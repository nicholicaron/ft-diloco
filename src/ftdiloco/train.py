"""Single training entrypoint for both modes.

  baseline:  python -m ftdiloco.train --config configs/train/m0_tiny.yaml
  diloco:    REPLICA_GROUP_ID=0 NUM_REPLICA_GROUPS=2 TORCHFT_LIGHTHOUSE=http://worker1:29510 \
                 python -m ftdiloco.train --config configs/train/m1_diloco.yaml

This file (plus ft.py) is the only torchft touchpoint in the repo.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig, config_dict, load_config
from .data import TokenBins
from .eval import evaluate
from .metrics import RunLogger
from .model import GPT


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    min_lr = cfg.lr * cfg.min_lr_frac
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (cfg.lr - min_lr)


def train_loop(
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    bins: TokenBins,
    rng: np.random.Generator,
    logger: RunLogger,
    autocast_ctx,
    post_step_hook=None,
    should_eval=None,
) -> None:
    """Shared inner loop. In diloco mode the DiLoCo context manager (entered by the
    caller) hooks `optimizer.step()` to run syncs; `post_step_hook(step)` lets ft.py
    log sync/digest events at boundaries, and `should_eval(step)` aligns evals to them."""
    block = cfg.model_cfg.block_size
    tokens_per_step = cfg.batch_size * cfg.grad_accum * block
    tokens = 0
    raw_model = getattr(model, "_orig_mod", model)
    if should_eval is None:
        should_eval = lambda step: step % cfg.eval_every == 0  # noqa: E731

    def do_eval(step: int) -> None:
        res = evaluate(raw_model, bins, cfg.batch_size, cfg.eval_batches, autocast_ctx)
        logger.log("eval", step=step, tokens=tokens, **res)

    do_eval(0)
    t_last = time.monotonic()
    for step in range(1, cfg.max_steps + 1):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = bins.get_batch(rng, cfg.batch_size)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            loss_acc += loss.item()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tokens += tokens_per_step

        if post_step_hook is not None:
            post_step_hook(step)
        if step % cfg.log_every == 0:
            now = time.monotonic()
            dt_ms = (now - t_last) * 1000 / cfg.log_every
            t_last = now
            logger.log("step", step=step, loss=loss_acc, lr=lr, tokens=tokens, dt_ms=dt_ms)
        if should_eval(step) or step == cfg.max_steps:
            do_eval(step)
            t_last = time.monotonic()


def run(cfg: TrainConfig) -> None:
    replica_id = int(os.environ.get("REPLICA_GROUP_ID", "0"))
    num_replicas = int(os.environ.get("NUM_REPLICA_GROUPS", "1")) if cfg.mode == "diloco" else 1

    torch.manual_seed(cfg.seed + replica_id)
    rng = np.random.default_rng(cfg.seed + 1000 * replica_id)
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[cfg.dtype]
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if "cuda" in device
        else nullcontext()
    )

    out_dir = Path(cfg.out_dir) / cfg.run_id
    logger = RunLogger(out_dir, cfg.run_id, replica_id)
    logger.log_start(config_dict(cfg), device=device, num_replicas=num_replicas)

    bins = TokenBins(cfg.data_dir, cfg.model_cfg.block_size, replica_id, num_replicas, device)
    model = GPT(cfg.model_cfg).to(device)
    logger.log("lifecycle", phase="model", n_params=model.num_params())
    if cfg.compile:
        model = torch.compile(model)
    inner_opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    try:
        if cfg.mode == "diloco":
            from .ft import run_diloco

            run_diloco(cfg, model, inner_opt, bins, rng, logger, autocast_ctx, train_loop)
        else:
            train_loop(cfg, model, inner_opt, bins, rng, logger, autocast_ctx)
        raw_model = getattr(model, "_orig_mod", model)
        torch.save(raw_model.state_dict(), out_dir / f"checkpoint_final_r{replica_id}.pt")
        logger.close(status="ok")
    except BaseException as e:
        logger.close(status=f"error: {type(e).__name__}: {e}")
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="train config yaml")
    p.add_argument(
        "--set", action="append", default=[], dest="overrides", help="key=value override"
    )
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    run(cfg)


if __name__ == "__main__":
    main()
