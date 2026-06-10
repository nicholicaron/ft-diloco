"""torchft integration: Manager construction, DiLoCo wiring, sync-boundary telemetry.

Mirrors the canonical upstream example (train_diloco.py @ pinned commit, see
pyproject.toml) with three deliberate differences:
- no CUDA_VISIBLE_DEVICES pinning (our replicas deliberately share one GPU),
- Gloo for the manager ProcessGroup in all topologies (outer sync is rare; avoids
  mixed CPU/GPU quorum risk and the NCCL same-host trap),
- HTTPTransport for checkpoint recovery (sidesteps torchft #323 PGTransport timeout).

Eval cadence in diloco mode aligns to outer-sync boundaries: right after a committed
sync all replicas hold identical params, so the eval measures the *global* model.
"""

import os
import time
from datetime import timedelta

import numpy as np
import torch

from .config import TrainConfig
from .data import TokenBins
from .metrics import RunLogger, tensor_digest


def run_diloco(
    cfg: TrainConfig,
    model: torch.nn.Module,
    inner_opt: torch.optim.Optimizer,
    bins: TokenBins,
    rng: np.random.Generator,
    logger: RunLogger,
    autocast_ctx,
    train_loop_fn,
) -> None:
    from torchft import Manager, ProcessGroupGloo
    from torchft.checkpointing.http_transport import HTTPTransport
    from torchft.local_sgd import DiLoCo

    replica_id = int(os.environ.get("REPLICA_GROUP_ID", "0"))
    raw_model = getattr(model, "_orig_mod", model)
    outer_opt = torch.optim.SGD(
        raw_model.parameters(), lr=cfg.outer_lr, momentum=cfg.outer_momentum, nesterov=True
    )

    def state_dict() -> dict:
        return {
            "model": raw_model.state_dict(),
            "inner_optim": inner_opt.state_dict(),
            "outer_optim": outer_opt.state_dict(),
        }

    def load_state_dict(sd: dict) -> None:
        raw_model.load_state_dict(sd["model"])
        inner_opt.load_state_dict(sd["inner_optim"])
        outer_opt.load_state_dict(sd["outer_optim"])

    manager = Manager(
        pg=ProcessGroupGloo(timeout=timedelta(seconds=30)),
        use_async_quorum=False,  # torchft #316: async quorum SIGSEVs on long runs
        min_replica_size=cfg.min_replica_size,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"ftd_{replica_id}",
        timeout=timedelta(seconds=cfg.quorum_timeout_s),
        checkpoint_transport=HTTPTransport(timeout=timedelta(seconds=60), num_chunks=0),
    )

    H = cfg.sync_every
    syncs_per_eval = max(1, round(cfg.eval_every / H))
    payload_bytes = 4 * sum(p.numel() for p in raw_model.parameters())  # fp32 pseudo-grads
    last_outer = {"step": manager.current_step(), "mono": time.monotonic()}

    def post_step(step: int) -> None:
        if step % H != 0:
            return
        now = time.monotonic()
        outer = manager.current_step()
        committed = outer > last_outer["step"]
        logger.log(
            "outer_sync",
            step=step,
            outer_step=outer,
            committed=committed,
            num_participants=manager.num_participants(),
            t_since_prev_s=now - last_outer["mono"],
            bytes_analytic=payload_bytes,
        )
        last_outer["step"] = outer
        last_outer["mono"] = now
        params = [p for p in raw_model.parameters()]
        logger.log("digest", step=step, outer_step=outer, kind="params", **tensor_digest(params))
        mom = [
            s["momentum_buffer"]
            for s in outer_opt.state.values()
            if isinstance(s, dict) and "momentum_buffer" in s
        ]
        if mom:
            logger.log(
                "digest", step=step, outer_step=outer, kind="outer_momentum", **tensor_digest(mom)
            )

    def should_eval(step: int) -> bool:
        return step % H == 0 and (step // H) % syncs_per_eval == 0

    logger.log(
        "lifecycle",
        phase="diloco_setup",
        sync_every=H,
        min_replica_size=cfg.min_replica_size,
        lighthouse=os.environ.get("TORCHFT_LIGHTHOUSE", ""),
        payload_bytes=payload_bytes,
    )
    try:
        with DiLoCo(
            manager,
            [raw_model],
            inner_opt,
            outer_opt,
            sync_every=H,
            backup_device=torch.device("cpu"),
            should_quantize=False,
        ):
            train_loop_fn(
                cfg,
                model,
                inner_opt,
                bins,
                rng,
                logger,
                autocast_ctx,
                post_step_hook=post_step,
                should_eval=should_eval,
            )
    finally:
        shutdown = getattr(manager, "shutdown", None)
        if callable(shutdown):
            shutdown()
