"""YAML config loading with CLI overrides.

A train config (configs/train/*.yaml) names a model config (configs/model/*.yaml)
via its `model:` key; both live under the repo's configs/ root.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .model import GPTConfig


@dataclass
class TrainConfig:
    # identity
    run_id: str = "dev"
    out_dir: str = "experiments"
    mode: str = "baseline"  # baseline | diloco
    seed: int = 1337
    # model (resolved from configs/model/<model>.yaml)
    model: str = "tiny50m"
    model_cfg: GPTConfig = field(default_factory=GPTConfig)
    # data
    data_dir: str = "data/tinystories"
    # optimization (inner)
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 6000
    lr: float = 4e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_frac: float = 0.1
    # diloco (outer)
    sync_every: int = 100  # H
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    min_replica_size: int = 1
    quorum_timeout_s: float = 120.0
    ckpt_every_syncs: int = 5  # commit-coupled durable checkpoint cadence (0 = off)
    # cadence
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 50
    # system
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False


def load_config(train_yaml: str | None, overrides: list[str], configs_root: str | Path = "configs") -> TrainConfig:
    cfg = TrainConfig()
    raw: dict = {}
    if train_yaml:
        raw = yaml.safe_load(Path(train_yaml).read_text()) or {}
    for kv in overrides:
        k, _, v = kv.partition("=")
        raw[k.strip()] = yaml.safe_load(v)
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    model_yaml = Path(configs_root) / "model" / f"{cfg.model}.yaml"
    mraw = yaml.safe_load(model_yaml.read_text())
    cfg.model_cfg = GPTConfig(**mraw)
    return cfg


def config_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["model_cfg"] = asdict(cfg.model_cfg)
    return d
