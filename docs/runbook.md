# Runbook

## Hosts

| Host | Role | Project dir | Venv |
|---|---|---|---|
| worker4 (5950X + RTX 3060 12GB) | GPU trainer; all replicas in M1–M3 | `/srv/fpga/ft-diloco` | `.venv` (torch cu128) |
| worker1 (8-core, CPU) | Lighthouse; later CPU replica | `~/ft-diloco` | `.venv` (torch cpu + torchft-nightly) |

Dev happens on the Mac; `make sync` pushes code to worker4, `make fetch` pulls
`experiments/`. Long jobs always run inside tmux on worker4 (the link flaps).

## torchft

- Wheel: `torchft-nightly==2026.6.10` (both hosts).
- Source reference + fix branches: `github.com/nicholicaron/torchft`
  (fork of meta-pytorch/torchft), pinned at upstream `4157be16` (2026-06-09).
  Editable source build (needs Rust + protobuf) only if/when we patch torchft.
- Friction log: [findings-171.md](findings-171.md) — every workaround/surprise goes
  there at the moment it happens, tagged candidate-issue / candidate-pr / candidate-doc.

## Standard runs

```bash
# one-time data prep (worker4)
.venv/bin/python -m ftdiloco.data --dataset tinystories --out data/tinystories

# M0 baseline (worker4, tmux)
tmux new -d -s m0 '.venv/bin/python -m ftdiloco.train --config configs/train/m0_tiny.yaml \
    --set run_id=m0-tiny-s1337 --set seed=1337 2>&1 | tee /tmp/m0.log'

# lighthouse (worker1)
tmux new -d -s lighthouse '~/ft-diloco/.venv/bin/torchft_lighthouse --min_replicas 1 \
    --quorum_tick_ms 100 --join_timeout_ms 10000 2>&1 | tee /tmp/lighthouse.log'

# M1 sweep (worker4, tmux)
tmux new -d -s m1sweep 'scripts/run_m1_sweep.sh 1337 2>&1 | tee /tmp/m1sweep.log'
```

## Conventions

- Every run gets `experiments/<run_id>/`: per-replica JSONL + worker logs (+ netmon.jsonl
  when the netns harness is up). Curated runs are committed.
- run_id: `<milestone>-<variant>-s<seed>`, e.g. `m1-h100-s1337`.
- Convergence comparisons are vs total tokens across replicas; wall-clock only within
  identical-contention conditions.
