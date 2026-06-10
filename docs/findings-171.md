# torchft friction log & findings (feeds the issue #171 comment)

Anchor: [meta-pytorch/torchft#171](https://github.com/meta-pytorch/torchft/issues/171)
— "Towards Native Fault Tolerance for Semi-Synchronous Training". Open as of 2026-06-10.
Pinned commit for all observations: `4157be16` (+ `torchft-nightly==2026.6.10`).

Every entry: date, what happened, repro, tag (`candidate-issue` / `candidate-pr` /
`candidate-doc` / `evidence-171`). Measured evidence for the eventual #171 comment:
momentum-recovery digests, T_detect/T_resume/T_rejoin, 1-survivor sync semantics,
kill-mid-allreduce behavior on Gloo.

---

## 2026-06-10 — setup notes

- `train_diloco.py` (repo root) pins `CUDA_VISIBLE_DEVICES = REPLICA_GROUP_ID % 4` at
  module import; any same-GPU multi-replica setup must avoid that pattern. The example
  is also MLP/dummy-data only — no small-scale LM example exists. [candidate-doc]
- Known live bugs steered around from day one: #316 (async-quorum SIGSEV → we run
  `use_async_quorum=False`), #323 (PGTransport timeout ineffective → we use
  HTTPTransport). [evidence-171: config guidance]
