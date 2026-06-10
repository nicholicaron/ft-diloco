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

- `Manager` hard-requires torchrun-style env (`MASTER_ADDR`/`MASTER_PORT`, and the
  TCPStore it implies) even for a single-process replica group — standalone (non-torchrun)
  DiLoCo usage isn't documented anywhere; repro: construct Manager without torchrun →
  `KeyError: 'MASTER_ADDR'` in manager.py store setup. Workaround: export
  MASTER_ADDR=localhost + unique MASTER_PORT per replica group on shared hosts.
  [candidate-doc]
- Root cause of the standalone hang: `Manager.__init__` connects to the replica-group
  TCPStore with `is_master=False` (manager.py:291) — it assumes torchrun is hosting the
  store. Without torchrun the constructor blocks forever (no timeout, no error). A
  standalone launcher must host a `TCPStore(is_master=True)` on group rank 0 first.
  Repro: construct Manager with MASTER_ADDR/PORT set but no store server → hang.
  [candidate-doc; candidate-pr: a store-hosting fallback or a clear timeout error]

## 2026-06-10 — M0.5 kill/rejoin validation (run m05-rejoin) [evidence-171]

Setup: 2 replica groups (micro 3.3M-param LM, CPU/Gloo, H=20) on worker4, lighthouse on
worker1, `min_replica_size=1`, sync quorum, HTTPTransport.

- `kill -9` replica 1 mid-run (22:53:59): survivor's next `should_commit` succeeded one
  sync period later; quorum shrank 2→1 with zero stall; solo sync cadence ~2x faster
  (no peer wait). No manual intervention.
- Relaunch (22:56:08): manager logged `healing is required step=0, max_step=41,
  recover_src_replica_rank=0` → P2P state transfer from the survivor; first sync after
  rejoin committed at outer_step 42 with 2 participants.
- **State equality proof: 30/30 digest points (params + outer Nesterov momentum buffers)
  bit-identical across replicas at every post-rejoin sync boundary** (sha256 over fp32
  bytes). Outer optimizer state demonstrably survives kill→rejoin via live recovery —
  no checkpoint needed.
- Negative control (run m05-kill, accidental): a worker rejoining an EMPTY cluster (all
  peers exited) starts from scratch at step 0 — live recovery requires a living peer;
  full-cluster death needs durable checkpoints (the M3 supervisor design point).
- Cross-namespace (≈ multi-host) standalone DiLoCo needs three non-obvious settings:
  (1) `Manager(hostname=...)` override — default `socket.gethostname()` advertises an
  address peers can't reach; (2) `MASTER_ADDR` must be the externally-reachable IP, NOT
  localhost — the quorum advertises it to peers as the replica's store address for PG
  configure (localhost times out at c10d socket.cpp:1030); (3) `GLOO_SOCKET_IFNAME`
  pinned to the right interface. None of this is documented for non-torchrun use.
  [candidate-doc]
- One lighthouse = one logical training job. Two concurrent runs registering the same
  replica_ids ("ftd_0"/"ftd_1") against a shared lighthouse merge into a single broken
  quorum (observed num_participants=4, commits fail). Operational rule: kill prior runs
  (or use a dedicated lighthouse / distinct replica_id namespace) before starting a new
  cluster. Ghost members age out after heartbeat_timeout (5s default). [candidate-doc]
- **BUG (candidate-pr): `HTTPTransport.address()` hardcodes `socket.gethostname()`**
  (checkpointing/http_transport.py), ignoring `Manager(hostname=...)`. Any topology
  where the local hostname isn't peer-resolvable/reachable (netns, NAT, multi-homed
  cross-DC nodes — the #171 scenario) makes P2P recovery fail with connection-refused;
  the recovering replica loops in heal while the donor's allreduce times out
  ("Application timeout caused pair closure" at gloo unbound_buffer.cc:78). Repro:
  two replica groups in separate netns, kill+rejoin one. Fix: plumb an advertised
  hostname into HTTPTransport (mirror Manager's hostname param); we run a subclass
  override meanwhile. [candidate-pr]
