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

## 2026-06-11 — M3 storm finding: restart churn wipes global state at small N [evidence-171]

Storm (2 replica groups, min_replica_size=1, supervisor auto-restart 15s, Poisson kills
mean 120s): global eval loss REGRESSED repeatedly (2.39 → 3.6 → 2.4 → 4.0) despite 86.7%
committed-sync throughput. Mechanism: a kill landing while the only other member is
alive-but-unhealed leaves a fresh-init worker as a singleton quorum; its near-random
weights become the cluster state and the victim heals FROM it (manager logs show heals
with donor max_step as low as 0). Live P2P recovery is necessary but NOT sufficient
under churn at small replica counts — torchft's blog setup (30 groups) makes this
practically unreachable, but cross-datacenter DiLoCo (#171's regime, few big members)
hits it head-on. Mitigations implemented here: (a) commit-coupled checkpoints (each
replica persists state every K commits; restarts init from the newest checkpoint, so a
wiped quorum resumes from durable state); (b) experiment hygiene: kills only execute
against clusters with a HEALTHY donor (committed in current process generation).
[evidence-171; candidate-doc]

## 2026-06-11 — M4 WAN sweep findings [evidence-171]

netem sweep (veth, 20ms RTT, fp32 204.8MB payload, 51M model, 2 replica groups):
- Per-step sync (H=1, DDP comm pattern) degrades from 5.3k tok/s at 1 Gbps (already 7x
  slower than DiLoCo on the SAME link — the payload costs ~2s/step even at gigabit) to
  858 tok/s at 50 Mbps, and FAILS ENTIRELY at 10 Mbps.
- DiLoCo H=100 holds 27-37k tok/s from 1 Gbps down to 50 Mbps (sync cost amortized).
- **At 10 Mbps both fail, and the mechanism matters: the data plane starves the control
  plane.** The ~200s allreduce saturates the link; lighthouse heartbeats/quorum gRPC
  share it; quorum times out mid-transfer ("lighthouse quorum failed: Timeout expired")
  and the cluster cascades. Mitigations to explore: quantized sync (upstream
  should_quantize=True halves/quarters payload), Streaming DiLoCo fragments, or QoS
  prioritization of the control plane. [evidence-171; candidate-doc]
- **Rendezvous skew on heterogeneous workers:** first sync happens after H local steps,
  so workers with different step times (GPU 0.4s vs CPU 0.8s + differing init costs)
  request their first quorum up to minutes apart. `Manager(quorum_timeout=)` defaults to
  60s — independently of `timeout=` — and the faster worker dies at first rendezvous if
  skew exceeds it (observed: GPU at +41s, CPU at +103s, quorum formed 2s after the GPU
  worker timed out). Heterogeneous/cross-DC deployments need quorum_timeout >= worst-case
  H x step-time spread. [candidate-doc]

## 2026-06-12 — M4 cloud smoke: cross-internet sync works [evidence-171]

worker4 (home, RTX 3060) + Vast.ai RTX 4090 VM (Virginia) as 2 DiLoCo replica groups
over a tailscale mesh (~52 ms RTT, real internet). First outer sync committed with
num_participants=2; param digests bit-identical across the WAN (a573c3de001da30a both
sides). The 204 MB fp32 pseudo-gradient allreduce over gloo-on-tailscale succeeded.
Confirms torchft DiLoCo works across NAT'd, geographically-distributed commodity nodes
with no code changes beyond the hostname/store-bind fixes already documented above —
exactly #171's cross-datacenter target. (Operational gotchas in docs/cloud.md.)
[evidence-171]
