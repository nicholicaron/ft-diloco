#!/usr/bin/env bash
# N-way failure storm on a single commodity host (worker4): N CPU/micro DiLoCo
# replicas in per-replica netns + one supervisor + Poisson chaos for a fixed window,
# then teardown. Lighthouse runs on THIS host (reachable from every netns at the
# bridge gateway 10.77.0.1) so the whole storm is self-contained — no second machine
# in the critical path. Generalized from run_m3_storm.sh (which hardcoded 2 GPU
# replicas). Run ON worker4 inside tmux:
#   tmux new -d -s storm 'bash scripts/run_storm.sh storm-n32 32 configs/chaos/storm_n32.yaml 2400 2>&1 | tee /tmp/storm.log'
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
N="${2:?num replicas}"
SCHEDULE="${3:?chaos schedule yaml}"
DURATION="${4:-2400}"
CONFIG="${5:-configs/train/storm_micro.yaml}"
WARMUP="${WARMUP:-60}"
COOLDOWN="${COOLDOWN:-90}"
DELAY="${DELAY:-15}"
LH_PORT="${LH_PORT:-29510}"
export LH="http://10.77.0.1:${LH_PORT}"
# CPU pinning (PIN=1): give each replica a dedicated thread so step times stay uniform
# and replicas reach each sync barrier together (raises quorum participation at N≈cores).
export PIN="${PIN:-0}"
export PIN_CORES="${PIN_CORES:-$(nproc)}"

echo "=== storm $RUN: N=$N, ${DURATION}s, schedule $SCHEDULE, config $CONFIG $(date -Is)"
bash scripts/kill_run.sh "$RUN" || true
for R in $(seq 0 $((N - 1))); do tmux kill-session -t "ftdr$R" 2>/dev/null || true; done
tmux kill-session -t ftdsup 2>/dev/null || true
rm -rf "experiments/$RUN"; mkdir -p "experiments/$RUN"

# fresh netns 0..N-1 (idempotent rebuild)
sudo bash scripts/netns_cluster.sh down "$N" 2>/dev/null || true
sudo bash scripts/netns_cluster.sh up "$N"

# lighthouse on THIS host, dual-stack [::] so netns can dial it at 10.77.0.1
tmux kill-session -t ftdlh 2>/dev/null || true
tmux new -d -s ftdlh \
  "RUST_LOG=info .venv/bin/torchft_lighthouse --bind '[::]:${LH_PORT}' \
   --min_replicas 2 --quorum_tick_ms 100 --join_timeout_ms 10000 \
   >> experiments/$RUN/lighthouse.log 2>&1"
sleep 3

# N workers
for R in $(seq 0 $((N - 1))); do bash scripts/launch_storm_replica.sh "$RUN" "$R" "$N" "$CONFIG"; done

# one supervisor watching all N
tmux new -d -s ftdsup \
  "cd /srv/fpga/ft-diloco && PIN=$PIN PIN_CORES=$PIN_CORES DELAY=$DELAY \
   bash scripts/supervisor_n.sh $RUN $N $DELAY $CONFIG \
   >> experiments/$RUN/supervisor.log 2>&1"

echo "=== warmup ${WARMUP}s before first fault $(date -Is)"
sleep "$WARMUP"
echo "=== chaos begins $(date -Is)"
.venv/bin/python -m chaos.controller --schedule "$SCHEDULE" --run-dir "experiments/$RUN" --replicas "$N"
echo "=== chaos schedule done $(date -Is); cooldown ${COOLDOWN}s"
sleep "$COOLDOWN"

echo "=== teardown $(date -Is)"
tmux kill-session -t ftdsup 2>/dev/null || true
bash scripts/kill_run.sh "$RUN" || true
for R in $(seq 0 $((N - 1))); do tmux kill-session -t "ftdr$R" 2>/dev/null || true; done
tmux kill-session -t ftdlh 2>/dev/null || true
sudo bash scripts/netns_cluster.sh down "$N" 2>/dev/null || true
echo "MARK storm_complete $RUN $(date -Is)"
