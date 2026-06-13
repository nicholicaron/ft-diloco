#!/usr/bin/env bash
# Launch ONE CPU DiLoCo replica for an N-way storm, inside its netns, in tmux.
# Generalized from launch_m2_replica.sh: any N, CPU/gloo, micro model, 1 OMP thread
# (N single-threaded procs share the host's cores — avoids 32 procs each grabbing
# all cores and thrashing). Idempotent per replica (kills any prior ftdr$R session).
#   launch_storm_replica.sh <run_id> <replica_idx> <num_replicas> [config]
# Env: LH (lighthouse URL, default the bridge gateway = lighthouse on this host),
#      STEPS (max_steps cap, default huge — storm is wall-clock bounded).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
R="${2:?replica idx}"
N="${3:?num replicas}"
CONFIG="${4:-configs/train/storm_micro.yaml}"
STEPS="${STEPS:-1000000}"
LH="${LH:-http://10.77.0.1:29510}"
# Pin each replica to a dedicated CPU thread (PIN=1). At N≈cores, unpinned procs get
# migrated/timesliced unevenly by the scheduler → step times desync → only a subset
# reach each H-boundary together → low quorum participation. A fixed 1:1 pin makes
# step times uniform so replicas hit the barrier together. PIN_CORES caps the modulus.
PIN="${PIN:-0}"
PIN_CORES="${PIN_CORES:-$(nproc)}"
TASKSET=""; PIN_ENV=""
if [ "$PIN" = "1" ]; then
  CORE=$((R % PIN_CORES))
  TASKSET="taskset -c $CORE"        # initial hint (torch resets it; FTD_PIN_CORE re-asserts in-process)
  PIN_ENV="FTD_PIN_CORE=$CORE"
fi
mkdir -p "experiments/$RUN"
tmux kill-session -t "ftdr$R" 2>/dev/null || true
tmux new -d -s "ftdr$R" \
  "cd /srv/fpga/ft-diloco && sudo ip netns exec ftd$R sudo -u claude env \
   MASTER_ADDR=10.77.0.1$R MASTER_PORT=$((29600 + R)) RANK=0 WORLD_SIZE=1 \
   GLOO_SOCKET_IFNAME=eth0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PIN_ENV \
   REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=$N \
   TORCHFT_LIGHTHOUSE=$LH FTD_ADVERTISE_HOST=10.77.0.1$R \
   $TASKSET /srv/fpga/ft-diloco/.venv/bin/python -m ftdiloco.train \
     --config $CONFIG \
     --set run_id=$RUN --set max_steps=$STEPS \
     >> experiments/$RUN/worker$R.log 2>&1"
echo "replica $R/$N launched for $RUN (lighthouse $LH)"
