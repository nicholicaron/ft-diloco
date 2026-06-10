#!/usr/bin/env bash
# M1 H-sweep on worker4: for each H, two DiLoCo replica groups (same GPU), each inside
# its own network namespace (ftd0/ftd1) so sync traffic crosses measurable veths.
# Requires: sudo scripts/netns_cluster.sh up 2   (passwordless sudo available)
# Run inside tmux:
#   tmux new -d -s m1sweep 'scripts/run_m1_sweep.sh 1337 2>&1 | tee /tmp/m1sweep.log'
# Env overrides: HS="25 50 100 200 500", STEPS=3000, LIGHTHOUSE_V4.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-1337}"
HS=(${HS:-25 50 100 200 500})
STEPS="${STEPS:-3000}"
# v4 only inside the namespaces (no v6 route over the veth/NAT path)
LIGHTHOUSE_V4="${LIGHTHOUSE_V4:-http://192.168.1.104:29510}"

sudo scripts/netns_cluster.sh up 2 || true

for H in "${HS[@]}"; do
  RUN="m1-h${H}-s${SEED}"
  echo "=== $RUN (H=$H, steps=$STEPS/replica) $(date -Is) ==="
  mkdir -p "experiments/$RUN"
  .venv/bin/python -m ftdiloco.netmon --ifaces vftd0 vftd1 \
    --out "experiments/$RUN/netmon.jsonl" --interval 0.5 &
  NETMON_PID=$!
  pids=()
  for R in 0 1; do
    sudo ip netns exec "ftd$R" sudo -u "$(whoami)" env \
      MASTER_ADDR="10.77.0.1$R" MASTER_PORT=$((29600 + R)) RANK=0 WORLD_SIZE=1 \
      GLOO_SOCKET_IFNAME=eth0 \
      REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCHFT_LIGHTHOUSE="$LIGHTHOUSE_V4" FTD_ADVERTISE_HOST="10.77.0.1$R" \
      "$PWD/.venv/bin/python" -m ftdiloco.train --config configs/train/m1_diloco.yaml \
      --set run_id="$RUN" --set sync_every="$H" --set seed="$SEED" --set max_steps="$STEPS" \
      > "experiments/$RUN/worker$R.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  kill "$NETMON_PID" 2>/dev/null || true
  echo "=== $RUN done rc=$rc $(date -Is) ==="
done
echo "MARK sweep_complete $(date -Is)"
