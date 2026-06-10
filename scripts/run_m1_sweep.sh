#!/usr/bin/env bash
# M1 H-sweep on worker4: for each H, run 2 DiLoCo replica groups (same GPU) to
# completion, sequentially across H values. Run inside tmux (link flaps kill bare ssh).
#
#   tmux new -d -s m1sweep 'scripts/run_m1_sweep.sh 1337 2>&1 | tee /tmp/m1sweep.log'
#
# Env overrides: HS="25 50 100 200 500", STEPS=3000, TORCHFT_LIGHTHOUSE.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-1337}"
HS=(${HS:-25 50 100 200 500})
STEPS="${STEPS:-3000}"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://worker1:29510}"

for H in "${HS[@]}"; do
  RUN="m1-h${H}-s${SEED}"
  echo "=== $RUN (H=$H, steps=$STEPS/replica) $(date -Is) ==="
  mkdir -p "experiments/$RUN"
  pids=()
  for R in 0 1; do
    REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 \
      .venv/bin/python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
      --set run_id="$RUN" --set sync_every="$H" --set seed="$SEED" --set max_steps="$STEPS" \
      > "experiments/$RUN/worker$R.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  echo "=== $RUN done rc=$rc $(date -Is) ==="
done
echo "sweep complete"
