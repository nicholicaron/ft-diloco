#!/usr/bin/env bash
# Relaunch the netns micro validation run (m1-nstest): 2 CPU replicas inside
# ftd0/ftd1 namespaces, micro model, synthetic data. Run ON worker4.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=m1-nstest
bash scripts/kill_run.sh "$RUN"
sleep 8
mkdir -p "experiments/$RUN"
rm -f experiments/$RUN/replica*.jsonl experiments/$RUN/worker*.log
for R in 0 1; do
  tmux kill-session -t "ftdns$R" 2>/dev/null || true
  tmux new -d -s "ftdns$R" \
    "cd /srv/fpga/ft-diloco && sudo ip netns exec ftd$R sudo -u claude env \
     MASTER_ADDR=10.77.0.1$R MASTER_PORT=$((29700 + R)) RANK=0 WORLD_SIZE=1 \
     GLOO_SOCKET_IFNAME=eth0 REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 \
     TORCHFT_LIGHTHOUSE=http://192.168.1.104:29510 FTD_ADVERTISE_HOST=10.77.0.1$R \
     /srv/fpga/ft-diloco/.venv/bin/python -m ftdiloco.train \
       --config configs/train/m1_diloco.yaml \
       --set run_id=$RUN --set model=micro --set data_dir=/tmp/ftd-smoke-data \
       --set device=cpu --set dtype=float32 --set batch_size=4 --set max_steps=200 \
       --set sync_every=20 --set eval_every=100 --set eval_batches=2 --set log_every=20 \
       >> experiments/$RUN/worker$R.log 2>&1"
done
echo "NSTEST_RELAUNCHED $(date -Is)"
