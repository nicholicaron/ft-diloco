#!/usr/bin/env bash
# Launch one cloud replica on a Vast VM. Run ON the VM (avoids nested ssh+tmux quoting).
#   bash launch_replica.sh <replica_id> <n_replicas> <run_id> <min_replica_size> <max_steps> [config]
set -uo pipefail
cd /root/ft-diloco
RID="${1:?rid}"; N="${2:?n}"; RUN="${3:?run}"; MINREP="${4:?minrep}"; STEPS="${5:?steps}"
CONFIG="${6:-configs/train/m4_cloud.yaml}"
pkill -f "ftdiloco.train" 2>/dev/null; sleep 3
mkdir -p "experiments/$RUN"
TS_IP=$(tailscale ip -4)
nohup env MASTER_ADDR="$TS_IP" MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
  GLOO_SOCKET_IFNAME=tailscale0 REPLICA_GROUP_ID="$RID" NUM_REPLICA_GROUPS="$N" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCHFT_LIGHTHOUSE=http://100.86.208.63:29510 FTD_ADVERTISE_HOST="$TS_IP" \
  python3 -m ftdiloco.train --config "$CONFIG" \
    --set run_id="$RUN" --set device=cuda --set min_replica_size="$MINREP" --set max_steps="$STEPS" \
    > "experiments/$RUN/worker$RID.log" 2>&1 < /dev/null &
sleep 4
echo "LAUNCHED replica=$RID ts=$TS_IP pid=$! log_lines=$(wc -l < experiments/$RUN/worker$RID.log)"
