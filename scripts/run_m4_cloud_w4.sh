#!/usr/bin/env bash
# worker4's replica for a cloud-hybrid run: advertises its TAILSCALE address so cloud
# peers can reach the store/gloo/recovery endpoints across the real internet.
# Writes an inner launch script first (nested ssh+tmux quoting is a footgun).
#   bash scripts/run_m4_cloud_w4.sh <run_id> <n_replicas> [steps] [min_replica_size] [config]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
N="${2:?n_replicas}"
STEPS="${3:-600}"
MINREP="${4:-2}"
CONFIG="${5:-configs/train/m4_cloud.yaml}"
TS_IP=$(tailscale ip -4)
mkdir -p "experiments/$RUN"

cat > /tmp/ftd_w4_inner.sh <<INNER
#!/usr/bin/env bash
cd /srv/fpga/ft-diloco
export MASTER_ADDR=$TS_IP MASTER_PORT=29600 RANK=0 WORLD_SIZE=1
export GLOO_SOCKET_IFNAME=tailscale0 REPLICA_GROUP_ID=0 NUM_REPLICA_GROUPS=$N
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHFT_LIGHTHOUSE=http://100.86.208.63:29510 FTD_ADVERTISE_HOST=$TS_IP
exec .venv/bin/python -m ftdiloco.train --config $CONFIG \\
  --set run_id=$RUN --set device=cuda --set min_replica_size=$MINREP --set max_steps=$STEPS \\
  >> experiments/$RUN/worker0.log 2>&1
INNER
chmod +x /tmp/ftd_w4_inner.sh

tmux kill-session -t ftdcloud0 2>/dev/null || true
tmux new -d -s ftdcloud0 'bash /tmp/ftd_w4_inner.sh'
echo "W4_CLOUD_REPLICA_LAUNCHED $RUN ts=$TS_IP minrep=$MINREP"
