#!/usr/bin/env bash
# Onstart/bootstrap for a Vast.ai instance joining the ft-diloco cluster.
# Expects env (set via vastai create instance --env or exported before running):
#   TS_AUTHKEY      tailscale auth key (reusable+ephemeral recommended)
#   REPLICA_ID      this node's replica group id (2, 3, ...)
#   N_REPLICAS      total replica groups in the cluster
#   LIGHTHOUSE      http://<worker1-tailscale-ip>:29510
#   H               sync_every (default 100)   STEPS (default 3000)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/5] tailscale"
curl -fsSL https://tailscale.com/install.sh | sh
if [ -e /dev/net/tun ]; then
  (tailscaled > /var/log/tailscaled.log 2>&1 &)
else
  echo "no /dev/net/tun — userspace netstack mode"
  (tailscaled --tun=userspace-networking > /var/log/tailscaled.log 2>&1 &)
fi
sleep 3
tailscale up --authkey "$TS_AUTHKEY" --hostname "ftd-cloud-$REPLICA_ID"
TS_IP=$(tailscale ip -4)
echo "tailscale up: $TS_IP"

echo "=== [2/5] repo + deps"
cd /root
git clone --depth 1 https://github.com/Neumann-Labs/ft-diloco.git
cd ft-diloco
pip install -q -e ".[prep]" torchft-nightly==2026.6.10

echo "=== [3/5] data (tokenize from HF — cloud pipe is fast)"
python -m ftdiloco.data --dataset tinystories --out data/tinystories

echo "=== [4/5] connectivity check"
curl -s -o /dev/null -w "lighthouse HTTP %{http_code}\n" "$LIGHTHOUSE" || true

echo "=== [5/5] launch replica $REPLICA_ID/$N_REPLICAS"
mkdir -p "experiments/m4-cloud"
MASTER_ADDR="$TS_IP" MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
GLOO_SOCKET_IFNAME="${GLOO_IFACE:-eth0}" \
REPLICA_GROUP_ID="$REPLICA_ID" NUM_REPLICA_GROUPS="$N_REPLICAS" \
TORCHFT_LIGHTHOUSE="$LIGHTHOUSE" FTD_ADVERTISE_HOST="$TS_IP" \
nohup python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
  --set run_id=m4-cloud --set sync_every="${H:-100}" --set max_steps="${STEPS:-3000}" \
  --set quorum_timeout_s=600 --set pg_timeout_s=600 \
  > experiments/m4-cloud/worker$REPLICA_ID.log 2>&1 &
echo "BOOTSTRAP_DONE replica=$REPLICA_ID ts_ip=$TS_IP"
