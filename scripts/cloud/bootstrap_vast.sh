#!/usr/bin/env bash
# Onstart for a Vast.ai *VM* instance joining the ft-diloco cluster. Run as an
# --onstart script (executes in-guest on real VMs). REQUIRES a true VM, not a
# docker container — see docs/cloud.md: launch with the official "Ubuntu 22.04 VM"
# template_hash or a vastai/kvm:* image. Docker instances fall back to tailscale
# userspace-netstack, which this stack cannot use (store bind + gloo both fail).
#
# Env (set via `vastai create instance --env`):
#   TS_AUTHKEY   tailscale auth key (reusable + ephemeral)
#   REPLICA_ID   this node's replica group id (1, 2, ...)
#   N_REPLICAS   total replica groups in the cluster
#   LIGHTHOUSE   http://<worker1-tailscale-ip>:29510
#   PUBKEY       (optional) ssh pubkey to authorize, with perms fixed (VM images
#                ship authorized_keys with bad modes → sshd refuses; see cloud.md)
#   H            sync_every (default 100)    STEPS (default 3000)
#   DEVICE       cuda|cpu (default cuda)      TORCH_INDEX (default cu128)
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
exec > /var/log/ftd-bootstrap.log 2>&1

echo "=== [0/6] fix ssh key perms (VM images ship bad modes → lockout)"
mkdir -p /root/.ssh
[ -n "${PUBKEY:-}" ] && echo "$PUBKEY" >> /root/.ssh/authorized_keys
chown -R root:root /root/.ssh && chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true

echo "=== [1/6] base deps"
apt-get update -qq
apt-get install -y -qq python3-pip git curl >/dev/null 2>&1 || true

echo "=== [2/6] tailscale (real VM: systemd + /dev/net/tun)"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
systemctl enable --now tailscaled
sleep 3
tailscale up --authkey "$TS_AUTHKEY" --hostname "ftd-cloud-$REPLICA_ID"
TS_IP=$(tailscale ip -4)
echo "tailscale up: $TS_IP (iface: $(ip link show tailscale0 >/dev/null 2>&1 && echo tailscale0 || echo MISSING))"

echo "=== [3/6] repo + deps"
cd /root
[ -d ft-diloco ] || git clone --depth 1 https://github.com/Neumann-Labs/ft-diloco.git
cd ft-diloco
TORCH_INDEX="${TORCH_INDEX:-cu128}"
python3 -c "import torch" 2>/dev/null || \
  pip3 install -q torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
pip3 install -q -e ".[prep]" torchft-nightly==2026.6.10

echo "=== [4/6] data"
[ -f data/tinystories/val.bin ] || python3 -m ftdiloco.data --dataset tinystories --out data/tinystories

echo "=== [5/6] connectivity"
curl -s -o /dev/null -w "lighthouse HTTP %{http_code}\n" "$LIGHTHOUSE" || true

echo "=== [6/6] launch replica $REPLICA_ID/$N_REPLICAS"
mkdir -p experiments/m4-cloud
MASTER_ADDR="$TS_IP" MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
GLOO_SOCKET_IFNAME=tailscale0 \
REPLICA_GROUP_ID="$REPLICA_ID" NUM_REPLICA_GROUPS="$N_REPLICAS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHFT_LIGHTHOUSE="$LIGHTHOUSE" FTD_ADVERTISE_HOST="$TS_IP" \
nohup python3 -m ftdiloco.train --config configs/train/m1_diloco.yaml \
  --set run_id="${RUN_ID:-m4-cloud}" --set sync_every="${H:-100}" --set max_steps="${STEPS:-3000}" \
  --set device="${DEVICE:-cuda}" --set quorum_timeout_s=600 --set pg_timeout_s=600 \
  > experiments/m4-cloud/worker$REPLICA_ID.log 2>&1 &
echo "BOOTSTRAP_DONE replica=$REPLICA_ID ts_ip=$TS_IP"
