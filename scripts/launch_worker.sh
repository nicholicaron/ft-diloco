#!/usr/bin/env bash
# One DiLoCo replica group. Usage:
#   scripts/launch_worker.sh <replica_id> <num_replicas> <train-config> [--set k=v ...]
# Env: TORCHFT_LIGHTHOUSE (default http://worker1:29510)
set -euo pipefail
cd "$(dirname "$0")/.."
export REPLICA_GROUP_ID="$1"; shift
export NUM_REPLICA_GROUPS="$1"; shift
CONFIG="$1"; shift
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://worker1:29510}"
exec .venv/bin/python -m ftdiloco.train --config "$CONFIG" "$@"
