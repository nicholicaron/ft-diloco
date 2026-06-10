#!/usr/bin/env bash
# Lighthouse coordinator — run on worker1.
# Port/flags verified at M0.5 against the pinned torchft commit.
set -euo pipefail
BIND_PORT="${LIGHTHOUSE_PORT:-29510}"
exec env RUST_BACKTRACE=1 RUST_LOG="${RUST_LOG:-info}" torchft_lighthouse \
  --bind "[::]:${BIND_PORT}" \
  --min_replicas "${MIN_REPLICAS:-1}" \
  --quorum_tick_ms 100 \
  --join_timeout_ms 10000
