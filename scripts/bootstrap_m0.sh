#!/usr/bin/env bash
# One-shot unattended M0 bootstrap on worker4 (run inside tmux; survives link flaps):
#   1. offline pip install from /srv/fpga/ftd-wheels
#   2. CUDA sanity check
#   3. TinyStories download + tokenize (worker4's own internet; slow but stable)
#   4. tiny50m baseline runs, seeds 1337/1338/1339, sequential
# Progress markers go to stdout (tee'd to /tmp/ftd-m0.log by the launcher).
set -euo pipefail
cd /srv/fpga/ft-diloco

echo "=== [1/4] offline install $(date -Is)"
if ! .venv/bin/python -c "import torch, torchft" 2>/dev/null; then
  .venv/bin/pip install -e ".[prep]" torchft-nightly --no-index --find-links /srv/fpga/ftd-wheels
fi
echo "MARK install_done"

echo "=== [2/4] sanity $(date -Is)"
.venv/bin/python - <<'EOF'
import torch, torchft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().norm().item() > 0)
print("torchft import ok")
EOF
echo "MARK sanity_done"

echo "=== [3/4] tinystories prep $(date -Is)"
if [ ! -f data/tinystories/train.bin ]; then
  .venv/bin/python -m ftdiloco.data --dataset tinystories --out data/tinystories
fi
ls -l --block-size=M data/tinystories/
echo "MARK data_done"

echo "=== [4/4] tiny50m baselines $(date -Is)"
for SEED in 1337 1338 1339; do
  RUN="m0-tiny-s${SEED}"
  if [ -f "experiments/$RUN/checkpoint_final_r0.pt" ]; then
    echo "skip $RUN (already done)"; continue
  fi
  echo "--- $RUN $(date -Is)"
  .venv/bin/python -m ftdiloco.train --config configs/train/m0_tiny.yaml \
    --set run_id="$RUN" --set seed="$SEED"
  echo "MARK baseline_${SEED}_done"
done
echo "MARK all_done $(date -Is)"
