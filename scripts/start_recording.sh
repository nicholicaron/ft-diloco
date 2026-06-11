#!/usr/bin/env bash
# Start the GIF recording in a detached tmux session, from the repo root.
# (Exists as a file because nested quoting through ssh+tmux is a proven footgun.)
#   bash scripts/start_recording.sh <run_id> [duration_s]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
DUR="${2:-210}"
tmux kill-session -t ftdrec 2>/dev/null || true
tmux new -d -s ftdrec \
  "cd /srv/fpga/ft-diloco && bash scripts/record_gif.sh $RUN $DUR > /tmp/record.log 2>&1"
echo "RECORDER_STARTED $RUN ${DUR}s $(date -Is)"
