#!/usr/bin/env bash
# Record the kill-a-node demo as an asciinema cast. Run ON worker4 after
# scripts/launch_m2.sh <run> is warm. Produces /tmp/ftd-demo.cast; render to GIF
# on the Mac with `agg /tmp/ftd-demo.cast demo.gif`.
#
#   bash scripts/record_gif.sh m2-gif [duration_s]
#
# Layout (tmux session "demo", 4 panes):
#   ┌──────────────┬──────────────┐
#   │ worker 0     │ worker 1     │   live tickers (status + loss sparkline)
#   ├──────────────┼──────────────┤
#   │ cluster      │ $ shell      │   cluster view + the pane where the kill happens
#   └──────────────┴──────────────┘
# The "actor" keystrokes are injected with tmux send-keys (typed live on camera).
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-m2-gif}"
DUR="${2:-210}"
PY=/srv/fpga/ft-diloco/.venv/bin/python

tmux kill-session -t demo 2>/dev/null || true
tmux new-session -d -s demo -x 130 -y 34
tmux split-window -h -t demo
tmux split-window -v -t demo:0.0
tmux split-window -v -t demo:0.1
# panes: 0.0 top-left, 0.2 bottom-left, 0.1 top-right, 0.3 bottom-right
tmux send-keys -t demo:0.0 "$PY analysis/ticker.py --run experiments/$RUN --replica 0 --name 'worker 0 (GPU)'" C-m
tmux send-keys -t demo:0.1 "$PY analysis/ticker.py --run experiments/$RUN --replica 1 --name 'worker 1 (GPU)'" C-m
tmux send-keys -t demo:0.2 "$PY analysis/ticker.py --run experiments/$RUN --cluster" C-m
tmux send-keys -t demo:0.3 "clear" C-m

# Drive the demo: type the kill into the shell pane mid-recording, relaunch later.
(
  sleep 30
  PID=$($PY -c "
from pathlib import Path
import sys; sys.path.insert(0, 'chaos')
from faults import resolve_pid
print(resolve_pid(Path('experiments/$RUN'), 1))
")
  CMD="kill -9 $PID   # worker 1 dies mid-training"
  for ((i = 0; i < ${#CMD}; i++)); do
    tmux send-keys -t demo:0.3 -l "${CMD:$i:1}"
    sleep 0.06
  done
  sleep 1.2
  tmux send-keys -t demo:0.3 C-m
  sleep 75
  CMD2="bash scripts/launch_m2_replica.sh $RUN 1   # bring it back"
  for ((i = 0; i < ${#CMD2}; i++)); do
    tmux send-keys -t demo:0.3 -l "${CMD2:$i:1}"
    sleep 0.05
  done
  sleep 1
  tmux send-keys -t demo:0.3 C-m
) &
DRIVER=$!

# Record the composed layout for DUR seconds.
.venv/bin/asciinema rec --overwrite --command "timeout $DUR tmux attach -t demo" /tmp/ftd-demo.cast
wait "$DRIVER" 2>/dev/null || true
tmux kill-session -t demo 2>/dev/null || true
echo "CAST_DONE /tmp/ftd-demo.cast"
