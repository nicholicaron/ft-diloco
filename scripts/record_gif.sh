#!/usr/bin/env bash
# Record the kill-a-node demo as an asciinema cast. Run ON worker4 (via
# scripts/start_recording.sh) after scripts/launch_m2.sh <run> is warm.
# Produces /tmp/ftd-demo.cast; render on the Mac with agg.
#
#   bash scripts/record_gif.sh m2-gif [duration_s]
#
# 2x2 tiled layout, pane ids captured explicitly (tmux renumbers on split):
#   worker 0 ticker | worker 1 ticker
#   cluster view    | shell (the kill happens here, typed live)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-m2-gif}"
DUR="${2:-240}"
PY=/srv/fpga/ft-diloco/.venv/bin/python

tmux kill-session -t demo 2>/dev/null || true
P0=$(tmux new-session -d -s demo -x 130 -y 34 -P -F '#{pane_id}')
P1=$(tmux split-window -h -t "$P0" -P -F '#{pane_id}')
P2=$(tmux split-window -v -t "$P0" -P -F '#{pane_id}')
P3=$(tmux split-window -v -t "$P1" -P -F '#{pane_id}')
tmux select-layout -t demo tiled

tmux send-keys -t "$P0" "$PY analysis/ticker.py --run experiments/$RUN --replica 0 --name 'worker 0 (GPU)'" C-m
tmux send-keys -t "$P1" "$PY analysis/ticker.py --run experiments/$RUN --replica 1 --name 'worker 1 (GPU)'" C-m
tmux send-keys -t "$P2" "$PY analysis/ticker.py --run experiments/$RUN --cluster" C-m
tmux send-keys -t "$P3" "clear" C-m

# Drive the demo: kill at +20s, relaunch at +80s, recovery visible by ~+150s.
(
  sleep 20
  PID=$($PY -c "
from pathlib import Path
import sys; sys.path.insert(0, 'chaos')
from faults import resolve_pid
print(resolve_pid(Path('experiments/$RUN'), 1))
")
  CMD="kill -9 $PID   # worker 1 dies mid-training"
  for ((i = 0; i < ${#CMD}; i++)); do
    tmux send-keys -t "$P3" -l "${CMD:$i:1}"
    sleep 0.06
  done
  sleep 1.2
  tmux send-keys -t "$P3" C-m
  sleep 58
  CMD2="bash scripts/launch_m2_replica.sh $RUN 1   # bring it back"
  for ((i = 0; i < ${#CMD2}; i++)); do
    tmux send-keys -t "$P3" -l "${CMD2:$i:1}"
    sleep 0.05
  done
  sleep 1
  tmux send-keys -t "$P3" C-m
) &
DRIVER=$!

# Record the composed layout; TMUX= allows the nested attach.
TMUX= .venv/bin/asciinema rec --overwrite \
  --command "timeout $DUR tmux attach -t demo" /tmp/ftd-demo.cast
wait "$DRIVER" 2>/dev/null || true
tmux kill-session -t demo 2>/dev/null || true

# Trim any post-timeout teardown frames ([terminated] flash) from the cast tail.
$PY - <<EOF
import json
lines = open("/tmp/ftd-demo.cast").read().splitlines()
header, events = lines[0], [json.loads(line) for line in lines[1:] if line]
cutoff = $DUR - 0.5
keep = [e for e in events if e[0] < cutoff]
with open("/tmp/ftd-demo.cast", "w") as f:
    f.write(header + "\n")
    for e in keep:
        f.write(json.dumps(e) + "\n")
print(f"trimmed {len(events) - len(keep)} tail events")
EOF
echo "CAST_DONE /tmp/ftd-demo.cast"
