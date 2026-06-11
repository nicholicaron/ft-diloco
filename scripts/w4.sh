#!/usr/bin/env bash
# Run a command on worker4, retrying across Tailscale link flaps until the
# sentinel "W4_DONE" comes back. Falls back to jumping via worker1 over the LAN
# when the direct Tailscale path is down. Usage: scripts/w4.sh '<remote command>'
# The remote command runs under bash -c on worker4 from /srv/fpga/ft-diloco.
set -uo pipefail
CMD="${1:?remote command}"
TRIES="${TRIES:-30}"
JUMP=(-o ProxyJump=worker1 -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/coord_to_workers")
for _ in $(seq "$TRIES"); do
  for target in "worker4" "claude@192.168.1.66"; do
    if [ "$target" = "worker4" ]; then
      out=$(ssh -o ConnectTimeout=15 worker4 "cd /srv/fpga/ft-diloco && { $CMD ; } && echo W4_DONE" 2>/dev/null)
    else
      out=$(ssh -o ConnectTimeout=15 "${JUMP[@]}" "$target" "cd /srv/fpga/ft-diloco && { $CMD ; } && echo W4_DONE" 2>/dev/null)
    fi
    if printf '%s' "$out" | grep -q "W4_DONE"; then
      printf '%s\n' "$out" | grep -v "^W4_DONE$"
      exit 0
    fi
  done
  sleep 45
done
echo "W4_GAVE_UP after $TRIES tries" >&2
exit 1
