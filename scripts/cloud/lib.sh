#!/usr/bin/env bash
# Shared helpers for cloud orchestration. source this.
VAST=".venv/bin/vastai"
TS_KEY_FILE="$HOME/.config/ftd_ts_authkey"
PUBKEY_FILE="$HOME/.ssh/id_rsa.pub"
KEY="$HOME/.ssh/id_rsa"
VM_TEMPLATE="b7942f6bbc4374893ff66eb78145bbac"  # official "Ubuntu 22.04 VM"
LIGHTHOUSE="http://100.86.208.63:29510"

# vm_ssh <instance_id> -> echoes "HOST PORT" (direct public IP + mapped 22)
vm_ssh() {
  $VAST show instance "$1" --raw 2>/dev/null | python3 -c '
import json,sys,re
raw=sys.stdin.read(); m=re.search(r"\{.*\}",raw,re.S)
d=json.loads(m.group(0))
ports=d.get("ports") or {}
p=ports.get("22/tcp")
print(d.get("public_ipaddr",""), p[0]["HostPort"] if p else "")
'
}

vm_status() {  # echoes actual_status
  $VAST show instance "$1" --raw 2>/dev/null | python3 -c '
import json,sys,re
raw=sys.stdin.read(); m=re.search(r"\{.*\}",raw,re.S)
print(json.loads(m.group(0)).get("actual_status","?"))
'
}

vm_run() {  # vm_run <id> <remote cmd> ; ssh via direct IP, retries
  local id="$1"; shift
  read -r host port < <(vm_ssh "$id")
  [ -z "$port" ] && { echo "no ssh port for $id" >&2; return 1; }
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i "$KEY" -p "$port" "root@$host" "$@" 2>&1 | grep -vE "Welcome to|Have fun|Warning: Permanently"
}
