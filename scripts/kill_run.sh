#!/usr/bin/env bash
# Kill all training processes of one run_id. Pattern is built at runtime so the
# caller's command line never contains it (pkill -f self-match trap).
set -uo pipefail
RUN="${1:?run_id}"
pat="run_id=${RUN}"
pkill -f "$pat" 2>/dev/null
exit 0
