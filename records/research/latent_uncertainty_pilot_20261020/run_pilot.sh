#!/bin/zsh
# Usage: ./run_pilot.sh <out_dir> [pilot.py arguments]
# Checks the preconditions, then runs one pilot.py job under caffeinate. Log: <out_dir>.log
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
OUT=$1; shift

echo "== preconditions $(date '+%Y-%m-%d %H:%M:%S')"
OTHER=$(ps ax -o pid=,command= | grep -E 'harness\.py|smcheck\.py|latent_capacity|pilot\.py' | grep -v grep)
if [[ -n "$OTHER" ]]; then echo "ABORT: another GPU job runs:"; echo "$OTHER"; exit 1; fi
echo "python processes:"; ps ax -o pid=,etime=,command= | grep -i '[p]ython' || echo "  none"
SWAP=$(sysctl -n vm.swapusage); echo "swap: $SWAP"
USED=$(echo "$SWAP" | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
if (( ${USED%.*} > 2048 )); then echo "ABORT: swap used ${USED}M > 2048M"; exit 1; fi
AC=$(pmset -g ac); echo "$AC"
WATT=$(echo "$AC" | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p')
if [[ -z "$WATT" || "$WATT" -lt 60 ]]; then echo "ABORT: charger wattage '${WATT}' < 60 W"; exit 1; fi

echo "== start $(date '+%Y-%m-%d %H:%M:%S')"
PYTORCH_ENABLE_MPS_FALLBACK=0 $PY pilot.py --out "$OUT" "$@" > "$OUT.log" 2>&1 &
PID=$!
caffeinate -i -w $PID &
echo "pid $PID (caffeinate -i -w $PID)"
wait $PID
RC=$?
echo "== end $(date '+%Y-%m-%d %H:%M:%S') rc=$RC"
exit $RC
