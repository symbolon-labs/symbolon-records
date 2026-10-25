#!/bin/zsh
# Usage: ./run_followup.sh <out_dir> <limit_min> [followup.py arguments]
# Checks the preconditions, then runs one followup.py job under caffeinate. Log: <out_dir>.log (tee -a).
# If the job is still active after <limit_min> minutes, the wrapper stops it (PLAN.md section 5).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
OUT=$1; LIMIT_MIN=$2; shift 2

echo "== preconditions $(date '+%Y-%m-%d %H:%M:%S')"
OTHER=$(ps ax -o pid=,command= | grep -E 'harness\.py|smcheck\.py|latent_capacity|latent_r06|pilot\.py|followup\.py' | grep -v grep)
if [[ -n "$OTHER" ]]; then echo "ABORT: another GPU job runs:"; echo "$OTHER"; exit 1; fi
echo "python processes:"; ps ax -o pid=,etime=,command= | grep -i '[p]ython' || echo "  none"
SWAP=$(sysctl -n vm.swapusage); echo "swap: $SWAP"
USED=$(echo "$SWAP" | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
if (( ${USED%.*} > 2048 )); then echo "ABORT: swap used ${USED}M > 2048M"; exit 1; fi
AC=$(pmset -g ac); echo "$AC"
WATT=$(echo "$AC" | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p')
if [[ -z "$WATT" || "$WATT" -lt 60 ]]; then echo "ABORT: charger wattage '${WATT}' < 60 W"; exit 1; fi

echo "== start $(date '+%Y-%m-%d %H:%M:%S') (time limit ${LIMIT_MIN} min)"
PYTORCH_ENABLE_MPS_FALLBACK=0 $PY followup.py --out "$OUT" "$@" > >(tee -a "$OUT.log") 2>&1 &
PID=$!
caffeinate -i -w $PID &
( sleep $(( LIMIT_MIN * 60 )); if kill -0 $PID 2>/dev/null; then
    echo "== STOP $(date '+%Y-%m-%d %H:%M:%S'): time limit ${LIMIT_MIN} min reached, kill $PID"; kill $PID; fi ) &
WD=$!
echo "pid $PID (caffeinate -i -w $PID; watchdog $WD)"
wait $PID
RC=$?
pkill -P $WD 2>/dev/null; kill $WD 2>/dev/null
echo "== end $(date '+%Y-%m-%d %H:%M:%S') rc=$RC"
echo "swap after: $(sysctl -n vm.swapusage)"
exit $RC
