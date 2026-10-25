#!/bin/bash
# Main run after J0: the job list comes from runs/j0/decision.json by the plan rule
# (PREREGISTRATION.md section 11). Each job starts only if the job before it exits 0.
ROOT=$(cd "$(dirname "$0")" && pwd)
PY="$ROOT/../../.venv/bin/python"
CLOG="$ROOT/runs/chain.log"
SPECS=$("$PY" -B "$ROOT/jobs.py" plan) || { echo "[chain $(date '+%F %T')] plan REFUSED: $SPECS" >> "$CLOG"; exit 6; }
echo "[chain $(date '+%F %T')] plan: $(echo "$SPECS" | tr '\n' ';')" >> "$CLOG"
while read -r line; do
  [ -z "$line" ] && continue
  set -- $line
  echo "[chain $(date '+%F %T')] start $1" >> "$CLOG"
  bash "$ROOT/mrp_launch.sh" "$@" > /dev/null 2>&1; RC=$?
  echo "[chain $(date '+%F %T')] $1 exit=$RC" >> "$CLOG"
  [ "$RC" -eq 0 ] || exit "$RC"
done <<< "$SPECS"
echo "[chain $(date '+%F %T')] all jobs exit 0" >> "$CLOG"
exit 0
