#!/bin/bash
# Usage: mrp_launch.sh <job> <script.py> [script args...]
# Gate (mrp_gate.sh) -> python under caffeinate -i -w, log by tee -a, watchdog (mrp_watchdog.sh).
# Exit code = python's (3 if the gate fails). jobs.py also gets --job <job>.
ROOT=$(cd "$(dirname "$0")" && pwd)
JOB=$1; SCRIPT=$2; shift 2
mkdir -p "$ROOT/runs/logs"
LOG="$ROOT/runs/logs/$JOB.log"
bash "$ROOT/mrp_gate.sh" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "GATE FAILED: job $JOB not started" | tee -a "$LOG"; exit 3; fi
PY="$ROOT/../../.venv/bin/python"
EXTRA=()
[ "$SCRIPT" = "jobs.py" ] && EXTRA=(--job "$JOB")
"$PY" -B "$ROOT/$SCRIPT" "$@" "${EXTRA[@]}" > >(tee -a "$LOG") 2>&1 &
PID=$!
echo "[launch $(date '+%F %T')] job=$JOB pid=$PID script=$SCRIPT args=$* ${EXTRA[*]}" | tee -a "$LOG"
caffeinate -i -w "$PID" &
bash "$ROOT/mrp_watchdog.sh" "$PID" >> "$LOG" 2>&1 &
WD=$!
wait "$PID"; RC=$?
kill "$WD" 2>/dev/null
sleep 1
echo "[launch $(date '+%F %T')] job=$JOB exit=$RC" | tee -a "$LOG"
exit $RC
