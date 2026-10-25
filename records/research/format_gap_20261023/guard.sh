#!/bin/bash
# Power guard and time limit for one ugap.py job (PREREGISTRATION.md G27). run_ugap.sh starts it.
#
# Usage: guard.sh <pid> <limit_min> <marker_prefix>
#
# Every 20 s:
#   - SIGSTOP (pause) the job if the Mac is on battery power, the battery is at 50% or less, or the charger gives
#     less than 60 W;
#   - SIGCONT (resume) the job when the Mac is on AC power, the battery is at 55% or more and the charger gives 60 W
#     or more.
# The run time counts only while the job is not paused; a step of more than 60 s (for example a system sleep) counts
# as 60 s. At <limit_min> minutes of run time: SIGTERM and the file <marker_prefix>.limit. If the battery falls below
# 20%: SIGTERM and the file <marker_prefix>.battery. If pmset gives no battery percentage, the guard keeps the last
# value (start value 100) and writes a note.
PID=$1
LIMIT_S=$(( $2 * 60 ))
MARK=$3
PAUSED=0
RUN=0
PCT=100
LAST=$(date +%s)
power() {
  BATT=$(pmset -g batt)
  local p
  p=$(echo "$BATT" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  if [ -n "$p" ]; then PCT=$p; else echo "[guard $(date '+%F %T')] note: no battery percentage; keep $PCT%"; fi
  if echo "$BATT" | grep -q "AC Power"; then AC=1; else AC=0; fi
  W=$(pmset -g ac | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p' | head -1)
  W=${W:-0}
}
echo "[guard $(date '+%F %T')] pid $PID; run-time limit $2 min"
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s)
  STEP=$(( NOW - LAST ))
  if [ "$STEP" -gt 60 ]; then STEP=60; fi
  if [ "$PAUSED" -eq 0 ]; then RUN=$(( RUN + STEP )); fi
  LAST=$NOW
  power
  if [ "$PAUSED" -eq 0 ] && { [ "$AC" -eq 0 ] || [ "$PCT" -le 50 ] || [ "$W" -lt 60 ]; }; then
    kill -STOP "$PID"; PAUSED=1
    echo "[guard $(date '+%F %T')] PAUSE (SIGSTOP) ac=$AC battery=$PCT% charger=${W}W run=${RUN}s"
  elif [ "$PAUSED" -eq 1 ] && [ "$AC" -eq 1 ] && [ "$PCT" -ge 55 ] && [ "$W" -ge 60 ]; then
    kill -CONT "$PID"; PAUSED=0
    echo "[guard $(date '+%F %T')] RESUME (SIGCONT) ac=$AC battery=$PCT% charger=${W}W run=${RUN}s"
  fi
  if [ "$PCT" -lt 20 ]; then
    echo "[guard $(date '+%F %T')] STOP (SIGTERM): battery $PCT% < 20%"
    touch "$MARK.battery"; kill -TERM "$PID"; kill -CONT "$PID"; break
  fi
  if [ "$RUN" -ge "$LIMIT_S" ]; then
    echo "[guard $(date '+%F %T')] STOP (SIGTERM): run-time limit $2 min reached"
    touch "$MARK.limit"; kill -TERM "$PID"; kill -CONT "$PID"; break
  fi
  sleep 20
done
echo "[guard $(date '+%F %T')] end; run=${RUN}s"
