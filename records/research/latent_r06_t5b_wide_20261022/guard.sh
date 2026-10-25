#!/bin/bash
# EXPLORATORY. Power guard (PLAN.md section 4, step 5).
# Every 60 s: SIGSTOP the job on battery or at 50% battery or less;
# SIGCONT only on AC power at 60 W or more with the battery above 50%.
# Usage: guard.sh <pid> <log>
set -u
pid=$1
log=$2
paused=0
while kill -0 "$pid" 2>/dev/null; do
  batt=$(pmset -g batt)
  ac=$(echo "$batt" | grep -c "AC Power")
  pct=$(echo "$batt" | grep -Eo "[0-9]+%" | head -1 | tr -d %)
  watts=$(pmset -g ac | awk -F'= ' '/Wattage/ {gsub(/W/, "", $2); print $2 + 0}')
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ "$paused" -eq 0 ] && { [ "$ac" -eq 0 ] || [ "${pct:-0}" -le 50 ]; }; then
    kill -STOP "$pid"
    paused=1
    echo "{\"event\":\"guard_pause\",\"utc\":\"$now\",\"pid\":$pid,\"ac\":$ac,\"battery_pct\":${pct:-0},\"watts\":${watts:-0}}" | tee -a "$log"
  elif [ "$paused" -eq 1 ] && [ "$ac" -eq 1 ] && [ "${watts:-0}" -ge 60 ] && [ "${pct:-0}" -gt 50 ]; then
    kill -CONT "$pid"
    paused=0
    echo "{\"event\":\"guard_resume\",\"utc\":\"$now\",\"pid\":$pid,\"battery_pct\":${pct:-0},\"watts\":${watts:-0}}" | tee -a "$log"
  fi
  sleep 60
done
echo "{\"event\":\"guard_end\",\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"pid\":$pid}" | tee -a "$log"
