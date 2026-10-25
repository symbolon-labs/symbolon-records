#!/bin/bash
# EXPLORATORY. Launch one job with preflight, caffeinate, power guard and an appended log.
# Usage: launch.sh smoke | launch.sh t5b | launch.sh c1b
set -u
cd "$(dirname "$0")"
job=$1
log="run_${job}.log"
PY=.venv/bin/python
SCRIPT="$(pwd)/t5b.py"
./preflight.sh 2>&1 | tee -a "$log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "{\"event\":\"launch_refused\",\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$log"
  exit 1
fi
if [ "$job" = "smoke" ]; then
  "$PY" -I -B "$SCRIPT" smoke 2>&1 | tee -a "$log" &
  pidfile="smoke.pid"
else
  "$PY" -I -B "$SCRIPT" run --arm "$job" 2>&1 | tee -a "$log" &
  pidfile="run/$job/pid"
fi
pid=""
for _ in $(seq 1 120); do
  if [ -f "$pidfile" ]; then pid=$(cat "$pidfile"); break; fi
  sleep 1
done
if [ -z "$pid" ]; then
  echo "{\"event\":\"launch_no_pid\",\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | tee -a "$log"
  wait
  exit 1
fi
echo "{\"event\":\"launch\",\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"job\":\"$job\",\"pid\":$pid}" | tee -a "$log"
caffeinate -i -w "$pid" &
./guard.sh "$pid" "$log" &
wait
