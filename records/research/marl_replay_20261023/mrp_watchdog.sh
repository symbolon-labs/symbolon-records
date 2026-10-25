#!/bin/bash
# Watchdog for one job (stop rule S4). SIGSTOP if on battery, battery <= 50% or charger < 60 W;
# SIGCONT when on AC, battery >= 55% and charger >= 60 W; SIGTERM if battery < 20% or after
# 125 min (backstop; jobs.py stops itself at 115 min).
PID=$1
PAUSED=0
T0=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
  BATT=$(pmset -g batt)
  PCT=$(echo "$BATT" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  if echo "$BATT" | grep -q "AC Power"; then AC=1; else AC=0; fi
  W=$(pmset -g ac | awk -F'= ' '/Wattage/{gsub("W","",$2); print $2}')
  W=${W:-0}
  if [ "$PAUSED" -eq 0 ] && { [ "$AC" -eq 0 ] || [ "$PCT" -le 50 ] || [ "$W" -lt 60 ]; }; then
    kill -STOP "$PID"; PAUSED=1; echo "[watchdog $(date '+%T')] SIGSTOP ac=$AC battery=$PCT% charger=${W}W"
  elif [ "$PAUSED" -eq 1 ] && [ "$AC" -eq 1 ] && [ "$PCT" -ge 55 ] && [ "$W" -ge 60 ]; then
    kill -CONT "$PID"; PAUSED=0; echo "[watchdog $(date '+%T')] SIGCONT ac=$AC battery=$PCT% charger=${W}W"
  fi
  if [ "$PCT" -lt 20 ] || [ $(( $(date +%s) - T0 )) -gt 7500 ]; then
    echo "[watchdog $(date '+%T')] SIGTERM battery=$PCT% elapsed=$(( $(date +%s) - T0 ))s"
    kill -TERM "$PID"; kill -CONT "$PID"; break
  fi
  sleep 20
done
