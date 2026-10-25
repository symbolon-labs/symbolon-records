#!/bin/bash
# Wait for test B v3: its DONE file exists and no process of its folder runs (bsp_*.py, bsp_chain.sh).
# Checks every 60 s. Prints one line at the end.
B3="$(cd "$(dirname "$0")/.." && pwd)/bridge_safety_pair17_20261023"
while true; do
  if [ -f "$B3/DONE" ] && ! ps -axo command= | grep -E "bridge_safety_pair17_20261023/(bsp_[a-z_]+\.py|bsp_chain\.sh|bsp_launch\.sh)" | grep -v grep > /dev/null; then
    echo "[wait $(date '+%F %T')] test B v3 DONE and no bsp process"; exit 0
  fi
  sleep 60
done
