#!/bin/bash
# EXPLORATORY. Preflight for one long CPU run (T5b PLAN.md, section A 2.3 and section C 1).
# Exit 1 if a blocking process is active or the power rule fails.
set -u
echo "{\"event\":\"preflight\",\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
blocking=$(ps -axo pid,command | grep -E "pilot\.py|followup\.py|uconfirm\.py|phase0\.py|harness\.py|smcheck\.py|latent_capacity|python.*latent_r06" | grep -v grep)
if [ -n "$blocking" ]; then
  echo "PREFLIGHT FAIL: blocking process"
  echo "$blocking"
  exit 1
fi
sysctl vm.swapusage
pmset -g ac | grep -E "Wattage|Adapter Name"
pmset -g batt
watts=$(pmset -g ac | awk -F'= ' '/Wattage/ {gsub(/W/, "", $2); print $2 + 0}')
if pmset -g batt | grep -q "AC Power" && [ "${watts:-0}" -ge 60 ]; then
  echo "PREFLIGHT PASS: AC ${watts} W"
else
  echo "PREFLIGHT FAIL: power (AC ${watts:-none} W)"
  exit 1
fi
