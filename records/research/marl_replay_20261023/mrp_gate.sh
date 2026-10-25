#!/bin/bash
# Gate before each job (PREREGISTRATION.md section 13). Exit 0 only if all checks pass.
# Checks: no other heavy lab job; AC power, battery > 50%, charger >= 60 W; swap used < 2000 MB;
# free disk >= 10 GB; FREEZE.json hashes equal the files; unit tests pass (stop rule S1).
ROOT=$(cd "$(dirname "$0")" && pwd)
PY="$ROOT/../../.venv/bin/python"
PAT='bsp_run\.py|bsp_judge\.py|bsj_run|bsj_judge|bsr_run|pilot\.py|followup\.py|t5\.py|t5b\.py|latent_r06|phase0\.py|uconfirm\.py|ugap|canary_c2c\.py|lim_run\.py|harness\.py|smcheck\.py|latent_capacity|\.venv_c2c|marl_replay_20261023/jobs\.py'
fail=0
echo "[gate $(date '+%F %T')]"
BUSY=$(ps -axo pid=,command= | grep -E "$PAT" | grep -v -E "grep -E")
if [ -n "$BUSY" ]; then echo "  busy: $(echo "$BUSY" | cut -c1-200)"; fail=1; else echo "  lab jobs: none"; fi
HOT=$(ps -axo pid=,pcpu=,command= | awk '$2 > 50 && /[Pp]ython/' | grep -v "mrp_gate")
if [ -n "$HOT" ]; then echo "  busy python (>50% CPU): $(echo "$HOT" | cut -c1-200)"; fail=1; else echo "  hot python: none"; fi
BATT=$(pmset -g batt)
PCT=$(echo "$BATT" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
if echo "$BATT" | grep -q "AC Power"; then AC=1; else AC=0; fi
W=$(pmset -g ac | awk -F'= ' '/Wattage/{gsub("W","",$2); print $2}')
echo "  power: ac=$AC battery=${PCT}% charger=${W:-0}W"
[ "$AC" -eq 1 ] || { echo "  FAIL: not on AC"; fail=1; }
[ "${PCT:-0}" -gt 50 ] || { echo "  FAIL: battery <= 50%"; fail=1; }
[ "${W:-0}" -ge 60 ] || { echo "  FAIL: charger < 60 W"; fail=1; }
SWAP=$(sysctl -n vm.swapusage)
USED=$(echo "$SWAP" | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
echo "  swap: $SWAP"
awk -v u="$USED" 'BEGIN{exit !(u < 2000)}' || { echo "  FAIL: swap used >= 2000 MB"; fail=1; }
FREE=$(df -g "$ROOT" | awk 'NR==2{print $4}')
echo "  free disk: ${FREE} GB"
[ "${FREE:-0}" -ge 10 ] || { echo "  FAIL: free disk < 10 GB"; fail=1; }
FOUT=$("$PY" -B "$ROOT/freeze.py" verify 2>&1); FRC=$?
echo "  freeze: $(echo "$FOUT" | tail -1)"
[ "$FRC" -eq 0 ] || { echo "  FAIL: freeze check"; fail=1; }
TOUT=$("$PY" -B "$ROOT/test_marl.py" 2>&1); TRC=$?
echo "  unit tests: $(echo "$TOUT" | grep -E '^Ran|^OK|^FAILED' | tr '\n' ' ')"
[ "$TRC" -eq 0 ] || { echo "  FAIL: unit tests (S1)"; fail=1; }
echo "  gate result: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
