#!/bin/bash
# Wrapper for one ugap.py job (PREREGISTRATION.md G26 to G28). Run each model job and the analysis through it.
#
# Usage: ./run_ugap.sh <job> <limit_min> <ugap.py arguments>    job: a short name; limit: 1 to 120 min of run time
#        ./run_ugap.sh --check-only                             the machine checks only; no job starts
#   ./run_ugap.sh Dp 45 dev --stages gate
#   ./run_ugap.sh S1 120 confirm --stages sender --pool V
#
# Checks before the job (all must pass; the wrapper reports every failed check):
#   - no other heavy job of the lab: no Python process whose command line or working folder is in the project tree
#     (latent-communication-recovery), and no process with a known job name;
#   - the swap use is known and at most 3,584 MB (sysctl vm.swapusage; sealed run P32);
#   - the Mac draws from AC power, the charger gives 60 W or more (pmset -g ac) and the battery is at 55% or more;
#   - 10 GiB or more of free disk; the Python of the venv exists.
# Then: caffeinate -i -w <pid>; the job log goes to logs/<job>.log (tee -a), the wrapper log to
# logs/<job>_wrapper.log and the guard log to logs/<job>_guard.log; the models load offline; guard.sh pauses the job
# on battery power or at 50% battery or less, and stops it at <limit_min> minutes of run time.
# Exit code: the code of ugap.py; 124 if the guard stopped the job at its limit; 125 if it stopped it for the battery.
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
TAG=latent-communication-recovery
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1; JOB=check
else
  if [[ $# -lt 3 ]]; then echo "usage: $0 <job> <limit_min> <ugap.py arguments> | --check-only"; exit 2; fi
  JOB=$1; LIMIT_MIN=$2; shift 2
  if ! [[ $JOB =~ ^[A-Za-z0-9_-]+$ ]]; then echo "ABORT: job name '$JOB'"; exit 2; fi
  if ! [[ $LIMIT_MIN =~ ^[0-9]+$ ]] || (( LIMIT_MIN < 1 || LIMIT_MIN > 120 )); then
    echo "ABORT: limit '$LIMIT_MIN' must be 1 to 120 minutes"; exit 2
  fi
fi
mkdir -p logs
WLOG="logs/${JOB}_wrapper.log"
say() { if (( CHECK_ONLY )); then echo "$@"; else echo "$@" | tee -a "$WLOG"; fi }
FAIL=()

say "== preconditions $(date '+%F %T') job $JOB"
OTHER=""
while read -r pid comm; do
  [[ -z "${pid:-}" ]] && continue
  base=${comm##*/}
  [[ $base == [Pp]ython* ]] || continue
  cmd=$(ps -o command= -p "$pid" 2>/dev/null) || continue
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
  if [[ "$cmd" == *"$TAG"* || "$cwd" == *"$TAG"* || "$cmd" =~ (pilot|followup|t5|phase0|harness|smcheck|uconfirm|ugap|bsp_run|bsp_judge)\.py ]]; then
    OTHER+="  pid $pid (cwd $cwd): $cmd"$'\n'
  fi
done < <(ps ax -o pid=,comm=)
if [[ -n "$OTHER" ]]; then say "another heavy job runs:"; say "$OTHER"; FAIL+=("other job"); fi
SWAP=$(sysctl -n vm.swapusage 2>/dev/null); say "swap: $SWAP"
USED=$(echo "$SWAP" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')
if [[ -z "$USED" ]]; then FAIL+=("swap use unknown"); elif (( ${USED%.*} > 3584 )); then FAIL+=("swap used ${USED}M > 3584M"); fi
BATT=$(pmset -g batt 2>/dev/null); say "battery: $(echo "$BATT" | tr '\n' ' ')"
PCT=$(echo "$BATT" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
if ! echo "$BATT" | grep -q "AC Power"; then FAIL+=("not on AC power"); fi
if [[ -z "$PCT" || "$PCT" -lt 55 ]]; then FAIL+=("battery '${PCT}'% < 55%"); fi
AC=$(pmset -g ac 2>/dev/null); say "charger: $(echo "$AC" | tr '\n' ' ')"
WATT=$(echo "$AC" | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p' | head -1)
if [[ -z "$WATT" || "$WATT" -lt 60 ]]; then FAIL+=("charger wattage '${WATT}' < 60 W"); fi
FREE_KB=$(df -k . | awk 'NR==2 {print $4}'); say "free disk: $(( FREE_KB / 1024 / 1024 )) GiB"
if (( FREE_KB < 10 * 1024 * 1024 )); then FAIL+=("less than 10 GiB free disk"); fi
if [[ ! -x $PY ]]; then FAIL+=("$PY not found"); fi

if (( ${#FAIL[@]} )); then
  for x in "${FAIL[@]}"; do say "ABORT: $x"; done
  exit 1
fi
if (( CHECK_ONLY )); then echo "CHECK-ONLY: all checks pass; no job started"; exit 0; fi

rm -f "logs/${JOB}.limit" "logs/${JOB}.battery"
say "== start $(date '+%F %T') (run-time limit ${LIMIT_MIN} min): ugap.py $*"
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  $PY -I -B ugap.py "$@" > >(tee -a "logs/${JOB}.log") 2>&1 &
PID=$!
caffeinate -i -w $PID &
bash ./guard.sh $PID "$LIMIT_MIN" "logs/${JOB}" >> "logs/${JOB}_guard.log" 2>&1 &
GD=$!
say "pid $PID (caffeinate -i -w $PID; guard $GD)"
wait $PID
RC=$?
sleep 1
kill $GD 2>/dev/null
if [[ -f "logs/${JOB}.limit" ]]; then RC=124; say "== STOP: run-time limit reached; the job is incomplete"; fi
if [[ -f "logs/${JOB}.battery" ]]; then RC=125; say "== STOP: battery below 20%; the job is incomplete"; fi
say "== end $(date '+%F %T') rc=$RC"
say "swap after: $(sysctl -n vm.swapusage)"
exit $RC
