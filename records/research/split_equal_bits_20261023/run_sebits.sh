#!/bin/zsh
# Wrapper for one sebits.py job (split learning at equal bits). [researcher] approved the CIFAR-10 download, J0 and
# J1 to J6 on 2026-10-23 (PREREGISTRATION.md, Part 12). Unit tests do not need this wrapper:
#   ../../.venv/bin/python -I -B sebits.py test
#
# Usage: ./run_sebits.sh <job> <limit_min> [sebits.py arguments]    job: data, j0 to j6; limit 1 to 120
#        ./run_sebits.sh --check-only                                 the machine checks only; no job starts
#   ./run_sebits.sh data 10 data/download/cifar-10-binary.tar.gz
#   ./run_sebits.sh j0 120
#   ./run_sebits.sh j2 120 --part 1
#
# Checks before the job (all must pass; the wrapper reports every failed check):
#   - no other heavy job of the lab: a Python process whose command names a lab script, a lab venv or a
#     folder of ember/research (this wrapper's own sebits.py is not counted);
#   - the swap use is known and at most 3,584 MB (sysctl vm.swapusage);
#   - the charger gives 60 W or more (pmset -g ac), the Mac draws from AC power and the battery is above 50%;
#   - 10 GiB or more of free disk; the Python of the venv exists.
# During the job: caffeinate -i -w <pid>; the job log goes to logs/<job>.log and the wrapper log to
# logs/<job>_wrapper.log (tee -a). Every 30 s a guard reads pmset -g batt. If the Mac is on battery or the
# battery is at 50% or less, the guard pauses the job (SIGSTOP). It resumes the job (SIGCONT) when AC power
# returns and the battery is above 50%. The time limit counts running time only. At the limit the guard
# stops the job (SIGTERM).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1; JOB=check
else
  if [[ $# -lt 2 ]]; then echo "usage: $0 <data|j0..j6> <limit_min> [sebits.py arguments] | --check-only"; exit 2; fi
  JOB=$1; LIMIT_MIN=$2; shift 2
  case $JOB in data|j0|j1|j2|j3|j4|j5|j6) ;; *) echo "ABORT: job '$JOB' is not data or j0 to j6"; exit 2;; esac
  if [[ ! $LIMIT_MIN == <-> ]] || (( LIMIT_MIN < 1 || LIMIT_MIN > 120 )); then
    echo "ABORT: limit '$LIMIT_MIN' must be 1 to 120 minutes"; exit 2
  fi
fi
mkdir -p logs
WLOG="logs/${JOB}_wrapper.log"
say() { if (( CHECK_ONLY )); then echo "$@"; else echo "$@" | tee -a "$WLOG"; fi }
power_ok() {  # prints "<source>|<percent>"; returns 0 if AC power and the battery is above 50%
  local B SRC PCT
  B=$(pmset -g batt 2>/dev/null)
  SRC=$(echo "$B" | sed -nE "s/.*drawing from '([^']+)'.*/\1/p" | head -1)
  PCT=$(echo "$B" | sed -nE 's/.*[[:space:]]([0-9]+)%;.*/\1/p' | head -1)
  echo "${SRC}|${PCT}"
  [[ "$SRC" == "AC Power" && -n "$PCT" ]] && (( PCT > 50 ))
}
FAIL=()

say "== preconditions $(date '+%Y-%m-%d %H:%M:%S') job $JOB"
HEAVY='ember/research/|\.venv(_c2c)?/bin/python|pilot\.py|followup\.py|t5\.py|latent_r06|phase0\.py|harness\.py|smcheck\.py|latent_capacity|uconfirm\.py|ugap\.py|bsp_run|marl|beacon|format_gap'
OTHER=$(ps ax -o pid=,command= | awk 'tolower($2) ~ /python/' | grep -E "$HEAVY" | grep -v -E 'sebits\.py|grep' )
if [[ -n "$OTHER" ]]; then say "another heavy job runs:"; say "$OTHER"; FAIL+=("other job"); fi
say "python processes:"; say "$(ps ax -o pid=,etime=,command= | grep -i '[p]ython' | cut -c1-200 || echo '  none')"
SWAP=$(sysctl -n vm.swapusage 2>/dev/null); say "swap: $SWAP"
USED=$(echo "$SWAP" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')
if [[ -z "$USED" ]]; then FAIL+=("swap use unknown"); elif (( ${USED%.*} > 3584 )); then FAIL+=("swap used ${USED}M > 3584M"); fi
AC=$(pmset -g ac 2>/dev/null); say "charger: $(echo "$AC" | tr '\n' ' ')"
WATT=$(echo "$AC" | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p' | head -1)
if [[ -z "$WATT" || "$WATT" -lt 60 ]]; then FAIL+=("charger wattage '${WATT}' < 60 W"); fi
PW=$(power_ok); PWOK=$?; say "power source and battery: $PW"
if (( PWOK != 0 )); then FAIL+=("not on AC power or battery at 50% or less ($PW)"); fi
FREE_KB=$(df -k . | awk 'NR==2 {print $4}'); say "free disk: $(( FREE_KB / 1024 / 1024 )) GiB"
if (( FREE_KB < 10 * 1024 * 1024 )); then FAIL+=("less than 10 GiB free disk"); fi
if [[ ! -x $PY ]]; then FAIL+=("$PY not found"); fi

if (( ${#FAIL[@]} )); then
  for x in "${FAIL[@]}"; do say "ABORT: $x"; done
  exit 1
fi
if (( CHECK_ONLY )); then echo "CHECK-ONLY: all checks pass; no job started"; exit 0; fi

say "== start $(date '+%Y-%m-%d %H:%M:%S') (running-time limit ${LIMIT_MIN} min): sebits.py $JOB $*"
PYTORCH_ENABLE_MPS_FALLBACK=0 $PY -I -B sebits.py "$JOB" "$@" > >(tee -a "logs/$JOB.log") 2>&1 &
PID=$!
caffeinate -i -w $PID &
say "pid $PID (caffeinate -i -w $PID)"
RUN=0; PAUSED=0
while kill -0 $PID 2>/dev/null; do
  sleep 30
  kill -0 $PID 2>/dev/null || break
  PW=$(power_ok); PWOK=$?
  if (( PWOK != 0 )); then
    if (( ! PAUSED )); then kill -STOP $PID; PAUSED=1; say "== PAUSE $(date '+%Y-%m-%d %H:%M:%S'): power $PW"; fi
  else
    if (( PAUSED )); then kill -CONT $PID; PAUSED=0; say "== RESUME $(date '+%Y-%m-%d %H:%M:%S'): power $PW"; fi
    RUN=$(( RUN + 30 ))
  fi
  if (( RUN >= LIMIT_MIN * 60 )); then
    say "== STOP $(date '+%Y-%m-%d %H:%M:%S'): running-time limit ${LIMIT_MIN} min reached, kill $PID"
    kill -CONT $PID 2>/dev/null; kill $PID 2>/dev/null; break
  fi
done
wait $PID
RC=$?
say "== end $(date '+%Y-%m-%d %H:%M:%S') rc=$RC (running time about $(( RUN / 60 )) min)"
say "swap after: $(sysctl -n vm.swapusage)"
exit $RC
