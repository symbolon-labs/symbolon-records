#!/bin/zsh
# DRAFT (EXPLORATORY code, not frozen). Wrapper for one uconfirm.py job. Run a job only after [researcher]'s yes for that job
# (APPROVAL_REQUEST.md). The unit tests do not need this wrapper: ../../.venv/bin/python -I -B uconfirm.py test
#
# Usage: ./run_uconfirm.sh <mode> <limit_min> [uconfirm.py arguments]     mode: smoke, dev or confirm; limit 1 to 120
#        ./run_uconfirm.sh --check-only                                   the machine checks only; no job starts
#   ./run_uconfirm.sh smoke 45
#   ./run_uconfirm.sh dev 120 --stages sender,gates
#
# Checks before the job (all must pass; the wrapper reports every failed check):
#   - no other heavy job of the lab: pilot.py, followup.py, t5.py, latent_r06, phase0.py, harness.py, smcheck.py,
#     latent_capacity, uconfirm.py;
#   - the swap use is known and at most 3,584 MB (sysctl vm.swapusage; was 2,048 MB until 2026-10-21 16:05);
#   - the charger gives 60 W or more (pmset -g ac);
#   - 10 GiB or more of free disk; the two data files exist; the Python of the venv exists.
# Then: caffeinate -i -w <pid>; the job log goes to <mode>.log and the wrapper log to <mode>_wrapper.log (tee -a);
# the models load offline; a watchdog stops the job after <limit_min> minutes (design section 17: 120 min at most).
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1; MODE=check
else
  if [[ $# -lt 2 ]]; then echo "usage: $0 <smoke|dev|confirm> <limit_min> [uconfirm.py arguments] | --check-only"; exit 2; fi
  MODE=$1; LIMIT_MIN=$2; shift 2
  case $MODE in smoke|dev|confirm) ;; *) echo "ABORT: mode '$MODE' is not smoke, dev or confirm"; exit 2;; esac
  if [[ ! $LIMIT_MIN == <-> ]] || (( LIMIT_MIN < 1 || LIMIT_MIN > 120 )); then
    echo "ABORT: limit '$LIMIT_MIN' must be 1 to 120 minutes"; exit 2
  fi
fi
WLOG="${MODE}_wrapper.log"
say() { if (( CHECK_ONLY )); then echo "$@"; else echo "$@" | tee -a "$WLOG"; fi }
FAIL=()

say "== preconditions $(date '+%Y-%m-%d %H:%M:%S') mode $MODE"
OTHER=$(ps ax -o pid=,command= | grep -E 'pilot\.py|followup\.py|t5\.py|latent_r06|phase0\.py|harness\.py|smcheck\.py|latent_capacity|uconfirm\.py' | grep -v grep)
if [[ -n "$OTHER" ]]; then say "another heavy job runs:"; say "$OTHER"; FAIL+=("other job"); fi
say "python processes:"; say "$(ps ax -o pid=,etime=,command= | grep -i '[p]ython' || echo '  none')"
SWAP=$(sysctl -n vm.swapusage 2>/dev/null); say "swap: $SWAP"
USED=$(echo "$SWAP" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')
if [[ -z "$USED" ]]; then FAIL+=("swap use unknown"); elif (( ${USED%.*} > 3584 )); then FAIL+=("swap used ${USED}M > 3584M"); fi
AC=$(pmset -g ac 2>/dev/null); say "charger: $(echo "$AC" | tr '\n' ' ')"
WATT=$(echo "$AC" | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p' | head -1)
if [[ -z "$WATT" || "$WATT" -lt 60 ]]; then FAIL+=("charger wattage '${WATT}' < 60 W"); fi
FREE_KB=$(df -k . | awk 'NR==2 {print $4}'); say "free disk: $(( FREE_KB / 1024 / 1024 )) GiB"
if (( FREE_KB < 10 * 1024 * 1024 )); then FAIL+=("less than 10 GiB free disk"); fi
for f in data/trivia_qa/rc.wikipedia/validation-00000-of-00001.parquet data/trivia_qa/rc.wikipedia/train-00000-of-00007.parquet; do
  if [[ ! -f $f ]]; then FAIL+=("$f missing (fetch_data.py needs [researcher]'s yes)"); fi
done
if [[ ! -x $PY ]]; then FAIL+=("$PY not found"); fi

if (( ${#FAIL[@]} )); then
  for x in "${FAIL[@]}"; do say "ABORT: $x"; done
  exit 1
fi
if (( CHECK_ONLY )); then echo "CHECK-ONLY: all checks pass; no job started"; exit 0; fi

say "== start $(date '+%Y-%m-%d %H:%M:%S') (time limit ${LIMIT_MIN} min): uconfirm.py $MODE $*"
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  $PY -I -B uconfirm.py "$MODE" "$@" > >(tee -a "$MODE.log") 2>&1 &
PID=$!
caffeinate -i -w $PID &
( sleep $(( LIMIT_MIN * 60 )); if kill -0 $PID 2>/dev/null; then
    echo "== STOP $(date '+%Y-%m-%d %H:%M:%S'): time limit ${LIMIT_MIN} min reached, kill $PID" | tee -a "$WLOG"; kill $PID; fi ) &
WD=$!
say "pid $PID (caffeinate -i -w $PID; watchdog $WD)"
wait $PID
RC=$?
pkill -P $WD 2>/dev/null; kill $WD 2>/dev/null
say "== end $(date '+%Y-%m-%d %H:%M:%S') rc=$RC"
say "swap after: $(sysctl -n vm.swapusage)"
exit $RC
