#!/bin/zsh
# Wrapper for one sleak.py job (replication of the leakage result of split_equal_bits_20261023).
# [researcher] approved the runs on 2026-10-23 and 2026-10-24 (PREREGISTRATION.md, Part 12). Unit tests do not need it:
#   ../../.venv/bin/python -I -B sleak.py test
#
# Usage: ./run_sleak.sh seed <limit_min> <run seed>     one seed job (MPS)
#        ./run_sleak.sh analysis <limit_min>            the analysis (CPU; the only job that reads test labels)
#        ./run_sleak.sh --check-only                    the machine checks only; no job starts
#        ./run_sleak.sh selftest <limit_min>            test_run_sleak.zsh only: runs SLEAK_TEST_CMD (no Python job)
#                                                       with the power state from the file SLEAK_TEST_POWER
#
# Lock: one wrapper at a time (the folder <logdir>/.lock with the wrapper's pid; a lock of a dead pid is stale).
# Checks before the job. If a check fails, the wrapper WAITS and checks again every 60 s (it does not fail), for at
# most SLEAK_MAXWAIT_MIN minutes (default 720). Then it stops with rc 5.
#   - no other heavy job of the lab: a Python process whose command names a lab script, a lab venv or a folder of
#     ember/research (another sleak.py job counts too);
#   - the swap use is known and at most 3,584 MB (sysctl vm.swapusage);
#   - the charger gives 60 W or more (pmset -g ac), the Mac draws from AC power and the battery is at 55% or more.
# Hard checks (the wrapper stops at once with rc 1): 10 GiB or more of free disk; the Python of the venv exists.
# During the job: caffeinate -i -w <pid>; the job log goes to logs/<job>.log and the wrapper log to
# logs/<job>_wrapper.log (tee -a). Every 20 s a guard reads pmset. It pauses the job (SIGSTOP) if the Mac is on
# battery power, the battery is at 50% or less, or the charger gives less than 60 W. It resumes the job (SIGCONT) when
# the Mac is on AC power, the battery is at 55% or more and the charger gives 60 W or more. While the job is paused,
# the guard writes a line every 10 min. The time limit counts running time only (a step of more than 60 s counts as
# 60 s). The guard stops the job (SIGTERM) at the time limit (rc 124), after 120 min in one pause (rc 125), or if the
# battery falls below 20% (rc 126).
# Exit code: zsh loses the exit code of a job that was paused (wait gives 145). Thus sleak.py writes its exit code
# into the file SLEAK_EXIT_FILE, and the wrapper uses that file. With no file: the guard's code if the guard stopped
# the job, else the code of wait, but 9 in place of 0.
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
GUARD_S=20; WAIT_S=60; PAUSE_MAX_S=7200; LOGDIR=logs; WAIT_MAX_N=${SLEAK_MAXWAIT_MIN:-720}; SELFTEST=0
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1; JOB=check; NAME=check; LIMIT_MIN=0
else
  if [[ $# -lt 2 ]]; then echo "usage: $0 <seed|analysis> <limit_min> [run seed] | --check-only"; exit 2; fi
  JOB=$1; LIMIT_MIN=$2; shift 2
  case $JOB in
    seed) if [[ $# -ne 1 || ! $1 == <-> ]]; then echo "ABORT: seed needs one run seed"; exit 2; fi; NAME="seed_$1";;
    analysis) if [[ $# -ne 0 ]]; then echo "ABORT: analysis takes no argument"; exit 2; fi; NAME=analysis;;
    selftest)
      if [[ $# -ne 0 || -z "${SLEAK_TEST_CMD:-}" || -z "${SLEAK_TEST_POWER:-}" || -z "${SLEAK_TEST_LOGDIR:-}" ]]; then
        echo "ABORT: selftest needs SLEAK_TEST_CMD, SLEAK_TEST_POWER and SLEAK_TEST_LOGDIR"; exit 2
      fi
      NAME=selftest; SELFTEST=1; GUARD_S=1; WAIT_S=1; LOGDIR=$SLEAK_TEST_LOGDIR
      PAUSE_MAX_S=${SLEAK_TEST_PAUSE_MAX_S:-7200}; WAIT_MAX_N=${SLEAK_TEST_MAXWAIT_N:-60};;
    *) echo "ABORT: job '$JOB' is not seed or analysis"; exit 2;;
  esac
  if [[ ! $LIMIT_MIN == <-> ]] || (( LIMIT_MIN < 1 || LIMIT_MIN > 115 )); then
    echo "ABORT: limit '$LIMIT_MIN' must be 1 to 115 minutes"; exit 2
  fi
fi
LIMIT_S=$(( LIMIT_MIN * 60 ))
if (( SELFTEST )) && [[ -n "${SLEAK_TEST_LIMIT_S:-}" ]]; then LIMIT_S=$SLEAK_TEST_LIMIT_S; fi
WLOG="$LOGDIR/${NAME}_wrapper.log"
say() { if (( CHECK_ONLY )); then echo "$@"; else echo "$@" | tee -a "$WLOG"; fi }
power() {  # sets SRC, PCT, WATT; returns 0 if AC power, battery >= $1 % and charger >= 60 W
  local B
  if (( SELFTEST )); then
    IFS='|' read -r SRC PCT WATT < "$SLEAK_TEST_POWER"
  else
    B=$(pmset -g batt 2>/dev/null)
    SRC=$(echo "$B" | sed -nE "s/.*drawing from '([^']+)'.*/\1/p" | head -1)
    PCT=$(echo "$B" | sed -nE 's/.*[[:space:]]([0-9]+)%;.*/\1/p' | head -1)
    WATT=$(pmset -g ac 2>/dev/null | sed -nE 's/.*Wattage = ([0-9]+)W.*/\1/p' | head -1)
  fi
  WATT=${WATT:-0}
  [[ "$SRC" == "AC Power" && -n "$PCT" ]] && (( PCT >= $1 && WATT >= 60 ))
}
HEAVY='ember/research/|\.venv(_c2c)?/bin/python|pilot\.py|followup\.py|t5\.py|latent_r06|phase0\.py|harness\.py|smcheck\.py|latent_capacity|uconfirm\.py|ugap\.py|bsp_run|marl|beacon|format_gap|sebits\.py|sleak\.py|bridge_safety'
SWAP="(not read)"
soft_checks() {  # fills SOFT with the failed waiting checks
  SOFT=()
  if (( ! SELFTEST )); then
    OTHER=$(ps ax -o pid=,command= | awk 'tolower($2) ~ /python/' | grep -E "$HEAVY" | grep -v grep)
    if [[ -n "$OTHER" ]]; then SOFT+=("another heavy job runs: $(echo "$OTHER" | cut -c1-160 | tr '\n' ';')"); fi
    SWAP=$(sysctl -n vm.swapusage 2>/dev/null)
    USED=$(echo "$SWAP" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')
    if [[ -z "$USED" ]]; then SOFT+=("swap use unknown"); elif (( ${USED%.*} > 3584 )); then SOFT+=("swap used ${USED}M > 3584M"); fi
  fi
  if ! power 55; then SOFT+=("power: source '$SRC', battery ${PCT}%, charger ${WATT} W (need AC, >= 55%, >= 60 W)"); fi
}

if (( ! CHECK_ONLY )); then
  mkdir -p "$LOGDIR"
  LOCK="$LOGDIR/.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    OPID=$(cat "$LOCK/pid" 2>/dev/null)
    if [[ -n "$OPID" ]] && kill -0 "$OPID" 2>/dev/null; then say "ABORT: another run_sleak.sh runs (pid $OPID)"; exit 1; fi
    rm -rf "$LOCK"; mkdir "$LOCK" || { say "ABORT: cannot make $LOCK"; exit 1; }
    say "note: stale lock of pid '$OPID' removed"
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
fi
say "== preconditions $(date '+%Y-%m-%d %H:%M:%S') job $NAME"
HARD=()
FREE_KB=$(df -k . | awk 'NR==2 {print $4}'); say "free disk: $(( FREE_KB / 1024 / 1024 )) GiB"
if (( FREE_KB < 10 * 1024 * 1024 )); then HARD+=("less than 10 GiB free disk"); fi
if [[ ! -x $PY ]]; then HARD+=("$PY not found"); fi
if (( ${#HARD[@]} )); then for x in "${HARD[@]}"; do say "ABORT: $x"; done; exit 1; fi
WAITED=0; LAST=""
while true; do
  soft_checks
  if (( ${#SOFT[@]} == 0 )); then break; fi
  if (( CHECK_ONLY )); then for x in "${SOFT[@]}"; do say "WOULD WAIT: $x"; done; exit 1; fi
  NOW_R="${SOFT[*]}"
  if [[ "$NOW_R" != "$LAST" ]] || (( WAITED % 30 == 0 )); then
    for x in "${SOFT[@]}"; do say "WAIT $(date '+%H:%M:%S') (${WAITED} checks): $x"; done
    LAST="$NOW_R"
  fi
  if (( WAITED >= WAIT_MAX_N )); then say "ABORT: the checks did not pass in ${WAIT_MAX_N} checks"; exit 5; fi
  sleep $WAIT_S; WAITED=$(( WAITED + 1 ))
done
say "python processes:"; say "$(ps ax -o pid=,etime=,command= | grep -i '[p]ython' | cut -c1-200 || echo '  none')"
say "swap: $SWAP"
power 55; say "power source and battery: ${SRC}|${PCT}%|${WATT}W"
if (( CHECK_ONLY )); then echo "CHECK-ONLY: all checks pass; no job started"; exit 0; fi

EXITF="${LOGDIR:A}/${NAME}.exit"
rm -f "$EXITF"
say "== start $(date '+%Y-%m-%d %H:%M:%S') (running-time limit ${LIMIT_S} s): $JOB $*"
if (( SELFTEST )); then
  SLEAK_EXIT_FILE="$EXITF" sh -c "$SLEAK_TEST_CMD" > >(tee -a "$LOGDIR/$NAME.log") 2>&1 &
else
  SLEAK_EXIT_FILE="$EXITF" PYTORCH_ENABLE_MPS_FALLBACK=0 $PY -I -B sleak.py "$JOB" "$@" > >(tee -a "$LOGDIR/$NAME.log") 2>&1 &
fi
PID=$!
caffeinate -i -w $PID &
say "pid $PID (caffeinate -i -w $PID)"
RUN=0; PAUSED=0; PSEC=0; NEXTLOG=600; GSTOP=0; LASTT=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  sleep $GUARD_S
  kill -0 $PID 2>/dev/null || break
  NOWT=$(date +%s); STEP=$(( NOWT - LASTT )); LASTT=$NOWT
  if (( STEP > 60 )); then STEP=60; fi
  if (( PAUSED )); then
    PSEC=$(( PSEC + STEP ))
    if power 55; then
      kill -CONT $PID; PAUSED=0; say "== RESUME $(date '+%Y-%m-%d %H:%M:%S'): ${SRC}|${PCT}%|${WATT}W after ${PSEC}s paused; run ${RUN}s"
    else
      if (( PSEC >= NEXTLOG )); then say "== STILL PAUSED $(date '+%Y-%m-%d %H:%M:%S'): ${SRC}|${PCT}%|${WATT}W for ${PSEC}s"; NEXTLOG=$(( NEXTLOG + 600 )); fi
      if (( PSEC >= PAUSE_MAX_S )); then
        say "== STOP $(date '+%Y-%m-%d %H:%M:%S'): paused for ${PSEC}s (limit ${PAUSE_MAX_S}s), kill $PID"
        GSTOP=125; kill -CONT $PID 2>/dev/null; kill $PID 2>/dev/null; break
      fi
    fi
  else
    RUN=$(( RUN + STEP ))
    if ! power 51; then
      kill -STOP $PID; PAUSED=1; PSEC=0; NEXTLOG=600
      say "== PAUSE $(date '+%Y-%m-%d %H:%M:%S'): ${SRC}|${PCT}%|${WATT}W run ${RUN}s"
    fi
  fi
  if [[ -n "$PCT" ]] && (( PCT < 20 )); then
    say "== STOP $(date '+%Y-%m-%d %H:%M:%S'): battery ${PCT}% < 20%, kill $PID"
    GSTOP=126; kill -CONT $PID 2>/dev/null; kill $PID 2>/dev/null; break
  fi
  if (( RUN >= LIMIT_S )); then
    say "== STOP $(date '+%Y-%m-%d %H:%M:%S'): running-time limit ${LIMIT_S}s reached, kill $PID"
    GSTOP=124; kill -CONT $PID 2>/dev/null; kill $PID 2>/dev/null; break
  fi
done
wait $PID
RC=$?
FRC=""
if [[ -f "$EXITF" ]]; then FRC=$(<"$EXITF"); fi
if [[ "$FRC" == <-> ]]; then
  if (( FRC != RC )); then say "note: wait gave rc $RC; the exit file gives $FRC"; fi
  RC=$FRC
elif (( GSTOP )); then
  say "note: wait gave rc $RC; the guard stopped the job (rc $GSTOP)"; RC=$GSTOP
elif (( RC == 0 )); then
  say "note: no exit file; rc 9 in place of 0"; RC=9
fi
say "== end $(date '+%Y-%m-%d %H:%M:%S') rc=$RC (running time about $(( RUN / 60 )) min)"
if (( ! SELFTEST )); then say "swap after: $(sysctl -n vm.swapusage)"; fi
exit $RC
