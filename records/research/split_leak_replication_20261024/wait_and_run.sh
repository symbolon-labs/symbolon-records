#!/bin/zsh
# Waiter and job list of the replication (PREREGISTRATION.md, Part 9). Start it after the freeze, in the background.
#   1. Wait until ../bridge_safety_deep_20261024/DONE exists and no process of that folder runs (a Python process
#      whose command names the folder, or whose current directory is in the folder). Check every 60 s; log every
#      10 min. After 1,440 checks (about 24 h) with no start, stop with rc 7.
#   2. Wait 120 s more, then run the seed jobs in the order of SEEDS and the analysis, each through run_sleak.sh
#      (run_sleak.sh does the machine checks and waits while a check fails).
#   3. Stop at the first job with rc != 0. Write logs/waiter.log.
# Lock: one waiter at a time (logs/.waiter_lock with the pid; a lock of a dead pid is stale).
# Self-test (test_run_sleak.zsh only): with SLEAK_TEST_BR and SLEAK_TEST_LOGDIR set, the waiter watches the folder
# SLEAK_TEST_BR, checks every 1 s, stops after SLEAK_TEST_MAXWAIT_N checks (default 30) and exits 0 at step 2
# without a job.
set -u
cd "$(dirname "$0")"
SEEDS=(20261121 20261122 20261123 20261124 20261125 20261126)
SEED_LIMIT_MIN=100
ANALYSIS_LIMIT_MIN=30
BR=$(cd .. && pwd)/bridge_safety_deep_20261024
LOGDIR=logs; STEP_S=60; MAXN=1440; TEST=0
if [[ -n "${SLEAK_TEST_BR:-}" ]]; then
  TEST=1; BR=$SLEAK_TEST_BR; LOGDIR=${SLEAK_TEST_LOGDIR:?}; STEP_S=1; MAXN=${SLEAK_TEST_MAXWAIT_N:-30}
fi
BR=${BR:A}  # the real path (lsof gives real paths)
mkdir -p "$LOGDIR"
LOG=$LOGDIR/waiter.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
LOCK="$LOGDIR/.waiter_lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OPID=$(cat "$LOCK/pid" 2>/dev/null)
  if [[ -n "$OPID" ]] && kill -0 "$OPID" 2>/dev/null; then say "another waiter runs (pid $OPID); stop"; exit 1; fi
  rm -rf "$LOCK"; mkdir "$LOCK" || { say "cannot make $LOCK; stop"; exit 1; }
  say "note: stale waiter lock of pid '$OPID' removed"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
folder_procs() {  # pids of Python processes of the watched folder
  local p c d
  for p in $(ps ax -o pid=,command= | awk 'tolower($2) ~ /python/ {print $1}'); do
    c=$(ps -p $p -o command= 2>/dev/null)
    d=$(lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
    if [[ "$c" == *${BR:t}* || "$d" == "$BR" || "$d" == "$BR"/* ]]; then echo -n "$p "; fi
  done
}

say "waiter start (pid $$); waits for $BR/DONE and no process of that folder; then seeds ${SEEDS[*]} and the analysis"
N=0
while true; do
  P=$(folder_procs)
  if [[ -f "$BR/DONE" && -z "${P// /}" ]]; then break; fi
  if (( N % 10 == 0 )); then
    say "waiting: DONE $([[ -f "$BR/DONE" ]] && echo exists || echo absent); folder exists $([[ -d "$BR" ]] && echo yes || echo no); processes '${P}'"
  fi
  if (( N >= MAXN )); then say "stop: no start after $N checks"; exit 7; fi
  sleep $STEP_S; N=$(( N + 1 ))
done
say "DONE exists and no process of that folder runs (after $N checks)"
if (( TEST )); then exit 0; fi
say "wait 120 s"
sleep 120
P=$(folder_procs)
if [[ -n "${P// /}" ]]; then say "a process of the bridge-safety folder runs again: '$P'; stop"; exit 6; fi
for S in "${SEEDS[@]}"; do
  say "seed $S: start"
  ./run_sleak.sh seed $SEED_LIMIT_MIN $S
  RC=$?
  say "seed $S: rc $RC"
  if (( RC != 0 )); then say "stop: seed $S ended with rc $RC"; exit $RC; fi
done
say "analysis: start"
./run_sleak.sh analysis $ANALYSIS_LIMIT_MIN
RC=$?
say "analysis: rc $RC"
if (( RC != 0 )); then exit $RC; fi
say "all jobs done"
exit 0
