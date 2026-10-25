#!/bin/zsh
# Self-test of run_sleak.sh (job "selftest": a shell command in place of sleak.py, the power state from a file) and of
# the wait part of wait_and_run.sh (SLEAK_TEST_BR). No Python job of the study starts; no pmset value is used.
# test_sleak.py runs this file. It prints one line for each case and exits 0 if every case passes.
set -u
HERE=${0:A:h}
VPY=$HERE/../../.venv/bin/python
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
OK_P="AC Power|90|140"
setp() { echo "$1" > "$T/power"; }
check() {  # check <case> <got rc> <want rc> [pattern that the case log must contain]
  local log="$T/logs_$1/selftest_wrapper.log"
  if (( $2 != $3 )); then echo "FAIL $1: rc $2, want $3"; cat "$T/out_$1"; FAILS=$(( FAILS + 1 )); return; fi
  if [[ -n "${4:-}" ]] && ! grep -q -- "$4" "$log" 2>/dev/null; then
    echo "FAIL $1: the log has no '$4'"; cat "$T/out_$1"; FAILS=$(( FAILS + 1 )); return
  fi
  echo "ok   $1 (rc $2)"
}
run() {  # run <case> <command> [VAR=value ...]; sets RC
  local name=$1 cmd=$2; shift 2
  env "$@" SLEAK_TEST_CMD="$cmd" SLEAK_TEST_POWER="$T/power" SLEAK_TEST_LOGDIR="$T/logs_$name" \
    "$HERE/run_sleak.sh" selftest 1 > "$T/out_$name" 2>&1
  RC=$?
}
W='echo $X > "$SLEAK_EXIT_FILE"; exit $X'

setp "$OK_P"; run plain "sleep 1; X=7; $W"; check plain $RC 7
setp "$OK_P"; ( sleep 2; setp "Battery Power|90|0"; sleep 3; setp "$OK_P" ) &
run pause0 "sleep 7; X=0; $W"; wait; check pause0 $RC 0 "RESUME"
setp "$OK_P"; ( sleep 2; setp "AC Power|90|8"; sleep 3; setp "$OK_P" ) &
run pause3 "sleep 7; X=3; $W"; wait; check pause3 $RC 3 "PAUSE"
setp "AC Power|90|8"; ( sleep 3; setp "$OK_P" ) &
run startwait "X=0; $W"; wait; check startwait $RC 0 "WAIT"
setp "AC Power|90|8"; run waitlimit "X=0; $W" SLEAK_TEST_MAXWAIT_N=3; check waitlimit $RC 5 "did not pass"
setp "$OK_P"; ( sleep 2; setp "AC Power|90|8" ) &
run pauselimit "sleep 9; X=0; $W" SLEAK_TEST_PAUSE_MAX_S=3; wait; check pauselimit $RC 125 "paused for"
setp "$OK_P"; run timelimit "sleep 9; X=0; $W" SLEAK_TEST_LIMIT_S=3; check timelimit $RC 124 "running-time limit"
setp "$OK_P"; ( sleep 2; setp "Battery Power|15|0" ) &
run battery "sleep 9; X=0; $W" SLEAK_TEST_PAUSE_MAX_S=60; wait; check battery $RC 126 "< 20%"
setp "$OK_P"; run noexit0 "exit 0"; check noexit0 $RC 9 "no exit file"
setp "$OK_P"; run noexit4 "exit 4"; check noexit4 $RC 4
mkdir -p "$T/logs_lock/.lock"; echo $$ > "$T/logs_lock/.lock/pid"
setp "$OK_P"; run lock "X=0; $W"; check lock $RC 1 "another run_sleak.sh runs"
mkdir -p "$T/logs_stale/.lock"; echo 999999 > "$T/logs_stale/.lock/pid"
setp "$OK_P"; run stale "X=0; $W"; check stale $RC 0 "stale lock"

wcheck() {  # wcheck <case> <got rc> <want rc> <pattern>
  if (( $2 == $3 )) && grep -q -- "$4" "$T/wlogs_$1/waiter.log"; then echo "ok   waiter $1 (rc $2)"
  else echo "FAIL waiter $1: rc $2, want $3, pattern '$4'"; cat "$T/wlogs_$1/waiter.log"; FAILS=$(( FAILS + 1 )); fi
}
wrun() {  # wrun <case> [VAR=value ...]; sets RC
  local name=$1; shift
  env "$@" SLEAK_TEST_BR="$T/bridge_folder_x" SLEAK_TEST_LOGDIR="$T/wlogs_$name" "$HERE/wait_and_run.sh" > /dev/null 2>&1
  RC=$?
}
mkdir -p "$T/bridge_folder_x"
wrun nodone SLEAK_TEST_MAXWAIT_N=3; wcheck nodone $RC 7 "no start after"
touch "$T/bridge_folder_x/DONE"; wrun done; wcheck done $RC 0 "DONE exists"
( cd "$T/bridge_folder_x" && "$VPY" -I -c 'import time; time.sleep(4)' ) &
sleep 1; T0=$(date +%s); wrun cwdproc; DT=$(( $(date +%s) - T0 )); wait
wcheck cwdproc $RC 0 "processes '[0-9]"
if (( DT < 2 )); then echo "FAIL waiter cwdproc: it did not wait (${DT} s)"; FAILS=$(( FAILS + 1 )); fi
"$VPY" -I -c 'import time; time.sleep(4)' "$T/bridge_folder_x/job.py" &
sleep 1; wrun cmdproc; wait; wcheck cmdproc $RC 0 "processes '[0-9]"

echo "failures: $FAILS"
exit $(( FAILS > 0 ))
