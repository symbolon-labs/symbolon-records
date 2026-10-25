#!/bin/bash
# Job chain of the format-gap test (PREREGISTRATION.md G32 and G33).
#
# Usage: ./chain.sh dev    wait for ../marl_replay_20261023/DONE, then for the machine checks; then job Dp (Phase D')
#        ./chain.sh M      jobs S1 to S4, Q1 to Q4 and A of Plan M, in this order, each through run_ugap.sh
#
# Before each job, the chain waits until ./run_ugap.sh --check-only passes (a check every 60 s, at most 6 h). The
# check covers every Python process of the project tree, the MARL job included. A job that does not end with exit
# code 0 ends the chain; the chain does not start it again. A job that ended with 0 gets the file logs/<job>.ok.
set -u
cd "$(dirname "$0")"
mkdir -p logs
CLOG=logs/chain.log
MARL=../marl_replay_20261023
say() { echo "[chain $(date '+%F %T')] $*" | tee -a "$CLOG"; }
wait_machine() {
  local t0 n=0
  t0=$(date +%s)
  until ./run_ugap.sh --check-only > logs/check_last.txt 2>&1; do
    if (( n % 30 == 0 )); then say "wait for the machine checks: $(grep ABORT logs/check_last.txt | tr '\n' ' ')"; fi
    n=$(( n + 1 ))
    if (( $(date +%s) - t0 > 21600 )); then say "STOP: the machine checks did not pass in 6 h"; return 1; fi
    sleep 60
  done
  say "machine checks pass"
}
case ${1:-} in
  dev)
    say "wait for $MARL/DONE"
    until [[ -f $MARL/DONE ]]; do sleep 60; done
    say "$MARL/DONE exists"
    wait_machine || exit 1
    say "start Dp: dev --stages gate"
    ./run_ugap.sh Dp 45 dev --stages gate
    RC=$?
    say "Dp exit=$RC"
    [[ $RC -eq 0 ]] && touch logs/Dp.ok
    exit $RC
    ;;
  M)
    for spec in "S1|sender --pool V" "S2|sender --pool S0" "S3|sender --pool S1 --chunks 0:8" \
                "S4|sender --pool S1 --chunks 8:16" "Q1|seeds --t0c --seeds 20261101" "Q2|seeds --seeds 20261102" \
                "Q3|seeds --seeds 20261103" "Q4|seeds --seeds 20261104,20261105,20261106" "A|analysis"; do
      JOB=${spec%%|*}
      ARGS=${spec#*|}
      if [[ -f logs/$JOB.ok ]]; then say "$JOB: ended with 0 before; skip"; continue; fi
      wait_machine || exit 1
      say "start $JOB: confirm --stages $ARGS"
      # shellcheck disable=SC2086
      ./run_ugap.sh "$JOB" 120 confirm --stages $ARGS
      RC=$?
      say "$JOB exit=$RC"
      [[ $RC -eq 0 ]] || exit $RC
      touch "logs/$JOB.ok"
    done
    say "Plan M chain complete"
    ;;
  *)
    echo "usage: $0 dev | M"; exit 2
    ;;
esac
