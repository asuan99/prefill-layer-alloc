#!/bin/bash
# G16 block 1 (job 884336) durable SLURM polling loop.
#
# Runs detached from any interactive/agent session (launched via setsid+nohup
# by experiment-runner, 2026-08-16) so it survives session expiry -- the
# prior session's poll log lived under /tmp/.../scratchpad/ and was lost when
# that session ended. This one lives in the results directory instead.
#
# Appends one `sacct` snapshot line every POLL_INTERVAL_S seconds to
# $LOGFILE, and stops on its own once the job reaches a terminal state
# (COMPLETED/FAILED/TIMEOUT/CANCELLED/OUT_OF_MEMORY/NODE_FAIL/...).
#
# NOT committed to git (matches the .out/.err convention for this project --
# see task instructions / CLAUDE.md "루트의 .out/.err는 커밋하지 않는다").
#
# Usage: setsid nohup bash g16_blk1_poll.sh <JOBID> [POLL_INTERVAL_S] > /dev/null 2>&1 &
set -uo pipefail

JOB="${1:?usage: g16_blk1_poll.sh <JOBID> [POLL_INTERVAL_S]}"
INTERVAL="${2:-300}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$HERE/g16_blk1_${JOB}_poll.log"

TERMINAL_STATES="COMPLETED FAILED TIMEOUT CANCELLED OUT_OF_MEMORY NODE_FAIL PREEMPTED BOOT_FAIL DEADLINE REVOKED"

is_terminal() {
  local state="$1"
  for t in $TERMINAL_STATES; do
    # sacct sometimes appends " by <uid>" to CANCELLED -- prefix match.
    case "$state" in
      "$t"*) return 0 ;;
    esac
  done
  return 1
}

{
  echo "POLL_START job=$JOB interval_s=$INTERVAL pid=$$ started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOGFILE"

while true; do
  LINE="$(sacct -j "$JOB" --format=JobID,State,Elapsed,ExitCode,NodeList -P --noheader 2>&1 | head -1)"
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ -z "$LINE" ]; then
    echo "$TS job=$JOB SACCT_EMPTY_ROW (job not visible to sacct yet)" >> "$LOGFILE"
    STATE=""
  else
    echo "$TS job=$JOB $LINE" >> "$LOGFILE"
    STATE="$(echo "$LINE" | awk -F'|' '{print $2}')"
  fi

  if [ -n "$STATE" ] && is_terminal "$STATE"; then
    echo "$TS job=$JOB POLL_STOP reason=terminal_state state=$STATE" >> "$LOGFILE"
    break
  fi

  sleep "$INTERVAL"
done

echo "POLL_END job=$JOB ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOGFILE"
