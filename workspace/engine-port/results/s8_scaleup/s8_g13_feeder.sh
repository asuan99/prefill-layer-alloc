#!/bin/bash
# gate #13 campaign drip-feeder.
#
# The partition's QOS (aanv8) allows 4 SUBMITTED and 2 RUNNING jobs per user, so
# the registered plan's 32 jobs cannot be queued at once.  This keeps the queue
# topped up and stops on a failure rate that mirrors the harness's own
# pre-registered stop rule (s8_c2r.sbatch, PREREG rev2 SS5: > 25%).
#
# Registered plan (DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md sec2.2.1):
#     M8   16 jobs x BOOTS=3     Ha8  16 jobs x BOOTS=6
# Work items are INTERLEAVED so that an interruption leaves both arms with a
# comparable number of jobs rather than one arm complete and the other empty.
#
# State lives in G13_FEEDER_STATE.tsv, so a restart resumes instead of
# double-submitting.  Run detached:
#     setsid nohup ./s8_g13_feeder.sh > g13_feeder.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$HERE/s8_c2r.sbatch"
STATE="$HERE/G13_FEEDER_STATE.tsv"
STOP="$HERE/G13_FEEDER_STOP"          # touch this file to halt cleanly
GATE="$HERE/G13_GATE26_OK"            # written by gate26_ok() once plumbing is proven
MAX_SUBMITTED=4
POLL_S="${POLL_S:-300}"
K_PER_ARM="${K_PER_ARM:-16}"

[ -f "$STATE" ] || printf 'idx\tarm\tarm_ids\tboots\tjobid\tsubmitted_utc\n' > "$STATE"

# Work list: interleaved M8/Ha8, K_PER_ARM each.
declare -a W_ARM W_IDS W_BOOTS
for ((i = 0; i < K_PER_ARM; i++)); do
  W_ARM+=(M8);  W_IDS+=(0); W_BOOTS+=(3)
  W_ARM+=(Ha8); W_IDS+=(1); W_BOOTS+=(6)
done
TOTAL=${#W_ARM[@]}

n_done()      { awk 'NR>1' "$STATE" | wc -l; }
n_in_queue()  { squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l; }

# gate #26 (plumbing smoke before a large campaign).  Until the FIRST work item
# has (a) finished COMPLETED, (b) echoed the BOOTS this feeder asked for, and
# (c) reached SWEEP_DONE, only ONE job is kept in flight.  The check is
# self-verifying: nobody has to be awake to open the gate, and a wrong env
# passthrough costs one job instead of four.
gate26_ok() {
  [ -f "$GATE" ] && return 0
  local jid boots st out
  jid=$(awk 'NR==2{print $5}' "$STATE"); boots=$(awk 'NR==2{print $4}' "$STATE")
  [ -z "$jid" ] && return 1
  st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
  [ "$st" = COMPLETED ] || return 1
  out=$(ls "$HERE"/s8g13-*_"$jid".out 2>/dev/null | head -1)
  [ -n "$out" ] || out=$(ls "$HERE"/*_"$jid".out 2>/dev/null | head -1)
  [ -n "$out" ] || return 1
  grep -q "BOOTS=$boots " "$out" || {
      echo "[feeder] GATE26 FAIL: $out does not echo BOOTS=$boots -- env passthrough broken"
      touch "$STOP"; return 1; }
  grep -q "SWEEP_DONE" "$out" || return 1
  echo "[feeder] GATE26 PASS via job $jid ($out)"; touch "$GATE"; return 0
}

# Failure guard: of the campaign jobs that have already finished, how many did
# not reach COMPLETED.  Mirrors the harness stop rule (>25%).
fail_rate() {
  local ids fin=0 bad=0 st
  ids=$(awk 'NR>1{print $5}' "$STATE" | paste -sd, -)
  [ -z "$ids" ] && { echo "0 0"; return; }
  while read -r jid st; do
    case "$st" in
      COMPLETED)                     fin=$((fin+1)) ;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
                                     fin=$((fin+1)); bad=$((bad+1)) ;;
    esac
  done < <(sacct -j "$ids" -X -n -o JobID,State 2>/dev/null | awk '{print $1, $2}')
  echo "$bad $fin"
}

echo "[feeder] start $(date -u +%FT%TZ)  total=$TOTAL already=$(n_done)"
while :; do
  [ -f "$STOP" ] && { echo "[feeder] STOP file present -- exiting"; break; }
  done_n=$(n_done)
  [ "$done_n" -ge "$TOTAL" ] && { echo "[feeder] all $TOTAL work items submitted"; break; }

  read -r bad fin < <(fail_rate)
  if [ "$fin" -ge 4 ]; then
    if awk -v b="$bad" -v f="$fin" 'BEGIN{exit !(b/f > 0.25)}'; then
      echo "[feeder] STOP: failure rate $bad/$fin > 25% -- halting (harness PREREG rev2 SS5)"
      break
    fi
  fi

  if gate26_ok; then CAP=$MAX_SUBMITTED; else CAP=1; fi
  inq=$(n_in_queue)
  slots=$(( CAP - inq ))
  while [ "$slots" -gt 0 ] && [ "$done_n" -lt "$TOTAL" ]; do
    arm=${W_ARM[$done_n]}; ids=${W_IDS[$done_n]}; boots=${W_BOOTS[$done_n]}
    out=$(sbatch --parsable --job-name="s8g13-${arm}" \
          --export=ALL,ARM_IDS="$ids",BOOTS="$boots" "$SB" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$done_n" "$arm" "$ids" "$boots" "$out" \
             "$(date -u +%FT%TZ)" >> "$STATE"
      echo "[feeder] $(date -u +%FT%TZ) submitted idx=$done_n $arm boots=$boots -> $out"
      done_n=$((done_n + 1)); slots=$((slots - 1))
    else
      echo "[feeder] submit refused ($out) -- backing off"
      break
    fi
  done
  sleep "$POLL_S"
done
echo "[feeder] end $(date -u +%FT%TZ)  submitted=$(n_done)/$TOTAL"
