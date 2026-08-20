#!/bin/bash
# G18 rate-axis interior probe feeder (2026-08-20).
# Same QOS reality as the gate #13 feeder (4 submitted / 2 running per user), so
# the four probe jobs are drip-fed.  Cap is the SAME global 4; both feeders back
# off on refusal, so they interleave work-conservingly.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="$HERE/g16_grid.sbatch"
STATE="$HERE/G18_PROBE_STATE.tsv"
STOP="$HERE/G18_PROBE_STOP"
MAX_TOTAL=4
POLL_S="${POLL_S:-300}"

[ -f "$STATE" ] || printf 'idx\ttag\trlo\trhi\tjobid\tsubmitted_utc\n' > "$STATE"
# rev2 (claims-auditor 2026-08-20): the arm grid is back to all seven, and the
# Stage-1 ladder is the DENSE TOP of the interval (4.25, 4.75 against a 5.084
# minimum-arm capacity), not four equally spaced points.  One boot covers both
# rates via the harness's own two-phase round, so n=2 blocks is 2 jobs.
TAGS=(r425x475rep1 r425x475rep2)
RLOS=(4.25         4.25)
RHIS=(4.75         4.75)
TOTAL=${#TAGS[@]}

n_done()     { awk 'NR>1' "$STATE" | wc -l; }
n_in_queue() { squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l; }

echo "[g18probe] start $(date -u +%FT%TZ) total=$TOTAL already=$(n_done)"
while :; do
  [ -f "$STOP" ] && { echo "[g18probe] STOP file -- exiting"; break; }
  d=$(n_done); [ "$d" -ge "$TOTAL" ] && { echo "[g18probe] all submitted"; break; }
  slots=$(( MAX_TOTAL - $(n_in_queue) ))
  while [ "$slots" -gt 0 ] && [ "$d" -lt "$TOTAL" ]; do
    t=${TAGS[$d]}; lo=${RLOS[$d]}; hi=${RHIS[$d]}
    # counterbalance the visiting order across replicates exactly as G16 does
    ao=$(python3 "$HERE/g16_arm_order.py" $(( d + 1 )))
    out=$(sbatch --parsable --job-name="g18p-${t}" \
          --export=ALL,G18_PROBE=1,G16_RLO="$lo",G16_RHI="$hi",G18_ARMS="$ao" \
          "$SB" "$t" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$t" "$lo" "$hi" "$out" \
             "$(date -u +%FT%TZ)" >> "$STATE"
      echo "[g18probe] $(date -u +%FT%TZ) submitted $t rates=($lo,$hi) -> $out"
      d=$((d+1)); slots=$((slots-1))
    else
      echo "[g18probe] submit refused ($out) -- backing off"; break
    fi
  done
  sleep "$POLL_S"
done
echo "[g18probe] end $(date -u +%FT%TZ) submitted=$(n_done)/$TOTAL"
