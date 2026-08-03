#!/bin/bash
# M3 pin verification (DESIGN.md sec 4.3.8(e), REVISED 2026-08-02).
#
# ★WHY THIS IS NOT A GPU JOB.  sec 4.3.8(e) originally required "a separate
# short trace-force-ON run" because job 867298's PIN_CHECK died on an argument
# order bug and left no pin data.  That requirement was wrong on its own terms:
#
#   - The time-weighted pin gate does NOT need PDMUX_TRACE_FORCE_PREFILL.
#     e1_capacity_scan.sbatch never sets it (engine default "0",
#     multiplex/multiplexing_mixin.py:104) and still produced passing gates on
#     every cell M3 uses -- T8 d16/d24/d44 pin_frac 0.950-0.985, Ha8
#     d16/d24/d44/d54 0.994-1.000, n_episodes 31-144.  Trace-force only adds
#     prefill-active samples; it is not a precondition.
#   - Forcing it would inject the very observer effect sec 4.7.1 exists to keep
#     out of an ITL measurement (+2.0% [+0.78, +3.21] on d92's itl_p95).
#   - The capacity-scan gates are aggregated over ALL probe rates, so they do
#     not by themselves certify the operating point: CONSENSUS sec 1-22 shows
#     green-context reverts to no-split when decode is empty, and that is
#     MORE likely at low rate.  What is needed is a rate-2-only gate.
#
# M3's telemetry is exactly that.  Each e1m3_<arm>_<cell>_<block> file holds a
# single rate-2 probe, so running the gate over the whole file IS the
# rate-restricted check -- no window arguments, no reconstruction, no GPU.
#
# Cite-blocking: a cell failing here is VOIDED, exactly like the in-run gates.
# Run BEFORE reading m3_analyze.py's verdict.
#
# Usage:
#   source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate   # needs scipy
#   ./m3_pin_check.sh <job_id>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="${1:?usage: m3_pin_check.sh <job_id>}"
MIN_LOWER95="${MIN_LOWER95:-0.80}"   # legacy gate (identity; kept for continuity)
COND_PIN_MIN="${COND_PIN_MIN:-0.99}"  # ★pre-registered 2026-08-03 for the
                                      # conditional gate. Binding on jobs after
                                      # this revision; applying it to 872077 is
                                      # a POST-HOC re-score and is labelled so.

shopt -s nullglob
FILES=("$HERE"/e1m3_*_"$JOB"_telemetry.jsonl)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "no M3 telemetry for job=$JOB under $HERE"; exit 1
fi

echo "=== M3 PIN GATES (rate-2 only, trace-force OFF) job=$JOB ==="
echo "★2026-08-03: THREE numbers per cell now. The legacy gate is kept because"
echo "  every prior report was scored on it, but it is an IDENTITY -- verified"
echo "  over 120 files / 77,688 prefill-active snapshots, ZERO violations of"
echo "     prefill_sms != target  <=>  decode_running_batch_size == 0"
echo "  -- so it scores the DESIGNED decode-empty auto-revert"
echo "  (multiplexing_mixin.py:773,792-794; CONSENSUS 1-22) as a failure."
echo "  COND_PIN is the gate that asks the pin question (population restricted"
echo "  to prefill-active AND decode-busy). DECODE_REALIZED is the axis that"
echo "  was never gated at all. Re-scoring 872077 with COND_PIN is POST HOC"
echo "  and flagged as such; COND_PIN_MIN=$COND_PIN_MIN binds forward."
NPASS=0; NFAIL=0
for T in "${FILES[@]}"; do
  BASE=$(basename "$T" "_telemetry.jsonl")
  CELL=$(echo "$BASE" | sed -E 's/^e1m3_[^_]+_(d[0-9]+)_.*/\1/')
  DSM="${CELL#d}"
  OUT=$(python3 "$HERE/e1_pin_check.py" "$T" "$DSM" "$MIN_LOWER95" 2>&1)
  LEG=$(echo "$OUT" | grep "E1_PIN_GATE"        | sed 's/^E1_PIN_GATE //')
  CND=$(echo "$OUT" | grep "E1_COND_PIN"        | sed 's/.*cond_pin_frac=//')
  DEC=$(echo "$OUT" | grep "E1_DECODE_REALIZED" | sed 's/.*decode-active time = //')
  CVAL=$(echo "$CND" | awk '{print $1}')
  OK=$(python3 -c "import sys; v='$CVAL'
try: print('PASS' if float(v) >= $COND_PIN_MIN else 'FAIL')
except Exception: print('NODATA')")
  [ "$OK" = "PASS" ] && NPASS=$((NPASS+1)) || NFAIL=$((NFAIL+1))
  printf '  %-30s COND_PIN=%s -> %s\n' "$BASE" "$CVAL" "$OK"
  printf '  %-30s   legacy: %s\n' "" "$LEG"
  printf '  %-30s   decode_realized: %s\n' "" "$DEC"
done
echo "--- COND_PIN: $NPASS PASS / $NFAIL FAIL (threshold $COND_PIN_MIN) ---"
if [ "$NFAIL" -gt 0 ]; then
  echo "M3PIN_ESCALATE: $NFAIL cell-blocks failed the CONDITIONAL pin gate."
  exit 2
fi
echo "all cells held their target partition whenever the question was well posed."
