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
MIN_LOWER95="${MIN_LOWER95:-0.80}"

shopt -s nullglob
FILES=("$HERE"/e1m3_*_"$JOB"_telemetry.jsonl)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "no M3 telemetry for job=$JOB under $HERE"; exit 1
fi

echo "=== M3 PIN GATE (rate-2 only, trace-force OFF) job=$JOB, min_lower95=$MIN_LOWER95 ==="
echo "one telemetry file = one rate-2 probe, so the whole-file gate is the operating-point gate"
NPASS=0; NFAIL=0
for T in "${FILES[@]}"; do
  BASE=$(basename "$T" "_telemetry.jsonl")          # e1m3_<arm>_<cell>_<block>_<job>
  CELL=$(echo "$BASE" | sed -E 's/^e1m3_[^_]+_(d[0-9]+)_.*/\1/')
  DSM="${CELL#d}"
  OUT=$(python3 "$HERE/e1_pin_check.py" "$T" "$DSM" "$MIN_LOWER95" 2>&1)
  LINE=$(echo "$OUT" | grep "E1_PIN_GATE" || echo "E1_PIN_GATE (no output) $OUT")
  HIST=$(echo "$OUT" | grep "E1_REALIZED_hist" | sed 's/.*): //')
  if echo "$LINE" | grep -q PASS; then NPASS=$((NPASS+1)); else NFAIL=$((NFAIL+1)); fi
  printf '  %-34s %s\n' "$BASE" "$(echo "$LINE" | sed 's/^E1_PIN_GATE //')"
  printf '  %-34s   realized: %s\n' "" "$HIST"
done
echo "--- $NPASS PASS / $NFAIL FAIL ---"
if [ "$NFAIL" -gt 0 ]; then
  echo "M3PIN_ESCALATE: $NFAIL cell-blocks did not hold their target partition at the"
  echo "  operating rate. Those cells are VOIDED -- do not read the verdict for any arm"
  echo "  whose d16 or d54 is among them (DESIGN.md sec 4.3.8(e))."
  exit 2
fi
echo "all cells held their target partition at rate 2; the verdict may be read."
