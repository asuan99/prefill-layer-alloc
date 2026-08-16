#!/bin/bash
#SBATCH --job-name=s8c2r
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_scaleup/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_scaleup/%x_%j.err
# ============================================================================
# C2-R: M8 (Mamba-Codestral-7B) / Ha8 (Zamba2-7B) re-measurement.
# Implements PREREG_C2R_RULES_REV2_2026-08-15.md, GO'd by claims-auditor on
# 2026-08-15 with 10 caveats (C1-C10), 5 of which (C3,C4,C6,C7,C8) changed
# this harness -- see "COORDINATOR GO + CAVEATS" below. DO NOT edit this
# header's estimand/grid without re-reading that file first.
#
# WHAT THIS MEASURES (PREREG rev2 SS1, verbatim reason this campaign exists):
#   Job 865493 (the only run with a WORKING keepalive device, i.e. the prefill
#   partition was actually held resident) is missing `t0_monotonic_s` for
#   M8's all 5 cells and Ha8's d16 cell -- a client-summary field, not a
#   telemetry field, but the two must be joined via it to attribute ITL
#   samples to a realized (decode_sms, decode_bs) state. Without it those
#   cells cannot be batch-matched at all. The ONLY other run that has the
#   field (865533) had its keepalive collapse completely (F4: KEEPA_REPS=224
#   overflowed CTXCAP=1792 by 1 token, every keepalive request rejected,
#   partition never actually held resident) -- so today's M8/Ha8 C2 headline
#   ratios rest on data from a run where the very device that was supposed to
#   keep the partition split active didn't work. This campaign re-measures
#   M8 (all legs) and Ha8 (d16 leg; d92 already has a working-keepalive
#   sample) under a WORKING keepalive device, batch-matched at b=16 (the
#   value the audit confirmed clears the realized decode-active denominator
#   in 8/8 prior cells).
#
# WHAT THIS DOES NOT MEASURE / CLOSE (PREREG rev2 SS6, SS5-1 C1/C2):
#   - Does NOT validate or invalidate the existing M8/Ha8 headline ratios
#     (2.909 / 2.852). Normal keepalive has ZERO snapshots at the batch those
#     numbers were computed at (M8 b=12, Ha8 b=9) -- there is no slice to
#     compare against, so this campaign produces a REPLACEMENT number, not a
#     verdict on the old one (C2).
#   - Does NOT settle arm-vs-arm ORDERING. r_a(16) is an arm-internal ratio;
#     cross-arm order is a function of batch b and the curves cross, so a
#     single b=16 ordering is just another slice (rev2's "stop (a) is
#     permanently retired", not resolved).
#   - Does NOT bound low-realization sampling bias, touch HE0/tension-A/
#     Gate-2, or claim ctx=4096 / other (B,L) portability.
#   - Backend: both arms run triton, so this campaign is not exposed to the
#     Hs8/flashinfer confound -- but if D2 (4-arm range) is ever cited, Hs8
#     (flashinfer) must be flagged alongside it.
#
# GRID (PREREG rev2 SS3, do not change without a new pre-registration):
#   ctx=1024 (CTXCAP=1792) x arm{M8,Ha8} x decode_sm{16,92} (prefill pinned
#   16 SM throughout -- pdmux_p16_d{16,92}.yml) x 6 independent boots/cell
#   (24 boots total) x batch b=16 (conc=16, closed-loop).
#
# COORDINATOR GO + CAVEATS (2026-08-15, addressed in this harness):
#   C3 (-> H7 below): assert t0_monotonic_s + chunk_times_s + a
#     phase=="benchmark" telemetry record with matching run_id/workload_id
#     exist right after each boot's measurement; abort that CELL (not just
#     that boot) if not, rather than repeating a diagnosed-broken cell
#     6 times over.
#   C4: boots/cell 4 -> 6 (24 total, not 16) -- "boot SD ~=0.3%" (rev2's
#     original justification for n=4) was itself found unmeasurable
#     (872920/872921 have no d92 leg and a collapsed keepalive), so boot
#     variance is UNKNOWN and 6 buys slack against the D1 threshold of >=4
#     adopted boots, not a resolved-variance number.
#   C6 (the big one): keepalive is NOT re-derived from H2's headroom. It
#     replicates 865493's ACTUAL setting: KEEPA_N=8, KEEPA_REPS=32 (~258
#     tokens for both M8 and Ha8's tokenizers, confirmed offline
#     2026-08-15 -- see KEEPALIVE_TOKEN_CHECK_OK banner at boot). Do not
#     "improve" this value -- a longer prompt changes prefill co-residency
#     duty cycle and decode_bs occupancy, i.e. a DIFFERENT regime than the
#     one being re-measured. H2's assert threshold (<=1536 tokens) is a
#     REGRESSION GUARD only (catches a future F4-style overflow), not the
#     operating point.
#   C7: Ha8's d16 leg sits on a guard cliff (same-state adjacent telemetry
#     snapshot gap p50=2.72s, max=2.97s, vs GUARD=3.0s in the analysis
#     script) -- zero qualifying intervals for that cell is a possible,
#     legitimate MEASUREMENT OUTCOME, not a harness bug (methodology gate
#     #21). H7's check therefore reports the snapshot-gap distribution
#     (diagnostic only, does not gate) alongside its pass/fail.
#   C8: the two legs (d16, d92) are boot-interleaved in BLOCKS that
#     alternate visiting order by block parity (BLOCK 1: d16,d92; BLOCK 2:
#     d92,d16; ...), so cell identity is not confounded with wall-clock /
#     boot-order drift across the run. This mirrors s8_sweep_d54.sbatch's
#     BLOCK design, which is also where this script borrows its run_cell()
#     structure from.
#   (C1, C2, C5, C10 are result-interpretation caveats for whoever scores
#   this campaign's output, not harness requirements; not implemented here.)
#
# HARNESS REQUIREMENTS FROM PREREG rev2 SS4 (H1-H6) + coordinator's H7:
#   H1 - boot is the independent unit, not an outer wrapper around REPS.
#        Implemented: run_cell() boots fresh per BLOCK visit to a cell,
#        ports/telemetry/logs are all boot-scoped ("_blk<N>" in every
#        filename).
#   H2 - keepalive prompt is tokenized with THAT ARM'S actual tokenizer and
#        asserted <= CTXCAP-256 tokens BEFORE any server boots for that arm;
#        violation is a hard `exit` (not a skip), because CTX/KEEPA_REPS are
#        shared across arms in one job invocation and a violation likely
#        indicates a shared misconfiguration. Prompt file name embeds the
#        arm (tokenizers differ across arms in this project, even though M8
#        and Ha8 happen to agree numerically).
#   H3 - see s8_c2r_client.py (a COPY of s0dc_client.py, not an edit of it).
#        keepalive_worker there body-parses the response instead of trusting
#        HTTP status alone.
#   H4 - KEEPA_REPS/KEEPA_N are fixed defaults (32/8, see C6), overridable
#        via env for debugging, and printed into every arm's result-file
#        header.
#   H5 - realized_pin_check.py output is recorded every boot but never used
#        to drop a cell (its own exit code is always 0 by design).
#   H6 - see the #SBATCH block above; --comment is inside it, above the
#        first executable line.
#   H7 - see run_cell(): the artifact-integrity check described under C3.
#
# STOP RULE (PREREG rev2 SS5, arm-scoped): if boot failure rate exceeds 25%
# (evaluated once >=4 boots have been attempted for that arm), remaining
# BLOCKS for that arm are skipped and partial results are reported. A cell
# that fails H7 is disabled for the rest of that arm's blocks specifically
# (the sibling cell and other blocks continue) -- see CELL_DISABLED below.
#
# PIPELINE SMOKE (methodology gate #26 -- required before the real campaign,
# not a performance run): SMOKE=1 narrows the grid to 1 arm (M8) x 1 leg
# (d16) x 1 boot x measure=10s, to confirm H2's assert and H3's body-parsing
# actually fire (they have never fired in this codebase before).
# SMOKE_FORCE_BAD_KEEPALIVE=1 additionally forces KEEPA_REPS=224 (the exact
# F4 value, 1793-1794 tokens > the 1536 threshold for both arms) so the H2
# abort path can be exercised deliberately.
#
# Usage:
#   Real campaign, both arms sequentially in one job:
#     sbatch s8_c2r.sbatch
#   Real campaign, split by arm (QOS aanv8: 4 submitted / 2 running):
#     sbatch --export=ALL,ARM_IDS="0" s8_c2r.sbatch   # M8
#     sbatch --export=ALL,ARM_IDS="1" s8_c2r.sbatch   # Ha8
#   Pipeline smoke (gate #26):
#     sbatch --export=ALL,SMOKE=1 s8_c2r.sbatch
#     sbatch --export=ALL,SMOKE=1,SMOKE_FORCE_BAD_KEEPALIVE=1 s8_c2r.sbatch
#
# NOT a claim job by itself. claims-auditor GO'd the DESIGN in PREREG rev2;
# scoring the output still needs the positive-control gate (C5) before any
# number from this campaign is cited.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_scaleup
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_c2r_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest_c2r.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

# ---------------------------------------------------------------------------
# Grid / knobs (PREREG rev2 SS3; SMOKE overrides below narrow this for the
# pipeline smoke, they do not change any of these defaults).
# ---------------------------------------------------------------------------
ARM_LABELS=(M8 Ha8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
)
ARM_BACKENDS=(triton triton)   # both triton -- PREREG rev2 SS3, no backend axis here

SMOKE="${SMOKE:-0}"
SMOKE_FORCE_BAD_KEEPALIVE="${SMOKE_FORCE_BAD_KEEPALIVE:-0}"

CTX="${CTX:-1024}"
CONC="${CONC:-16}"                 # b=16, PREREG rev2 SS3
OUTTOK="${OUTTOK:-512}"
BOOTS="${BOOTS:-6}"                 # independent boots per (arm,leg) cell -- C4
REPS_PER_BOOT="${REPS_PER_BOOT:-1}" # H1 pseudocode: "run client once" per boot
KEEPA_N="${KEEPA_N:-8}"             # C6: replicate 865493, not a fresh guess
PIN_GATE="${PIN_GATE:-0.80}"        # H5: reported only, never gates a cell

if [ "$SMOKE" = 1 ]; then
  WARMUP_S="${WARMUP_S:-5}"
  MEASURE_S="${MEASURE_S:-10}"
else
  WARMUP_S="${WARMUP_S:-20}"
  MEASURE_S="${MEASURE_S:-60}"
fi

CTXCAP=$(( CTX + OUTTOK + 256 ))
KEEPA_TOKEN_LIMIT=$(( CTXCAP - 256 ))   # H2 regression-guard threshold

# C6: KEEPA_REPS=32 replicates 865493's ACTUAL keepalive prompt (confirmed
# via that job's *_result.txt header: "keepN=8(short prompt)"; the specific
# prompt file 865493 used is gone from disk, but its generation is
# deterministic from KEEPA_REPS so regenerating with the same rep count
# reproduces it byte-for-byte). SMOKE_FORCE_BAD_KEEPALIVE deliberately picks
# the historical F4 value instead, to exercise the H2 abort path.
KEEPA_REPS_C6=32          # the ONE value that satisfies C6; used to police the label
if [ "$SMOKE" = 1 ] && [ "$SMOKE_FORCE_BAD_KEEPALIVE" = 1 ]; then
  KEEPA_REPS="${KEEPA_REPS:-224}"
else
  KEEPA_REPS="${KEEPA_REPS:-$KEEPA_REPS_C6}"
fi

if [ "$SMOKE" = 1 ]; then
  ARM_IDS="0"       # M8 only
  BOOTS=1
  REPS_PER_BOOT=1
fi
ARM_IDS="${ARM_IDS:-0 1}"

echo "SMOKE=$SMOKE SMOKE_FORCE_BAD_KEEPALIVE=$SMOKE_FORCE_BAD_KEEPALIVE" \
     "CTX=$CTX CTXCAP=$CTXCAP KEEPA_TOKEN_LIMIT=$KEEPA_TOKEN_LIMIT" \
     "CONC=$CONC OUTTOK=$OUTTOK WARMUP_S=$WARMUP_S MEASURE_S=$MEASURE_S" \
     "BOOTS=$BOOTS REPS_PER_BOOT=$REPS_PER_BOOT KEEPA_N=$KEEPA_N KEEPA_REPS=$KEEPA_REPS" \
     "ARM_IDS=$ARM_IDS PIN_GATE=$PIN_GATE"

# Main-workload prompt: unchanged generation rule from every prior s8_scaleup
# campaign, shared across arms (only the keepalive prompt is arm-specific).
PREPS=$(( CTX / 8 ))
PF="$HERE/s8_C${CTX}_prompt.txt"
[ -f "$PF" ] || python3 - "$PREPS" "$PF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

JOBTAG="${SLURM_JOB_ID:-local}"
declare -A CELL_CFG=( [d16]="$HERE/pdmux_p16_d16.yml:16" [d92]="$HERE/pdmux_p16_d92.yml:92" )
declare -A CELL_DISABLED

# ---------------------------------------------------------------------------
# run_cell: ONE independent boot of ONE cell (H1). Returns:
#   0 = boot + measurement + H7 artifact check all passed
#   1 = boot failed OR the client measurement failed (counts toward the
#       arm's boot-failure-rate stop rule; the cell stays enabled -- a boot
#       flake does not indict the cell)
#   2 = H7 artifact-integrity check failed (counts toward the stop rule AND
#       the caller disables this cell for the rest of the arm -- C3)
# ---------------------------------------------------------------------------
run_cell () {
  local ARM="$1" MODEL="$2" BACKEND="$3" NAME="$4" BLK="$5" RESULT="$6"
  local cfg="${CELL_CFG[$NAME]%%:*}" DSM="${CELL_CFG[$NAME]##*:}"
  local PORT
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  local RUNID="s8c2r_${ARM}_C${CTX}_${NAME}_${JOBTAG}_blk${BLK}"
  local SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  local WORKLOAD_ID="c2r_${ARM}_C${CTX}_${NAME}_blk${BLK}"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="$WORKLOAD_ID"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  TOTAL_BOOTS_ATTEMPTED=$(( TOTAL_BOOTS_ATTEMPTED + 1 ))
  echo "=== ARM=$ARM block=$BLK/$BOOTS cell=$NAME decode_sm=$DSM cfg=$(basename "$cfg") port=$PORT ===" \
    | tee -a "$RESULT"

  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 360); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] \
      && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED block=$BLK cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok block=$BLK cell=$NAME" | tee -a "$RESULT"

  if [ "$ok" != 1 ]; then
    TOTAL_BOOTS_FAILED=$(( TOTAL_BOOTS_FAILED + 1 ))
    echo "BOOT_FAILED block=$BLK cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
    kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
    return 1
  fi

  # SM clock sampler spans the whole boot's measurement (all REPS_PER_BOOT
  # reps). Idle-SM boost headroom differs by how many SMs are active, biasing
  # toward UNDER-estimating decode-SM sensitivity if unrecorded (inherited
  # rationale from s8_sweep.sbatch / s0_deconfound/DESIGN.md SS3). Reported
  # per cell (C7).
  local CLK="$HERE/${RUNID}_clocks.csv"
  local CLK_ITERS=$(( REPS_PER_BOOT * ((WARMUP_S + MEASURE_S) / 5 + 2) ))
  ( for _ in $(seq 1 $CLK_ITERS); do
      nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu \
        --format=csv,noheader,nounits >> "$CLK" 2>/dev/null; sleep 5
    done ) &
  local CLKPID=$!

  local RC=0
  for r in $(seq 1 $REPS_PER_BOOT); do
    local RAW="$HERE/${RUNID}_rep${r}_raw.jsonl"
    local SUMMARY="$HERE/${RUNID}_rep${r}_summary.json"
    python3 "$HERE/s8_c2r_client.py" --url "http://127.0.0.1:$PORT" \
      --prompt-file "$PF" --conc $CONC --out-tokens $OUTTOK \
      --warmup-s $WARMUP_S --measure-s $MEASURE_S \
      --keepalive-prefill $KEEPA_N --keepalive-prompt-file "$KPF" \
      --raw-out "$RAW" --tag "c2r_${ARM}_C${CTX}_${NAME}_blk${BLK}_r${r}" \
      2>>"$RESULT" > "$SUMMARY"
    local CLIENT_RC=$?
    cat "$SUMMARY" | tee -a "$RESULT"
    if [ "$CLIENT_RC" != 0 ]; then
      echo "CLIENT_FAILED block=$BLK cell=$NAME rep=$r rc=$CLIENT_RC" | tee -a "$RESULT"
      [ "$RC" = 2 ] || RC=1
      continue
    fi

    # H7 (coordinator caveat C3): the exact defect that made 865493's
    # M8/Ha8-d16 cells unusable was silent -- the client ran fine, only the
    # downstream join failed months later. Check it now, per boot, instead.
    python3 - "$SUMMARY" "$RAW" "$PDMUX_TELEMETRY_PATH" "$RUNID" "$WORKLOAD_ID" <<'PY' | tee -a "$RESULT"
import json, sys

summary_path, raw_path, telem_path, run_id, workload_id = sys.argv[1:6]
ok, reasons = True, []

try:
    lines = [l for l in open(summary_path) if l.strip()]
    summary = json.loads(lines[-1]) if lines else {}
except Exception as exc:  # noqa: BLE001
    ok, summary = False, {}
    reasons.append(f"summary unparseable: {exc}")
if summary.get("t0_monotonic_s") is None:
    ok = False
    reasons.append("t0_monotonic_s missing/null in client summary")

raw_ok = False
try:
    for line in open(raw_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if len(rec.get("chunk_times_s") or []) >= 2:
            raw_ok = True
            break
except Exception as exc:  # noqa: BLE001
    reasons.append(f"raw_out unreadable: {exc}")
if not raw_ok:
    ok = False
    reasons.append("no raw_out record with chunk_times_s >= 2 samples")

bench_n = mism_run = mism_wl = 0
gaps, last_ts = [], None
try:
    for line in open(telem_path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        bench_n += 1
        if e.get("run_id") != run_id:
            mism_run += 1
        if e.get("workload_id") != workload_id:
            mism_wl += 1
        ts = e.get("timestamp_monotonic_s")
        if ts is not None:
            if last_ts is not None:
                gaps.append(ts - last_ts)
            last_ts = ts
except Exception as exc:  # noqa: BLE001
    reasons.append(f"telemetry unreadable: {exc}")

if bench_n == 0:
    ok = False
    reasons.append("no phase=benchmark runtime_snapshot records in telemetry")
if bench_n and mism_run:
    ok = False
    reasons.append(f"{mism_run}/{bench_n} benchmark records have run_id != {run_id}")
if bench_n and mism_wl:
    ok = False
    reasons.append(f"{mism_wl}/{bench_n} benchmark records have workload_id != {workload_id}")


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


gaps_sorted = sorted(gaps)
gmax = max(gaps) if gaps else float("nan")
# C7: report-only diagnostic (does NOT gate). A GUARD=3.0s cliff means a
# cell can legitimately produce zero qualifying intervals at the analysis
# stage; that is a measurement outcome, not a harness failure (gate #21).
print(f"H7_CHECK run_id={run_id} bench_snapshots={bench_n} n_gaps={len(gaps_sorted)} "
      f"snapshot_gap_s_p50={pct(gaps_sorted, 0.5):.3f} p90={pct(gaps_sorted, 0.9):.3f} "
      f"max={gmax:.3f} (GUARD=3.0s cliff diagnostic per C7)")
if not ok:
    print("H7_FAILED " + "; ".join(reasons))
    sys.exit(1)
print("H7_OK")
PY
    local H7_RC=${PIPESTATUS[0]}
    if [ "$H7_RC" != 0 ]; then
      echo "H7_ABORT_CELL arm=$ARM block=$BLK cell=$NAME rep=$r: artifact integrity check" \
           "failed -- disabling cell=$NAME for the rest of this arm's blocks" | tee -a "$RESULT"
      RC=2
    fi
  done

  kill $CLKPID 2>/dev/null; wait $CLKPID 2>/dev/null
  python3 - "$CLK" <<'PY' | tee -a "$RESULT"
import statistics, sys
try:
    rows = [l.split(",") for l in open(sys.argv[1]) if l.strip()]
    sm = [float(r[0]) for r in rows]
    print(f"CLOCKS sm_mhz_mean={statistics.fmean(sm):.0f} sm_mhz_min={min(sm):.0f} n={len(sm)}")
except Exception as exc:
    print(f"CLOCKS unavailable: {exc}")
PY

  # H5 (audit target 9): report + sensitivity axis ONLY. Its own exit status
  # is always 0 by construction (realized_pin_check.py's docstring) and is
  # never used here to drop a cell -- Ha8 would be eliminated wholesale if it
  # were (rev2's own rationale for demoting PIN 0.80 from a gate).
  python3 "$HERE/realized_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "$PIN_GATE" | tee -a "$RESULT"

  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5

  if [ "$RC" = 2 ]; then
    TOTAL_BOOTS_FAILED=$(( TOTAL_BOOTS_FAILED + 1 ))
    return 2
  fi
  if [ "$RC" = 1 ]; then
    TOTAL_BOOTS_FAILED=$(( TOTAL_BOOTS_FAILED + 1 ))
    return 1
  fi
  return 0
}

for AIDX in $ARM_IDS; do
  ARM="${ARM_LABELS[$AIDX]}"
  MODEL="${ARM_MODELS[$AIDX]}"
  BACKEND="${ARM_BACKENDS[$AIDX]}"
  RESULT="$HERE/s8c2r_${ARM}_C${CTX}_${JOBTAG}_result.txt"; : > "$RESULT"
  # The C6 provenance note describes KEEPA_REPS=32 (job 865493's actual setting).
  # SMOKE_FORCE_BAD_KEEPALIVE deliberately substitutes the corrupt historical value
  # 224, so the note must NOT travel with that value -- a label that says
  # "replicates 865493" next to 224 is exactly the class of lying label this
  # project has been burned by. Emit the note only when it is true.
  if [ "$KEEPA_REPS" = "$KEEPA_REPS_C6" ]; then
    KEEPA_NOTE="(C6: replicates job 865493's actual keepalive setting)"
  else
    KEEPA_NOTE="(C6 VIOLATED: KEEPA_REPS=$KEEPA_REPS is NOT 865493's setting ($KEEPA_REPS_C6) -- not a valid C2-R measurement)"
  fi
  echo "C2R-rev2 ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CTX=$CTX CTXCAP=$CTXCAP" \
       "conc=$CONC out=$OUTTOK warmup=${WARMUP_S}s measure=${MEASURE_S}s" \
       "BOOTS=$BOOTS REPS_PER_BOOT=$REPS_PER_BOOT KEEPA_N=$KEEPA_N KEEPA_REPS=$KEEPA_REPS" \
       "$KEEPA_NOTE" \
       "SMOKE=$SMOKE SMOKE_FORCE_BAD_KEEPALIVE=$SMOKE_FORCE_BAD_KEEPALIVE" | tee -a "$RESULT"

  # ---- H2: arm-specific keepalive prompt + pre-boot tokenizer assert ----
  KPF="$HERE/s8_keepalive_${ARM}_${KEEPA_REPS}.txt"
  [ -f "$KPF" ] || python3 - "$KEEPA_REPS" "$KPF" <<'PY'
import sys
reps, pf = int(sys.argv[1]), sys.argv[2]
open(pf, "w").write("the robot walked slowly across the red planet " * reps)
PY

  python3 - "$MODEL" "$KPF" "$KEEPA_TOKEN_LIMIT" "$ARM" <<'PY' | tee -a "$RESULT"
import sys
from transformers import AutoTokenizer

model, pf, limit, arm = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
n_tok = len(tok(open(pf).read())["input_ids"])
print(f"KEEPALIVE_TOKEN_CHECK arm={arm} file={pf} n_tok={n_tok} limit={limit} "
      f"margin={limit - n_tok}")
if n_tok > limit:
    print(f"KEEPALIVE_TOKEN_CHECK_FAILED arm={arm} n_tok={n_tok} exceeds limit={limit} -- "
          "ABORT_BEFORE_BOOT (PREREG rev2 H2, regression guard for F4 1793>1792)")
    sys.exit(1)
print(f"KEEPALIVE_TOKEN_CHECK_OK arm={arm} n_tok={n_tok}")
PY
  KTC_RC=${PIPESTATUS[0]}
  if [ "$KTC_RC" != 0 ]; then
    echo "ABORT: keepalive token check failed pre-boot for arm=$ARM (see" \
         "KEEPALIVE_TOKEN_CHECK_FAILED above). CTX/KEEPA_REPS are shared across" \
         "arms in this job invocation, so this is treated as a hard failure," \
         "not a per-arm skip." | tee -a "$RESULT"
    exit 12
  fi

  TOTAL_BOOTS_ATTEMPTED=0
  TOTAL_BOOTS_FAILED=0
  CELL_DISABLED=( [d16]=0 [d92]=0 )

  for BLK in $(seq 1 $BOOTS); do
    if [ "$SMOKE" = 1 ]; then
      ORDER="d16"
    elif [ $(( BLK % 2 )) -eq 1 ]; then
      ORDER="d16 d92"     # C8: alternate visiting order by block parity so
    else                  # cell identity is not confounded with boot-order /
      ORDER="d92 d16"     # wall-clock drift.
    fi
    echo "BLOCK $BLK/$BOOTS order=[$ORDER]" | tee -a "$RESULT"
    for NAME in $ORDER; do
      if [ "${CELL_DISABLED[$NAME]}" = 1 ]; then
        echo "cell=$NAME SKIPPED block=$BLK (H7 disabled it earlier this arm)" | tee -a "$RESULT"
        continue
      fi
      run_cell "$ARM" "$MODEL" "$BACKEND" "$NAME" "$BLK" "$RESULT"
      CELL_RC=$?
      [ "$CELL_RC" = 2 ] && CELL_DISABLED[$NAME]=1
    done

    # Pre-registered stop rule (PREREG rev2 SS5): boot failure rate > 25%,
    # evaluated arm-scoped, once at least 4 boots have been attempted (avoid
    # a single early flake tripping the whole arm).
    if [ "$TOTAL_BOOTS_ATTEMPTED" -ge 4 ]; then
      STOP=$(awk -v f="$TOTAL_BOOTS_FAILED" -v a="$TOTAL_BOOTS_ATTEMPTED" \
        'BEGIN{print (f/a > 0.25) ? 1 : 0}')
      if [ "$STOP" = 1 ]; then
        FRAC=$(awk -v f="$TOTAL_BOOTS_FAILED" -v a="$TOTAL_BOOTS_ATTEMPTED" \
          'BEGIN{printf "%.3f", f/a}')
        echo "STOP_RULE_TRIGGERED arm=$ARM boot_failure_rate=$FRAC" \
             "attempted=$TOTAL_BOOTS_ATTEMPTED failed=$TOTAL_BOOTS_FAILED --" \
             "aborting remaining blocks for this arm (PREREG rev2 SS5)" | tee -a "$RESULT"
        break
      fi
    fi
  done

  echo "SWEEP_DONE ARM=$ARM CTX=$CTX BOOTS_ATTEMPTED=$TOTAL_BOOTS_ATTEMPTED" \
       "BOOTS_FAILED=$TOTAL_BOOTS_FAILED CELL_DISABLED_d16=${CELL_DISABLED[d16]}" \
       "CELL_DISABLED_d92=${CELL_DISABLED[d92]}" | tee -a "$RESULT"
done
