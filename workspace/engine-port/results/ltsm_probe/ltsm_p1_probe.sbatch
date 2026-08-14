#!/bin/bash
#SBATCH --job-name=ltsm_p1_probe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# LTSM P1 -- per-layer-type SM response, single-cell ADMISSIBILITY PROBE.
#
# Pre-registration (READ FIRST, this script is not self-contained authority):
#   PREREG_LTSM_P1_PROBE_2026-08-14.md (same directory).
# Design this probe is drawn from (NOT a pre-registration -- sec12/13 of that
# doc explicitly forbid citing it as one):
#   workspace/engine-port/reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md
#
# WHAT THIS DOES: boots Zamba2-7B-Instruct ONCE (eager, per-layer-type CUDA-
# event timing already resident in the synced tree, zero dev-tree writes),
# pins decode to two SM values {44,92} at two prompt lengths {1024,2048}
# tokens (matched target concurrency, NOT the (L=2048,B~=88) cell the design
# doc's sec5 literally wrote -- see prereg sec3 correction 2 for why that
# cell is infeasible), and renders four admissibility verdicts (A1 physical
# invariant, A2 device-vs-host-paced, E2 instrumentation self-cost, E8
# mamba-track structural check) BEFORE any GPU hour is spent on the full
# (L,B,SM) grid this probe exists to gate.
#
# WHAT THIS DOES NOT DO: it does not produce, and must not be cited for, any
# claim about which layer type is more SM-sensitive (that is L2, gated on
# this job passing), any operating-point (cudagraph-ON) number, or any
# policy/goodput/TTFT/ITL claim. See prereg sec2 (self-invalidation) and
# sec9 (payoff, gate #28).
#
# NEW FILE (not an edit of results/r0c/decode_knee_vs_ctx_v2.sbatch): that
# harness's byte content is pinned by
# workspace/engine-port/tests/test_p5_gate_tools.py:40-41,176-183,237-243 --
# an in-place edit here would silently break that regression and the E1-b
# gate that depends on it (g2s_run.sbatch:91). Verified this session by
# reading the test file directly, not assumed.
#
# REUSES, UNMODIFIED: results/r0c/analyze_decode_knee_vs_ctx_v2.py (GATE C,
# GATE N / gate_n_conditional, host_boundness_rows, batch_mixture -- all
# already generic in (n_attn, n_mamba), so a 7B model is a CLI-flag change,
# not a code change: --n-attn 13 --n-mamba 81). This job writes its result/
# server-log files under THIS directory using that analyzer's hard-coded
# `deckneectx2_result_C*_r*_<jobid>.txt` / `deckneectx2_srv_C*_r*_<jobid>.log`
# naming convention so `load()` works with zero modification -- that is a
# deliberate reuse of the unmodified tool's file-discovery glob, not a
# copy-paste artefact from r0c.
#
# Manifest trip-wire: P1 changes NOTHING in the shared dev tree (design doc
# sec13, prereg sec1). This job must reproduce the SAME 15-file manifest as
# jobs 877756/877757/877974 EXACTLY -- it does NOT freeze a new reference
# (task instruction: reuse, don't re-baseline).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/ltsm_probe
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
JOBTAG="${SLURM_JOB_ID:-manual}"
export TRITON_CACHE_DIR="$OUT/.triton_cache_${JOBTAG}"   # job-scoped (methodology gate: triton cache race)

MODEL=Zyphra/Zamba2-7B-Instruct
# V-confirmed THIS SESSION (not carried over from the design doc without
# recheck): hf_cache/.../Zamba2-7B-Instruct/config.json ->
# num_hidden_layers=81, len(hybrid_layer_ids)=13, max_position_embeddings=4096.
N_ATTN=13
N_MAMBA=81
CTX_CAP=2560       # >= 2048 + OUTTOK(32) + margin; < max_position_embeddings(4096)
CONC=64
OUTTOK=32
TARGET_CAP=48      # correction 2 (prereg sec3): NOT the design doc's literal
                   # B~=88 at L=2048 -- Ha8 B_max(mem) at L=2048 is 57 (design
                   # doc sec6.2), so 88 OOMs by the design's own arithmetic.
                   # 48 is safe with margin at BOTH L=1024 (headroom to 98)
                   # and L=2048 (headroom to 57), and matches the L=2048 cell
                   # the design doc's OWN sec6.3 full-grid table settles on.
MEAS_ROUNDS=2
BLOCK_EVERY=8

echo "==================== PRE-REGISTERED DECISION RULES ===================="
sed -n '/^# ========================= PRE-REGISTERED RULES/,/^# =====================/p' "$0"
echo "=========================================================================="

# ---------------------------------------------------------------- manifest trip-wire
REF_MANIFEST="workspace/engine-port/results/p1_gates/gate1/runtime_source_manifest_gate1c_877974.sha256"
MANIFEST="$OUT/runtime_source_manifest_ltsmp1_${JOBTAG}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || { echo "SYNC FAILED" >&2; exit 3; }
DIFF="$OUT/manifest_diff_ltsmp1_${JOBTAG}.txt"
if ! diff <(sort "$REF_MANIFEST") <(sort "$MANIFEST") > "$DIFF"; then
  echo "MANIFEST_CHECK=FAIL -- tree drifted from the 877756/877757/877974-verified state." >&2
  echo "P1 is a NO-CHANGE-TO-TREE experiment (design doc sec13); this must be an exact match, not added==N." >&2
  cat "$DIFF" >&2
  echo "MANIFEST_MISMATCH_ABORT" >&2
  exit 4
fi
echo "MANIFEST_CHECK=PASS (byte-identical to reference, 0 diff lines)"

python -m unittest discover -s workspace/engine-port/tests -v 2>&1 \
  || { echo "CPU_REGRESSION_FAILED" >&2; exit 5; }
echo "CPU_REGRESSION=PASS"

PORT=$((35500 + ${SLURM_JOB_ID:-0} % 300))
BOOT_TIMEOUT_S=220

boot_server () {   # $1 = extra server flags, $2 = srvlog path, $3 = 1/0 export TIMING
  # Sets globals BOOT_OK / BOOT_SRV_PID -- NOT a return-via-stdout design.
  # A backgrounded server started inside a `$(...)` command-substitution
  # subshell is reparented when that subshell exits; the PID stays valid in
  # practice, but that is a fragile thing to depend on inside a job script
  # nobody is watching interactively. Globals avoid the subshell entirely.
  if [ "$3" = "1" ]; then
    export SGLANG_ZAMBA_TIMING=1
    export SGLANG_ZAMBA_TIMING_EVERY=$BLOCK_EVERY
    export SGLANG_ZAMBA_CLOSURE_MIN=0.85
  else
    # trap (prereg sec3 correction 1, confirmed this session by direct read
    # of zamba2.py:544): `bool(os.environ.get("SGLANG_ZAMBA_TIMING"))` is
    # True for the STRING "0" (any non-empty string is truthy in Python).
    # The OFF arm MUST unset, never set-to-"0".
    unset SGLANG_ZAMBA_TIMING SGLANG_ZAMBA_TIMING_EVERY SGLANG_ZAMBA_CLOSURE_MIN || true
  fi
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  # shellcheck disable=SC2086
  python -m sglang.launch_server --model-path "$MODEL" \
    --trust-remote-code --dtype bfloat16 --attention-backend triton \
    --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests $TARGET_CAP \
    --context-length $CTX_CAP $1 \
    --host 127.0.0.1 --port $PORT > "$2" 2>&1 &
  BOOT_SRV_PID=$!
  BOOT_OK=0
  for i in $(seq 1 $BOOT_TIMEOUT_S); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { BOOT_OK=1; break; }
    kill -0 $BOOT_SRV_PID 2>/dev/null || { echo "SERVER DIED"; break; }
    sleep 1
  done
}

teardown_server () {   # $1 = pid
  kill "$1" 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 2
}

make_body () {   # $1 = ctx (tokens, approx), $2 = out tokens, $3 = output file
  python3 - "$1" "$2" "$3" <<'PY'
import json, sys
ctx, out, bf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
reps = max(1, ctx // 8)
text = 'the robot walked slowly across the red planet ' * reps
json.dump({'text': text, 'sampling_params': {'max_new_tokens': out, 'temperature': 0.7, 'ignore_eos': True}}, open(bf, 'w'))
PY
}

fire_round () {   # $1 = smfile content, $2 = smfile path, $3 = body file, $4 = conc
  echo "$1" > "$2"
  local pids=""
  for r in $(seq 1 "$4"); do
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      --data-binary @"$3" >/dev/null 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  sleep 2
}

# ============================================================================
# PHASE 0 -- PIPELINE SMOKE (methodology gate #26, amended 2026-08-14).
#
# This job's harness (this file) and its output-schema wiring (writing into a
# NEW directory under the r0c filename convention) are new relative to the
# last audited campaign (job 873783 / P5_GATES_BATCH1). Per gate #26's
# combined-batch trigger, this is submitted as part of a batch whose combined
# cost (this probe + the E1-b/c and %smid work noted in
# handoff-report/session_handoff_2026-08-13.md sec2.4-2.5) crosses 1 GPU-hr,
# even though THIS job alone is ~0.4-0.6 GPU-hr. This phase is the minimum-
# rep (n=1), real-execution-path, real-schema smoke this campaign owes before
# paying for the real cells below.
#
# WHAT A PASS HERE MEANS: the harness boots this model with these flags, the
# instrumentation emits ZBLT2 BLOCK lines the unmodified r0c analyzer can
# parse, and the analyzer exits 0 against this job's file-naming convention.
# WHAT A PASS HERE DOES NOT MEAN: nothing about A1/A2/E2/E8 -- those are
# unset by a 4-request, 8-output-token, single round. A pipeline-smoke PASS
# licenses NO performance or admissibility conclusion (gate #26 verbatim).
# ============================================================================
echo "==================== PHASE 0: PIPELINE SMOKE ===================="
# SMOKE_TAG is DELIBERATELY DISTINCT from JOBTAG (not just a different
# ctx/rep under the same tag): the r0c analyzer's file glob is
# `deckneectx2_result_C*_r*_<jobid>.txt` -- if the smoke's result file used
# the bare JOBTAG suffix, it would still be sitting in $OUT (only SMFILE/
# BODY temp files are deleted below, deliberately -- the smoke RESULT/srvlog
# are kept as an artefact) when PHASE 3 globs for "$JOBTAG", and its 1-round,
# 4-request, first-block-dropped data would get silently pooled into the
# real GATE C/N computation. A distinct suffix makes that structurally
# impossible rather than relying on remembering to delete it in time.
SMOKE_TAG="${JOBTAG}s"
SMOKE_SRVLOG="$OUT/deckneectx2_srv_C99_r9_${SMOKE_TAG}.log"
SMOKE_RESULT="$OUT/deckneectx2_result_C99_r9_${SMOKE_TAG}.txt"; : > "$SMOKE_RESULT"
{
  echo "HEADER model=$MODEL ctx=99 rep=9 conc=4 outtok=8 port=$PORT SMOKE=1"
  echo "HEADER cudagraph=DISABLED (--disable-cuda-graph --disable-piecewise-cuda-graph) EAGER"
  echo "HEADER git_head=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
} | tee -a "$SMOKE_RESULT"
boot_server "" "$SMOKE_SRVLOG" 1
SMOKE_OK=$BOOT_OK; SMOKE_PID=$BOOT_SRV_PID
if [ "$SMOKE_OK" != 1 ]; then
  echo "PHASE0_SMOKE=FAIL boot_ok=0" | tee -a "$SMOKE_RESULT" >&2
  tail -30 "$SMOKE_SRVLOG" | grep -viE "pynvml|Future"
  teardown_server "$SMOKE_PID"
  exit 7
fi
SMOKE_SMFILE="$OUT/.ltsmp1_smoke_${SMOKE_TAG}"
SMOKE_BODY="$OUT/.ltsmp1_smoke_body_${SMOKE_TAG}.json"
# outtok=24, block_every(inherited from the TIMING=1 export above)=8 -> 3 raw
# BLOCK emits; keep_steady() drops the first of the round, so ~2 should
# survive -- enough to exercise the KEPT-block path (build_cells/build_
# subcells/gate_n_conditional), not just the raw-BLOCK-line regex.
make_body 99 24 "$SMOKE_BODY"
fire_round "92@99r1" "$SMOKE_SMFILE" "$SMOKE_BODY" 4
grep "ZBLT2 mode=92@99r1 " "$SMOKE_SRVLOG" | sed "s|^|BLOCK ctx=99 rep=9 sm=92 round=1 |" >> "$SMOKE_RESULT"
echo "ARMDONE ctx=99 rep=9 sm=92 closure_fail_meas=0 closure_fail_warmup=0" | tee -a "$SMOKE_RESULT"
teardown_server "$SMOKE_PID"

N_SMOKE_BLOCKS=$(grep -c '^BLOCK ' "$SMOKE_RESULT" || true)
if [ "$N_SMOKE_BLOCKS" -lt 1 ]; then
  echo "PHASE0_SMOKE=FAIL: 0 BLOCK lines emitted (SGLANG_ZAMBA_TIMING not taking effect, or ZBLT2 regex drift)" >&2
  exit 8
fi
# NOTE: main() in the unmodified analyzer returns exit 1 by DESIGN whenever
# GATE C/N do not cleanly PASS ("gates first, numbers second") -- with a
# single n=1 smoke round that is the EXPECTED outcome, not a failure. So the
# smoke checks for the ABSENCE of a crash (traceback) and the PRESENCE of
# the expected structural sections, never the process exit code.
SMOKE_OUT="$OUT/smoke_analyzer_out_${JOBTAG}.txt"
python3 workspace/engine-port/results/r0c/analyze_decode_knee_vs_ctx_v2.py "$SMOKE_TAG" \
  --dir "$OUT" --n-attn "$N_ATTN" --n-mamba "$N_MAMBA" > "$SMOKE_OUT" 2>&1
if grep -q "Traceback (most recent call last)" "$SMOKE_OUT"; then
  echo "PHASE0_SMOKE=FAIL: unmodified r0c analyzer raised on this job's output (schema contract violation)" >&2
  tail -40 "$SMOKE_OUT" >&2
  exit 9
fi
MISSING=""
for marker in "GATE C" "GATE N" "HOST-BOUNDNESS OBSERVABLES"; do
  grep -q "$marker" "$SMOKE_OUT" || MISSING="$MISSING [$marker]"
done
if [ -n "$MISSING" ]; then
  echo "PHASE0_SMOKE=FAIL: analyzer ran without a traceback but did not print expected section(s):$MISSING (contract drift)" >&2
  tail -40 "$SMOKE_OUT" >&2
  exit 10
fi
echo "PHASE0_SMOKE=PASS  n_blocks=$N_SMOKE_BLOCKS  (PASS licenses NO performance/admissibility claim -- gate #26;"
echo "  a non-PASS GATE C/N verdict inside the smoke output is EXPECTED at n=1 and is not itself a smoke failure)"
rm -f "$SMOKE_SMFILE" "$SMOKE_BODY"

# ============================================================================
# PHASE 1 -- PRIMARY PROBE (SGLANG_ZAMBA_TIMING=1), rep=1 (n_indep=1, single
# boot -- same aggregation-unit caveat as P5_GATES_BATCH1: "reproducibility"
# here means block-to-block spread within ONE boot, not across boots).
# Cells: (ctx=1024, target_cap=48) and (ctx=2048, target_cap=48), each visited
# at SM in {44,92}. Visit order randomised (methodology gate: arm-order bias,
# P5_GATES defect 2 lineage).
# ============================================================================
echo "==================== PHASE 1: PRIMARY PROBE (TIMING=1) ===================="
SEED_BASE=${1:-20260814}
CELL_ORDER=$(python3 - "$SEED_BASE" <<'PY'
import random, sys
r = random.Random(int(sys.argv[1]))
cells = [(sm, ctx) for sm in ("44", "92") for ctx in (1024, 2048)]
r.shuffle(cells)
print(" ".join(f"{sm}:{ctx}" for sm, ctx in cells))
PY
)
echo "cell_order=$CELL_ORDER"

SRVLOG="$OUT/deckneectx2_srv_C0_r1_${JOBTAG}.log"   # one combined boot serves both ctx
RESULT_1024="$OUT/deckneectx2_result_C1024_r1_${JOBTAG}.txt"; : > "$RESULT_1024"
RESULT_2048="$OUT/deckneectx2_result_C2048_r1_${JOBTAG}.txt"; : > "$RESULT_2048"
SERVER_FLAGS="--trust-remote-code --dtype bfloat16 --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests $TARGET_CAP --context-length $CTX_CAP"
for RF in "$RESULT_1024" "$RESULT_2048"; do
  {
    echo "HEADER model=$MODEL ctxcap=$CTX_CAP rep=1 conc=$CONC outtok=$OUTTOK port=$PORT target_cap=$TARGET_CAP"
    echo "HEADER job=${JOBTAG} seed_base=$SEED_BASE cell_order=$CELL_ORDER"
    echo "HEADER block_every=$BLOCK_EVERY closure_min=0.85 n_attn=$N_ATTN n_mamba=$N_MAMBA"
    echo "HEADER cudagraph=DISABLED (--disable-cuda-graph --disable-piecewise-cuda-graph) EAGER: admissibility probe, NOT the operating point (structural, per-layer spans do not execute under cudagraph replay -- see prereg sec1 V-fact)"
    echo "HEADER server_flags=$SERVER_FLAGS"
    echo "HEADER zamba2_sha256=$(sha256sum /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/models/zamba2.py | cut -d' ' -f1)"
    echo "HEADER git_head=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
  } | tee -a "$RF"
done

boot_server "" "$SRVLOG" 1
OK1=$BOOT_OK; SRV1=$BOOT_SRV_PID
if [ "$OK1" != 1 ]; then
  echo "PHASE1_BOOT=FAIL" | tee -a "$RESULT_1024" "$RESULT_2048" >&2
  tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
  teardown_server "$SRV1"
  exit 11
fi
echo "boot_ok=1" | tee -a "$RESULT_1024" "$RESULT_2048"

# E8 structural check (prereg sec3 correction 3): --mamba-track-interval is
# NOT passed at all -- confirmed this session (schedule_batch.py:2144-2175,
# server_args.py:964-967) that the checkpoint mask is only ever computed
# inside `if enable_mamba_extra_buffer()`, which requires
# --mamba-scheduler-strategy=extra_buffer; the default `auto` resolves to
# `no_buffer` and --disable-radix-cache never overrides that resolution. So
# there is no counter to read; this queries the LIVE server's resolved
# strategy as a boot-time precondition, not an empirical firing count.
STRATEGY=$(curl -s http://127.0.0.1:$PORT/server_info | python3 -c "import json,sys; print(json.load(sys.stdin).get('mamba_scheduler_strategy','MISSING'))" 2>/dev/null || echo "QUERY_FAILED")
echo "E8_STRATEGY_CHECK mamba_scheduler_strategy=$STRATEGY" | tee -a "$RESULT_1024" "$RESULT_2048"
if [ "$STRATEGY" != "no_buffer" ]; then
  echo "E8_CHECK=UNEXPECTED (expected no_buffer) -- this session's code-trace assumption is violated; ABORTING rather than silently mis-scoring E8 (see prereg sec3 correction 3)" | tee -a "$RESULT_1024" "$RESULT_2048" >&2
  teardown_server "$SRV1"
  exit 12
fi
echo "E8_CHECK=STRUCTURALLY_INAPPLICABLE (no_buffer confirmed; checkpoint path does not execute in this config, not a measured-zero)" | tee -a "$RESULT_1024" "$RESULT_2048"

BODY_1024="$OUT/.ltsmp1_body_1024_${JOBTAG}.json"
BODY_2048="$OUT/.ltsmp1_body_2048_${JOBTAG}.json"
make_body 1024 $OUTTOK "$BODY_1024"
make_body 2048 $OUTTOK "$BODY_2048"
SMFILE="$OUT/.ltsmp1_${JOBTAG}"

declare -A BODY_OF=( [1024]="$BODY_1024" [2048]="$BODY_2048" )
declare -A RESULT_OF=( [1024]="$RESULT_1024" [2048]="$RESULT_2048" )
# wall-clock of each measured round, keyed "sm:ctx:round" -- feeds PHASE 2 (E2)
declare -A ROUND_WALL_S=()

for CELL in $CELL_ORDER; do
  SM="${CELL%%:*}"; CTX="${CELL##*:}"
  BODY="${BODY_OF[$CTX]}"; RF="${RESULT_OF[$CTX]}"
  fire_round "${SM}@${CTX}rw" "$SMFILE" "$BODY" "$CONC"
  echo "WARMUP ctx=$CTX rep=1 sm=$SM blocks=$(grep -c "mode=$SM@${CTX}rw " "$SRVLOG")" | tee -a "$RF"
  for RND in $(seq 1 $MEAS_ROUNDS); do
    T0=$(date +%s.%N)
    fire_round "${SM}@${CTX}r${RND}" "$SMFILE" "$BODY" "$CONC"
    T1=$(date +%s.%N)
    ROUND_WALL_S["${SM}:${CTX}:${RND}"]=$(python3 -c "print(f'{$T1-$T0:.4f}')")
    grep "ZBLT2 mode=$SM@${CTX}r${RND} " "$SRVLOG" \
      | sed "s|^|BLOCK ctx=$CTX rep=1 sm=$SM round=$RND |" >> "$RF"
  done
  CF_MEAS=$(grep -cE "ZBLT_CLOSURE_FAIL mode=$SM@${CTX}r[0-9]" "$SRVLOG")
  CF_WARM=$(grep -cE "ZBLT_CLOSURE_FAIL mode=$SM@${CTX}rw" "$SRVLOG")
  echo "ARMDONE ctx=$CTX rep=1 sm=$SM closure_fail_meas=$CF_MEAS closure_fail_warmup=$CF_WARM" | tee -a "$RF"
done
teardown_server "$SRV1"

echo "phase1_round_wall_s (SM:CTX:ROUND -> seconds):"
for k in "${!ROUND_WALL_S[@]}"; do echo "  $k = ${ROUND_WALL_S[$k]}"; done | tee -a "$OUT/phase1_round_wall_${JOBTAG}.txt"

# ============================================================================
# PHASE 2 -- E2 BASELINE (SGLANG_ZAMBA_TIMING unset): fresh boot, SAME server
# flags, SM=92 fixed, ctx=1024 only (prereg sec4.2: chosen as the single
# designated E2 cell -- highest-SM arm, so device work per step is smallest,
# making a fixed host-side instrumentation cost the LARGEST relative
# fraction of step time at this arm; a conservative choice for detecting
# overhead, not a favourable one). No ZBLT2 lines are expected (TIMING is
# unset); this measures ROUND WALL-CLOCK only, compared against Phase 1's
# sm=92,ctx=1024 rounds. Descriptive only -- no CI, no threshold, no gate
# (prereg sec5: this is a size-the-overhead measurement, not a hypothesis
# test; methodology gate #14 does not apply because no interval is computed).
# ============================================================================
echo "==================== PHASE 2: E2 BASELINE (TIMING unset) ===================="
SRVLOG2="$OUT/deckneectx2_srv_C1024_r2_${JOBTAG}.log"
boot_server "" "$SRVLOG2" 0
OK2=$BOOT_OK; SRV2=$BOOT_SRV_PID
if [ "$OK2" != 1 ]; then
  echo "PHASE2_BOOT=FAIL (E2 baseline not obtained -- report as UNDETERMINED, not as overhead=0)" >&2
  tail -30 "$SRVLOG2" | grep -viE "pynvml|Future"
  teardown_server "$SRV2"
  exit 13
fi
E2_SMFILE="$OUT/.ltsmp1_e2_${JOBTAG}"
declare -A E2_ROUND_WALL_S=()
fire_round "92@1024rw" "$E2_SMFILE" "$BODY_1024" "$CONC"
for RND in $(seq 1 $MEAS_ROUNDS); do
  T0=$(date +%s.%N)
  fire_round "92@1024r${RND}" "$E2_SMFILE" "$BODY_1024" "$CONC"
  T1=$(date +%s.%N)
  E2_ROUND_WALL_S["$RND"]=$(python3 -c "print(f'{$T1-$T0:.4f}')")
done
teardown_server "$SRV2"

E2_REPORT="$OUT/e2_overhead_${JOBTAG}.txt"
python3 - "$JOBTAG" "${ROUND_WALL_S[92:1024:1]:-nan}" "${ROUND_WALL_S[92:1024:2]:-nan}" \
          "${E2_ROUND_WALL_S[1]:-nan}" "${E2_ROUND_WALL_S[2]:-nan}" > "$E2_REPORT" <<'PY'
import sys, statistics
job, w1_1, w1_2, w0_1, w0_2 = sys.argv[1:6]
with_timing = [float(x) for x in (w1_1, w1_2) if x != "nan"]
without_timing = [float(x) for x in (w0_1, w0_2) if x != "nan"]
print(f"job={job}")
print(f"with_timing_rounds_s={with_timing}")
print(f"without_timing_rounds_s={without_timing}")
if with_timing and without_timing:
    mt, mo = statistics.mean(with_timing), statistics.mean(without_timing)
    pct = (mt - mo) / mo * 100.0 if mo > 0 else float("nan")
    print(f"E2_mean_with_timing_s={mt:.4f}")
    print(f"E2_mean_without_timing_s={mo:.4f}")
    print(f"E2_overhead_pct={pct:.2f}")
    print(f"E2_ADVISORY_FLAG={'YES (>=10%, review before L2)' if pct >= 10.0 else 'no'}")
    print("NOTE: n=2 rounds per side, wall-clock only, no CI computed -- descriptive")
    print("sizing per PREREG_LTSM_P1_PROBE_2026-08-14.md sec5, not a hypothesis test.")
else:
    print("E2_UNDETERMINED: missing round(s) on one side")
PY
cat "$E2_REPORT"

# ============================================================================
# PHASE 3 -- SCORE (reuses the UNMODIFIED r0c analyzer; this echo is a
# convenience read, NOT the analysis of record -- an independent re-parse
# of the raw BLOCK lines by result-analyst is required before any of this
# is cited, per this project's standing practice).
# ============================================================================
echo "==================== PHASE 3: SCORE (convenience echo, not analysis-of-record) ===================="
# jobid="$JOBTAG" (NOT the smoke's SMOKE_TAG) -- glob is
# deckneectx2_result_C*_r*_<jobid>.txt, an exact-suffix match, so this
# picks up ONLY RESULT_1024/RESULT_2048 (rep=1) and structurally cannot
# match the smoke's *_${SMOKE_TAG}.txt files (see PHASE 0 comment).
python3 workspace/engine-port/results/r0c/analyze_decode_knee_vs_ctx_v2.py "$JOBTAG" \
  --dir "$OUT" --n-attn "$N_ATTN" --n-mamba "$N_MAMBA" \
  | tee "$OUT/ltsmp1_analyzer_out_${JOBTAG}.txt"

rm -f "$SMFILE" "$E2_SMFILE" "$BODY_1024" "$BODY_2048"
echo "DONE job=${JOBTAG}"

# ========================= PRE-REGISTERED RULES =========================
# (echoed verbatim at job start -- see PHASE 0's `sed` above; keep this
# block's content in sync with PREREG_LTSM_P1_PROBE_2026-08-14.md sec5-6.)
#
# A1 (physical invariant, per SM arm, at TARGET_CAP=48's realized bs):
#   per_mamba ratio across ctx {1024,2048} at the SAME realized bs must be
#   <= 1.05 (gate_n_conditional, threshold PRE-REGISTERED, NOT re-tuned here
#   -- methodology gate #8). Any VIOLATION -> that arm FAILS A1.
#   UNSCOREABLE (no realized-bs overlap between the two ctx points) also
#   does NOT pass -- it is reported and campaign proceeds to sec9 (not
#   silently treated as OK). per_mlp (negative control) must show OK at
#   both arms as a sanity check; a per_mlp VIOLATION invalidates the whole
#   run (measurement corruption beyond the known host-pacing mechanism).
# A2 (device- vs host-paced, at ctx=1024): FAIL if fwd_ms(SM=44) < fwd_ms
#   (SM=92) [reversed/physically-suspect direction] OR if
#   fwd_ms(44)/fwd_ms(92) < 1.05 [not distinguishable beyond ~10x this
#   harness's demonstrated measurement noise, P5_GATES_BATCH1 Sec5:
#   round1/round2 agreement <=0.5%, cross-position <=0.31%]. 1.05 is reused
#   from A1's own already-registered value (design doc sec8 "새 자유모수
#   회계"), not a new free parameter chosen from this run's data.
# E2 (instrumentation self-cost): DESCRIPTIVE. No pass/fail. >=10% overhead
#   is an ADVISORY flag for human review before authorising L2, not an
#   auto-fail of this job.
# E8 (mamba-track-interval): BOOT-TIME STRUCTURAL PRECONDITION, not an
#   empirical firing count -- see PHASE 1 code comment. Abort (exit 12) if
#   violated, do not silently proceed.
# TOP-LEVEL: "ELIGIBLE_FOR_L2" iff GATE C == PASS (0.85, reused threshold)
#   AND A1 == OK for BOTH sm=44 and sm=92 AND per_mlp == OK for both AND
#   A2 == PASS at ctx=1024 AND E8 boot precondition held. Otherwise
#   "NOT_ELIGIBLE_FOR_L2" -- route to the design doc's own documented next
#   step (nsys path, sec5 "차선책"), do not re-run this job with adjusted
#   thresholds against its own outcome (methodology gate #8).
# A3 (monotonicity) is NOT evaluated here -- only 2 SM points exist in this
#   probe; monotonicity needs >=3 and is an L2-grid concern.
# ==========================================================================
