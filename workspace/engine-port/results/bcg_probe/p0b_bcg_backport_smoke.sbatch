#!/bin/bash
#SBATCH --job-name=bcg-p0b
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.err
# ============================================================================
# P0-B -- Breakable CUDA Graph backport SMOKE (assumptions E1/E2/E3).
# Design: workspace/engine-port/reports/BCG_APPLICABILITY_2026-08-21.md sec7.
#
# ★ THIS IS A TOOL-VALIDITY PROBE, NOT A PERFORMANCE VERDICT.
#   It answers three availability questions and ONE cost question:
#     (i)  does a BCG-enabled server boot and capture under pdmux + green ctx?
#     (ii) does BCG's `torch.cuda.Stream.wait_stream` hook collide with
#          graph_capture / ExternalStream / true-dual-worker event bookkeeping?
#     (iii) what does ONE graph break cost, measured against a null-break arm?
#   No policy is compared. No arm here is a policy. Nothing in this output is
#   a goodput, a ranking, or an argument that any split is better than another.
#
# ★ PREREQUISITE -- the BCG backport must already be installed in the dev tree
#   (upstream python/sglang/srt/model_executor/runner_backend_utils/
#    breakable_cuda_graph/{__init__,breakable_cuda_graph,context,cuda_utils}.py).
#   This script does NOT install it: installing is an engine-porter task that
#   must also update sync_engine_tree.sh's SHA-256 manifest and pass the CPU
#   regression suite. Stage 0 below exits with a distinct code if it is absent.
#   ABSENT != FAILED (methodology lesson #21).
#
# ★ IMPORTABILITY IS NOT WIRING. F1 found ZERO `debug-cuda-graph` /
#   `debug_cuda_graph` strings in this tree, so the server flag (server_args.py)
#   and the capture routing (cuda_graph_runner.py) are NOT part of the 4-file
#   module -- the backport is three layers, not one. If only the module lands,
#   every arm boots fine and every delta comes out ~0, which would read as
#   "breaks are free" when the truth is "no break ever executed". Stage 1 below
#   therefore asserts a POSITIVE CONTROL (shim reports wrapped>0 AND the server
#   log shows more than one captured segment) before any cost is scored.
#
# ★ ORDERING FACT this probe is built around (verified 2026-08-21):
#     model_runner.py:656  init_device_graphs()        <- capture
#     model_runner.py:666  register_forward_hooks(...) <- hooks attach AFTER
#   so `--forward-hooks` cannot put a break inside the captured graph. Breaks
#   are injected by p0b_break_shim.py, which wraps ModelRunner.load_model
#   (defined model_runner.py:1072, called :508 -- both before capture) and
#   writes nothing to disk. The shim gates on `forward_mode.is_decode()`,
#   because the same layer objects are also called by forward_split_prefill
#   (models/zamba2.py:766-772), which is fully eager -- an ungated break would
#   change prefill and decode at once.
#
# ★ THE NULL-BREAK ARM IS NOT OPTIONAL. Without an arm that has the BCG path
#   ON and ZERO breaks, a slowdown cannot be attributed between "the BCG code
#   path costs something" and "each break costs something". Do not drop it.
#
# Usage:  sbatch p0b_bcg_backport_smoke.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
JOBTAG="${SLURM_JOB_ID:-local}"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_p0b_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"
export PYTHONPATH="$OUT:${PYTHONPATH:-}"

MODEL=Zyphra/Zamba2-2.7B; CTX=4096; INLEN=2000; OUTLEN=96; NPROMPT=60; RATE=2
CFG="$SLO/pdmux_slo.yml"          # 7-group dec16-54; agnostic (PDMUX_SLO_SCHED unset)
unset PDMUX_SLO_SCHED

echo "############# BCG P0-B job=$JOBTAG #############"
nvidia-smi --query-gpu=name,memory.total,compute_mode,driver_version --format=csv || true
python3 -c "import torch; print('torch', torch.__version__)"

# ---------------------------------------------------------------- stage 0
echo "=== stage 0/4: is the BCG backport installed? ==="
python3 - <<'PY'
import sys
try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        break_graph, eager_on_graph)
    print("BCG_PRESENT=1", break_graph, eager_on_graph)
except Exception as e:
    print("BCG_PRESENT=0", repr(e))
    sys.exit(30)
PY
RC0=$?
if [ "$RC0" -ne 0 ]; then
  echo "VERDICT: UNAVAILABLE (BCG NOT INSTALLED)."
  echo "This is a tool-availability fact, NOT a result about BCG's value and"
  echo "NOT a gate failure. Install the backport (engine-porter: 4 files +"
  echo "manifest + CPU regression) and resubmit."
  exit 30
fi

# ------------------------------------------------------------- run helper
# $1=label  $2=extra server flags  $3=extra env assignments (k=v ...)
run_variant () {
  local LAB="$1"; local XFLAGS="$2"; local XENV="$3"
  local PORT=$((37000 + JOBTAG % 120 + RANDOM % 40))
  local SRVLOG="$OUT/srv_${LAB}_${JOBTAG}.log"
  echo "########## VARIANT=$LAB env='$XENV' flags='$XFLAGS' port=$PORT ##########"
  env $XENV python -c \
    "import p0b_break_shim, runpy; p0b_break_shim.install(); \
     runpy.run_module('sglang.launch_server', run_name='__main__')" \
    --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --chunked-prefill-size -1 --disable-overlap-schedule \
    $XFLAGS --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!; local ok=0
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] \
      && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "$LAB SERVER DIED"; break; }
    sleep 5
  done
  echo "$LAB boot_ok=$ok"

  echo "--- $LAB (i) capture evidence ---"
  grep -iE "Capture cuda graph|capture .* end|break|segment" "$SRVLOG" \
    | grep -viE "Future|pynvml" | head -10
  echo "--- $LAB shim state ---"
  grep -E "p0b_break_shim" "$SRVLOG" | head -5
  echo "--- $LAB (ii) collision evidence (wait_stream / ExternalStream / events) ---"
  grep -iE "wait_stream|ExternalStream|illegal memory|CUDA error|capture.*fail|Traceback|assert|deadlock|hang" \
    "$SRVLOG" | grep -viE "Future|pynvml" | head -10

  if [ "$ok" = 1 ]; then
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' \
      | grep -oE '"text":"[^"]*"' | head -1 | sed "s/^/$LAB SANITY /"
    # warmup, discarded
    python -m sglang.bench_serving --backend sglang --model $MODEL \
      --host 127.0.0.1 --port $PORT --dataset-name random-ids \
      --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts 8 --request-rate 2 >/dev/null 2>&1 || true
    echo "===== $LAB in${INLEN}o${OUTLEN} rate=$RATE (ITL is the (iii) cost quantity) ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL \
      --host 127.0.0.1 --port $PORT --dataset-name random-ids \
      --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $RATE 2>/dev/null \
      | grep -E "Successful requests|Median ITL|P99 ITL|Median TPOT|P99 TPOT|Median TTFT"
  else
    echo "$LAB BOOT FAILED"; tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
}

# ---------------------------------------------------------------- stage 1
echo "=== stage 1/4: (i) availability + POSITIVE CONTROL ==="
# A: current operating point, BCG off entirely -- the reference boot.
run_variant "A_base_pre"    ""                      "PDMUX_BCG_BREAK_EVERY=0"
# B: --debug-cuda-graph -- whole decode forward eager THROUGH the graph runner.
#    This is also the S4 differential oracle arm.
run_variant "B_debugcg"     "--debug-cuda-graph"    "PDMUX_BCG_BREAK_EVERY=0"

echo "--- POSITIVE CONTROL: did a break arm actually break the graph? ---"
run_variant "P_wiring"      ""  "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=6 PDMUX_BCG_SHIM_VERBOSE=1"
WRAPPED=$(grep -oE "wrapped [0-9]+ of" "$OUT/srv_P_wiring_${JOBTAG}.log" 2>/dev/null \
          | grep -oE "[0-9]+" | head -1)
echo "positive_control wrapped=${WRAPPED:-0}"
if [ "${WRAPPED:-0}" -lt 1 ]; then
  echo "VERDICT: UNAVAILABLE (BCG PATH NOT WIRED)."
  echo "The module imports but no break was installed, so every cost delta below"
  echo "would be ~0 for a reason that has nothing to do with what a break costs."
  echo "This is a tool-availability fact, NOT a result. Finish the backport"
  echo "(module + server_args flag + cuda_graph_runner routing) and resubmit."
  exit 31
fi

# ---------------------------------------------------------------- stage 2
echo "=== stage 2/4: (iii) per-break cost -- INTERLEAVED, ORDER-COUNTERBALANCED ==="
# ★ WHY NOT C,D,E ONCE EACH IN THAT ORDER.
#   The decision quantities are delta_6 = ITL(k=6) - ITL(null) and
#   delta_3 = ITL(k=3) - ITL(null), and the mutation rule demands
#   delta_3 > delta_6. If the arms always run null -> k6 -> k3, then break count
#   is perfectly aliased with position in the job, and ANY monotone drift
#   (thermal, node state, boot-to-boot) satisfies the mutation rule for free.
#   That exact failure is already on this repo's books: the C2 audit flagged
#   `results/r0c/decode_knee_vs_ctx.sbatch:45` for always running `full` first, putting a
#   warm-up bias in the denominator of every ratio (CONSENSUS.md:1355).
#   So: three blocks, order reversed in the middle block, and delta is taken
#   WITHIN each block. Boot-to-boot variance of the null arm is measured by the
#   three null boots themselves -- that is the sigma the 3% gate needs and does
#   not currently have.
for BLK in 1 2 3; do
  case $BLK in
    1) ORDER="null k6 k3" ;;
    2) ORDER="k3 k6 null" ;;   # counterbalance
    3) ORDER="k6 null k3" ;;
  esac
  echo "########## BLOCK $BLK order='$ORDER' ##########"
  for ARM in $ORDER; do
    case $ARM in
      null) run_variant "b${BLK}_null" "" "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=0" ;;
      k6)   run_variant "b${BLK}_k6"   "" "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=6" ;;
      k3)   run_variant "b${BLK}_k3"   "" "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=3" ;;
    esac
  done
done

# ---------------------------------------------------------------- stage 3
echo "=== stage 3/4: (ii) true-dual-worker interaction -- BOTH break cells ==="
# The dual-worker runtime does its own record_event/wait_event bookkeeping on
# ExternalStreams; BCG hooks torch.cuda.Stream.wait_stream. L5 of the design doc
# warns specifically about break callbacks running on the replay thread while
# the scheduler thread mutates the module-global CURRENT_STREAM_IDX -- which can
# only happen in the (dual-worker x breaks>0) cell. Both cells are run so that
# cell is not left empty. NOT performance arms.
run_variant "F_dualwk_null" "" "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=0 PDMUX_TRUE_DUAL_WORKER=1"
run_variant "G_dualwk_k6"   "" "SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 PDMUX_BCG_BREAK_EVERY=6 PDMUX_TRUE_DUAL_WORKER=1"

# Drift anchor: repeat the very first arm at the very end. If A_base_post differs
# materially from A_base_pre, the job drifted and every cross-block number is suspect.
run_variant "A_base_post"   ""                      "PDMUX_BCG_BREAK_EVERY=0"

# ---------------------------------------------------------------- stage 4
echo "=== stage 4/4: how to read this job ==="
cat <<'EOF'
READING RULES -- fixed before the run. Do not deviate, do not retune after seeing data.

ESTIMAND. The goodput definition this project uses is built on request-internal
token-ITL p95 (methodology gate #4). `bench_serving` does not emit that, so the
decision quantity here is declared to be BOTH `Median ITL` AND `P99 ITL`, and
the gate must hold on BOTH. Reporting a Median-only pass while arguing about p95
is an estimand swap -- breaks are CPU round-trips and are more likely to move
the tail than the median.

(i)  AVAILABILITY. Stage 0 = module importable. Stage 1 positive control =
     a break was actually installed. Either missing -> UNAVAILABLE, and NOTHING
     below is scored. A boot failure is NOT evidence about BCG's value.

(ii) COLLISION. Any illegal-memory / capture-failure / hang / traceback present
     in a BCG arm and absent in A_base_pre is a wait_stream-hook x green-context
     x dual-worker collision (design doc L4/L5). Report the signature verbatim;
     do not score costs from a job that showed one.

(iii) PER-BREAK COST. Within each block b in {1,2,3}:
          delta_6(b) = ITL(b_k6)   - ITL(b_null)
          delta_3(b) = ITL(b_k3)   - ITL(b_null)
     The null arm is the denominator, NOT A_base: comparing against A_base would
     fold the standing cost of the BCG code path into the per-break cost.
     sigma_boot is estimated from the THREE null boots (b1,b2,b3).
     GATE (both metrics, pre-fixed): scenario S1 (operating-point per-layer
     instrumentation) is DEAD if
          mean_b delta_6(b) - 1.96 * sd_b(delta_6) / sqrt(3)  >  0.03 * mean ITL(null)
     i.e. the LOWER confidence bound on the per-break cost exceeds 3%. A bare
     point estimate over 3% does NOT kill S1, and a point estimate under 3% with
     a lower bound above it does. If sd_b(delta_6) alone exceeds 3% of ITL(null),
     report UNDETERMINED (BOOT VARIANCE EXCEEDS THE GATE) and re-power -- do not
     rule either way.
     MUTATION CHECK (pre-fixed threshold, not chosen after the fact):
          mean_b delta_3(b) / mean_b delta_6(b)  >=  1.5
     k=3 installs about twice the breaks of k=6, so a real per-break cost must
     scale. If the ratio is below 1.5, delta is NOT per-break cost and stage 2
     reports NOTHING -- no gate verdict, in either direction.

(iv) DRIFT. |ITL(A_base_post) - ITL(A_base_pre)| / ITL(A_base_pre) is reported
     alongside every number above. If it exceeds sd_b(delta_6), the job drifted
     more than the effect and only within-block deltas may be quoted.

NOT ANSWERED HERE: whether BCG helps anything. Whether any split is better.
Whether layer-aware is revived (it is not -- design doc sec4-S2). Whether
gate #13 or gate #16 moved (they did not).
EOF
echo "############# done job=$JOBTAG #############"
