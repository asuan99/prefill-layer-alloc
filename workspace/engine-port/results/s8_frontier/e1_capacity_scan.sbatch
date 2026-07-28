#!/bin/bash
#SBATCH --job-name=e1capscan
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# E1 CAPACITY PRE-SCAN -- required before e1_sweep.sbatch (DESIGN.md sec 4,
# methodology gate #6: never measure at a TTFT~SLO metric cliff, pick the
# offered rate from a measured off-cliff band, not by assumption).
#
# For each (arm, D-cell) this boots the server ONCE and fires a short
# ShareGPT open-loop probe (sglang.bench_serving, same tool the main sweep
# uses) at each candidate rate in RATES, in increasing order, reporting raw
# TTFT/ITL percentiles + achieved throughput + error rate per probe. No
# goodput/SLO judgement is computed here on purpose -- SLO_TTFT_MS/
# SLO_ITL_P95_MS are themselves not yet chosen (DESIGN.md sec 4.3); a human
# (or result-analyst) reads the printed table, finds the elbow (where median
# TTFT stops being flat), and picks RATE_LO/RATE_HI for e1_sweep.sbatch
# comfortably below it -- same "locate cliff" logic g2_0_decliff/g2_0_pre
# used, adapted to a per-(arm,D) grid because CAPACITY DIFFERS BY D (a
# decode-starved cell like D16 has lower decode throughput; a
# prefill-starved cell like D92 has lower prefill throughput -- see
# DESIGN.md sec 4.2 "different arms/cells may have different off-cliff
# bands").
#
# NOT a claim job, no goodput computed, nothing here is cited without
# claims-auditor. Usage:
#   sbatch --export=ALL,ARM_IDS="3",CELLS="d16 d24 d44 d54 d92" e1_capacity_scan.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_e1cap_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(M8 Ha8 Hs8 T8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
  "nvidia/Nemotron-H-8B-Base-8K"
  "Qwen/Qwen2.5-7B"
)
# Same unavoidable backend asymmetry as s8_scaleup/s8p_prefill.
ARM_BACKENDS=(triton triton flashinfer triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

ARM_IDS="${ARM_IDS:-0 1 2 3}"
CELLS="${CELLS:-d16 d24 d44 d54 d92}"
RATES="${RATES:-1 2 3 4 6 8 12 16}"
PROBE_TARGET_S="${PROBE_TARGET_S:-20}"   # aim each probe at roughly this many seconds
NP_MIN="${NP_MIN:-15}"
JOBTAG="${SLURM_JOB_ID:-local}"

for AIDX in $ARM_IDS; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/e1cap_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CELLS=[$CELLS] RATES=[$RATES]" | tee -a "$RESULT"

for NAME in $CELLS; do
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG"; continue; }
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="e1cap_${ARM}_${NAME}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="capscan_${ARM}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== ARM=$ARM cell=$NAME target_decode_sm=$DSM cfg=$(basename $CFG) port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend $BACKEND \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok cell=$NAME" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for RATE in $RATES; do
      NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")
      JOUT="$HERE/${RUNID}_r${RATE}.jsonl"; : > "$JOUT"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE --output-details --output-file "$JOUT" \
        > "$HERE/${RUNID}_r${RATE}_bench.log" 2>&1
      BENCH_RC=$?
      if [ "$BENCH_RC" != 0 ]; then
        echo "PROBE_FAILED cell=$NAME rate=$RATE rc=$BENCH_RC (see ${RUNID}_r${RATE}_bench.log)" | tee -a "$RESULT"
        continue
      fi
      python3 - "$JOUT" "$NAME" "$RATE" <<'PY' | tee -a "$RESULT"
import json, sys
def pct(a, q):
    if not a: return float("nan")
    a = sorted(a); return a[min(len(a) - 1, int(q * len(a)))]
name, rate = sys.argv[2], sys.argv[3]
tt, il, errs, n = [], [], 0, 0
dur = 0.0
for ln in open(sys.argv[1]):
    ln = ln.strip()
    if not ln:
        continue
    o = json.loads(ln)
    T = o.get("ttfts") or []; I = o.get("itls") or []; E = o.get("errors") or []
    dur += o.get("duration") or 0.0
    for i, t in enumerate(T):
        n += 1
        err = E[i] if i < len(E) else ""
        if err:
            errs += 1
            continue
        m = I[i] if i < len(I) and I[i] else []
        p95 = pct([x * 1000.0 for x in m], 0.95) if m else float("nan")
        tt.append(t); il.append(p95)
il = [x for x in il if x == x]
print(f"CAPSCAN cell={name} rate={rate} n={n} errs={errs} dur={dur:.1f}s "
      f"achieved_rps={((n-errs)/dur if dur else 0):.2f} "
      f"TTFT_ms p50/p95/p99={pct(tt,.5)*1000:.0f}/{pct(tt,.95)*1000:.0f}/{pct(tt,.99)*1000:.0f} "
      f"ITLp95_ms p50/p95={pct(il,.5):.1f}/{pct(il,.95):.1f}")
PY
      # Light settle between rate probes (NOT the full running==0/queued==0
      # drain-detection loop e1_sweep.sbatch uses between LO/HI phases --
      # this scan is elbow-finding only, never cited directly, so a fixed
      # pause is an acceptable cost/rigor tradeoff; DESIGN.md sec 4.2 notes
      # this explicitly).
      sleep 5
    done
    python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" "${PIN_GATE:-0.80}" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "CAPSCAN_DONE ARM=$ARM" | tee -a "$RESULT"
done
rm -rf "$TRITON_CACHE_DIR"
