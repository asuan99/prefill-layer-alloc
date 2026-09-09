#!/bin/bash
#SBATCH --job-name=p4skew
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# P4 SKEW -- PREREG_PROBES_2026-09-08.md sec 4.  Measures the instrument
# itself, not the treatment: (a) does PDMUX_DUAL_WORKER_TRACE_EVERY=32 (the
# default subsampling used by P1/P2/every other probe/campaign) bias `f`? and
# (b) does PDMUX_STICKY_PARTITION=1 actually raise `f` (claimed by code
# reading only so far, `FINDING_DUTYCYCLE`)?
#
# Same cell as P1's d44 arm (d44 config, ctx 8192, rate 1.0, 160 requests,
# same seed=17 for workload comparability) x TRACE_EVERY=1, 2 boots:
#   boot A: sticky OFF (baseline, isolates the trace-every effect against P1)
#   boot B: PDMUX_STICKY_PARTITION=1 (isolates the sticky effect against boot A)
#
# ★REQUIRES P1_D44_SUMMARY env var pointing at P1's `summary_d44.json`
# (produced by p1_split_fires.sbatch) -- prediction 1 needs it.  Example:
#   sbatch --export=ALL,P1_D44_SUMMARY=.../p1_<jobid>/summary_d44.json p4_skew.sbatch
#
# Registered predictions (prereg sec 4, "틀리면 그게 결과다" -- printed, not judged):
#   1. |f_count(TRACE_EVERY=1,sticky OFF) - f_count(P1 d44, TRACE_EVERY=32)|
#      / f_count(P1 d44) <= 0.20  => subsampling harmless; else FINDING_BIN0's
#      repair must extend to every bin.
#   2. f_count(sticky ON) >= 3 * f_count(sticky OFF)  => FINDING_DUTYCYCLE sec2's
#      code-reading claim is supported; else withdrawn.
#   3. frac(split ∧ prefill_active_batch_size==0) is LARGER under sticky ON
#      than OFF; else the record-skew mechanism explanation is wrong.
#
# usage: sbatch --export=ALL,P1_D44_SUMMARY=<path> p4_skew.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
STAGE0=$ROOT/workspace/engine-port/results/stage0_xctrl
OUT="$PROBES/p4_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_FIXED_DECODE_SM_FILE \
      PDMUX_TRACE_FORCE_PREFILL

if [ -z "${P1_D44_SUMMARY:-}" ] || [ ! -f "${P1_D44_SUMMARY:-/nonexistent}" ]; then
  echo "P4_ABORT: P1_D44_SUMMARY not set or not found (need P1's summary_d44.json)" \
    | tee "$OUT/BOOT_FAILURES.txt"
  exit 3
fi
echo "P1_D44_SUMMARY=$P1_D44_SUMMARY"
cp "$P1_D44_SUMMARY" "$OUT/p1_d44_summary_reference.json"

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$STAGE0/pdmux_d44.yml"
CTX=8192
OUTTOK=96
CTXCAP=$((CTX + 512))
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED=17
RATE=1.0
NP=160

echo "P4 model=$MODEL backend=$BACKEND ctx=$CTX out=$OUTTOK rate=$RATE np=$NP job=$SLURM_JOB_ID"
echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

boot_and_bench () {   # $1=tag  $2=sticky(0/1)
  local TAG=$1 STICKY=$2
  local PORT=$((39900 + (SLURM_JOB_ID + STICKY + RANDOM) % 400))
  local SRVLOG="$OUT/srv_${TAG}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/p4_${TAG}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="p4_${TAG}_${SLURM_JOB_ID}"
  export PDMUX_WORKLOAD_ID="p4_${TAG}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=44
  export PDMUX_DUAL_WORKER_TRACE_EVERY=1
  if [ "$STICKY" = "1" ]; then
    export PDMUX_STICKY_PARTITION=1
  else
    unset PDMUX_STICKY_PARTITION
  fi

  echo ""
  echo "########## P4 tag=$TAG sticky=$STICKY port=$PORT ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  local PID=$!

  local ok=1
  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 tag=$TAG"
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
      --output-details --output-file "$OUT/bench_${TAG}.jsonl" \
      > "$OUT/bench_${TAG}.log" 2>&1
    echo "bench_rc=$? tag=$TAG"
    grep -E "Successful requests|Median TTFT|Median ITL|Concurrency:" "$OUT/bench_${TAG}.log"
    python3 "$PROBES/analyze_probes.py" "$PDMUX_TELEMETRY_PATH" 44 \
      --label "p4_${TAG}" --out "$OUT/summary_${TAG}.json" || echo "ANALYZE_FAILED tag=$TAG"
  else
    echo "BOOT_FAILED tag=$TAG" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    ok=0
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
  return $((1 - ok))
}

FAIL_STREAK=0
boot_and_bench sticky_off 0 || FAIL_STREAK=$((FAIL_STREAK + 1))
boot_and_bench sticky_on 1 || FAIL_STREAK=$((FAIL_STREAK + 1))
if [ "$FAIL_STREAK" -ge 2 ]; then
  echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES (both P4 boots failed, 순서·중단 규칙 #3)" \
    | tee -a "$OUT/BOOT_FAILURES.txt"
fi

echo "=== P4 registered predictions (printed, not judged -- prereg sec 4) ==="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

p1 = load(os.path.join(out, "p1_d44_summary_reference.json"))
off = load(os.path.join(out, "summary_sticky_off.json"))
on = load(os.path.join(out, "summary_sticky_on.json"))

result = {"probe": "P4_SKEW"}
if p1 is None or off is None:
    result["prediction_1"] = {"status": "UNRESOLVED", "reason": "missing P1 reference or sticky-off summary"}
else:
    f_p1 = p1["f_count"]
    f_off = off["f_count"]
    rel = abs(f_off - f_p1) / f_p1 if f_p1 else None
    result["prediction_1"] = {
        "f_count_p1_trace32": f_p1, "f_count_trace1_sticky_off": f_off,
        "rel_diff": rel,
        "within_20pct": (rel <= 0.20) if rel is not None else None,
        "reading": "subsampling harmless" if (rel is not None and rel <= 0.20)
                   else "FINDING_BIN0 repair scope must extend to all bins" if rel is not None
                   else "UNDEFINED (f_count_p1 == 0)",
    }

if off is None or on is None:
    result["prediction_2"] = {"status": "UNRESOLVED", "reason": "missing sticky on/off summary"}
else:
    f_off = off["f_count"]
    f_on = on["f_count"]
    ratio = (f_on / f_off) if f_off else None
    result["prediction_2"] = {
        "f_count_sticky_off": f_off, "f_count_sticky_on": f_on,
        "ratio_on_over_off": ratio,
        "ge_3x": (ratio >= 3.0) if ratio is not None else None,
        "reading": "FINDING_DUTYCYCLE sec2 code-reading supported" if (ratio is not None and ratio >= 3.0)
                   else "FINDING_DUTYCYCLE sec2 code-reading withdrawn" if ratio is not None
                   else "UNDEFINED (f_count_sticky_off == 0)",
    }

if off is None or on is None:
    result["prediction_3"] = {"status": "UNRESOLVED", "reason": "missing sticky on/off summary"}
else:
    fr_off = off["frac_split_pab0_of_split"]
    fr_on = on["frac_split_pab0_of_split"]
    larger = None
    if fr_off == fr_off and fr_on == fr_on:  # not NaN
        larger = fr_on > fr_off
    result["prediction_3"] = {
        "frac_split_pab0_of_split_off": fr_off, "frac_split_pab0_of_split_on": fr_on,
        "on_gt_off": larger,
        "reading": "record-skew mechanism supported" if larger
                   else "record-skew mechanism explanation is wrong" if larger is False
                   else "UNDEFINED (NaN input)",
    }

json.dump(result, open(os.path.join(out, "P4_LABEL.json"), "w"), indent=2, sort_keys=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "P4_DONE_${SLURM_JOB_ID}"
