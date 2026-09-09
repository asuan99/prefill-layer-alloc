#!/bin/bash
#SBATCH --job-name=p1split
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
# P1 SPLIT_FIRES -- PREREG_PROBES_2026-09-08.md sec 1 (★MOST IMPORTANT probe).
# "Does the SM split actually fire for Nemotron + flashinfer at CONCURRENCY >
# 1?"  Zero repo evidence either way: AF-1 was concurrency 1 (L-10 => the
# split group is structurally unreachable there), W1 did not record partition
# telemetry.  --request-rate 1.0 (Poisson arrivals, no --max-concurrency cap)
# is the mechanism that produces concurrency > 1 here.
#
# 2 boots: d44 (existing stage0_xctrl/pdmux_d44.yml, target 44) and d92
# (pdmux_d92.yml, target 92).  ctx 8192, out 96, rate 1.0, 160 requests each.
#
# Labels (prereg sec 1, applied programmatically below from the canonical
# `compute_decode_realized` output -- NO size threshold, EXISTENCE only,
# per the prereg's explicit registered-against-a-threshold failure lesson):
#   SPLIT_ABSENT      either arm has 0 decode-active snapshots at its own
#                      target D  -> "RATIO family wholesale absent" result
#   SPLIT_FIRES        both arms have >0 AND each arm's decode_hist mode
#                      equals its own target (44 / 92)
#   SPLIT_UNRESOLVED    boot failure / telemetry missing -> 1 retry
#
# usage: sbatch p1_split_fires.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
STAGE0=$ROOT/workspace/engine-port/results/stage0_xctrl
OUT="$PROBES/p1_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CTX=8192
OUTTOK=96
CTXCAP=$((CTX + 512))
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED=17
RATE=1.0
NP=160

echo "P1 model=$MODEL backend=$BACKEND ctx=$CTX out=$OUTTOK rate=$RATE np=$NP job=$SLURM_JOB_ID"
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

declare -A CFG=( [44]="$STAGE0/pdmux_d44.yml" [92]="$STAGE0/pdmux_d92.yml" )
FAIL_STREAK=0
declare -A BOOTED

for D in 44 92; do
  PORT=$((39500 + (SLURM_JOB_ID + D + RANDOM) % 400))
  SRVLOG="$OUT/srv_d${D}.log"
  RUNID="p1_d${D}_${SLURM_JOB_ID}"
  export PDMUX_TELEMETRY_PATH="$OUT/p1_d${D}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"
  export PDMUX_WORKLOAD_ID="p1_d${D}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## P1 D=$D port=$PORT cfg=${CFG[$D]} ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "${CFG[$D]}" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 D=$D"
    FAIL_STREAK=0
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
      --output-details --output-file "$OUT/bench_d${D}.jsonl" \
      > "$OUT/bench_d${D}.log" 2>&1
    BRC=$?
    echo "bench_rc=$BRC D=$D"
    grep -E "Successful requests|Median TTFT|Median ITL|Concurrency:" "$OUT/bench_d${D}.log"
    BOOTED[$D]=1
    python3 "$PROBES/analyze_probes.py" "$PDMUX_TELEMETRY_PATH" "$D" \
      --label "p1_d${D}" --out "$OUT/summary_d${D}.json" || echo "ANALYZE_FAILED D=$D"
  else
    echo "BOOT_FAILED D=$D" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES (순서·중단 규칙 #3)" | tee -a "$OUT/BOOT_FAILURES.txt"
    break
  fi
done

echo "=== P1 label ==="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
labels = {}
missing = []
for d in (44, 92):
    p = os.path.join(out, f"summary_d{d}.json")
    if not os.path.exists(p):
        missing.append(d)
        continue
    labels[d] = json.load(open(p))

if missing:
    verdict = "SPLIT_UNRESOLVED"
    reason = f"missing summaries for D={missing} (boot failure or telemetry loss)"
else:
    n44 = labels[44]["n_decode_active_at_target"]
    n92 = labels[92]["n_decode_active_at_target"]
    mode44 = labels[44]["decode_hist_mode"]
    mode92 = labels[92]["decode_hist_mode"]
    if n44 == 0 or n92 == 0:
        verdict = "SPLIT_ABSENT"
        reason = f"n_decode_active_at_target: d44={n44} d92={n92} (>=1 arm is 0)"
    elif n44 > 0 and n92 > 0 and mode44 == 44 and mode92 == 92:
        verdict = "SPLIT_FIRES"
        reason = f"n_decode_active_at_target: d44={n44} d92={n92}; decode_hist_mode: d44={mode44} d92={mode92}"
    else:
        verdict = "SPLIT_UNRESOLVED"
        reason = (f"both arms have split snapshots (d44={n44}, d92={n92}) but mode "
                   f"mismatch (d44 mode={mode44} expect 44; d92 mode={mode92} expect 92)")

result = {"probe": "P1_SPLIT_FIRES", "verdict": verdict, "reason": reason,
          "f_time": {d: labels[d]["f_time"] for d in labels},
          "f_count": {d: labels[d]["f_count"] for d in labels},
          "decode_hist": {d: labels[d]["decode_hist"] for d in labels}}
json.dump(result, open(os.path.join(out, "P1_LABEL.json"), "w"), indent=2, sort_keys=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "P1_DONE_${SLURM_JOB_ID}"
