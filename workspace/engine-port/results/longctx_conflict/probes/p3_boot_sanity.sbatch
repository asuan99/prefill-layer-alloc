#!/bin/bash
#SBATCH --job-name=p3boot
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:50:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# P3 BOOT_SANITY -- PREREG_PROBES_2026-09-08.md sec 3.  "Is the sm_group_num:5
# / 3-division homogenized config (pdmux_homog5.yml) actually bootable, for
# all THREE registered PDMUX_R2_FIXED_DSM targets (16/44/92)?"  Up to now this
# was verified by code reading only (decode_states={16,44} guard passes;
# _r2_decide_idx matches target 92 to sm_counts idx3=(16,92) by exact
# decode_sms equality, independent of decode_states membership).
#
# ★MUST RUN FIRST (prereg sec "순서·중단 규칙" #1): if the homogenized config
# does not boot, the rest of the probe set's design premise is void.
# ★abort rule (sec "순서·중단 규칙" #3): 2 consecutive boot failures -> stop.
#
# Labels (prereg sec 3):
#   BOOT_OK   3/3 boots + capture completed + sm_counts banner shows 5 rows
#   BOOT_FAIL any boot fails -> 3F3 repair invalid, redesign required
#
# usage: sbatch p3_boot_sanity.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
OUT="$PROBES/p3_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$PROBES/pdmux_homog5.yml"
CTX=16384
CTXCAP=$((CTX + 512))
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "P3 model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX ctxcap=$CTXCAP job=$SLURM_JOB_ID"
echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

wait_health () {   # $1=port $2=pid
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

FAIL_STREAK=0
BOOT_OK_COUNT=0
for D in 16 44 92; do
  PORT=$((39000 + (SLURM_JOB_ID + D + RANDOM) % 400))
  SRVLOG="$OUT/srv_d${D}.log"
  RUNID="p3_d${D}_${SLURM_JOB_ID}"
  export PDMUX_TELEMETRY_PATH="$OUT/p3_d${D}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"
  export PDMUX_WORKLOAD_ID="p3_d${D}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## P3 D=$D port=$PORT ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 D=$D"
    BOOT_OK_COUNT=$((BOOT_OK_COUNT + 1))
    FAIL_STREAK=0
    BANNER="$OUT/banner_d${D}.txt"; : > "$BANNER"
    echo "--- PD-mux enable banner + sm_counts ---" | tee -a "$BANNER"
    grep -oE "PD-Multiplexing enabled with .*" "$SRVLOG" | sort -u | tee -a "$BANNER"
    echo "--- max_total_num_tokens ---" | tee -a "$BANNER"
    grep -oE "max_total_num_tokens=[0-9]+.*" "$SRVLOG" | head -1 | tee -a "$BANNER"
    echo "--- mamba (grep -i, best-effort: no single dedicated mamba-pool-size log line exists upstream) ---" | tee -a "$BANNER"
    grep -i "mamba" "$SRVLOG" | sort -u | sed 's/^/MAMBA /' | tee -a "$BANNER"
    echo "--- cudagraph capture list ---" | tee -a "$BANNER"
    grep -oE "Capture cuda graph bs \[[^]]*\]" "$SRVLOG" | sort -u | tee -a "$BANNER"
    grep -c "Capturing batches" "$SRVLOG" | sed 's/^/N_CAPTURING_BATCHES_LOG_LINES /' | tee -a "$BANNER"
    grep -c "Capture cuda graph failed" "$SRVLOG" | sed 's/^/N_CAPTURE_FAILURES /' | tee -a "$BANNER"

    echo "--- smoke: 8 requests (in=512 out=32, plumbing only) ---" | tee -a "$BANNER"
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len 512 --random-output-len 32 --random-range-ratio 1.0 \
      --num-prompts 8 --seed 7 \
      --output-details --output-file "$OUT/smoke_d${D}.jsonl" \
      > "$OUT/bench_d${D}.log" 2>&1
    BRC=$?
    echo "smoke_rc=$BRC D=$D" | tee -a "$BANNER"
    grep -E "Successful requests|Median TTFT|Median ITL" "$OUT/bench_d${D}.log" | tee -a "$BANNER"
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

echo "P3_BOOT_OK_COUNT=$BOOT_OK_COUNT/3"
if [ "$BOOT_OK_COUNT" -eq 3 ]; then
  echo "P3_LABEL=BOOT_OK"
else
  echo "P3_LABEL=BOOT_FAIL"
fi
echo "P3_DONE_${SLURM_JOB_ID}"
