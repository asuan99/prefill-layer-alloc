#!/bin/bash
#SBATCH --job-name=ccap
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# PROBE C -- CAPACITY.  PREREG_CAPACITY_2026-09-09.md.
#
# Buys the three things the R~=0.3-0.45 static-split sweep cannot be designed
# without, and that RESULT_CAPACITY_PRECOMP_2026-09-09.md could NOT settle on
# paper:
#   (1) mu_p(d16) -- the d16 arm has never been measured on this workload;
#       0.94 req/s is a two-point EXTRAPOLATION, not a measurement.
#   (2) the knee location/shape per arm.  P1 ran at 49x and 394x floor_ref --
#       far past the knee -- so "rate above the knee" is currently unbounded
#       above and methodology gate #6 (metric cliff) is unaddressed.
#   (3) whether the closed-form decode-population ceiling
#         L_decode = [(1-R)/R] * (itl_load/itl_solo) / s(D)
#       (validated to 1.4% / 0.5% on the two existing P1 arms, zero GPU)
#       survives a change of `out`, i.e. whether R really is what caps the
#       decode population.
#
# 12 boots = 3 arms {16,44,92} x 3 rates (out=96) + 3 arms x 1 rate (out=384).
# ONE boot per cell, ONE telemetry file per boot -- deliberately NOT a rate
# ladder inside a single boot, because slicing one telemetry file by time
# window would reintroduce exactly the interval-attribution machinery this
# track keeps dying on (CONSENSUS sec 3 item 120).
#
# All arms use the HOMOGENIZED config pdmux_homog5.yml (P3 job 905701:
# all three targets boot, banner identical, cost 0.051%).  This differs from
# P1, which used per-arm pdmux_d44/d92.yml -- registered in prereg sec 2.3 as
# a known difference: the anchors mu_p(d44)=0.663 and mu_p(d92)=0.185 come
# from the per-arm configs, so the rate ladder derived from them is
# provisional.  The knee rule is arm-relative and existence-based, so a
# moderate shift does not invalidate it; a large shift shows up as
# KNEE_NOT_BRACKETED / UNRESOLVED and is itself the deliverable.
#
# NO policy verdict.  NO arm comparison: the rate ladder is arm-specific by
# construction (each arm is probed relative to its OWN predicted capacity), so
# cells across arms are NOT matched and must not be compared.
#
# usage: sbatch c_capacity.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
OUT="$PROBES/c_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$PROBES/pdmux_homog5.yml"
CTX=8192
CTXCAP=$((CTX + 512))
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED=41

# cell spec: "name D rate outtok numprompts"
# rates are rho = rate/mu_p_predicted in {0.60, 0.90, 1.30}; mu_p predicted as
#   d16 0.94 (EXTRAPOLATED)  d44 0.663 (measured P1)  d92 0.185 (measured P1)
# rho=1.30 is chosen so the saturated side lands at ach/off ~= 1/1.30 = 0.77,
# comfortably under the registered 0.90 -- prereg sec 3 attainability check.
CELLS=(
  "d16_r0_o96  16 0.56  96  110"
  "d16_r1_o96  16 0.85  96  170"
  "d16_r2_o96  16 1.22  96  170"
  "d44_r0_o96  44 0.40  96   80"
  "d44_r1_o96  44 0.60  96  120"
  "d44_r2_o96  44 0.86  96  130"
  "d92_r0_o96  92 0.111 96   40"
  "d92_r1_o96  92 0.167 96   40"
  "d92_r2_o96  92 0.241 96   45"
  "d16_r1_o384 16 0.85 384  130"
  "d44_r1_o384 44 0.60 384  100"
  "d92_r1_o384 92 0.167 384  40"
)

echo "PROBE C model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX job=$SLURM_JOB_ID"
echo "n_cells=${#CELLS[@]}"
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

FAIL_STREAK=0
for SPEC in "${CELLS[@]}"; do
  read -r NAME D RATE OUTTOK NP <<< "$SPEC"
  PORT=$((39100 + (SLURM_JOB_ID + D + RANDOM) % 400))
  SRVLOG="$OUT/srv_${NAME}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_${NAME}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="c_${NAME}_${SLURM_JOB_ID}"
  export PDMUX_WORKLOAD_ID="c_${NAME}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## CELL $NAME  D=$D rate=$RATE out=$OUTTOK np=$NP port=$PORT ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 cell=$NAME"
    FAIL_STREAK=0

    # warmup: 8 sequential requests, discarded.  Keeps triton/JIT compilation
    # out of the measured TTFTs -- which matters here because the knee is read
    # off TTFT divergence.
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts 8 --max-concurrency 1 --seed 7 \
      > "$OUT/warmup_${NAME}.log" 2>&1
    echo "warmup_rc=$?"

    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
      --output-details --output-file "$OUT/bench_${NAME}.jsonl" \
      > "$OUT/bench_${NAME}.log" 2>&1
    echo "bench_rc=$? cell=$NAME"
    grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median ITL|Concurrency:" \
      "$OUT/bench_${NAME}.log"

    python3 "$PROBES/c_capacity_analyze.py" "$OUT/bench_${NAME}.jsonl" \
      --telemetry "$PDMUX_TELEMETRY_PATH" --expect-d "$D" \
      --label "$NAME" --out "$OUT/cell_${NAME}.json" \
      > "$OUT/analyze_${NAME}.log" 2>&1 || echo "ANALYZE_FAILED cell=$NAME"
    python3 -c "
import json,sys
d=json.load(open('$OUT/cell_${NAME}.json'))
print('  ach/off=%.3f  ach=%.4f req/s  TTFTp50=%.2fs (%.1fx floor)  L_dec=%.3f  L_pre=%.2f  obs/pred=%.3f  xerr=%.1e'
      % (d['achieved_over_offered'], d['achieved_rate'], d['ttft_p50_s'],
         d['ttft_p50_over_floor_ref'], d['L_decode_exact'], d['L_pre_exact'],
         d['L_decode_obs_over_pred'], d['crosscheck_rel_err']))
" 2>/dev/null || true

    # telemetry is large; keep it but note the size in the log
    du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"
    break
  fi
done

echo ""
echo "=== PROBE C registered rules (PREREG_CAPACITY sec 4) ==="
python3 "$PROBES/c_capacity_label.py" "$OUT" || echo "LABEL_FAILED"

echo "C_DONE_${SLURM_JOB_ID}"
