#!/bin/bash
#SBATCH --job-name=p7sig
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# P7 ITL_SIGMA -- PREREG_P7_ITL_SIGMA_2026-09-09.md.
#
# PURE INSTRUMENTATION.  No pre-registered prediction, no decision rule, no
# threshold except "did it boot".  By construction there is nothing here for a
# rule-layer audit to judge -- that is the point (P6 died because its two
# predictions were the two premises its own prescriber had already registered).
#
# Buys the one number this project has never had: the BETWEEN-BOOT standard
# deviation of mu_p, itl95_cell and L_decode_exact.  Every prior measurement in
# this track, probe C included, is n_boot = 1, and RESULT_RULEPOWER sec 7
# forbids importing that variance from another campaign (gate #32).
#
# Design constants are NOT free -- they were produced by the registered rule
# W := 0.85 * min_D mu_p(D) applied to Step 1 (job 905958) measurements:
#   mu_p = 0.9143 / 0.6666 / 0.5772  ->  W = 0.491,  rho = 0.537/0.737/0.851
#   N = 90 for every arm -> arrival window 183.3 s IDENTICAL across arms
#     (death cause Y5 / X5-c: match the ARRIVAL window, not the bench length)
#
# 4 rounds x 3 arms = 12 boots.  Arms round-robin INSIDE each round with the
# order REVERSED on alternate rounds (cancels first-order drift; probe C ran
# arm-blocked and confounded arm with in-job time drift -- blocker B6).
# Seeds are REDRAWN per round and SHARED across arms within a round, so
# within-boot spread is arrival-realisation variance (random) while
# between-boot spread is the boot random effect -- the decomposition that a
# fixed shared seed set structurally cannot give (blocker C4).
#
# Each boot: warmup(8, conc 1) -> 3 MAIN benches (one per round seed)
#            -> 1 SAT bench (prereg sec 3.1 amendment) for mu_p only.
# ★SAT benches are deliberately overloaded and their TTFT/ITL are NOT
# operating-point values; the analyser keeps them in a separate `sat` key.
#
# usage: sbatch p7_itl_sigma.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
OUT="$PROBES/p7_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$PROBES/pdmux_homog6.yml"
CTX=8192
CTXCAP=$((CTX + 512))
OUTTOK=96
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

W=0.491          # registered rule output, prereg sec 1.2
NMAIN=90         # identical across arms -> identical arrival window
NSAT=45
declare -A RATE_SAT=( [16]=1.463 [44]=1.067 [54]=0.924 )   # 1.6 x mu_p (Step 1)

# round -> arm order (reversed on even rounds) and that round's seed set
declare -A ROUND_ARMS=( [1]="16 44 54" [2]="54 44 16" [3]="16 44 54" [4]="54 44 16" )
declare -A ROUND_SEEDS=( [1]="41 42 43" [2]="44 45 46" [3]="47 48 49" [4]="50 51 52" )

echo "P7 model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX out=$OUTTOK W=$W N=$NMAIN job=$SLURM_JOB_ID"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm --format=csv 2>/dev/null | tee "$OUT/gpu_info.txt"
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
for RND in 1 2 3 4; do
  for D in ${ROUND_ARMS[$RND]}; do
    PORT=$((38400 + (SLURM_JOB_ID + RND * 7 + D + RANDOM) % 400))
    SRVLOG="$OUT/srv_r${RND}_d${D}.log"
    export PDMUX_TELEMETRY_PATH="$OUT/tel_r${RND}_d${D}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="p7_r${RND}_d${D}_${SLURM_JOB_ID}"
    export PDMUX_WORKLOAD_ID="p7_r${RND}_d${D}"
    export PDMUX_R2_POLICY=fixed
    export PDMUX_R2_FIXED_DSM=$D

    echo ""
    echo "########## P7 round=$RND D=$D port=$PORT  seeds=[${ROUND_SEEDS[$RND]}] ##########"
    echo "boot_start=$(date -Is) node=$(hostname)" | tee "$OUT/meta_r${RND}_d${D}.txt"
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader 2>/dev/null \
      | tee -a "$OUT/meta_r${RND}_d${D}.txt"

    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend "$BACKEND" \
      --enable-pdmux --pdmux-config-path "$CFG" \
      --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
      --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
      --context-length "$CTXCAP" \
      --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
    PID=$!

    if wait_health "$PORT" "$PID"; then
      echo "boot_ok=1 round=$RND D=$D"
      FAIL_STREAK=0
      grep -m1 -oE "sm_counts[^\n]*" "$SRVLOG" | tee -a "$OUT/meta_r${RND}_d${D}.txt"

      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
        --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
        --num-prompts 8 --max-concurrency 1 --seed 7 \
        > "$OUT/warmup_r${RND}_d${D}.log" 2>&1
      echo "warmup_rc=$?"

      # --- MAIN: 3 benches, one per seed of this round (common W) ---
      for S in ${ROUND_SEEDS[$RND]}; do
        TAG="r${RND}_d${D}_s${S}"
        python -m sglang.bench_serving --backend sglang --model "$MODEL" \
          --host 127.0.0.1 --port "$PORT" \
          --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
          --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
          --num-prompts "$NMAIN" --request-rate "$W" --seed "$S" \
          --output-details --output-file "$OUT/bench_${TAG}.jsonl" \
          > "$OUT/bench_${TAG}.log" 2>&1
        echo "main_rc=$? $TAG"
        grep -E "Request throughput|Median TTFT|Median ITL" "$OUT/bench_${TAG}.log" | sed "s/^/  $TAG /"
      done

      # --- SAT: 1 bench for mu_p only (prereg sec 3.1). NOT an operating point.
      SSAT=$(echo ${ROUND_SEEDS[$RND]} | awk '{print $1}')
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
        --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
        --num-prompts "$NSAT" --request-rate "${RATE_SAT[$D]}" --seed "$SSAT" \
        --output-details --output-file "$OUT/sat_r${RND}_d${D}.jsonl" \
        > "$OUT/sat_r${RND}_d${D}.log" 2>&1
      echo "sat_rc=$? r${RND}_d${D}"
      grep -E "Request throughput" "$OUT/sat_r${RND}_d${D}.log" | sed "s/^/  sat r${RND}_d${D} /"
    else
      echo "BOOT_FAILED round=$RND D=$D" | tee -a "$OUT/BOOT_FAILURES.txt"
      tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
      FAIL_STREAK=$((FAIL_STREAK + 1))
    fi
    kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

    if [ "$FAIL_STREAK" -ge 2 ]; then
      echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"
      break 2
    fi
  done
done

echo ""
echo "=== P7 analysis (measurement only, no verdict) ==="
python3 "$PROBES/p7_analyze.py" "$OUT" --out "$OUT/P7_RESULT.json" \
  > "$OUT/p7_analyze.log" 2>&1 || echo "ANALYZE_FAILED"
python3 -c "
import json
d=json.load(open('$OUT/P7_RESULT.json'))
for a in (16,44,54):
    v=d['variance'][str(a)] if str(a) in d['variance'] else d['variance'][a]
    i=v['itl95_median_ms']; m=v['mu_p_saturated']
    print(f\"  d{a}: itl95 mean={i.get('grand_mean',float('nan')):.2f}ms  sigma_boot={i.get('sigma_boot_rel_pct')}%  sigma_seed={i.get('sigma_seed_rel_pct')}%\")
    print(f\"        mu_p mean={m.get('mean')}  rel_sd={m.get('rel_sd_pct')}%\")
print('  TTFT dominance violation frac:', d['ttft_dominance'].get('violation_frac'))
print('  crosscheck_rel_err_max:', d['crosscheck_rel_err_max'], ' main benches:', d['n_main_benches'])
print('  missing:', d['missing'])
" 2>/dev/null || true

echo "P7_DONE_${SLURM_JOB_ID}"
