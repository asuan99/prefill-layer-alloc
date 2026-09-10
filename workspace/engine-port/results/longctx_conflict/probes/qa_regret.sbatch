#!/bin/bash
#SBATCH --job-name=qaregret
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# Q-A REGRET -- PREREG_QA_REGRET_REV7_2026-09-10.md (submission revision).
# Format authority copied from probes/p7_itl_sigma.sbatch (model, backend,
# server flags, sync_engine_tree.sh, wait_health, 2-consecutive-boot-failure
# abort, telemetry env). NOTHING outside the registered grid/seeds/N-rule/
# rate/--random-range-ratio/max-running-requests/model/backend/config is
# changed relative to that template.
#
# PURE MEASUREMENT HARNESS. Estimand = paired difference matrix (rev5 sec1);
# min/argmin/envelope/"which arm is best" are NOT computed here (stake #1,
# rev5-rev7 sec "제출 후 금지"). Analysis lives in a separate qa_analyze.py
# (out of scope for this script).
#
# ---- registered grid x seed table (rev5 sec3, hardcoded, do not derive) ----
#   arm d16: (0.09,70)(0.09,67)(0.15,61)(0.15,68)(0.25,63)(0.42,71)(0.70,61)  7 cells
#   arm d44: (0.09,70)(0.09,67)(0.15,61)(0.15,68)(0.25,63)(0.42,71)          6 cells
#   arm d54: (0.09,70)(0.09,67)(0.15,61)(0.15,68)(0.25,63)(0.42,71)          6 cells
#   arm d92: (0.09,70)(0.09,67)(0.1162,70)(0.15,61)(0.15,68)                5 cells
#   N(lambda) = round(lambda*150)+1: 0.09->15 0.1162->18 0.15->23 0.25->39
#               0.42->64 0.70->106 (registered values, not recomputed here)
#   shapes: A = in8192/out96, B = in8192/out384. Both shapes run for EVERY
#   cell (A(lambda,B) := A(lambda,A), rev5 sec4) -> 24 cells x 2 shapes = 48
#   main benches per round. Saturation: 1x shape-B per boot (4) + 1x shape-A
#   for the d92 boot only (1) = 5 per round. 4 rounds -> main 192, sat 20.
#
# ---- round structure ----
#   arm boot order:  odd round  "16 44 54 92" | even round "92 54 44 16"
#   shape run order (applied uniformly to every cell in that round's boots):
#                    odd round  A,B          | even round B,A
#   rung order inside a boot: ASCENDING lambda, exactly the table order above
#   (never reversed -- only arm-boot-order and shape-order alternate by
#   round parity; this is a literal reading of the prereg text "rung
#   오름차순으로 돌되" with no stated exception for even rounds).
#
# ---- saturation seed ambiguity resolution (LOG THIS) ----
#   Task text allows SAT_SEED in {70, 61} "depending on round order" and
#   asks the seed actually used to be logged. Under the ascending-rung-
#   always rule above, the FIRST cell processed in every arm's boot, in
#   every round, is (lambda=0.09, seed=70) -- 0.09 is every arm's lowest
#   registered rung and 70 is listed first among its two seeds (rev5 sec3).
#   There is no boot/round in this design whose first processed cell is
#   seed 61. We therefore fix SAT_SEED=70 for ALL boots/rounds and print
#   this resolution at job start and in every boot's meta file so it is
#   auditable. See report deviation note (b) at submission time.
#
# ---- self-check ----
#   static_selfcheck() below recomputes 48/5 per round and 192/20 total
#   PURELY from the tables (no execution) and exits 2 immediately if the
#   arithmetic does not match -- runs before module load / GPU touch in
#   both DRYRUN and real mode.
#
# ---- dry run ----
#   PDMUX_DRYRUN=1 bash qa_regret.sbatch [ROUND]   # prints planned commands
#   only (no sbatch/module/GPU needed); ROUND omitted -> previews all 4.
#
# ---- submission (4 separate jobs, one per round; not an array) ----
#   for R in 1 2 3 4; do
#     sbatch --job-name=qaregret_r$R probes/qa_regret.sbatch $R
#   done
#
# usage: sbatch --job-name=qaregret_rN qa_regret.sbatch <ROUND in 1..4>

# =========================================================================
# ---- registered tables (shared by DRYRUN preview and real execution) ----
# =========================================================================
declare -A ROUND_ARMS=( [1]="16 44 54 92" [2]="92 54 44 16" [3]="16 44 54 92" [4]="92 54 44 16" )
declare -A ROUND_SHAPES=( [1]="A B" [2]="B A" [3]="A B" [4]="B A" )
declare -A ARM_CELLS=(
  [16]="0.09:70 0.09:67 0.15:61 0.15:68 0.25:63 0.42:71 0.70:61"
  [44]="0.09:70 0.09:67 0.15:61 0.15:68 0.25:63 0.42:71"
  [54]="0.09:70 0.09:67 0.15:61 0.15:68 0.25:63 0.42:71"
  [92]="0.09:70 0.09:67 0.1162:70 0.15:61 0.15:68"
)
declare -A NMAP=( [0.09]=15 [0.1162]=18 [0.15]=23 [0.25]=39 [0.42]=64 [0.70]=106 )
declare -A MUP_REF=( [16]=0.8962 [44]=0.6632 [54]=0.5736 [92]=0.1885 )
declare -A RATE_SAT=( [16]=1.434 [44]=1.061 [54]=0.918 [92]=0.302 )
declare -A SHAPE_OUTLEN=( [A]=96 [B]=384 )
SAT_SEED=70   # see "saturation seed ambiguity resolution" comment above

EXPECT_MAIN_PER_ROUND=48
EXPECT_SAT_PER_ROUND=5
EXPECT_MAIN_TOTAL=192
EXPECT_SAT_TOTAL=20

# =========================================================================
# ---- static self-check: pure arithmetic on the tables, no execution ----
# =========================================================================
static_selfcheck () {
  local ok=1
  local main_per_round=0
  local D n_cells
  for D in 16 44 54 92; do
    n_cells=$(wc -w <<< "${ARM_CELLS[$D]}")
    main_per_round=$((main_per_round + n_cells * 2))
  done
  local sat_per_round=5   # 4 boots x 1 (shape B) + 1 extra (d92, shape A)
  local main_total=$((main_per_round * 4))
  local sat_total=$((sat_per_round * 4))

  echo "SELFCHECK main_per_round=$main_per_round (expect $EXPECT_MAIN_PER_ROUND)"
  echo "SELFCHECK sat_per_round=$sat_per_round (expect $EXPECT_SAT_PER_ROUND)"
  echo "SELFCHECK main_total=$main_total (expect $EXPECT_MAIN_TOTAL)"
  echo "SELFCHECK sat_total=$sat_total (expect $EXPECT_SAT_TOTAL)"

  [ "$main_per_round" -eq "$EXPECT_MAIN_PER_ROUND" ] || ok=0
  [ "$sat_per_round" -eq "$EXPECT_SAT_PER_ROUND" ] || ok=0
  [ "$main_total" -eq "$EXPECT_MAIN_TOTAL" ] || ok=0
  [ "$sat_total" -eq "$EXPECT_SAT_TOTAL" ] || ok=0

  if [ "$ok" -ne 1 ]; then
    echo "SELFCHECK_FAILED -- cell-count arithmetic does not match the" \
         "registered budget (main 192 / sat 20). Aborting before any GPU" \
         "or module action." >&2
    return 1
  fi
  echo "SELFCHECK_PASSED"
  return 0
}

static_selfcheck || exit 2

# =========================================================================
# ---- dry run: preview only, no SLURM/module/GPU required ----
# =========================================================================
if [ "${PDMUX_DRYRUN:-0}" = "1" ]; then
  DRY_ROUNDS="${1:-1 2 3 4}"
  echo "=== PDMUX_DRYRUN=1 preview (no execution) ==="
  total_main=0
  total_sat=0
  for R in $DRY_ROUNDS; do
    echo ""
    echo "---- ROUND $R  arm_order=[${ROUND_ARMS[$R]}]  shape_order=[${ROUND_SHAPES[$R]}] ----"
    read -r SHAPE1 SHAPE2 <<< "${ROUND_SHAPES[$R]}"
    round_main=0
    round_sat=0
    for D in ${ROUND_ARMS[$R]}; do
      echo "  BOOT arm=d$D  PDMUX_R2_FIXED_DSM=$D  PDMUX_R2_POLICY=fixed  cfg=pdmux_homog6.yml  ctx=8192 context-length=8704"
      for CELL in ${ARM_CELLS[$D]}; do
        LAMBDA="${CELL%%:*}"; SEED="${CELL##*:}"; N="${NMAP[$LAMBDA]}"
        for SHAPE in "$SHAPE1" "$SHAPE2"; do
          OUTLEN="${SHAPE_OUTLEN[$SHAPE]}"
          TAG="r${R}_d${D}_s${SHAPE}_l${LAMBDA}_seed${SEED}"
          echo "    BENCH $TAG  in=8192 out=$OUTLEN N=$N rate=$LAMBDA seed=$SEED -> bench_${TAG}.jsonl"
          round_main=$((round_main + 1))
        done
      done
      SATSHAPES="B"; [ "$D" = "92" ] && SATSHAPES="B A"
      for SHAPE in $SATSHAPES; do
        OUTLEN="${SHAPE_OUTLEN[$SHAPE]}"
        echo "    SAT   r${R}_d${D}_s${SHAPE}  in=8192 out=$OUTLEN N=45 rate=${RATE_SAT[$D]} seed=$SAT_SEED -> sat_r${R}_d${D}_s${SHAPE}.jsonl"
        round_sat=$((round_sat + 1))
      done
    done
    echo "  ROUND $R totals: main=$round_main sat=$round_sat"
    total_main=$((total_main + round_main))
    total_sat=$((total_sat + round_sat))
  done
  echo ""
  echo "=== DRYRUN grand totals shown: main=$total_main sat=$total_sat" \
       "(campaign totals are 192/20 only when all 4 rounds are previewed) ==="
  exit 0
fi

# =========================================================================
# ---- real execution ----
# =========================================================================
set -uo pipefail

ROUND="${1:-${SLURM_ARRAY_TASK_ID:-}}"
case "${ROUND:-}" in
  1|2|3|4) ;;
  *) echo "ERROR: round number required as \$1 (or SLURM_ARRAY_TASK_ID), got '${ROUND:-}'. Usage: sbatch qa_regret.sbatch <1..4>" >&2; exit 2 ;;
esac

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
OUT="$PROBES/qa_r${ROUND}_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$PROBES/pdmux_homog6.yml"
CTX=8192
CTXCAP=8704   # CTX+512; shape B out=384 -> 8576 < 8704 (prereg literal)
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "QA_REGRET round=$ROUND model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX ctxcap=$CTXCAP job=$SLURM_JOB_ID"
echo "node=$(hostname) date=$(date -Is)"
echo "arm_order=[${ROUND_ARMS[$ROUND]}] shape_order=[${ROUND_SHAPES[$ROUND]}] SAT_SEED=$SAT_SEED"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm --format=csv 2>/dev/null | tee "$OUT/gpu_info.txt"
echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2
echo "=== pkg versions (registered output 7) ==="
PKGVERS=$(python3 -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__)")
echo "$PKGVERS" | tee "$OUT/pkg_versions.txt"

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

read -r SHAPE1 SHAPE2 <<< "${ROUND_SHAPES[$ROUND]}"
MAIN_COUNT=0
SAT_COUNT=0
FAIL_STREAK=0

for D in ${ROUND_ARMS[$ROUND]}; do
  PORT=$((38400 + (SLURM_JOB_ID + ROUND * 7 + D + RANDOM) % 400))
  SRVLOG="$OUT/srv_r${ROUND}_d${D}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_r${ROUND}_d${D}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="qa_r${ROUND}_d${D}_${SLURM_JOB_ID}"
  export PDMUX_WORKLOAD_ID="qa_r${ROUND}_d${D}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## QA round=$ROUND D=$D port=$PORT shape_order=[$SHAPE1,$SHAPE2] ##########"
  META="$OUT/meta_r${ROUND}_d${D}.txt"
  echo "boot_start=$(date -Is) node=$(hostname)" | tee "$META"
  echo "$PKGVERS" | tee -a "$META"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader 2>/dev/null \
    | tee -a "$META"

  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 round=$ROUND D=$D"
    FAIL_STREAK=0
    grep -m1 -oE "sm_counts[^\n]*" "$SRVLOG" | tee -a "$META"

    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len 96 --random-range-ratio 1.0 \
      --num-prompts 8 --max-concurrency 1 --seed 7 \
      > "$OUT/warmup_r${ROUND}_d${D}.log" 2>&1
    echo "warmup_rc=$?"

    # --- MAIN: registered grid x seed table, rung ASCENDING, shape ALTERNATING ---
    for CELL in ${ARM_CELLS[$D]}; do
      LAMBDA="${CELL%%:*}"; SEED="${CELL##*:}"; N="${NMAP[$LAMBDA]}"
      for SHAPE in "$SHAPE1" "$SHAPE2"; do
        OUTLEN="${SHAPE_OUTLEN[$SHAPE]}"
        TAG="r${ROUND}_d${D}_s${SHAPE}_l${LAMBDA}_seed${SEED}"
        python -m sglang.bench_serving --backend sglang --model "$MODEL" \
          --host 127.0.0.1 --port "$PORT" \
          --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
          --random-input-len "$CTX" --random-output-len "$OUTLEN" --random-range-ratio 1.0 \
          --num-prompts "$N" --request-rate "$LAMBDA" --seed "$SEED" \
          --output-details --output-file "$OUT/bench_${TAG}.jsonl" \
          > "$OUT/bench_${TAG}.log" 2>&1
        echo "main_rc=$? $TAG"
        grep -E "Request throughput|Median TTFT|Median ITL" "$OUT/bench_${TAG}.log" | sed "s/^/  $TAG /"
        MAIN_COUNT=$((MAIN_COUNT + 1))
      done
    done

    # --- SAT: shape B for every boot; shape A additionally for the d92 boot ---
    SATSHAPES="B"; [ "$D" = "92" ] && SATSHAPES="B A"
    for SHAPE in $SATSHAPES; do
      OUTLEN="${SHAPE_OUTLEN[$SHAPE]}"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
        --random-input-len "$CTX" --random-output-len "$OUTLEN" --random-range-ratio 1.0 \
        --num-prompts 45 --request-rate "${RATE_SAT[$D]}" --seed "$SAT_SEED" \
        --output-details --output-file "$OUT/sat_r${ROUND}_d${D}_s${SHAPE}.jsonl" \
        > "$OUT/sat_r${ROUND}_d${D}_s${SHAPE}.log" 2>&1
      echo "sat_rc=$? r${ROUND}_d${D}_s${SHAPE}"
      grep -E "Request throughput" "$OUT/sat_r${ROUND}_d${D}_s${SHAPE}.log" | sed "s/^/  sat r${ROUND}_d${D}_s${SHAPE} /"
      SAT_COUNT=$((SAT_COUNT + 1))
    done
  else
    echo "BOOT_FAILED round=$ROUND D=$D" | tee -a "$OUT/BOOT_FAILURES.txt"
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
echo "=== QA round=$ROUND runtime totals: main=$MAIN_COUNT (expect $EXPECT_MAIN_PER_ROUND) sat=$SAT_COUNT (expect $EXPECT_SAT_PER_ROUND) ==="
if [ "$MAIN_COUNT" -ne "$EXPECT_MAIN_PER_ROUND" ] || [ "$SAT_COUNT" -ne "$EXPECT_SAT_PER_ROUND" ]; then
  echo "RUNTIME_COUNT_MISMATCH (likely a boot-failure abort mid-round -- check BOOT_FAILURES.txt)" | tee -a "$OUT/BOOT_FAILURES.txt"
fi

echo "QA_REGRET_DONE round=$ROUND job=${SLURM_JOB_ID}"
