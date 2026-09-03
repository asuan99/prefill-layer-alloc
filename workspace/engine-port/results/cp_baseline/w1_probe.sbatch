#!/bin/bash
#SBATCH --job-name=w1probe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.err
#
# W1 -- the long-prompt tail's RATE GRID and DISTRIBUTION POSITIONS.
# PREREG_W1_2026-09-02.md.  Measurement stage: no policy verdict, no arm ranking,
# no goodput comparison.  Buys where the knee is and where the TTFT/ITL modes sit
# on a workload where chunked prefill ACTUALLY FIRES (census: 200/200 prompts
# exceed cps 2048, against 0/200 exceeding cps 8192 on the canonical trace).
#
# The workload swap is a DATA swap: same sampler, same harness, different
# --dataset-path (built by build_sharegpt_long.py).  No engine or harness patch.
#
# ★ --disable-piecewise-cuda-graph is a COMMON flag on EVERY arm
# (DECISION_PIECEWISE_OFF_2026-09-02.md, user instruction): piecewise capture
# crashes on falcon_h1 / nemotron_h in this environment, so the optimisation is
# unavailable for the long-context models and is switched off everywhere for
# comparability.  It is in the common flags, not the per-arm branch, because a
# per-arm branch is how the next arm loses it (gate #66).  Consequence: `cps` is
# a SINGLE lever again (prefill budget only), which closes the F3 / G4 confound.
# Cost: the cp family loses a prefill optimisation it could otherwise use -- every
# result must carry that scope (decision doc sec 4).
#
# usage:  sbatch w1_probe.sbatch
#         sbatch --export=ALL,W1_ARMS="cp2048",W1_RATES="0.5" --time=00:30:00 w1_probe.sbatch  # smoke

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base}"
CTX="${CTX:-32768}"
BACKEND="${BACKEND:-flashinfer}"
ARMS="${W1_ARMS:-cp512 cp2048 cp8192 d44}"
RATES="${W1_RATES:-0.25 0.5 1.0 2.0}"
SEED="${W1_SEED:-11}"

export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# Two data tracks (user decision 2026-09-02, option C):
#   sharegpt -> the REAL long-prompt tail (primary).  Supply caps the usable floor
#               at 2048 (census: only 290 prompts corpus-wide exceed 4096), so this
#               track can exercise cps <= 2048 and nothing above it.
#   random   -> SYNTHETIC, input length is a design variable, so cps 8192 (the
#               engine default) can be made to fire.  Not a real trace; secondary.
DSMODE="${W1_DSMODE:-sharegpt}"
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ "$DSMODE" = "random" ]; then
  DATASET="$SGPT_RAW"                       # random sampler draws its text from here
  INLEN="${W1_INLEN:-16000}"
  DSARGS="--dataset-name random --dataset-path $DATASET --random-input-len $INLEN --random-output-len ${W1_OUTLEN:-300} --random-range-ratio 1.0"
else
  DATASET="${W1_DATASET:-$HF_HOME/raw/ShareGPT_long2048_cap8192.json}"
  DSARGS="--dataset-name sharegpt --dataset-path $DATASET --sharegpt-context-len $CTX"
fi
[ -f "$DATASET" ] || { echo "MISSING DATASET $DATASET -- run build_sharegpt_long.py first"; exit 2; }

OUT="$ROOT/w1_probe_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"

# measurement stage: the chunk probe stays OFF (its observer effect is n=3 with a
# CI outside the margin -- P1-f), and no SLO scheduler is engaged.
unset PDMUX_CHUNK_PROBE_PATH PDMUX_CHUNK_PROBE_ARM PDMUX_CHUNK_PROBE_RUN_ID \
      PDMUX_CHUNK_PROBE_WORKLOAD_ID PDMUX_SLO_SCHED PDMUX_SLO_MODE

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

echo "W1 arms=[$ARMS] rates=[$RATES] seed=$SEED ctx=$CTX job=$SLURM_JOB_ID"
echo "W1 dsmode=$DSMODE dataset=$DATASET dsargs=[$DSARGS]"
[ -f "$DATASET.meta.json" ] && { echo "--- dataset meta ---"; cat "$DATASET.meta.json"; }

arm_flags () {
  case "$1" in
    fused_default) echo "" ;;
    cp512)  echo "--chunked-prefill-size 512" ;;
    cp1024) echo "--chunked-prefill-size 1024" ;;
    cp2048) echo "--chunked-prefill-size 2048" ;;
    cp8192) echo "--chunked-prefill-size 8192" ;;
    cp4096) echo "--chunked-prefill-size 4096" ;;
    d16|d24|d34|d44|d54)
      echo "--enable-pdmux --pdmux-config-path $SLO/pdmux_$1.yml --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "__BAD__" ;;
  esac
}

kill_server () { kill "$1" 2>/dev/null; sleep 3; pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 5; }

wait_ready () {
  local PID="$1" PORT="$2" ok=0 i
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

SUMMARY="$OUT/w1_summary.jsonl"; : > "$SUMMARY"

for ARM in $ARMS; do
  FLAGS=$(arm_flags "$ARM")
  [ "$FLAGS" = "__BAD__" ] && { echo "bad arm=$ARM"; continue; }
  PORT=$((38200 + (SLURM_JOB_ID + RANDOM) % 400))
  SRVLOG="$OUT/srv_${ARM}.log"
  echo "############ boot arm=$ARM port=$PORT flags=[$FLAGS] ############"
  T0=$(date +%s)
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$BACKEND" $FLAGS \
    --disable-piecewise-cuda-graph \
    --disable-radix-cache --mem-fraction-static 0.82 \
    --max-running-requests 48 --context-length "$CTX" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!
  OK=$(wait_ready "$PID" "$PORT")
  T1=$(date +%s)
  echo "boot arm=$ARM ok=$OK boot_s=$((T1-T0))"
  echo "--- realized cps / piecewise / pdmux banner ($ARM) ---"
  grep -oE "chunked_prefill_size=[-0-9]+|piecewise_cuda_graph_max_tokens=[0-9]+|enable_pdmux=[A-Za-z]+|disable_overlap_schedule=[A-Za-z]+|disable_piecewise_cuda_graph=[A-Za-z]+" \
    "$SRVLOG" | sort -u | head -8
  grep -E "max_total_num_tokens" "$SRVLOG" | head -1

  if [ "$OK" != 1 ]; then
    echo "BOOT_FAILED arm=$ARM" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    echo "{\"arm\":\"$ARM\",\"boot_ok\":false}" >> "$SUMMARY"
    kill_server "$PID" "$PORT"; continue
  fi

  for RATE in $RATES; do
    # NP scales with rate so every rung takes a comparable wall time (prereg sec 3).
    NP=$(python3 -c "print(max(32,min(120,round($RATE*130))))")
    JF="$OUT/w1_${ARM}_r${RATE}.jsonl"; : > "$JF"
    echo "== arm=$ARM rate=$RATE np=$NP =="
    RT0=$(date +%s)
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      $DSARGS \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
      --output-details --output-file "$JF" \
      > "$OUT/bench_${ARM}_r${RATE}.log" 2>&1
    BRC=$?
    RT1=$(date +%s)
    echo "  rc=$BRC wall=$((RT1-RT0))s"
    grep -E "Request throughput|Median TTFT|Median ITL|P95 ITL|Successful requests" \
      "$OUT/bench_${ARM}_r${RATE}.log" | head -6
    echo "{\"arm\":\"$ARM\",\"rate\":$RATE,\"np\":$NP,\"rc\":$BRC,\"wall_s\":$((RT1-RT0)),\"file\":\"$JF\"}" >> "$SUMMARY"
  done
  kill_server "$PID" "$PORT"
done

echo "=== analysis ==="
PYTHONPATH=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks \
  python3 "$ROOT/w1_analyze.py" "$OUT" || echo "ANALYSIS_FAILED (artifacts kept)"
echo "DONE_w1_${SLURM_JOB_ID}"
