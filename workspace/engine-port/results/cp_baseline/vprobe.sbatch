#!/bin/bash
#SBATCH --job-name=vprobe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.err
#
# V-PROBE -- the UNTREATED VARIANCE of the paired change-trace goodput estimator.
# PREREG_VPROBE_2026-09-01.md.  Buys ONE number per arm: the SD of a paired Delta
# between two boots OF THE SAME ARM, in the estimator a later comparison would use.
#
# ★ WHY THIS EXISTS: the 5th rule-layer audit killed CP-2 rev1 on H1 -- the
# registered 3% equivalence margin sits BELOW the noise floor of its own estimator
# (archive re-scoring put the paired Delta SD at 3.2-4.0%), so `equivalent` was
# unreachable at n=4 and a direction label needed |Delta| >= 6.1-6.9%.  Methodology
# gate #86 asks for exactly this purchase BEFORE a continuous axis is registered:
# "연속 축 등록 전에 그 통계량의 무처치 분산을 재고 유도 밴드가 그보다 넓은지 산술로
# 보여라".  CP-2 rev1 skipped it and deferred the arithmetic (its 12.6), and that
# deferral is what the audit named a free parameter.
#
# ★ WHAT IT DOES NOT BUY (prereg sec 0, enforced by the design not by a promise):
#   NO policy verdict, NO arm ranking, NO chunked-prefill-vs-PD-mux comparison.
#   ONE JOB HOLDS ONE ARM.  A cross-arm Delta is therefore a CROSS-JOB comparison,
#   which CONSENSUS 1-31 already forbids citing without the (unrun) delta probe.
#   The comparison is not "discouraged" here; it is not computable within a job.
#
# usage:
#   sbatch --export=ALL,VP_ARM=fused_default,VP_SEED=11 vprobe.sbatch   # smoke first
#   (then the remaining 7 cells, --time set from the smoke's measured wall time)

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched

# ---- registered design constants (PREREG_VPROBE_2026-09-01.md sec 3) ----------
ARM="${VP_ARM:?VP_ARM must be set (fused_default|d44)}"
SEED="${VP_SEED:?VP_SEED must be set (11|23|37|53)}"
MODEL="${MODEL:-Zyphra/Zamba2-2.7B}"
CTX="${CTX:-4096}"
BACKEND="${BACKEND:-triton}"
NP="${NP:-200}"
ROUNDS="${ROUNDS:-3}"
RLO="${RLO:-3}"
RHI="${RHI:-12}"
BOOTS="${VP_BOOTS:-2}"          # two boots of the SAME arm = one null Delta

OUT="$ROOT/vprobe_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
SGPT="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# the chunk probe stays OFF: this job measures variance, not chunking, and the
# probe's observer effect is itself unmeasured (P1-f, n=3, CI outside the margin).
unset PDMUX_CHUNK_PROBE_PATH PDMUX_CHUNK_PROBE_ARM PDMUX_CHUNK_PROBE_RUN_ID \
      PDMUX_CHUNK_PROBE_WORKLOAD_ID PDMUX_SLO_SCHED PDMUX_SLO_MODE

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || exit 2

echo "VPROBE arm=$ARM seed=$SEED boots=$BOOTS rounds=$ROUNDS LO=$RLO HI=$RHI np=$NP job=$SLURM_JOB_ID"

# ---- arm -> server flags.  The ONLY place the two families differ. ------------
arm_flags () {
  case "$1" in
    fused_default) echo "" ;;                     # cps default (8192), overlap ON,
                                                  # piecewise ON -- the CP family at
                                                  # its own operating point
    cp512)  echo "--chunked-prefill-size 512" ;;
    d24|d34|d44|d54|d16)
      echo "--enable-pdmux --pdmux-config-path $SLO/pdmux_$1.yml --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "__BAD__" ;;
  esac
}
FLAGS=$(arm_flags "$ARM")
[ "$FLAGS" = "__BAD__" ] && { echo "bad VP_ARM=$ARM"; exit 2; }

kill_server () {
  local PID="$1" PORT="$2"
  kill "$PID" 2>/dev/null
  sleep 3
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 5
}

wait_ready () {
  local PID="$1" PORT="$2" ok=0 i
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

SUMMARY="$OUT/vprobe_summary.jsonl"; : > "$SUMMARY"

for B in $(seq 1 "$BOOTS"); do
  PORT=$((37100 + (SLURM_JOB_ID + B * 37) % 400))
  TAG="${ARM}_s${SEED}_b${B}"
  SRVLOG="$OUT/srv_${TAG}.log"
  JLO="$OUT/vpLo_${TAG}.jsonl"; JHI="$OUT/vpHi_${TAG}.jsonl"; : > "$JLO"; : > "$JHI"
  echo "############ boot $B/$BOOTS arm=$ARM seed=$SEED port=$PORT ############"
  T0=$(date +%s)
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$BACKEND" $FLAGS \
    --disable-radix-cache --mem-fraction-static 0.82 \
    --max-running-requests 48 --context-length "$CTX" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!
  OK=$(wait_ready "$PID" "$PORT")
  T1=$(date +%s)
  echo "boot_ok=$OK boot_s=$((T1-T0)) tag=$TAG"
  # F3 evidence: record what cps/piecewise the engine actually realized, per boot.
  echo "--- realized cps / piecewise banner ($TAG) ---"
  grep -iE "chunked_prefill_size|piecewise_cuda_graph|enable_pdmux" "$SRVLOG" \
    | grep -viE "Future|pynvml" | head -4

  if [ "$OK" != 1 ]; then
    echo "BOOT_FAILED tag=$TAG" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -30 "$SRVLOG" | grep -viE "pynvml|Future"
    echo "{\"arm\":\"$ARM\",\"seed\":$SEED,\"boot\":$B,\"boot_ok\":false}" >> "$SUMMARY"
    kill_server "$PID" "$PORT"
    continue
  fi

  BT0=$(date +%s)
  for r in $(seq 1 "$ROUNDS"); do
    for PH in LO HI; do
      [ "$PH" = LO ] && { RATE=$RLO; JF="$JLO"; } || { RATE=$RHI; JF="$JHI"; }
      echo "== boot $B round $r $PH (rate $RATE) =="
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --dataset-name sharegpt \
        --dataset-path "$SGPT" --sharegpt-context-len 4000 \
        --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
        --output-details --output-file "$JF" \
        >> "$OUT/bench_${TAG}.log" 2>&1 \
        || echo "BENCH_FAILED tag=$TAG round=$r phase=$PH" | tee -a "$OUT/BOOT_FAILURES.txt"
    done
  done
  BT1=$(date +%s)
  echo "bench_s=$((BT1-BT0)) tag=$TAG"
  echo "{\"arm\":\"$ARM\",\"seed\":$SEED,\"boot\":$B,\"boot_ok\":true,\"boot_s\":$((T1-T0)),\"bench_s\":$((BT1-BT0)),\"lo\":\"$JLO\",\"hi\":\"$JHI\"}" >> "$SUMMARY"
  kill_server "$PID" "$PORT"
done

echo "=== per-boot scoring (canonical p95 predicate, gate #4) ==="
PYTHONPATH=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks \
  python3 "$ROOT/vprobe_score.py" "$OUT" || echo "SCORING_FAILED (artifacts kept)"
echo "DONE_vprobe_${ARM}_s${SEED}_${SLURM_JOB_ID}"
