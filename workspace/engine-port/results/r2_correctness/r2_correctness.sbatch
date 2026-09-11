#!/bin/bash
#SBATCH --job-name=r2corr
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/%x_%j.err
#
# R2 GPU CORRECTNESS GATE -- does PDMUX_TRUE_DUAL_WORKER=1 compute the same
# function as the legacy single-thread PD-mux loop, at the operating point?
#
# ★ INVARIANT UNDER TEST
#     greedy(prompt | true_dual fixed D) == greedy(prompt | legacy fixed D)
#   token ids, every prompt, server seed identical, cudagraph ON in both.
#   The arm pair is exactly the P2 / Claim D pair as scripts/r2_eval/
#   policy_adapter.sh builds it (legacy:fixed vs true_dual:fixed), on the
#   canonical R2 state set benchmarks/configs/pdmux_r2.yml.
#
# ★ BOOTS  (default order L TD L TD -> labels L1 TD1 L2 TD2)
#   The second L and TD boots are the NULL CONTROL: under concurrency batch
#   composition depends on timing, so a legacy-vs-legacy difference is the
#   noise floor any legacy-vs-true_dual difference has to be judged against.
#   Phase S (one request at a time) must be identical across ALL boots.
#
# ★ CUDAGRAPH MUST BE ON AND MUST STAY ON
#   No --disable-cuda-graph.  --decode-log-interval 1 makes every decode step
#   log "cuda graph: True/False"; the checker requires True on every Decode
#   line of every boot.  In true-dual mode every decode run_batch is issued
#   from the decode worker thread (multiplexing_mixin.py event_loop_pdmux,
#   decode_future), so "all True" there means graph REPLAY happened on that
#   thread -- capture at boot alone would not show it.  The same signature
#   separates cgON/cgOFF on existing data (results/cudagraph_probe/
#   srv_pdmux_cgON_847391.log 98/98 True; srv_pdmux_cgOFF_847391.log 0/41).
#
# ★ DEVIATIONS FROM scripts/r2_eval/engine_bench_runner.sh (deliberate)
#   --context-length 4096 (runner default 16384 exceeds Zamba2-2.7B
#   max_position_embeddings=4096 and SGLang refuses to boot without
#   SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN); --decode-log-interval 1.
#   PDMUX_TRACE_FORCE_PREFILL=1 (observation only, both arms) so the
#   checker can see prefill/decode overlap in telemetry.
#
# ★ WHAT A PASS DOES NOT MEAN
#   Not a performance result, not Claim D.  It certifies that the two
#   architectures agree on outputs at this split and that cudagraph stayed on.
#   Other splits (D16/D24/D34), generic/hybrid policies, sampling (T>0) and
#   other models are not covered.
#
# Cost estimate: warmup boot (cold import ~4 min) + 4 scored boots x ~2 min
# (boot ~45 s incl. 6 stream groups x 10 graph sizes, phase S 16 x 64 tok,
# phase C 32 staggered requests) ~= 15-25 min => ~0.25-0.4 GPU-h; hard cap
# 1.25 GPU-h (wall limit).
#
# usage:
#   sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
#   R2C_DSM=24 sbatch ...          # other split (must exist in pdmux_r2.yml)
#   R2C_ORDER="L TD TD L" sbatch ...

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
PROJECT=/scratch/ehmoon/whlee/prefill-layer-alloc
cd "$PROJECT" || exit 2

HERE=$PROJECT/workspace/engine-port/results/r2_correctness
CONFIG=$PROJECT/workspace/engine-port/benchmarks/configs/pdmux_r2.yml
OUT="$HERE/job_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"

MODEL="${R2C_MODEL:-Zyphra/Zamba2-2.7B}"
DSM="${R2C_DSM:-44}"
SEED="${R2C_SEED:-1}"
ORDER="${R2C_ORDER:-L TD L TD}"

# Start from a clean PD-mux environment (policy_adapter.sh does the same).
for var in $(env | grep -o '^PDMUX_[A-Z0-9_]*' || true); do unset "$var"; done
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

COMMIT=$(git -C "$PROJECT" rev-parse HEAD)
{
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname) date=$(date -Is)"
  echo "commit=$COMMIT"
  echo "src_dirty:"; git -C "$PROJECT" status --porcelain -- workspace/engine-port/src
  nvidia-smi -L
  python -c "from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local as f; print('thread_local_role_patch', f())"
  echo "model=$MODEL dsm=$DSM seed=$SEED order=[$ORDER] config=$CONFIG"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

server_cmd () {   # $1 = port
  echo python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length 4096 \
    --random-seed "$SEED" --host 127.0.0.1 --port "$1" \
    --enable-pdmux --pdmux-config-path "$CONFIG" --chunked-prefill-size -1 \
    --disable-overlap-schedule --decode-log-interval 1
}

wait_health () {  # $1 = port, $2 = pid, $3 = max polls (5 s each)
  for _ in $(seq 1 "$3"); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null || true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

stop_server () {  # $1 = port, $2 = pid
  kill "$2" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $1" 2>/dev/null; sleep 15
}

BASEPORT=$((30000 + (${SLURM_JOB_ID:-0} % 5000) * 6))

# Warm the import + weight page cache on a throwaway boot so the first scored
# boot does not time out (cp_baseline job 902397 lost two arms to exactly
# that; its BOOT_FAILED was a harness artifact, not an engine fact).
echo "=== warmup boot (not scored) ==="
env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" \
  $(server_cmd "$BASEPORT") > "$OUT/srv_warmup.log" 2>&1 &
WPID=$!
wait_health "$BASEPORT" "$WPID" 180 && echo "  warmup up" || echo "  warmup did not come up (continuing)"
stop_server "$BASEPORT" "$WPID"

: > "$OUT/boots.txt"
n_l=0; n_td=0; i=0
for ARM in $ORDER; do
  i=$((i + 1))
  case "$ARM" in
    L)  n_l=$((n_l + 1)); LABEL="L$n_l"; TD_ENV="" ;;
    TD) n_td=$((n_td + 1)); LABEL="TD$n_td"; TD_ENV="PDMUX_TRUE_DUAL_WORKER=1" ;;
    *)  echo "bad arm $ARM"; continue ;;
  esac
  PORT=$((BASEPORT + i))
  echo ""
  echo "########## boot=$LABEL arm=$ARM port=$PORT ##########"
  echo -n "$LABEL " >> "$OUT/boots.txt"
  env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" \
    PDMUX_TELEMETRY_PATH="$OUT/tel_${LABEL}.jsonl" \
    PDMUX_RUN_ID="r2corr_${SLURM_JOB_ID:-local}_${LABEL}" \
    PDMUX_WORKLOAD_ID="r2corr_seed${SEED}" \
    PDMUX_TRACE_FORCE_PREFILL=1 PDMUX_ENGINE_COMMIT="$COMMIT" $TD_ENV \
    $(server_cmd "$PORT") > "$OUT/srv_${LABEL}.log" 2>&1 &
  PID=$!
  if ! wait_health "$PORT" "$PID" 96; then
    echo "BOOT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
    grep -n "Traceback\|Error" "$OUT/srv_${LABEL}.log" | tail -5
    stop_server "$PORT" "$PID"
    continue
  fi
  grep -oE "disable_cuda_graph=[A-Za-z]+|random_seed=[0-9]+|enable_pdmux=[A-Za-z]+" \
    "$OUT/srv_${LABEL}.log" | sort -u | tr '\n' ' '; echo
  grep -m1 "PD-Multiplexing enabled" "$OUT/srv_${LABEL}.log"
  grep -m1 "true dual-worker enabled" "$OUT/srv_${LABEL}.log" || echo "  (legacy loop: no true-dual banner)"
  timeout 900 python3 "$HERE/r2_correctness_client.py" --port "$PORT" \
    --boot "$LABEL" --out "$OUT/gen_${LABEL}.json" \
    || echo "CLIENT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  kill -0 "$PID" 2>/dev/null || echo "SERVER_DIED_DURING_CLIENT boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  stop_server "$PORT" "$PID"
done

echo ""
echo "=== correctness check ==="
python3 "$HERE/r2_correctness_check.py" "$OUT" --dsm "$DSM" | tee "$OUT/verdict.txt"
echo "R2CORR_DONE_${SLURM_JOB_ID:-local}"
