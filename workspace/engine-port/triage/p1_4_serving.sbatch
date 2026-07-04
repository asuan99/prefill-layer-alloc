#!/bin/bash
#SBATCH --job-name=p1_4_serving
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
PORT=31088
RESULT="$HERE/p1_4_serving_${SLURM_JOB_ID}.txt"; : > "$RESULT"

run_policy () {   # $1 = fused | agnostic
  local POL="$1"; local SRVLOG="$HERE/p1_4_srv_${SLURM_JOB_ID}_${POL}.log"
  local EXTRA=""
  [ "$POL" = "agnostic" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"
  python -m sglang.launch_server \
    --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
    --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length 8192 \
    $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 150); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 5
  done
  if [ "$ok" = 1 ]; then
    python "$HERE/bench_client.py" --port $PORT --rate 2 --num 24 --in-len 2000 --out-len 96 --tag "$POL-warmup" >/dev/null 2>&1 || true
    for RATE in 3 6 10; do
      python "$HERE/bench_client.py" --port $PORT --rate $RATE --num 60 --in-len 2000 --out-len 96 --tag "$POL" | tee -a "$RESULT"
    done
  else
    echo "BENCH $POL SERVER_FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" >> "$RESULT"
  fi
  kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null; sleep 2
}

echo "===== NemotronH serving: fused vs agnostic(pdmux), in=2000/out=96 =====" | tee -a "$RESULT"
run_policy fused
run_policy agnostic
echo "===== DONE ====="; cat "$RESULT"
