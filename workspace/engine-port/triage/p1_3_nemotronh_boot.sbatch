#!/bin/bash
#SBATCH --job-name=p1_3_nemotronh_boot
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:25:00
#SBATCH --comment="field=efficientai;appl=pytorch"
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
MODE="${1:-plain}"   # plain | pdmux
PORT=31041; SRVLOG="$HERE/p1_3_server_${SLURM_JOB_ID}.log"
EXTRA=""
[ "$MODE" = "pdmux" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"

echo "===== NemotronH-8B boot (mode=$MODE) ====="
python -m sglang.launch_server \
  --model-path nvidia/Nemotron-H-8B-Base-8K \
  --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph \
  --disable-radix-cache --mem-fraction-static 0.75 --max-running-requests 16 \
  --context-length 4096 $EXTRA \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 96); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null || true)
  [ "$code" = "200" ] && { ok=1; echo "healthy after ${i}0s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }
  sleep 10
done
grep -iE "PD-Multiplexing enabled|stream group" "$SRVLOG" | head -2
if [ "$ok" = 1 ]; then
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":12,"temperature":0}}' | head -c 400; echo
  echo "===== P1.3 RESULT: NEMOTRONH_${MODE}_OK ====="
else
  echo "===== P1.3 RESULT: NEMOTRONH_${MODE}_FAILED ====="; tail -40 "$SRVLOG"
fi
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null
echo "===== DONE ====="
