#!/bin/bash
#SBATCH --job-name=p1_4_nh_smsens
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
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
export SGLANG_NH_LAYER_TIMING=1
PORT=31055
RESULT="$HERE/p1_4_nh_smsens_${SLURM_JOB_ID}.txt"
: > "$RESULT"

run_one () {
  local SM="$1"   # "full" or an int
  local SRVLOG="$HERE/p1_4_srv_${SLURM_JOB_ID}_${SM}.log"
  if [ "$SM" = "full" ]; then unset PDMUX_FIXED_DECODE_SM; else export PDMUX_FIXED_DECODE_SM="$SM"; fi
  python -m sglang.launch_server \
    --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
    --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
    --mem-fraction-static 0.8 --max-running-requests 40 --context-length 4096 \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 5
  done
  if [ "$ok" = 1 ]; then
    # sustained decode load: 32 concurrent requests, 200 new tokens each -> ~200 decode steps
    for r in $(seq 1 32); do
      curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
        -d '{"text":"Write a long story about a robot.","sampling_params":{"max_new_tokens":200,"temperature":0.7,"ignore_eos":true}}' >/dev/null 2>&1 &
    done
    wait
    sleep 2
    local line=$(grep "NHLT" "$SRVLOG" | tail -1)
    echo "SM=$SM | $line" | tee -a "$RESULT"
  else
    echo "SM=$SM | SERVER_FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" >> "$RESULT"
  fi
  kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null; sleep 2
}

echo "===== NemotronH decode per-layer-type SM sensitivity =====" | tee -a "$RESULT"
for SM in full 64 32 16; do
  echo "--- running SM=$SM ---"
  run_one "$SM"
done
echo "===== DONE ====="; cat "$RESULT"
