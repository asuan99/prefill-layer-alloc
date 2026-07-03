#!/bin/bash
#SBATCH --job-name=p1_4_nh_ctxsens
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:35:00
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
SMFILE="$HERE/.nhc_decode_sm_${SLURM_JOB_ID}"
export SGLANG_NH_LAYER_TIMING=1 PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full@0 > "$SMFILE"
PORT=31071
SRVLOG="$HERE/p1_4_nhc_srv_${SLURM_JOB_ID}.log"
RESULT="$HERE/p1_4_nh_ctxsens_${SLURM_JOB_ID}.txt"; : > "$RESULT"

python -m sglang.launch_server \
  --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.8 --max-running-requests 24 --context-length 8192 \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; echo healthy; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done

send_load () {  # $1=ctx tokens
  python3 - "$1" "$PORT" <<'PY' &
import sys, json, urllib.request, concurrent.futures
ctx, port = int(sys.argv[1]), sys.argv[2]
prompt = ("the robot walked slowly across " * ((ctx//5)+1))
payload = json.dumps({"text": prompt, "sampling_params": {"max_new_tokens": 100, "temperature": 0.7, "ignore_eos": True}}).encode()
def one(_):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=payload, headers={"Content-Type":"application/json"})
    try: urllib.request.urlopen(req, timeout=300).read()
    except Exception as e: pass
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(one, range(16)))
PY
  echo $!
}

if [ "$ok" = 1 ]; then
  for CTX in 256 2048 5000; do
    for SM in full 44 16; do
      echo "$SM@$CTX" > "$SMFILE"
      echo "--- ctx=$CTX sm=$SM ---"
      lp=$(send_load "$CTX"); wait "$lp"; sleep 2
      line=$(grep "NHLT mode=$SM@$CTX " "$SRVLOG" | tail -1)
      echo "RESULT $line" | tee -a "$RESULT"
    done
  done
else
  echo "SERVER_FAILED"; tail -30 "$SRVLOG" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null
echo "===== DONE ====="; cat "$RESULT"
