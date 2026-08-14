#!/bin/bash
#SBATCH --job-name=p1_6_bench_one
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
POL="${1:-fused}"
PORT=$((31600 + SLURM_JOB_ID % 400))
LAFILE="$HERE/.one_mode_${SLURM_JOB_ID}"
SRVLOG="$HERE/p1_6one_srv_${SLURM_JOB_ID}.log"
BENCHLOG="$HERE/p1_6one_bench_${SLURM_JOB_ID}.log"
JSONL="$HERE/p1_6one_${POL}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
RATES="1 2 3 4 6"; NPROMPT=120; INLEN=2000; OUTLEN=96; MODEL=nvidia/Nemotron-H-8B-Base-8K

EXTRA=""
[ "$POL" != "fused" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"
if [ "$POL" = "layer_aware" ]; then echo layer_aware > "$LAFILE"; export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_LA_RESERVE="M-" PDMUX_LA_FLOOR_SM=16
elif [ "$POL" = "agnostic_v2" ]; then echo agnostic_v2 > "$LAFILE"; export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_AGN2_SM=16; fi

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length 8192 \
  $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 160); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "POLICY=$POL  boot_ok=$ok  port=$PORT"
if [ "$ok" = 1 ]; then
  echo "### warmup + first bench (stderr visible) ###"
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
    --num-prompts 16 --request-rate 2 2>&1 | tail -20
  for R in $RATES; do
    echo "===== $POL rate=$R ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$JSONL" 2>>"$BENCHLOG" \
      | grep -E "Request rate|Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E"
  done
  echo "### goodput@SLO (TTFT<=3s, TPOT<=60ms) ###"
  python3 - "$JSONL" <<'PY'
import json,sys
for ln in open(sys.argv[1]):
    d=json.loads(ln); rr=d.get("request_rate"); tt=d.get("ttfts") or []; il=d.get("itls") or []
    dur=d.get("duration") or 0.0
    good=sum(1 for i,t in enumerate(tt) if t<=3.0 and ((sum(il[i])/len(il[i])) if i<len(il) and il[i] else 9)<=0.06)
    print(f"  rate={rr} good={good}/{len(tt)} goodput={good/dur:.2f}req/s dur={dur:.1f}s")
PY
else echo "BOOT FAILED"; tail -15 "$SRVLOG" | grep -viE "pynvml|FutureWarning"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE_$POL"
