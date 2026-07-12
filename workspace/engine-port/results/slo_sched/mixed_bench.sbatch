#!/bin/bash
#SBATCH --job-name=slo_mixed
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# B: MIXED/phase-alternating workload. Alternates phase A (in3600/o32=prefill-bound) and
# phase B (in2000/o96=decode-heavy) on ONE server -> no single static split is optimal throughout.
#   arg1 LABEL ; arg2 REP ; arg3 CONFIG ; arg4 SLOFLAG ; arg5 RATE
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"
LABEL="${1:-mix}"; REP="${2:-1}"; CFG="$OUT/${3:-pdmux_slo.yml}"; SLOF="${4:-1}"; RATE_A="${5:-2}"; RATE_B="${6:-4}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT=48; ROUNDS=3
if [ "$SLOF" = "1" ]; then export PDMUX_SLO_SCHED=1; export PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; else unset PDMUX_SLO_SCHED; fi
PORT=$((35600 + SLURM_JOB_ID % 150))
SRVLOG="$OUT/mixsrv_${LABEL}_rA${RATE_A}B${RATE_B}_${SLURM_JOB_ID}.log"; BENCHLOG="$OUT/mix_${LABEL}_rA${RATE_A}B${RATE_B}_${SLURM_JOB_ID}.log"
JA="$OUT/mixA_${LABEL}_rA${RATE_A}B${RATE_B}_${SLURM_JOB_ID}.jsonl"; JB="$OUT/mixB_${LABEL}_rA${RATE_A}B${RATE_B}_${SLURM_JOB_ID}.jsonl"; : > "$JA"; : > "$JB"
echo "MIXED LABEL=$LABEL cfg=${3} SLO=$SLOF RATE_A=$RATE_A RATE_B=$RATE_B port=$PORT job=$SLURM_JOB_ID"; cat "$CFG"
python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!; ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "boot_ok=$ok"
if [ "$ok" = 1 ]; then
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^/SANITY /'
  for round in $(seq 1 $ROUNDS); do
    echo "===== round $round PHASE A (in3600/o32 prefill-bound) ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len 3600 --random-output-len 32 --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $RATE_A --output-details --output-file "$JA" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT|throughput"
    echo "===== round $round PHASE B (in2000/o96 decode-heavy) ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $RATE_B --output-details --output-file "$JB" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT|throughput"
  done
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
python3 - "$JA" "$JB" "$LABEL" "${RATE_A}/${RATE_B}" <<'PY'
import json,sys
def load(f):
    g=d=0
    for ln in open(f):
        ln=ln.strip()
        if not ln: continue
        o=json.loads(ln); tt=o.get("ttfts") or []; il=o.get("itls") or []; dur=o.get("duration") or 0
        good=sum(1 for i,t in enumerate(tt) if t<=3.0 and ((sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9)<=0.06)
        g+=good; d+=dur
    return g,d
ja,jb,lab,rate=sys.argv[1:5]
gA,dA=load(ja); gB,dB=load(jb)
gpA=gA/dA if dA else 0; gpB=gB/dB if dB else 0; gpC=(gA+gB)/(dA+dB) if (dA+dB) else 0
print(f"MIXED_RESULT label={lab} rate={rate} | phaseA(prefill) goodput={gpA:.3f} | phaseB(decode) goodput={gpB:.3f} | COMBINED={gpC:.3f} | min-phase={min(gpA,gpB):.3f}")
PY
echo "DONE_${LABEL}"
