#!/bin/bash
#SBATCH --job-name=sgpt
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
# ============================================================================
# REAL-TRACE validation (open item, handoff 2026-07-15): all prior conclusions were
# on synthetic fixed-length workloads (range-ratio 1.0). A real trace has variable
# prompt/output lengths & mix -> the instantaneous optimal split may MOVE over time.
# Q: on ShareGPT, (a) does the best static split differ / move, (b) can dynamic beat
# best-static? If dynamic > best-static -> the optimum moved & dynamic exploited it.
#   arg1 MODE (d16|d24|d34|d44|slo|bind) ; arg2 REP ; arg3 RATE
# cudagraph ON. Compares static sweep vs slo/bind. goodput@SLO + TTFT/ITL percentiles.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"

MODE="${1:-bind}"; REP="${2:-1}"; RATE="${3:-4}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT="${NPROMPT:-400}"
SGPT="/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

case "$MODE" in
  d16|d24|d34|d44|d54) CFG="$OUT/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  agn) CFG="$OUT/pdmux_slo.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  slo) CFG="$OUT/pdmux_slo.yml"; export PDMUX_SLO_SCHED=1 PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; unset PDMUX_SLO_MODE ;;
  bind) CFG="${SLO_CFG:-$OUT/pdmux_slo.yml}"; export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding PDMUX_TPOT_SLO_MS=60 \
        PDMUX_TTFT_SLO_MS="${PDMUX_TTFT_SLO_MS:-3000}" PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}" ;;
  *) echo "bad MODE=$MODE"; exit 1 ;;
esac
if [ -n "${PDMUX_NO_CG:-}" ]; then CGFLAGS="--disable-cuda-graph --disable-piecewise-cuda-graph"; else CGFLAGS=""; fi

PORT=$((36400 + SLURM_JOB_ID % 150))
TAG="${MODE}_rep${REP}_r${RATE}_${SLURM_JOB_ID}"
SRVLOG="$OUT/sgptsrv_${TAG}.log"; BENCHLOG="$OUT/sgpt_${TAG}.log"
J="$OUT/sgpt_${TAG}.jsonl"; : > "$J"
echo "SGPT TAG=$TAG MODE=$MODE cg=$([ -z "${PDMUX_NO_CG:-}" ] && echo ON || echo OFF) rate=$RATE nprompt=$NPROMPT port=$PORT job=$SLURM_JOB_ID"; cat "$CFG"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton $CGFLAGS --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!; ok=0
for i in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "boot_ok=$ok"
if [ "$ok" = 1 ]; then
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
    --num-prompts $NPROMPT --request-rate $RATE --output-details --output-file "$J" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT|request throughput|Input token|Output token"
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
python3 - "$J" "$TAG" "$RATE" <<'PY'
import json,sys
def pct(a,q):
    if not a: return 0.0
    a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
g=0;dur=0;tts=[];itls=[];ins=[]
for ln in open(sys.argv[1]):
    ln=ln.strip()
    if not ln: continue
    o=json.loads(ln); tt=o.get("ttfts") or []; il=o.get("itls") or []; d=o.get("duration") or 0
    il_in=o.get("input_lens") or o.get("prompt_len") or []
    dur=max(dur,d)  # OK here: this file holds exactly ONE round (cf. sharegpt_vary_bench.sbatch bugfix)
    for i,t in enumerate(tt):
        m=(sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9.0
        tts.append(t); itls.append(m*1000.0)
        if t<=3.0 and m<=0.06: g+=1
tag,rate=sys.argv[2:4]
gp=g/dur if dur else 0
print(f"SGPT_RESULT tag={tag} rate={rate} | n={len(tts)} good={g} goodput={gp:.3f}")
print(f"SGPT_PCT tag={tag} | TTFTs p50/p95/p99={pct(tts,.5):.2f}/{pct(tts,.95):.2f}/{pct(tts,.99):.2f}s "
      f"| ITLms p50/p95/p99={pct(itls,.5):.1f}/{pct(itls,.95):.1f}/{pct(itls,.99):.1f}")
PY
echo "DONE_${TAG}"
