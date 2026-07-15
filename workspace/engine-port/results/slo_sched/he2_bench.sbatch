#!/bin/bash
#SBATCH --job-name=he2
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# HE2 (design §E.5/§F): time-varying (NON-saturated) workload where the binding
# constraint ALTERNATES over time -> the ONLY regime where dynamic can BEAT static
# (each phase's optimal static differs; no single static is optimal throughout).
# Phase A = in3600/o32 (prefill-bound, best static d24); Phase B = in2000/o96
# (decode-heavy, best static d44). cudagraph ON. Compares agn/d24/d44/slo/bind.
#   arg1 MODE (agn|d24|d44|slo|bind) ; arg2 REP ; arg3 RATE_A ; arg4 RATE_B
# Then: aggregate COMBINED goodput per mode (dynamic should beat best single static).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"

MODE="${1:-bind}"; REP="${2:-1}"; RATE_A="${3:-3}"; RATE_B="${4:-5}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT="${NPROMPT:-48}"; ROUNDS="${ROUNDS:-3}"
# phase workloads env-configurable. defaults = original (both prefill-bound under cudagraph).
# to make phase B genuinely DECODE-bound: LB=256 OB=512 (tiny prefill, long decode, high concurrency).
LA="${LA:-3600}"; OA="${OA:-32}"; NA="${NA:-$NPROMPT}"
LB="${LB:-2000}"; OB="${OB:-96}"; NB="${NB:-$NPROMPT}"

# config + policy env per mode
case "$MODE" in
  d16|d24|d34|d44|d54) CFG="$OUT/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  agn) CFG="$OUT/pdmux_slo.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  slo) CFG="$OUT/pdmux_slo.yml"; export PDMUX_SLO_SCHED=1 PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; unset PDMUX_SLO_MODE ;;
  bind) CFG="${SLO_CFG:-$OUT/pdmux_slo.yml}"; export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding PDMUX_TPOT_SLO_MS=60 \
        PDMUX_TTFT_SLO_MS="${PDMUX_TTFT_SLO_MS:-3000}" PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}" ;;
  *) echo "bad MODE=$MODE"; exit 1 ;;
esac
if [ -n "${PDMUX_NO_CG:-}" ]; then CGFLAGS="--disable-cuda-graph --disable-piecewise-cuda-graph"; else CGFLAGS=""; fi

PORT=$((36100 + SLURM_JOB_ID % 150))
TAG="${MODE}_rep${REP}_rA${RATE_A}B${RATE_B}_${SLURM_JOB_ID}"
SRVLOG="$OUT/he2srv_${TAG}.log"; BENCHLOG="$OUT/he2_${TAG}.log"
JA="$OUT/he2A_${TAG}.jsonl"; JB="$OUT/he2B_${TAG}.jsonl"; : > "$JA"; : > "$JB"
echo "HE2 TAG=$TAG MODE=$MODE cg=$([ -z "${PDMUX_NO_CG:-}" ] && echo ON || echo OFF) rA=$RATE_A rB=$RATE_B port=$PORT job=$SLURM_JOB_ID"; cat "$CFG"

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
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^/SANITY /'
  for round in $(seq 1 $ROUNDS); do
    echo "== round $round PHASE A (in${LA}/o${OA} @${RATE_A}) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $LA --random-output-len $OA --random-range-ratio 1.0 \
      --num-prompts $NA --request-rate $RATE_A --output-details --output-file "$JA" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
    echo "== round $round PHASE B (in${LB}/o${OB} @${RATE_B}) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $LB --random-output-len $OB --random-range-ratio 1.0 \
      --num-prompts $NB --request-rate $RATE_B --output-details --output-file "$JB" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
  done
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
python3 - "$JA" "$JB" "$TAG" "${RATE_A}/${RATE_B}" <<'PY'
import json,sys
def pct(a,q):
    if not a: return 0.0
    a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
def load(f):
    g=0;dur=0;tts=[];itls=[]
    for ln in open(f):
        ln=ln.strip()
        if not ln: continue
        o=json.loads(ln); tt=o.get("ttfts") or []; il=o.get("itls") or []; d=o.get("duration") or 0
        dur=max(dur,d)
        for i,t in enumerate(tt):
            m=(sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9.0
            tts.append(t); itls.append(m*1000.0)
            if t<=3.0 and m<=0.06: g+=1
    return g,dur,tts,itls
ja,jb,tag,rate=sys.argv[1:5]
gA,dA,ttA,ilA=load(ja); gB,dB,ttB,ilB=load(jb)
gpA=gA/dA if dA else 0; gpB=gB/dB if dB else 0; gpC=(gA+gB)/(dA+dB) if (dA+dB) else 0
print(f"HE2_RESULT tag={tag} rate={rate} | phaseA(prefill) gp={gpA:.3f} | phaseB(decode) gp={gpB:.3f} | COMBINED={gpC:.3f} | min-phase={min(gpA,gpB):.3f}")
def pl(nm,tt,il):
    return (f"HE2_PCT tag={tag} {nm} | TTFTs p50/p95/p99={pct(tt,.5):.2f}/{pct(tt,.95):.2f}/{pct(tt,.99):.2f} "
            f"| ITLms p50/p95/p99={pct(il,.5):.1f}/{pct(il,.95):.1f}/{pct(il,.99):.1f}")
print(pl("A",ttA,ilA)); print(pl("B",ttB,ilB))
PY
echo "DONE_${TAG}"
