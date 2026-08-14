#!/bin/bash
#SBATCH --job-name=slo_lff
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# Step D (SKETCH): context-length FEEDFORWARD split. Heterogeneous-L workload = two
# CONCURRENT streams on one server: short-flood (in~512, hi rate) + long-trickle
# (in~16000, lo rate). A single static split can't serve both; feedback-only SLO reacts
# only after latency degrades; L-feedforward should preemptively give the long requests
# prefill SM (protect their TTFT tail) without starving the short requests' decode.
#   arg1 CFGNAME (pdmux_slo|pdmux_d24|pdmux_d44) ; arg2 MODE (agn|slo|lff) ; arg3 REP
# Compare: agnostic (pdmux_slo cfg + MODE=agn) / d24 (pdmux_d24 + agn) /
#          SLO-v7b (pdmux_slo + slo) / SLO+LFF (pdmux_slo + lff).
# Then: python analyze_lff.py  (per length-bucket TTFT p50/p99 + goodput@SLO + TPOT).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1   # long-trickle L=16000 > Zamba2 4096 (timing valid)

CFGNAME="${1:-pdmux_slo}"; MODE="${2:-lff}"; REP="${3:-1}"
CFG="$OUT/${CFGNAME}.yml"
# load params env-overridable (first run at short8/long16k saturated: TTFT 17-57s >> 3s SLO).
# calibrate NEAR the SLO boundary so policy differences are visible (not saturated, not idle).
MODEL=Zyphra/Zamba2-2.7B
L_SHORT="${L_SHORT:-512}";   R_SHORT="${R_SHORT:-3}";   N_SHORT="${N_SHORT:-60}"   # short-flood
L_LONG="${L_LONG:-8000}";    R_LONG="${R_LONG:-0.4}";   N_LONG="${N_LONG:-10}"     # long-trickle
OUTTOK="${OUTTOK:-64}"
CTX=$((L_LONG + 1000))

# policy env
if [ "$MODE" = "slo" ]; then export PDMUX_SLO_SCHED=1; export PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; unset PDMUX_SLO_LFF PDMUX_SLO_MODE
elif [ "$MODE" = "lff" ]; then export PDMUX_SLO_SCHED=1; export PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; export PDMUX_SLO_LFF="${PDMUX_SLO_LFF:-0.75}" PDMUX_L_REF_TOK="${PDMUX_L_REF_TOK:-3600}"; unset PDMUX_SLO_MODE
elif [ "$MODE" = "bind" ]; then export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding; export PDMUX_TPOT_SLO_MS=60 PDMUX_TTFT_SLO_MS="${PDMUX_TTFT_SLO_MS:-3000}" PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}" PDMUX_SLO_PF_URGENCY="${PDMUX_SLO_PF_URGENCY:-0.5}"; unset PDMUX_SLO_LFF   # Step E binding-first (anchor idx2=d24)
else unset PDMUX_SLO_SCHED PDMUX_SLO_LFF PDMUX_SLO_MODE; fi   # agn = agnostic (decode_bs) baseline

PORT=$((35800 + SLURM_JOB_ID % 150))
TAG="${CFGNAME}_${MODE}_rep${REP}_${SLURM_JOB_ID}"
SRVLOG="$OUT/lffsrv_${TAG}.log"; BENCHLOG="$OUT/lff_${TAG}.log"
JS="$OUT/lffShort_${TAG}.jsonl"; JL="$OUT/lffLong_${TAG}.jsonl"; : > "$JS"; : > "$JL"
echo "LFF TAG=$TAG cfg=$CFGNAME MODE=$MODE LFF=${PDMUX_SLO_LFF:-off} Lref=${PDMUX_L_REF_TOK:-na} port=$PORT job=$SLURM_JOB_ID"; cat "$CFG"

# cudagraph ON by default (decode ITL margin -> binding shifts to TTFT = the regime where L-feedforward
# can pay off). Set PDMUX_NO_CG=1 to reproduce the no-cudagraph (decode-ITL-bound) runs.
if [ -n "${PDMUX_NO_CG:-}" ]; then CGFLAGS="--disable-cuda-graph --disable-piecewise-cuda-graph"; else CGFLAGS=""; fi
echo "cudagraph=$([ -z "${PDMUX_NO_CG:-}" ] && echo ON || echo OFF)"
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
  # launch the two streams CONCURRENTLY -> heterogeneous-L load co-resident
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len $L_LONG --random-output-len $OUTTOK --random-range-ratio 1.0 \
    --num-prompts $N_LONG --request-rate $R_LONG --output-details --output-file "$JL" 2>>"$BENCHLOG" &
  PL=$!
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len $L_SHORT --random-output-len $OUTTOK --random-range-ratio 1.0 \
    --num-prompts $N_SHORT --request-rate $R_SHORT --output-details --output-file "$JS" 2>>"$BENCHLOG" &
  PS=$!
  wait $PL $PS
  echo "streams done"
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2

# inline summary (detailed per-bucket analysis in analyze_lff.py)
python3 - "$JS" "$JL" "$TAG" <<'PY'
import json,sys
def load(f):
    tt=[]; g=d=0
    for ln in open(f):
        ln=ln.strip()
        if not ln: continue
        o=json.loads(ln); ttfts=o.get("ttfts") or []; il=o.get("itls") or []; dur=o.get("duration") or 0
        tt+=ttfts
        g+=sum(1 for i,t in enumerate(ttfts) if t<=3.0 and ((sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9)<=0.06)
        d=max(d,dur)
    tt.sort(); n=len(tt)
    p=lambda q: tt[min(n-1,int(q*n))] if n else 0
    return g,d,(p(.5),p(.99)),n
js,jl,tag=sys.argv[1:4]
gS,dS,tS,nS=load(js); gL,dL,tL,nL=load(jl)
dur=max(dS,dL) or 1
print(f"LFF_RESULT tag={tag} | overall_goodput={(gS+gL)/dur:.3f} "
      f"| SHORT n={nS} good={gS} TTFT_p50={tS[0]:.3f} p99={tS[1]:.3f} "
      f"| LONG n={nL} good={gL} TTFT_p50={tL[0]:.3f} p99={tL[1]:.3f}")
PY
echo "DONE_${TAG}"
