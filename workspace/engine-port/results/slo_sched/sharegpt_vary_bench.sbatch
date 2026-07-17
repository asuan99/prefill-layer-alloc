#!/bin/bash
#SBATCH --job-name=sgptv
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
# TIME-VARYING real trace (user, 2026-07-15): ShareGPT with the arrival RATE alternating
# LO<->HI over time. Mechanism: optimal split follows DECODE LOAD (rate x output). LO phase
# = decode light -> prefill-heavy optimal (d16-ish); HI phase = decode binds -> decode-heavy
# optimal (d24/d44). No single static is optimal in BOTH -> the ONLY regime where dynamic
# can beat best-static. Reports per-phase goodput + SWITCH COUNT (for overhead/positioning
# decomposition) + TTFT/ITL percentiles.
#   arg1 MODE (d16|d24|d44|slo|bind) ; arg2 REP ; arg3 RATE_LO ; arg4 RATE_HI
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"

MODE="${1:-bind}"; REP="${2:-1}"; RLO="${3:-3}"; RHI="${4:-12}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NP="${NP:-200}"; ROUNDS="${ROUNDS:-3}"
SGPT="/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

case "$MODE" in
  d16|d24|d34|d44|d54) CFG="$OUT/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  agn) CFG="$OUT/pdmux_slo.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  slo) CFG="$OUT/pdmux_slo.yml"; export PDMUX_SLO_SCHED=1 PDMUX_TPOT_SLO_MS=60 PDMUX_QDEPTH_TARGET=4; unset PDMUX_SLO_MODE ;;
  bind) CFG="${SLO_CFG:-$OUT/pdmux_slo.yml}"; export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding PDMUX_TPOT_SLO_MS=60 \
        PDMUX_TTFT_SLO_MS="${PDMUX_TTFT_SLO_MS:-3000}" PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}" ;;
  *) echo "bad MODE=$MODE"; exit 1 ;;
esac
[ -n "${PDMUX_NO_CG:-}" ] && CGFLAGS="--disable-cuda-graph --disable-piecewise-cuda-graph" || CGFLAGS=""

PORT=$((36600 + SLURM_JOB_ID % 150))
TAG="${MODE}_rep${REP}_L${RLO}H${RHI}_${SLURM_JOB_ID}"
SRVLOG="$OUT/sgptvsrv_${TAG}.log"; BENCHLOG="$OUT/sgptv_${TAG}.log"
JLO="$OUT/sgptvLo_${TAG}.jsonl"; JHI="$OUT/sgptvHi_${TAG}.jsonl"; : > "$JLO"; : > "$JHI"
echo "SGPTV TAG=$TAG MODE=$MODE cg=$([ -z "${PDMUX_NO_CG:-}" ] && echo ON || echo OFF) RLO=$RLO RHI=$RHI port=$PORT job=$SLURM_JOB_ID"

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
  for r in $(seq 1 $ROUNDS); do
    echo "== round $r LO (rate $RLO) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts $NP --request-rate $RLO --output-details --output-file "$JLO" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
    echo "== round $r HI (rate $RHI) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts $NP --request-rate $RHI --output-details --output-file "$JHI" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
  done
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
# switch count (dynamic modes)
SW=$(grep -cE "SLO-BIND|SLO-SCHED" "$SRVLOG" 2>/dev/null); echo "SWITCHES tag=$TAG count=$SW"
python3 - "$JLO" "$JHI" "$TAG" "$RLO/$RHI" "$SW" <<'PY'
import json,sys
def pct(a,q):
    if not a: return 0.0
    a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
def load(f):
    g=0;dur=0;tt=[];il=[]
    for ln in open(f):
        ln=ln.strip()
        if not ln: continue
        o=json.loads(ln); T=o.get("ttfts") or []; I=o.get("itls") or []; d=o.get("duration") or 0
        # BUGFIX (2026-07-17): this file holds ONE json object PER ROUND. `good` accumulates over
        # all rounds, so the denominator must be the SUM of round durations, not the max —
        # max() inflated every reported goodput by ~3x (= ROUNDS). Ranking was unaffected
        # (pure scale factor), absolute values were not.
        dur+=d
        for i,t in enumerate(T):
            m=(sum(I[i])/len(I[i])) if (i<len(I) and I[i]) else 9.0
            tt.append(t); il.append(m*1000)
            if t<=3.0 and m<=0.06: g+=1
    return g,dur,tt,il
jl,jh,tag,rate,sw=sys.argv[1:6]
gL,dL,ttL,ilL=load(jl); gH,dH,ttH,ilH=load(jh)
gpL=gL/dL if dL else 0; gpH=gH/dH if dH else 0; gpC=(gL+gH)/(dL+dH) if (dL+dH) else 0
print(f"SGPTV_RESULT tag={tag} rate={rate} sw={sw} | LO gp={gpL:.3f} | HI gp={gpH:.3f} | COMBINED={gpC:.3f}")
print(f"SGPTV_PCT tag={tag} LO | TTFT p50/p95/p99={pct(ttL,.5):.2f}/{pct(ttL,.95):.2f}/{pct(ttL,.99):.2f} ITL {pct(ilL,.5):.0f}/{pct(ilL,.95):.0f}/{pct(ilL,.99):.0f}")
print(f"SGPTV_PCT tag={tag} HI | TTFT p50/p95/p99={pct(ttH,.5):.2f}/{pct(ttH,.95):.2f}/{pct(ttH,.99):.2f} ITL {pct(ilH,.5):.0f}/{pct(ilH,.95):.0f}/{pct(ilH,.99):.0f}")
PY
echo "DONE_${TAG}"
