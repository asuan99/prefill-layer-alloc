#!/bin/bash
#SBATCH --job-name=iact
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
# INTERACTIVE-SLO controller retune + P90-attainment (plan: reports/interactive_slo_retune_plan.md).
# Unlike the 3s reanalysis (post-hoc rescore), the CONTROLLER SLO is set to the interactive value
# here -> controller behavior actually changes. Evaluation SLO == controller SLO (objective aligned).
#   arg1 MODE (d16|d24|d34|d44|d54|slo|bind) ; arg2 REP ; arg3 "RATE_LIST" (one boot, sweep rates)
#   env SLO_TTFT_MS (default 300=chat) SLO_ITL_MS (default 50) -> BOTH controller & scoring.
#   env for Stage B: PDMUX_SLO_PF_URGENCY / PDMUX_SLO_ANCHOR_IDX / PDMUX_SLO_FEAS_GATE / PDMUX_SLO_FEAS_OCC
# cudagraph ON. Reports attainment% (frac meeting TTFT&ITL SLO), goodput(SLO/s), P50/P90/P99 TTFT/ITL,
# and per-length-bucket attainment. P90 is the target percentile (p99 hits the physical long-req wall).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/../../triage/.triton_cache"

MODE="${1:-d24}"; REP="${2:-1}"; RATES="${3:-2 3 4 5 6}"
SLO_TTFT_MS="${SLO_TTFT_MS:-300}"; SLO_ITL_MS="${SLO_ITL_MS:-50}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT="${NPROMPT:-400}"
SGPT="/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SLO_ITL_S=$(python3 -c "print($SLO_ITL_MS/1000.0)"); SLO_TTFT_S=$(python3 -c "print($SLO_TTFT_MS/1000.0)")

case "$MODE" in
  d16|d24|d34|d44|d54) CFG="$OUT/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  slo) CFG="$OUT/pdmux_slo.yml"; export PDMUX_SLO_SCHED=1 PDMUX_TPOT_SLO_MS=$SLO_ITL_MS PDMUX_QDEPTH_TARGET=4; unset PDMUX_SLO_MODE ;;
  bind) CFG="${SLO_CFG:-$OUT/pdmux_slo.yml}"; export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding \
        PDMUX_TPOT_SLO_MS=$SLO_ITL_MS PDMUX_TTFT_SLO_MS=$SLO_TTFT_MS \
        PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}" PDMUX_SLO_PF_URGENCY="${PDMUX_SLO_PF_URGENCY:-0.5}" ;;
  *) echo "bad MODE=$MODE"; exit 1 ;;
esac
[ -n "${PDMUX_NO_CG:-}" ] && CGFLAGS="--disable-cuda-graph --disable-piecewise-cuda-graph" || CGFLAGS=""

PORT=$((37200 + SLURM_JOB_ID % 150))
TAG="${MODE}_rep${REP}_slo${SLO_TTFT_MS}-${SLO_ITL_MS}_${SLURM_JOB_ID}"
SRVLOG="$OUT/iactsrv_${TAG}.log"; BENCHLOG="$OUT/iact_${TAG}.log"
echo "IACT TAG=$TAG MODE=$MODE SLO=${SLO_TTFT_MS}ms/${SLO_ITL_MS}ms gate=${PDMUX_SLO_FEAS_GATE:-unset} rates=[$RATES] port=$PORT job=$SLURM_JOB_ID"

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
  for R in $RATES; do
    J="$OUT/iact_${TAG}_r${R}.jsonl"; : > "$J"
    echo "===== rate $R ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len 4000 \
      --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$J" 2>>"$BENCHLOG" \
      | grep -E "Request throughput|Median TTFT" || true
    python3 - "$J" "$TAG" "$R" "$SLO_TTFT_S" "$SLO_ITL_S" <<'PY'
import json,sys
def pct(a,q):
    if not a: return 0.0
    a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
J,tag,rate,ttft_slo,itl_slo=sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4]),float(sys.argv[5])
g=0;dur=0.0;tts=[];itls=[];ins=[]
for ln in open(J):
    ln=ln.strip()
    if not ln: continue
    o=json.loads(ln); tt=o.get("ttfts") or []; il=o.get("itls") or []; IL=o.get("input_lens") or []
    dur=max(dur,o.get("duration") or 0)   # one round per file
    for i,t in enumerate(tt):
        m=(sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9.0
        tts.append(t); itls.append(m*1000.0); ins.append(IL[i] if i<len(IL) else 0)
        if t<=ttft_slo and m<=itl_slo: g+=1
n=len(tts); att=100.0*g/n if n else 0; gp=g/dur if dur else 0
# per-length-bucket attainment
edges=[500,1000,2000]; blab=['<500','500-1k','1k-2k','>=2k']; bk=[[0,0] for _ in range(4)]
for i in range(n):
    bi=sum(1 for e in edges if ins[i]>=e)
    bk[bi][1]+=1; bk[bi][0]+= (1 if (tts[i]<=ttft_slo and itls[i]<=itl_slo*1000.0) else 0)
bstr=' '.join(f"{blab[i]} {100*bk[i][0]/bk[i][1] if bk[i][1] else 0:.0f}%" for i in range(4))
print(f"IACT_RESULT tag={tag} rate={rate} slo={ttft_slo*1000:.0f}/{itl_slo*1000:.0f}ms | n={n} att={att:.1f}% goodput={gp:.3f} | "
      f"TTFT p50/p90/p99={pct(tts,.5):.3f}/{pct(tts,.9):.3f}/{pct(tts,.99):.3f}s ITL p50/p90/p99={pct(itls,.5):.0f}/{pct(itls,.9):.0f}/{pct(itls,.99):.0f}ms")
print(f"IACT_BUCKET tag={tag} rate={rate} | {bstr}")
PY
  done
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
SW=$(grep -cE "SLO-BIND|SLO-SCHED" "$SRVLOG" 2>/dev/null); FR=$(grep -c "SLO-FEAS refused" "$SRVLOG" 2>/dev/null)
echo "IACT_META tag=$TAG switches=$SW feas_refused=$FR"
echo "DONE_${TAG}"
