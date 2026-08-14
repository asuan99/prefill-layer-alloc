#!/bin/bash
#SBATCH --job-name=r0c_coord
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# R0c — COORDINATED fixed-split probe (Zamba2-2.7B, in3600/out32).
# Uses pdmux's OWN coordinated pairs (manual_divisions) — decode & prefill on
# exclusive complementary partitions (no independent green-ctx, no contention).
# Tests: (a) true coordinated decode knee, (b) does any prefill-heavy coordinated
# split beat agnostic's dynamic? (precondition for per-type layer-aware to help)
#   arg1 = agn | D  (D = decode_sm; forces coordinated split prefill=108-D / decode=D)
#   arg2 = REP
# NO layer-aware here; pure agnostic policy on a forced coordinated split.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
TRIAGE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r0c
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"

LABEL="${1:-agn}"; REP="${2:-1}"; RATES="${3:-1 2 3 4 6}"
INLEN="${4:-3600}"; OUTLEN="${5:-32}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT=120
WL="in${INLEN}o${OUTLEN}"; FLAB="${LABEL}_${WL}"
PORT=$((34000 + SLURM_JOB_ID % 300))
SRVLOG="$OUT/srv_${FLAB}_rep${REP}_${SLURM_JOB_ID}.log"
BENCHLOG="$OUT/bench_${FLAB}_rep${REP}_${SLURM_JOB_ID}.log"
JSONL="$OUT/zb_${FLAB}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"

# pick pdmux config: forced coordinated split, or default auto (agnostic)
if [ "$LABEL" = "agn" ]; then
  CFG="$TRIAGE/pdmux_a100_smoke.yml"; MAPDESC="agnostic auto (34/54 dynamic)"
else
  D=$LABEL; P=$((108 - D)); CFG="$OUT/.pdmux_coord_d${D}_${SLURM_JOB_ID}.yml"
  printf 'sm_group_num: 3\nmanual_divisions:\n  - [%d, %d, 0]\nsplit_forward_token_budget: 65536\ndecode_bs_divisor: 36\n' "$P" "$D" > "$CFG"
  MAPDESC="coordinated prefill:${P}/decode:${D}"
fi

echo "R0c LABEL=$LABEL REP=$REP split=[$MAPDESC] cfg=$CFG in=$INLEN out=$OUTLEN port=$PORT job=$SLURM_JOB_ID"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "boot_ok=$ok"
if [ "$ok" = 1 ]; then
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' \
    | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^/SANITY /'
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
    --num-prompts 16 --request-rate 2 >/dev/null 2>>"$BENCHLOG" || true
  for R in $RATES; do
    echo "===== $LABEL rep$REP rate=$R ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$JSONL" 2>>"$BENCHLOG" \
      | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT"
  done
else echo "BOOT FAILED"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2

python3 - "$JSONL" "$FLAB" "$REP" "$SLURM_JOB_ID" "$MODEL" "$INLEN" "$OUTLEN" "$OUT" "$MAPDESC" <<'PY'
import json, sys, os
jsonl, lab, rep, jid, model, inlen, outlen, outdir, mapdesc = sys.argv[1:10]
def pct(xs, p):
    if not xs: return float('nan')
    s=sorted(xs); k=(len(s)-1)*p/100.0; f=int(k); c=min(f+1,len(s)-1)
    return s[f] if f==c else s[f]+(s[c]-s[f])*(k-f)
rows=[]
for ln in open(jsonl):
    ln=ln.strip()
    if not ln: continue
    d=json.loads(ln); rr=d.get("request_rate"); tt=d.get("ttfts") or []; il=d.get("itls") or []
    dur=d.get("duration") or 0.0; n=len(tt)
    tpot=[(sum(il[i])/len(il[i])*1000) if (i<len(il) and il[i]) else float('nan') for i in range(n)]
    tpot=[x for x in tpot if x==x]; ttft=[t*1000 for t in tt]
    good=sum(1 for i,t in enumerate(tt) if t<=3.0 and ((sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9)<=0.06)
    thr=(n/dur) if dur else float('nan'); gp=(good/dur) if dur else float('nan')
    rows.append([model,lab,mapdesc.replace(",",";"),rep,jid,inlen,outlen,f"{rr:g}",n,n,
        f"{thr:.3f}",f"{pct(ttft,50):.1f}",f"{pct(ttft,99):.1f}",f"{pct(tpot,50):.2f}",f"{pct(tpot,99):.2f}",
        good,f"{gp:.3f}",f"{dur:.1f}","measured"])
with open(os.path.join(outdir, f"_rows_{lab}_rep{rep}_{jid}.csv"),"w") as f:
    for r in rows: f.write(",".join(str(x) for x in r)+"\n")
print(f"CSV_DONE {lab} rep{rep} [{mapdesc}]: {len(rows)} rows")
for r in rows: print("  rate="+r[7],"thr="+r[10],"ttft50="+r[11],"tpot50="+r[13],"good="+r[15],"gp="+r[16])
PY
echo "DONE_${LABEL}_rep${REP}"
