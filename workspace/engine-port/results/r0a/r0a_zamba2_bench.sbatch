#!/bin/bash
#SBATCH --job-name=r0a_zb_bench
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
# R0a — Zamba2-2.7B clean async re-measure (replaces old GIL-client §04 panel).
# Client = python -m sglang.bench_serving (async, random-ids).  NO bench_client.py.
# Workload in3600/out32 (matches old measurement for comparability), triton, ctx 4096.
# Policies fused / agnostic / layer_aware (confirmed flags, see r0a NOTES).
# One policy per job, sweeps all rates.  New outputs ONLY under results/r0a/.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
TRIAGE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r0a
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"     # reuse warm triton cache (read/append only)

POL="${1:-agnostic}"
REP="${2:-1}"
RATES="${3:-1 2 3 4 6}"
MODEL=Zyphra/Zamba2-2.7B
INLEN=3600; OUTLEN=32; CTX=4096; NPROMPT=120
PORT=$((33200 + SLURM_JOB_ID % 300))     # disjoint from p1_4_zb(32000-499)/p1_7(32600-899)/p1_6(31600-999)
LAFILE="$OUT/.r0a_mode_${SLURM_JOB_ID}"
SRVLOG="$OUT/srv_${POL}_rep${REP}_${SLURM_JOB_ID}.log"
BENCHLOG="$OUT/bench_${POL}_rep${REP}_${SLURM_JOB_ID}.log"
JSONL="$OUT/zamba2_async_${POL}_rep${REP}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"

# ---- policy flags (confirmed: zamba2.py L360-403; notes.md:173; p1_7_bench_one.sbatch) ----
EXTRA=""
if [ "$POL" != "fused" ]; then
  EXTRA="--enable-pdmux --pdmux-config-path $TRIAGE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"
fi
if [ "$POL" = "layer_aware" ]; then
  echo layer_aware > "$LAFILE"
  export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_LA_FLOOR_SM=16   # Zamba2 reserves by layer-type; PDMUX_LA_RESERVE not read
fi

echo "R0a MODEL=$MODEL POLICY=$POL REP=$REP in=$INLEN out=$OUTLEN ctx=$CTX rates='$RATES' port=$PORT job=$SLURM_JOB_ID"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
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
  # warmup (not recorded)
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
    --num-prompts 16 --request-rate 2 >/dev/null 2>>"$BENCHLOG" || true
  for R in $RATES; do
    echo "===== $POL rep$REP rate=$R ====="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$JSONL" 2>>"$BENCHLOG" \
      | grep -E "Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E"
  done
else
  echo "BOOT FAILED"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future"
fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2

# ---- post-process JSONL -> per-run CSVs + per-job rows file (merged later; no shared-file races) ----
python3 - "$JSONL" "$POL" "$REP" "$SLURM_JOB_ID" "$MODEL" "$INLEN" "$OUTLEN" "$OUT" <<'PY'
import json, sys, os
jsonl, pol, rep, jid, model, inlen, outlen, outdir = sys.argv[1:9]
def pct(xs, p):
    if not xs: return float('nan')
    s=sorted(xs); k=(len(s)-1)*p/100.0; f=int(k); c=min(f+1,len(s)-1)
    return s[f] if f==c else s[f]+(s[c]-s[f])*(k-f)
rows=[]
hdr=("model,policy,rep,job_id,in_len,out_len,rate,n_prompts,n_success,"
     "throughput_reqs_per_s,ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,"
     "good_count,goodput_reqs_per_s,duration_s,label")
for ln in open(jsonl):
    ln=ln.strip()
    if not ln: continue
    d=json.loads(ln)
    rr=d.get("request_rate"); tt=d.get("ttfts") or []; il=d.get("itls") or []
    dur=d.get("duration") or 0.0
    n=len(tt)
    # per-request TPOT = mean inter-token latency (s); goodput uses TTFT<=3s AND TPOT<=60ms
    tpots=[(sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else float('nan') for i in range(n)]
    ttft_ms=[t*1000 for t in tt]
    tpot_ms=[t*1000 for t in tpots if t==t]
    good=sum(1 for i,t in enumerate(tt) if t<=3.0 and ((sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9)<=0.06)
    thr=(n/dur) if dur else float('nan'); gp=(good/dur) if dur else float('nan')
    row=[model,pol,rep,jid,inlen,outlen,
         (f"{rr:g}" if isinstance(rr,(int,float)) else rr), n, n,
         f"{thr:.3f}", f"{pct(ttft_ms,50):.1f}", f"{pct(ttft_ms,99):.1f}",
         f"{pct(tpot_ms,50):.2f}", f"{pct(tpot_ms,99):.2f}",
         good, f"{gp:.3f}", f"{dur:.1f}", "measured"]
    rows.append([str(x) for x in row])
    # per-run CSV
    rpath=os.path.join(outdir, f"zamba2_async_{pol}_r{rr:g}_rep{rep}.csv" if isinstance(rr,(int,float)) else f"zamba2_async_{pol}_r{rr}_rep{rep}.csv")
    with open(rpath,"w") as f:
        f.write(hdr+"\n"); f.write(",".join(str(x) for x in row)+"\n")
# per-job rows file (no header) for later merge into r0a_summary.csv
with open(os.path.join(outdir, f"_rows_{pol}_rep{rep}_{jid}.csv"),"w") as f:
    for r in rows: f.write(",".join(r)+"\n")
print(f"CSV_DONE {pol} rep{rep}: {len(rows)} rate-rows")
for r in rows:
    print("  ", ",".join(r[6:16]))
PY
echo "DONE_${POL}_rep${REP}"
