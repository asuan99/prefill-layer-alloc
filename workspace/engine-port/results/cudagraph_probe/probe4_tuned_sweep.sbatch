#!/bin/bash
#SBATCH --job-name=cg_probe4
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
# Probe 4: tuned decode-SM sweep UNDER CUDAGRAPH. Tests whether cudagraph shifts the
# step-fixed optimum split (does starving decode more win now that attn-decode is cheap?).
# If the optimum moves to a smaller decode partition, the "reservation" prefill benefit
# is captured BY TUNED -> no layer-awareness needed (collapse argument).
#   arg1 = REGIME (in3600o32 | in2000o96) ; arg2 = space-sep config basenames (d08 d16 d24 d34 d44 d54)
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
TRIAGE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cudagraph_probe
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT=100
REGIME="${1:-in3600o32}"; CONFIGS="${2:-d08 d16 d24 d34}"
case "$REGIME" in
  in3600o32) INLEN=3600; OUTLEN=32; RATES="2 3 4";;
  in2000o96) INLEN=2000; OUTLEN=96; RATES="2 3 4 6";;
  *) echo "bad regime"; exit 1;;
esac
JSONL="$OUT/jl_probe4_${REGIME}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
unset PDMUX_SLO_SCHED PDMUX_LA_COORD PDMUX_LA_COORD_OPT

find_cfg () { for d in "$OUT" "$SLO"; do [ -f "$d/pdmux_$1.yml" ] && { echo "$d/pdmux_$1.yml"; return; }; done; }

run_cfg () {
  local NAME="$1"; local CFG; CFG=$(find_cfg "$NAME")
  [ -z "$CFG" ] && { echo "MISSING cfg $NAME"; return; }
  local PORT=$((39000 + SLURM_JOB_ID % 100 + RANDOM % 60))
  local SRVLOG="$OUT/srv_p4_${REGIME}_${NAME}_${SLURM_JOB_ID}.log"
  echo "########## SPLIT=$NAME ($(grep -o '\[.*\]' $CFG)) REGIME=$REGIME port=$PORT ##########"
  python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
    --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  local SRV=$!; local ok=0
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "$NAME DIED"; break; }; sleep 5
  done
  echo "$NAME boot_ok=$ok cudagraph_captured=$(grep -c 'Capture cuda graph end' "$SRVLOG")"
  if [ "$ok" = 1 ]; then
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts 8 --request-rate 2 >/dev/null 2>&1 || true
    for R in $RATES; do
      echo "===== $NAME $REGIME rate=$R ====="
      python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
        --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
        --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$JSONL" 2>/dev/null \
        | grep -E "Successful requests|Median TTFT|Median TPOT|P99 TPOT|Median ITL"
    done
  else echo "$NAME BOOT FAILED"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future"; fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
}

echo "PROBE4 regime=$REGIME configs=[$CONFIGS] cudagraph=ON job=$SLURM_JOB_ID"
for C in $CONFIGS; do run_cfg "$C"; done
python3 - "$JSONL" "$REGIME" <<'PY'
import json,sys
jsonl,regime=sys.argv[1:3]
def pct(xs,p):
    if not xs: return float('nan')
    s=sorted(xs);k=(len(s)-1)*p/100;f=int(k);c=min(f+1,len(s)-1)
    return s[f] if f==c else s[f]+(s[c]-s[f])*(k-f)
print(f"--- goodput@SLO rows (order = config x rate), {regime} ---")
for ln in open(jsonl):
    ln=ln.strip()
    if not ln: continue
    d=json.loads(ln); rr=d.get("request_rate"); tt=d.get("ttfts") or []; il=d.get("itls") or []; dur=d.get("duration") or 0
    good=sum(1 for i,t in enumerate(tt) if t<=3.0 and ((sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9)<=0.06)
    gp=good/dur if dur else float('nan'); ttft=[t*1000 for t in tt]
    print(f"  rate={rr:g} ttft50={pct(ttft,50):.0f} good={good} gp={gp:.3f}")
PY
echo "DONE_PROBE4_${REGIME}"
