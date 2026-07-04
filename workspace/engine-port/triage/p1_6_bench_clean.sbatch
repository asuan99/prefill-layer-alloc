#!/bin/bash
#SBATCH --job-name=p1_6_bench_clean
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --comment=pytorch
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
PORT=$((31600 + SLURM_JOB_ID % 400))
LAFILE="$HERE/.clean_mode_${SLURM_JOB_ID}"
RESULT="$HERE/p1_6_bench_clean_${SLURM_JOB_ID}.txt"; : > "$RESULT"
# server saturates ~4.3 req/s (410 tok/s / 96 out) -> sweep BELOW + at + past saturation
RATES="1 2 3 4 6"
NPROMPT=120; INLEN=2000; OUTLEN=96

# goodput@SLO from bench_serving's per-request jsonl dump (ttfts + itls)
gp_py='
import json,sys
slo_ttft,slo_tpot=3.0,0.06
for ln in open(sys.argv[1]):
    d=json.loads(ln)
    rr=d.get("request_rate"); ttfts=d.get("ttfts") or []; itls=d.get("itls") or []
    dur=d.get("duration") or 0.0; comp=d.get("completed") or len(ttfts)
    good=0
    for i,tt in enumerate(ttfts):
        il=itls[i] if i<len(itls) else []
        tpot=(sum(il)/len(il)) if il else 0.0
        if tt<=slo_ttft and tpot<=slo_tpot: good+=1
    gp=good/dur if dur>0 else 0.0
    print(f"    rate={rr} completed={comp} good={good}/{len(ttfts)} goodput@SLO={gp:.2f}req/s")
'

run_policy () {   # $1 = fused | agnostic | agnostic_v2 | layer_aware
  local POL="$1"; local SRVLOG="$HERE/p1_6_srv_${SLURM_JOB_ID}_${POL}.log"
  local JSONL="$HERE/p1_6_${POL}_${SLURM_JOB_ID}.jsonl"; : > "$JSONL"
  local EXTRA=""
  ( unset PDMUX_FIXED_DECODE_SM_FILE PDMUX_LA_RESERVE PDMUX_LA_FLOOR_SM PDMUX_AGN2_SM
    [ "$POL" != "fused" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"
    if [ "$POL" = "layer_aware" ]; then echo layer_aware > "$LAFILE"; export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_LA_RESERVE="M-" PDMUX_LA_FLOOR_SM=16
    elif [ "$POL" = "agnostic_v2" ]; then echo agnostic_v2 > "$LAFILE"; export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_AGN2_SM=16; fi
    python -m sglang.launch_server \
      --model-path nvidia/Nemotron-H-8B-Base-8K --trust-remote-code --dtype bfloat16 \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
      --mem-fraction-static 0.82 --max-running-requests 48 --context-length 8192 \
      $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 ) &
  local SRV=$!
  local ok=0
  for i in $(seq 1 160); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 5
  done
  echo "===== $POL =====" | tee -a "$RESULT"
  if [ "$ok" = 1 ]; then
    # warmup
    python -m sglang.bench_serving --backend sglang --model nvidia/Nemotron-H-8B-Base-8K --host 127.0.0.1 --port $PORT \
      --dataset-name random --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
      --num-prompts 24 --request-rate 2 >/dev/null 2>&1 || true
    for R in $RATES; do
      python -m sglang.bench_serving --backend sglang --model nvidia/Nemotron-H-8B-Base-8K --host 127.0.0.1 --port $PORT \
        --dataset-name random --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 1.0 \
        --num-prompts $NPROMPT --request-rate $R --output-details --output-file "$JSONL" 2>/dev/null \
        | grep -E "Request rate|Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT|Median E2E" \
        | sed "s/^/  [$POL r$R] /" | tee -a "$RESULT"
    done
    echo "  --- goodput@SLO (TTFT<=3s, TPOT<=60ms) from per-request dump ---" | tee -a "$RESULT"
    python3 -c "$gp_py" "$JSONL" | tee -a "$RESULT"
  else
    echo "  SERVER_FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" >> "$RESULT"
  fi
  kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null; sleep 2
}

echo "===== NemotronH CLEAN re-measure (sglang.bench_serving async client, sub-saturation sweep) =====" | tee -a "$RESULT"
echo "  in=$INLEN out=$OUTLEN nprompts=$NPROMPT rates=[$RATES]  (server saturates ~4.3 req/s)" | tee -a "$RESULT"
run_policy fused
run_policy agnostic
run_policy agnostic_v2
run_policy layer_aware
echo "===== DONE ====="; cat "$RESULT"
