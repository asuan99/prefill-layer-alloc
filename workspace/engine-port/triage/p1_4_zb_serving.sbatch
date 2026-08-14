#!/bin/bash
#SBATCH --job-name=p1_4_zb_serving
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:45:00
#SBATCH --comment="field=efficientai;appl=pytorch"
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
PORT=$((32000 + SLURM_JOB_ID % 500))   # disjoint range from p1_4_serving (NemotronH) so cross-model concurrent runs don't collide
LAFILE="$HERE/.zb_la_mode_${SLURM_JOB_ID}"; echo layer_aware > "$LAFILE"
RESULT="$HERE/p1_4_zb_serving_${SLURM_JOB_ID}.txt"; : > "$RESULT"
INLEN=3600; OUTLEN=32; CTX=4096   # prefill-bound regime: short output so decode isn't the bottleneck
TPOTSLO=0.12   # triton-no-cudagraph decode floors TPOT ~76ms; 60ms SLO is unreachable by ANY policy
              # (harness artifact, not a policy diff). Match SLO to the backend floor so TTFT differentiates.

run_policy () {   # $1 = fused | agnostic | agnostic_v2 | layer_aware
  local POL="$1"; local SRVLOG="$HERE/p1_4_zbsrv_${SLURM_JOB_ID}_${POL}.log"
  local EXTRA=""
  ( unset PDMUX_FIXED_DECODE_SM_FILE PDMUX_LA_FLOOR_SM PDMUX_AGN2_SM
    [ "$POL" != "fused" ] && EXTRA="--enable-pdmux --pdmux-config-path $HERE/pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule"
    if [ "$POL" = "layer_aware" ]; then
      echo layer_aware > "$LAFILE"
      export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_LA_FLOOR_SM=16
    elif [ "$POL" = "agnostic_v2" ]; then
      echo agnostic_v2 > "$LAFILE"
      export PDMUX_FIXED_DECODE_SM_FILE="$LAFILE" PDMUX_AGN2_SM=16
    fi
    python -m sglang.launch_server \
      --model-path Zyphra/Zamba2-2.7B --trust-remote-code --dtype bfloat16 \
      --attention-backend triton \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
      --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
      $EXTRA --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 ) &
  local SRV=$!
  local ok=0
  for i in $(seq 1 160); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break; sleep 5
  done
  if [ "$ok" = 1 ]; then
    python "$HERE/bench_client.py" --port $PORT --rate 4 --num 24 --in-len $INLEN --out-len $OUTLEN --tpot-slo $TPOTSLO --tag "$POL-warmup" >/dev/null 2>&1 || true
    for RATE in 6 12 18; do
      python "$HERE/bench_client.py" --port $PORT --rate $RATE --num 72 --in-len $INLEN --out-len $OUTLEN --tpot-slo $TPOTSLO --tag "$POL" | tee -a "$RESULT"
    done
    curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' | grep -oE '"text":"[^"]*"' | head -1 | sed "s/^/SANITY $POL /" | tee -a "$RESULT"
  else
    echo "BENCH $POL SERVER_FAILED" | tee -a "$RESULT"; tail -25 "$SRVLOG" >> "$RESULT"
  fi
  kill $SRV 2>/dev/null; sleep 3; kill -9 $SRV 2>/dev/null; sleep 2
}

echo "===== Zamba2-2.7B serving (prefill-bound): fused vs agnostic vs agnostic_v2 vs layer_aware, in=$INLEN/out=$OUTLEN tpot_slo=${TPOTSLO}s triton =====" | tee -a "$RESULT"
run_policy fused
run_policy agnostic
run_policy agnostic_v2
run_policy layer_aware
echo "===== DONE ====="; cat "$RESULT"
