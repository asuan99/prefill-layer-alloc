#!/bin/bash
#SBATCH --job-name=n9smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/smoke
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/smoke/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/smoke/%x_%j.err
# ============================================================================
# Nemotron-Nano-9B-v2-Base BOOT SMOKE (gate #25/#26).  NOT an experiment.
#
# ★NO SCORED VERDICT.  No rule is applied; nothing is attributed or labelled.
# It answers plumbing questions only:
#   S1  does the model boot in this sglang tree at all (arch recognised)?
#   S2  does it boot WITH --enable-pdmux (green-context split) ?
#   S3  what context length actually fits (131072 requested) ?
#   S4  do requests complete, and does a LONG prompt produce controller
#       decision points (SLO-BIND / SLO-FEAS lines) -- the thing TC1 P3 lacked?
# S4 is OBSERVED, not scored: n=1, no prereg rule is applied to it here.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
export HF_HOME=$ROOT/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"
MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base
# ★S4 수리(2026-08-28): 첫 판본은 컨트롤러를 켜지 않은 채 SLO-BIND/FEAS 라인을
# 세었다 -- 0이 나오는 것이 당연했고 P3의 DEAD와 혼동될 수 있었다.  CTRL=1이면
# 정본 bind+GATE 배선을 그대로 켠다(anchor는 정본 상수 idx 2).
if [ "${CTRL:-0}" = "1" ]; then
  export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding PDMUX_SLO_FEAS_GATE=1 \
         PDMUX_TPOT_SLO_MS=60 PDMUX_TTFT_SLO_MS=3000 PDMUX_SLO_ANCHOR_IDX=2
  echo "CTRL=1 bind+GATE anchor=2"
else
  echo "CTRL=0 plumbing only (컨트롤러 미배선 -- SLO 라인 0은 당연하다)"
fi
CFG=$ROOT/workspace/engine-port/results/slo_sched/pdmux_slo.yml
CTX="${CTX:-131072}"
PORT=$((37100 + SLURM_JOB_ID % 150))
SRV=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/model_roster/smoke/n9srv_${SLURM_JOB_ID}.log
echo "SMOKE model=$MODEL ctx=$CTX port=$PORT job=$SLURM_JOB_ID"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend ${ATTN:-flashinfer} --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRV" 2>&1 &
P=$!; ok=0
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $P 2>/dev/null || { echo "S1/S2 SERVER DIED"; break; }; sleep 5
done
echo "boot_ok=$ok  ctx_requested=$CTX"
if [ "$ok" = 1 ]; then
  echo "== S4a short prompts (TC1 P3 regime: no decision points expected) =="
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len 200 --random-output-len 96 --random-range-ratio 1.0 \
    --num-prompts 40 --request-rate 8 2>&1 | grep -E "Median TTFT|Median TPOT|Successful"
  SW1=$(grep -cE "SLO-BIND|SLO-FEAS refused" "$SRV" 2>/dev/null)
  echo "  ctrl_lines_after_short=$SW1"
  echo "== S4b LONG prompts in=8000 (the regime P3 lacked) =="
  python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
    --dataset-name random-ids --random-input-len 8000 --random-output-len 96 --random-range-ratio 1.0 \
    --num-prompts 24 --request-rate 4 2>&1 | grep -E "Median TTFT|Median TPOT|Successful"
  SW2=$(grep -cE "SLO-BIND|SLO-FEAS refused" "$SRV" 2>/dev/null)
  echo "  ctrl_lines_after_long=$SW2  (delta=$((SW2-SW1)))"
  grep -oE "sm_group_num=[0-9]+|hybrid_override_pattern[^,]{0,20}" "$SRV" 2>/dev/null | sort -u | head -3
else
  echo "BOOT FAILED — tail:"; tail -25 "$SRV" | grep -viE "pynvml|FutureWarning"
fi
kill $P 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "SMOKE_DONE job=$SLURM_JOB_ID"
