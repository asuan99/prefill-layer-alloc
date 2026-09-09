#!/bin/bash
#SBATCH --job-name=p2floor
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/%x_%j.err
#
# P2 FLOOR -- PREREG_PROBES_2026-09-08.md sec 2.  Measures floor_ref(ctx) and
# itl_solo DIRECTLY instead of extrapolating from a different model.  Every
# `RATIO` cell coordinate/arrival-rate/SLO-threshold/budget to date sat on
# extrapolated floors from a different model; this buys the real numbers.
#
# Concurrency 1 == "--max-concurrency 1" with --request-rate left at its
# default (inf, i.e. all "arrive" at t=0 but the semaphore serializes
# execution to exactly one in flight) -- true sequential submission.
# 4 boots, one per ctx in {1024,4096,8192,16384}, all d44 (single-division,
# PDMUX_R2_FIXED_DSM=44; L-10 predicts NO split fires at concurrency 1 either
# way, so the arm choice does not matter for this probe -- prereg sec 2,
# "동시성 1에서는 분할이 발화하지 않으므로 이 값은 arm 무관 물리량이다").
#
# No policy verdict.  Registered check ONLY: is floor_ref within +/-30% of
# the prior extrapolated values (0.13 / 0.49 / 0.96 / 1.90 s for ctx
# 1024/4096/8192/16384)?  Outside -> RATIO's cell coordinates/k/budget must
# be recomputed (prereg sec 2) -- that is itself the deliverable, not a
# judgement about which is "right".
#
# usage: sbatch p2_floor.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
STAGE0=$ROOT/workspace/engine-port/results/stage0_xctrl
OUT="$PROBES/p2_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$STAGE0/pdmux_d44.yml"
OUTTOK=96
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED=23
NP=22

echo "P2 model=$MODEL backend=$BACKEND cfg=$CFG out=$OUTTOK np=$NP job=$SLURM_JOB_ID"
echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

FAIL_STREAK=0
for CTX in 1024 4096 8192 16384; do
  CTXCAP=$((CTX + 512))
  PORT=$((39700 + (SLURM_JOB_ID + CTX + RANDOM) % 400))
  SRVLOG="$OUT/srv_c${CTX}.log"
  RUNID="p2_c${CTX}_${SLURM_JOB_ID}"
  export PDMUX_TELEMETRY_PATH="$OUT/p2_c${CTX}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"
  export PDMUX_WORKLOAD_ID="p2_c${CTX}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=44

  echo ""
  echo "########## P2 ctx=$CTX port=$PORT ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 ctx=$CTX"
    FAIL_STREAK=0
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts "$NP" --max-concurrency 1 --seed "$SEED" \
      --output-details --output-file "$OUT/bench_c${CTX}.jsonl" \
      > "$OUT/bench_c${CTX}.log" 2>&1
    BRC=$?
    echo "bench_rc=$BRC ctx=$CTX"
    grep -E "Successful requests|Median TTFT|Median ITL|Concurrency:" "$OUT/bench_c${CTX}.log"
    python3 "$PROBES/p2_floor_analyze.py" "$CTX" "$OUT/bench_c${CTX}.jsonl" \
      --warmup 2 --win-lo 9 --win-hi 32 --out "$OUT/floor_c${CTX}.json" \
      || echo "P2_ANALYZE_FAILED ctx=$CTX"
  else
    echo "BOOT_FAILED ctx=$CTX" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES (순서·중단 규칙 #3)" | tee -a "$OUT/BOOT_FAILURES.txt"
    break
  fi
done

echo "=== P2 registered +/-30% check (calibration bookkeeping, NOT a verdict) ==="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
# prereg sec 2: prior extrapolated floor_ref values, aligned to the registered ctx list.
PRIOR = {1024: 0.13, 4096: 0.49, 8192: 0.96, 16384: 1.90}
rows = {}
for ctx in (1024, 4096, 8192, 16384):
    p = os.path.join(out, f"floor_c{ctx}.json")
    if not os.path.exists(p):
        rows[ctx] = {"status": "MISSING"}
        continue
    d = json.load(open(p))
    floor = d.get("floor_ref_ttft_p50_s")
    prior = PRIOR[ctx]
    within_30 = (abs(floor - prior) / prior) <= 0.30 if floor == floor else None
    rows[ctx] = {"floor_ref_ttft_p50_s": floor, "itl_solo_p50_s": d.get("itl_solo_p50_s"),
                 "prior_extrapolated_s": prior, "within_30pct_of_prior": within_30,
                 "input_lens_realized": d.get("input_lens_realized")}
result = {"probe": "P2_FLOOR", "cells": rows,
          "any_outside_30pct": any(r.get("within_30pct_of_prior") is False for r in rows.values()),
          "any_missing": any(r.get("status") == "MISSING" for r in rows.values())}
json.dump(result, open(os.path.join(out, "P2_LABEL.json"), "w"), indent=2, sort_keys=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "P2_DONE_${SLURM_JOB_ID}"
