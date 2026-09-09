#!/bin/bash
#SBATCH --job-name=s1arm
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
# STEP 1 -- ARM-SET REPAIR.  audit_p6_rules_2026-09-09/VERDICT.md "Step 1".
#
# PURE INSTRUMENTATION.  No pre-registered prediction, no decision rule, no
# threshold except "did it boot".  Nothing here can produce a policy verdict,
# an arm ranking, or a performance claim -- by construction there is no rule
# to audit at the rule layer.
#
# Buys exactly three things:
#   (1) BOOT_OK   -- does pdmux_homog6.yml (6 groups, 4 divisions) boot for all
#                    four targets D in {16,44,54,92}?  Same check P3 (job
#                    905701) ran for homog5.
#   (2) MU_P      -- saturated throughput for all four arms UNDER homog6.
#                    D=54 has never been run.  D in {16,44,92} act as CONTROLS
#                    on whether the 5->6 group change shifts mu_p at all
#                    (homog5 anchors, job 905835: 0.933 / 0.675 / 0.187).
#   (3) DIV_COST  -- max_total_num_tokens vs homog5, i.e. the KV cost of the
#                    extra division (P3-C format, job 905710 measured 0.051%
#                    for homog5 vs per-arm configs).
#
# Every cell is deliberately OVERLOADED (rho ~ 1.5-2.1) because saturated
# throughput is the estimand.  TTFT/ITL from this job are NOT operating-point
# measurements and must not be cited as such.
#
# usage: sbatch s1_armfix.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PROBES=$ROOT/workspace/engine-port/results/longctx_conflict/probes
OUT="$PROBES/s1_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$PROBES/pdmux_homog6.yml"
CTX=8192
CTXCAP=$((CTX + 512))
OUTTOK=96
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED=41

# "D rate N"  -- rate chosen so rho ~ 1.5-2.1 against the homog5 anchor
# (d54 uses the extrapolated 0.566, which is exactly what this job tests).
CELLS=( "16 1.40 60" "44 1.20 45" "54 1.20 45" "92 0.35 30" )

echo "STEP1 model=$MODEL backend=$BACKEND cfg=$CFG ctx=$CTX job=$SLURM_JOB_ID"
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
for SPEC in "${CELLS[@]}"; do
  read -r D RATE NP <<< "$SPEC"
  PORT=$((38900 + (SLURM_JOB_ID + D + RANDOM) % 400))
  SRVLOG="$OUT/srv_d${D}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_d${D}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="s1_d${D}_${SLURM_JOB_ID}"
  export PDMUX_WORKLOAD_ID="s1_d${D}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$D

  echo ""
  echo "########## STEP1 D=$D rate=$RATE N=$NP port=$PORT ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 \
    --context-length "$CTXCAP" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 D=$D"
    FAIL_STREAK=0
    # banner + KV size (DIV_COST input)
    grep -m1 -oE "sm_counts[^\n]*" "$SRVLOG" | tee "$OUT/banner_d${D}.txt"
    grep -m1 -oE "max_total_num_tokens=[0-9]+" "$SRVLOG" | tee -a "$OUT/banner_d${D}.txt"

    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts 8 --max-concurrency 1 --seed 7 > "$OUT/warmup_d${D}.log" 2>&1
    echo "warmup_rc=$?"

    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$CTX" --random-output-len "$OUTTOK" --random-range-ratio 1.0 \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
      --output-details --output-file "$OUT/bench_d${D}.jsonl" \
      > "$OUT/bench_d${D}.log" 2>&1
    echo "bench_rc=$? D=$D"
    grep -E "Successful requests|Request throughput|Median TTFT|Median ITL" "$OUT/bench_d${D}.log"
  else
    echo "BOOT_FAILED D=$D" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"
    break
  fi
done

echo ""
echo "=== STEP1 labels (measurement only, no verdict) ==="
python3 - "$OUT" <<'PY'
import json, os, sys, glob
out = sys.argv[1]
# homog5 anchors, job 905835 (out=96, ctx=8192).  Controls: 16/44/92.
ANCHOR = {16: 0.9331, 44: 0.6753, 92: 0.1867}
res = {"probe": "S1_ARMFIX", "cells": {}}
booted = []
for d in (16, 44, 54, 92):
    p = os.path.join(out, f"bench_d{d}.jsonl")
    if not os.path.exists(p):
        res["cells"][d] = {"status": "UNRESOLVED (no bench artifact)"}
        continue
    b = json.loads(open(p).read().strip().split("\n")[0])
    mu = b["request_throughput"]
    row = {"mu_p_saturated_req_per_s": mu,
           "offered_rate": b["request_rate"],
           "ach_over_off": mu / b["request_rate"],
           "completed": b["completed"],
           "n_errors": sum(1 for e in b["errors"] if e != ""),
           "median_itl_ms": b["median_itl_ms"]}
    if d in ANCHOR:
        row["homog5_anchor"] = ANCHOR[d]
        row["ratio_to_anchor"] = mu / ANCHOR[d]
    else:
        row["prior_extrapolation"] = 0.566
        row["ratio_to_extrapolation"] = mu / 0.566
    res["cells"][d] = row
    booted.append(d)

res["boot"] = {"booted": booted,
               "verdict": "BOOT_OK" if len(booted) == 4 else "BOOT_INCOMPLETE"}
# ★spread must be computed over the P7 ARM SET {16,44,54}, not over every
# booted arm.  d92 is booted only as a CONTROL on the 5->6 group change (sec 3
# of RESULT_S1_ARMFIX) and is excluded from P7 by PREREG_P7 sec 2 (registered
# before this job ran; source = audit_p6_rules Step 2 + rev1 improvement #5).
# max/min cannot shrink when an INTERIOR arm is added, so reporting the 4-arm
# spread against a target derived from the 3-arm set is a category error.
# Both are reported; the P7 one is the one the target applies to.
def spread(ds):
    v = [res["cells"][d]["mu_p_saturated_req_per_s"] for d in ds
         if "mu_p_saturated_req_per_s" in res["cells"].get(d, {})]
    return (max(v) / min(v)) if len(v) >= 2 else None
res["mu_p_spread_p7_armset"] = spread([16, 44, 54])
res["mu_p_spread_all_booted"] = spread(booted)
res["mu_p_spread_note"] = ("target <= 2.4x applies to the P7 arm set {16,44,54} only "
                           "(PREREG_P7 sec 2); d92 is a config-change control. "
                           "homog5 arm set {16,44,92} was 4.99x")
res["banners"] = {d: open(os.path.join(out, f"banner_d{d}.txt")).read().strip()
                  for d in (16, 44, 54, 92)
                  if os.path.exists(os.path.join(out, f"banner_d{d}.txt"))}
text = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False, default=str)
print(text)
open(os.path.join(out, "S1_LABEL.json"), "w").write(text)
PY

echo "S1_DONE_${SLURM_JOB_ID}"
