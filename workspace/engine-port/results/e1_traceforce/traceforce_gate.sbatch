#!/bin/bash
#SBATCH --job-name=tfgate
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/e1_traceforce
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/e1_traceforce/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/e1_traceforce/%x_%j.err
# ============================================================================
# OBSERVER-EFFECT GATE for PDMUX_TRACE_FORCE_PREFILL (engine patch, not a
# claim job -- no goodput, no policy comparison, nothing citable).
#
# Question: does forcing a runtime_snapshot on every prefill-in-flight sync
# (bypassing the count subsampling that makes fast-prefill cells invisible)
# perturb serving latency?
#
# Design: ABBA over server boots (OFF ON ON OFF) per cell, 2 measured probes
# per boot (seeds 11/12) after one discarded warmup probe per boot, so
# per-cell n=4 per condition, paired by seed, position-balanced against
# boot order / triton-cache warmth.  Cells:
#   d16 [92,16] @ rate 4 -- the cell the patch exists for (0 informative
#                           prefill episodes without it)
#   d92 [16,92] @ rate 2 -- the MAX added-volume cell (slow prefill occupies
#                           the most sync calls; predicted +28% emissions vs
#                           +1.2% at d16), i.e. the worst case for overhead
# Rates are the off-cliff values already used by the 866066 sweep (LO=2,
# HI=4); TTFT p99 at d92/rate4 was 2.7s (3s-cliff), hence rate 2 there.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/e1_traceforce
FRONTIER=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_tfgate_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }
grep -q "PDMUX_TRACE_FORCE_PREFILL" \
  /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py \
  || { echo "PATCH_NOT_INSTALLED"; exit 4; }

MODEL="Qwen/Qwen2.5-7B"; BACKEND=triton; ARM=T8
SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN=3600; CTXCAP=$(( CTX + 512 + 256 ))
NP="${NP:-150}"
SEEDS="${SEEDS:-11 12}"
JOBTAG="${SLURM_JOB_ID:-local}"
RESULT="$HERE/tfgate_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
PROBES="$HERE/tfgate_${ARM}_${JOBTAG}_probes.jsonl"; : > "$PROBES"

CELLS="${CELLS:-d16 d92}"
declare -A CELL_RATE=( [d16]=4 [d92]=2 )
FLAG_ORDER="${FLAG_ORDER:-0 1 1 0}"

echo "ARM=$ARM MODEL=$MODEL CELLS=[$CELLS] NP=$NP SEEDS=[$SEEDS] ABBA=[$FLAG_ORDER]" | tee -a "$RESULT"

for NAME in $CELLS; do
  CFG="$FRONTIER/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"; RATE="${CELL_RATE[$NAME]}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME"; continue; }
  BOOT=0
  for FLAG in $FLAG_ORDER; do
    BOOT=$((BOOT+1))
    PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
    RUNID="tfgate_${ARM}_${NAME}_f${FLAG}_b${BOOT}_${JOBTAG}"
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="tfgate_${NAME}_r${RATE}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    export PDMUX_TRACE_FORCE_PREFILL=$FLAG

    echo "=== cell=$NAME boot=$BOOT force=$FLAG dsm=$DSM rate=$RATE port=$PORT ===" | tee -a "$RESULT"
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend $BACKEND \
      --enable-pdmux --pdmux-config-path "$CFG" \
      --disable-piecewise-cuda-graph \
      --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
      --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
      --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
    SRV=$!
    ok=0
    for i in $(seq 1 240); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
      kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME boot=$BOOT" | tee -a "$RESULT"; break; }
      sleep 5
    done
    echo "boot_ok=$ok cell=$NAME boot=$BOOT force=$FLAG" | tee -a "$RESULT"
    [ "$ok" = 1 ] || { pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; continue; }

    # discarded per-boot warmup (cold kernels / allocator), not analyzed
    python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
      --num-prompts 30 --request-rate $RATE --seed 99 --output-details \
      --output-file "$HERE/${RUNID}_warmup.jsonl" > "$HERE/${RUNID}_warmup_bench.log" 2>&1
    sleep 5

    for SEED in $SEEDS; do
      JOUT="$HERE/${RUNID}_s${SEED}.jsonl"; : > "$JOUT"
      T0=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE --seed $SEED --output-details --output-file "$JOUT" \
        > "$HERE/${RUNID}_s${SEED}_bench.log" 2>&1
      RC=$?
      T1=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
      if [ "$RC" != 0 ]; then
        echo "PROBE_FAILED cell=$NAME boot=$BOOT force=$FLAG seed=$SEED rc=$RC" | tee -a "$RESULT"
        continue
      fi
      python3 - "$JOUT" "$NAME" "$FLAG" "$BOOT" "$SEED" "$RATE" "$T0" "$T1" \
        "$PDMUX_TELEMETRY_PATH" >> "$PROBES" <<'PY'
import json, sys
raw, cell, flag, boot, seed, rate, t0, t1, telem = sys.argv[1:10]
d = json.load(open(raw))
ttfts = [t * 1000.0 for t, e in zip(d["ttfts"], d["errors"]) if not e]
itls = [x for r, e in zip(d["itls"], d["errors"]) if not e for x in r]

def pct(a, q):
    if not a: return float("nan")
    a = sorted(a); return a[min(len(a) - 1, int(q * len(a)))]

# telemetry volume actually realized inside this probe's window
n_snap = n_forced = n_pref = 0; drops = 0
lo, hi = float(t0), float(t1)
for line in open(telem):
    line = line.strip()
    if not line: continue
    try: r = json.loads(line)
    except Exception: continue
    if r.get("event") != "runtime_snapshot": continue
    ts = r.get("timestamp_monotonic_s", 0.0)
    if not (lo <= ts <= hi): continue
    n_snap += 1
    if r.get("trace_forced"): n_forced += 1
    if (r.get("prefill_active_batch_size") or 0) > 0: n_pref += 1
    drops = max(drops, r.get("dropped_events", 0) or 0)
print(json.dumps({
    "cell": cell, "force": int(flag), "boot": int(boot), "seed": int(seed),
    "rate": float(rate), "t0": lo, "t1": hi, "dur_s": hi - lo,
    "n_ok": len(ttfts), "n_err": sum(1 for e in d["errors"] if e),
    "ttft_p50": pct(ttfts, .50), "ttft_p95": pct(ttfts, .95), "ttft_p99": pct(ttfts, .99),
    "ttft_mean": sum(ttfts) / max(1, len(ttfts)),
    "itl_p50": pct(itls, .50), "itl_p95": pct(itls, .95), "itl_p99": pct(itls, .99),
    "itl_mean": sum(itls) / max(1, len(itls)),
    "output_throughput": d.get("output_throughput"),
    "n_snapshots": n_snap, "n_forced": n_forced, "n_prefill_active": n_pref,
    "dropped_events": drops,
}))
PY
      tail -1 "$PROBES" | tee -a "$RESULT"
      echo "PIN_CHECK cell=$NAME force=$FLAG boot=$BOOT seed=$SEED:" | tee -a "$RESULT"
      # e1_pin_check.py CLI signature as of the 2026-07-30 bug #6 revision
      # (time-weighted PRIMARY gate): <telemetry> <expect_d> [min_lower95]
      # [raw_jsonl seed rate] --t0 --t1.  min_lower95 is the 3rd positional
      # (it was raw_jsonl in the bug #4/#5 era); this call site kept the stale
      # order and so every PIN_CHECK in job 867298 died in float(argv[2]) on a
      # path string.  Mirror e1_sweep.sbatch:258 -- do not reorder without
      # re-checking e1_pin_check.py's main().
      python3 "$FRONTIER/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" 0.80 "$JOUT" "$SEED" "$RATE" \
        --t0 "$T0" --t1 "$T1" 2>&1 | tee -a "$RESULT"
      sleep 5
    done
    kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 15
  done
done

echo "=== PAIRED SUMMARY ===" | tee -a "$RESULT"
python3 "$HERE/tfgate_analyze.py" "$PROBES" | tee -a "$RESULT"
echo "TFGATE_DONE job=$JOBTAG" | tee -a "$RESULT"
