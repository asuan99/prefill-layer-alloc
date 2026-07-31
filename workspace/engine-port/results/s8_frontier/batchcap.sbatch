#!/bin/bash
#SBATCH --job-name=e1bcap
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# E1 BATCH-CAP EXPERIMENT -- claims-auditor's minimum decisive experiment
# (2026-07-31 audit of job 867231, item 1 of its priority list).
#
# QUESTION. Is T8's per-cell ITL "ceiling" a property of the MODEL or of the
# HARNESS CONFIGURATION?
#
# The audit found `--max-running-requests 48` set in e1_capacity_scan.sbatch
# and e1_sweep.sbatch while appearing NOWHERE in DESIGN.md -- an unregistered
# constant that on the evidence single-handedly determines the ITL axis, and
# hence the pre-registered ITL-NONBINDING verdict and the whole sec 4.4
# decision rule. Reproduced independently by decode_duty_check.py on 867231:
#
#   cell  max decode_running_batch_size   kv_occupancy there
#   d16   48                              0.0241
#   d44   48                              0.0237
#   d92   38 (prefill admission binds first, never reaches the cap)
#
# kv_occupancy ~0.024 means memory is ~40x from binding: the batch is capped
# by CONFIG, not by resources. And d16's ITL-p95 is flat at 50.2-50.6ms from
# rate 12 all the way to 32 -- an open queue would keep climbing with the
# batch. If the ceilings {d16 50.6, d24 37.9, d44 26.2, d54 23.9, d92 19.1 ms}
# move with the cap, then "the 60ms primary SLO is out of reach for T8" is a
# statement about this harness, not about Qwen2.5-7B, and T8 becomes
# decidable again at a higher cap.
#
# PRE-REGISTERED PREDICTION AND DECISION (written before the run):
#   - If d16's ITL-p95 rises monotonically with the cap and CLEARS 60ms at
#     cap>=96: the ceiling is a configuration property. CONFIRMED. E1 must
#     then either re-site its operating point at a registered cap or report
#     every ITL conclusion scoped to `max_running_requests=48`.
#   - If the ITL-p95 stays flat while the batch demonstrably grows past 48
#     (i.e. the cap really was lifted): the ceiling is NOT the cap, and only
#     then does the "saturated decode step time" reading become CONFIRMED.
#   - If the batch does NOT grow past 48 even with the cap lifted, the binding
#     constraint is elsewhere (admission/KV/arrival) and BOTH readings above
#     are unsupported -- report as inconclusive, do not pick one.
#
# CONTROLLED: exactly ONE knob moves (--max-running-requests). Arm, cells,
# rate, seeds, context, backend, cudagraph setting, mem-fraction and the
# pdmux configs are byte-identical to e1_capacity_scan.sbatch's, and
# PROBE_TARGET_S/NP_MIN/WARMUP_S are pinned to the values job 867231
# ACTUALLY ran with (8/45/3 -- note 8 is NOT DESIGN.md sec 4.2's documented
# 20; 867231 was submitted with an unrecorded override, which the audit
# surfaced, so it is pinned explicitly here rather than inherited).
#
# NOT a claim job. Nothing here is cited without claims-auditor.
# Usage:
#   sbatch batchcap.sbatch
#   sbatch --export=ALL,CELLS="d16 d44",CAPS="48 96 192",RATE=16 batchcap.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_e1bcap_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM="T8"
MODEL="Qwen/Qwen2.5-7B"
BACKEND="triton"

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

CELLS="${CELLS:-d16 d44}"
CAPS="${CAPS:-48 96 192}"
RATE="${RATE:-16}"
# 867231's ACTUAL values, pinned (see header). PROBE_TARGET_S=8 is a
# deliberate match to the comparison run, not DESIGN.md sec 4.2's 20.
PROBE_TARGET_S="${PROBE_TARGET_S:-8}"
NP_MIN="${NP_MIN:-45}"
WARMUP_S="${WARMUP_S:-3}"
# Four seeds, not two: the audit established that --seed changes the PROMPT
# SET (bench_serving.py:1705 seeds stdlib random; datasets/sharegpt.py:98
# consumes it in random.shuffle), so each seed is a distinct workload and an
# n=2 mean estimates nothing. Same seeds across every (cell, cap) so the cap
# comparison stays paired.
SEEDS="${SEEDS:-1 2 3 4}"
JOBTAG="${SLURM_JOB_ID:-local}"

NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")
RESULT="$HERE/e1bcap_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CELLS=[$CELLS] CAPS=[$CAPS] RATE=$RATE NP=$NP SEEDS=[$SEEDS]" | tee -a "$RESULT"
echo "PINNED PROBE_TARGET_S=$PROBE_TARGET_S NP_MIN=$NP_MIN WARMUP_S=$WARMUP_S SGPT_CTXLEN=$SGPT_CTXLEN CTXCAP=$CTXCAP" | tee -a "$RESULT"

for NAME in $CELLS; do
for CAP in $CAPS; do
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG" | tee -a "$RESULT"; continue; }
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="e1bcap_${ARM}_${NAME}_c${CAP}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="batchcap_${ARM}_${NAME}_c${CAP}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
  # Main sweep runs force-trace OFF (DESIGN.md sec 4.7.1, job 867298's
  # asymmetric itl_p95 perturbation at d92). This experiment's headline IS an
  # ITL comparison, so it inherits that: OFF.
  export PDMUX_TRACE_FORCE_PREFILL=0

  echo "=== ARM=$ARM cell=$NAME target_decode_sm=$DSM max_running_requests=$CAP port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend $BACKEND \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests $CAP --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME cap=$CAP" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok cell=$NAME cap=$CAP" | tee -a "$RESULT"

  # Confirm the server ACTUALLY took the cap rather than trusting the flag --
  # methodology gate: verify with instrumentation, never by assertion (this is
  # the discipline whose absence produced Stage 0's D108 anchor failure).
  curl -s "http://127.0.0.1:$PORT/get_server_info" 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print(f'SERVER_INFO_UNPARSEABLE: {e}'); raise SystemExit
def dig(o,k):
    if isinstance(o,dict):
        if k in o: return o[k]
        for v in o.values():
            r=dig(v,k)
            if r is not None: return r
    return None
print(f'SERVER_INFO_CHECK max_running_requests={dig(d,\"max_running_requests\")} '
      f'(requested $CAP)')" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for SEED in $SEEDS; do
      JOUT="$HERE/${RUNID}_s${SEED}.jsonl"; : > "$JOUT"
      T0=$(python3 -c "import time; print(time.monotonic())")
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE --seed $SEED --output-details --output-file "$JOUT" \
        > "$HERE/${RUNID}_s${SEED}_bench.log" 2>&1
      BENCH_RC=$?
      T1=$(python3 -c "import time; print(time.monotonic())")
      if [ "$BENCH_RC" != 0 ]; then
        echo "PROBE_FAILED cell=$NAME cap=$CAP seed=$SEED rc=$BENCH_RC" | tee -a "$RESULT"
        continue
      fi
      python3 - "$JOUT" "$NAME" "$CAP" "$RATE" "$WARMUP_S" "$SEED" <<'PY' | tee -a "$RESULT"
import json, sys

def pct(a, q):
    if not a: return float("nan")
    a = sorted(a)
    pos = q * (len(a) - 1); lo = int(pos); hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)

path, name, cap_s, rate_s, warm_s, seed_s = sys.argv[1:7]
rate, warm, seed, cap = float(rate_s), float(warm_s), int(seed_s), int(cap_s)
o = json.loads(open(path).read().strip().splitlines()[0])
tt, il, er = o.get("ttfts") or [], o.get("itls") or [], o.get("errors") or []
# Pre-registered warmup discard in DISPATCH order (DESIGN.md sec 4.2.2),
# identical convention to e1_analyze.py's scan mode.
import math
nw = math.ceil(warm * rate)
ttk = [t * 1000.0 for i, t in enumerate(tt) if i >= nw and (er[i] if i < len(er) else "") == ""]
ilk = []
for i, seq in enumerate(il):
    if i < nw: continue
    if (er[i] if i < len(er) else "") != "": continue
    ms = [x * 1000.0 for x in seq]
    if ms: ilk.append(pct(ms, 0.95))
print(f"BCAP cell={name} cap={cap} seed={seed} n={len(tt)} warmup_n={nw} "
      f"n_steady={len(ttk)} errs={sum(1 for e in er if e)} "
      f"TTFT_ms p50/p95={pct(ttk,.50):.0f}/{pct(ttk,.95):.0f} "
      f"reqITLp95_ms p50/p95={pct(ilk,.50):.1f}/{pct(ilk,.95):.1f} "
      f"frac_itl_le_50={sum(1 for x in ilk if x<=50)/max(1,len(ilk)):.3f} "
      f"frac_itl_le_60={sum(1 for x in ilk if x<=60)/max(1,len(ilk)):.3f} "
      f"thr={o.get('output_throughput')}")
PY
      echo "BCAP_DUTY cell=$NAME cap=$CAP seed=$SEED:" | tee -a "$RESULT"
      python3 "$HERE/decode_duty_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" \
        --t0 "$T0" --t1 "$T1" 2>&1 | tee -a "$RESULT"
      sleep 5
    done
    echo "BCAP_PIN cell=$NAME cap=$CAP (whole-file prefill pin, diagnostic):" | tee -a "$RESULT"
    python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" 2>&1 | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME cap=$CAP" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10
done
done
echo "BCAP_DONE ARM=$ARM job=$JOBTAG" | tee -a "$RESULT"
rm -rf "$TRITON_CACHE_DIR"
