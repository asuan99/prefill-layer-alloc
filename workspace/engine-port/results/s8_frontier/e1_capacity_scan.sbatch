#!/bin/bash
#SBATCH --job-name=e1capscan
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# E1 CAPACITY PRE-SCAN -- required before e1_sweep.sbatch (DESIGN.md sec 4,
# methodology gate #6: never measure at a TTFT~SLO metric cliff, pick the
# offered rate from a measured off-cliff band, not by assumption).
#
# For each (arm, D-cell) this boots the server ONCE and fires a short
# ShareGPT open-loop probe (sglang.bench_serving, same tool the main sweep
# uses) at each candidate rate in RATES, in increasing order, reporting raw
# TTFT/ITL percentiles (warmup-discarded, DESIGN.md sec 4.2.2) + two
# DISTINCT rate numbers -- arrival_rps (validity check) and completion_rps
# (tail-diluted for short probes, do not read alone) -- + error rate per
# probe (DESIGN.md sec 4.2.1, 2026-07-29 revision: the old single
# `achieved_rps` was tail-diluted and produced a physically-impossible TTFT
# reversal at 866868, see that section for the full diagnosis). No
# goodput/SLO judgement is computed here on purpose -- SLO_TTFT_MS/
# SLO_ITL_P95_MS are themselves not yet chosen (DESIGN.md sec 4.3). ELBOW-
# FINDING PRIMARY SIGNAL = the TTFT/ITL percentile TREND across rates, NOT
# either throughput number (DESIGN.md sec 4.2.1) -- a human (or
# result-analyst) reads the printed table, finds where that trend stops
# being flat, and picks RATE_LO/RATE_HI for e1_sweep.sbatch comfortably
# below it -- same "locate cliff" logic g2_0_decliff/g2_0_pre used, adapted
# to a per-(arm,D) grid because CAPACITY DIFFERS BY D (a decode-starved
# cell like D16 has lower decode throughput; a prefill-starved cell like
# D92 has lower prefill throughput -- see DESIGN.md sec 4.2 "different
# arms/cells may have different off-cliff bands").
#
# NOT a claim job, no goodput computed, nothing here is cited without
# claims-auditor. Usage:
#   sbatch --export=ALL,ARM_IDS="3",CELLS="d16 d24 d44 d54 d92" e1_capacity_scan.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_e1cap_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM_LABELS=(M8 Ha8 Hs8 T8)
ARM_MODELS=(
  "$ROOT/hf_cache/mamba-codestral-7b-sglang"
  "Zyphra/Zamba2-7B-Instruct"
  "nvidia/Nemotron-H-8B-Base-8K"
  "Qwen/Qwen2.5-7B"
)
# Same unavoidable backend asymmetry as s8_scaleup/s8p_prefill.
ARM_BACKENDS=(triton triton flashinfer triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

ARM_IDS="${ARM_IDS:-0 1 2 3}"
CELLS="${CELLS:-d16 d24 d44 d54 d92}"
RATES="${RATES:-1 2 3 4 6 8 12 16}"
PROBE_TARGET_S="${PROBE_TARGET_S:-20}"   # aim each probe's ARRIVAL window at
                                          # roughly this many seconds (NOT the
                                          # measured completion duration, which
                                          # runs longer due to the tail --
                                          # DESIGN.md sec 4.2.1, 2026-07-29 revision)
# NP_MIN raised 15->45 2026-07-29 (866868 diagnosis): the relative std of a
# Poisson arrival span over n draws is ~1/sqrt(n), RATE-INDEPENDENT (mean and
# std of the sum both scale as 1/rate, the ratio cancels). n=45 targets
# ~15% relative std (1/sqrt(45)=0.149) regardless of which RATE is being
# probed -- this is a sample-count floor, not a duration floor, and
# PROBE_TARGET_S alone does not guarantee it at low rates (n=15 gave a
# realized arrival span of 5.5s vs an expected 9.5s for rate=2 in 866868 --
# a ~42% deviation from nominal, see DESIGN.md sec 4.2.1 for the full
# reconstruction).
NP_MIN="${NP_MIN:-45}"
WARMUP_S="${WARMUP_S:-3}"                # pre-registered warmup discard
                                          # (DESIGN.md sec 4.2.2): the first
                                          # WARMUP_N=ceil(WARMUP_S*RATE)
                                          # dispatch-order requests are
                                          # excluded from TTFT/ITL percentiles
                                          # (866868 found TTFT p50 at rate=2
                                          # HIGHER than at rate=4, a cold-start
                                          # transient in a 20-request probe,
                                          # not a real capacity signal)
SEEDS="${SEEDS:-1 2}"                    # ★2026-07-29 (coordinator directive,
                                          # sec 4.5/item 4): every rate is
                                          # probed at >=2 seeds (default 1,2 --
                                          # 1 for continuity with 866868/all
                                          # prior smokes, which used
                                          # bench_serving's implicit default).
                                          # A single arrival realization is
                                          # not enough to trust an elbow
                                          # location; the two probes' TTFT/
                                          # arrival_rps are diffed below and
                                          # flagged if they diverge >25%
                                          # (relative, TTFT p50) -- that rate
                                          # is then an unstable region for
                                          # off-cliff siting, not a clean
                                          # single-realization read.
JOBTAG="${SLURM_JOB_ID:-local}"

for AIDX in $ARM_IDS; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/e1cap_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND CELLS=[$CELLS] RATES=[$RATES]" | tee -a "$RESULT"

for NAME in $CELLS; do
  CFG="$HERE/pdmux_e1_${NAME}.yml"
  DSM="${NAME#d}"
  [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG"; continue; }
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  RUNID="e1cap_${ARM}_${NAME}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="capscan_${ARM}_${NAME}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM

  echo "=== ARM=$ARM cell=$NAME target_decode_sm=$DSM cfg=$(basename $CFG) port=$PORT ===" | tee -a "$RESULT"
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
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok cell=$NAME" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    for RATE in $RATES; do
      NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")
      SEED_JOUTS=()
      for SEED in $SEEDS; do
      JOUT="$HERE/${RUNID}_r${RATE}_s${SEED}.jsonl"; : > "$JOUT"
      SEED_JOUTS+=("$JOUT")
      python -m sglang.bench_serving --backend sglang --model "$MODEL" --host 127.0.0.1 --port $PORT \
        --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
        --num-prompts $NP --request-rate $RATE --seed $SEED --output-details --output-file "$JOUT" \
        > "$HERE/${RUNID}_r${RATE}_s${SEED}_bench.log" 2>&1
      BENCH_RC=$?
      if [ "$BENCH_RC" != 0 ]; then
        echo "PROBE_FAILED cell=$NAME rate=$RATE seed=$SEED rc=$BENCH_RC (see ${RUNID}_r${RATE}_s${SEED}_bench.log)" | tee -a "$RESULT"
        continue
      fi
      python3 - "$JOUT" "$NAME" "$RATE" "$NP" "$WARMUP_S" "$SEED" <<'PY' | tee -a "$RESULT"
import json, sys
import numpy as np

def pct(a, q):
    if not a: return float("nan")
    a = sorted(a); return a[min(len(a) - 1, int(q * len(a)))]

name, rate_s, np_s, warmup_s_s, seed_s = sys.argv[2:7]
rate, npc, warmup_s, seed = float(rate_s), int(np_s), float(warmup_s_s), int(seed_s)

# --- ARRIVAL-time reconstruction via seeded RNG replay (DESIGN.md sec 4.2.1,
# now driven by the ACTUAL --seed passed to this probe, sec 4.5 2026-07-29 --
# every probe now uses an EXPLICIT seed, no longer the implicit bench_serving
# default). sglang.bench_serving draws inter-arrival intervals from
# np.random.exponential(1/rate) inside get_request() (bench_serving.py:948),
# request i (0-indexed, DISPATCH order) arriving at cumsum(intervals[0:i]).
# np.random.seed(args.seed) is set once near the top of run_benchmark
# (bench_serving.py:1706) BEFORE dataset sampling; for --dataset-name
# sharegpt specifically, dataset prep only calls Python's stdlib
# random.shuffle (sglang/benchmark/datasets/sharegpt.py:98), NOT np.random,
# so the np.random stream is untouched between the seed call and the first
# get_request() draw -- this replay is exact FOR THIS (sharegpt) dataset.
# VALIDATED (2026-07-29, CPU-only, no server needed): called sglang.bench_serving's
# own get_request() directly against dummy inputs and measured actual
# asyncio-scheduled dispatch wall-clock times for seed in {1,2,7} x rate in
# {2,8} -- max abs error vs this replay was 2-70ms out of several seconds
# (event-loop scheduling jitter), confirming the replay generalizes to ANY
# seed, not just the previously-only-tested default (seed=1). FRAGILE: if
# the dataset/backend changes to one that DOES consume np.random during prep
# (e.g. --dataset-name random-ids, sglang/benchmark/datasets/random.py:147),
# this reconstruction silently desyncs -- re-verify against the then-current
# sglang commit before reusing this block for a different dataset/backend.
np.random.seed(seed)
intervals = [np.random.exponential(1.0 / rate) for _ in range(npc)]
arrival = np.cumsum([0.0] + intervals[:-1])  # arrival[i] = dispatch time of request i
arrival_span_s = float(arrival[-1]) if npc > 1 else float("nan")
arrival_rps = (npc - 1) / arrival_span_s if arrival_span_s else float("nan")

warmup_n = max(0, int(np.ceil(warmup_s * rate)))

tt, il, errs, n = [], [], 0, 0
dur = 0.0
for ln in open(sys.argv[1]):
    ln = ln.strip()
    if not ln:
        continue
    o = json.loads(ln)
    T = o.get("ttfts") or []; I = o.get("itls") or []; E = o.get("errors") or []
    dur += o.get("duration") or 0.0
    for i, t in enumerate(T):
        n += 1
        err = E[i] if i < len(E) else ""
        if err:
            errs += 1
            continue
        if i < warmup_n:
            continue   # pre-registered warmup discard (DESIGN.md sec 4.2.2) -- excluded from percentiles only, still counted in n/errs above
        m = I[i] if i < len(I) and I[i] else []
        p95 = pct([x * 1000.0 for x in m], 0.95) if m else float("nan")
        tt.append(t); il.append(p95)
il = [x for x in il if x == x]
n_steady = max(0, n - errs - warmup_n)
# ★2026-07-29 (coordinator directive, sec 4.5/item 5): arrival_rps leads --
# it is the label that should be trusted for cross-probe comparison/x-axis,
# nominal_rate is kept only as the human-readable reference the operator
# requested this probe at.
print(f"CAPSCAN cell={name} arrival_rps={arrival_rps:.3f} nominal_rate={rate} seed={seed} "
      f"n={n} errs={errs} warmup_n={warmup_n} n_steady={n_steady} "
      f"dur={dur:.1f}s arrival_span_s={arrival_span_s:.1f} "
      f"completion_rps={((n-errs)/dur if dur else 0):.2f} "
      f"TTFT_ms(post-warmup) p50/p95/p99={pct(tt,.5)*1000:.0f}/{pct(tt,.95)*1000:.0f}/{pct(tt,.99)*1000:.0f} "
      f"ITLp95_ms(post-warmup) p50/p95={pct(il,.5):.1f}/{pct(il,.95):.1f}")
# Machine-readable line the seed-divergence step (below) parses.
print(f"CAPSCAN_JSON {json.dumps(dict(cell=name, nominal_rate=rate, seed=seed, arrival_rps=arrival_rps, ttft_p50=pct(tt,.5)*1000, ttft_p95=pct(tt,.95)*1000, itl_p50=pct(il,.5), n_steady=n_steady))}")
print("NOTE: arrival_rps is the TRUSTED label (item 5) -- it is a VALIDITY "
      "check that should track nominal_rate closely (open-loop dispatch "
      "does not wait on the server); a large deviation flags small-n "
      "Poisson variance at that (rate,seed), not server saturation "
      "(DESIGN.md sec 4.2.1). completion_rps is diluted by the completion "
      "tail for short probes and should NOT be read alone as a saturation "
      "signal. ELBOW-FINDING PRIMARY SIGNAL = the TTFT/ITL percentile TREND "
      "across arrival_rps values (now warmup-corrected), not either "
      "throughput number.")
PY
      # Light settle between rate probes (NOT the full running==0/queued==0
      # drain-detection loop e1_sweep.sbatch uses between LO/HI phases --
      # this scan is elbow-finding only, never cited directly, so a fixed
      # pause is an acceptable cost/rigor tradeoff; DESIGN.md sec 4.2 notes
      # this explicitly).
      sleep 5
      done   # SEEDS
      # ★2026-07-29 (coordinator directive, item 4): seed-divergence report --
      # if two seeds at the SAME nominal rate disagree substantially, that
      # rate is itself an unstable region for off-cliff siting, not a clean
      # single-realization measurement. Threshold (25% relative on TTFT p50)
      # is a judgment call, not data-derived -- flagged as such below.
      if [ "${#SEED_JOUTS[@]}" -ge 2 ]; then
        # Re-derive the two just-produced CAPSCAN_JSON rows by scanning the
        # campaign RESULT file (each per-seed python invocation above ran as
        # a separate process and already tee'd its CAPSCAN_JSON line there).
        python3 - "$RESULT" "$NAME" "$RATE" <<'PY' | tee -a "$RESULT"
import json, sys
result_path, cell, rate = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
for line in open(result_path):
    if not line.startswith("CAPSCAN_JSON "):
        continue
    try:
        o = json.loads(line[len("CAPSCAN_JSON "):])
    except Exception:
        continue
    if o.get("cell") == cell and o.get("nominal_rate") == rate:
        rows.append(o)
if len(rows) >= 2:
    rows = rows[-2:]  # the two seeds just run for this rate
    a, b = rows
    def reldiff(x, y):
        m = max(abs(x), abs(y))
        return abs(x - y) / m if m else 0.0
    d_ttft = reldiff(a["ttft_p50"], b["ttft_p50"])
    d_arr = reldiff(a["arrival_rps"], b["arrival_rps"])
    flag = "  <-- LARGE SEED DIVERGENCE: this rate is an unstable region for off-cliff siting" \
        if d_ttft > 0.25 else ""
    print(f"CAPSCAN_SEED_DIVERGENCE cell={cell} nominal_rate={rate} "
          f"seeds=({a['seed']},{b['seed']}) TTFT_p50_reldiff={d_ttft:.3f} "
          f"arrival_rps_reldiff={d_arr:.3f} "
          f"arrival_rps=({a['arrival_rps']:.2f},{b['arrival_rps']:.2f}) "
          f"TTFT_p50=({a['ttft_p50']:.0f},{b['ttft_p50']:.0f}){flag}")
PY
      fi
    done   # RATES
    python3 "$HERE/e1_pin_check.py" "$PDMUX_TELEMETRY_PATH" "$DSM" | tee -a "$RESULT"
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$RESULT"
    grep -viE "pynvml|Future" "$SRVLOG" | tail -25 | tee -a "$RESULT"
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
done
echo "CAPSCAN_DONE ARM=$ARM" | tee -a "$RESULT"
done
rm -rf "$TRITON_CACHE_DIR"
