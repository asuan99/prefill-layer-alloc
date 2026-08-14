#!/bin/bash
#SBATCH --job-name=e1m3ctl
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier/%x_%j.err
# ============================================================================
# M3 -- TRANSFORMER-CONTROL CONTRAST.  Pre-registered in DESIGN.md sec 4.3.8(c);
# do not change anything below without editing that section FIRST.
#
# QUESTION (one, narrow): at serving concurrency, does the per-request ITL tail
# respond to the decode-SM split at all -- and does the answer differ between a
# pure Transformer (T8, positive control) and a hybrid (Ha8)?
#
# It does NOT ask whether the lever pays off in goodput. That question is only
# askable if the answer here is yes, which is why this job exists.
#
# WHY THIS SHAPE (all of it forced by the 2026-08-02 claims-auditor referral and
# the M1/M2/M4 follow-ups -- see FINDINGS_M1_M2_2026-08-02.md):
#   - ONE COMMON RATE for every arm and cell.  The published 4-arm rung table
#     was silently assembled at rate 12 for T8 and rate 2 for the others, which
#     sec 4.3.5(b) forbids in as many words as a rate-confound.
#   - NP_MIN, NOT PROBE_TARGET_S, is the probe-size knob.  At rate <= 5 the
#     probe size is max(NP_MIN, PROBE_TARGET_S*rate), so NP_MIN binds and
#     raising PROBE_TARGET_S from 8 to 20 would change nothing.  n=45 left 39
#     post-warmup requests and a p95 CI 8-95% wide against a 15% classification
#     margin: not a test.  NP_MIN=200 here (~194 post-warmup, 5x that), the
#     rest of the budget going to BLOCKS -- see the next item.
#   - BLOCKS ARE THE REPLICATION UNIT, NOT SEEDS.  g pairs d16 with d54, and a
#     static split cannot change without a reboot (multiplexing_mixin.py:155),
#     so the two cells of g always come from DIFFERENT boots and boot-to-boot
#     variance enters g directly instead of cancelling.  Extra seeds inside one
#     boot buy no independent replication against it -- exactly the
#     pseudo-replication the audit found in batch-cap.  So: BLOCKS independent
#     replicates, one boot per cell per block, one seed per block, cell order
#     alternating fwd/rev by block so cell is crossed with time order.
#   - CAP UNCHANGED AT 48, BUT THE MAMBA POOL PINNED EXPLICITLY.  A cap of 96
#     was drafted and rejected: it would double Ha8's SSM state pool
#     (6.78 -> 13.5 GB) out of an attention KV pool already only 52%
#     subscribed, for no benefit, since 48 already clears the measured 12-30
#     concurrency.  --max-running-requests 48 was an unregistered constant
#     (sec 4.3.6) and, on SSM arms, a DOUBLE knob:
#     model_runner_kv_cache_mixin.py:223-230 sets max_mamba_cache_size = cap
#     when radix cache is off, making kv_mamba_occupancy = batch/cap an
#     IDENTITY rather than a memory signal.  Setting it explicitly takes the
#     :218 branch and decouples the two.
#   - NO CAP EXPERIMENT.  A cap TOST at this rate would be vacuous: measured max
#     concurrency here is 12-30 against a cap of 48, so no cap effect can exist
#     (sec 4.3.8(b)).  An assertion replaces it.
#
# NOT a claim job on its own; nothing here is cited without claims-auditor.
# Usage (defaults are the pre-registered values -- override only to debug):
#   sbatch e1_m3_control.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_e1m3_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

# --- arms: T8 is the POSITIVE CONTROL, Ha8 the arm under test -----------------
# Same indices/backends as e1_capacity_scan.sbatch so nothing silently diverges.
# The triton/flashinfer asymmetry does not arise here: both arms use triton.
ARM_LABELS=(T8 Ha8)
ARM_MODELS=("Qwen/Qwen2.5-7B" "Zyphra/Zamba2-7B-Instruct")
ARM_BACKENDS=(triton triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

# --- PRE-REGISTERED PARAMETERS (DESIGN.md sec 4.3.8(c)) ----------------------
RATE="${RATE:-2}"                 # the ONE common rate, sec 4.3.7(b): highest
                                  # probe rate at which every non-d92 cell of
                                  # every arm is off-cliff under all 9 knee
                                  # estimator variants (binding cell: Ha8 d54,
                                  # min knee 3.08)
CELLS_FWD="${CELLS_FWD:-d16 d24 d44 d54}"   # d92 is NOT measured here: sec
                                  # 4.3.5(b) excludes it from the argmax and
                                  # sec 4.3.8(a) forbids reading A_free there
                                  # (prefill holds 16 SM, so sub-threshold
                                  # prompts block too and the decomposition
                                  # stays contaminated).  Adding it back would
                                  # cost 1/4 of the job for an unreadable cell.
BLOCKS="${BLOCKS:-8}"             # ★★THE replication parameter.  See sec
                                  # 4.3.8(c): a BLOCK, not a seed, is the
                                  # independent unit for g, because
                                  # g = A_free(d16)/A_free(d54) pairs two cells
                                  # measured in DIFFERENT boots (a static split
                                  # cannot be changed without a reboot,
                                  # multiplexing_mixin.py:155), so boot-to-boot
                                  # variance does NOT cancel in g -- it enters
                                  # it directly.  Extra seeds INSIDE one boot
                                  # therefore buy no independent replication
                                  # against it; that was the pseudo-replication
                                  # the 2026-08-02 audit found in the batch-cap
                                  # run.  One seed per block, seed = block
                                  # index, so workload and boot vary together
                                  # and n_independent = BLOCKS unambiguously.
                                  # 8 tolerates sd(g) <= 0.227, which covers the
                                  # worst case observed anywhere in this
                                  # campaign (M8: g = 1.387 vs 1.071 across two
                                  # seeds, sd 0.223).  2 blocks would tolerate
                                  # only 0.021 = never fires.
NP_MIN="${NP_MIN:-200}"           # lowered 300->200 to pay for BLOCKS 2->8.
                                  # This moves power from WITHIN a cell to
                                  # BETWEEN independent replicates, and the
                                  # verdict's CI depends only on the latter.
                                  # 200 still leaves ~194 post-warmup requests,
                                  # 5x the 39 that made the sec 4.3.7 per-cell
                                  # labels UNPOWERED.
PROBE_TARGET_S="${PROBE_TARGET_S:-8}"   # documented value is 20 but every
                                  # prior job (867231, 870295-297, 870301) ran
                                  # at 8; pinned to 8 for comparability.  At
                                  # rate 2 NP_MIN binds anyway (300 > 8*2).
WARMUP_S="${WARMUP_S:-3}"
CAP="${CAP:-48}"                  # sec 4.3.8(b).  ★48, NOT 96.  The point of
                                  # this parameter is only that the cap must be
                                  # unable to bind, and at RATE=2 the measured
                                  # max concurrency is 12-30 (44 at Ha8 d92,
                                  # which this job does not run) -- 48 already
                                  # clears that with ~60% headroom.  Raising it
                                  # to 96 would DOUBLE the SSM state pool on Ha8
                                  # (0.141 GB/slot: 6.78 -> 13.5 GB) and that
                                  # memory comes straight out of the attention
                                  # KV pool, which the 2026-08-02 audit already
                                  # measured at only 52% of 48 x ctx.  That is a
                                  # memory-split confound bought for no benefit.
MAMBA_POOL="${MAMBA_POOL:-$CAP}"  # arm-common RULE (= cap), NOT an arm-common
                                  # absolute constant: per-slot cost differs
                                  # (M8 0.255 / Ha8 0.141 / Hs8 0.096 GB), so a
                                  # shared absolute pool would force a different
                                  # memory split on each arm.  Setting it
                                  # EXPLICITLY is the whole change: it takes the
                                  # kv_cache_mixin:218 branch instead of :223,
                                  # so kv_mamba_occupancy stops being the
                                  # identity batch/cap.  Memory is unchanged
                                  # from every prior job in this campaign.
CONC_ASSERT_FRAC="${CONC_ASSERT_FRAC:-0.8}"

# Plateau TTFT-p50 (ms) per (arm, cell) from the sec 4.3.7 knee scan, measured
# at cap 48.  Used ONLY for the in-run off-cliff assertion described in sec
# 4.3.8(c): this job changes the cap, and rather than re-run the capacity scan
# we assert in-run that rate RATE is still on the plateau.  A cell failing this
# is escalated and VOIDED, not silently kept.
declare -A PLATEAU
PLATEAU[T8_d16]=48;  PLATEAU[T8_d24]=53;  PLATEAU[T8_d44]=57;  PLATEAU[T8_d54]=64
PLATEAU[Ha8_d16]=152; PLATEAU[Ha8_d24]=162; PLATEAU[Ha8_d44]=172; PLATEAU[Ha8_d54]=193

JOBTAG="${SLURM_JOB_ID:-local}"
NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")

echo "=== M3 TRANSFORMER-CONTROL (DESIGN.md sec 4.3.8) job=$JOBTAG ==="
echo "RATE=$RATE NP=$NP (NP_MIN=$NP_MIN PROBE_TARGET_S=$PROBE_TARGET_S) WARMUP_S=$WARMUP_S"
echo "CAP=$CAP MAMBA_POOL=$MAMBA_POOL CELLS_FWD=[$CELLS_FWD] BLOCKS=$BLOCKS (1 seed each, seed=block)"
echo "SGPT_CTXLEN=$SGPT_CTXLEN CTXCAP=$CTXCAP"

CELLS_REV=$(echo $CELLS_FWD | tr ' ' '\n' | tac | tr '\n' ' ')

for AIDX in 0 1; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/e1m3_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND RATE=$RATE NP=$NP CAP=$CAP" | tee -a "$RESULT"

# A BLOCK = one independent replicate = one fresh boot of every cell, run at
# one seed.  Odd blocks use the forward cell order, even blocks the reverse, so
# cell is CROSSED with boot/time order rather than confounded with it.
for BI in $(seq 1 $BLOCKS); do
  BLOCK="b${BI}"
  if [ $(( BI % 2 )) -eq 1 ]; then CELLS="$CELLS_FWD"; else CELLS="$CELLS_REV"; fi
  BSEEDS="$BI"                    # seed == block index: one workload draw per
                                  # independent replicate (sec 4.3.8(c))
  echo "--- block=$BLOCK cell_order=[$CELLS] seed=$BSEEDS ---" | tee -a "$RESULT"

  for NAME in $CELLS; do
    CFG="$HERE/pdmux_e1_${NAME}.yml"
    DSM="${NAME#d}"
    [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG" | tee -a "$RESULT"; continue; }
    PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
    RUNID="e1m3_${ARM}_${NAME}_${BLOCK}_${JOBTAG}"
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="m3_${ARM}_${NAME}_${BLOCK}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    # sec 4.7.1: the observer-effect gate found PDMUX_TRACE_FORCE_PREFILL moves
    # d92 itl_p95 by +2.0% [+0.78, +3.21].  ITL-p95 is exactly what this job
    # measures, so force-trace is OFF here and pin verification is a SEPARATE
    # short run.
    export PDMUX_TRACE_FORCE_PREFILL=0

    echo "=== ARM=$ARM cell=$NAME block=$BLOCK target_decode_sm=$DSM port=$PORT ===" | tee -a "$RESULT"
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend $BACKEND \
      --enable-pdmux --pdmux-config-path "$CFG" \
      --disable-piecewise-cuda-graph \
      --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
      --mem-fraction-static 0.80 \
      --max-running-requests $CAP --max-mamba-cache-size $MAMBA_POOL \
      --context-length $CTXCAP \
      --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
    SRV=$!
    ok=0
    for i in $(seq 1 240); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
      kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED cell=$NAME block=$BLOCK" | tee -a "$RESULT"; break; }
      sleep 5
    done
    echo "boot_ok=$ok cell=$NAME block=$BLOCK" | tee -a "$RESULT"

    # ★ sec 4.3.8(b) item 3: the pool must be independent of the cap now.  On a
    # pure Transformer arm there is no mamba pool line at all, which is itself
    # the check that --max-mamba-cache-size did not silently do something else.
    grep -E "Mamba Cache is allocated|max_mamba_cache_size|KV Cache is allocated" \
         "$SRVLOG" | head -4 | sed "s/^/  MEMLINE $ARM $NAME $BLOCK: /" | tee -a "$RESULT"

    if [ "$ok" = 1 ]; then
      for SEED in $BSEEDS; do
        JOUT="$HERE/${RUNID}_s${SEED}.jsonl"; : > "$JOUT"
        python -m sglang.bench_serving --backend sglang --model "$MODEL" \
          --host 127.0.0.1 --port $PORT \
          --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
          --num-prompts $NP --request-rate $RATE --seed $SEED --output-details \
          --output-file "$JOUT" > "$HERE/${RUNID}_s${SEED}_bench.log" 2>&1
        BENCH_RC=$?
        if [ "$BENCH_RC" != 0 ]; then
          echo "PROBE_FAILED arm=$ARM cell=$NAME block=$BLOCK seed=$SEED rc=$BENCH_RC" | tee -a "$RESULT"
          continue
        fi
        # Per-probe gates.  These print PASS/FAIL lines that the analyzer reads;
        # nothing is silently dropped.
        python3 - "$JOUT" "$ARM" "$NAME" "$BLOCK" "$SEED" "$RATE" "$WARMUP_S" \
                  "$CAP" "$CONC_ASSERT_FRAC" "${PLATEAU[${ARM}_${NAME}]:-0}" <<'PY' | tee -a "$RESULT"
import json, math, sys
path, arm, cell, block, seed, rate, warmup_s, cap, frac, plateau = sys.argv[1:11]
rate, warmup_s, cap, frac, plateau = float(rate), float(warmup_s), int(cap), float(frac), float(plateau)

def pctl(a, q):
    if not a: return float("nan")
    a = sorted(a); p = q * (len(a) - 1); lo = int(p); hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)

o = json.loads(open(path).read().strip().split("\n")[0])
tt, il, err = o.get("ttfts") or [], o.get("itls") or [], o.get("errors") or []
inp, out = o.get("input_lens") or [], o.get("output_lens") or []
nw = math.ceil(warmup_s * rate)
keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
ttft = [tt[i] * 1000.0 for i in keep]
nerr = sum(1 for e in err if e)

# GATE 1 (sec 4.3.8(b) item 3): the cap must not be able to bind.
conc = o.get("max_concurrent_requests")
g1 = "PASS" if (conc is not None and conc < frac * cap) else "FAIL"

# GATE 2 (sec 4.3.8(c)): rate must still be off-cliff under the new cap.
t50 = pctl(ttft, 0.50)
g2 = "PASS" if (plateau > 0 and t50 <= 2.0 * plateau) else ("FAIL" if plateau > 0 else "NOPLATEAU")

# GATE 3: errors.
g3 = "PASS" if nerr == 0 else "FAIL"

# The two pre-registered ITL statistics (sec 4.3.8(a)).  A_free drops any ITL
# overlapping the prefill window of a >=1024-token prompt.  Arrival times come
# from the RNG replay used everywhere else in this campaign.
import numpy as np
np.random.seed(int(seed))
iv = [np.random.exponential(1.0 / rate) for _ in range(len(tt))]
arr = np.cumsum([0.0] + iv[:-1])
PREFILL_BLOCK_TOK = 1024
win = [(arr[j], arr[j] + tt[j]) for j in range(len(tt))
       if j < len(inp) and inp[j] >= PREFILL_BLOCK_TOK]
pa, pf = [], []
for i in keep:
    if not il[i]: continue
    t = arr[i] + tt[i]; allv = []; free = []
    for d in il[i]:
        a, b = t, t + d
        allv.append(d * 1000.0)
        if not any(b > w0 and a < w1 for w0, w1 in win): free.append(d * 1000.0)
        t = b
    if allv: pa.append(pctl(allv, 0.95))
    if free: pf.append(pctl(free, 0.95))

print(f"M3PROBE arm={arm} cell={cell} block={block} seed={seed} n_keep={len(keep)} "
      f"n_err={nerr} conc={conc} cap={cap} "
      f"ttft_p50={t50:.1f} ttft_p95={pctl(ttft,0.95):.1f} plateau={plateau:.0f} "
      f"A_all={pctl(pa,0.95):.2f} A_free={pctl(pf,0.95):.2f} "
      f"n_blocking_windows={len(win)} "
      f"GATE_CONC={g1} GATE_OFFCLIFF={g2} GATE_ERR={g3}")
if "FAIL" in (g1, g2, g3):
    print(f"M3ESCALATE arm={arm} cell={cell} block={block} seed={seed} "
          f"conc={g1} offcliff={g2} err={g3} -- cell VOIDED, see sec 4.3.8(c)")
PY
      done
    fi
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 5
  done
done
done

echo "=== M3 done job=$JOBTAG.  Analyze with:"
echo "    python3 m3_analyze.py --dir $HERE --job $JOBTAG"
echo "=== Nothing here is citable before claims-auditor. ==="
