#!/bin/bash
#SBATCH --job-name=deckneectx2
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --array=0-11
#SBATCH --output=%x_T%a_%A.out
#SBATCH --error=%x_T%a_%A.err
# ============================================================================
# P5 RE-MEASUREMENT harness (decode knee vs context length), 2026-08-04.
#
# SUPERSEDES decode_knee_vs_ctx.sbatch (job 858811), whose numbers are NOT to be
# reused: three instrumentation defects were confirmed by audit on 2026-08-04.
#   (1) bucket asymmetry  -- "attn" timed only the RadixAttention core while
#       "mamba" timed the whole mixer; qkv/o_proj/MLP/layernorm were unbucketed.
#   (2) running-mean accumulator + `tail -1` -- the reported sample was the one
#       MOST diluted by cold start, and `full` was always the first arm, so the
#       bias sat in the denominator of every ratio.
#   (3) consequent physical-invariant violation (mamba decode must be O(1) in ctx).
#
# What changed HERE (harness side; the model-side fix is in src/models/zamba2.py):
#   * SM arm ORDER IS RANDOMISED per task from a recorded seed  -> "full first"
#     can no longer sit in every denominator.
#   * one WARM-UP round per arm is run and DISCARDED (own mode tag `r w`).
#   * ALL block-level emits are kept (no `tail -1`); the steady state is chosen in
#     analysis (analyze_decode_knee_vs_ctx_v2.py drops the first block of each
#     round and every block flagged dirty).
#   * closure gate: server-side `ZBLT_CLOSURE_FAIL` lines are counted into the
#     result file; a cell whose buckets do not cover the layer loop is not a
#     measurement.
#   * NEGATIVE CONTROL: per-mamba-layer decode time must be O(1) in ctx AT FIXED
#     BATCH. Each cell records its own value (CELL line); the cross-ctx invariance
#     check is done by the analyzer, conditional on the REALISED batch size.
#     A violation means the instrumentation is STILL broken.
#   * grid 4 ctx x 5 SM x n=3 reps (same grid as 858811, reps added).
#
# TOOLING FIXES 2026-08-05 (harness side; mechanical, no design change -- the run
# grid, the arm set, the server flags and the gate thresholds are UNCHANGED):
#   D1  closure-fail counting is now round-scoped (`r[0-9]` vs `rw`), so a block
#       from the DISCARDED warm-up round can no longer fail the gate.
#   D3  the CELL line carries the realised-batch MIXTURE (bs_mode/bs_set/bs_med)
#       instead of a bare `statistics.mode`.
#   (D2/D4/D5 are analyzer-side: see analyze_decode_knee_vs_ctx_v2.py.)
#
# Cuda-graph: `--disable-cuda-graph --disable-piecewise-cuda-graph` is KEPT from
# 858811 for comparability. This is EAGER, i.e. NOT the operating point; it is a
# composition measurement, not a serving claim. Recorded in the result header.
#
# Run:  sbatch decode_knee_vs_ctx_v2.sbatch [SEED]
# Then: python analyze_decode_knee_vs_ctx_v2.py <jobid>
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
HERE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r0c
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$HERE/.triton_cache"
MODEL=Zyphra/Zamba2-2.7B

SEED_BASE=${1:-20260804}
CTX_LIST=(256 1024 4096 16384)
NCTX=${#CTX_LIST[@]}
TASK=${SLURM_ARRAY_TASK_ID:-0}
CTX=${CTX_LIST[$(( TASK % NCTX ))]}
REP=$(( TASK / NCTX + 1 ))
TASK_SEED=$(( SEED_BASE * 1000 + TASK ))

# arm order: deterministic from (SEED_BASE, task) but NOT "full first".
SM_ORDER=$(python3 - "$TASK_SEED" <<'PY'
import random, sys
r = random.Random(int(sys.argv[1]))
arms = ["full", "44", "24", "16", "8"]
r.shuffle(arms)
print(" ".join(arms))
PY
)
read -r -a SM_LIST <<< "$SM_ORDER"

CONC=32; OUTTOK=32
MEAS_ROUNDS=2          # measured rounds per arm (plus 1 discarded warm-up)
CTXCAP=$(( CTX + 512 ))

export SGLANG_ZAMBA_TIMING=1
export SGLANG_ZAMBA_TIMING_EVERY=8      # block size (forwards); legacy default was 30
export SGLANG_ZAMBA_CLOSURE_MIN=0.85    # closure gate threshold: calibrated 2026-08-05
                                         # from job 873617 (smoke, n=49): healthy closure
                                         # min/p50/max = 0.9390/0.9403/0.9435, vs legacy
                                         # (defective) bucket recomputed = 0.6997.
                                         # 0.85 sits in the gap between the two
                                         # populations, not at the healthy-population
                                         # floor (avoids endpoint-selection bias).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SMFILE="$OUT/.deckneectx2_${SLURM_ARRAY_JOB_ID}_${TASK}"
export PDMUX_FIXED_DECODE_SM_FILE="$SMFILE"
echo full > "$SMFILE"
PORT=$((34500 + (SLURM_ARRAY_JOB_ID + TASK) % 300))
SRVLOG="$OUT/deckneectx2_srv_C${CTX}_r${REP}_${SLURM_ARRAY_JOB_ID}.log"
RESULT="$OUT/deckneectx2_result_C${CTX}_r${REP}_${SLURM_ARRAY_JOB_ID}.txt"; : > "$RESULT"

SERVER_FLAGS="--trust-remote-code --dtype bfloat16 --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP"
{
  echo "HEADER model=$MODEL ctx=$CTX ctxcap=$CTXCAP rep=$REP conc=$CONC outtok=$OUTTOK port=$PORT"
  echo "HEADER job=${SLURM_ARRAY_JOB_ID}_${TASK} seed_base=$SEED_BASE task_seed=$TASK_SEED"
  echo "HEADER sm_order=$SM_ORDER   # randomised: 'full' is NOT always first (858811 defect 2)"
  echo "HEADER warmup_rounds=1 meas_rounds=$MEAS_ROUNDS block_every=$SGLANG_ZAMBA_TIMING_EVERY closure_min=$SGLANG_ZAMBA_CLOSURE_MIN"
  echo "HEADER cudagraph=DISABLED (--disable-cuda-graph --disable-piecewise-cuda-graph) EAGER: composition measurement, not the operating point"
  echo "HEADER server_flags=$SERVER_FLAGS"
  echo "HEADER zamba2_sha256=$(sha256sum /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/models/zamba2.py | cut -d' ' -f1)"
  echo "HEADER git_head=$(cd /scratch/ehmoon/whlee/prefill-layer-alloc && git rev-parse --short HEAD 2>/dev/null || echo NA)"
} | tee -a "$RESULT"

python -m sglang.launch_server --model-path "$MODEL" $SERVER_FLAGS \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 220); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RESULT"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"

if [ "$ok" = 1 ]; then
  # prompt ~= CTX tokens (~8 tok / phrase-rep); built via FILE to dodge ARG_MAX at long ctx
  REPS=$(( CTX / 8 )); [ "$REPS" -lt 1 ] && REPS=1
  BODYFILE="$OUT/.deckneectx2_body_${SLURM_ARRAY_JOB_ID}_${TASK}.json"
  python3 - "$REPS" "$OUTTOK" "$BODYFILE" <<'PY'
import json, sys
reps, out, bf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
text = 'the robot walked slowly across the red planet ' * reps
json.dump({'text': text, 'sampling_params': {'max_new_tokens': out, 'temperature': 0.7, 'ignore_eos': True}}, open(bf, 'w'))
PY

  run_round () {   # $1 = sm arm, $2 = round tag
    echo "$1@${CTX}r$2" > "$SMFILE"
    local pids=""
    for r in $(seq 1 $CONC); do
      curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
        --data-binary @"$BODYFILE" >/dev/null 2>&1 &
      pids="$pids $!"
    done
    wait $pids; sleep 2
  }

  for SM in "${SM_LIST[@]}"; do
    run_round "$SM" "w"          # WARM-UP: measured but discarded (own mode tag)
    echo "WARMUP ctx=$CTX rep=$REP sm=$SM blocks=$(grep -c "mode=$SM@${CTX}rw " "$SRVLOG")" | tee -a "$RESULT"
    for RND in $(seq 1 $MEAS_ROUNDS); do
      run_round "$SM" "$RND"
      # keep EVERY block emit (no `tail -1`); steady state is selected in analysis
      grep "ZBLT2 mode=$SM@${CTX}r${RND} " "$SRVLOG" \
        | sed "s|^|BLOCK ctx=$CTX rep=$REP sm=$SM round=$RND |" >> "$RESULT"
    done
    # D1 (2026-08-05): the old single counter greped `mode=$SM@${CTX}r`, which also
    # matches the DISCARDED warm-up tag `rw`, so a warm-up block failed the gate and
    # the analyzer contradicted itself (GATE C PASS, GATES closure=FAIL).  Measured
    # rounds and warm-up are now counted separately; warm-up is reported, not scored.
    CF_MEAS=$(grep -cE "ZBLT_CLOSURE_FAIL mode=$SM@${CTX}r[0-9]" "$SRVLOG")
    CF_WARM=$(grep -cE "ZBLT_CLOSURE_FAIL mode=$SM@${CTX}rw" "$SRVLOG")
    echo "ARMDONE ctx=$CTX rep=$REP sm=$SM closure_fail_meas=$CF_MEAS closure_fail_warmup=$CF_WARM" | tee -a "$RESULT"
  done

  # per-cell summary + per-cell record of the O(1)-in-ctx negative control input.
  # (the cross-ctx invariance verdict needs all four tasks -> analyzer)
  python3 - "$RESULT" "$CTX" "$REP" <<'PY' | tee -a "$RESULT"
import re, statistics, sys
res, ctx, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat = re.compile(
    r"BLOCK ctx=(\d+) rep=(\d+) sm=(\S+) round=(\d+) .*?"
    r"blk=(\d+) blk_n=(\d+) dirty=(\d+) shape=(\d+)x(\d+) "
    r"closure=([\d.naif-]+) closure_min=([\d.naif-]+) fwd_ms=([\d.]+) .*?"
    r"b_per_attn=([\d.]+) b_per_mamba=([\d.]+) b_per_attn_core=([\d.]+)"
)
cells = {}
for ln in open(res):
    m = pat.search(ln)
    if not m:
        continue
    (_c, _r, sm, rnd, blk, bn, dirty, bs, ntok,
     clo, clomin, fwd, pa, pm, pac) = m.groups()
    rec = dict(sm=sm, rnd=int(rnd), blk=int(blk), n=int(bn), dirty=int(dirty),
               bs=int(bs), ntok=int(ntok), closure=float(clo), fwd=float(fwd),
               per_attn=float(pa), per_mamba=float(pm), per_attn_core=float(pac))
    cells.setdefault(sm, []).append(rec)

for sm, recs in cells.items():
    # steady state = drop the first block of each round (post-switch transient)
    # and every block whose batch shape changed mid-block (dirty).
    first = {}
    for r in recs:
        first[r["rnd"]] = min(first.get(r["rnd"], 10**9), r["blk"])
    keep = [r for r in recs if r["dirty"] == 0 and r["blk"] != first[r["rnd"]]]
    if not keep:
        print(f"CELL ctx={ctx} rep={rep} sm={sm} STATUS=NO_STEADY_BLOCKS n_raw={len(recs)}")
        continue
    med = lambda k: statistics.median([r[k] for r in keep])
    lo_clo = min(r["closure"] for r in keep)
    # D3 (2026-08-05): the realised batch of a cell can be a MIXTURE (ctx16384 in job
    # 873783 ran 6 blocks at bs=19 then 8 at bs=13).  `statistics.mode` alone printed
    # `bs=13` and hid that from the label while the median still saw it.  Emit the
    # whole mixture: mode, the {bs: n_blocks} map, and the median.  No single-value
    # batch field is emitted on its own.
    bs_counts = {}
    for r in keep:
        bs_counts[r["bs"]] = bs_counts.get(r["bs"], 0) + 1
    bs_set = ",".join(f"{b}x{n}" for b, n in sorted(bs_counts.items(), reverse=True))
    print(
        f"CELL ctx={ctx} rep={rep} sm={sm} STATUS=OK n_blocks={len(keep)}/{len(recs)} "
        f"bs_mode={statistics.mode([r['bs'] for r in keep])} bs_set={bs_set} "
        f"bs_med={statistics.median([r['bs'] for r in keep]):.1f} "
        f"closure_med={med('closure'):.4f} closure_min={lo_clo:.4f} "
        f"fwd_ms={med('fwd'):.4f} per_attn={med('per_attn'):.5f} "
        f"per_mamba={med('per_mamba'):.5f} per_attn_core={med('per_attn_core'):.5f}"
    )
PY
else
  echo "BOOT FAILED" | tee -a "$RESULT"; tail -20 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE ctx=$CTX rep=$REP" | tee -a "$RESULT"
