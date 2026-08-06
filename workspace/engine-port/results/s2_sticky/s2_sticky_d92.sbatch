#!/bin/bash
#SBATCH --job-name=s2stickyD92
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky/%x_%j.err
# ============================================================================
# S2-alpha -- STICKY-ON, HIGH-D CONTROL CELL.  This is gate (alpha) from
# CONSENSUS.md sec 1-31 / PROJECT_STATUS.md "8B decode-SM 프론티어" "다음
# 실험 gate" #8: the ONE cheap experiment that can falsify the 2026-08-05
# behavioural closure of Sec 0 (872077 vs C2 decode-16SM ITL 2.6x mismatch).
# Pre-registered here, in this file, BEFORE submission (methodology gate #8
# forbids re-scoring thresholds after seeing results):
#
#   treatment : sticky ON (PDMUX_STICKY_PARTITION=1) + decode SM fixed near
#               92 (the s8_scaleup/pdmux_c_d92.yml precedent; this campaign
#               reuses the identical, already-existing, UNMODIFIED
#               s8_frontier/pdmux_e1_d92.yml -- prefill=16 SM / decode=92 SM,
#               guard-satisfier row [64,44,48] never selected under
#               FixedPolicy(92), same reasoning as pdmux_e1_d54.yml).
#   rest      : IDENTICAL to s2_sticky.sbatch (job 873015) -- T8
#               (Qwen/Qwen2.5-7B) arm only, ShareGPT rate 2 open-loop, NO
#               keepalive, cudagraph ON (only --disable-piecewise-cuda-graph,
#               never --disable-cuda-graph -- verified against
#               e1_m3_control.sbatch:212-218, server args VERBATIM except
#               --pdmux-config-path), 4 BLOCKS (not 8 -- this gate's
#               pre-registration in the task brief scopes it to 4 blocks,
#               replication unit = block = independent boot + seed=block
#               index, same convention as s2_sticky.sbatch).
#
#   primary   : pooled per-token ITL p50, unit = one ITL interval = one
#               emitted token, post-warmup, error-free. Computed TWO ways,
#               both printed by this job (sec "S2A pooled diagnostics"
#               below):
#                 (1) "telemetry path" -- m3_conditional.label_probe's
#                     a_free-filtered, lag-aligned population (same pipeline
#                     s2_sticky.sbatch used for its S2_POOLED p50_ALL/
#                     p50_SPLIT numbers).
#                 (2) "raw-itls path" -- direct flatten of the client
#                     bench_serving JSONL `itls` field, post-warmup +
#                     error-free ONLY, NO a_free filter, NO split_frac label,
#                     NO telemetry file touched at all. This is the path
#                     S2_REPLICATION_2026-08-05.md sec 1 says was the actual
#                     basis for the 28.92ms headline ("317,342 client ITL
#                     intervals, raw median").
#
#   gate      : E1_DECODE_REALIZED (time-weighted, decode-busy-conditional,
#               e1_pin_check.compute_decode_realized) >= 0.90 per block. If
#               unmet the treatment did not land and the readout is void.
#
#   PRE-REGISTERED PREDICTION / VERDICT RULE (fixed BEFORE submission, do not
#   edit after seeing results -- methodology gate #8):
#     p50 near 12-13 ms  -> SM count sets the level -> Sec 0 closure STANDS
#                           (claims-auditor recomputed C2 d92 = 12.88ms)
#     p50 near 29 ms     -> the level is made by the STICKY TREATMENT ITSELF,
#                           not SM count -> Sec 0 closure COLLAPSES
#     in between         -> INDETERMINATE, report as-is, no verdict forced
#
# This job computes and prints the gate + primary + the three-way interval
# comparison ONLY.  No throughput/latency/goodput/g/G_LEVER/G_FLAT/policy
# judgment is made here -- that is result-analyst / claims-auditor's call on
# this same data, per this repo's role separation.
#
# Nothing in s2_sticky.sbatch or its artifacts (job 873015) is modified by
# this file -- this is a new, standalone script.
# Usage:
#   sbatch s2_sticky_d92.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/s2_sticky
E1=$ROOT/workspace/engine-port/results/s8_frontier   # configs + gate/analysis modules live here, unchanged
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_s2stickyd92_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest_s2d92.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

# --- arm: T8 ONLY, identical to s2_sticky.sbatch (job 873015). -------------
ARM_LABELS=(T8)
ARM_MODELS=("Qwen/Qwen2.5-7B")
ARM_BACKENDS=(triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

# --- PRE-REGISTERED PARAMETERS -- IDENTICAL to s2_sticky.sbatch except
# CELLS_FWD (single d92 cell) and BLOCKS (4, per this gate's task brief). ---
RATE="${RATE:-2}"
CELLS_FWD="${CELLS_FWD:-d92}"      # single high-D control cell, gate (alpha)
BLOCKS="${BLOCKS:-4}"              # replication unit = block (independent boot +
                                   # seed=block index), same rationale as
                                   # s2_sticky.sbatch / e1_m3_control.sbatch.
NP_MIN="${NP_MIN:-200}"
PROBE_TARGET_S="${PROBE_TARGET_S:-8}"
WARMUP_S="${WARMUP_S:-3}"
CAP="${CAP:-48}"
MAMBA_POOL="${MAMBA_POOL:-$CAP}"    # no-op for T8 (pure Transformer)
CONC_ASSERT_FRAC="${CONC_ASSERT_FRAC:-0.8}"

# No PLATEAU entry exists for d92 anywhere in this campaign line
# (e1_m3_control.sbatch's PLATEAU table only covers d16/d24/d44/d54) -- the
# per-probe GATE_OFFCLIFF check below reports NOPLATEAU for this cell rather
# than fabricating a threshold; this is the script's existing, documented
# fallback behaviour (plateau=0 -> NOPLATEAU, not FAIL), not a new gate.
declare -A PLATEAU

JOBTAG="${SLURM_JOB_ID:-local}"
NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")

echo "=== S2-ALPHA STICKY HIGH-D CONTROL (gate alpha, CONSENSUS.md sec 1-31) job=$JOBTAG ==="
echo "PDMUX_STICKY_PARTITION=1. Prereg verdict rule: p50 12-13ms => Sec0 STANDS; ~29ms => Sec0 COLLAPSES; else INDETERMINATE."
echo "RATE=$RATE NP=$NP (NP_MIN=$NP_MIN PROBE_TARGET_S=$PROBE_TARGET_S) WARMUP_S=$WARMUP_S"
echo "CAP=$CAP MAMBA_POOL=$MAMBA_POOL CELLS_FWD=[$CELLS_FWD] BLOCKS=$BLOCKS (1 seed each, seed=block)"
echo "SGPT_CTXLEN=$SGPT_CTXLEN CTXCAP=$CTXCAP"

for AIDX in 0; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/s2a_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND RATE=$RATE NP=$NP CAP=$CAP STICKY=1" | tee -a "$RESULT"

for BI in $(seq 1 $BLOCKS); do
  BLOCK="b${BI}"
  CELLS="$CELLS_FWD"
  BSEEDS="$BI"
  echo "--- block=$BLOCK cell_order=[$CELLS] seed=$BSEEDS ---" | tee -a "$RESULT"

  for NAME in $CELLS; do
    CFG="$E1/pdmux_e1_${NAME}.yml"
    DSM="${NAME#d}"
    [ -f "$CFG" ] || { echo "MISSING_CONFIG cell=$NAME cfg=$CFG" | tee -a "$RESULT"; continue; }
    PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
    RUNID="e1m3_${ARM}_${NAME}_${BLOCK}_${JOBTAG}"    # naming kept identical to
                                                       # s2_sticky.sbatch/
                                                       # m3_conditional.py's tag
                                                       # convention on purpose.
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="s2a_${ARM}_${NAME}_${BLOCK}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    export PDMUX_TRACE_FORCE_PREFILL=0
    export PDMUX_STICKY_PARTITION=1    # ★ the ONE substantive flag change vs
                                        # a plain e1_m3_control.sbatch probe --
                                        # identical to s2_sticky.sbatch's usage.

    echo "=== ARM=$ARM cell=$NAME block=$BLOCK target_decode_sm=$DSM port=$PORT sticky=1 ===" | tee -a "$RESULT"
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
    grep -c "sticky partition ENABLED" "$SRVLOG" \
      | sed "s/^/  STICKY_LOG_LINES cell=$NAME block=$BLOCK: /" | tee -a "$RESULT"

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
        # Per-probe gates -- SAME logic/thresholds as e1_m3_control.sbatch /
        # s2_sticky.sbatch (GATE_CONC / GATE_OFFCLIFF / GATE_ERR), unchanged.
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

conc = o.get("max_concurrent_requests")
g1 = "PASS" if (conc is not None and conc < frac * cap) else "FAIL"

t50 = pctl(ttft, 0.50)
g2 = "PASS" if (plateau > 0 and t50 <= 2.0 * plateau) else ("FAIL" if plateau > 0 else "NOPLATEAU")

g3 = "PASS" if nerr == 0 else "FAIL"

# ---- RAW-ITLS PATH (this gate's primary, path 2): no a_free filter, no
# split_frac, no telemetry file touched -- direct flatten of the client
# bench_serving JSONL itls field, post-warmup + error-free only. ----------
raw_itl = [d * 1000.0 for i in keep for d in (il[i] or [])]
raw_p50 = pctl(raw_itl, 0.50)
raw_p95 = pctl(raw_itl, 0.95)

print(f"M3PROBE arm={arm} cell={cell} block={block} seed={seed} n_keep={len(keep)} "
      f"n_err={nerr} conc={conc} cap={cap} "
      f"ttft_p50={t50:.1f} ttft_p95={pctl(ttft,0.95):.1f} plateau={plateau:.0f} "
      f"GATE_CONC={g1} GATE_OFFCLIFF={g2} GATE_ERR={g3}")
print(f"S2A_RAW_ITL arm={arm} cell={cell} block={block} seed={seed} "
      f"n_itl={len(raw_itl)} raw_p50={raw_p50:.3f} raw_p95={raw_p95:.3f} "
      f"(raw-itls path: no a_free filter, no split_frac, no telemetry)")
if "FAIL" in (g1, g2, g3):
    print(f"M3ESCALATE arm={arm} cell={cell} block={block} seed={seed} "
          f"conc={g1} offcliff={g2} err={g3} -- cell VOIDED, see sec 4.3.8(c)")
PY

        # ------------------------------------------------------------------
        # ★ S2A IN-RUN GATES/DIAGNOSTICS -- E1_DECODE_REALIZED computed with
        # the EXISTING e1_pin_check.py (unmodified), exactly as in
        # s2_sticky.sbatch. ALIGN_R/AMBIG_FRAC/MIN_N_SPLIT/p50(SPLIT) vs
        # p50(all) also reused unmodified from m3_conditional.py -- these are
        # DIAGNOSTIC, not outcome gates (per S2_REPLICATION_2026-08-05.md
        # sec 7, they degenerate to near-identities once sticky ON makes
        # SPLIT ~= the whole population; reported anyway for continuity with
        # the S2 line, not scored against an invented threshold here).
        # ------------------------------------------------------------------
        python3 - "$HERE" "$ARM" "$NAME" "$BI" "$JOBTAG" "$DSM" "$RATE" "$E1" <<'PY' | tee -a "$RESULT"
import sys
d, arm, cell, block_s, job, dsm_s, rate_s, moddir = sys.argv[1:9]
block, dsm, rate = int(block_s), int(dsm_s), float(rate_s)
sys.path.insert(0, moddir)
import e1_pin_check as pc
import m3_conditional as mc

tele = f"{d}/e1m3_{arm}_{cell}_b{block}_{job}_telemetry.jsonl"

dr = pc.compute_decode_realized(tele, dsm)
frac = dr["decode_realized_frac"]
verdict = "PASS" if (frac == frac and frac >= 0.90) else "FAIL"
hist = ", ".join(f"D{k}:{v:.1f}s" for k, v in
                  sorted(dr["decode_hist"].items(), key=lambda kv: -kv[1])[:4])
print(f"S2A_GATE_DECODE_REALIZED arm={arm} cell={cell} block={block} D={dsm} "
      f"frac={frac:.4f} verdict={verdict} (gate: >=0.90, task-brief alpha gate) "
      f"t_decode_active={dr['t_decode_active_s']:.1f}s hist=[{hist}]")

p = mc.label_probe(d, arm, cell, block, block, job, rate=rate)  # seed==block, campaign convention
align_flag = "" if p["align_r"] >= mc.ALIGN_R_MIN else "ALIGN-WEAK"
print(f"S2A_GATE_ALIGN_R arm={arm} cell={cell} block={block} lag={p['lag']:.2f} "
      f"align_r={p['align_r']:.4f} flag={align_flag or 'none'} "
      f"(threshold {mc.ALIGN_R_MIN}, FLAG-ONLY, never excluded)")

recs_af = [r for r in p["recs"] if r[1]]   # a_free_only population
n_af = len(recs_af)
n_split = sum(1 for r in recs_af if r[2] >= mc.SPLIT_HI)
n_unsplit = sum(1 for r in recs_af if r[2] <= mc.UNSPLIT_LO)
n_ambig = n_af - n_split - n_unsplit
ambig_frac = (n_ambig / n_af) if n_af else float("nan")
print(f"S2A_DIAG_AMBIG_MIN_N_SPLIT arm={arm} cell={cell} block={block} "
      f"n_a_free={n_af} n_split={n_split} n_unsplit={n_unsplit} n_ambig={n_ambig} "
      f"ambig_frac={ambig_frac:.4f} (diagnostic only, per S2_REPLICATION_2026-08-05.md sec 7)")

sp = mc.tokens(p["recs"], mc.sel_split)
al = mc.tokens(p["recs"], mc.sel_all)
p50_sp, p50_all = mc.pctl(sp, .50), mc.pctl(al, .50)
agree = (abs(p50_sp - p50_all) <= 0.05 * max(p50_sp, p50_all, 1e-9)) if (sp and al) else False
print(f"S2A_DIAG_P50 arm={arm} cell={cell} block={block} n_split_tok={len(sp)} n_all_tok={len(al)} "
      f"p50_SPLIT={p50_sp:.2f}ms p50_ALL={p50_all:.2f}ms agree_within_5pct={agree} "
      f"(telemetry-path pooled per-block value; pooled/4-block figure is in the "
      f"S2A pooled diagnostics section below)")
PY
      done
    fi
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 5
  done
done
done

# ============================================================================
# S2A pooled diagnostics -- BOTH primary paths, pooled over all 4 blocks.
# (1) telemetry path: m3_conditional.label_probe's a_free-filtered,
#     lag-aligned p50(SPLIT)/p50(all) -- same pipeline as s2_sticky.sbatch's
#     S2_POOLED block.
# (2) raw-itls path: direct pool of the per-block S2A_RAW_ITL raw_itl arrays
#     printed above, recomputed here from the client JSONL files directly
#     (no telemetry, no a_free filter, no split_frac).
# Printed for monitoring/triage ONLY -- the sec-3-way verdict readout
# (12-13ms / 29ms / in-between) is reported in the job report, but the
# throughput/latency/goodput/g/policy INTERPRETATION of that readout is
# result-analyst / claims-auditor's call on this same data, not this job's.
# ============================================================================
echo "=== S2A pooled diagnostics (gate alpha; NOT a throughput/policy verdict) ===" | tee -a "$HERE/s2a_pooled_${JOBTAG}.txt"
python3 - "$HERE" "$JOBTAG" "$RATE" "$E1" "$BLOCKS" "$WARMUP_S" <<'PY' | tee -a "$HERE/s2a_pooled_${JOBTAG}.txt"
import json, math, sys
d, job, rate_s, moddir, blocks_s, warmup_s_s = sys.argv[1:7]
rate, blocks, warmup_s = float(rate_s), int(blocks_s), float(warmup_s_s)
sys.path.insert(0, moddir)
import m3_conditional as mc

ARM = "T8"
CELL = "d92"
BLOCKS = range(1, blocks + 1)

def pctl(a, q):
    if not a: return float("nan")
    a = sorted(a); p = q * (len(a) - 1); lo = int(p); hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)

# --- (1) telemetry path (m3_conditional pipeline, unmodified) --------------
SP, AL = [], []
aligns = []
n_af_tot = n_split_tot = n_unsplit_tot = 0
per_block_p50_all = []
for b in BLOCKS:
    p = mc.label_probe(d, ARM, CELL, b, b, job, rate=rate)
    aligns.append(p["align_r"])
    recs_af = [r for r in p["recs"] if r[1]]
    n_af_tot += len(recs_af)
    n_split_tot += sum(1 for r in recs_af if r[2] >= mc.SPLIT_HI)
    n_unsplit_tot += sum(1 for r in recs_af if r[2] <= mc.UNSPLIT_LO)
    al_b = mc.tokens(p["recs"], mc.sel_all)
    per_block_p50_all.append(mc.pctl(al_b, .50))
    SP += mc.tokens(p["recs"], mc.sel_split)
    AL += mc.tokens(p["recs"], mc.sel_all)
p50_sp, p95_sp = mc.pctl(SP, .50), mc.pctl(SP, .95)
p50_all = mc.pctl(AL, .50)
agree = (abs(p50_sp - p50_all) <= 0.05 * max(p50_sp, p50_all, 1e-9)) if (SP and AL) else False
ambig_frac_pooled = (n_af_tot - n_split_tot - n_unsplit_tot) / n_af_tot if n_af_tot else float("nan")
import statistics
TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
m = statistics.fmean(per_block_p50_all)
sd = statistics.stdev(per_block_p50_all) if len(per_block_p50_all) > 1 else float("nan")
h = TCRIT.get(len(per_block_p50_all), 1.96) * sd / math.sqrt(len(per_block_p50_all)) if sd == sd else float("nan")
print(f"S2A_POOLED_TELEMETRY_PATH arm={ARM} cell={CELL} n_split_tokens={len(SP)} n_all_tokens={len(AL)} "
      f"p50_SPLIT={p50_sp:.2f}ms p95_SPLIT={p95_sp:.2f}ms p50_ALL={p50_all:.2f}ms "
      f"agree_within_5pct={agree} align_r_range=[{min(aligns):.3f},{max(aligns):.3f}] "
      f"n_a_free={n_af_tot} n_split_label={n_split_tot} n_unsplit_label={n_unsplit_tot} "
      f"ambig_frac={ambig_frac_pooled:.4f}")
print(f"S2A_POOLED_TELEMETRY_PATH_BLOCKCI arm={ARM} cell={CELL} "
      f"per_block_p50_all={['%.2f'%x for x in per_block_p50_all]} "
      f"mean={m:.3f} sd={sd:.3f} t95_half_width={h:.3f} "
      f"t95_CI=[{m-h:.3f},{m+h:.3f}] (n={len(per_block_p50_all)} blocks)")

# --- (2) raw-itls path (independent of telemetry / m3_conditional / a_free) -
RAW = []
per_block_raw_p50 = []
for b in BLOCKS:
    o = json.loads(open(f"{d}/e1m3_{ARM}_{CELL}_b{b}_{job}_s{b}.jsonl").read().strip().split("\n")[0])
    tt, il = o["ttfts"], o["itls"]
    err = o.get("errors") or []
    nw = math.ceil(warmup_s * rate)
    keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
    raw_b = [dd * 1000.0 for i in keep for dd in (il[i] or [])]
    per_block_raw_p50.append(pctl(raw_b, .50))
    RAW += raw_b
raw_p50 = pctl(RAW, .50)
raw_p95 = pctl(RAW, .95)
m2 = statistics.fmean(per_block_raw_p50)
sd2 = statistics.stdev(per_block_raw_p50) if len(per_block_raw_p50) > 1 else float("nan")
h2 = TCRIT.get(len(per_block_raw_p50), 1.96) * sd2 / math.sqrt(len(per_block_raw_p50)) if sd2 == sd2 else float("nan")
print(f"S2A_POOLED_RAW_ITLS_PATH arm={ARM} cell={CELL} n_itl={len(RAW)} "
      f"p50={raw_p50:.3f}ms p95={raw_p95:.3f}ms "
      f"(no a_free filter, no split_frac, no telemetry file read)")
print(f"S2A_POOLED_RAW_ITLS_PATH_BLOCKCI arm={ARM} cell={CELL} "
      f"per_block_p50={['%.2f'%x for x in per_block_raw_p50]} "
      f"mean={m2:.3f} sd={sd2:.3f} t95_half_width={h2:.3f} "
      f"t95_CI=[{m2-h2:.3f},{m2+h2:.3f}] (n={len(per_block_raw_p50)} blocks)")

# --- prereg verdict-rule readout (VALUES ONLY, no interpretation forced) ---
for label, val in (("telemetry_path", p50_all), ("raw_itls_path", raw_p50)):
    if 12.0 <= val <= 13.0:
        band = "12-13ms band (Sec0-STANDS prediction)"
    elif 28.0 <= val <= 30.0:
        band = "~29ms band (Sec0-COLLAPSES prediction)"
    else:
        band = "INDETERMINATE (outside both pre-registered bands)"
    print(f"S2A_PREREG_BAND path={label} p50={val:.3f}ms -> {band}")

print("NOTE: bands above are a MECHANICAL restatement of the task brief's three")
print("pre-registered intervals, fixed before submission. No throughput / goodput /")
print("g / G_LEVER / G_FLAT / policy judgment is made by this job.")
PY

echo "=== S2A done job=$JOBTAG. Artifacts under $HERE. Analysis/interpretation: result-analyst/claims-auditor. ==="
