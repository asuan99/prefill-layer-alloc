#!/bin/bash
#SBATCH --job-name=s2sticky
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s2_sticky/%x_%j.err
# ============================================================================
# S2 -- FIRST STICKY-PARTITION CAUSAL RUN.  Pre-registered in
# ../s8_frontier/PREREG_S2_STICKY_ITL_2026-08-03.md; do not change the run
# shape or the sec 3 prediction intervals without editing that file FIRST
# (methodology gate #8: re-scoring after seeing results is forbidden).
#
# QUESTION (one, narrow, per the pre-registration): with
# PDMUX_STICKY_PARTITION=1 holding the green-context division through
# decode-busy/prefill-idle gaps (DESIGN.md sec 4.3.11), does T8's d16
# per-token ITL move toward C2's measured [prefill-fixed-16] range
# (~28-34 ms), or does it stay near 872077's sticky-OFF ~11 ms?  That
# distinguishes "the label under-isolated a real partition effect" from
# "the green context is not honored for graph-replayed decode".
#
# THIS IS A DIRECT DERIVATION of ../s8_frontier/e1_m3_control.sbatch.
# Changed: (1) PDMUX_STICKY_PARTITION=1 for every probe (sec 4.3.11),
# (2) arm set reduced to T8 ONLY (Ha8 excluded -- the pre-registration
# scoped this run to T8, sec 1), (3) cell set reduced to {d16, d54} ONLY
# (d24/d44 excluded, out of scope for S2), (4) output directory/job name,
# (5) an in-run gate/diagnostic block appended after each probe using the
# EXISTING e1_pin_check.py (E1_DECODE_REALIZED, the real gate under sticky)
# and the EXISTING m3_conditional.py (ALIGN_R flag, AMBIG_FRAC/MIN_N_SPLIT
# diagnostics, and the sec 3.1 p50(SPLIT) vs p50(all) agreement check).
# NOTHING ELSE IS CHANGED: server args are e1_m3_control.sbatch:212-218
# VERBATIM, cell configs (pdmux_e1_d16.yml / pdmux_e1_d54.yml) are the SAME
# files referenced from ../s8_frontier (not copied -- avoids drift), CAP,
# MAMBA_POOL, NP_MIN, PROBE_TARGET_S, WARMUP_S, RATE, BLOCKS, seed policy,
# and cell-order-alternates-by-block-parity are all unchanged.
#
# ★ NO KEEPALIVE.  This is a ShareGPT-trace probe (bench_serving, one
# request stream), exactly like e1_m3_control.sbatch -- there is no
# keepalive worker anywhere in that file or in this derivation, so the
# s8_scaleup keepalive/context-overflow defect (results/s8_scaleup/
# NOTES_D54_ANCHOR_2026-08-03.md Finding 1) does not apply here. Confirmed
# by inspection before submission, not just by assumption.
#
# ★ GUARD-SATISFIER ROW.  pdmux_e1_d54.yml's second manual_divisions row
# ([64,44,48], never selected under a plain fixed target) is the same file
# already exercised by test_on_with_fixed_target_ignores_the_guard_satisfier_row
# (DESIGN.md sec 4.3.11): sticky pins the FixedPolicy-RESOLVED index, not the
# decode-bs selector, so this run cannot drift onto the guard row. No yml
# change was needed or made.
#
# Nothing here is a performance/policy claim.  In-run gate/diagnostic output
# is descriptive only; the sec 4 three-way readout is for result-analyst /
# claims-auditor, not this job.
# Usage:
#   sbatch s2_sticky.sbatch
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
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_s2sticky_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest_s2.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

# --- arm: T8 ONLY (positive control from e1_m3_control.sbatch). Pre-registration
# sec 1 scoped this run to T8; Ha8 is explicitly excluded, not merely omitted. --
ARM_LABELS=(T8)
ARM_MODELS=("Qwen/Qwen2.5-7B")
ARM_BACKENDS=(triton)

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN="${SGPT_CTXLEN:-3600}"
CTXCAP=$(( CTX + 512 + 256 ))

# --- PRE-REGISTERED PARAMETERS, unchanged from e1_m3_control.sbatch --------
RATE="${RATE:-2}"
CELLS_FWD="${CELLS_FWD:-d16 d54}"   # PREREG_S2 sec 1: d16 primary, d54 companion.
                                   # d24/d44 are out of scope for S2 -- not run.
BLOCKS="${BLOCKS:-8}"              # replication unit = block (independent boot +
                                   # seed=block index), unchanged rationale from
                                   # e1_m3_control.sbatch (sec 4.3.8(c)).
NP_MIN="${NP_MIN:-200}"
PROBE_TARGET_S="${PROBE_TARGET_S:-8}"
WARMUP_S="${WARMUP_S:-3}"
CAP="${CAP:-48}"
MAMBA_POOL="${MAMBA_POOL:-$CAP}"    # no-op for T8 (pure Transformer, no mamba pool
                                   # line in its server log) -- kept only for
                                   # structural parity with e1_m3_control.sbatch.
CONC_ASSERT_FRAC="${CONC_ASSERT_FRAC:-0.8}"

# Plateau TTFT-p50 (ms), T8 only, same values as e1_m3_control.sbatch's PLATEAU
# table (measured at cap 48, off-cliff assertion only -- see that file's
# comments for the full rationale, unchanged here).
declare -A PLATEAU
PLATEAU[T8_d16]=48; PLATEAU[T8_d54]=64

JOBTAG="${SLURM_JOB_ID:-local}"
NP=$(python3 -c "import math; print(max($NP_MIN, math.ceil($RATE*$PROBE_TARGET_S)))")

echo "=== S2 STICKY (PREREG_S2_STICKY_ITL_2026-08-03.md) job=$JOBTAG ==="
echo "PDMUX_STICKY_PARTITION=1 for every probe (sec 4.3.11)."
echo "RATE=$RATE NP=$NP (NP_MIN=$NP_MIN PROBE_TARGET_S=$PROBE_TARGET_S) WARMUP_S=$WARMUP_S"
echo "CAP=$CAP MAMBA_POOL=$MAMBA_POOL CELLS_FWD=[$CELLS_FWD] BLOCKS=$BLOCKS (1 seed each, seed=block)"
echo "SGPT_CTXLEN=$SGPT_CTXLEN CTXCAP=$CTXCAP"

CELLS_REV=$(echo $CELLS_FWD | tr ' ' '\n' | tac | tr '\n' ' ')

for AIDX in 0; do
ARM="${ARM_LABELS[$AIDX]}"
MODEL="${ARM_MODELS[$AIDX]}"
BACKEND="${ARM_BACKENDS[$AIDX]}"
RESULT="$HERE/s2_${ARM}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "ARM=$ARM MODEL=$MODEL BACKEND=$BACKEND RATE=$RATE NP=$NP CAP=$CAP STICKY=1" | tee -a "$RESULT"

# A BLOCK = one independent replicate = one fresh boot of every cell, run at
# one seed.  Odd blocks use the forward cell order, even blocks the reverse, so
# cell is CROSSED with boot/time order (same convention as e1_m3_control.sbatch
# and s8_sweep_d54.sbatch -- guards against order confounding).
for BI in $(seq 1 $BLOCKS); do
  BLOCK="b${BI}"
  if [ $(( BI % 2 )) -eq 1 ]; then CELLS="$CELLS_FWD"; else CELLS="$CELLS_REV"; fi
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
                                                       # e1_m3_control.sbatch/
                                                       # m3_conditional.py's tag
                                                       # convention on purpose --
                                                       # this lets the EXISTING
                                                       # m3_conditional.label_probe()
                                                       # read these files with no
                                                       # path/name translation.
    SRVLOG="$HERE/${RUNID}_srv.log"
    export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
    export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="s2_${ARM}_${NAME}_${BLOCK}"
    export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
    export PDMUX_TRACE_FORCE_PREFILL=0
    export PDMUX_STICKY_PARTITION=1    # ★ the ONE substantive flag change vs
                                        # e1_m3_control.sbatch (PREREG_S2 sec 1)

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
        # Per-probe gates -- SAME logic/thresholds as e1_m3_control.sbatch
        # (GATE_CONC / GATE_OFFCLIFF / GATE_ERR), unchanged.
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

        # ------------------------------------------------------------------
        # ★ S2 IN-RUN GATES/DIAGNOSTICS (PREREG_S2 sec 2).  E1_DECODE_REALIZED
        # is computed with the EXISTING e1_pin_check.py (a real gate under
        # sticky, per DESIGN.md sec 4.3.11).  ALIGN_R (flag-only, never a
        # silent drop), AMBIG_FRAC, MIN_N_SPLIT and the sec 3.1 p50(SPLIT) vs
        # p50(all) agreement check use the EXISTING m3_conditional.py's
        # label_probe()/tokens()/pctl() -- no new gate logic is invented here,
        # and no pass/fail judgment beyond E1_DECODE_REALIZED>=0.90 (the one
        # value the pre-registration actually fixes a number for) is made.
        # AMBIG_FRAC/MIN_N_SPLIT are reported next to the 872077 reference
        # range for result-analyst/claims-auditor, not scored against an
        # invented threshold.
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
print(f"S2_GATE_DECODE_REALIZED arm={arm} cell={cell} block={block} D={dsm} "
      f"frac={frac:.4f} verdict={verdict} (gate: >=0.90, PREREG_S2 sec 2) "
      f"t_decode_active={dr['t_decode_active_s']:.1f}s hist=[{hist}]")

p = mc.label_probe(d, arm, cell, block, block, job, rate=rate)  # seed==block, campaign convention
align_flag = "" if p["align_r"] >= mc.ALIGN_R_MIN else "ALIGN-WEAK"
print(f"S2_GATE_ALIGN_R arm={arm} cell={cell} block={block} lag={p['lag']:.2f} "
      f"align_r={p['align_r']:.4f} flag={align_flag or 'none'} "
      f"(threshold {mc.ALIGN_R_MIN}, FLAG-ONLY per PREREG_S2 sec 2 -- never excluded)")

recs_af = [r for r in p["recs"] if r[1]]   # a_free_only population
n_af = len(recs_af)
n_split = sum(1 for r in recs_af if r[2] >= mc.SPLIT_HI)
n_unsplit = sum(1 for r in recs_af if r[2] <= mc.UNSPLIT_LO)
n_ambig = n_af - n_split - n_unsplit
ambig_frac = (n_ambig / n_af) if n_af else float("nan")
print(f"S2_DIAG_AMBIG_MIN_N_SPLIT arm={arm} cell={cell} block={block} "
      f"n_a_free={n_af} n_split={n_split} n_unsplit={n_unsplit} n_ambig={n_ambig} "
      f"ambig_frac={ambig_frac:.4f} "
      f"(872077 reference range: ambig_frac 0.003-0.014, n_split 1390-7045 per "
      f"DESIGN.md sec 4.3.12(c) -- NOT scored against an invented threshold here, "
      f"reported for result-analyst/claims-auditor)")

sp = mc.tokens(p["recs"], mc.sel_split)
al = mc.tokens(p["recs"], mc.sel_all)
p50_sp, p50_all = mc.pctl(sp, .50), mc.pctl(al, .50)
agree = (abs(p50_sp - p50_all) <= 0.05 * max(p50_sp, p50_all, 1e-9)) if (sp and al) else False
print(f"S2_DIAG_P50 arm={arm} cell={cell} block={block} n_split_tok={len(sp)} n_all_tok={len(al)} "
      f"p50_SPLIT={p50_sp:.2f}ms p50_ALL={p50_all:.2f}ms agree_within_5pct={agree} "
      f"(PREREG_S2 sec 3.1: p50(SPLIT) and p50(all) must agree within 5% pooled over "
      f"8 blocks or the run is UNMEASURABLE rather than negative -- per-block value "
      f"printed here for monitoring ONLY, the pooled/8-block judgment is not made by "
      f"this job)")
PY
      done
    fi
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    sleep 5
  done
done
done

# ============================================================================
# Pooled diagnostics (PREREG_S2 sec 3/3.1) -- ALL 8 blocks, both cells. Cheap
# (CPU-only re-labelling of files already on disk). Printed for monitoring/
# triage ONLY: this job does not adopt the sec 4 three-way readout or any
# g/G_LEVER/throughput/goodput judgment -- that is result-analyst/
# claims-auditor's call, on this same data.
# ============================================================================
echo "=== S2 pooled diagnostics (NOT a verdict; see PREREG_S2_STICKY_ITL_2026-08-03.md sec 3/3.1/4) ===" | tee -a "$HERE/s2_pooled_${JOBTAG}.txt"
python3 - "$HERE" "$JOBTAG" "$RATE" "$E1" <<'PY' | tee -a "$HERE/s2_pooled_${JOBTAG}.txt"
import sys
d, job, rate_s, moddir = sys.argv[1:5]
rate = float(rate_s)
sys.path.insert(0, moddir)
import m3_conditional as mc

ARM = "T8"
CELLS = ("d16", "d54")
BLOCKS = range(1, 9)

for cell in CELLS:
    SP, AL = [], []
    aligns = []
    n_af_tot = n_split_tot = n_unsplit_tot = 0
    for b in BLOCKS:
        p = mc.label_probe(d, ARM, cell, b, b, job, rate=rate)
        aligns.append(p["align_r"])
        recs_af = [r for r in p["recs"] if r[1]]
        n_af_tot += len(recs_af)
        n_split_tot += sum(1 for r in recs_af if r[2] >= mc.SPLIT_HI)
        n_unsplit_tot += sum(1 for r in recs_af if r[2] <= mc.UNSPLIT_LO)
        SP += mc.tokens(p["recs"], mc.sel_split)
        AL += mc.tokens(p["recs"], mc.sel_all)
    p50_sp, p95_sp = mc.pctl(SP, .50), mc.pctl(SP, .95)
    p50_all = mc.pctl(AL, .50)
    agree = (abs(p50_sp - p50_all) <= 0.05 * max(p50_sp, p50_all, 1e-9)) if (SP and AL) else False
    ambig_frac_pooled = (n_af_tot - n_split_tot - n_unsplit_tot) / n_af_tot if n_af_tot else float("nan")
    print(f"S2_POOLED arm={ARM} cell={cell} n_split_tokens={len(SP)} n_all_tokens={len(AL)} "
          f"p50_SPLIT={p50_sp:.2f}ms p95_SPLIT={p95_sp:.2f}ms p50_ALL={p50_all:.2f}ms "
          f"agree_within_5pct={agree} align_r_range=[{min(aligns):.3f},{max(aligns):.3f}] "
          f"n_a_free={n_af_tot} n_split_label={n_split_tot} n_unsplit_label={n_unsplit_tot} "
          f"ambig_frac={ambig_frac_pooled:.4f}")
print("NOTE: numbers above are DIAGNOSTIC ONLY, per-file breakdown is in the "
      "*_result.txt logs. The pre-registration's sec 4 three-way readout and any "
      "g/G_LEVER/throughput/goodput judgment are for result-analyst / "
      "claims-auditor, not this job.")
PY

echo "=== S2 done job=$JOBTAG. Artifacts under $HERE. Analysis: result-analyst/claims-auditor. ==="
