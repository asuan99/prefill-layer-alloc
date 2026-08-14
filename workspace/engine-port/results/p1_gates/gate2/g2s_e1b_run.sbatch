#!/bin/bash
#SBATCH --job-name=g2se1b
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --time=02:30:00
#SBATCH --output=g2se1b_%j.out
#SBATCH --error=g2se1b_%j.err
#
# E1-b / E1-c -- PREREG_G2S_E1B_E1C_2026-08-14.md  (engine-porter, Phase 0)
#
# ============================ CANONICAL ORDER =============================
# THE PRE-REGISTRATION IS CANONICAL AND THIS SCRIPT FOLLOWS IT.  If running
# this reveals a need to deviate, STOP AND REPORT -- do not edit the rules
# here.  Methodology gate #20 is "the prereg table has a comparison the scorer
# never computes"; its reverse ("the harness does something the prereg never
# registered") is the same failure.
#
# THIS IS A NEW FILE, DERIVED FROM g2s_run.sbatch BY COPY.  g2s_run.sbatch is
# the audited producer of the rev20 Gate 2-S result and is NOT edited: editing
# it in place would (a) change the harness that produced 877756/877757 and
# (b) invalidate the byte-provenance of that campaign's artifacts.
#
# ========================= PRE-REGISTERED RULES ===========================
# (echoed to stdout below, before any measurement)
#
#  ARMS (2 -- both are Gate 2-S arm A4 "agnostic"; NO other arm exists here):
#    a4_force1  --enable-pdmux, no R2 policy, PDMUX_TRACE_FORCE_PREFILL=1
#    a4_force0  --enable-pdmux, no R2 policy, force flag UNSET   (paired control)
#  Everything else is byte-identical to g2s_run.sbatch's A4 boot line.
#
#  CELLS: Zamba2 {2,3} + Zamba2 {6}=E1-c CONTROL / Granite {3,4}.  n = 6 paired.
#  seed = 9000 + 100*rep + rate  (same formula as 873944/875344/875346/877756).
#  Arm order is a fresh random permutation per rep.  Rates run ASCENDING inside
#  one boot (same shape as G1-b job 875293, which is where the r6 prior=40 came
#  from), so the control is measured under the same within-boot carryover.
#
#  DECISION QUANTITIES (prereg sec5):
#    P1  pooled max(decode_running_batch_size) over force=1 rows vs 36
#    P2  producer withdrawal rule frac((54,54)) >= 0.01  -- consulted ONLY when
#        P1 does not clear (the identity max<36 => frac=0 makes it redundant
#        otherwise; methodology gate #9 FIX, not a confession)
#    P3  D = max(decode_bs | cohabiting, dense) - max(decode_bs | cohabiting,
#        scheduled-only), paired t 95% CI.  BOOTSTRAP FORBIDDEN (gate #14).
#    C1  E1-c control on Zamba2 r6: the screen must be able to fire (gate #15)
#    O1  force1-vs-force0 alpha ratio, self-nullification only, RATIO ONLY
#
#  THIS CAMPAIGN WRITES NO Gate 2-S PREMISE LABEL, CHANGES NO PERFORMANCE
#  VERDICT, AND LICENSES NO UPGRADE OF sec1-1 ATTRIBUTION.
#
#  MANIFEST TRIPWIRE: inherited verbatim from g2s_run.sbatch:81-89 (added==2,
#  removed==0 vs 873944).  This experiment changes NO engine source, so the
#  tripwire must hold as-is.  *** IF G1-a IS APPLIED TO THE TREE FIRST, THIS
#  JOB DIES WITH exit 4.  E1-b/E1-c MUST BE SUBMITTED BEFORE G1-a. ***
# ==========================================================================
set -uo pipefail

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
cd "$ROOT"

TAG="${G2S_TAG:?set G2S_TAG=zamba2-27b|granite-40-h-micro-base}"
MODE="${E1B_MODE:-campaign}"          # campaign | smoke   (prereg sec10)
case "$TAG" in
  zamba2-27b)              MODEL="Zyphra/Zamba2-2.7B";                   CTX=4096; BACKEND=triton;     RATES="2 3 6" ;;
  granite-40-h-micro-base) MODEL="ibm-granite/granite-4.0-h-micro-base"; CTX=8192; BACKEND=flashinfer; RATES="3 4"   ;;
  *) echo "unknown G2S_TAG=$TAG" >&2; exit 2 ;;
esac
# ^ RATES is per-model and hard-coded, exactly as in g2s_run.sbatch:54-55.  The
#   ONLY difference is the appended `6` on Zamba2: that is the E1-c positive
#   control cell (G1-b job 875293 observed max(decode_bs)=40 >= 36 there, and
#   this campaign's own Gate 2-S capscan observed 31 < 36 -- i.e. the threshold
#   is bracketed and the control can genuinely fail).  Granite is unchanged.

OUT="$ROOT/workspace/engine-port/results/p1_gates/gate2"
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
PREFIX=g2se1b
ARMS="a4_force1 a4_force0"
if [ "$MODE" = smoke ]; then
  NREP=2; NPROMPT=20; WARMUP=2; NCONC_GATE=4
else
  NREP=6; NPROMPT=120; WARMUP=8; NCONC_GATE=16
fi
export G2_GREEDY_MAX_TIME=180
export G2_GREEDY_CONNECT_TIMEOUT=10

echo "==================== PRE-REGISTERED DECISION RULES ===================="
sed -n '/^# ========================= PRE-REGISTERED RULES/,/^# =====================/p' "$0"
echo "======================================================================="
echo "tag=$TAG model=$MODEL ctx=$CTX backend=$BACKEND rates=[$RATES] nrep=$NREP mode=$MODE job=$SLURM_JOB_ID"
if [ "$MODE" = smoke ]; then
  echo "*** PLUMBING SMOKE (methodology gate #26).  DO NOT CITE ANY NUMBER FROM THIS JOB."
  echo "*** A smoke PASS carries ZERO information about any decision quantity."
  echo "*** The scorer is EXPECTED to print 'I4 rep count 2 != 6' -> UNDETERMINED."
fi

# ---------------------------------------------------------------- manifest
MANIFEST="$OUT/runtime_source_manifest_e1b_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || { echo "SYNC FAILED" >&2; exit 3; }
DIFF="$OUT/manifest_diff_${PREFIX}_${TAG}_${SLURM_JOB_ID}.txt"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$DIFF"
N_ADD=$(grep -c '^>' "$DIFF" || true); N_DEL=$(grep -c '^<' "$DIFF" || true)
echo "MANIFEST added=$N_ADD removed=$N_DEL (expect added=2 removed=0 -- HOLB probe files)"
if [ "$N_ADD" != "2" ] || [ "$N_DEL" != "0" ]; then
  echo "MANIFEST_CHECK=FAIL -- engine tree drifted from the 873944-verified state." >&2
  echo "  If this fired because G1-a was applied first, that is the documented" >&2
  echo "  ordering violation (prereg sec7): revert G1-a, run E1-b/c, then G1-a." >&2
  cat "$DIFF" >&2; exit 4
fi
echo "MANIFEST_CHECK=PASS"
echo "NOTE (gate #33): manifest sha equality covers the 15 tracked files only; it is"
echo "     not a proof of runtime byte-identity (sgl_kernel/spatial.py and"
echo "     sglang/srt/multiplex/pdmux_context.py are outside it)."

python -m unittest discover -s workspace/engine-port/tests -v 2>&1 \
  || { echo "CPU REGRESSION FAILED" >&2; exit 5; }
echo "CPU_REGRESSION=PASS"

python3 "$OUT/g2s_e1b_premise.py" --selftest \
  || { echo "SCORER SELFTEST FAILED (a declared diagnostic field has no production path)" >&2; exit 6; }

PORT=$((34000 + SLURM_JOB_ID % 400))
source "$OUT/g2_holb_phaseA_lib.sh"   # bounded greedy_call (gate #21 fix)

arm_flags () {   # identical for both arms -- the ONLY difference is env
  echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule"
}
arm_env () {   # emits `export` lines; TELEM=<path> must be preset by caller
  # NOTE: boot() unsets PDMUX_TRACE_FORCE_PREFILL *before* this eval runs
  # (g2s_run.sbatch:117-120 has the same ordering), so exporting the flag here
  # -- and ONLY here -- is the correct and only working place for it.  Setting
  # it in the SLURM environment does NOT reach the server.
  case "$1" in
    a4_force1) echo "export PDMUX_TELEMETRY_PATH=$TELEM PDMUX_DUAL_WORKER_TRACE_EVERY=32 PDMUX_TRACE_FORCE_PREFILL=1" ;;
    a4_force0) echo "export PDMUX_TELEMETRY_PATH=$TELEM PDMUX_DUAL_WORKER_TRACE_EVERY=32" ;;
  esac
}

boot () {   # $1=arm $2=port $3=srvlog $4=telemetry -> pid
  local ARM="$1" P="$2" LOG="$3"; export TELEM="$4"
  ( unset PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM PDMUX_STICKY_PARTITION PDMUX_TELEMETRY_PATH \
          PDMUX_DUAL_WORKER PDMUX_TRUE_DUAL_WORKER PDMUX_HOLB_PATH PDMUX_SLO_SCHED \
          PDMUX_LA_COORD PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
          PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_AGN2_SM PDMUX_LA_FLOOR_SM PDMUX_LA_RESERVE \
          SGLANG_ZAMBA_TIMING SGLANG_ZAMBA_TIMING_EVERY SGLANG_GRANITE_TIMING \
          SGLANG_NH_LAYER_TIMING
    # ^ the four SGLANG_*_TIMING names are unset, never set to "0": the model
    #   code reads them as bool(os.environ.get(NAME)) (zamba2.py:544,
    #   granitemoehybrid.py:474; nemotron_h.py:646 is the same shape), so the
    #   string "0" is TRUTHY and would silently switch per-layer timing ON.
    #   The code defect is filed for engine-porter and deliberately NOT fixed in
    #   this batch (fixing it would change the tree and trip the manifest gate).
    eval "$(arm_env "$ARM")"
    export PDMUX_RUN_ID="${PREFIX}_${TAG}_${SLURM_JOB_ID}" PDMUX_WORKLOAD_ID="$ARM"
    exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
      --dtype bfloat16 --attention-backend "$BACKEND" --disable-radix-cache \
      --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
      $(arm_flags "$ARM") --host 127.0.0.1 --port "$P" ) > "$LOG" 2>&1 &
  echo $!
}
wait_ready () { local PID="$1" P="$2" ok=0; for i in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED" >&2; break; }; sleep 5; done; echo $ok; }
kill_server () { kill "$1" 2>/dev/null; pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 3; }

record_flags () {  # realized flags + KV pool from the LOG (g2s_run.sbatch:133-150)
  local ARM="$1" P="$2" LOG="$3" F="$4"
  curl -s "http://127.0.0.1:$P/get_server_info" > "$F" || true
  python3 - "$LOG" "$F" <<'PYEOF'
import json, re, sys
log, fj = sys.argv[1:3]
txt = open(log, errors="ignore").read()
kv = re.search(r"KV Cache is allocated\. #tokens: (\d+)", txt)
cap = re.search(r"Capture cuda graph end\. Time elapsed: ([\d.]+) s\. mem usage=([\d.]+) GB", txt)
try: d = json.load(open(fj))
except Exception: d = {}
d["_kv_cache_tokens"] = int(kv.group(1)) if kv else None
d["_capture_s"] = float(cap.group(1)) if cap else None
d["_capture_gb"] = float(cap.group(2)) if cap else None
json.dump(d, open(fj, "w"), indent=1)
print(f"  flags: kv_tokens={d['_kv_cache_tokens']} capture={d['_capture_s']}s/{d['_capture_gb']}GB")
PYEOF
}

bench () {  # $1=port $2=rate $3=nprompt $4=seed $5=outfile $6=warmup $7=row-tag
  # --output-details is MANDATORY (bench_serving.py:1630 drops result_details
  # without it; that defect cost jobs 877107/877109).  Identical bench line to
  # g2s_run.sbatch:160-164 so the workload is the Gate 2-S cell's workload.
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$1" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 \
    --random-range-ratio 1.0 --request-rate "$2" --num-prompts "$3" --seed "$4" \
    --warmup-requests "$6" --output-details --output-file "$5" --tag "$7" \
    >> "$OUT/${PREFIX}_bench_${TAG}_${SLURM_JOB_ID}.log" 2>&1
}

# ================== correctness gate (prereg sec4 G-1/G-2/G-3) =============
echo; echo "===== CORRECTNESS GATE ====="
PROMPT="$OUT/${PREFIX}_prompt_${TAG}_${SLURM_JOB_ID}.txt"
python3 -c "open('$PROMPT','w').write(' '.join(['token']*2000))"
for ARM in $ARMS; do
  LOG="$OUT/${PREFIX}_${TAG}_srv_G1_${ARM}_${SLURM_JOB_ID}.log"
  TEL="$OUT/${PREFIX}_${TAG}_telemetry_G1_${ARM}_${SLURM_JOB_ID}.jsonl"
  PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
  [ "$OK" = 1 ] || { echo "G1 boot FAILED arm=$ARM"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; }
  record_flags "$ARM" "$PORT" "$LOG" "$OUT/${PREFIX}_${TAG}_flags_${ARM}_${SLURM_JOB_ID}.json"
  python3 "$OUT/g2s_gate.py" --port "$PORT" --arm "$ARM" --prompt-file "$PROMPT" \
    --n-concurrent "$NCONC_GATE" \
    --baseline "$OUT/${PREFIX}_${TAG}_selfbaseline_${ARM}_${SLURM_JOB_ID}.json" \
    --out "$OUT/${PREFIX}_${TAG}_correctness_${ARM}_${SLURM_JOB_ID}.json"
  kill_server "$PID" "$PORT"
done
# G-3: the force flag is documented as OBSERVATION ONLY (multiplexing_mixin.py:
# 502-525 "nothing below this line feeds scheduling, partition selection or
# batch composition").  This is the direct test of that claim: batch=1 greedy
# output must be identical across the two arms.  A mismatch is reported, never
# silently accepted -- but per gate #21 a transport failure is UNDETERMINED.
python3 - "$OUT" "$PREFIX" "$TAG" "$SLURM_JOB_ID" <<'PYEOF'
import json, os, sys
out, pre, tag, job = sys.argv[1:5]
shas = {}
for arm in ("a4_force1", "a4_force0"):
    p = f"{out}/{pre}_{tag}_selfbaseline_{arm}_{job}.json"
    shas[arm] = json.load(open(p))["sha"] if os.path.exists(p) else None
if None in shas.values():
    print(f"G3_FORCE_FLAG_NEUTRALITY=UNDETERMINED (missing baseline) {shas}")
elif shas["a4_force1"] == shas["a4_force0"]:
    print("G3_FORCE_FLAG_NEUTRALITY=PASS (batch=1 greedy sha identical across force 1/0)")
else:
    print(f"G3_FORCE_FLAG_NEUTRALITY=MISMATCH {shas} -- prereg sec7 self-nullification 4")
json.dump(shas, open(f"{out}/{pre}_{tag}_g3_force_neutrality_{job}.json", "w"), indent=1)
PYEOF

# ===================== Phase 1 -- 2 arms x rates x reps ====================
echo; echo "===== PHASE 1: [$ARMS] x [$RATES] x $NREP reps, per-(arm,rep) boot ====="
for REP in $(seq 1 $NREP); do
  ORDER="$(shuf -e $ARMS | tr '\n' ' ')"
  echo "rep=$REP arm_order=$ORDER"
  echo "{\"rep\":$REP,\"order\":\"$ORDER\"}" >> "$OUT/${PREFIX}_${TAG}_arm_order_${SLURM_JOB_ID}.jsonl"
  for ARM in $ORDER; do
    LOG="$OUT/${PREFIX}_${TAG}_srv_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
    TEL="$OUT/${PREFIX}_${TAG}_telemetry_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"
    PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
    if [ "$OK" != 1 ]; then echo "  BOOT FAILED arm=$ARM rep=$REP"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; fi
    # Per-CELL (per-rate) telemetry slicing.  Telemetry is per-BOOT and a boot
    # serves every rate, so record line offsets around each rate's bench.  This
    # is the g2s_run.sbatch:254-265 bookkeeping, preserved VERBATIM -- the
    # scorer's rep-boundary reconstruction depends on these `_teloffset_` files
    # and refuses gap heuristics.
    OFF="$OUT/${PREFIX}_${TAG}_teloffset_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
    echo "{" > "$OFF"; FIRST=1
    for R in $RATES; do
      S0=$(wc -l < "$TEL" 2>/dev/null || echo 0)
      bench "$PORT" "$R" "$NPROMPT" $((9000 + 100*REP + R)) \
        "$OUT/${PREFIX}_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl" "$WARMUP" \
        "rep${REP}_${ARM}_r${R}"
      S1=$(wc -l < "$TEL" 2>/dev/null || echo 0)
      [ $FIRST -eq 1 ] || echo "," >> "$OFF"; FIRST=0
      printf '"r%s":[%s,%s]' "$R" "$S0" "$S1" >> "$OFF"
    done
    echo "}" >> "$OFF"
    kill_server "$PID" "$PORT"
  done
done

# Per-CELL concatenated telemetry.  NOT consumed by g2s_e1b_premise.py (which
# reads the per-boot slices directly, so E1's I3/I5 cell-file cross-checks have
# nothing to cross-check and are deliberately not inherited -- prereg sec4).
# Built for auditors only.
python3 - "$OUT" "$PREFIX" "$TAG" "$SLURM_JOB_ID" "$RATES" "$ARMS" <<'PYEOF'
import glob, json, os, sys
out, pre, tag, job, rates, arms = sys.argv[1:7]
for arm in arms.split():
    for r in rates.split():
        rows = []
        for tel in sorted(glob.glob(f"{out}/{pre}_{tag}_telemetry_{arm}_rep*_{job}.jsonl")):
            off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
            if not os.path.exists(off):
                continue
            try: sl = json.load(open(off)).get(f"r{r}")
            except Exception: sl = None
            if not sl:
                continue
            with open(tel) as fh:
                rows += fh.readlines()[sl[0]:sl[1]]
        dst = f"{out}/{pre}_{tag}_telemetry_{arm}_r{r}_{job}.jsonl"
        open(dst, "w").writelines(rows)
        print(f"  cell telemetry {arm} r{r}: {len(rows)} rows -> {os.path.basename(dst)}")
PYEOF

# ==========================================================================
# MEASUREMENT INVENTORY PREFLIGHT (methodology gate #21 and its mirror image).
#   exit 0  = MEASURED + SCORED
#   exit 20 = MEASURED_OK / SCORING_FAILED   -> rescore offline, 0 GPU
#   exit 21 = MEASUREMENT_INCOMPLETE         -> a GPU re-run is required
#   exit 22 = INDETERMINATE                  -> human judgment, do NOT spend GPU
# A VERDICT IS NEVER AN EXIT CODE.  UNDETERMINED/REFUTED/PREMISE_HOLDS are
# scientific outcomes and all of them exit 0.
# ==========================================================================
echo; echo "===== MEASUREMENT INVENTORY PREFLIGHT ====="
python3 - "$OUT" "$PREFIX" "$TAG" "$SLURM_JOB_ID" "$RATES" "$ARMS" "$NREP" <<'PYEOF'
import glob, json, os, sys
out, pre, tag, job, rates, arms, nrep = sys.argv[1:8]
nrep = int(nrep); bad = []; have = 0; want = 0
for arm in arms.split():
    for rep in range(1, nrep + 1):
        f = f"{out}/{pre}_{tag}_{arm}_rep{rep}_{job}.jsonl"
        tel = f"{out}/{pre}_{tag}_telemetry_{arm}_rep{rep}_{job}.jsonl"
        off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
        seen = set()
        if os.path.exists(f):
            for ln in open(f):
                try: row = json.loads(ln)
                except Exception: continue
                t = str(row.get("tag", ""))
                if row.get("itls") and t.startswith(f"rep{rep}_{arm}_r"):
                    seen.add(t.rsplit("_r", 1)[1])
        for r in rates.split():
            want += 1
            if r in seen: have += 1
            else: bad.append(f"{arm} rep{rep} r{r}: no bench row with itls")
        for p in (tel, off):
            if not os.path.exists(p):
                bad.append(f"{arm} rep{rep}: missing {os.path.basename(p)}")
print(f"INVENTORY have={have}/{want} bench cells; defects={len(bad)}")
for b in bad[:20]: print("   ", b)
json.dump(dict(have=have, want=want, defects=bad),
          open(f"{out}/{pre}_preflight_{tag}_{job}.json", "w"), indent=1)
sys.exit(0 if not bad else 7)
PYEOF
PREFLIGHT_RC=$?
echo "PREFLIGHT_RC=$PREFLIGHT_RC (0=MEASUREMENT_OK, 7=MEASUREMENT_INCOMPLETE)"

echo; echo "===== SCORING ====="
python3 "$OUT/g2s_e1b_premise.py" --outdir "$OUT" --tag "$TAG" \
  --job-force1 "$SLURM_JOB_ID" --job-force0 "$SLURM_JOB_ID" \
  --json-out "$OUT/${PREFIX}_report_${TAG}_${SLURM_JOB_ID}.json" \
  2>&1 | tee "$OUT/${PREFIX}_scoreout_${TAG}_${SLURM_JOB_ID}.txt"
SCORE_RC=${PIPESTATUS[0]}

if   [ "$SCORE_RC" = 0 ] && [ "$PREFLIGHT_RC" = 0 ]; then STATUS=MEASURED_AND_SCORED;              RC=0
elif [ "$SCORE_RC" = 0 ];     then                        STATUS=SCORED_BUT_INVENTORY_UNCERTIFIED; RC=22
elif [ "$PREFLIGHT_RC" = 0 ]; then                        STATUS=MEASURED_OK_SCORING_FAILED;       RC=20
elif [ "$PREFLIGHT_RC" = 7 ]; then                        STATUS=MEASUREMENT_INCOMPLETE;           RC=21
else                                                      STATUS=INDETERMINATE;                    RC=22
fi

if [ "$MODE" = smoke ]; then
  ls "$OUT"/${PREFIX}_*"${SLURM_JOB_ID}"* > "$OUT/${PREFIX}smoke_manifest_${SLURM_JOB_ID}.txt" 2>/dev/null
  echo "SMOKE MANIFEST -> ${PREFIX}smoke_manifest_${SLURM_JOB_ID}.txt  (ALL DO-NOT-CITE)"
  echo "A smoke PASS is plumbing evidence only and carries ZERO information about"
  echo "P1/P2/P3/C1/O1 or about any Gate 2-S claim."
fi
echo "CAMPAIGN_STATUS=$STATUS scorer_rc=$SCORE_RC preflight_rc=$PREFLIGHT_RC exit=$RC"
[ "$RC" = 21 ] && echo "  -> GPU RE-RUN REQUIRED. Confirm against the preflight defect list first."
[ "$RC" = 22 ] && echo "  -> INDETERMINATE: human judgment. Do NOT re-run on GPU on this verdict alone."
echo "DONE tag=$TAG job=$SLURM_JOB_ID mode=$MODE"
exit "$RC"
