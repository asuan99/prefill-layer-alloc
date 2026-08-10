#!/bin/bash
#SBATCH --job-name=g2s
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --comment=pytorch
#SBATCH --time=06:00:00
#SBATCH --output=g2s_%j.out
#SBATCH --error=g2s_%j.err
#
# Gate 2-S main campaign -- PREREG_GATE2S_2026-08-09.md rev6 (submission build).
#
# ============================ CANONICAL ORDER =============================
# THE PRE-REGISTRATION IS CANONICAL AND THIS SCRIPT FOLLOWS IT.  If running
# this reveals a need to deviate, STOP AND REPORT -- do not edit the rules
# here.  Methodology gate #20 is "the prereg table has a comparison the scorer
# never computes"; its reverse ("the harness does something the prereg never
# registered") is the same failure.
#
# ========================= PRE-REGISTERED RULES ===========================
# (echoed to stdout below, before any measurement -- sec7 of the brief)
#
#  ARMS (5, sec2, all four cells):
#    A2  plainaux               --chunked-prefill-size -1 --disable-overlap-schedule
#    C   pdmux_nosplit          + --enable-pdmux  PDMUX_R2_POLICY=fixed R2_FIXED_DSM=108
#    T   pdmux_split34          + PDMUX_R2_FIXED_DSM=34  PDMUX_STICKY_PARTITION=1
#    T'  pdmux_split34_nosticky + PDMUX_R2_FIXED_DSM=34  (no sticky)   <- confirmatory
#    A4  agnostic               + no R2 policy (873944 selector)       <- anchor only
#  All three pdmux arms share ONE yml (same green-ctx count, same capture set,
#  therefore identical KV pool -- asserted in sec6.3-8).
#
#  CELLS: Zamba2 {2,3} / Granite {3,4}.  n = 10 paired.  DESIGNATED = Zamba2 r3.
#  seed = 9000 + 100*rep + rate  (same formula as 873944/875344/875346).
#  Arm order is a fresh random permutation of all 5 per rep (sec2.1).
#  Boot is per (arm, rep); probes ride those boots (sec6.2.1, zero extra boots).
#
#  VERDICTS: paired t 95% CI only (bootstrap FORBIDDEN, gate #14).
#            Holm p source = ALPHA only (sec5.6c).
#            F1 = confirmatory (C - T', 4 cells); F2 = secondary (C - T).
#            Claim printed only AFTER Holm (sec5.6d).
#            sec8.9.1 falsifier is ONE-WAY: {refuted, unchanged}; never "verified".
#            Delta_conc (C vs A2) is SIGN ONLY -- magnitude citation forbidden.
#  ANCHOR (sec5.5) is UNCONDITIONAL and must never be written as a decomposition
#  of the sec1-1 A4-vs-fused gap.
# ==========================================================================
set -uo pipefail

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
cd "$ROOT"

TAG="${G2S_TAG:?set G2S_TAG=zamba2-27b|granite-40-h-micro-base}"
case "$TAG" in
  zamba2-27b)              MODEL="Zyphra/Zamba2-2.7B";                    CTX=4096; BACKEND=triton;     RATES="2 3" ;;
  granite-40-h-micro-base) MODEL="ibm-granite/granite-4.0-h-micro-base";  CTX=8192; BACKEND=flashinfer; RATES="3 4" ;;
  *) echo "unknown G2S_TAG=$TAG" >&2; exit 2 ;;
esac

OUT="$ROOT/workspace/engine-port/results/p1_gates/gate2"
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
NREP=10
CAPSCAN_RATES="1 2 3 4 6"
CAPSCAN_SEEDS="7 17 27"
CAPSCAN_ARMS="plainaux pdmux_nosplit pdmux_split34 agnostic"   # sec6.4.1: T' NOT in capscan
SCORE_ARMS="plainaux pdmux_nosplit pdmux_split34 pdmux_split34_nosticky agnostic"
PILOT_ARMS="pdmux_nosplit pdmux_split34"                        # sec3.4(a), designated cell, n=4
NCONC_GATE=16
export G2_GREEDY_MAX_TIME=180
export G2_GREEDY_CONNECT_TIMEOUT=10

echo "==================== PRE-REGISTERED DECISION RULES ===================="
sed -n '/^# ========================= PRE-REGISTERED RULES/,/^# =====================/p' "$0"
echo "======================================================================="
echo "§9 아티팩트 레이아웃 주: 스코어러는 §9가 개별 파일로 예시한 env/delivery/premise/kvpool/pilot/bimodality 산출을 단일 g2s_report_<tag>_<jobid>.json의 키로 통합한다. 내용 보존은 제출 전 확인됨(experiment-runner, 2026-08-10). 절차 이탈 아님."
echo "tag=$TAG model=$MODEL ctx=$CTX backend=$BACKEND rates=[$RATES] nrep=$NREP job=$SLURM_JOB_ID"

# ---------------------------------------------------------------- manifest
MANIFEST="$OUT/runtime_source_manifest_s_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || { echo "SYNC FAILED" >&2; exit 3; }
DIFF="$OUT/manifest_diff_g2s_${TAG}_${SLURM_JOB_ID}.txt"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$DIFF"
N_ADD=$(grep -c '^>' "$DIFF" || true); N_DEL=$(grep -c '^<' "$DIFF" || true)
echo "MANIFEST added=$N_ADD removed=$N_DEL (expect added=2 removed=0 -- HOLB probe files)"
if [ "$N_ADD" != "2" ] || [ "$N_DEL" != "0" ]; then
  echo "MANIFEST_CHECK=FAIL -- engine tree drifted from the 873944-verified state" >&2
  cat "$DIFF" >&2; exit 4
fi
echo "MANIFEST_CHECK=PASS"

python -m unittest discover -s workspace/engine-port/tests -v 2>&1 \
  || { echo "CPU REGRESSION FAILED" >&2; exit 5; }
echo "CPU_REGRESSION=PASS"

PORT=$((34000 + SLURM_JOB_ID % 400))
source "$OUT/g2_holb_phaseA_lib.sh"   # bounded greedy_call (gate #21 fix)

arm_flags () {
  case "$1" in
    plainaux) echo "--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    pdmux_nosplit|pdmux_split34|pdmux_split34_nosticky|agnostic)
      echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
  esac
}
arm_env () {   # emits `export` lines; TELEM=<path> must be preset by caller
  case "$1" in
    plainaux) ;;                                            # no pdmux env at all
    pdmux_nosplit)          echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=108 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    pdmux_split34)          echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=34 PDMUX_STICKY_PARTITION=1 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    pdmux_split34_nosticky) echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=34 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    agnostic)               echo "export PDMUX_TELEMETRY_PATH=$TELEM" ;;
  esac
}

boot () {   # $1=arm $2=port $3=srvlog $4=telemetry -> pid
  local ARM="$1" P="$2" LOG="$3"; export TELEM="$4"
  ( unset PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM PDMUX_STICKY_PARTITION PDMUX_TELEMETRY_PATH \
          PDMUX_DUAL_WORKER PDMUX_TRUE_DUAL_WORKER PDMUX_HOLB_PATH PDMUX_SLO_SCHED \
          PDMUX_LA_COORD PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL
    eval "$(arm_env "$ARM")"
    export PDMUX_RUN_ID="g2s_${TAG}_${SLURM_JOB_ID}" PDMUX_WORKLOAD_ID="$ARM"
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

record_flags () {  # sec1.2 realized flags + sec6.3-8 KV pool from the LOG
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
d["_kv_cache_tokens"] = int(kv.group(1)) if kv else None      # sec2.3-b: /get_server_info
d["_capture_s"] = float(cap.group(1)) if cap else None        # gives max_total_tokens=null
d["_capture_gb"] = float(cap.group(2)) if cap else None       # in 114/114 rev4 files
json.dump(d, open(fj, "w"), indent=1)
print(f"  flags: kv_tokens={d['_kv_cache_tokens']} capture={d['_capture_s']}s/{d['_capture_gb']}GB")
PYEOF
}

bench () {  # $1=port $2=rate $3=nprompt $4=seed $5=outfile $6=warmup $7=row-tag
  # --output-details is MANDATORY: without it bench_serving.py:1630 dumps only the
  # aggregate `result` and drops `result_details` (itls/ttfts/input_lens/output_lens).
  # The pre-registered primaries (sec4.2 alpha=ITLp95_mean, beta=pooled p99) and the
  # sec6.4 F-series are computed from per-request `itls`/`ttfts`; the aggregate
  # p95_itl_ms is a run-level estimand and is NOT a substitute (post-hoc metric swap).
  # Same flag as g2_run.sbatch:345 (capscan) and g2_run.sbatch:403 (phase 1).
  # --tag is provenance only (bench_serving.py:1551); no scorer decision reads it.
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$1" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 \
    --random-range-ratio 1.0 --request-rate "$2" --num-prompts "$3" --seed "$4" \
    --warmup-requests "$6" --output-details --output-file "$5" --tag "$7" \
    >> "$OUT/g2s_bench_${TAG}_${SLURM_JOB_ID}.log" 2>&1
}

# ======================= Phase 0a -- capscan (4 arms) ======================
echo; echo "===== PHASE 0a capscan (arms: $CAPSCAN_ARMS ; T' excluded by sec6.4.1) ====="
for ARM in $CAPSCAN_ARMS; do
  for SEED in $CAPSCAN_SEEDS; do
    LOG="$OUT/g2s_${TAG}_srv_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.log"
    TEL="$OUT/g2s_${TAG}_telemetry_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.jsonl"
    PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
    [ "$OK" = 1 ] || { echo "capscan boot FAILED arm=$ARM seed=$SEED"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; }
    for R in $CAPSCAN_RATES; do
      bench "$PORT" "$R" 60 "$SEED" "$OUT/g2s_${TAG}_capscan_${ARM}_seed${SEED}_${SLURM_JOB_ID}.jsonl" 4 \
            "capscan_${ARM}_seed${SEED}_r${R}"
    done
    kill_server "$PID" "$PORT"
    echo "  capscan done arm=$ARM seed=$SEED"
  done
done

# =================== Phase 0b -- T/C pilot (sec3.4a) =======================
echo; echo "===== PHASE 0b T/C pilot -- designated cell only, n=4, scoring config ====="
DES_RATE=$(echo $RATES | awk '{print $NF}')
if [ "$TAG" = zamba2-27b ]; then
  for ARM in $PILOT_ARMS; do
    for REP in 1 2 3 4; do
      LOG="$OUT/g2s_${TAG}_srv_pilot_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
      TEL="$OUT/g2s_${TAG}_telemetry_pilot_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"
      PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
      [ "$OK" = 1 ] || { kill_server "$PID" "$PORT"; continue; }
      bench "$PORT" 3 120 $((9000 + 100*REP + 3)) \
        "$OUT/g2s_${TAG}_pilot_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl" 8 \
        "pilot_${ARM}_rep${REP}_r3"
      kill_server "$PID" "$PORT"
    done
  done
  echo "  pilot done (df=3; sigma_D CI ~ [0.57x, 3.73x] -- report with every MDE)"
else
  echo "  pilot is Zamba2-only (designated cell) -- skipped"
fi

# ================== correctness gate G-1/G-2 (sec6.3) ======================
echo; echo "===== CORRECTNESS GATE G-1/G-2 (sec6.3-4) ====="
PROMPT="$OUT/g2s_prompt_${TAG}_${SLURM_JOB_ID}.txt"
python3 -c "
import sys
sys.path.insert(0,'$ROOT/workspace/engine-port')
open('$PROMPT','w').write(' '.join(['token']*2000))
"
for ARM in $SCORE_ARMS; do
  LOG="$OUT/g2s_${TAG}_srv_G1_${ARM}_${SLURM_JOB_ID}.log"
  TEL="$OUT/g2s_${TAG}_telemetry_G1_${ARM}_${SLURM_JOB_ID}.jsonl"
  PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
  [ "$OK" = 1 ] || { echo "G1 boot FAILED arm=$ARM"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; }
  record_flags "$ARM" "$PORT" "$LOG" "$OUT/g2s_${TAG}_flags_${ARM}_${SLURM_JOB_ID}.json"
  python3 "$OUT/g2s_gate.py" --port "$PORT" --arm "$ARM" --prompt-file "$PROMPT" \
    --n-concurrent "$NCONC_GATE" \
    --baseline "$OUT/g2s_${TAG}_selfbaseline_${ARM}_${SLURM_JOB_ID}.json" \
    --out "$OUT/g2s_${TAG}_correctness_${ARM}_${SLURM_JOB_ID}.json"
  kill_server "$PID" "$PORT"
done
python3 - "$OUT" "$TAG" "$SLURM_JOB_ID" <<'PYEOF'
import glob, json, sys
out, tag, job = sys.argv[1:4]
ks = {}
for f in glob.glob(f"{out}/g2s_{tag}_correctness_*_{job}.json"):
    d = json.load(open(f)); ks[d["arm"]] = d["verdict"]
print("  correctness verdicts:", ks)
print("  NOTE gate #21: UNDETERMINED is a MEASUREMENT failure, never a mismatch,")
print("       and there is no code path from UNDETERMINED to FAIL.")
PYEOF

# ===================== Phase 1 -- scoring, 5 arms ==========================
echo; echo "===== PHASE 1 scoring: 5 arms x [$RATES] x $NREP reps, per-rep boot ====="
for REP in $(seq 1 $NREP); do
  ORDER="$(shuf -e $SCORE_ARMS | tr '\n' ' ')"
  echo "rep=$REP arm_order=$ORDER"
  echo "{\"rep\":$REP,\"order\":\"$ORDER\"}" >> "$OUT/g2s_${TAG}_arm_order_${SLURM_JOB_ID}.jsonl"
  for ARM in $ORDER; do
    LOG="$OUT/g2s_${TAG}_srv_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
    TEL="$OUT/g2s_${TAG}_telemetry_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"
    PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
    if [ "$OK" != 1 ]; then echo "  BOOT FAILED arm=$ARM rep=$REP"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; fi
    # probes ride this boot, BEFORE the scored bench (sec6.2.1)
    python3 "$OUT/g2s_probe.py" --port "$PORT" --arm "$ARM" --rep "$REP" \
      --out "$OUT/g2s_${TAG}_probes_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
    # sec8.9.1 needs a PER-CELL (per-rate) telemetry slice, but telemetry is
    # per-BOOT and a boot serves both rates.  Record line offsets around each
    # rate's bench so the scorer can slice exactly.  This is harness bookkeeping,
    # NOT a change to the pre-registered analysis.
    OFF="$OUT/g2s_${TAG}_teloffset_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
    echo "{" > "$OFF"; FIRST=1
    for R in $RATES; do
      S0=$(wc -l < "$TEL" 2>/dev/null || echo 0)
      bench "$PORT" "$R" 120 $((9000 + 100*REP + R)) \
        "$OUT/g2s_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl" 8 \
        "rep${REP}_${ARM}_r${R}"
      S1=$(wc -l < "$TEL" 2>/dev/null || echo 0)
      [ $FIRST -eq 1 ] || echo "," >> "$OFF"; FIRST=0
      printf '"r%s":[%s,%s]' "$R" "$S0" "$S1" >> "$OFF"
    done
    echo "}" >> "$OFF"
    kill_server "$PID" "$PORT"
  done
done

# sec8.9.1: build per-CELL telemetry from the per-BOOT files using the recorded
# offsets.  Only A4 is consumed by the falsifier; the rest are built for audit.
python3 - "$OUT" "$TAG" "$SLURM_JOB_ID" "$RATES" <<'PYEOF'
import glob, json, os, sys
out, tag, job, rates = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
for arm in ("plainaux","pdmux_nosplit","pdmux_split34","pdmux_split34_nosticky","agnostic"):
    for r in rates:
        rows = []
        for tel in sorted(glob.glob(f"{out}/g2s_{tag}_telemetry_{arm}_rep*_{job}.jsonl")):
            off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
            if not os.path.exists(off):
                continue
            try: sl = json.load(open(off)).get(f"r{r}")
            except Exception: sl = None
            if not sl:
                continue
            with open(tel) as fh:
                rows += fh.readlines()[sl[0]:sl[1]]
        dst = f"{out}/g2s_{tag}_telemetry_{arm}_r{r}_{job}.jsonl"
        open(dst, "w").writelines(rows)
        print(f"  cell telemetry {arm} r{r}: {len(rows)} rows -> {os.path.basename(dst)}")
PYEOF

# ==========================================================================
# SCORING.  Methodology gate #21: a scoring failure must NOT be reported as a
# successful job, and it must be distinguishable from a MEASUREMENT failure.
# Job 877107/877109 exited 0:0 while the scorer died on KeyError:'itls' --
# `set -uo pipefail` without `-e` means the pipeline's status was discarded and
# the script's exit code was the trailing `echo`.
#
# ★ THE COST OF MISCLASSIFYING IS ASYMMETRIC:
#     exit 21 => somebody burns ~6.4 GPU-hr re-running the campaign
#     exit 20 => somebody reruns the scorer on the existing files, 0 GPU-hr
#   Job 877593 exited 21 while the measurement was PERFECT (all 5 rep files had
#   itls/ttfts); the violations came from telemetry files a loose glob had pulled
#   in.  Trusting that verdict would have wasted 6.4 GPU-hr.  This is the MIRROR
#   IMAGE of gate #21 -- a SCORER failure reported as a MEASUREMENT failure -- and
#   it happened inside the mechanism built to tell those apart.  When the evidence
#   is ambiguous the classifier must fall to 22, not round up to 21.
#
#   exit 0  = MEASURED + SCORED
#   exit 20 = MEASURED_OK / SCORING_FAILED  -> data on disk is complete (or the
#             only bad rows are files that are not bench output); RESCORE OFFLINE
#   exit 21 = MEASUREMENT_INCOMPLETE        -> accepted bench files really lack the
#             required keys; a GPU re-run is required
#   exit 22 = INDETERMINATE                 -> cannot separate the two; HUMAN
#             JUDGMENT required.  Do NOT spend GPU on this verdict alone.
# ==========================================================================
echo; echo "===== PRE-SCORING MEASUREMENT PREFLIGHT (gate #21 + its mirror image) ====="
python3 "$OUT/g2s_analyze.py" --tag "$TAG" --job "$SLURM_JOB_ID" --outdir "$OUT" \
  --check-contract-only 2>&1 | tee "$OUT/g2s_preflight_${TAG}_${SLURM_JOB_ID}.txt"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "PREFLIGHT_RC=$PREFLIGHT_RC (0=MEASUREMENT_OK, 7=MEASUREMENT_INCOMPLETE[GPU],"\
     "8=SCORER_SELECTION_DEFECT[no GPU], 9=INDETERMINATE[human])"

echo; echo "===== SCORING ====="
python3 "$OUT/g2s_analyze.py" --tag "$TAG" --job "$SLURM_JOB_ID" --outdir "$OUT" \
  2>&1 | tee "$OUT/g2s_scoreout_${TAG}_${SLURM_JOB_ID}.txt"
SCORE_RC=${PIPESTATUS[0]}

# A scorer exit of 0 is NOT on its own a success: the scorer will happily emit a
# report full of NaNs from an empty directory.  Success requires BOTH the scorer to
# finish AND the preflight to certify the inventory.  If they disagree, that is a
# contradiction to surface (22), never something to paper over with 0.
if   [ "$SCORE_RC" = 0 ] && [ "$PREFLIGHT_RC" = 0 ]; then
                                   STATUS=MEASURED_AND_SCORED;              RC=0
elif [ "$SCORE_RC" = 0 ];     then STATUS=SCORED_BUT_INVENTORY_UNCERTIFIED; RC=22
elif [ "$PREFLIGHT_RC" = 0 ]; then STATUS=MEASURED_OK_SCORING_FAILED;       RC=20
elif [ "$PREFLIGHT_RC" = 8 ]; then STATUS=MEASURED_OK_SCORER_SELECTION_DEFECT; RC=20
elif [ "$PREFLIGHT_RC" = 7 ]; then STATUS=MEASUREMENT_INCOMPLETE;           RC=21
else                               STATUS=INDETERMINATE;                    RC=22
fi
python3 - "$OUT" "$TAG" "$SLURM_JOB_ID" "$STATUS" "$RC" "$SCORE_RC" "$PREFLIGHT_RC" <<'PYEOF'
import json, sys
out, tag, job, status, rc, src, prc = sys.argv[1:8]
json.dump(dict(tag=tag, job=job, status=status, exit_code=int(rc),
               scorer_rc=int(src), preflight_rc=int(prc),
               rescore_offline_sufficient=status in (
                   "MEASURED_OK_SCORING_FAILED", "MEASURED_OK_SCORER_SELECTION_DEFECT"),
               gpu_rerun_required=(status == "MEASUREMENT_INCOMPLETE"),
               human_judgment_required=status in (
                   "INDETERMINATE", "SCORED_BUT_INVENTORY_UNCERTIFIED"),
               cost_note=("exit 21 costs ~6.4 GPU-hr, exit 20/22 cost 0 -- never "
                          "round an ambiguous case up to 21")),
          open(f"{out}/g2s_status_{tag}_{job}.json", "w"), indent=1)
PYEOF
echo "CAMPAIGN_STATUS=$STATUS scorer_rc=$SCORE_RC preflight_rc=$PREFLIGHT_RC exit=$RC"
[ "$RC" = 21 ] && echo "  -> GPU RE-RUN REQUIRED (~6.4 GPU-hr). Confirm against the preflight attribution lines before spending it."
[ "$RC" = 22 ] && echo "  -> INDETERMINATE: human judgment required. Do NOT re-run on GPU on this verdict alone."
echo "DONE tag=$TAG job=$SLURM_JOB_ID"
exit "$RC"
