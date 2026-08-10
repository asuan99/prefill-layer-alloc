#!/bin/bash
#SBATCH --job-name=g2ssmoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --comment=pytorch
#SBATCH --time=00:15:00
#SBATCH --output=g2ssmoke_%j.out
#SBATCH --error=g2ssmoke_%j.err
#
# ==============================================================================
# Gate 2-S HARNESS SMOKE -- NOT the pre-registered campaign, NOT a performance
# run. Coordinator brief 2026-08-10, after 877107/877109 completed measurement
# (exit 0:0) but produced ZERO scoreable primaries: g2s_run.sbatch's bench()
# call had no --output-details, so bench_serving.py:1630 wrote only the
# aggregate `result` and dropped `result_details` (itls/ttfts/input_lens/
# output_lens). engine-porter applied 4 fixes to g2s_run.sbatch/g2s_analyze.py/
# g2s_selftest.py (uncommitted, GPU-0-verified only). This script is the ~10 min
# GPU check that those fixes actually produce a scoreable artifact end-to-end
# BEFORE committing to the 6.4 GPU-hr re-run.
#
# PREREG_GATE2S_2026-08-09.md is NOT touched, NOT re-implemented here, and this
# job's numbers must never be cited as a finding -- this is plumbing
# verification only ("does the pipeline hand the scorer what it eats"), not a
# measurement of anything in sec2-sec8.
#
# THREE THINGS THIS JOB CHECKS (coordinator brief, verified independently by
# experiment-runner after the job completes, not just by this script's own
# self-report):
#   1. the produced rep jsonl actually contains non-empty "itls"/"ttfts" whose
#      lengths match the request count (the exact field bench_serving.py:1630
#      was dropping).
#   2. g2s_analyze.py runs on that REAL output with rc=0 and reports all 19
#      sec6.5 blocks present in the block manifest.
#   3. the exit-code wiring g2s_run.sbatch now carries (0=MEASURED_AND_SCORED,
#      20=MEASURED_OK_SCORING_FAILED, 21=MEASUREMENT_INCOMPLETE) actually
#      yields 0 on this successful path.
#
# WHY 5 ARMS (not literally "1 arm" as the brief's example suggested) --
# discretion the brief explicitly delegated ("정확한 파라미터는 네가 정하되"):
# g2s_analyze.py's primary contrasts are C-T' and C-T (build_diffs needs BOTH
# arms of a pair present at the same rate/rep to produce a non-empty diff
# list), kv_pool compares T vs C, probe_pos compares T vs C, probe_carve
# compares A2 vs C. A 1-arm smoke would make every one of those trivially
# empty (still non-crashing per the code's NaN/None-on-insufficient-n
# handling, confirmed by reading paired_t()/build_diffs()/f_series()/load_kv()
# before writing this script) but would not actually exercise the contrast
# machinery that is the entire point of validating "the scorer eats this".
# 5 arms x 1 rep x 1 rate x --num-prompts 20 still fits comfortably inside the
# ~10 min target (measured boot-cycle cost from 877107/877109: ~1 min/arm
# including cudagraph capture; this smoke's bench calls are ~10x shorter than
# a real 120-prompt Phase-1 call).
#
# WHAT'S DELIBERATELY SKIPPED (coordinator brief explicitly allows this):
#   - Phase 0a capscan (30 eq-run alone in the real campaign) -- f_series()
#     degrades F_B_i to (None, evaluated=False) with an empty capscan dict,
#     confirmed by reading the code; does not crash.
#   - Phase 0b T/C pilot -- feeds sigma_D context only, not required for
#     build_diffs()/paired_t() to run.
#   - the dedicated G-1/G-2 correctness gate (g2s_gate.py) section -- instead,
#     record_flags() (the KV-pool-log-parsing step, cheap: one curl + one
#     regex pass) is called once per Phase-1-style boot below so kv_pool gets
#     REAL data too, at near-zero extra cost, rather than reporting
#     UNDETERMINED for a block that was easy to cover for real.
#   - NREP=10 -> 1, rate={2,3} -> {2} only. Rate 3 cells will show up empty/
#     UNDETERMINED in the report -- that is expected and itself confirms the
#     partial-coverage path doesn't crash either.
#
# NAMING (coordinator instruction: never mix with pre-registered data):
#   - this script, its SLURM job name, and its own stdout/stderr are
#     g2ssmoke_* / job-name=g2ssmoke -- squeue/sacct/ls all show it as
#     obviously distinct from the g2s_* campaign jobs (877107/877109, and any
#     future real re-run).
#   - g2s_analyze.py's glob patterns are HARD-CODED to the literal prefix
#     "g2s_" + tag + ... + job-id (not a variable prefix) -- the per-arm rep/
#     telemetry/flags/probe files this smoke produces MUST carry that exact
#     "g2s_<tag>_..._<SLURM_JOB_ID>.*" shape or g2s_analyze.py will not find
#     them, which would silently defeat checks 2-3. Separation from the real
#     campaign is therefore by SLURM_JOB_ID (always fresh/unique, never
#     colliding with 877107/877109 or a future re-run's ids), not by prefix.
#     g2ssmoke_manifest_<jobid>.txt (written at the end) enumerates every
#     g2s_-prefixed file this job wrote, stamped DO-NOT-CITE, so the boundary
#     stays auditable even though the raw filenames share the prefix.
# ==============================================================================
set -uo pipefail

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
cd "$ROOT"

TAG=zamba2-27b   # fixed: the bug being validated (bench()/g2s_analyze.py) is
                 # model-agnostic; Zamba2 is the smaller/faster-booting model.
MODEL="Zyphra/Zamba2-2.7B"; CTX=4096; BACKEND=triton
RATE=2           # single rate, the lower of CELLS["zamba2-27b"]=(2.0,3.0) --
                 # a real pre-registered rate value, not an invented one.

OUT="$ROOT/workspace/engine-port/results/p1_gates/gate2"
PDMUX_CFG="$OUT/pdmux_a100_smoke.yml"
SCORE_ARMS="plainaux pdmux_nosplit pdmux_split34 pdmux_split34_nosticky agnostic"
export G2_GREEDY_MAX_TIME=180
export G2_GREEDY_CONNECT_TIMEOUT=10

echo "==================== G2S HARNESS SMOKE (NOT the prereg campaign) ===================="
echo "purpose: verify --output-details fix produces scoreable itls/ttfts, and that"
echo "g2s_analyze.py + its exit-code wiring succeed end-to-end on REAL output."
echo "tag=$TAG model=$MODEL rate=$RATE arms=[$SCORE_ARMS] nrep=1 job=$SLURM_JOB_ID"
echo "DO NOT CITE any number from this job. DO NOT mix its artifacts with 877107/877109"
echo "or any future prereg re-run -- see g2ssmoke_manifest_${SLURM_JOB_ID}.txt at the end."
echo "======================================================================================"

MANIFEST="$OUT/runtime_source_manifest_ssmoke_${TAG}_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || { echo "SYNC FAILED" >&2; exit 3; }
DIFF="$OUT/manifest_diff_g2ssmoke_${TAG}_${SLURM_JOB_ID}.txt"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$DIFF"
N_ADD=$(grep -c '^>' "$DIFF" || true); N_DEL=$(grep -c '^<' "$DIFF" || true)
echo "MANIFEST added=$N_ADD removed=$N_DEL (expect added=2 removed=0)"
if [ "$N_ADD" != "2" ] || [ "$N_DEL" != "0" ]; then
  echo "MANIFEST_CHECK=FAIL" >&2; cat "$DIFF" >&2; exit 4
fi
echo "MANIFEST_CHECK=PASS"

PORT=$((34000 + SLURM_JOB_ID % 400))
source "$OUT/g2_holb_phaseA_lib.sh"

# ---- verbatim from the fixed (uncommitted) g2s_run.sbatch: arm_flags/arm_env/
# boot/wait_ready/kill_server/record_flags/bench. Copied, not sourced, because
# g2s_run.sbatch defines these inline (no shared library) -- kept byte-identical
# so this smoke tests the SAME functions the real campaign will run, not a
# reimplementation that could silently diverge.
arm_flags () {
  case "$1" in
    plainaux) echo "--chunked-prefill-size -1 --disable-overlap-schedule" ;;
    pdmux_nosplit|pdmux_split34|pdmux_split34_nosticky|agnostic)
      echo "--enable-pdmux --pdmux-config-path $PDMUX_CFG --chunked-prefill-size -1 --disable-overlap-schedule" ;;
  esac
}
arm_env () {
  case "$1" in
    plainaux) ;;
    pdmux_nosplit)          echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=108 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    pdmux_split34)          echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=34 PDMUX_STICKY_PARTITION=1 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    pdmux_split34_nosticky) echo "export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=34 PDMUX_TELEMETRY_PATH=$TELEM" ;;
    agnostic)               echo "export PDMUX_TELEMETRY_PATH=$TELEM" ;;
  esac
}
boot () {
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
record_flags () {
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
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$1" \
    --dataset-name random-ids --random-input-len 2000 --random-output-len 96 \
    --random-range-ratio 1.0 --request-rate "$2" --num-prompts "$3" --seed "$4" \
    --warmup-requests "$6" --output-details --output-file "$5" --tag "$7" \
    >> "$OUT/g2s_bench_${TAG}_${SLURM_JOB_ID}.log" 2>&1
}

echo; echo "===== SMOKE PHASE 1: 5 arms x 1 rep x rate=$RATE, --num-prompts 20 (real bench, real analyzer) ====="
REP=1
ORDER="$(shuf -e $SCORE_ARMS | tr '\n' ' ')"
echo "arm_order=$ORDER"
for ARM in $ORDER; do
  LOG="$OUT/g2s_${TAG}_srv_${ARM}_rep${REP}_${SLURM_JOB_ID}.log"
  TEL="$OUT/g2s_${TAG}_telemetry_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl"
  PID=$(boot "$ARM" "$PORT" "$LOG" "$TEL"); OK=$(wait_ready "$PID" "$PORT")
  if [ "$OK" != 1 ]; then echo "  BOOT FAILED arm=$ARM"; tail -20 "$LOG"; kill_server "$PID" "$PORT"; continue; fi
  record_flags "$ARM" "$PORT" "$LOG" "$OUT/g2s_${TAG}_flags_${ARM}_${SLURM_JOB_ID}.json"
  python3 "$OUT/g2s_probe.py" --port "$PORT" --arm "$ARM" --rep "$REP" \
    --out "$OUT/g2s_${TAG}_probes_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
  OFF="$OUT/g2s_${TAG}_teloffset_${ARM}_rep${REP}_${SLURM_JOB_ID}.json"
  S0=$(wc -l < "$TEL" 2>/dev/null || echo 0)
  bench "$PORT" "$RATE" 20 $((9000 + 100*REP + RATE)) \
    "$OUT/g2s_${TAG}_${ARM}_rep${REP}_${SLURM_JOB_ID}.jsonl" 2 \
    "smoke_rep${REP}_${ARM}_r${RATE}"
  S1=$(wc -l < "$TEL" 2>/dev/null || echo 0)
  printf '{"r%s":[%s,%s]}\n' "$RATE" "$S0" "$S1" > "$OFF"
  kill_server "$PID" "$PORT"
  echo "  arm=$ARM done"
done

echo; echo "===== CHECK 1 (in-job, quick): does the rep jsonl actually contain itls/ttfts? ====="
python3 - "$OUT" "$TAG" "$SLURM_JOB_ID" "$SCORE_ARMS" <<'PYEOF'
import json, sys
out, tag, job, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
ok = True
for arm in arms:
    f = f"{out}/g2s_{tag}_{arm}_rep1_{job}.jsonl"
    try:
        lines = open(f).readlines()
    except FileNotFoundError:
        print(f"  arm={arm:24s} FILE MISSING ({f})"); ok = False; continue
    if not lines:
        print(f"  arm={arm:24s} FILE EMPTY"); ok = False; continue
    d = json.loads(lines[0])
    itls, ttfts = d.get("itls"), d.get("ttfts")
    completed = d.get("completed")
    n_itls, n_ttfts = (len(itls) if itls else 0), (len(ttfts) if ttfts else 0)
    good = bool(itls) and bool(ttfts) and n_ttfts == completed
    print(f"  arm={arm:24s} completed={completed} n_itls={n_itls} n_ttfts={n_ttfts} "
          f"itls_present={bool(itls)} ttfts_present={bool(ttfts)} len_matches_completed={n_ttfts==completed} "
          f"-> {'OK' if good else 'FAIL'}")
    ok = ok and good
print(f"CHECK1_ITLS_TTFTS_PRESENT={'PASS' if ok else 'FAIL'}")
PYEOF

# sec8.9.1-style per-cell telemetry build (1 rate only, same code shape as the
# real campaign's offset-slicing step).
python3 - "$OUT" "$TAG" "$SLURM_JOB_ID" "$RATE" <<'PYEOF'
import glob, json, os, sys
out, tag, job, rate = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
for arm in ("plainaux","pdmux_nosplit","pdmux_split34","pdmux_split34_nosticky","agnostic"):
    rows = []
    for tel in sorted(glob.glob(f"{out}/g2s_{tag}_telemetry_{arm}_rep*_{job}.jsonl")):
        off = tel.replace("_telemetry_", "_teloffset_").replace(".jsonl", ".json")
        if not os.path.exists(off):
            continue
        try: sl = json.load(open(off)).get(f"r{rate}")
        except Exception: sl = None
        if not sl:
            continue
        with open(tel) as fh:
            rows += fh.readlines()[sl[0]:sl[1]]
    dst = f"{out}/g2s_{tag}_telemetry_{arm}_r{rate}_{job}.jsonl"
    open(dst, "w").writelines(rows)
    print(f"  cell telemetry {arm} r{rate}: {len(rows)} rows -> {os.path.basename(dst)}")
PYEOF

echo; echo "===== CHECK 2+3: g2s_analyze.py on REAL smoke output + exit-code wiring ====="
python3 "$OUT/g2s_analyze.py" --tag "$TAG" --job "$SLURM_JOB_ID" --outdir "$OUT" \
  --check-contract-only 2>&1 | tee "$OUT/g2ssmoke_preflight_${TAG}_${SLURM_JOB_ID}.txt"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "PREFLIGHT_RC=$PREFLIGHT_RC"

python3 "$OUT/g2s_analyze.py" --tag "$TAG" --job "$SLURM_JOB_ID" --outdir "$OUT" \
  2>&1 | tee "$OUT/g2ssmoke_scoreout_${TAG}_${SLURM_JOB_ID}.txt"
SCORE_RC=${PIPESTATUS[0]}

if   [ "$SCORE_RC" = 0 ];     then STATUS=MEASURED_AND_SCORED;   RC=0
elif [ "$PREFLIGHT_RC" = 0 ]; then STATUS=MEASURED_OK_SCORING_FAILED; RC=20
else                               STATUS=MEASUREMENT_INCOMPLETE; RC=21
fi
echo "SMOKE_STATUS=$STATUS scorer_rc=$SCORE_RC preflight_rc=$PREFLIGHT_RC exit=$RC"

N_BLOCKS_IN_REPORT=$(python3 -c "
import json
try:
    d = json.load(open('$OUT/g2s_report_${TAG}_${SLURM_JOB_ID}.json'))
    print(len(d.get('blocks', [])))
except Exception as e:
    print(f'ERROR:{e}')
")
echo "N_BLOCKS_IN_REPORT=$N_BLOCKS_IN_REPORT (expect 19)"

# ---- final smoke verdict (in-job self-report; experiment-runner re-verifies
# independently against the raw artifacts after the job ends, per brief).
{
  echo "Gate 2-S HARNESS SMOKE verdict -- job=$SLURM_JOB_ID tag=$TAG (NOT the prereg campaign)"
  echo "CHECK1 (itls/ttfts present in rep jsonl, len matches completed): see stdout CHECK1_ITLS_TTFTS_PRESENT above"
  echo "CHECK2 (g2s_analyze.py rc=0 on real output, 19/19 blocks): scorer_rc=$SCORE_RC N_BLOCKS_IN_REPORT=$N_BLOCKS_IN_REPORT"
  echo "CHECK3 (exit-code wiring = 0 on success path): SMOKE_STATUS=$STATUS exit=$RC"
} > "$OUT/g2ssmoke_verdict_${SLURM_JOB_ID}.txt"
cat "$OUT/g2ssmoke_verdict_${SLURM_JOB_ID}.txt"

# ---- manifest of every g2s_-prefixed (real-campaign-shaped) artifact this
# smoke job wrote, so the g2s_/g2ssmoke_ naming boundary stays auditable.
{
  echo "DO NOT CITE. DO NOT mix with 877107/877109 or any prereg re-run."
  echo "All files below share the required g2s_ prefix (g2s_analyze.py's glob"
  echo "patterns are hard-coded to it) but carry SLURM_JOB_ID=$SLURM_JOB_ID,"
  echo "which is unique and never collides with a real campaign job id."
  ls -la "$OUT"/g2s_*_${SLURM_JOB_ID}.* "$OUT"/g2s_*_${SLURM_JOB_ID} 2>/dev/null
} > "$OUT/g2ssmoke_manifest_${SLURM_JOB_ID}.txt"

echo "DONE_G2SSMOKE job=$SLURM_JOB_ID status=$STATUS"
exit "$RC"
