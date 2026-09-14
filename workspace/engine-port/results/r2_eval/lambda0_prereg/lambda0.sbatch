#!/bin/bash
#SBATCH --job-name=lambda0
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/%x_%j.err
#
# CAMPAIGN STAGE 0 -- lambda*(shape) on the reference arm B1.
# Pre-registration: $PREREG_MD (default PREREG_LAMBDA0_REV5_2026-09-14.md)
#   (rev1 NO-GO -> rev2 NO-GO -> rev3 NO-GO -> rev4 NO-GO -> rev5; all kept,
#    none deleted).  ★rev5 exists because the rev4 audit found condition (b) of
#    the anchor predicate was an effective IDENTITY on live artifacts (N1) and
#    that repairing it FLIPS the branch ANCHORED -> FALLBACK (N2).  The repair is
#    in `lambda0_lambda_inf.py` (F1); the flip is registered in rev5 (F2).
#    ★F6: the digest of every file that can change the verdict, and of the
#    pre-registration itself, is written to $OUT/REGISTRATION_SHA256.txt BEFORE
#    the decision is taken.
#
# ONE BOOT PER CELL, ONE telemetry file per boot -- deliberately NOT a rate
# ladder inside a single boot, because slicing one telemetry file by time window
# reintroduces exactly the interval-attribution machinery this track keeps dying
# on (`CONSENSUS sec 3 item 120`).
#   ★ The rev1 audit corrected rev1's citation here: `c_capacity.sbatch:30-35` is
#     the COMMENT that states this rule; the MECHANISM inherited is the
#     `for SPEC in "${CELLS[@]}"` loop at `c_capacity.sbatch:111` plus the
#     per-cell launch / kill / sleep 5 / pkill / sleep 10 at `:123-178`.  That
#     mechanism is what is reproduced below, not the comment.
#   ★ Stronger still, and independent of this loop: the judged quantity is
#     CLIENT SIDE (`achieved = completed/duration`, `span` from the replayed
#     RNG) and never touches telemetry.
#
# NO policy verdict.  NO arm comparison -- every ladder is relative to the
# reference arm's own predicted capacity, and the same quantity already varies
# 5x across splits (D16 0.933 / D44 0.675 / D92 0.187 req/s, job 905835).
#
# usage:
#   sbatch lambda0.sbatch          # anchored-vs-fallback is decided by FILES
#
# ★D18 (rev2 audit N2-2): there is NO operator switch.  rev2 branched on
#   `PDMUX_LAMBDA0_FALLBACK`, and that discretion FLIPS a label (at
#   lambda*(B)=1.10 the anchored window [0.5157,1.835] brackets and the fallback
#   window [0.4302,1.026] does not).  `lambda0_lambda_inf.py` decides from the
#   newpair I3 artifacts instead: both cells present, run at rate inf with
#   positive throughput, and -- ★rev5/F1 -- EACH OF THE TWO CELLS reaching
#   #running-req >= 48 inside ITS OWN interval of `srv_warmup.log`, recomputed
#   from `I_log_offsets.txt` under all three registered conventions (newpair F5).
#   rev4 read one GLOBAL-max line out of `I3_max_running_req.txt` instead, which
#   made the disqualification path an identity; that file is now provenance and a
#   cross-check only -- it can veto the anchor, never grant it.
#   `PDMUX_LAMBDA0_INSTR` says only WHERE to look, never WHAT to conclude.
#
# ★ WHY NOT ${BASH_SOURCE[0]}: sbatch COPIES the submitted script into the node
#   spool, so in a running batch job ${BASH_SOURCE[0]} is
#   /var/spool/slurmd/job<ID>/slurm_script and every path derived from it points
#   at nothing (the defect fixed in commit 09a8075 for r2_eval.sbatch).  The
#   project root is pinned as an absolute constant, like the committed fix does
#   (scripts/r2_eval/r2_eval.sbatch:38 @HEAD bddff6a,
#   `project_root="${PDMUX_PROJECT_ROOT:-/scratch/ehmoon/whlee/prefill-layer-alloc}"`).
#   NB commit 09a8075's message cites r2_correctness.sbatch:99 for the same
#   precedent; that line number was true on 2026-09-12 and is NOT true now.  ★And
#   this line's own replacement ("the constant sits at :138 @HEAD") HAS ALREADY
#   GONE STALE -- :138 was true at HEAD bddff6a and the same constant
#   (`PROJECT=/scratch/ehmoon/whlee/prefill-layer-alloc`) sits at :166 at HEAD
#   8507cee, because another track added ~28 lines above it.  That is lam5A-11 in
#   this very file, so the citation is given as CODE TEXT + a commit pin instead
#   of a bare line number, and only the stable anchor above is relied on.
set -uo pipefail

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
PREREG=$ROOT/workspace/engine-port/results/r2_eval/lambda0_prereg

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd "$ROOT" || exit 2

JOBID="${SLURM_JOB_ID:-local$$}"
OUT="$PREREG/lam0_${JOBID}"; mkdir -p "$OUT"

export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
# ctx above the model's own limit silently corrupts results -- keep the engine's
# guard armed (same reasoning as commit 09a8075).
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
unset PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD PDMUX_STICKY_PARTITION \
      PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL \
      PDMUX_DUAL_WORKER_TRACE_EVERY PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER

# ------------------------------------------------------------ registered arm
# B1 = legacy loop + PD-mux fixed D44.  Registered control factors (prereg rev3
# sec 2); the differences from probe C are registered in sec 2.4.
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="$ROOT/workspace/engine-port/results/longctx_conflict/probes/pdmux_homog5.yml"
CTX=16384
MEM_FRACTION=0.82
MAX_RUNNING=48
SERVER_SEED=1                     # D11: the campaign passes --random-seed; probe C did not
FIXED_DSM=44
SGPT_RAW="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# ------------------------------------------------- F6: registration integrity
# The pre-registration in force, and the digest of every file that can change
# the verdict, recorded BEFORE the decision.  rev4 ran with all 14 executables
# uncommitted and no digests, so a post-hoc edit would not have been detectable.
PREREG_MD="${PDMUX_LAMBDA0_PREREG:-$PREREG/PREREG_LAMBDA0_REV5_2026-09-14.md}"
# ★the DECISION PATH: 7 files.  A change to any of them can change the label.
DECISION_PATH=(lambda0_lambda_inf.py lambda0_plan.py lambda0_label.py
               lambda0_analyze.py lambda0_cells.py lambda0_reachability.py
               lambda0_mutation_check.py)
echo "STAGE0 lambda* $(basename "$PREREG_MD") model=$MODEL backend=$BACKEND ctx=$CTX mem=$MEM_FRACTION job=$JOBID"
for f in "${DECISION_PATH[@]}" lambda0_cellprint.py lambda0.sbatch; do
  [ -f "$PREREG/$f" ] || { echo "ABORT_F6 decision-path file missing: $f" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 3; }
done
if [ ! -f "$PREREG_MD" ]; then
  echo "ABORT_F6 pre-registration document missing: $PREREG_MD (set PDMUX_LAMBDA0_PREREG if it is named differently)" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
  exit 3
fi
{
  echo "# pre-registration in force: $(basename "$PREREG_MD")"
  sha256sum "$PREREG_MD"
  echo "# decision path (${#DECISION_PATH[@]} files) + renderer + runner"
  for f in "${DECISION_PATH[@]}" lambda0_cellprint.py lambda0.sbatch; do
    sha256sum "$PREREG/$f"
  done
  echo "# every pre-registration revision kept beside it"
  sha256sum "$PREREG"/PREREG_LAMBDA0*.md
  echo "# tree: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo NO_GIT)" \
       "uncommitted=$(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c .)"
} | tee "$OUT/REGISTRATION_SHA256.txt"
N_SHA=$(grep -c "^[0-9a-f]\{64\} " "$OUT/REGISTRATION_SHA256.txt")
if [ "${N_SHA:-0}" -lt 10 ]; then
  echo "ABORT_F6 only $N_SHA digests recorded, registered minimum 10" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
  exit 3
fi

# ------------------------------------------------- D12: provenance BEFORE GPU
echo "=== env sync (runtime source manifest) ==="
"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$OUT/runtime_source_manifest.sha256" || exit 2
N_MANIFEST=$(grep -c . "$OUT/runtime_source_manifest.sha256")
echo "manifest_entries=$N_MANIFEST"
if ! grep -q "models/nemotron_h.py" "$OUT/runtime_source_manifest.sha256"; then
  echo "ABORT_D12 manifest does not cover the served model implementation" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
  exit 3
fi
if [ "${N_MANIFEST:-0}" -lt 24 ]; then
  echo "ABORT_D12 manifest has $N_MANIFEST entries, registered minimum 24" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
  exit 3
fi

# ---------------------------------------------- the plan (deterministic, D3)
# D18: the anchored/fallback branch is a predicate over files, not a switch.
# Default = the MEASURED artifacts (job 907959, 2026-09-13).  E6: the
# resolved absolute path and a sha256 per input are recorded below, because
# the rev3 audit showed the directory choice can flip a shape-A label.
INSTR="${PDMUX_LAMBDA0_INSTR:-$ROOT/workspace/engine-port/results/r2_correctness/job_907959/instrument}"
# =========================================================== lam5A-2 (rev5 addendum sec B-1)
# BIND THE SELFTEST'S INPUT TO THE BRANCH'S INPUT.  The 5th rules-layer audit
# showed BY EXECUTION (W4') that `PDMUX_LAMBDA0_INSTR` -- what `decide()` reads --
# and `LAMBDA0_I3_JOB_DIR` -- what the predicate's LIVE-artifact selftest leg
# recounts -- were INDEPENDENT.  So a substituted instrument dir with hand-written
# offsets answered ANCHORED (lambda_inf 3.0939 / 0.6956, zero disqualifications)
# WHILE THE SHIPPED SELFTEST STILL RETURNED rc 0: the gate certified job 907959
# and the branch was taken on somebody else's bytes.  One export closes it -- the
# selftest now recounts THE SAME JOB the branch is decided from, so a substituted
# dir fails `set(rc["cells"]) == set(LIVE_RECOUNT)` and `|| exit 2` below stops
# the job instead of running an unregistered plan.
#   NB the mutation harness sets this variable EXPLICITLY for its temp-dir copies
#   (`lambda0_mutation_check.py`, `env = dict(os.environ, ..., LAMBDA0_I3_JOB_DIR=
#   str(I3_JOB))`), so it is unaffected by this export.
export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"
python3 "$PREREG/lambda0_lambda_inf.py" --selftest || exit 2
# ★F1: the per-cell #running-req recount condition (b) is decided on, in its own
# file, so the run record carries the numbers even if the warm-up log is later
# rotated away.  (rev4 passed a `--record` flag the tool never had: argv was
# length 3, so the tool printed its docstring and returned 2 -- and `tee`
# swallowed that status, leaving MODE empty and the job on the fallback branch
# for the WRONG reason.)
python3 "$PREREG/lambda0_lambda_inf.py" --recount "$INSTR" \
  | tee "$OUT/RUNNING_REQ_RECOUNT.txt" || exit 2
python3 "$PREREG/lambda0_lambda_inf.py" "$INSTR" \
  | tee "$OUT/LAMBDA_INF_DECISION.txt" || exit 2
# E6: the machine-readable twin travels with the cells so the label JSON embeds it.
python3 -c "
import json,sys
sys.path.insert(0,'$PREREG')
from lambda0_lambda_inf import decide
json.dump(decide('$INSTR'), open('$OUT/LAMBDA_INF_DECISION.json','w'), indent=2, sort_keys=True)
"
LAMBDA0_MODE=$(awk -F= '$1=="LAMBDA0_MODE"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
# ★fail closed on a decision that did not happen.  An EMPTY mode used to fall
# through to the `else` of the branch below, i.e. the job would have taken the
# fallback branch because the predicate CRASHED -- a measurement failure wearing
# a verdict's clothes (lesson 21 / gate #21).  ★That `else` no longer exists
# (lam5A-9's sibling lam5A-4 made the FALLBACK plan unconditional and turned
# ANCHORED into an abort), but this `case` is still the guard that separates "the
# predicate said FALLBACK" from "the predicate never answered".
case "$LAMBDA0_MODE" in
  ANCHORED|FALLBACK) ;;
  *) echo "ABORT_D18 predicate produced no mode (got '$LAMBDA0_MODE')" \
       | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; exit 2;;
esac
LAMBDA0_A=$(awk -F= '$1=="LAMBDA0_LAMBDA_INF_A"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
LAMBDA0_B=$(awk -F= '$1=="LAMBDA0_LAMBDA_INF_B"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
# =========================================================== lam5A-4 (rev5 addendum sec B-3)
# rev5 REGISTERS FALLBACK.  On the registered inputs the ANCHORED branch is
# UNREACHABLE: `lambda0_lambda_inf.py --selftest` asserts
# `decide(live)["mode"] == "FALLBACK"` and the `|| exit 2` above makes that
# assertion fatal.  An unreachable branch that would silently build a DIFFERENT
# ladder is the same defect class as an operator switch (D18), so it is NOT
# deleted -- the predicate and both of its outputs are preserved and the
# unregistered outcome becomes a fail-closed abort.
#   lam5C-1: at this cap the anchor is STRUCTURALLY unreachable for shape B
#   (`max_prefill_tokens=16384` caps a prefill-dominated decode batch at ~2), so
#   reviving it means redesigning the F5 threshold, which is a SEPARATE
#   pre-registration and is not authorised here.
if [ "$LAMBDA0_MODE" = "ANCHORED" ]; then
  echo "ABORT_D18 unexpected ANCHORED (rev5 registers FALLBACK) A=$LAMBDA0_A B=$LAMBDA0_B (from $INSTR)" \
    | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
  exit 2
fi
PLAN_ARGS=(--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback)
echo "PLAN=fallback (registered literal ladders; see LAMBDA_INF_DECISION.txt)"

python3 "$PREREG/lambda0_plan.py" --selftest || exit 2
python3 "$PREREG/lambda0_label.py" --selftest || exit 2
python3 "$PREREG/lambda0_cells.py" --selftest || exit 2
# ★E2: every registered verdict must have a non-empty domain under the
# registered model.  rev3 shipped with KNEE_BRACKETED[B] and LADDER_TOO_HIGH
# structurally unreachable and nothing in the run would have said so.
python3 "$PREREG/lambda0_reachability.py" --selftest || exit 2
python3 "$PREREG/lambda0_reachability.py" > "$OUT/REACHABILITY.txt" 2>&1 || exit 2
# =========================================================== lam5A-9 (rev5 addendum sec B-4)
# DO NOT LABEL A MEASUREMENT FAILURE AS A VERDICT (lesson 21 / gate #21).  The
# harness has two distinct non-zero exits and the old one-liner printed
# `ABORT_MUTATION_ESCAPE` for BOTH:
#   rc 1  >= 1 registered mutation PASSED its routed selftest      = AN ESCAPE
#   rc 2  the harness could not run -- archived probe-C job or I3 job missing,
#         or the unmutated CONTROL failed on the temp-dir copy     = NO MEASUREMENT
# Both still abort (fail closed); only the NAME and the exit code differ, so the
# run record cannot be read as "a mutation escaped" when nothing was scored.
python3 "$PREREG/lambda0_mutation_check.py" > "$OUT/mutation_check.txt" 2>&1
MUT_RC=$?
case "$MUT_RC" in
  0) ;;
  1) echo "ABORT_MUTATION_ESCAPE (rc=1: a registered mutation survived its selftest)" \
       | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
     tail -5 "$OUT/mutation_check.txt"; exit 5;;
  2) echo "ABORT_MUTATION_HARNESS_CANNOT_RUN (rc=2: MEASUREMENT FAILURE, not an escape -- the harness never scored the mutations)" \
       | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
     tail -5 "$OUT/mutation_check.txt"; exit 6;;
  *) echo "ABORT_MUTATION_UNEXPECTED_RC=$MUT_RC (unregistered exit status; treated as a MEASUREMENT FAILURE, not an escape)" \
       | tee -a "$OUT/PREFLIGHT_FAILURES.txt"
     tail -5 "$OUT/mutation_check.txt"; exit 6;;
esac
python3 "$PREREG/lambda0_plan.py" "${PLAN_ARGS[@]}" | tee "$OUT/PLAN.txt"
python3 "$PREREG/lambda0_plan.py" "${PLAN_ARGS[@]}" --json > "$OUT/plan.json" || exit 2

SEED1=$(python3 -c "import json;print(json.load(open('$OUT/plan.json'))['seed1'])")
SEED2=$(python3 -c "import json;print(json.load(open('$OUT/plan.json'))['seed2'])")
# cell spec: "name shape in out rate numprompts seed kappa drain_pred lowside T"
# (kappa/drain_pred are the placeholder `-` on rungs with no ceiling)
mapfile -t CELLS < <(python3 "$PREREG/lambda0_cells.py" "$OUT/plan.json")
echo "n_cells=${#CELLS[@]} seed1=$SEED1 seed2=$SEED2 mode=$LAMBDA0_MODE"

# ★D23: EXPECT/REPEAT are built from plan.json BEFORE the loop, so a cell that
# is never attempted (FAIL_STREAK cut-off, wall-clock exhaustion) still reaches
# the rule as a MISSING EXPECTED CELL and its shape lands on UNRESOLVED.  rev2
# appended the names inside the loop, so a truncated run made a whole shape
# vanish from the label JSON -- outside the registered output space.
EXPECT=""; REPEAT=""
for SPEC in "${CELLS[@]}"; do
  read -r NAME _ <<< "$SPEC"
  case "$NAME" in
    *_s2) REPEAT="${REPEAT:+$REPEAT,}$NAME";;
    *)    EXPECT="${EXPECT:+$EXPECT,}$NAME";;
  esac
done
echo "expect=$EXPECT"
echo "repeat=$REPEAT"

wait_health () {
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

FAIL_STREAK=0
for SPEC in "${CELLS[@]}"; do
  read -r NAME SHAPE INLEN OUTLEN RATE NP CSEED KAPPA LHAT LOWSIDE TMEAS <<< "$SPEC"
  # `-` is the renderer's placeholder for "no ceiling applies" (a saturated
  # rung).  It exists because `read` collapses whitespace: an empty column would
  # not occupy a field and every column to its right would shift left.
  [ "$KAPPA" = "-" ] && KAPPA=""
  [ "$LHAT" = "-" ] && LHAT=""

  # Port collision avoidance: a stale server from a previous cell (or another
  # job on the same node) must never be talked to.  Pick a free port and verify
  # nothing answers on it before launching.
  PORT=0
  for try in $(seq 1 50); do
    CAND=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$CAND/health" 2>/dev/null; then
      if ! ss -ltn 2>/dev/null | grep -q ":$CAND "; then PORT=$CAND; break; fi
    fi
  done
  [ "$PORT" = 0 ] && { echo "ABORT_NO_FREE_PORT"; exit 4; }

  SRVLOG="$OUT/srv_${NAME}.log"
  export PDMUX_TELEMETRY_PATH="$OUT/tel_${NAME}.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="lam0_${NAME}_${JOBID}"
  export PDMUX_WORKLOAD_ID="lam0_${NAME}"
  export PDMUX_R2_POLICY=fixed
  export PDMUX_R2_FIXED_DSM=$FIXED_DSM

  echo ""
  echo "########## CELL $NAME shape=$SHAPE in=$INLEN out=$OUTLEN rate=$RATE np=$NP seed=$CSEED T=${TMEAS}s kappa=${KAPPA:-SAT} lowside=$LOWSIDE port=$PORT ##########"
  # cudagraph ON (the operating point).  --disable-piecewise-cuda-graph is NOT
  # passed: D11 registers that probe C disabled it and the campaign's
  # cudagraph-ON path does not (engine_bench_runner.sh:99 adds it only on the
  # cudagraph-OFF branch), and this stage matches the CAMPAIGN.
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --enable-pdmux --pdmux-config-path "$CFG" \
    --disable-overlap-schedule --chunked-prefill-size -1 \
    --disable-radix-cache --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --context-length "$CTX" --random-seed "$SERVER_SEED" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!

  if wait_health "$PORT" "$PID"; then
    echo "boot_ok=1 cell=$NAME"
    FAIL_STREAK=0
    # sec 2.4 verification item: the binding resource must still be 48.
    grep -E "max_mamba_cache_size|max_total_num_tokens|available_gpu_mem" "$SRVLOG" \
      | tail -3 | sed 's/^/  banner /'

    # warmup: 8 sequential requests, discarded (probe C :139-145).  Costs ~54 s
    # at (256,512) vs ~17 s at (8192,96) -- registered in the D15 budget.
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0 \
      --num-prompts 8 --max-concurrency 1 --seed 7 \
      > "$OUT/warmup_${NAME}.log" 2>&1
    echo "warmup_rc=$?"

    # D10: --dataset-path + --tokenize-prompt make the achieved INPUT length
    # exact (probe C: Total input tokens = 130 x 8192, error 0); --output-details
    # --output-file are required for any later re-scoring.
    # ★D20: --random-range-ratio 1.0 is ALSO a necessary condition for the
    # offline arrival replay (benchmark/datasets/common.py:56-64 consumes the
    # numpy stream unless the range is degenerate).  The analyzer reads the
    # ratio back out of the record and refuses the cell if it is not 1.0.
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$INLEN" --random-output-len "$OUTLEN" --random-range-ratio 1.0 \
      --num-prompts "$NP" --request-rate "$RATE" --seed "$CSEED" \
      --output-details --output-file "$OUT/bench_${NAME}.jsonl" \
      > "$OUT/bench_${NAME}.log" 2>&1
    echo "bench_rc=$? cell=$NAME"
    grep -E "Successful requests|Request throughput|Total input tokens|Total generated tokens|Median TTFT|P99 TTFT|Median ITL|Concurrency:" \
      "$OUT/bench_${NAME}.log"

    python3 "$PREREG/lambda0_analyze.py" "$OUT/bench_${NAME}.jsonl" \
      --shape "$SHAPE" --label "$NAME" --num-prompts "$NP" --seed "$CSEED" \
      --low-side-candidate "$LOWSIDE" --t-measure-s "$TMEAS" \
      ${KAPPA:+--kappa-pred "$KAPPA"} ${LHAT:+--drain-pred-s "$LHAT"} \
      --out "$OUT/cell_${NAME}.json" > "$OUT/analyze_${NAME}.log" 2>&1 \
      || echo "ANALYZE_FAILED cell=$NAME"
    python3 "$PREREG/lambda0_cellprint.py" "$OUT/cell_${NAME}.json" 2>/dev/null || true
    du -h "$PDMUX_TELEMETRY_PATH" | sed 's/^/  telemetry /'
  else
    echo "BOOT_FAILED cell=$NAME" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$OUT/BOOT_FAILURES.txt"
    FAIL_STREAK=$((FAIL_STREAK + 1))
  fi
  kill "$PID" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 10

  if [ "$FAIL_STREAK" -ge 2 ]; then
    echo "ABORT_2_CONSECUTIVE_BOOT_FAILURES" | tee -a "$OUT/BOOT_FAILURES.txt"
    break
  fi
done

echo ""
# lam5A-6 (rev5 addendum sec A4): rev5 sec 5 SUPERSEDES rev4 sec 5 (and rev4
# sec 3/4/6/7).  Still live from rev4: sec 9 citation bans, sec 10 mandatory
# companions, sec 11 citation table.
echo "=== registered rules (PREREG_LAMBDA0_REV5 sec 5 + ADDENDUM sec A6; rev4 sec 5 SUPERSEDED, rev4 sec 9/10/11 still live) ==="
# ★ --expect, not a glob: a cell whose boot failed (or was never reached) must
#   put its shape on UNRESOLVED rather than silently shortening the ladder
#   (rev1 audit S9 / D4, rev2 audit D23).
python3 "$PREREG/lambda0_label.py" --cells-dir "$OUT" \
  --expect "$EXPECT" --repeat "$REPEAT" --out "$OUT/LAMBDA0_LABEL.json" \
  || echo "LABEL_FAILED"

echo "LAMBDA0_DONE_${JOBID}"
