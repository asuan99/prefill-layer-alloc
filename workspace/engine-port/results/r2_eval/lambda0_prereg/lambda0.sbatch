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
# Pre-registration: PREREG_LAMBDA0_REV4_2026-09-13.md
#   (rev1 NO-GO -> rev2 NO-GO -> rev3 NO-GO -> rev4; all kept, none deleted)
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
#   positive throughput, and every reported #running-req >= 48 (newpair F5).
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
#   precedent; that line number was true on 2026-09-12 and is NOT true now (the
#   constant sits at :138 @HEAD), so the stable anchor is cited instead.
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

echo "STAGE0 lambda* rev3 model=$MODEL backend=$BACKEND ctx=$CTX mem=$MEM_FRACTION job=$JOBID"

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
python3 "$PREREG/lambda0_lambda_inf.py" --selftest || exit 2
python3 "$PREREG/lambda0_lambda_inf.py" "$INSTR" --record \
  | tee "$OUT/LAMBDA_INF_DECISION.txt"
# E6: the machine-readable twin travels with the cells so the label JSON embeds it.
python3 -c "
import json,sys
sys.path.insert(0,'$PREREG')
from lambda0_lambda_inf import decide
json.dump(decide('$INSTR'), open('$OUT/LAMBDA_INF_DECISION.json','w'), indent=2, sort_keys=True)
"
LAMBDA0_MODE=$(awk -F= '$1=="LAMBDA0_MODE"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
LAMBDA0_A=$(awk -F= '$1=="LAMBDA0_LAMBDA_INF_A"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
LAMBDA0_B=$(awk -F= '$1=="LAMBDA0_LAMBDA_INF_B"{print $2}' "$OUT/LAMBDA_INF_DECISION.txt")
if [ "$LAMBDA0_MODE" = "ANCHORED" ]; then
  PLAN_ARGS=(--lambda-inf-a "$LAMBDA0_A" --lambda-inf-b "$LAMBDA0_B")
  echo "PLAN=anchored A=$LAMBDA0_A B=$LAMBDA0_B (from $INSTR)"
else
  PLAN_ARGS=(--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback)
  echo "PLAN=fallback (registered literal ladders; see LAMBDA_INF_DECISION.txt)"
fi

python3 "$PREREG/lambda0_plan.py" --selftest || exit 2
python3 "$PREREG/lambda0_label.py" --selftest || exit 2
python3 "$PREREG/lambda0_cells.py" --selftest || exit 2
# ★E2: every registered verdict must have a non-empty domain under the
# registered model.  rev3 shipped with KNEE_BRACKETED[B] and LADDER_TOO_HIGH
# structurally unreachable and nothing in the run would have said so.
python3 "$PREREG/lambda0_reachability.py" --selftest || exit 2
python3 "$PREREG/lambda0_reachability.py" > "$OUT/REACHABILITY.txt" 2>&1 || exit 2
python3 "$PREREG/lambda0_mutation_check.py" > "$OUT/mutation_check.txt" 2>&1 \
  || { echo "ABORT_MUTATION_ESCAPE"; tail -5 "$OUT/mutation_check.txt"; exit 5; }
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
echo "=== registered rules (PREREG_LAMBDA0_REV4 sec 5) ==="
# ★ --expect, not a glob: a cell whose boot failed (or was never reached) must
#   put its shape on UNRESOLVED rather than silently shortening the ladder
#   (rev1 audit S9 / D4, rev2 audit D23).
python3 "$PREREG/lambda0_label.py" --cells-dir "$OUT" \
  --expect "$EXPECT" --repeat "$REPEAT" --out "$OUT/LAMBDA0_LABEL.json" \
  || echo "LABEL_FAILED"

echo "LAMBDA0_DONE_${JOBID}"
