#!/bin/bash
#SBATCH --job-name=g1probe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.err
#
# G1 PROBE -- is CP-0's registered capacity axis IDENTIFIABLE on the ShareGPT
# workload CP-0 actually runs?  PREREG_G1_PROBE_2026-08-28.md sec 2, decision
# rule cp0_g1_rule.py (RULE_REV=1).  Buys THREE numbers per (arm, rate) cell:
# achieved throughput, its seed variance, and whether the registered knee band
# is wider than that variance.  Ranks NO arm, computes NO goodput, licenses NO
# claim about chunked prefill (prereg sec 5, forbidden sentences).
#
# ** NOT SUBMITTED BY THE ENGINE-PORTER. ** Authorized by
# OVERRIDE_P1_SUBMIT_2026-08-28.md sec "범위 확장 (2026-09-01)" -- the G1
# probe's exclusion from the original override is explicitly lifted there
# because the reason for the exclusion (needs its own registration) no longer
# holds now that PREREG_G1_PROBE_2026-08-28.md exists.  No reachability
# certificate is issued for G1 by design (prereg sec 3: G1 has no
# evidence-based restriction, so an empty `restrict` would trip the tool's
# RESTRICTIONS_INERT check vacuously -- see RETRACTION_reqinactive_2026-08-28.md).
#
# Design (prereg sec 2, verbatim):
#   arm   fused_default (no cps flag -> 8192 default) | cp512 (--chunked-prefill-size 512)
#   rate  {2, 3, 4, 6, 8}      req/s, nominal offered
#   seed  {1, 7, 17}           ARRIVAL REALIZATIONS, not reps of one realization
#   boot  2 arm x 3 seed = 6 boots, each boot serves the 5 rates SEQUENTIALLY.
#         1 job, same node.
#   flags: same as CP-0 sec 6, but with the STAGED direction REVERSED from P1:
#     --disable-piecewise-cuda-graph on EVERY arm (G1 does not observe the
#     chunking mechanism, so cps must be the only lever) -- decode cudagraph
#     stays ON. --disable-radix-cache --mem-fraction-static 0.82
#     --max-running-requests 48 --context-length 4096.
#   workload: ShareGPT, --num-prompts 200 FIXED at every rate (not scaled by
#     rate), --sharegpt-context-len 4000 -- the same served population as
#     CP-0 (served_population.json).
#   instrument: the chunk probe stays OFF (PDMUX_CHUNK_PROBE_PATH unset) --
#     G1 measures capacity, not chunking, and the probe's observer effect is
#     unmeasured (this is exactly the P1-f question, not this job's).
#
# ** STAGED SUBMISSION (methodology gate #26) **: the prereg's sec 4 budget
# (~0.55 GPU-hr) is an ESTIMATE with no measured basis for the per-rate
# duration.  Run this script FIRST as a 1-boot smoke
# (G1_ARMS="cp512" G1_SEEDS="1", --time short) to get a real number, THEN set
# --time for the 6-boot run from that.  A smoke PASS carries no information
# about the axis question (prereg sec 0) -- it only tells you the harness
# runs end to end and how long it took.
#
# usage:
#   sbatch --time=00:20:00 --export=ALL,G1_ARMS="cp512",G1_SEEDS="1" g1_probe.sbatch   # smoke
#   sbatch --time=01:00:00 g1_probe.sbatch                                              # full (default grid)

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${MODEL:-Zyphra/Zamba2-2.7B}"
CTX="${CTX:-4096}"
BACKEND="${BACKEND:-triton}"
ARMS="${G1_ARMS:-fused_default cp512}"
SEEDS="${G1_SEEDS:-1 7 17}"
RATES="${G1_RATES:-2 3 4 6 8}"
NP="${G1_NP:-200}"

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
OUT="$ROOT/g1_probe_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
SGPT="$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json"

# G1's instrument is OFF, unconditionally -- this job measures capacity, not chunking.
unset PDMUX_CHUNK_PROBE_PATH PDMUX_CHUNK_PROBE_ARM PDMUX_CHUNK_PROBE_RUN_ID PDMUX_CHUNK_PROBE_WORKLOAD_ID

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || exit 2
echo "arms=[$ARMS] seeds=[$SEEDS] rates=[$RATES] np=$NP model=$MODEL backend=$BACKEND"

PORT_BASE=$((35000 + SLURM_JOB_ID % 400))

cps_flag_for_arm () {
  case "$1" in
    fused_default) echo "" ;;
    cp512)         echo "--chunked-prefill-size 512" ;;
    *) echo "UNKNOWN_ARM $1" >&2; exit 3 ;;
  esac
}

descendants () {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    descendants "$child"
    echo "$child"
  done
}

kill_server () {  # $1=pid $2=port -- SIGTERM-first, SIGKILL only as a timeout backstop
  local PID="$1" PORT="$2" p i
  local GRACE_S="${KILL_GRACE_S:-20}" SETTLE_S="${KILL_SETTLE_S:-3}"
  for p in $(descendants "$PID"); do kill -TERM "$p" 2>/dev/null; done
  kill -TERM "$PID" 2>/dev/null
  for i in $(seq 1 "$GRACE_S"); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "kill_server: pid $PID still alive after ${GRACE_S}s grace -> SIGKILL"
    for p in $(descendants "$PID"); do kill -9 "$p" 2>/dev/null; done
    kill -9 "$PID" 2>/dev/null
  fi
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep "$SETTLE_S"
}

wait_ready () {
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

banner_evidence () {  # $1=srvlog $2=tag -- confirms the STAGED direction (opposite of P1)
  echo "--- piecewise/cudagraph banner ($2) ---"
  grep -iE "disable_piecewise_cuda_graph|piecewise_cuda_graph_max_tokens|chunked_prefill_size" "$1" \
    | grep -viE "Future|pynvml" | head -5
}

SUMMARY="$OUT/g1_summary.jsonl"; : > "$SUMMARY"

for ARM in $ARMS; do
  CPS_FLAG=$(cps_flag_for_arm "$ARM")
  for SEED in $SEEDS; do
    PORT=$((PORT_BASE + RANDOM % 300))
    SRVLOG="$OUT/srv_${ARM}_s${SEED}.log"
    BOOT_T0=$(date +%s)
    echo "############ boot arm=$ARM seed=$SEED cps_flag=[$CPS_FLAG] port=$PORT ############"
    python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
      --dtype bfloat16 --attention-backend "$BACKEND" $CPS_FLAG \
      --disable-piecewise-cuda-graph \
      --disable-radix-cache --mem-fraction-static 0.82 \
      --max-running-requests 48 --context-length "$CTX" \
      --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
    PID=$!
    OK=$(wait_ready "$PID" "$PORT")
    BOOT_T1=$(date +%s)
    echo "boot arm=$ARM seed=$SEED ok=$OK boot_s=$((BOOT_T1-BOOT_T0))"
    banner_evidence "$SRVLOG" "${ARM}_s${SEED}"
    if [ "$OK" != 1 ]; then
      echo "BOOT_FAILED arm=$ARM seed=$SEED" | tee -a "$OUT/BOOT_FAILURES.txt"
      tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
      echo "{\"arm\":\"$ARM\",\"seed\":$SEED,\"boot_ok\":false}" >> "$SUMMARY"
      kill_server "$PID" "$PORT"
      continue
    fi

    RATE_T0=$(date +%s)
    RATE_ROWS=()
    for RATE in $RATES; do
      JOUT="$OUT/bench_${ARM}_s${SEED}_r${RATE}.jsonl"; : > "$JOUT"
      T0=$(date +%s.%N)
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --dataset-name sharegpt \
        --dataset-path "$SGPT" --sharegpt-context-len 4000 \
        --num-prompts "$NP" --request-rate "$RATE" --seed "$SEED" \
        --output-details --output-file "$JOUT" \
        --tag "g1_${ARM}_s${SEED}_r${RATE}" \
        > "$OUT/benchlog_${ARM}_s${SEED}_r${RATE}.log" 2>&1
      BRC=$?
      T1=$(date +%s.%N)
      WALL=$(python3 -c "print(f'{$T1-$T0:.1f}')")
      echo "  rate=$RATE bench_rc=$BRC wall_s=$WALL"
      if [ "$BRC" != 0 ]; then
        echo "PROBE_FAILED arm=$ARM seed=$SEED rate=$RATE rc=$BRC" | tee -a "$OUT/BOOT_FAILURES.txt"
        tail -20 "$OUT/benchlog_${ARM}_s${SEED}_r${RATE}.log"
      fi
    done
    RATE_T1=$(date +%s)
    echo "boot arm=$ARM seed=$SEED all_rates_wall_s=$((RATE_T1-RATE_T0))"

    kill_server "$PID" "$PORT"

    # per-boot roll-up: completed/duration (achieved) vs offered=rate, attainment=achieved/offered
    python3 - "$OUT" "$ARM" "$SEED" "$RATES" "$NP" "$SUMMARY" "$SRVLOG" <<'PY'
import json, sys, re

out_dir, arm, seed, rates_s, np_s, summary_path, srvlog = sys.argv[1:8]
seed = int(seed); np_ = int(np_s); rates = [int(r) for r in rates_s.split()]

banner = {"disable_piecewise_cuda_graph": None, "piecewise_cuda_graph_max_tokens": None,
          "chunked_prefill_size": None, "disable_piecewise_reason": None}
try:
    log = open(srvlog, errors="replace").read()
    # server_args= prints the dataclass repr (key=value, no quotes on the key);
    # bench_serving's --output-details server_info field prints a dict repr
    # ('key': value) instead -- accept both so the parse is not tied to which
    # one this particular log line happened to use (g1_probe smoke 900054
    # found the first regex only matched the dict form and silently left
    # every field null against a dataclass-repr log).
    m = re.search(r"disable_piecewise_cuda_graph['\"]?\s*[:=]\s*(True|False)", log)
    if m: banner["disable_piecewise_cuda_graph"] = (m.group(1) == "True")
    m = re.search(r"piecewise_cuda_graph_max_tokens['\"]?\s*[:=]\s*(-?\d+)", log)
    if m: banner["piecewise_cuda_graph_max_tokens"] = int(m.group(1))
    m = re.search(r"chunked_prefill_size['\"]?\s*[:=]\s*(-?\d+)", log)
    if m: banner["chunked_prefill_size"] = int(m.group(1))
    m = re.search(r"Disable piecewise CUDA graph because ([^\n]*)", log)
    if m: banner["disable_piecewise_reason"] = m.group(1).strip()
except OSError:
    pass

rows = []
for rate in rates:
    jf = f"{out_dir}/bench_{arm}_s{seed}_r{rate}.jsonl"
    row = {"rate": rate, "offered_np": np_, "completed": None, "duration": None,
           "achieved_rps": None, "attainment": None, "extraction": "empty"}
    try:
        recs = [json.loads(l) for l in open(jf) if l.strip()]
        if recs:
            r = recs[-1]
            row["completed"] = r.get("completed")
            row["duration"] = r.get("duration")
            row["mean_ttft_ms"] = r.get("mean_ttft_ms")
            row["p95_itl_ms"] = r.get("p95_itl_ms")
            if row["completed"] is not None and row["duration"]:
                row["achieved_rps"] = row["completed"] / row["duration"]
                row["attainment"] = row["achieved_rps"] / rate
            row["extraction"] = "ok"
    except (OSError, json.JSONDecodeError):
        pass
    rows.append(row)

rec = {"arm": arm, "seed": seed, "boot_ok": True, "banner": banner, "rates": rows}
with open(f"{out_dir}/g1_{arm}_s{seed}.json", "w") as f:
    json.dump(rec, f, indent=2)
with open(summary_path, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"wrote g1_{arm}_s{seed}.json:", [(r["rate"], r["extraction"], r["attainment"]) for r in rows])
PY
  done
done

echo "############ artifacts in $OUT ############"
ls -la "$OUT"
echo "############ g1_summary.jsonl ############"
cat "$SUMMARY"
