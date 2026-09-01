#!/bin/bash
#SBATCH --job-name=cp0p1
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# CP-0 prerequisite P1 -- the SEVEN acceptance tests of PREREG_CP0_2026-08-28.md
# section 2.2.  This job measures the INSTRUMENT, not the arms.  It produces no
# goodput number, ranks nothing, and its PASS carries no information about
# whether chunked prefill helps (prereg section 9; PROJECT_STATUS.md "방법론
# 게이트" #26: "대형 캠페인 제출 전 배관 스모크를 규율로 등재한다 -- 스모크의
# PASS는 성능 판정에 아무 정보도 주지 않는다").
#
# ** NOT SUBMITTED BY THE ENGINE-PORTER. **  presubmit.py is globally blocked by
# two other tracks (M4R SINGLE_LABEL_FORCED, TC1 RESTRICTIONS_INERT) that are not
# CP's to fix.  Submission is the user's call.
#
# Boots (all cudagraph ON, the CP-0 operating point; flags identical to prereg
# section 6 except for the chunked-prefill-size lever):
#   1  cps 512   probe ON   -> P1-a, P1-e (phases S and B), P1-c ON half
#   2  cps -1    probe ON   -> P1-b            (this arm's prefill runs eager)
#   3  cps 4096  probe ON   -> P1-d            (batch-boundary positive control)
#   4  cps 512   probe OFF  -> P1-c OFF half
#   5+ cps 512   ON/OFF x NREP paired          -> P1-f (neutrality)
#   P1-g is recorded in every ON boot's chunk_open line and cross-checked
#   against the server log banner.
#
# usage: sbatch p1_accept.sbatch [MODEL] [CTX] [BACKEND] [NREP]

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

MODEL="${1:-Zyphra/Zamba2-2.7B}"
CTX="${2:-4096}"
BACKEND="${3:-triton}"
NREP="${4:-3}"

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
OUT="$ROOT/p1_accept_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST" || exit 2
grep -E "chunk_probe|scheduler\.py" "$MANIFEST"

echo "=== CPU regression (the same gate CLAUDE.md registers) ==="
python -m unittest discover -s workspace/engine-port/tests 2>&1 | tail -3

PORT_BASE=$((34000 + SLURM_JOB_ID % 400))

boot_server () {   # $1=cps $2=port $3=srvlog $4=probe_path("" = OFF) $5=arm
  local CPS="$1" PORT="$2" SRVLOG="$3" PROBE="$4" ARM="$5"
  if [ -n "$PROBE" ]; then
    export PDMUX_CHUNK_PROBE_PATH="$PROBE"
    export PDMUX_CHUNK_PROBE_ARM="$ARM"
    export PDMUX_CHUNK_PROBE_RUN_ID="p1accept_${SLURM_JOB_ID}"
    export PDMUX_CHUNK_PROBE_WORKLOAD_ID="$ARM"
  else
    unset PDMUX_CHUNK_PROBE_PATH PDMUX_CHUNK_PROBE_ARM \
          PDMUX_CHUNK_PROBE_RUN_ID PDMUX_CHUNK_PROBE_WORKLOAD_ID
  fi
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$BACKEND" \
    --chunked-prefill-size "$CPS" \
    --disable-radix-cache --mem-fraction-static 0.82 \
    --max-running-requests 48 --context-length "$CTX" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  echo $!
}

wait_ready () {   # $1=pid $2=port -> echoes 1/0
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

descendants () {  # $1=pid -> every descendant pid, deepest first
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    descendants "$child"
    echo "$child"
  done
}

kill_server () {  # $1=pid $2=port
  # Job 899768 lost two entire boots here (RESULT_P1_899768_2026-08-28.md): the
  # old body was `kill $PID; pkill -9 ...` with NO grace period, so nothing
  # downstream ever got to flush.  Two facts make the naive sequence lossy:
  #
  #   * the chunk probe lives in the SCHEDULER process, not in the launcher, and
  #     SGLang's launcher reaps its children with SIGKILL
  #     (sglang/launch_server.py `finally: kill_process_tree(os.getpid(),
  #     include_parent=False)` -> srt/utils/common.py:1054 `child.kill()`), so a
  #     SIGTERM aimed only at the launcher never reaches the probe at all;
  #   * Python's default SIGTERM disposition does not run `atexit`.
  #
  # So: signal the children FIRST (their SIGTERM handler closes the probe and
  # writes the shutdown summary), give them a flush window, only then ask the
  # launcher to drain, and reach for SIGKILL solely as a timeout backstop.  The
  # probe additionally flushes on a wall-clock interval, which is what covers
  # the case where this sequence is bypassed entirely.
  local PID="$1" PORT="$2" p i
  local GRACE_S="${KILL_GRACE_S:-30}" FLUSH_S="${KILL_FLUSH_S:-3}"
  local SETTLE_S="${KILL_SETTLE_S:-5}"
  for p in $(descendants "$PID"); do kill -TERM "$p" 2>/dev/null; done
  sleep "$FLUSH_S"
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

banner_evidence () {  # $1=srvlog $2=tag -- P1-g cross-check from the server log
  echo "--- piecewise/cudagraph banner ($2) ---"
  grep -iE "piecewise|capture size|Capture cuda graph|Disable piecewise" "$1" \
    | grep -viE "Future|pynvml" | head -10
}

# --------------------------------------------------------------------------
# boots 1-4: the functional acceptance tests
# --------------------------------------------------------------------------
run_probe_boot () {  # $1=cps $2=arm $3=mode $4=extra_mode("" or greedy)
  local CPS="$1" ARM="$2" MODE="$3"
  local PORT=$((PORT_BASE + RANDOM % 100))
  local SRVLOG="$OUT/srv_${ARM}.log"
  local JSONL="$OUT/chunk_${ARM}.jsonl"
  local PID; PID=$(boot_server "$CPS" "$PORT" "$SRVLOG" "$JSONL" "$ARM")
  local OK; OK=$(wait_ready "$PID" "$PORT")
  echo "boot arm=$ARM cps=$CPS ok=$OK port=$PORT"
  banner_evidence "$SRVLOG" "$ARM"
  if [ "$OK" != 1 ]; then
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
    echo '{"boot_ok": false, "arm": "'"$ARM"'"}' > "$OUT/boot_${ARM}.json"
    kill_server "$PID" "$PORT"; return 1
  fi
  python "$ROOT/p1_accept_workload.py" --port "$PORT" --mode "$MODE" --cps "$CPS" \
    --out "$OUT/workload_${ARM}.json" --out-segments "$OUT/segments_${ARM}.json"
  echo "workload rc=$?"
  if [ "$ARM" = "cps512_on" ]; then
    python "$ROOT/p1_accept_workload.py" --port "$PORT" --mode greedy --cps "$CPS" \
      --out "$OUT/greedy_on.json"
  fi
  # Flush ping.  The shutdown summary comes from an atexit hook that a SIGKILL
  # would skip; a single 1-token generate after the periodic interval has
  # elapsed forces one more summary that already contains every counter, so a
  # hard kill cannot leave the roll-up without a cross-check.  It lands outside
  # every registered segment window, so it changes no segment statistic.
  #
  # 2026-09-01: this ping was already here in job 899768 and did NOT save those
  # boots -- it only ENQUEUES a summary, and the writer had no reason to flush
  # it.  It is kept because it is still the cheapest way to get one more
  # complete roll-up on the record, but the thing that makes the enqueued
  # records reach the disk is the writer's interval flush plus the SIGTERM-first
  # sequence in kill_server.  Do not read this ping as the durability mechanism.
  sleep 7
  curl -s -o /dev/null http://127.0.0.1:$PORT/generate \
    -H 'Content-Type: application/json' \
    -d '{"input_ids":[1],"sampling_params":{"max_new_tokens":1,"temperature":0}}'
  sleep 1
  kill_server "$PID" "$PORT"
  python "$ROOT/analyze_chunk_probe.py" --jsonl "$JSONL" \
    --segments "$OUT/segments_${ARM}.json" --out "$OUT/boot_${ARM}.json"
  echo "analyze rc=$? -> $OUT/boot_${ARM}.json"
}

echo "############ boot 1: cps 512, probe ON (P1-a, P1-e, P1-c ON) ############"
run_probe_boot 512 cps512_on p1e

echo "############ boot 2: cps -1, probe ON (P1-b) ############"
run_probe_boot -1 mono_on mono

echo "############ boot 3: cps 4096, probe ON (P1-d) ############"
run_probe_boot 4096 cps4096_on d

echo "############ boot 4: cps 512, probe OFF (P1-c OFF half) ############"
PORT=$((PORT_BASE + RANDOM % 100))
SRVLOG="$OUT/srv_cps512_off.log"
PID=$(boot_server 512 "$PORT" "$SRVLOG" "" "cps512_off")
OK=$(wait_ready "$PID" "$PORT")
echo "boot arm=cps512_off ok=$OK"
banner_evidence "$SRVLOG" "cps512_off"
if [ "$OK" = 1 ]; then
  python "$ROOT/p1_accept_workload.py" --port "$PORT" --mode greedy --cps 512 \
    --out "$OUT/greedy_off.json"
fi
kill_server "$PID" "$PORT"
# The OFF boot must contain no probe output at all -- the strongest form of
# "default OFF" is that the file does not exist.
if [ -e "$OUT/chunk_cps512_off.jsonl" ]; then
  echo "P1-c VIOLATION: the probe wrote output on a boot where it was OFF"
fi

# --------------------------------------------------------------------------
# boot 5+: P1-f neutrality, NREP paired ON/OFF boots, order coin-flipped
# --------------------------------------------------------------------------
echo "############ P1-f: $NREP paired ON/OFF boots ############"
THR="$OUT/throughput_pairs.json"
echo '{"pairs": [' > "$THR"
for REP in $(seq 1 "$NREP"); do
  ON_TP=""; OFF_TP=""
  if [ $((RANDOM % 2)) = 0 ]; then ORDER="on off"; else ORDER="off on"; fi
  for STATE in $ORDER; do
    PORT=$((PORT_BASE + 200 + RANDOM % 100))
    SRVLOG="$OUT/srv_f_${STATE}_rep${REP}.log"
    if [ "$STATE" = on ]; then
      PROBE="$OUT/chunk_f_rep${REP}.jsonl"
    else
      PROBE=""
    fi
    PID=$(boot_server 512 "$PORT" "$SRVLOG" "$PROBE" "f_${STATE}_rep${REP}")
    OK=$(wait_ready "$PID" "$PORT")
    if [ "$OK" = 1 ]; then
      BJ="$OUT/bench_f_${STATE}_rep${REP}.jsonl"; : > "$BJ"
      python -m sglang.bench_serving --backend sglang --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" --dataset-name sharegpt \
        --dataset-path "$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --sharegpt-context-len 4000 --num-prompts 200 --request-rate 6 \
        --warmup-requests 4 --output-details --output-file "$BJ" \
        --tag "p1f_${STATE}_rep${REP}" \
        | grep -E "Successful requests|Request throughput"
      TP=$(python - "$BJ" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(rows[-1].get("request_throughput", "null") if rows else "null")
PY
)
      if [ "$STATE" = on ]; then ON_TP="$TP"; else OFF_TP="$TP"; fi
    fi
    kill_server "$PID" "$PORT"
  done
  SEP=","; [ "$REP" = "$NREP" ] && SEP=""
  echo "  {\"rep\": $REP, \"on\": ${ON_TP:-null}, \"off\": ${OFF_TP:-null}}$SEP" >> "$THR"
done
echo ']}' >> "$THR"

# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
echo "############ scoring the seven acceptance tests ############"
python "$ROOT/check_p1_acceptance.py" \
  --expectations "$ROOT/p1e_expectations.json" \
  --boot cps512="$OUT/boot_cps512_on.json" \
  --boot mono="$OUT/boot_mono_on.json" \
  --boot cps4096="$OUT/boot_cps4096_on.json" \
  --phase p1e="$OUT/workload_cps512_on.json" \
  --phase d="$OUT/workload_cps4096_on.json" \
  --greedy-on "$OUT/greedy_on.json" --greedy-off "$OUT/greedy_off.json" \
  --throughput "$THR" \
  --out "$OUT/P1_VERDICT.json"
echo "checker rc=$?"
python -c "import json,sys;d=json.load(open('$OUT/P1_VERDICT.json'));print(d['verdict']);print(' measurement:',d['verdict_measurement'],'| expectation:',d['verdict_expectation'],'| refuted:',d['refuted_tests']);[print(' ',k,v['label'],v['note'][:120]) for k,v in d['tests'].items()]"

echo "############ artifacts in $OUT ############"
ls -la "$OUT"
