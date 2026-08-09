#!/bin/bash
#SBATCH --job-name=g2eaprobe
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2/%x_%j.err
#
# ==============================================================================
# E-A Stage 1 -- HARD GATE: does --enable-mixed-chunk actually activate at
# runtime for the two flag combinations E-A's arms A1m/A3m need?
#   combo default = (unset --chunked-prefill-size, i.e. falls to the 8192
#                    A100 default) + --enable-mixed-chunk   -> matches A1m
#   combo 512     = --chunked-prefill-size 512 + --enable-mixed-chunk -> A3m
# Probed on BOTH Gate-2 models (Zamba2-2.7B/triton, Granite-4.0-h-micro-base/
# flashinfer) since the gating condition is read from server_args at scheduler
# construction time per-process and nothing rules out a model/backend-specific
# silent-disable path a priori (server_args.py has several; see task brief).
#
# Ground truth for "activation" (source-verified 2026-08-07, NOT inferred from
# behaviour alone):
#   scheduler.py:892-896  self.is_mixed_chunk = (chunked_prefill_size is not
#                          None) and server_args.enable_mixed_chunk
#   scheduler.py:2536-2549 mix_with_running() fires only if self.is_mixed_chunk
#                          and the running (decode) batch is non-empty.
# Silent-disable paths audited in server_args.py (grep, 2026-08-07): all require
# dual_chunk_flash_attn backend, multimodal+transformers backend, speculative
# decoding, pp_size>1, or diffusion-LLM mode -- NONE apply to Zamba2/Granite
# plain-decode triton/flashinfer serving. This probe verifies that audit against
# the live server rather than trusting it blind.
#
# Three independent runtime signals collected per combo x model:
#   (1) /server_info top-level dataclasses.asdict(server_args) (tokenizer-
#       manager-side copy)
#   (2) /server_info -> internal_states[0] = vars(get_global_server_args())
#       returned live by the SCHEDULER process via IPC (scheduler.py:3102) --
#       this is the object init_chunked_prefill() actually reads.
#   (3) HOLB probe's first jsonl record (event=holb_open), tags assembled at
#       scheduler-construction time from the same live server_args
#       (holb_probe.py:568-576) -- independent code path, third witness.
# AVAILABLE requires all three to agree: enable_mixed_chunk=true AND
# chunked_prefill_size not null/<=0.
#
# Functional smoke (not a full G-1/G-2 gate -- that's Stage 2 territory):
# 2 concurrent long (~2000 tok) greedy generate calls, staggered so the 2nd
# request's chunked prefill admits while the 1st is mid-decode -- the only
# way mix_with_running() actually fires. Must both return HTTP 200 with no
# server-side traceback/assert.
# ==============================================================================
set -uo pipefail

echo "=== HEADER: node placement + concurrent jobs ==="
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "--- squeue --all -p amd_a100nv_8 at job start ---"
squeue --all -p amd_a100nv_8 -o "%.10i %.9P %.20j %.8T %.10M %N %.8u" 2>&1 || true
echo "=== END HEADER ==="

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2
mkdir -p "$OUT"
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache_probe_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== env sync ==="
MANIFEST="$OUT/runtime_source_manifest_g2eaprobe_${SLURM_JOB_ID}.sha256"
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh "$MANIFEST"
echo "manifest -> $MANIFEST"
diff <(sort workspace/engine-port/results/p1_gates/gate1/reference_manifest_873944.sha256) \
     <(sort "$MANIFEST") > "$OUT/manifest_diff_g2eaprobe_${SLURM_JOB_ID}.txt" || true
echo "--- manifest diff vs 873944 reference (informational; probe does not abort on drift) ---"
cat "$OUT/manifest_diff_g2eaprobe_${SLURM_JOB_ID}.txt"

PORT_BASE=$((33000 + SLURM_JOB_ID % 400))

wait_ready () {   # $1=pid $2=port -> echoes 1/0
  local PID="$1" PORT="$2" ok=0
  for i in $(seq 1 170); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "SERVER DIED (pid $PID)" >&2; break; }
    sleep 5
  done
  echo $ok
}

kill_server () {  # $1=pid $2=port
  local PID="$1" PORT="$2"
  kill "$PID" 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 3
}

capture_evidence () {  # $1=srvlog
  local SRVLOG="$1"
  echo "--- capture evidence ($SRVLOG) ---"
  grep -iE "captur|cuda graph|cudagraph" "$SRVLOG" | grep -viE "Future|pynvml" | head -8
  echo "--- mixed-chunk / chunked-prefill mentions ($SRVLOG) ---"
  grep -iE "mixed.chunk|chunked prefill|chunked_prefill" "$SRVLOG" | grep -viE "Future|pynvml" | head -15
  echo "--- error evidence ($SRVLOG) ---"
  grep -iE "error|traceback|not support|illegal|assert" "$SRVLOG" | grep -viE "Future|pynvml" | head -15
  echo "--- HOLB banner ($SRVLOG) ---"
  grep "HOLB probe ENABLED" "$SRVLOG" || true
}

source "$OUT/g2_holb_phaseA_lib.sh"   # build_g2holb_prompt, greedy_call

run_probe () {   # $1=model $2=ctx $3=backend $4=combo_tag $5=extra_flags
  local MODEL="$1" CTX="$2" BACKEND="$3" TAG="$4" EXTRA="$5"
  local PORT=$((PORT_BASE + RANDOM % 100))
  local SRVLOG="$OUT/g2eaprobe_srv_${TAG}_${SLURM_JOB_ID}.log"
  local INFO_JSON="$OUT/g2eaprobe_serverinfo_${TAG}_${SLURM_JOB_ID}.json"
  local HOLB="$OUT/g2eaprobe_${TAG}_${SLURM_JOB_ID}.holb.jsonl"
  export PDMUX_HOLB_PATH="$HOLB"
  export PDMUX_HOLB_RUN_ID="${SLURM_JOB_ID}_probe_${TAG}"
  export PDMUX_HOLB_WORKLOAD_ID="probe"
  echo ""
  echo "############# PROBE tag=$TAG model=$MODEL flags=[$EXTRA] port=$PORT #############"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
    --attention-backend "$BACKEND" \
    --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
    $EXTRA --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  local PID=$!
  local OK; OK=$(wait_ready "$PID" "$PORT")
  echo "boot_ok=$OK tag=$TAG"
  capture_evidence "$SRVLOG"
  if [ "$OK" != 1 ]; then
    echo "BOOT_FAILED tag=$TAG"
    tail -40 "$SRVLOG" | grep -viE "pynvml|Future"
    kill_server "$PID" "$PORT"
    echo "PROBE_RESULT tag=$TAG verdict=BOOT_FAILED" >> "$OUT/g2eaprobe_summary_${SLURM_JOB_ID}.txt"
    return
  fi

  echo "--- /server_info top-level (tokenizer-manager copy) ---"
  curl -s "http://127.0.0.1:$PORT/server_info" > "$INFO_JSON" || true
  python3 - "$INFO_JSON" "$TAG" <<'PYEOF'
import json, sys
path, tag = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"SERVER_INFO_PARSE_FAILED tag={tag} err={e}")
    sys.exit(0)
top = {
    "enable_mixed_chunk": d.get("enable_mixed_chunk"),
    "chunked_prefill_size": d.get("chunked_prefill_size"),
    "disable_overlap_schedule": d.get("disable_overlap_schedule"),
    "piecewise_cuda_graph_max_tokens": d.get("piecewise_cuda_graph_max_tokens"),
    "max_prefill_tokens": d.get("max_prefill_tokens"),
}
print(f"SERVER_INFO_TOP tag={tag} {json.dumps(top)}")
states = d.get("internal_states") or []
if states:
    s0 = states[0]
    sched = {
        "enable_mixed_chunk": s0.get("enable_mixed_chunk"),
        "chunked_prefill_size": s0.get("chunked_prefill_size"),
        "disable_overlap_schedule": s0.get("disable_overlap_schedule"),
    }
    print(f"SERVER_INFO_SCHEDULER tag={tag} {json.dumps(sched)}")
else:
    print(f"SERVER_INFO_SCHEDULER tag={tag} MISSING_internal_states")
PYEOF

  # correctness sanity
  local RESP
  RESP=$(curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":8,"temperature":0}}')
  echo "SANITY_RAW tag=$TAG: $RESP"
  if echo "$RESP" | grep -q "Paris"; then echo "CORRECTNESS tag=$TAG PASS"; else echo "CORRECTNESS tag=$TAG FAIL"; fi

  # functional smoke: 2 concurrent long (~2000 tok) greedy requests, staggered
  # so request 2's prefill (possibly chunked) admits while request 1 is mid-decode
  local PROMPT_TXT="$OUT/g2eaprobe_prompt_${MODEL##*/}_${SLURM_JOB_ID}.txt"
  local PROMPT_META="$OUT/g2eaprobe_prompt_meta_${MODEL##*/}_${SLURM_JOB_ID}.json"
  if [ ! -s "$PROMPT_TXT" ]; then
    local INLEN=2000
    build_g2holb_prompt
  fi
  local BURST_BODY; BURST_BODY="$(python3 -c "import json,sys;print(json.dumps({'text':open('$PROMPT_TXT').read(),'sampling_params':{'max_new_tokens':150,'temperature':0}}))")"
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      -d "$BURST_BODY" \
      -o "$OUT/g2eaprobe_burst1_${TAG}_${SLURM_JOB_ID}.json" -w "BURST1_HTTP=%{http_code}\n" &
  local BPID1=$!
  sleep 1.2
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
      -d "$BURST_BODY" \
      -o "$OUT/g2eaprobe_burst2_${TAG}_${SLURM_JOB_ID}.json" -w "BURST2_HTTP=%{http_code}\n" &
  local BPID2=$!
  wait "$BPID1" "$BPID2"
  local B1ERR B2ERR
  B1ERR=$(python3 -c "import json;d=json.load(open('$OUT/g2eaprobe_burst1_${TAG}_${SLURM_JOB_ID}.json'));print('error' in d)" 2>&1)
  B2ERR=$(python3 -c "import json;d=json.load(open('$OUT/g2eaprobe_burst2_${TAG}_${SLURM_JOB_ID}.json'));print('error' in d)" 2>&1)
  echo "BURST_CHECK tag=$TAG req1_has_error=$B1ERR req2_has_error=$B2ERR"
  echo "--- post-burst error scan ---"
  grep -iE "error|traceback|assert" "$SRVLOG" | grep -viE "Future|pynvml" | tail -20

  kill -0 "$PID" 2>/dev/null && echo "SERVER_ALIVE_AFTER_BURST tag=$TAG yes" || echo "SERVER_ALIVE_AFTER_BURST tag=$TAG no (CRASHED)"

  echo "--- HOLB holb_open record (scheduler-process witness) ---"
  if [ -f "$HOLB" ]; then
    head -1 "$HOLB"
    python3 -c "
import json
with open('$HOLB') as f:
    line = f.readline()
d = json.loads(line)
print('HOLB_TAGS tag=$TAG arm_enable_mixed_chunk=' + str(d.get('arm_enable_mixed_chunk')) + ' arm_chunked_prefill_size=' + str(d.get('arm_chunked_prefill_size')) + ' arm_enable_overlap=' + str(d.get('arm_enable_overlap')))
" 2>&1
  else
    echo "HOLB_FILE_MISSING tag=$TAG path=$HOLB"
  fi

  kill_server "$PID" "$PORT"

  # verdict
  python3 - "$INFO_JSON" "$HOLB" "$TAG" "$B1ERR" "$B2ERR" "$OUT/g2eaprobe_summary_${SLURM_JOB_ID}.txt" <<'PYEOF'
import json, sys
info_path, holb_path, tag, b1err, b2err, summary_path = sys.argv[1:7]
def bad(v):
    return None
try:
    d = json.load(open(info_path))
except Exception:
    d = {}
top_mixed = d.get("enable_mixed_chunk")
top_cps = d.get("chunked_prefill_size")
sched_mixed = None
sched_cps = None
states = d.get("internal_states") or []
if states:
    sched_mixed = states[0].get("enable_mixed_chunk")
    sched_cps = states[0].get("chunked_prefill_size")
holb_mixed = None
holb_cps = None
try:
    with open(holb_path) as f:
        rec = json.loads(f.readline())
    holb_mixed = rec.get("arm_enable_mixed_chunk")
    holb_cps = rec.get("arm_chunked_prefill_size")
except Exception:
    pass

def cps_active(cps):
    return cps is not None and (not isinstance(cps, (int, float)) or cps > 0)

witnesses = [
    ("server_info_top", top_mixed, cps_active(top_cps)),
    ("server_info_scheduler", sched_mixed, cps_active(sched_cps)),
    ("holb_open", holb_mixed, cps_active(holb_cps)),
]
all_active = all((w[1] is True and w[2]) for w in witnesses)
any_missing = any(w[1] is None for w in witnesses)
burst_ok = (b1err.strip() == "False") and (b2err.strip() == "False")

if any_missing:
    verdict = "UNDETERMINED_MISSING_WITNESS"
elif all_active and burst_ok:
    verdict = "AVAILABLE"
elif all_active and not burst_ok:
    verdict = "AVAILABLE_BUT_BURST_ERROR"
else:
    verdict = "UNAVAILABLE"

line = (f"PROBE_RESULT tag={tag} verdict={verdict} "
        f"top=({top_mixed},{top_cps}) scheduler=({sched_mixed},{sched_cps}) "
        f"holb=({holb_mixed},{holb_cps}) burst_ok={burst_ok}")
print(line)
with open(summary_path, "a") as f:
    f.write(line + "\n")
PYEOF
}

: > "$OUT/g2eaprobe_summary_${SLURM_JOB_ID}.txt"

# Zamba2-2.7B (triton, ctx4096) -- both combos
run_probe "Zyphra/Zamba2-2.7B" 4096 triton "zamba2_default_mixed"  "--enable-mixed-chunk"
run_probe "Zyphra/Zamba2-2.7B" 4096 triton "zamba2_chunk512_mixed" "--chunked-prefill-size 512 --enable-mixed-chunk"

# Granite-4.0-h-micro-base (flashinfer, ctx8192) -- both combos
run_probe "ibm-granite/granite-4.0-h-micro-base" 8192 flashinfer "granite_default_mixed"  "--enable-mixed-chunk"
run_probe "ibm-granite/granite-4.0-h-micro-base" 8192 flashinfer "granite_chunk512_mixed" "--chunked-prefill-size 512 --enable-mixed-chunk"

echo ""
echo "############# STAGE 1 PROBE SUMMARY #############"
cat "$OUT/g2eaprobe_summary_${SLURM_JOB_ID}.txt"
echo "DONE_G2EAPROBE"
