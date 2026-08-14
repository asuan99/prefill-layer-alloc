#!/bin/bash
#SBATCH --job-name=stksmoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/sticky_smoke
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/sticky_smoke/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/sticky_smoke/%x_%j.err
# ============================================================================
# PDMUX_STICKY_PARTITION -- CORRECTNESS SMOKE.  NOT a performance experiment.
#
# This job answers exactly two questions and nothing else:
#   (A) CORRECTNESS: with the flag ON, does the engine produce the SAME tokens
#       as with the flag OFF?  Greedy (temperature 0), one request at a time,
#       fixed prompts, so the only difference between the two arms is which
#       stream group decode runs on.  Byte-inequality = gate FAIL.
#   (B) OBSERVATION (not a gate here): what does E1_DECODE_REALIZED -- the
#       decode-active TIME-WEIGHTED fraction of decode work that ran at the
#       target D SM -- read on each arm?  Job 872077 measured 0.104-0.187 on
#       Ha8 with the flag absent.  The pre-registered target for the flag is
#       >= 0.90, but this run only REPORTS the number; it is not tuned to it.
#
# It makes NO throughput/latency/goodput claim and the probe is far too short
# to support one.  Any performance statement needs experiment-runner +
# result-analyst + claims-auditor, not this file.
#
# SHAPE.  One arm (Ha8 = Zamba2-7B-Instruct, the hybrid family under study),
# one cell (d16 = prefill 92 / decode 16 SM: the largest OFF-vs-ON contrast,
# and the E1 cell with the lowest measured realized fraction), two boots
# (STICKY=0, STICKY=1).  Server args are copied verbatim from
# e1_m3_control.sbatch so the only difference from job 872077 is the flag.
#
# TELEMETRY SLICING.  Both phases of an arm append to one JSONL, so the line
# count is recorded after the correctness phase and the probe-only slice is
# `tail -n +N`.  Whole-file numbers are printed too; the probe-only slice is
# the one comparable to the 872077 baseline.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/sticky_smoke
E1=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_stk_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM=Ha8
MODEL="Zyphra/Zamba2-7B-Instruct"
BACKEND=triton
CELL="${CELL:-d16}"
CFG="$E1/pdmux_e1_${CELL}.yml"
DSM="${CELL#d}"

SGPT="$ROOT/hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
CTX=4096; SGPT_CTXLEN=3600; CTXCAP=$(( CTX + 512 + 256 ))
CAP=48; MAMBA_POOL=$CAP
RATE="${RATE:-2}"; NP="${NP:-100}"; SEED=1

JOBTAG="${SLURM_JOB_ID:-local}"
RESULT="$HERE/stksmoke_${ARM}_${CELL}_${JOBTAG}_result.txt"; : > "$RESULT"
echo "=== STICKY SMOKE arm=$ARM cell=$CELL D=$DSM rate=$RATE np=$NP job=$JOBTAG ===" | tee -a "$RESULT"
[ -f "$CFG" ] || { echo "MISSING_CONFIG $CFG" | tee -a "$RESULT"; exit 4; }

for STICKY in 0 1; do
  RUNID="stk${STICKY}_${ARM}_${CELL}_${JOBTAG}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  TELE="$HERE/${RUNID}_telemetry.jsonl"; : > "$TELE"
  GEN="$HERE/${RUNID}_greedy.json"
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")

  export PDMUX_TELEMETRY_PATH="$TELE"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="sticky_smoke_${ARM}_${CELL}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
  export PDMUX_TRACE_FORCE_PREFILL=0
  export PDMUX_STICKY_PARTITION=$STICKY

  echo "=== STICKY=$STICKY port=$PORT runid=$RUNID ===" | tee -a "$RESULT"
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
  for i in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED sticky=$STICKY" | tee -a "$RESULT"; break; }
    sleep 5
  done
  echo "boot_ok=$ok sticky=$STICKY" | tee -a "$RESULT"
  # The flag must be visible in the server log when ON and absent when OFF.
  grep -c "sticky partition ENABLED" "$SRVLOG" \
    | sed "s/^/  STICKY_LOG_LINES sticky=$STICKY: /" | tee -a "$RESULT"
  if [ "$ok" != 1 ]; then kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; continue; fi

  # ---- (A) correctness: greedy, strictly sequential, one request in flight --
  python3 - "$PORT" "$GEN" <<'PY' | tee -a "$RESULT"
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 + 25?\nA:",
    "Once upon a time in a small village by the sea,",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List the first five prime numbers:",
    "Translate to French: The weather is nice today.",
]
res = []
for p in PROMPTS:
    body = json.dumps({"text": p,
                       "sampling_params": {"temperature": 0, "max_new_tokens": 48}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res.append(json.loads(r.read().decode())["text"])
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(f"GREEDY_DONE n={len(res)} first={res[0][:60]!r}")
PY
  LINES_AFTER_GREEDY=$(wc -l < "$TELE")
  echo "TELE_LINES_AFTER_GREEDY sticky=$STICKY n=$LINES_AFTER_GREEDY" | tee -a "$RESULT"
  echo "$LINES_AFTER_GREEDY" > "$HERE/${RUNID}_greedy_lines.txt"

  # ---- (B) short load probe, telemetry only ---------------------------------
  JOUT="$HERE/${RUNID}_probe.jsonl"; : > "$JOUT"
  python -m sglang.bench_serving --backend sglang --model "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --dataset-name sharegpt --dataset-path "$SGPT" --sharegpt-context-len $SGPT_CTXLEN \
    --num-prompts $NP --request-rate $RATE --seed $SEED --output-details \
    --output-file "$JOUT" > "$HERE/${RUNID}_probe_bench.log" 2>&1
  echo "probe_rc=$? sticky=$STICKY" | tee -a "$RESULT"

  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sleep 5
done

# ---- verdicts ---------------------------------------------------------------
python3 - "$HERE" "$ARM" "$CELL" "$JOBTAG" "$DSM" "$E1" <<'PY' | tee -a "$RESULT"
import json, sys, subprocess
here, arm, cell, job, dsm, e1 = sys.argv[1:7]
sys.path.insert(0, e1)
import e1_pin_check as pc

# (A) token-level equivalence
a = json.load(open(f"{here}/stk0_{arm}_{cell}_{job}_greedy.json"))
b = json.load(open(f"{here}/stk1_{arm}_{cell}_{job}_greedy.json"))
same = a == b
print(f"STICKY_CORRECTNESS n={len(a)} identical={same} "
      f"verdict={'PASS' if same else 'FAIL'}")
if not same:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  DIFF[{i}] off={x[:120]!r}\n         on={y[:120]!r}")

# (B) realized decode partition, whole file and probe-only slice
for sticky in (0, 1):
    run = f"stk{sticky}_{arm}_{cell}_{job}"
    tele = f"{here}/{run}_telemetry.jsonl"
    n = int(open(f"{here}/{run}_greedy_lines.txt").read().strip())
    probe = f"{here}/{run}_probe_slice.jsonl"
    with open(probe, "w") as f:
        subprocess.run(["tail", "-n", f"+{n+1}", tele], stdout=f, check=True)
    for label, path in (("ALL", tele), ("PROBE_ONLY", probe)):
        d = pc.compute_decode_realized(path, int(dsm))
        h = ", ".join(f"D{k}:{v:.1f}s" for k, v in
                      sorted(d["decode_hist"].items(), key=lambda kv: -kv[1])[:4])
        print(f"E1_DECODE_REALIZED sticky={sticky} scope={label} D={dsm} "
              f"frac={d['decode_realized_frac']:.4f} "
              f"t_decode_active={d['t_decode_active_s']:.1f}s hist=[{h}]")
    c = pc.compute_conditional_pin_gate(tele, int(dsm), None, None)
    print(f"E1_COND_PIN sticky={sticky} D={dsm} frac={c['cond_pin_frac']:.4f} "
          f"(n={c['n_concurrent_snapshots']}, t={c['t_concurrent_s']:.2f}s)")
print("=== OBSERVATION ONLY.  No performance claim is licensed by this job. ===")
PY

echo "=== sticky smoke done job=$JOBTAG ==="
