#!/bin/bash
#SBATCH --job-name=cgate
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%j.err
#
# CORRECTNESS GATE -- do the arms compute the SAME FUNCTION?
#
# Every performance comparison this track plans assumes chunked prefill and SM
# partitioning are SCHEDULING choices: they change WHEN work happens, not WHAT is
# computed.  That assumption has never been tested on this tree for this model.
# If it is false, every goodput number is comparing two different functions and
# the whole comparison is void -- so this runs BEFORE any more policy spend.
#
# ★ THE INVARIANT UNDER TEST (this is the whole gate):
#     greedy(prompt | arm) == greedy(prompt | plain)   for every arm, every prompt
#   `plain` = no chunking, no pdmux = the reference configuration.
#   Sampling is temperature 0 and requests are sent ONE AT A TIME, so batch-shape
#   floating-point nondeterminism is excluded from this tier by construction.
#   (Whether CONCURRENCY changes outputs is a different question and is NOT asked
#   here -- naming it keeps a later session from reading this gate as covering it.)
#
# ★ WHAT A PASS DOES NOT MEAN: it does not mean the model is accurate, well-suited,
#   or that its outputs are good.  It means the arms agree.  Model quality is not
#   a property this project measures.
#
# usage: sbatch correctness_gate.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
OUT="$ROOT/cgate_${SLURM_JOB_ID}"; mkdir -p "$OUT"

export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base}"
BACKEND="${BACKEND:-flashinfer}"
CTX="${CTX:-32768}"
ARMS="${CG_ARMS:-plain cp512 cp2048 cp8192 d44}"
MAXNEW="${CG_MAXNEW:-48}"
PROMPTS="$ROOT/correctness_prompts.json"
[ -f "$PROMPTS" ] || { echo "MISSING $PROMPTS"; exit 2; }

unset PDMUX_CHUNK_PROBE_PATH PDMUX_SLO_SCHED PDMUX_SLO_MODE

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2
echo "CGATE model=$MODEL backend=$BACKEND ctx=$CTX arms=[$ARMS] max_new=$MAXNEW job=$SLURM_JOB_ID"

# Warm the import + weight page cache on a throwaway boot, so the FIRST MEASURED arm
# does not pay the cold-start cost and time out (job 902397: plain and cp512 lost to
# exactly that, and their BOOT_FAILED was a harness artifact, not a model fact).
echo "=== warmup boot (not scored) ==="
WPORT=$((41000 + RANDOM % 400))
python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
  --dtype bfloat16 --attention-backend "$BACKEND" --chunked-prefill-size -1 \
  --disable-piecewise-cuda-graph --disable-radix-cache --mem-fraction-static 0.82 \
  --max-running-requests 8 --context-length 4096 \
  --host 127.0.0.1 --port "$WPORT" > "$OUT/srv_warmup.log" 2>&1 &
WPID=$!
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$WPORT/health 2>/dev/null||true)" = 200 ] && break
  kill -0 "$WPID" 2>/dev/null || break
  sleep 5
done
echo "  warmup done"
kill "$WPID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $WPORT" 2>/dev/null; sleep 15

arm_flags () {
  case "$1" in
    plain)  echo "--chunked-prefill-size -1" ;;          # reference: no chunking, no pdmux
    cp512)  echo "--chunked-prefill-size 512" ;;
    cp2048) echo "--chunked-prefill-size 2048" ;;
    cp8192) echo "--chunked-prefill-size 8192" ;;
    d44)    echo "--enable-pdmux --pdmux-config-path $SLO/pdmux_d44.yml --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "__BAD__" ;;
  esac
}

for ARM in $ARMS; do
  FLAGS=$(arm_flags "$ARM"); [ "$FLAGS" = "__BAD__" ] && { echo "bad arm=$ARM"; continue; }
  PORT=$((40000 + RANDOM % 500))
  SRVLOG="$OUT/srv_${ARM}.log"
  echo ""
  echo "########## arm=$ARM port=$PORT flags=[$FLAGS] ##########"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$BACKEND" $FLAGS \
    --disable-piecewise-cuda-graph \
    --disable-radix-cache --mem-fraction-static 0.82 \
    --max-running-requests 48 --context-length "$CTX" \
    --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { OK=1; break; }
    kill -0 "$PID" 2>/dev/null || { echo "  SERVER DIED"; break; }
    sleep 5
  done
  echo "  boot_ok=$OK"
  grep -oE "chunked_prefill_size=[-0-9]+|enable_pdmux=[A-Za-z]+|disable_piecewise_cuda_graph=[A-Za-z]+" "$SRVLOG" | sort -u | tr '\n' ' '; echo
  if [ "$OK" != 1 ]; then
    echo "BOOT_FAILED arm=$ARM" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -12 "$SRVLOG" | grep -viE "pynvml|Future|^\s*$" | tail -4
    kill "$PID" 2>/dev/null; sleep 3; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 5
    continue
  fi

  # ONE AT A TIME, temperature 0 -- see the invariant note in the header.
  python3 - "$PROMPTS" "$PORT" "$OUT/gen_${ARM}.json" "$MAXNEW" "$ARM" <<'PY'
import json, sys, time, urllib.request
prompts, port, outp, maxnew, arm = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
res = []
for p in prompts:
    body = json.dumps({"text": p["prompt"],
                       "sampling_params": {"temperature": 0.0, "max_new_tokens": maxnew}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        txt = out.get("text", "")
        err = None
    except Exception as e:
        txt, err = "", f"{type(e).__name__}: {e}"
    res.append({"id": p["id"], "kind": p["kind"], "n_tokens": p["n_tokens"],
                "text": txt, "error": err, "wall_s": round(time.time() - t0, 2)})
    print(f"    {p['id']:<10} {p['n_tokens']:>6} tok  {time.time()-t0:>6.1f}s  "
          f"{'ERR ' + err if err else repr(txt[:60])}", flush=True)
json.dump({"arm": arm, "results": res}, open(outp, "w"), ensure_ascii=False)
PY
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 15
done

echo ""
echo "=== correctness check ==="
python3 "$ROOT/correctness_check.py" "$OUT" || echo "CHECK_FAILED (artifacts kept)"
echo "CGATE_DONE_${SLURM_JOB_ID}"
