#!/bin/bash
#SBATCH --job-name=r0d_parity
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:35:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Correctness gate for coordinated per-type la: greedy outputs under sustained
# concurrent load (exercises the decode-window ∥ prefill interleave) must MATCH
# a normal-pdmux server. arg1 = coord | agn.
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
TRIAGE=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/triage
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r0d_coord_la
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"
MODE="${1:-coord}"; MODEL=Zyphra/Zamba2-2.7B; PORT=$((34600 + SLURM_JOB_ID % 200))
SRVLOG="$OUT/parity_srv_${MODE}_${SLURM_JOB_ID}.log"; RES="$OUT/parity_${MODE}.txt"; : > "$RES"
if [ "$MODE" = "coord" ]; then export PDMUX_LA_COORD=1; CFG="$OUT/pdmux_coord_la.yml"; else CFG="$TRIAGE/pdmux_a100_smoke.yml"; fi
echo "MODE=$MODE cfg=$CFG port=$PORT coord=${PDMUX_LA_COORD:-unset} job=$SLURM_JOB_ID" | tee -a "$RES"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length 4096 \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!; ok=0
for i in $(seq 1 170); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED" | tee -a "$RES"; break; }; sleep 5
done
echo "boot_ok=$ok" | tee -a "$RES"
if [ "$ok" = 1 ]; then
python3 - "$PORT" "$RES" <<'PY'
import sys, json, urllib.request, concurrent.futures as cf, time, collections
port, res = sys.argv[1], sys.argv[2]
# distinct greedy prompts (varied length so prefill+decode overlap under load)
base = [
  "The capital of France is",
  "Water is made of hydrogen and",
  "The opposite of hot is",
  "One plus one equals",
  "The sun rises in the",
  "Roses are red, violets are",
]
long_ctx = "the robot walked slowly across the red planet " * 60  # ~540 tok, longer prefill
prompts = base + [long_ctx + " " + p for p in base]   # short + long-ctx variants
def gen(p):
    body = json.dumps({"text": p, "sampling_params": {"max_new_tokens": 24, "temperature": 0.0}}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body, headers={"Content-Type":"application/json"})
    return p, json.loads(urllib.request.urlopen(r, timeout=120).read())["text"]
# fire each prompt 6x concurrently (48 in flight) + staggered -> sustained prefill∥decode overlap
tasks = prompts * 6
out = collections.defaultdict(list)
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    futs = []
    for i, p in enumerate(tasks):
        futs.append(ex.submit(gen, p)); time.sleep(0.05)  # stagger arrivals
    for f in cf.as_completed(futs):
        p, t = f.result(); out[p].append(t)
# intra-server determinism: all repeats of a prompt must be identical (greedy)
intra_ok = all(len(set(v)) == 1 for v in out.values())
with open(res, "a") as fh:
    fh.write(f"INTRA_DETERMINISTIC={intra_ok}  n_prompts={len(out)}  n_req={len(tasks)}\n")
    for p in prompts:
        key = ("L:" + p[-40:]) if len(p) > 60 else p
        fh.write(f"OUT|||{key}|||{out[p][0]!r}\n")
print("INTRA_DETERMINISTIC", intra_ok)
PY
else echo "BOOT FAILED" | tee -a "$RES"; tail -25 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RES"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
echo "DONE_$MODE"
