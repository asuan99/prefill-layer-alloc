#!/bin/bash
# R1: compare the legacy PD-mux scheduler with the opt-in cooperative
# dual-worker state path under identical whole-phase SM partitions.
#
# The job intentionally keeps per-layer switching and coordinated scheduling
# disabled. It first validates the CPU-only state tests and the external
# editable SGLang tree, then runs five synthetic workloads for both policies.
#
# Usage:
#   sbatch workspace/engine-port/triage/r1_dual_worker.sbatch
#   sbatch workspace/engine-port/triage/r1_dual_worker_no_smi.sbatch
#   NPROMPTS=24 TRACE_EVERY=16 sbatch .../r1_dual_worker.sbatch

#SBATCH --job-name=r1_dual_worker_smi
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:30:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1=""
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
ENGINE_ROOT="$REPO_ROOT/workspace/engine-port"
ENGINE_DEV=/scratch/ehmoon/whlee/sglang_engine_dev/python
TRIAGE="$ENGINE_ROOT/triage"
OUT_ROOT="$REPO_ROOT/workspace/engine-port/results/r1_dual_worker"
JOB_ID="${SLURM_JOB_ID:-$$}"
RUN_DIR="$OUT_ROOT/job_${JOB_ID}"
MODEL="Zyphra/Zamba2-2.7B"
CTX=4096
NPROMPTS="${NPROMPTS:-48}"
TRACE_EVERY="${TRACE_EVERY:-32}"
PORT=$((37000 + JOB_ID % 200))

source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

export HF_HOME="$REPO_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$TRIAGE/.triton_cache"
mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

cat > "$RUN_DIR/manifest.txt" <<EOF
job_id=$JOB_ID
git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
host=$(hostname)
model=$MODEL
context_length=$CTX
max_running_requests=48
gpu_partition=amd_a100nv_8
port=$PORT
nprompts=$NPROMPTS
trace_every=$TRACE_EVERY
EOF

echo "===== R1 dual-worker preflight ====="
echo "job=$JOB_ID host=$(hostname) model=$MODEL run_dir=$RUN_DIR"
if [[ "${R1_USE_NVIDIA_SMI:-1}" == 1 ]]; then
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
else
  echo "nvidia-smi=SKIPPED (R1_USE_NVIDIA_SMI=0)"
fi

for required in \
  "$ENGINE_DEV/sglang/srt/multiplex/dual_worker.py" \
  "$ENGINE_DEV/sglang/srt/multiplex/multiplexing_mixin.py"; do
  test -s "$required" || { echo "MISSING external runtime file: $required"; exit 2; }
done

has_pattern() {
  local pattern="$1" file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q -- "$pattern" "$file"
  else
    grep -Eq -- "$pattern" "$file"
  fi
}

has_pattern 'class DualWorkerState' "$ENGINE_DEV/sglang/srt/multiplex/dual_worker.py"
has_pattern 'PDMUX_DUAL_WORKER' "$ENGINE_DEV/sglang/srt/multiplex/multiplexing_mixin.py"
has_pattern 'PDMUX_DUAL_WORKER_TRACE' "$ENGINE_DEV/sglang/srt/multiplex/multiplexing_mixin.py"

python -m unittest discover \
  -s "$ENGINE_ROOT/tests" -p 'test_dual_worker.py'

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
name = torch.cuda.get_device_name(0)
print(f"[preflight] GPU={name}")
assert "A100" in name, f"R1 requires A100, got {name!r}"
PY

SUMMARY="$RUN_DIR/summary.csv"

start_server() {
  local policy="$1" scenario="$2" cfg="$3"
  local server_log="$RUN_DIR/server_${scenario}_${policy}.log"
  local trace_file="$RUN_DIR/trace_${scenario}_${policy}.jsonl"
  : > "$server_log"
  : > "$trace_file"

  echo "===== START policy=$policy scenario=$scenario cfg=$cfg port=$PORT ====="
  if [[ "$policy" == dual ]]; then
    PDMUX_DUAL_WORKER=1 \
    PDMUX_DUAL_WORKER_TRACE="$trace_file" \
    PDMUX_DUAL_WORKER_TRACE_EVERY="$TRACE_EVERY" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
      --attention-backend triton \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
      --mem-fraction-static 0.82 --max-running-requests 48 \
      --context-length "$CTX" \
      --enable-pdmux --pdmux-config-path "$cfg" \
      --chunked-prefill-size -1 --disable-overlap-schedule \
      --host 127.0.0.1 --port "$PORT" > "$server_log" 2>&1 &
  else
    env -u PDMUX_DUAL_WORKER -u PDMUX_DUAL_WORKER_TRACE \
      -u PDMUX_DUAL_WORKER_TRACE_EVERY \
      python -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
        --attention-backend triton \
        --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache \
        --mem-fraction-static 0.82 --max-running-requests 48 \
        --context-length "$CTX" \
        --enable-pdmux --pdmux-config-path "$cfg" \
        --chunked-prefill-size -1 --disable-overlap-schedule \
        --host 127.0.0.1 --port "$PORT" > "$server_log" 2>&1 &
  fi
  SERVER_PID=$!

  for _ in $(seq 1 180); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:$PORT/health" 2>/dev/null || true)" == 200 ]]; then
      echo "healthy policy=$policy scenario=$scenario"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "SERVER_DIED policy=$policy scenario=$scenario"
      tail -30 "$server_log"
      return 1
    fi
    sleep 5
  done
  echo "SERVER_TIMEOUT policy=$policy scenario=$scenario"
  tail -30 "$server_log"
  return 1
}

run_correctness() {
  local policy="$1" scenario="$2"
  local output="$RUN_DIR/correctness_${scenario}_${policy}.json"
  python3 - "$PORT" "$output" <<'PY'
import json
import sys
import urllib.request

port, output = sys.argv[1:]
prompts = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "One plus one equals",
]
outputs = []
for prompt in prompts:
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        text = json.load(response).get("text", "")
    if not text:
        raise RuntimeError(f"empty output for prompt {prompt!r}")
    outputs.append({"prompt": prompt, "text": text})
with open(output, "w", encoding="utf-8") as stream:
    json.dump(outputs, stream, ensure_ascii=False, indent=2)
print(f"CORRECTNESS_OK output={output}")
PY
}

run_bench() {
  local policy="$1" scenario="$2" phase="$3" input_len="$4" output_len="$5" rate="$6"
  local jsonl="$RUN_DIR/bench_${scenario}_${policy}_${phase}.jsonl"
  local bench_log="$RUN_DIR/bench_${scenario}_${policy}_${phase}.log"
  : > "$jsonl"
  echo "===== BENCH policy=$policy scenario=$scenario phase=$phase in=$input_len out=$output_len rate=$rate ====="
  python -m sglang.bench_serving \
    --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids \
    --random-input-len "$input_len" --random-output-len "$output_len" \
    --random-range-ratio 1.0 --num-prompts "$NPROMPTS" \
    --request-rate "$rate" --output-details --output-file "$jsonl" \
    > "$bench_log" 2>&1
  grep -E 'Successful requests|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT' \
    "$bench_log" || true
  echo "$jsonl"
}

warmup() {
  local policy="$1" scenario="$2"
  python -m sglang.bench_serving \
    --backend sglang --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-ids --random-input-len 128 --random-output-len 16 \
    --random-range-ratio 1.0 --num-prompts 8 --request-rate 2 \
    > "$RUN_DIR/warmup_${scenario}_${policy}.log" 2>&1 || true
}

summarize() {
  local policy="$1" scenario="$2" trace_file="$3"
  shift 3
  python3 - "$SUMMARY" "$policy" "$scenario" "$trace_file" "$@" <<'PY'
import csv
import json
import math
import os
import sys
from collections import Counter

summary, policy, scenario, trace_file, *bench_files = sys.argv[1:]

def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * quantile
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

ttft_ms = []
itls_ms = []
good = 0
requests = 0
duration_s = 0.0
rates = []
for path in bench_files:
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            ttfts = record.get("ttfts") or []
            itls = record.get("itls") or []
            requests += len(ttfts)
            duration_s += float(record.get("duration") or 0.0)
            if record.get("request_rate") is not None:
                rates.append(str(record["request_rate"]))
            for index, ttft in enumerate(ttfts):
                ttft_ms.append(float(ttft) * 1000.0)
                request_itls = itls[index] if index < len(itls) else []
                mean_itl = (
                    sum(request_itls) / len(request_itls)
                    if request_itls else math.inf
                )
                itls_ms.append(mean_itl * 1000.0)
                if float(ttft) <= 3.0 and mean_itl <= 0.060:
                    good += 1

trace = []
if os.path.exists(trace_file):
    with open(trace_file, encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream if line.strip()]
queue_depth = [float(row["prefill_queue_depth"]) for row in trace]
queue_age = [float(row["prefill_queue_age_ms"]) for row in trace]
decode_batch = [float(row["decode_running_batch_size"]) for row in trace]
tpot = [float(row["decode_last_tpot_ms"]) for row in trace if row.get("decode_last_tpot_ms")]
blocked = Counter(row.get("prefill_admission_block_reason", "") for row in trace)
sm_pairs = Counter(f'{row.get("prefill_sms")}/{row.get("decode_sms")}' for row in trace)

row = {
    "policy": policy,
    "scenario": scenario,
    "request_rates": ";".join(rates),
    "requests": requests,
    "good_requests": good,
    "duration_s": f"{duration_s:.3f}",
    "throughput_req_s": f"{requests / duration_s:.6f}" if duration_s else "nan",
    "goodput_req_s": f"{good / duration_s:.6f}" if duration_s else "nan",
    "ttft_p50_ms": f"{percentile(ttft_ms, .50):.3f}",
    "ttft_p95_ms": f"{percentile(ttft_ms, .95):.3f}",
    "ttft_p99_ms": f"{percentile(ttft_ms, .99):.3f}",
    "itl_p50_ms": f"{percentile(itls_ms, .50):.3f}",
    "itl_p95_ms": f"{percentile(itls_ms, .95):.3f}",
    "itl_p99_ms": f"{percentile(itls_ms, .99):.3f}",
    "trace_samples": len(trace),
    "queue_depth_max": f"{max(queue_depth, default=0):.3f}",
    "queue_age_max_ms": f"{max(queue_age, default=0):.3f}",
    "decode_batch_max": f"{max(decode_batch, default=0):.3f}",
    "trace_tpot_p50_ms": f"{percentile(tpot, .50):.3f}",
    "trace_tpot_p95_ms": f"{percentile(tpot, .95):.3f}",
    "blocked_reasons": json.dumps(dict(blocked), sort_keys=True),
    "sm_pairs": json.dumps(dict(sm_pairs), sort_keys=True),
}
fields = list(row)
exists = os.path.exists(summary) and os.path.getsize(summary) > 0
with open(summary, "a", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    if not exists:
        writer.writeheader()
    writer.writerow(row)
print(f"SUMMARY policy={policy} scenario={scenario} requests={requests} good={good} trace={len(trace)}")
PY
}

compare_correctness() {
  local scenario="$1"
  python3 - "$RUN_DIR/correctness_${scenario}_legacy.json" \
    "$RUN_DIR/correctness_${scenario}_dual.json" <<'PY'
import json
import sys

left, right = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
if left != right:
    raise SystemExit("CORRECTNESS_MISMATCH legacy != dual")
print("CORRECTNESS_PARITY_OK")
PY
}

run_policy_scenario() {
  local policy="$1" scenario="$2" cfg="$3"
  local trace_file="$RUN_DIR/trace_${scenario}_${policy}.jsonl"
  start_server "$policy" "$scenario" "$cfg"
  run_correctness "$policy" "$scenario"
  warmup "$policy" "$scenario"

  case "$scenario" in
    decode_heavy)
      run_bench "$policy" "$scenario" main 2000 96 8 > "$RUN_DIR/bench_path_${scenario}_${policy}.txt"
      ;;
    prefill_burst)
      run_bench "$policy" "$scenario" main 3600 32 2 > "$RUN_DIR/bench_path_${scenario}_${policy}.txt"
      ;;
    alternating)
      run_bench "$policy" "$scenario" prefill 3600 32 2 > /dev/null
      run_bench "$policy" "$scenario" decode 2000 96 8 > /dev/null
      ;;
    low_decode_sm)
      run_bench "$policy" "$scenario" main 2000 96 8 > "$RUN_DIR/bench_path_${scenario}_${policy}.txt"
      ;;
    overload)
      run_bench "$policy" "$scenario" main 3600 96 12 > "$RUN_DIR/bench_path_${scenario}_${policy}.txt"
      ;;
  esac
  cleanup

  if [[ "$scenario" == alternating ]]; then
    summarize "$policy" "${scenario}_prefill" "$trace_file" \
      "$RUN_DIR/bench_${scenario}_${policy}_prefill.jsonl"
    summarize "$policy" "${scenario}_decode" "$trace_file" \
      "$RUN_DIR/bench_${scenario}_${policy}_decode.jsonl"
    summarize "$policy" "$scenario" "$trace_file" \
      "$RUN_DIR/bench_${scenario}_${policy}_prefill.jsonl" \
      "$RUN_DIR/bench_${scenario}_${policy}_decode.jsonl"
  else
    summarize "$policy" "$scenario" "$trace_file" \
      "$RUN_DIR/bench_${scenario}_${policy}_main.jsonl"
  fi
}

declare -A CONFIGS=(
  [decode_heavy]="$ENGINE_ROOT/results/slo_sched/pdmux_d24.yml"
  [prefill_burst]="$ENGINE_ROOT/results/slo_sched/pdmux_d24.yml"
  [alternating]="$ENGINE_ROOT/results/slo_sched/pdmux_d24.yml"
  [low_decode_sm]="$ENGINE_ROOT/results/slo_sched/pdmux_d16.yml"
  [overload]="$ENGINE_ROOT/results/slo_sched/pdmux_d24.yml"
)

for scenario in decode_heavy prefill_burst alternating low_decode_sm overload; do
  cfg="${CONFIGS[$scenario]}"
  test -s "$cfg"
  run_policy_scenario legacy "$scenario" "$cfg"
  run_policy_scenario dual "$scenario" "$cfg"
  compare_correctness "$scenario"
done

echo "===== R1 DONE ====="
echo "summary=$SUMMARY"
cat "$SUMMARY"
