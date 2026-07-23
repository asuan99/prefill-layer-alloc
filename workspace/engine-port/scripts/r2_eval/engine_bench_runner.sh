#!/bin/bash
# Canonical single-GPU SGLang server + immutable trace replay adapter.
set -euo pipefail

run_record="${1:?run record required}"
run_dir="${2:?run directory required}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
engine_dev="${SGLANG_ENGINE_DEV:-/scratch/ehmoon/whlee/sglang_engine_dev/python}"
model="${PDMUX_MODEL:-Zyphra/Zamba2-2.7B}"
config="${PDMUX_R2_CONFIG:-${engine_root}/benchmarks/configs/pdmux_r2.yml}"
port=$((32000 + (${SLURM_JOB_ID:-1} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000))

readarray -t fields < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["trace_path"]); print(d["server_seed"]); print(d["max_running_requests"]); print(d["ttft_slo_ms"]); print(d["itl_slo_ms"]); print(d["workload"])' \
    "${run_record}"
)
trace_path="${fields[0]}"
server_seed="${fields[1]}"
max_running="${fields[2]}"
ttft_slo="${fields[3]}"
itl_slo="${fields[4]}"
workload="${fields[5]}"

source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
export PDMUX_WORKLOAD_ID="${workload}"

server_args=(
  --model-path "${model}"
  --trust-remote-code
  --dtype bfloat16
  --attention-backend triton
  --disable-radix-cache
  --mem-fraction-static "${PDMUX_MEM_FRACTION:-0.82}"
  --max-running-requests "${max_running}"
  --context-length "${PDMUX_CONTEXT_LENGTH:-16384}"
  --random-seed "${server_seed}"
  --host 127.0.0.1
  --port "${port}"
)
if [[ "${PDMUX_ENGINE_MODE:-pdmux}" == pdmux ]]; then
  server_args+=(
    --enable-pdmux
    --pdmux-config-path "${config}"
    --chunked-prefill-size "${PDMUX_CHUNKED_PREFILL_SIZE:--1}"
    --disable-overlap-schedule
  )
fi
if [[ "${PDMUX_DISABLE_CUDA_GRAPH:-0}" == 1 ]]; then
  server_args+=(--disable-cuda-graph --disable-piecewise-cuda-graph)
fi

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python -m sglang.launch_server "${server_args[@]}" \
  >"${run_dir}/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 180); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${port}/health" 2>/dev/null || true)" == 200 ]]; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -100 "${run_dir}/server.log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
curl -fsS "http://127.0.0.1:${port}/model_info" \
  >"${run_dir}/model_info.json"

python -m pdmux_eval.trace_loadgen \
  --endpoint "http://127.0.0.1:${port}/generate" \
  --trace "${trace_path}" \
  --output "${run_dir}/requests.jsonl" \
  --summary "${run_dir}/summary.json" \
  --ttft-slo-ms "${ttft_slo}" \
  --itl-slo-ms "${itl_slo}"
