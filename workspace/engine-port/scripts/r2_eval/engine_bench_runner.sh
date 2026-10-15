#!/bin/bash
# Canonical single-GPU SGLang server + immutable trace replay adapter.
set -euo pipefail

run_record="${1:?run record required}"
run_dir="${2:?run directory required}"
shift 2
# ---- multi-GPU options (2026-10-12, multi-GPU track M2).  ALL default to the single-GPU
# behaviour: with none of them given, server_args.txt is byte-identical to before.
#   --tp-size N     tensor parallel over N GPUs.  --max-running-requests becomes 48*N, i.e. the run
#                   record's (per-GPU) value times N (user decision 2026-10-12, design section 9-4).
#   --dp-size N     N independent replicas, round_robin routing (fixed).  Without DP-attention,
#                   SGLang's --max-running-requests is PER REPLICA (model_runner_kv_cache_mixin.py
#                   divides by dp_size only when enable_dp_attention; scheduler reports
#                   effective_max_running_requests_per_dp), so the record's value (48) is passed
#                   unchanged: 48 per replica.
#   --gpu-ids L     comma list, e.g. 0,1,2,3 -> CUDA_VISIBLE_DEVICES (pre-registrable subset of an
#                   x8 node).  Its length must equal tp*dp.
# Env fallbacks PDMUX_TP_SIZE / PDMUX_DP_SIZE / PDMUX_GPU_IDS (a flag wins).  TP and DP are not mixed.
tp_size="${PDMUX_TP_SIZE:-1}"
dp_size="${PDMUX_DP_SIZE:-1}"
gpu_ids="${PDMUX_GPU_IDS:-}"
while (( $# )); do
  case "$1" in
    --tp-size) tp_size="${2:?--tp-size needs a value}"; shift 2 ;;
    --dp-size) dp_size="${2:?--dp-size needs a value}"; shift 2 ;;
    --gpu-ids) gpu_ids="${2:?--gpu-ids needs a value}"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if ! [[ "${tp_size}" =~ ^[1-9][0-9]*$ && "${dp_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --tp-size/--dp-size must be positive integers (got tp=${tp_size} dp=${dp_size})" >&2
  exit 2
fi
if (( tp_size > 1 && dp_size > 1 )); then
  echo "ERROR: TP and DP are not mixed in one cell (tp=${tp_size} dp=${dp_size})" >&2
  exit 2
fi
n_gpus=$(( tp_size * dp_size ))
if [[ -n "${gpu_ids}" ]]; then
  if ! [[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --gpu-ids must be a comma list of GPU indices (got '${gpu_ids}')" >&2; exit 2
  fi
  n_ids=$(awk -F, '{print NF}' <<<"${gpu_ids}")
  n_distinct=$(tr ',' '\n' <<<"${gpu_ids}" | sort -u | wc -l)
  if (( n_ids != n_gpus || n_distinct != n_ids )); then
    echo "ERROR: --gpu-ids '${gpu_ids}' must list exactly tp*dp=${n_gpus} distinct GPUs" >&2; exit 2
  fi
  export CUDA_VISIBLE_DEVICES="${gpu_ids}"
fi
mkdir -p "${run_dir}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${script_dir}/../.." && pwd)"
# PDMUX_ROOT = execution host's work root (see sync_engine_tree.sh and
# CLAUDE.md "환경 / 실행"). Only builds *defaults* below; SGLANG_ENGINE_DEV
# and PDMUX_VENV still win outright when set, unchanged from before.
pdmux_root="${PDMUX_ROOT:-$(cd "${engine_root}/../../.." && pwd)}"
engine_dev="${SGLANG_ENGINE_DEV:-${pdmux_root}/sglang_engine_dev/python}"
venv_dir="${PDMUX_VENV:-${pdmux_root}/sglang_engine_venv}"
config="${PDMUX_R2_CONFIG:-${engine_root}/benchmarks/configs/pdmux_r2.yml}"
port=$((32000 + (${SLURM_JOB_ID:-1} + ${SLURM_ARRAY_TASK_ID:-0}) % 20000))

# The run record is the provenance of what was served, so `model`,
# `context_length` and `cuda_graph` are read FROM IT (campaign schema v2).  A v1
# record has none of the three; the `or ""` keeps those campaigns runnable by
# falling through to the env/literal defaults below.
readarray -t fields < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["trace_path"]); print(d["server_seed"]); print(d["max_running_requests"]); print(d["ttft_slo_ms"]); print(d["itl_slo_ms"]); print(d["workload"]); print(d.get("model") or ""); print("" if d.get("context_length") is None else d["context_length"]); print("" if d.get("cuda_graph") is None else int(bool(d["cuda_graph"])))' \
    "${run_record}"
)
# Same caveat as r2_eval.sbatch: a dead `python` inside the process
# substitution leaves `fields` short instead of failing the script.
if (( ${#fields[@]} != 9 )) || [[ -z "${fields[0]}" ]]; then
  echo "ERROR: could not read the run record ${run_record}" >&2
  exit 2
fi
trace_path="${fields[0]}"
server_seed="${fields[1]}"
max_running="${fields[2]}"
ttft_slo="${fields[3]}"
itl_slo="${fields[4]}"
workload="${fields[5]}"
record_model="${fields[6]}"
record_ctx="${fields[7]}"
record_cuda_graph="${fields[8]}"

# Precedence: environment (set by r2_eval.sbatch, which has already validated it
# against the record) > run record > literal default.  Written as two statements
# rather than one nested expansion so the literal stays a plain default
# expansion that tests/test_r2_eval_runner.py can compare against the copies in
# r2_eval.sbatch and generate_campaign.sh.
if [[ -z "${PDMUX_MODEL:-}" && -n "${record_model}" ]]; then
  PDMUX_MODEL="${record_model}"
fi
model="${PDMUX_MODEL:-Zyphra/Zamba2-2.7B}"
if [[ -z "${PDMUX_CONTEXT_LENGTH:-}" && -n "${record_ctx}" ]]; then
  PDMUX_CONTEXT_LENGTH="${record_ctx}"
fi
# `cuda_graph` used to be a field nothing read.  Make it decide the flags, and
# refuse rather than silently pick a winner when the environment disagrees: the
# campaign manifest declares the operating point, and cudagraph-ON vs -OFF is
# precisely the axis the canon calls the operating point.
if [[ -n "${record_cuda_graph}" ]]; then
  record_disable_cg=$((1 - record_cuda_graph))
  if [[ -n "${PDMUX_DISABLE_CUDA_GRAPH:-}" \
        && "${PDMUX_DISABLE_CUDA_GRAPH}" != "${record_disable_cg}" ]]; then
    echo "ERROR: PDMUX_DISABLE_CUDA_GRAPH=${PDMUX_DISABLE_CUDA_GRAPH} contradicts" \
      "the run record's cuda_graph=${record_cuda_graph} (${run_record})." >&2
    echo "       Regenerate the campaign with/without --no-cuda-graph instead." >&2
    exit 2
  fi
  PDMUX_DISABLE_CUDA_GRAPH="${record_disable_cg}"
fi

source "${venv_dir}/bin/activate"
export PYTHONPATH="${engine_root}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
export PDMUX_WORKLOAD_ID="${workload}"

# Effective cap: TP scales with N (48*N); DP and N=1 keep the record's value (per replica).
max_running_effective="${max_running}"
if (( tp_size > 1 )); then
  max_running_effective=$(( max_running * tp_size ))
fi

server_args=(
  --model-path "${model}"
  --trust-remote-code
  --dtype bfloat16
  # `triton` is the established R2 value and stays the default.  It is a knob
  # because SGLang hard-refuses NemotronHForCausalLM on triton (server_args.py:
  # "does not support triton attention backend, as the first layer might not be
  # an attention layer"), so a NemotronH campaign must set
  # PDMUX_ATTENTION_BACKEND=flashinfer.  Switching backends changes the kernel,
  # so arms measured under different values are not comparable -- this is a
  # measurement-design choice that belongs to whoever designs the campaign.
  --attention-backend "${PDMUX_ATTENTION_BACKEND:-triton}"
  --disable-radix-cache
  --mem-fraction-static "${PDMUX_MEM_FRACTION:-0.82}"
  --max-running-requests "${max_running_effective}"
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
if (( tp_size > 1 )); then
  server_args+=(--tp-size "${tp_size}")
fi
if (( dp_size > 1 )); then
  # round_robin is the registered routing (design section 4-3); shortest-queue-style methods
  # are a sensitivity arm and would need their own, explicit launcher change.
  server_args+=(--dp-size "${dp_size}" --load-balance-method round_robin)
fi

# Provenance: the exact tuple this boot was launched with, one argument per
# line, written BEFORE the launch so it exists even if the server dies during
# boot.  results/r2_correctness/r2_correctness.sbatch relies on the same kind of
# dump to prove a CLI-only perturbation (R2C_NUM_KV_SPLITS) was really applied;
# without it, "which context length did that run use" is unanswerable after the
# fact.  PDMUX_DRY_RUN=1 stops here, so the tuple can be inspected on a login
# node with no GPU and no model load.
printf '%s\n' "${server_args[@]}" >"${run_dir}/server_args.txt"
# Communication env: record "unset" by default (a value is a PREREG decision; a PREREG can pin
# values through PDMUX_COMM_ENV_PIN="K=V,K=V", exported here so record and run cannot differ).
if [[ -n "${PDMUX_COMM_ENV_PIN:-}" ]]; then
  while IFS='=' read -r pin_k pin_v; do
    [[ -n "${pin_k}" ]] && export "${pin_k}=${pin_v}"
  done < <(tr ',' '\n' <<<"${PDMUX_COMM_ENV_PIN}")
fi
jit_flag=()
[[ "${PDMUX_ALLOW_JIT_ALL_REDUCE:-0}" == 1 ]] && jit_flag=(--allow-jit-all-reduce)
if (( n_gpus > 1 )); then
  python "${script_dir}/multigpu_cell_meta.py" comm-env --out "${run_dir}/comm_env.json" \
    ${jit_flag[@]+"${jit_flag[@]}"} || exit $?
else
  python "${script_dir}/multigpu_cell_meta.py" comm-env --out "${run_dir}/comm_env.json" \
    --allow-jit-all-reduce || true
fi
if [[ "${PDMUX_DRY_RUN:-0}" == 1 ]]; then
  printf '%s\n' "${server_args[@]}"
  exit 0
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
# KV / mamba capacity the server actually got (audit C-5) -- recorded for every cell.
curl -fsS "http://127.0.0.1:${port}/server_info" >"${run_dir}/server_info.json" 2>/dev/null || true
python "${script_dir}/multigpu_cell_meta.py" server-facts --log "${run_dir}/server.log" \
  --server-info "${run_dir}/server_info.json" --out "${run_dir}/cell_meta.json" \
  --tp "${tp_size}" --dp "${dp_size}" --max-running-requested "${max_running_effective}" \
  --gpu-ids "${gpu_ids}" || true

python -m pdmux_eval.trace_loadgen \
  --endpoint "http://127.0.0.1:${port}/generate" \
  --trace "${trace_path}" \
  --output "${run_dir}/requests.jsonl" \
  --summary "${run_dir}/summary.json" \
  --ttft-slo-ms "${ttft_slo}" \
  --itl-slo-ms "${itl_slo}" \
  ${PDMUX_CLIENT_CONNECTOR_LIMIT:+--connector-limit "${PDMUX_CLIENT_CONNECTOR_LIMIT}"}
