#!/bin/bash
# Translate a campaign run record into an engine policy environment, then call
# the pinned server+loadgen implementation.
set -euo pipefail

run_record="${1:?run record required}"
run_dir="${2:?run directory required}"
readarray -t fields < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["baseline"]); print(d["architecture"]); print(d["policy"]); print(d["decode_sms"]); print(int(d["requires_offline_oracle"])); print(d["run_id"]); print(d["workload"])' \
    "${run_record}"
)
baseline="${fields[0]}"
architecture="${fields[1]}"
policy="${fields[2]}"
decode_sms="${fields[3]}"
requires_oracle="${fields[4]}"
export PDMUX_RUN_ID="${fields[5]}"
export PDMUX_WORKLOAD_ID="${fields[6]}"

unset PDMUX_DUAL_WORKER PDMUX_TRUE_DUAL_WORKER PDMUX_R2_POLICY
unset PDMUX_SLO_SCHED PDMUX_MODEL_PROFILE PDMUX_R2_FIXED_DSM
unset PDMUX_LAYER_GRANULAR_BASELINE

if [[ "${requires_oracle}" == 1 ]]; then
  echo "ERROR: ${baseline} requires a completed offline sweep/oracle schedule." >&2
  exit 2
fi

case "${architecture}:${policy}" in
  vanilla:vanilla)
    export PDMUX_ENGINE_MODE=vanilla
    ;;
  legacy:fixed)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_R2_POLICY=fixed
    export PDMUX_R2_FIXED_DSM="${decode_sms}"
    ;;
  legacy:single_worker_dynamic)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_SLO_SCHED=1
    ;;
  true_dual:fixed)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_TRUE_DUAL_WORKER=1
    export PDMUX_R2_POLICY=fixed
    export PDMUX_R2_FIXED_DSM="${decode_sms}"
    ;;
  true_dual:generic)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_TRUE_DUAL_WORKER=1
    export PDMUX_R2_POLICY=generic
    ;;
  true_dual:hybrid)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_TRUE_DUAL_WORKER=1
    export PDMUX_R2_POLICY=hybrid
    export PDMUX_MODEL_PROFILE="${PDMUX_MODEL_PROFILE_PATH:?set PDMUX_MODEL_PROFILE_PATH}"
    ;;
  legacy:layer_granular)
    export PDMUX_ENGINE_MODE=pdmux
    export PDMUX_LAYER_GRANULAR_BASELINE=1
    export PDMUX_LA_COORD=1
    ;;
  *)
    echo "ERROR: unsupported campaign arm ${architecture}:${policy}" >&2
    exit 2
    ;;
esac

export PDMUX_BASELINE="${baseline}"
export PDMUX_RUN_RECORD="${run_record}"
export PDMUX_RUN_DIR="${run_dir}"
exec "${PDMUX_ENGINE_BENCH_RUNNER:?set PDMUX_ENGINE_BENCH_RUNNER}" \
  "${run_record}" "${run_dir}"
