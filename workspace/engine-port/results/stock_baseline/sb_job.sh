#!/bin/bash
# Job target of PREREG_STOCK_BASE rev2.1 (SEALED 2026-11-02): J1, J1-ext, J1b, J2, J3.  Refuses unless PDMUX_SB_SEALED=1 or
# PDMUX_SB_TEST_MODE=1 (CPU tests).  Nothing is launched by this file being committed.
#
#   launch.sh --campaign stock_<job> --target workspace/engine-port/results/stock_baseline/sb_job.sh \
#             --env PDMUX_SB_JOB=J2 --env PDMUX_SB_ADDENDUM=/data/stock/addendum_<sha>.json --env PDMUX_REQUIRE_PRISTINE=1
#
# J1 needs no addendum; J1b needs lambda_star_burst + refcell_rate (after J1); J2/J3 need lambda_star_ref (after J1b).
# Image: every arm runs in ONE image, PDMUX_VESSL_IMAGE_SBASE; VL-stock boots /opt/vllm-venv/bin/python inside it.  Select it at
# submit time without changing launch.sh: PDMUX_VESSL_ENV=scripts/vessl/vessl.sbase.env scripts/vessl/launch.sh ...
# sb_runner refuses any boot whose image pip-freeze sha256 (SGLang /opt/pdmux, vLLM /opt/vllm-venv) differs from the registered values.
# rc: 0 ok, 2 setup, 13 ABORT_ENGINE_DIGEST/ABORT_HARNESS_DIGEST/ABORT_DATASET, 15 ABORT_PRISTINE, 16 config/unsealed/substrate,
#     17 boot failures, 18 raw export, 124/137 hard cap.  Nothing here prints the environment.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../.." && pwd)"
job="${PDMUX_SB_JOB:-}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
raw="${PDMUX_SB_OUT:-${data}/runs/${name}/raw}"
summary="${PDMUX_SB_SUMMARY:-${here}/runs/${name}}"
w3tok="${PDMUX_SB_W3_TOKENS:-${data}/datasets/arxsum_x/arxsum_x_tokens_v1.bin}"
w3sha="4c32f1af8eff07d7d0ad39006423163895181381c0d83a73a5f0f5b1ce76888f"   # results/multigpu/mg_const.py X_TOKENS_SHA256
expected_digest="2836112bd710a712a26a31fb516a685a25c6d893308b2058bb7b5590deda5ed7"

case "${job}" in
  J1|J1-ext|J1b|J2|J3) ;;
  *) echo "ABORT_CONFIG: PDMUX_SB_JOB must be J1|J1-ext|J1b|J2|J3; got '${job}'" >&2; exit 16 ;;
esac
if [[ "${PDMUX_SB_SEALED:-}" != 1 && "${PDMUX_SB_TEST_MODE:-}" != 1 ]]; then
  echo "ABORT_UNSEALED: PDMUX_SB_SEALED=1 not set (PREREG_STOCK_BASE rev2.1 is sealed by HARNESS_DIGEST.txt)" >&2; exit 16
fi
if [[ "${PDMUX_SB_SEALED:-}" == 1 ]] && { [[ -n "${PDMUX_SB_ENGINE_DIGEST_FILE:-}" ]] || [[ -n "${PDMUX_SB_HARNESS_DIGEST:-}" ]] || [[ -n "${PDMUX_SB_MANIFEST:-}" ]]; }; then
  echo "ABORT_SEALED_OVERRIDE: digest/manifest paths cannot be overridden when PDMUX_SB_SEALED=1" >&2; exit 16
fi
edigest="${PDMUX_SB_ENGINE_DIGEST_FILE:-${here}/ENGINE_DIGEST.txt}"
if [[ ! -f "${edigest}" ]] || [[ "$(tr -d ' \n' < "${edigest}")" != "${expected_digest}" ]]; then
  echo "ABORT_ENGINE_DIGEST: ${edigest} is not the registered engine digest ${expected_digest:0:12}..." >&2; exit 13
fi
if [[ "${PDMUX_SB_SEALED:-}" == 1 ]]; then
  python3 "${here}/sb_digest.py" check --file "${PDMUX_SB_HARNESS_DIGEST:-${here}/HARNESS_DIGEST.txt}" \
    || { echo "ABORT_HARNESS_DIGEST: sealed harness bytes differ" >&2; exit 13; }
fi
add_args=()
if [[ -n "${PDMUX_SB_ADDENDUM:-}" ]]; then add_args=(--addendum "${PDMUX_SB_ADDENDUM}"); fi
plan_json="$(python3 "${here}/sb_runner.py" plan --job "${job}" "${add_args[@]}" 2>/dev/null)" \
  || { echo "ABORT_PLAN: plan for ${job} could not be built (addendum missing?)" >&2; exit 16; }
cap_min="$(printf '%s' "${plan_json}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert not d["problems"], d["problems"]; print(d["hard_cap_min"])')" \
  || { echo "ABORT_PLAN: plan problems" >&2; exit 16; }
mkdir -p "${raw}" "${summary}" || { echo "ERROR: cannot create ${raw} / ${summary}" >&2; exit 2; }
printf '%s\n' "${plan_json}" > "${summary}/plan_summary.json"

if [[ "${PDMUX_SB_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
  echo "gpu_count=${ngpu}" > "${summary}/preflight.txt"
  [[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, stock-baseline Jobs need exactly 1" >&2; exit 16; }
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader > "${summary}/gpu.txt" 2>/dev/null || true
  # rev1: one combined image for every arm (VL-stock in /opt/vllm-venv); refuse a Job launched on any other image up front
  echo "image=${PDMUX_IMAGE:-}" >> "${summary}/preflight.txt"
  python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import sb_spec as S; sys.exit(0 if S.image_is_sbase(sys.argv[2]) else 1)' \
    "${here}" "${PDMUX_IMAGE:-}" || { echo "ABORT_SUBSTRATE: image '${PDMUX_IMAGE:-}' is not PDMUX_VESSL_IMAGE_SBASE" >&2; exit 16; }
  manifest="${PDMUX_SB_MANIFEST:-${data}/runs/${name}/meta/runtime_source_manifest.sha256}"
  python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed-file "${edigest}" \
    --out "${summary}/ENGINE_DIGEST_CHECK.json" > "${summary}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
    || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
  # pristine tree (PREREG §1): the SG-stock/rep/mix boots need the preserved copy; refuse up front instead of 3 dead boots
  pst="$(dirname "$(dirname "${SGLANG_ENGINE_DEV:-${PDMUX_ROOT:-/opt/pdmux}/sglang_engine_dev/python}")")/sglang_pristine/PRESERVE_STATUS"
  cp "${pst}" "${summary}/PRESERVE_STATUS" 2>/dev/null || true
  grep -qx 'status=OK' "${pst}" 2>/dev/null || { echo "ABORT_PRISTINE: ${pst} missing or not OK (set PDMUX_REQUIRE_PRISTINE=1)" >&2; exit 15; }
  sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
  [[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing (stage_data.sh)" >&2; exit 13; }
  if [[ "${job}" == J3 ]]; then
    [[ "$(sha256sum "${w3tok}" 2>/dev/null | cut -d' ' -f1)" == "${w3sha}" ]] \
      || { echo "ABORT_DATASET: ${w3tok} missing or sha256 != ${w3sha}" >&2; exit 13; }
  fi
  tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
  sha256sum "${sg}" > "${summary}/sharegpt.sha256"
  python3 "${eng}/results/cap_char/policy_harness/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${raw}/w1_lengths.json" \
    > "${summary}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }
else
  [[ -f "${PDMUX_SB_W1_LENGTHS:-}" ]] && cp "${PDMUX_SB_W1_LENGTHS}" "${raw}/w1_lengths.json"
fi

args=(run --job "${job}" --out "${raw}" --w1-lengths "${raw}/w1_lengths.json" "${add_args[@]}")
[[ "${job}" == J3 ]] && args+=(--w3-tokens "${w3tok}")
[[ -n "${PDMUX_SB_ADDENDUM:-}" ]] && sha256sum "${PDMUX_SB_ADDENDUM}" > "${summary}/addendum.sha256"
[[ "${PDMUX_SB_RESUME:-}" == 1 ]] && args+=(--resume)

timeout -k 60 "$(( cap_min * 60 ))" python3 "${here}/sb_runner.py" "${args[@]}"
rc=$?
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}, \"cap_min\": ${cap_min}}" > "${summary}/HARDCAP_KILLED.json"
[[ -f "${data}/runs/${name}/meta/clk.csv" ]] && { mkdir -p "${raw}/clk"; cp "${data}/runs/${name}/meta/clk.csv" "${raw}/clk/${name}.csv" 2>/dev/null || true; }
cp "${raw}/RUN_DONE.json" "${raw}/plan.json" "${summary}/" 2>/dev/null || true
(cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS) && cp "${raw}/SHA256SUMS" "${summary}/RAW_SHA256SUMS"
if [[ "${raw}" != "${summary}"/* && "${raw}" != "${summary}" ]]; then
  if mkdir -p "${summary}/raw" && cp -a "${raw}/." "${summary}/raw/" && (cd "${summary}/raw" && sha256sum --quiet -c SHA256SUMS) \
     && [[ "$(find "${raw}" -type f | wc -l)" == "$(find "${summary}/raw" -type f | wc -l)" ]]; then
    echo "raw exported to ${summary}/raw ($(find "${summary}/raw" -type f | wc -l) files, sha256 verified)" > "${summary}/RAW_EXPORT_OK"
  else
    echo "ABORT_RAW_EXPORT: could not export/verify ${raw} -> ${summary}/raw" >&2
    [[ "${rc}" -eq 0 ]] && rc=18
  fi
fi
echo "stock ${job}: rc=${rc} raw=${raw}"
exit "${rc}"
