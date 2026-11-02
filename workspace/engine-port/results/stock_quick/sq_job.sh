#!/bin/bash
# Job target: quick default-engine SLO-attainment ladders, see sq_plan.py.  Descriptive, not PREREG STOCK-BASE.
#   PDMUX_VESSL_ENV=scripts/vessl/vessl.sbase.env scripts/vessl/launch.sh --campaign stock_quick \
#     --target workspace/engine-port/results/stock_quick/sq_job.sh --env PDMUX_SQ_ARMS=SG-stock --env PDMUX_REQUIRE_PRISTINE=1
# Preflight mirrors stock_baseline/sb_job.sh (same image, engine digest, pristine tree, datasets).
# rc: 0 ok, 2 setup, 13 digest/dataset, 15 pristine, 16 config/substrate, 17 boot failures, 18 raw export, 124/137 hard cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sb="${here}/../stock_baseline"
eng="$(cd "${here}/../.." && pwd)"
arms="${PDMUX_SQ_ARMS:-SG-stock}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
raw="${data}/runs/${name}/raw"
summary="${here}/runs/${name}"
w3tok="${data}/datasets/arxsum_x/arxsum_x_tokens_v1.bin"
w3sha="4c32f1af8eff07d7d0ad39006423163895181381c0d83a73a5f0f5b1ce76888f"
edigest="${sb}/ENGINE_DIGEST.txt"

plan_json="$(python3 "${here}/sq_plan.py" plan --arms "${arms}")" || { echo "ABORT_PLAN: ${plan_json}" >&2; exit 16; }
cap_min="$(printf '%s' "${plan_json}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hard_cap_min"])')"
mkdir -p "${raw}" "${summary}" || { echo "ERROR: cannot create ${raw} / ${summary}" >&2; exit 2; }
printf '%s\n' "${plan_json}" > "${summary}/plan_summary.json"

ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
[[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, need exactly 1" >&2; exit 16; }
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader > "${summary}/gpu.txt" 2>/dev/null || true
python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import sb_spec as S; sys.exit(0 if S.image_is_sbase(sys.argv[2]) else 1)' \
  "${sb}" "${PDMUX_IMAGE:-}" || { echo "ABORT_SUBSTRATE: image '${PDMUX_IMAGE:-}' is not PDMUX_VESSL_IMAGE_SBASE" >&2; exit 16; }
manifest="${data}/runs/${name}/meta/runtime_source_manifest.sha256"
python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed-file "${edigest}" \
  --out "${summary}/ENGINE_DIGEST_CHECK.json" > "${summary}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
  || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
pst="$(dirname "$(dirname "${SGLANG_ENGINE_DEV:-${PDMUX_ROOT:-/opt/pdmux}/sglang_engine_dev/python}")")/sglang_pristine/PRESERVE_STATUS"
cp "${pst}" "${summary}/PRESERVE_STATUS" 2>/dev/null || true
grep -qx 'status=OK' "${pst}" 2>/dev/null || { echo "ABORT_PRISTINE: ${pst} missing or not OK (set PDMUX_REQUIRE_PRISTINE=1)" >&2; exit 15; }
sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
[[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing" >&2; exit 13; }
[[ "$(sha256sum "${w3tok}" 2>/dev/null | cut -d' ' -f1)" == "${w3sha}" ]] || { echo "ABORT_DATASET: ${w3tok} missing or sha256 != ${w3sha}" >&2; exit 13; }
tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
python3 "${eng}/results/cap_char/policy_harness/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${raw}/w1_lengths.json" \
  > "${summary}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }

timeout -k 60 "$(( cap_min * 60 ))" python3 "${here}/sq_plan.py" run --arms "${arms}" --out "${raw}" --w1-lengths "${raw}/w1_lengths.json" --w3-tokens "${w3tok}"
rc=$?
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}, \"cap_min\": ${cap_min}}" > "${summary}/HARDCAP_KILLED.json"
cp "${raw}/RUN_DONE.json" "${raw}/plan.json" "${summary}/" 2>/dev/null || true
(cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS) && cp "${raw}/SHA256SUMS" "${summary}/RAW_SHA256SUMS"
if mkdir -p "${summary}/raw" && cp -a "${raw}/." "${summary}/raw/" && (cd "${summary}/raw" && sha256sum --quiet -c SHA256SUMS); then
  echo "raw exported to ${summary}/raw" > "${summary}/RAW_EXPORT_OK"
else
  echo "ABORT_RAW_EXPORT" >&2; [[ "${rc}" -eq 0 ]] && rc=18
fi
echo "stock_quick: rc=${rc} raw=${raw}"
exit "${rc}"
