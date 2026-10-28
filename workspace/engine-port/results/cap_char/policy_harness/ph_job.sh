#!/bin/bash
# Job target of the policy PREREG (rev0.1): CAP1, CAP2, TUNE, EVAL-a, EVAL-b, HD-1..HD-8.  NOT SEALED: refuses unless
# PDMUX_POLICY_SEALED=1 (after the seal) or PDMUX_POLICY_TEST_MODE=1 (CPU tests).  The harness is not launched by this commit.
#
#   launch.sh --campaign policy_<job> --target workspace/engine-port/results/cap_char/policy_harness/ph_job.sh \
#             --env PDMUX_POLICY_JOB=HD-3 [--env PDMUX_POLICY_ADDENDUM=/data/policy/addendum_<sha>.json] [--env PDMUX_POLICY_RESUME=1]
#
# CAP1/CAP2 need an addendum with cap_ladder.{W1,W3}.rmin only; TUNE/EVAL/HD need the post-capacity addendum
# (lambda_star, W3_floor_ms, tuned_config).  rc: 0 ok, 2 setup, 13 ABORT_ENGINE_DIGEST/ABORT_HARNESS_DIGEST/ABORT_DATASET,
# 16 config/unsealed/substrate, 124/137 hard cap.  Nothing here prints the environment.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../../.." && pwd)"
ct="${here}/../costterms_harness"
job="${PDMUX_POLICY_JOB:-}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
raw="${PDMUX_POLICY_OUT:-${data}/runs/${name}/raw}"
summary="${PDMUX_POLICY_SUMMARY:-${here}/../policy_runs/${name}}"
w3tok="${PDMUX_POLICY_W3_TOKENS:-${data}/datasets/arxsum_x/arxsum_x_tokens_v1.bin}"
w3sha="4c32f1af8eff07d7d0ad39006423163895181381c0d83a73a5f0f5b1ce76888f"   # results/multigpu/mg_const.py X_TOKENS_SHA256

case "${job}" in           # hard caps (min) = ph_spec.HARD_CAP_MIN
  CAP1) cap_min=196 ;; CAP2) cap_min=272 ;; TUNE) cap_min=247 ;; EVAL-a|EVAL-b) cap_min=252 ;;
  HD-[1-8]) cap_min=312 ;;
  *) echo "ABORT_CONFIG: PDMUX_POLICY_JOB must be CAP1|CAP2|TUNE|EVAL-a|EVAL-b|HD-1..8; got '${job}'" >&2; exit 16 ;;
esac
if [[ "${PDMUX_POLICY_SEALED:-}" != 1 && "${PDMUX_POLICY_TEST_MODE:-}" != 1 ]]; then
  echo "ABORT_UNSEALED: policy harness not sealed (PREREG_HD_HE0cell rev0.1)" >&2; exit 16
fi
# D1: the engine digest of the FINAL tree (shared with the diag Job) lives in this directory; a placeholder must never run.
edigest="${PDMUX_POLICY_ENGINE_DIGEST_FILE:-${here}/ENGINE_DIGEST.txt}"
if [[ "${PDMUX_POLICY_TEST_MODE:-}" != 1 ]] && { [[ ! -f "${edigest}" ]] || [[ "$(tr -d ' \n' < "${edigest}")" == "PENDING_FINAL_TREE" ]]; }; then
  echo "ABORT_ENGINE_DIGEST_PENDING: ${edigest} is a placeholder (final engine tree digest not delivered) -- refusing to run" >&2; exit 13
fi
if [[ "${PDMUX_POLICY_SEALED:-}" == 1 ]]; then
  python3 "${here}/ph_digest.py" check --file "${PDMUX_POLICY_HARNESS_DIGEST:-${here}/HARNESS_DIGEST.txt}" \
    || { echo "ABORT_HARNESS_DIGEST: sealed harness bytes differ" >&2; exit 13; }
fi
mkdir -p "${raw}" "${summary}" || { echo "ERROR: cannot create ${raw} / ${summary}" >&2; exit 2; }

if [[ "${PDMUX_POLICY_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
  echo "gpu_count=${ngpu}" > "${summary}/preflight.txt"
  [[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, policy Jobs need exactly 1" >&2; exit 16; }
  # engine digest (same sealed engine as the cost-term campaign)
  manifest="${PDMUX_POLICY_MANIFEST:-${data}/runs/${name}/meta/runtime_source_manifest.sha256}"
  python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed-file "${edigest}" \
    --out "${summary}/ENGINE_DIGEST_CHECK.json" > "${summary}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
    || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
  # datasets: ShareGPT json + arXiv-Sum token file (sha registered)
  sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
  [[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing (stage_data.sh)" >&2; exit 13; }
  if [[ "${job}" == CAP2 || "${job}" == HD-* ]]; then
    [[ "$(sha256sum "${w3tok}" 2>/dev/null | cut -d' ' -f1)" == "${w3sha}" ]] \
      || { echo "ABORT_DATASET: ${w3tok} missing or sha256 != ${w3sha} (see PREREG Q-2 staging procedure)" >&2; exit 13; }
  fi
  tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
  sha256sum "${sg}" > "${summary}/sharegpt.sha256"     # audit m8: pinned per Job; w1_lengths sha is checked against the addendum
  python3 "${here}/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${raw}/w1_lengths.json" \
    > "${summary}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }
else
  [[ -f "${PDMUX_POLICY_W1_LENGTHS:-}" ]] && cp "${PDMUX_POLICY_W1_LENGTHS}" "${raw}/w1_lengths.json"
fi

args=(run --job "${job}" --out "${raw}" --w1-lengths "${raw}/w1_lengths.json")
[[ "${job}" == CAP2 || "${job}" == HD-* ]] && args+=(--w3-tokens "${w3tok}")
[[ -n "${PDMUX_POLICY_ADDENDUM:-}" ]] && args+=(--addendum "${PDMUX_POLICY_ADDENDUM}") && sha256sum "${PDMUX_POLICY_ADDENDUM}" > "${summary}/addendum.sha256"
[[ "${PDMUX_POLICY_RESUME:-}" == 1 ]] && args+=(--resume)

timeout -k 60 "$(( cap_min * 60 ))" python3 "${here}/ph_runner.py" "${args[@]}"
rc=$?
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"HARDCAP_KILLED\", \"rc\": ${rc}}" > "${summary}/HARDCAP_KILLED.json"
[[ -f "${data}/runs/${name}/meta/clk.csv" ]] && { mkdir -p "${raw}/clk"; cp "${data}/runs/${name}/meta/clk.csv" "${raw}/clk/${name}.csv" 2>/dev/null || true; }
cp "${raw}/RUN_DONE.json" "${raw}/plan.json" "${summary}/" 2>/dev/null || true
(cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS) && cp "${raw}/SHA256SUMS" "${summary}/RAW_SHA256SUMS"
echo "policy ${job}: rc=${rc} raw=${raw}"
exit "${rc}"
