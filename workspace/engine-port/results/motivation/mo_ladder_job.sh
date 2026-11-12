#!/bin/bash
# VESSL Job target of the pre-ladder per workload (2026-11-12).  NOT A MEASUREMENT: a short rate search whose result only chooses the
# rate grids of the later measurements; provisional cap 48 / mem 0.6 (stated in every arm name), no anchor, no rehearsal gate.
# Nothing is launched by this file being committed; it has not run on a GPU.
#
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sg312.env scripts/vessl/launch.sh --campaign motivation_pre_ladder \
#     --target workspace/engine-port/results/motivation/mo_ladder_job.sh \
#     --env PDMUX_MOLAD_JOB=<job> --env PDMUX_MOLAD_ARMS=<key1,key2,...> \
#     --env PDMUX_MOLAD_GRIDS=<target:rmin[:factor:n],...> --env PDMUX_MOLAD_BUDGET_MIN=<min> [--env PDMUX_MOLAD_STAGE1_N=<n>]
# (launch.sh passes only PDMUX_* names.)  Targets: sharegpt_first, openchat_rounds, arxiv_summ, interference_short (the sharegpt_first
# stream at context 96,000).  The lowest rate of a target and the arm list have no default (mo_ladder.py options prints the Job time of
# every arm choice).
# Preflight (rc != 0 stops the Job before any boot): exactly one GPU, engine digest 8d8f6720, pristine tree preserved, ShareGPT source
#   present; the token files of the Job's workloads are resolved by the runner (sha256 = registered) before the first boot.
# rc: 0 ok, 3 analysis found the Job invalid, 13 digest/dataset, 15 pristine, 16 config/substrate, 17 boot failures, 19 abort,
#     124/137 budget cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../.." && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
job="${PDMUX_MOLAD_JOB:-}"
arms="${PDMUX_MOLAD_ARMS:-}"
grids="${PDMUX_MOLAD_GRIDS:-}"
budget="${PDMUX_MOLAD_BUDGET_MIN:-}"
expected_digest="8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"
out="${PDMUX_MOLAD_OUT:-${eng}/results/motivation_pre_ladder_runs/${name}}"
raw="${out}/raw"
export PDMUX_HEARTBEAT_S=0

[[ -n "${job}" && -n "${arms}" && -n "${grids}" && "${budget}" =~ ^[0-9]+$ ]] || { echo "ABORT_CONFIG: PDMUX_MOLAD_JOB, PDMUX_MOLAD_ARMS, PDMUX_MOLAD_GRIDS, PDMUX_MOLAD_BUDGET_MIN (minutes) required" >&2; exit 16; }
mkdir -p "${out}" || { echo "ERROR: cannot create ${out}" >&2; exit 2; }

if [[ "${PDMUX_MOLAD_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  { echo "gpu_count=${ngpu}"; echo "image=${PDMUX_IMAGE:-}"; echo "arms=${arms}"; echo "grids=${grids}"; echo "NOT_A_MEASUREMENT"; } > "${out}/preflight.txt"
  [[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, this Job needs exactly 1" >&2; exit 16; }
  python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${data}/runs/${name}/meta/runtime_source_manifest.sha256" \
    --sealed "${expected_digest}" --out "${out}/ENGINE_DIGEST_CHECK.json" > "${out}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
    || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
  pst="$(dirname "$(dirname "${SGLANG_ENGINE_DEV:-${PDMUX_ROOT:-/opt/pdmux}/sglang_engine_dev/python}")")/sglang_pristine/PRESERVE_STATUS"
  cp "${pst}" "${out}/PRESERVE_STATUS" 2>/dev/null || true
  grep -qx 'status=OK' "${pst}" 2>/dev/null || { echo "ABORT_PRISTINE: ${pst} missing or not OK" >&2; exit 15; }
  sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
  tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
  [[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing" >&2; exit 13; }
  sha256sum "${sg}" > "${out}/sharegpt.sha256"
  python3 "${eng}/results/cap_char/policy_harness/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${out}/w1_lengths.json" \
    > "${out}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }
else
  cp "${PDMUX_MOLAD_W1_LENGTHS:?}" "${out}/w1_lengths.json"
fi

common=(--job "${job}" --arms "${arms}" --grids "${grids}" --w1-lengths "${out}/w1_lengths.json")
python3 "${here}/mo_ladder.py" card "${common[@]}" --out "${out}/card.json" > "${out}/card_summary.json" \
  || { echo "ABORT_CARD: card build failed (unknown target / grid?)" >&2; exit 16; }
args=(run "${common[@]}" --card "${out}/card.json" --out "${raw}")
[[ "${PDMUX_MOLAD_RESUME:-}" == 1 ]] && args+=(--resume)
timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_ladder.py" "${args[@]}"
rc=$?
if [[ -f "${raw}/plan.json" ]]; then
  an=(python3 "${here}/mo_ladder_analyze.py" "${raw}" --out "${out}/analysis")
  [[ -n "${PDMUX_MOLAD_STAGE1_N:-}" ]] && an+=(--stage1-n "${PDMUX_MOLAD_STAGE1_N}")
  "${an[@]}" > "${out}/analysis.txt" 2>&1
  arc=$?
  [[ ${rc} -eq 0 && ${arc} -ne 0 ]] && rc=${arc}
fi
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"BUDGET_CAP_KILLED\", \"rc\": ${rc}, \"budget_min\": ${budget}}" > "${out}/BUDGET_CAP_KILLED.json"
[[ -d "${raw}" ]] && (cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "motivation pre-ladder (NOT A MEASUREMENT): rc=${rc} out=${out}"
exit "${rc}"
