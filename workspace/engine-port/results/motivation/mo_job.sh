#!/bin/bash
# VESSL Job target of the motivation harness (2026-11-11).  One Job = one evaluation unit (mo_spec.make_plan: every arm x rung x seed,
# anchor first and last, one split-check boot).  Nothing is launched by this file being committed.
#
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sg312.env scripts/vessl/launch.sh --campaign motivation \
#     --target workspace/engine-port/results/motivation/mo_job.sh \
#     --env MO_MODE=rehearse --env MO_WORKLOAD=openchat_rounds --env MO_JOB=<job> --env MO_BUDGET_MIN=<min>
#   ... --env MO_MODE=run --env MO_WORKLOAD=<w> --env MO_JOB=<job> --env MO_RMIN=<r> --env MO_FACTOR=<f> --env MO_N=<rungs> \
#       --env MO_N_REQUESTS=<n> --env MO_BUDGET_MIN=<min> [--env MO_ROLE=extension] [--env MO_EXPECT_CARD_SHA=<sha>]
#
# MO_MODE=rehearse: full pass over the real arm list (one rung, few requests) on MO_WORKLOAD + the CPU trace/token check of every
#   registered workload; writes the rehearsal record (results/motivation/rehearsals/<digest16>.json) that the launch gate requires.
# MO_MODE=run: plan -> card (refused if MO_EXPECT_CARD_SHA is set and differs) -> run (the runner refuses W1, placeholders in the card,
#   and a plan without a passed rehearsal record for this harness digest) -> analysis with the Job's clock trace.
# Preflight (rc != 0 stops the Job before any boot): exactly one GPU, engine digest 8d8f6720 (dev_tree_edits #40), pristine tree
#   preserved, ShareGPT source present (warm-up / W1 pool), token files resolved by the runner (sha256 = registered).
# rc: 0 ok, 3 analysis found the Job invalid, 13 digest/dataset, 15 pristine, 16 config/substrate, 17 boot failures, 19 abort,
#     124/137 budget cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../.." && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
mode="${MO_MODE:-}"
wl="${MO_WORKLOAD:-}"
job="${MO_JOB:-}"
budget="${MO_BUDGET_MIN:-}"
expected_digest="8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"
out="${PDMUX_MO_OUT:-${eng}/results/motivation_runs/${name}}"
raw="${out}/raw"

case "${mode}" in rehearse|run) ;; *) echo "ABORT_CONFIG: MO_MODE must be rehearse|run; got '${mode}'" >&2; exit 16 ;; esac
[[ -n "${wl}" && -n "${job}" && "${budget}" =~ ^[0-9]+$ ]] || { echo "ABORT_CONFIG: MO_WORKLOAD, MO_JOB, MO_BUDGET_MIN (minutes) required" >&2; exit 16; }
if [[ "${mode}" == run ]]; then
  for v in MO_RMIN MO_FACTOR MO_N MO_N_REQUESTS; do [[ -n "${!v:-}" ]] || { echo "ABORT_CONFIG: ${v} required for MO_MODE=run" >&2; exit 16; }; done
fi
mkdir -p "${out}" || { echo "ERROR: cannot create ${out}" >&2; exit 2; }

if [[ "${PDMUX_MO_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  { echo "gpu_count=${ngpu}"; echo "image=${PDMUX_IMAGE:-}"; echo "mode=${mode}"; echo "workload=${wl}"; } > "${out}/preflight.txt"
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
  cp "${PDMUX_MO_W1_LENGTHS:?}" "${out}/w1_lengths.json"
fi

common=(--job "${job}" --workload "${wl}" --w1-lengths "${out}/w1_lengths.json")
if [[ "${mode}" == rehearse ]]; then
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_runner.py" rehearse "${common[@]}" --out "${raw}" > "${out}/rehearse.json"
  rc=$?
else
  grid=(--rmin "${MO_RMIN}" --factor "${MO_FACTOR}" --n "${MO_N}" --n-requests "${MO_N_REQUESTS}" --role "${MO_ROLE:-base}")
  python3 "${here}/mo_runner.py" card "${common[@]}" "${grid[@]}" --out "${out}/card.json" > "${out}/card_summary.json" \
    || { echo "ABORT_CARD: card build failed" >&2; exit 16; }
  if [[ -n "${MO_EXPECT_CARD_SHA:-}" ]]; then
    got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["card_sha"])' "${out}/card.json")"
    [[ "${got}" == "${MO_EXPECT_CARD_SHA}" ]] || { echo "ABORT_CARD: card sha ${got:0:16} != registered ${MO_EXPECT_CARD_SHA:0:16}" >&2; exit 16; }
  fi
  args=(run "${common[@]}" "${grid[@]}" --card "${out}/card.json" --out "${raw}")
  [[ "${MO_RESUME:-}" == 1 ]] && args+=(--resume)
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_runner.py" "${args[@]}"
  rc=$?
  if [[ -f "${raw}/plan.json" ]]; then
    clk="${data}/runs/${name}/meta/clk.csv"
    an=(python3 "${here}/mo_analyze.py" "${raw}" --out "${out}/analysis")
    [[ -f "${clk}" ]] && an+=(--clk "${clk}")
    "${an[@]}" > "${out}/analysis.txt" 2>&1
    arc=$?
    [[ ${rc} -eq 0 && ${arc} -ne 0 ]] && rc=${arc}
  fi
fi
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"BUDGET_CAP_KILLED\", \"rc\": ${rc}, \"budget_min\": ${budget}}" > "${out}/BUDGET_CAP_KILLED.json"
[[ -d "${raw}" ]] && (cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "motivation ${mode} ${wl}: rc=${rc} out=${out}"
exit "${rc}"
