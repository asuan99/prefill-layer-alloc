#!/bin/bash
# VESSL Job target of the LooGLE interference cell over static D arms (2026-11-12).  Nothing is launched by this file being committed; it
# has not run on a GPU.  One Job = one full block (every arm x with / without LooGLE x 3 short rates x 4 seeds) = one evaluation unit.
# The measurement is two such Jobs (block 1 and block 2, disjoint registered seeds); they are combined by mo_intf_analyze.py combine.
#
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sg312.env scripts/vessl/launch.sh --campaign motivation_interference \
#     --target workspace/engine-port/results/motivation/mo_intf_job.sh \
#     --env PDMUX_MOINTF_MODE=rehearse --env PDMUX_MOINTF_JOB=<job> --env PDMUX_MOINTF_RATES=<r1> --env PDMUX_MOINTF_BUDGET_MIN=<min>
#   ... --env PDMUX_MOINTF_MODE=run --env PDMUX_MOINTF_JOB=<job> --env PDMUX_MOINTF_BLOCK=1|2 --env PDMUX_MOINTF_RATES=<r1,r2,r3> \
#       --env PDMUX_MOINTF_BUDGET_MIN=<min> [--env PDMUX_MOINTF_EXPECT_CARD_SHA=<sha>]
# (launch.sh passes only PDMUX_* names.)  The three short rates come from the pre-ladder; there is no default.
#
# MODE=rehearse: full pass over the real arm list (D16 / D24 / D34 / D44 + baseline, the four split-check boots, anchors) with one
#   with / without pair of 20 s cells; writes the rehearsal record of this kind, which the launch gate of MODE=run requires.
# MODE=run: plan -> card (refused if PDMUX_MOINTF_EXPECT_CARD_SHA is set and differs) -> run (the runner refuses placeholders in the card
#   -- mem / cap come from the cap measurement -- and a plan without a passed rehearsal record) -> block analysis.
# Preflight (rc != 0 stops the Job before any boot): exactly one GPU, engine digest 8d8f6720, pristine tree preserved, ShareGPT source
#   present (warm-up pool); the LooGLE token file is resolved by the runner (sha256 = registered) before the first boot.
# rc: 0 ok, 3 analysis found the Job invalid, 13 digest/dataset, 15 pristine, 16 config/substrate, 17 boot failures, 19 abort,
#     124/137 budget cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../.." && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
mode="${PDMUX_MOINTF_MODE:-}"
job="${PDMUX_MOINTF_JOB:-}"
block="${PDMUX_MOINTF_BLOCK:-}"
rates="${PDMUX_MOINTF_RATES:-}"
budget="${PDMUX_MOINTF_BUDGET_MIN:-}"
expected_digest="8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"
out="${PDMUX_MOINTF_OUT:-${eng}/results/motivation_interference_runs/${name}}"
raw="${out}/raw"
export PDMUX_HEARTBEAT_S=0        # measurement Job: no heartbeat (patched boots get the same value from the runner's server env)

case "${mode}" in rehearse|run) ;; *) echo "ABORT_CONFIG: PDMUX_MOINTF_MODE must be rehearse|run; got '${mode}'" >&2; exit 16 ;; esac
[[ -n "${job}" && -n "${rates}" && "${budget}" =~ ^[0-9]+$ ]] || { echo "ABORT_CONFIG: PDMUX_MOINTF_JOB, PDMUX_MOINTF_RATES, PDMUX_MOINTF_BUDGET_MIN (minutes) required" >&2; exit 16; }
if [[ "${mode}" == run ]]; then
  [[ "${block}" =~ ^[12]$ ]] || { echo "ABORT_CONFIG: PDMUX_MOINTF_BLOCK (1|2) required: one Job per block" >&2; exit 16; }
  [[ "${rates}" =~ ^[0-9.]+,[0-9.]+,[0-9.]+$ ]] || { echo "ABORT_CONFIG: PDMUX_MOINTF_RATES must be three short rates r1,r2,r3" >&2; exit 16; }
fi
mkdir -p "${out}" || { echo "ERROR: cannot create ${out}" >&2; exit 2; }

if [[ "${PDMUX_MOINTF_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  { echo "gpu_count=${ngpu}"; echo "image=${PDMUX_IMAGE:-}"; echo "mode=${mode}"; echo "block=${block}"; echo "rates=${rates}"; } > "${out}/preflight.txt"
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
  cp "${PDMUX_MOINTF_W1_LENGTHS:?}" "${out}/w1_lengths.json"
fi

common=(--job "${job}" --rates "${rates}" --w1-lengths "${out}/w1_lengths.json")
[[ -n "${block}" ]] && common+=(--block "${block}")
if [[ "${mode}" == rehearse ]]; then
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_intf_runner.py" rehearse "${common[@]}" --out "${raw}" > "${out}/rehearse.json"
  rc=$?
else
  python3 "${here}/mo_intf_runner.py" card "${common[@]}" --out "${out}/card.json" > "${out}/card_summary.json" \
    || { echo "ABORT_CARD: card build failed" >&2; exit 16; }
  if [[ -n "${PDMUX_MOINTF_EXPECT_CARD_SHA:-}" ]]; then
    got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["card_sha"])' "${out}/card.json")"
    [[ "${got}" == "${PDMUX_MOINTF_EXPECT_CARD_SHA}" ]] || { echo "ABORT_CARD: card sha ${got:0:16} != registered ${PDMUX_MOINTF_EXPECT_CARD_SHA:0:16}" >&2; exit 16; }
  fi
  args=(run "${common[@]}" --card "${out}/card.json" --out "${raw}")
  [[ "${PDMUX_MOINTF_RESUME:-}" == 1 ]] && args+=(--resume)
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_intf_runner.py" "${args[@]}"
  rc=$?
  if [[ -f "${raw}/plan.json" ]]; then
    clk="${data}/runs/${name}/meta/clk.csv"
    an=(python3 "${here}/mo_intf_analyze.py" block "${raw}" --out "${out}/analysis")
    [[ -f "${clk}" ]] && an+=(--clk "${clk}")
    "${an[@]}" > "${out}/analysis.txt" 2>&1
    arc=$?
    [[ ${rc} -eq 0 && ${arc} -ne 0 ]] && rc=${arc}
  fi
fi
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"BUDGET_CAP_KILLED\", \"rc\": ${rc}, \"budget_min\": ${budget}}" > "${out}/BUDGET_CAP_KILLED.json"
[[ -d "${raw}" ]] && (cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "motivation interference ${mode} block=${block}: rc=${rc} out=${out}"
exit "${rc}"
