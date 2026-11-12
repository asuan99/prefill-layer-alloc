#!/bin/bash
# VESSL Job target of the concurrency cap and decode memory traffic measurement (2026-11-12).  Nothing is launched by this file being
# committed; it has not run on a GPU.  One Job = one part (fixed batch OR serving cap/mem comparison) = one evaluation unit.
#
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sg312.env scripts/vessl/launch.sh --campaign motivation_cap \
#     --target workspace/engine-port/results/motivation/mo_cap_job.sh \
#     --env PDMUX_MOCAP_PART=fixed --env PDMUX_MOCAP_MODE=rehearse --env PDMUX_MOCAP_JOB=<job> --env PDMUX_MOCAP_BUDGET_MIN=<min>
#   ... --env PDMUX_MOCAP_PART=fixed   --env PDMUX_MOCAP_MODE=run --env PDMUX_MOCAP_JOB=<job> --env PDMUX_MOCAP_BUDGET_MIN=<min> \
#       --env PDMUX_MOCAP_ROUND=1|2      (one Job per round, user decision 2026-11-12; the two rounds are compared by mo_cap_analyze.py rounds)
#   ... --env PDMUX_MOCAP_PART=serving --env PDMUX_MOCAP_MODE=run --env PDMUX_MOCAP_JOB=<job> --env PDMUX_MOCAP_BUDGET_MIN=<min> \
#       --env PDMUX_MOCAP_WORKLOAD=<w> --env PDMUX_MOCAP_RMIN=<r> --env PDMUX_MOCAP_FACTOR=<f> [--env PDMUX_MOCAP_EXPECT_CARD_SHA=<sha>]
# (launch.sh passes only PDMUX_* names, hence the prefix.)
#
# PART=fixed  : context 2^7..2^18 x batch 1..256, memory-permitted cells, Nano + a31, mem 0.82, pristine tree, no instrumentation.
# PART=serving: mem {0.6, 0.75} at cap 48 + cap {48, 128, 256} at mem 0.75, 2 arms x 4 rungs x 4 seeds; the rate grid has no default.
# MODE=rehearse: full pass over the real boot list with one cell (fixed) / one rung and few requests (serving); writes the rehearsal
#   record of that part, which the launch gate of MODE=run requires.
# MODE=run: plan -> card (refused if PDMUX_MOCAP_EXPECT_CARD_SHA is set and differs) -> run (the runner refuses unconfirmed design values,
#   placeholders in the card, and a plan without a passed rehearsal record for this harness digest) -> analysis.
# Preflight (rc != 0 stops the Job before any boot): exactly one GPU, engine digest 8d8f6720, pristine tree preserved, Nano snapshot
#   present (a31 variant source), ShareGPT source present (serving warm-up / W1 pool).
# rc: 0 ok, 3 analysis found the Job invalid, 13 digest/dataset, 15 pristine, 16 config/substrate, 17 boot failures, 19 abort,
#     124/137 budget cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eng="$(cd "${here}/../.." && pwd)"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
part="${PDMUX_MOCAP_PART:-}"
mode="${PDMUX_MOCAP_MODE:-}"
job="${PDMUX_MOCAP_JOB:-}"
budget="${PDMUX_MOCAP_BUDGET_MIN:-}"
wl="${PDMUX_MOCAP_WORKLOAD:-}"
expected_digest="8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858"
out="${PDMUX_MOCAP_OUT:-${eng}/results/motivation_cap_runs/${name}}"
raw="${out}/raw"
export PDMUX_HEARTBEAT_S=0        # measurement Job: no heartbeat (patched boots get the same value from the runner's server env)

case "${part}" in fixed|serving) ;; *) echo "ABORT_CONFIG: PDMUX_MOCAP_PART must be fixed|serving; got '${part}'" >&2; exit 16 ;; esac
case "${mode}" in rehearse|run) ;; *) echo "ABORT_CONFIG: PDMUX_MOCAP_MODE must be rehearse|run; got '${mode}'" >&2; exit 16 ;; esac
[[ -n "${job}" && "${budget}" =~ ^[0-9]+$ ]] || { echo "ABORT_CONFIG: PDMUX_MOCAP_JOB and PDMUX_MOCAP_BUDGET_MIN (minutes) required" >&2; exit 16; }
if [[ "${part}" == fixed && "${mode}" == run ]]; then
  [[ "${PDMUX_MOCAP_ROUND:-}" =~ ^[12]$ ]] || { echo "ABORT_CONFIG: PDMUX_MOCAP_ROUND (1|2) required: one Job per round" >&2; exit 16; }
fi
if [[ "${part}" == serving && "${mode}" == run ]]; then
  for v in PDMUX_MOCAP_WORKLOAD PDMUX_MOCAP_RMIN PDMUX_MOCAP_FACTOR; do
    [[ -n "${!v:-}" ]] || { echo "ABORT_CONFIG: ${v} required for the serving part (the rate grid comes from the pre-ladder)" >&2; exit 16; }
  done
fi
mkdir -p "${out}" || { echo "ERROR: cannot create ${out}" >&2; exit 2; }

snap="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052"
if [[ "${PDMUX_MOCAP_TEST_MODE:-}" != 1 ]]; then
  ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  { echo "gpu_count=${ngpu}"; echo "image=${PDMUX_IMAGE:-}"; echo "part=${part}"; echo "mode=${mode}"; echo "workload=${wl}"; } > "${out}/preflight.txt"
  [[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, this Job needs exactly 1" >&2; exit 16; }
  python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${data}/runs/${name}/meta/runtime_source_manifest.sha256" \
    --sealed "${expected_digest}" --out "${out}/ENGINE_DIGEST_CHECK.json" > "${out}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
    || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
  pst="$(dirname "$(dirname "${SGLANG_ENGINE_DEV:-${PDMUX_ROOT:-/opt/pdmux}/sglang_engine_dev/python}")")/sglang_pristine/PRESERVE_STATUS"
  cp "${pst}" "${out}/PRESERVE_STATUS" 2>/dev/null || true
  grep -qx 'status=OK' "${pst}" 2>/dev/null || { echo "ABORT_PRISTINE: ${pst} missing or not OK" >&2; exit 15; }
  [[ -f "${snap}/config.json" ]] || { echo "ABORT_DATASET: Nano snapshot ${snap} missing" >&2; exit 13; }
  if [[ "${part}" == serving ]]; then
    sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
    [[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing" >&2; exit 13; }
    sha256sum "${sg}" > "${out}/sharegpt.sha256"
    python3 "${eng}/results/cap_char/policy_harness/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${snap}/tokenizer.json" \
      --out "${out}/w1_lengths.json" > "${out}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }
  fi
else
  snap="${PDMUX_MOCAP_VARIANT_SRC:-${snap}}"
  [[ "${part}" == serving ]] && cp "${PDMUX_MOCAP_W1_LENGTHS:?}" "${out}/w1_lengths.json"
fi

common=(--part "${part}" --job "${job}")
if [[ "${part}" == fixed ]]; then
  common+=(--variant-src "${snap}")
  [[ "${mode}" == run ]] && common+=(--round "${PDMUX_MOCAP_ROUND}")
else
  common+=(--w1-lengths "${out}/w1_lengths.json")
  [[ -n "${wl}" ]] && common+=(--workload "${wl}")
  [[ -n "${PDMUX_MOCAP_RMIN:-}" ]] && common+=(--rmin "${PDMUX_MOCAP_RMIN}")
  [[ -n "${PDMUX_MOCAP_FACTOR:-}" ]] && common+=(--factor "${PDMUX_MOCAP_FACTOR}")
fi
if [[ "${mode}" == rehearse ]]; then
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_cap_runner.py" rehearse "${common[@]}" --out "${raw}" > "${out}/rehearse.json"
  rc=$?
else
  python3 "${here}/mo_cap_runner.py" card "${common[@]}" --out "${out}/card.json" > "${out}/card_summary.json" \
    || { echo "ABORT_CARD: card build failed" >&2; exit 16; }
  if [[ -n "${PDMUX_MOCAP_EXPECT_CARD_SHA:-}" ]]; then
    got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["card_sha"])' "${out}/card.json")"
    [[ "${got}" == "${PDMUX_MOCAP_EXPECT_CARD_SHA}" ]] || { echo "ABORT_CARD: card sha ${got:0:16} != registered ${PDMUX_MOCAP_EXPECT_CARD_SHA:0:16}" >&2; exit 16; }
  fi
  args=(run "${common[@]}" --card "${out}/card.json" --out "${raw}")
  [[ "${PDMUX_MOCAP_RESUME:-}" == 1 ]] && args+=(--resume)
  timeout -k 60 "$(( budget * 60 ))" python3 "${here}/mo_cap_runner.py" "${args[@]}"
  rc=$?
  if [[ -f "${raw}/plan.json" ]]; then
    clk="${data}/runs/${name}/meta/clk.csv"
    an=(python3 "${here}/mo_cap_analyze.py" "${part}" "${raw}" --out "${out}/analysis")
    [[ "${part}" == serving && -f "${clk}" ]] && an+=(--clk "${clk}")
    "${an[@]}" > "${out}/analysis.txt" 2>&1
    arc=$?
    [[ ${rc} -eq 0 && ${arc} -ne 0 ]] && rc=${arc}
  fi
fi
[[ ${rc} -eq 124 || ${rc} -eq 137 ]] && echo "{\"verdict\": \"BUDGET_CAP_KILLED\", \"rc\": ${rc}, \"budget_min\": ${budget}}" > "${out}/BUDGET_CAP_KILLED.json"
[[ -d "${raw}" ]] && (cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "motivation cap ${part} ${mode}: rc=${rc} out=${out}"
exit "${rc}"
