#!/bin/bash
# DIAGNOSTIC, not a measurement.  Job target for the diagnostic boots B2 -> B5 -> B1 -> B3 -> B4 (DIAGBOOT_itl_gap_2026-11-03.md),
# used because Workspace start is refused for capacity.  Nothing here is a registered cell; no number from it may be cited as a
# measurement (single run per variant, no pairing, one shared node).  Same env/path contract as sb_job.sh, called by job_entry.sh.
#
#   PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.sbase.env scripts/vessl/launch.sh --campaign stock_baseline_diag \
#     --target workspace/engine-port/results/stock_baseline/diagboot/diag_job.sh --env PDMUX_REQUIRE_PRISTINE=1
#
# env: PDMUX_DIAG_HARDCAP_MIN (default 90, whole script), PDMUX_DIAG_VARIANTS (default "B2 B5 B1 B3 B4"),
#      PDMUX_DIAG_TEST_MODE=1 (CPU mock: skips digest/pristine/dataset checks; still stops at the GPU check).
# rc: 0 ok (every variant attempted; boot failures are recorded in <variant>/BOOT_FAIL.txt, not fatal), 2 setup,
#     13 ABORT_ENGINE_DIGEST/ABORT_DATASET, 15 ABORT_PRISTINE, 16 substrate (needs 1 GPU), 18 raw export, 124/137 hard cap.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../results/stock_baseline/diagboot
sbd="$(cd "${here}/.." && pwd)"
eng="$(cd "${here}/../../.." && pwd)"                      # workspace/engine-port
export PDMUX_PROJECT_ROOT="${PDMUX_PROJECT_ROOT:-$(cd "${eng}/../.." && pwd)}"
name="${PDMUX_JOB_NAME:-local-$(date -u +%Y%m%d%H%M%S)}"
data="${PDMUX_DATA:-/data}"
raw="${PDMUX_SB_OUT:-${data}/runs/${name}/raw}"
summary="${PDMUX_SB_SUMMARY:-${sbd}/runs/${name}}"
expected_digest="2836112bd710a712a26a31fb516a685a25c6d893308b2058bb7b5590deda5ed7"
cap_min="${PDMUX_DIAG_HARDCAP_MIN:-90}"
variants="${PDMUX_DIAG_VARIANTS:-B2 B5 B1 B3 B4}"
test_mode="${PDMUX_DIAG_TEST_MODE:-}"
[[ "${cap_min}" =~ ^[0-9]+$ ]] || { echo "ABORT_CONFIG: PDMUX_DIAG_HARDCAP_MIN='${cap_min}'" >&2; exit 2; }
mkdir -p "${raw}" "${summary}" || { echo "ERROR: cannot create ${raw} / ${summary}" >&2; exit 2; }
cat > "${summary}/DIAGNOSTIC_NOT_A_MEASUREMENT.txt" <<TXT
DIAGNOSTIC, not a measurement.  Boots ${variants} of results/stock_baseline/diagboot (one run per variant, no pairing).
Purpose: locate the ITL gap (DIAGBOOT_itl_gap_2026-11-03.md).  Do not cite any number here as a measurement or a cell.
TXT

cp "${summary}/DIAGNOSTIC_NOT_A_MEASUREMENT.txt" "${raw}/"
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
echo "gpu_count=${ngpu}" > "${summary}/preflight.txt"
if [[ "${test_mode}" != 1 ]]; then
  edigest="${sbd}/ENGINE_DIGEST.txt"
  if [[ ! -f "${edigest}" ]] || [[ "$(tr -d ' \n' < "${edigest}")" != "${expected_digest}" ]]; then
    echo "ABORT_ENGINE_DIGEST: ${edigest} is not the registered engine digest ${expected_digest:0:12}..." >&2; exit 13; fi
  [[ "${ngpu}" == 1 ]] || { echo "ABORT_SUBSTRATE: ${ngpu} GPUs visible, need exactly 1" >&2; exit 16; }
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader > "${summary}/gpu.txt" 2>/dev/null || true
  echo "image=${PDMUX_IMAGE:-}" >> "${summary}/preflight.txt"
  manifest="${PDMUX_SB_MANIFEST:-${data}/runs/${name}/meta/runtime_source_manifest.sha256}"
  python3 "${eng}/scripts/bootstrap/engine_digest.py" check --manifest "${manifest}" --sealed-file "${edigest}" \
    --out "${summary}/ENGINE_DIGEST_CHECK.json" > "${summary}/ENGINE_DIGEST_CHECK.txt" 2>&1 \
    || { echo "ABORT_ENGINE_DIGEST (see ENGINE_DIGEST_CHECK.txt)" >&2; exit 13; }
  pst="$(dirname "$(dirname "${SGLANG_ENGINE_DEV:-${PDMUX_ROOT:-/opt/pdmux}/sglang_engine_dev/python}")")/sglang_pristine/PRESERVE_STATUS"
  cp "${pst}" "${summary}/PRESERVE_STATUS" 2>/dev/null || true
  grep -qx 'status=OK' "${pst}" 2>/dev/null || { echo "ABORT_PRISTINE: ${pst} missing or not OK (set PDMUX_REQUIRE_PRISTINE=1)" >&2; exit 15; }
  sg="${HF_HOME:-${data}/hf}/raw/ShareGPT_V3_unfiltered_cleaned_split.json"
  [[ -f "${sg}" ]] || { echo "ABORT_DATASET: ${sg} missing (stage_data.sh)" >&2; exit 13; }
  tok="${HF_HOME:-${data}/hf}/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/tokenizer.json"
  sha256sum "${sg}" > "${summary}/sharegpt.sha256"
  # same generation path as sb_job.sh (identical to the first Job's raw w1_lengths.json)
  python3 "${eng}/results/cap_char/policy_harness/ph_trace.py" build-w1 --sharegpt "${sg}" --tokenizer "${tok}" --out "${raw}/w1_lengths.json" \
    > "${summary}/w1_lengths.txt" || { echo "ABORT_DATASET: W1 length build failed" >&2; exit 13; }
else
  [[ -f "${PDMUX_SB_W1_LENGTHS:-}" ]] && cp "${PDMUX_SB_W1_LENGTHS}" "${raw}/w1_lengths.json"
  echo "TEST_MODE: digest/pristine/dataset checks skipped" >> "${summary}/preflight.txt"
fi
if [[ "${ngpu}" -lt 1 ]]; then
  echo "ABORT_SUBSTRATE: 서버 부팅은 GPU 필요 (no GPU visible; DIAG boots B-variants cannot run on CPU)" >&2; exit 16; fi

start=$SECONDS; deadline=$(( cap_min * 60 ))
for v in ${variants}; do
  left=$(( deadline - (SECONDS - start) ))
  if (( left <= 60 )); then echo "HARDCAP: ${v} and later variants not started (${cap_min} min)" | tee -a "${summary}/HARDCAP_KILLED.txt"; rc=124; break; fi
  vd="${raw}/${v}"; mkdir -p "${vd}"
  echo "[DIAG $(date -u +%H:%M:%SZ)] variant ${v} (left ${left}s)"
  timeout -k 30 "${left}" python3 "${here}/diag_boot.py" "${v}" "${vd}" "${raw}/w1_lengths.json" > "${vd}/diag_boot.stdout" 2> "${vd}/diag_boot.stderr"
  vrc=$?
  echo "${vrc}" > "${vd}/rc.txt"
  runjson="$(ls "${vd}"/runs/*.json 2>/dev/null | head -1)"
  if [[ ${vrc} -ne 0 || -z "${runjson}" ]]; then
    { echo "variant=${v} rc=${vrc} (no run json: $([[ -z "${runjson}" ]] && echo yes || echo no))"; echo "--- stderr (tail) ---"; tail -n 200 "${vd}/diag_boot.stderr"; } > "${vd}/BOOT_FAIL.txt"
    echo "[DIAG] ${v}: BOOT_FAIL rc=${vrc}, continuing"
    continue
  fi
  python3 "${here}/diag_stats.py" "${runjson}" > "${vd}/stats.json" 2> "${vd}/stats.stderr" \
    || { echo "variant=${v} diag_stats failed" > "${vd}/STATS_FAIL.txt"; tail -n 50 "${vd}/stats.stderr" >> "${vd}/STATS_FAIL.txt"; }
done
rc="${rc:-0}"
[[ -f "${data}/runs/${name}/meta/clk.csv" ]] && { mkdir -p "${raw}/clk"; cp "${data}/runs/${name}/meta/clk.csv" "${raw}/clk/${name}.csv" 2>/dev/null || true; }
(cd "${raw}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS) && cp "${raw}/SHA256SUMS" "${summary}/RAW_SHA256SUMS"
if [[ "${raw}" != "${summary}"/* && "${raw}" != "${summary}" ]]; then
  if mkdir -p "${summary}/raw" && cp -a "${raw}/." "${summary}/raw/" && (cd "${summary}/raw" && sha256sum --quiet -c SHA256SUMS) \
     && [[ "$(find "${raw}" -type f | wc -l)" == "$(find "${summary}/raw" -type f | wc -l)" ]]; then
    echo "raw exported to ${summary}/raw ($(find "${summary}/raw" -type f | wc -l) files, sha256 verified); DIAGNOSTIC, not a measurement" > "${summary}/RAW_EXPORT_OK"
  else echo "ABORT_RAW_EXPORT: could not export/verify ${raw} -> ${summary}/raw" >&2; [[ "${rc}" -eq 0 ]] && rc=18; fi
fi
echo "diag: rc=${rc} raw=${raw} (DIAGNOSTIC, not a measurement)"
exit "${rc}"
