#!/bin/bash
# Retrieve one VESSL Job's export and verify it.  A job is "done" only after this passes.
#
#   fetch.sh <campaign> <job_name>
#
# 1. downloads results/<job_name>/ from the object volume into a staging dir outside the repo,
# 2. requires DONE (written last by job_entry.sh) and `sha256sum -c SHA256SUMS`,
# 3. merges results/ into workspace/engine-port/results/ WITHOUT overwriting any existing file
#    (a differing existing file aborts the merge and is reported),
# 4. copies meta/ + logs/ to workspace/engine-port/results/<campaign>/vessl/<job_name>/,
#    computes elapsed time and cost, and marks the ledger row fetched.
# Plan: handoff-report/vessl_operating_model_2026-10-02.md.
set -euo pipefail

campaign="${1:?usage: fetch.sh <campaign> <job_name>}"
name="${2:?usage: fetch.sh <campaign> <job_name>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${here}" rev-parse --show-toplevel)"
# shellcheck disable=SC1090
source "${PDMUX_VESSL_ENV:-${here}/vessl.env}"
results="${repo}/workspace/engine-port/results"
stage="${PDMUX_VESSL_STAGE:-$(dirname "$(dirname "${repo}")")/_vessl_fetch}/${name}"

mkdir -p "${stage}"
vesslctl job show "${name}" > "${stage}.job_show.txt" 2>&1 || true   # status at fetch time
vesslctl volume download "${PDMUX_VESSL_OBJECT_VOLUME}" "${stage}" --remote-prefix "results/${name}/"
root="${stage}"; [[ -f "${root}/DONE" ]] || root="${stage}/results/${name}"   # prefix kept or stripped
[[ -f "${root}/DONE" ]] || { echo "NOT DONE: no DONE marker (job still running or export failed)" >&2; exit 3; }
(cd "${root}" && sha256sum --quiet -c SHA256SUMS) || { echo "CHECKSUM FAILED" >&2; exit 3; }

conflicts=0
while IFS= read -r -d '' f; do
  rel="${f#"${root}/results/"}"
  dst="${results}/${rel}"
  if [[ -e "${dst}" ]] && ! cmp -s "${f}" "${dst}"; then
    echo "CONFLICT (not overwritten): ${rel}" >&2; conflicts=$((conflicts + 1))
  fi
done < <(find "${root}/results" -type f -print0 2>/dev/null)
[[ "${conflicts}" -eq 0 ]] || { echo "${conflicts} conflict(s); staging kept at ${root}" >&2; exit 4; }
[[ -d "${root}/results" ]] && cp -an "${root}/results/." "${results}/"

dest="${results}/${campaign}/vessl/${name}"
mkdir -p "${dest}"
cp -a "${root}/meta" "${root}/logs" "${root}/SHA256SUMS" "${root}/DONE" "${dest}/"
cp "${stage}.job_show.txt" "${dest}/job_show_at_fetch.txt"
python3 - "${dest}/meta/timing.txt" "${PDMUX_VESSL_RATE_USD_H}" > "${dest}/cost.txt" <<'PY'
import datetime as d, sys
kv = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l)
f = lambda s: d.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
h = (f(kv["end_utc"]) - f(kv["start_utc"])).total_seconds() / 3600
print(f"container_hours={h:.3f} rate_usd_h={sys.argv[2]} cost_usd_lower_bound={h*float(sys.argv[2]):.2f}")
print("note: excludes scheduling/image-pull time billed before job_entry.sh started")
PY
ledger="${results}/${campaign}/vessl_jobs.tsv"
[[ -f "${ledger}" ]] && awk -F'\t' -v OFS='\t' -v n="${name}" '$2==n{$9="yes"}1' "${ledger}" > "${ledger}.tmp" \
  && mv "${ledger}.tmp" "${ledger}"
echo "FETCHED ${name}: $(cat "${root}/DONE") -> ${dest}"
cat "${dest}/cost.txt"
[[ -f "${dest}/meta/SUBSTRATE_WARNING" ]] && cat "${dest}/meta/SUBSTRATE_WARNING"
grep -q '"throttle_flag": true' "${dest}/meta/clk_summary.json" 2>/dev/null && echo "WARNING: throttle active during run"
exit 0
