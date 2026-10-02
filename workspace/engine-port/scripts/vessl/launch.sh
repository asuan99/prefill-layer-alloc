#!/bin/bash
# Local launcher for VESSL Cloud measurement Jobs.  Dry-run by default.
#
#   launch.sh --campaign <label> --target <repo-relative script> [--array <spec>]
#             [--env KEY=VALUE]... [--submit]
#
# Without --submit it prints the exact `vesslctl job create` command, the hourly rate and
# the job name, and uploads nothing.  With --submit it (1) refuses a dirty tree, (2) uploads
# a git bundle of HEAD + job_entry.sh to the object volume under code/<commit>/ (skipped if
# already there), (3) creates the Job, (4) appends a row to
# workspace/engine-port/results/<campaign>/vessl_jobs.tsv (commit it with the results).
# Submitting costs money: run --submit only after the user approved this exact command.
# Plan: handoff-report/vessl_operating_model_2026-10-02.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${here}" rev-parse --show-toplevel)"
cfg="${PDMUX_VESSL_ENV:-${here}/vessl.env}"
[[ -f "${cfg}" ]] || { echo "ERROR: ${cfg} missing (copy vessl.env.example and fill it)" >&2; exit 2; }
# shellcheck disable=SC1090
source "${cfg}"

campaign="" target="" array="" submit=0 extra_env=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) campaign="$2"; shift 2 ;;
    --target)   target="$2"; shift 2 ;;
    --array)    array="$2"; shift 2 ;;
    --env)      extra_env+=("$2"); shift 2 ;;
    --submit)   submit=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${campaign}" && -n "${target}" ]] || { echo "need --campaign and --target" >&2; exit 2; }
[[ "${campaign}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo "campaign: lowercase [a-z0-9_-]" >&2; exit 2; }
[[ -f "${repo}/${target}" ]] || { echo "target not in repo: ${target}" >&2; exit 2; }
for v in PDMUX_VESSL_SPEC PDMUX_VESSL_RATE_USD_H PDMUX_VESSL_CLUSTER_VOLUME \
         PDMUX_VESSL_OBJECT_VOLUME PDMUX_VESSL_IMAGE; do
  [[ -n "${!v:-}" && "${!v}" != *"<"* ]] || { echo "ERROR: ${v} not filled in ${cfg}" >&2; exit 2; }
done
[[ "${PDMUX_VESSL_IMAGE}" == *@sha256:* ]] || { echo "ERROR: image must be pinned by digest" >&2; exit 2; }
for kv in "${extra_env[@]+"${extra_env[@]}"}"; do
  [[ "${kv}" =~ ^PDMUX_[A-Z0-9_]+=.*$ ]] || { echo "ERROR: --env only for PDMUX_* knobs: ${kv}" >&2; exit 2; }
done

commit="$(git -C "${repo}" rev-parse HEAD)"
name="pdmux-${campaign//_/-}-${commit:0:7}-$(date -u +%Y%m%d%H%M%S)"
code_prefix="code/${commit}/"
env_args=(-e "PDMUX_COMMIT=${commit}" -e "PDMUX_JOB_NAME=${name}" -e "PDMUX_CAMPAIGN=${campaign}"
          -e "PDMUX_TARGET=${target}" -e "PDMUX_IMAGE=${PDMUX_VESSL_IMAGE}"
          -e "PDMUX_SPEC=${PDMUX_VESSL_SPEC}")
[[ -n "${array}" ]] && env_args+=(-e "PDMUX_ARRAY_SPEC=${array}")
for kv in "${extra_env[@]+"${extra_env[@]}"}"; do env_args+=(-e "${kv}"); done
cmd=(vesslctl job create -n "${name}" -r "${PDMUX_VESSL_SPEC}" -i "${PDMUX_VESSL_IMAGE}"
     --image-pull-policy IfNotPresent
     --cluster-volume "${PDMUX_VESSL_CLUSTER_VOLUME}:/data"
     --object-volume "${PDMUX_VESSL_OBJECT_VOLUME}:/io"
     "${env_args[@]}" --tag "${campaign}"
     --cmd "bash /io/${code_prefix}job_entry.sh")

echo "job name : ${name}"
echo "commit   : ${commit}"
echo "rate     : \$${PDMUX_VESSL_RATE_USD_H}/h (${PDMUX_VESSL_SPEC}) -- billed while running"
printf 'command  :'; printf ' %q' "${cmd[@]}"; echo
if [[ "${submit}" -ne 1 ]]; then
  echo "(dry-run: nothing uploaded or created; add --submit after approval)"
  exit 0
fi

if [[ -n "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked changes not committed; the job runs HEAD only" >&2; exit 2
fi
if ! vesslctl volume ls "${PDMUX_VESSL_OBJECT_VOLUME}" --prefix "${code_prefix}" 2>/dev/null \
     | grep -q "repo.bundle"; then
  stage="$(mktemp -d)"; trap 'rm -rf "${stage}"' EXIT
  git -C "${repo}" bundle create "${stage}/repo.bundle" HEAD
  (cd "${stage}" && sha256sum repo.bundle > repo.bundle.sha256)
  cp "${here}/job_entry.sh" "${stage}/job_entry.sh"
  vesslctl volume upload "${PDMUX_VESSL_OBJECT_VOLUME}" "${stage}" --remote-prefix "${code_prefix}"
fi
out="$("${cmd[@]}" 2>&1)"; echo "${out}"
ledger="${repo}/workspace/engine-port/results/${campaign}/vessl_jobs.tsv"
mkdir -p "$(dirname "${ledger}")"
[[ -f "${ledger}" ]] || printf 'submitted_utc\tjob_name\tcommit\tspec\trate_usd_h\timage\ttarget\tarray\tfetched\n' > "${ledger}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${commit}" \
  "${PDMUX_VESSL_SPEC}" "${PDMUX_VESSL_RATE_USD_H}" "${PDMUX_VESSL_IMAGE}" "${target}" "${array}" "no" >> "${ledger}"
echo "recorded in ${ledger}; fetch with: scripts/vessl/fetch.sh ${campaign} ${name}"
