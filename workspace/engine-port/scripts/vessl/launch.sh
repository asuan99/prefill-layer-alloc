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

campaign="" target="" array="" submit=0 extra_env=() spec_override=0 rate_given=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) campaign="$2"; shift 2 ;;
    --target)   target="$2"; shift 2 ;;
    --array)    array="$2"; shift 2 ;;
    --env)      extra_env+=("$2"); shift 2 ;;
    --spec)     PDMUX_VESSL_SPEC="$2"; spec_override=1; shift 2 ;;   # e.g. resourcespec-a100cpu (CPU-only Job)
    --rate)     PDMUX_VESSL_RATE_USD_H="$2"; rate_given=1; shift 2 ;;
    --submit)   submit=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${campaign}" && -n "${target}" ]] || { echo "need --campaign and --target" >&2; exit 2; }
[[ "${campaign}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || { echo "campaign: lowercase [a-z0-9_-]" >&2; exit 2; }
[[ -f "${repo}/${target}" ]] || { echo "target not in repo: ${target}" >&2; exit 2; }
if [[ "${spec_override}" == 1 ]]; then   # a spec other than vessl.env's A100: the rate must be given or known
  if [[ "${PDMUX_VESSL_SPEC}" == resourcespec-a100cpu && "${rate_given}" == 0 ]]; then
    PDMUX_VESSL_RATE_USD_H=0.30           # CPU Only, betelgeuse, 3.5 CPU / 13 GiB (resource-spec list 2026-10-16)
  elif [[ "${rate_given}" == 0 ]]; then
    echo "WARNING: spec ${PDMUX_VESSL_SPEC}: rate shown is vessl.env's (not this spec's) -- pass --rate" >&2
  fi
fi
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
     "${env_args[@]}" --tag "${campaign//_/-}"   # VESSL tags: [a-z0-9-] only (underscore -> rc=1 after create, 2026-10-04)
     --cmd "bash /io/${code_prefix}job_entry.sh")

echo "job name : ${name}"
echo "commit   : ${commit}"
echo "rate     : \$${PDMUX_VESSL_RATE_USD_H}/h (${PDMUX_VESSL_SPEC}) -- billed while running"
printf 'command  :'; printf ' %q' "${cmd[@]}"; echo
# rev11 R-5: an M3 Job on J4/J8 needs J2's committed pass table + its sha256 (G0(ii) result for the TP axis); make their presence visible in the dry-run.
# Missing either one does not abort the Job -- it silently turns the Job's TP configs into NOT_TESTED -- so the approver must see it here.
if [[ "${target}" == */m3_job.sh ]]; then
  mgjob=""; own_sha=""; j2_sha=""
  for kv in "${extra_env[@]+"${extra_env[@]}"}"; do
    case "${kv}" in
      PDMUX_MG_JOB=*) mgjob="${kv#*=}" ;;
      PDMUX_M2G_PASS_SHA256=*) own_sha="${kv#*=}" ;;
      PDMUX_M2G_PASS_SHA256_J2=*) j2_sha="${kv#*=}" ;;
    esac
  done
  mgdir="${repo}/workspace/engine-port/results/multigpu"
  tabstate () {   # $1 = table basename: tracked+clean in HEAD?
    local t="${mgdir}/$1" c; c="${mgdir}/$1.sha256"
    [[ -f "${t}" ]] && printf 'table present' || printf 'table ABSENT'
    if git -C "${repo}" ls-files --error-unmatch "${c}" >/dev/null 2>&1 && git -C "${repo}" diff --quiet HEAD -- "${c}" 2>/dev/null; then
      printf ', committed .sha256 tracked+clean'
    else printf ', committed .sha256 NOT tracked/clean'; fi
    if [[ -f "${t}" && -f "${c}" ]]; then
      [[ "$(sha256sum "${t}" | cut -d' ' -f1)" == "$(cut -d' ' -f1 < "${c}")" ]] && printf ', file sha == committed' || printf ', file sha != committed (MISMATCH)'
    fi
  }
  echo "m3 job   : PDMUX_MG_JOB=${mgjob:-<unset>}"
  echo "own table: M2G_PASS_TABLE_${mgjob:-?}.json -- $(tabstate "M2G_PASS_TABLE_${mgjob:-?}.json"); env PDMUX_M2G_PASS_SHA256 $([[ -n "${own_sha}" ]] && echo "set (${#own_sha} chars)" || echo "NOT SET -> the Job aborts ABORT_M2G_NOT_PASSED")"
  if [[ "${mgjob}" == J4 || "${mgjob}" == J8 ]]; then
    echo "J2 table : M2G_PASS_TABLE_J2.json -- $(tabstate M2G_PASS_TABLE_J2.json); env PDMUX_M2G_PASS_SHA256_J2 $([[ -n "${j2_sha}" ]] && echo "set (${#j2_sha} chars)" || echo "NOT SET -> TP configs NOT_TESTED (DP only)")"
  fi
fi
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
create_rc=0
out="$("${cmd[@]}" 2>&1)" || create_rc=$?
echo "${out}"
# vesslctl may exit non-zero after the job was in fact created (seen on the first B5 submit,
# 2026-10-04: job existed, ledger row lost because `set -e` killed this script).  Trust the
# platform, not the exit code: look the job up by name before deciding.
if ! vesslctl job list 2>/dev/null | awk -v n="${name}" '$2==n{f=1} END{exit !f}'; then
  echo "ERROR: job ${name} not found after create (rc=${create_rc}); nothing recorded" >&2; exit 3
fi
[[ "${create_rc}" -eq 0 ]] || echo "WARNING: vesslctl job create rc=${create_rc} but the job exists; recording it" >&2
ledger="${repo}/workspace/engine-port/results/${campaign}/vessl_jobs.tsv"
mkdir -p "$(dirname "${ledger}")"
[[ -f "${ledger}" ]] || printf 'submitted_utc\tjob_name\tcommit\tspec\trate_usd_h\timage\ttarget\tarray\tfetched\n' > "${ledger}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${commit}" \
  "${PDMUX_VESSL_SPEC}" "${PDMUX_VESSL_RATE_USD_H}" "${PDMUX_VESSL_IMAGE}" "${target}" "${array}" "no" >> "${ledger}"
echo "recorded in ${ledger}; fetch with: scripts/vessl/fetch.sh ${campaign} ${name}"
