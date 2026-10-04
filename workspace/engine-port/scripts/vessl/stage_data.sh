#!/bin/bash
# One-time data staging inside the CPU Workspace (pdmux-build): models + traces onto the cluster
# volume, revision-pinned, with a manifest.  Never run inside a measurement Job (Jobs are offline).
#
#   stage_data.sh <revisions.tsv> <model-id>...      e.g. nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base
#
#   <revisions.tsv>  the migration bundle's D_misc/hf_hub_revisions.tsv (uploaded to /io/staging/):
#                    "models--<org>--<name>\t<commit>" per line -- the exact revisions every KISTI
#                    campaign served.  A model id missing from it is refused (no floating "main").
# ENV  HF_HOME (default /data/hf)   PDMUX_IO (default /io)
# Traces: /io/staging/hf_cache_raw_traces.tar.zst -> $HF_HOME/raw/ (the $HF_HOME/raw convention the
# campaign scripts read), verified against /io/staging/SHA256SUMS first.
# Gated models need `hf auth login` typed by the user beforehand, and `hf auth logout` afterwards.
# Plan: handoff-report/vessl_operating_model_2026-10-02.md sec 2 / sec 7.
set -euo pipefail

revs="${1:?usage: stage_data.sh <revisions.tsv> <model-id>...}"; shift
export HF_HOME="${HF_HOME:-/data/hf}"
io="${PDMUX_IO:-/io}"
mkdir -p "${HF_HOME}"
log="${HF_HOME}/STAGING_LOG.tsv"
[[ -f "${log}" ]] || printf 'utc\tkind\tid\trevision\tstatus\n' > "${log}"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---- traces --------------------------------------------------------------------------------
if [[ ! -f "${HF_HOME}/raw/ShareGPT_V3_unfiltered_cleaned_split.json" ]]; then
  (cd "${io}/staging" && sha256sum -c --quiet SHA256SUMS)
  tar --zstd -xf "${io}/staging/hf_cache_raw_traces.tar.zst" -C "${HF_HOME}" --strip-components=1
  printf '%s\ttraces\thf_cache_raw_traces.tar.zst\t%s\tok\n' "$(utc)" \
    "$(sha256sum "${io}/staging/hf_cache_raw_traces.tar.zst" | cut -c1-16)" >> "${log}"
fi

# ---- models (revision-pinned) ----------------------------------------------------------------
for model in "$@"; do
  key="models--${model//\//--}"
  rev="$(awk -F'\t' -v k="${key}" '$1==k{print $2}' "${revs}")"
  [[ -n "${rev}" ]] || { echo "ERROR: ${model} not in ${revs}; refusing floating revision" >&2; exit 2; }
  if hf download "${model}" --revision "${rev}" > /dev/null; then
    printf '%s\tmodel\t%s\t%s\tok\n' "$(utc)" "${model}" "${rev}" >> "${log}"
  else
    printf '%s\tmodel\t%s\t%s\tFAILED\n' "$(utc)" "${model}" "${rev}" >> "${log}"; exit 3
  fi
done

# ---- manifest of what is now on the volume ---------------------------------------------------
(cd "${HF_HOME}" && find raw hub -type f ! -path '*/.locks/*' 2>/dev/null | sort | xargs -r sha256sum) \
  > "${HF_HOME}/STAGED_SHA256SUMS"
tail -n +2 "${log}"
du -sh "${HF_HOME}"
