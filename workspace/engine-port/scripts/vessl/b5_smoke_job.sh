#!/bin/bash
# B5 target for a VESSL Job: run the substrate probe INSIDE the Job pipeline (job_entry.sh), so one
# run proves image pull + bundle checkout + install_runtime + GPU work + export + fetch.sh end to end.
# Launch: scripts/vessl/launch.sh --campaign b5_smoke --target workspace/engine-port/scripts/vessl/b5_smoke_job.sh
# Output lands under results/b5_smoke/ (job_entry exports it); manifest = the one job_entry just wrote.
set -euo pipefail
: "${PDMUX_PROJECT_ROOT:?}" "${PDMUX_JOB_NAME:?}"
exec bash "${PDMUX_PROJECT_ROOT}/workspace/engine-port/scripts/vessl/substrate_probe.sh" \
  --manifest "${PDMUX_DATA:-/data}/runs/${PDMUX_JOB_NAME}/meta/runtime_source_manifest.sha256" \
  --out-root "${PDMUX_PROJECT_ROOT}/workspace/engine-port/results/b5_smoke"
