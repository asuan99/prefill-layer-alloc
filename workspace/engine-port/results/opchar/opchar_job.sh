#!/bin/bash
# opchar Job target -- operator-level characterization of Nemotron-Nano-9B-v2 (Bullet ASPLOS'26
# Fig 2 / Table 2 / Fig 3 analogue).  Plan: ~/.claude/plans/figure2-table2-linear-cherny.md (2026-10-12).
# CHARACTERIZATION ONLY: isolated per-operator kernels on one green partition; no serving claim,
# no policy claim (layer-type policy death and HE0 are untouched).
#
#   launch.sh --campaign opchar --env PDMUX_OPCHAR_MODE=probe \
#     --target workspace/engine-port/results/opchar/opchar_job.sh
#   PDMUX_OPCHAR_MODE = probe (Step 1: capability facts + memcpy ceiling + builder smoke)
#                     | full  (Step 2: full grid; run only after the probe was read)
# Output: results/opchar/<mode>_<job name>/ (exported by job_entry.sh with SHA256SUMS + DONE).
set -uo pipefail
ROOT="${PDMUX_PROJECT_ROOT:?PDMUX_PROJECT_ROOT must be set (job_entry.sh exports it)}"
MODE="${PDMUX_OPCHAR_MODE:-}"
case "$MODE" in probe|full|ctrl|ctrl2) ;; *) echo "ABORT_ENV PDMUX_OPCHAR_MODE must be probe, full, ctrl or ctrl2 (got '$MODE')"; exit 2;; esac
ALLOW=" PDMUX_COMMIT PDMUX_JOB_NAME PDMUX_CAMPAIGN PDMUX_TARGET PDMUX_IMAGE PDMUX_SPEC PDMUX_ROOT PDMUX_PROJECT_ROOT \
PDMUX_IO PDMUX_DATA PDMUX_OPT PDMUX_TEST_SKIP_SYNC PDMUX_ENERGY_SAMPLER PDMUX_OPCHAR_MODE "
for v in $(compgen -e | grep '^PDMUX_' || true); do
  case "$ALLOW" in *" $v "*) ;; *) echo "ABORT_ENV unregistered knob $v is set"; exit 2;; esac
done
OUT="$ROOT/workspace/engine-port/results/opchar/${MODE}_${PDMUX_JOB_NAME:-local$$}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$ROOT/workspace/characterization" || exit 2
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.csv" 2>&1
git -C "$ROOT" rev-parse HEAD > "$OUT/commit.txt" 2>&1
python3 experiments/opchar/run_opchar.py --mode "$MODE" --out-root "$OUT"
rc=$?
echo "OPCHAR_DONE mode=$MODE rc=$rc"
exit "$rc"
