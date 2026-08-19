#!/bin/bash
#SBATCH --job-name=kmech-p1-greenctx
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --constrain=hwperf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/p1_probe/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/p1_probe/%x_%j.err
#
# kernel_mech Stage 0, P1 (PROBE 1 ONLY -- not the rest of Stage 0).
#
# Question: can `ncu` resolve a single CUDA kernel when that kernel runs
# inside a CUDA Green Context stream (SM count restricted below full GPU)?
# See workspace/engine-port/reports/EXPERIMENT_PLAN_2026-08-19.md §B and
# .../results/kernel_mech/DESIGN_KERNEL_MECH_REV2_2026-08-16.md §7-0 item 1.
#
# --exclusive --constrain=hwperf mirrors the one precedent in this repo that
# is known to have produced real ncu hardware-counter captures on this
# cluster (workspace/characterization/slurm/run_ncu_profile.sh). This is a
# deliberately conservative choice for a 1-shot probe: minimize confounds
# (e.g. ERR_NVGPUCTRPERM from a non-exclusive allocation) that could be
# mis-attributed to the CUPTI x Green-Context question under test. All nodes
# in amd_a100nv_8 already carry the hwperf feature (verified via `sinfo -p
# amd_a100nv_8 -o "%N %f"` on 2026-08-19), so --constrain=hwperf does not
# narrow the eligible node set here -- it is included to match the request
# shape of the known-working precedent, not because it changes placement.
#
# Runs TWO ncu-attached subprocess launches of the SAME target script
# (p1_greenctx_target.py, a single bf16 GEMM):
#   1. greenctx : kernel launched on a Green Context stream restricted to
#                 16 SMs (smctrl.set_sm_count(16)) -- CONDITION UNDER TEST.
#   2. full     : same kernel, default (unrestricted) stream -- CONTROL,
#                 to narrow attribution if greenctx fails (per task: "원인을
#                 단정하지 마라" -- do not assume which of the 3 registered
#                 competing attributions for ncu error code 9 applies without
#                 a control).

set -uo pipefail  # NOT -e: ncu is expected to possibly exit non-zero (that
                   # IS the P1=UNAVAILABLE observation) and this script must
                   # still print the summary in that case.

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

RESULT_ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/p1_probe
OUT="${RESULT_ROOT}/job_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
TARGET="${RESULT_ROOT}/p1_greenctx_target.py"

# NCU_PYTHON pin -- re-derived root cause from
# DESIGN_KERNEL_MECH_REV2_2026-08-16.md §9: ncu requires the profiled
# process's CUDA runtime to match ncu's own CUDA major version, or profiling
# dies with "Failed to prepare kernel for profiling / Unknown Error on
# device 0" -- the SAME error signature this probe is trying to attribute to
# CUPTI x Green Context. Pin explicitly (sys.executable, not `command -v`,
# per that section's finding) and log both versions side by side so a silent
# toolkit mismatch on a future re-run is caught here, not misread as a
# green-context finding (Stage 0 item 0 recommendation).
export NCU_PYTHON="$(python -c 'import sys; print(sys.executable)')"

{
  echo "=== P1 probe (kernel_mech Stage 0, item 1) | job=${SLURM_JOB_ID} | node=${SLURMD_NODENAME:-unknown} ==="
  echo "GPU        : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "ncu path   : $(command -v ncu)"
  echo "ncu version: $(ncu --version 2>&1 | sed -n '3p')"
  echo "torch cuda : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
  echo "NCU_PYTHON : ${NCU_PYTHON}"
  echo "target     : ${TARGET}"
  echo ""
} | tee "${OUT}/env_diagnostic.txt"

# Same metric set NCURunner (workspace/characterization/src/profiling/
# ncu_runner.py) uses for ncu>=2025 (dot-notation, this cluster's ncu is
# 2025.3.1). These are genuine CUPTI hardware counters (not just launch-
# config metadata) -- required so this probe actually exercises the CUPTI
# path the incompatibility claim is about, not just software-only launch
# stats. Grid Size / Block Size come back as CSV columns automatically
# (ncu_runner.py:34-39) -- launch__grid_size / launch__waves_per_multiprocessor
# were confirmed ABSENT from `ncu --query-metrics --chip ga100` on this
# cluster's ncu 2025.3.1 (checked 2026-08-19, login node, 0 matches) so this
# probe reports grid_size (CSV column) + an analytically-derived wave count
# (grid_size / realized_sm_count, the same derivation ncu_runner.py's
# _derive_sm_util already performs) rather than a metric name that does not
# exist in this ncu version. This is a naming-generation gap, not a
# deviation from what item 1 asks for.
METRICS="sm__cycles_active.sum,sm__cycles_elapsed.sum,smsp__warps_active.sum,gpu__time_duration.sum"

run_ncu () {
  local label="$1"; shift
  echo "############################################################"
  echo "# ${label}"
  echo "############################################################"
  ncu --metrics "${METRICS}" --csv --target-processes all --clock-control none \
      "${NCU_PYTHON}" "${TARGET}" "$@" \
      > "${OUT}/${label}.csv.log" 2> "${OUT}/${label}.err.log"
  local rc=$?
  echo "exit code: ${rc}"
  echo "--- stdout tail ---"
  tail -n 60 "${OUT}/${label}.csv.log"
  echo "--- stderr tail ---"
  tail -n 60 "${OUT}/${label}.err.log"
  echo ""
  return "${rc}"
}

run_ncu greenctx --mode greenctx --sm-count 16 --n-warmup 10 --n-measure 3
GREENCTX_RC=$?

run_ncu control --mode full --n-warmup 10 --n-measure 3
CONTROL_RC=$?

{
  echo "=== P1 SUMMARY ==="
  echo "greenctx exit=${GREENCTX_RC}  control exit=${CONTROL_RC}"

  SIG_FOUND=0
  if grep -qE "Unknown Error on device 0|Failed to prepare kernel for profiling" \
       "${OUT}/greenctx.err.log" "${OUT}/greenctx.csv.log" 2>/dev/null; then
    SIG_FOUND=1
  fi

  if [ "${SIG_FOUND}" -eq 1 ]; then
    echo "P1_VERDICT=UNAVAILABLE (CUPTI x GREEN-CONTEXT)"
    echo "  -- documented failure signature observed in greenctx run."
  elif [ "${GREENCTX_RC}" -eq 0 ]; then
    echo "P1_VERDICT=AVAILABLE"
    echo "  -- greenctx run exited 0; inspect greenctx.csv.log for"
    echo "     sm__cycles_active.sum / Grid Size / Block Size columns."
  else
    echo "P1_VERDICT=INCONCLUSIVE"
    echo "  -- greenctx run failed (exit=${GREENCTX_RC}) but the documented"
    echo "     CUPTI x Green-Context signature was NOT found. Do not label"
    echo "     this UNAVAILABLE (CUPTI x GREEN-CONTEXT) -- inspect"
    echo "     greenctx.err.log / greenctx.csv.log for the actual cause"
    echo "     before attributing (registry: 3 competing attributions for"
    echo "     ncu error code 9)."
  fi
  echo "=== done ==="
} | tee "${OUT}/summary.txt"
