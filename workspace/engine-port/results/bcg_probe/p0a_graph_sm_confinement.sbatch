#!/bin/bash
#SBATCH --job-name=bcg-p0a
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.err
# ============================================================================
# P0-A -- does CUDA-graph replay preserve Green-Context SM confinement?
# Design: workspace/engine-port/reports/BCG_APPLICABILITY_2026-08-21.md sec7.
# Question origin: SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md:77-96
#                  ("구멍 C" / assumption E5) -- registered, never measured.
#
# ★ NO SERVER IS STARTED. No model is loaded. No request is issued. This job
#   launches spin kernels and reads the %smid hardware register. It produces
#   NO latency, NO throughput, NO goodput -- by construction there is nothing
#   in its output that can be quoted as a performance number. This is a
#   TOOL-VALIDITY probe (2026-08-20 job 886718 precedent), not a performance
#   verdict.
#
# ★ THE DEV TREE IS NOT MODIFIED and no manifest is re-frozen. The probe
#   records the sha256 of the three critical-path files that
#   sync_engine_tree.sh's manifest does NOT cover (pdmux_context.py,
#   sgl_kernel/spatial.py, smid_l0_census.py) into the raw artefact instead
#   (methodology gate #33).
#
# ★ results/smid_census/smid_l0_census.py IS IMPORTED, NEVER EDITED. It is a
#   pre-registered artefact (PREREG_SMID_R0_2026-08-14.md).
#
# ★ PREREQUISITE -- READ THIS BEFORE SUBMITTING.
#   The base census R0 ("does %smid give a globally consistent physical label
#   inside a green context?") has NEVER been run on a GPU: results/smid_census/
#   contains only smid_l0_cpu_selftest_*.json, no smid_l0_raw_*.json. If R0 is
#   not GLOBALLY_CONSISTENT_LABEL, this probe's green legs are uninterpretable.
#   Submit results/smid_census/smid_l0_run.sbatch FIRST (~10 min) and read its
#   verdict. This script's P4 precondition will refuse to score rather than
#   guess, but that wastes the slot.
#
# Cost: ~10 minutes wall, of which the census itself is seconds.
# Usage:  sbatch p0a_graph_sm_confinement.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe
JOBTAG="${SLURM_JOB_ID:-local}"
export PYTHONUNBUFFERED=1
# $HOME is at its file quota; a default ~/.triton cache aborts compilation with
# "Disk quota exceeded" before a single kernel is built (observed 2026-08-14).
export TRITON_CACHE_DIR="$OUT/.triton_cache_p0a_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

echo "############# BCG P0-A job=$JOBTAG #############"
nvidia-smi --query-gpu=name,memory.total,compute_mode,driver_version --format=csv || true
python3 -c "import torch; print('torch', torch.__version__)"

echo "=== stage 1/3: analyzer self-test (incl. mutation tests, lesson #53) ==="
python3 "$OUT/p0a_graph_sm_confinement.py" --selftest-analyzer
RC_AN=$?
echo "selftest-analyzer rc=$RC_AN"
if [ "$RC_AN" -ne 0 ]; then
  echo "PLUMBING FAILED -- refusing to run the probe. This is a MEASUREMENT"
  echo "condition, not a result about the engine (methodology gate #21)."
  exit 20
fi

echo "=== stage 2/3: GPU legs (raw only; NO VERDICT emitted here) ==="
python3 "$OUT/p0a_graph_sm_confinement.py" --run --tag "$JOBTAG" --outdir "$OUT"
RC_RUN=$?
echo "run rc=$RC_RUN"
if [ "$RC_RUN" -ne 0 ]; then
  echo "PROBE DID NOT COMPLETE -> UNDETERMINED (MEASUREMENT ABSENT)."
  echo "Do not score this job. Do not report any statement from it."
  exit 21
fi

echo "=== stage 3/3: scoring against the pre-fixed rule (docstring sec 'PRE-FIXED') ==="
python3 "$OUT/p0a_graph_sm_confinement.py" --analyze "$OUT/p0a_raw_${JOBTAG}.json" \
        --tag "$JOBTAG" --outdir "$OUT" | tee "$OUT/p0a_verdict_${JOBTAG}.txt"

echo "############# done job=$JOBTAG #############"
echo "REMINDER: every outcome of this probe is a TOOL-VALIDITY / SM-IDENTITY"
echo "statement. PRESERVED licenses no performance claim and closes no gate."
echo "LOST is a reason to re-examine the decode-side SM-split narrative, not a"
echo "performance result. A capture failure is NOT a confinement verdict."
