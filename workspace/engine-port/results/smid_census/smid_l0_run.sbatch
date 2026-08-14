#!/bin/bash
#SBATCH --job-name=smidl0
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/smid_census/smidl0_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/smid_census/smidl0_%j.err
# ============================================================================
# L0 standalone `%smid` census -- P1 (R0/R1/R2/R5) + P2 (driver read-out).
# Pre-registration: PREREG_SMID_R0_2026-08-14.md (this directory). Read it
# first; every threshold and verdict string is fixed there, not here.
#
# ★ NO SERVER IS STARTED. No model is loaded. No request is issued. This job
#   launches spin kernels on CUDA streams and reads a hardware register. It
#   produces NO latency, NO throughput, NO goodput -- there is nothing in its
#   output that can be quoted as a performance number, by construction.
#
# ★ THE DEV TREE IS NOT MODIFIED and no manifest is re-frozen. This job does
#   not call sync_engine_tree.sh: the tree is byte-identical to the state of
#   jobs 877756/877757/877974 and running sync would be a write where the
#   pre-registration promised zero writes. It records the sha256 of the two
#   critical-path files that the manifest does NOT cover
#   (sglang/srt/multiplex/pdmux_context.py, sgl_kernel/spatial.py --
#   methodology gate #33) directly into the raw artefact instead.
#
# Cost: ~10 minutes wall, of which the census itself is seconds. Everything
# else is module load + import.
#
# Usage:   sbatch smid_l0_run.sbatch
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/smid_census
JOBTAG="${SLURM_JOB_ID:-local}"
export PYTHONUNBUFFERED=1
# $HOME is at its file quota; a default ~/.triton cache aborts compilation with
# "Disk quota exceeded" before a single kernel is built. Observed on glogin01
# 2026-08-14 while writing this script.
export TRITON_CACHE_DIR="$OUT/.triton_cache_smid_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

echo "############# smid L0 job=$JOBTAG (PREREG_SMID_R0_2026-08-14.md) #############"
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv || true
echo "--- compute-node profiling permission (design probe P3, RECORD ONLY;"
echo "--- CUPTI is already dead by principle, cuda 13.0.2 headers carry no smId) ---"
grep -i "RmProfilingAdminOnly" /proc/driver/nvidia/params || echo "(field absent)"

echo "=== stage 1/4: CPU self-test (instrument, no GPU kernels) ==="
python3 "$OUT/smid_l0_census.py" --selftest-cpu --tag "$JOBTAG" --outdir "$OUT"
RC_CPU=$?
echo "selftest-cpu rc=$RC_CPU"

echo "=== stage 2/4: analyzer self-test (gate #21 branch fixed in code) ==="
python3 "$OUT/smid_l0_census.py" --selftest-analyzer
RC_AN=$?
echo "selftest-analyzer rc=$RC_AN"

if [ "$RC_CPU" -ne 0 ] || [ "$RC_AN" -ne 0 ]; then
  echo "PLUMBING FAILED -- refusing to run the census. This is a MEASUREMENT"
  echo "condition, not a result about the engine (methodology gate #21)."
  exit 20
fi

echo "=== stage 3/4: GPU census (R0/R1/R2/R5 raw; NO VERDICT emitted here) ==="
python3 "$OUT/smid_l0_census.py" --run --tag "$JOBTAG" --outdir "$OUT"
RC_RUN=$?
echo "run rc=$RC_RUN"
if [ "$RC_RUN" -ne 0 ]; then
  echo "CENSUS DID NOT COMPLETE -> UNDETERMINED (MEASUREMENT ABSENT)."
  echo "Do not score this job. Do not report any set statement from it."
  exit 21
fi

echo "=== stage 4/4: scoring against the pre-registered rules ==="
python3 "$OUT/smid_l0_census.py" --analyze "$OUT/smid_l0_raw_${JOBTAG}.json" \
        --tag "$JOBTAG" --outdir "$OUT" | tee "$OUT/smid_l0_verdict_${JOBTAG}.txt"

echo "############# done job=$JOBTAG #############"
echo "REMINDER: a GLOBALLY_CONSISTENT_LABEL verdict licenses no performance"
echo "claim, does not close the CONSENSUS sec1-1 attribution, and does not add"
echo "a single magnitude-citable cell to Gate 2-S (prereg sec3)."
