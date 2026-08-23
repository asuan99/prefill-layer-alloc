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
# P0-A -- does CUDA-graph replay preserve green-context SM confinement?
#
# ★ CANON: workspace/engine-port/results/bcg_probe/PREREG_P0A_2026-08-22.md
#   (rev7). This script implements it; where they differ, the prereg wins.
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
# ★ LABEL CEILING: R0 (job 889631, 2026-08-22) established that %smid is a
#   GLOBALLY CONSISTENT LABEL across a green context on this substrate -- NOT
#   that it is a physical SM index. Label-set cardinalities are not compute
#   fractions. Nothing here may be written as "physical SM".
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
# ★ PREREQUISITE -- SATISFIED (this replaces the pre-2026-08-22 warning that
#   R0 "has NEVER been run", prereg sec10-15). R0 ran as job 889631 on
#   2026-08-22 (gpu40, git_head 28a972c) and returned GLOBALLY_CONSISTENT_LABEL
#   with sizes [74, 34] matching divide_sm(108,(8,0),2)[0]; claims-auditor
#   scored it CONFIRMED(scoped). Cite results/smid_census/
#   smid_l0_verdict_889631.json ONLY -- the .txt is truncated (gate #56).
#   If that verdict is ever overturned, this probe becomes UNINTERPRETABLE
#   (prereg sec9-2).
#
# Cost: <= 20 min wall (stage 1 ~3 min CPU, stage 2 ~2-3 min GPU legs).
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
git -C /scratch/ehmoon/whlee/prefill-layer-alloc rev-parse HEAD || true

echo "=== stage 1/3: self-test -- FULL world space + adapter mutants ==="
echo "    (516,096 worlds through the adapter against p0a_rule_totality;"
echo "     nine contract mutants; the R1-R3 read-out battery. ~3 min CPU.)"
python3 "$OUT/p0a_graph_sm_confinement.py" --selftest
RC_ST=$?
echo "selftest rc=$RC_ST"
if [ "$RC_ST" -ne 0 ]; then
  echo "PLUMBING FAILED -- refusing to run the probe. This is a MEASUREMENT"
  echo "condition, not a result about the engine (methodology gate #21)."
  exit 20
fi

echo "=== stage 2/3: GPU legs (raw only; NO VERDICT emitted here) ==="
# The five DECISION legs are flushed to disk as each one completes, and the
# two descriptive cross legs run afterwards. A failure in the descriptive legs
# is recorded as `descriptive_legs_incomplete` and does NOT block scoring
# (prereg sec5.3 / sec10-14); a failure during the decision legs exits non-zero
# here and the run is UNDETERMINED (MEASUREMENT ABSENT).
python3 "$OUT/p0a_graph_sm_confinement.py" --run --tag "$JOBTAG" --outdir "$OUT"
RC_RUN=$?
echo "run rc=$RC_RUN"
if [ "$RC_RUN" -ne 0 ]; then
  echo "PROBE DID NOT COMPLETE -> UNDETERMINED (MEASUREMENT ABSENT)."
  echo "Do not score this job. Do not report any statement from it."
  exit 21
fi

echo "=== stage 3/3: scoring against the pre-registered rule ==="
python3 "$OUT/p0a_graph_sm_confinement.py" --analyze "$OUT/p0a_raw_${JOBTAG}.json" \
        --tag "$JOBTAG" --outdir "$OUT" | tee "$OUT/p0a_verdict_${JOBTAG}.txt"

echo "############# done job=$JOBTAG #############"
echo "★ CITE p0a_verdict_${JOBTAG}.json ONLY. The .txt above is a console"
echo "  transcript and is NOT citable (gate #56, R0 defect N1)."
echo "REMINDER: every outcome of this probe is a TOOL-VALIDITY / LABEL-IDENTITY"
echo "statement. PRESERVED licenses no performance claim, closes no gate, and"
echo "does NOT answer 구멍 C or R4 (prereg sec0.3). LOST is a reason to"
echo "re-examine the decode-side SM-split narrative, not a performance result."
echo "A capture or replay failure is NOT a confinement verdict."
