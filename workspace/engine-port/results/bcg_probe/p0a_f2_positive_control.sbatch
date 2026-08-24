#!/bin/bash
#SBATCH --job-name=bcg-p0af2
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe/%x_%j.err
# ============================================================================
# P0-A **F2 POSITIVE CONTROL** -- is `replay` what writes the census?
#
# ★ CANON: audit_p0a_result_890893_2026-08-23/VERDICT.md, condition C4 /
#   follow-up F2. That audit left exactly one premise of job 890893 untested:
#   that the labels in the census tensors were written by the GRAPH REPLAY.
#   If something else writes them, 890893's PRESERVED is not wrong -- it is
#   VACUOUS, which is the failure shape this repo has paid for repeatedly.
#
# ★ TWO LEGS ON THE SAME GREEN-DECODE STREAM, DIFFERING IN ONE STEP:
#     capture_only   warmup -> capture            ->  must observe NOTHING
#     capture_replay warmup -> capture -> replay  ->  must observe labels
#   The shim is IMPORTED from p0a_graph_sm_confinement.py (subclassed, replay
#   removed) so both legs take the byte-identical launch path that produced
#   890893. ★THAT FILE IS NOT MODIFIED -- its harness_sha256 is stamped into
#   the cited verdict, and the raw artefact records it for comparison.
#
# ★ NO SERVER. NO MODEL. NO REQUEST. Spin kernels and one hardware register.
#   Nothing in the output can be quoted as a performance number.
#
# ★ WHAT IT CANNOT DO: REPLAY_IS_THE_WRITER closes C4 and nothing else -- it
#   licenses no performance claim, closes no gate, says nothing about
#   confinement, and does not answer 구멍 C or R4.
#
# Cost: <= 15 min wall (import of the cited harness alone costs ~70 s).
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
export TRITON_CACHE_DIR="$OUT/.triton_cache_p0af2_${JOBTAG}"
mkdir -p "$TRITON_CACHE_DIR"

echo "############# BCG P0-A job=$JOBTAG #############"
nvidia-smi --query-gpu=name,memory.total,compute_mode,driver_version --format=csv || true
python3 -c "import torch; print('torch', torch.__version__)"
git -C /scratch/ehmoon/whlee/prefill-layer-alloc rev-parse HEAD || true

echo "=== stage 1/3: self-test -- FULL world space + adapter mutants ==="
echo "    (516,096 worlds through the adapter against p0a_rule_totality;"
echo "     nine contract mutants; the R1-R3 read-out battery. ~3 min CPU.)"
python3 "$OUT/p0a_f2_positive_control.py" --selftest
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
python3 "$OUT/p0a_f2_positive_control.py" --run --tag "$JOBTAG" --outdir "$OUT"
RC_RUN=$?
echo "run rc=$RC_RUN"
if [ "$RC_RUN" -ne 0 ]; then
  echo "PROBE DID NOT COMPLETE -> UNDETERMINED (MEASUREMENT ABSENT)."
  echo "Do not score this job. Do not report any statement from it."
  exit 21
fi

echo "=== stage 3/3: scoring against the pre-registered rule ==="
# ★harness-layer audit G6. Until 2026-08-23 this stage had NO rc check, so a
#   scoring exception ended the job at exit 0 with a traceback in the .txt and
#   NO p0a_f2_verdict_<jobid>.json at all. That defect is already on record for
#   R0 (PROJECT_STATUS.md:4375 caveat (c), smid_l0_run.sbatch:86, left
#   unrepaired) and it was inherited here -- lesson #71, a fork inherits the
#   original's hardwiring. `pipefail` alone does not help: `tee` succeeds.
python3 "$OUT/p0a_f2_positive_control.py" --analyze "$OUT/p0a_f2_raw_${JOBTAG}.json" \
        --tag "$JOBTAG" --outdir "$OUT" | tee "$OUT/p0a_f2_verdict_${JOBTAG}.txt"
RC_AN=${PIPESTATUS[0]}
echo "analyze rc=$RC_AN"
if [ "$RC_AN" -ne 0 ]; then
  echo "SCORING FAILED -> no verdict exists for this job. This is a MEASUREMENT"
  echo "condition, not a result. Do not quote the .txt (it holds a traceback)."
  exit 22
fi
if [ ! -s "$OUT/p0a_f2_verdict_${JOBTAG}.json" ]; then
  echo "SCORING PRODUCED NO .json -- the only citable artefact is missing."
  exit 22
fi

echo "############# done job=$JOBTAG #############"
echo "★ CITE p0a_f2_verdict_${JOBTAG}.json ONLY. The .txt above is a console"
echo "  transcript and is NOT citable (gate #56, R0 defect N1)."
echo "REMINDER: every outcome of this probe is a TOOL-VALIDITY / LABEL-IDENTITY"
echo "statement. PRESERVED licenses no performance claim, closes no gate, and"
echo "does NOT answer 구멍 C or R4 (prereg sec0.3). LOST is a reason to"
echo "re-examine the decode-side SM-split narrative, not a performance result."
echo "A capture or replay failure is NOT a confinement verdict."
