#!/bin/bash
#SBATCH --job-name=smsplit_realized
#SBATCH --partition=amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/smsplit_realized/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/smsplit_realized/%x_%j.err

# E-3 -- read the realized SM counts that the green-context split primitive
# computes and the Python wrapper discards.  Seconds of GPU; the 15-minute
# wall clock is boot/import slack, not work.
#
# Deliberately does NOT run sync_engine_tree.sh.  The probe exercises no engine
# behaviour: it needs `pdmux_context.divide_sm` (a file the sync script does not
# install -- see methodology gate #33) and the `sgl_kernel` wheel in the venv.
# Running sync here would rewrite the default manifest path for no benefit, and
# the E1-b/E1-c tripwire is keyed on that baseline.

set -uo pipefail

TRACK_ROOT="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port"
OUT="${TRACK_ROOT}/results/smsplit_realized"
JOBID="${SLURM_JOB_ID:-manual}"

module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
# shellcheck disable=SC1091
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

# $HOME is at its inode quota (98,976/100,000 on 2026-08-14); any library that
# silently creates a cache under ~ dies with `OSError: [Errno 122]`.  Keep every
# cache on scratch.
export TRITON_CACHE_DIR="${OUT}/.triton_cache_${JOBID}"
export TORCHINDUCTOR_CACHE_DIR="${OUT}/.inductor_cache_${JOBID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "host=$(hostname) job=${JOBID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 "${OUT}/smsplit_realized_probe.py" \
  --device 0 \
  --out "${OUT}/smsplit_realized_${JOBID}.json"
rc=$?

echo "probe_exit=${rc}"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exit "$rc"
