#!/bin/bash
#SBATCH --job-name=re7b_e3e5
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=slurm/logs/re7b_%j.out
#SBATCH --error=slurm/logs/re7b_%j.err
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13
module load cuda/13.0.2
module load gcc/15.2.0
PROJECT_ROOT=/scratch/$USER/whlee/prefill-layer-alloc
source "$PROJECT_ROOT/bin/activate"
export TOKENIZERS_PARALLELISM=false HF_HOME="$PROJECT_ROOT/hf_cache" PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$PROJECT_ROOT/.triton_cache"     # the fix that resolved decode_ssm OSError
cd "$PROJECT_ROOT/workspace/characterization"
nvidia-smi --query-gpu=name --format=csv,noheader || true
BATCHES="1 2 4 8 16 32 64 128 256 512"

echo "########## [1/2] E3 decode floor (zamba2_7b, attn+ssm) — TRITON fix ##########"
python experiments/e3_decode_floor/run_decode_floor.py --models zamba2_7b \
    --batches $BATCHES --context-lens 4096 --layer-types attn ssm
echo "--- E3 floor ssm 행 확인 ---"
python -c "
import pandas as pd
df=pd.read_csv('results_v2/e3/decode_floor_zamba2_7b_a100_sxm4_80gb.csv',skiprows=1)
df.columns=[c.split('__')[0] for c in df.columns]
print('  layer_type별:', dict(df.layer_type.value_counts()))
"
echo "########## [2/2] E5 serving_coexec (opt+protect) — green_ctx_protect 재측정 ##########"
python experiments/e5_serving/run_serving_coexec.py --models zamba2_7b \
    --prefill-mode full --prefill-tokens 256 --chunk 256 --prefill-batch 8 \
    --context-lens 4096 --decode-batches $BATCHES \
    --decode-protect --opt-decode --output-dir results_v2/e5_sim_b8_opt
echo "--- E5 green_ctx_protect ssm ok 확인 ---"
python -c "
import pandas as pd
df=pd.read_csv('results_v2/e5_sim_b8_opt/serving_coexec_full_zamba2_7b_a100_sxm4_80gb.csv',skiprows=1)
df.columns=[c.split('__')[0] for c in df.columns]
s=df[(df.prefill_layer=='ssm')&(df.decode_layer=='ssm')&(df.backend=='green_ctx_protect')]
print('  ssm green_ctx_protect ok:', (s.status=='ok').sum(),'/',len(s))
"
echo "ALL DONE."
