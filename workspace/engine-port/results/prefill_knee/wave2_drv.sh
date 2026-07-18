#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/prefill_knee
while [ "$(squeue -j 857371 -h -o '%i' 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date +%H:%M) wave1 (L=256..2048) done"
J2=$(sbatch --array=0-3 --export=ALL,OFF=4 knee2d_wide.sbatch 2>&1 | grep -oE "[0-9]+$")
echo "$(date +%H:%M) wave2 (L=4096..32768) = $J2"
while [ "$(squeue -j $J2 -h -o '%i' 2>/dev/null | wc -l)" -gt 0 ]; do sleep 60; done
echo "ALL_WIDE_DONE wave1=857371 wave2=$J2"
