#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export LA=2048 OA=32 LB=2048 OB=512 NPROMPT=64 ROUNDS=3
IDS=""
for cfg in "d16 81" "d24 81" "d34 81" "d44 81" "bind 81" "d16 82" "d24 82" "d34 82" "d44 82" "bind 82"; do
  mode=${cfg% *}; rep=${cfg#* }
  while [ "$(squeue -u ehmoon -h -o '%i' 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
  j=$(sbatch --export=ALL he2_bench.sbatch $mode $rep 8 5 2>&1 | grep -oE "[0-9]+$")
  echo "$(date +%H:%M) [$mode rep$rep rA8B5] = $j"; IDS="$IDS $j"; sleep 3
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "DONE_EXTREME$IDS"
