#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export LA=2048 OA=32 LB=256 OB=512 NPROMPT=64 ROUNDS=3
IDS=""
# static sweep (prefill-heavy d16 ... decode-heavy d44) + dynamic bind, n=2
for cfg in "d16 71" "d16 72" "d24 71" "d24 72" "d44 71" "d44 72" "d34 71" "d34 72" "bind 71" "bind 72"; do
  mode=${cfg% *}; rep=${cfg#* }
  while [ "$(squeue -u ehmoon -h -o '%i' 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
  j=$(sbatch --export=ALL he2_bench.sbatch $mode $rep 6 5 2>&1 | grep -oE "[0-9]+$")
  echo "$(date +%H:%M) [$mode rep$rep rA6B5] = $j"; IDS="$IDS $j"
  sleep 3
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "===== MIX-SWING RESULTS (phaseA=prefill-heavy in2048/o32, phaseB=decode-heavy in256/o512) ====="
for j in $IDS; do grep -h "HE2_RESULT" he2_*_${j}.log 2>/dev/null; done
echo "DONE_MIXSWING"
