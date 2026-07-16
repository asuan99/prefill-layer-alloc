#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
bj=""
while [ -z "$bj" ]; do
  q=$(squeue -u ehmoon -h -o "%i"|wc -l)
  [ "$q" -lt 4 ] && bj=$(sbatch sharegpt_vary_bench.sbatch bind 1 3 12 2>&1 | grep -oE "[0-9]+$")
  sleep 30
done
echo "bind = $bj"
ALL="$(cat .sgptv_jobs) $bj"
while :; do q=$(squeue -j $(echo $ALL|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 30; done
echo "=== VARYING ShareGPT (rate 3<->12) DONE ==="
for j in $ALL; do grep -hE "SGPTV_RESULT" sgptv_${j}.out 2>/dev/null; done | sed -E 's/tag=(\w+)_rep1_[^ ]*/[\1]/'
echo "--- percentiles ---"; for j in $ALL; do grep -hE "SGPTV_PCT" sgptv_${j}.out 2>/dev/null; done | sed -E 's/tag=(\w+)_rep1_[^ ]*/[\1]/'
echo "DONE_SGPTV_AGG"
