#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
bj=""
while [ -z "$bj" ]; do
  q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
  if [ "$q" -lt 4 ]; then bj=$(sbatch sharegpt_bench.sbatch bind 1 8 2>&1 | grep -oE "[0-9]+$"); [ -n "$bj" ] && echo "bind rate8 = $bj"; fi
  sleep 30
done
ALL="852216 852217 852218 852219 $bj"
while :; do q=$(squeue -j $(echo $ALL|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 30; done
echo "=== ALL SHAREGPT (rate 8) DONE ==="
echo "--- goodput ---"; grep -h "SGPT_RESULT" sgpt_852216.out sgpt_852217.out sgpt_852218.out sgpt_852219.out sgpt_${bj}.out 2>/dev/null | sed -E 's/tag=(\w+)_rep1_r8_[0-9]+/[\1]/'
echo "--- percentiles ---"; grep -h "SGPT_PCT" sgpt_852216.out sgpt_852217.out sgpt_852218.out sgpt_852219.out sgpt_${bj}.out 2>/dev/null | sed -E 's/tag=(\w+)_rep1_r8_[0-9]+/[\1]/'
echo "DONE_SGPT_AGG"
