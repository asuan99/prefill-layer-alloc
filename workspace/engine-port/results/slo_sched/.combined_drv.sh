#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
# submit bind-loose (stationary r8) when a slot frees
lj=""
while [ -z "$lj" ]; do
  q=$(squeue -u ehmoon -h -o "%i"|wc -l)
  [ "$q" -lt 4 ] && lj=$(PDMUX_SLO_PF_URGENCY=0.1 PDMUX_SLO_DWELL=10 sbatch --export=ALL sharegpt_bench.sbatch bind 2 8 2>&1 | grep -oE "[0-9]+$")
  sleep 30
done
echo "bind-loose stationary = $lj"
# wait for loose job + varying agg
while :; do
  qs=$(squeue -j $lj -h -o "%i" 2>/dev/null|wc -l)
  done_v=$(grep -c DONE_SGPTV_AGG .sgptv_drv.out 2>/dev/null)
  [ "$qs" -eq 0 ] && [ "$done_v" -ge 1 ] && break
  sleep 40
done
echo "===== BIND-LOOSE (stationary ShareGPT r8, pf_urg=0.1) ====="
grep -hE "SGPT_RESULT|SGPT_PCT" sgpt_${lj}.out 2>/dev/null
echo "loose switch count: $(grep -cE 'SLO-BIND' sgptsrv_bind_rep2_r8_${lj}.log 2>/dev/null)"
echo ""
echo "===== VARYING TRACE (rate 3<->12) ====="
sed -n '/VARYING ShareGPT/,/DONE_SGPTV_AGG/p' .sgptv_drv.out 2>/dev/null
echo "DONE_COMBINED"
