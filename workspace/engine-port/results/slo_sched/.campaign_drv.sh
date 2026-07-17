#!/bin/bash
# (3)+(4) on the VALID bench (varying trace, low variance). No resource isolation needed.
#  - d44 : the "static wins" claim rests on n=1 (9.706) -> need n>=4
#  - d34 : NEVER measured; the gate ratchets to d34 and freezes -> tests "bind+GATE == static d34"
#  - bind+GATE / bind no-gate : n>=4
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
G="PDMUX_SLO_FEAS_GATE=1 PDMUX_SLO_PF_URGENCY=0.1"
N="PDMUX_SLO_PF_URGENCY=0.1"
JOBS=()
for r in 41 42 43; do JOBS+=("d44 n=$r||sharegpt_vary_bench.sbatch d44 $r 3 12"); done
for r in 41 42 43 44; do JOBS+=("d34 n=$r||sharegpt_vary_bench.sbatch d34 $r 3 12"); done
for r in 41 42 43; do JOBS+=("bind+GATE n=$r|$G|sharegpt_vary_bench.sbatch bind $r 3 12"); done
for r in 41 42 43; do JOBS+=("bind nogate n=$r|$N|sharegpt_vary_bench.sbatch bind $r 3 12"); done
IDS=""
for e in "${JOBS[@]}"; do
  lbl="${e%%|*}"; rest="${e#*|}"; ev="${rest%%|*}"; cmd="${rest#*|}"
  j=""
  while [ -z "$j" ]; do
    q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
    if [ "$q" -lt 4 ]; then
      if [ -n "$ev" ]; then j=$(env $ev sbatch --export=ALL $cmd 2>&1 | grep -oE "[0-9]+$")
      else j=$(sbatch $cmd 2>&1 | grep -oE "[0-9]+$"); fi
      [ -n "$j" ] && echo "$(date +%H:%M) [$lbl] = $j"
    fi
    [ -z "$j" ] && sleep 90
  done
  IDS="$IDS $j"
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 90; done
echo "===== (3)+(4) VARYING-TRACE CAMPAIGN RESULTS ====="
for j in $IDS; do
  r=$(grep -h "SGPTV_RESULT" sgptv_${j}.out 2>/dev/null)
  [ -z "$r" ] && continue
  S=$(ls sgptvsrv_*_${j}.log 2>/dev/null | head -1)
  echo "$r | feas=$(grep -c 'SLO-FEAS refused' "$S" 2>/dev/null) ctlcost=$(grep -h 'SLO-CTLCOST' "$S" 2>/dev/null | tail -1 | grep -oE 'mean=[0-9.]+ms max=[0-9.]+ms')"
done
echo "DONE_CAMPAIGN"
