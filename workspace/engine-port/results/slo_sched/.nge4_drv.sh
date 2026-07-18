#!/bin/bash
# Raise d24/d16/slo from n=1 to n>=4 so the length-norm-SLO reanalysis rests on robust
# baselines (user: "먼저 검증 강화"). Same VALID varying-trace bench (rate 3<->12), 3 rounds.
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
JOBS=()
for m in d24 d16 slo; do for r in 44 45 46; do JOBS+=("$m $r"); done; done
IDS=""
for e in "${JOBS[@]}"; do
  set -- $e; m=$1; r=$2
  j=""
  while [ -z "$j" ]; do
    q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
    if [ "$q" -lt 4 ]; then
      j=$(sbatch sharegpt_vary_bench.sbatch $m $r 3 12 2>&1 | grep -oE "[0-9]+$")
      [ -n "$j" ] && echo "$(date +%H:%M) [$m rep$r] = $j"
    fi
    [ -z "$j" ] && sleep 60
  done
  IDS="$IDS $j"
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "===== n>=4 BASELINE CAMPAIGN DONE ====="
for j in $IDS; do grep -h "SGPTV_RESULT" sgptv_*_${j}.out 2>/dev/null; done
echo "DONE_NGE4"
