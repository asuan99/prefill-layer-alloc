#!/bin/bash
# Stage A n>=4 confirmation at the boundary: d44/d34 (best statics) vs bind/bind+GATE (retuned dynamic).
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
RATES="6 7 8"
JOBS=()
for r in 2 3 4; do
 JOBS+=("d44 r$r|SLO_TTFT_MS=300 SLO_ITL_MS=50|d44 $r")
 JOBS+=("d34 r$r|SLO_TTFT_MS=300 SLO_ITL_MS=50|d34 $r")
 JOBS+=("bind r$r|SLO_TTFT_MS=300 SLO_ITL_MS=50|bind $r")
 JOBS+=("bindg r$r|SLO_TTFT_MS=300 SLO_ITL_MS=50 PDMUX_SLO_FEAS_GATE=1|bind $r")
done
IDS=""
for e in "${JOBS[@]}"; do
  lbl="${e%%|*}"; rest="${e#*|}"; env="${rest%%|*}"; args="${rest##*|}"
  j=""
  while [ -z "$j" ]; do
    q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
    if [ "$q" -lt 4 ]; then
      j=$(env $env sbatch --export=ALL interactive_bench.sbatch $args "$RATES" 2>&1 | grep -oE "[0-9]+$")
      [ -n "$j" ] && echo "$(date +%H:%M) [$lbl] = $j"
    fi
    [ -z "$j" ] && sleep 60
  done
  IDS="$IDS $j"
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "===== STAGE A n>=4 DONE ====="
for j in $IDS; do grep -hE "IACT_RESULT.*rate=8 |IACT_META" iact_${j}.out 2>/dev/null; done
echo "DONE_STAGEA_N4"
