#!/bin/bash
# Stage A: chat SLO (300/50) full policy scan at the boundary regime (rate 6,7,8).
# rep1 scan first to locate where policies split & whether it's a cliff; n>=4 follows on the winner rate.
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
RATES="6 7 8"
# label|env|mode
JOBS=(
 "d16|SLO_TTFT_MS=300 SLO_ITL_MS=50|d16"
 "d24|SLO_TTFT_MS=300 SLO_ITL_MS=50|d24"
 "d34|SLO_TTFT_MS=300 SLO_ITL_MS=50|d34"
 "d44|SLO_TTFT_MS=300 SLO_ITL_MS=50|d44"
 "slo|SLO_TTFT_MS=300 SLO_ITL_MS=50|slo"
 "bind|SLO_TTFT_MS=300 SLO_ITL_MS=50|bind"
 "bindg|SLO_TTFT_MS=300 SLO_ITL_MS=50 PDMUX_SLO_FEAS_GATE=1|bind"
)
IDS=""
for e in "${JOBS[@]}"; do
  lbl="${e%%|*}"; rest="${e#*|}"; env="${rest%%|*}"; mode="${rest##*|}"
  j=""
  while [ -z "$j" ]; do
    q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
    if [ "$q" -lt 3 ]; then
      j=$(env $env sbatch --export=ALL interactive_bench.sbatch $mode 1 "$RATES" 2>&1 | grep -oE "[0-9]+$")
      [ -n "$j" ] && echo "$(date +%H:%M) [$lbl] = $j"
    fi
    [ -z "$j" ] && sleep 60
  done
  IDS="$IDS $j"
done
echo "SUBMITTED:$IDS"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "===== STAGE A SCAN DONE ====="
for j in $IDS; do grep -hE "IACT_RESULT|IACT_META" iact_${j}.out 2>/dev/null; done
echo "DONE_STAGEA"
