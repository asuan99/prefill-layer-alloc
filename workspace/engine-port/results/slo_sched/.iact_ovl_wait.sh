cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
IDS="860448 860449"
while :; do q=$(squeue -j $(echo $IDS|tr ' ' ',') -h -o "%i" 2>/dev/null|wc -l); [ "$q" -eq 0 ] && break; sleep 60; done
echo "===== CHAT-SLO OVERLOAD PROBE DONE ====="
for j in $IDS; do echo "-- job $j"; grep -hE "IACT_RESULT|IACT_BUCKET|boot_ok|BOOT FAILED|IACT_META" iact_${j}.out 2>/dev/null; done
echo "DONE_OVL"
