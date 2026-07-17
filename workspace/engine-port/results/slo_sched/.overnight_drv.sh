#!/bin/bash
# Feeds the QOS-limited (4) queue overnight, working through the remaining validation plan.
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
G="PDMUX_SLO_FEAS_GATE=1 PDMUX_SLO_PF_URGENCY=0.1"
N="PDMUX_SLO_PF_URGENCY=0.1"
# each entry: "<label>|<env>|<harness> <args>"
JOBS=(
  "HG1 bind+GATE rep14|$G|sharegpt_bench.sbatch bind 14 8"
  "HG1 bind+GATE rep15|$G|sharegpt_bench.sbatch bind 15 8"
  "TRAPRATE no-gate rep16|$N|sharegpt_bench.sbatch bind 16 8"
  "TRAPRATE no-gate rep17|$N|sharegpt_bench.sbatch bind 17 8"
  "HG-iso lowload gate rep18|$G|sharegpt_bench.sbatch bind 18 4"
  "HG2 varying +GATE rep5|$G|sharegpt_vary_bench.sbatch bind 5 3 12"
  "HG2 varying +GATE rep6|$G|sharegpt_vary_bench.sbatch bind 6 3 12"
)
SUB=""
for entry in "${JOBS[@]}"; do
  lbl="${entry%%|*}"; rest="${entry#*|}"; env_s="${rest%%|*}"; cmd="${rest#*|}"
  jid=""
  while [ -z "$jid" ]; do
    q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
    if [ "$q" -lt 4 ]; then
      jid=$(env $env_s sbatch --export=ALL $cmd 2>&1 | grep -oE "[0-9]+$")
      [ -n "$jid" ] && echo "$(date +%H:%M) submitted [$lbl] = $jid"
    fi
    [ -z "$jid" ] && sleep 120
  done
  SUB="$SUB $jid"
done
echo "ALL_SUBMITTED:$SUB"
# wait for everything (incl. the 4 already queued) to drain
while :; do q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l); [ "$q" -eq 0 ] && break; sleep 120; done
echo "===== OVERNIGHT RESULTS ====="
echo "--- stationary r8 (goodput | switches | FEAS-refused) ---"
for f in sgpt_*.out; do
  r=$(grep -hoE "SGPT_RESULT tag=\S+ rate=8 .*goodput=[0-9.]+" "$f" 2>/dev/null)
  [ -z "$r" ] && continue
  j="${f#sgpt_}"; j="${j%.out}"
  S=$(ls sgptsrv_*_r*_${j}.log 2>/dev/null | head -1)
  echo "$r | sw=$(grep -c 'SLO-BIND' "$S" 2>/dev/null) feas=$(grep -c 'SLO-FEAS refused' "$S" 2>/dev/null)"
done
echo "--- varying trace ---"; grep -h "SGPTV_RESULT" sgptv_*.out 2>/dev/null
echo "DONE_OVERNIGHT"
