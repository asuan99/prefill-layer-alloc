#!/bin/bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
E="R_SHORT=4 N_SHORT=80 R_LONG=0.6 N_LONG=14 L_LONG=8000"
declare -a TOSUB=("slo 7" "lff 7")
newids=""
# submit remaining rep7 jobs as QOS slots free
while [ ${#TOSUB[@]} -gt 0 ]; do
  q=$(squeue -u ehmoon -h -o "%i" 2>/dev/null | wc -l)
  if [ "$q" -lt 4 ]; then
    set -- ${TOSUB[0]}; mode=$1; rep=$2
    jid=$(env $E sbatch --export=ALL lff_bench.sbatch pdmux_slo $mode $rep 2>&1 | grep -oE "[0-9]+$")
    if [ -n "$jid" ]; then echo "submitted rep$rep $mode = $jid"; newids="$newids $jid"; TOSUB=("${TOSUB[@]:1}"); fi
  fi
  sleep 30
done
# wait for ALL rep5/6/7 target jobs to finish
ALL="849524 849525 849526 849579 849580 849581 849582 $newids"
while :; do
  q=$(squeue -j $(echo $ALL|tr ' ' ',') -h -o "%i" 2>/dev/null | wc -l)
  [ "$q" -eq 0 ] && break; sleep 30
done
echo "=== ALL REPS DONE. aggregating rep5/6/7 (intermediate load, cudagraph ON) ==="
python3 - <<'PY'
import glob,re,collections
rows=collections.defaultdict(list)  # mode -> list of (gp, short_p99, long_p99, short_good, long_good)
for f in glob.glob("slo_lff_*.out"):
    t=open(f).read()
    m=re.search(r"LFF_RESULT tag=(\S+) \| overall_goodput=([\d.]+) \| SHORT n=\d+ good=(\d+) TTFT_p50=[\d.]+ p99=([\d.]+) \| LONG n=\d+ good=(\d+) TTFT_p50=[\d.]+ p99=([\d.]+)",t)
    if not m: continue
    tag=m.group(1)
    rp=re.search(r"_rep(\d)_",tag)
    if not rp or rp.group(1) not in "567": continue
    if "d24" in tag: mode="d24"
    elif "_slo_rep" in tag: mode="slo"
    elif "_lff_rep" in tag: mode="lff"
    else: continue
    rows[mode].append((float(m.group(2)),int(m.group(3)),float(m.group(4)),int(m.group(5)),float(m.group(6))))
def ms(x): 
    import statistics; return (statistics.mean(x), min(x), max(x))
print(f"{'mode':5} {'n':>2} {'goodput mean[min-max]':>26} {'SHORT p99 mean[rng]':>22} {'LONG p99 mean[rng]':>22}")
for mode in ["d24","slo","lff"]:
    r=rows.get(mode,[])
    if not r: print(f"{mode}: no data"); continue
    gp=[x[0] for x in r]; sp=[x[2] for x in r]; lp=[x[4] for x in r]
    import statistics
    print(f"{mode:5} {len(r):>2} {statistics.mean(gp):>10.3f}[{min(gp):.2f}-{max(gp):.2f}]      {statistics.mean(sp):>6.2f}[{min(sp):.1f}-{max(sp):.1f}]      {statistics.mean(lp):>6.2f}[{min(lp):.1f}-{max(lp):.1f}]")
PY
echo "DONE_AGG"
