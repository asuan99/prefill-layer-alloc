import json,glob,math,sys,statistics as st
from collections import defaultdict
def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
target=sys.argv[1]; want=[float(x) for x in sys.argv[2].split(",")] if sys.argv[2]!="all" else None
cells=defaultdict(list)
for root in sys.argv[3:]:
    for p in glob.glob(root+"/raw/runs/*.json"):
        d=json.load(open(p))
        if d["spec"]["target"]!=target: continue
        arm="vLLM" if d["engine"]=="vllm" else ("PD-mux" if "bundle" in d["arm"] else "SG mixed")
        cells[(arm,d["spec"]["rate"])]+=[r for s in d["segments"] for r in s["requests"]]
def ms(v,u=1.0,nd=0): return f"{st.fmean(v)/u:.{nd}f} ± {st.pstdev(v)/u:.{nd}f}"
print("| arm | 도착률 | n | TTFT s (≤3 s) | 요청 ITL p95 ms (≤60) | 평균 TPOT ms (≤60) | 최대 간격 ms | 둘 다(프로젝트) |")
print("|---|---|---|---|---|---|---|---|")
for key in sorted(cells):
    if want and not any(abs(key[1]-w)<0.01 for w in want): continue
    rq=cells[key]; g=[r for r in rq if not r.get("error") and r.get("ttft_ms") is not None and r.get("token_itl_ms")]
    n=len(rq)
    tt=[r["ttft_ms"] for r in g]; p=[pct(r["token_itl_ms"],.95) for r in g]; tp=[st.fmean(r["token_itl_ms"]) for r in g]; mx=[max(r["token_itl_ms"]) for r in g]
    f=lambda c: f"{sum(c)/n:.2f}"
    print(f"| {key[0]} | {key[1]:.2f} | {n} | {ms(tt,1000,2)} ({f([x<=3000 for x in tt])}) | {ms(p,1,1)} ({f([x<=60 for x in p])}) | {ms(tp,1,1)} ({f([x<=60 for x in tp])}) | {ms(mx)} | {f([a<=3000 and b<=60 for a,b in zip(tt,p)])} |")
