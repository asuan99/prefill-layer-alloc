import json,glob,math,sys,statistics as st
from collections import defaultdict
def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
cells=defaultdict(list)
for p in glob.glob(sys.argv[1]+"/runs/*.json"):
    d=json.load(open(p)); cells[(d["arm"]["D"],d["spec"]["rate"])]+=[r for s in d["segments"] for r in s["requests"]]
r0=cells[min(cells)][0]; print("request keys:",sorted(r0.keys()))
allr=[r for v in cells.values() for r in v]
print("input p50/mean",st.median(r["input_tokens"] for r in allr),round(st.fmean(r["input_tokens"] for r in allr)),"output p50/mean",st.median(r["observed_output_tokens"] for r in allr),round(st.fmean(r["observed_output_tokens"] for r in allr)))
SLO_T=2814.84
def ms(v,u=1.0,nd=0): return f"{st.fmean(v)/u:.{nd}f} ± {st.pstdev(v)/u:.{nd}f}"
want=[float(x) for x in sys.argv[2].split(",")]
print("| 도착률 | D | TTFT s (≤2.81 s) | 평균 TPOT ms | 요청 ITL p95 ms (≤60) | 최대 간격 ms | 생성 시간 s | 완료 시각 s | 둘 다 | 1k입력토큰당 TTFT ms p50 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for rate in want:
  for D in (16,24,34,44,64):
    key=next((k for k in cells if k[0]==D and abs(k[1]-rate)<0.005),None)
    if not key: continue
    rq=cells[key]; g=[r for r in rq if not r.get("error") and r.get("ttft_ms") is not None and r.get("token_itl_ms")]; n=len(rq)
    tt=[r["ttft_ms"] for r in g]; p=[pct(r["token_itl_ms"],.95) for r in g]; tp=[st.fmean(r["token_itl_ms"]) for r in g]; mx=[max(r["token_itl_ms"]) for r in g]
    gen=[sum(r["token_itl_ms"]) for r in g]; e2e=[a+b for a,b in zip(tt,gen)]
    f=lambda c: f"{sum(c)/n:.2f}"
    per=st.median(r["ttft_ms"]/r["input_tokens"]*1000 for r in g)
    print(f"| {key[1]:.2f} | D{D} | {ms(tt,1000,2)} ({f([x<=SLO_T for x in tt])}) | {ms(tp,1,1)} | {ms(p,1,1)} ({f([x<=60 for x in p])}) | {ms(mx)} | {ms(gen,1000,1)} | {ms(e2e,1000,1)} | {f([a<=SLO_T and b<=60 for a,b in zip(tt,p)])} | {per:.0f} |")
