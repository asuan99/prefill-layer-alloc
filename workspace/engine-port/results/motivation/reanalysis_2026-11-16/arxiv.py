import json,glob,math,sys,statistics as st
from collections import defaultdict
def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
SLO={"strict":(1000,40),"primary":(3000,60),"loose":(5000,100)}
cells=defaultdict(list)
for p in glob.glob(sys.argv[1]+"/raw/runs/*.json"):
    d=json.load(open(p))
    if d["spec"]["target"]!="arxiv_summ": continue
    arm="VLLM" if d["engine"]=="vllm" else ("PDMUX" if "bundle" in d["arm"] else "SG_MIXED")
    cells[(arm,d["spec"]["rate"])].append(d)
last=None
for key in sorted(cells):
    runs=cells[key]; reqs=[r for d in runs for s in d["segments"] for r in s["requests"]]
    if key[0]!=last:
        last=key[0]; print("\n##",key[0],"| n/셀",runs[0]["spec"]["n_requests"],"seeds",len(runs))
        print("rate    n err | strict att/ttftOnly/itlOnly/both | primary | loose | seed별 strict | ttft p50/p95/max | in p50 | out p50 | reqITLp95 p50 | step | gapfrac | gaplen")
    ok=[r for r in reqs if not r.get("error") and r.get("ttft_ms") is not None]
    row=[]
    for k,(T,I) in SLO.items():
        tf=itf=both=0
        for r in ok:
            a=r["ttft_ms"]>T; itl=r.get("token_itl_ms") or []; b=bool(itl) and pct(itl,0.95)>I
            if a and b: both+=1
            elif a: tf+=1
            elif b: itf+=1
        n=len(reqs); row.append(f"{(len(ok)-tf-itf-both)/n:.2f}/{tf/n:.2f}/{itf/n:.2f}/{both/n:.2f}")
    def att(d):
        rq=[r for s in d["segments"] for r in s["requests"]]
        g=sum(1 for r in rq if not r.get("error") and r.get("ttft_ms") is not None and r["ttft_ms"]<=1000 and (not r.get("token_itl_ms") or pct(r["token_itl_ms"],.95)<=40))
        return round(g/len(rq),2)
    tt=[r["ttft_ms"] for r in ok]; allitl=[x for r in ok for x in (r.get("token_itl_ms") or [])]
    step=st.median(allitl); gaps=[x for x in allitl if x>2*step]
    p95s=[pct(r["token_itl_ms"],0.95) for r in ok if r.get("token_itl_ms")]
    print(f"{key[1]:6.3f} {len(reqs):3d} {len(reqs)-len(ok):2d} | "+" | ".join(row)+f" | {sorted(att(d) for d in runs)} | {pct(tt,.5):6.0f}/{pct(tt,.95):6.0f}/{max(tt):6.0f} | {st.median(r['input_tokens'] for r in ok):6.0f} | {st.median(r['observed_output_tokens'] for r in ok):4.0f} | {st.median(p95s):6.1f} | {step:5.1f} | {len(gaps)/len(allitl):.3f} | {st.median(gaps) if gaps else float('nan'):6.1f}")
# lowest rate: TTFT vs input length
print("\n## 최저 도착률에서 입력 길이 구간별 TTFT (ms) p50 / 1 s 초과 비율")
for arm in sorted({k[0] for k in cells}):
    r0=min(k[1] for k in cells if k[0]==arm)
    reqs=[r for d in cells[(arm,r0)] for s in d["segments"] for r in s["requests"] if r.get("ttft_ms") is not None]
    for lo,hi in ((0,2000),(2000,4000),(4000,6000),(6000,8000),(8000,12000),(12000,99999)):
        b=[r["ttft_ms"] for r in reqs if lo<=r["input_tokens"]<hi]
        if b: print(f"  {arm:8s} rate {r0} in[{lo},{hi}) n={len(b):3d} ttft p50 {st.median(b):6.0f}  >1s {sum(x>1000 for x in b)/len(b):.2f}")
