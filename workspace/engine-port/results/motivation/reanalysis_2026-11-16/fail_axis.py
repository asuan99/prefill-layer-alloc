import json,glob,math,sys,statistics as st
from collections import defaultdict
def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
SLO={"strict":(1000,40),"primary":(3000,60),"loose":(5000,100)}
cells=defaultdict(list)
for root in sys.argv[1:]:
    for p in glob.glob(root+"/raw/runs/*.json"):
        d=json.load(open(p))
        key=(d["spec"]["target"],d["arm"].split("_preladder")[0][-40:],d["spec"]["rate"])
        cells[key]+=[r for s in d["segments"] for r in s["requests"]]
last=None
for key in sorted(cells):
    reqs=cells[key]
    if key[:2]!=last:
        print("\n##",key[0],key[1]); last=key[:2]
        print("rate   n  err | " + " | ".join(f"{k}: att ttftOnly itlOnly both" for k in SLO)+" | ttft p50/p95 ms | reqITLp95 p50 | step p50 | gap>2x frac | gap len p50")
    ok=[r for r in reqs if not r.get("error") and r.get("ttft_ms") is not None]
    err=len(reqs)-len(ok)
    row=[]
    for k,(T,I) in SLO.items():
        tf=itf=both=0
        for r in ok:
            a=r["ttft_ms"]>T; itl=r.get("token_itl_ms") or []; b=bool(itl) and pct(itl,0.95)>I
            if a and b: both+=1
            elif a: tf+=1
            elif b: itf+=1
        n=len(reqs); row.append(f"{(len(ok)-tf-itf-both)/n:.2f} {tf/n:.2f} {itf/n:.2f} {both/n:.2f}")
    tt=[r["ttft_ms"] for r in ok]
    p95s=[pct(r["token_itl_ms"],0.95) for r in ok if r.get("token_itl_ms")]
    allitl=[x for r in ok for x in (r.get("token_itl_ms") or [])]
    if not allitl: print(f"{key[2]:6.3f} {len(reqs)} {err}"); continue
    step=st.median(allitl); gaps=[x for x in allitl if x>2*step]
    print(f"{key[2]:6.3f} {len(reqs):3d} {err:3d} | "+" | ".join(row)+f" | {pct(tt,.5):7.0f}/{pct(tt,.95):7.0f} | {st.median(p95s):6.1f} | {step:5.1f} | {len(gaps)/len(allitl):.3f} | {st.median(gaps) if gaps else float('nan'):6.1f}")
