import json,glob,math,sys,statistics as st
from collections import defaultdict
def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
target=sys.argv[1]
cells=defaultdict(list)
for root in sys.argv[2:]:
    for p in glob.glob(root+"/raw/runs/*.json"):
        d=json.load(open(p))
        if d["spec"]["target"]!=target: continue
        arm="VLLM" if d["engine"]=="vllm" else ("PDMUX" if "bundle" in d["arm"] else "SG_MIXED")
        cells[(arm,d["spec"]["rate"])]+=[r for s in d["segments"] for r in s["requests"]]
def ok(r): return not r.get("error") and r.get("ttft_ms") is not None
def p95(r): return pct(r["token_itl_ms"],.95) if r.get("token_itl_ms") else 0
def tpot(r): return st.fmean(r["token_itl_ms"]) if r.get("token_itl_ms") else 0
C={ "proj(3s,ITLp95 60)":lambda r: r["ttft_ms"]<=3000 and p95(r)<=60,
    "3s,TPOT 60":lambda r: r["ttft_ms"]<=3000 and tpot(r)<=60,
    "DistServe13B(.25s,TPOT100)":lambda r: r["ttft_ms"]<=250 and tpot(r)<=100,
    "semiPD8B(.3s,TPOT150)":lambda r: r["ttft_ms"]<=300 and tpot(r)<=150,
    "Bullet(3ms/tok,150)":lambda r: r["ttft_ms"]<=3.0*r["input_tokens"] and tpot(r)<=150,
    "Bullet(1.5ms/tok,175)":lambda r: r["ttft_ms"]<=1.5*r["input_tokens"] and tpot(r)<=175,
    "TaiChi(4s,TPOT70)":lambda r: r["ttft_ms"]<=4000 and tpot(r)<=70,
    "semiPD8B-LB(2.25s,130)":lambda r: r["ttft_ms"]<=2250 and tpot(r)<=130,
    "MuxWise8B(TBTp95 50)":lambda r: p95(r)<=50}
use=[k for k in C if (target=="arxiv_summ")==(k in("Bullet(1.5ms/tok,175)","TaiChi(4s,TPOT70)","semiPD8B-LB(2.25s,130)")) or k in("proj(3s,ITLp95 60)","3s,TPOT 60","MuxWise8B(TBTp95 50)")]
last=None
for key in sorted(cells):
    rq=cells[key]
    if key[0]!=last:
        last=key[0]; print("\n##",target,key[0]); print("rate   | "+" | ".join(use)+" | TPOT mean p50/p95 | allTokITL p99 | in p50")
    g=[r for r in rq if ok(r)]
    al=[x for r in g for x in (r.get("token_itl_ms") or [])]
    tp=[tpot(r) for r in g]
    print(f"{key[1]:6.3f} | "+" | ".join(f"{sum(1 for r in g if C[k](r))/len(rq):.2f}" for k in use)+f" | {pct(tp,.5):5.1f}/{pct(tp,.95):5.1f} | {pct(al,.99):6.1f} | {st.median(r['input_tokens'] for r in g):.0f}")
