import json,statistics
S=str(__import__("pathlib").Path(__file__).resolve().parent)
rows=json.load(open(f"{S}/rows.json"))
MiB=1048576; GiB=1024**3
print("k     T   | dDN-TD1 (GiB)  d/T (MiB/tok) | 0.3R(T) GiB  R(T) GiB  d/R | allocmax DN1(GiB) TD1(GiB) dalloc(GiB)")
for r in rows:
    k=r["k"]; T=r["tok"]["DN1"]; p=r["peak"]; a=r["alloc"]
    if T is None or p["DN1"] is None: continue
    d=p["DN1"]-p["TD1"]; R=T*1.2750*MiB
    da=a["DN1"]-a["TD1"]
    print(f"{k:3d} {T:6d} | {d/GiB:12.4f} {d/T/MiB:14.4f} | {0.3*R/GiB:11.4f} {R/GiB:9.4f} {d/R:6.3f} | {a['DN1']/GiB:13.4f} {a['TD1']/GiB:9.4f} {da/GiB:10.4f}")
# correlation dDN-TD1 vs T
xs=[(r["tok"]["DN1"], r["peak"]["DN1"]-r["peak"]["TD1"]) for r in rows if r["tok"]["DN1"] is not None and r["peak"]["DN1"] is not None]
import math
n=len(xs); mx=sum(x for x,_ in xs)/n; my=sum(y for _,y in xs)/n
cov=sum((x-mx)*(y-my) for x,y in xs); sx=math.sqrt(sum((x-mx)**2 for x,_ in xs)); sy=math.sqrt(sum((y-my)**2 for _,y in xs))
print("\nn matched epochs =",n, " pearson r(T, delta) =", round(cov/(sx*sy),4))
ds=[y for _,y in xs]
q=lambda p: statistics.quantiles(ds,n=100,method="inclusive")[p-1]
print("delta bytes: min",min(ds),"p25",int(q(25)),"median",int(statistics.median(ds)),"p75",int(q(75)),"max",max(ds))
print("sign: positive",sum(1 for d in ds if d>0),"zero",sum(1 for d in ds if d==0),"negative",sum(1 for d in ds if d<0))
ds2=[r["peak"]["DN1"]-r["peak"]["TD2"] for r in rows if r["tok"]["DN1"] is not None and r["peak"]["DN1"] is not None]
print("vs TD2: positive",sum(1 for d in ds2 if d>0),"neg",sum(1 for d in ds2 if d<0),"min",min(ds2),"max",max(ds2))
