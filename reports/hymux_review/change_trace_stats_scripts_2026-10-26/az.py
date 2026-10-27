import sys, pandas as pd, numpy as np
d = sys.argv[1]
def load(p):
    df = pd.read_csv(p, usecols=[0,1,2], dtype={1:'int32',2:'int32'})
    df.columns=['ts','inp','out']
    df['ts']=pd.to_datetime(df['ts'], format='mixed', utc=True)
    return df
def q(s, ps=(5,50,95,99)): return " ".join(f"p{p}={np.percentile(s,p):.3g}" for p in ps)
res={}
for tag in sys.argv[2:]:
    df=load(f"{d}/{tag}.csv"); res[tag]=df
    dur=(df.ts.max()-df.ts.min()).total_seconds()
    print(f"\n=== {tag}: n={len(df)} span={df.ts.min()}..{df.ts.max()} ({dur/3600:.2f} h) mean rate={len(df)/dur:.3f}/s")
    print(" input:", q(df.inp), " mean", round(df.inp.mean()))
    print(" output:", q(df.out), " mean", round(df.out.mean()))
    t=df.set_index('ts')
    for w in ['1s','10s','1min','10min','1h']:
        c=t.inp.resample(w).count()
        if len(c)<5: continue
        sec=pd.Timedelta(w).total_seconds()
        r=c/sec
        print(f" rate[{w}]: mean={r.mean():.3g} {q(r)} max={r.max():.3g} peak/mean={r.max()/r.mean():.2f} p95/p5={np.percentile(r,95)/max(np.percentile(r,5),1e-9):.2f}")
        if w in ('1min','10min'):
            nz=r[r>0]; ratio=(nz/nz.shift(1)).dropna()
            ratio=np.maximum(ratio,1/ratio)
            print(f"   adjacent-window rate ratio (max/min): {q(ratio)} frac>=2x={np.mean(ratio>=2):.3f} frac>=4x={np.mean(ratio>=4):.4f}")
    # long fraction
    for thr in [2048, 4096, int(np.percentile(df.inp,70))]:
        g=t.inp.resample('10min').agg(lambda s: np.mean(s>=thr) if len(s)>=20 else np.nan).dropna()
        h=t.inp.resample('1h').agg(lambda s: np.mean(s>=thr) if len(s)>=50 else np.nan).dropna()
        print(f" long(inp>={thr}) overall={np.mean(df.inp>=thr):.3f}; 10min frac {q(g)} min={g.min():.3f} max={g.max():.3f}; 1h frac min={h.min():.3f} max={h.max():.3f}")
    # hourly median input
    hm=t.inp.resample('1h').median().dropna()
    print(f" hourly median input: min={hm.min()} max={hm.max()}  hourly mean output min={t.out.resample('1h').mean().min():.1f} max={t.out.resample('1h').mean().max():.1f}")
# mix
if len(res)==2:
    a,b=list(res)
    ca=res[a].set_index('ts').inp.resample('1h').count(); cb=res[b].set_index('ts').inp.resample('1h').count()
    j=pd.concat([ca,cb],axis=1).fillna(0); sh=j.iloc[:,0]/(j.sum(axis=1))
    sh=sh[j.sum(axis=1)>100]
    print(f"\n share of {a} in combined per hour: {q(sh)} min={sh.min():.3f} max={sh.max():.3f}")
