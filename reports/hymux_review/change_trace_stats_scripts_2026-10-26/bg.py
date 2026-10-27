import sys, pandas as pd, numpy as np
df=pd.read_csv(sys.argv[1]); df.columns=['t','model','inp','out','tot','log']
df=df[df.out>0]
print("n",len(df),"span days %.1f"%(df.t.max()/86400)); print(df.groupby(['model','log']).size())
for key,g in [('ALL',df)]+list(df.groupby(['model','log'])):
    if len(g)<20000: continue
    ts=pd.to_datetime(g.t,unit='s'); s=pd.Series(1,index=ts)
    print(f"\n== {key} n={len(g)} inp p50={g.inp.median():.0f} p95={g.inp.quantile(.95):.0f} out p50={g.out.median():.0f} p95={g.out.quantile(.95):.0f}")
    for w in ['1min','10min','1h']:
        r=s.resample(w).count()/pd.Timedelta(w).total_seconds()
        print(f"  rate[{w}] mean={r.mean():.3g} p50={r.median():.3g} p95={r.quantile(.95):.3g} p99={r.quantile(.99):.3g} max={r.max():.3g} peak/mean={r.max()/r.mean():.1f} frac_zero={np.mean(r==0):.2f}")
    r=s.resample('1min').count().astype(float)
    for win in ['15min','1h','3h']:
        rr=(r.rolling(win).max()/(r.rolling(win).mean())).dropna()
        print(f"  1-min max / rolling-{win} mean: p50={rr.median():.2f} p95={rr.quantile(.95):.2f}")
    inp=pd.Series(g.inp.values,index=ts)
    for thr in [1024,2048]:
        lf=inp.resample('1h').agg(lambda x: np.mean(x>=thr) if len(x)>=50 else np.nan).dropna()
        print(f"  long(inp>={thr}) overall={np.mean(g.inp>=thr):.3f} hourly p5={lf.quantile(.05):.3f} p50={lf.median():.3f} p95={lf.quantile(.95):.3f}")
# share API vs conv per day
ts=pd.to_datetime(df.t,unit='s'); api=pd.Series((df.log=='API log').astype(int).values,index=ts)
d=api.resample('1D').mean(); h=api.resample('1h').agg(lambda x: x.mean() if len(x)>=50 else np.nan).dropna()
print("\nAPI share daily p5 %.2f p50 %.2f p95 %.2f; hourly p5 %.2f p95 %.2f"%(d.quantile(.05),d.median(),d.quantile(.95),h.quantile(.05),h.quantile(.95)))
