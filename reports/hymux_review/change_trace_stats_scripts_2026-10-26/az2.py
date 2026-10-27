import sys, pandas as pd, numpy as np
d=sys.argv[1]
S={}
for tag in ['az24_code','az24_conv']:
    df=pd.read_csv(f"{d}/{tag}.csv",usecols=[0,1,2]); df.columns=['ts','inp','out']
    df['ts']=pd.to_datetime(df.ts,format='mixed',utc=True); t=df.set_index('ts'); S[tag]=t
    r=t.inp.resample('1min').count()/60
    h=t.inp.resample('1h').count()/3600
    day=h.groupby(h.index.date).agg(['max','min'])
    print(tag,"per-day hourly max/min:",(day['max']/day['min']).round(2).tolist())
    wk=h.groupby(h.index.dayofweek).mean(); print("  mean rate by weekday(0=Mon):",wk.round(1).tolist())
    hod=h.groupby(h.index.hour).mean(); print("  mean rate by UTC hour:",hod.round(1).tolist())
    for win in ['5min','15min','1h','3h','6h']:
        rr=r.rolling(win).max()/r.rolling(win).min()
        rr=rr.replace([np.inf],np.nan).dropna()
        print(f"  1-min rate max/min within rolling {win}: p50={rr.median():.2f} p95={rr.quantile(.95):.2f} max={rr.max():.2f}")
    # time from trough to peak each day
    thr=np.percentile(t.inp,70)
    lf=t.inp.resample('10min').agg(lambda s: np.mean(s>=2048) if len(s)>=20 else np.nan)
    for win in ['30min','1h','3h','6h']:
        rr=(lf.rolling(win).max()-lf.rolling(win).min()).dropna()
        print(f"  long(>=2048) frac range within rolling {win}: p50={rr.median():.3f} p95={rr.quantile(.95):.3f} max={rr.max():.3f}")
    hl=lf.groupby(lf.index.hour).mean(); print("  long frac by UTC hour:",hl.round(3).tolist())
a,b=S['az24_code'],S['az24_conv']
lo=max(a.index.min(),b.index.min()); hi=min(a.index.max(),b.index.max())
ca=a.loc[lo:hi].inp.resample('1h').count(); cb=b.loc[lo:hi].inp.resample('1h').count()
sh=ca/(ca+cb); print("overlap",lo,hi,"code share hourly: min %.3f p50 %.3f max %.3f"%(sh.min(),sh.median(),sh.max()))
# combined long frac
comb=pd.concat([a.loc[lo:hi].inp,b.loc[lo:hi].inp]).sort_index()
lf=comb.resample('10min').agg(lambda s: np.mean(s>=2048)); h=comb.resample('1h').agg(lambda s: np.mean(s>=2048))
print("combined long(>=2048) 10min: p5 %.3f p50 %.3f p95 %.3f min %.3f max %.3f; hourly min %.3f max %.3f"%(lf.quantile(.05),lf.median(),lf.quantile(.95),lf.min(),lf.max(),h.min(),h.max()))
tok=comb.resample('1min').sum()/60; print("combined input tokens/s per-min: p5 %.0f p50 %.0f p95 %.0f max %.0f"%(tok.quantile(.05),tok.median(),tok.quantile(.95),tok.max()))
