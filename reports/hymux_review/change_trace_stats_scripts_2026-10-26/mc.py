import sys, json, numpy as np, pandas as pd
for p in sys.argv[1:]:
    df=pd.read_json(p,lines=True); t=df.timestamp/1000.0  # ms
    print(f"\n== {p.split('/')[-1]} n={len(df)} span={t.max():.0f}s ({t.max()/3600:.2f}h) mean rate={len(df)/t.max():.2f}/s")
    print("  input p50 %d p95 %d mean %d ; output p50 %d p95 %d"%(df.input_length.median(),df.input_length.quantile(.95),df.input_length.mean(),df.output_length.median(),df.output_length.quantile(.95)))
    s=pd.Series(df.input_length.values,index=pd.to_datetime(t,unit='s'))
    for w in ['10s','1min','5min']:
        r=s.resample(w).count()/pd.Timedelta(w).total_seconds()
        print(f"  rate[{w}] p5={r.quantile(.05):.2f} p50={r.median():.2f} p95={r.quantile(.95):.2f} max={r.max():.2f} max/min={r.max()/max(r.min(),1e-9):.2f}")
    for thr in [8192,16384]:
        lf=s.resample('5min').agg(lambda x: np.mean(x>=thr))
        print(f"  long(>={thr}) overall {np.mean(df.input_length>=thr):.3f} 5min min {lf.min():.3f} max {lf.max():.3f}")
