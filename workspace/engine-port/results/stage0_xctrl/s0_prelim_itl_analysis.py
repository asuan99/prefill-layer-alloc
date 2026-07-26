import json, glob, statistics as st, numpy as np, re, collections

def itls_for(arm, ctx, D):
    per_rep_p50=[]; per_rep_mean=[]; pooled=[]
    for r in (1,2,3):
        fn=f"s0_{arm}_C{ctx}_D{D}_864230_rep{r}_raw.jsonl"
        try: lines=open(fn).read().splitlines()
        except FileNotFoundError: return None
        rep=[]
        for ln in lines:
            if not ln.strip(): continue
            ts=json.loads(ln)["token_ts"]
            d=[(ts[i+1]-ts[i])*1000.0 for i in range(len(ts)-1)]
            rep+=d
        pooled+=rep
        per_rep_p50.append(np.percentile(rep,50))
        per_rep_mean.append(np.mean(rep))
    pooled=np.array(pooled)
    return dict(
      p50=np.percentile(pooled,50), p95=np.percentile(pooled,95),
      p99=np.percentile(pooled,99), mean=pooled.mean(), n=len(pooled),
      rep_p50=per_rep_p50, rep_mean=per_rep_mean,
      rep_p50_sd=st.pstdev(per_rep_p50), rep_mean_sd=st.pstdev(per_rep_mean))

arms=["T","H"]; ctxs=[4096,8192,16384]; Ds=[16,44,92,108]
print(f"{'arm':>3} {'ctx':>6} {'D':>4} | {'p50':>7} {'p95':>7} {'mean':>7} | {'repP50sd':>8} {'repMnsd':>8} | {'n':>5}")
data={}
for arm in arms:
  for ctx in ctxs:
    for D in Ds:
      s=itls_for(arm,ctx,D)
      if s is None: 
        print(f"{arm:>3} {ctx:>6} {D:>4} | MISSING"); continue
      data[(arm,ctx,D)]=s
      print(f"{arm:>3} {ctx:>6} {D:>4} | {s['p50']:7.2f} {s['p95']:7.2f} {s['mean']:7.2f} | {s['rep_p50_sd']:8.3f} {s['rep_mean_sd']:8.3f} | {s['n']:5d}")

print("\n=== SM-sensitivity: ITL(16)/ITL(92) ratio (active-split curve), D108 ref separate ===")
print(f"{'arm':>3} {'ctx':>6} | {'p50_16/92':>10} {'mean_16/92':>10} {'p50_44/92':>10} | {'D108_p50':>9}")
for arm in arms:
  for ctx in ctxs:
    d16=data[(arm,ctx,16)]; d44=data[(arm,ctx,44)]; d92=data[(arm,ctx,92)]; d108=data[(arm,ctx,108)]
    print(f"{arm:>3} {ctx:>6} | {d16['p50']/d92['p50']:10.3f} {d16['mean']/d92['mean']:10.3f} {d44['p50']/d92['p50']:10.3f} | {d108['p50']:9.2f}")
