import json, collections
S="/tmp/claude-100018302/-scratch-ehmoon-whlee/1d13f22f-029f-454a-9ec0-68a0e6ef664e/scratchpad/fn2"
recs=json.load(open(f"{S}/recs.json"))
summ={}
for b,r in recs.items():
    by=collections.OrderedDict()
    for x in r: by.setdefault(x["ep"],[]).append(x)
    nonmono=[]
    tab={}
    for ep,xs in by.items():
        pk=[x["peak"] for x in xs]
        mono=all(pk[i]<=pk[i+1] for i in range(len(pk)-1))
        if not mono: nonmono.append(ep)
        tab[ep]=dict(n=len(xs), peak_max=max(pk), peak_first=pk[0], peak_last=pk[-1],
                     pbs_max=max((x["pbs"] or 0) for x in xs),
                     pbs_vals=sorted({x["pbs"] for x in xs}),
                     alloc_max=max((x["alloc"] or 0) for x in xs),
                     resv_max=max((x["resv"] or 0) for x in xs),
                     t0=xs[0]["t"], t1=xs[-1]["t"],
                     si0=xs[0]["si"], si1=xs[-1]["si"],
                     dbs_max=max((x["dbs"] or 0) for x in xs),
                     ctxmax=max((x["ctxmax"] or 0) for x in xs))
    summ[b]=tab
    missing=[e for e in range(0,max(by)+1) if e not in by]
    print(b,"epochs present",len(by),"missing",missing,"non-monotone-within-epoch epochs",nonmono[:10], len(nonmono),
          "peak_last==peak_max for all?", all(tab[e]["peak_last"]==tab[e]["peak_max"] for e in tab))
json.dump({b:{str(k):v for k,v in t.items()} for b,t in summ.items()}, open(f"{S}/epochs.json","w"))
