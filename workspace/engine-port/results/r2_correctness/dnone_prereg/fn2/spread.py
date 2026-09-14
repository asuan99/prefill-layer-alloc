import json,statistics
S="/tmp/claude-100018302/-scratch-ehmoon-whlee/1d13f22f-029f-454a-9ec0-68a0e6ef664e/scratchpad/fn2"
rows=json.load(open(f"{S}/rows.json"))
MiB=1048576; GiB=1024**3
four=["TD1","TD2","L1","L2"]
print("k     T |  4-boot(inference_mode) spread B  | sep? |  dDN-TD1 B      dDN-TD2 B      dDN-L1 B       dDN-L2 B     | d/spread")
sep=0; tot=0; ratios=[]
for r in rows:
    k=r["k"]; T=r["tok"]["DN1"]; p=r["peak"]
    vals={b:p[b] for b in four if p[b] is not None}
    if len(vals)<4: continue
    spread=max(vals.values())-min(vals.values())
    tdset={p["TD1"],p["TD2"]}; lset={p["L1"],p["L2"]}
    full = (len(tdset)==1 and len(lset)==1 and tdset!=lset)
    tot+=1
    if full: sep+=1
    if p["DN1"] is None:
        print(f"{k:3d} {str(T):>6} | {spread:14d} ({spread/MiB:8.3f} MiB) | {'YES' if full else '  -'} | DN1 epoch missing")
        continue
    ds=[p["DN1"]-p[b] for b in four]
    rat = (min(ds)/spread) if spread>0 else float('inf')
    ratios.append(rat)
    print(f"{k:3d} {T:6d} | {spread:14d} ({spread/MiB:8.3f} MiB) | {'YES' if full else '  -'} | "
          + " ".join(f"{d:15d}" for d in ds) + f" | {rat:10.1f}" if spread>0 else
          f"{k:3d} {T:6d} | {spread:14d} ({spread/MiB:8.3f} MiB) | {'YES' if full else '  -'} | "
          + " ".join(f"{d:15d}" for d in ds) + "   inf(spread=0)")
print()
print("epochs with all 4 inference_mode boots present:",tot,"  full TD-vs-L arm separation (TD1==TD2 != L1==L2):",sep)
print("min ratio min|dDN|/4boot-spread over matched epochs:", round(min(ratios),1) if ratios else None)
sp=[]
for r in rows:
    p=r["peak"]; vals=[p[b] for b in four if p[b] is not None]
    if len(vals)==4: sp.append(max(vals)-min(vals))
print("4-boot spread bytes: min",min(sp),"median",int(statistics.median(sp)),"max",max(sp), "  max MiB", round(max(sp)/MiB,3))
