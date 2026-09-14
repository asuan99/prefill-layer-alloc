import json,collections
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908534"
S=str(__import__("pathlib").Path(__file__).resolve().parent)
batches=json.load(open(f"{S}/batches.json"))
for b in ["L1","TD1","L2","TD2","DN1"]:
    first={}
    with open(f"{D}/tel_{b}.jsonl") as f:
        for line in f:
            d=json.loads(line)
            if d.get("event")!="runtime_snapshot" or "gpu_mem_peak_epoch" not in d: continue
            e=d["gpu_mem_peak_epoch"]
            if e not in first: first[e]=d.get("timestamp_s")
    deltas=[]
    for r in batches[b]:
        k=r["k"]
        if k in first and first[k] is not None:
            deltas.append(round(first[k]-r["wall"],2))
    print(b,"epoch_first_snapshot_wall - batch_log_wall (s): min",min(deltas),"max",max(deltas),
          "n",len(deltas), "frac in [-1.0, 2.0]", sum(1 for x in deltas if -1.0<=x<=2.0)/len(deltas))
    print("   outliers:",[ (r['k'],round(first[r['k']]-r['wall'],2)) for r in batches[b] if r['k'] in first and not -1.0<=first[r['k']]-r['wall']<=2.0][:10])

print("=== residual after removing constant clock offset (offset = min delta, log has 1 s resolution) ===")
for b in ["L1","TD1","L2","TD2","DN1"]:
    first={}
    with open(f"{D}/tel_{b}.jsonl") as f:
        for line in f:
            d=json.loads(line)
            if d.get("event")!="runtime_snapshot" or "gpu_mem_peak_epoch" not in d: continue
            e=d["gpu_mem_peak_epoch"]
            if e not in first: first[e]=d.get("timestamp_s")
    ds=[(r["k"], first[r["k"]]-r["wall"]) for r in batches[b] if r["k"] in first]
    off=min(x for _,x in ds)
    res=[(k, round(x-off,2)) for k,x in ds]
    print(b,"residual min",min(x for _,x in res),"max",max(x for _,x in res),
          "n<=2.0s:",sum(1 for _,x in res if x<=2.0),"/",len(res))
    print("   worst:",sorted(res,key=lambda t:-t[1])[:5])
