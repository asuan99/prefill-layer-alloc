import json,collections,sys
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908534"
GiB=1024**3
for b in ["DN1","TD1"]:
    by=collections.defaultdict(list)
    with open(f"{D}/tel_{b}.jsonl") as f:
        for line in f:
            d=json.loads(line)
            if d.get("event")!="runtime_snapshot" or "gpu_mem_peak_epoch" not in d: continue
            by[d["gpu_mem_peak_epoch"]].append(d)
    for ep in (23,39):
        xs=by.get(ep,[])
        if not xs: print(b,ep,"MISSING"); continue
        t0=xs[0]["timestamp_monotonic_s"]
        print(f"### {b} epoch {ep} (n={len(xs)})  dt  peakG  allocG  resvG  pbs dbs chunk tf")
        for x in xs[:26]:
            print(f"   {x['timestamp_monotonic_s']-t0:7.3f} {x['gpu_mem_peak_allocated_b']/GiB:8.4f} {x['gpu_mem_allocated_b']/GiB:8.4f} {x['gpu_mem_reserved_b']/GiB:8.4f} {x['prefill_active_batch_size']:3d} {x['decode_batch_size']:3d} {x.get('prefill_chunk_progress')} {x.get('trace_forced')}")
        if len(xs)>26: print("   ...")
