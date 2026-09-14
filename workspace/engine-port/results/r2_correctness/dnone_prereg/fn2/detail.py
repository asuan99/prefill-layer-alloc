import json,collections
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908534"
GiB=1024**3
for b in ["DN1","TD1","TD2","L1","L2"]:
    by=collections.defaultdict(list)
    with open(f"{D}/tel_{b}.jsonl") as f:
        for line in f:
            d=json.loads(line)
            if d.get("event")!="runtime_snapshot" or "gpu_mem_peak_epoch" not in d: continue
            by[d["gpu_mem_peak_epoch"]].append(d)
    for ep in (38,39):
        xs=by.get(ep,[])
        if not xs: print(b,ep,"MISSING"); continue
        last=xs[-1]
        print(f"{b} ep{ep}: n={len(xs):4d} dur={xs[-1]['timestamp_monotonic_s']-xs[0]['timestamp_monotonic_s']:.3f}s "
              f"peak_last={last['gpu_mem_peak_allocated_b']} ({last['gpu_mem_peak_allocated_b']/GiB:.4f}G) "
              f"alloc_last={last['gpu_mem_allocated_b']/GiB:.4f}G resv_last={last['gpu_mem_reserved_b']/GiB:.4f}G "
              f"pbs={last['prefill_active_batch_size']} dbs={last['decode_batch_size']} drbs={last.get('decode_running_batch_size')} "
              f"kvtot={last.get('kv_total_occupancy')} chunk={last.get('prefill_chunk_progress')}")
    # next epoch start after 39
    print(f"   {b} max epoch = {max(by)}  last snapshot t = {by[max(by)][-1]['timestamp_monotonic_s']:.3f}")
