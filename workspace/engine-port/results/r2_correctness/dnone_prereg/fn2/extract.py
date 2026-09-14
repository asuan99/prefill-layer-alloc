import json, collections, os, sys
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908534"
out={}
for boot in ["L1","TD1","L2","TD2","DN1"]:
    recs=[]
    with open(f"{D}/tel_{boot}.jsonl") as f:
        for i,line in enumerate(f):
            d=json.loads(line)
            if d.get("event")!="runtime_snapshot": continue
            if "gpu_mem_peak_allocated_b" not in d: continue
            recs.append(dict(
                line=i,
                ep=d.get("gpu_mem_peak_epoch"),
                peak=d.get("gpu_mem_peak_allocated_b"),
                alloc=d.get("gpu_mem_allocated_b"),
                resv=d.get("gpu_mem_reserved_b"),
                si=d.get("sample_index"),
                pbs=d.get("prefill_active_batch_size"),
                dbs=d.get("decode_batch_size"),
                drbs=d.get("decode_running_batch_size"),
                phase=d.get("phase"),
                tf=d.get("trace_forced"),
                t=d.get("timestamp_monotonic_s"),
                chunk=d.get("prefill_chunk_progress"),
                ctxmax=d.get("context_max"),
                arch=d.get("architecture"),
                guard=d.get("worker_grad_guard"),
            ))
    out[boot]=recs
with open("/tmp/claude-100018302/-scratch-ehmoon-whlee/1d13f22f-029f-454a-9ec0-68a0e6ef664e/scratchpad/fn2/recs.json","w") as f:
    json.dump(out,f)
for b,r in out.items():
    eps=[x["ep"] for x in r]
    print(b, "n=",len(r), "ep range", min(eps), max(eps), "distinct eps", len(set(eps)),
          "monotone_ep", all(eps[i]<=eps[i+1] for i in range(len(eps)-1)),
          "guard", sorted({x["guard"] for x in r}), "arch", sorted({x["arch"] for x in r}))
