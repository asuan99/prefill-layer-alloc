import json,re,collections,datetime
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908534"
S=str(__import__("pathlib").Path(__file__).resolve().parent)
pat=re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Prefill batch, #new-seq: (\d+), #new-token: (\d+)")
ep=json.load(open(f"{S}/epochs.json"))
batches={}
for b in ["L1","TD1","L2","TD2","DN1"]:
    rows=[]
    for line in open(f"{D}/srv_{b}.log"):
        m=pat.search(line)
        if m:
            ts=datetime.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").timestamp()
            rows.append(dict(k=len(rows)+1, nseq=int(m.group(2)), ntok=int(m.group(3)), wall=ts))
    batches[b]=rows
    t=ep[b]
    print(f"--- {b}: batches={len(rows)} epochs(1..)={sum(1 for k in t if int(k)>=1)}")
    mism=[]
    for r in rows:
        k=str(r["k"])
        if k not in t: 
            mism.append((r["k"],r["nseq"],r["ntok"],"NO-EPOCH-SNAPSHOT")); continue
        if t[k]["pbs_max"]!=r["nseq"]:
            mism.append((r["k"],r["nseq"],r["ntok"],"pbs_max="+str(t[k]["pbs_max"]),"pbs_vals="+str(t[k]["pbs_vals"])))
    print("  nseq<->pbs_max mismatches:",len(mism))
    for m_ in mism: print("   ",m_)
json.dump(batches,open(f"{S}/batches.json","w"))
