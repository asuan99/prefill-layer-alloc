import json,sys,statistics as st
def q(a,p): a=sorted(a); return a[min(len(a)-1,int(p*len(a)))]
for f in sys.argv[1:]:
    R=[json.loads(l) for l in open(f)]
    I=[r["input_length"] for r in R]; O=[r["output_length"] for r in R]; T=[r["timestamp"] for r in R]
    dur=(max(T)-min(T))/1000
    # prefix reuse: fraction of hash blocks seen before
    seen=set(); tot=hit=0
    for r in R:
        for h in r.get("hash_ids",[]):
            tot+=1; hit+= h in seen; seen.add(h)
    print(f.split('/')[-1], "n=%d dur=%.0fs rate=%.2f/s"%(len(R),dur,len(R)/dur))
    print("  in  mean %.0f p50 %d p90 %d p99 %d max %d | >32k %.1f%% >64k %.1f%% >128k %.2f%%"%(st.mean(I),q(I,.5),q(I,.9),q(I,.99),max(I),100*sum(i>32768 for i in I)/len(I),100*sum(i>65536 for i in I)/len(I),100*sum(i>131072 for i in I)/len(I)))
    print("  out mean %.0f p50 %d p90 %d p99 %d max %d"%(st.mean(O),q(O,.5),q(O,.9),q(O,.99),max(O)))
    print("  block reuse %.1f%%  corr(in,out) %.2f"%(100*hit/max(tot,1), st.correlation(I,O)))
    # Little: decode concurrency ~ rate*out*step(20ms)
    print("  decode tokens/s demand %.0f ; prefill tokens/s demand %.0f"%(sum(O)/dur, sum(I)/dur))
