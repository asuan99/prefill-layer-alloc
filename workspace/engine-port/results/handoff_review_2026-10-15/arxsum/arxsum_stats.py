import sys, glob, os, statistics as st, pyarrow.parquet as pq
from tokenizers import Tokenizer
D=sys.argv[1]
t=pq.read_table(os.path.join(D,"test.parquet")).to_pydict()
arts, abss = t["article"], t["abstract"]
print("docs", len(arts))
def q(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))]
for f in sorted(glob.glob(os.path.join(D,"tok","*.json"))):
    tk=Tokenizer.from_file(f); name=os.path.basename(f)[:-5]
    I=[len(e.ids) for e in tk.encode_batch(arts, add_special_tokens=False)]
    O=[len(e.ids) for e in tk.encode_batch(abss, add_special_tokens=False)]
    keep=[(i,o) for i,o in zip(I,O) if 0<i<=16384]
    Ik=[i for i,_ in keep]; Ok=[o for _,o in keep]
    print(f"\n{name}")
    print(f"  all in : mean {st.mean(I):7.0f} p50 {q(I,.5)} p90 {q(I,.9)} p99 {q(I,.99)} max {max(I)}")
    print(f"  keep<=16384: {len(keep)}/{len(I)} = {len(keep)/len(I):.1%} (empty {sum(1 for i in I if i==0)})")
    print(f"  kept in: mean {st.mean(Ik):7.0f} p10 {q(Ik,.1)} p50 {q(Ik,.5)} p90 {q(Ik,.9)} p99 {q(Ik,.99)}")
    print(f"  kept out(abstract): mean {st.mean(Ok):5.0f} p50 {q(Ok,.5)} p90 {q(Ok,.9)} p99 {q(Ok,.99)}")
    for th in (2048,4096,8192,8760,12288):
        print(f"    kept in > {th}: {sum(1 for i in Ik if i>th)/len(Ik):.1%}")
