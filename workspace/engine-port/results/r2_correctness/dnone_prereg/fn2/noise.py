import json,statistics
S=str(__import__("pathlib").Path(__file__).resolve().parent)
rows=json.load(open(f"{S}/rows.json")); MiB=1048576
tdd=[]; lld=[]; sep=[]
for r in rows:
    p=r["peak"]
    if None in (p["TD1"],p["TD2"]): continue
    tdd.append((r["k"],p["TD1"]-p["TD2"]))
    lld.append((r["k"],p["L1"]-p["L2"]))
    if p["TD1"]==p["TD2"] and p["L1"]==p["L2"] and p["TD1"]!=p["L1"]:
        sep.append((r["k"], r["tok"]["TD1"], p["TD1"]-p["L1"]))
print("same-config boot noise, |TD1-TD2| over 41 epochs:")
a=[abs(x) for _,x in tdd]; print("  zero epochs:",sum(1 for x in a if x==0),"/41  max",max(a),f"({max(a)/MiB:.3f} MiB)  nonzero:",[(k,v) for k,v in tdd if v!=0])
print("same-config boot noise, |L1-L2| over 41 epochs:")
b=[abs(x) for _,x in lld]; print("  zero epochs:",sum(1 for x in b if x==0),"/41  max",max(b),f"({max(b)/MiB:.3f} MiB)  nonzero:",[(k,v) for k,v in lld if v!=0])
print("\nGATE #232 demonstration: epochs with TD1==TD2 != L1==L2 (full architecture-arm separation):")
for k,T,d in sep: print(f"  epoch {k:3d} T={T:5d}  peak(TD)-peak(L) = {d:+12d} B = {d/MiB:+8.3f} MiB")
print("  count =",len(sep),"/41")
