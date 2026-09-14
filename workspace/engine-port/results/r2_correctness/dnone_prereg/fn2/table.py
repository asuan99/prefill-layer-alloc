import json
S="/tmp/claude-100018302/-scratch-ehmoon-whlee/1d13f22f-029f-454a-9ec0-68a0e6ef664e/scratchpad/fn2"
ep=json.load(open(f"{S}/epochs.json")); ba=json.load(open(f"{S}/batches.json"))
GiB=1024**3; MiB=1024**2
boots=["DN1","TD1","TD2","L1","L2"]
tok={b:{r["k"]:r["ntok"] for r in ba[b]} for b in boots}
nseq={b:{r["k"]:r["nseq"] for r in ba[b]} for b in boots}
print("k  tok(DN1) nseq  | peak DN1 / TD1 / TD2 / L1 / L2  (bytes) | dDN-TD1  dDN-TD2 | tokmatch")
rows=[]
for k in range(1,42):
    line={"k":k}
    line["tok"]={b:tok[b].get(k) for b in boots}
    line["nseq"]={b:nseq[b].get(k) for b in boots}
    line["peak"]={b:(ep[b].get(str(k),{}) or {}).get("peak_max") for b in boots}
    line["n"]={b:(ep[b].get(str(k),{}) or {}).get("n") for b in boots}
    line["pbs"]={b:(ep[b].get(str(k),{}) or {}).get("pbs_max") for b in boots}
    line["alloc"]={b:(ep[b].get(str(k),{}) or {}).get("alloc_max") for b in boots}
    line["resv"]={b:(ep[b].get(str(k),{}) or {}).get("resv_max") for b in boots}
    rows.append(line)
json.dump(rows,open(f"{S}/rows.json","w"))
for r in rows:
    p=r["peak"]; t=r["tok"]
    tm = (t["DN1"] is not None and t["DN1"]==t["TD1"]==t["TD2"])
    d1 = (p["DN1"]-p["TD1"]) if (p["DN1"] is not None and p["TD1"] is not None) else None
    d2 = (p["DN1"]-p["TD2"]) if (p["DN1"] is not None and p["TD2"] is not None) else None
    print(f"{r['k']:3d} {str(t['DN1']):>8} {str(r['nseq']['DN1']):>4} | "
          + " ".join(f"{(str(p[b]) if p[b] is not None else '-'):>14}" for b in boots)
          + f" | {str(d1):>12} {str(d2):>12} | {'YES' if tm else 'NO'}")
