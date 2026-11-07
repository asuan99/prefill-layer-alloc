#!/usr/bin/env python3
"""Post-hoc (not registered) view: add 'largest token gap <= 1 s' to the attainment predicate.  Descriptive.  Writes sq_maxitl.json."""
import json
import sq_analyze as A
V = A.V
LIM = 1000.0
def good(r, ttft, itl=60):
    return V.req_good(r, ttft, itl) and max(r.get("token_itl_ms") or [0.0]) <= LIM
def nogap(r):
    return not r.get("error") and r.get("ttft_ms") is not None and max(r.get("token_itl_ms") or [0.0]) <= LIM
out = {}
for w, tag, pre in (("W1", "W1lad", "CAP1R"), ("W3", "W3lad", "CAP2")):
    arms = {"SG-stock": A.load(A.JOB / f"SQ_SG-stock_{tag}_*.json")}
    for D in (16, 24, 34, 44, 64):
        arms[f"D{D}"] = A.load(A.SOURCES[w] / f"{pre}_D{D}_*.json")
    out[w] = {}
    for a, by in arms.items():
        out[w][a] = {}
        for rate, ds in sorted(by.items()):
            rs = [r for d in ds for r in A.reqs(d)]
            mx = sorted(max(r.get("token_itl_ms") or [0.0]) for r in rs)
            bys = {str(d["spec"]["seed"]): sum(good(r, A.TTFT[w]) for r in A.reqs(d)) / len(A.reqs(d)) for d in sorted(ds, key=lambda d: d["spec"]["seed"])}
            out[w][a][str(rate)] = {"base": sum(V.req_good(r, A.TTFT[w], 60) for r in rs) / len(rs), "with_gap": sum(good(r, A.TTFT[w]) for r in rs) / len(rs),
                                    "gap_ok": sum(nogap(r) for r in rs) / len(rs), "max_gap_p50_ms": V.percentile(mx, .5), "max_gap_p100_ms": mx[-1], "good": sum(good(r, A.TTFT[w]) for r in rs), "n": len(rs), "by_seed": bys}
json.dump(out, open(A.HERE / "sq_maxitl.json", "w"), indent=1)
for w in out:
    print(w, "with_gap: rate | SG | D16 D24 D34 D44 D64 || SG base | SG gap_ok | SG maxgap p50 | D-all max gap")
    for rk, c in out[w]["SG-stock"].items():
        ds = [out[w][f"D{D}"][rk] for D in (16, 24, 34, 44, 64)]
        print(f"{float(rk):7.3f} | {c['with_gap']:.3f} ({c['good']}/{c['n']}) {[round(v,2) for v in c['by_seed'].values()]} | " + " ".join(f"{x['with_gap']:.3f}" for x in ds) + f" || {c['base']:.3f} | {c['gap_ok']:.3f} | {c['max_gap_p50_ms']:.0f} | " + " ".join(f"{x['max_gap_p100_ms']:.0f}" for x in ds))
