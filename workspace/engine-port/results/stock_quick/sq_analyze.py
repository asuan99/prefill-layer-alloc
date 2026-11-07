#!/usr/bin/env python3
"""SLO-attainment tables for the stock_quick SG-stock Job vs the existing CAP1-R / CAP2 PD-mux runs.  Descriptive: different Jobs and
GPUs, 2 seeds, no CI, no comparison label.  Attainment predicate = ph_verdict.req_good (TTFT <= S and request-internal ITL p95 <= T)."""
import glob, json, os, sys
from pathlib import Path
HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "stock_baseline"))
import sb_runner as R  # noqa: E402
V = R.V
JOB = HERE / "runs/pdmux-stock-quick-6dfe05d-20261102071129/raw/runs"
CAP = HERE.parent / "cap_char/policy_runs"
SOURCES = {"W1": CAP / "pdmux-policy-cap1r-4bef3cf-20261029003539/raw/runs", "W3": CAP / "pdmux-policy-cap2-e90e233-20261029023336/raw/runs"}
TTFT = {"W1": 3000.0, "W3": 2814.84}     # CAP1-R / CAP2 SLOs, used for every arm so the arms share one SLO
ITLS = (40, 50, 60, 70, 80, 100, 150)

def reqs(d):
    return [r for s in d["segments"] for r in s["requests"]]

def cell(rs, ttft):
    itlp = [V.percentile(r["token_itl_ms"], 0.95) for r in rs if r.get("token_itl_ms")]
    itlp.sort()
    tt = sorted(r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None)
    return {"n": len(rs), "errors": sum(1 for r in rs if r.get("error")), "ttft_ok": sum(1 for r in rs if r.get("ttft_ms") is not None and r["ttft_ms"] <= ttft) / len(rs),
            "ttft_p50_ms": V.percentile(tt, 0.5), "ttft_p95_ms": V.percentile(tt, 0.95), "req_itl_p95_p50_ms": V.percentile(itlp, 0.5), "req_itl_p95_p90_ms": V.percentile(itlp, 0.9),
            "attain": {str(t): sum(1 for r in rs if V.req_good(r, ttft, t)) / len(rs) for t in ITLS}}

def load(pattern):
    out = {}
    for f in glob.glob(str(pattern)):
        d = json.load(open(f))
        out.setdefault(d["spec"]["rate"], []).append(d)
    return out

def main():
    res = {}
    for w, tag in (("W1", "W1lad"), ("W3", "W3lad")):
        sg = load(JOB / f"SQ_SG-stock_{tag}_*.json")
        res[w] = {"SG-stock": {}}
        for rate, ds in sorted(sg.items()):
            c = cell([r for d in ds for r in reqs(d)], TTFT[w])
            c["by_seed"] = {str(d["spec"]["seed"]): sum(1 for r in reqs(d) if V.req_good(r, TTFT[w], 60)) / len(reqs(d)) for d in sorted(ds, key=lambda d: d["spec"]["seed"])}
            res[w]["SG-stock"][str(rate)] = c
        pre = "CAP1R" if w == "W1" else "CAP2"
        for D in (16, 24, 34, 44, 64):
            res[w][f"D{D}"] = {str(rate): cell([r for d in ds for r in reqs(d)], TTFT[w]) for rate, ds in sorted(load(SOURCES[w] / f"{pre}_D{D}_*.json").items())}
    (HERE / "sq_tables.json").write_text(json.dumps(res, indent=1))
    for w in res:
        print(w, "rate | SG 60/80 ttft_ok itlp50 | D16 D24 D34 D44 D64 @60 | D44@80")
        for rk, c in res[w]["SG-stock"].items():
            o = [res[w][f"D{D}"].get(rk, {}).get("attain", {}).get("60") for D in (16, 24, 34, 44, 64)]
            d80 = res[w]["D44"].get(rk, {}).get("attain", {}).get("80")
            f = lambda x: "  -  " if x is None else f"{x:.3f}"
            print(f"{float(rk):7.3f} | {c['attain']['60']:.3f} {c['attain']['80']:.3f} {c['ttft_ok']:.3f} {c['req_itl_p95_p50_ms']:.1f} seeds {[round(v,2) for v in c['by_seed'].values()]} err {c['errors']} | " + " ".join(f(x) for x in o) + f" | {f(d80)}")

if __name__ == "__main__":
    main()
