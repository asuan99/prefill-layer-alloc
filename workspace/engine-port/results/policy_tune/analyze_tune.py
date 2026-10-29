#!/usr/bin/env python3
"""TUNE analysis with sealed harness (select_tune via ph_analyze.tune_pick).  Usage: analyze_tune.py <job_dir> <out_json>"""
import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cap_char/policy_harness"))
import ph_spec as P, ph_verdict as V, ph_analyze as A
job, outp = Path(sys.argv[1]), Path(sys.argv[2]); raw = job / "raw"
runs = A.load_runs(raw); pick = A.tune_pick(runs)
res = {"n_runs": len(runs), "pick": pick, "runs": {}, "configs": {}}
for r in runs:
    g = lambda ph=None: V.he0_goodput(r, P.W1_TTFT_MS, P.ITL_MS, ph)
    reqs = lambda ph=None: [q for s in r["segments"] if ph is None or s["phase"] == ph for q in s["requests"]]
    def pc(ph):
        q = reqs(ph); tt = [x["ttft_ms"] for x in q if x.get("ttft_ms") is not None]
        itl = [V.percentile(x["token_itl_ms"], .95) for x in q if x.get("token_itl_ms")]
        allitl = [v for x in q for v in x.get("token_itl_ms", [])]
        return {"n": len(q), "ttft": [V.percentile(tt, p) for p in (.5, .95, .99)], "itl_tok": [V.percentile(allitl, p) for p in (.5, .95, .99)],
                "good_frac": sum(1 for x in q if V.req_good(x, P.W1_TTFT_MS, P.ITL_MS)) / len(q), "cliff_frac": V.cliff_fraction(q, P.W1_TTFT_MS)}
    res["runs"][r["run_id"]] = {"cfg": r["arm"]["name"], "seed": r["seed"], "goodput": g(), "LO": g("LO"), "HI": g("HI"), "switch_count": r["switch_count"],
        "split_dwell": r["split_dwell"], "n_segments": len(r["segments"]), "n_req": len(reqs()), "total_duration_s": sum(s["duration_s"] for s in r["segments"]),
        "trace_hash": r["trace_hash"], "all": pc(None), "LO_stats": pc("LO"), "HI_stats": pc("HI")}
for c in sorted({v["cfg"] for v in res["runs"].values()}):
    rs = [v for v in res["runs"].values() if v["cfg"] == c]
    res["configs"][c] = {"mean_goodput": statistics.fmean(v["goodput"] for v in rs), "mean_LO": statistics.fmean(v["LO"] for v in rs), "mean_HI": statistics.fmean(v["HI"] for v in rs),
        "mean_switch": statistics.fmean(v["switch_count"] for v in rs), "per_seed": {v["seed"]: v["goodput"] for v in rs}}
by_seed = {}
for v in res["runs"].values(): by_seed.setdefault(v["seed"], set()).add(v["trace_hash"])
res["trace_identity_across_configs"] = {s: len(h) == 1 for s, h in by_seed.items()}
json.dump(res, open(outp, "w"), indent=1, sort_keys=True, default=str)
print(json.dumps(pick, default=str)[:600])
for c, v in sorted(res["configs"].items(), key=lambda x: -x[1]["mean_goodput"]): print(c, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "per_seed"}, {s: round(x, 3) for s, x in v["per_seed"].items()})
print(res["trace_identity_across_configs"])
