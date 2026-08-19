#!/usr/bin/env python3
"""Cache per-file request-level ITL-p95 vectors for the survey audit."""
import json, math, sys, pickle
from pathlib import Path
ROOT = Path("/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port")
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "benchmarks"))
from pdmux_eval.analyze import load_bench_serving_rounds, percentile
out = {}
for d in sorted(RES.iterdir()):
    if not d.is_dir(): continue
    for p in d.rglob("*.jsonl"):
        if "telemetry" in p.name: continue
        try:
            reqs, dur = load_bench_serving_rounds(p)
        except Exception:
            continue
        if not reqs: continue
        out[str(p.relative_to(RES))] = [
            percentile(r.token_itl_ms, .95) if r.token_itl_ms else math.inf for r in reqs]
pickle.dump(out, open(Path(__file__).parent/"itl95_cache.pkl","wb"))
print("cached", len(out), "files")
