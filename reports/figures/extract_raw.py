#!/usr/bin/env python3
"""Extract plottable series straight from campaign artifacts.

Only reads; writes a single JSON so the plotting layer never re-parses logs.
Every series carries `source` (file + job) and `grade` so a figure can never
render a number without its evidence level travelling with it.

Run:  python reports/figures/extract_raw.py
Out:  reports/figures/raw_extracted.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIAGE = ROOT / "workspace/engine-port/triage"
R0C = ROOT / "workspace/engine-port/results/r0c"
OUT = Path(__file__).resolve().parent / "raw_extracted.json"

GOODPUT_RE = re.compile(
    r"rate=(?P<rate>[\d.]+)\s+good=(?P<good>\d+)/(?P<total>\d+)\s+"
    r"goodput=(?P<gp>[\d.]+)req/s\s+dur=(?P<dur>[\d.]+)s"
)
TPOT_RE = re.compile(r"Median TPOT \(ms\):\s+([\d.]+)")

# job id -> (model label, policy). p1_6_* is Nemotron-H (MODEL is in the sbatch,
# not echoed into the .out), p1_7_* echoes MODEL= itself.
P1_RUNS = {
    "p1_6_bench_one_831609.out": ("Nemotron-H-8B", "fused"),
    "p1_6_bench_one_831610.out": ("Nemotron-H-8B", "agnostic"),
    "p1_6_bench_one_831611.out": ("Nemotron-H-8B", "agnostic_v2"),
    "p1_6_bench_one_831612.out": ("Nemotron-H-8B", "layer_aware"),
    "p1_7_bench_one_832638.out": ("Granite-4.0-h-micro", "fused"),
    "p1_7_bench_one_832639.out": ("Granite-4.0-h-micro", "agnostic"),
    "p1_7_bench_one_832640.out": ("Granite-4.0-h-micro", "agnostic_v2"),
    "p1_7_bench_one_832690.out": ("Granite-4.0-h-micro", "layer_aware"),
    "p1_7_bench_one_832575.out": ("Falcon-H1-3B", "fused"),
    "p1_7_bench_one_832576.out": ("Falcon-H1-3B", "agnostic"),
    "p1_7_bench_one_832701.out": ("Zamba2-2.7B", "agnostic"),
    "p1_7_bench_one_832702.out": ("Zamba2-2.7B", "layer_aware"),
}


def parse_bench(path: Path):
    """Pull the goodput@SLO block and the per-rate median TPOT row."""
    text = path.read_text(errors="replace")
    rows = [m.groupdict() for m in GOODPUT_RE.finditer(text)]
    if not rows:
        return None
    # The TPOT summary line repeats one value per rate, in rate order.
    tpot_line = [ln for ln in text.splitlines() if ln.count("Median TPOT") > 1]
    tpots = [float(v) for v in TPOT_RE.findall(tpot_line[-1])] if tpot_line else []
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "rate": float(r["rate"]),
                "goodput": float(r["gp"]),
                "good": int(r["good"]),
                "total": int(r["total"]),
                "duration_s": float(r["dur"]),
                "tpot_median_ms": tpots[i] if i < len(tpots) else None,
            }
        )
    return out


def extract_p1():
    series = {}
    for fname, (model, policy) in P1_RUNS.items():
        path = TRIAGE / fname
        if not path.exists():
            continue
        rows = parse_bench(path)
        if rows:
            series.setdefault(model, {})[policy] = {
                "rows": rows,
                "source": f"workspace/engine-port/triage/{fname}",
                "job": fname.split("_")[-1].removesuffix(".out"),
            }
    return series


ZBLT_RE = re.compile(
    r"RESULT CTX=(?P<ctx>\d+) sm=(?P<sm>\w+).*?"
    r"attn_total=(?P<attn>[\d.]+) mamba_total=(?P<mamba>[\d.]+).*?"
    r"per-attn\(9\)=(?P<pa>[\d.]+) per-mamba\(54\)=(?P<pm>[\d.]+)"
)


def extract_decode_knee():
    """job 858811 decode-only micro. RETRACTED for ratio use -- 3 instrumentation
    defects (asymmetric buckets / `full`-first warm-up / physics-invariant
    violation). Kept so the figure can show *why* it was withdrawn."""
    rows = []
    for path in sorted(R0C.glob("deckneectx_result_C*_858811.txt")):
        for m in ZBLT_RE.finditer(path.read_text(errors="replace")):
            d = m.groupdict()
            rows.append(
                {
                    "ctx": int(d["ctx"]),
                    "sm": d["sm"],
                    "attn_total_ms": float(d["attn"]),
                    "mamba_total_ms": float(d["mamba"]),
                    "per_attn_ms": float(d["pa"]),
                    "per_mamba_ms": float(d["pm"]),
                    "source": f"workspace/engine-port/results/r0c/{path.name}",
                }
            )
    return rows


def main():
    payload = {
        "p1_4model_goodput": {
            "grade": "serving, no-cudagraph (non-operating point), n=1/cell",
            "workload": "synthetic random-ids in2000/out96, range-ratio 1.0, "
            "120 prompts, goodput@(TTFT<=3s AND TPOT<=60ms)",
            "series": extract_p1(),
        },
        "decode_knee_vs_ctx_858811": {
            "grade": "micro, no-cudagraph, CITATION-BARRED (3 instrumentation defects)",
            "note": "asymmetric buckets (attn=core only, mamba=whole mixer); "
            "`full` always first arm (warm-up biases every denominator); "
            "mamba SSD decode must be O(1) in ctx but scatters 1.98x, and "
            "full(108SM) mamba is 2.34x SLOWER than sm44 -- physically impossible.",
            "rows": extract_decode_knee(),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    n_p1 = sum(len(v) for v in payload["p1_4model_goodput"]["series"].values())
    print(f"wrote {OUT}")
    print(f"  p1 runs        : {n_p1} (models: "
          f"{sorted(payload['p1_4model_goodput']['series'])})")
    print(f"  decode-knee rows: {len(payload['decode_knee_vs_ctx_858811']['rows'])}")


if __name__ == "__main__":
    main()
