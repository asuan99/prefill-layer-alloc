"""SLO-lens comparison of prefill/decode SM backends — judge on DECODE LATENCY, not
just throughput. Answers the A5 / dynamic-partition question:

  Does a *decode-protective* split (green_ctx_protect: decode reserved its E3 floor)
  keep decode latency low (SLO) while not wrecking throughput, vs plain co-schedule
  (two_stream) and prefill-favoring static split (green_ctx@f)?

For each cell it reads two metrics already in the E5 CSV:
  • decode_inflation_pct  — SLO proxy (decode latency vs solo; lower=better)
  • speedup_vs_seq        — throughput (higher=better)

GPU-free. Run on full CSVs produced with --decode-protect:
  python -m experiments.e5_serving.analyze_slo_partition [--csv-glob ...]

IMPORTANT (honest scope): a single-cell microbench has ONE prefill + ONE decode
kernel, so two_stream already barely inflates decode (no sustained contention). The
real MuxWise/Bullet SLO benefit needs a *request stream / queue* (many concurrent
prefills pressuring decode tails) — NOT reproducible here. This script bounds the
microbench question; the queue dynamic is a separate, larger build.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
if _CHAR not in sys.path:
    sys.path.insert(0, _CHAR)

KEYC = ["prefill_layer", "decode_layer", "decode_batch", "context_len"]


def _load(path):
    import pandas as pd
    df = pd.read_csv(path, skiprows=1)
    df.columns = [c.split("__")[0] for c in df.columns]
    for col in ("decode_inflation_pct", "speedup_vs_seq", "decode_batch"):
        if col in df:
            df[col] = __import__("pandas").to_numeric(df[col], errors="coerce")
    return df


def analyze(path):
    import pandas as pd
    df = _load(path)
    name = os.path.basename(path).replace("serving_coexec_", "").replace("serving_coexec", "").replace(".csv", "")
    bks = [b for b in ["two_stream", "green_ctx", "green_ctx_protect"] if b in set(df.backend)]
    print(f"\n### {name}   backends: {bks}")
    if "green_ctx_protect" not in bks:
        print("  [no green_ctx_protect rows] → run with --decode-protect first.")
    # mean SLO (decode inflation) + mean throughput per backend
    for b in bks:
        s = df[(df.backend == b) & (df.status == "ok")]
        print(f"  {b:18} decode_inflation%% mean={s.decode_inflation_pct.mean():6.1f} max={s.decode_inflation_pct.max():6.1f}"
              f"   throughput speedup mean={s.speedup_vs_seq.mean():.3f}")
    if "green_ctx_protect" not in bks:
        return
    # per-cell: does protect beat two_stream on SLO (lower inflation) and on throughput?
    ts = df[df.backend == "two_stream"][KEYC + ["decode_inflation_pct", "speedup_vs_seq", "concurrent_ms"]]
    dp = df[df.backend == "green_ctx_protect"][KEYC + ["decode_inflation_pct", "speedup_vs_seq", "concurrent_ms"]]
    j = ts.merge(dp, on=KEYC, suffixes=("_ts", "_dp"))
    if not len(j):
        print("  (no overlapping ok cells)")
        return
    slo_win = (j.decode_inflation_pct_dp <= j.decode_inflation_pct_ts + 1.0).sum()   # protect ≤ two_stream latency
    thr_win = (j.concurrent_ms_dp < j.concurrent_ms_ts).sum()                        # protect faster wall-clock
    both = ((j.decode_inflation_pct_dp <= j.decode_inflation_pct_ts + 1.0) & (j.concurrent_ms_dp < j.concurrent_ms_ts)).sum()
    print(f"  vs two_stream over {len(j)} cells: protect wins SLO(decode-lat)={slo_win}  wins throughput={thr_win}  wins BOTH={both}")
    # show the best SLO-and-throughput cells if any
    cand = j[(j.decode_inflation_pct_dp <= j.decode_inflation_pct_ts + 1.0) & (j.concurrent_ms_dp < j.concurrent_ms_ts)]
    for _, r in cand.sort_values("concurrent_ms_dp").head(5).iterrows():
        print(f"    BOTH-win: pf={r.prefill_layer} dec={r.decode_layer} db{int(r.decode_batch)}  "
              f"infl {r.decode_inflation_pct_ts:.0f}→{r.decode_inflation_pct_dp:.0f}%  "
              f"conc {r.concurrent_ms_ts:.2f}→{r.concurrent_ms_dp:.2f}ms")


def main():
    p = argparse.ArgumentParser(description="SLO-lens analysis of decode-protective partition")
    p.add_argument("--csv-glob", default=None,
                   help="default: results_v2/e5/serving_coexec_full_*.csv")
    args = p.parse_args()
    g = args.csv_glob or os.path.join(_CHAR, "results_v2", "e5", "serving_coexec_full_*.csv")
    files = sorted(glob.glob(g))
    if not files:
        raise SystemExit(f"no CSVs match {g}")
    print("SLO-lens: green_ctx_protect (decode=E3 floor) vs two_stream — decode latency & throughput")
    print("NOTE: microbench bound only; real SLO win needs a request-stream/queue (see module docstring).")
    for f in files:
        analyze(f)


if __name__ == "__main__":
    main()
