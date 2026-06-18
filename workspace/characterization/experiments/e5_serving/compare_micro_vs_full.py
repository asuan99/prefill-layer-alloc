"""Compare E5 microbench (v2) vs real-prefill (full) on the IDENTICAL sweep grid.

The full-mode run uses the *same* sweep axes as v2 (decode_batch, context, the
prefill×decode 2×2 cells, green_ctx fracs); the only treatment is prefill fidelity
(GEMM-inclusive ssm_full/attn_full, optional multi-chunk / prefill-batch). This
script joins the two CSVs on the shared keys and reports whether the three v2
conclusions survive real prefill:

  C1 (partition) : max(two_stream/green_ctx) stays < 1.0 ?         (green_ctx still loses)
  C2 (overlap)   : speedup_vs_seq(two_stream) micro vs full        (does the ~2x shrink?)
  C3 (window)    : speedup_vs_seq by decode_batch, micro vs full   (still closes with batch?)
  A5             : decode_inflation_pct micro vs full, per backend

GPU-free (reads CSVs only). Safe to run before the full sweep exists — it then
reports which full CSVs are missing and exits 0.

Usage:
  python -m experiments.e5_serving.compare_micro_vs_full [--results-dir DIR] [--models ...]
Outputs (per model): results_v2/e5/compare_micro_full_{model}_{device}.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
if _CHAR not in sys.path:
    sys.path.insert(0, _CHAR)

KEY = ["prefill_layer", "decode_layer", "decode_batch", "context_len", "backend"]


def _load(path):
    import pandas as pd
    df = pd.read_csv(path, skiprows=1)               # skip the value_kind comment line
    df.columns = [c.split("__")[0] for c in df.columns]
    return df


def _ratio_two_over_green(df):
    """max/min of two_stream_ms / green_ctx_ms per cell (>=1.0 => green_ctx wins)."""
    import pandas as pd
    cellkey = ["prefill_layer", "decode_layer", "decode_batch", "context_len"]
    piv = df.pivot_table(index=cellkey, columns="backend", values="concurrent_ms",
                         aggfunc="first")
    if "two_stream" not in piv or "green_ctx" not in piv:
        return None
    piv = piv.dropna(subset=["two_stream", "green_ctx"])
    r = piv["two_stream"] / piv["green_ctx"]
    return {"min": float(r.min()), "max": float(r.max()),
            "n_ge_1": int((r >= 1.0).sum()), "n": int(len(r))}


def compare_model(micro_path, full_path, out_dir: Path):
    import pandas as pd
    m, f = _load(micro_path), _load(full_path)
    name = os.path.basename(micro_path).replace("serving_coexec_", "").replace(".csv", "")
    print(f"\n### {name}")
    fmeta = f[["prefill_mode", "prefill_batch", "n_chunks"]].iloc[0].to_dict() if "prefill_mode" in f else {}
    print(f"  full config: {fmeta}")

    # ---- C1: green_ctx vs two_stream, micro vs full ----
    rm, rf = _ratio_two_over_green(m), _ratio_two_over_green(f)
    if rm and rf:
        print(f"  C1 partition  two_stream/green_ctx max: micro={rm['max']:.4f}({rm['n_ge_1']}/{rm['n']}≥1)"
              f"  full={rf['max']:.4f}({rf['n_ge_1']}/{rf['n']}≥1)"
              f"  -> green_ctx {'STILL loses' if rf['max'] < 1.0 else 'WINS some cells (review!)'} in full")

    # ---- C2/C3: two_stream speedup micro vs full, joined per cell ----
    mts = m[m.backend == "two_stream"][KEY + ["speedup_vs_seq", "concurrent_ms"]]
    fts = f[f.backend == "two_stream"][KEY + ["speedup_vs_seq", "concurrent_ms"]]
    j = mts.merge(fts, on=KEY, suffixes=("_micro", "_full"))
    j["speedup_delta"] = j["speedup_vs_seq_full"] - j["speedup_vs_seq_micro"]

    # C2: best cell (pf=ssm,dec=ssm) at smallest db
    best = j[(j.prefill_layer == "ssm") & (j.decode_layer == "ssm")].sort_values("decode_batch")
    if len(best):
        lo = best.iloc[0]
        print(f"  C2 overlap    pf=ssm×dec=ssm @db{int(lo.decode_batch)}: "
              f"micro {lo.speedup_vs_seq_micro:.2f}x -> full {lo.speedup_vs_seq_full:.2f}x "
              f"(Δ{lo.speedup_delta:+.2f})  [2x shrink under real prefill?]")
        # C3: window curve micro vs full
        print("  C3 window (pf=ssm×dec=ssm) speedup micro->full by db:")
        print("     " + "  ".join(f"db{int(r.decode_batch)}:{r.speedup_vs_seq_micro:.2f}->{r.speedup_vs_seq_full:.2f}"
                                  for _, r in best.iterrows()))

    # ---- A5: decode_inflation per backend, micro vs full ----
    for be in ["two_stream", "green_ctx"]:
        im = m[m.backend == be]["decode_inflation_pct"]
        iff = f[f.backend == be]["decode_inflation_pct"]
        if len(im) and len(iff):
            print(f"  A5 decode_inflation%% [{be}]: micro mean={im.mean():.1f} -> full mean={iff.mean():.1f}")

    out = out_dir / f"compare_micro_full_{name}.csv"
    j.sort_values(KEY).to_csv(out, index=False)
    print(f"  wrote per-cell comparison -> {out}")
    return j


def main():
    p = argparse.ArgumentParser(description="Compare E5 micro vs full (real-prefill) sweeps")
    p.add_argument("--results-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e5")
    p.add_argument("--models", nargs="+", default=None, help="default: all micro CSVs found")
    args = p.parse_args()

    d = args.results_dir
    micro = sorted(g for g in glob.glob(str(d / "serving_coexec_*.csv"))
                   if "serving_coexec_full_" not in os.path.basename(g))
    if args.models:
        micro = [g for g in micro if any(mm in os.path.basename(g) for mm in args.models)]
    if not micro:
        raise SystemExit(f"no micro CSVs in {d}")

    any_full = False
    for mp in micro:
        fp = mp.replace("serving_coexec_", "serving_coexec_full_")
        if not os.path.exists(fp):
            print(f"  [skip] full CSV missing for {os.path.basename(mp)} "
                  f"-> run with --prefill-mode full first")
            continue
        any_full = True
        compare_model(mp, fp, d)
    if not any_full:
        print("\nNo full-mode CSVs yet. Run:\n"
              "  env -u BASH_ENV bash experiments/slurm/submit_size_sweep.sh e5 -- "
              "--prefill-mode full --prefill-tokens 4096")


if __name__ == "__main__":
    main()
