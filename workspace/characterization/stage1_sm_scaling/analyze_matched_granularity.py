"""
analyze_matched_granularity.py — matched-granularity SSM-vs-Attn comparison (Phase 2).

Re-derives the asymmetry table (corrected report's 2.4–4.1× story) under MATCHED
granularity: both SSM and Attention measured in prefill_chunk_tokens chunks with
cumulative state (results/stage1_v2/{ssm,attn}_chunked_*.csv).  Uses the single
saturation_point() implementation.

Outputs:
  (A) matched-granularity asymmetry table per (seq,bs): ssm vs attn sm_sat,
      14→108 speedup, and an explicit verdict — asymmetry MAINTAINED / REDUCED /
      ELIMINATED relative to the prior full-seq-attn result.
  (B) granularity effect on attention: chunked attn vs the existing full-seq attn
      (results/stage1/attn_scaling_*.csv) — how much chunking advances attn
      saturation.

This script is analysis-only (no GPU). Run after run_ssm_chunked_v2_sweep.py and
run_attn_chunked_sweep.py have produced the stage1_v2 CSVs.

  python stage1_sm_scaling/analyze_matched_granularity.py --model zamba2
"""

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import argparse
from pathlib import Path

import pandas as pd

from shared import sweep_spec as spec


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # skip the leading "# value_kind:" comment line
    return pd.read_csv(path, comment="#")


def _sat_and_speedup(group: pd.DataFrame):
    """Return (sm_sat, lat@min_sm, lat@max_sm, speedup) for one (seq,bs) group."""
    g = group.sort_values("sm_count")
    sat = spec.saturation_point(g)
    lo = g.iloc[0]
    hi = g.iloc[-1]
    lat_lo = float(lo["latency_ms"])
    lat_hi = float(hi["latency_ms"])
    speedup = lat_lo / lat_hi if lat_hi > 0 else float("nan")
    return sat, lat_lo, lat_hi, speedup


def asymmetry_table(ssm: pd.DataFrame, attn: pd.DataFrame) -> list[str]:
    lines = ["## (A) Matched-granularity SSM vs Attn asymmetry", ""]
    lines.append("| seq | bs | ssm sm_sat | attn sm_sat | ssm 14→108× | attn 14→108× | verdict |")
    lines.append("|----:|---:|:---------:|:----------:|:-----------:|:------------:|:--------|")

    keys = sorted(set(map(tuple, ssm[["seq_len", "batch_size"]].values))
                  & set(map(tuple, attn[["seq_len", "batch_size"]].values)))
    maintained = reduced = eliminated = 0
    for seq, bs in keys:
        sg = ssm[(ssm.seq_len == seq) & (ssm.batch_size == bs)]
        ag = attn[(attn.seq_len == seq) & (attn.batch_size == bs)]
        if len(sg) < 2 or len(ag) < 2:
            continue
        s_sat, _, _, s_sp = _sat_and_speedup(sg)
        a_sat, _, _, a_sp = _sat_and_speedup(ag)
        # Asymmetry = attn saturates LATER (needs more SM) than ssm.
        if s_sat is None or a_sat is None:
            verdict = "n/a"
        elif a_sat > s_sat:
            verdict = "MAINTAINED (attn later)"; maintained += 1
        elif a_sat == s_sat:
            verdict = "REDUCED (equal sat)"; reduced += 1
        else:
            verdict = "ELIMINATED (ssm later)"; eliminated += 1
        lines.append(
            f"| {seq} | {bs} | {s_sat} | {a_sat} | {s_sp:.2f} | {a_sp:.2f} | {verdict} |"
        )

    lines += ["",
              f"**Summary:** MAINTAINED={maintained}  REDUCED={reduced}  "
              f"ELIMINATED={eliminated} (of {maintained+reduced+eliminated} comparable cells).",
              ""]
    return lines


def granularity_effect(attn_chunked: pd.DataFrame, attn_full: pd.DataFrame | None) -> list[str]:
    lines = ["## (B) Granularity effect on Attention (chunked vs full-seq)", ""]
    if attn_full is None:
        lines += ["_full-seq attn CSV (results/stage1/attn_scaling_*) not found — skipped._", ""]
        return lines
    lines.append("| seq | bs | full-seq sm_sat | chunked sm_sat | Δ (chunked advances by) |")
    lines.append("|----:|---:|:--------------:|:--------------:|:------------------------|")
    # normalise full-seq column names (sm_ratio present there)
    keys = sorted(set(map(tuple, attn_chunked[["seq_len", "batch_size"]].values)))
    for seq, bs in keys:
        cg = attn_chunked[(attn_chunked.seq_len == seq) & (attn_chunked.batch_size == bs)]
        fg = attn_full[(attn_full.seq_len == seq) & (attn_full.batch_size == bs)]
        if len(cg) < 2 or len(fg) < 2:
            continue
        c_sat = spec.saturation_point(cg)
        f_sat = spec.saturation_point(fg)
        if c_sat is None or f_sat is None:
            delta = "n/a"
        else:
            d = f_sat - c_sat
            delta = f"{d:+d} SM" + (" (earlier)" if d > 0 else (" (later)" if d < 0 else " (same)"))
        lines.append(f"| {seq} | {bs} | {f_sat} | {c_sat} | {delta} |")
    lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser(description="Matched-granularity SSM/Attn analysis.")
    p.add_argument("--model", default="zamba2",
                   choices=["zamba2", "falcon_h1", "nemotron_h"])
    p.add_argument("--v2-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "stage1_v2")
    p.add_argument("--stage1-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "stage1")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent.parent.parent.parent
                   / "reports" / "matched_granularity.md")
    args = p.parse_args()

    ssm = _read(args.v2_dir / spec.canonical_filename("ssm_chunked", args.model))
    attn = _read(args.v2_dir / spec.canonical_filename("attn_chunked", args.model))
    if ssm is None or attn is None:
        sys.exit(
            f"Missing v2 CSV(s) in {args.v2_dir} for {args.model}. "
            f"Run run_ssm_chunked_v2_sweep.py and run_attn_chunked_sweep.py first."
        )

    # Schema diff-0 check (Phase 2 checklist).
    diff = set(ssm.columns) ^ set(attn.columns)
    schema_note = ("schema diff 0 ✓" if not diff
                   else f"SCHEMA DIFF: {sorted(diff)}")

    # full-seq attn (existing) for the granularity-effect table.
    attn_full = None
    for tag in (spec.device_name(), "a100-sxm4-80gb", "a100_sxm4_80gb"):
        cand = args.stage1_dir / f"attn_scaling_{args.model}_{tag}.csv"
        if cand.exists():
            attn_full = _read(cand)
            break

    lines = [f"# Matched-granularity analysis — {args.model}", "",
             f"device: {spec.device_name()} · chunk: {spec.prefill_chunk_tokens()} · "
             f"saturation threshold: {spec.saturation_threshold()}",
             f"column check: {schema_note}", ""]
    lines += asymmetry_table(ssm, attn)
    lines += granularity_effect(attn, attn_full)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ written: {args.out}")


if __name__ == "__main__":
    main()
