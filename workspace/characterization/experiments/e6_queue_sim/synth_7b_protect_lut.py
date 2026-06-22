"""Synthesize a green_ctx_protect LUT for zamba2_7b from MEASURED green_ctx data.

WHY THIS EXISTS (read before trusting the output):
The 2.7b layer_aware result (run_layer_aware) needs a `green_ctx_protect` backend:
decode reserved its E3 saturation floor, prefill = the rest. For 7B that backend is
absent — every cell is `failed:no_room(decode_floor=None)` because **E3 decode-floor was
never measured for 7B** (no decode_floor_zamba2_7b CSV; new GPU measurement is blocked:
broken mamba_ssm/CUDA13 locally, no A100-SXM4 in slurm).

What 7B DOES have is the measured `green_ctx` f-sweep: prefill∥decode on a FIXED SM split,
at f=0.5 (prefill 54 / decode 54) and f=0.7 (prefill 76 / decode 32). We use those measured
partition stream times to stand in for green_ctx_protect, choosing the floor per (layer_type,
decode_batch) as the smallest measured decode_sm that keeps decode within TOL of solo.

HONEST LIMITATIONS (these BOUND the result, they don't invalidate it):
  - Measured 7B fracs are only {0.5, 0.7} ⇒ decode floor is snapped to {32, 54};
    prefill is NEVER starved below 54 SMs. The real 2.7b LUT (E3-driven) starves prefill
    down to 14 SMs at high batch — which is exactly where agnostic_protect bleeds and
    layer_aware wins big. So this 7B construction UNDER-states agnostic starvation ⇒ it is a
    CONSERVATIVE LOWER BOUND on layer_aware's advantage at 7B.
  - If decode@54SM is still >TOL inflated (true floor > 54, common for 7B attn), we reserve
    54 anyway (the most protection the measured grid offers) and flag it. Same reservation is
    applied to attn under BOTH agnostic and layer_aware, so the A/B contrast stays fair.

Output: results_v2/e5_sim_b8_opt/serving_coexec_full_zamba2_7b_a100_sxm4_80gb_synth.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))

TOL = 0.15  # decode within 15% of solo == "protected / saturated"
# default source = 7B decode-protect run (has green_ctx f-sweep); --model swaps it
_SRC_DIR = {"zamba2_7b": "e5_dp", "zamba2_2.7b": "e5_dp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zamba2_7b")
    ap.add_argument("--src", default=None, help="override source LUT path")
    a = ap.parse_args()
    SRC = Path(a.src) if a.src else (Path(_CHAR) / "results_v2" / _SRC_DIR.get(a.model, "e5_dp")
                                     / f"serving_coexec_full_{a.model}_a100_sxm4_80gb.csv")
    OUT = (Path(_CHAR) / "results_v2" / "e5_sim_b8_opt"
           / f"serving_coexec_full_{a.model}_a100_sxm4_80gb_synth.csv")
    with open(SRC) as fh:
        comment = fh.readline().rstrip("\n")          # "# value_kind: ..."
        header = fh.readline().rstrip("\n")           # col__kind names
    raw_cols = header.split(",")
    short = [c.split("__")[0] for c in raw_cols]

    df = pd.read_csv(SRC, skiprows=1)
    df.columns = [c.split("__")[0] for c in df.columns]
    for c in ("decode_batch", "prefill_sm", "decode_sm", "prefill_stream_ms",
              "decode_stream_ms", "solo_prefill_ms", "solo_decode_ms", "concurrent_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    same = df[df.prefill_layer == df.decode_layer].copy()
    ok = same[same.status == "ok"]

    out_rows = []
    # 1) keep measured two_stream rows verbatim
    ts = ok[ok.backend == "two_stream"]
    for _, r in ts.iterrows():
        out_rows.append(r[short].to_dict())

    # 2) build green_ctx_protect rows from measured green_ctx (best floor per cell)
    gc = ok[ok.backend == "green_ctx"]
    log = []
    for lt in ("attn", "ssm"):
        sub = gc[gc.prefill_layer == lt]
        for b in sorted(sub.decode_batch.dropna().unique()):
            cells = sub[sub.decode_batch == b].sort_values("decode_sm")  # 32 then 54
            if cells.empty:
                continue
            solo = float(cells.iloc[0].solo_decode_ms)
            chosen = None
            for _, c in cells.iterrows():           # smallest decode_sm first (max prefill room)
                if float(c.decode_stream_ms) <= (1 + TOL) * solo:
                    chosen = c
                    break
            flag = "saturated"
            if chosen is None:                       # even max decode_sm still inflated → reserve it
                chosen = cells.iloc[-1]              # largest decode_sm (54)
                flag = f"under_protected(infl={100*(chosen.decode_stream_ms/solo-1):.0f}%)"
            row = chosen[short].to_dict()
            row["backend"] = "green_ctx_protect"
            out_rows.append(row)
            log.append((lt, int(b), int(chosen.decode_sm), int(chosen.prefill_sm),
                        round(float(chosen.decode_stream_ms), 4), round(solo, 4), flag))

    out = pd.DataFrame(out_rows)[short]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write(comment + "\n")
        f.write(",".join(raw_cols) + "\n")
        out.to_csv(f, index=False, header=False)

    print(f"wrote {OUT}  ({len(out)} rows: {len(ts)} two_stream + {len(log)} green_ctx_protect)")
    print(f"\nfloor choice per (layer,batch)  [TOL={TOL}]:")
    print(f"{'lt':>5}{'b':>5}{'dec_sm':>8}{'pf_sm':>7}{'dec_ms':>9}{'solo':>9}  flag")
    for lt, b, dsm, psm, dms, solo, flag in log:
        print(f"{lt:>5}{b:>5}{dsm:>8}{psm:>7}{dms:>9}{solo:>9}  {flag}")


if __name__ == "__main__":
    main()
