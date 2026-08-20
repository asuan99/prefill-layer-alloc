#!/usr/bin/env python3
"""S0(a) -- within-boot half-split residual of the C2 ratio r.

GPU spend: 0.  Re-scores telemetry that is already committed.

WHAT THIS IS
------------
The audit's recommendation C-g is to alternate `--conc` INSIDE a boot so that
Delta_batch becomes a within-boot paired contrast instead of a between-boot one.
`design_g13_stats.paired_conc()` can only report the ceiling (perfect
cancellation) and the floor (none), because the within-boot residual SD of r
"HAS NOT BEEN MEASURED -- that is exactly S0(a)".  This measures it.

NO NEW ESTIMAND: r is computed by `s8_c2r_score`'s own bracket logic, verbatim
(same guard, same state-match, same per-boot median of matched ITL intervals).
The only thing added is a REPORTING AXIS: each boot's accepted intervals are
split into an early and a late half by snapshot index (which is monotone in
time), and r is formed within each half.

★WHAT THIS IS NOT (rev3 §3, and the audit's criterion-5 finding)
    S0(a) is NOT a gate.  It was removed from gate #13's critical path by the
    rev3 repair, and as a gate it would have been null-accepting (methodology
    lesson #20): six boots cannot resolve the cancellation factor f to +-0.05,
    and no threshold for it exists in any document.  This script therefore
    reports a MEASUREMENT WITH ITS UNCERTAINTY and refuses to emit a pass/fail.

★DIRECTION OF THE BOUND (state before reading the output)
    A half-split separates the two halves by the LARGEST within-boot lag
    available, while an alternating design pairs ADJACENT segments.  So the
    half-split residual is an UPPER bound on the residual an alternating
    contrast would carry, hence a LOWER bound on cancellation.  Separately, a
    half has half the intervals, so the half medians are noisier than full-boot
    medians -- that sampling noise inflates the raw within-boot number and is
    estimated (interval bootstrap) and reported alongside, never silently
    subtracted from the headline.

FALSIFIABLE POSITIVE CONTROLS
    PC1  per-boot medians recomputed here must match the committed
         C2R_RESULTS_2026-08-16.json per_boot_median exactly.
    PC2  the between-boot SD of r recomputed here must match
         design_g13_stats.pairsd()'s sigma_boot (the number rev3 prices the
         whole Ha8 leg on) to 3 decimals.
    Neither is an identity: PC1 fails if the half-split axis perturbs the
    interval set, PC2 fails if the boot pairing here differs from the one the
    design inputs use.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import pickle
import random
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = 3.0
BATCH = 16
BOOT_SEED = 20260820
BOOT_DRAWS = 1000


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _load("s8_c2r_score", "s8_c2r_score.py")
G13 = _load("design_g13_stats", "design_g13_stats.py")
LEG_SM = {"d16": 16, "d92": 92}


def _boot_cells(boots):
    """-> {arm: {blk: {cell: [ (snap_i, itl_ms), ... ]}}} at the canonical cell."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for bt in boots:
        if bt["cell"] not in LEG_SM:
            continue
        want = LEG_SM[bt["cell"]]
        v = [(t[5], t[2]) for t in bt["intervals"]
             if t[0] == want and t[1] == BATCH and t[3] <= GUARD]
        if v:
            out[bt["arm"]][bt["blk"]][bt["cell"]] = sorted(v)
    return out


def _half_split(v):
    """Early/late halves by snapshot index -- the boot's own time axis."""
    idx = [i for i, _ in v]
    cut = st.median(idx)
    early = [x for i, x in v if i <= cut]
    late = [x for i, x in v if i > cut]
    return early, late


def _ratio_se_bootstrap(a, b, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Sampling SE of median(a)/median(b) from resampling INTERVALS.

    This is the noise a half inherits purely from having half the samples; it
    is NOT the temporal residual we are after, which is why it is reported
    separately rather than subtracted from the headline.
    """
    rng = np.random.default_rng(seed)
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra = np.median(rng.choice(aa, size=(draws, aa.size), replace=True), axis=1)
    rb = np.median(rng.choice(bb, size=(draws, bb.size), replace=True), axis=1)
    rs = ra / rb
    return float(rs.std(ddof=1)), float(rs.mean())


CACHE = os.path.join(HERE, ".s0a_cells_cache.pkl")


def _cells_cached():
    """The scoring pass parses ~171 MB of telemetry; cache its (small) output.

    The cache holds ONLY the extracted (snapshot_index, itl_ms) lists at the
    canonical cell, so PC1/PC2 still re-derive every reported number from it.
    """
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    boots = SC.score(SC.discover("c2r", ctx="1024"), keep_slack=True)
    cells = {a: {b: dict(c) for b, c in d.items()} for a, d in _boot_cells(boots).items()}
    with open(CACHE, "wb") as f:
        pickle.dump(cells, f)
    return cells


def run() -> dict:
    cells = _cells_cached()
    committed = json.load(open(os.path.join(HERE, "C2R_RESULTS_2026-08-16.json")))
    ps = G13.pairsd()

    res = {"what": "S0(a): within-boot half-split residual of r",
           "gpu_spend": 0, "is_a_gate": False,
           "guard_s": GUARD, "batch": BATCH,
           "estimand": "unchanged -- s8_c2r_score bracket logic, verbatim",
           "arms": {}, "positive_controls": []}

    pc1_bad = []
    for arm in ("M8", "Ha8"):
        blks = sorted(b for b, d in cells[arm].items() if len(d) == 2)
        rows, r_full, within_pairs = [], [], []
        for blk in blks:
            d16, d92 = cells[arm][blk]["d16"], cells[arm][blk]["d92"]
            m16 = st.median([x for _, x in d16])
            m92 = st.median([x for _, x in d92])
            r_full.append(m16 / m92)
            # PC1: the committed per-boot medians
            for cell, val in (("d16", m16), ("d92", m92)):
                key = f"{arm}/{cell}/SM{LEG_SM[cell]}/b{BATCH}"
                ref = committed["cells"][key]["per_boot_median"]
                hit = [v for k, v in ref.items() if k.endswith(f"blk{blk}")]
                if not hit or abs(hit[0] - val) > 1e-9:
                    pc1_bad.append((arm, cell, blk, val, hit[0] if hit else None))
            e16, l16 = _half_split(d16)
            e92, l92 = _half_split(d92)
            if min(len(e16), len(l16), len(e92), len(l92)) < 30:
                rows.append({"blk": blk, "skipped": "half with <30 intervals"})
                continue
            r_e = st.median(e16) / st.median(e92)
            r_l = st.median(l16) / st.median(l92)
            se_e, _ = _ratio_se_bootstrap(e16, e92, seed=BOOT_SEED + blk)
            se_l, _ = _ratio_se_bootstrap(l16, l92, seed=BOOT_SEED + 100 + blk)
            # within-pair SD of r across the two halves (n=2 -> SD = |diff|/sqrt(2))
            sd_within = abs(r_e - r_l) / math.sqrt(2.0)
            sd_sampling = math.hypot(se_e, se_l) / math.sqrt(2.0)
            within_pairs.append((sd_within, sd_sampling, (r_e + r_l) / 2.0))
            rows.append({"blk": blk, "r_early": r_e, "r_late": r_l,
                         "n_early": [len(e16), len(e92)], "n_late": [len(l16), len(l92)],
                         "sd_within_pct": sd_within / ((r_e + r_l) / 2.0) * 100.0,
                         "sd_sampling_pct": sd_sampling / ((r_e + r_l) / 2.0) * 100.0,
                         "r_full_boot": st.median([x for _, x in d16]) / st.median([x for _, x in d92])})
        # pooled: RMS of the per-pair within SDs, in percent of r
        if within_pairs:
            rel_w = [s / mu * 100.0 for s, _, mu in within_pairs]
            rel_s = [s / mu * 100.0 for _, s, mu in within_pairs]
            sw = math.sqrt(st.fmean([x ** 2 for x in rel_w]))
            ss = math.sqrt(st.fmean([x ** 2 for x in rel_s]))
        else:
            sw = ss = float("nan")
        sd_between = st.stdev(r_full) / st.fmean(r_full) * 100.0
        resid = math.sqrt(max(0.0, sw ** 2 - ss ** 2))
        res["arms"][arm] = {
            "n_boot_pairs": len(blks),
            "sigma_boot_between_pct_recomputed": sd_between,
            "sigma_boot_between_pct_design_input": ps["arms"][arm]["sigma_boot_paired_pct"],
            "within_boot_halfsplit_sd_pct_raw": sw,
            "halfsplit_sampling_sd_pct": ss,
            "within_boot_temporal_residual_pct_noise_removed": resid,
            "ratio_raw_over_between": (sw / sd_between) if sd_between else None,
            "ratio_residual_over_between": (resid / sd_between) if sd_between else None,
            "boots_scaling_vs_between_contrast_raw": (sw / sd_between) ** 2 if sd_between else None,
            "per_block": rows,
            "caveats": [
                "half-split uses the LARGEST within-boot lag -> upper bound on the "
                "residual an alternating design carries -> lower bound on cancellation",
                "each within-pair SD is n=2 (|diff|/sqrt2); the pooled figure is an "
                "RMS over 6 such pairs, i.e. df=6, not a precise variance",
                "noise-removed column subtracts an interval-bootstrap sampling term; "
                "it is reported BESIDE the raw number, never instead of it",
            ],
        }

    res["positive_controls"].append({
        "id": "PC1", "what": "per-boot medians match the committed C2R JSON exactly",
        "n_mismatch": len(pc1_bad), "mismatches": pc1_bad[:5], "ok": not pc1_bad})
    pc2 = []
    for arm in ("M8", "Ha8"):
        a = res["arms"][arm]
        pc2.append(abs(a["sigma_boot_between_pct_recomputed"]
                       - a["sigma_boot_between_pct_design_input"]))
    res["positive_controls"].append({
        "id": "PC2", "what": "between-boot SD of r matches design_g13_stats.pairsd()",
        "abs_diff_pct_points": pc2, "tolerance": 1e-3, "ok": all(x <= 1e-3 for x in pc2)})
    res["CONTROLS"] = "PASS" if all(c["ok"] for c in res["positive_controls"]) else "FAIL"
    res["VERDICT_TYPE"] = ("measurement with uncertainty -- NOT a gate, NOT a "
                           "pass/fail, NOT pre-registered, NOT audited")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = run()
    txt = json.dumps(r, indent=2, default=str)
    if a.out:
        dest = os.path.join(HERE, a.out)
        if os.path.exists(dest) and not a.force:
            sys.stderr.write("REFUSING to overwrite: %s\n" % dest)
            raise SystemExit(2)
        open(dest, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
