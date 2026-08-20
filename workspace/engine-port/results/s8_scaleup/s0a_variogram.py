#!/usr/bin/env python3
"""S0(a) extension -- lag-resolved within-boot structure of r (variogram).

GPU 0.  Reads the same cached extraction as `s0a_within_boot.py`.

WHY: the half-split answers the WRONG question for C-g.  A half-split separates
the two halves by the LARGEST available within-boot lag, and it came out 1.6-1.7x
the BETWEEN-boot SD -- i.e. as a ceiling on cancellation it is uninformative
(>1).  What C-g actually needs is whether pairing at a SHORT lag (adjacent
alternating segments) sits BELOW sigma_boot.  That is a lag-dependence question,
so measure the lag dependence.

Estimator: bin each boot's accepted intervals into `NBIN` equal-count bins by
snapshot index (the boot's own time axis), form r per bin, and report the
semivariogram
        gamma(h) = mean over bin pairs (i, j) with |i-j| = h  of  (r_i - r_j)^2 / 2
in percent-of-r units.  sqrt(gamma(h)) is the SD a paired contrast at lag h
would carry.  A bin-level sampling term is estimated by bootstrap and reported
alongside -- never subtracted from the headline.

NOT a gate, NOT pre-registered, NOT audited.  No new estimand: r is still the
ratio of `s8_c2r_score` bracket-matched medians.
"""
import json, math, os, pickle, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".s0a_cells_cache.pkl")
NBIN = 8
DRAWS = 400
SEED = 20260820


def _bins(v, nbin):
    """Equal-COUNT bins along the boot's own time axis (snapshot index)."""
    v = sorted(v)
    n = len(v)
    return [[x for _, x in v[i * n // nbin:(i + 1) * n // nbin]] for i in range(nbin)]


def _median_se(a, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(a, dtype=float)
    return float(np.median(rng.choice(arr, size=(draws, arr.size), replace=True),
                           axis=1).std(ddof=1))


def run():
    cells = pickle.load(open(CACHE, "rb"))
    out = {"what": "lag-resolved within-boot variogram of r", "gpu_spend": 0,
           "n_bins_per_boot": NBIN, "is_a_gate": False, "arms": {}}
    for arm in ("M8", "Ha8"):
        blks = sorted(b for b, d in cells[arm].items() if len(d) == 2)
        gam = {h: [] for h in range(1, NBIN)}
        samp = {h: [] for h in range(1, NBIN)}
        r_full = []
        for blk in blks:
            b16 = _bins(cells[arm][blk]["d16"], NBIN)
            b92 = _bins(cells[arm][blk]["d92"], NBIN)
            if min(len(x) for x in b16 + b92) < 40:
                continue
            r = [st.median(a) / st.median(b) for a, b in zip(b16, b92)]
            se = [math.hypot(_median_se(a, seed=SEED + i) / st.median(a),
                             _median_se(b, seed=SEED + 50 + i) / st.median(b)) * 100.0
                  for i, (a, b) in enumerate(zip(b16, b92))]
            mu = st.fmean(r)
            r_full.append(st.median([x for _, x in cells[arm][blk]["d16"]])
                          / st.median([x for _, x in cells[arm][blk]["d92"]]))
            for h in range(1, NBIN):
                for i in range(NBIN - h):
                    gam[h].append(((r[i] - r[i + h]) / mu * 100.0) ** 2 / 2.0)
                    samp[h].append((se[i] ** 2 + se[i + h] ** 2) / 2.0)
        sd_between = st.stdev(r_full) / st.fmean(r_full) * 100.0
        rows = []
        for h in range(1, NBIN):
            g = st.fmean(gam[h]) if gam[h] else float("nan")
            s = st.fmean(samp[h]) if samp[h] else float("nan")
            rows.append({"lag_bins": h, "n_pairs": len(gam[h]),
                         "sd_paired_at_lag_pct": math.sqrt(g),
                         "sampling_term_pct": math.sqrt(s),
                         "sd_noise_removed_pct": math.sqrt(max(0.0, g - s)),
                         "vs_sigma_boot": math.sqrt(g) / sd_between})
        out["arms"][arm] = {
            "sigma_boot_between_pct": sd_between,
            "n_boot_pairs_used": len(r_full),
            "variogram": rows,
            "reading": ("sd_paired_at_lag < sigma_boot means a within-boot paired "
                        "contrast at that lag is quieter than a between-boot one; "
                        ">= means C-g buys nothing at that lag"),
        }
    return out


if __name__ == "__main__":
    r = run()
    txt = json.dumps(r, indent=2)
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        dest = os.path.join(HERE, sys.argv[2])
        if os.path.exists(dest):
            sys.stderr.write("REFUSING to overwrite: %s\n" % dest); raise SystemExit(2)
        open(dest, "w").write(txt)
    print(txt)
