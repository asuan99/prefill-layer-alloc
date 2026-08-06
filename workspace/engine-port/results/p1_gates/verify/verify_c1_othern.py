"""C-1 blast-radius addendum: coverage of the canonical percentile bootstrap at
other repetition counts used elsewhere in this repository (n=3,4,5,6,8,10).

Corrects an off-by-one in the t critical table of the first pass of
``verify_c1_coverage.py`` (that pass indexed the table by n-1 while the table was
keyed by n, so the reported t coverage/width ratio in its 'other n' block used
df = n-2).  The BOOTSTRAP numbers in that block were unaffected (they never touch
the t table) and are reproduced here for cross-check.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import verify_c1_coverage as V

# t_{0.975, df}
TCRIT_BY_DF = {2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836,
               7: 2.364624252, 9: 2.262157163}


def main():
    rng = np.random.default_rng(20260806)
    out = {}
    print(f"{'n':>3s} {'boot cov':>9s} {'t cov':>7s} {'boot width':>11s} {'t width':>9s} "
          f"{'width ratio':>11s}")
    for n in (3, 4, 5, 6, 8, 10):
        _, counts = V.recover_design(n=n)
        V.N = n
        V.TCRIT4 = TCRIT_BY_DF[n - 1]
        r = V.coverage(lambda m, k=n: rng.normal(size=(m, k)), 100000, counts, label=f"n={n}")
        out[n] = r
        print(f"{n:3d} {r['coverage_bootstrap']:9.4f} {r['coverage_t']:7.4f} "
              f"{r['mean_width_bootstrap']:11.4f} {r['mean_width_t']:9.4f} "
              f"{r['width_ratio_of_means']:11.4f}")
    # theoretical width ratio: 2*1.96*sqrt((n-1)/n)/ (2*t_{.975,n-1})
    print("\n(reference) normal-approx ratio 1.96*sqrt((n-1)/n)/t_{.975,n-1}:")
    for n in (3, 4, 5, 6, 8, 10):
        print(f"   n={n:2d}: {1.959964 * math.sqrt((n - 1) / n) / TCRIT_BY_DF[n - 1]:.4f}")
    Path(Path(__file__).parent / "c1_other_n.json").write_text(json.dumps(out, indent=1))
    print("\nwrote c1_other_n.json")


if __name__ == "__main__":
    main()
