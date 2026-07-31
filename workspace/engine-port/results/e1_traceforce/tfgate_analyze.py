"""Paired OFF-vs-ON summary for the PDMUX_TRACE_FORCE_PREFILL observer gate.

Reads the probe JSONL written by traceforce_gate.sbatch.  Pairing is
(cell, seed, ABBA position): the ABBA boot order OFF ON ON OFF gives an
outer pair (boot1 OFF, boot4 OFF) and an inner pair (boot2 ON, boot3 ON);
outer-OFF/inner-ON are paired within seed so each pair is workload-matched
and roughly balanced for boot order and triton-cache warmth.

Reported per metric: OFF mean, ON mean, relative delta, and a paired
bootstrap 95% CI on the relative delta.  n=4 pairs per cell is small by
design (this is an overhead gate, not a policy comparison): the CI width
IS the answer -- read it as "the patch is not detectably worse than X",
never as evidence of an improvement.
"""
import json
import sys

import numpy as np

METRICS = [
    "ttft_p50", "ttft_p95", "ttft_p99", "ttft_mean",
    "itl_p50", "itl_p95", "itl_p99", "itl_mean",
    "output_throughput",
]


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def paired(rows, cell):
    off = sorted([r for r in rows if r["cell"] == cell and r["force"] == 0],
                 key=lambda r: (r["boot"], r["seed"]))
    on = sorted([r for r in rows if r["cell"] == cell and r["force"] == 1],
                key=lambda r: (r["boot"], r["seed"]))
    pairs = []
    for o in off:
        cands = [x for x in on if x["seed"] == o["seed"]
                 and x not in [p[1] for p in pairs]]
        if not cands:
            continue
        # inner ON boot pairs with the nearest OFF boot
        cands.sort(key=lambda x: abs(x["boot"] - o["boot"]))
        pairs.append((o, cands[0]))
    return pairs


def boot_ci(deltas, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows = load(sys.argv[1])
    for cell in sorted({r["cell"] for r in rows}):
        pairs = paired(rows, cell)
        print(f"\n--- cell={cell} n_pairs={len(pairs)} "
              f"(OFF boots {[o['boot'] for o, _ in pairs]} vs "
              f"ON boots {[n['boot'] for _, n in pairs]}) ---")
        vol_off = [o for o, _ in pairs]
        vol_on = [n for _, n in pairs]
        print("TELEMETRY_VOLUME snapshots/probe OFF={:.0f} ON={:.0f} (+{:.1f}%)"
              "  forced/probe ON={:.0f}  prefill_active/probe OFF={:.1f} ON={:.1f}"
              "  dropped_events max OFF={} ON={}".format(
                  np.mean([r["n_snapshots"] for r in vol_off]),
                  np.mean([r["n_snapshots"] for r in vol_on]),
                  100.0 * (np.mean([r["n_snapshots"] for r in vol_on])
                           / max(1e-9, np.mean([r["n_snapshots"] for r in vol_off])) - 1.0),
                  np.mean([r["n_forced"] for r in vol_on]),
                  np.mean([r["n_prefill_active"] for r in vol_off]),
                  np.mean([r["n_prefill_active"] for r in vol_on]),
                  max([r["dropped_events"] for r in vol_off], default=0),
                  max([r["dropped_events"] for r in vol_on], default=0)))
        for m in METRICS:
            a = np.array([o[m] for o, _ in pairs], dtype=float)
            b = np.array([n[m] for _, n in pairs], dtype=float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            rel = (b - a) / np.where(a == 0, np.nan, a)
            lo, hi = boot_ci(rel)
            # Student-t CI as a cross-check: the percentile bootstrap is
            # anti-conservative at n=4 (it cannot resample outside the
            # observed 4 values), so the t interval is the one to quote when
            # the two disagree.
            n = len(rel)
            tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
            half = tcrit * rel.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            tlo, thi = rel.mean() - half, rel.mean() + half
            flag = "" if (tlo <= 0.0 <= thi) else "  <-- t-CI EXCLUDES 0"
            print(f"{m:>18s}: OFF {a.mean():9.2f} (sd {a.std(ddof=1):7.2f})  "
                  f"ON {b.mean():9.2f} (sd {b.std(ddof=1):7.2f})  "
                  f"rel {100*rel.mean():+6.2f}%  boot95 [{100*lo:+6.2f}%, {100*hi:+6.2f}%]"
                  f"  t95 [{100*tlo:+6.2f}%, {100*thi:+6.2f}%]{flag}")


if __name__ == "__main__":
    main()
