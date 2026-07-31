"""Diagnostic: does forced prefill sampling put snapshots INSIDE the episode?

`e1_pin_check.compute_episode_gate` reads the realized partition and
prefill activity at the bracket ENDPOINTS only -- `ii` = last snapshot at or
before admission `a`, `jj` = first snapshot at or after first-token `b`.
Both lie OUTSIDE the request's own prefill window by construction, so at a
cell whose prefill is fast the endpoints see the runtime's idle/auto
partition (866066/d16: realized hist over accepted brackets was
P0 69 / P108 28, never the P92 target) and every bracket is "vacuous".

This script asks the complementary question the endpoint rule cannot:
within (a, b), how many snapshots are there, how many show prefill active,
and what partition do THOSE show?  It changes nothing in the harness -- it
is read-only evidence about what the forced samples make available.

Usage:
  tfgate_inside_bracket.py <telemetry.jsonl> <expected_D> <raw_bench.jsonl> \
      <seed> <rate> <t0> <t1>
"""
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "s8_frontier"),
)
import e1_pin_check as pin  # noqa: E402


def main():
    telem, expect_d, raw, seed, rate, t0, t1 = sys.argv[1:8]
    expect_d = int(expect_d)
    expect_p = pin.D_TO_PREFILL[expect_d]
    seed, rate = int(seed), float(rate)
    t1 = float(t1)

    d = json.load(open(raw))
    ttfts = [t * 1000.0 for t in d["ttfts"]]
    ttfts = [t / 1000.0 for t in ttfts]  # seconds, as compute_episode_gate uses
    itls, errors = d["itls"], d["errors"]

    ts, psm, pab, forced = [], [], [], []
    for line in open(telem):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
            continue
        ts.append(e["timestamp_monotonic_s"])
        psm.append(e.get("prefill_sms"))
        pab.append(e.get("prefill_active_batch_size", 0))
        forced.append(bool(e.get("trace_forced")))

    arrival_rel, completion_rel = pin.reconstruct_arrival_and_completion(
        seed, rate, ttfts, itls
    )
    t0_abs = t1 - max(completion_rel)

    import bisect

    n_ep = n_with_inside = n_with_active_inside = 0
    hist = {}
    inside_counts = []
    for i in range(len(ttfts)):
        if errors and i < len(errors) and errors[i]:
            continue
        a = t0_abs + arrival_rel[i]
        b = a + ttfts[i]
        lo = bisect.bisect_right(ts, a)
        hi = bisect.bisect_left(ts, b)
        n_ep += 1
        inside = list(range(lo, hi))
        inside_counts.append(len(inside))
        if inside:
            n_with_inside += 1
        act = [k for k in inside if pab[k] > 0]
        if act:
            n_with_active_inside += 1
            for k in act:
                hist[psm[k]] = hist.get(psm[k], 0) + 1

    tot = sum(hist.values())
    on_target = hist.get(expect_p, 0)
    print(
        "TFGATE_INSIDE D={} (expect P{}): episodes={} with>=1 snapshot inside={} "
        "with>=1 PREFILL-ACTIVE snapshot inside={} ({:.3f})  "
        "mean snapshots inside={:.1f}  forced_records_in_file={}".format(
            expect_d, expect_p, n_ep, n_with_inside, n_with_active_inside,
            (n_with_active_inside / n_ep) if n_ep else float("nan"),
            (sum(inside_counts) / max(1, len(inside_counts))),
            sum(forced),
        )
    )
    print(
        "TFGATE_INSIDE_realized(prefill-active samples inside brackets, n={}): {}"
        "  on_target_frac={}".format(
            tot,
            ", ".join(f"P{k}:{v}" for k, v in sorted(hist.items(), key=lambda x: -x[1])),
            f"{on_target / tot:.3f}" if tot else "nan",
        )
    )


if __name__ == "__main__":
    main()
