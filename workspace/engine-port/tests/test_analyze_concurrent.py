"""Concurrent-replica duration rule of benchmarks/pdmux_eval/analyze.py (multi-GPU track, M2).

Pins (design reports/multigpu_scaling_design_2026-10-12.md section 4-2, audit section 5):
  * 1-GPU / sequential-round results are numerically unchanged (literal values computed with
    the pre-change module);
  * concurrent replicas: round denominator = shared client wall clock (first send to last
    completion over all replicas), summed across rounds; aggregate AND per-replica goodput use it;
  * three mutants must be KILLED by the same scenario check:
      (i)   concurrent replica durations summed instead of shared,
      (ii)  a replica divided by its own first-send/last-completion span,
      (iii) sequential rounds not summed.
    An unmutated control must pass (a check that never fails kills nothing).
"""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmarks"))
from pdmux_eval import analyze as B  # noqa: E402  (1-GPU module, must stay unchanged)
from pdmux_eval import analyze_concurrent as A  # noqa: E402

TTFT, ITL = 1000.0, 50.0


def T(replica, sent, comp, ttft=None, itls=()):
    req = None if ttft is None else B.RequestResult(
        f"{replica}{sent}", ttft, tuple(itls), comp)
    return A.TimedRequest(replica, float(sent), float(comp), req)


ROUND0 = [
    T("A", 0, 5, 100, [10, 20]),            # pass
    T("A", 2, 8, 1500, [10]),               # ttft_fail
    T("B", 1, 6, 200, [100, 100]),          # itl_fail
    T("B", 3, 10, 2000, [100]),             # both_fail
]
ROUND1 = [
    T("A", 0, 6, 100, [10]),                # pass
    T("A", 1, 4),                           # failed request (occupies the clock)
    T("B", 2, 12, 300, [20]),               # pass
    T("B", 3, 9, 100, []),                  # no ITL -> undefined p95 -> itl_fail (as RequestResult.passes)
]
ROUNDS = [ROUND0, ROUND1]
# shared wall: round0 0->10 = 10, round1 0->12 = 12  => 22
# own spans:   A 8 / 6, B 9 / 10 ;  sum of own spans per round: 17 / 16 => 33


def scenario_failures():
    """Return the list of violated expectations for the CURRENT module state."""
    bad = []

    def expect(name, got, want):
        ok = (math.isclose(got, want, rel_tol=1e-12) if isinstance(want, float)
              else got == want)
        if not ok:
            bad.append(f"{name}: got {got!r}, want {want!r}")

    out = A.summarize_concurrent_rounds(ROUNDS, TTFT, ITL)
    expect("shared wall", out["shared_wall_clock_s"], 22.0)
    expect("round walls", out["round_wall_clock_s"], [10.0, 12.0])
    agg = out["aggregate"]
    expect("agg duration", agg["duration_s"], 22.0)
    expect("agg goodput", agg["slo_goodput_req_s"], 3 / 22)
    expect("agg throughput", agg["throughput_req_s"], 7 / 22)
    for rep, good in (("A", 2), ("B", 1)):
        cell = out["per_replica"][rep]
        expect(f"{rep} duration", cell["duration_s"], 22.0)
        expect(f"{rep} goodput", cell["slo_goodput_req_s"], good / 22)
    expect("replica goodputs add up to aggregate",
           sum(c["slo_goodput_req_s"] for c in out["per_replica"].values()),
           agg["slo_goodput_req_s"])
    # sequential rule: sum
    expect("sequential sum", A.sequential_duration_s([10.0, 20.0]), 30.0)
    return bad


class ConcurrentDuration(unittest.TestCase):
    def test_control_passes(self):
        self.assertEqual(scenario_failures(), [])

    def test_mutant_i_replica_durations_summed_is_killed(self):
        def summed(round_requests):
            spans = {}
            for it in round_requests:
                lo, hi = spans.get(it.replica, (it.sent_s, it.completion_s))
                spans[it.replica] = (min(lo, it.sent_s), max(hi, it.completion_s))
            return sum(hi - lo for lo, hi in spans.values())
        with mock.patch.object(A, "shared_wall_clock_s", summed):
            self.assertTrue(scenario_failures())

    def test_mutant_ii_replica_own_span_is_killed(self):
        def own(shared_wall_s, replica_requests):
            return (max(i.completion_s for i in replica_requests)
                    - min(i.sent_s for i in replica_requests))
        with mock.patch.object(A, "replica_denominator_s", own):
            bad = scenario_failures()
        self.assertTrue(bad)
        # the sum-of-replicas == aggregate identity is among what breaks
        self.assertTrue(any("add up" in b or "duration" in b for b in bad))

    def test_mutant_iii_sequential_not_summed_is_killed(self):
        with mock.patch.object(A, "sequential_duration_s",
                               lambda xs: float(max(xs))):
            self.assertTrue(scenario_failures())

    def test_failed_request_occupies_wall_clock(self):
        late_fail = ROUND0 + [T("A", 0, 30)]   # a failure completing last stretches the round
        out = A.summarize_concurrent_rounds([late_fail], TTFT, ITL)
        self.assertEqual(out["round_wall_clock_s"], [30.0])
        self.assertEqual(out["aggregate"]["failed_requests"], 1.0)

    def test_empty_and_degenerate_rounds_rejected(self):
        with self.assertRaises(ValueError):
            A.summarize_concurrent_rounds([], TTFT, ITL)
        with self.assertRaises(ValueError):
            A.shared_wall_clock_s([T("A", 5, 5, 1, [1])])


class SloFailureBreakdown(unittest.TestCase):
    def test_partition_matches_goodput_numerator(self):
        out = A.summarize_concurrent_rounds(ROUNDS, TTFT, ITL)
        b = out["aggregate"]["slo_failure_breakdown"]
        self.assertEqual((b["pass"], b["ttft_fail"], b["itl_fail"], b["both_fail"]),
                         (3.0, 1.0, 2.0, 1.0))
        self.assertEqual(b["pass"] + b["ttft_fail"] + b["itl_fail"] + b["both_fail"],
                         b["requests"])
        reqs = [i.request for r in ROUNDS for i in r if i.request]
        self.assertEqual(b["pass"], sum(q.passes(TTFT, ITL) for q in reqs))
        self.assertEqual(out["per_replica"]["B"]["slo_failure_breakdown"]["both_fail"], 1.0)

    def test_boundary_values_pass(self):
        q = B.RequestResult("x", TTFT, (ITL,), 1.0)
        self.assertEqual(A.slo_failure_breakdown([q], TTFT, ITL)["pass"], 1.0)


class OneGpuUnchanged(unittest.TestCase):
    """Literal values produced by the module BEFORE this change (git HEAD 20e184d)."""

    def test_sequential_loader_and_summary_pinned(self):
        rounds = [
            {"duration": 10.0, "ttfts": [0.1, 0.5, 2.0], "itls": [[0.01, 0.02], [0.03, 0.2], [0.01]]},
            {"duration": 20.0, "ttfts": [0.2, 0.3], "itls": [[0.04, 0.05, 0.06], []]},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "fx.jsonl")
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rounds) + "\n")
            from pathlib import Path
            reqs, dur = B.load_bench_serving_rounds(Path(path))
        self.assertEqual(dur, 30.0)
        s = B.summarize_requests(reqs, 0.0, dur, 1000.0, 50.0)
        pinned = {"duration_s": 30.0, "legacy_mean_itl_goodput_req_s": 0.1, "requests": 5.0,
                  "slo_goodput_req_s": 0.03333333333333333, "slo_violation_ratio": 0.8,
                  "throughput_req_s": 0.16666666666666666, "token_itl_p50_ms": 35.0,
                  "token_itl_p90_ms": 101.99999999999997, "token_itl_p95_ms": 150.99999999999994,
                  "token_itl_p99_ms": 190.19999999999996, "ttft_p50_ms": 300.0,
                  "ttft_p90_ms": 1400.0, "ttft_p95_ms": 1699.9999999999998,
                  "ttft_p99_ms": 1940.0}
        self.assertEqual(s, pinned)           # exact: same key set, bit-identical floats

    def test_analyze_module_has_no_concurrent_api(self):
        # the sealed-digest module must not grow the new API (it lives in analyze_concurrent)
        self.assertFalse(hasattr(B, "summarize_concurrent_rounds"))

    def test_summarize_requests_has_no_new_keys(self):
        q = B.RequestResult("x", 1.0, (1.0,), 1.0)
        self.assertNotIn("slo_failure_breakdown", B.summarize_requests([q], 0.0, 1.0, 10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
