"""Pure-function checks for results/multigpu_probe/m15_client_capacity_probe.py (no network)."""

import importlib.util
import math
import unittest
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "results" / "multigpu_probe" / "m15_client_capacity_probe.py"
spec = importlib.util.spec_from_file_location("m15", P)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def point(rate, sched99=0.5, wire99=0.5, ttft99=11.0, itl95=26.0, errors=0):
    return {"rate_req_s": rate, "errors": errors, "sched_lag_ms": {"p99": sched99},
            "wire_raw_ms": {"p50": 0.2, "p99": 0.2 + wire99}, "ttft_ms": {"p50": 10.0, "p99": ttft99},
            "itl_ms": {"p95": itl95}}


class Probe(unittest.TestCase):
    def test_trace_is_deterministic_and_poisson_rate(self):
        a = M.make_trace(50.0, 100.0, 10, 5, seed=3, tag="t")
        self.assertEqual(a, M.make_trace(50.0, 100.0, 10, 5, seed=3, tag="t"))
        self.assertAlmostEqual(len(a) / 100.0, 50.0, delta=5.0)
        self.assertTrue(all(x["record"] == "request" for x in a))

    def test_judge_discriminates_each_leg(self):
        ok = M.judge([point(2), point(64)])
        self.assertTrue(ok[1]["PASS_candidate"])
        for bad in (point(64, sched99=9.0), point(64, wire99=9.0), point(64, ttft99=20.0),
                    point(64, itl95=30.0), point(64, errors=1)):
            self.assertFalse(M.judge([point(2), bad])[1]["PASS_candidate"], bad)

    def test_percentile(self):
        self.assertEqual(M.pct([1, 2, 3, 4], 0.5), 2.5)
        self.assertTrue(math.isnan(M.pct([], 0.5)))


class TraceLoadgenDefault(unittest.TestCase):
    def test_connector_limit_defaults_to_stock(self):
        import inspect
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
        from pdmux_eval import trace_loadgen as TL
        self.assertIsNone(inspect.signature(TL.replay).parameters["connector_limit"].default)
        src = inspect.getsource(TL.replay)
        self.assertIn("if connector_limit is None else", src)   # None -> no connector kwarg at all


if __name__ == "__main__":
    unittest.main()
