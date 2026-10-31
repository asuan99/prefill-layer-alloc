"""CPU tests for results/stock_baseline (PLAN_stock_baseline_metrics_2026-11-01, Phase A)."""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "results", "stock_baseline"))

import stock_metrics as sm  # noqa: E402
import r_ser_holb as rh  # noqa: E402


def req(ttft, itls, i=100, o=None, e2e=None, ok=True):
    return sm.request(ttft, itls, i, o if o is not None else len(itls) + 1, e2e, ok)


class TestMetrics(unittest.TestCase):
    def test_duration_is_summed_not_maxed(self):
        r1 = (10.0, [req(0.1, [0.01] * 4)] * 10)
        r2 = (10.0, [req(0.1, [0.01] * 4)] * 10)
        m = sm.metrics([r1, r2])
        self.assertEqual(m["duration_s"], 20.0)
        self.assertAlmostEqual(m["req_per_s"], 1.0)  # max() would give 2.0 (gate 7 bug)

    def test_throughputs(self):
        m = sm.metrics([(2.0, [req(0.1, [0.01] * 9, i=100, o=10)] * 4)])
        self.assertAlmostEqual(m["out_tok_per_s"], 20.0)
        self.assertAlmostEqual(m["total_tok_per_s"], 220.0)

    def test_missing_itl_is_inf_and_fails_goodput(self):
        m = sm.metrics([(1.0, [req(0.1, [], o=1), req(0.1, [0.01, 0.01])])], 3.0, 0.06)
        self.assertEqual(m["goodput_req_per_s"], 1.0)
        self.assertTrue(math.isinf(m["tpot_p99_s"]))

    def test_goodput_uses_request_internal_p95_not_mean(self):
        # mean ITL 0.0275 s < 60 ms, but p95 of [0.01]*19 + [0.36] is > 60 ms
        itls = [0.01] * 18 + [0.36, 0.36]
        m = sm.metrics([(1.0, [req(0.1, itls)])], 3.0, 0.06)
        self.assertEqual(m["goodput_req_per_s"], 0.0)

    def test_errors_excluded(self):
        m = sm.metrics([(1.0, [req(0.1, [0.01]), req(None, [], ok=False), req(0.2, [0.01], ok=False)])])
        self.assertEqual((m["n_sent"], m["n_ok"]), (3, 1))

    def test_e2e_source_flag(self):
        self.assertEqual(sm.metrics([(1.0, [req(0.1, [0.01] * 3)])])["e2e_source"], "ttft+itl")
        m = sm.metrics([(1.0, [req(0.1, [0.01] * 3, e2e=0.5)])])
        self.assertEqual(m["e2e_source"], "measured")
        self.assertAlmostEqual(m["e2e_p50_s"], 0.5)

    def test_pctl_matches_linear_and_inf(self):
        self.assertAlmostEqual(sm.pctl([1, 2, 3, 4], 0.5), 2.5)
        self.assertTrue(math.isinf(sm.pctl([1, math.inf, math.inf], 0.9)))

    def test_bench_row_loader(self):
        row = {"duration": 4.0, "ttfts": [0.1, 0.2], "itls": [[0.01], []], "errors": ["", "boom"],
               "input_lens": [10, 10], "output_lens": [2, 2]}
        dur, reqs = sm.from_bench_serving_row(row)
        self.assertEqual(dur, 4.0)
        self.assertEqual([r["ok"] for r in reqs], [True, False])


class TestHolb(unittest.TestCase):
    @staticmethod
    def gap(ts, dec=7.0, same=0.0, n_other=0, mode="DECODE", bs=1, cls="strict"):
        return {"host_ts": ts, "decode_dur_ms": dec, "same_stream_fw_ms": same, "n_other_fw": n_other,
                "decode_mode": mode, "decode_bs": bs, "gap_class": cls, "inconsistent": False}

    def test_segments_drop_warmup_and_cut_at_largest_pause(self):
        g = [self.gap(t) for t in (0.0, 0.1)]                        # warm-up
        g += [self.gap(10 + 0.01 * i) for i in range(200)]           # rate 1, with an in-run 4 s silence
        g += [self.gap(16 + 0.01 * i) for i in range(200)]
        g += [self.gap(40 + 0.01 * i) for i in range(200)]           # rate 2 after a 22 s pause
        segs = rh.segments(g, 2)
        self.assertEqual([len(s) for s in segs], [400, 200])

    def test_r_ser_and_r_mix(self):
        seg = [self.gap(i, dec=10.0) for i in range(10)]
        seg.append(self.gap(11, dec=10.0, same=150.0, n_other=1))
        seg.append(self.gap(12, dec=40.0, mode="MIXED"))
        st = rh.seg_stats(seg)
        self.assertAlmostEqual(st["r_ser_p50"], 15.0)
        self.assertAlmostEqual(st["r_mix_p50"], 3.0)
        self.assertEqual(st["frac_r_ser_gt1"], 1.0)

    def test_first_gap_excluded(self):
        seg = [self.gap(i) for i in range(10)] + [self.gap(11, same=500.0, n_other=1, cls="first")]
        self.assertEqual(rh.seg_stats(seg)["n_r_ser"], 0)


if __name__ == "__main__":
    unittest.main()
