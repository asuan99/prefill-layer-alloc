"""CPU tests for n2_probe_reduce.py (PREREG_N2_3A rev8, R7-1).
Registered run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_probe_reduce -v"""
import os
import statistics
import unittest
import n2_probe_reduce as r
import n2_feasibility as f

N3_DIR = os.environ.get("N2_N3_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "n3_capacity", "n3b_pdmux-n3-capacity-38ed3f4-20261006092813")


def _span(seq, t0, tok_s, p=64, bs=32, n_iter=5):
    t1 = t0 + 16384.0 / tok_s
    ev = [dict(event="prefill_span_start", span_seq=seq, timestamp_monotonic_s=t0, extend_tokens=16384),
          dict(event="prefill_span_end", span_seq=seq, timestamp_monotonic_s=t1)]
    for i in range(n_iter):
        t = t0 + (t1 - t0) * (i + 0.5) / n_iter
        ev.append(dict(event="decode_iteration", t_launch=t, bs=bs, prefill_sms=p, decode_sms=108 - p))
    return ev, t1


def one_span_run_boot():
    """25 load calls with ONE (slow, 4,748 tok/s) 16,384-token span each + 1 load call with 25 spans
    (first slow, rest 5,693). Registered composition keeps 24 x 5,693."""
    recs, calls, t, seq = [], [], 0.0, 0
    for _ in range(25):
        seq += 1; ev, t1 = _span(seq, t, 4748.0); recs += ev; calls.append((t, t1 + 0.1)); t = t1 + 1.0
    a = t
    for i in range(25):
        seq += 1; ev, t1 = _span(seq, t, 4748.0 if i == 0 else 5693.0); recs += ev; t = t1 + 0.01
    calls.append((a, t))
    return [("synthetic", recs, calls)]


class TestTheta(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(N3_DIR), "N3 run-3 raw telemetry not present locally")
    def test_positive_control_n3(self):
        th, ex, dr = r.n3_theta_positive_control(N3_DIR)
        self.assertEqual(set(th), {64})
        self.assertAlmostEqual(statistics.median(th[64]), 5693.0, delta=10.0)
        self.assertEqual((len(th[64]), len(ex), len(dr)), (288, 26, 0))

    def test_first_span_exclusion_one_span_runs(self):
        th, ex, dr = r.reduce_theta(one_span_run_boot())
        self.assertEqual(len(th[64]), 24)
        self.assertAlmostEqual(statistics.median(th[64]), 5693.0, places=3)
        self.assertEqual(len(ex), 26)
        self.assertTrue(all(abs(e["tok_s"] - 4748.0) < 1e-6 for e in ex))

    def test_decode_load_and_realized_split(self):
        ev1, t = _span(1, 0.0, 5693.0)                 # first -> excluded
        ev2, t = _span(2, t + 1, 5693.0, bs=40)        # above binding bs -> dropped
        ev3, t = _span(3, t + 1, 5693.0, p=0)          # no realized split (p=0 -> (0,108)) -> dropped
        ev4, t = _span(4, t + 1, 5693.0)               # kept
        th, ex, dr = r.reduce_theta([("b", ev1 + ev2 + ev3 + ev4, [])])
        self.assertEqual(len(th[64]), 1)
        self.assertEqual(sorted(d["reason"][:8] for d in dr), ["decode l", "no reali"])


class TestCellsAndJson(unittest.TestCase):
    def _window(self, n, dt_ms, t0=0.0):
        return [dict(t_launch=t0 + i * dt_ms / 1000.0) for i in range(n)]

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(r.ReduceRejected):
            r.loads_strict('{"a": 1, "a": 2}')

    def test_same_cell_multiple_measurements_pooled(self):
        # pooled median of ALL intervals (30) differs from the median of per-measurement medians (10)
        w1, w2, w3 = self._window(129, 10.0), self._window(129, 10.0), self._window(600, 30.0)
        self.assertAlmostEqual(r.pooled_median_ms([w1, w2, w3]), 30.0, places=6)

    def test_short_window_rejected(self):
        with self.assertRaises(r.ReduceRejected):
            r.s_per_intervals(self._window(100, 10.0))

    def test_end_to_end_probe_accepted(self):
        boots = one_span_run_boot()
        # add a p=92 run of 21 spans after exclusion (22 spans in one call)
        recs, calls, t = [], [], 10000.0
        a = t
        for i in range(22):
            ev, t = _span(1000 + i, t, 5693.0 * 92 / 64, p=92); recs += ev; t += 0.01
        boots.append(("b92", recs, [(a, t)]))
        def win(ms):
            return [self._window(129, ms)]
        tau = {(16, 32, 8192): win(f.tau_alone(16, 32, 8192, "DN") * 1000), (44, 32, 8192): win(f.tau_alone(44, 32, 8192, "DN") * 1000)}
        kap = {(16, 32, 8192): {"C": win(13.0), "A_on": win(10.0)}, (44, 32, 8192): {"C": win(13.0), "A_on": win(10.0)}}
        probe = r.build_probe(boots, tau, kap, [30], [0.0], [22.0])
        self.assertEqual(len(probe["theta_spans"][64]), 24)
        self.assertEqual(len(probe["theta_spans"][92]), 21)
        g = f.post_probe_gate({k: v for k, v in probe.items() if not k.startswith("theta_") or k == "theta_spans"})
        self.assertTrue(g["seal_allowed"])


if __name__ == "__main__":
    unittest.main()
