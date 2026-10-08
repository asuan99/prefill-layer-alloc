"""CPU tests for n2_probe_reduce.py (PREREG_N2_3A rev9: R7-1, R8-1/2/3).
Registered run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_probe_reduce -v"""
import json
import os
import statistics
import tempfile
import unittest
import n2_probe_reduce as r
import n2_feasibility as f

N3_DIR = os.environ.get("N2_N3_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "n3_capacity", "n3b_pdmux-n3-capacity-38ed3f4-20261006092813")


def _span(seq, t0, tok_s, p=64, bs=32, n_iter=5, extra=None):
    t1 = t0 + 16384.0 / tok_s
    ev = [dict(event="prefill_span_start", span_seq=seq, timestamp_monotonic_s=t0, extend_tokens=16384, n_new=2),
          dict(event="prefill_span_end", span_seq=seq, timestamp_monotonic_s=t1)]
    for i in range(n_iter):
        t = t0 + (t1 - t0) * (i + 0.5) / n_iter
        ev.append(dict(event="decode_iteration", t_launch=t, bs=bs, prefill_sms=p, decode_sms=108 - p))
    for (pp, dd, bb) in (extra or []):
        ev.append(dict(event="decode_iteration", t_launch=(t0 + t1) / 2, bs=bb, prefill_sms=pp, decode_sms=dd))
    return ev, t1


def one_span_run_boot():
    recs, calls, t, seq = [], [], 0.0, 0
    for _ in range(25):
        seq += 1; ev, t1 = _span(seq, t, 4748.0); recs += ev; calls.append((t, t1 + 0.1)); t = t1 + 1.0
    a = t
    for i in range(25):
        seq += 1; ev, t1 = _span(seq, t, 4748.0 if i == 0 else 5693.0); recs += ev; t = t1 + 0.01
    calls.append((a, t))
    return [("synthetic", recs, calls)]


def raw_cell(d=44, b=32, c=8192, kind="KAPPA", alone_ms=20.0, co_ms=26.0, a2_ms=22.0, n_c=200,
             leaks=0, contaminated_snapshot=True):
    """Synthetic RAW cell: burst span, alone segment through the ctx band, co-load spans with C iterations
    (optionally `leaks` (0,108) not-in-flight iterations inside W_co), then A2."""
    pair, pair108 = r.PAIR[d], r.PAIR[108]
    recs, t, q = [], 0.0, 0
    recs += [dict(event="prefill_span_start", span_seq=1, timestamp_monotonic_s=0.0, extend_tokens=b * 100, n_new=b),
             dict(event="prefill_span_end", span_seq=1, timestamp_monotonic_s=1.0)]
    t = 1.0
    recs.append(dict(event="runtime_snapshot", decode_iterations=1, context_p50=c - 144, active_decode_sequences=b))

    def it(step_ms, in_flight, pr, bs=b):
        nonlocal t, q
        q += 1; t += step_ms / 1000.0
        recs.append(dict(event="decode_iteration", iter_seq=q, t_launch=t, bs=bs, prefill_in_flight=in_flight,
                         prefill_sms=pr[0], decode_sms=pr[1]))
    for _ in range(230):
        it(alone_ms, False, pair)
    marks = {}
    if kind == "KAPPA":
        marks["coload_sent"] = t + 0.001
        t0 = t + 0.002
        recs.append(dict(event="prefill_span_start", span_seq=2, timestamp_monotonic_s=t0, extend_tokens=16384, n_new=2))
        if contaminated_snapshot:
            recs.append(dict(event="runtime_snapshot", decode_iterations=q + 5, context_p50=8193,
                             active_decode_sequences=b + 2))
        for i in range(n_c):
            it(co_ms, True, pair)
        for i in range(leaks):
            it(10.0, False, pair108)
        t1 = t + 0.001
        recs.append(dict(event="prefill_span_end", span_seq=2, timestamp_monotonic_s=t1))
        t = t1
        for _ in range(200):
            it(a2_ms, False, pair)
    return dict(cell=(d, b, c), kind=kind, records=recs, marks=marks)


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

    def test_decode_load_and_realized_split(self):
        ev1, t = _span(1, 0.0, 5693.0)
        ev2, t = _span(2, t + 1, 5693.0, bs=40)
        ev3, t = _span(3, t + 1, 5693.0, p=0)
        ev4, t = _span(4, t + 1, 5693.0)
        th, ex, dr = r.reduce_theta([("b", ev1 + ev2 + ev3 + ev4, [])])
        self.assertEqual(len(th[64]), 1)
        self.assertEqual(sorted(d["reason"][:8] for d in dr), ["decode l", "no reali"])

    def test_split_rule_at_least_one_and_unique_p(self):          # R8-7, kills B01 / B02
        ev1, t = _span(1, 0.0, 5693.0)
        ev2, t = _span(2, t + 1, 5693.0, extra=[(0, 108, 32)])     # one unsplit iteration -> still kept (>=1 rule)
        ev3, t = _span(3, t + 1, 5693.0, extra=[(92, 16, 32)])     # second split p -> dropped "mixed"
        th, ex, dr = r.reduce_theta([("b", ev1 + ev2 + ev3, [])])
        self.assertEqual(len(th[64]), 1)
        self.assertEqual([d["reason"][:5] for d in dr], ["mixed"])

    def test_decode_load_counts_all_iterations(self):             # kills B03
        ev1, t = _span(1, 0.0, 5693.0)
        ev2, t = _span(2, t + 1, 5693.0, extra=[(0, 108, 40)])     # bs 40 on an unsplit iteration -> dropped
        th, ex, dr = r.reduce_theta([("b", ev1 + ev2, [])])
        self.assertNotIn(64, th)

    def test_load_call_first_span_by_start_time(self):            # R8-7, kills B04
        ev0, t = _span(1, 0.0, 5693.0)                              # boot first -> excluded
        evA, tA = _span(2, 9.0, 8192.0)                             # starts before the call, ends inside
        evB, tB = _span(3, 12.0, 5693.0)                            # first span STARTING inside the call
        evC, tC = _span(4, 16.0, 5693.0)
        th, ex, dr = r.reduce_theta([("b", ev0 + evA + evB + evC, [(10.0, 30.0)])])
        self.assertEqual(sorted(e["span_seq"] for e in ex), [1, 3])

    def test_load_calls_from_marks(self):
        calls = r.load_calls_from_marks(["load_call_start 1 5.0", "load_call_end 1 9.0",
                                         "load_call_start 2 10.0", "load_call_end 2 12.0"])
        self.assertEqual(calls, [(5.0, 9.0), (10.0, 12.0)])
        with self.assertRaises(r.ReduceRejected):
            r.load_calls_from_marks(["load_call_start 1 5.0"])


class TestWindows(unittest.TestCase):
    def test_s_per_consecutive_pairs_only(self):                 # R8-2, kills B06-type
        w = [dict(iter_seq=q, t_launch=q * 0.010) for q in range(1, 161) if q % 10 != 0]   # gaps every 10th
        seqs = {e["iter_seq"] for e in w}
        iv = r.s_per_intervals(w)
        self.assertAlmostEqual(max(iv), 10.0, places=6)              # only 10 ms pairs; no 20 ms across gaps
        self.assertEqual(len(iv), sum(1 for q in seqs if q + 1 in seqs))

    def test_window_boundary_128(self):                            # kills B16
        mk = lambda n: [dict(iter_seq=q, t_launch=q * 0.01) for q in range(n)]
        with self.assertRaises(r.ReduceRejected):
            r.s_per_intervals(mk(127))
        self.assertEqual(len(r.s_per_intervals(mk(128))), 127)

    def test_tau_window_from_raw(self):
        m = raw_cell(kind="TAU")
        win = r.extract_cell_windows(m["records"], m["marks"], m["cell"], "TAU", 44)
        self.assertEqual(len(win["W"]), 128)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["W"])), 20.0, places=6)

    def test_kappa_parts_pinned(self):                             # kills B18 / B19
        m = raw_cell()
        win = r.extract_cell_windows(m["records"], m["marks"], m["cell"], "KAPPA", 44)
        c_iv, a_on = r.kappa_parts(win)
        self.assertAlmostEqual(statistics.median(c_iv), 26.0, places=6)
        mm = win["ctx_mean"]
        self.assertAlmostEqual(a_on, 20.0 + 2.0 * (mm["C"] - mm["A1"]) / (mm["A2"] - mm["A1"]), places=6)
        self.assertTrue(20.0 < a_on < 22.0)

    def test_c_window_excludes_leaks(self):                        # C contamination mutant must fail
        m = raw_cell(leaks=300)
        win = r.extract_cell_windows(m["records"], m["marks"], m["cell"], "KAPPA", 44)
        self.assertEqual(len(win["C"]), 200)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["C"])), 26.0, places=6)

    def test_contaminated_snapshot_excluded(self):
        m = raw_cell()
        win = r.extract_cell_windows(m["records"], m["marks"], m["cell"], "KAPPA", 44)
        self.assertAlmostEqual(win["ctx_mean"]["A1"], 8192 - 0.5, places=6)
        # C iterations iter_seq 231..430 -> ctx = (c - 144) + (seq - 1); a used contaminated snapshot would give ~8290
        self.assertAlmostEqual(win["ctx_mean"]["C"], 8377.5, places=6)

    def test_missing_coload_mark_rejected(self):
        m = raw_cell(); m["marks"] = {}
        with self.assertRaises(r.ReduceRejected):
            r.extract_cell_windows(m["records"], m["marks"], m["cell"], "KAPPA", 44)

    def test_same_cell_multiple_measurements_pooled(self):
        mk = lambda n, ms: [dict(iter_seq=q, t_launch=q * ms / 1000.0) for q in range(n)]
        self.assertAlmostEqual(r.pooled_median_ms([mk(129, 10.0), mk(129, 10.0), mk(600, 30.0)]), 30.0, places=6)


class TestProbeJson(unittest.TestCase):
    def _probe(self):
        boots = one_span_run_boot()
        recs, t = [], 10000.0
        a = t
        for i in range(22):
            ev, t = _span(1000 + i, t, 5693.0 * 92 / 64, p=92); recs += ev; t += 0.01
        boots.append(("b92", recs, [(a, t)]))
        cells = [raw_cell(d=16, kind="TAU", alone_ms=f.tau_alone(16, 32, 8192, "DN") * 1000),
                 raw_cell(d=44, kind="TAU", alone_ms=f.tau_alone(44, 32, 8192, "DN") * 1000),
                 raw_cell(d=16, alone_ms=20.0, co_ms=26.0, a2_ms=20.0),
                 raw_cell(d=44, alone_ms=20.0, co_ms=26.0, a2_ms=20.0)]
        return r.build_probe(boots, cells, [30], [0.0], [22.0], raw_files={"tel_x.jsonl": "0" * 64})

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(r.ReduceRejected):
            r.loads_strict('{"a": 1, "a": 2}')

    def test_provenance_recorded(self):                            # R8-3
        p = self._probe()
        prov = p["provenance"]
        self.assertEqual(prov["reducer_sha256"], r.self_sha256())
        self.assertIn("tel_x.jsonl", prov["raw_files"])
        self.assertEqual(len(prov["load_calls"]["synthetic"]), 26)
        self.assertEqual(len(prov["cell_windows"]), 4)
        self.assertIn("C", prov["cell_windows"][2])
        self.assertAlmostEqual(p["kappa_cells"][(16, 32, 8192)]["c_median_ms"] / p["kappa_cells"][(16, 32, 8192)]["a_on_ms"], 1.3, places=6)

    def test_round_trip_through_calculator_cli(self):              # kills B21; reducer sha accepted
        p = self._probe()
        with tempfile.TemporaryDirectory() as td:
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K.json")
            with open(pi, "w") as fh:
                fh.write(r.to_json(p))
            rc = f.main(["--probe", pi, "--out", po])
            with open(po) as fh:
                rec = json.load(fh)
        self.assertEqual(rc, 0, rec.get("reasons"))
        self.assertEqual(rec["reducer_sha256"], r.self_sha256())
        self.assertIn("budget", rec)

    def test_reducer_sha_mismatch_blocks_seal(self):               # R8-3
        p = self._probe(); p["provenance"]["reducer_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as td:
            pi = os.path.join(td, "probe.json")
            with open(pi, "w") as fh:
                fh.write(r.to_json(p))
            self.assertEqual(f.main(["--probe", pi, "--out", os.path.join(td, "K.json")]), 2)


if __name__ == "__main__":
    unittest.main()
