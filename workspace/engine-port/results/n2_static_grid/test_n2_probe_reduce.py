"""CPU tests for n2_probe_reduce.py (PREREG_N2_3A rev10: R7-1, R8-1/2/3, R9-1/2/5/6/8).
Registered run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_probe_reduce -v
(The N3 positive control loads ~2 GB of JSONL into one process; run suites sequentially, R9-8.)"""
import hashlib
import inspect
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


def raw_cell(d=44, b=32, c=8192, alone_ms=20.0, co_ms=26.0, a2_ms=22.0, n_c=200, leaks=0, on_leaks=0,
             contaminated_snapshot=True, t0=0.0, q0=0, seq0=0, lag=0, non_co_after=False, a2_inflight=0):
    """Synthetic RAW sticky-ON probe cell starting at time t0 / iter_seq q0: burst span, alone segment through the
    ctx band, coload_sent, (lag alone iterations), co span with n_c C iterations (+ `leaks` (0,108) OFF-type and
    `on_leaks` PAIR[d] ON-type not-in-flight iterations inside W_co), then A2: 16 trim + 128 at a2_ms, then 100 at
    40 ms (so a wrong A2 position is visible), optionally `a2_inflight` in-flight iterations after the first 20.
    Returns (records, marks dict)."""
    pair, pair108 = r.PAIR[d], r.PAIR[108]
    recs, t, q = [], t0, q0
    recs += [dict(event="prefill_span_start", span_seq=seq0 + 1, timestamp_monotonic_s=t0 + 0.0001,
                  extend_tokens=b * 100, n_new=b),
             dict(event="prefill_span_end", span_seq=seq0 + 1, timestamp_monotonic_s=t0 + 1.0)]
    t = t0 + 1.0
    recs.append(dict(event="runtime_snapshot", timestamp_monotonic_s=t, decode_iterations=q0 + 1,
                     context_p50=c - 144, active_decode_sequences=b))

    def it(step_ms, in_flight, pr, bs=b):
        nonlocal t, q
        q += 1; t += step_ms / 1000.0
        recs.append(dict(event="decode_iteration", iter_seq=q, t_launch=t, bs=bs, prefill_in_flight=in_flight,
                         prefill_sms=pr[0], decode_sms=pr[1]))
    for _ in range(230):
        it(alone_ms, False, pair)
    marks = {"coload_sent": t + 0.0005}
    t += 0.001
    for _ in range(lag):
        it(alone_ms, False, pair)
    if non_co_after:
        recs += [dict(event="prefill_span_start", span_seq=seq0 + 3, timestamp_monotonic_s=t + 0.0001,
                      extend_tokens=b * 100, n_new=b),
                 dict(event="prefill_span_end", span_seq=seq0 + 3, timestamp_monotonic_s=t + 0.0002)]
        t += 0.0003
    tc0 = t + 0.0005
    recs.append(dict(event="prefill_span_start", span_seq=seq0 + 2, timestamp_monotonic_s=tc0, extend_tokens=16384,
                     n_new=2))
    t = tc0
    if contaminated_snapshot:
        recs.append(dict(event="runtime_snapshot", timestamp_monotonic_s=t, decode_iterations=q + 5, context_p50=8193,
                         active_decode_sequences=b + 2))
    for _ in range(n_c):
        it(co_ms, True, pair)
    for _ in range(leaks):
        it(10.0, False, pair108)
    for _ in range(on_leaks):
        it(10.0, False, pair)
    t1 = t + 0.001
    recs.append(dict(event="prefill_span_end", span_seq=seq0 + 2, timestamp_monotonic_s=t1))
    t = t1
    for i in range(144):
        if i == 20:
            for _ in range(a2_inflight):
                it(50.0, True, pair)
        it(a2_ms, False, pair)
    for _ in range(100):
        it(40.0, False, pair)
    marks["t1"] = t + 0.01
    return recs, marks


def write_boot(td, name, cells=(), theta=None, delta_l=0.0, fix_s=22.0, drop_mark=None):
    """Write one probe boot (telemetry JSONL + marks) to td. cells: list of raw_cell kwargs dicts, laid out
    sequentially; each kappa cell is also one load call (harness spec). theta: (records, calls) appended after."""
    recs, lines, t, q, seq = [], [], 0.0, 0, 0
    for i, kw in enumerate(cells):
        d, b, c = kw.get("d", 44), kw.get("b", 32), kw.get("c", 8192)
        cr, mk = raw_cell(t0=t, q0=q, seq0=seq, **kw)
        recs += cr
        cid = f"{name}_c{i}"
        lines += [f"cell_start {cid} {d} {b} {c} {t}", f"coload_sent {cid} {mk['coload_sent']}",
                  f"cell_end {cid} {mk['t1']}", f"delta_l {cid} {delta_l}", f"fix_s {cid} {fix_s}",
                  f"load_call_start {cid} {mk['coload_sent']}", f"load_call_end {cid} {mk['t1']}"]
        q = max(e.get("iter_seq", 0) for e in cr) + 10
        seq += 10
        t = mk["t1"] + 5.0
    if theta is not None:
        trecs, tcalls = theta
        off = t + 10.0
        for e in trecs:
            e = dict(e)
            for k in ("timestamp_monotonic_s", "t_launch"):
                if k in e:
                    e[k] += off
            if "span_seq" in e:
                e["span_seq"] += 100000
            recs.append(e)
        for j, (a, bb) in enumerate(tcalls):
            lines += [f"load_call_start th{j} {a + off}", f"load_call_end th{j} {bb + off}"]
    if drop_mark:
        lines = [ln for ln in lines if not ln.startswith(drop_mark)]
    tp, mp = os.path.join(td, f"tel_{name}.jsonl"), os.path.join(td, f"marks_{name}.txt")
    with open(tp, "w") as fh:
        fh.write("\n".join(json.dumps(e) for e in recs) + "\n")
    with open(mp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(boot_id=name, telemetry=tp, marks=mp)


def theta_boots():
    (_, recs64, calls64), = one_span_run_boot()
    recs, t, a = [], 10000.0, 10000.0
    for i in range(22):
        ev, t = _span(1000 + i, t, 5693.0 * 92 / 64, p=92); recs += ev; t += 0.01
    return (recs64, calls64), (recs, [(a, t)])


def binding_cells(kappa=1.3):
    out = []
    for d in (16, 44):
        ms = f.tau_alone(d, 32, 8192, "DN") * 1000
        out.append(dict(d=d, alone_ms=ms, a2_ms=ms, co_ms=ms * kappa))
    return out


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

    def test_split_rule_at_least_one_and_unique_p(self):          # R8-7, kills B01 / B02 (assertion form, R9-8)
        ev1, t = _span(1, 0.0, 5693.0)
        ev2, t = _span(2, t + 1, 5693.0, extra=[(0, 108, 32)])     # one unsplit iteration -> still kept (>=1 rule)
        ev3, t = _span(3, t + 1, 5693.0, extra=[(92, 16, 32)])     # second split p -> dropped "mixed"
        th, ex, dr = r.reduce_theta([("b", ev1 + ev2 + ev3, [])])
        self.assertEqual(len(th.get(64, [])), 1)
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
    def _win(self, **kw):
        recs, mk = raw_cell(**kw)
        return r.extract_cell_windows(recs, {"coload_sent": mk["coload_sent"]}, (kw.get("d", 44), 32, 8192))

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

    def test_alone_window_from_raw(self):
        win = self._win()
        self.assertEqual(len(win["W"]), 128)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["W"])), 20.0, places=6)

    def test_kappa_parts_pinned(self):                             # kills B18 / B19 / X08 / M04
        win = self._win()
        c_med, a_on = r.kappa_parts(win)
        self.assertAlmostEqual(c_med, 26.0, places=6)
        mm = win["ctx_mid"]
        self.assertAlmostEqual(a_on, 20.0 + 2.0 * (mm["C"] - mm["A1"]) / (mm["A2"] - mm["A1"]), places=6)
        self.assertTrue(20.0 < a_on < 22.0)

    def test_c_window_excludes_off_type_leaks(self):               # C14
        win = self._win(leaks=300)
        self.assertEqual(len(win["C"]), 200)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["C"])), 26.0, places=6)

    def test_c_window_excludes_on_type_leaks(self):                # R9-3 X05: ON leaks carry PAIR[d]
        win = self._win(on_leaks=300)
        self.assertEqual(len(win["C"]), 200)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["C"])), 26.0, places=6)

    def test_alone_windows_exclude_in_flight_and_trim(self):       # R9-3 X14 / X07: A2 position pinned
        win = self._win(a2_inflight=20)
        a2_first = 230 + 200 + 16 + 1                              # 230 alone + 200 C + 16 trimmed, 1-based
        self.assertEqual(win["ranges"]["A2"], (a2_first, a2_first + 127 + 20))
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["A2"])), 22.0, places=6)

    def test_contaminated_snapshot_excluded(self):
        win = self._win()
        self.assertAlmostEqual(win["ctx_mid"]["A1"], 8192 - 0.5, places=6)
        # C iterations iter_seq 231..430 -> ctx = (c - 144) + (seq - 1); a used contaminated snapshot would give ~8290
        self.assertAlmostEqual(win["ctx_mid"]["C"], 8377.5, places=6)

    def test_l_lag_from_raw(self):
        self.assertEqual(self._win(lag=7)["l_lag_iters"], 7)

    def test_a_on_containment_guard(self):                         # R9-6
        self.assertAlmostEqual(r.a_on_interp(20.0, 0.0, 22.0, 10.0, 5.0), 21.0)
        for x in (-0.1, 10.1):
            with self.assertRaises(r.MeasurementRejected) as cm:
                r.a_on_interp(20.0, 0.0, 22.0, 10.0, x)
            self.assertEqual(cm.exception.klass, r.VALUE_DEPENDENT)

    def test_same_cell_multiple_windows_pooled(self):
        mk = lambda n, ms: [dict(iter_seq=q, t_launch=q * ms / 1000.0) for q in range(n)]
        self.assertAlmostEqual(r.pooled_median_ms([mk(129, 10.0), mk(129, 10.0), mk(600, 30.0)]), 30.0, places=6)


class TestReduceProbe(unittest.TestCase):
    def _reduce(self, cells, theta=False, **kw):
        with tempfile.TemporaryDirectory() as td:
            boots = [write_boot(td, "k", cells, **kw)]
            if theta:
                t64, t92 = theta_boots()
                boots += [write_boot(td, "t64", theta=t64), write_boot(td, "t92", theta=t92)]
            p = r.reduce_probe(boots)
            shas = {b[k]: hashlib.sha256(open(b[k], "rb").read()).hexdigest() for b in boots
                    for k in ("telemetry", "marks")}
        return p, shas

    def test_completeness_and_censoring(self):                     # R9-1, kills X04
        p, _ = self._reduce([dict(), dict(n_c=100), dict()])
        st = [m["status"] for m in p["measurements"]]
        self.assertEqual(st, ["REDUCED", r.KAPPA_CENSORED, "REDUCED"])
        self.assertEqual(p["probe_status"], r.KAPPA_CENSORED)
        self.assertEqual(p["kappa_cells"][(44, 32, 8192)]["n_measurements"], 2)

    def test_value_dependent_reject_non_co_span(self):             # R9-3 X09, §7-2(c)
        p, _ = self._reduce([dict(), dict(non_co_after=True)])
        self.assertEqual([m["status"] for m in p["measurements"]], ["REDUCED", r.VALUE_DEPENDENT])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)

    def test_value_independent_missing_mark(self):
        p, _ = self._reduce([dict()], drop_mark="coload_sent")
        self.assertEqual(p["measurements"][0]["status"], r.VALUE_INDEPENDENT)
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)

    def test_censored_dominates_status(self):
        p, _ = self._reduce([dict(n_c=100), dict(non_co_after=True)], drop_mark=None)
        self.assertEqual(p["probe_status"], r.KAPPA_CENSORED)

    def test_kappa_equal_weight_median(self):                      # rev10 estimator, kills X01 / X02 variants
        cells = [dict(alone_ms=20.0, a2_ms=20.0, co_ms=32.0, n_c=500),   # kappa 1.6, long C
                 dict(alone_ms=20.0, a2_ms=20.0, co_ms=40.0, n_c=200),   # kappa 2.0
                 dict(alone_ms=20.0, a2_ms=20.0, co_ms=40.0, n_c=200)]   # kappa 2.0
        p, _ = self._reduce(cells)
        k = p["kappa_cells"][(44, 32, 8192)]
        self.assertAlmostEqual(k["kappa_median"], 2.0, places=6)   # pooled C would give 1.6, mean 1.867
        self.assertEqual(k["n_measurements"], 3)

    def test_tau_from_kappa_a1(self):                              # R9-2
        p, _ = self._reduce(binding_cells())
        self.assertEqual(set(p["tau_cells"]), {(16, 32, 8192), (44, 32, 8192)})
        self.assertAlmostEqual(p["tau_cells"][(44, 32, 8192)]["s_per_median_ms"],
                               f.tau_alone(44, 32, 8192, "DN") * 1000, places=6)

    def test_provenance_self_computed(self):                       # R9-5
        self.assertNotIn("reducer_sha256", inspect.signature(r.reduce_probe).parameters)
        p, shas = self._reduce([dict()])
        self.assertEqual(p["provenance"]["reducer_sha256"], r.self_sha256())
        self.assertEqual(p["provenance"]["raw_files"], shas)
        m = p["measurements"][0]
        self.assertEqual(m["cell"], [44, 32, 8192])
        self.assertIn("C", m["ranges"])

    def test_scalars_from_marks_and_raw(self):
        p, _ = self._reduce([dict(lag=3), dict(lag=5)], delta_l=0.25, fix_s=21.0)
        self.assertEqual(p["l_lag_iters"], [3, 5])
        self.assertEqual(p["delta_l_s"], [0.25, 0.25])
        self.assertEqual(p["fix_s"], [21.0, 21.0])

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(r.ReduceRejected):
            r.loads_strict('{"a": 1, "a": 2}')

    def _cli(self, p, tamper=None):
        with tempfile.TemporaryDirectory() as td:
            pi, po, fz = (os.path.join(td, n) for n in ("probe.json", "K.json", "FREEZE.json"))
            txt = r.to_json(p)
            if tamper:
                d = json.loads(txt); tamper(d); txt = json.dumps(d)
            with open(pi, "w") as fh:
                fh.write(txt)
            with open(fz, "w") as fh:
                json.dump({"n2_feasibility.py": f._sha256_file(f.__file__), "n2_probe_reduce.py": r.self_sha256()}, fh)
            rc = f.main(["--probe", pi, "--out", po, "--freeze", fz])
            with open(po) as fh:
                return rc, json.load(fh)

    def test_round_trip_through_calculator_cli(self):              # kills B21; end to end raw files -> K
        p, _ = self._reduce(binding_cells(), theta=True)
        self.assertEqual(p["probe_status"], "OK")
        rc, rec = self._cli(p)
        self.assertEqual(rc, 0, rec.get("reasons"))
        self.assertEqual(rec["K"], {"16": 22, "44": 10})
        self.assertEqual(rec["reducer_sha256"], r.self_sha256())
        self.assertIn("budget", rec)

    def test_reducer_sha_mismatch_blocks_seal(self):               # R8-3
        p, _ = self._reduce(binding_cells(), theta=True)
        rc, rec = self._cli(p, tamper=lambda d: d["provenance"].update(reducer_sha256="f" * 64))
        self.assertEqual(rc, 2)

    def test_censored_probe_cli_no_reprobe(self):                  # R9-1 end to end
        cells = binding_cells() + [dict(n_c=100)]
        p, _ = self._reduce(cells, theta=True)
        rc, rec = self._cli(p)
        self.assertEqual((rc, rec["K"], rec["reprobe_allowed"]), (2, None, False))


if __name__ == "__main__":
    unittest.main()
