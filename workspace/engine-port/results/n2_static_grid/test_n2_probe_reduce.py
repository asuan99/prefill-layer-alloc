"""CPU tests for n2_probe_reduce.py (PREREG_N2_3A rev14 smoke/shared-window tool; rev11: R7-1, R8-1/2/3, R9-1/2/5/6/8, R10-1…R10-5).
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
             contaminated_snapshot=True, t0=0.0, q0=0, seq0=0, lag=0, non_co_after=False, a2_inflight=0,
             w_off=0, inflight_off_leaks=0, co_spans=1, span_tok_s=None, span_req=2, coload=None):
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
    for k in range(230):                       # iter_seq 81..208 is the ctx band; w_off of them run unsplit (0,108)
        it(alone_ms, False, pair108 if 100 <= k < 100 + w_off else pair)
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
    if span_tok_s is not None:                 # co_spans contiguous 2-request spans at throughput span_tok_s
        D = 8192.0 * span_req / span_tok_s
        n_c = int(co_spans * D * 1000.0 / co_ms) - 1
        for k in range(co_spans):
            recs += [dict(event="prefill_span_start", span_seq=seq0 + 4 + k, timestamp_monotonic_s=tc0 + k * D,
                          extend_tokens=8192 * span_req, n_new=span_req),
                     dict(event="prefill_span_end", span_seq=seq0 + 4 + k, timestamp_monotonic_s=tc0 + (k + 1) * D)]
    else:                                      # one co span carrying the registered co-load K_PROBE[d] (R11-1)
        kp = r.K_PROBE.get(d, 8) if coload is None else coload
        recs.append(dict(event="prefill_span_start", span_seq=seq0 + 2, timestamp_monotonic_s=tc0,
                         extend_tokens=8192 * kp, n_new=kp))
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
    for _ in range(inflight_off_leaks):
        it(10.0, True, pair108)
    if span_tok_s is not None:
        t = tc0 + co_spans * 8192.0 * span_req / span_tok_s + 0.0005
    else:
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


def write_boot(td, name, cells=(), theta=None, delta_l=0.0, fix_s=22.0, drop_mark=None, extra_lines=(),
               telemetry_tail=None, plan=None):
    """Write one probe boot (telemetry JSONL + marks, §14-2 MARK FORMAT) to td. cells: raw_cell kwargs dicts laid out
    sequentially, each wrapped by its own load call; boot_plan = the cells' d counts unless `plan` is given.
    Per-cell test knobs: _omit (mark tags to leave out), _abort (cell_abort reason), _abort_t ("end" | "mid" | float),
    _id (cell id), _no_marks (write no marks for the cell), _no_call (no load-call wrap), _end_extra (seconds added
    to cell_end). theta: (records, calls) appended after the cells. telemetry_tail: raw text appended to the JSONL."""
    recs, lines, t, q, seq = [], [], 0.0, 0, 0
    counts = {}
    for kw in cells:
        counts[kw.get("d", 44)] = counts.get(kw.get("d", 44), 0) + 1
    for d, n in sorted((counts if plan is None else plan).items()):
        lines.append(f"boot_plan {d} {n}")
    for i, kw in enumerate(cells):
        kw = dict(kw)
        omit, abort, abort_t = set(kw.pop("_omit", ())), kw.pop("_abort", None), kw.pop("_abort_t", "end")
        cid, no_marks, no_call = kw.pop("_id", f"{name}_c{i}"), kw.pop("_no_marks", False), kw.pop("_no_call", False)
        end_extra = kw.pop("_end_extra", 0.0)
        d, b, c = kw.get("d", 44), kw.get("b", 32), kw.get("c", 8192)
        cr, mk = raw_cell(t0=t, q0=q, seq0=seq, **kw)
        recs += cr
        cl = [("cell_start", f"cell_start {cid} {d} {b} {c} {t}"),
              ("coload_sent", f"coload_sent {cid} {mk['coload_sent']}"),
              ("cell_end", f"cell_end {cid} {mk['t1'] + end_extra}"), ("delta_l", f"delta_l {cid} {delta_l}"),
              ("fix_s", f"fix_s {cid} {fix_s}")]
        if abort:
            ta = {"end": mk["t1"], "mid": mk["coload_sent"] + 0.01}.get(abort_t, abort_t)
            cl += [("cell_abort", f"cell_abort {cid} {abort} {ta}"), ("abort_detail", f"abort_detail {cid} TimeoutError")]
        if not no_call:
            cl += [("load_call_start", f"load_call_start {cid} {mk['coload_sent'] - 0.0001}"),
                   ("load_call_end", f"load_call_end {cid} {mk['t1']}")]
        if not no_marks:
            lines += [ln for tag, ln in cl if tag not in omit]
        q = max(e.get("iter_seq", 0) for e in cr) + 10
        seq += 100
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
    lines += list(extra_lines)
    tp, mp = os.path.join(td, f"tel_{name}.jsonl"), os.path.join(td, f"marks_{name}.txt")
    with open(tp, "w") as fh:
        fh.write("\n".join(json.dumps(e) for e in recs) + "\n" + (telemetry_tail or ""))
    with open(mp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(boot_id=name, telemetry=tp, marks=mp)


def expected_of(cells):
    out = {16: 0, 44: 0}
    for kw in cells:
        out[kw.get("d", 44)] = out.get(kw.get("d", 44), 0) + 1
    return out


def full_cells(kappa=1.3, tau_ratio=1.0, theta64=5693.0, spans=(4, 9)):
    """The registered probe: 14 binding-cell calls at d = 44 and 7 at d = 16 (R9-7); each call's co-load is
    `spans` contiguous 2-request spans (K 8 -> 4, K 18 -> 9) whose throughput is theta64 (p = 64) or
    theta64 * 92/64 (p = 92) -- theta comes only from the calls' own co spans, as registered."""
    out = []
    for d, n, ns, th in ((44, r.PROBE_CALLS[44], spans[0], theta64), (16, r.PROBE_CALLS[16], spans[1], theta64 * 92 / 64)):
        ms = f.tau_alone(d, 32, 8192, "DN") * 1000 * tau_ratio
        out += [dict(d=d, alone_ms=ms, a2_ms=ms, co_ms=ms * kappa, co_spans=ns, span_tok_s=th) for _ in range(n)]
    return out


def full_boots(td, cells=None, **kw):
    return [write_boot(td, "k", full_cells() if cells is None else cells, **kw)]


FROZEN = ("n2_feasibility.py", "n2_probe_reduce.py", "test_n2_feasibility.py", "test_n2_probe_reduce.py")


def freeze_repo(td, drop=(), override=None, writers=True, commit=None, dirty=False):
    """Freeze-checkout fixture (H-2, R11-5): a git repository td/frz holding copies of the frozen files and a stand-in
    harness mark writer, committed; FREEZE.json (untracked, like the real gate copy) records commit = HEAD and the
    file sha256. Knobs: drop (leave files out), override (extra/changed entries), writers=False, commit (a different
    commit sha), dirty (edit the mark writer after the commit AND update its manifest sha -- only git sees it)."""
    import shutil
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(td, "frz")
    os.makedirs(repo, exist_ok=True)
    for fn in FROZEN:
        shutil.copy(os.path.join(here, fn), os.path.join(repo, fn))
    with open(os.path.join(repo, "mark_writer.py"), "w") as fh:
        fh.write("# stand-in for the harness mark writer\n")
    g = lambda *a: subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo,
                                  capture_output=True, text=True, check=True).stdout.strip()
    g("init", "-q"); g("add", "-A"); g("commit", "-q", "-m", "freeze")
    head = g("rev-parse", "HEAD")
    if dirty:
        with open(os.path.join(repo, "mark_writer.py"), "a") as fh:
            fh.write("# edited after the freeze commit\n")
    sha = lambda fn: hashlib.sha256(open(os.path.join(repo, fn), "rb").read()).hexdigest()
    files = {fn: sha(fn) for fn in FROZEN + ("mark_writer.py",) if fn not in drop}
    files.update(override or {})
    man = {"commit": commit or head, "files": files, "harness_mark_writers": ["mark_writer.py"] if writers else []}
    path = os.path.join(repo, "FREEZE.json")
    with open(path, "w") as fh:
        json.dump(man, fh)
    return path


def theta_boots(theta64=5693.0, n92=22):
    (_, recs64, calls64), = one_span_run_boot()
    if theta64 != 5693.0:
        recs64, calls64, t = [], [], 0.0
        for i in range(26):
            ev, t1 = _span(i + 1, t, theta64); recs64 += ev; t = t1 + 0.01
        calls64 = [(0.0, t)]
    recs, t, a = [], 10000.0, 10000.0
    for i in range(n92):
        ev, t = _span(1000 + i, t, theta64 * 92 / 64, p=92); recs += ev; t += 0.01
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

    def test_c_value_window_is_first_128(self):                    # rev14 (A') truncation (kills J07 / J08)
        win = self._win()
        self.assertEqual(len(win["C"]), 128)
        self.assertEqual(win["ranges"]["C"], (231, 358))               # first 128 co-located iterations by iter_seq
        self.assertEqual(win["n_colocated_all"], 200)
        self.assertEqual(win["ranges"]["C_all"], (231, 430))
        self.assertAlmostEqual(win["ctx_mid"]["C"], 8048 + 293.5, places=6)      # ctx of seq 231..358
        self.assertAlmostEqual(win["ctx_mid"]["C_all"], 8048 + 329.5, places=6)

    def test_c_window_excludes_off_type_leaks(self):               # C14
        win = self._win(leaks=300)
        self.assertEqual(len(win["C"]), 128)
        self.assertEqual(win["n_colocated_all"], 200)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["C"])), 26.0, places=6)

    def test_c_window_excludes_on_type_leaks(self):                # R9-3 X05: ON leaks carry PAIR[d]
        win = self._win(on_leaks=300)
        self.assertEqual(len(win["C"]), 128)
        self.assertEqual(win["n_colocated_all"], 200)
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["C"])), 26.0, places=6)

    def test_alone_windows_exclude_in_flight_and_trim(self):       # R9-3 X14 / X07: A2 position pinned
        win = self._win(a2_inflight=20)
        a2_first = 230 + 200 + 16 + 1                              # 230 alone + 200 C + 16 trimmed, 1-based
        self.assertEqual(win["ranges"]["A2"], (a2_first, a2_first + 127 + 20))
        self.assertAlmostEqual(statistics.median(r.s_per_intervals(win["A2"])), 22.0, places=6)

    def test_contaminated_snapshot_excluded(self):
        win = self._win()
        self.assertAlmostEqual(win["ctx_mid"]["A1"], 8192 - 0.5, places=6)
        # C value window iter_seq 231..358 -> ctx = (c - 144) + (seq - 1); a used contaminated snapshot would give ~8290
        self.assertAlmostEqual(win["ctx_mid"]["C"], 8341.5, places=6)

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
    def _reduce(self, cells, theta=False, expected=None, **kw):
        with tempfile.TemporaryDirectory() as td:
            boots = [write_boot(td, "k", cells, **kw)]
            if theta:
                t64, t92 = theta_boots()
                boots += [write_boot(td, "t64", theta=t64), write_boot(td, "t92", theta=t92)]
            p = r.reduce_probe(boots, expected_calls=expected_of(cells) if expected is None else expected)
            shas = {b[k]: hashlib.sha256(open(b[k], "rb").read()).hexdigest() for b in boots
                    for k in ("telemetry", "marks")}
        return p, shas

    def _st(self, p):
        return [m["status"] for m in p["measurements"]]

    def test_completeness_and_censoring(self):                     # R9-1, kills X04
        p, _ = self._reduce([dict(), dict(n_c=100), dict()])
        self.assertEqual(self._st(p), ["REDUCED", r.KAPPA_CENSORED, "REDUCED"])
        self.assertEqual(p["probe_status"], r.KAPPA_CENSORED)
        self.assertEqual(p["kappa_cells"][(44, 32, 8192)]["n_measurements"], 2)

    def test_censor_records_w_co_and_filter_losses(self):          # R10-5, kills Y04
        p, _ = self._reduce([dict(n_c=100, leaks=300)])
        m = p["measurements"][0]
        self.assertEqual(m["status"], r.KAPPA_CENSORED)
        self.assertEqual(m["censor_detail"], dict(w_co=400, c=100, loss_bs_ne_b=0, loss_not_in_flight=300,
                                                  loss_pair_mismatch=0))

    def test_censor_boundary_127_128(self):                        # kills Y10
        p, _ = self._reduce([dict(n_c=127), dict(n_c=128)])
        self.assertEqual(self._st(p), [r.KAPPA_CENSORED, "REDUCED"])

    def test_value_dependent_reject_non_co_span(self):             # R9-3 X09, §7-2(c)
        p, _ = self._reduce([dict(), dict(non_co_after=True)])
        self.assertEqual(self._st(p), ["REDUCED", r.VALUE_DEPENDENT])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)

    def test_partial_split_window_rejected(self):                  # kills X10 (PE 0.906 < 0.99)
        p, _ = self._reduce([dict(w_off=12)])
        self.assertEqual(self._st(p), [r.VALUE_DEPENDENT])
        self.assertIn("PE-iter", p["measurements"][0]["reason"])

    def test_c_excludes_in_flight_unsplit_leaks(self):             # kills X06 (pair condition) under the rev14 truncation
        p, _ = self._reduce([dict(inflight_off_leaks=300)])
        self.assertEqual(self._st(p), ["REDUCED"])
        self.assertAlmostEqual(p["measurements"][0]["c_median_ms"], 26.0, places=6)
        win = r.extract_cell_windows(*(lambda recs, mk: (recs, {"coload_sent": mk["coload_sent"]}, (44, 32, 8192)))(*raw_cell(inflight_off_leaks=300)))
        self.assertEqual(win["n_colocated_all"], 200)             # in-flight (0,108) leaks are not co-located iterations
        p, _ = self._reduce([dict(n_c=100, inflight_off_leaks=300)])
        self.assertEqual(self._st(p), [r.KAPPA_CENSORED])        # without the pair filter the leaks would fill C

    # ---- R10-1: class = most severe applicable check; mark loss never launders censored / VD into VI ----
    def test_mark_loss_does_not_launder(self):                     # audit S1-S3, S6
        for omit, want in ((("fix_s",), r.KAPPA_CENSORED), (("delta_l",), r.KAPPA_CENSORED),
                           (("cell_end",), r.VALUE_DEPENDENT), (("coload_sent",), r.VALUE_DEPENDENT)):
            p, _ = self._reduce([dict(n_c=100, _omit=omit)])
            self.assertEqual(p["probe_status"], want, omit)
        p, _ = self._reduce([dict(non_co_after=True, _omit=("fix_s",))])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)

    def test_missing_scalar_mark_on_good_cell_is_vi(self):         # kills Y05
        p, _ = self._reduce([dict(_omit=("fix_s",))])
        self.assertEqual(self._st(p), [r.VALUE_INDEPENDENT])

    def test_abort_reason_table(self):                             # H-1 reason -> class
        for reason, want in (("server_down", r.VALUE_INDEPENDENT), ("cell_timeout", r.VALUE_DEPENDENT),
                             ("whatever", r.VALUE_DEPENDENT)):
            p, _ = self._reduce([dict(_abort=reason, _omit=("cell_end",))])
            self.assertEqual(self._st(p), [want], reason)

    def test_vd_and_vi_mixed_status_is_vd(self):                   # kills Y01
        p, _ = self._reduce([dict(non_co_after=True), dict(_omit=("fix_s",))])
        self.assertEqual(sorted(self._st(p)), sorted([r.VALUE_DEPENDENT, r.VALUE_INDEPENDENT]))
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)

    def test_censored_dominates_status(self):
        p, _ = self._reduce([dict(n_c=100), dict(non_co_after=True)])
        self.assertEqual(p["probe_status"], r.KAPPA_CENSORED)

    def test_probe_d_violation_vi(self):                           # kills Y06
        p, _ = self._reduce([dict(d=24)], expected={16: 0, 44: 0})
        self.assertEqual(self._st(p), [r.VALUE_INDEPENDENT])

    def test_malformed_mark_line_vi(self):                         # kills Y07
        p, _ = self._reduce([dict()], extra_lines=["cell_end oops"])
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)

    # ---- R10-2: completeness against the raw co spans, unique ids, exact counts, load-call wrapping ----
    def test_omitted_cell_is_orphan_co_span(self):                 # audit S4
        p, _ = self._reduce([dict(), dict(n_c=100, _no_marks=True)])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        self.assertTrue(any("co span" in x["reason"] for x in p["boot_issues"]))

    def test_cell_id_reuse_rejected(self):                         # audit S5b
        p, _ = self._reduce([dict(_id="x"), dict(_id="x")])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        self.assertTrue(any("duplicate" in x["reason"] for x in p["boot_issues"]))

    def _codes(self, p):
        return {m["reason_code"] for m in p["measurements"] if m["status"] != "REDUCED"} | \
               {b["reason_code"] for b in p["boot_issues"]}

    def test_plan_sum_and_counts(self):                             # R11-2 (audit S7 now via boot_plan)
        cells = [dict(), dict()]
        p, _ = self._reduce(cells)
        self.assertEqual(p["probe_status"], "OK")
        p, _ = self._reduce(cells, expected={16: 0, 44: 3})        # plans do not add up to the registered count
        self.assertIn("BOOT_PLAN_SUM", self._codes(p)); self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        p, _ = self._reduce(cells, plan={44: 3}, expected={16: 0, 44: 3})   # started fewer than planned, no death
        self.assertIn("CELL_COUNT_SHORT", self._codes(p)); self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        p, _ = self._reduce(cells, plan={44: 1}, expected={16: 0, 44: 1})   # excess
        self.assertIn("CELL_COUNT_EXCESS", self._codes(p))
        p, _ = self._reduce(cells, plan={})                         # no boot_plan
        self.assertIn("BOOT_PLAN_MISSING", self._codes(p))

    def test_per_boot_death(self):                                  # R11-2, audit (c') / Z06
        tail = '{"event": "decode_it'
        p, _ = self._reduce([dict()], plan={44: 3}, expected={16: 0, 44: 3}, telemetry_tail=tail)
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)   # this boot died after its last cell
        self.assertIn("CELL_COUNT_SHORT_DEATH", self._codes(p))
        p, _ = self._reduce([dict()], plan={44: 3}, expected={16: 0, 44: 3},
                            extra_lines=["boot_abort server_down 1e12"])
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)
        p, _ = self._reduce([dict(), dict()], plan={44: 3}, expected={16: 0, 44: 3},
                            extra_lines=["boot_abort server_down 0.5"])        # died BEFORE its last cell -> VD
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        p, _ = self._reduce([dict(), dict()], plan={44: 1}, expected={16: 0, 44: 1}, telemetry_tail=tail)
        self.assertIn("CELL_COUNT_EXCESS", self._codes(p))       # excess + death stays VD (Z06)
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        with tempfile.TemporaryDirectory() as td:                    # another boot's death does not excuse a shortfall
            a = write_boot(td, "a", [dict()], telemetry_tail=tail)
            b = write_boot(td, "b", [dict(_id="b0")], plan={44: 2})
            p = r.reduce_probe([a, b], expected_calls={16: 0, 44: 3})
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        self.assertIn("CELL_COUNT_SHORT", self._codes(p))

    def test_boot_health_timeout_rule(self):                        # rev13 G-8
        with tempfile.TemporaryDirectory() as td:                    # no cell started in this boot -> VI
            b = write_boot(td, "k", [], plan={44: 1}, extra_lines=["boot_abort boot_health_timeout 5.0"])
            p = r.reduce_probe([b], expected_calls={16: 0, 44: 1})
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)
        self.assertIn("CELL_COUNT_SHORT_DEATH", self._codes(p))
        p, _ = self._reduce([dict()], extra_lines=["boot_abort boot_health_timeout 1e12"])   # a cell exists -> VD
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        for reason in ("client_exception", "wiring_check_failed", "boot_health_timeout"):     # as cell aborts -> VD
            p, _ = self._reduce([dict(_abort=reason)])
            self.assertEqual(self._st(p), [r.VALUE_DEPENDENT], reason)
        self.assertEqual(set(r.ABORT_REASONS), {"server_down", "http_connection_error", "cell_timeout",
                                                "coload_payload_invalid", "stream_cancel_failed", "runner_stop",
                                                "client_exception", "wiring_check_failed", "boot_health_timeout"})

    def test_boot_abort_reasons(self):                              # H-5 runner_stop, Z15
        for line, want in (("boot_abort server_down 1e12", r.VALUE_INDEPENDENT),
                           ("boot_abort runner_stop 1e12", r.VALUE_DEPENDENT),
                           ("boot_abort budget_stop 1e12", r.VALUE_DEPENDENT)):
            p, _ = self._reduce([dict()], extra_lines=[line])
            self.assertEqual(p["probe_status"], want, line)

    def test_raw_missing_is_vi(self):                               # Z20 (redefined): raw absence only -> VI
        with tempfile.TemporaryDirectory() as td:
            b = dict(write_boot(td, "k", [dict()]), telemetry=os.path.join(td, "absent.jsonl"))
            p = r.reduce_probe([b], expected_calls={16: 0, 44: 1})
        self.assertEqual((p["probe_status"], p["reason_codes"]), (r.VALUE_INDEPENDENT, ["RAW_MISSING"]))

    def test_reducer_failure_is_vd(self):                           # Z16, Z12
        with tempfile.TemporaryDirectory() as td:
            b = write_boot(td, "k", [dict()])
            with open(b["telemetry"]) as fh:
                lines = fh.read().split("\n")
            lines = [ln for ln in lines if ln]
            lines[-1] = lines[-1][:10]; lines[-2] = lines[-2][:10]       # two damaged lines at the end
            with open(b["telemetry"], "w") as fh:
                fh.write("\n".join(lines))
            p = r.reduce_probe([b], expected_calls={16: 0, 44: 1})
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        self.assertEqual([x["status"] for x in p["boot_issues"] if x["reason_code"] == "REDUCER_FAILURE"],
                         [r.VALUE_DEPENDENT])

    def test_coload_size_bc_and_calls(self):                        # R11-1
        p, _ = self._reduce([dict(coload=10)])
        self.assertEqual(p["measurements"][0]["reason_code"], "COLOAD_SIZE_MISMATCH")
        p, _ = self._reduce([dict(b=8)])
        self.assertIn("CELL_BC_INVALID", [x[1] for x in p["measurements"][0]["all_issues"]])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        p, _ = self._reduce([dict(_no_call=True), dict(_no_call=True)],
                            extra_lines=["load_call_start big 0.0", "load_call_end big 1e9"])
        self.assertIn("CALL_CELLS", self._codes(p))
        p, _ = self._reduce([dict(_id="x0"), dict(_id="x1")],
                            extra_lines=["load_call_start x0 1e9", "load_call_end x0 1e9"])
        self.assertIn("CALL_DUPLICATE", self._codes(p))

    def test_span_in_two_cells_and_call_overlap(self):              # Z04, Z05
        p, _ = self._reduce([dict(_end_extra=1000.0), dict()])
        self.assertIn("CO_SPAN_CELLS", self._codes(p))
        p, _ = self._reduce([dict()], extra_lines=["load_call_start extra 0.0", "load_call_end extra 1e9"])
        self.assertIn("CALL_WRAP", [x[1] for x in p["measurements"][0]["all_issues"]])

    def test_abort_reasons_and_censor_promotion(self):              # R11-3, Z01, Z02
        for reason in ("coload_payload_invalid", "stream_cancel_failed", "cell_timeout", "runner_stop"):
            p, _ = self._reduce([dict(_abort=reason)])
            self.assertEqual(self._st(p), [r.VALUE_DEPENDENT], reason)
        p, _ = self._reduce([dict(n_c=100, _abort="server_down")])              # complete co-load then VI abort
        self.assertEqual((p["probe_status"], p["reason_codes"]), (r.KAPPA_CENSORED, ["C_CENSORED_ABORTED"]))
        p, _ = self._reduce([dict(n_c=100, _abort="server_down", _abort_t="mid")])  # aborted during the co-load
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)
        p, _ = self._reduce([dict(_abort="server_down", _omit=("cell_end",))])  # Z18: no cell_end, own spans fine
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)
        self.assertEqual(p["measurements"][0]["abort_detail"], "TimeoutError")

    def test_duplicate_cell_measurement_status(self):              # Z11
        p, _ = self._reduce([dict(_id="x"), dict(_id="x")])
        self.assertEqual(p["measurements"][0]["status"], r.VALUE_DEPENDENT)
        self.assertIn("MARK_DUPLICATE", [x[1] for x in p["measurements"][0]["all_issues"]])

    def test_reason_codes_enumerated_and_stable(self):             # R11-5 stop rule compares codes, not text
        seen = []
        for kw in (dict(n_c=100), dict(non_co_after=True), dict(_omit=("fix_s",)), dict(b=8), dict(coload=9),
                   dict(_abort="weird"), dict(w_off=12)):
            p, _ = self._reduce([kw])
            for m in p["measurements"] + p["boot_issues"]:
                if m["status"] != "REDUCED":
                    self.assertIn(m["reason_code"], r.REASON_CODES)
                    seen.append(m["reason_code"])
        self.assertIn("ABORT_UNREGISTERED", seen)
        codes = []
        for name in ("try1", "try2"):                                # same cause, different paths -> same codes
            with tempfile.TemporaryDirectory() as td:
                b = dict(write_boot(td, name, [dict()]), marks=os.path.join(td, name, "gone.txt"))
                codes.append(r.reduce_probe([b], expected_calls={16: 0, 44: 1})["reason_codes"])
        self.assertEqual(codes[0], codes[1])

    def test_registered_counts_default(self):
        self.assertEqual(r.PROBE_CALLS, {16: 7, 44: 14})
        self.assertEqual(r.K_PROBE, {16: 18, 44: 8})                # smoke co-load size only (rev14: NOT the E-alpha K)
        self.assertNotEqual(r.K_PROBE, f.K_CO)
        self.assertEqual(r.BIND_CELL, (32, 8192))
        with tempfile.TemporaryDirectory() as td:                   # default = registered 14 / 7
            p = r.reduce_probe([write_boot(td, "k", [dict()])])
        self.assertEqual(p["probe_status"], r.VALUE_DEPENDENT)
        self.assertEqual(p["provenance"]["expected_calls"], {16: 7, 44: 14})

    def test_cell_must_be_in_one_load_call(self):
        p, _ = self._reduce([dict(_no_call=True)])
        self.assertEqual(self._st(p), [r.VALUE_DEPENDENT])

    def test_raw_failures_classified(self):                        # R10-1 crash handling
        p, _ = self._reduce([dict()], telemetry_tail='{"event": "decode_it')
        self.assertEqual(self._st(p), ["REDUCED"])
        self.assertEqual(p["probe_status"], r.VALUE_INDEPENDENT)
        with tempfile.TemporaryDirectory() as td:
            b = write_boot(td, "k", [dict()])
            with open(b["telemetry"]) as fh:
                lines = fh.read().split("\n")
            lines[5] = lines[5][:10]
            with open(b["telemetry"], "w") as fh:
                fh.write("\n".join(lines))
            self.assertEqual(r.reduce_probe([b], expected_calls={16: 0, 44: 1})["probe_status"], r.VALUE_DEPENDENT)
            b2 = dict(b, telemetry=os.path.join(td, "absent.jsonl"))
            self.assertEqual(r.reduce_probe([b2], expected_calls={16: 0, 44: 0})["probe_status"], r.VALUE_INDEPENDENT)
            b3 = write_boot(td, "u", [dict()], extra_lines=["load_call_start orphan 1.0"])
            self.assertEqual(r.reduce_probe([b3], expected_calls={16: 0, 44: 1})["probe_status"], r.VALUE_DEPENDENT)

    # ---- estimator / sources ----
    def test_kappa_equal_weight_median(self):                      # kills X01 / X02 variants
        cells = [dict(alone_ms=20.0, a2_ms=20.0, co_ms=32.0, n_c=500),   # kappa 1.6, long C
                 dict(alone_ms=20.0, a2_ms=20.0, co_ms=40.0, n_c=200),   # kappa 2.0
                 dict(alone_ms=20.0, a2_ms=20.0, co_ms=40.0, n_c=200)]   # kappa 2.0
        p, _ = self._reduce(cells)
        k = p["kappa_cells"][(44, 32, 8192)]
        self.assertAlmostEqual(k["kappa_median"], 2.0, places=6)   # pooled C would give 1.6, mean 1.867
        self.assertEqual(k["n_measurements"], 3)

    def test_kappa_even_n_median(self):                            # kills Y02 (median_low) / Y03 (max)
        cells = [dict(alone_ms=20.0, a2_ms=20.0, co_ms=ms) for ms in (32.0, 32.0, 40.0, 40.0)]
        p, _ = self._reduce(cells)
        self.assertAlmostEqual(p["kappa_cells"][(44, 32, 8192)]["kappa_median"], 1.8, places=6)

    def test_tau_from_kappa_a1(self):                              # R9-2; kills Y12 (a2_ms != alone_ms)
        cells = [dict(d=d, alone_ms=20.0, a2_ms=30.0, co_ms=26.0) for d in (16, 44)]
        p, _ = self._reduce(cells)
        self.assertEqual(set(p["tau_cells"]), {(16, 32, 8192), (44, 32, 8192)})
        self.assertAlmostEqual(p["tau_cells"][(44, 32, 8192)]["s_per_median_ms"], 20.0, places=6)

    def test_provenance_self_computed(self):                       # R9-5
        self.assertNotIn("reducer_sha256", inspect.signature(r.reduce_probe).parameters)
        p, shas = self._reduce([dict()])
        self.assertEqual(p["provenance"]["reducer_sha256"], r.self_sha256())
        self.assertEqual(p["provenance"]["raw_files"], shas)
        self.assertEqual(len(p["provenance"]["boots"]), 1)
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


class TestCalculatorCli(unittest.TestCase):
    """raw files -> reduce_probe -> to_json -> calculator CLI (re-reduction R10-3, freeze manifest H-2).
    The CLI mutates calculator module state (one process per real run); each test restores it."""
    from test_n2_feasibility import _STATE

    def setUp(self):
        import copy
        self._snap = {n: copy.deepcopy(getattr(f, n)) for n in self._STATE}

    def tearDown(self):
        import copy
        for n, v in self._snap.items():
            cur = getattr(f, n)
            if isinstance(cur, dict):
                cur.clear(); cur.update(copy.deepcopy(v))
            else:
                setattr(f, n, copy.deepcopy(v))

    def _run(self, cells=None, tamper=None, manifest=None, **kw):
        with tempfile.TemporaryDirectory() as td:
            boots = full_boots(td, cells, **kw)
            p = r.reduce_probe(boots)
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K.json")
            txt = r.to_json(p)
            if tamper:
                d = json.loads(txt); tamper(d); txt = json.dumps(d, indent=1, sort_keys=True)
            with open(pi, "w") as fh:
                fh.write(txt)
            fz = freeze_repo(td, **(manifest or {})) if manifest is not False else None
            argv = ["--probe", pi, "--out", po] + (["--freeze", fz] if fz else [])
            rc = f.main(argv)
            with open(po) as fh:
                return rc, json.load(fh), p

    def test_round_trip_through_calculator_cli(self):              # kills B21; end to end raw files -> K
        rc, rec, p = self._run()
        self.assertEqual(p["probe_status"], "OK")
        self.assertEqual(rc, 0, rec.get("reasons"))
        self.assertEqual(rec["K"], {"16": 33, "44": 15})                # rev14: the registered constant, never derived
        self.assertFalse(rec["judgment_input"])
        self.assertEqual(rec["reducer_sha256"], r.self_sha256())
        self.assertIn("budget", rec)
        self.assertFalse(rec["reprobe_allowed"])

    def test_tampered_probe_json_rejected(self):                   # R10-3 (audit tamper.py)
        cells = full_cells() + [dict(n_c=100)]
        def launder(d):
            d["measurements"] = [m for m in d["measurements"] if m["status"] != r.KAPPA_CENSORED]
            d["probe_status"] = "OK"
        rc, rec, p = self._run(cells, tamper=launder)
        self.assertEqual(p["probe_status"], r.KAPPA_CENSORED)
        self.assertEqual((rc, rec["K"], rec["reprobe_allowed"]), (2, None, False))
        self.assertTrue(any("re-reduction" in x for x in rec["reasons"]))

    def test_reducer_sha_mismatch_blocks_seal(self):               # R8-3 (also caught by re-reduction)
        rc, rec, _ = self._run(tamper=lambda d: d["provenance"].update(reducer_sha256="f" * 64))
        self.assertEqual(rc, 2)

    def test_status_routes(self):                                  # R9-1 end to end
        for cells, want in ((full_cells() + [dict(n_c=100)], False),
                            (full_cells()[:-1] + [dict(full_cells()[-1], non_co_after=True)], False),
                            (full_cells()[:-1] + [dict(full_cells()[-1], _omit=("fix_s",))], True)):
            rc, rec, p = self._run(cells)
            self.assertEqual((rc, rec["K"], rec["reprobe_allowed"]), (2, None, want), p["probe_status"])
            self.assertEqual(rec["reason_codes"], p["reason_codes"])
        self.assertEqual(rec["reason_codes"], ["SCALAR_MARK_MISSING"])

    def test_theta_short_value_independent(self):                  # n < 20 -> re-probe allowed
        cells = [dict(kw, span_req=1, co_spans=18) if kw["d"] == 16 else kw for kw in full_cells()]
        rc, rec, p = self._run(cells=cells)                       # d16 all 1-request spans: n(p=92) = 0 (N2C-56)
        self.assertEqual(p["probe_status"], "OK")
        self.assertEqual(rc, 2)
        self.assertTrue(rec["reprobe_allowed"])

    def test_not_sealable_rc2(self):                               # A11
        rc, rec, _ = self._run(cells=full_cells(tau_ratio=0.5, theta64=4500.0))
        self.assertEqual(rc, 2)
        self.assertFalse(rec["seal_allowed"])

    def test_value_tamper_and_expected_calls(self):                # Z08, Z07
        def bump(d):
            k = next(iter(d["kappa_cells"])); d["kappa_cells"][k]["kappa_median"] *= 1.5
        rc, rec, p = self._run(tamper=bump)
        self.assertEqual(p["probe_status"], "OK")
        self.assertEqual((rc, rec["reason_codes"]), (2, ["PROBE_BYTES_MISMATCH"]))
        with tempfile.TemporaryDirectory() as td:                    # reduced with a non-registered expected count
            cells = full_cells()[:-1]
            boots = full_boots(td, cells)
            p = r.reduce_probe(boots, expected_calls={16: 6, 44: 14})
            self.assertEqual(p["probe_status"], "OK")
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K.json")
            with open(pi, "w") as fh:
                fh.write(r.to_json(p))
            rc = f.main(["--probe", pi, "--out", po, "--freeze", freeze_repo(td)])
            with open(po) as fh:
                rec = json.load(fh)
        self.assertEqual((rc, rec["reason_codes"]), (2, ["PROBE_BYTES_MISMATCH"]))

    def test_raw_path_missing_reported_separately(self):            # R11-4 diagnosis
        with tempfile.TemporaryDirectory() as td:
            boots = full_boots(td)
            p = r.reduce_probe(boots)
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K.json")
            with open(pi, "w") as fh:
                fh.write(r.to_json(p))
            os.rename(boots[0]["marks"], boots[0]["marks"] + ".moved")
            rc = f.main(["--probe", pi, "--out", po, "--freeze", freeze_repo(td)])
            with open(po) as fh:
                rec = json.load(fh)
        self.assertEqual((rc, rec["reason_codes"], rec["reprobe_allowed"]), (2, ["RAW_PATH_MISSING"], False))

    def test_foreign_reducer_module_in_process(self):               # R11-4: M02 / N16 (audit m02run.py)
        import importlib.util
        import sys
        src = open(r.__file__).read().replace('kap.setdefault(cell, []).append(out["c_med"] / out["a_on"])',
                                              'kap.setdefault(cell, []).append(0.8 * out["c_med"] / out["a_on"])')
        saved = sys.modules.get("n2_probe_reduce")
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "foreign"))
            fp = os.path.join(td, "foreign", "n2_probe_reduce.py")
            with open(fp, "w") as fh:
                fh.write(src)
            spec = importlib.util.spec_from_file_location("n2_probe_reduce", fp)
            foreign = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foreign)
            sys.modules["n2_probe_reduce"] = foreign
            try:
                p = foreign.reduce_probe(full_boots(td))
                pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K.json")
                with open(pi, "w") as fh:
                    fh.write(foreign.to_json(p))
                rc = f.main(["--probe", pi, "--out", po, "--freeze", freeze_repo(td)])
                with open(po) as fh:
                    rec = json.load(fh)
            finally:
                if saved is not None:
                    sys.modules["n2_probe_reduce"] = saved
        self.assertEqual((rc, rec["reason_codes"]), (2, ["PROBE_BYTES_MISMATCH"]))
        self.assertTrue(any("reducer sha256 missing or differs" in x for x in rec["reasons"]))

    def test_git_commit_and_dirty(self):                            # R11-5: Z17 / Z10
        rc, rec, _ = self._run(manifest=dict(commit="0" * 40))
        self.assertEqual(rc, 2)
        self.assertTrue(any("HEAD is not manifest.commit" in x for x in rec["reasons"]))
        rc, rec, _ = self._run(manifest=dict(dirty=True))
        self.assertEqual(rc, 2)
        self.assertTrue(any("uncommitted changes" in x for x in rec["reasons"]))

    def test_freeze_manifest(self):                                # R9-4 / H-2
        for man in (False, dict(drop=("n2_feasibility.py",)), dict(drop=("n2_probe_reduce.py",)),
                    dict(writers=False), dict(override={"extra_frozen.py": "0" * 64})):
            rc, rec, _ = self._run(manifest=man)
            self.assertEqual(rc, 2, man)
            self.assertTrue(rec["reasons"])
        rc, rec, _ = self._run(manifest={})
        self.assertEqual(rc, 0, rec["reasons"])


if __name__ == "__main__":
    unittest.main()
