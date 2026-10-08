"""CPU tests for the N2(3-a) U-2 probe HARNESS (n2_marks.py, n2_loadgen.py, n2_probe_run.py) against a fake server
(fake_n2_engine.py) -- PREREG_N2_3A rev12 §14-2 mark format spec, H-1, H-4…H-7, R11-1…R11-3, R11-7, §3-1, §4-2.
Registered-style run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_harness -v   (needs aiohttp; git not needed)
Every contract item has a test that FAILS on the corresponding harness mutant (n2_harness_mutants.py, lesson 53)."""
import asyncio
import importlib.util
import json
import os
import re
import tempfile
import time
import unittest

import aiohttp

import fake_n2_engine as fk
import n2_loadgen as lg
import n2_marks
import n2_probe_run as pr

HERE = os.path.dirname(os.path.abspath(__file__))
RED = pr.load_by_path(pr.REDUCER, "_n2_test_harness_reducer")
CALC = pr.load_by_path(pr.CALCULATOR, "_n2_test_harness_calculator")
TEST_CELL = dict(b=32, c=8192, c_in=7992, O_EA=2000, delta=8)       # fake-scale cell (trigger 271 tokens after c_in)


def _rd(path):
    with open(path) as fh:
        return fh.read()


def _js(path):
    with open(path) as fh:
        return json.load(fh)


def plan_of(calls):
    return {"calls": {str(d): n for d, n in calls.items()},
            "cells": {str(d): dict(TEST_CELL, d=d, K=RED.K_PROBE[d]) for d in (44, 16)}}


def run_opts(**kw):
    o = dict(stop={}, gate_preview=False, flush_interval_s=0.01, warmup_n=1, environ={},
             cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=10))
    o.update(kw)
    return o


def attempt(td, calls, engine_kw=None, server_factory=None, **kw):
    opts = run_opts(expected_calls=dict(calls), **kw)
    mk = server_factory or (lambda: fk.FakeServerHandle(dict({"chunk_every": 4}, **(engine_kw or {}))))
    return asyncio.run(pr.run_attempt(td, plan_of(calls), mk, opts))


def marks(adir, boot):
    with open(os.path.join(adir, f"marks_{boot}.txt")) as fh:
        return [ln.split() for ln in fh.read().splitlines() if ln.strip()]


def tel(adir, boot):
    with open(os.path.join(adir, f"tel_{boot}.jsonl")) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def probe(adir):
    with open(os.path.join(adir, "probe.json")) as fh:
        return json.load(fh)


async def one_cell_delta(delta):
    return await one_cell(delta=delta)


async def one_cell(engine_kw=None, d=44, base=None, exit_fn=None, session_kw=None, cell_kw=None, k=None, mw_dir=None,
                   delta=8):
    """A single probe cell against a fake engine; returns (marks lines, info or CellAborted, engine)."""
    td = mw_dir or tempfile.mkdtemp()
    srv = fk.FakeServerHandle(dict({"chunk_every": 4}, **(engine_kw or {})))
    tp = os.path.join(td, "tel.jsonl")
    open(tp, "x").close()
    srv.start({"PDMUX_R2_FIXED_DSM": str(d), "PDMUX_TELEMETRY_PATH": tp, "PDMUX_STICKY_PARTITION": "1"},
              os.path.join(td, "srv.log"))
    await srv.wait_healthy(None)
    mw = n2_marks.MarkWriter(os.path.join(td, "marks.txt"))
    session = lg.new_session(**(session_kw or {}))
    spec = lg.CellSpec(d=d, b=32, c=8192, c_in=7992, o_ea=2000, k=RED.K_PROBE[d] if k is None else k, delta=delta)
    fn = exit_fn or (lambda: srv.exit_code())
    res = None
    try:
        res = await lg.run_probe_cell(session, base or srv.base, mw, spec, "c0", "call-c0",
                                      lambda e: lg.classify_exception(e, fn), lg.TelemetryTail(tp),
                                      **dict(dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=5),
                                             **(cell_kw or {})))
    except n2_marks.CellAborted as e:
        res = e
    finally:
        await session.close()
        mw.close()
        await srv.stop()
    return [ln.split() for ln in mw.lines], res, srv.engine, tp


# ===================================================================================================================
class TestMarkWriter(unittest.TestCase):
    def test_arity_and_reasons_match_reducer(self):                 # C1: the writer IS the reducer's format
        self.assertEqual(n2_marks.ARITY, {"cell_start": 5, "cell_end": 2, "coload_sent": 2, "delta_l": 2, "fix_s": 2,
                                          "cell_abort": 3, "abort_detail": 2, "load_call_start": 2,
                                          "load_call_end": 2, "boot_plan": 2, "boot_abort": 2})
        src = _rd(pr.REDUCER)
        m = re.search(r"arity = (\{[^}]*\})", src)
        self.assertEqual(eval(m.group(1)), n2_marks.ARITY)
        self.assertEqual(set(n2_marks.REGISTERED_REASONS), set(RED.ABORT_REASONS))
        self.assertFalse(set(n2_marks.UNREGISTERED_REASONS) & set(RED.ABORT_REASONS))

    def test_unregistered_tag_and_reason_refused(self):
        with tempfile.TemporaryDirectory() as td:
            mw = n2_marks.MarkWriter(os.path.join(td, "m.txt"))
            with self.assertRaises(n2_marks.MarkError):
                mw.write("bench_start", 1.0)
            with self.assertRaises(n2_marks.MarkError):
                mw.boot_abort("budget_stop")
            with self.assertRaises(n2_marks.MarkError):
                mw.write("cell_end", "a b", 1.0)
            mw.close()

    def test_ids_unique_across_writers_of_one_attempt(self):         # C8 (H-1 셀 id 유일, R11-1 호출 id 유일)
        with tempfile.TemporaryDirectory() as td:
            reg = n2_marks.IdRegistry()
            a = n2_marks.MarkWriter(os.path.join(td, "a.txt"), reg)
            b = n2_marks.MarkWriter(os.path.join(td, "b.txt"), reg)
            a.cell_start("x1", 44, 32, 8192)
            with self.assertRaises(n2_marks.MarkError):
                b.cell_start("x1", 16, 32, 8192)
            a.load_call_start("call-x1")
            with self.assertRaises(n2_marks.MarkError):
                b.load_call_start("call-x1")
            a.close(); b.close()

    def test_marks_file_never_overwritten(self):                     # H-6
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.txt")
            w = n2_marks.MarkWriter(p); w.boot_plan(44, 14); w.close()
            with self.assertRaises(FileExistsError):
                n2_marks.MarkWriter(p)

    def test_clock_contract(self):                                   # C2 / ABORT_CLOCK
        self.assertTrue(n2_marks.clock_contract_ok())


# ===================================================================================================================
class TestClassify(unittest.TestCase):                                # H-4, registered order
    def rf(self, inner, nbytes=0):
        return lg.RequestFailed("r", inner, nbytes)

    def test_server_timeout_is_cell_timeout_first(self):
        e = aiohttp.ServerTimeoutError("sock_read")
        self.assertIsInstance(e, aiohttp.ClientConnectionError)    # the trap the order exists for
        self.assertEqual(lg.classify_exception(self.rf(e)), "cell_timeout")
        self.assertEqual(lg.classify_exception(self.rf(e), lambda: 1), "cell_timeout")
        self.assertEqual(lg.classify_exception(asyncio.TimeoutError()), "cell_timeout")

    def test_connector_error_before_bytes_only(self):
        ce = aiohttp.ClientConnectorError(aiohttp.client_reqrep.ConnectionKey("h", 1, False, False, None, None, None),
                                          ConnectionRefusedError(111, "refused"))
        self.assertEqual(lg.classify_exception(self.rf(ce, 0)), "http_connection_error")
        self.assertEqual(lg.classify_exception(self.rf(ce, 0), lambda: 1), "http_connection_error")
        self.assertEqual(lg.classify_exception(self.rf(ce, 10)), "client_exception")
        self.assertEqual(lg.classify_exception(self.rf(aiohttp.ServerDisconnectedError(), 0)), "client_exception")

    def test_server_down_only_with_exit_code(self):
        e = self.rf(aiohttp.ServerDisconnectedError(), 500)
        self.assertEqual(lg.classify_exception(e, lambda: None), "client_exception")
        self.assertEqual(lg.classify_exception(e, lambda: 1), "server_down")
        self.assertEqual(lg.classify_exception(lg.CoLoadPayloadInvalid("x"), lambda: 1), "server_down")

    def test_payload_cancel_runner_stop_other(self):
        self.assertEqual(lg.classify_exception(lg.CoLoadPayloadInvalid("x")), "coload_payload_invalid")
        self.assertEqual(lg.classify_exception(lg.StreamCancelFailed("x")), "stream_cancel_failed")
        self.assertEqual(lg.classify_exception(lg.RunnerStop("x"), lambda: 1), "runner_stop")
        self.assertEqual(lg.classify_exception(ValueError("x")), "client_exception")
        self.assertEqual(lg.classify_exception(lg.WiringCheckFailed("x")), "wiring_check_failed")
        for r in ("client_exception", "wiring_check_failed", "boot_health_timeout"):     # rev13 G-8: registered VD
            self.assertEqual(RED.abort_code(r), "ABORT_" + r.upper())
            self.assertEqual(RED.ABORT_REASONS[r], RED.VALUE_DEPENDENT)
        self.assertEqual(n2_marks.UNREGISTERED_REASONS, ())
        self.assertEqual(pr._reason_class("boot_health_timeout", 0), RED.VALUE_INDEPENDENT)
        self.assertEqual(pr._reason_class("boot_health_timeout", 1), RED.VALUE_DEPENDENT)
        for r in ("client_crash", "node_failure"):                                    # G-9: removed
            self.assertNotIn(r, RED.ABORT_REASONS)

    def test_coload_payload_trace_rule(self):                        # §3-1 ABORT_TRACE
        self.assertEqual(lg.payload([1, 2], 1, "r", coload=True)["sampling_params"]["max_new_tokens"], 1)
        for bad in (0, 2, None):
            with self.assertRaises(lg.TraceInvalid):
                lg.payload([1], bad, "r", coload=True)

    def test_lower_median_rule(self):                                 # §4-2: L_(int((b-1)*0.5)), server _percentile
        self.assertEqual(lg.lower_median([4, 1, 3, 2]), 2)
        self.assertEqual(lg.lower_median(list(range(32))), 15)
        self.assertEqual(lg.lower_median([7]), 7)

    def test_seed_rule(self):                                         # §4-2 seed = SHA-256 first 8 bytes, unsigned
        import hashlib
        self.assertEqual(lg.seed_of("n2|PROBE|32|8192|1"),
                         int.from_bytes(hashlib.sha256(b"n2|PROBE|32|8192|1").digest()[:8], "big"))
        self.assertEqual(lg.decode_prompt("PROBE", 32, 8192, 1, 10), lg.decode_prompt("PROBE", 32, 8192, 1, 20)[:10])
        self.assertNotEqual(lg.coload_prompt(32, 8192, 1, 0), lg.coload_prompt(32, 8192, 1, 1))


# ===================================================================================================================
class TestCell(unittest.TestCase):
    def test_good_cell_marks_order_clock_and_reducer(self):          # C1 C2 C4 C5 C16 C17 C18
        lines, info, eng, tp = asyncio.run(one_cell())
        self.assertIsInstance(info, dict, info)
        tags = [l[0] for l in lines]
        self.assertEqual(tags, ["cell_start", "load_call_start", "coload_sent", "load_call_end", "cell_end",
                                "fix_s", "delta_l"])
        t = {l[0]: float(l[-1]) for l in lines if l[0] in ("cell_start", "load_call_start", "coload_sent",
                                                            "load_call_end", "cell_end")}
        self.assertTrue(t["cell_start"] <= t["load_call_start"] <= t["coload_sent"] <= t["load_call_end"]
                        <= t["cell_end"])
        # same clock as the engine: marks bracket the engine's own events
        ev = [json.loads(x) for x in _rd(tp).splitlines() if x.strip()]
        spans = [e for e in ev if e["event"] == "prefill_span_start"]
        co = [e for e in spans if e["extend_tokens"] == e["n_new"] * 8192]
        self.assertLess(t["cell_start"], spans[0]["timestamp_monotonic_s"])
        self.assertLessEqual(t["coload_sent"], min(eng.coload_arrivals))           # read BEFORE the first send
        self.assertLessEqual(min(eng.coload_arrivals), co[0]["timestamp_monotonic_s"])
        # exactly K co-load requests, all at once (no pacing): every one arrived before the first co span ended
        self.assertEqual(len(eng.coload_arrivals), RED.K_PROBE[44])
        self.assertEqual(sum(e["n_new"] for e in co), RED.K_PROBE[44])
        self.assertLess(max(eng.coload_arrivals) - min(eng.coload_arrivals), 0.03)
        # drain: no decode iteration launched between cell_end and the end of the file
        self.assertFalse([e for e in ev if e["event"] == "decode_iteration" and e["t_launch"] > t["cell_end"]])
        # the registered reducer reads this cell (marks + telemetry) as REDUCED
        cells, calls, plan, aborts, issues = RED.parse_marks([" ".join(l) for l in lines])
        self.assertEqual(issues, [])
        recs = [e for e in ev]
        out = RED.extract_cell_windows(RED._slice(recs, t["cell_start"], t["cell_end"]),
                                       {"coload_sent": t["coload_sent"]}, (44, 32, 8192))
        self.assertEqual(len(out["W"]), 128)
        self.assertGreaterEqual(len(out["A2"]), 128)
        # fix_s is the overhead, not the cell duration (harness definition G-6)
        fix = float([l for l in lines if l[0] == "fix_s"][0][2])
        self.assertLess(fix, 0.5 * (t["cell_end"] - t["cell_start"]))

    def test_cancel_threshold_includes_delta(self):                   # §4-2 step 5: base + 16 + 128 + delta
        lines, info, eng, tp = asyncio.run(one_cell_delta(300))
        self.assertIsInstance(info, dict, info)
        self.assertGreaterEqual(info["median_at_cancel"] - info["base"], 16 + 128 + 300)

    def test_delta_l_sign(self):                                      # Δℓ = ℓ_d − ℓ_co (§8-4)
        lines, info, eng, tp = asyncio.run(one_cell({"co_response_delay_s": 0.05}))
        dl = float([l for l in lines if l[0] == "delta_l"][0][2])
        self.assertLess(dl, -0.035)
        self.assertGreater(dl, -0.08)

    def test_d16_k18(self):
        lines, info, eng, tp = asyncio.run(one_cell(d=16))
        self.assertIsInstance(info, dict, info)
        self.assertEqual(len(eng.coload_arrivals), 18)

    def _aborted(self, res, reason):
        lines, err, eng, tp = res
        self.assertIsInstance(err, n2_marks.CellAborted, err)
        self.assertEqual(err.reason, reason)
        tags = [l[0] for l in lines]
        self.assertEqual(tags[-1], "cell_end")                        # finally, always last
        ab = [l for l in lines if l[0] == "cell_abort"]
        self.assertEqual(len(ab), 1)
        self.assertEqual(ab[0][2], reason)
        self.assertLessEqual(float(ab[0][3]), float(lines[-1][2]))   # t = decision time <= cell_end
        self.assertIn("abort_detail", tags)
        if "load_call_start" in tags:
            self.assertIn("load_call_end", tags)
        self.assertNotIn("fix_s", tags)
        cells, calls, plan, aborts, issues = RED.parse_marks([" ".join(l) for l in lines])
        self.assertEqual(issues, [])
        return lines

    def test_read_timeout_is_cell_timeout(self):
        lines = self._aborted(asyncio.run(one_cell({"mode": ["hang"]}, session_kw={"sock_read": 0.3})),
                              "cell_timeout")
        detail = [l for l in lines if l[0] == "abort_detail"][0][2]
        self.assertIn(detail, ("ServerTimeoutError", "SocketTimeoutError"))   # aiohttp 3.13: Socket < Server

    def test_cell_wall_timeout(self):
        self._aborted(asyncio.run(one_cell({"tau_s": 0.02}, cell_kw={"cell_timeout_s": 0.4})), "cell_timeout")

    def test_refused_is_http_connection_error(self):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        self._aborted(asyncio.run(one_cell(base=f"http://127.0.0.1:{port}")), "http_connection_error")

    def test_mid_stream_drop_with_exit_is_server_down(self):
        self._aborted(asyncio.run(one_cell({"drop_after": 40, "mode": ["exit_on_drop"]})), "server_down")

    def test_mid_stream_drop_without_exit_is_vd(self):
        self._aborted(asyncio.run(one_cell({"drop_after": 40})), "client_exception")

    def test_bad_coload_payload(self):                                # F3
        self._aborted(asyncio.run(one_cell({"mode": ["bad_coload"]})), "coload_payload_invalid")

    def test_drain_without_idle_evidence_is_cancel_failure(self):     # §4-2 step 6
        self._aborted(asyncio.run(one_cell({"mode": ["no_idle_snapshot"]}, cell_kw={"drain_timeout_s": 0.5})),
                      "stream_cancel_failed")


# ===================================================================================================================
class TestRunner(unittest.TestCase):
    def test_small_attempt_reduces_ok_and_paths(self):               # C9 C12 C14 C19 (+ H-6 relative paths)
        with tempfile.TemporaryDirectory() as td:
            adir, s = attempt(td, {44: 2, 16: 1})
            self.assertEqual(os.path.basename(adir), "probe_attempt_1")
            p = probe(adir)
            self.assertEqual(p["probe_status"], "OK", p["measurements"] + p["boot_issues"])
            self.assertEqual([b["telemetry"] for b in p["provenance"]["boots"]], ["tel_b44.jsonl", "tel_b16.jsonl"])
            m = marks(adir, "b44")
            self.assertEqual(m[0], ["boot_plan", "44", "2"])
            sub = _js(os.path.join(adir, "substrate.json"))
            self.assertEqual(sub["energy_sampler"]["state"], "off")
            self.assertTrue(sub["clock"]["contract_ok"])
            # F-6: a record after the last cell_end
            last_end = max(float(l[2]) for l in m if l[0] == "cell_end")
            self.assertTrue(any(e.get("timestamp_monotonic_s", 0) > last_end and e["event"] == "prefill_span_start"
                                for e in tel(adir, "b44")))
            # second attempt goes to a new directory only if the stop rule allows it (no gate result -> refused)
            with self.assertRaises(SystemExit):
                pr.new_attempt_dir(td)

    def test_h5_flush_wait_before_stop(self):                         # H-5
        with tempfile.TemporaryDirectory() as td:
            handles = []

            def mk():
                h = fk.FakeServerHandle({"chunk_every": 4}); handles.append(h); return h
            adir, s = attempt(td, {44: 1}, server_factory=mk, flush_interval_s=0.3, boot_order=(44,))
            last_end = max(float(l[2]) for l in marks(adir, "b44") if l[0] == "cell_end")
            self.assertGreaterEqual(handles[0].stop_called_at - last_end, 2 * 0.3)

    def test_runner_stop_deadline(self):                              # H-5 boot_abort runner_stop (VD)
        with tempfile.TemporaryDirectory() as td:
            adir, s = attempt(td, {44: 2}, deadline=time.monotonic() - 1, boot_order=(44,))
            m = marks(adir, "b44")
            self.assertEqual([l[1] for l in m if l[0] == "boot_abort"], ["runner_stop"])
            p = probe(adir)
            self.assertEqual(p["probe_status"], "VALUE_DEPENDENT")
            self.assertIn("ABORT_RUNNER_STOP", p["reason_codes"])

    def test_server_exit_before_health_is_vi_and_next_boot_runs(self):   # R11-2 per-boot death
        with tempfile.TemporaryDirectory() as td:
            hs = iter([fk.FakeServerHandle(fail_start="exit"), fk.FakeServerHandle({"chunk_every": 4})])
            adir, s = attempt(td, {44: 1, 16: 1}, server_factory=lambda: next(hs))
            self.assertEqual([l[:2] for l in marks(adir, "b44")], [["boot_plan", "44"], ["boot_abort", "server_down"]])
            self.assertEqual([b["n_cells"] for b in s["boots"]], [0, 1])
            p = probe(adir)
            self.assertEqual(p["probe_status"], "VALUE_INDEPENDENT", p["boot_issues"])
            self.assertIn("CELL_COUNT_SHORT_DEATH", p["reason_codes"])

    def test_health_timeout_classified_before_stop(self):             # H-4 (3): our SIGTERM is not a server exit
        with tempfile.TemporaryDirectory() as td:
            adir, s = attempt(td, {44: 1}, server_factory=lambda: fk.FakeServerHandle(fail_start="timeout"),
                              boot_order=(44,))
            self.assertEqual([l[1] for l in marks(adir, "b44") if l[0] == "boot_abort"], ["boot_health_timeout"])

    def test_wiring_check_classified_before_stop(self):               # sticky ON wiring; H-4 (3) not our SIGTERM
        with tempfile.TemporaryDirectory() as td:
            adir, s = attempt(td, {44: 1}, engine_kw={"sticky_log": False}, boot_order=(44,))
            self.assertEqual([l[1] for l in marks(adir, "b44") if l[0] == "boot_abort"], ["wiring_check_failed"])

    def test_server_env_and_flags(self):                              # §3-1
        env = pr.server_env({"PDMUX_TRUE_DUAL_WORKER": "1", "PDMUX_LA_COORD": "1", "PDMUX_ROOT": "/r", "X": "1"},
                            44, "/t.jsonl", "rid", "/g.json")
        self.assertEqual((env["PDMUX_STICKY_PARTITION"], env["PDMUX_R2_POLICY"], env["PDMUX_R2_FIXED_DSM"],
                          env["PDMUX_PHASE_EVENTS"], env["PDMUX_GREEN_READOUT"]), ("1", "fixed", "44", "1", "1"))
        self.assertNotIn("PDMUX_TRUE_DUAL_WORKER", env)
        self.assertNotIn("PDMUX_LA_COORD", env)
        self.assertEqual(env["PDMUX_ROOT"], "/r")
        self.assertEqual(pr.env_problems({"PDMUX_ROOT": "/r", "PDMUX_FOO": "1"}), ["PDMUX_FOO"])
        # the N3 rev6 server text (same flags, same values, cudagraph ON)
        sh = _rd(os.environ.get("N2_N3_SH") or os.path.join(HERE, "..", "n3_capacity", "n3_capacity.sh"))
        block = sh[sh.index("SRV=(python -m sglang.launch_server"):]
        block = block[:block.index(")")]
        vals = {"$MODEL": pr.MODEL, "$BACKEND": "flashinfer", "$CFG": "CFG", "$MEM_FRACTION": "0.82",
                "$MAX_RUNNING": "48", "$CTX": "16384", "$SERVER_SEED": "1"}
        toks = [vals.get(t.strip('"'), t.strip('"')) for t in block.split()[3:]]
        toks = toks[:toks.index("--host")]
        self.assertEqual(toks, pr.server_flags("CFG"))
        self.assertNotIn("--disable-cuda-graph", pr.server_flags("CFG"))

    def test_plan_checks(self):                                       # R11-1 constants come from the reducer
        good = {"calls": {"44": 14, "16": 7}, "cells": {str(d): dict(TEST_CELL, d=d, K=RED.K_PROBE[d])
                                                         for d in (44, 16)}}
        self.assertEqual(pr.plan_problems(good, RED), [])
        for mut in (lambda p: p["cells"]["44"].update(K=9), lambda p: p["cells"]["16"].update(b=8),
                    lambda p: p["calls"].update({"44": 13}), lambda p: p["cells"]["44"].update(c_in=8192)):
            bad = json.loads(json.dumps(good)); mut(bad)
            self.assertTrue(pr.plan_problems(bad, RED))

    def test_emit_plan_pre_probe_values(self):
        with tempfile.TemporaryDirectory() as td:
            p = pr.emit_plan(os.path.join(td, "PROBE_PLAN.json"))
            der = CALC.derived_constants()["ealpha"]
            self.assertEqual(p["cells"]["44"]["c_in"], der["44,32,8192"]["c_in"])
            self.assertEqual(p["cells"]["16"]["K"], 18)
            self.assertEqual(pr.plan_problems(p, RED, check_sources=True), [])
            stale = dict(p, calculator_sha256="0" * 64)
            self.assertTrue(pr.plan_problems(stale, RED, check_sources=True))
            with self.assertRaises(FileExistsError):
                pr.emit_plan(os.path.join(td, "PROBE_PLAN.json"))

    def test_stop_rule(self):                                         # H-7: reason_codes SET, not strings
        ok = lambda codes: {"reprobe_allowed": True, "reason_codes": codes}
        self.assertTrue(pr.next_attempt_allowed([])[0])
        self.assertTrue(pr.next_attempt_allowed([ok(["RAW_MISSING"])])[0])
        self.assertFalse(pr.next_attempt_allowed([ok(["RAW_MISSING"]), ok(["RAW_MISSING"])])[0])
        self.assertFalse(pr.next_attempt_allowed([ok(["A", "B"]), ok(["B", "A"])])[0])
        self.assertTrue(pr.next_attempt_allowed([ok(["RAW_MISSING"]), ok(["THETA_N_SHORT"])])[0])
        self.assertFalse(pr.next_attempt_allowed([{"reprobe_allowed": False, "reason_codes": []}])[0])
        self.assertFalse(pr.next_attempt_allowed([None])[0])

    def test_attempt_dirs_exclusive(self):                             # H-6
        with tempfile.TemporaryDirectory() as td:
            n, p1 = pr.new_attempt_dir(td)
            with open(os.path.join(p1, "GATE_PREVIEW.json"), "w") as fh:                 # rev13 G-15: a preview
                json.dump({"reprobe_allowed": True, "reason_codes": ["RAW_MISSING"]}, fh)   # never allows a re-probe
            with self.assertRaises(SystemExit):
                pr.new_attempt_dir(td)
            with open(os.path.join(p1, "K_POST_PROBE.json"), "w") as fh:
                json.dump({"reprobe_allowed": True, "reason_codes": ["RAW_MISSING"]}, fh)
            n2, p2 = pr.new_attempt_dir(td)
            self.assertEqual((n, n2), (1, 2))
            self.assertNotEqual(p1, p2)
            with open(os.path.join(p2, "K_POST_PROBE.json"), "w") as fh:
                json.dump({"reprobe_allowed": True, "reason_codes": ["RAW_MISSING"]}, fh)
            with self.assertRaises(SystemExit):
                pr.new_attempt_dir(td)


class TestEndToEnd(unittest.TestCase):
    def test_registered_probe_14_7_reduces_and_reproduces(self):     # harness -> reducer -> calculator re-reduction
        with tempfile.TemporaryDirectory() as td:
            opts = run_opts(gate_preview=True)
            adir, s = asyncio.run(pr.run_attempt(td, plan_of(RED.PROBE_CALLS),
                                                 lambda: fk.FakeServerHandle({"chunk_every": 4}), opts))
            p = probe(adir)
            self.assertEqual(p["probe_status"], "OK", [m for m in p["measurements"] if m["status"] != "REDUCED"]
                             + p["boot_issues"])
            self.assertEqual(len(p["measurements"]), 21)
            self.assertEqual(p["provenance"]["expected_calls"], {"16": 7, "44": 14})
            self.assertGreaterEqual(len(p["theta_spans"]["64"]), 20)
            self.assertGreaterEqual(len(p["theta_spans"]["92"]), 20)
            self.assertEqual(len(p["delta_l_s"]), 21)
            self.assertEqual(len(p["fix_s"]), 21)
            # the calculator re-reduces the raw files from the attempt directory (gate-location rule) byte-identically
            cwd = os.getcwd()
            os.chdir(adir)
            try:
                self.assertIsNone(CALC.reproduce_probe("probe.json", _js("probe.json")))
            finally:
                os.chdir(cwd)
            prev = _js(os.path.join(adir, "GATE_PREVIEW.json"))
            self.assertEqual(prev["probe_status"], "OK")
            self.assertNotIn("PROBE_BYTES_MISMATCH", prev.get("reason_codes", []))
            self.assertNotIn("RAW_PATH_MISSING", prev.get("reason_codes", []))


if __name__ == "__main__":
    unittest.main()
