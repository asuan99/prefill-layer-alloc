"""CPU tests for the PDMUX_PHASE_DIAG GPU-gate scorers (scripts/vessl/engine_gate_diag.py).

The scorers are fed (a) the telemetry a real `event_loop_pdmux` run writes on CPU with PDMUX_PHASE_DIAG ON
(pdmux_loop_fakes + the lagging fake CUDA events of test_phase_diag), which must PASS, and (b) single-defect
corruptions of it, each of which must FAIL (negative controls, lesson 53: a scorer that passes everything is an
identity).  The GPU driver (`run`) is not exercised here.
"""

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts" / "vessl"))

import engine_gate_diag as GD  # noqa: E402
import test_phase_diag as TPD  # noqa: E402
from pdmux_loop_fakes import LoopTestBase, run_loop  # noqa: E402


class StampingTelemetry:
    """RecordingTelemetry + the writer's own timestamp field and event name, as tel.jsonl carries them."""

    def __init__(self, log):
        self.log = log
        self.records = []

    def emit(self, event, phase, request_id="", **fields):
        rec = dict(fields, event=event, phase=phase, timestamp_monotonic_s=time.perf_counter())
        self.records.append(rec)
        self.log.append(("telemetry", event, dict(fields)))
        return True

    def mark_phase(self, phase):
        return self.emit("phase_marker", phase)

    def close(self, timeout_s=5.0):
        return None


class ScorerTest(LoopTestBase):
    def _records(self, slo=False):
        if slo:
            os.environ["PDMUX_SLO_SCHED"] = "1"
        sched = self.install(TPD._build(True, True, lag=3, slo=slo))
        tel = StampingTelemetry(sched.log)
        sched.pdmux_telemetry = tel
        run_loop(sched)
        return tel.records, [list(x) for x in sched.sm_counts]

    def test_real_loop_records_pass(self):
        for slo in (False, True):
            with self.subTest(slo=slo):
                recs, sm = self._records(slo)
                res = GD.check_diag_records(recs, sm)
                self.assertEqual(res["verdict"], GD.PASS, res["why"])
                self.assertGreater(res["facts"]["n_chunks"], 0)
                if slo:
                    self.assertEqual(GD.check_slo_switch(recs)["verdict"], GD.PASS)

    def test_each_defect_fails(self):
        recs, sm = self._records(slo=True)

        def first(name):
            return next(i for i, r in enumerate(recs) if r["event"] == name)

        mutations = {
            "gpu_ms_null": lambda rs: rs[first("prefill_chunk")].update(chunk_gpu_ms=None),
            "gpu_ms_exceeds_wall": lambda rs: [r.update(chunk_gpu_ms=1e6) for r in rs if r["event"] == "prefill_chunk"],
            "issue_reversed": lambda rs: rs[first("prefill_chunk")].update(
                t_issue_start=rs[first("prefill_chunk")]["t_issue_end"] + 1),
            "split_overflow": lambda rs: rs[first("prefill_chunk")].update(split_index=99),
            "chunk_dropped": lambda rs: rs.pop(first("prefill_chunk")),
            "rids_missing": lambda rs: rs[first("prefill_span_start")].pop("rids"),
            "drain_negative": lambda rs: rs[first("drain")].update(decode_sync_ms=-1.0),
            "drain_reason_unknown": lambda rs: rs[first("drain")].update(reason="other"),
            "sm_not_configured": lambda rs: rs[first("prefill_chunk")].update(prefill_sms=1),
            "no_boot": lambda rs: rs.pop(first("phase_diag_boot")),
            "no_drains": lambda rs: [rs.remove(r) for r in [r for r in rs if r["event"] == "drain"]],
        }
        for name, mut in mutations.items():
            with self.subTest(mutation=name):
                rs = copy.deepcopy(recs)
                mut(rs)
                self.assertEqual(GD.check_diag_records(rs, sm)["verdict"], GD.FAIL, name)
        self.assertEqual(GD.check_diag_records(recs, sm, "x PDMUX_PHASE_DIAG disabled after failure")["verdict"],
                         GD.FAIL)
        same = copy.deepcopy(recs)
        for r in same:
            if r["event"] == "drain" and r["reason"] == "slo_switch":
                r["stream_idx_to"] = r["stream_idx_from"]
        self.assertEqual(GD.check_slo_switch(same)["verdict"], GD.FAIL)
        none = [r for r in recs if not (r["event"] == "drain" and r["reason"] == "slo_switch")]
        self.assertEqual(GD.check_slo_switch(none)["verdict"], GD.UNRESOLVED)


class PureScorerTest(unittest.TestCase):
    def test_identity(self):
        a = {"x": [1, 2, 3], "y": [4]}
        self.assertEqual(GD.check_identity(a, dict(a))["verdict"], GD.PASS)
        self.assertEqual(GD.check_identity(a, {"x": [1, 2, 3], "y": [5]})["verdict"], GD.FAIL)
        self.assertEqual(GD.check_identity(a, {"x": [1, 2, 3]})["verdict"], GD.FAIL)
        self.assertEqual(GD.check_identity(a, {"x": [1, 2, 3], "y": []})["verdict"], GD.FAIL)
        self.assertEqual(GD.check_identity({}, {})["verdict"], GD.FAIL)

    def test_decode_probe(self):
        good = [{"event": "decode_forward", "decode_fwd_gpu_ms": 3.1}]
        self.assertEqual(GD.check_decode_probe(good)["verdict"], GD.PASS)
        self.assertEqual(GD.check_decode_probe([{"event": "decode_forward", "decode_fwd_gpu_ms": None}])["verdict"],
                         GD.FAIL)
        self.assertEqual(GD.check_decode_probe(good, "decode probe disabled after failure")["verdict"], GD.FAIL)

    def _clock_case(self):
        t = time.perf_counter()
        boot = {"event": "phase_diag_boot", "t_perf": t, "t_monotonic": t + 1e-6,
                "perf_counter_impl": "clock_gettime(CLOCK_MONOTONIC)"}
        span = {"event": "prefill_span_start", "rids": ["a"], "timestamp_monotonic_s": t + 1.0}
        client = {"request_id": "a", "error": "", "t_sent_mono": t + 0.5, "token_t_mono": [t + 1.5, t + 1.6]}
        return boot, span, client

    def test_clocks(self):
        boot, span, client = self._clock_case()
        self.assertEqual(GD.check_clocks([boot, span], [client])["verdict"], GD.PASS)
        for name, edit in (
            ("impl", lambda b, s, c: b.update(perf_counter_impl="clock_gettime(CLOCK_REALTIME)")),
            ("offset", lambda b, s, c: b.update(t_monotonic=b["t_perf"] + 0.1)),
            ("sent_after_span", lambda b, s, c: c.update(t_sent_mono=s["timestamp_monotonic_s"] + 0.1)),
            ("token_before_span", lambda b, s, c: c.update(token_t_mono=[s["timestamp_monotonic_s"] - 0.1])),
            ("rid_missing", lambda b, s, c: s.update(rids=["b"])),
        ):
            with self.subTest(name=name):
                b, s, c = self._clock_case()
                edit(b, s, c)
                self.assertEqual(GD.check_clocks([b, s], [c])["verdict"], GD.FAIL, name)

    def test_score_dir_off_must_be_off(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for d in ("off", "on"):
                (root / d).mkdir()
            (root / "plan.json").write_text(json.dumps({"sm_counts": [[108, 0], [0, 108]]}))
            (root / "off" / "seq.json").write_text(json.dumps({"k": [1]}))
            (root / "on" / "seq.json").write_text(json.dumps({"k": [1]}))
            (root / "off" / "tel.jsonl").write_text("")            # OFF wrote telemetry -> D2 FAIL
            out = GD.score_dir(root)
            self.assertEqual(out["checks"]["D2_levels"]["verdict"], GD.FAIL)
            self.assertEqual(out["checks"]["D1_identity"]["verdict"], GD.PASS)
            self.assertEqual(out["verdict"], GD.FAIL)


if __name__ == "__main__":
    unittest.main()
