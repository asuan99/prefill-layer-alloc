"""Gate for PDMUX_PHASE_EVENTS (E-1 cap campaign instrument, default OFF).

PREREG: results/e1_cap_vessl/PREREG_E1_CAP_VESSL_2026-10-04.md sec 5.

The real `event_loop_pdmux` is driven on CPU through pdmux_loop_fakes (legacy
loop, no R2 policy, no SLO env -- the static-arm path E-1 runs).  What is pinned:

  default OFF     resolve_phase_events: unset / "0" -> OFF; anything but "1"
                  is refused; "1" without PDMUX_TELEMETRY_PATH or with
                  PDMUX_LA_COORD is refused.
  OFF parity      with the flag OFF no phase event is written, and the
                  scheduler-visible trace (admissions, prefill chunks, every
                  other telemetry record) is IDENTICAL to the flag-ON run:
                  the instrument observes, it does not steer.
  ON content      one decode_iteration per decode step (== the loop's own
                  decode counter), start/end strictly alternate with matching
                  span_seq, prefill_in_flight matches the iterations where a
                  split-prefill chunk actually ran beside decode, decode_sms is
                  the partition the step ran on, admission state is recorded.
  isolation       an emit that raises disables the events (logged once) and
                  leaves the scheduling trace unchanged.
  installed       the installed runtime carries the flag (stale-tree check).

Negative controls (lesson 53): `test_mutants_are_killed` loads mutated copies
of the mixin (each of the five call-site hooks removed in turn where it can be
removed alone, the flag forced on by default, prefill_in_flight inverted) via
PDMUX_MIXIN_UNDER_TEST in a subprocess and requires this module to FAIL on
each.  An unmutated copy must PASS (control for false kills, lesson 254).
The init hook is not mutated alone: removing it leaves the attribute unset,
which every hook reads as OFF -- that IS the default-OFF path, covered by the
OFF-parity test, not a separate defect.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdmux_loop_fakes import (  # noqa: E402
    MIXIN_SOURCE,
    FakeReq,
    LoopScheduler,
    LoopTestBase,
    mixin_module,
    run_loop,
)

PHASE = ("prefill_span_start", "prefill_span_end", "decode_iteration")


def _arrivals():
    # staggered arrivals so that some prefills overlap a running decode batch
    arr = {}
    rid = 0
    for it in (0, 1, 3, 6, 7, 12, 20, 21, 22, 30):
        arr.setdefault(it, []).append(FakeReq(f"r{rid}", max_new=6 + (rid % 4)))
        rid += 1
    return arr


def _build(phase_on, admit=1, cap=48):
    sched = LoopScheduler(policy=None, policy_name="", arrivals=_arrivals(),
                          max_iters=60, admit_per_batch=admit,
                          max_running_requests=cap)
    sched.pdmux_phase_events = phase_on
    sched._phase_span_seq = 0
    sched._phase_events_error_logged = False
    sched._phase_decode_mark = None
    # record which iterations ran a decode step and/or a prefill chunk
    sched.steps = []
    real_run = sched.run_batch

    def run_batch(batch, _real=real_run, _s=sched):
        mode = "prefill" if batch.forward_mode == mixin_module.ForwardMode.SPLIT_PREFILL else "decode"
        _s.steps.append((_s.iteration, mode))
        return _real(batch)

    sched.run_batch = run_batch
    return sched


def _scheduling_trace(sched):
    return [e for e in sched.log
            if not (e[0] == "telemetry" and e[1] in PHASE)]


class ResolveTest(unittest.TestCase):
    def test_default_off_and_refusals(self):
        r = mixin_module.resolve_phase_events
        self.assertFalse(r("", {}))
        self.assertFalse(r("/x.jsonl", {}))
        self.assertFalse(r("/x.jsonl", {"PDMUX_PHASE_EVENTS": "0"}))
        self.assertTrue(r("/x.jsonl", {"PDMUX_PHASE_EVENTS": "1"}))
        with self.assertRaises(RuntimeError):
            r("", {"PDMUX_PHASE_EVENTS": "1"})
        with self.assertRaises(RuntimeError):
            r("/x.jsonl", {"PDMUX_PHASE_EVENTS": "1", "PDMUX_LA_COORD": "1"})
        for bad in ("true", "yes", "on", "2", "True"):
            with self.assertRaises(ValueError, msg=bad):
                r("/x.jsonl", {"PDMUX_PHASE_EVENTS": bad})

    def test_installed_runtime_has_the_flag(self):
        text = Path(MIXIN_SOURCE).read_text()
        self.assertIn("def resolve_phase_events", text)
        self.assertIn("self.dual_worker_trace_error_logged = self._init_phase_events(trace_path)", text)
        self.assertIn("self.pdmux_phase_events = resolve_phase_events(trace_path)", text)


class PhaseEventsLoopTest(LoopTestBase):
    def _run(self, phase_on, **kw):
        sched = self.install(_build(phase_on, **kw))
        run_loop(sched)
        return sched

    def test_off_writes_nothing_and_on_does_not_steer(self):
        off = self._run(False)
        on = self._run(True)
        self.assertEqual([e for e in off.log if e[0] == "telemetry" and e[1] in PHASE], [])
        self.assertGreater(len(on.events("decode_iteration")), 0)
        self.assertEqual(_scheduling_trace(off), _scheduling_trace(on))
        self.assertEqual(off.steps, on.steps)

    def test_one_event_per_decode_step(self):
        on = self._run(True)
        dec = on.events("decode_iteration")
        n_decode_steps = sum(1 for _, m in on.steps if m == "decode")
        self.assertEqual(len(dec), n_decode_steps)
        self.assertEqual(len(dec), on._r2_decode_iterations)
        self.assertEqual([d[2]["iter_seq"] for d in dec], list(range(1, len(dec) + 1)))
        for d in dec:
            f = d[2]
            self.assertLessEqual(f["t_launch"], f["t_sync"])
            self.assertEqual((f["prefill_sms"], f["decode_sms"]), tuple(on.sm_counts[f["stream_idx"]]))
            self.assertGreaterEqual(f["bs"], 1)

    def test_prefill_in_flight_matches_coresident_iterations(self):
        on = self._run(True)
        by_iter = {}
        for it, mode in on.steps:
            by_iter.setdefault(it, set()).add(mode)
        co_iters = sorted(it for it, modes in by_iter.items() if modes == {"prefill", "decode"})
        self.assertGreater(len(co_iters), 0, "fixture must produce co-resident iterations")
        dec = on.events("decode_iteration")
        flagged = sum(1 for d in dec if d[2]["prefill_in_flight"] and not d[2]["prefill_kernel_done_wait"])
        self.assertEqual(flagged, len(co_iters))

    def test_spans_alternate_and_carry_admission_state(self):
        on = self._run(True, admit=2, cap=48)
        seq = [(e[1], e[2]["span_seq"]) for e in on.log
               if e[0] == "telemetry" and e[1] in ("prefill_span_start", "prefill_span_end")]
        self.assertGreater(len(seq), 0)
        for i, (name, s) in enumerate(seq):
            self.assertEqual(name, "prefill_span_start" if i % 2 == 0 else "prefill_span_end")
            self.assertEqual(s, i // 2 + 1)
        starts = on.events("prefill_span_start")
        admitted = sum(1 for e in on.log if e[0] == "admit")
        self.assertEqual(sum(s[2]["n_new"] for s in starts), admitted)
        for s in starts:
            f = s[2]
            self.assertEqual(f["cap"], 48)
            self.assertLessEqual(f["running_bs"] + f["n_new"], 48)
            self.assertEqual((f["prefill_sms"], f["decode_sms"]), tuple(on.sm_counts[f["stream_idx"]]))
        ends = on.events("prefill_span_end")
        self.assertEqual([e[2]["n_done"] for e in ends], [s[2]["n_new"] for s in starts[:len(ends)]])

    def test_emit_failure_disables_without_steering(self):
        off = self._run(False)
        sched = self.install(_build(True))

        class Boom:
            def __init__(self, log):
                self.log = log

            def emit(self, event, phase, request_id="", **fields):
                if event in PHASE:
                    raise TypeError("boom")
                self.log.append(("telemetry", event, dict(fields)))
                return True

            def mark_phase(self, phase):
                return self.emit("phase_marker", phase)

            def close(self, timeout_s=5.0):
                return None

        sched.pdmux_telemetry = Boom(sched.log)
        run_loop(sched)
        self.assertFalse(sched.pdmux_phase_events)
        self.assertTrue(sched._phase_events_error_logged)
        self.assertEqual(_scheduling_trace(off), _scheduling_trace(sched))


# ------------------------------------------------------------------- mutants
_MUTANTS = {
    "sync_emitter_removed": (
        "decode_stream.synchronize(); self._phase_on_decode_sync(decode_done)",
        "decode_stream.synchronize()",
    ),
    "launch_hook_removed": (
        "decode_done = True; self._phase_on_decode_launch(stream_idx, wait_prefill_kernel_done)",
        "decode_done = True",
    ),
    "flag_forced_on": (
        '    raw = (env.get("PDMUX_PHASE_EVENTS", "") or "").strip()\n',
        '    raw = (env.get("PDMUX_PHASE_EVENTS", "1") or "1").strip()\n',
    ),
    "in_flight_inverted": (
        "                self.split_prefill_batch is not None,\n                bool(kernel_done_wait),",
        "                self.split_prefill_batch is None,\n                bool(kernel_done_wait),",
    ),
    "end_event_dropped": (
        "self._phase_on_prefill_end(batch); runtime = ",
        "runtime = ",
    ),
    "start_event_dropped": (
        "self._r2_reset_memory_peak(); self._phase_on_prefill_start(batch)",
        "self._r2_reset_memory_peak()",
    ),
}


def _run_self_against(path):
    env = dict(os.environ, PDMUX_MIXIN_UNDER_TEST=str(path), PHASE_EVENTS_NO_MUTANTS="1",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", "test_phase_events"],
        cwd=os.path.dirname(os.path.abspath(__file__)), env=env,
        capture_output=True, text=True, timeout=600,
    )
    return proc.returncode


@unittest.skipIf(os.environ.get("PHASE_EVENTS_NO_MUTANTS"), "inside a mutant run")
class MutantTest(unittest.TestCase):
    def test_mutants_are_killed(self):
        src = Path(MIXIN_SOURCE).read_text()
        with tempfile.TemporaryDirectory() as td:
            control = Path(td) / "control.py"
            control.write_text(src)
            self.assertEqual(_run_self_against(control), 0, "unmutated copy must pass")
            for name, (old, new) in _MUTANTS.items():
                self.assertEqual(src.count(old), 1, f"{name}: anchor not unique/absent")
                p = Path(td) / f"{name}.py"
                p.write_text(src.replace(old, new))
                self.assertNotEqual(_run_self_against(p), 0, f"mutant {name} SURVIVED")


if __name__ == "__main__":
    unittest.main()
