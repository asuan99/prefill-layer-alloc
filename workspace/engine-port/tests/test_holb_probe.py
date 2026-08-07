"""Gate for the direct head-of-line-blocking (HOLB) probe.

WHY THIS EXISTS.  Gate 2 changed its decision variable three times and two of
those died in audit, both times because a *threshold indicator* saturated at one
end for one arm (rev1: pdmux floor-censored; rev2: plain ceiling-censored).
`PREREG_GATE2_2026-08-06.md` section 9-8 pre-committed the escape: measure the
blocking duration directly in the engine instead of inventing a fourth
statistic.  These tests pin the definition so that the escape does not quietly
acquire the same defects.

What is pinned here:

  1. the classification table (`advances_decode`) -- in particular that MIXED
     counts as decode-advancing, which is the one genuinely ambiguous case and
     the one that differs structurally between the chunked and non-chunked arms;
  2. the boundary conditions of the accounting: idle, `pending == 0`, empty
     gaps, first/trailing spans, the strict-vs-ambiguous split;
  3. that a *different* stream never fills a gap (this is what makes the pdmux
     arm's prefill correctly count as concurrent rather than as blocking) while
     the *same* stream does (fused arms);
  4. that `blocking_fw_ms` -- the "recommended" definition -- really is
     identically zero on a pdmux-shaped trace, i.e. that the code reproduces the
     identity the design doc forbids using as a primary;
  5. that the window decomposition identity holds exactly, since that identity
     is the whole arm-symmetry argument;
  6. that the probe is OFF unless `PDMUX_HOLB_PATH` is set, and that when ON the
     CUDA plumbing (event pool, lazy drain, chain across a stream switch) is
     correct -- exercised with fake events so it runs on CPU.

The module is loaded from the installed runtime tree (same convention as
test_dual_worker.py / test_sticky_partition.py), so this also checks that
sync_engine_tree.sh actually installed it.
"""

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# test_profile_controller.py installs src/multiplex/profile.py as
# sys.modules["profile"], shadowing the stdlib module and breaking the sglang
# import chain below.  Same guard as test_sticky_partition.py.
sys.modules.pop("profile", None)

from sglang.srt.multiplex.holb_probe import (  # noqa: E402
    GAP_AMBIGUOUS,
    GAP_FIRST,
    GAP_INVALID,
    GAP_STRICT,
    HolbAccountant,
    HolbProbe,
    SpanRecord,
    _pending_decode_count,
    advances_decode,
    maybe_create_holb_probe,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STREAM_A = 111  # the single fused stream / the pdmux decode stream
STREAM_B = 222  # the pdmux prefill stream


def span(kind, dur, pending, stream=STREAM_A, gap=None, mode=None, seq=0, host_ts=0.0, fw_bs=None):
    if mode is None:
        mode = "DECODE" if kind == "decode" else "EXTEND"
    return SpanRecord(
        kind=kind,
        mode=mode,
        pending=pending,
        fw_bs=pending if fw_bs is None else fw_bs,
        stream_key=stream,
        dur_ms=dur,
        host_ts=host_ts,
        seq=seq,
        gap_ms=gap,
    )


def feed(acct, spans):
    return [acct.record(s) for s in spans]


class FakeMode:
    """Stands in for ForwardMode without importing the enum."""

    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_mixed(self):
        return self.name == "MIXED"

    def is_split_prefill(self):
        return self.name == "SPLIT_PREFILL"


class FakeEvent:
    """Deterministic stand-in for torch.cuda.Event(enable_timing=True).

    `record(stream)` stamps a virtual GPU clock owned by the stream; the clock
    is advanced explicitly by the test.  Reuse (the probe's event pool) is
    modelled faithfully: re-recording overwrites the stamp.
    """

    def __init__(self):
        self.t = None
        self.ready = True
        self.n_records = 0

    def record(self, stream):
        self.t = stream.clock
        self.n_records += 1

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True

    def elapsed_time(self, other):
        return other.t - self.t


class FakeStream:
    def __init__(self, key):
        self.cuda_stream = key
        self.clock = 0.0


class FakeTelemetry:
    def __init__(self):
        self.events = []
        self.closed = False

    def emit(self, event, phase, request_id="", **fields):
        self.events.append((event, phase, fields))
        return True

    def close(self, timeout_s=5.0):
        self.closed = True

    def of(self, name):
        return [f for e, _p, f in self.events if e == name]


# ---------------------------------------------------------------------------
# 1. classification table
# ---------------------------------------------------------------------------


class TestClassification(unittest.TestCase):
    def test_decode_and_mixed_advance_decode(self):
        self.assertTrue(advances_decode(FakeMode("DECODE")))
        # MIXED contains decode requests -> it ends a gap.  Pre-registered in
        # DIRECT_BLOCKING_DESIGN.md section 3.2 together with the bias it adds.
        self.assertTrue(advances_decode(FakeMode("MIXED")))

    def test_prefill_shapes_do_not_advance_decode(self):
        for name in ("EXTEND", "SPLIT_PREFILL", "IDLE", "TARGET_VERIFY", "PREBUILT"):
            with self.subTest(name=name):
                self.assertFalse(advances_decode(FakeMode(name)))


class TestPendingCount(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(_pending_decode_count(None), 0)
        self.assertEqual(_pending_decode_count(SimpleNamespace(reqs=[])), 0)

    def test_finished_requests_are_not_pending(self):
        reqs = [
            SimpleNamespace(finished=lambda: False),
            SimpleNamespace(finished=lambda: True),
            SimpleNamespace(finished=lambda: False),
        ]
        self.assertEqual(_pending_decode_count(SimpleNamespace(reqs=reqs)), 2)

    def test_prefill_only_batch_is_not_decode_pending(self):
        reqs = [SimpleNamespace(finished=lambda: False)]
        rb = SimpleNamespace(reqs=reqs, is_prefill_only=True)
        self.assertEqual(_pending_decode_count(rb), 0)


# ---------------------------------------------------------------------------
# 2. accounting boundary conditions
# ---------------------------------------------------------------------------


class TestAccountantBoundaries(unittest.TestCase):
    def test_first_decode_has_no_gap(self):
        acct = HolbAccountant()
        ev = acct.record(span("decode", 5.0, 3))
        self.assertEqual(ev["gap_class"], GAP_FIRST)
        self.assertIsNone(ev["gap_ms"])
        self.assertEqual(acct.c["stall_ms"], 0.0)
        self.assertEqual(acct.c["n_stall_episodes"], 0)

    def test_pre_first_decode_forwards_are_counted_separately(self):
        acct = HolbAccountant()
        feed(acct, [span("other", 9.0, 0), span("other", 9.0, 0)])
        acct.record(span("decode", 5.0, 2))
        self.assertEqual(acct.c["n_pre_first_decode_fw"], 2)
        self.assertEqual(acct.c["stall_ms"], 0.0)

    def test_trailing_forwards_are_counted_separately(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 2))
        acct.record(span("other", 7.0, 2))
        acct.finalize()
        self.assertEqual(acct.c["n_trailing_other_fw"], 1)

    def test_prefill_between_decodes_is_a_strict_stall(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 4))
        acct.record(span("other", 30.0, 4))
        ev = acct.record(span("decode", 5.0, 4, gap=32.0))
        self.assertEqual(ev["gap_class"], GAP_STRICT)
        self.assertAlmostEqual(acct.c["stall_ms"], 32.0)
        self.assertEqual(acct.c["n_stall_episodes"], 1)
        # same stream -> the prefill fills the gap; the rest is a bubble
        self.assertAlmostEqual(acct.c["same_stream_fw_ms"], 30.0)
        self.assertAlmostEqual(acct.c["other_stream_fw_ms"], 0.0)
        self.assertAlmostEqual(acct.c["bubble_ms"], 2.0)
        # request-weighted burden
        self.assertAlmostEqual(acct.c["stall_req_ms"], 32.0 * 4)

    def test_pending_zero_inside_gap_makes_it_ambiguous(self):
        """Queue drained, request arrived, prefill ran, decode resumed.

        The long wait was arrival time, not head-of-line blocking.  It must not
        land in the primary.
        """
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 1))
        acct.record(span("other", 40.0, 0))  # pending == 0 observed
        ev = acct.record(span("decode", 5.0, 1, gap=900.0))
        self.assertEqual(ev["gap_class"], GAP_AMBIGUOUS)
        self.assertEqual(acct.c["stall_ms"], 0.0)
        self.assertAlmostEqual(acct.c["stall_ambiguous_ms"], 900.0)
        self.assertEqual(acct.c["n_ambiguous_gaps"], 1)
        self.assertEqual(acct.c["n_stall_episodes"], 0)

    def test_empty_gap_with_no_intervening_forward_is_strict(self):
        """A pure GPU bubble while decode is pending is still decode not
        progressing.  This is the shape the pdmux arm produces (drain +
        per-iteration host overhead) and it is why the primary is non-trivial
        for that arm."""
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 6))
        ev = acct.record(span("decode", 5.0, 6, gap=3.5))
        self.assertEqual(ev["gap_class"], GAP_STRICT)
        self.assertEqual(ev["n_other_fw"], 0)
        self.assertAlmostEqual(acct.c["stall_ms"], 3.5)
        self.assertAlmostEqual(acct.c["bubble_ms"], 3.5)
        self.assertAlmostEqual(acct.c["same_stream_fw_ms"], 0.0)

    def test_negative_or_nan_gap_is_invalid_and_excluded(self):
        for bad in (-1.0, float("nan"), None):
            with self.subTest(bad=bad):
                acct = HolbAccountant()
                acct.record(span("decode", 5.0, 2))
                ev = acct.record(span("decode", 5.0, 2, gap=bad))
                self.assertEqual(ev["gap_class"], GAP_INVALID)
                self.assertEqual(acct.c["n_invalid_gaps"], 1)
                self.assertEqual(acct.c["stall_ms"], 0.0)

    def test_same_stream_longer_than_gap_is_flagged_inconsistent(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 2))
        acct.record(span("other", 50.0, 2))
        ev = acct.record(span("decode", 5.0, 2, gap=10.0))
        self.assertTrue(ev["inconsistent"])
        self.assertEqual(acct.c["n_inconsistent_gaps"], 1)
        # the gap still counts once, never more than the gap itself
        self.assertAlmostEqual(acct.c["stall_ms"], 10.0)
        self.assertAlmostEqual(acct.c["same_stream_fw_ms"], 10.0)
        self.assertAlmostEqual(acct.c["bubble_ms"], 0.0)

    def test_mixed_forward_ends_a_gap_and_is_counted_separately(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 2))
        acct.record(span("other", 20.0, 2))
        ev = acct.record(span("decode", 6.0, 2, gap=21.0, mode="MIXED"))
        self.assertEqual(ev["gap_class"], GAP_STRICT)
        self.assertEqual(acct.c["n_mixed"], 1)
        self.assertAlmostEqual(acct.c["mixed_fw_ms"], 6.0)
        self.assertAlmostEqual(acct.c["decode_fw_ms"], 11.0)

    def test_mode_counts_expose_unexpected_modes(self):
        acct = HolbAccountant()
        acct.record(span("other", 1.0, 0, mode="IDLE"))
        acct.record(span("decode", 1.0, 1))
        self.assertEqual(acct.summary()["mode_counts"], {"IDLE": 1, "DECODE": 1})


# ---------------------------------------------------------------------------
# 3. stream semantics -- the pdmux/fused distinction
# ---------------------------------------------------------------------------


class TestStreamSemantics(unittest.TestCase):
    def test_other_stream_forward_does_not_fill_the_gap(self):
        """pdmux shape: prefill runs concurrently on the prefill stream."""
        acct = HolbAccountant()
        acct.record(span("decode", 8.0, 5, stream=STREAM_A))
        acct.record(span("other", 40.0, 5, stream=STREAM_B, mode="SPLIT_PREFILL"))
        ev = acct.record(span("decode", 8.0, 5, stream=STREAM_A, gap=2.0))
        self.assertAlmostEqual(ev["same_stream_fw_ms"], 0.0)
        self.assertAlmostEqual(ev["other_stream_fw_ms"], 40.0)
        self.assertAlmostEqual(ev["bubble_ms"], 2.0)
        # the concurrent 40 ms prefill did NOT inflate the stall
        self.assertAlmostEqual(acct.c["stall_ms"], 2.0)

    def test_blocking_fw_ms_is_identically_zero_on_a_pdmux_shaped_trace(self):
        """The 'recommended' definition reproduced -- and shown to be an
        identity for pdmux, which is exactly why it must not be the primary
        (DIRECT_BLOCKING_DESIGN.md section 1.7)."""
        acct = HolbAccountant()
        acct.record(span("decode", 8.0, 5, stream=STREAM_A))
        for _ in range(20):
            acct.record(span("other", 40.0, 5, stream=STREAM_B, mode="SPLIT_PREFILL"))
            acct.record(span("decode", 8.0, 5, stream=STREAM_A, gap=2.0))
        # non-zero, because pending > 0 during those prefills ...
        self.assertGreater(acct.c["blocking_fw_ms"], 0.0)
        # ... but none of it reached the decode timeline
        self.assertAlmostEqual(acct.c["same_stream_fw_ms"], 0.0)
        self.assertAlmostEqual(acct.c["stall_ms"], 2.0 * 20)

    def test_fused_shape_puts_the_prefill_on_the_decode_timeline(self):
        acct = HolbAccountant()
        acct.record(span("decode", 8.0, 5))
        for _ in range(20):
            acct.record(span("other", 40.0, 5))
            acct.record(span("decode", 8.0, 5, gap=41.0))
        self.assertAlmostEqual(acct.c["same_stream_fw_ms"], 40.0 * 20)
        self.assertAlmostEqual(acct.c["stall_ms"], 41.0 * 20)
        self.assertAlmostEqual(acct.c["blocking_fw_ms"], 40.0 * 20)


# ---------------------------------------------------------------------------
# 4. window identity + episode distribution
# ---------------------------------------------------------------------------


class TestWindowIdentity(unittest.TestCase):
    def test_decomposition_is_exact(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 3))
        acct.record(span("other", 20.0, 3))
        acct.record(span("decode", 5.0, 3, gap=21.0))
        acct.record(span("other", 20.0, 0))
        acct.record(span("decode", 5.0, 1, gap=500.0))
        s = acct.summary()
        self.assertAlmostEqual(
            s["timeline_ms"],
            s["decode_fw_ms"] + s["stall_ms"] + s["stall_ambiguous_ms"],
        )
        self.assertAlmostEqual(s["timeline_ms"], 15.0 + 21.0 + 500.0)

    def test_timeline_over_host_selfcheck(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 3, host_ts=100.000))
        acct.record(span("decode", 5.0, 3, gap=20.0, host_ts=100.030))
        s = acct.summary()
        self.assertAlmostEqual(s["host_span_ms"], 30.0)
        self.assertAlmostEqual(s["timeline_ms"], 30.0)
        self.assertAlmostEqual(s["timeline_over_host"], 1.0)

    def test_episode_percentiles(self):
        acct = HolbAccountant()
        acct.record(span("decode", 1.0, 2))
        for g in (10.0, 20.0, 30.0, 40.0, 100.0):
            acct.record(span("other", g - 1.0, 2))
            acct.record(span("decode", 1.0, 2, gap=g))
        s = acct.summary()
        self.assertEqual(s["n_stall_episodes"], 5)
        self.assertAlmostEqual(s["episode_p50_ms"], 30.0)
        self.assertAlmostEqual(s["episode_max_ms"], 100.0)
        self.assertAlmostEqual(s["stall_frac"], 200.0 / 206.0)

    def test_no_episodes_yields_none_not_zero(self):
        acct = HolbAccountant()
        acct.record(span("decode", 5.0, 3))
        s = acct.summary()
        self.assertIsNone(s["episode_p50_ms"])
        self.assertIsNone(s["episode_max_ms"])

    def test_episode_cap_is_reported(self):
        acct = HolbAccountant(episode_cap=3)
        acct.record(span("decode", 1.0, 1))
        for _ in range(5):
            acct.record(span("decode", 1.0, 1, gap=7.0))
        self.assertEqual(len(acct.episodes), 3)
        self.assertEqual(acct.episodes_truncated, 2)
        self.assertEqual(acct.c["n_stall_episodes"], 5)  # counter is not capped
        self.assertAlmostEqual(acct.c["stall_ms"], 35.0)  # nor is the sum


# ---------------------------------------------------------------------------
# 5. probe plumbing (fake CUDA events, runs on CPU)
# ---------------------------------------------------------------------------


class TestProbePlumbing(unittest.TestCase):
    def _probe(self, **kw):
        tel = FakeTelemetry()
        probe = HolbProbe(
            None,
            event_factory=FakeEvent,
            telemetry=tel,
            summary_every_s=1e9,  # no periodic summaries during the test
            **kw,
        )
        return probe, tel

    def _run(self, probe, stream, mode_name, pending, dur, fw_bs=None):
        """Issue one forward on `stream` taking `dur` ms of virtual GPU time."""
        batch = SimpleNamespace(
            forward_mode=FakeMode(mode_name),
            reqs=[object()] * (pending if fw_bs is None else fw_bs),
        )
        rb = SimpleNamespace(
            reqs=[SimpleNamespace(finished=lambda: False)] * pending,
            is_prefill_only=False,
        )
        import sglang.srt.multiplex.holb_probe as hp

        orig = hp.torch
        hp.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: stream))
        try:
            s = probe.begin(batch, rb)
            stream.clock += dur
            probe.end(s)
        finally:
            hp.torch = orig

    def test_off_by_default(self):
        os.environ.pop("PDMUX_HOLB_PATH", None)
        self.assertIsNone(maybe_create_holb_probe(SimpleNamespace()))

    def test_off_when_path_is_blank(self):
        os.environ["PDMUX_HOLB_PATH"] = "   "
        try:
            self.assertIsNone(maybe_create_holb_probe(SimpleNamespace()))
        finally:
            os.environ.pop("PDMUX_HOLB_PATH", None)

    def test_true_dual_worker_combination_fails_fast(self):
        """Two host issue threads would break the single-FIFO ordering the
        accountant relies on.  Refuse rather than emit a wrong number."""
        os.environ["PDMUX_HOLB_PATH"] = "/dev/null"
        os.environ["PDMUX_TRUE_DUAL_WORKER"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                maybe_create_holb_probe(SimpleNamespace())
        finally:
            os.environ.pop("PDMUX_HOLB_PATH", None)
            os.environ.pop("PDMUX_TRUE_DUAL_WORKER", None)

    def test_unresolved_spans_are_reported_at_close_not_waited_on(self):
        probe, tel = self._probe()
        a = FakeStream(STREAM_A)
        self._run(probe, a, "DECODE", 2, 5.0)

        def factory():
            e = FakeEvent()
            e.ready = False
            return e

        probe._event_factory = FakeEvent  # keep pool clean
        probe._pool = []
        probe._event_factory = factory
        a.clock += 4.0
        self._run(probe, a, "DECODE", 2, 5.0)  # never completes
        probe.close()
        s = tel.of("holb_summary")[-1]
        self.assertEqual(s["n_unresolved_at_close"], 1)
        # the resolved part is still reported; nothing hangs
        self.assertEqual(s["n_decode_fw"], 1)

    def test_end_to_end_fused_shape(self):
        probe, tel = self._probe()
        a = FakeStream(STREAM_A)
        self._run(probe, a, "DECODE", 4, 5.0)
        a.clock += 1.0  # host bubble
        self._run(probe, a, "EXTEND", 4, 30.0)
        a.clock += 1.0
        self._run(probe, a, "DECODE", 4, 5.0)
        gaps = tel.of("holb_gap")
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["gap_class"], GAP_FIRST)
        self.assertEqual(gaps[1]["gap_class"], GAP_STRICT)
        self.assertAlmostEqual(gaps[1]["gap_ms"], 32.0)
        self.assertAlmostEqual(gaps[1]["same_stream_fw_ms"], 30.0)
        self.assertAlmostEqual(gaps[1]["bubble_ms"], 2.0)

    def test_end_to_end_pdmux_shape_two_streams(self):
        probe, tel = self._probe()
        a, b = FakeStream(STREAM_A), FakeStream(STREAM_B)
        self._run(probe, a, "DECODE", 6, 8.0)
        # concurrent prefill on the other stream; the decode stream only idles
        # for the host-side overhead
        self._run(probe, b, "SPLIT_PREFILL", 6, 40.0)
        a.clock += 2.0
        self._run(probe, a, "DECODE", 6, 8.0)
        g = tel.of("holb_gap")[-1]
        self.assertAlmostEqual(g["gap_ms"], 2.0)
        self.assertAlmostEqual(g["same_stream_fw_ms"], 0.0)
        self.assertAlmostEqual(g["other_stream_fw_ms"], 40.0)

    def test_chain_survives_a_decode_stream_switch(self):
        """pdmux changes stream objects when the partition index changes; the
        decode-to-decode chain is cross-stream and must still resolve."""
        probe, tel = self._probe()
        a, b = FakeStream(STREAM_A), FakeStream(STREAM_B)
        b.clock = 0.0
        self._run(probe, a, "DECODE", 3, 5.0)
        b.clock = a.clock + 4.0  # common GPU clock, new stream object
        self._run(probe, b, "DECODE", 3, 5.0)
        g = tel.of("holb_gap")[-1]
        self.assertEqual(g["gap_class"], GAP_STRICT)
        self.assertAlmostEqual(g["gap_ms"], 4.0)

    def test_lazy_drain_never_synchronises(self):
        """A forward whose end event has not completed must not be waited on;
        it stays in the backlog and resolves later."""
        probe, tel = self._probe()
        a = FakeStream(STREAM_A)
        created = []

        def factory():
            e = FakeEvent()
            e.ready = False  # GPU has not reached it yet
            created.append(e)
            return e

        probe._event_factory = factory
        self._run(probe, a, "DECODE", 2, 5.0)
        a.clock += 3.0
        self._run(probe, a, "DECODE", 2, 5.0)
        self.assertEqual(len(tel.of("holb_gap")), 0)  # nothing resolved, nothing blocked
        self.assertEqual(len(probe._inflight), 2)
        for e in created:
            e.ready = True
        probe._event_factory = FakeEvent
        a.clock += 3.0
        self._run(probe, a, "DECODE", 2, 5.0)  # this end() drains the backlog
        self.assertEqual(len(tel.of("holb_gap")), 3)
        self.assertEqual(len(probe._inflight), 0)
        self.assertAlmostEqual(probe.acct.c["stall_ms"], 6.0)

    def test_event_pool_recycles(self):
        probe, _ = self._probe()
        a = FakeStream(STREAM_A)
        for _ in range(50):
            self._run(probe, a, "DECODE", 2, 5.0)
        # 2 events per forward would be 100 without recycling; the pool plus the
        # single held decode-end event bound it to a small constant.
        self.assertLessEqual(len(probe._pool) + 1, 4)

    def test_max_inflight_drops_instead_of_blocking(self):
        probe, _ = self._probe(max_inflight=1)
        a = FakeStream(STREAM_A)
        batch = SimpleNamespace(forward_mode=FakeMode("DECODE"), reqs=[object()])
        rb = SimpleNamespace(reqs=[SimpleNamespace(finished=lambda: False)], is_prefill_only=False)
        import sglang.srt.multiplex.holb_probe as hp

        orig = hp.torch
        hp.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: a))
        try:
            s1 = probe.begin(batch, rb)
            e = FakeEvent()
            e.ready = False
            probe._inflight.append(
                SimpleNamespace(kind="decode", end_event=e)
            )  # occupy the queue
            s2 = probe.begin(batch, rb)
        finally:
            hp.torch = orig
        self.assertIsNotNone(s1)
        self.assertIsNone(s2)
        self.assertEqual(probe.dropped_spans, 1)

    def test_begin_errors_are_swallowed_and_counted(self):
        probe, _ = self._probe()
        bad = SimpleNamespace()  # no forward_mode
        self.assertIsNone(probe.begin(bad, None))
        self.assertEqual(probe.n_errors, 1)

    def test_summary_record_on_close(self):
        probe, tel = self._probe()
        a = FakeStream(STREAM_A)
        self._run(probe, a, "DECODE", 2, 5.0)
        a.clock += 10.0
        self._run(probe, a, "DECODE", 2, 5.0)
        probe.close()
        summaries = tel.of("holb_summary")
        self.assertTrue(summaries)
        s = summaries[-1]
        self.assertAlmostEqual(s["stall_ms"], 10.0)
        self.assertAlmostEqual(s["decode_fw_ms"], 10.0)
        self.assertEqual(s["dropped_spans"], 0)
        self.assertEqual(s["n_errors"], 0)
        self.assertTrue(tel.closed)
        self.assertTrue(math.isfinite(s["timeline_ms"]))


# ---------------------------------------------------------------------------
# 6. installed-tree hooks
# ---------------------------------------------------------------------------


class TestSchedulerHooksInstalled(unittest.TestCase):
    """The probe is worthless if it only exists on the pdmux arm.  Pin that the
    hooks are present in scheduler.py (all event loops) and that the OFF path is
    a bare `is not None` check."""

    def _scheduler_source(self):
        import sglang.srt.managers.scheduler as sched

        with open(sched.__file__, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_hooks_present_in_all_three_dispatch_branches(self):
        src = self._scheduler_source()
        self.assertIn("from sglang.srt.multiplex.holb_probe import maybe_create_holb_probe", src)
        self.assertIn("self.holb_probe = maybe_create_holb_probe(self)", src)
        # one begin/end pair per forward-dispatch branch of run_batch
        self.assertEqual(src.count("self.holb_probe.begin(batch, self.running_batch)"), 3)
        self.assertEqual(src.count("self.holb_probe.end(_holb)"), 3)

    def test_off_path_is_a_none_check(self):
        src = self._scheduler_source()
        self.assertEqual(src.count("if self.holb_probe is not None"), 3)
        self.assertEqual(src.count("if _holb is not None:"), 3)

    def test_class_level_default_is_none(self):
        from sglang.srt.managers.scheduler import Scheduler

        self.assertIsNone(Scheduler.holb_probe)

    def test_patch_reapplies_the_hooks_and_nothing_else(self):
        """The mirrored patch must reproduce the installed tree exactly, so the
        dev-tree edit is reversible and reproducible from src/."""
        import subprocess
        import tempfile

        import sglang.srt.managers.scheduler as sched

        track_root = Path(__file__).resolve().parents[1]
        patch_file = track_root / "src" / "patches" / "holb_probe_scheduler_hooks.patch"
        self.assertTrue(patch_file.is_file(), patch_file)
        installed = Path(sched.__file__).read_text()
        # reverse the patch in a scratch tree, then re-apply it
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sglang" / "srt" / "managers"
            target.mkdir(parents=True)
            (target / "scheduler.py").write_text(installed)
            for direction in ("--reverse", "--forward"):
                r = subprocess.run(
                    ["patch", direction, "--batch", "-p1", "-d", tmp],
                    stdin=patch_file.open("rb"),
                    capture_output=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr.decode())
            self.assertEqual((target / "scheduler.py").read_text(), installed)


class TestMirrorSync(unittest.TestCase):
    def test_tracked_mirror_matches_installed_runtime(self):
        import sglang.srt.multiplex.holb_probe as hp

        mirror = (
            Path(__file__).resolve().parents[1] / "src" / "multiplex" / "holb_probe.py"
        )
        runtime = Path(hp.__file__)
        if not runtime.is_file():
            self.skipTest(f"runtime tree absent: {runtime}")
        self.assertEqual(
            mirror.read_text(),
            runtime.read_text(),
            "src/multiplex/holb_probe.py and the installed runtime diverged; "
            "re-run scripts/bootstrap/sync_engine_tree.sh",
        )


if __name__ == "__main__":
    unittest.main()
