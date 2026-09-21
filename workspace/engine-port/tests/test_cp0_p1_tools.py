"""Gate for the CP-0 P1 offline tools: the roll-up and the acceptance checker.

WHY THESE ARE TESTED SEPARATELY FROM THE PROBE.  The probe writes JSONL and the
tools read it; a schema drift between them would show up on the GPU as an empty
result and be misread as "the engine did not chunk".  So the fixtures here are
**produced by the real ``ChunkProbe``**, not hand-written to match the reader --
if the writer and the reader disagree about a field name, these tests fail on
CPU instead of a GPU job failing silently.

What is pinned:

  1. the roll-up bins by ``host_ts`` into the harness's windows, and does NOT
     assume the per-segment ``n_requests_chunked`` sum to the boot total (they
     are distinct-rid counts, and a request can be chunked in two windows) --
     but it does check the relation in BOTH directions (repair 3, 2026-09-01):
     overlapping windows, an additive counter summing to more than the boot
     total, and an rid covered by no window are all reported, because job
     899768's segments summed to 8 rids against a boot total of 5 and the
     record still said ``status: OK``;
  2. the roll-up's integrity checks actually fire -- dropped telemetry events
     and a disagreement between the event stream and the accountant both flip
     ``status`` to ``EXTRACTION_UNRELIABLE``;
  3. the acceptance checker's four-way failure classification
     (PREREG_CP0_2026-08-28.md section 2.3), in particular that a prefill
     forward exceeding the budget is labelled ``ENGINE_BUDGET_EXCEEDED`` -- a
     finding about the engine -- and NOT folded into "instrument failure"
     (PROJECT_STATUS.md "방법론 게이트" #21, bidirectional);
  4. that P1-d and P1-e actually discriminate: a checker fed a run in which the
     three candidate definitions coincide must FAIL, and a P1-d burst that
     contained a prompt longer than cps must FAIL rather than pass on the
     strength of a length-driven count;
  5. that the verdict is a PRODUCT of two axes (repair 2, 2026-09-01): a
     measurement failure may not swallow a ``FAIL``, which is what happened to
     P1-e in job 899768, and the reverted one-ladder rule is written out as a
     mutation so the check cannot be vacuous;
  6. the harness's own kill sequence, run as bash against a launcher/scheduler
     mock: the process that owns the probe must get a SIGTERM before anything
     gets a SIGKILL (repair 1).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
CP = TRACK_ROOT / "results" / "cp_baseline"
EXPECT_PATH = CP / "p1e_expectations.json"

RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(TRACK_ROOT.parents[2])),
                     "sglang_engine_dev", "python")
)
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def _extract_shell_function(src, name):
    """Pull one `name () { ... }` block out of the sbatch, verbatim.

    The harness tests below run the harness's OWN code rather than a copy of
    it, so a drift between the sbatch and the test cannot hide (the same
    reason test_chunk_probe.py builds its fixtures with the real writer).
    """
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"{name} ()"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN = _load("cp0_analyze_chunk_probe", CP / "analyze_chunk_probe.py")
CK = _load("cp0_check_p1_acceptance", CP / "check_p1_acceptance.py")
WL = _load("cp0_p1_accept_workload", CP / "p1_accept_workload.py")


def write_probe_jsonl(path, arm="cps512", cfg=None, chunks=(), forwards=()):
    """Produce a fixture with the REAL writer, so the schema cannot drift."""
    from sglang.srt.multiplex.chunk_probe import ChunkProbe

    base = {
        "requested_chunked_prefill_size": 512,
        "realized_chunked_prefill_size": 512,
        "page_size": 1,
        "disable_piecewise_cuda_graph": False,
        "piecewise_cuda_graph_max_tokens": 512,
        "piecewise_cuda_graph_tokens_len": 40,
        "piecewise_cuda_graph_tokens_min": 4,
        "piecewise_cuda_graph_tokens_max": 512,
    }
    base.update(cfg or {})
    probe = ChunkProbe(path, run_id="rid", arm=arm, config=base, summary_every_s=1e9)
    import sglang.srt.multiplex.chunk_probe as cpmod

    real_time = cpmod.time.time
    for ts, rid, mech, before, after in chunks:
        cpmod.time.time = lambda ts=ts: ts
        probe.on_chunk(rid, mech, before, after)
    for ts, mode, ext, bs in forwards:
        cpmod.time.time = lambda ts=ts: ts
        probe.on_forward(mode, ext, bs, False)
    cpmod.time.time = real_time
    probe.close()


def boot_record(tmp, **kw):
    """A fixture boot record shaped like analyze_chunk_probe.py's output."""
    tot = {
        "n_requests_chunked": kw.pop("n_requests_chunked", 2),
        "n_chunk_events": kw.pop("n_chunk_events", 3),
        "n_chunk_events_admission": kw.pop("n_chunk_events_admission", 2),
        "n_chunk_events_continuation": kw.pop("n_chunk_events_continuation", 1),
        "n_dllm_trunc_events": 0,
        "mech_counts": {},
        "extend_tokens_per_forward": {
            "max": kw.pop("max_extend", 512), "min": 1,
            "mean": 375.0, "sum": 2625,
            "n_forwards": kw.pop("n_forwards", 7),
        },
    }
    rec = {
        "schema": AN.SCHEMA,
        "status": kw.pop("status", "OK"),
        "problems": kw.pop("problems", []),
        "run_id": "rid",
        "arm": kw.pop("arm", "cps512"),
        "requested_chunked_prefill_size": kw.pop("requested_cps", 512),
        "realized_chunked_prefill_size": kw.pop("realized_cps", 512),
        "disable_piecewise_cuda_graph": kw.pop("disable_piecewise", False),
        "piecewise_cuda_graph_max_tokens": 512,
        "piecewise_cuda_graph_tokens": kw.pop(
            "pw_tokens", {"len": 40, "min": 4, "max": 512}
        ),
        "segments": kw.pop("segments", []),
        "boot_total_from_events": tot,
        "boot_total_from_accountant": tot,
        "health": {"dropped_events": 0},
    }
    rec.update(kw)
    p = Path(tmp) / f"boot_{rec['arm']}.json"
    p.write_text(json.dumps(rec))
    return str(p), rec


# ---------------------------------------------------------------------------
# 1. the roll-up
# ---------------------------------------------------------------------------


class TestRollup(unittest.TestCase):
    def test_segments_bin_by_host_ts(self):
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[
                    (100.0, "a", "admission_ignore_eos", 1500, 512),
                    (100.5, "a", "continuation", 988, 512),
                    (200.0, "b", "admission_ignore_eos", 900, 512),
                ],
                forwards=[
                    (100.0, "EXTEND", 512, 1), (100.5, "EXTEND", 512, 1),
                    (101.0, "EXTEND", 476, 1), (150.0, "DECODE", 0, 4),
                    (200.0, "EXTEND", 512, 1),
                ],
            )
            rows, bad = AN.read_jsonl(jsonl)
            segdoc = {"segments": [
                {"segment_id": "rate3", "rate": 3, "t_start": 99.0, "t_end": 120.0},
                {"segment_id": "rate12", "rate": 12, "t_start": 199.0, "t_end": 220.0},
            ]}
            rec = AN.build_record(rows, segdoc, bad_lines=bad)
        self.assertEqual(rec["status"], "OK", rec["problems"])
        s0, s1 = rec["segments"]
        self.assertEqual(s0["rate"], 3)
        self.assertEqual(s0["n_requests_chunked"], 1)
        self.assertEqual(s0["n_chunk_events"], 2)
        self.assertEqual(s0["n_chunk_events_continuation"], 1)
        self.assertEqual(s0["extend_tokens_per_forward"]["n_forwards"], 3)
        self.assertEqual(s0["extend_tokens_per_forward"]["max"], 512)
        self.assertAlmostEqual(s0["extend_tokens_per_forward"]["mean"], 1500 / 3)
        self.assertEqual(s1["n_requests_chunked"], 1)
        self.assertEqual(s1["n_chunk_events"], 1)
        self.assertEqual(rec["boot_total_from_events"]["n_requests_chunked"], 2)
        self.assertEqual(rec["boot_total_from_events"]["n_chunk_events"], 3)
        # decode forwards never enter channel 2
        self.assertEqual(
            rec["boot_total_from_events"]["extend_tokens_per_forward"]["n_forwards"], 4
        )

    def test_distinct_rids_are_not_forced_to_be_additive(self):
        """The same request chunked in two windows counts once per window and
        once for the boot.  The tool must not 'fix' that into a sum."""
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512),
                        (200.0, "a", "continuation", 988, 512)],
                forwards=[(100.0, "EXTEND", 512, 1), (200.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {"segments": [
                {"segment_id": "s1", "t_start": 99.0, "t_end": 120.0},
                {"segment_id": "s2", "t_start": 199.0, "t_end": 220.0},
            ]}, bad_lines=bad)
        self.assertEqual(rec["status"], "OK", rec["problems"])
        self.assertEqual(sum(s["n_requests_chunked"] for s in rec["segments"]), 2)
        self.assertEqual(rec["boot_total_from_events"]["n_requests_chunked"], 1)

    # -- segment <-> boot consistency, BOTH directions (P1 repair 3) --------
    #
    # Until 2026-09-01 the roll-up only complained when the segments summed to
    # LESS than the boot.  Job 899768's segments summed to 8 distinct rids
    # against a boot total of 5, and to 9 chunk events against 6, and the record
    # said `status: OK` -- because `p1_accept_workload.py` registers
    # `t_end + 1.0`, so the phase-S window overlapped the start of phase B.

    def _overlapping_fixture(self, d):
        jsonl = Path(d) / "chunk.jsonl"
        write_probe_jsonl(
            jsonl,
            chunks=[
                (100.0, "a", "admission_ignore_eos", 1500, 512),
                (110.5, "b", "admission_ignore_eos", 900, 512),   # in the overlap
            ],
            forwards=[(100.0, "EXTEND", 512, 1), (110.5, "EXTEND", 400, 1)],
        )
        rows, bad = AN.read_jsonl(jsonl)
        segdoc = {"segments": [
            # exactly the shape p1_accept_workload.py produces: the first
            # window's end has +1.0 of slack and lands inside the second
            {"segment_id": "p1e_S", "t_start": 99.0, "t_end": 111.0},
            {"segment_id": "p1e_B", "t_start": 110.0, "t_end": 120.0},
        ]}
        return AN.build_record(rows, segdoc, bad_lines=bad)

    def test_overlapping_windows_are_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._overlapping_fixture(d)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE", rec["problems"])
        self.assertTrue(
            any("overlap" in p for p in rec["problems"]),
            f"the overlap itself was not reported: {rec['problems']}",
        )
        self.assertEqual(
            rec["segment_consistency"]["windows_overlap"][0]["seconds"], 1.0
        )

    def test_segments_summing_to_MORE_than_the_boot_is_a_problem(self):
        """★The direction job 899768 walked through in silence."""
        with tempfile.TemporaryDirectory() as d:
            rec = self._overlapping_fixture(d)
        con = rec["segment_consistency"]
        self.assertGreater(
            con["additive_sums"]["n_chunk_events"],
            con["additive_boot"]["n_chunk_events"],
            "the fixture no longer double counts, so it tests nothing",
        )
        self.assertTrue(
            any("cannot exceed the whole it partitions" in p
                for p in rec["problems"]),
            f"an additive counter exceeded the boot total silently: "
            f"{rec['problems']}",
        )

    def test_the_shortfall_direction_still_fires(self):
        """The pre-existing check must survive its own generalisation."""
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512),
                        (500.0, "z", "admission_ignore_eos", 900, 512)],
                forwards=[(100.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {"segments": [
                {"segment_id": "s1", "t_start": 99.0, "t_end": 120.0},
            ]}, bad_lines=bad)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE")
        self.assertTrue(
            any("no registered window" in p for p in rec["problems"]),
            f"the uncovered rid was not reported: {rec['problems']}",
        )

    def test_one_rid_in_two_disjoint_windows_is_a_counted_warning(self):
        """The legitimate excess the old one-sided check was excusing.  It must
        stay legal, and it must stop being invisible."""
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512),
                        (200.0, "a", "continuation", 988, 512)],
                forwards=[(100.0, "EXTEND", 512, 1), (200.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {"segments": [
                {"segment_id": "s1", "t_start": 99.0, "t_end": 120.0},
                {"segment_id": "s2", "t_start": 199.0, "t_end": 220.0},
            ]}, bad_lines=bad)
        self.assertEqual(rec["status"], "OK", rec["problems"])
        con = rec["segment_consistency"]
        self.assertEqual(con["sum_of_per_segment_distinct_rids"], 2)
        self.assertEqual(con["n_distinct_rids_boot"], 1)
        self.assertEqual(con["n_rid_segment_pairs_beyond_the_first"], 1)
        self.assertTrue(
            any("more than one window" in w for w in rec["warnings"]),
            f"the legitimate excess was silent: {rec['warnings']}",
        )

    def test_a_non_prefill_forward_is_excluded_from_channel_2_and_warned_about(self):
        """Repair 1 (2026-09-01).  On job 899768's cps512 boot 101 of the 116
        `prefill_forward` lines had forward_mode DECODE, each carrying the
        STALE `extend_num_tokens` of the preceding prefill, so
        `extend_tokens_per_forward` was not "prefill tokens per prefill
        forward" and could not be compared with the pre-registered P1-e numbers
        at all.

        The reader now classifies by `forward_mode` and counts prefill forwards
        only.  This is not "silently redefining a registered counter", which is
        what the pre-repair comment here worried about: the excluded records
        stay in the raw JSONL, their count and their token sum are reported,
        and the exclusion is warned about.  What was silently redefined was the
        OTHER direction -- decode steps were being reported as prefill.
        """
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                forwards=[(100.0, "EXTEND", 512, 1), (101.0, "DECODE", 476, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertTrue(
            any("not a prefill" in w for w in rec["warnings"]),
            f"channel-2 contamination was silent: {rec['warnings']}",
        )
        tot = rec["boot_total_from_events"]
        self.assertEqual(tot["extend_tokens_per_forward"]["sum"], 512,
                         "the decode forward's stale count must not be summed")
        self.assertEqual(tot["extend_tokens_per_forward"]["n_forwards"], 1)
        self.assertEqual(tot["n_nonprefill_forwards_with_extend_tokens"], 1)
        self.assertEqual(tot["sum_extend_tokens_not_counted"], 476,
                         "excluded, but not lost")
        self.assertEqual(tot["forward_mode_counts"], {"EXTEND": 1, "DECODE": 1},
                         "the raw mode counts stay visible")
        self.assertEqual(rec["health"]["n_nonprefill_forward_lines"], 1)

    def test_an_old_artifact_labels_decode_forwards_prefill_and_is_reclassified(self):
        """The 899768 file is in the pre-repair schema and must still be readable.

        Written by hand ON PURPOSE -- the exception to this file's "fixtures come
        from the real writer" rule -- because the repaired writer can no longer
        produce it: before 2026-09-01 a DECODE forward with a stale extend count
        was emitted as `event: prefill_forward`.  The reader must classify on
        `forward_mode`, not on the event name, or the re-read of job 899768
        would inherit the defect it is meant to correct.
        """
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "old.jsonl"
            lines = [
                {"event": "chunk_open", "schema_chunk": "pdmux.chunk-probe/v1",
                 "arm": "cps512", "run_id": "rid", "host_ts": 99.0,
                 "realized_chunked_prefill_size": 512},
                {"event": "prefill_forward", "host_ts": 100.0,
                 "forward_mode": "EXTEND", "extend_num_tokens": 512,
                 "batch_size": 1},
                {"event": "prefill_forward", "host_ts": 101.0,
                 "forward_mode": "DECODE", "extend_num_tokens": 512,
                 "batch_size": 1},
                {"event": "chunk_summary", "phase": "shutdown", "host_ts": 102.0,
                 "schema_chunk": "pdmux.chunk-probe/v1",
                 "n_requests_chunked": 0, "n_chunk_events": 0,
                 "n_chunk_events_admission": 0, "n_chunk_events_continuation": 0,
                 "extend_tokens_per_forward": {"max": 512, "min": 512,
                                               "mean": 512.0, "sum": 1024,
                                               "n_forwards": 2},
                 "n_forwards_total": 2, "n_errors": 0},
            ]
            jsonl.write_text("".join(json.dumps(r) + "\n" for r in lines))
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {}, bad_lines=bad)
        tot = rec["boot_total_from_events"]
        self.assertEqual(tot["extend_tokens_per_forward"]["n_forwards"], 1,
                         "the DECODE line was labelled prefill_forward by the "
                         "old writer; the reader must not believe the label")
        self.assertEqual(tot["extend_tokens_per_forward"]["sum"], 512)
        self.assertEqual(rec["writer_schema"], "pdmux.chunk-probe/v1")

    def test_a_pre_repair_accountant_disagreement_is_a_warning_not_a_defect(self):
        """...but only downward, and only for the pre-repair schema.

        The archived accountant totals in a v1 artifact were computed with the
        old rule, so they legitimately EXCEED the repaired event-stream count;
        calling that an instrument defect would be the mislabelling gate #21
        forbids.  Every other case stays a problem, or this downgrade would be
        a hole big enough to hide a real disagreement in.
        """
        def build(schema, acct_forwards, acct_sum):
            with tempfile.TemporaryDirectory() as d:
                jsonl = Path(d) / "x.jsonl"
                lines = [
                    {"event": "chunk_open", "schema_chunk": schema, "arm": "a",
                     "host_ts": 99.0},
                    {"event": "prefill_forward", "host_ts": 100.0,
                     "forward_mode": "EXTEND", "extend_num_tokens": 512,
                     "batch_size": 1},
                    {"event": "chunk_summary", "phase": "shutdown",
                     "host_ts": 102.0, "schema_chunk": schema,
                     "n_requests_chunked": 0, "n_chunk_events": 0,
                     "n_chunk_events_admission": 0,
                     "n_chunk_events_continuation": 0,
                     "extend_tokens_per_forward": {
                         "max": 512, "min": 512, "mean": 512.0,
                         "sum": acct_sum, "n_forwards": acct_forwards},
                     "n_forwards_total": acct_forwards, "n_errors": 0},
                ]
                jsonl.write_text("".join(json.dumps(r) + "\n" for r in lines))
                rows, bad = AN.read_jsonl(jsonl)
                return AN.build_record(rows, {}, bad_lines=bad)

        old_over = build("pdmux.chunk-probe/v1", 9, 4096)
        self.assertEqual(old_over["status"], "OK", old_over["problems"])
        self.assertTrue(any("pre-repair" in w for w in old_over["warnings"]),
                        old_over["warnings"])
        old_under = build("pdmux.chunk-probe/v1", 0, 0)
        self.assertEqual(old_under["status"], "EXTRACTION_UNRELIABLE",
                         "the old rule could only OVER-count; an accountant "
                         "below the event stream is a real disagreement")
        new_over = build(AN.WRITER_SCHEMA_CURRENT, 9, 4096)
        self.assertEqual(new_over["status"], "EXTRACTION_UNRELIABLE",
                         "a post-repair artifact gets no amnesty")

    def test_unknown_forward_modes_are_counted_and_warned_about(self):
        """A mode the reader cannot classify is a hole, and must be visible.

        Dropping it would erase the answer; counting it into channel 2 would
        reproduce the stale-field defect for any future mode.
        """
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "x.jsonl"
            lines = [
                {"event": "chunk_open", "schema_chunk": AN.WRITER_SCHEMA_CURRENT,
                 "arm": "a", "host_ts": 99.0},
                {"event": "prefill_forward", "host_ts": 100.0,
                 "forward_mode": "EXTEND", "extend_num_tokens": 512},
                {"event": "nonprefill_forward", "host_ts": 100.5,
                 "forward_mode": "A_MODE_FROM_THE_FUTURE",
                 "extend_num_tokens": 7, "is_prefill": False},
                {"event": "prefill_forward", "host_ts": 101.0,
                 "extend_num_tokens": 7},
            ]
            jsonl.write_text("".join(json.dumps(r) + "\n" for r in lines))
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["health"]["n_unknown_mode_forward_lines"], 2)
        self.assertTrue(any("unknown forward mode" in w for w in rec["warnings"]),
                        rec["warnings"])
        self.assertEqual(
            rec["boot_total_from_events"]["extend_tokens_per_forward"]["sum"], 512
        )

    def test_a_prefill_mode_with_no_extend_tokens_is_not_counted_either(self):
        """The reader's predicate must be the WRITER's predicate, both halves.

        The probe counts a forward into channel 2 only when the mode is a
        prefill mode AND the extend count is positive.  A reader that dropped
        the second half would let a zero-token prefill line into the statistics
        (it would become the `min`), and the two passes over the same stream
        would no longer be the same rule.  The engine does not emit such a line
        today; the point is that the two copies of the rule agree, not that the
        input is reachable.
        """
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "x.jsonl"
            lines = [
                {"event": "chunk_open", "schema_chunk": AN.WRITER_SCHEMA_CURRENT,
                 "arm": "a", "host_ts": 99.0},
                {"event": "prefill_forward", "host_ts": 100.0,
                 "forward_mode": "EXTEND", "extend_num_tokens": 512},
                {"event": "prefill_forward", "host_ts": 100.5,
                 "forward_mode": "EXTEND", "extend_num_tokens": 0},
            ]
            jsonl.write_text("".join(json.dumps(r) + "\n" for r in lines))
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {}, bad_lines=bad)
        e = rec["boot_total_from_events"]["extend_tokens_per_forward"]
        self.assertEqual(e["n_forwards"], 1)
        self.assertEqual(e["min"], 512, "a zero-token line must not set the min")
        self.assertEqual(e["observed_multiset"], [512])

        from sglang.srt.multiplex.chunk_probe import ChunkAccountant
        a = ChunkAccountant()
        a.record_forward("EXTEND", 512)
        a.record_forward("EXTEND", 0)
        w = a.summary()["extend_tokens_per_forward"]
        self.assertEqual((w["n_forwards"], w["min"]), (e["n_forwards"], e["min"]),
                         "reader and writer disagree about the same input")

    def test_the_reader_and_the_writer_agree_on_which_modes_are_prefill(self):
        """Two copies of the partition would drift; this pins them together.

        The reader cannot import the probe (it runs offline, on hosts where the
        engine may not be installed), so the sets are duplicated -- and a
        duplicated decision rule is only safe if something fails when the
        copies diverge.
        """
        from sglang.srt.multiplex import chunk_probe as CP_MOD

        self.assertEqual(AN.PREFILL_FORWARD_MODES, CP_MOD.PREFILL_FORWARD_MODES)
        self.assertEqual(AN.NONPREFILL_FORWARD_MODES,
                         CP_MOD.NONPREFILL_FORWARD_MODES)
        self.assertEqual(AN.WRITER_SCHEMA_CURRENT, CP_MOD.CHUNK_SCHEMA)

    def test_dropped_events_make_the_extraction_unreliable(self):
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512)],
                forwards=[(100.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            for r in rows:
                r["dropped_events"] = 7
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE")
        self.assertTrue(any("dropped" in p for p in rec["problems"]))

    def test_event_stream_and_accountant_must_agree(self):
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512),
                        (100.5, "b", "admission_ignore_eos", 900, 512)],
                forwards=[(100.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rows = [r for r in rows if not (
                r.get("event") == "chunk_event" and r.get("rid") == "b")]
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE")
        self.assertTrue(any("accountant says" in p for p in rec["problems"]))

    def test_no_summary_at_all_is_an_extraction_problem(self):
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(jsonl, forwards=[(1.0, "EXTEND", 512, 1)])
            rows, bad = AN.read_jsonl(jsonl)
            rows = [r for r in rows if r.get("event") != "chunk_summary"]
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE")
        self.assertEqual(rec["cross_check"], "none")

    def test_missing_shutdown_summary_is_a_warning_not_a_defect(self):
        """The shutdown summary comes from an atexit hook a SIGKILL skips.
        Calling that an instrument defect is the mislabelling gate #21
        forbids; the periodic summary still bounds what can be hidden."""
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512)],
                forwards=[(100.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            for r in rows:
                if r.get("phase") == "shutdown":
                    r["phase"] = "running"   # a periodic summary, not a final one
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["status"], "OK", rec["problems"])
        self.assertEqual(rec["cross_check"], "partial_no_shutdown_summary")
        self.assertTrue(any("shutdown" in w for w in rec["warnings"]))

    def test_a_periodic_summary_that_is_AHEAD_of_the_events_is_still_a_defect(self):
        """Partial cross-checking only forgives the summary LAGGING the stream.
        A summary that counted more than the event lines is a real mismatch."""
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl,
                chunks=[(100.0, "a", "admission_ignore_eos", 1500, 512),
                        (100.5, "b", "admission_ignore_eos", 900, 512)],
                forwards=[(100.0, "EXTEND", 512, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rows = [r for r in rows
                    if not (r.get("event") == "chunk_event" and r.get("rid") == "b")]
            for r in rows:
                if r.get("phase") == "shutdown":
                    r["phase"] = "running"
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["status"], "EXTRACTION_UNRELIABLE")
        self.assertTrue(any("accountant says" in p for p in rec["problems"]))

    def test_configuration_caveats_are_warned_about_not_corrected(self):
        """--enable-mixed-chunk folds decode tokens into extend_num_tokens
        (schedule_batch.py:1899), and true dual worker races channel 2.  Both
        must show up in the record; neither may silently alter a number."""
        for key, needle in (
            ("enable_mixed_chunk", "mixed-chunk"),
            ("true_dual_worker", "TRUE_DUAL_WORKER"),
            ("enable_dynamic_chunking", "dynamic-chunking"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                jsonl = Path(d) / "chunk.jsonl"
                write_probe_jsonl(jsonl, cfg={key: True},
                                  forwards=[(1.0, "EXTEND", 512, 1)])
                rows, bad = AN.read_jsonl(jsonl)
                rec = AN.build_record(rows, {}, bad_lines=bad)
                self.assertEqual(rec["status"], "OK", rec["problems"])
                self.assertTrue(
                    any(needle in w for w in rec["warnings"]),
                    f"no warning for {key}: {rec['warnings']}",
                )
                self.assertEqual(
                    rec["boot_total_from_events"][
                        "extend_tokens_per_forward"]["sum"], 512,
                    "a caveat must not change the recorded number",
                )

    def test_p1g_banner_survives_the_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            jsonl = Path(d) / "chunk.jsonl"
            write_probe_jsonl(
                jsonl, arm="fused_mono",
                cfg={"requested_chunked_prefill_size": -1,
                     "realized_chunked_prefill_size": None,
                     "piecewise_cuda_graph_tokens_len": 0,
                     "piecewise_cuda_graph_tokens_min": None,
                     "piecewise_cuda_graph_tokens_max": None,
                     "piecewise_cuda_graph_max_tokens": -1},
                forwards=[(1.0, "EXTEND", 1500, 1)],
            )
            rows, bad = AN.read_jsonl(jsonl)
            rec = AN.build_record(rows, {}, bad_lines=bad)
        self.assertEqual(rec["requested_chunked_prefill_size"], -1)
        self.assertIsNone(rec["realized_chunked_prefill_size"])
        self.assertEqual(rec["piecewise_cuda_graph_tokens"]["len"], 0)
        self.assertEqual(rec["boot_total_from_events"][
            "extend_tokens_per_forward"]["max"], 1500)


# ---------------------------------------------------------------------------
# 2. the acceptance checker
# ---------------------------------------------------------------------------


class TestChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = json.loads(EXPECT_PATH.read_text())

    def test_p1a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            _, rec = boot_record(d)
        self.assertEqual(CK.check_p1a(rec)[0], CK.PASS)

    def test_p1a_budget_overrun_is_an_engine_finding_not_an_instrument_failure(self):
        with tempfile.TemporaryDirectory() as d:
            _, rec = boot_record(d, max_extend=600)
        label, note, _ = CK.check_p1a(rec)
        self.assertEqual(label, CK.ENGINE_BUDGET_EXCEEDED)
        self.assertIn("finding about the engine", note)
        self.assertNotEqual(label, CK.EXTRACTION_EMPTY)

    def test_p1a_boot_failure_is_not_a_gate_failure(self):
        self.assertEqual(CK.check_p1a({"boot_ok": False})[0], CK.BOOT_FAILED)
        self.assertEqual(
            CK.check_p1a({"status": "EXTRACTION_UNRELIABLE", "problems": ["x"]})[0],
            CK.EXTRACTION_EMPTY,
        )

    def test_p1a_zero_chunking_fails(self):
        with tempfile.TemporaryDirectory() as d:
            _, rec = boot_record(d, n_requests_chunked=0, n_chunk_events=0)
        self.assertEqual(CK.check_p1a(rec)[0], CK.FAIL)

    def test_p1b_pass_and_the_two_ways_it_fails(self):
        with tempfile.TemporaryDirectory() as d:
            _, ok = boot_record(
                d, arm="mono", requested_cps=-1, realized_cps=None,
                n_requests_chunked=0, n_chunk_events=0,
                n_chunk_events_admission=0, n_chunk_events_continuation=0,
                max_extend=1500,
            )
            _, small = boot_record(
                d, arm="mono2", requested_cps=-1, realized_cps=None,
                n_requests_chunked=0, n_chunk_events=0,
                n_chunk_events_admission=0, n_chunk_events_continuation=0,
                max_extend=400,
            )
            _, chunked = boot_record(
                d, arm="mono3", requested_cps=-1, realized_cps=None,
                max_extend=1500,
            )
        self.assertEqual(CK.check_p1b(ok)[0], CK.PASS)
        self.assertEqual(CK.check_p1b(small)[0], CK.FAIL)
        self.assertEqual(CK.check_p1b(chunked)[0], CK.FAIL)

    def test_p1c_detects_a_perturbed_decode(self):
        on = {"responses": [{"output_ids": [1, 2, 3], "text": "a",
                             "finish_reason": "length"}]}
        off = {"responses": [{"output_ids": [1, 2, 3], "text": "a",
                              "finish_reason": "length"}]}
        self.assertEqual(CK.check_p1c(on, off)[0], CK.PASS)
        off2 = {"responses": [{"output_ids": [1, 2, 4], "text": "a",
                               "finish_reason": "length"}]}
        self.assertEqual(CK.check_p1c(on, off2)[0], CK.PERTURBED)
        off3 = {"responses": [{"output_ids": [1, 2, 3], "text": "b",
                               "finish_reason": "length"}]}
        self.assertEqual(
            CK.check_p1c(on, off3)[0], CK.PERTURBED,
            "text-only differences must also count",
        )

    def test_p1d_fails_when_the_burst_smuggled_in_a_long_prompt(self):
        """Without this, a 'count prompts longer than cps' instrument passes the
        positive control."""
        with tempfile.TemporaryDirectory() as d:
            _, rec = boot_record(
                d, arm="cps4096", requested_cps=4096, realized_cps=4096,
                n_requests_chunked=1, n_chunk_events=1, max_extend=4096,
            )
        good = {"n_prompts": 16, "n_prompts_longer_than_cps": 0,
                "requested_lens": [341] * 16, "served_lens": [341] * 16}
        bad = dict(good, n_prompts_longer_than_cps=1)
        self.assertEqual(CK.check_p1d(rec, good, self.exp)[0], CK.PASS)
        self.assertEqual(CK.check_p1d(rec, bad, self.exp)[0], CK.FAIL)

    def test_p1d_zero_chunking_fails_with_the_retry_instruction(self):
        with tempfile.TemporaryDirectory() as d:
            _, rec = boot_record(
                d, arm="cps4096", requested_cps=4096, realized_cps=4096,
                n_requests_chunked=0, n_chunk_events=0, max_extend=4096,
            )
        label, note, _ = CK.check_p1d(
            rec, {"n_prompts": 16, "n_prompts_longer_than_cps": 0,
                  "requested_lens": [341] * 16, "served_lens": [341] * 16},
            self.exp,
        )
        self.assertEqual(label, CK.FAIL)
        self.assertIn("larger burst", note)

    def _p1e_inputs(self, sdev=None, bdev=None):
        s = _exp_phase(self.exp, "S")["expected"]
        b = _exp_phase(self.exp, "B")["expected"]
        s = dict(s, **(sdev or {}))
        b = dict(b, **(bdev or {}))

        def seg(pid, e):
            return {
                "segment_id": f"p1e_{pid}",
                "n_requests_chunked": e["n_requests_chunked"],
                "n_chunk_events": e["n_chunk_events"],
                "n_chunk_events_admission": e["n_chunk_events_admission"],
                "n_chunk_events_continuation": e["n_chunk_events_continuation"],
                "extend_tokens_per_forward": {
                    "n_forwards": e["n_prefill_forwards"],
                    "sum": e["sum_extend_tokens"],
                    "max": e["max_extend_tokens"],
                    "min": 1, "mean": e["mean_extend_tokens"],
                },
            }

        rec = {
            "status": "OK",
            "boot_total_from_events": {"extend_tokens_per_forward": {"n_forwards": 11}},
            "segments": [seg("S", s), seg("B", b)],
        }
        phases = {"phases": {
            "S": {"n_prompts_longer_than_cps": s["n_prompts_longer_than_cps"],
                  "length_mismatch": False},
            "B": {"n_prompts_longer_than_cps": b["n_prompts_longer_than_cps"],
                  "length_mismatch": False},
        }}
        return rec, phases

    def test_p1e_pass_on_the_registered_values(self):
        rec, phases = self._p1e_inputs()
        label, note, obs = CK.check_p1e(rec, phases, self.exp)
        self.assertEqual(label, CK.PASS, note)
        self.assertEqual(len({
            obs["combined"]["n_requests_chunked"],
            obs["combined"]["n_chunk_events"],
            obs["combined"]["n_prompts_longer_than_cps"],
        }), 3)

    def test_p1e_fails_when_the_instrument_measures_prompt_length(self):
        """Simulate definition (3): the counter equals the number of prompts
        longer than cps, so phase B reports nothing."""
        rec, phases = self._p1e_inputs(
            sdev={"n_chunk_events": 2, "n_chunk_events_continuation": 0},
            bdev={"n_requests_chunked": 0, "n_chunk_events": 0,
                  "n_chunk_events_admission": 0},
        )
        label, note, _ = CK.check_p1e(rec, phases, self.exp)
        self.assertEqual(label, CK.FAIL)
        self.assertIn("n_requests_chunked", note)

    def test_p1e_fails_when_the_phases_collide(self):
        rec, phases = self._p1e_inputs(
            sdev={"n_chunk_events": 2, "n_chunk_events_continuation": 0}
        )
        label, note, _ = CK.check_p1e(rec, phases, self.exp)
        self.assertEqual(label, CK.FAIL)

    def test_p1e_rejects_a_served_length_mismatch(self):
        rec, phases = self._p1e_inputs()
        phases["phases"]["S"]["length_mismatch"] = True
        phases["phases"]["S"]["served_lens"] = [1501, 513, 514, 101]
        self.assertEqual(CK.check_p1e(rec, phases, self.exp)[0], CK.EXTRACTION_EMPTY)

    def test_p1f_margin_and_thinness(self):
        near = {"pairs": [{"on": 5.00, "off": 5.02}, {"on": 5.03, "off": 5.01},
                          {"on": 4.99, "off": 5.00}]}
        far = {"pairs": [{"on": 4.5, "off": 5.0}, {"on": 4.6, "off": 5.0},
                         {"on": 4.4, "off": 5.0}]}
        thin = {"pairs": [{"on": 5.0, "off": 5.0}, {"on": 5.0, "off": 5.0}]}
        self.assertEqual(CK.check_p1f(near)[0], CK.PASS)
        self.assertEqual(CK.check_p1f(far)[0], CK.PERTURBED)
        self.assertEqual(CK.check_p1f(thin)[0], CK.EXTRACTION_EMPTY)
        _, _, obs = CK.check_p1f(near)
        self.assertIn("normality", obs["ci"]["assumption"])

    def test_p1g_requires_the_banner_on_every_boot(self):
        with tempfile.TemporaryDirectory() as d:
            _, ok = boot_record(d)
            _, missing = boot_record(
                d, arm="mono", disable_piecewise=None,
                pw_tokens={"len": None, "min": None, "max": None},
            )
        self.assertEqual(CK.check_p1g({"cps512": ok})[0], CK.PASS)
        self.assertEqual(
            CK.check_p1g({"cps512": ok, "mono": missing})[0], CK.EXTRACTION_EMPTY
        )

    def test_margin_is_the_project_constant(self):
        self.assertEqual(CK.NEUTRALITY_MARGIN, 0.03)


# ---------------------------------------------------------------------------
# 3. the verdict is a PRODUCT OF TWO AXES (P1 repair 2)
#
# Rule rev 1 had one ladder, and `FAIL` only reached its `else`.  In job 899768
# three EXTRACTION_EMPTY labels therefore swallowed P1-e's FAIL: an arithmetic
# disagreement with pre-registered, hand-computed expectations disappeared from
# the verdict because a DIFFERENT test's plumbing had broken.
# ---------------------------------------------------------------------------


class TestVerdictAxes(unittest.TestCase):
    MEASUREMENT_LABELS = {
        CK.ENGINE_BUDGET_EXCEEDED: "P1_BLOCKED_ENGINE_FINDING",
        CK.PERTURBED: "P1_BLOCKED_NOT_AN_OBSERVER",
        CK.EXTRACTION_EMPTY: "P1_BLOCKED_INSTRUMENT",
        CK.BOOT_FAILED: "P1_INCONCLUSIVE_BOOT",
        CK.NOT_RUN: "P1_INCOMPLETE",
    }

    def test_the_prereg_four_way_classification_is_unchanged(self):
        """Repair 2 may not renegotiate PREREG section 2.3.  Same labels, same
        priority, same names -- only the FAIL axis is new."""
        for label, want in self.MEASUREMENT_LABELS.items():
            with self.subTest(label=label):
                self.assertEqual(
                    CK.measurement_verdict({CK.PASS, label}), want
                )
        self.assertEqual(
            CK.measurement_verdict({CK.PASS}), CK.MEASUREMENT_CLEAN
        )
        self.assertEqual(
            CK.measurement_verdict({CK.PASS, CK.FAIL}), CK.MEASUREMENT_CLEAN,
            "FAIL is not a member of the section 2.3 classification",
        )

    def test_a_fail_survives_every_measurement_label(self):
        """★The hole.  Neither axis may absorb the other."""
        for label, blocked in self.MEASUREMENT_LABELS.items():
            with self.subTest(label=label):
                labels = {CK.PASS, CK.FAIL, label}
                m = CK.measurement_verdict(labels)
                e = CK.expectation_verdict(labels)
                v = CK.compose_verdict(m, e)
                self.assertEqual(m, blocked)
                self.assertEqual(e, CK.EXPECTATION_REFUTED)
                self.assertIn(blocked, v)
                self.assertIn(
                    CK.EXPECTATION_REFUTED, v,
                    f"{blocked} swallowed the refutation: {v}",
                )

    def test_the_rule_rev_1_ladder_would_fail_this_check(self):
        """MUTATION (PROJECT_STATUS.md "방법론 게이트" #53): the reverted rule,
        written out, must not pass the check above."""
        def rev1(labels):
            if labels == {CK.PASS}:
                return "P1_ACCEPTED"
            for label, name in self.MEASUREMENT_LABELS.items():
                if labels & {label}:
                    return name
            return "P1_REJECTED"

        v = rev1({CK.PASS, CK.FAIL, CK.EXTRACTION_EMPTY})
        self.assertEqual(v, "P1_BLOCKED_INSTRUMENT")
        self.assertNotIn(
            CK.EXPECTATION_REFUTED, v,
            "the reverted rule already reported the refutation, so repair 2 "
            "would be a no-op",
        )

    def test_a_blocked_run_is_never_reported_as_expectations_held(self):
        labels = {CK.EXTRACTION_EMPTY, CK.BOOT_FAILED}
        self.assertEqual(
            CK.expectation_verdict(labels), CK.EXPECTATION_UNEVALUATED
        )
        v = CK.compose_verdict(
            CK.measurement_verdict(labels), CK.expectation_verdict(labels)
        )
        self.assertIn(CK.EXPECTATION_UNEVALUATED, v)
        self.assertNotIn("P1_ACCEPTED", v)

    def test_only_all_pass_accepts(self):
        self.assertEqual(
            CK.compose_verdict(
                CK.measurement_verdict({CK.PASS}),
                CK.expectation_verdict({CK.PASS}),
            ),
            "P1_ACCEPTED",
        )
        for labels in (
            {CK.PASS, CK.FAIL},
            {CK.PASS, CK.NOT_RUN},
            {CK.PASS, CK.EXTRACTION_EMPTY},
            {CK.PASS, CK.FAIL, CK.PERTURBED},
            {CK.EXTRACTION_EMPTY},
        ):
            with self.subTest(labels=sorted(labels)):
                v = CK.compose_verdict(
                    CK.measurement_verdict(labels),
                    CK.expectation_verdict(labels),
                )
                self.assertNotEqual(v, "P1_ACCEPTED")

    def test_the_clean_plus_refuted_name_is_the_rev1_name(self):
        """The vocabulary must not fork: a clean run whose predicate is false
        is still P1_REJECTED."""
        self.assertEqual(
            CK.compose_verdict(
                CK.measurement_verdict({CK.PASS, CK.FAIL}),
                CK.expectation_verdict({CK.PASS, CK.FAIL}),
            ),
            "P1_REJECTED",
        )


class TestCheckerOnJob899768(unittest.TestCase):
    """Replay the repaired checker over the artefacts that produced the hole.

    This is a re-read of existing files, NOT a new measurement: no GPU is
    touched and no arm is scored (PREREG section 9).
    """

    JOB = CP / "p1_accept_899768"

    def setUp(self):
        path = self.JOB / "P1_VERDICT.json"
        if not path.is_file():
            self.skipTest(f"job artefacts absent: {self.JOB}")
        self.old = json.loads(path.read_text())
        if "verdict_expectation" in self.old:
            # someone re-scored the archive with rule rev 2; the historical
            # fixture is gone and there is nothing left here to check
            self.skipTest("the archived verdict was re-scored with rule rev 2")

    def test_the_recorded_verdict_hid_a_FAIL(self):
        old = self.old
        self.assertEqual(old["verdict"], "P1_BLOCKED_INSTRUMENT")
        self.assertEqual(old["tests"]["P1-e"]["label"], CK.FAIL)

    def test_the_repaired_rule_reports_both_axes_on_the_same_labels(self):
        labels = {t["label"] for t in self.old["tests"].values()}
        m = CK.measurement_verdict(labels)
        e = CK.expectation_verdict(labels)
        v = CK.compose_verdict(m, e)
        self.assertEqual(m, "P1_BLOCKED_INSTRUMENT")
        self.assertEqual(e, CK.EXPECTATION_REFUTED)
        self.assertEqual(v, "P1_BLOCKED_INSTRUMENT AND P1_EXPECTATION_REFUTED")


def _exp_phase(exp, pid):
    return next(p for p in exp["phases"] if p["id"] == pid)


class TestExpectationsFileIsSelfConsistent(unittest.TestCase):
    """The hand-derivation must at least be arithmetically closed: the forward
    multiset has to reproduce the totals it is quoted alongside."""

    @classmethod
    def setUpClass(cls):
        cls.exp = json.loads(EXPECT_PATH.read_text())

    def test_multisets_match_the_totals(self):
        for phase in self.exp["phases"]:
            e = phase["expected"]
            m = e["extend_tokens_per_forward_multiset"]
            with self.subTest(phase=phase["id"]):
                self.assertEqual(len(m), e["n_prefill_forwards"])
                self.assertEqual(sum(m), e["sum_extend_tokens"])
                self.assertEqual(max(m), e["max_extend_tokens"])
                self.assertAlmostEqual(
                    sum(m) / len(m), e["mean_extend_tokens"], places=6
                )
                self.assertEqual(
                    sum(m), sum(phase["prompt_lens"]),
                    "the forwards must carry exactly the prompt tokens",
                )
                self.assertEqual(
                    e["n_chunk_events"],
                    e["n_chunk_events_admission"] + e["n_chunk_events_continuation"],
                )
                self.assertLessEqual(
                    e["n_prompts_longer_than_cps"], e["n_requests_chunked"],
                    "a prompt longer than cps must be chunked, so (3) <= (1)",
                )

    def test_combined_claim_is_the_sum_of_the_phases(self):
        c = self.exp["separation_claim"]["combined_boot_S_plus_B"]
        for key in ("n_requests_chunked", "n_chunk_events",
                    "n_prompts_longer_than_cps"):
            self.assertEqual(
                c[key], sum(p["expected"][key] for p in self.exp["phases"]), key
            )
        self.assertEqual(len(set(c.values())), 3, "the claim is not a separation")

    def test_every_phase_declares_where_it_is_exact(self):
        for phase in self.exp["phases"]:
            self.assertIn("exact_on", phase)
            if "gpu" not in phase["exact_on"]:
                self.assertIn(
                    "gpu_assertions", phase,
                    "a phase that is not exact on GPU must register what IS "
                    "checked there, or the GPU run checks nothing",
                )


class TestHarnessWiring(unittest.TestCase):
    SBATCH = CP / "p1_accept.sbatch"

    def test_sbatch_has_the_mandatory_comment_before_the_first_command(self):
        lines = self.SBATCH.read_text().splitlines()
        first_cmd = next(
            i for i, l in enumerate(lines)
            if l.strip() and not l.startswith("#")
        )
        comments = [
            i for i, l in enumerate(lines)
            if l.startswith('#SBATCH --comment="field=efficientai;appl=pytorch"')
        ]
        self.assertTrue(comments, "the mandatory --comment directive is missing")
        self.assertLess(
            comments[0], first_cmd,
            "Slurm ignores #SBATCH after the first command line",
        )

    def test_sbatch_boots_all_three_arms_and_an_off_boot(self):
        src = self.SBATCH.read_text()
        for frag in ("run_probe_boot 512 cps512_on p1e",
                     "run_probe_boot -1 mono_on mono",
                     "run_probe_boot 4096 cps4096_on d",
                     "greedy_off.json"):
            self.assertIn(frag, src)

    def test_sbatch_forces_a_summary_before_killing_the_server(self):
        """Without the flush ping a SIGKILL can leave the roll-up with no
        cross-check at all."""
        src = self.SBATCH.read_text()
        self.assertIn("Flush ping", src)
        self.assertLess(
            src.index("Flush ping"), src.index('kill_server "$PID" "$PORT"\n  python'),
            "the flush ping must happen before the server is killed",
        )

    # -- kill sequence (P1 repair 1, after job 899768) ----------------------
    #
    # The old body was `kill $PID; pkill -9 ...` with no grace period.  Two
    # facts made that lossy: the probe lives in the SCHEDULER process, which
    # SGLang's launcher reaps with SIGKILL (launch_server.py `finally:
    # kill_process_tree(os.getpid(), include_parent=False)` ->
    # srt/utils/common.py:1054 `child.kill()`), and Python's default SIGTERM
    # disposition skips `atexit`.  So the harness must signal the CHILDREN, not
    # only the launcher, and must wait before it hammers.
    _MOCK_TREE = textwrap.dedent(
        """
        set -uo pipefail
        MARK="$1"
        # "scheduler": traps SIGTERM the way the probe's handler does
        bash -c 'trap "echo CHILD_SIGTERM >> \\"$0\\"; exit 0" TERM
                 while true; do sleep 0.05; done' "$MARK" &
        CHILD=$!
        # "launcher": on SIGTERM it reaps the child with SIGKILL, like SGLang
        trap 'kill -9 $CHILD 2>/dev/null; exit 0' TERM
        echo $$ > "$MARK.parent"
        wait
        """
    )

    def _run_kill_sequence(self, body):
        """Drive `body` against a launcher/scheduler mock; return the marker."""
        src = self.SBATCH.read_text()
        funcs = "\n".join(
            block for block in (
                _extract_shell_function(src, "descendants"),
                _extract_shell_function(src, "kill_server"),
            )
        )
        with tempfile.TemporaryDirectory() as d:
            mark = Path(d) / "mark"
            mock = Path(d) / "mock.sh"
            mock.write_text(self._MOCK_TREE)
            script = (
                "set -uo pipefail\n"
                f"{funcs}\n"
                f"bash {mock} {mark} &\n"
                "PID=$!\n"
                "for i in $(seq 1 100); do "
                f"[ -e {mark}.parent ] && break; sleep 0.05; done\n"
                # give the mock's own child time to install its trap
                "sleep 0.3\n"
                f"{body}\n"
                "wait $PID 2>/dev/null\n"
            )
            env = dict(os.environ,
                       KILL_FLUSH_S="0.4", KILL_GRACE_S="3", KILL_SETTLE_S="0.2")
            subprocess.run(["bash", "-c", script], env=env, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return mark.read_text() if mark.exists() else ""

    def test_kill_server_signals_the_scheduler_before_it_hammers(self):
        got = self._run_kill_sequence('kill_server "$PID" 65000')
        self.assertIn(
            "CHILD_SIGTERM", got,
            "kill_server never gave the process that owns the probe a SIGTERM; "
            "that is how job 899768 produced 0-byte probe files",
        )

    def test_the_old_kill_sequence_fails_the_same_check(self):
        """MUTATION (PROJECT_STATUS.md "방법론 게이트" #53).  The pre-repair
        body must not pass the check above, or the check proves nothing."""
        got = self._run_kill_sequence(
            'kill "$PID" 2>/dev/null; '
            'pkill -9 -f "launch_server.*--port 65000" 2>/dev/null; sleep 0.4'
        )
        self.assertNotIn(
            "CHILD_SIGTERM", got,
            "the pre-repair sequence also reached the child, so the check "
            "above is not discriminating",
        )

    def test_kill_server_waits_for_the_launcher_before_sigkill(self):
        src = self.SBATCH.read_text()
        # comments quote the OLD sequence, so the ordering check must read the
        # executable lines only (`pkill -9` also contains "kill -9")
        body = "\n".join(
            l for l in _extract_shell_function(src, "kill_server").splitlines()
            if not l.lstrip().startswith("#")
        )
        self.assertLess(
            body.index("kill -TERM"), body.index("kill -9"),
            "SIGKILL must not precede SIGTERM",
        )
        self.assertIn("kill -0", body, "no exit polling between TERM and KILL")
        self.assertIn("KILL_GRACE_S", body, "the grace period is not tunable")

    def test_off_boot_passes_an_empty_probe_path(self):
        """P1-c is only meaningful if the OFF boot really has the probe off."""
        src = self.SBATCH.read_text()
        self.assertIn('unset PDMUX_CHUNK_PROBE_PATH', src)
        self.assertIn('boot_server 512 "$PORT" "$SRVLOG" "" "cps512_off"', src)



# ---------------------------------------------------------------------------
# 4. the phase windows the harness registers (repair 2, 2026-09-01)
# ---------------------------------------------------------------------------


class TestWorkloadSegmentWindows(unittest.TestCase):
    """`p1_accept_workload.py` used to register `t_end + 1.0` for every phase.

    Phase S is awaited to completion and phase B starts about 1.5 ms later, so
    the grace pushed the S window a full second INTO phase B and the roll-up
    counted phase B's first forwards in both.  That is where job 899768's
    "phase S: n_prefill_forwards expected 7, got 43" came from.

    The registered expectations were never wrong: they are stated for
    non-overlapping phase windows.  What was wrong is the instrument, so the
    repair is here and `p1e_expectations.json` is untouched.
    """

    JOB = CP / "p1_accept_899768"

    def _real_phase_times(self):
        path = self.JOB / "workload_cps512_on.json"
        if not path.is_file():
            self.skipTest(f"job artefacts absent: {self.JOB}")
        ph = json.loads(path.read_text())["phases"]
        return [
            {"segment_id": f"p1e_{pid}", "phase": pid,
             "t_start": ph[pid]["t_start"], "t_end": ph[pid]["t_end"]}
            for pid in ("S", "B")
        ]

    def test_the_windows_are_disjoint_on_the_real_phase_times(self):
        segs = WL.close_segments(self._real_phase_times())
        s, b = segs
        self.assertLessEqual(s["t_end"], b["t_start"],
                             "phase S must not reach into phase B")
        self.assertGreater(s["t_end"], s["t_start"])
        rec = AN.build_record([], {"segments": segs})
        self.assertEqual(rec.get("segment_consistency", {}).get("windows_overlap"),
                         [], "the roll-up still sees an overlap")

    def test_the_pre_repair_grace_is_what_899768_registered(self):
        """The negative control: without the clamp the check would be vacuous.

        This reproduces the defective windows from the same real timestamps and
        shows the roll-up flags them -- so the assertion above is testing the
        clamp and not an accident of the data.
        """
        raw = self._real_phase_times()
        bad = [dict(s, t_end=s["t_end"] + 1.0) for s in raw]
        self.assertGreater(bad[0]["t_end"], bad[1]["t_start"])
        rec = AN.build_record([], {"segments": bad})
        self.assertTrue(rec["segment_consistency"]["windows_overlap"])
        self.assertTrue(any("overlap" in p for p in rec["problems"]))
        recorded = self.JOB / "segments_cps512_on.json"
        if recorded.is_file():
            got = json.loads(recorded.read_text())["segments"]
            self.assertAlmostEqual(got[0]["t_end"], bad[0]["t_end"], places=6,
                                   msg="this is the window the job registered")

    def test_the_last_segment_keeps_its_grace(self):
        """The clamp is against the next window, not a blanket truncation: the
        trailing forwards of the last phase must still be inside it."""
        segs = WL.close_segments([{"segment_id": "only", "t_start": 10.0,
                                   "t_end": 20.0}])
        self.assertAlmostEqual(segs[0]["t_end"], 20.0 + WL.SEGMENT_GRACE_S)

    def test_a_clamped_window_records_what_it_clamped(self):
        segs = WL.close_segments([
            {"segment_id": "a", "t_start": 0.0, "t_end": 10.0},
            {"segment_id": "b", "t_start": 10.5, "t_end": 20.0},
        ])
        self.assertAlmostEqual(segs[0]["t_end"], 10.5)
        self.assertAlmostEqual(segs[0]["raw_t_end"], 10.0)
        self.assertTrue(segs[0]["clamped_to_next_segment"])
        self.assertFalse(segs[1]["clamped_to_next_segment"])

    def test_windows_are_never_inverted(self):
        """Two phases that really do overlap in wall-clock time must not
        produce t_end < t_start; the overlap is reported by the roll-up, and
        the writer's job is only to not manufacture a negative window."""
        segs = WL.close_segments([
            {"segment_id": "a", "t_start": 0.0, "t_end": 30.0},
            {"segment_id": "b", "t_start": 5.0, "t_end": 20.0},
        ])
        for s in segs:
            self.assertGreaterEqual(s["t_end"], s["t_start"])

    def test_no_phase_registers_a_bare_grace_any_more(self):
        """Source-level guard: three call sites had the same `+ 1.0`."""
        src = (CP / "p1_accept_workload.py").read_text()
        body = src[src.index("def main("):]
        self.assertEqual(
            body.count('"t_end": s["t_end"] + 1.0'), 0,
            "a phase still registers a bare grace instead of a clamped window",
        )
        self.assertEqual(body.count("close_segments("), 1,
                         "every mode must close its windows through one path")


# ---------------------------------------------------------------------------
# 5. re-reading job 899768 with both repairs (no GPU, no new measurement)
# ---------------------------------------------------------------------------


class TestReanalysisOfJob899768(unittest.TestCase):
    """Re-READ of the archived raw data with the repaired reader and windows.

    ★This is not a P1-e pass.  Nothing was measured here: the assertion is that
    the repaired instrument, counting the SAME bytes, reproduces the registered
    expectations.  P1-e on a new engine run is still unexecuted, and the
    re-read cannot show that the repaired PROBE emits the right lines on a GPU
    -- only that the repaired READER reads these ones correctly.
    """

    JOB = CP / "p1_accept_899768"

    @classmethod
    def setUpClass(cls):
        cls.jsonl = cls.JOB / "chunk_cps512_on.jsonl"
        cls.wl = cls.JOB / "workload_cps512_on.json"
        if not (cls.jsonl.is_file() and cls.wl.is_file()):
            raise unittest.SkipTest(f"job artefacts absent: {cls.JOB}")
        cls.exp = json.loads(EXPECT_PATH.read_text())
        ph = json.loads(cls.wl.read_text())["phases"]
        segs = WL.close_segments([
            {"segment_id": f"p1e_{pid}", "phase": pid,
             "t_start": ph[pid]["t_start"], "t_end": ph[pid]["t_end"]}
            for pid in ("S", "B")
        ])
        rows, bad = AN.read_jsonl(cls.jsonl)
        cls.rec = AN.build_record(rows, {"segments": segs}, bad_lines=bad)

    def _seg(self, pid):
        return next(s for s in self.rec["segments"] if s["segment_id"] == f"p1e_{pid}")

    def test_the_raw_file_still_contains_the_decode_records(self):
        """The evidence may not be deleted by the repair (nothing rewrote it)."""
        rows, _ = AN.read_jsonl(self.jsonl)
        modes = [r.get("forward_mode") for r in rows
                 if r.get("event") in ("prefill_forward", "nonprefill_forward")]
        self.assertEqual(len(modes), 116)
        self.assertEqual(modes.count("DECODE"), 101)
        self.assertEqual(modes.count("EXTEND"), 15)

    def test_phase_S_reproduces_the_registered_numbers(self):
        exp = _exp_phase(self.exp, "S")["expected"]
        got = self._seg("S")
        e = got["extend_tokens_per_forward"]
        self.assertEqual(e["n_forwards"], exp["n_prefill_forwards"])
        self.assertEqual(e["sum"], exp["sum_extend_tokens"])
        self.assertEqual(e["max"], exp["max_extend_tokens"])
        self.assertEqual(got["n_requests_chunked"], exp["n_requests_chunked"])
        self.assertEqual(got["n_chunk_events"], exp["n_chunk_events"])
        self.assertEqual(got["n_chunk_events_admission"],
                         exp["n_chunk_events_admission"])
        self.assertEqual(got["n_chunk_events_continuation"],
                         exp["n_chunk_events_continuation"])

    def test_phase_S_reproduces_the_registered_multiset(self):
        exp = _exp_phase(self.exp, "S")["expected"]
        self.assertEqual(
            sorted(self._seg("S")["extend_tokens_per_forward"]["observed_multiset"]),
            sorted(exp["extend_tokens_per_forward_multiset"]),
        )

    def test_phase_B_satisfies_its_registered_gpu_assertions(self):
        phase = _exp_phase(self.exp, "B")
        got = self._seg("B")
        g = phase["gpu_assertions"]
        self.assertGreaterEqual(got["n_requests_chunked"], g["n_requests_chunked_min"])
        self.assertGreaterEqual(got["n_chunk_events"], g["n_chunk_events_min"])
        self.assertLessEqual(got["extend_tokens_per_forward"]["max"],
                             g["max_extend_tokens_max"])
        # this boot also happened to hit the exact CPU-loop numbers
        exp = phase["expected"]
        self.assertEqual(got["extend_tokens_per_forward"]["n_forwards"],
                         exp["n_prefill_forwards"])
        self.assertEqual(got["extend_tokens_per_forward"]["sum"],
                         exp["sum_extend_tokens"])
        self.assertEqual(
            sorted(got["extend_tokens_per_forward"]["observed_multiset"]),
            sorted(exp["extend_tokens_per_forward_multiset"]),
        )

    def test_the_windows_no_longer_overlap_and_the_counters_are_subsets(self):
        self.assertEqual(self.rec["segment_consistency"]["windows_overlap"], [])
        self.assertNotIn(
            "EXTRACTION_UNRELIABLE", self.rec["status"],
            f"problems: {self.rec['problems']}",
        )

    def test_the_checker_scores_p1e_on_the_reread(self):
        """The registered predicate, re-scored on the re-read record.

        Still not a measurement: `check_p1e` is a comparison, and the numbers it
        compares came out of the same archived bytes.
        """
        phases_doc = json.loads(self.wl.read_text())
        label, note, obs = CK.check_p1e(self.rec, phases_doc, self.exp)
        self.assertEqual(label, CK.PASS, note)
        self.assertEqual(obs["combined"]["n_requests_chunked"], 5)
        self.assertEqual(obs["combined"]["n_chunk_events"], 6)
        self.assertEqual(obs["combined"]["n_prompts_longer_than_cps"], 2)

    def test_the_archived_windows_still_fail_the_same_check(self):
        """Mutation: with the windows the job actually registered, the re-read
        must still fail.  Otherwise the repair above is not what fixed it."""
        recorded = self.JOB / "segments_cps512_on.json"
        if not recorded.is_file():
            self.skipTest("archived segments file absent")
        rows, bad = AN.read_jsonl(self.jsonl)
        rec = AN.build_record(rows, json.loads(recorded.read_text()), bad_lines=bad)
        phases_doc = json.loads(self.wl.read_text())
        label, note, _ = CK.check_p1e(rec, phases_doc, self.exp)
        self.assertNotEqual(label, CK.PASS,
                            "the overlapping windows must still be caught")

if __name__ == "__main__":
    unittest.main()
