"""Gate-logic tests for the P5 re-measurement analyzer (decode_knee_vs_ctx_v2).

WHY THIS EXISTS.  The batch-1 gate scoring of job 873783 (result-analyst,
`results/r0c/P5_GATES_BATCH1_2026-08-05.md`, UNAUDITED) found three tool defects
and one gate mis-specification.  Every one of them was a *silent* failure of the
scoring code, not of the measurement, and nothing asserted the scoring logic:

  D1  the analyzer printed `GATE C ... PASS` and then `GATES: closure=FAIL`,
      because the harness counted closure failures with a pattern
      (`mode=$SM@${CTX}r`) that also matches the DISCARDED warm-up tag `rw`.
      The single failure in the whole job was a warm-up block.
  D2  `mamba_spread` was max/min over reps -> identically 1.00 at n_reps=1
      (methodology gate #6: a check that always passes is an identity).
  D3  CELL `bs` was `statistics.mode`, which labelled a 6 x bs19 + 8 x bs13
      mixture `bs=13` while the median still saw both.
  D4  GATE N compared per_mamba across ctx cells whose realised batch differed,
      confounding ctx with batch: bs 19->13 alone moves per_mamba by
      1.069/1.104/1.139/1.163/1.180 (full/44/24/16/8), i.e. 3 of 5 arms cross
      the 1.15 threshold on batch alone.
  D5  no host-boundness observable was printed at all.

These tests pin the FIXES.  They are pure-CPU checks of the scoring functions on
synthetic records; the timing itself needs a GPU and is covered by the sbatch.

Two properties are load-bearing and deliberately over-tested:
  * `UNSCOREABLE` must be reachable and must NOT be collapsed into PASS -- a gate
    that silently passes an un-compared column is the D4 failure again;
  * the 1.15 threshold and the 0.85 closure threshold are PRE-REGISTERED
    constants.  A test asserts their values so that re-tuning them against
    observed data (methodology gate #8) cannot happen quietly.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = TRACK_ROOT / "results" / "r0c" / "analyze_decode_knee_vs_ctx_v2.py"
SBATCH = TRACK_ROOT / "results" / "r0c" / "decode_knee_vs_ctx_v2.sbatch"

_spec = importlib.util.spec_from_file_location("p5_analyzer", ANALYZER)
p5 = importlib.util.module_from_spec(_spec)
sys.modules["p5_analyzer"] = p5
_spec.loader.exec_module(p5)


def block(ctx, sm, rep=1, rnd=1, blk=1, bs=32, dirty=0, closure=0.99, fwd=50.0,
          per_mamba=0.5, per_attn=1.0, per_attn_core=0.9, b_mlp=1.35, b_other=1.8):
    """A synthetic BLOCK record in the analyzer's parsed form."""
    return dict(ctx=ctx, rep=rep, sm=sm, round=rnd, blk=blk, blk_n=8, dirty=dirty,
                bs=bs, ntok=bs, closure=closure, closure_min=closure, fwd=fwd,
                b_attn=per_attn * 9, b_mlp=b_mlp, b_other=b_other,
                b_mamba=per_mamba * 54, b_attn_core=per_attn_core * 9,
                per_attn=per_attn, per_mamba=per_mamba, per_attn_core=per_attn_core)


def block_line(ctx, sm, rep=1, rnd=1, blk=0, bs=32, dirty=0, closure=0.9890,
               fwd=86.86, per_attn=8.14246, per_mamba=0.17571, per_attn_core=7.98362):
    """A verbatim-format BLOCK line as the harness writes it into the result file."""
    return (
        f"BLOCK ctx={ctx} rep={rep} sm={sm} round={rnd} [2026-08-05 10:03:21] "
        f"ZBLT2 mode={sm}@{ctx}r{rnd} ctxlen={ctx + 10} n=8 bs={bs} ntok={bs} | "
        f"attn_total=73.282 mamba_total=9.488 | per-attn(9)={per_attn:.4f} "
        f"per-mamba(54)={per_mamba:.4f} | attn_core_total=71.853 "
        f"per-attn_core(9)={per_attn_core:.4f} closure_cum=0.9890 || "
        f"blk={blk} blk_n=8 dirty={dirty} shape={bs}x{bs} closure={closure:.4f} "
        f"closure_min={closure:.4f} fwd_ms={fwd:.4f} | b_attn=73.2822 b_mlp=1.3549 "
        f"b_other=1.7760 b_mamba=9.4884 b_attn_core=71.8525 | "
        f"b_per_attn={per_attn:.5f} b_per_mamba={per_mamba:.5f} "
        f"b_per_attn_core={per_attn_core:.5f}"
    )


class TestPreRegisteredThresholds(unittest.TestCase):
    """The thresholds are pre-registered; re-tuning them against observed data is
    methodology gate #8.  If a future change edits them, this test must be the
    thing that fails."""

    def test_thresholds_unchanged(self):
        self.assertEqual(p5.GATE_N_MAX_DEFAULT, 1.15)
        self.assertEqual(p5.GATE_C_MIN_DEFAULT, 0.85)

    def test_cli_defaults_match_constants(self):
        # A CLI default that drifts away from the constant is the same failure,
        # only quieter.  The parser is built inside main(), so pin the source.
        src = ANALYZER.read_text()
        self.assertIn("default=GATE_N_MAX_DEFAULT", src)
        self.assertIn("default=GATE_C_MIN_DEFAULT", src)


class TestD1WarmupExclusion(unittest.TestCase):
    """Warm-up (`rw`) blocks are excluded from scoring but still reported."""

    FAIL_LINE = ("[2026-08-05 10:00:14] ZBLT_CLOSURE_FAIL mode=44@1024rw blk=0 "
                 "closure=0.7119 min_fwd=0.2588 threshold=0.85 bucket_ms=44.2103 "
                 "loop_ms=62.1005 -- buckets do not cover the layer loop")
    MEAS_LINE = FAIL_LINE.replace("rw", "r2")

    def test_warmup_tag_is_classified_as_warmup(self):
        rec = p5.parse_closure_fail(self.FAIL_LINE)
        self.assertEqual((rec["sm"], rec["ctx"], rec["round"]), ("44", 1024, "w"))
        self.assertTrue(rec["warmup"])
        self.assertAlmostEqual(rec["closure"], 0.7119)

    def test_measured_round_tag_is_not_warmup(self):
        rec = p5.parse_closure_fail(self.MEAS_LINE)
        self.assertEqual(rec["round"], "2")
        self.assertFalse(rec["warmup"])

    def test_resolver_splits_legacy_ambiguous_counter_via_server_log(self):
        # legacy ARMDONE: one count, cannot tell warm-up from measured
        armdone = {(1024, "44", 1): {"ctx": "1024", "sm": "44", "rep": "1",
                                     "closure_fail": "1"}}
        srv = {(1024, "44", 1): [p5.parse_closure_fail(self.FAIL_LINE)]}
        out = p5.resolve_closure_fails(armdone, srv, srv_files_found=True)
        self.assertEqual(out[(1024, "44", 1)]["measured"], 0)
        self.assertEqual(out[(1024, "44", 1)]["warmup"], 1)
        self.assertEqual(out[(1024, "44", 1)]["source"], "srvlog")

    def test_gate_c_passes_when_the_only_failure_is_warmup(self):
        """The D1 self-contradiction: GATE C PASS then GATES closure=FAIL."""
        cells = {(1024, "44", 1): dict(status="OK", closure=0.9578)}
        resolved = {(1024, "44", 1): dict(measured=0, warmup=1, source="srvlog")}
        verdict, bad, meas, ambiguous = p5.gate_c(cells, resolved, 0.85)
        self.assertEqual(verdict, "PASS")
        self.assertEqual((bad, meas, ambiguous), ([], [], []))

    def test_gate_c_still_fails_on_a_measured_round_failure(self):
        cells = {(1024, "44", 1): dict(status="OK", closure=0.9578)}
        resolved = {(1024, "44", 1): dict(measured=1, warmup=0, source="srvlog")}
        verdict, _, meas, _ = p5.gate_c(cells, resolved, 0.85)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(meas, [((1024, "44", 1), 1)])

    def test_gate_c_is_unscoreable_when_the_counter_cannot_be_split(self):
        """No server log + legacy non-zero counter: never a silent PASS."""
        armdone = {(1024, "44", 1): {"ctx": "1024", "sm": "44", "rep": "1",
                                     "closure_fail": "1"}}
        resolved = p5.resolve_closure_fails(armdone, {}, srv_files_found=False)
        self.assertEqual(resolved[(1024, "44", 1)]["source"], "AMBIGUOUS")
        cells = {(1024, "44", 1): dict(status="OK", closure=0.99)}
        verdict, _, _, ambiguous = p5.gate_c(cells, resolved, 0.85)
        self.assertEqual(verdict, "UNSCOREABLE")
        self.assertEqual(ambiguous, [(1024, "44", 1)])

    def test_disagreement_between_log_scan_and_harness_counter_is_ambiguous(self):
        """The legacy counter matches BOTH round classes, so it must equal the sum
        of the two.  A silent 0 when the counter says 1 is exactly the kind of
        quiet failure this whole batch of fixes exists for."""
        armdone = {(1024, "44", 1): {"ctx": "1024", "sm": "44", "rep": "1",
                                     "closure_fail": "3"}}
        srv = {(1024, "44", 1): [p5.parse_closure_fail(self.FAIL_LINE)]}
        out = p5.resolve_closure_fails(armdone, srv, srv_files_found=True)
        self.assertEqual(out[(1024, "44", 1)]["source"], "AMBIGUOUS")
        self.assertIsNone(out[(1024, "44", 1)]["measured"])

    def test_arm_with_no_failure_line_resolves_to_zero(self):
        armdone = {(256, "8", 1): {"ctx": "256", "sm": "8", "rep": "1",
                                   "closure_fail": "0"}}
        out = p5.resolve_closure_fails(armdone, {}, srv_files_found=True)
        self.assertEqual((out[(256, "8", 1)]["measured"], out[(256, "8", 1)]["warmup"]),
                         (0, 0))
        self.assertEqual(out[(256, "8", 1)]["source"], "srvlog")

    def test_round_scoped_field_is_preferred_when_present(self):
        armdone = {(256, "8", 1): {"ctx": "256", "sm": "8", "rep": "1",
                                   "closure_fail_meas": "0",
                                   "closure_fail_warmup": "3"}}
        out = p5.resolve_closure_fails(armdone, {}, srv_files_found=False)
        self.assertEqual(out[(256, "8", 1)]["measured"], 0)
        self.assertEqual(out[(256, "8", 1)]["warmup"], 3)
        self.assertEqual(out[(256, "8", 1)]["source"], "armdone_round_scoped")

    def test_harness_grep_is_round_scoped(self):
        """The defect lived in the sbatch, so pin the sbatch too."""
        src = SBATCH.read_text()
        self.assertIn('ZBLT_CLOSURE_FAIL mode=$SM@${CTX}r[0-9]', src)
        self.assertIn('ZBLT_CLOSURE_FAIL mode=$SM@${CTX}rw', src)
        # the ambiguous pattern (no round qualifier, immediately closed by a quote)
        # must be gone
        self.assertNotIn('ZBLT_CLOSURE_FAIL mode=$SM@${CTX}r"', src)


class TestD2SpreadIsNotAnIdentity(unittest.TestCase):
    """`rep_spread` is max/min ACROSS REPS: undefined at n_reps=1, and never the
    same name as the within-cell block spread."""

    def test_rep_spread_is_none_at_one_rep(self):
        cells = {(256, "full", 1): dict(status="OK", per_mamba=0.5546)}
        self.assertIsNone(p5.rep_spread(cells, 256, "full"))

    def test_rep_spread_is_a_number_at_two_reps(self):
        cells = {(256, "full", 1): dict(status="OK", per_mamba=0.50),
                 (256, "full", 2): dict(status="OK", per_mamba=0.55)}
        self.assertAlmostEqual(p5.rep_spread(cells, 256, "full"), 1.1)

    def test_block_spread_is_a_separate_field(self):
        recs = [block(256, "full", per_mamba=v) for v in (0.50, 0.55, 0.60)]
        s = p5.summarize(recs, 9, 54)
        self.assertIn("blk_spread_pct", s)
        self.assertNotIn("rep_spread", s)
        self.assertAlmostEqual(s["blk_spread_pct"], (0.60 - 0.50) / 0.55 * 100.0)

    def test_the_two_quantities_disagree_on_the_same_cell(self):
        """A cell with block-to-block scatter and a single rep: the identity-valued
        quantity must not be able to masquerade as the informative one."""
        cells = {(256, "full", 1): dict(status="OK", per_mamba=0.55)}
        recs = [block(256, "full", per_mamba=v) for v in (0.50, 0.55, 0.60)]
        self.assertIsNone(p5.rep_spread(cells, 256, "full"))
        self.assertGreater(p5.summarize(recs, 9, 54)["blk_spread_pct"], 15.0)


class TestD3BatchMixtureIsVisible(unittest.TestCase):
    """A cell's realised batch can be a mixture; the label must carry it."""

    MIX = [block(16384, "full", blk=i, bs=19) for i in range(1, 7)] + \
          [block(16384, "full", blk=i, bs=13) for i in range(7, 15)]

    def test_mode_alone_hides_the_mixture(self):
        mix = p5.batch_mixture(self.MIX)
        self.assertEqual(mix["bs_mode"], 13)       # what the old label printed
        self.assertTrue(mix["mixed"])              # what it hid
        self.assertEqual(mix["bs_counts"], {19: 6, 13: 8})
        self.assertEqual(mix["bs_med"], 13)

    def test_bs_set_is_rendered_with_counts(self):
        self.assertEqual(p5.fmt_bs_set(p5.batch_mixture(self.MIX)["bs_counts"]),
                         "19x6,13x8")

    def test_uniform_cell_reports_not_mixed(self):
        mix = p5.batch_mixture([block(256, "full", blk=i) for i in range(1, 7)])
        self.assertFalse(mix["mixed"])
        self.assertEqual(mix["bs_counts"], {32: 6})

    def test_harness_cell_line_emits_the_mixture(self):
        src = SBATCH.read_text()
        self.assertIn("bs_mode=", src)
        self.assertIn("bs_set=", src)
        self.assertIn("bs_med=", src)
        # a bare single-value batch field must not be the only one emitted
        self.assertNotIn("bs={statistics.mode(", src)


class TestD4GateNConditionalOnRealisedBatch(unittest.TestCase):
    """GATE N is scored only within a fixed realised bs, and reports UNSCOREABLE
    when no comparable pair exists."""

    SMS = ["full"]

    def test_unscoreable_when_each_bs_appears_at_one_ctx_only(self):
        sub = {(16384, "full", 19): dict(per_mamba=0.1753, n=6),
               (16384, "full", 13): dict(per_mamba=0.1640, n=8)}
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)[0]
        self.assertEqual(res["verdict"], "UNSCOREABLE")
        self.assertTrue(all(g["verdict"] == "UNSCOREABLE" for g in res["groups"]))
        self.assertTrue(all(g["ratio"] is None for g in res["groups"]))
        self.assertEqual(p5.combine_gate_n([res]), "UNSCOREABLE")

    def test_unscoreable_is_not_pass_and_not_fail(self):
        sub = {(16384, "full", 19): dict(per_mamba=0.1753, n=6)}
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)
        self.assertNotIn(p5.combine_gate_n(res), ("PASS", "FAIL"))

    def test_batch_effect_alone_cannot_trip_the_gate(self):
        """The D4 confound, reproduced with batch-1's own numbers: per_mamba at
        ctx16384 is 0.1753 @bs19 vs 0.1640 @bs13 (1.069x, and 1.180x at sm8).
        Scored as written that is a cross-cell ratio; scored conditionally there
        is no comparison at all."""
        sub = {(16384, "full", 19): dict(per_mamba=0.6994, n=6),   # sm8 numbers
               (16384, "full", 13): dict(per_mamba=0.5930, n=8)}
        naive = max(r["per_mamba"] for r in sub.values()) / \
            min(r["per_mamba"] for r in sub.values())
        self.assertGreater(naive, 1.15)            # would FAIL as specified
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)[0]
        self.assertEqual(res["verdict"], "UNSCOREABLE")   # is not scored at all

    def test_scores_within_the_same_bs_across_ctx(self):
        sub = {(256, "full", 32): dict(per_mamba=0.5546, n=6),
               (1024, "full", 32): dict(per_mamba=0.4623, n=6),
               (4096, "full", 32): dict(per_mamba=0.1995, n=6)}
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)[0]
        g = res["groups"][0]
        self.assertEqual(g["bs"], 32)
        self.assertAlmostEqual(g["ratio"], 0.5546 / 0.1995, places=3)
        self.assertEqual(g["verdict"], "VIOLATION")
        self.assertEqual(p5.combine_gate_n([res]), "FAIL")

    def test_clean_arm_passes_only_with_full_ctx_coverage(self):
        clean = {(256, "full", 32): dict(per_mamba=0.5305, n=6),
                 (1024, "full", 32): dict(per_mamba=0.5296, n=6),
                 (4096, "full", 32): dict(per_mamba=0.5312, n=6)}
        res = p5.gate_n_conditional(clean, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["verdict"], "OK")
        self.assertEqual(p5.combine_gate_n(res), "PASS")
        # add an uncomparable ctx column -> the gate no longer covers it
        clean[(16384, "full", 13)] = dict(per_mamba=0.5300, n=8)
        res = p5.gate_n_conditional(clean, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["verdict"], "OK_PARTIAL")
        self.assertEqual(res[0]["uncovered_ctx"], [16384])
        self.assertEqual(p5.combine_gate_n(res), "UNSCOREABLE")

    def test_stray_bs_at_an_already_covered_ctx_does_not_make_it_partial(self):
        sub = {(256, "full", 32): dict(per_mamba=0.5305, n=6),
               (1024, "full", 32): dict(per_mamba=0.5296, n=6),
               (256, "full", 31): dict(per_mamba=0.5290, n=1)}   # one odd block
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["verdict"], "OK")
        self.assertEqual(res[0]["uncovered_ctx"], [])

    def test_violation_dominates_partial_coverage(self):
        sub = {(256, "full", 32): dict(per_mamba=0.5546, n=6),
               (4096, "full", 32): dict(per_mamba=0.1995, n=6),
               (16384, "full", 13): dict(per_mamba=0.1640, n=8)}
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["verdict"], "VIOLATION")
        self.assertEqual(p5.combine_gate_n(res), "FAIL")

    def test_threshold_is_a_strict_upper_bound(self):
        sub = {(256, "full", 32): dict(per_mamba=1.0, n=6),
               (1024, "full", 32): dict(per_mamba=1.0 / 1.15, n=6)}
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["groups"][0]["verdict"], "OK")   # exactly 1.15 -> OK
        sub[(1024, "full", 32)] = dict(per_mamba=1.0 / 1.16, n=6)
        res = p5.gate_n_conditional(sub, "per_mamba", 1.15, self.SMS)
        self.assertEqual(res[0]["groups"][0]["verdict"], "VIOLATION")


class TestD4ComplementNegativeControl(unittest.TestCase):
    """Methodology gate #10: the same O(1) check runs on the complement buckets.
    Without it the gate cannot be told apart from estimator insensitivity."""

    def test_complement_buckets_are_derived_per_module(self):
        recs = [block(256, "full", b_mlp=1.35, b_other=5.4)]
        s = p5.summarize(recs, 9, 54)
        self.assertAlmostEqual(s["per_mlp"], 1.35 / 9)
        self.assertAlmostEqual(s["per_other"], 5.4 / 54)

    def test_ratio_is_invariant_to_the_per_module_divisor(self):
        """So the gate verdict cannot depend on the layer-count convention."""
        recs = [block(256, "full", b_mlp=1.35)]
        a = p5.summarize(recs, 9, 54)["per_mlp"]
        b = p5.summarize(recs, 3, 54)["per_mlp"]
        self.assertNotAlmostEqual(a, b)
        recs2 = [block(1024, "full", b_mlp=2.70)]
        a2 = p5.summarize(recs2, 9, 54)["per_mlp"]
        b2 = p5.summarize(recs2, 3, 54)["per_mlp"]
        self.assertAlmostEqual(a2 / a, b2 / b)

    def test_control_can_dissociate_from_the_gated_metric(self):
        """mlp clean while mamba fails -- the batch-1 pattern that separates
        'the mamba bucket is broken' from 'the estimator is dead'."""
        sub = {(256, "full", 32): dict(per_mamba=0.5546, per_mlp=0.1510, n=6),
               (4096, "full", 32): dict(per_mamba=0.1995, per_mlp=0.1517, n=6)}
        mamba = p5.gate_n_conditional(sub, "per_mamba", 1.15, ["full"])
        mlp = p5.gate_n_conditional(sub, "per_mlp", 1.15, ["full"])
        self.assertEqual(mamba[0]["verdict"], "VIOLATION")
        self.assertEqual(mlp[0]["verdict"], "OK")

    def test_control_can_fail_together_with_the_gated_metric(self):
        """`other` failing in lockstep is a cross-bucket attribution signature;
        the tool must be able to express it, not just the clean case."""
        sub = {(256, "full", 32): dict(per_mamba=0.5546, per_other=0.1037, n=6),
               (4096, "full", 32): dict(per_mamba=0.1995, per_other=0.0325, n=6)}
        mamba = p5.gate_n_conditional(sub, "per_mamba", 1.15, ["full"])
        other = p5.gate_n_conditional(sub, "per_other", 1.15, ["full"])
        self.assertEqual(mamba[0]["verdict"], "VIOLATION")
        self.assertEqual(other[0]["verdict"], "VIOLATION")


class TestD5HostBoundnessObservables(unittest.TestCase):
    """Values only.  The tool must NOT filter, drop or judge on host-boundness --
    any such cut has to be pre-registered (methodology gate #8)."""

    SUB = {(256, "full", 32): dict(fwd=43.978, closure=0.9454, unbkt=2.403, n=6),
           (256, "44", 32): dict(fwd=45.361, closure=0.9300, unbkt=3.178, n=6),
           (256, "8", 32): dict(fwd=92.960, closure=0.9918, unbkt=0.767, n=6)}

    def test_ratio_is_against_the_min_at_the_same_ctx_and_bs(self):
        rows = {(r["ctx"], r["sm"]): r for r in p5.host_boundness_rows(self.SUB)}
        self.assertAlmostEqual(rows[(256, "full")]["fwd_ratio"], 1.0)
        self.assertAlmostEqual(rows[(256, "8")]["fwd_ratio"], 92.960 / 43.978, places=6)
        self.assertAlmostEqual(rows[(256, "44")]["fwd_min"], 43.978)

    def test_floors_are_not_shared_across_different_realised_bs(self):
        sub = dict(self.SUB)
        sub[(256, "full", 13)] = dict(fwd=20.0, closure=0.99, unbkt=0.2, n=8)
        rows = {(r["ctx"], r["sm"], r["bs"]): r for r in p5.host_boundness_rows(sub)}
        self.assertAlmostEqual(rows[(256, "full", 32)]["fwd_min"], 43.978)
        self.assertAlmostEqual(rows[(256, "full", 13)]["fwd_min"], 20.0)

    def test_no_cell_is_dropped_and_no_cut_is_applied(self):
        rows = p5.host_boundness_rows(self.SUB)
        self.assertEqual(len(rows), len(self.SUB))
        for r in rows:
            self.assertNotIn("admissible", r)
            self.assertNotIn("excluded", r)

    def test_gate_verdicts_do_not_consume_host_boundness(self):
        """A gate that quietly excluded host-bound cells would be a design choice
        made from batch-1 data.  Assert the gate signature cannot see them."""
        import inspect

        params = inspect.signature(p5.gate_n_conditional).parameters
        self.assertEqual(list(params), ["subcells", "metric", "threshold", "sms"])
        src = inspect.getsource(p5.gate_n_conditional)
        for token in ("fwd", "host", "unbkt"):
            self.assertNotIn(token, src)


class TestParsingAndSteadyState(unittest.TestCase):
    """The parse path the gates stand on."""

    def test_block_line_roundtrip(self):
        rec = p5.parse_block(block_line(16384, "full", blk=0, bs=19))
        self.assertEqual((rec["ctx"], rec["sm"], rec["bs"], rec["blk"]), (16384, "full", 19, 0))
        self.assertAlmostEqual(rec["per_mamba"], 0.17571)
        self.assertAlmostEqual(rec["b_mlp"], 1.3549)

    def test_non_block_line_returns_none(self):
        self.assertIsNone(p5.parse_block("CELL ctx=256 rep=1 sm=full STATUS=OK"))

    def test_cell_line_parses_both_legacy_and_mixture_labels(self):
        legacy = ("CELL ctx=16384 rep=1 sm=full STATUS=OK n_blocks=14/16 bs=13 "
                  "closure_med=0.9850 closure_min=0.9842 fwd_ms=62.2717 "
                  "per_attn=5.47617 per_mamba=0.16431 per_attn_core=5.32498")
        kv = p5.parse_kv_line(legacy, "CELL")
        self.assertEqual(kv["bs"], "13")
        self.assertEqual(kv["STATUS"], "OK")
        new = legacy.replace("bs=13", "bs_mode=13 bs_set=19x6,13x8 bs_med=13.0")
        kv = p5.parse_kv_line(new, "CELL")
        self.assertEqual(kv["bs_set"], "19x6,13x8")
        self.assertNotIn("bs", kv)

    def test_steady_state_rule_matches_the_harness(self):
        recs = ([block(256, "full", rnd=1, blk=i) for i in range(4)] +
                [block(256, "full", rnd=2, blk=i) for i in range(4)])
        recs[5]["dirty"] = 1
        kept = p5.keep_steady(recs)
        self.assertEqual(len(kept), 5)                       # 8 - 2 first - 1 dirty
        self.assertTrue(all(r["blk"] != 0 for r in kept))

    def test_subcells_split_a_mixed_cell_by_realised_bs(self):
        blocks = ([block(16384, "full", blk=i, bs=19) for i in range(0, 4)] +
                  [block(16384, "full", blk=i, bs=13) for i in range(4, 8)])
        sub = p5.build_subcells(blocks, 9, 54)
        self.assertEqual(sorted(bs for _, _, bs in sub), [13, 19])
        self.assertEqual(sub[(16384, "full", 19)]["n"], 3)   # blk 0 dropped
        self.assertEqual(sub[(16384, "full", 13)]["n"], 4)

    def test_subcell_rep_spread_is_none_at_one_rep(self):
        blocks = [block(256, "full", blk=i) for i in range(1, 4)]
        sub = p5.build_subcells(blocks, 9, 54)
        self.assertIsNone(sub[(256, "full", 32)]["per_mamba_rep_spread"])
        self.assertEqual(sub[(256, "full", 32)]["n_reps"], 1)


class TestRawArtifactsUntouched(unittest.TestCase):
    """The analyzer re-interprets; it must never write into the results dir."""

    def test_analyzer_opens_files_read_only(self):
        import ast

        tree = ast.parse(ANALYZER.read_text())
        opens, mutators = 0, []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "open":
                opens += 1
                mode = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                self.assertIn(mode, (None, "r", "rt", "rb"),
                              f"open() with mode {mode!r}: the analyzer must not write")
            if name in ("write", "writelines", "remove", "unlink", "rename",
                        "replace", "mkdir", "makedirs", "rmtree", "copy", "truncate"):
                mutators.append(name)
        self.assertGreater(opens, 0)
        self.assertEqual(mutators, [], f"filesystem-mutating calls found: {mutators}")


if __name__ == "__main__":
    unittest.main()
