"""Campaign stage 0 (lambda*) pre-registration rev3: plan, analyzer, rule, runner.

Two rules-layer audits stand behind this file.

rev1 audit (`VERDICT_lambda0_rules_2026-09-13.md`, `NO-GO`, D1-D15)
  D2/N2-S1  an unsaturated cell's `achieved/offered` is the ARRIVAL REALIZATION
            factor 1/Ebar, not a saturation measure.
  D3/N2-S2  the ladder must be a deterministic function of a measured input.
  D4/S9     `UNRESOLVED` was structurally unreachable under a glob: a boot
            failure printed as `KNEE_NOT_BRACKETED`.
  D5/S10    no analyzer and no sbatch existed; the rule crashed with
            `KeyError: 'shape'` on every cell JSON in the repository.

rev2 audit (`VERDICT_lambda0_rev2_2026-09-13.md`, `NO-GO`, D16-D23) -- 13 of the
15 above were confirmed closed, and the kill cause was in rev2's OWN repair:
  D16/D17   `achieved/realized_offered` is `(N/(N-1))*span/(span+L)`, an
            IDENTITY in the run's timing, so an unsaturated cell reads
            `kappa`, not 1.  rev2 validated it only where the drain is
            negligible (out=96) and applied it where the drain dominates
            (out=512), which pushed every shape-A low rung's CEILING to
            0.91-0.95 -- under the 0.95 threshold it was compared against.
  D18       the anchored/fallback branch was an operator env var, and it flips
            a label.
  D19       the "at most one blind-spot rung" theorem was an arithmetic fact
            about constants, false for the real estimator.
  D20       `--random-range-ratio 1.0` is a NECESSARY CONDITION for the arrival
            replay, not just for length exactness.
  D21       an unevaluated forecast was being recorded as REFUTED.
  D22       the mutation harness reached one file; 5 of the auditor's 6
            independent mutations escaped.
  D23       a truncated run made a whole shape vanish from the label JSON.

rev4 audit (`VERDICT_lambda0_rev4_2026-09-14.md`, `NO-GO`, N1-N2)
  N1        condition (b) of the anchor predicate was an EFFECTIVE IDENTITY on
            live artifacts: the producer writes ONE global-max line into
            `I3_max_running_req.txt`, so "every value >= 48" meant "some cell
            touched 48" and the F5 disqualification path was dead.  rev5/F1
            recomputes it PER CELL from `I_log_offsets.txt` + `srv_warmup.log`.
  N2        that repair FLIPS the branch ANCHORED -> FALLBACK (I3a = 48 but
            I3b = 2), which changes ladders, seeds, budget and 8 of 25 labels.
  F5        `SELFTEST_OF` routed plan/analyze mutations away from the
            reachability map, and 11 of the auditor's 19 mutations escaped; two
            of them MOVE a label on the map the registration publishes.

★ Negative controls, because a repair test that also passes on the unrepaired
code certifies nothing (lesson #53):
  `test_d4_injection_fails_on_rev1_glob`        rev1's glob gets it WRONG.
  `test_rev2_low_rungs_are_rejected`            rev2's 170 s shape-A low rungs
                                                fail the new ceiling.
  `test_drain_identity_on_archived_cells`       the identity that killed rev2,
                                                re-derived from raw artifacts.
  `test_rev4_global_max_predicate_would_anchor` rev4's predicate ANCHORS on the
                                                same bytes rev5 refuses.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PREREG = TRACK_ROOT / "results" / "r2_eval" / "lambda0_prereg"
ARCHIVE = TRACK_ROOT / "results" / "longctx_conflict" / "probes" / "c_905835"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PREREG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PREREG))
    spec.loader.exec_module(mod)
    return mod


PLAN = _load("lambda0_plan")
LABEL = _load("lambda0_label")
ANALYZE = _load("lambda0_analyze")
LAMINF = _load("lambda0_lambda_inf")
CELLS = _load("lambda0_cells")
REACH = _load("lambda0_reachability")
MUT = _load("lambda0_mutation_check")
MEASURED_INSTR = (TRACK_ROOT / "results" / "r2_correctness" / "job_907959"
                  / "instrument")
PRODUCER = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness.sbatch"
PROJECT_ROOT = TRACK_ROOT.parents[1]


def _sbatch_block(text, start_needle, end_needle, drop=()):
    """The runner's OWN lines between two anchors, so a test can EXECUTE the
    shipped bytes instead of a paraphrase of them.

    ★Why extraction and not a rewritten fixture: the 5th rules-layer audit found
    W4' by running the real script, and a hand-copied excerpt is the identity
    trap this track keeps dying on (`CONSENSUS sec 3` item 18 / lesson 9) -- a
    gate that tests its own copy of the code certifies the copy.  `drop` deletes
    whole lines, which is how the reverted-repair mutants below are built.
    """
    lines = text.splitlines()
    starts = [k for k, l in enumerate(lines) if start_needle in l]
    if len(starts) != 1:
        raise AssertionError("anchor %r matched %d lines, expected 1"
                             % (start_needle, len(starts)))
    i = starts[0]
    j = next(k for k in range(i, len(lines)) if end_needle in lines[k])
    kept = [l for l in lines[i:j + 1] if not any(d in l for d in drop)]
    return "\n".join(kept) + "\n"


def _live_job_present():
    """★F1 reads two artifacts of job 907959.  `srv_warmup.log` is 3.5 MB and
    matched by `workspace/engine-port/.gitignore:30` (`*.log`), so it is NOT in
    the repository: on a fresh clone these legs skip, and the sbatch fails
    closed instead (`lambda0_lambda_inf.selftest` hard-fails without it)."""
    return (MEASURED_INSTR.exists()
            and (MEASURED_INSTR.parent / "srv_warmup.log").exists()
            and (MEASURED_INSTR / "I_log_offsets.txt").exists())


class TestShippedSelftests(unittest.TestCase):
    def test_plan_selftest(self):
        PLAN.selftest()

    def test_label_selftest(self):
        LABEL.selftest()

    def test_lambda_inf_predicate_selftest(self):
        LAMINF.selftest()

    def test_cells_renderer_selftest(self):
        CELLS.selftest()

    def test_reachability_selftest(self):
        REACH.selftest()


class TestDrainIdentity(unittest.TestCase):
    """D16: the fact that killed rev2, re-derived from the raw bench records."""

    def test_drain_identity_on_archived_cells(self):
        if not ARCHIVE.exists():
            self.skipTest("archived probe C job not present")
        import numpy as np
        nps = {"d16_r0_o96": 110, "d16_r1_o96": 170, "d16_r2_o96": 170,
               "d44_r0_o96": 80, "d44_r1_o96": 120, "d44_r2_o96": 130,
               "d92_r0_o96": 40, "d92_r1_o96": 40, "d92_r2_o96": 45}
        worst_dur, worst_id = 0.0, 0.0
        for name, n in nps.items():
            d = json.loads((ARCHIVE / f"bench_{name}.jsonl").read_text()
                           .strip().split("\n")[0])
            np.random.seed(41)
            iv = np.random.exponential(1.0 / d["request_rate"], size=n - 1)
            span = float(iv.sum())
            ratio = d["request_throughput"] / ((n - 1) / span)
            # (a) the estimator IS (N/(N-1))*span/duration
            worst_id = max(worst_id,
                           abs(ratio - (n / (n - 1)) * span / d["duration"]))
            # (b) duration IS the last completion, so L = duration - span is the
            #     tail drain and NOT a queueing shortfall
            arr = np.concatenate([[0.0], np.cumsum(iv)])
            e2e = [d["ttfts"][i] + sum(d["itls"][i]) for i in range(len(d["ttfts"]))]
            worst_dur = max(worst_dur, abs(max(a + e for a, e in zip(arr, e2e))
                                           - d["duration"]))
        self.assertLess(worst_id, 1e-12, "the estimator is not that identity")
        self.assertLess(worst_dur, 0.30, "duration is not the last completion")

    def test_kappa_is_a_ceiling_not_one(self):
        """An unsaturated shape-A rung at rev2's 170 s window cannot reach the
        low-side threshold: that is the whole kill cause, in one assertion."""
        c = PLAN.cell_at("A", 0.40 * 2.10, 170.0, 400)
        self.assertLess(c["kappa_pred"], PLAN.ACH_HI)
        self.assertGreater(c["kappa_pred"], PLAN.ACH_LO)      # in the blind spot

    def test_rev2_low_rungs_are_rejected(self):
        """NEGATIVE CONTROL for D16/D17: `size_rung` must refuse to certify the
        design rev3 replaces.  If this passes, the repair does nothing."""
        for lam in (2.10, 2.51, 1.08, 3.91):
            for m in (0.40, 0.70):                 # rev2's MULT[A][0], [1]
                r = PLAN.size_rung("A", m * lam, extendable=False)
                self.assertFalse(r["low_side_candidate"], (lam, m, r))

    def test_window_extension_is_what_fixes_it(self):
        """D17: the same rung, only the window changed, becomes certifiable."""
        x = 0.25 * 2.10
        short = PLAN.size_rung("A", x, extendable=False)
        long_ = PLAN.size_rung("A", x, extendable=True)
        self.assertFalse(short["low_side_candidate"])
        self.assertTrue(long_["low_side_candidate"])
        self.assertGreater(long_["T_measure_s"], PLAN.T_SHORT_S)
        self.assertGreaterEqual(long_["kappa_pred"], PLAN.ACH_HI + 0.02)


class TestPlanIsDeterministic(unittest.TestCase):
    """D3: nothing in the measurement plan may be chosen after seeing data."""

    def test_plan_is_a_pure_function_of_lambda_inf(self):
        a, b = PLAN.plan(2.1, 0.675), PLAN.plan(2.1, 0.675)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_every_registered_scenario_has_a_low_side_candidate(self):
        for title, la, lb in PLAN.REGISTERED_GRID:
            p = PLAN.plan(la, lb)
            for shape in ("A", "B"):
                d = p["shapes"][shape]
                self.assertGreaterEqual(d["n_low_side_candidates"], 1, (title, shape))
                for c in d["cells"]:
                    if c["low_side_candidate"]:
                        self.assertGreaterEqual(c["kappa_pred"], 0.97, (title, c))

    def test_bracket_window_is_the_audited_closed_form(self):
        p = PLAN.plan(2.1, 0.675)
        for shape in ("A", "B"):
            d = p["shapes"][shape]
            xs = [c["offered_realized_pred"] for c in d["cells"]]
            self.assertAlmostEqual(d["window_realized"][0], 0.95 * min(xs), places=12)
            self.assertAlmostEqual(d["window_realized"][1], 0.90 * max(xs), places=12)

    def test_fallback_reproduces_the_audit_recommended_windows(self):
        """D6: the registered fallback is the rev1 audit's literal ladder and
        must reproduce its published windows exactly."""
        lad = {k: tuple(v) for k, v in PLAN.FALLBACK_LADDER.items()}
        p = PLAN.plan(2.10, 0.675, ladders=lad)
        self.assertAlmostEqual(p["shapes"]["A"]["window_nominal"][0], 1.045, places=6)
        self.assertAlmostEqual(p["shapes"]["A"]["window_nominal"][1], 7.20, places=6)
        self.assertAlmostEqual(p["shapes"]["B"]["window_nominal"][0], 0.4275, places=6)
        self.assertAlmostEqual(p["shapes"]["B"]["window_nominal"][1], 1.035, places=6)
        # ...and the fallback must ALSO clear the ceiling; rev2's did not.
        self.assertGreaterEqual(p["shapes"]["A"]["n_low_side_candidates"], 1)

    def test_lower_rung_does_not_assume_lambda_inf_bounds_from_below(self):
        """The adjacent newpair pre-registration (D7)/NP-3' registers that I3
        fixes only the UPPER end of the ladder.  rev2 put x_min at
        0.40*lambda_inf; rev3 lowers it and adds a data-checked branch."""
        self.assertEqual(PLAN.MULT["A"][0], 0.25)
        p = PLAN.plan(2.10, 0.675)
        self.assertLessEqual(p["shapes"]["A"]["coverage_band_ratio"][0], 0.24)
        # LADDER_TOO_HIGH is the registered response, with a direction.
        too_high = LABEL.rule({"a": [LABEL._cell(r, 0.70) for r in (4.0, 8.0)]})["a"]
        self.assertEqual(too_high["verdict"], "LADDER_TOO_HIGH")
        self.assertEqual(too_high["redesign_factor"], 1.0 / LABEL.REDESIGN_FACTOR)

    def test_eleven_boots_and_budget(self):
        """E5: a rung above capacity finishes in N/lambda*, not N/x, so the
        registered request must cover the BOTTOM of the coverage band."""
        for title, la, lb in PLAN.REGISTERED_GRID:
            p = PLAN.plan(la, lb)
            self.assertEqual(p["n_boot"], 11, title)
            worst = max(PLAN.budget(p, r)["total_gpu_h"]
                        for r in PLAN.BUDGET_RATIO_GRID)
            self.assertLessEqual(worst, PLAN.REQUESTED_BUDGET_GPU_H, (title, worst))
            # ...and the 0.25x corner really is the expensive one, i.e. the
            # repair is load bearing rather than decorative.
            self.assertGreater(PLAN.budget(p, 0.25)["total_gpu_h"],
                               1.35 * PLAN.budget(p, 1.0)["total_gpu_h"], title)

    def test_measured_lambda_inf_is_registered(self):
        titles = [t for t, _, _ in PLAN.REGISTERED_GRID]
        self.assertTrue(any("907959" in t for t in titles), titles)
        self.assertEqual(PLAN.LAMBDA_INF_MEASURED, (3.0939, 0.6956))


class TestArrivalRealization(unittest.TestCase):
    """D2 and D20."""

    def test_scalar_and_array_exponential_streams_agree(self):
        import numpy as np
        np.random.seed(41)
        one = [np.random.exponential(1 / 0.86) for _ in range(11)]
        np.random.seed(41)
        arr = list(np.random.exponential(1 / 0.86, size=11))
        for x, y in zip(one, arr):
            self.assertEqual(x, y)

    def test_range_ratio_one_is_necessary_for_the_replay(self):
        """D20: `benchmark/datasets/common.py:56-64` draws randint twice per
        run.  Only a degenerate range consumes zero stream, so rr=1.0 is a
        precondition of the replay -- rev2 called it a length-exactness flag."""
        import numpy as np

        def after(rr):
            np.random.seed(1376)
            for full in (256, 512):
                np.random.randint(max(int(full * rr), 1), full + 1, size=140)
            return np.random.exponential(1.0, size=3)
        np.random.seed(1376)
        clean = np.random.exponential(1.0, size=3)
        self.assertTrue(np.allclose(after(1.0), clean))
        self.assertFalse(np.allclose(after(0.9), clean))

    def test_realized_denominator_removes_the_artifact_on_archived_cells(self):
        if not ARCHIVE.exists():
            self.skipTest("archived probe C job not present")
        for bench, npr, nominal in (("bench_d16_r0_o96.jsonl", 110, 1.1700),
                                    ("bench_d44_r0_o96.jsonl", 80, 1.1422),
                                    ("bench_d92_r0_o96.jsonl", 40, 1.1886)):
            rec = ANALYZE.analyze(ARCHIVE / bench, "B", bench, npr, 41)
            self.assertAlmostEqual(rec["achieved_over_offered"], nominal, places=3)
            self.assertAlmostEqual(rec["achieved_over_realized"], 1.0, delta=0.02,
                                   msg=bench)

    def test_analyzer_records_the_drain_decomposition(self):
        if not ARCHIVE.exists():
            self.skipTest("archived probe C job not present")
        rec = ANALYZE.analyze(ARCHIVE / "bench_d44_r2_o96.jsonl", "B", "b_r2", 130, 41)
        self.assertEqual(rec["shape"], "B")
        self.assertTrue(rec["input_len_exact"] and rec["output_len_exact"])
        self.assertAlmostEqual(rec["span_s"] + rec["drain_s"], rec["duration_s"],
                               places=9)
        self.assertFalse(rec["ebar_guard_ok"])   # seed 41 was never registered here


class TestRuleReproducesTheAuditedVerdicts(unittest.TestCase):
    def test_probe_c_lambda_star_bit_for_bit(self):
        if not ARCHIVE.exists():
            self.skipTest("archived probe C job not present")
        got = LABEL.rule(LABEL._probe_c_records(), enforce_guards=False)
        self.assertEqual(got["d16"]["lambda_star"], 0.9331188685063582)
        self.assertEqual(got["d44"]["lambda_star"], 0.675302644015995)
        self.assertEqual(got["d92"]["lambda_star"], 0.18668462822130108)
        for arm in ("d16", "d44", "d92"):
            self.assertEqual(got[arm]["verdict"], "KNEE_BRACKETED")

    def test_guards_reject_the_unregistered_seed(self):
        if not ARCHIVE.exists():
            self.skipTest("archived probe C job not present")
        got = LABEL.rule(LABEL._probe_c_records())        # guards armed
        self.assertEqual(got["d16"]["verdict"], "UNRESOLVED")

    def test_low_side_requires_a_registered_candidate(self):
        """D16 in the rule: a cell whose CEILING sits under the threshold may
        not supply the low side however good its reading."""
        bad = [LABEL._cell(1.0, 0.99, cand=False, kappa=0.93),
               LABEL._cell(4.0, 0.50, achieved=2.0, cand=False)]
        self.assertNotEqual(LABEL.rule({"a": bad})["a"]["verdict"], "KNEE_BRACKETED")
        good = [LABEL._cell(1.0, 0.99, cand=True),
                LABEL._cell(4.0, 0.50, achieved=2.0, cand=False)]
        self.assertEqual(LABEL.rule({"a": good})["a"]["verdict"], "KNEE_BRACKETED")

    def test_unevaluated_seed_repeat_is_unresolved(self):
        """D21: rev2 recorded a forecast it never evaluated as REFUTED."""
        lad = [LABEL._cell(r, 1.00) for r in (1.0, 2.0)]
        res = LABEL.rule({"a": lad}, repeats={"a": LABEL._cell(2.0, 1.00)})["a"]
        self.assertIsNone(res["lambda_star"])
        self.assertEqual(res["seed_repeat"]["verdict"], "UNRESOLVED")

    def test_blind_spot_cell_is_neither_side(self):
        """D19 in the rule: the withdrawn theorem is replaced by behaviour."""
        x = [LABEL._cell(1.0, 0.99, achieved=0.99),
             LABEL._cell(2.0, 0.93, achieved=1.86, cand=False),
             LABEL._cell(4.0, 0.45, achieved=1.80, cand=False)]
        r = LABEL.rule({"a": x})["a"]
        self.assertEqual(r["verdict"], "KNEE_BRACKETED")
        self.assertAlmostEqual(r["lambda_star"], 1.80)      # not 1.86
        self.assertEqual(r["blind_spot_cells"], [2.0])


class TestEveryVerdictIsReachable(unittest.TestCase):
    """E2 -- the generalisation of the attainability pre-computation.

    Both kill causes were verdicts with empty domains that nothing in the
    registration noticed: rev2 registered a low-side ceiling it could not
    reach; rev3 registered a bracket whose high-side evidence its own guard
    discarded.  rev4 asserts the whole map."""

    def test_all_four_verdicts_have_non_empty_domains(self):
        _, seen, _codes = REACH.table()
        for shape in ("A", "B"):
            for verdict in ("KNEE_BRACKETED", "KNEE_NOT_BRACKETED",
                            "LADDER_TOO_HIGH", "LADDER_TOO_LOW"):
                self.assertTrue(seen[shape].get(verdict),
                                (shape, verdict, sorted(seen[shape])))

    def test_shape_b_brackets_at_the_measured_anchor(self):
        """The rev3 kill cause, at the COUNTERFACTUAL anchor: lambda_inf(B) =
        0.6956 (job 907959) sits inside the band rev3 threw away wholesale.

        ★rev5/F1: 0.6956 is NOT what will be served -- the per-cell recount
        disqualifies it (cell I3b peaked at 2 admitted requests, newpair F5), so
        the run takes the FALLBACK ladder.  This leg keeps covering the rev3 kill
        cause at that point of the map; it makes no claim about the anchor's
        status (audit R4C-2 forbids citing 0.6956 as a saturation throughput).
        """
        r = REACH.emit(*PLAN.LAMBDA_INF_MEASURED, 1.00)["B"]
        self.assertEqual(r["verdict"], "KNEE_BRACKETED", r)
        self.assertLess(abs(r["rel_err"]), 0.05, r)

    def test_rev3_rule_still_fails_here(self):
        """NEGATIVE CONTROL (lesson 53): with the drain guard back in R0 the
        same input must return UNRESOLVED, or E1 changed nothing."""
        # ★`_load` builds a fresh module object per call, while
        # `lambda0_reachability` imported `rule` from the copy that landed in
        # sys.modules.  Patching the wrong one silently does nothing and this
        # negative control would pass for the wrong reason.
        mod = sys.modules["lambda0_label"]
        self.assertIs(mod.rule, REACH.rule, "patching the wrong module object")
        saved = mod._usable_low, mod._cell_invalid
        base = saved[1]

        def rev3_usable_low(c):
            return bool(c.get("low_side_candidate", False))

        def rev3_cell_invalid(c):
            r = base(c)
            if r:
                return r
            if c.get("low_side_candidate", False) and not c.get("drain_model_ok", False):
                return "rev3: drain past prediction voids the SHAPE"
            return None
        mod._usable_low, mod._cell_invalid = rev3_usable_low, rev3_cell_invalid
        try:
            r = REACH.emit(*PLAN.LAMBDA_INF_MEASURED, 1.00)["B"]
        finally:
            mod._usable_low, mod._cell_invalid = saved
        self.assertEqual(r["verdict"], "UNRESOLVED", r)

    def test_drain_guard_no_longer_voids_the_shape(self):
        """E1 stated as behaviour: a blown drain costs the cell its LOW-side
        role and nothing else; the HIGH side carries no drain condition."""
        lowbad = [LABEL._cell(1.0, 0.99, drain_ok=False),
                  LABEL._cell(4.0, 0.50, achieved=2.0, cand=False)]
        r = LABEL.rule({"a": lowbad})["a"]
        self.assertNotEqual(r["verdict"], "UNRESOLVED")
        self.assertEqual(r["n_usable_low_side"], 0)
        high = [LABEL._cell(1.0, 0.99),
                LABEL._cell(4.0, 0.50, achieved=2.0, drain_ok=False)]
        self.assertEqual(LABEL.rule({"a": high})["a"]["verdict"], "KNEE_BRACKETED")

    def test_r0_is_measurement_validity_only(self):
        src = (PREREG / "lambda0_label.py").read_text()
        r0 = src.split("def _cell_invalid", 1)[1].split("def _usable_low", 1)[0]
        self.assertNotIn("drain_model_ok", r0)
        self.assertIn("ebar_guard_ok", r0)
        self.assertIn("range_ratio_ok", r0)


class TestUnresolvedIsReachable(unittest.TestCase):
    """D4 / rev1 audit S9, with the negative control the audit asked for."""

    @staticmethod
    def _ladder_dir(td):
        recs = [("a_r0", 1.0, 1.00, 1.00, True), ("a_r1", 2.0, 0.99, 1.98, True),
                ("a_r2", 4.0, 0.50, 2.00, False)]
        for name, off, ratio, ach, cand in recs:
            Path(td, f"cell_{name}.json").write_text(json.dumps({
                "shape": "A", "label": name, "offered_rate": off,
                "achieved_rate": ach, "achieved_over_realized": ratio,
                "achieved_over_offered": ratio, "low_side_candidate": cand,
                "kappa_pred": 0.98, "ebar_guard_ok": True,
                "range_ratio_ok": True, "drain_model_ok": True}))
        return [r[0] for r in recs]

    def test_missing_top_cell_is_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            names = self._ladder_dir(td)
            os.remove(Path(td, "cell_a_r2.json"))
            cells, repeats, missing = LABEL.load(td, names)
            self.assertEqual(LABEL.rule(cells, repeats)["A"]["verdict"], "UNRESOLVED")
            self.assertEqual(missing, ["a_r2"])

    def test_d4_injection_fails_on_rev1_glob(self):
        """NEGATIVE CONTROL: rev1 globbed the directory, so a missing cell
        vanished and the survivors were judged as if they were the ladder."""
        with tempfile.TemporaryDirectory() as td:
            self._ladder_dir(td)
            os.remove(Path(td, "cell_a_r2.json"))
            rev1 = {}
            for f in sorted(Path(td).glob("cell_*.json")):
                rec = json.loads(f.read_text())
                rev1.setdefault(rec["shape"], []).append(rec)
            v = LABEL.rule(rev1, {})["A"]["verdict"]
            self.assertNotEqual(v, "UNRESOLVED")
            self.assertEqual(v, "LADDER_TOO_LOW")

    def test_shipped_mutation_entry_point(self):
        self.assertEqual(LABEL.mutation_missing_cell(verbose=False), "UNRESOLVED")


class TestLambdaInfPredicate(unittest.TestCase):
    """D18 + E6: the anchored/fallback branch is a predicate over files, and
    the artifacts it read are recorded by digest."""

    def test_measured_artifacts_select_fallback_after_the_f1_repair(self):
        """★N2, the flip rev5 registers BEFORE the run.

        rev4 answered ANCHORED with lambda_inf = (3.0939, 0.6956) on exactly
        these bytes.  The per-cell recount says cell I3b peaked at 2 admitted
        requests, not 48, so neither rate may be cited as a saturation
        throughput (newpair F5) and the ladder falls back to the literals.
        """
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        r = LAMINF.decide(MEASURED_INSTR)
        self.assertEqual(r["mode"], "FALLBACK", r)
        self.assertEqual(r["lambda_inf"], {}, r)
        why = [x for x in r["disqualifications"] if "I3b_shapeB" in x]
        self.assertEqual(len(why), 1, r["disqualifications"])
        self.assertIn("F5", why[0])
        # I3a DID reach 48 and I2 is not judged, so neither may be a cause.
        self.assertFalse([x for x in r["disqualifications"]
                          if x.startswith("I3a_shapeA:") or x.startswith("I2")], r)

    def test_the_registered_per_cell_recount_matches_the_live_bytes(self):
        """★F1: the 3 cells x 3 conventions table the pre-registration prints,
        recomputed here from the producer's own artifacts."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        rc = LAMINF.recount(MEASURED_INSTR)
        self.assertEqual(rc["errors"], [])
        self.assertEqual(rc["server_log_lines"], 18382)     # `wc -l`
        self.assertEqual(rc["server_log_global_max"], 48)   # the file's value
        want = {"I2": (85, 4226, 1), "I3a_shapeA": (4227, 8434, 48),
                "I3b_shapeB": (8435, 18380, 2)}
        self.assertEqual(set(rc["cells"]), set(want))
        for stem, (lo, hi, mx) in want.items():
            m = rc["cells"][stem]
            self.assertEqual(m["interval_1based"], [lo, hi], stem)
            # ★all three registered conventions give the SAME answer, so the
            # F5 verdict does not depend on which one is privileged (gate #113)
            for conv in LAMINF.CONVENTIONS:
                self.assertEqual(m[conv], mx, (stem, conv))

    def test_only_the_two_cited_cells_are_judged(self):
        """I2 runs at concurrency 1 by construction, so requiring 48 of it
        would be nonsense; it is reported and never judged.  Both choices give
        FALLBACK here, and this pins which one is registered."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        rc = LAMINF.recount(MEASURED_INSTR)
        self.assertEqual(rc["judged_cells"], ["I3a_shapeA", "I3b_shapeB"])
        self.assertEqual([Path(f).stem for f in
                          sorted(LAMINF.CELL_FILES.values())],
                         rc["judged_cells"])

    def test_wc_l_line_convention_is_load_bearing(self):
        """★the free surface F1 creates, measured rather than argued.

        The offsets come from `wc -l`, which counts newline bytes.  Python's
        `splitlines()` also breaks on the 116 bare CRs this log carries, which
        shifts every offset by 116 lines and makes cell I3b read 11 instead of
        2.  Both fail F5 -- so the BRANCH is robust -- but only one is the
        producer's numbering, and the registered table prints 2.
        """
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        raw = (MEASURED_INSTR.parent / LAMINF.SERVER_LOG).read_bytes()
        wc = LAMINF.log_lines(raw)
        sl = [l.encode() for l in raw.decode("utf-8", "replace").splitlines()]
        self.assertEqual(len(wc), 18382)
        self.assertEqual(len(sl), 18498)
        self.assertEqual(raw.count(b"\r"), 116)
        off = LAMINF.read_offsets(
            (MEASURED_INSTR / LAMINF.OFFSETS_FILE).read_text())
        s, e = off["I3b_shapeB_start"], off["I3b_shapeB_end"]
        self.assertEqual(LAMINF.cell_running_req(wc, s, e)["all_lines"], 2)
        self.assertEqual(LAMINF.cell_running_req(sl, s, e)["all_lines"], 11)

    def test_interval_boundary_cannot_be_discriminated_on_this_artifact(self):
        """★registered honestly: the three boundary lines (84, 4226, 8434,
        18380) carry no `#running-req`, so start-exclusive vs start-inclusive
        changes NO value on job 907959.  The convention is therefore pinned by
        the shipped selftest's real-log-slice leg, not by this artifact."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        lines = LAMINF.log_lines(
            (MEASURED_INSTR.parent / LAMINF.SERVER_LOG).read_bytes())
        off = LAMINF.read_offsets(
            (MEASURED_INSTR / LAMINF.OFFSETS_FILE).read_text())
        for stem in ("I2", "I3a_shapeA", "I3b_shapeB"):
            s, e = off[stem + "_start"], off[stem + "_end"]
            reg = LAMINF.cell_running_req(lines, s, e)
            inc = LAMINF.cell_running_req(lines, s - 1, e)
            for conv in LAMINF.CONVENTIONS:
                self.assertEqual(reg[conv], inc[conv], (stem, conv))
            # ...and the line that WOULD be pulled in is the reason why
            self.assertNotIn(b"#running-req", lines[s - 1], stem)

    def test_rev4_global_max_predicate_would_anchor(self):
        """★NEGATIVE CONTROL (lesson 53) for F1 itself.  rev4's condition (b)
        -- `min(every integer in I3_max_running_req.txt) >= 48` -- is computed
        here on the same bytes and ANCHORS.  If this ever stops anchoring, the
        repair is no longer repairing anything."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        vals = LAMINF._last_int_per_line(
            (MEASURED_INSTR / LAMINF.RUNNING_FILE).read_text())
        self.assertEqual(vals, [48])
        self.assertGreaterEqual(min(vals), LAMINF.MIN_RUNNING_REQ)  # rev4: ANCHOR
        self.assertEqual(LAMINF.decide(MEASURED_INSTR)["mode"], "FALLBACK")

    def test_the_legacy_file_can_veto_but_never_grant(self):
        """`I3_max_running_req.txt` is provenance + a cross-check after F1.  It
        cannot create an anchor (cell B still reads 2), and if it disagrees with
        our recount of the whole log the artifacts are not one run."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        r = LAMINF.decide(MEASURED_INSTR)
        self.assertEqual(r["running_req_file_value"], 48)
        self.assertEqual(r["mode"], "FALLBACK")          # 48 grants nothing

    def test_e6_provenance_is_recorded(self):
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        r = LAMINF.decide(MEASURED_INSTR)
        self.assertTrue(Path(r["instrument_dir"]).is_absolute())
        # ★F1 added two inputs: the offsets file and the server log itself.
        self.assertEqual(set(r["inputs"]),
                         set(LAMINF.CELL_FILES.values())
                         | {LAMINF.RUNNING_FILE, LAMINF.OFFSETS_FILE,
                            LAMINF.SERVER_LOG})
        for fn, meta in r["inputs"].items():
            self.assertEqual(len(meta["sha256"]), 64, fn)
            self.assertTrue(Path(meta["abspath"]).is_absolute(), fn)

    def test_sbatch_defaults_to_the_measured_artifacts(self):
        text = (PREREG / "lambda0.sbatch").read_text()
        self.assertIn("job_907959/instrument", text)
        self.assertIn("LAMBDA_INF_DECISION.json", text)

    def test_missing_artifacts_force_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(LAMINF.decide(td)["mode"], "FALLBACK")

    def test_f5_is_decided_per_cell_on_real_log_slices(self):
        """The fixtures are SLICES OF THE PRODUCER'S OWN LOG (rev4 died of a
        fixture format the producer never writes).  Same summary file "48" in
        every case; only the cell's own interval decides."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        with tempfile.TemporaryDirectory() as td:
            ok = {"I3a_shapeA": LAMINF.W_OK_A, "I3b_shapeB": LAMINF.W_OK_B}
            d = LAMINF._fixture(td, ok)
            self.assertEqual(LAMINF.decide(d)["mode"], "ANCHORED")
            # 35 < 48 in the cell's OWN window -> disqualified (auditor Z5
            # lowers the threshold to 32, which this refuses)
            d = LAMINF._fixture(td, {"I3a_shapeA": LAMINF.W_OK_A,
                                     "I3b_shapeB": LAMINF.W_LOW})
            r = LAMINF.decide(d)
            self.assertEqual(r["mode"], "FALLBACK")
            self.assertTrue(any("F5" in x and "35" in x
                                for x in r["disqualifications"]), r)
            # and the interval convention is load bearing on a REAL boundary
            # line: 7903 reports 48, 7904.. peaks at 43.
            d = LAMINF._fixture(td, {"I3a_shapeA": LAMINF.W_OK_A,
                                     "I3b_shapeB": LAMINF.W_BOUND})
            self.assertEqual(LAMINF.decide(d)["mode"], "FALLBACK")
            m = LAMINF.recount(d)["cells"]["I3b_shapeB"]
            self.assertEqual([m[c] for c in LAMINF.CONVENTIONS], [43, 43, 43])
            d = LAMINF._fixture(td, {"I3a_shapeA": LAMINF.W_OK_A,
                                     "I3b_shapeB": (LAMINF.W_BOUND[0] - 1,
                                                    LAMINF.W_BOUND[1])})
            m = LAMINF.recount(d)["cells"]["I3b_shapeB"]
            self.assertEqual([m[c] for c in LAMINF.CONVENTIONS], [48, 48, 43])
            r = LAMINF.decide(d)
            self.assertEqual(r["mode"], "FALLBACK")   # conventions disagree
            self.assertTrue(any("DISAGREE" in x for x in r["disqualifications"]))

    def test_missing_recount_inputs_force_fallback(self):
        """Fail closed: without the offsets file or the server log, condition
        (b) cannot be evaluated at all, so the anchor is refused."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        with tempfile.TemporaryDirectory() as td:
            ok = {"I3a_shapeA": LAMINF.W_OK_A, "I3b_shapeB": LAMINF.W_OK_B}
            for kw, needle in ((dict(offsets=False), LAMINF.OFFSETS_FILE),
                               (dict(log=False), LAMINF.SERVER_LOG),
                               (dict(running=None), LAMINF.RUNNING_FILE)):
                d = LAMINF._fixture(td, ok, **kw)
                r = LAMINF.decide(d)
                self.assertEqual(r["mode"], "FALLBACK", (kw, r))
                self.assertTrue(any(needle in x for x in r["disqualifications"]),
                                (kw, r))


class TestMutationHarness(unittest.TestCase):
    """D9 + D22: every registered mutation of the whole decision path must
    break a selftest, and the harness must reach the analyzer and the plan."""

    def test_no_escapes(self):
        r = subprocess.run(
            [sys.executable, str(PREREG / "lambda0_mutation_check.py")],
            capture_output=True, text=True, timeout=3600)
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-2000:])
        harness = _load("lambda0_mutation_check")
        for entry in harness.CONTROL_ENTRIES:
            self.assertIn(f"CONTROL unmutated {entry:<24s}(temp-dir copy) PASSES",
                          r.stdout, entry)
        self.assertNotIn("ESCAPE", r.stdout)
        self.assertNotIn("SKIPPED", r.stdout)
        self.assertIn("escapes: none", r.stdout)

    def test_harness_covers_the_whole_decision_path(self):
        """D22 + E3.  ★This assertion used to read "exactly these three
        modules", which FROZE the blind spot the rev3 audit then walked through:
        `lambda0_cells.py` carries the low-side certification to the analyzer,
        was outside MODULES, and the auditor's Y5 (certify every rung) passed
        36/36 untouched while replicating rev3's kill cause into shape A."""
        harness = _load("lambda0_mutation_check")
        mods = {mod for mod, _, _ in harness.MUTATIONS.values()}
        self.assertIn("lambda0_cells.py", mods)
        self.assertTrue(mods.issubset(set(harness.MODULES)), mods)
        # every module the harness claims to cover must actually be mutated
        for m in harness.MODULES:
            self.assertIn(m, mods, f"{m} is in MODULES but never mutated")
        for name in ("X3", "X4", "X5", "Y1", "Y2", "Y4", "Y5", "Y6",
                     # ★rev5/F5-2: the rev4 auditor's escapes.  Z1 and Z4 each
                     # MOVE a label on the reachability map, Z5 is the F5
                     # threshold and Z19 the drain guard's boundary direction.
                     "Z1", "Z4", "Z5", "Z19",
                     # ★rev5/F1's own revert-mutants (lesson 53): F1a is
                     # literally rev4's global-max predicate.
                     "F1a", "F1b", "F1c"):
            self.assertTrue(any(k.startswith(name) for k in harness.MUTATIONS),
                            f"the auditors' {name} must be registered")

    def test_plan_and_analyze_mutations_reach_the_reachability_map(self):
        """★rev5/F5-1.  The rev4 audit's structural finding: `SELFTEST_OF` sent
        plan and analyze mutations to the plan / label selftests ONLY, so a
        mutation that moves a label on the published map escaped.  Z1 and Z4 are
        caught by the reachability entry and by nothing else."""
        harness = _load("lambda0_mutation_check")
        for mod in ("lambda0_plan.py", "lambda0_analyze.py"):
            self.assertIn("lambda0_reachability.py", harness.SELFTEST_OF[mod], mod)
        # every routed entry must be a real selftest of a module in MODULES
        for mod, entries in harness.SELFTEST_OF.items():
            self.assertIsInstance(entries, tuple, mod)
            for e in entries:
                self.assertIn(e, harness.MODULES, (mod, e))

    def test_the_registered_reachability_map_is_asserted_not_just_censused(self):
        """A pooled non-emptiness census cannot see a label MOVE between two
        non-empty verdicts.  `REGISTERED_MAP` pins every point."""
        self.assertEqual(set(REACH.REGISTERED_MAP),
                         {title for title, _, _ in REACH.SCENARIOS})
        for title, (a, b) in REACH.REGISTERED_MAP.items():
            self.assertEqual(len(a), len(REACH.LAMBDA_RATIO_GRID), title)
            self.assertEqual(len(b), len(REACH.LAMBDA_RATIO_GRID), title)
            self.assertTrue(set(a) | set(b) <= set(REACH.VERDICT_CODE.values()))
        # ★and the FALLBACK scenario -- the one that runs after F1 -- cannot
        # emit LADDER_TOO_LOW for shape A under the registered model (audit V4 /
        # R4C-5).  Registered as a KNOWN LIMIT, not hidden behind the pool.
        self.assertNotIn("L", REACH.REGISTERED_MAP["FALLBACK literal ladders"][0])

    def test_control_is_scored_on_the_mutation_path(self):
        """G-lam3-5 / E3: rev3 called `_run(None, None, env)` and threw the
        result away, re-running in place instead -- so a broken temp-dir copy
        would have let every mutation print "FAILS (good)" vacuously."""
        harness = _load("lambda0_mutation_check")
        src = (PREREG / "lambda0_mutation_check.py").read_text()
        self.assertNotIn("r = _run(None, None, env)\n        r = subprocess.run", src)
        self.assertIn("CONTROL_ENTRIES", src)
        # the control must go through the same copy-to-temp-dir path
        body = src.split("ctl = {entry", 1)[1].split("if bad_control")[0]
        # ★rev5: the runs are dispatched to a thread pool (the reachability
        # entry costs ~90 s and F5-1 routes ~20 mutations into it), so the
        # control is submitted rather than called -- but still through `_run`,
        # i.e. still over the temp-dir copy, and still never `subprocess.run`.
        self.assertIn("pool.submit(_run, None, None, env, entry)", body)
        self.assertNotIn("subprocess.run", body)
        self.assertIn("results are collected and printed in REGISTRY", src)


class TestSbatchDiscipline(unittest.TestCase):
    def setUp(self):
        self.sb = PREREG / "lambda0.sbatch"
        self.text = self.sb.read_text()

    def test_exists_and_parses(self):
        r = subprocess.run(["bash", "-n", str(self.sb)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_comment_directive_is_inside_the_sbatch_block(self):
        lines = self.text.splitlines()
        first_cmd = next(i for i, l in enumerate(lines)
                         if l.strip() and not l.startswith("#"))
        comment = [i for i, l in enumerate(lines) if l.startswith("#SBATCH --comment=")]
        self.assertEqual(len(comment), 1)
        self.assertLess(comment[0], first_cmd)
        self.assertIn("field=efficientai;appl=pytorch", lines[comment[0]])

    def test_does_not_derive_its_own_path(self):
        self.assertIn("ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc", self.text)

    def test_one_boot_per_cell(self):
        body = self.text.split('for SPEC in "${CELLS[@]}"')[-1]
        self.assertIn("sglang.launch_server", body)
        self.assertIn('pkill -9 -f "launch_server.*--port $PORT"', body)

    def test_environment_preamble(self):
        for needle in ("module load conda/pytorch_2.9.1_cuda13",
                       "source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate",
                       "export HF_HOME=", "TRITON_CACHE_DIR", "sync_engine_tree.sh"):
            self.assertIn(needle, self.text, needle)

    def test_port_is_probed_before_use(self):
        self.assertIn("ABORT_NO_FREE_PORT", self.text)

    def test_harness_flags_the_audits_required(self):
        for flag in ("--dataset-path", "--tokenize-prompt", "--output-details",
                     "--output-file", "--random-range-ratio 1.0",
                     "--model", "--host", "--port"):
            self.assertIn(flag, self.text, flag)

    def test_provenance_gate_runs_before_any_boot(self):
        self.assertLess(self.text.index("ABORT_D12"),
                        self.text.index("sglang.launch_server"))
        self.assertIn("models/nemotron_h.py", self.text)

    def test_registered_control_factors(self):
        launch = self.text.split("python -m sglang.launch_server", 1)[1]
        launch = launch.split("PID=$!", 1)[0]
        self.assertNotIn("--disable-piecewise-cuda-graph", launch)
        self.assertIn("--random-seed", launch)
        self.assertIn("--attention-backend", launch)
        self.assertIn("CTX=16384", self.text)
        self.assertIn("MEM_FRACTION=0.82", self.text)
        # the unregistered difference must still be WRITTEN DOWN (D11).
        self.assertIn("--disable-piecewise-cuda-graph", self.text)

    # ---------------- rev2 audit conditions ----------------
    def test_d18_no_operator_switch(self):
        """The branch that flips a label may not be a human choice.

        Checked on the EXECUTABLE lines only: the comment block deliberately
        names the removed variable so a reader learns why it is gone."""
        code = "\n".join(l for l in self.text.splitlines()
                          if l.strip() and not l.lstrip().startswith("#"))
        self.assertNotIn("PDMUX_LAMBDA0_FALLBACK", code)
        self.assertIn("PDMUX_LAMBDA0_FALLBACK", self.text,
                      "the removal must still be explained in the header")
        self.assertIn("lambda0_lambda_inf.py", code)
        self.assertLess(code.index("lambda0_lambda_inf.py"),
                        code.index("lambda0_plan.py"))
        # the only env var left may say WHERE to look, never WHAT to conclude.
        self.assertIn("PDMUX_LAMBDA0_INSTR", code)

    def test_d23_expect_is_built_before_the_loop(self):
        """A cell never attempted must still reach the rule as missing."""
        build = self.text.index('EXPECT=""; REPEAT=""')
        loop = self.text.index('for SPEC in "${CELLS[@]}"',
                               self.text.index("FAIL_STREAK=0"))
        self.assertLess(build, loop)
        # and the loop body must NOT append to them any more
        body = self.text[loop:]
        self.assertNotIn('EXPECT="${EXPECT:+$EXPECT,}', body)

    def test_placeholder_survives_bash_read(self):
        """`read` collapses whitespace, so an empty column would shift every
        column to its right.  The `-` placeholder and its translation must both
        be present, and the renderer must emit 11 fields on saturated rows."""
        self.assertIn('[ "$KAPPA" = "-" ] && KAPPA=""', self.text)
        self.assertIn('[ "$LHAT" = "-" ] && LHAT=""', self.text)
        p = PLAN.plan(2.10, 0.675)
        for row in CELLS.rows(p):
            self.assertEqual(len(row.split()), 11, row)

    def test_analyzer_receives_the_registered_ceiling(self):
        self.assertIn("--low-side-candidate", self.text)
        self.assertIn("--kappa-pred", self.text)
        self.assertIn("--drain-pred-s", self.text)

    def test_mutation_harness_gates_the_run(self):
        self.assertIn("lambda0_mutation_check.py", self.text)
        self.assertIn("ABORT_MUTATION_ESCAPE", self.text)

    def test_all_shipped_selftests_gate_the_run(self):
        """E3: the renderer and the reachability map were not gated in rev3."""
        for m in ("lambda0_plan.py", "lambda0_label.py", "lambda0_cells.py",
                  "lambda0_lambda_inf.py", "lambda0_reachability.py"):
            self.assertIn(f"{m} --selftest".replace(".py --selftest",
                                                    '.py" --selftest'), self.text, m)

    def test_wall_time_covers_the_registered_worst_case(self):
        """E5: --time must cover the 0.25x corner of the coverage band.

        ★rev5/F5-4: the FALLBACK plan is included, because after F1 that is the
        plan that actually runs and its worst corner (3.509 GPU-h) is above
        every REGISTERED_GRID corner (3.224).
        """
        hhmmss = [l for l in self.text.splitlines()
                  if l.startswith("#SBATCH --time=")][0].split("=")[1]
        h, m, sec = (int(x) for x in hhmmss.split(":"))
        wall_h = h + m / 60.0 + sec / 3600.0
        fb = PLAN.plan(2.10, 0.675,
                       ladders={k: tuple(v)
                                for k, v in PLAN.FALLBACK_LADDER.items()})
        worst = max([PLAN.budget(PLAN.plan(la, lb), r)["total_gpu_h"]
                     for _, la, lb in PLAN.REGISTERED_GRID
                     for r in PLAN.BUDGET_RATIO_GRID]
                    + [PLAN.budget(fb, r)["total_gpu_h"]
                       for r in PLAN.BUDGET_RATIO_GRID])
        self.assertGreater(wall_h, worst, (wall_h, worst))
        self.assertLessEqual(worst, PLAN.REQUESTED_BUDGET_GPU_H, worst)

    def test_f6_records_the_decision_path_and_the_prereg_digest(self):
        """★F6: rev4 ran with 14 uncommitted executables and no digests, so a
        post-hoc edit of any of them was undetectable."""
        self.assertIn("REGISTRATION_SHA256.txt", self.text)
        self.assertIn("DECISION_PATH=(", self.text)
        for f in ("lambda0_lambda_inf.py", "lambda0_plan.py", "lambda0_label.py",
                  "lambda0_analyze.py", "lambda0_cells.py",
                  "lambda0_reachability.py", "lambda0_mutation_check.py"):
            self.assertIn(f, self.text, f)
        self.assertIn('sha256sum "$PREREG_MD"', self.text)
        self.assertIn('sha256sum "$PREREG"/PREREG_LAMBDA0*.md', self.text)
        # and the manifest must be written BEFORE the branch is decided
        self.assertLess(self.text.index("REGISTRATION_SHA256.txt"),
                        self.text.index("lambda0_lambda_inf.py\" --selftest"))
        # ...and a missing pre-registration must abort, not run unregistered
        self.assertIn("ABORT_F6 pre-registration document missing", self.text)

    def test_no_phantom_flags_and_no_stale_revision_banner(self):
        """rev4 printed "rev3" in its banner and passed a `--record` flag the
        predicate never had (argv length 3 -> docstring + rc 2, swallowed by
        `tee`, leaving the job on the fallback branch for the WRONG reason)."""
        # the flag must not be PASSED (it may be named in a comment saying so)
        for line in self.text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            self.assertNotIn("--record", line, line)
        self.assertNotIn("lambda* rev3", self.text)
        self.assertIn('basename "$PREREG_MD"', self.text)
        # the per-cell recount is archived with the decision
        self.assertIn("--recount", self.text)
        self.assertIn("RUNNING_REQ_RECOUNT.txt", self.text)

    def test_an_empty_mode_aborts_instead_of_falling_back(self):
        """★the other half of the `--record` defect: a predicate that crashed
        left `$LAMBDA0_MODE` empty, and the `if ANCHORED ... else` took the
        FALLBACK branch -- a measurement failure printed as a verdict."""
        self.assertIn("ABORT_D18 predicate produced no mode", self.text)
        head = self.text.split('if [ "$LAMBDA0_MODE" = "ANCHORED" ]')[0]
        self.assertIn("ANCHORED|FALLBACK) ;;", head)
        # and the decision command's exit status may not be swallowed by `tee`
        self.assertIn('| tee "$OUT/LAMBDA_INF_DECISION.txt" || exit 2', self.text)

    def test_the_phantom_flag_would_really_have_failed(self):
        """NEGATIVE CONTROL for the line above: the tool must reject the extra
        argument, so this was a live defect and not a cosmetic one."""
        r = subprocess.run([sys.executable, str(PREREG / "lambda0_lambda_inf.py"),
                            str(MEASURED_INSTR), "--record"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout[-500:])
        self.assertNotIn("LAMBDA0_MODE=", r.stdout)


class TestNoStaleRules(unittest.TestCase):
    """★F5-3 / gate #179: a constant left behind with a comment describing a
    RETRACTED rule reads, months later, as a live rule."""

    def test_plan_has_no_dead_drain_tolerance(self):
        src = (PREREG / "lambda0_plan.py").read_text()
        # no ASSIGNMENT (the deletion is recorded in a comment, which is fine)
        self.assertFalse([l for l in src.splitlines()
                          if l.startswith("DRAIN_MODEL_TOL")], src[:0])
        self.assertFalse(hasattr(PLAN, "DRAIN_MODEL_TOL"))
        # the live one lives in the module that applies it
        self.assertEqual(ANALYZE.DRAIN_MODEL_TOL, 2.0)
        self.assertIn("DRAIN_MODEL_TOL", (PREREG / "lambda0_analyze.py").read_text())

    def test_the_retracted_rev3_wording_is_gone(self):
        """E1 retracted "a blown drain voids the SHAPE"; the plan's comment
        still said so."""
        src = (PREREG / "lambda0_plan.py").read_text()
        # the rev3 wording may only survive as a QUOTATION marked retracted
        for line in src.splitlines():
            if "measured L > TOL * Lhat -> UNRESOLVED" in line:
                self.assertIn("RETRACTED", src)
                self.assertTrue(line.lstrip().startswith("#"), line)

    def test_plan_no_longer_claims_the_engine_cap_was_reached(self):
        """★N1: "with I3_max_running_req = 48 (the engine cap was reached,
        newpair F5)" was FALSE at cell granularity, inside a rule-canon file
        (gate #80: a false provenance claim is most dangerous in code)."""
        src = (PREREG / "lambda0_plan.py").read_text()
        # kept only as a quotation that is explicitly marked false
        self.assertIn("WAS FALSE AT CELL GRANULARITY", src)
        self.assertIn("I3b = 2 (F5 NOT met)", src)
        self.assertIn("THIS SCENARIO IS NO LONGER THE ONE THAT RUNS", src)


class TestPreregDocument(unittest.TestCase):
    def setUp(self):
        self.doc = (PREREG / "PREREG_LAMBDA0_REV5_2026-09-14.md").read_text()

    def test_earlier_revisions_are_marked_superseded_and_kept(self):
        for old in ("PREREG_LAMBDA0_2026-09-13.md",
                    "PREREG_LAMBDA0_REV2_2026-09-13.md",
                    "PREREG_LAMBDA0_REV3_2026-09-13.md",
                    "PREREG_LAMBDA0_REV4_2026-09-13.md"):
            p = PREREG / old
            self.assertTrue(p.exists(), f"{old} must not be deleted")
            self.assertIn("SUPERSEDED", p.read_text()[:2500], old)

    def test_thresholds_match_the_code(self):
        self.assertEqual(LABEL.ACH_HI, 0.95)
        self.assertEqual(LABEL.ACH_LO, 0.90)
        self.assertEqual(PLAN.KAPPA_MIN, 0.97)
        self.assertIn("0.97", self.doc)

    def test_carries_the_registered_citation_bans(self):
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "L1", "L2", "L3",
                  "R3C-1", "R3C-2", "R3C-3", "R3C-4", "NPC-I"):
            self.assertIn(q, self.doc, q)
        # E7-2: the clauses the rev3 audit found had drifted out.
        self.assertIn("이 단계 자신이 두 shape에서 다른 값을 낸다", self.doc)
        self.assertIn("3b", self.doc)

    def test_states_that_gate_6_is_not_closed(self):
        """The single most important sentence both audits told rev3 to keep."""
        self.assertIn("#6", self.doc)
        self.assertIn("0.59", self.doc)

    def test_w4_approximation_language_is_gone(self):
        self.assertNotIn("(64, 512)", self.doc.replace("(in 64, out 512)", ""))
        self.assertIn("workloads.py:159-160", self.doc)

    def test_records_the_drain_identity_and_its_consequence(self):
        for needle in ("span + L", "kappa", "511", "19.92"):
            self.assertIn(needle, self.doc, needle)

    def test_records_the_measured_lambda_inf_and_its_scope(self):
        for needle in ("907959", "3.0939", "0.6956", "48"):
            self.assertIn(needle, self.doc, needle)

    def test_records_that_the_correctness_gate_is_now_adjudicated(self):
        """rev4 was written while the new (model, backend) pair's true-dual
        correctness gate was still open (907959 FAILed, CUDA OOM).  It is
        adjudicated now: 908179 PASSed on the repaired engine.  rev5 must
        record BOTH the adjudication and its registered limit -- that PASS
        certifies one sentence and does not authorise P2 (NP-8).  F3-(5) of
        VERDICT_lambda0_rev4_2026-09-14.md required the wording and this test
        to move together; a test still demanding "판정 전" would have pinned a
        sentence that became false on 2026-09-14."""
        self.assertIn("908179", self.doc)
        self.assertIn("PASS", self.doc)
        self.assertIn("판정됐", self.doc)
        self.assertIn("NP-8", self.doc)
        self.assertIn("P2 착수를 승인하지 않는다", self.doc)


if __name__ == "__main__":
    unittest.main()

class TestRev5AddendumCodeRepairs(unittest.TestCase):
    """★rev5 addendum sec B -- the five CODE-side recommendations of the 5th
    rules-layer audit (`VERDICT_lambda0_rev5_2026-09-14.md` sec 4,
    `GO-with-caveats`, sha `33dac116c37fe758f39d9b1aa67f55124c0d714a142545e25d5
    44dc45abd45b0`).  The auditor demonstrated every one of them BY EXECUTION,
    so every test here executes the shipped bytes rather than reading them:

      lam5A-2  `PDMUX_LAMBDA0_INSTR` (what `decide()` reads) and
               `LAMBDA0_I3_JOB_DIR` (what the predicate's selftest recounts) were
               INDEPENDENT -- a substituted instrument dir answered ANCHORED
               while the shipped selftest still returned rc 0 (W4').
      lam5A-3  every plan was titled `### measured lambda_inf`, including the
               FALLBACK one, whose lambda_inf is a registered PRIOR (lam5C-4).
      lam5A-4  the ANCHORED branch was unreachable on the registered inputs and
               would have built a DIFFERENT ladder in silence.
      lam5A-9  every non-zero rc of the mutation harness printed
               `ABORT_MUTATION_ESCAPE`, including rc 2 = `HARNESS CANNOT RUN`,
               which is a MEASUREMENT FAILURE (lesson 21 / gate #21).
      lam5A-11 the producer's line citations were true only of one uncommitted
               working tree, and another track is editing that file.
    """

    def setUp(self):
        self.sb = (PREREG / "lambda0.sbatch").read_text()

    # -------------------------------------------------------------- helpers
    def _bash(self, block, env=None, prereg=None, out=None):
        """Run an extracted block under the sbatch's own shell options."""
        script = ("set -uo pipefail\n"
                  "ROOT=%s\nPREREG=%s\nOUT=%s\n" % (PROJECT_ROOT,
                                                    prereg or PREREG,
                                                    out or PREREG)) + block
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(script)
            path = fh.name
        try:
            return subprocess.run(["bash", path], capture_output=True, text=True,
                                  env=dict(os.environ, **(env or {})))
        finally:
            os.unlink(path)

    def _anchoring_fixture(self, td):
        """The auditor's W4' input: a SUBSTITUTE instrument dir whose
        hand-written offsets flip the branch to ANCHORED with zero
        disqualifications, carrying the very lambda_inf pair the live cells
        reported before F1 disqualified cell B.  Asserted here so the tests
        below cannot pass vacuously on an input that never flips anything."""
        d = LAMINF._fixture(td, {"I3a_shapeA": LAMINF.W_OK_A,
                                 "I3b_shapeB": LAMINF.W_OK_B},
                            thr_a=3.0939, thr_b=0.6956)
        r = LAMINF.decide(d)
        self.assertEqual(r["mode"], "ANCHORED", r)
        self.assertEqual(r["disqualifications"], [], r)
        self.assertEqual(r["lambda_inf"], {"A": 3.0939, "B": 0.6956}, r)
        return d

    # -------------------------------------------------------------- lam5A-2
    def test_lam5a2_the_selftest_input_is_bound_to_the_branch_input(self):
        """The prescribed one-liner, verified the way the auditor verified it:
        rc != 0 on a substituted dir, rc 0 on the reference case."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        block = _sbatch_block(self.sb, 'INSTR="${PDMUX_LAMBDA0_INSTR:-',
                              'lambda0_lambda_inf.py" --selftest')
        self.assertIn('export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"', block)
        with tempfile.TemporaryDirectory() as td:
            d = self._anchoring_fixture(td)
            # (i) the substituted dir no longer passes the gate
            r = self._bash(block, {"PDMUX_LAMBDA0_INSTR": str(d)})
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("AssertionError", r.stderr)
            # ...and the reason is the registered recount table, not something else
            sub = subprocess.run(
                [sys.executable, str(PREREG / "lambda0_lambda_inf.py"),
                 "--selftest"], capture_output=True, text=True,
                env=dict(os.environ, LAMBDA0_I3_JOB_DIR=str(Path(d).parent)))
            self.assertEqual(sub.returncode, 1, sub.stderr[-800:])
            self.assertIn("AssertionError: ['I3a_shapeA', 'I3b_shapeB']",
                          sub.stderr)
        # (ii) the prescription does NOT disqualify the reference case -- both
        # with the dir named explicitly and with the registered default.
        for env in ({"PDMUX_LAMBDA0_INSTR": str(MEASURED_INSTR)}, {}):
            r = self._bash(block, env)
            self.assertEqual(r.returncode, 0, (env, r.stdout, r.stderr[-800:]))

    def test_lam5a2_negative_control_the_unbound_gate_certified_another_job(self):
        """LESSON 53 -- the defect was LIVE, not cosmetic.  Delete the one export
        line (= the shipped state the audit read) and the SAME substituted dir
        passes with rc 0: the gate recounts job 907959 while the branch is
        decided from somebody else's bytes."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        block = _sbatch_block(self.sb, 'INSTR="${PDMUX_LAMBDA0_INSTR:-',
                              'lambda0_lambda_inf.py" --selftest',
                              drop=('export LAMBDA0_I3_JOB_DIR',))
        # the EXECUTABLE lines no longer bind it (the comment still names it,
        # which is how a reader learns why the export has to be there)
        self.assertFalse([l for l in block.splitlines()
                          if "LAMBDA0_I3_JOB_DIR" in l
                          and not l.lstrip().startswith("#")], block)
        with tempfile.TemporaryDirectory() as td:
            d = self._anchoring_fixture(td)
            r = self._bash(block, {"PDMUX_LAMBDA0_INSTR": str(d)})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_lam5a2_the_export_does_not_leak_into_the_mutation_harness(self):
        """The harness builds its children's environment EXPLICITLY, so the new
        export cannot retarget it (addendum sec B-1 note).  Checked by polluting
        the parent environment and running the harness's own `_run` on the
        control -- not by reading the source only."""
        if not _live_job_present():
            self.skipTest("job 907959 warm-up log / instrument dir not present")
        self.assertIn("LAMBDA0_I3_JOB_DIR=str(I3_JOB)",
                      (PREREG / "lambda0_mutation_check.py").read_text())
        polluted = dict(os.environ, LAMBDA0_I3_JOB_DIR="/nonexistent/elsewhere")
        env = dict(polluted, LAMBDA0_PROBE_C_DIR=str(MUT.ARCHIVE),
                   LAMBDA0_I3_JOB_DIR=str(MUT.I3_JOB))
        r = MUT._run(None, None, env, "lambda0_lambda_inf.py")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        # negative control: WITHOUT the explicit override the same call fails,
        # i.e. the override is what protects the harness.
        bad = MUT._run(None, None, polluted, "lambda0_lambda_inf.py")
        self.assertNotEqual(bad.returncode, 0, bad.stdout[-400:])

    # -------------------------------------------------------------- lam5A-4
    def test_lam5a4_unexpected_anchored_aborts_instead_of_replanning(self):
        """The unreachable branch fails CLOSED and the predicate is preserved."""
        self.assertNotIn("PLAN=anchored", self.sb)
        self.assertIn("ANCHORED|FALLBACK) ;;", self.sb)   # still a predicate
        block = _sbatch_block(self.sb, 'if [ "$LAMBDA0_MODE" = "ANCHORED" ]',
                              'echo "PLAN=fallback')
        common = {"LAMBDA0_A": "3.0939", "LAMBDA0_B": "0.6956",
                  "INSTR": str(MEASURED_INSTR)}
        with tempfile.TemporaryDirectory() as out:
            r = self._bash(block, dict(common, LAMBDA0_MODE="ANCHORED"), out=out)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("ABORT_D18 unexpected ANCHORED", r.stdout)
            self.assertIn("rev5 registers FALLBACK", r.stdout)
            self.assertIn("ABORT_D18 unexpected ANCHORED",
                          (Path(out) / "PREFLIGHT_FAILURES.txt").read_text())
            # ...and the registered FALLBACK ladder is still what gets built
            r = self._bash(block + 'printf "%s\n" "${PLAN_ARGS[@]}"\n',
                           dict(common, LAMBDA0_MODE="FALLBACK"), out=out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("PLAN=fallback", r.stdout)
            for needle in ("--lambda-inf-a", "2.10", "--lambda-inf-b", "0.675",
                           "--fallback"):
                self.assertIn(needle, r.stdout, needle)

    def test_lam5a4_negative_control_the_old_branch_would_have_replanned(self):
        """LESSON 53: on the shipped-before state the same ANCHORED mode built a
        DIFFERENT ladder (from the substituted lambda_inf) and printed rc 0 --
        an unregistered plan, silently."""
        old = ('if [ "$LAMBDA0_MODE" = "ANCHORED" ]; then\n'
               '  PLAN_ARGS=(--lambda-inf-a "$LAMBDA0_A" --lambda-inf-b "$LAMBDA0_B")\n'
               '  echo "PLAN=anchored A=$LAMBDA0_A B=$LAMBDA0_B (from $INSTR)"\n'
               'else\n'
               '  PLAN_ARGS=(--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback)\n'
               '  echo "PLAN=fallback"\n'
               'fi\n'
               'printf "%s\n" "${PLAN_ARGS[@]}"\n')
        r = self._bash(old, {"LAMBDA0_MODE": "ANCHORED", "LAMBDA0_A": "3.0939",
                             "LAMBDA0_B": "0.6956", "INSTR": str(MEASURED_INSTR)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PLAN=anchored", r.stdout)
        self.assertIn("3.0939", r.stdout)
        self.assertNotIn("--fallback", r.stdout)
        # and that text is GONE from the shipped runner
        self.assertNotIn('PLAN_ARGS=(--lambda-inf-a "$LAMBDA0_A"', self.sb)

    # -------------------------------------------------------------- lam5A-9
    def test_lam5a9_a_harness_that_cannot_run_is_not_an_escape(self):
        """rc 1 = an escape (a verdict).  rc 2 = no measurement.  The runner must
        not print the first when it means the second (lesson 21 / gate #21)."""
        block = _sbatch_block(self.sb,
                              'python3 "$PREREG/lambda0_mutation_check.py"',
                              'esac')
        cases = ((0, 0, None, None),
                 (1, 5, "ABORT_MUTATION_ESCAPE",
                  "ABORT_MUTATION_HARNESS_CANNOT_RUN"),
                 (2, 6, "ABORT_MUTATION_HARNESS_CANNOT_RUN",
                  "ABORT_MUTATION_ESCAPE"),
                 (7, 6, "ABORT_MUTATION_UNEXPECTED_RC=7",
                  "ABORT_MUTATION_ESCAPE"))
        for stub_rc, want_rc, want, forbidden in cases:
            with tempfile.TemporaryDirectory() as td:
                Path(td, "lambda0_mutation_check.py").write_text(
                    "import sys\nprint('stub')\nsys.exit(%d)\n" % stub_rc)
                r = self._bash(block, prereg=td, out=td)
                self.assertEqual(r.returncode, want_rc,
                                 (stub_rc, r.returncode, r.stdout, r.stderr))
                if want:
                    self.assertIn(want, r.stdout, (stub_rc, r.stdout))
                    self.assertIn(want, Path(td, "PREFLIGHT_FAILURES.txt").read_text())
                if forbidden:
                    self.assertNotIn(forbidden, r.stdout, (stub_rc, r.stdout))
                if stub_rc == 0:
                    self.assertFalse(Path(td, "PREFLIGHT_FAILURES.txt").exists())
                # the measurement-failure labels must SAY they are measurement
                # failures, so the run record cannot be misread later
                if stub_rc in (2, 7):
                    self.assertIn("MEASUREMENT FAILURE", r.stdout)

    def test_lam5a9_negative_control_the_old_line_mislabelled_rc2(self):
        """LESSON 53: the reverted one-liner prints ABORT_MUTATION_ESCAPE for rc
        2 as well -- the exact misattribution lam5A-9 names."""
        old = ('python3 "$PREREG/lambda0_mutation_check.py" > "$OUT/mutation_check.txt" 2>&1 \\\n'
               '  || { echo "ABORT_MUTATION_ESCAPE"; tail -5 "$OUT/mutation_check.txt"; exit 5; }\n')
        for stub_rc in (1, 2):
            with tempfile.TemporaryDirectory() as td:
                Path(td, "lambda0_mutation_check.py").write_text(
                    "import sys\nsys.exit(%d)\n" % stub_rc)
                r = self._bash(old, prereg=td, out=td)
                self.assertEqual(r.returncode, 5, (stub_rc, r.stdout))
                self.assertIn("ABORT_MUTATION_ESCAPE", r.stdout)
        # and the shipped runner no longer contains that undifferentiated form
        self.assertNotIn('|| { echo "ABORT_MUTATION_ESCAPE"', self.sb)

    def test_lam5a9_rc2_is_a_real_state_of_the_shipped_harness(self):
        """rc 2 is not hypothetical.  Both artifacts the harness needs are
        gitignored, so on a fresh clone the SHIPPED harness prints `HARNESS
        CANNOT RUN` and returns 2 before scoring a single mutation."""
        with tempfile.TemporaryDirectory() as td:
            deep = Path(td, "a", "b", "c")
            deep.mkdir(parents=True)
            for f in tuple(MUT.MODULES) + ("lambda0_mutation_check.py",):
                shutil.copy(PREREG / f, deep)
            r = subprocess.run([sys.executable,
                                str(deep / "lambda0_mutation_check.py")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stdout[-600:])
            self.assertIn("HARNESS CANNOT RUN", r.stdout)
            self.assertNotIn("escapes:", r.stdout)   # nothing was scored

    # -------------------------------------------------------------- lam5A-3
    def test_lam5a3_the_fallback_plan_is_not_titled_measured(self):
        """`PLAN.txt` is the run record; on the FALLBACK branch lambda_inf is the
        registered PRIOR (A 2.10 / B 0.675) and neither number is measured."""
        def run(args):
            return subprocess.run([sys.executable,
                                   str(PREREG / "lambda0_plan.py")] + args,
                                  capture_output=True, text=True)
        fb = run(["--lambda-inf-a", "2.10", "--lambda-inf-b", "0.675",
                  "--fallback"])
        self.assertEqual(fb.returncode, 0, fb.stderr[-500:])
        head = fb.stdout.splitlines()[0]
        self.assertTrue(head.startswith("### FALLBACK prior lambda_inf "
                                        "(NOT measured"), head)
        self.assertNotIn("### measured", head)
        # the heading now describes the numbers printed beneath it
        self.assertIn("lambda_inf=2.1", fb.stdout)
        self.assertIn("lambda_inf=0.675", fb.stdout)
        self.assertEqual(PLAN.LAMBDA_STAR_PRIOR["A"][0], 2.1)
        # ...and it names the disqualified pair as disqualified, not as absent
        for needle in ("3.0939", "0.6956", "DISQUALIFIED", "F5"):
            self.assertIn(needle, head, needle)
        # the anchored invocation is untouched (the branch still exists)
        an = run(["--lambda-inf-a", "3.0939", "--lambda-inf-b", "0.6956"])
        self.assertEqual(an.returncode, 0, an.stderr[-500:])
        self.assertEqual(an.stdout.splitlines()[0], "### measured lambda_inf")
        # the runner only ever asks for the FALLBACK form now (lam5A-4)
        self.assertIn("--fallback", self.sb)

    def test_lam5a3_negative_control_one_constant_title_was_false(self):
        """LESSON 53: with the title hard-coded -- the shipped-before state --
        the FALLBACK plan is headed `### measured lambda_inf` above 2.1/0.675."""
        p = PLAN.plan(2.10, 0.675, ladders={k: tuple(v) for k, v
                                            in PLAN.FALLBACK_LADDER.items()})
        reverted = PLAN._fmt_plan(p, "measured lambda_inf")
        self.assertEqual(reverted.splitlines()[0], "### measured lambda_inf")
        self.assertIn("lambda_inf=2.1", reverted)     # ...and it is not measured
        # the constant form must be gone from `main()`
        src = (PREREG / "lambda0_plan.py").read_text()
        self.assertNotIn('_fmt_plan(p, "measured lambda_inf")', src)
        self.assertIn("FALLBACK_TITLE", src)

    # ------------------------------------------------------------- lam5A-11
    def test_lam5a11_the_producer_is_cited_by_code_text_not_by_line(self):
        """Another track is editing `r2_correctness.sbatch` (573 -> 684 lines on
        2026-09-14), so a line citation there is true of one unnamed snapshot."""
        src = (PREREG / "lambda0_lambda_inf.py").read_text()
        self.assertEqual(re.findall(r"r2_correctness\.sbatch:\d+", src), [])
        quoted = ('grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log" | awk',
                  'sort -n | tail -1 > "$IOUT/I3_max_running_req.txt"',
                  "i_mark ()")
        for q in quoted:
            self.assertIn(q, src, q)
            if PRODUCER.exists():
                # a code-text citation is only better than a line number if it
                # RESOLVES -- and resolves to exactly one place.
                self.assertEqual(PRODUCER.read_text().count(q), 1, q)
        # the line numbers that remain are attributed to NAMED snapshots
        for needle in (":496-497", "HEAD 8507cee", ":580-581",
                       "uncommitted working tree", ":429-430"):
            self.assertIn(needle, src, needle)

    def test_lam5a11_the_pinned_snapshot_really_holds_those_lines(self):
        """The provenance claim is itself checkable, so it is checked: at commit
        8507cee the quoted block sits at 496-497 and the file is 573 lines."""
        blob = subprocess.run(
            ["git", "show", "8507cee:workspace/engine-port/results/"
             "r2_correctness/r2_correctness.sbatch"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if blob.returncode != 0:
            self.skipTest("commit 8507cee not present in this clone")
        lines = blob.stdout.splitlines()
        self.assertEqual(len(lines), 573, len(lines))
        self.assertIn('grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log"',
                      lines[495])
        self.assertIn('tail -1 > "$IOUT/I3_max_running_req.txt"', lines[496])

    # -------------------------------------------------------------- lam5A-6
    def test_lam5a6_the_printed_rules_banner_names_the_live_revision(self):
        """rev5 sec 5 SUPERSEDES rev4 sec 5; addendum sec A4 fixes in writing
        which rev4 sections survive (sec 9, 10, 11)."""
        code = "\n".join(l for l in self.sb.splitlines()
                         if l.strip() and not l.lstrip().startswith("#"))
        self.assertNotIn("PREREG_LAMBDA0_REV4 sec 5", code)
        self.assertIn("PREREG_LAMBDA0_REV5 sec 5", code)
        self.assertIn("rev4 sec 9/10/11", code)
        self.assertIn("SUPERSEDED", code)
        # addendum sec A4 is the WRITTEN source of that split
        addendum = (PREREG / "PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md"
                    ).read_text()
        for needle in ("§9 인용금지", "§10 필수병기",
                       "§11 인용 고정표", "§5 판정 규칙"):
            self.assertIn(needle, addendum, needle)
