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

★ Negative controls, because a repair test that also passes on the unrepaired
code certifies nothing (lesson #53):
  `test_d4_injection_fails_on_rev1_glob`        rev1's glob gets it WRONG.
  `test_rev2_low_rungs_are_rejected`            rev2's 170 s shape-A low rungs
                                                fail the new ceiling.
  `test_drain_identity_on_archived_cells`       the identity that killed rev2,
                                                re-derived from raw artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
MEASURED_INSTR = (TRACK_ROOT / "results" / "r2_correctness" / "job_907959"
                  / "instrument")


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
        _, seen = REACH.table()
        for shape in ("A", "B"):
            for verdict in ("KNEE_BRACKETED", "KNEE_NOT_BRACKETED",
                            "LADDER_TOO_HIGH", "LADDER_TOO_LOW"):
                self.assertTrue(seen[shape].get(verdict),
                                (shape, verdict, sorted(seen[shape])))

    def test_shape_b_brackets_at_the_measured_anchor(self):
        """The rev3 kill cause, at the value that will actually be served:
        lambda_inf(B) = 0.6956 (job 907959) sits inside the band rev3 threw
        away wholesale."""
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

    def test_measured_artifacts_select_anchored(self):
        if not MEASURED_INSTR.exists():
            self.skipTest("job 907959 instrument dir not present")
        r = LAMINF.decide(MEASURED_INSTR)
        self.assertEqual(r["mode"], "ANCHORED", r)
        self.assertAlmostEqual(r["lambda_inf"]["A"], 3.0939, places=3)
        self.assertAlmostEqual(r["lambda_inf"]["B"], 0.6956, places=3)
        self.assertEqual(r["disqualifications"], [])

    def test_e6_provenance_is_recorded(self):
        if not MEASURED_INSTR.exists():
            self.skipTest("job 907959 instrument dir not present")
        r = LAMINF.decide(MEASURED_INSTR)
        self.assertTrue(Path(r["instrument_dir"]).is_absolute())
        self.assertEqual(len(r["inputs"]), 3)
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

    def test_f5_running_req_below_48_forces_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            for fn in LAMINF.CELL_FILES.values():
                Path(td, fn).write_text(json.dumps(
                    {"request_throughput": 2.4, "request_rate": None}))
            Path(td, LAMINF.RUNNING_FILE).write_text("48")
            self.assertEqual(LAMINF.decide(td)["mode"], "ANCHORED")
            Path(td, LAMINF.RUNNING_FILE).write_text("31")
            r = LAMINF.decide(td)
            self.assertEqual(r["mode"], "FALLBACK")
            self.assertTrue(any("F5" in x for x in r["disqualifications"]))


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
        for name in ("X3", "X4", "X5", "Y1", "Y2", "Y4", "Y5", "Y6"):
            self.assertTrue(any(k.startswith(name) for k in harness.MUTATIONS),
                            f"the auditors' {name} must be registered")

    def test_control_is_scored_on_the_mutation_path(self):
        """G-lam3-5 / E3: rev3 called `_run(None, None, env)` and threw the
        result away, re-running in place instead -- so a broken temp-dir copy
        would have let every mutation print "FAILS (good)" vacuously."""
        harness = _load("lambda0_mutation_check")
        src = (PREREG / "lambda0_mutation_check.py").read_text()
        self.assertNotIn("r = _run(None, None, env)\n        r = subprocess.run", src)
        self.assertIn("CONTROL_ENTRIES", src)
        # the control must go through the same copy-to-temp-dir path
        body = src.split("for entry in CONTROL_ENTRIES:", 1)[1].split("if bad_control")[0]
        self.assertIn("_run(None, None, env, entry=entry)", body)
        self.assertNotIn("subprocess.run", body)


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
        """E5: --time must cover the 0.25x corner of the coverage band."""
        hhmmss = [l for l in self.text.splitlines()
                  if l.startswith("#SBATCH --time=")][0].split("=")[1]
        h, m, sec = (int(x) for x in hhmmss.split(":"))
        wall_h = h + m / 60.0 + sec / 3600.0
        worst = max(PLAN.budget(PLAN.plan(la, lb), r)["total_gpu_h"]
                    for _, la, lb in PLAN.REGISTERED_GRID
                    for r in PLAN.BUDGET_RATIO_GRID)
        self.assertGreater(wall_h, worst, (wall_h, worst))


class TestPreregDocument(unittest.TestCase):
    def setUp(self):
        self.doc = (PREREG / "PREREG_LAMBDA0_REV4_2026-09-13.md").read_text()

    def test_earlier_revisions_are_marked_superseded_and_kept(self):
        for old in ("PREREG_LAMBDA0_2026-09-13.md",
                    "PREREG_LAMBDA0_REV2_2026-09-13.md",
                    "PREREG_LAMBDA0_REV3_2026-09-13.md"):
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

    def test_does_not_prejudge_the_correctness_gate_outcome(self):
        """The same job's true-dual correctness gate FAILED (CUDA OOM in the C
        tier).  What that blocks for the legacy-arm stage 0 is not adjudicated
        yet, and this registration must not assume either way."""
        self.assertIn("FAIL", self.doc)
        self.assertIn("판정 전", self.doc)


if __name__ == "__main__":
    unittest.main()
