"""Gate for the methodology-gate-#14 repair of ``pdmux_eval/analyze.py``.

WHY THIS EXISTS.  Gate #14 (``PROJECT_STATUS.md`` #14, 2026-08-06) says: at n<=8
repetitions the paired/unpaired percentile-bootstrap interval may not decide a
verdict -- the primary is the t interval, the bootstrap is reported alongside.
For a month the file the gate NAMES said the opposite: it shipped no t-CI, its
``unpaired_bootstrap_ci`` docstring called the paired BOOTSTRAP "preferred", and
its CLI read ``headline_improvement`` straight off the bootstrap interval with no
n guard.  That is lesson #97 ("a gate registered in the canon but contradicted by
the tool recurs").  These tests pin the repair, and -- because the repair had to
be additive -- they also pin what the repair was NOT allowed to touch.

Four groups:
  A  BEHAVIOUR PRESERVATION.  Every pre-existing key of both bootstrap functions
     is bit-identical to the pre-repair implementation on ``main``.  This is the
     cheap, direct proof of the hard constraint: ``results/p1_gates/verify/
     verify_c1_coverage.py`` recovers the 10000xN resample-count matrix by
     replaying ``random.Random(1)`` and VALIDATES it against these functions'
     own ci95_low/high, so a changed seed, sample count, draw order or key would
     break the evidence path of gate #14 itself.
  B  ONE t IMPLEMENTATION.  ``t_crit_for``/``paired_t_ci`` agree numerically with
     the audited original in ``results/cp_baseline/d1_predicates.py`` (lesson:
     the project must not grow a second, unaudited t).
  C  THE GUARD IS NON-DESTRUCTIVE.  n<=8 marks ``decision_eligible`` False and
     warns; it never raises, because the n=3 and n=5 callers in this repository
     are gate #14's own evidence generators.
  D  THE CLI DECIDES ON THE t INTERVAL.  Including a dataset where the two
     estimators DISAGREE, so "it now calls paired_t_ci" cannot pass vacuously.

``PDMUX_ANALYZE_PATH`` overrides which analyze.py is loaded; ``mutation_gate14.py``
uses it to check that reverting the repair makes these tests fail (gate #53).
"""

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CANONICAL = REPO / "workspace/engine-port/benchmarks/pdmux_eval/analyze.py"
ANALYZE_PATH = Path(os.environ.get("PDMUX_ANALYZE_PATH", CANONICAL))
D1_PREDICATES = REPO / "workspace/engine-port/results/cp_baseline/d1_predicates.py"
REL_ANALYZE = "workspace/engine-port/benchmarks/pdmux_eval/analyze.py"


def load_module(path, name):
    """Load an analyze.py by FILE PATH (it is stdlib-only and imports nothing
    from its own package, so no sys.path surgery is needed and two revisions can
    coexist in one process)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # registered before exec: ``@dataclass`` resolves ``cls.__module__`` through
    # ``sys.modules`` and fails on an unregistered module
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module(ANALYZE_PATH, "analyze_under_test")

try:
    d1 = load_module(D1_PREDICATES, "d1_predicates_reference")
except Exception:  # pragma: no cover - the reference is optional evidence
    d1 = None


def _main_revision():
    """The pre-repair analyze.py from ``main``, or None if git cannot supply it."""
    try:
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", "main:" + REL_ANALYZE],
            capture_output=True, check=True, text=True,
        ).stdout
    except Exception:
        return None
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_analyze_main.py", delete=False, encoding="utf-8"
    )
    handle.write(blob)
    handle.close()
    return load_module(handle.name, "analyze_main_revision")


BASELINE_REVISION = _main_revision()


def same(a, b):
    """Bit equality, with NaN == NaN (both are legitimate returns here)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return type(a) is type(b) and a == b


def random_pair(n, seed):
    rng = random.Random(seed)
    keys = ["r%d" % i for i in range(n)]
    base = {k: 8.0 + rng.gauss(0.0, 1.5) for k in keys}
    prop = {k: base[k] + rng.gauss(0.4, 1.0) for k in keys}
    return base, prop


class BehaviourPreservationTest(unittest.TestCase):
    """A -- the two bootstrap functions are frozen (additive repair only)."""

    @unittest.skipIf(BASELINE_REVISION is None, "git could not supply main's analyze.py")
    def test_paired_bootstrap_keys_are_bit_identical_to_main(self):
        checked = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in range(3, 11):
                for seed in range(5):
                    base, prop = random_pair(n, 1000 * n + seed)
                    old = BASELINE_REVISION.paired_bootstrap_ci(base, prop)
                    new = analyze.paired_bootstrap_ci(base, prop)
                    self.assertTrue(
                        set(old) <= set(new),
                        "n=%d seed=%d: keys were removed: %s"
                        % (n, seed, sorted(set(old) - set(new))),
                    )
                    for key, value in old.items():
                        self.assertTrue(
                            same(value, new[key]),
                            "n=%d seed=%d key=%r: %r != %r (pre-repair)"
                            % (n, seed, key, new[key], value),
                        )
                        checked += 1
        self.assertGreaterEqual(checked, 8 * 5 * 7)

    @unittest.skipIf(BASELINE_REVISION is None, "git could not supply main's analyze.py")
    def test_unpaired_bootstrap_keys_are_bit_identical_to_main(self):
        checked = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for n in range(3, 11):
                for seed in range(5):
                    base, prop = random_pair(n, 7000 * n + seed)
                    a = list(base.values())
                    b = list(prop.values())[: max(2, n - 1)]
                    old = BASELINE_REVISION.unpaired_bootstrap_ci(a, b)
                    new = analyze.unpaired_bootstrap_ci(a, b)
                    self.assertTrue(set(old) <= set(new))
                    for key, value in old.items():
                        self.assertTrue(
                            same(value, new[key]),
                            "n=%d seed=%d key=%r: %r != %r (pre-repair)"
                            % (n, seed, key, new[key], value),
                        )
                        checked += 1
        self.assertGreaterEqual(checked, 8 * 5 * 10)

    def test_frozen_bootstrap_conventions(self):
        """seed=1 / samples=10000 are load-bearing for verify_c1_coverage.py."""
        import inspect

        for function in (analyze.paired_bootstrap_ci, analyze.unpaired_bootstrap_ci):
            parameters = inspect.signature(function).parameters
            self.assertEqual(parameters["seed"].default, 1)
            self.assertEqual(parameters["samples"].default, 10000)


class SingleTImplementationTest(unittest.TestCase):
    """B -- the t machinery is the audited one, not a second private copy."""

    def test_t_crit_reproduces_the_published_table(self):
        self.assertAlmostEqual(analyze.t_crit_for(0.05, 5), 2.571, places=3)
        self.assertAlmostEqual(analyze.t_crit_for(0.05, 4), 2.776, places=3)
        self.assertAlmostEqual(analyze.t_crit_for(0.05, 2), 4.303, places=3)
        self.assertAlmostEqual(analyze.t_crit_for(0.01, 4), 4.604, places=3)

    @unittest.skipIf(d1 is None, "d1_predicates.py not importable")
    def test_t_crit_matches_d1_predicates(self):
        for df in (2, 3, 4, 5, 6, 8, 12, 30, 4.7):
            for alpha in (0.05, 0.01, 0.10):
                self.assertEqual(
                    analyze.t_crit_for(alpha, df), d1.t_crit_for(alpha, df)
                )

    @unittest.skipIf(d1 is None, "d1_predicates.py not importable")
    def test_paired_t_ci_matches_d1_predicates(self):
        for n in range(2, 9):
            base, prop = random_pair(n, 31337 + n)
            keys = sorted(base)
            deltas = [prop[k] - base[k] for k in keys]
            # ``1.0 - 0.95`` is 0.050000000000000044, not 0.05, and the
            # bisection is sharp enough to see it (~3 ulp on the endpoint).  Feed
            # d1 the alpha ``paired_t_ci`` actually uses so this compares the
            # ESTIMATORS; alpha=0.05 agreement is asserted separately in
            # ``test_t_crit_matches_d1_predicates``.
            t_crit = d1.t_crit_for(1.0 - 0.95, n - 1)
            lo, hi = d1.paired_t_ci(deltas, 0.95, t_crit)
            got = analyze.paired_t_ci(base, prop)
            self.assertEqual(got["ci95_low"], lo)
            self.assertEqual(got["ci95_high"], hi)
            self.assertEqual(got["t_crit"], t_crit)
            self.assertEqual(got["p_value"], d1.paired_t_p(deltas))
            # and the alpha spelling is immaterial at any reporting scale
            literal = d1.paired_t_ci(deltas, 0.95, d1.t_crit_for(0.05, n - 1))
            self.assertAlmostEqual(got["ci95_low"] / literal[0], 1.0, places=12)

    def test_paired_t_ci_reports_the_required_fields(self):
        base, prop = random_pair(5, 4)
        got = analyze.paired_t_ci(base, prop)
        for key in ("mean_effect", "effect_percent", "ci95_low", "ci95_high",
                    "t_stat", "t_crit", "df", "pairs"):
            self.assertIn(key, got)
        self.assertEqual(got["pairs"], 5.0)
        self.assertEqual(got["df"], 4.0)
        self.assertLess(got["ci95_low"], got["ci95_high"])

    def test_paired_t_ci_shares_the_bootstrap_pairing_convention(self):
        """Same two-Mapping signature AND the same pair set, so the primary and
        the companion can only disagree about width, never about matching."""
        base = {"r0": 10.0, "r1": 11.0, "r2": 9.5, "unmatched_b": 99.0}
        prop = {"r0": 11.0, "r1": 11.4, "r2": 10.2, "unmatched_p": -99.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = analyze.paired_bootstrap_ci(base, prop)
            t_ci = analyze.paired_t_ci(base, prop)
        self.assertEqual(t_ci["pairs"], boot["pairs"])
        self.assertAlmostEqual(t_ci["mean_effect"], boot["mean_effect"])
        self.assertAlmostEqual(t_ci["effect_percent"], boot["effect_percent"])
        with self.assertRaises(ValueError):
            analyze.paired_t_ci({"a": 1.0}, {"a": 2.0})

    def test_unpaired_t_ci_is_welch_not_pooled(self):
        """Unequal variances: Welch's df must be strictly below the pooled df
        (n_a+n_b-2) and the interval must widen accordingly."""
        tight = [10.0, 10.1, 9.9, 10.05, 9.95]
        loose = [12.0, 6.0, 15.0, 4.0, 13.0]
        got = analyze.unpaired_t_ci(tight, loose)
        self.assertLess(got["df"], len(tight) + len(loose) - 2)
        self.assertAlmostEqual(got["df"], 4.0, delta=0.2)
        for key in ("mean_effect", "effect_percent", "ci95_low", "ci95_high",
                    "t_stat", "t_crit", "df", "p_value"):
            self.assertIn(key, got)
        self.assertLess(got["ci95_low"], 0.0)
        self.assertGreater(got["ci95_high"], 0.0)
        with self.assertRaises(ValueError):
            analyze.unpaired_t_ci([1.0], [2.0, 3.0])


class Gate14GuardTest(unittest.TestCase):
    """C -- the guard adds a refusal; it does not remove an ability."""

    def test_small_n_is_marked_ineligible_and_warns(self):
        for n in (3, 4, 5, 6, 8):
            base, prop = random_pair(n, 500 + n)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = analyze.paired_bootstrap_ci(base, prop, samples=200)
            self.assertFalse(
                result["decision_eligible"],
                "n=%d must not be verdict-eligible under gate #14" % n,
            )
            self.assertIn("#14", result["gate14_note"])
            self.assertTrue(
                any(issubclass(w.category, UserWarning) for w in caught),
                "n=%d must warn" % n,
            )

    def test_large_n_is_eligible_and_silent(self):
        base, prop = random_pair(9, 99)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze.paired_bootstrap_ci(base, prop, samples=200)
        self.assertTrue(result["decision_eligible"])
        self.assertEqual(
            [w for w in caught if "gate #14" in str(w.message)], []
        )

    def test_unpaired_guard_uses_the_smaller_arm(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small = analyze.unpaired_bootstrap_ci([1.0] * 9 + [2.0], [1.0, 3.0, 2.0],
                                                  samples=200)
            large = analyze.unpaired_bootstrap_ci([1.0, 2.0] * 6, [1.0, 3.0] * 6,
                                                  samples=200)
        self.assertFalse(small["decision_eligible"])
        self.assertTrue(large["decision_eligible"])

    def test_guard_never_raises_at_the_evidence_generating_n(self):
        """n=3 (tests/test_benchmark_tools.py) and n=5 (verify_c1_coverage.py)
        are gate #14's own evidence generators and must keep computing."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # even promoted to errors...
            warnings.simplefilter("ignore", analyze.Gate14SmallSampleWarning)
            for n in (3, 5):
                base, prop = random_pair(n, n)
                self.assertIn("ci95_low", analyze.paired_bootstrap_ci(base, prop,
                                                                      samples=200))

    def test_docstrings_no_longer_steer_to_the_bootstrap(self):
        module_doc = analyze.__doc__ or ""
        self.assertNotIn("paired bootstrap analysis", module_doc)
        self.assertIn("gate #14", module_doc)
        unpaired_doc = analyze.unpaired_bootstrap_ci.__doc__ or ""
        self.assertNotIn("is preferred whenever repetitions are matched",
                         unpaired_doc)
        for doc in (analyze.paired_bootstrap_ci.__doc__ or "", unpaired_doc):
            self.assertIn("gate #14", doc)
            self.assertIn("COMPANION ONLY", doc)
        paired_doc = analyze.paired_bootstrap_ci.__doc__ or ""
        for coverage in ("0.798", "0.840", "0.859", "0.888"):
            self.assertIn(coverage, paired_doc)


class CliVerdictTest(unittest.TestCase):
    """D -- the CLI verdict is read off the t interval, with the bootstrap kept."""

    # Effects 1.5 / 1.0 / 0.2 on a baseline of ~10: the bootstrap interval
    # cannot contain 0 (every resample of three positive numbers is positive) and
    # the effect is 9%, so the PRE-REPAIR CLI called this a headline.  The paired
    # t interval at df=2 (t_crit 4.303) straddles 0, so the repaired CLI must not.
    ROWS = [
        {"pair_id": "p0", "baseline": "B1", "slo_goodput_req_s": 10.0},
        {"pair_id": "p1", "baseline": "B1", "slo_goodput_req_s": 11.0},
        {"pair_id": "p2", "baseline": "B1", "slo_goodput_req_s": 9.0},
        {"pair_id": "p0", "baseline": "B6", "slo_goodput_req_s": 11.5},
        {"pair_id": "p1", "baseline": "B6", "slo_goodput_req_s": 12.0},
        {"pair_id": "p2", "baseline": "B6", "slo_goodput_req_s": 9.2},
    ]

    def _run_cli(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "paired.json"
            output = Path(directory) / "out.json"
            summary.write_text(json.dumps(rows), encoding="utf-8")
            argv = sys.argv
            sys.argv = [
                "analyze",
                "--paired-summary", str(summary),
                "--output", str(output),
            ]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    analyze.main()
            finally:
                sys.argv = argv
            return json.loads(output.read_text(encoding="utf-8"))

    def test_cli_declares_the_primary_and_keeps_the_bootstrap_alongside(self):
        result = self._run_cli(self.ROWS)
        self.assertEqual(result["primary_estimator"], "paired_t_ci")
        self.assertTrue(result["gate14_compliant"])
        self.assertIn("companion_bootstrap", result)
        self.assertIn("ci95_low", result["companion_bootstrap"])
        self.assertFalse(result["companion_bootstrap"]["decision_eligible"])
        # the top-level interval is the t interval, not the bootstrap one
        self.assertNotEqual(result["ci95_low"],
                            result["companion_bootstrap"]["ci95_low"])
        self.assertIn("t_crit", result)

    def test_cli_headline_follows_the_t_ci_where_the_estimators_disagree(self):
        result = self._run_cli(self.ROWS)
        companion = result["companion_bootstrap"]
        # the disagreement this fixture is built on -- assert it really exists,
        # so the test cannot pass vacuously if the fixture drifts
        self.assertGreater(companion["ci95_low"], 0.0)
        self.assertGreaterEqual(companion["effect_percent"], 3.0)
        self.assertLess(result["ci95_low"], 0.0)
        self.assertGreaterEqual(result["effect_percent"], 3.0)
        self.assertFalse(
            result["headline_improvement"],
            "gate #14: the bootstrap interval may not decide the headline",
        )

    def test_cli_still_reports_a_headline_when_the_t_ci_supports_one(self):
        rows = [dict(row) for row in self.ROWS]
        for row in rows:
            if row["baseline"] == "B6":
                row["slo_goodput_req_s"] += 1.3
        result = self._run_cli(rows)
        self.assertGreater(result["ci95_low"], 0.0)
        self.assertTrue(result["headline_improvement"])


if __name__ == "__main__":
    unittest.main()
