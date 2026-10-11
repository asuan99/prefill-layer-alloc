"""CPU checks for the M2g gate SKELETON's pure comparators (2026-10-12).

Only the decision functions are tested; the GPU tiers have not run.  Each
comparator gets a case it must FAIL on, so a comparator that always returns
PASS (the identity gate of lesson 9) is caught.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m2g_correctness_gate", ENG / "scripts" / "multigpu" / "m2g_correctness_gate.py"
)
M2G = importlib.util.module_from_spec(SPEC)
sys.modules["m2g_correctness_gate"] = M2G
SPEC.loader.exec_module(M2G)
GUARD_PY = ENG / "src" / "multiplex" / "multigpu_guard.py"
CTX_PATCH = ENG / "src" / "patches" / "pdmux_context_device_assert.patch"


def pos(token, lp, top=None):
    return {"token": token, "lp": lp, "top": top or [[token, lp]]}


class TestG1(unittest.TestCase):
    def test_equal_passes(self):
        r = M2G.score_g1({"S00": [1, 2, 3]}, {"S00": [1, 2, 3]})
        self.assertEqual(r["verdict"], "PASS")

    def test_divergence_fails_with_first_index(self):
        r = M2G.score_g1({"S00": [1, 2, 3]}, {"S00": [1, 9, 3]})
        self.assertEqual((r["verdict"], r["first_diff"]["S00"]), ("FAIL", 1))

    def test_missing_or_empty_is_a_failure_not_a_skip(self):
        self.assertEqual(M2G.score_g1({"S00": [1]}, {}), M2G.score_g1({"S00": [1]}, {}))
        self.assertEqual(M2G.score_g1({"S00": [1]}, {})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g1({"S00": []}, {"S00": []})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g1({}, {})["verdict"], "FAIL")

    def test_prefix_is_a_divergence(self):
        r = M2G.score_g1({"S00": [1, 2, 3]}, {"S00": [1, 2]})
        self.assertEqual((r["verdict"], r["first_diff"]["S00"]), ("FAIL", 2))


class TestG2(unittest.TestCase):
    def test_refuses_without_registered_tolerances(self):
        with self.assertRaises(ValueError):
            M2G.score_g2({}, {}, None, 1.0)
        with self.assertRaises(ValueError):
            M2G.score_g2({}, {}, 0.01, None)

    def test_within_tolerance_passes(self):
        ref = {"P0": [pos(5, -0.10, [[5, -0.10], [6, -3.0]])]}
        test = {"P0": [pos(5, -0.105, [[5, -0.105], [6, -3.0]])]}
        self.assertEqual(M2G.score_g2(ref, test, 0.01, 1.0)["verdict"], "PASS")

    def test_outside_tolerance_fails(self):
        ref = {"P0": [pos(5, -0.10)]}
        test = {"P0": [pos(5, -0.30)]}
        self.assertEqual(M2G.score_g2(ref, test, 0.01, 1.0)["verdict"], "FAIL")

    def test_top1_flip_above_margin_fails_even_within_atol(self):
        ref = {"P0": [pos(5, -0.10, [[5, -0.10], [6, -2.0]])]}
        test = {"P0": [pos(5, -0.10, [[6, -0.09], [5, -0.10]])]}
        self.assertEqual(M2G.score_g2(ref, test, 0.5, 1.0)["verdict"], "FAIL")

    def test_top1_flip_below_margin_is_tolerated(self):
        ref = {"P0": [pos(5, -0.70, [[5, -0.70], [6, -0.72]])]}
        test = {"P0": [pos(5, -0.71, [[6, -0.70], [5, -0.71]])]}
        self.assertEqual(M2G.score_g2(ref, test, 0.05, 1.0)["verdict"], "PASS")

    def test_different_teacher_tokens_fail(self):
        self.assertEqual(M2G.score_g2({"P0": [pos(5, -0.1)]}, {"P0": [pos(6, -0.1)]}, 1.0, 1.0)["verdict"], "FAIL")


class TestG0(unittest.TestCase):
    def test_markers_match_the_engine_text(self):
        """The log markers must be substrings of what the engine actually raises."""
        self.assertIn(M2G.GUARD_REFUSAL_TEXT, GUARD_PY.read_text(encoding="utf-8"))
        self.assertIn(M2G.DEVICE_ASSERT_TEXT, CTX_PATCH.read_text(encoding="utf-8"))

    def test_scoring(self):
        logs = {"gen": "RuntimeError: PD-mux with tp_size=2 is only defined ...", "ok": "server ready"}
        self.assertEqual(M2G.score_g0(logs, {"gen": "refused", "ok": "boots"})["verdict"], "PASS")
        self.assertEqual(M2G.score_g0(logs, {"ok": "refused"})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g0(logs, {"gen": "boots"})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g0({}, {"gen": "refused"})["verdict"], "FAIL")

    def test_plan_covers_every_refused_env_and_both_g1_arms(self):
        boots = M2G.plan([1, 2], "m", "c.yml")
        g0 = [b for b in boots if b["tier"] == "G0" and b["tp"] == 2 and b["expect"] == "refused"]
        self.assertEqual(len(g0), len(M2G.G0_REFUSED_ENVS))
        self.assertFalse([b for b in boots if b["tier"] == "G0" and b["tp"] == 1])
        self.assertEqual(len([b for b in boots if b.get("expect") == "device_assert"]), 1)
        g1 = [(b["tier"], b["tp"], b["pdmux"]) for b in boots if b["tier"].startswith("G1")]
        self.assertEqual(sorted(g1), [("G1", 2, False), ("G1", 2, True), ("G1-control", 1, False), ("G1-control", 1, True)])


if __name__ == "__main__":
    unittest.main()
