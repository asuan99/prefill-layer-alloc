"""M2g G5(a): agnostic manual table == auto grid selector (PREREG_MULTIGPU rev6 sec 3 G5(a)).

Deterministic CPU test over decode_bs 0..512 x prefill-in-flight, using the
runtime tree's own `pdmux_context.py` and the verbatim `adjust_stream_groups`
(scripts/multigpu/g5a_selector_equiv.py).  Failure blocks harness sealing.

Lesson 53: each mutant below must be KILLED (verdict FAIL).  Lesson 254: the
unmutated run and a no-op mutator are the control and must PASS, so a FAIL
cannot come from the mutation machinery itself.

The absolute expectation in `test_selector_matches_prereg_formula` comes from
the PREREG text (sec 2: idx = max(1, min(2, decode_bs*2//36)) while a prefill
batch is in flight, sm_counts [(108,0),(74,34),(54,54),(0,108)]), not from the
code under test, so a mutant that changes BOTH paths identically (which the
equivalence check cannot see) is still caught.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g5a_selector_equiv", ENG / "scripts" / "multigpu" / "g5a_selector_equiv.py")
G5A = importlib.util.module_from_spec(SPEC)
sys.modules["g5a_selector_equiv"] = G5A
SPEC.loader.exec_module(G5A)

RUNTIME = G5A.find_runtime()
MIRROR_MIXIN = ENG / "src" / "multiplex" / "multiplexing_mixin.py"


def _runtime_mixin():
    return RUNTIME / "sglang" / "srt" / "multiplex" / "multiplexing_mixin.py"


def _set_row(i, row):
    def mut(cfg):
        cfg.manual_divisions[i] = list(row)
        return cfg
    return mut


class TestG5aRuntimePresent(unittest.TestCase):
    def test_runtime_found_when_declared(self):
        # A skip would read as rc 0 to a Job step that only checks the exit status, so a
        # declared-but-missing runtime is a FAILURE, not a skip.  (Sealing should call
        # scripts/multigpu/g5a_selector_equiv.py, whose rc 2 = runtime not found.)
        if os.environ.get("SGLANG_ENGINE_DEV") or os.environ.get("PDMUX_REQUIRE_RUNTIME") == "1":
            self.assertIsNotNone(RUNTIME, "SGLANG_ENGINE_DEV declared but no pdmux_context.py there")


@unittest.skipIf(RUNTIME is None, "runtime tree (SGLANG_ENGINE_DEV) not found -- G5(a) NOT RUN")
class TestG5aEquivalence(unittest.TestCase):
    def test_config_file_literal(self):
        ctx = G5A.load_pdmux_context(RUNTIME)
        cfg = ctx.load_pdmux_config(str(G5A.MANUAL_CFG))
        self.assertEqual(cfg.sm_group_num, 4)
        self.assertEqual(cfg.manual_divisions, [[74, 34, 0], [54, 54, 36]])
        self.assertEqual(cfg.decode_bs_divisor, 36)
        self.assertEqual(cfg.split_forward_token_budget, 65536)

    def test_auto_grid_is_the_registered_division(self):
        ctx = G5A.load_pdmux_context(RUNTIME)
        self.assertEqual(ctx.divide_sm(108, (8, 0), 2), [(74, 34), (54, 54)])

    def test_control_unmutated_passes_runtime_and_mirror(self):
        for mixin in (_runtime_mixin(), MIRROR_MIXIN):
            with self.subTest(mixin=str(mixin)):
                res = G5A.compare(RUNTIME, mixin)
                self.assertEqual(res["verdict"], "PASS", res)
                self.assertEqual(res["n_states"], 2 * 513)
                self.assertTrue(res["sm_counts_match_registered"])

    def test_control_noop_mutators_pass(self):
        res = G5A.compare(RUNTIME, _runtime_mixin(), manual_mutator=lambda c: c,
                          auto_mutator=lambda c: c, src_mutator=lambda s: s)
        self.assertEqual(res["verdict"], "PASS", res)

    def test_selector_matches_prereg_formula(self):
        res = G5A.run_path(RUNTIME, _runtime_mixin(), G5A.MANUAL_CFG)
        for (bs, pf), (idx, sm) in res["table"].items():
            if bs == 0:
                want = 0
            elif not pf:
                want = 3
            else:
                want = max(1, min(2, bs * 2 // 36))
            self.assertEqual(idx, want, (bs, pf))
            self.assertEqual(sm, G5A.EXPECTED_SM_COUNTS[want])

    # ------------------------------------------------------------------ mutants
    def assertKilled(self, **kw):
        res = G5A.compare(RUNTIME, _runtime_mixin(), **kw)
        self.assertEqual(res["verdict"], "FAIL", "mutant SURVIVED")
        return res

    def test_mutant_threshold_36_to_35_killed(self):
        res = self.assertKilled(manual_mutator=_set_row(1, [54, 54, 35]))
        self.assertEqual(res["n_mismatch"], 1)  # only decode_bs == 35 with prefill in flight
        self.assertEqual(res["first_mismatches"][0]["decode_bs"], 35)

    def test_mutant_threshold_36_to_37_killed(self):
        self.assertKilled(manual_mutator=_set_row(1, [54, 54, 37]))

    def test_mutant_first_threshold_nonzero_killed(self):
        # decode_bs 1 has no qualifying row -> selector raises or diverges; either is a FAIL.
        try:
            self.assertKilled(manual_mutator=_set_row(0, [74, 34, 2]))
        except UnboundLocalError:
            pass

    def test_mutant_auto_divisor_killed(self):
        def mut(cfg):
            cfg.decode_bs_divisor = 35
            return cfg
        self.assertKilled(auto_mutator=mut)

    def test_mutant_row_sm_killed(self):
        res = self.assertKilled(manual_mutator=_set_row(0, [72, 36, 0]))
        self.assertFalse(res["sm_counts_equal"])

    def test_mutant_rows_swapped_killed(self):
        def mut(cfg):
            cfg.manual_divisions = [[54, 54, 0], [74, 34, 36]]
            return cfg
        self.assertKilled(manual_mutator=mut)

    def test_mutant_selector_strict_inequality_killed(self):
        src = lambda s: s.replace("if decode_bs >= threshold:", "if decode_bs > threshold:")
        self.assertKilled(src_mutator=src)

    def test_mutant_selector_source_actually_changed(self):
        # Guard for the mutant above: the target text must exist in the engine source,
        # otherwise the mutant is a silent no-op and "KILLED" would be luck.
        text = G5A.extract_method_source(_runtime_mixin(), "SchedulerMultiplexMixin", "adjust_stream_groups")
        self.assertEqual(text.count("if decode_bs >= threshold:"), 1)


if __name__ == "__main__":
    unittest.main()
