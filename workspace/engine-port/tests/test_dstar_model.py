"""CPU checks for scripts/costmodel/dstar_model.py (proposal, decision 61).

The model copies measured coefficients by hand; these tests pin them against the descriptive values quoted in
results/cap_char/RESULT_costterms_2026-10-26.md section 2-3 and the T2' table (section 4).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "costmodel"))

import dstar_model as dm  # noqa: E402


class CoefficientTranscription(unittest.TestCase):
    def test_t1_reproduces_quoted_medians(self):
        # (mode, bs, ctx, J-A1 median, J-A2 median) from RESULT_costterms section 2-3
        for mode, b, c, m1, m2 in ((44, 32, 8854, 34.48, 35.26), ("108u", 8, 65536, 20.31, 20.14),
                                   (16, 96, 2150, 180.3, 180.7)):
            pred = dm.t_dec(mode, b, c)
            self.assertLess(abs(pred / ((m1 + m2) / 2) - 1), 0.03, (mode, pred))

    def test_kappa_near_one(self):
        for D in dm.D_GRID:
            self.assertLess(abs(dm.kappa(D) - 1), 0.03, D)

    def test_s_pre_matches_t2prime_at_32k(self):
        for D, (r1, r2) in dm.T2P_32K.items():
            self.assertAlmostEqual(dm.s_pre(32768, D), (r1 + r2) / 2, delta=1e-6 * r1)

    def test_a0_is_analytic_prior(self):
        self.assertAlmostEqual(dm.hardware_layer()["a0_ms_derived"], 8.140, places=2)


class ServingModelShape(unittest.TestCase):
    SUMMARY = {"W1": {"in_mean": 230.0, "out_mean": 286.0, "in_sample": [33, 251, 693] * 50,
                      "out_sample": [251, 286, 585] * 50},
               "W3": {"in_mean": 7105.0, "out_mean": 1007.0, "in_sample": [3000, 6469, 12471] * 50}}

    def test_more_decode_sm_never_lowers_decode_step(self):
        for b in (1, 16, 48):
            steps = [dm.t_dec(D, b, 400) for D in dm.D_GRID]
            self.assertEqual(steps, sorted(steps, reverse=True), b)

    def test_capacity_monotone_in_load_feasibility(self):
        wl = dm.workload(self.SUMMARY, "W1")
        mom = dm.moments(wl["Ls"], 44)
        cap = dm.lam_cap(44, wl, 1.16, mom)["lam_cap"]
        self.assertGreater(cap, 0)
        self.assertFalse(all(dm.state(cap * 1.5, 44, wl, 1.16, mom)["ok"].values()))

    def test_upper_wall_rules_out_d64_for_long_inputs(self):
        self.assertLess(dm.d_window(self.SUMMARY, "W3")["upper_wall_D"], 64)


if __name__ == "__main__":
    unittest.main()
