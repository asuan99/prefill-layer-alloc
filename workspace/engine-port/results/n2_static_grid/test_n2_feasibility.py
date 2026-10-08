"""CPU tests for n2_feasibility.py (design calculator, PREREG_N2_3A rev8).
Registered run (bytecode off): PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_feasibility -v
Every test restores the full module state from a deepcopy snapshot (R5-1)."""
import copy
import json
import math
import os
import tempfile
import unittest
import n2_feasibility as f

_STATE = ["K_CO", "COEF", "RHO_MAX_REQUIRED", "L_LAG", "DELTA_L_S", "FIX_S", "C_WINDOW_MIN", "X_CONST",
          "KV_TOKENS", "BOOT_OVERHEAD_S"]


class _Iso(unittest.TestCase):
    def setUp(self):
        self._snap = {n: copy.deepcopy(getattr(f, n)) for n in _STATE}

    def tearDown(self):
        for n, v in self._snap.items():
            cur = getattr(f, n)
            if isinstance(cur, dict):
                cur.clear(); cur.update(copy.deepcopy(v))
            else:
                setattr(f, n, copy.deepcopy(v))


def _model_ms(D, b, c):
    return f.TAU_D44_ALONE_MS[f._kb(b)] * f.CTX_FACTOR[c] * f.D_FACTOR_DESIGN[D]


def n3_like_probe(theta64=5693.0, kappas=(1.3, 1.3), tau_ratio=1.0, outlier=False, extra_kappa=None, tau_extra=None):
    """N3-like synthetic probe aggregates (design recount value 5,693 tok/s at p = 64; cite-forbidden)."""
    sp64 = [theta64 + (i % 5 - 2) * 5 for i in range(25)] + ([3943.0] if outlier else [])
    sp92 = [x * 92 / 64 for x in sp64]
    tc = {cell: {"s_per_median_ms": _model_ms(*cell) * tau_ratio, "window_iters": 128} for cell in f.BINDING_EA_CELLS}
    if tau_extra:
        tc.update(tau_extra)
    kc = {cell: {"c_median_ms": k * 10.0, "a_on_ms": 10.0, "window_iters": 128, "a_on_window_iters": 128}
          for cell, k in zip(f.BINDING_EA_CELLS, kappas)}
    if extra_kappa:
        kc.update(extra_kappa)
    return {"theta_spans": {64: sp64, 92: sp92}, "tau_cells": tc, "kappa_cells": kc,
            "l_lag_iters": [32], "delta_l_s": [0.0], "fix_s": [22.48]}


def dispersed_probe():
    """probe with different min/max across cells and p (pins budget/env/rho directions, R7-4)."""
    sp64 = [5693.0 + (i % 5 - 2) * 5 for i in range(25)]
    sp92 = [x * 92 / 64 * 1.1 for x in sp64]
    tc = {(16, 32, 8192): {"s_per_median_ms": _model_ms(16, 32, 8192) * 1.1, "window_iters": 128},
          (44, 32, 8192): {"s_per_median_ms": _model_ms(44, 32, 8192) * 1.2, "window_iters": 128},
          (16, 8, 2048): {"s_per_median_ms": _model_ms(16, 8, 2048) * 0.8, "window_iters": 128}}
    kc = {cell: {"c_median_ms": c, "a_on_ms": 10.0, "window_iters": 128, "a_on_window_iters": 128}
          for cell, c in (((16, 32, 8192), 13.0), ((44, 32, 8192), 15.0), ((16, 8, 2048), 11.0))}
    return {"theta_spans": {64: sp64, 92: sp92}, "tau_cells": tc, "kappa_cells": kc,
            "l_lag_iters": [30, 40], "delta_l_s": [0.5, 0.2], "fix_s": [20, 22, 30]}


class TestRegistered(_Iso):
    def test_pre_probe_k_and_requirement(self):
        self.assertEqual(f.K_CO, {16: 18, 44: 8})
        self.assertEqual((f.RHO_MAX_REQUIRED, f.C_WINDOW_MIN), (2.0, 128))
        self.assertFalse(hasattr(f, "K_SEARCH_MAX"))            # R6-2

    def test_l_lag_in_o_ealpha(self):
        self.assertEqual(f.L_LAG, 32)
        self.assertEqual(round(f.o_ealpha(44, 8, 2048)), (2048 + 71 - 1903) + 32 + 1038 + 16 + 128 + 0 + 64)

    def test_tau_worst_rule_single(self):
        base = f.tau_alone(16, 32, 8192, "DN")
        self.assertAlmostEqual(f.tau_co(16, 32, 8192, "DW") / base, 2.4)
        self.assertAlmostEqual(f.tau_co(16, 32, 8192, "DN") / base, 1.3)

    def test_x_rule_and_zero_spread(self):
        for D, b, c in [(44, 8, 2048), (16, 32, 8192), (108, 48, 512)]:
            self.assertEqual(f.c_in_of(D, b, c), max(256, int(c - (f.s_hat(D, b, c) + 145))))
        self.assertEqual(f.c_in_of(44, 8, 2048), 2048 - 145)


class TestEalpha(_Iso):
    def test_pair_uses_c_in_of_d(self):
        for d in f.EA_D:
            for b in f.EA_B:
                for c in f.EA_C:
                    self.assertEqual(f.ealpha_c_in(d, b, c), f.c_in_of(d, b, c))

    def test_pinned_job_ea_boot_values(self):
        # R6-4 (1)(7): pins catch call-site c_in(Dalone) and L_LAG deletion in `pre`
        exp = {True: ({"ON16": 548.303, "ON44": 541.47, "OFF16": 472.871, "OFF44": 501.554},
                      {"ON16": 830.935, "ON44": 816.906, "OFF16": 715.772, "OFF44": 756.007}),
               False: ({"ON16": 732.454, "ON44": 751.846, "OFF16": 559.192, "OFF44": 659.336},
                       {"ON16": 1113.596, "ON44": 1134.905, "OFF16": 848.269, "OFF44": 994.507})}
        for cancel, (nom, worst) in exp.items():
            r = f.job_budget(f.boots_ea_job(), cancel, 0)
            for k, v in nom.items():
                self.assertAlmostEqual(r["nom"]["per_boot"][k], v, places=2)
            for k, v in worst.items():
                self.assertAlmostEqual(r["worst"]["per_boot"][k], v, places=2)

    def test_delta_window_tau_and_call_site(self):
        # R5-5 + R6-4 (2): OFF cancel time grows by (delta(108) - 8) * tau_alone(108)
        base = f.cell_time_ealpha(108, 16, 32, 8192, "nom", True)
        f.DELTA_L_S = 0.5
        d108, d16 = f.delta_tokens(108, 32, 8192), f.delta_tokens(16, 32, 8192)
        self.assertGreater(d108, d16)
        grown = f.cell_time_ealpha(108, 16, 32, 8192, "nom", True)
        self.assertAlmostEqual(grown - base, (d108 - 8) * f.tau_alone(108, 32, 8192, "DN", "budget"), places=6)


class TestRhoTable(_Iso):
    def test_registered_k_passes(self):
        rows = {r["cell"]: r for r in f.rho_table()}
        self.assertTrue(all(r["passes"] for r in rows.values()))
        self.assertAlmostEqual(rows[(16, 32, 8192)]["rho_ub_worst"], 2.12, delta=0.01)

    def test_k8_reproduces_d8(self):
        f.K_CO[16] = 8
        self.assertAlmostEqual(f.w_co(16, 32, 8192, "PN", "DN"), 139, delta=1)
        self.assertLess(f.w_co(16, 32, 8192, "PN", "DW"), 128)

    def test_k16_reproduces_188(self):
        """DESIGN-COEFFICIENT basis only (R6-6): K=16 was rejected at 1.885 under design coefficients;
        the post-probe U-9 rule may pick a K below 18 (or below 16) under calibrated coefficients."""
        f.K_CO[16] = 16
        r = {x["cell"]: x for x in f.rho_table()}[(16, 32, 8192)]
        self.assertAlmostEqual(r["rho_ub_worst"], 1.885, delta=0.005)
        self.assertFalse(r["passes"])

    def test_span_structure(self):
        sc = f.limits_check()["span_checks"]
        self.assertEqual(sc[16], dict(K=18, n_new_max=2, spans_min=9, spans_max=18, sum_n_new=18))
        self.assertEqual(sc[44], dict(K=8, n_new_max=2, spans_min=4, spans_max=8, sum_n_new=8))


class TestKRule(_Iso):
    def test_design_coefficients(self):
        self.assertEqual((f.choose_k(16), f.choose_k(44)), (17, 8))
        self.assertTrue(f.is_minimal_k(16, 17) and f.is_minimal_k(44, 8))

    def test_nonminimal_chooser_caught(self):
        self.assertFalse(f.is_minimal_k(16, f.choose_k(16) + 1))

    def test_search_beyond_64(self):
        f.apply_probe(n3_like_probe(kappas=(3.0, 3.0), tau_ratio=2.0))   # R6-2: no artificial cap
        self.assertEqual(f.choose_k(16), 80)

    def test_w_co_condition_load(self):
        # R6-4 (5): kappa_DN = 2.0 at design theta -> (21, 10); without the W_co condition it would be (17, 8)
        f.apply_probe(n3_like_probe(theta64=4500.0, kappas=(2.0, 2.0)))
        self.assertEqual((f.choose_k(16), f.choose_k(44)), (21, 10))


class TestProbeMapping(_Iso):
    def test_n3_like_gives_22_10(self):
        f.apply_probe(n3_like_probe())
        self.assertEqual((f.choose_k(16), f.choose_k(44)), (22, 10))

    def test_outlier_span_does_not_move_k(self):
        f.apply_probe(n3_like_probe(outlier=True))                     # D-9: median, not min
        self.assertEqual((f.choose_k(16), f.choose_k(44)), (22, 10))

    def test_reject_too_few_spans_and_missing_p(self):
        p = n3_like_probe(); p["theta_spans"][64] = p["theta_spans"][64][:19]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)
        p = n3_like_probe(); del p["theta_spans"][92]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)

    def test_reject_iteration_level_raw(self):
        p = n3_like_probe(); p["tau_cells"][(16, 32, 8192)]["s_per_median_ms"] = [50.0, 51.0, 900.0]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)
        p = n3_like_probe(); p["kappa_cells"][(44, 32, 8192)]["c_median_ms"] = [18.8, 1366.8]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)

    def test_reject_missing_binding_cell_and_short_window(self):
        p = n3_like_probe(); del p["kappa_cells"][(16, 32, 8192)]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)
        p = n3_like_probe(); p["tau_cells"][(44, 32, 8192)]["window_iters"] = 64
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)

    def test_reject_d108_tau_cell(self):
        p = n3_like_probe(tau_extra={(108, 32, 8192): {"s_per_median_ms": 50.0, "window_iters": 128}})
        with self.assertRaises(f.ProbeRejected):                           # R6-3
            f.apply_probe(p)

    def test_tau_direction_max_min_and_s_other(self):
        extra = {(16, 8, 2048): {"s_per_median_ms": _model_ms(16, 8, 2048) * 0.9, "window_iters": 128},
                 (44, 8, 2048): {"s_per_median_ms": _model_ms(44, 8, 2048) * 1.2, "window_iters": 128}}
        p = n3_like_probe(tau_ratio=1.1, tau_extra=extra)
        f.apply_probe(p)
        self.assertAlmostEqual(f.COEF["rho"]["d_factor"][16], 1.6 * 1.1)       # max (R6-4 (4))
        self.assertAlmostEqual(f.COEF["env"]["d_factor"][16], 1.6 * 0.9)       # min (R6-5)
        self.assertAlmostEqual(f.COEF["rho"]["d_factor"][24], 1.35 * 1.2)      # max(s16, s44) (R6-3 / R6-4 (3))
        self.assertAlmostEqual(f.COEF["env"]["d_factor"][108], 0.75 * 0.9)     # min(s16, s44)

    def test_kappa_direction_three_cells(self):
        extra = {(16, 8, 2048): {"c_median_ms": 15.0, "a_on_ms": 10.0, "window_iters": 128, "a_on_window_iters": 128}}
        f.apply_probe(n3_like_probe(kappas=(1.2, 1.3), extra_kappa=extra))     # elements 1.2, 1.3, 1.5
        self.assertAlmostEqual(f.COEF["rho"]["kappa_dn"], 1.5)                   # max (R6-4 (6))
        self.assertAlmostEqual(f.COEF["env"]["kappa_dn"], 1.2)
        self.assertAlmostEqual(f.kappa("DW", "rho"), 1.5 * 1.6 / 1.3)

    def test_scalars_maxima_and_overhead_unchanged(self):
        oh = dict(f.BOOT_OVERHEAD_S)
        p = n3_like_probe(); p.update({"l_lag_iters": [10, 41.2], "delta_l_s": [0.01, 0.03], "fix_s": [10, 12, 30]})
        f.apply_probe(p)
        self.assertEqual((f.L_LAG, f.DELTA_L_S, f.FIX_S["nom"], f.FIX_S["worst"]), (42, 0.03, 12, 30))
        self.assertEqual(f.BOOT_OVERHEAD_S, oh)


class TestDirections(_Iso):
    def _probe_with_spread(self):
        extra = {(16, 8, 2048): {"s_per_median_ms": _model_ms(16, 8, 2048) * 0.8, "window_iters": 128}}
        extra_k = {(16, 8, 2048): {"c_median_ms": 11.0, "a_on_ms": 10.0, "window_iters": 128, "a_on_window_iters": 128}}
        sp = n3_like_probe(tau_ratio=1.1, tau_extra=extra, extra_kappa=extra_k)
        sp["theta_spans"][92] = [x * 1.1 for x in sp["theta_spans"][92]]   # p=92 faster -> theta rho/env differ
        return sp

    def test_theta_cross_p_direction(self):
        f.apply_probe(self._probe_with_spread())
        self.assertAlmostEqual(f.COEF["rho"]["theta"], 5693.0 * 1.1, places=3)     # MAX over p (p=92 faster)
        self.assertAlmostEqual(f.COEF["env"]["theta"], 5693.0, places=3)           # MIN over p
        self.assertAlmostEqual(f.COEF["budget"]["theta"], 5693.0, places=3)

    def test_s_hat_uses_env_direction(self):
        f.apply_probe(self._probe_with_spread())
        env = f.s_hat(16, 32, 8192)
        rho = max(f.burst_spread(16, 32, 8192, pc, dc, "rho") for pc, dc in f.CORNERS)
        self.assertGreater(env, rho)                     # R6-5: envelope from slow-prefill / fast-decode set

    def test_n_co_max_uses_env_direction(self):
        f.apply_probe(self._probe_with_spread())
        rho = math.ceil(max(f.w_co(16, 32, 8192, pc, dc, "rho") for pc, dc in f.CORNERS))
        self.assertGreater(f.n_co_max(16, 32, 8192), rho)


class TestRev8(_Iso):
    def test_budget_direction_pins(self):                       # R7-4 A01/A02
        f.apply_probe(dispersed_probe())
        self.assertAlmostEqual(f.COEF["budget"]["theta"], 5693.0, places=3)
        self.assertAlmostEqual(f.COEF["budget"]["kappa_dn"], 1.5)
        self.assertAlmostEqual(f.COEF["budget"]["d_factor"][16], 1.6 * 1.1)
        self.assertAlmostEqual(f.COEF["budget"]["d_factor"][24], 1.35 * 1.2)

    def test_worst_boot_cost_pins_dispersed(self):              # R7-4 A04/A05
        f.apply_probe(dispersed_probe())
        ea = f.job_budget(f.boots_ea_job(), True, 0)["worst"]["per_boot"]
        pr = f.job_budget(f.boots_profile_job(), True, 5)["worst"]["per_boot"]
        for k, v in {"ON16": 823.54, "ON44": 767.153, "OFF16": 685.447, "OFF44": 704.432}.items():
            self.assertAlmostEqual(ea[k], v, places=2)
        for k, v in {"ON16": 1301.948, "ON24": 1371.863, "ON44": 1293.451, "OFF16": 1434.868, "OBS44": 346.958}.items():
            self.assertAlmostEqual(pr[k], v, places=2)

    def test_exec_delta_env_budget_delta_budget(self):          # R7-5 / A13
        f.apply_probe(dispersed_probe())
        self.assertEqual((f.delta_exec(16, 32, 8192), f.delta_exec(108, 32, 8192)), (12, 24))
        self.assertEqual(f.delta_tokens(108, 32, 8192), 16)

    def test_reject_missing_keys(self):                          # R7-6
        for key in f.REQUIRED_PROBE_KEYS:
            p = n3_like_probe(); del p[key]
            with self.assertRaises(f.ProbeRejected):
                f.apply_probe(p)

    def test_reject_bad_values(self):                            # R7-6 incl. negative theta (no infinite loop)
        for mut in ("neg_theta", "zero_tau", "nan_kappa", "p7", "kappa_d108", "tau_b_off_grid", "empty_fix"):
            p = n3_like_probe()
            if mut == "neg_theta":
                p["theta_spans"][64] = [-5693.0] * 25
            elif mut == "zero_tau":
                p["tau_cells"][(16, 32, 8192)]["s_per_median_ms"] = 0.0
            elif mut == "nan_kappa":
                p["kappa_cells"][(44, 32, 8192)]["c_median_ms"] = float("nan")
            elif mut == "p7":
                p["theta_spans"][7] = [100.0] * 25
            elif mut == "kappa_d108":
                p["kappa_cells"][(108, 32, 8192)] = {"c_median_ms": 13.0, "a_on_ms": 10.0, "window_iters": 128,
                                                     "a_on_window_iters": 128}
            elif mut == "tau_b_off_grid":
                p["tau_cells"][(16, 30, 8192)] = {"s_per_median_ms": 50.0, "window_iters": 128}
            elif mut == "empty_fix":
                p["fix_s"] = []
            with self.assertRaises(f.ProbeRejected, msg=mut):
                f.apply_probe(p)

    def test_signed_delta_l_allowed(self):
        p = n3_like_probe(); p["delta_l_s"] = [-0.2, -0.1]
        f.apply_probe(p)
        self.assertEqual(f.DELTA_L_S, 0.0)

    def test_reject_missing_tau_binding_and_p64_and_short_a_on(self):   # R7-4 A06/A08/A07
        # another D44 tau cell stays, so ONLY the binding-cell rule can reject (R7-4 A06)
        p = n3_like_probe(tau_extra={(44, 8, 2048): {"s_per_median_ms": _model_ms(44, 8, 2048), "window_iters": 128}})
        del p["tau_cells"][(44, 32, 8192)]
        with self.assertRaisesRegex(f.ProbeRejected, "binding cell"):
            f.apply_probe(p)
        p = n3_like_probe(); del p["theta_spans"][64]
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)
        p = n3_like_probe(); p["kappa_cells"][(16, 32, 8192)]["a_on_window_iters"] = 64
        with self.assertRaises(f.ProbeRejected):
            f.apply_probe(p)

    def test_gate_derived_constants_post_probe(self):           # R7-2: not design-coefficient recompute
        design_oea = f.o_ealpha(16, 32, 8192)
        g = f.post_probe_gate(dispersed_probe())
        self.assertTrue(g["seal_allowed"])
        d = g["derived"]["ealpha"]["16,32,8192"]
        self.assertEqual(d["K"], g["K"][16])
        self.assertAlmostEqual(d["O_EA"], f.o_ealpha(16, 32, 8192))
        self.assertNotAlmostEqual(d["O_EA"], design_oea, places=0)
        self.assertEqual(g["derived"]["L_LAG"], 40)
        self.assertEqual(len(g["derived"]["profile"]), 60)


class TestSealGate(_Iso):
    def test_gate_ok(self):
        g = f.post_probe_gate(n3_like_probe())
        self.assertTrue(g["seal_allowed"])
        self.assertEqual(g["K"], {16: 22, 44: 10})

    def test_gate_limits_fail_profile(self):
        g = f.post_probe_gate(n3_like_probe(theta64=4500.0, tau_ratio=0.5))     # R6-7: PROFILE longest 16,617
        self.assertFalse(g["seal_allowed"])
        self.assertTrue(any("limits_check failed" in r for r in g["reasons"]))

    def test_gate_no_k(self):
        g = f.post_probe_gate(n3_like_probe(theta64=3600.0, tau_ratio=0.5))
        self.assertFalse(g["seal_allowed"])
        self.assertIsNone(g["K"][44])

    def test_cli_writes_single_k_source(self):
        p = n3_like_probe()
        raw = {k: ({json.dumps(list(kk)) if isinstance(kk, tuple) else str(kk): vv for kk, vv in v.items()}
                   if isinstance(v, dict) else v) for k, v in p.items()}
        with tempfile.TemporaryDirectory() as td:
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "K_POST_PROBE.json")
            with open(pi, "w") as fh:
                json.dump(raw, fh)
            rc = f.main(["--probe", pi, "--out", po])
            with open(po) as fh:
                rec = json.load(fh)
            psha = f._sha256_file(pi)
        self.assertEqual(rc, 0)
        self.assertEqual(rec["K"], {"16": 22, "44": 10})
        self.assertEqual(rec["probe_sha256"], psha)                                   # R7-4 A10
        self.assertEqual(rec["calculator_sha256"], f._sha256_file(f.__file__))
        self.assertNotEqual(rec["probe_sha256"], rec["calculator_sha256"])
        self.assertIn("derived", rec)

    def test_cli_rc2_when_not_sealable(self):                                         # R7-4 A11
        p = n3_like_probe(theta64=4500.0, tau_ratio=0.5)
        raw = {k: ({json.dumps(list(kk)) if isinstance(kk, tuple) else str(kk): vv for kk, vv in v.items()}
                   if isinstance(v, dict) else v) for k, v in p.items()}
        with tempfile.TemporaryDirectory() as td:
            pi, po = os.path.join(td, "probe.json"), os.path.join(td, "out.json")
            with open(pi, "w") as fh:
                json.dump(raw, fh)
            self.assertEqual(f.main(["--probe", pi, "--out", po]), 2)

    def test_cli_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as td:
            pi = os.path.join(td, "probe.json")
            with open(pi, "w") as fh:
                fh.write('{"l_lag_iters": [1], "l_lag_iters": [2]}')
            with self.assertRaises(f.ProbeRejected):
                f.main(["--probe", pi, "--out", os.path.join(td, "o.json")])


class TestBudget(_Iso):
    def test_cap_rule_and_counts(self):
        for boots, probes, cells in ((f.boots_profile_job(), 5, 296), (f.boots_ea_job(), 0, 64)):
            for cancel in (False, True):
                r = f.job_budget(boots, cancel, probes)
                self.assertEqual(r["cap"], math.ceil(1.1 * r["worst"]["section"] / 100) * 100)
                self.assertEqual(r["nom"]["cells"], cells)

    def test_limits(self):
        lim = f.limits_check()
        self.assertTrue(lim["ok"])
        self.assertEqual(lim["max_len_cell"], ("PROFILE", 108, 32, 8192))


if __name__ == "__main__":
    unittest.main()
