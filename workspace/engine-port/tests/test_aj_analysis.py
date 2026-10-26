"""CPU tests of scripts/costmodel/aj_analysis.py (PREREG rev3.6' §1-9 AJ-1 ... AJ-4, §12 item 25).

What this proves: on FAKE fit results / fake analysis.json dicts, every §1-9 label flips on the registered side of its
threshold, the NA paths (NA_FORM, NA_POINTS, W-INDEP gating, THETA_SINGLE_JOB) fire, Delta_ideal is computed from
analytic_terms, and every analysis.json field the module reads exists in the real smoke analysis.json / harness source.
What it does NOT prove: anything about measured data (none existed at commit time).  The fake numbers are invented.
"""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ENG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENG / "scripts" / "costmodel"))
import aj_analysis as AJ  # noqa: E402

SMOKE = ENG / "results" / "cap_char" / "costterms_smoke_20261026T054338Z" / "summary"
HARNESS = ENG / "results" / "cap_char" / "costterms_harness"
D_REAL = {"D16": 16, "D24": 24, "D34": 36, "D44": 44, "D64": 64}   # D34 realized 36 (4-SM granularity), fake


def fit_entry(a, b=0.2, c=1e-5, label=None):
    return {"F1": {"coef": [a, b, c], "form": "F1"}, "chosen": {"form": "F1"}, "label": label}


def ja_analysis(job, a_by_mode, rnd="r1", tag="real_a4", t2p=None, t1_1x2k=0.5, fit_labels=None, gpu_scale=None,
                b_by_mode=None, c_by_mode=None):
    """Fake J-A analysis.json.  T1 cells are generated EXACTLY from F1 (a, b, c) so the gpu refit recovers a."""
    tables, fits = {"t1": {}, "t2prime": t2p or {}}, {"t1": {}}
    cells = [(1, 2048), (8, 2048), (8, 8192), (32, 2048), (32, 8192), (48, 32768)]
    for mode, a in a_by_mode.items():
        key = f"{job}|{rnd}|{tag}|{mode}|cap128"
        b = (b_by_mode or {}).get(mode, 0.2)
        c = (c_by_mode or {}).get(mode, 1e-5)
        g = (gpu_scale or {}).get(mode, 1.0)
        cv = {f"{bs}x{cc}": a + b * bs + c * bs * cc for bs, cc in cells}
        cv[AJ.LAG_T1_CELL] = t1_1x2k
        tables["t1"][key] = {"D_real": D_REAL.get(mode, 108), "cells": cv,
                             "ctx_actual": {f"{bs}x{cc}": float(cc) for bs, cc in cells},
                             "gpu_ms": {f"{bs}x{cc}": g * a + b * bs + c * bs * cc for bs, cc in cells}}
        fits["t1"][key] = fit_entry(a, b, c, (fit_labels or {}).get(mode))
    return {"tables": tables, "fits": fits, "smoke": False}


def a_inv(t0=1.0, kappa=200.0, sat=None):
    """a_D = t0 + kappa / min(D, D_sat) on the realized D axis (bandwidth saturates at D_sat); 108u from the same law."""
    f = lambda d: t0 + kappa / min(d, sat or math.inf)   # noqa: E731
    out = {m: f(d) for m, d in D_REAL.items()}
    out["108u"] = f(108)
    return out


def t2p_tab(job, theta, rnd="r1", tag="real_a4", drop=None, unconfirm=None, T0=1000.0):
    P = {"D16": 92, "D24": 84, "D34": 72, "D44": 64, "D64": 44}
    tab = {}
    for m, p in P.items():
        pts = {str(L): T0 * (L / 32768) * (92 / p) ** theta for L in AJ.T2P_L_GRID}
        conf = dict(pts)
        for L in (unconfirm or {}).get(m, []):
            conf.pop(str(L), None)
        if drop == m:
            continue
        tab[f"{job}|{rnd}|{tag}|{m}"] = {"P": p, "points": pts, "points_confirmed": conf}
    return tab


class TestAJ1(unittest.TestCase):
    def test_inv_d_threshold(self):
        self.assertEqual(AJ.label_inv_d(0.03, 0.0), "BWD_INV_D_HELD_LE64")
        self.assertEqual(AJ.label_inv_d(0.0301, 0.0), "BWD_INV_D_MISS_LE64")
        self.assertEqual(AJ.label_inv_d(0.0, -1e-9), "BWD_INV_D_MISS_LE64")

    def test_s108_thresholds(self):
        self.assertEqual(AJ.label_s108(0.0329), "BWD_FLAT_BY_64")
        self.assertEqual(AJ.label_s108(0.033), "BWD_KNEE_64_108")
        self.assertEqual(AJ.label_s108(0.1009), "BWD_KNEE_64_108")
        self.assertEqual(AJ.label_s108(0.101), "BWD_NO_KNEE_TO_108")

    def test_exact_inv_d_held_and_no_knee(self):
        r = AJ.aj1_job(ja_analysis("J-A1", a_inv()), "J-A1")["J-A1|r1|real_a4"]
        self.assertEqual(r["S_eng"]["i"]["label"], "BWD_INV_D_HELD_LE64")
        self.assertAlmostEqual(r["S_eng"]["i"]["t0_ms"], 1.0, places=6)
        self.assertAlmostEqual(r["S_eng"]["i"]["kappa_ms_sm"], 200.0, places=4)
        s = (200 / 64 - 200 / 108) / (200 / 16 - 200 / 64)          # 1/D to 108: matches the 0.136 derived prior
        self.assertAlmostEqual(r["S_eng"]["ii"]["s_108"], s, places=9)
        self.assertAlmostEqual(s, 0.1358, places=3)
        self.assertEqual(r["S_eng"]["ii"]["label"], "BWD_NO_KNEE_TO_108")
        self.assertAlmostEqual(r["S_eng"]["iv"]["r_a"], (1 + 200 / 16) / (1 + 200 / 64))
        self.assertIsNone(r["estimator_split"]["i"]["label"])
        self.assertTrue(r["S_eng_label_citable"])

    def test_saturation_miss_and_flat(self):
        r = AJ.aj1_job(ja_analysis("J-A1", a_inv(t0=0.0, sat=34)), "J-A1")["J-A1|r1|real_a4"]
        self.assertEqual(r["S_eng"]["i"]["label"], "BWD_INV_D_MISS_LE64")
        self.assertGreater(r["S_eng"]["i"]["max_rel_resid"], 0.03)
        self.assertEqual(r["S_eng"]["ii"]["label"], "BWD_FLAT_BY_64")       # D_sat <= 64 -> s = 0

    def test_knee_label_from_dsat80(self):
        r = AJ.aj1_job(ja_analysis("J-A1", a_inv(t0=0.0, sat=80)), "J-A1")["J-A1|r1|real_a4"]
        self.assertAlmostEqual(r["S_eng"]["ii"]["s_108"], 0.067, places=3)
        self.assertEqual(r["S_eng"]["ii"]["label"], "BWD_KNEE_64_108")

    def test_na_form_and_na_points(self):
        A = ja_analysis("J-A1", a_inv(), fit_labels={"D24": "FORM_INADEQUATE_T1_D24"})
        r = AJ.aj1_job(A, "J-A1")["J-A1|r1|real_a4"]
        self.assertEqual(r["S_eng"]["i"]["label"], "NA_FORM")
        self.assertEqual(r["S_eng"]["ii"]["label"], "BWD_NO_KNEE_TO_108")    # (ii) does not use D24
        a = a_inv()
        a.pop("D64")
        r = AJ.aj1_job(ja_analysis("J-A1", a), "J-A1")["J-A1|r1|real_a4"]
        self.assertEqual(r["S_eng"]["i"]["label"], "NA_POINTS")
        self.assertEqual(r["S_eng"]["ii"]["label"], "NA_POINTS")
        self.assertIsNone(r["estimator_split"]["i"]["label"])

    def test_estimator_split(self):
        # gpu estimator: a_D saturates (fixed part inflated only at high D) -> different (i) label
        sat = a_inv(sat=34)
        base = a_inv()
        A = ja_analysis("J-A1", base, gpu_scale={m: sat[m] / base[m] for m in base})
        r = AJ.aj1_job(A, "J-A1")["J-A1|r1|real_a4"]
        self.assertEqual(r["gpu_ms"]["i"]["label"], "BWD_INV_D_MISS_LE64")
        self.assertEqual(r["S_eng"]["i"]["label"], "BWD_INV_D_HELD_LE64")
        self.assertEqual(r["estimator_split"]["i"]["label"], "BWD_ESTIMATOR_SPLIT")
        self.assertFalse(r["S_eng_label_citable"])

    def test_cross_job(self):
        g1 = AJ.aj1_job(ja_analysis("J-A1", a_inv()), "J-A1")["J-A1|r1|real_a4"]
        g2 = AJ.aj1_job(ja_analysis("J-A2", a_inv(t0=0.0, sat=34), rnd="r2"), "J-A2")["J-A2|r2|real_a4"]
        c = AJ.aj1_cross(g1, g2)
        self.assertEqual(c["i"]["label"], "BWD_UNRESOLVED")
        self.assertEqual(c["ii"]["label"], "BWD_UNRESOLVED")
        self.assertIsNone(AJ.aj1_cross(g1, copy.deepcopy(g1))["i"]["label"])
        self.assertIsNone(AJ.aj1_cross(g1, None)["i"]["label"])


class TestAJ2(unittest.TestCase):
    def test_prior_and_cknee(self):
        self.assertAlmostEqual(AJ.knee_prior_tokens(), 17523, delta=1)
        A = ja_analysis("J-A1", a_inv(), b_by_mode={"D44": 0.1}, c_by_mode={"D44": 1e-5})
        row = AJ.aj2_job(A, "J-A1")["J-A1|r1|real_a4"]["rows"]["D44"]
        self.assertAlmostEqual(row["c_knee_tokens"], 10000.0)
        self.assertAlmostEqual(row["ratio_to_prior"], 10000.0 / AJ.knee_prior_tokens())

    def test_unstable_threshold(self):
        self.assertFalse(AJ.coef_unstable(1.0, 1.5))
        self.assertTrue(AJ.coef_unstable(1.0, 1.5001))
        self.assertTrue(AJ.coef_unstable(1.0, -1.0))
        self.assertTrue(AJ.coef_unstable(0.0, 1.0))
        r1 = AJ.aj2_job(ja_analysis("J-A1", a_inv()), "J-A1")["J-A1|r1|real_a4"]
        r2 = AJ.aj2_job(ja_analysis("J-A2", a_inv(), rnd="r2", c_by_mode={"D16": 2e-5}), "J-A2")["J-A2|r2|real_a4"]
        cr = AJ.aj2_cross(r1, r2)
        self.assertEqual(cr["D16"]["tag"], "KNEE_COEF_UNSTABLE")      # c doubled: 100 % > 50 %
        self.assertIsNone(cr["D24"]["tag"])

    def test_na_form(self):
        A = ja_analysis("J-A1", a_inv(), fit_labels={"D16": "FORM_INADEQUATE_T1_D16"})
        self.assertEqual(AJ.aj2_job(A, "J-A1")["J-A1|r1|real_a4"]["rows"]["D16"], {"status": "NA_FORM"})


class TestAJ3(unittest.TestCase):
    def run_job(self, theta, **kw):
        A = ja_analysis("J-A1", a_inv(), t2p=t2p_tab("J-A1", theta, **kw))
        return AJ.aj3_job(A, "J-A1")["J-A1|r1|real_a4"]

    def test_theta_boundaries(self):
        self.assertEqual(AJ.label_theta(0.9), "T2P_COEXIST_EXP_LINEAR_SCOPED")
        self.assertEqual(AJ.label_theta(1.1), "T2P_COEXIST_EXP_LINEAR_SCOPED")
        self.assertEqual(AJ.label_theta(0.8999), "T2P_COEXIST_EXP_SUB_SCOPED")
        self.assertEqual(AJ.label_theta(1.1001), "T2P_COEXIST_EXP_SUPER_SCOPED")

    def test_recovers_theta(self):
        for th, lab in ((1.0, "LINEAR"), (0.85, "SUB"), (1.2, "SUPER")):
            r = self.run_job(th)
            self.assertAlmostEqual(r["theta_prime"], th, places=9)
            self.assertEqual(r["label"], f"T2P_COEXIST_EXP_{lab}_SCOPED")
            self.assertEqual(r["n_points"], 8)
            self.assertAlmostEqual(r["per_L_descriptive"]["1024"]["theta_origin"], th, places=9)

    def test_na_points(self):
        self.assertEqual(self.run_job(1.0, drop="D44")["status"], "NA_POINTS")
        r = self.run_job(1.0, unconfirm={"D16": [65536]})                 # reference unconfirmed at one L
        self.assertEqual(r["status"], "NA_POINTS")
        self.assertIsNone(r["label"])
        self.assertIn("D16@L65536", r["missing"])

    def test_lag_bound(self):
        # T2'(32k, D16) = 1000 ms -> lag 2 x 15 / 1000 = 3.0 % (not > 3 %), 15.01 -> exceeded
        for s, lab in ((15.0, "T2P_COEXIST_EXP_LINEAR_SCOPED"), (15.01, None)):
            A = ja_analysis("J-A1", a_inv(), t2p=t2p_tab("J-A1", 1.0), t1_1x2k=s)
            r = AJ.aj3_job(A, "J-A1")["J-A1|r1|real_a4"]
            self.assertEqual(r["label"], lab)
            if lab is None:
                self.assertEqual(r["status"], "THETA_LAG_BOUND_EXCEEDED")

    def test_lag_missing_withholds(self):
        A = ja_analysis("J-A1", a_inv(), t2p=t2p_tab("J-A1", 1.0))
        for k in A["tables"]["t1"]:
            A["tables"]["t1"][k]["cells"].pop(AJ.LAG_T1_CELL)
        r = AJ.aj3_job(A, "J-A1")["J-A1|r1|real_a4"]
        self.assertIsNone(r["label"])
        self.assertEqual(r["status"], "LAG_NA")

    def test_cross(self):
        lin, sup = self.run_job(1.0), self.run_job(1.2)
        self.assertEqual(AJ.aj3_cross(lin, sup)["label"], "THETA_UNRESOLVED")
        same = AJ.aj3_cross(lin, copy.deepcopy(lin))
        self.assertIsNone(same["label"])
        self.assertFalse(same["pooled"])
        na = self.run_job(1.0, drop="D24")
        self.assertEqual(AJ.aj3_cross(lin, na)["label"], "THETA_SINGLE_JOB")
        self.assertIsNone(AJ.aj3_cross(na, na)["label"])


def jh1_analysis(t1_fn, t2_fn, w1="WEIGHT_INDEP_HELD", w2="WEIGHT_INDEP_HELD", drop=None):
    t1, t2 = {}, {}
    for n in AJ.VARIANTS:
        cells = {f"{bs}x{c}": t1_fn(n, bs, c) for bs, c in AJ.JH_COMMON_CELLS}
        if drop and drop[0] == n:
            cells.pop(drop[1], None)
        t1[f"J-H1|r1|dummy_a{n}|108v|cap128"] = {"D_real": 108, "cells": cells,
                                                  "ctx_actual": {f"{bs}x{c}": float(c) for bs, c in AJ.JH_COMMON_CELLS}}
        t2[f"J-H1|r1|dummy_a{n}|108v"] = {"points": {str(L): t2_fn(n, L) for L in AJ.JH_T2_L}}
    return {"tables": {"t1": t1, "t2": t2}, "jh1": {"w_indep": {"T1": {"label": w1}, "T2": {"label": w2}}}}


class TestAJ4(unittest.TestCase):
    T1 = staticmethod(lambda n, bs, c: 10 + 0.3 * n + bs * (0.05 + 0.002 * (31 - n)) + bs * c * 1e-6 * n)
    T2 = staticmethod(lambda n, L: 100 + 2.0 * n + L * 0.01)

    def test_exact_regression_and_ideal(self):
        r = AJ.aj4(jh1_analysis(self.T1, self.T2))
        t2 = r["T2"]["16384"]
        self.assertEqual(t2["status"], "OK")
        self.assertAlmostEqual(t2["beta1"], 2.0)
        self.assertAlmostEqual(t2["r2"], 1.0)
        d = (AJ.ideal_prefill_ms(31, 16384) - AJ.ideal_prefill_ms(4, 16384)) / 27      # linear in n_A
        self.assertAlmostEqual(t2["delta_ideal_ms"], d, places=12)
        self.assertAlmostEqual(t2["ratio_beta1_to_ideal"], 2.0 / d)
        t1 = r["T1"]["8x2048"]
        self.assertTrue(t1["level_includes_weight_term"])
        self.assertAlmostEqual(t1["beta1"], 0.3 - 8 * 0.002 + 8 * 2048 * 1e-6)
        dd = r["T1_ctx_diff_of_diff"]["8x8192-8x2048"]
        self.assertAlmostEqual(dd["beta1_diff"], 8 * (8192 - 2048) * 1e-6)
        self.assertIn("memory", t1["delta_ideal_ms"])
        for n in AJ.VARIANTS:                     # roofline = max(compute, memory); bs 8 decode is memory-bound
            tot, comp, mem = AJ.ideal_decode_ms(n, 8, 2048)
            self.assertGreater(mem, comp)
            self.assertEqual(tot, max(comp, mem))
        self.assertAlmostEqual(t1["delta_ideal_ms"]["roofline"], t1["delta_ideal_ms"]["memory"], places=12)

    def test_na_points_no_three_point(self):
        r = AJ.aj4(jh1_analysis(self.T1, self.T2, drop=(13, "32x2048")))
        self.assertEqual(r["T1"]["32x2048"]["status"], "NA_POINTS")
        self.assertIsNone(r["T1"]["32x2048"]["ratio_beta1_to_ideal"])
        self.assertEqual(r["T1_ctx_diff_of_diff"]["32x8192-32x2048"]["status"], "NA_POINTS")
        self.assertEqual(r["T1"]["8x2048"]["status"], "OK")

    def test_windep_per_term(self):
        r = AJ.aj4(jh1_analysis(self.T1, self.T2, w1="WEIGHT_DEPENDENT"))
        self.assertTrue(r["T1"]["status"].startswith("NOT_EVALUATED"))
        self.assertEqual(r["T2"]["4096"]["status"], "OK")
        r = AJ.aj4(jh1_analysis(self.T1, self.T2, w2=None))
        self.assertTrue(r["T2"]["status"].startswith("NOT_EVALUATED"))
        self.assertTrue(AJ.aj2_jh1(jh1_analysis(self.T1, self.T2, w1="WEIGHT_DEPENDENT"))["status"].startswith("NOT_EVALUATED"))

    def test_aj2_jh1_side_table(self):
        r = AJ.aj2_jh1(jh1_analysis(self.T1, self.T2))
        self.assertEqual(r["status"], "OK")
        self.assertAlmostEqual(r["slope_b_vs_nM"], 0.002, places=9)
        self.assertAlmostEqual(r["b_a31"], 0.05, places=9)


class TestDriverAndFields(unittest.TestCase):
    def test_run_end_to_end_fake(self):
        A1 = ja_analysis("J-A1", a_inv(), t2p=t2p_tab("J-A1", 1.0))
        A2 = ja_analysis("J-A2", a_inv(), rnd="r2", t2p=t2p_tab("J-A2", 1.0, rnd="r2"))
        out = AJ.run(A1, A2, jh1_analysis(TestAJ4.T1, TestAJ4.T2))
        json.dumps(out)
        self.assertIsNone(out["AJ-1"]["cross_job"]["i"]["label"])
        self.assertEqual(out["AJ-1"]["cross_job"]["i"]["J-A1"], "BWD_INV_D_HELD_LE64")
        self.assertFalse(out["AJ-3"]["cross_job"]["pooled"])
        self.assertTrue(all("[해석]" in s for s in out["interpretations"]))

    def test_fields_exist_in_smoke_analysis(self):
        """Every analysis.json field aj_analysis reads exists in the real smoke output or in the harness writer."""
        SA = json.loads((SMOKE / "SA_analysis.json").read_text())
        SC = json.loads((SMOKE / "SC_analysis.json").read_text())
        for k in ("tables", "fits", "smoke"):
            self.assertIn(k, SA)
        for k in ("t1", "t2", "t2prime"):
            self.assertIn(k, SA["tables"])
        self.assertIn("t1", SA["fits"])
        k1 = "J-A1|r1|real_a4|D44|cap128"
        self.assertEqual(AJ.ja_groups(SA, "J-A1"), {("r1", "real_a4"): {"D44": k1}})
        for f in ("D_real", "cells", "ctx_actual", "gpu_ms"):
            self.assertIn(f, SA["tables"]["t1"][k1])
        tp = SA["tables"]["t2prime"]["J-A1|r1|real_a4|D44"]
        for f in ("P", "points_confirmed"):
            self.assertIn(f, tp)
        self.assertIn("jh1", SC)
        self.assertIn("w_indep", SC["jh1"])
        self.assertIsNotNone(AJ.jh1_t1_cells(SC, 4))
        # fits.t1 / tables.t2 entries are empty in the smoke (too few cells): check the writer instead (read-only)
        src = (HARNESS / "ct_runner.py").read_text()
        self.assertIn('fits[key] = {"chosen": _strip(fit["chosen"]), "F1": _strip(fit["F1"]), "label": fit["label"]}', src)
        self.assertIn('t2tab[key] = {"points": {str(L): t for L, t in pts}', src)
        self.assertIn('key = f"{v[\'job\']}|r{v[\'round\']}|{ident[\'tag\']}|{ident[\'mode\']}"', src)
        self.assertIn('"coef": f["coef"]', (HARNESS / "ct_fit.py").read_text())


if __name__ == "__main__":
    unittest.main()
