"""CPU tests of the policy harness (results/cap_char/policy_harness, PREREG_HD_HE0cell rev0.1).  No GPU, no server.
PH_DIR (env) points the tests at a copy of the harness -- ph_mutants.py uses it to prove every registered rule is guarded:
a test that cannot fail under the reverted rule is an identity gate (lesson 53)."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(os.path.abspath(__file__)).parents[1]
PH = Path(os.environ.get("PH_DIR", ENG / "results/cap_char/policy_harness"))
sys.path.insert(0, str(PH))
import ph_spec as P  # noqa: E402
import ph_trace as T  # noqa: E402
import ph_verdict as V  # noqa: E402
import ph_runner as RU  # noqa: E402
import ph_analyze as A  # noqa: E402

ADD = {"cap_ladder": {"W1": {"rmin": 1.0}, "W3": {"rmin": 0.1}}, "lambda_star": {"W1": 3.0, "W3": 0.3},
       "W3_floor_ms": 1000.0, "tuned_config": {"anchor_idx": 3, "dwell": 3}}
W1_POOL = [[100 + i, 50 + i] for i in range(300)]
W3_POOL = [[i, 6000 + i, 200] for i in range(100)]


def rq(ttft, itls=(30.0,), err="", phase="LO", i=0):
    return {"request_id": f"r{i}", "phase": phase, "ttft_ms": ttft, "token_itl_ms": list(itls), "error": err}


def mkrun(rid, arm, seed, segs):
    return {"run_id": rid, "arm": arm, "seed": seed, "segments": segs, "switch_count": 0, "split_dwell": {}}


def seg(phase, n_good, n_bad, dur, plen=None, good_ttft=500.0, bad_ttft=8000.0):
    reqs = [rq(good_ttft, phase=phase, i=i) for i in range(n_good)] + [rq(bad_ttft, phase=phase, i=n_good + i) for i in range(n_bad)]
    return {"segment_id": f"{phase}", "phase": phase, "round": 1, "duration_s": dur, "phase_len_s": plen, "requests": reqs}


def he0_arm(arm_name, seeds, g_lo, g_hi, jitter=None):
    arm = {"kind": "dynamic"} if arm_name == "dyn" else {"kind": "static", "D": int(arm_name[1:])}
    out = {}
    for s in seeds:
        j = (jitter or {}).get(s, 0)
        out[s] = mkrun(f"{arm_name}_{s}", arm, s, [seg("LO", g_lo + j, 100 - g_lo - j, 100.0), seg("HI", g_hi + j, 100 - g_hi - j, 50.0)])
    return out


def he0_runs(dyn_lo, dyn_hi, seeds=(1, 2, 3, 4), dyn_jitter=None):
    base = {"s16": (70, 40), "s24": (75, 45), "s34": (80, 50), "s44": (90, 60), "s64": (60, 30)}
    runs = {a: he0_arm(a, seeds, *v) for a, v in base.items()}
    runs["dyn"] = he0_arm("dyn", seeds, dyn_lo, dyn_hi, dyn_jitter)
    return runs


class SpecTests(unittest.TestCase):
    def test_idx_mapping_and_ratio(self):
        self.assertEqual(P.D_OF_IDX, {1: 16, 2: 24, 3: 34, 4: 44, 5: 64})
        r = P.he0_rates(3.0)
        self.assertAlmostEqual(r["hi"] / r["lo"], 4.0)
        self.assertAlmostEqual(r["hi"], 4.5)

    def test_seed_sets_disjoint(self):
        self.assertFalse(set(P.TUNE_SEEDS) & set(P.EVAL_SEEDS))
        self.assertFalse(set(P.CAP_SEEDS) & set(P.EVAL_SEEDS))

    def test_plans_valid_and_deterministic(self):
        for j in ("CAP1", "CAP2", "TUNE", "EVAL-a", "EVAL-b", *[f"HD-{i}" for i in range(1, 9)]):
            p = P.make_plan(j, ADD)
            self.assertEqual(P.plan_problems(p), [], j)
            self.assertEqual(P.sha(p), P.sha(P.make_plan(j, ADD)))
        hd = P.make_plan("HD-3", ADD)
        self.assertEqual(len(hd["boots"]), 5)
        self.assertEqual(sorted(b["arm"]["D"] for b in hd["boots"]), [16, 24, 34, 44, 64])
        self.assertTrue(all(len(b["runs"]) == 3 for b in hd["boots"]))

    def test_eval_dynamic_reboots_per_seed(self):
        p = P.make_plan("EVAL-a", ADD)
        dyn = [b for b in p["boots"] if b["arm"]["kind"] == "dynamic"]
        self.assertEqual(len(dyn), 2)
        self.assertTrue(all(len(b["runs"]) == 1 for b in dyn))
        st = [b for b in p["boots"] if b["arm"]["kind"] == "static"]
        self.assertEqual(len(st), 5)
        self.assertTrue(all(len(b["runs"]) == 2 for b in st))

    def test_hd_start_phase_alternates_by_seed(self):
        self.assertEqual(P.make_plan("HD-1", ADD)["boots"][0]["runs"][0]["start"], "W1")
        self.assertEqual(P.make_plan("HD-2", ADD)["boots"][0]["runs"][0]["start"], "W3")

    def test_rate_rule_q1(self):
        r = P.make_plan("HD-1", ADD)["boots"][0]["runs"][0]
        self.assertAlmostEqual(r["rate_w1"], 3.0)
        self.assertAlmostEqual(r["rate_w3"], 0.3)
        self.assertAlmostEqual(r["w3_ttft_ms"], 3000.0)

    def test_addendum_required(self):
        for j in ("TUNE", "EVAL-a", "HD-1", "CAP1"):
            with self.assertRaises(RuntimeError):
                P.make_plan(j, None)

    def test_arm_env(self):
        st = P.arm_env(P.static_arm(44))
        self.assertNotIn("PDMUX_STICKY_PARTITION", st)             # decision 49: static arms are non-sticky in EVERY Job
        self.assertEqual(st["PDMUX_R2_FIXED_DSM"], "44")
        self.assertEqual(P.arm_env(P.static_arm(44, sticky=True))["PDMUX_STICKY_PARTITION"], "1")
        dy = P.arm_env(P.dynamic_arm(3, 6))
        self.assertNotIn("PDMUX_STICKY_PARTITION", dy)          # engine rejects sticky + SLO_SCHED
        self.assertNotIn("PDMUX_R2_POLICY", dy)
        self.assertEqual((dy["PDMUX_SLO_ANCHOR_IDX"], dy["PDMUX_SLO_DWELL"], dy["PDMUX_TPOT_SLO_MS"]), ("3", "6", "60"))

    def test_tune_grid_reboots_per_config_and_seed(self):
        p = P.make_plan("TUNE", ADD)
        self.assertEqual(len(p["boots"]), 12)                      # controller state lives in the boot (audit M3)
        self.assertTrue(all(len(b["runs"]) == 1 for b in p["boots"]))
        self.assertEqual({r["seed"] for b in p["boots"] for r in b["runs"]}, set(P.TUNE_SEEDS))

    def test_every_job_has_nonsticky_static_arms(self):
        self.assertIs(P.STATIC_STICKY, False)
        for j in ("CAP1", "CAP2", "EVAL-a", "EVAL-b", *[f"HD-{i}" for i in range(1, 9)]):
            for b in P.make_plan(j, ADD)["boots"]:
                if b["arm"]["kind"] == "static":
                    self.assertIs(b["arm"]["sticky"], False, j)
                    self.assertNotIn("PDMUX_STICKY_PARTITION", P.arm_env(b["arm"]), j)

    def test_hd_first_boot_records_floor_covariate_only(self):
        self.assertEqual(P.make_plan("HD-1", ADD)["boots"][0]["floor"], "covariate")
        self.assertIs(P.make_plan("CAP2", ADD)["boots"][0]["floor"], True)

    def test_he0_rates_refuses_missing_lambda(self):
        with self.assertRaises(RuntimeError):
            P.he0_rates(None)

    def test_engine_digest_is_placeholder_or_hex(self):
        v = P.ENGINE_DIGEST_FILE.read_text().strip()
        self.assertTrue(v == P.ENGINE_DIGEST_PENDING or (len(v) == 64 and all(c in "0123456789abcdef" for c in v)))


class TraceTests(unittest.TestCase):
    def test_hd_clean_and_cont(self):
        c = T.hd_trace("clean", 120, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 1, "t")
        self.assertEqual([s["phase"] for s in c], ["W1", "W3"])
        self.assertTrue(all(r["arrival_s"] < 120 for s in c for r in s["requests"]))
        k = T.hd_trace("cont", 120, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 1, "t")
        self.assertEqual(len(k), 1)
        w3 = [r["arrival_s"] for r in k[0]["requests"] if r["phase"] == "W3"]
        self.assertTrue(w3 and min(w3) >= 120)                   # second phase offset by the phase length
        self.assertTrue(all(r["arrival_s"] < 120 for r in k[0]["requests"] if r["phase"] == "W1"))

    def test_start_order(self):
        k = T.hd_trace("clean", 120, "W3", 3.0, 0.3, W1_POOL, W3_POOL, 2, "t")
        self.assertEqual([s["phase"] for s in k], ["W3", "W1"])

    def test_same_requests_for_every_arm(self):
        a = T.hd_trace("clean", 600, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 5, "t")
        b = T.hd_trace("clean", 600, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 5, "t")
        self.assertEqual(T.trace_hash(a), T.trace_hash(b))
        c = T.hd_trace("clean", 600, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 6, "t")
        self.assertNotEqual(T.trace_hash(a), T.trace_hash(c))

    def test_requests_do_not_depend_on_arm_tag(self):
        """run_id (hence the tag) differs per arm; the sampled requests must not."""
        a = T.hd_trace("clean", 600, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 5, "HD-5_s16_clean600")
        b = T.hd_trace("clean", 600, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 5, "HD-5_s44_clean600")
        sig = lambda segs: [[(r["arrival_s"], r["input_tokens"], r["output_tokens"]) for r in s["requests"]] for s in segs]
        self.assertEqual(sig(a), sig(b))
        h = T.he0_trace(W1_POOL, 1.0, 4.0, 50, 2, 3, "EVAL-a_s16_s3")
        h2 = T.he0_trace(W1_POOL, 1.0, 4.0, 50, 2, 3, "EVAL-a_dyn_s3")
        self.assertEqual(sig(h), sig(h2))

    def test_workload_eos_semantics(self):
        c = T.hd_trace("clean", 120, "W1", 3.0, 0.3, W1_POOL, W3_POOL, 1, "t")
        self.assertTrue(all(r["ignore_eos"] is True for r in c[0]["requests"]))       # W1 fixed output
        self.assertTrue(all(r["ignore_eos"] is False for r in c[1]["requests"]))      # W3 natural output (decision 47)
        self.assertTrue(all(r["output_tokens"] == P.W3_MAX_NEW for r in c[1]["requests"]))

    def test_he0_structure(self):
        segs = T.he0_trace(W1_POOL, 1.125, 4.5, 200, 3, 1, "x")
        self.assertEqual([s["phase"] for s in segs], ["LO", "HI"] * 3)
        self.assertTrue(all(len(s["requests"]) == 200 for s in segs))
        lo = max(r["arrival_s"] for r in segs[0]["requests"])
        hi = max(r["arrival_s"] for r in segs[1]["requests"])
        self.assertGreater(lo / hi, 2.5)                          # same count at 4x rate -> ~4x shorter segment
        self.assertLess(lo / hi, 6.0)


class VerdictTests(unittest.TestCase):
    def test_goodput_uses_request_p95_not_mean(self):
        spiky = rq(500, [10.0] * 9 + [400.0])                     # mean 49 <= 60, p95 ~ 215 > 60
        smooth = rq(500, [59.0] * 10)
        self.assertFalse(V.req_good(spiky, 3000, 60))
        self.assertTrue(V.req_good(smooth, 3000, 60))
        self.assertFalse(V.req_good(rq(3001), 3000, 60))
        self.assertFalse(V.req_good(rq(500, err="x"), 3000, 60))

    def test_duration_is_summed(self):
        run = mkrun("r", {}, 1, [seg("LO", 30, 0, 10.0), seg("HI", 30, 0, 20.0)])
        self.assertAlmostEqual(V.he0_goodput(run, 3000, 60), 60 / 30.0)    # max() would give 3.0
        self.assertAlmostEqual(V.he0_goodput(run, 3000, 60, phase="HI"), 1.5)

    def test_cliff_rule(self):
        near = [rq(3000 * 1.04, i=i) for i in range(20)] + [rq(500, i=100 + i) for i in range(80)]   # 20 % in the band
        far = [rq(3000 * 1.2, i=i) for i in range(20)] + [rq(500, i=100 + i) for i in range(80)]
        few = [rq(3000 * 1.04, i=i) for i in range(5)] + [rq(500, i=100 + i) for i in range(95)]     # 5 % in the band
        self.assertTrue(V.is_cliff(near, 3000))
        self.assertFalse(V.is_cliff(far, 3000))
        self.assertFalse(V.is_cliff(few, 3000))

    def test_t_ci_exact(self):
        m, lo, hi, n = V.paired_t_ci([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(m, 2.5)
        self.assertAlmostEqual(hi - m, 3.182 * 1.2909944 / 2, places=3)
        self.assertEqual(n, 4)
        self.assertTrue(math.isnan(V.paired_t_ci([1.0])[1]))

    def test_no_bootstrap_in_verdict_source(self):
        src = (PH / "ph_verdict.py").read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("#", '"""')))
        self.assertNotIn("import random", code)
        self.assertNotIn("resample", code)

    def test_he0_labels(self):
        below = V.he0_compare(he0_runs(81, 54), 3000, 60)
        self.assertEqual(below["label"], "DYN_BELOW_STATIC")
        equal = V.he0_compare(he0_runs(90, 60), 3000, 60)
        self.assertEqual(equal["label"], "DYN_EQUIV_STATIC")
        above = V.he0_compare(he0_runs(99, 66), 3000, 60)
        self.assertEqual(above["label"], "DYN_ABOVE_STATIC")
        self.assertEqual(above["best_static"], "s44")
        noisy = V.he0_compare(he0_runs(90, 60, dyn_jitter={1: 9, 2: -9, 3: 9, 4: -9}), 3000, 60)
        self.assertEqual(noisy["label"], "HE0CELL_UNRESOLVED")

    def test_above_needs_both_references(self):
        # dyn beats the a-priori D44 but NOT the post-hoc best static: build s34 as best static, D44 weaker
        runs = he0_runs(95, 60)
        runs["s34"] = he0_arm("s34", (1, 2, 3, 4), 96, 62)
        runs["s44"] = he0_arm("s44", (1, 2, 3, 4), 70, 40)
        r = V.he0_compare(runs, 3000, 60)
        self.assertEqual(r["best_static"], "s34")
        self.assertNotEqual(r["label"], "DYN_ABOVE_STATIC")

    def test_n_below_four_unresolved(self):
        r = V.he0_compare(he0_runs(81, 54, seeds=(1, 2, 3)), 3000, 60)
        self.assertEqual(r["label"], "HE0CELL_UNRESOLVED")

    def test_cliff_blocks_label(self):
        runs = he0_runs(81, 54)
        for run in runs["dyn"].values():                          # 30 % of dyn's LO requests sit at TTFT ~ SLO
            for i, q in enumerate(run["segments"][0]["requests"][:30]):
                q["ttft_ms"] = 3050.0
        r = V.he0_compare(runs, 3000, 60)
        self.assertEqual(r["label"], "HE0CELL_UNRESOLVED")
        self.assertEqual(r["why"], "CLIFF_ADJACENT")

    def test_tune_selection(self):
        t = {"a": [{"goodput": 1.00, "switch_count": 10}, {"goodput": 1.00, "switch_count": 10}],
             "b": [{"goodput": 0.995, "switch_count": 2}, {"goodput": 0.995, "switch_count": 2}],     # within 1 % -> fewer switches
             "c": [{"goodput": 0.90, "switch_count": 0}, {"goodput": 0.90, "switch_count": 0}]}
        self.assertEqual(V.select_tune(t)[0], "b")
        t["b"] = [{"goodput": 0.97, "switch_count": 2}] * 2
        self.assertEqual(V.select_tune(t)[0], "a")

    # ---- H-D
    @staticmethod
    def tab(f):
        return {D: {s: f(D, s) for s in range(1, 9)} for D in P.D_GRID}

    def test_hd_present_when_phase_optima_differ(self):
        def f(D, s):
            w1 = 10.0 if D == 16 else 2.0
            w3 = 10.0 if D == 44 else 2.0
            return {"W1": w1, "W3": w3, "total": (w1 + w3) / 2}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        self.assertAlmostEqual(cf["H"], math.log(10 / 6), places=6)
        for r in cf["roles"]:
            self.assertEqual(r["pick"], {"W1": 16, "W3": 44})      # one D per phase KIND
        lab, _ = V.hd_label({120: cf, 600: cf}, {120: False, 600: False})
        self.assertEqual(lab, "DYNAMIC_HEADROOM_PRESENT")

    def test_hd_absent_needs_all_registered_q(self):
        def f(D, s):
            return {"W1": 6.0, "W3": 6.0, "total": 6.0}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(V.hd_label({120: cf, 600: cf}, {}, qs_run=(1.0,))[0], "HEADROOM_UNRESOLVED")
        self.assertEqual(V.hd_label({120: cf, 600: cf}, {}, qs_run=(0.8, 1.0))[0], "DYNAMIC_HEADROOM_ABSENT")

    def test_hd_needs_both_lengths_and_cliff_blocks(self):
        def f(D, s):
            w1 = 10.0 if D == 16 else 2.0
            w3 = 10.0 if D == 44 else 2.0
            return {"W1": w1, "W3": w3, "total": (w1 + w3) / 2}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        flat = V.crossfit_H(self.tab(lambda D, s: {"W1": 6.0, "W3": 6.0, "total": 6.0}), [1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(V.hd_label({120: cf, 600: flat}, {})[0], "HEADROOM_UNRESOLVED")      # one length only
        self.assertEqual(V.hd_label({120: cf, 600: cf}, {600: True})[0], "HEADROOM_UNRESOLVED")

    def test_hd_selection_is_cross_fitted(self):
        def f(D, s):
            w1 = 10.0 if (D == 16 and s <= 4) or (D == 24 and s >= 5) else 4.0
            return {"W1": w1, "W3": 4.0, "total": (w1 + 4.0) / 2}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(cf["roles"][0]["pick"]["W1"], 16)         # chosen on seeds 1-4, scored on 5-8 (not the other way)
        self.assertEqual(cf["roles"][1]["pick"]["W1"], 24)

    def test_hd_gbest_is_chosen_on_evaluation_seeds(self):
        def f(D, s):
            hi = (D == 16 and s <= 4) or (D == 44 and s >= 5)
            g = 8.0 if hi else 4.0
            return {"W1": g, "W3": g, "total": g}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(cf["roles"][0]["best"], 44)               # role 0 evaluates on seeds 5-8, where D44 is best
        self.assertEqual(cf["roles"][1]["best"], 16)

    def test_hd_widest_ci_of_the_two_roles(self):
        def f(D, s):
            noise = (0.5 if s % 2 else -0.5) if s <= 4 else 0.0     # seeds 1-4 noisy, 5-8 clean
            w1 = 10.0 + noise if D == 16 else 2.0
            w3 = 10.0 if D == 44 else 2.0
            return {"W1": w1, "W3": w3, "total": (w1 + w3) / 2}
        cf = V.crossfit_H(self.tab(f), [1, 2, 3, 4], [5, 6, 7, 8])
        los = [r["lo"] for r in cf["roles"]]
        his = [r["hi"] for r in cf["roles"]]
        self.assertNotEqual(los[0], los[1])
        self.assertEqual(cf["lo"], min(los))
        self.assertEqual(cf["hi"], max(his))

    # ---- capacity
    def test_capacity(self):
        lad = [1.0, 1.35, 1.82]
        cells = [{"rate": 1.0, "n": 100, "good": 95}, {"rate": 1.35, "n": 100, "good": 91}, {"rate": 1.82, "n": 100, "good": 50}]
        c = V.capacity_per_D(cells, lad)
        self.assertEqual((c["capacity"], c["first_fail"]), (1.35, 1.82))
        # 89 % fails the 90 % rule; a pass above the first failure flags non-monotone and does not raise the capacity
        cells2 = [{"rate": 1.0, "n": 100, "good": 95}, {"rate": 1.35, "n": 100, "good": 89}, {"rate": 1.82, "n": 100, "good": 95}]
        c2 = V.capacity_per_D(cells2, lad)
        self.assertEqual(c2["capacity"], 1.0)
        self.assertTrue(c2["nonmonotone"])
        lam, D, st = V.lambda_star_best_static({16: c, 44: c2, 64: {"capacity": None, "first_fail": 1.0}})
        self.assertEqual((lam, D, st), (1.35, 16, "CAPACITY_BRACKETED"))

    def test_truncated_capacity_is_not_accepted(self):
        lad = [1.0, 1.35, 1.82]
        top_passes = V.capacity_per_D([{"rate": r, "n": 100, "good": 97} for r in lad], lad)
        self.assertIsNone(top_passes["first_fail"])
        lam, D, st = V.lambda_star_best_static({16: top_passes})
        self.assertEqual((lam, st), (None, "CAPACITY_NOT_BRACKETED"))      # q=1.0 would sit below capacity (H=0 identity)
        none_pass = V.capacity_per_D([{"rate": r, "n": 100, "good": 10} for r in lad], lad)
        self.assertEqual(V.lambda_star_best_static({16: none_pass})[2], "NO_PASSING_RATE")
        self.assertAlmostEqual(V.next_ladder_rmin("CAPACITY_NOT_BRACKETED", 1.0, 1.5, 6), 1.5 ** 5)
        self.assertAlmostEqual(V.next_ladder_rmin("NO_PASSING_RATE", 1.0, 1.5, 6), 1.5 ** -5)
        self.assertIsNone(V.next_ladder_rmin("CAPACITY_BRACKETED", 1.0, 1.5, 6))

    def test_empty_itl_is_good_single_token_output(self):
        self.assertTrue(V.req_good(rq(500, itls=()), 3000, 60))

    def test_hd_phase_attribution_by_arrival(self):
        """A W1 request that ARRIVED in the W1 phase but completed during W3 still belongs to W1; a W3 request never counts for W1."""
        reqs = [dict(rq(500, phase="W1", i=0), scheduled_arrival_s=5.0), dict(rq(9000, phase="W1", i=1), scheduled_arrival_s=6.0),
                dict(rq(500, phase="W3", i=2), scheduled_arrival_s=125.0), dict(rq(500, phase="W3", i=3), scheduled_arrival_s=140.0)]
        run = mkrun("r", {}, 1, [{"segment_id": "c", "phase": "CONT", "round": 0, "duration_s": 300.0, "phase_len_s": 120, "requests": reqs}])
        self.assertAlmostEqual(V.hd_phase_goodput(run, "W1", 3000, 3000, 60), 1 / 120)
        self.assertAlmostEqual(V.hd_phase_goodput(run, "W3", 3000, 3000, 60), 2 / 120)
        # warm exclusion drops arrivals in the first 10 % (12 s) of each phase and shrinks the denominator
        self.assertAlmostEqual(V.hd_phase_goodput(run, "W1", 3000, 3000, 60, exclude_warm=0.10), 0.0)
        self.assertAlmostEqual(V.hd_phase_goodput(run, "W3", 3000, 3000, 60, exclude_warm=0.10), 1 / 108)

    def test_he0_requires_all_five_static_arms(self):
        runs = he0_runs(81, 54)
        del runs["s16"]
        with self.assertRaises(ValueError):
            V.he0_compare(runs, 3000, 60)

    def test_he0_reports_phase_level_pairs_descriptively(self):
        r = V.he0_compare(he0_runs(81, 54), 3000, 60)
        self.assertEqual(set(r["phase_ci"]), {"LO:a", "LO:b", "HI:a", "HI:b"})

    def test_stationarity_s1_s2(self):
        flat = [dict(rq(500, i=i), scheduled_arrival_s=float(i)) for i in range(60)]
        drift = [dict(rq(500 if i < 30 else 2500, i=i), scheduled_arrival_s=float(i)) for i in range(60)]   # last third 5x slower
        self.assertEqual(V.stationarity(flat, 3000, 60)["label"], "STATIONARY")
        self.assertEqual(V.stationarity(drift, 3000, 60)["label"], "UNSTABLE")
        late_fail = [dict(rq(500, itls=(30.0,) if i < 40 else (400.0,), i=i), scheduled_arrival_s=float(i)) for i in range(60)]   # TTFT flat, ITL degrades
        lf = V.stationarity(late_fail, 3000, 60)
        self.assertLess(lf["S1"], 1.7)
        self.assertEqual(lf["label"], "UNSTABLE")                                                           # S2 alone (attainment drop)

    def test_hd_gate5_has_oracle_switch_count(self):
        runs = {16: {1: mkrun("a", {}, 1, [seg("W1", 5, 0, 10.0, plen=10)])}}
        g = V.hd_gate5(runs, {"W1": 16, "W3": 44})
        self.assertEqual(g["oracle"]["switches_per_cycle"], 1)
        self.assertEqual(V.hd_gate5(runs, {"W1": 44, "W3": 44})["oracle"]["switches_per_cycle"], 0)
        self.assertIn("p99", g[16]["ttft_ms"])


class RunnerTests(unittest.TestCase):
    def test_boot_env_strips_instrumentation(self):
        base = {"PDMUX_PHASE_EVENTS": "1", "PDMUX_DECODE_PROBE_PATH": "/x", "PDMUX_TELEMETRY_PATH": "/y", "PDMUX_ROOT": "/r", "HOME": "/h"}
        env = RU.boot_env(P.static_arm(44), Path("/tmp/b"), base=base)
        self.assertEqual(RU.instr_violation(env), [])
        self.assertEqual(env["PDMUX_ROOT"], "/r")
        self.assertNotIn("PDMUX_STICKY_PARTITION", RU.boot_env(P.dynamic_arm(2, 3), Path("/tmp/b"), base=base))
        self.assertEqual(RU.instr_violation({"PDMUX_PHASE_EVENTS": "1"}), ["PDMUX_PHASE_EVENTS"])

    def test_parse_switches(self):
        log = ("[2026-10-28 10:00:00] boot\n[2026-10-28 10:00:10] SLO-BIND 3->2 dec_sm=24\n"
               "[2026-10-28 10:00:40] SLO-BIND 2->4 dec_sm=44\n[2026-10-28 10:01:00] done\n")
        n, dw = RU.parse_switches(log, 3)
        self.assertEqual(n, 2)
        self.assertEqual({k: round(v) for k, v in dw.items()}, {"34": 10, "24": 30, "44": 20})

    def _fakes(self, tmp, wrong_tokens=False, calls=None):
        import datetime as dt
        live = {}

        class Srv:
            def __init__(s, bdir, argv, env):
                s.log, s.port = Path(bdir) / "server.log", 1234
                s.log.write_text("[2026-10-28 10:00:00] boot\n")
                live["srv"] = s
            def start(s): return "ok"
            def stop(s): return {"returncode": 0}

        def replay(endpoint, reqs, out_path):
            if calls is not None:
                calls.append(reqs)
            first = reqs[0]["request_id"] if reqs else ""
            if first.endswith("r1LO-00000"):                           # one controller switch inside every HE0 run (round 1, LO)
                with open(live["srv"].log, "a") as f:
                    f.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] SLO-BIND 3->4 dec_sm=44\n")
            recs = [{"request_id": r["request_id"], "phase": r["phase"], "ttft_ms": 500.0, "token_itl_ms": [30.0] * 3,
                     "error": "", "error_kind": "", "scheduled_arrival_s": r["arrival_s"], "completion_s": r["arrival_s"] + 1.0,
                     "input_tokens": r["input_tokens"], "requested_output_tokens": r["output_tokens"],
                     "observed_output_tokens": (1 if wrong_tokens else r["output_tokens"])}
                    for r in reqs]
            return (max(r["arrival_s"] for r in reqs) + 1.0 if reqs else 1.0), recs
        return Srv, replay

    def test_run_job_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            Srv, replay = self._fakes(td)
            plan = P.make_plan("EVAL-a", ADD)
            out = Path(td) / "raw"
            res = RU.run_job(plan, out, Srv, replay, W1_POOL, W3_POOL)
            self.assertEqual(RU.exit_code(res), 0)
            runs = A.load_runs(out)
            self.assertEqual(len(runs), 12)
            dyn = [r for r in runs if r["arm"]["kind"] == "dynamic"]
            self.assertTrue(all(r["switch_count"] == 1 for r in dyn))
            self.assertTrue(all(len(r["segments"]) == 6 for r in runs))
            self.assertEqual(A.eval_trace_problems(runs), [])            # every arm of a seed saw the same requests (audit M7)
            self.assertTrue((out / "RUN_DONE.json").exists())
            with self.assertRaises(RuntimeError):
                RU.run_job(plan, out, Srv, replay, W1_POOL, W3_POOL)                 # no --resume: ABORT_RAW_NOT_EMPTY
            before = {p.name: p.stat().st_mtime_ns for p in (out / "runs").glob("*.json")}
            RU.run_job(plan, out, Srv, replay, W1_POOL, W3_POOL, resume=True)
            after = {p.name: p.stat().st_mtime_ns for p in (out / "runs").glob("*.json")}
            self.assertEqual(before, after)                                           # finished runs kept

    def test_switch_count_is_sliced_per_run(self):
        """Two runs in one boot: the second run's count must not include the first run's (or warm-up's) switches (audit M2)."""
        with tempfile.TemporaryDirectory() as td:
            Srv, replay = self._fakes(td)
            rates = P.he0_rates(3.0)
            boot = {"boot_id": "b", "arm": P.dynamic_arm(3, 3),
                    "runs": [P.he0_run("ra", 1, rates, np_=20, rounds=1), P.he0_run("rb", 2, rates, np_=20, rounds=1)]}
            # the fake keys the switch on a request id ending "r1LO-00000" -> one per run
            RU.run_boot(boot, Path(td) / "raw", Srv, replay, W1_POOL, W3_POOL)
            ra, rb = (json.loads((Path(td) / "raw/runs" / f"{x}.json").read_text()) for x in ("ra", "rb"))
            self.assertEqual((ra["switch_count"], rb["switch_count"]), (1, 1))
            self.assertAlmostEqual(sum(ra["split_dwell"].values()), sum(rb["split_dwell"].values()), delta=3.0)

    def test_warmup_is_60s_and_correctness_gate(self):
        with tempfile.TemporaryDirectory() as td:
            calls = []
            Srv, replay = self._fakes(td, calls=calls)
            plan = P.make_plan("TUNE", ADD)
            plan["boots"] = plan["boots"][:1]
            RU.run_boot(plan["boots"][0], Path(td) / "raw", Srv, replay, W1_POOL, W3_POOL)
            warm = calls[0]
            self.assertGreater(max(r["arrival_s"] for r in warm), 50.0)            # ~60 s of arrivals, not 10 requests
            self.assertLessEqual(max(r["arrival_s"] for r in warm), 60.0)
        with tempfile.TemporaryDirectory() as td:
            Srv, replay = self._fakes(td, wrong_tokens=True)
            r = RU.run_boot(P.make_plan("TUNE", ADD)["boots"][0], Path(td) / "raw", Srv, replay, W1_POOL, W3_POOL)
            self.assertEqual(r["status"], "INVALID_CORRECTNESS")
            self.assertEqual(r["runs"], [])
            self.assertEqual(RU.exit_code([r]), 17)

    def test_exit_code_flags_dead_boot(self):
        self.assertEqual(RU.exit_code([{"status": "done"}]), 0)
        self.assertEqual(RU.exit_code([{"status": "done"}, {"status": "died"}]), 17)

    def test_cap_trace_identical_for_every_arm(self):
        p = P.make_plan("CAP1", ADD)
        specs = {}
        for b in p["boots"]:
            for r in b["runs"]:
                segs = RU.build_segments(r, W1_POOL, W3_POOL, r["run_id"])
                specs.setdefault((r["rate"], r["seed"]), set()).add(T.trace_hash(segs))
        self.assertTrue(all(len(v) == 1 for v in specs.values()))                   # same (rate, seed) -> same requests across D
        self.assertEqual(len(specs), 2 * P.CAP_N_RATES["W1"])

    def test_w1_sha_mismatch_aborts_before_any_server(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = Path(td) / "w1.json"
            w1.write_text(json.dumps(W1_POOL))
            add = dict(ADD, w1_lengths_sha256="0" * 64)
            (Path(td) / "a.json").write_text(json.dumps(add))
            rc = RU.main(["run", "--job", "TUNE", "--out", str(Path(td) / "raw"), "--w1-lengths", str(w1), "--addendum", str(Path(td) / "a.json")])
            self.assertEqual(rc, 13)

    def test_cap2_measures_floor(self):
        with tempfile.TemporaryDirectory() as td:
            Srv, replay = self._fakes(td)
            plan = P.make_plan("CAP2", ADD)
            for b in plan["boots"]:
                b["runs"] = b["runs"][:1]
            RU.run_job(plan, Path(td) / "raw", Srv, replay, W1_POOL, W3_POOL)
            fl = json.loads((Path(td) / "raw" / "floor.json").read_text())
            self.assertEqual(fl["floor_ms"], 500.0)

    def test_analysis_glue(self):
        with tempfile.TemporaryDirectory() as td:
            Srv, replay = self._fakes(td)
            out = Path(td) / "raw"
            RU.run_job(P.make_plan("TUNE", ADD), out, Srv, replay, W1_POOL, W3_POOL)
            pick = A.tune_pick(A.load_runs(out))
            self.assertIn("anchor_idx", pick["tuned_config"])


sys.path.insert(0, str(ENG / "tests"))
try:
    import test_loadgen_outcome as LO
    HAVE_LO = bool(LO.HAVE_AIOHTTP)
except Exception:  # noqa: BLE001
    HAVE_LO = False


class SealedLoadgenTests(unittest.TestCase):
    def test_trace_loadgen_untouched(self):
        src = (ENG / "benchmarks/pdmux_eval/trace_loadgen.py").read_text()
        self.assertIn('"ignore_eos": True,', src)                      # sealed bytes: payload still hard-codes True
        self.assertNotIn('item.get("ignore_eos"', src)
        self.assertNotIn("PL.", src)


@unittest.skipUnless(HAVE_LO, "aiohttp / test_loadgen_outcome not available")
class NaturalOutputWrapperTests(unittest.TestCase):
    def _run(self, abs_mono):
        import asyncio
        import ph_loadgen as PL

        seen = {}

        class Srv(LO._FakeServer):
            async def generate(self, req):
                body = await req.json()
                seen[body["rid"]] = body["sampling_params"]["ignore_eos"]
                return await super().generate(req)

        reqs = [{"request_id": "short_nat", "phase": "W3", "arrival_s": 0.0, "input_tokens": 4, "output_tokens": 8, "ignore_eos": False},
                {"request_id": "short_fix", "phase": "W1", "arrival_s": 0.0, "input_tokens": 4, "output_tokens": 8},
                {"request_id": "full_nat", "phase": "W3", "arrival_s": 0.0, "input_tokens": 4, "output_tokens": 8, "ignore_eos": False}]
        with tempfile.TemporaryDirectory() as td, Srv() as srv:
            out = Path(td) / "o.jsonl"
            summ = asyncio.run(PL.replay_natural(f"http://127.0.0.1:{srv.port}/generate", reqs, out, 3000.0, 60.0, abs_mono=abs_mono))
            recs = {json.loads(l)["request_id"]: json.loads(l) for l in out.read_text().splitlines()}
        import aiohttp
        return seen, recs, summ, aiohttp

    def test_natural_flag_and_classification(self):
        seen, recs, summ, aiohttp = self._run(False)
        self.assertEqual(seen, {"short_nat": False, "short_fix": True, "full_nat": False})
        self.assertEqual(recs["short_nat"]["error"], "")                # natural short stream is NOT OUTPUT_SHORT
        self.assertEqual(recs["short_fix"]["error_kind"], "OUTPUT_SHORT")  # fixed-length request still is
        self.assertEqual(recs["full_nat"]["error"], "")
        self.assertNotIn("t_sent_mono", recs["full_nat"])
        self.assertTrue(aiohttp.ClientSession.__module__.startswith("aiohttp"))   # patch removed afterwards

    def test_connector_limit_is_unlimited(self):
        import aiohttp
        from unittest import mock
        limits = []
        real = aiohttp.TCPConnector

        def spy(*a, **k):
            limits.append(k.get("limit"))
            return real(*a, **k)
        with mock.patch.object(aiohttp, "TCPConnector", spy):
            self._run(False)
        self.assertEqual(limits, [0])                              # default 100 would queue streams in the client (audit M1)

    def test_composes_with_abs_mono(self):
        seen, recs, summ, _ = self._run(True)
        self.assertEqual(seen["short_nat"], False)
        self.assertEqual(seen["short_fix"], True)
        self.assertIn("t_sent_mono", recs["full_nat"])
        self.assertIn("benchmark_started_mono", summ)


class JobShellTests(unittest.TestCase):
    def run_sh(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        e.update(env)
        return subprocess.run(["bash", str(PH / "ph_job.sh")], env=e, capture_output=True, text=True, timeout=60)

    def test_unsealed_refuses(self):
        r = self.run_sh({"PDMUX_POLICY_JOB": "HD-1"})
        self.assertEqual(r.returncode, 16)
        self.assertIn("ABORT_UNSEALED", r.stderr)

    def test_bad_job_refuses(self):
        r = self.run_sh({"PDMUX_POLICY_JOB": "HD-9", "PDMUX_POLICY_TEST_MODE": "1"})
        self.assertEqual(r.returncode, 16)

    def test_pending_engine_digest_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            pend = Path(td) / "ED.txt"
            pend.write_text("PENDING_FINAL_TREE\n")
            r = self.run_sh({"PDMUX_POLICY_JOB": "CAP1", "PDMUX_POLICY_SEALED": "1", "PDMUX_POLICY_ENGINE_DIGEST_FILE": str(pend),
                             "PDMUX_POLICY_OUT": f"{td}/raw", "PDMUX_POLICY_SUMMARY": f"{td}/sum"})
            self.assertEqual(r.returncode, 13)
            self.assertIn("ABORT_ENGINE_DIGEST_PENDING", r.stderr)

    def test_sealed_without_digest_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_sh({"PDMUX_POLICY_JOB": "CAP1", "PDMUX_POLICY_SEALED": "1", "PDMUX_POLICY_HARNESS_DIGEST": f"{td}/none",
                             "PDMUX_POLICY_OUT": f"{td}/raw", "PDMUX_POLICY_SUMMARY": f"{td}/sum"})
            self.assertEqual(r.returncode, 13)


class MutantsTests(unittest.TestCase):
    @unittest.skipIf(os.environ.get("PH_IN_MUTANT"), "inside a mutant run")
    def test_every_mutant_is_killed(self):
        sys.path.insert(0, str(PH))
        import ph_mutants
        surv = ph_mutants.run_all()
        self.assertEqual(surv, [], f"surviving mutants (a guard is an identity gate): {surv}")


if __name__ == "__main__":
    unittest.main()
