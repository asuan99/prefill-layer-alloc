"""CPU tests of the cost-term measurement harness, stages 1 + 2 (results/cap_char/costterms_harness/).

What this proves: the harness reproduces the PREREG rev3.4 registered values (cell grid, feasibility table, T4 pre-data
table), parses real SGLang boot logs of this project (compo probe), recovers the KNOWN ground truth of a fake engine
end to end (client -> records -> windows -> fits -> labels) on a virtual clock and over HTTP with a real process, and
FAILS under every stage-1 mutant of PREREG §11-2 / §12 (lesson 53).
What it does NOT prove: anything about the real engine, sglang, CUDA, a GPU or any performance claim.  The fake's
timing model is invented; its numbers are not predictions.
"""
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
H = ENG / "results" / "cap_char" / "costterms_harness"
PROBE = ENG / "results" / "cap_char" / "compo_probe" / "compo_probe_20261021T131621Z"
SRC = PROBE / "variants" / "a4"
sys.path.insert(0, str(H))
os.environ["PDMUX_COSTTERMS_TEST_MODE"] = "1"

S = importlib.import_module("ct_spec")
X = importlib.import_module("ct_extract")
F = importlib.import_module("ct_fit")
R = importlib.import_module("ct_runner")
M = importlib.import_module("ct_mutants")
FE = importlib.import_module("ct_fake_engine")

K = 1024


def mock_attr(obj, name, value):
    from unittest import mock
    return mock.patch.object(obj, name, value)


def tmpdir(tc):
    d = Path(tempfile.mkdtemp(prefix="ct_test_"))
    tc.addCleanup(shutil.rmtree, d, True)
    return d


# =============================================================================== registered values
class TestSpec(unittest.TestCase):
    def test_window_constants(self):
        self.assertEqual(S.O_BASE, S.TRIM_ITERS + S.WINDOW_ITERS + S.MARGIN_ITERS)
        self.assertEqual(S.X_TOKENS * 2, S.O_BASE)
        self.assertEqual(S.ABORT_AT_TOKENS, 319)
        self.assertLess(S.ABORT_AT_TOKENS, S.O_BASE)

    def test_ramp_rule(self):
        """rev3.4 §1-1 / §3-1: O = O_BASE + ramp bound; single-span cells have no ramp; J-A O is uniform over modes."""
        self.assertEqual(S.span_batches(32, 8192 - 166), [3] * 10 + [2])           # closes after crossing 16,384
        self.assertEqual(S.span_batches(9, 2048 - 166), [9])
        for cell in ((1, 2 * K), (8, 2 * K), (1, 64 * K)):
            self.assertEqual(S.o_tokens(*cell, "JA"), S.O_BASE)
        self.assertEqual(S.ja_pmin(8, 8 * K), 44)                                   # base grid includes D64
        self.assertEqual(S.ja_pmin(32, 16 * K), 64)                                 # 16k column: D44 / 108u
        self.assertEqual(S.ja_pmin(16, 16 * K), 44)                                 # OFF-only cell
        self.assertEqual(S.o_tokens(32, 8 * K, "JA"), 6173)                         # PREREG rev3.4 §3-1 table
        self.assertEqual(S.o_tokens(8, 32 * K, "VANILLA"), S.O_BASE + 64)
        self.assertGreater(S.ramp_steps_ub(32, 8 * K, 44, sigma=1.5), S.ramp_steps_ub(32, 8 * K, 44))

    def test_prompt_seed_rule(self):
        import hashlib
        want = int.from_bytes(hashlib.sha256(b"cost|T1|32|8192|1").digest()[:8], "big")
        self.assertEqual(S.prompt_seed("T1", 32, 8192, 1), want)
        a, b = S.prompt_ids("T1", 32, 8192, 1, 50), S.prompt_ids("T1", 32, 8192, 1, 50)
        self.assertEqual(a, b)
        self.assertNotEqual(a, S.prompt_ids("T1", 32, 8192, 2, 50))
        self.assertTrue(all(S.TOKEN_LO <= t < S.TOKEN_HI for t in a))

    def test_default_capture_list_has_required(self):
        lst = S.default_capture_list(128)
        self.assertTrue(set(S.CAPTURE_BS_REQUIRED) <= set(lst))
        self.assertEqual(X.padded_bs(lst, 83), 88)
        self.assertEqual(X.padded_bs(lst, 55), 56)
        self.assertEqual(X.padded_bs(lst, 64), 64)

    def test_spec_digest_is_stable_json(self):
        self.assertEqual(S.spec_sha256(), S.spec_sha256())
        self.assertIn("A-1", json.loads(S.spec_json())["AMBIGUITIES"])

    def test_feasibility_table_reproduces_prereg_3_1(self):
        """PREREG rev3.4 §3-1 table at the derived cap-128 pool (895,781 tokens): bs (c_in + O_cell) <= 0.95 KV."""
        f = {"kv_cache_tokens": 895781, "kv_k_gib": 1.0, "kv_v_gib": 1.0, "capture_bs": S.default_capture_list(128)}
        per_mode = {}
        for mode in S.ALL_T1_MODES:
            cells = X.t1_cells_for_mode(mode)
            rows = X.feasibility(cells, f, 128, {c: S.o_tokens(*c, "JA") for c in cells})
            self.assertEqual(len(rows), len(cells))                                 # every registered cell is a row
            run = {(r["bs"], r["c"]) for r in rows if r["run"]}
            per_mode[mode] = len(run)
            for c in ((96, 8 * K), (8, 64 * K), (32, 32 * K), (64, 8 * K), (64, 12 * K), (88, 8 * K), (48, 16 * K)):
                self.assertNotIn(c, run)
            for c in ((96, 2 * K), (48, 8 * K), (8, 32 * K), (1, 64 * K)):
                self.assertIn(c, run)
            if mode in S.C16K_MODES:
                self.assertIn((32, 16 * K), run)
        self.assertEqual(per_mode, {"D16": 12, "D24": 15, "D34": 15, "D44": 15, "D64": 12, "108u": 15, "108v": 15})
        self.assertEqual(sum(per_mode.values()), 99)                                # rev3.3: 133

    def test_prereg_t4_table(self):
        self.assertEqual(M.o_prereg_table(), [])

    def test_analytic_priors_link(self):
        p = F.priors(4)
        self.assertAlmostEqual(p["a0_ms"], 8.14, delta=0.05)                      # ANALYTIC_COST_FORMULAS §6
        self.assertAlmostEqual(p["c0_ms_per_req_token"] * 1e6, 8.0, delta=0.1)
        self.assertEqual(p["gamma_flop_per_token2"], 40960)


# =============================================================================== boot log parser (real logs)
class TestBootLog(unittest.TestCase):
    def facts(self, boot):
        return X.parse_boot_log((PROBE / boot / "server.log").read_text(errors="replace"), {}, None)

    def test_real_pdmux_log(self):
        f = self.facts("dummy_a4")
        self.assertEqual(f["kv_cache_tokens"], 886804)
        self.assertEqual(f["max_mamba_cache_size"], 128)
        self.assertEqual(f["model_arch"], "NemotronHForCausalLM")
        self.assertEqual(f["server_args"]["load_format"], "dummy")
        self.assertEqual(f["server_args"]["enable_pdmux"], True)
        self.assertEqual(f["pdmux_sm_counts"], [(108, 0), (74, 34), (54, 54), (0, 108)])
        self.assertTrue(set(S.CAPTURE_BS_REQUIRED) <= set(f["capture_bs"]))
        self.assertEqual(X.logged_cap(f), 128)
        self.assertEqual(f["final"]["max_prefill_tokens"], 16384)
        self.assertAlmostEqual(f["init_new_token_ratio_derived"], 0.7)
        mt = X.mem_terms(f)
        self.assertAlmostEqual(mt["s"] / 143.5e6, 1.0, delta=0.005)
        self.assertAlmostEqual(mt["k"] / 16384, 1.0, delta=0.002)

    def test_real_vanilla_log_labels(self):
        f = self.facts("dummy_a4_vanilla")
        plan = R._boot("J-H1", 1, "x", "T1T2", "nano", "vanilla", 128, variant=4, dummy=True)
        self.assertEqual([p for p in X.label_problems(plan, f) if not p.startswith("ABORT_CONTEXT")], [])
        bad = X.label_problems(dict(plan, model="llama"), f)
        self.assertTrue(any(p.startswith("ABORT_MODEL_LABEL") for p in bad))
        bad = X.label_problems(dict(plan, dummy=False), f)
        self.assertTrue(any(p.startswith("ABORT_LOAD_FORMAT_LABEL") for p in bad))
        bad = X.label_problems(dict(plan, cap=48), f)
        self.assertTrue(any(p.startswith("ABORT_CAP_MISMATCH") for p in bad))
        bad = X.label_problems(dict(plan, variant=13), f)
        self.assertTrue(any(p.startswith("ABORT_VARIANT_LABEL") for p in bad))

    def test_real_a31_log(self):
        mt = X.mem_terms(self.facts("dummy_a31"))
        self.assertEqual(mt["s"], 0.0)
        self.assertAlmostEqual(mt["k"] / 126976, 1.0, delta=0.002)
        self.assertEqual(F.compo_mem(31, mt["s"], mt["k"], mt["slots"])["label"], "COMPO_MEM_HELD")


# =============================================================================== extractor units
class TestExtractUnits(unittest.TestCase):
    def test_ctx_band_inclusive(self):
        self.assertEqual(M.o_ctx_band(), [])

    def test_window_status_paths(self):
        c = 8192
        c_in = c - 166
        recs, cell = M.synth_window(c, c_in)
        w = X.t1_window(recs, cell, [1, 2, 4, 8])
        self.assertEqual(w["status"], "VALID")
        self.assertAlmostEqual(w["S_eng_ms"], 10.0)
        self.assertEqual(w["n_intervals"], S.WINDOW_ITERS)
        self.assertEqual(w["seq_range"], (1 + S.TRIM_ITERS, 1 + S.TRIM_ITERS + S.WINDOW_ITERS))
        self.assertAlmostEqual(w["ctx_mean"], c_in + S.TRIM_ITERS + (S.WINDOW_ITERS - 1) / 2)
        r2 = [dict(e) for e in recs]
        del r2[100]                                                    # seq gap inside the window
        self.assertEqual(X.t1_window(r2, cell, [1, 2, 4, 8])["status"], "RECORDER_UNPROVEN")
        # ramp: records below bs first -> the window starts TRIM after the first bs_real == bs record
        rr = [dict(e, bs_real=(2 if e["seq"] <= 5 else 4)) for e in recs]
        wr = X.t1_window(rr, cell, [1, 2, 4, 8])
        self.assertEqual(wr["status"], "VALID")
        self.assertEqual(wr["seq_range"][0], 6 + S.TRIM_ITERS)
        # bs changes / mixed forwards inside the window are excluded and counted
        r5 = [dict(e) for e in recs]
        for e in r5[50:60]:
            e["bs_real"] = 3
        r5[70]["mode"] = "mixed"
        w5 = X.t1_window(r5, cell, [1, 2, 4, 8])
        self.assertEqual((w5["status"], w5["excluded"]["bs_change"], w5["excluded"]["mixed"]), ("VALID", 10, 1))
        for e in r5[100:140]:
            e["bs_real"] = 3
        self.assertEqual(X.t1_window(r5, cell, [1, 2, 4, 8])["status"], "WINDOW_CONTAMINATED")
        r6 = [dict(e) for e in recs]
        r6[200]["bs_real"] = 5                                          # a foreign request
        self.assertEqual(X.t1_window(r6, cell, [1, 2, 4, 8])["status"], "FOREIGN_RESIDUAL")
        r7 = [dict(e, t_host_start=e["t_host_start"] - 1.0) if e["seq"] <= 3 else e for e in recs]
        self.assertEqual(X.t1_window(r7, dict(cell, t_first_send=-2.0), [1, 2, 4, 8])["status"], "FOREIGN_RESIDUAL")
        r3 = [dict(e) for e in recs]
        r3[100]["graph"] = "eager"
        self.assertEqual(X.t1_window(r3, cell, [1, 2, 4, 8])["status"], "CG_UNPROVEN")
        self.assertEqual(X.t1_window(recs, cell, [1, 2, 8])["status"], "PAD_MISMATCH")      # 4 -> 8 expected
        r4 = [dict(e) for e in recs]
        for e in r4[20:30]:
            e["decode_fwd_gpu_ms"] = None
        w4 = X.t1_window(r4, cell, [1, 2, 4, 8])
        self.assertEqual(w4["status"], "VALID")
        self.assertIn("I4_UNRESOLVED", w4["labels"])
        self.assertEqual(X.t1_window(recs, cell, [1, 2, 4, 8], "x #retracted_reqs: 2, y")["status"], "RETRACT")
        short = recs[:S.TRIM_ITERS + S.WINDOW_ITERS - 1]
        self.assertEqual(X.t1_window(short, cell, [1, 2, 4, 8])["status"], "WINDOW_SHORT")
        few = dict(cell, requests=[dict(r, completion_tokens=S.ABORT_AT_TOKENS - 1) for r in cell["requests"]])
        self.assertEqual(X.t1_window(recs, few, [1, 2, 4, 8])["status"], "BENCH_INVALID")

    def test_t2_excludes_first_rep(self):
        reps = [{"rep": i, "ttft_s": (2.0 if i == 0 else 1.0 + 0.001 * i), "ok": True} for i in range(8)]
        v = X.t2_value({"L": 4096, "reps": reps})
        self.assertEqual(v["n_used"], 7)
        self.assertNotIn(0, v["used_reps"])
        self.assertAlmostEqual(v["ttft_ms"], 1004.0)
        reps[3]["ok"] = False
        self.assertEqual(X.t2_value({"L": 4096, "reps": reps})["status"], "INVALID")

    def test_fill_incomplete_rule(self):
        def req(t_first, t_done):
            return {"chunks": [[t_first, 1], [t_done, 10]], "prompt_tokens": 100, "completion_tokens": 10,
                    "finish": "length", "error": None}
        recs = [{"event": "decode_forward", "mode": "decode", "bs_real": 3, "t_host_start": 1.5}]
        ok = {"L": 100, "R": 10, "N": 4, "t_first_send": 0, "t_end": 9,
              "requests": [req(1, 5), req(1.1, 5.1), req(1.2, 5.2), req(5.5, 8)]}
        self.assertNotIn("FILL_INCOMPLETE", X.t4_fill(ok, recs)["labels"])
        late = dict(ok, requests=[req(1, 5), req(1.1, 5.1), req(5.3, 8), req(5.5, 8)])
        self.assertIn("FILL_INCOMPLETE", X.t4_fill(late, recs)["labels"])

    def test_realized_sm_rule(self):
        b = R._boot("J", 1, "x", "T1T2", "nano", "pdmux_sticky", 128, D=34)

        def green(dec, pre):
            core = {"groups": [{"stream_index": i} for i in range(7)], "sticky_target_index": 3}
            core["groups"][3] = {"stream_index": 3, "decode": {"green_sm": {"smCount": dec}},
                                 "prefill": {"green_sm": {"smCount": pre}}}
            return core
        self.assertTrue(X.realized_sm(b, green(36, 72))["ok"])
        self.assertEqual(X.realized_sm(b, green(36, 72))["D_real"], 36)
        self.assertEqual(X.realized_sm(b, green(40, 68))["label"], "SM_NOT_REALIZED_D34")    # |40-34| > 4
        self.assertEqual(X.realized_sm(b, green(36, 76))["label"], "SM_NOT_REALIZED_D34")    # 36+76 > 108
        self.assertEqual(X.realized_sm(b, None)["label"], "SM_NOT_REALIZED_D34")

    def test_instrumentation_matrix(self):
        for kind in ("T1T2", "T4", "OFF"):
            b = R._boot("J", 1, "x", kind, "nano", "pdmux_sticky", 128, D=44)
            env = R.boot_env(b, Path("/nonexistent"), base={})
            self.assertEqual(X.instrumentation_problems(kind, env, None, None), [], kind)
        env = R.boot_env(R._boot("J", 1, "x", "OFF", "nano", "vanilla", 128), Path("/n"), base={})
        env["PDMUX_DECODE_PROBE_PATH"] = "/n/x"
        self.assertTrue(X.instrumentation_problems("OFF", env, None, None))


# =============================================================================== fit units
class TestFitUnits(unittest.TestCase):
    def test_f1_recovers_and_extrapolates(self):
        pts = [(b, c, 8.0 + 0.2 * b + 9e-6 * b * c) for b in (1, 8, 32, 96) for c in (2048, 8192, 32768) if b * c < 1e6]
        fit = F.fit_t1(pts, "D44")
        self.assertIsNone(fit["label"])
        co = fit["chosen"]["coef"]
        self.assertAlmostEqual(co[0], 8.0, places=6)
        self.assertAlmostEqual(co[1], 0.2, places=6)
        self.assertAlmostEqual(co[2], 9e-6, places=12)
        self.assertEqual(fit["chosen"]["extrap"]["ctx"]["level"], 32768)
        self.assertEqual(fit["chosen"]["extrap"]["bs"]["level"], 96)

    def test_fallback_to_f2_and_inadequate(self):
        pts = [(b, c, 8.0 + 0.2 * b + 0.02 * b * b + 9e-6 * b * c) for b in (1, 8, 32, 48, 96) for c in (2048, 8192)]
        fit = F.fit_t1(pts, "D16")
        self.assertEqual(fit["chosen"]["form"], "F2")
        bad = [(b, c, 10.0 * (1 + ((b * 7 + c) % 5) / 10)) for b in (1, 8, 32, 96) for c in (2048, 8192, 32768)]
        self.assertEqual(F.fit_t1(bad, "D16")["label"], "FORM_INADEQUATE_T1_D16")

    def test_extrap_na(self):
        pts = [(8, c, 10 + c * 1e-4) for c in (2048, 8192)]
        self.assertTrue(F.extrap_test(pts, "F1", "ctx").get("NA"))

    def test_t2_quadratic_and_q(self):
        Ls = list(S.T2_L_GRID)
        pts = [(L, 15 + 0.1 * L + 3e-7 * L * L) for L in Ls]
        ft = F.fit_t2(pts)
        self.assertTrue(ft["accepted"])
        q = F.q_index(pts, ft)
        want = ((15 + 0.1 * 65536 + 3e-7 * 65536 ** 2) / 65536) / ((15 + 0.1 * 8800 + 3e-7 * 8800 ** 2) / 8800)
        self.assertAlmostEqual(q["q"], want)
        self.assertFalse(q["from_fit"])
        q2 = F.q_index([p for p in pts if p[0] != 8800], ft)
        self.assertTrue(q2["from_fit"])
        self.assertEqual(F.q_label([1.2, 1.3]), "PREFILL_SUPERLINEAR_SCOPED")
        self.assertEqual(F.q_label([1.0, 1.05]), "PREFILL_NEAR_LINEAR_SCOPED")
        self.assertEqual(F.q_label([1.05, 1.15]), "Q_UNRESOLVED")

    def test_labels(self):
        ag = F.agreement_labels([{"model": "m", "mode": "108v", "bs": 8, "S_eng_ms": 10.0, "client_gap_ms": 10.29},
                                 {"model": "m", "mode": "108v", "bs": 32, "S_eng_ms": 10.0, "client_gap_ms": 10.31},
                                 {"model": "m", "mode": "108v", "bs": 48, "S_eng_ms": None, "client_gap_ms": 10.0}])
        self.assertEqual(ag[("m", "108v", 8)]["label"], "CLIENT_ENGINE_AGREE_bs8")
        self.assertEqual(ag[("m", "108v", 32)]["label"], "CLIENT_ENGINE_MISMATCH_bs32")
        self.assertEqual(ag[("m", "108v", 48)]["label"], "NOT_COLLECTED")
        self.assertEqual(F.w_indep({1: 10.0}, {1: 10.19})["label"], "WEIGHT_INDEP_HELD")
        self.assertEqual(F.w_indep({1: 10.0}, {1: 10.21})["label"], "WEIGHT_DEPENDENT")
        self.assertEqual(F.ctx_slope_prop({4: 1.0, 13: 3.25, 22: 5.5, 31: 7.75 * 1.24}, "WEIGHT_INDEP_HELD")["label"],
                         "CTX_SLOPE_PROP_HELD")
        self.assertEqual(F.ctx_slope_prop({4: 1.0, 13: 3.25 * 1.3}, "WEIGHT_INDEP_HELD")["label"], "CTX_SLOPE_PROP_MISS")
        self.assertIsNone(F.ctx_slope_prop({4: 1.0, 13: 3.25}, "WEIGHT_DEPENDENT")["label"])
        self.assertEqual(F.q_monotone({4: 1.1, 13: 1.2, 22: 1.3, 31: 1.4}, "WEIGHT_INDEP_HELD")["label"], "PREFILL_Q_MONOTONE")
        self.assertEqual(F.q_monotone({4: 1.1, 13: 1.2, 22: 1.2, 31: 1.4}, "WEIGHT_INDEP_HELD")["label"],
                         "PREFILL_Q_NOT_MONOTONE")
        cells = {c: 10.0 for c in S.CAP_INV_CELLS}
        self.assertEqual(F.cap_invariance(cells, {c: 10.25 for c in cells})["label"], "CAP_INVARIANT")
        self.assertEqual(F.cap_invariance(cells, {c: 10.4 for c in cells})["label"], "CAP_DEPENDENT")
        pred = X.cmem_predictions(S.PREREG_POOL_BYTES, S.PREREG_S_BYTES, S.PREREG_K_BYTES, 48, 65536, 8000)
        lab = F.t4_labels({"L": 65536, "n_real": 21, "labels": []}, pred)["labels"]
        self.assertEqual(lab, ["CMEM_ACCT_HELD", "CMEM_ORIG_OVERPREDICTS"])
        self.assertEqual(F.c_mem_drift({"a": 100.0, "b": 101.0, "c": 104.0})["label"], "C_MEM_DRIFT")


# =============================================================================== virtual end to end
class TestVirtualE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="ct_e2e_"))
        cls.runs = {}
        for sc in ("JH1", "JH1F", "JAF", "JB2", "JA0", "T3S", "JC"):
            d = cls.root / sc
            cls.runs[sc] = (d, M.run_scenario(sc, d, SRC))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, True)

    def truth_coef(self, n_attn, d_sm=108, vanilla=True, truth=None):
        t = dict(FE.DEFAULT_TRUTH, **(truth or {}))
        core = FE.EngineCore(["--model-path", str(self.root / "JH1" / "variants" / f"a{n_attn}"),
                              "--max-running-requests", "128"], {}, t)
        sc = core._scales()
        r = 108 / d_sm
        m = t["vanilla_mult"] if vanilla else 1.0
        return (t["a108"] * sc["a"] * r ** t["ga"] * m, t["b108"] * sc["b"] * r ** t["gb"] * m,
                t["c108"] * sc["c"] * r ** t["gc"] * m)

    def test_jh1_ground_truth_and_labels(self):
        _, A = self.runs["JH1"]
        for n in S.VARIANTS:
            fit = A["fits"]["t1"][f"J-H1|r1|dummy_a{n}|108v|cap128"]["chosen"]
            a, b, c = self.truth_coef(n)
            # rev3.4: cells are fitted at the ACTUAL window mean ctx, so F1 returns the truth; the client count lags the
            # engine by the one in-flight token (A-26), which moves at most 1 x c into b
            got = fit["coef"]
            self.assertAlmostEqual(got[0] / a, 1.0, delta=1e-6)
            self.assertAlmostEqual(got[2] / c, 1.0, delta=1e-6)
            self.assertLessEqual(abs(got[1] - b), 1.0001 * c)
        jh = A["jh1"]
        self.assertEqual(jh["w_indep"]["T1"]["label"], "WEIGHT_INDEP_HELD")
        self.assertEqual(jh["w_indep"]["T2"]["label"], "WEIGHT_INDEP_HELD")
        self.assertTrue(all(v["label"] == "COMPO_MEM_HELD" for v in jh["P1_compo_mem"].values()))
        self.assertEqual({v["label"] for v in jh["P2_cmem"].values()},
                         {f"CMEM_ACCT_HELD_a{n}" for n in S.VARIANTS})
        self.assertEqual(jh["P3_ctx_slope"]["label"], "CTX_SLOPE_PROP_HELD")
        self.assertEqual(jh["P4_q"]["label"], "PREFILL_Q_MONOTONE")
        self.assertTrue(all("AGREE" in v["label"] for v in A["labels"]["agreement"].values()))
        self.assertEqual(M.o_infeasible_rows(*self.runs["JH1"]), [])
        self.assertEqual(M.o_model_labels(*self.runs["JH1"]), [])
        for v in A["instr_effect"].values():
            self.assertEqual(v["T1"]["label"], "INSTR_EFFECT_SMALL")

    def test_jh1_faults_detected(self):
        d, A = self.runs["JH1F"]
        self.assertEqual(A["jh1"]["w_indep"]["T1"]["label"], "WEIGHT_DEPENDENT")
        self.assertIn("NOT_EVALUATED", A["jh1"]["P3_ctx_slope"]["status"])
        self.assertEqual(M.o_mismatch_excluded(d, A), [])
        self.assertEqual(M.o_gpu_field(d, A), [])
        self.assertEqual(M.o_seng_is_engine(d, A), [])

    def test_ja_fast_gamma_and_axes(self):
        d, A = self.runs["JAF"]
        g = A["fits"]["gamma_j"]["J-A1|r1"]
        self.assertAlmostEqual(g["a"], 0.5, delta=0.01)
        self.assertAlmostEqual(g["b"], 0.8, delta=0.01)
        self.assertAlmostEqual(g["c"], 0.9, delta=0.01)
        self.assertEqual(M.o_realized_axis(d, A), [])
        self.assertEqual(M.o_excluded_boot(d, A), [])
        self.assertEqual(M.o_modes_separate(d, A), [])
        self.assertEqual(M.o_off_clean(d, A), [])
        # decision 28 A-2: T2 only in 108u / 108v; the sticky-D P axis is T2' with engine-confirmed co-residence
        t2, t2p = A["tables"]["t2"], A["tables"]["t2prime"]
        self.assertEqual(sorted(k.split("|")[3] for k in t2), ["108u", "108v"])
        self.assertEqual(t2p["J-A1|r1|real_a4|D44"]["P"], 64)
        self.assertEqual(t2p["J-A1|r1|real_a4|D34"]["P"], 72)                      # realized 36 -> 108 - 36
        self.assertEqual(sorted(t2p["J-A1|r1|real_a4|D44"]["points_confirmed"]), sorted(map(str, S.T2P_L_GRID)))

    def test_ja_ramp_aware_windows(self):
        """Decision 28 A-1 with the fake's (invented) prefill speed: every feasible PD-mux cell is VALID (rev3.3:
        5-6 of 14-21), the window starts after the first full-batch record, the fit at the actual ctx returns the
        truth, and vanilla T2' is not engine-confirmed (prefill first) while PD-mux T2' is."""
        d, A = self.runs["JA0"]
        self.assertEqual(M.o_window_reach(d, A), [])
        self.assertEqual(M.o_fit_truth(d, A), [])
        self.assertEqual(M.o_no_residual(d, A), [])
        self.assertEqual(M.o_t2p_confirm(d, A), [])
        self.assertEqual(M.o_t2_p108(), [])
        ws = A["boots"]["ja1_D44"]["t1_windows"]
        self.assertTrue(all(w["ramp_records"] <= w["ramp_steps_ub"] for w in ws))
        self.assertTrue(any(w["ramp_records"] > 0 and "CTX_OFF_NOMINAL" in w["labels"] for w in ws))

    def test_jb2_t4(self):
        d, A = self.runs["JB2"]
        t4 = A["tables"]["t4"]
        self.assertEqual(t4["jb2_cap48|L65536"]["labels"], ["CMEM_ACCT_HELD", "CMEM_ORIG_OVERPREDICTS"])
        self.assertEqual((t4["jb2_cap48|L65536"]["n_A"], t4["jb2_cap48|L65536"]["n_B"]), (24, 21))
        self.assertIn("CMEM_ACCT_HELD", t4["jb2_capCB_L65536|L65536"]["labels"])
        self.assertIsNone(A["labels"]["c_mem_drift"]["label"])
        self.assertEqual(M.o_sealed(d, A), [])
        self.assertEqual(M.o_n_real(d, A), [])
        for x in t4.values():
            if "FILL_INCOMPLETE" in x["labels"]:
                self.assertTrue(x["excluded"])
                self.assertFalse(any(l.startswith("CMEM_") for l in x["labels"]))

    def test_virtual_determinism_and_resume(self):
        d = tmpdir(self)
        plan = M.sub_plan("J-H1", ["jh1_llama"])
        A1 = M.run_scenario("JH1", d / "a", SRC, plan=plan)
        A2 = M.run_scenario("JH1", d / "b", SRC, plan=plan)
        self.assertEqual(A1["tables"], A2["tables"])
        before = (d / "a" / "boots" / "jh1_llama" / "cells.jsonl").read_bytes()
        R.run_job("J-H1", d / "a", 1, SRC, d / "a" / "variants",
                  make_server=lambda b, a, e: R.VirtualServer(b, a, e, {}), plan=plan)     # DONE -> skipped
        self.assertEqual((d / "a" / "boots" / "jh1_llama" / "cells.jsonl").read_bytes(), before)

    # ---------------------------------------------------------------------- mutation ledger (lesson 53)
    def test_mutants_caught(self):
        caught, rows = 0, []
        for mid, desc, stage, sc, fac, oracle, sub in M.MUTANTS:
            with self.subTest(mutant=mid, desc=desc):
                if stage == "unit":
                    base = oracle(None, None)
                    with M.applied(fac):
                        mut = oracle(None, None)
                elif stage == "analysis":
                    d, A = self.runs[sc]
                    base = oracle(d, A)
                    with M.applied(fac):
                        B = R.analyze(d)
                    mut = oracle(d, B)
                    R.analyze(d)
                else:
                    plan = M.cg_plan() if sc == "CG" else M.sub_plan(*sub)
                    root = tmpdir(self)
                    base = oracle(root / "base", M.run_scenario(sc, root / "base", SRC, plan=plan))
                    with M.applied(fac):
                        try:
                            mut = oracle(root / "mut", M.run_scenario(sc, root / "mut", SRC, plan=plan))
                        except Exception as exc:  # noqa: BLE001 - a crash is also a detection
                            mut = [f"exception {exc!r}"]
                self.assertEqual(base, [], f"{mid}: oracle fails on the unmutated harness")
                self.assertTrue(mut, f"{mid} ({desc}) NOT caught")
                caught += bool(mut) and not base
                rows.append((mid, bool(mut)))
        self.assertEqual(caught, len(M.MUTANTS))


# =============================================================================== HTTP + process end to end
class TestHttpE2E(unittest.TestCase):
    def test_http_fake_server(self):
        d = tmpdir(self)
        env = {"PDMUX_CT_SERVER_CMD": f"{sys.executable} {H / 'ct_fake_engine.py'}",
               "FAKE_CT_TRUTH": json.dumps({"a108": 2.0, "b108": 0.02, "c108": 1e-6, "pf_u": 0.5, "pf_v": 1e-5,
                                            "pf_w4": 0.0})}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            mk = lambda **kw: R._boot("J-TEST", 1, kw.pop("bid"), kw.pop("kind"), "nano", kw.pop("engine"), 128, **kw)  # noqa: E731
            plan = {"job": "J-TEST", "deferred": [], "boots": [
                mk(bid="http_v", kind="T1T2", engine="vanilla", t1=[(8, 2048)], t2=[1024, 4096], t2p=[4096]),
                mk(bid="http_bad", kind="T1T2", engine="pdmux_sticky", D=50, t1=[(8, 2048)]),    # no D=50 division
                mk(bid="http_D44", kind="T4", engine="pdmux_sticky", D=44, t1=[(8, 2048)], t4=[(4096, 64)]),
                mk(bid="http_off", kind="OFF", engine="vanilla", t1=[(8, 2048)])]}
            res = {r["boot_id"]: r for r in R.run_job("J-TEST", d, plan=plan)}
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(res["http_bad"]["labels"], ["SERVER_DEATH"])
        self.assertEqual({k: r["start"] for k, r in res.items() if k != "http_bad"}, dict.fromkeys(("http_v", "http_D44", "http_off"), "ok"))
        A = R.analyze(d)
        v = A["boots"]["http_v"]
        w = v["t1_windows"][0]
        self.assertEqual(w["status"], "VALID")
        want = (2.0 + 0.02 * 8 + 1e-6 * 8 * 2048) * 1.03
        self.assertAlmostEqual(w["S_eng_ms"] / want, 1.0, delta=0.15)      # real-time sleeps: loose
        self.assertTrue(all("AGREE" in x["label"] for x in A["labels"]["agreement"].values()))
        self.assertEqual([x["status"] for x in v["t2"] + v["t2p"]], ["VALID"] * 3)
        t4 = A["boots"]["http_D44"]["t4"][0]
        self.assertEqual(t4["n_real"], 128)
        self.assertEqual(A["boots"]["http_D44"]["t4_prediction_order"], "sealed before fill")
        self.assertEqual(A["boots"]["http_D44"]["sm"]["D_real"], 44)
        off = A["boots"]["http_off"]
        self.assertEqual(off["t1_windows"][0]["engine_records"], "NOT_COLLECTED")
        self.assertEqual(off["problems"], [])
        self.assertTrue((d / "RUN_DONE").exists() and (d / "SHA256SUMS").exists())


# =============================================================================== entry points
class TestEntry(unittest.TestCase):
    def test_bash_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(H / "costterms_job.sh")]).returncode, 0)

    def test_unsealed_refusal(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_COSTTERMS")}
        r = subprocess.run([sys.executable, str(H / "ct_runner.py"), "run", "--job", "J-H1", "--out", str(tmpdir(self))],
                           env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 16)
        self.assertIn("ABORT_UNSEALED", r.stderr)
        r = subprocess.run(["bash", str(H / "costterms_job.sh")], env=dict(env, PDMUX_COSTTERMS_JOB="J-H1"),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 16)
        r = subprocess.run(["bash", str(H / "costterms_job.sh")], env=dict(env, PDMUX_COSTTERMS_JOB="J-X"),
                           capture_output=True, text=True)
        self.assertIn("ABORT_CONFIG", r.stderr)
        for job in ("J-H2", "J-C"):                                               # stage 2 jobs are known, unsealed
            r = subprocess.run(["bash", str(H / "costterms_job.sh")], env=dict(env, PDMUX_COSTTERMS_JOB=job),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 16)
            self.assertIn("ABORT_UNSEALED", r.stderr)

    def test_engine_digest_gate(self):
        """H-2: a manifest that does not reproduce ENGINE_DIGEST.txt (here: absent) aborts with rc 13 before any run."""
        self.assertEqual((H / "ENGINE_DIGEST.txt").read_text().strip(), S.ENGINE_DIGEST_EXPECTED)
        d = tmpdir(self)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_COSTTERMS")}
        env.update(PDMUX_COSTTERMS_JOB="J-B2", PDMUX_COSTTERMS_TEST_MODE="1", PDMUX_DATA=str(d), PDMUX_JOB_NAME="t",
                   PDMUX_COSTTERMS_SUMMARY=str(d / "sum"), PDMUX_COSTTERMS_OUT=str(d / "raw"))
        r = subprocess.run(["bash", str(H / "costterms_job.sh")], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 13, r.stderr)
        self.assertIn("ABORT_ENGINE_DIGEST", r.stderr)
        self.assertFalse((d / "raw" / "plan.json").exists())

    def test_plans(self):
        for job, rnd in (("J-H1", 1), ("J-A", 1), ("J-A", 2), ("J-B2", 1)):
            p = R.make_plan(job, rnd)
            self.assertEqual(R.plan_problems(p), [], (job, rnd))
        a1, a2 = R.make_plan("J-A", 1), R.make_plan("J-A", 2)
        self.assertEqual([b["boot_id"].replace("ja1_", "") for b in a1["boots"] if "capinv" not in b["boot_id"]][::-1],
                         [b["boot_id"].replace("ja2_", "") for b in a2["boots"]])
        self.assertEqual(len(R.make_plan("J-B2")["boots"]), 10)
        self.assertEqual(len(R.make_plan("J-H1")["boots"]), 7)
        self.assertEqual(len(R.make_plan("J-A", 1)["boots"]), 12)                  # + T3 boots D24 / D44
        self.assertEqual(len(R.make_plan("J-A", 2)["boots"]), 11)
        for job in ("J-H2", "J-C"):
            self.assertEqual(R.plan_problems(R.make_plan(job)), [], job)
        jh2 = R.make_plan("J-H2")["boots"]
        self.assertEqual(sorted((b["variant"], b["D"]) for b in jh2), sorted((n, d) for n in S.VARIANTS for d in S.JH2_D))
        self.assertTrue(all(b["kind"] == "T3" and b["dummy"] for b in jh2))
        jc = R.make_plan("J-C")["boots"]
        self.assertEqual(len(jc), 10)                                               # 96k deleted by default (O-19)
        self.assertEqual({(b["tp"], b["cap"], b["gpus"]) for b in jc}, {(2, 128, "0,1"), (1, 48, "0")})
        r1 = [b["boot_id"] for b in jc if b["boot_id"].startswith("jc1_")]
        r2 = [b["boot_id"] for b in jc if b["boot_id"].startswith("jc2_")]
        self.assertEqual([x.replace("jc1_", "") for x in r1][::-1], [x.replace("jc2_", "") for x in r2])   # ABBA
        with mock_attr(S, "JC_96K_MODE", "separate_boot"):
            p = R.make_plan("J-C")
            self.assertEqual(p["boots"][-1]["long"], [S.JC_LONG_L])
            self.assertEqual(R.plan_problems(p), [])
            p["boots"].insert(0, p["boots"].pop())                               # the 96k boot must be last
            self.assertTrue(any("LAST" in x for x in R.plan_problems(p)))
        with self.assertRaises(SystemExit):
            os.environ.pop("PDMUX_COSTTERMS_TEST_MODE")
            try:
                R.run_job("J-H1", tmpdir(self), plan={"boots": []})
            finally:
                os.environ["PDMUX_COSTTERMS_TEST_MODE"] = "1"

    def test_config_candidate_matches_spec(self):
        groups = FE.read_pdmux_config(H / S.CONFIG_CANDIDATE)
        self.assertEqual(tuple(groups), S.STREAM_GROUPS)
        for d, i in S.STICKY_IDX.items():
            self.assertEqual(groups[i][1], d)


# =============================================================================== stage 2 (T3, J-H2, J-C, Job entry)
class TestStage2Spec(unittest.TestCase):
    PROBE_POOLS = {4: 886804, 13: 419117, 22: 334083, 31: 298407}     # compo probe banners (design input, not citable)

    def test_sigma_single_knob(self):
        """H-6: RAMP_SIGMA alone moves every registered output length (T1 burst, T3 decode, sigma' reference)."""
        base = (S.o_tokens(32, 8 * K, "JA"), S.t3_o_tokens(32, 8 * K, 44), S.t3_ref_o_tokens(32 * K, 44),
                S.o_tokens(32, 8 * K, "JC|44"), S.o_tokens(32, 8 * K, "JH2|13|24"))
        with mock_attr(S, "RAMP_SIGMA", 0.5):
            low = (S.o_tokens(32, 8 * K, "JA"), S.t3_o_tokens(32, 8 * K, 44), S.t3_ref_o_tokens(32 * K, 44),
                   S.o_tokens(32, 8 * K, "JC|44"), S.o_tokens(32, 8 * K, "JH2|13|24"))
        self.assertTrue(all(a > b for a, b in zip(base, low)), (base, low))
        self.assertEqual(S.o_tokens(32, 8 * K, "JA"), 6173)                          # stage-1 values unchanged
        self.assertEqual(S.T3_PRE_TOKENS, 169)
        self.assertEqual(S.T3_POST_TOKENS, 154)

    def test_jh2_feasibility_on_probe_pools(self):
        """rev3.5 draft §4' table: bs 8 T3 cells run in every variant; bs 32 only where the pool allows."""
        cap = S.default_capture_list(128)
        run = {}
        for n in S.VARIANTS:
            for d in S.JH2_D:
                a = S.variant_arch(n)
                for bs, L in [(b, l) for b in S.T3_BS for l in S.T3_L]:
                    r = S.feasible_t3(bs, L, S.t3_o_tokens(bs, L, d, a), self.PROBE_POOLS[n], 128, bs in cap) + \
                        S.feasible_t3(1, L, S.t3_ref_o_tokens(L, d, a), self.PROBE_POOLS[n], 128, True)
                    run[(n, d, bs, L)] = not r
                for bs, c in S.JH2_T1_CELLS:
                    run[(n, d, "t1", c)] = not S.feasible_cell(bs, c, S.o_tokens(bs, c, f"JH2|{n}|{d}"),
                                                               self.PROBE_POOLS[n], 128, True)
        self.assertTrue(all(run[(n, d, 8, L)] for n in S.VARIANTS for d in S.JH2_D for L in S.T3_L))
        self.assertTrue(all(run[(4, d, 32, L)] for d in S.JH2_D for L in S.T3_L))
        self.assertEqual([k for k, v in run.items() if k[0] == 13 and k[2] == 32 and v], [(13, 24, 32, 8 * K)])
        self.assertFalse(any(v for k, v in run.items() if k[0] in (22, 31) and k[2] == 32))
        self.assertEqual({k for k, v in run.items() if k[2] == "t1" and not v},
                         {(n, d, "t1", 8 * K) for n in (22, 31) for d in S.JH2_D})

    def test_jc_feasibility(self):
        """O-19 / A-32: one O per (cell, mode) for TP2 and the anchor; the anchor's cap-48 pool loses (32, 32k) D44."""
        cap48 = S.default_capture_list(48)
        for bs, c in S.JC_T1_CELLS:
            O = S.o_tokens(bs, c, "JC|44")
            n1 = S.feasible_cell(bs, c, O, 1588000, 48, bs in cap48)
            tp2 = S.feasible_cell(bs, c, O, 3720000, 128, True)
            self.assertEqual(not n1, (bs, c) != (32, 32 * K), (bs, c, n1))
            self.assertEqual(tp2, [], (bs, c))
            self.assertEqual(S.feasible_cell(bs, c, S.o_tokens(bs, c, "VANILLA"), 1588000, 48, True), [])
        self.assertGreater(S.o_tokens(32, 8 * K, "JC|44"), S.o_tokens(32, 8 * K, "JA"))
        self.assertEqual(int(2 ** 31 // (S.JC_INPROJ_N1 / 2)), 189573)              # 2 x 94,786.5

    def test_time_model_and_caps(self):
        tm = importlib.import_module("ct_timemodel")
        h = {k: v[0] for k, v in tm.main().items()}
        self.assertAlmostEqual(h["J-A|1"], 3.4, delta=0.05)                         # PREREG §7-4' J-A1 3.4 h
        self.assertAlmostEqual(h["J-A|2"], 3.3, delta=0.05)
        self.assertAlmostEqual(h["J-H1|1"], 3.4, delta=0.05)
        for k in ("J-A|1", "J-A|2", "J-H1"):
            self.assertGreaterEqual(S.HARD_CAP_MIN[k], 1.2 * h[k if "|" in k else f"{k}|1"] * 60, k)
        self.assertGreaterEqual(S.HARD_CAP_MIN["J-H2"], 1.2 * h["J-H2|1"] * 60)
        self.assertGreaterEqual(S.HARD_CAP_MIN["J-C"], 1.2 * h["J-C|1"] * 60)
        txt = (H / "costterms_job.sh").read_text()
        import re
        for job, cap in (("J-H2", S.HARD_CAP_MIN["J-H2"]), ("J-C", S.HARD_CAP_MIN["J-C"])):
            self.assertRegex(txt, rf"{job}\) +cap_min={cap};")
        launch = (ENG / "scripts" / "vessl" / "launch.sh").read_text()
        self.assertIn(f"J-H2:*) ctcap={S.HARD_CAP_MIN['J-H2']}", launch)
        self.assertIn(f"J-C:*) ctcap={S.HARD_CAP_MIN['J-C']}", launch)

    def test_real_tp2_banner_lines(self):
        """Per-rank banner parse on lines copied verbatim from an M2G TP2 server log (2026-10-13)."""
        text = ("[2026-10-13 09:08:59.130 TP0] Load weight end. elapsed=7.78 s, type=NemotronHForCausalLM, avail mem=75.72 GB, mem usage=2.20 GB.\n"
                "[2026-10-13 09:08:59.158 TP1] Load weight end. elapsed=7.83 s, type=NemotronHForCausalLM, avail mem=75.58 GB, mem usage=2.20 GB.\n"
                "[2026-10-13 09:08:59.167 TP1] Mamba Cache is allocated. max_mamba_cache_size: 384, conv_state size: 0.09GB, ssm_state size: 6.34GB \n"
                "[2026-10-13 09:08:59.174 TP0] Mamba Cache is allocated. max_mamba_cache_size: 384, conv_state size: 0.09GB, ssm_state size: 6.34GB \n"
                "[2026-10-13 09:08:59.209 TP1] KV Cache is allocated. #tokens: 28920927, K size: 27.58 GB, V size: 27.58 GB\n")
        rb = X.rank_banners(text)
        self.assertEqual(sorted(rb), [0, 1])
        self.assertEqual(rb[1]["kv_cache_tokens"], 28920927)
        self.assertEqual(rb[0]["max_mamba_cache_size"], 384)
        self.assertNotIn("kv_cache_tokens", rb[0])
        self.assertFalse(X.tp_facts(rb, 2)["F5"]["held"])                            # rank 0 has no KV banner here


class TestStage2E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="ct_s2_"))
        cls.runs = {sc: (cls.root / sc, M.run_scenario(sc, cls.root / sc, SRC)) for sc in ("T3S", "JC")}
        d = cls.root / "JH2"
        R.run_job("J-H2", d, 1, SRC, d / "variants", plan=M.sub_plan("J-H2", ["jh2_a31_D24", "jh2_a4_D44"]),
                  make_server=lambda b, a, e: R.VirtualServer(b, a, e, dict(M.FAST_PF)))
        cls.runs["JH2"] = (d, R.analyze(d))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, True)

    def test_t3_truth(self):
        d, A = self.runs["T3S"]
        self.assertEqual(M.o_t3_window(d, A), [])
        self.assertEqual(M.o_t3_sigma_engine(d, A), [])
        self.assertEqual(M.o_t3_rho_stall(d, A), [])
        n = 0
        for key, ck, w in M._t3_reps(A):
            n += 1
            bs = w["bs"]
            truth = (1 + 0.004 * bs) / (1 + 0.004)                                   # fake pf_coex, 1-decode reference
            self.assertLessEqual(abs(w["sigma_prime"] / truth - 1), w["span_obs_lag_bound_rel"] + 1e-9, (key, ck))
            self.assertGreaterEqual(w["n_it_obs"], S.T3_MIN_ITERS)
            self.assertGreater(w["stall_co"]["n_stall"], 0)
            self.assertEqual(w["stall_solo"]["n_stall"], 0)
            self.assertEqual(w["status"], "VALID")
        self.assertEqual(n, 16)                                                      # 2 boots x 4 cells x 2 reps
        tab = A["tables"]["t3"]["J-A1|r1|real_a4|D44"]
        self.assertEqual(sorted(tab["cells"]), ["32x32768", "32x8192", "8x32768", "8x8192"])
        self.assertAlmostEqual(tab["cells"]["8x8192"]["ln_rho_mean"], math.log(1.25), delta=0.01)
        raw = json.loads((d / "t3_iterations.json").read_text())
        self.assertEqual(len(raw["J-A1|r1|real_a4|D44"]["8x8192"]["0"]["solo_ms"]), 300)
        self.assertNotIn("raw_co_ms", json.dumps(A))

    def test_t3_sigma_not_collected_without_tel(self):
        """A T3 cell analysed without TEL records: sigma' NA (never filled), SIGMA_NOT_COLLECTED, rho still computed."""
        d, A = self.runs["T3S"]
        c = tmpdir(self) / "copy"
        shutil.copytree(d, c)
        (c / "boots" / "ja1_t3_D44" / "tel.jsonl").unlink()
        B = R.analyze(c)
        v = B["boots"]["ja1_t3_D44"]
        self.assertIn("SIGMA_NOT_COLLECTED", v["labels"])
        tab = B["tables"]["t3"]["J-A1|r1|real_a4|D44"]["cells"]["8x8192"]
        self.assertIsNone(tab["ln_sigma_prime_mean"])
        self.assertEqual(tab["sigma_label"], "SIGMA_NOT_COLLECTED")
        self.assertTrue(all(w.get("sigma_prime") is None for w in tab["reps"]))

    def test_jc(self):
        d, A = self.runs["JC"]
        self.assertEqual(M.o_jc_ratios(d, A), [])
        self.assertEqual(M.o_raw_not_exported(d, A), [])
        jc = A["jc"]
        self.assertIsNone(jc["F5_label"])
        self.assertEqual(sorted(jc["ratios_t1"]["r1|D44"]), sorted(f"{b}x{c}" for b, c in S.JC_T1_CELLS
                                                                    if (b, c) != (32, 32 * K)))
        # fake span x tp_pf_mult 0.6; TTFT adds queue wait + observation at a decode boundary (decode x 0.62)
        self.assertTrue(all(0.6 - 1e-6 <= x <= 0.62 + 1e-6 for x in jc["ratios_t2p"]["r1|D44"].values()),
                        jc["ratios_t2p"]["r1|D44"])
        self.assertEqual({x["labels"][0] for x in jc["t4"]["jc_tp2_t4"]}, {"CMEM_ACCT_HELD"})
        self.assertTrue(all(r["bs_sequence_equal"] for r in jc["rank"].values()))
        v = A["boots"]["jc1_n1_D44"]
        self.assertEqual(v["identity"]["cap"], 48)
        self.assertTrue((d / "boots" / "jc1_tp2_D44" / "decode_probe.tp1.jsonl").exists())
        self.assertEqual(json.loads((d / "boots" / "jc1_tp2_D44" / "boot.json").read_text())["cuda_visible_devices"], "0,1")
        self.assertEqual(json.loads((d / "boots" / "jc1_n1_D44" / "boot.json").read_text())["cuda_visible_devices"], "0")

    def test_jh2(self):
        d, A = self.runs["JH2"]
        for bid in ("jh2_a31_D24", "jh2_a4_D44"):
            v = A["boots"][bid]
            self.assertTrue(v["usable"], v["problems"])
            self.assertEqual(len(v["feasibility_t3"]), 4)                     # every registered T3 cell is a row
            self.assertEqual(v["kind"], "T3")
            self.assertTrue((d / "boots" / bid / "tel.jsonl").exists())        # TEL ON in both D boots (symmetric)
        self.assertEqual(A["boots"]["jh2_a31_D24"]["identity"]["n_attn"], 31)
        self.assertIn("J-H2|r1|dummy_a31|D24", A["jh2"]["dummy_a31"]["t3_keys"])
        self.assertIsNotNone(A["tables"]["t3"]["J-H2|r1|dummy_a31|D24"]["a31_note"])

    def test_smoke_and_resume(self):
        self.assertEqual(M.o_smoke(), [])
        self.assertEqual(M.o_resume(), [])
        self.assertEqual(M.o_jc_anchor(), [])
        p = M.sub_plan("J-A", ["ja1_D44"])
        b = p["boots"][0]
        with self.assertRaises(SystemExit):
            R.select_cells(b, select=["ja1_D44:t2:4096"])                     # T2 is not registered in sticky D44
        keep = R.select_cells(b, cells_per_boot=2)
        self.assertEqual([k for k in keep if k.startswith("t1|")], ["t1|1x2048|0", "t1|8x2048|0"])
        with self.assertRaises(SystemExit):
            R.run_job("J-A", tmpdir(self), plan=p, select=["ja1_D44:t1:8x2048"])   # subsets are smoke-only
        d = tmpdir(self)
        mk = lambda b_, a, e: R.VirtualServer(b_, a, e, dict(M.FAST_PF))  # noqa: E731
        R.run_job("J-A", d, plan=p, make_server=mk, smoke=True, select=["ja1_D44:t1:8x2048"])
        os.environ.pop("PDMUX_COSTTERMS_TEST_MODE")
        try:
            with self.assertRaises(SystemExit):                                   # non-empty out without RESUME=1
                R.run_job("J-A", d, plan=None, smoke=True, boots=["ja1_D44"], select=["ja1_D44:t1:8x2048"])
        finally:
            os.environ["PDMUX_COSTTERMS_TEST_MODE"] = "1"


if __name__ == "__main__":
    unittest.main()
