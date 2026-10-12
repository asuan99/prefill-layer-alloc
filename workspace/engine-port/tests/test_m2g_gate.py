"""CPU checks for the M2g gate harness (PREREG_MULTIGPU_2026-10-12.md rev6 sec 3 / 3-1).

Nothing here touches a GPU.  Every scorer gets inputs it must FAIL on (an always-PASS
scorer is the identity gate of lesson 9), every registered margin gets a boundary
case plus a mutant that moves the margin and must flip that case (lesson 53), and
the unmutated module is the control (lesson 254).  Expected values are derived by
hand in the comments from the PREREG text, not computed by the code under test.
Supersedes tests/test_m2g_skeleton.py (the skeleton's comparators were replaced).
"""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ENG = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M2G = _load("m2g_correctness_gate", ENG / "scripts" / "multigpu" / "m2g_correctness_gate.py")
STRESS = _load("m2g_comm_reverse_stress", ENG / "scripts" / "multigpu" / "m2g_comm_reverse_stress.py")
GUARD = _load("multigpu_guard_t", ENG / "src" / "multiplex" / "multigpu_guard.py")
MIX = _load("mg_realized_mix_t", ENG / "results" / "multigpu" / "mg_realized_mix.py")
GUARD_PY = ENG / "src" / "multiplex" / "multigpu_guard.py"
CTX_PATCH = ENG / "src" / "patches" / "pdmux_context_device_assert.patch"
PREREG = (ENG / "results" / "multigpu" / "PREREG_MULTIGPU_2026-10-12.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------- registered constants
class TestRegisteredConstants(unittest.TestCase):
    def test_values(self):
        self.assertEqual((M2G.G1_M_MARGIN, M2G.G1_Q_MARGIN, M2G.G1_CTL_UNINFORMATIVE_BELOW), (0.05, 0.10, 0.5))
        self.assertEqual((M2G.G1_N_PROMPTS, M2G.G1_MAX_NEW, M2G.G1_CONCURRENCY), (64, 128, 32))
        self.assertEqual((M2G.G2_FLOOR_MULT, M2G.G2_ABS_NAT, M2G.G2_TOP1_MISMATCH_MAX, M2G.G2_PCTL), (4.0, 0.05, 0.02, 99.0))
        self.assertEqual((M2G.G3_REPEATS, M2G.G3_WATCHDOG_S), (500, 120.0))
        self.assertEqual(M2G.G3_COMBOS, (("plain", None), ("green", (16, 92)), ("green", (44, 64))))
        self.assertEqual((M2G.G3B_OFFER_MULT, M2G.G3B_PRIOR_A, M2G.G3B_DURATION_S, M2G.G3B_PROGRESS_WINDOW_S),
                         (1.5, 2.99, 600, 60.0))
        self.assertEqual(M2G.HEALTH_WAIT_S, 1800)
        self.assertEqual(M2G.NVML_EMPTY_GPU_MIB, 1024)
        self.assertEqual(M2G.HARD_CAP_S, 4 * 3600)               # rev8: M2g 4 h per Job
        self.assertEqual(set(M2G.C.HARD_CAP_S["m2g"].values()), {4 * 3600})
        self.assertEqual(M2G.C.HARD_CAP_S["m3"], {"J1": 10 * 3600, "J2": 14 * 3600, "J4": 14 * 3600, "J8": 14 * 3600})
        self.assertEqual(M2G.C.PREREG_REV, 10)

    def test_stress_script_uses_the_same_registered_values(self):
        self.assertEqual(STRESS.G3_REPEATS, M2G.G3_REPEATS)
        self.assertEqual(STRESS.G3_WATCHDOG_S, M2G.G3_WATCHDOG_S)
        self.assertEqual(tuple(STRESS.G3_GREEN_SPLITS), tuple(sp for s, sp in M2G.G3_COMBOS if sp))
        self.assertEqual(STRESS.expected_sums(4), (10.0, 16.0))  # 1+2+3+4, 1+3+5+7

    def test_prereg_text_carries_the_literals(self):
        for lit in ("−0.05", "q + 0.10", "max(4 × 바닥 p99, 0.05 nat)", "2%", "R = 500", "watchdog 120 s",
                    "1800 s", "16/92, 44/64", "1.5 × (N × 2.99)", "600 s", "마지막 60 s"):
            self.assertIn(lit, PREREG, lit)

    def test_markers_match_the_engine_text(self):
        self.assertIn(M2G.GUARD_REFUSAL_TEXT, GUARD_PY.read_text(encoding="utf-8"))
        self.assertIn(M2G.DEVICE_ASSERT_TEXT, CTX_PATCH.read_text(encoding="utf-8"))

    def test_g0ii_envs_are_the_five_and_each_is_refused_by_the_guard(self):
        self.assertEqual(len(M2G.G0_REFUSED_ENVS), 5)
        keys = [tuple(sorted(e.items()))[0] if len(e) == 1 else ("PDMUX_R2_POLICY", e["PDMUX_R2_POLICY"])
                for e in M2G.G0_REFUSED_ENVS]
        self.assertEqual(len(set(keys)), 5)
        base = {"PDMUX_R2_POLICY": "fixed"}  # the d44 arm env the refusal boots override
        for env in M2G.G0_REFUSED_ENVS:
            self.assertTrue(GUARD.tp_pdmux_violations(2, {**base, **env}), env)
        self.assertEqual(GUARD.tp_pdmux_violations(2, base), [])  # control: allowed arm not refused


# ------------------------------------------------------------------------------ G0
class TestG0(unittest.TestCase):
    def test_boot(self):
        self.assertEqual(M2G.score_g0_boot({"healthy": True})["verdict"], "PASS")
        self.assertEqual(M2G.score_g0_boot({"healthy": False, "health_wait_cap_s": 1800})["verdict"], "FAIL")
        # MGP-23: a failure under the runner's old 360 s wait is NOT_TESTED, not FAIL
        self.assertEqual(M2G.score_g0_boot({"healthy": False, "health_wait_cap_s": 360})["verdict"], "NOT_TESTED")
        self.assertEqual(M2G.score_g0_boot(None)["verdict"], "NOT_TESTED")

    def test_refusal_needs_text_and_abnormal_exit(self):
        txt = "RuntimeError: PD-mux with tp_size=2 is only defined ..."
        ok = {"healthy": False, "exit_code": 1, "log_text": txt}
        self.assertEqual(M2G.score_g0_refusal(ok, M2G.GUARD_REFUSAL_TEXT)["verdict"], "PASS")
        self.assertEqual(M2G.score_g0_refusal(dict(ok, healthy=True, exit_code=None), M2G.GUARD_REFUSAL_TEXT)["verdict"], "FAIL")
        self.assertEqual(M2G.score_g0_refusal(dict(ok, log_text="OOM"), M2G.GUARD_REFUSAL_TEXT)["verdict"], "FAIL")
        self.assertEqual(M2G.score_g0_refusal(dict(ok, exit_code=0), M2G.GUARD_REFUSAL_TEXT)["verdict"], "FAIL")

    def test_g0ii_all_five(self):
        b = {"healthy": False, "exit_code": 1, "log_text": "PD-mux with tp_size=2"}
        self.assertEqual(M2G.score_g0_ii([b] * 5)["verdict"], "PASS")
        self.assertEqual(M2G.score_g0_ii([b] * 4)["verdict"], "NOT_TESTED")
        self.assertEqual(M2G.score_g0_ii([b] * 4 + [{"healthy": True, "log_text": ""}])["verdict"], "FAIL")


# ------------------------------------------------------------------------------ G1
def outs(n=64, length=128, diverge=None):
    """diverge: {prompt_index: first mismatch position}."""
    a = {f"P{i:02d}": list(range(length)) for i in range(n)}
    b = {k: list(v) for k, v in a.items()}
    for i, pos in (diverge or {}).items():
        b[f"P{i:02d}"][pos] = -1
    return a, b


def sg1(pair, ctl, abnormal_exit=False, exposure=None):
    """score_g1 with an explicit 'both servers alive' survival input (the scorer's default is now None)."""
    return M2G.score_g1(pair, ctl, abnormal_exit, exposure)


class TestG1(unittest.TestCase):
    def test_metrics_by_hand(self):
        a, b = outs(4, 10, {0: 5, 1: 0})
        m = M2G.g1_metrics(a, b)
        # m = [0.5, 0.0, 1, 1] -> median 0.75, q = 2/4
        self.assertEqual(m["m"]["P00"], 0.5)
        self.assertEqual(m["m"]["P01"], 0.0)
        self.assertEqual(m["m_median"], 0.75)
        self.assertEqual(m["q"], 0.5)
        self.assertEqual(m["zero_length"], [])

    def test_prefix_counts_as_divergence(self):
        m = M2G.g1_metrics({"P": [1, 2, 3, 4]}, {"P": [1, 2]})
        self.assertEqual(m["m"]["P"], 0.5)

    def ctl(self, diverge=None):
        return M2G.g1_metrics(*outs(diverge=diverge))

    def test_identical_passes(self):
        self.assertEqual(sg1(self.ctl(), self.ctl())["verdict"], "PASS")

    def _pair_with_median(self, k_half):
        # 33 of 64 prompts diverge at position p -> m median = p/128
        return M2G.g1_metrics(*outs(diverge={i: k_half for i in range(33)}))

    def test_m_margin_boundary_and_mutant(self):
        ctl = M2G.g1_metrics(*outs(diverge={i: 128 - 32 for i in range(33)}))  # median 0.75
        at = self._pair_with_median(128 - 32 - 6)  # median 90/128 = 0.703 -> >= 0.70 PASS... q equal
        below = self._pair_with_median(128 - 32 - 7)  # 89/128 = 0.6953 < 0.70 FAIL
        self.assertEqual(sg1(at, ctl)["verdict"], "PASS")
        self.assertEqual(sg1(below, ctl)["verdict"], "FAIL")
        with mock.patch.object(M2G, "G1_M_MARGIN", 0.0):  # mutant: margin dropped
            self.assertEqual(sg1(at, ctl)["verdict"], "FAIL")
        with mock.patch.object(M2G, "G1_M_MARGIN", 0.10):  # mutant: margin widened
            self.assertEqual(sg1(below, ctl)["verdict"], "PASS")

    def test_q_margin_boundary_and_mutant(self):
        ctl = M2G.g1_metrics(*outs(diverge={i: 127 for i in range(6)}))  # q = 6/64
        ok = M2G.g1_metrics(*outs(diverge={i: 127 for i in range(12)}))  # q 12/64 = 0.1875 <= 0.09375+0.10
        bad = M2G.g1_metrics(*outs(diverge={i: 127 for i in range(13)}))  # 13/64 = 0.203 > 0.19375
        self.assertEqual(sg1(ok, ctl)["verdict"], "PASS")
        self.assertEqual(sg1(bad, ctl)["verdict"], "FAIL")
        with mock.patch.object(M2G, "G1_Q_MARGIN", 0.0):
            self.assertEqual(sg1(ok, ctl)["verdict"], "FAIL")

    def test_uninformative_control(self):
        ctl = M2G.g1_metrics(*outs(diverge={i: 10 for i in range(40)}))  # median 10/128 < 0.5
        pair = M2G.g1_metrics(*outs(diverge={i: 0 for i in range(64)}))
        self.assertEqual(sg1(pair, ctl)["verdict"], "UNINFORMATIVE")
        with mock.patch.object(M2G, "G1_CTL_UNINFORMATIVE_BELOW", 0.0):  # mutant: rule removed
            self.assertEqual(sg1(pair, ctl)["verdict"], "FAIL")

    def test_hard_conditions(self):
        a, b = outs()
        b["P00"] = []
        self.assertEqual(sg1(M2G.g1_metrics(a, b), self.ctl())["verdict"], "FAIL")
        self.assertEqual(sg1(self.ctl(), self.ctl(), abnormal_exit=True)["verdict"], "FAIL")
        self.assertEqual(sg1(None, self.ctl())["verdict"], "NOT_TESTED")
        self.assertEqual(sg1(self.ctl(), None)["verdict"], "NOT_TESTED")
        few = M2G.g1_metrics(*outs(n=10))
        self.assertEqual(sg1(few, self.ctl())["verdict"], "FAIL")

    def test_exposure_reported(self):
        self.assertEqual(sg1(self.ctl(), self.ctl(), exposure=0.42)["exposure_split_steps"], 0.42)

    def test_survival_is_tristate_and_default_fails_closed(self):
        """F2: the old Job path passed a hardcoded False; now None (no record) is NOT_TESTED."""
        c = self.ctl()
        self.assertEqual(M2G.score_g1(c, c)["verdict"], "NOT_TESTED")  # default = no survival record
        self.assertEqual(M2G.score_g1(c, c, None)["verdict"], "NOT_TESTED")
        self.assertEqual(M2G.score_g1(c, c, False)["verdict"], "PASS")  # control
        self.assertEqual(M2G.score_g1(c, c, True)["verdict"], "FAIL")
        a, b = outs()
        b["P00"] = []
        self.assertEqual(M2G.score_g1(M2G.g1_metrics(a, b), c, None)["verdict"], "FAIL")  # zero-length still FAIL
        div = M2G.g1_metrics(*outs(diverge={i: 0 for i in range(64)}))  # every prompt diverges at 0
        self.assertEqual(M2G.score_g1(div, c, None)["verdict"], "FAIL")  # observed divergence is not relabelled
        r = M2G.score_g1(c, c, None)
        self.assertEqual((r["verdict"], r["verdict_if_alive"]), ("NOT_TESTED", "PASS"))

    def test_abnormal_exit_of_truth_table(self):
        f = M2G.abnormal_exit_of
        self.assertIs(f(True, True), False)
        self.assertIs(f(True, False), True)
        self.assertIs(f(None, False), True)  # a seen death wins over a missing record
        self.assertIsNone(f(True, None))
        self.assertIsNone(f(None, None))


# ------------------------------------------------------------------------------ G2
def tf(n_prompts, n_pos, lp_fn, id_fn=lambda i, j: 7):
    return {f"P{i}": [{"teacher_token": 7, "top1_id": id_fn(i, j), "top1_lp": lp_fn(i, j)} for j in range(n_pos)]
            for i in range(n_prompts)}


class TestG2(unittest.TestCase):
    def test_percentile_linear(self):
        self.assertEqual(M2G.percentile([0, 10], 99), 9.9)
        self.assertEqual(M2G.percentile(list(range(101)), 99), 99.0)

    def test_floor_zero_uses_absolute(self):
        a = tf(10, 10, lambda i, j: -0.1)
        self.assertEqual(M2G.score_g2(a, a, tf(10, 10, lambda i, j: -0.1 - 0.049))["verdict"], "PASS")
        r = M2G.score_g2(a, a, tf(10, 10, lambda i, j: -0.1 - 0.051))
        self.assertEqual((r["verdict"], r["label"]), ("FAIL", "TP_NUMERIC_DIVERGENT"))

    def test_floor_multiple_and_mutant(self):
        a = tf(10, 10, lambda i, j: -1.0)
        b = tf(10, 10, lambda i, j: -1.0 - 0.02)  # floor p99 = 0.02 -> threshold max(0.08, 0.05) = 0.08
        t = tf(10, 10, lambda i, j: -1.0 - 0.07)
        self.assertAlmostEqual(M2G.score_g2(a, b, t)["threshold_nat"], 0.08)
        self.assertEqual(M2G.score_g2(a, b, t)["verdict"], "PASS")
        with mock.patch.object(M2G, "G2_FLOOR_MULT", 1.0):  # mutant: floor not multiplied
            self.assertEqual(M2G.score_g2(a, b, t)["verdict"], "FAIL")

    def test_top1_rate(self):
        a = tf(10, 10, lambda i, j: -0.1)
        ok = tf(10, 10, lambda i, j: -0.1, id_fn=lambda i, j: 8 if (i, j) in ((0, 0), (1, 1)) else 7)  # 2/100
        bad = tf(10, 10, lambda i, j: -0.1, id_fn=lambda i, j: 8 if (i, j) in ((0, 0), (1, 1), (2, 2)) else 7)
        self.assertEqual(M2G.score_g2(a, a, ok)["verdict"], "PASS")
        self.assertEqual(M2G.score_g2(a, a, bad)["verdict"], "FAIL")
        with mock.patch.object(M2G, "G2_TOP1_MISMATCH_MAX", 0.01):
            self.assertEqual(M2G.score_g2(a, a, ok)["verdict"], "FAIL")

    def test_structural_problems(self):
        a = tf(3, 4, lambda i, j: -0.1)
        short = {k: v[:3] for k, v in a.items()}
        self.assertEqual(M2G.score_g2(a, a, short)["verdict"], "FAIL")
        nan = tf(3, 4, lambda i, j: math.nan)
        self.assertEqual(M2G.score_g2(a, a, nan)["verdict"], "FAIL")
        self.assertEqual(M2G.score_g2(a, short, a)["verdict"], "NOT_TESTED")  # floor invalid
        self.assertEqual(M2G.score_g2(None, a, a)["verdict"], "NOT_TESTED")


# ------------------------------------------------------------------------------ G3
def write_g3(d, world, repeats=500, errors=0, completed=None, hang=None, drop=None):
    d.mkdir(parents=True, exist_ok=True)
    for r in range(world):
        if r == drop:
            continue
        (d / f"rank{r}.json").write_text(json.dumps({"rank": r, "world": world, "repeats": repeats,
                                                     "completed": repeats if completed is None else completed,
                                                     "errors": errors if r == 0 else 0, "elapsed_s": 1.0}))
    if hang is not None:
        (d / f"HANG.rank{hang}.json").write_text("{}")


class TestG3(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_combo(self):
        cases = {"ok": ({}, "PASS"), "hang": ({"hang": 1}, "FAIL"), "err": ({"errors": 1}, "FAIL"),
                 "short": ({"completed": 499}, "FAIL"), "drop": ({"drop": 1}, "FAIL"),
                 "wrongR": ({"repeats": 100, "completed": 100}, "FAIL")}
        for name, (kw, want) in cases.items():
            write_g3(self.d / name, 2, **kw)
            self.assertEqual(M2G.score_g3_combo(self.d / name, 2)["verdict"], want, name)
        self.assertEqual(M2G.score_g3_combo(self.d / "absent", 2)["verdict"], "NOT_TESTED")

    def test_cli_layout_all_three_combos(self):
        for tag in ("plain", "g16_92", "g44_64"):
            write_g3(self.d / f"g3_TP4_{tag}", 4)
        self.assertEqual(M2G.cli_score_g3(self.d, 4)["verdict"], "PASS")
        self.assertTrue((self.d / "g3_TP4.json").exists())
        write_g3(self.d / "g3_TP4_g44_64", 4, hang=3)
        self.assertEqual(M2G.cli_score_g3(self.d, 4)["verdict"], "FAIL")
        import shutil
        shutil.rmtree(self.d / "g3_TP4_g16_92")
        # an observed hang is not relabelled NOT_TESTED by a missing sibling combination
        self.assertEqual(M2G.cli_score_g3(self.d, 4)["verdict"], "FAIL")
        shutil.rmtree(self.d / "g3_TP4_g44_64")
        write_g3(self.d / "g3_TP4_g44_64", 4)
        self.assertEqual(M2G.cli_score_g3(self.d, 4)["verdict"], "NOT_TESTED")

    def test_progress_watchdog_fires_only_without_progress(self):
        fired = []
        wd = STRESS.ProgressWatchdog(0.05, lambda: fired.append(1))
        wd.kick()
        import time
        for _ in range(5):
            time.sleep(0.02)
            wd.kick()
        wd.cancel()
        self.assertEqual(fired, [])
        wd.kick()
        time.sleep(0.15)
        self.assertEqual(fired, [1])


# ------------------------------------------------------------------------------ G3b
def req(sent, ttft_ms, itls, error=""):
    return {"sent_s": sent, "ttft_ms": ttft_ms, "token_itl_ms": itls, "error": error,
            "completion_s": sent + (ttft_ms or 0) / 1000 + sum(itls) / 1000}


class TestG3b(unittest.TestCase):
    def test_progress_in_last_60s(self):
        r1 = req(530.0, 5000.0, [1000.0] * 10)  # tokens 535 .. 545 -> inside [540, 600]
        self.assertEqual(M2G.score_g3b([r1], True)["verdict"], "PASS")
        r2 = req(100.0, 1000.0, [100.0] * 10)  # all tokens before 540
        self.assertEqual(M2G.score_g3b([r2], True)["verdict"], "FAIL")
        self.assertEqual(M2G.score_g3b([r1], False)["verdict"], "FAIL")
        r3 = req(530.0, None, [], error="timeout")
        self.assertEqual(M2G.score_g3b([r3], True)["verdict"], "FAIL")  # no completed request
        self.assertEqual(M2G.score_g3b(None, True)["verdict"], "NOT_TESTED")

    def test_g_descriptive_uses_injected_function(self):
        ML = _load("mg_label_t", ENG / "results" / "multigpu" / "mg_label.py")
        reqs = [req(i * 0.5, 100.0 + 10 * i, [10.0]) for i in range(100)]
        r = M2G.score_g3b(reqs, True, g_fn=ML.g_of)
        self.assertIsInstance(r["g_descriptive"]["g"], float)
        self.assertEqual(r["verdict"], "FAIL")  # descriptive g never changes the verdict (no token >= 540)

    def test_offered(self):
        self.assertAlmostEqual(M2G.g3b_offered_rps(8), 1.5 * 8 * 2.99)


# ------------------------------------------------------------------------------ G4
class TestG4(unittest.TestCase):
    LOG = ("[TP0] Capture cuda graph begin. avail\n[TP1] Capture cuda graph begin.\n"
           "[TP0] Capture cuda graph end. Time\n[TP1] Capture cuda graph end. Time\n")

    def test_capture(self):
        self.assertEqual(M2G.score_g4(self.LOG, {})["verdict"], "PASS")
        self.assertEqual(M2G.score_g4(self.LOG.rsplit("\n", 2)[0], {})["verdict"], "FAIL")  # one end missing
        self.assertEqual(M2G.score_g4(self.LOG + "Capture cuda graph failed: x", {})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g4(self.LOG, {"disable_cuda_graph": True})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g4("", {})["verdict"], "FAIL")
        self.assertEqual(M2G.score_g4(None)["verdict"], "NOT_TESTED")

    def test_piecewise_recorded(self):
        r = M2G.score_g4(self.LOG + "Capture piecewise CUDA graph begin\n", {"piecewise_cuda_graph_tokens": [8192]})
        self.assertEqual(r["piecewise"]["begin"], 1)
        self.assertEqual(r["piecewise"]["piecewise_cuda_graph_tokens"], [8192])



def _runtime():
    g5a = _load("g5a_t", ENG / "scripts" / "multigpu" / "g5a_selector_equiv.py")
    return g5a.find_runtime()


class TestG4EngineStrings(unittest.TestCase):
    @unittest.skipIf(_runtime() is None, "runtime tree not found")
    def test_strings(self):
        rt = _runtime() / "sglang" / "srt" / "model_executor"
        mr = (rt / "model_runner.py").read_text()
        cg = (rt / "cuda_graph_runner.py").read_text()
        self.assertIn('f"Capture {graph_backend[self.device]} begin', mr)  # renders "Capture cuda graph begin"
        self.assertIn('f"Capture {graph_backend[self.device]} end', mr)
        self.assertIn('lambda: "cuda graph"', mr)
        self.assertIn(M2G.CAPTURE_FAILED, cg)
        self.assertIn(M2G.PIECEWISE_BEGIN, mr)
        self.assertIn(M2G.PIECEWISE_END, mr)



# ------------------------------------------------------------------------------ G5
def write_snaps(path, rows):
    recs = [{"event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": float(i),
             "decode_running_batch_size": 4, "stream_index": idx, "prefill_sms": p, "decode_sms": d}
            for i, (idx, p, d) in enumerate(rows)]
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


class TestG5b(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_realized_in_table(self):
        base = str(self.d / "tel.jsonl")
        f0, f1 = MIX.rank_files(base, "dp", 2)
        write_snaps(f0, [(1, 74, 34), (2, 54, 54), (3, 0, 108)])
        write_snaps(f1, [(1, 74, 34), (0, 108, 0)])
        self.assertEqual(M2G.score_g5b_arm(2, "agnostic", base)["verdict"], "PASS")
        write_snaps(f1, [(1, 64, 44)])  # a d44 division under the agnostic arm
        self.assertEqual(M2G.score_g5b_arm(2, "agnostic", base)["verdict"], "FAIL")
        write_snaps(f1, [(2, 74, 34)])  # in the table but at the wrong index
        self.assertEqual(M2G.score_g5b_arm(2, "agnostic", base)["verdict"], "FAIL")
        Path(f1).unlink()
        self.assertEqual(M2G.score_g5b_arm(2, "agnostic", base)["verdict"], "FAIL")

    def test_aggregate(self):
        self.assertEqual(M2G.score_g5b({})["verdict"], "NOT_TESTED")
        self.assertEqual(M2G.score_g5b({"agnostic": {"verdict": "PASS"}, "d44": None})["verdict"], "PASS")
        self.assertEqual(M2G.score_g5b({"agnostic": {"verdict": "PASS"}, "d44": {"verdict": "FAIL"}})["verdict"], "FAIL")


def meta(cap=96, mamba=None, max_total=None):
    return {"max_running_requests_requested": cap, "max_mamba_cache_size": mamba if mamba is not None else [cap],
            "max_total_num_tokens": max_total if max_total is not None else [800000]}


class TestG5cMemory(unittest.TestCase):
    def test_rules(self):
        ok = {a: meta() for a in M2G.ARMS}
        self.assertEqual(M2G.score_g5c_memory(ok)["verdict"], "PASS")
        # (i) mamba slots != cap
        self.assertEqual(M2G.score_g5c_memory(dict(ok, d44=meta(mamba=[95])))["verdict"], "FAIL")
        # (ii) KV bound: TP2 cap 96 x 8256 = 792,576 (= B 396,288 x 2)
        self.assertEqual(M2G.score_g5c_memory({a: meta(max_total=[792576]) for a in M2G.ARMS})["verdict"], "PASS")
        self.assertEqual(M2G.score_g5c_memory({a: meta(max_total=[792575]) for a in M2G.ARMS})["verdict"], "FAIL")
        # (iii) RETIRED (rev9 §16-2 item 1, post-hoc): an arm spread > 1 % is no longer a MEMORY_MISMATCH -- it is recorded under both
        # denominators as MEMORY_RULE_III_COUNTERFACTUAL.  (Large totals so rule (ii) cannot fire.)
        big = {a: meta(max_total=[900000]) for a in M2G.ARMS}
        spread = dict(big, agnostic=meta(max_total=[900000 * 0.985]))
        r = M2G.score_g5c_memory(spread)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["MEMORY_RULE_III_COUNTERFACTUAL"], {"denominator_fused_sym": True, "denominator_min_arm": True})
        self.assertAlmostEqual(r["arm_rel_diff"]["fused_sym"], 0.015, places=6)
        self.assertAlmostEqual(r["arm_rel_diff"]["min_arm"], 0.015 / 0.985, places=6)
        tight = dict(big, agnostic=meta(max_total=[900000 * 0.995]))
        r = M2G.score_g5c_memory(tight)
        self.assertEqual((r["verdict"], r["rule_iii_would_have_flagged"]), ("PASS", False))
        # four arms required
        r = M2G.score_g5c_memory({a: meta() for a in M2G.ARMS[:3]})
        self.assertEqual((r["verdict"], r["memory_mismatch"]), ("NOT_TESTED", True))


class FakeNvml:
    def __init__(self, table):
        self.table = table  # gpu -> (used_bytes, n_procs)
        self.inited = 0

    def nvmlInit(self):
        self.inited += 1

    def nvmlShutdown(self):
        self.inited -= 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=self.table[h][0])

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [object()] * self.table[h][1]


class TestContamination(unittest.TestCase):
    def test_nvml_snapshot(self):
        nv = FakeNvml({0: (512 * 2**20, 0), 1: (2 * 2**30, 3)})
        snap = M2G.nvml_snapshot([0, 1], nv)
        self.assertEqual(snap[0], {"memory_used_mib": 512.0, "n_procs": 0})
        self.assertEqual(snap[1], {"memory_used_mib": 2048.0, "n_procs": 3})
        self.assertEqual(nv.inited, 0)

    def test_pre_and_post(self):
        clean = {0: {"memory_used_mib": 500.0, "n_procs": 0}}
        self.assertFalse(M2G.check_contamination(clean)["gpu_contaminated"])
        self.assertTrue(M2G.check_contamination({0: {"memory_used_mib": 1024.0, "n_procs": 0}})["gpu_contaminated"])
        self.assertTrue(M2G.check_contamination({0: {"memory_used_mib": 10.0, "n_procs": 1}})["gpu_contaminated"])
        # NVML hides entries -> only (가)
        self.assertFalse(M2G.check_contamination({0: {"memory_used_mib": 10.0, "n_procs": 1}},
                                                 procs_visible=False)["gpu_contaminated"])
        post = {0: {"memory_used_mib": 9e4, "n_procs": 2}}
        self.assertFalse(M2G.check_contamination(clean, post, {0: 2})["gpu_contaminated"])
        self.assertTrue(M2G.check_contamination(clean, post, {0: 1})["gpu_contaminated"])

    def test_visibility(self):
        self.assertFalse(M2G.procs_visible_from_table({"TP2:fused": {0: 0, 1: 0}}))
        self.assertTrue(M2G.procs_visible_from_table({"TP2:fused": {0: 0, 1: 1}}))


# ------------------------------------------------------------------------- PASS table
def all_pass_results():
    P = {"verdict": "PASS"}
    res = {"g0i": {}, "g1": {}, "g2": {}, "g3": {}, "g3b": {}, "g4": {}, "g5b": {}, "g5c": {}, "contam": {},
           "nvml_post": {}, "g0ii": dict(P), "g0iii": dict(P)}
    for cfg in ("N1",) + M2G.CONFIGS_MULTI:
        mode, n = M2G.cfg_mode_n(cfg)
        for arm in M2G.ARMS:
            k = M2G.cell_key(cfg, arm)
            res["g0i"][k] = dict(P)
            res["g4"][k] = dict(P)
            res["contam"][k] = {"gpu_contaminated": False}
            res["nvml_post"][k] = {g: 1 for g in range(n)}
            if arm in M2G.PDMUX_ARMS and cfg != "N1":
                res["g1"][k] = dict(P)
            if mode == "tp" and arm in M2G.G3B_ARMS:
                res["g3b"][k] = dict(P)
        if mode == "tp":
            res["g3"][cfg] = dict(P)
            res["g2"][cfg] = dict(P)
        if mode == "dp":
            res["g5b"][cfg] = dict(P)
        res["g5c"][cfg] = {"verdict": "PASS", "memory_mismatch": False}
    return res


def excluded(table):
    return sorted(k for k, v in table["cells"].items() if not v["included"])


def cells_of(cfgs, arms=None):
    return sorted(M2G.cell_key(c, a) for c in cfgs for a in (arms or M2G.ARMS))


TP = ("TP2", "TP4", "TP8")


class TestPassTable(unittest.TestCase):
    def table(self, mutate=None):
        res = all_pass_results()
        if mutate:
            mutate(res)
        return M2G.build_pass_table(json.loads(json.dumps(res)))  # JSON round trip, as on disk

    def test_job_scoped_table_covers_only_its_configs_and_g0ii_is_j2s(self):
        res = json.loads(json.dumps(all_pass_results()))
        res["g0ii"], res["g0iii"] = {}, {}                              # J4 never ran the refusal boots
        t = M2G.build_pass_table(res, cfg_list=("N1", "TP4", "DP4"), g0ii_in_job=False, job="J4")
        self.assertEqual(sorted({k.split(":")[0] for k in t["cells"]}), ["DP4", "N1", "TP4"])
        self.assertEqual(excluded(t), [])                               # no TP exclusion from a G0(ii) this Job did not run
        self.assertEqual((t["job"], t["g0ii"], t["g0iii"]), ("J4", "NOT_RUN_IN_JOB", "NOT_RUN_IN_JOB"))
        self.assertEqual(t["findings"], [])
        res2 = json.loads(json.dumps(all_pass_results()))
        res2["g0ii"] = {"verdict": "FAIL"}
        self.assertEqual(excluded(M2G.build_pass_table(res2, cfg_list=("N1", "TP2", "DP2"), job="J2")), cells_of(["TP2"]))   # J2 judges it itself

    def test_control_all_included(self):
        t = self.table()
        self.assertEqual(excluded(t), [])
        self.assertEqual(len(t["cells"]), 7 * 4)
        self.assertTrue(t["nvml_procs_visible"])
        self.assertEqual(t["findings"], [])

    def setv(self, sec, key, v):
        def m(res):
            res[sec][key] = {"verdict": v}
        return m

    def test_fused_boot_fail_excludes_config(self):
        self.assertEqual(excluded(self.table(self.setv("g0i", "TP4:fused", "FAIL"))), cells_of(["TP4"]))

    def test_fused_sym_boot_fail_excludes_arm_and_undefines_gsym(self):
        t = self.table(self.setv("g0i", "DP2:fused_sym", "FAIL"))
        self.assertEqual(excluded(t), ["DP2:fused_sym"])
        self.assertTrue(t["configs"]["DP2"]["g_sym_undefined"])
        self.assertFalse(t["configs"]["DP4"]["g_sym_undefined"])

    def test_pdmux_boot_fail_excludes_arm(self):
        self.assertEqual(excluded(self.table(self.setv("g0i", "TP8:d44", "FAIL"))), ["TP8:d44"])

    def test_g0ii_fail_excludes_tp_axis_only(self):
        t = self.table(lambda r: r.update(g0ii={"verdict": "FAIL"}))
        self.assertEqual(excluded(t), cells_of(TP))

    def test_g0ii_not_tested_excludes_tp_axis(self):
        t = self.table(lambda r: r.pop("g0ii"))
        self.assertEqual(excluded(t), cells_of(TP))
        self.assertTrue(all(t["cells"][k]["not_tested"] for k in cells_of(TP)))

    def test_g0iii_fail_is_finding_only(self):
        t = self.table(lambda r: r.update(g0iii={"verdict": "FAIL"}))
        self.assertEqual(excluded(t), [])
        self.assertEqual(len(t["findings"]), 1)

    def test_g1(self):
        self.assertEqual(excluded(self.table(self.setv("g1", "DP4:agnostic", "FAIL"))), ["DP4:agnostic"])
        self.assertEqual(excluded(self.table(self.setv("g1", "TP2:d44", "FAIL"))), ["TP2:d44"])
        self.assertEqual(excluded(self.table(self.setv("g1", "TP2:d44", "NOT_TESTED"))), ["TP2:d44"])
        t = self.table(self.setv("g1", "TP2:agnostic", "UNINFORMATIVE"))
        self.assertEqual(excluded(t), [])
        self.assertIn("agnostic:G1_UNINFORMATIVE", t["configs"]["TP2"]["flags"])

    def test_g3_and_g3b(self):
        self.assertEqual(excluded(self.table(self.setv("g3", "TP8", "FAIL"))), cells_of(["TP8"]))
        self.assertEqual(excluded(self.table(self.setv("g3b", "TP4:d44", "FAIL"))), cells_of(["TP4"]))
        self.assertEqual(excluded(self.table(lambda r: r["g3b"].pop("TP2:agnostic"))), cells_of(["TP2"]))

    def test_g4(self):
        self.assertEqual(excluded(self.table(self.setv("g4", "N1:fused_sym", "FAIL"))), ["N1:fused_sym"])

    def test_g5b(self):
        self.assertEqual(excluded(self.table(self.setv("g5b", "DP8", "FAIL"))), cells_of(["DP8"]))

    def test_g2_is_flag_only(self):
        t = self.table(self.setv("g2", "TP4", "FAIL"))
        self.assertEqual(excluded(t), [])
        self.assertTrue(t["configs"]["TP4"]["tp_numeric_divergent"])

    def test_memory_is_flag_only(self):
        t = self.table(lambda r: r["g5c"].update(DP2={"verdict": "FAIL", "memory_mismatch": True}))
        self.assertEqual(excluded(t), [])
        self.assertTrue(t["configs"]["DP2"]["memory_mismatch"])
        t = self.table(lambda r: r["g5c"].pop("TP2"))
        self.assertIn("MEMORY_MISMATCH(NOT_TESTED)", t["configs"]["TP2"]["flags"])

    def test_contamination(self):
        t = self.table(lambda r: r["contam"].update({"TP2:agnostic": {"gpu_contaminated": True}}))
        self.assertEqual(excluded(t), ["TP2:agnostic"])
        t = self.table(lambda r: r["contam"].pop("DP2:fused"))
        self.assertEqual(excluded(t), ["DP2:fused"])

    def test_empty_results_exclude_everything(self):
        t = M2G.build_pass_table({})
        self.assertEqual(len(excluded(t)), 28)

    def test_digest_is_of_canonical_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "M2G_PASS_TABLE.json"
            dig = M2G.write_pass_table(self.table(), p)
            self.assertEqual(dig, hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(M2G.write_pass_table(self.table(), p), dig)  # deterministic


# ------------------------------------------------------------------------- Job sequence
class FakeIO:
    """In-memory runner: every boot healthy unless listed in `fail`; writes synthetic telemetry/green."""

    def __init__(self, root, fail=(), clock_step=1.0):
        self.root = Path(root)
        self.fail = set(fail)
        self.t = 0.0
        self.clock_step = clock_step
        self.boots = []
        self.live = {}
        self.n_boot = 0

    def clock(self):
        self.t += self.clock_step
        return self.t

    def sleep(self, s):
        self.t += s

    def prompts(self):
        return [(f"P{i:02d}", "x") for i in range(64)]

    def nvml(self, ids):
        busy = any(h for h in self.live.values())
        return {g: {"memory_used_mib": 50000.0 if busy else 100.0, "n_procs": 1 if busy else 0} for g in ids}

    def boot(self, spec):
        self.n_boot += 1
        self.boots.append(spec)
        env = spec["env"]
        refused = spec["config"].startswith("TP") and bool(GUARD.tp_pdmux_violations(spec["n"], {**(
            {"PDMUX_R2_POLICY": "fixed"} if spec["arm"] == "d44" else {}), **env}))
        if refused:
            return {"healthy": False, "exit_code": 1, "health_wait_cap_s": 1800, "handle": None,
                    "log_text": "RuntimeError: PD-mux with tp_size=2 is only defined"}
        if spec.get("pdmux_config_rel") and spec["config"] != "N1":
            return {"healthy": False, "exit_code": 1, "health_wait_cap_s": 1800, "handle": None,
                    "log_text": "AssertionError: PD-mux auto grid (divide_sm): ..."}
        if spec["tag"] in self.fail:
            return {"healthy": False, "exit_code": 1, "health_wait_cap_s": 1800, "handle": None, "log_text": "boom"}
        h = f"h{self.n_boot}"
        self.live[h] = True
        mode, n = spec["mode"], spec["n"]
        base = self.root / spec["tag"]
        tel, grn = str(base / "tel.jsonl"), str(base / "green.json")
        base.mkdir(parents=True, exist_ok=True)
        if spec["arm"] in M2G.PDMUX_ARMS:
            sm = MIX.REGISTERED_SM_COUNTS[spec["arm"]]
            idx = 1 if spec["arm"] == "agnostic" else 4
            for tf_, gf in zip(MIX.rank_files(tel, mode, n), MIX.rank_files(grn, mode, n)):
                write_snaps(tf_, [(idx, *sm[idx]), (idx, *sm[idx]), (len(sm) - 1, *sm[-1])])
                groups = [{"stream_index": i, "target_prefill_sm": p, "target_decode_sm": d,
                           "prefill": ({"green_ctx_is_null": False, "green_sm": {"smCount": p}} if 0 < i < len(sm) - 1
                                       else {"green_ctx_is_null": True}),
                           "decode": ({"green_ctx_is_null": False, "green_sm": {"smCount": d}} if 0 < i < len(sm) - 1
                                      else {"green_ctx_is_null": True})} for i, (p, d) in enumerate(sm)]
                Path(gf).write_text(json.dumps({"sticky_target_index": None, "groups": groups}))
        cap = 48 * n if mode == "tp" else 48
        reps = n if mode == "dp" else 1
        return {"healthy": True, "health_s": 100.0, "health_wait_cap_s": 1800, "exit_code": None, "handle": h,
                "telemetry": tel, "green": grn, "server_args": {},
                "meta": {"max_running_requests_requested": cap, "max_mamba_cache_size": [cap] * reps,
                         "max_total_num_tokens": [cap * 9000] * reps}}

    def log_text(self, h):
        return "[TP0] Capture cuda graph begin\n[TP0] Capture cuda graph end\n"

    def alive(self, h):
        return self.live.get(h, False)

    def teardown(self, h):
        self.live[h] = False

    def generate(self, h, prompts, max_new, conc):
        return {rid: list(range(max_new)) for rid, _ in prompts}

    def teacher_forced(self, h, prompts, conts):
        return {rid: [{"teacher_token": t, "top1_id": t, "top1_lp": -0.1} for t in conts[rid]] for rid, _ in prompts}

    def serve_stress(self, h, rps, dur, shape):
        return [req(550.0, 1000.0, [100.0] * 5)]

    def comm_stress(self, n, streams, split, repeats, wd):
        d = self.root / f"g3_{n}_{streams}_{split}"
        write_g3(d, n, repeats)
        return str(d), 0


class TestM2gRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_sequence_all_included(self):
        io = FakeIO(self.tmp.name)
        res = M2G.M2gRun(io).run()
        t = M2G.build_pass_table(json.loads(json.dumps(res, default=str)))
        self.assertEqual(excluded(t), [], json.dumps({k: v for k, v in t["cells"].items() if not v["included"]}, indent=1))
        self.assertEqual(res["g0ii"]["verdict"], "PASS")
        self.assertEqual(res["g0iii"]["verdict"], "PASS")
        self.assertEqual(res["descriptive"]["g5a_auto_status"], "OK")
        self.assertFalse(any(io.live.values()))  # every boot torn down
        # boots: 4 N1 + 1 auto + 5 refusal + 1 autogrid + 6 x 4
        self.assertEqual(io.n_boot, 4 + 1 + 5 + 1 + 24)

    def test_failing_boot_propagates(self):
        io = FakeIO(self.tmp.name, fail={"DP4_fused"})
        t = M2G.build_pass_table(json.loads(json.dumps(M2G.M2gRun(io).run(), default=str)))
        self.assertEqual(excluded(t), cells_of(["DP4"]))

    def test_hard_cap_truncates_to_not_tested(self):
        io = FakeIO(self.tmp.name, clock_step=1000.0)  # every clock read costs 1000 s -> cap hit early
        res = M2G.M2gRun(io).run()
        self.assertTrue(res["truncated_by_hard_cap"])
        t = M2G.build_pass_table(json.loads(json.dumps(res, default=str)))
        self.assertGreater(len(excluded(t)), 0)
        self.assertTrue(any(v["not_tested"] for v in t["cells"].values()))

    def test_io_contract_enforced(self):
        with self.assertRaises(TypeError):
            M2G.M2gRun(SimpleNamespace(clock=lambda: 0.0))


# ------------------------------------------------------------------------- CLI adapters
def green_doc(sm):
    return {"sticky_target_index": None, "groups": [
        {"stream_index": i, "target_prefill_sm": p, "target_decode_sm": d,
         "prefill": ({"green_ctx_is_null": False, "green_sm": {"smCount": p}} if 0 < i < len(sm) - 1
                     else {"green_ctx_is_null": True}),
         "decode": ({"green_ctx_is_null": False, "green_sm": {"smCount": d}} if 0 < i < len(sm) - 1
                    else {"green_ctx_is_null": True})} for i, (p, d) in enumerate(sm)]}


class TestCliAdapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.g = Path(self.tmp.name) / "gates"  # Job layout: OUT/gates next to OUT/boots
        self.g.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, cfg, arm, outs_):
        (self.g / f"g1_out_{cfg}_{arm}.json").write_text(json.dumps({"outputs": outs_, "errors": {}}))

    def alive(self, cfg, arm, rc=None, status=200, boots=None):
        bd = (boots or self.g.parent / "boots") / f"m2g_{cfg}_{arm}"
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "alive_after.json").write_text(json.dumps(M2G.make_alive_record(f"m2g_{cfg}_{arm}", rc, status)))

    def test_score_g1_contract(self):
        a, b = outs()
        for cfg in ("N1", "TP2"):
            for arm in M2G.ARMS:
                self.put(cfg, arm, a)
        self.put("TP2", "d44", {k: v[:64] + [-1] * 64 for k, v in a.items()})  # every prompt diverges at 64
        # F2: no survival record anywhere -> NOT_TESTED (the old hardcoded False said PASS/FAIL here)
        self.assertEqual(M2G.cli_score_g1(self.g, "TP2")["verdicts"],
                         {"fused|agnostic": "NOT_TESTED", "fused|d44": "FAIL"})  # d44 FAIL: outputs diverge
        for arm in M2G.ARMS:
            self.alive("TP2", arm)
        doc = M2G.cli_score_g1(self.g, "TP2")
        self.assertEqual(doc["verdicts"], {"fused|agnostic": "PASS", "fused|d44": "FAIL"})
        self.assertTrue((self.g / "g1_TP2.json").exists())
        self.assertEqual(M2G.cli_score_g1(self.g, "DP2")["verdicts"]["fused|agnostic"], "NOT_TESTED")

    def test_score_g1_survival_from_records_and_mutant(self):
        a, _ = outs()
        for cfg in ("N1", "TP4"):
            for arm in M2G.ARMS:
                self.put(cfg, arm, a)
        for arm in M2G.ARMS:
            self.alive("TP4", arm)
        self.assertEqual(M2G.cli_score_g1(self.g, "TP4")["verdicts"]["fused|agnostic"], "PASS")  # control
        self.alive("TP4", "agnostic", rc=-9, status=None)  # server exited after its last step
        self.assertEqual(M2G.cli_score_g1(self.g, "TP4")["verdicts"]["fused|agnostic"], "FAIL")
        self.alive("TP4", "agnostic", rc=None, status=503)  # process up, /health not OK
        self.assertEqual(M2G.cli_score_g1(self.g, "TP4")["verdicts"]["fused|agnostic"], "FAIL")
        (self.g.parent / "boots" / "m2g_TP4_fused" / "alive_after.json").unlink()
        self.assertEqual(M2G.cli_score_g1(self.g, "TP4")["verdicts"]["fused|d44"], "NOT_TESTED")
        # mutant = the pre-F2 code (abnormal exit hardcoded False): the missing record would read PASS
        with mock.patch.object(M2G, "abnormal_exit_of", lambda *x: False):
            self.assertEqual(M2G.cli_score_g1(self.g, "TP4")["verdicts"]["fused|d44"], "PASS")
        # explicit --boots directory is honoured
        other = Path(self.tmp.name) / "elsewhere"
        for arm in M2G.ARMS:
            self.alive("TP4", arm, boots=other)
        self.assertEqual(M2G.cli_score_g1(self.g, "TP4", boots=other)["verdicts"]["fused|d44"], "PASS")

    def test_score_g1_exposure_given_and_from_g1_window(self):
        a, _ = outs()
        for cfg in ("N1", "TP2"):
            for arm in M2G.ARMS:
                self.put(cfg, arm, a)
        boots = self.g.parent / "boots"
        for arm in M2G.ARMS:
            self.alive("TP2", arm)
        doc = M2G.cli_score_g1(self.g, "TP2", exposure={"agnostic": 0.25})
        self.assertEqual(doc["detail"]["fused|agnostic"]["exposure_split_steps"], 0.25)
        self.assertIsNone(doc["detail"]["fused|d44"]["exposure_split_steps"])
        # telemetry of the TP2 agnostic boot (rank 0 file), steps attributed per snapshot (by hand):
        #   t  idx di   window [0.5, 3.5] -> t1 (split, +10), t2 (idx 0, +5), t3 (split, +15) = 25/30
        #   0   1   0   whole file          -> +10 split, +10, +5, +15, t4 idx3 (+10)  = 35/50 = 0.7
        #   1   1  10
        #   2   0  20
        #   3   1  25
        #   4   3  40
        #   5   1  50
        sm = MIX.REGISTERED_SM_COUNTS["agnostic"]
        bd = boots / "m2g_TP2_agnostic"
        tel = MIX.rank_files(str(bd / "tel.jsonl"), "tp", 2)[0]
        grn = MIX.rank_files(str(bd / "green.json"), "tp", 2)[0]
        rows = [(0, 1, 0), (1, 1, 10), (2, 0, 20), (3, 1, 25), (4, 3, 40), (5, 1, 50)]
        Path(tel).write_text("".join(json.dumps({
            "event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": float(t),
            "decode_running_batch_size": 4, "stream_index": i, "prefill_sms": sm[i][0], "decode_sms": sm[i][1],
            "decode_iterations": di}) + "\n" for t, i, di in rows))
        Path(grn).write_text(json.dumps(green_doc(sm)))
        # no G1 window recorded -> NO_G1_WINDOW, value None (never the whole-boot value)
        d0 = M2G.cli_score_g1(self.g, "TP2", exposure_from_boots=True)["detail"]["fused|agnostic"]
        self.assertEqual(d0["exposure_detail"]["status"], "NO_G1_WINDOW")
        self.assertIsNone(d0["exposure_split_steps"])
        out_doc = json.loads((self.g / "g1_out_TP2_agnostic.json").read_text())
        out_doc["g1_window_perf_counter_s"] = [0.5, 3.5]
        (self.g / "g1_out_TP2_agnostic.json").write_text(json.dumps(out_doc))
        d1 = M2G.cli_score_g1(self.g, "TP2", exposure_from_boots=True)["detail"]["fused|agnostic"]
        self.assertEqual(d1["exposure_detail"]["status"], "OK")
        self.assertAlmostEqual(d1["exposure_split_steps"], 25 / 30)
        self.assertEqual(d1["verdict"], "PASS")  # descriptive only
        # mutant: window dropped -> whole-boot value 0.7, killed by the 25/30 expectation
        real = M2G._mix

        def no_window():
            m = real()
            orig = m.cell_e_time
            m.cell_e_time = lambda arm, mode, n, t, g, window=None, sm_counts=None: orig(arm, mode, n, t, g, None)
            return m
        with mock.patch.object(M2G, "_mix", no_window):
            dm = M2G.cli_score_g1(self.g, "TP2", exposure_from_boots=True)["detail"]["fused|agnostic"]
        self.assertAlmostEqual(dm["exposure_split_steps"], 0.7)
        self.assertNotAlmostEqual(dm["exposure_split_steps"], 25 / 30)
        # window outside the telemetry span -> MIX_FAILED, None
        out_doc["g1_window_perf_counter_s"] = [100.0, 200.0]
        (self.g / "g1_out_TP2_agnostic.json").write_text(json.dumps(out_doc))
        d2 = M2G.cli_score_g1(self.g, "TP2", exposure_from_boots=True)["detail"]["fused|agnostic"]
        self.assertEqual(d2["exposure_detail"]["status"], "MIX_FAILED")
        self.assertIsNone(d2["exposure_split_steps"])

    def test_cli_score_g1_flags(self):
        a, _ = outs()
        for cfg in ("N1", "TP2"):
            for arm in M2G.ARMS:
                self.put(cfg, arm, a)
        for arm in M2G.ARMS:
            self.alive("TP2", arm)
        ej = Path(self.tmp.name) / "exp.json"
        ej.write_text(json.dumps({"agnostic": 0.5, "d44": 0.6}))
        self.assertEqual(M2G.main(["score-g1", "--out", str(self.g), "--config", "TP2", "--exposure-json", str(ej)]), 0)
        doc = json.loads((self.g / "g1_TP2.json").read_text())
        self.assertEqual(doc["detail"]["fused|d44"]["exposure_split_steps"], 0.6)
        self.assertEqual(doc["verdicts"]["fused|d44"], "PASS")

    def test_score_g2_contract_and_alignment_guard(self):
        a = tf(64, 4, lambda i, j: -0.1)
        for name in ("g2_tf_N1_r0", "g2_tf_N1_r1", "g2_tf_TP4_r0"):
            (self.g / f"{name}.json").write_text(json.dumps(a))
        self.assertEqual(M2G.cli_score_g2(self.g, "TP4")["verdict"], "PASS")
        mis = tf(64, 4, lambda i, j: -0.1, id_fn=lambda i, j: 9)  # TP=1 top-1 never equals its own teacher
        (self.g / "g2_tf_N1_r0.json").write_text(json.dumps(mis))
        self.assertEqual(M2G.cli_score_g2(self.g, "TP4")["verdict"], "NOT_TESTED")

    def test_registered_prompts(self):
        p = M2G.registered_prompts()
        self.assertEqual(len(p), 64)
        self.assertEqual(len({r for r, _ in p}), 64)
        self.assertEqual(p, M2G.registered_prompts())  # seed fixed -> identical

    def test_http_clients_with_fake_post(self):
        def post(endpoint, body):
            if "text" in body and body.get("return_logprob"):
                return {"meta_info": {"input_token_logprobs": [[None, 11, None], [-0.5, 12, None]]}}
            if "text" in body:
                return {"output_ids": [1, 2, 3]}
            ids = body["input_ids"]
            start = body["logprob_start_len"]
            self.assertEqual(ids[:2], [11, 12])
            return {"meta_info": {"input_top_logprobs": [[[-0.2, t, None]] for t in ids[start:]]}}
        g = M2G.http_generate("http://x", [("A", "t"), ("B", "u")], post=post)
        self.assertEqual(g["outputs"], {"A": [1, 2, 3], "B": [1, 2, 3]})
        tfr = M2G.http_teacher_forced("http://x", [("A", "t")], {"A": [5, 6]}, post=post)
        self.assertEqual(tfr["A"], [{"teacher_token": 5, "top1_id": 5, "top1_lp": -0.2},
                                    {"teacher_token": 6, "top1_id": 6, "top1_lp": -0.2}])
        self.assertEqual(M2G.teacher_agreement(tfr), 1.0)

# ------------------------------------------------------------------ F1: G3 execution
FAKE_PY = """#!/usr/bin/env python3
# Stand-in for `python`: `-c CODE mods...` = preflight (ok), `-m torch.distributed.run ... --out D` =
# launcher.  FAKE_G3_MODE: pass | hang_rank1 | crash | sleep | spawn_sleep
import json, os, subprocess, sys, time
a = sys.argv[1:]
if a[0] == "-c":
    sys.exit(int(os.environ.get("FAKE_PREFLIGHT_RC", "0")))
assert a[:2] == ["-m", "torch.distributed.run"], a
n = int(a[a.index("--nproc-per-node") + 1]); out = a[a.index("--out") + 1]; R = int(a[a.index("--repeats") + 1])
os.makedirs(out, exist_ok=True)
mode = os.environ.get("FAKE_G3_MODE", "pass")
print("launcher argv", a); print("stderr line", file=sys.stderr)
if mode == "sleep":
    time.sleep(60)
if mode == "spawn_sleep":
    c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open(os.path.join(out, "grandchild.pid"), "w").write(str(c.pid)); time.sleep(60)
if mode == "crash":
    sys.exit(1)
for r in range(n):
    if mode == "hang_rank1" and r == 1:
        json.dump({"rank": r}, open(os.path.join(out, "HANG.rank1.json"), "w")); continue
    json.dump({"rank": r, "world": n, "repeats": R, "completed": R, "errors": 0, "elapsed_s": 0.1,
               "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}, open(os.path.join(out, f"rank{r}.json"), "w"))
sys.exit(3 if mode == "hang_rank1" else 0)
"""


def _alive_pid(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # zombie = dead
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


class TestG3Execution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.py = self.d / "fakepython"
        self.py.write_text(FAKE_PY.replace("#!/usr/bin/env python3", "#!" + sys.executable))
        self.py.chmod(0o755)
        self.env = {"PATH": os.environ.get("PATH", "")}

    def tearDown(self):
        self.tmp.cleanup()

    def env_mode(self, mode, **kw):
        return dict(self.env, FAKE_G3_MODE=mode, **kw)

    def test_launcher_argv_uses_python_m_and_parses_in_the_stress_script(self):
        argv = M2G.g3_launcher_argv(4, "green", (16, 92), self.d / "o", python="PY")
        self.assertEqual(argv[:6], ["PY", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "4"])
        self.assertEqual(argv[6], str(M2G.G3_STRESS_SCRIPT))
        tail = argv[7:]
        self.assertEqual(tail[tail.index("--split") + 1:tail.index("--split") + 3], ["16", "92"])
        self.assertEqual(tail[tail.index("--repeats") + 1], "500")
        # the stress script's own argparse accepts exactly these flags (parse succeeds, then `import torch` fails)
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(ImportError):
                STRESS.main(tail)
            with self.assertRaises(SystemExit):  # control: an unregistered split is refused by the parser
                STRESS.main(["--streams", "green", "--split", "16", "90", "--out", "x"])
        plain = M2G.g3_launcher_argv(2, "plain", None, self.d / "o", python="PY")
        self.assertNotIn("--split", plain)

    def test_preflight(self):
        self.assertTrue(M2G.g3_preflight(sys.executable, modules=("json",))["ok"])
        bad = M2G.g3_preflight(sys.executable, modules=("no_such_module_m2g_xyz",))
        self.assertFalse(bad["ok"])
        self.assertIn("no_such_module_m2g_xyz", bad["stderr_tail"])
        nf = M2G.g3_preflight(str(self.d / "nonexistent_python"))
        self.assertEqual((nf["ok"], nf["status"]), (False, "LAUNCHER_NOT_FOUND"))

    def run_combo(self, mode, timeout=30.0, **kw):
        od = self.d / f"combo_{mode}"
        return M2G.run_g3_combo(2, "plain", None, od, self.env_mode(mode), timeout, str(self.py), **kw), od

    def test_pass_and_files_saved(self):
        r, od = self.run_combo("pass")
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["launcher_status"], "RAN")
        self.assertIn("launcher argv", (od / "launcher.stdout.txt").read_text())
        self.assertIn("stderr line", (od / "launcher.stderr.txt").read_text())
        self.assertEqual(json.loads((od / "launcher.json").read_text())["rc"], 0)

    def test_hang_is_fail(self):
        r, _ = self.run_combo("hang_rank1")
        self.assertEqual(r["verdict"], "FAIL")
        self.assertEqual(r["hangs"], ["HANG.rank1.json"])

    def test_crash_without_any_rank_file_is_harness_error(self):
        r, _ = self.run_combo("crash")
        self.assertEqual((r["verdict"], r["not_tested_reason"]), ("NOT_TESTED", "HARNESS_ERROR"))

    def test_launcher_not_found_is_harness_error(self):
        od = self.d / "nf"
        r = M2G.run_g3_combo(2, "plain", None, od, self.env, 5.0, str(self.d / "no_python_here"))
        self.assertEqual((r["verdict"], r["not_tested_reason"], r["launcher_status"]),
                         ("NOT_TESTED", "HARNESS_ERROR", "LAUNCHER_NOT_FOUND"))
        self.assertTrue((od / "launcher.json").exists())

    def test_timeout_kills_the_process_group(self):
        r, od = self.run_combo("spawn_sleep", timeout=1.5)
        self.assertEqual((r["verdict"], r["not_tested_reason"], r["launcher_status"]),
                         ("NOT_TESTED", "HARNESS_ERROR", "TIMEOUT"))
        gpid = int((od / "grandchild.pid").read_text())
        import time
        for _ in range(50):
            if not _alive_pid(gpid):
                break
            time.sleep(0.1)
        self.assertFalse(_alive_pid(gpid), "grandchild (a rank) survived the launcher timeout")

    def test_preflight_failure_does_not_launch(self):
        r, od = self.run_combo("pass", preflight={"ok": False, "stderr_tail": "No module named torch"})
        self.assertEqual((r["verdict"], r["not_tested_reason"], r["launcher_status"]),
                         ("NOT_TESTED", "HARNESS_ERROR", "PREFLIGHT_FAILED"))
        self.assertFalse((od / "rank0.json").exists())

    def test_partial_rank_evidence_stays_fail_and_unreadable_only_is_harness_error(self):
        write_g3(self.d / "partial", 2, drop=1)
        self.assertEqual(M2G.score_g3_combo(self.d / "partial", 2)["verdict"], "FAIL")
        (self.d / "garbled").mkdir()
        (self.d / "garbled" / "rank0.json").write_text("{trunc")
        r = M2G.score_g3_combo(self.d / "garbled", 2)
        self.assertEqual((r["verdict"], r.get("not_tested_reason")), ("NOT_TESTED", "HARNESS_ERROR"))
        (self.d / "empty").mkdir()
        self.assertEqual(M2G.score_g3_combo(self.d / "empty", 2).get("not_tested_reason"), "HARNESS_ERROR")

    def test_mutant_old_rule_empty_dir_is_fail_killed(self):
        """lesson 53: re-introducing 'no rank file = FAIL' must flip the empty-directory case."""
        (self.d / "empty").mkdir()
        src = (ENG / "scripts" / "multigpu" / "m2g_correctness_gate.py").read_text()
        old = "    if not hangs and not ranks:\n"
        self.assertEqual(src.count(old), 1)
        mut = _load_src("m2g_mut_g3", src.replace(old, "    if False:\n"))
        self.assertEqual(mut.score_g3_combo(self.d / "empty", 2)["verdict"], "FAIL")  # mutant
        self.assertEqual(M2G.score_g3_combo(self.d / "empty", 2)["verdict"], "NOT_TESTED")  # control

    def test_run_g3_end_to_end_and_cli(self):
        gates = self.d / "gates"
        with mock.patch.dict(os.environ, self.env_mode("pass")):
            r = M2G.run_g3(4, gates, python=str(self.py), timeout_s=30.0, base_env=self.env_mode("pass"))
        self.assertEqual(r["verdict"], "PASS")
        self.assertTrue(r["preflight"]["ok"])
        for tag in ("plain", "g16_92", "g44_64"):
            rk = json.loads((gates / f"g3_TP4_{tag}" / "rank3.json").read_text())
            self.assertEqual(rk["visible"], M2G.C.GPU_SUBSETS["TP4"])  # registered GPU subset
        lj = json.loads((gates / "g3_TP4_g44_64" / "launcher.json").read_text())
        self.assertEqual(lj["argv"][lj["argv"].index("--split") + 1:][:2], ["44", "64"])
        self.assertEqual(json.loads((gates / "g3_TP4.json").read_text())["verdict"], "PASS")
        # preflight failure -> all three NOT_TESTED(HARNESS_ERROR), table label carries the reason
        r2 = M2G.run_g3(2, gates, python=str(self.py), timeout_s=30.0,
                        base_env=self.env_mode("pass", FAKE_PREFLIGHT_RC="1"))
        self.assertEqual((r2["verdict"], r2["not_tested_reason"]), ("NOT_TESTED", "HARNESS_ERROR"))
        # deadline: nothing started -> NOT_TESTED without HARNESS_ERROR, listed as not attempted
        r3 = M2G.run_g3(8, gates, python=str(self.py), timeout_s=30.0, base_env=self.env_mode("pass"),
                        deadline_epoch=0.0)
        self.assertEqual((r3["verdict"], r3.get("not_tested_reason"), len(r3["not_attempted"])), ("NOT_TESTED", None, 3))
        self.assertFalse((gates / "g3_TP8_plain").exists())
        import contextlib
        import io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M2G.main(["run-g3", "--out", str(self.d / "g2"), "--tp", "2", "--python", str(self.py)])
        self.assertEqual((rc, json.loads(buf.getvalue())["verdict"]), (0, "PASS"))

    def test_pass_table_labels_harness_error(self):
        res = all_pass_results()
        res["g3"]["TP4"] = {"verdict": "NOT_TESTED", "not_tested_reason": "HARNESS_ERROR"}
        t = M2G.build_pass_table(json.loads(json.dumps(res)))
        self.assertEqual(excluded(t), cells_of(["TP4"]))  # excluded like FAIL (sec 3-1 last row)
        cell = t["cells"]["TP4:agnostic"]
        self.assertIn("G3 comm stress NOT_TESTED(HARNESS_ERROR)", cell["reasons"])
        self.assertTrue(cell["not_tested"])
        res["g3"]["TP4"] = {"verdict": "FAIL"}
        cell = M2G.build_pass_table(json.loads(json.dumps(res)))["cells"]["TP4:agnostic"]
        self.assertFalse(cell["not_tested"])

    def test_score_g3_fail_dominates_and_reason_propagates(self):
        P, F = {"verdict": "PASS"}, {"verdict": "FAIL"}
        H = {"verdict": "NOT_TESTED", "not_tested_reason": "HARNESS_ERROR"}
        self.assertEqual(M2G.score_g3({"plain": P, "green_16_92": F, "green_44_64": H})["verdict"], "FAIL")
        r = M2G.score_g3({"plain": P, "green_16_92": P, "green_44_64": H})
        self.assertEqual((r["verdict"], r["not_tested_reason"]), ("NOT_TESTED", "HARNESS_ERROR"))
        r = M2G.score_g3({"plain": P, "green_16_92": P})
        self.assertEqual((r["verdict"], r.get("not_tested_reason")), ("NOT_TESTED", None))


def _load_src(name, src):
    mod = type(sys)(name)
    mod.__file__ = str(ENG / "scripts" / "multigpu" / "m2g_correctness_gate.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------ F2: survival record
class FakeProc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


class TestAliveRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_read(self):
        calls = []

        def health(ep, t):
            calls.append(ep)
            return 200, None
        rec = M2G.record_alive_after(self.d / "b", "m2g_TP2_d44", FakeProc(None), "http://127.0.0.1:1",
                                     {"index": 3, "type": "cell", "status": "OK"}, 4, health=health)
        self.assertTrue(rec["alive"])
        self.assertEqual(calls, ["http://127.0.0.1:1"])
        on_disk = json.loads((self.d / "b" / "alive_after.json").read_text())
        self.assertEqual(on_disk["schema"], "m2g_alive_after/v1")
        self.assertEqual(set(on_disk), {"schema", "boot_id", "t_epoch", "after_step", "steps_total",
                                        "poll_returncode", "health_http_status", "health_error", "alive"})
        self.assertIs(M2G.alive_after(self.d / "b"), True)

    def test_dead_process_skips_health(self):
        rec = M2G.record_alive_after(self.d / "b", "x", FakeProc(-11), "http://e",
                                     health=lambda *a: self.fail("health must not be called"))
        self.assertFalse(rec["alive"])
        self.assertIs(M2G.alive_after(self.d / "b"), False)

    def test_health_not_ok(self):
        M2G.record_alive_after(self.d / "b", "x", FakeProc(None), "http://e", health=lambda *a: (None, "refused"))
        self.assertIs(M2G.alive_after(self.d / "b"), False)

    def test_reader_rederives_and_fails_closed(self):
        self.assertIsNone(M2G.alive_after(self.d / "none"))
        (self.d / "b").mkdir()
        f = self.d / "b" / "alive_after.json"
        rec = M2G.make_alive_record("x", 1, None)
        rec["alive"] = True  # forged derived field: ignored
        f.write_text(json.dumps(rec))
        self.assertIs(M2G.alive_after(self.d / "b"), False)
        f.write_text(json.dumps({"schema": "other", "poll_returncode": None, "health_http_status": 200}))
        self.assertIsNone(M2G.alive_after(self.d / "b"))
        f.write_text("{garbled")
        self.assertIsNone(M2G.alive_after(self.d / "b"))
        f.write_text(json.dumps({"schema": "m2g_alive_after/v1", "alive": True}))  # raw fields missing
        self.assertIsNone(M2G.alive_after(self.d / "b"))

    def test_real_health_probe_against_a_closed_port(self):
        import socket
        s_ = socket.socket()
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
        s_.close()
        st, err = M2G._http_health(f"http://127.0.0.1:{port}", 2.0)
        self.assertIsNone(st)
        self.assertIsNotNone(err)


class TestG3bArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.o = Path(self.tmp.name)
        cd = self.o / "cells" / "m2g_TP2_agnostic__A_G3B_s1"
        cd.mkdir(parents=True)
        (cd / "requests.jsonl").write_text(json.dumps(req(550.0, 1000.0, [100.0] * 5)) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def alive(self, rc, st):
        bd = self.o / "boots" / "m2g_TP2_agnostic"
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "alive_after.json").write_text(json.dumps(M2G.make_alive_record("m2g_TP2_agnostic", rc, st)))

    def test_survival_after_last_step(self):
        r = M2G.score_g3b_artifacts(self.o, "TP2", "agnostic")
        self.assertEqual(r["verdict"], "NOT_TESTED")  # no record
        self.alive(None, 200)
        self.assertEqual(M2G.score_g3b_artifacts(self.o, "TP2", "agnostic")["verdict"], "PASS")
        self.alive(-9, None)
        self.assertEqual(M2G.score_g3b_artifacts(self.o, "TP2", "agnostic")["verdict"], "FAIL")
        self.assertEqual(M2G.score_g3b_artifacts(self.o, "TP2", "d44")["verdict"], "NOT_TESTED")  # no cell
        g = M2G.score_g3b_artifacts(self.o, "TP2", "agnostic", g_fn=lambda rs: (0.5, 10.0, [1.0]))
        self.assertEqual(g["g_descriptive"]["g"], 0.5)


if __name__ == "__main__":
    unittest.main()
