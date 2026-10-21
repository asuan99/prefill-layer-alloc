"""CPU tests for scripts/vessl/engine_gate_checks.py (split GPU gate scorers; no GPU data exists yet).

Each scorer is exercised on constructed records with a known answer, and every mutant the gate is
meant to catch (host-time I-4, bs_padded := bs_real) is shown to FAIL -- or, when the data cannot tell
the mutant apart, the scorer says UNRESOLVED instead of PASS (lesson 53).
"""

import importlib.util
import math
import random
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine_gate_checks_t", ENG / "scripts/vessl/engine_gate_checks.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)

CAPTURE = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]


def pdmux_data(n=300, gpu=20.0, host=2.0, tail_ms=0.5, gap_s=0.004):
    dec, ph, t = [], [], 10.0
    for i in range(n):
        dec.append({"event": "decode_forward", "mode": "decode", "seq": i + 1, "t_host_start": t,
                    "decode_fwd_gpu_ms": gpu, "decode_fwd_host_ms": host})
        ph.append({"event": "decode_iteration", "t_launch": t + host / 1000 + 0.0005,
                   "t_sync": t + (gpu + tail_ms) / 1000})
        t += (gpu + tail_ms) / 1000 + gap_s
    return dec, ph


def fused_data(n=300, gpu=20.0, host=2.0, period_ms=22.0):
    return [{"event": "decode_forward", "mode": "decode", "seq": i + 1, "t_host_start": 5.0 + i * period_ms / 1000,
             "decode_fwd_gpu_ms": gpu, "decode_fwd_host_ms": host} for i in range(n)]


class TestI4(unittest.TestCase):
    def test_pdmux_pass_and_mutant_fails(self):
        dec, ph = pdmux_data()
        r = gc.check_i4_pdmux(dec, ph)
        self.assertEqual(r["verdict"], gc.PASS, r)
        self.assertAlmostEqual(r["ratio"], 20 / 20.5, places=6)
        self.assertEqual(r["host_mutant"]["verdict"], gc.FAIL)
        # the mutant's records themselves fail the same scorer
        m = gc.check_i4_pdmux(gc.host_mutant(dec), ph)
        self.assertEqual(m["verdict"], gc.FAIL)

    def test_pdmux_non_discriminating_is_unresolved(self):
        dec, ph = pdmux_data(gpu=2.0, host=1.98, tail_ms=0.05)
        r = gc.check_i4_pdmux(dec, ph)
        self.assertEqual(r["verdict"], gc.UNRESOLVED, r)

    def test_pdmux_containment_violation_fails(self):
        dec, ph = pdmux_data()
        dec[10]["decode_fwd_gpu_ms"] = 30.0
        self.assertEqual(gc.check_i4_pdmux(dec, ph)["verdict"], gc.FAIL)

    def test_pdmux_low_ratio_fails(self):
        dec, ph = pdmux_data(gpu=15.0, tail_ms=6.0)
        self.assertEqual(gc.check_i4_pdmux(dec, ph)["verdict"], gc.FAIL)

    def test_too_few_records_unresolved(self):
        dec, ph = pdmux_data(n=50)
        self.assertEqual(gc.check_i4_pdmux(dec, ph)["verdict"], gc.UNRESOLVED)

    def test_join_ignores_mixed_and_unresolved(self):
        dec, ph = pdmux_data()
        dec.append({"event": "decode_forward", "mode": "mixed", "t_host_start": 0.0,
                    "decode_fwd_gpu_ms": 999.0, "decode_fwd_host_ms": 1.0})
        dec[5]["decode_fwd_gpu_ms"] = None
        pairs = gc.i4_pdmux_pairs(dec, ph)
        self.assertEqual(len(pairs), 299)
        self.assertTrue(all(g == 20.0 for g, _ in pairs))

    def test_fused_pass_and_mutant_fails(self):
        dec = fused_data()
        r = gc.check_i4_fused(dec, (-1e18, 1e18))
        self.assertEqual(r["verdict"], gc.PASS, r)
        self.assertEqual(gc.check_i4_fused(gc.host_mutant(dec), (-1e18, 1e18))["verdict"], gc.FAIL)

    def test_fused_overlap_violation(self):
        dec = fused_data(period_ms=15.0)  # gpu 20 ms cannot fit a 15 ms bracket
        self.assertEqual(gc.check_i4_fused(dec, (-1e18, 1e18))["verdict"], gc.FAIL)


class TestI6(unittest.TestCase):
    def recs(self, bss, graph="hit", pad_fn=None):
        out = []
        for i, b in enumerate(bss):
            padded = (pad_fn or (lambda x: gc.expected_padded_bs(CAPTURE, x)))(b) if graph == "hit" else b
            out.append({"event": "decode_forward", "mode": "decode", "seq": i, "bs_real": b,
                        "bs_padded": padded, "graph": graph,
                        "eager_reason": None if graph == "hit" else "graph_runner_absent"})
        return out

    def test_pass_with_padding(self):
        r = gc.check_i6(self.recs([5, 8, 9, 33, 48]), {"capture_bs": CAPTURE})
        self.assertEqual(r["verdict"], gc.PASS, r)
        self.assertEqual(r["n_padded_hits"], 3)

    def test_pad_real_mutant_fails(self):
        recs = gc.pad_real_mutant(self.recs([5, 8, 9]))
        self.assertEqual(gc.check_i6(recs, {"capture_bs": CAPTURE})["verdict"], gc.FAIL)

    def test_only_exact_sizes_unresolved(self):
        r = gc.check_i6(self.recs([8, 16, 48]), {"capture_bs": CAPTURE})
        self.assertEqual(r["verdict"], gc.UNRESOLVED)
        # and the mutant would indeed have passed there
        self.assertEqual(gc.check_i6(gc.pad_real_mutant(self.recs([8, 16, 48])), {"capture_bs": CAPTURE})["verdict"],
                         gc.UNRESOLVED)

    def test_eager_control(self):
        boot = {"capture_bs": []}
        ok = gc.check_i6(self.recs([5, 9], graph="eager"), boot, expect_all_eager_reason="graph_runner_absent")
        self.assertEqual(ok["verdict"], gc.PASS)
        bad = self.recs([5], graph="eager") + self.recs([5])
        self.assertEqual(gc.check_i6(bad, {"capture_bs": CAPTURE}, expect_all_eager_reason="graph_runner_absent")["verdict"],
                         gc.FAIL)
        padded_eager = self.recs([5], graph="eager")
        padded_eager[0]["bs_padded"] = 8
        self.assertEqual(gc.check_i6(padded_eager, boot, expect_all_eager_reason="graph_runner_absent")["verdict"], gc.FAIL)

    def test_expected_padded_matches_probe_rule(self):
        spec2 = importlib.util.spec_from_file_location("dp_for_gate", ENG / "src/multiplex/decode_probe.py")
        dp = importlib.util.module_from_spec(spec2)
        import sys
        sys.modules[spec2.name] = dp
        spec2.loader.exec_module(dp)
        for raw in range(0, 60):
            for dpad in (False, True):
                self.assertEqual(gc.expected_padded_bs(CAPTURE, raw, dpad), dp.expected_padded_bs(CAPTURE, raw, dpad))


class TestEmission(unittest.TestCase):
    def test_vanilla_path_must_emit(self):
        rows = [{"event": "decode_forward", "mode": "decode", "decode_fwd_gpu_ms": 1.0} for _ in range(100)]
        rows.append({"event": "decode_probe_window", "n_decode": 100, "n_pending": 0})
        adm = [{"event": "admission_window", "n_decode_iter": 100}]
        self.assertEqual(gc.check_emission(rows, adm)["verdict"], gc.PASS)
        self.assertEqual(gc.check_emission(rows, [])["verdict"], gc.FAIL)
        none = [{"event": "decode_probe_window", "n_decode": 0, "n_pending": 0}]
        r = gc.check_emission(none, adm)
        self.assertEqual(r["verdict"], gc.FAIL)
        self.assertIn("NOT_COLLECTED", r["why"])
        half = rows[:50] + [{"event": "decode_forward", "mode": "decode", "decode_fwd_gpu_ms": None}] * 50 + rows[100:]
        self.assertEqual(gc.check_emission(half, adm)["verdict"], gc.FAIL)


class TestTost(unittest.TestCase):
    def test_t_table(self):
        self.assertEqual(gc._T95[7 - 1], 1.895)

    def test_equivalent(self):
        rnd = random.Random(1)
        pairs = [(100 * math.exp(rnd.gauss(0, 0.003)), 100.0) for _ in range(8)]
        r = gc.tost_log_ratio(pairs)
        self.assertEqual(r["verdict"], gc.PASS, r)

    def test_shift_is_observer_effect(self):
        rnd = random.Random(2)
        pairs = [(100 * math.exp(-0.03 + rnd.gauss(0, 0.003)), 100.0) for _ in range(8)]
        self.assertEqual(gc.tost_log_ratio(pairs)["verdict"], "OBSERVER_EFFECT")

    def test_wide_ci_is_not_pass(self):
        rnd = random.Random(3)
        pairs = [(100 * math.exp(rnd.gauss(0, 0.03)), 100.0) for _ in range(8)]
        self.assertEqual(gc.tost_log_ratio(pairs)["verdict"], "UNRESOLVED_POWER")  # CI contains 0, exceeds delta

    def test_too_few_pairs(self):
        self.assertEqual(gc.tost_log_ratio([(1.0, 1.0)] * 7)["verdict"], gc.UNRESOLVED)


class TestAuditR1(unittest.TestCase):
    def test_fused_window_excludes_phase_gap(self):
        # s-pass bs=1 stretch at 22 ms period, then a 6 s idle gap, then the isolated phase: unwindowed ratio is wrecked
        a = fused_data(n=300)
        gap = [dict(r, t_host_start=r["t_host_start"] + 400.0) for r in fused_data(n=300)]
        dec = a + gap
        self.assertEqual(gc.check_i4_fused(dec, (-1e18, 1e18))["verdict"], gc.FAIL)   # the audit's inversion
        r = gc.check_i4_fused(dec, (399.0, 1e18))
        self.assertEqual(r["verdict"], gc.PASS, r)
        self.assertEqual(gc.check_i4_fused(gc.host_mutant(dec), (399.0, 1e18))["verdict"], gc.FAIL)
        self.assertEqual(gc.check_i4_fused(dec, None)["verdict"], gc.UNRESOLVED)
        self.assertEqual(gc.check_i4_fused(dec, (1e9, 2e9))["verdict"], gc.UNRESOLVED)

    def test_tost_three_way_and_extension(self):
        rnd = random.Random(5)
        pairs = [(100 * math.exp(rnd.gauss(0, 0.03)), 100.0) for _ in range(10)]
        self.assertEqual(gc.tost_log_ratio(pairs)["verdict"], "UNRESOLVED_POWER")
        self.assertEqual(gc.tost_log_ratio(pairs * 2)["verdict"], gc.UNRESOLVED)           # 20 > registered 10
        self.assertIn(gc.tost_log_ratio(pairs + pairs[:6], look="extended")["verdict"], gc.REG["tost_verdicts"])
        self.assertEqual(gc.tost_log_ratio(pairs + pairs[:7], look="extended")["verdict"], gc.UNRESOLVED)  # 17 > 16
        self.assertEqual(gc.tost_log_ratio([(100.0, 100.0)] * 8 + [(100.0, 100.0000001)] * 2)["verdict"], gc.PASS)
        shift = [(100 * math.exp(0.02 + rnd.gauss(0, 0.002)), 100.0) for _ in range(10)]
        self.assertEqual(gc.tost_log_ratio(shift)["verdict"], "OBSERVER_EFFECT")

    def test_invalid_pair_asymmetry(self):
        ok = {"a": {"verdict": gc.PASS}, "b": {"verdict": gc.PASS}}
        self.assertEqual(gc.gamma_family_verdict(ok, 0, 0)["verdict"], gc.PASS)
        self.assertEqual(gc.gamma_family_verdict(ok, 1, 0)["verdict"], gc.PASS)
        self.assertEqual(gc.gamma_family_verdict(ok, 2, 0)["verdict"], "OBSERVER_EFFECT")
        self.assertEqual(gc.gamma_family_verdict(ok, 3, 1)["verdict"], "OBSERVER_EFFECT")
        self.assertEqual(gc.gamma_family_verdict(ok, 0, 5)["verdict"], gc.PASS)           # OFF-side excess is not an observer effect
        pw = {"a": {"verdict": gc.PASS}, "b": {"verdict": "UNRESOLVED_POWER"}}
        self.assertEqual(gc.gamma_family_verdict(pw, 0, 0)["verdict"], "UNRESOLVED_POWER")

    def test_instrumentation(self):
        base = {"on": False, "log_decode_enabled": False, "log_admission_enabled": False,
                "files": {"decode_probe": False, "admission": False, "telemetry": False},
                "sglang_file": "/opt/t/sglang/__init__.py", "tree": "/opt/t"}
        self.assertEqual(gc.check_instrumentation(base)["verdict"], gc.PASS)
        for k, v in (("log_decode_enabled", True), ("log_admission_enabled", True)):
            self.assertEqual(gc.check_instrumentation(dict(base, **{k: v}))["verdict"], gc.FAIL)
        self.assertEqual(gc.check_instrumentation(dict(base, files={"decode_probe": True}))["verdict"], gc.FAIL)
        self.assertEqual(gc.check_instrumentation(dict(base, sglang_file="/usr/lib/sglang/__init__.py"))["verdict"], gc.FAIL)
        on = dict(base, on=True, log_decode_enabled=True, log_admission_enabled=True,
                  files={"decode_probe": True, "admission": True})
        self.assertEqual(gc.check_instrumentation(on)["verdict"], gc.PASS)
        self.assertEqual(gc.check_instrumentation(dict(on, log_admission_enabled=False))["verdict"], gc.FAIL)
        self.assertEqual(gc.check_instrumentation(dict(on, files={"decode_probe": True}))["verdict"], gc.FAIL)

    def test_i5_i6(self):
        dec = [{"event": "decode_forward", "mode": "decode", "bs_real": b, "t_host_start": 10.0 + i * 0.01}
               for i, b in enumerate(list(range(40, 0, -1)))]
        dec += [{"event": "decode_forward", "mode": "decode", "bs_real": 48, "t_host_start": 50.0 + i * 0.01} for i in range(5)]
        adm = [{"event": "admission_window", "t_mono_start": 49.0, "t_mono_end": 51.0, "decode_bs_hist": {"48": 5},
                "n_decode_iter_at_or_above_cap": 5, "terminal_by_category": {"cap": 3}}]
        r = gc.check_i5_i6(dec, adm, (9.0, 11.0), (49.5, 51.0))
        self.assertEqual(r["verdict"], gc.PASS, r)
        bad = [dict(a, terminal_by_category={"cap": 0}) for a in adm]
        self.assertEqual(gc.check_i5_i6(dec, bad, (9.0, 11.0), (49.5, 51.0))["verdict"], gc.FAIL)
        self.assertEqual(gc.check_i5_i6(dec[1:], adm, (9.0, 11.0), (49.5, 51.0))["verdict"], gc.FAIL)   # bs 40 never visited
        low = [dict(a, decode_bs_hist={"47": 5}) for a in adm]
        self.assertEqual(gc.check_i5_i6(dec, low, (9.0, 11.0), (49.5, 51.0))["verdict"], gc.FAIL)

    # ---- r2 (2026-10-20): PD-mux stair criterion = contiguous 1..M, M == I-5 stair max, M >= floor
    def _pd_stair(self, bs_set, i5_max, cap_ok=True):
        dec = [{"event": "decode_forward", "mode": "decode", "bs_real": b, "t_host_start": 10.0 + i * 0.01}
               for i, b in enumerate(bs_set)]
        dec += [{"event": "decode_forward", "mode": "decode", "bs_real": 48, "t_host_start": 50.0 + i * 0.01} for i in range(5)]
        adm = [{"event": "admission_window", "t_mono_start": 9.0, "t_mono_end": 12.0, "decode_bs_hist": {str(i5_max): 5}},
               {"event": "admission_window", "t_mono_start": 49.0, "t_mono_end": 51.0, "decode_bs_hist": {"48": 5},
                "n_decode_iter_at_or_above_cap": 5, "terminal_by_category": {"cap": 3}}]
        return gc.check_i5_i6(dec, adm, (9.0, 11.0), (49.5, 51.0), pdmux=True)

    def test_pdmux_stair_criterion(self):
        ok = self._pd_stair(range(39, 0, -1), 39)                       # the G-alpha round-1 B2 shape: 1..39, I-5 max 39
        self.assertEqual(ok["verdict"], gc.PASS, ok)
        self.assertEqual((ok["stair_M"], ok["stair_i5_max"]), (39, 39))
        # the same data under the fused criterion (1..40) fails: the PD-mux branch is what changes the outcome
        dec39 = [{"event": "decode_forward", "mode": "decode", "bs_real": b, "t_host_start": 10.0 + i * 0.01}
                 for i, b in enumerate(range(39, 0, -1))]
        # mutation 1: a hole
        holey = self._pd_stair([b for b in range(39, 0, -1) if b != 17], 39)
        self.assertEqual(holey["verdict"], gc.FAIL)
        self.assertTrue(any("contiguous" in w for w in holey["why"]))
        # mutation 2: M != I-5 max (either direction)
        self.assertEqual(self._pd_stair(range(39, 0, -1), 40)["verdict"], gc.FAIL)
        self.assertEqual(self._pd_stair(range(35, 0, -1), 39)["verdict"], gc.FAIL)
        # mutation 3: M below the floor even though it matches I-5 and is contiguous
        low = self._pd_stair(range(31, 0, -1), 31)
        self.assertEqual(low["verdict"], gc.FAIL)
        self.assertTrue(any("floor" in w for w in low["why"]))
        self.assertEqual(self._pd_stair(range(32, 0, -1), 32)["verdict"], gc.PASS)   # boundary
        # no stair data at all
        self.assertEqual(self._pd_stair([], 0)["verdict"], gc.FAIL)

    def test_fused_stair_criterion_unchanged(self):
        dec = [{"event": "decode_forward", "mode": "decode", "bs_real": b, "t_host_start": 10.0 + i * 0.01}
               for i, b in enumerate(range(39, 0, -1))]
        dec += [{"event": "decode_forward", "mode": "decode", "bs_real": 48, "t_host_start": 50.0 + i * 0.01} for i in range(5)]
        adm = [{"event": "admission_window", "t_mono_start": 49.0, "t_mono_end": 51.0, "decode_bs_hist": {"48": 5},
                "n_decode_iter_at_or_above_cap": 5, "terminal_by_category": {"cap": 3}}]
        self.assertEqual(gc.check_i5_i6(dec, adm, (9.0, 11.0), (49.5, 51.0))["verdict"], gc.FAIL)   # bs 40 not visited (default arm)

    def test_stall_boundary_label(self):
        mk = lambda frac, n=1000: [40.0] * int(n * frac) + [10.0] * (n - int(n * frac))
        self.assertEqual(gc.stall_boundary(mk(0.05))["label"], "STALL_BOUNDARY")
        self.assertEqual(gc.stall_boundary(mk(0.04))["label"], "STALL_BOUNDARY")
        self.assertEqual(gc.stall_boundary(mk(0.06))["label"], "STALL_BOUNDARY")
        self.assertIsNone(gc.stall_boundary(mk(0.035))["label"])
        self.assertIsNone(gc.stall_boundary(mk(0.09))["label"])
        self.assertIsNone(gc.stall_boundary([])["stall_frac"])
        self.assertIsNone(gc.stall_boundary([30.0] * 100)["label"])         # exactly 30 ms is not a stall (> 30)


class TestMisc(unittest.TestCase):
    def test_outputs_identical(self):
        self.assertEqual(gc.outputs_identical({"a": [1, 2]}, {"a": [1, 2]})["verdict"], gc.PASS)
        self.assertEqual(gc.outputs_identical({"a": [1, 2]}, {"a": [1, 3]})["verdict"], gc.FAIL)
        self.assertEqual(gc.outputs_identical({"a": [1, 2]}, {})["verdict"], gc.FAIL)
        self.assertEqual(gc.outputs_identical({}, {})["verdict"], gc.FAIL)

    def test_reg_registered(self):
        self.assertEqual(gc.REG["tost_min_pairs"], 8)
        self.assertAlmostEqual(gc.REG["tost_delta_log"], math.log(1.015))
        self.assertEqual(len(gc.reg_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
