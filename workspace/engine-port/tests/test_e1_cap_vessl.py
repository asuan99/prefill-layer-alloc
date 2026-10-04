"""E-1 cap campaign (PREREG_E1_CAP_VESSL_2026-10-04.md): runner + helper.

What is tested, and the negative control that makes each test non-vacuous
(lesson 53: a test that also passes on the broken code certifies nothing):

  HE0 command lines     server / bench flag sets are extracted from BOTH the HE0
                        sbatch and the runner; the runner may differ ONLY by the
                        registered additions.  Control: dropping a flag from a
                        copy of the runner breaks the comparison.
  arm wiring (dry-run)  the runner's DRYRUN command log carries, per arm, the
                        registered cap / pool / telemetry / controller env, in
                        the registered block order, plus the CG2 pair.
  allowlist / block     an unregistered PDMUX_* aborts; a test knob outside
                        DRYRUN aborts; an unregistered block aborts.
  fail-closed wiring    every drift check is wired to an ABORT.
  budget enforcer       a fake run that would overrun is cut at the deadline.
                        Control: both kill layers removed -> it overruns.  One
                        layer removed is an EQUIVALENT mutant (recorded, lesson 254).
  analysis              co-residency (a), cap-bound fraction and pin check are
                        exact on a synthetic run with known geometry.
  label mutants         the registered rule is re-loaded from mutated source; each
                        mutant must change at least one registered outcome.  An
                        unmutated reload must not (control for false kills).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
ROOT = ENG.parents[1]
HERE = ENG / "results" / "e1_cap_vessl"
RUNNER = HERE / "e1_cap_vessl.sh"
HELPER = HERE / "e1_plan.py"
HE0 = ENG / "results" / "slo_sched" / "sharegpt_vary_bench.sbatch"
BASH = "/usr/bin/bash"

sys.path.insert(0, str(HERE))
import e1_plan as E  # noqa: E402


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env.update(extra)
    return env


def _flags(block: str):
    return set(re.findall(r"(--[a-z0-9-]+)", block))


def _between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


class TestHE0CommandLines(unittest.TestCase):
    SRV_ADDED = {"--random-seed"}                 # prereg sec 3-2
    SRV_CONDITIONAL = {"--max-mamba-cache-size"}  # M arms only
    BENCH_ADDED = {"--seed"}                      # block seed (HE0 used the default 1)

    def _pairs(self, runner_text):
        he0 = HE0.read_text()
        srv0 = _between(he0, "python -m sglang.launch_server", "SRV=$!")
        bench0 = _between(he0, "python -m sglang.bench_serving", "| grep")
        srv1 = _between(runner_text, "srv_cmd () {", "\n}\n")
        bench1 = _between(runner_text, "bench_cmd () {", "\n}\n")
        return (srv0, srv1), (bench0, bench1)

    def _check(self, runner_text):
        (s0, s1), (b0, b1) = self._pairs(runner_text)
        f_s0, f_s1, f_b0, f_b1 = _flags(s0), _flags(s1), _flags(b0), _flags(b1)
        f_s0.discard("--disable-cuda-graph")       # HE0 CGFLAGS are empty unless PDMUX_NO_CG
        f_s0.discard("--disable-piecewise-cuda-graph")
        return (f_s1 - f_s0 == self.SRV_ADDED | self.SRV_CONDITIONAL and f_s0 <= f_s1
                and f_b1 - f_b0 == self.BENCH_ADDED and f_b0 <= f_b1)

    def test_runner_is_he0_plus_registered_additions(self):
        self.assertTrue(self._check(RUNNER.read_text()))

    def test_control_dropped_flag_is_detected(self):
        src = RUNNER.read_text()
        mutant = src.replace(" --chunked-prefill-size -1", "", 1)
        self.assertNotEqual(mutant, src)
        self.assertFalse(self._check(mutant))
        mutant2 = src.replace(' --sharegpt-context-len "$SGCTX"', "", 1)
        self.assertNotEqual(mutant2, src)
        self.assertFalse(self._check(mutant2))

    def test_constants_match_he0(self):
        he0 = HE0.read_text()
        self.assertIn('MODEL="${MODEL:-Zyphra/Zamba2-2.7B}"', he0)
        self.assertEqual(E.MODEL, "Zyphra/Zamba2-2.7B")
        self.assertIn('CTX="${CTX:-4096}"', he0)
        self.assertEqual(E.CTX, 4096)
        self.assertIn("--mem-fraction-static 0.82 --max-running-requests 48", he0)
        self.assertEqual((E.MEM_FRACTION, E.ARMS["S44_C48"]["cap"]), (0.82, 48))
        self.assertIn('NP="${NP:-200}"; ROUNDS="${ROUNDS:-3}"', he0)
        self.assertEqual((E.NP, E.ROUNDS), (200, 3))
        self.assertIn('RLO="${3:-3}"; RHI="${4:-12}"', he0)
        self.assertEqual((E.RATE_LO, E.RATE_HI), (3, 12))
        self.assertIn("--attention-backend triton", he0)
        self.assertIn("--sharegpt-context-len 4000", he0)
        self.assertIn("PDMUX_SLO_MODE=binding PDMUX_TPOT_SLO_MS=60", he0)
        self.assertIn('PDMUX_SLO_ANCHOR_IDX="${PDMUX_SLO_ANCHOR_IDX:-2}"', he0)


class TestHelper(unittest.TestCase):
    def test_selftest(self):
        E.selftest()

    def test_configs_are_registered_bytes(self):
        self.assertEqual(E.verify_configs(ENG), [])

    def test_config_drift_is_reported(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "results" / "slo_sched"
            d.mkdir(parents=True)
            for n in E.CONFIG_DIGESTS:
                shutil.copy(ENG / "results" / "slo_sched" / n, d / n)
            (d / "pdmux_d24.yml").write_text((d / "pdmux_d24.yml").read_text().replace("84, 24", "86, 22"))
            bad = E.verify_configs(Path(td))
            self.assertEqual(len(bad), 1)
            self.assertIn("pdmux_d24.yml", bad[0])

    def test_arm_table(self):
        self.assertEqual(len(E.ARMS), 12)
        pools = {a: E.expected_pool(a) for a in E.ARMS}
        self.assertEqual(pools["S24_C48"], 48)
        self.assertEqual(pools["S24_M"], 192)
        self.assertEqual(pools["S24_C192"], 192)
        # cap test M -> C192: exactly one knob (cap) differs; pool control C48 -> M: one knob (pool)
        for d in (24, 44):
            m, c192, c48 = E.ARMS[f"S{d}_M"], E.ARMS[f"S{d}_C192"], E.ARMS[f"S{d}_C48"]
            self.assertEqual((m["cap"], pools[f"S{d}_M"]), (48, 192))
            self.assertEqual((c192["cap"], pools[f"S{d}_C192"]), (192, 192))
            self.assertEqual((c48["cap"], pools[f"S{d}_C48"]), (48, 48))
        # observer pairs differ only in tel
        for on, off in E.OBSERVER_PAIRS:
            a, b = dict(E.ARMS[on]), dict(E.ARMS[off])
            self.assertEqual((a.pop("tel"), b.pop("tel")), (1, 0))
            a.pop("role"); b.pop("role")
            self.assertEqual(a, b)

    def test_orders_are_permutations_and_fixed(self):
        for k in E.BLOCK_SEEDS:
            o = E.block_order(k)
            self.assertEqual(sorted(o), sorted(E.ARMS))
            self.assertEqual(o, E.block_order(k))
        self.assertGreater(len({tuple(E.block_order(k)) for k in E.BLOCK_SEEDS}), 6)

    def test_budget_numbers(self):
        b = E.budget()
        self.assertLessEqual(b["job_est_gpu_h"], 4.0)          # ops model: Job <= 3-4 GPU-h
        self.assertEqual(b["job_cap_plus_tail_s"], E.HARD_CAP_S + E.TAIL_RESERVE_S)


class TestAnalysisGeometry(unittest.TestCase):
    """Synthetic run with known geometry: per admission a 0.10 s prefill span and
    eight 0.019 s decode steps every 0.02 s from the span start; steps 0-4 overlap
    the span.  => co-resident decode time / decode time = 5/8 exactly."""

    def test_coresidency_capbound_pin(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            E._synth_run(rd, "S24_C48", 1000.0, cap_bound=0.7)
            r = E.analyze_run(rd, "S24_C48")
            self.assertTrue(r["valid"], r["invalid_reasons"])
            hi = r["telemetry"]["phase"]["HI"]
            self.assertAlmostEqual(hi["coresidency_a_decode"], 5 / 8, places=6)
            self.assertAlmostEqual(hi["cap_bound_admission_frac"], 0.7, places=9)
            self.assertAlmostEqual(r["server"]["log_cap_bound_frac"]["HI"], 0.7, places=9)
            self.assertEqual(r["telemetry"]["phase"]["LO"]["cap_bound_admission_frac"], 0.0)
            self.assertEqual(hi["pin_frac"], 1.0)
            self.assertEqual(r["HI"]["completed"], [200, 200, 200])

    def test_banner_mismatch_invalidates(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            E._synth_run(rd, "S24_M", 1000.0)
            log = rd / "srv_S24_M.log"
            log.write_text(log.read_text().replace("max_mamba_cache_size: 192", "max_mamba_cache_size: 48"))
            r = E.analyze_run(rd, "S24_M")
            self.assertFalse(r["valid"])
            self.assertTrue(any("max_mamba_cache_size" in x for x in r["invalid_reasons"]))

    def test_missing_round_invalidates(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            E._synth_run(rd, "S44_C192", 1000.0)
            f = rd / "bench_HI_S44_C192.jsonl"
            f.write_text("\n".join(f.read_text().splitlines()[:2]) + "\n")
            self.assertFalse(E.analyze_run(rd, "S44_C192")["valid"])

    def test_off_arm_needs_no_telemetry(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            E._synth_run(rd, "S34_C48_OFF", 1000.0)
            self.assertFalse((rd / "tel_S34_C48_OFF.jsonl").exists())
            r = E.analyze_run(rd, "S34_C48_OFF")
            self.assertTrue(r["valid"], r["invalid_reasons"])
            self.assertNotIn("telemetry", r)


def _reload(src: str):
    mod = types.ModuleType("e1_plan_mutant")
    mod.__file__ = str(HELPER)
    exec(compile(src, str(HELPER), "exec"), mod.__dict__)
    return mod


FIXTURES = [
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90}, {}),
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 1.10}, {}),
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 1.02}, {}),   # residual ~2%: inside 3% gate, outside 1%
    ({"S24_C48": 1.10, "S24_M": 1.05, "S24_C192": 1.05}, {}),   # pool matters, cap does not
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90}, {"cap_bound48": 0.3}),
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90}, {"cap_bound192": 0.3}),
    ({"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90}, {"kv": 0.95}),
    ({"S24_C48": 1.01, "S24_M": 1.10, "S24_C192": 0.90}, {}),
    ({"S24_C48": 1.10, "S24_M": 1.00, "S24_C192": 0.90}, {}),   # sign lost when only the pool moves
]


def _outcomes(M, dirs) -> tuple:
    """Registered label on fixed synthetic campaigns (analysed once, labelled by M)."""
    return tuple(M.label_blocks(M.collect(d)[0])["label"] for d in dirs)


class TestLabelMutants(unittest.TestCase):
    MUTANTS = {
        "gate_1pct": ("GATE = math.log(1.03)", "GATE = math.log(1.01)"),
        "cap_test_vs_c48": ('did_cap = mean_ci([p["dM"] - p["d192"] for p in per])',
                            'did_cap = mean_ci([p["d48"] - p["d192"] for p in per])'),
        "m1_dropped": ('    if not manip["M1_ok"]:\n        return "UNRESOLVED_CAP48_NOT_BINDING"\n', ""),
        "m2_dropped": ('    if not manip["M2_ok"]:\n        return "UNRESOLVED_CAP192_BINDS"\n', ""),
        "m3_dropped": ('    if not manip["M3_ok"]:\n        return "UNRESOLVED_KV_BINDS"\n', ""),
        "pool_check_dropped": ('    if not sig_pos(dM):\n        return "UNRESOLVED_SIGN_LOST_AT_POOL192"\n', ""),
        "baseline_dropped": ('    if not sig_pos(d48):\n        return "UNRESOLVED_BASELINE_REVERSED" if sig_neg(d48) else "UNRESOLVED_BASELINE_NOT_REPRODUCED"\n', ""),
        "reversed_before_did": ('    if sig_neg(d192):\n        return "CAP_BOUND_REVERSED"\n', ""),
        "capbound_queue_ignored": ('int(s["running_bs"] + s["n_new"] >= s["cap"] and s["queue_len"] > 0)',
                                   'int(s["running_bs"] + s["n_new"] >= s["cap"])'),
    }

    @classmethod
    def setUpClass(cls):
        cls.src = HELPER.read_text()
        cls._td = tempfile.TemporaryDirectory()
        cls.dirs = []
        for i, (eff, kw) in enumerate(FIXTURES):
            d = Path(cls._td.name) / f"c{i}"
            E._synth_campaign(d, eff, noise=0.003, **kw)
            cls.dirs.append(d)
        cls.base = _outcomes(_reload(cls.src), cls.dirs)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_registered_outcomes(self):
        self.assertEqual(self.base, (
            "CAP_BOUND_REVERSED", "NOT_CAP_BOUND", "CAP_BOUND_VANISHED", "NOT_CAP_BOUND",
            "UNRESOLVED_CAP48_NOT_BINDING", "UNRESOLVED_CAP192_BINDS", "UNRESOLVED_KV_BINDS",
            "UNRESOLVED_BASELINE_NOT_REPRODUCED", "UNRESOLVED_SIGN_LOST_AT_POOL192"))

    def test_control_unmutated_reload_agrees(self):
        self.assertEqual(_outcomes(_reload(self.src), self.dirs), self.base)

    def test_each_mutant_changes_an_outcome(self):
        survivors = []
        for name, (old, new) in self.MUTANTS.items():
            self.assertEqual(self.src.count(old), 1, f"{name}: anchor")
            if _outcomes(_reload(self.src.replace(old, new)), self.dirs) == self.base:
                survivors.append(name)
        # capbound_queue_ignored acts in the ANALYSIS stage, which these fixtures
        # ran once with the unmutated module; it is pinned by the dedicated test
        # below instead.  Recorded, not hidden (lesson 254).
        self.assertEqual(survivors, ["capbound_queue_ignored"])

    def test_unmeasured_manipulation_is_load_bearing(self):
        # an unparsed server log must not read as "KV not binding"
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "srv.log"
            log.write_text("[2026-10-04 00:00:00] max_total_num_tokens=1, x, max_running_requests=48\n")
            self.assertTrue(math.isnan(E.parse_server_log(log, [], 48)["max_full_token_usage"]))
        old = '    if not manip.get("measured", True):\n        return "UNRESOLVED_MANIPULATION_UNMEASURED"\n'
        self.assertEqual(self.src.count(old), 1)
        with self.assertRaises(AssertionError):           # the mutant fails the registered selftest
            _reload(self.src.replace(old, "")).selftest()
        _reload(self.src).selftest()                      # control

    def test_capbound_queue_condition_is_load_bearing(self):
        ev = [{"event": "prefill_span_start", "timestamp_monotonic_s": 1.0 + i, "n_new": 1,
               "running_bs": 47, "queue_len": 0 if i % 2 else 2, "cap": 48, "dropped_events": 0}
              for i in range(10)]
        ev += [{"event": "decode_iteration", "timestamp_monotonic_s": 11.0, "t_launch": 10.9,
                "t_sync": 11.0, "decode_sms": 44, "prefill_in_flight": False,
                "prefill_kernel_done_wait": False, "dropped_events": 0}]
        marks = [{"round": 1, "phase": "HI", "mono_start": 0.0, "mono_end": 20.0,
                  "wall_start": 0, "wall_end": 0, "rc": 0}]
        ok = E.telemetry_metrics(ev, marks, 48, 44)["phase"]["HI"]["cap_bound_admission_frac"]
        self.assertAlmostEqual(ok, 0.5)
        old, new = self.MUTANTS["capbound_queue_ignored"]
        M = _reload(self.src.replace(old, new))
        self.assertAlmostEqual(M.telemetry_metrics(ev, marks, 48, 44)["phase"]["HI"]["cap_bound_admission_frac"], 1.0)


class TestRev11DescriptiveFields(unittest.TestCase):
    """rev1.1 (audit R5, R9): descriptive fields only -- they must never feed decide()."""

    MARKS = [{"round": 1, "phase": "HI", "mono_start": 0.0, "mono_end": 100.0,
              "wall_start": 0, "wall_end": 0, "rc": 0}]

    @staticmethod
    def _dec(t, bs, q, inflight):
        return {"event": "decode_iteration", "timestamp_monotonic_s": t + 0.01, "t_launch": t,
                "t_sync": t + 0.01, "bs": bs, "queue_len": q, "decode_sms": 44,
                "prefill_in_flight": inflight, "prefill_kernel_done_wait": False, "dropped_events": 0}

    def _events(self):
        # the phase window opens at the first work event (here an admission at 0.5 s)
        ev = [{"event": "prefill_span_start", "timestamp_monotonic_s": 0.5, "n_new": 1, "running_bs": 1,
               "queue_len": 0, "cap": 48, "dropped_events": 0}]
        ev += [self._dec(1.0 + 0.02 * i, 30, 2, False) for i in range(5)]       # non-cap block (count)
        ev += [self._dec(2.0 + 0.02 * i, 30, 2, True) for i in range(5)]        # prefill in flight: not M2b
        ev += [self._dec(3.0 + 0.02 * i, 48, 2, False) for i in range(5)]       # at cap: not M2b
        ev += [self._dec(4.0 + 0.02 * i, 30, 0, False) for i in range(5)]       # no queue: not M2b
        ev += [self._dec(10.0, 30, 0, False)]                                   # 5.9 s gap -> one stall
        return ev

    def test_m2b_and_stalls(self):
        ph = E.telemetry_metrics(self._events(), self.MARKS, 48, 44)["phase"]["HI"]
        self.assertAlmostEqual(ph["M2b_noncap_block_frac_count"], 5 / 21)
        self.assertEqual(ph["stalls"]["count"], 1)
        self.assertGreater(ph["stalls"]["max_s"], 5.0)

    def test_m2b_mutant_killed(self):
        src = HELPER.read_text()
        old = 'and not e.get("prefill_in_flight") and int(e["bs"]) < cap]'
        self.assertEqual(src.count(old), 1)
        M = _reload(src.replace(old, 'and int(e["bs"]) < cap]'))
        self.assertNotAlmostEqual(
            M.telemetry_metrics(self._events(), self.MARKS, 48, 44)["phase"]["HI"]["M2b_noncap_block_frac_count"], 5 / 21)

    def test_log_silence(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "srv.log"
            log.write_text("\n".join([
                "[2026-10-04 00:00:00] max_total_num_tokens=1, x, max_running_requests=48",
                "[2026-10-04 00:00:10] Decode batch, #running-req: 1, #full token: 2, full token usage: 0.10, mamba num: 1, #queue-req: 0",
                "[2026-10-04 00:00:11] Decode batch, #running-req: 1, #full token: 2, full token usage: 0.10, mamba num: 1, #queue-req: 0",
                "[2026-10-04 00:00:15] Decode batch, #running-req: 1, #full token: 2, full token usage: 0.10, mamba num: 1, #queue-req: 0",
            ]) + "\n")
            mk = [{"round": 1, "phase": "HI", "mono_start": 0, "mono_end": 1,
                   "wall_start": E._naive("2026-10-04 00:00:10"), "wall_end": E._naive("2026-10-04 00:00:20"), "rc": 0}]
            sil = E.parse_server_log(log, mk, 48)["log_silences_HI"]
            self.assertEqual((sil["count"], sil["max_s"]), (1, 3.0))
            # mutant: forgetting the 1 s stamp resolution counts adjacent seconds as silence
            src = HELPER.read_text()
            old = "if b - a - 1.0 >= STALL_S]"
            self.assertEqual(src.count(old), 1)
            M = _reload(src.replace(old, "if b - a >= STALL_S]"))
            self.assertNotEqual(M.parse_server_log(log, mk, 48)["log_silences_HI"]["count"], 1)

    def test_descriptive_fields_do_not_reach_decide(self):
        import inspect
        body = inspect.getsource(E.decide)
        for k in ("M2b", "stall", "per_arm", "silence"):
            self.assertNotIn(k, body)

    def test_allowed_forbidden_text_rev11(self):
        self.assertEqual(len(E.FORBIDDEN_TEXT), 8)
        self.assertIn("goodput", E.FORBIDDEN_TEXT[-1])
        self.assertNotIn("was not reduced", E.ALLOWED_TEXT["NOT_CAP_BOUND"])
        self.assertIn("DiD_cap", E.ALLOWED_TEXT["NOT_CAP_BOUND"])
        self.assertNotIn("remained.", E.ALLOWED_TEXT["CAP_ATTENUATED"])
        self.assertIn("not judged", E.ALLOWED_TEXT["CAP_ATTENUATED"])

    def test_prereg_carries_e1c_caveats_verbatim(self):
        verdict = (HERE / "VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md").read_text()
        prereg = (HERE / "PREREG_E1_CAP_VESSL_2026-10-04.md").read_text()
        bullets = [ln for ln in verdict.splitlines() if ln.startswith("- **E1C-")]
        self.assertEqual(len(bullets), 9)
        for ln in bullets:
            self.assertIn(ln, prereg, ln[:40])


class TestRunnerEnv(unittest.TestCase):
    def _run(self, **env):
        with tempfile.TemporaryDirectory() as td:
            e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=td, PDMUX_IO=td,
                           PDMUX_JOB_NAME="t_env", PDMUX_E1_TEST_OUT_ROOT=td, **env)
            r = subprocess.run([BASH, str(RUNNER)], env=e, capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout + r.stderr

    def test_unregistered_knob_aborts(self):
        rc, out = self._run(PDMUX_E1_DRYRUN="1", PDMUX_E1_BLOCK="1", PDMUX_STICKY_PARTITION="1")
        self.assertEqual(rc, 2, out)
        self.assertIn("ABORT_ENV unregistered knob PDMUX_STICKY_PARTITION", out)

    def test_phase_events_cannot_be_injected(self):
        rc, out = self._run(PDMUX_E1_DRYRUN="1", PDMUX_E1_BLOCK="1", PDMUX_PHASE_EVENTS="1")
        self.assertEqual(rc, 2, out)

    def test_test_knob_without_dryrun_aborts(self):
        rc, out = self._run(PDMUX_E1_BLOCK="1", PDMUX_E1_TEST_HARD_CAP_S="700")
        self.assertEqual(rc, 2, out)
        self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", out)

    def test_unregistered_block_aborts(self):
        for b in ("", "0", "9", "x"):
            rc, out = self._run(PDMUX_E1_DRYRUN="1", PDMUX_E1_BLOCK=b)
            self.assertEqual(rc, 2, (b, out))
            self.assertIn("ABORT_BLOCK", out)


class TestFailClosedWiring(unittest.TestCase):
    def test_checks_abort(self):
        src = RUNNER.read_text()
        for check, abort in (('python3 "$HELPER" --verify-configs', "ABORT_CONFIG_DRIFT"),
                             ('"$MODEL_DIR/refs/main"', "ABORT_MODEL_REVISION"),
                             ('sha256sum "$SGPT"', "ABORT_TRACE"),
                             ("perf_counter_clock=clock_gettime(CLOCK_MONOTONIC)", "ABORT_CLOCK"),
                             ('cmp -s "$MIXIN"', "ABORT_ENGINE"),
                             ('--compare-cg2', "ABORT_CG2_MISMATCH")):
            i = src.rindex(check)
            self.assertIn(abort, src[i:i + 400], check)

    def test_admission_precedes_boot(self):
        src = RUNNER.read_text()
        body = src[src.index("run_arm () {"):]
        self.assertLess(body.index('if [ "$REM" -lt "$NEED" ]; then'),
                        body.index('setsid "${SRV[@]}"'))

    def test_cg2_runs_before_any_arm(self):
        src = RUNNER.read_text()
        self.assertLess(src.index("\nrun_cg2\n"), src.index('for ARM in "${ORDER[@]}"; do run_arm'))


class TestDryRun(unittest.TestCase):
    def test_commands_per_arm(self):
        with tempfile.TemporaryDirectory() as td:
            e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=td, PDMUX_IO=td,
                           PDMUX_JOB_NAME="t_dry", PDMUX_E1_TEST_OUT_ROOT=td,
                           PDMUX_E1_DRYRUN="1", PDMUX_E1_BLOCK="3",
                           PDMUX_E1_TEST_SKIP_PREFLIGHT="1")
            r = subprocess.run([BASH, str(RUNNER)], env=e, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            out = Path(td) / "e1v_t_dry"
            cmds = (out / "DRYRUN_COMMANDS.txt").read_text()
            self.assertEqual((out / "block_3" / "ORDER.txt").read_text().split(), E.block_order(3))
            arms = re.findall(r"^ARM (\S+)$", cmds, re.M)
            self.assertEqual(arms, E.block_order(3))
            self.assertEqual(cmds.count("CG2 on server:"), 1)
            self.assertEqual(cmds.count("CG2 off server:"), 1)
            chunks = dict(zip(arms, re.split(r"^ARM \S+$", cmds, flags=re.M)[1:]))
            for arm, ch in chunks.items():
                a = E.ARMS[arm]
                self.assertIn(f"--max-running-requests {a['cap']}", ch, arm)
                self.assertEqual("--max-mamba-cache-size 192" in ch, a["pool"] == 192, arm)
                self.assertEqual("PDMUX_TELEMETRY_PATH=" in ch, bool(a["tel"]), arm)
                self.assertEqual("PDMUX_PHASE_EVENTS=1" in ch, bool(a["tel"]), arm)
                self.assertEqual("PDMUX_SLO_FEAS_GATE=1" in ch, a["ctrl"] == "bind_gate", arm)
                self.assertIn(E.config_name(arm), ch, arm)
                self.assertIn("PDMUX_GREEN_READOUT=1", ch, arm)
                self.assertIn(f"--seed {E.BLOCK_SEEDS[3]}", ch, arm)
                self.assertIn("--request-rate 3", ch)
                self.assertIn("--request-rate 12", ch)
            summ = json.loads((out / "block_3" / "BLOCK_SUMMARY.json").read_text())
            self.assertFalse(summ["primary_valid"])          # a dry run is never a valid block


class TestBudgetEnforcer(unittest.TestCase):
    """Fake runs (DRYRUN) under the same timeout/watchdog/admission machinery.
    cap 620 s - tail 600 s => deadline ~20 s after start; a fake run sleeps 60 s."""

    CAP, FAKE, LIMIT_S = "620", "60", 120

    def _run(self, script: Path):
        with tempfile.TemporaryDirectory() as td:
            e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=td, PDMUX_IO=td,
                           PDMUX_JOB_NAME="t_budget", PDMUX_E1_TEST_OUT_ROOT=td,
                           PDMUX_E1_DRYRUN="1", PDMUX_E1_BLOCK="1",
                           PDMUX_E1_TEST_SKIP_PREFLIGHT="1",
                           PDMUX_E1_TEST_HARD_CAP_S=self.CAP, PDMUX_E1_TEST_FAKE_RUN_S=self.FAKE)
            t0 = time.monotonic()
            try:
                r = subprocess.run([BASH, str(script)], env=e, capture_output=True,
                                   text=True, timeout=self.LIMIT_S)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "e1_fake_run"], check=False)
                subprocess.run(["pkill", "-f", "sleep 100000"], check=False)
                return None, time.monotonic() - t0, {}
            wall = time.monotonic() - t0
            out = Path(td) / "e1v_t_budget"
            files = {p.name: p.read_text() for p in out.glob("*.txt")}
            return r.returncode, wall, files

    def test_overrunning_run_is_cut_at_the_deadline(self):
        rc, wall, f = self._run(RUNNER)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 50.0)
        fake = f.get("FAKE_RUNS.txt", "")
        self.assertRegex(fake, r"fake_rc=(124|143|137) arm=")
        self.assertIn("NOT_ATTEMPTED_BUDGET", f.get("BUDGET_SKIPPED.txt", ""))
        self.assertEqual(fake.count("fake_rc="), 1, fake)

    def test_control_both_kill_layers_removed_overruns(self):
        src = RUNNER.read_text()
        m = src.replace('timeout -k 2 "$CAPT" bash -c', 'bash -c')
        m = m.replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        self.assertEqual(m.count("sleep 100000"), 1)
        self.assertNotIn('timeout -k 2 "$CAPT" bash -c', m)
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e1_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._run(mut)
        self.assertGreaterEqual(wall, float(self.FAKE), "mutant was still cut -- the test is vacuous")

    def test_equivalent_mutant_one_layer_removed_still_cuts(self):
        m = RUNNER.read_text().replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n',
                                       '  sleep 100000\n')
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e1_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._run(mut)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 50.0)


if __name__ == "__main__":
    unittest.main()
