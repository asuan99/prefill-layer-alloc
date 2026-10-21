"""CPU tests of scripts/vessl/engine_gate_decomp.py (observer-effect decomposition Job) against the FAKE server / loadgen.

Proves: Williams plan + registered sha, per-level instrument assertions (only the level's probe files / log lines / I-4 nullness),
registered order enforcement, small export tree with raw data off it, the selection rule on synthetic effects, and three mutants
(level assertion failure, order violation, export size exceeded).  Proves nothing about the real engine, sglang or a GPU.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import test_engine_gate_driver as TD  # noqa: E402

D = TD.D
ENG = TD.ENG
sys.path.insert(0, str(ENG / "scripts/vessl"))
import engine_gate_decomp as DM  # noqa: E402

FX = None
BASE = {}


def setUpModule():
    global FX
    FX = TD.Fixture().finish()
    DM.bind(D)


def tearDownModule():
    shutil.rmtree(FX.tmp, ignore_errors=True)


def run(name, **kw):
    root = FX.tmp / name
    kw.setdefault("FAKE_BG_ACTIVE", "4")
    rc, res = FX.run("decomp", root, job=name, **kw)
    return root, rc, res


def raw_of(root):
    return FX.tmp / "data" / "runs" / root.name / "raw"


class TestPlan(unittest.TestCase):
    def test_rectangle_balanced_and_registered(self):
        self.assertTrue(DM.balanced_ok())
        self.assertEqual(DM.order_sha(), DM.ORDER_SHA)
        seq = DM.sequence()
        self.assertEqual(len(seq), 20)
        self.assertEqual([k for k, *_ in seq], list(range(1, 21)))
        for lv in DM.LEVELS:      # every level once per block and once per position
            self.assertEqual(sorted(b for _k, b, _p, l, _s in seq if l == lv), [1, 2, 3, 4])
            self.assertEqual(len({p for _k, _b, p, l, _s in seq if l == lv}), 4)      # 4 distinct positions of 5

    def test_unbalanced_plan_detected(self):
        bad = (DM.PLAN[0],) * 4
        self.assertFalse(DM.balanced_ok(bad))
        L = DM.LEVELS
        same_pos = (L, L[1:] + L[:1], L[2:] + L[:2], L[:1] + L[2:] + L[1:2])      # last row: OFF again at position 1
        self.assertFalse(DM.balanced_ok(same_pos))
        pairs = (L, tuple(reversed(L)), L, tuple(reversed(L)))
        self.assertFalse(DM.balanced_ok(pairs))
        pc = {}
        for r in DM.PLAN:
            for i in range(4):
                pc[(r[i], r[i + 1])] = pc.get((r[i], r[i + 1]), 0) + 1
        self.assertEqual((len(pc), max(pc.values())), (15, 2))           # disclosed limit: one repeated neighbour pair

    def test_level_env_separates_instruments_without_engine_change(self):
        self.assertEqual(DM.LEVEL_ENV["OFF"], {})
        self.assertEqual(set(DM.LEVEL_ENV["I5"]), {"PDMUX_ADMISSION_PATH"})
        self.assertEqual(DM.LEVEL_ENV["I6"]["PDMUX_DECODE_PROBE_GPU_EVENTS"], "0")
        self.assertEqual(DM.LEVEL_ENV["I46"]["PDMUX_DECODE_PROBE_GPU_EVENTS"], "1")
        self.assertEqual(DM.LEVEL_ENV["TEL"], {"PDMUX_TELEMETRY_PATH": "{dir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1"})
        for lv, env in DM.LEVEL_ENV.items():
            if lv != "TEL":
                self.assertFalse({"PDMUX_TELEMETRY_PATH", "PDMUX_PHASE_EVENTS"} & set(env), lv)

    def test_plan_info_and_shell(self):
        out = D.plan_info("decomp", None, 1.48, 0)
        self.assertIn("hard cap : 250 min", out)
        self.assertIn("20 instances", out)
        import subprocess
        for f in ("engine_gate_decomp.sh", "engine_gate_common.sh", "launch.sh"):
            r = subprocess.run(["bash", "-n", str(ENG / "scripts/vessl" / f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f + r.stderr)


class TestFullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.rc, cls.res = run("decomp_base")
        cls.raw = raw_of(cls.root)

    def test_complete_and_order(self):
        self.assertEqual(self.rc, 0, self.res.get("checks", {}).get("abort"))
        self.assertEqual(self.res["verdict"], "DIAGNOSTIC_COMPLETE")
        self.assertEqual(self.res["checks"]["order"]["n_executed"], 20)
        for k in ("order", "level_assertions", "digest_3x", "export_size"):
            self.assertEqual(self.res["checks"][k]["verdict"], "PASS", k)
        self.assertEqual(len(self.res["digests"]), 3)

    def test_only_the_levels_probe_files_exist(self):
        want = {"OFF": set(), "I5": {"admission.jsonl"}, "I6": {"decode_probe.jsonl"}, "I46": {"decode_probe.jsonl"},
                "TEL": {"tel.jsonl"}}
        for lv, files in want.items():
            dirs = sorted(self.raw.glob(f"b*p*_{lv}"))
            self.assertEqual(len(dirs), 4)
            for d in dirs:
                have = {n for n in ("decode_probe.jsonl", "admission.jsonl", "tel.jsonl") if (d / n).exists()}
                self.assertEqual(have, files, d.name)

    def test_i4_field_null_only_without_events(self):
        for lv, expect_null in (("I6", True), ("I46", False)):
            rows = [r for r in D.read_jsonl(next(iter(sorted(self.raw.glob(f"b*p*_{lv}")))) / "decode_probe.jsonl")
                    if r.get("event") == "decode_forward"]
            self.assertTrue(rows)
            nn = sum(1 for r in rows if r["decode_fwd_gpu_ms"] is not None)
            self.assertEqual(nn == 0, expect_null, lv)

    def test_bulk_data_stays_off_the_export_tree(self):
        names = {p.name for p in self.root.rglob("*") if p.is_file()}
        for big in ("requests.jsonl", "decode_probe.jsonl", "admission.jsonl", "server.log", "trace.jsonl"):
            self.assertNotIn(big, names)
            self.assertTrue(list(self.raw.rglob(big)) or big == "admission.jsonl" or big == "decode_probe.jsonl")
        self.assertTrue(list(self.raw.rglob("requests.jsonl")))
        mb = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) / 1e6
        self.assertLess(mb, 5)
        man = json.loads((self.root / "EXPORT_MANIFEST.json").read_text())
        self.assertGreater(len(man["files"]), 50)
        for f in man["files"]:
            self.assertEqual((self.root / f["path"]).stat().st_size, f["bytes"])
        idx = json.loads(next(iter(sorted(self.root.glob("b1p*_*")))).joinpath("raw_index.json").read_text())
        self.assertTrue(any(f["path"].endswith("requests.jsonl") for f in idx["files"]))

    def test_no_effect_not_flagged(self):
        st = self.res["checks"]["decomp"]
        for m in st["selection"].values():
            self.assertEqual(m["flagged"], [])
            self.assertEqual(m["largest"], "NONE_RESOLVED")
        self.assertEqual(st["metrics"]["lambda_thr"]["I5"]["n"], 4)


class TestSelectionRule(unittest.TestCase):
    def test_synthetic_effects(self):
        def mk(lam):
            return {(b, lv): {"valid": True, "hi": {"achieved": 100 * lam.get(lv, 1.0) * (1 + 0.001 * b)},
                              "lo": {"itl_p95_ms": 20.0 * (1 / lam.get(lv, 1.0)) * (1 + 0.001 * ((b * 7) % 3))}}
                    for b in range(1, 5) for lv in DM.LEVELS}
        # I5 -3 %, I6 -1 %, I46 -1 % (=> I4 increment 0): lambda flagged I5 and I6, largest I5
        st = DM.decomp_stats(mk({"OFF": 1.0, "I5": 0.97, "I6": 0.99, "I46": 0.99}))
        sel = st["selection"]["lambda_thr"]
        self.assertEqual(sel["flagged"], ["I5", "I6"])
        self.assertEqual(sel["largest"], "I5")
        self.assertNotIn("I4_increment", sel["flagged"])
        # an improvement is OPPOSITE, never flagged
        st = DM.decomp_stats(mk({"OFF": 1.0, "I5": 1.05, "I6": 1.0, "I46": 1.0}))
        self.assertEqual(st["metrics"]["lambda_thr"]["I5"]["flag"], "OPPOSITE")
        self.assertEqual(st["selection"]["lambda_thr"]["largest"], "NONE_RESOLVED")
        # I-4 increment is the I46 vs I6 contrast
        st = DM.decomp_stats(mk({"OFF": 1.0, "I5": 1.0, "I6": 1.0, "I46": 0.95}))
        self.assertEqual(st["selection"]["lambda_thr"]["flagged"], ["I4_increment"])
        self.assertLess(st["metrics"]["lambda_thr"]["I46_total"]["pct"], -4)

    def test_tel_is_a_candidate_and_flagged(self):
        insts = {(b, lv): {"valid": True, "hi": {"achieved": 100 * (0.97 if lv == "TEL" else 1.0) * (1 + 0.001 * b)},
                           "lo": {"itl_p95_ms": 20.0}} for b in range(1, 5) for lv in DM.LEVELS}
        sel = DM.decomp_stats(insts)["selection"]["lambda_thr"]
        self.assertEqual((sel["flagged"], sel["largest"]), (["TEL"], "TEL"))

    def test_too_few_blocks_insufficient(self):
        insts = {(b, lv): {"valid": b <= 2, "hi": {"achieved": 1.0}, "lo": {"itl_p95_ms": 1.0}} for b in range(1, 5) for lv in DM.LEVELS}
        st = DM.decomp_stats(insts)
        self.assertEqual(st["metrics"]["lambda_thr"]["I5"]["flag"], "INSUFFICIENT")

    def test_slowdown_detected_end_to_end(self):
        root, rc, res = run("decomp_slow", FAKE_ON_SLOW="0.04")
        sel = res["checks"]["decomp"]["selection"]["lambda_thr"]
        self.assertEqual(rc, 0)
        self.assertEqual(set(sel["flagged"]), {"I5", "I6", "TEL"})
        self.assertNotIn("I4_increment", sel["flagged"])


class TestMutants(unittest.TestCase):
    def test_level_assertion_failure_log_line(self):
        root, rc, res = run("mut_level_log", FAKE_MODE="no_enabled_log")
        self.assertEqual(res["verdict"], "ABORT_LEVEL")
        self.assertEqual(rc, 16)
        # aborted at the first instrumented boot: OFF (position 1 of block 1) ran, then the first non-OFF boot stopped it
        self.assertLessEqual(len(list(root.glob("b*p*_*/instance.json"))), 2)

    def test_level_assertion_failure_wrong_events(self):
        bad = dict(DM.LEVEL_ENV, I6={"PDMUX_DECODE_PROBE_PATH": "{dir}/decode_probe.jsonl", "PDMUX_DECODE_PROBE_GPU_EVENTS": "1"})
        with mock.patch.object(DM, "LEVEL_ENV", bad):
            root, rc, res = run("mut_level_events")
        self.assertEqual(res["verdict"], "ABORT_LEVEL")
        self.assertIn("I6", res["checks"]["abort"]["why"])

    def test_tel_level_assertions(self):
        d = Path(tempfile.mkdtemp(dir=FX.tmp))
        (d / "server.log").write_text("x\n")
        D.dump({"pdmux_knobs": ["PDMUX_TELEMETRY_PATH", "PDMUX_PHASE_EVENTS"]}, d / "boot.json")
        (d / "tel.jsonl").write_text(json.dumps({"event": "decode_iteration"}) + "\n")
        self.assertTrue(DM.post_level_check(d, "TEL")["ok"])
        self.assertFalse(DM.post_level_check(d, "OFF")["ok"])                  # telemetry file where none is allowed
        (d / "tel.jsonl").write_text(json.dumps({"event": "runtime_snapshot"}) + "\n")
        self.assertFalse(DM.post_level_check(d, "TEL")["ok"])                  # no phase events
        (d / "tel.jsonl").write_text(json.dumps({"event": "decode_iteration"}) + "\n")
        (d / "decode_probe.jsonl").write_text(json.dumps({"event": "decode_forward", "decode_fwd_gpu_ms": None}) + "\n")
        self.assertFalse(DM.post_level_check(d, "TEL")["ok"])                  # a probe file must not exist
        chk = DM.level_check("TEL")
        base = dict(log_decode_enabled=False, log_admission_enabled=False, files={"telemetry": True},
                    sglang_file="/t/sglang/x.py", tree="/t")
        self.assertEqual(chk(base)["verdict"], "PASS")
        self.assertEqual(chk(dict(base, files={"decode_probe": True}))["verdict"], "FAIL")
        self.assertEqual(chk(dict(base, log_admission_enabled=True))["verdict"], "FAIL")

    def test_post_check_catches_events_on_i6(self):
        d = Path(tempfile.mkdtemp(dir=FX.tmp))
        (d / "decode_probe.jsonl").write_text(json.dumps({"event": "decode_forward", "decode_fwd_gpu_ms": 1.2}) + "\n")
        self.assertFalse(DM.post_level_check(d, "I6")["ok"])
        self.assertFalse(DM.post_level_check(d, "I5")["ok"])        # wrong file set for I5
        (d / "decode_probe.jsonl").write_text(json.dumps({"event": "decode_forward", "decode_fwd_gpu_ms": None}) + "\n")
        self.assertTrue(DM.post_level_check(d, "I6")["ok"])
        self.assertFalse(DM.post_level_check(d, "I46")["ok"])       # I-4 events not resolving

    def test_order_violation_unregistered_plan_aborts(self):
        swapped = (DM.PLAN[1], DM.PLAN[0]) + DM.PLAN[2:]
        with mock.patch.object(DM, "PLAN", swapped):
            root, rc, res = run("mut_order_plan")
        self.assertEqual(res["verdict"], "ABORT_REG")
        self.assertFalse(list(root.glob("b*p*_*/instance.json")))

    def test_order_violation_executed_sequence_detected(self):
        root = FX.tmp / "mut_order_exec"
        shutil.copytree(FX.tmp / "decomp_base", root, ignore=shutil.ignore_patterns("cache"))
        f2, f3 = sorted(root.glob("b1p2_*/instance.json"))[0], sorted(root.glob("b1p3_*/instance.json"))[0]
        a, b = json.loads(f2.read_text()), json.loads(f3.read_text())
        a["k"], b["k"] = b["k"], a["k"]          # the two instances ran in swapped order
        f2.write_text(json.dumps(a)); f3.write_text(json.dumps(b))
        chk = DM.score_decomp(root)
        self.assertEqual(chk["order"]["verdict"], "FAIL")
        self.assertEqual(D.verdict_of("decomp", chk), "FAIL")

    def test_export_size_exceeded(self):
        root = FX.tmp / "mut_export"
        shutil.copytree(FX.tmp / "decomp_base", root, ignore=shutil.ignore_patterns("cache"))
        (root / "b1p1_OFF" / "requests_leak.jsonl").write_bytes(b"x" * 3_000_000)
        with mock.patch.dict(os.environ, {"PDMUX_DECOMP_MAX_EXPORT_MB": "1"}):
            self.assertEqual(DM.score_decomp(root)["export_size"]["verdict"], "FAIL")
            g = D.Gate("decomp", root, 120)
            raw = FX.tmp / "mut_export_raw"
            ex = DM.export_check(g, raw)
        self.assertEqual(ex["verdict"], "FAIL")
        self.assertIn("b1p1_OFF/requests_leak.jsonl", ex["spilled"])
        self.assertFalse((root / "b1p1_OFF" / "requests_leak.jsonl").exists())      # the export tree is safe again
        self.assertTrue(list((raw / "spilled_from_export").glob("*requests_leak*")))

    def test_raw_dir_inside_export_tree_aborts(self):
        with mock.patch.object(DM, "_raw_root", lambda gate: gate.root / "raw"):
            root, rc, res = run("mut_raw_in_root")
        self.assertEqual(res["verdict"], "ABORT_CONFIG")


if __name__ == "__main__":
    unittest.main()
