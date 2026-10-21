"""CPU tests of scripts/vessl/engine_gate_driver.py against a FAKE server / loadgen / unit harness.

What this proves: the Job driver's PASS / FAIL / UNRESOLVED / ABORT paths, the expected-death judgement of G-beta, the budget
handling, the instrumentation-level assertions and the mutants listed in the task (changed registered constant, inverted
death expectation, digest mismatch).  What it does NOT prove: anything about the real engine, sglang, CUDA or a GPU.
"""
import copy
import difflib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
T = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


D = _load("engine_gate_driver_t", ENG / "scripts/vessl/engine_gate_driver.py")
ED = _load("engine_digest_gt", ENG / "scripts/bootstrap/engine_digest.py")
GC = D.GC
REL = D.CONV_REL
PRISTINE = "def kernel():\n    chunk_offset = tl.program_id(1)\n    start = tl.load(q)\n    return 1\n"
PATCHED = ("def kernel():\n    chunk_offset = tl.program_id(1).to(tl.int64)  # int64: token offsets\n"
           "    start = tl.load(q).to(tl.int64)\n    return 1\n")


class Fixture:
    """A 30-entry fake engine tree whose conv file is the fake pristine/patched pair, a real unified-diff patch, digests."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gate_t_"))
        self.tree = self.tmp / "python"
        for i, rel in enumerate(ED.registered_relpaths()):
            f = self.tree / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# file {i} {rel}\n")
        (self.tree / "sglang/__init__.py").write_text("")
        (self.tree / REL).write_text(PATCHED)
        self.patch = self.tmp / "int64.patch"
        self.patch.write_text("".join(difflib.unified_diff(PRISTINE.splitlines(True), PATCHED.splitlines(True),
                                                           "a/" + REL, "b/" + REL)))
        self.base = D.tree_digest(self.tree)[0] if self._env() else None

    def _env(self):
        os.environ.update(PDMUX_GATE_TEST_MODE="1", PDMUX_GATE_PATCH_FILE=str(self.patch), PDMUX_GATE_TREE=str(self.tree))
        return True

    def finish(self):
        with mock.patch.dict(os.environ, {"PDMUX_GATE_TEST_MODE": "1", "PDMUX_GATE_PATCH_FILE": str(self.patch),
                                          "PDMUX_GATE_TREE": str(self.tree)}):
            D.reverse_patch(self.tree)
            self.rev = D.tree_digest(self.tree)[0]
            self.pristine_sha = D.sha256_file(self.tree / REL)
            D.reapply_patch(self.tree)
            self.base = D.tree_digest(self.tree)[0]
        return self

    def env(self, root, job="gate-test", **kw):
        e = {"PDMUX_GATE_TEST_MODE": "1", "PDMUX_GATE_ROOT": str(root), "PDMUX_GATE_TREE": str(self.tree),
             "PDMUX_GATE_PATCH_FILE": str(self.patch), "PDMUX_GATE_DIGEST_BASE": self.base, "PDMUX_GATE_DIGEST_REV": self.rev,
             "PDMUX_GATE_PRISTINE_SHA256": self.pristine_sha, "PDMUX_GATE_SERVER_CMD": f"{sys.executable} {T}/fake_gate_server.py",
             "PDMUX_GATE_UNIT_CMD": f"{sys.executable} {T}/fake_gate_unit.py",
             "PDMUX_GATE_CALIB_CMD": f"{sys.executable} {T}/fake_gate_calib.py",
             "PDMUX_GATE_LOADGEN_CMD": f"{sys.executable} {T}/fake_gate_loadgen.py",
             "PDMUX_GATE_GPU_MEM_CMD": "echo 100", "PDMUX_GATE_SGLANG_FILE": str(self.tree / "sglang/__init__.py"),
             "PDMUX_GATE_FAKE_ENCODER": "1", "PDMUX_GATE_SKIP_MODEL_CHECK": "1", "PDMUX_GATE_LS_B": "256,512",
             "PDMUX_GATE_TD_WAIT_S": "3", "PDMUX_GATE_DEATH_WAIT_S": "25", "PDMUX_GATE_CACHE_BASE": str(Path(root) / "cache"),
             "PDMUX_JOB_NAME": job, "PDMUX_COMMIT": "0" * 40, "PDMUX_GATE_HARDCAP_MIN": "120",
             "FAKE_GATE_CONV": str(self.tree / REL), "PDMUX_GATE_HI_RATE": "60", "PDMUX_GATE_HI_N": "30",
             "PDMUX_GATE_LO_RATE": "20", "PDMUX_GATE_LO_N": "24", "PDMUX_DATA": str(self.tmp / "data")}
        e.update({k: str(v) for k, v in kw.items()})
        return e

    def run(self, gate, root, family=None, **kw):
        env = self.env(root, **kw)
        with mock.patch.dict(os.environ, env):
            rc = D.run_gate(gate, family)
        return rc, json.loads((Path(root) / "GATE_RESULT.json").read_text())


FX = None
BASE = {}


def setUpModule():
    global FX
    FX = Fixture().finish()


def tearDownModule():
    shutil.rmtree(FX.tmp, ignore_errors=True)


class TestRegistered(unittest.TestCase):
    def test_constants_registered(self):
        self.assertEqual(D.REG_SHA, GC.reg_sha256())
        self.assertEqual(len(D.DIGEST_BASE), 64)
        self.assertNotEqual(D.DIGEST_BASE, D.DIGEST_REV)

    def test_reg_mutation_aborts(self):
        root = FX.tmp / "reg"
        with mock.patch.dict(GC.REG, {"tost_min_pairs": 7}):
            rc, res = FX.run("alpha", root)
        self.assertEqual(rc, 16)
        self.assertEqual(res["verdict"], "ABORT_REG")

    def test_test_hooks_ignored_outside_test_mode(self):
        with mock.patch.dict(os.environ, {"PDMUX_GATE_TEST_MODE": "0", "PDMUX_GATE_TREE": "/nonexistent"}):
            self.assertIsNone(D.tget("TREE"))

    def test_plan_info_and_launch_dry_run_fields(self):
        out = D.plan_info("gamma", None, 1.48, 0)
        self.assertIn("hard cap : 270 min", out)
        self.assertIn("WARNING", out)
        self.assertNotIn("WARNING", D.plan_info("gamma", "A", 1.48, 0))
        self.assertIn("120 min", D.plan_info("alpha", None, 1.48, 0))

    def test_shell_syntax(self):
        for f in ("engine_gate_common.sh", "engine_gate_alpha.sh", "engine_gate_beta.sh", "engine_gate_gamma.sh", "launch.sh"):
            r = subprocess.run(["bash", "-n", str(ENG / "scripts/vessl" / f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f + r.stderr)


class TestDigestAndBudget(unittest.TestCase):
    def test_digest_mismatch_aborts_before_any_boot(self):
        rc, res = FX.run("alpha", FX.tmp / "dig", PDMUX_GATE_DIGEST_BASE="0" * 64)
        self.assertEqual(rc, 13)
        self.assertEqual(res["verdict"], "ABORT_ENGINE_DIGEST")
        self.assertFalse((FX.tmp / "dig" / "B1").exists())

    def test_budget_exhausted_is_unresolved_not_pass(self):
        rc, res = FX.run("alpha", FX.tmp / "bud", PDMUX_GATE_HARDCAP_MIN="11")
        self.assertEqual(res["verdict"], "UNRESOLVED")
        self.assertEqual(rc, 4)
        self.assertTrue(all(s["status"] == "SKIPPED" for k, s in res["stages"].items() if k.startswith("B")))

    def test_unreleased_gpu_memory_is_unresolved(self):
        rc, res = FX.run("alpha", FX.tmp / "mem", PDMUX_GATE_GPU_MEM_CMD="echo 5000", PDMUX_GATE_ONLY_BOOTS="B1")
        self.assertNotEqual(res["verdict"], "PASS")
        self.assertEqual(res["checks"]["gpu_memory_released"]["verdict"], "UNRESOLVED")
        self.assertEqual(res["checks"]["gpu_memory_released"]["kind"], "HARNESS_UNRESOLVED")


class TestAlpha(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = FX.tmp / "alpha_base"
        cls.rc, cls.res = FX.run("alpha", cls.root)

    def test_baseline_pass(self):
        bad = {k: v for k, v in self.res["checks"].items() if v["verdict"] not in ("PASS", "UNINFORMATIVE")}
        self.assertEqual(self.res["verdict"], "PASS", bad)
        self.assertEqual(self.rc, 0)
        self.assertTrue(self.res["complete"])
        self.assertEqual([d["ok"] for d in self.res["digests"]], [True, True, True])
        self.assertEqual(D.conv_state(FX.tree), "applied")             # tree restored

    def test_checks_exist(self):
        for k in ("a_spass_agnostic", "a_cpass_llama", "b_int64_le64k_sanity_fused_sym", "emission_B7", "i6_B7", "i4_B2",
                  "i4_B6", "i5i6_B4", "schema_all_on", "instrumentation_level", "survival_all_steps", "digest_3x"):
            self.assertIn(k, self.res["checks"])

    def test_boot_records(self):
        b = json.loads((self.root / "B8/boot.json").read_text())
        self.assertEqual(b["tree_state"], "reversed")
        self.assertIn("triton_reversed", b["triton_cache_dir"])
        self.assertIn("triton_patched", json.loads((self.root / "B1/boot.json").read_text())["triton_cache_dir"])
        self.assertIn("triton_key", b)
        self.assertFalse((self.root / "B1/decode_probe.jsonl").exists())        # OFF is OFF
        self.assertTrue((self.root / "B2/decode_probe.jsonl").exists())
        self.assertTrue((self.root / "substrate.json").exists())

    def rescore(self, mutate):
        d = Path(tempfile.mkdtemp(dir=FX.tmp))
        shutil.copytree(self.root, d, dirs_exist_ok=True)
        mutate(d)
        with mock.patch.dict(os.environ, FX.env(d)):
            return D.score_alpha(d, FX.tree)

    def test_rescore_mutants_fail(self):
        def host(d):
            rows = D.read_jsonl(d / "B2/decode_probe.jsonl")
            for r in rows:
                if r.get("event") == "decode_forward":
                    r["decode_fwd_gpu_ms"] = r["decode_fwd_host_ms"]
            (d / "B2/decode_probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.assertEqual(self.rescore(host)["i4_B2"]["verdict"], "FAIL")

        def diverge(d):
            j = json.loads((d / "B4/s_pass.json").read_text())
            k = sorted(j["outputs"])[0]
            j["outputs"][k][-1] += 1
            (d / "B4/s_pass.json").write_text(json.dumps(j))
        self.assertEqual(self.rescore(diverge)["a_spass_fused_sym"]["verdict"], "FAIL")

        def int64_change(d):
            j = json.loads((d / "B9/long.json").read_text())
            j[sorted(j)[0]]["out"][0] += 1
            (d / "B9/long.json").write_text(json.dumps(j))
        self.assertEqual(self.rescore(int64_change)["b_int64_le64k_sanity_fused_sym"]["verdict"], "FAIL")

        def dead(d):
            f = d / "B2/alive_after_iso.json"
            j = json.loads(f.read_text())
            j["poll_returncode"] = 1
            f.write_text(json.dumps(j))
        self.assertEqual(self.rescore(dead)["survival_all_steps"]["verdict"], "FAIL")

        def cap_not_recorded(d):
            rows = D.read_jsonl(d / "B4/admission.jsonl")
            for r in rows:
                if r.get("event") == "admission_window":
                    r["terminal_by_category"]["cap"] = 0
                    r["n_decisions"] = 0
            (d / "B4/admission.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.assertEqual(self.rescore(cap_not_recorded)["i5i6_B4"]["verdict"], "FAIL")

        def digest(d):
            j = json.loads((d / "digests.json").read_text())
            j[1]["ok"] = False
            (d / "digests.json").write_text(json.dumps(j))
        self.assertEqual(self.rescore(digest)["digest_3x"]["verdict"], "FAIL")

        def probe_on_off(d):
            (d / "B1/decode_probe.jsonl").write_text("{}\n")
            j = json.loads((d / "B1/instrumentation.json").read_text())
            j["files"]["decode_probe"] = True
            (d / "B1/instrumentation.json").write_text(json.dumps(j))
        self.assertEqual(self.rescore(probe_on_off)["instrumentation_level"]["verdict"], "FAIL")


class TestAlphaLive(unittest.TestCase):
    def live(self, name, **kw):
        return FX.run("alpha", FX.tmp / name, PDMUX_GATE_ONLY_BOOTS="B1,B2", **kw)

    def test_host_time_mutant_fails(self):
        rc, res = self.live("m_host", FAKE_MODE="host_time")
        self.assertEqual(res["checks"]["i4_B2"]["verdict"], "FAIL")
        self.assertNotEqual(res["verdict"], "PASS")

    def test_on_boot_without_enabled_log_fails(self):
        rc, res = self.live("m_nolog", FAKE_MODE="no_enabled_log")
        self.assertEqual(res["checks"]["instrumentation_level"]["verdict"], "FAIL")

    def test_off_boot_writing_probe_file_fails(self):
        root = FX.tmp / "m_off"
        rc, res = FX.run("alpha", root, PDMUX_GATE_ONLY_BOOTS="B1", FAKE_MODE="off_writes_probe",
                         FAKE_STRAY_PATH=str(root / "B1" / "decode_probe.jsonl"))
        self.assertEqual(res["checks"]["instrumentation_level"]["verdict"], "FAIL")

    def test_wrong_sglang_import_fails(self):
        rc, res = self.live("m_sg", PDMUX_GATE_SGLANG_FILE="/usr/lib/python3/sglang/__init__.py")
        self.assertEqual(res["checks"]["instrumentation_level"]["verdict"], "FAIL")

    def test_unpadded_graph_hits_fail_i6(self):
        rc, res = self.live("m_pad", FAKE_MODE="no_pad")          # bs_padded := bs_real while the capture list says otherwise
        self.assertEqual(res["checks"]["i6_B2"]["verdict"], "FAIL")

    def test_server_dying_mid_boot_is_fail(self):
        rc, res = self.live("m_die", FAKE_MODE="die_after_stream")
        self.assertEqual(res["checks"]["survival_all_steps"]["verdict"], "FAIL")


class TestBeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = FX.tmp / "beta_base"
        cls.rc, cls.res = FX.run("beta", cls.root)

    def test_baseline_pass_with_expected_death(self):
        bad = {k: v for k, v in self.res["checks"].items() if v["verdict"] != "PASS"}
        self.assertEqual(self.res["verdict"], "PASS", bad)
        self.assertEqual(self.rc, 0)
        b1 = json.loads((self.root / "BETA2_boot1/beta2.json").read_text())
        b2 = json.loads((self.root / "BETA2_boot2_blocking/beta2.json").read_text())
        self.assertTrue(b1["died"] and b1["sig_illegal"] and b1["signature_ok"])
        self.assertFalse(b1["sig_conv"] or b1["traceback_conv_file"])        # G-beta round 1 reality: async crash names no kernel file
        self.assertTrue(b2["died"] and b2["traceback_conv_file"] and b2["signature_ok"])
        self.assertTrue(b1["pre_boundary_94500"]["ok"])
        self.assertEqual([d["ok"] for d in self.res["digests"]], [True, True, True])
        self.assertEqual(D.conv_state(FX.tree), "applied")

    def test_unit_case_count_and_isolation(self):
        n = len(list((self.root / "beta0/cases").glob("*.json")))
        self.assertEqual(n, len(D.beta0_cases()))
        self.assertTrue((self.root / "beta0/green_event_calib.json").exists())

    def test_pristine_survives_with_wrong_output_still_pass(self):
        rc, res = FX.run("beta", FX.tmp / "beta_surv", FAKE_MODE="pristine_survive")
        self.assertEqual(res["checks"]["beta2_BETA2_boot1"]["verdict"], "PASS")
        self.assertEqual(res["verdict"], "PASS")

    def test_pristine_identical_output_is_fail(self):
        rc, res = FX.run("beta", FX.tmp / "beta_same", FAKE_MODE="pristine_same")
        self.assertEqual(res["checks"]["beta2_BETA2_boot1"]["verdict"], "FAIL")
        self.assertEqual(res["verdict"], "FAIL")

    def test_inverted_death_expectation_fails(self):
        want = D.beta0_cases()
        inv = [dict(c, expect="MATCH" if c["expect"] == "BROKEN" else c["expect"]) for c in want]
        with mock.patch.object(D, "beta0_cases", lambda: inv):
            rc, res = FX.run("beta", FX.tmp / "beta_inv", PDMUX_GATE_ONLY="x")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "FAIL")
        self.assertEqual(res["verdict"], "FAIL")

    def test_patched_kernel_crash_is_fail(self):
        rc, res = FX.run("beta", FX.tmp / "beta_pc", FAKE_UNIT_MODE="patched_crash")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "FAIL")

    def test_green_calibration_fail_propagates(self):
        rc, res = FX.run("beta", FX.tmp / "beta_cal", FAKE_CALIB="FAIL")
        self.assertEqual(res["checks"]["beta0_green_event_calib"]["verdict"], "FAIL")
        self.assertEqual(res["verdict"], "FAIL")

    def test_digest_mismatch_after_reverse_aborts_and_restores_tree(self):
        rc, res = FX.run("beta", FX.tmp / "beta_dig", PDMUX_GATE_DIGEST_REV="1" * 64)
        self.assertEqual(rc, 13)
        self.assertEqual(D.conv_state(FX.tree), "applied")

    def test_case_classifier(self):
        self.assertEqual(D.classify_case(0, False), "OK")
        self.assertEqual(D.classify_case(1, False), "STATE_MISMATCH")
        self.assertEqual(D.classify_case(2, False), "ERROR")
        self.assertEqual(D.classify_case(-6, False), "CRASH")
        self.assertEqual(D.classify_case(None, True), "CRASH")

    def test_judge_case_table(self):
        ref = {"outcome": "OK", "out_sha256": "aa"}
        ok = lambda sha: {"outcome": "OK", "out_sha256": sha}
        m, b, r = dict(expect="MATCH"), dict(expect="BROKEN"), dict(expect="REF")
        J = D.judge_case
        self.assertEqual(J(m, ok("aa"), ref)[0], "PASS")
        self.assertEqual(J(m, ok("bb"), ref)[0], "FAIL")
        self.assertEqual(J(m, {"outcome": "CRASH"}, ref)[0], "FAIL")
        self.assertEqual(J(m, {"outcome": "STATE_MISMATCH", "out_sha256": "aa"}, ref)[0], "FAIL")
        self.assertEqual(J(m, ok("aa"), None)[0], "UNRESOLVED")                 # no reference -> never PASS
        self.assertEqual(J(m, ok("aa"), {"outcome": "CRASH"})[0], "UNRESOLVED")
        self.assertEqual(J(m, {"outcome": "ERROR"}, ref)[0], "UNRESOLVED")      # input error is a harness defect, not FAIL
        self.assertEqual(J(b, {"outcome": "CRASH"}, ref)[0], "PASS")
        self.assertEqual(J(b, {"outcome": "STATE_MISMATCH"}, ref)[0], "PASS")
        self.assertEqual(J(b, ok("bb"), ref)[0], "PASS")
        self.assertEqual(J(b, ok("aa"), ref)[0], "FAIL")                        # equal to the reference = not broken
        self.assertEqual(J(b, ok("bb"), None)[0], "UNRESOLVED")
        self.assertEqual(J(r, ok("aa"), None)[0], "PASS")
        self.assertEqual(J(r, {"outcome": "CRASH"}, None)[0], "UNRESOLVED")

    def test_case_geometry_and_refs(self):
        cases = {c["name"]: c for c in D.beta0_cases()}
        self.assertEqual(len(cases), len(D.beta0_cases()))
        for c in cases.values():
            self.assertLessEqual(c["offset"] + c["dim"], c["width"], c["name"])          # round 1: every w11328 case was rc=2
            if c["ref"]:
                self.assertIn(c["ref"], cases, c["name"])
                self.assertEqual(cases[c["ref"]]["dim"], c["dim"])
                self.assertEqual(cases[c["ref"]]["T"], c["T"])
                self.assertEqual(cases[c["ref"]]["layout"], "contig")
                self.assertEqual(cases[c["ref"]]["expect"] == "REF" or cases[c["ref"]]["name"].startswith("patched_contig"), True)
            else:
                self.assertEqual(c["expect"], "REF")
        tp2 = [c for c in cases.values() if c["width"] == 11328]
        self.assertTrue(tp2 and all((c["dim"], c["offset"]) == (6144, 5120) for c in tp2))
        order = list(cases)
        for c in cases.values():
            if c["ref"]:
                self.assertLess(order.index(c["ref"]), order.index(c["name"]))           # references run first
        self.assertEqual(cases["co_only_view_w22656_T94790_r0"]["ref"], "patched_contig_w22656_T94790_r0")
        # int32 boundary arithmetic of the registered predictions
        for w, d, edge in ((22656, 12288, 94786), (11328, 6144, 189573)):
            self.assertLessEqual((edge - 1) * w + d - 1, 2 ** 31 - 1)
            self.assertGreater(edge * w + d - 1, 2 ** 31 - 1)

    def test_no_input_errors_in_baseline(self):
        for f in (self.root / "beta0/cases").glob("*.json"):
            self.assertNotEqual(json.loads(f.read_text())["outcome"], "ERROR", f.name)

    def test_descriptive_atol_noise_never_fails(self):
        d = json.loads((self.root / "beta0/cases/patched_view_w22656_T95100_r0.json").read_text())
        self.assertGreater(d["max_abs_diff"], 0.02)          # fake reports 0.05 like round 1; the case still passed
        self.assertTrue(d["case_ok"])

    def test_wrong_patched_output_is_fail(self):
        rc, res = FX.run("beta", FX.tmp / "beta_pw", FAKE_UNIT_MODE="patched_wrong")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "FAIL")
        self.assertIn("patched_view_w22656_T95100_r0", res["checks"]["beta0_cases"]["bad"])

    def test_python_exception_rc1_is_crash_not_state_mismatch(self):
        rc, res = FX.run("beta", FX.tmp / "beta_exc", FAKE_UNIT_MODE="py_exception")
        d = json.loads((FX.tmp / "beta_exc/beta0/cases/patched_view_w22656_T95100_r0.json").read_text())
        self.assertEqual(d["outcome"], "CRASH")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "FAIL")

    def test_pristine_wrong_output_instead_of_crash_is_broken(self):
        rc, res = FX.run("beta", FX.tmp / "beta_pm", FAKE_BROKEN="mismatch")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "PASS")

    def test_chunk_offset_mutant_verdicts(self):
        rc, res = FX.run("beta", FX.tmp / "beta_cos", FAKE_UNIT_MODE="co_state")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "PASS")                # state mismatch = BROKEN
        rc, res = FX.run("beta", FX.tmp / "beta_cosame", FAKE_UNIT_MODE="co_same")
        self.assertEqual(res["checks"]["beta0_cases"]["verdict"], "FAIL")                # not broken -> the mutant test is vacuous
        self.assertIn("co_only_view_w22656_T94790_r0", res["checks"]["beta0_cases"]["bad"])

    def _rescore_beta(self, name, mutate_case=None, mutate_b2=None):
        d = FX.tmp / name
        shutil.copytree(self.root, d, dirs_exist_ok=True)
        if mutate_case:
            f = d / "beta0/cases" / mutate_case[0]
            j = json.loads(f.read_text()); mutate_case[1](j); f.write_text(json.dumps(j))
        if mutate_b2:
            f = d / mutate_b2[0] / "beta2.json"
            j = json.loads(f.read_text()); mutate_b2[1](j); f.write_text(json.dumps(j))
        return D.score_beta(d)

    def test_rescore_ignores_stored_case_ok(self):
        # stored case_ok says True, but the sha no longer equals the reference: the scorer must re-judge from the files
        r = self._rescore_beta("beta_rs1", ("patched_view_w22656_T94800_r1.json", lambda j: j.update(out_sha256="0" * 64)))
        self.assertEqual(r["beta0_cases"]["verdict"], "FAIL")
        r = self._rescore_beta("beta_rs2", ("pristine_contig_w22656_T94800_r0.json", lambda j: j.update(outcome="CRASH")))
        self.assertNotEqual(r["beta0_cases"]["verdict"], "PASS")                         # reference lost -> never PASS

    def test_nonblocking_signature_needs_death_and_illegal_only(self):
        self.assertTrue(D.beta2_signature_ok({"died": True, "sig_illegal": True}, False))
        self.assertFalse(D.beta2_signature_ok({"died": True, "sig_illegal": False, "sig_conv": True}, False))
        self.assertFalse(D.beta2_signature_ok({"died": True, "sig_illegal": True, "sig_conv": True}, True))   # blocking needs the file
        self.assertTrue(D.beta2_signature_ok({"died": True, "sig_illegal": True, "sig_conv": True, "traceback_conv_file": True}, True))
        self.assertIsNone(D.beta2_signature_ok({"died": False}, True))
        r = self._rescore_beta("beta_rs3", mutate_b2=("BETA2_boot1", lambda j: j.update(sig_illegal=False)))
        self.assertEqual(r["beta2_BETA2_boot1"]["verdict"], "FAIL")
        r = self._rescore_beta("beta_rs4", mutate_b2=("BETA2_boot2_blocking", lambda j: j.update(died=True, sig_illegal=True, sig_conv=True, traceback_conv_file=False)))
        self.assertEqual(r["beta2_BETA2_boot2_blocking"]["verdict"], "FAIL")


class TestGreenCalibScorer(unittest.TestCase):
    CAL = _load("green_calib_t", ENG / "scripts/vessl/green_event_calib_unit.py")

    def cells(self, **kw):
        c = {"green": {"pre": 9.6, "post": 4.4, "ref": 4.45}, "default": {"pre": 9.1, "post": 4.0, "ref": 4.02}}
        for k, v in kw.items():
            s, f = k.split("__")
            c[s][f] = v
        return c

    def test_registered_predictions(self):
        self.assertEqual(self.CAL.score(self.cells())["verdict"], "PASS")
        self.assertEqual(self.CAL.score(self.cells(green__pre=4.5))["verdict"], "FAIL")      # sleep before not included
        self.assertEqual(self.CAL.score(self.cells(default__post=9.0))["verdict"], "FAIL")   # sleep after included
        self.assertEqual(self.CAL.score(self.cells(green__post=4.6))["verdict"], "FAIL")     # >2 % off the reference
        self.assertEqual(self.CAL.score({"default": self.cells()["default"]})["verdict"], "UNRESOLVED")
        self.assertIn("USER DECISION", self.CAL.score(self.cells(green__pre=4.5))["on_fail"])


class TestGamma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = FX.tmp / "gamma_base"
        cls.rc, cls.res = FX.run("gamma", cls.root, family="F")

    def test_baseline_pass(self):
        self.assertEqual(self.res["verdict"], "PASS", self.res["checks"])
        self.assertEqual(self.rc, 0)
        self.assertEqual(self.res["checks"]["family"]["n_valid"], 10)
        self.assertEqual([d["tag"] for d in self.res["digests"]], ["start", "middle", "end"])

    def test_abba_order(self):
        order = [o for _p, _s, o in D.gamma_pairs_plan("main")]
        self.assertEqual(order[0], ("on", "off"))
        self.assertEqual(order[1], ("off", "on"))

    def rescore(self, mutate, phase="main"):
        d = Path(tempfile.mkdtemp(dir=FX.tmp))
        shutil.copytree(self.root, d, dirs_exist_ok=True)
        for f in d.glob("pair*_*/instance.json"):
            j = json.loads(f.read_text())
            mutate(j)
            f.write_text(json.dumps(j))
        return D.score_gamma(d, phase)

    def test_observer_shift_is_observer_effect(self):
        def slow(j):
            if j["level"] == "on":
                j["hi"]["achieved"] *= 0.95
        r = self.rescore(slow)
        self.assertEqual(r["tost_lambda_star_thr"]["verdict"], "OBSERVER_EFFECT")
        self.assertEqual(D.verdict_of("gamma", r), "OBSERVER_EFFECT")

    def test_noise_is_unresolved_power_not_effect(self):
        import random
        rnd = random.Random(9)

        def noisy(j):
            if j["level"] == "on":
                j["hi"]["achieved"] *= 1 + rnd.gauss(0, 0.04)
        r = self.rescore(noisy)
        self.assertEqual(r["tost_lambda_star_thr"]["verdict"], "UNRESOLVED_POWER", r["tost_lambda_star_thr"])
        self.assertEqual(D.verdict_of("gamma", r), "UNRESOLVED_POWER")

    def test_oo_pairs_not_in_default_plan(self):
        c = self.res["checks"]
        self.assertEqual(D.gamma_oo_plan("main"), [])
        self.assertEqual(D.gamma_oo_plan("ext"), [])
        self.assertFalse(list(self.root.glob("pair[12][0-9][0-9]_*")))
        self.assertEqual(c["diag_off_off"]["n_valid_oo"], 0)
        self.assertEqual(c["family"]["n_pairs"], 10)
        self.assertEqual(c["family"]["n_valid"], 10)
        self.assertEqual(c["diag_stall_boundary"]["verdict"], "DIAGNOSTIC")
        self.assertEqual(len(list(self.root.glob("pair*_o[nf]*/instance.json"))), 20)

    def test_oo_option_runs_and_is_inert(self):
        with mock.patch.dict(os.environ, {"PDMUX_GATE_GAMMA_OO_PAIRS": "2"}):
            self.assertEqual([p for p, _s, _o in D.gamma_oo_plan("main")], [101, 102])
            self.assertEqual([p for p, _s, _o in D.gamma_oo_plan("ext")], [201, 202])
        rc, res = FX.run("gamma", FX.tmp / "g_oo", family="F", PDMUX_GATE_GAMMA_OO_PAIRS="2")
        self.assertEqual(res["verdict"], "PASS", res["checks"])
        self.assertEqual(res["checks"]["diag_off_off"]["n_valid_oo"], 2)
        self.assertEqual(res["checks"]["family"]["n_pairs"], 10)                        # OO pairs never enter the TOST

    def test_stall_boundary_label_does_not_touch_run_gate_rollup(self):
        # whole run_gate path (not rescore): every LO cell sits at ~5 % stall -> STALL_BOUNDARY on every cell
        rc, res = FX.run("gamma", FX.tmp / "g_stall", family="F", FAKE_STALL="1")
        self.assertEqual(res["verdict"], "PASS", res["checks"])
        self.assertEqual(rc, 0)
        st = res["checks"]["diag_stall_boundary"]
        self.assertGreater(st["n_boundary"], 0)
        self.assertEqual(st["n_boundary"], len([c for c in st["cells"]]))
        self.assertFalse([k for k, v in res["stages"].items() if v["status"] in ("SKIPPED", "UNRESOLVED", "DIAG_SKIPPED")])

    @staticmethod
    def _fits_after(n_pairs):
        """Gate.fits stub: the per-pair budget check (secs == 2*EST['inst']) succeeds n_pairs times, then fails."""
        calls = {"n": 0}
        real = D.Gate.fits

        def fits(self, secs):
            if secs == 2 * D.EST["inst"]:
                calls["n"] += 1
                return calls["n"] <= n_pairs
            return real(self, secs)
        return fits

    def test_unrun_oo_pair_is_diag_skipped_not_unresolved(self):
        with mock.patch.object(D.Gate, "fits", self._fits_after(10)):
            rc, res = FX.run("gamma", FX.tmp / "g_oo_skip", family="F", PDMUX_GATE_GAMMA_OO_PAIRS="1")
        self.assertEqual(res["stages"]["pair101"]["status"], "DIAG_SKIPPED", res["stages"].get("pair101"))
        self.assertEqual(res["verdict"], "PASS", res["checks"])
        self.assertEqual(rc, 0)

    def test_skipped_on_off_pair_still_lowers_verdict(self):
        with mock.patch.object(D.Gate, "fits", self._fits_after(9)):
            rc, res = FX.run("gamma", FX.tmp / "g_pair_skip", family="F")
        self.assertEqual(res["stages"]["pair10"]["status"], "SKIPPED")
        self.assertNotEqual(res["verdict"], "PASS")

    def test_diagnostics_never_change_verdict(self):
        def junk(j):                                                # wreck OO pairs and stall labels
            if j["pair"] >= 100:
                j["hi"]["achieved"] *= 0.5 if j["level"] == "offb" else 1
            j["lo"]["stall"] = {"stall_frac": 0.05, "label": "STALL_BOUNDARY", "n": 10}
        r = self.rescore(junk)
        self.assertEqual(D.verdict_of("gamma", r), "PASS")
        self.assertGreater(r["diag_stall_boundary"]["n_boundary"], 0)

    def test_asymmetric_invalid_pairs_are_observer_effect(self):
        def brk(j):
            if j["level"] == "on" and j["pair"] in (1, 2):
                j["valid"] = False
        r = self.rescore(brk)
        self.assertEqual(r["family"]["verdict"], "OBSERVER_EFFECT")
        self.assertEqual(r["invalid_pair_asymmetry"]["invalid_on"], 2)

    def test_one_asymmetric_invalid_pair_is_tolerated(self):
        def brk(j):
            if j["level"] == "on" and j["pair"] == 1:
                j["valid"] = False
        self.assertEqual(self.rescore(brk)["family"]["verdict"], "PASS")

    def test_too_few_valid_pairs_unresolved(self):
        def brk(j):
            if j["pair"] <= 3:
                j["valid"] = False
        r = self.rescore(brk)
        self.assertEqual(r["family"]["verdict"], "UNRESOLVED")

    def test_digest_mutation_fails_job(self):
        r = self.rescore(lambda j: None)
        r["digest_3x"] = {"verdict": "FAIL"}
        self.assertEqual(D.verdict_of("gamma", r), "FAIL")

    def test_live_slow_observer_detected(self):
        rc, res = FX.run("gamma", FX.tmp / "gamma_slow", family="A", FAKE_ON_SLOW="0.06", PDMUX_GATE_PAIRS="8")
        self.assertEqual(res["verdict"], "OBSERVER_EFFECT", res["checks"].get("family"))
        self.assertEqual(rc, 1)

    def test_extension_pooling_uses_extended_look(self):
        d = FX.tmp / "g_pool"
        shutil.copytree(self.root, d)
        ext = FX.tmp / "g_ext"
        ext.mkdir()
        for p in range(11, 17):
            for lvl in ("on", "off"):
                src = json.loads((self.root / f"pair0{p - 10}_{lvl}/instance.json").read_text())
                src["pair"] = p
                (ext / f"pair{p}_{lvl}").mkdir()
                (ext / f"pair{p}_{lvl}/instance.json").write_text(json.dumps(src))
        r = D.score_gamma(d, "pooled", extra_dirs=[ext])
        self.assertEqual(r["family"]["look"], "extended")
        self.assertEqual(r["family"]["n_pairs"], 16)

    def test_real_loadgen_integration_cell(self):
        """mg_loadgen.py against the fake server's SSE: the cell pipeline (trace -> loadgen -> analyze_cell) runs end to end."""
        root = FX.tmp / "g_real"
        env = FX.env(root, PDMUX_GATE_LOADGEN_CMD="", PDMUX_GATE_HI_N="24", PDMUX_GATE_HI_RATE="24", FAKE_TOK_S="0.0005")
        with mock.patch.dict(os.environ, env):
            gate = D.Gate("gamma", root, 120, "F")
            srv = D.Server(gate, "pairX_on", dict(arm="fused_sym", on=True, ctx=None), "patched")
            self.assertEqual(srv.start(30), "ok")
            try:
                cell = D.run_cell(srv, srv.dir / "cell_HI", "HI", 4386)
            finally:
                srv.stop()
        self.assertEqual(cell["rc"], 0)
        self.assertEqual(cell["n"], 24)
        self.assertIsNotNone(cell["itl_p95_ms"])


if __name__ == "__main__":
    unittest.main()
