"""CPU tests for scripts/vessl/engine_tree.py (pristine vs patched SGLang tree per boot, 2026-11-01).

Two layers:
  * fake trees: preserve / verify / plan_boot fail-fast rules, each with a mutation that must flip it;
  * the real v0.5.10 tree (skipped unless the local SGLang checkout with tag v0.5.10 exists): the committed
    reference equals `git archive v0.5.10 python`, upstream carries no project marker, an image-like tree
    (+ setuptools_scm _version.py, egg-info, __pycache__) verifies, and the tree produced by the REAL
    install_runtime.sh fails verification with markers found.
"""
import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
sys.path.insert(0, str(ENGINE / "scripts" / "vessl"))
import engine_tree as et  # noqa: E402

REF = ENGINE / "env" / "sglang_v0.5.10_python_tree.sha256"
SGLANG_GIT = Path(os.environ.get("SGLANG_SRC", REPO.parent / "sglang_engine_dev"))


def _has_tag():
    if not (SGLANG_GIT / ".git").exists() or shutil.which("git") is None:
        return False
    r = subprocess.run(["git", "-C", str(SGLANG_GIT), "rev-parse", "-q", "--verify", "v0.5.10^{commit}"],
                       capture_output=True)
    return r.returncode == 0


def _fake_tree(py: Path):
    (py / "sglang" / "srt").mkdir(parents=True)
    (py / "sglang" / "__init__.py").write_text("X = 0\n")
    (py / "sglang" / "srt" / "x.py").write_text("_PDMUX_PREFILL_TP_GROUP = None  # upstream-style name\n")
    os.symlink("x.py", py / "sglang" / "srt" / "link.py")


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.td, True)
        self.opt = self.td / "opt"
        self.img = self.opt / "sglang_engine_dev" / "python"
        _fake_tree(self.img)
        self.ref = self.td / "ref.sha256"
        self.ref.write_text(et.render(et.scan(self.img)))
        (self.img / "sglang" / "_version.py").write_text("v = '0.0.0.dev0'\n")     # allowed generated extra
        (self.img / "sglang.egg-info").mkdir()
        (self.img / "sglang.egg-info" / "PKG-INFO").write_text("x\n")             # excluded cache
        (self.img / "sglang" / "__pycache__").mkdir()
        (self.img / "sglang" / "__pycache__" / "a.pyc").write_bytes(b"\0")

    def preserve(self, src=None, ref=None, overridden=True):
        argv = ["preserve", "--src", str(src or self.img), "--dst", str(self.opt / "sglang_pristine"),
                "--ref", str(ref or self.ref), "--meta", str(self.td / "meta")]
        with contextlib.redirect_stdout(io.StringIO()):
            return et.main(argv + (["--ref-overridden"] if overridden else []))

    def status(self):
        return et._kv_read(self.opt / "sglang_pristine" / "PRESERVE_STATUS")

    def env(self, **kw):
        e = {"PATH": os.environ.get("PATH", ""), "PDMUX_ROOT": str(self.opt), "SGLANG_ENGINE_DEV": str(self.img),
             "PDMUX_COMMIT": "c", "PDMUX_JOB_NAME": "j", "PYTHONPATH": str(self.img)}
        e.update(kw)
        return {k: v for k, v in e.items() if v is not None}

    def patch_image(self):
        with open(self.img / "sglang" / "srt" / "x.py", "a") as f:
            f.write("def pdmux_role_is_thread_local(): pass\n# maybe_create_holb_probe maybe_install_chunk_probe\n")


class ManifestTest(_Base):
    def test_scan_excludes_caches_and_hashes_symlink_target(self):
        m = et.scan(self.img)
        self.assertNotIn("sglang.egg-info/PKG-INFO", m)
        self.assertFalse(any("__pycache__" in k for k in m))
        self.assertEqual(m["sglang/srt/link.py"], hashlib.sha256(b"symlink:x.py").hexdigest())

    def test_render_parse_roundtrip(self):
        m = et.scan(self.img)
        self.assertEqual(et.parse_manifest(et.render(m)), m)

    def test_upstream_style_name_is_not_a_marker(self):
        self.assertEqual(et.find_markers(self.img, et.scan(self.img)), {})


class PreserveTest(_Base):
    def test_ok_copy_and_records(self):
        self.assertEqual(self.preserve(), 0)
        st = self.status()
        self.assertEqual((st["status"], st["origin"], st["extras"]), ("OK", "copied", "sglang/_version.py"))
        self.assertEqual(st["tree_manifest_sha256"], hashlib.sha256(self.ref.read_bytes()).hexdigest())
        self.assertTrue((self.td / "meta" / "engine_tree_pristine_status.txt").exists())
        self.assertTrue((self.opt / "sglang_pristine" / "python" / "sglang" / "srt" / "link.py").is_symlink())

    def test_rerun_keeps_verified_copy(self):
        self.preserve()
        self.patch_image()                               # Workspace: install ran after the first preserve
        self.assertEqual(self.preserve(), 0)
        self.assertEqual(self.status()["origin"], "existing")

    def test_patched_source_fails(self):
        self.patch_image()
        self.assertEqual(self.preserve(), 5)
        self.assertIn("not pristine", self.status()["reason"])
        self.assertFalse((self.opt / "sglang_pristine" / "python").exists())

    def test_unknown_extra_fails_but_version_py_is_allowed(self):
        (self.img / "sglang" / "srt" / "new.py").write_text("y = 1\n")
        self.assertEqual(self.preserve(), 5)
        (self.img / "sglang" / "srt" / "new.py").unlink()
        self.assertEqual(self.preserve(), 0)

    def test_missing_src(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = et.main(["preserve", "--src", "", "--dst", str(self.opt / "sglang_pristine"), "--ref", str(self.ref)])
        self.assertEqual(rc, 5)
        self.assertIn("SGLANG_ENGINE_DEV unset", self.status()["reason"])


class PlanBootTest(_Base):
    CMD = [sys.executable, "-c", "pass"]

    def test_patched_default_passes_env_through(self):
        self.patch_image()
        env = self.env()
        child, stamp = et.plan_boot(env, self.CMD)
        self.assertEqual(child, env)
        self.assertEqual(stamp["engine_tree"], "patched")
        self.assertIn("pdmux_role_is_thread_local", stamp["engine_tree_markers_found"])

    def test_patched_without_markers_refuses(self):
        with self.assertRaisesRegex(et.TreeError, "lacks project markers"):
            et.plan_boot(self.env(), self.CMD)

    def test_patched_resolving_elsewhere_refuses(self):
        self.patch_image()
        other = self.td / "other"
        _fake_tree(other)
        with self.assertRaisesRegex(et.TreeError, "resolves to"):
            et.plan_boot(self.env(PYTHONPATH=str(other)), self.CMD)

    def test_pristine_ok_scrubs_and_switches_path(self):
        self.preserve()
        self.patch_image()
        child, stamp = et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine"), self.CMD)
        self.assertEqual([k for k in child if k.startswith("PDMUX_")], [])
        self.assertEqual(child["PYTHONPATH"], str(self.opt / "sglang_pristine" / "python"))
        self.assertEqual(stamp["pythonpath_dropped"], [str(self.img)])
        self.assertEqual(stamp["engine_tree_manifest_sha256"], hashlib.sha256(self.ref.read_bytes()).hexdigest())
        self.assertTrue(stamp["engine_tree_ref_overridden"])
        self.assertIn("PDMUX_COMMIT", stamp["pdmux_env_scrubbed"])

    def test_pristine_refuses_engine_knob_unless_declared(self):
        self.preserve()
        with self.assertRaisesRegex(et.TreeError, "PDMUX_TELEMETRY_PATH"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine", PDMUX_TELEMETRY_PATH="/x"), self.CMD)
        child, _ = et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine", PDMUX_GATE_FAMILY="a"), self.CMD,
                                allow_env=["PDMUX_GATE_FAMILY"])
        self.assertNotIn("PDMUX_GATE_FAMILY", child)

    def test_pristine_refuses_pdmux_args(self):
        self.preserve()
        for a in ("--enable-pdmux", "--pdmux-config-path=/c.yml"):
            with self.assertRaisesRegex(et.TreeError, "PD-mux argument"):
                et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine"), self.CMD + [a])

    def test_pristine_tampered_copy_refuses(self):
        self.preserve()
        p = self.opt / "sglang_pristine" / "python" / "sglang" / "__init__.py"
        p.write_text("X = 1\n")
        with self.assertRaisesRegex(et.TreeError, "changed since preservation"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine"), self.CMD)

    def test_pristine_without_preserve_or_failed_preserve_refuses(self):
        with self.assertRaisesRegex(et.TreeError, "missing"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine"), self.CMD)
        self.patch_image()
        self.preserve()
        with self.assertRaisesRegex(et.TreeError, "not available"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine"), self.CMD)

    def test_pristine_ref_must_be_the_commits_unless_overridden(self):
        self.preserve(overridden=False)                  # verified against a non-committed ref, not flagged
        proj = self.td / "proj"
        (proj / Path(et.REF_REL).parent).mkdir(parents=True)
        (proj / et.REF_REL).write_text("0" * 64 + "  other\n")
        with self.assertRaisesRegex(et.TreeError, "different reference"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine", PDMUX_PROJECT_ROOT=str(proj)), self.CMD)
        shutil.copyfile(self.ref, proj / et.REF_REL)     # same bytes -> accepted, not overridden
        _, stamp = et.plan_boot(self.env(PDMUX_ENGINE_TREE="pristine", PDMUX_PROJECT_ROOT=str(proj)), self.CMD)
        self.assertFalse(stamp["engine_tree_ref_overridden"])

    def test_bad_tree_value(self):
        with self.assertRaisesRegex(et.TreeError, "expected one of"):
            et.plan_boot(self.env(PDMUX_ENGINE_TREE="stock"), self.CMD)


@unittest.skipUnless(_has_tag(), "local SGLang checkout with tag v0.5.10 not found")
class RealTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = Path(tempfile.mkdtemp())
        arch = subprocess.run(["git", "-C", str(SGLANG_GIT), "archive", "v0.5.10", "python"],
                              capture_output=True, check=True).stdout
        for n in ("ref", "img"):
            (cls.td / n).mkdir()
            subprocess.run(["tar", "-x", "-C", str(cls.td / n)], input=arch, check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, True)

    def test_committed_reference_is_git_archive_v0510(self):
        self.assertEqual(et.render(et.scan(self.td / "ref" / "python")), REF.read_text())
        src = et._kv_read(REF.with_suffix(".source"))
        self.assertEqual(src["manifest_sha256"], hashlib.sha256(REF.read_bytes()).hexdigest())

    def test_upstream_has_no_marker(self):
        root = self.td / "ref" / "python"
        self.assertEqual(et.find_markers(root, et.scan(root)), {})

    def test_image_like_tree_verifies_and_real_install_fails(self):
        py = self.td / "img" / "python"
        (py / "sglang" / "_version.py").write_text("__version__ = '0.0.0.dev0'\n")
        (py / "sglang.egg-info").mkdir(exist_ok=True)
        (py / "sglang.egg-info" / "PKG-INFO").write_text("Name: sglang\n")
        res = et.verify_pristine(py, REF.read_text())
        self.assertTrue(res["ok"], res["reasons"])
        if shutil.which("patch") is None or shutil.which("flock") is None:
            self.skipTest("patch/flock missing")
        env = dict(os.environ, SGLANG_ENGINE_DEV=str(py), PDMUX_PROJECT_ROOT=str(REPO))
        env.pop("PDMUX_SYNC_LOCK_HELD", None)
        r = subprocess.run(["bash", str(ENGINE / "scripts" / "vessl" / "install_runtime.sh"),
                            str(self.td / "m.sha256")], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        res = et.verify_pristine(py, REF.read_text())
        self.assertFalse(res["ok"])
        found = {m for v in res["markers"].values() for m in v}
        self.assertTrue(set(et.REQUIRED_PATCHED_MARKERS) <= found, found)


if __name__ == "__main__":
    unittest.main()
