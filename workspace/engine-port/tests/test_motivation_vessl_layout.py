"""Entry scripts of the motivation harness import in the VESSL Job layout (results/ = symlink into the run volume, scripts/ only in the checkout).

2026-11-12: two Jobs died before the first boot with ModuleNotFoundError (analytic_terms, mg_x_pretokenize) because the script directory
Python puts on sys.path, and Path.resolve(), follow the symlink to a tree that has no scripts/.  Local runs have no symlink, so no other
test saw it.  This test rebuilds that layout from the tracked .py files and imports every entry script through the symlinked path.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(os.path.abspath(__file__)).parents[1]
ENTRIES = ("mo_runner.py", "mo_analyze.py", "mo_cap_runner.py", "mo_cap_analyze.py", "mo_intf_runner.py", "mo_intf_analyze.py",
           "mo_ladder.py", "mo_ladder_analyze.py")


class TestVesslLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mo_layout_"))
        eng, run = cls.tmp / "checkout" / "workspace" / "engine-port", cls.tmp / "run" / "results"
        eng.mkdir(parents=True)
        for d in (x for x in ENG.iterdir() if x.is_dir() and x.name not in ("results", "tests") and not x.name.startswith(".")):
            os.symlink(d, eng / d.name)                      # scripts/, benchmarks/, ...: in the checkout only
        files = subprocess.run(["git", "ls-files", "results"], cwd=ENG, capture_output=True, text=True, check=True).stdout.split("\n")
        for rel in (f for f in files if f.endswith((".py", ".json")) and (ENG / f).is_file()):
            dst = run.parent / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ENG / rel, dst)
        os.symlink(run, eng / "results")                       # job_entry.sh: ln -s ${RUN}/results .../workspace/engine-port/results
        cls.mot = eng / "results" / "motivation"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_entry_script_imports_through_the_symlink(self):
        for name in ENTRIES:
            with self.subTest(entry=name):
                r = subprocess.run([sys.executable, str(self.mot / name), "--help"], capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, f"{name}: {r.stderr[-600:]}")

    def test_the_layout_fails_without_the_path_fix(self):
        """The layout is a real reproduction: a script that imports a sealed spec with only Python's own script dir fails in it."""
        probe = self.mot / "_probe_layout.py"
        probe.write_text("import mo_spec\n")
        r = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ModuleNotFoundError", r.stderr)


if __name__ == "__main__":
    unittest.main()
