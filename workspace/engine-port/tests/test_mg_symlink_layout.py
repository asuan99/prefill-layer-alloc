"""Job-layout regression for the multi-GPU harness (lesson 270 recurrence, M2g Job pdmux-multigpu-cc181b4-20261012080743).

Inside a VESSL Job `workspace/engine-port/results` is a SYMLINK to /data/runs/<job>/results (job_entry.sh:113-117).
Deriving the engine root with `Path(__file__).resolve()` from a file under results/ therefore lands in /data/runs/<job>
and loses `scripts/` and `benchmarks/` (that Job died with ModuleNotFoundError: m2g_correctness_gate, and G3/G5(a) could
not find their scripts).  This test rebuilds that layout in a temp dir and checks every harness module that derives a
path from __file__ still points at the repo tree.  Mutation check: putting `.resolve()` back makes it fail.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(os.path.abspath(__file__)).parents[1]

PROBE = r"""
import importlib.util, os, sys
from pathlib import Path
path, attr = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("m", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
v = getattr(m, attr)
print(Path(v))
"""

# (file relative to engine-port, attribute holding a path, how many parents up to engine-port from that attribute)
MODULES = [
    ("results/multigpu/mg_drive.py", "ENGINE", 0),
    ("results/multigpu/mg_plan.py", "HERE", 2),
    ("results/multigpu/mg_label.py", "HERE", 2),
    ("results/multigpu/mg_realized_mix.py", "ENG", 0),
]


class JobSymlinkLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mg_layout_"))
        self.ep = self.tmp / "opt" / "workspace" / "engine-port"
        self.ep.mkdir(parents=True)
        for name in ("scripts", "benchmarks", "src", "env", "triage"):
            if (ENG / name).exists():
                (self.ep / name).symlink_to(ENG / name)
        runs = self.tmp / "data" / "runs" / "job" / "results"
        runs.mkdir(parents=True)
        shutil.copytree(ENG / "results" / "multigpu", runs / "multigpu",
                        ignore=shutil.ignore_patterns("m2g_*", "m3_pdmux*", "vessl", "__pycache__"))
        for extra in ("r2_correctness", "e2_sticky_prereg"):
            if (ENG / "results" / extra).exists():
                (runs / extra).symlink_to(ENG / "results" / extra)
        (self.ep / "results").symlink_to(runs)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _root_seen_by(self, rel, attr, up):
        out = subprocess.run([sys.executable, "-c", PROBE, str(self.ep / rel), attr],
                             capture_output=True, text=True, cwd=self.ep, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        p = Path(out.stdout.strip().splitlines()[-1])
        for _ in range(up):
            p = p.parent
        return p

    def test_engine_root_stays_in_repo_tree(self):
        for rel, attr, up in MODULES:
            with self.subTest(module=rel):
                root = self._root_seen_by(rel, attr, up)
                self.assertEqual(os.path.normpath(str(root)), os.path.normpath(str(self.ep)))
                self.assertTrue((root / "scripts" / "multigpu" / "m2g_correctness_gate.py").exists())
                self.assertTrue((root / "scripts" / "multigpu" / "g5a_selector_equiv.py").exists())

    def test_no_resolve_on_file_paths(self):
        # static guard: no harness module derives a path from __file__ via .resolve()
        files = list((ENG / "results" / "multigpu").glob("*.py")) + list((ENG / "scripts" / "multigpu").glob("*.py")) \
            + [ENG / "scripts" / "bootstrap" / "engine_digest.py"]
        bad = [str(f) for f in files if "Path(__file__).resolve()" in f.read_text()]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
