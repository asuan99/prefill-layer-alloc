"""Job-layout regression for the multi-GPU harness (lesson 270 recurrence, M2g Job pdmux-multigpu-cc181b4-20261012080743).

Inside a VESSL Job `workspace/engine-port/results` is a SYMLINK to /data/runs/<job>/results (job_entry.sh:113-117).
Deriving the engine root with `Path(__file__).resolve()` from a file under results/ therefore lands in /data/runs/<job>
and loses `scripts/` and `benchmarks/` (that Job died with ModuleNotFoundError: m2g_correctness_gate, and G3/G5(a) could
not find their scripts).  This test rebuilds that layout in a temp dir and checks every harness module that derives a
path from __file__ still points at the repo tree.  Mutation check: putting `.resolve()` back makes it fail.
"""
from __future__ import annotations

import os
import re
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
    ("results/multigpu/mg_drive.py", "HERE", 2),            # rev11 R-2: the rev10 audit found these two covered by the literal guard only
    ("results/multigpu/mg_substrate.py", "HERE", 2),
    ("results/multigpu/mg_substrate.py", "ENG", 0),
]

# rev11 R-2: any path-canonicalising call in a harness module breaks the Job layout (results/ is a symlink into /data/runs/<job>)
CANON = re.compile(r"\bresolve\s*\(|\brealpath\b")


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

    def _harness_files(self):
        return list((ENG / "results" / "multigpu").glob("*.py")) + list((ENG / "scripts" / "multigpu").glob("*.py")) \
            + [ENG / "scripts" / "bootstrap" / "engine_digest.py"]

    def test_no_resolve_on_file_paths(self):
        # static guard (regex, not the literal "Path(__file__).resolve()"): no harness module calls resolve( or realpath at all
        bad = [f"{f}:{m.group(0)}" for f in self._harness_files() for m in CANON.finditer(f.read_text())]
        self.assertEqual(bad, [])

    def test_static_guard_regex_catches_the_variants_the_literal_guard_missed(self):
        for variant in ("Path(__file__).resolve()", "p = Path(x)\nROOT = p.resolve().parents[1]", "os.path.realpath(__file__)",
                        "from os.path import realpath", "Path( __file__ ).resolve ()"):
            self.assertTrue(CANON.search(variant), variant)
        for fine in ("Path(os.path.abspath(__file__)).parent", "# resolved at import"):
            self.assertIsNone(CANON.search(fine), fine)

    def test_this_test_is_in_the_registered_digest_list(self):
        # rev11 R-2: the registered set (mg_job_common.sh MANDATORY_REG) must contain this regression test
        src = (ENG / "results" / "multigpu" / "mg_job_common.sh").read_text()
        self.assertIn("test_mg_symlink_layout.py", src[src.index("MANDATORY_REG=("):src.index("OPTIONAL_REG=")])
        self.assertTrue((ENG / "results" / "multigpu" / "mg_substrate.py").exists())


if __name__ == "__main__":
    unittest.main()
