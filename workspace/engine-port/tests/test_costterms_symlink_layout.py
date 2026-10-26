"""Job layout regression: job_entry.sh replaces workspace/engine-port/results with a symlink to the persistent volume
(`ln -s "${RUN}/results" "${PROJECT}/workspace/engine-port/results"`).  J-A1 (job-bnkxoji00zmy, 2026-10-26) aborted with
ABORT_HARNESS_DIGEST because the harness used Path(__file__).resolve(), which follows that symlink and loses the
workspace/engine-port root (external files ABSENT, spec/plans sha ERROR).  The harness must navigate by logical path."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
HARNESS_REL = Path("results/cap_char/costterms_harness")
EXTERNAL_DIRS = ("scripts/costmodel", "scripts/compo", "scripts/bootstrap", "scripts/vessl")


class TestSymlinkedResultsLayout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ct_symlink_"))
        eng = self.tmp / "proj" / "workspace" / "engine-port"
        for d in EXTERNAL_DIRS:
            shutil.copytree(ENG / d, eng / d, ignore=shutil.ignore_patterns("__pycache__"))
        real = self.tmp / "volume" / "results"
        shutil.copytree(ENG / HARNESS_REL, real / "cap_char" / "costterms_harness",
                        ignore=shutil.ignore_patterns("__pycache__"))
        os.symlink(real, eng / "results")           # same shape as job_entry.sh
        self.harness = eng / HARNESS_REL              # logical path through the symlink

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _py(self, *args):
        return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=str(self.harness))

    def test_digest_make_succeeds_through_symlink(self):
        r = self._py(str(self.harness / "ct_harness_digest.py"), "make", "--dir", str(self.harness))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("ABSENT", r.stdout)
        self.assertNotIn("ERROR", r.stdout)

    def test_spec_and_plans_sha_through_symlink(self):
        for args in (("ct_spec.py", "--sha256"), ("ct_runner.py", "plans-sha")):
            r = self._py(str(self.harness / args[0]), *args[1:])
            self.assertEqual(r.returncode, 0, f"{args}: {r.stderr[-500:]}")
            self.assertRegex(r.stdout.strip(), r"^[0-9a-f]{64}$")

    def test_sealed_digest_matches_through_symlink(self):
        """The committed HARNESS_DIGEST.txt (sealed in the plain checkout) must verify in the Job layout too."""
        digest = ENG / HARNESS_REL / "HARNESS_DIGEST.txt"
        if not digest.exists():
            self.skipTest("no HARNESS_DIGEST.txt (unsealed tree)")
        r = self._py(str(self.harness / "ct_harness_digest.py"), "check", "--dir", str(self.harness), "--file", str(digest))
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
