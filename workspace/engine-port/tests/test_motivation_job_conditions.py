"""Rehearsals under the two conditions a GPU Job has and the other CPU tests do not (2026-11-14).

  (a) the output directory lies UNDER the engine-port root (a Job writes to {ENG}/results/<campaign>_runs/<job>/raw);
  (b) the interpreter's venv digest is a real sha256, not the rehearsal card's placeholder.

GPU rehearsals 4 and 5 of the fixed-batch part aborted on (b) and (a) while every CPU test passed: the fakes returned the placeholder
itself and used a temp directory outside the root.  Each rehearsal kind is run here under both conditions with the same fakes its own
test file uses.  Reverting either fix (mo_card.check_arm rehearsal digest rule, mo_card.norm_argv replacement order) fails the
fixed-batch case.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_motivation_cap as TC  # noqa: E402
import test_motivation_harness as TH  # noqa: E402
import test_motivation_interference as TI  # noqa: E402

SP = TH.SP
REAL_DIGEST = "8a6b00442c0f761459b81d63aba5ab171b767e3f01eb4d2e2605ff8cb6ec8be3"      # read by GPU rehearsals 4 and 5


def real_freeze(py):
    return SP.VLLM_VENV_FREEZE_SHA256 if py == SP.VLLM_PY else REAL_DIGEST


def job_dir():
    return tempfile.TemporaryDirectory(dir=str(Path(SP.ENG) / "results"), prefix="_jobcond_")


class TestRehearsalsUnderJobConditions(unittest.TestCase):
    def check(self, rec):
        self.assertTrue(rec["passed"], rec["reason"])
        self.assertEqual(list(rec["observed_venv_freeze"]), [REAL_DIGEST])

    def test_fixed_batch(self):
        with job_dir() as td:
            self.check(TC.CR.rehearse_fixed("JCF", Path(td) / "raw", TC.fixed_server(), rehearsal_dir=Path(td) / "rec", digest="d" * 64,
                                            py_version_fn=TH.fake_py, freeze_fn=real_freeze, variant_fn=TC.variant_fn))

    def test_serving_cap_mem(self):
        with job_dir() as td:
            rec = TC.CR.rehearse_serving("JCS", TH.POOL, Path(td) / "raw", lambda bdir, a: TC.KvServer(bdir, a), TH.fake_replay,
                                         rehearsal_dir=Path(td) / "rec", digest="f" * 64, py_version_fn=TH.fake_py, freeze_fn=real_freeze)
            self.assertTrue(rec["passed"], rec["reason"])

    def test_grid(self):
        with job_dir() as td:
            rec = TH.MH.rehearse("JCG", TH.POOL, Path(td) / "raw", TH.make_server(), TH.fake_replay, rehearsal_dir=Path(td) / "rec",
                                 digest="d" * 64, py_version_fn=TH.fake_py, freeze_fn=real_freeze)
            self.assertTrue(rec["passed"], rec["reason"])

    def test_interference(self):
        with job_dir() as td, mock.patch.object(TI.MW, "store_for", lambda wl, token_dir=None: TI.FakeStore() if wl in TI.MW.INTERFERENCE else None):
            rec = TI.IR.rehearse("JCI", TH.POOL, Path(td) / "raw", lambda bdir, a: TH.FakeServer(bdir, a, None), TI.make_replay(TI.SCENARIO),
                                 TI.RATES, rehearsal_dir=Path(td) / "rec", digest="a" * 64, py_version_fn=TH.fake_py, freeze_fn=real_freeze)
            self.assertTrue(rec["passed"], rec["reason"])


if __name__ == "__main__":
    unittest.main()
