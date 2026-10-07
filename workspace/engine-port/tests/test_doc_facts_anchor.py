"""Gate #245 structural repair (2026-10-07): the DESIGN procedure-2 regression
field is checked against a frozen commit, not the live tests/ tree.

Lesson 53: a repair's check must fail on the mutant that reverts the repair.
"""
import importlib.util, os, subprocess, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "scripts", "discipline", "check_doc_facts.py")


def _load():
    spec = importlib.util.spec_from_file_location("_cdf", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _have_anchor(m):
    try:
        subprocess.run(["git", "-C", m.TRACK, "cat-file", "-e", m.PROC2_ANCHOR_SHA],
                       check=True, capture_output=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class TestProc2Anchor(unittest.TestCase):
    def setUp(self):
        self.m = _load()
        if not _have_anchor(self.m):
            self.skipTest("anchor commit not in this clone (shallow clone?)")

    def test_anchor_count_is_155(self):
        self.assertEqual(self.m._regression_tests_at(self.m.PROC2_ANCHOR_SHA)(), 155)

    def test_fact_uses_frozen_source(self):
        src, _doc, _pat = self.m.FACTS["regression tests @proc2 anchor"]
        self.assertEqual(src(), 155)

    def test_new_work_tree_test_does_not_move_anchor(self):
        tdir = os.path.join(self.m.TRACK, "tests")
        fd, path = tempfile.mkstemp(prefix="test_zz_anchor_probe_", suffix=".py", dir=tdir)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("import unittest\nclass T(unittest.TestCase):\n"
                         "    def test_a(self): pass\n    def test_b(self): pass\n")
            self.assertEqual(self.m._regression_tests_at(self.m.PROC2_ANCHOR_SHA)(), 155)
        finally:
            os.unlink(path)

    def test_mutant_live_tree_source_disagrees(self):
        # Reverting the repair (live-tree truth) must disagree with the anchored
        # doc value, i.e. the old checker would FAIL this document again.
        self.assertNotEqual(self.m._regression_tests()(), 155)

    def test_bad_sha_raises_not_passes(self):
        with self.assertRaises(Exception):
            self.m._regression_tests_at("0" * 40)()


if __name__ == "__main__":
    unittest.main()
