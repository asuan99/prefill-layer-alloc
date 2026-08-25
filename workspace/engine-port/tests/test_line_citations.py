"""Gate for `scripts/discipline/check_line_citations.py`.

WHY THIS EXISTS.  The tool was written to stop lesson #80 (a `file:line`
citation going stale under the reader's feet) and it worked -- and then its own
declared limitation fired within one commit: re-running `--snapshot` on an
edited file silently re-baselined a citation whose target had moved, so
`--check` certified `a1_q3k1_rule.py:83` as `N_MIN_DECODE_STEPS` after the
constant had moved to :98.  A discipline tool with no tests is a discipline
tool that fails quietly.

These pin the four behaviours the re-audit asked for:
  1. DRIFT is detected and the corrected citation is offered;
  2. a changed baseline is REFUSED by --snapshot unless --force;
  3. bare `:NNN` citations inherit the last full-form file (12 of 21 citations
     in the A1 design use that form, and the one surviving stale number was
     among them);
  4. ambiguity is split correctly: identical duplicates resolve to the tree
     that runs, genuinely different files are refused.
"""

import os
import sys
import tempfile
import unittest

sys.modules.pop("profile", None)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts", "discipline"))

import check_line_citations as C  # noqa: E402


class _Tree:
    """A doc + a target file in a temp dir, wired into the tool's resolver."""

    def __init__(self, target_text, doc_text, name="widget.py"):
        self.dir = tempfile.TemporaryDirectory()
        self.target = os.path.join(self.dir.name, name)
        with open(self.target, "w") as fh:
            fh.write(target_text)
        self.doc = os.path.join(self.dir.name, "doc.md")
        with open(self.doc, "w") as fh:
            fh.write(doc_text)
        self._saved_index = C._index
        C._index = None
        self._saved_roots = C.SEARCH_ROOTS
        C.SEARCH_ROOTS = (self.dir.name,)

    def rewrite_target(self, text):
        with open(self.target, "w") as fh:
            fh.write(text)

    def close(self):
        C.SEARCH_ROOTS = self._saved_roots
        C._index = self._saved_index
        self.dir.cleanup()


BODY = "import os\n\nALPHA = 1\nBETA = 2\n\n\ndef gamma():\n    return ALPHA\n"
SHIFTED = "import os\n# inserted\n# inserted\n\nALPHA = 1\nBETA = 2\n\n\ndef gamma():\n    return ALPHA\n"


class DriftTest(unittest.TestCase):
    def setUp(self):
        self.t = _Tree(BODY, "see `widget.py:3` for the constant\n")
        self.addCleanup(self.t.close)

    def test_clean_snapshot_then_check_passes(self):
        man = {}
        v, rec, _ = C.process([self.t.doc], True, man)
        self.assertEqual(v, [])
        self.assertEqual(rec, 1)
        v, _, checked = C.process([self.t.doc], False, man)
        self.assertEqual(v, [])
        self.assertEqual(checked, 1)

    def test_drift_is_detected_and_the_fix_is_offered(self):
        man = {}
        C.process([self.t.doc], True, man)
        self.t.rewrite_target(SHIFTED)          # ALPHA moves 3 -> 5
        v, _, _ = C.process([self.t.doc], False, man)
        self.assertEqual(len(v), 1)
        self.assertIn("DRIFT", v[0])
        self.assertIn("widget.py:5", v[0])

    def test_snapshot_refuses_to_rebase_a_changed_baseline(self):
        """★The failure that actually happened, as a test."""
        man = {}
        C.process([self.t.doc], True, man)
        before = dict(man[list(man)[0]])
        self.t.rewrite_target(SHIFTED)
        v, rec, _ = C.process([self.t.doc], True, man)
        self.assertEqual(rec, 0)
        self.assertEqual(len(v), 1)
        self.assertIn("REBASE-REFUSED", v[0])
        self.assertEqual(man[list(man)[0]], before)   # baseline untouched

    def test_force_rebases_deliberately(self):
        man = {}
        C.process([self.t.doc], True, man)
        self.t.rewrite_target(SHIFTED)
        v, rec, _ = C.process([self.t.doc], True, man, force=True)
        self.assertEqual(v, [])
        self.assertEqual(rec, 1)


class RangeAndBlankTest(unittest.TestCase):
    def test_a_citation_past_the_end_is_reported(self):
        t = _Tree(BODY, "see `widget.py:900`\n")
        self.addCleanup(t.close)
        v, _, _ = C.process([t.doc], True, {})
        self.assertEqual(len(v), 1)
        self.assertIn("RANGE", v[0])

    def test_an_entirely_blank_range_is_reported(self):
        t = _Tree(BODY, "see `widget.py:5-6`\n")   # two blank lines
        self.addCleanup(t.close)
        v, _, _ = C.process([t.doc], True, {})
        self.assertEqual(len(v), 1)
        self.assertIn("BLANK", v[0])


class BareCitationTest(unittest.TestCase):
    def test_bare_citation_inherits_the_last_named_file(self):
        t = _Tree(BODY, "`widget.py:3` and also `:4` for the other one\n")
        self.addCleanup(t.close)
        cites = C.citations_in(t.doc)
        self.assertEqual(cites, [("widget.py", 3, 3), ("widget.py", 4, 4)])

    def test_a_bare_citation_with_no_antecedent_is_ignored(self):
        t = _Tree(BODY, "just `:4` on its own\n")
        self.addCleanup(t.close)
        self.assertEqual(C.citations_in(t.doc), [])

    def test_hist_marker_exempts_a_deliberately_old_citation(self):
        t = _Tree(BODY, "the audit wrote `widget.py:900` [HIST] but the tree says 3\n")
        self.addCleanup(t.close)
        self.assertEqual(C.citations_in(t.doc), [])


class AmbiguityTest(unittest.TestCase):
    """The split that encodes lesson #31: an identical duplicate is the sync,
    a divergent one is the wrong-tree grep."""

    def _two_trees(self, second_text):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        a = os.path.join(d.name, "sglang_engine_dev", "pkg")
        b = os.path.join(d.name, "overlay")
        os.makedirs(a), os.makedirs(b)
        for root, text in ((a, BODY), (b, second_text)):
            with open(os.path.join(root, "widget.py"), "w") as fh:
                fh.write(text)
        saved_i, saved_r = C._index, C.SEARCH_ROOTS
        self.addCleanup(lambda: setattr(C, "SEARCH_ROOTS", saved_r))
        self.addCleanup(lambda: setattr(C, "_index", saved_i))
        C._index, C.SEARCH_ROOTS = None, (a, b)
        return a

    def test_identical_duplicates_resolve_to_the_tree_that_runs(self):
        dev = self._two_trees(BODY)
        path, why = C.resolve("widget.py")
        self.assertIsNone(why)
        self.assertEqual(os.path.dirname(path), dev)

    def test_divergent_duplicates_are_refused(self):
        self._two_trees(BODY + "# overlay has drifted\n")
        path, why = C.resolve("widget.py")
        self.assertIsNone(path)
        self.assertIn("DIFFERENT contents", why)


if __name__ == "__main__":
    unittest.main()


class OrphanAndUntestedDecisionsTest(unittest.TestCase):
    """The three design decisions the 3rd audit found surviving mutation, plus
    the orphan path it named.

    ★B10: mutating `anchor` to the first non-empty line, `relocate` to a partial
    match, and dropping the `anchor_offset` correction all passed the original
    eleven tests -- while the tool's own docstring records the first of those as
    a repair it made after observing fifteen relocation candidates.  A repair
    with no test is a repair that can be undone silently (lesson #53).
    """

    def setUp(self):
        self.t = _Tree(BODY, "see `widget.py:3-4` for the constants\n")
        self.addCleanup(self.t.close)

    def test_orphan_key_is_reported(self):
        """★B9: correcting a citation's NUMBER makes a new key, so
        REBASE-REFUSED never fires and the old key lingers unverified -- the
        path the :83 -> :98 -> :105 incident actually took."""
        man = {}
        C.process([self.t.doc], True, man)
        with open(self.t.doc, "w") as fh:
            fh.write("see `widget.py:4-4` for the constant\n")
        v, _, _ = C.process([self.t.doc], False, man)
        self.assertTrue(any("ORPHAN" in x and "widget.py:3-4" in x for x in v), v)

    def test_prune_drops_the_orphan(self):
        man = {}
        C.process([self.t.doc], True, man)
        with open(self.t.doc, "w") as fh:
            fh.write("see `widget.py:4-4` for the constant\n")
        rel = os.path.relpath(os.path.abspath(self.t.doc), C.TRACK_ROOT)
        live = {f"{c}:{a}-{b}" for c, a, b in C.citations_in(self.t.doc)}
        for key in list(man.get(rel, {})):
            if key not in live:
                del man[rel][key]
        v, _, _ = C.process([self.t.doc], True, man)
        self.assertEqual(v, [])

    def test_anchor_is_the_longest_line_not_the_first(self):
        """The repair the docstring records: `ALPHA = 1` is shorter than
        `BETA = 2`?  No -- equal.  Use a range whose first line is the SHORT
        one, so first-line and longest-line disagree."""
        chunk = ["    x = 1", "    some_much_longer_identifier = 2"]
        _sha, anchor, offset = C.fingerprint(chunk)
        self.assertEqual(anchor, "some_much_longer_identifier = 2")
        self.assertEqual(offset, 1)

    def test_relocate_requires_an_exact_line_match(self):
        """A partial match would relocate `ALPHA = 1` onto `ALPHA = 10`."""
        with open(self.t.target, "w") as fh:
            fh.write("ALPHA = 10\nBETA = 2\n")
        self.assertEqual(C.relocate(self.t.target, "ALPHA = 1", 1), [])
        self.assertEqual(C.relocate(self.t.target, "ALPHA = 10", 1), [1])

    def test_the_offered_fix_subtracts_the_anchor_offset(self):
        """Without the correction the tool points at the ANCHOR's new line
        rather than the range's new start, which is wrong for every
        multi-line citation -- and the live manifest has offsets of 1 and 2."""
        man = {}
        C.process([self.t.doc], True, man)
        self.t.rewrite_target("# pad\n# pad\n" + BODY)
        v, _, _ = C.process([self.t.doc], False, man)
        self.assertEqual(len(v), 1)
        self.assertIn("widget.py:5-6", v[0])      # start moved 3 -> 5, not 4 -> 6
