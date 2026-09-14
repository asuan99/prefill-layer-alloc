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


class CommaListTest(unittest.TestCase):
    """The canon writes multi-site citations as one span: `f.py:44,113-160`.

    The single-item form required a backtick right after the first number, so a
    whole comma list was invisible -- which is how `profile.py`'s
    `is_compatible` and `engine_source_hash` anchors escaped the registry while
    a 2026-09-12 edit moved them.
    """

    def test_a_comma_list_yields_every_item(self):
        t = _Tree(BODY, "see `widget.py:3,4,7-8` for all of it\n")
        self.addCleanup(t.close)
        self.assertEqual(
            C.citations_in(t.doc),
            [("widget.py", 3, 3), ("widget.py", 4, 4), ("widget.py", 7, 8)],
        )

    def test_a_plain_citation_is_unchanged(self):
        t = _Tree(BODY, "see `widget.py:3-4`\n")
        self.addCleanup(t.close)
        self.assertEqual(C.citations_in(t.doc), [("widget.py", 3, 4)])

    def test_an_arrow_update_note_yields_nothing_at_all(self):
        """`A -> B` notes are history on the left and current on the right.

        A scanner cannot tell which side it is on, so it must not turn either
        into a live claim.  It yields NOTHING -- not even the left half -- and
        that is why the canon's "broken citation update" lines have to be
        reviewed by hand rather than registered.  Pinned here so a future
        loosening of CITE cannot start baselining the historical side.
        """
        t = _Tree(BODY, "moved: `widget.py:3 -> :4`\n")
        self.addCleanup(t.close)
        self.assertEqual(C.citations_in(t.doc), [])


class PartialRegistrationScopeTest(unittest.TestCase):
    """A long ledger is registered on PART of its citations, and says so.

    Without a declared scope there are only two options for `PROJECT_STATUS.md`:
    baseline its historical citations too (the registry becomes an identity
    function) or leave the gate permanently red (a red gate is a dead gate).
    """

    def setUp(self):
        self.t = _Tree(
            BODY,
            "good `widget.py:3` and broken `widget.py:900` and blank `widget.py:5-6`\n",
        )
        self.addCleanup(self.t.close)

    def test_only_registers_just_the_selected_key(self):
        man = {}
        v, rec, _ = C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        self.assertEqual(v, [])
        self.assertEqual(rec, 1)
        rel = os.path.relpath(os.path.abspath(self.t.doc), C.TRACK_ROOT)
        self.assertEqual(set(man[rel]) - {C.SCOPE_KEY}, {"widget.py:3-3"})

    def test_the_scope_is_recorded_in_the_manifest(self):
        man = {}
        C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        rel = os.path.relpath(os.path.abspath(self.t.doc), C.TRACK_ROOT)
        self.assertEqual(man[rel][C.SCOPE_KEY], {"only": ["widget.py:3-3"]})

    def test_check_reads_the_scope_back_and_ignores_out_of_scope_breakage(self):
        man = {}
        C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        v, _, checked = C.process([self.t.doc], False, man)
        self.assertEqual(v, [], "out-of-scope RANGE/BLANK must not fail a scoped doc")
        self.assertEqual(checked, 1)

    def test_drift_inside_the_scope_is_still_caught(self):
        """★The scope narrows WHAT is watched, never HOW HARD."""
        man = {}
        C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        self.t.rewrite_target(SHIFTED)
        v, _, _ = C.process([self.t.doc], False, man)
        self.assertEqual(len(v), 1)
        self.assertIn("DRIFT", v[0])
        self.assertIn("widget.py:5", v[0])

    def test_an_unscoped_document_keeps_the_unrestricted_checks(self):
        """The original guarantee must survive for fully-registered docs."""
        man = {}
        C.process([self.t.doc], True, man)      # no --only
        v, _, _ = C.process([self.t.doc], False, man)
        kinds = sorted(x.split()[2] for x in v)
        self.assertEqual(kinds, ["BLANK", "RANGE"])

    def test_a_re_snapshot_without_only_drops_the_scope(self):
        man = {}
        C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        rel = os.path.relpath(os.path.abspath(self.t.doc), C.TRACK_ROOT)
        self.assertIn(C.SCOPE_KEY, man[rel])
        C.process([self.t.doc], True, man)
        self.assertNotIn(C.SCOPE_KEY, man[rel])

    def test_the_scope_key_is_never_reported_as_an_orphan(self):
        man = {}
        C.process([self.t.doc], True, man, only=["widget.py:3-3"])
        v, _, _ = C.process([self.t.doc], False, man)
        self.assertFalse([x for x in v if C.SCOPE_KEY in x], v)


class LiveRegistryTest(unittest.TestCase):
    """★The registry must not be an identity function on the REAL canon docs.

    2026-09-12 saw `controller.py` / `profile.py` citations drift twice in one
    session while `--check --all` reported OK, because no manifest entry pointed
    at either file.  These two tests say: the entries now exist, and a line
    shift in either file makes every one of them fail.
    """

    CANON = (
        "../../PROJECT_STATUS.md",
        "../../reports/paper/CLAIM_EVIDENCE_MATRIX.md",
        "../../reports/paper/EXPERIMENT_ROADMAP.md",
    )

    def setUp(self):
        self.man = C.load_manifest()
        missing = [d for d in self.CANON if d not in self.man]
        if missing:
            self.fail(f"canon documents are not registered at all: {missing}")

    def test_the_canon_documents_target_controller_and_profile(self):
        targets = {
            entry["target"]
            for doc in self.CANON
            for key, entry in self.man[doc].items()
            if key != C.SCOPE_KEY
        }
        self.assertTrue(
            any(t.endswith("multiplex/controller.py") for t in targets), targets
        )
        self.assertTrue(
            any(t.endswith("multiplex/profile.py") for t in targets), targets
        )

    def test_every_canon_document_declares_a_partial_scope(self):
        for doc in self.CANON:
            self.assertIn(C.SCOPE_KEY, self.man[doc], doc)
            self.assertTrue(self.man[doc][C.SCOPE_KEY]["only"], doc)

    # ★Every directory a registered key resolves through must be mutated, or the
    # key that lives outside the mutated set comes back UNRESOLVED instead of
    # DRIFT and the assertion below reads it as "survived".  `workloads.py`
    # (registered 2026-09-14 for W4's phase shapes) is the case that forced this
    # to be a list rather than two hard-coded names.
    MUTATED = (
        (("src", "multiplex"), "controller.py"),
        (("src", "multiplex"), "profile.py"),
        (("benchmarks", "pdmux_eval"), "workloads.py"),
    )

    def test_shifting_the_real_files_fails_every_registered_key(self):
        import copy
        import shutil

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        watched = set()
        for parts, name in self.MUTATED:
            src = os.path.join(C.TRACK_ROOT, *parts)
            with open(os.path.join(src, name)) as fh:
                text = fh.read().split("\n")
            text[20:20] = ["# line-shift mutation", "# line-shift mutation"]
            with open(os.path.join(tmp.name, name), "w") as fh:
                fh.write("\n".join(text))
            watched.add(name)
        for parts, _ in self.MUTATED:
            src = os.path.join(C.TRACK_ROOT, *parts)
            for other in os.listdir(src):
                if other.endswith(".py") and other not in watched:
                    shutil.copy(os.path.join(src, other), tmp.name)
                    watched.add(other)

        saved_i, saved_r = C._index, C.SEARCH_ROOTS
        self.addCleanup(lambda: setattr(C, "SEARCH_ROOTS", saved_r))
        self.addCleanup(lambda: setattr(C, "_index", saved_i))
        C._index, C.SEARCH_ROOTS = None, (tmp.name,)

        docs = [os.path.join(C.TRACK_ROOT, d) for d in self.CANON]
        violations, _, _ = C.process(docs, False, copy.deepcopy(self.man))
        drifted = {v.split()[1] for v in violations if " DRIFT " in v}

        # Keys written with a repo-relative path resolve through the real tree
        # (resolve() prefers an explicit path that exists), so the temp shift
        # cannot reach them; they are excluded rather than silently expected.
        expected = {
            key
            for doc in self.CANON
            for key in self.man[doc]
            if key != C.SCOPE_KEY and "/" not in key
        }
        self.assertTrue(expected, "nothing registered to mutate")
        self.assertEqual(
            expected - drifted, set(),
            "a registered citation survived a line shift -- it is an identity",
        )


class BenchmarksSearchRootTest(unittest.TestCase):
    """★`benchmarks/` was not a search root until 2026-09-14.

    Every citation of `workloads.py` / `campaign.py` / `lambda_star.py` -- the
    whole W4 and lambda* workstream -- resolved to UNRESOLVED, so those anchors
    COULD NOT BE REGISTERED and `--check` skipped them in silence.  That is the
    same failure the module comment above WORKSPACE_ROOT records for the engine
    tree, repeated for a second root.  Deleting `benchmarks` from SEARCH_ROOTS
    must make these fail.
    """

    def test_a_bare_benchmarks_basename_resolves(self):
        for name in ("workloads.py", "campaign.py", "lambda_star.py"):
            path, why = C.resolve(name)
            self.assertIsNotNone(path, f"{name}: {why}")
            self.assertTrue(
                path.endswith(os.path.join("benchmarks", "pdmux_eval", name)),
                path,
            )

    def test_benchmarks_is_listed_as_a_search_root(self):
        self.assertIn(
            os.path.join(C.TRACK_ROOT, "benchmarks"), C.SEARCH_ROOTS,
        )

    def test_the_canon_documents_register_the_w4_phase_shapes(self):
        man = C.load_manifest()
        docs = (
            "../../PROJECT_STATUS.md",
            "../../reports/paper/CLAIM_EVIDENCE_MATRIX.md",
            "../../reports/paper/EXPERIMENT_ROADMAP.md",
        )
        for doc in docs:
            targets = {
                entry["target"]
                for key, entry in man.get(doc, {}).items()
                if key != C.SCOPE_KEY
            }
            self.assertTrue(
                any(t.endswith("benchmarks/pdmux_eval/workloads.py")
                    for t in targets),
                f"{doc} does not register the W4 shape anchor: {targets}",
            )

    def test_the_registered_w4_anchor_is_the_shape_pair(self):
        """The claim the canon makes there is (8192,64) / (256,512)."""
        man = C.load_manifest()
        entry = man["../../reports/paper/EXPERIMENT_ROADMAP.md"][
            "workloads.py:159-160"]
        path, _ = C.resolve("workloads.py")
        chunk, _ = C.read_range(path, 159, 160)
        sha, anchor, _ = C.fingerprint(chunk)
        self.assertEqual(sha, entry["sha"])
        self.assertIn("8192 if is_prefill else 256", "\n".join(chunk))
        self.assertIn("64 if is_prefill else 512", "\n".join(chunk))
