"""CPU checks for scripts/bootstrap/engine_digest.py (pathless ENGINE_DIGEST, harness verdict 2026-10-12 F7(b)).

The property under test: the same 30 (27 before 2026-10-20; 29 in the undeployed b0f6cfeb) file contents give the same digest whether they sit
under a local dev tree or under /opt/pdmux/... in a VESSL container, while any content
change, a missing/extra/reordered entry, or a different entry set changes the digest or
is refused.  Manifests are produced by the real `sha256sum` binary with absolute paths,
exactly as sync_engine_tree.sh writes them.  The known-answer value is written out from
the docstring definition, not computed by the tool (lesson 9); each structural guard has
a mutant that disables it and must flip a case (lesson 53); unmodified inputs are the
control (lesson 254).
"""

import contextlib
import hashlib
import importlib.util
import io
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
SPEC = importlib.util.spec_from_file_location("engine_digest_t", ENG / "scripts" / "bootstrap" / "engine_digest.py")
ED = importlib.util.module_from_spec(SPEC)
sys.modules["engine_digest_t"] = ED
SPEC.loader.exec_module(ED)
RELS = ED.registered_relpaths()


def make_tree(root: Path, salt: str = "") -> Path:
    py = root / "python"
    for i, rel in enumerate(RELS):
        f = py / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# file {i} {rel}{salt}\n")
    return py


def write_manifest(py: Path, out: Path, rels=None) -> Path:
    """The sync script's own command: sha256sum over absolute paths in registered order."""
    r = subprocess.run(["sha256sum", *[str(py / rel) for rel in (rels or RELS)]], capture_output=True, text=True,
                       check=True)
    out.write_text(r.stdout)
    return out


def run_cli(*argv):
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = ED.main(list(argv))
    return rc, so.getvalue(), se.getvalue()


class TestRegisteredEntries(unittest.TestCase):
    def test_parsed_from_the_sync_script(self):
        self.assertEqual(len(RELS), 30)
        self.assertEqual(RELS[0], "sglang/srt/distributed/parallel_state.py")
        self.assertEqual(RELS[24], "sglang/srt/layers/attention/flashinfer_backend.py")
        self.assertEqual(RELS[25:27], ["sglang/srt/multiplex/pdmux_context.py", "sglang/srt/multiplex/multigpu_guard.py"])
        # 2026-10-20 (decision 15): appended, never inserted
        self.assertEqual(RELS[27:29], ["sglang/srt/multiplex/admission_probe.py",
                                       "sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"])
        # same day (decision 16 O-2, same digest change): I-4/I-6 probe appended last
        self.assertEqual(RELS[29:], ["sglang/srt/multiplex/decode_probe.py"])

    def test_sync_script_change_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            text = ED.SYNC_SCRIPT.read_text()
            line = '  "${runtime_python}/sglang/srt/multiplex/multigpu_guard.py" \\\n'
            self.assertEqual(text.count(line), 1)
            p = Path(d) / "sync.sh"
            p.write_text(text.replace(line, ""))  # mutant: an entry dropped
            with self.assertRaises(ED.DigestInputError):
                ED.registered_relpaths(p)


class TestDigest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.local = make_tree(self.d / "home" / "wonho" / "sglang_engine_dev_multigpu")
        self.opt = make_tree(self.d / "opt" / "pdmux" / "sglang_engine_dev")
        self.m_local = write_manifest(self.local, self.d / "local.sha256")
        self.m_opt = write_manifest(self.opt, self.d / "opt.sha256")

    def tearDown(self):
        self.tmp.cleanup()

    def test_known_answer_from_definition(self):
        hs = sorted(hashlib.sha256(f"# file {i} {rel}\n".encode()).hexdigest() for i, rel in enumerate(RELS))
        want = hashlib.sha256(("\n".join(hs) + "\n").encode("ascii")).hexdigest()
        self.assertEqual(ED.digest_of_entries(ED.load_entries(runtime=str(self.local))), want)

    def test_path_free_and_tree_equals_manifest(self):
        a = ED.digest_of_entries(ED.load_entries(manifest=str(self.m_local)))
        b = ED.digest_of_entries(ED.load_entries(manifest=str(self.m_opt)))
        c = ED.digest_of_entries(ED.load_entries(runtime=str(self.opt)))
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        # why the old check could not be sealed locally: the manifest FILE hashes differ by path alone
        self.assertNotEqual(hashlib.sha256(self.m_local.read_bytes()).hexdigest(),
                            hashlib.sha256(self.m_opt.read_bytes()).hexdigest())

    def test_one_byte_changes_the_digest(self):
        base = ED.digest_of_entries(ED.load_entries(runtime=str(self.opt)))
        f = self.opt / RELS[13]
        f.write_text(f.read_text() + " ")
        self.assertNotEqual(ED.digest_of_entries(ED.load_entries(runtime=str(self.opt))), base)

    def test_structural_guards(self):
        lines = self.m_opt.read_text().splitlines(keepends=True)
        bad = self.d / "bad.sha256"
        cases = {
            "26 entries": lines[:-1],
            "28 entries": lines + [lines[0]],
            "swapped": [lines[1], lines[0]] + lines[2:],
            "not hex": ["zz" + lines[0][2:]] + lines[1:],
        }
        for name, ls in cases.items():
            bad.write_text("".join(ls))
            with self.subTest(name), self.assertRaises(ED.DigestInputError):
                ED.load_entries(manifest=str(bad))
        with self.assertRaises(ED.DigestInputError):
            ED.load_entries(manifest=str(self.d / "absent.sha256"))
        (self.opt / RELS[5]).unlink()
        with self.assertRaises(ED.DigestInputError):
            ED.load_entries(runtime=str(self.opt))

    def test_mutant_order_check_removed_is_killed(self):
        """Without the suffix check a swapped manifest would yield the SAME (sorted) digest silently."""
        lines = self.m_opt.read_text().splitlines(keepends=True)
        sw = self.d / "sw.sha256"
        sw.write_text("".join([lines[1], lines[0]] + lines[2:]))
        real = ED.manifest_entries

        def no_suffix_check(path, rels, require_all=True):
            rows = ED.parse_manifest(path)
            return [(rel, h) for (h, _), rel in zip(rows, rels)]
        with mock.patch.object(ED, "manifest_entries", no_suffix_check):
            d_mut = ED.digest_of_entries(ED.load_entries(manifest=str(sw)))
        self.assertEqual(d_mut, ED.digest_of_entries(ED.load_entries(manifest=str(self.m_opt))))  # mutant: silent
        with self.assertRaises(ED.DigestInputError):  # real: refused
            real(sw, RELS)

    def test_check_cli(self):
        dig = ED.digest_of_entries(ED.load_entries(runtime=str(self.local)))
        sealed = self.d / "ENGINE_DIGEST.txt"
        sealed.write_text(dig + "\n")
        rc, out, _ = run_cli("check", "--manifest", str(self.m_opt), "--sealed-file", str(sealed),
                             "--out", str(self.d / "chk.json"))
        self.assertEqual(rc, 0, out)
        self.assertTrue(json.loads((self.d / "chk.json").read_text())["match"])
        sealed.write_text("UNSEALED\n")
        self.assertEqual(run_cli("check", "--manifest", str(self.m_opt), "--sealed-file", str(sealed))[0], 1)
        self.assertEqual(run_cli("check", "--manifest", str(self.m_opt), "--sealed", "0" * 64)[0], 1)
        self.assertEqual(run_cli("check", "--manifest", str(self.m_opt), "--sealed-file", str(self.d / "none"))[0], 1)
        self.assertEqual(run_cli("check", "--manifest", str(self.d / "absent"), "--sealed", dig)[0], 2)
        rc, out, _ = run_cli("digest", "--runtime", str(self.opt))
        self.assertEqual((rc, out.strip()), (0, dig))


class TestComparePrefix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.py = make_tree(self.d / "tree")
        self.res = self.d / "results"

    def tearDown(self):
        self.tmp.cleanup()

    def job_manifest(self, campaign, job, rels, py=None):
        meta = self.res / campaign / "vessl" / job / "meta"
        meta.mkdir(parents=True)
        return write_manifest(py or self.py, meta / "runtime_source_manifest.sha256", rels)

    def test_latest_by_job_timestamp_not_mtime(self):
        old = self.job_manifest("a", "pdmux-a-1111111-20261011182445", RELS[:25])
        new = self.job_manifest("b", "pdmux-b-2222222-20261001000000", RELS[:25])
        os.utime(new, (2e9, 2e9))  # newest mtime, oldest timestamp
        self.assertEqual(ED.latest_vessl_manifest(self.res), old)

    def test_prefix_report(self):
        other = make_tree(self.d / "other")
        (other / RELS[2]).write_text("drift\n")
        self.job_manifest("a", "pdmux-a-1111111-20261011182445", RELS[:25], py=other)
        rc, out, _ = run_cli("compare-prefix", "--runtime", str(self.py), "--reference-root", str(self.res),
                             "--out", str(self.d / "p.json"))
        doc = json.loads((self.d / "p.json").read_text())
        self.assertEqual(rc, 1)
        self.assertEqual((doc["n_equal"], doc["differing"]), (24, [RELS[2]]))
        self.assertEqual([e["rel"] for e in doc["beyond_prefix"]], RELS[25:])
        shutil.rmtree(self.res)
        self.job_manifest("a", "pdmux-a-1111111-20261011182445", RELS[:25])
        self.assertEqual(run_cli("compare-prefix", "--runtime", str(self.py), "--reference-root", str(self.res))[0], 0)
        self.assertEqual(run_cli("compare-prefix", "--runtime", str(self.py), "--reference-root",
                                 str(self.d / "empty"))[0], 2)

    def test_short_reference_refused(self):
        ref = self.job_manifest("a", "pdmux-a-1111111-20261011182445", RELS[:17])
        self.assertEqual(run_cli("compare-prefix", "--runtime", str(self.py), "--reference", str(ref))[0], 2)


if __name__ == "__main__":
    unittest.main()
