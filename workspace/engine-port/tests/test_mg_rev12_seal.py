"""M3 PREREG rev12 SEAL record (results/multigpu/REGISTRATION_rev12_SEAL.txt) -- CPU only, no GPU, no model.

The seal record is the committed list of every byte the rev12 M3 Job registers (mg_job_common.sh MANDATORY_REG, in Job
order) plus the rule / harness digests and the external pins (engine digest, X token file, tokenizer, model revision).
These tests tie the record to (a) the bytes in the tree, (b) what m3_job.sh itself registers in a dry run, and (c) the
constants the harness reads -- and show the verifier is not an identity (negative controls).

Digest definition (same in the record header): sha256 over the section's lines "<sha256>  <repo-relative path>\\n", in
the listed order.  Adding, removing or editing any registered file (e.g. a new VERDICT_*.md beside the PREREG, which the
MANDATORY_REG glob picks up) makes test_job_registration_equals_the_sealed_list fail until the record is reissued --
that is the intended fail-closed outcome.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
HERE = ENGINE / "results" / "multigpu"
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import mg_const as C  # noqa: E402

SEAL = HERE / "REGISTRATION_rev12_SEAL.txt"
_ROW = re.compile(r"^([0-9a-f]{64})  (\S+)$")
_KV = re.compile(r"^([A-Za-z0-9_]+) (\S+)$")


def parse(text):
    """{section title: [lines]} for every '## TITLE' section (comment lines '#' and blank lines dropped)."""
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None and line.strip() and not line.startswith("#"):
            out[cur].append(line.rstrip("\n"))
    return out


def rows(sec):
    got = []
    for line in sec:
        m = _ROW.match(line)
        if not m:
            raise ValueError(f"not a '<sha256>  <path>' row: {line!r}")
        got.append((m.group(1), m.group(2)))
    return got


def kv(sec):
    d = {}
    for line in sec:
        m = _KV.match(line)
        if not m:
            raise ValueError(f"not a 'KEY value' row: {line!r}")
        d[m.group(1)] = m.group(2)
    return d


def listing_digest(rs):
    return hashlib.sha256("".join(f"{h}  {p}\n" for h, p in rs).encode("ascii")).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(text, root=REPO):
    """Problems (empty = the record is consistent with the bytes under root and with its own digests)."""
    bad = []
    try:
        s = parse(text)
        rule, harness = rows(s["RULE SEAL FILES"]), rows(s["HARNESS SEAL FILES"])
        opt, ext = rows(s["OPTIONAL_REG"]), rows(s["CITED SPEC"])
        dig = kv(s["DIGESTS"])
    except (KeyError, ValueError) as e:
        return [f"record unparsable: {e!r}"]
    for h, p in rule + harness + opt + ext:
        f = root / p
        if not f.is_file():
            bad.append(f"missing {p}")
        elif sha_file(f) != h:
            bad.append(f"bytes differ {p}")
    if dig.get("rule_seal_digest") != listing_digest(rule):
        bad.append("rule_seal_digest != digest of RULE SEAL FILES")
    if dig.get("harness_seal_digest") != listing_digest(harness):
        bad.append("harness_seal_digest != digest of HARNESS SEAL FILES")
    if len(rule) != 2 or len({p for _, p in rule + harness}) != len(rule + harness):
        bad.append("rule files != 2 or duplicate registered path")
    return bad


class Rev12SealRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SEAL.read_text(encoding="utf-8")
        cls.s = parse(cls.text)

    def test_record_matches_tree_bytes_and_its_digests(self):
        self.assertEqual(verify(self.text), [])

    def test_rule_files_are_rev11_and_the_adopted_rev12_file(self):
        rule = [p for _, p in rows(self.s["RULE SEAL FILES"])]
        self.assertEqual(rule, [f"workspace/engine-port/results/multigpu/{C.PREREG}",
                                f"workspace/engine-port/results/multigpu/{C.PREREG_REV12}"])
        self.assertEqual(C.PREREG_REV, 12)
        self.assertNotIn("DRAFT", C.PREREG_REV12)                                          # r2+ ⑬: the adopted name
        self.assertFalse((HERE / "PREREG_MULTIGPU_rev12_DRAFT_2026-10-15.md").exists())
        self.assertTrue((HERE / C.PREREG_REV12).is_file())

    def test_external_pins_equal_the_harness_constants(self):
        d = kv(self.s["DIGESTS"])
        self.assertEqual(d["engine_digest"], (HERE / "ENGINE_DIGEST.txt").read_text().strip())
        self.assertEqual(d["engine_digest_scheme"], "pathless-v1")
        self.assertEqual(d["model_revision"], C.MODEL_REVISION)
        self.assertEqual(d["x_tokens_sha256"], C.X_TOKENS_SHA256)
        self.assertEqual(d["x_tokenizer_json_sha256"], C.X_TOKENIZER_SHA256)
        self.assertEqual(d["x_tokenizers_version"], C.X_TOKENIZERS_VERSION)
        self.assertEqual(d["x_lengths_sha256"], C.X_LENGTHS_SHA256)
        self.assertEqual(d["x_parquet_sha256"], C.X_PARQUET_SHA256)
        self.assertEqual(int(d["x_floor_doc_id"]), C.FLOOR_DOC_ID)
        perms = {int(k[len("x_perm_sha256_seed"):]): v for k, v in d.items() if k.startswith("x_perm_sha256_seed")}
        self.assertEqual(perms, C.X_PERM_SHA256)

    def test_job_registration_equals_the_sealed_list(self):
        """m3_job.sh (dry run) registers exactly RULE + HARNESS (Job order, same bytes) and the OPTIONAL slot."""
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "data"
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_MG_DRYRUN="1", PDMUX_MG_TEST_OUT_ROOT=td, PDMUX_MG_JOB="J1",
                       PDMUX_JOB_NAME="sealunit", PDMUX_DATA=str(data))
            p = subprocess.run(["bash", str(HERE / "m3_job.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout[-1500:] + p.stderr[-1500:])
            reg = (Path(td) / "m3_sealunit" / "REGISTRATION_SHA256.txt").read_text()
        job = []
        for line in reg.splitlines():
            m = re.match(r"^([0-9a-f]{64})  (/\S+)$", line)
            if m:
                job.append((m.group(1), os.path.relpath(m.group(2), REPO)))
        self.assertIn("# mandatory_missing=0", reg)
        sealed = rows(self.s["RULE SEAL FILES"]) + rows(self.s["HARNESS SEAL FILES"]) + rows(self.s["OPTIONAL_REG"])
        self.assertEqual(job, sealed)
        self.assertGreaterEqual(len(sealed), 40)


class Rev12SealVerifierIsNotAnIdentity(unittest.TestCase):
    """Negative controls: each mutation of the record must produce a problem (lesson 53)."""

    @classmethod
    def setUpClass(cls):
        cls.text = SEAL.read_text(encoding="utf-8")

    def _mut(self, old, new):
        self.assertIn(old, self.text)
        return verify(self.text.replace(old, new, 1))

    def test_flipped_file_sha(self):
        h, p = rows(parse(self.text)["HARNESS SEAL FILES"])[0]
        self.assertTrue(any("bytes differ" in b for b in self._mut(f"{h}  {p}", f"{'0' * 64}  {p}")))

    def test_dropped_harness_row_breaks_the_digest(self):
        h, p = rows(parse(self.text)["HARNESS SEAL FILES"])[-1]
        self.assertIn("harness_seal_digest != digest of HARNESS SEAL FILES", self._mut(f"{h}  {p}\n", ""))

    def test_wrong_rule_digest(self):
        d = kv(parse(self.text)["DIGESTS"])["rule_seal_digest"]
        self.assertIn("rule_seal_digest != digest of RULE SEAL FILES", self._mut(f"rule_seal_digest {d}", f"rule_seal_digest {'1' * 64}"))

    def test_changed_bytes_under_a_copied_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for h, p in rows(parse(self.text)["RULE SEAL FILES"]):
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                (root / p).write_bytes((REPO / p).read_bytes() + b" ")
            self.assertTrue(any("bytes differ" in b and "PREREG" in b for b in verify(self.text, root=root)))

    def test_missing_section(self):
        self.assertTrue(verify(self.text.replace("## DIGESTS", "## DIGESTZ", 1))[0].startswith("record unparsable"))


if __name__ == "__main__":
    unittest.main()
