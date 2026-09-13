"""`results/r2_correctness/r2_correctness.sbatch`: context length is DERIVED.

WHAT WAS WRONG
--------------
The server command line carried the literal `--context-length 4096` (X3 audit
finding D1).  `R2C_MODEL` has existed for a while, so pointing the correctness
gate at any model whose `max_position_embeddings` is not 4096 served a context
length the model never declared -- either refusing to boot (limit < 4096) or
silently truncating nothing and certifying a window far below what the campaign
it is supposed to certify will actually serve (limit > 4096).

WHAT IS PINNED HERE
-------------------
1.  The literal is gone and `$CTX` is what reaches `--context-length`.
2.  The default is the model's own derived limit, computed by the SAME module
    (`benchmarks/pdmux_eval/context_limit.py`) that
    `scripts/r2_eval/r2_eval.sbatch` pre-flights the campaign with.
3.  **Reproduction guard**: for `Zyphra/Zamba2-2.7B` that derivation yields
    exactly 4096, so jobs 907100 / 907456 are reproduced by the default, with no
    `R2C_CTX` in the environment.  This is the whole reason the repair is safe.
4.  `R2C_CTX` wins when set.
5.  ★NEGATIVE CONTROL (lesson #53): a mutant that restores the literal 4096 must
    make test (1) fail, and a mutant that deletes the `R2C_CTX` branch must make
    test (4) fail.  Without these, both tests would also pass on the unrepaired
    script and would certify nothing.

The resolution block is executed for real, extracted verbatim from the sbatch
between its two marker comments, so the test exercises the shipped text rather
than a paraphrase of it.  Nothing here boots a server or touches a GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[1]
BENCHMARKS = TRACK_ROOT / "benchmarks"
SBATCH = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness.sbatch"
CHECKER = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness_check.py"
HF_HOME = PROJECT_ROOT / "hf_cache"

BEGIN = "# --- BEGIN r2c ctx resolution"
END = "# --- END r2c ctx resolution ---"

# The verdict rule must not move when the harness changes.  Pinned in the task
# brief and in the job's own provenance dump.
CHECKER_SHA256 = "ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30"


def sbatch_text() -> str:
    return SBATCH.read_text(encoding="utf-8")


def ctx_block(text: str) -> str:
    """The resolution block, verbatim, without its marker lines."""
    start = text.index(BEGIN)
    start = text.index("\n", start) + 1
    stop = text.index(END)
    return text[start:stop]


def non_comment_lines(text: str):
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            yield line


class TestServerCommandHasNoLiteralContext(unittest.TestCase):
    def test_context_length_argument_is_the_variable(self):
        served = [
            line
            for line in non_comment_lines(sbatch_text())
            if "--context-length" in line
        ]
        self.assertEqual(len(served), 1, served)
        self.assertIn('--context-length "$CTX"', served[0])

    def test_no_literal_context_length_anywhere_executable(self):
        offending = [
            line
            for line in non_comment_lines(sbatch_text())
            if re.search(r"--context-length\s+[0-9]", line)
        ]
        self.assertEqual(offending, [], offending)

    def test_mutant_restoring_the_literal_is_caught(self):
        """★NEGATIVE CONTROL for the two tests above."""
        mutant = sbatch_text().replace(
            '--context-length "$CTX"', "--context-length 4096"
        )
        self.assertNotEqual(mutant, sbatch_text())
        offending = [
            line
            for line in non_comment_lines(mutant)
            if re.search(r"--context-length\s+[0-9]", line)
        ]
        self.assertNotEqual(offending, [], "the guard must fire on the mutant")

    def test_does_not_enable_the_context_override(self):
        for line in non_comment_lines(sbatch_text()):
            self.assertNotIn("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1", line)

    def test_attention_backend_default_is_unchanged(self):
        """The backend is a knob now, but the 907100/907456 value is the default.

        NemotronHForCausalLM is hard-refused on triton by
        `ServerArgs.__post_init__`, so a long-context NemotronH boot needs
        `R2C_ATTN_BACKEND=flashinfer`; making that expressible must not move the
        default, or the existing Zamba2 result stops reproducing.
        """
        text = sbatch_text()
        self.assertIn('ABE="${R2C_ATTN_BACKEND:-triton}"', text)
        served = [
            line
            for line in non_comment_lines(text)
            if "--attention-backend" in line
        ]
        self.assertEqual(len(served), 1, served)
        self.assertIn('--attention-backend "$ABE"', served[0])

    def test_runner_attention_backend_default_is_unchanged(self):
        runner = (
            TRACK_ROOT / "scripts" / "r2_eval" / "engine_bench_runner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--attention-backend "${PDMUX_ATTENTION_BACKEND:-triton}"',
                      runner)

    def test_verdict_rule_file_is_untouched(self):
        """The judging rule is pinned; a harness change may not move it."""
        import hashlib

        digest = hashlib.sha256(CHECKER.read_bytes()).hexdigest()
        self.assertEqual(digest, CHECKER_SHA256)


class _BlockRunner(unittest.TestCase):
    """Run the extracted resolution block in bash, with a real interpreter."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_block(self, model, env_extra=None, block=None, expect_rc=0):
        if block is None:
            block = ctx_block(sbatch_text())
        script = self.tmp / "block.sh"
        script.write_text(
            "set -uo pipefail\n"
            f'PROJECT={PROJECT_ROOT}\n'
            f'MODEL={json.dumps(str(model))}\n'
            + block
            + 'printf "RESOLVED=%s\\n" "$CTX"\n',
            encoding="utf-8",
        )
        env = dict(os.environ)
        env.pop("R2C_CTX", None)
        env["HF_HOME"] = str(HF_HOME)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
        env.update(env_extra or {})
        result = subprocess.run(
            ["/usr/bin/bash", str(script)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        self.assertEqual(
            result.returncode, expect_rc, result.stdout + result.stderr
        )
        return result

    def resolved(self, result):
        for line in result.stdout.splitlines():
            if line.startswith("RESOLVED="):
                return line.split("=", 1)[1]
        self.fail("block printed no RESOLVED line: " + result.stdout)


class TestCtxResolution(_BlockRunner):
    def test_local_config_directory_is_derived(self):
        model = self.tmp / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"max_position_embeddings": 5120}), encoding="utf-8"
        )
        self.assertEqual(self.resolved(self.run_block(model)), "5120")

    def test_r2c_ctx_overrides(self):
        model = self.tmp / "model2"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"max_position_embeddings": 5120}), encoding="utf-8"
        )
        result = self.run_block(model, {"R2C_CTX": "777"})
        self.assertEqual(self.resolved(result), "777")
        self.assertIn("R2C_CTX", result.stdout)

    def test_mutant_without_the_override_branch_ignores_r2c_ctx(self):
        """★NEGATIVE CONTROL for test_r2c_ctx_overrides."""
        block = ctx_block(sbatch_text())
        self.assertIn('if [ -n "${R2C_CTX:-}" ]; then', block)
        mutant = block.replace('if [ -n "${R2C_CTX:-}" ]; then', "if false; then")
        model = self.tmp / "model3"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"max_position_embeddings": 5120}), encoding="utf-8"
        )
        result = self.run_block(model, {"R2C_CTX": "777"}, block=mutant)
        self.assertEqual(self.resolved(result), "5120")

    def test_undeliverable_model_is_fatal_not_a_fallback(self):
        missing = self.tmp / "no_such_model_anywhere"
        self.run_block(missing, expect_rc=2)

    def test_non_numeric_override_is_rejected(self):
        model = self.tmp / "model4"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
        )
        self.run_block(model, {"R2C_CTX": "4096x"}, expect_rc=2)


class TestCachedModelsResolveAsExpected(_BlockRunner):
    """The reproduction guard, against the real cached checkpoints.

    `model_context_limit` reads `config.json` out of the HF cache, so this needs
    no weights load, no network and no GPU -- but it does need the cache entry,
    hence the skips.
    """

    @staticmethod
    def _cached(repo_id: str) -> bool:
        name = "models--" + repo_id.replace("/", "--")
        hits = list((HF_HOME / "hub" / name).glob("snapshots/*/config.json"))
        return bool(hits)

    def test_zamba2_2_7b_still_derives_4096(self):
        """★907100 / 907456 reproduction: the default MUST still be 4096."""
        if not self._cached("Zyphra/Zamba2-2.7B"):
            self.skipTest("Zyphra/Zamba2-2.7B not in hf_cache")
        result = self.run_block("Zyphra/Zamba2-2.7B")
        self.assertEqual(self.resolved(result), "4096")
        self.assertIn("derived from", result.stdout)

    def test_nemotron_nano_9b_derives_its_declared_131072(self):
        """The reason the model swap is possible at all; also a warning label.

        131072 boots, but it is not the context length the r2_eval campaign
        resolves for W1-W9, which is why the sbatch header tells you to set
        R2C_CTX explicitly for this model.
        """
        repo = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
        if not self._cached(repo):
            self.skipTest(f"{repo} not in hf_cache")
        self.assertEqual(self.resolved(self.run_block(repo)), "131072")


class TestPythonPathIsSetBeforeUse(unittest.TestCase):
    def test_block_exports_pythonpath_for_the_benchmarks_package(self):
        block = ctx_block(sbatch_text())
        self.assertIn("PYTHONPATH=", block)
        self.assertIn("pdmux_eval.context_limit", block)

    def test_hf_home_is_exported_before_the_block(self):
        text = sbatch_text()
        self.assertLess(
            text.index("export HF_HOME="),
            text.index(BEGIN),
            "the derivation reads the offline HF cache; HF_HOME must precede it",
        )


if __name__ == "__main__":
    sys.path.insert(0, str(BENCHMARKS))
    unittest.main()
