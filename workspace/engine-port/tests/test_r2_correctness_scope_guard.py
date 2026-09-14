"""`r2_correctness.sbatch`: the substrate a job serves must match the one its
pre-registration registered, and a mismatch must cost seconds, not a job.

WHAT HAPPENED (job 908020, 2026-09-14, 0.15250 GPU-h).
The pre-registration `rerun_prereg/PREREG_RERUN_2026-09-13.md` (rev2) registered
the scope tuple (NemotronH-Nano-9B-v2, flashinfer, ctx 16384).  Its section 9
submit command carried no `R2C_*` prefix, and the harness defaults are
`Zyphra/Zamba2-2.7B` (`:177`) and `triton` (`:191`), with the context length
derived from whichever model won.  So the only registered way to run the
experiment DETERMINISTICALLY produced a different substrate -- one on which
`mamba_ngroups == 1`, i.e. the fused `weight.data` branch of
`Mixer2RMSNormGated`, which is the branch the registered hypothesis is NOT
about.  Every registered estimator (theta, epsilon, 1.2750 MiB/token, P=38,
6245) is a NemotronH constant and is undefined there.

Nothing malfunctioned.  Three lines of defence -- the pre-registration author,
two rule-layer audits, and the submitter -- each had the information and none
compared the command against the scope table.

WHAT THIS PINS
  1. the guard exists, and sits AFTER the provenance dump and BEFORE the first
     boot, so a mismatch performs no GPU work;
  2. an unset expectation is fatal (a forgotten expectation cannot silently
     disable the check);
  3. a mismatch on ANY of model / backend / ctx is fatal, including the exact
     908020 tuple, which is carried here as a regression fixture;
  4. a match is silent and exit 0, and both the resolved and the expected
     triples are appended to `provenance.txt` (realized, not just target);
  5. NEGATIVE CONTROLS (lesson 53): mutants that drop `exit 2`, that neuter one
     comparison, or that drop the unset check must each make one test fail.

The block is executed for real, extracted verbatim from the sbatch between its
two marker comments.  Nothing here boots a server or touches a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
SBATCH = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness.sbatch"

BEGIN = "# --- BEGIN r2c scope guard"
END = "# --- END r2c scope guard ---"

# The substrate job 908020 actually served, and the one rev2 registered.
WRONG = ("Zyphra/Zamba2-2.7B", "triton", "4096")
REGISTERED = ("nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", "flashinfer", "16384")


def _sbatch_text() -> str:
    return SBATCH.read_text(errors="replace")


def _block(text: str | None = None) -> str:
    text = text if text is not None else _sbatch_text()
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(BEGIN))
    stop = next(i for i, l in enumerate(lines) if l.startswith(END))
    return "\n".join(lines[start : stop + 1])


def run_guard(resolved, expected, block=None):
    """Execute the extracted block with the surrounding shell state faked.

    `resolved` is what the harness would actually serve ($MODEL/$ABE/$CTX after
    the derivation block); `expected` is the R2C_EXPECT_* triple, with None
    meaning "not set at all".
    """
    model, backend, ctx = resolved
    with tempfile.TemporaryDirectory() as out:
        script = textwrap.dedent(
            f"""
            OUT={out!r}
            MODEL={model!r}
            ABE={backend!r}
            CTX={ctx!r}
            """
        ) + "\n" + (block if block is not None else _block()) + "\nexit 0\n"
        env = dict(os.environ)
        for key in ("R2C_EXPECT_MODEL", "R2C_EXPECT_BACKEND", "R2C_EXPECT_CTX"):
            env.pop(key, None)
        for key, value in zip(
            ("R2C_EXPECT_MODEL", "R2C_EXPECT_BACKEND", "R2C_EXPECT_CTX"), expected
        ):
            if value is not None:
                env[key] = value
        proc = subprocess.run(
            ["/usr/bin/bash", "-c", script],
            capture_output=True, text=True, env=env,
        )
        provenance = Path(out) / "provenance.txt"
        return proc, provenance.read_text() if provenance.is_file() else ""


class ScopeGuardPlacementTest(unittest.TestCase):
    def test_guard_is_present(self):
        text = _sbatch_text()
        self.assertIn(BEGIN, text)
        self.assertIn(END, text)

    def test_guard_runs_after_provenance_and_before_any_boot(self):
        """A mismatch must cost seconds, not a job."""
        text = _sbatch_text()
        provenance = text.index('cat "$OUT/provenance.txt"')
        guard = text.index(BEGIN)
        ctx_end = text.index("# --- END r2c ctx resolution ---")
        warmup = text.index("# --- BEGIN r2c warmup launch")
        self.assertLess(ctx_end, guard, "guard must see the DERIVED $CTX")
        self.assertLess(provenance, guard, "guard must run after the provenance dump")
        self.assertLess(guard, warmup, "guard must run before the first boot")

    def test_guard_does_not_sit_inside_a_verbatim_pinned_block(self):
        """The three byte-pinned blocks must stay byte-pinned."""
        text = _sbatch_text()
        guard = text.index(BEGIN)
        for begin, end in (
            ("# --- BEGIN r2c ctx resolution", "# --- END r2c ctx resolution ---"),
            ("# --- BEGIN r2c warmup launch", "# --- END r2c warmup launch ---"),
            ("# --- BEGIN r2c instrumentation", "# --- END r2c instrumentation ---"),
        ):
            with self.subTest(block=begin):
                self.assertFalse(
                    text.index(begin) < guard < text.index(end),
                    f"the scope guard was inserted inside {begin}",
                )


class ScopeGuardBehaviourTest(unittest.TestCase):
    def test_matching_substrate_passes_quietly(self):
        proc, provenance = run_guard(REGISTERED, REGISTERED)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("scope_guard: OK", provenance)
        self.assertIn(f"model={REGISTERED[0]}", provenance)
        self.assertIn(f"backend={REGISTERED[1]}", provenance)
        self.assertIn(f"ctx={REGISTERED[2]}", provenance)

    def test_the_908020_situation_is_refused(self):
        """The exact regression: harness defaults vs the registered tuple."""
        proc, provenance = run_guard(WRONG, REGISTERED)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("refusing to boot", proc.stderr)
        self.assertIn("MISMATCH", provenance)
        for axis in ("model", "backend", "context_length"):
            self.assertIn(f"SCOPE_GUARD: {axis} ", proc.stderr)

    def test_each_axis_alone_is_fatal(self):
        for index, axis in enumerate(("model", "backend", "context_length")):
            resolved = list(REGISTERED)
            resolved[index] = WRONG[index]
            with self.subTest(axis=axis):
                proc, _ = run_guard(tuple(resolved), REGISTERED)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(f"SCOPE_GUARD: {axis} ", proc.stderr)

    def test_unset_expectation_is_fatal(self):
        """A forgotten expectation must not silently disable the check."""
        for index, name in enumerate(
            ("R2C_EXPECT_MODEL", "R2C_EXPECT_BACKEND", "R2C_EXPECT_CTX")
        ):
            expected = list(REGISTERED)
            expected[index] = None
            with self.subTest(missing=name):
                proc, _ = run_guard(REGISTERED, tuple(expected))
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(f"SCOPE_GUARD: {name} is not set", proc.stderr)

    def test_all_three_unset_is_fatal(self):
        """i.e. a bare `sbatch` with no env prefix at all."""
        proc, _ = run_guard(WRONG, (None, None, None))
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)


class ScopeGuardMutationTest(unittest.TestCase):
    """Lesson 53: a repair whose test still passes on the un-repaired form
    proves nothing.  Each mutant must break one of the tests above."""

    def test_mutant_without_exit_is_caught(self):
        mutant = _block().replace('  exit 2\n', '  :\n')
        self.assertNotEqual(mutant, _block(), "mutation did not apply")
        proc, _ = run_guard(WRONG, REGISTERED, block=mutant)
        self.assertEqual(
            proc.returncode, 0,
            "the mutant still exited non-zero: the exit-2 assertion above does "
            "not depend on that line",
        )

    def test_mutant_with_a_neutered_model_comparison_is_caught(self):
        mutant = _block().replace(
            """[ "$MODEL" = "$R2C_EXPECT_MODEL" ] || { echo "SCOPE_GUARD: model '$MODEL' != expected '$R2C_EXPECT_MODEL'" >&2; scope_fail=1; }""",
            ':',
        )
        self.assertNotEqual(mutant, _block(), "mutation did not apply")
        resolved = (WRONG[0], REGISTERED[1], REGISTERED[2])
        proc, _ = run_guard(resolved, REGISTERED, block=mutant)
        self.assertEqual(
            proc.returncode, 0,
            "a model-only mismatch was still refused without the model "
            "comparison: test_each_axis_alone_is_fatal does not cover it",
        )

    def test_mutant_without_the_unset_check_is_caught(self):
        block = _block()
        start = block.index("scope_fail=0")
        stop = block.index('if [ "$scope_fail" = 0 ]; then')
        mutant = block[:start] + "scope_fail=0\n" + block[stop:]
        self.assertNotEqual(mutant, block, "mutation did not apply")
        # with no expectation set, the comparisons compare against empty strings
        proc, _ = run_guard(("", "", ""), (None, None, None), block=mutant)
        self.assertEqual(
            proc.returncode, 0,
            "the unset case was still refused without the unset loop: "
            "test_unset_expectation_is_fatal does not depend on it",
        )


if __name__ == "__main__":
    unittest.main()
