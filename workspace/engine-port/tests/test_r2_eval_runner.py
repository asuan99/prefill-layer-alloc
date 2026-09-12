"""R2 eval runner: path resolution and context-length pre-flight.

Two defects kept `results/r2_eval/` from ever being created:

  (1) `r2_eval.sbatch` derived its own location from `${BASH_SOURCE[0]}`.  sbatch
      COPIES the submitted script into the node spool, so in a batch job that is
      `/var/spool/slurmd/job<ID>/slurm_script` and every path under the derived
      `engine_root` -- sync_engine_tree.sh, policy_adapter.sh, benchmarks/ --
      pointed at nothing.

  (2) `engine_bench_runner.sh` defaults to `--context-length 16384` and nothing
      in the campaign path ever set `PDMUX_CONTEXT_LENGTH`.  Zamba2-2.7B declares
      `max_position_embeddings=4096`, so SGLang refuses to boot.

★ The end-to-end tests here are paired with a NEGATIVE CONTROL
(`test_spool_copy_*_mutant_*`): the same test run against a mutant that restores
the old `${BASH_SOURCE[0]}` derivation must FAIL.  A repair test that passes on
the unrepaired code certifies nothing (lesson #53).
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[1]
BENCHMARKS = TRACK_ROOT / "benchmarks"
SBATCH = TRACK_ROOT / "scripts" / "r2_eval" / "r2_eval.sbatch"
RUNNER = TRACK_ROOT / "scripts" / "r2_eval" / "engine_bench_runner.sh"
ENGINE_DEV = Path(
    os.environ.get("SGLANG_ENGINE_DEV", "/scratch/ehmoon/whlee/sglang_engine_dev/python")
)

sys.path.insert(0, str(BENCHMARKS))
from pdmux_eval import context_limit as CL           # noqa: E402
from pdmux_eval.campaign import build_runs           # noqa: E402


def _write_trace(path: Path, requests) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"record": "workload", "workload_id": "T1"}) + "\n")
        for index, (inp, outp) in enumerate(requests):
            out.write(
                json.dumps(
                    {
                        "record": "request",
                        "request_id": f"T1-{index:04d}",
                        "arrival_s": float(index),
                        "input_tokens": inp,
                        "output_tokens": outp,
                        "context_tokens": inp,
                        "phase": "steady",
                    }
                )
                + "\n"
            )


class TestDerivedContextLength(unittest.TestCase):
    def test_mirrors_engine_key_list(self):
        """Our copy of CONTEXT_LENGTH_KEYS must equal the engine's, in order.

        The copy exists so the pre-flight can run on a login node without
        importing torch.  A copy that silently diverges would let the pre-flight
        bless a context length the engine then rejects -- exactly the class of
        defect this file is about.  Read the engine source with ast rather than
        importing it, so the check costs nothing.
        """
        source = (ENGINE_DEV / "sglang/srt/utils/hf_transformers_utils.py").read_text(
            errors="replace"
        )
        tree = ast.parse(source)
        engine_keys = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "CONTEXT_LENGTH_KEYS" in targets:
                    engine_keys = tuple(ast.literal_eval(node.value))
        self.assertIsNotNone(engine_keys, "CONTEXT_LENGTH_KEYS not found in engine")
        self.assertEqual(tuple(CL.CONTEXT_LENGTH_KEYS), engine_keys)

    def test_plain_max_position_embeddings(self):
        self.assertEqual(CL.derived_context_length({"max_position_embeddings": 4096}), 4096)

    def test_key_priority_matches_engine_order(self):
        config = {"max_seq_len": 8192, "max_position_embeddings": 4096}
        self.assertEqual(CL.derived_context_length(config), 8192)

    def test_rope_scaling_factor_multiplies(self):
        config = {"max_position_embeddings": 4096, "rope_scaling": {"factor": 4}}
        self.assertEqual(CL.derived_context_length(config), 16384)

    def test_rope_scaling_original_max_disables_factor(self):
        config = {
            "max_position_embeddings": 4096,
            "rope_scaling": {"factor": 4, "original_max_position_embeddings": 2048},
        }
        self.assertEqual(CL.derived_context_length(config), 4096)

    def test_rope_type_llama3_disables_factor(self):
        config = {
            "max_position_embeddings": 4096,
            "rope_scaling": {"factor": 8, "rope_type": "llama3"},
        }
        self.assertEqual(CL.derived_context_length(config), 4096)

    def test_nested_text_config(self):
        config = {"text_config": {"max_position_embeddings": 1024}}
        self.assertEqual(CL.derived_context_length(config), 1024)

    def test_fallback(self):
        self.assertEqual(CL.derived_context_length({}), CL.DEFAULT_FALLBACK_CONTEXT)

    def test_local_directory_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 777}), encoding="utf-8"
            )
            self.assertEqual(CL.model_context_limit(tmp), 777)


class TestTracePreflight(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_fitting_trace_has_no_problems(self):
        trace = self.tmp / "ok.jsonl"
        _write_trace(trace, [(2048, 128), (256, 512)])
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            resolved, problems = CL.check(model_dir, None, [trace])
        self.assertEqual(resolved, 4096)
        self.assertEqual(problems, [])

    def test_oversized_prompt_is_refused(self):
        """The W2/W4/W5 shape: an 8192-token prompt at a 4096 context."""
        trace = self.tmp / "big.jsonl"
        _write_trace(trace, [(8192, 64)])
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            _, problems = CL.check(model_dir, None, [trace])
        self.assertTrue(any("input_tokens >=" in p for p in problems), problems)

    def test_silent_output_truncation_is_refused(self):
        """input fits, input+output does not -> engine shortens max_new_tokens.

        This is the dangerous case: the request succeeds, is scored, and quietly
        contributes fewer decode steps than the trace prescribed.
        """
        trace = self.tmp / "trunc.jsonl"
        _write_trace(trace, [(4000, 512)])
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            _, problems = CL.check(model_dir, None, [trace])
        self.assertTrue(any("input+output >" in p for p in problems), problems)

    def test_requested_context_above_model_limit_is_refused(self):
        """The 16384 default against Zamba2's 4096 -- the original boot failure.

        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN is deliberately NOT the escape
        hatch: it extrapolates RoPE past training instead of lengthening the
        model, which contaminates the very tokens a serving result measures.
        """
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            _, problems = CL.check(model_dir, 16384, [])
        self.assertTrue(any("exceeds" in p for p in problems), problems)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(CL.main(["--model", model_dir]), 0)
                self.assertEqual(
                    CL.main(["--model", model_dir, "--context-length", "16384"]), 2
                )
            self.assertEqual(buffer.getvalue().strip(), "4096")


class TestSbatchSourceText(unittest.TestCase):
    def test_no_bash_source_derivation(self):
        """Regression guard on defect (1)."""
        text = SBATCH.read_text(encoding="utf-8")
        offending = [
            line
            for line in text.splitlines()
            if "BASH_SOURCE" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(offending, [], "r2_eval.sbatch must not locate itself")

    def test_project_root_is_absolute(self):
        text = SBATCH.read_text(encoding="utf-8")
        self.assertIn('project_root="${PDMUX_PROJECT_ROOT:-/', text)

    def test_default_model_agrees_with_runner(self):
        """The default model is written twice; drift would silently re-break ctx.

        The runner keeps its own default so it stays usable standalone, and the
        sbatch needs the value before the runner is reached (to derive the
        context length).  Two constants, one meaning -> assert they are equal.
        """
        import re

        pattern = re.compile(r'PDMUX_MODEL:-([^"}]+)')
        sbatch_default = pattern.search(SBATCH.read_text(encoding="utf-8"))
        runner_default = pattern.search(RUNNER.read_text(encoding="utf-8"))
        self.assertIsNotNone(sbatch_default)
        self.assertIsNotNone(runner_default)
        self.assertEqual(sbatch_default.group(1), runner_default.group(1))

    def test_does_not_enable_the_context_override(self):
        for path in (SBATCH, RUNNER):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                self.assertNotIn(
                    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1", line,
                    f"{path.name} must not override the model's context limit",
                )

    def test_slurm_comment_directive_present(self):
        text = SBATCH.read_text(encoding="utf-8")
        self.assertIn('#SBATCH --comment="field=efficientai;appl=pytorch"', text)


class _SbatchDriver(unittest.TestCase):
    """Shared machinery: build a campaign, run a script copy, capture args."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        if not (ENGINE_DEV / "sglang/srt/distributed/parallel_state.py").is_file():
            raise unittest.SkipTest(f"no editable SGLang tree at {ENGINE_DEV}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.model_dir = self.tmp / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
        )

        trace_dir = self.tmp / "traces"
        _write_trace(trace_dir / "W1.jsonl", [(2048, 128), (2048, 128)])
        runs = build_runs(["W1"], ["B4"], trace_dir, repetitions=5, seed=1)
        self.campaign = self.tmp / "campaign.json"
        self.campaign.write_text(
            json.dumps({"schema": "pdmux.campaign/v1", "runs": [r.__dict__ for r in runs]}),
            encoding="utf-8",
        )

        self.env = dict(os.environ)
        self.env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + self.env["PATH"]
        self.env["PDMUX_CAMPAIGN"] = str(self.campaign)
        self.env["PDMUX_RESULT_ROOT"] = str(self.tmp / "out")
        self.env["PDMUX_DRY_RUN"] = "1"
        self.env["PDMUX_MODEL"] = str(self.model_dir)
        self.env.pop("PDMUX_CONTEXT_LENGTH", None)
        self.env.pop("SLURM_ARRAY_TASK_ID", None)
        self.env.pop("SLURM_JOB_ID", None)

    def spool_copy(self, mutate=None) -> Path:
        """Emulate what sbatch does: copy the script somewhere else entirely."""
        spool = self.tmp / "spool" / "job999"
        spool.mkdir(parents=True, exist_ok=True)
        target = spool / "slurm_script"
        text = SBATCH.read_text(encoding="utf-8")
        if mutate is not None:
            text = mutate(text)
        target.write_text(text, encoding="utf-8")
        target.chmod(0o755)
        return target

    def run_script(self, script: Path, cwd: Path, env_extra=None):
        env = dict(self.env)
        env.update(env_extra or {})
        return subprocess.run(
            ["/usr/bin/bash", str(script)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )

    def assert_served_args(self, result, expected_ctx="4096"):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("--context-length", lines)
        self.assertEqual(lines[lines.index("--context-length") + 1], expected_ctx)
        self.assertIn("--enable-pdmux", lines)
        return lines


class TestSbatchPathResolution(_SbatchDriver):
    def test_a_spool_copy_unrelated_cwd(self):
        """(a) sbatch: the script runs from the spool, cwd is not the repo."""
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        result = self.run_script(self.spool_copy(), elsewhere,
                                 {"SLURM_JOB_ID": "999001"})
        self.assert_served_args(result)
        self.assertTrue(
            (Path(self.env["PDMUX_RESULT_ROOT"]) / "job_999001" / "run_0"
             / "server_args.txt").is_file()
        )

    def test_b_login_node_direct_run(self):
        """(b) `bash .../r2_eval.sbatch` from the repo, no SLURM_* at all."""
        result = self.run_script(SBATCH, PROJECT_ROOT)
        self.assert_served_args(result)
        self.assertTrue(
            (Path(self.env["PDMUX_RESULT_ROOT"]) / "job_local" / "run_0"
             / "server_args.txt").is_file()
        )

    def test_c_array_task_selects_its_own_run(self):
        """(c) array: SLURM_ARRAY_TASK_ID picks the run record, not cwd."""
        result = self.run_script(
            self.spool_copy(), Path("/"),
            {"SLURM_ARRAY_TASK_ID": "3", "SLURM_ARRAY_JOB_ID": "999100",
             "SLURM_JOB_ID": "999103"},
        )
        self.assert_served_args(result)
        record = json.loads(
            (Path(self.env["PDMUX_RESULT_ROOT"]) / "job_999103" / "run_3"
             / "run.json").read_text()
        )
        campaign = json.loads(self.campaign.read_text())
        self.assertEqual(record["run_id"], campaign["runs"][3]["run_id"])

    def test_mutant_restoring_bash_source_fails_from_spool(self):
        """★NEGATIVE CONTROL.  Undo the repair; (a) must break again.

        Without this, `test_a_spool_copy_unrelated_cwd` would also pass on a
        script that never needed repairing, and would certify nothing.
        """
        def mutate(text: str) -> str:
            old = 'project_root="${PDMUX_PROJECT_ROOT:-/scratch/ehmoon/whlee/prefill-layer-alloc}"'
            self.assertIn(old, text)
            return text.replace(
                old,
                'script_dir_m="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
                'project_root="$(cd "${script_dir_m}/../../../.." && pwd)"',
            )

        elsewhere = self.tmp / "elsewhere_m"
        elsewhere.mkdir()
        result = self.run_script(self.spool_copy(mutate), elsewhere,
                                 {"SLURM_JOB_ID": "999002"})
        self.assertNotEqual(
            result.returncode, 0,
            "the pre-repair derivation must NOT resolve from a spool copy",
        )

    def test_mutant_dropping_context_preflight_reintroduces_16384(self):
        """★NEGATIVE CONTROL for defect (2).

        Strip the pre-flight export and the runner falls back to its 16384
        default -- the value SGLang refuses to boot Zamba2-2.7B with.
        """
        def mutate(text: str) -> str:
            old = 'export PDMUX_CONTEXT_LENGTH="${resolved_ctx}"'
            self.assertIn(old, text)
            # Empty, not unset: `set -u` would kill the echo two lines down
            # and we would be testing the wrong failure.  Empty is what "the
            # campaign never set it" looked like, and `${VAR:-16384}` then fires.
            return text.replace(old, 'export PDMUX_CONTEXT_LENGTH=""')

        result = self.run_script(self.spool_copy(mutate), Path("/"),
                                 {"SLURM_JOB_ID": "999003"})
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[lines.index("--context-length") + 1], "16384")

    def test_oversized_trace_is_refused_before_any_allocation_is_used(self):
        trace_dir = self.tmp / "traces_big"
        _write_trace(trace_dir / "W2.jsonl", [(8192, 64)])
        runs = build_runs(["W2"], ["B4"], trace_dir, repetitions=5, seed=1)
        campaign = self.tmp / "campaign_big.json"
        campaign.write_text(
            json.dumps({"schema": "pdmux.campaign/v1", "runs": [r.__dict__ for r in runs]}),
            encoding="utf-8",
        )
        result = self.run_script(
            self.spool_copy(), Path("/"),
            {"PDMUX_CAMPAIGN": str(campaign), "SLURM_JOB_ID": "999004"},
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("REFUSED", result.stderr)

    def test_wrong_project_root_fails_immediately(self):
        result = self.run_script(
            self.spool_copy(), Path("/"),
            {"PDMUX_PROJECT_ROOT": str(self.tmp / "nope"), "SLURM_JOB_ID": "999005"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("is wrong: missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
