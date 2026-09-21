"""campaign.json must say what it served, and `cuda_graph` must be read.

TWO DEFECTS THIS FILE PINS SHUT
-------------------------------
(P1) A `CampaignRun` had no `model` and no `context_length`.  The checkpoint came
     from `PDMUX_MODEL` and the window from whatever `r2_eval.sbatch` derived at
     run time, so a finished run record could not answer "what did this number
     measure".  With the served model being swapped (Zamba2-2.7B, ctx 4096 ->
     NVIDIA-Nemotron-Nano-9B-v2-Base, ctx 131072) that is not a bookkeeping
     nicety: two campaigns with identical records would have served different
     models.

(P2) `cuda_graph: true` was in every record and READ BY NOTHING -- decoration on
     the one axis the project canon calls the operating point.  It now drives
     `PDMUX_DISABLE_CUDA_GRAPH`, and a contradicting environment is refused.

HOW IT IS TESTED
----------------
`PDMUX_DRY_RUN=1` makes `engine_bench_runner.sh` print the exact `launch_server`
argument vector and exit before any GPU work, so the whole pipeline
(campaign.json -> run.json -> policy_adapter -> runner -> server args) is
exercised on a login node.  Every positive test is paired with a ★NEGATIVE
CONTROL mutant in which the repair is undone and the same assertion must fail
(lesson #53): otherwise these tests would pass on the unrepaired scripts.
"""

from __future__ import annotations

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
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(TRACK_ROOT.parents[2])),
                     "sglang_engine_dev", "python")
)

sys.path.insert(0, str(BENCHMARKS))
from pdmux_eval.campaign import DEFAULT_MODEL, build_runs   # noqa: E402


# A trace header must now declare the per-shape lambda* it was generated from
# (campaign schema v3); `campaign.build_runs` REFUSES a trace that declares
# none.  These fixtures are about model/context provenance, so they declare a
# measured one and stay on the accept path.  The refusal itself is tested in
# tests/test_lambda_star_per_shape.py.
MEASURED_LAMBDA_STAR = {
    "schema": "pdmux.lambda_star/v1",
    "definition": "slo_sustainable",
    "measured_by": "unit-test fixture",
    "rate_derivation": "single_shape",
    "source": "measured",
    "caveats": [],
    "shapes": {
        "2048x128": {"req_per_s": 1.0, "source": "measured", "evidence": "fixture"}
    },
}


def _write_trace(path: Path, requests) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "record": "workload",
                    "workload_id": "T1",
                    "lambda_star": MEASURED_LAMBDA_STAR,
                }
            )
            + "\n"
        )
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


class TestCampaignFields(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.trace_dir = self.tmp / "traces"
        _write_trace(self.trace_dir / "W1.jsonl", [(2048, 128)])

    def test_every_run_carries_model_and_context_length(self):
        runs = build_runs(
            ["W1"], ["B1", "B4"], self.trace_dir, 5, seed=1,
            model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", context_length=16384,
        )
        self.assertTrue(runs)
        for run in runs:
            self.assertEqual(run.model, "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base")
            self.assertEqual(run.context_length, 16384)
            self.assertTrue(run.cuda_graph)

    def test_defaults_preserve_the_old_behaviour(self):
        runs = build_runs(["W1"], ["B1"], self.trace_dir, 5, seed=1)
        self.assertEqual({r.model for r in runs}, {DEFAULT_MODEL})
        self.assertEqual({r.context_length for r in runs}, {None},
                         "None must mean 'derive at run time', not 16384")

    def test_no_cuda_graph_is_expressible(self):
        runs = build_runs(["W1"], ["B1"], self.trace_dir, 5, seed=1,
                          cuda_graph=False)
        self.assertEqual({r.cuda_graph for r in runs}, {False})

    def test_pairs_still_share_trace_and_seed_after_the_new_fields(self):
        runs = build_runs(["W1"], ["B1", "B4"], self.trace_dir, 5, seed=1)
        by_pair = {}
        for run in runs:
            by_pair.setdefault(run.pair_id, []).append(run)
        for members in by_pair.values():
            self.assertEqual(len({m.trace_sha256 for m in members}), 1)
            self.assertEqual(len({m.server_seed for m in members}), 1)
            self.assertEqual(len({m.model for m in members}), 1)
            self.assertEqual(len({m.context_length for m in members}), 1)

    def test_cli_writes_schema_v3_and_top_level_provenance(self):
        out = self.tmp / "campaign.json"
        result = subprocess.run(
            [sys.executable, "-m", "pdmux_eval.campaign",
             "--trace-dir", str(self.trace_dir), "--workloads", "W1",
             "--baselines", "B1", "B4", "--repetitions", "5",
             "--model", "some/model", "--context-length", "12288",
             "--output", str(out)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(BENCHMARKS)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text())
        self.assertEqual(data["schema"], "pdmux.campaign/v3")
        self.assertEqual(data["model"], "some/model")
        self.assertEqual(data["context_length"], 12288)
        self.assertTrue(data["cuda_graph"])
        self.assertTrue(all(r["model"] == "some/model" for r in data["runs"]))

    def test_cli_rejects_a_nonpositive_context_length(self):
        out = self.tmp / "bad.json"
        result = subprocess.run(
            [sys.executable, "-m", "pdmux_eval.campaign",
             "--trace-dir", str(self.trace_dir), "--workloads", "W1",
             "--baselines", "B1", "--repetitions", "5",
             "--context-length", "0", "--output", str(out)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(BENCHMARKS)},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(out.exists())


class _DryRunDriver(unittest.TestCase):
    """Drive the real sbatch -> adapter -> runner chain with PDMUX_DRY_RUN=1."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        if not (ENGINE_DEV / "sglang/srt/distributed/parallel_state.py").is_file():
            raise unittest.SkipTest(f"no editable SGLang tree at {ENGINE_DEV}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.model_dir = self.tmp / "declared_model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 9216}), encoding="utf-8"
        )
        self.other_model = self.tmp / "other_model"
        self.other_model.mkdir()
        (self.other_model / "config.json").write_text(
            json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
        )

        self.trace_dir = self.tmp / "traces"
        _write_trace(self.trace_dir / "W1.jsonl", [(2048, 128), (2048, 128)])

        self.env = dict(os.environ)
        self.env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + self.env["PATH"]
        self.env["PDMUX_RESULT_ROOT"] = str(self.tmp / "out")
        self.env["PDMUX_DRY_RUN"] = "1"
        for var in ("PDMUX_MODEL", "PDMUX_CONTEXT_LENGTH",
                    "PDMUX_DISABLE_CUDA_GRAPH", "SLURM_ARRAY_TASK_ID",
                    "SLURM_JOB_ID"):
            self.env.pop(var, None)

    def campaign(self, **kwargs) -> Path:
        runs = build_runs(["W1"], ["B4"], self.trace_dir, 5, seed=1, **kwargs)
        path = self.tmp / f"campaign_{len(list(self.tmp.glob('campaign_*')))}.json"
        path.write_text(
            json.dumps({"schema": "pdmux.campaign/v2",
                        "runs": [r.__dict__ for r in runs]}),
            encoding="utf-8",
        )
        return path

    def v1_campaign(self) -> Path:
        """A pre-v2 manifest: no model / context_length / cuda_graph keys."""
        runs = build_runs(["W1"], ["B4"], self.trace_dir, 5, seed=1)
        stripped = []
        for run in runs:
            record = dict(run.__dict__)
            for key in ("model", "context_length", "cuda_graph",
                        "lambda_star_source"):
                record.pop(key)
            stripped.append(record)
        path = self.tmp / "campaign_v1.json"
        path.write_text(
            json.dumps({"schema": "pdmux.campaign/v1", "runs": stripped}),
            encoding="utf-8",
        )
        return path

    def run_chain(self, campaign_path, env_extra=None, mutate=None, job="999900"):
        script = self.tmp / f"slurm_script_{job}"
        text = SBATCH.read_text(encoding="utf-8")
        if mutate is not None:
            text = mutate(text)
        script.write_text(text, encoding="utf-8")
        script.chmod(0o755)
        env = dict(self.env)
        env["PDMUX_CAMPAIGN"] = str(campaign_path)
        env["SLURM_JOB_ID"] = job
        env.update(env_extra or {})
        return subprocess.run(
            ["/usr/bin/bash", str(script)],
            cwd="/", env=env, capture_output=True, text=True, timeout=900,
        )

    def served(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.splitlines()

    def arg_value(self, lines, flag):
        self.assertIn(flag, lines, lines)
        return lines[lines.index(flag) + 1]


class TestRecordDrivesWhatIsServed(_DryRunDriver):
    def test_model_comes_from_the_record(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir)), job="999910"))
        self.assertEqual(self.arg_value(lines, "--model-path"),
                         str(self.model_dir))

    def test_context_length_comes_from_the_record(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir), context_length=8192),
            job="999911"))
        self.assertEqual(self.arg_value(lines, "--context-length"), "8192")

    def test_null_context_length_derives_the_model_limit(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir)), job="999912"))
        self.assertEqual(self.arg_value(lines, "--context-length"), "9216")

    def test_record_context_length_above_the_model_limit_is_refused(self):
        result = self.run_chain(
            self.campaign(model=str(self.model_dir), context_length=131072),
            job="999913")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("REFUSED", result.stderr)

    def test_contradicting_env_model_is_refused(self):
        result = self.run_chain(
            self.campaign(model=str(self.model_dir)),
            {"PDMUX_MODEL": str(self.other_model)}, job="999914")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("contradicts the run record", result.stderr)

    def test_agreeing_env_model_is_accepted(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir)),
            {"PDMUX_MODEL": str(self.model_dir)}, job="999915"))
        self.assertEqual(self.arg_value(lines, "--model-path"),
                         str(self.model_dir))

    def test_v1_record_still_runs_through_the_env(self):
        """Back-compatibility: a pre-v2 manifest has none of the three keys."""
        lines = self.served(self.run_chain(
            self.v1_campaign(), {"PDMUX_MODEL": str(self.model_dir)},
            job="999916"))
        self.assertEqual(self.arg_value(lines, "--model-path"),
                         str(self.model_dir))

    def test_mutant_ignoring_the_record_model_falls_back(self):
        """★NEGATIVE CONTROL for test_model_comes_from_the_record.

        Force `record_model` empty -- what the code did before the field
        existed -- and the declared model must stop reaching the server.  The
        literal fallback is then used, which has no local config, so the
        pre-flight refuses; either way the assertion that the DECLARED model was
        served must no longer hold.
        """
        def mutate(text: str) -> str:
            old = 'record_model="${record_fields[1]}"'
            self.assertIn(old, text)
            return text.replace(old, 'record_model=""')

        result = self.run_chain(self.campaign(model=str(self.model_dir)),
                                mutate=mutate, job="999917")
        served_declared = (
            result.returncode == 0
            and str(self.model_dir) in result.stdout.splitlines()
        )
        self.assertFalse(
            served_declared,
            "the record-read is what makes the declared model win",
        )


class TestCudaGraphFieldIsRead(_DryRunDriver):
    CG_FLAGS = ("--disable-cuda-graph", "--disable-piecewise-cuda-graph")

    def test_cuda_graph_true_means_no_disable_flags(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir)), job="999920"))
        for flag in self.CG_FLAGS:
            self.assertNotIn(flag, lines)

    def test_cuda_graph_false_adds_the_disable_flags(self):
        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir), cuda_graph=False),
            job="999921"))
        for flag in self.CG_FLAGS:
            self.assertIn(flag, lines)

    def test_env_contradicting_cuda_graph_true_is_refused(self):
        result = self.run_chain(
            self.campaign(model=str(self.model_dir)),
            {"PDMUX_DISABLE_CUDA_GRAPH": "1"}, job="999922")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("contradicts", result.stderr)

    def test_env_contradicting_cuda_graph_false_is_refused(self):
        result = self.run_chain(
            self.campaign(model=str(self.model_dir), cuda_graph=False),
            {"PDMUX_DISABLE_CUDA_GRAPH": "0"}, job="999923")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("contradicts", result.stderr)

    def test_mutant_ignoring_the_field_makes_it_decoration_again(self):
        """★NEGATIVE CONTROL: without the read, cuda_graph=False changes nothing.

        Both the sbatch and the runner read it, so both reads are removed; if
        either survives the field is still load-bearing and the control is
        meaningless.
        """
        def mutate(text: str) -> str:
            old = 'record_cuda_graph="${record_fields[3]}"'
            self.assertIn(old, text)
            return text.replace(old, 'record_cuda_graph=""')

        runner_text = RUNNER.read_text(encoding="utf-8")
        old_runner = 'record_cuda_graph="${fields[8]}"'
        self.assertIn(old_runner, runner_text)
        stub_runner = self.tmp / "runner_mutant.sh"
        stub_runner.write_text(
            runner_text.replace(old_runner, 'record_cuda_graph=""'),
            encoding="utf-8",
        )
        stub_runner.chmod(0o755)

        lines = self.served(self.run_chain(
            self.campaign(model=str(self.model_dir), cuda_graph=False),
            {"PDMUX_ENGINE_BENCH_RUNNER": str(stub_runner)},
            mutate=mutate, job="999924"))
        for flag in self.CG_FLAGS:
            self.assertNotIn(
                flag, lines,
                "with both reads removed the field must go back to decoration",
            )


class TestRunRecordIsCopiedVerbatim(_DryRunDriver):
    def test_run_json_keeps_the_declared_fields(self):
        campaign_path = self.campaign(model=str(self.model_dir),
                                      context_length=8192)
        self.served(self.run_chain(campaign_path, job="999930"))
        record = json.loads(
            (Path(self.env["PDMUX_RESULT_ROOT"]) / "job_999930" / "run_0"
             / "run.json").read_text()
        )
        self.assertEqual(record["model"], str(self.model_dir))
        self.assertEqual(record["context_length"], 8192)
        self.assertTrue(record["cuda_graph"])

    def test_server_args_txt_records_the_realized_tuple(self):
        campaign_path = self.campaign(model=str(self.model_dir),
                                      context_length=8192)
        self.served(self.run_chain(campaign_path, job="999931"))
        args = (Path(self.env["PDMUX_RESULT_ROOT"]) / "job_999931" / "run_0"
                / "server_args.txt").read_text().splitlines()
        self.assertEqual(args[args.index("--model-path") + 1],
                         str(self.model_dir))
        self.assertEqual(args[args.index("--context-length") + 1], "8192")


if __name__ == "__main__":
    unittest.main()
