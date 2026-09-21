"""Per-shape lambda*: the campaign may not invent the capacity it scales by.

WHAT THIS FILE PINS (workstream C, 2026-09-13, GPU 0)
-----------------------------------------------------
(C1) FAIL-CLOSED.  `PDMUX_SUSTAINABLE_RATE` defaulted to 4 req/s and had never
     been measured on any model.  Every workload is a FRACTION of lambda*, so
     with that default "W8 near saturation" and "W9 overload" were assertions
     about a capacity nobody had measured (gate #6).  There is now no default:
     a missing table is an error, and the removed scalar is refused by name.

(C2) PER-PHASE lambda* (user decision (a) on EXPERIMENT_ROADMAP "P2").  W4's
     phases are (8192,64) and (256,512); their capacities differ ~5x, so one
     scalar cannot put both at 0.80x -- 0.675 gives prefill 0.79x and decode
     0.21-0.25x, 2.1 gives decode 0.79x and prefill 2.47x.  Each phase now
     scales by its OWN shape's rate.  `test_one_scalar_cannot_...` reproduces
     the pre-revision behaviour and shows it missing the target, so the repair
     is not certified by a test that would also pass without it (lesson #53).

(C3) UNMEASURED SHAPES ARE RECORDED AND REFUSED.  Only (256,512) and (8192,64)
     are slated for measurement; the other five shapes belong to workloads that
     must not run silently.  The refusal is repeated at three layers -- trace
     generation, campaign build, and the sbatch before it spends an allocation
     -- and each layer is shown to be load-bearing by a mutant that removes it.

(C4) THE CITED SOURCE LINES DO NOT MOVE.  Four canonical documents cite
     `workloads.py:159-160` for "W4's decode phase is (256,512), not (64,512)".
     This revision was written to keep those lines in place; the pin below
     turns a future accidental shift into a failing test instead of silent
     citation drift (lesson #80).

NOTHING HERE MEASURES ANYTHING OR SUPPORTS ANY PERFORMANCE CLAIM.  lambda* is
still unmeasured for every shape; this file only pins how it must be supplied.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[1]
BENCHMARKS = TRACK_ROOT / "benchmarks"
SBATCH = TRACK_ROOT / "scripts" / "r2_eval" / "r2_eval.sbatch"
GENERATE = TRACK_ROOT / "scripts" / "r2_eval" / "generate_campaign.sh"
EXAMPLE_TABLE = TRACK_ROOT / "scripts" / "r2_eval" / "lambda_star.example.json"
WORKLOADS_PY = BENCHMARKS / "pdmux_eval" / "workloads.py"
ENGINE_DEV = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(TRACK_ROOT.parents[2])),
                     "sglang_engine_dev", "python")
)

sys.path.insert(0, str(BENCHMARKS))
from pdmux_eval import campaign, lambda_star, workloads   # noqa: E402

# The two shapes campaign stage 0 is slated to measure.  A = every W3 request
# and W4's decode phase; B = W2 and W4's prefill phase.
SHAPE_A = (256, 512)
SHAPE_B = (8192, 64)
# Numbers used ONLY to exercise arithmetic.  They are not measurements: 0.675 is
# job 905835's d44 knee for a DIFFERENT shape (8192,96) and 2.1 is an
# extrapolation, both `unmeasured` in the shipped template.
RATE_A = 2.1
RATE_B = 0.675


def table(rates, source="measured", definition="slo_sustainable"):
    return lambda_star.parse(
        {
            "schema": lambda_star.SCHEMA,
            "definition": definition,
            "measured_by": "unit-test fixture, not a measurement",
            "shapes": {
                lambda_star.shape_key(shape): {
                    "req_per_s": rate,
                    "source": source,
                    "evidence": "fixture",
                }
                for shape, rate in rates.items()
            },
        }
    )


def phase_rates(requests):
    """Realized arrivals/second inside each 30 s W4 phase."""
    counts = {}
    for request in requests:
        key = (request.phase, int(request.arrival_s // 30.0))
        counts[key] = counts.get(key, 0) + 1
    rates = {"prefill": [], "decode": []}
    for (phase, _), count in counts.items():
        rates[phase].append(count / 30.0)
    return rates


class ShapeMapIsDerivedFromTheGenerator(unittest.TestCase):
    """Gate #198: read shapes off the generator, never off the spec prose."""

    def setUp(self):
        every_shape = {
            (2048, 128): 4.0, (8192, 64): 4.0, (256, 512): 4.0,
            (1024, 128): 4.0, (8192, 128): 4.0, (2048, 32): 4.0,
            (2048, 512): 4.0,
        }
        self.table = table(every_shape)

    def test_declared_shapes_equal_generated_shapes_for_all_nine(self):
        for workload in sorted(workloads.builtin_workloads()):
            with self.subTest(workload=workload):
                emitted = {
                    (item.input_tokens, item.output_tokens)
                    for item in workloads.generate_requests(
                        workload, self.table, 256, seed=1
                    )
                }
                self.assertEqual(set(workloads.workload_shapes(workload, 256)), emitted)

    def test_w4_decode_phase_is_256x512_not_the_64x512_phantom(self):
        """The 2026-09-13(3) canonical correction, asserted against the code."""
        emitted = {
            item.phase: (item.input_tokens, item.output_tokens)
            for item in workloads.generate_requests("W4", self.table, 256, seed=1)
        }
        self.assertEqual(emitted["prefill"], SHAPE_B)
        self.assertEqual(emitted["decode"], SHAPE_A)
        self.assertEqual(
            emitted["decode"],
            workloads.workload_shapes("W3")[0],
            "W4's decode phase IS W3's shape -- not an approximation of it",
        )
        every = {
            shape
            for workload in workloads.builtin_workloads()
            for shape in workloads.workload_shapes(workload, 256)
        }
        self.assertNotIn((64, 512), every, "the (64,512) shape does not exist")

    def test_rate_derivation_is_declared_per_workload(self):
        self.assertEqual(workloads.workload_rate_derivation("W4"), "per_phase")
        for workload in ("W1", "W2", "W3", "W7", "W8", "W9"):
            self.assertEqual(
                workloads.workload_rate_derivation(workload), "single_shape"
            )
        for workload in ("W5", "W6"):
            self.assertEqual(
                workloads.workload_rate_derivation(workload),
                "interleaved_mixture_harmonic",
                "two shapes in one Poisson stream have no per-shape rate",
            )


class PerPhaseLambdaStar(unittest.TestCase):
    def test_each_w4_phase_hits_080_of_its_own_capacity(self):
        requests = workloads.generate_requests(
            "W4", table({SHAPE_A: RATE_A, SHAPE_B: RATE_B}), 256, seed=1
        )
        rates = phase_rates(requests)
        for realized in rates["prefill"]:
            self.assertAlmostEqual(realized / RATE_B, 0.80, delta=0.03)
        for realized in rates["decode"]:
            self.assertAlmostEqual(realized / RATE_A, 0.80, delta=0.03)

    def test_one_scalar_cannot_put_both_phases_at_080(self):
        """★NEGATIVE CONTROL: the pre-revision parameterisation, re-created.

        Feeding the SAME rate to both shapes is exactly what the single
        `PDMUX_SUSTAINABLE_RATE` did.  The assertion below must fail for one
        phase whichever scalar is chosen -- otherwise the per-phase repair
        above would be certified by an experiment that never needed it.
        """
        for scalar, off_phase, off_shape in (
            (RATE_B, "decode", SHAPE_A),
            (RATE_A, "prefill", SHAPE_B),
        ):
            with self.subTest(scalar=scalar):
                requests = workloads.generate_requests(
                    "W4", table({SHAPE_A: scalar, SHAPE_B: scalar}), 256, seed=1
                )
                true_rate = {SHAPE_A: RATE_A, SHAPE_B: RATE_B}[off_shape]
                realized = phase_rates(requests)[off_phase][0] / true_rate
                self.assertNotAlmostEqual(realized, 0.80, delta=0.03)
                self.assertTrue(
                    realized < 0.30 or realized > 2.0,
                    f"{off_phase} landed at {realized:.2f}x its own capacity",
                )

    def test_w2_scales_by_the_8192x64_rate(self):
        requests = workloads.generate_requests(
            "W2", table({SHAPE_A: RATE_A, SHAPE_B: RATE_B}), 256, seed=1
        )
        self.assertEqual(
            {(r.input_tokens, r.output_tokens) for r in requests}, {SHAPE_B}
        )
        count = sum(1 for r in requests if r.phase == "burst-0")
        # Integer rounding dominates at this rate: 4 x 0.20 x 0.675 x 10 s =
        # 5.4 requests -> 5, i.e. 0.74x rather than 0.80x.  Assert the exact
        # intended arithmetic, and the rate only to the width rounding allows.
        self.assertEqual(count, round(4.0 * 0.20 * RATE_B * 10.0))
        self.assertAlmostEqual((count / 10.0) / RATE_B, 0.80, delta=0.08)

    def test_single_shape_stream_uses_its_own_rate(self):
        requests = workloads.generate_requests(
            "W3", table({SHAPE_A: RATE_A}), 256, seed=1
        )
        span = max(r.arrival_s for r in requests) - min(r.arrival_s for r in requests)
        self.assertAlmostEqual(len(requests) / span / RATE_A, 0.80, delta=0.10)

    def test_mixture_of_equal_rates_is_that_rate(self):
        mixed = table({(1024, 128): 3.0, (8192, 128): 3.0})
        self.assertAlmostEqual(mixed.mixture_rate([(1024, 128), (8192, 128)]), 3.0)

    def test_mixture_is_harmonic_not_arithmetic(self):
        mixed = table({(1024, 128): 4.0, (8192, 128): 1.0})
        # 1 / ((1/4 + 1/1) / 2) = 1.6, not the arithmetic mean 2.5.
        self.assertAlmostEqual(
            mixed.mixture_rate([(1024, 128), (8192, 128)]), 1.6, places=6
        )


class FailClosed(unittest.TestCase):
    def test_generate_requests_refuses_a_bare_scalar(self):
        with self.assertRaises(TypeError) as caught:
            workloads.generate_requests("W3", 4.0, 16, seed=1)
        self.assertIn("campaign stage 0", str(caught.exception))
        self.assertIn("캠페인 0단계", str(caught.exception))

    def test_missing_shape_in_the_table_is_a_hard_error(self):
        partial = table({SHAPE_A: RATE_A, SHAPE_B: RATE_B})
        with self.assertRaises(KeyError) as caught:
            workloads.generate_requests("W1", partial, 16, seed=1)
        self.assertIn("2048x128", str(caught.exception))

    def test_table_rejects_a_missing_source(self):
        with self.assertRaises(ValueError):
            lambda_star.parse(
                {
                    "schema": lambda_star.SCHEMA,
                    "definition": "slo_sustainable",
                    "measured_by": "x",
                    "shapes": {"256x512": {"req_per_s": 1.0, "evidence": "e"}},
                }
            )

    def test_table_rejects_an_unknown_key_instead_of_defaulting(self):
        with self.assertRaises(ValueError) as caught:
            lambda_star.parse(
                {
                    "schema": lambda_star.SCHEMA,
                    "definition": "slo_sustainable",
                    "measured_by": "x",
                    "shapes": {
                        "256x512": {
                            "req_per_s": 1.0,
                            "sources": "measured",
                            "evidence": "e",
                        }
                    },
                }
            )
        self.assertIn("sources", str(caught.exception))

    def test_table_rejects_an_unknown_definition(self):
        with self.assertRaises(ValueError):
            lambda_star.parse(
                {
                    "schema": lambda_star.SCHEMA,
                    "definition": "whatever_we_measured",
                    "measured_by": "x",
                    "shapes": {
                        "256x512": {
                            "req_per_s": 1.0,
                            "source": "measured",
                            "evidence": "e",
                        }
                    },
                }
            )

    def test_shipped_template_loads_and_is_entirely_unmeasured(self):
        shipped = lambda_star.load(EXAMPLE_TABLE)
        self.assertEqual(
            {lambda_star.shape_key(s) for s in shipped.entries},
            {"256x512", "8192x64"},
        )
        for shape in shipped.entries:
            self.assertEqual(shipped.source(shape), "unmeasured")


class ProvenanceRecord(unittest.TestCase):
    def test_measured_single_shape_workload_has_no_caveats(self):
        record = workloads.lambda_star_record("W3", table({SHAPE_A: RATE_A}))
        self.assertEqual(record["source"], "measured")
        self.assertEqual(record["caveats"], [])
        self.assertEqual(record["rate_derivation"], "single_shape")
        self.assertEqual(record["shapes"]["256x512"]["req_per_s"], RATE_A)

    def test_unmeasured_shape_becomes_an_unmeasured_rollup(self):
        record = workloads.lambda_star_record(
            "W3", table({SHAPE_A: RATE_A}, source="unmeasured")
        )
        self.assertEqual(record["source"], "unmeasured")
        self.assertIn("unmeasured_lambda_star:256x512", record["caveats"])

    def test_throughput_definition_is_a_caveat_even_when_measured(self):
        """Declaring the knee does not satisfy gate #6 (verdict 3(1b))."""
        record = workloads.lambda_star_record(
            "W3", table({SHAPE_A: RATE_A}, definition="throughput_saturation")
        )
        self.assertEqual(record["source"], "unmeasured")
        self.assertIn(
            "lambda_star_definition:throughput_saturation", record["caveats"]
        )

    def test_interleaved_workloads_disclose_the_undecided_derivation(self):
        record = workloads.lambda_star_record(
            "W5", table({(1024, 128): 1.0, (8192, 128): 0.5})
        )
        self.assertEqual(record["source"], "unmeasured")
        self.assertIn(
            "undecided_rate_derivation:interleaved_mixture_harmonic",
            record["caveats"],
        )

    def test_w4_record_carries_both_phase_shapes(self):
        record = workloads.lambda_star_record(
            "W4", table({SHAPE_A: RATE_A, SHAPE_B: RATE_B})
        )
        self.assertEqual(set(record["shapes"]), {"256x512", "8192x64"})
        self.assertEqual(record["rate_derivation"], "per_phase")
        self.assertEqual(record["source"], "measured")


class _TraceFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.trace_dir = self.tmp / "traces"

    def write(self, workload="W3", rates=None, source="measured", block="record"):
        rates = rates or {SHAPE_A: RATE_A, SHAPE_B: RATE_B}
        built = table(rates, source=source)
        record = None
        if block == "record":
            record = workloads.lambda_star_record(workload, built)
        workloads.write_trace(
            self.trace_dir / f"{workload}.jsonl",
            workloads.builtin_workloads()[workload],
            workloads.generate_requests(workload, built, 16, seed=1),
            lambda_star=record,
        )
        return self.trace_dir / f"{workload}.jsonl"


class CampaignReadsTheTraceHeader(_TraceFixture):
    def test_measured_trace_builds_and_stamps_every_run(self):
        self.write()
        runs = campaign.build_runs(["W3"], ["B1", "B4"], self.trace_dir, 5, seed=1)
        self.assertTrue(runs)
        self.assertEqual({run.lambda_star_source for run in runs}, {"measured"})

    def test_unmeasured_trace_is_refused(self):
        self.write(source="unmeasured")
        with self.assertRaises(ValueError) as caught:
            campaign.build_runs(["W3"], ["B1"], self.trace_dir, 5, seed=1)
        self.assertIn("REFUSED", str(caught.exception))
        self.assertIn("campaign stage 0", str(caught.exception))

    def test_unmeasured_trace_passes_with_the_optin_and_is_recorded(self):
        self.write(source="unmeasured")
        runs = campaign.build_runs(
            ["W3"], ["B1"], self.trace_dir, 5, seed=1,
            allow_unmeasured_lambda_star=True,
        )
        self.assertEqual({run.lambda_star_source for run in runs}, {"unmeasured"})

    def test_trace_without_a_header_block_is_undeclared_not_fine(self):
        self.write(block="none")
        with self.assertRaises(ValueError) as caught:
            campaign.build_runs(["W3"], ["B1"], self.trace_dir, 5, seed=1)
        self.assertIn("undeclared", str(caught.exception))

    def test_mutant_ignoring_the_header_lets_an_unmeasured_trace_through(self):
        """★NEGATIVE CONTROL: without the read, the refusal is unreachable."""
        self.write(source="unmeasured")
        original = campaign.trace_lambda_star
        campaign.trace_lambda_star = lambda path: {"source": "measured"}
        try:
            runs = campaign.build_runs(["W3"], ["B1"], self.trace_dir, 5, seed=1)
        finally:
            campaign.trace_lambda_star = original
        self.assertEqual({run.lambda_star_source for run in runs}, {"measured"})


class CliSurfaces(_TraceFixture):
    def env(self):
        return {**os.environ, "PYTHONPATH": str(BENCHMARKS)}

    def run_workload_cli(self, *extra, table_path=None):
        table_path = table_path or EXAMPLE_TABLE
        return subprocess.run(
            [sys.executable, "-m", "pdmux_eval.workloads", "--workload", "W3",
             "--lambda-star-table", str(table_path), "--count", "8",
             "--output", str(self.tmp / "out.jsonl"), *extra],
            capture_output=True, text=True, env=self.env(),
        )

    def test_workloads_cli_requires_a_table(self):
        result = subprocess.run(
            [sys.executable, "-m", "pdmux_eval.workloads", "--workload", "W3",
             "--output", str(self.tmp / "out.jsonl")],
            capture_output=True, text=True, env=self.env(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--lambda-star-table", result.stderr)

    def test_workloads_cli_refuses_the_removed_scalar_flag(self):
        result = self.run_workload_cli("--sustainable-rate", "4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removed", result.stderr)

    def test_workloads_cli_refuses_the_unmeasured_template(self):
        result = self.run_workload_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSED", result.stderr)
        self.assertFalse((self.tmp / "out.jsonl").exists())

    def test_workloads_cli_optin_writes_the_caveats_into_the_header(self):
        result = self.run_workload_cli("--allow-unmeasured-lambda-star")
        self.assertEqual(result.returncode, 0, result.stderr)
        header = json.loads(
            (self.tmp / "out.jsonl").read_text().splitlines()[0]
        )
        self.assertEqual(header["lambda_star"]["source"], "unmeasured")
        self.assertIn(
            "unmeasured_lambda_star:256x512", header["lambda_star"]["caveats"]
        )

    def test_campaign_json_mirrors_the_table_and_bumps_the_schema(self):
        self.write()
        out = self.tmp / "campaign.json"
        result = subprocess.run(
            [sys.executable, "-m", "pdmux_eval.campaign", "--trace-dir",
             str(self.trace_dir), "--workloads", "W3", "--baselines", "B1",
             "--repetitions", "5", "--output", str(out)],
            capture_output=True, text=True, env=self.env(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(out.read_text())
        self.assertEqual(data["schema"], "pdmux.campaign/v3")
        self.assertEqual(data["lambda_star"]["W3"]["source"], "measured")
        self.assertEqual(
            data["lambda_star"]["W3"]["shapes"]["256x512"]["req_per_s"], RATE_A
        )
        self.assertTrue(
            all(run["lambda_star_source"] == "measured" for run in data["runs"])
        )

    def test_campaign_cli_refuses_an_unmeasured_trace(self):
        self.write(source="unmeasured")
        out = self.tmp / "campaign_bad.json"
        result = subprocess.run(
            [sys.executable, "-m", "pdmux_eval.campaign", "--trace-dir",
             str(self.trace_dir), "--workloads", "W3", "--baselines", "B1",
             "--repetitions", "5", "--output", str(out)],
            capture_output=True, text=True, env=self.env(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSED", result.stderr)
        self.assertFalse(out.exists())


class ShellRefusals(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_generate(self, **extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("PDMUX_LAMBDA_STAR_TABLE", "PDMUX_SUSTAINABLE_RATE")}
        env.update(extra)
        return subprocess.run(
            ["/usr/bin/bash", str(GENERATE), str(self.tmp / "traces"),
             str(self.tmp / "campaign.json")],
            capture_output=True, text=True, env=env, timeout=300,
        )

    def test_generate_campaign_refuses_without_a_table(self):
        result = self.run_generate()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("PDMUX_LAMBDA_STAR_TABLE is not set", result.stderr)
        self.assertIn("캠페인 0단계", result.stderr)
        self.assertFalse((self.tmp / "campaign.json").exists())

    def test_generate_campaign_refuses_the_removed_scalar(self):
        result = self.run_generate(
            PDMUX_LAMBDA_STAR_TABLE=str(EXAMPLE_TABLE),
            PDMUX_SUSTAINABLE_RATE="4",
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("was removed", result.stderr)

    def test_generate_campaign_refuses_the_unmeasured_template(self):
        result = self.run_generate(
            PDMUX_LAMBDA_STAR_TABLE=str(EXAMPLE_TABLE),
            PDMUX_WORKLOADS="W3 W4",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSED", result.stderr)

    def test_generate_campaign_runs_w3_w4_on_a_measured_table(self):
        measured = self.tmp / "measured.json"
        measured.write_text(
            json.dumps(
                {
                    "schema": lambda_star.SCHEMA,
                    "definition": "slo_sustainable",
                    "measured_by": "fixture, not a measurement",
                    "shapes": {
                        "256x512": {"req_per_s": RATE_A, "source": "measured",
                                    "evidence": "fixture"},
                        "8192x64": {"req_per_s": RATE_B, "source": "measured",
                                    "evidence": "fixture"},
                    },
                }
            ),
            encoding="utf-8",
        )
        result = self.run_generate(
            PDMUX_LAMBDA_STAR_TABLE=str(measured),
            PDMUX_WORKLOADS="W3 W4",
            PDMUX_REQUEST_COUNT="8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads((self.tmp / "campaign.json").read_text())
        self.assertEqual(set(data["lambda_star"]), {"W3", "W4"})
        self.assertEqual(
            data["lambda_star"]["W4"]["rate_derivation"], "per_phase"
        )
        self.assertEqual({run["workload"] for run in data["runs"]}, {"W3", "W4"})


class SbatchReadsTheField(unittest.TestCase):
    """The last refusal before an allocation is spent, plus its mutant."""

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
        self.trace_dir = self.tmp / "traces"
        built = table({SHAPE_A: RATE_A})
        workloads.write_trace(
            self.trace_dir / "W3.jsonl",
            workloads.builtin_workloads()["W3"],
            workloads.generate_requests("W3", built, 4, seed=1),
            lambda_star=workloads.lambda_star_record("W3", built),
        )
        self.env = dict(os.environ)
        self.env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + self.env["PATH"]
        self.env["PDMUX_RESULT_ROOT"] = str(self.tmp / "out")
        self.env["PDMUX_DRY_RUN"] = "1"
        for var in ("PDMUX_MODEL", "PDMUX_CONTEXT_LENGTH", "PDMUX_DISABLE_CUDA_GRAPH",
                    "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID",
                    "PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR"):
            self.env.pop(var, None)

    def campaign_json(self, source):
        runs = campaign.build_runs(
            ["W3"], ["B4"], self.trace_dir, 5, seed=1,
            model=str(self.model_dir), allow_unmeasured_lambda_star=True,
        )
        records = []
        for run in runs:
            record = dict(run.__dict__)
            record["lambda_star_source"] = source
            records.append(record)
        path = self.tmp / f"campaign_{source}.json"
        path.write_text(
            json.dumps({"schema": "pdmux.campaign/v3", "runs": records}),
            encoding="utf-8",
        )
        return path

    def run_chain(self, campaign_path, job, env_extra=None, mutate=None):
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
            ["/usr/bin/bash", str(script)], cwd="/", env=env,
            capture_output=True, text=True, timeout=900,
        )

    def test_measured_record_boots(self):
        result = self.run_chain(self.campaign_json("measured"), "999940")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--enable-pdmux", result.stdout.splitlines())

    def test_unmeasured_record_is_refused_before_any_boot(self):
        result = self.run_chain(self.campaign_json("unmeasured"), "999941")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("lambda_star_source=unmeasured", result.stderr)
        self.assertNotIn("--enable-pdmux", result.stdout.splitlines())

    def test_unmeasured_record_boots_with_the_explicit_optin(self):
        result = self.run_chain(
            self.campaign_json("unmeasured"), "999942",
            {"PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--enable-pdmux", result.stdout.splitlines())

    def test_mutant_dropping_the_read_makes_the_field_decoration_again(self):
        """★NEGATIVE CONTROL for test_unmeasured_record_is_refused."""
        def mutate(text):
            old = 'record_lambda_star_source="${record_fields[4]}"'
            self.assertIn(old, text)
            return text.replace(old, 'record_lambda_star_source=""')

        result = self.run_chain(
            self.campaign_json("unmeasured"), "999943", mutate=mutate
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "--enable-pdmux", result.stdout.splitlines(),
            "with the read removed the field must stop gating anything",
        )


# ★The pre-revision reference is a FIXED COMMIT, not `HEAD` (2026-09-14,
# engine-porter).  It was written as `HEAD:` while the revision was still
# uncommitted, and the moment commit `8507cee` landed, `HEAD` BECAME the
# post-revision file -- whose first statement is `from .lambda_star import ...`,
# which `exec()` of a bare module cannot resolve.  Both controls in this class
# then raised `ImportError: attempted relative import with no known parent
# package` in `unittest discover` (measured: 645 tests, 2 errors).  A control
# anchored to a moving reference destroys itself at the instant the change it
# controls for is recorded; the `try/except TypeError -> skipTest` guards below
# were meant to catch that transition and could not, because the module now dies
# before any call.  `bddff6a` is the last commit whose `workloads.py` takes the
# scalar `sustainable_rate`.
PRE_REVISION_COMMIT = "bddff6a"


class NoUnintendedBehaviourChange(unittest.TestCase):
    """The revision must be a no-op when every shape shares one lambda*."""

    def old_module(self):
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "show",
             f"{PRE_REVISION_COMMIT}:workspace/engine-port/benchmarks/"
             "pdmux_eval/workloads.py"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(
                f"git {PRE_REVISION_COMMIT} copy of workloads.py unavailable")
        module = types.ModuleType("_pre_revision_workloads")
        sys.modules["_pre_revision_workloads"] = module
        self.addCleanup(sys.modules.pop, "_pre_revision_workloads", None)
        exec(compile(result.stdout, f"<{PRE_REVISION_COMMIT} workloads.py>",
                     "exec"), module.__dict__)
        return module

    def test_shared_rate_reproduces_the_pre_revision_traces_exactly(self):
        old = self.old_module()
        if not hasattr(old, "generate_requests"):
            self.skipTest("HEAD copy has no generate_requests")
        try:
            baseline = old.generate_requests("W3", 4.0, 8, seed=7)
        except TypeError:
            self.skipTest("HEAD already takes a table; equivalence already pinned")
        shared = table(
            {
                (2048, 128): 4.0, (8192, 64): 4.0, (256, 512): 4.0,
                (1024, 128): 4.0, (8192, 128): 4.0, (2048, 32): 4.0,
                (2048, 512): 4.0,
            }
        )
        self.assertTrue(baseline)
        for workload in sorted(workloads.builtin_workloads()):
            with self.subTest(workload=workload):
                before = old.generate_requests(workload, 4.0, 64, seed=7)
                after = workloads.generate_requests(workload, shared, 64, seed=7)
                self.assertEqual(
                    [tuple(item.__dict__.values()) for item in before],
                    [tuple(item.__dict__.values()) for item in after],
                )

    def test_pre_revision_w4_misses_its_decode_target(self):
        """★FAIL-BEFORE, against the real pre-revision code (not a re-creation)."""
        old = self.old_module()
        try:
            requests = old.generate_requests("W4", RATE_B, 256, seed=1)
        except TypeError:
            self.skipTest("HEAD already takes a table")
        decode = [r for r in requests if r.phase == "decode"]
        self.assertEqual(
            {(r.input_tokens, r.output_tokens) for r in decode}, {SHAPE_A}
        )
        realized = (len(decode) / 3) / 30.0 / RATE_A
        self.assertLess(
            realized, 0.30,
            "the single-scalar generator put W4's decode phase far below 0.80x",
        )


class CitedLinesArePinned(unittest.TestCase):
    """Lesson #80: a citation goes stale between verification and reading."""

    CITED = {
        148: 'add(arrivals, 8192, 64, f"burst-{cycle}")',
        159: "8192 if is_prefill else 256,",
        160: "64 if is_prefill else 512,",
    }

    def test_workloads_py_cited_lines_did_not_move(self):
        lines = WORKLOADS_PY.read_text(encoding="utf-8").splitlines()
        for number, expected in self.CITED.items():
            with self.subTest(line=number):
                self.assertEqual(
                    lines[number - 1].strip(), expected,
                    f"workloads.py:{number} is cited by PROJECT_STATUS.md, "
                    "reports/CONSENSUS.md, reports/paper/EXPERIMENT_ROADMAP.md, "
                    "reports/paper/CLAIM_EVIDENCE_MATRIX.md and two verdicts "
                    "(the 2026-09-13(3) 'W4 decode phase is (256,512)' "
                    "correction).  Moving it silently falsifies those "
                    "documents: update them in the same change, then update "
                    "this pin.",
                )


if __name__ == "__main__":
    unittest.main()
