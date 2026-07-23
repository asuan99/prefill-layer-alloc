import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from pdmux_eval import analyze, campaign, trace_loadgen, workloads


class WorkloadTest(unittest.TestCase):
    def test_trace_generation_is_deterministic(self):
        first = workloads.generate_requests("W3", 4.0, 16, seed=7)
        second = workloads.generate_requests("W3", 4.0, 16, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all(item.input_tokens == 256 for item in first))
        self.assertTrue(all(item.output_tokens == 512 for item in first))

    def test_campaign_pairs_share_trace_and_server_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "W3.jsonl"
            workloads.write_trace(
                trace,
                workloads.builtin_workloads()["W3"],
                workloads.generate_requests("W3", 4.0, 8, seed=1),
            )
            runs = campaign.build_runs(["W3"], ["B1", "B6"], root, 5, seed=3)
        self.assertEqual(len(runs), 10)
        for repetition in range(5):
            pair = [run for run in runs if run.repetition == repetition]
            self.assertEqual(len({run.server_seed for run in pair}), 1)
            self.assertEqual(len({run.trace_sha256 for run in pair}), 1)

    def test_trace_loadgen_ignores_metadata_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                '{"record":"workload","workload_id":"W1"}\n'
                '{"record":"request","request_id":"r0","arrival_s":0,'
                '"input_tokens":2,"output_tokens":3,"context_tokens":2,'
                '"phase":"steady"}\n'
            )
            requests = trace_loadgen.read_requests(path)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["request_id"], "r0")


class AnalysisTest(unittest.TestCase):
    def test_primary_goodput_uses_request_token_itl_p95(self):
        results = [
            analyze.RequestResult("good", 100, (10, 20, 30), 1.0),
            analyze.RequestResult("bad_tail", 100, (10,) * 18 + (100, 100), 2.0),
        ]
        summary = analyze.summarize_requests(results, 0, 2, 3000, 60)
        self.assertEqual(summary["slo_goodput_req_s"], 0.5)
        self.assertEqual(summary["legacy_mean_itl_goodput_req_s"], 1.0)

    def test_paired_bootstrap_reports_effect(self):
        result = analyze.paired_bootstrap_ci(
            {"r1": 10.0, "r2": 11.0, "r3": 9.0},
            {"r1": 11.0, "r2": 12.0, "r3": 10.0},
            samples=1000,
        )
        self.assertAlmostEqual(result["mean_effect"], 1.0)
        self.assertGreater(result["ci95_low"], 0)

    def test_residency_is_time_weighted_not_sample_count(self):
        events = [
            {
                "event": "runtime_snapshot",
                "phase": "benchmark",
                "timestamp_monotonic_s": 0.0,
                "decode_sms": 24,
            },
            {
                "event": "runtime_snapshot",
                "phase": "benchmark",
                "timestamp_monotonic_s": 9.0,
                "decode_sms": 44,
            },
            {
                "event": "runtime_snapshot",
                "phase": "benchmark",
                "timestamp_monotonic_s": 10.0,
                "decode_sms": 44,
            },
        ]
        summary = analyze.controller_summary(events)
        self.assertEqual(summary["split_transitions"], 1)
        self.assertAlmostEqual(summary["residency_fraction"][24], 0.9)

    def test_static_selection_and_trace_oracle(self):
        rows = [
            {"workload": "W1", "decode_sms": 24, "slo_goodput_req_s": 8},
            {"workload": "W1", "decode_sms": 44, "slo_goodput_req_s": 7},
            {"workload": "W3", "decode_sms": 24, "slo_goodput_req_s": 2},
            {"workload": "W3", "decode_sms": 44, "slo_goodput_req_s": 9},
        ]
        selected = analyze.select_static_baselines(rows)
        self.assertEqual(selected["global_decode_sms"], 44)
        self.assertEqual(selected["per_workload_decode_sms"]["W1"], 24)
        oracle = analyze.trace_aware_oracle(
            [{24: 10, 44: 1}, {24: 1, 44: 10}],
            states=(24, 44),
            transition_cost=1,
        )
        self.assertEqual(oracle["schedule"], [24, 44])
        self.assertEqual(oracle["total_reward"], 19)


if __name__ == "__main__":
    unittest.main()
