"""engine_bench_runner.sh multi-GPU options + multigpu_cell_meta.py (multi-GPU track, M2).

All CPU: the runner is driven with PDMUX_DRY_RUN=1 (stops after writing server_args.txt and
comm_env.json) against a fake empty venv.  Every behaviour is paired with its negative control.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
RUNNER = ENGINE / "scripts" / "r2_eval" / "engine_bench_runner.sh"
sys.path.insert(0, str(ENGINE / "scripts" / "r2_eval"))
import multigpu_cell_meta as M  # noqa: E402

# argv of a single-GPU dry run recorded from this runner at git HEAD 20e184d (before the change)
CLEAN_ENV_KEYS = ("PDMUX_TP_SIZE", "PDMUX_DP_SIZE", "PDMUX_GPU_IDS", "PDMUX_COMM_ENV_PIN",
                  "PDMUX_ALLOW_JIT_ALL_REDUCE", "SGLANG_USE_JIT_ALL_REDUCE", "CUDA_VISIBLE_DEVICES",
                  "PDMUX_ENGINE_MODE", "PDMUX_MODEL", "PDMUX_CONTEXT_LENGTH",
                  "PDMUX_DISABLE_CUDA_GRAPH")


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        root = Path(self.td.name)
        (root / "venv" / "bin").mkdir(parents=True)
        (root / "venv" / "bin" / "activate").write_text("")
        self.record = root / "run.json"
        self.record.write_text(json.dumps({
            "trace_path": "t.jsonl", "server_seed": 1, "max_running_requests": 48,
            "ttft_slo_ms": 1000, "itl_slo_ms": 50, "workload": "w"}))
        self.root = root

    def run_runner(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV_KEYS}
        e.update({"PDMUX_DRY_RUN": "1", "PDMUX_VENV": str(self.root / "venv")})
        e.update(env or {})
        out = self.root / f"out{len(list(self.root.glob('out*')))}"
        p = subprocess.run(["bash", str(RUNNER), str(self.record), str(out), *args],
                           env=e, capture_output=True, text=True)
        return p, out

    @staticmethod
    def argv(out):
        return (out / "server_args.txt").read_text().splitlines()

    @staticmethod
    def value(argv, flag):
        return argv[argv.index(flag) + 1]


class SingleGpuUnchanged(RunnerCase):
    def test_no_new_server_args(self):
        p, out = self.run_runner()
        self.assertEqual(p.returncode, 0, p.stderr)
        a = self.argv(out)
        for flag in ("--tp-size", "--dp-size", "--load-balance-method"):
            self.assertNotIn(flag, a)
        self.assertEqual(self.value(a, "--max-running-requests"), "48")
        self.assertIn("--disable-radix-cache", a)

    def test_explicit_n1_equals_default(self):
        _, o1 = self.run_runner()
        _, o2 = self.run_runner("--tp-size", "1", "--dp-size", "1")
        self.assertEqual(self.argv(o1), self.argv(o2))

    def test_comm_env_default_is_unset_record(self):
        _, out = self.run_runner()
        rec = json.loads((out / "comm_env.json").read_text())
        self.assertTrue(all(v["state"] == "unset" and v["value"] is None
                            for v in rec["vars"].values()))
        for name in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS",
                     "CUDA_DEVICE_MAX_CONNECTIONS", "SGLANG_USE_JIT_ALL_REDUCE",
                     "SGLANG_CUSTOM_ALLREDUCE_ALGO"):
            self.assertIn(name, rec["vars"])


class TpDp(RunnerCase):
    def test_tp_cap_is_48_times_n(self):
        for n in (2, 4, 8):
            p, out = self.run_runner("--tp-size", str(n))
            self.assertEqual(p.returncode, 0, p.stderr)
            a = self.argv(out)
            self.assertEqual(self.value(a, "--tp-size"), str(n))
            self.assertEqual(self.value(a, "--max-running-requests"), str(48 * n))
            self.assertNotIn("--dp-size", a)

    def test_dp_cap_stays_48_per_replica_and_round_robin(self):
        p, out = self.run_runner("--dp-size", "4")
        self.assertEqual(p.returncode, 0, p.stderr)
        a = self.argv(out)
        self.assertEqual(self.value(a, "--dp-size"), "4")
        self.assertEqual(self.value(a, "--load-balance-method"), "round_robin")
        self.assertEqual(self.value(a, "--max-running-requests"), "48")
        self.assertIn("--disable-radix-cache", a)

    def test_env_fallback_and_flag_precedence(self):
        _, out = self.run_runner(env={"PDMUX_TP_SIZE": "2"})
        self.assertEqual(self.value(self.argv(out), "--tp-size"), "2")
        _, out = self.run_runner("--tp-size", "4", env={"PDMUX_TP_SIZE": "2"})
        self.assertEqual(self.value(self.argv(out), "--tp-size"), "4")

    def test_rejections(self):
        for args in (["--tp-size", "2", "--dp-size", "2"], ["--tp-size", "0"],
                     ["--dp-size", "x"], ["--bogus"]):
            p, _ = self.run_runner(*args)
            self.assertEqual(p.returncode, 2, args)


class GpuSubset(RunnerCase):
    def test_gpu_ids_must_match_count_and_be_distinct(self):
        p, _ = self.run_runner("--tp-size", "2", "--gpu-ids", "0,1,2")
        self.assertEqual(p.returncode, 2)
        p, _ = self.run_runner("--dp-size", "2", "--gpu-ids", "3,3")
        self.assertEqual(p.returncode, 2)
        p, _ = self.run_runner("--dp-size", "2", "--gpu-ids", "a,b")
        self.assertEqual(p.returncode, 2)
        p, out = self.run_runner("--dp-size", "2", "--gpu-ids", "4,5")
        self.assertEqual(p.returncode, 0, p.stderr)


class CommEnv(RunnerCase):
    def test_jit_all_reduce_refused_for_multi_gpu_unless_allowed(self):
        env = {"SGLANG_USE_JIT_ALL_REDUCE": "1"}
        p, _ = self.run_runner("--tp-size", "2", env=env)
        self.assertEqual(p.returncode, 3)
        p, out = self.run_runner("--tp-size", "2", env=dict(env, PDMUX_ALLOW_JIT_ALL_REDUCE="1"))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(json.loads((out / "comm_env.json").read_text())["jit_all_reduce_enabled"])
        # negative control: unset passes
        p, _ = self.run_runner("--tp-size", "2")
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_pin_is_recorded_as_pinned_and_unregistered_pin_rejected(self):
        p, out = self.run_runner("--tp-size", "2",
                                 env={"PDMUX_COMM_ENV_PIN": "NCCL_ALGO=Ring,NCCL_PROTO=Simple"})
        self.assertEqual(p.returncode, 0, p.stderr)
        v = json.loads((out / "comm_env.json").read_text())["vars"]
        self.assertEqual((v["NCCL_ALGO"]["value"], v["NCCL_ALGO"]["state"]), ("Ring", "pinned"))
        self.assertEqual(v["NCCL_MAX_NCHANNELS"]["state"], "unset")
        with self.assertRaises(ValueError):
            M.parse_pin("SOME_OTHER=1")

    def test_inherited_value_is_labelled(self):
        rec = M.comm_env_record({"NCCL_ALGO": "Tree"})
        self.assertEqual(rec["NCCL_ALGO"]["state"], "inherited")


class ServerFacts(unittest.TestCase):
    LOG = ("max_total_num_tokens=70000, chunked_prefill_size=-1, max_prefill_tokens=16384, "
           "max_running_requests=96, context_len=16384, available_gpu_mem=10.00 GB\n"
           "Mamba Cache is allocated. max_mamba_cache_size: 96, conv_state size: 1.0GB\n"
           "KV Cache is allocated. #tokens: 70000, K size: 2.00 GB, V size: 2.00 GB\n")

    def test_extracts_capacity(self):
        f = M.server_facts(self.LOG, tp=2, dp=1, max_running_requested=96)
        self.assertEqual(f["max_total_num_tokens"], [70000])
        self.assertEqual(f["max_mamba_cache_size"], [96])
        self.assertEqual(f["max_running_requests_logged"], [96])
        self.assertEqual(f["kv_cache_tokens"], [70000])
        self.assertTrue(f["capacity_recorded"] and f["scheduler_lines_match_dp"])

    def test_dp_line_count_mismatch_is_flagged(self):
        f = M.server_facts(self.LOG, tp=1, dp=4, max_running_requested=48)
        self.assertFalse(f["scheduler_lines_match_dp"])

    def test_empty_log_flags_not_recorded(self):
        self.assertFalse(M.server_facts("")["capacity_recorded"])

    def test_server_info_states(self):
        info = {"internal_states": [{"memory_usage": {"token_capacity": 5},
                                     "effective_max_running_requests_per_dp": 48}]}
        f = M.server_facts(self.LOG, info)
        self.assertEqual(f["server_info"]["token_capacity"], [5])
        self.assertEqual(f["server_info"]["effective_max_running_requests_per_dp"], [48])


JOB_ENTRY = ENGINE / "scripts" / "vessl" / "job_entry.sh"


def _heredoc(marker):
    """Python body of the job_entry.sh heredoc whose first line contains `marker`."""
    text = JOB_ENTRY.read_text()
    for m in __import__("re").finditer(r"<<'PY'[^\n]*\n(.*?)\nPY\n", text, __import__("re").S):
        if marker in m.group(1):
            return m.group(1)
    raise AssertionError(marker)


class JobEntrySubstrate(unittest.TestCase):
    """Substrate stamp / clock summary of job_entry.sh on 1 and N GPUs (CPU: heredocs only)."""

    G = "NVIDIA A100-SXM4-80GB, GPU-{u}, 580.105.08, 8.0, 81920 MiB, 1410 MHz, 400.00 W, Disabled\n"
    ENV = {"PDMUX_COMMIT": "c", "PDMUX_IMAGE": "i", "PDMUX_SPEC": "s", "PDMUX_JOB_NAME": "j",
           "PDMUX_CAMPAIGN": "k", "PDMUX_TARGET": "t"}

    def stamp(self, uuids):
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td)
            (meta / "gpu.csv").write_text("".join(self.G.format(u=u) for u in uuids))
            env = dict(os.environ, **self.ENV, PDMUX_OPT=td)
            subprocess.run([sys.executable, "-c", _heredoc("substrate.json"), td], env=env, check=True)
            return json.loads((meta / "substrate.json").read_text())

    def test_one_gpu_keeps_old_fields(self):
        r = self.stamp(["a"])
        self.assertEqual(r["gpu"]["uuid"], "GPU-a")
        self.assertEqual(r["gpu"]["driver_version"], "580.105.08")
        self.assertEqual(r["gpu_count"], 1)
        self.assertEqual(r["gpu_uuids"], ["GPU-a"])
        for k in ("hostname", "commit", "image", "spec", "job_name", "campaign", "target"):
            self.assertIn(k, r)

    def test_all_gpus_recorded(self):
        r = self.stamp(["a", "b", "c", "d"])
        self.assertEqual(r["gpu_count"], 4)
        self.assertEqual(r["gpu_uuids"], ["GPU-a", "GPU-b", "GPU-c", "GPU-d"])
        self.assertEqual([g["index"] for g in r["gpus"]], [0, 1, 2, 3])
        self.assertEqual(r["gpu"]["uuid"], "GPU-a")       # legacy view = first row

    def clk_summary(self, rows):
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td)
            (meta / "clk.csv").write_text(rows)
            subprocess.run([sys.executable, "-c", _heredoc("clk_summary.json"), td], check=True)
            return json.loads((meta / "clk_summary.json").read_text())

    HDR5 = "timestamp, clocks.sm [MHz], power.draw [W], temperature.gpu, clocks_throttle_reasons.active\n"

    def test_clk_summary_old_five_column_layout_unchanged(self):
        s = self.clk_summary(self.HDR5 + "t, 1410 MHz, 300 W, 40, 0x0000000000000001\n"
                             "t, 1200 MHz, 300 W, 40, 0x0000000000000004\n")
        self.assertEqual(s, {"samples": 2, "throttle_active_samples": 1, "throttle_flag": True})

    def test_clk_summary_per_gpu(self):
        s = self.clk_summary(self.HDR5.rstrip() + ", index\n"
                             "t, 1410 MHz, 300 W, 40, 0x1, 0\nt, 1410 MHz, 300 W, 40, 0x1, 1\n"
                             "t, 1100 MHz, 300 W, 40, 0x4, 1\n")
        self.assertEqual((s["samples"], s["throttle_active_samples"]), (3, 1))
        self.assertFalse(s["per_gpu"]["0"]["throttle_flag"])
        self.assertTrue(s["per_gpu"]["1"]["throttle_flag"])
        self.assertEqual(s["per_gpu"]["1"]["samples"], 2)

    def test_clk_query_appends_index_last(self):
        text = JOB_ENTRY.read_text()
        self.assertIn("clocks_throttle_reasons.active,index", text)
        self.assertIn("topo -m", text)


if __name__ == "__main__":
    unittest.main()
