"""CPU regression for the opchar campaign (Bullet Fig 2 / Table 2 / Fig 3 analogue).

Pins the parts that decide numbers without a GPU: wave-quantization math (positive control
= Bullet ASPLOS'26 Table 2), occupancy residency, the Nemotron-Nano-9B-v2 operator inventory
and its analytical work, Kineto range attribution, the run plan, and the Job script guards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ENG = Path(__file__).resolve().parents[1]
_WS = _ENG.parent
_CHAR = _WS / "characterization"
for _p in (str(_WS), str(_CHAR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.common import wave_model as WM  # noqa: E402
from experiments.opchar import ops_spec as S  # noqa: E402
from experiments.opchar import run_opchar as R  # noqa: E402
from experiments.opchar import trace_parse as TP  # noqa: E402


class WaveQuantization(unittest.TestCase):
    def test_bullet_table2_arithmetic_identity(self):
        # Llama-3.1-8B on A100 (108 SM), b=1, 128x256 output tiles. Bullet Table 2 columns
        # QKV (N=6144) and O (N=4096), rows seq 1024/2048/4096/16384, in %.
        # IDENTITY CHECK (claims-auditor 2026-10-13): the tile was CHOSEN so the published table is
        # reproduced, so this pins only the idle-formula arithmetic -- it says nothing about which
        # tile cuBLAS uses (the recorded graph census contradicts the assumed tile on this image).
        bullet = {1024: (11.1, 40.7), 2048: (11.1, 21.0), 4096: (11.1, 5.2), 16384: (1.9, 0.2)}
        for L, (qkv, o) in bullet.items():
            got_qkv = 100 * WM.wave_quantization(WM.gemm_tile_blocks(L, 6144, 128, 256), 108)["idle_ratio"]
            got_o = 100 * WM.wave_quantization(WM.gemm_tile_blocks(L, 4096, 128, 256), 108)["idle_ratio"]
            self.assertAlmostEqual(got_qkv, qkv, delta=0.05, msg=f"QKV L={L}")
            self.assertAlmostEqual(got_o, o, delta=0.05, msg=f"O L={L}")

    def test_edges(self):
        self.assertEqual(WM.wave_quantization(108, 108)["idle_ratio"], 0.0)
        r = WM.wave_quantization(109, 108)
        self.assertEqual((r["waves"], r["tail_sms"]), (2, 1))
        self.assertAlmostEqual(r["idle_ratio"], 107 / 216)
        r = WM.wave_quantization(10, 108, blocks_per_sm=4)    # under one wave
        self.assertEqual(r["waves"], 1)
        self.assertEqual(r["tail_sms"], 3)
        # fewer SMs -> the same grid can quantize differently (the partition axis)
        self.assertNotEqual(WM.wave_quantization(128, 108)["idle_ratio"],
                            WM.wave_quantization(128, 44)["idle_ratio"])
        with self.assertRaises(ValueError):
            WM.wave_quantization(0, 108)


class Occupancy(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(TP.resident_blocks_per_sm(256, 32, 0), 8)       # threads-bound
        self.assertEqual(TP.resident_blocks_per_sm(128, 255, 0), 2)      # register-bound
        self.assertEqual(TP.resident_blocks_per_sm(128, 32, 100 * 1024), 1)  # smem-bound
        self.assertEqual(TP.resident_blocks_per_sm(32, 16, 0), 32)       # block-count cap
        # 33 regs/thread: per-warp allocation rounds 1056 -> 1280 (unit 256): 51 warps -> 6 blocks
        self.assertEqual(TP.resident_blocks_per_sm(256, 33, 0), 6)


class OpsSpec(unittest.TestCase):
    def setUp(self):
        self.c = S.cfg_from_models_yaml("nemotron_nano_9b_v2")

    def test_config_derivations(self):
        c = self.c
        self.assertEqual(S.layer_counts(c.pattern), {"M": 27, "*": 4, "-": 25})
        self.assertEqual(len(c.pattern), 56)
        self.assertEqual(c.mamba_inner, 10240)
        self.assertEqual(c.conv_dim, 12288)
        self.assertEqual(c.in_proj_out, 22656)
        self.assertEqual(c.qkv_out, 7168)

    def test_inventory(self):
        p = S.prefill_ops(self.c, 1024)
        d = S.decode_ops(self.c, 32, 2048)
        self.assertEqual([o.name for o in p["M"]],
                         ["norm", "in_proj", "conv1d", "ssd_scan", "gated_norm", "out_proj"])
        self.assertEqual([o.name for o in d["M"]],
                         ["norm", "in_proj", "conv_update", "ssm_update", "gated_norm", "out_proj"])
        for ops in (p, d):
            self.assertEqual([o.name for o in ops["*"]], ["norm", "qkv_proj", "kv_write", "attn", "o_proj"])
            self.assertEqual([o.name for o in ops["-"]], ["norm", "up_proj", "relu2", "down_proj"])
            keys = S.op_keys(ops)
            self.assertEqual(len(keys), len(set(keys)))
            for layer in ops.values():
                for o in layer:
                    self.assertIn(o.category, S.CATEGORIES)
                    self.assertGreater(o.bytes_total, 0)

    def test_scaling(self):
        c = self.c
        a1 = S.prefill_ops(c, 1024)["*"][3].flops
        a4 = S.prefill_ops(c, 4096)["*"][3].flops
        self.assertAlmostEqual(a4 / a1, 16.0)                 # causal attention ~ L^2
        s1 = S.ssd_scan_work(c, 1024).flops
        s4 = S.ssd_scan_work(c, 4096).flops
        self.assertAlmostEqual(s4 / s1, 4.0)                  # SSD ~ L
        g = S.prefill_ops(c, 1024)["M"][1]
        self.assertEqual(g.flops, 2 * 1024 * c.hidden * c.in_proj_out)
        # decode attention bytes dominated by KV read, linear in ctx
        d1 = S.decode_ops(c, 32, 1024)["*"][3].bytes_read
        d2 = S.decode_ops(c, 32, 2048)["*"][3].bytes_read
        self.assertGreater(d2 / d1, 1.9)

    def test_model_shares(self):
        counts = {"M": 27, "*": 4, "-": 25}
        sh = S.model_shares({"M:a": 1.0, "*:b": 1.0, "-:c": 2.0}, counts)
        self.assertAlmostEqual(sum(sh.values()), 1.0)
        self.assertAlmostEqual(sh["M:a"], 27 / (27 + 4 + 50))

    def test_moe_pattern_refused(self):
        with self.assertRaises(ValueError):
            S.layer_counts("ME-")


def _trace():
    ev = []
    def X(cat, name, ts, dur, tid=1, **args):
        ev.append({"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 1,
                   "tid": tid, "args": args})
    X("user_annotation", "opchar:M:in_proj", 0, 100)
    X("user_annotation", "opchar:*:attn", 200, 100)
    X("user_annotation", "opchar:outer", 190, 200)            # enclosing, inner must win
    X("cuda_runtime", "cudaLaunchKernel", 10, 2, correlation=1)
    X("cuda_runtime", "cudaLaunchKernel", 20, 2, correlation=2)
    X("cuda_driver", "cuLaunchKernel", 210, 2, correlation=3)  # triton launch path
    X("cuda_runtime", "cudaLaunchKernel", 500, 2, correlation=4)  # outside any range
    X("kernel", "sm80_xmma_gemm_bf16", 1000, 50, tid=7, correlation=1, grid=[109, 1, 1],
      block=[256, 1, 1], **{"registers per thread": 32, "shared memory": 0})
    X("kernel", "sm80_xmma_gemm_bf16", 1100, 50, tid=7, correlation=2, grid=[109, 1, 1],
      block=[256, 1, 1], **{"registers per thread": 32, "shared memory": 0})
    X("kernel", "BatchPrefillWithRaggedKVCacheKernel", 1200, 80, tid=7, correlation=3,
      grid=[40, 1, 1], block=[128, 1, 1], **{"registers per thread": 255, "shared memory": 0})
    X("kernel", "stray_kernel", 1300, 5, tid=7, correlation=4, grid=[1, 1, 1], block=[32, 1, 1],
      **{"registers per thread": 16, "shared memory": 0})
    return {"traceEvents": ev}


class TraceAttribution(unittest.TestCase):
    def test_attribution_and_table(self):
        s = TP.parse_trace(_trace())
        ops = {k.name: k.op for k in s.kernels}
        self.assertEqual(ops["BatchPrefillWithRaggedKVCacheKernel"], "*:attn")
        self.assertEqual(ops["stray_kernel"], "unattributed")
        self.assertEqual(s.by_op()["M:in_proj"][0].grid_blocks, 109)
        rows = TP.op_kernel_table(s, {"M:in_proj": 2, "*:attn": 1}, sm_count=108)
        gemm = next(r for r in rows if r["op"] == "M:in_proj")
        self.assertEqual(gemm["calls_per_iter"], 1.0)
        self.assertEqual(gemm["dur_us_per_iter"], 50.0)
        self.assertEqual(gemm["resident_blocks_per_sm"], 8)
        self.assertEqual(gemm["waves"], 1)                    # 109 blocks fit in 108*8 slots
        self.assertEqual(gemm["family"], "gemm")
        attn = next(r for r in rows if r["op"] == "*:attn")
        self.assertEqual(attn["family"], "attn")
        idle = TP.time_weighted_idle(rows)
        self.assertAlmostEqual(idle["M:in_proj#coverage"], 1.0)

    def test_kernel_time_follows_launch_not_kernel_timestamp(self):
        # moving the GPU kernel far away in time must not change its operator
        t = _trace()
        for e in t["traceEvents"]:
            if e["name"] == "BatchPrefillWithRaggedKVCacheKernel":
                e["ts"] = 50
        ops = {k.name: k.op for k in TP.parse_trace(t).kernels}
        self.assertEqual(ops["BatchPrefillWithRaggedKVCacheKernel"], "*:attn")


class WorkerHelpers(unittest.TestCase):
    def test_realized_sm_reads_nested_green_readout(self):
        # shape recorded by job-3fdczheaizqq (green_readout.green_sm_readout); a dict here crashed
        from experiments.opchar import worker as W
        rd = {"cuStreamGetGreenCtx_rc": 0,
              "green_sm": {"smCount": 44, "minSmPartitionSize": 4, "smCoscheduledAlignment": 2}}
        self.assertEqual(W.realized_sm(rd), 44)
        self.assertIsNone(W.realized_sm({"error": "x"}))


class TritonLaunch(unittest.TestCase):
    """Recorder against a fake triton.compiler.compiler with the 3.5.1 call signature."""

    def setUp(self):
        import types
        from experiments.opchar import triton_launch as TL
        self.TL = TL
        calls = self.calls = []

        class CompiledKernel:
            def __init__(self, name, warps, shared, regs):
                self.name, self.n_regs, self.n_spills = name, regs, 0
                self.metadata = types.SimpleNamespace(num_warps=warps, shared=shared, num_ctas=1)
                self.inited = False

            def _init_handles(self):
                self.inited = True

            def launch_metadata(self, grid, stream, *args):
                calls.append((self.name, tuple(grid), stream, args))
                return "orig"
        self.CK = CompiledKernel
        mod = types.ModuleType(TL.TRITON_COMPILER_MODULE)
        mod.CompiledKernel = CompiledKernel
        self._saved = {k: sys.modules.get(k) for k in ("triton", "triton.compiler", TL.TRITON_COMPILER_MODULE)}
        sys.modules["triton"] = types.ModuleType("triton")
        sys.modules["triton.compiler"] = types.ModuleType("triton.compiler")
        sys.modules[TL.TRITON_COMPILER_MODULE] = mod

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_records_only_inside_op_and_passes_through(self):
        rec = self.TL.TritonLaunchRecorder().install()
        k = self.CK("_chunk_scan_fwd_kernel", 4, 32768, 128)
        self.assertEqual(k.launch_metadata((216, 1), "s", 7), "orig")      # outside op: not recorded
        rec.begin_op("M:ssd_scan")
        k.launch_metadata((216,), "s", 7)
        k.launch_metadata((216, 2, 1), "s", 7)
        rec.end_op()
        rec.uninstall()
        k.launch_metadata((1,), "s")                                       # after uninstall
        self.assertEqual(len(self.calls), 4)                               # original always called
        self.assertEqual([r["grid"] for r in rec.records], [[216, 1, 1], [216, 2, 1]])
        self.assertTrue(k.inited)
        self.assertEqual(rec.records[0]["n_regs"], 128)
        self.assertFalse(getattr(self.CK.launch_metadata, "_opchar_wrapped", False))

    def test_double_install_refused(self):
        rec = self.TL.TritonLaunchRecorder().install()
        with self.assertRaises(RuntimeError):
            self.TL.TritonLaunchRecorder().install()
        rec.uninstall()

    def test_launch_table(self):
        recs = [{"op": "M:ssd_scan", "kernel": "k", "grid": [109, 1, 1], "num_warps": 4, "num_ctas": 1,
                 "shared": 0, "n_regs": 32, "n_spills": 0},
                {"op": "M:ssd_scan", "kernel": "k", "grid": [109, 1, 1], "num_warps": 4, "num_ctas": 1,
                 "shared": 0, "n_regs": 32, "n_spills": 0}]
        rows = self.TL.launch_table(recs, {0: 1.0, 1: 3.0}, {"M:ssd_scan": 2}, sm_count=108)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["calls"], r["calls_per_iter"], r["segment_ms_per_iter"]), (2, 1.0, 2.0))
        self.assertEqual(r["resident_blocks_per_sm"], 16)                  # 128 thr, 32 regs
        self.assertEqual(r["waves"], 1)
        self.assertAlmostEqual(self.TL.op_idle(rows)["M:ssd_scan"], r["idle_ratio"])


class GraphCensus(unittest.TestCase):
    def test_struct_layout_matches_cuda_h(self):
        import ctypes
        from experiments.opchar import graph_census as GC
        P = GC.KernelNodeParamsV2
        self.assertEqual(ctypes.sizeof(P), 72)                  # 8 + 7*4 (+4 pad) + 4*8
        self.assertEqual(P.kernelParams.offset, 40)
        self.assertEqual(P.kern.offset, 56)

    def test_census_rows_waves_on_partition(self):
        from experiments.opchar import graph_census as GC

        def k(grid, threads, regs, dyn=0, static=0):
            return {"node_kind": "kernel", "kernel": "k", "grid": [grid, 1, 1], "block": [threads, 1, 1],
                    "dyn_smem": dyn, "static_smem": static, "regs": regs, "max_threads": 1024}
        # a grid sized for the full device (108 CTAs, 1 resident/SM via smem) on a 104-SM partition
        nodes = [k(108, 128, 64, dyn=100 * 1024), {"node_kind": "memset"}, k(104, 256, 32)]
        rows = GC.census_rows(nodes, sm_count=104)
        self.assertEqual([r["node_kind"] for r in rows], ["kernel", "memset", "kernel"])
        self.assertEqual(rows[0]["resident_blocks_per_sm"], 1)
        self.assertEqual((rows[0]["waves"], rows[0]["tail_sms"]), (2, 4))
        self.assertAlmostEqual(rows[0]["idle_ratio"], 100 / 208)
        self.assertEqual(GC.census_rows(nodes[:1], sm_count=108)[0]["waves"], 1)   # same grid, full GPU
        self.assertEqual(rows[2]["waves"], 1)

    def test_census_walk_with_fake_driver(self):
        from experiments.opchar import graph_census as GC

        class P:
            def __init__(self, grid):
                self.gridDimX, self.gridDimY, self.gridDimZ = grid
                self.blockDimX, self.blockDimY, self.blockDimZ = (128, 1, 1)
                self.sharedMemBytes = 0

        class FakeDrv:
            def nodes(self, g): return {1: [10, 11, 12], 2: [20]}[g]
            def node_type(self, n): return {10: 0, 11: 2, 12: 4, 20: 0}[n]
            def child_graph(self, n): return 2
            def kernel_params(self, n): return P((n, 1, 1))
            def func_name(self, p): return f"k{p.gridDimX}"
            def func_attr(self, p, key, device=0): return {"static_smem": 0, "regs": 40, "max_threads": 1024}[key]

        out = GC.census(1, FakeDrv())
        self.assertEqual([o["node_kind"] for o in out], ["kernel", "memset", "kernel"])
        self.assertEqual(out[2]["kernel"], "k20")                 # child graph flattened
        self.assertEqual(out[0]["grid"], [10, 1, 1])


class Plan(unittest.TestCase):
    def test_grid(self):
        for sm in R.SM_GRID:
            self.assertTrue(sm == 108 or (sm % 2 == 0 and sm >= 16), sm)
        full = R.plan("full")
        n = len(R.SM_GRID)
        self.assertEqual(len(full), 3 * n)
        self.assertTrue(all("--launch-log" in c for l, c in full if not l.startswith("memcpy")))
        ctrl = R.plan("ctrl")
        self.assertEqual(len(ctrl), 2 * len(R.CTRL_SM))
        self.assertIn(104, R.CTRL_SM)
        self.assertIn(108, R.CTRL_SM)
        self.assertTrue(all("--graph-census" in c for _, c in ctrl))
        probe = R.plan("probe")
        self.assertEqual(probe[0][0], "probe")
        self.assertTrue(any(l.startswith("smoke_decode") for l, _ in probe))


class JobScript(unittest.TestCase):
    SCRIPT = _ENG / "results" / "opchar" / "opchar_job.sh"

    def _run(self, env_extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        env.update(env_extra)
        return subprocess.run(["/usr/bin/bash", str(self.SCRIPT)], env=env, capture_output=True,
                              text=True, timeout=60)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["/usr/bin/bash", "-n", str(self.SCRIPT)]).returncode, 0)

    def test_bad_mode_and_knob_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run({"PDMUX_PROJECT_ROOT": d, "PDMUX_OPCHAR_MODE": "bogus"})
            self.assertEqual(r.returncode, 2)
            self.assertIn("ABORT_ENV", r.stdout)
            r = self._run({"PDMUX_PROJECT_ROOT": d, "PDMUX_OPCHAR_MODE": "probe",
                           "PDMUX_R2_POLICY": "fixed"})
            self.assertEqual(r.returncode, 2)
            self.assertIn("unregistered knob", r.stdout)


if __name__ == "__main__":
    unittest.main()
