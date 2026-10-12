"""Multi-GPU harness H1–H10 (PREREG_MULTIGPU rev6): launcher, plan, budget, labels, drivers.  All CPU.

Every behaviour is paired with its negative control (lesson 53/254).  The two integration tests run the Job
scripts end to end against a FAKE runner / `--dry-run` (no GPU, no server).
"""

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "results" / "multigpu"))
sys.path.insert(0, str(ENGINE / "scripts" / "multigpu"))
sys.path.insert(0, str(ENGINE / "benchmarks"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import mg_boot as B  # noqa: E402
import mg_const as C  # noqa: E402
import mg_drive as D  # noqa: E402
import mg_label as L  # noqa: E402
import mg_plan as P  # noqa: E402

REPO = ENGINE.parents[1]
STOCK_RUNNER = ENGINE / "scripts" / "r2_eval" / "engine_bench_runner.sh"


def spec_for(cfg, arm):
    cs = P.config_spec(cfg)
    return dict(cs, arm=arm, kind="boot1", block="main", boot_id=f"t_{cfg}_{arm}", steps=[])


# =========================================================================== H3 / H2 / H4 / H5 / H6 / cap
class ServerArgs(unittest.TestCase):
    def args(self, cfg, arm):
        return B.build_server_args(spec_for(cfg, arm), 12345, "/cfg.yml" if arm in C.PDMUX_ARMS else None)

    def test_h3_fused_sym_is_pdmux_args_minus_pdmux_flags(self):
        for cfg in ("N1", "TP4", "DP8"):
            ag, fs = self.args(cfg, "agnostic"), self.args(cfg, "fused_sym")
            self.assertEqual([x for x in ag if x not in ("--enable-pdmux", "--pdmux-config-path", "/cfg.yml")], fs)

    def test_fused_vs_fused_sym_server_args_diff_is_pinned(self):
        """fused = vanilla + --disable-piecewise-cuda-graph; vs fused_sym the ONLY differences are
        {chunked_prefill_size, disable_overlap_schedule (fused_sym sets them)} and {disable_piecewise_cuda_graph (fused sets it)}."""
        def flags(args):
            return {x for x in args if x.startswith("--")}
        for cfg in ("N1", "TP2", "TP8", "DP4"):
            f, fs = self.args(cfg, "fused"), self.args(cfg, "fused_sym")
            self.assertEqual(flags(f) ^ flags(fs), {"--chunked-prefill-size", "--disable-overlap-schedule",
                                                    "--disable-piecewise-cuda-graph"})
            self.assertIn("--disable-piecewise-cuda-graph", f)
            self.assertNotIn("--disable-piecewise-cuda-graph", fs)
            self.assertEqual([x for x in f if x != "--disable-piecewise-cuda-graph"],
                             [x for x in fs if x not in C.SCHED_KNOBS])
        for arm in ("fused_sym", "agnostic", "d44"):                                  # negative control: not on the other arms
            self.assertNotIn("--disable-piecewise-cuda-graph", self.args("TP2", arm))

    def test_h3_knobs_byte_identical_in_text(self):
        knobs = " ".join(C.SCHED_KNOBS)
        self.assertEqual(knobs, "--chunked-prefill-size -1 --disable-overlap-schedule")
        for arm in ("fused_sym", "agnostic", "d44"):
            self.assertIn(knobs, " ".join(self.args("TP2", arm)))
        self.assertNotIn("--disable-overlap-schedule", self.args("TP2", "fused"))      # negative control
        self.assertNotIn("--chunked-prefill-size", self.args("TP2", "fused"))

    def test_h3_equals_stock_runner_pdmux_knobs(self):
        """The knobs of the STOCK engine_bench_runner.sh PD-mux branch (dry run) are exactly SCHED_KNOBS."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "venv" / "bin").mkdir(parents=True)
            (root / "venv" / "bin" / "activate").write_text("")
            rec = root / "run.json"
            rec.write_text(json.dumps({"trace_path": "t.jsonl", "server_seed": 1, "max_running_requests": 48,
                                       "ttft_slo_ms": 1000, "itl_slo_ms": 50, "workload": "w"}))
            env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
            env.update(PDMUX_DRY_RUN="1", PDMUX_VENV=str(root / "venv"), PDMUX_ENGINE_MODE="pdmux")
            p = subprocess.run(["bash", str(STOCK_RUNNER), str(rec), str(root / "out")], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            a = (root / "out" / "server_args.txt").read_text().splitlines()
            i = a.index("--chunked-prefill-size")
            self.assertEqual(tuple(a[i:i + 3]), C.SCHED_KNOBS)

    def test_pdmux_arms_differ_only_by_config(self):
        ag, d4 = self.args("N1", "agnostic"), self.args("N1", "d44")
        self.assertEqual(ag, d4)      # same flags; the arm difference is the config file (given as abs path) + env

    def test_h4_backend_flashinfer_everywhere(self):
        for arm in C.ARMS:
            a = self.args("TP2", arm)
            self.assertEqual(a[a.index("--attention-backend") + 1], "flashinfer")

    def test_cap_rules(self):
        cap = lambda cfg: self.args(cfg, "fused")[self.args(cfg, "fused").index("--max-running-requests") + 1]
        self.assertEqual((cap("N1"), cap("TP2"), cap("TP8"), cap("DP2"), cap("DP8")), ("48", "96", "384", "48", "48"))

    def test_dp_and_tp_flags(self):
        a = self.args("DP4", "fused")
        self.assertEqual(a[a.index("--dp-size") + 1], "4")
        self.assertEqual(a[a.index("--load-balance-method") + 1], "round_robin")
        self.assertNotIn("--tp-size", a)
        self.assertEqual(self.args("TP8", "fused")[-2:], ["--tp-size", "8"])
        self.assertNotIn("--dp-size", self.args("TP8", "fused"))

    def test_decode_log_interval_registered(self):
        a = self.args("N1", "fused")
        self.assertEqual(a[a.index("--decode-log-interval") + 1], "40")

    def test_h2_health_wait_1800(self):
        self.assertEqual(C.HEALTH_WAIT_S, 1800)

    def test_env_per_arm(self):
        s_d44, u = B.build_env(spec_for("N1", "d44"), "/b")
        self.assertEqual((s_d44["PDMUX_R2_POLICY"], s_d44["PDMUX_R2_FIXED_DSM"]), ("fixed", "44"))
        s_ag, u = B.build_env(spec_for("N1", "agnostic"), "/b")
        self.assertNotIn("PDMUX_R2_POLICY", s_ag)
        self.assertIn("PDMUX_R2_POLICY", u)
        for arm in ("fused", "fused_sym"):
            s, _ = B.build_env(spec_for("N1", arm), "/b")
            self.assertEqual(s["PDMUX_ENGINE_MODE"], "vanilla")
            self.assertNotIn("PDMUX_R2_POLICY", s)
            self.assertNotIn("PDMUX_TELEMETRY_PATH", s)
        self.assertEqual(s_d44["SGLANG_LOG_MS"], "1")

    def test_unknown_arm_and_tp_dp_mix_rejected(self):
        with self.assertRaises(ValueError):
            B.build_server_args(dict(spec_for("N1", "fused"), arm="x"), 1)
        with self.assertRaises(ValueError):
            B.build_server_args(dict(spec_for("N1", "fused"), tp=2, dp=2), 1)


class RunnerDry(unittest.TestCase):
    def test_dry_run_writes_records_and_aborts_on_jit(self):
        with tempfile.TemporaryDirectory() as td:
            spec = dict(spec_for("TP2", "fused_sym"), out_root=td)
            sp = Path(td) / "s.json"
            sp.write_text(json.dumps(spec))
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            p = subprocess.run(["bash", str(ENGINE / "scripts/multigpu/mg_runner.sh"), str(sp), "--dry-run"],
                               env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            bd = Path(td) / "boots" / "t_TP2_fused_sym"
            for f in ("server_args.txt", "env.json", "comm_env.json", "steps.json", "boot.json"):
                self.assertTrue((bd / f).exists(), f)
            self.assertTrue((Path(td) / "ULIMIT_N.txt").exists())
            comm = json.loads((bd / "comm_env.json").read_text())
            self.assertTrue(all(v["state"] == "unset" for v in comm["vars"].values()))
            env2 = dict(env, SGLANG_USE_JIT_ALL_REDUCE="1")                # negative control: ABORT_COMM_ENV
            p = subprocess.run(["bash", str(ENGINE / "scripts/multigpu/mg_runner.sh"), str(sp), "--dry-run"],
                               env=env2, capture_output=True, text=True)
            self.assertEqual(p.returncode, 3)
            self.assertEqual(json.loads((bd / "boot.json").read_text())["status"], "ABORT_COMM_ENV")

    def test_gpu_ids_must_match_n(self):
        with tempfile.TemporaryDirectory() as td:
            spec = dict(spec_for("TP2", "fused"), out_root=td, gpu_ids="0")
            self.assertEqual(B.run_boot(spec, dry=True), 2)


class Contamination(unittest.TestCase):
    def snap(self, used, procs):
        return {"available": True, "gpus": {"0": {"used_mib": used, "n_compute_procs": procs}}}

    def test_cases(self):
        self.assertFalse(B.contamination_verdict(self.snap(300, 0), False)["gpu_contaminated"])
        self.assertTrue(B.contamination_verdict(self.snap(2000, 0), False)["gpu_contaminated"])        # (가)
        self.assertTrue(B.contamination_verdict(self.snap(300, 2), False)["gpu_contaminated"])         # (나)
        self.assertFalse(B.contamination_verdict(self.snap(300, 2), True)["gpu_contaminated"])         # count hidden -> (가) only
        v = B.contamination_verdict({"available": False}, False)
        self.assertFalse(v["gpu_contaminated"] or v["verified"])


class Piecewise(unittest.TestCase):
    def test_h8_record(self):
        log = ("server_args=ServerArgs(disable_piecewise_cuda_graph=False, piecewise_cuda_graph_tokens=[4, 8, 16], "
               "chunked_prefill_size=8192)\nCapture cuda graph begin\nPiecewise CUDA graph capture\nCapture cuda graph end")
        rec, lines = B.piecewise_record(log)
        self.assertEqual(rec["disable_piecewise_cuda_graph"], "False")
        self.assertEqual(rec["piecewise_lines"], 1)
        self.assertGreaterEqual(len(lines), 3)


class TraceGen(unittest.TestCase):
    def test_identical_args_identical_bytes_and_seed_changes(self):
        with tempfile.TemporaryDirectory() as td:
            a = B.gen_trace(Path(td) / "a", "A", 4.0, 100, 4386)
            b = B.gen_trace(Path(td) / "b", "A", 4.0, 100, 4386)
            c = B.gen_trace(Path(td) / "c", "A", 4.0, 100, 4162)
            self.assertEqual(a, b)
            self.assertNotEqual(a, c)
            rows = [json.loads(x) for x in (Path(td) / "a").read_text().splitlines()]
            self.assertEqual(rows[1]["input_tokens"], 256)
            self.assertEqual(rows[1]["output_tokens"], 512)


# =========================================================================== H5 / H6 loadgen
class LoadgenConstants(unittest.TestCase):
    def test_h5_sock_read_registered_in_trace_loadgen(self):
        src = (ENGINE / "benchmarks/pdmux_eval/trace_loadgen.py").read_text()
        self.assertIn(f"sock_read={C.SOCK_READ_S}", src)
        import mg_loadgen
        self.assertTrue(mg_loadgen.sock_read_registered())

    def test_h6_fd_raise_and_hard_limit_abort(self):
        import mg_loadgen
        code = ("import sys;sys.path.insert(0,%r);import mg_loadgen,json;"
                "import resource;resource.setrlimit(resource.RLIMIT_NOFILE,(%d,%d));"
                "print(json.dumps(mg_loadgen.raise_fd_limit()))")
        env = dict(os.environ, PYTHONPATH=str(ENGINE / "scripts/multigpu"))
        p = subprocess.run([sys.executable, "-c", code % (str(ENGINE / "scripts/multigpu"), 1024, 1048576)],
                           capture_output=True, text=True, env=env)
        rec = json.loads(p.stdout)
        self.assertTrue(rec["ok"] and rec["soft_after"] >= 8192, p.stdout + p.stderr)
        p = subprocess.run([sys.executable, "-c", code % (str(ENGINE / "scripts/multigpu"), 1024, 4096)],
                           capture_output=True, text=True, env=env)
        self.assertFalse(json.loads(p.stdout)["ok"])          # hard limit 4096 < 8192 -> ABORT_FD_LIMIT path


# =========================================================================== plan / budget
class Plan(unittest.TestCase):
    def test_rung_arithmetic(self):
        self.assertEqual(P.n_req_for(1.6 * 2.99), 810)
        self.assertEqual(P.n_req_for(1.6 * 0.69), 190)
        self.assertEqual(P.n_req_for(1.6 * 8 * 2.99), 4000)            # upper clamp
        self.assertEqual(P.n_req_for(0.05), 40)                         # lower clamp
        self.assertAlmostEqual(P.hi_rung(4, "B")["offered"], 1.6 * 4 * 0.69, 6)
        self.assertAlmostEqual(P.hi_rung(2, "A", 1)["offered"], 2 * 1.6 * 2 * 2.99, 5)

    def test_order_gpu_dev_first_and_counts(self):
        st = P.m3_static_boots()
        self.assertEqual(st[0][0], "gpu_dev")
        self.assertEqual(len(st[0][1]), 8)
        self.assertEqual([n.split(".")[0] for n, _ in st[1:15:2]], list(C.CONFIG_ORDER))
        self.assertEqual([n for n, _ in st[-3:]], ["numa", "nsys", "anchor2"])
        self.assertEqual(sum(len(b) for _, b in st), 70)
        self.assertEqual(sum(1 for _, bs in st for b in bs for s in b["steps"] if s["type"] == "cell"), 365)

    def test_gpu_dev_cell_is_fused_A_seed4386_each_gpu(self):
        for g, b in enumerate(P.boots_gpu_dev()):
            self.assertEqual((b["arm"], b["gpu_ids"], b["tp"], b["dp"]), ("fused", str(g), 1, 1))
            cells = [s for s in b["steps"] if s["type"] == "cell"]
            self.assertEqual(len(cells), 1)
            self.assertEqual((cells[0]["shape"], cells[0]["seed"], cells[0]["rung"]), ("A", 4386, "HI"))

    def test_boot1_has_both_shapes_four_seeds_one_boot(self):                       # H1
        b = P.boots_boot1("TP4")[0]
        cells = [s for s in b["steps"] if s["type"] == "cell"]
        self.assertEqual(len(cells), 8)
        self.assertEqual({(c["shape"], c["seed"]) for c in cells}, {(s, d) for s in "AB" for d in C.HI_SEEDS})
        self.assertEqual(sum(1 for s in b["steps"] if s["type"] == "warmup"), 2)    # warm-up phase separated from cells

    def test_arm_order_is_deterministic_permutation_and_varies_by_config(self):
        self.assertEqual(sorted(P.arm_order("TP2")), sorted(C.ARMS))
        self.assertEqual(P.arm_order("TP2"), P.arm_order("TP2"))
        self.assertTrue(len({tuple(P.arm_order(c)) for c in C.CONFIGS}) > 1)

    def test_lo_cells_identical_across_arms_MGP9(self):
        x = {"A": P.lo_rung(7.0), "B": P.lo_rung(0.8)}
        sets = []
        for b in P.boots_boot2("DP4", x):
            sets.append(sorted((s["shape"], s["seed"], s["offered"], s["n_req"]) for s in b["steps"] if s["type"] == "cell"))
        self.assertTrue(all(s == sets[0] for s in sets))
        self.assertEqual({c[1] for c in sets[0]}, set(C.LO_SEEDS))

    def test_exclusion_drops_boots(self):
        self.assertEqual(len(P.boots_boot1("TP2", exclude={"TP2:d44"})), 3)

    def test_numa_and_anchor_blocks(self):
        self.assertEqual({b["gpu_ids"] for b in P.boots_numa()}, {"0,1,4,5"})
        self.assertEqual([b["arm"] for b in P.boots_anchor2()], ["fused", "fused_sym", "agnostic"])
        self.assertTrue(all(s["round"] == 2 for b in P.boots_anchor2() for s in b["steps"] if s["type"] == "cell"))


class Budget(unittest.TestCase):
    def test_lambda_eq_prior_in_audit_band_and_tp_below_prior_over_cap(self):
        b1 = P.m3_budget()
        lo, hi = C.AUDIT_FORECAST_H["lambda_eq_prior"]
        self.assertTrue(lo - 1.0 <= b1["total_h"] <= hi + 1.0, b1["total_h"])
        b2 = P.m3_budget(lambda cfg: 0.7 if cfg.startswith("TP") else 1.0)
        self.assertGreater(b2["total_h"], b1["total_h"])
        self.assertTrue(b2["over_cap"])
        self.assertLess(P.m3_budget(lambda c: 1.0)["total_h"], P.m3_budget(lambda c: 0.5)["total_h"])    # monotone

    def test_extension_adds_boots(self):
        base = P.m3_budget()
        ext = P.m3_budget(assume_ext=True)
        self.assertGreater(ext["boots"], base["boots"])
        self.assertGreater(ext["total_h"], base["total_h"])

    def test_m2g_within_cap(self):
        r = P.m2g_budget()
        self.assertFalse(r["over_cap"])
        self.assertGreater(r["boots"], 30)

    def test_m2g_covers_every_registered_boot(self):
        ids = [b["boot_id"] for b in P.m2g_boots()]
        for cfg in ("N1",) + P.M2G_CFGS:
            for arm in C.ARMS:
                self.assertIn(f"m2g_{cfg}_{arm}", ids)
        self.assertEqual(sum(1 for b in P.m2g_boots() if b["kind"] == "g0ii"), 5)
        self.assertEqual(sum(1 for b in P.m2g_boots() if b["kind"] == "g0iii"), 1)
        self.assertEqual(sum(1 for b in P.m2g_boots() if b["kind"] == "g4"), 2)
        g3b = [b for b in P.m2g_boots() if any(s.get("rung") == "G3B" for s in b["steps"])]
        self.assertEqual(len(g3b), 3 * 2)                                                  # TP{2,4,8} x {agnostic,d44}
        s = next(s for s in g3b[0]["steps"] if s.get("rung") == "G3B")
        self.assertAlmostEqual(s["offered"] / (1.5 * g3b[0]["N"] * 2.99), 1.0, 5)


class PassTable(unittest.TestCase):
    def test_abort_cases(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.json"
            self.assertEqual(P.load_pass_table(p, "x")[1], "ABORT_M2G_NOT_PASSED")                 # missing
            raw = json.dumps({"cells": {"TP2:fused": {"included": True}}, "g5a_unit_test": "PASS"})
            p.write_text(raw)
            sha = hashlib.sha256(raw.encode()).hexdigest()
            self.assertIsNone(P.load_pass_table(p, sha)[1])
            self.assertEqual(P.load_pass_table(p, "0" * 64)[1], "ABORT_M2G_NOT_PASSED")           # digest mismatch
            for bad in ("NOT_TESTED", "FAIL", "TREE_MISSING", "SKIPPED_DRYRUN", None):             # R5: g5a unit test must be PASS
                d = {"cells": {}}
                if bad is not None:
                    d["g5a_unit_test"] = bad
                raw2 = json.dumps(d)
                p.write_text(raw2)
                self.assertEqual(P.load_pass_table(p, hashlib.sha256(raw2.encode()).hexdigest())[1], "ABORT_M2G_NOT_PASSED", bad)
            p.write_text(raw)
            self.assertEqual(P.load_pass_table(p, "")[1], "ABORT_M2G_NOT_PASSED")                 # no registered digest
            p.write_text("{not json")
            self.assertEqual(P.load_pass_table(p, hashlib.sha256(b"{not json").hexdigest())[1], "ABORT_M2G_NOT_PASSED")


# =========================================================================== labels
def cell_rec(cfg, arm, shape, rung, seed, achieved, g, status="OK", ext=0, rnd=1, **kw):
    r = {"cell_id": f"{cfg}{arm}{shape}{rung}{seed}", "status": status, "g": g, "achieved": achieved, "truncated": False,
         "harness_limited": False, "frontend_saturated": False,
         "meta": dict(config=cfg, arm=arm, shape=shape, rung=rung, seed=seed, ext=ext, round=rnd, block="main")}
    r.update(kw)
    return r


def key_cells(cfg, arm, shape, lam, g_hi=0.4, g_lo=0.01, seed_noise=(1.0, 1.01, 0.99, 1.005), rnd=1, **kw):
    out = [cell_rec(cfg, arm, shape, "HI", s, lam * n, g_hi, rnd=rnd, **kw) for s, n in zip(C.HI_SEEDS, seed_noise)]
    out += [cell_rec(cfg, arm, shape, "LO", s, lam * 0.6, g_lo, rnd=rnd) for s in C.LO_SEEDS]
    return out


class Labels(unittest.TestCase):
    def test_mutation_checks_all_killed_control_survives(self):
        ok, v, per = L.run_mutation_checks()
        self.assertTrue(ok, json.dumps(v))
        self.assertFalse(v["CONTROL_unmutated"]["killed"])
        for m in ("a_server_decode_lines", "b_sample_mean", "c_keep_errors_in_Qc", "d_running_eq_cap"):
            self.assertTrue(v[m]["killed"], m)

    def test_each_mutant_killed_on_its_own_scenario(self):
        ok, v, per = L.run_mutation_checks()
        self.assertIn("S4_prefill_priority_saturated", v["a_server_decode_lines"]["killed_on"])
        self.assertIn("S3_burst_unsaturated", v["b_sample_mean"]["killed_on"])
        self.assertIn("S7_errors_near_threshold", v["c_keep_errors_in_Qc"]["killed_on"])
        self.assertIn("S5_cap_bound_unsaturated", v["d_running_eq_cap"]["killed_on"])

    def test_qc_formula_and_error_conventions_H9(self):
        reqs = [{"sent_s": 0.0, "ttft_ms": 2000.0, "error": ""}, {"sent_s": 1.0, "ttft_ms": None, "error": "x"},
                {"sent_s": 2.0, "ttft_ms": 500.0, "error": "midstream"}, {"sent_s": 2.0, "ttft_ms": 5000.0, "error": ""}]
        self.assertEqual(L.qc_value(reqs, 1.5), 1)                 # the no-ttft error request is NOT in Q_c
        self.assertEqual(L.qc_value(reqs, 2.2), 2)                 # the error WITH ttft keeps its interval
        self.assertEqual(L.qc_value(reqs, 2.6), 1)
        rec = L.analyze_cell(reqs, {"cell_id": "x"})
        self.assertEqual((rec["n_errors"], rec["n_errors_without_ttft"], rec["n_errors_with_ttft"]), (2, 1, 1))
        self.assertEqual(rec["status"], "CLIENT_ERROR")            # 2/4 > 1 %

    def test_client_error_threshold_boundary(self):
        base = [{"sent_s": float(i), "scheduled_arrival_s": float(i), "ttft_ms": 100.0, "error": "", "completion_s": i + 3.0}
                for i in range(200)]
        one = [dict(r) for r in base]
        one[50] = dict(one[50], error="boom", ttft_ms=None)
        self.assertEqual(L.analyze_cell(one, {})["status"], "OK")                 # 0.5 %
        two = [dict(r) for r in one]
        two[60] = dict(two[60], error="boom", ttft_ms=None)
        two[70] = dict(two[70], error="boom", ttft_ms=None)
        self.assertEqual(L.analyze_cell(two, {})["status"], "CLIENT_ERROR")       # 1.5 %

    def test_saturation_unresolved_below_20(self):
        reqs = [{"sent_s": float(i), "scheduled_arrival_s": float(i), "ttft_ms": 100.0, "error": "", "completion_s": i + 1.0}
                for i in range(10)]
        self.assertEqual(L.analyze_cell(reqs, {})["status"], "SATURATION_UNRESOLVED")

    def test_g_time_integral_matches_fluid_on_saturated_fifo(self):
        reqs = L._fifo_cell(6.4, 4.0, 120.0)
        g, _, _ = L.g_of(reqs)
        self.assertGreater(g, 0.25)
        self.assertLess(g, 0.40)                                   # fluid 1 - 4/6.4 = 0.375 minus edge

    def test_harness_limited_lag_and_throttle(self):
        reqs = L._fifo_cell(2.0, 4.0, 60.0)
        for r in reqs:
            r["sent_s"] += 0.0
        self.assertFalse(L.analyze_cell(reqs, {})["harness_limited"])
        lagged = [dict(r, sent_s=r["scheduled_arrival_s"] + 0.020) for r in reqs]
        self.assertTrue(L.analyze_cell(lagged, {})["harness_limited"])           # p99 20 ms > 5 ms
        b = "nr_periods 100\nnr_throttled 0\n"
        self.assertTrue(L.analyze_cell(reqs, {}, b, "nr_periods 200\nnr_throttled 2\n")["harness_limited"])   # 2 % > 1 %
        self.assertFalse(L.analyze_cell(reqs, {}, b, "nr_periods 200\nnr_throttled 1\n")["harness_limited"])  # 1 % not >
        self.assertTrue(L.analyze_cell(reqs, {}, "", "")["throttle_unmeasured"])

    def test_frontend_saturated_per_second_delta_and_thread_max(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as td:
            p = Path(td) / "proc.csv"
            rows = ["t_mono,role,pid,tid,ticks"]
            for k in range(11):          # detokenizer: 1.0 core for 3 of 10 seconds; tokenizer_manager 0.5 core always
                rows.append(f"{100 + k},detokenizer,5,-1,{(k if k <= 3 else 3) * 100 + max(0, 0)}")
                rows.append(f"{100 + k},tokenizer_manager,4,-1,{k * 50}")
                rows.append(f"{100 + k},tokenizer_manager,4,41,{k * 50}")
            p.write_text("\n".join(rows) + "\n")
            fe = L.frontend_stats(str(p), (100.0, 110.0))
            self.assertTrue(fe["frontend_saturated"])                                    # 3/10 = 30 % >= 20 %
            self.assertAlmostEqual(fe["roles"]["detokenizer"]["frac_seconds_ge_0.90"], 0.3, 6)
            self.assertAlmostEqual(fe["roles"]["tokenizer_manager"]["frac_seconds_ge_0.90"], 0.0, 6)
            self.assertAlmostEqual(fe["per_thread_max_cores"]["tokenizer_manager"], 0.5, 6)
            rows2 = [r for r in rows if "detokenizer" not in r]
            p.write_text("\n".join(rows2) + "\n")
            self.assertFalse(L.frontend_stats(str(p), (100.0, 110.0))["frontend_saturated"])       # negative control

    def test_key_hi_lo_knee_and_lambda_star(self):
        lab = L.key_labels(key_cells("N1", "fused", "A", 10.0))
        self.assertEqual((lab["hi_label"], lab["lo_label"], lab["knee"]), ("HI_SATURATED", "LO_UNSATURATED", "KNEE_BRACKETED"))
        self.assertAlmostEqual(lab["lambda_star"], 10.0 * 1.0025, 6)               # median of 4 seed values
        self.assertFalse(lab["ext_trigger"])
        lab = L.key_labels(key_cells("N1", "fused", "A", 10.0, g_hi=0.08))
        self.assertTrue(lab["ext_trigger"] and lab["hi_label"] == "HI_NOT_SATURATED" and lab["knee"] == "KNEE_NOT_BRACKETED")
        lab = L.key_labels(key_cells("N1", "fused", "A", 10.0, g_lo=0.06))
        self.assertEqual((lab["lo_label"], lab["knee"]), ("LO_ABOVE_0.05", "KNEE_NOT_BRACKETED"))
        self.assertIsNotNone(lab["lambda_star"])                                   # lambda* recorded, G judgement excluded

    def test_client_error_does_not_trigger_extension_and_blocks_lambda(self):
        cs = key_cells("N1", "fused", "A", 10.0)
        cs[0] = cell_rec("N1", "fused", "A", "HI", C.HI_SEEDS[0], None, 0.0, status="CLIENT_ERROR")
        lab = L.key_labels(cs)
        self.assertEqual(lab["hi_label"], "HI_NOT_JUDGED")
        self.assertFalse(lab["ext_trigger"])
        self.assertIsNone(lab["lambda_star"])

    def test_ladder_too_low_after_extension(self):
        cs = [c for c in key_cells("N1", "fused", "A", 10.0, g_hi=0.08, ext=1)]
        self.assertEqual(L.key_labels(cs)["hi_label"], "LADDER_TOO_LOW")

    def test_extension_cells_supersede_first_hi(self):
        first = key_cells("N1", "fused", "A", 10.0, g_hi=0.05, ext=0)[:4]
        ext = [cell_rec("N1", "fused", "A", "HI", s, 12.0, 0.4, ext=1) for s in C.HI_SEEDS]
        lab = L.key_labels(first + ext)
        self.assertEqual(lab["hi_label"], "HI_SATURATED")
        self.assertAlmostEqual(lab["lambda_star"], 12.0)

    def test_partition_not_exercised(self):
        cs = key_cells("TP2", "agnostic", "A", 10.0)
        for c in cs:
            if c["meta"]["rung"] == "HI":
                c["etime"] = {"status": "OK", "e_time": 0.3}
        self.assertTrue(L.key_labels(cs)["partition_not_exercised"])
        for c in cs:
            if c["meta"]["rung"] == "HI":
                c["etime"] = {"status": "OK", "e_time": 0.7}
        self.assertFalse(L.key_labels(cs).get("partition_not_exercised", False))

    def test_paired_ci_uses_seed_pairing(self):
        # strongly correlated seeds: paired CI is far narrower than an unpaired ratio-of-independent-resamples would be
        num = [10.0, 14.0, 12.0, 16.0]
        den = [5.0, 7.0, 6.0, 8.0]
        p, lo, hi = L.paired_ratio_ci(num, den)
        self.assertEqual((round(lo, 9), round(hi, 9), round(p, 9)), (2.0, 2.0, 2.0))
        self.assertEqual(L.paired_ratio_ci(num, den), L.paired_ratio_ci(num, den))       # seed 1 deterministic

    def test_trend_joint_resample_and_labels(self):
        up = [(1, [2, 2.1, 1.9, 2.0], [1, 1, 1, 1]), (2, [3, 3.1, 2.9, 3.0], [1, 1, 1, 1]), (4, [4.5, 4.6, 4.4, 4.5], [1, 1, 1, 1])]
        self.assertEqual(L.trend(up)["label"], "G_INCREASES_WITH_N")
        down = [(n, [v for v in nu], d) for (n, nu, d) in up][::-1]
        flat = [(n, [2.0] * 4, [1.0] * 4) for n in (1, 2, 4, 8)]
        t = L.trend(flat)
        self.assertEqual((t["label"], t["magnitude"]), ("NO_TREND_DETECTED", "DIRECTION_NOT_CITABLE_UNDER_3PCT"))
        dn = [(1, [4.5] * 4, [1] * 4), (2, [3] * 4, [1] * 4), (4, [2] * 4, [1] * 4)]
        self.assertEqual(L.trend(dn)["label"], "G_DECREASES_WITH_N")
        self.assertEqual(L.trend(up[:2])["label"], "TREND_UNDEFINED")
        self.assertEqual(L.trend(up[1:] + [(8, [5, 5, 5, 5], [1] * 4)])["magnitude"], "MAGNITUDE_UNDEFINED")   # no N=1
        # magnitude 2.25x -> citable
        self.assertTrue(L.trend(up)["direction_citable"])

    def build_cells(self, agn_boost=lambda n: 1.0, drop=(), **over):
        cells = []
        for cfg in C.CONFIGS:
            n = 1 if cfg == "N1" else int(cfg[2:])
            for shape in "AB":
                lam = P.prior(n, shape)
                for arm in C.ARMS:
                    if (cfg, arm) in drop:
                        continue
                    l = lam * (agn_boost(n) if arm == "agnostic" else 1.0)
                    cells += key_cells(cfg, arm, shape, l, **over)
        return cells

    def test_aggregate_defined_ratio_and_trend(self):
        doc = L.aggregate(self.build_cells(agn_boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1)), None, {})
        e = doc["estimands"]["G_sym"]["TP4"]["A"]
        self.assertTrue(e["defined"])
        self.assertAlmostEqual(e["point"], 1.2, 6)
        t = doc["trends"]["G_sym|TP|A"]
        self.assertEqual(t["label"], "G_INCREASES_WITH_N")
        self.assertEqual(t["undefined_count"], 0)
        self.assertTrue(doc["trends"]["G_sym|TP|A"]["primary"] and not doc["trends"]["G_def|TP|A"]["primary"])
        self.assertTrue(doc["P_DP"]["broken"] and doc["P_DP"]["MGP13_TP_labels_not_citable"])      # same boost on DP

    def test_aggregate_flat_dp_keeps_p_dp(self):
        doc = L.aggregate(self.build_cells(), None, {})
        self.assertEqual(doc["trends"]["G_sym|DP|B"]["label"], "NO_TREND_DETECTED")
        self.assertFalse(doc["P_DP"]["broken"])

    def test_undefined_cells_listed_with_reasons_MGP34(self):
        cells = self.build_cells(drop={("TP2", "fused_sym")})
        doc = L.aggregate(cells, None, {})
        self.assertFalse(doc["estimands"]["G_sym"]["TP2"]["A"]["defined"])
        t = doc["trends"]["G_sym|TP|A"]
        self.assertEqual(t["undefined_count"], 1)
        self.assertIn("fused_sym:NOT_RUN", t["undefined"][0]["reasons"])
        self.assertTrue(doc["estimands"]["G_def"]["TP2"]["A"]["defined"])          # G_def NOT substituted for G_sym (MGP-38)
        self.assertEqual(doc["trends"]["G_def|TP|A"]["undefined_count"], 0)

    def test_knee_not_bracketed_and_harness_limited_make_g_undefined(self):
        cells = self.build_cells()
        for c in cells:
            m = c["meta"]
            if (m["config"], m["arm"], m["shape"], m["rung"]) == ("DP4", "agnostic", "A", "LO"):
                c["g"] = 0.2
            if (m["config"], m["arm"], m["shape"]) == ("DP2", "fused", "B"):
                c["harness_limited"] = True
        doc = L.aggregate(cells, None, {})
        self.assertFalse(doc["estimands"]["G_sym"]["DP4"]["A"]["defined"])
        self.assertTrue(any("KNEE_NOT_BRACKETED" in r for r in doc["estimands"]["G_sym"]["DP4"]["A"]["undefined_reasons"]))
        self.assertFalse(doc["estimands"]["G_def"]["DP2"]["B"]["defined"])
        self.assertTrue(doc["estimands"]["G_sym"]["DP2"]["B"]["defined"])        # fused_sym/agnostic unaffected

    def test_config_flags_and_pass_table_exclusion(self):
        cells = self.build_cells()
        doc = L.aggregate(cells, {"TP8:agnostic": {"included": False, "reason": "G1 FAIL"}},
                          {"TP4": {"memory_mismatch": True}, "DP2": {"gpu_contaminated": True}})
        self.assertFalse(doc["estimands"]["G_sym"]["TP4"]["A"]["defined"])
        self.assertIn("MEMORY_MISMATCH", doc["estimands"]["G_sym"]["TP4"]["A"]["undefined_reasons"])
        self.assertIn("GPU_CONTAMINATED", doc["estimands"]["G_sym"]["DP2"]["B"]["undefined_reasons"])
        self.assertFalse(doc["estimands"]["G_sym"]["TP8"]["A"]["defined"])
        self.assertTrue(doc["estimands"]["G_sym"]["TP8"]["A"]["undefined_reasons"][0].startswith("agnostic:M2G_EXCLUDED"))
        self.assertTrue(doc["estimands"]["G_def"]["TP8"]["A"]["undefined_reasons"][0].startswith("agnostic:M2G_EXCLUDED"))

    def test_r3_memory_not_tested_arm_makes_only_its_pairs_undefined(self):                  # R3
        cells = self.build_cells()
        flags = {"TP4": {"memory_not_tested_arms": ["d44"]}}
        doc = L.aggregate(cells, None, flags)
        for est in doc["estimands"].values():
            for shape in ("A", "B"):
                ent = est.get("TP4", {}).get(shape)
                if ent is None:
                    continue
                pair_has_d44 = any("d44" in r for r in ent["undefined_reasons"])
                if not ent["defined"]:
                    self.assertTrue(pair_has_d44 or any("NOT_RUN" in r or "KNEE" in r or "M2G" in r for r in ent["undefined_reasons"]))
        und = [(n, sh) for n, est in doc["estimands"].items() for sh, e in est.get("TP4", {}).items()
               if "d44:MEMORY_NOT_TESTED" in e["undefined_reasons"]]
        self.assertTrue(und)
        # a pair WITHOUT d44 stays defined (pair outside arm: unaffected)
        base = L.aggregate(self.build_cells(), None, {})
        for n, est in doc["estimands"].items():
            for sh, e in est.get("TP4", {}).items():
                if "d44:MEMORY_NOT_TESTED" not in e["undefined_reasons"]:
                    self.assertEqual(e["defined"], base["estimands"][n]["TP4"][sh]["defined"])

    def test_r3_mutation_ignoring_memory_not_tested_arms_is_killed(self):
        cells = self.build_cells()
        flags = {"TP4": {"memory_not_tested_arms": ["d44"]}}
        doc = L.aggregate(cells, None, flags)
        mut = L.aggregate(cells, None, {"TP4": {"memory_not_tested_arms": []}})                # = the removed R3 branch
        self.assertNotEqual(json.dumps(doc["estimands"], sort_keys=True), json.dumps(mut["estimands"], sort_keys=True))

    def test_r4_gpu_contaminated_is_scoped_to_pairs_containing_the_boot_arm(self):           # R4
        cells = self.build_cells()
        flags = {"DP2": {"gpu_contaminated": True, "contam_arms": ["d44"]}}
        doc = L.aggregate(cells, None, flags)
        base = L.aggregate(self.build_cells(), None, {})
        hit = clean = 0
        for n, est in doc["estimands"].items():
            for sh, e in est.get("DP2", {}).items():
                if "d44:GPU_CONTAMINATED" in e["undefined_reasons"]:
                    hit += 1
                    self.assertFalse(e["defined"])
                else:
                    clean += 1
                    self.assertFalse(any("GPU_CONTAMINATED" in r for r in e["undefined_reasons"]))
                    self.assertEqual(e["defined"], base["estimands"][n]["DP2"][sh]["defined"])
        self.assertTrue(hit and clean)
        # legacy flag without contam_arms keeps the conservative whole-config reading
        leg = L.aggregate(self.build_cells(), None, {"DP2": {"gpu_contaminated": True}})
        self.assertIn("GPU_CONTAMINATED", leg["estimands"]["G_sym"]["DP2"]["B"]["undefined_reasons"])

    def test_r4_build_config_flags_records_the_boot_arm(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "specs").mkdir()
            (root / "specs" / "b1.json").write_text(json.dumps({"boot_id": "b1", "config": "DP2", "arm": "d44", "steps": []}))
            (root / "boots" / "b1").mkdir(parents=True)
            (root / "boots" / "b1" / "contam.json").write_text(json.dumps({"gpu_contaminated": True}))
            fl = L.build_config_flags(root, [])
            self.assertEqual(fl["DP2"]["contam_arms"], ["d44"])

    def test_frontend_saturated_flag_is_do_not_cite_not_undefined(self):
        cells = self.build_cells()
        for c in cells:
            if c["meta"]["config"] == "DP8" and c["meta"]["rung"] == "HI" and c["meta"]["arm"] == "fused":
                c["frontend_saturated"] = True
        doc = L.aggregate(cells, None, {})
        e = doc["estimands"]["G_sym"]["DP8"]["A"]
        self.assertTrue(e["defined"] and e["frontend_saturated_do_not_cite_as_scaling"])

    def test_anchor_drift(self):
        cells = self.build_cells()
        cells += key_cells("N1", "fused", "A", 10.0 * P.prior(1, "A") / 10.0 * 1.04, rnd=2)
        cells += key_cells("N1", "fused_sym", "A", P.prior(1, "A"), rnd=2)
        cells += key_cells("N1", "agnostic", "A", P.prior(1, "A"), rnd=2)
        doc = L.aggregate(cells, None, {})
        self.assertTrue(doc["anchor"]["ANCHOR_DRIFT"])                         # fused moved +4 %
        self.assertTrue(doc["anchor"]["G_sym_drift_checked"])
        cells2 = self.build_cells() + key_cells("N1", "fused", "A", P.prior(1, "A"), rnd=2)
        doc2 = L.aggregate(cells2, None, {})
        self.assertFalse(doc2["anchor"]["ANCHOR_DRIFT"])
        self.assertFalse(doc2["anchor"]["G_sym_drift_checked"])
        self.assertIn("MGP-38", doc2["anchor"]["note"])

    def test_gpu_heterogeneous_block(self):
        cells = self.build_cells()
        for g, a in enumerate([10, 10.1, 10.2, 10.5, 10, 10, 10, 10]):
            c = cell_rec("N1", "fused", "A", "HI", 4386, a, 0.4)
            c["meta"].update(block="gpu_dev", gpu=g)
            cells.append(c)
        doc = L.aggregate(cells, None, {})
        self.assertTrue(doc["blocks"]["gpu_dev"]["GPU_HETEROGENEOUS"])         # 5 % > 3 %
        self.assertNotIn("TP4X", doc["configs"])                                # descriptive blocks never enter the keys

    def test_memory_check_three_conditions(self):
        ok = {a: dict(mamba=[96], max_total=[900000], eff_cap=96) for a in C.ARMS}
        self.assertFalse(L.memory_check(ok)["memory_mismatch"])
        bad = dict(ok, d44=dict(mamba=[48], max_total=[900000], eff_cap=96))
        self.assertTrue(L.memory_check(bad)["memory_mismatch"])                                   # (i)
        bad = dict(ok, d44=dict(mamba=[96], max_total=[10000], eff_cap=96))
        self.assertTrue(L.memory_check(bad)["memory_mismatch"])                                   # (ii)
        bad = dict(ok, d44=dict(mamba=[96], max_total=[860000], eff_cap=96))
        self.assertTrue(L.memory_check(bad)["memory_mismatch"])                                   # (iii) 4 % spread
        near = dict(ok, d44=dict(mamba=[96], max_total=[895000], eff_cap=96))
        self.assertFalse(L.memory_check(near)["memory_mismatch"])                                 # 0.6 % tolerated (not exact)

    def test_server_log_descriptive_never_changes_label(self):
        s = L.synthetic_scenarios()["S1_saturated_fifo"]
        d = L.server_log_descriptive(s["log"], 48)
        self.assertGreater(d["decode_lines"], 10)
        self.assertEqual(L.correct_class(s), "HI")


# =========================================================================== sampler (real /proc)
class Sampler(unittest.TestCase):
    def test_busy_vs_idle_child(self):
        with tempfile.TemporaryDirectory() as td:
            busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
            self.addCleanup(lambda: (busy.kill(), busy.wait()))
            out = Path(td) / "p.csv"
            s = subprocess.Popen([sys.executable, str(ENGINE / "scripts/multigpu/mg_proc_sampler.py"),
                                  "--root-pid", str(busy.pid), "--out", str(out), "--period", "0.5"])
            time.sleep(3.2)
            s.send_signal(signal.SIGTERM)
            s.wait(timeout=5)
            fe = L.frontend_stats(str(out), (0.0, 1e18))
            self.assertTrue(fe["frontend_saturated"], fe)
            self.assertGreater(fe["roles"]["tokenizer_manager"]["max_cores"], 0.9)
            idle = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            self.addCleanup(lambda: (idle.kill(), idle.wait()))
            out2 = Path(td) / "q.csv"
            s = subprocess.Popen([sys.executable, str(ENGINE / "scripts/multigpu/mg_proc_sampler.py"),
                                  "--root-pid", str(idle.pid), "--out", str(out2), "--period", "0.5"])
            time.sleep(2.2)
            s.send_signal(signal.SIGTERM)
            s.wait(timeout=5)
            self.assertFalse(L.frontend_stats(str(out2), (0.0, 1e18))["frontend_saturated"])


# =========================================================================== drivers (fake runner / dry-run)
FAKE_RUNNER_PY = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["MG_RESULTS_DIR"])
import mg_label as L, mg_plan as P, mg_const as C
spec = json.load(open(sys.argv[1]))
if "--dry-run" in sys.argv: sys.exit(0)
out = Path(spec["out_root"]); bd = out / "boots" / spec["boot_id"]; bd.mkdir(parents=True, exist_ok=True)
ratio = float(os.environ.get("FAKE_RATIO", "1.0"))
boost = float(os.environ.get("FAKE_AGN_BOOST", "1.0"))
N = spec["N"]; cap = 48 * spec["tp"] if spec["tp"] > 1 else 48
json.dump({"max_mamba_cache_size": [cap], "max_total_num_tokens": [900000 * spec["tp"]], "max_running_requests_requested": cap}, open(bd / "cell_meta.json", "w"))
json.dump({"status": "OK", "boot_id": spec["boot_id"], "t_health_epoch": 1}, open(bd / "boot.json", "w"))
for st in spec["steps"]:
    if st["type"] != "cell": continue
    lam = ratio * P.prior(N, st["shape"]) * (boost if spec["arm"] == "agnostic" else 1.0) * (1 + 0.004 * ((st["seed"] % 7) - 3))
    x, n = st["offered"], st["n_req"]
    reqs = L._fifo_cell(x, lam, n / x)
    cid = f"{spec['boot_id']}__{st['shape']}_{st['rung']}{'x' if st.get('ext') else ''}_s{st['seed']}"
    cd = out / "cells" / cid; cd.mkdir(parents=True, exist_ok=True)
    with open(cd / "requests.jsonl", "w") as f:
        for r in reqs: f.write(json.dumps(r) + "\n")
    meta = {"cell_id": cid, "boot_dir": f"boots/{spec['boot_id']}", "config": spec["config"], "mode": spec["mode"], "N": N,
            "arm": spec["arm"], "shape": st["shape"], "rung": st["rung"], "seed": st["seed"], "offered": x, "n_req": n,
            "ext": st.get("ext", 0), "round": st.get("round", 1), "block": st.get("block", "main"), "gpu": st.get("gpu")}
    json.dump(meta, open(cd / "cell.json", "w"))
    (cd / "cpu_stat_before.txt").write_text("nr_periods 10\nnr_throttled 0\n"); (cd / "cpu_stat_after.txt").write_text("nr_periods 110\nnr_throttled 0\n")
'''


class DriverWithFakeRunner(unittest.TestCase):
    def run_driver(self, td, ratio, boost, extra_args=(), configs_note=""):
        fake = Path(td) / "fake_runner.sh"
        (Path(td) / "fake.py").write_text(FAKE_RUNNER_PY)
        fake.write_text(f'#!/bin/bash\nexec python3 {td}/fake.py "$@"\n')
        out = Path(td) / "out"
        env = dict(os.environ, PDMUX_MG_RUNNER=str(fake), MG_RESULTS_DIR=str(ENGINE / "results/multigpu"),
                   FAKE_RATIO=str(ratio), FAKE_AGN_BOOST=str(boost), PYTHONDONTWRITEBYTECODE="1")
        raw = json.dumps({"cells": {}, "g5a_unit_test": "PASS"})
        tbl = Path(td) / "t.json"
        tbl.write_text(raw)
        import hashlib
        p = subprocess.run([sys.executable, str(ENGINE / "results/multigpu/mg_drive.py"), "m3", "--out", str(out),
                            "--pass-table", str(tbl), "--pass-sha256", hashlib.sha256(raw.encode()).hexdigest(),
                            *extra_args], env=env, capture_output=True, text=True)
        return p, out

    def test_no_extension_when_lambda_equals_prior_and_lo_shared_across_arms(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertEqual(doc["n_cells"], 365)
            self.assertTrue(all(not d["ext_shapes"] for d in doc["driver"]["decisions"].values()))
            e = doc["estimands"]["G_sym"]["TP8"]["A"]
            self.assertTrue(e["defined"], e)
            self.assertAlmostEqual(e["point"], 1.0, 2)
            for cfg in C.CONFIG_ORDER:                      # x_lo identical across the four arms (MGP-9)
                for sh in "AB":
                    xs = set()
                    for d in (out / "cells").glob(f"m3_{cfg}_*_boot2__{sh}_LO_s4386"):
                        m = json.loads((d / "cell.json").read_text())
                        xs.add((m["offered"], m["n_req"]))
                    self.assertEqual(len(xs), 1, (cfg, sh, xs))
            self.assertFalse(doc["anchor"]["ANCHOR_DRIFT"])
            self.assertEqual(doc["trends"]["G_sym|DP|A"]["label"], "NO_TREND_DETECTED")

    def test_joint_extension_when_lambda_far_above_prior(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.7, 1.0)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            dec = doc["driver"]["decisions"]
            self.assertTrue(all(d["ext_shapes"] == ["A", "B"] for d in dec.values()), dec)
            self.assertTrue(list((out / "specs").glob("m3_TP8_*_boot1ext.json")))
            specs = [json.loads(p.read_text()) for p in (out / "specs").glob("m3_TP8_*_boot1ext.json")]
            self.assertEqual(len(specs), 4)                     # all four arms jointly (MGP-9)
            # x_lo derives from the extended fused HI median, ext=1 on LO cells
            self.assertTrue(all(s["ext"] == 1 for sp in (out / "specs").glob("m3_TP8_*_boot2.json")
                                for s in json.loads(sp.read_text())["steps"] if s["type"] == "cell"))

    def test_constant_boost_tiny_slope_is_not_citable_under_3pct_rule(self):
        """Deterministic synthetic data: even a ~0.1 % N-dependence has a CI excluding 0; the magnitude rule (§4-4) is
        what keeps it from being cited as a direction."""
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.15)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertAlmostEqual(doc["estimands"]["G_sym"]["DP2"]["A"]["point"], 1.15, 1)
            t = doc["trends"]["G_sym|TP|A"]
            self.assertEqual(t["undefined_count"], 0)
            if t["label"] != "NO_TREND_DETECTED":
                self.assertEqual(t["magnitude"], "DIRECTION_NOT_CITABLE_UNDER_3PCT")
                self.assertLess(abs(t["magnitude_ratio"] - 1.0), 0.03)

    def test_hard_cap_admission_records_not_attempted(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0, ["--hard-cap-s", "1500"])
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            st = json.loads((out / "DRIVER_STATUS.json").read_text())
            self.assertGreater(st["not_attempted"], 20)
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertTrue(doc["driver"]["not_attempted"])
            self.assertLess(doc["n_cells"], 365)

    def test_missing_pass_table_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            p = subprocess.run([sys.executable, str(ENGINE / "results/multigpu/mg_drive.py"), "m3", "--out", str(out),
                                "--pass-table", str(Path(td) / "none.json"), "--pass-sha256", "0" * 64],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3)
            self.assertIn("ABORT_M2G_NOT_PASSED", (out / "ABORT").read_text())


class AssemblePassTable(unittest.TestCase):
    """mg_drive.assemble_pass_table reads Job files and delegates scoring + the §3-1 mapping to m2g_correctness_gate."""

    def test_nothing_run_excludes_everything_not_tested(self):
        with tempfile.TemporaryDirectory() as td:
            t = D.assemble_pass_table(Path(td))
            self.assertTrue(all(not v["included"] for v in t["cells"].values()))
            self.assertIn("NOT_TESTED", t["cells"]["TP2:fused"]["reason"])
            self.assertEqual(len((Path(td) / "M2G_PASS_TABLE.json.sha256").read_text().strip()), 64)

    @staticmethod
    def build(out, drop_boot=()):
        for k in range(5):
            d = out / "boots" / f"m2g_refuse_{k}"
            d.mkdir(parents=True)
            (d / "boot.json").write_text(json.dumps({"kind": "g0ii", "status": "REFUSED_AS_EXPECTED", "t_health_epoch": None,
                                                     "server_exit_code": 1, "health_wait_cap_s": 1800}))
            (d / "server.log").write_text("RuntimeError: PD-mux with tp_size=2 is only defined ...\n")
        d = out / "boots" / "m2g_autogrid_TP2"
        d.mkdir(parents=True)
        (d / "boot.json").write_text(json.dumps({"kind": "g0iii", "t_health_epoch": None, "server_exit_code": 1, "health_wait_cap_s": 1800}))
        (d / "server.log").write_text("AssertionError: PD-mux auto grid (divide_sm) ...\n")
        (out / "gates").mkdir(exist_ok=True)
        for cfg in ("N1",) + P.M2G_CFGS:
            n = 1 if cfg == "N1" else int(cfg[2:])
            for arm in C.ARMS:
                d = out / "boots" / f"m2g_{cfg}_{arm}"
                d.mkdir(parents=True)
                if (cfg, arm) in drop_boot:
                    (d / "boot.json").write_text(json.dumps({"status": "BOOT_FAILED", "health_wait_cap_s": 1800, "server_exit_code": 1}))
                else:
                    (d / "boot.json").write_text(json.dumps({"status": "OK", "t_health_epoch": 1, "boot_seconds": 100,
                                                             "health_wait_cap_s": 1800, "server_exit_code": None, "steps": []}))
                (d / "server.log").write_text("Capture cuda graph begin\nCapture cuda graph end\n")
                (d / "alive_after.json").write_text(json.dumps(__import__("m2g_correctness_gate").make_alive_record(d.name, None, 200)))     # F2
                (d / "piecewise.json").write_text(json.dumps({"disable_piecewise_cuda_graph": "False"}))
                cap = 48 * n if cfg.startswith("TP") else 48
                (d / "cell_meta.json").write_text(json.dumps({"max_mamba_cache_size": [cap], "max_total_num_tokens": [900000 * n],
                                                              "max_running_requests_requested": cap}))
                (d / "proc_counts_post_boot.json").write_text(json.dumps({"0": 1}))
                (d / "contam_pre.json").write_text(json.dumps({"snapshot": {"gpus": {"0": {"used_mib": 5.0, "n_compute_procs": 0}}}}))
            if cfg != "N1":
                (out / "gates" / f"g1_{cfg}.json").write_text(json.dumps({"verdicts": {"fused|agnostic": "PASS", "fused|d44": "PASS"}}))
            if cfg.startswith("TP"):
                (out / "gates" / f"g3_TP{n}.json").write_text(json.dumps({"verdict": "PASS"}))
                (out / "gates" / f"g2_{cfg}.json").write_text(json.dumps({"verdict": "PASS"}))
                for arm in C.G3B_ARMS:
                    cd = out / "cells" / f"m2g_{cfg}_{arm}__A_G3B_s4386"
                    cd.mkdir(parents=True)
                    reqs = L._fifo_cell(3.0, 3.6, 600.0)
                    (cd / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in reqs) + "\n")

    def test_g4_not_tested_when_boot_not_ok(self):
        """A boot that never became healthy has no capture to judge: G4 NOT_TESTED, even if its log has capture lines."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out, drop_boot={("TP2", "fused")})
            t = D.assemble_pass_table(out)
            self.assertIn("G4 capture NOT_TESTED", t["cells"]["TP2:fused"]["reasons"])
            self.assertNotIn("G4 capture NOT_TESTED", " ".join(t["cells"]["TP2:d44"]["reasons"]))   # negative control: healthy boot

    def test_full_pass_and_per_gate_exclusions(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out)
            t = D.assemble_pass_table(out)
            self.assertTrue(all(t["cells"][f"TP{n}:{a}"]["included"] for n in (2, 4, 8) for a in C.ARMS),
                            t["cells"]["TP2:fused"])
            self.assertTrue(all(t["cells"][f"N1:{a}"]["included"] for a in C.ARMS))
            # DP: no telemetry for the realized-partition check -> G5(b) NOT_TESTED excludes the whole DP=N (§3-1)
            self.assertTrue(all(not t["cells"][f"DP{n}:fused"]["included"] and "G5(b)" in t["cells"][f"DP{n}:fused"]["reason"]
                                for n in (2, 4, 8)))
            self.assertFalse(t["configs"]["TP2"]["memory_mismatch"])
            self.assertTrue(t["nvml_procs_visible"])
            self.assertEqual(t["nvml_expected_procs"]["TP2:d44"], {"0": 1})
            # negative controls: each failed gate removes exactly its registered range
            (out / "gates" / "g1_TP2.json").write_text(json.dumps({"verdicts": {"fused|agnostic": "FAIL", "fused|d44": "UNINFORMATIVE"}}))
            (out / "gates" / "g3_TP4.json").write_text(json.dumps({"verdict": "FAIL"}))
            (out / "boots" / "m2g_TP8_fused_sym" / "boot.json").write_text(json.dumps({"status": "BOOT_FAILED", "health_wait_cap_s": 1800,
                                                                                         "server_exit_code": 1}))
            (out / "boots" / "m2g_refuse_0" / "boot.json").write_text(json.dumps({"kind": "g0ii", "t_health_epoch": 1, "server_exit_code": None,
                                                                                    "health_wait_cap_s": 1800}))
            t = D.assemble_pass_table(out)
            self.assertTrue(all(not t["cells"][f"TP{n}:fused"]["included"] for n in (2, 4, 8)))        # G0(ii) -> TP axis out
            (out / "boots" / "m2g_refuse_0" / "boot.json").write_text(json.dumps({"kind": "g0ii", "t_health_epoch": None,
                                                                                    "server_exit_code": 1, "health_wait_cap_s": 1800}))
            t = D.assemble_pass_table(out)
            self.assertTrue(t["cells"]["TP2:fused"]["included"] and t["cells"]["TP2:d44"]["included"])  # UNINFORMATIVE excludes nothing
            self.assertFalse(t["cells"]["TP2:agnostic"]["included"])
            self.assertTrue(t["cells"]["TP2:fused_sym"]["included"])
            self.assertFalse(t["cells"]["TP4:fused"]["included"])                                       # G3 FAIL: whole TP4
            self.assertFalse(t["cells"]["TP8:fused_sym"]["included"])                                   # G0(i): that arm only
            self.assertTrue(t["cells"]["TP8:fused"]["included"] and t["cells"]["TP8:agnostic"]["included"])
            self.assertTrue(t["configs"]["TP8"]["g_sym_undefined"])

    def test_short_health_wait_is_not_tested_not_fail(self):                                           # MGP-23
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out)
            (out / "boots" / "m2g_TP2_d44" / "boot.json").write_text(json.dumps({"status": "BOOT_TIMEOUT", "health_wait_cap_s": 300,
                                                                                  "server_exit_code": None}))
            t = D.assemble_pass_table(out)
            r = t["cells"]["TP2:d44"]
            self.assertFalse(r["included"])
            self.assertTrue(r["not_tested"])

    def test_m3_reads_the_table_it_got(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out)
            t = D.assemble_pass_table(out)
            self.assertEqual(P.excluded_set(t) & {"TP2:fused", "TP8:d44"}, set())
            self.assertIn("TP2", P.runnable_configs(t))
            self.assertNotIn("DP2", P.runnable_configs(t))


# =========================================================================== rev6 harness audit F1–F17 (2026-10-12)
def etime_ok(cells, v):
    for c in cells:
        if c["meta"]["rung"] == "HI":
            c["etime"] = {"status": "OK", "e_time": v}
    return cells


class ExtensionRules(unittest.TestCase):                                                   # F10
    def cells(self, trigger):
        out = []
        for arm in ("fused", "fused_sym", "agnostic", "d44"):
            if arm == "agnostic" and trigger:
                out += key_cells("TP2", arm, "A", 10.0, g_hi=0.05, ext=0)[:4]               # first pass NOT saturated -> triggers
                out += [cell_rec("TP2", arm, "A", "HI", s, 12.0, 0.4, ext=1) for s in C.HI_SEEDS]
                out += [cell_rec("TP2", arm, "A", "LO", s, 7.0, 0.01, ext=1) for s in C.LO_SEEDS]
            else:
                out += key_cells("TP2", arm, "A", 10.0)
        return out

    def test_arm_without_extension_cells_after_joint_trigger_is_ext_not_run(self):
        doc = L.aggregate(self.cells(True), None, {})
        lab = doc["keys"]["TP2|fused_sym|A|r1"]
        self.assertEqual(lab["hi_label"], "HI_NOT_JUDGED")
        self.assertIn("HI_NOT_JUDGED(EXT_NOT_RUN)", lab["reasons"])
        self.assertIsNone(lab["lambda_star"])
        e = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertFalse(e["defined"])                                                   # extended agnostic vs first-pass fused_sym: no pairing
        self.assertTrue(any("fused_sym:KNEE_NOT_BRACKETED" in r for r in e["undefined_reasons"]))
        self.assertEqual(doc["keys"]["TP2|agnostic|A|r1"]["hi_label"], "HI_SATURATED")  # the extended arm itself is judged

    def test_no_trigger_no_penalty_and_old_behaviour_would_pair(self):
        doc = L.aggregate(self.cells(False), None, {})
        self.assertTrue(doc["estimands"]["G_sym"]["TP2"]["A"]["defined"])
        # mutant: without ext_fired the not-extended arm reads as HI_SATURATED (the pre-F10 conditional reversal)
        cs = [c for c in self.cells(True) if c["meta"]["arm"] == "fused_sym"]
        self.assertEqual(L.key_labels(cs)["hi_label"], "HI_SATURATED")
        self.assertEqual(L.key_labels(cs, ext_fired=True)["hi_label"], "HI_NOT_JUDGED")


class PDPAndDisclosures(unittest.TestCase):                                                # F9, F12, F13, F14
    build = Labels.build_cells

    def test_p_dp_three_values_and_tp_disclosure(self):
        doc = L.aggregate(self.build(), None, {})
        self.assertEqual(doc["P_DP"]["status"], "HELD")
        self.assertNotIn("P-DP 미검정", doc["trends"]["G_sym|TP|A"]["disclosures"])
        drop = {(f"DP{n}", "fused_sym") for n in (2, 4, 8)}
        doc = L.aggregate(self.build(drop=drop), None, {})
        self.assertEqual(doc["P_DP"]["status"], "UNTESTED")
        self.assertFalse(doc["P_DP"]["broken"])                                           # the old code said 'not broken' here
        self.assertTrue(doc["P_DP"]["TP_labels_need_P_DP_UNTESTED_disclosure"])
        for sh in "AB":
            self.assertIn("P-DP 미검정", doc["trends"][f"G_sym|TP|{sh}"]["disclosures"])
        self.assertIn("P-DP 미검정", L.render_text(doc))
        doc = L.aggregate(self.build(agn_boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1)), None, {})
        self.assertEqual(doc["P_DP"]["status"], "BROKEN")

    def test_mix_failed_and_e_time_undefined_propagate_to_estimand_and_trend(self):
        cs = self.build()
        for c in cs:
            if c["meta"]["arm"] in C.PDMUX_ARMS and c["meta"]["rung"] == "HI":
                c["etime"] = {"status": "OK", "e_time": 0.9}
        doc = L.aggregate(cs, None, {})
        e = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertEqual(e["partition"], {"agnostic": "PARTITION_EXERCISED"})
        self.assertFalse(e["partition_exercised_unknown"])
        for c in cs:
            m = c["meta"]
            if (m["config"], m["arm"], m["shape"], m["seed"], m["rung"]) == ("TP2", "agnostic", "A", C.HI_SEEDS[1], "HI"):
                c["etime"] = {"status": "MIX_FAILED", "e_time": None, "reasons": ["x"]}
        doc = L.aggregate(cs, None, {})
        e = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertEqual(e["partition"]["agnostic"], "E_TIME_UNDEFINED")
        self.assertTrue(e["partition_exercised_unknown"] and e["mix_failed_arms"] == ["agnostic"])
        self.assertTrue(e["defined"])                                                     # a flag, not an exclusion
        t = doc["trends"]["G_sym|TP|A"]
        self.assertTrue(t["partition_exercised_unknown"])
        self.assertIn("E_TIME_UNDEFINED", L.render_text(doc))

    def test_unmeasured_throttle_sendlag_frontend_and_missing_roles_are_explicit(self):
        lab = L.key_labels(key_cells("DP2", "agnostic", "A", 10.0, throttle_unmeasured=True, send_lag_p99_ms=None,
                                     frontend_unmeasured=True))
        gaps = " ".join(lab["measurement_gaps"])
        for tag in ("THROTTLE_UNMEASURED", "SENDLAG_UNMEASURED", "FRONTEND_UNMEASURED"):
            self.assertIn(tag, gaps)
        ok = key_cells("DP2", "agnostic", "A", 10.0, throttle_unmeasured=False, send_lag_p99_ms=1.0, frontend_unmeasured=False,
                       frontend={"roles": {"tokenizer_manager": {}, "detokenizer": {}}, "frontend_saturated": False})
        for c in ok:
            c.update(throttle_unmeasured=False, send_lag_p99_ms=1.0, frontend_unmeasured=False)    # LO cells carry no kw in key_cells
            c["meta"]["mode"] = "DP"
        lab = L.key_labels(ok)
        self.assertIn("FRONTEND_ROLE_MISSING:dp_controller", lab["measurement_gaps"])      # DP needs the controller role
        self.assertFalse(any("UNMEASURED" in g for g in lab["measurement_gaps"]))
        for c in ok:
            c["meta"]["mode"] = "TP"
        self.assertEqual(L.key_labels(ok)["measurement_gaps"], [])                         # negative control: TP needs only two roles

    def test_m2g_config_flags_travel_into_labels(self):
        mc = {"TP2": {"flags": ["TP_NUMERIC_DIVERGENT", "agnostic:G1_UNINFORMATIVE"], "memory_mismatch": False},
              "TP4": {"flags": ["G2_NOT_TESTED"], "memory_mismatch": False},
              "TP8": {"flags": ["MEMORY_MISMATCH(NOT_TESTED)"], "memory_mismatch": True}}
        doc = L.aggregate(self.build(), None, {}, mc)
        e2 = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertTrue(e2["defined"])
        self.assertIn("TP_NUMERIC_DIVERGENT", e2["disclosures"])
        self.assertIn("agnostic:G1_UNINFORMATIVE", e2["disclosures"])
        self.assertIn("G2_NOT_TESTED", doc["estimands"]["G_sym"]["TP4"]["A"]["disclosures"])
        e8 = doc["estimands"]["G_sym"]["TP8"]["A"]
        self.assertFalse(e8["defined"])                                                   # M2g MEMORY_MISMATCH carries (G5(c) meaning unified)
        self.assertIn("MEMORY_MISMATCH", e8["undefined_reasons"])
        t = doc["trends"]["G_sym|TP|A"]
        self.assertIn("TP_NUMERIC_DIVERGENT", t["disclosures"])                           # beside the TP trend (MGH-3)
        self.assertEqual(doc["estimands"]["G_sym"]["DP2"]["A"]["disclosures"], [])


class ConfigFlagScopes(unittest.TestCase):                                                 # F5, F11, F14
    def mk(self, td, boots):
        """boots: (boot_id, config, arm, block, round, contam, unverified, cell_meta|None)"""
        root = Path(td)
        (root / "specs").mkdir(exist_ok=True)
        for bid, cfg, arm, block, rnd, contam, unv, meta in boots:
            (root / "specs" / f"{bid}.json").write_text(json.dumps(
                {"boot_id": bid, "config": cfg, "arm": arm, "block": block,
                 "steps": [{"type": "cell", "round": rnd}]}))
            bd = root / "boots" / bid
            bd.mkdir(parents=True)
            if contam:
                (bd / "contam.json").write_text(json.dumps({"gpu_contaminated": True}))
            if unv:
                (bd / "contam_pre.json").write_text(json.dumps({"snapshot": {"available": False},
                                                                "verdict": {"contam_unverified": True, "verified": False}}))
            if meta:
                (bd / "cell_meta.json").write_text(json.dumps(meta))
        return root

    OKM = {"max_mamba_cache_size": [48], "max_total_num_tokens": [900000], "max_running_requests_requested": 48}

    def test_gpu_dev_numa_and_anchor_do_not_erase_round1_main_G(self):
        with tempfile.TemporaryDirectory() as td:
            root = self.mk(td, [
                ("m3_gpudev_gpu3", "N1", "fused", "gpu_dev", 1, True, False, None),
                ("m3_N1_fused_anchor2", "N1", "fused", "anchor", 2, True, False, None),
                ("m3_TP4X_fused_numa", "TP4X", "fused", "numa", 1, True, False, None),
                ("m3_N1_fused_boot1", "N1", "fused", "main", 1, False, False, self.OKM)])
            fl = L.build_config_flags(root, [])
            self.assertFalse(fl["N1"]["gpu_contaminated"])                                # round-1 main untouched
            self.assertTrue(fl["N1"]["by_group"]["r2/anchor"]["gpu_contaminated"])
            self.assertTrue(fl["N1"]["by_group"]["r1/gpu_dev"]["gpu_contaminated"])
            self.assertNotIn("r1/main", fl.get("TP4X", {}).get("by_group", {}))

    def test_or_summation_within_a_group_and_memory_per_boot_set(self):
        with tempfile.TemporaryDirectory() as td:
            bad = dict(self.OKM, max_mamba_cache_size=[24])
            root = self.mk(td, [
                ("a_boot1", "TP2", "fused", "main", 1, False, False, self.OKM),
                ("a_boot2", "TP2", "fused", "main", 1, True, False, bad),                  # the LATER boot is the dirty/mismatched one
                ("b_boot1", "TP2", "agnostic", "main", 1, False, False, self.OKM)])
            fl = L.build_config_flags(root, [])["TP2"]
            self.assertTrue(fl["gpu_contaminated"] and fl["memory_mismatch"])               # old code: last-writer-wins on the facts
            self.assertEqual(fl["memory_not_tested_arms"], ["fused_sym", "d44"])           # expected arms without facts: NOT_TESTED, not FAIL
            fl2 = L.build_config_flags(root, [], excluded={"TP2:fused_sym", "TP2:d44"})["TP2"]
            self.assertEqual(fl2["memory_not_tested_arms"], [])

    def test_contam_unverified_is_a_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = self.mk(td, [("c_boot1", "DP2", "fused", "main", 1, False, True, None)])
            fl = L.build_config_flags(root, [])["DP2"]
            self.assertTrue(fl["contam_unverified"] and not fl["gpu_contaminated"])
            self.assertEqual(fl["unverified_boots"], ["c_boot1"])
            fl = dict(fl, memory_not_tested_arms=[])        # R3: this fake boot set has no memory facts for the other arms; isolate the disclosure
            doc = L.aggregate(Labels.build_cells(self), None, {"DP2": fl})
            self.assertIn("CONTAM_UNVERIFIED", doc["estimands"]["G_sym"]["DP2"]["A"]["disclosures"])


class PlanF4F15F17(unittest.TestCase):
    def test_g5a_gpu_boot_is_in_the_m2g_plan(self):                                         # F4
        b = [x for x in P.m2g_boots() if x["boot_id"] == "m2g_g5a_auto_N1"]
        self.assertEqual(len(b), 1)
        self.assertEqual((b[0]["arm"], b[0]["config"], b[0]["config_rel"]), ("agnostic", "N1", C.AUTO_GRID_CONFIG_REL))
        self.assertTrue((ENGINE / C.AUTO_GRID_CONFIG_REL).exists())
        args = B.build_server_args(dict(b[0]), 1234, str(ENGINE / C.AUTO_GRID_CONFIG_REL))
        self.assertIn(str(ENGINE / C.AUTO_GRID_CONFIG_REL), args)
        self.assertNotIn(str(ENGINE / C.AGNOSTIC_CONFIG_REL), args)

    def test_budget_audited_model_unchanged_and_calibration_adds_time(self):                # F15
        a = P.m3_budget()
        self.assertAlmostEqual(a["total_h"], 29.96, 1)                                      # the audited number must not move
        v = P.m3_budget(calib=P.CALIBS["vessl_boot"])
        self.assertAlmostEqual(v["total_h"] - a["total_h"], 70 * C.EST_VESSL_BOOT_EXTRA_S / 3600.0, 3)
        u = P.m3_budget(calib=P.CALIBS["vessl_boot_cell"])
        self.assertGreater(u["total_h"], v["total_h"])
        s3 = P.m3_budget(ratio_arm_of=P.SCENARIOS["S3 extension: agnostic 1.48"])
        self.assertGreater(s3["boots"], a["boots"])                                          # ONE arm above 1.42 extends all four arms
        self.assertEqual(s3["boots"] - a["boots"], 7 * 4)

    def test_forecast_domain(self):
        rows = {(r["scenario"], r["calib"]): r for r in P.m3_forecast_table()}
        r = rows[("S1 lambda=prior (all)", "audited")]
        self.assertEqual((r["n_done"], r["trend_TP"], r["trend_DP"], r["anchor_boots_done"]), (70, "defined", "defined", 3))
        r = rows[("S1 lambda=prior (all)", "vessl_boot")]
        self.assertEqual(r["anchor_boots_done"], 0)                                          # the drift-measuring anchor is what gets cut (MGH-6)
        r = rows[("S3 extension: agnostic 1.48", "audited")]
        self.assertEqual((r["trend_TP"], r["trend_DP"]), ("TREND_UNDEFINED", "TREND_UNDEFINED"))   # MGH-7
        self.assertEqual(r["N_defined_TP"], [1, 8])
        blk = {(b["scenario"], b["calib"], b["k"]): b for b in P.block_plan_table()}
        self.assertLess(blk[("S1 lambda=prior (all)", "audited", 3)]["max_block_h"], 30.0)
        self.assertGreater(blk[("S1 lambda=prior (all)", "audited", 3)]["total_h"], P.m3_budget()["total_h"])   # k anchors + k preflights

    def test_pass_table_three_way_digest(self):                                             # F17
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.json"
            raw = json.dumps({"cells": {}, "g5a_unit_test": "PASS"})
            p.write_text(raw)
            sha = hashlib.sha256(raw.encode()).hexdigest()
            cs = Path(td) / "t.json.sha256"
            cs.write_text(sha + "\n")
            self.assertIsNone(P.load_pass_table(p, sha, cs)[1])                              # env == committed == bytes
            cs.write_text("0" * 64 + "\n")
            self.assertEqual(P.load_pass_table(p, sha, cs)[1], "ABORT_M2G_NOT_PASSED")      # committed differs
            cs.unlink()
            self.assertEqual(P.load_pass_table(p, sha, cs)[1], "ABORT_M2G_NOT_PASSED")       # committed missing
            self.assertIsNone(P.load_pass_table(p, sha)[1])                                  # legacy 2-way call still works
            self.assertEqual(P.load_pass_table(p, "1" * 64, None)[1], "ABORT_M2G_NOT_PASSED")


class BootLiveness(unittest.TestCase):                                                      # F2, F5 unit level
    class Fake:
        def __init__(self, rc):
            self.rc = rc

        def poll(self):
            return self.rc

    def rec(self, rc, td):
        import m2g_correctness_gate as G
        steps = [{"type": "cmd"}]
        B.write_alive(td, "b1", self.Fake(rc), 1, steps, [{"step": steps[0], "status": "OK"}])
        return G.alive_after(Path(td)), json.loads((Path(td) / "alive_after.json").read_text())

    def test_alive_record_cases(self):
        with tempfile.TemporaryDirectory() as td:
            alive, rec = self.rec(137, td)                                                  # exited: never asked, never alive
            self.assertFalse(alive)
            self.assertEqual((rec["poll_returncode"], rec["after_step"]["status"], rec["steps_total"]), (137, "OK", 1))
        with tempfile.TemporaryDirectory() as td:
            alive, rec = self.rec(None, td)                                                 # running but /health unreachable: not alive
            self.assertFalse(alive)
            self.assertIsNone(rec["poll_returncode"])
        with tempfile.TemporaryDirectory() as td:                                           # writer failure -> absent record, never 'alive'
            import m2g_correctness_gate as G
            B.write_alive(Path("/proc/forbidden"), "b", self.Fake(None), 1, [], [])
            self.assertIsNone(G.alive_after(Path("/proc/forbidden")))

    def test_nvml_unavailable_is_contam_unverified_not_clean(self):
        v = B.contamination_verdict({"available": False, "error": "x"}, False)
        self.assertTrue(v["contam_unverified"] and v["flag"] == "CONTAM_UNVERIFIED" and not v["verified"])
        self.assertFalse(v["gpu_contaminated"])
        self.assertFalse(B.contamination_verdict({"available": True, "gpus": {"0": {"used_mib": 1.0, "n_compute_procs": 0}}},
                                                  False)["contam_unverified"])


# ---- F8: the JOB path (mg_drive.run_m2g + assemble_pass_table) against a fake runner and a fake torchrun
FAKE_M2G_RUNNER_PY = r'''
import json, os, sys, time
from pathlib import Path
ENG = Path(os.environ["MG_ENGINE"])
sys.path[:0] = [str(ENG / "results/multigpu"), str(ENG / "scripts/multigpu"), str(ENG / "tests")]
import mg_label as L, mg_const as C, mg_realized_mix as RM
import m2g_correctness_gate as G
import test_mg_realized_mix as T
spec = json.load(open(sys.argv[1]))
if "--dry-run" in sys.argv: sys.exit(0)
out = Path(spec["out_root"]); bid = spec["boot_id"]; bd = out / "boots" / bid; bd.mkdir(parents=True, exist_ok=True)
with open(os.environ["FAKE_ORDER_LOG"], "a") as f: f.write("BOOT " + bid + "\n")
dead_final = os.environ.get("FAKE_DEAD", "").split(",")
no_alive = os.environ.get("FAKE_NO_ALIVE", "").split(",")
kind = spec.get("kind"); cfg = spec["config"]; arm = spec["arm"]; n = spec["N"]
log = bd / "server.log"
if kind in ("g0ii", "g0iii"):
    json.dump({"kind": kind, "status": "REFUSED_AS_EXPECTED", "t_health_epoch": None, "server_exit_code": 1, "health_wait_cap_s": 1800}, open(bd / "boot.json", "w"))
    log.write_text(("RuntimeError: " + spec["expect_refusal"] + " ...\n") + (G.DEVICE_ASSERT_TEXT + "\n" if kind == "g0iii" else ""))
    sys.exit(0)
json.dump({"status": "OK", "t_health_epoch": 1, "boot_seconds": 100, "health_wait_cap_s": 1800, "server_exit_code": None, "steps": []}, open(bd / "boot.json", "w"))
log.write_text("Capture cuda graph begin\nCapture cuda graph end\n")
json.dump({"disable_piecewise_cuda_graph": "False"}, open(bd / "piecewise.json", "w"))
cap = 48 * n if spec["mode"] == "TP" else 48
json.dump({"max_mamba_cache_size": [cap], "max_total_num_tokens": [900000 * (n if spec["mode"] == "TP" else 1)], "max_running_requests_requested": cap}, open(bd / "cell_meta.json", "w"))
json.dump({"0": 1}, open(bd / "proc_counts_post_boot.json", "w"))
json.dump({"snapshot": {"available": True, "gpus": {"0": {"used_mib": 5.0, "n_compute_procs": 0}}}, "verdict": {"verified": True}}, open(bd / "contam_pre.json", "w"))
if bid not in no_alive:                                    # the gate module's own record writer: one schema
    dead = bid in dead_final
    json.dump(G.make_alive_record(bid, -9 if dead else None, None if dead else 200), open(bd / "alive_after.json", "w"))
(out / "gates").mkdir(exist_ok=True)
json.dump({"outputs": {f"r{i}": [1, 2, 3, 4 + i] for i in range(64)}, "errors": 0, "config": cfg, "arm": arm,
           "g1_window_perf_counter_s": [-0.5, 3.5]}, open(out / "gates" / f"g1_out_{cfg}_{arm}.json", "w"))
if arm in C.PDMUX_ARMS and spec["mode"] in ("TP", "DP"):
    mode = spec["mode"].lower()
    rows, sm = (T.AGN_ROWS, T.AGN) if arm == "agnostic" else (T.R2_ROWS, T.R2)
    for tf, gf in zip(RM.rank_files(str(bd / "tel.jsonl"), mode, n), RM.rank_files(str(bd / "green.json"), mode, n)):
        T.write_tel(tf, rows, sm)
        json.dump(T.green(sm), open(gf, "w"))
    json.dump({"t0_mono": -0.5, "t1_mono": 3.5, "status": "OK"}, open(bd / f"cmd_g1_{arm}.window.json", "w"))
if spec["mode"] == "TP" and arm in C.G3B_ARMS:
    cd = out / "cells" / f"{bid}__A_G3B_s4386"; cd.mkdir(parents=True, exist_ok=True)
    with open(cd / "requests.jsonl", "w") as f:
        for r in L._fifo_cell(3.0, 3.6, 600.0): f.write(json.dumps(r) + "\n")
'''

FAKE_TORCHRUN_PY = r'''
import json, os, sys, time
from pathlib import Path
a = sys.argv[1:]
n = int(a[a.index("--nproc-per-node") + 1]); out = Path(a[a.index("--out") + 1])
with open(os.environ["FAKE_ORDER_LOG"], "a") as f: f.write(f"G3 TP{n} {out.name}\n")
if f"TP{n}" in os.environ.get("FAKE_G3_BROKEN", "").split(","):
    out.mkdir(parents=True, exist_ok=True)                  # directory exists, no rank file, no HANG: the harness never ran the script
    sys.exit(7)
out.mkdir(parents=True, exist_ok=True)
for r in range(n):
    json.dump({"rank": r, "world": n, "repeats": 500, "completed": 500, "errors": 0, "elapsed_s": 1.0}, open(out / f"rank{r}.json", "w"))
'''


class M2gJobPath(unittest.TestCase):
    """F8: order (G3 first), liveness (F2), exposure (F3), G3 launcher failure (F1) through the real run_m2g / assemble_pass_table."""

    def run_job(self, **env):
        from unittest import mock
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        (root / "fake_runner.py").write_text(FAKE_M2G_RUNNER_PY)
        (root / "fake_runner.sh").write_text(f'#!/bin/bash\nexec python3 {root}/fake_runner.py "$@"\n')
        (root / "fake_torchrun.py").write_text(FAKE_TORCHRUN_PY)
        (root / "fakepy.sh").write_text(f'#!/bin/bash\nif [ "$1" = "-c" ]; then exit 0; fi\n'
                                        f'if [ "$1" = "-m" ]; then shift 2; exec python3 {root}/fake_torchrun.py "$@"; fi\nexit 9\n')
        os.chmod(root / "fakepy.sh", 0o755)
        order = root / "order.log"
        e = {"MG_ENGINE": str(ENGINE), "FAKE_ORDER_LOG": str(order), "PYTHONDONTWRITEBYTECODE": "1"}
        e.update(env)
        out = root / "out"
        out.mkdir()
        ctx = D.Ctx(out, False, time.time() + 8 * 3600, time.time(), "m2g")
        ctx.g3_python = str(root / "fakepy.sh")                  # `fakepy.sh -m torch.distributed.run ...` -> fake_torchrun.py
        ctx.g5a_argv = [sys.executable, "-c", "print('g5a ok')"]
        with mock.patch.dict(os.environ, e), mock.patch.object(D, "RUNNER", str(root / "fake_runner.sh")):
            table = D.run_m2g(ctx)
        return table, out, order.read_text().splitlines(), ctx

    def test_order_g3_before_every_serving_boot_and_clean_pass(self):
        table, out, order, ctx = self.run_job()
        kinds = [ln.split()[0] for ln in order]
        self.assertEqual(kinds.count("G3"), 9)                                                # 3 TP x 3 combinations
        self.assertEqual(max(i for i, k in enumerate(kinds) if k == "G3") + 1, 9)             # all G3 lines precede ...
        self.assertTrue(all(k == "BOOT" for k in kinds[9:]))                                  # ... every BOOT line
        self.assertEqual(sum(1 for k in kinds if k == "BOOT"), len(P.m2g_boots()))
        self.assertTrue(all(table["cells"][f"TP{n}:{a}"]["included"] for n in (2, 4, 8) for a in C.ARMS),
                        {k: v["reason"] for k, v in table["cells"].items() if k.startswith("TP") and not v["included"]})
        self.assertEqual(table["g5a_unit_test"], "PASS")
        self.assertEqual(table["g5a_gpu_descriptive"], "RECORDED")                           # F4: the GPU boot ran and was read
        self.assertTrue((out / "gates" / "g3_launch.json").exists() and len(list((out / "gates" / "g3_logs").glob("run_g3_TP*.txt"))) == 3)
        self.assertTrue((out / "gates" / "g3_TP4_g44_64" / "launcher.json").exists())          # per-combination launcher record (gate module)
        self.assertEqual(table["contam_unverified"], [])

    def test_server_dead_after_the_last_step_fails_g3b_and_unknown_liveness_is_not_tested(self):    # F2
        table, out, order, ctx = self.run_job(FAKE_DEAD="m2g_TP2_d44", FAKE_NO_ALIVE="m2g_TP4_agnostic")
        self.assertFalse(table["cells"]["TP2:fused"]["included"])
        self.assertIn("G3b d44 FAIL", table["cells"]["TP2:fused"]["reason"])
        self.assertIn("G1 fused<->d44 FAIL", table["cells"]["TP2:d44"]["reason"])             # a pair with a dead server is an abnormal exit
        self.assertIn("G3b agnostic NOT_TESTED", table["cells"]["TP4:fused"]["reason"])       # no record != alive
        self.assertIn("G1 fused<->agnostic NOT_TESTED", table["cells"]["TP4:agnostic"]["reason"])
        self.assertTrue(table["cells"]["TP8:fused"]["included"])                              # negative control: the others are untouched

    def test_abnormal_exit_fails_g1_while_the_other_arm_of_the_config_passes(self):          # F2, G1 '비정상 종료 0'
        table, out, order, ctx = self.run_job(FAKE_DEAD="m2g_DP2_agnostic")
        g1 = json.loads((out / "gates" / "g1_DP2.json").read_text())
        self.assertEqual((g1["verdicts"]["fused|agnostic"], g1["verdicts"]["fused|d44"]), ("FAIL", "PASS"))
        self.assertTrue(g1["detail"]["fused|agnostic"]["abnormal_exit"])
        self.assertIn("G1 fused<->agnostic FAIL", table["cells"]["DP2:agnostic"]["reason"])
        self.assertNotIn("G1 fused<->d44", table["cells"]["DP2:d44"]["reason"])

    def test_g1_window_exposure_is_computed_and_delivered(self):                              # F3
        table, out, order, ctx = self.run_job()
        e = json.loads((out / "gates" / "g1_exposure_TP2.json").read_text())                 # the driver's own (cmd-window) cross-check
        # hand-derived from tests/test_mg_realized_mix.py rows: window [-0.5, 3.5] keeps 4 snapshots (agnostic: split 30 of 40 steps;
        # d44: 10 of 20); the WHOLE telemetry would give 0.8 / 0.6667 -- the window matters
        self.assertAlmostEqual(e["agnostic"]["exposure_steps"], 0.75, 6)
        self.assertAlmostEqual(e["d44"]["exposure_steps"], 0.5, 6)
        g1 = json.loads((out / "gates" / "g1_TP2.json").read_text())                          # what the scorer received (G1 window of serve-g1)
        self.assertAlmostEqual(g1["detail"]["fused|agnostic"]["exposure_split_steps"], 0.75, 6)
        self.assertAlmostEqual(g1["detail"]["fused|d44"]["exposure_split_steps"], 0.5, 6)
        self.assertAlmostEqual(table["g1_exposure_split_steps"]["TP2:agnostic"], 0.75, 6)
        self.assertEqual(json.loads((out / "gates" / "g1_exposure_DP4.json").read_text())["d44"]["status"], "OK")

    def test_g3_launcher_failure_is_not_tested_harness_error_not_fail(self):                  # F1
        table, out, order, ctx = self.run_job(FAKE_G3_BROKEN="TP4")
        g3 = json.loads((out / "gates" / "g3_TP4.json").read_text())
        self.assertEqual(g3["verdict"], "NOT_TESTED")                                         # lesson 21: not FAIL
        self.assertIn("HARNESS_ERROR", json.dumps(g3))
        self.assertIn("G3 comm stress NOT_TESTED", table["cells"]["TP4:fused"]["reason"])
        self.assertTrue(table["cells"]["TP2:fused"]["included"])
        self.assertTrue((out / "gates" / "g3_logs" / "run_g3_TP4.txt").exists())

    def test_r1_g3_not_pass_tp_serving_boots_are_not_attempted_and_pass_tp_runs(self):         # R1
        table, out, order, ctx = self.run_job(FAKE_G3_BROKEN="TP4")
        booted = [ln.split()[1] for ln in order if ln.split()[0] == "BOOT"]
        self.assertFalse([b for b in booted if b.startswith("m2g_TP4_") or b.startswith("m2g_g4_TP4_")], booted)
        skipped = {e["boot_id"] for e in ctx.not_attempted if e.get("reason") == "NOT_ATTEMPTED_G3_EXCLUDED"}
        want = {b["boot_id"] for b in P.m2g_boots() if b["boot_id"].startswith(("m2g_TP4_", "m2g_g4_TP4_"))}
        self.assertTrue(want and skipped == want, (want, skipped))
        self.assertFalse((out / "boots" / "m2g_TP4_fused").exists())
        for n in (2, 8):                                                                     # negative control: G3 PASS TPs run
            self.assertIn(f"m2g_TP{n}_fused", booted)
        self.assertIn("m2g_DP4_fused", booted)                                              # DP has no G3 dependence
        self.assertFalse(table["cells"]["TP4:fused"]["included"])

    def test_r1_mutation_without_the_skip_boots_the_excluded_tp(self):                       # R1 mutation: KILLED
        from unittest import mock
        with mock.patch.object(D, "g3_excluded_boot", lambda bid, ex: False):
            table, out, order, ctx = self.run_job(FAKE_G3_BROKEN="TP4")
        booted = [ln.split()[1] for ln in order if ln.split()[0] == "BOOT"]
        self.assertTrue(any(b.startswith("m2g_TP4_") for b in booted))                       # the R1 test above would fail -> KILLED

    def test_r1_rescore_keeps_preflight_and_not_attempted(self):                              # recommendation: merge
        table, out, order, ctx = self.run_job()
        g = json.loads((out / "gates" / "g3_TP2.json").read_text())
        self.assertIn("preflight", g)
        self.assertIn("not_attempted", g)
        import subprocess as sp
        sp.run([sys.executable, str(ENGINE / "scripts" / "multigpu" / "m2g_correctness_gate.py"), "score-g3", "--out",
                str(out / "gates"), "--tp", "2"], check=True, capture_output=True)
        g2 = json.loads((out / "gates" / "g3_TP2.json").read_text())
        self.assertEqual(g2["preflight"], g["preflight"])
        self.assertEqual(g2["not_attempted"], g["not_attempted"])

    def test_preboot_nvml_check_is_recorded_once_before_the_first_boot(self):
        from unittest import mock
        import m2g_correctness_gate as G
        calls = []
        def fake(ids, nv=None):
            calls.append(1)
            return {int(g): {"memory_used_mib": 5.0, "n_procs": 0} for g in ids}
        with mock.patch.object(G, "nvml_snapshot", fake):
            table, out, order, ctx = self.run_job()
        rec = json.loads((out / "gates" / "nvml_preboot.json").read_text())
        self.assertTrue(rec["checked"] and rec["not_empty"] == [] and rec["attempts"] == 1)

    def test_no_interpreter_for_the_launcher_leaves_every_tp_not_tested(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = D.Ctx(Path(td), False, time.time() + 8 * 3600, time.time(), "m2g")
            ctx.g3_python = "/nonexistent/python"
            s = D.run_g3(ctx)
            self.assertEqual(set(s), {"TP2", "TP4", "TP8"})
            for n in (2, 4, 8):
                self.assertEqual(json.loads((Path(td) / "gates" / f"g3_TP{n}.json").read_text())["verdict"], "NOT_TESTED")

    def test_nvml_unavailable_cell_is_listed_and_not_included(self):                          # F5
        table, out, order, ctx = self.run_job()
        # re-assemble after rewriting one pre-boot record as 'NVML unavailable'
        p = out / "boots" / "m2g_TP2_agnostic" / "contam_pre.json"
        p.write_text(json.dumps({"snapshot": {"available": False, "error": "x"}, "verdict": {"contam_unverified": True, "verified": False}}))
        t2 = D.assemble_pass_table(out)
        self.assertEqual(t2["contam_unverified"], ["TP2:agnostic"])
        self.assertFalse(t2["cells"]["TP2:agnostic"]["included"])
        self.assertIn("CONTAM_UNVERIFIED", t2["cells"]["TP2:agnostic"]["reason"])
        self.assertTrue(table["cells"]["TP2:agnostic"]["included"])                              # before the rewrite it was included


class M3ExtensionDriver(unittest.TestCase):                                                  # F10 driver half
    def test_incomplete_extension_blocks_boot2_and_labels_the_arm(self):
        with tempfile.TemporaryDirectory() as td:
            drv = DriverWithFakeRunner()
            skip = "m3_TP8_agnostic_boot1ext"
            fake_py = FAKE_RUNNER_PY.replace('spec = json.load(open(sys.argv[1]))',
                                             'spec = json.load(open(sys.argv[1]))\nif spec["boot_id"] == os.environ.get("FAKE_SKIP"): sys.exit(1)')
            fake = Path(td) / "fake_runner.sh"
            (Path(td) / "fake.py").write_text(fake_py)
            fake.write_text(f'#!/bin/bash\nexec python3 {td}/fake.py "$@"\n')
            import hashlib
            raw = json.dumps({"cells": {}, "g5a_unit_test": "PASS"})
            tbl = Path(td) / "t.json"
            tbl.write_text(raw)
            env = dict(os.environ, PDMUX_MG_RUNNER=str(fake), MG_RESULTS_DIR=str(ENGINE / "results/multigpu"), FAKE_RATIO="1.7",
                       FAKE_AGN_BOOST="1.0", FAKE_SKIP=skip, PYTHONDONTWRITEBYTECODE="1")
            out = Path(td) / "out"
            p = subprocess.run([sys.executable, str(ENGINE / "results/multigpu/mg_drive.py"), "m3", "--out", str(out),
                                "--pass-table", str(tbl), "--pass-sha256", hashlib.sha256(raw.encode()).hexdigest()],
                               env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            doc = json.loads((out / "LABELS.json").read_text())
            dec = doc["driver"]["decisions"]["TP8"]
            self.assertFalse(dec["ext_complete"])
            self.assertIn("agnostic", dec["ext_incomplete_reason"])
            self.assertEqual(list((out / "specs").glob("m3_TP8_*_boot2.json")), [])             # boot2 never entered
            self.assertEqual(list((out / "cells").glob("m3_TP8_*_boot2__*")), [])               # no TP8 LO cell was run
            self.assertTrue(list((out / "cells").glob("m3_TP2_*_boot2__*")))                    # negative control: other configs still ran LO
            reasons = [x["reason"] for x in doc["driver"]["not_attempted"] if x["boot_id"].startswith("m3_TP8_")]
            self.assertEqual(reasons, ["NOT_ATTEMPTED_EXT_INCOMPLETE"] * 4)
            lab = doc["keys"]["TP8|agnostic|A|r1"]
            self.assertEqual(lab["hi_label"], "HI_NOT_JUDGED")                                  # first-pass cells exist, extension does not
            self.assertIn("HI_NOT_JUDGED(EXT_NOT_RUN)", lab["reasons"])
            self.assertFalse(doc["estimands"]["G_sym"]["TP8"]["A"]["defined"])
            self.assertTrue(doc["estimands"]["G_sym"]["TP2"]["A"]["defined"])


class JobScriptsDryRun(unittest.TestCase):
    def run_job(self, name):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_MG_DRYRUN="1", PDMUX_MG_TEST_OUT_ROOT=td,
                       PDMUX_JOB_NAME="unit", PDMUX_DATA=str(Path(td) / "data"))
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu" / name)], env=env, capture_output=True, text=True)
            out = Path(td) / f"{name.split('_')[0]}_unit"
            files = {f.name for f in out.iterdir()} if out.exists() else set()
            boots = sorted(x.name for x in (out / "boots").glob("*")) if (out / "boots").exists() else []
            notes = (out / "DRYRUN_NOTES.txt").read_text() if (out / "DRYRUN_NOTES.txt").exists() else ""
            return p, files, boots, notes

    def test_m3_dry_run(self):
        p, files, boots, notes = self.run_job("m3_job.sh")
        self.assertEqual(p.returncode, 0, p.stderr[-1500:] + p.stdout[-800:])
        self.assertEqual(len(boots), 70)
        for a in ("ABORT_SUBSTRATE", "ABORT_MODEL_REVISION", "ABORT_ENGINE_DIGEST"):
            self.assertIn(a, notes)                              # would abort a real run on this machine
        self.assertIn("DRYRUN", files)
        self.assertIn("BUDGET_DRYRUN.txt", files)

    def test_m2g_dry_run(self):
        p, files, boots, notes = self.run_job("m2g_job.sh")
        self.assertEqual(p.returncode, 0, p.stderr[-1500:] + p.stdout[-800:])
        self.assertEqual(len(boots), len(P.m2g_boots()))
        self.assertIn("M2G_PASS_TABLE.json", files)

    def test_real_run_aborts_without_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_JOB_NAME="unitreal", PDMUX_DATA=str(Path(td) / "data"))
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu/m3_job.sh")], env=env, capture_output=True, text=True,
                               cwd=td)
            self.assertEqual(p.returncode, 11, p.stdout[-600:])                       # ABORT_SUBSTRATE (no 8x A100 here)
            out = ENGINE / "results/multigpu/m3_unitreal"
            try:
                self.assertIn("ABORT_SUBSTRATE", (out / "ABORT").read_text())
            finally:
                import shutil
                shutil.rmtree(out, ignore_errors=True)

    def test_jit_all_reduce_aborts_comm_env(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_JOB_NAME="unitjit", PDMUX_DATA=str(Path(td) / "data"),
                       SGLANG_USE_JIT_ALL_REDUCE="1")
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu/m2g_job.sh")], env=env, capture_output=True, text=True, cwd=td)
            out = ENGINE / "results/multigpu/m2g_unitjit"
            try:
                self.assertEqual(p.returncode, 14, p.stdout[-600:])
            finally:
                import shutil
                shutil.rmtree(out, ignore_errors=True)


FAKE_LAUNCH_SERVER = r"""
import asyncio, json, os, sys, time
from aiohttp import web
mode = os.environ.get("FAKE_MODE", "ok")
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
cap = int(args[args.index("--max-running-requests") + 1])
print("server_args=ServerArgs(disable_piecewise_cuda_graph=False, piecewise_cuda_graph_tokens=[4, 8], chunked_prefill_size=8192)", flush=True)
if mode == "die_text":
    print("RuntimeError: PD-mux with tp_size=2 is only defined for the legacy ...", flush=True); sys.exit(1)
if mode == "die":
    print("CUDA error: boom", flush=True); sys.exit(1)
if mode == "sleep":
    time.sleep(600)
print(f"max_total_num_tokens=900000, max_running_requests={cap}", flush=True)
print(f"max_mamba_cache_size: {cap}", flush=True)
print("Capture cuda graph begin", flush=True)
itl = float(os.environ.get("FAKE_ITL", "0.002"))
async def health(_r): return web.Response(text="ok")
async def info(_r): return web.json_response({"max_total_num_tokens": 900000, "max_running_requests": cap, "internal_states": []})
async def generate(req):
    body = await req.json(); n = int(body["sampling_params"]["max_new_tokens"])
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await resp.prepare(req)
    await asyncio.sleep(0.01)
    for k in range(1, n + 1):
        await resp.write(b"data: " + json.dumps({"text": "t" * k, "meta_info": {"completion_tokens": k}}).encode() + b"\n\n")
        await asyncio.sleep(itl)
    await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
app = web.Application(); app.router.add_get("/health", health); app.router.add_get("/model_info", info)
app.router.add_get("/server_info", info); app.router.add_post("/generate", generate)
web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)
"""


class RunBootFakeServer(unittest.TestCase):
    """The REAL mg_boot.run_boot / mg_loadgen / sampler against a fake `sglang.launch_server` (no GPU, no model)."""

    def setUp(self):
        from unittest import mock
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        root = Path(self.td.name)
        (root / "fake" / "sglang").mkdir(parents=True)
        (root / "fake" / "sglang" / "__init__.py").write_text("")
        (root / "fake" / "sglang" / "launch_server.py").write_text(FAKE_LAUNCH_SERVER)
        self.root = root
        clean = {"available": True, "gpus": {"0": {"used_mib": 10.0, "n_compute_procs": 0},
                                              "1": {"used_mib": 10.0, "n_compute_procs": 0}}}
        p = mock.patch.object(B, "nvml_snapshot", return_value=clean)
        p.start()
        self.addCleanup(p.stop)
        env = {"PYTHONPATH": str(root / "fake") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        e = mock.patch.dict(os.environ, env)
        e.start()
        self.addCleanup(e.stop)
        for k in [k for k in os.environ if k.startswith(("PDMUX_", "SGLANG_USE", "NCCL_"))]:
            os.environ.pop(k)

    def spec(self, arm="d44", cfg="N1", steps=None, **kw):
        s = dict(spec_for(cfg, arm), out_root=str(self.root / "out"), steps=steps or [], health_wait_s=60)
        s.update(kw)
        return s

    def cell(self, shape="A", rung="HI", seed=4386, offered=20.0, n=40, **kw):
        return dict(type="cell", shape=shape, rung=rung, seed=seed, offered=offered, n_req=n, ext=0, round=1, block="main", **kw)

    def test_full_boot_with_two_cells_and_label_roundtrip(self):
        spec = self.spec(steps=[{"type": "warmup", "shape": "A"}, self.cell(), self.cell(seed=4162, rung="LO", offered=10.0, n=30)],
                         expected_proc_counts={"0": 0})
        rc = B.run_boot(spec)
        out = self.root / "out"
        bd = out / "boots" / spec["boot_id"]
        self.assertEqual(rc, 0, (bd / "server.log").read_text() if (bd / "server.log").exists() else "")
        boot = json.loads((bd / "boot.json").read_text())
        self.assertEqual(boot["status"], "OK")
        self.assertEqual(boot["registered"]["health_wait_s"], 60)
        for f in ("server_args.txt", "server.log", "cell_meta.json", "piecewise.json", "cuda_graph_capture.log",
                  "comm_env.json", "env.json", "model_info.json", "server_info.json"):
            self.assertTrue((bd / f).exists(), f)
        cm = json.loads((bd / "cell_meta.json").read_text())
        self.assertEqual(cm["piecewise"]["disable_piecewise_cuda_graph"], "False")          # H8 inside cell_meta.json
        self.assertEqual(cm["max_total_num_tokens"], [900000])
        cells = sorted((out / "cells").glob("*"))
        self.assertEqual(len(cells), 2)
        for cd in cells:
            for f in ("requests.jsonl", "cell.json", "cpu_stat_before.txt", "cpu_stat_after.txt", "proc_samples.csv",
                      "loadgen_clock.json", "fd_limit.json", "trace.sha256", "summary.json"):
                self.assertTrue((cd / f).exists(), (cd.name, f))
            rec = L.load_cell(cd)
            self.assertEqual(rec["status"], "OK", rec)
            self.assertIsNotNone(rec["g"])
            self.assertIsNotNone(rec["achieved"])
            self.assertFalse(rec["frontend_unmeasured"])           # sampler rows exist inside the send window
        self.assertTrue(any(r["meta"]["trace_sha256"] for r in map(L.load_cell, cells)))
        self.assertGreaterEqual(json.loads((cells[0] / "fd_limit.json").read_text())["soft_after"], 8192)

    def test_same_cell_same_trace_hash_across_arms(self):
        hs = set()
        for arm in ("fused", "d44"):
            r = self.root / f"o_{arm}"
            spec = self.spec(arm=arm, steps=[self.cell(n=25)], out_root=str(r), boot_id=f"b_{arm}")
            self.assertEqual(B.run_boot(spec), 0)
            hs.add((next((r / "cells").glob("*")) / "trace.sha256").read_text().strip())
        self.assertEqual(len(hs), 1)                                # same seed/shape/offered/n -> same trace (pair rule)

    def test_expected_refusal(self):
        os.environ["FAKE_MODE"] = "die_text"
        self.addCleanup(os.environ.pop, "FAKE_MODE", None)
        spec = self.spec(cfg="TP2", arm="agnostic", expect_refusal="PD-mux with tp_size=")
        self.assertEqual(B.run_boot(spec), 0)
        self.assertEqual(json.loads((self.root / "out/boots" / spec["boot_id"] / "boot.json").read_text())["status"], "REFUSED_AS_EXPECTED")
        spec2 = self.spec(cfg="TP2", arm="agnostic", expect_refusal="a text that never appears", boot_id="b2")      # negative control
        self.assertEqual(B.run_boot(spec2), 1)
        self.assertEqual(json.loads((self.root / "out/boots/b2/boot.json").read_text())["status"], "REFUSAL_FAILED")
        os.environ["FAKE_MODE"] = "ok"
        spec3 = self.spec(cfg="TP2", arm="agnostic", expect_refusal="PD-mux with tp_size=", boot_id="b3")           # healthy server: refusal failed
        self.assertEqual(B.run_boot(spec3), 1)

    def test_boot_failure_and_timeout(self):
        os.environ["FAKE_MODE"] = "die"
        self.addCleanup(os.environ.pop, "FAKE_MODE", None)
        self.assertEqual(B.run_boot(self.spec(boot_id="bf")), 1)
        self.assertEqual(json.loads((self.root / "out/boots/bf/boot.json").read_text())["status"], "BOOT_FAILED")
        os.environ["FAKE_MODE"] = "sleep"
        self.assertEqual(B.run_boot(self.spec(boot_id="bt", health_wait_s=3)), 1)
        self.assertEqual(json.loads((self.root / "out/boots/bt/boot.json").read_text())["status"], "BOOT_TIMEOUT")

    def test_deadline_truncates_running_cell_and_skips_the_rest(self):
        slow = self.cell(offered=0.5, n=60)                         # ~120 s send window
        spec = self.spec(steps=[{"type": "warmup", "shape": "A"}, slow, self.cell(seed=4162)], deadline_epoch=time.time() + 12)
        rc = B.run_boot(spec)
        boot = json.loads((self.root / "out/boots" / spec["boot_id"] / "boot.json").read_text())
        statuses = [s["status"] for s in boot["steps"]]
        self.assertIn("TRUNCATED", statuses, statuses)
        self.assertEqual(statuses[-1], "NOT_STARTED_DEADLINE")
        cd = next((self.root / "out/cells").glob("*_s4386"))
        self.assertTrue(json.loads((cd / "cell.json").read_text())["truncated"])
        self.assertEqual(L.load_cell(cd)["status"], "EMPTY")        # no requests.jsonl -> never a label input

    def test_contaminated_gpu_blocks_the_boot(self):
        from unittest import mock
        dirty = {"available": True, "gpus": {"0": {"used_mib": 5000.0, "n_compute_procs": 1}}}
        with mock.patch.object(B, "nvml_snapshot", return_value=dirty), mock.patch.object(B.time, "sleep", lambda *_: None):
            spec = self.spec(boot_id="bc")
            self.assertEqual(B.run_boot(spec), 5)
        bd = self.root / "out/boots/bc"
        self.assertEqual(json.loads((bd / "boot.json").read_text())["status"], "GPU_CONTAMINATED")
        self.assertTrue(json.loads((bd / "contam.json").read_text())["gpu_contaminated"])
        self.assertFalse((bd / "server.log").exists())              # nothing was launched on a contaminated GPU


if __name__ == "__main__":
    unittest.main()

    def test_alive_after_and_cmd_window_are_written_by_the_real_boot(self):                  # F2, F3
        import m2g_correctness_gate as G
        steps = [{"type": "cmd", "name": "g1_d44", "argv": [sys.executable, "-c", "print('g1')"]}]
        spec = self.spec(steps=steps)
        self.assertEqual(B.run_boot(spec), 0)
        bd = self.root / "out/boots" / spec["boot_id"]
        self.assertTrue(G.alive_after(bd))                                                    # running + /health 200 AFTER the last step
        rec = json.loads((bd / "alive_after.json").read_text())
        self.assertEqual((rec["after_step"]["type"], rec["after_step"]["status"], rec["steps_total"]), ("cmd", "OK", 1))
        w = json.loads((bd / "cmd_g1_d44.window.json").read_text())
        self.assertLess(w["t0_mono"], w["t1_mono"])
        self.assertLess(abs(w["t1_mono"] - time.monotonic()), 120)                           # CLOCK_MONOTONIC, comparable with the telemetry clock

    def test_nvml_unavailable_boot_runs_but_is_marked_contam_unverified(self):               # F5
        from unittest import mock
        with mock.patch.object(B, "nvml_snapshot", return_value={"available": False, "error": "no nvml"}):
            spec = self.spec(boot_id="bu")
            self.assertEqual(B.run_boot(spec), 0)
        bd = self.root / "out/boots/bu"
        self.assertEqual(json.loads((bd / "boot.json").read_text())["contam_status"], "CONTAM_UNVERIFIED")
        self.assertTrue(json.loads((bd / "contam_pre.json").read_text())["verdict"]["contam_unverified"])
        self.assertFalse((bd / "contam.json").exists())                                      # not 'contaminated', not 'clean': unverified
