"""Multi-GPU harness H1–H10 (PREREG_MULTIGPU rev10: one Job = one N): launcher, plan, budget, labels, drivers.  All CPU.

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

    def test_job_plans_order_counts_and_blocks(self):
        # J8: gpu_dev (8 concurrent) -> anchor (2) -> the two configs in the REGISTERED order [DP, TP]; no numa / nsys / anchor2
        st = P.m3_static_boots("J8")
        self.assertEqual([n for n, _ in st[:2]], ["gpu_dev", "anchor"])
        self.assertEqual(len(st[0][1]), 8)
        self.assertEqual(len(st[1][1]), 2)
        self.assertEqual([n.split(".")[0] for n, _ in st[2::2]], ["DP8", "TP8"])
        self.assertEqual(sum(len(b) for _, b in st), 8 + 2 + 16)
        self.assertEqual(sum(1 for _, bs in st for b in bs for s in b["steps"] if s["type"] == "cell"), 8 + 8 + 2 * 4 * 2 * 6 + 0 * 1 + 0)
        for banned in ("numa", "nsys", "anchor2"):
            self.assertFalse(any(n.startswith(banned) for n, _ in st))
        # J2 / J4: no gpu_dev block; J1: neither gpu_dev nor anchor (it is the reference), N=1 only
        self.assertEqual([n for n, _ in P.m3_static_boots("J2")][:1], ["anchor"])
        self.assertEqual([n for n, _ in P.m3_static_boots("J1")], ["N1.boot1", "N1.boot2"])
        self.assertEqual(sum(len(b) for _, b in P.m3_static_boots("J1")), 8)

    def test_anchor_boots_h_spec(self):
        for job in ("J2", "J4", "J8"):
            bs = P.boots_anchor(job)
            self.assertEqual(sorted(b["arm"] for b in bs), ["agnostic", "fused_sym"])          # exactly two boots, §16-3
            for b in bs:
                cells = [s for s in b["steps"] if s["type"] == "cell"]
                self.assertEqual((b["gpu_ids"], b["block"], b["N"]), ("0", "anchor", 1))
                self.assertEqual({(c["shape"], c["rung"], c["seed"], c["ext"]) for c in cells}, {("A", "HI", sd, 0) for sd in C.HI_SEEDS})
                # THE SAME rung as J1 boot1 (1.6 x prior(1), shape A, ext 0)
                j1 = [s for s in P.boots_boot1("N1")[0]["steps"] if s["type"] == "cell" and s["shape"] == "A"][0]
                self.assertEqual((cells[0]["offered"], cells[0]["n_req"]), (j1["offered"], j1["n_req"]))
        self.assertEqual(P.boots_anchor("J1"), [])

    def test_registered_config_order_and_its_mutant_H2(self):
        self.assertEqual(C.config_order_types(2), ["DP", "TP"])
        for n in (2, 4, 8):
            self.assertEqual(C.config_order_types(n), C.CONFIG_ORDER_EXPECTED[n])
        self.assertEqual({j: P.config_order(j) for j in C.JOBS}, {"J1": ["N1"], "J2": ["DP2", "TP2"], "J4": ["DP4", "TP4"], "J8": ["DP8", "TP8"]})
        self.assertTrue(P.assert_config_order_registered())
        with self.assertRaises(AssertionError):                                          # mutant: initial order [DP, TP] flips the shuffle -> KILLED
            P.assert_config_order_registered(["DP", "TP"])
        self.assertEqual(C.config_order_types(2, ["DP", "TP"]), ["TP", "DP"])

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

class Budget(unittest.TestCase):
    def test_job_budgets_within_job_caps_and_audited_band(self):
        # lambda = prior, audited model: J1 ~4.0 h, J2 ~8.8 h, J4 ~8.7 h, J8 ~7.9 h (rev8 audit MGP-49: 4.0-4.3 / 8.8-9.4 / 8.7-9.4 / 7.9-8.6 h)
        got = {j: P.m3_budget(j) for j in C.JOBS}
        for j, (lo, hi) in {"J1": (3.9, 4.4), "J2": (8.6, 9.5), "J4": (8.5, 9.5), "J8": (7.7, 8.7)}.items():
            self.assertTrue(lo <= got[j]["job_h"] <= hi, (j, got[j]["job_h"]))
            self.assertFalse(got[j]["over_cap"], j)
        self.assertEqual({j: got[j]["cap_h"] for j in C.JOBS}, {"J1": 10.0, "J2": 14.0, "J4": 14.0, "J8": 14.0})
        # reservation view: reserved = spec GPUs x Job wall; J1 uses 1 of its 2 reserved GPUs
        self.assertAlmostEqual(got["J1"]["reserved_gpu_h"], 2 * got["J1"]["job_h"], 6)
        self.assertLess(got["J1"]["utilization"], 0.51)
        for j in ("J2", "J4", "J8"):
            self.assertGreater(got[j]["utilization"], 0.85)
        # TP below prior and S3 extension push a Job over its cap (MGP-49)
        self.assertGreater(P.m3_budget("J2", lambda cfg: 0.7 if cfg.startswith("TP") else 1.0)["total_h"], got["J2"]["total_h"])
        self.assertLess(P.m3_budget("J2", lambda c: 1.0)["total_h"], P.m3_budget("J2", lambda c: 0.5)["total_h"])    # monotone

    def test_s3_extension_cuts_configs_per_job(self):
        rows = P.m3_forecast_table("J2")
        s3 = [r for r in rows if r["scenario"].startswith("S3")]
        self.assertTrue(all(r["total_h_uncapped"] > 14.0 for r in s3))
        self.assertTrue(all(len(r["configs_defined"]) < len(r["configs_total"]) for r in s3))      # the Job cap cuts the SECOND (TP) config first

    def test_extension_adds_boots(self):
        base = P.m3_budget("J4")
        ext = P.m3_budget("J4", assume_ext=True)
        self.assertGreater(ext["boots"], base["boots"])
        self.assertGreater(ext["total_h"], base["total_h"])

    def test_m2g_within_cap_per_job(self):
        for j in C.JOBS:
            r = P.m2g_budget(j)
            self.assertFalse(r["over_cap"], j)
            self.assertEqual(r["cap_h"], 4.0)
            self.assertGreater(r["boots"], 3)
        self.assertEqual(P.m2g_budget("J2")["boots"], len(P.m2g_boots("J2")))

    def test_m2g_covers_every_registered_boot(self):
        for job, cfgs in (("J1", ("N1",)), ("J2", ("N1", "TP2", "DP2")), ("J4", ("N1", "TP4", "DP4")), ("J8", ("N1", "TP8", "DP8"))):
            ids = [b["boot_id"] for b in P.m2g_boots(job)]
            for cfg in cfgs:
                for arm in C.ARMS:
                    self.assertIn(f"m2g_{cfg}_{arm}", ids)           # N1 carries fused / fused_sym / agnostic / d44 in EVERY Job (G1-ctl, G2 floor)
            kinds = lambda k: sum(1 for b in P.m2g_boots(job) if b["kind"] == k)
            self.assertEqual((kinds("g0ii"), kinds("g0iii")), (5, 1) if job == "J2" else (0, 0))      # G0(ii)/(iii) only in J2
            self.assertEqual(kinds("g5a"), 1 if job == "J1" else 0)
            self.assertEqual(kinds("g4"), {"J1": 0, "J2": 1, "J4": 0, "J8": 1}[job])
            g3b = [b for b in P.m2g_boots(job) if any(s.get("rung") == "G3B" for s in b["steps"])]
            self.assertEqual(len(g3b), 0 if job == "J1" else 2)                                  # TP_N x {agnostic, d44}
            for b in g3b:
                s = next(s for s in b["steps"] if s.get("rung") == "G3B")
                self.assertAlmostEqual(s["offered"] / (1.5 * b["N"] * 2.99), 1.0, 5)

    def test_job_table_prints_every_job_with_reservation_columns(self):
        t = P.job_table()
        for j in C.JOBS:
            self.assertIn(j, t)
        self.assertIn("util", t)


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


def job_docs(cells, anchor_G=lambda n: 1.0, anchor_over=None, mcs=None, driver=None):
    """Split a flat cell list (config N1 / TPn / DPn) into the four Jobs, add the N=1 anchor cells (block "anchor", config N1, shape A HI) to
    J2/J4/J8 -- agnostic anchor = anchor_G(n) x fused_sym anchor, anchor_G(n) None -> no anchor -- and label each Job."""
    by = {j: [] for j in C.JOBS}
    for c in cells:
        cfg = c["meta"]["config"]
        by["J1" if cfg == "N1" else f"J{int(cfg[2:])}"].append(c)
    for n in (2, 4, 8):
        g = anchor_G(n)
        if g is None:
            continue
        for arm in C.ANCHOR_ARMS:
            lam = P.prior(1, "A") * (g if arm == "agnostic" else 1.0)
            cs = [cell_rec("N1", arm, "A", "HI", sd, lam * nz, 0.4) for sd, nz in zip(C.HI_SEEDS, (1.0, 1.01, 0.99, 1.005))]
            for c in cs:
                c["meta"]["block"] = "anchor"
                c.update((anchor_over or {}).get((n, arm), {}))
            by[f"J{n}"] += cs
    docs = {}
    for j in C.JOBS:
        docs[j] = L.aggregate(by[j], None, {}, mcs, job=j)
        if driver and j in driver:
            docs[j]["driver"] = driver[j]
    return docs


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
        t = L.trend(up[1:] + [(8, [5, 5, 5, 5], [1] * 4)])                         # no N=1: P9-a, the rev8 'direction label only' rule is retired
        self.assertEqual((t["label"], t["magnitude"], t["direction_citable"]), (L.G1_UNDEFINED, L.G1_UNDEFINED, False))
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
        cells = self.build_cells(agn_boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1))
        docs = job_docs(cells)
        e = docs["J4"]["estimands"]["G_sym"]["TP4"]["A"]
        self.assertTrue(e["defined"])
        self.assertAlmostEqual(e["point"], 1.2, 6)
        self.assertEqual(set(docs["J4"]["estimands"]["G_sym"]), {"TP4", "DP4"})              # a Job labels only its own configs
        self.assertNotIn("N1@anchor", docs["J4"]["estimands"]["G_sym"])                     # the anchor is never an estimand config / trend point
        doc = L.aggregate_jobs(docs)
        t = doc["trends"]["G_sym|TP|A"]
        self.assertEqual((t["label"], t["Ns"]), ("G_INCREASES_WITH_N", [1, 2, 4, 8]))
        self.assertEqual({J: s["status"] for J, s in doc["job_effect"].items()}, {"J2": "CLEAR", "J4": "CLEAR", "J8": "CLEAR"})
        self.assertEqual(t["undefined_count"], 0)
        self.assertTrue(doc["trends"]["G_sym|TP|A"]["primary"] and not doc["trends"]["G_def|TP|A"]["primary"])
        self.assertTrue(doc["P_DP"]["broken"] and doc["P_DP"]["MGP13_TP_labels_not_citable"])      # same boost on DP

    def test_aggregate_flat_dp_keeps_p_dp(self):
        doc = L.aggregate_jobs(job_docs(self.build_cells()))
        self.assertEqual(doc["trends"]["G_sym|DP|B"]["label"], "NO_TREND_DETECTED")
        self.assertFalse(doc["P_DP"]["broken"])
        self.assertEqual(doc["P_DP"]["status"], "HELD")

    def test_undefined_cells_listed_with_reasons_MGP34(self):
        cells = self.build_cells(drop={("TP2", "fused_sym")})
        docs = job_docs(cells)
        self.assertFalse(docs["J2"]["estimands"]["G_sym"]["TP2"]["A"]["defined"])
        doc = L.aggregate_jobs(docs)
        t = doc["trends"]["G_sym|TP|A"]
        self.assertEqual(t["undefined_count"], 1)
        self.assertIn("fused_sym:NOT_RUN", t["undefined"][0]["reasons"])
        self.assertEqual(t["undefined"][0]["job"], "J2")
        self.assertTrue(docs["J2"]["estimands"]["G_def"]["TP2"]["A"]["defined"])          # G_def NOT substituted for G_sym (MGP-38)
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

    def test_job_effect_clear_fired_untested_from_cells(self):
        base = lambda **kw: L.aggregate_jobs(job_docs(self.build_cells(), **kw))
        self.assertEqual(base()["job_effect"]["J4"]["status"], "CLEAR")
        fired = base(anchor_G=lambda n: 1.05 if n == 4 else 1.0)["job_effect"]
        self.assertEqual((fired["J4"]["status"], fired["J2"]["status"]), ("FIRED", "CLEAR"))
        self.assertAlmostEqual(fired["J4"]["G_rel"], 0.05, 6)
        # the fused_sym lambda* alone can fire it: both anchor arms +4 % -> G unchanged, lambda*(fused_sym) +4 % > 3 %
        docs = job_docs(self.build_cells())
        for arm in C.ANCHOR_ARMS:
            e = docs["J8"]["anchor"]["arms"][arm]
            e["lambda_star"] *= 1.04
        self.assertEqual(L.aggregate_jobs(docs)["job_effect"]["J8"]["status"], "FIRED")
        # no anchor / anchor cell not saturated / HARNESS_LIMITED / CLIENT_ERROR / reference cell not clean -> UNTESTED (H-3, §16-1)
        self.assertEqual(base(anchor_G=lambda n: None if n == 2 else 1.0)["job_effect"]["J2"]["status"], "UNTESTED")
        for field, val in (("g", 0.05), ("harness_limited", True), ("status", "CLIENT_ERROR")):
            d = base(anchor_over={(4, "agnostic"): {field: val}})["job_effect"]
            self.assertEqual(d["J4"]["status"], "UNTESTED", field)
            self.assertEqual(d["J2"]["status"], "CLEAR", field)                       # scoped to the Job that owns the anchor
        for field, val in (("g", 0.05), ("harness_limited", True), ("status", "CLIENT_ERROR")):       # H-3: the J1 boot1 reference cell
            cells = self.build_cells()
            for c in cells:
                m = c["meta"]
                if (m["config"], m["arm"], m["shape"], m["rung"]) == ("N1", "fused_sym", "A", "HI"):
                    c[field] = val
            d = L.aggregate_jobs(job_docs(cells))["job_effect"]
            self.assertEqual({d[j]["status"] for j in ("J2", "J4", "J8")}, {"UNTESTED"}, field)

    def test_job_effect_reference_is_boot1_never_the_extension_rung(self):
        cells = self.build_cells()
        for c in cells:                                 # J1 fused_sym/agnostic shape A: ext-0 HI at 1.0 x prior is NOT saturated; ext-1 at 1.3 x
            m = c["meta"]
            if m["config"] == "N1" and m["shape"] == "A" and m["arm"] in C.ANCHOR_ARMS and m["rung"] == "HI":
                c["g"] = 0.05
        for arm in C.ANCHOR_ARMS:
            cells += [cell_rec("N1", arm, "A", "HI", s, P.prior(1, "A") * 1.3 * nz, 0.4, ext=1) for s, nz in zip(C.HI_SEEDS, (1.0, 1.01, 0.99, 1.005))]
        docs = job_docs(cells)
        ref = docs["J1"]["ref"]["fused_sym"]
        self.assertAlmostEqual(ref["boot1_lambda_star"], P.prior(1, "A") * 1.0025, 6)
        self.assertAlmostEqual(ref["final_lambda_star"], P.prior(1, "A") * 1.3 * 1.0025, 6)
        self.assertEqual(ref["boot1_hi_label"], "HI_NOT_SATURATED")          # H-3 reference-cell condition fires on the boot1 cell
        self.assertEqual(L.aggregate_jobs(docs)["job_effect"]["J2"]["status"], "UNTESTED")

    def test_p9a_g1_undefined_fail_closed(self):
        docs = job_docs(self.build_cells(drop={("N1", "fused_sym")}))
        doc = L.aggregate_jobs(docs)
        for k, t in doc["trends"].items():
            if k.startswith("G_sym|"):
                self.assertEqual(t["label"], L.G1_UNDEFINED, k)
                self.assertFalse(t["direction_citable"])
        self.assertTrue(all(s["status"] == "UNTESTED" for s in doc["job_effect"].values()))
        self.assertEqual(doc["P_DP"]["status"], "UNTESTED")
        self.assertFalse(doc["P_DP"]["broken"])

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
        wide = dict(ok, d44=dict(mamba=[96], max_total=[860000], eff_cap=96))
        r = L.memory_check(wide)
        self.assertFalse(r["memory_mismatch"])                                                    # (iii) RETIRED (rev9, post-hoc): 4 % spread is no reason
        self.assertEqual(r["MEMORY_RULE_III_COUNTERFACTUAL"], {"denominator_fused_sym": True, "denominator_min_arm": True})
        self.assertAlmostEqual(r["arm_rel_diff"]["fused_sym"], 40000 / 900000, 9)
        self.assertAlmostEqual(r["arm_rel_diff"]["min_arm"], 40000 / 860000, 9)                   # the two denominators differ
        near = dict(ok, d44=dict(mamba=[96], max_total=[895000], eff_cap=96))
        r = L.memory_check(near)
        self.assertFalse(r["memory_mismatch"] or r["rule_iii_would_have_flagged"])                # 0.6 % is below the counterfactual threshold
        self.assertTrue(L.memory_check(wide, _mut="iii")["memory_mismatch"])                      # mutant '(iii) still present' KILLED
        self.assertFalse(L.memory_check(dict(ok, d44=dict(mamba=[96], max_total=[10000], eff_cap=96)), _mut="ii")["memory_mismatch"])   # mutant '(ii) removed' KILLED

    def test_server_log_descriptive_never_changes_label(self):
        s = L.synthetic_scenarios()["S1_saturated_fifo"]
        d = L.server_log_descriptive(s["log"], 48)
        self.assertGreater(d["decode_lines"], 10)
        self.assertEqual(L.correct_class(s), "HI")


# =========================================================================== rev9 / rev10 Job layer: mutants KILLED, controls survive
class Rev10JobLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ok, cls.verdict, cls.per = L.run_job_mutation_checks()

    def test_all_mutants_killed_and_control_unkilled(self):
        self.assertTrue(self.ok, json.dumps(self.verdict, sort_keys=True))
        self.assertFalse(self.verdict["CONTROL"]["killed"], self.verdict["CONTROL"])           # lesson 254: the unmutated control matches ALL scenarios
        for m in ("i", "ii", "iii", "iv", "h3", "h4", "h2_ignore_truncation", "untested_first", "mem_iii_present", "mem_ii_removed"):
            self.assertTrue(self.verdict[m]["killed"], m)

    def test_each_mutant_dies_on_its_own_scenario(self):
        k = lambda m: set(self.verdict[m]["killed_on"])
        self.assertIn("S_i_g1_undefined_J8_plus2pct", k("i"))                    # (i) P9-a: G(1) undefined + J8 +2 % must not be cited
        self.assertIn("S_ii_extension_rung_reference", k("ii"))                   # (ii) reference = boot1, not the extension rung
        self.assertIn("S_iii_other_jobs_anchor", k("iii"))                        # (iii) another Job's anchor
        self.assertIn("S_iv_corrected_below_3pct", k("iv"))                       # (iv) corrected trend needs the 3 % rule too
        self.assertIn("S_h3_boot1_vs_extension_G1", k("h3"))                      # H-3 correction ratio from the extended-rung G(1)
        self.assertIn("S_h4_untested_anchor_point_excluded", k("h4"))             # H-4 raw value substituted for an anchor-UNTESTED point
        self.assertIn("S_h2_truncated_dp_blocks_tp", k("h2_ignore_truncation"))   # H-2 truncation reason ignored
        self.assertTrue({"P_A_job_sensitive_B_citable_increase", "P_A_g1_undefined_B_increase"} <= k("untested_first"))   # H-1
        self.assertIn("M_iii_spread_only", k("mem_iii_present"))                  # H-5
        self.assertIn("M_ii_kv_bound_violated", k("mem_ii_removed"))

    def test_scenario_outcomes_spelled_out(self):
        self.assertEqual(self.per["S_i_g1_undefined_J8_plus2pct"]["CONTROL"], L.G1_UNDEFINED)
        self.assertEqual(self.per["S_i_g1_undefined_J8_plus2pct"]["i"], "G_INCREASES_WITH_N")      # what the retired rule would have cited
        self.assertEqual(self.per["S_iii_other_jobs_anchor"]["iii"], "G_INCREASES_WITH_N")
        self.assertEqual(self.per["S_h3_boot1_vs_extension_G1"]["CONTROL"], L.TREND_JOB_SENSITIVE)
        self.assertEqual(self.per["S_h2_truncated_dp_blocks_tp"]["CONTROL"], "TP_UNCONTROLLED_TRUNCATION")
        self.assertEqual(self.per["S_h2_nontruncation_dp_gap_cites_tp"]["CONTROL"], "G_INCREASES_WITH_N")   # P9-c: disclose, then cite
        self.assertEqual(self.per["P_A_job_sensitive_B_citable_increase"]["CONTROL"], "BROKEN")
        self.assertEqual(self.per["P_A_g1_undefined_B_increase"]["untested_first"], "UNTESTED")

    def test_h2_tp_blocked_only_with_the_disclosure_and_p_dp_untested(self):
        docs, key, _, _ = L.JOB_MUT_SCENARIOS["S_h2_truncated_dp_blocks_tp"]
        doc = L.aggregate_jobs(docs)
        t = doc["trends"][key]
        self.assertEqual((t["label"], t["direction_citable"], t["label_before_truncation_rule"]), ("TP_UNCONTROLLED_TRUNCATION", False, "G_INCREASES_WITH_N"))
        self.assertIn("TP_UNCONTROLLED_TRUNCATION", t["disclosures"])
        self.assertEqual(doc["P_DP"]["status"], "UNTESTED")
        self.assertTrue(doc["truncation"]["dp_trend_undefined_by_truncation"])
        docs, key, _, _ = L.JOB_MUT_SCENARIOS["S_h2_nontruncation_dp_gap_cites_tp"]
        t = L.aggregate_jobs(docs)["trends"][key]
        self.assertEqual((t["label"], t["direction_citable"]), ("G_INCREASES_WITH_N", True))
        self.assertIn("P-DP 미검정", t["disclosures"])

    def test_h1_per_shape_p_dp_any_broken_fires_mgp13(self):
        tr = {"G_sym|DP|A": {"label": "TREND_JOB_SENSITIVE", "g1_defined": True}, "G_sym|DP|B": {"label": "G_INCREASES_WITH_N", "g1_defined": True}}
        r = L.p_dp_status(tr)
        self.assertEqual((r["status"], r["MGP13_fired"], r["per_shape"]["G_sym|DP|A"]), ("BROKEN", True, "UNTESTED"))
        tr["G_sym|DP|B"]["g1_defined"] = False                 # B directional but G(1,B) undefined -> not BROKEN
        self.assertEqual(L.p_dp_status(tr)["status"], "UNTESTED")
        self.assertFalse(L.p_dp_status(tr)["MGP13_fired"])

    def test_h4_correction_trend_points_are_excluded_not_substituted(self):
        docs, key, _, _ = L.JOB_MUT_SCENARIOS["S_h4_untested_anchor_point_excluded"]
        t = L.aggregate_jobs(docs)["trends"][key]
        self.assertEqual(t["label"], L.TREND_JOB_SENSITIVE)
        self.assertEqual(t["corrected"]["label"], "TREND_UNDEFINED")
        self.assertEqual(t["corrected"]["Ns"], [1, 4])
        self.assertIn("CORRECTED_TREND_UNDEFINED", t["job_sensitive_reason"])

    def test_p9c_non_truncation_reason_and_registered_order(self):
        self.assertEqual(P.config_order("J2")[0], "DP2")                       # DP is cut LAST, TP first (MGP-49')

    def test_substrate_gate_mutant_v_killed_and_exact_name(self):
        import mg_substrate as S
        ok, verdict, per = S.run_mutation_checks()
        self.assertTrue(ok, verdict)
        self.assertEqual(per["pcie_J2"]["CONTROL"], "ABORT_SUBSTRATE")
        self.assertEqual(per["pcie_J2"]["v"], "PASS")                           # the substring test would have passed the PCIe card
        self.assertEqual(per["count_J1_needs_2"]["CONTROL"], "ABORT_SUBSTRATE")  # J1 is judged on the x2 SPEC's 2 GPUs
        self.assertEqual(per["ok_J1_two_gpus"]["CONTROL"], "PASS")
        self.assertEqual(S.judge_substrate(S._csv(C.EXACT_GPU_NAME, 2), S._topo(2, "SYS"), "J2")["status"], "ABORT_SUBSTRATE")    # not NV#

    def test_launcher_has_no_global_sweep_for_a_concurrent_group_and_preassigned_ports(self):
        src = (ENGINE / "scripts/multigpu/mg_boot.py").read_text()
        self.assertIn('if not spec.get("concurrent_group")', src)
        self.assertIn('spec.get("port")', src)
        ports = [b["port"] for b in P.boots_gpu_dev()]
        self.assertEqual((len(set(ports)), all(b["concurrent_group"] == "gpu_dev" for b in P.boots_gpu_dev())), (8, True))


# =========================================================================== rev11 (harness audit of 0898540: M-1 … M-4, R-1 … R-5)
def all_cells(drop=lambda cfg, arm, shape, rung: False, agn_boost=lambda n: 1.0):
    """Every (config, arm, shape) key with HI+LO cells, minus the ones `drop` removes (a refused boot2 leaves no LO cells)."""
    cells = []
    for cfg in C.CONFIGS:
        n = 1 if cfg == "N1" else int(cfg[2:])
        for shape in "AB":
            for arm in C.ARMS:
                for c in key_cells(cfg, arm, shape, P.prior(n, shape) * (agn_boost(n) if arm == "agnostic" else 1.0)):
                    if not drop(cfg, arm, shape, c["meta"]["rung"]):
                        cells.append(c)
    return cells


def driver_refusals(boot_ids_specs):
    """Driver records produced by the REAL mg_drive.run_boot admission (deadline already passed), not hand-written dicts."""
    out = []
    for cfg, arm, kind in boot_ids_specs:
        with tempfile.TemporaryDirectory() as td:
            ctx = D.Ctx(td, False, time.time() + 1.0, time.time(), "m3", job="J4")
            spec = dict(P.config_spec(cfg), arm=arm, kind=kind, block="main", boot_id=f"m3_{cfg}_{arm}_{kind}",
                        steps=P.lo_steps(cfg, arm, {"A": {"offered": 1.0, "n_req": 50}, "B": {"offered": 1.0, "n_req": 50}}))
            assert D.run_boot(ctx, spec) is None
            out += ctx.not_attempted
    return out


class Rev11TruncationM1(unittest.TestCase):
    """M-1: truncation is read from the driver's records per (config, arm); documents come from the REAL aggregate()."""
    ARMS_ALL = C.ARMS

    def docs(self, cut_cfgs, drivers_extra=None, drop_extra=lambda *a: False, boost=lambda n: 1.0):
        drop = lambda cfg, arm, shape, rung: (cfg in cut_cfgs and rung == "LO") or drop_extra(cfg, arm, shape, rung)
        cells = all_cells(drop, agn_boost=boost)
        drv = {}
        for J in ("J2", "J4", "J8"):
            n = int(J[1:])
            refused = driver_refusals([(f"{m}{n}", a, "boot2") for m in ("TP", "DP") for a in C.ARMS if f"{m}{n}" in cut_cfgs])
            if refused:
                drv[J] = {"not_attempted": refused}
        for J, extra in (drivers_extra or {}).items():
            drv.setdefault(J, {"not_attempted": []})["not_attempted"] += extra
        return job_docs(cells, driver=drv)

    def test_driver_record_shape_names_config_arm_block(self):
        rec = driver_refusals([("DP4", "agnostic", "boot2")])[0]
        self.assertEqual((rec["reason"], rec["config"], rec["arm"], rec["block"]), ("NOT_ATTEMPTED_BUDGET", "DP4", "agnostic", "main"))

    def test_a_dp_lo_refused_blocks_tp_and_reasons_carry_no_truncation_text(self):
        docs = self.docs({"DP4", "DP8"}, boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1))
        e = docs["J4"]["estimands"]["G_sym"]["DP4"]["A"]
        self.assertFalse(e["defined"])
        self.assertFalse(any("TRUNCATED" in r or "NOT_RUN" in r for r in e["undefined_reasons"]), e["undefined_reasons"])   # the rev10 predicate saw nothing
        self.assertTrue(any("LO_NOT_JUDGED" in r for r in e["undefined_reasons"]))
        doc = L.aggregate_jobs(docs)
        t = doc["trends"]["G_sym|TP|A"]
        self.assertEqual((t["label"], t["direction_citable"], t["label_before_truncation_rule"]), ("TP_UNCONTROLLED_TRUNCATION", False, "G_INCREASES_WITH_N"))
        self.assertEqual(doc["P_DP"]["status"], "UNTESTED")
        self.assertIn("G_sym|DP|A", doc["truncation"]["dp_trend_undefined_by_truncation"])
        u = [x for x in doc["trends"]["G_sym|DP|A"]["undefined"] if x["config"] == "DP4"][0]
        self.assertTrue(u["truncated"])
        self.assertEqual(u["cut_by"], {"agnostic": ["NOT_ATTEMPTED_BUDGET"], "fused_sym": ["NOT_ATTEMPTED_BUDGET"]})

    def test_a_mutant_reason_text_predicate_is_killed_here(self):
        docs = self.docs({"DP4", "DP8"}, boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1))
        mut = L.aggregate_jobs(docs, _mut="trunc_reason")
        self.assertEqual(mut["trends"]["G_sym|TP|A"]["label"], "G_INCREASES_WITH_N")             # mutant cites TP -> differs from TP_UNCONTROLLED_TRUNCATION
        self.assertNotEqual(mut["trends"]["G_sym|TP|A"]["label"], L.aggregate_jobs(docs)["trends"]["G_sym|TP|A"]["label"])

    def test_b_tp_lo_refused_marks_dp_p_dp_only(self):
        docs = self.docs({"TP4", "TP8"})
        doc = L.aggregate_jobs(docs)
        self.assertTrue(doc["truncation"]["tp_trend_undefined_by_truncation"])
        self.assertTrue(all(t.get("used_for_p_dp_only") for k, t in doc["trends"].items() if "|DP|" in k and k.startswith("G_sym")))
        mut = L.aggregate_jobs(docs, _mut="trunc_reason")
        self.assertFalse(any(t.get("used_for_p_dp_only") for k, t in mut["trends"].items() if "|DP|" in k), "mutant misses the TP-side cut")
        # control: nothing refused -> no marker
        clean = L.aggregate_jobs(job_docs(all_cells()))
        self.assertFalse(any(t.get("used_for_p_dp_only") for t in clean["trends"].values()))

    def test_cut_is_per_config_and_arm_not_per_config(self):
        """A d44 boot2 refusal (d44 is not in G_sym) does not truncate G_sym; the same refusal truncates G_d44sym."""
        docs = self.docs(set(), drivers_extra={"J4": driver_refusals([("DP4", "d44", "boot2")])},
                         drop_extra=lambda cfg, arm, shape, rung: cfg == "DP4" and arm == "d44" and rung == "LO")
        doc = L.aggregate_jobs(docs)
        u = lambda key: [x for x in doc["trends"][key]["undefined"] if x["config"] == "DP4"]
        self.assertEqual(u("G_sym|DP|A"), [])                                                       # G_sym(DP4) is defined: nothing undefined
        self.assertTrue(u("G_d44sym|DP|A")[0]["truncated"])

    def test_ext_not_run_text_is_not_truncation_even_with_a_cut_of_another_arm(self):
        """DP4 shape A: fused* arms ran the extension, agnostic did not -> agnostic HI_NOT_JUDGED(EXT_NOT_RUN) (text contains NOT_RUN); the only
        driver cut is d44's.  rev10 matched the substring and called it truncation; rev11 reads the (cfg, arm) records."""
        def extra(cfg, arm, shape, rung):
            return cfg == "DP4" and arm == "d44" and rung == "LO"
        cells = [c for c in all_cells(drop=lambda cfg, arm, shape, rung: extra(cfg, arm, shape, rung))]
        for arm in ("fused", "fused_sym", "d44"):
            cells += [cell_rec("DP4", arm, "A", "HI", sd, P.prior(4, "A") * 1.3 * nz, 0.4, ext=1) for sd, nz in zip(C.HI_SEEDS, (1.0, 1.01, 0.99, 1.005))]
        for c in cells:                                                                               # ext0 of fused trips the joint trigger
            m = c["meta"]
            if (m["config"], m["arm"], m["shape"], m["rung"], m["ext"]) == ("DP4", "fused", "A", "HI", 0):
                c["g"] = 0.05
        docs = job_docs(cells, driver={"J4": {"not_attempted": driver_refusals([("DP4", "d44", "boot2")])}})
        e = docs["J4"]["estimands"]["G_sym"]["DP4"]["A"]
        self.assertTrue(any("EXT_NOT_RUN" in r for r in e["undefined_reasons"]), e["undefined_reasons"])
        doc = L.aggregate_jobs(docs)
        u = [x for x in doc["trends"]["G_sym|DP|A"]["undefined"] if x["config"] == "DP4"][0]
        self.assertFalse(u["truncated"])
        self.assertTrue([x for x in L.aggregate_jobs(docs, _mut="trunc_reason")["trends"]["G_sym|DP|A"]["undefined"] if x["config"] == "DP4"][0]["truncated"],
                        "the rev10 predicate matched the NOT_RUN substring")

    def test_partial_boot_and_truncated_cell_are_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "specs").mkdir()
            (root / "boots" / "m3_DP4_agnostic_boot2").mkdir(parents=True)
            (root / "boots" / "m3_DP4_fused_boot2").mkdir(parents=True)
            (root / "specs" / "m3_DP4_agnostic_boot2.json").write_text(json.dumps({"boot_id": "m3_DP4_agnostic_boot2", "config": "DP4", "arm": "agnostic", "block": "main"}))
            (root / "boots" / "m3_DP4_agnostic_boot2" / "boot.json").write_text(json.dumps(
                {"status": "OK", "steps": [{"type": "cell", "status": "OK"}, {"type": "cell", "status": "NOT_STARTED_DEADLINE"}]}))
            (root / "boots" / "m3_DP4_fused_boot2" / "boot.json").write_text(json.dumps({"status": "OK", "steps": [{"type": "cell", "status": "OK"}]}))
            pb = L.partial_boots_of(root)
            self.assertEqual([(b["boot_id"], b["config"], b["arm"]) for b in pb], [("m3_DP4_agnostic_boot2", "DP4", "agnostic")])
        self.assertEqual(L.driver_cut_arms({"driver": {"partial_boots": pb}}), {("DP4", "agnostic"): ["PARTIAL_BOOT"]})
        # a TP_NOT_TESTED / G3-excluded / anchor / gpu_dev record is NOT truncation
        drv = {"not_attempted": [{"config": "TP4", "reason": "TP_NOT_TESTED"}, {"boot_id": "m2g_TP4_fused", "reason": "NOT_ATTEMPTED_G3_EXCLUDED"},
                                 {"boot_id": "m3_N1_agnostic_anchor", "reason": "NOT_ATTEMPTED_BUDGET"}, {"boot_id": "m3_gpudev_gpu3", "reason": "NOT_ATTEMPTED_BUDGET"}]}
        self.assertEqual(L.driver_cut_arms({"driver": drv}), {})
        # a TRUNCATED cell (runner flag) cuts its arm
        cells = all_cells()
        for c in cells:
            m = c["meta"]
            if (m["config"], m["arm"], m["shape"], m["rung"]) == ("DP4", "agnostic", "A", "HI"):
                c["truncated"], c["status"] = True, "TRUNCATED"
        docs = job_docs(cells)
        self.assertEqual(L.driver_cut_arms(docs["J4"]), {("DP4", "agnostic"): ["TRUNCATED_CELL"]})
        self.assertIn("agnostic:TRUNCATED", docs["J4"]["estimands"]["G_sym"]["DP4"]["A"]["undefined_reasons"])
        self.assertTrue(L.aggregate_jobs(docs)["trends"]["G_sym|DP|A"]["undefined"][0]["truncated"])

    def test_driver_finish_writes_config_arm_and_partial_boots(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = D.Ctx(td, False, time.time() + 1.0, time.time(), "m3", job="J2")
            spec = dict(P.config_spec("TP2"), arm="fused", kind="boot1", block="main", boot_id="m3_TP2_fused_boot1", steps=[])
            D.run_boot(ctx, spec)
            D.finish(ctx, None)
            doc = json.loads((Path(td) / "LABELS.json").read_text())
            self.assertEqual(doc["driver"]["partial_boots"], [])
            self.assertEqual(doc["driver"]["not_attempted"][0]["config"], "TP2")


class Rev11JobEffectM2(unittest.TestCase):
    def j1_cells(self, lo_harness_limited=False, boot1_harness_limited=True):
        cells = [c for c in all_cells() if c["meta"]["config"] != "N1" or c["meta"]["shape"] != "A"]
        for arm in C.ARMS:
            n1 = key_cells("N1", arm, "A", P.prior(1, "A"), g_hi=0.4)
            for c in n1:
                m = c["meta"]
                if arm == "fused" and m["rung"] == "HI":
                    c["g"] = 0.05                                           # the JOINT extension trigger (one arm is enough)
                if arm in C.ANCHOR_ARMS and m["rung"] == "HI" and boot1_harness_limited:
                    c["harness_limited"], c["send_lag_p99_ms"] = True, 9.0   # boot1 send lag 9 ms > 5 ms
                if m["rung"] == "LO" and lo_harness_limited:
                    c["harness_limited"] = True
            cells += n1
            cells += [cell_rec("N1", arm, "A", "HI", sd, P.prior(1, "A") * 1.5 * nz, 0.4, ext=1) for sd, nz in zip(C.HI_SEEDS, (1.0, 1.01, 0.99, 1.005))]
        return cells

    def test_boot1_flags_are_read_from_the_ext0_hi_cells_not_the_final_set(self):
        docs = job_docs(self.j1_cells())
        ref = docs["J1"]["ref"]
        for arm in C.ANCHOR_ARMS:
            self.assertEqual(ref[arm]["final_ext"], 1)
            self.assertTrue(ref[arm]["boot1_harness_limited"], arm)
            self.assertFalse(ref[arm]["harness_limited"], f"{arm}: the final (extension) set is clean -- rev10 read this flag")
            self.assertEqual(ref[arm]["boot1_hi_label"], "HI_SATURATED")
        doc = L.aggregate_jobs(docs)
        for J in ("J2", "J4", "J8"):
            self.assertEqual(doc["job_effect"][J]["status"], "UNTESTED", J)
            self.assertTrue(any("J1_REFERENCE_HARNESS_LIMITED" in r for r in doc["job_effect"][J]["reasons"]))
        mut = L.aggregate_jobs(docs, _mut="flags_final")                      # mutant: final-set flags
        self.assertNotEqual({mut["job_effect"][J]["status"] for J in ("J2", "J4", "J8")}, {"UNTESTED"})

    def test_lo_cells_and_clean_boot1_do_not_make_the_reference_unclean(self):
        docs = job_docs(self.j1_cells(lo_harness_limited=True, boot1_harness_limited=False))
        self.assertFalse(docs["J1"]["ref"]["fused_sym"]["boot1_harness_limited"])
        # (the harness-limited LO cells do make the G_sym(1, A) estimand undefined -> P9-a; the Job-effect check itself reads boot1 HI cells only)
        self.assertEqual(L.job_effect(docs["J1"]["ref"], docs["J2"]["anchor"])["status"], "CLEAR")

    def test_reference_without_boot1_flags_is_untested_fail_closed(self):
        ref = L._syn_ref()
        for a in ref:
            del ref[a]["boot1_harness_limited"]
        self.assertEqual(L.job_effect(ref, L._syn_anchor(1.0))["status"], "UNTESTED")


class Rev11JoinM3(unittest.TestCase):
    """M-3: `mg_label.py --jobs` admits a Job directory only with its own sha-checked M2G_PASS_TABLE.used.json and BUDGET.txt job=."""
    def make_jobs(self, td, tables=None, jobs=("J1", "J2"), budget_job=None, abort=None):
        import hashlib
        from unittest import mock
        root, shas = Path(td) / "jobs", Path(td) / "sha"
        shas.mkdir()
        dirs = {}
        for J in jobs:
            d = root / J
            d.mkdir(parents=True)
            (d / "BUDGET.txt").write_text(f"kind=m3 job={(budget_job or {}).get(J, J)} hard_cap_s=100 t0_epoch=1 deadline_epoch=101 dryrun=0\n")
            (d / "LABELS.json").write_text(json.dumps({"driver": {"not_attempted": []}}))
            raw = json.dumps((tables or {}).get(J, {"cells": {}, "configs": {}, "g5a_unit_test": "PASS"}))
            (d / "M2G_PASS_TABLE.used.json").write_text(raw)
            (shas / f"M2G_PASS_TABLE_{J}.json.sha256").write_text(hashlib.sha256(raw.encode()).hexdigest() + "  M2G_PASS_TABLE.json\n")
            if abort and J in abort:
                (d / "ABORT").write_text("ABORT_X\n")
            dirs[J] = d
        return dirs, shas

    def run_cli(self, dirs, shas, extra=()):
        """The real CLI entry (main) with load_cell/find_cells replaced by synthetic cells of the Job's own N."""
        from unittest import mock
        import io
        cells = all_cells()
        by = {j: [] for j in C.JOBS}
        for c in cells:
            cfg = c["meta"]["config"]
            by["J1" if cfg == "N1" else f"J{int(cfg[2:])}"].append(c)
        which = {str(d): by[J] for J, d in dirs.items()}
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(L, "find_cells", lambda root: [Path(f"{root}/c{i}") for i in range(len(which[str(root)]))]), \
             mock.patch.object(L, "load_cell", lambda cd: which[str(Path(cd).parent)][int(Path(cd).name[1:])]), \
             mock.patch.object(L, "job_covariates", lambda root: {}), \
             mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            rc = L.main(["--jobs", *[f"{J}={d}" for J, d in dirs.items()], "--committed-sha-dir", str(shas), *extra])
        return rc, out.getvalue(), err.getvalue()

    def test_clean_join_uses_each_jobs_used_table(self):
        with tempfile.TemporaryDirectory() as td:
            tables = {"J2": {"cells": {}, "configs": {"DP2": {"flags": ["MEMORY_MISMATCH"]}}, "g5a_unit_test": "PASS"}}
            dirs, shas = self.make_jobs(td, tables)
            rc, out, err = self.run_cli(dirs, shas, ["--out", str(Path(td) / "joined.json")])
            self.assertEqual(rc, 0, err)
            doc = json.loads((Path(td) / "joined.json").read_text())
            # J2's table says MEMORY_MISMATCH for DP2: DP2 is undefined, so the DP trend has a hole beside the label
            self.assertTrue(any(u["config"] == "DP2" and any("MEMORY_MISMATCH" in r for r in u["reasons"]) for u in doc["trends"]["G_sym|DP|A"]["undefined"]))

    def test_join_without_a_table_is_refused_and_the_table_free_mutant_is_killed(self):
        with tempfile.TemporaryDirectory() as td:
            tables = {"J2": {"cells": {}, "configs": {"DP2": {"flags": ["MEMORY_MISMATCH"]}}, "g5a_unit_test": "PASS"}}
            dirs, shas = self.make_jobs(td, tables)
            (dirs["J2"] / "M2G_PASS_TABLE.used.json").unlink()
            rc, out, err = self.run_cli(dirs, shas)
            self.assertEqual(rc, 3)
            self.assertIn("JOIN_REFUSED", err)
            self.assertIn("no pass table", err)
            self.assertEqual(out, "")
            # mutant: admission without the table -> the join succeeds and the MEMORY_MISMATCH vanishes (the rev10 behaviour)
            used, t = L.job_dir_admission("J2", dirs["J2"], shas, _mut="no_table")
            self.assertEqual((used, t), (None, {}))

    def test_edited_or_wrong_jobs_table_fails_the_sha_check(self):
        with tempfile.TemporaryDirectory() as td:
            dirs, shas = self.make_jobs(td)
            (dirs["J2"] / "M2G_PASS_TABLE.used.json").write_text(json.dumps({"cells": {}, "configs": {}, "g5a_unit_test": "PASS", "x": 1}))
            rc, _o, err = self.run_cli(dirs, shas)
            self.assertEqual(rc, 3)
            self.assertIn("committed .sha256", err)
            dirs, shas = self.make_jobs(td + "/b" if False else tempfile.mkdtemp(dir=td))
            (shas / "M2G_PASS_TABLE_J2.json.sha256").unlink()
            rc, _o, err = self.run_cli(dirs, shas)
            self.assertEqual(rc, 3)
            self.assertIn("missing", err)

    def test_budget_job_must_equal_cli_job_and_abort_dryrun_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            dirs, shas = self.make_jobs(td, budget_job={"J2": "J4"})
            rc, _o, err = self.run_cli(dirs, shas)
            self.assertEqual(rc, 3)
            self.assertIn("BUDGET.txt job='J4' != CLI J2", err)
        with tempfile.TemporaryDirectory() as td:
            dirs, shas = self.make_jobs(td, abort={"J2"})
            self.assertEqual(self.run_cli(dirs, shas)[0], 3)
        with tempfile.TemporaryDirectory() as td:
            dirs, shas = self.make_jobs(td)
            (dirs["J1"] / "DRYRUN").write_text("x")
            self.assertEqual(self.run_cli(dirs, shas)[0], 3)
        with tempfile.TemporaryDirectory() as td:
            dirs, shas = self.make_jobs(td)
            (dirs["J2"] / "LABELS.json").unlink()
            rc, _o, err = self.run_cli(dirs, shas)
            self.assertEqual(rc, 3)
            self.assertIn("driver", err)


class Rev11PDPM4(unittest.TestCase):
    """M-4 (PREREG rev11 §18, user decision fail-closed): the audit's reversal scenario."""
    @staticmethod
    def scenario_docs():
        return L._syn_docs(1.0, {"DP2": 1.0, "DP4": 1.0, "DP8": 1.0, "TP2": 1.2, "TP4": 1.5, "TP8": 2.0}, {2: 1.00, 4: 1.10, 8: 1.21},
                           series_b={"DP2": 1.0, "DP4": 1.10, "DP8": 1.21})

    def test_reversal_scenario_p_dp_broken_and_mgp13_fires(self):
        doc = L.aggregate_jobs(self.scenario_docs())
        self.assertEqual({J: doc["job_effect"][J]["status"] for J in ("J2", "J4", "J8")}, {"J2": "CLEAR", "J4": "FIRED", "J8": "FIRED"})
        a, b = doc["trends"]["G_sym|DP|A"], doc["trends"]["G_sym|DP|B"]
        self.assertEqual(b["label"], L.TREND_JOB_SENSITIVE)                       # the cross-shape correction still lowers the published label ...
        self.assertEqual(b["raw_label"], "G_INCREASES_WITH_N")
        self.assertEqual(a["label"], L.TREND_JOB_SENSITIVE)                       # shape A: the same-shape correction (current behaviour) is untouched
        self.assertEqual(doc["P_DP"]["per_shape"], {"G_sym|DP|A": "UNTESTED", "G_sym|DP|B": "BROKEN"})
        self.assertEqual((doc["P_DP"]["status"], doc["P_DP"]["MGP13_fired"], doc["P_DP"]["MGP13_TP_labels_not_citable"]), ("BROKEN", True, True))
        self.assertIn("P-DP BROKEN: MGP-13 TP labels not citable", doc["trends"]["G_sym|TP|A"]["disclosures"])
        self.assertIn("G_sym|DP|B", doc["P_DP"]["raw_label_used"])

    def test_rev10_reading_is_killed(self):
        mut = L.aggregate_jobs(self.scenario_docs(), _mut="same_shape_only")
        self.assertEqual(mut["P_DP"]["status"], "UNTESTED")                          # what the rev10 harness reported
        self.assertFalse(mut["P_DP"]["MGP13_fired"])

    def test_shape_a_same_shape_correction_still_lowers_to_untested(self):
        tr = {"G_sym|DP|A": {"label": L.TREND_JOB_SENSITIVE, "raw_label": "G_INCREASES_WITH_N", "g1_defined": True},
              "G_sym|DP|B": {"label": "NO_TREND_DETECTED", "g1_defined": True}}
        self.assertEqual(L.p_dp_status(tr)["status"], "UNTESTED")
        tr["G_sym|DP|B"] = {"label": L.TREND_JOB_SENSITIVE, "raw_label": "G_INCREASES_WITH_N", "g1_defined": True}
        self.assertEqual(L.p_dp_status(tr)["status"], "BROKEN")
        tr["G_sym|DP|B"]["g1_defined"] = False                                       # G_sym(1, B) undefined: not BROKEN (P9-a stays)
        self.assertEqual(L.p_dp_status(tr)["status"], "UNTESTED")
        tr["G_sym|DP|B"] = {"label": L.TREND_JOB_SENSITIVE, "raw_label": "TREND_UNDEFINED", "g1_defined": True}      # raw trend not defined -> not BROKEN
        self.assertEqual(L.p_dp_status(tr)["status"], "UNTESTED")


class Rev11Misc(unittest.TestCase):
    def test_prereg_rev_is_11_and_documents_say_so(self):
        self.assertEqual(C.PREREG_REV, 11)
        docs = job_docs(all_cells())
        self.assertTrue(docs["J1"]["prereg"].endswith("rev11"))
        self.assertTrue(L.aggregate_jobs(docs)["prereg"].endswith("rev11"))

    def test_r1_group_sweep_runs_once_after_the_group(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            calls = []
            fake = Path(td) / "ok.sh"
            fake.write_text("#!/bin/bash\nexit 0\n")
            with mock.patch.object(D, "RUNNER", str(fake)), mock.patch.object(D.subprocess, "run", lambda *a, **k: calls.append(a[0])):
                ctx = D.Ctx(td, False, time.time() + 36000, time.time(), "m3", job="J8")
                rcs = D.run_boot_group(ctx, P.boots_gpu_dev())
            self.assertEqual(set(rcs.values()), {0})
            sweeps = [c for c in calls if c[:2] == ["pkill", "-9"]]
            self.assertEqual(len(sweeps), 1)                                         # exactly once, though no rc was -9
            self.assertEqual([e for e in ctx.log if e.get("group") == "gpu_dev"][-1]["global_sweep"], True)
            # dry-run: no sweep
            calls.clear()
            with mock.patch.object(D, "RUNNER", str(fake)), mock.patch.object(D.subprocess, "run", lambda *a, **k: calls.append(a[0])):
                D.run_boot_group(D.Ctx(td + "/d", True, time.time() + 36000, time.time(), "m3", job="J8"), P.boots_gpu_dev())
            self.assertEqual([c for c in calls if c[:2] == ["pkill", "-9"]], [])

    def test_r3_g4_nsys_boot_is_not_run_when_nsys_is_absent(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            ran = []
            for have in (False, True):
                ran.clear()
                ctx = D.Ctx(Path(td) / str(have), False, time.time() + 36000, time.time(), "m2g", job="J2")
                ctx.nsys_path = "/usr/bin/nsys" if have else None
                ctx.g5a_argv = [sys.executable, "-c", "pass"]
                with mock.patch.object(D, "run_boot", lambda c, spec, expected_counts=None: ran.append(spec["boot_id"])), \
                     mock.patch.object(D, "run_g3", lambda c: None), mock.patch.object(D, "g3_excluded_tps", lambda c: set()), \
                     mock.patch.object(D, "preboot_nvml_check", lambda c: None), \
                     mock.patch.object(D, "g1_exposure", lambda c: None), mock.patch.object(D, "score_g1_all", lambda c: None), \
                     mock.patch.object(D, "g5a_gpu_descriptive", lambda c: None), mock.patch.object(D.subprocess, "run", lambda *a, **k: None):
                    D.run_m2g(ctx)
                g4 = [b for b in ran if b.startswith("m2g_g4_")]
                if have:
                    self.assertEqual(g4, ["m2g_g4_TP2_d44"])
                else:
                    self.assertEqual(g4, [])
                    self.assertEqual([x for x in ctx.not_attempted if x["reason"] == "NOT_RUN_NSYS_ABSENT"][0]["boot_id"], "m2g_g4_TP2_d44")

    def test_r5_launch_dry_run_prints_j2_table_and_sha_presence(self):
        env_file = Path(td := tempfile.mkdtemp()) / "env"
        env_file.write_text('PDMUX_VESSL_SPEC=resourcespec-a100x4\nPDMUX_VESSL_RATE_USD_H=5.92\nPDMUX_VESSL_CLUSTER_VOLUME=cv\n'
                            'PDMUX_VESSL_OBJECT_VOLUME=ov\nPDMUX_VESSL_IMAGE=img@sha256:' + "0" * 64 + '\n')
        launch = str(ENGINE / "scripts/vessl/launch.sh")
        base = ["bash", launch, "--campaign", "multigpu", "--target", "workspace/engine-port/results/multigpu/m3_job.sh"]
        env = dict(os.environ, PDMUX_VESSL_ENV=str(env_file))
        run = lambda *e: subprocess.run(base + [x for kv in e for x in ("--env", kv)], env=env, capture_output=True, text=True)
        try:
            r = run("PDMUX_MG_JOB=J4", "PDMUX_M2G_PASS_SHA256=" + "a" * 64)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("J2 table : M2G_PASS_TABLE_J2.json", r.stdout)
            self.assertIn("env PDMUX_M2G_PASS_SHA256_J2 NOT SET -> TP configs NOT_TESTED", r.stdout)
            self.assertIn("own table: M2G_PASS_TABLE_J4.json", r.stdout)
            r = run("PDMUX_MG_JOB=J8", "PDMUX_M2G_PASS_SHA256=" + "a" * 64, "PDMUX_M2G_PASS_SHA256_J2=" + "b" * 64)
            self.assertIn("PDMUX_M2G_PASS_SHA256_J2 set (64 chars)", r.stdout)
            r = run("PDMUX_MG_JOB=J2", "PDMUX_M2G_PASS_SHA256=" + "a" * 64)
            self.assertNotIn("J2 table :", r.stdout)
        finally:
            shutil_rmtree(td)


def shutil_rmtree(p):
    import shutil
    shutil.rmtree(p, ignore_errors=True)


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
if spec["boot_id"] == os.environ.get("FAKE_SKIP"): sys.exit(1)
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
    def run_driver(self, td, ratio, boost, extra_args=(), job="J2", table=None, j2_table=None, out_name="out", fake_skip=""):
        import hashlib
        fake = Path(td) / "fake_runner.sh"
        (Path(td) / "fake.py").write_text(FAKE_RUNNER_PY)
        fake.write_text(f'#!/bin/bash\nexec python3 {td}/fake.py "$@"\n')
        out = Path(td) / out_name
        env = dict(os.environ, PDMUX_MG_RUNNER=str(fake), MG_RESULTS_DIR=str(ENGINE / "results/multigpu"),
                   FAKE_RATIO=str(ratio), FAKE_AGN_BOOST=str(boost), FAKE_SKIP=fake_skip, PYTHONDONTWRITEBYTECODE="1")
        raw = json.dumps(table or {"cells": {}, "g5a_unit_test": "PASS", "g0ii": "PASS"})
        tbl = Path(td) / f"t_{job}_{out_name}.json"
        tbl.write_text(raw)
        args = ["m3", "--job", job, "--out", str(out), "--pass-table", str(tbl), "--pass-sha256", hashlib.sha256(raw.encode()).hexdigest()]
        if j2_table is None and job in ("J4", "J8"):
            j2_table = {"cells": {}, "g5a_unit_test": "PASS", "g0ii": "PASS"}            # default: J2 released the TP axis
        if j2_table is not None and j2_table is not False:
            raw2 = json.dumps(j2_table)
            t2 = Path(td) / f"j2_{out_name}.json"
            t2.write_text(raw2)
            args += ["--j2-pass-table", str(t2), "--j2-pass-sha256", hashlib.sha256(raw2.encode()).hexdigest()]
        p = subprocess.run([sys.executable, str(ENGINE / "results/multigpu/mg_drive.py"), *args, *extra_args], env=env, capture_output=True, text=True)
        return p, out

    def events(self, out):
        return [json.loads(l) for l in (out / "DRIVER_LOG.jsonl").read_text().splitlines()]

    def test_j2_order_anchor_first_no_extension_and_lo_shared_across_arms(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0, job="J2")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertEqual(doc["job"], "J2")
            self.assertEqual(doc["n_cells"], 8 + 2 * 4 * 2 * 6)                                   # 2 anchor boots x 4 + (DP2, TP2) x 4 arms x 2 shapes x 6
            ev = self.events(out)
            self.assertEqual(next(e for e in ev if "config_order" in e)["config_order"], ["DP2", "TP2"])      # registered order, logged
            boots = [e["boot"] for e in ev if "boot" in e and "rc" in e]
            self.assertTrue(all(b.startswith("m3_N1_") and b.endswith("_anchor") for b in boots[:2]), boots[:3])    # anchors BEFORE any config
            first_cfg = next(i for i, b in enumerate(boots) if not b.startswith("m3_N1_"))
            self.assertTrue(boots[first_cfg].startswith("m3_DP2_") and not any(b.startswith("m3_TP2_") for b in boots[:first_cfg]))
            self.assertEqual(sorted(d["ext_shapes"] for k, d in doc["driver"]["decisions"].items() if k in ("DP2", "TP2")), [[], []])
            e = doc["estimands"]["G_sym"]["TP2"]["A"]
            self.assertTrue(e["defined"], e)
            self.assertAlmostEqual(e["point"], 1.0, 2)
            for cfg in ("DP2", "TP2"):                      # x_lo identical across the four arms (MGP-9)
                for sh in "AB":
                    xs = set()
                    for d in (out / "cells").glob(f"m3_{cfg}_*_boot2__{sh}_LO_s4386"):
                        m = json.loads((d / "cell.json").read_text())
                        xs.add((m["offered"], m["n_req"]))
                    self.assertEqual(len(xs), 1, (cfg, sh, xs))
            for banned in ("numa", "nsys", "anchor2"):
                self.assertFalse(list((out / "specs").glob(f"*{banned}*")))
            a = doc["anchor"]["arms"]
            self.assertTrue(all(a[x]["hi_label"] == "HI_SATURATED" and a[x]["present"] for x in C.ANCHOR_ARMS), a)
            self.assertIn("N1@anchor", doc["keys"] and {k.split("|")[0] for k in doc["keys"]})
            self.assertNotIn("N1@anchor", doc["configs"])

    def test_j1_n1_only_no_anchor_and_j8_gpu_dev_block_first(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0, job="J1")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertEqual((doc["n_cells"], doc["anchor"], sorted(doc["estimands"]["G_sym"])), (48, {}, ["N1"]))
            self.assertTrue(doc["ref"]["fused_sym"]["present"] and doc["ref"]["agnostic"]["boot1_lambda_star"])
            p, out = self.run_driver(td, 1.0, 1.0, job="J8", out_name="out8")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertEqual(doc["n_cells"], 8 + 8 + 96)
            self.assertEqual(doc["blocks"]["gpu_dev"]["n_gpus"], 8)
            ev = self.events(out)
            grp = next(i for i, e in enumerate(ev) if e.get("group") == "gpu_dev")
            self.assertTrue(all(not (("boot" in e) and e["boot"].startswith("m3_N1_")) for e in ev[:grp]))      # nothing before the concurrent block
            self.assertEqual(sorted(json.loads(sp.read_text())["port"] for sp in (out / "specs").glob("m3_gpudev_*.json")), list(range(20700, 20708)))
            self.assertEqual(next(e for e in ev if "config_order" in e)["config_order"], ["DP8", "TP8"])

    def test_joint_extension_when_lambda_far_above_prior(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.7, 1.0, job="J4")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            doc = json.loads((out / "LABELS.json").read_text())
            dec = {k: v for k, v in doc["driver"]["decisions"].items() if k in ("DP4", "TP4")}
            self.assertTrue(all(d["ext_shapes"] == ["A", "B"] for d in dec.values()), dec)
            specs = [json.loads(p.read_text()) for p in (out / "specs").glob("m3_TP4_*_boot1ext.json")]
            self.assertEqual(len(specs), 4)                     # all four arms jointly (MGP-9)
            self.assertFalse(list((out / "specs").glob("m3_N1_*ext*")))        # an anchor is never extended
            self.assertTrue(all(s["ext"] == 1 for sp in (out / "specs").glob("m3_TP4_*_boot2.json")
                                for s in json.loads(sp.read_text())["steps"] if s["type"] == "cell"))

    def test_four_jobs_join_constant_boost_job_effect_clear_and_magnitude_rule(self):
        """Deterministic synthetic data through all four Job drivers and the cross-Job join: a CONSTANT 1.15x boost -> every anchor equals
        the J1 reference (JOB_EFFECT CLEAR) and no N-trend is citable under the 3 % rule (a ~0.1 % N-dependence may still have a CI excluding 0)."""
        with tempfile.TemporaryDirectory() as td:
            docs = {}
            for j in C.JOBS:
                p, out = self.run_driver(td, 1.0, 1.15, job=j, out_name=f"o{j}")
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                docs[j] = json.loads((out / "LABELS.json").read_text())
            self.assertAlmostEqual(docs["J2"]["estimands"]["G_sym"]["DP2"]["A"]["point"], 1.15, 1)
            doc = L.aggregate_jobs(docs)
            self.assertEqual({j: s["status"] for j, s in doc["job_effect"].items()}, {"J2": "CLEAR", "J4": "CLEAR", "J8": "CLEAR"})
            t = doc["trends"]["G_sym|TP|A"]
            self.assertEqual((t["undefined_count"], t["Ns"]), (0, [1, 2, 4, 8]))
            if t["label"] != "NO_TREND_DETECTED":
                self.assertEqual(t["magnitude"], "DIRECTION_NOT_CITABLE_UNDER_3PCT")
                self.assertLess(abs(t["magnitude_ratio"] - 1.0), 0.03)
            self.assertIn("JOB_EFFECT", L.render_jobs_text(doc))

    def test_hard_cap_admission_records_not_attempted(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0, ["--hard-cap-s", "1500"], job="J2")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            st = json.loads((out / "DRIVER_STATUS.json").read_text())
            self.assertGreater(st["not_attempted"], 5)
            doc = json.loads((out / "LABELS.json").read_text())
            self.assertTrue(doc["driver"]["not_attempted"])
            self.assertLess(doc["n_cells"], 104)

    def test_missing_pass_table_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            p = subprocess.run([sys.executable, str(ENGINE / "results/multigpu/mg_drive.py"), "m3", "--job", "J2", "--out", str(out),
                                "--pass-table", str(Path(td) / "none.json"), "--pass-sha256", "0" * 64],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3)
            self.assertIn("ABORT_M2G_NOT_PASSED", (out / "ABORT").read_text())

    def test_tp_axis_needs_j2_g0ii_and_missing_j2_table_leaves_dp_running(self):
        """J4/J8 read J2's table for G0(ii): no table -> TP NOT_TESTED, DP proceeds, NO abort.  J2's own table decides J2's TP."""
        tbl4 = {"cells": {}, "g5a_unit_test": "PASS", "g0ii": "NOT_RUN_IN_JOB"}
        with tempfile.TemporaryDirectory() as td:
            p, out = self.run_driver(td, 1.0, 1.0, job="J4", table=tbl4, j2_table=False)                      # no J2 table at all
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertFalse(list((out / "specs").glob("m3_TP4_*")))
            self.assertTrue(list((out / "specs").glob("m3_DP4_*")))
            na = json.loads((out / "LABELS.json").read_text())["driver"]["not_attempted"]
            self.assertTrue(any(x.get("config") == "TP4" and x["reason"] == "TP_NOT_TESTED" for x in na))
            p, out = self.run_driver(td, 1.0, 1.0, job="J4", table=tbl4, j2_table={"cells": {}, "g5a_unit_test": "PASS", "g0ii": "PASS"}, out_name="ok")
            self.assertTrue(list((out / "specs").glob("m3_TP4_*")))                          # control: a PASS J2 table releases TP
            p, out = self.run_driver(td, 1.0, 1.0, job="J4", table=tbl4, j2_table={"cells": {}, "g5a_unit_test": "PASS", "g0ii": "FAIL"}, out_name="fail")
            self.assertFalse(list((out / "specs").glob("m3_TP4_*")))
            p, out = self.run_driver(td, 1.0, 1.0, job="J2", table={"cells": {}, "g5a_unit_test": "PASS", "g0ii": "FAIL"}, out_name="j2fail")
            self.assertFalse(list((out / "specs").glob("m3_TP2_*")))
            self.assertTrue(list((out / "specs").glob("m3_DP2_*")))


class AssemblePassTable(unittest.TestCase):
    """mg_drive.assemble_pass_table(out, job) reads ONE Job's files and delegates scoring + the §3-1 mapping to m2g_correctness_gate."""

    def test_nothing_run_excludes_everything_not_tested(self):
        with tempfile.TemporaryDirectory() as td:
            t = D.assemble_pass_table(Path(td), "J2")
            self.assertEqual(sorted({k.split(":")[0] for k in t["cells"]}), ["DP2", "N1", "TP2"])          # only this Job's configs
            self.assertTrue(all(not v["included"] for v in t["cells"].values()))
            self.assertIn("NOT_TESTED", t["cells"]["TP2:fused"]["reason"])
            self.assertEqual(len((Path(td) / "M2G_PASS_TABLE_J2.json.sha256").read_text().strip()), 64)
            self.assertFalse((Path(td) / "M2G_PASS_TABLE.json").exists())
            j1 = D.assemble_pass_table(Path(td), "J1")
            self.assertEqual(sorted({k.split(":")[0] for k in j1["cells"]}), ["N1"])

    @staticmethod
    def build(out, job="J2", drop_boot=()):
        """Healthy artifacts for ONE Job: its configs (N1 + TP_N / DP_N); J2 additionally has the five G0(ii) refusals and the G0(iii) boot."""
        if job == "J2":
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
        (out / "boots").mkdir(exist_ok=True)
        for cfg in P.m2g_configs(job):
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
            self.build(out, "J2", drop_boot={("TP2", "fused")})
            t = D.assemble_pass_table(out, "J2")
            self.assertIn("G4 capture NOT_TESTED", t["cells"]["TP2:fused"]["reasons"])
            self.assertNotIn("G4 capture NOT_TESTED", " ".join(t["cells"]["TP2:d44"]["reasons"]))   # negative control: healthy boot

    def test_j2_full_pass_and_per_gate_exclusions(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out, "J2")
            t = D.assemble_pass_table(out, "J2")
            self.assertEqual((t["job"], t["g0ii"]), ("J2", "PASS"))
            self.assertTrue(all(t["cells"][f"TP2:{a}"]["included"] for a in C.ARMS), t["cells"]["TP2:fused"])
            self.assertTrue(all(t["cells"][f"N1:{a}"]["included"] for a in C.ARMS))
            # DP: no telemetry for the realized-partition check -> G5(b) NOT_TESTED excludes the whole DP=N (§3-1)
            self.assertTrue(all(not t["cells"][f"DP2:{a}"]["included"] and "G5(b)" in t["cells"][f"DP2:{a}"]["reason"] for a in C.ARMS))
            self.assertFalse(t["configs"]["TP2"]["memory_mismatch"])
            self.assertTrue(t["nvml_procs_visible"])
            self.assertEqual(t["nvml_expected_procs"]["TP2:d44"], {"0": 1})
            # negative controls: each failed gate removes exactly its registered range
            (out / "gates" / "g1_TP2.json").write_text(json.dumps({"verdicts": {"fused|agnostic": "FAIL", "fused|d44": "UNINFORMATIVE"}}))
            (out / "boots" / "m2g_refuse_0" / "boot.json").write_text(json.dumps({"kind": "g0ii", "t_health_epoch": 1, "server_exit_code": None,
                                                                                    "health_wait_cap_s": 1800}))
            t = D.assemble_pass_table(out, "J2")
            self.assertTrue(all(not t["cells"][f"TP2:{a}"]["included"] for a in C.ARMS))              # G0(ii) -> TP axis out
            self.assertTrue(t["cells"]["N1:fused"]["included"])
            (out / "boots" / "m2g_refuse_0" / "boot.json").write_text(json.dumps({"kind": "g0ii", "t_health_epoch": None,
                                                                                    "server_exit_code": 1, "health_wait_cap_s": 1800}))
            t = D.assemble_pass_table(out, "J2")
            self.assertTrue(t["cells"]["TP2:fused"]["included"] and t["cells"]["TP2:d44"]["included"])  # UNINFORMATIVE excludes nothing
            self.assertFalse(t["cells"]["TP2:agnostic"]["included"])
            self.assertTrue(t["cells"]["TP2:fused_sym"]["included"])

    def test_j4_table_does_not_judge_g0ii_and_scopes_g3_boot_failures_to_its_units(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out, "J4")
            t = D.assemble_pass_table(out, "J4")
            self.assertEqual((t["g0ii"], t["g0iii"]), ("NOT_RUN_IN_JOB", "NOT_RUN_IN_JOB"))
            self.assertTrue(all(t["cells"][f"TP4:{a}"]["included"] for a in C.ARMS))             # no refusal boots in J4 -> no TP exclusion from them
            self.assertEqual(sorted({k.split(":")[0] for k in t["cells"]}), ["DP4", "N1", "TP4"])
            (out / "gates" / "g3_TP4.json").write_text(json.dumps({"verdict": "FAIL"}))
            (out / "boots" / "m2g_TP4_fused_sym" / "boot.json").write_text(json.dumps({"status": "BOOT_FAILED", "health_wait_cap_s": 1800,
                                                                                         "server_exit_code": 1}))
            t = D.assemble_pass_table(out, "J4")
            self.assertFalse(t["cells"]["TP4:fused"]["included"])                                  # G3 FAIL: whole TP4
            self.assertTrue(t["cells"]["N1:fused"]["included"])
            (out / "gates" / "g3_TP4.json").write_text(json.dumps({"verdict": "PASS"}))
            t = D.assemble_pass_table(out, "J4")
            self.assertFalse(t["cells"]["TP4:fused_sym"]["included"])                              # G0(i): that arm only
            self.assertTrue(t["cells"]["TP4:fused"]["included"] and t["cells"]["TP4:agnostic"]["included"])
            self.assertTrue(t["configs"]["TP4"]["g_sym_undefined"])
            self.assertEqual(D.job_tps("J1"), ())
            self.assertEqual(D.job_tps("J8"), (8,))

    def test_short_health_wait_is_not_tested_not_fail(self):                                           # MGP-23
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out, "J2")
            (out / "boots" / "m2g_TP2_d44" / "boot.json").write_text(json.dumps({"status": "BOOT_TIMEOUT", "health_wait_cap_s": 300,
                                                                                  "server_exit_code": None}))
            t = D.assemble_pass_table(out, "J2")
            r = t["cells"]["TP2:d44"]
            self.assertFalse(r["included"])
            self.assertTrue(r["not_tested"])

    def test_m3_reads_the_table_it_got(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.build(out, "J2")
            t = D.assemble_pass_table(out, "J2")
            self.assertEqual(P.excluded_set(t) & {"TP2:fused", "TP2:d44"}, set())
            self.assertEqual(P.runnable_configs(t, "J2"), ["TP2"])                                  # DP2 excluded by G5(b); order = registered [DP, TP]
            self.assertEqual(P.anchor_exclude(t), set())
            self.assertEqual(P.tp_allowed_by_j2(t), (True, None))
            self.assertFalse(P.tp_allowed_by_j2(None)[0])
            self.assertFalse(P.tp_allowed_by_j2(dict(t, g0ii="FAIL"))[0])
            self.assertFalse(P.tp_allowed_by_j2({"g0ii": "NOT_RUN_IN_JOB"})[0])


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
        doc = L.aggregate_jobs(job_docs(self.build()))
        self.assertEqual(doc["P_DP"]["status"], "HELD")
        self.assertNotIn("P-DP 미검정", doc["trends"]["G_sym|TP|A"]["disclosures"])
        drop = {(f"DP{n}", "fused_sym") for n in (2, 4, 8)}
        doc = L.aggregate_jobs(job_docs(self.build(drop=drop)))
        self.assertEqual(doc["P_DP"]["status"], "UNTESTED")
        self.assertFalse(doc["P_DP"]["broken"])                                           # the old code said 'not broken' here
        self.assertTrue(doc["P_DP"]["TP_labels_need_P_DP_UNTESTED_disclosure"])
        for sh in "AB":
            self.assertIn("P-DP 미검정", doc["trends"][f"G_sym|TP|{sh}"]["disclosures"])
        self.assertIn("P-DP 미검정", L.render_jobs_text(doc))
        doc = L.aggregate_jobs(job_docs(self.build(agn_boost=lambda n: 1.0 + 0.1 * (n.bit_length() - 1))))
        self.assertEqual(doc["P_DP"]["status"], "BROKEN")

    def test_mix_failed_and_e_time_undefined_propagate_to_estimand_and_trend(self):
        cs = self.build()
        for c in cs:
            if c["meta"]["arm"] in C.PDMUX_ARMS and c["meta"]["rung"] == "HI":
                c["etime"] = {"status": "OK", "e_time": 0.9}
        docs = job_docs(cs)
        doc = docs["J2"]
        e = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertEqual(e["partition"], {"agnostic": "PARTITION_EXERCISED"})
        self.assertFalse(e["partition_exercised_unknown"])
        for c in cs:
            m = c["meta"]
            if (m["config"], m["arm"], m["shape"], m["seed"], m["rung"]) == ("TP2", "agnostic", "A", C.HI_SEEDS[1], "HI"):
                c["etime"] = {"status": "MIX_FAILED", "e_time": None, "reasons": ["x"]}
        docs = job_docs(cs)
        doc = docs["J2"]
        e = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertEqual(e["partition"]["agnostic"], "E_TIME_UNDEFINED")
        self.assertTrue(e["partition_exercised_unknown"] and e["mix_failed_arms"] == ["agnostic"])
        self.assertTrue(e["defined"])                                                     # a flag, not an exclusion
        t = L.aggregate_jobs(docs)["trends"]["G_sym|TP|A"]
        self.assertTrue(t["partition_exercised_unknown"])

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
        docs = job_docs(self.build(), mcs=mc)
        doc = {"estimands": {n: {c: e for J in docs.values() for c, e in J["estimands"][n].items()} for n in docs["J2"]["estimands"]}}
        e2 = doc["estimands"]["G_sym"]["TP2"]["A"]
        self.assertTrue(e2["defined"])
        self.assertIn("TP_NUMERIC_DIVERGENT", e2["disclosures"])
        self.assertIn("agnostic:G1_UNINFORMATIVE", e2["disclosures"])
        self.assertIn("G2_NOT_TESTED", doc["estimands"]["G_sym"]["TP4"]["A"]["disclosures"])
        e8 = doc["estimands"]["G_sym"]["TP8"]["A"]
        self.assertFalse(e8["defined"])                                                   # M2g MEMORY_MISMATCH carries (G5(c) meaning unified)
        self.assertIn("MEMORY_MISMATCH", e8["undefined_reasons"])
        t = L.aggregate_jobs(docs)["trends"]["G_sym|TP|A"]
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
    def test_g5a_gpu_boot_is_in_the_m2g_plan(self):                                         # F4 (J1 only: §15-2, the GPU half is not per Job)
        b = [x for x in P.m2g_boots("J1") if x["boot_id"] == "m2g_g5a_auto_N1"]
        self.assertEqual(len(b), 1)
        self.assertFalse([x for j in ("J2", "J4", "J8") for x in P.m2g_boots(j) if x["boot_id"] == "m2g_g5a_auto_N1"])
        self.assertEqual((b[0]["arm"], b[0]["config"], b[0]["config_rel"]), ("agnostic", "N1", C.AUTO_GRID_CONFIG_REL))
        self.assertTrue((ENGINE / C.AUTO_GRID_CONFIG_REL).exists())
        args = B.build_server_args(dict(b[0]), 1234, str(ENGINE / C.AUTO_GRID_CONFIG_REL))
        self.assertIn(str(ENGINE / C.AUTO_GRID_CONFIG_REL), args)
        self.assertNotIn(str(ENGINE / C.AGNOSTIC_CONFIG_REL), args)

    def test_budget_audited_model_unchanged_and_calibration_adds_time(self):                # F15
        a = P.m3_budget("J2")
        self.assertAlmostEqual(a["total_h"], 8.76 - C.EST_PREFLIGHT_S / 3600.0, 1)         # the audited per-Job number must not move
        v = P.m3_budget("J2", calib=P.CALIBS["vessl_boot"])
        self.assertAlmostEqual(v["total_h"] - a["total_h"], a["boots"] * C.EST_VESSL_BOOT_EXTRA_S / 3600.0, 3)
        u = P.m3_budget("J2", calib=P.CALIBS["vessl_boot_cell"])
        self.assertGreater(u["total_h"], v["total_h"])
        s3 = P.m3_budget("J2", ratio_arm_of=P.SCENARIOS["S3 extension: agnostic 1.48"])
        self.assertGreater(s3["boots"], a["boots"])                                          # ONE arm above 1.42 extends all four arms
        self.assertEqual(s3["boots"] - a["boots"], 2 * 4)                                    # 2 configs x 4 arms; the anchors do not extend

    def test_forecast_domain(self):
        rows = {(r["scenario"], r["calib"]): r for r in P.m3_forecast_table("J2")}
        r = rows[("S1 lambda=prior (all)", "audited")]
        self.assertEqual((r["n_done"], r["anchor_boots_done"], r["configs_defined"]), (18, 2, ["DP2", "TP2"]))
        r = rows[("S3 extension: agnostic 1.48", "audited")]
        self.assertEqual(r["configs_defined"], ["DP2"])                                      # the SECOND config in the registered order (TP) is cut first (MGP-49')
        self.assertTrue(any(k.startswith("TP2") for k in r["cut"]))
        self.assertEqual(r["anchor_boots_done"], 2)                                          # anchors run first and survive the cut
        self.assertGreater(r["total_h_uncapped"], 14.0)
        j1 = {(r["scenario"], r["calib"]): r for r in P.m3_forecast_table("J1")}[("S1 lambda=prior (all)", "audited")]
        self.assertEqual((j1["anchor_boots"], j1["configs_defined"]), (0, ["N1"]))

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

    def run_job(self, job="J2", **env):
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
        ctx = D.Ctx(out, False, time.time() + 4 * 3600, time.time(), "m2g", job)
        ctx.g3_python = str(root / "fakepy.sh")                  # `fakepy.sh -m torch.distributed.run ...` -> fake_torchrun.py
        ctx.g5a_argv = [sys.executable, "-c", "print('g5a ok')"]
        with mock.patch.dict(os.environ, e), mock.patch.object(D, "RUNNER", str(root / "fake_runner.sh")):
            table = D.run_m2g(ctx)
        return table, out, order.read_text().splitlines(), ctx

    def test_order_g3_before_every_serving_boot_and_clean_pass(self):
        table, out, order, ctx = self.run_job("J4")
        kinds = [ln.split()[0] for ln in order]
        self.assertEqual(kinds.count("G3"), 3)                                                # ONLY this Job's TP (4) x 3 combinations
        self.assertTrue(all(ln.split()[1] == "TP4" for ln in order if ln.startswith("G3")))
        self.assertEqual(max(i for i, k in enumerate(kinds) if k == "G3") + 1, 3)             # all G3 lines precede ...
        self.assertTrue(all(k == "BOOT" for k in kinds[3:]))                                  # ... every BOOT line
        self.assertEqual(sum(1 for k in kinds if k == "BOOT"), len(P.m2g_boots("J4")))
        self.assertTrue(all(table["cells"][f"TP4:{a}"]["included"] for a in C.ARMS),
                        {k: v["reason"] for k, v in table["cells"].items() if k.startswith("TP") and not v["included"]})
        self.assertEqual(sorted({k.split(":")[0] for k in table["cells"]}), ["DP4", "N1", "TP4"])
        self.assertEqual((table["job"], table["g0ii"]), ("J4", "NOT_RUN_IN_JOB"))
        self.assertEqual(table["g5a_unit_test"], "PASS")
        self.assertEqual(table["g5a_gpu_descriptive"], "NOT_IN_THIS_JOB")                    # the GPU descriptive boot is J1's (§15-2)
        self.assertTrue((out / "M2G_PASS_TABLE_J4.json").exists() and (out / "M2G_PASS_TABLE_J4.json.sha256").exists())
        self.assertTrue((out / "gates" / "g3_launch.json").exists() and len(list((out / "gates" / "g3_logs").glob("run_g3_TP*.txt"))) == 1)
        self.assertTrue((out / "gates" / "g3_TP4_g44_64" / "launcher.json").exists())          # per-combination launcher record (gate module)
        self.assertEqual(table["contam_unverified"], [])

    def test_j1_has_no_g3_runs_the_g5a_gpu_boot_and_j2_runs_the_refusals(self):
        table, out, order, ctx = self.run_job("J1")
        self.assertFalse([ln for ln in order if ln.startswith("G3")])
        self.assertEqual(table["g5a_gpu_descriptive"], "RECORDED")                           # F4: the GPU boot ran and was read
        self.assertEqual(sorted({k.split(":")[0] for k in table["cells"]}), ["N1"])
        self.assertTrue(all(table["cells"][f"N1:{a}"]["included"] for a in C.ARMS))
        table, out, order, ctx = self.run_job("J2")
        booted = [ln.split()[1] for ln in order if ln.startswith("BOOT")]
        self.assertEqual(sum(1 for b in booted if b.startswith("m2g_refuse_")), 5)
        self.assertIn("m2g_autogrid_TP2", booted)
        self.assertEqual(table["g0ii"], "PASS")
        self.assertTrue(all(table["cells"][f"TP2:{a}"]["included"] for a in C.ARMS))

    def test_server_dead_after_the_last_step_fails_g3b_and_unknown_liveness_is_not_tested(self):    # F2
        table, out, order, ctx = self.run_job("J2", FAKE_DEAD="m2g_TP2_d44", FAKE_NO_ALIVE="m2g_TP2_agnostic")
        self.assertFalse(table["cells"]["TP2:fused"]["included"])
        self.assertIn("G3b d44 FAIL", table["cells"]["TP2:fused"]["reason"])
        self.assertIn("G1 fused<->d44 FAIL", table["cells"]["TP2:d44"]["reason"])             # a pair with a dead server is an abnormal exit
        self.assertIn("G3b agnostic NOT_TESTED", table["cells"]["TP2:fused"]["reason"])       # no record != alive
        self.assertIn("G1 fused<->agnostic NOT_TESTED", table["cells"]["TP2:agnostic"]["reason"])
        self.assertTrue(table["cells"]["N1:fused"]["included"])                               # negative control: the others are untouched

    def test_abnormal_exit_fails_g1_while_the_other_arm_of_the_config_passes(self):          # F2, G1 '비정상 종료 0'
        table, out, order, ctx = self.run_job("J2", FAKE_DEAD="m2g_DP2_agnostic")
        g1 = json.loads((out / "gates" / "g1_DP2.json").read_text())
        self.assertEqual((g1["verdicts"]["fused|agnostic"], g1["verdicts"]["fused|d44"]), ("FAIL", "PASS"))
        self.assertTrue(g1["detail"]["fused|agnostic"]["abnormal_exit"])
        self.assertIn("G1 fused<->agnostic FAIL", table["cells"]["DP2:agnostic"]["reason"])
        self.assertNotIn("G1 fused<->d44", table["cells"]["DP2:d44"]["reason"])

    def test_g1_window_exposure_is_computed_and_delivered(self):                              # F3
        table, out, order, ctx = self.run_job("J2")
        e = json.loads((out / "gates" / "g1_exposure_TP2.json").read_text())                 # the driver's own (cmd-window) cross-check
        # hand-derived from tests/test_mg_realized_mix.py rows: window [-0.5, 3.5] keeps 4 snapshots (agnostic: split 30 of 40 steps;
        # d44: 10 of 20); the WHOLE telemetry would give 0.8 / 0.6667 -- the window matters
        self.assertAlmostEqual(e["agnostic"]["exposure_steps"], 0.75, 6)
        self.assertAlmostEqual(e["d44"]["exposure_steps"], 0.5, 6)
        g1 = json.loads((out / "gates" / "g1_TP2.json").read_text())                          # what the scorer received (G1 window of serve-g1)
        self.assertAlmostEqual(g1["detail"]["fused|agnostic"]["exposure_split_steps"], 0.75, 6)
        self.assertAlmostEqual(g1["detail"]["fused|d44"]["exposure_split_steps"], 0.5, 6)
        self.assertAlmostEqual(table["g1_exposure_split_steps"]["TP2:agnostic"], 0.75, 6)
        self.assertEqual(json.loads((out / "gates" / "g1_exposure_DP2.json").read_text())["d44"]["status"], "OK")

    def test_g3_launcher_failure_is_not_tested_harness_error_not_fail(self):                  # F1
        table, out, order, ctx = self.run_job("J4", FAKE_G3_BROKEN="TP4")
        g3 = json.loads((out / "gates" / "g3_TP4.json").read_text())
        self.assertEqual(g3["verdict"], "NOT_TESTED")                                         # lesson 21: not FAIL
        self.assertIn("HARNESS_ERROR", json.dumps(g3))
        self.assertIn("G3 comm stress NOT_TESTED", table["cells"]["TP4:fused"]["reason"])
        self.assertTrue(table["cells"]["N1:fused"]["included"])
        self.assertTrue((out / "gates" / "g3_logs" / "run_g3_TP4.txt").exists())

    def test_r1_g3_not_pass_tp_serving_boots_are_not_attempted_and_pass_tp_runs(self):         # R1
        table, out, order, ctx = self.run_job("J4", FAKE_G3_BROKEN="TP4")
        booted = [ln.split()[1] for ln in order if ln.split()[0] == "BOOT"]
        self.assertFalse([b for b in booted if b.startswith("m2g_TP4_") or b.startswith("m2g_g4_TP4_")], booted)
        skipped = {e["boot_id"] for e in ctx.not_attempted if e.get("reason") == "NOT_ATTEMPTED_G3_EXCLUDED"}
        want = {b["boot_id"] for b in P.m2g_boots("J4") if b["boot_id"].startswith(("m2g_TP4_", "m2g_g4_TP4_"))}
        self.assertTrue(want and skipped == want, (want, skipped))
        self.assertFalse((out / "boots" / "m2g_TP4_fused").exists())
        self.assertIn("m2g_DP4_fused", booted)                                              # DP has no G3 dependence
        self.assertIn("m2g_N1_fused", booted)
        self.assertFalse(table["cells"]["TP4:fused"]["included"])
        table, out, order, ctx = self.run_job("J4")                                          # negative control: G3 PASS -> TP4 boots run
        self.assertIn("m2g_TP4_fused", [ln.split()[1] for ln in order if ln.split()[0] == "BOOT"])

    def test_r1_mutation_without_the_skip_boots_the_excluded_tp(self):                       # R1 mutation: KILLED
        from unittest import mock
        with mock.patch.object(D, "g3_excluded_boot", lambda bid, ex: False):
            table, out, order, ctx = self.run_job("J4", FAKE_G3_BROKEN="TP4")
        booted = [ln.split()[1] for ln in order if ln.split()[0] == "BOOT"]
        self.assertTrue(any(b.startswith("m2g_TP4_") for b in booted))                       # the R1 test above would fail -> KILLED

    def test_r1_rescore_keeps_preflight_and_not_attempted(self):                              # recommendation: merge
        table, out, order, ctx = self.run_job("J2")
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
            table, out, order, ctx = self.run_job("J2")
        rec = json.loads((out / "gates" / "nvml_preboot.json").read_text())
        self.assertTrue(rec["checked"] and rec["not_empty"] == [] and rec["attempts"] == 1)

    def test_no_interpreter_for_the_launcher_leaves_every_tp_not_tested(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = D.Ctx(Path(td), False, time.time() + 4 * 3600, time.time(), "m2g", "J4")
            ctx.g3_python = "/nonexistent/python"
            s = D.run_g3(ctx)
            self.assertEqual(set(s), {"TP4"})                                                # only the Job's own TP
            self.assertEqual(json.loads((Path(td) / "gates" / "g3_TP4.json").read_text())["verdict"], "NOT_TESTED")
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(D.run_g3(D.Ctx(Path(td), False, time.time() + 4 * 3600, time.time(), "m2g", "J1")), {})

    def test_nvml_unavailable_cell_is_listed_and_not_included(self):                          # F5
        table, out, order, ctx = self.run_job("J2")
        # re-assemble after rewriting one pre-boot record as 'NVML unavailable'
        p = out / "boots" / "m2g_TP2_agnostic" / "contam_pre.json"
        p.write_text(json.dumps({"snapshot": {"available": False, "error": "x"}, "verdict": {"contam_unverified": True, "verified": False}}))
        t2 = D.assemble_pass_table(out, "J2")
        self.assertEqual(t2["contam_unverified"], ["TP2:agnostic"])
        self.assertFalse(t2["cells"]["TP2:agnostic"]["included"])
        self.assertIn("CONTAM_UNVERIFIED", t2["cells"]["TP2:agnostic"]["reason"])
        self.assertTrue(table["cells"]["TP2:agnostic"]["included"])                              # before the rewrite it was included


class M3ExtensionDriver(unittest.TestCase):                                                  # F10 driver half
    def test_incomplete_extension_blocks_boot2_and_labels_the_arm(self):
        with tempfile.TemporaryDirectory() as td:
            drv = DriverWithFakeRunner()
            p, out = drv.run_driver(td, 1.7, 1.0, job="J2", fake_skip="m3_TP2_agnostic_boot1ext")
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            doc = json.loads((out / "LABELS.json").read_text())
            dec = doc["driver"]["decisions"]["TP2"]
            self.assertFalse(dec["ext_complete"])
            self.assertIn("agnostic", dec["ext_incomplete_reason"])
            self.assertEqual(list((out / "specs").glob("m3_TP2_*_boot2.json")), [])             # boot2 never entered
            self.assertEqual(list((out / "cells").glob("m3_TP2_*_boot2__*")), [])               # no TP2 LO cell was run
            self.assertTrue(list((out / "cells").glob("m3_DP2_*_boot2__*")))                    # negative control: the other config still ran LO
            reasons = [x["reason"] for x in doc["driver"]["not_attempted"] if str(x.get("boot_id", "")).startswith("m3_TP2_")]
            self.assertEqual(reasons, ["NOT_ATTEMPTED_EXT_INCOMPLETE"] * 4)
            lab = doc["keys"]["TP2|agnostic|A|r1"]
            self.assertEqual(lab["hi_label"], "HI_NOT_JUDGED")                                  # first-pass cells exist, extension does not
            self.assertIn("HI_NOT_JUDGED(EXT_NOT_RUN)", lab["reasons"])
            self.assertFalse(doc["estimands"]["G_sym"]["TP2"]["A"]["defined"])
            self.assertTrue(doc["estimands"]["G_sym"]["DP2"]["A"]["defined"])


class JobScriptsDryRun(unittest.TestCase):
    def run_job(self, name, job="J2", **extra):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_MG_DRYRUN="1", PDMUX_MG_TEST_OUT_ROOT=td, PDMUX_MG_JOB=job,
                       PDMUX_JOB_NAME="unit", PDMUX_DATA=str(Path(td) / "data"))
            env.update(extra)
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu" / name)], env=env, capture_output=True, text=True)
            out = Path(td) / f"{name.split('_')[0]}_unit"
            files = {f.name for f in out.iterdir()} if out.exists() else set()
            boots = sorted(x.name for x in (out / "boots").glob("*")) if (out / "boots").exists() else []
            notes = (out / "DRYRUN_NOTES.txt").read_text() if (out / "DRYRUN_NOTES.txt").exists() else ""
            order = (out / "CONFIG_ORDER.txt").read_text() if (out / "CONFIG_ORDER.txt").exists() else ""
            budget = (out / "BUDGET.txt").read_text() if (out / "BUDGET.txt").exists() else ""
            return p, files, boots, notes, order, budget

    def test_m3_dry_run_every_job(self):
        want = {"J1": 8, "J2": 18, "J4": 18, "J8": 26}
        caps = {"J1": 36000, "J2": 50400, "J4": 50400, "J8": 50400}
        gpus = {"J1": 2, "J2": 2, "J4": 4, "J8": 8}
        for job, nboots in want.items():
            p, files, boots, notes, order, budget = self.run_job("m3_job.sh", job)
            self.assertEqual(p.returncode, 0, (job, p.stderr[-1500:] + p.stdout[-800:]))
            self.assertEqual(len(boots), nboots, job)
            for a in ("ABORT_SUBSTRATE", "ABORT_MODEL_REVISION", "ABORT_ENGINE_DIGEST"):
                self.assertIn(a, notes)                              # would abort a real run on this machine
            self.assertIn("DRYRUN", files)
            self.assertIn("BUDGET_DRYRUN.txt", files)
            self.assertIn("registered J2=J4=J8=[DP,TP]", order)       # H-2: the dry-run prints the order and asserts it
            self.assertIn("N1" if job == "J1" else f"DP{job[1:]} TP{job[1:]}", order)
            self.assertIn(f"job={job}", budget)
            self.assertIn(f"hard_cap_s={caps[job]}", budget)
            self.assertIn(f"expected_gpus={gpus[job]}", budget)

    def test_m2g_dry_run_every_job(self):
        for job in C.JOBS:
            p, files, boots, notes, order, budget = self.run_job("m2g_job.sh", job)
            self.assertEqual(p.returncode, 0, (job, p.stderr[-1500:] + p.stdout[-800:]))
            self.assertEqual(len(boots), len(P.m2g_boots(job)), job)
            self.assertIn(f"M2G_PASS_TABLE_{job}.json", files)
            self.assertIn("hard_cap_s=14400", budget)

    def test_job_selection_is_mandatory_and_spec_must_match(self):
        p, *_ = self.run_job("m3_job.sh", job="")
        self.assertEqual(p.returncode, 16, p.stdout[-400:])                                  # no PDMUX_MG_JOB -> ABORT_ENV
        p, *_ = self.run_job("m3_job.sh", job="J9")
        self.assertEqual(p.returncode, 16)
        with tempfile.TemporaryDirectory() as td:                                            # real run: J4 submitted on the x2 spec -> ABORT_ENV
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_JOB_NAME="unitspec", PDMUX_DATA=str(Path(td) / "data"), PDMUX_MG_JOB="J4",
                       PDMUX_SPEC="resourcespec-a100x2")
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu/m3_job.sh")], env=env, capture_output=True, text=True, cwd=td)
            out = ENGINE / "results/multigpu/m3_unitspec"
            try:
                self.assertEqual(p.returncode, 16, p.stdout[-600:])
                self.assertIn("wants spec resourcespec-a100x4", p.stdout)
            finally:
                import shutil
                shutil.rmtree(out, ignore_errors=True)

    def test_real_run_aborts_without_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_JOB_NAME="unitreal", PDMUX_DATA=str(Path(td) / "data"), PDMUX_MG_JOB="J2")
            p = subprocess.run(["bash", str(ENGINE / "results/multigpu/m3_job.sh")], env=env, capture_output=True, text=True,
                               cwd=td)
            self.assertEqual(p.returncode, 11, p.stdout[-600:])                       # ABORT_SUBSTRATE (no 2x A100 here)
            out = ENGINE / "results/multigpu/m3_unitreal"
            try:
                self.assertIn("ABORT_SUBSTRATE", (out / "ABORT").read_text())
            finally:
                import shutil
                shutil.rmtree(out, ignore_errors=True)

    def test_jit_all_reduce_aborts_comm_env(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PDMUX_", "SGLANG_", "NCCL_"))}
            env.update(PDMUX_PROJECT_ROOT=str(REPO), PDMUX_JOB_NAME="unitjit", PDMUX_DATA=str(Path(td) / "data"), PDMUX_MG_JOB="J2",
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
