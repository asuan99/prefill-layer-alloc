"""CPU tests of the stock-baseline harness (results/stock_baseline/sb_*, PREREG_STOCK_BASE rev0 DRAFT).  No GPU, no server.
SB_DIR (env) points the tests at a copy of the harness -- sb_mutants.py uses it: every registered rule must make a test FAIL when it is
reverted (lesson 53)."""
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(os.path.abspath(__file__)).parents[1]
SB = Path(os.environ.get("SB_DIR", ENG / "results/stock_baseline"))
sys.path.insert(0, str(SB))
import sb_spec as SP  # noqa: E402
import sb_runner as RU  # noqa: E402
import sb_analyze as AN  # noqa: E402
import sb_loadgen as LG  # noqa: E402

W1_POOL = [[100 + i, 20 + (i % 7)] for i in range(400)]
W3_POOL = [[i, 6000 + i, 300] for i in range(50)]
ADD = {"lambda_star_burst": {a: 6.0 for a in SP.LADDER_ARMS}, "lambda_star_ref": 5.0, "refcell_rate": 6.0}


def pristine_stamp(argv=()):
    return {"engine_tree": "pristine", "engine_tree_manifest_sha256": SP.PRISTINE_REF_SHA256, "engine_tree_ref_sha256": SP.PRISTINE_REF_SHA256,
            "engine_tree_ref_overridden": False, "engine_tree_markers_found": [], "argv": list(argv)}


def patched_stamp(sha="p" * 64):
    return {"engine_tree": "patched", "engine_tree_manifest_sha256": sha, "engine_tree_markers_found": list(SP.REQUIRED_PATCHED_MARKERS) + [r"\bPDMUX_[A-Z]"],
            "argv": []}


class FakeServer:
    """Writes the engine-tree stamp engine_tree.py would write; replays are answered by the fake replay below."""
    def __init__(self, bdir, a, stamp_override=None, die=False):
        self.bdir, self.a, self.die, self.so = Path(bdir), a, die, stamp_override
        self.log = self.bdir / "server.log"
        self.cmd = ["fake"]

    def start(self):
        self.port = 1
        self.log.write_text("server_args=ServerArgs(attention_backend='flashinfer', mem_fraction_static=0.83, chunked_prefill_size=8192)\n")
        if self.a["engine"] == "sglang":
            st = self.so or (pristine_stamp() if self.a["tree"] == "pristine" else patched_stamp())
            (self.bdir / f"tree_{self.a['name']}.json").write_text(json.dumps(st))
        return "died" if self.die else "ok"

    @property
    def base_url(self):
        return f"fake://{self.a['name']}"

    def stop(self):
        return {"returncode": 0}


def make_replay(ttft_of=None, itl=20.0, dur_scale=1.0, mixed_log=None, srv_log=None):
    """Fake replay: every request succeeds with its exact length; TTFT from ttft_of(arm, req, rate_hint)."""
    def replay(api, base_url, reqs, out_path):
        arm = base_url.split("://")[1]
        recs = []
        t_end = 0.0
        for q in reqs:
            n = int(q["output_tokens"])
            tt = (ttft_of or (lambda a, r: 200.0))(arm, q)
            sent = float(q["arrival_s"]) - float(reqs[0]["arrival_s"])
            comp = sent + tt / 1000.0 + itl * (n - 1) / 1000.0
            t_end = max(t_end, comp)
            recs.append({"request_id": q["request_id"], "phase": q["phase"], "input_tokens": q["input_tokens"],
                         "requested_output_tokens": n, "observed_output_tokens": n, "ttft_ms": tt, "token_itl_ms": [itl] * (n - 1),
                         "error": "", "error_kind": "", "scheduled_arrival_s": sent, "sent_s": sent, "completion_s": comp,
                         "t_sent_mono": 1000.0 + sent, "t_finished_mono": 1000.0 + comp, "token_t_mono": [1000.0 + sent + tt / 1000.0]})
        if mixed_log is not None and srv_log is not None:
            with open(srv_log[0], "a") as f:
                f.write(mixed_log)
        return (t_end + 0.5) * dur_scale, recs
    return replay


def fake_metrics():
    st = {"n": 0}

    def m(url):
        st["n"] += 1
        k = st["n"] // 2
        return f"sglang:time_to_first_token_seconds_sum {k * 0.198}\nsglang:time_to_first_token_seconds_count {k}\n"
    return m


def run_fake_job(td, job, add=None, ttft_of=None, server_kw=None):
    plan = SP.make_plan(job, add)
    out = Path(td) / job
    res = RU.run_job(plan, out, lambda bdir, a: FakeServer(bdir, a, **(server_kw or {})), make_replay(ttft_of), W1_POOL, W3_POOL,
                     metrics_fn=fake_metrics())
    return plan, out, res


# ============================================================================================================== spec
class SpecTests(unittest.TestCase):
    def test_arm_set_and_trees(self):
        self.assertEqual(SP.ARM_NAMES, ("SG-stock", "SG-rep", "SG-mix", "SG-off", "VL-stock", "PM-ref", "PM-ref34"))
        for n in ("SG-stock", "SG-rep", "SG-mix"):
            a = SP.arm(n)
            self.assertEqual(a["tree"], "pristine")
            self.assertEqual(a["env"], {})
            self.assertNotIn("--enable-pdmux", a["cmd"])
        for n in ("SG-off", "PM-ref", "PM-ref34"):
            self.assertEqual(SP.arm(n)["tree"], "patched")

    def test_stock_argv_defaults(self):
        st, mix, off = SP.arm("SG-stock")["cmd"], SP.arm("SG-mix")["cmd"], SP.arm("SG-off")["cmd"]
        self.assertEqual(off, st, "SG-off must differ from SG-stock only by the tree")
        self.assertEqual(mix, st + ["--enable-mixed-chunk"])
        for forbidden in ("--chunked-prefill-size", "--disable-overlap-schedule", "--enable-mixed-chunk", "--mem-fraction-static"):
            self.assertNotIn(forbidden, st)
        for x in ("--disable-radix-cache", "--disable-piecewise-cuda-graph"):
            self.assertIn(x, st)
        self.assertEqual(st[st.index("--max-running-requests") + 1], "48")
        self.assertEqual(st[st.index("--context-length") + 1], "20480")
        self.assertEqual(st[st.index("--max-prefill-tokens") + 1], "16384")

    def test_pm_argv_is_policy_harness_bridge(self):
        import ph_runner as PR
        import ph_spec as P
        for name, D in (("PM-ref", 44), ("PM-ref34", 34)):
            a = RU.resolve(SP.arm(name), "/x")
            self.assertEqual(a["cmd"][2:], PR.server_argv(P.static_arm(D)))
            self.assertEqual(a["env"]["PDMUX_R2_FIXED_DSM"], str(D))
            self.assertFalse(SP.arm(name)["sticky"])

    def test_vllm_arm(self):
        a = SP.arm("VL-stock")
        self.assertEqual(a["image_key"], "PDMUX_VESSL_IMAGE_VLLM")
        self.assertEqual(a["cmd"][:2], ["vllm", "serve"])
        self.assertEqual(a["cmd"][a["cmd"].index("--max-num-seqs") + 1], "48")
        self.assertIn("--no-enable-prefix-caching", a["cmd"])
        self.assertEqual(SP.VLLM_PLACEMENT, "PENDING_USER_DECISION")
        with self.assertRaises(RuntimeError):
            RU.server_cmd(a, "/x", 1)

    def test_ladder_grid(self):
        r = SP.ladder_rates()
        self.assertEqual(len(r), 10)
        for k, x in enumerate(r):
            self.assertAlmostEqual(x, 5.516 * 1.15 ** k, places=4)
        self.assertAlmostEqual(SP.ladder_rates(ext=True)[0], 5.516 * 1.15 ** 10, places=4)

    def test_J1_plan(self):
        p = SP.make_plan("J1")
        self.assertEqual(p["boots"][0]["role"], "smoke")
        lad = [b for b in p["boots"] if b["role"] == "meas"]
        self.assertEqual(sorted(b["arm"] for b in lad), sorted(["SG-stock", "SG-mix", "SG-off", "PM-ref"]))
        self.assertIn("VL-stock", p["pending"])
        for b in lad:
            self.assertEqual(len(b["runs"]), 20)
            self.assertEqual({r["seed"] for r in b["runs"]}, {201, 202})
            self.assertTrue(all(r["n_requests"] == 120 for r in b["runs"]))
        self.assertEqual(SP.plan_problems(p), [])

    def test_J1b_plan(self):
        p = SP.make_plan("J1b", ADD)
        self.assertEqual([b["role"] for b in (p["boots"][0], p["boots"][-1])], ["refcell", "refcell"])
        for b in p["boots"][1:-1]:
            self.assertEqual(sorted(r["factor"] for r in b["runs"]), [0.8, 1.0])
            self.assertEqual(b["extra_run"]["factor"], 0.5)
            for r in b["runs"]:
                self.assertEqual(r["n_requests"], int(round(r["rate"] * 600)))
        with self.assertRaises(RuntimeError):
            SP.make_plan("J1b", None)

    def test_J2_abba_and_cells(self):
        p = SP.make_plan("J2", ADD)
        meas = [b for b in p["boots"] if b["role"] == "meas"]
        r1 = [b["arm"] for b in meas if b["abba_round"] == 1]
        r2 = [b["arm"] for b in meas if b["abba_round"] == 2]
        self.assertEqual(r2, list(reversed(r1)))
        self.assertEqual(sorted(r1), sorted(SP.SG_ARMS))
        h1, h2 = p["seed_halves"]
        for b in meas:
            seeds = {r["seed"] for r in b["runs"]}
            self.assertEqual(seeds, set(h1 if b["abba_round"] == 1 else h2))
        self.assertAlmostEqual(p["rates"]["R1"], 2.5)
        self.assertAlmostEqual(p["rates"]["R2"], 4.0)
        roles = [b["role"] for b in p["boots"]]
        self.assertEqual(roles[0], "refcell")
        self.assertEqual(roles[-1], "refcell")
        self.assertTrue(all(i > max(k for k, r in enumerate(roles) if r == "meas") for i, r in enumerate(roles) if r == "instr"))
        sg = [b for b in p["boots"] if b["boot_id"].endswith("instr_SG")][0]
        self.assertEqual(sg["arm_def"]["tree"], "patched")
        self.assertEqual(sg["arm_def"]["env"]["PDMUX_PROBE_EXTEND_TIME"], "1")
        self.assertEqual(sg["runs"][0]["rate"], p["rates"]["R2"])
        self.assertEqual(SP.plan_problems(p), [])

    def test_J3_plan(self):
        p = SP.make_plan("J3", ADD)
        tr = [r for b in p["boots"] for r in b["runs"] if r["kind"] == "trace"]
        self.assertTrue(all(abs(r["rate_lo"] - 1.0) < 1e-9 and abs(r["rate_hi"] - 4.0) < 1e-9 for r in tr))
        self.assertTrue(p["boots"][0]["floor"])
        r3 = [b for b in p["boots"] if b.get("stage_gated") == "R3"]
        self.assertEqual(sorted({b["arm"] for b in r3}), sorted(a for a in SP.W3_ARMS if a != "VL-stock"))
        lad = [b for b in p["boots"] if b["role"] == "w3ladder"][0]
        self.assertEqual(len(lad["runs"]), 12)
        self.assertEqual(SP.plan_problems(p), [])

    def test_plan_problems_catch_bad_boots(self):
        p = SP.make_plan("J2", ADD)
        b = [x for x in p["boots"] if x["arm"] == "SG-stock" and x["role"] == "meas"][0]
        b["arm_def"]["env"] = {"PDMUX_R2_POLICY": "fixed"}
        self.assertTrue(any("pristine boot" in x for x in SP.plan_problems(p)))
        p = SP.make_plan("J2", ADD)
        b = [x for x in p["boots"] if x["arm"] == "SG-off" and x["role"] == "meas"][0]
        b["arm_def"]["env"] = {"PDMUX_DECODE_PROBE_PATH": "/x"}
        self.assertTrue(any("instrumentation env" in x for x in SP.plan_problems(p)))
        p = SP.make_plan("J2", ADD)
        p["boots"] = [x for x in p["boots"] if x["arm"] != "SG-rep"]
        self.assertTrue(any("placebo" in x for x in SP.plan_problems(p)))

    def test_plan_is_path_independent(self):
        txt = SP.stable(SP.make_plan("J2", ADD))
        self.assertNotIn(str(SP.ENG), txt)

    def test_time_model_reports_J3_over_4h(self):
        self.assertGreater(SP.make_plan("J3", ADD)["est_min"], 240)


# ============================================================================================================== runner
class RunnerEnvTests(unittest.TestCase):
    def test_pristine_env_has_no_engine_knob(self):
        base = {"PDMUX_R2_POLICY": "fixed", "PDMUX_JOB_NAME": "j", "PDMUX_TELEMETRY_PATH": "/t", "PATH": "/bin"}
        env = RU.server_env(SP.arm("SG-stock"), "/b", base)
        self.assertEqual(sorted(k for k in env if k.startswith("PDMUX_")), ["PDMUX_ENGINE_TREE", "PDMUX_JOB_NAME"])
        self.assertEqual(env["PDMUX_ENGINE_TREE"], "pristine")
        off = RU.server_env(SP.arm("SG-off"), "/b", base)
        self.assertNotIn("PDMUX_R2_POLICY", off)
        self.assertEqual(off["PDMUX_ENGINE_TREE"], "patched")
        pm = RU.server_env(RU.resolve(SP.arm("PM-ref"), "/b"), "/b", base)
        self.assertEqual(pm["PDMUX_R2_FIXED_DSM"], "44")

    def test_sglang_cmd_goes_through_engine_tree(self):
        cmd = RU.server_cmd(RU.resolve(SP.arm("SG-stock"), "/b"), "/b", 1234, py="py")
        self.assertEqual(cmd[1:4], [str(SP.ENGINE_TREE_PY), "boot", "--stamp"])
        self.assertEqual(cmd[4], "/b/tree_SG-stock.json")
        i = cmd.index("--")
        self.assertEqual(cmd[i + 1:i + 4], ["py", "-m", "sglang.launch_server"])
        self.assertEqual(cmd[-2:], ["--port", "1234"])

    def test_lambda_pick_tie_is_conservative(self):
        pts = [(5.0, 1.0), (5.0, 0.97), (6.0, 0.95), (6.0, 0.85), (7.0, 0.95), (7.0, 0.95), (8.0, 0.1), (8.0, 0.1)]
        self.assertEqual(RU.lambda_pick(pts), (5.0, "BRACKETED"))
        self.assertEqual(RU.lambda_pick([(5.0, 0.95), (6.0, 0.95)])[1], "LADDER_TOP_PASS")
        self.assertEqual(RU.lambda_pick([(5.0, 0.5), (6.0, 0.95)]), (None, "LADDER_BOTTOM_FAIL"))

    def test_mixed_proxy(self):
        log = "Prefill batch, #new-seq: 1, #new-token: 5, token usage: 0.1, #running-req: 3, #queue-req: 0\n" \
              "Prefill batch, #new-seq: 1, #new-token: 5, token usage: 0.1, #running-req: 0, #queue-req: 0\n"
        self.assertEqual(RU.mixed_forward_proxy(log), 1)

    def test_sustain_eval_rule(self):
        def rec(rate, ttft_first, ttft_last, drop=0.0):
            n = int(rate * 600)
            reqs = []
            for i in range(n):
                t = i / rate
                tt = ttft_first if t < 120 else ttft_last if t >= 480 else ttft_first
                lost = drop and (i % int(1 / drop) == 0)
                reqs.append({"scheduled_arrival_s": t, "sent_s": t, "ttft_ms": tt, "completion_s": None if lost else t + 1,
                             "error": "x" if lost else "", "token_itl_ms": [10.0]})
            return {"spec": {"rate": rate}, "segments": [{"requests": reqs}]}
        self.assertTrue(RU.sustain_eval(rec(2.0, 100.0, 150.0))["pass"])
        self.assertFalse(RU.sustain_eval(rec(2.0, 100.0, 250.0))["pass"])
        self.assertFalse(RU.sustain_eval(rec(2.0, 100.0, 100.0, drop=0.1))["pass"])


class RunnerJobTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("PDMUX_JOB_NAME", "PDMUX_GPU_UUID")}
        os.environ["PDMUX_JOB_NAME"] = "jobA"
        os.environ["PDMUX_GPU_UUID"] = "GPU-a"
        RU._GPU.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        RU._GPU.clear()

    def test_J2_records_stamps_and_windows(self):
        with tempfile.TemporaryDirectory() as td:
            plan, out, res = run_fake_job(td, "J2", ADD)
            self.assertEqual(RU.exit_code(res), 0)
            runs = [json.loads(p.read_text()) for p in (out / "runs").glob("*.json")]
            n_req = sum(len(b["runs"]) for b in plan["boots"])
            self.assertEqual(len(runs), n_req)
            for r in runs:
                self.assertEqual((r["job_name"], r["gpu_uuid"]), ("jobA", "GPU-a"))
                self.assertIn(r["engine_tree"], ("pristine", "patched"))
                self.assertLessEqual(r["t_load_start_epoch"], r["t_load_end_epoch"])
                q = r["segments"][0]["requests"][0]
                for k in ("t_sent_mono", "t_finished_mono", "token_t_mono"):
                    self.assertIn(k, q)
                self.assertTrue(r["tree_stamp"])
            boots = [json.loads(p.read_text()) for p in (out / "boots").glob("*/boot.json")]
            self.assertTrue(all(b["job_name"] == "jobA" and b["gpu_uuid"] == "GPU-a" for b in boots))

    def test_dead_boot_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            _p, _o, res = run_fake_job(td, "J1", None, server_kw={"die": True})
            self.assertEqual(RU.exit_code(res), 17)

    def test_sustain_extra_point_only_when_both_fail(self):
        with tempfile.TemporaryDirectory() as td:
            _p, out, _r = run_fake_job(td, "J1b", ADD, ttft_of=lambda a, q: 9000.0 if a == "SG-mix" else 100.0)
            ev = json.loads((out / "boots" / "J1b_sus_SG-mix" / "SUSTAIN_EVAL.json").read_text())
            ev_ok = json.loads((out / "boots" / "J1b_sus_SG-stock" / "SUSTAIN_EVAL.json").read_text())
            self.assertEqual(ev_ok["extra_needed"], False)
            names = [p.name for p in (out / "runs").glob("*SG-mix_sus*")]
            names_ok = [p.name for p in (out / "runs").glob("*SG-stock_sus*")]
            self.assertEqual(len(names_ok), 2)
            # TTFT 9000 everywhere still passes the completion/growth rule (sustain rule is not the SLO) -> no extra either;
            # force a failure through completion: rerun with growth
        with tempfile.TemporaryDirectory() as td:
            def tt(a, q):
                return 100.0 + (2000.0 * (q["arrival_s"] > 480)) if a == "SG-mix" else 100.0
            _p, out, _r = run_fake_job(td, "J1b", ADD, ttft_of=tt)
            ev = json.loads((out / "boots" / "J1b_sus_SG-mix" / "SUSTAIN_EVAL.json").read_text())
            self.assertTrue(ev["extra_needed"])
            self.assertEqual(len(list((out / "runs").glob("*SG-mix_sus*"))), 3)

    def test_J3_fills_R3_from_in_job_ladder(self):
        with tempfile.TemporaryDirectory() as td:
            def tt(a, q):
                return 9e6 if q["phase"] == "W3" and q["request_id"].split("_W3lad_r")[-1].startswith(("0.50", "0.75")) else 100.0
            plan, out, res = run_fake_job(td, "J3", ADD, ttft_of=tt)
            w3 = json.loads((out / "W3_LAMBDA.json").read_text())
            self.assertEqual(w3["status"], "BRACKETED")
            self.assertAlmostEqual(w3["lambda_star_W3"], 0.3375)
            r3 = [json.loads(p.read_text()) for p in (out / "runs").glob("*_R3_s*.json")]
            self.assertTrue(r3)
            self.assertTrue(all(abs(r["spec"]["rate"] - 0.27) < 1e-9 for r in r3))
            self.assertEqual(RU.exit_code(res), 0)

    def test_J3_unbracketed_W3_ladder_skips_R3(self):
        with tempfile.TemporaryDirectory() as td:
            plan, out, res = run_fake_job(td, "J3", ADD)                      # every W3 rung passes -> LADDER_TOP_PASS
            w3 = json.loads((out / "W3_LAMBDA.json").read_text())
            self.assertEqual(w3["status"], "LADDER_TOP_PASS")
            self.assertIsNone(w3["r3_rate"])
            self.assertEqual(list((out / "runs").glob("*_R3_s*.json")), [])
            self.assertEqual(RU.exit_code(res), 17)

    def test_resume_refused_without_flag(self):
        with tempfile.TemporaryDirectory() as td:
            plan, out, _ = run_fake_job(td, "J1", None)
            with self.assertRaises(RuntimeError):
                RU.run_job(plan, out, lambda b, a: FakeServer(b, a), make_replay(), W1_POOL, W3_POOL)


# ============================================================================================================== analysis
def mk_run(rid, arm, seed, cell, segs, job="jobA", gpu="GPU-a", tree=None, boot_id=None, th="h", **kw):
    t = tree or ("vllm" if arm == "VL-stock" else "pristine" if arm in SP.PRISTINE_ARMS else "patched")
    return {"run_id": rid, "arm": arm, "seed": seed, "cell": cell, "segments": segs, "job_name": job, "gpu_uuid": gpu,
            "engine_tree": t, "boot_id": boot_id or f"b_{arm}", "boot_role": "meas", "trace_hash": f"{th}{seed}", "spec": kw.get("spec", {}), **kw.get("extra", {})}


def seg(phase, n_good, n_bad, dur, good_ttft=500.0, bad_ttft=8000.0, err=""):
    reqs = [{"request_id": f"g{i}", "phase": phase, "ttft_ms": good_ttft, "token_itl_ms": [20.0], "error": ""} for i in range(n_good)]
    reqs += [{"request_id": f"b{i}", "phase": phase, "ttft_ms": bad_ttft, "token_itl_ms": [20.0], "error": err} for i in range(n_bad)]
    return {"phase": phase, "duration_s": dur, "requests": reqs}


def boot_rec(arm, tree_stamp=None):
    if arm == "VL-stock":
        return {"boot_id": f"b_{arm}", "engine": "vllm", "engine_tree": "vllm", "status": "ok", "tree_stamp": {}}
    t = "pristine" if arm in SP.PRISTINE_ARMS else "patched"
    return {"boot_id": f"b_{arm}", "engine": "sglang", "engine_tree": t, "status": "ok",
            "tree_stamp": tree_stamp or (pristine_stamp() if t == "pristine" else patched_stamp())}


def job_of(good_by_arm, cell="R2", jitter=None, job_override=None, stamp_override=None, phases=("W1",), refcells=None):
    runs = []
    for a, g in good_by_arm.items():
        for s in SP.EVAL_SEEDS:
            j = (jitter or {}).get((a, s), 0)
            segs = [seg(ph, g + j, 100 - g - j, 100.0) for ph in phases]
            runs.append(mk_run(f"{a}_{s}", a, s, cell, segs, job=(job_override or {}).get((a, s), "jobA")))
    runs += refcells or []
    boots = {f"b_{a}": boot_rec(a, (stamp_override or {}).get(a)) for a in good_by_arm}
    return {"runs": runs, "boots": boots, "plan": {"pending": []}}


SMOKE_OK = {"status": "CLIENT_DEF_OK"}
EQ = {"SG-stock": 80, "SG-rep": 80, "SG-mix": 80, "SG-off": 80, "PM-ref": 80, "PM-ref34": 80, "VL-stock": 80}
JIT = {(a, s): d for a in EQ for s, d in zip(SP.EVAL_SEEDS, (0, 1, -1, 0))}


class AnalyzeTests(unittest.TestCase):
    def test_goodput_sums_durations(self):
        r = {"run_id": "x", "segments": [seg("LO", 50, 0, 100.0), seg("HI", 50, 0, 50.0)]}
        self.assertAlmostEqual(AN.goodput(r, 3000.0), 100 / 150.0)

    def test_label_rules(self):
        t = ("PM", "SG")
        self.assertEqual(AN.label((0.05, 0.035, 0.07, 4), t), "PM_ABOVE_SG")
        self.assertEqual(AN.label((0.02, 0.01, 0.03, 4), t), "UNRESOLVED_INTERVAL")      # CI > 0 but mean < ln1.03
        self.assertEqual(AN.label((-0.05, -0.07, -0.035, 4), t), "PM_BELOW_SG")
        self.assertEqual(AN.label((0.0, -0.01, 0.01, 4), t), "EQUIV_WITHIN_3PCT")
        self.assertEqual(AN.label((0.0, -0.04, 0.01, 4), t), "UNRESOLVED_INTERVAL")
        self.assertEqual(AN.label((0.05, 0.04, 0.07, 3), t), "UNRESOLVED_INTERVAL")

    def test_placebo_k3(self):
        ok = {"n": 4, "mean": 0.005, "lo": -0.004, "hi": 0.014}
        self.assertFalse(AN.placebo_invalid(ok))
        self.assertTrue(AN.placebo_invalid(dict(ok, n=3)))
        self.assertTrue(AN.placebo_invalid(dict(ok, mean=0.031, lo=-0.01, hi=0.07)))
        self.assertTrue(AN.placebo_invalid(dict(ok, mean=0.01, lo=0.002, hi=0.018)))
        self.assertTrue(AN.placebo_invalid(dict(ok, mean=-0.01, lo=-0.018, hi=-0.002)))
        self.assertTrue(AN.placebo_invalid(None))

    def _pair(self, job, key, smoke=SMOKE_OK, cell="R2"):
        return AN.analyze_cell(job, cell, None, smoke, 3000.0)["pairs"][key]

    def test_equiv_cell_valid(self):
        p = self._pair(job_of(EQ, jitter=JIT), "Q8:SG-off_vs_SG-stock")
        self.assertEqual(p["label"], "EQUIV_WITHIN_3PCT", p)
        self.assertEqual(p["validity"], [])

    def test_above(self):
        g = dict(EQ, **{"PM-ref": 92})
        self.assertEqual(self._pair(job_of(g, jitter=JIT), "Q2:PM-ref_vs_SG-stock")["label"], "PM_ABOVE_SG")

    def test_placebo_failure_is_boot_noise(self):
        g = dict(EQ, **{"SG-rep": 86})
        self.assertEqual(self._pair(job_of(g, jitter=JIT), "Q8:SG-off_vs_SG-stock")["label"], "BOOT_NOISE")

    def test_cross_job_pair_invalid(self):
        j = job_of(EQ, jitter=JIT, job_override={("SG-off", 3): "jobB"})
        p = self._pair(j, "Q8:SG-off_vs_SG-stock")
        self.assertEqual(p["label"], "INVALID")
        self.assertTrue(any(x.startswith("CROSS_JOB_OR_GPU") for x in p["validity"]))

    def test_engine_tree_mismatch_invalid(self):
        j = job_of(EQ, jitter=JIT, stamp_override={"SG-stock": patched_stamp()})
        for r in j["runs"]:
            if r["arm"] == "SG-stock":
                r["engine_tree"] = "patched"
        p = self._pair(j, "Q8:SG-off_vs_SG-stock")
        self.assertEqual(p["label"], "INVALID")
        self.assertTrue(any(x.startswith("TREE_MISMATCH") for x in p["validity"]))

    def test_pristine_manifest_checked(self):
        bad = dict(pristine_stamp(), engine_tree_manifest_sha256="0" * 64)
        p = self._pair(job_of(EQ, jitter=JIT, stamp_override={"SG-stock": bad}), "Q8:SG-off_vs_SG-stock")
        self.assertIn("PRISTINE_MANIFEST:SG-stock", p["validity"])
        bad = dict(pristine_stamp(), engine_tree_markers_found=["maybe_create_holb_probe"])
        p = self._pair(job_of(EQ, jitter=JIT, stamp_override={"SG-stock": bad}), "Q8:SG-off_vs_SG-stock")
        self.assertIn("PRISTINE_MARKERS:SG-stock", p["validity"])

    def test_patched_manifest_must_agree_in_job(self):
        j = job_of(EQ, jitter=JIT, stamp_override={"PM-ref": patched_stamp("q" * 64)})
        p = self._pair(j, "Q8:SG-off_vs_SG-stock")
        self.assertTrue(any(x.startswith("PATCHED_MANIFEST_DIFFERS") for x in p["validity"]))

    def test_trace_hash_and_failures(self):
        j = job_of(EQ, jitter=JIT)
        for r in j["runs"]:
            if r["arm"] == "SG-off" and r["seed"] == 2:
                r["trace_hash"] = "other"
        self.assertIn("TRACE_HASH_DIFFERS:s2", self._pair(j, "Q8:SG-off_vs_SG-stock")["validity"])
        j = job_of(EQ, jitter=JIT)
        j["runs"][0]["segments"][0]["requests"][0]["error"] = "boom"
        self.assertTrue(any(x.startswith("REQUEST_FAILURES") for x in self._pair(j, "Q8:SG-off_vs_SG-stock")["validity"]))

    def test_cliff(self):
        j = job_of(EQ, jitter=JIT)
        for r in j["runs"]:
            if r["arm"] == "SG-off":
                for q in r["segments"][0]["requests"][:20]:
                    q["ttft_ms"] = 2990.0
        self.assertIn("CLIFF_ADJACENT:SG-off:W1", self._pair(j, "Q8:SG-off_vs_SG-stock")["validity"])

    def test_engine_pair_needs_smoke(self):
        j = job_of(EQ, jitter=JIT)
        p = self._pair(j, "Q3:VL-stock_vs_SG-stock", smoke={"status": "CLIENT_DEF_UNCHECKED"})
        self.assertEqual(p["label"], "INVALID")
        self.assertEqual(self._pair(j, "Q3:VL-stock_vs_SG-stock")["label"], "EQUIV_WITHIN_3PCT")
        self.assertEqual(self._pair(j, "Q8:SG-off_vs_SG-stock", smoke={"status": "CLIENT_DEF_UNCHECKED"})["label"], "EQUIV_WITHIN_3PCT")

    def test_smoke_verdict(self):
        def j1(d_sg, d_vl):
            rows = lambda d: [{"client_ttft_ms": 100.0 + d, "server_ttft_ms": 100.0, "ok": True} for _ in range(20)]
            return {"runs": [{"kind": "smoke", "engine": "sglang", "rows": rows(d_sg)}, {"kind": "smoke", "engine": "vllm", "rows": rows(d_vl)}]}
        self.assertEqual(AN.smoke_verdict(j1(2.0, 4.0))["status"], "CLIENT_DEF_OK")
        self.assertEqual(AN.smoke_verdict(j1(2.0, 9.0))["status"], "CLIENT_DEF_MISMATCH")
        self.assertEqual(AN.smoke_verdict({"runs": [j1(1, 1)["runs"][0]]})["status"], "CLIENT_DEF_UNCHECKED")

    def test_out_of_band(self):
        g = dict(EQ)
        j = job_of({a: 95 for a in g}, cell="TRACE", jitter=JIT, phases=("LO", "HI"))
        c = AN.analyze_cell(j, "TRACE", None, SMOKE_OK, 3000.0)
        self.assertTrue(c["cell_out_of_band"])
        self.assertIn("CELL_OUT_OF_BAND", c["pairs"]["Q8:SG-off_vs_SG-stock"]["tags"])
        j = job_of({a: 75 for a in g}, cell="TRACE", jitter=JIT, phases=("LO", "HI"))
        self.assertFalse(AN.analyze_cell(j, "TRACE", None, SMOKE_OK, 3000.0)["cell_out_of_band"])
        j = job_of({a: 40 for a in g}, cell="TRACE", jitter=JIT, phases=("LO", "HI"))
        self.assertTrue(AN.analyze_cell(j, "TRACE", None, SMOKE_OK, 3000.0)["cell_out_of_band"])

    def test_substrate(self):
        base = mk_run("j1", "SG-stock", 201, "LADDER", [seg("W1", 95, 5, 20.0)], th="REF")
        j1 = {"runs": [base]}
        ref = lambda tag, g: mk_run(f"r{tag}", "SG-stock", 201, "REFCELL", [seg("W1", g, 100 - g, 20.0)], th="REF",
                                    spec={"tag": tag})
        self.assertEqual(AN.substrate({"runs": [ref("start", 93), ref("end", 90)]}, j1)["tag"], "SUBSTRATE_OK")
        self.assertEqual(AN.substrate({"runs": [ref("start", 93), ref("end", 84)]}, j1)["tag"], "SUBSTRATE_SHIFT")
        self.assertEqual(AN.substrate({"runs": [ref("start", 93)]}, {"runs": []})["tag"], "SUBSTRATE_UNCHECKED")

    def test_mix_not_fired(self):
        j = job_of(EQ, jitter=JIT)
        for r in j["runs"]:
            if r["arm"] == "SG-mix":
                r["mixed_forward_proxy"] = 10
        self.assertEqual(self._pair(j, "Q5:SG-mix_vs_SG-stock")["label"], "MIX_NOT_FIRED")
        for r in j["runs"]:
            if r["arm"] == "SG-mix":
                r["mixed_forward_proxy"] = 60
        self.assertEqual(self._pair(j, "Q5:SG-mix_vs_SG-stock")["label"], "EQUIV_WITHIN_3PCT")

    def test_ladder_lambda_and_ref_min(self):
        def lad(arm, rates_ok, rates_fail):
            out = []
            for r in rates_ok:
                for s in (201, 202):
                    out.append(mk_run(f"{arm}{r}{s}", arm, s, "LADDER", [seg("W1", 118, 2, 20.0)], spec={"rate": r}))
            for r in rates_fail:
                for s in (201, 202):
                    out.append(mk_run(f"{arm}{r}{s}f", arm, s, "LADDER", [seg("W1", 50, 70, 20.0)], spec={"rate": r}))
            return out
        j1 = {"runs": lad("SG-stock", [5.0, 6.0], [7.0]) + lad("PM-ref", [5.0], [6.0, 7.0])}
        def sus(arm, rate, ok):
            n = int(rate * 600)
            reqs = [{"scheduled_arrival_s": i / rate, "sent_s": i / rate, "ttft_ms": 100.0, "completion_s": i / rate + 1, "error": "",
                     "token_itl_ms": [5.0]} for i in range(n if ok else n // 2)]
            return mk_run(f"s{arm}{rate}", arm, 201, "SUSTAIN", [{"phase": "W1", "duration_s": 600.0, "requests": reqs}], spec={"rate": rate})
        j1b = {"runs": [sus("SG-stock", 6.0, False), sus("SG-stock", 4.8, True), sus("PM-ref", 5.0, True), sus("PM-ref", 4.0, True)]}
        add = AN.build_addendum(j1, None, j1b)
        self.assertEqual(add["lambda_star_burst"], {"SG-stock": 6.0, "PM-ref": 5.0})
        self.assertEqual(add["refcell_rate"], 6.0)
        self.assertEqual(add["lambda_star_ref"], 4.8)
        self.assertIn("VL-stock", add["lambda_star_ref_missing_arms"])


class InstrTests(unittest.TestCase):
    """PREREG §5-1 / §5-3 load-window filter: warm-up records outside [t_load_start, t_load_end] must not enter r_ser / r_span."""
    @staticmethod
    def probe():
        boot = {"event": "decode_probe_boot", "extend_time_enabled": True, "t_realtime": 1000.0, "t_host": 50.0}
        d = lambda seq, ms, t, bs=4, di=None: {"event": "decode_forward", "seq": seq, "mode": "decode", "bs_real": bs, "stream_key": 1,
                                               "decode_fwd_gpu_ms": ms, "t_host_start": t, "decode_iter_seq": di}
        e = lambda seq, ms, t, L=2048: {"event": "extend_forward", "seq": seq, "mode": "extend", "extend_gpu_ms": ms, "stream_key": 1,
                                        "extend_tokens": L, "n_reqs": 1, "t_host_start": t}
        recs = [boot] + [d(i, 10.0, 50.0 + i) for i in range(1, 7)] + [e(7, 500.0, 57.0), d(8, 10.0, 58.0)]
        recs += [d(i, 10.0, 150.0 + i, di=i) for i in range(9, 15)] + [e(15, 20.0, 165.0), d(16, 10.0, 166.0, di=16), e(17, 40.0, 167.0),
                                                                      d(18, 10.0, 168.0, di=18)]
        return recs

    def test_r_ser_uses_load_window_only(self):
        run = {"t_load_start_epoch": 1100.0, "t_load_end_epoch": 1200.0}
        res = AN.r_ser_from_instr(self.probe(), run)
        self.assertEqual(res["window_probe_clock"], (150.0, 250.0))
        self.assertEqual(res["n_r_ser"], 2)
        self.assertAlmostEqual(res["r_ser_p50"], 3.0)
        self.assertEqual(res["by_L"][2048]["n"], 2)

    def test_r_span_uses_load_window_only(self):
        probe = [{"event": "decode_forward", "decode_fwd_gpu_ms": 10.0, "decode_iter_seq": k, "t_host_start": t}
                 for k, t in ((1, 10.0), (2, 160.0), (3, 170.0))]
        tel = [{"event": "prefill_chunk", "span_seq": 1, "chunk_gpu_ms": 100.0, "decode_launched": True, "decode_iter_seq": 1, "t_issue_start": 10.0},
               {"event": "prefill_chunk", "span_seq": 2, "chunk_gpu_ms": 12.0, "decode_launched": True, "decode_iter_seq": 2, "t_issue_start": 160.0},
               {"event": "prefill_chunk", "span_seq": 3, "chunk_gpu_ms": 8.0, "decode_launched": True, "decode_iter_seq": 3, "t_issue_start": 170.0}]
        r = AN.pc_reduce_windowed(tel, probe, (150.0, 250.0))
        self.assertEqual(r["n_spans"], 2)
        self.assertAlmostEqual(r["r_span_p50"], 1.2)
        self.assertEqual(r["share_backlog"], 0.0)

    def test_r_span_chunk_and_decode_windows_both_apply(self):
        probe = [{"event": "decode_forward", "decode_fwd_gpu_ms": 10.0, "decode_iter_seq": k, "t_host_start": t}
                 for k, t in ((1, 10.0), (2, 160.0))]
        # chunk OUTSIDE the window joined to a decode INSIDE it (warm-up chunk paired late) -> must be dropped
        tel = [{"event": "prefill_chunk", "span_seq": 1, "chunk_gpu_ms": 100.0, "decode_launched": True, "decode_iter_seq": 2, "t_issue_start": 10.0},
               {"event": "prefill_chunk", "span_seq": 2, "chunk_gpu_ms": 12.0, "decode_launched": True, "decode_iter_seq": 2, "t_issue_start": 160.0}]
        r = AN.pc_reduce_windowed(tel, probe, (150.0, 250.0))
        self.assertEqual((r["n_spans"], r["r_span_p50"]), (1, 1.2))
        # chunk INSIDE the window joined to a decode OUTSIDE it -> must be dropped
        tel = [{"event": "prefill_chunk", "span_seq": 1, "chunk_gpu_ms": 100.0, "decode_launched": True, "decode_iter_seq": 1, "t_issue_start": 170.0},
               {"event": "prefill_chunk", "span_seq": 2, "chunk_gpu_ms": 12.0, "decode_launched": True, "decode_iter_seq": 2, "t_issue_start": 160.0}]
        r = AN.pc_reduce_windowed(tel, probe, (150.0, 250.0))
        self.assertEqual((r["n_spans"], r["r_span_p50"]), (1, 1.2))

    def test_probe_window_needs_boot_clock(self):
        with self.assertRaises(ValueError):
            AN.probe_window([{"event": "decode_probe_boot"}], 0, 1)


class ReportTests(unittest.TestCase):
    def test_not_measured_lists_pending_and_deviation(self):
        j = job_of(EQ, jitter=JIT)
        j["plan"] = {"pending": ["VL-stock"]}
        nm = AN.not_measured({"J2": j, "J3": None}, {"R2": AN.analyze_cell(j, "R2", None, {"status": "CLIENT_DEF_UNCHECKED"}, 3000.0)},
                             {"status": "CLIENT_DEF_UNCHECKED"}, {})
        txt = "\n".join(nm)
        self.assertIn("VL-stock pending", txt)
        self.assertIn("D-1", txt)
        self.assertIn("J3: Job raw tree absent", txt)
        self.assertIn("definition smoke", txt)
        self.assertIn("측정하지 않은 것", AN.render_not_measured(nm))

    def test_predictions(self):
        add = {"lambda_star_burst": {"SG-stock": 7.0, "VL-stock": 9.5}}
        cells = {"R2": {"pairs": {"Q2:PM-ref_vs_SG-stock": {"n": 4, "mean": 0.05, "lo": 0.01, "hi": 0.09, "label": "PM_ABOVE_SG"}},
                        "gate5": {"SG-mix": {"itl_ms": {"p95": 30.0}}, "SG-stock": {"itl_ms": {"p95": 40.0}}}}}
        instr = {"SG": {"by_L": {2048: {"n": 12, "r_ser_p50": 15.0, "bs_p50": 1}}}}
        P = AN.score_predictions(add, cells, instr)
        self.assertEqual(P["P-1"]["score"], "HIT")
        self.assertEqual(P["P-2"]["score"], "MISS")
        self.assertEqual(P["P-3"]["score"], "HIT(+)")
        self.assertEqual(P["P-4"]["forecast_cell"], {"L": 2048, "bs": 1, "ctx": 2048})
        self.assertEqual(P["P-4"]["score"], "HIT")
        self.assertEqual(P["P-5"]["score"], "HIT")


class LoadgenTests(unittest.TestCase):
    def test_openai_stream_counts_usage(self):
        st = LG.OpenAIStreamState()
        t = iter(range(100))
        clk = lambda: float(next(t))
        for ct in (2, 2, 5):
            st.feed_line(f'data: {{"choices":[{{"text":"x","finish_reason":null}}],"usage":{{"completion_tokens":{ct}}}}}'.encode(), clk)
        st.feed_line(b'data: {"choices":[],"usage":{"completion_tokens":5}}', clk)
        st.feed_line(b"data: [DONE]", clk)
        self.assertEqual(len(st.token_times), 5)
        self.assertEqual(st.token_times, [0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(st.classify(5, True), ("", "", ""))
        self.assertTrue(st.classify(6, True)[0])

    def test_openai_payload(self):
        p = LG.openai_payload({"input_tokens": 3, "output_tokens": 7, "ignore_eos": True}, "m")
        self.assertEqual(p["prompt"], [1, 1, 1])
        self.assertEqual(p["max_tokens"], 7)
        self.assertTrue(p["ignore_eos"])
        self.assertTrue(p["stream_options"]["continuous_usage_stats"])
        self.assertFalse(LG.openai_payload({"input_tokens": 1, "output_tokens": 1, "ignore_eos": False}, "m")["ignore_eos"])

    def test_metric_parse_and_smoke_delta(self):
        txt = '# HELP x\nvllm:time_to_first_token_seconds_sum{model_name="m"} 1.5\nvllm:time_to_first_token_seconds_count{model_name="m"} 3\n'
        self.assertEqual(LG.parse_ttft_metric(txt), (1.5, 3.0))
        state = {"s": 0.0, "c": 0}

        def metrics(url):
            return f"sglang:time_to_first_token_seconds_sum {state['s']}\nsglang:time_to_first_token_seconds_count {state['c']}\n"

        def replay(api, url, reqs, p):
            state["s"] += 0.100
            state["c"] += 1
            return 1.0, [{"ttft_ms": 103.0, "error": ""}]
        rows = LG.smoke_sequence("sglang", "u", [{"request_id": "a", "arrival_s": 3.0}, {"request_id": "b", "arrival_s": 4.0}], "/tmp",
                                 metrics_fn=metrics, replay_fn=replay, settle_s=0)
        self.assertTrue(all(r["ok"] for r in rows))
        self.assertAlmostEqual(rows[0]["client_ttft_ms"] - rows[0]["server_ttft_ms"], 3.0)

        def replay2(api, url, reqs, p):          # a foreign request landed in the same scrape interval: count delta 2
            state["s"] += 0.300
            state["c"] += 2
            return 1.0, [{"ttft_ms": 103.0, "error": ""}]
        rows = LG.smoke_sequence("sglang", "u", [{"request_id": "a", "arrival_s": 0.0}], "/tmp", metrics_fn=metrics, replay_fn=replay2,
                                 settle_s=0)
        self.assertFalse(rows[0]["ok"])
        self.assertIsNone(rows[0]["server_ttft_ms"])


class DigestAndJobShellTests(unittest.TestCase):
    def test_engine_digest_file(self):
        self.assertEqual((SB / "ENGINE_DIGEST.txt").read_text().strip(), SP.ENGINE_DIGEST_EXPECTED)
        self.assertTrue(SP.ENGINE_DIGEST_EXPECTED.startswith("2836112b"))

    def test_digest_make_covers_files(self):
        import sb_digest as DG
        d = DG.make()
        self.assertIn("results/stock_baseline/sb_analyze.py", d["files"])
        self.assertIn("src/multiplex/decode_probe.py", d["reused"])
        self.assertNotIn("results/stock_baseline/sb_mutants.py", d["files"])
        self.assertEqual(set(d["plans_sha"]), set(SP.JOBS))

    def _job(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        e.update(env)
        return subprocess.run(["bash", str(SB / "sb_job.sh")], env=e, capture_output=True, text=True, timeout=120)

    def test_job_refuses_unsealed_and_bad_job(self):
        self.assertEqual(self._job({"PDMUX_SB_JOB": "J1"}).returncode, 16)
        self.assertEqual(self._job({"PDMUX_SB_JOB": "JX", "PDMUX_SB_TEST_MODE": "1"}).returncode, 16)

    def test_job_refuses_wrong_engine_digest(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ED.txt"
            f.write_text("65cbb49d" + "0" * 56 + "\n")
            r = self._job({"PDMUX_SB_JOB": "J1", "PDMUX_SB_TEST_MODE": "1", "PDMUX_SB_ENGINE_DIGEST_FILE": str(f),
                           "PDMUX_SB_OUT": f"{td}/raw", "PDMUX_SB_SUMMARY": f"{td}/sum"})
            self.assertEqual(r.returncode, 13, r.stderr)

    def test_job_needs_addendum_for_J2(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._job({"PDMUX_SB_JOB": "J2", "PDMUX_SB_TEST_MODE": "1", "PDMUX_SB_OUT": f"{td}/raw", "PDMUX_SB_SUMMARY": f"{td}/sum"})
            self.assertEqual(r.returncode, 16, r.stderr)
            self.assertIn("ABORT_PLAN", r.stderr)


@unittest.skipIf(os.environ.get("SB_IN_MUTANT") == "1", "mutants run inside the mutant harness")
class MutantsTests(unittest.TestCase):
    def test_all_mutants_killed(self):
        if os.environ.get("SB_RUN_MUTANTS") != "1":
            self.skipTest("set SB_RUN_MUTANTS=1 (slow: one unittest run per mutant)")
        import sb_mutants as M
        self.assertEqual(M.run_all(), [])


if __name__ == "__main__":
    unittest.main()
