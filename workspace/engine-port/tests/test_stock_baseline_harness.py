"""CPU tests of the stock-baseline harness (results/stock_baseline/sb_*, PREREG_STOCK_BASE rev3, 2026-11-03).  No GPU, no server.
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
# rev3: the first Job's real addendum shape -- the SGLang-default arms have no burst lambda* (LADDER_BOTTOM_FAIL)
ADD_J1_REAL = {"lambda_star_burst": {"VL-stock": 8.389146, "PM-ref": 7.29491}, "refcell_rate": 7.29491, "refcell_arm": "PM-ref"}


def pristine_stamp(argv=()):
    return {"engine_tree": "pristine", "engine_tree_manifest_sha256": SP.PRISTINE_REF_SHA256, "engine_tree_ref_sha256": SP.PRISTINE_REF_SHA256,
            "engine_tree_ref_overridden": False, "engine_tree_markers_found": [], "argv": list(argv)}


def vllm_stamp(**kw):
    return dict({"engine_tree": "vllm", "python": SP.VLLM_PY, "vllm_venv_freeze_sha256": SP.VLLM_VENV_FREEZE_SHA256,
                 "sglang_freeze_sha256": SP.SGLANG_FREEZE_SHA256, "attention_backend": "FLASHINFER", "cudagraph_mode": "FULL_AND_PIECEWISE"}, **kw)


IMG_OK = {"sglang_ok": True, "vllm_ok": True, "sglang_freeze_sha256": SP.SGLANG_FREEZE_SHA256,
          "vllm_venv_freeze_sha256": SP.VLLM_VENV_FREEZE_SHA256, "vllm_venv_freeze_sha256_file": SP.VLLM_VENV_FREEZE_SHA256}
VL_LOG = ("INFO [cuda.py:300] Using FLASHINFER backend.\n"
          "INFO [core.py:90] compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'level': 3}\n")


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
        mix = "True" if "--enable-mixed-chunk" in self.a["cmd"] else "False"
        self.log.write_text(VL_LOG if self.a["engine"] == "vllm" else
                            "server_args=ServerArgs(attention_backend='flashinfer', mem_fraction_static=0.83, chunked_prefill_size=8192, "
                            f"enable_mixed_chunk={mix})\n"
                            "Prefill batch, #new-seq: 1, #new-token: 5, token usage: 0.1, #running-req: 3, #queue-req: 0\n")
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


def run_fake_job(td, job, add=None, ttft_of=None, server_kw=None, image_check_fn=lambda: dict(IMG_OK), clock=None):
    plan = SP.make_plan(job, add)
    out = Path(td) / job
    kw = {"clock": clock} if clock is not None else {}
    res = RU.run_job(plan, out, lambda bdir, a: FakeServer(bdir, a, **(server_kw or {})), make_replay(ttft_of), W1_POOL, W3_POOL,
                     metrics_fn=fake_metrics(), image_check_fn=image_check_fn, **kw)
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
        self.assertEqual(SP.VLLM_PLACEMENT, "same_image_venv")
        self.assertEqual(a["image_key"], "PDMUX_VESSL_IMAGE_SBASE")
        self.assertEqual(a["cmd"][:3], ["/opt/vllm-venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"])
        self.assertEqual(a["cmd"][a["cmd"].index("--model") + 1], SP.S.NANO_ID)
        self.assertEqual(a["cmd"][a["cmd"].index("--max-num-seqs") + 1], "48")
        self.assertIn("--no-enable-prefix-caching", a["cmd"])
        self.assertEqual(a["env"], {"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0"})
        self.assertEqual(RU.server_cmd(a, "/x", 7), a["cmd"] + ["--port", "7"])

    def test_one_image_for_every_arm(self):
        self.assertTrue(SP.IMAGE_SBASE_DIGEST.startswith("sha256:a5f9bf0f"))
        self.assertTrue(SP.image_is_sbase(SP.image_digests()["PDMUX_VESSL_IMAGE_SBASE"]))
        self.assertFalse(SP.image_is_sbase(SP.image_digests()["PDMUX_VESSL_IMAGE"]))
        self.assertFalse(SP.image_is_sbase(""))
        self.assertIn(SP.IMAGE_SBASE_DIGEST, SP.image_digests().get("PDMUX_VESSL_IMAGE_SBASE", ""))
        for n in SP.ARM_NAMES:
            self.assertEqual(SP.arm(n)["image_key"], "PDMUX_VESSL_IMAGE_SBASE", n)
        for j in SP.JOBS:
            p = SP.make_plan(j, ADD if j != "J1" else None)
            self.assertEqual({b["arm_def"]["image_key"] for b in p["boots"]}, {"PDMUX_VESSL_IMAGE_SBASE"})
        p = SP.make_plan("J2", ADD)
        p["boots"][1]["arm_def"]["image_key"] = "PDMUX_VESSL_IMAGE_VLLM"
        self.assertTrue(any("image key" in x for x in SP.plan_problems(p)))
        p = SP.make_plan("J2", ADD)
        b = [x for x in p["boots"] if x["arm"] == "VL-stock"][0]
        b["arm_def"]["env"] = dict(b["arm_def"]["env"], PYTHONPATH="/opt/pdmux/sglang_engine_dev/python")
        self.assertTrue(any("vLLM boot must use" in x for x in SP.plan_problems(p)))

    def test_ladder_grid(self):
        r = SP.ladder_rates()
        self.assertEqual(len(r), 10)
        for k, x in enumerate(r):
            self.assertAlmostEqual(x, 5.516 * 1.15 ** k, places=4)
        self.assertAlmostEqual(SP.ladder_rates(ext=True)[0], 5.516 * 1.15 ** 10, places=4)

    def test_J1_plan(self):
        p = SP.make_plan("J1")
        self.assertEqual(p["boots"][0]["role"], "smoke")
        self.assertEqual([b["arm"] for b in p["boots"] if b["role"] == "smoke"], ["SMOKE-SG-stock", "SMOKE-VL-stock"])
        lad = [b for b in p["boots"] if b["role"] == "meas"]
        self.assertEqual(sorted(b["arm"] for b in lad), sorted(["SG-stock", "SG-mix", "SG-off", "PM-ref", "VL-stock"]))
        self.assertEqual(p["pending"], [])
        for b in lad:
            self.assertEqual(len(b["runs"]), 20)
            self.assertEqual({r["seed"] for r in b["runs"]}, {201, 202})
            self.assertTrue(all(r["n_requests"] == 120 for r in b["runs"]))
        self.assertEqual(SP.plan_problems(p), [])

    def test_J1b_plan(self):
        p = SP.make_plan("J1b", ADD)
        self.assertEqual([b["role"] for b in (p["boots"][0], p["boots"][-1])], ["refcell", "refcell"])
        self.assertEqual(sorted(b["arm"] for b in p["boots"][1:-1]), ["PM-ref", "VL-stock"])        # rev3: REF_ARMS only
        for b in p["boots"][1:-1]:
            self.assertEqual(sorted(r["factor"] for r in b["runs"]), [0.8, 1.0])
            self.assertEqual(b["extra_run"]["factor"], 0.5)
            for r in b["runs"]:
                self.assertEqual(r["n_requests"], int(round(r["rate"] * 600)))
        with self.assertRaises(RuntimeError):
            SP.make_plan("J1b", None)

    def test_J1b_plan_from_first_job_addendum(self):
        """rev3: the second Job is plannable from the first Job's real addendum (no SGLang-default burst lambda*), sustained cells for
        VL-stock / PM-ref only at {1.0, 0.8} x their burst lambda*, refcell = PM-ref at 7.29491 (a first-Job rung), no J1-down needed."""
        self.assertEqual(SP.REF_ARMS, ("VL-stock", "PM-ref"))
        p = SP.make_plan("J1b", ADD_J1_REAL)
        self.assertEqual(SP.plan_problems(p), [])
        sus = {b["arm"]: sorted(r["rate"] for r in b["runs"]) for b in p["boots"] if b["role"] == "meas"}
        self.assertEqual(sus, {"VL-stock": [6.711317, 8.389146], "PM-ref": [5.835928, 7.29491]})
        self.assertEqual({b["arm"]: b["extra_run"]["rate"] for b in p["boots"] if b["role"] == "meas"},
                         {"VL-stock": 4.194573, "PM-ref": 3.647455})
        refs = [b for b in p["boots"] if b["role"] == "refcell"]
        self.assertEqual([(b["arm"], b["runs"][0]["rate"], b["runs"][0]["seed"], b["runs"][0]["n_requests"]) for b in refs],
                         [("PM-ref", 7.29491, 201, 120)] * 2)
        self.assertIn(7.29491, SP.ladder_rates())                                            # identical cell exists in Job 1
        bad = json.loads(json.dumps(p))
        bad["boots"][0] = SP.refcell_boot("J1b", "start", 7.29491)
        bad["boots"][0]["arm"] = "SG-stock"
        self.assertTrue(any("refcell boot arm" in x for x in SP.plan_problems(bad)))
        bad = json.loads(json.dumps(p))
        bad["boots"].insert(1, SP.boot("J1b_sus_SG-stock", SP.arm("SG-stock"), []))
        self.assertTrue(any("REF_ARMS" in x for x in SP.plan_problems(bad)))

    def test_J1_down_plan(self):
        """rev3: SG 3 arms x 5.516 / 1.15^k (k = 1..4) x seeds {201, 202} x 120 requests, one boot per arm, no smoke, no addendum."""
        p = SP.make_plan("J1-down")
        self.assertEqual(p["kind"], "LADDER_DOWN")
        self.assertEqual(p["ladder"], [4.796522, 4.170888, 3.62686, 3.153791])
        self.assertEqual(sorted(b["arm"] for b in p["boots"]), ["SG-mix", "SG-off", "SG-stock"])
        for b in p["boots"]:
            self.assertEqual(b["role"], "meas")
            self.assertEqual(sorted((r["rate"], r["seed"]) for r in b["runs"]),
                             sorted((r, s) for r in (4.796522, 4.170888, 3.62686, 3.153791) for s in (201, 202)))
            self.assertTrue(all(r["n_requests"] == 120 and r["cell"] == "LADDER_DOWN" for r in b["runs"]))
        self.assertEqual({b["arm"]: b["arm_def"]["tree"] for b in p["boots"]}, {"SG-stock": "pristine", "SG-mix": "pristine", "SG-off": "patched"})
        self.assertEqual(SP.plan_problems(p), [])
        self.assertAlmostEqual(p["est_min"], 3 * (SP.BOOT_OH_MIN + 8 * SP.CAP_CELL_MIN), places=6)
        bad = json.loads(json.dumps(p))
        bad["boots"][0]["runs"][0]["rate"] = 5.516
        self.assertTrue(any("J1-down" in x for x in SP.plan_problems(bad)))
        bad = json.loads(json.dumps(p))
        bad["boots"].append(SP.boot("J1-down_lad_VL-stock", SP.arm("VL-stock"), []))
        self.assertTrue(any("one measurement boot per" in x for x in SP.plan_problems(bad)))
        self.assertIn("J1-down", SP.JOBS)

    def test_J2_abba_and_cells(self):
        p = SP.make_plan("J2", ADD)
        meas = [b for b in p["boots"] if b["role"] == "meas"]
        r1 = [b["arm"] for b in meas if b["abba_round"] == 1]
        r2 = [b["arm"] for b in meas if b["abba_round"] == 2]
        self.assertEqual(r2, list(reversed(r1)))
        self.assertEqual(sorted(r1), sorted(SP.MEAS_ARMS))
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
        mt = [b for b in p["boots"] if b["role"] == "instr_metrics"]
        self.assertEqual(len(mt), 1)
        mt = mt[0]
        self.assertEqual((mt["arm_def"]["tree"], mt["arm_def"]["env"]), ("pristine", {}))
        self.assertEqual(mt["arm_def"]["cmd"], SP.arm("SG-stock")["cmd"] + ["--enable-metrics"])
        self.assertEqual([(r["rate"], r["seed"], r["n_requests"], r["cell"]) for r in mt["runs"]], [(p["rates"]["R1"], 301, 200, "INSTR_METRICS")])
        self.assertGreater(roles.index("instr_metrics"), max(k for k, r in enumerate(roles) if r == "meas"))
        self.assertTrue(any("INSTR-VL-stock" in x for x in p["pending"]))
        self.assertEqual(SP.plan_problems(p), [])

    def test_J3_plan(self):
        """rev1: J3 = W3 only -- refcell(+floor) -> SG-stock W3 ladder -> R3 ABBA over W3_ARMS -> refcell; no W1 change trace."""
        p = SP.make_plan("J3", ADD)
        self.assertEqual(p["kind"], "W3")
        self.assertFalse(hasattr(SP, "trace_run") or hasattr(SP, "TRACE_LO"))
        roles = [b["role"] for b in p["boots"]]
        self.assertEqual(roles, ["refcell", "w3ladder"] + ["meas"] * (2 * len(SP.W3_ARMS)) + ["refcell"])
        self.assertFalse(p["boots"][0].get("floor"))                       # rev3: refcell arm is PM-ref; floor stays on SG-stock
        self.assertEqual((p["boots"][1]["arm"], p["boots"][1]["floor"]), ("SG-stock", True))
        self.assertEqual(p["boots"][0]["arm"], SP.REFCELL_ARM)
        runs = [r for b in p["boots"] if b["role"] in ("meas", "w3ladder") for r in b["runs"]]
        self.assertTrue(runs and all(r["workload"] == "W3" for r in runs))
        self.assertEqual({r["cell"] for b in p["boots"] if b["role"] == "meas" for r in b["runs"]}, {"R3"})
        r3 = [b for b in p["boots"] if b.get("stage_gated") == "R3"]
        self.assertEqual(sorted({b["arm"] for b in r3}), sorted(SP.W3_ARMS))
        self.assertIn("VL-stock", SP.W3_ARMS)
        lad = [b for b in p["boots"] if b["role"] == "w3ladder"][0]
        self.assertEqual(len(lad["runs"]), 12)
        self.assertEqual(SP.plan_problems(p), [])
        bad = json.loads(json.dumps(p))
        bad["boots"].insert(2, SP.boot("J3_x", SP.arm("SG-stock"), [SP.cap_run("J3_x_R2", "W1", 4.0, 1, 200, kind="steady", cell="R2")]))
        self.assertTrue(any("W3 only" in x for x in SP.plan_problems(bad)))
        self.assertTrue(any("plan kind" in x for x in SP.plan_problems(dict(p, kind="TRACE_W3"))))

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

    def test_time_model_and_hard_cap(self):
        for j in SP.JOBS:
            p = SP.make_plan(j, ADD if j != "J1" else None)
            self.assertEqual(p["hard_cap_min"], int(-(-p["est_min"] * SP.HARD_CAP_FACTOR // 5) * 5))
        p = SP.make_plan("J3", ADD)
        self.assertAlmostEqual(p["est_min"], sum(SP.boot_est_min(b) for b in p["boots"]), places=6)
        self.assertLess(p["est_min"], 400)                  # rev0 (trace + W3) was ~670 min

    def test_synth_requests_and_plan(self):
        """§5-1 (2026-11-02): lead-in + 10 x 512 + 10 x 2k random-id prompts, output 64, ignore_eos, 2 s apart, seed 301, SG probe boot only."""
        self.assertEqual(SP.SYNTH_LENGTHS, SP.Q6_L_GRID)
        q = SP.synth_requests(SP.INSTR_SEED, "x")
        self.assertEqual(q, SP.synth_requests(SP.INSTR_SEED, "x"))
        self.assertNotEqual([r["input_ids"] for r in q], [r["input_ids"] for r in SP.synth_requests(SP.INSTR_SEED + 1, "x")])
        self.assertEqual(len(q), 21)
        self.assertEqual((q[0]["synth_role"], q[0]["input_tokens"]), ("lead_in", SP.SYNTH_LEAD_IN_TOKENS))
        body = q[1:]
        self.assertEqual(sorted(r["input_tokens"] for r in body), [512] * 10 + [2048] * 10)
        self.assertEqual([r["arrival_s"] for r in q], [2.0 * k for k in range(21)])
        for r in q:
            self.assertEqual((r["output_tokens"], r["ignore_eos"], r["phase"]), (64, True, SP.SYNTH_PHASE))
            self.assertEqual(len(r["input_ids"]), r["input_tokens"])
            self.assertTrue(all(SP.SYNTH_TOKEN_RANGE[0] <= t < SP.SYNTH_TOKEN_RANGE[1] for t in r["input_ids"]))
        self.assertFalse(any(abs(SP.SYNTH_LEAD_IN_TOKENS - L) <= SP.Q6_L_BAND * L for L in SP.Q6_L_GRID))
        p = SP.make_plan("J2", ADD)
        ins = {b["boot_id"]: b for b in p["boots"] if b["role"] == "instr"}
        sg, pm = ins["J2_instr_SG"]["runs"][0], ins["J2_instr_PM"]["runs"][0]
        self.assertEqual((sg["seed"], sg["n_requests"], sg["synth"]), (SP.INSTR_SEED, SP.INSTR_N, SP.synth_spec()))
        self.assertNotIn("synth", pm)
        self.assertFalse(any(r.get("synth") for b in p["boots"] if b["role"] != "instr" for r in b["runs"]))
        self.assertAlmostEqual(SP.run_est_min(sg) - SP.run_est_min(pm), SP.synth_est_min(SP.synth_spec()), places=6)
        bad = json.loads(json.dumps(p))
        m = [b for b in bad["boots"] if b["role"] == "meas"][0]
        m["runs"][0]["synth"] = SP.synth_spec()
        self.assertTrue(any("synthetic prefill window" in x for x in SP.plan_problems(bad)))


# ============================================================================================================== runner
class RunnerEnvTests(unittest.TestCase):
    def test_vllm_env(self):
        base = {"PDMUX_JOB_NAME": "j", "PDMUX_ROOT": "/r", "PDMUX_R2_POLICY": "fixed", "PYTHONPATH": "/opt/pdmux/sglang_engine_dev/python",
                "PATH": "/opt/conda/bin:/bin"}
        env = RU.server_env(SP.arm("VL-stock"), "/b", base)
        self.assertEqual([k for k in env if k.startswith("PDMUX_")], [])
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual((env["CUDA_HOME"], env["TORCH_CUDA_ARCH_LIST"]), ("/usr/local/cuda", "8.0"))
        self.assertTrue(env["PATH"].startswith("/opt/vllm-venv/bin" + os.pathsep))

    def test_image_env_check(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            opt, venv = Path(td) / "opt", Path(td) / "venv"
            opt.mkdir()
            venv.mkdir()
            (opt / "pip_freeze.txt").write_text("sglang==0.5.10\n")
            (venv / "pip_freeze.txt").write_text("vllm==0.31.0\n")
            sg = hashlib.sha256(b"sglang==0.5.10\n").hexdigest()
            vl = hashlib.sha256(b"vllm==0.31.0\n").hexdigest()
            (venv / "pip_freeze.sha256").write_text(vl + "\n")
            ic = RU.image_env_check(opt, venv)
            self.assertEqual((ic["sglang_freeze_sha256"], ic["vllm_venv_freeze_sha256"], ic["vllm_venv_freeze_sha256_file"]), (sg, vl, vl))
            self.assertFalse(ic["sglang_ok"] or ic["vllm_ok"])               # not the registered bytes
            saved = (SP.SGLANG_FREEZE_SHA256, SP.VLLM_VENV_FREEZE_SHA256)
            try:
                SP.SGLANG_FREEZE_SHA256, SP.VLLM_VENV_FREEZE_SHA256 = sg, vl
                ic = RU.image_env_check(opt, venv)
                self.assertTrue(ic["sglang_ok"] and ic["vllm_ok"])
                (venv / "pip_freeze.sha256").write_text("0" * 64 + "\n")      # recorded sha disagrees with the bytes
                self.assertFalse(RU.image_env_check(opt, venv)["vllm_ok"])
            finally:
                SP.SGLANG_FREEZE_SHA256, SP.VLLM_VENV_FREEZE_SHA256 = saved
        self.assertIsNone(RU.image_refusal(IMG_OK, "vllm"))
        self.assertIn("VLLM_VENV_FREEZE", RU.image_refusal(dict(IMG_OK, vllm_ok=False), "vllm"))
        self.assertIsNone(RU.image_refusal(dict(IMG_OK, vllm_ok=False), "sglang"))
        self.assertIn("SGLANG_FREEZE", RU.image_refusal(dict(IMG_OK, sglang_ok=False), "vllm"))
        self.assertIn("SGLANG_FREEZE", RU.image_refusal(dict(IMG_OK, sglang_ok=False), "sglang"))

    def test_vllm_log_facts(self):
        f = RU.parse_vllm_log(VL_LOG)
        self.assertEqual((f["attention_backend"], f["cudagraph_mode"]), ("FLASHINFER", "FULL_AND_PIECEWISE"))
        self.assertEqual(RU.parse_vllm_log("nothing")["attention_backend"], None)

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

    def test_mixing_opportunities_and_server_args_firing(self):
        """rev2 (audit r1 C): the Prefill-batch count is only a mixing-OPPORTUNITY count; firing = server_args."""
        log = "Prefill batch, #new-seq: 1, #new-token: 5, token usage: 0.1, #running-req: 3, #queue-req: 0\n" \
              "Prefill batch, #new-seq: 1, #new-token: 5, token usage: 0.1, #running-req: 0, #queue-req: 0\n"
        self.assertEqual(RU.mixing_opportunities(log), 1)
        self.assertFalse(hasattr(RU, "mixed_forward_proxy"))
        on = RU.parse_sglang_log("server_args=ServerArgs(chunked_prefill_size=8192, enable_mixed_chunk=True, x=1)")
        self.assertTrue(RU.mix_fired_from_server_args(on))
        self.assertFalse(RU.mix_fired_from_server_args(dict(on, enable_mixed_chunk="False")))
        self.assertFalse(RU.mix_fired_from_server_args(dict(on, chunked_prefill_size="-1")))      # chunked prefill off -> no mixing
        self.assertFalse(RU.mix_fired_from_server_args(dict(on, chunked_prefill_size="None")))
        self.assertFalse(RU.mix_fired_from_server_args({}))

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

    # CAP1-R D44 E2E quantiles (p0, p5, ..., p100) at 5.516 / 6.343 / 7.295 req/s, 720 requests, E2E = completion_s - scheduled_arrival_s
    # (results/cap_char/policy_runs/pdmux-policy-cap1r-4bef3cf-20261029003539/raw/runs/CAP1R_D44_r{5.516,6.3434,7.29491}_s*.json)
    CAP1R_D44_E2E_Q = (0.16, 0.83, 1.57, 2.37, 3.38, 4.22, 4.89, 5.65, 6.38, 7.1, 7.81, 8.35, 9.03, 9.56, 10.31, 11.24, 12.44, 13.98,
                       15.55, 18.45, 23.5)

    def healthy_run(self, rate, seed, ttft=150.0):
        """Healthy server model (audit r1 A): Poisson arrivals, n = rate x 600 (some arrive after 600 s), E2E drawn from CAP1-R D44."""
        import random
        q = self.CAP1R_D44_E2E_Q
        rng, t, reqs = random.Random(seed), 0.0, []
        for _ in range(int(round(rate * SP.SUSTAIN_S))):
            t += rng.expovariate(rate)
            x = rng.random() * 20
            i = min(19, int(x))
            e = q[i] + (q[i + 1] - q[i]) * (x - i)
            reqs.append({"scheduled_arrival_s": t, "sent_s": t, "ttft_ms": ttft, "completion_s": t + e, "error": "", "token_itl_ms": [20.0]})
        return {"spec": {"rate": rate}, "segments": [{"requests": reqs}]}

    def test_sustain_healthy_server_passes(self):
        """rev2 rule passes the healthy model at every rate/seed; the rev1 rule (completed inside the window / arrived inside it) failed
        it with probability ~0.97 (first window fill transient + completions after 600 s)."""
        for rate in (3.65, 5.84, 7.29):
            for seed in range(8):
                ev = RU.sustain_eval(self.healthy_run(rate, seed))
                self.assertTrue(ev["pass"], (rate, seed, [round(w["completion_ratio"], 3) for w in ev["windows"]]))
                self.assertEqual(len(ev["windows"]), 5)

    def test_sustain_e2e_bound_and_late_arrivals(self):
        """E2E > 60 s does not count even if the request completes and TTFT does not grow; arrivals >= 600 s are excluded."""
        r = self.healthy_run(5.0, 1)
        for q in r["segments"][0]["requests"]:
            if 240.0 <= q["scheduled_arrival_s"] < 360.0 and int(q["scheduled_arrival_s"] * 10) % 10 < 2:
                q["completion_s"] = q["scheduled_arrival_s"] + 90.0           # ~20 % of window 2 slow, TTFT unchanged
        ev = RU.sustain_eval(r)
        self.assertFalse(ev["pass"])
        self.assertLess(ev["windows"][2]["completion_ratio"], 0.95)
        self.assertLess(ev["ttft_growth"], 2.0)
        r = self.healthy_run(5.0, 2)
        late = [q for q in r["segments"][0]["requests"] if q["scheduled_arrival_s"] >= SP.SUSTAIN_S]
        for k in range(60):                                                    # client cut-off after the 600 s arrival period
            t = SP.SUSTAIN_S + 0.1 * k
            r["segments"][0]["requests"].append({"scheduled_arrival_s": t, "sent_s": t, "ttft_ms": None, "completion_s": None,
                                                 "error": "cut", "token_itl_ms": []})
        ev = RU.sustain_eval(r)
        self.assertTrue(ev["pass"], [w["completion_ratio"] for w in ev["windows"]])
        self.assertEqual(ev["n_late_excluded"], len(late) + 60)
        self.assertTrue(all(w["arrivals"] > 0 for w in ev["windows"]))

    def test_sustain_ttft_growth_rule(self):
        r = self.healthy_run(5.0, 3)
        for q in r["segments"][0]["requests"]:
            if q["scheduled_arrival_s"] >= 480.0:
                q["ttft_ms"] = 400.0                                           # 150 -> 400 ms: growth 2.67 > 2
        ev = RU.sustain_eval(r)
        self.assertFalse(ev["pass"])
        self.assertTrue(all(w["completion_ratio"] >= 0.95 for w in ev["windows"]))

    def test_sustain_denominator_is_observed_arrivals(self):
        """Healthy run whose window arrival counts swing around the nominal 240 by Poisson-like noise (+-8 %): every request completes in its
        window, so it must PASS; a nominal-rate denominator (2.0 x 120 = 240) would fail it on the 220-arrival windows (220/240 < 0.95)."""
        reqs = []
        for w, n in enumerate((240, 220, 260, 220, 260)):
            for i in range(n):
                t = w * 120.0 + i * (118.0 / n)
                reqs.append({"scheduled_arrival_s": t, "sent_s": t, "ttft_ms": 100.0, "completion_s": t + 1.0, "error": "", "token_itl_ms": [10.0]})
        ev = RU.sustain_eval({"spec": {"rate": 2.0}, "segments": [{"requests": reqs}]})
        self.assertTrue(ev["pass"], ev["windows"])
        self.assertEqual([w["arrivals"] for w in ev["windows"]], [240, 220, 260, 220, 260])
        self.assertTrue(all(w["nominal_arrivals"] == 240.0 for w in ev["windows"]))


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
                self.assertIn(r["engine_tree"], ("pristine", "patched", "vllm"))
                scraped = r["arm"] == "VL-stock" or r["cell"] == "INSTR_METRICS"
                self.assertEqual("server_ttft_metric" in r, scraped, r["run_id"])
                if scraped:
                    self.assertIsNotNone(r["server_ttft_metric"]["before"]["count"])
                    self.assertIsNotNone(r["server_ttft_metric"]["after"]["count"])
                self.assertLessEqual(r["t_load_start_epoch"], r["t_load_end_epoch"])
                q = r["segments"][0]["requests"][0]
                for k in ("t_sent_mono", "t_finished_mono", "token_t_mono"):
                    self.assertIn(k, q)
                self.assertTrue(r["tree_stamp"])
            boots = [json.loads(p.read_text()) for p in (out / "boots").glob("*/boot.json")]
            self.assertTrue(all(b["job_name"] == "jobA" and b["gpu_uuid"] == "GPU-a" for b in boots))
            self.assertTrue(all(b["image_env"]["sglang_ok"] for b in boots))
            vl = [b for b in boots if b["engine"] == "vllm"]
            self.assertEqual(len(vl), 2)
            for b in vl:
                st = b["tree_stamp"]
                self.assertEqual(st["engine_tree"], "vllm")
                self.assertEqual(st["vllm_venv_freeze_sha256"], SP.VLLM_VENV_FREEZE_SHA256)
                self.assertEqual((st["attention_backend"], st["cudagraph_mode"]), ("FLASHINFER", "FULL_AND_PIECEWISE"))

    def test_J2_synth_window_recorded_apart(self):
        """The SG probe run records the W1 load window and the synthetic window separately (W1 window ends before the synthetic one)."""
        with tempfile.TemporaryDirectory() as td:
            plan, out, res = run_fake_job(td, "J2", ADD)
            r = json.loads((out / "runs" / "J2_INSTR_SG.json").read_text())
            self.assertEqual([s["phase"] for s in r["segments"]], ["W1", SP.SYNTH_PHASE])
            self.assertEqual(len(r["segments"][1]["requests"]), 21)
            self.assertEqual(r["synth"]["n_requests"], 21)
            self.assertEqual(r["t_load_end_epoch"], r["segments"][0]["t_load_end_epoch"])
            self.assertLessEqual(r["t_load_end_epoch"], r["t_synth_start_epoch"])
            self.assertLessEqual(r["t_load_end_mono"], r["t_synth_start_mono"])
            self.assertLess(r["t_synth_start_epoch"], r["t_synth_end_epoch"])
            for other in ("J2_INSTR_PM", "J2_INSTR_METRICS"):
                o = json.loads((out / "runs" / f"{other}.json").read_text())
                self.assertNotIn("t_synth_start_epoch", o)
                self.assertEqual([s["phase"] for s in o["segments"]], ["W1"])

    def test_image_mismatch_refuses_boot(self):
        with tempfile.TemporaryDirectory() as td:
            _p, out, res = run_fake_job(td, "J2", ADD, image_check_fn=lambda: dict(IMG_OK, vllm_ok=False))
            st = {b["boot_id"]: b["status"] for b in res}
            self.assertTrue(all(v == "REFUSED_IMAGE_ENV" for k, v in st.items() if "VL-stock" in k))
            self.assertTrue(all(v == "done" for k, v in st.items() if "VL-stock" not in k))
            self.assertEqual(RU.exit_code(res), 17)
            b = json.loads((out / "boots" / [k for k in st if "VL-stock" in k][0] / "boot.json").read_text())
            self.assertIn("VLLM_VENV_FREEZE", b["why"])
            self.assertEqual(list((out / "runs").glob("*VL-stock*")), [])
        with tempfile.TemporaryDirectory() as td:
            _p, _o, res = run_fake_job(td, "J1", None, image_check_fn=lambda: dict(IMG_OK, sglang_ok=False))
            self.assertTrue(all(b["status"] == "REFUSED_IMAGE_ENV" for b in res))

    def test_dead_boot_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            _p, _o, res = run_fake_job(td, "J1", None, server_kw={"die": True})
            self.assertEqual(RU.exit_code(res), 17)

    def test_sustain_extra_point_only_when_both_fail(self):
        with tempfile.TemporaryDirectory() as td:
            _p, out, _r = run_fake_job(td, "J1b", ADD, ttft_of=lambda a, q: 9000.0 if a == "VL-stock" else 100.0)
            ev = json.loads((out / "boots" / "J1b_sus_VL-stock" / "SUSTAIN_EVAL.json").read_text())
            ev_ok = json.loads((out / "boots" / "J1b_sus_PM-ref" / "SUSTAIN_EVAL.json").read_text())
            self.assertEqual(ev_ok["extra_needed"], False)
            names = [p.name for p in (out / "runs").glob("*VL-stock_sus*")]
            names_ok = [p.name for p in (out / "runs").glob("*PM-ref_sus*")]
            self.assertEqual(len(names_ok), 2)
            # TTFT 9000 everywhere still passes the completion/growth rule (sustain rule is not the SLO) -> no extra either;
            # force a failure through completion: rerun with growth
        with tempfile.TemporaryDirectory() as td:
            def tt(a, q):
                return 100.0 + (2000.0 * (q["arrival_s"] > 480)) if a == "VL-stock" else 100.0
            _p, out, _r = run_fake_job(td, "J1b", ADD, ttft_of=tt)
            ev = json.loads((out / "boots" / "J1b_sus_VL-stock" / "SUSTAIN_EVAL.json").read_text())
            self.assertTrue(ev["extra_needed"])
            self.assertEqual(len(list((out / "runs").glob("*VL-stock_sus*"))), 3)

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

    def test_server_args_and_opportunities_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            _p, out, _r = run_fake_job(td, "J2", ADD)
            runs = [json.loads(p.read_text()) for p in (out / "runs").glob("*.json")]
            mix = [r for r in runs if r["arm"] == "SG-mix"]
            stock = [r for r in runs if r["arm"] == "SG-stock" and r["boot_role"] == "meas"]
            self.assertTrue(mix and stock)
            self.assertTrue(all(RU.mix_fired_from_server_args(r["server_args"]) for r in mix))
            self.assertFalse(any(RU.mix_fired_from_server_args(r["server_args"]) for r in stock))
            self.assertTrue(all(r.get("mixing_opportunities") is not None for r in mix + stock))
            self.assertTrue(all("mixing_opportunities" not in r for r in runs if r["arm"] == "SG-off"))

    def test_J3_r3_budget_guard_all_or_nothing(self):
        """rev2 (audit r1 ⑨): right after the W3 ladder, time left < R3 estimate -> EVERY R3 boot R3_SKIPPED_BUDGET (no half ABBA)."""
        def tt(a, q):
            return 9e6 if q["phase"] == "W3" and q["request_id"].split("_W3lad_r")[-1].startswith(("0.50", "0.75")) else 100.0
        with tempfile.TemporaryDirectory() as td:
            plan, out, res = run_fake_job(td, "J3", ADD, ttft_of=tt, clock=lambda: 0.0)       # no time spent -> R3 runs
            bud = json.loads((out / "R3_BUDGET.json").read_text())
            self.assertFalse(bud["skip"])
            self.assertTrue(list((out / "runs").glob("*_R3_s*.json")))
        with tempfile.TemporaryDirectory() as td:
            plan = SP.make_plan("J3", ADD)
            calls = {"n": 0}                                   # start at 0; at the decision elapsed = hard cap - 1 min -> left 1 min < need
            def clock():
                calls["n"] += 1
                return 0.0 if calls["n"] == 1 else (plan["hard_cap_min"] - 1) * 60.0
            plan2, out, res = run_fake_job(td, "J3", ADD, ttft_of=tt, clock=clock)
            bud = json.loads((out / "R3_BUDGET.json").read_text())
            self.assertTrue(bud["skip"])
            self.assertEqual(bud["status"], SP.R3_BUDGET_STATUS)
            self.assertGreater(bud["need_min"], bud["left_min"])
            st = {b["boot_id"]: b["status"] for b in res}
            r3 = [k for k in st if "_R3_" in k]
            self.assertEqual(len(r3), 2 * len(SP.W3_ARMS))
            self.assertTrue(all(st[k] == SP.R3_BUDGET_STATUS for k in r3))
            self.assertEqual(list((out / "runs").glob("*_R3_s*.json")), [])
            self.assertEqual(st["J3_ref_end"], "done")
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
        return {"boot_id": f"b_{arm}", "engine": "vllm", "engine_tree": "vllm", "status": "ok", "tree_stamp": tree_stamp or vllm_stamp()}
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
OFFSET_OK = {"status": "TTFT_OFFSET_OK"}
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

    def _pair(self, job, key, smoke=SMOKE_OK, cell="R2", offset=OFFSET_OK):
        return AN.analyze_cell(job, cell, None, smoke, 3000.0, offset)["pairs"][key]

    def test_equiv_cell_valid(self):
        p = self._pair(job_of(EQ, jitter=JIT), "Q8:SG-off_vs_SG-stock")
        self.assertEqual(p["label"], "SIGN_UNRESOLVED", p)                 # rev3: Q8 is sign-only (EQUIV is a magnitude statement)
        self.assertEqual(p["raw_label"], "EQUIV_WITHIN_3PCT", p)
        self.assertEqual(p["validity"], [])
        p = self._pair(job_of(EQ, jitter=JIT), "Q4:PM-ref_vs_VL-stock")
        self.assertEqual(p["label"], "EQUIV_WITHIN_3PCT", p)

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
        self.assertEqual(self._pair(j, "Q3:VL-stock_vs_SG-stock")["label"], "SIGN_UNRESOLVED")
        self.assertEqual(self._pair(j, "Q4:PM-ref_vs_VL-stock")["label"], "EQUIV_WITHIN_3PCT")
        self.assertEqual(self._pair(j, "Q8:SG-off_vs_SG-stock", smoke={"status": "CLIENT_DEF_UNCHECKED"})["label"], "SIGN_UNRESOLVED")

    def test_engine_pair_needs_in_load_offset(self):
        j = job_of(EQ, jitter=JIT)
        p = self._pair(j, "Q3:VL-stock_vs_SG-stock", offset={"status": "TTFT_OFFSET_SHIFT"})
        self.assertEqual(p["label"], "INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT")
        self.assertIn("TTFT_OFFSET_SHIFT", p["tags"])
        p = self._pair(j, "Q4:PM-ref_vs_VL-stock", offset=None)
        self.assertEqual(p["label"], "INVALID")
        self.assertIn("TTFT_OFFSET_UNCHECKED", p["validity"])
        self.assertEqual(self._pair(j, "Q8:SG-off_vs_SG-stock", offset={"status": "TTFT_OFFSET_SHIFT"})["label"], "SIGN_UNRESOLVED")

    def test_vllm_stamp_rules(self):
        p = self._pair(job_of(EQ, jitter=JIT, stamp_override={"VL-stock": vllm_stamp(vllm_venv_freeze_sha256="0" * 64)}),
                       "Q3:VL-stock_vs_SG-stock")
        self.assertEqual(p["label"], "INVALID")
        self.assertIn("VLLM_VENV_FREEZE:VL-stock", p["validity"])
        p = self._pair(job_of(EQ, jitter=JIT, stamp_override={"VL-stock": vllm_stamp(sglang_freeze_sha256="1" * 64)}),
                       "Q3:VL-stock_vs_SG-stock")
        self.assertIn("SGLANG_FREEZE:VL-stock", p["validity"])
        j = job_of(EQ, jitter=JIT)
        j["boots"]["b_smoke_vl"] = dict(boot_rec("VL-stock", vllm_stamp(cudagraph_mode="PIECEWISE")), boot_id="b_smoke_vl")
        p = self._pair(j, "Q3:VL-stock_vs_SG-stock")
        self.assertEqual(p["label"], "INVALID")
        self.assertIn("VLLM_STAMP_DIFFERS_IN_JOB:2", p["validity"])
        self.assertEqual(self._pair(j, "Q8:SG-off_vs_SG-stock")["label"], "SIGN_UNRESOLVED")

    @staticmethod
    def _offset_j2(sg_server_ms, vl_server_ms, client_ms=200.0, n=200, vl_count_delta=None):
        def run(rid, arm, cell, role, server_ms, dc=None):
            reqs = [{"request_id": f"q{i}", "phase": "W1", "ttft_ms": client_ms, "token_itl_ms": [20.0], "error": ""} for i in range(n)]
            m = {"before": {"sum_s": 10.0, "count": 50}, "after": {"sum_s": 10.0 + n * server_ms / 1000.0, "count": 50 + (dc or n)}}
            return {"run_id": rid, "arm": arm, "cell": cell, "boot_role": role, "seed": 1, "segments": [{"phase": "W1", "duration_s": 60.0,
                                                                                                        "requests": reqs}],
                    "server_ttft_metric": m}
        runs = [run("m", "INSTR-METRICS-SG-stock", "INSTR_METRICS", "instr_metrics", sg_server_ms)]
        runs += [run(f"v{s}", "VL-stock", "R1", "meas", vl_server_ms, vl_count_delta) for s in SP.EVAL_SEEDS]
        runs.append(run("v_r2", "VL-stock", "R2", "meas", 0.0))                  # R2 runs are not part of the comparison
        return {"runs": runs, "boots": {}, "plan": {"kind": "STEADY"}}

    def test_ttft_offset_verdict(self):
        smoke = {"per_engine": {"sglang": {"median_ms": 4.0}, "vllm": {"median_ms": 6.0}}}
        self.assertEqual(AN.ttft_offset_verdict(smoke, self._offset_j2(197.0, 195.0))["status"], "TTFT_OFFSET_OK")     # 3 / 5 ms
        v = AN.ttft_offset_verdict(smoke, self._offset_j2(170.0, 195.0))                                                # SG 30 ms
        self.assertEqual(v["status"], "TTFT_OFFSET_SHIFT")
        self.assertAlmostEqual([r for r in v["rows"] if r["engine"] == "sglang"][0]["load_offset_ms"], 30.0)
        self.assertEqual(AN.ttft_offset_verdict(smoke, self._offset_j2(197.0, 187.0))["status"], "TTFT_OFFSET_OK")      # VL 13 ms vs 6: no 2x rule
        self.assertEqual(AN.ttft_offset_verdict(smoke, self._offset_j2(197.0, 170.0))["status"], "TTFT_OFFSET_SHIFT")   # VL 30 ms vs 6
        self.assertEqual(AN.ttft_offset_verdict(smoke, self._offset_j2(197.0, 195.0, vl_count_delta=150))["status"], "TTFT_OFFSET_UNCHECKED")
        self.assertEqual(AN.ttft_offset_verdict({"per_engine": {"sglang": {"median_ms": 4.0}}}, self._offset_j2(197.0, 195.0))["status"],
                         "TTFT_OFFSET_UNCHECKED")
        self.assertEqual(AN.ttft_offset_verdict(smoke, None)["status"], "TTFT_OFFSET_UNCHECKED")
        self.assertTrue(AN.offset_shift(25.0, 4.0))
        self.assertFalse(AN.offset_shift(9.0, 4.0))             # factor 2.25 but 5 ms apart: absolute 20 ms rule only (rev1.1)
        self.assertFalse(AN.offset_shift(23.0, 4.0))            # 19 ms
        self.assertTrue(AN.offset_shift(3.0, 25.0))
        self.assertTrue(AN.offset_shift(65.0, 40.0))
        self.assertFalse(AN.offset_shift(55.0, 40.0))

    def test_smoke_verdict(self):
        """rev2 (§5-0 (a) wording): per-engine offset sd <= 5 ms; the two engines' constants need not agree."""
        def j1(d_sg, d_vl, spread_sg=0.0, spread_vl=0.0):
            rows = lambda d, sp: [{"client_ttft_ms": 100.0 + d + (sp if i % 2 else -sp), "server_ttft_ms": 100.0, "ok": True} for i in range(20)]
            return {"runs": [{"kind": "smoke", "engine": "sglang", "rows": rows(d_sg, spread_sg)},
                             {"kind": "smoke", "engine": "vllm", "rows": rows(d_vl, spread_vl)}]}
        self.assertEqual(AN.smoke_verdict(j1(2.0, 4.0))["status"], "CLIENT_DEF_OK")
        self.assertEqual(AN.smoke_verdict(j1(2.0, 30.0))["status"], "CLIENT_DEF_OK")          # different constants: allowed
        self.assertEqual(AN.smoke_verdict(j1(2.0, 4.0, spread_vl=9.0))["status"], "CLIENT_DEF_MISMATCH")   # vLLM sd 9 ms
        self.assertEqual(AN.smoke_verdict(j1(2.0, 4.0, spread_sg=9.0))["status"], "CLIENT_DEF_MISMATCH")   # SGLang sd 9 ms
        self.assertEqual(AN.smoke_verdict(j1(2.0, 4.0, spread_sg=4.0))["status"], "CLIENT_DEF_OK")
        self.assertEqual(AN.smoke_verdict({"runs": [j1(1, 1)["runs"][0]]})["status"], "CLIENT_DEF_UNCHECKED")

    def test_out_of_band(self):
        """rev1: judged on the steady-state cells R2 (W1) and R3 (W3) -- all arms > 0.9 or best < 0.5; R1 is never tagged."""
        for cell, ph in (("R2", "W1"), ("R3", "W3")):
            j = job_of({a: 95 for a in EQ}, cell=cell, jitter=JIT, phases=(ph,))
            c = AN.analyze_cell(j, cell, None, SMOKE_OK, 3000.0, OFFSET_OK)
            self.assertTrue(c["cell_out_of_band"], cell)
            self.assertIn("CELL_OUT_OF_BAND", c["pairs"]["Q2:PM-ref_vs_SG-stock"]["tags"])
            j = job_of({a: 75 for a in EQ}, cell=cell, jitter=JIT, phases=(ph,))
            self.assertFalse(AN.analyze_cell(j, cell, None, SMOKE_OK, 3000.0, OFFSET_OK)["cell_out_of_band"])
            j = job_of({a: 40 for a in EQ}, cell=cell, jitter=JIT, phases=(ph,))
            self.assertTrue(AN.analyze_cell(j, cell, None, SMOKE_OK, 3000.0, OFFSET_OK)["cell_out_of_band"])
        j = job_of({a: 95 for a in EQ}, cell="R1", jitter=JIT)
        self.assertIsNone(AN.analyze_cell(j, "R1", None, SMOKE_OK, 3000.0, OFFSET_OK)["cell_out_of_band"])

    def test_substrate(self):
        self.assertEqual(SP.REFCELL_ARM, "PM-ref")                          # rev3
        base = mk_run("j1", "PM-ref", 201, "LADDER", [seg("W1", 95, 5, 20.0)], th="REF")
        other = mk_run("j1x", "SG-stock", 201, "LADDER", [seg("W1", 20, 80, 20.0)], th="REF")     # same trace, other arm: not the baseline
        j1 = {"runs": [other, base]}
        ref = lambda tag, g: mk_run(f"r{tag}", "PM-ref", 201, "REFCELL", [seg("W1", g, 100 - g, 20.0)], th="REF",
                                    spec={"tag": tag})
        self.assertEqual(AN.substrate({"runs": [ref("start", 93), ref("end", 90)]}, j1)["tag"], "SUBSTRATE_OK")
        self.assertEqual(AN.substrate({"runs": [ref("start", 93), ref("end", 84)]}, j1)["tag"], "SUBSTRATE_SHIFT")
        self.assertEqual(AN.substrate({"runs": [ref("start", 93)]}, {"runs": []})["tag"], "SUBSTRATE_UNCHECKED")

    def test_mix_not_fired(self):
        """rev2: firing = server_args enable_mixed_chunk=True + chunked prefill on; the opportunity count does not decide."""
        on = {"enable_mixed_chunk": "True", "chunked_prefill_size": "8192"}
        j = job_of(EQ, jitter=JIT)
        for r in j["runs"]:
            if r["arm"] == "SG-mix":
                r["mixing_opportunities"] = 500
                r["server_args"] = dict(on, enable_mixed_chunk="False")
        p = self._pair(j, "Q5:SG-mix_vs_SG-stock")
        self.assertEqual(p["label"], "MIX_NOT_FIRED")
        self.assertTrue(all(v["label"] == "MIX_NOT_FIRED" for v in p["latency"].values()))
        for r in j["runs"]:
            if r["arm"] == "SG-mix":
                r["mixing_opportunities"] = 0
                r["server_args"] = dict(on)
            if r["arm"] == "SG-stock":
                r["mixing_opportunities"] = 7
        c = AN.analyze_cell(j, "R2", None, SMOKE_OK, 3000.0, OFFSET_OK)
        self.assertEqual(c["pairs"]["Q5:SG-mix_vs_SG-stock"]["label"], "SIGN_UNRESOLVED")
        self.assertEqual(set(c["mixing_opportunities"]), {"SG-mix", "SG-stock"})     # negative control reported
        self.assertEqual(set(c["mixing_opportunities"]["SG-stock"].values()), {7})
        j2 = job_of(EQ, jitter=JIT)                                                   # server_args from the boot record also count
        j2["boots"]["b_SG-mix"]["server_args"] = dict(on)
        self.assertEqual(self._pair(j2, "Q5:SG-mix_vs_SG-stock")["label"], "SIGN_UNRESOLVED")
        self.assertEqual(self._pair(job_of(EQ, jitter=JIT), "Q5:SG-mix_vs_SG-stock")["label"], "MIX_NOT_FIRED")

    def test_both_saturated(self):
        """rev2 (audit r1 B): both arms >= 0.98 on all 4 seeds -> BOTH_SATURATED, never EQUIV; one arm below -> ordinary label."""
        sat = {a: 99 for a in EQ}
        jit = {(a, s): d for a in EQ for s, d in zip(SP.EVAL_SEEDS, (0, 1, -1, 0))}
        for cell in ("R1", "R2"):
            c = AN.analyze_cell(job_of(sat, cell=cell, jitter=jit), cell, None, SMOKE_OK, 3000.0, OFFSET_OK)
            p = c["pairs"]["Q8:SG-off_vs_SG-stock"]
            self.assertEqual(p["label"], "BOTH_SATURATED", cell)
            self.assertEqual(p["raw_label"], "EQUIV_WITHIN_3PCT")
            self.assertTrue(p["both_saturated"])
        p = self._pair(job_of(dict(sat, **{"SG-off": 90}), jitter=jit), "Q8:SG-off_vs_SG-stock")
        self.assertFalse(p["both_saturated"])
        self.assertEqual(p["label"], "OFF_BELOW_STOCK_SIGN")                # rev3: sign only
        self.assertEqual(p["raw_label"], "OFF_BELOW_STOCK")
        p = self._pair(job_of(EQ, jitter=JIT), "Q8:SG-off_vs_SG-stock")
        self.assertFalse(p["both_saturated"])

    def test_latency_co_metrics_q8_q5(self):
        """rev2 (audit r1 B): Q8/Q5 carry seed-level paired ln TPOT mean / TTFT p50 / ITL p95-median labels (same 3 % + t(3) rule)."""
        j = job_of({a: 99 for a in EQ}, jitter={(a, s): d for a in EQ for s, d in zip(SP.EVAL_SEEDS, (0, 1, -1, 0))})
        for r in j["runs"]:
            r["server_args"] = {"enable_mixed_chunk": "True", "chunked_prefill_size": "8192"}
            if r["arm"] == "SG-off":
                for q in r["segments"][0]["requests"]:
                    q["token_itl_ms"] = [22.0, 22.0, 22.0]                              # 10 % slower per token
            elif r["arm"] in ("SG-stock", "SG-mix"):
                for q in r["segments"][0]["requests"]:
                    q["token_itl_ms"] = [20.0, 20.0, 20.0]
        c = AN.analyze_cell(j, "R2", None, SMOKE_OK, 3000.0, OFFSET_OK)
        q8 = c["pairs"]["Q8:SG-off_vs_SG-stock"]
        self.assertEqual(q8["label"], "BOTH_SATURATED")                                # goodput blind to the 10 % ...
        self.assertEqual(q8["latency"]["TPOT"]["label"], "OFF_ABOVE_STOCK_TPOT")        # ... latency is not
        self.assertEqual(q8["latency"]["ITL"]["label"], "OFF_ABOVE_STOCK_ITL")
        self.assertEqual(q8["latency"]["TTFT"]["label"], "EQUIV_WITHIN_3PCT_TTFT")
        self.assertAlmostEqual(q8["latency"]["TPOT"]["mean"], math.log(1.1))
        self.assertEqual(c["pairs"]["Q5:SG-mix_vs_SG-stock"]["latency"]["TPOT"]["label"], "EQUIV_WITHIN_3PCT_TPOT")
        self.assertEqual(set(SP.LATENCY_PAIRS), {"Q3", "Q5", "Q8"})
        self.assertEqual(c["pairs"]["Q2:PM-ref_vs_SG-stock"]["latency"], {})
        self.assertEqual(set(c["pairs"]["Q3:VL-stock_vs_SG-stock"]["latency"]), {"TPOT", "TTFT", "ITL"})      # rev3: Q3 gains them
        self.assertEqual(c["pairs"]["Q3:VL-stock_vs_SG-stock"]["latency"]["TPOT"]["label"], "EQUIV_WITHIN_3PCT_TPOT")
        [r for r in j["runs"] if r["arm"] == "SG-off"][0]["segments"][0]["requests"][0]["error"] = "boom"   # validity reaches latency
        c = AN.analyze_cell(j, "R2", None, SMOKE_OK, 3000.0, OFFSET_OK)
        self.assertEqual(c["pairs"]["Q8:SG-off_vs_SG-stock"]["latency"]["TPOT"]["label"], "INVALID")

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
        j1 = {"runs": lad("PM-ref", [5.0], [6.0, 7.0]) + lad("VL-stock", [5.0, 6.0], [7.0]) + lad("SG-stock", [], [5.516, 6.0])}
        def sus(arm, rate, ok):
            n = int(rate * 600)
            reqs = [{"scheduled_arrival_s": i / rate, "sent_s": i / rate, "ttft_ms": 100.0, "completion_s": i / rate + 1, "error": "",
                     "token_itl_ms": [5.0]} for i in range(n if ok else n // 2)]
            return mk_run(f"s{arm}{rate}", arm, 201, "SUSTAIN", [{"phase": "W1", "duration_s": 600.0, "requests": reqs}], spec={"rate": rate})
        j1b = {"runs": [sus("PM-ref", 6.0, False), sus("PM-ref", 4.8, True), sus("SG-stock", 3.0, True)]}
        add = AN.build_addendum(j1, None, j1b)
        self.assertEqual(add["lambda_star_burst"], {"PM-ref": 5.0, "VL-stock": 6.0})
        self.assertEqual(add["ladder"]["SG-stock"]["status"], "LADDER_BOTTOM_FAIL")
        self.assertEqual((add["refcell_arm"], add["refcell_rate"]), ("PM-ref", 5.0))       # rev3: PM-ref burst rung, not VL (6.0) / max
        self.assertEqual(add["lambda_star_ref"], 4.8)                        # SG-stock's 3.0 is excluded from the min (rev3)
        self.assertEqual(add["lambda_star_ref_scope"], ["PM-ref"])
        self.assertEqual(add["lambda_star_ref_missing_arms"], ["VL-stock"])
        self.assertIn("SG-stock", add["lambda_star_ref_excluded"])
        j1b2 = {"runs": j1b["runs"] + [sus("VL-stock", 5.0, True), sus("VL-stock", 4.0, True)]}
        self.assertEqual(AN.build_addendum(j1, None, j1b2)["lambda_star_ref"], 4.8)
        j1b3 = {"runs": j1b2["runs"] + [sus("SG-off", 3.5, False), sus("SG-off", 2.8, False)]}     # an SG arm with None: still excluded
        self.assertEqual(AN.build_addendum(j1, None, j1b3)["lambda_star_ref"], 4.8)

    def test_ladder_down(self):
        """rev3: pick rule over J1 + J1-down points; bottom failure again -> LADDER_BOTTOM_FAIL_2 (no value); Q1 only."""
        def lad(arm, cell, rates_ok, rates_fail):
            out = []
            for r in rates_ok + rates_fail:
                for s in (201, 202):
                    g = 118 if r in rates_ok else 30
                    out.append(mk_run(f"{arm}{cell}{r}{s}", arm, s, cell, [seg("W1", g, 120 - g, 20.0)], spec={"rate": r}))
            return out
        down = SP.ladder_rates(down=True)                                    # [4.796522, 4.170888, 3.62686, 3.153791]
        j1 = {"runs": lad("SG-stock", "LADDER", [], [5.516, 6.3434]) + lad("SG-mix", "LADDER", [], [5.516]) +
                      lad("SG-off", "LADDER", [], [5.516]) + lad("PM-ref", "LADDER", [5.516, 7.29491], [8.389146])}
        jd = {"runs": lad("SG-stock", "LADDER_DOWN", down[2:] + [down[1]], [down[0]]) + lad("SG-mix", "LADDER_DOWN", [], down) +
                      lad("SG-off", "LADDER_DOWN", down, [])}
        add = AN.build_addendum(j1, None, None, j1down=jd)
        self.assertEqual(add["ladder_down"]["SG-stock"]["status"], "BRACKETED")
        self.assertEqual(add["lambda_star_burst_down"]["SG-stock"], 4.170888)
        self.assertEqual(add["ladder_down"]["SG-mix"]["status"], "LADDER_BOTTOM_FAIL_2")
        self.assertIsNone(add["lambda_star_burst_down"]["SG-mix"])
        self.assertEqual((add["ladder_down"]["SG-off"]["status"], add["lambda_star_burst_down"]["SG-off"]), ("BRACKETED", 4.796522))
        self.assertEqual(add["lambda_star_burst"], {"PM-ref": 7.29491})      # J1-down never feeds the J1b input
        self.assertEqual(add["refcell_rate"], 7.29491)
        self.assertEqual(AN.q1_burst(add), (4.170888, "J1 + J1-down ladder"))
        rows = AN.question_rows(add, {}, {})
        self.assertEqual((rows["Q1"]["status"], rows["Q1"]["detail"]["burst"]), ("PARTIAL", 4.170888))
        P = AN.score_predictions(add, {}, {})
        self.assertEqual((P["Q1-SG"]["value"], P["Q1-SG"]["score"]), (4.170888, "HIT"))
        self.assertIsNone(P["P-1"]["score"])                                 # P-1 stays on the first-Job ladder (unchanged rev2.1 rule)
        bad = AN.build_addendum(j1, None, None, j1down={"runs": lad("SG-stock", "LADDER_DOWN", [], down)})
        self.assertEqual(AN.score_predictions(bad, {}, {})["Q1-SG"]["score"], "UNSCORABLE")
        self.assertIn("LADDER_BOTTOM_FAIL_2", AN.score_predictions(bad, {}, {})["Q1-SG"]["why"])
        self.assertEqual(AN.score_predictions(AN.build_addendum(j1), {}, {})["Q1-SG"]["score"], "UNSCORABLE")
        nm = "\n".join(AN.not_measured({}, {}, {"status": "CLIENT_DEF_OK"}, {}, {"status": "TTFT_OFFSET_OK"}, bad))
        self.assertIn("J1-down SG-stock: LADDER_BOTTOM_FAIL_2", nm)
        self.assertIn("기준 용량", nm)

    def test_sign_only_pairs(self):
        """rev3: Q3 / Q5 / Q8 pairs carry the goodput sign only (+ MAGNITUDE_NOT_CITABLE); Q2 and Q4 keep the registered labels."""
        self.assertEqual(SP.SIGN_ONLY_QUESTIONS, ("Q3", "Q5", "Q8"))
        g = dict(EQ, **{"SG-off": 92, "VL-stock": 70, "PM-ref": 92, "SG-mix": 80})
        j = job_of(g, jitter=JIT)
        for r in j["runs"]:
            if r["arm"] == "SG-mix":
                r["server_args"] = {"enable_mixed_chunk": "True", "chunked_prefill_size": "8192"}
        c = AN.analyze_cell(j, "R2", None, SMOKE_OK, 3000.0, OFFSET_OK)
        q8, q3, q2 = c["pairs"]["Q8:SG-off_vs_SG-stock"], c["pairs"]["Q3:VL-stock_vs_SG-stock"], c["pairs"]["Q2:PM-ref_vs_SG-stock"]
        self.assertEqual((q8["raw_label"], q8["label"]), ("OFF_ABOVE_STOCK", "OFF_ABOVE_STOCK_SIGN"))
        self.assertIn("MAGNITUDE_NOT_CITABLE", q8["tags"])
        self.assertEqual(q3["label"], "VL_BELOW_SG_SIGN")
        self.assertEqual(c["pairs"]["Q5:SG-mix_vs_SG-stock"]["label"], "SIGN_UNRESOLVED")
        self.assertEqual(q2["label"], "PM_ABOVE_SG")
        self.assertNotIn("MAGNITUDE_NOT_CITABLE", q2["tags"])
        self.assertFalse(c["pairs"]["Q4:PM-ref_vs_VL-stock"]["sign_only"])
        self.assertEqual(AN.sign_label((0.01, 0.002, 0.02, 4), ("A", "B")), "A_ABOVE_B_SIGN")       # no 3 % margin on the sign
        self.assertEqual(AN.sign_label((0.01, 0.002, 0.02, 3), ("A", "B")), "SIGN_UNRESOLVED")
        nm = "\n".join(AN.not_measured({}, c and {"R2": c}, SMOKE_OK, {}, OFFSET_OK, None))
        self.assertIn("Q8:SG-off_vs_SG-stock: rev3 부호", nm)


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
        res = AN.r_ser_from_instr(self.probe(), run)["windows"]["W1"]
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

    def test_q6_grid_and_other_stream_invalid(self):
        """rev2 (audit r1 C): grid {512, 2k} (8k removed); one gap with an extend on another stream -> Q6 INVALID."""
        self.assertEqual(SP.Q6_L_GRID, (512, 2048))
        run = {"t_load_start_epoch": 1100.0, "t_load_end_epoch": 1200.0}
        self.assertEqual(AN.r_ser_from_instr(self.probe(), run)["q6_status"], "Q6_NO_SYNTH_WINDOW")
        recs = self.probe()
        recs.append({"event": "extend_forward", "seq": 19, "mode": "extend", "extend_gpu_ms": 5.0, "stream_key": 2, "extend_tokens": 512,
                     "n_reqs": 1, "t_host_start": 169.0})
        recs.append({"event": "decode_forward", "seq": 20, "mode": "decode", "bs_real": 4, "stream_key": 1, "decode_fwd_gpu_ms": 10.0,
                     "t_host_start": 170.0, "decode_iter_seq": 20})
        res = AN.r_ser_from_instr(recs, run)
        self.assertEqual(res["q6_status"], "Q6_INVALID:OTHER_STREAM_EXTEND")
        self.assertEqual(res["n_other_stream_gaps"], 1)
        P = AN.score_predictions({}, {}, {"SG": dict(res, windows={"SYNTH": {"by_L": {2048: {"n": 12, "r_ser_p50": 15.0, "bs_p50": 1}}}})})
        self.assertEqual(P["P-4"]["score"], "UNSCORABLE")

    def test_p4_unscorable_below_10(self):
        sg = lambda n: {"q6_status": "Q6_FEW_SPANS", "windows": {"SYNTH": {"by_L": {2048: {"n": n, "r_ser_p50": 15.0, "bs_p50": 1}}}}}
        P = AN.score_predictions({}, {}, {"SG": sg(9)})
        self.assertEqual(P["P-4"]["score"], "UNSCORABLE")
        self.assertIsNone(P["P-4"]["value"])
        P = AN.score_predictions({}, {}, {"SG": sg(10)})
        self.assertIn(P["P-4"]["score"], ("HIT", "MISS"))
        self.assertEqual(AN.score_predictions({}, {}, {})["P-4"]["score"], "UNSCORABLE")

    @staticmethod
    def two_window_probe(n512=10, n2k=10, multi2k=0, w1_single=10):
        """W1 window = host [150, 250] (epoch [1100, 1200]): bs-4 decodes at 10 ms, w1_single single-request 2k extends at 30 ms (r_ser 3).
        Synthetic window = host [350, 450] (epoch [1300, 1400]): lead-in extend (64 tok, first gap -> dropped), bs-1 decodes at 5 ms, single
        512 extends at 25 ms (r_ser 5), single 2k extends at 75 ms (r_ser 15), multi-request 2k extends (n_reqs 2) at 75 ms."""
        recs = [{"event": "decode_probe_boot", "extend_time_enabled": True, "t_realtime": 1000.0, "t_host": 50.0}]
        st = {"seq": 0}

        def add(ev, t, **kw):
            st["seq"] += 1
            base = {"event": ev, "seq": st["seq"], "stream_key": 1, "t_host_start": t}
            base.update({"mode": "decode", "decode_iter_seq": st["seq"]} if ev == "decode_forward" else {"mode": "extend"})
            base.update(kw)
            recs.append(base)

        def decodes(t, bs, ms, n=6):
            for i in range(n):
                add("decode_forward", t + 0.01 * i, bs_real=bs, decode_fwd_gpu_ms=ms)
        t = 151.0
        decodes(t, 4, 10.0)
        for _ in range(w1_single):
            t += 0.5
            add("extend_forward", t, extend_gpu_ms=30.0, extend_tokens=2048, n_reqs=1)
            decodes(t + 0.05, 4, 10.0)
        t = 351.0
        add("extend_forward", t, extend_gpu_ms=4.0, extend_tokens=64, n_reqs=1)
        decodes(t + 0.05, 1, 5.0)
        for L, ms, nr, n in ((512, 25.0, 1, n512), (2048, 75.0, 1, n2k), (2048, 75.0, 2, multi2k)):
            for _ in range(n):
                t += 2.0
                add("extend_forward", t, extend_gpu_ms=ms, extend_tokens=L, n_reqs=nr)
                decodes(t + 0.05, 1, 5.0)
        return recs

    RUN2 = {"t_load_start_epoch": 1100.0, "t_load_end_epoch": 1200.0, "t_synth_start_epoch": 1300.0, "t_synth_end_epoch": 1400.0}

    def test_synth_window_reduced_apart_from_w1(self):
        """§5-1 (2026-11-02): W1 and synthetic windows are reduced separately; P-4 = synthetic 2k single-request spans only; 512 descriptive."""
        res = AN.r_ser_from_instr(self.two_window_probe(), self.RUN2)
        w1, sy = res["windows"]["W1"], res["windows"]["SYNTH"]
        self.assertEqual((w1["window_probe_clock"], sy["window_probe_clock"]), ((150.0, 250.0), (350.0, 450.0)))
        self.assertAlmostEqual(w1["r_ser_p50"], 3.0)                    # pooled windows would move this (W1 3.0 + synth 5 / 15)
        self.assertEqual(w1["by_L"], {2048: {"n": 10, "r_ser_p50": 3.0, "bs_p50": 4}})
        self.assertEqual(sy["by_L"][512], {"n": 10, "r_ser_p50": 5.0, "bs_p50": 1})
        self.assertEqual(sy["by_L"][2048], {"n": 10, "r_ser_p50": 15.0, "bs_p50": 1})
        self.assertEqual(sy["excluded"]["n_first"], 1)                  # the lead-in's gap, never counted
        self.assertEqual(res["n_single_request_spans"], {"W1": 10, "SYNTH": 20})
        self.assertEqual(res["q6_status"], "Q6_OK")
        P = AN.score_predictions({}, {}, {"SG": res})["P-4"]
        self.assertEqual((P["source"], P["value"], P["n"], P["forecast_cell"]), ("SYNTH", 15.0, 10, {"L": 2048, "bs": 1, "ctx": 2048}))
        self.assertEqual(P["score"], "HIT")                              # forecast 17.53, |15 - 17.53| <= 0.3 x 17.53
        self.assertEqual((P["descriptive_512"]["value"], P["descriptive_512"]["score"]), (5.0, "NOT_SCORED"))
        # W1 window alone never scores P-4, even with >= 10 single-request 2k spans in it
        P = AN.score_predictions({}, {}, {"SG": dict(res, windows={"W1": w1, "SYNTH": None})})["P-4"]
        self.assertEqual(P["score"], "UNSCORABLE")
        few = AN.r_ser_from_instr(self.two_window_probe(w1_single=9), self.RUN2)
        self.assertEqual(few["q6_status"], "Q6_FEW_SPANS")                # 9 + 20 < 30

    def test_multi_request_prefill_is_not_single(self):
        """A 2k-token EXTEND that carries 2 requests is not a single-request span: 9 single + 2 multi -> n = 9 -> P-4 UNSCORABLE."""
        res = AN.r_ser_from_instr(self.two_window_probe(n2k=9, multi2k=2), self.RUN2)
        sy = res["windows"]["SYNTH"]
        self.assertEqual(sy["by_L"][2048]["n"], 9)
        self.assertEqual(sy["n_single_request_spans"], 19)
        self.assertEqual(res["q6_status"], "Q6_FEW_SPANS")
        self.assertEqual(AN.score_predictions({}, {}, {"SG": res})["P-4"]["score"], "UNSCORABLE")
        self.assertGreater(sy["n_r_ser"], sy["n_single_request_spans"])  # the multi-request spans exist and yield r_ser, just not counted

    def test_probe_window_needs_boot_clock(self):
        with self.assertRaises(ValueError):
            AN.probe_window([{"event": "decode_probe_boot"}], 0, 1)


class ReportTests(unittest.TestCase):
    def test_not_measured_lists_pending_and_deviation(self):
        j = job_of(EQ, jitter=JIT)
        j["plan"] = {"pending": [SP.VL_PROFILER_PENDING], "kind": "STEADY"}
        nm = AN.not_measured({"J2": j, "J3": None}, {"R2": AN.analyze_cell(j, "R2", None, {"status": "CLIENT_DEF_UNCHECKED"}, 3000.0)},
                             {"status": "CLIENT_DEF_UNCHECKED"}, {})
        txt = "\n".join(nm)
        self.assertIn("INSTR-VL-stock", txt)
        self.assertIn("TTFT_OFFSET_UNCHECKED", txt)
        self.assertIn("변화 trace 미적용", txt)
        self.assertIn("D-1", txt)
        self.assertIn("J3: Job raw tree absent", txt)
        self.assertIn("definition smoke", txt)
        self.assertIn("측정하지 않은 것", AN.render_not_measured(nm))

    def test_question_rows_forced(self):
        """rev2 (audit r1 ⑦): Q1-Q8 rows always exist, also with no data at all."""
        nm = AN.not_measured({"J1": None, "J1b": None, "J2": None, "J3": None}, {}, {"status": "CLIENT_DEF_UNCHECKED"}, {}, None, None)
        for q in SP.QUESTIONS:
            self.assertTrue(any(x.startswith(f"{q} NOT_MEASURED") for x in nm), q)
        self.assertTrue(any("기준 용량" in x and "addendum 없음" in x for x in nm))
        r = AN.report(None, None, None, None, add={})
        self.assertEqual(list(r["questions"]), list(SP.QUESTIONS))
        j = job_of(EQ, jitter=JIT)
        rows = AN.question_rows({"lambda_star_burst": {"SG-stock": 6.0}}, {"R2": AN.analyze_cell(j, "R2", None, SMOKE_OK, 3000.0, OFFSET_OK)}, {})
        self.assertEqual(rows["Q1"]["status"], "PARTIAL")
        self.assertEqual(rows["Q8"]["status"], "PARTIAL")                   # R2 measured, R1 cell absent
        self.assertEqual(rows["Q7"]["status"], "NOT_MEASURED")
        self.assertEqual(rows["Q6"]["status"], "NOT_MEASURED")

    def test_ref_capacity_none_reason(self):
        nm = AN.ref_capacity_rows({"lambda_star_ref": None, "lambda_star_ref_why_none": ["VL-stock"], "lambda_star_ref_missing_arms": []})
        self.assertTrue(nm and "VL-stock" in nm[0] and "None" in nm[0])
        self.assertEqual(AN.ref_capacity_rows({"lambda_star_ref": 4.0, "lambda_star_ref_missing_arms": []}), [])

    def test_not_measured_reads_done_json(self):
        """Skipped / correctness-failed / budget-skipped boots (DONE.json status) and never-finished boots are listed."""
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td)
            plan = {"kind": "W3", "pending": [], "boots": [{"boot_id": b} for b in ("a", "b", "c", "d")]}
            (raw / "plan.json").write_text(json.dumps(plan))
            for bid, st in (("a", "done"), ("b", "INVALID_CORRECTNESS"), ("c", SP.R3_BUDGET_STATUS)):
                (raw / "boots" / bid).mkdir(parents=True)
                (raw / "boots" / bid / "DONE.json").write_text(json.dumps({"boot_id": bid, "status": st}))
            (raw / "boots" / "b" / "boot.json").write_text(json.dumps({"boot_id": "b", "status": "ok"}))
            (raw / "runs").mkdir()
            j = AN.load_job(raw)
            txt = "\n".join(AN.not_measured({"J3": j}, {}, {"status": "CLIENT_DEF_OK"}, {}, {"status": "TTFT_OFFSET_OK"}, None))
            self.assertIn("boot b INVALID_CORRECTNESS", txt)
            self.assertIn(f"boot c {SP.R3_BUDGET_STATUS}", txt)
            self.assertIn("boot d DONE.json 없음", txt)
            self.assertNotIn("boot a ", txt)

    def test_report_refuses_wrong_plan_kind(self):
        j3 = job_of(EQ, cell="R3", jitter=JIT, phases=("W3",))
        j3["plan"] = {"kind": "TRACE_W3", "pending": []}
        j3["floor"] = {"floor_ms": 1000.0}
        r = AN.report(None, None, None, j3, add={})
        self.assertIn("J3", r["kind_problems"])
        self.assertNotIn("R3", r["cells"])
        j3["plan"]["kind"] = "W3"
        r = AN.report(None, None, None, j3, add={})
        self.assertEqual(r["kind_problems"], {})
        self.assertIn("R3", r["cells"])

    def test_predictions(self):
        add = {"lambda_star_burst": {"SG-stock": 7.0, "VL-stock": 9.5}}
        cells = {"R2": {"pairs": {"Q2:PM-ref_vs_SG-stock": {"n": 4, "mean": 0.05, "lo": 0.01, "hi": 0.09, "label": "PM_ABOVE_SG"}},
                        "gate5": {"SG-mix": {"itl_ms": {"p95": 30.0}}, "SG-stock": {"itl_ms": {"p95": 40.0}}}}}
        instr = {"SG": {"windows": {"SYNTH": {"by_L": {2048: {"n": 12, "r_ser_p50": 15.0, "bs_p50": 1}}}}}}
        P = AN.score_predictions(add, cells, instr)
        self.assertEqual(P["P-1"]["score"], "HIT")
        self.assertEqual(P["P-2"]["score"], "MISS")
        self.assertEqual(P["P-3"]["score"], "HIT(+)")
        self.assertEqual(P["P-4"]["forecast_cell"], {"L": 2048, "bs": 1, "ctx": 2048})
        self.assertEqual(P["P-4"]["score"], "HIT")
        self.assertEqual(P["P-5"]["score"], "HIT")

    def test_p3_unscorable_when_both_saturated(self):
        """rev2.1 (re-audit r2 (a)): a BOTH_SATURATED 80 % cell pair is never sign-scored, even with a CI clear of 0 (drain sign only)."""
        sat = {"n": 4, "mean": 0.006, "lo": 0.001, "hi": 0.011, "raw_label": "UNRESOLVED_INTERVAL", "label": "BOTH_SATURATED",
               "both_saturated": True}
        for r2 in (sat, dict(sat, lo=-0.011, hi=-0.001, mean=-0.006)):
            P = AN.score_predictions({}, {"R2": {"pairs": {"Q2:PM-ref_vs_SG-stock": r2}}}, {})["P-3"]
            self.assertEqual(P["score"], "UNSCORABLE")
            self.assertEqual(P["why"], "BOTH_SATURATED")
        unsat = dict(sat, label="UNRESOLVED_INTERVAL", both_saturated=False)
        self.assertEqual(AN.score_predictions({}, {"R2": {"pairs": {"Q2:PM-ref_vs_SG-stock": unsat}}}, {})["P-3"]["score"], "HIT(+)")


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
        for f in ("sb_job.sh", "ENGINE_DIGEST.txt", "R_SER_FORECAST_NANO.csv", "PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md",
                  "PREDICTION_phaseB_expected_effects_2026-11-02.md", "AUDIT_harness_stock_base_r1_2026-11-02.md",
                  "AUDIT_harness_stock_base_r2_2026-11-02.md"):          # rev2.1: sealed documents are pinned (re-audit r2 (a))
            self.assertIn(f"results/stock_baseline/{f}", d["files"])
        self.assertEqual(set(d["plans_sha"]), set(SP.JOBS))

    def _job(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        e.update(env)
        return subprocess.run(["bash", str(SB / "sb_job.sh")], env=e, capture_output=True, text=True, timeout=120)

    def test_job_refuses_unsealed_and_bad_job(self):
        self.assertEqual(self._job({"PDMUX_SB_JOB": "J1"}).returncode, 16)
        self.assertEqual(self._job({"PDMUX_SB_JOB": "JX", "PDMUX_SB_TEST_MODE": "1"}).returncode, 16)

    def test_job_accepts_J1_down(self):
        """rev3: J1-down is in the PDMUX_SB_JOB allow-list (passes the config gate; stops here at the engine-digest gate, rc 13)."""
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ED.txt"
            f.write_text("0" * 64 + "\n")
            r = self._job({"PDMUX_SB_JOB": "J1-down", "PDMUX_SB_TEST_MODE": "1", "PDMUX_SB_ENGINE_DIGEST_FILE": str(f),
                           "PDMUX_SB_OUT": f"{td}/raw", "PDMUX_SB_SUMMARY": f"{td}/sum"})
            self.assertEqual(r.returncode, 13, r.stderr)
            self.assertNotIn("ABORT_CONFIG", r.stderr)

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
