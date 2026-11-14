"""CPU tests of the pre-ladder per workload (results/motivation/mo_ladder*.py, 2026-11-12) and of the registered-values file of mo_spec.
No GPU, no engine: fake servers and a fake client whose outcome is a known function of rate / capacity.  The pre-ladder is NOT A
MEASUREMENT; these tests check that it is labelled so, that it does not wait for the cap measurement, and that its reading is the
registered pick rule.  TestMutants: each guard holds on the code as written and fails when it is broken."""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
MO = ENG / "results" / "motivation"
sys.path.insert(0, str(MO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mo_ladder as LD  # noqa: E402
import mo_ladder_analyze as LA  # noqa: E402
import mo_spec as MS  # noqa: E402
import mo_card as MC  # noqa: E402
import mo_runner as MR  # noqa: E402
import mo_loadgen as ML  # noqa: E402
import mo_analyze as MA  # noqa: E402
import mo_rehearsal as MH  # noqa: E402
import test_motivation_harness as TH  # noqa: E402

SP, MW, L = MS.SP, MS.MW, MS.MW.L
GRIDS = "sharegpt_first:2,openchat_rounds:2,arxiv_summ:0.5,interference_short:2"
ARMS = ("SG_MIXED_8192", "PDMUX_BUNDLE", "PDMUX_D16", "PDMUX_D44")
CAP = {"SG_MIXED": 4.0, "D16": 3.0, "D44": 6.0, "arxiv": 1.0}


class FakeStore:
    path = "fake"

    def input_ids(self, doc_id, n_input):
        return [1] * int(n_input)


def cap_of(arm, out_path):
    if "arxiv_summ" in str(out_path):
        return CAP["arxiv"]
    if "PDMUX-D16" in arm:
        return CAP["D16"]
    return CAP["D44"] if "PDMUX-D44" in arm else CAP["SG_MIXED"]


def replay(api, base_url, reqs, out_path, cap_fn=cap_of):
    """Client twin: every request of a run has TTFT 500 x^4 ms and ITL 20 x^2 ms, x = rate / capacity of the arm (rate from the run id in
    the record path).  The strictest SLO pair (1000, 40) therefore holds up to x = 2^(1/4), the loosest (5000, 100) up to 10^(1/4)."""
    arm = base_url.replace("fake://", "")
    m = re.search(r"_r([0-9.]+)_s\d+", str(out_path))
    x = (float(m.group(1)) / cap_fn(arm, out_path)) if m else 0.5                # the warm-up has no run id: light load
    ttft, itl = 500.0 * x ** 4, 20.0 * x ** 2
    out = []
    for q in reqs:
        n, at = int(q["output_tokens"]), float(q["arrival_s"])
        stamps = [1000.0 + at + ttft / 1000.0 + k * itl / 1000.0 for k in range(n)]
        out.append({"request_id": q["request_id"], "phase": q["phase"], "input_tokens": q["input_tokens"], "requested_output_tokens": n,
                    "observed_output_tokens": n, "scheduled_arrival_s": at, "sent_s": at, "completion_s": at + ttft / 1000.0 + n * itl / 1000.0,
                    "t_sent_mono": 1000.0 + at, "t_finished_mono": stamps[-1], "ttft_ms": ttft, "token_t_mono": stamps,
                    "token_itl_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])], "error": "", "error_kind": ""})
    by = {q["request_id"]: q for q in reqs}
    return float(max(2.0, (reqs[-1]["arrival_s"] if reqs else 0.0) + 1.0)), [ML.normalize(r, by) for r in out]


def run_ladder(td, grids=GRIDS, arms=ARMS, job="JL", tamper=None, rp=replay):
    plan = LD.make_plan(job, LD.parse_grids(grids), arms)
    tb = tamper or {}
    with mock.patch.object(MW, "store_for", lambda wl, token_dir=None: FakeStore() if MW.WORKLOADS[wl]["tokens"] else None):
        card = LD.build_card(plan, TH.POOL)
        out = Path(td) / "raw"
        res = LD.run_job(plan, card, out, lambda bdir, a: TH.FakeServer(bdir, a, tb.get(Path(bdir).name)), rp, TH.POOL,
                         py_version_fn=TH.fake_py, freeze_fn=lambda py: "whatever-the-image-has", warmup_s=2.0)
    return out, plan, card, res


# ---- 1. plan ---------------------------------------------------------------------------------------------------------------------------
class TestPlan(unittest.TestCase):
    def setUp(self):
        self.grids = LD.parse_grids(GRIDS)
        self.plan = LD.make_plan("JL", self.grids, ARMS)

    def test_ladder_shape_follows_the_earlier_capacity_ladders(self):
        self.assertEqual(self.grids["sharegpt_first"], {"rmin": 2.0, "factor": 1.35, "n": 10, "n_requests": 120})
        self.assertEqual(self.grids["arxiv_summ"], {"rmin": 0.5, "factor": 1.5, "n": 6, "n_requests": 48})
        self.assertEqual(LD.parse_grids("sharegpt_first:3:1.2:5")["sharegpt_first"], {"rmin": 3.0, "factor": 1.2, "n": 5, "n_requests": 120})
        for bad in ("unknown:2", "sharegpt_first", "sharegpt_first:2:1.3"):
            with self.assertRaises(ValueError):
                LD.parse_grids(bad)
        self.assertEqual(len(LD.SEEDS), 2)

    def test_not_blocked_by_the_cap_measurement_or_the_venv_digest(self):
        self.assertTrue(MS.is_pending(MS.default_fixed()["mem"]) and MS.is_pending(MS.default_fixed()["cap"]))
        self.assertEqual(LD.plan_problems(self.plan), [])
        self.assertEqual(MS.plan_problems(self.plan), [])                              # mo_spec dispatches by plan kind
        card = LD.build_card(self.plan, TH.POOL)
        self.assertEqual(MC.card_placeholders(card), [])
        self.assertTrue(all(a["expect"]["venv_freeze"] is None for a in card["arms"].values()))      # observed and recorded, not compared

    def test_provisional_cap_and_mem_are_stated_in_name_argv_and_plan(self):
        self.assertEqual(LD.PROVISIONAL, {"mem": 0.6, "cap": 48})
        for b in self.plan["boots"]:
            a, ctx = b["arm_def"], LD.TARGETS[b["target"]]["context"]
            self.assertIn(f"_preladder_cap48_mem0.6_ctx{ctx}", a["name"])
            self.assertEqual((TH._flag(a["cmd"], "--max-running-requests"), TH._flag(a["cmd"], "--mem-fraction-static"),
                              TH._flag(a["cmd"], "--context-length")), ("48", "0.6", str(ctx)))
            self.assertEqual(a["provisional"], LD.PROVISIONAL)
        self.assertEqual(self.plan["label"], LD.NOT_A_MEASUREMENT)
        self.assertIn("NOT_A_MEASUREMENT", self.plan["label"])

    def test_targets_and_arms(self):
        got = sorted((b["target"], b["arm_def"]["arm_key"]) for b in self.plan["boots"])
        want = sorted([(t, k) for t in ("sharegpt_first", "openchat_rounds", "arxiv_summ") for k in ("SG_MIXED_8192", "PDMUX_BUNDLE")]
                      + [("interference_short", k) for k in ("SG_MIXED_8192", "PDMUX_D16", "PDMUX_D44")])
        self.assertEqual(got, want)
        short = [b for b in self.plan["boots"] if b["target"] == "interference_short"]
        self.assertTrue(all(TH._flag(b["arm_def"]["cmd"], "--context-length") == "96000" and b["runs"][0]["workload"] == "sharegpt_first" for b in short))
        self.assertTrue(all(b["role"] == "pre_ladder" for b in self.plan["boots"]))     # no anchor boot, no split-check boot
        for b in self.plan["boots"]:
            g = self.grids[b["target"]]
            self.assertEqual(len(b["runs"]), g["n"] * 2)
            self.assertEqual({r["seed"] for r in b["runs"]}, set(L.PRE_LADDER_SEEDS))

    def test_seeds_are_disjoint_from_every_measurement_seed_and_registered(self):
        self.assertFalse(set(L.PRE_LADDER_SEEDS) & (set(L.SEEDS) | set(L.SPARE_SEEDS) | set(L.SECOND_BLOCK_SEEDS)))
        reg = json.loads((MO / "workloads" / "trace_registry.json").read_text())
        self.assertEqual(reg["pre_ladder_seeds"], list(L.PRE_LADDER_SEEDS))
        for w in ("sharegpt_first", "openchat_rounds", "arxiv_summ"):
            self.assertTrue(all(str(s) in reg["master_hashes"][w] for s in L.PRE_LADDER_SEEDS))
        self.assertEqual(MW.G.seal_data()["master_hashes"], reg["master_hashes"])

    def test_order_is_seeded(self):
        o = LD.boot_order("A", self.grids, ARMS)
        self.assertEqual(o, LD.boot_order("A", self.grids, ARMS))
        self.assertNotEqual(o, LD.boot_order("B", self.grids, ARMS))
        self.assertNotEqual(o, LD.boot_keys(self.grids, ARMS))

    def test_checker_rejects_each_mutation(self):
        def mut(fn):
            p = copy.deepcopy(self.plan)
            fn(p)
            p["boot_order"] = [b["arm"] for b in p["boots"]]
            return " | ".join(LD.plan_problems(p))
        self.assertIn("registered permutation", mut(lambda p: p["boots"].insert(1, p["boots"].pop(0))))
        self.assertIn("ladder rungs x pre-ladder seeds", mut(lambda p: p["boots"][0]["runs"].pop()))
        self.assertIn("provisional cap / mem", mut(lambda p: p["boots"][0]["arm_def"]["cmd"].__setitem__(
            p["boots"][0]["arm_def"]["cmd"].index("--max-running-requests") + 1, "256")))
        self.assertIn("arm name does not state", mut(lambda p: p["boots"][0]["arm_def"].update(name="x")))
        self.assertIn("NOT_A_MEASUREMENT", mut(lambda p: p.update(label="")))
        self.assertIn("disjoint from every measurement seed", mut(lambda p: p.update(seeds=list(L.SEEDS[:2]))))
        self.assertIn("instrumentation env", mut(lambda p: next(b for b in p["boots"] if b["arm_def"]["tree"] == "patched")["arm_def"]["env"]
                                                 .update(PDMUX_TELEMETRY_PATH="x")))
        self.assertIn("none of the arms", " | ".join(LD.plan_problems(LD.make_plan("J", LD.parse_grids("interference_short:2"), ("VLLM",)))))

    def test_arm_choice_has_no_default_and_its_cost_is_computed(self):
        self.assertEqual(LD.main(["plan", "--grids", GRIDS]), 16)
        opt = LD.options(self.grids)
        self.assertEqual(sorted(opt), sorted(LD.ARM_OPTIONS))
        self.assertGreater(opt["all_arms"]["total_h"], opt["baseline_and_bundle"]["total_h"])
        self.assertGreater(opt["baseline_and_bundle"]["total_h"], opt["primary_baseline_only"]["total_h"])
        self.assertEqual(opt["all_arms"]["boots"], 3 * 7 + 5)
        e = LD.estimate(self.plan)
        self.assertAlmostEqual(e["total_h"], (e["load_h"] * 60 + e["drain_min"] + e["boot_min"] + 20) / 60, delta=0.02)


# ---- 2. Job through the fakes -> reading -------------------------------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.out, cls.plan, cls.card, cls.res = run_ladder(cls.td.name)
        cls.an = LA.analyze(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_job_ran_without_a_rehearsal_record_and_is_labelled(self):
        self.assertEqual(MR.exit_code(self.res), 0)
        self.assertEqual(json.loads((self.out / "RUN_DONE.json").read_text())["label"], LD.NOT_A_MEASUREMENT)
        self.assertEqual(self.an["label"], LD.NOT_A_MEASUREMENT)
        self.assertEqual(self.an["provisional"], {"mem": 0.6, "cap": 48})
        job = MA.load_job(self.out)
        self.assertEqual({r["boot_role"] for r in job["runs"]}, {"pre_ladder"})
        self.assertEqual(job["boots"][0]["observed"]["venv_freeze"], "whatever-the-image-has")      # recorded, not compared

    def test_brackets_follow_the_pick_rule_at_the_strictest_and_loosest_slo(self):
        self.assertEqual((LA.STRICT, LA.LOOSE), ((1000.0, 40.0), (5000.0, 100.0)))
        row = {(r["target"], r["arm_key"]): r for r in self.an["rows"]}
        ladder = LD.rates_of(self.plan["grids"]["sharegpt_first"])
        mixed = row[("sharegpt_first", "SG_MIXED_8192")]                            # capacity 4: strict holds to 4.76, loose to 7.11
        self.assertEqual((mixed["strict"]["last_pass"], mixed["strict"]["first_fail"]), (ladder[2], ladder[3]))
        self.assertEqual((mixed["loose"]["last_pass"], mixed["loose"]["first_fail"]), (ladder[4], ladder[5]))
        bundle = row[("sharegpt_first", "PDMUX_BUNDLE")]                            # capacity 6: strict to 7.14, loose to 10.67
        self.assertEqual((bundle["strict"]["last_pass"], bundle["loose"]["first_fail"]), (ladder[4], ladder[6]))
        self.assertTrue(all(r["strict"]["status"] == "BRACKETED" for r in self.an["rows"]))

    def test_envelope_contains_every_arm_and_gives_the_grids(self):
        t = self.an["targets"]["sharegpt_first"]
        ladder = t["ladder"]
        self.assertEqual((t["status"], t["envelope"]), ("BRACKETED", [ladder[2], ladder[6]]))
        self.assertEqual((t["lowest_strict_arm"], t["highest_loose_arm"]), ("SG_MIXED_8192", "PDMUX_BUNDLE"))
        g = t["grids"]["serving_cap_mem"]
        self.assertEqual((g["n"], g["rmin"], g["status"]), (4, ladder[2], "PROPOSED"))
        self.assertAlmostEqual(g["rates"][-1], ladder[6], delta=1e-3)                # the base grid ends at the envelope's upper end
        self.assertEqual(g["rates"], MS.make_grid(g["rmin"], g["factor"], 4)["rates"])   # the registered grid form
        self.assertEqual(len(g["extension_rates"]), MS.EXT_COUNT)
        self.assertEqual(t["grids"]["stage1"], {"n": LA.STAGE1_N, "status": "PENDING_N"})
        self.assertTrue(MS.is_pending(self.an["pending"]["stage1_n"]))
        six = LA.analyze(self.out, stage1_n=6)["targets"]["sharegpt_first"]["grids"]["stage1"]
        self.assertEqual((six["n"], len(six["rates"]), six["rmin"]), (6, 6, ladder[2]))
        self.assertEqual(self.an["targets"]["arxiv_summ"]["status"], "BRACKETED")

    def test_interference_short_rates(self):
        t = self.an["targets"]["interference_short"]
        ir = t["interference_rates"]
        ladder = t["ladder"]
        self.assertEqual(ir["limiting_arm"], "PDMUX_D16")                            # capacity 3: the primary SLO holds to 4.69
        self.assertEqual(ir["eligible_rungs"], ladder[:3])
        self.assertEqual(ir["rehearsal_rate_candidate"], ladder[0])
        self.assertEqual(ir["candidate_unconfirmed"], ladder[:3])
        self.assertTrue(MS.is_pending(ir["rule"]))                                    # how three rates are chosen is not registered
        self.assertNotIn("interference_rates", self.an["targets"]["sharegpt_first"])

    def test_outputs_carry_the_label(self):
        p = LA.write_outputs(self.an, Path(self.td.name) / "an")
        lines = p.read_text().splitlines()
        self.assertIn("NOT_A_MEASUREMENT", lines[0])
        self.assertEqual(len(lines), 2 + len(self.plan["boots"]))
        self.assertEqual(LA.main([str(self.out), "--out", str(Path(self.td.name) / "an2"), "--stage1-n", "6"]), 0)

    def test_measurement_analyzers_refuse_a_pre_ladder(self):
        with self.assertRaises(MA.JobInvalid):
            MA.analyze_job(self.out)                                                  # no anchor, not a grid Job: never a verdict table

    def test_two_jobs_are_not_merged(self):
        job = MA.load_job(self.out)
        job["runs"][0]["job_name"] = "another-job"
        with self.assertRaises(MA.JobInvalid) as cm:
            LA.analyze(job)
        self.assertIn("never merged", str(cm.exception))


class TestEdges(unittest.TestCase):
    def test_ladder_that_does_not_bracket_gives_no_grid(self):
        with tempfile.TemporaryDirectory() as td:
            out, _, _, _ = run_ladder(td, grids="sharegpt_first:6:1.35:3", arms=("SG_MIXED_8192", "PDMUX_BUNDLE"))
            t = LA.analyze(out)["targets"]["sharegpt_first"]                         # 6 is above the strict boundary of the mixed arm
            self.assertEqual((t["status"], t["grids"]), ("NOT_BRACKETED", None))
            self.assertIn("EXTEND_DOWN", t["extend"])
        with tempfile.TemporaryDirectory() as td:
            out, _, _, _ = run_ladder(td, grids="sharegpt_first:1:1.35:3", arms=("SG_MIXED_8192",))
            t = LA.analyze(out)["targets"]["sharegpt_first"]
            self.assertEqual((t["status"], t["extend"]), ("NOT_BRACKETED", ["EXTEND_UP"]))

    def test_failed_boot_loses_only_its_target(self):
        plan = LD.make_plan("JL", LD.parse_grids(GRIDS), ARMS)
        target = next(b["boot_id"] for b in plan["boots"] if b["target"] == "openchat_rounds")
        with tempfile.TemporaryDirectory() as td:
            out, _, _, res = run_ladder(td, tamper={target: {"status": "died"}})
            self.assertEqual(MR.exit_code(res), MR.EXIT_BOOT_FAIL)
            an = LA.analyze(out)
            self.assertEqual(an["targets"]["openchat_rounds"]["status"], "INCOMPLETE")
            self.assertEqual(an["targets"]["sharegpt_first"]["status"], "BRACKETED")

    def test_card_tamper_aborts_and_missing_tokens_abort_before_any_boot(self):
        plan = LD.make_plan("JL", LD.parse_grids(GRIDS), ARMS)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:
                run_ladder(td, tamper={plan["boots"][0]["boot_id"]: {"banner": {"max_running_requests": 256, "max_mamba_cache_size": 256}}})
            self.assertIn("[cap]", cm.exception.reason)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"PDMUX_MO_TOKEN_DIR": td}):
            card = LD.build_card(plan, TH.POOL)
            with self.assertRaises(MR.JobAbort) as cm:
                LD.run_job(plan, card, Path(td) / "raw", TH.make_server(), replay, TH.POOL, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze)
            self.assertEqual(cm.exception.status, "ABORT_TOKENS")
            self.assertFalse((Path(td) / "raw" / "boots").exists())

    def test_harness_digest_covers_the_ladder_modules(self):
        names = {f.name for f in MH._harness_files()}
        self.assertTrue({"mo_ladder.py", "mo_ladder_analyze.py"} <= names)
        self.assertNotIn("registered_fixed.json", names)


# ---- 3. registered values are data, not code -------------------------------------------------------------------------------------------------
class TestRegisteredValues(unittest.TestCase):
    def write(self, td, values):
        p = Path(td) / "registered_fixed.json"
        p.write_text(json.dumps({"values": values}))
        return p

    def test_committed_file_registers_the_venv_digest_only(self):
        """2026-11-15: the py3.12 venv freeze digest read by fixed-batch rehearsal 7 (and rehearsals 4 / 5, pre-ladder run 3)."""
        self.assertEqual(MS.registered_values(), {"sg_venv_freeze": "8a6b00442c0f761459b81d63aba5ab171b767e3f01eb4d2e2605ff8cb6ec8be3"})
        f = MS.default_fixed()
        self.assertTrue(all(MS.is_pending(f[k]) for k in MS.REGISTERABLE if k != "sg_venv_freeze"))
        self.assertFalse(MS.is_pending(f["sg_venv_freeze"]))

    def test_a_registered_value_replaces_its_placeholder_without_touching_the_code_digest(self):
        d0 = MH.harness_digest()
        with tempfile.TemporaryDirectory() as td:
            p = self.write(td, {"sg_venv_freeze": {"value": "a" * 64, "source": "rehearsal record of Job X"},
                                "mem": {"value": 0.75, "source": "cap measurement Job Y"}, "cap": {"value": 128, "source": "cap measurement Job Y"}})
            with mock.patch.object(MS, "REGISTERED_FILE", p):
                f = MS.default_fixed()
                self.assertEqual((f["sg_venv_freeze"], f["mem"], f["cap"]), ("a" * 64, 0.75, 128))
                self.assertTrue(MS.is_pending(f["vllm_chunk"]))
                arm = MS.build_arm(MS.ARM_PARAMS[0])
                self.assertEqual(TH._flag(arm["cmd"], "--max-running-requests"), "128")
                self.assertEqual(MH.harness_digest(), d0)                              # the rehearsal record stays valid

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            for bad in ({"context": {"value": 1, "source": "x"}}, {"mem": {"value": 0.7}}, {"mem": 0.7},
                        {"cap": {"value": MS.pend("x"), "source": "y"}}):
                with self.assertRaises(ValueError):
                    MS.registered_values(self.write(td, bad))


# ---- 4. Job script ----------------------------------------------------------------------------------------------------------------------------
class TestJobScript(unittest.TestCase):
    P = MO / "mo_ladder_job.sh"

    def run_sh(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_MOLAD_")}
        e.update(env)
        return subprocess.run(["bash", str(self.P)], env=e, capture_output=True, text=True, timeout=120)

    def test_syntax_executable_env_names_label(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(self.P)]).returncode, 0)
        self.assertTrue(os.access(self.P, os.X_OK))
        txt = self.P.read_text()
        self.assertIn("8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858", txt)
        self.assertIn("NOT A MEASUREMENT", txt)
        names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:}]", txt))
        self.assertEqual({n for n in names if not n.startswith(("PDMUX_", "HF_HOME", "SGLANG_ENGINE_DEV", "BASH_SOURCE"))}, set())
        head = "\n".join(ln for ln in txt.splitlines() if ln.startswith("#"))
        self.assertTrue(all(k.startswith("PDMUX_") for k in re.findall(r"--env ([A-Z0-9_]+)=", head)))

    def test_config_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td) / "w1.json"
            pool.write_text(json.dumps(TH.POOL))
            base = {"PDMUX_MOLAD_OUT": str(Path(td) / "o"), "PDMUX_MOLAD_TEST_MODE": "1", "PDMUX_MOLAD_W1_LENGTHS": str(pool)}
            self.assertEqual(self.run_sh(dict(base)).returncode, 16)
            self.assertEqual(self.run_sh(dict(base, PDMUX_MOLAD_JOB="J", PDMUX_MOLAD_GRIDS=GRIDS, PDMUX_MOLAD_BUDGET_MIN="5")).returncode, 16)   # no arms
            r = self.run_sh(dict(base, PDMUX_MOLAD_JOB="J", PDMUX_MOLAD_ARMS="SG_MIXED_8192", PDMUX_MOLAD_GRIDS="nope:2", PDMUX_MOLAD_BUDGET_MIN="5"))
            self.assertEqual(r.returncode, 16)
            self.assertIn("ABORT_CARD", r.stderr)


# ---- 5. mutants ---------------------------------------------------------------------------------------------------------------------------------
class TestMutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.out, cls.plan, _, _ = run_ladder(cls.td.name, grids="sharegpt_first:2,interference_short:2")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def ladder(self):
        return LD.rates_of(self.plan["grids"]["sharegpt_first"])

    def o_envelope(self):
        return LA.analyze(self.out)["targets"]["sharegpt_first"]["envelope"] == [self.ladder()[2], self.ladder()[6]]

    def o_grid_top(self):
        g = LA.analyze(self.out)["targets"]["sharegpt_first"]["grids"]["serving_cap_mem"]
        return abs(g["rates"][-1] - self.ladder()[6]) < 1e-3 and g["rmin"] == self.ladder()[2]

    def o_eligible(self):
        return LA.analyze(self.out)["targets"]["interference_short"]["interference_rates"]["eligible_rungs"] == self.ladder()[:3]

    def o_plan_clean_and_named(self):
        p = LD.make_plan("JM", LD.parse_grids(GRIDS), ARMS)
        return LD.plan_problems(p) == [] and all("_preladder_cap48_mem0.6_" in b["arm"] for b in p["boots"])

    def o_no_placeholder(self):
        return MC.card_placeholders(LD.build_card(LD.make_plan("JM", LD.parse_grids("sharegpt_first:2"), ARMS), TH.POOL)) == []

    def o_refused_by_measurement_analyzer(self):
        try:
            MA.analyze_job(self.out)
        except MA.JobInvalid:
            return True
        return False

    def test_each_mutant_is_caught(self):
        real_pick, real_grid = LA.RU.lambda_pick, LA.propose_grid

        def pick_any_passing_rung(points, attain=MS.CAP_ATTAIN):      # the highest passing point, not the ascending prefix
            ok = [r for r, g in points if g >= attain]
            return (max(ok), "BRACKETED") if ok else (None, "LADDER_BOTTOM_FAIL")

        def grid_one_rung_short(lo, hi, n):
            return real_grid(lo, hi, n) if MS.is_pending(n) else real_grid(lo, lo * (hi / lo) ** ((n - 1) / n), n)      # factor = (hi/lo)^(1/n)

        def noisy(api, base_url, reqs, out_path):                     # one seed of the top rungs passes by chance
            return replay(api, base_url, reqs, out_path, cap_fn=lambda arm, p: cap_of(arm, p) * (4.0 if str(p).endswith(f"_s{L.PRE_LADDER_SEEDS[0]}.jsonl")
                                                                                             and "_r16." in str(p) else 1.0))
        with tempfile.TemporaryDirectory() as td:
            out_noisy, _, _, _ = run_ladder(td, grids="sharegpt_first:2", arms=("SG_MIXED_8192", "PDMUX_BUNDLE"), rp=noisy)
            lad = self.ladder()

            def o_prefix():
                return LA.analyze(out_noisy)["targets"]["sharegpt_first"]["envelope"] == [lad[2], lad[6]]

            mutants = [
                ("strict SLO replaced by the loose one", mock.patch.object(LA, "STRICT", LA.LOOSE), self.o_envelope),
                ("loose SLO replaced by the strict one", mock.patch.object(LA, "LOOSE", LA.STRICT), self.o_envelope),
                ("boundary level lowered", mock.patch.object(MS, "SAT_HI", 0.0), self.o_eligible),
                ("pick rule without the ascending prefix", mock.patch.object(LA.RU, "lambda_pick", pick_any_passing_rung), o_prefix),
                ("proposed grid does not reach the envelope top", mock.patch.object(LA, "propose_grid", grid_one_rung_short), self.o_grid_top),
                ("provisional cap changed silently", mock.patch.dict(LD.PROVISIONAL, {"cap": 256}), self.o_plan_clean_and_named),
                ("pre-ladder on measurement seeds", mock.patch.object(LD, "SEEDS", tuple(L.SEEDS[:2])), self.o_plan_clean_and_named),
                ("pre-ladder waits for the venv digest", mock.patch.object(MC, "arm_expect", lambda a, fixed: {
                    **{f: None for f in MC.ARM_FIELDS}, "venv_freeze": MS.pend("x"), "argv": []}), self.o_no_placeholder),
                ("pre-ladder accepted as a grid Job", mock.patch.object(MA, "job_problems", lambda job, **kw: ([], [])),
                 self.o_refused_by_measurement_analyzer),
            ]
            self.assertIs(real_pick, LA.RU.lambda_pick)
            for name, patch, oracle in mutants:
                self.assertTrue(oracle(), f"oracle of '{name}' does not hold on the unmutated code")
                with patch:
                    try:
                        held = bool(oracle())
                    except Exception:  # noqa: BLE001
                        held = False
                self.assertFalse(held, f"mutant '{name}' survives: its oracle is an identity")
                self.assertTrue(oracle(), f"'{name}': the patch leaked")


if __name__ == "__main__":
    unittest.main()
