"""CPU tests of the LooGLE interference cell (results/motivation/mo_intf_*.py, mo_workload interference pair; 2026-11-12).  No GPU, no
engine: fake servers and a fake client with a known per-request outcome profile.  Nothing here shows the cell works on a GPU.

TestMutants: every guard listed there holds on the code as written and FAILS when the guard is replaced by its broken variant."""
import copy
import hashlib
import json
import math
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
import mo_intf_spec as IS  # noqa: E402
import mo_intf_analyze as IA  # noqa: E402
import mo_intf_runner as IR  # noqa: E402
import mo_spec as MS  # noqa: E402
import mo_card as MC  # noqa: E402
import mo_runner as MR  # noqa: E402
import mo_schema as SC  # noqa: E402
import mo_loadgen as ML  # noqa: E402
import mo_rehearsal as MH  # noqa: E402
import mo_analyze as MA  # noqa: E402
import test_motivation_harness as TH  # noqa: E402

SP, MW, L, G = MS.SP, MS.MW, MS.MW.L, MS.MW.G
RU_V = IA.V                                   # the sealed verdict module (ph_verdict) the analyzers share
RATES = [2.0, 3.0, 4.0]
TOK_DIR = MO / "workloads" / "tokens"
HAVE_LOOGLE = (TOK_DIR / MW.WORKLOADS[IS.WITH]["tokens"]["name"]).exists()
REH = lambda: dict(MS.REHEARSAL_FIXED)       # noqa: E731


class FakeStore:
    """Token store twin: every doc_id resolves to n placeholder ids (the real file is exercised by TestRealTokens when present)."""
    path = "fake"

    def input_ids(self, doc_id, n_input):
        return [1] * int(n_input)


def _u(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def make_replay(pfn, sent=None):
    """Client twin with a known profile: a request is bad (TTFT 4.5 s, ITL 90 ms) with probability pfn(arm name, with_long, request).
    The draw depends on the request id only, so the same short request gets the same draw in every arm and in both members of a pair."""
    def replay(api, base_url, reqs, out_path):
        arm = base_url.replace("fake://", "")
        with_long = any(q["phase"] == MW.LONG_PHASE for q in reqs)
        out = []
        for q in reqs:
            bad = _u(q["request_id"]) < pfn(arm, with_long, q)
            ttft, itl = (4500.0, 90.0) if bad else (700.0, 25.0)
            n, at = int(q["output_tokens"]), float(q["arrival_s"])
            stamps = [1000.0 + at + ttft / 1000.0 + k * itl / 1000.0 for k in range(n)]
            out.append({"request_id": q["request_id"], "phase": q["phase"], "input_tokens": q["input_tokens"], "requested_output_tokens": n,
                        "observed_output_tokens": n, "scheduled_arrival_s": at, "sent_s": at, "completion_s": at + ttft / 1000.0 + n * itl / 1000.0,
                        "t_sent_mono": 1000.0 + at, "t_finished_mono": stamps[-1], "ttft_ms": ttft, "token_t_mono": stamps,
                        "token_itl_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])], "error": "", "error_kind": ""})
        if sent is not None:
            sent.extend(reqs)
        srv = TH.FAKE_SERVERS.get(base_url)
        if srv is not None:
            srv.emit_steps()
        by = {q["request_id"]: q for q in reqs}
        return float(max(2.0, (reqs[-1]["arrival_s"] if reqs else 0.0) + 1.0)), [ML.normalize(r, by) for r in out]
    return replay


def d_of(arm):
    m = re.search(r"PDMUX-D(\d+)", arm)
    return int(m.group(1)) if m else None


def profile(short_with, long_with=None):
    """pfn: bad probability of the input<=512 class WITH LooGLE by D (None = baseline), 0 everywhere else unless long_with gives the
    bad probability of the LooGLE requests by D."""
    def pfn(arm, with_long, q):
        d = d_of(arm)
        if not with_long:
            return 0.0
        if q["phase"] == MW.LONG_PHASE:
            return (long_with or {}).get(d, 0.0)
        return short_with.get(d, 0.0) if q["input_tokens"] <= 512 else 0.0
    return pfn


SHORT_DURATION = 40.0


def run_block(td, block, pfn, job=None, tamper=None, rates=RATES, sent=None):
    """One measurement block through the fakes (cells shortened to 40 s so the test stays small; the registered 300 s is plan-tested)."""
    job = job or f"JI{block}"
    with mock.patch.object(IS, "DURATION_S", SHORT_DURATION), mock.patch.object(MS, "default_fixed", REH), \
            mock.patch.object(MW, "store_for", lambda wl, token_dir=None: FakeStore() if wl in MW.INTERFERENCE else None):
        plan = IS.make_plan(job, rates, block)
        card = MC.build_card(plan, TH.POOL)
        out = Path(td) / f"raw{block}"
        tb = tamper or {}
        res = MR.run_job(plan, card, out, lambda bdir, a: TH.FakeServer(bdir, a, tb.get(Path(bdir).name)), make_replay(pfn, sent), TH.POOL,
                         gate_fn=lambda p: None, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze, warmup_s=2.0)
    return out, plan, res


SCENARIO = profile({16: 0.30, 24: 0.06})          # D16 starves the short class, D24 is hit but stays above 0.90, D34 / D44 untouched


# ---- 1. arms -----------------------------------------------------------------------------------------------------------------------------
class TestArms(unittest.TestCase):
    def test_five_arms_no_d64(self):
        self.assertEqual(IS.D_GRID, (16, 24, 34, 44))
        self.assertEqual(IS.ARM_KEYS, ("PDMUX_D16", "PDMUX_D24", "PDMUX_D34", "PDMUX_D44", "SG_MIXED_8192"))
        with self.assertRaises(ValueError):
            IS.d_arm(64, IS.fixed_values(True))

    def test_d_arm_is_the_registered_bundle_with_only_d_changed(self):
        fixed = IS.fixed_values(True)
        bundle = MS.build_arm(next(p for p in MS.ARM_PARAMS if p["key"] == MS.BUNDLE_KEY), fixed)
        for d in IS.D_GRID:
            a = IS.d_arm(d, fixed)
            self.assertEqual(a["cmd"], bundle["cmd"])                                   # every flag identical
            self.assertEqual({k for k in a["env"] if a["env"][k] != bundle["env"].get(k)}, set() if d == 44 else {"PDMUX_R2_FIXED_DSM"})
            self.assertEqual(a["env"]["PDMUX_R2_FIXED_DSM"], str(d))
            self.assertEqual(a["env"], SP.P.arm_env(SP.P.static_arm(d)))                 # the sealed policy harness's non-sticky static D env
            self.assertEqual((a["tree"], a["python"], a["D"]), ("patched", MS.PY312, d))
            for part in (f"PDMUX-D{d}-", "nonsticky", "budget65536", "chunk-1", "piecewiseOFF", "overlapOFF", "bundle"):
                self.assertIn(part, a["name"])
            self.assertEqual(a["name"], MS.bundle_name(a["params"]))
        self.assertEqual(IS.d_arm(44, fixed)["name"], bundle["name"])                 # D44 IS the registered bundle arm
        self.assertEqual(len({IS.d_arm(d, fixed)["name"] for d in IS.D_GRID}), 4)

    def test_every_boot_has_the_same_96k_context(self):
        plan = IS.make_plan("J", RATES, 1, rehearsal=True)
        for b in plan["boots"]:
            self.assertEqual(TH._flag(b["arm_def"]["cmd"], "--context-length"), "96000", b["boot_id"])
        self.assertEqual(IS.CONTEXT, L.LOOGLE_CTX)
        rows = MW.long_pool()
        self.assertLess(max(i + o for _r, i, o in rows), IS.CONTEXT)                 # every LooGLE request fits the boot context

    def test_baseline_is_the_registered_primary_baseline(self):
        a = IS.build_arm(IS.BASELINE_KEY, IS.fixed_values(True))
        self.assertEqual((a["tree"], a["env"]), ("pristine", {}))
        self.assertIn("--enable-mixed-chunk", a["cmd"])
        self.assertEqual(a["name"], MS.name_of(MS.PRIMARY_KEY))


# ---- 2. pair / plan ---------------------------------------------------------------------------------------------------------------------------
class TestPairAndPlan(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(MS, "default_fixed", REH)
        p.start()
        self.addCleanup(p.stop)
        self.plan = IS.make_plan("JP", RATES, 1)

    def segs(self, wl, rate=3.0, seed=L.SEEDS[0], dur=300.0):
        return MW.build_segments({"run_id": "x", "workload": wl, "rate": rate, "seed": seed, "duration_s": dur}, None)[0]["requests"]

    def test_pair_shares_the_short_stream_and_differs_by_the_long_stream_only(self):
        w, wo = self.segs(IS.WITH), self.segs(IS.WITHOUT)
        long = [r for r in w if r["phase"] == MW.LONG_PHASE]
        self.assertEqual([r for r in w if r["phase"] != MW.LONG_PHASE], wo)             # byte-identical short records
        self.assertEqual(len(long), round(0.05 / 0.95 * 3.0 * 300.0))
        self.assertTrue(all("doc_id" in r and r["input_tokens"] > 2048 for r in long))
        self.assertTrue(all("doc_id" not in r for r in wo))
        self.assertAlmostEqual(len(long) / len(w), 0.05, delta=0.01)
        self.assertEqual(MW.n_interference(L.SEEDS[0], 3.0, 300.0, True), len(w))

    def test_both_members_report_the_same_pool_facts_and_token_file(self):
        f = [MW.pool_facts(wl, TH.POOL, MC.pool_facts) for wl in IS.MEMBERS]
        self.assertEqual(f[0], f[1])
        self.assertEqual(f[0]["token_file_sha256"], "9586fbbeeeb4117ef0c14b5dd6e7981bad26a320716b78d0531fa082807ceb17")
        self.assertEqual(MW.WORKLOADS[IS.WITH]["tokens"], MW.WORKLOADS[IS.WITHOUT]["tokens"])

    def test_full_block(self):
        self.assertEqual(IS.plan_problems(self.plan), [])
        self.assertEqual(MS.plan_problems(self.plan), [])                              # mo_spec dispatches by plan kind
        roles = [b["role"] for b in self.plan["boots"]]
        self.assertEqual(roles, ["anchor"] + ["meas"] * 5 + ["split_check"] * 4 + ["anchor"])
        for b in self.plan["boots"]:
            if b["role"] == "meas":
                self.assertEqual(len(b["runs"]), 3 * 4 * 2)                            # rates x seeds x (with, without), ONE boot
                self.assertEqual({r["duration_s"] for r in b["runs"]}, {300.0})
        self.assertEqual([b["arm_def"]["D"] for b in self.plan["boots"] if b["role"] == "split_check"], [16, 24, 34, 44])
        self.assertEqual(self.plan["seeds"], list(L.SEEDS))

    def test_same_trace_for_every_arm(self):
        card = MC.build_card(self.plan, TH.POOL)
        by = {}
        for b in self.plan["boots"]:
            if b["role"] == "meas":
                for r in b["runs"]:
                    by.setdefault((r["workload"], r["rate"], r["seed"]), set()).add(card["trace"][r["run_id"]]["hash"])
        self.assertEqual(len(by), 24)
        self.assertTrue(all(len(v) == 1 for v in by.values()))

    def test_two_blocks_use_disjoint_registered_seeds_and_one_design(self):
        p2 = IS.make_plan("JP2", RATES, 2)
        self.assertEqual(IS.plan_problems(p2), [])
        self.assertEqual(p2["seeds"], [20261117, 20261118, 20261119, 20261120])
        self.assertFalse(set(p2["seeds"]) & set(self.plan["seeds"]))
        self.assertFalse(set(p2["seeds"]) & set(L.SPARE_SEEDS))                         # the spare seeds keep their role
        self.assertEqual(p2["design_sha"], self.plan["design_sha"])
        self.assertNotEqual(IS.make_plan("JP3", [2.0, 3.0, 5.0], 2)["design_sha"], self.plan["design_sha"])
        reg = json.loads((MO / "workloads" / "trace_registry.json").read_text())
        self.assertEqual(reg["second_block_seeds"], p2["seeds"])
        for w in ("sharegpt_first", "loogle_longdep"):
            self.assertTrue(all(str(s) in reg["master_hashes"][w] for s in p2["seeds"]))
        self.assertEqual(G.seal_data()["master_hashes"], reg["master_hashes"])

    def test_order_is_a_seeded_permutation(self):
        self.assertEqual(IS.boot_order("A"), IS.boot_order("A"))
        self.assertNotEqual(IS.boot_order("A"), IS.boot_order("B"))
        self.assertEqual(sorted(IS.boot_order("A")), sorted(IS.ARM_KEYS))

    def test_checker_rejects_each_mutation(self):
        i = next(k for k, b in enumerate(self.plan["boots"]) if b["role"] == "meas" and b["arm_def"].get("D") == 24)

        def mut(fn):
            p = copy.deepcopy(self.plan)
            fn(p)
            p["boot_order"] = [b["arm"] for b in p["boots"]]
            return " | ".join(IS.plan_problems(p))

        def set_d64(p):
            a = p["boots"][i]["arm_def"]
            a["params"]["D"] = a["D"] = 64
            a["env"]["PDMUX_R2_FIXED_DSM"] = "64"
            a["name"] = p["boots"][i]["arm"] = MS.bundle_name(a["params"])

        self.assertIn("registered permutation", mut(lambda p: p["boots"].insert(i + 1, p["boots"].pop(i))))
        self.assertIn("registered permutation", mut(set_d64))
        self.assertIn("only PDMUX_R2_FIXED_DSM changed", mut(lambda p: p["boots"][i]["arm_def"]["cmd"].append("--enable-mixed-chunk")))
        self.assertIn("only PDMUX_R2_FIXED_DSM changed", mut(lambda p: p["boots"][i]["arm_def"]["env"].update(PDMUX_STICKY_PARTITION="1")))
        self.assertIn("only PDMUX_R2_FIXED_DSM changed", mut(lambda p: p["boots"][i]["arm_def"]["env"].update(PDMUX_R2_FIXED_DSM="44")))
        self.assertIn("full-block", mut(lambda p: p["boots"][i]["runs"].pop()))
        self.assertIn("full-block", mut(lambda p: p["boots"][i].update(runs=[r for r in p["boots"][i]["runs"] if r["workload"] == IS.WITH])))
        self.assertIn("context is not 96000", mut(lambda p: p["boots"][1]["arm_def"]["cmd"].__setitem__(
            p["boots"][1]["arm_def"]["cmd"].index("--context-length") + 1, "20480")))
        self.assertIn("one split-check boot per D", mut(lambda p: p["boots"].pop(-2)))
        self.assertIn("anchor boot must be first and last", mut(lambda p: p["boots"].append(p["boots"].pop(0))))
        self.assertIn("registered seeds of block", mut(lambda p: p.update(seeds=p["seeds"][:3])))
        self.assertIn("3 distinct short rates", mut(lambda p: p["grid"].update(rates=p["grid"]["rates"][:2])))
        self.assertIn("not a registered interference cell", mut(lambda p: p["boots"][i]["runs"][0].update(n_requests=1)))
        self.assertIn("duration", mut(lambda p: p.update(duration_s=100.0)))

    def test_measurement_plan_waits_for_the_cap_measurement(self):
        with mock.patch.object(MS, "default_fixed", lambda: {**MS.REHEARSAL_FIXED, "mem": MS.pend("x"), "cap": MS.pend("y")}):
            p = IS.make_plan("JP", RATES, 1)
            self.assertTrue(any("PLACEHOLDER" in x for x in IS.plan_problems(p)))
            self.assertEqual(IS.plan_problems(IS.make_plan("JP", RATES, 1, rehearsal=True)), [])

    def test_time_estimate(self):
        e = IS.estimate(self.plan)
        self.assertEqual((e["boots"], e["runs"]), (11, 5 * 24 + 4 + 2))
        self.assertAlmostEqual(e["load_h"], 126 * 300 / 3600, delta=0.01)
        self.assertAlmostEqual(e["total_h"], (126 * 300 / 60 + 126 * SP.DRAIN_MIN + 11 * 6 + 20) / 60, delta=0.01)


# ---- 3. one block through the fakes, two blocks combined -----------------------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.sent = []
        cls.out1, cls.plan1, cls.res1 = run_block(cls.td.name, 1, SCENARIO, sent=cls.sent)
        cls.out2, cls.plan2, cls.res2 = run_block(cls.td.name, 2, SCENARIO)
        cls.b1, cls.b2 = IA.analyze_block(cls.out1), IA.analyze_block(cls.out2)
        cls.co = IA.combine_blocks([cls.b1, cls.b2])

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_block_is_valid_and_raw_follows_the_schema(self):
        self.assertEqual(MR.exit_code(self.res1), 0)
        job = MA.load_job(self.out1)
        self.assertEqual(len(job["runs"]), 126)
        self.assertEqual([m for r in job["runs"] for m in SC.validate_run(r)], [])
        self.assertEqual(self.b1["flags"], [])
        self.assertEqual(self.b1["split_checks"]["status"], "MATCH")
        self.assertEqual(sorted(self.b1["split_checks"]["by_D"]), ["D16", "D24", "D34", "D44"])

    def test_cached_request_rule_equals_the_sealed_one(self):
        job = MA.load_job(self.out1)
        reqs = [x for r in job["runs"][::9] for x in MA.reqs_of(r)]
        extra = [dict(reqs[0], error="x"), dict(reqs[0], ttft_ms=None), dict(reqs[0], token_itl_ms=[]), dict(reqs[0], token_itl_ms=[10.0] * 18 + [900.0] * 2)]
        self.assertGreater(len(reqs), 500)
        for x in reqs + extra:
            for t, i in MA.SLO_PAIRS:
                self.assertEqual(IA.good(dict(x), t, i), RU_V.req_good(x, t, i))
        self.assertEqual(IA.attainment_grid([dict(x) for x in reqs]), MA.attainment_grid(reqs))

    def test_long_requests_are_sent_with_token_ids_short_ones_without(self):
        long = [q for q in self.sent if q["phase"] == MW.LONG_PHASE]
        self.assertTrue(long and all(len(q["input_ids"]) == q["input_tokens"] for q in long))
        self.assertTrue(all("input_ids" not in q for q in self.sent if q["phase"] == "sharegpt_first"))

    def test_per_class_table_and_loogle_only_cell(self):
        pc = self.b1["per_class"]
        self.assertEqual(len(pc), 5 * 3 * 2 * 3)                                         # arms x rates x members x classes
        self.assertEqual({r["class"] for r in pc}, {"in<=512", "in<=2048", "in>2048"})
        row = next(r for r in pc if r["D"] == 16 and r["rate"] == 3.0 and r["with_long"] and r["class"] == "in<=512")
        self.assertAlmostEqual(row["attainment"], 0.70, delta=0.07)
        self.assertTrue(all(k in row for k in ("ttft_p50_ms", "ttft_p95_ms", "req_itl_p95_median_ms", "n")))
        lo = self.b1["loogle_only"]
        self.assertEqual(len(lo), 5 * 3)                                                # only the WITH member has LooGLE requests
        self.assertTrue(all(r["descriptive_only"] and r["n"] > 0 for r in lo))
        n_long = round(0.05 / 0.95 * 3.0 * SHORT_DURATION) * 4
        self.assertEqual(next(r for r in lo if r["D"] == 34 and r["rate"] == 3.0)["n"], n_long)

    def test_interference_reading(self):
        lab = {(r["D"], r["rate"]): r["label"] for r in self.co["interference"]}
        for rt in RATES:
            self.assertEqual(lab[(16, rt)], "INTERFERENCE")
            self.assertEqual(lab[(34, rt)], "NO_INTERFERENCE")
            self.assertEqual(lab[(44, rt)], "NO_INTERFERENCE")
            self.assertEqual(lab[(None, rt)], "NO_INTERFERENCE")                        # the baseline arm is read the same way
        r = next(x for x in self.co["interference"] if x["D"] == 16 and x["rate"] == 3.0)
        self.assertAlmostEqual(r["mean"], 0.30, delta=0.07)
        self.assertEqual((r["n"], r["n_blocks"], r["df"]), (8, 2, 6))
        self.assertTrue(r["t_lo"] < r["mean"] < r["t_hi"] and r["boot_lo"] <= r["mean"] <= r["boot_hi"])
        self.assertGreater(r["t_lo"], MS.ATT_MARGIN)

    def test_d_shift_reading(self):
        rows = {(r["with_long"], r["rate"], r["D"]): r for r in self.co["d_shift"]}
        for rt in RATES:
            self.assertIn(rows[(True, rt, 34)]["label"], ("BEST", "WITHIN"))
            self.assertIn(rows[(True, rt, 44)]["label"], ("BEST", "WITHIN"))
            self.assertEqual(rows[(True, rt, 16)]["label"], "BELOW_BEST")                # D16 loses ~30 % of the short class
            self.assertLess(rows[(True, rt, 16)]["ratio"], 0.97)
            self.assertLess(rows[(True, rt, 16)]["t_hi"], 0.0)
            self.assertTrue(all(rows[(False, rt, d)]["label"] in ("BEST", "WITHIN") for d in IS.D_GRID))   # without LooGLE every D is equal
        self.assertTrue(rows[(True, 3.0, 34)]["judged"] and not rows[(True, 3.0, 16)]["judged"])

    def test_starved_class_and_action_row(self):
        st = {(r["D"], r["with_long"], r["rate"], r["class"]): r for r in self.co["starved"]}
        self.assertTrue(st[(16, True, 3.0, "in<=512")]["starved"])
        self.assertFalse(st[(24, True, 3.0, "in<=512")]["starved"])                      # hit (0.94) but above 0.90
        self.assertFalse(st[(16, False, 3.0, "in<=512")]["starved"])
        for a in self.co["actions"]:
            self.assertEqual(a["row"], 3)                                                # starved at D16, resolved by the other D
            self.assertEqual(a["resolving_D"], [24, 34, 44])
            self.assertFalse(a["every_D_has_a_starved_class"])
        self.assertEqual(self.co["status"], "JUDGED")

    def test_loogle_only_cell_is_carried_not_judged(self):
        self.assertEqual(len(self.co["loogle_only"]), 2 * 15)
        self.assertNotIn("loogle", json.dumps(self.co["actions"]).lower())

    def test_block_alone_is_one_block(self):
        one = IA.combine_blocks([self.b1])
        self.assertEqual(one["status"], "ONE_BLOCK")
        self.assertEqual(one["interference"][0]["df"], 3)

    def test_outputs_and_cli(self):
        self.assertTrue(IA.write_block(self.b1, Path(self.td.name) / "ab").exists())
        self.assertEqual(len((Path(self.td.name) / "ab" / "loogle_only.csv").read_text().splitlines()), 16)
        p = IA.write_combined(self.co, Path(self.td.name) / "ac")
        self.assertEqual(len(p.read_text().splitlines()), 4)
        self.assertEqual(IA.main(["combine", str(self.out1), str(self.out2), "--out", str(Path(self.td.name) / "ad")]), 0)
        self.assertEqual(IA.main(["combine", str(self.out1), str(self.out1), "--out", str(Path(self.td.name) / "ae")]), 3)

    def test_combination_refusals(self):
        def bad(blocks):
            with self.assertRaises(MA.JobInvalid) as cm:
                IA.combine_blocks(blocks)
            return str(cm.exception)
        self.assertIn("same Job given twice", bad([self.b1, self.b1]))
        self.assertIn("share a seed", bad([self.b1, dict(self.b2, seeds=self.b1["seeds"])]))
        self.assertIn("different designs", bad([self.b1, dict(self.b2, design_sha="0" * 64)]))
        self.assertIn("rehearsal is not a block", bad([self.b1, dict(self.b2, rehearsal=True)]))
        self.assertIn("same block number", bad([self.b1, dict(self.b2, block=1)]))
        with self.assertRaises(MA.CrossJobMergeError):                                  # the raw-point rule is untouched
            MA.merge_points(MA.load_job(self.out1), MA.load_job(self.out2))

    def test_flagged_block_is_not_judged(self):
        self.assertEqual(IA.combine_blocks([self.b1, dict(self.b2, flags=["ANCHOR_DRIFT"])])["status"], "NOT_JUDGEABLE")

    def test_split_check_mismatch_of_one_d_flags_the_block(self):
        with tempfile.TemporaryDirectory() as td:
            out, plan, _ = run_block(td, 1, SCENARIO, job="JSC", tamper={"JSC_splitcheck_D24": {"green_d": 44}})
            b = IA.analyze_block(out)
            self.assertEqual(b["split_checks"]["by_D"]["D24"]["status"], "MISMATCH")
            self.assertEqual(b["split_checks"]["by_D"]["D34"]["status"], "MATCH")
            self.assertIn("SPLIT_CHECK_D24_MISMATCH", b["flags"])

    def test_card_tamper_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            target = next(b["boot_id"] for b in IS.make_plan("JCT", RATES, 1, rehearsal=True)["boots"] if b["role"] == "meas")
            with self.assertRaises(MR.JobAbort) as cm:
                run_block(td, 1, SCENARIO, job="JCT", tamper={target: {"banner": {"context_length": 20480}}})
            self.assertIn("[context]", cm.exception.reason)


# ---- 4. block combination statistics ----------------------------------------------------------------------------------------------------------
class TestBlockEstimate(unittest.TestCase):
    def test_block_level_shift_does_not_widen_the_interval(self):
        e = IA.block_estimate([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]], "t")
        self.assertAlmostEqual(e["mean"], 7.5)
        self.assertEqual((e["n"], e["n_blocks"], e["df"]), (8, 2, 6))
        se = math.sqrt(10.0 / 6 / 8)
        self.assertAlmostEqual(e["se"], se)
        self.assertAlmostEqual(e["t_hi"] - e["mean"], 2.447 * se, places=6)
        self.assertEqual(e["block_means"], [2.5, 12.5])
        same = IA.block_estimate([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], "t")
        self.assertAlmostEqual(same["se"], e["se"])                                     # the shift between blocks changes nothing

    def test_one_block_is_the_paired_t(self):
        e = IA.block_estimate([[0.1, 0.2, 0.3, 0.6]], "t")
        sd = math.sqrt(sum((v - 0.3) ** 2 for v in (0.1, 0.2, 0.3, 0.6)) / 3)
        self.assertAlmostEqual(e["se"], sd / 2)
        self.assertAlmostEqual(e["t_lo"], 0.3 - 3.182 * sd / 2, places=6)

    def test_bootstrap_is_seeded_and_stratified(self):
        a = IA.block_estimate([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]], "t")
        b = IA.block_estimate([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]], "t")
        self.assertEqual((a["boot_lo"], a["boot_hi"]), (b["boot_lo"], b["boot_hi"]))
        self.assertTrue(6.0 < a["boot_lo"] < 7.5 < a["boot_hi"] < 9.0)                  # resampled inside each block: never near 2.5 or 12.5

    def test_sign_agreement_and_empty(self):
        self.assertFalse(IA.block_estimate([[0.1, 0.2], [-0.1, -0.3]], "t")["blocks_same_sign"])
        self.assertTrue(IA.block_estimate([[0.1, 0.2], [0.1, 0.3]], "t")["blocks_same_sign"])
        self.assertEqual(IA.block_estimate([[], []])["n"], 0)

    def test_t_table(self):
        self.assertEqual((IA.T975[3], IA.T975[6], IA.T975[30]), (3.182, 2.447, 2.042))


# ---- 5. action rows on hand-built blocks ----------------------------------------------------------------------------------------------------------
def synth(att=None, good=None, rates=(3.0,), eps=0.004):
    """Two blocks of per-seed values.  att(d, with_long, cls) -> attainment, good(d, with_long) -> goodput; a small seed-dependent term
    keeps the paired differences from being exactly constant."""
    att = att or (lambda d, wl, c: 1.0)
    good = good or (lambda d, wl: 3.0)
    blocks = []
    for bi, seeds in ((1, IS.BLOCK_SEEDS[1]), (2, IS.BLOCK_SEEDS[2])):
        sv, vr = [], []
        for d in IS.D_GRID + (None,):
            key = IS.BASELINE_KEY if d is None else f"PDMUX_D{d}"
            for rt in rates:
                for wl in (True, False):
                    for k, s in enumerate(seeds):
                        j = eps * ((k + bi + (d or 0) // 10) % 3 - 1)
                        sv.append({"arm": key, "arm_key": key, "D": d, "rate": rt, "with_long": wl, "seed": s,
                                   "goodput_rps": good(d, wl) * (1 + j), "att": {c: min(1.0, att(d, wl, c) + j) for c in IA.CLASSES},
                                   "n": {c: 60 for c in IA.CLASSES}, "att_all": 1.0})
                vr.append({"arm_key": key, "D": d, "rate": rt, "eligible": True, "label_read": "A>B", "att_without": 1.0, "att_with": 1.0})
        blocks.append({"job": f"S{bi}", "kind": IS.KIND, "block": bi, "seeds": list(seeds), "rates": list(rates), "design_sha": "d", "rehearsal": False,
                       "flags": [], "seed_values": sv, "verdict_rows": vr, "loogle_only": []})
    return blocks


class TestActionRows(unittest.TestCase):
    def row(self, **kw):
        return IA.combine_blocks(synth(**kw))["actions"][0]

    def test_row1_d34_within_three_percent_everywhere(self):
        a = self.row(good=lambda d, wl: 3.0 * {16: 0.98, 24: 0.99, 34: 0.985, 44: 1.0}[d] if d else 3.0)
        self.assertEqual(a["row"], 1)                                                    # 1.5 % below the best D is not "more than 3 %"

    def test_row2_shift_needs_f_times_x(self):
        a = self.row(good=lambda d, wl: 3.0 * ({16: 1.0, 24: 0.99, 34: 0.90, 44: 0.85}[d] if wl else 1.0) if d else 3.0)
        self.assertEqual(a["row"], 2)                                                    # D16 loses nowhere, D34 loses 10 % with LooGLE
        self.assertAlmostEqual(a["x_D34_loss"], 0.10, delta=0.01)
        self.assertAlmostEqual(a["f_threshold"], 0.15, delta=0.02)                       # roadmap: x 10 % -> f 0.15
        self.assertEqual(a["below_best_by_D"]["D34"], ["with"])

    def test_row3_starved_and_some_d_resolves(self):
        a = self.row(att=lambda d, wl, c: 0.80 if (d in (34, 44) and wl and c == "in<=512") else 1.0)
        self.assertEqual((a["row"], a["resolving_D"]), (3, [16, 24]))

    def test_row4_starved_the_same_at_every_d(self):
        a = self.row(att=lambda d, wl, c: 0.70 if (wl and c == "in>2048") else 1.0)
        self.assertEqual(a["row"], 4)
        self.assertTrue(a["every_D_has_a_starved_class"])

    def test_row5_different_classes_starved_at_different_d(self):
        a = self.row(att=lambda d, wl, c: 0.70 if wl and c == ("in<=512" if d in (34, 44) else "in>2048") else 1.0)
        self.assertEqual(a["row"], 5)

    def test_row5_every_d_loses_on_one_side(self):
        a = self.row(good=lambda d, wl: 3.0 * (({16: 1.0, 24: 0.95, 34: 0.90, 44: 0.85} if wl else {16: 0.85, 24: 0.90, 34: 0.95, 44: 1.0})[d]) if d else 3.0)
        self.assertEqual(a["row"], 5)
        self.assertEqual(a["below_best_by_D"]["D16"], ["without"])

    def test_three_percent_needs_the_interval_too(self):
        noisy = synth(good=lambda d, wl: 3.0 * (0.90 if d == 34 and wl else 1.0) if d else 3.0, eps=0.2)   # 10 % below, but +-20 % seed noise
        r = next(x for x in IA.combine_blocks(noisy)["d_shift"] if x["D"] == 34 and x["with_long"])
        self.assertEqual(r["label"], "WITHIN")
        self.assertGreater(r["t_hi"], 0.0)

    def test_interference_needs_margin_eligibility_and_reading_rule(self):
        hit = lambda d, wl, c: 0.90 if (wl and c == "in<=512") else 1.0            # noqa: E731
        b = synth(att=hit)
        self.assertEqual({r["label"] for r in IA.combine_blocks(b)["interference"]}, {"INTERFERENCE"})
        b2 = synth(att=hit)
        for r in b2[1]["verdict_rows"]:
            r["label_read"] = "SLO_DEPENDENT"
        self.assertEqual({r["label"] for r in IA.combine_blocks(b2)["interference"]}, {"READING_RULE_NOT_PASSED"})
        b3 = synth(att=hit)
        for r in b3[0]["verdict_rows"]:
            r["eligible"] = False
        self.assertEqual({r["label"] for r in IA.combine_blocks(b3)["interference"]}, {"NOT_ELIGIBLE"})
        small = synth(att=lambda d, wl, c: 0.98 if (wl and c == "in<=512") else 1.0)     # 0.02 < ATT_MARGIN
        self.assertEqual({r["label"] for r in IA.combine_blocks(small)["interference"]}, {"NO_INTERFERENCE"})

    def test_registered_constants(self):
        c = IA.combine_blocks(synth())["constants"]
        self.assertEqual(c, {"ATT_MARGIN": 0.03, "SAT_HI": 0.98, "CAP_ATTAIN": 0.90, "GOODPUT_LOSS": 0.03, "MIN_CLASS_N": 20, "primary_slo": [3000.0, 60.0]})
        self.assertEqual(sorted(IA.ACTION_ROWS), [1, 2, 3, 4, 5])


# ---- 6. rehearsal ---------------------------------------------------------------------------------------------------------------------------------
class TestRehearsal(unittest.TestCase):
    def reh(self, td, tamper=None):
        tb = tamper or {}
        with mock.patch.object(MW, "store_for", lambda wl, token_dir=None: FakeStore() if wl in MW.INTERFERENCE else None):
            return IR.rehearse("JRI", TH.POOL, Path(td) / "raw", lambda bdir, a: TH.FakeServer(bdir, a, tb.get(Path(bdir).name)),
                               make_replay(SCENARIO), RATES, rehearsal_dir=Path(td) / "rec", digest="a" * 64, py_version_fn=TH.fake_py,
                               freeze_fn=TH.fake_freeze)

    def test_full_pass_over_the_real_arm_list(self):
        with tempfile.TemporaryDirectory() as td:
            rec = self.reh(td)
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertEqual(rec["kind"], IS.KIND)
            self.assertEqual(len(rec["arm_names"]), 5 + 4)                           # five arms + the four split-check arms
            self.assertEqual(rec["verdict_rows"], 5)
            self.assertEqual(rec["workloads_resolved"][IS.WITH], "OK")
            self.assertEqual(list(rec["observed_venv_freeze"]), ["rehearsal-any"])   # what the boots saw; a GPU pass records the real digest
            self.assertTrue((Path(td) / "rec" / ("a" * 16 + f".{IS.KIND}.json")).exists())
            self.assertTrue((Path(td) / "raw" / "analysis" / "per_class.csv").exists())
            with mock.patch.object(MS, "default_fixed", REH):
                self.assertIsNone(MH.launch_gate(IS.make_plan("JREAL", RATES, 1), Path(td) / "rec", "a" * 64))
                self.assertIsNone(MH.launch_gate(IS.make_plan("JREAL2", RATES, 2), Path(td) / "rec", "a" * 64))
                self.assertIn("NO_REHEARSAL_RECORD", MH.launch_gate(MS.make_plan("JG", TH.GRID3, fixed=REH()), Path(td) / "rec", "a" * 64))

    def test_failing_stage_is_named(self):
        with tempfile.TemporaryDirectory() as td:
            rec = self.reh(td, {"JRI_splitcheck_D34": {"banner": {"mem_fraction_static": 0.9}}})
            self.assertFalse(rec["passed"])
            self.assertIn("ABORT_CARD_MISMATCH", rec["reason"])

    def test_gate_needs_the_token_file_resolved(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"PDMUX_MO_TOKEN_DIR": td}):
            rec = IR.rehearse("JRT", TH.POOL, Path(td) / "raw", TH.make_server(), make_replay(SCENARIO), RATES, rehearsal_dir=Path(td) / "rec",
                              digest="b" * 64, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze)
            self.assertFalse(rec["passed"])
            self.assertIn("ABORT_TOKENS", rec["reason"])                              # before any boot
            self.assertFalse((Path(td) / "raw" / "boots").exists())

    def test_harness_digest_covers_the_interference_modules(self):
        names = {f.name for f in MH._harness_files()}
        self.assertTrue({"mo_intf_spec.py", "mo_intf_analyze.py", "mo_intf_runner.py", "gen_traces.py", "mw_lib.py"} <= names)


@unittest.skipUnless(HAVE_LOOGLE, "local LooGLE token file absent (results/motivation/workloads/tokens/)")
class TestRealTokens(unittest.TestCase):
    def test_loogle_ids_match_lengths(self):
        st = MW.TokenStore(IS.WITH, TOK_DIR)
        run = {"run_id": "x", "workload": IS.WITH, "rate": 3.0, "seed": L.SEEDS[0], "duration_s": 60.0}
        segs = MW.with_input_ids(MW.build_segments(run, None), st)
        long = [r for r in segs[0]["requests"] if r["phase"] == MW.LONG_PHASE]
        self.assertTrue(long and all(len(r["input_ids"]) == r["input_tokens"] for r in long))
        self.assertTrue(all("input_ids" not in r for r in segs[0]["requests"] if r["phase"] != MW.LONG_PHASE))


# ---- 7. Job script ----------------------------------------------------------------------------------------------------------------------------------
class TestJobScript(unittest.TestCase):
    P = MO / "mo_intf_job.sh"

    def run_sh(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_MOINTF_")}
        e.update(env)
        return subprocess.run(["bash", str(self.P)], env=e, capture_output=True, text=True, timeout=120)

    def test_syntax_executable_digest_env_names(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(self.P)]).returncode, 0)
        self.assertTrue(os.access(self.P, os.X_OK))
        txt = self.P.read_text()
        self.assertIn("8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858", txt)
        self.assertIn("export PDMUX_HEARTBEAT_S=0", txt)
        names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:}]", txt))
        self.assertEqual({n for n in names if not n.startswith(("PDMUX_", "HF_HOME", "SGLANG_ENGINE_DEV", "BASH_SOURCE"))}, set())

    def test_config_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            base = {"PDMUX_MOINTF_OUT": td, "PDMUX_MOINTF_TEST_MODE": "1"}
            self.assertEqual(self.run_sh(dict(base)).returncode, 16)
            ok = dict(base, PDMUX_MOINTF_MODE="run", PDMUX_MOINTF_JOB="J", PDMUX_MOINTF_BUDGET_MIN="5")
            self.assertEqual(self.run_sh(dict(ok)).returncode, 16)                                        # no rates: no default
            r = self.run_sh(dict(ok, PDMUX_MOINTF_RATES="2,3,4"))
            self.assertEqual(r.returncode, 16)
            self.assertIn("PDMUX_MOINTF_BLOCK", r.stderr)
            r = self.run_sh(dict(ok, PDMUX_MOINTF_RATES="2,3", PDMUX_MOINTF_BLOCK="1"))
            self.assertIn("three short rates", r.stderr)

    def test_run_refuses_while_mem_and_cap_are_placeholders(self):
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td) / "w1.json"
            pool.write_text(json.dumps(TH.POOL))
            r = self.run_sh({"PDMUX_MOINTF_OUT": str(Path(td) / "o"), "PDMUX_MOINTF_TEST_MODE": "1", "PDMUX_MOINTF_W1_LENGTHS": str(pool),
                             "PDMUX_MOINTF_MODE": "run", "PDMUX_MOINTF_JOB": "JSH", "PDMUX_MOINTF_BLOCK": "1", "PDMUX_MOINTF_RATES": "2,3,4",
                             "PDMUX_MOINTF_BUDGET_MIN": "5"})
            self.assertEqual(r.returncode, 19, r.stderr[-400:])
            self.assertIn("ABORT_PLAN", r.stderr + r.stdout)
            self.assertFalse((Path(td) / "o" / "raw" / "boots").exists())


class TestGridJobScriptEnvNames(unittest.TestCase):
    """mo_job.sh: launch.sh passes --env only for PDMUX_* names (the first header documented MO_<NAME>, which launch.sh refuses)."""
    P = MO / "mo_job.sh"

    def test_pdmux_prefixed_knobs_are_read(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("MO_", "PDMUX_MO_"))}
        with tempfile.TemporaryDirectory() as td:
            base = dict(e, PDMUX_MO_OUT=td, PDMUX_MO_TEST_MODE="1", PDMUX_MO_MODE="run", PDMUX_MO_WORKLOAD="openchat_rounds", PDMUX_MO_JOB="J",
                        PDMUX_MO_BUDGET_MIN="10")
            r = subprocess.run(["bash", str(self.P)], env=base, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 16)
            self.assertIn("PDMUX_MO_RMIN required", r.stderr)                         # mode / workload / job / budget were accepted
            r = subprocess.run(["bash", str(self.P)], env=dict(e, PDMUX_MO_OUT=td, PDMUX_MO_TEST_MODE="1"), capture_output=True, text=True, timeout=60)
            self.assertIn("PDMUX_MO_MODE must be", r.stderr)

    def test_header_examples_use_only_names_launch_sh_accepts(self):
        head = "\n".join(ln for ln in self.P.read_text().splitlines() if ln.startswith("#"))
        knobs = re.findall(r"--env ([A-Z0-9_]+)=", head)
        self.assertTrue(knobs)
        self.assertTrue(all(re.match(r"^PDMUX_[A-Z0-9_]+$", k) for k in knobs), knobs)
        for script in ("mo_cap_job.sh", "mo_intf_job.sh"):
            h = "\n".join(ln for ln in (MO / script).read_text().splitlines() if ln.startswith("#"))
            self.assertTrue(all(k.startswith("PDMUX_") for k in re.findall(r"--env ([A-Z0-9_]+)=", h)), script)


# ---- 8. mutants -----------------------------------------------------------------------------------------------------------------------------------------
class TestMutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.out1, _, _ = run_block(cls.td.name, 1, SCENARIO)
        cls.out2, _, _ = run_block(cls.td.name, 2, SCENARIO)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def combined(self):
        return IA.combine_blocks([IA.analyze_block(self.out1), IA.analyze_block(self.out2)])

    def o_interference(self):
        lab = {(r["D"], r["rate"]): r["label"] for r in self.combined()["interference"]}
        return lab[(16, 3.0)] == "INTERFERENCE" and lab[(34, 3.0)] == "NO_INTERFERENCE"

    def o_row3(self):
        return all(a["row"] == 3 and a["resolving_D"] == [24, 34, 44] for a in self.combined()["actions"])

    def o_d16_below(self):
        return all(r["label"] == "BELOW_BEST" for r in self.combined()["d_shift"] if r["D"] == 16 and r["with_long"])

    def o_d_env(self):
        f = IS.fixed_values(True)
        return [IS.d_arm(d, f)["env"]["PDMUX_R2_FIXED_DSM"] for d in IS.D_GRID] == ["16", "24", "34", "44"] and \
            IS.plan_problems(IS.make_plan("JM", RATES, 1, rehearsal=True)) == []

    def o_pair(self):
        seg = lambda wl: MW.build_segments({"run_id": "x", "workload": wl, "rate": 3.0, "seed": L.SEEDS[0], "duration_s": 300.0}, None)[0]["requests"]  # noqa: E731
        w, wo = seg(IS.WITH), seg(IS.WITHOUT)
        return len(w) > len(wo) and [r for r in w if r["phase"] != MW.LONG_PHASE] == wo and IS.plan_problems(IS.make_plan("JM", RATES, 1, rehearsal=True)) == []

    def o_context(self):
        return IS.plan_problems(IS.make_plan("JM", RATES, 1, rehearsal=True)) == []

    def o_block_ci(self):
        e = IA.block_estimate([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]], "t")
        return e["t_hi"] - e["t_lo"] < 2.5

    def o_refuse_same_job(self):
        b = IA.analyze_block(self.out1)
        try:
            IA.combine_blocks([b, b])
        except MA.JobInvalid:
            return True
        return False

    def o_seeds_disjoint(self):
        return not (set(IS.BLOCK_SEEDS[1]) & set(IS.BLOCK_SEEDS[2])) and \
            "share a seed" not in " ".join(IA.combine_problems([IA.analyze_block(self.out1), IA.analyze_block(self.out2)]))

    def o_split_per_d(self):
        with tempfile.TemporaryDirectory() as td:
            out, _, _ = run_block(td, 1, SCENARIO, job="JSC", tamper={"JSC_splitcheck_D24": {"green_d": 44}})
            return "SPLIT_CHECK_D24_MISMATCH" in IA.analyze_block(out)["flags"]

    def o_three_percent(self):
        a = IA.combine_blocks(synth(good=lambda d, wl: 3.0 * {16: 0.98, 24: 0.99, 34: 0.985, 44: 1.0}[d] if d else 3.0))["actions"][0]
        return a["row"] == 1

    def o_interval_gate(self):
        noisy = synth(good=lambda d, wl: 3.0 * (0.90 if d == 34 and wl else 1.0) if d else 3.0, eps=0.2)
        return next(x for x in IA.combine_blocks(noisy)["d_shift"] if x["D"] == 34 and x["with_long"])["label"] == "WITHIN"

    def o_long_tag(self):
        b = IA.analyze_block(self.out1)
        return all(r["n"] == round(0.05 / 0.95 * r["rate"] * SHORT_DURATION) * 4 for r in b["loogle_only"])

    def test_each_mutant_is_caught(self):
        real_read, real_est, real_interf, real_d_arm, real_fixed = MA.read_label, IA.block_estimate, G.interference, IS.d_arm, IS.fixed_values

        def pooled_variance(by_block, label=""):                  # ignores the block: one sample, variance around the grand mean
            return real_est([[v for b in by_block for v in b]], label)

        def no_interval(by_block, label=""):                      # the point estimate alone decides "more than 3 % below"
            e = real_est(by_block, label)
            if "t_hi" in e:
                e["t_hi"] = -1.0
            return e

        def d_arm_keeps_d44(D, fixed, split_check=False):
            a = real_d_arm(D, fixed, split_check)
            a["env"]["PDMUX_R2_FIXED_DSM"] = "44"
            return a

        mutants = [
            ("with / without swapped in the reading", mock.patch.object(MA, "read_label", lambda a, b: real_read(b, a)), self.o_interference),
            ("interference read without the margin", mock.patch.object(MS, "ATT_MARGIN", 0.5), self.o_interference),
            ("starved threshold lowered", mock.patch.object(MS, "CAP_ATTAIN", 0.5), self.o_row3),
            ("goodput loss threshold raised", mock.patch.object(IS, "GOODPUT_LOSS", 0.6), self.o_d16_below),
            ("three percent read as zero", mock.patch.object(IS, "GOODPUT_LOSS", 0.0), self.o_three_percent),
            ("interval not required for the D shift", mock.patch.object(IA, "block_estimate", no_interval), self.o_interval_gate),
            ("D arm keeps the bundle's D44", mock.patch.object(IS, "d_arm", d_arm_keeps_d44), self.o_d_env),
            ("'without' member carries the long stream", mock.patch.object(G, "interference", lambda s, sp, lp, r, d, wl: real_interf(s, sp, lp, r, d, True)),
             self.o_pair),
            ("boots keep the 20,480 context", mock.patch.object(IS, "fixed_values", lambda rehearsal=False: dict(real_fixed(rehearsal), context=20480)),
             self.o_context),
            ("variance pooled across blocks", mock.patch.object(IA, "block_estimate", pooled_variance), self.o_block_ci),
            ("combination accepts anything", mock.patch.object(IA, "combine_problems", lambda blocks: []), self.o_refuse_same_job),
            ("second block reuses the first block's seeds", mock.patch.dict(IS.BLOCK_SEEDS, {2: IS.BLOCK_SEEDS[1]}), self.o_seeds_disjoint),
            ("split check read for one D only", mock.patch.object(IS, "D_GRID", (44,)), self.o_split_per_d),
            ("LooGLE cell counts every long-input request", mock.patch.object(MW, "LONG_PHASE", "sharegpt_first"), self.o_long_tag),
        ]
        for name, patch, oracle in mutants:
            self.assertTrue(oracle(), f"oracle of '{name}' does not hold on the unmutated code")
            with patch:
                try:
                    held = bool(oracle())
                except Exception:  # noqa: BLE001 -- a mutant that makes the pass crash is caught as well
                    held = False
            self.assertFalse(held, f"mutant '{name}' survives: its oracle is an identity")
            self.assertTrue(oracle(), f"'{name}': the patch leaked")


if __name__ == "__main__":
    unittest.main()
