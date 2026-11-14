"""CPU tests of the concurrency cap and decode memory traffic measurement (results/motivation/mo_cap_*.py, 2026-11-12).  No GPU, no engine:
the fixed-batch part runs against the cost-term harness's fake engine on its virtual clock, the serving part against the fake server / fake
client of test_motivation_harness.  Nothing here shows the harness works on a GPU.

TestMutants: every guard listed there is checked twice -- the oracle holds on the code as written and FAILS when the guard is replaced by
its broken variant (a check that passes either way is an identity, not a check)."""
import copy
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
os.environ["PDMUX_COSTTERMS_TEST_MODE"] = "1"
import mo_cap_spec as CS  # noqa: E402
import mo_cap_runner as CR  # noqa: E402
import mo_cap_analyze as CA  # noqa: E402
import mo_spec as MS  # noqa: E402
import mo_card as MC  # noqa: E402
import mo_runner as MR  # noqa: E402
import mo_schema as SC  # noqa: E402
import mo_rehearsal as MH  # noqa: E402
import mo_analyze as MA  # noqa: E402
import ct_runner as R  # noqa: E402  (fake-engine VirtualServer)
import test_motivation_harness as TH  # noqa: E402  (FakeServer / fake_replay / POOL of the grid harness tests)

SP = MS.SP
SRC = ENG / "results" / "cap_char" / "compo_probe" / "compo_probe_20261021T131621Z" / "variants" / "a4"
ALL = frozenset(CS.PROPOSED)
SMALL_BUDGET = 50_816_000_000          # the cost-term fake's own default: weights + pool at mem 0.6
TH_OK = {"min_rungs_agree": 2, "mem_tie_break": "lower", "cap_tie_break": "lower", "min_kv_tokens_per_slot": 1000}


# ---- fakes ---------------------------------------------------------------------------------------------------------------------------
class FakeFixedServer:
    """The cost-term fake engine (virtual clock) behind the interface run_fixed_boot uses, plus what engine_tree.py boot would leave:
    the tree stamp and a ServerArgs banner with the three fields the card reads that the cost-term fake does not print."""

    def __init__(self, bdir, a, tamper=None, truth=None, budget=None):
        self.bdir, self.a, self.t = Path(bdir), a, tamper or {}
        self.v = R.VirtualServer(self.bdir, list(a["cmd"]), dict(a["env"]), truth or {})
        self.log, self.cmd = self.v.log, None
        self.budget = int(0.82 * CS.AVAIL_BYTES) if budget is None else budget      # the fake's weights + pool bytes (its own default is a mem-0.6 value)

    def start(self):
        a = self.a
        if self.t.get("status"):
            self.log.write_text("")
            return self.t["status"]
        with mock.patch.object(self.v.FE, "NANO_BUDGET", self.budget):
            st = self.v.start()
        argv = [a["python"]] + list(a["cmd"]) + ["--port", "1"]
        self.cmd = [sys.executable, str(SP.ENGINE_TREE_PY), "boot", "--stamp", str(self.bdir / f"tree_{a['name']}.json"), "--"] + argv
        stamp = {"engine_tree": a["tree"], "argv": argv, "pdmux_env": {**TH.PLUMB, **a["env"]}}
        stamp.update(self.t.get("stamp", {}))
        (self.bdir / f"tree_{a['name']}.json").write_text(json.dumps(stamp))
        if st == "ok":
            self.v.lf.flush()
            c = a["cmd"]
            extra = (f", enable_mixed_chunk={'--enable-mixed-chunk' in c}, disable_overlap_schedule={'--disable-overlap-schedule' in c}, "
                     f"disable_piecewise_cuda_graph={'--disable-piecewise-cuda-graph' in c}, tp_size=")
            txt = self.log.read_text().replace(", tp_size=", extra, 1)
            for k, v in (self.t.get("banner") or {}).items():
                txt = re.sub(rf"\b{k}=[^,)]+", f"{k}={v}", txt, count=1)
            self.log.write_text(txt)
            self.tr = self.v.tr
        return st

    def alive(self):
        return self.v.alive()

    def stop(self):
        return self.v.stop() if not self.t.get("status") else {"returncode": 1}


def fixed_server(tamper_by_boot=None, truth=None, budget=None):
    tb = tamper_by_boot or {}
    return lambda bdir, a: FakeFixedServer(bdir, a, tb.get(Path(bdir).name), truth, budget)


def variant_fn(dst):
    return CR.build_variant(SRC, dst)


def run_fixed(plan, td, tamper=None, truth=None, gate=lambda p: None, budget=None):
    card = CR.fixed_card(plan)
    out = Path(td) / "raw"
    res = CR.run_fixed_job(plan, card, out, fixed_server(tamper, truth, budget), gate_fn=gate, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze,
                           variant_fn=variant_fn)
    return out, card, res


def small_fixed_plan(job="JF", keep=((("nano", 8), ((8, 512), (8, 8192))), (("a31", 4), ((4, 256), (4, 2048))), (("nano", 2), ((2, 131072),)))):
    """A smoke selection: a few chosen cells of the table (the full table is exercised by the plan tests, not sent through the fake)."""
    return CS.fixed_plan(job, rehearsal=True, only=dict(keep))


def truth_step_ms(model, bs, ctx):
    """Ground truth of the fake engine (vanilla, 108 SM): (a + b * bs_padded + c * bs * ctx) * vanilla_mult, scaled by composition."""
    import ct_fake_engine as FE
    t = FE.DEFAULT_TRUTH
    a, nano = CS.arch(model), CS.arch("nano")
    sa = CS.AT.params_total(a) / CS.AT.params_total(nano)
    sb = 0.2 + 0.8 * a.n_M / 27.0
    sc = CS.k_bytes(model) / 16384.0
    return (t["a108"] * sa + t["b108"] * sb * bs + t["c108"] * sc * bs * ctx) * t["vanilla_mult"]


class KvServer(TH.FakeServer):
    """Grid-harness fake server + the max_total_num_tokens banner line the cap analysis reads."""
    KV = {48: 2_000_000, 128: 1_600_000, 256: 540_000}

    def start(self):
        st = super().start()
        cap = int(TH._flag(self.a["cmd"], "--max-running-requests", "48"))
        with open(self.log, "a") as f:
            f.write(f"max_total_num_tokens={self.KV[cap]}, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests={cap}, context_len=20480\n")
        return st


def perf_by_combo(capacity):
    """fake_replay perf: SLO capacity (req/s) keyed by '(mem, cap)' read from the arm name in the fake base_url."""
    def perf(base_url, rate):
        m = re.search(r"_cap(\d+)_mem([0-9.]+)", base_url)
        return capacity[(float(m.group(2)), int(m.group(1)))]
    return perf


CAPACITY = {(0.6, 48): 4.0, (0.75, 48): 4.0, (0.75, 128): 6.0, (0.75, 256): 6.0}


def run_serving(td, job="JS", capacity=None, tamper=None, seeds=None, rehearsal=False, server=KvServer):
    with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
        plan = CS.serving_plan(job, "W1", 3.0, 1.3, run_s=4.0, seeds=seeds, rehearsal=rehearsal)
        card = MC.build_card(plan, TH.POOL)
        out = Path(td) / "raw"
        tb = tamper or {}
        res = MR.run_job(plan, card, out, lambda bdir, a: server(bdir, a, tb.get(a["name"])), coupled_replay(capacity or CAPACITY), TH.POOL,
                         gate_fn=lambda p: None, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze, warmup_s=4.0)
    return out, plan, card, res


def coupled_replay(capacity):
    """TH.fake_replay with the per-request draw taken from the request id WITHOUT the arm name: the same (rate, seed, index) request gets
    the same draw in every arm, so two combinations of equal capacity produce identical outcomes (no sampling noise between arms)."""
    real_u = TH._u

    def replay(*a):
        with mock.patch.object(TH, "_u", lambda s: real_u(re.sub(r"(SGL|VLLM)0\.\S*?_cap\d+_mem[0-9.]+", "ARM", s))):
            return TH.fake_replay(*a, perf=perf_by_combo(capacity))
    return replay


# ---- 1. cell table -----------------------------------------------------------------------------------------------------------------------
class TestCellTable(unittest.TestCase):
    def test_grid_is_the_user_definition(self):
        self.assertEqual(CS.CONTEXTS, tuple(2 ** i for i in range(7, 18)))          # 2^18 dropped: user decision 2026-11-13
        self.assertEqual(CS.BATCHES, (1, 2, 4, 8, 16, 32, 64, 128, 256))
        self.assertEqual(CS.FB_MEM, 0.82)

    def test_s_and_k_come_from_analytic_terms(self):
        self.assertEqual(CS.s_bytes("nano"), CS.AT.state_bytes_per_request(CS.AT.nano9b_v2()))
        self.assertEqual(CS.k_bytes("nano"), CS.AT.kv_bytes_per_token(CS.AT.nano9b_v2()))
        self.assertEqual(CS.s_bytes("a31"), 0)
        self.assertEqual(CS.k_bytes("a31"), CS.AT.kv_bytes_per_token(CS.AT.nano9b_v2(31, 0)))

    def test_documented_rule_gives_86_and_71_not_the_stated_75_and_60(self):
        """The first stated counts were not reproduced; the user decided the rule's table (2026-11-12: 90 / 72 with the 2^18 column;
        2026-11-13: 86 / 71 without it).  75 / 60 stays as a record only."""
        self.assertEqual({m: len(CS.predicted_cells(m)) for m in CS.MODELS}, {"nano": 86, "a31": 71})
        self.assertEqual(CS.cell_count_gap(), {"nano": 11, "a31": 11})
        with mock.patch.object(CS, "CONTEXTS", CS.CONTEXTS + (262144,)):         # the dropped column = exactly these five cells
            gone = {m: sorted(set(CS.predicted_cells(m)) - {x for x in CS.predicted_cells(m) if x[1] <= 131072}) for m in CS.MODELS}
        self.assertEqual(gone, {"nano": [(1, 262144), (2, 262144), (4, 262144), (8, 262144)], "a31": [(1, 262144)]})
        self.assertEqual(CS.USER_STATED_CELLS, {"nano": 75, "a31": 60})

    def test_user_decisions_of_2026_11_12_are_confirmed_and_new_values_are_not(self):
        self.assertEqual(CS.open_items(), [])
        self.assertEqual(set(CS.CONFIRMED), set(CS.PROPOSED))
        self.assertEqual((CS.P("fb_boot_rule"), CS.P("sv_arms"), CS.P("fb_rounds")), ("cap_equals_batch", ("SG_MIXED_8192", "PDMUX_BUNDLE"), (1, 2)))
        with mock.patch.dict(CS.PROPOSED, {"new_value": (1, "no decision yet")}):
            self.assertEqual(CS.open_items(), ["new_value"])                    # listed explicitly: nothing is confirmed by default
            self.assertTrue(any("UNCONFIRMED" in x for x in CS.fixed_plan_problems(CS.fixed_plan("J", rounds=(1,)))))
        self.assertTrue(all(MS.is_pending(v) for v in CS.FIX_RULE.values()))     # the rule thresholds stay with the pre-registration

    def test_same_table_as_the_prediction_script(self):
        d = json.loads((ENG.parents[1] / "reports" / "motivation_gap_2026-11-09" / "predict_motivation_charts.json").read_text())
        self.assertEqual(d["regime_fit"], {"Nano": 90, "a31": 72})
        got = {(r["m"], r["b"], r["c"]) for r in d["regime"] if r["fit"] and r["c"] <= CS.NANO_CONTEXT_LIMIT}      # the script's 2^18 column is not measured
        self.assertEqual(got, {("Nano" if m == "nano" else m, b, c) for m in CS.MODELS for b, c in CS.predicted_cells(m)})

    def test_table_reacts_to_mem_and_stays_within_the_context_limit(self):
        self.assertLess(len(CS.predicted_cells("nano", 0.6)), len(CS.predicted_cells("nano", 0.82)))
        self.assertEqual(max(CS.CONTEXTS), CS.NANO_CONTEXT_LIMIT)
        self.assertEqual(CS.cell_table_summary()["nano"]["max_context_by_batch"][1], 131072)

    def test_banner_rule(self):
        ctx = CS.P("fb_context_length")
        need = 8 * (CS.c_in(8192) + CS.o_tokens(8192))
        self.assertEqual(CS.fits_banner(8, 8192, math.ceil(need / 0.95), 8, ctx), [])
        self.assertTrue(any("0.95" in r for r in CS.fits_banner(8, 8192, math.ceil(need / 0.95) - 1, 8, ctx)))      # one token short
        self.assertTrue(any("cap" in r for r in CS.fits_banner(8, 8192, 10 ** 9, 4, ctx)))
        self.assertTrue(any("capture" in r for r in CS.fits_banner(8, 8192, 10 ** 9, 8, ctx, capturable=False)))
        self.assertTrue(any("banner" in r for r in CS.fits_banner(8, 8192, None, 8, ctx)))
        self.assertTrue(any("context_length" in r for r in CS.fits_banner(1, 8192, 10 ** 9, 8, 8192)))
        self.assertTrue(any("context_length" in r for r in CS.fits_banner(1, 131072, 10 ** 9, 8, 65536)))      # a clamped context refuses, never shrinks

    def test_every_predicted_cell_passes_the_banner_rule_under_its_predicted_boot(self):
        for rule in ("cap_equals_batch", "min_boots"):
            for m in CS.MODELS:
                self.assertEqual(CS.predicted_infeasible(m, rule), [], (rule, m))
                self.assertEqual(sorted(c for cells in CS.boot_cells(m, rule).values() for c in cells), sorted(CS.predicted_cells(m)))
        self.assertEqual({m: len(CS.boot_cells(m, "min_boots")) for m in CS.MODELS}, {"nano": 3, "a31": 1})

    def test_predicted_pool_is_close_to_the_probe_banner(self):
        """py3.12 boot probe 2026-11-09: cap 16, mem 0.82 -> max_total_num_tokens=3004104 (the only mem-0.82 banner there is)."""
        self.assertAlmostEqual(CS.predicted_kv_tokens("nano", 16) / 3004104, 1.0, delta=0.01)

    def test_window_geometry(self):
        for c in CS.CONTEXTS:
            w = CS.window_iters(c)
            self.assertEqual(w, min(300, c // 2))
            self.assertEqual(CS.abort_at(c), 1 + 16 + w + 2)
            self.assertGreater(CS.c_in(c), 0)
            self.assertLessEqual(CS.c_in(c) + CS.o_tokens(c), CS.P("fb_ctx_total_max"))
            self.assertLess(CS.c_in(c) + CS.o_tokens(c), CS.P("fb_context_length"))      # input + output fit the boot's context length
            if c < 131072:                                    # the window is centred on c
                self.assertEqual(CS.c_in(c) + 1 + 16 + w // 2, c)
        self.assertEqual(CS.P("fb_ctx_total_max"), 131072 - 112)
        self.assertLess(CS.c_in(131072), 131072 - CS.x_tokens(131072))       # the top context is shifted down, not centred
        self.assertEqual(CS.c_in(131072) + CS.o_tokens(131072), CS.P("fb_ctx_total_max"))
        with mock.patch.dict(CS.PROPOSED, {"fb_ctx_total_max": (10 ** 9, "")}):   # without the shift the top cells are refused, not sent
            self.assertTrue(any("context_length" in r for r in CS.fits_banner(1, 131072, 10 ** 9, 1, CS.P("fb_context_length"))))


# ---- 2. plans ------------------------------------------------------------------------------------------------------------------------------
class TestFixedPlan(unittest.TestCase):
    def setUp(self):
        self.p = mock.patch.object(CS, "CONFIRMED", ALL)
        self.p.start()
        self.addCleanup(self.p.stop)
        self.fx = mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED))
        self.fx.start()
        self.addCleanup(self.fx.stop)

    def test_round_plan_is_clean_and_covers_the_table_once(self):
        plan = CS.fixed_plan("JF", rounds=(1,))
        self.assertEqual(CS.fixed_plan_problems(plan), [])
        self.assertEqual(MS.plan_problems(plan), [])                                  # mo_spec dispatches by plan kind
        meas = [b for b in plan["boots"] if b["role"] == "meas"]
        self.assertEqual(len(meas), 18)
        self.assertEqual(sum(len(b["cells"]) for b in meas), 86 + 71)
        self.assertEqual((plan["boots"][0]["role"], plan["boots"][-1]["role"]), ("anchor", "anchor"))
        self.assertEqual(plan["boots"][0]["cells"], plan["boots"][-1]["cells"])

    def test_boot_is_unpartitioned_uninstrumented_chunked_at_mem_082(self):
        for b in CS.fixed_plan("JF", rounds=(1,))["boots"]:
            a = b["arm_def"]
            self.assertEqual((a["tree"], a["env"]), ("pristine", {}))
            self.assertNotIn("--enable-pdmux", a["cmd"])
            self.assertNotIn("--enable-mixed-chunk", a["cmd"])
            self.assertEqual(TH._flag(a["cmd"], "--chunked-prefill-size"), "8192")
            self.assertEqual(TH._flag(a["cmd"], "--mem-fraction-static"), "0.82")
            self.assertEqual(TH._flag(a["cmd"], "--max-running-requests"), str(b["cap"]))
            self.assertEqual(TH._flag(a["cmd"], "--context-length"), "131072")
            self.assertEqual("--load-format" in a["cmd"], b["model"] == "a31")
            for part in (f"cap{b['cap']}", "mem0.82", "ctx131072", "pristinetree", "chunked", CS.MODELS[b["model"]]["tag"]):
                self.assertIn(part, a["name"])

    def test_order_is_seeded_by_job_and_round(self):
        self.assertEqual(CS.fb_boot_order("J1", 1), CS.fb_boot_order("J1", 1))
        self.assertNotEqual(CS.fb_boot_order("J1", 1), CS.fb_boot_order("J2", 1))
        self.assertNotEqual(CS.fb_boot_order("J1", 1), CS.fb_boot_order("J1", 2))
        self.assertNotEqual(CS.fb_boot_order("J1", 1), CS.fb_boot_keys())               # not the registration order

    def test_no_boot_exceeds_the_context_length_the_model_derives(self):
        """2026-11-13: every boot of the rehearsal Job died because --context-length 262144 > 131072 (Nano-9B-v2's derived limit) and the
        boot path removes SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.  Every plan of both parts, argv and plan field."""
        limit = 131072
        plans = [CS.fixed_plan("JF", rounds=(1,)), CS.fixed_plan("JF", rounds=(2,)), CS.fixed_plan("JF", rehearsal=True), small_fixed_plan(),
                 CS.fixed_plan("JF", rounds=(1,), rule="min_boots"),
                 CS.serving_plan("JS", "W1", 3.0, 1.3), CS.serving_plan("JS", "W1", 3.0, 1.3, rehearsal=True)]
        n = 0
        for plan in plans:
            self.assertLessEqual(plan.get("context_length", 0), limit)
            for b in plan["boots"]:
                a = b["arm_def"]
                self.assertEqual(a["engine"], "sglang")
                self.assertLessEqual(int(TH._flag(a["cmd"], "--context-length")), limit, b["boot_id"])
                self.assertLessEqual(a["fixed"]["context"], limit)
                self.assertNotIn("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", a["env"])
                for c in b.get("cells", []):                           # fixed batch: the cell itself fits the boot's context
                    self.assertLess(CS.c_in(c[1]) + CS.o_tokens(c[1]), int(TH._flag(a["cmd"], "--context-length")))
                n += 1
        self.assertGreater(n, 60)
        self.assertLessEqual(max(CS.CONTEXTS), limit)
        p = CS.fixed_plan("JF", rounds=(1,))                           # the plan checker states it as well
        cmd = p["boots"][3]["arm_def"]["cmd"]
        cmd[cmd.index("--context-length") + 1] = "262144"
        self.assertTrue(any("derived limit" in x for x in CS.fixed_plan_problems(p)))

    def test_checker_rejects_each_mutation(self):
        base = CS.fixed_plan("JF", rounds=(1,))
        i = next(k for k, b in enumerate(base["boots"]) if b["role"] == "meas" and len(b["cells"]) > 2)

        def mut(fn):
            p = copy.deepcopy(base)
            fn(p)
            p["boot_order"] = [b["arm"] for b in p["boots"]]
            return " | ".join(CS.fixed_plan_problems(p))

        self.assertIn("registered permutation", mut(lambda p: p["boots"].insert(2, p["boots"].pop(1))))
        self.assertIn("anchor boot must be first and last", mut(lambda p: p["boots"].append(p["boots"].pop(0))))
        self.assertIn("cell table", mut(lambda p: p["boots"][i]["cells"].pop(0)))
        self.assertIn("does not carry the cap / mem", mut(lambda p: p["boots"][i]["arm_def"]["cmd"].__setitem__(
            p["boots"][i]["arm_def"]["cmd"].index("--mem-fraction-static") + 1, "0.6")))
        self.assertIn("un-instrumented", mut(lambda p: p["boots"][i]["arm_def"]["env"].update(PDMUX_DECODE_PROBE_PATH="x")))
        self.assertIn("arm name does not state", mut(lambda p: p["boots"][i]["arm_def"].update(name="x")))
        self.assertIn("not the registered anchor", mut(lambda p: p["boots"][0].update(cells=[[8, 4096]])))
        self.assertIn("weight loading", mut(lambda p: p["boots"][i]["arm_def"]["cmd"].extend(["--load-format", "dummy"])
                                            if p["boots"][i]["model"] == "nano" else p["boots"][i]["arm_def"]["cmd"].remove("--load-format")))

    def test_one_job_per_round(self):
        with self.assertRaises(ValueError):
            CS.fixed_plan("JF")                                                     # a measurement plan names its round
        both = CS.fixed_plan("JF", rounds=(1, 2))
        self.assertTrue(any("exactly one registered round" in x for x in CS.fixed_plan_problems(both)))
        self.assertTrue(any("exactly one registered round" in x for x in CS.fixed_plan_problems(CS.fixed_plan("JF", rounds=(3,)))))
        r1, r2 = CS.fixed_plan("JF", rounds=(1,)), CS.fixed_plan("JF", rounds=(2,))
        self.assertEqual(CS.fixed_plan_problems(r2), [])
        self.assertNotEqual(r1["boot_order"], r2["boot_order"])                    # each round has its own seeded order
        self.assertLess(CS.estimate_fixed(r1)["total_h"], 6.0)                     # the reason for the split
        self.assertGreater(CS.estimate_fixed(both)["total_h"], 6.0)

    def test_unconfirmed_values_and_placeholders_block_a_measurement_plan_only(self):
        self.p.stop()
        self.fx.stop()
        try:
            with mock.patch.object(CS, "CONFIRMED", frozenset()):
                self.assertIn("UNCONFIRMED design values", " | ".join(CS.fixed_plan_problems(CS.fixed_plan("JF", rounds=(1,)))))
            bad = " | ".join(CS.fixed_plan_problems(CS.fixed_plan("JF", rounds=(1,))))
            self.assertNotIn("UNCONFIRMED", bad)                                # decided 2026-11-12
            self.assertIn("PLACEHOLDER", bad)                                   # the py3.12 venv freeze digest is still unregistered
            self.assertNotIn("mem", re.findall(r"PLACEHOLDER[^|]*", bad)[0])    # ... but mem / cap are NOT placeholders here
            self.assertEqual(CS.fixed_plan_problems(CS.fixed_plan("JF", rehearsal=True)), [])
        finally:
            self.p.start()
            self.fx.start()

    def test_time_estimate_adds_up(self):
        plan = CS.fixed_plan("JF", rounds=(1,))
        e = CS.estimate_fixed(plan)
        self.assertEqual((e["boots"], e["cells"]), (20, 159))
        self.assertAlmostEqual(e["total_min"], e["cell_min"] + e["boot_min"] + e["start_min"], delta=0.11)
        self.assertEqual(e["boot_min"], 20 * 6.0)
        self.assertGreater(CS.est_cell_s("nano", 8, 131072), CS.est_cell_s("nano", 8, 8192))
        self.assertGreater(CS.est_cell_s("a31", 1, 65536), CS.est_cell_s("nano", 1, 65536))     # more attention layers


class TestServingPlan(unittest.TestCase):
    def setUp(self):
        self.p = mock.patch.object(CS, "CONFIRMED", ALL)
        self.p.start()
        self.addCleanup(self.p.stop)
        self.fx = mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED))
        self.fx.start()
        self.addCleanup(self.fx.stop)
        self.plan = CS.serving_plan("JS", "W1", 3.0, 1.3)

    def test_four_combinations_two_arms_four_rungs_four_seeds(self):
        self.assertEqual(CS.SERVING_COMBOS, ((0.6, 48), (0.75, 48), (0.75, 128), (0.75, 256)))
        self.assertEqual(CS.serving_plan_problems(self.plan), [])
        self.assertEqual(MS.plan_problems(self.plan), [])
        meas = [b for b in self.plan["boots"] if b["role"] == "meas"]
        self.assertEqual(len(meas), 4 * 2)
        self.assertTrue(all(len(b["runs"]) == 4 * 4 for b in meas))
        self.assertEqual(sum(len(b["runs"]) for b in meas), 128)
        self.assertEqual([b["role"] for b in self.plan["boots"]][0::len(self.plan["boots"]) - 1], ["anchor", "anchor"])
        self.assertEqual(sum(1 for b in self.plan["boots"] if b["role"] == "split_check"), 1)

    def test_each_boot_carries_its_own_mem_and_cap_in_name_argv_and_card(self):
        card = MC.build_card(self.plan, TH.POOL)
        names = set()
        for b in self.plan["boots"]:
            a = b["arm_def"]
            mem, cap = a["params"]["mem"], a["params"]["cap"]
            self.assertIn(f"_cap{cap}_mem{mem}", a["name"])
            self.assertEqual((TH._flag(a["cmd"], "--mem-fraction-static"), TH._flag(a["cmd"], "--max-running-requests")), (str(mem), str(cap)))
            e = card["arms"][a["name"]]["expect"]
            self.assertEqual((e["mem"], e["cap"]), (mem, cap))
            if b["role"] == "meas":
                names.add(a["name"])
        self.assertEqual(len(names), 8)                                        # eight different bundles
        self.assertEqual(MC.card_placeholders(card), [])                       # not blocked by mo_spec's PENDING mem / cap

    def test_run_length_and_same_trace_across_arms(self):
        for b in self.plan["boots"]:
            for r in b["runs"]:
                if b["role"] == "meas":
                    self.assertEqual(r["n_requests"], max(1, round(r["rate"] * 300.0)))
        e = CS.estimate_serving(self.plan)
        self.assertAlmostEqual(e["load_h"], 128 * 300 / 3600, delta=0.15)       # the roadmap's 10.7 load-hours

    def test_checker_rejects_each_mutation(self):
        i = next(k for k, b in enumerate(self.plan["boots"]) if b["role"] == "meas")

        def mut(fn):
            p = copy.deepcopy(self.plan)
            fn(p)
            p["boot_order"] = [b["arm"] for b in p["boots"]]
            return " | ".join(CS.serving_plan_problems(p))

        self.assertIn("registered permutation", mut(lambda p: p["boots"].insert(i + 1, p["boots"].pop(i))))
        self.assertIn("registered permutation", mut(lambda p: p["boots"].pop(i)))
        self.assertIn("full-block", mut(lambda p: p["boots"][i]["runs"].pop()))
        self.assertIn("anchor boot must be first and last", mut(lambda p: p["boots"].append(p["boots"].pop(0))))
        self.assertIn("gate 3", mut(lambda p: p.update(seeds=p["seeds"][:3])))
        self.assertIn("does not carry the cap / mem", mut(lambda p: p["boots"][i]["arm_def"]["cmd"].__setitem__(
            p["boots"][i]["arm_def"]["cmd"].index("--max-running-requests") + 1, "64")))
        self.assertIn("not a registered combination", mut(lambda p: p["boots"][i]["arm_def"]["params"].update(mem=0.7)))
        self.assertIn("instrumentation env", mut(lambda p: next(b for b in p["boots"] if b["arm_def"]["tree"] == "patched" and b["role"] == "meas")
                                                 ["arm_def"]["env"].update(PDMUX_TELEMETRY_PATH="x")))

    def test_rate_grid_has_no_default(self):
        self.assertEqual(CR.main(["plan", "--part", "serving", "--job", "J"]), 16)


# ---- 3. card ---------------------------------------------------------------------------------------------------------------------------------
class TestFixedCard(unittest.TestCase):
    def test_card_binds_plan_cells_and_values(self):
        plan = small_fixed_plan()
        card = CR.fixed_card(plan)
        self.assertTrue(MC.card_sha_ok(card))
        self.assertEqual(card["plan_sha"], MS.sha(plan))
        self.assertEqual(card["cells"][plan["boots"][0]["boot_id"]][0][:2], [8, 8192])
        self.assertEqual(card["analytic"], CS.AT.VERSION)
        p2 = copy.deepcopy(plan)
        p2["boots"][1]["cells"][0][1] *= 2
        self.assertNotEqual(CR.fixed_card(p2)["card_sha"], card["card_sha"])

    def test_banner_or_argv_tamper_aborts_the_job_before_any_cell(self):
        plan = small_fixed_plan()
        target = plan["boots"][1]["boot_id"]
        for tamper, field in (({"banner": {"mem_fraction_static": 0.9}}, "mem"), ({"banner": {"max_running_requests": 999}}, "cap"),
                              ({"banner": {"chunked_prefill_size": -1}}, "chunk"), ({"stamp": {"engine_tree": "patched"}}, "tree"),
                              ({"banner": {"enable_mixed_chunk": True}}, "mixed")):
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(CR.JobAbort) as cm:
                    run_fixed(plan, td, {target: tamper})
                self.assertEqual(cm.exception.status, "ABORT_CARD_MISMATCH")
                self.assertIn(f"[{field}]", cm.exception.reason)
                self.assertFalse((Path(td) / "raw" / "boots" / target / "cells.jsonl").exists())
                self.assertTrue((Path(td) / "raw" / "JOB_ABORTED.json").exists())

    def test_card_of_another_plan_is_refused(self):
        plan = small_fixed_plan()
        other = CR.fixed_card(small_fixed_plan("JOTHER"))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(CR.JobAbort) as cm:
                CR.run_fixed_job(plan, other, Path(td) / "raw", fixed_server(), variant_fn=variant_fn)
            self.assertEqual(cm.exception.status, "ABORT_CARD_PLAN")


# ---- 4. fixed batch: raw -> analysis table ----------------------------------------------------------------------------------------------------
class TestFixedEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.plan = small_fixed_plan()
        cls.out, cls.card, cls.res = run_fixed(cls.plan, cls.td.name)
        cls.job = CA.load_fixed(cls.out)
        cls.an = CA.analyze_fixed(cls.job)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_every_boot_done_and_feasibility_written_first(self):
        self.assertTrue(all(b["status"] == "done" for b in self.res))
        for b in self.job["boots"]:
            planned = [tuple(c) for c in next(p for p in self.plan["boots"] if p["boot_id"] == b["boot"]["boot_id"])["cells"]]
            self.assertEqual([(f["bs"], f["c"]) for f in b["feasibility"]], planned)
            self.assertEqual(b["feasibility"][0]["kv_tokens"], b["cells"][0]["banner"]["max_total_num_tokens"])

    def test_raw_cells_satisfy_the_schema_and_each_removed_field_is_caught(self):
        cells = [c for b in self.job["boots"] for c in b["cells"]]
        self.assertEqual(len(cells), 7)                                  # 5 measurement cells + 2 anchor cells
        for c in cells:
            self.assertEqual(SC.validate_fixed_cell(c), [])
            self.assertEqual(len(c["requests"]), c["bs"])
        c = cells[0]
        for k in SC.FIXED_CELL_KEYS:
            self.assertTrue(SC.validate_fixed_cell({x: v for x, v in c.items() if x != k}), k)
        bad = copy.deepcopy(c)
        bad["requests"][0]["chunks"][3][0] = bad["requests"][0]["chunks"][1][0] - 1.0
        self.assertTrue(any("monotone" in m for m in SC.validate_fixed_cell(bad)))
        bad = copy.deepcopy(c)
        bad["requests"].pop()
        self.assertTrue(any("requests != bs" in m for m in SC.validate_fixed_cell(bad)))

    def test_step_time_matches_the_fake_engine_truth(self):
        rows = [r for r in self.an["cell_rows"] if r["status"] == "VALID"]
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertAlmostEqual(r["step_ms"] / truth_step_ms(r["model"], r["bs"], r["ctx_mean"]), 1.0, delta=0.005, msg=r["cell_id"])
            self.assertAlmostEqual(r["tokens_per_s"], r["bs"] * 1000.0 / r["step_ms"])
            self.assertEqual(r["n_gaps"] >= r["bs"] * (CS.window_iters(r["c"]) - 1), True)

    def test_window_is_centred_on_the_cell_context_except_at_the_top(self):
        for r in self.an["cell_rows"]:
            if r["c"] < 131072:
                self.assertLess(abs(r["ctx_mean"] - r["c"]), 2.0, r["cell_id"])
                self.assertNotIn("CTX_OFF_NOMINAL", r["labels"])
            else:
                self.assertIn("CTX_OFF_NOMINAL", r["labels"])
        self.assertTrue(any(r["c"] == 131072 for r in self.an["cell_rows"]))

    def test_traffic_uses_analytic_terms_at_the_realized_context(self):
        for r in (x for x in self.an["cell_rows"] if x["status"] == "VALID"):
            ds = CS.AT.decode_step(CS.arch(r["model"]), r["bs"], int(round(r["ctx_mean"])))
            self.assertEqual(r["bytes_per_step"], ds["bytes_total"])
            self.assertEqual(r["bytes_kv"], r["bs"] * int(round(r["ctx_mean"])) * CS.k_bytes(r["model"]))
            self.assertAlmostEqual(r["bytes_per_s"], ds["bytes_total"] / (r["step_ms"] / 1000.0))
            self.assertAlmostEqual(r["step_over_roofline"], r["step_ms"] / CS.CT.step_ms_lb(r["bs"], r["ctx_mean"], CS.arch(r["model"])))
        a31 = next(r for r in self.an["cell_rows"] if r["model"] == "a31" and r["c"] == 2048)
        self.assertEqual((a31["bytes_state"], a31["largest_term"]), (0, "weights"))

    def test_anchor_is_a_drift_check_not_a_table_row(self):
        self.assertEqual(self.an["anchor"]["abs_ln_ratio"] < CS.P("fb_anchor_drift_ln"), True)
        self.assertNotIn("ANCHOR_DRIFT", self.an["flags"])
        self.assertFalse(any("anchor" in (r.get("boot_id") or "") for r in self.an["cell_rows"]))
        self.assertEqual(len(self.an["verdict_rows"]), 5)

    def test_outputs_written(self):
        p = CA.write_fixed(self.an, Path(self.td.name) / "an")
        txt = p.read_text().splitlines()
        self.assertEqual(txt[0].split(","), list(CA.FIXED_COLS))
        self.assertEqual(len(txt), 6)
        self.assertEqual(CA.main(["fixed", str(self.out), "--out", str(Path(self.td.name) / "an2")]), 0)

    def test_analyzer_refuses_a_broken_job(self):
        job = copy.deepcopy(self.job)
        job["boots"][1]["cells"].pop()
        with self.assertRaises(MA.JobInvalid) as cm:
            CA.analyze_fixed(job)
        self.assertIn("incomplete block", str(cm.exception))
        job = copy.deepcopy(self.job)
        job["boots"].pop()
        with self.assertRaises(MA.JobInvalid):
            CA.analyze_fixed(job)
        job = copy.deepcopy(self.job)
        job["boots"][1]["cells"][0]["card_sha"] = "0" * 64
        with self.assertRaises(MA.JobInvalid):
            CA.analyze_fixed(job)


class TestFixedFullTable(unittest.TestCase):
    def test_whole_table_two_round_jobs_raw_to_cell_table_and_round_agreement(self):
        """The two measurement plans themselves (one Job per round: 20 boots, 157 cells each) through the fake engine: every cell VALID,
        the seven top-context cells (c = the context length, input shifted down) carry CTX_OFF_NOMINAL, each Job gives its own 157-row table, and the rounds are compared without merging."""
        ans = []
        with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)), \
                tempfile.TemporaryDirectory() as td:
            for rnd in (1, 2):
                with mock.patch.dict(os.environ, {"PDMUX_JOB_NAME": f"job-round{rnd}"}):
                    out, _, res = run_fixed(CS.fixed_plan(f"JFULL{rnd}", rounds=(rnd,)), Path(td) / str(rnd))
                    self.assertTrue(all(b["status"] == "done" for b in res))
                    ans.append(CA.analyze_fixed(out))
        an = ans[0]
        self.assertEqual(an["counts"], {"nano": {"planned": 86, "valid": 86, "infeasible": 0, "server_death": 0},
                                            "a31": {"planned": 71, "valid": 71, "infeasible": 0, "server_death": 0}})
        self.assertEqual((an["flags"], len(an["verdict_rows"]), an["rounds"]), ([], 157, [1]))
        ra = CA.round_agreement(ans[0], ans[1])
        self.assertEqual((ra["n"], ra["merged"], ra["rounds"]), ({"AGREE": 157, "ROUND_DISAGREE": 0, "ONE_ROUND_ONLY": 0}, False, [1, 2]))
        slow = copy.deepcopy(ans[1])
        slow["verdict_rows"][0]["step_ms"] *= 1.05
        slow["verdict_rows"][1]["step_ms"] = None
        self.assertEqual(CA.round_agreement(ans[0], slow)["n"], {"AGREE": 155, "ROUND_DISAGREE": 1, "ONE_ROUND_ONLY": 1})
        with self.assertRaises(ValueError):
            CA.round_agreement(ans[0], ans[0])                                   # the same round / the same Job twice
        with self.assertRaises(ValueError):
            CA.round_agreement(ans[0], dict(ans[1], rehearsal=True))
        self.assertTrue(all(r["n_rounds_valid"] == 1 for r in an["verdict_rows"]))
        self.assertEqual(sorted((r["model"], r["bs"], r["c"]) for r in an["verdict_rows"] if "CTX_OFF_NOMINAL" in r["labels"]),
                         [("a31", 1, 131072), ("a31", 2, 131072)] + [("nano", b, 131072) for b in (1, 2, 4, 8, 16)])
        worst = max(abs(r["step_ms"] / truth_step_ms(r["model"], r["bs"], r["ctx_mean"]) - 1) for r in an["cell_rows"])
        self.assertLess(worst, 0.005)
        gain = {(g["model"], g["c"], g["bs"]): g["gain"] for g in an["doubling_gain"]}
        self.assertLess(gain[("nano", 2048, 256)], gain[("nano", 2048, 2)])           # the fake's per-request term: gain falls with the batch
        self.assertTrue(all(1.0 < v <= 2.0 for v in gain.values()))


class TestFixedRunTimeRules(unittest.TestCase):
    def test_cell_refused_by_the_banner_is_reported_not_sent(self):
        """A server whose pool is smaller than the mem-0.82 prediction: a planned cell that the banner refuses gets a row, no request."""
        plan = small_fixed_plan(keep=((("nano", 64), ((64, 128), (64, 32768))),))
        with tempfile.TemporaryDirectory() as td:
            out, _, res = run_fixed(plan, td, budget=SMALL_BUDGET)
            an = CA.analyze_fixed(out)
            row = {(r["bs"], r["c"]): r for r in an["cell_rows"]}
            self.assertEqual(row[(64, 32768)]["status"], "INFEASIBLE_BY_RULE")
            self.assertIn("0.95", row[(64, 32768)]["detail"])
            self.assertEqual(row[(64, 128)]["status"], "VALID")
            self.assertEqual(an["counts"]["nano"], {"planned": 2, "valid": 1, "infeasible": 1, "server_death": 0})
            cells = [c for b in CA.load_fixed(out)["boots"] for c in b["cells"]]
            self.assertFalse(any(c["c"] == 32768 for c in cells))

    def test_boot_failure_is_recorded_and_the_job_continues(self):
        plan = small_fixed_plan()
        target = plan["boots"][2]["boot_id"]
        with tempfile.TemporaryDirectory() as td:
            out, _, res = run_fixed(plan, td, {target: {"status": "died"}})
            self.assertEqual([b["status"] for b in res].count("died"), 1)
            self.assertEqual(res[-1]["status"], "done")
            with self.assertRaises(MA.JobInvalid) as cm:
                CA.analyze_fixed(out)
            self.assertIn("status died", str(cm.exception))

    def test_a_cell_that_kills_the_server_is_a_row_and_the_boot_goes_on(self):
        # user decision 2026-11-14: a server death inside a cell marks that cell, restarts the server and runs the remaining cells
        keep = ((("nano", 8), ((8, 512),)), (("nano", 2), ((2, 131072), (2, 512), (2, 2048))))
        plan = small_fixed_plan("JD", keep)
        target = next(b for b in plan["boots"] if b["role"] == "meas" and b["cap"] == 2)
        first_is_death = tuple(target["cells"][-1]) != (2, 131072)       # a death at the last cell needs no restart
        with tempfile.TemporaryDirectory() as td:
            out, _, res = run_fixed(plan, td, truth={"die_on_input_ge": 100000})
            self.assertEqual({b["status"] for b in res}, {"done"})
            r = next(b for b in res if b["boot_id"] == target["boot_id"])
            self.assertEqual(len(r["cells"]), 3)                           # every planned cell has a record; the dead one is not re-sent
            self.assertEqual(r["server_deaths"], [f"{target['boot_id']}_b2_c131072"])
            bdir = out / "boots" / target["boot_id"]
            self.assertEqual((bdir / "restart_1.json").exists(), first_is_death)
            self.assertEqual((bdir / "server.death1.log").exists(), first_is_death)
            if first_is_death:
                self.assertIn("illegal memory access", (bdir / "server.death1.log").read_text())
                self.assertTrue(json.loads((bdir / "restart_1.json").read_text())["warmup"]["greedy_ok"])
            an = CA.analyze_fixed(out)
            st = {(x["bs"], x["c"]): x["status"] for x in an["cell_rows"] if x["boot_id"] == target["boot_id"]}
            self.assertEqual(st, {(2, 131072): "SERVER_DEATH", (2, 512): "VALID", (2, 2048): "VALID"})
            self.assertEqual(an["counts"]["nano"]["server_death"], 1)
            row = next(x for x in an["verdict_rows"] if (x["model"], x["bs"], x["c"]) == ("nano", 2, 131072))
            self.assertEqual((row["n_rounds_valid"], row["statuses"]), (0, ["SERVER_DEATH"]))

    def test_restart_that_shows_another_pool_ends_the_boot(self):
        keep = ((("nano", 8), ((8, 512),)), (("nano", 2), ((2, 131072), (2, 512), (2, 2048))))
        plan = small_fixed_plan("JD", keep)
        target = next(b for b in plan["boots"] if b["role"] == "meas" and b["cap"] == 2)
        if tuple(target["cells"][-1]) == (2, 131072):
            self.skipTest("seeded order puts the death last: no restart in this plan")
        real = CR.restart_after_death

        def other_pool(srv, b, a, card, bdir, make_server, n, banner, *rest):
            return real(srv, b, a, card, bdir, make_server, n, dict(banner, max_total_num_tokens=banner["max_total_num_tokens"] + 1), *rest)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(CR, "restart_after_death", other_pool):
            out, _, res = run_fixed(plan, td, truth={"die_on_input_ge": 100000})
            self.assertEqual(next(b for b in res if b["boot_id"] == target["boot_id"])["status"], "RESTART_BANNER_DIFFERS")
            with self.assertRaises(MA.JobInvalid):
                CA.analyze_fixed(out)

    def test_anchor_drift_is_flagged(self):
        plan = small_fixed_plan()
        with tempfile.TemporaryDirectory() as td:
            out, _, _ = run_fixed(plan, td)
            job = CA.load_fixed(out)
            end = next(b for b in job["boots"] if b["boot"].get("anchor") == "end")
            for r in end["cells"][0]["requests"]:                       # the end anchor 10 % slower
                t0 = r["chunks"][0][0]
                r["chunks"] = [[t0 + (t - t0) * 1.10, n] for t, n in r["chunks"]]
            self.assertIn("ANCHOR_DRIFT", CA.analyze_fixed(job)["flags"])

    def test_real_plan_refuses_to_launch_without_a_rehearsal_record(self):
        with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
            plan = CS.fixed_plan("JREAL", rounds=(1,))
            card = CR.fixed_card(plan)
            with tempfile.TemporaryDirectory() as td:
                gate = lambda p: MH.launch_gate(p, Path(td) / "none", "c" * 64)       # noqa: E731
                with self.assertRaises(CR.JobAbort) as cm:
                    CR.run_fixed_job(plan, card, Path(td) / "raw", fixed_server(), gate_fn=gate, variant_fn=variant_fn)
                self.assertEqual(cm.exception.status, "ABORT_NO_REHEARSAL")
                self.assertFalse((Path(td) / "raw" / "boots").exists())

    def test_unconfirmed_values_refuse_a_measurement_run(self):
        with mock.patch.object(CS, "CONFIRMED", frozenset()):
            plan = CS.fixed_plan("JREAL", rounds=(1,))
        with tempfile.TemporaryDirectory() as td, mock.patch.object(CS, "CONFIRMED", frozenset()):
            with self.assertRaises(CR.JobAbort) as cm:
                CR.run_fixed_job(plan, CR.fixed_card(plan), Path(td) / "raw", fixed_server(), gate_fn=lambda p: None, variant_fn=variant_fn)
            self.assertEqual(cm.exception.status, "ABORT_PLAN")
            self.assertIn("UNCONFIRMED", cm.exception.reason)


class TestCellWindow(unittest.TestCase):
    """Hand-built cell: the first request decodes alone (40 ms / token) before the second joins; from then on both step at 10 ms."""

    def cell(self, trim=4, win=20):
        n = 1 + trim + win + 2
        late = [[10.0 + 0.010 * i, i + 1] for i in range(n)]
        early = [[9.0 + 0.040 * i, i + 1] for i in range(25)] + [[10.0 + 0.010 * i, 26 + i] for i in range(n)]
        mk = lambda rid, ch: {"rid": rid, "t_send": 8.0, "chunks": ch, "prompt_tokens": 100, "completion_tokens": ch[-1][1],   # noqa: E731
                              "finish": "abort", "error": None, "aborted": True}
        return {"cell_id": "x", "model": "nano", "bs": 2, "c": 128, "c_in": 100, "O": 200, "abort_at": n, "window_iters": win, "trim_iters": trim,
                "round": 1, "requests": [mk("a", early), mk("b", late)]}

    def test_window_starts_after_the_last_join(self):
        w = CA.cell_window(self.cell())
        self.assertEqual(w["status"], "VALID")
        self.assertAlmostEqual(w["step_ms"], 10.0, places=6)
        self.assertEqual(w["n_gaps"], 2 * 20)
        self.assertEqual(w["tokens_before_last_join"], 25)

    def test_invalid_and_short_windows(self):
        c = self.cell()
        c["requests"][1]["prompt_tokens"] = 99
        self.assertEqual(CA.cell_window(c)["status"], "BENCH_INVALID")
        c = self.cell()
        c["requests"][0]["error"] = "HTTP 500"
        self.assertEqual(CA.cell_window(c)["status"], "BENCH_INVALID")
        c = self.cell()
        c["requests"][1]["chunks"] = c["requests"][1]["chunks"][:10]
        c["requests"][1]["completion_tokens"] = 10
        self.assertEqual(CA.cell_window(c)["status"], "BENCH_INVALID")          # fewer than abort_at tokens
        c = self.cell()
        c["requests"][0].update(finish="length")
        c["requests"][0]["chunks"] = c["requests"][0]["chunks"][:35]              # ended by length inside the window
        c["requests"][0]["completion_tokens"] = 35
        c["abort_at"] = 20
        self.assertEqual(CA.cell_window(c)["status"], "WINDOW_SHORT")


# ---- 5. serving cap/mem comparison --------------------------------------------------------------------------------------------------------------
class TestServingEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.out, cls.plan, cls.card, cls.res = run_serving(cls.td.name)
        cls.an = CA.analyze_serving(cls.out, thresholds=TH_OK)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_job_ran_every_boot_and_every_run_has_the_grid_schema(self):
        self.assertEqual(MR.exit_code(self.res), 0)
        job = MA.load_job(self.out)
        self.assertEqual(len(job["runs"]), 128 + 3)
        self.assertEqual([m for r in job["runs"] for m in SC.validate_run(r)], [])
        self.assertEqual(self.an["flags"], [])

    def test_pairs_differ_in_one_factor_only(self):
        pairs = CA.serving_pairs()
        self.assertEqual(sorted(p[2] for p in pairs), ["cap", "cap", "cap", "mem"])
        for hi, lo, f in pairs:
            self.assertEqual(sum(1 for x, y in zip(hi, lo) if x != y), 1)
            self.assertGreater(hi[0 if f == "mem" else 1], lo[0 if f == "mem" else 1])
        self.assertEqual(len(self.an["verdict_rows"]), 2 * 4 * 4)                # arms x pairs x rungs

    def test_labels_follow_the_fake_profile(self):
        def labels(hi, lo):
            return {(r["arm_key"], r["rate"]): r["label"] for r in self.an["verdict_rows"] if (r["hi"], r["lo"]) == (hi, lo)}
        self.assertEqual(set(labels([0.75, 48], [0.6, 48]).values()) - {"UNINFORMATIVE", "SLO_DEPENDENT"}, set())     # equal capacity: no sign
        up = labels([0.75, 128], [0.75, 48])
        self.assertEqual(sum(1 for v in up.values() if v == "A>B"), 4)          # the two top rungs of both arms
        self.assertFalse(any(v == "A<B" for v in up.values()))
        self.assertFalse(any(v in ("A>B", "A<B") for v in labels([0.75, 256], [0.75, 128]).values()))

    def test_fix_rule_on_the_fake_profile(self):
        f = self.an["fix"]                 # cap 48 -> 128 is UP, 128 -> 256 shows nothing; cap 128 was measured at mem 0.75 only
        self.assertEqual((f["status"], f["mem"], f["cap"]), ("FIXED", 0.75, 128), f["trace"])
        self.assertEqual(CA.apply_fix(f)["cap"], 128)
        self.assertEqual(CA.apply_fix(f)["mem"], 0.75)
        an = CA.analyze_serving(self.out, thresholds=dict(TH_OK, cap_tie_break="higher"))
        self.assertEqual((an["fix"]["status"], an["fix"]["cap"]), ("FIXED", 256), an["fix"]["trace"])
        an = CA.analyze_serving(self.out, thresholds=dict(TH_OK, cap_tie_break="higher", min_kv_tokens_per_slot=2200))
        self.assertEqual(an["fix"]["status"], "USER_DECISION")                 # 540,000 / 256 = 2,109 tokens per slot

    def test_thresholds_are_pending_until_registered(self):
        self.assertTrue(all(MS.is_pending(v) for v in CS.FIX_RULE.values()))
        f = CA.analyze_serving(self.out)["fix"]
        self.assertEqual(f["status"], "PENDING_THRESHOLD")
        self.assertEqual((f["mem"], f["cap"]), (None, None))
        fixed = CA.apply_fix(f)
        self.assertTrue(MS.is_pending(fixed["mem"]) and MS.is_pending(fixed["cap"]))

    def test_banner_pool_is_read_per_combination(self):
        got = {(c["mem"], c["cap"]): c["max_total_num_tokens"] for c in self.an["combos"]}
        self.assertEqual(got, {(0.6, 48): 2_000_000, (0.75, 48): 2_000_000, (0.75, 128): 1_600_000, (0.75, 256): 540_000})

    def test_outputs_written(self):
        p = CA.write_serving(self.an, Path(self.td.name) / "an")
        self.assertEqual(len(p.read_text().splitlines()), 33)
        self.assertEqual(json.loads((Path(self.td.name) / "an" / "cap_mem_fix.json").read_text())["status"], "FIXED")

    def test_card_tamper_on_one_combination_aborts(self):
        target = next(b["arm"] for b in self.plan["boots"] if b["role"] == "meas" and b["arm_def"]["params"]["cap"] == 256)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:
                run_serving(td, tamper={target: {"banner": {"max_running_requests": 128, "max_mamba_cache_size": 128}}})
            self.assertIn("[cap]", cm.exception.reason)


class TestFixRule(unittest.TestCase):
    """The rule on hand-built pair rows (arms K1, K2; four rungs)."""

    def res(self, mem=("", ""), c128=("", ""), c256=("", ""), kv=None, flags=(), rehearsal=False):
        def rows(key, hi, lo, pat):
            pat = (pat + "....")[:4]
            return [{"arm_key": key, "hi": hi, "lo": lo, "rate": i, "label": {"u": "A>B", "d": "A<B"}.get(ch, "SLO_DEPENDENT")}
                    for i, ch in enumerate(pat)]
        out = []
        for k, i in (("K1", 0), ("K2", 1)):
            out += rows(k, [0.75, 48], [0.6, 48], mem[i]) + rows(k, [0.75, 128], [0.75, 48], c128[i]) + rows(k, [0.75, 256], [0.75, 128], c256[i])
        kv = kv or {(0.6, 48): 2e6, (0.75, 48): 2e6, (0.75, 128): 1.6e6, (0.75, 256): 5.4e5}
        return {"rehearsal": rehearsal, "flags": list(flags), "verdict_rows": out,
                "combos": [{"arm_key": k, "mem": m, "cap": c, "max_total_num_tokens": v} for k in ("K1", "K2") for (m, c), v in kv.items()]}

    def fix(self, th=None, **kw):
        f = CA.fix_cap_mem(self.res(**kw), dict(TH_OK, **(th or {})))
        return f["status"], f["mem"], f["cap"]

    def test_directions(self):
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("uu", "uu"), c256=("uu", "")), ("FIXED", 0.75, 256))
        self.assertEqual(self.fix(mem=("uu", ""), c128=("uu", "uu"), c256=("dd", "")), ("FIXED", 0.75, 128))
        self.assertEqual(self.fix(mem=("dd", "dd")), ("FIXED", 0.6, 48))
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("dd", "dd"), c256=("uu", "uu")), ("FIXED", 0.75, 48))     # the ladder stops

    def test_one_rung_is_not_a_direction(self):
        self.assertEqual(self.fix(mem=("u", "u")), ("FIXED", 0.6, 48))                      # 1 rung < min_rungs_agree -> tie break "lower"
        self.assertEqual(self.fix(mem=("u", "u"), th={"min_rungs_agree": 1}), ("FIXED", 0.75, 48))

    def test_tie_breaks(self):
        self.assertEqual(self.fix(th={"mem_tie_break": "higher"}), ("FIXED", 0.75, 48))
        self.assertEqual(self.fix(th={"mem_tie_break": "higher", "cap_tie_break": "higher", "min_kv_tokens_per_slot": 100}), ("FIXED", 0.75, 256))

    def test_disagreement_goes_to_the_user(self):
        self.assertEqual(self.fix(mem=("uu", "dd"))[0], "USER_DECISION")                    # arms disagree
        self.assertEqual(self.fix(mem=("ud", ""))[0], "USER_DECISION")                      # rungs disagree within an arm
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("uu", "dd"))[0], "USER_DECISION")

    def test_unmeasured_pair_goes_to_the_user(self):
        self.assertEqual(self.fix(mem=("dd", "dd"), c128=("uu", "uu")), ("USER_DECISION", 0.6, 128))

    def test_banner_floor(self):
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("uu", "uu"), c256=("uu", "uu"))[0], "FIXED")           # 540k / 256 = 2109 >= 1000
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("uu", "uu"), c256=("uu", "uu"), th={"min_kv_tokens_per_slot": 2200})[0], "USER_DECISION")
        kv = {(0.6, 48): 2e6, (0.75, 48): 2e6, (0.75, 128): None, (0.75, 256): 5.4e5}
        self.assertEqual(self.fix(mem=("uu", "uu"), c128=("uu", "uu"), kv=kv)[0], "USER_DECISION")

    def test_not_judgeable_and_pending(self):
        self.assertEqual(self.fix(rehearsal=True)[0], "NOT_JUDGEABLE")
        self.assertEqual(self.fix(flags=("ANCHOR_DRIFT",))[0], "NOT_JUDGEABLE")
        self.assertEqual(CA.fix_cap_mem(self.res(mem=("uu", "uu")))["status"], "PENDING_THRESHOLD")
        self.assertEqual(CA.fix_cap_mem(self.res(), dict(TH_OK, cap_tie_break=MS.pend("x")))["status"], "PENDING_THRESHOLD")

    def test_missing_pair_rows_are_never_no_difference(self):
        r = self.res(mem=("uu", "uu"))
        r["verdict_rows"] = [x for x in r["verdict_rows"] if not (x["arm_key"] == "K2" and x["hi"] == [0.75, 128])]
        self.assertEqual(CA.fix_cap_mem(r, TH_OK)["status"], "USER_DECISION")

    def test_fewer_than_four_seeds_is_not_judgeable(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(CS, "SERVING_SEEDS_MIN", 2), mock.patch.object(MS, "MIN_SEEDS", 2):
                out, _, _, _ = run_serving(td, seeds=SP.EVAL_SEEDS[:2])
            an = CA.analyze_serving(out, thresholds=TH_OK)
            self.assertIn("N_LT_4_SEEDS", an["flags"])
            self.assertTrue(all(r["label"] == "NOT_JUDGEABLE_N" for r in an["verdict_rows"]))
            self.assertEqual(an["fix"]["status"], "NOT_JUDGEABLE")


# ---- 6. rehearsal -----------------------------------------------------------------------------------------------------------------------------------
class TestRehearsal(unittest.TestCase):
    def test_fixed_full_pass_over_the_real_boot_list(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "rec"
            rec = CR.rehearse_fixed("JRF", Path(td) / "raw", fixed_server(), rehearsal_dir=rd, digest="d" * 64, py_version_fn=TH.fake_py,
                                    freeze_fn=TH.fake_freeze, variant_fn=variant_fn)
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertEqual(rec["kind"], CS.KIND_FIXED)
            self.assertEqual(len(rec["arm_names"]), 18)                         # every (model, cap); the anchor arm is one of them
            top = sum(len(CS.rehearsal_top_cells(m)) for m in CS.MODELS)
            self.assertEqual(top, 4)                                            # per model: smallest and largest batch at the top context
            self.assertEqual(rec["verdict_rows"], 18 + top)                     # one cell per boot + the top-context cells (2026-11-13)
            plan = CS.fixed_plan("JRF", rehearsal=True)
            ran = {(b["model"], tuple(c)) for b in plan["boots"] for c in b["cells"]}
            for m in CS.MODELS:
                for cell in CS.rehearsal_top_cells(m):
                    self.assertEqual(cell[1], CS.CONTEXTS[-1])
                    self.assertIn((m, cell), ran)
            self.assertTrue((rd / ("d" * 16 + f".{CS.KIND_FIXED}.json")).exists())
            self.assertFalse((rd / ("d" * 16 + ".json")).exists())              # not the grid measurement's record
            self.assertTrue((Path(td) / "raw" / "analysis" / "cell_table.csv").exists())
            with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
                self.assertIsNone(MH.launch_gate(CS.fixed_plan("JREAL", rounds=(1,)), rd, "d" * 64))
                self.assertIn("NO_REHEARSAL_RECORD", MH.launch_gate(CS.serving_plan("JS", "W1", 3.0, 1.3), rd, "d" * 64))
                self.assertIn("NO_REHEARSAL_RECORD", MH.launch_gate(MS.make_plan("JG", TH.GRID3, fixed=dict(MS.REHEARSAL_FIXED)), rd, "d" * 64))

    def test_fixed_rehearsal_passes_with_deaths_at_the_top_context_only(self):
        with tempfile.TemporaryDirectory() as td:
            rec = CR.rehearse_fixed("JRD", Path(td) / "raw", fixed_server(truth={"die_on_input_ge": 100000}), rehearsal_dir=Path(td) / "rec",
                                    digest="d" * 64, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze, variant_fn=variant_fn)
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertEqual(len(rec["server_deaths"]), 4)                  # the four top-context cells, named in the record
            self.assertTrue(all(c.endswith(f"_c{CS.CONTEXTS[-1]}") for c in rec["server_deaths"]))
        with tempfile.TemporaryDirectory() as td:                           # a death below the top context is a defect, not a finding
            rec = CR.rehearse_fixed("JRD", Path(td) / "raw", fixed_server(truth={"die_on_input_ge": 5000}), rehearsal_dir=Path(td) / "rec",
                                    digest="d" * 64, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze, variant_fn=variant_fn)
            self.assertFalse(rec["passed"])

    def test_fixed_rehearsal_names_the_failing_stage(self):
        plan = CS.fixed_plan("JRF", rehearsal=True)
        target = plan["boots"][3]["boot_id"]
        with tempfile.TemporaryDirectory() as td:
            rec = CR.rehearse_fixed("JRF", Path(td) / "raw", fixed_server({target: {"banner": {"mem_fraction_static": 0.6}}}),
                                    rehearsal_dir=Path(td) / "rec", digest="e" * 64, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze,
                                    variant_fn=variant_fn)
            self.assertFalse(rec["passed"])
            self.assertIn("=CARD_MISMATCH", rec["reason"])
            self.assertIn("[mem]", rec["reason"])
            self.assertFalse(rec["stages"]["verdict_table"])
            self.assertIn("NOT_PASSED", MH.launch_gate(plan, Path(td) / "rec", "e" * 64))

    def test_serving_full_pass_over_every_combination(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "rec"
            rec = CR.rehearse_serving("JRS", TH.POOL, Path(td) / "raw", lambda bdir, a: KvServer(bdir, a), TH.fake_replay, rehearsal_dir=rd,
                                      digest="f" * 64, py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze)
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertEqual(rec["kind"], CS.KIND_SERVING)
            self.assertEqual(len(rec["arm_names"]), 8 + 1)                      # 2 arms x 4 combinations + the split-check arm
            self.assertEqual(rec["verdict_rows"], 2 * 4)                        # arms x pairs, one rung
            self.assertTrue((rd / ("f" * 16 + f".{CS.KIND_SERVING}.json")).exists())
            fix = json.loads((Path(td) / "raw" / "analysis" / "cap_mem_fix.json").read_text())
            self.assertEqual(fix["status"], "NOT_JUDGEABLE")                    # a rehearsal never fixes mem / cap

    def test_grid_rehearsal_record_name_is_unchanged(self):
        self.assertEqual(MH.record_path("/x", "a" * 64, None).name, "a" * 16 + ".json")
        self.assertEqual(MH.record_path("/x", "a" * 64, "MOTIVATION_GRID").name, "a" * 16 + ".json")

    def test_harness_digest_covers_the_cap_modules_and_the_imported_sealed_code(self):
        names = {f.name for f in MH._harness_files()}
        for n in ("mo_cap_spec.py", "mo_cap_runner.py", "mo_cap_analyze.py", "ct_spec.py", "ct_client.py", "ct_extract.py", "analytic_terms.py",
                  "dstar_model.py", "make_variant.py"):
            self.assertIn(n, names)


# ---- 7. Job script ------------------------------------------------------------------------------------------------------------------------------------
class TestJobScript(unittest.TestCase):
    P = MO / "mo_cap_job.sh"

    def run_sh(self, env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_MOCAP_")}
        e.update(env)
        return subprocess.run(["bash", str(self.P)], env=e, capture_output=True, text=True, timeout=120)

    def test_syntax_executable_digest_heartbeat(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(self.P)]).returncode, 0)
        self.assertTrue(os.access(self.P, os.X_OK))
        txt = self.P.read_text()
        self.assertIn("8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858", txt)
        self.assertIn("export PDMUX_HEARTBEAT_S=0", txt)
        self.assertEqual(txt.count("8d8f6720"), (MO / "mo_job.sh").read_text().count("8d8f6720"))     # same engine digest as mo_job.sh

    def test_only_pdmux_env_names(self):
        """scripts/vessl/launch.sh passes --env only for PDMUX_* names."""
        names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:}]", self.P.read_text()))
        self.assertEqual({n for n in names if not n.startswith(("PDMUX_", "HF_HOME", "SGLANG_ENGINE_DEV", "BASH_SOURCE"))}, set())

    def test_config_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            base = {"PDMUX_MOCAP_OUT": td, "PDMUX_MOCAP_TEST_MODE": "1"}
            self.assertEqual(self.run_sh(dict(base)).returncode, 16)
            self.assertEqual(self.run_sh(dict(base, PDMUX_MOCAP_PART="fixed")).returncode, 16)
            self.assertEqual(self.run_sh(dict(base, PDMUX_MOCAP_PART="fixed", PDMUX_MOCAP_MODE="run", PDMUX_MOCAP_JOB="J")).returncode, 16)
            r = self.run_sh(dict(base, PDMUX_MOCAP_PART="serving", PDMUX_MOCAP_MODE="run", PDMUX_MOCAP_JOB="J", PDMUX_MOCAP_BUDGET_MIN="5"))
            self.assertEqual(r.returncode, 16)
            self.assertIn("PDMUX_MOCAP_WORKLOAD", r.stderr)

    def test_run_refuses_before_any_boot_while_the_card_has_a_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_sh({"PDMUX_MOCAP_OUT": str(Path(td) / "o"), "PDMUX_MOCAP_TEST_MODE": "1", "PDMUX_MOCAP_VARIANT_SRC": str(SRC),
                             "PDMUX_MOCAP_PART": "fixed", "PDMUX_MOCAP_MODE": "run", "PDMUX_MOCAP_JOB": "JSH", "PDMUX_MOCAP_BUDGET_MIN": "5",
                             "PDMUX_MOCAP_ROUND": "1"})
            self.assertEqual(r.returncode, 19, r.stderr[-400:])
            self.assertIn("ABORT_PLAN", r.stderr + r.stdout)
            self.assertFalse((Path(td) / "o" / "raw" / "boots").exists())


# ---- 8. mutants: each guard fails when it is broken ---------------------------------------------------------------------------------------------------
class TestMutants(unittest.TestCase):
    """(name, patch, oracle).  The oracle is True on the code as written and must be False (or raise) under the patch."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.fout, _, _ = run_fixed(small_fixed_plan(), cls.td.name + "/f")
        cls.sout, _, _, _ = run_serving(cls.td.name + "/s")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    # -- oracles
    def o_step_truth(self):
        rows = [r for r in CA.analyze_fixed(self.fout)["cell_rows"] if r["status"] == "VALID"]
        return len(rows) == 5 and all(abs(r["step_ms"] / truth_step_ms(r["model"], r["bs"], r["ctx_mean"]) - 1) < 0.005 for r in rows)

    def o_tokens_per_s(self):
        return all(abs(r["tokens_per_s"] - r["bs"] * 1000.0 / truth_step_ms(r["model"], r["bs"], r["ctx_mean"])) / r["tokens_per_s"] < 0.005
                   for r in CA.analyze_fixed(self.fout)["cell_rows"] if r["status"] == "VALID")

    def o_centred(self):
        with tempfile.TemporaryDirectory() as td:
            out, _, _ = run_fixed(small_fixed_plan(keep=((("nano", 8), ((8, 512),)),)), td)
            return all(abs(r["ctx_mean"] - r["c"]) < 2 for r in CA.analyze_fixed(out)["cell_rows"])

    def o_kv_traffic(self):
        return all(r["bytes_kv"] == r["bs"] * int(round(r["ctx_mean"])) * CS.k_bytes(r["model"]) and (r["model"] != "a31" or r["bytes_state"] == 0)
                   for r in CA.analyze_fixed(self.fout)["cell_rows"] if r["status"] == "VALID")

    def o_infeasible_not_sent(self):
        with tempfile.TemporaryDirectory() as td:
            out, _, res = run_fixed(small_fixed_plan(keep=((("nano", 64), ((64, 128), (64, 32768))),)), td, budget=SMALL_BUDGET)
            an = CA.analyze_fixed(out)
            return all(b["status"] == "done" for b in res) and an["counts"]["nano"] == {"planned": 2, "valid": 1, "infeasible": 1, "server_death": 0}

    def o_last_join(self):
        return abs(CA.cell_window(TestCellWindow().cell())["step_ms"] - 10.0) < 1e-6

    def o_card_catches_mem(self):
        plan = small_fixed_plan()
        with tempfile.TemporaryDirectory() as td:
            try:
                run_fixed(plan, td, {plan["boots"][1]["boot_id"]: {"banner": {"mem_fraction_static": 0.9}}})
            except CR.JobAbort as e:
                return e.status == "ABORT_CARD_MISMATCH"
        return False

    def o_argv_states_values(self):
        with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
            p = CS.serving_plan("JS", "W1", 3.0, 1.3)
            a = p["boots"][1]["arm_def"]
            a["cmd"][a["cmd"].index("--mem-fraction-static") + 1] = "0.9"
            return any("does not carry" in x for x in CS.serving_plan_problems(p))

    def o_names_distinct(self):
        with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
            p = CS.serving_plan("JS", "W1", 3.0, 1.3)
            return len({b["arm"] for b in p["boots"] if b["role"] == "meas"}) == 8 and CS.serving_plan_problems(p) == []

    def o_order_checked(self):
        with mock.patch.object(CS, "CONFIRMED", ALL), mock.patch.object(MS, "default_fixed", lambda: dict(MS.REHEARSAL_FIXED)):
            p = CS.fixed_plan("JF", rounds=(1,))
            p["boots"].insert(2, p["boots"].pop(1))
            p["boot_order"] = [b["arm"] for b in p["boots"]]
            return any("registered permutation" in x for x in CS.fixed_plan_problems(p))

    def o_order_seeded(self):
        a, b = [CS.fixed_plan(j, rehearsal=True)["boot_order"] for j in ("J1", "J2")]
        return a != b and sorted(a) == sorted(b)

    def o_cap_step_up(self):
        f = CA.analyze_serving(self.sout, thresholds=TH_OK)["fix"]
        return (f["status"], f["mem"], f["cap"]) == ("FIXED", 0.75, 128)

    def o_disagreement(self):
        return TestFixRule().fix(mem=("uu", "dd"))[0] == "USER_DECISION"

    def o_unmeasured(self):
        return TestFixRule().fix(mem=("dd", "dd"), c128=("uu", "uu"))[0] == "USER_DECISION"

    def o_pending(self):
        return CA.fix_cap_mem(TestFixRule().res(mem=("uu", "uu")))["status"] == "PENDING_THRESHOLD"

    def o_one_factor_pairs(self):
        return all(sum(1 for x, y in zip(hi, lo) if x != y) == 1 for hi, lo, _ in CA.serving_pairs()) and len(CA.serving_pairs()) == 4

    def o_table(self):
        return {m: len(CS.predicted_cells(m)) for m in CS.MODELS} == {"nano": 86, "a31": 71}

    def o_anchor_drift(self):
        job = CA.load_fixed(self.fout)
        end = next(b for b in job["boots"] if b["boot"].get("anchor") == "end")
        for r in end["cells"][0]["requests"]:
            t0 = r["chunks"][0][0]
            r["chunks"] = [[t0 + (t - t0) * 1.10, n] for t, n in r["chunks"]]
        return "ANCHOR_DRIFT" in CA.analyze_fixed(job)["flags"]

    def test_each_mutant_is_caught(self):
        real_window, real_fits, real_combine, real_traffic = CA.cell_window, CS.fits_banner, CA._combine, CA.traffic

        def window_no_join(rec):                          # the window opens at the FIRST request's first token (ramp included)
            r0 = min(rec["requests"], key=lambda r: r["chunks"][0][0])
            return real_window(dict(rec, requests=[r0] * rec["bs"]))

        def window_per_request(rec):                      # forgets the batch: tokens/s of ONE request
            w = real_window(rec)
            if w["status"] == "VALID":
                w["tokens_per_s"] = 1000.0 / w["step_ms"]
            return w

        def window_wall_mean(rec):                        # (t_abort - t_first_send) / tokens instead of the windowed gap
            w = real_window(rec)
            if w["status"] == "VALID":
                w["step_ms"] = (rec["t_abort"] - rec["t_first_send"]) * 1000.0 / max(r["completion_tokens"] for r in rec["requests"])
            return w

        mutants = [
            ("window opens before the last join", mock.patch.object(CA, "cell_window", window_no_join), self.o_last_join),
            ("step = wall time / tokens (fill included)", mock.patch.object(CA, "cell_window", window_wall_mean), self.o_step_truth),
            ("tokens/s forgets the batch", mock.patch.object(CA, "cell_window", window_per_request), self.o_tokens_per_s),
            ("input is not shortened by the half window", mock.patch.object(CS, "x_tokens", lambda c: 0), self.o_centred),
            ("traffic always uses the Nano composition", mock.patch.object(CA, "traffic", lambda m, bs, ctx, st: real_traffic("nano", bs, ctx, st)),
             self.o_kv_traffic),
            ("banner rule ignores the token pool", mock.patch.object(CS, "fits_banner", lambda b, c, kv, cap, ctx, capturable=True:
                                                                     real_fits(b, c, 10 ** 12, cap, ctx, capturable)), self.o_infeasible_not_sent),
            ("card check never fails", mock.patch.object(MC, "check_arm", lambda card, arm, obs: []), self.o_card_catches_mem),
            ("argv check is an identity", mock.patch.object(CS, "arm_states_its_values", lambda a: True), self.o_argv_states_values),
            ("arm name without cap / mem", mock.patch.object(CS, "sv_name", lambda p: MS.bundle_name(p)), self.o_names_distinct),
            ("boot order is not seeded", mock.patch.object(CS, "fb_boot_order", lambda job, rnd, rule=None: CS.fb_boot_keys(rule)), self.o_order_seeded),
            ("boot order is not checked", mock.patch.object(CS, "fb_boot_order", lambda job, rnd, rule=None: mock.ANY), self.o_order_checked),
            ("reading rule never sees a difference", mock.patch.object(MA, "read_label", lambda a, b: ("SLO_DEPENDENT", 9)), self.o_cap_step_up),
            ("arm disagreement resolved silently", mock.patch.object(CA, "_combine", lambda d: "UP" if "UP" in set(d) else real_combine(d)),
             self.o_disagreement),
            ("unmeasured combination accepted", mock.patch.object(CS, "SERVING_COMBOS", CS.SERVING_COMBOS + ((0.6, 128),)), self.o_unmeasured),
            ("PENDING thresholds treated as values", mock.patch.object(MS, "is_pending", lambda v: False), self.o_pending),
            ("pairs across two factors", mock.patch.object(CS, "SERVING_COMBOS", ((0.6, 48), (0.75, 128), (0.75, 256), (0.75, 48), (0.6, 256))),
             self.o_one_factor_pairs),
            ("cell rule drops the state term", mock.patch.object(CS, "s_bytes", lambda m: 0), self.o_table),
            ("cell rule at another mem", mock.patch.object(CS, "FB_MEM", 0.6), lambda: len(CS.predicted_cells("nano", CS.FB_MEM)) == 86),
            ("anchor drift bound is infinite", mock.patch.dict(CS.PROPOSED, {"fb_anchor_drift_ln": (float("inf"), "")}), self.o_anchor_drift),
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
