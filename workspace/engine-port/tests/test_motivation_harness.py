"""CPU tests of the motivation harness (results/motivation/mo_*.py, 2026-11-10).  No GPU, no engine; the only sockets are localhost fake
SSE servers used to check the per-token / chunk-delta recording of the two sealed client paths.  Nothing here shows the harness works on a
GPU: the server-log patterns, the py3.12 image and the vLLM banner are unverified."""
import asyncio
import copy
import hashlib
import json
import math
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
MO = ENG / "results" / "motivation"
sys.path.insert(0, str(MO))
import mo_spec as MS  # noqa: E402
import mo_card as MC  # noqa: E402
import mo_schema as SC  # noqa: E402
import mo_loadgen as ML  # noqa: E402
import mo_runner as MR  # noqa: E402
import mo_analyze as MA  # noqa: E402
import mo_rehearsal as MH  # noqa: E402
import engine_tree as ET  # noqa: E402

SP = MS.SP
POOL = [[40 + (i * 37) % 3000, 16 + (i % 9)] for i in range(500)]       # sb pool format [[input_tokens, output_tokens], ...]
GRID1 = MS.make_grid(3.0, 1.15, 1)
GRID3 = MS.make_grid(3.0, 1.5, 3)
PLUMB = {"PDMUX_RUN_ID": "j", "PDMUX_WORKLOAD_ID": "w", "PDMUX_HEARTBEAT_S": "0", "PDMUX_JOB_NAME": "j", "PDMUX_ENGINE_TREE": "x"}


# ---- fakes ---------------------------------------------------------------------------------------------------------------------
def _flag(cmd, name, default=None):
    return cmd[cmd.index(name) + 1] if name in cmd else default


def banner_of(a, over=None):
    c = a["cmd"]
    d = {"attention_backend": "'%s'" % _flag(c, "--attention-backend", "auto"), "mem_fraction_static": _flag(c, "--mem-fraction-static", "0.8"),
         "chunked_prefill_size": _flag(c, "--chunked-prefill-size", "8192"), "enable_mixed_chunk": str("--enable-mixed-chunk" in c),
         "disable_overlap_schedule": str("--disable-overlap-schedule" in c), "disable_piecewise_cuda_graph": str("--disable-piecewise-cuda-graph" in c),
         "max_running_requests": _flag(c, "--max-running-requests", "48"), "max_mamba_cache_size": _flag(c, "--max-mamba-cache-size", "48"),
         "context_length": _flag(c, "--context-length", "20480"), "random_seed": _flag(c, "--random-seed", "1")}
    d.update(over or {})
    return "server_args=ServerArgs(model_path='x', " + ", ".join(f"{k}={v}" for k, v in d.items()) + ", other=1)\n"


class FakeServer:
    """What engine_tree.py boot + the server would leave behind: command, tree stamp, log banner.  `tamper` perturbs exactly one thing."""

    def __init__(self, bdir, a, tamper=None):
        self.bdir, self.a, self.t = Path(bdir), a, tamper or {}
        self.log = self.bdir / "server.log"
        self.base_url = f"fake://{a['name']}"
        self.cmd = None

    def start(self):
        a, t = self.a, self.t
        if a["engine"] == "sglang":
            argv = [a["python"]] + list(a["cmd"]) + ["--port", "1"]
            self.cmd = [sys.executable, str(SP.ENGINE_TREE_PY), "boot", "--stamp", str(self.bdir / f"tree_{a['name']}.json"), "--"] + \
                       argv + t.get("cmd_extra", [])
            stamp = {"engine_tree": a["tree"], "argv": argv, "pdmux_env": {**PLUMB, **a["env"]}}
            stamp.update(t.get("stamp", {}))
            (self.bdir / f"tree_{a['name']}.json").write_text(json.dumps(stamp))
            txt = banner_of(a, t.get("banner"))
            if "--disable-piecewise-cuda-graph" not in a["cmd"] and not t.get("no_capture"):
                txt += "Capture piecewise CUDA graph end. Time elapsed: 3 s\n"
            if t.get("pw_failed"):
                txt += "Piecewise CUDA Graph failed with error: boom\n"
        else:
            self.cmd = list(a["cmd"]) + ["--port", "1"]
            txt = "INFO Using FLASHINFER backend.\nmax_num_batched_tokens=8192\n"
        self.log.write_text(txt)
        return t.get("status", "ok")

    def stop(self):
        return {"returncode": 0}


def _u(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


CAPS = {"PDMUX": 6.0, "mixed_chunk8192_piecewiseOFF": 4.0, "chunked": 3.5, "mixed_chunk2048": 4.0, "VLLM": 5.0}


def _arm_cap(name):
    if "pdmuxflagsOFF" in name:
        return 4.0
    if "piecewiseON" in name and "PDMUX" not in name:
        return 2.5
    for k, v in CAPS.items():
        if k in name:
            return v
    return 4.0


def fake_replay(api, base_url, reqs, out_path, perf=None):
    """Client twin: arm and rate are read from the request ids; a request fails its SLO with a probability that grows with rate/capacity."""
    out = []
    for q in reqs:
        rid = q["request_id"]
        m = re.search(r"_r([0-9.]+)_s", rid)
        rate = float(m.group(1)) if m else 1.0
        cap = _arm_cap(base_url) if perf is None else perf(base_url, rate)
        bad = _u(rid) < max(0.0, min(1.0, (rate - cap) / cap * 2.0))
        ttft, itl = (4500.0, 90.0) if bad else (700.0, 25.0)
        n, sent = int(q["output_tokens"]), float(q["arrival_s"])
        t0 = 1000.0 + sent + ttft / 1000.0
        stamps = []
        k = 0
        while len(stamps) < n:
            ch = 2 if (k % 4 == 3) else 1
            ch = min(ch, n - len(stamps))
            stamps += [t0 + k * itl / 1000.0] * ch
            k += 1
        rec = {"request_id": rid, "phase": q["phase"], "input_tokens": q["input_tokens"], "requested_output_tokens": n, "observed_output_tokens": n,
               "scheduled_arrival_s": sent, "sent_s": sent, "completion_s": sent + ttft / 1000.0 + n * itl / 1000.0, "t_sent_mono": 1000.0 + sent,
               "t_finished_mono": stamps[-1], "ttft_ms": ttft, "token_t_mono": stamps,
               "token_itl_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])], "error": "", "error_kind": ""}
        out.append(rec)
    by_id = {q["request_id"]: q for q in reqs}
    return float(max(2.0, (reqs[-1]["arrival_s"] if reqs else 0.0) + 1.0)), [ML.normalize(r, by_id) for r in out]


def fake_py(py):
    return "3.12"


def fake_freeze(py):
    return SP.VLLM_VENV_FREEZE_SHA256 if py == SP.VLLM_PY else "rehearsal-any"


def make_server(tamper_by_arm=None):
    tb = tamper_by_arm or {}
    return lambda bdir, a: FakeServer(bdir, a, tb.get(a["name"]))


def rehearsal_setup(job="JR"):
    plan = MH.rehearsal_plan(job)
    return plan, MC.build_card(plan, POOL)


def real_plan(job="JREAL", grid=None, seeds=SP.EVAL_SEEDS, n=12):
    return MS.make_plan(job, grid or GRID3, fixed=dict(MS.REHEARSAL_FIXED), seeds=seeds, n_requests=n, rehearsal=False)


def run_fake_job(plan, tmp, gate=lambda p: None, tamper=None, perf=None):
    card = MC.build_card(plan, POOL)
    out = Path(tmp) / "raw"
    rp = (lambda *a: fake_replay(*a, perf=perf)) if perf else fake_replay
    res = MR.run_job(plan, card, out, make_server(tamper), rp, POOL, gate_fn=gate, py_version_fn=fake_py, freeze_fn=fake_freeze, warmup_s=5.0)
    return out, card, res


# ---- 1. interpreter per arm / arm table --------------------------------------------------------------------------------------------
class TestArms(unittest.TestCase):
    def setUp(self):
        self.arms = {a["key"]: a for a in MS.arm_table(dict(MS.REHEARSAL_FIXED))}

    def test_required_arms_present(self):
        self.assertEqual(set(self.arms), {"SG_MIXED_8192", "SG_CHUNKED_8192", "SG_MIXED_2048", "VLLM", "PDMUX_BUNDLE", "SG_PATCH_OFF", "SG_PIECEWISE_ON"})

    def test_sglang_arms_use_py312_vllm_unchanged(self):
        for k, a in self.arms.items():
            if a["engine"] == "sglang":
                self.assertEqual(a["python"], "/opt/sg312-venv/bin/python", k)
            else:
                self.assertEqual(a["python"], SP.VLLM_PY)
                self.assertEqual(a["cmd"][0], SP.VLLM_PY)
        base = SP.arm("VL-stock")["cmd"]
        mine = self.arms["VLLM"]["cmd"]
        self.assertEqual(base, mine, "rehearsal cap 48 == the sealed value: the vLLM argv is otherwise unchanged")

    def test_boot_cmd_puts_the_arm_interpreter_after_the_engine_tree_wrapper(self):
        a = self.arms["SG_MIXED_8192"]
        cmd = MR.server_cmd(a, "/tmp/x", 1234)
        i = cmd.index("--")
        self.assertEqual(cmd[i + 1], MS.PY312)
        self.assertEqual(cmd[1:4], [str(SP.ENGINE_TREE_PY), "boot", "--stamp"])
        self.assertEqual(ET._interp(cmd[i + 1:]), MS.PY312)          # engine_tree resolves `import sglang` under THIS interpreter
        v = MR.server_cmd(self.arms["VLLM"], "/tmp/x", 1234)
        self.assertEqual(v[0], SP.VLLM_PY)
        self.assertNotIn("--", v)

    def test_names_state_the_bundle_and_react_to_every_treatment_field(self):
        names = [a["name"] for a in self.arms.values()]
        self.assertEqual(len(set(names)), len(names))
        for p in MS.ARM_PARAMS:
            if p["engine"] != "sglang":
                continue
            n0 = MS.bundle_name(p)
            for f, v in (("tree", "patched" if p["tree"] == "pristine" else "pristine"), ("mixed", not p["mixed"]), ("chunk", 777),
                         ("piecewise", "off" if p["piecewise"] == "on" else "on"), ("overlap", "off" if p["overlap"] == "on" else "on")):
                if p["key"] == "PDMUX_BUNDLE" and f == "mixed":
                    continue                                              # chunk -1 disables chunked prefill: no mixed treatment in the bundle
                self.assertNotEqual(MS.bundle_name(dict(p, **{f: v})), n0, (p["key"], f))
        pm = self.arms["PDMUX_BUNDLE"]["name"]
        for token in ("PDMUX-D44", "nonsticky", "budget65536", "chunk-1", "piecewiseOFF", "overlapOFF", "bundle", "patchedtree"):
            self.assertIn(token, pm)
        self.assertIn("pdmuxflagsOFF", self.arms["SG_PATCH_OFF"]["name"])
        self.assertIn("piecewiseON", self.arms["SG_PIECEWISE_ON"]["name"])

    def test_registered_treatments(self):
        a = self.arms
        c = lambda k: a[k]["cmd"]
        self.assertIn("--enable-mixed-chunk", c("SG_MIXED_8192"))
        self.assertNotIn("--enable-mixed-chunk", c("SG_CHUNKED_8192"))
        self.assertEqual(_flag(c("SG_MIXED_2048"), "--chunked-prefill-size"), "2048")
        self.assertIn("--disable-piecewise-cuda-graph", c("SG_MIXED_8192"))
        self.assertIn("--enable-pdmux", c("PDMUX_BUNDLE"))
        self.assertEqual(a["PDMUX_BUNDLE"]["env"]["PDMUX_R2_FIXED_DSM"], "44")
        self.assertIs(a["PDMUX_BUNDLE"]["sticky"], False)

    def test_single_variable_controls(self):
        base = self.arms["SG_MIXED_8192"]
        po = self.arms["SG_PATCH_OFF"]
        self.assertEqual(po["cmd"], base["cmd"])                          # patch-OFF: identical argv, only the tree differs
        self.assertEqual((base["tree"], po["tree"]), ("pristine", "patched"))
        self.assertEqual(po["env"], {})
        pw = self.arms["SG_PIECEWISE_ON"]
        self.assertEqual([x for x in base["cmd"] if x not in pw["cmd"]], ["--disable-piecewise-cuda-graph"])
        self.assertEqual([x for x in pw["cmd"] if x not in base["cmd"]], [])
        self.assertEqual(pw["tree"], base["tree"])
        ch = self.arms["SG_CHUNKED_8192"]
        self.assertEqual([x for x in base["cmd"] if x not in ch["cmd"]], ["--enable-mixed-chunk"])

    def test_common_fixed_flags_are_identical_across_sglang_arms(self):
        for flag in ("--attention-backend", "--max-running-requests", "--max-mamba-cache-size", "--mem-fraction-static", "--context-length",
                     "--random-seed", "--max-prefill-tokens"):
            vals = {_flag(a["cmd"], flag) for a in self.arms.values() if a["engine"] == "sglang"}
            self.assertEqual(len(vals), 1, flag)

    def test_mem_and_cap_are_placeholders_until_the_cap_measurement(self):
        d = MS.arm_table()                                                 # default_fixed()
        for a in d:
            self.assertIn(MS.CAP_TOKEN, a["cmd"], a["name"])
            if a["engine"] == "sglang":
                self.assertIn(MS.MEM_TOKEN, a["cmd"])
        plan = MS.make_plan("JP", GRID3, n_requests=8)
        self.assertTrue(any("PLACEHOLDER" in m for m in MS.plan_problems(plan)))
        self.assertTrue(MC.card_placeholders(MC.build_card(plan, POOL)))
        self.assertEqual(MS.plan_problems(real_plan()), [])


# ---- 3. configuration card -----------------------------------------------------------------------------------------------------------
def observe_for(plan, card, boot, tamper=None, py=fake_py, fr=fake_freeze):
    with tempfile.TemporaryDirectory() as td:
        a = MR.RU.resolve(boot["arm_def"], td)
        srv = FakeServer(td, a, tamper)
        srv.start()
        stamp = json.loads((Path(td) / f"tree_{a['name']}.json").read_text()) if a["engine"] == "sglang" else MR.RU.vllm_stamp(a, {}, srv.log.read_text())
        return MC.observe_boot(a, srv.cmd, stamp, srv.log.read_text(), py, fr, td)


class TestCard(unittest.TestCase):
    def setUp(self):
        self.plan, self.card = rehearsal_setup()
        self.boots = {b["arm_def"]["key"]: b for b in self.plan["boots"] if b["role"] == "meas"}

    def clean(self, key):
        b = self.boots[key]
        obs = observe_for(self.plan, self.card, b)
        return b, obs, MC.check_boot(self.card, b, obs, self.plan["boot_order"][:2], POOL)

    def test_clean_boot_of_every_arm_passes(self):
        for k in self.boots:
            b = self.boots[k]
            obs = observe_for(self.plan, self.card, b)
            executed = self.plan["boot_order"][:self.plan["boot_order"].index(b["arm"]) + 1]
            self.assertEqual(MC.check_boot(self.card, b, obs, executed, POOL), [], (k, obs))

    def test_every_card_field_is_covered_by_a_mutation(self):
        self.assertEqual(set(MUTATIONS), set(MC.CARD_FIELDS))
        self.assertEqual(set(E2E_TAMPER), set(MC.ARM_FIELDS))          # the end-to-end abort is also shown for every arm-level field

    def test_each_field_mutation_fails_the_check_and_names_the_field(self):
        b = self.boots["SG_MIXED_2048"]
        obs0 = observe_for(self.plan, self.card, b)
        executed0 = self.plan["boot_order"][:self.plan["boot_order"].index(b["arm"]) + 1]
        for field, mut in MUTATIONS.items():
            obs, executed, facts = copy.deepcopy(obs0), list(executed0), None
            fn = mut(obs, executed)
            bad = MC.check_arm(self.card, b["arm"], obs) + MC.check_order(self.card, executed)
            if field in ("trace_hash", "trace_quantiles"):
                run = b["runs"][0]
                f = MC.trace_facts(MR.RU.build_segments(run, POOL, []))
                fn(f)
                bad = MC.check_trace(self.card, run["run_id"], f)
            self.assertEqual({m["field"] for m in bad}, {field}, (field, bad))
            self.assertIn(field, MC.format_mismatches("B", "A", bad))

    def test_pool_mutation_fails_hash_and_quantiles(self):
        self.assertEqual(MC.check_pool(self.card, POOL), [])
        pool2 = [list(x) for x in POOL]
        pool2[0][0] += 1                                                # one length changed: sha differs, quantiles may not
        self.assertEqual({m["field"] for m in MC.check_pool(self.card, pool2)}, {"trace_hash"})
        pool3 = [[x[0] + 5000, x[1]] for x in POOL]                     # whole distribution shifted: both fields
        self.assertEqual({m["field"] for m in MC.check_pool(self.card, pool3)}, {"trace_hash", "trace_quantiles"})

    def test_vllm_arm_mutations(self):
        b = self.boots["VLLM"]
        obs0 = observe_for(self.plan, self.card, b)
        order = self.plan["boot_order"][:self.plan["boot_order"].index(b["arm"]) + 1]
        self.assertEqual(MC.check_boot(self.card, b, obs0, order, POOL), [])
        for f, v in (("python_version", "3.14"), ("venv_freeze", "0" * 64), ("cap", 47), ("backend", "TRITON"), ("chunk", 4096), ("context", 4096),
                     ("env", {"CUDA_HOME": "/x"}), ("argv", ["x"])):
            obs = dict(obs0, **{f: v})
            self.assertIn(f, {m["field"] for m in MC.check_arm(self.card, b["arm"], obs)}, f)

    def test_e2e_tamper_aborts_the_job_with_a_clear_reason(self):
        plan, card = self.plan, self.card
        target = next(b for b in plan["boots"] if b["role"] == "meas" and b["arm_def"]["key"] == "SG_MIXED_2048")["arm"]
        for field, (tamper, kw) in E2E_TAMPER.items():
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(MR.JobAbort, msg=field) as cm:
                    MR.run_job(plan, card, Path(td) / "raw", make_server({target: tamper} if tamper else {}), fake_replay, POOL,
                               py_version_fn=kw.get("py", fake_py), freeze_fn=kw.get("fr", fake_freeze), warmup_s=5.0)
                self.assertEqual(cm.exception.status, "ABORT_CARD_MISMATCH", field)
                self.assertIn(f"[{field}]", cm.exception.reason)
                self.assertIn("CARD_MISMATCH", cm.exception.reason)
                self.assertTrue((Path(td) / "raw" / "JOB_ABORTED.json").exists())
                aborted = json.loads((Path(td) / "raw" / "JOB_ABORTED.json").read_text())
                self.assertEqual(aborted["status"], "ABORT_CARD_MISMATCH")
                # the Job stopped at the bad boot: no later boot directory exists
                bad_idx = 0 if field in ("python_version", "venv_freeze") else plan["boot_order"].index(target)   # global fns trip the first boot
                self.assertEqual(len(list((Path(td) / "raw" / "boots").iterdir())), bad_idx + 1, field)
                self.assertFalse(list((Path(td) / "raw" / "runs").glob("*")) and field in ("trace_hash",), field)

    def test_e2e_boot_order_violation_aborts(self):
        plan, card = self.plan, self.card
        swapped = copy.deepcopy(plan)
        swapped["boots"][1], swapped["boots"][2] = swapped["boots"][2], swapped["boots"][1]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:            # the plan itself is invalid first (registered permutation)
                MR.run_job(swapped, card, Path(td) / "raw", make_server(), fake_replay, POOL, py_version_fn=fake_py, freeze_fn=fake_freeze)
            self.assertIn("ABORT_PLAN", cm.exception.status)
        with tempfile.TemporaryDirectory() as td:                 # the card itself is the authority at boot time
            ctx = {"executed": [plan["boots"][2]["arm"]]}
            with self.assertRaises(MR.JobAbort) as cm:
                MR.run_boot(plan["boots"][0], card, Path(td), make_server(), fake_replay, POOL, ctx, fake_py, fake_freeze, warmup_s=5.0)
            self.assertIn("[boot_order]", cm.exception.reason)

    def test_unregistered_card_is_refused_by_run_and_by_check(self):
        plan = MS.make_plan("JU", GRID1, n_requests=8)
        card = MC.build_card(plan, POOL)
        self.assertTrue(MC.card_placeholders(card))
        b = next(x for x in plan["boots"] if x["role"] == "meas")
        bad = MC.check_arm(card, b["arm"], {})
        self.assertTrue(any(m["reason"].startswith("UNREGISTERED") for m in bad))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:
                MR.run_job(plan, card, Path(td) / "raw", make_server(), fake_replay, POOL, gate_fn=lambda p: None)
            self.assertTrue(cm.exception.status in ("ABORT_PLAN", "ABORT_CARD_PLACEHOLDER"))

    def test_card_is_bound_to_its_plan(self):
        plan2 = MH.rehearsal_plan("JR", n_requests=9)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:
                MR.run_job(plan2, self.card, Path(td) / "raw", make_server(), fake_replay, POOL, py_version_fn=fake_py, freeze_fn=fake_freeze)
            self.assertEqual(cm.exception.status, "ABORT_CARD_PLAN")


def _set(k, v):
    def m(obs, executed):
        obs[k] = v
        return None
    return m


def _order(obs, executed):
    executed[-1], executed[-2] = executed[-2], executed[-1]


def _hash(facts_fn=None):
    def m(obs, executed):
        return lambda f: f.update(hash="0" * 64)
    return m


def _quant(obs, executed):
    return lambda f: f.update(in_q=[1, 2, 3])


MUTATIONS = {
    "python_version": _set("python_version", "3.14"), "venv_freeze": _set("venv_freeze", "f" * 64), "tree": _set("tree", "patched"),
    "piecewise": _set("piecewise", "on"), "chunk": _set("chunk", 4096), "mixed": _set("mixed", False), "overlap": _set("overlap", "off"),
    "mem": _set("mem", 0.75), "cap": _set("cap", 128), "context": _set("context", 40960), "backend": _set("backend", "triton"), "seed": _set("seed", 2),
    "env": _set("env", {"PDMUX_R2_POLICY": "fixed"}), "argv": _set("argv", ["x"]), "boot_order": _order, "trace_hash": _hash(), "trace_quantiles": _quant,
}
E2E_TAMPER = {   # field -> (FakeServer tamper, extra run_job kwargs)
    "python_version": (None, {"py": lambda py: "3.14"}), "venv_freeze": (None, {"fr": lambda py: "0" * 64}),
    "tree": ({"stamp": {"engine_tree": "patched"}}, {}), "piecewise": ({"banner": {"disable_piecewise_cuda_graph": "False"}}, {}), "chunk": ({"banner": {"chunked_prefill_size": 4096}}, {}),
    "mixed": ({"banner": {"enable_mixed_chunk": "False"}}, {}), "overlap": ({"banner": {"disable_overlap_schedule": "True"}}, {}),
    "mem": ({"banner": {"mem_fraction_static": 0.75}}, {}), "cap": ({"banner": {"max_running_requests": 128, "max_mamba_cache_size": 128}}, {}),
    "context": ({"banner": {"context_length": 40960}}, {}), "backend": ({"banner": {"attention_backend": "'triton'"}}, {}),
    "seed": ({"banner": {"random_seed": 2}}, {}), "env": ({"stamp": {"pdmux_env": {**PLUMB, "PDMUX_SLO_SCHED": "1"}}}, {}),
    "argv": ({"cmd_extra": ["--enable-pdmux"]}, {}),
}


# ---- 4. order, anchor, grid -----------------------------------------------------------------------------------------------------------
class TestOrderAnchorGrid(unittest.TestCase):
    def test_order_is_seeded_permutation_differs_by_job(self):
        orders = {tuple(MS.boot_order(f"job{i}")) for i in range(12)}
        self.assertGreater(len(orders), 4)
        self.assertEqual(MS.boot_order("jobX"), MS.boot_order("jobX"))
        for o in orders:
            self.assertEqual(sorted(o), sorted(MS.ARM_KEYS))

    def test_anchor_first_and_last_with_registered_cell(self):
        plan = real_plan()
        self.assertEqual([b["role"] for b in plan["boots"]][0], "anchor")
        self.assertEqual(plan["boots"][-1]["role"], "anchor")
        self.assertEqual(plan["boots"][0]["arm"], plan["boots"][-1]["arm"])
        self.assertEqual(plan["boots"][0]["arm"], MS.name_of(MS.ANCHOR_KEY))
        r0, r1 = plan["boots"][0]["runs"][0], plan["boots"][-1]["runs"][0]
        self.assertEqual((r0["rate"], r0["seed"], r0["n_requests"]), (r1["rate"], r1["seed"], r1["n_requests"]))
        self.assertEqual(MS.plan_problems(plan), [])
        self.assertEqual(plan["boot_order"], [b["arm"] for b in plan["boots"]])

    def test_plan_checker_rejects_mutated_order_anchor_and_blocks(self):
        plan = real_plan()
        p = copy.deepcopy(plan)
        p["boots"][2], p["boots"][3] = p["boots"][3], p["boots"][2]
        self.assertTrue(any("registered permutation" in m for m in MS.plan_problems(p)))
        p = copy.deepcopy(plan)
        p["boots"][-1]["role"] = "meas"
        self.assertTrue(MS.plan_problems(p))
        p = copy.deepcopy(plan)
        del p["boots"][3]["runs"][0]
        self.assertTrue(any("full-block" in m for m in MS.plan_problems(p)))
        p = copy.deepcopy(plan)
        p["boots"][1]["arm_def"]["python"] = "/usr/bin/python3"
        self.assertTrue(any("must use" in m for m in MS.plan_problems(p)))
        p = copy.deepcopy(plan)
        p["seeds"] = [1, 2, 3]
        for b in p["boots"]:
            if b["role"] == "meas":
                b["runs"] = [r for r in b["runs"] if r["seed"] in (1, 2, 3)]
        self.assertTrue(any("gate 3" in m for m in MS.plan_problems(p)))

    def test_grid_single_extension(self):
        g = MS.make_grid(3.0, 1.15, 4)
        self.assertEqual(len(g["extension_rates"]), MS.EXT_COUNT)
        self.assertGreater(min(g["extension_rates"]), max(g["rates"]))
        e = MS.make_grid(3.0, 1.15, 4, role="extension")
        self.assertEqual(e["rates"][0], g["rates"][-1])                 # overlap rung, then the extension rungs
        self.assertEqual(e["rates"][1:], g["extension_rates"])
        self.assertIsNone(MS.may_extend(True, ["base"]))
        self.assertIn("NOT_TRIGGERED", MS.may_extend(False, ["base"]))
        self.assertIn("ALREADY_USED", MS.may_extend(True, ["base", "extension"]))
        with self.assertRaises(MS.GridError):
            MS.make_grid(3.0, 1.15, 4, role="second_extension")

    def test_extension_job_plans_like_any_job(self):
        p = MS.make_plan("JEXT", MS.make_grid(3.0, 1.15, 4, role="extension"), fixed=dict(MS.REHEARSAL_FIXED))
        self.assertEqual(MS.plan_problems(p), [])
        self.assertEqual(p["grid"]["role"], "extension")


# ---- 5. raw schema / chunk delta -----------------------------------------------------------------------------------------------------------
def _serve(handler_factory):
    from aiohttp import web
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box = {}

    def run():
        asyncio.set_event_loop(loop)
        app = web.Application()
        handler_factory(app, web)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        box["port"] = site._server.sockets[0].getsockname()[1]
        box["runner"] = runner
        ready.set()
        loop.run_forever()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    ready.wait(10)

    def stop():
        asyncio.run_coroutine_threadsafe(box["runner"].cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
    return box["port"], stop


PATTERN = [1, 1, 3, 1, 2]       # tokens per stream chunk the fake servers emit (8 output tokens)


def _sglang_app(app, web):
    async def gen(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        total = 0
        for k in PATTERN:
            total += k
            await resp.write(("data: " + json.dumps({"text": "x", "meta_info": {"completion_tokens": total, "finish_reason": None}}) + "\n\n").encode())
            await asyncio.sleep(0.02)
        await resp.write(b"data: [DONE]\n\n")
        return resp
    app.router.add_post("/generate", gen)


def _openai_app(app, web):
    async def comp(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        total = 0
        for k in PATTERN:
            total += k
            await resp.write(("data: " + json.dumps({"choices": [{"text": "x", "finish_reason": None}], "usage": {"completion_tokens": total}}) + "\n\n").encode())
            await asyncio.sleep(0.02)
        await resp.write(("data: " + json.dumps({"choices": [], "usage": {"completion_tokens": total}}) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def models(request):
        return web.json_response({"data": [{"id": "m"}]})
    app.router.add_post("/v1/completions", comp)
    app.router.add_get("/v1/models", models)


class TestRawSchemaAndChunkDelta(unittest.TestCase):
    def _reqs(self):
        return [{"request_id": f"q{i}", "phase": "W1", "arrival_s": 0.05 * i, "input_tokens": (100, 1000, 3000)[i], "output_tokens": 8, "ignore_eos": True}
                for i in range(3)]

    def _check(self, api, app):
        port, stop = _serve(app)
        try:
            with tempfile.TemporaryDirectory() as td:
                dur, recs = ML.replay(api, f"http://127.0.0.1:{port}", self._reqs(), Path(td) / "seg.jsonl")
        finally:
            stop()
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertEqual(r["observed_output_tokens"], 8)
            self.assertEqual(len(r["token_t_mono"]), 8)                       # per-token client timestamps
            self.assertEqual(r["chunk_delta"], PATTERN, api)                  # tokens per stream chunk recorded
            self.assertEqual(SC.validate_request(r), [], api)
            self.assertEqual(r["token_itl_ms"].count(0.0), sum(k - 1 for k in PATTERN))
        self.assertEqual([r["req_class"] for r in recs], ["in<=512", "in<=2048", "in>2048"])

    def test_chunk_delta_recorded_for_sglang_client_path(self):
        self._check("sglang", _sglang_app)

    def test_chunk_delta_recorded_for_vllm_client_path(self):
        self._check("openai", _openai_app)

    def test_stream_state_parsers_group_tokens_per_chunk(self):
        from pdmux_eval import trace_loadgen as TL
        import sb_loadgen as LG
        for state, mk in ((TL.StreamState(), lambda c: {"meta_info": {"completion_tokens": c}}),
                          (LG.OpenAIStreamState(), lambda c: {"choices": [{"text": "x", "finish_reason": None}], "usage": {"completion_tokens": c}})):
            tick = iter(range(1, 100))
            total = 0
            for k in PATTERN:
                total += k
                state.feed_line(("data: " + json.dumps(mk(total))).encode(), lambda: float(next(tick)))
            self.assertEqual(SC.chunk_deltas_from_stamps(state.token_times), PATTERN)

    def _good_run(self):
        plan, card = rehearsal_setup()
        with tempfile.TemporaryDirectory() as td:
            out, card, res = run_fake_job(plan, td)
            return plan, [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]

    def test_run_record_has_the_fixed_schema_and_each_removed_field_is_caught(self):
        plan, runs = self._good_run()
        rec = runs[0]
        self.assertEqual(SC.validate_run(rec), [])
        self.assertTrue(rec["realized_split"]["available"] is False and rec["realized_split"]["reason"])
        self.assertEqual(rec["card_sha"], MC.build_card(plan, POOL)["card_sha"])
        for k in SC.RUN_KEYS:
            r = copy.deepcopy(rec)
            del r[k]
            self.assertTrue(SC.validate_run(r), k)
        for k in SC.REQUEST_KEYS:
            r = copy.deepcopy(rec)
            del r["segments"][0]["requests"][0][k]
            self.assertTrue(SC.validate_run(r), k)

    def test_validator_catches_inconsistent_chunk_delta_and_stamps(self):
        _, runs = self._good_run()
        r = copy.deepcopy(runs[0])
        q = r["segments"][0]["requests"][0]
        q["chunk_delta"] = [q["observed_output_tokens"]]
        self.assertTrue(any("chunk_delta" in m for m in SC.validate_run(r)))
        r = copy.deepcopy(runs[0])
        q = r["segments"][0]["requests"][0]
        q["token_t_mono"] = q["token_t_mono"][:-1]
        self.assertTrue(SC.validate_run(r))
        r = copy.deepcopy(runs[0])
        r["segments"][0]["requests"][0]["token_itl_ms"][0] += 5.0
        self.assertTrue(any("gap list" in m for m in SC.validate_run(r)))
        r = copy.deepcopy(runs[0])
        r["realized_split"] = {"available": False}
        self.assertTrue(SC.validate_run(r))


# ---- 6. rehearsal and the launch gate -----------------------------------------------------------------------------------------------------
class TestRehearsal(unittest.TestCase):
    def test_full_pass_over_the_real_arm_list(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "rec"
            rec = MH.rehearse("JREH", POOL, Path(td) / "raw", make_server(), fake_replay, rehearsal_dir=rd, digest="d" * 64,
                              py_version_fn=fake_py, freeze_fn=fake_freeze)
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertTrue(all(rec["stages"].values()), rec["stages"])
            self.assertEqual(len(rec["arm_names"]), len(MS.ARM_KEYS))
            self.assertEqual(rec["verdict_rows"], 1)                              # one rung
            self.assertTrue((Path(td) / "raw" / "analysis" / "verdict_table.csv").exists())
            runs = list((Path(td) / "raw" / "runs").glob("*.json"))
            self.assertEqual(len(runs), len(MS.ARM_KEYS) + 2)                     # each arm once, one rung, one seed + 2 anchor cells
            self.assertTrue((rd / ("d" * 16 + ".json")).exists())

    def test_rehearsal_reports_the_failing_stage_and_writes_no_pass(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "rec"
            plan = MH.rehearsal_plan("JREH")
            target = next(b["arm"] for b in plan["boots"] if b["role"] == "meas")
            rec = MH.rehearse("JREH", POOL, Path(td) / "raw", make_server({target: {"banner": {"mem_fraction_static": 0.9}}}), fake_replay,
                              rehearsal_dir=rd, digest="e" * 64, py_version_fn=fake_py, freeze_fn=fake_freeze)
            self.assertFalse(rec["passed"])
            self.assertIn("ABORT_CARD_MISMATCH", rec["reason"])
            self.assertFalse(rec["stages"]["verdict_table"])

    def test_launch_gate(self):
        plan = real_plan()
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            self.assertIn("NO_REHEARSAL_RECORD", MH.launch_gate(plan, rd, "a" * 64))
            rec = {"harness_digest": "a" * 64, "passed": True, "stages": {s: True for s in MH.STAGES}, "arm_names": sorted({b["arm"] for b in plan["boots"]})}
            (rd / ("a" * 16 + ".json")).write_text(json.dumps(rec))
            self.assertIsNone(MH.launch_gate(plan, rd, "a" * 64))
            self.assertIn("NO_REHEARSAL_RECORD", MH.launch_gate(plan, rd, "b" * 64))            # another code state
            (rd / ("a" * 16 + ".json")).write_text(json.dumps(dict(rec, passed=False, reason="x")))
            self.assertIn("NOT_PASSED", MH.launch_gate(plan, rd, "a" * 64))
            (rd / ("a" * 16 + ".json")).write_text(json.dumps(dict(rec, stages={**rec["stages"], "verdict_table": False})))
            self.assertIn("NOT_PASSED", MH.launch_gate(plan, rd, "a" * 64))
            (rd / ("a" * 16 + ".json")).write_text(json.dumps(dict(rec, arm_names=rec["arm_names"][:-1])))
            self.assertIn("ARM_LIST_DIFFERS", MH.launch_gate(plan, rd, "a" * 64))

    def test_harness_digest_follows_code(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "mo_x.py"
            f.write_text("a=1\n")
            d0 = MH.harness_digest([f])
            f.write_text("a=2\n")
            self.assertNotEqual(MH.harness_digest([f]), d0)
        self.assertEqual(MH.harness_digest(), MH.harness_digest())

    def test_real_plan_refuses_to_launch_without_a_rehearsal_record(self):
        plan = real_plan(n=6)
        card = MC.build_card(plan, POOL)
        with tempfile.TemporaryDirectory() as td:
            gate = lambda p: MH.launch_gate(p, Path(td) / "none", "c" * 64)
            with self.assertRaises(MR.JobAbort) as cm:
                MR.run_job(plan, card, Path(td) / "raw", make_server(), fake_replay, POOL, gate_fn=gate, py_version_fn=fake_py, freeze_fn=fake_freeze)
            self.assertEqual(cm.exception.status, "ABORT_NO_REHEARSAL")
            self.assertFalse((Path(td) / "raw" / "boots").exists())               # nothing booted
            out, _, res = run_fake_job(plan, Path(td) / "ok")                         # with a passing gate the same plan runs
            self.assertEqual(MR.exit_code(res), 0)

    def test_cli_gate_exit_code(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = MR.main(["gate", "--job", "JCLI", "--n", "2"])
        self.assertEqual(rc, 0 if "REHEARSAL_RECORD_OK" in buf.getvalue() else 1)


# ---- 7/8. analyzer -------------------------------------------------------------------------------------------------------------------------
class TestAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.plan = real_plan("JAN", n=40)
        cls.out, cls.card, cls.res = run_fake_job(cls.plan, cls.td.name)
        cls.job = MA.load_job(cls.out)
        cls.result = MA.analyze_job(cls.job)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_job_is_valid_and_labelled(self):
        self.assertEqual(MA.job_problems(self.job)[0], [])
        self.assertEqual(len(self.result["verdict_rows"]), len(GRID3["rates"]))
        self.assertEqual(self.result["cross_job"], "NOT_MERGED")
        self.assertEqual(self.result["flags"], [])

    def test_attainment_per_registered_slo_grid_is_post_hoc(self):
        run = next(r for r in self.job["runs"] if r["boot_role"] == "meas")
        g = MA.attainment_grid(MA.reqs_of(run))
        self.assertEqual(set(g), set((t, i) for t in MS.TTFT_GRID_MS for i in MS.ITL_GRID_MS))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in g.values()))
        # looser SLO never lowers attainment; the same raw answers every grid point without re-running anything
        self.assertGreaterEqual(g[(5000.0, 100.0)], g[(1000.0, 40.0)])
        key = f"{run['arm']}|{run['spec']['rate']}"
        self.assertEqual(set(self.result["attainment_grid"][key]), {f"{t:g}/{i:g}" for t in MS.TTFT_GRID_MS for i in MS.ITL_GRID_MS})

    def test_reading_rule(self):
        P = MA.SLO_PAIRS
        hi = {p: 0.95 for p in P}
        lo = {p: 0.50 for p in P}
        self.assertEqual(MA.read_label(hi, lo)[0], "A>B")
        self.assertEqual(MA.read_label(lo, hi)[0], "A<B")
        mixed = {p: (0.95 if k % 2 else 0.30) for k, p in enumerate(P)}
        self.assertEqual(MA.read_label(mixed, lo)[0], "SLO_DEPENDENT")
        sat = {p: 0.99 for p in P}
        self.assertEqual(MA.read_label(sat, sat)[0], "UNINFORMATIVE")
        dead = {p: 0.0 for p in P}
        self.assertEqual(MA.read_label(dead, dead)[0], "UNINFORMATIVE")
        small = {p: 0.51 for p in P}
        self.assertEqual(MA.read_label(small, {p: 0.50 for p in P})[0], "SLO_DEPENDENT")        # difference inside the margin is no sign

    def test_capacity_estimators_primary_and_secondary(self):
        caps = self.result["capacity"]
        self.assertEqual(set(caps), set(a for a in MA.arms_of(self.job)))
        for a, c in caps.items():
            self.assertIn(c["primary_status"], ("BRACKETED", "LADDER_TOP_PASS", "LADDER_BOTTOM_FAIL"))
            self.assertIn("status", c["secondary_interp"])
        pm = next(a for a in caps if "PDMUX" in a)
        mixed = next(a for a in caps if a.endswith("mixed_chunk8192_piecewiseOFF_overlapON") and "pdmuxflagsOFF" not in a)
        pv, mv = caps[pm]["primary_burst"], caps[mixed]["primary_burst"]
        self.assertTrue(pv is None or mv is None or pv >= mv)                 # fake profile: bundle capacity 6 vs 4
        # unit: interpolation in log-rate between the last passing and the first failing rung
        r = MA._interp([(2.0, 1.0, 1), (4.0, 0.8, 1)])
        self.assertAlmostEqual(r["value"], round(math.exp(math.log(2.0) + 0.5 * math.log(2.0)), 6), places=5)
        self.assertEqual(MA._interp([(2.0, 0.5, 1)])["status"], "BELOW_GRID")
        self.assertEqual(MA._interp([(2.0, 1.0, 1), (3.0, 0.95, 1)])["status"], "ABOVE_GRID_TOP")

    def test_per_class_reporting(self):
        rows = self.result["per_class"]
        self.assertEqual({r["class"] for r in rows}, {"in<=512", "in<=2048", "in>2048"})
        self.assertTrue(all("flag" in r and r["n"] > 0 for r in rows))
        self.assertTrue(all((r["flag"] == "SMALL_N") == (r["n"] < MS.MIN_CLASS_N) for r in rows))

    def test_best_baseline_per_load(self):
        rows = self.result["verdict_rows"]
        base = [a for a in MA.arms_of(self.job) if MA.role_of(self.job, a) == "baseline"]
        self.assertEqual(len(base), 4)
        tab = MA.rung_table(self.job)
        bundle = next(a for a in MA.arms_of(self.job) if MA.role_of(self.job, a) == "bundle")
        for r in rows:
            best = max(tab[(a, r["rate"])]["grid"][MS.PRIMARY_SLO] for a in base)
            self.assertAlmostEqual(r["best_baseline_att_primary"], best)
            self.assertAlmostEqual(r["diff_pp_primary"], tab[(bundle, r["rate"])]["grid"][MS.PRIMARY_SLO] - best)
            self.assertTrue(set(r["vs_each_baseline_primary"]) == set(base))
            self.assertTrue(r["best_baseline_primary"])
        self.assertIn("control", self.result["controls"][0])
        self.assertEqual(len(self.result["controls"]), 2 * len(GRID3["rates"]))

    def test_pdmux_wins_only_where_the_fake_profile_says_so(self):
        lab = {r["rate"]: r["label"] for r in self.result["verdict_rows"]}
        top = max(lab)
        self.assertIn(lab[top], ("A>B", "SLO_DEPENDENT"))

    def test_analyzer_refuses_to_merge_jobs(self):
        with self.assertRaises(MA.CrossJobMergeError):
            MA.merge_points(self.job, self.job)
        self.assertEqual(MA.merge_points(self.job), self.job["runs"])
        with tempfile.TemporaryDirectory() as td:
            j2 = copy.deepcopy(self.job)
            for r in j2["runs"][:3]:
                r["job_name"] = "OTHER-JOB"
            self.assertTrue(any("several Jobs" in m for m in MA.job_problems(j2)[0]))
            with self.assertRaises(MA.JobInvalid):
                MA.analyze_job(j2)

    def test_analyze_jobs_returns_separate_results(self):
        res = MA.analyze_jobs([self.out, self.out])
        self.assertEqual(len(res), 2)
        self.assertTrue(all(r["cross_job"] == "NOT_MERGED" for r in res))

    def test_refusals(self):
        def broken(f):
            j = copy.deepcopy(self.job)
            f(j)
            return MA.job_problems(j)[0]
        self.assertTrue(any("incomplete block" in m for m in broken(lambda j: j["runs"].pop(5))))
        self.assertTrue(any("card sha" in m for m in broken(lambda j: j["runs"][0].update(card_sha="x"))))
        self.assertTrue(any("boot order" in m for m in broken(lambda j: j["boots"].reverse())))
        self.assertTrue(any("trace hash" in m for m in broken(lambda j: next(r for r in j["runs"] if r["boot_role"] == "meas").update(trace_hash="0"))))
        self.assertTrue(any("Job aborted" in m for m in broken(lambda j: j.update(aborted={"status": "ABORT_CARD_MISMATCH", "reason": "x"}))))
        self.assertTrue(any("status" in m for m in broken(lambda j: j["done"].pop(next(iter(j["done"]))))))
        self.assertTrue(any("anchor" in m for m in broken(lambda j: j["runs"].remove(next(r for r in j["runs"] if r["cell"] == "ANCHOR_end")))))

    def test_anchor_drift_flagged(self):
        j = copy.deepcopy(self.job)
        end = next(r for r in j["runs"] if r["cell"] == "ANCHOR_end")
        for q in MA.reqs_of(end):
            q["ttft_ms"] = 99999.0
        self.assertIn("ANCHOR_DRIFT", MA.job_problems(j)[1])

    def test_fewer_than_four_seeds_is_not_judgeable(self):
        plan = real_plan("JN3", seeds=(1, 2, 3), n=40)
        plan["rehearsal"] = False
        with tempfile.TemporaryDirectory() as td:
            card = MC.build_card(plan, POOL)
            problems = MS.plan_problems(plan)
            self.assertTrue(any("gate 3" in m for m in problems))
        # an analyzed Job with 3 seeds (rehearsal-style plan forced through) is labelled, not judged
        plan["rehearsal"] = True
        with tempfile.TemporaryDirectory() as td:
            out, card, res = run_fake_job(plan, td)
            r = MA.analyze_job(out)
            self.assertTrue(all(x["label"] == "REHEARSAL" for x in r["verdict_rows"]))
            job = MA.load_job(out)
            job["plan"]["rehearsal"] = False
            self.assertIn("N_LT_4_SEEDS", MA.job_problems(job)[1])

    def test_outputs_written(self):
        with tempfile.TemporaryDirectory() as td:
            p = MA.write_outputs(self.result, td)
            self.assertTrue(p.exists())
            self.assertIn("rate,", p.read_text().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
