"""CPU tests for the N2(3-a) Job-PROFILE harness (n2_plan, n2_job, n2_capacity.sh, n2_window, n2_realized_mix,
Job extensions of n2_marks / n2_loadgen / fake_n2_engine). HARNESS-SEAL file.
Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_job -v      (needs aiohttp; label rows: n2_reach_tests)
Every contract item has a test that FAILS on its harness mutant in n2_job_mutants.py (lesson 53)."""
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import fake_n2_engine as fk
import n2_analyze as A
import n2_job as J
import n2_loadgen as lg
import n2_marks
import n2_plan as P
import n2_realized_mix as MX
import n2_window as NW

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REG = P.load_registered()
UNITS = P.units("PROFILE", REG)
CFG = os.path.join(P.ENG, P.CONFIG_REL)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def write_reg(td, mut):
    """A synthetic registered file (copy of the sealed one, mutated) and its own sha."""
    with open(P.REGISTERED_JSON) as fh:
        reg = json.load(fh)
    mut(reg)
    raw = json.dumps(reg, indent=1, sort_keys=True).encode()
    p = os.path.join(td, "N2_REGISTERED.json")
    with open(p, "wb") as fh:
        fh.write(raw)
    return p, _sha(raw)


# ===================================================================================================== n2_plan
class TestRegSource(unittest.TestCase):
    def test_sealed_file_loads(self):
        reg = P.load_registered()
        self.assertEqual(reg["K"], {"16": 33, "44": 15})
        self.assertEqual(P.verify_rule_seal(), [])

    def test_hand_edited_file_with_intact_self_report_refused(self):     # H14-2 / H15-5
        with tempfile.TemporaryDirectory() as td:
            def mut(r):
                r["K"] = {"16": 18, "44": 8}
                r["derive_fixed_k"] = {"16": 18, "44": 8}                 # self-report fields stay consistent
                r["budget"]["PROFILE_b"]["cap"] = 99999
            p, own = write_reg(td, mut)
            with self.assertRaises(P.RegSourceError) as cm:
                P.load_registered(p)                                       # sealed digest
            self.assertIn("sha256", cm.exception.msg)
            reg = P.load_registered(p, expected_sha256=own)                # only the byte digest refused it
            self.assertEqual(reg["K"]["16"], 18)

    def test_missing_and_aux_fields(self):
        with self.assertRaises(P.RegSourceError):
            P.load_registered("/nonexistent/N2_REGISTERED.json")
        with tempfile.TemporaryDirectory() as td:
            for mut, why in ((lambda r: r.update(K_matches_derivation=False), "K_matches_derivation"),
                             (lambda r: r.update(calculator_sha256="0" * 64), "calculator_sha256"),
                             (lambda r: r["limits"].update(ok=False), "limits.ok"),
                             (lambda r: r.update(K={"16": 32, "44": 15}), "derive_fixed_k")):
                p, own = write_reg(td, mut)
                with self.assertRaises(P.RegSourceError) as cm:
                    P.load_registered(p, expected_sha256=own)
                self.assertIn(why, cm.exception.msg)
                os.remove(p)
            p = os.path.join(td, "N2_REGISTERED.json")
            raw = b'{"K": 1, "K": 2}'
            with open(p, "wb") as fh:
                fh.write(raw)
            with self.assertRaises(P.RegSourceError):
                P.load_registered(p, expected_sha256=_sha(raw))

    def test_values_come_from_the_registered_file(self):             # single source: no hard-coded copy, no recalc
        with tempfile.TemporaryDirectory() as td:
            def mut(r):
                r["derived"]["profile"]["44,8,2048"]["c_in"] = 1777
                r["derived"]["profile"]["44,8,2048"]["O_P"] = 300.5
                r["budget"]["PROFILE_b"]["per_boot_worst"]["ON44"] = 1234.0
                r["budget"]["PROFILE_b"]["cap"] = 4321
            p, own = write_reg(td, mut)
            reg = P.load_registered(p, expected_sha256=own)
            u = next(x for x in P.units("PROFILE", reg) if x["name"] == "r1_ON44")
            c = next(cc for cc in u["cells"] if (cc["b"], cc["c"]) == (8, 2048))
            self.assertEqual((c["c_in"], c["O"]), (1777, 301))
            self.assertEqual(P.need_s("PROFILE", reg, "r2_ON44"), 1234.0)
            self.assertEqual(P.budget("PROFILE", reg)["cap_s"], 4321.0)
        with open(os.path.join(HERE, "n2_plan.py")) as fh:
            src = fh.read()
        self.assertNotIn("import n2_feasibility", src)
        self.assertNotIn("K_POST_PROBE.json\")", src)

    def test_rule_seal_tamper_detected(self):
        with tempfile.TemporaryDirectory() as td:
            for fn, _ in P.RULE_FILES:
                shutil.copy(os.path.join(HERE, fn), td)
            shutil.copy(os.path.join(HERE, "REGISTRATION_SHA256.txt"), td)
            self.assertEqual(P.verify_rule_seal(td), [])
            with open(os.path.join(td, "n2_probe_reduce.py"), "a") as fh:
                fh.write("\n# tamper\n")
            self.assertTrue(any("n2_probe_reduce.py" in x for x in P.verify_rule_seal(td)))

    def test_harness_seal_section(self):                                 # rev17 R17-2 / H17-6: two-stage seal
        st, bad = P.verify_harness_seal()
        self.assertIn(st, ("UNSEALED", "SEALED"))
        with tempfile.TemporaryDirectory() as td:
            for fn in P.HARNESS_FILES:
                shutil.copy(os.path.join(HERE, fn), td)
            prof, ea = seal_section(td, "PROFILE"), seal_section(td, "EA")
            head = f"## RULE SEAL DIGEST\n{P.RULE_SEAL_DIGEST}\n\n"
            reg_txt = os.path.join(td, "REGISTRATION_SHA256.txt")

            def check(txt, scope):
                with open(reg_txt, "w") as fh:
                    fh.write(head + txt)
                return P.verify_harness_seal(td, reg_txt, scope=scope)
            self.assertEqual(check(prof, "PROFILE"), ("SEALED", []))
            self.assertEqual(check(prof, "EA")[0], "UNSEALED")                  # EA needs the EA section too
            self.assertEqual(check(prof + "\n" + ea, "EA"), ("SEALED", []))
            self.assertEqual(check(ea + "\n" + prof, "EA"), ("SEALED", []))
            self.assertEqual(check(ea, "EA")[0], "UNSEALED")                    # ... and the PROFILE section
            self.assertEqual(check(ea, "PROFILE")[0], "UNSEALED")
            self.assertEqual(check(prof + "\n" + prof, "PROFILE")[0], "BROKEN")    # one section per scope
            self.assertEqual(check(prof.replace("scope PROFILE\n", ""), "PROFILE")[0], "BROKEN")   # no scope tag
            self.assertEqual(check(prof.replace("scope PROFILE", "scope ALL"), "PROFILE")[0], "BROKEN")
            self.assertEqual(check(prof.replace("scope PROFILE\n", "scope PROFILE\nscope EA\n"), "EA")[0], "BROKEN")
            self.assertEqual(check(prof.replace(f"rule_seal_digest {P.RULE_SEAL_DIGEST}", "rule_seal_digest " + "0" * 64),
                                   "PROFILE")[0], "BROKEN")
            self.assertEqual(check(prof + "\n" + ea.replace(f"rule_seal_digest {P.RULE_SEAL_DIGEST}",
                                                             "rule_seal_digest " + "0" * 64), "EA")[0], "BROKEN")
            self.assertEqual(check(prof + "\n" + ea.replace(f"rule_seal_digest {P.RULE_SEAL_DIGEST}",   # EA content
                                                             "rule_seal_digest " + "0" * 64), "PROFILE"),   # is not
                             ("SEALED", []))                                                    # a PROFILE input
            with open(os.path.join(td, "n2_window.py"), "a") as fh:
                fh.write("\n#x\n")
            self.assertEqual(check(prof, "PROFILE")[0], "BROKEN")
            self.assertEqual(check(prof + "\n" + ea, "EA")[0], "BROKEN")
            self.assertEqual(check("", "PROFILE")[0], "UNSEALED")
            self.assertEqual(P.verify_harness_seal(td, reg_txt, scope="BOTH")[0], "BROKEN")
            r = subprocess.run([sys.executable, os.path.join(HERE, "n2_plan.py"), "--verify-harness-seal", "--scope", "EA"],
                               capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertIn("scope=EA", r.stdout)
            self.assertEqual(r.returncode, 0 if r.stdout.startswith("SEALED") else 3)


def seal_section(here, scope):
    """A well-formed harness-seal section of `scope` over the CURRENT bytes of the harness files in `here`."""
    listing = P.harness_listing(here)
    return (f"## HARNESS SEAL\nscope {scope}\nrule_seal_digest {P.RULE_SEAL_DIGEST}\n{listing}"
            f"harness_digest {_sha(listing.encode())}\n")


class TestUnits(unittest.TestCase):
    def test_structure(self):                                            # §5-1
        self.assertEqual(P.plan_problems("PROFILE", UNITS), [])
        self.assertEqual(len(UNITS), 28)
        self.assertEqual(sum(len(u["cells"]) for u in UNITS), 296)
        for r in (1, 2, 3, 4):
            tags = [u["cost_key"] for u in UNITS if u["round"] == r]
            self.assertEqual(tags, list(P.PROFILE_ORDER[r]))
        self.assertEqual(P.PROFILE_ORDER[1], ("OBS44", "ON44", "OFF44", "ON24", "ON16", "OFF16", "ON34"))
        self.assertEqual(P.PROFILE_ORDER[4][-3:], ("OFF44", "ON44", "OBS44"))        # r4 tail (N2C-17)
        self.assertEqual({u["name"] for u in UNITS if u["greedy"]}, P.GREEDY_BOOTS)
        for u in UNITS:
            self.assertEqual(u["sticky"], u["kind"] in ("ON", "OBS"))
            self.assertEqual(u["phase_events"], u["kind"] != "OBS")
            self.assertEqual(u["alone_D"], 108 if u["kind"] == "OFF" else u["D"])
            self.assertEqual(len(u["cells"]), 2 if u["kind"] == "OBS" else 12)

    def test_cell_order_alternation(self):                               # §5-3 / §14-3 (b outer, c inner)
        order = {r: [(c["b"], c["c"]) for c in next(u for u in UNITS if u["round"] == r and u["kind"] == "ON")["cells"]]
                 for r in (1, 2, 3, 4)}
        self.assertEqual(order[1][:4], [(1, 512), (1, 2048), (1, 8192), (8, 512)])
        self.assertEqual(order[2][:4], [(1, 8192), (1, 2048), (1, 512), (8, 8192)])
        self.assertEqual(order[3][:4], [(48, 512), (48, 2048), (48, 8192), (32, 512)])
        self.assertEqual(order[4][:4], [(48, 8192), (48, 2048), (48, 512), (32, 8192)])
        obs = {r: [(c["b"], c["c"]) for c in next(u for u in UNITS if u["round"] == r and u["kind"] == "OBS")["cells"]]
               for r in (1, 2, 3, 4)}
        self.assertEqual(obs[1], [(1, 2048), (32, 2048)])
        self.assertEqual(obs[3], [(32, 2048), (1, 2048)])

    def test_cells_from_registered_values(self):                         # §4-2 / §8-4 (iii) OBS uses c_in(44)
        der = REG["derived"]["profile"]
        for u in UNITS:
            for c in u["cells"]:
                d = der[f"{u['alone_D']},{c['b']},{c['c']}"]
                self.assertEqual(c["c_in"], int(d["c_in"]))
                self.assertEqual(c["O"], -(-d["O_P"] // 1))
                self.assertEqual(c["seed_kind"], "PROFILE")
                self.assertEqual(c["r"], u["round"])
        obs = next(u for u in UNITS if u["kind"] == "OBS")
        self.assertEqual(obs["alone_D"], 44)

    def test_budget_and_admission(self):                                 # §14-3 STOP rule inputs
        b = P.budget("PROFILE", REG)
        self.assertEqual(b["column"], "PROFILE_b")
        self.assertEqual(b["cap_s"], 36700.0)
        self.assertEqual(P.need_s("PROFILE", REG, "r3_OBS44"), REG["budget"]["PROFILE_b"]["per_boot_worst"]["OBS44"])
        self.assertTrue(P.admission(1385.0, 1384.6))
        self.assertFalse(P.admission(1384.0, 1384.6))

    def test_trace_problems(self):
        u = dict(UNITS[1], cells=[dict(UNITS[1]["cells"][0], c_in=8192), dict(UNITS[1]["cells"][1], O=0)])
        self.assertEqual(len(P.trace_problems(u)), 2)

    def test_boot_env(self):                                              # §3-1 per-boot knobs
        base = {"PDMUX_TRUE_DUAL_WORKER": "1", "PDMUX_STICKY_PARTITION": "1", "PDMUX_PHASE_EVENTS": "1",
                "PDMUX_ROOT": "/r", "PDMUX_ENERGY_SAMPLER": "0", "X": "1"}
        by = {u["cost_key"]: u for u in UNITS}
        on, off, obs = (P.boot_env(base, by[k], "t", "g", "r") for k in ("ON16", "OFF44", "OBS44"))
        self.assertEqual((on["PDMUX_STICKY_PARTITION"], on["PDMUX_PHASE_EVENTS"], on["PDMUX_R2_FIXED_DSM"]),
                         ("1", "1", "16"))
        self.assertNotIn("PDMUX_STICKY_PARTITION", off)
        self.assertEqual((off["PDMUX_PHASE_EVENTS"], off["PDMUX_R2_FIXED_DSM"]), ("1", "44"))
        self.assertNotIn("PDMUX_PHASE_EVENTS", obs)
        self.assertEqual(obs["PDMUX_STICKY_PARTITION"], "1")
        for e in (on, off, obs):
            self.assertNotIn("PDMUX_TRUE_DUAL_WORKER", e)
            self.assertEqual((e["PDMUX_R2_POLICY"], e["PDMUX_GREEN_READOUT"], e["PDMUX_ROOT"]), ("fixed", "1", "/r"))

    def test_server_flags_equal_n3_text(self):                           # §3-1 same text as N3 rev6
        with open(os.path.join(HERE, "..", "n3_capacity", "n3_capacity.sh")) as fh:
            sh = fh.read()
        block = sh[sh.index("SRV=(python -m sglang.launch_server"):]
        block = block[:block.index(")")]
        vals = {"$MODEL": P.MODEL, "$BACKEND": "flashinfer", "$CFG": "CFG", "$MEM_FRACTION": "0.82",
                "$MAX_RUNNING": "48", "$CTX": "16384", "$SERVER_SEED": "1"}
        toks = [vals.get(t.strip('"'), t.strip('"')) for t in block.split()[3:]]
        self.assertEqual(toks[:toks.index("--host")], P.server_flags("CFG"))


class TestChecks(unittest.TestCase):
    def test_config_parser(self):                                         # §8-1, lesson 258
        self.assertEqual(P.check_config(CFG), [])
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "c.yml")
            with open(CFG) as fh:
                txt = fh.read()
            with open(bad, "w") as fh:
                fh.write(txt.replace("  - [84, 24, 18]", "  - [84, 24, 18]\n  - [92, 16, 20]"))
            probs = P.check_config(bad)
            self.assertTrue(any("D16" in x for x in probs))
            self.assertTrue(any("digest" in x for x in probs))

    def test_green(self):                                                 # §11-1
        self.assertEqual(P.check_green(fk.green_readout(4), True, 44), [])
        self.assertEqual(P.check_green(fk.green_readout(None), False, 16), [])
        self.assertTrue(P.check_green(fk.green_readout(3), True, 44))
        self.assertTrue(P.check_green(fk.green_readout(None), True, 44))
        self.assertTrue(P.check_green(fk.green_readout(1), False, 16))
        g = fk.green_readout(4)
        g["sticky_target_index"] = 3                 # header disagrees with the (correct) per-group flags
        self.assertTrue(P.check_green(g, True, 44))
        g = fk.green_readout(4)
        g["groups"][5]["decode"] = {"green_ctx_is_null": False, "green_sm": {"smCount": 108}}
        self.assertTrue(P.check_green(g, True, 44))
        g = fk.green_readout(4)
        g["groups"][2]["prefill"]["green_sm"]["smCount"] = 80
        self.assertTrue(P.check_green(g, True, 44))

    def test_wiring(self):
        line = "PD-mux sticky partition ENABLED: decode holds ... (fixed target index=4); the unpartitioned ..."
        self.assertEqual(P.check_wiring(line, True, 44), [])
        self.assertTrue(P.check_wiring(line, True, 34))
        self.assertTrue(P.check_wiring(line + "\n" + line, True, 44))
        self.assertTrue(P.check_wiring("", True, 44))
        self.assertTrue(P.check_wiring(line, False, 44))
        self.assertEqual(P.check_wiring("nothing", False, 44), [])

    def test_p3_matches_n3_rule(self):                                    # §3-2 same text/constants as N3 R6
        spec = importlib.util.spec_from_file_location("_n3p", os.path.join(HERE, "..", "n3_capacity", "n3_plan.py"))
        n3 = importlib.util.module_from_spec(spec)
        sys.dont_write_bytecode = True
        spec.loader.exec_module(n3)
        for rows in ([(48, 2722025)] * 9 + [(48, 2722022)], [(48, 2721290), (48, 2722025)], [(47, 2722025)],
                     [(48, 700000)], [(48, 2721290), (48, 2700000)]):
            ok, _ = P.p3_verdict(rows)
            ok3, _ = n3.p3_verdict([[str(m), str(t)] for m, t in rows], 48, 16384)
            self.assertEqual(ok, ok3, rows)
        self.assertFalse(P.p3_verdict([(None, 2722025)])[0])

    def test_cg_bs(self):
        self.assertEqual(P.cg_bs_problems("Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]"), [])
        self.assertTrue(P.cg_bs_problems("Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40]"))
        self.assertTrue(P.cg_bs_problems("no line"))

    def test_substrate(self):
        self.assertEqual(P.check_substrate("NVIDIA A100-SXM4-80GB, 8.0, 81920, 550.54.15, Disabled", 108), [])
        self.assertTrue(P.check_substrate("NVIDIA GeForce RTX 5060 Ti, 12.0, 16311, 580.1, [N/A]", 36))

    def test_engine_manifest_equals_n3_registered(self):
        spec = importlib.util.spec_from_file_location("_n3p2", os.path.join(HERE, "..", "n3_capacity", "n3_plan.py"))
        n3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(n3)
        self.assertEqual(P.ENGINE_MANIFEST, n3.ENGINE_MANIFEST)


# ===================================================================================================== realized mix
class TestRealizedMix(unittest.TestCase):
    def test_positive_and_negative_control(self):                         # §15 n2_realized_mix row
        st, det = MX.positive_control()
        if st == "MISSING":
            self.skipTest("λ0 VESSL archive not present (git-ignored)")
        self.assertEqual((st, det), ("OK", []))

    def test_blind_scope(self):                                            # H15-4
        ev = [dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=float(i), decode_iterations=i,
                   decode_running_batch_size=1, prefill_sms=64, decode_sms=44, prefill_queue_depth=0) for i in range(30)]
        with self.assertRaises(MX.BlindedScope):
            MX.estimators(ev, (64, 44), "cell")
        with self.assertRaises(MX.BlindedScope):
            MX.estimators(ev, (64, 44), "boot")
        r = MX.estimators(ev, (64, 44), "cell", blind_exception=True)
        self.assertIn("N2C-73", r["blind_exception"])
        w = MX.estimators(ev, (64, 44), "window", 0.0, 29.0)
        self.assertEqual(w["e_cnt"]["pct"], 100.0)
        self.assertIn("empty denominator", w["e_qcond"]["na"])
        w = MX.estimators(ev, (64, 44), "window", 0.0, 8.0)                 # 9 busy snapshots -> NA
        self.assertIsNone(w["e_time"]["pct"])
        self.assertEqual(MX.estimators(ev, (64, 44), "boot", job="EA")["e_cnt"]["pct"], 100.0)

    def test_a2_pair_definitions(self):
        ev = [dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t, decode_iterations=it,
                   decode_running_batch_size=busy, prefill_sms=p, decode_sms=d, prefill_queue_depth=q)
              for t, it, busy, p, d, q in ((0.0, 0, 1, 64, 44, 1), (1.0, 2, 1, 0, 108, 0), (4.0, 3, 0, 108, 0, 0),
                                           (5.0, 7, 1, 64, 44, 0), (6.0, 8, 1, 64, 44, 0))]
        r = MX.a2_pair(ev, (64, 44))
        self.assertEqual((r["e_cnt"]["num"], r["e_cnt"]["den"]), (3, 4))
        self.assertEqual((r["e_time"]["num_s"], r["e_time"]["den_s"]), (2.0, 5.0))    # gaps 1, 3 (busy), 1
        self.assertEqual((r["e_iter"]["num"], r["e_iter"]["den"]), (3, 4))           # deltas 2 (pi), 1 (108), 1
        self.assertEqual((r["e_qcond"]["num"], r["e_qcond"]["den"]), (1, 1))


# ===================================================================================================== n2_window units
class TestWindowUnits(unittest.TestCase):
    def test_delta_dropped_and_f6(self):                                  # §6-4
        recs = [dict(event="x", timestamp_monotonic_s=t, dropped_events=d) for t, d in ((1, 0), (2, 0), (3, 1), (5, 1))]
        self.assertEqual(NW.delta_dropped(recs, 1.5, 2.5), 1)               # loss shows in the FIRST record after
        self.assertEqual(NW.delta_dropped(recs, 3.5, 4.5), 0)
        self.assertIsNone(NW.delta_dropped(recs, 5.5, 6.0))
        self.assertEqual(NW.delta_dropped(recs, 0.0, 0.5), 0)

    def test_bench_valid(self):                                           # HARNESS READING under U-1 (b)
        cell = dict(b=2, c_in=100, O=10)
        q = lambda **k: dict(dict(rid="a", n=5, prompt_tokens=100, done=False, finish=None), **k)
        self.assertTrue(NW.bench_valid(dict(cancel_fired=True, requests=[q(), q()]), cell)[0])
        self.assertFalse(NW.bench_valid(dict(cancel_fired=True, requests=[q()]), cell)[0])
        self.assertFalse(NW.bench_valid(dict(cancel_fired=True, requests=[q(), q(prompt_tokens=99)]), cell)[0])
        self.assertFalse(NW.bench_valid(dict(cancel_fired=True, requests=[q(), q(prompt_tokens=None)]), cell)[0])
        self.assertFalse(NW.bench_valid(dict(cancel_fired=False, requests=[q(), q()]), cell)[0])
        self.assertTrue(NW.bench_valid(dict(cancel_fired=False, requests=[q(n=10, done=True, finish="length")] * 2),
                                       cell)[0])
        self.assertFalse(NW.bench_valid(dict(aborted="cell_timeout"), cell)[0])

    def test_warmup_cg_canary(self):                                      # §18-2-d #16 (H17-3), pure
        a, z = (1000.0, 10.0), (1010.0, 20.0)                             # wall = 990 + mono
        T, F = "Decode batch, #running-req: 1, cuda graph: True", "Decode batch, #running-req: 1, cuda graph: False"
        lines = [(1000, F), (1009, F), (1001, T), (1008, T), (1005, "Prefill batch, #new-seq: 1, cuda graph: False")]
        self.assertEqual(NW.warmup_cg_canary(lines, a, z, 10.5, 19.5), (True, 2, 0, 1001, 1008))   # [ceil, floor - 1]
        self.assertEqual(NW.warmup_cg_canary(lines + [(1004, F)], a, z, 10.5, 19.5)[:3], (False, 3, 1))
        self.assertEqual(NW.warmup_cg_canary(lines[:2], a, z, 10.5, 19.5)[:3], (False, 0, 0))    # none inside
        self.assertEqual(NW.warmup_cg_canary([(1001, "Decode batch, #running-req: 1")], a, z, 10.5, 19.5)[:3],
                         (False, 1, 1))                                   # no cuda-graph field = not True
        self.assertFalse(NW.warmup_cg_canary(lines, None, z, 10.5, 19.5)[0])
        self.assertEqual(NW.warmup_cg_canary(lines, a, None, 10.5, 19.5)[3:], (1001, 1008))    # start offset only
        gm = 9 * 3600                                                     # log stamps are local time (§18-2-d #15)
        ll = NW.parse_log(time.strftime("[%Y-%m-%d %H:%M:%S] ", time.gmtime(1001 + gm)) + T + "\n", gm)
        self.assertEqual(ll, [(1001, T)])
        self.assertTrue(NW.warmup_cg_canary(ll, a, z, 10.5, 19.5)[0])

    def test_obs_client_window(self):                                     # §11-4
        mk = lambda n, dt: dict(times=[1.0 + k * dt for k in range(n)])
        c, n = NW.obs_client_window(dict(requests=[mk(200, 0.01), mk(200, 0.01)], t_cancel=None))
        self.assertAlmostEqual(c, 10.0, places=6)
        self.assertEqual(n, 198)                                          # up to (not incl.) the first end
        c, n = NW.obs_client_window(dict(requests=[mk(120, 0.01)]))
        self.assertIsNone(c)


# ===================================================================================================== runner (fake)
FAST = dict(chunk_every=1, tau_s=0.0005, burst_span_s=0.001, decode_log_every=8)


def run_mini(td, names, cells, engine_kw=None, handles=None, deadline=None, **opt):
    sw = J.ScaledWall(200.0)
    us = J.select_test_units(UNITS, set(names), set(cells))
    us = sorted(us, key=lambda u: names.index(u["name"]))
    kw = dict(FAST, wall=sw, **(engine_kw or {}))

    def make(u):
        h = (handles or {}).get(u["name"]) or fk.FakeServerHandle(dict(kw), job_mode=True)
        if isinstance(h, dict):
            h = fk.FakeServerHandle(dict(kw, **h.get("engine_kw", {})), job_mode=True,
                                    green_override=h.get("green"), log_extra=h.get("log_extra", ""),
                                    fail_start=h.get("fail_start"))
        return h
    opts = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=1, install_signals=False,
                cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20), **opt)
    if deadline is not None:
        opts["deadline"] = time.monotonic() + deadline
    led = asyncio.run(J.run_job(td, us, make, opts))
    return led


def marks(td, b):
    with open(os.path.join(td, f"marks_{b}.txt")) as fh:
        return [ln.split() for ln in fh.read().splitlines() if ln.strip()]


def tel(td, b):
    with open(os.path.join(td, f"tel_{b}.jsonl")) as fh:
        return [json.loads(x) for x in fh if x.strip()]


class TestRunnerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="n2job_e2e_")
        cls.names = ["r1_OBS44", "r1_ON44", "r1_OFF44"]
        cls.led = run_mini(cls.td, cls.names, [(1, 512), (1, 2048), (32, 2048)])
        cls.res = A.analyze(cls.td, reg=REG)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def test_ledger_and_files(self):
        self.assertEqual([r["status"] for r in self.led], ["ADMITTED"] * 3)
        for b in self.names:
            for k in ("marks", "tel", "srv", "green", "wall", "client", "boot", "cpu"):
                self.assertTrue(any(f.startswith(f"{k}_{b}.") for f in os.listdir(self.td)), (k, b))
        self.assertTrue(os.path.exists(os.path.join(self.td, "greedy_r1_ON44.json")))
        self.assertFalse(os.path.exists(os.path.join(self.td, "greedy_r1_OFF44.json")))

    def test_marks_protocol(self):                                         # boot_plan first, warm-up outside cells
        m = marks(self.td, "r1_ON44")
        self.assertEqual(m[0], ["boot_plan", "44", "3"])
        self.assertEqual([x[0] for x in m[1:3]], ["warmup_start", "warmup_end"])
        warm = {}
        cells, calls, plan, aborts, issues = J.RED.parse_marks([" ".join(x) for x in m], warm)
        self.assertEqual(issues, [])
        w0, w1 = warm["r1_ON44"]["warmup_start"], warm["r1_ON44"]["warmup_end"]
        self.assertTrue(all(cm["cell_start"][1] > w1 for cm in cells.values()))
        self.assertEqual(len(cells), 3)
        self.assertEqual(sorted(cm["cell_start"][0] for cm in cells.values()),
                         sorted([(44, 1, 512), (44, 1, 2048), (44, 32, 2048)]))
        self.assertTrue(all("fix_s" in cm and "cell_end" in cm for cm in cells.values()))
        self.assertEqual(marks(self.td, "r1_OFF44")[0], ["boot_plan", "44", "3"])

    def test_f6_record_after_last_cell(self):                              # §6-4 F-6
        last_end = max(float(x[2]) for x in marks(self.td, "r1_OFF44") if x[0] == "cell_end")
        self.assertTrue(any(e["event"] == "prefill_span_start" and e["timestamp_monotonic_s"] > last_end
                            for e in tel(self.td, "r1_OFF44")))

    def test_partitions_and_phase_events(self):                            # sticky ON vs OFF, OBS PHASE_EVENTS off
        on = [e for e in tel(self.td, "r1_ON44") if e["event"] == "decode_iteration" and not e["prefill_in_flight"]]
        off = [e for e in tel(self.td, "r1_OFF44") if e["event"] == "decode_iteration" and not e["prefill_in_flight"]]
        self.assertEqual({(e["prefill_sms"], e["decode_sms"]) for e in on}, {(64, 44)})
        self.assertEqual({(e["prefill_sms"], e["decode_sms"]) for e in off}, {(0, 108)})
        self.assertFalse([e for e in tel(self.td, "r1_OBS44") if e["event"] == "decode_iteration"])

    def test_replicates_valid_end_to_end(self):                           # runner -> n2_window -> n2_analyze
        st = {r["cell_id"]: r["status"] for r in self.res["replicates"] if r["boot"] in self.names}
        for b in ("r1_ON44", "r1_OFF44"):
            for bc in ("b1-c512", "b1-c2048", "b32-c2048"):
                self.assertEqual(st[f"{b}-{bc}"], "VALID", (b, bc, st))
        self.assertEqual(st["r1_OBS44-b1-c2048"], "VALID")
        self.assertEqual(st["r1_OBS44-b32-c2048"], "VALID")
        self.assertEqual(self.res["neg_control"]["label"], "NEG_CONTROL_HELD")
        cell = next(c for c in self.res["observer"]["cells"].values() if c["n"] == 1)
        self.assertLess(abs(cell["x"][0]), 1.0)

    def test_cancel_rule_and_client_record(self):                          # §4-2 step 5 (PROFILE)
        with open(os.path.join(self.td, "client_r1_ON44.jsonl")) as fh:
            recs = {r["cell_id"]: r for r in map(json.loads, fh)}
        r = recs["r1_ON44-b32-c2048"]
        self.assertTrue(r["cancel_fired"])
        lo = lg.lower_median([q["n"] for q in r["requests"]])
        self.assertGreaterEqual(r["c_in"] + lo, 2048 + 71)
        self.assertEqual(len(r["requests"]), 32)
        self.assertEqual({q["prompt_tokens"] for q in r["requests"]}, {r["c_in"]})


class TestRunnerAborts(unittest.TestCase):
    CELLS = [(1, 512)]

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="n2job_ab_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_stop_rule_no_skipping(self):                                  # §14-3: first refusal stops the Job
        led = run_mini(self.td, ["r1_ON44", "r1_OBS44"], self.CELLS, deadline=1000.0)
        self.assertEqual([r["status"] for r in led], ["NOT_ATTEMPTED_BUDGET", "NOT_ATTEMPTED_BUDGET"])
        self.assertFalse(os.path.exists(os.path.join(self.td, "marks_r1_OBS44.txt")))
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            self.assertEqual(json.load(fh)["stopped"], "budget")

    def test_admitted_when_budget_covers(self):
        led = run_mini(self.td, ["r1_OBS44"], [(1, 2048)], deadline=400.0)
        self.assertEqual(led[0]["status"], "ADMITTED")

    def test_wiring_abort_exit6(self):                                     # §11-1
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_ON44", "r1_OFF44"], self.CELLS,
                     handles={"r1_ON44": dict(engine_kw=dict(sticky_log=False))})
        self.assertEqual((cm.exception.code, cm.exception.tag), (6, "ABORT_STICKY"))
        self.assertIn(["boot_abort", "wiring_check_failed"], [x[:2] for x in marks(self.td, "r1_ON44")])
        self.assertFalse(os.path.exists(os.path.join(self.td, "marks_r1_OFF44.txt")))
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            led = json.load(fh)
        self.assertEqual((led["stopped"], led["job_abort"]["tag"]), ("job_abort", "ABORT_STICKY"))
        res = A.analyze(self.td, reg=REG)                                   # H17-2 / §18-2-d #10
        self.assertEqual(res["forecasts"]["F-N2-7P"], "UNRESOLVED")

    def test_green_abort_exit6(self):
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_OFF44"], self.CELLS, handles={"r1_OFF44": dict(green=fk.green_readout(4))})
        self.assertEqual((cm.exception.code, cm.exception.tag), (6, "ABORT_GREEN"))

    def test_p3_abort_exit8(self):
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_ON44"], self.CELLS,
                     handles={"r1_ON44": dict(log_extra="max_total_num_tokens=700000\n")})
        self.assertEqual(cm.exception.code, 8)

    def test_cg_bs_abort_exit3(self):
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_ON44"], self.CELLS,
                     handles={"r1_ON44": dict(log_extra="Capture cuda graph bs [1, 2, 4, 8]\n")})
        self.assertEqual((cm.exception.code, cm.exception.tag), (3, "ABORT_CG_BS"))

    def test_greedy_empty_exit9(self):                                      # §11-2
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_ON44"], self.CELLS, handles={"r1_ON44": dict(engine_kw=dict(greedy_mode="empty"))})
        self.assertEqual(cm.exception.code, 9)

    def test_vi_boot_death_next_boot_runs(self):                           # G-10
        led = run_mini(self.td, ["r1_ON44", "r1_OFF44"], self.CELLS, handles={"r1_ON44": dict(fail_start="exit")})
        self.assertEqual((led[0]["aborted"], led[0]["abort_class"]), ("server_down", "VALUE_INDEPENDENT"))
        self.assertEqual((led[1]["status"], led[1]["aborted"]), ("ADMITTED", None))

    def test_vd_cell_abort_next_boot_runs(self):                          # G-10 Job (§18-2-d #7, rev17 user decision)
        led = run_mini(self.td, ["r1_ON44", "r1_OFF44"], [(1, 512), (1, 2048)],
                       handles={"r1_ON44": dict(engine_kw=dict(drop_after=100))})   # warm-up output 64 < 100
        self.assertEqual((led[0]["aborted"], led[0]["abort_class"]), ("client_exception", "VALUE_DEPENDENT"))
        self.assertEqual(led[0]["cells_done"], 0)
        self.assertEqual((led[1]["status"], led[1].get("aborted"), led[1].get("cells_done")), ("ADMITTED", None, 2))
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            jl = json.load(fh)
        self.assertEqual((jl["stopped"], jl["job_abort"]), (None, None))
        m = marks(self.td, "r1_ON44")
        self.assertIn("cell_abort", [x[0] for x in m])
        self.assertEqual([x[0] for x in m].count("cell_start"), 1)          # the boot ended at the aborted cell
        res = A.analyze(self.td, reg=REG)
        st = {r["cell_id"]: (r["status"], r["reason"]) for r in res["replicates"]}
        self.assertEqual(st["r1_ON44-b1-c512"][0], "BENCH_INVALID")
        self.assertEqual(st["r1_ON44-b1-c2048"], ("BENCH_INVALID", "not_attempted_boot_aborted"))
        self.assertEqual(st["r1_OFF44-b1-c512"][0], "VALID")
        self.assertEqual(st["r1_OFF44-b1-c2048"][0], "VALID")

    def test_runner_crash_marks_crash_and_f7p_unresolved(self):          # H17-8 (§18-2-d #10)
        # D6: a server.start failure is now a CLASSIFIED boot abort (test_start_failure_is_classified_boot_abort);
        # the runner crash path is exercised by a failure OUTSIDE run_boot (the server factory of run_job).
        sw = J.ScaledWall(200.0)
        us = sorted(J.select_test_units(UNITS, {"r1_ON44", "r1_OFF44"}, set(self.CELLS)),
                    key=lambda u: u["name"] != "r1_ON44")

        def make(u):
            if u["name"] == "r1_OFF44":
                raise OSError("synthetic runner crash")
            return fk.FakeServerHandle(dict(FAST, wall=sw), job_mode=True)
        opts = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=1, install_signals=False,
                    cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        with self.assertRaises(OSError):
            asyncio.run(J.run_job(self.td, us, make, opts))
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            jl = json.load(fh)
        self.assertEqual((jl["stopped"], jl["job_abort"]), ("crash", None))
        res = A.analyze(self.td, reg=REG)
        self.assertEqual(res["forecasts"]["F-N2-7P"], "UNRESOLVED")

    def test_start_failure_is_classified_boot_abort(self):                # D6: settle / start inside the try
        stopped = {}

        class StartFails(fk.FakeServerHandle):
            def start(self, env, log_path):
                raise OSError("synthetic start failure")

            async def stop(self):
                stopped[True] = time.monotonic()
                await super().stop()

        class SettleFails(fk.FakeServerHandle):
            async def settle(self, baseline_mib, record_path):
                raise RuntimeError("synthetic settle failure")
        sw = J.ScaledWall(200.0)
        led = run_mini(self.td, ["r1_ON44", "r1_ON24", "r1_OFF44"], self.CELLS,
                       handles={"r1_ON44": StartFails(dict(FAST, wall=sw), job_mode=True),
                                "r1_ON24": SettleFails(dict(FAST, wall=sw), job_mode=True)})
        self.assertEqual([(r["aborted"], r["abort_class"]) for r in led[:2]],
                         [("client_exception", "VALUE_DEPENDENT")] * 2)
        self.assertIn(True, stopped)                                     # stop() ran although start never did
        self.assertEqual((led[2]["status"], led[2].get("aborted"), led[2].get("cells_done")), ("ADMITTED", None, 1))
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            self.assertEqual(json.load(fh)["stopped"], None)             # not a runner crash
        for b in ("r1_ON44", "r1_ON24"):
            self.assertIn(["boot_abort", "client_exception"], [x[:2] for x in marks(self.td, b)])
            with open(os.path.join(self.td, f"boot_{b}.json")) as fh:
                self.assertIn("synthetic", json.load(fh)["error"])

    def test_stop_flag_honoured_before_warmup(self):                       # D5: health / warm-up loops
        sw = J.ScaledWall(200.0)
        box = {}

        class StopAtHealth(fk.FakeServerHandle):
            async def wait_healthy(self, session):
                ok = await super().wait_healthy(session)
                box["opts"]["stop"]["flag"] = True                         # watchdog SIGTERM during the /health wait
                return ok
        h = StopAtHealth(dict(FAST, wall=sw), job_mode=True)
        us = J.select_test_units(UNITS, {"r1_OFF44"}, set(self.CELLS))
        box["opts"] = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=2, install_signals=False, stop={},
                           cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        led = asyncio.run(J.run_job(self.td, us, lambda u: h, box["opts"]))
        self.assertEqual(led[0]["aborted"], "runner_stop")
        self.assertFalse([r for r in h.engine.arrivals if "warm" in r])  # no warm-up request after the stop
        self.assertNotIn("warmup_start", [x[0] for x in marks(self.td, "r1_OFF44")])
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            self.assertEqual(json.load(fh)["stopped"], "budget")

    def test_stop_flag_inside_warmup_loop(self):                           # D5: checked per warm-up request
        sw = J.ScaledWall(200.0)
        box = {}
        h = fk.FakeServerHandle(dict(FAST, wall=sw), job_mode=True)
        us = J.select_test_units(UNITS, {"r1_OFF44"}, set(self.CELLS))
        box["opts"] = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=3, install_signals=False, stop={},
                           cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        orig_simple = lg.simple_request
        calls = []

        async def counting(session, base, ids, mx, rid):
            calls.append(rid)
            r = await orig_simple(session, base, ids, mx, rid)
            box["opts"]["stop"]["flag"] = True
            return r
        lg.simple_request = counting
        try:
            led = asyncio.run(J.run_job(self.td, us, lambda u: h, box["opts"]))
        finally:
            lg.simple_request = orig_simple
        self.assertEqual(led[0]["aborted"], "runner_stop")
        self.assertEqual(calls, ["r1_OFF44-warm0"])                       # the 2nd / 3rd warm-up request never sent
        m = [x[0] for x in marks(self.td, "r1_OFF44")]
        self.assertEqual(m.count("warmup_start"), m.count("warmup_end"))   # the warm-up interval stays paired
        self.assertNotIn("cell_start", m)

    def test_runner_stop_in_last_boot_marks_budget(self):                 # runner_stop ends the cell section
        box = {}
        sw = J.ScaledWall(200.0)

        def make(u):
            h = fk.FakeServerHandle(dict(FAST, wall=sw, tau_s=0.004), job_mode=True)
            orig = h.wait_healthy

            async def wh(session):
                ok = await orig(session)
                asyncio.get_running_loop().call_later(0.6, box["opts"]["on_term"])
                return ok
            h.wait_healthy = wh
            return h
        us = J.select_test_units(UNITS, {"r1_OFF44"}, {(1, 512), (1, 2048)})
        box["opts"] = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=1, install_signals=False,
                           cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        led = asyncio.run(J.run_job(self.td, us, make, box["opts"]))
        self.assertEqual(led[0]["aborted"], "runner_stop")
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            self.assertEqual(json.load(fh)["stopped"], "budget")

    def test_cg_canary_cuda_graph_false_exit3(self):                       # §18-2-d #16 (H17-3)
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_OFF44", "r1_ON44"], self.CELLS,
                     handles={"r1_OFF44": dict(engine_kw=dict(cuda_graph=False))})
        self.assertEqual((cm.exception.code, cm.exception.tag), (3, "ABORT_CG_CANARY"))
        self.assertFalse(os.path.exists(os.path.join(self.td, "marks_r1_ON44.txt")))     # the Job stopped
        m = marks(self.td, "r1_OFF44")
        self.assertNotIn("cell_start", [x[0] for x in m])                    # aborted before any cell
        with open(os.path.join(self.td, "boot_r1_OFF44.json")) as fh:
            rec = json.load(fh)
        self.assertFalse(rec["cg_canary"]["ok"])
        self.assertGreaterEqual(rec["cg_canary"]["decode_lines"], 1)
        self.assertEqual(rec["cg_canary"]["not_cuda_graph_true"], rec["cg_canary"]["decode_lines"])
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            self.assertEqual(json.load(fh)["job_abort"]["tag"], "ABORT_CG_CANARY")

    def test_cg_canary_one_false_line_exit3(self):                          # "그 줄 전부" True
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_OFF44"], self.CELLS,
                     handles={"r1_OFF44": dict(engine_kw=dict(cg_false_at_iter=48))})   # warm-up tail only
        self.assertEqual(cm.exception.tag, "ABORT_CG_CANARY")
        with open(os.path.join(self.td, "boot_r1_OFF44.json")) as fh:
            rec = json.load(fh)["cg_canary"]
        self.assertGreater(rec["decode_lines"], rec["not_cuda_graph_true"])
        self.assertGreaterEqual(rec["not_cuda_graph_true"], 1)

    def test_cg_canary_no_decode_line_exit3(self):                          # ">= 1 line" (zero is not a pass)
        with self.assertRaises(J.JobAbort) as cm:
            run_mini(self.td, ["r1_OFF44"], self.CELLS,
                     handles={"r1_OFF44": dict(engine_kw=dict(decode_log_every=10 ** 9))})
        self.assertEqual(cm.exception.tag, "ABORT_CG_CANARY")
        with open(os.path.join(self.td, "boot_r1_OFF44.json")) as fh:
            self.assertEqual(json.load(fh)["cg_canary"]["decode_lines"], 0)

    def test_cg_canary_passes_cells_keep_cg_unproven(self):                # canary = cost guard, not a judgment
        led = run_mini(self.td, ["r1_OFF44"], self.CELLS,
                       handles={"r1_OFF44": dict(engine_kw=dict(cg_false_at_iter=80))})   # after the warm-up
        self.assertEqual((led[0]["status"], led[0]["aborted"]), ("ADMITTED", None))
        with open(os.path.join(self.td, "boot_r1_OFF44.json")) as fh:
            self.assertTrue(json.load(fh)["cg_canary"]["ok"])
        res = A.analyze(self.td, reg=REG)
        st = {r["cell_id"]: r["status"] for r in res["replicates"]}
        self.assertEqual(st["r1_OFF44-b1-c512"], "CG_UNPROVEN")

    def test_greedy_output_ids_absent_recorded_and_unresolved(self):       # §18-2-d #4 (H17-1): label, not abort
        led = run_mini(self.td, ["r1_ON44", "r1_OFF16"], self.CELLS,
                       engine_kw=dict(greedy_mode="no_ids"))
        self.assertEqual([r["status"] for r in led], ["ADMITTED", "ADMITTED"])
        with open(os.path.join(self.td, "boot_r1_ON44.json")) as fh:
            self.assertEqual(json.load(fh)["greedy_output_ids_absent"], 6)
        with open(os.path.join(self.td, "GREEDY_OUTPUT_IDS.txt")) as fh:
            txt = fh.read()
        self.assertIn("r1_ON44 outputs=6 output_ids_absent_or_empty=6", txt)
        res = A.analyze(self.td, reg=REG)
        self.assertEqual(res["sticky_correctness"]["44"]["label"], "STICKY_CORRECTNESS_UNRESOLVED_D44")
        self.assertEqual(res["sticky_correctness"]["44"]["why"], "output_ids absent")

    def test_greedy_with_ids_byte_identical(self):                          # the positive side of the same path
        run_mini(self.td, ["r1_ON44", "r1_OFF16"], self.CELLS)
        with open(os.path.join(self.td, "boot_r1_ON44.json")) as fh:
            self.assertEqual(json.load(fh)["greedy_output_ids_absent"], 0)
        res = A.analyze(self.td, reg=REG)
        self.assertEqual(res["sticky_correctness"]["44"]["label"], "STICKY_CORRECTNESS_BYTE_IDENTICAL_D44")

    def test_runner_stop_on_watchdog_term_mid_cell(self):                 # SIGTERM path -> runner_stop (H-5)
        box = {}
        sw = J.ScaledWall(200.0)

        def make(u):
            h = fk.FakeServerHandle(dict(FAST, wall=sw, tau_s=0.004), job_mode=True)
            orig = h.wait_healthy

            async def wh(session):
                ok = await orig(session)
                asyncio.get_running_loop().call_later(0.6, box["opts"]["on_term"])   # fires inside the first cell
                return ok
            h.wait_healthy = wh
            return h
        us = sorted(J.select_test_units(UNITS, {"r1_ON44", "r1_OFF44"}, {(1, 512), (1, 2048)}),
                    key=lambda u: u["name"] != "r1_ON44")
        box["opts"] = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=1, install_signals=False,
                           cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        led = asyncio.run(J.run_job(self.td, us, make, box["opts"]))
        self.assertEqual(led[0]["aborted"], "runner_stop")
        self.assertEqual(led[1]["status"], "NOT_ATTEMPTED_BUDGET")
        m = marks(self.td, "r1_ON44")
        self.assertIn(["cell_abort", "r1_ON44-b1-c512", "runner_stop"], [x[:3] for x in m])
        self.assertIn(["boot_abort", "runner_stop"], [x[:2] for x in m])

    def test_flush_wait_before_stop(self):                                  # H-5
        hs = {}

        sw = J.ScaledWall(200.0)                     # one wall clock for the fake's log stamps and the runner (CG canary)

        def keep(u):
            hs[u] = fk.FakeServerHandle(dict(FAST, wall=sw), job_mode=True)
            return hs[u]
        us = J.select_test_units(UNITS, {"r1_OBS44"}, {(1, 2048)})
        opts = dict(reg=REG, wall=sw, flush_interval_s=0.3, warmup_n=1, install_signals=False,
                    cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20))
        asyncio.run(J.run_job(self.td, us, lambda u: keep(u["name"]), opts))
        last_end = max(float(x[2]) for x in marks(self.td, "r1_OBS44") if x[0] == "cell_end")
        self.assertGreaterEqual(hs["r1_OBS44"].stop_called_at - last_end, 2 * 0.3)


class TestProcessServer(unittest.TestCase):                                 # D4 / D5 (work-review carry-over)
    def _group_alive(self, pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def test_stop_cleans_dead_leader_group_and_closes_log(self):          # D4: poll() is not a reason to skip
        with tempfile.TemporaryDirectory() as td:
            srv = J.ProcessServer("cfg", 1)
            srv.log = open(os.path.join(td, "srv.log"), "x")
            # leader exits at once, its child keeps the process group alive (an orphaned sglang worker)
            srv.proc = subprocess.Popen(["/usr/bin/bash", "-c", "sleep 300 & exit 0"], start_new_session=True)
            srv.proc.wait(timeout=10)
            pgid = srv.proc.pid
            self.assertIsNotNone(srv.proc.poll())
            self.assertTrue(self._group_alive(pgid))
            asyncio.run(srv.stop(term_grace_s=0.5))
            for _ in range(50):
                if not self._group_alive(pgid):
                    break
                time.sleep(0.1)
            self.assertFalse(self._group_alive(pgid), "the dead leader's group must be killed")
            self.assertTrue(srv.log.closed)

    def test_stop_without_start_is_safe(self):                              # D4 + D6: stop() after a failed start
        srv = J.ProcessServer("cfg", 1)
        asyncio.run(srv.stop(term_grace_s=0.1))
        self.assertEqual(srv.log_text(), "")

    def test_settle_is_async_and_honours_stop(self):                       # D5
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, "SETTLE.txt")
            stop = {}
            srv = J.ProcessServer("cfg", 1, stop_flag=stop)
            ticks = []

            async def go():
                async def ticker():
                    for _ in range(10):
                        ticks.append(time.monotonic())
                        await asyncio.sleep(0.02)
                    stop["flag"] = True
                t = asyncio.ensure_future(ticker())
                t0 = time.monotonic()
                with self.assertRaises(lg.RunnerStop):
                    await srv.settle(None, rec, max_s=60.0, poll_s=0.05)   # baseline None: never settles by itself
                await t
                return time.monotonic() - t0
            dt = asyncio.run(go())
            self.assertEqual(len(ticks), 10)                                 # the loop kept running (no time.sleep)
            self.assertLess(dt, 10.0)
            with open(rec) as fh:
                self.assertIn("end=stop_flag", fh.read())

    def test_wait_healthy_honours_stop(self):                               # D5: not held for 1200 s
        srv = J.ProcessServer("cfg", 1, stop_flag={"flag": True})
        srv.proc = subprocess.Popen(["/usr/bin/bash", "-c", "sleep 30"], start_new_session=True)
        try:
            with self.assertRaises(lg.RunnerStop):
                asyncio.run(srv.wait_healthy(None, poll_s=0.05))
        finally:
            asyncio.run(srv.stop(term_grace_s=1.0))


class TestCoShapeWarmup(unittest.TestCase):                                # G-3 (b) (Job-EA, implemented here)
    def test_one_then_two_request_spans_inside_warmup(self):
        async def go(td):
            srv = fk.FakeServerHandle(dict(FAST), job_mode=True)
            tp = os.path.join(td, "tel.jsonl")
            open(tp, "x").close()
            srv.start({"PDMUX_R2_FIXED_DSM": "44", "PDMUX_TELEMETRY_PATH": tp, "PDMUX_STICKY_PARTITION": "1",
                       "PDMUX_PHASE_EVENTS": "1"}, os.path.join(td, "srv.log"))
            await srv.wait_healthy(None)
            mw = n2_marks.MarkWriter(os.path.join(td, "m.txt"))
            s = lg.new_session()
            try:
                await lg.run_coshape_warmup(s, srv.base, mw, "bootX", 44, 1903, 4852)
            finally:
                await s.close()
                mw.close()
                await srv.stop()
            return mw.lines, srv.engine, tp
        with tempfile.TemporaryDirectory() as td:
            lines, eng, tp = asyncio.run(go(td))
            self.assertEqual([x.split()[0] for x in lines], ["warmup_start", "warmup_end"])
            self.assertEqual(len(eng.coload_arrivals), 3)
            with open(tp) as fh:
                ev = [json.loads(x) for x in fh if x.strip()]
            co = [e["n_new"] for e in ev if e["event"] == "prefill_span_start" and e["extend_tokens"] == e["n_new"] * 8192]
            self.assertEqual(sum(co), 3)                    # 1 + 2 sent; a 2-request span is not guaranteed (gap)
            self.assertEqual(co[0], 1)
            arr = sorted(eng.coload_arrivals)
            self.assertLess(arr[2] - arr[1], 0.03)          # the second group is sent concurrently (no pacing)
            self.assertGreater(arr[1] - arr[0], arr[2] - arr[1])   # ... and only after the first one completed
            warm = {}
            J.RED.parse_marks(lines, warm)
            t0, t1 = warm["bootX"]["warmup_start"], warm["bootX"]["warmup_end"]
            self.assertTrue(all(t0 <= e["timestamp_monotonic_s"] <= t1 for e in ev
                                if e["event"] == "prefill_span_start" and e["n_new"] * 8192 == e["extend_tokens"]))


# ===================================================================================================== Job-EA
UNITS_EA = P.units("EA", REG)


class TestEAPlan(unittest.TestCase):                                         # §5-2, §4-2, §3-1 co-load row
    def test_structure_and_registered_values(self):
        self.assertEqual(P.plan_problems("EA", UNITS_EA, REG), [])
        self.assertEqual(J.ea_payload_problems(UNITS_EA), [])
        self.assertEqual([u["name"] for u in UNITS_EA[:4]], ["r1_ON44", "r1_OFF44", "r1_ON16", "r1_OFF16"])
        for u in UNITS_EA:
            self.assertEqual((u["sticky"], u["phase_events"], u["greedy"]), (u["kind"] == "ON", True, False))
            self.assertEqual(u["fixed_dsm"], u["D"])
            self.assertEqual([(c["b"], c["c"]) for c in u["cells"]], list(P.EA_CELLS))
            for c in u["cells"]:
                e = REG["derived"]["ealpha"][f"{u['D']},{c['b']},{c['c']}"]
                self.assertEqual((c["K"], c["delta"], c["c_in"], c["O"]),
                                 (REG["K"][str(u["D"])], REG["DELTA_FIXED"], e["c_in"], math.ceil(e["O_EA"])))
                self.assertEqual(c["seed_kind"], "EALPHA")
        on = {(u["round"], c["b"], c["c"]): c["c_in"] for u in UNITS_EA if u["kind"] == "ON" and u["D"] == 16
              for c in u["cells"]}
        off = {(u["round"], c["b"], c["c"]): c["c_in"] for u in UNITS_EA if u["kind"] == "OFF" and u["D"] == 16
               for c in u["cells"]}
        self.assertEqual(on, off)                                            # R5-2: the pair shares c_in(d)

    def test_structure_violations_detected(self):
        bad = [dict(u) for u in UNITS_EA]
        bad[0], bad[1] = bad[1], bad[0]                                       # r1 order OFF44, ON44
        self.assertTrue(any("EA round 1 order" in x for x in P.plan_problems("EA", bad, REG)))
        k = [dict(u, cells=[dict(c, K=c["K"] - 1) for c in u["cells"]]) if i == 0 else u
             for i, u in enumerate(UNITS_EA)]
        self.assertTrue(any("K 14 != registered K_44 15" in x for x in P.plan_problems("EA", k, REG)))
        dl = [dict(u, cells=[dict(c, delta=8) for c in u["cells"]]) if i == 0 else u for i, u in enumerate(UNITS_EA)]
        self.assertTrue(any("delta 8" in x for x in P.plan_problems("EA", dl, REG)))
        cin = [dict(u, cells=[dict(c, c_in=c["c_in"] + 1) for c in u["cells"]]) if i == 1 else u
               for i, u in enumerate(UNITS_EA)]                               # OFF with its own c_in (rev5 defect)
        self.assertTrue(any("c_in/O" in x for x in P.plan_problems("EA", cin, REG)))
        order = [dict(u, cells=list(reversed(u["cells"]))) if i == 0 else u for i, u in enumerate(UNITS_EA)]
        self.assertTrue(any("E-alpha cell order" in x for x in P.plan_problems("EA", order, REG)))

    def test_payload_trace_check(self):                                     # F1/F2: max_new_tokens == 1 explicit
        orig = lg.payload

        def zero(ids, mx, rid, coload=False):
            out = orig(ids, mx, rid, coload)
            if coload:
                out["sampling_params"]["max_new_tokens"] = 0
            return out
        lg.payload = zero
        try:
            self.assertTrue(J.ea_payload_problems(UNITS_EA[:1]))
        finally:
            lg.payload = orig
        with self.assertRaises(lg.TraceInvalid):
            lg.payload([1, 2], None, "x", coload=True)

    def test_warmup_inputs(self):                                            # G-3 (a)/(b) for Job-EA boots
        for u in UNITS_EA[:4]:
            c = J.warmup_c_in(u, REG)
            self.assertEqual(c, REG["derived"]["ealpha"][f"{u['D']},32,8192"]["c_in"])
            self.assertNotEqual(c % 8192, 0)
            cw = J.coshape_warmup_args(u, REG)
            e = REG["derived"]["ealpha"][f"{u['D']},8,2048"]
            self.assertEqual(cw, (e["c_in"], math.ceil(e["O_EA"])))

    def test_dryrun_cli(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_") or k == "PDMUX_PROJECT_ROOT"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            r = subprocess.run([sys.executable, os.path.join(HERE, "n2_job.py"), "--job", "EA", "--out", td,
                                "--dry-run"], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("DRYRUN 16 boots, 64 cells", r.stdout)
            with open(os.path.join(td, "DRYRUN_COMMANDS.txt")) as fh:
                txt = fh.read()
            self.assertEqual(txt.count("co_max_new_tokens=1"), 64)
            self.assertEqual(txt.count("K=33 "), 32)
            self.assertEqual(txt.count("K=15 "), 32)
            r = subprocess.run([sys.executable, os.path.join(HERE, "n2_job.py"), "--job", "XX", "--out", td,
                                "--dry-run"], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 2)


EA_FAST = dict(FAST, co_span_s=0.04)


def run_mini_ea(td, names, cells, engine_kw=None, handles=None, **opt):
    sw = J.ScaledWall(200.0)
    us = sorted(J.select_test_units(UNITS_EA, set(names), set(cells)), key=lambda u: names.index(u["name"]))
    kw = dict(EA_FAST, wall=sw, **(engine_kw or {}))

    def make(u):
        h = (handles or {}).get(u["name"])
        if isinstance(h, dict):
            return fk.FakeServerHandle(dict(kw, **h.get("engine_kw", {})), job_mode=True)
        return h or fk.FakeServerHandle(dict(kw), job_mode=True)
    opts = dict(reg=REG, wall=sw, flush_interval_s=0.02, warmup_n=1, install_signals=False,
                cell_kw=dict(drain_poll_s=0.02, drain_settle_s=0.05, drain_timeout_s=20), **opt)
    return asyncio.run(J.run_job(td, us, make, opts))


class TestRunnerEA(unittest.TestCase):                                       # Job-EA end to end on the CPU fake
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="n2job_ea_")
        cls.names = ["r1_ON44", "r1_OFF44"]
        cls.led = run_mini_ea(cls.td, cls.names, [(8, 2048), (32, 2048)])
        cls.res = A.analyze_ea(cls.td, reg=REG)
        cls.reps = {r["cell_id"]: r for r in cls.res["replicates"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def test_ledger_budget_column(self):
        self.assertEqual([r["status"] for r in self.led], ["ADMITTED"] * 2)
        with open(os.path.join(self.td, "JOB_LEDGER.json")) as fh:
            jl = json.load(fh)
        self.assertEqual((jl["job"], jl["cap_s"]), ("EA", float(REG["budget"]["EA_b"]["cap"])))
        self.assertEqual(self.led[0]["need_s"], REG["budget"]["EA_b"]["per_boot_worst"]["ON44"])

    def test_one_warmup_interval_holds_both_warmups(self):                 # G-3 (a) + (b), §7-2 (R)
        m = marks(self.td, "r1_OFF44")
        tags = [x[0] for x in m]
        self.assertEqual(tags[:3], ["boot_plan", "warmup_start", "warmup_end"])
        self.assertEqual(tags.count("warmup_start"), 1)
        w0, w1 = float(m[1][2]), float(m[2][2])
        ev = tel(self.td, "r1_OFF44")
        co_w = [e for e in ev if e["event"] == "prefill_span_start" and e["extend_tokens"] == e["n_new"] * 8192
                and w0 <= e["timestamp_monotonic_s"] <= w1]
        self.assertEqual(sum(e["n_new"] for e in co_w), 3)                  # 1 + 2 co-shape requests
        cells, _calls, _plan, _ab, issues = J.RED.parse_marks([" ".join(x) for x in m], {})
        self.assertEqual(issues, [])
        self.assertTrue(all(cm["cell_start"][1] > w1 for cm in cells.values()))

    def test_cell_marks_and_client_record(self):                            # §4-2 steps 4-5 (E-alpha rule)
        m = marks(self.td, "r1_ON44")
        cells, *_ = J.RED.parse_marks([" ".join(x) for x in m], {})
        self.assertEqual(sorted(cm["cell_start"][0] for cm in cells.values()), [(44, 8, 2048), (44, 32, 2048)])
        self.assertTrue(all("coload_sent" in cm and "fix_s" in cm for cm in cells.values()))
        with open(os.path.join(self.td, "client_r1_ON44.jsonl")) as fh:
            recs = {r["cell_id"]: r for r in map(json.loads, fh)}
        r = recs["r1_ON44-ea-b8-c2048"]
        self.assertEqual((len(r["requests"]), len(r["co_requests"]), r["K"]), (8, 15, 15))
        self.assertEqual({q["completion_tokens"] for q in r["co_requests"]}, {1})
        self.assertTrue(r["cancel_fired"])
        self.assertGreaterEqual(r["median_at_cancel"], r["base"] + 16 + 128 + 64)
        self.assertLessEqual(r["t_coload"], min(q["times"][0] for q in r["co_requests"]))
        cm = cells["r1_ON44-ea-b8-c2048"]
        ev = tel(self.td, "r1_ON44")
        lo = [e for e in ev if e["event"] == "prefill_span_start" and e["timestamp_monotonic_s"] >= cm["coload_sent"]
              and e["timestamp_monotonic_s"] <= cm["cell_end"]]
        self.assertEqual(sum(e["n_new"] for e in lo), 15)                    # K co requests, nothing else

    def test_trigger_after_a1_window(self):                                 # §4-2 step 4: trigger c + 71
        with open(os.path.join(self.td, "client_r1_OFF44.jsonl")) as fh:
            r = {x["cell_id"]: x for x in map(json.loads, fh)}["r1_OFF44-ea-b8-c2048"]
        before = [sum(1 for t in q["times"] if t < r["t_coload"]) for q in r["requests"]]
        self.assertGreaterEqual(r["c_in"] + lg.lower_median(before), 2048 + 71)

    def test_windows_and_rho_end_to_end(self):                              # runner -> n2_window -> n2_analyze
        for cid in ("r1_ON44-ea-b8-c2048", "r1_OFF44-ea-b8-c2048", "r1_ON44-ea-b32-c2048", "r1_OFF44-ea-b32-c2048"):
            self.assertEqual(self.reps[cid]["windows"], {"A1": "VALID", "A2": "VALID", "C": "VALID"}, cid)
        off = self.reps["r1_OFF44-ea-b8-c2048"]
        self.assertGreaterEqual(off["values"]["A1"]["pe_iter_idx5"], 0.99)  # OFF alone on (0,108)
        self.assertEqual(off["values"]["C"]["pe_iter"], 1.0)                # OFF C on (64,44)
        cr = self.res["cells"]["44,8,2048"]
        for which in ("rho_off", "rho_on"):
            p = cr[which]["pairs"]["1"]
            self.assertTrue(p["valid"], p)
            self.assertAlmostEqual(p["ln_rho"], math.log(p["c_med"] / p["a_on"]))
            self.assertEqual(p["on_boot"], "r1_ON44")
        self.assertEqual(cr["rho_off"]["pairs"]["1"]["c_boot"], "r1_OFF44")
        self.assertEqual(cr["rho_on"]["pairs"]["1"]["c_boot"], "r1_ON44")      # rho_on: same boot (boot effect out)
        self.assertEqual(cr["rho_off"]["label"], "UNRESOLVED_N_BELOW_4")
        self.assertEqual(self.res["coload"]["label"], "COLOAD_NOT_JOINED")
        self.assertEqual(self.res["neg_control_ea"]["label"], "NEG_CONTROL_EA_HELD")
        self.assertEqual(self.res["neg_control_ea"]["n"], 4)
        self.assertEqual(self.res["ds3"], "DS3_UNRESOLVED")

    def test_cli_writes_ea_outputs_and_seal(self):                          # --job EA, --seal-ea-result
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_") or k == "PDMUX_PROJECT_ROOT"}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "ea")
            shutil.copytree(self.td, out)
            r = subprocess.run([sys.executable, os.path.join(HERE, "n2_analyze.py"), "--job", "EA", "--out-dir", out],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            for f in A.EA_RESULT_FILES:
                self.assertTrue(os.path.exists(os.path.join(out, f)))
            r = subprocess.run([sys.executable, os.path.join(HERE, "n2_analyze.py"), "--job", "EA", "--out-dir", out,
                                "--seal-ea-result"], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            seal = A.verify_ea_result_seal(os.path.join(out, A.EA_SEAL_FILE))
            self.assertTrue(seal["verified"], seal["problems"])
            with open(os.path.join(out, "N2_EA_LABELS.txt"), "a") as fh:  # tampering after the seal
                fh.write("x\n")
            self.assertFalse(A.verify_ea_result_seal(os.path.join(out, A.EA_SEAL_FILE))["verified"])


class TestEAAborts(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="n2job_eaab_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_coload_payload_invalid_ends_boot(self):                        # F3 -> H-4 (4) -> G-10 Job
        led = run_mini_ea(self.td, ["r1_ON44", "r1_OFF44"], [(8, 2048), (32, 2048)],
                          handles={"r1_ON44": dict(engine_kw=dict(mode=("bad_coload",)))})
        self.assertEqual((led[0]["aborted"], led[0]["abort_class"], led[0]["cells_done"]),
                         ("coload_payload_invalid", "VALUE_DEPENDENT", 0))
        self.assertEqual((led[1]["status"], led[1].get("aborted"), led[1].get("cells_done")), ("ADMITTED", None, 2))
        res = A.analyze_ea(self.td, reg=REG)
        st = {r["cell_id"]: (r["status"], r["reason"]) for r in res["replicates"]}
        self.assertEqual(st["r1_ON44-ea-b8-c2048"][0], "COLOAD_PAYLOAD_INVALID")
        self.assertEqual(st["r1_ON44-ea-b32-c2048"], ("BENCH_INVALID", "not_attempted_boot_aborted"))
        self.assertEqual(st["r1_OFF44-ea-b8-c2048"][0], "MEASURED")
        self.assertEqual(res["cells"]["44,8,2048"]["rho_off"]["pairs"]["1"]["reason"].split(" (")[0], "A_on invalid")


class TestPostEASeal(unittest.TestCase):                                     # §7-3 산출 시점 / §18-2-d #11
    def test_profile_cell_boot_scope_only_after_a_verified_ea_seal(self):
        with tempfile.TemporaryDirectory() as td:
            prof, ea = os.path.join(td, "prof"), os.path.join(td, "ea")
            os.makedirs(prof)
            os.makedirs(ea)
            run_mini(prof, ["r1_OFF44"], [(1, 512)])
            seal_path = os.path.join(ea, A.EA_SEAL_FILE)
            with self.assertRaisesRegex(MX.BlindedScope, "not verified"):     # no seal yet
                A.profile_post_ea(prof, seal_path, reg=REG)
            for f in A.EA_RESULT_FILES:
                with open(os.path.join(ea, f), "w") as fh:
                    fh.write("{}\n")
            A.seal_ea_result(ea)
            res = A.profile_post_ea(prof, seal_path, reg=REG)
            self.assertEqual(len(res["cells"]), 1)
            est = res["cells"][0]["estimators"]
            self.assertEqual(sorted(est), ["0,108", "64,44"])                 # OFF boot: alone + FIXED pair
            self.assertTrue(all("post_ea_result_seal" in v and "blind_exception" not in v for v in est.values()))
            self.assertEqual(len(res["boots"]), 1)
            with open(os.path.join(ea, "N2_EA_RESULT.json"), "a") as fh:      # result changed after the seal
                fh.write(" ")
            with self.assertRaisesRegex(MX.BlindedScope, "not verified"):
                A.profile_post_ea(prof, seal_path, reg=REG)
            with self.assertRaises(FileExistsError):                          # the seal is written once
                A.seal_ea_result(ea)

    def test_estimators_refuse_unverified_seal(self):
        ev = [dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=1.0, decode_running_batch_size=1,
                   prefill_sms=64, decode_sms=44, decode_iterations=1)]
        with self.assertRaises(MX.BlindedScope):
            MX.estimators(ev, (64, 44), "cell", job="PROFILE", ea_result_seal=dict(verified=False,
                                                                                     ea_result_sha256="x"))
        out = MX.estimators(ev, (64, 44), "cell", job="PROFILE", ea_result_seal=dict(verified=True,
                                                                                      ea_result_sha256="x"))
        self.assertIn("post_ea_result_seal", out)
        self.assertNotIn("blind_exception", out)
        out = MX.estimators(ev, (64, 44), "cell", job="EA")
        self.assertNotIn("blind_exception", out)


# ===================================================================================================== mutant runner
def _runner():
    spec = importlib.util.spec_from_file_location("_n2_mutants_rt", os.path.join(HERE, "n2_mutants.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOY = '''def add(a, b):
    return a + b


def twice(x):
    return add(x, x)


class Box:
    def stop(self):
        return 1


def other():
    return 0
'''
TOY_TEST = '''import unittest
import toy


class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(toy.add(1, 2), 3)


class TestTwice(unittest.TestCase):
    def test_twice(self):
        self.assertEqual(toy.twice(2), 4)


class TestOther(unittest.TestCase):
    def test_other(self):
        self.assertEqual(toy.other(), 0)

    def test_other2(self):
        self.assertEqual(toy.other(), 0)

    def test_other3(self):
        self.assertEqual(toy.other(), 0)
'''


class TestMutantRunner(unittest.TestCase):                                  # rev18 candidate (user approval 2026-10-12)
    """The shared targeted / parallel runner (n2_mutants.py, used by n2_job_mutants.py) on a toy package -- no
    registered mutant is executed here."""

    @classmethod
    def setUpClass(cls):
        cls.R = _runner()

    def test_selection_follows_callers_only(self):
        sel = self.R.targeted_tests({"toy.py": TOY}, {"test_toy.py": TOY_TEST}, [("toy.py", "return a + b", "x")])
        self.assertEqual(sel, ["test_toy.TestAdd.test_add", "test_toy.TestTwice.test_twice"])
        sel = self.R.targeted_tests({"toy.py": TOY}, {"test_toy.py": TOY_TEST}, [("toy.py", "return add(x, x)", "x")])
        self.assertEqual(sel, ["test_toy.TestTwice.test_twice"])

    def test_large_selection_runs_full(self):                              # >= 90 % of the tests: no targeted phase
        big = TOY_TEST.replace("toy.other()", "toy.add(0, 0) * 0 + toy.other()")
        self.assertEqual(self.R.targeted_tests({"toy.py": TOY}, {"test_toy.py": big},
                                               [("toy.py", "return a + b", "x")]), [])

    def _judge(self, mutated, expect="kill"):
        with tempfile.TemporaryDirectory() as td:
            ctrl, mut = os.path.join(td, "ctrl"), os.path.join(td, "mut")
            for d, text in ((ctrl, TOY), (mut, mutated)):
                os.makedirs(d)
                with open(os.path.join(d, "toy.py"), "w") as fh:
                    fh.write(text)
                with open(os.path.join(d, "test_toy.py"), "w") as fh:
                    fh.write(TOY_TEST)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            controls = self.R.TargetControls(lambda: (ctrl, lambda: None), env)
            calls = []

            def full():
                calls.append("full")
                return self.R.run_unittest(mut, env, ["test_toy"])

            def targets(nm):
                calls.append("targeted")
                return self.R.run_unittest(mut, env, nm, failfast=True)
            sel = self.R.targeted_tests({"toy.py": TOY}, {"test_toy.py": TOY_TEST}, [("toy.py", "return add(x, x)", "")])
            return self.R.judge("M1", "toy", expect, full, sel, targets, controls), calls

    def test_targeted_kill_needs_no_full_run(self):
        row, calls = self._judge(TOY.replace("return add(x, x)", "return add(x, 1)"))
        self.assertEqual((row[2], calls), ("KILLED", ["targeted"]))
        self.assertIn("[targeted 1, failfast]", row[3])

    def test_targeted_survivor_runs_the_full_suite(self):                  # ledger meaning: ANY test failing kills
        row, calls = self._judge(TOY.replace("return add(x, x)", "return add(x, x)\n\n\nX = 1")
                                 .replace("return 0", "return 9"))          # only an untargeted test fails
        self.assertEqual((row[2], calls), ("KILLED", ["targeted", "full"]))
        self.assertIn("survived -> full", row[3])
        row, calls = self._judge(TOY)                                      # nothing fails anywhere
        self.assertEqual((row[2], calls), ("SURVIVED", ["targeted", "full"]))

    def test_targeted_failure_with_failing_control_is_not_a_kill(self):    # isolation artefact never kills
        R = self.R

        class Never:
            def passes(self, names):
                return False
        env = dict(os.environ)
        row = R.judge("M2", "toy", "kill", lambda: (False, "OK", False), ["x"], lambda nm: (True, "FAILED", False),
                      Never())
        self.assertEqual(row[2], "SURVIVED")
        self.assertIn("targeted control failed -> full", row[3])
        row = R.judge("M3", "toy", "equivalent", lambda: (False, "OK", False), ["x"], lambda nm: (True, "FAILED", False),
                      Never())
        self.assertEqual(row[2], "EQUIVALENT-SURVIVED")                     # equivalents always run the full suites
        del env

    def test_run_rows_order_and_parallel(self):
        import threading
        seen = []
        lock = threading.Lock()

        def mk(i):
            def go():
                time.sleep(0.05 * (3 - i))
                return (f"M{i}", "", "KILLED", "")
            return go
        rows = self.R.run_rows([(i, mk(i)) for i in range(3)], 3, lambda r: (lock.acquire(), seen.append(r[0]),
                                                                              lock.release()))
        self.assertEqual([r[0] for r in rows], ["M0", "M1", "M2"])          # table order kept
        self.assertEqual(sorted(seen), ["M0", "M1", "M2"])
        self.assertEqual(self.R.parse_runner_opts(["--jobs", "4", "--set", "core", "--no-targeted", "x"]),
                         (4, False, ["--set", "core", "x"]))
        with self.assertRaises(SystemExit):
            self.R.parse_runner_opts(["--jobs", "0"])


# ===================================================================================================== n2_capacity.sh
def job_layout(td):
    """Lesson 270: the VESSL Job layout -- workspace/engine-port/results is a SYMLINK to the run volume."""
    proj = os.path.join(td, "proj")
    eng = os.path.join(proj, "workspace", "engine-port")
    run = os.path.join(td, "run", "results")
    os.makedirs(os.path.join(run, "n3_capacity"))
    os.makedirs(eng)
    shutil.copytree(HERE, os.path.join(run, "n2_static_grid"),
                    ignore=shutil.ignore_patterns("__pycache__", "n2p_*", "*.pyc"))
    shutil.copytree(os.path.join(HERE, "..", "n3_capacity", "configs"), os.path.join(run, "n3_capacity", "configs"))
    shutil.copy(os.path.join(HERE, "..", "n3_capacity", "n3_capacity.sh"), os.path.join(run, "n3_capacity"))
    os.symlink(run, os.path.join(eng, "results"))
    for d in ("benchmarks", "src", "scripts"):
        os.symlink(os.path.join(P.ENG, d), os.path.join(eng, d))
    return proj, os.path.join(run, "n2_static_grid")


def sh(proj, env_extra, td):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env.update(PDMUX_PROJECT_ROOT=proj, PDMUX_DATA=os.path.join(td, "data"), PDMUX_JOB_NAME="t1",
               PDMUX_N2_TEST_SKIP_PREFLIGHT="1", PYTHONDONTWRITEBYTECODE="1", **env_extra)
    script = os.path.join(proj, "workspace", "engine-port", "results", "n2_static_grid", "n2_capacity.sh")
    return subprocess.run(["/usr/bin/bash", script], env=env, capture_output=True, text=True, timeout=600)


class TestCapacityScript(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="n2cap_")
        self.proj, self.here = job_layout(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_dryrun_in_symlinked_job_layout(self):
        r = sh(self.proj, dict(PDMUX_N2_JOB="PROFILE", PDMUX_N2_DRYRUN="1"), self.td)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = os.path.join(self.here, "n2p_t1")
        with open(os.path.join(out, "DRYRUN_COMMANDS.txt")) as fh:
            txt = fh.read()
        self.assertEqual(txt.count("BOOT "), 28)
        self.assertEqual(txt.count("  cell "), 296)
        with open(os.path.join(out, "REG_CHECK.txt")) as fh:
            self.assertIn("REG_OK", fh.read())
        with open(os.path.join(out, "HARNESS_SEAL_CHECK.txt")) as fh:
            self.assertIn("scope=PROFILE", fh.read())

    def test_tampered_registered_file_aborts(self):                         # ABORT_REG_SOURCE (runner side)
        p = os.path.join(self.here, "N2_REGISTERED.json")
        with open(p) as fh:
            reg = json.load(fh)
        reg["K"]["16"] = 18
        with open(p, "w") as fh:
            json.dump(reg, fh, indent=1, sort_keys=True)
        r = sh(self.proj, dict(PDMUX_N2_JOB="PROFILE", PDMUX_N2_DRYRUN="1"), self.td)
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("ABORT_REG_SOURCE (see REG_CHECK.txt)", r.stdout)     # refused AT the registration step
        self.assertFalse(os.path.exists(os.path.join(self.here, "n2p_t1", "UNITS.txt")))

    def test_knob_rules(self):
        r = sh(self.proj, dict(PDMUX_N2_JOB="PROFILE", PDMUX_N2_DRYRUN="1", PDMUX_LA_COORD="1"), self.td)
        self.assertEqual(r.returncode, 2)
        self.assertIn("ABORT_ENV", r.stdout)
        r = sh(self.proj, dict(PDMUX_N2_JOB="PROFILE", PDMUX_N2_DRYRUN="1", PDMUX_COMMIT="abc"), self.td)
        self.assertEqual(r.returncode, 2)
        self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", r.stdout)
        self._write_sections("")            # the layout copy may carry real seal sections (H17-9 class): drop them
        r = sh(self.proj, dict(PDMUX_N2_JOB="EA", PDMUX_N2_DRYRUN="1"), self.td)        # H17-6: every mode
        self.assertEqual(r.returncode, 3)
        self.assertIn("ABORT_HARNESS_UNSEALED_EA", r.stdout)
        r = sh(self.proj, dict(PDMUX_N2_JOB="XX"), self.td)
        self.assertEqual(r.returncode, 2)

    def test_preflight_cwd_import_in_symlinked_job_layout(self):             # H17-10 (Job 52a3b97 ABORT_PREFLIGHT)
        cwd = os.path.join(self.proj, "workspace", "engine-port", "results", "n2_static_grid")   # logical (symlink)
        code = "import n2_analyze, n2_realized_mix as M; print(n2_analyze.ENG); print(M.ENG)"
        base = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        base.update(PYTHONDONTWRITEBYTECODE="1")
        bad = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=base, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0, "layout must reproduce the Job failure without PDMUX_PROJECT_ROOT")
        self.assertIn("analyze.py", bad.stderr)
        ok = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=dict(base, PDMUX_PROJECT_ROOT=self.proj),
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        want = os.path.join(os.path.abspath(self.proj), "workspace", "engine-port")
        self.assertEqual(ok.stdout.split(), [want, want])

    def test_preflight_n3_dir_without_jsonl(self):                           # H17-10: Job clone has no N3 JSONL
        n3b = os.path.join(self.proj, "workspace", "engine-port", "results", "n3_capacity",
                           "n3b_pdmux-n3-capacity-38ed3f4-20261006092813")
        os.makedirs(n3b)
        with open(os.path.join(n3b, "RESULT.md"), "w") as fh:                # tracked non-JSONL file only
            fh.write("x\n")
        cwd = os.path.join(self.proj, "workspace", "engine-port", "results", "n2_static_grid")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_") and k != "N2_N3_DIR"}
        env.update(PYTHONDONTWRITEBYTECODE="1", PDMUX_PROJECT_ROOT=self.proj)
        t = "test_n2_probe_reduce.TestTheta.test_positive_control_n3"
        bad = subprocess.run([sys.executable, "-m", "unittest", t], cwd=cwd, env=env, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0, "an empty tracked N3 dir must fail the positive control (the hazard)")
        with open(os.path.join(HERE, "n2_capacity.sh")) as fh:
            m = re.search(r'cd "\$HERE" && N2_N3_DIR="(\$OUT/[A-Z0-9_]+)" timeout', fh.read())
        self.assertIsNotNone(m, "preflight must override N2_N3_DIR")
        ok = subprocess.run([sys.executable, "-m", "unittest", "-v", t], cwd=cwd, capture_output=True, text=True,
                            env=dict(env, N2_N3_DIR=m.group(1).replace("$OUT", self.td)))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("skipped", ok.stderr)

    def _write_sections(self, txt):
        reg = os.path.join(self.here, "REGISTRATION_SHA256.txt")
        with open(reg) as fh:                                               # H17-9: drop any existing harness section
            base = "".join(b for b in re.split(r"(?m)^(?=## )", fh.read()) if not b.startswith("## HARNESS SEAL"))
        with open(reg, "w") as fh:
            fh.write(base + "\n" + txt)

    def test_ea_needs_profile_and_ea_sections(self):                        # rev17 R17-2 / H17-6 (every mode)
        prof, ea = seal_section(self.here, "PROFILE"), seal_section(self.here, "EA")
        for txt, fake in ((prof, False), (ea, False), (prof, True), ("", True)):
            self._write_sections(txt)
            shutil.rmtree(os.path.join(self.here, "n2p_t1"), ignore_errors=True)
            mode = dict(PDMUX_N2_TEST_FAKE="1") if fake else dict(PDMUX_N2_DRYRUN="1")
            r = sh(self.proj, dict(PDMUX_N2_JOB="EA", **mode), self.td)
            self.assertEqual(r.returncode, 3, (txt[:40], r.stdout))
            self.assertIn("ABORT_HARNESS_UNSEALED_EA", r.stdout)
            out = os.path.join(self.here, "n2p_t1")
            with open(os.path.join(out, "REG_CHECK.txt")) as fh:            # the registration check ran first
                self.assertIn("REG_OK", fh.read())
            self.assertFalse(os.path.exists(os.path.join(out, "UNITS.txt")))    # refused AT the seal check
        self._write_sections(prof + "\n" + ea)
        shutil.rmtree(os.path.join(self.here, "n2p_t1"), ignore_errors=True)
        r = sh(self.proj, dict(PDMUX_N2_JOB="EA", PDMUX_N2_DRYRUN="1"), self.td)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = os.path.join(self.here, "n2p_t1")
        with open(os.path.join(out, "HARNESS_SEAL_CHECK.txt")) as fh:
            self.assertIn("SEALED scope=EA", fh.read())
        with open(os.path.join(out, "DRYRUN_COMMANDS.txt")) as fh:
            txt = fh.read()
        self.assertEqual((txt.count("BOOT "), txt.count("  cell ")), (16, 64))
        with open(os.path.join(out, "BUDGET.txt")) as fh:
            self.assertIn(f"column=EA_b cap_s={REG['budget']['EA_b']['cap']}", fh.read())
        self.assertIn("N2_DONE_EA_t1", r.stdout)

    def test_fake_ea_job_end_to_end(self):                                  # the whole script, Job-EA, CPU fake
        self._write_sections(seal_section(self.here, "PROFILE") + "\n" + seal_section(self.here, "EA"))
        r = sh(self.proj, dict(PDMUX_N2_JOB="EA", PDMUX_N2_TEST_FAKE="1", PDMUX_N2_TEST_UNITS="r1_ON44,r1_OFF44",
                               PDMUX_N2_TEST_CELLS="8,2048", PDMUX_N2_TEST_WALL_SCALE="200"), self.td)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        out = os.path.join(self.here, "n2p_t1")
        with open(os.path.join(out, "N2_EA_RESULT.json")) as fh:
            res = json.load(fh)
        st = {x["cell_id"]: x["windows"] for x in res["replicates"]}
        for cid in ("r1_ON44-ea-b8-c2048", "r1_OFF44-ea-b8-c2048"):
            self.assertEqual(st[cid], {"A1": "VALID", "A2": "VALID", "C": "VALID"}, (cid, st[cid]))
        self.assertTrue(res["cells"]["44,8,2048"]["rho_off"]["pairs"]["1"]["valid"])
        self.assertTrue(os.path.exists(os.path.join(out, "N2_EA_LABELS.txt")))
        self.assertIn("N2_DONE_EA_t1", r.stdout)

    def test_fake_job_end_to_end(self):                                     # the whole script on the CPU fake
        r = sh(self.proj, dict(PDMUX_N2_JOB="PROFILE", PDMUX_N2_TEST_FAKE="1", PDMUX_N2_TEST_UNITS="r1_ON44,r1_OFF44",
                               PDMUX_N2_TEST_CELLS="1,512;8,2048", PDMUX_N2_TEST_WALL_SCALE="200"), self.td)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        out = os.path.join(self.here, "n2p_t1")
        with open(os.path.join(out, "N2_PROFILE_RESULT.json")) as fh:
            res = json.load(fh)
        st = {x["cell_id"]: x["status"] for x in res["replicates"]}
        self.assertEqual([st[k] for k in ("r1_ON44-b1-c512", "r1_ON44-b8-c2048", "r1_OFF44-b1-c512",
                                          "r1_OFF44-b8-c2048")], ["VALID"] * 4, st)
        for f in ("N2_PROFILE_POINTS.csv", "N2_PROFILE_LABELS.txt", "JOB_LEDGER.json", "substrate_n2.json",
                  "REGISTRATION_RUN_SHA256.txt", "TEST_FAKE_ENGINE"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertIn("N2_DONE_PROFILE_t1", r.stdout)


if __name__ == "__main__":
    unittest.main()
