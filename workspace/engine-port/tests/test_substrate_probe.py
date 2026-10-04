"""CPU tests for scripts/vessl/substrate_probe_helper.py (the B0/B3/B4 judgments).

The GPU legs cannot run here; what can be pinned on a CPU node is
  * that every judgment DISCRIMINATES -- each rule is driven into PASS and into
    FAIL/UNRESOLVED by inputs that differ in exactly the fact it is meant to read
    (a judge that returned a constant would fail at least one assertion here);
  * that the registered inputs are still the ones the cited sources hold
    (grid = config files, B4 prompts/argv = e2_sticky.sbatch), so a later edit of
    either side shows up as a test failure instead of silent drift;
  * that the KISTI verdict artefacts map to PASS under the same mapping the VESSL
    run will use (R0 889631, P0-A 890893).
"""

import ast
import importlib.util
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path) as f:
        return f.read()


ENGINE = os.path.dirname(HERE)
HELPER = os.path.join(ENGINE, "scripts", "vessl", "substrate_probe_helper.py")
SHELL = os.path.join(ENGINE, "scripts", "vessl", "substrate_probe.sh")
E2 = os.path.join(ENGINE, "results", "r2_eval", "e2_sticky_prereg", "e2_sticky.sbatch")

spec = importlib.util.spec_from_file_location("substrate_probe_helper", HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def census(labels, saturated=True, min_hits=5):
    last = sorted(labels)
    prev = last if saturated else last[:-1]
    return {"ladder": [{"n_blocks": 864, "union": prev}, {"n_blocks": 1728, "union": last}],
            "union": last, "min_hits": min_hits}


def readout(n):
    return {"green_ctx_is_null": False, "cuStreamGetGreenCtx_rc": 0,
            "green_sm": {"smCount": n, "minSmPartitionSize": 2, "smCoscheduledAlignment": 2}}


def conc_block(p_labels, d_labels, overlap=True):
    pt = (0, 1000) if overlap else (0, 100)
    dt = (500, 1500) if overlap else (200, 300)
    return {"n_blocks": len(p_labels),
            "prefill": {"smid": list(p_labels), "t0": [pt[0]] * len(p_labels),
                        "t1": [pt[1]] * len(p_labels)},
            "decode": {"smid": list(d_labels), "t0": [dt[0]] * len(d_labels),
                       "t1": [dt[1]] * len(d_labels)}}


D = set(range(108))


def pair_rec(p, d, sp=None, sd=None, conc=None, created=True, sat=True):
    sp = list(range(p)) if sp is None else sp
    sd = list(range(p, p + d)) if sd is None else sd
    rec = {"requested": [p, d], "created": created, "create_error": None if created else "boom"}
    if created:
        rec.update(prefill=census(sp, sat), decode=census(sd, sat),
                   driver={"prefill": readout(p), "decode": readout(d)},
                   concurrent=conc if conc is not None else [conc_block(sp, sd)])
    return rec


class RegisteredInputs(unittest.TestCase):
    def test_prompts_are_the_e2_literals(self):
        src = _read(E2)
        m = re.search(r"PROMPTS = (\[.*?\n\])", src, re.S)
        self.assertIsNotNone(m)
        self.assertEqual(tuple(ast.literal_eval(m.group(1))), H.P13_PROMPTS)
        self.assertIn('"sampling_params": {"temperature": 0, "max_new_tokens": 48}', src)

    def test_grid_comes_from_the_configs(self):
        pairs, src = H.grid_from_configs(H.GRID_CONFIGS)
        self.assertEqual(pairs, [(92, 16), (84, 24), (74, 34), (64, 44), (16, 92)])
        self.assertIn("16/92", src)

    def test_b4_control_factors_match_e2(self):
        e2, sh = _read(E2), _read(SHELL)
        for key in ("MODEL", "BACKEND"):
            v = re.search(rf'^{key}="([^"]+)"', e2, re.M).group(1)
            self.assertIn(f'{key}="{v}"', sh)
        for key in ("CTX", "MEM_FRACTION", "MAX_RUNNING", "SERVER_SEED", "FIXED_DSM"):
            v = re.search(rf"^{key}=(\S+)", e2, re.M).group(1)
            self.assertRegex(sh, rf"\b{key}={re.escape(v)}\b")
        self.assertIn("pdmux_homog5.yml", sh)
        launch = re.search(r"python -m sglang.launch_server(.*?)> \"\$SRVLOG\"", e2, re.S).group(1)
        flags = re.findall(r"--[a-z-]+", launch)
        for f in flags:
            if f not in ("--port",):
                self.assertIn(f, sh, f"B4 argv lacks e2 flag {f}")
        self.assertNotIn("--disable-cuda-graph", sh.split("SRV_ARGS=(")[1].split(")")[0])


class B3Judgments(unittest.TestCase):
    def test_pair_pass(self):
        a, b = H.judge_pair(pair_rec(74, 34), D)
        self.assertEqual((a["status"], b["status"]), ("PASS", "PASS"))
        self.assertTrue(a["facts"]["tiles_D"])

    def test_pair_requested_ne_realized_fails(self):
        a, _ = H.judge_pair(pair_rec(16, 92, sp=list(range(18)), sd=list(range(18, 108))), D)
        self.assertEqual(a["status"], "FAIL")
        self.assertIn("requested 16/92 != realized 18/90", a["reason"])

    def test_pair_overlap_fails(self):
        a, _ = H.judge_pair(pair_rec(16, 92, sp=list(range(16)), sd=list(range(10, 102))), D)
        self.assertEqual(a["status"], "FAIL")

    def test_pair_outside_D_fails(self):
        a, _ = H.judge_pair(pair_rec(16, 92, sd=list(range(16, 107)) + [200]), D)
        self.assertEqual(a["status"], "FAIL")

    def test_pair_unsaturated_unresolved(self):
        a, b = H.judge_pair(pair_rec(74, 34, sat=False), D)
        self.assertEqual((a["status"], b["status"]), ("UNRESOLVED", "UNRESOLVED"))

    def test_pair_creation_failure_fails(self):
        a, b = H.judge_pair(pair_rec(74, 34, created=False), D)
        self.assertEqual((a["status"], b["status"]), ("FAIL", "FAIL"))

    def test_driver_not_attached_unresolved(self):
        rec = pair_rec(74, 34)
        rec["driver"]["decode"] = {"green_ctx_is_null": True}
        a, _ = H.judge_pair(rec, D)
        self.assertEqual(a["status"], "UNRESOLVED")

    def test_concurrent_shared_label_fails(self):
        sp, sd = list(range(74)), list(range(74, 108))
        a, b = H.judge_pair(pair_rec(74, 34, conc=[conc_block(sp, sd + [0])]), D)
        self.assertEqual((a["status"], b["status"]), ("PASS", "FAIL"))

    def test_concurrent_escape_fails(self):
        sp, sd = list(range(74)), list(range(74, 108))
        _, b = H.judge_pair(pair_rec(74, 34, sd=sd, conc=[conc_block(sp[:70], sd[:30] + [5])]), D)
        self.assertEqual(b["status"], "FAIL")

    def test_concurrent_without_time_overlap_unresolved(self):
        sp, sd = list(range(74)), list(range(74, 108))
        _, b = H.judge_pair(pair_rec(74, 34, conc=[conc_block(sp, sd, overlap=False)]), D)
        self.assertEqual(b["status"], "UNRESOLVED")

    def _grid_raw(self, **kw):
        raw = {"complete": True, "total_sm_reported": 108, "multi_processor_count": 108,
               "plain_pre": census(range(108)), "plain_post": census(range(108)),
               "plain_driver": {"plain_pre": {"green_ctx_is_null": True},
                                "plain_post": {"green_ctx_is_null": True}},
               "runtime_ptx_smid_sites": 1, "runtime_spin_back_edge": True,
               "pairs": [pair_rec(92, 16), pair_rec(64, 44)]}
        raw.update(kw)
        return raw

    def test_grid_pass(self):
        plain, a, b = H.judge_grid(self._grid_raw())
        self.assertEqual([plain["status"], a["status"], b["status"]], ["PASS"] * 3)

    def test_grid_plain_carved_fails_and_blocks(self):
        plain, a, _ = H.judge_grid(self._grid_raw(plain_post=census(range(100))))
        self.assertEqual(plain["status"], "FAIL")
        self.assertEqual(a["status"], "UNRESOLVED")

    def test_grid_D_ne_device_fails(self):
        plain, _, _ = H.judge_grid(self._grid_raw(plain_pre=census(range(100)),
                                                  plain_post=census(range(100))))
        self.assertEqual(plain["status"], "FAIL")

    def test_grid_dead_instrument_unresolved(self):
        _, a, _ = H.judge_grid(self._grid_raw(runtime_ptx_smid_sites=0))
        self.assertEqual(a["status"], "UNRESOLVED")

    def test_grid_one_bad_pair_fails_stage(self):
        raw = self._grid_raw()
        raw["pairs"].append(pair_rec(16, 92, sp=list(range(18)), sd=list(range(18, 108))))
        _, a, _ = H.judge_grid(raw)
        self.assertEqual(a["status"], "FAIL")

    def test_grid_partial_unresolved(self):
        _, a, b = H.judge_grid(self._grid_raw(complete=False))
        self.assertEqual((a["status"], b["status"]), ("UNRESOLVED", "UNRESOLVED"))

    def test_gran_outcomes(self):
        ok = dict(pair_rec(18, 90), complete=True, runtime_ptx_smid_sites=1,
                  runtime_spin_back_edge=True)
        rounded = dict(pair_rec(17, 91, sp=list(range(18)), sd=list(range(18, 108))),
                       complete=True, runtime_ptx_smid_sites=1, runtime_spin_back_edge=True)
        rejected = {"requested": [1, 107], "created": False, "create_error": "CUDA_ERROR_X",
                    "complete": True}
        v = H.judge_gran([((18, 90), ok), ((17, 91), rounded), ((1, 107), rejected),
                          ((3, 105), None)], D)
        outs = [r["outcome"] for r in v["facts"]["rows"]]
        self.assertEqual(outs, ["EXACT", "ROUNDED", "REJECTED", "NO_RESULT"])
        self.assertEqual(v["status"], "UNRESOLVED")
        v2 = H.judge_gran([((18, 90), ok), ((17, 91), rounded), ((1, 107), rejected)], D)
        self.assertEqual(v2["status"], "PASS")
        breach = dict(pair_rec(16, 17, sp=list(range(16)), sd=list(range(10, 28))),
                      complete=True, runtime_ptx_smid_sites=1, runtime_spin_back_edge=True)
        self.assertEqual(H.judge_gran([((16, 17), breach)], D)["status"], "FAIL")

    def test_overlap_window_formula(self):
        ov, wp, wd = H.overlap_window(conc_block([1, 2], [3]))
        self.assertEqual((ov, wp, wd), (500, {1, 2}, {3}))
        self.assertEqual(H.overlap_window(conc_block([1], [3], overlap=False))[0], 0)

    def test_kisti_artefacts_map_to_pass(self):
        r0 = json.loads(_read(os.path.join(H.SMID_DIR, "smid_l0_verdict_889631.json")))
        p0a = json.loads(_read(os.path.join(H.BCG_DIR, "p0a_verdict_890893.json")))
        self.assertEqual(H.judge_r0(r0, r0)["status"], "PASS")
        self.assertEqual(H.judge_p0a(p0a, p0a)["status"], "PASS")
        self.assertEqual(H.judge_r0(None, r0)["status"], "UNRESOLVED")
        lost = dict(p0a, verdict="CONFINEMENT_LOST_THROUGH_GRAPH_REPLAY (FULL)")
        self.assertEqual(H.judge_p0a(lost, p0a)["status"], "FAIL")
        nocap = dict(p0a, verdict="UNDETERMINED (GRAPH CAPTURE UNAVAILABLE ON GREEN STREAM)")
        self.assertEqual(H.judge_p0a(nocap, p0a)["status"], "UNRESOLVED")
        bad = {"R0": {"verdict": "NOT_A_GLOBALLY_CONSISTENT_LABEL", "facts": {}}}
        self.assertEqual(H.judge_r0(bad, r0)["status"], "FAIL")


class B0Judgment(unittest.TestCase):
    CSV = ("NVIDIA A100-SXM4-80GB, GPU-abc, 580.105.08, 8.0, 81920 MiB, 1410 MHz, "
           "400.00 W, Disabled, Default\n")
    PY = {"cuda_available": True, "compute_capability": "8.0", "multi_processor_count": 108,
          "get_sm_available_0": 108}

    def test_pass(self):
        v = H.judge_b0(H.parse_gpu_csv(self.CSV), self.PY)
        self.assertEqual(v["status"], "PASS")
        self.assertTrue(all(x["same"] for x in v["facts"]["vs_VESSL_A100_CONTEXT"].values()))

    def test_fail_cases(self):
        self.assertEqual(H.judge_b0(H.parse_gpu_csv(self.CSV.replace(" 8.0,", " 9.0,")),
                                    self.PY)["status"], "FAIL")
        self.assertEqual(H.judge_b0(H.parse_gpu_csv(self.CSV.replace("Disabled", "Enabled")),
                                    self.PY)["status"], "FAIL")
        self.assertEqual(H.judge_b0(H.parse_gpu_csv(self.CSV),
                                    dict(self.PY, get_sm_available_0=132))["status"], "FAIL")
        self.assertEqual(H.judge_b0(H.parse_gpu_csv(self.CSV),
                                    dict(self.PY, sgl_kernel_error="x"))["status"], "FAIL")
        self.assertEqual(H.judge_b0(H.parse_gpu_csv(self.CSV * 2), self.PY)["status"], "FAIL")
        self.assertEqual(H.judge_b0(None, self.PY)["status"], "FAIL")

    def test_driver_drift_is_recorded_not_failed(self):
        v = H.judge_b0(H.parse_gpu_csv(self.CSV.replace("580.105.08", "590.1")), self.PY)
        self.assertEqual(v["status"], "PASS")
        self.assertFalse(v["facts"]["vs_VESSL_A100_CONTEXT"]["driver_version"]["same"])


class B4Judgment(unittest.TestCase):
    LOG = ("server_args=ServerArgs(model_path='x', enable_pdmux=True, disable_cuda_graph=False)\n"
           "Capture cuda graph bs [1, 2, 4]\nmax_total_num_tokens=12345\n"
           "max_mamba_cache_size: 48\n")

    def reqs(self, text="ok"):
        r = [{"text": text}] * len(H.P13_PROMPTS)
        return {"sequential": r, "concurrent": r}

    def test_pass(self):
        v = H.judge_b4(True, self.reqs(), H.parse_server_log(self.LOG))
        self.assertEqual(v["status"], "PASS")
        self.assertEqual(v["facts"]["server_log"]["max_mamba_cache_size"], 48)

    def test_fail_cases(self):
        lf = H.parse_server_log(self.LOG)
        self.assertEqual(H.judge_b4(False, self.reqs(), lf)["status"], "FAIL")
        self.assertEqual(H.judge_b4(True, self.reqs(""), lf)["status"], "FAIL")
        self.assertEqual(H.judge_b4(True, None, lf)["status"], "FAIL")
        eager = H.parse_server_log(self.LOG.replace("disable_cuda_graph=False",
                                                    "disable_cuda_graph=True"))
        self.assertEqual(H.judge_b4(True, self.reqs(), eager)["status"], "FAIL")
        tb = H.parse_server_log(self.LOG + "Traceback (most recent call last):\n")
        self.assertEqual(H.judge_b4(True, self.reqs(), tb)["status"], "FAIL")
        nocap = H.parse_server_log(self.LOG.replace("Capture cuda graph bs", "x"))
        self.assertEqual(H.judge_b4(True, self.reqs(), nocap)["status"], "UNRESOLVED")

    def test_green_readout(self):
        def grp(i, tp, td):
            if tp and td:
                return {"stream_index": i, "target_prefill_sm": tp, "target_decode_sm": td,
                        "prefill": readout(tp), "decode": readout(td)}
            return {"stream_index": i, "target_prefill_sm": tp, "target_decode_sm": td,
                    "prefill": {"green_ctx_is_null": True}, "decode": {"green_ctx_is_null": True}}
        gro = {"groups": [grp(0, 108, 0), grp(1, 92, 16), grp(2, 64, 44), grp(3, 0, 108)]}
        self.assertEqual(H.judge_green_readout(gro, None)["status"], "PASS")
        gro["groups"][1]["decode"] = readout(18)
        self.assertEqual(H.judge_green_readout(gro, None)["status"], "FAIL")
        self.assertEqual(H.judge_green_readout(None, None)["status"], "UNRESOLVED")


class ClockSummary(unittest.TestCase):
    def test_idle_bit_is_not_a_throttle(self):
        csv = ("timestamp, clocks.current.sm [MHz], power.draw [W], temperature.gpu, "
               "clocks_throttle_reasons.active\n"
               "2026/10/04 01:00:00.100, 1410 MHz, 60.00 W, 30, 0x0000000000000001\n"
               "2026/10/04 01:00:00.200, 1410 MHz, 300.00 W, 50, 0x0000000000000000\n"
               "2026/10/04 01:00:00.300, 1275 MHz, 400.00 W, 70, 0x0000000000000004\n")
        rows = H.parse_clk(csv)
        self.assertEqual(len(rows), 3)
        t0, t1 = rows[0][0] - 1, rows[-1][0] + 1
        s = H.clock_summary(rows, {"ALL": (t0, t1)})["ALL"]
        self.assertEqual((s["samples"], s["nonidle_reason_samples"], s["sm_mhz_min"]),
                         (3, 1, 1275))
        self.assertTrue(s["flag"])


if __name__ == "__main__":
    unittest.main()
