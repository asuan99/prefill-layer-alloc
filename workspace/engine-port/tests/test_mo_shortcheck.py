"""CPU tests of the second short confirmation Job (scripts/vessl/mo_shortcheck.py, 2026-11-10).  No GPU, no engine; the only socket is a
localhost fake of SGLang's /generate that applies the same length rule as tokenizer_manager._validate_one_request.  Nothing here shows that
the Job boots or that any flag works on a GPU: those are what the Job itself is for."""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
VESSL = ENG / "scripts" / "vessl"
sys.path.insert(0, str(VESSL))
sys.path.insert(0, str(ENG / "results" / "motivation"))
import mo_shortcheck as M  # noqa: E402
import py312_probe as P  # noqa: E402


class FakeGenerate(BaseHTTPRequestHandler):
    ctx = 262144
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        n_in, n_out = len(body["input_ids"]), body["sampling_params"]["max_new_tokens"]
        FakeGenerate.seen.append((n_in, n_out))
        if n_in >= self.ctx or n_in + n_out >= self.ctx:          # tokenizer_manager.py:795 / :815 (strict)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "longer than the model context length"}')
            return
        out = json.dumps({"text": "t" * n_out, "meta_info": {"completion_tokens": n_out, "prompt_tokens": n_in}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(out)


class TestLengthRule(unittest.TestCase):
    def test_requests_follow_the_boundary_rule(self):
        reqs = M.ctx_requests()
        ctx = M.CTX_262144
        by = {e: [] for e in ("ok", "http_400")}
        for n_in, n_out, e in reqs:
            by[e].append((n_in, n_out))
        self.assertEqual(len(by["http_400"]), 1)
        n_in, n_out = by["http_400"][0]
        self.assertEqual(n_in + n_out, ctx)                       # first rejected total
        for n_in, n_out in by["ok"]:
            self.assertLess(n_in + n_out, ctx)
            self.assertGreaterEqual(n_in, ctx - 128)              # 'just below 2^18'
        self.assertIn(ctx - M.CTX_OUT - 1, [a for a, _ in by["ok"]])   # the tightest accepted length is exercised

    def test_first_probe_failure_reproduced_by_old_requests(self):
        # the first probe sent input == context length: the fake (= SGLang rule) rejects it
        srv = HTTPServer(("127.0.0.1", 0), FakeGenerate)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = M.gen(srv.server_port, M.CTX_262144, 16, 1, 60)
            self.assertEqual(r["status"], 400)
            self.assertFalse(r["ok"])
        finally:
            srv.shutdown()

    def test_driver_ok_against_rule_and_bad_expectation_fails(self):
        FakeGenerate.seen.clear()
        srv = HTTPServer(("127.0.0.1", 0), FakeGenerate)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            res = M.drive_ctx(srv.server_port, "d", 60)
            self.assertTrue(res["ok"], res)
            self.assertTrue(res["timing_use_only"])
            self.assertEqual([x["status"] for x in res["rows"]], [400, 200, 200])
            self.assertEqual([x["prompt_tokens"] for x in res["rows"][1:]], [x["n_in"] for x in res["rows"][1:]])
            # mutant: an engine that accepts everything must fail the negative control
            with mock.patch.object(FakeGenerate, "ctx", 10 ** 9):
                bad = M.drive_ctx(srv.server_port, "d", 60)
            self.assertFalse(bad["ok"])
            # mutant: an engine that truncates silently (prompt_tokens != n_in) must fail
            orig = M.gen

            def trunc(port, n_in, n_out, seed, timeout):
                r = orig(port, n_in, n_out, seed, timeout)
                if r["ok"]:
                    r["prompt_tokens"] = n_in - 1
                return r
            with mock.patch.object(M, "gen", trunc):
                self.assertFalse(M.drive_ctx(srv.server_port, "d", 60)["ok"])
        finally:
            srv.shutdown()

    def test_ctx_boot_is_the_first_probe_boot(self):
        b = M.ctx_cmd()
        self.assertEqual(flag_value(b["cmd"], "--context-length"), "262144")
        self.assertEqual(b["env"], {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"})
        self.assertEqual(b["tree"], "pristine")


def flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.out = Path("/tmp/OUT")
        self.pl = M.plan(self.out)

    def test_s2_flags(self):
        by = {b["id"]: b for b in self.pl["S2"]}
        self.assertEqual(set(by), {"pm_off_main", "pm_on_default", "pm_on_max8192", "pm_on_max16384"})
        for k, b in by.items():
            self.assertIn("--enable-pdmux", b["cmd"], k)
            self.assertEqual(flag_value(b["cmd"], "--chunked-prefill-size"), "-1", k)       # the cause the first probe found
            self.assertEqual(b["tree"], "patched")
        self.assertIn("--disable-piecewise-cuda-graph", by["pm_off_main"]["cmd"])
        for k in ("pm_on_default", "pm_on_max8192", "pm_on_max16384"):
            self.assertNotIn("--disable-piecewise-cuda-graph", by[k]["cmd"], k)
        self.assertNotIn("--piecewise-cuda-graph-max-tokens", by["pm_on_default"]["cmd"])
        self.assertEqual(flag_value(by["pm_on_max8192"]["cmd"], "--piecewise-cuda-graph-max-tokens"), "8192")
        self.assertEqual(flag_value(by["pm_on_max16384"]["cmd"], "--piecewise-cuda-graph-max-tokens"), "16384")
        # only the treatment differs: strip the piecewise flags and the argv are identical
        strip = lambda c: [x for i, x in enumerate(c) if x not in ("--disable-piecewise-cuda-graph", "--piecewise-cuda-graph-max-tokens")
                           and c[i - 1] != "--piecewise-cuda-graph-max-tokens"]
        ref = strip(by["pm_off_main"]["cmd"])
        for k in by:
            self.assertEqual(strip(by[k]["cmd"]), ref, k)
            self.assertEqual(by[k]["env"], by["pm_off_main"]["env"], k)

    def test_s2_main_arm_equals_registered_bundle(self):
        import mo_spec as MS
        reg = P.resolve(MS.build_arm(M.arm_params("PDMUX_BUNDLE"), M.BOOT_FIXED)["cmd"])
        self.assertEqual({b["id"]: b for b in self.pl["S2"]}["pm_off_main"]["cmd"], reg)

    def test_s4_design(self):
        s4 = self.pl["S4"]
        self.assertEqual(len(s4), 8)
        cfgs, orders = M.abba_orders()
        for cfg in cfgs:
            self.assertIn(orders[cfg], (["OFF", "ON", "ON", "OFF"], ["ON", "OFF", "OFF", "ON"]))
        ids = [b["id"] for b in s4]
        self.assertEqual(len(set(ids)), 8)
        for cfg in ("SG", "PM"):
            on = [b for b in s4 if b["id"].startswith(f"s4_{cfg}_ON")]
            off = [b for b in s4 if b["id"].startswith(f"s4_{cfg}_OFF")]
            self.assertEqual((len(on), len(off)), (2, 2))
            for b in on + off:                                                              # same argv, same tree, same driver
                self.assertEqual(b["cmd"], on[0]["cmd"])
                self.assertEqual(b["tree"], "patched")
                self.assertIs(b["driver"], M.drive_instr)
            for b in off:                                                                   # OFF: no observation knob at all
                self.assertFalse(set(M.ON_WINDOW_FLAGS + M.PM_ONLY_FLAGS) & set(b["env"]), b["id"])
            want = set(M.ON_WINDOW_FLAGS) | (set(M.PM_ONLY_FLAGS) if cfg == "PM" else set())
            for b in on:
                self.assertTrue(want <= set(b["env"]), (b["id"], want - set(b["env"])))
                if cfg == "SG":
                    self.assertFalse(set(M.PM_ONLY_FLAGS) & set(b["env"]))
                for k, v in b["env"].items():
                    if k.endswith("_PATH"):
                        self.assertTrue(v.startswith(f"/tmp/OUT/{b['id']}/"), (k, v))        # each boot writes into its own dir
                    elif k in want and not k.endswith("_PATH"):
                        self.assertEqual(v, "1")
            # everything except the ON knobs equals between ON and OFF
            drop = lambda e: {k: v for k, v in e.items() if k not in M.ON_WINDOW_FLAGS + M.PM_ONLY_FLAGS and k != "PDMUX_WORKLOAD_ID"}
            self.assertEqual(drop(on[0]["env"]), drop(off[0]["env"]))
        sg = next(b for b in s4 if "_SG_" in b["id"])
        self.assertIn("--enable-mixed-chunk", sg["cmd"])
        self.assertNotIn("--enable-pdmux", sg["cmd"])
        self.assertTrue(sg["want_piecewise"])
        pm = next(b for b in s4 if "_PM_" in b["id"])
        self.assertIn("--enable-pdmux", pm["cmd"])
        self.assertIn("--disable-piecewise-cuda-graph", pm["cmd"])

    def test_on_flag_names_exist_in_engine_sources(self):
        src = "".join((ENG / "src" / "multiplex" / n).read_text() for n in ("decode_probe.py", "admission_probe.py", "multiplexing_mixin.py"))
        for k in M.ON_WINDOW_FLAGS + M.PM_ONLY_FLAGS:
            self.assertIn(k, src, k)

    def test_orders_are_seeded(self):
        self.assertEqual(M.s2_order(), M.s2_order())
        self.assertEqual(sorted(M.a31_order()), sorted(M.A31_CANDS))


class TestA31(unittest.TestCase):
    def test_vllm_cmd(self):
        cmd = M.vllm_cmd("/v/a31", M.a31_flags("block16"))
        self.assertEqual(flag_value(cmd, "--model"), "/v/a31")
        self.assertNotIn("--revision", cmd)
        self.assertEqual(flag_value(cmd, "--load-format"), "dummy")
        self.assertEqual(flag_value(cmd, "--block-size"), "16")
        self.assertNotIn("--block-size", M.vllm_cmd("/v/a31", M.a31_flags("as_is")))
        self.assertIn("--disable-hybrid-kv-cache-manager", M.vllm_cmd("/v/a31", M.a31_flags("nohybridkv")))
        for c in M.A31_CANDS:                                                              # only the candidate flag differs from as_is
            base = M.vllm_cmd("/v/a31", [])
            extra = M.vllm_cmd("/v/a31", M.a31_flags(c))[len(base):]
            self.assertEqual(extra, M.a31_flags(c))

    def test_neutralize_does_not_mutate(self):
        cfg = {"mamba_num_heads": 128, "x": 1}
        out = M.neutralize_mamba(cfg)
        self.assertEqual(out["mamba_num_heads"], 0)
        self.assertEqual(cfg["mamba_num_heads"], 128)

    def test_block_facts_and_verdict(self):
        txt = ("INFO Setting attention block size to 1312 tokens to ensure that attention page size is >= mamba page size.\n"
               "INFO GPU KV cache size: 123,456 tokens\nINFO unrelated line\n")
        f = M.vllm_block_facts(txt)
        self.assertEqual(f["block_size_attn_line"], 1312)
        self.assertEqual(f["kv_cache_tokens"], [123456])
        d = dict(f, boot_ok=True)
        self.assertEqual(M.a31_verdict(d, dict(d)), "UNCHANGED")
        c = dict(M.vllm_block_facts("INFO Setting attention block size to 16 tokens\n"), boot_ok=True)
        self.assertEqual(M.a31_verdict(d, c), "CHANGED")
        self.assertEqual(M.a31_verdict(d, {"boot_ok": False}), "BOOT_FAILED")
        self.assertEqual(M.a31_verdict(d, {"boot_ok": True, "block_size_attn_line": None, "block_size_last": None}), "UNPARSED")


class TestCaptureFacts(unittest.TestCase):
    def test_sizes_and_reasons(self):
        txt = ("server_args=ServerArgs(piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 16], x=1)\n"
               "Capture piecewise CUDA graph begin. avail mem=40.00 GB\nCapture piecewise CUDA graph end. Time elapsed: 30 s\n"
               "max_total_num_tokens=1000, available_gpu_mem=20.5 GB\n")
        f = M.sg_capture_facts(txt)
        self.assertEqual((f["n_capture_sizes"], f["max_capture_size"], f["captured"], f["failed"]), (3, 16, True, False))
        self.assertEqual(f["max_total_num_tokens"], 1000)
        off = M.sg_capture_facts("piecewise_cuda_graph_max_tokens=-1, piecewise_cuda_graph_tokens=[], \n"
                                 "Disable piecewise CUDA graph because the capture size is not set\n")
        self.assertFalse(off["captured"])
        self.assertEqual(off["n_capture_sizes"], 0)
        self.assertIn("capture size is not set", off["disabled_reason"])


def eqrec(h, n=4):
    return [{"sha": f"{h}{i}", "n": 64, "ok": True} for i in range(n)]


class TestAnalysis(unittest.TestCase):
    def boots(self, off_b="x", on_a="x", on_b="x", rep_a=None):
        mk = lambda h, r=None: {"eq1": eqrec(h), "eq2": eqrec(r or h), "mix_wall_s": [10.0, 10.2, 10.1], "dec_wall_s": [8.0, 8.1]}
        return {"OFF_a": mk("x"), "OFF_b": mk(off_b), "ON_a": mk(on_a, rep_a), "ON_b": mk(on_b)}

    def test_equiv_pass(self):
        self.assertEqual(M.equivalence(self.boots())["verdict"], "EQUIV_PASS")

    def test_equiv_fail_when_on_differs_and_floor_zero(self):
        v = M.equivalence(self.boots(on_a="y", on_b="y"))
        self.assertEqual(v["verdict"], "EQUIV_FAIL")

    def test_floor_nonzero(self):
        self.assertEqual(M.equivalence(self.boots(off_b="y"))["verdict"], "FLOOR_NONZERO")
        self.assertEqual(M.equivalence(self.boots(rep_a="y"))["verdict"], "FLOOR_NONZERO")     # within-boot repeat differs

    def test_incomplete(self):
        b = self.boots()
        del b["ON_b"]
        self.assertEqual(M.equivalence(b)["verdict"], "INCOMPLETE")
        b = self.boots()
        b["ON_b"]["eq1"] = None
        self.assertEqual(M.equivalence(b)["verdict"], "INCOMPLETE")

    def test_one_token_count_difference_counts(self):
        b = self.boots()
        for boot in ("ON_a", "ON_b"):
            for k in ("eq1", "eq2"):
                b[boot][k][0] = {"sha": "x0", "n": 63, "ok": False}
        self.assertEqual(M.equivalence(b)["verdict"], "EQUIV_FAIL")

    def test_overhead(self):
        b = self.boots()
        for k in ("ON_a", "ON_b"):
            b[k]["mix_wall_s"] = [20.0, 20.0, 20.0]
        o = M.overhead(b)
        self.assertAlmostEqual(o["mix_wall_s"]["ratio_on_over_off"], 20.0 / 10.1, places=6)
        self.assertEqual(o["mix_wall_s"]["status"], "PROBE_VALUE_NOT_CITABLE")
        self.assertAlmostEqual(o["dec_wall_s"]["ratio_on_over_off"], 1.0)
        del b["OFF_b"]
        self.assertEqual(M.overhead(b)["mix_wall_s"]["status"], "INCOMPLETE")

    def test_record_facts(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "decode_probe.jsonl").write_text("\n".join(json.dumps({"event": e}) for e in ("decode_forward", "extend_forward", "forward_interior")) + "\n")
            (d / "admission.jsonl").write_text(json.dumps({"event": "admission_request"}) + "\n")
            (d / "tel.jsonl").write_text(json.dumps({"event": "decode_iteration", "t_issue_start": 1.0}) + "\nnot json\n")
            f = M.record_facts(d, True, True)
            self.assertEqual(f["missing_kinds"], ["mixed_opportunity"])
            self.assertTrue(f["t_issue_start_seen"])
            self.assertEqual(M.record_facts(d, False, True)["missing_kinds"], ["mixed_opportunity"])
            self.assertFalse(M.record_facts(d, True, False)["off_files_absent"])        # OFF boot that wrote a file is flagged
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(M.record_facts(t, False, False)["off_files_absent"])


class TestDriverHook(unittest.TestCase):
    """py312_probe.run_boot: a stage driver replaces the fixed request list and decides PASS (mutation: driver false -> FAIL)."""

    def run_one(self, ok):
        with tempfile.TemporaryDirectory() as t:
            proc = mock.Mock(pid=12345)
            proc.poll.return_value = None
            proc.wait.return_value = 0
            b = M.boot_dict("x", "patched", ["-m", "sglang.launch_server"], {}, False, driver=lambda port, d, rt: {"ok": ok, "seen": [port, str(d)]})
            with mock.patch.object(P.subprocess, "Popen", return_value=proc), mock.patch.object(P.urllib.request, "urlopen"), \
                    mock.patch.object(P, "gpu_used_mib", return_value=0), mock.patch.object(P.os, "killpg"), mock.patch.object(P.time, "sleep"):
                res = P.run_boot(b, t, boot_timeout=30, req_timeout=30)
            return res

    def test_driver_decides(self):
        self.assertEqual(self.run_one(True)["status"], "PASS")
        r = self.run_one(False)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["requests"], [])


class TestJobScript(unittest.TestCase):
    def test_shell_syntax_and_digest(self):
        p = VESSL / "mo_shortcheck_job.sh"
        self.assertEqual(subprocess.run(["bash", "-n", str(p)]).returncode, 0)
        self.assertTrue(os.access(p, os.X_OK))
        txt = p.read_text()
        self.assertIn(M.ENGINE_DIGEST_EXPECTED, txt)
        self.assertIn(M.ENGINE_DIGEST_EXPECTED, (ENG / "env" / "dev_tree_edits.md").read_text())
        self.assertIn("gpu_count", txt)

    def test_env_file_pins_the_probe_image(self):
        env = (VESSL / "vessl.sg312.env").read_text()
        sg312 = re.search(r'^PDMUX_VESSL_IMAGE_SG312="([^"]+)"', (VESSL / "vessl.env").read_text(), re.M).group(1)
        img = re.search(r'^PDMUX_VESSL_IMAGE="([^"]+)"', env, re.M).group(1)
        self.assertEqual(img, sg312)
        self.assertIn("@sha256:4414b5f9", img)
        other = lambda s: [ln for ln in s.splitlines() if not ln.startswith(("PDMUX_VESSL_IMAGE=", "#"))]
        self.assertEqual(other(env), other((VESSL / "vessl.env").read_text()))


if __name__ == "__main__":
    unittest.main()
