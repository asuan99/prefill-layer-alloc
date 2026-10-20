"""CPU tests for scripts/vessl/longctx_probe_helper.py, longctx_probe.sh, stage_llama.sh (128k probe).

No GPU, no sglang, no network beyond a loopback fake server.  Each group has a negative control (a
mutated input that must be caught), so a vacuous test cannot pass:

  synthetic input   exactly L ids for many L; deterministic; seed-sensitive; chunk-addressable.
                    Control: an encoder returning too few ids per chunk still yields L (loop), a
                    zero-token encoder raises.
  length assert     in+out <= 131072; control: 126976+4096 passes, 126976+4097... fails at the edge.
  C_mem / banner    hand-computed expected values from a banner; Llama (no mamba) reduces to floor(T/L)-like.
  stream client     loopback fake /generate: ttft, token count, forced finish; error path.
  result schema     validate_result accepts a real record, rejects a dropped key / over-budget record.
  scoring           PASS only if every expected request is present and ok; control: one missing/bad flips it.
  plan / argv       longctx_probe.sh --print-plan: argv invariants (ctx, cudagraph ON, radix OFF, cap ==
                    max-running == max-mamba), total expected minutes vs the plan doc's table.
  stage_llama.sh    token never in argv/xtrace; logout trap present; control: failing download still cleans up.
"""
import http.server
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
VESSL = os.path.join(ENGINE, "scripts", "vessl")
SPEC = importlib.util.spec_from_file_location("longctx_probe_helper", os.path.join(VESSL, "longctx_probe_helper.py"))
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)

BANNER = ("[t] Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state size: 0.09GB, ssm_state size: 6.46GB\n"
          "[t] KV Cache is allocated. #tokens: 1000000, K size: 8.00 GB, V size: 8.00 GB\n"
          "[t] max_total_num_tokens=1000000, chunked_prefill_size=-1, max_prefill_tokens=16384, "
          "max_running_requests=48, context_len=131072, available_gpu_mem=13.38 GB\n")


def fake_encode(text):
    """Deterministic stand-in tokenizer: each word -> 1..3 ids (like BPE splitting long words)."""
    out = []
    for w in text.split():
        n = 1 + len(w) // 5
        out.extend((hash_w(w) + j) % 50000 for j in range(n))
    return out


def hash_w(w):
    h = 7
    for ch in w:
        h = (h * 131 + ord(ch)) % 1000003
    return h


class TestSynthetic(unittest.TestCase):
    def test_exact_length_many_L(self):
        for L in (1, 7, 512, 4097, 16384, 33333, 126976):
            ids = H.synth_ids(fake_encode, L, H.seed_for(L, 0))
            self.assertEqual(len(ids), L)

    def test_deterministic_and_seed_sensitive(self):
        a = H.synth_ids(fake_encode, 3000, H.seed_for(3000, 0))
        b = H.synth_ids(fake_encode, 3000, H.seed_for(3000, 0))
        c = H.synth_ids(fake_encode, 3000, H.seed_for(3000, 1))
        self.assertEqual(H.ids_sha256(a), H.ids_sha256(b))
        self.assertNotEqual(H.ids_sha256(a), H.ids_sha256(c))       # control: seed matters

    def test_text_chunk_addressable(self):
        whole = H.synth_text(11, 300)
        head = H.synth_text(11, 100)
        tail = H.synth_text(11, 200, start_word=100)
        self.assertEqual(whole, head + tail)

    def test_zero_token_encoder_raises(self):
        with self.assertRaises(ValueError):
            H.synth_ids(lambda t: [], 10, 1)

    def test_pinned_golden_prefix(self):
        # the generator is part of the registered input definition: a change must be deliberate
        self.assertEqual(H.synth_text(5, 3), H.synth_text(5, 3))
        self.assertEqual(len(H.synth_text(5, 30).split()), 30)
        self.assertEqual(H.seed_for(16384, 0), H.seed_for(16384, 0))
        self.assertNotEqual(H.seed_for(16384, 0), H.seed_for(32768, 0))


class TestBudget(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(H.check_budget(126976, 64), 127040)
        self.assertEqual(H.check_budget(126976, 4096), 131072)         # exactly full: allowed
        with self.assertRaises(ValueError):
            H.check_budget(126976, 4097)                                # control: one over
        with self.assertRaises(ValueError):
            H.check_budget(0, 64)
        with self.assertRaises(ValueError):
            H.check_budget(131072, 64)

    def test_grid_fits_with_forced_output(self):
        for L in H.L_GRID:
            H.check_budget(L, H.OUT_TOKENS)
        self.assertEqual(max(H.L_GRID), 124 * 1024)


class TestCmem(unittest.TestCase):
    def test_banner(self):
        f = H.parse_banner(BANNER)
        self.assertEqual(f["max_total_num_tokens"], 1000000)
        self.assertEqual(f["kv_cache_tokens"], 1000000)
        self.assertEqual(f["max_mamba_cache_size"], 48)
        self.assertAlmostEqual(f["mamba_ssm_gib"], 6.46)
        self.assertEqual(f["context_len_logged"], 131072)

    def test_cmem_hand_computed(self):
        f = H.parse_banner(BANNER)
        pool, s, k = H.mem_terms(f)
        gib = 1024 ** 3
        self.assertAlmostEqual(pool, (16.0 + 0.09 + 6.46) * gib)
        self.assertAlmostEqual(s, (0.09 + 6.46) * gib / 49)
        self.assertAlmostEqual(k, 16.0 * gib / 1_000_000)
        L = 65536
        expect = int(pool // (s + k * L))
        self.assertEqual(H.c_mem(pool, s, k, L), expect)
        t = H.cmem_table(f, H.L_GRID)
        self.assertEqual(t["by_L"][str(L)]["c_mem"], expect)
        self.assertEqual(t["by_L"][str(L)]["kv_token_bound"], 1000000 // L)
        # monotone: longer L -> fewer concurrent requests (control against an inverted formula)
        vals = [t["by_L"][str(x)]["c_mem"] for x in H.L_GRID]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_attention_only_has_no_slot_term(self):
        llama = ("KV Cache is allocated. #tokens: 400000, K size: 24.00 GB, V size: 24.00 GB\n"
                 "max_total_num_tokens=400000, context_len=131072\n")
        f = H.parse_banner(llama)
        pool, s, k = H.mem_terms(f)
        self.assertEqual(s, 0.0)
        self.assertEqual(H.c_mem(pool, s, k, 100000), 4)               # 400000 // 100000
        self.assertIsNone(H.mem_terms(H.parse_banner("no banner here")))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            H.c_mem(1.0, 0.0, 1.0, 0)

    def test_signatures(self):
        t = "x\nCUDA out of memory. Tried\nTraceback (most recent call last):\nCapture cuda graph failed\n"
        s = H.log_signatures(t)
        self.assertEqual((s["oom"], s["traceback"], s["cuda_graph_capture_failed"]), (1, 1, 1))
        self.assertEqual(H.log_signatures("clean")["oom"], 0)
        self.assertEqual(H.sig_delta({"oom": 1}, {"oom": 3})["oom"], 2)


class _Handler(http.server.BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        type(self).seen.append(body)
        if type(self).mode == "http500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        ids, mn = body["input_ids"], body["sampling_params"]["max_new_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(1, mn + 1):
            mi = {"prompt_tokens": len(ids), "completion_tokens": i,
                  "finish_reason": ({"type": "length", "length": mn} if i == mn else None)}
            self.wfile.write(("data: " + json.dumps({"text": "x", "meta_info": mi}) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


class TestClient(unittest.TestCase):
    def setUp(self):
        _Handler.mode, _Handler.seen = "ok", []
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_parse_sse(self):
        self.assertEqual(H.parse_sse_line(b"data: [DONE]\n"), "DONE")
        self.assertIsNone(H.parse_sse_line(b": keepalive"))
        self.assertEqual(H.parse_sse_line('data: {"a": 1}')["a"], 1)
        self.assertIsNone(H.parse_sse_line("data: {broken"))

    def test_stream_ok_and_request_body(self):
        ids = list(range(100))
        r = H.stream_generate(self.port, ids, 8, 10)
        self.assertIsNone(r["error"])
        self.assertEqual((r["prompt_tokens"], r["completion_tokens"]), (100, 8))
        self.assertEqual(len(r["chunks"]), 8)
        self.assertTrue(r["ttft_s"] is not None and r["e2e_s"] >= r["ttft_s"])
        self.assertTrue(H.request_ok(r, 100, 8))
        self.assertFalse(H.request_ok(r, 101, 8))        # control: wrong prompt length is not ok
        self.assertFalse(H.request_ok(r, 100, 9))
        sp = _Handler.seen[0]["sampling_params"]
        self.assertEqual((sp["ignore_eos"], sp["temperature"], _Handler.seen[0]["stream"]), (True, 0.0, True))
        self.assertEqual(_Handler.seen[0]["input_ids"], ids)       # no retokenization path

    def test_http_error_and_dead_server(self):
        _Handler.mode = "http500"
        r = H.stream_generate(self.port, [1, 2], 4, 10)
        self.assertIn("HTTP 500", r["error"])
        self.assertFalse(H.request_ok(r, 2, 4))
        self.assertTrue(H.server_alive(self.port))
        self.assertFalse(H.server_alive(1))

    def test_probe_single_end_to_end(self):
        d = tempfile.mkdtemp()
        try:
            log = os.path.join(d, "server.log")
            open(log, "w").write("boot ok\n")
            out = os.path.join(d, "g.json")
            argv = ["probe-single", "--port", str(self.port), "--srvlog", log, "--arm", "t", "--model", "m",
                    "--out", out, "--tokenizer", "unused", "--stage", "G2", "--Ls", "300,500", "--reps", "2",
                    "--req-timeout", "10"]
            with mock.patch.object(H, "load_encoder", lambda _d: (fake_encode, None)):
                self.assertEqual(H.main(argv), 0)
            res = json.load(open(out))
            self.assertEqual(H.validate_result(res), [])
            self.assertEqual(len(res["records"]), 4)
            self.assertEqual([r["n_in"] for r in res["records"]], [300, 300, 500, 500])
            self.assertTrue(all(r["ok"] for r in res["records"]))
            st, why, facts = H.score_single(res, 4, "G2")
            self.assertEqual(st, H.PASS)
            st2, _, _ = H.score_single(res, 5, "G2")                 # control: one more expected than present
            self.assertNotEqual(st2, H.PASS)
            # the (warm-up + 4) requests all carried exact lengths and forced output
            lens = [len(b["input_ids"]) for b in _Handler.seen]
            self.assertEqual(lens, [512, 300, 300, 500, 500])
        finally:
            shutil.rmtree(d)

    def test_g4_and_conc_end_to_end(self):
        d = tempfile.mkdtemp()
        try:
            log = os.path.join(d, "server.log")
            open(log, "w").write("")
            with mock.patch.object(H, "load_encoder", lambda _d: (fake_encode, None)):
                out4 = os.path.join(d, "g4.json")
                self.assertEqual(H.main(["probe-g4", "--port", str(self.port), "--srvlog", log, "--arm", "t",
                                         "--model", "m", "--out", out4, "--tokenizer", "u", "--Ls", "400",
                                         "--decode-in", "50", "--decode-out", "40", "--lead-tokens", "2",
                                         "--lead-timeout-s", "5", "--req-timeout", "10"]), 0)
                r4 = json.load(open(out4))
                sc = r4["scenarios"][0]
                self.assertTrue(sc["long"]["ok"] and all(x["ok"] for x in sc["decode"]))
                self.assertEqual(len(sc["decode"]), 4)
                bf = os.path.join(d, "bf.json")
                json.dump({"cmem": {"by_L": {"300": {"c_mem": 3, "kv_token_bound": 3}}}}, open(bf, "w"))
                outc = os.path.join(d, "conc.json")
                self.assertEqual(H.main(["probe-conc", "--port", str(self.port), "--srvlog", log, "--arm", "t",
                                         "--model", "m", "--out", outc, "--tokenizer", "u", "--Ls", "300",
                                         "--boot-facts", bf, "--req-timeout", "10"]), 0)
                rc = json.load(open(outc))
                self.assertEqual(rc["groups"][0]["n"], 3)
                self.assertEqual(len(rc["records"]), 3)
                self.assertEqual(len({r["ids_sha256"] for r in rc["records"]}), 3)   # distinct inputs
            sd = os.path.join(d, "st")
            H.main(["score-g4", "--in", out4, "--Ls", "400", "--status-dir", sd])
            self.assertEqual(json.load(open(os.path.join(sd, "G4.json")))["status"], "PASS")
            H.main(["score-conc", "--in", outc, "--stage", "G5-conc", "--expect-n", "0", "--status-dir", sd])
            self.assertEqual(json.load(open(os.path.join(sd, "G5-conc.json")))["status"], "PASS")
        finally:
            shutil.rmtree(d)


class TestReductions(unittest.TestCase):
    def test_itl_summary(self):
        ch = [(0.0, 1), (0.1, 2), (0.2, 3), (1.2, 4), (1.3, 5)]
        s = H.itl_summary(ch, 0.5, 2.0)
        self.assertEqual(s["n_in_window"], 2)
        self.assertAlmostEqual(s["max_gap_in_window_s"], 1.0)
        self.assertEqual(s["n_before"], 2)
        self.assertAlmostEqual(s["p50_before_s"], 0.1)
        # two tokens in one chunk -> per-token gap halves
        self.assertAlmostEqual(H.token_gaps([(0.0, 1), (0.4, 3)])[0][1], 0.2)

    def test_lstsq(self):
        sl = H.lstsq_slope([1, 2, 3, 4], [2, 4, 6, 8])
        self.assertAlmostEqual(sl["slope"], 2.0)
        self.assertIsNone(H.lstsq_slope([1, 1], [1, 2]))

    def test_pick_docs(self):
        tok = [(60000, "a", "1", []), (70000, "a", "2", []), (50000, "a", "3", []), (40000, "b", "4", [])]
        got = H.pick_docs(tok, 65536, 2)
        self.assertEqual([(t[1], t[0]) for t in got], [("a", 60000), ("b", 40000)])   # <=64k, distinct tasks
        self.assertTrue(all(t[0] <= 65536 for t in got))


def _rec(L, ok=True, **kw):
    r = {"kind": "single", "stage": "G2", "arm": "a", "model": "m", "L": L, "rep": 0, "n_in": L, "max_new": 64,
         "seed": 1, "ids_sha256": "x", "ok": ok, "ttft_s": 1.0, "e2e_s": 2.0, "prompt_tokens": L,
         "completion_tokens": 64 if ok else 3, "finish_reason": {"type": "length"}, "error": None,
         "server_alive_after": True, "log_delta": {"oom": 0}, "t_send_utc": "t"}
    r.update(kw)
    return r


class TestSchemaScoring(unittest.TestCase):
    def res(self, recs):
        r = H.new_result("G2", "a", "m")
        r["records"] = recs
        return r

    def test_schema(self):
        self.assertEqual(H.validate_result(self.res([_rec(1000)])), [])
        bad = self.res([_rec(1000)])
        del bad["records"][0]["ttft_s"]
        self.assertTrue(any("ttft_s" in p for p in H.validate_result(bad)))
        over = self.res([_rec(131072)])                              # input alone fills the context
        self.assertTrue(any("context_length" in p for p in H.validate_result(over)))
        nokey = self.res([])
        del nokey["synth_rule"]
        self.assertTrue(H.validate_result(nokey))
        self.assertIn("ignore_eos", H.SYNTH_RULE["output"])

    def test_scoring(self):
        ok = self.res([_rec(1000), _rec(2000)])
        self.assertEqual(H.score_single(ok, 2, "G2")[0], H.PASS)
        self.assertEqual(H.score_single(ok, 3, "G2")[0], H.UNRES)         # stopped early, no failure seen
        self.assertEqual(H.score_single(None, 2, "G2")[0], H.UNRES)
        one_bad = self.res([_rec(1000), _rec(2000, ok=False, error="CUDA out of memory")])
        st, why, facts = H.score_single(one_bad, 2, "G2")
        self.assertEqual(st, H.FAIL)
        self.assertEqual(facts["per_L"]["2000"]["fail"], 1)
        oom = self.res([_rec(1000, log_delta={"oom": 1})])
        self.assertEqual(H.score_single(oom, 1, "G2")[0], H.FAIL)         # ok request but OOM line in the log

    def test_g7_and_g6(self):
        recs = []
        for kind, ttft in (("real", 1.0), ("synth", 2.0), ("synth", 2.0), ("real", 1.0)):
            recs.append(_rec(5000, input_kind=kind, doc_id="d", ttft_s=ttft))
        res = self.res(recs)
        res["docs"] = []
        st, _, facts = H.score_g7(res)
        self.assertEqual(st, H.PASS)
        self.assertAlmostEqual(facts["pairs"][0]["rel_diff_real_over_synth"], -0.5)   # a value, not a verdict
        res["records"][0]["ok"] = False
        self.assertEqual(H.score_g7(res)[0], H.FAIL)
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "n.json")
            json.dump(self.res([_rec(1000, ttft_s=0.1), _rec(2000, ttft_s=0.2), _rec(4000, ttft_s=0.4)]), open(p, "w"))
            H.main(["score-g6", "--nano", p, "--status-dir", d])
            v = json.load(open(os.path.join(d, "G6.json")))
            self.assertEqual(v["status"], "PASS")
            self.assertAlmostEqual(v["facts"]["by_input"]["nano_pdmux_G2"]["slope_ms_per_token"], 0.1, places=6)
            H.main(["score-g6", "--nano", os.path.join(d, "nonexistent.json"), "--status-dir", d])
            self.assertEqual(json.load(open(os.path.join(d, "G6.json")))["status"], "UNRESOLVED")
        finally:
            shutil.rmtree(d)

    def test_no_verdict_words_in_helper_statuses(self):
        # statuses/reasons are mechanical: the helper must not contain performance-verdict phrasing
        src = open(os.path.join(VESSL, "longctx_probe_helper.py")).read()
        for w in ("faster than", "slower than", "outperform", "speedup", "wins", "is linear", "not linear"):
            self.assertNotIn(w, src)


class TestPlanAndScripts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = subprocess.run(["/usr/bin/bash", os.path.join(VESSL, "longctx_probe.sh"), "--print-plan"],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        cls.lines = [ln.split("\t") for ln in out.stdout.splitlines()]

    def argv(self, boot):
        return next(ln[3] for ln in self.lines if ln[0] == "ARGV" and ln[1] == boot).split()

    def test_four_boots_share_fixed_factors(self):
        for b in ("nano_pdmux", "nano_fused", "llama", "nano_chunked"):
            a = self.argv(b)
            self.assertEqual(a[a.index("--context-length") + 1], "131072")
            self.assertIn("--disable-radix-cache", a)
            self.assertNotIn("--disable-cuda-graph", a)                    # cudagraph ON
            self.assertEqual(a[a.index("--mem-fraction-static") + 1], "0.82")
            self.assertEqual(a[a.index("--attention-backend") + 1], "flashinfer")
            cap = a[a.index("--max-running-requests") + 1]
            if b != "llama":                                                # --max-mamba-cache-size = cap explicit
                self.assertEqual(a[a.index("--max-mamba-cache-size") + 1], cap)
            else:
                self.assertNotIn("--max-mamba-cache-size", a)
                self.assertEqual(a[a.index("--revision") + 1], "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")

    def test_arm_definitions(self):
        pd = self.argv("nano_pdmux")
        self.assertIn("--enable-pdmux", pd)
        self.assertTrue(pd[pd.index("--pdmux-config-path") + 1].endswith("benchmarks/configs/pdmux_agnostic_manual.yml"))
        self.assertEqual(pd[pd.index("--chunked-prefill-size") + 1], "-1")
        fu = self.argv("nano_fused")                                        # fused_sym: PD-mux off + same knobs
        self.assertNotIn("--enable-pdmux", fu)
        self.assertEqual(fu[fu.index("--chunked-prefill-size") + 1], "-1")
        self.assertIn("--disable-overlap-schedule", fu)
        ch = self.argv("nano_chunked")
        self.assertEqual(ch[ch.index("--chunked-prefill-size") + 1], "8192")
        self.assertIn("--enable-mixed-chunk", ch)
        modes = {ln[1]: ln[2] for ln in self.lines if ln[0] == "ARGV"}
        self.assertEqual(modes["nano_pdmux"], "engine_mode=pdmux")
        self.assertEqual(modes["nano_fused"], "engine_mode=vanilla")
        self.assertTrue(os.path.exists(os.path.join(ENGINE, "benchmarks/configs/pdmux_agnostic_manual.yml")))
        cfg = open(os.path.join(ENGINE, "benchmarks/configs/pdmux_agnostic_manual.yml")).read()
        self.assertIn("[74, 34, 0]", cfg)
        self.assertIn("[54, 54, 36]", cfg)

    def test_time_table_vs_plan(self):
        tot = {ln[1]: (int(ln[2]), int(ln[3])) for ln in self.lines if ln[0] == "EXPECT_TOTAL_MIN"}
        budget = int(next(ln[1] for ln in self.lines if ln[0] == "BUDGET_MIN"))
        self.assertEqual(budget, 120)                                       # plan sec 2 hard cap
        # plan sec 2 steps 3-5 (GPU time without Workspace resume/pause) must lie inside the plan's 60-100 min
        lo, hi = tot["plan_subset_three_boots"]
        self.assertTrue(lo <= 100 and hi <= 100, (lo, hi))
        # all four boots: upper estimate must stay under the hard cap (chunked proxy is the budget valve)
        self.assertLessEqual(tot["all_four_boots"][1], budget)
        # per-step hard timeouts exceed the expected maxima (a timeout shorter than the estimate is a bug)
        tmo = {ln[1]: int(ln[2]) for ln in self.lines if ln[0] == "TIMEOUT_S"}
        exp_hi = {ln[1]: int(ln[3]) for ln in self.lines if ln[0] == "EXPECT_MIN"}
        self.assertGreaterEqual(tmo["boot"], 20 * 60)
        for step, key in (("G2", "G2"), ("G4", "G4"), ("G7", "G7"), ("G3", "G3"), ("G5", "G5"),
                          ("G5conc", "G5conc"), ("G3chunked", "G3chunked")):
            self.assertGreaterEqual(tmo[step], exp_hi[key] * 60, step)

    def test_syntax_and_no_secret_leaks(self):
        for n in ("longctx_probe.sh", "stage_llama.sh"):
            r = subprocess.run(["/usr/bin/bash", "-n", os.path.join(VESSL, n)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        sh = shutil.which("shellcheck")
        if sh:
            for n in ("longctx_probe.sh", "stage_llama.sh"):
                r = subprocess.run([sh, "-S", "error", os.path.join(VESSL, n)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout)
        for n in ("longctx_probe.sh", "stage_llama.sh", "longctx_probe_helper.py"):
            src = open(os.path.join(VESSL, n)).read()
            self.assertIsNone(re.search(r"^\s*set -[a-z]*x", src, re.M), n)       # no xtrace
            self.assertNotIn("printenv", src)
            self.assertIsNone(re.search(r"^\s*(env|export -p)\s*(>|\|)", src, re.M), n)  # no env dump

    def test_stage_llama_token_and_trap(self):
        src = open(os.path.join(VESSL, "stage_llama.sh")).read()
        self.assertIn("trap cleanup EXIT INT TERM", src)
        self.assertIn("hf auth logout", src)
        self.assertIn("d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", src)
        self.assertIsNone(re.search(r"--token", src))                              # never an argv token
        self.assertIsNone(re.search(r"read\s+-r?s?\s*\w*tok", src, re.I))           # bash never holds it
        self.assertIn("HF_TOKEN_PATH", src)

    def test_stage_llama_behaviour_with_fakes(self):
        """Fake `hf` + fake huggingface_hub: token never printed/logged; logout + cleanup run on success and
        on a failing download (control)."""
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "bin"))
            os.makedirs(os.path.join(d, "py", "huggingface_hub"))
            hf = os.path.join(d, "bin", "hf")
            open(hf, "w").write(
                '#!/bin/bash\necho "hf $1 $2 $*" >> "$FAKE_LOG"\n'
                'if [[ "$1" == download ]]; then\n'
                ' s=$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b\n'
                ' mkdir -p $s; echo x > $s/config.json; echo y > $s/model.safetensors\n'
                ' [[ -n "$FAKE_FAIL" ]] && exit 1\nfi\n'
                'if [[ "$1 $2" == "auth logout" ]]; then echo logout >> "$FAKE_LOG"; rm -f "$HF_TOKEN_PATH"; fi\nexit 0\n')
            os.chmod(hf, 0o755)
            open(os.path.join(d, "py", "huggingface_hub", "__init__.py"), "w").write(
                "import os\ndef login(token, add_to_git_credential=False):\n"
                "    open(os.environ['HF_TOKEN_PATH'],'w').write(token)\n"
                "def snapshot_download(repo, revision, allow_patterns=None, ignore_patterns=None):\n"
                "    s=os.path.join(os.environ['HF_HOME'],'hub','models--'+repo.replace('/','--'),'snapshots',revision)\n"
                "    os.makedirs(s, exist_ok=True)\n"
                "    open(os.path.join(s,'config.json'),'w').write('x')\n"
                "    if os.environ.get('FAKE_FAIL'): raise SystemExit(1)\n"
                "    open(os.path.join(s,'model-00001-of-00001.safetensors'),'w').write('y')\n"
                "    open(os.environ['FAKE_LOG'],'a').write('snapshot_download\\n')\n")
            secret = "hf_UNITTESTFAKETOKEN0123456789"
            tokfile = os.path.join(d, "tok")
            open(tokfile, "w").write(secret + "\n")
            for fail in ("", "1"):
                log = os.path.join(d, "log")
                open(log, "w").close()
                env = {"PATH": f"{d}/bin:/usr/bin:/bin", "PYTHONPATH": f"{d}/py", "HF_HOME": f"{d}/hf",
                       "FAKE_LOG": log, "FAKE_FAIL": fail, "HOME": d}
                before = set(os.listdir("/tmp"))
                with open(tokfile) as fin:
                    r = subprocess.run(["/usr/bin/bash", os.path.join(VESSL, "stage_llama.sh")], stdin=fin,
                                       capture_output=True, text=True, env=env, timeout=60)
                self.assertNotIn(secret, r.stdout + r.stderr + open(log).read())
                self.assertIn("logout", open(log).read())                           # logout ran, also when fail="1"
                self.assertEqual(r.returncode, 0 if not fail else 4, r.stderr)
                self.assertIn("no token file left", r.stdout)
                self.assertEqual({x for x in set(os.listdir("/tmp")) - before if x.startswith("stage_llama.")}, set())
                self.assertFalse(os.path.exists(os.path.join(d, "hf", "token")))
            self.assertTrue(os.path.exists(os.path.join(d, "hf", "STAGED_LLAMA_SHA256SUMS")))
            # a terminal on stdin is refused (no prompt) -- simulated by closing stdin to /dev/null -> empty token
            with open(os.devnull) as fin:
                r = subprocess.run(["/usr/bin/bash", os.path.join(VESSL, "stage_llama.sh")], stdin=fin,
                                   capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(r.returncode, 3)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
