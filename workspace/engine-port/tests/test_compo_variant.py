"""CPU tests for scripts/compo/make_variant.py, scripts/vessl/compo_probe_helper.py, compo_probe.sh (decision 24).

No GPU, no network.  Each group carries a negative control so a vacuous test cannot pass:

  swap rule        counts, exact positions (hand-listed), determinism, MLP layers untouched, n_attn=4 identity,
                   n_attn=0 pure-SSM; control: the off-by-half rule floor(j*M/n) gives different positions.
  predictions      original config: s = 143,548,416 B (boot log 143.5 MB), k = 16,384 B (boot log 16,382 B),
                   params 8,888,227,328, weights vs boot log 16.58 GiB; SGLang's own Mamba2CacheParams agrees
                   for every variant (skipped if the sglang config modules cannot be imported).
  banner check     a real A100 boot banner (results/longctx_conflict/probes/qa_r3_906506/srv_r3_d44.log) passes
                   PRED for the original prediction with implied layer counts 4/27; control: the a13
                   prediction against the same banner FAILs.
  variant.json     schema keys, weights excluded, config differs only in hybrid_override_pattern, sha256s.
  plan / argv      compo_probe.sh --print-plan: dummy flag exactly on dummy boots, PD-mux on all but the vanilla
                   boot, fixed factors, a31 retry drops --max-mamba-cache-size, time table within the budget.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
VESSL = os.path.join(ENGINE, "scripts", "vessl")
COMPO = os.path.join(ENGINE, "scripts", "compo")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MV = _load("make_variant", os.path.join(COMPO, "make_variant.py"))
CH = _load("compo_probe_helper", os.path.join(VESSL, "compo_probe_helper.py"))

# config.json of nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base rev dc0661c829b14e5b9246c05cfa89094a0875e052 (fields used)
ORIG_CFG = {
    "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h",
    "auto_map": {"AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                 "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"},
    "attention_bias": False, "chunk_size": 128, "conv_kernel": 4, "head_dim": 128, "hidden_size": 4480,
    "hybrid_override_pattern": "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-",
    "intermediate_size": 15680, "mamba_head_dim": 80, "mamba_num_groups": 8, "mamba_num_heads": 128,
    "mamba_proj_bias": False, "mamba_state_dim": 128, "mlp_bias": False, "n_groups": 8,
    "num_attention_heads": 40, "num_hidden_layers": 56, "num_key_value_heads": 8, "ssm_state_size": 128,
    "tie_word_embeddings": False, "time_step_limit": [0.0, float("inf")], "use_bias": False,
    "use_conv_bias": True, "vocab_size": 131072, "torch_dtype": "bfloat16",
}

# real boot log lines (A100, PD-mux, cap 48, original weights), plus the server_args fields the helper reads
REAL_BANNER = (
    "[2026-09-10 21:35:01] server_args=ServerArgs(model_path='nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base', "
    "load_format='auto', trust_remote_code=True, context_length=8704)\n"
    "[2026-09-10 21:35:10] PD-Multiplexing enabled with 6 stream groups, sm_counts (prefill_sm, decode_sm): "
    "[(108, 0), (92, 16), (64, 44), (54, 54), (16, 92), (0, 108)]\n"
    "[2026-09-10 21:35:20] Load weight end. elapsed=8.04 s, type=NemotronHForCausalLM, avail mem=62.10 GB, mem usage=16.58 GB.\n"
    "[2026-09-10 21:35:20] Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state size: 0.09GB, ssm_state size: 6.46GB \n"
    "[2026-09-10 21:35:20] KV Cache is allocated. #tokens: 2618152, K size: 19.97 GB, V size: 19.97 GB\n"
    "[2026-09-10 21:35:20] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]\n"
    "[2026-09-10 21:35:29] Capture cuda graph end. Time elapsed: 8.32 s. mem usage=1.27 GB. avail mem=13.17 GB.\n"
    "[2026-09-10 21:35:30] max_total_num_tokens=2618152, chunked_prefill_size=-1, max_prefill_tokens=16384, "
    "max_running_requests=48, context_len=8704, available_gpu_mem=13.17 GB\n")


def _rj(path):
    with open(path) as f:
        return json.load(f)


def _txt(path):
    with open(path) as f:
        return f.read()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def variant_pattern(n):
    return MV.make_pattern(ORIG_CFG["hybrid_override_pattern"], n)[0]


class TestSwapRule(unittest.TestCase):
    def test_counts_and_fixed_mlp(self):
        orig = ORIG_CFG["hybrid_override_pattern"]
        self.assertEqual((orig.count("M"), orig.count("*"), orig.count("-"), len(orig)), (27, 4, 25, 56))
        mlp = [i for i, c in enumerate(orig) if c == "-"]
        for n in MV.N_ATTN_CHOICES:
            p = variant_pattern(n)
            self.assertEqual(len(p), 56)
            self.assertEqual(p.count("*"), n)
            self.assertEqual(p.count("M"), 31 - n)
            self.assertEqual([i for i, c in enumerate(p) if c == "-"], mlp)       # MLP layers never move

    def test_exact_positions(self):
        exp = {13: [2, 7, 13, 20, 27, 34, 41, 46, 52],
               22: [0, 4, 6, 9, 11, 16, 18, 23, 25, 29, 32, 36, 38, 43, 44, 48, 50, 54],
               31: [i for i, c in enumerate(ORIG_CFG["hybrid_override_pattern"]) if c == "M"]}
        for n, layers in exp.items():
            _, got, ords, direction = MV.make_pattern(ORIG_CFG["hybrid_override_pattern"], n)
            self.assertEqual(got, layers, n)
            self.assertEqual(direction, "M->*")
            m = 27
            self.assertEqual(ords, [math.floor((j + 0.5) * m / (n - 4)) for j in range(n - 4)])
        # negative control: an off-by-half rule picks different layers for a13
        m_pos = MV.mamba_positions(ORIG_CFG["hybrid_override_pattern"])
        alt = [m_pos[(j * 27) // 9] for j in range(9)]
        self.assertNotEqual(alt, exp[13])

    def test_identity_and_pure_ssm(self):
        orig = ORIG_CFG["hybrid_override_pattern"]
        p, layers, ords, d = MV.make_pattern(orig, 4)
        self.assertEqual((p, layers, ords, d), (orig, [], [], "none"))
        p0, l0, _, d0 = MV.make_pattern(orig, 0)
        self.assertEqual(p0, orig.replace("*", "M"))
        self.assertEqual((l0, d0), ([14, 21, 30, 39], "*->M"))
        with self.assertRaises(ValueError):
            MV.make_pattern(orig, 2)

    def test_deterministic_and_spread(self):
        for n_swap in range(1, 28):
            a, b = MV.swap_ordinals(27, n_swap), MV.swap_ordinals(27, n_swap)
            self.assertEqual(a, b)
            self.assertEqual(len(set(a)), n_swap)
            for j, o in enumerate(a):            # floor of each equal-width bin's midpoint: one pick per bin
                self.assertTrue(math.floor(j * 27 / n_swap) <= o < (j + 1) * 27 / n_swap, (n_swap, j, o))
        with self.assertRaises(ValueError):
            MV.swap_ordinals(27, 28)


class TestPredictions(unittest.TestCase):
    def test_original_values(self):
        pr = MV.predict(ORIG_CFG, ORIG_CFG["hybrid_override_pattern"])
        self.assertEqual(pr["conv_dim"], 12288)
        self.assertEqual(pr["state_bytes_per_req_s"], 143_548_416)
        self.assertLess(abs(pr["state_bytes_per_req_s"] / 143.5e6 - 1), 1e-3)     # boot log 143.5 MB
        self.assertEqual(pr["kv_bytes_per_token_k"], 16_384)
        self.assertLess(abs(pr["kv_bytes_per_token_k"] / 16_382 - 1), 1e-3)      # boot log 16,382 B
        self.assertEqual(pr["params_total"], 8_888_227_328)
        self.assertLess(abs(pr["weight_gib"] / 16.58 - 1), 0.005)                 # boot log mem usage=16.58 GB
        self.assertAlmostEqual(pr["L_star_tokens"], 143_548_416 / 16_384)
        self.assertEqual(sum(pr["params_by_type"].values()), pr["params_total"])

    def test_grid_monotone_and_extremes(self):
        prev = None
        for n in (4, 13, 22, 31):
            pr = MV.predict(ORIG_CFG, variant_pattern(n))
            self.assertEqual(pr["kv_bytes_per_token_k"], n * 4096)
            self.assertEqual(pr["state_bytes_per_req_s"], (31 - n) * 5_316_608)
            if prev:
                self.assertLess(pr["params_total"], prev["params_total"])          # attention layer < Mamba layer
            prev = pr
        self.assertEqual(MV.predict(ORIG_CFG, variant_pattern(31))["L_star_tokens"], 0.0)
        self.assertIsNone(MV.predict(ORIG_CFG, variant_pattern(0))["L_star_tokens"])

    def test_assumption_guards(self):
        bad = dict(ORIG_CFG, mamba_ssm_dtype="bfloat16")
        with self.assertRaises(ValueError):
            MV.predict(bad, bad["hybrid_override_pattern"])
        with self.assertRaises(ValueError):
            MV.predict(dict(ORIG_CFG, attention_bias=True), ORIG_CFG["hybrid_override_pattern"])

    def test_sglang_cache_params_agree(self):
        try:
            from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype
            from sglang.srt.configs.nemotron_h import NemotronHConfig
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"sglang config modules not importable: {exc!r}")
        for n in MV.N_ATTN_CHOICES:
            d = dict(ORIG_CFG, hybrid_override_pattern=variant_pattern(n))
            d.pop("auto_map")
            c = NemotronHConfig(**copy.deepcopy(d))
            self.assertEqual(c.hybrid_override_pattern, d["hybrid_override_pattern"])
            sh = Mamba2StateShape.create(tp_world_size=1, intermediate_size=c.mamba_num_heads * c.mamba_head_dim,
                                         n_groups=c.n_groups, num_heads=c.mamba_num_heads, head_dim=c.mamba_head_dim,
                                         state_size=c.ssm_state_size, conv_kernel=c.conv_kernel)
            cp = Mamba2CacheParams(shape=sh, layers=c.mamba_layer_ids, dtype=mamba2_state_dtype(c))
            pr = MV.predict(d, d["hybrid_override_pattern"])
            self.assertEqual(cp.mamba_cache_per_req, pr["state_bytes_per_req_s"], n)
            self.assertEqual(sh.conv_dim, pr["conv_dim"])
            self.assertEqual(len(c.full_attention_layer_ids), n)


class TestBannerCheck(unittest.TestCase):
    def test_real_banner_matches_original_prediction(self):
        f = CH.parse_boot_log(REAL_BANNER)
        self.assertEqual(f["weight_mem_usage_gib"], 16.58)
        self.assertEqual(f["cuda_graph_bs_lists"], [[1, 2, 4, 8, 12, 16, 24, 32, 40, 48]])
        self.assertEqual(f["pdmux_stream_groups"], 6)
        self.assertEqual(f["load_format_logged"], "auto")
        pr = MV.predict(ORIG_CFG, ORIG_CFG["hybrid_override_pattern"])
        out = CH.compare_prediction(pr, f, 8 * 128 * 2 * 2, pr["ssm_state_bytes_per_layer"])
        self.assertEqual(out["verdict"], CH.PASS, out)
        self.assertEqual(out["implied_layer_counts"], {"n_attn": 4, "n_mamba": 27})
        self.assertEqual(out["banner"]["mamba_slots"], 49)

    def test_wrong_variant_fails(self):
        f = CH.parse_boot_log(REAL_BANNER)
        pr = MV.predict(ORIG_CFG, variant_pattern(13))
        out = CH.compare_prediction(pr, f, 4096, pr["ssm_state_bytes_per_layer"])
        self.assertEqual(out["verdict"], CH.FAIL)
        self.assertEqual(out["checks"]["layer_counts"]["verdict"], CH.FAIL)

    def test_zero_state_variant(self):
        pr = MV.predict(ORIG_CFG, variant_pattern(31))
        banner = REAL_BANNER.replace("conv_state size: 0.09GB, ssm_state size: 6.46GB",
                                     "conv_state size: 0.00GB, ssm_state size: 0.00GB")
        banner = banner.replace("K size: 19.97 GB, V size: 19.97 GB", "K size: 154.79 GB, V size: 154.79 GB")
        banner = banner.replace("mem usage=16.58 GB", "mem usage=11.91 GB")
        out = CH.compare_prediction(pr, CH.parse_boot_log(banner), 4096, pr["ssm_state_bytes_per_layer"])
        self.assertEqual(out["checks"]["s"]["verdict"], CH.PASS)
        self.assertEqual(out["implied_layer_counts"], {"n_attn": 31, "n_mamba": 0})
        self.assertEqual(out["verdict"], CH.PASS, out)
        # control: a non-empty Mamba pool against a zero prediction fails
        out2 = CH.compare_prediction(pr, CH.parse_boot_log(REAL_BANNER), 4096, pr["ssm_state_bytes_per_layer"])
        self.assertEqual(out2["checks"]["s"]["verdict"], CH.FAIL)

    def test_boot_verdict(self):
        f = CH.parse_boot_log(REAL_BANNER.replace("context_len=8704", "context_len=73728"))
        sig = CH.signatures(REAL_BANNER)
        self.assertEqual(CH.boot_verdict(f, sig, True, False, True)[0], CH.PASS)
        self.assertEqual(CH.boot_verdict(f, sig, True, True, True)[0], CH.FAIL)       # wanted dummy, got auto
        self.assertEqual(CH.boot_verdict(f, sig, True, False, False)[0], CH.FAIL)     # PD-mux line on vanilla
        self.assertEqual(CH.boot_verdict(f, sig, False, False, True)[0], CH.FAIL)
        bad = dict(sig, cuda_graph_capture_failed=1)
        self.assertEqual(CH.boot_verdict(f, bad, True, False, True)[0], CH.FAIL)
        f2 = dict(f, cuda_graph_bs_lists=[])
        self.assertEqual(CH.boot_verdict(f2, sig, True, False, True)[0], CH.UNRES)

    def test_request_checks(self):
        ok = {"error": None, "prompt_tokens": 2048, "completion_tokens": 64, "finish_reason": {"type": "length"}}
        res = {"plain": [dict(ok) for _ in range(8)], "logprob": dict(ok, logprobs=[-1.0] * 64)}
        zero = {k: 0 for k in CH.signatures("")}
        self.assertEqual(CH.req_verdict(res, zero)[0], CH.PASS)
        self.assertEqual(CH.req_verdict(dict(res, logprob=dict(ok, logprobs=[-1.0] * 63 + [None])), zero)[0], CH.FAIL)
        self.assertEqual(CH.req_verdict(dict(res, logprob=dict(ok, logprobs=[float("nan")] + [-1.0] * 63)), zero)[0], CH.FAIL)
        self.assertEqual(CH.req_verdict(dict(res, logprob=dict(ok, error="HTTP 400", logprobs=None)), zero)[0], CH.UNRES)
        res2 = copy.deepcopy(res)
        res2["plain"][3]["completion_tokens"] = 12
        self.assertEqual(CH.req_verdict(res2, zero)[0], CH.FAIL)
        self.assertEqual(CH.req_verdict(res, dict(zero, nan_logits=1))[0], CH.FAIL)


class TestVariantJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="compo_variant_test_")
        self.src = os.path.join(self.tmp, "snap")
        os.makedirs(self.src)
        with open(os.path.join(self.src, "config.json"), "w") as f:
            json.dump(ORIG_CFG, f, indent=2)
        for n, body in (("tokenizer.json", "{}"), ("tokenizer_config.json", "{}"), ("generation_config.json", "{}"),
                        ("configuration_nemotron_h.py", "# stub\n"), ("model-00001-of-00004.safetensors", "w"),
                        ("model.safetensors.index.json", "{}")):
            with open(os.path.join(self.src, n), "w") as f:
                f.write(body)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_schema_and_files(self):
        out = os.path.join(self.tmp, "a13")
        rec = MV.build_variant(self.src, out, 13)
        on_disk = _rj(os.path.join(out, "variant.json"))
        self.assertEqual(on_disk, json.loads(json.dumps(rec, sort_keys=True)))
        for k in ("schema", "rule", "source", "n_attn", "n_mamba", "n_mlp", "pattern_orig", "pattern", "direction",
                  "changed_layers", "changed_mamba_ordinals", "config_sha256", "copied_files", "excluded_files",
                  "weights_copied", "load_format_required", "prediction", "note"):
            self.assertIn(k, on_disk)
        self.assertEqual(on_disk["schema"], "compo_variant/v1")
        self.assertEqual(on_disk["rule"]["name"], "spread_floor_mid_v1")
        self.assertEqual((on_disk["n_attn"], on_disk["n_mamba"], on_disk["n_mlp"]), (13, 18, 25))
        self.assertEqual(on_disk["changed_layers"], [2, 7, 13, 20, 27, 34, 41, 46, 52])
        src_sha = _sha(os.path.join(self.src, "config.json"))
        self.assertEqual(on_disk["source"]["config_sha256"], src_sha)
        self.assertEqual(on_disk["config_sha256"],
                         _sha(os.path.join(out, "config.json")))
        names = sorted(os.listdir(out))
        self.assertEqual(names, ["config.json", "configuration_nemotron_h.py", "generation_config.json",
                                 "tokenizer.json", "tokenizer_config.json", "variant.json"])
        self.assertEqual(sorted(e["name"] for e in on_disk["excluded_files"]),
                         ["model-00001-of-00004.safetensors", "model.safetensors.index.json"])
        new_cfg = _rj(os.path.join(out, "config.json"))
        diff = {k for k in set(new_cfg) | set(ORIG_CFG) if new_cfg.get(k) != ORIG_CFG.get(k)}
        self.assertEqual(diff, {"hybrid_override_pattern"})
        self.assertEqual(new_cfg["time_step_limit"][1], float("inf"))
        p = on_disk["prediction"]
        for k in ("state_bytes_per_req_s", "kv_bytes_per_token_k", "L_star_tokens", "params_by_type", "params_total",
                  "weight_bytes", "weight_gib"):
            self.assertIn(k, p)
        with self.assertRaises(FileExistsError):
            MV.build_variant(self.src, out, 13)

    def test_original_dummy_variant_is_identical(self):
        out = os.path.join(self.tmp, "a4")
        rec = MV.build_variant(self.src, out, 4)
        self.assertEqual(rec["pattern"], rec["pattern_orig"])
        self.assertEqual(_rj(os.path.join(out, "config.json")), ORIG_CFG)
        self.assertEqual(rec["prediction"]["state_bytes_per_req_s"], 143_548_416)

    def test_refuses_foreign_pattern(self):
        cfg = dict(ORIG_CFG, hybrid_override_pattern="M*" * 28)
        with open(os.path.join(self.src, "config.json"), "w") as f:
            json.dump(cfg, f)
        with self.assertRaises(ValueError):
            MV.build_variant(self.src, os.path.join(self.tmp, "x"), 13)

    def test_cli_print(self):
        r = subprocess.run(["python3", os.path.join(COMPO, "make_variant.py"), "--src", self.src, "--print"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = [json.loads(x) for x in r.stdout.splitlines()]
        self.assertEqual([x["n_attn"] for x in rows], [0, 4, 13, 22, 31])


class TestPlanAndScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = subprocess.run(["/usr/bin/bash", os.path.join(VESSL, "compo_probe.sh"), "--print-plan"],
                             capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        cls.lines = [ln.split("\t") for ln in out.stdout.splitlines()]
        cls.argv = {ln[1]: ln[3].split() for ln in cls.lines if ln[0] in ("ARGV", "ARGV_RETRY")}
        cls.mode = {ln[1]: ln[2] for ln in cls.lines if ln[0] in ("ARGV", "ARGV_RETRY")}

    def test_boot_set(self):
        self.assertEqual([ln[1] for ln in self.lines if ln[0] == "ARGV"],
                         ["real_orig", "dummy_a4", "dummy_a13", "dummy_a22", "dummy_a31", "dummy_a4_vanilla"])
        self.assertEqual([ln[1] for ln in self.lines if ln[0] == "ARGV_RETRY"], ["dummy_a31_retry"])

    def test_dummy_flag_only_on_dummy_boots(self):
        for b, a in self.argv.items():
            has = any(a[i] == "--load-format" and a[i + 1] == "dummy" for i in range(len(a) - 1))
            self.assertEqual(has, b.startswith("dummy_"), b)
            self.assertEqual(a.count("--load-format"), 1 if b.startswith("dummy_") else 0, b)
        ro = self.argv["real_orig"]
        self.assertEqual(ro[ro.index("--model-path") + 1], "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base")
        self.assertEqual(ro[ro.index("--revision") + 1], "dc0661c829b14e5b9246c05cfa89094a0875e052")
        for b, var in (("dummy_a4", "a4"), ("dummy_a13", "a13"), ("dummy_a22", "a22"), ("dummy_a31", "a31"),
                       ("dummy_a31_retry", "a31"), ("dummy_a4_vanilla", "a4")):
            self.assertTrue(self.argv[b][self.argv[b].index("--model-path") + 1].endswith("/variants/" + var), b)
            self.assertNotIn("--revision", self.argv[b])

    def test_fixed_factors(self):
        for b, a in self.argv.items():
            self.assertEqual(a[a.index("--context-length") + 1], "73728")
            self.assertEqual(a[a.index("--mem-fraction-static") + 1], "0.6")
            self.assertEqual(a[a.index("--max-running-requests") + 1], "128")
            self.assertEqual(a[a.index("--chunked-prefill-size") + 1], "-1")
            for flag in ("--disable-radix-cache", "--disable-overlap-schedule", "--disable-piecewise-cuda-graph",
                         "--trust-remote-code"):
                self.assertIn(flag, a, b)
            self.assertNotIn("--disable-cuda-graph", a)                                   # cudagraph ON
            if b == "dummy_a31_retry":
                self.assertNotIn("--max-mamba-cache-size", a)
            else:
                self.assertEqual(a[a.index("--max-mamba-cache-size") + 1], "128")
            pd = "--enable-pdmux" in a
            self.assertEqual(pd, b != "dummy_a4_vanilla", b)
            self.assertEqual(self.mode[b], "engine_mode=" + ("vanilla" if b == "dummy_a4_vanilla" else "pdmux"))
            if pd:
                self.assertTrue(a[a.index("--pdmux-config-path") + 1].endswith("benchmarks/configs/pdmux_agnostic_manual.yml"))
        # the retry differs from dummy_a31 by exactly the mamba-cap flag pair
        a31, rt = self.argv["dummy_a31"], self.argv["dummy_a31_retry"]
        i = a31.index("--max-mamba-cache-size")
        self.assertEqual(a31[:i] + a31[i + 2:], rt)

    def test_time_table(self):
        budget = int(next(ln[1] for ln in self.lines if ln[0] == "BUDGET_MIN"))
        self.assertEqual(budget, 90)
        tot = {ln[1]: (int(ln[2]), int(ln[3])) for ln in self.lines if ln[0] == "EXPECT_TOTAL_MIN"}
        self.assertLessEqual(tot["all_boots_with_a31_retry"][1], budget)
        exp = {ln[1]: (int(ln[2]), int(ln[3])) for ln in self.lines if ln[0] == "EXPECT_MIN"}
        self.assertEqual(sum(v[1] for v in exp.values()), tot["all_boots_with_a31_retry"][1])
        tmo = {ln[1]: int(ln[2]) for ln in self.lines if ln[0] == "TIMEOUT_S"}
        self.assertGreaterEqual(tmo["boot"], 10 * 60)
        self.assertGreaterEqual(tmo["REQ"], 2 * 60)

    def test_syntax_and_no_env_dump(self):
        r = subprocess.run(["/usr/bin/bash", "-n", os.path.join(VESSL, "compo_probe.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        sh = shutil.which("shellcheck")
        if sh:
            r = subprocess.run([sh, "-S", "error", os.path.join(VESSL, "compo_probe.sh")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout)
        for p in (os.path.join(VESSL, "compo_probe.sh"), os.path.join(VESSL, "compo_probe_helper.py"),
                  os.path.join(COMPO, "make_variant.py")):
            src = _txt(p)
            self.assertIsNone(re.search(r"^\s*set -[a-z]*x", src, re.M), p)
            self.assertNotIn("printenv", src)
            self.assertIsNone(re.search(r"^\s*(env|export -p)\s*(>|\|)", src, re.M), p)
        src = _txt(os.path.join(VESSL, "compo_probe.sh"))
        self.assertIn("unset SGLANG_MAMBA_SSM_DTYPE SGLANG_MAMBA_CONV_DTYPE", src)     # predictions assume defaults
        self.assertIn("PDMUX_TELEMETRY_PATH", src)                                     # telemetry unset (OFF)

    def test_no_verdict_words(self):
        src = _txt(os.path.join(VESSL, "compo_probe_helper.py"))
        for w in ("faster than", "slower than", "outperform", "speedup", "wins"):
            self.assertNotIn(w, src)


if __name__ == "__main__":
    unittest.main()
