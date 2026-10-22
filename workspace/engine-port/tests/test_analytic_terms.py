"""Fixed analytic cost formulas (decision 27): the code must reproduce the document's numbers and agree with
independent sources (config, compo make_variant predictions, boot-banner measurements)."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "costmodel"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "compo"))
import analytic_terms as t  # noqa: E402

CFG = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/"
                         "dc0661c829b14e5b9246c05cfa89094a0875e052/config.json")


class TestNanoValues(unittest.TestCase):
    def setUp(self):
        self.a = t.nano9b_v2()

    def test_per_layer_params(self):
        self.assertEqual(t.params_attn(self.a), 55_050_240)
        self.assertEqual(t.params_mamba_matmul(self.a), 147_374_080)
        self.assertEqual(t.params_mlp(self.a), 140_492_800)

    def test_total_params_matches_independent_count(self):
        # compo make_variant.py (independent derivation) and boot banner 16.58 GB
        self.assertEqual(t.params_total(self.a), 8_888_227_328)
        self.assertAlmostEqual(t.params_total(self.a) * 2 / 2**30, 16.58, delta=0.05)   # banner "GB" is GiB

    def test_resident_memory_matches_banner(self):
        self.assertEqual(t.state_bytes_per_request(self.a), 143_548_416)      # banner 143.5 MB
        self.assertEqual(t.kv_bytes_per_token(self.a), 16_384)                # banner 16,382 B (GiB rounding)
        self.assertAlmostEqual(t.L_star(self.a), 8761.5, delta=1)

    def test_flop_terms(self):
        self.assertEqual(t.flops_ssd_prefill_per_token(self.a), 8_126_464)
        self.assertEqual(t.flops_ssm_decode_per_token(self.a), 6_553_600)
        pf = t.prefill_flops(self.a, 65536)
        self.assertEqual(pf["quad_coeff"], 40_960)
        self.assertAlmostEqual(pf["linear_per_token"] / 1e9, 15.645, delta=0.001)
        self.assertAlmostEqual(pf["L_quad_eq_linear"], 381_966, delta=50)

    def test_decode_traffic(self):
        d = t.decode_step(self.a, 32, 8192)
        self.assertEqual(d["bytes_kv_read"], 32 * 8192 * 16384)
        self.assertAlmostEqual(d["c_state_eq_kv"], 17_523, delta=1)
        self.assertLess(d["intensity_flop_per_byte"], 30)   # memory-bound on A100 (ridge ~150)

    def test_attention_scaling_is_proportional(self):
        # registered prediction P3 rests on k (and the decode ctx coefficient) being exactly ∝ n_A
        k = [t.kv_bytes_per_token(t.nano9b_v2(n, 31 - n)) for n in (4, 13, 22, 31)]
        self.assertEqual([x // 4096 for x in k], [4, 13, 22, 31])


class TestCrossSources(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(CFG), "Nano config not in local HF cache")
    def test_config_parse_equals_hardcoded(self):
        self.assertEqual(t.arch_from_config(json.load(open(CFG))), t.nano9b_v2())

    @unittest.skipUnless(os.path.exists(CFG), "Nano config not in local HF cache")
    def test_agrees_with_make_variant(self):
        import make_variant as mv
        cfg = json.load(open(CFG))
        for n_attn in (4, 13, 22, 31):
            pat = mv.make_pattern(cfg["hybrid_override_pattern"], n_attn)[0]
            pr = mv.predict(cfg, pat)
            a = t.nano9b_v2(n_attn, 31 - n_attn)
            self.assertEqual(pr["params_total"], t.params_total(a))
            self.assertEqual(round(pr["L_star_tokens"]) if pr["L_star_tokens"] else 0, round(t.L_star(a)))


if __name__ == "__main__":
    unittest.main()
