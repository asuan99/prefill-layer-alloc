"""CPU regression for the motivation workload builders and trace generator (no datasets needed)."""
import json
import os
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
WL = HERE.parent / "results" / "motivation" / "workloads"
sys.path.insert(0, str(WL))
import mw_lib as L  # noqa: E402
import gen_traces as G  # noqa: E402
import ph_trace as PH  # noqa: E402


class MasterAndRung(unittest.TestCase):
    def setUp(self):
        self.pool = [[None, 10 + i, 20 + i] for i in range(1000)]

    def test_master_deterministic_and_seed_sensitive(self):
        a = G.master("sharegpt_first", 1, self.pool)
        self.assertEqual(a, G.master("sharegpt_first", 1, self.pool))
        self.assertNotEqual(a, G.master("sharegpt_first", 2, self.pool))

    def test_rung_prefix_scales_with_rate(self):
        high = G.instantiate("sharegpt_first", 1, self.pool, 10.0, duration=100.0)[0]["requests"]
        low = G.instantiate("sharegpt_first", 1, self.pool, 2.0, duration=100.0)[0]["requests"]
        self.assertGreater(len(high), len(low))
        n = len(low)
        for a, b in zip(low, high[:n]):
            self.assertEqual(a["input_tokens"], b["input_tokens"])
            self.assertAlmostEqual(a["arrival_s"] * 2.0, b["arrival_s"] * 10.0, delta=1e-5)

    def test_rung_window_and_guard(self):
        reqs = G.instantiate("sharegpt_first", 1, self.pool, 2.0, duration=50.0)[0]["requests"]
        self.assertTrue(all(r["arrival_s"] < 50.0 for r in reqs))
        with self.assertRaises(ValueError):
            G.instantiate("sharegpt_first", 1, self.pool, 0.1, n=10 ** 6)


class Interference(unittest.TestCase):
    def setUp(self):
        self.short = [[None, 10 + (i % 50), 20] for i in range(5000)]
        self.long = [[i, 30000 + i, 8] for i in range(200)]

    def test_short_stream_identical_with_and_without(self):
        w, sa = G.interference(7, self.short, self.long, 3.0, 300.0, True)
        wo, sb = G.interference(7, self.short, self.long, 3.0, 300.0, False)
        self.assertEqual(sa, sb)
        self.assertEqual(G.short_hash(sa), G.short_hash(sb))
        long_recs = [r for r in w[0]["requests"] if r["request_class"] == "long"]
        self.assertEqual(len(long_recs), int(round(0.05 / 0.95 * 3.0 * 300.0)))
        self.assertTrue(all(0 <= r["arrival_s"] < 300.0 for r in long_recs))
        self.assertEqual(len(wo[0]["requests"]), len(sa))
        self.assertEqual(sum(r["request_class"] == "short" for r in w[0]["requests"]), len(sa))
        arrivals = [r["arrival_s"] for r in w[0]["requests"]]
        self.assertEqual(arrivals, sorted(arrivals))

    def test_long_count_is_fixed_across_seeds(self):
        counts = {len([r for r in G.interference(s, self.short, self.long, 3.0, 300.0, True)[0][0]["requests"] if r["request_class"] == "long"]) for s in range(5)}
        self.assertEqual(len(counts), 1)

    def test_trace_hash_changes_with_long_stream(self):
        w, _ = G.interference(7, self.short, self.long, 3.0, 300.0, True)
        wo, _ = G.interference(7, self.short, self.long, 3.0, 300.0, False)
        self.assertNotEqual(PH.trace_hash(w), PH.trace_hash(wo))


class Outputs(unittest.TestCase):
    def test_arxiv_main_condition_is_the_abstract_length(self):
        r = G.request("arxiv_summ", [12, 800, 200], 1.0, 0, "x", "arxiv_summ")
        self.assertIs(r["ignore_eos"], True)
        self.assertEqual(r["output_tokens"], 200)
        self.assertEqual(r["doc_id"], 12)

    def test_arxiv_variant_changes_only_the_output(self):
        seg = G.segment("t", "arxiv_summ", [G.request("arxiv_summ", [12, 800, 200], 1.0, 0, "x", "arxiv_summ")])
        v = G.fix_output([seg], G.ARXIV_VARIANT_OUTPUT)[0]
        r, o = v["requests"][0], seg["requests"][0]
        self.assertEqual((r["output_tokens"], r["ignore_eos"]), (1024, True))
        self.assertEqual({k: x for k, x in r.items() if k != "output_tokens"}, {k: x for k, x in o.items() if k != "output_tokens"})
        self.assertEqual(o["output_tokens"], 200)

    def test_loogle_fixed_output_record(self):
        r = G.request("loogle_longdep", [3, 40000, 7], 1.0, 0, "x", "LONG")
        self.assertIs(r["ignore_eos"], True)
        self.assertEqual(r["output_tokens"], 7)


class TokenFile(unittest.TestCase):
    def test_roundtrip_with_shared_segments(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.bin")
            payload = array("i", list(range(100, 130)))
            records = [[0, [[0, 10], [20, 5]], 15, 3], [1, [[0, 10]], 10, 2]]
            h = L.write_token_file(p, payload, records, {"workload": "x", "tokenizer_sha256": "y"})
            h2 = L.read_token_header(p)
            self.assertEqual(h2["records"], h["records"])
            self.assertEqual(L.record_input_ids(p, h2, 0), list(range(100, 110)) + list(range(120, 125)))
            self.assertEqual(L.record_input_ids(p, h2, 1), list(range(100, 110)))


class Registry(unittest.TestCase):
    def test_registry_consistent_when_present(self):
        p = WL / "trace_registry.json"
        if not p.exists():
            self.skipTest("registry not sealed")
        reg = json.load(open(p))
        new = G.seal_data()
        self.assertEqual(reg["master_hashes"], new["master_hashes"])
        self.assertEqual(reg["provisional_interference_pairs"], new["provisional_interference_pairs"])


if __name__ == "__main__":
    unittest.main()
