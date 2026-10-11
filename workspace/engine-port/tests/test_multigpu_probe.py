"""CPU tests for scripts/vessl/multigpu_probe_helper.py pure parsers (DRAFT, 2026-10-12)."""
import importlib.util
import os
import unittest

P = os.path.join(os.path.dirname(__file__), "..", "scripts", "vessl", "multigpu_probe_helper.py")
spec = importlib.util.spec_from_file_location("multigpu_probe_helper", P)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

TOPO4 = ("\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\n"
         "GPU0\t X \tNV12\tNV12\tSYS\t0-21\t0\n"
         "GPU1\tNV12\t X \tNV12\tNV12\t0-21\t0\n"
         "GPU2\tNV12\tNV12\t X \tNV12\t0-21\t0\n"
         "GPU3\tSYS\tNV12\tNV12\t X \t0-21\t0\n\nLegend:\n  X = Self\n")
NCCL = """#       size   count  type redop root   time   algbw   busbw #wrong   time   algbw   busbw #wrong
        1024     256 float   sum   -1    20.1    0.05    0.09      0    19.8    0.05    0.09      0
     1048576  262144 float   sum   -1    40.0   26.21   45.87      0    39.0   26.9    47.0      0
# Avg bus bandwidth    : 23.4
"""


class T(unittest.TestCase):
    def test_link_class(self):
        self.assertEqual(H.link_class("NV12"), "NVLINK(12)")
        self.assertEqual(H.link_class("PIX"), "PIX")
        self.assertEqual(H.link_class("junk"), "OTHER")

    def test_parse_topo(self):
        r = H.parse_topo(TOPO4)
        self.assertEqual(r["n_gpu"], 4)
        self.assertEqual(r["links"][0][3], "SYS")
        self.assertFalse(r["all_pairs_nvlink"])
        self.assertEqual(r["classes"]["SYS"], 2)

    def test_parse_topo_ansi_header(self):
        # real VESSL a100x8 output (2026-10-12): header underlined with ESC[4m ... ESC[0m
        t = "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\x1b[0m\nGPU0\t X \tNV12\t0-23\t0\nGPU1\tNV12\t X \t0-23\t0\n"
        r = H.parse_topo(t)
        self.assertEqual(r["n_gpu"], 2)
        self.assertTrue(r["all_pairs_nvlink"])
        self.assertEqual(r["classes"], {"NVLINK(12)": 2})

    def test_parse_topo_single_column_not_vacuous(self):
        self.assertFalse(H.parse_topo("\tGPU0\tCPU Affinity\nGPU0\t X \t0-23\n")["all_pairs_nvlink"])

    def test_parse_topo_all_nvlink_and_garbage(self):
        t = "\tGPU0\tGPU1\nGPU0\t X \tNV8\nGPU1\tNV8\t X \n"
        self.assertTrue(H.parse_topo(t)["all_pairs_nvlink"])
        self.assertEqual(H.parse_topo("nothing")["n_gpu"], 0)

    def test_parse_nccl(self):
        r = H.parse_nccl_tests(NCCL)
        self.assertEqual(len(r["rows"]), 2)
        self.assertEqual(r["avg_busbw"], 23.4)
        self.assertEqual(r["wrong_total"], 0)

    def test_busbw(self):
        alg, bus = H.busbw_allreduce(10**9, 1.0, 8)
        self.assertAlmostEqual(alg, 1.0)
        self.assertAlmostEqual(bus, 1.75)


if __name__ == "__main__":
    unittest.main()
