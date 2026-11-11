"""mo_clock: GPU clock-limit covariate per run / arm (CPU only)."""
import csv
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
MO = ENG / "results" / "motivation"
sys.path.insert(0, str(MO))
sys.path.insert(0, str(ENG / "results" / "stock_baseline"))
import mo_clock as CK  # noqa: E402
import mo_analyze as MA  # noqa: E402

HDR = "timestamp, clocks.current.sm [MHz], power.draw [W], temperature.gpu, clocks_event_reasons.active, index"
T0 = datetime(2026, 11, 11, 4, 0, 0, tzinfo=timezone.utc).timestamp()


def write_csv(path, rows):
    """rows: (offset_s, mhz, watts, reasons_int)"""
    with open(path, "w") as f:
        f.write(HDR + "\n")
        for off, mhz, w, rs in rows:
            ts = datetime.fromtimestamp(T0 + off, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            f.write(f"{ts}, {mhz} MHz, {w} W, 40, 0x{rs:016x}, 0\n")


class TestClock(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.p = Path(self.td.name) / "clk.csv"
        rows = [(i * 0.1, 1410, 300, 0x0) for i in range(10)]                  # arm A: 0..0.9 s, never limited
        rows += [(1.0 + i * 0.1, 1200, 400, 0x4 if i < 5 else 0x1) for i in range(10)]   # arm B: half sw power cap, half idle-only
        write_csv(self.p, rows)
        self.runs = [{"run_id": "r1", "arm": "A", "t_load_start_epoch": T0, "t_load_end_epoch": T0 + 0.95},
                     {"run_id": "r2", "arm": "B", "t_load_start_epoch": T0 + 1.0, "t_load_end_epoch": T0 + 1.95}]

    def tearDown(self):
        self.td.cleanup()

    def test_utc_parse_and_idle_bit_not_limited(self):
        s, bad = CK.read_clk(self.p)
        self.assertEqual((len(s), bad), (20, 0))
        self.assertAlmostEqual(s[0]["t"], T0, places=3)
        self.assertFalse(any(x["limited"] for x in s if x["reasons"] == 0x1))

    def test_per_arm_windows(self):
        s, _ = CK.read_clk(self.p)
        rep = CK.report(self.runs, s)
        self.assertEqual(rep["per_arm"]["A"]["limited_frac"], 0.0)
        self.assertEqual(rep["per_arm"]["B"]["limited_frac"], 0.5)
        self.assertEqual(rep["per_arm"]["B"]["reasons_seen"], ["sw_power_cap"])
        self.assertEqual(rep["per_arm"]["A"]["sm_mhz_p50"], 1410)
        self.assertAlmostEqual(rep["spread"]["limited_frac_range"], 0.5)

    def test_pooling_weights_samples_not_runs(self):
        s, _ = CK.read_clk(self.p)
        runs = self.runs + [{"run_id": "r3", "arm": "B", "t_load_start_epoch": T0, "t_load_end_epoch": T0 + 0.25}]
        rep = CK.report(runs, s)
        self.assertAlmostEqual(rep["per_arm"]["B"]["limited_frac"], 5 / 13)

    def test_empty_window(self):
        s, _ = CK.read_clk(self.p)
        row = CK.summarize(CK.window(s, T0 + 100, T0 + 101))
        self.assertEqual(row["n_samples"], 0)
        self.assertIsNone(row["limited_frac"])

    def test_analyzer_hook(self):
        job = {"runs": self.runs, "raw": None}
        self.assertFalse(MA.clock_report(job)["available"])
        ck = MA.clock_report(job, self.p)
        self.assertTrue(ck["available"])
        self.assertEqual(ck["per_arm"]["B"]["n_limited"], 5)

    def test_find_clk_walks_up(self):
        root = Path(self.td.name) / "run"
        raw = root / "results" / "motivation_runs" / "job1"
        raw.mkdir(parents=True)
        (root / "meta").mkdir()
        (root / "meta" / "clk.csv").write_text(HDR + "\n")
        self.assertEqual(MA.find_clk(raw), (root / "meta" / "clk.csv").resolve())


if __name__ == "__main__":
    unittest.main()
