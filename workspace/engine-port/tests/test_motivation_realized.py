"""mo_realized: realized decode SM count from split-check telemetry + green read-out (CPU only)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENG / "results" / "motivation"))
import mo_realized as R  # noqa: E402


def ev(t, co, d, idx):
    return {"event": "decode_iteration", "timestamp_monotonic_s": t, "t_issue_start": t - 0.01, "t_sync": t, "prefill_in_flight": co,
            "decode_sms": d, "stream_idx": idx}


GREEN = {"groups": [{"stream_index": 4, "decode": {"green_sm": {"smCount": 44}}, "prefill": {"green_sm": {"smCount": 64}}},
                    {"stream_index": 2, "decode": {"green_sm": {"smCount": 16}}, "prefill": {"green_sm": {"smCount": 92}}}]}


class TestRealized(unittest.TestCase):
    def test_window_and_states(self):
        evs = [ev(1.0, True, 44, 4), ev(2.0, False, 108, 6), ev(3.0, True, 44, 4), ev(9.0, True, 16, 2)]
        d = R.dwell(evs, 0.5, 5.0, 44)
        self.assertEqual((d["n_steps"], d["n_coresident"], d["coresident_share_on_target"]), (3, 2, 1.0))
        self.assertEqual(d["coresident_stream_idx"], [4])
        d = R.dwell(evs, 0.5, 10.0, 44)
        self.assertAlmostEqual(d["coresident_share_on_target"], 2 / 3)

    def test_green_tolerance(self):
        rows, ok, why = R.green_groups(GREEN, [4], 44)
        self.assertTrue(ok)
        rows, ok, why = R.green_groups(GREEN, [2], 44)          # the "D108 was 16 SM" shape
        self.assertFalse(ok)
        rows, ok, why = R.green_groups(GREEN, [5], 44)
        self.assertFalse(ok)
        self.assertIn("absent", why)
        rows, ok, why = R.green_groups({"error": "driver"}, [4], 44)
        self.assertFalse(ok)

    def test_realized_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td)
            (b / "tel.jsonl").write_text("\n".join(json.dumps(e) for e in [ev(1.0, True, 44, 4), ev(2.0, False, 108, 6)]) + "\n")
            (b / "green.json").write_text(json.dumps(GREEN))
            run = {"t_load_start_mono": 0.0, "t_load_end_mono": 5.0}
            r = R.realized(b, run, 44)
            self.assertTrue(r["available"] and r["matches_target"])
            r = R.realized(b, {"t_load_start_mono": 3.0, "t_load_end_mono": 5.0}, 44)
            self.assertIsNone(r["matches_target"])
            (b / "tel.jsonl").unlink()
            self.assertFalse(R.realized(b, run, 44)["available"])

    def test_env_paths_live_in_the_boot_dir(self):
        for k in ("PDMUX_TELEMETRY_PATH", "PDMUX_GREEN_READOUT_PATH"):
            self.assertTrue(R.SPLIT_CHECK_ENV[k].startswith("{dir}/"))


if __name__ == "__main__":
    unittest.main()
