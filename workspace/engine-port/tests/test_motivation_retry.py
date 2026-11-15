"""A long Job is not lost to one transient fault (2026-11-14, after pre-ladder run 3 lost a boot to one dropped warm-up connection and
rehearsals 4 / 5 each stopped at their first card mismatch).

  - warm-up: bad requests that are all client-side dropped connections (no token received) -> the warm-up is replayed once;
    a wrong length is final.
  - boot: a boot that ended before its first run / cell is redone once on a fresh server; the failed attempt is kept.
  - rehearsal: a boot that differs from the card is recorded and the pass goes on; the record names every differing boot.
    A measurement Job still ends at the first mismatch.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_motivation_cap as TC  # noqa: E402
import test_motivation_harness as TH  # noqa: E402

MR, MH, MC, POOL = TH.MR, TH.MH, TH.MC, TH.POOL


def flaky_replay(drop_calls=(0,), kind="drop"):
    """fake_replay whose warm-up call number k in drop_calls breaks one request: a dropped connection or a short output."""
    calls = {"warm": 0}

    def rp(api, base_url, reqs, out_path, perf=None):
        dur, recs = TH.fake_replay(api, base_url, reqs, out_path, perf=perf)
        if recs and recs[0]["phase"] == "WARM":
            k = calls["warm"]
            calls["warm"] += 1
            if k in drop_calls:
                if kind == "drop":
                    recs[1].update(error="ServerDisconnectedError('Server disconnected')", error_kind="CLIENT_EXCEPTION", observed_output_tokens=0)
                else:
                    recs[1].update(observed_output_tokens=recs[1]["requested_output_tokens"] - 1)
        return dur, recs
    rp.calls = calls
    return rp


def run(plan, td, server=None, replay=TH.fake_replay, **kw):
    card = MC.build_card(plan, POOL)
    return MR.run_job(plan, card, Path(td) / "raw", server or TH.make_server(), replay, POOL, gate_fn=lambda p: None, py_version_fn=TH.fake_py,
                      freeze_fn=TH.fake_freeze, warmup_s=5.0, **kw)


class TestWarmupRetry(unittest.TestCase):
    def test_one_dropped_connection_replays_the_warmup_and_the_boot_is_measured(self):
        plan = MH.rehearsal_plan("JW1")
        rp = flaky_replay(drop_calls=(0,))
        with tempfile.TemporaryDirectory() as td:
            res = run(plan, td, replay=rp)
            self.assertTrue(all(b["status"] == "done" and "failed_attempts" not in b for b in res))
            first = json.loads((Path(td) / "raw" / "boots" / plan["boots"][0]["boot_id"] / "correctness.json").read_text())
            self.assertEqual([a["n_bad"] for a in first["attempts"]], [1, 0])
            self.assertTrue(first["attempts"][0]["client_connection_only"])
            self.assertEqual(rp.calls["warm"], len(plan["boots"]) + 1)

    def test_a_wrong_length_is_not_replayed_and_the_boot_is_redone_once(self):
        plan = MH.rehearsal_plan("JW2")
        rp = flaky_replay(drop_calls=(0,), kind="short")
        with tempfile.TemporaryDirectory() as td:
            res = run(plan, td, replay=rp)
            self.assertEqual(res[0]["status"], "done")
            self.assertEqual(res[0]["failed_attempts"], ["INVALID_CORRECTNESS"])
            kept = Path(td) / "raw" / "failed_boots" / f"{plan['boots'][0]['boot_id']}.attempt1"
            self.assertEqual(len(json.loads((kept / "correctness.json").read_text())["attempts"]), 1)       # no replay of a wrong length
            self.assertEqual(json.loads((kept / "DONE.json").read_text())["status"], "INVALID_CORRECTNESS")

    def test_a_fault_that_repeats_stays_a_failed_boot_and_the_job_goes_on(self):
        plan = MH.rehearsal_plan("JW3")
        rp = flaky_replay(drop_calls=(0, 1, 2, 3))            # both warm-up passes of both attempts of the first boot
        with tempfile.TemporaryDirectory() as td:
            res = run(plan, td, replay=rp)
            self.assertEqual(res[0]["status"], "INVALID_CORRECTNESS")
            self.assertEqual(res[0]["failed_attempts"], ["INVALID_CORRECTNESS"])
            self.assertTrue(all(b["status"] == "done" for b in res[1:]))
            self.assertEqual(MR.exit_code(res), MR.EXIT_BOOT_FAIL)


class TestBootRedo(unittest.TestCase):
    def test_a_start_failure_is_redone_once_in_the_same_place_of_the_boot_order(self):
        plan = MH.rehearsal_plan("JB1")
        target, n = plan["boots"][1], {"k": 0}

        def server(bdir, a):
            if a["name"] == target["arm"]:
                n["k"] += 1
                return TH.FakeServer(bdir, a, {"status": "BOOT_FAILED"} if n["k"] == 1 else None)
            return TH.FakeServer(bdir, a, None)
        with tempfile.TemporaryDirectory() as td:
            res = run(plan, td, server=server)
            self.assertTrue(all(b["status"] == "done" for b in res))             # the order check of every later boot passed
            self.assertEqual(res[1]["failed_attempts"], ["BOOT_FAILED"])
            self.assertEqual(n["k"], 2)
            self.assertTrue((Path(td) / "raw" / "failed_boots" / f"{target['boot_id']}.attempt1" / "boot.json").exists())
            self.assertEqual(json.loads((Path(td) / "raw" / "boots" / target["boot_id"] / "boot.json").read_text())["index"], 1)

    def test_fixed_batch_boot_is_redone_and_cells_are_never_resent(self):
        plan = TC.small_fixed_plan()
        target, n = plan["boots"][1], {"k": 0}

        def server(bdir, a):
            t = None
            if Path(bdir).name == target["boot_id"]:
                n["k"] += 1
                t = {"status": "BOOT_FAILED"} if n["k"] == 1 else None
            return TC.FakeFixedServer(bdir, a, t, None, None)
        with tempfile.TemporaryDirectory() as td:
            res = TC.CR.run_fixed_job(plan, TC.CR.fixed_card(plan), Path(td) / "raw", server, gate_fn=lambda p: None, py_version_fn=TH.fake_py,
                                      freeze_fn=TH.fake_freeze, variant_fn=TC.variant_fn)
            self.assertTrue(all(b["status"] == "done" for b in res), res)
            self.assertEqual(res[1]["failed_attempts"], ["BOOT_FAILED"])
        self.assertNotIn("SERVER_DEATH", MR.NO_RETRY)          # ... but a boot that already ran a cell is not redone:
        r = MR.boot_with_retry(lambda: {"boot_id": "x", "status": "SERVER_DEATH", "cells": ["c1"]}, {"boot_id": "x"}, td, {"executed": ["a"]})
        self.assertNotIn("failed_attempts", r)


class TestRehearsalNamesEveryMismatch(unittest.TestCase):
    def test_two_differing_boots_are_both_in_the_record(self):
        plan = MH.rehearsal_plan("JM1")
        arms = [b["arm"] for b in plan["boots"] if b["role"] == "meas"][:2]
        tam = {arms[0]: {"banner": {"mem_fraction_static": 0.9}}, arms[1]: {"banner": {"max_running_requests": 999}}}
        with tempfile.TemporaryDirectory() as td:
            rec = MH.rehearse("JM1", POOL, Path(td) / "raw", TH.make_server(tam), TH.fake_replay, rehearsal_dir=Path(td) / "rec", digest="c" * 64,
                              py_version_fn=TH.fake_py, freeze_fn=TH.fake_freeze)
            self.assertFalse(rec["passed"])
            self.assertIn("[mem]", rec["reason"])
            self.assertIn("[cap]", rec["reason"])
            self.assertGreaterEqual(rec["reason"].count("=CARD_MISMATCH"), 2)
            done = [json.loads(p.read_text())["status"] for p in sorted((Path(td) / "raw" / "boots").glob("*/DONE.json"))]
            self.assertEqual(len(done), len(plan["boots"]))                      # the pass went through every boot
            self.assertEqual(sorted(set(done)), ["CARD_MISMATCH", "done"])

    def test_a_job_that_is_not_a_rehearsal_pass_still_ends_at_the_first_mismatch(self):
        plan = MH.rehearsal_plan("JM2")
        arm = next(b["arm"] for b in plan["boots"] if b["role"] == "meas")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(MR.JobAbort) as cm:
                run(plan, td, server=TH.make_server({arm: {"banner": {"mem_fraction_static": 0.9}}}))
            self.assertEqual(cm.exception.status, "ABORT_CARD_MISMATCH")


class TestRehearsalInsideTheMeasurementJob(unittest.TestCase):
    """Static only: the in-Job rehearsal path has not run on a GPU or under a fake engine."""

    def test_each_job_script_rehearses_before_the_card_when_asked_and_stops_if_it_did_not_pass(self):
        mo = Path(TH.MR.__file__).parent
        for f, pfx, runner in (("mo_cap_job.sh", "PDMUX_MOCAP", "mo_cap_runner.py"), ("mo_intf_job.sh", "PDMUX_MOINTF", "mo_intf_runner.py"),
                               ("mo_job.sh", "PDMUX_MO", "mo_runner.py")):
            txt = (mo / f).read_text()
            i, j = txt.index(f'"${{{pfx}_REHEARSE_FIRST:-}}" == 1'), txt.index(f'"${{here}}/{runner}" card')
            self.assertLess(i, j, f)
            blk = txt[i:j]
            self.assertIn(f'{runner}" rehearse "${{common[@]}}" --out "${{out}}/rehearsal_raw"', blk, f)
            self.assertIn("exit 19", blk, f)


class TestCapJobScriptSealGuards(unittest.TestCase):
    def test_digest_guard_rehearsal_budget_and_clock_are_wired(self):
        sh = (Path(__file__).resolve().parent.parent / "results" / "motivation" / "mo_cap_job.sh").read_text()
        self.assertIn("PDMUX_MOCAP_EXPECT_HARNESS_DIGEST", sh)
        self.assertIn("ABORT_HARNESS_DIGEST", sh)
        self.assertLess(sh.index("ABORT_HARNESS_DIGEST"), sh.index("mo_cap_runner.py\" rehearse"))      # before any boot
        self.assertIn('PDMUX_MOCAP_REHEARSE_BUDGET_MIN:-60', sh)
        self.assertIn('[[ -f "${clk}" ]] && an+=(--clk "${clk}")', sh)


if __name__ == "__main__":
    unittest.main()
