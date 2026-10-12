"""mg_realized_mix (PREREG_MULTIGPU rev6 sec 4-3 E-time, P-D2) on synthetic telemetry.

Expected values are hand-derived in the comments below from the registered text
(busy = decode_running_batch_size > 0; weight = gap to the next non-startup
snapshot; numerator = stream_index not in {0, last}), not computed by the code
under test.  Mutants (lesson 53) replace one registered predicate each and must
change the result on BOTH the agnostic and the r2 (d44) fixture; the unmutated
module and a pass-through patch are the controls (lesson 254).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mg_realized_mix", ENG / "results" / "multigpu" / "mg_realized_mix.py")
MIX = importlib.util.module_from_spec(SPEC)
sys.modules["mg_realized_mix"] = MIX
SPEC.loader.exec_module(MIX)

AGN = MIX.REGISTERED_SM_COUNTS["agnostic"]
R2 = MIX.REGISTERED_SM_COUNTS["d44"]

# (t, decode_running_batch_size, stream_index)
# agnostic: busy weights 2.0(i1) 1.0(i1) 0.5(i2) 1.5(i3) [idle 1.0 at i0] 2.0(i1); last has no weight
#   den = 7.0, num(split = idx 1,2) = 5.5 -> 0.7857142857
AGN_ROWS = [(0.0, 4, 1), (2.0, 4, 1), (3.0, 40, 2), (3.5, 4, 3), (5.0, 0, 0), (6.0, 3, 1), (8.0, 3, 3)]
AGN_E = 5.5 / 7.0
# r2/d44: busy 3.0(i4) 1.0(i5) 1.0(i4) [idle 1.0 at i0]; den 5, num 4 -> 0.8
R2_ROWS = [(0.0, 10, 4), (3.0, 10, 5), (4.0, 10, 4), (5.0, 0, 0), (6.0, 2, 4)]
R2_E = 4.0 / 5.0


def snap(t, bs, idx, sm_counts, it):
    p, d = sm_counts[idx]
    return {"event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": t,
            "decode_running_batch_size": bs, "decode_batch_size": max(1, bs),
            "stream_index": idx, "prefill_sms": p, "decode_sms": d, "decode_iterations": it,
            "run_id": "x"}


def write_tel(path, rows, sm_counts, extra=()):
    recs = [{"event": "runtime_snapshot", "phase": "startup", "timestamp_monotonic_s": -1.0,
             "decode_running_batch_size": 9, "stream_index": 2, "prefill_sms": sm_counts[2][0],
             "decode_sms": sm_counts[2][1]},
            {"event": "split_transition", "phase": "benchmark", "timestamp_monotonic_s": 0.1}]
    recs += [snap(t, bs, idx, sm_counts, 10 * k) for k, (t, bs, idx) in enumerate(rows)]
    recs += list(extra)
    recs.sort(key=lambda r: r["timestamp_monotonic_s"])
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def green(sm_counts, override=None):
    groups = []
    for i, (p, d) in enumerate(sm_counts):
        div = 1 <= i <= len(sm_counts) - 2
        side = lambda sm: ({"green_ctx_is_null": False, "green_sm": {"smCount": sm}} if div
                           else {"green_ctx_is_null": True})
        groups.append({"stream_index": i, "target_prefill_sm": p, "target_decode_sm": d,
                       "is_sticky_target": False, "prefill": side(p), "decode": side(d)})
    doc = {"probe": "pdmux_green_readout", "sticky_target_index": None, "groups": groups}
    if override:
        override(doc)
    return doc


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cell(self, arm, rows, mode="n1", n=1, green_override=None, rows_by_rank=None, window=None):
        sm = MIX.REGISTERED_SM_COUNTS[arm]
        tel, grn = str(self.d / "tel.jsonl"), str(self.d / "green.json")
        tfiles, gfiles = MIX.rank_files(tel, mode, n), MIX.rank_files(grn, mode, n)
        for k, (tf, gf) in enumerate(zip(tfiles, gfiles)):
            write_tel(tf, (rows_by_rank or {}).get(k, rows), sm)
            Path(gf).write_text(json.dumps(green(sm, green_override)), encoding="utf-8")
        return MIX.cell_e_time(arm, mode, n, tel, grn, window)


class TestRegistered(Fixture):
    def test_config_tables_match_files(self):
        for arm, rel in MIX.ARM_CONFIGS.items():
            self.assertEqual(MIX.sm_counts_from_config(ENG / rel), MIX.REGISTERED_SM_COUNTS[arm], arm)

    def test_control_agnostic(self):
        r = self.cell("agnostic", AGN_ROWS)
        self.assertEqual(r["status"], "OK", r)
        self.assertAlmostEqual(r["e_time"], AGN_E, places=12)

    def test_control_r2(self):
        r = self.cell("d44", R2_ROWS)
        self.assertEqual(r["status"], "OK", r)
        self.assertAlmostEqual(r["e_time"], R2_E, places=12)

    def test_control_passthrough_patch(self):
        with mock.patch.object(MIX, "in_split_group", MIX.in_split_group):
            self.assertAlmostEqual(self.cell("agnostic", AGN_ROWS)["e_time"], AGN_E, places=12)

    def test_fused_not_applicable(self):
        self.assertEqual(MIX.cell_e_time("fused", "tp", 2, "a", "b")["status"], "NOT_APPLICABLE")
        self.assertEqual(MIX.cell_e_time("fused_sym", "tp", 2, "a", "b")["status"], "NOT_APPLICABLE")

    def test_exposure_steps(self):
        # decode_iterations rise 10 per snapshot; busy-left pairs: i1,i1,i2,i3,i1 (idle i0 excluded)
        # split steps = 4 of 5 -> 0.8
        self.assertAlmostEqual(self.cell("agnostic", AGN_ROWS)["exposure_steps"], 0.8)


MUTANTS = {
    "idx==2 literal (e2 D44_STREAM_INDEX)": ("in_split_group", lambda idx, real: idx == 2),
    "exclude only index 0": ("in_split_group", lambda idx, real: idx != 0),
    "busy = decode_batch_size (A2 violation)": ("is_decode_busy", lambda rec: (rec.get("decode_batch_size") or 0) > 0),
    "weight = gap to previous": ("snapshot_weights", lambda s: [
        (i, s[i]["timestamp_monotonic_s"] - s[i - 1]["timestamp_monotonic_s"]) for i in range(1, len(s))]),
    "uniform weight (count)": ("snapshot_weights", lambda s: [(i, 1.0) for i in range(len(s) - 1)]),
}


class TestMutants(Fixture):
    def test_every_mutant_killed_on_agnostic_and_r2(self):
        # Which fixture kills which mutant (hand-derived):
        #   idx==2      agnostic 0.5/7=0.071, r2 0/5=0           -> both killed
        #   only-0      agnostic 7/7=1.0,     r2 5/5=1.0          -> both killed
        #   busy-field  agnostic 5.5/8=0.6875, r2 4/6=0.667       -> both killed
        #   prev-gap    agnostic 4/6.5,        r2 (1+... ) != 0.8 -> both killed
        #   uniform     agnostic 4/5=0.8,      r2 2/3             -> both killed
        for name, (attr, fn) in MUTANTS.items():
            for arm, rows, want in (("agnostic", AGN_ROWS, AGN_E), ("d44", R2_ROWS, R2_E)):
                with self.subTest(mutant=name, arm=arm), mock.patch.object(MIX, attr, fn):
                    got = self.cell(arm, rows)["e_time"]
                    self.assertTrue(got is None or abs(got - want) > 1e-6, f"SURVIVED {name} on {arm}: {got}")

    def test_idx2_literal_is_exactly_wrong_where_expected(self):
        with mock.patch.object(MIX, "in_split_group", MUTANTS["idx==2 literal (e2 D44_STREAM_INDEX)"][1]):
            self.assertAlmostEqual(self.cell("agnostic", AGN_ROWS)["e_time"], 0.5 / 7.0)
            self.assertEqual(self.cell("d44", R2_ROWS)["e_time"], 0.0)


class TestCellRules(Fixture):
    def test_tp_reads_rank0_only(self):
        rows1 = [(0.0, 4, 3), (1.0, 4, 3)]  # rank 1 would give 0.0 if it were read
        r = self.cell("agnostic", AGN_ROWS, mode="tp", n=2, rows_by_rank={0: AGN_ROWS, 1: rows1})
        self.assertEqual(len(r["per_rank"]), 1)
        self.assertTrue(r["per_rank"][0]["telemetry"].endswith("tel.tp0.jsonl"))
        self.assertAlmostEqual(r["e_time"], AGN_E)

    def test_dp_mean_over_replicas(self):
        r = self.cell("agnostic", AGN_ROWS, mode="dp", n=2, rows_by_rank={0: AGN_ROWS, 1: [(0.0, 4, 1), (1.0, 4, 3), (2.0, 4, 3)]})
        # replica 1: weights 1.0(i1) 1.0(i3) -> 0.5 ; mean (5.5/7 + 0.5)/2
        self.assertEqual([Path(p["telemetry"]).name for p in r["per_rank"]], ["tel.dp0.jsonl", "tel.dp1.jsonl"])
        self.assertAlmostEqual(r["e_time"], (AGN_E + 0.5) / 2)

    def test_green_mismatch_is_mix_failed(self):
        def bad(doc):
            doc["groups"][1]["decode"]["green_sm"]["smCount"] = 36
        r = self.cell("agnostic", AGN_ROWS, green_override=bad)
        self.assertEqual(r["status"], "MIX_FAILED")
        self.assertIsNone(r["e_time"])

    def test_wrong_arm_table_is_mix_failed(self):
        sm = MIX.REGISTERED_SM_COUNTS["d44"]
        tel, grn = self.d / "t.jsonl", self.d / "g.json"
        write_tel(tel, R2_ROWS, sm)
        grn.write_text(json.dumps(green(sm)))
        r = MIX.cell_e_time("agnostic", "n1", 1, str(tel), str(grn))  # d44 data scored as agnostic
        self.assertEqual(r["status"], "MIX_FAILED")

    def test_plain_group_with_green_ctx_is_mix_failed(self):
        def bad(doc):
            doc["groups"][0]["prefill"] = {"green_ctx_is_null": False, "green_sm": {"smCount": 108}}
        self.assertEqual(self.cell("agnostic", AGN_ROWS, green_override=bad)["status"], "MIX_FAILED")

    def test_missing_replica_is_mix_failed(self):
        sm = AGN
        tel, grn = str(self.d / "tel.jsonl"), str(self.d / "green.json")
        f0 = MIX.rank_files(tel, "dp", 2)[0]
        write_tel(f0, AGN_ROWS, sm)
        for g in MIX.rank_files(grn, "dp", 2):
            Path(g).write_text(json.dumps(green(sm)))
        self.assertEqual(MIX.cell_e_time("agnostic", "dp", 2, tel, grn)["status"], "MIX_FAILED")

    def test_window_selects_cell_and_rejects_foreign_clock(self):
        # window [3.0, 5.0]: busy snapshots t=3 (i2, 0.5) and t=3.5 (i3, 1.5) -> 0.25
        self.assertAlmostEqual(self.cell("agnostic", AGN_ROWS, window=(3.0, 5.0))["e_time"], 0.25)
        self.assertEqual(self.cell("agnostic", AGN_ROWS, window=(1e6, 1e6 + 1))["status"], "MIX_FAILED")

    def test_partition_label(self):
        self.assertEqual(MIX.partition_label([0.4, 0.4, 0.6, 0.6]), "PARTITION_EXERCISED")  # median 0.5 is not < 0.5
        self.assertEqual(MIX.partition_label([0.1, 0.2, 0.3, 0.9]), "PARTITION_NOT_EXERCISED")
        self.assertEqual(MIX.partition_label([0.9, 0.9, None, 0.9]), "E_TIME_UNDEFINED")
        self.assertEqual(MIX.partition_label([0.9, 0.9, 0.9]), "E_TIME_UNDEFINED")


if __name__ == "__main__":
    unittest.main()
