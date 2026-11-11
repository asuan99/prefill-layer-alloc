"""mo_workload: registered motivation workloads in the runner (traces from gen_traces, token ids from the registered token files)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_motivation_harness as H  # noqa: E402  (fixtures: fake server / replay / pool)

MS, MC, MR, MA, MW = H.MS, H.MC, H.MR, H.MA, H.MS.MW
TOK_DIR = ENG / "results" / "motivation" / "workloads" / "tokens"
HAVE_TOKENS = all((TOK_DIR / MW.WORKLOADS[w]["tokens"]["name"]).exists() for w in ("openchat_rounds", "arxiv_summ"))
GRID = {"rates": [2.0, 3.0], "role": "base"}


def plan(workload, job="JWL", n=8, rehearsal=False):
    return MS.make_plan(job, dict(GRID), fixed=dict(MS.REHEARSAL_FIXED), n_requests=n, rehearsal=rehearsal, workload=workload)


class TestPlan(unittest.TestCase):
    def test_registered_seeds_and_workload_field(self):
        p = plan("openchat_rounds")
        self.assertEqual(p["seeds"], list(MW.SEEDS))
        self.assertTrue(all(r["workload"] == "openchat_rounds" for b in p["boots"] for r in b["runs"]))
        self.assertEqual(MS.plan_problems(p, allow_placeholders=True), [])

    def test_unregistered_workload_refused(self):
        with self.assertRaises(ValueError):
            plan("loogle_longdep")

    def test_same_trace_for_every_arm_at_a_rate_seed(self):
        p = plan("arxiv_summ")
        th = {}
        for b in p["boots"]:
            if b["role"] != "meas":
                continue
            for r in b["runs"]:
                th.setdefault((r["rate"], r["seed"]), set()).add(MC.trace_facts(MW.build_segments(r, H.POOL))["hash"])
        self.assertTrue(th and all(len(v) == 1 for v in th.values()))

    def test_output_conditions(self):
        run = {"run_id": "x", "workload": "arxiv_summ", "rate": 2.0, "seed": MW.SEEDS[0], "n_requests": 30}
        main = MW.build_segments(run, H.POOL)[0]["requests"]
        var = MW.build_segments(dict(run, workload="arxiv_summ_out1024"), H.POOL)[0]["requests"]
        self.assertEqual([r["doc_id"] for r in main], [r["doc_id"] for r in var])
        self.assertEqual([r["arrival_s"] for r in main], [r["arrival_s"] for r in var])
        self.assertTrue(all(r["output_tokens"] == 1024 and r["ignore_eos"] for r in var))
        self.assertTrue(all(r["ignore_eos"] for r in main))
        self.assertGreater(len({r["output_tokens"] for r in main}), 5)          # abstract lengths, not one constant

    def test_card_records_workload_pool_and_token_sha(self):
        p = plan("openchat_rounds")
        card = MC.build_card(p, H.POOL)
        self.assertEqual(card["workload"]["workload"], "openchat_rounds")
        self.assertEqual(card["workload"]["token_file_sha256"], MW.WORKLOADS["openchat_rounds"]["tokens"]["sha256"])


@unittest.skipUnless(HAVE_TOKENS, "local token files absent (results/motivation/workloads/tokens/)")
class TestTokens(unittest.TestCase):
    def test_ids_match_lengths(self):
        for wl in ("openchat_rounds", "arxiv_summ"):
            st = MW.TokenStore(wl, TOK_DIR)
            run = {"run_id": "x", "workload": wl, "rate": 2.0, "seed": MW.SEEDS[0], "n_requests": 5}
            segs = MW.with_input_ids(MW.build_segments(run, H.POOL), st)
            for r in segs[0]["requests"]:
                self.assertEqual(len(r["input_ids"]), r["input_tokens"])
                self.assertTrue(all(isinstance(t, int) for t in r["input_ids"][:20]))

    def test_sharegpt_has_no_token_file(self):
        self.assertIsNone(MW.store_for("sharegpt_first"))

    def test_job_sends_input_ids_and_hash_excludes_them(self):
        sent = []

        def replay(api, base_url, reqs, out_path, perf=None):
            sent.extend(reqs)
            return H.fake_replay(api, base_url, reqs, out_path)

        p = plan("openchat_rounds", "JWLE", n=6)
        card = MC.build_card(p, H.POOL)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"PDMUX_MO_TOKEN_DIR": str(TOK_DIR)}):
            out = Path(td) / "raw"
            MR.run_job(p, card, out, H.make_server(), replay, H.POOL, gate_fn=lambda p: None, py_version_fn=H.fake_py,
                       freeze_fn=H.fake_freeze, warmup_s=2.0)
            job = MA.load_job(out)
            self.assertEqual(MA.job_problems(job)[0], [])
            res = MA.analyze_job(job)
        meas = [q for q in sent if "doc_id" in q]
        self.assertTrue(meas and all(len(q["input_ids"]) == q["input_tokens"] for q in meas))
        self.assertIn("verdict_rows", res)


class TestTokenAbort(unittest.TestCase):
    def test_missing_token_file_aborts_before_any_boot(self):
        p = plan("arxiv_summ", "JWLA", n=4)
        card = MC.build_card(p, H.POOL)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"PDMUX_MO_TOKEN_DIR": td}):
            with self.assertRaises(MR.JobAbort) as cm:
                MR.run_job(p, card, Path(td) / "raw", H.make_server(), H.fake_replay, H.POOL, gate_fn=lambda p: None,
                           py_version_fn=H.fake_py, freeze_fn=H.fake_freeze, warmup_s=2.0)
            self.assertEqual(cm.exception.status, "ABORT_TOKENS")
            self.assertFalse((Path(td) / "raw" / "boots").exists())

    def test_wrong_token_file_sha_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / MW.WORKLOADS["arxiv_summ"]["tokens"]["name"]).write_bytes(b"not the file")
            with self.assertRaises(ValueError) as cm:
                MW.TokenStore("arxiv_summ", td)
            self.assertIn("ABORT_TOKEN_FILE_SHA", str(cm.exception))

    def test_cli_refuses_w1_for_a_measurement_run(self):
        with tempfile.TemporaryDirectory() as td:
            rc = MR.main(["run", "--job", "JCLI", "--w1-lengths", str(Path(td) / "p.json"), "--card", "c.json", "--out", td])
        self.assertEqual(rc, MR.EXIT_ABORT)


if __name__ == "__main__":
    unittest.main()


class TestRehearsalWorkloads(unittest.TestCase):
    def _rec(self, resolved, workload="openchat_rounds"):
        p = plan(workload, "JGATE")
        return p, {"harness_digest": "e" * 64, "passed": True, "stages": {s: True for s in H.MH.STAGES},
                   "arm_names": sorted({b["arm"] for b in p["boots"]}), "workloads_resolved": resolved}

    def test_gate_needs_the_plan_workload_resolved(self):
        with tempfile.TemporaryDirectory() as td:
            p, rec = self._rec({"openchat_rounds": "FileNotFoundError: x"})
            (Path(td) / ("e" * 16 + ".json")).write_text(json.dumps(rec))
            self.assertIn("REHEARSAL_WORKLOAD_NOT_RESOLVED", H.MH.launch_gate(p, td, "e" * 64))
            rec["workloads_resolved"]["openchat_rounds"] = "OK"
            (Path(td) / ("e" * 16 + ".json")).write_text(json.dumps(rec))
            self.assertIsNone(H.MH.launch_gate(p, td, "e" * 64))

    def test_harness_digest_covers_trace_and_token_code(self):
        names = {f.name for f in H.MH._harness_files()}
        self.assertTrue({"gen_traces.py", "mw_lib.py", "mg_x_pretokenize.py", "mo_workload.py", "mo_realized.py", "mo_clock.py"} <= names)

    @unittest.skipUnless(HAVE_TOKENS, "local token files absent")
    def test_full_rehearsal_on_a_token_workload(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"PDMUX_MO_TOKEN_DIR": str(TOK_DIR)}):
            rec = H.MH.rehearse("JRWL", H.POOL, Path(td) / "raw", H.make_server(), H.fake_replay, rehearsal_dir=Path(td) / "rec",
                                digest="f" * 64, py_version_fn=H.fake_py, freeze_fn=H.fake_freeze, workload="openchat_rounds")
            self.assertTrue(rec["passed"], rec["reason"])
            self.assertEqual(rec["workload"], "openchat_rounds")
            self.assertEqual(set(rec["workloads_resolved"].values()), {"OK"})


class TestAnalyzerCli(unittest.TestCase):
    def test_invalid_job_writes_reason(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "raw" / "runs").mkdir(parents=True)
            rc = MA.main([str(Path(td) / "raw"), "--out", str(Path(td) / "an")])
            self.assertEqual(rc, 3)
            self.assertTrue((Path(td) / "an" / "JOB_INVALID.json").exists())


class TestJobScript(unittest.TestCase):
    P = ENG / "results" / "motivation" / "mo_job.sh"

    def run_sh(self, env):
        import subprocess
        e = {k: v for k, v in os.environ.items() if not k.startswith(("MO_", "PDMUX_MO_"))}
        e.update(env)
        return subprocess.run(["bash", str(self.P)], env=e, capture_output=True, text=True, timeout=60)

    def test_syntax_executable_digest(self):
        import subprocess
        self.assertEqual(subprocess.run(["bash", "-n", str(self.P)]).returncode, 0)
        self.assertTrue(os.access(self.P, os.X_OK))
        self.assertIn("8d8f67201199bd181f64aae043b6a8ec5c6772799dbf964081584684e9e9c858", self.P.read_text())

    def test_config_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            base = {"PDMUX_MO_OUT": td, "PDMUX_MO_TEST_MODE": "1"}
            self.assertEqual(self.run_sh(dict(base)).returncode, 16)
            self.assertEqual(self.run_sh(dict(base, MO_MODE="rehearse", MO_WORKLOAD="openchat_rounds", MO_JOB="J")).returncode, 16)   # no budget
            r = self.run_sh(dict(base, MO_MODE="run", MO_WORKLOAD="openchat_rounds", MO_JOB="J", MO_BUDGET_MIN="10"))
            self.assertEqual(r.returncode, 16)
            self.assertIn("MO_RMIN", r.stderr)

    def test_run_refuses_placeholder_card(self):
        """Until the cap measurement fixes mem/cap the card holds placeholders and the runner aborts before any boot."""
        with tempfile.TemporaryDirectory() as td:
            pool = Path(td) / "w1.json"
            pool.write_text(json.dumps(H.POOL))
            r = self.run_sh({"PDMUX_MO_OUT": str(Path(td) / "o"), "PDMUX_MO_TEST_MODE": "1", "PDMUX_MO_W1_LENGTHS": str(pool), "MO_MODE": "run",
                             "MO_WORKLOAD": "sharegpt_first", "MO_JOB": "JPH", "MO_BUDGET_MIN": "5", "MO_RMIN": "2", "MO_FACTOR": "1.2",
                             "MO_N": "2", "MO_N_REQUESTS": "8"})
            self.assertEqual(r.returncode, 19, r.stderr[-400:])
            self.assertIn("ABORT_PLAN", r.stderr + r.stdout)
            self.assertFalse((Path(td) / "o" / "raw" / "boots").exists())
