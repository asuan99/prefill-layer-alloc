"""λ0 VESSL substrate port (PREREG_LAMBDA0_VESSL_2026-10-04.md): runner + helper.

What is tested, and the negative control that makes each test non-vacuous
(lesson 53: a repair test that also passes on the unrepaired code certifies nothing):

  rev5 not copied           the runner calls `$PREREG/lambda0_*.py`; the helper
                            imports rev5 modules.  Control: a runner that inlines
                            its own label call fails the structural test.
  command lines == rev5     server/warmup/bench flag multisets are extracted from
                            BOTH scripts and compared.  Control: dropping one flag
                            from a copy of the runner breaks equality.
  digests                   live repo = 908623 bytes.  Control: a one-byte change in
                            a temp copy of a rev5 file is reported as DRIFT.
  env allowlist             an unregistered PDMUX_* aborts before any work; a test
                            knob without DRYRUN aborts.
  budget enforcer           a fake cell that would overrun is cut at the deadline.
                            Control: the mutant with BOTH kill layers removed
                            (per-step timeout AND watchdog) overruns.  Removing ONE
                            layer is an EQUIVALENT mutant (the other still kills) --
                            recorded, not hidden (lesson 254).
  admission precedes boot   structural: the admission check sits before the
                            server launch inside run_cell.  Control: deleting it
                            breaks the order test.

The behavioural tests need the git-ignored job-907959 inputs; without them they
skip (same convention as test_lambda0_prereg).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
ROOT = ENG.parents[1]
PREREG = ENG / "results" / "r2_eval" / "lambda0_prereg"
HERE = ENG / "results" / "r2_eval" / "lambda0_vessl"
RUNNER = HERE / "lambda0_vessl.sh"
SBATCH = PREREG / "lambda0.sbatch"
I3_JOB = ENG / "results" / "r2_correctness" / "job_907959"
BASH = "/usr/bin/bash"

sys.path.insert(0, str(HERE))
import lambda0_vessl_plan as V  # noqa: E402

HAVE_INPUTS = all((I3_JOB / rel).exists() for rel in V.STAGED_INPUTS)


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env.update(extra)
    return env


def _flags(block: str):
    return sorted(re.findall(r"(--[a-z0-9-]+)", block))


def _rev5_block(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


class TestRev5IsCalledNotCopied(unittest.TestCase):
    def setUp(self):
        self.src = RUNNER.read_text()

    def test_runner_calls_every_rev5_decision_file(self):
        for f in ("lambda0_lambda_inf.py", "lambda0_plan.py", "lambda0_label.py",
                  "lambda0_analyze.py", "lambda0_cells.py", "lambda0_reachability.py",
                  "lambda0_mutation_check.py", "lambda0_cellprint.py"):
            self.assertIn(f'"$PREREG/{f}"', self.src, f)

    def test_helper_imports_rev5_modules(self):
        self.assertEqual(Path(V.P.__file__).resolve().parent, PREREG.resolve())
        self.assertEqual(Path(V.L.__file__).resolve().parent, PREREG.resolve())
        self.assertEqual(Path(V.C.__file__).resolve().parent, PREREG.resolve())

    def test_label_runs_on_rev5_cells_only(self):
        # the label sees $OUT; variance cells live in $OUT/variance
        self.assertIn('run_cell "$SPEC" "$OUT/variance"', self.src)
        self.assertIn('"$PREREG/lambda0_label.py" --cells-dir "$OUT"', self.src)
        self.assertNotIn('--cells-dir "$OUT/variance"', self.src)


class TestCommandLinesMatchRev5(unittest.TestCase):
    """Server / warmup / bench flag multisets are rev5's (lambda0.sbatch:338-378)."""

    def _pairs(self, runner_text):
        sb = SBATCH.read_text()
        rv = runner_text
        srv5 = _rev5_block(sb, "python -m sglang.launch_server", "PID=$!")
        warm5 = _rev5_block(sb, "python -m sglang.bench_serving", "warmup_rc")
        i2 = sb.index("python -m sglang.bench_serving", sb.index("warmup_rc"))
        bench5 = sb[i2:sb.index("bench_rc", i2)]
        srvp = _rev5_block(rv, "local SRV=(", "local WARM=(")
        warmp = _rev5_block(rv, "local WARM=(", "local BENCH=(")
        benchp = _rev5_block(rv, "local BENCH=(", "if [ \"$DRYRUN\" = 1 ]")
        return [(srv5, srvp), (warm5, warmp), (bench5, benchp)]

    def test_flags_identical(self):
        for a, b in self._pairs(RUNNER.read_text()):
            self.assertEqual(_flags(a), _flags(b))

    def test_control_dropped_flag_is_detected(self):
        mutant = RUNNER.read_text().replace(" --tokenize-prompt\n    --random-input-len \"$INLEN\" --random-output-len \"$OUTLEN\" --random-range-ratio 1.0\n    --num-prompts \"$NP\"",
                                            "\n    --random-input-len \"$INLEN\" --random-output-len \"$OUTLEN\" --random-range-ratio 1.0\n    --num-prompts \"$NP\"")
        self.assertNotEqual(mutant, RUNNER.read_text(), "mutation did not apply")
        self.assertTrue(any(_flags(a) != _flags(b) for a, b in self._pairs(mutant)))

    def test_arm_constants_identical(self):
        sb, rv = SBATCH.read_text(), RUNNER.read_text()
        for key in ("MODEL", "BACKEND", "CTX", "MEM_FRACTION", "MAX_RUNNING",
                    "SERVER_SEED", "FIXED_DSM"):
            a = re.search(rf"^{key}=(\S+)", sb, re.M).group(1)
            b = re.search(rf"^{key}=(\S+)", rv, re.M).group(1)
            self.assertEqual(a, b, key)
        self.assertIn('PLAN_ARGS=(--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback)', rv)


class TestDigests(unittest.TestCase):
    @unittest.skipUnless(HAVE_INPUTS, "git-ignored job 907959 inputs absent")
    def test_live_repo_is_908623_bytes(self):
        self.assertEqual(V.verify_digests(PREREG, I3_JOB, ENG), [])

    def test_control_one_byte_drift_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "prereg"
            shutil.copytree(PREREG, d, ignore=shutil.ignore_patterns("lam0_*", "*.out", "*.err", "__pycache__"))
            p = d / "lambda0_label.py"
            p.write_text(p.read_text().replace("ACH_LO = 0.90", "ACH_LO = 0.91"))
            bad = V.verify_digests(d, I3_JOB, ENG)
            self.assertTrue(any("lambda0_label.py" in b for b in bad), bad)

    def test_engine_tree_diff_detects_change(self):
        ref = PREREG / "lam0_908623" / "runtime_source_manifest.sha256"
        self.assertEqual(V.engine_tree_diff(ref, ref), [])
        with tempfile.TemporaryDirectory() as td:
            m = Path(td) / "m.sha256"
            lines = ref.read_text().splitlines()
            lines[0] = "0" * 64 + lines[0][64:]
            m.write_text("\n".join(lines) + "\n")
            self.assertEqual(len(V.engine_tree_diff(m, ref)), 1)


class TestHelper(unittest.TestCase):
    def test_selftest(self):
        V.selftest()

    def test_hard_cap_is_rev5_time_limit(self):
        self.assertEqual(V.HARD_CAP_S, 4 * 3600 + 30 * 60)   # lambda0.sbatch `--time=04:30:00`
        self.assertIn("#SBATCH --time=04:30:00", SBATCH.read_text())

    def test_runner_takes_cap_from_helper_not_env(self):
        src = RUNNER.read_text()
        self.assertIn("print(V.HARD_CAP_S)", src)
        self.assertNotRegex(src, r"HARD_CAP_S=\"\$\{PDMUX_LAM0V_HARD_CAP")


class TestEnvAllowlist(unittest.TestCase):
    def _run(self, **env):
        with tempfile.TemporaryDirectory() as td:
            e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=td, PDMUX_IO=td,
                           PDMUX_JOB_NAME="t_env", PDMUX_LAM0V_TEST_OUT_ROOT=td, **env)
            r = subprocess.run([BASH, str(RUNNER)], env=e, capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout + r.stderr

    def test_unregistered_knob_aborts(self):
        rc, out = self._run(PDMUX_LAM0V_DRYRUN="1", PDMUX_STICKY_PARTITION="1")
        self.assertEqual(rc, 2, out)
        self.assertIn("ABORT_ENV unregistered knob PDMUX_STICKY_PARTITION", out)

    def test_rev5_operator_knob_is_gone(self):
        rc, out = self._run(PDMUX_LAM0V_DRYRUN="1", PDMUX_LAMBDA0_INSTR="/elsewhere")
        self.assertEqual(rc, 2, out)

    def test_test_knob_without_dryrun_aborts(self):
        rc, out = self._run(PDMUX_LAM0V_TEST_HARD_CAP_S="700")
        self.assertEqual(rc, 2, out)
        self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", out)


class TestAdmissionPrecedesBoot(unittest.TestCase):
    @staticmethod
    def _order_ok(src):
        body = src[src.index("run_cell () {"):]
        try:
            adm = body.index('if [ "$REM" -lt "$NEED" ]; then')
        except ValueError:
            return False
        return adm < body.index('setsid "${SRV[@]}"')

    def test_order(self):
        self.assertTrue(self._order_ok(RUNNER.read_text()))

    def test_control_removed_admission(self):
        src = RUNNER.read_text()
        mutant = re.sub(r'  if \[ "\$REM" -lt "\$NEED" \]; then\n.*?\n  fi\n', "", src, count=1, flags=re.S)
        self.assertNotEqual(mutant, src, "mutation did not apply")
        self.assertFalse(self._order_ok(mutant))

    def test_need_is_rev5_worst_corner(self):
        import json
        p = json.loads((PREREG / "lam0_908623" / "plan.json").read_text())
        self.assertEqual(V.ADMISSION_RATIO, min(V.P.BUDGET_RATIO_GRID))
        self.assertGreater(V.cell_cost_s(p, "a_r1", 0.25), V.cell_cost_s(p, "a_r1", 1.0))


class TestFailClosedWiring(unittest.TestCase):
    """The drift checks are wired to aborts, not just computed."""

    def test_digest_and_plan_and_engine_checks_abort(self):
        src = RUNNER.read_text()
        for check, abort in (('python3 "$HELPER" --verify-digests', "ABORT_F6_DRIFT"),
                             ('cmp -s "$OUT/plan.json" "$PREREG/lam0_908623/plan.json"', "ABORT_PLAN_DRIFT"),
                             ('python3 "$HELPER" --engine-diff', "ABORT_ENGINE_TREE_DRIFT")):
            i = src.index(check)
            self.assertIn(abort, src[i:i + 600], check)


@unittest.skipUnless(HAVE_INPUTS, "git-ignored job 907959 inputs absent")
class TestResume(unittest.TestCase):
    """Resume copies only `.complete` cells from a previous Job with the SAME plan
    and registration digests.  Control: a registration that differs is refused."""

    def _prev(self, td, registration_text):
        p = Path(td) / "data/runs/prevjob/results/r2_eval/lambda0_vessl/lam0v_prevjob"
        (p / "variance").mkdir(parents=True)
        (Path(td) / "data/runs/prevjob/meta").mkdir(parents=True)
        shutil.copy(PREREG / "lam0_908623" / "plan.json", p / "plan.json")
        shutil.copy(PREREG / "lam0_908623" / "cell_a_r0.json", p / "cell_a_r0.json")
        (p / "cell_a_r0.complete").write_text("x\n")
        (p / "REGISTRATION_SHA256.txt").write_text(registration_text)
        (Path(td) / "data/runs/prevjob/meta/substrate.json").write_text(
            '{"gpu": {"uuid": "GPU-prev", "driver_version": "580.105.08"}}')

    def _run(self, td, job):
        e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=f"{td}/data", PDMUX_IO=f"{td}/io",
                       PDMUX_JOB_NAME=job, PDMUX_LAM0V_TEST_OUT_ROOT=f"{td}/out",
                       PDMUX_LAM0V_DRYRUN="1", PDMUX_LAM0V_TEST_SKIP_PREFLIGHT="1",
                       PDMUX_LAM0V_RESUME_FROM="prevjob")
        r = subprocess.run([BASH, str(RUNNER)], env=e, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr, Path(td) / "out" / f"lam0v_{job}"

    def test_resume_and_its_control(self):
        with tempfile.TemporaryDirectory() as td:
            self._prev(td, "0" * 64 + "  lambda0_plan.py\n")          # wrong digests
            rc, out, _ = self._run(td, "j1")
            self.assertEqual(rc, 2, out)
            self.assertIn("ABORT_RESUME registration digests differ", out)
            reg = (Path(td) / "out" / "lam0v_j1" / "REGISTRATION_SHA256.txt").read_text()
            shutil.rmtree(Path(td) / "data/runs/prevjob")
            self._prev(td, reg)                                         # same digests
            rc, out, o = self._run(td, "j2")
            self.assertEqual(rc, 0, out)
            self.assertIn("SKIP_DONE cell=a_r0", out)
            self.assertIn("a_r0\tprevjob\tGPU-prev", (o / "RESUMED_CELLS.tsv").read_text())
            self.assertNotIn("CELL a_r0\n", (o / "DRYRUN_COMMANDS.txt").read_text())


@unittest.skipUnless(HAVE_INPUTS, "git-ignored job 907959 inputs absent")
class TestBudgetEnforcer(unittest.TestCase):
    """Fake cells (DRYRUN) under the same timeout/watchdog/admission machinery.

    cap 620 s - tail 600 s => the cell deadline is ~20 s after start.  A fake cell
    sleeps 60 s, so only a kill layer can end the run before ~65 s.  (Admission
    alone still stops the run one cell after the deadline, so the double-layer
    mutant overruns by one cell, not by the whole ladder.)"""

    CAP = "620"
    FAKE = "60"
    LIMIT_S = 120

    def _run(self, script: Path):
        with tempfile.TemporaryDirectory() as td:
            e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=td, PDMUX_IO=td,
                           PDMUX_JOB_NAME="t_budget", PDMUX_LAM0V_TEST_OUT_ROOT=td,
                           PDMUX_LAM0V_DRYRUN="1", PDMUX_LAM0V_TEST_SKIP_PREFLIGHT="1",
                           PDMUX_LAM0V_TEST_HARD_CAP_S=self.CAP,
                           PDMUX_LAM0V_TEST_FAKE_CELL_S=self.FAKE)
            t0 = time.monotonic()
            try:
                r = subprocess.run([BASH, str(script)], env=e, capture_output=True,
                                   text=True, timeout=self.LIMIT_S)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "lambda0_vessl_fake_cell"], check=False)
                subprocess.run(["pkill", "-f", "sleep 100000"], check=False)
                return None, time.monotonic() - t0, {}
            wall = time.monotonic() - t0
            out = Path(td) / "lam0v_t_budget"
            files = {p.name: p.read_text() for p in out.glob("*.txt")}
            files["_fired"] = (out / "BUDGET_HARDCAP_FIRED").exists()
            return r.returncode, wall, files

    def test_overrunning_cell_is_cut_at_the_deadline(self):
        rc, wall, f = self._run(RUNNER)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 45.0)                                   # deadline ~20 s + tail
        fake = f.get("FAKE_CELLS.txt", "")
        self.assertRegex(fake, r"fake_rc=(124|143|137) cell=a_r0")   # killed, not finished
        self.assertIn("NOT_ATTEMPTED_BUDGET", f.get("BUDGET_SKIPPED.txt", ""))
        self.assertEqual(fake.count("fake_rc="), 1, fake)             # nothing after the cut

    def test_control_both_kill_layers_removed_overruns(self):
        src = RUNNER.read_text()
        m = src.replace('timeout -k 2 "$CAP" bash -c', 'bash -c')
        m = m.replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n',
                      '  sleep 100000\n')
        self.assertEqual(m.count("sleep 100000"), 1)
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "lambda0_vessl_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._run(mut)
        # the fake cell runs to completion past the deadline (>= 60 s wall)
        self.assertGreaterEqual(wall, float(self.FAKE), "mutant was still cut -- the test is vacuous")

    def test_equivalent_mutant_one_layer_removed_still_cuts(self):
        # watchdog removed, per-step timeout kept: still cut (redundant layers)
        m = RUNNER.read_text().replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n',
                                       '  sleep 100000\n')
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "lambda0_vessl_mutant.sh"
            mut.write_text(m)
            rc, wall, f = self._run(mut)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 45.0)


if __name__ == "__main__":
    unittest.main()
