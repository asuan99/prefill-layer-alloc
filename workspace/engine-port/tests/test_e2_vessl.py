"""E2 sticky contrast, VESSL substrate port (PREREG_E2_VESSL_A13_2026-10-04.md): runner + helper.

What is tested, and the negative control that makes each test non-vacuous
(lesson 53: a repair test that also passes on the unrepaired code certifies nothing):

  rev3 called, not copied   the runner calls `$PREREG/e2_{realized_mix,label}.py`.
                            Control: a runner that inlines its own label fails.
  command lines == sbatch   server / warm-up / bench flag multisets extracted from BOTH
                            scripts and compared; the P13 probe body is byte-identical.
                            Control: dropping one flag from a copy breaks equality.
  20-boot order             helper units == the sbatch's literal construction.
  digests                   live repo == registered bytes (E2 decision path, A13, config).
                            Control: a one-byte change in a temp copy is reported.
  engine tree               908623 + exactly multiplexing_mixin.py at HEAD's bytes passes;
                            identical tree, a second file, or other mixin bytes fail.
  env allowlist             PDMUX_STICKY_PARTITION / PDMUX_PHASE_EVENTS inherited -> abort;
                            a test knob without DRYRUN -> abort.
  P14 + hard cap            a fake bench that would overrun is cut at the deadline and no
                            later boot starts.  Control: BOTH kill layers removed overruns;
                            ONE layer removed is an EQUIVALENT mutant (recorded, lesson 254).
  admission precedes boot   structural.  Control: deleting it breaks the order test.
  resume                    pair granularity; a pair with only one complete boot is re-run;
                            different registration digests are refused.

The behavioural tests need the git-ignored job-908623 selftest archive (seven telemetry
files); without it they skip (same convention as test_lambda0_vessl).
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
PREREG = ENG / "results" / "r2_eval" / "e2_sticky_prereg"
HERE = ENG / "results" / "r2_eval" / "e2_sticky_vessl"
RUNNER = HERE / "e2_sticky_vessl.sh"
SBATCH = PREREG / "e2_sticky.sbatch"
ARCHIVE = ENG / "results" / "r2_eval" / "lambda0_prereg" / "lam0_908623"
BASH = "/usr/bin/bash"

sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import e2_vessl_plan as V  # noqa: E402

HAVE_ARCHIVE = all((ARCHIVE / n).exists() for n in V.ARCHIVE_DIGESTS)


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _flags(block: str):
    return sorted(re.findall(r"(--[a-z0-9-]+)", block))


def _block(text: str, start: str, end: str, after: str | None = None) -> str:
    i = text.index(start, text.index(after) if after else 0)
    return text[i:text.index(end, i)]


def _sbatch_blocks(sb: str):
    srv = _block(sb, "python -m sglang.launch_server", "PID=$!")
    warm = _block(sb, "python -m sglang.bench_serving", "warmup_rc")
    bench = _block(sb, "python -m sglang.bench_serving", "bench_rc", after="warmup_rc")
    return srv, warm, bench


def _runner_blocks(rv: str):
    return (_block(rv, "SRV=(python", "WARM=("), _block(rv, "WARM=(python", "BENCH=("),
            _block(rv, "BENCH=(python", 'if [ "$DRYRUN" = 1 ]'))


class TestRev3IsCalledNotCopied(unittest.TestCase):
    def setUp(self):
        self.src = RUNNER.read_text()

    def test_runner_calls_rev3_files(self):
        self.assertIn('python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH"', self.src)
        self.assertIn('python3 "$PREREG/e2_label.py" --cells-dir "$OUT" --expect "$EXPECT"', self.src)
        self.assertIn('"$PREREG/e2_realized_mix.py" --selftest --archive "$ARCHIVE"', self.src)
        self.assertIn('"$PREREG/e2_label.py" --selftest', self.src)
        self.assertNotIn("def label(", self.src)
        self.assertFalse(list(HERE.glob("e2_label*.py")) or list(HERE.glob("e2_realized_mix*.py")))

    def test_helper_has_no_rule_or_estimator(self):
        txt = (HERE / "e2_vessl_plan.py").read_text()
        for banned in ("def label(", "def compute(", "paired_bootstrap", "R1_OFF_BAND"):
            self.assertNotIn(banned, txt)

    def test_estimator_call_arguments_match_sbatch(self):
        sb, rv = SBATCH.read_text(), self.src
        a = _block(sb, 'python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH"', "|| echo")
        b = _block(rv, 'python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH"', "|| echo")
        norm = lambda s: re.sub(r"\$\{(INLEN|OUTLEN|NP)\[\$CELL\]\}", r"$\1", s)  # noqa: E731
        self.assertEqual(_flags(norm(a)), _flags(norm(b)))
        self.assertEqual(" ".join(norm(a).split()), " ".join(norm(b).split()))


class TestCommandLinesMatchSbatch(unittest.TestCase):
    @staticmethod
    def _norm(s):
        return re.sub(r"\$\{(INLEN|OUTLEN|NP|RATE)\[\$CELL\]\}", r"$\1", s)

    def test_flags_identical(self):
        for a, b in zip(_sbatch_blocks(SBATCH.read_text()), _runner_blocks(RUNNER.read_text())):
            self.assertEqual(_flags(a), _flags(b))

    def test_control_dropped_flag_is_detected(self):
        src = RUNNER.read_text()
        mutant = src.replace('--seed "$CSEED" --output-details', '--seed "$CSEED"', 1)
        self.assertNotEqual(mutant, src, "mutation did not apply")
        self.assertTrue(any(_flags(a) != _flags(b) for a, b in
                            zip(_sbatch_blocks(SBATCH.read_text()), _runner_blocks(mutant))))

    def test_p13_probe_body_byte_identical(self):
        def body(t):
            i = t.index("<<'PY' | tee -a \"$OUT/P13_PROBE.txt\"")
            return t[t.index("\n", i) + 1:t.index("\nPY\n", i)]
        self.assertEqual(body(SBATCH.read_text()), body(RUNNER.read_text()))

    def test_p13_verdict_body_byte_identical(self):
        def body(t):
            i = t.index("<<'PY' | tee \"$OUT/P13_VERDICT.txt\"")
            return t[t.index("\n", i) + 1:t.index("\nPY\n", i)]
        self.assertEqual(body(SBATCH.read_text()), body(RUNNER.read_text()))

    def test_one_knob_text(self):
        knob = 'if [ "$ARM" = "ON" ]; then export PDMUX_STICKY_PARTITION=1; else unset PDMUX_STICKY_PARTITION; fi'
        self.assertIn(knob, SBATCH.read_text())
        self.assertIn(knob, RUNNER.read_text())
        # nothing else in the runner sets the knob
        self.assertEqual(RUNNER.read_text().count("export PDMUX_STICKY_PARTITION"), 1)

    def test_control_constants_identical(self):
        sb, rv = SBATCH.read_text(), RUNNER.read_text()
        for key in ("MODEL", "BACKEND", "CTX", "MEM_FRACTION", "MAX_RUNNING", "SERVER_SEED", "FIXED_DSM"):
            a = re.search(rf"^{key}=(\S+)", sb, re.M).group(1)
            b = re.search(rf"^{key}=(\S+)", rv, re.M).group(1)
            self.assertEqual(a, b, key)
        self.assertEqual(V.FIXED_DSM, 44)
        self.assertIn('CFG="$ENG/results/longctx_conflict/probes/pdmux_homog5.yml"', rv)

    def test_cells_and_order_match_sbatch(self):
        sb = SBATCH.read_text()
        for cell, (shape, inl, outl, rate, np_) in V.CELLS.items():
            self.assertIn(f"[{cell}]={shape}", sb)
            self.assertIn(f"[{cell}]={inl}", sb)
            self.assertIn(f"[{cell}]={outl}", sb)
            self.assertIn(f"[{cell}]={rate}", sb)
            self.assertIn(f"[{cell}]={np_}", sb)
        self.assertIn("SEEDS_A=(" + " ".join(V.SEEDS_A) + ")", sb)
        self.assertIn("SEEDS_B=(" + " ".join(V.SEEDS_B) + ")", sb)
        # sbatch: for each cell, for each seed, for ARM in OFF ON
        want = [(c, s, a) for c, seeds in (("a_r4", V.SEEDS_A), ("a_r2", V.SEEDS_A), ("b_r3", V.SEEDS_B))
                for s in seeds for a in ("OFF", "ON")]
        self.assertEqual(V.units(), want)
        self.assertIn("for ARM in OFF ON; do", sb)


class TestDigests(unittest.TestCase):
    def test_live_repo_is_registered_bytes(self):
        self.assertEqual(V.verify_digests(PREREG, HERE, ENG), [])

    def test_a13_digest_pinned(self):
        self.assertEqual(V.sha256(HERE / V.A13_NAME), V.A13_DIGEST,
                         "A13 edited without updating e2_vessl_plan.A13_DIGEST")

    def test_control_one_byte_drift_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "prereg"
            shutil.copytree(PREREG, d, ignore=shutil.ignore_patterns("e2_*_[0-9]*", "__pycache__", "*.out", "*.err"))
            p = d / "e2_label.py"
            p.write_text(p.read_text().replace("R1_OFF_BAND = 20.0", "R1_OFF_BAND = 21.0"))
            bad = V.verify_digests(d, HERE, ENG)
            self.assertTrue(any("e2_label.py" in b for b in bad), bad)

    @unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
    def test_archive_is_908623_bytes(self):
        self.assertEqual(V.verify_archive(ARCHIVE), [])


class TestEngineTree(unittest.TestCase):
    REF = ARCHIVE / "runtime_source_manifest.sha256"
    KEY = "sglang/srt/multiplex/multiplexing_mixin.py"

    def _head(self):
        return [ln.replace(V.ENGINE_ALLOWED_DIFF[self.KEY]["ref_908623"], V.ENGINE_ALLOWED_DIFF[self.KEY]["this_head"])
                for ln in self.REF.read_text().splitlines()]

    def _diff(self, lines):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.sha256"
            p.write_text("\n".join(lines) + "\n")
            return V.engine_diff(p, self.REF)[0]

    def test_head_mixin_is_src_mirror(self):
        self.assertEqual(V.sha256(ENG / "src" / "multiplex" / "multiplexing_mixin.py"),
                         V.ENGINE_ALLOWED_DIFF[self.KEY]["this_head"])

    def test_registered_one_file_difference_passes(self):
        self.assertEqual(self._diff(self._head()), [])

    def test_control_identical_tree_fails(self):
        self.assertTrue(self._diff(self.REF.read_text().splitlines()))

    def test_control_second_file_fails(self):
        lines = self._head()
        j = next(i for i, ln in enumerate(lines) if "nemotron_h.py" in ln)
        lines[j] = "0" * 64 + lines[j][64:]
        self.assertTrue(self._diff(lines))

    def test_control_other_mixin_bytes_fail(self):
        lines = [ln.replace(V.ENGINE_ALLOWED_DIFF[self.KEY]["ref_908623"], "1" * 64)
                 for ln in self.REF.read_text().splitlines()]
        self.assertTrue(self._diff(lines))


class TestHelper(unittest.TestCase):
    def test_selftest(self):
        V.selftest()

    def test_t_cap_formula_reproduces_sbatch_literals(self):
        sb = SBATCH.read_text()
        m = re.search(r"declare -A T_CAP=\( \[a_r4\]=(\d+) \[a_r2\]=(\d+) \[b_r3\]=(\d+) \)", sb)
        self.assertEqual(tuple(int(x) for x in m.groups()),
                         tuple(V.t_cap(V.KISTI_OFF_DURATION_S[c]) for c in ("a_r4", "a_r2", "b_r3")))
        self.assertEqual(V.T_CAP_S, {"a_r4": 401, "a_r2": 429, "b_r3": 870})

    def test_p14_margin_is_rev3_literal(self):
        self.assertIn("P14_MARGIN_S=200", SBATCH.read_text())
        self.assertEqual(V.P14_MARGIN_S, 200)

    def test_hard_cap_within_approval(self):
        self.assertLessEqual(V.HARD_CAP_S, 4 * 3600)
        src = RUNNER.read_text()
        self.assertIn("HARD_CAP_S=$(K HARD_CAP_S)", src)
        self.assertNotRegex(src, r"HARD_CAP_S=\"\$\{PDMUX_E2V_HARD")

    def test_budget_corners(self):
        b = V.budget_table()
        self.assertEqual(b["scenarios"]["normal"]["boots_admitted"], 20)
        self.assertEqual(b["scenarios"]["worst"]["boots_admitted"], 20)
        self.assertLess(b["scenarios"]["capped"]["boots_admitted"], 20)   # P14 is reachable


class TestFailClosedWiring(unittest.TestCase):
    def test_checks_are_wired_to_aborts(self):
        src = RUNNER.read_text()
        for check, abort in (('python3 "$HELPER" --verify-digests', "ABORT_REG_DRIFT"),
                             ('python3 "$HELPER" --verify-archive', "ABORT_ARCHIVE_DRIFT"),
                             ('python3 "$HELPER" --engine-diff', "ABORT_ENGINE_TREE_DRIFT"),
                             ('python3 "$HELPER" --check-substrate', "ABORT_SUBSTRATE"),
                             ('"$PREREG/e2_realized_mix.py" --selftest', "ABORT_SELFTEST")):
            i = src.index(check)
            self.assertIn(abort, src[i:i + 400], check)

    def test_test_knobs_refused_outside_dryrun(self):
        src = RUNNER.read_text()
        i = src.index('if [ "$DRYRUN" != 1 ]; then')
        blk = src[i:i + 400]
        for v in ("PDMUX_E2V_TEST_HARD_CAP_S", "PDMUX_E2V_TEST_FAKE_BENCH_S",
                  "PDMUX_E2V_TEST_SKIP_PREFLIGHT", "PDMUX_E2V_TEST_OUT_ROOT", "PDMUX_TEST_SKIP_SYNC"):
            self.assertIn(v, blk)

    def test_abort_paths_tear_the_server_down(self):
        src = RUNNER.read_text()
        for tag in ("ABORT_STICKY_STATE_MISMATCH ON", "ABORT_STICKY_STATE_MISMATCH OFF", "ABORT_P2",
                    "ABORT_P3", "empty completions"):
            i = src.index(tag)
            self.assertIn('teardown "$PID" "$PORT"', src[i - 200:i], tag)

    def test_green_readout_both_arms_before_knob(self):
        src = RUNNER.read_text()
        g = src.index('export PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH=')
        k = src.index('if [ "$ARM" = "ON" ]; then export PDMUX_STICKY_PARTITION=1')
        self.assertLess(g, k)   # unconditional, not inside the arm branch


class TestRealPathKillLayers(unittest.TestCase):
    """The REAL (non-DRYRUN) path, which TestBudgetEnforcer's fake bench never exercises
    (VERDICT_e2_vessl_rules_2026-10-04 E2VC-4 / required repair 5).  Each check has a
    control mutant that must make it fail (lesson 53)."""

    BENCH_LINE = 'timeout -k 30 "${BT}s" "${BENCH[@]}"'

    # ---- (a) the measurement bench is wrapped in the per-step timeout
    @classmethod
    def _bench_wrapped(cls, src):
        real = src[src.index("  : > \"$PDMUX_TELEMETRY_PATH\"\n  setsid"):]   # after the DRYRUN branch
        return cls.BENCH_LINE in real and 'BT=$TCAP; KIND=P7_CAPPED' in real

    def test_bench_timeout_present(self):
        self.assertTrue(self._bench_wrapped(RUNNER.read_text()))

    def test_control_bench_timeout_removed(self):
        src = RUNNER.read_text()
        m = src.replace(self.BENCH_LINE, '"${BENCH[@]}"')
        self.assertNotEqual(m, src)
        self.assertFalse(self._bench_wrapped(m))

    # ---- (b) the watchdog's pkill patterns actually match the launched command lines
    @staticmethod
    def _watchdog_ok(src):
        i = src.index('while [ "$(remaining)" -gt 0 ]; do sleep 5; done')
        wd = src[i:src.index("WD_PID=$!", i)]
        pats = {sig: re.findall(rf'pkill -{sig} -f "([^"]+)"', wd) for sig in ("TERM", "9")}
        srv = " ".join(_block(src, "SRV=(python", "WARM=(").split()[0:3]).replace("SRV=(", "")
        bench = " ".join(_block(src, "BENCH=(python", 'if [ "$DRYRUN" = 1 ]').split()[0:3]).replace("BENCH=(", "")
        # the pgrep/pkill -f pattern is an extended regex over the full command line
        return all(any(re.search(p, cmd) for p in pats[sig])
                   for sig in ("TERM", "9") for cmd in (srv, bench))

    def test_watchdog_patterns_match_srv_and_bench(self):
        self.assertTrue(self._watchdog_ok(RUNNER.read_text()))

    def test_control_watchdog_pattern_changed(self):
        src = RUNNER.read_text()
        m = src.replace('pkill -TERM -f "sglang.launch_server"', 'pkill -TERM -f "sglang.launch-server"')
        self.assertNotEqual(m, src)
        self.assertFalse(self._watchdog_ok(m))
        m2 = src.replace('pkill -9 -f "sglang.bench_serving"', 'pkill -9 -f "sglang.benchserving"')
        self.assertNotEqual(m2, src)
        self.assertFalse(self._watchdog_ok(m2))

    # ---- (c) a BUDGET cut never writes boot_<NAME>.complete (so a resume re-runs the pair);
    #          a P7 CAPPED cut does (registered outcome, never retried).  Behavioural: the
    #          runner's own classification + completion lines are executed with stubs.
    @staticmethod
    def _snippet(src):
        a = src[src.index('  CUT=""\n'):src.index('  grep -E "Successful requests')]
        b = src[src.index("  # complete = a registered outcome"):src.index("  if budget_fired; then STOP=budget")]
        return a + b

    def _run_snippet(self, snippet, brc, kind, fired):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "cell_X.json").write_text("{}\n")
            if fired:
                (out / "BUDGET_HARDCAP_FIRED").write_text("x\n")
            script = (f'OUT="{td}"; NAME=X; TCAP=401; BRC={brc}; KIND={kind}\n'
                      'budget_fired () { [ -e "$OUT/BUDGET_HARDCAP_FIRED" ]; }\n' + snippet)
            r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            return (out / "boot_X.complete").exists(), (out / "CAPPED.txt").exists()

    def _cases(self, snippet):
        return {"budget_timeout": self._run_snippet(snippet, 124, "BUDGET", False),
                "budget_watchdog": self._run_snippet(snippet, 143, "P7_CAPPED", True),
                "fired_after_clean_bench": self._run_snippet(snippet, 0, "P7_CAPPED", True),
                "p7_capped": self._run_snippet(snippet, 124, "P7_CAPPED", False),
                "clean": self._run_snippet(snippet, 0, "P7_CAPPED", False)}

    def test_budget_cut_writes_no_complete(self):
        c = self._cases(self._snippet(RUNNER.read_text()))
        self.assertEqual(c["budget_timeout"], (False, False))
        self.assertEqual(c["budget_watchdog"][0], False)
        self.assertEqual(c["fired_after_clean_bench"][0], False)
        self.assertEqual(c["p7_capped"], (True, True))
        self.assertEqual(c["clean"], (True, False))

    def test_control_budget_guard_removed(self):
        src = RUNNER.read_text()
        m = src.replace('if [ "$CUT" != BUDGET ] && [ -s "$OUT/cell_${NAME}.json" ]; then',
                        'if [ -s "$OUT/cell_${NAME}.json" ]; then')
        self.assertNotEqual(m, src)
        self.assertTrue(self._cases(self._snippet(m))["budget_timeout"][0],
                        "mutant still withholds .complete -- the test is vacuous")

    # ---- (d) recommended: preflight steps bounded by the cell deadline, tail steps by the cap end
    PREFLIGHT = ('"$ENG/scripts/bootstrap/sync_engine_tree.sh" "$OUT/runtime_source_manifest.sha256"',
                 'python3 "$PREREG/e2_realized_mix.py" --selftest',
                 'python3 "$PREREG/e2_label.py" --selftest',
                 'python3 "$HELPER" --selftest',
                 'python3 -m unittest discover -s "$ENG/tests" -p "test_sticky_partition.py"',
                 'cp -p "$STAGE/$f" "$ARCHIVE/$f"')
    TAIL = ('python3 - "$G_OFF" "$G_ON"', 'python3 "$HELPER" --green-summary "$OUT"',
            'python3 "$PREREG/e2_label.py" --cells-dir "$OUT"')

    @classmethod
    def _bounded(cls, src):
        bad = []
        for cmd, fn in [(c, "cap_s") for c in cls.PREFLIGHT] + [(c, "tail_s") for c in cls.TAIL]:
            for m in re.finditer(re.escape(cmd), src):
                line = src[src.rfind("\n", 0, m.start()) + 1:m.start()]
                if f'timeout -k 10 "$({fn})" ' not in line:
                    bad.append(cmd)
        return bad

    def test_preflight_and_tail_bounded(self):
        self.assertEqual(self._bounded(RUNNER.read_text()), [])
        src = RUNNER.read_text()
        self.assertIn("cap_s () { local r; r=$(remaining)", src)
        self.assertIn("tail_s () { local r=$((T0 + HARD_CAP_S - $(date +%s)))", src)

    def test_control_unbounded_label_is_detected(self):
        src = RUNNER.read_text()
        m = src.replace('timeout -k 10 "$(tail_s)" python3 "$PREREG/e2_label.py"', 'python3 "$PREREG/e2_label.py"', 1)
        self.assertNotEqual(m, src)
        self.assertTrue(self._bounded(m))


class TestAdmissionPrecedesBoot(unittest.TestCase):
    @staticmethod
    def _order_ok(src):
        body = src[src.index('for U in "${UNITS[@]}"; do'):]
        try:
            adm = body.index('if [ "$REM" -lt "$NEED" ] || budget_fired; then')
        except ValueError:
            return False
        return adm < body.index('setsid "${SRV[@]}"') and 'NEED=$(( TCAP + P14_MARGIN_S ))' in body[:adm]

    def test_order(self):
        self.assertTrue(self._order_ok(RUNNER.read_text()))

    def test_control_removed_admission(self):
        src = RUNNER.read_text()
        mutant = re.sub(r'  if \[ "\$REM" -lt "\$NEED" \] \|\| budget_fired; then\n.*?\n  fi\n', "", src,
                        count=1, flags=re.S)
        self.assertNotEqual(mutant, src, "mutation did not apply")
        self.assertFalse(self._order_ok(mutant))


def _run(script: Path, td: str, job: str, **env):
    limit = env.pop("_t", 180)
    e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=f"{td}/data", PDMUX_IO=f"{td}/io",
                   PDMUX_JOB_NAME=job, PDMUX_E2V_TEST_OUT_ROOT=f"{td}/out", **env)
    return subprocess.run([BASH, str(script)], env=e, capture_output=True, text=True, timeout=limit)


class TestEnvAllowlist(unittest.TestCase):
    def test_inherited_arm_knob_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_env1", PDMUX_E2V_DRYRUN="1", PDMUX_STICKY_PARTITION="1")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ABORT_ENV unregistered knob PDMUX_STICKY_PARTITION", r.stdout)

    def test_phase_events_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_env2", PDMUX_E2V_DRYRUN="1", PDMUX_PHASE_EVENTS="1")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ABORT_ENV unregistered knob PDMUX_PHASE_EVENTS", r.stdout)

    def test_operator_prereg_knob_is_gone(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_env3", PDMUX_E2V_DRYRUN="1", PDMUX_E2_PREREG="/elsewhere.md")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)

    def test_test_knob_without_dryrun_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_env4", PDMUX_E2V_TEST_HARD_CAP_S="700")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", r.stdout)


@unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
class TestDryRun(unittest.TestCase):
    """Full dry run (no server): 20 boots written in registered order; the ONE knob
    differs between arms; green read-out on both; label sees all pairs missing."""

    def test_dry_run_commands(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_dry", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1")
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
            out = Path(td) / "out" / "e2v_t_dry"
            cmds = (out / "DRYRUN_COMMANDS.txt").read_text()
            boots = re.findall(r"^BOOT (\S+)$", cmds, re.M)
            self.assertEqual(boots, [f"{c}_{a}_{s}" for c, s, a in V.units()])
            envs = re.findall(r"env   : (.*)$", cmds, re.M)
            self.assertEqual(len(envs), 20)
            for b, e in zip(boots, envs):
                self.assertIn("PDMUX_GREEN_READOUT=1", e)
                self.assertIn("PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=44", e)
                want = "PDMUX_STICKY_PARTITION=1" if "_ON_" in b else "PDMUX_STICKY_PARTITION=<unset>"
                self.assertIn(want, e, b)
                cell = b.split("_O")[0]
                self.assertIn(f"T_cap={V.T_CAP_S[cell]}", e)
            self.assertIn("DIGESTS OK", (out / "DIGEST_CHECK.txt").read_text())
            self.assertIn("ARCHIVE OK", (out / "ARCHIVE_CHECK.txt").read_text())
            self.assertTrue((out / "DRYRUN_LABEL_ALL_MISSING.json").exists())
            self.assertGreaterEqual(len(re.findall(r"^[0-9a-f]{64} ", (out / "REGISTRATION_SHA256.txt").read_text(), re.M)), 10)


@unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
class TestBudgetEnforcer(unittest.TestCase):
    """Fake benches (DRYRUN) under the same timeout / watchdog / P14 machinery.
    cap 620 s - tail 600 s => deadline ~20 s after start.  A fake bench sleeps 60 s
    (< T_cap 401 s), so only the hard-cap layers can end it before ~65 s."""

    CAP, FAKE, LIMIT = "620", "60", 150

    def _go(self, script: Path):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.monotonic()
            try:
                r = _run(script, td, "t_budget", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                         PDMUX_E2V_TEST_HARD_CAP_S=self.CAP, PDMUX_E2V_TEST_FAKE_BENCH_S=self.FAKE,
                         _t=self.LIMIT)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "e2v_fake_bench"], check=False)
                subprocess.run(["pkill", "-f", "sleep 100000"], check=False)
                return None, time.monotonic() - t0, {}
            wall = time.monotonic() - t0
            out = Path(td) / "out" / "e2v_t_budget"
            files = {p.name: p.read_text() for p in out.glob("*.txt")}
            return r.returncode, wall, files

    def test_overrunning_bench_is_cut_and_nothing_follows(self):
        rc, wall, f = self._go(RUNNER)
        self.assertEqual(rc, 0, f)
        self.assertLess(wall, 50.0)
        fake = f.get("FAKE_BENCH.txt", "")
        self.assertRegex(fake, r"fake_rc=(124|143|137) boot=a_r4_OFF_4386 kind_if_cut=BUDGET")
        self.assertEqual(fake.count("fake_rc="), 1, fake)
        self.assertIn("UNRESOLVED_BUDGET", f.get("BUDGET_STOP.txt", ""))

    def test_control_both_kill_layers_removed_overruns(self):
        src = RUNNER.read_text()
        m = src.replace('timeout -k 2 "${BT}s" bash -c', 'bash -c')
        m = m.replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        self.assertEqual(m.count("sleep 100000"), 1)
        self.assertNotIn('timeout -k 2 "${BT}s" bash -c', m)
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e2_sticky_vessl_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._go(mut)
        self.assertGreaterEqual(wall, float(self.FAKE), "mutant was still cut -- the test is vacuous")

    def test_equivalent_mutant_one_layer_removed_still_cuts(self):
        m = RUNNER.read_text().replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e2_sticky_vessl_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._go(mut)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 50.0)


@unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
class TestResume(unittest.TestCase):
    """Pair granularity.  Control: different registration digests are refused; a pair
    with only ONE complete boot is re-run, not resumed."""

    def _prev(self, td, reg_text, units_text):
        p = Path(td) / "data/runs/prevjob/results/r2_eval/e2_sticky_vessl/e2v_prevjob"
        p.mkdir(parents=True)
        (Path(td) / "data/runs/prevjob/meta").mkdir(parents=True)
        (p / "REGISTRATION_SHA256.txt").write_text(reg_text)
        (p / "UNITS.txt").write_text(units_text)
        for b in ("a_r4_OFF_4386", "a_r4_ON_4386", "a_r4_OFF_4162"):     # pair 1 full, pair 2 half
            (p / f"boot_{b}.complete").write_text("x\n")
            (p / f"cell_{b}.json").write_text('{"empty": true}\n')
        (p / "BANNERS.txt").write_text("a_r4_OFF_4386 max_mamba_cache_size=48 max_total_num_tokens=2722025\n"
                                       "a_r4_ON_4386 max_mamba_cache_size=48 max_total_num_tokens=2722025\n")
        (Path(td) / "data/runs/prevjob/meta/substrate.json").write_text(
            '{"gpu": {"uuid": "GPU-prev", "driver_version": "580.105.08"}}')

    def test_resume_and_its_controls(self):
        with tempfile.TemporaryDirectory() as td:
            units = "\n".join(V.unit_rows()) + "\n"
            self._prev(td, "0" * 64 + "  e2_label.py\n", units)
            r = _run(RUNNER, td, "j1", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                     PDMUX_E2V_RESUME_FROM="prevjob")
            self.assertEqual(r.returncode, 2, r.stdout[-2000:])
            self.assertIn("ABORT_RESUME registration digests differ", r.stdout)
            reg = (Path(td) / "out" / "e2v_j1" / "REGISTRATION_SHA256.txt").read_text()
            shutil.rmtree(Path(td) / "data/runs/prevjob")
            self._prev(td, reg, units)
            r = _run(RUNNER, td, "j2", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                     PDMUX_E2V_RESUME_FROM="prevjob")
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
            o = Path(td) / "out" / "e2v_j2"
            self.assertIn("SKIP_RESUMED_PAIR a_r4_OFF_4386", r.stdout)
            self.assertIn("SKIP_RESUMED_PAIR a_r4_ON_4386", r.stdout)
            self.assertIn("a_r4/4386\tprevjob\tGPU-prev", (o / "RESUMED_PAIRS.tsv").read_text())
            self.assertNotIn("a_r4/4162", (o / "RESUMED_PAIRS.tsv").read_text())
            cmds = (o / "DRYRUN_COMMANDS.txt").read_text()
            self.assertNotIn("BOOT a_r4_OFF_4386\n", cmds)
            self.assertIn("BOOT a_r4_OFF_4162\n", cmds)        # half-complete pair re-runs whole
            self.assertIn("BOOT a_r4_ON_4162\n", cmds)
            self.assertTrue((o / "cell_a_r4_ON_4386.json").exists())
            self.assertFalse((o / "cell_a_r4_OFF_4162.json").exists())

    def test_control_units_differ_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r0 = _run(RUNNER, td, "j0", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1")
            reg = (Path(td) / "out" / "e2v_j0" / "REGISTRATION_SHA256.txt").read_text()
            self.assertEqual(r0.returncode, 0)
            self._prev(td, reg, "a_r4 4386 OFF A 256 512 8.00 400 393\n")
            r = _run(RUNNER, td, "j3", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                     PDMUX_E2V_RESUME_FROM="prevjob")
            self.assertEqual(r.returncode, 2)
            self.assertIn("ABORT_RESUME UNITS.txt differs", r.stdout)


if __name__ == "__main__":
    unittest.main()
