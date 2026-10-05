"""E2 sticky contrast, VESSL re-run (PREREG_E2_VESSL_A14_2026-10-05.md): runner r2 + helper r2.

The first Job's runner / helper / tests (e2_sticky_vessl.sh, e2_vessl_plan.py,
tests/test_e2_vessl.py) are SEALED and are not edited; this file tests only the r2 copy and
what A14 adds.  Every check carries a negative control that must make it fail (lesson 53).

  copy fidelity        r2's server / warm-up / bench / estimator / label / P13 text == the first
                       runner's (and the sbatch's flag multisets).  Control: one dropped flag.
  digests              A14 + r2 runner pinned in e2_vessl_plan_r2; the first Job's six sealed
                       files unchanged; the harness seal table of VERDICT rev2 sec 5 (7 of 8 rows;
                       the OVERRIDE row is the consumed record).  Control: one-byte drift reported.
  P3 (ii)              the first Job's keys (...025 x9, ...022) PASS; mamba 47/48, all-47, 0.8 %
                       token drift, < 786,432, banner missing FAIL -- in the helper AND through
                       the runner's real P3 snippet.  Control: a neutralised case branch passes
                       mamba 47 (detected); the first runner's literal branch aborts on the Job keys.
  O1 / O2 / O3 / O4    listen window outside ephemeral range (overlap -> ABORT_PORT_RANGE; the old
                       39100 window must abort); `ss -Htan` sees TIME_WAIT, old `ss -ltn` misses it;
                       teardown kills `^sglang::`; settle BEFORE the P14 admission, not a gate.
                       Control: the draft placement (settle after admission) is detected.
  kill layers          bench timeout, watchdog patterns, BUDGET cut writes no .complete, preflight
                       and tail bounded (TestRealPathKillLayers equivalent).  Controls as there.
  resume               PDMUX_E2V_RESUME_FROM -> ABORT_RESUME before any boot; no resume code left.
  dry run / budget     20 boots in rev3 order, both digest checks OK, A14 + r2 in REGISTRATION;
                       a fake bench that would overrun is cut.  Control: both kill layers removed.
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
RUNNER = HERE / "e2_sticky_vessl_r2.sh"
RUNNER_R1 = HERE / "e2_sticky_vessl.sh"
SBATCH = PREREG / "e2_sticky.sbatch"
ARCHIVE = ENG / "results" / "r2_eval" / "lambda0_prereg" / "lam0_908623"
JOB1 = HERE / "e2v_pdmux-e2-sticky-vessl-e6f424e-20261004135228"
BASH = "/usr/bin/bash"

sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import e2_vessl_plan as V1  # noqa: E402
import e2_vessl_plan_r2 as V2  # noqa: E402

HAVE_ARCHIVE = all((ARCHIVE / n).exists() for n in V1.ARCHIVE_DIGESTS)

# VERDICT_e2_vessl_rules_rev2_2026-10-04.md sec 5 (harness seal of the first Job), repo-relative.
# The 8th row (OVERRIDE_E2_VESSL_2026-10-04.md, 54a810ca...) was the declared exception: it is
# now the consumed record (131abfff..., commit 9a29f84) and is pinned in V2.SEALED_R1.
HARNESS_SEAL_R1 = {
    "workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh":
        "921dfd2df4c532ee2e6e0452533ca5a0cb09fd541daca0dcd4cda927af00a6c4",
    "workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_vessl_plan.py":
        "57220a46bc8e57d60f470d45accabdbdba5e99135435a0e19c06eb3c5354700f",
    "workspace/engine-port/tests/test_e2_vessl.py":
        "d163d7020cbe4d737904036b906d63be548529c452e84f924bca415b5a99207c",
    "workspace/engine-port/results/r2_eval/e2_sticky_vessl/PREREG_E2_VESSL_A13_2026-10-04.md":
        "82055fe17f1d438a80b48d9412a21e4d7d300d5f2ce1ed15b284762247a719c0",
    "workspace/engine-port/results/r2_eval/e2_sticky_vessl/stage_e2_inputs.sh":
        "4b27bb0ab53149bf8fb9cd0179da71e5b86b4569fe849e79f7c5a034642d54cb",
    "scripts/discipline/design_reachability.py":
        "7eb2c190d0153489d1d6632e841d2626977a830c82594200e0dd0c4706abbc55",
    "workspace/engine-port/scripts/discipline/presubmit.py":
        "17821c4944d2d56d3804cf838ed61b0fb9092402e5af38bd01099f50d161093e",
}
# Kept as records beside A14 (not executed): the draft, its two patches, the two verdicts.
RECORDS_R1 = {
    "PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md": "f0118ba1cab9baf52a979d9c7fc3afa6849098b73cde04364c5b1670a6cfd729",
    "A14_DRAFT_runner_ops_2026-10-04.patch": "8dd0fa933588b7e0424d38e8d0d69e549e9bff1212f43e42d2347b07b8cd4dd0",
    "A14_DRAFT_runner_p3_tolerance_2026-10-04.patch": "ab14627bead1e4975bf7930c9416202f85dd456071ad96cf0092e6cede25e84f",
    "VERDICT_e2_vessl_rules_2026-10-04.md": "37d52fc470fc6954acfe44fe72c52b816f2bdfe17a4a096a6c071af6070bbd20",
    "VERDICT_e2_vessl_rules_rev2_2026-10-04.md": "44ffb47e2113403f7573e2ce2e8c9037c4972992320fce994e32a2022a91c680",
}


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


def _runner_blocks(rv: str):
    return (_block(rv, "SRV=(python", "WARM=("), _block(rv, "WARM=(python", "BENCH=("),
            _block(rv, "BENCH=(python", 'if [ "$DRYRUN" = 1 ]'))


def _sbatch_blocks(sb: str):
    return (_block(sb, "python -m sglang.launch_server", "PID=$!"),
            _block(sb, "python -m sglang.bench_serving", "warmup_rc"),
            _block(sb, "python -m sglang.bench_serving", "bench_rc", after="warmup_rc"))


def _heredoc(t: str, marker: str) -> str:
    i = t.index(marker)
    return t[t.index("\n", i) + 1:t.index("\nPY\n", i)]


# ============================================================================ copy fidelity
class TestCopyFidelity(unittest.TestCase):
    def setUp(self):
        self.r2, self.r1 = RUNNER.read_text(), RUNNER_R1.read_text()

    def test_measurement_blocks_byte_identical_to_first_runner(self):
        for a, b in zip(_runner_blocks(self.r1), _runner_blocks(self.r2)):
            self.assertEqual(a, b)

    def test_flags_identical_to_sbatch(self):
        for a, b in zip(_sbatch_blocks(SBATCH.read_text()), _runner_blocks(self.r2)):
            self.assertEqual(_flags(a), _flags(b))

    def test_control_dropped_flag_is_detected(self):
        m = self.r2.replace('--seed "$CSEED" --output-details', '--seed "$CSEED"', 1)
        self.assertNotEqual(m, self.r2)
        self.assertTrue(any(_flags(a) != _flags(b) for a, b in
                            zip(_sbatch_blocks(SBATCH.read_text()), _runner_blocks(m))))

    def test_p13_bodies_byte_identical(self):
        for marker in ("<<'PY' | tee -a \"$OUT/P13_PROBE.txt\"", "<<'PY' | tee \"$OUT/P13_VERDICT.txt\""):
            self.assertEqual(_heredoc(SBATCH.read_text(), marker), _heredoc(self.r2, marker))

    def test_rev3_called_not_copied(self):
        for frag in ('python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH"',
                     'python3 "$PREREG/e2_label.py" --cells-dir "$OUT" --expect "$EXPECT"',
                     '"$PREREG/e2_realized_mix.py" --selftest --archive "$ARCHIVE"',
                     '"$PREREG/e2_label.py" --selftest'):
            self.assertIn(frag, self.r2)
            self.assertEqual(self.r1.count(frag), self.r2.count(frag), frag)
        est = lambda s: _block(s, 'python3 "$PREREG/e2_realized_mix.py" "$PDMUX_TELEMETRY_PATH"', "|| echo")  # noqa: E731
        self.assertEqual(est(self.r1), est(self.r2))
        txt = (HERE / "e2_vessl_plan_r2.py").read_text()
        for banned in ("def label(", "def compute(", "paired_bootstrap", "R1_OFF_BAND"):
            self.assertNotIn(banned, txt)
            self.assertNotIn(banned, self.r2)
        self.assertFalse(list(HERE.glob("e2_label*.py")) or list(HERE.glob("e2_realized_mix*.py")))

    def test_one_knob_and_constants(self):
        knob = 'if [ "$ARM" = "ON" ]; then export PDMUX_STICKY_PARTITION=1; else unset PDMUX_STICKY_PARTITION; fi'
        self.assertIn(knob, self.r2)
        self.assertEqual(self.r2.count("export PDMUX_STICKY_PARTITION"), 1)
        for key in ("MODEL", "BACKEND", "CTX", "MEM_FRACTION", "MAX_RUNNING", "SERVER_SEED", "FIXED_DSM", "CFG"):
            a = re.search(rf"^{key}=(\S+)", self.r1, re.M).group(1)
            b = re.search(rf"^{key}=(\S+)", self.r2, re.M).group(1)
            self.assertEqual(a, b, key)
        self.assertEqual(V2.P3_MAX_RUNNING, int(re.search(r"^MAX_RUNNING=(\d+)", self.r2, re.M).group(1)))
        self.assertEqual(V2.P3_CTX, int(re.search(r"^CTX=(\d+)", self.r2, re.M).group(1)))

    def test_hard_cap_and_units_from_sealed_helper(self):
        for frag in ('K () { python3 "$HELPER" --const "$1"; }', "HARD_CAP_S=$(K HARD_CAP_S)",
                     'mapfile -t UNITS < <(python3 "$HELPER" --units)', 'EXPECT=$(python3 "$HELPER" --expect)',
                     'HELPER="$HERE/e2_vessl_plan.py"', 'HELPER_R2="$HERE/e2_vessl_plan_r2.py"'):
            self.assertIn(frag, self.r2)
        self.assertEqual(V1.HARD_CAP_S, 14400)

    def test_checks_wired_to_aborts(self):
        for check, abort in (('python3 "$HELPER" --verify-digests', "ABORT_REG_DRIFT"),
                             ('python3 "$HELPER_R2" --verify-digests-r2', "ABORT_REG_DRIFT"),
                             ('python3 "$HELPER" --verify-archive', "ABORT_ARCHIVE_DRIFT"),
                             ('python3 "$HELPER" --engine-diff', "ABORT_ENGINE_TREE_DRIFT"),
                             ('python3 "$HELPER" --check-substrate', "ABORT_SUBSTRATE"),
                             ('python3 "$HELPER_R2" --selftest', "ABORT_SELFTEST"),
                             ('"$PREREG/e2_realized_mix.py" --selftest', "ABORT_SELFTEST")):
            i = self.r2.index(check)
            self.assertIn(abort, self.r2[i:i + 400], check)

    def test_registration_lists_a14_and_r2(self):
        reg = _block(self.r2, '  echo "# pre-registration in force', '} | tee "$OUT/REGISTRATION_SHA256.txt"')
        for frag in ('"$A14_MD"', '"$HERE/e2_sticky_vessl_r2.sh"', '"$HELPER_R2"', '"$HELPER"',
                     '"$HERE/e2_sticky_vessl.sh"', '"$HERE"/OVERRIDE_E2_VESSL_*.md', '"$A13_MD"'):
            self.assertIn(frag, reg)
        self.assertIn('A14_MD="$HERE/PREREG_E2_VESSL_A14_2026-10-05.md"', self.r2)
        self.assertEqual(V2.A14_NAME, "PREREG_E2_VESSL_A14_2026-10-05.md")


# ============================================================================ digests
class TestDigests(unittest.TestCase):
    def test_live_repo_passes_both_checks(self):
        self.assertEqual(V1.verify_digests(PREREG, HERE, ENG), [])
        self.assertEqual(V2.verify_r2(HERE, ENG), [])

    def test_a14_and_runner_pinned(self):
        self.assertEqual(V1.sha256(HERE / V2.A14_NAME), V2.A14_DIGEST,
                         "A14 edited without updating e2_vessl_plan_r2.A14_DIGEST")
        self.assertEqual(V1.sha256(RUNNER), V2.RUNNER_R2_DIGEST,
                         "runner r2 edited without updating e2_vessl_plan_r2.RUNNER_R2_DIGEST")

    def test_first_job_harness_seal_unchanged(self):
        for rel, want in HARNESS_SEAL_R1.items():
            self.assertEqual(V1.sha256(ROOT / rel), want, rel)
        for rel, want in V2.SEALED_R1.items():
            self.assertEqual(V1.sha256(ENG / rel), want, rel)
        for name, want in RECORDS_R1.items():
            self.assertEqual(V1.sha256(HERE / name), want, name)
        # the helper's sealed set agrees with the verdict's table where they overlap
        for rel, want in V2.SEALED_R1.items():
            k = "workspace/engine-port/" + rel
            if k in HARNESS_SEAL_R1:
                self.assertEqual(HARNESS_SEAL_R1[k], want, rel)

    def test_control_one_byte_drift_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            eng = Path(td) / "eng"
            here = eng / "results" / "r2_eval" / "e2_sticky_vessl"
            here.mkdir(parents=True)
            (eng / "tests").mkdir()
            for rel in V2.SEALED_R1:
                shutil.copy2(ENG / rel, eng / rel)
            for name in (V2.A14_NAME, V2.RUNNER_R2_NAME):
                shutil.copy2(HERE / name, here / name)
            self.assertEqual(V2.verify_r2(here, eng), [], "unmutated copy must pass (null control)")
            p = here / "e2_sticky_vessl.sh"                  # the SEALED first-Job runner
            p.write_text(p.read_text().replace("39100", "39101", 1))
            bad = V2.verify_r2(here, eng)
            self.assertTrue(any("sealed-r1" in b and "e2_sticky_vessl.sh" in b for b in bad), bad)
            q = here / V2.A14_NAME
            q.write_text(q.read_text() + " ")
            self.assertTrue(any(V2.A14_NAME in b for b in V2.verify_r2(here, eng)))


# ============================================================================ P3 (ii)
JOB1_ROWS = [["48", "2722025"]] * 9 + [["48", "2722022"]]


class TestP3Tolerance(unittest.TestCase):
    MUTANTS = {
        "mamba 47/48": [["48", "2722025"], ["47", "2722025"]],
        "mamba all 47": [["47", "2722025"]] * 3,
        "tokens 0.8% apart": [["48", "2722025"], ["48", str(round(2722025 * 0.992))]],
        "below 786432": [["48", "786431"]] * 2,
        "banner missing": [["48", "2722025"], ["NONE", "NONE"]],
    }

    def test_job1_keys_pass(self):
        rows = JOB1_ROWS
        if (JOB1 / "P3_KEYS.txt").exists():
            rows = [ln.split() for ln in (JOB1 / "P3_KEYS.txt").read_text().splitlines() if ln.strip()]
            self.assertEqual(rows, JOB1_ROWS)
        ok, line = V2.p3_verdict(rows, 48, 16384)
        self.assertTrue(ok, line)
        self.assertIn("rel=1.102e-06", line)

    def test_mutants_fail(self):
        for name, rows in self.MUTANTS.items():
            ok, line = V2.p3_verdict(rows, 48, 16384)
            self.assertFalse(ok, name)
            self.assertTrue(line.startswith("FAIL"), (name, line))

    def test_constants(self):
        self.assertEqual((V2.P3_MAMBA_EXPECT, V2.P3_TOKEN_FLOOR, V2.P3_REL_TOL), (48, 786432, 1e-4))
        self.assertTrue(V2.p3_verdict([["48", "786432"]], 48, 16384)[0])
        self.assertFalse(V2.p3_verdict(JOB1_ROWS, 48, 8192)[0])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k.txt"
            p.write_text("".join(f"{a} {b}\n" for a, b in JOB1_ROWS))
            r = subprocess.run([sys.executable, str(HERE / "e2_vessl_plan_r2.py"), "--p3-verdict", str(p), "48", "16384"],
                               capture_output=True, text=True, env=_clean_env())
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(r.stdout.startswith("PASS "))
            p.write_text("48 2722025\n47 2722025\n")
            r = subprocess.run([sys.executable, str(HERE / "e2_vessl_plan_r2.py"), "--p3-verdict", str(p), "48", "16384"],
                               capture_output=True, text=True, env=_clean_env())
            self.assertEqual(r.returncode, 8)
            self.assertTrue(r.stdout.startswith("FAIL "))

    # ---- the runner's REAL P3 snippet, executed with stubs (behavioural, not textual)
    @staticmethod
    def _snippet(src):
        a = src.index('  MMB=$(grep -o "max_mamba_cache_size')
        b = src.index("  esac\n", a) + len("  esac\n")
        return src[a:b]

    def _run(self, snippet, prior_rows, banner, helper=None):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "P3_KEYS.txt").write_text("".join(f"{a} {b}\n" for a, b in prior_rows))
            (out / "srv.log").write_text(banner)
            h = helper or str(HERE / "e2_vessl_plan_r2.py")
            script = (f'OUT="{td}"; NAME=X; SRVLOG="{td}/srv.log"; HELPER_R2="{h}"; MAX_RUNNING=48; CTX=16384; PID=1; PORT=1\n'
                      'teardown () { echo TEARDOWN >> "$OUT/td.txt"; }\n'
                      'fail () { echo "$1"; exit "$2"; }\n' + snippet + "\necho P3_PASSED\n")
            r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60, env=_clean_env())
            td_called = (out / "td.txt").exists()
            verdicts = (out / "P3_VERDICTS.txt").read_text() if (out / "P3_VERDICTS.txt").exists() else ""
            return r.returncode, r.stdout, td_called, verdicts

    BANNER = "x max_mamba_cache_size: {m} y\nz max_total_num_tokens={t} w\n"

    def test_runner_snippet(self):
        snip = self._snippet(RUNNER.read_text())
        rc, out, td, v = self._run(snip, JOB1_ROWS[:9], self.BANNER.format(m=48, t=2722022))
        self.assertEqual(rc, 0, out)
        self.assertIn("P3_PASSED", out)
        self.assertIn("X P3 rc=0 PASS ", v)
        self.assertFalse(td)
        for name, (m, t) in {"mamba 47": (47, 2722025), "0.8% drift": (48, round(2722025 * 0.992)),
                             "below floor": (48, 786431)}.items():
            rc, out, td, v = self._run(snip, JOB1_ROWS[:9], self.BANNER.format(m=m, t=t))
            self.assertEqual(rc, 8, (name, out))
            self.assertIn("ABORT_P3", out)
            self.assertTrue(td, name)                        # server torn down before exit
            self.assertIn("X P3 rc=8 FAIL", v)
        rc, out, td, v = self._run(snip, JOB1_ROWS[:9], "no banner here\n")       # banner missing
        self.assertEqual(rc, 8, out)
        rc, out, td, v = self._run(snip, JOB1_ROWS[:9], self.BANNER.format(m=48, t=2722025),
                                   helper="/nonexistent/e2_vessl_plan_r2.py")    # fail-closed
        self.assertEqual(rc, 8, out)
        with tempfile.TemporaryDirectory() as td:
            liar = Path(td) / "liar.py"                     # prints PASS but exits non-zero
            liar.write_text('import sys\nprint("PASS fake")\nsys.exit(1)\n')
            noisy = Path(td) / "noisy.py"                   # warning on stderr, then a real PASS
            noisy.write_text('import sys\nprint("warning: x", file=sys.stderr)\nprint("PASS ok")\n')
            rc, out, _, _ = self._run(snip, JOB1_ROWS[:9], self.BANNER.format(m=48, t=2722025), helper=str(liar))
            self.assertEqual(rc, 8, out)
            rc, out, _, _ = self._run(snip, JOB1_ROWS[:9], self.BANNER.format(m=48, t=2722025), helper=str(noisy))
            self.assertEqual(rc, 0, out)

    def test_control_neutralised_branch_is_detected(self):
        src = RUNNER.read_text()
        m = src.replace('    0:PASS\\ *) ;;\n    *) teardown', '    *) ;;\n    NEVER) teardown', 1)
        self.assertNotEqual(m, src, "mutation did not apply")
        rc, out, _, _ = self._run(self._snippet(m), JOB1_ROWS[:9], self.BANNER.format(m=47, t=2722025))
        self.assertEqual(rc, 0, "neutralised P3 still aborted -- the test is vacuous")

    def test_first_runner_literal_branch_aborts_on_job1_keys(self):
        # what (ii) changes: the sealed first-Job branch aborts on exactly these keys
        src = RUNNER_R1.read_text()
        a = src.index('  MMB=$(grep -o "max_mamba_cache_size')
        b = src.index("  fi\n", src.index('if [ "$(sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]', a)) + 5
        rc, out, _, _ = self._run(src[a:b], JOB1_ROWS[:9], self.BANNER.format(m=48, t=2722022))
        self.assertEqual(rc, 8, out)
        self.assertNotIn('sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]; then', RUNNER.read_text())


# ============================================================================ O1-O4
class TestPortAndTeardown(unittest.TestCase):
    def setUp(self):
        self.src = RUNNER.read_text()

    def _fake_ss(self, td, port):
        ss = Path(td) / "ss"
        ss.write_text("#!/bin/bash\n"
                      "case \"$*\" in\n"
                      f"  *-Htan*) echo 'TIME-WAIT 0 0 127.0.0.1:{port} 127.0.0.1:39001';;\n"
                      "  *-ltn*) echo 'State Recv-Q Send-Q Local Address:Port Peer Address:Port';"
                      " echo 'LISTEN 0 4096 127.0.0.1:22 0.0.0.0:*';;\n"
                      "esac\n")
        ss.chmod(0o755)
        return {**_clean_env(), "PATH": f"{td}:{os.environ['PATH']}"}

    def _port_in_use_fn(self):
        i = self.src.index("port_in_use () {")
        return self.src[i:self.src.index("\n}\n", i) + 3]

    def test_o2_new_check_sees_time_wait_old_check_misses_it(self):
        fn = self._port_in_use_fn()
        with tempfile.TemporaryDirectory() as td:
            env = self._fake_ss(td, 20123)
            self.assertEqual(subprocess.run([BASH, "-c", f"{fn}\nport_in_use 20123"], env=env).returncode, 0,
                             "new check must report the TIME_WAIT port as in use")
            self.assertEqual(subprocess.run([BASH, "-c", f"{fn}\nport_in_use 20124"], env=env).returncode, 1,
                             "an unrelated port must be free")
            self.assertEqual(subprocess.run([BASH, "-c", 'ss -ltn 2>/dev/null | grep -q ":20123 "'], env=env).returncode, 1,
                             "control: the first runner's LISTEN-only check misses TIME_WAIT")
            # control on the REAL loop text: replacing the new check by the old one passes the port
            old_fn = "port_in_use () { ss -ltn 2>/dev/null | grep -q \":$1 \"; }\n"
            self.assertEqual(subprocess.run([BASH, "-c", f"{old_fn}\nport_in_use 20123"], env=env).returncode, 1)
        code = "\n".join(ln for ln in self.src.splitlines() if not ln.lstrip().startswith("#"))
        self.assertNotIn("ss -ltn", code)                    # comments may still explain the old check
        self.assertIn('if ! port_in_use "$CAND"; then PORT=$CAND; break; fi', self.src)

    def _range_block(self, src):
        i = src.index("PORT_LO=")
        return src[i:src.index("\nfi\n", i) + 4]

    def _run_range(self, block, lo, hi):
        with tempfile.TemporaryDirectory() as td:
            rng = Path(td) / "range"
            rng.write_text(f"{lo}\t{hi}\n")
            b = block.replace("/proc/sys/net/ipv4/ip_local_port_range", str(rng))
            r = subprocess.run([BASH, "-c", f'OUT={td}\nfail () {{ echo "$1"; exit "$2"; }}\n{b}\necho RANGE_OK\n'],
                               capture_output=True, text=True)
            return r.returncode, r.stdout

    def test_o1_window_outside_ephemeral_and_overlap_aborts(self):
        block = self._range_block(self.src)
        self.assertIn("PORT_LO=20000; PORT_SPAN=800", block)
        rc, out = self._run_range(block, 32768, 60999)
        self.assertEqual(rc, 0, out)
        rc, out = self._run_range(block, 10000, 30000)
        self.assertEqual(rc, 5, out)
        self.assertIn("ABORT_PORT_RANGE", out)
        rc, out = self._run_range(block, 20799, 30000)       # one-port overlap at the top edge
        self.assertEqual(rc, 5, out)
        old = block.replace("PORT_LO=20000", "PORT_LO=39100")  # control: the first runner's window
        rc, out = self._run_range(old, 32768, 60999)
        self.assertEqual(rc, 5, out)
        with tempfile.TemporaryDirectory() as td:            # unreadable range -> abort
            b = block.replace("/proc/sys/net/ipv4/ip_local_port_range", f"{td}/absent")
            r = subprocess.run([BASH, "-c", f'OUT={td}\nfail () {{ echo "$1"; exit "$2"; }}\n{b}\necho RANGE_OK\n'],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 5, r.stdout)
        self.assertNotIn("39100 +", self.src)
        self.assertIn("CAND=$((PORT_LO + (RANDOM + $$ + try) % PORT_SPAN))", self.src)

    def test_o3_teardown_kills_renamed_children_after_inherited_sequence(self):
        i = self.src.index("teardown () {")
        body = self.src[i:self.src.index("\n}\n", i)]
        a = body.index('pkill -9 -f "launch_server.*--port $2" 2>/dev/null; sleep 10')
        b = body.index('pkill -9 -f "^sglang::"')
        self.assertLess(a, b)
        self.assertIn('kill "$1" 2>/dev/null; sleep 5', body)
        self.assertTrue(re.search(r"^sglang::", "sglang::scheduler"))

    @staticmethod
    def _settle_before_admission(src):
        body = src[src.index('for U in "${UNITS[@]}"; do'):]
        try:
            s = body.index('[ "$DRYRUN" = 1 ] || settle "$NAME"')
        except ValueError:
            return False
        return s < body.index('if [ "$REM" -lt "$NEED" ] || budget_fired; then') < body.index("PORT=0")

    def test_o4_settle_before_admission_and_not_a_gate(self):
        self.assertTrue(self._settle_before_admission(self.src))
        i = self.src.index("settle () {")
        settle = self.src[i:self.src.index("\n}\n", i)]
        self.assertNotIn("fail ", settle)
        self.assertNotIn("exit", settle)
        self.assertIn("seq 1 36", settle)
        self.assertIn('>> "$OUT/SETTLE.txt"', settle)
        self.assertEqual(self.src.count('settle "$NAME"'), 1)

    def test_control_draft_placement_is_detected(self):
        # the draft patch put settle AFTER admission (inside P14's 200 s margin)
        m = self.src.replace('  [ "$DRYRUN" = 1 ] || settle "$NAME"\n', "", 1)
        m = m.replace("  PORT=0\n  for try", '  [ "$DRYRUN" = 1 ] || settle "$NAME"\n  PORT=0\n  for try', 1)
        self.assertNotEqual(m, self.src)
        self.assertFalse(self._settle_before_admission(m))


# ============================================================================ kill layers (real path)
class TestRealPathKillLayers(unittest.TestCase):
    BENCH_LINE = 'timeout -k 30 "${BT}s" "${BENCH[@]}"'

    @classmethod
    def _bench_wrapped(cls, src):
        real = src[src.index("  : > \"$PDMUX_TELEMETRY_PATH\"\n  setsid"):]
        return cls.BENCH_LINE in real and 'BT=$TCAP; KIND=P7_CAPPED' in real

    def test_bench_timeout_and_control(self):
        src = RUNNER.read_text()
        self.assertTrue(self._bench_wrapped(src))
        m = src.replace(self.BENCH_LINE, '"${BENCH[@]}"')
        self.assertNotEqual(m, src)
        self.assertFalse(self._bench_wrapped(m))

    @staticmethod
    def _watchdog_ok(src):
        i = src.index('while [ "$(remaining)" -gt 0 ]; do sleep 5; done')
        wd = src[i:src.index("WD_PID=$!", i)]
        pats = {sig: re.findall(rf'pkill -{sig} -f "([^"]+)"', wd) for sig in ("TERM", "9")}
        srv = " ".join(_block(src, "SRV=(python", "WARM=(").split()[0:3]).replace("SRV=(", "")
        bench = " ".join(_block(src, "BENCH=(python", 'if [ "$DRYRUN" = 1 ]').split()[0:3]).replace("BENCH=(", "")
        return all(any(re.search(p, cmd) for p in pats[sig]) for sig in ("TERM", "9") for cmd in (srv, bench))

    def test_watchdog_and_controls(self):
        src = RUNNER.read_text()
        self.assertTrue(self._watchdog_ok(src))
        m = src.replace('pkill -TERM -f "sglang.launch_server"', 'pkill -TERM -f "sglang.launch-server"')
        self.assertNotEqual(m, src)
        self.assertFalse(self._watchdog_ok(m))

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

    def test_budget_cut_writes_no_complete_and_control(self):
        src = RUNNER.read_text()
        s = self._snippet(src)
        self.assertEqual(self._run_snippet(s, 124, "BUDGET", False), (False, False))
        self.assertEqual(self._run_snippet(s, 143, "P7_CAPPED", True)[0], False)
        self.assertEqual(self._run_snippet(s, 124, "P7_CAPPED", False), (True, True))
        self.assertEqual(self._run_snippet(s, 0, "P7_CAPPED", False), (True, False))
        m = src.replace('if [ "$CUT" != BUDGET ] && [ -s "$OUT/cell_${NAME}.json" ]; then',
                        'if [ -s "$OUT/cell_${NAME}.json" ]; then')
        self.assertNotEqual(m, src)
        self.assertTrue(self._run_snippet(self._snippet(m), 124, "BUDGET", False)[0])

    PREFLIGHT = ('"$ENG/scripts/bootstrap/sync_engine_tree.sh" "$OUT/runtime_source_manifest.sha256"',
                 'python3 "$PREREG/e2_realized_mix.py" --selftest', 'python3 "$PREREG/e2_label.py" --selftest',
                 'python3 "$HELPER" --selftest', 'python3 "$HELPER_R2" --selftest',
                 'python3 -m unittest discover -s "$ENG/tests" -p "test_sticky_partition.py"',
                 'cp -p "$STAGE/$f" "$ARCHIVE/$f"')
    TAIL = ('python3 - "$G_OFF" "$G_ON"', 'python3 "$HELPER" --green-summary "$OUT"',
            'python3 "$PREREG/e2_label.py" --cells-dir "$OUT"')

    @classmethod
    def _bounded(cls, src):
        bad = []
        for cmd, fn in [(c, "cap_s") for c in cls.PREFLIGHT] + [(c, "tail_s") for c in cls.TAIL]:
            hits = list(re.finditer(re.escape(cmd), src))
            if not hits:
                bad.append("MISSING " + cmd)
            for m in hits:
                line = src[src.rfind("\n", 0, m.start()) + 1:m.start()]
                if f'timeout -k 10 "$({fn})" ' not in line:
                    bad.append(cmd)
        return bad

    def test_preflight_and_tail_bounded_and_control(self):
        src = RUNNER.read_text()
        self.assertEqual(self._bounded(src), [])
        m = src.replace('timeout -k 10 "$(cap_s)" python3 "$HELPER_R2" --selftest', 'python3 "$HELPER_R2" --selftest', 1)
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

    def test_order_and_control(self):
        src = RUNNER.read_text()
        self.assertTrue(self._order_ok(src))
        m = re.sub(r'  if \[ "\$REM" -lt "\$NEED" \] \|\| budget_fired; then\n.*?\n  fi\n', "", src, count=1, flags=re.S)
        self.assertNotEqual(m, src)
        self.assertFalse(self._order_ok(m))


# ============================================================================ runner executions
def _run(script: Path, td: str, job: str, **env):
    limit = env.pop("_t", 180)
    e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=f"{td}/data", PDMUX_IO=f"{td}/io",
                   PDMUX_JOB_NAME=job, PDMUX_E2V_TEST_OUT_ROOT=f"{td}/out", **env)
    return subprocess.run([BASH, str(script)], env=e, capture_output=True, text=True, timeout=limit)


class TestResumeRefused(unittest.TestCase):
    def test_resume_aborts_before_any_boot(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_res", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                     PDMUX_E2V_RESUME_FROM="pdmux-e2-sticky-vessl-e6f424e-20261004135228")
            self.assertEqual(r.returncode, 2, r.stdout[-2000:])
            self.assertIn("ABORT_RESUME", r.stdout)
            out = Path(td) / "out" / "e2v_t_res"
            self.assertFalse((out / "DRYRUN_COMMANDS.txt").exists())
            self.assertFalse((out / "REGISTRATION_SHA256.txt").exists())   # aborted before registration

    def test_no_resume_code_left(self):
        src = RUNNER.read_text()
        for gone in ("RESUMED[", "RESUMED_PAIRS", "greedy_OFF_from_", "SKIP_RESUMED_PAIR", 'PREV="$DATA/runs/'):
            self.assertNotIn(gone, src)
        i = src.index('[ -n "${PDMUX_E2V_RESUME_FROM:-}" ]')
        self.assertIn("ABORT_RESUME", src[i:i + 200])
        self.assertLess(i, src.index('python3 "$HELPER" --verify-digests'))

    @unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
    def test_control_without_the_guard_resume_is_ignored(self):
        # lesson 53: with the guard removed the runner runs all 20 boots (it has no resume code
        # left), so the ABORT above is the guard's doing, not a side effect of something else.
        src = RUNNER.read_text()
        m = re.sub(r'\[ -n "\$\{PDMUX_E2V_RESUME_FROM:-\}" \] \\\n  && fail "ABORT_RESUME[^\n]*\n', "", src, count=1)
        self.assertNotEqual(m, src, "mutation did not apply")
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e2_sticky_vessl_r2_mutant.sh"
            mut.write_text(m)
            r = _run(mut, td, "t_res_mut", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                     PDMUX_E2V_RESUME_FROM="pdmux-e2-sticky-vessl-e6f424e-20261004135228")
            self.assertEqual(r.returncode, 0, r.stdout[-2000:])
            cmds = (Path(td) / "out" / "e2v_t_res_mut" / "DRYRUN_COMMANDS.txt").read_text()
            self.assertEqual(len(re.findall(r"^BOOT ", cmds, re.M)), 20)


class TestEnvAllowlist(unittest.TestCase):
    def test_inherited_arm_knob_and_phase_events_abort(self):
        for knob in ("PDMUX_STICKY_PARTITION", "PDMUX_PHASE_EVENTS"):
            with tempfile.TemporaryDirectory() as td:
                r = _run(RUNNER, td, "t_env", PDMUX_E2V_DRYRUN="1", **{knob: "1"})
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn(f"ABORT_ENV unregistered knob {knob}", r.stdout)

    def test_test_knob_without_dryrun_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_env4", PDMUX_E2V_TEST_HARD_CAP_S="700")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", r.stdout)


@unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
class TestDryRun(unittest.TestCase):
    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run(RUNNER, td, "t_dry", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1")
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
            out = Path(td) / "out" / "e2v_t_dry"
            cmds = (out / "DRYRUN_COMMANDS.txt").read_text()
            boots = re.findall(r"^BOOT (\S+)$", cmds, re.M)
            self.assertEqual(boots, [f"{c}_{a}_{s}" for c, s, a in V1.units()])
            envs = re.findall(r"env   : (.*)$", cmds, re.M)
            self.assertEqual(len(envs), 20)
            for b, e in zip(boots, envs):
                self.assertIn("PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM=44", e)
                want = "PDMUX_STICKY_PARTITION=1" if "_ON_" in b else "PDMUX_STICKY_PARTITION=<unset>"
                self.assertIn(want, e, b)
                self.assertIn(f"T_cap={V1.T_CAP_S[b.split('_O')[0]]}", e)
            dig = (out / "DIGEST_CHECK.txt").read_text()
            self.assertIn("DIGESTS OK", dig)
            self.assertIn("DIGESTS_R2 OK", dig)
            reg = (out / "REGISTRATION_SHA256.txt").read_text()
            for name in (V2.A14_NAME, "e2_sticky_vessl_r2.sh", "e2_vessl_plan_r2.py", "e2_sticky_vessl.sh",
                         "PREREG_E2_VESSL_A13_2026-10-04.md"):
                self.assertIn(name, reg)
            self.assertIn(f"{V2.A14_DIGEST}  ", reg)
            self.assertIn(f"{V2.RUNNER_R2_DIGEST}  ", reg)
            bt = (out / "BUDGET_TABLE.json").read_text()
            for k in ("observed", "worst_settle", "capped"):
                self.assertIn(f'"{k}"', bt)
            pr = (out / "PORT_RANGE.txt").read_text()
            self.assertIn("listen_port_window=20000-20799", pr)
            self.assertIn("ip_local_port_range=", pr)
            self.assertTrue((out / "SETTLE.txt").read_text().startswith("baseline "))
            self.assertTrue((out / "DRYRUN_LABEL_ALL_MISSING.json").exists())


@unittest.skipUnless(HAVE_ARCHIVE, "git-ignored job 908623 telemetry absent")
class TestBudgetEnforcer(unittest.TestCase):
    CAP, FAKE, LIMIT = "620", "60", 150

    def _go(self, script: Path):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.monotonic()
            try:
                r = _run(script, td, "t_budget", PDMUX_E2V_DRYRUN="1", PDMUX_E2V_TEST_SKIP_PREFLIGHT="1",
                         PDMUX_E2V_TEST_HARD_CAP_S=self.CAP, PDMUX_E2V_TEST_FAKE_BENCH_S=self.FAKE, _t=self.LIMIT)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "e2v_fake_bench"], check=False)
                subprocess.run(["pkill", "-f", "sleep 100000"], check=False)
                return None, time.monotonic() - t0, {}
            wall = time.monotonic() - t0
            out = Path(td) / "out" / "e2v_t_budget"
            return r.returncode, wall, {p.name: p.read_text() for p in out.glob("*.txt")}

    def test_overrunning_bench_is_cut_and_nothing_follows(self):
        rc, wall, f = self._go(RUNNER)
        self.assertEqual(rc, 0, f)
        self.assertLess(wall, 50.0)
        fake = f.get("FAKE_BENCH.txt", "")
        self.assertRegex(fake, r"fake_rc=(124|143|137) boot=a_r4_OFF_4386 kind_if_cut=BUDGET")
        self.assertEqual(fake.count("fake_rc="), 1, fake)
        self.assertIn("UNRESOLVED_BUDGET", f.get("BUDGET_STOP.txt", ""))

    def test_control_both_kill_layers_removed_overruns(self):
        # The mutant lives in a temp dir, but HERE/ENG come from PDMUX_PROJECT_ROOT, so the
        # r2 digest check (which pins THIS runner) is not tripped by the mutant's own bytes.
        m = RUNNER.read_text().replace('timeout -k 2 "${BT}s" bash -c', 'bash -c')
        m = m.replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        self.assertEqual(m.count("sleep 100000"), 1)
        with tempfile.TemporaryDirectory() as td:
            mut = Path(td) / "e2_sticky_vessl_r2_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._go(mut)
        self.assertGreaterEqual(wall, float(self.FAKE), "mutant was still cut -- the test is vacuous")


class TestHelperR2(unittest.TestCase):
    def test_selftest(self):
        V2.selftest()

    def test_budget_layers(self):
        b = V2.budget_table()["scenarios"]
        self.assertEqual((b["normal"]["gpu_h"], b["normal"]["usd"]), (1.991, 2.95))
        self.assertEqual((b["observed"]["gpu_h"], b["observed"]["usd"]), (2.242, 3.32))
        self.assertEqual((b["worst"]["gpu_h"], b["worst"]["usd"]), (2.533, 3.75))
        self.assertEqual((b["worst_settle"]["gpu_h"], b["worst_settle"]["usd"]), (3.533, 5.23))
        for c in ("normal", "observed", "worst", "worst_settle"):
            self.assertEqual(b[c]["boots_admitted"], 20, c)
        self.assertLess(b["capped"]["boots_admitted"], 20)          # P14 still reachable
        self.assertEqual(V1.HARD_CAP_S / 3600.0, 4.0)

    def test_registered_corners_identical_to_sealed_helper(self):
        for c in ("normal", "worst", "capped"):
            self.assertEqual(V1.simulate(c)[0], V2.simulate_r2(c)[0], c)
            self.assertAlmostEqual(V1.simulate(c)[1], V2.simulate_r2(c)[1], places=6)


if __name__ == "__main__":
    unittest.main()
