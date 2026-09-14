"""`r2_correctness.sbatch`: the D-none control arm (`DN`) is runnable, carries
exactly one moved axis, and is INVISIBLE to the scorer.

WHY THIS EXISTS
---------------
Job 908179 PASSed on the repaired engine, so "the worker grad-guard repair fixed
the true-dual OOM" is `PLAUSIBLE(conditional)` and no more
(`results/r2_correctness/audit_908179_2026-09-14/VERDICT.md` sec 8.5).  The only
experiment that can raise it is a control arm that moves
`PDMUX_WORKER_GRAD_GUARD=none` and NOTHING else.  That arm was UNRUNNABLE:

  (i)  the harness unsets every `PDMUX_*` variable before booting, so
       `PDMUX_WORKER_GRAD_GUARD=none sbatch ...` is silently discarded;
  (ii) the scorer takes its label list from `boots.txt` and splits arms by label
       prefix (`r2_correctness_check.py:366`, then `startswith("TD")`), so a
       `none` boot entered as a SCORED boot joins the TD arm -- and since that
       boot is PREDICTED to crash, the gate would read `FAIL` BY CONSTRUCTION.

WHAT IS PINNED HERE
-------------------
1.  ★`DN` never reaches `boots.txt`.  Its label goes to `diag_boots.txt`, a name
    the scorer does not open, and the scorer has no glob/listdir/walk to find it
    another way (that half is pinned by `test_r2_correctness_instrument.py`).
    This is the STRUCTURAL guarantee that a predicted OOM cannot colour the gate
    label; it is not a promise written in prose.
2.  A `DN` boot's launch environment is the SCORED `TD` boot's environment plus
    exactly one entry, `PDMUX_WORKER_GRAD_GUARD=<R2C_GUARD>` -- same policy, same
    D, same telemetry path rule, same green read-out, same allocator telemetry,
    same run/workload ids -- and it receives the SAME client command (same seed,
    `--n-probes 8`).
3.  `R2C_GUARD` is FAIL-CLOSED: an unlisted value is `exit 2`, and a `DN` boot
    with no `R2C_GUARD` is `exit 2`.  No silent default (gate #6 / #237): the
    engine maps an empty guard name to the DEFAULT `inference_mode`
    (`multiplexing_mixin.py:136-138`), so a tolerated empty value would produce a
    boot that silently is not the `none` arm it claims to be.
4.  ★THE DEFAULT PATH IS BYTE-IDENTICAL TO THE HARNESS JOB 908179 RAN.  Not
    against a hand-written expectation (which would be an identity), but against
    the boot loop of commit `83d8cb9d…`, whose blob sha256
    `f39b167b…ac0403` is the value recorded in `job_908179/provenance.txt`.  The
    frozen fragment is carried below and both versions are EXECUTED under the
    same stubs; every launch argv, the client argv and `boots.txt` must match.
5.  A dead `DN` server does not stop the job: the loop continues to the next
    boot and the block still exits 0, so the verdict step runs.
6.  ★NEGATIVE CONTROLS (lesson #53).  Two mutants are REQUIRED to fail:
      (a) the guard re-export removed from the `DN` branch;
      (b) the `DN` label written to `boots.txt`.
    Plus: a `DN` branch made scored, and the fail-closed `exit 2` removed.

Nothing here boots a server or touches a GPU: the two blocks are extracted
verbatim from the sbatch between their marker comments and executed with `env`,
`server_cmd`, `wait_health`, `stop_server`, `kill` and `timeout` stubbed.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRACK_ROOT.parents[1]
SBATCH = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness.sbatch"
CHECKER = TRACK_ROOT / "results" / "r2_correctness" / "r2_correctness_check.py"

KBEGIN = "# --- BEGIN r2c grad-guard knob"
KEND = "# --- END r2c grad-guard knob ---"
LBEGIN = "# --- BEGIN r2c boot loop"
LEND = "# --- END r2c boot loop ---"

# The harness job 908179 ran: commit 83d8cb9d…, whose blob of this file hashes to
# f39b167b… -- the value job_908179/provenance.txt records.  Both are checked
# against git below, so the frozen fragment cannot drift into fiction.
LEGACY_COMMIT = "83d8cb9d217545da56cd6e3612108e7c560f7a71"
LEGACY_SBATCH_SHA256 = (
    "f39b167bb6bd713af168abfbe3495caee099a6c0087e41b05e70b9a8d8ac0403"
)
SBATCH_REL = "workspace/engine-port/results/r2_correctness/r2_correctness.sbatch"

# ★The 908179 boot loop, VERBATIM.  Do not tidy: it is a fixture, and any edit
# to it silently weakens test_default_path_is_byte_identical_to_908179.
LEGACY_LOOP = r""": > "$OUT/boots.txt"
n_l=0; n_td=0; i=0
for ARM in $ORDER; do
  i=$((i + 1))
  case "$ARM" in
    L)  n_l=$((n_l + 1)); LABEL="L$n_l"; TD_ENV="" ;;
    TD) n_td=$((n_td + 1)); LABEL="TD$n_td"; TD_ENV="PDMUX_TRUE_DUAL_WORKER=1" ;;
    *)  echo "bad arm $ARM"; continue ;;
  esac
  PORT=$((BASEPORT + i))
  echo ""
  echo "########## boot=$LABEL arm=$ARM port=$PORT ##########"
  echo -n "$LABEL " >> "$OUT/boots.txt"
  env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" \
    PDMUX_TELEMETRY_PATH="$OUT/tel_${LABEL}.jsonl" \
    PDMUX_RUN_ID="r2corr_${SLURM_JOB_ID:-local}_${LABEL}" \
    PDMUX_WORKLOAD_ID="r2corr_seed${SEED}" \
    PDMUX_TRACE_FORCE_PREFILL="$TFP" PDMUX_ENGINE_COMMIT="$COMMIT" \
    PDMUX_MEM_TELEMETRY="$MEMTEL" \
    PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$OUT/green_${LABEL}.json" $TD_ENV \
    $(server_cmd "$PORT") > "$OUT/srv_${LABEL}.log" 2>&1 &
  PID=$!
  if ! wait_health "$PORT" "$PID" 96; then
    echo "BOOT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
    grep -n "Traceback\|Error" "$OUT/srv_${LABEL}.log" | tail -5
    stop_server "$PORT" "$PID"
    continue
  fi
  grep -oE "disable_cuda_graph=[A-Za-z]+|random_seed=[0-9]+|enable_pdmux=[A-Za-z]+" \
    "$OUT/srv_${LABEL}.log" | sort -u | tr '\n' ' '; echo
  grep -m1 "PD-Multiplexing enabled" "$OUT/srv_${LABEL}.log"
  grep -m1 "true dual-worker enabled" "$OUT/srv_${LABEL}.log" || echo "  (legacy loop: no true-dual banner)"
  timeout 900 python3 "$HERE/r2_correctness_client.py" --port "$PORT" \
    --boot "$LABEL" --out "$OUT/gen_${LABEL}.json" --n-probes 8 \
    || echo "CLIENT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  kill -0 "$PID" 2>/dev/null || echo "SERVER_DIED_DURING_CLIENT boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  stop_server "$PORT" "$PID"
done"""


def sbatch_text() -> str:
    return SBATCH.read_text(encoding="utf-8")


def _between(begin: str, end: str, text: str | None = None) -> str:
    """The block between two marker lines, markers excluded, verbatim."""
    text = sbatch_text() if text is None else text
    start = text.index("\n", text.index(begin)) + 1
    return text[start : text.index(end)]


def knob_block(text: str | None = None) -> str:
    return _between(KBEGIN, KEND, text)


def loop_block(text: str | None = None) -> str:
    return _between(LBEGIN, LEND, text)


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
STUBS = r"""
env () { printf '%s\n' "$@" > "$OUT/argv_${LABEL}.txt"; }
server_cmd () { echo python -m sglang.launch_server --port "$1"; }
wait_health () { [ "$LABEL" = "$FAIL_HEALTH" ] && return 1; return 0; }
stop_server () { echo "STOP port=$1 pid=$2" >> "$OUT/stops.txt"; }
kill () { return 0; }
timeout () { printf '%s\n' "$@" > "$OUT/client_${LABEL}.txt"; \
  [ "$LABEL" = "$FAIL_CLIENT" ] && return 1; return 0; }
"""


class _LoopRunner(unittest.TestCase):
    """Execute a boot-loop block with every external effect stubbed out."""

    BASEPORT = 31234

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_loop(self, block, order="L TD L TD", guard="", out=None,
                 fail_health="", fail_client=""):
        out = Path(out or tempfile.mkdtemp(dir=self.tmp))
        script = out.parent / f"loop_{out.name}.sh"
        script.write_text(
            "set -uo pipefail\n"
            f"OUT={shlex.quote(str(out))}\n"
            f"HERE={shlex.quote(str(SBATCH.parent))}\n"
            f"ORDER={shlex.quote(order)}\n"
            f"GUARD={shlex.quote(guard)}\n"
            f"FAIL_HEALTH={shlex.quote(fail_health)}\n"
            f"FAIL_CLIENT={shlex.quote(fail_client)}\n"
            f"BASEPORT={self.BASEPORT}\n"
            "LABEL=unset\nDSM=44\nSEED=1\nTFP=1\nMEMTEL=1\nCOMMIT=cafef00d\n"
            + STUBS + block + "\nwait\n",
            encoding="utf-8",
        )
        proc = subprocess.run(["/usr/bin/bash", str(script)],
                              capture_output=True, text=True, timeout=120)
        return proc, out

    # -- artifact readers ---------------------------------------------------- #
    @staticmethod
    def argv(out, label):
        path = Path(out) / f"argv_{label}.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else None

    @staticmethod
    def client(out, label):
        path = Path(out) / f"client_{label}.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else None

    @staticmethod
    def boots(out):
        path = Path(out) / "boots.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    @staticmethod
    def diag(out):
        path = Path(out) / "diag_boots.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None


# --------------------------------------------------------------------------- #
# 4. the default path did not move
# --------------------------------------------------------------------------- #
class DefaultPathByteIdentityTest(_LoopRunner):
    """★The whole safety argument: adding `DN` changed nothing for L / TD."""

    ORDERS = ("L TD L TD", "L TD", "TD L", "L TD TD L")

    def test_frozen_legacy_fragment_is_the_harness_908179_ran(self):
        """The fixture's provenance, so the identity test cannot be vacuous."""
        got = subprocess.run(
            ["git", "show", f"{LEGACY_COMMIT}:{SBATCH_REL}"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if got.returncode != 0:
            self.skipTest("git history for the 908179 commit is not available")
        self.assertEqual(hashlib.sha256(got.stdout).hexdigest(),
                         LEGACY_SBATCH_SHA256)
        self.assertIn(LEGACY_LOOP, got.stdout.decode("utf-8"),
                      "the frozen loop fixture is not verbatim from that blob")

    def test_default_path_is_byte_identical_to_908179(self):
        """Both versions run in turn at the SAME absolute $OUT, so this is a
        literal byte comparison of the produced argv -- no normalisation, no
        hand-written expectation that could quietly become the identity."""
        for order in self.ORDERS:
            with self.subTest(order=order):
                work = self.tmp / "job_same_path"
                old = self._snapshot(LEGACY_LOOP, order, work)
                new = self._snapshot(loop_block(), order, work)
                self.assertEqual(sorted(new), sorted(old))
                self.assertEqual(new, old)
                self.assertIn("boots.txt", new)
                self.assertNotIn("diag_boots.txt", new,
                                 "the default path must not create it")
                # the comparison must actually have seen the launch argv
                for label in new["boots.txt"].decode().split():
                    self.assertIn(f"argv_{label}.txt", new)
                    self.assertIn(f"client_{label}.txt", new)

    def _snapshot(self, block, order, work):
        """Run `block` at exactly `work` and return {filename: bytes}."""
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        proc, _ = self.run_loop(block, order=order, out=work)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return {p.name: p.read_bytes() for p in sorted(work.iterdir())
                if p.name.startswith(("argv_", "client_", "boots", "diag_boots"))}

    def test_only_stdout_difference_is_the_scored_flag_in_the_banner(self):
        """Honest accounting of what DID change on the default path: the job
        log's boot banner gained `scored=<0|1>`.  Nothing reads it."""
        old, _ = self.run_loop(LEGACY_LOOP)
        new, _ = self.run_loop(loop_block())
        differing = set(new.stdout.splitlines()) ^ set(old.stdout.splitlines())
        for line in differing:
            self.assertTrue(line.startswith("##########"), differing)
        self.assertTrue(differing, "expected the banner to differ")

    def test_no_guard_variable_appears_without_a_dn_boot(self):
        _, out = self.run_loop(loop_block(), order="L TD L TD", guard="none")
        for label in ("L1", "TD1", "L2", "TD2"):
            argv = self.argv(out, label)
            self.assertFalse(any(a.startswith("PDMUX_WORKER_GRAD_GUARD") for a in argv),
                             f"{label} was given a guard override: {argv}")


# --------------------------------------------------------------------------- #
# 1 + 2. the DN boot
# --------------------------------------------------------------------------- #
class DnBootTest(_LoopRunner):
    ORDER = "L TD L TD DN"

    def setUp(self):
        super().setUp()
        self.proc, self.out = self.run_loop(loop_block(), order=self.ORDER,
                                            guard="none")
        self.assertEqual(self.proc.returncode, 0,
                         self.proc.stdout + self.proc.stderr)

    def test_dn_label_never_reaches_bootstxt(self):
        """★INVARIANT 1.  A predicted OOM may not colour the gate label."""
        self.assertEqual(self.boots(self.out), "L1 TD1 L2 TD2 ")
        self.assertNotIn("DN", self.boots(self.out))

    def test_dn_label_goes_to_the_diagnostic_list(self):
        self.assertEqual(self.diag(self.out), "DN1 ")

    def test_the_scorer_cannot_reach_the_diagnostic_list_or_label(self):
        src = CHECKER.read_text(encoding="utf-8")
        self.assertNotIn("diag_boots", src)
        self.assertNotIn('"DN', src)
        # the label list comes from boots.txt alone (the no-glob half of this
        # argument is pinned by tests/test_r2_correctness_instrument.py)
        self.assertIn('open(os.path.join(out, "boots.txt"))', src)

    def test_dn_boot_ran_and_produced_its_own_artifacts(self):
        self.assertIsNotNone(self.argv(self.out, "DN1"))
        self.assertIn("DIAGNOSTIC boot, NOT SCORED", self.proc.stdout)
        self.assertIn("scored=0", self.proc.stdout)

    def test_dn_env_is_the_td_env_plus_exactly_the_guard(self):
        """★INVARIANT 2, stated as a set difference.

        Label-bearing values (telemetry path, run id, green path) and the port
        differ BY CONSTRUCTION -- a boot's label and its port are what make it a
        distinct boot -- so both are normalised away; whatever survives is the
        real treatment.
        """
        td = self._normalise(self.argv(self.out, "TD1"), "TD1", self.BASEPORT + 2)
        dn = self._normalise(self.argv(self.out, "DN1"), "DN1", self.BASEPORT + 5)
        self.assertEqual([a for a in dn if a not in td],
                         ["PDMUX_WORKER_GRAD_GUARD=none"])
        self.assertEqual([a for a in td if a not in dn], [])
        self.assertEqual(len(dn), len(td) + 1)

    def test_dn_carries_every_scored_instrument(self):
        argv = self.argv(self.out, "DN1")
        for expected in (
            "PDMUX_R2_POLICY=fixed",
            "PDMUX_R2_FIXED_DSM=44",
            "PDMUX_TRUE_DUAL_WORKER=1",
            "PDMUX_WORKER_GRAD_GUARD=none",
            f"PDMUX_TELEMETRY_PATH={self.out}/tel_DN1.jsonl",
            "PDMUX_RUN_ID=r2corr_local_DN1",
            "PDMUX_WORKLOAD_ID=r2corr_seed1",
            "PDMUX_TRACE_FORCE_PREFILL=1",
            "PDMUX_MEM_TELEMETRY=1",
            "PDMUX_GREEN_READOUT=1",
            f"PDMUX_GREEN_READOUT_PATH={self.out}/green_DN1.json",
            "PDMUX_ENGINE_COMMIT=cafef00d",
        ):
            self.assertIn(expected, argv)

    def test_dn_gets_the_same_client_call_as_a_scored_boot(self):
        dn = self.client(self.out, "DN1")
        td = self.client(self.out, "TD1")
        self.assertEqual(
            [a.replace("DN1", "<L>").replace(str(self.BASEPORT + 5), "<P>")
             for a in dn],
            [a.replace("TD1", "<L>").replace(str(self.BASEPORT + 2), "<P>")
             for a in td],
        )
        self.assertEqual(dn[0], "900")
        self.assertIn("--n-probes", dn)
        self.assertEqual(dn[dn.index("--n-probes") + 1], "8")

    def test_every_guard_value_is_passed_through_verbatim(self):
        for value in ("none", "no_grad", "inference_mode"):
            with self.subTest(guard=value):
                _, out = self.run_loop(loop_block(), order="TD DN", guard=value)
                self.assertIn(f"PDMUX_WORKER_GRAD_GUARD={value}",
                              self.argv(out, "DN1"))

    def _normalise(self, argv, label, port):
        return [a.replace(label, "<L>").replace(str(port), "<P>") for a in argv]


# --------------------------------------------------------------------------- #
# 5. a dead DN server does not take the job with it
# --------------------------------------------------------------------------- #
class DnFailureIsContainedTest(_LoopRunner):
    """The DN boot is PREDICTED to die.  The loop must not care."""

    def test_boot_failure_does_not_stop_the_remaining_boots(self):
        proc, out = self.run_loop(loop_block(), order="L DN TD",
                                  guard="none", fail_health="DN1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("BOOT_FAILED boot=DN1", proc.stdout)
        self.assertIsNotNone(self.argv(out, "TD1"),
                             "the boot after the dead DN boot did not run")
        self.assertEqual(self.boots(out), "L1 TD1 ")

    def test_client_failure_does_not_stop_the_remaining_boots(self):
        proc, out = self.run_loop(loop_block(), order="L DN TD",
                                  guard="none", fail_client="DN1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("CLIENT_FAILED boot=DN1", proc.stdout)
        self.assertIsNotNone(self.argv(out, "TD1"))

    def test_a_dead_dn_boot_is_still_torn_down(self):
        _, out = self.run_loop(loop_block(), order="DN", guard="none",
                               fail_health="DN1")
        self.assertIn(f"port={self.BASEPORT + 1}",
                      (Path(out) / "stops.txt").read_text(encoding="utf-8"))

    def test_the_verdict_step_is_outside_the_loop_and_unconditional(self):
        """`set -e` is absent (`set -uo pipefail` only), and the checker call
        sits after the loop with no guard, so a dead boot cannot skip it."""
        text = sbatch_text()
        self.assertIn("set -uo pipefail", text)
        self.assertNotIn("set -euo", text)
        loop_end = text.index(LEND)
        verdict = text.index('python3 "$HERE/r2_correctness_check.py" "$OUT"')
        self.assertLess(loop_end, verdict)
        line = next(l for l in text.splitlines()
                    if 'python3 "$HERE/r2_correctness_check.py" "$OUT"' in l)
        self.assertFalse(line.lstrip().startswith(("if", "&&", "[")), line)
        self.assertEqual(line, line.lstrip(), "the verdict call is not nested")


# --------------------------------------------------------------------------- #
# 3. the knob is fail-closed
# --------------------------------------------------------------------------- #
class GuardKnobTest(unittest.TestCase):
    ECHO = ('echo "GUARD=[$GUARD] HAS_DN=$HAS_DN GUARD_PROV=[$GUARD_PROV] '
            'GUARD_APPLIED=[$GUARD_APPLIED] n_arms=$n_arms"')

    def run_knob(self, order="L TD L TD", guard=None, block=None):
        script = (
            "set -uo pipefail\n"
            f"ORDER={shlex.quote(order)}\n"
            + (block if block is not None else knob_block())
            + "\n" + self.ECHO + "\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "R2C_GUARD"}
        if guard is not None:
            env["R2C_GUARD"] = guard
        return subprocess.run(["/usr/bin/bash", "-c", script],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_unset_is_fine_without_a_dn_boot(self):
        proc = self.run_knob(guard=None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GUARD=[] HAS_DN=0 GUARD_PROV=[<engine default>]",
                      proc.stdout)
        self.assertIn("GUARD_APPLIED=[no]", proc.stdout)

    def test_empty_string_is_fine_without_a_dn_boot(self):
        """`R2C_GUARD=` exported empty must behave like unset, not like a value
        (the engine would resolve an empty name to the DEFAULT guard)."""
        proc = self.run_knob(guard="")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GUARD_PROV=[<engine default>]", proc.stdout)

    def test_a_dn_boot_without_a_guard_is_refused(self):
        """★No silent default (gate #6 / #237)."""
        proc = self.run_knob(order="L TD L TD DN", guard=None)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("R2C_GUARD is unset", proc.stderr)

    def test_a_dn_boot_with_an_empty_guard_is_refused(self):
        proc = self.run_knob(order="DN", guard="")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("R2C_GUARD is unset", proc.stderr)

    def test_an_unlisted_value_is_refused_even_without_a_dn_boot(self):
        """A typo must never survive: `noen` would otherwise be echoed into
        provenance while the engine served the default."""
        for bad in ("noen", "None", "NONE", "inference-mode", "0", "off",
                    "none extra"):
            for order in ("L TD L TD", "L TD DN"):
                with self.subTest(value=bad, order=order):
                    proc = self.run_knob(order=order, guard=bad)
                    self.assertEqual(proc.returncode, 2,
                                     proc.stdout + proc.stderr)
                    self.assertIn("is not one of", proc.stderr)

    def test_each_listed_value_resolves_and_is_reported(self):
        for value in ("none", "no_grad", "inference_mode"):
            with self.subTest(guard=value):
                proc = self.run_knob(order="L TD DN", guard=value)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(f"GUARD=[{value}] HAS_DN=1 GUARD_PROV=[{value}]",
                              proc.stdout)
                self.assertIn("GUARD_APPLIED=[DN-boots-only]", proc.stdout)

    def test_a_guard_without_a_dn_boot_is_recorded_as_ignored(self):
        proc = self.run_knob(order="L TD L TD", guard="none")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GUARD_PROV=[<engine default>]", proc.stdout)
        self.assertIn("set but ORDER has no DN boot", proc.stdout)

    def test_dn_is_matched_as_a_whole_token(self):
        """`DNX` or `ADN` is not a DN boot; the loop would reject it as a bad
        arm, and requiring a guard for it would be a false alarm."""
        for order in ("L TDN", "L DNX", "L ADN"):
            with self.subTest(order=order):
                proc = self.run_knob(order=order, guard=None)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("HAS_DN=0", proc.stdout)

    def test_the_port_budget_is_fail_closed_at_six_boots(self):
        proc = self.run_knob(order="L TD L TD DN", guard="none")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("n_arms=5", proc.stdout)
        proc = self.run_knob(order="L TD L TD DN DN", guard="none")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("port budget is 5", proc.stderr)

    def test_the_knob_does_not_break_set_u(self):
        """`set -u` is the harness's own setting; an unbound expansion here
        would kill a job that is otherwise fine."""
        script = ("set -uo pipefail\nORDER='L TD L TD'\n" + knob_block()
                  + "\necho SURVIVED\n")
        env = {k: v for k, v in os.environ.items() if k != "R2C_GUARD"}
        proc = subprocess.run(["/usr/bin/bash", "-c", script],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("SURVIVED", proc.stdout)
        self.assertNotIn("unbound variable", proc.stderr)


# --------------------------------------------------------------------------- #
# 6. provenance and placement
# --------------------------------------------------------------------------- #
class ProvenanceAndPlacementTest(unittest.TestCase):
    def test_the_resolved_guard_is_written_to_provenance(self):
        """It is part of the scope tuple, so a log line is not enough."""
        text = sbatch_text()
        open_brace = text.index("COMMIT=$(git -C")
        close = text.index('} > "$OUT/provenance.txt"')
        block = text[open_brace:close]
        self.assertIn(
            'echo "mem_telemetry=$MEMTEL worker_grad_guard=$GUARD_PROV', block)
        self.assertIn(
            'echo "r2c_guard=${R2C_GUARD:-<unset>} guard_applied=$GUARD_APPLIED '
            'dn_boots=$HAS_DN order=[$ORDER]"', block)

    def test_the_knob_resolves_before_the_first_gpu_second_is_spent(self):
        text = sbatch_text()
        knob = text.index(KEND)
        self.assertLess(knob, text.index("sync_engine_tree.sh"))
        self.assertLess(knob, text.index("# --- BEGIN r2c warmup launch"))

    def test_the_scope_guard_is_still_required(self):
        """Spec item 7: the substrate check stays mandatory and unchanged."""
        text = sbatch_text()
        self.assertIn("for _v in R2C_EXPECT_MODEL R2C_EXPECT_BACKEND R2C_EXPECT_CTX; do",
                      text)
        self.assertIn("SCOPE_GUARD: refusing to boot (no GPU work performed)", text)
        self.assertLess(text.index("# --- END r2c scope guard ---"),
                        text.index(LBEGIN))

    def test_the_dn_arm_is_documented_in_the_usage_header(self):
        text = sbatch_text()
        head = text[: text.index("set -uo pipefail")]
        self.assertIn('R2C_ORDER="L TD L TD DN" R2C_GUARD=none sbatch', head)
        self.assertIn("diag_boots.txt", head)


# --------------------------------------------------------------------------- #
# ★ NEGATIVE CONTROLS (lesson #53)
# --------------------------------------------------------------------------- #
class MutantTest(_LoopRunner):
    """Each mutant must break a test above.  A gate that passes on the
    un-repaired form is not a gate."""

    # ★The anchor is the WHOLE assignment, not the substring
    # ` PDMUX_WORKER_GRAD_GUARD=$GUARD`.  The first version of this test used the
    # substring, which ALSO occurs in the block's own comment -- so when the real
    # export was deleted the mutant test still found its anchor (in the comment),
    # mutated nothing that mattered, and PASSED.  A mutation test whose anchor is
    # not load-bearing is an identity (lesson #9), so the count is asserted.
    GUARD_EXPORT_LINE = ('        TD_ENV="PDMUX_TRUE_DUAL_WORKER=1 '
                         'PDMUX_WORKER_GRAD_GUARD=$GUARD" ;;')
    GUARD_EXPORT_REMOVED = '        TD_ENV="PDMUX_TRUE_DUAL_WORKER=1" ;;'

    def test_mutant_a_without_the_guard_re_export(self):
        """★REQUIRED MUTANT (a): delete the guard from the DN branch.

        Then DnBootTest.test_dn_env_is_the_td_env_plus_exactly_the_guard and
        test_dn_carries_every_scored_instrument must fail -- the DN boot would be
        an ordinary TD boot on a different port, i.e. no control arm at all.
        """
        block = loop_block()
        self.assertEqual(block.count(self.GUARD_EXPORT_LINE), 1,
                         "the mutation anchor is not unique")
        mutant = block.replace(self.GUARD_EXPORT_LINE, self.GUARD_EXPORT_REMOVED)
        _, out = self.run_loop(mutant, order="L TD L TD DN", guard="none")
        argv = self.argv(out, "DN1")
        self.assertIsNotNone(argv)
        self.assertNotIn("PDMUX_WORKER_GRAD_GUARD=none", argv,
                         "the mutant still exported the guard: the positive "
                         "tests do not depend on that line")
        td = self._norm(self.argv(out, "TD1"), "TD1", self.BASEPORT + 2)
        dn = self._norm(argv, "DN1", self.BASEPORT + 5)
        self.assertEqual(dn, td,
                         "without the guard the DN boot is indistinguishable "
                         "from a TD boot, which is exactly the failure the "
                         "positive test must catch")

    def test_mutant_b_writing_dn_into_bootstxt(self):
        """★REQUIRED MUTANT (b): send the DN label to boots.txt.

        Then DnBootTest.test_dn_label_never_reaches_bootstxt must fail, and the
        predicted OOM would be scored as a TD-arm boot -- the `FAIL` by
        construction that audit sec 8.5 identified.
        """
        block = loop_block()
        old = '    echo -n "$LABEL " >> "$OUT/diag_boots.txt"'
        self.assertEqual(block.count(old), 1, "anchor is not unique")
        mutant = block.replace(old, '    echo -n "$LABEL " >> "$OUT/boots.txt"')
        _, out = self.run_loop(mutant, order="L TD L TD DN", guard="none")
        self.assertIn("DN1", self.boots(out),
                      "the mutant did not reach boots.txt: the invariant test "
                      "does not depend on the diag_boots.txt line")
        self.assertIsNone(self.diag(out))

    def test_mutant_marking_the_dn_branch_as_scored(self):
        """The same leak through the other door: SCORED=1 on the DN arm."""
        block = loop_block()
        old = '    DN) n_dn=$((n_dn + 1)); LABEL="DN$n_dn"; SCORED=0'
        self.assertEqual(block.count(old), 1, "anchor is not unique")
        mutant = block.replace(old, old[:-1] + "1")
        _, out = self.run_loop(mutant, order="L TD L TD DN", guard="none")
        self.assertIn("DN1", self.boots(out))

    def test_mutant_dropping_the_fail_closed_exit(self):
        """Without `exit 2` an unlisted R2C_GUARD would sail through."""
        block = knob_block()
        old = ('  *) echo "ERROR: R2C_GUARD=\'$GUARD\' is not one of '
               'inference_mode|no_grad|none" >&2\n     exit 2 ;;')
        self.assertEqual(block.count(old), 1, "anchor is not unique")
        mutant = block.replace(old, "  *) : ;;")
        script = ("set -uo pipefail\nORDER='L TD DN'\n" + mutant
                  + "\necho \"GUARD_PROV=[$GUARD_PROV]\"\n")
        env = dict(os.environ, R2C_GUARD="noen")
        proc = subprocess.run(["/usr/bin/bash", "-c", script],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(proc.returncode, 0,
                         "the mutant still refused: the enumeration test does "
                         "not depend on that exit")
        self.assertIn("GUARD_PROV=[noen]", proc.stdout)

    def test_mutant_dropping_the_point_of_use_guard_check(self):
        """The loop's own `[ -n "${GUARD:-}" ]` is the second line of defence:
        with it gone, a DN boot with an empty guard launches and silently gets
        the engine DEFAULT."""
        block = loop_block()
        old = ('        [ -n "${GUARD:-}" ] || { echo "ERROR: DN boot with no '
               'R2C_GUARD" >&2; exit 2; }\n')
        self.assertEqual(block.count(old), 1, "anchor is not unique")
        mutant = block.replace(old, "")
        proc, out = self.run_loop(mutant, order="DN", guard="")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("PDMUX_WORKER_GRAD_GUARD=", self.argv(out, "DN1"))

    def test_the_point_of_use_guard_check_fires(self):
        """Positive side of the mutant above."""
        proc, out = self.run_loop(loop_block(), order="DN", guard="")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("DN boot with no R2C_GUARD", proc.stderr)
        self.assertIsNone(self.argv(out, "DN1"))

    def _norm(self, argv, label, port):
        return [a.replace(label, "<L>").replace(str(port), "<P>") for a in argv]


if __name__ == "__main__":
    sys.path.insert(0, str(TRACK_ROOT / "benchmarks"))
    unittest.main()
