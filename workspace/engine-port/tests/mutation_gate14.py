#!/usr/bin/env python3
"""Mutation harness for the gate-#14 repair of ``pdmux_eval/analyze.py``.

Methodology gate #53: *a check that verifies a repair must FAIL on a mutant that
reverts that repair*, otherwise the check is a positive control for nothing.
This applies three reverts to a COPY of ``analyze.py`` and runs
``tests/test_analyze_gate14.py`` against each through ``PDMUX_ANALYZE_PATH``:

  M1  delete the t-CI functions            (the repair's substance)
  M2  disable the ``decision_eligible`` guard (always eligible)
  M3  let the CLI decide on the BOOTSTRAP interval again (the pre-repair verdict)

A mutant that survives is reported as a hole in the test module.  Nothing here
touches the working tree: mutants live in a temp directory.

Usage:  python workspace/engine-port/tests/mutation_gate14.py
Exit 0 iff every mutant is killed.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZE = HERE.parents[0] / "benchmarks" / "pdmux_eval" / "analyze.py"
TEST_MODULE = "test_analyze_gate14.py"


def delete_def(source, name):
    start = source.index("\ndef %s(" % name)
    end = source.index("\ndef ", start + 1)
    return source[:start] + source[end:]


def m1_delete_t_ci(source):
    """Revert the addition itself: no t interval in the module."""
    return delete_def(delete_def(source, "paired_t_ci"), "unpaired_t_ci")


def m2_disable_guard(source):
    """Guard computes but never refuses."""
    old = "    eligible = n >= GATE14_DECISION_MIN_N\n"
    assert source.count(old) == 1, "M2 anchor drifted"
    return source.replace(old, "    eligible = True\n")


def m3_cli_decides_on_bootstrap(source):
    """The pre-repair CLI: headline read off the percentile bootstrap."""
    old = ('                primary["effect_percent"] >= 3.0 '
           'and primary["ci95_low"] > 0.0')
    assert source.count(old) == 1, "M3 anchor drifted"
    return source.replace(
        old,
        '                companion["effect_percent"] >= 3.0 '
        'and companion["ci95_low"] > 0.0',
    )


MUTANTS = (
    ("M1 delete the t-CI functions", m1_delete_t_ci),
    ("M2 disable the decision_eligible guard", m2_disable_guard),
    ("M3 CLI decides on the bootstrap interval again", m3_cli_decides_on_bootstrap),
)

FAIL_RE = re.compile(r"^(?:FAIL|ERROR): (\S+)", re.M)


def run_tests(analyze_path):
    env = dict(os.environ, PDMUX_ANALYZE_PATH=str(analyze_path))
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
         "-p", TEST_MODULE, "-v"],
        capture_output=True, text=True, env=env,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode, sorted(set(FAIL_RE.findall(output))), output


def main():
    source = ANALYZE.read_text(encoding="utf-8")
    code, failures, _ = run_tests(ANALYZE)
    print("BASELINE (unmutated): rc=%d failures=%s" % (code, failures or "none"))
    if code != 0:
        print("ABORT: the unmutated tree already fails; fix that first.")
        return 1
    survivors = []
    with tempfile.TemporaryDirectory() as directory:
        for label, mutate in MUTANTS:
            path = Path(directory) / (label.split()[0] + "_analyze.py")
            path.write_text(mutate(source), encoding="utf-8")
            code, failures, output = run_tests(path)
            killed = code != 0
            print("\n%s -> %s" % (label, "KILLED" if killed else "SURVIVED"))
            for name in failures:
                print("    fails: %s" % name)
            if not killed:
                survivors.append(label)
                print(output[-2000:])
    print("\n%d/%d mutants killed" % (len(MUTANTS) - len(survivors), len(MUTANTS)))
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
