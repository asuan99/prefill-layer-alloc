#!/usr/bin/env python3
"""Mutation harness for the stage-0 decision path (rev3).

De-confound lesson 53: *a check that verifies a repair must FAIL on a version
that reverts the repair.*  History of this file:

  rev1 audit  ran 11 mutations against rev1's selftest; 7 ESCAPED.
  rev2        blocked all 17 it registered -- but ★mutated `lambda0_label.py`
              ONLY.  The rev2 audit then wrote six mutations of its own and
              **five escaped**, including X3 on the primary estimator itself
              (the realized denominator N-1 -> N).  A harness whose reach stops
              at one file certifies one file (gate #181).
  rev3        mutates the whole decision path -- `lambda0_label.py`,
              `lambda0_analyze.py`, `lambda0_plan.py` -- and the selftest now
              contains an END-TO-END leg that feeds the analyzer's real output
              dict into `rule()`, which is what makes analyzer mutations visible.

Every row must read FAILS.  The unmutated CONTROL must read PASSES -- without it
the harness could be reporting "FAILS" because it cannot run the file at all
(the escape route that produced the first draft of the rev2 harness, where the
patched copy in a temp dir silently skipped the archived-artifact comparison).

usage: python3 lambda0_mutation_check.py        # exit 0 iff 0 escapes
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ARCHIVE = HERE.parent.parent / "longctx_conflict" / "probes" / "c_905835"
# ★rev4/E3: the DECISION PATH, not a subset.  rev3 stopped at three files and
# the rev3 audit's Y5 -- `lambda0_cells.py` certifying every rung -- replicated
# rev3's own kill cause into shape A while passing 36/36 untouched.
MODULES = ("lambda0_label.py", "lambda0_analyze.py", "lambda0_plan.py",
           "lambda0_cells.py", "lambda0_lambda_inf.py",
           "lambda0_reachability.py")

# name -> (module, old, new)
MUTATIONS = {
    # ---- rev2's registered 17, kept ------------------------------------
    "M1  max(saturated) -> min           ": ("lambda0_label.py",
        "lam = max(saturated) if", "lam = min(saturated) if"),
    "M2  drop tie/monotone guard         ": ("lambda0_label.py",
        "and hi[0] > lo[0]\n            for i, lo in enumerate(ladder)",
        "\n            for i, lo in enumerate(ladder)"),
    "M3  ACH_HI 0.95 -> 1.20             ": ("lambda0_label.py",
        "ACH_HI = 0.95", "ACH_HI = 1.20"),
    "M4  ACH_LO 0.90 -> 0.70             ": ("lambda0_label.py",
        "ACH_LO = 0.90", "ACH_LO = 0.70"),
    "M5  ACH_LO 0.90 -> 0.80             ": ("lambda0_label.py",
        "ACH_LO = 0.90", "ACH_LO = 0.80"),
    "M6  ACH_HI 0.95 -> 1.05             ": ("lambda0_label.py",
        "ACH_HI = 0.95", "ACH_HI = 1.05"),
    "M7  ACH_HI boundary >= -> >         ": ("lambda0_label.py",
        "lo[1] >= ACH_HI and", "lo[1] > ACH_HI and"),
    "M8  ACH_LO boundary <= -> <         ": ("lambda0_label.py",
        "hi[1] <= ACH_LO and hi[0]", "hi[1] < ACH_LO and hi[0]"),
    "M9  saturation sign flip            ": ("lambda0_label.py",
        "if r <= ACH_LO]", "if r >= ACH_LO]"),
    "M10 lambda* = max over ALL cells    ": ("lambda0_label.py",
        "saturated = [a for _, r, a, _, _ in ladder if r <= ACH_LO]",
        "saturated = [a for _, r, a, _, _ in ladder]"),
    "M11 sort by ratio, not offered rate ": ("lambda0_label.py",
        "for c in seq), key=lambda t: t[0])\n        # ★R1",
        "for c in seq), key=lambda t: t[1])\n        # ★R1"),
    "M12 use the NOMINAL denominator     ": ("lambda0_label.py",
        'RATIO_KEY = "achieved_over_realized"',
        'RATIO_KEY = "achieved_over_offered"'),
    "M13 revert D4: glob the directory   ": ("lambda0_label.py",
        "    for name in expect:\n        rec = read(name)",
        "    for name in [p.name[5:-5] for p in sorted(d.glob('cell_*.json'))]:"
        "\n        rec = read(name)"),
    "M14 seed-repeat tol 0.05 -> 0.50    ": ("lambda0_label.py",
        "SEED_REPEAT_TOL = 0.05", "SEED_REPEAT_TOL = 0.50"),
    "M15 lambda* without a bracket       ": ("lambda0_label.py",
        "lam = max(saturated) if (bracketed and saturated) else None",
        "lam = max(saturated) if saturated else None"),
    "M16 guards off by default           ": ("lambda0_label.py",
        "def rule(cells, repeats=None, enforce_guards=True):",
        "def rule(cells, repeats=None, enforce_guards=False):"),
    "M17 ebar guard defaults to pass     ": ("lambda0_label.py",
        'if not c.get("ebar_guard_ok", False):',
        'if not c.get("ebar_guard_ok", True):'),
    # ---- the rev2 auditor's independent six (X3-X5 escaped rev2) --------
    "X1  high side ACH_LO -> ACH_HI      ": ("lambda0_label.py",
        "hi[1] <= ACH_LO and hi[0]", "hi[1] <= ACH_HI and hi[0]"),
    "X2  saturated set ACH_LO -> ACH_HI  ": ("lambda0_label.py",
        "if r <= ACH_LO]", "if r <= ACH_HI]"),
    "X3  realized denominator N-1 -> N   ": ("lambda0_analyze.py",
        "off_real = (n_sent - 1) / span", "off_real = n_sent / span"),
    "X4  EBAR_GUARD band -> (0.5, 1.5)   ": ("lambda0_analyze.py",
        "EBAR_GUARD = (0.95, 1.05)", "EBAR_GUARD = (0.5, 1.5)"),
    "X5  analyzer drops the shape field  ": ("lambda0_analyze.py",
        '"shape": shape, "label": label,', '"shape": "?", "label": label,'),
    "X6  replay draws n instead of n-1   ": ("lambda0_analyze.py",
        "size=num_prompts - 1", "size=num_prompts"),
    # ---- rev3's own new surfaces ---------------------------------------
    "R3a low-side restriction removed    ": ("lambda0_label.py",
        "lo[3] and lo[1] >= ACH_HI", "lo[1] >= ACH_HI"),
    "R3b low-side flag defaults to True  ": ("lambda0_label.py",
        'certified = c.get("low_side_candidate", False)',
        'certified = c.get("low_side_candidate", True)'),
    "R3c range-ratio guard removed       ": ("lambda0_label.py",
        'if not c.get("range_ratio_ok", False):',
        'if False:'),
    "R3d drain guard defaults to pass    ": ("lambda0_label.py",
        'ceiling_intact = c.get("drain_model_ok", False)',
        'ceiling_intact = c.get("drain_model_ok", True)'),
    "R3e unevaluated repeat -> REFUTED   ": ("lambda0_label.py",
        '        elif lam is None:\n            # ★D21',
        '        elif False:\n            # ★D21'),
    "R3f drain sign flipped              ": ("lambda0_analyze.py",
        'drain = dur - span if span == span else float("nan")',
        'drain = span - dur if span == span else float("nan")'),
    "R3g drain model tolerance -> 1e9    ": ("lambda0_analyze.py",
        "DRAIN_MODEL_TOL = 2.0", "DRAIN_MODEL_TOL = 1e9"),
    "R3h range ratio requirement dropped ": ("lambda0_analyze.py",
        "REQUIRED_RANGE_RATIO = 1.0", "REQUIRED_RANGE_RATIO = 0.9"),
    "R3i KAPPA_MIN -> ACH_HI - 0.10      ": ("lambda0_plan.py",
        "KAPPA_MIN = ACH_HI + 0.02", "KAPPA_MIN = ACH_HI - 0.10"),
    "R3j window extension disabled       ": ("lambda0_plan.py",
        "T_LOW_GRID = tuple(range(600, 1801, 100))", "T_LOW_GRID = ()"),
    "R3k no rung is extendable           ": ("lambda0_plan.py",
        "N_EXTENDABLE = 2", "N_EXTENDABLE = 0"),
    "R3l drain model ignores the batch   ": ("lambda0_plan.py",
        "    if batch > MAX_RUNNING:\n        return None, None",
        "    if False:\n        return None, None"),
    "R3m step model intercept -> solo ITL": ("lambda0_plan.py",
        "STEP_A_S = 19.82e-3", "STEP_A_S = 13.06e-3"),
    # ---- the rev3 auditor's independent eight (Y1,Y2,Y4,Y5,Y6 escaped rev3) --
    "Y1  achieved from another field    ": ("lambda0_analyze.py",
        'achieved = d["request_throughput"]', 'achieved = d["output_throughput"]'),
    "Y2  TTFT_LOAD_FACTOR 3.2 -> 1.0    ": ("lambda0_plan.py",
        "TTFT_LOAD_FACTOR = 3.2", "TTFT_LOAD_FACTOR = 1.0"),
    "Y3  kappa margin +0.02 -> +0.001   ": ("lambda0_plan.py",
        "KAPPA_MIN = ACH_HI + 0.02", "KAPPA_MIN = ACH_HI + 0.001"),
    "Y4  load() ignores the record shape": ("lambda0_label.py",
        'shape = rec["shape"] if rec else name.split("_")[0].upper()\n'
        '        cells.setdefault(shape, []).append(rec)',
        'shape = name.split("_")[0].upper()\n'
        '        cells.setdefault(shape, []).append(rec)'),
    "Y5  renderer certifies every rung  ": ("lambda0_cells.py",
        'int(bool(c["low_side_candidate"])), c["T_measure_s"]))',
        '1, c["T_measure_s"]))'),
    "Y6  renderer swaps kappa/low-side  ": ("lambda0_cells.py",
        '        "-" if c["kappa_pred"] is None else c["kappa_pred"],\n'
        '        "-" if c["drain_pred_s"] is None else c["drain_pred_s"],\n'
        '        int(bool(c["low_side_candidate"])), c["T_measure_s"]))',
        '        int(bool(c["low_side_candidate"])),\n'
        '        "-" if c["drain_pred_s"] is None else c["drain_pred_s"],\n'
        '        "-" if c["kappa_pred"] is None else c["kappa_pred"], c["T_measure_s"]))'),
    # ---- rev4's own new surfaces -------------------------------------------
    "R4a E1 reverted: drain ignored     ": ("lambda0_label.py",
        "    return bool(certified and ceiling_intact)",
        "    return bool(certified)"),
    "R4b LADDER_TOO_LOW -> top rung >=HI": ("lambda0_label.py",
        "        elif not saturated:", "        elif ladder[-1][1] >= ACH_HI:"),
    "R4c predicate: F5 threshold 48 -> 1": ("lambda0_lambda_inf.py",
        "MIN_RUNNING_REQ = 48", "MIN_RUNNING_REQ = 1"),
    "R4d predicate ignores a missing file": ("lambda0_lambda_inf.py",
        'reasons.append(f"missing {fn}")', "pass"),
    "R4e reachability grid collapsed    ": ("lambda0_reachability.py",
        "LAMBDA_RATIO_GRID = (0.10, 0.20, 0.25, 0.35, 0.50, 0.70, 0.85,\n"
        "                     1.00, 1.20, 1.45, 1.80, 2.50)",
        "LAMBDA_RATIO_GRID = (1.00,)"),
}

# Which selftest is expected to notice a mutation of which module.
SELFTEST_OF = {"lambda0_label.py": "lambda0_label.py",
               "lambda0_analyze.py": "lambda0_label.py",   # via the e2e leg
               "lambda0_plan.py": "lambda0_plan.py",
               "lambda0_cells.py": "lambda0_cells.py",
               "lambda0_lambda_inf.py": "lambda0_lambda_inf.py",
               "lambda0_reachability.py": "lambda0_reachability.py"}
CONTROL_ENTRIES = ("lambda0_label.py", "lambda0_plan.py", "lambda0_cells.py",
                   "lambda0_lambda_inf.py", "lambda0_reachability.py")


def _run(patched_name, patched_text, env, entry=None):
    """Copy the whole decision path to a temp dir, optionally patch ONE file,
    and run the selftest that is supposed to notice.  Everything -- including
    the CONTROL -- runs through here, so a broken copy cannot masquerade as 'the
    mutation was blocked'."""
    td = tempfile.mkdtemp()
    for f in MODULES:
        shutil.copy(HERE / f, td)
    if patched_name:
        pathlib.Path(td, patched_name).write_text(patched_text)
    entry = entry or SELFTEST_OF.get(patched_name, "lambda0_label.py")
    return subprocess.run([sys.executable, os.path.join(td, entry), "--selftest"],
                          capture_output=True, text=True, env=env)


def main() -> int:
    src = {f: (HERE / f).read_text() for f in MODULES}
    # The patched copy runs from a temp dir, so hand it the archive explicitly;
    # the shipped file resolves it relative to itself and HARD FAILS if absent.
    env = dict(os.environ, LAMBDA0_PROBE_C_DIR=str(ARCHIVE))

    # ★rev4/E3 + gate G-lam3-5: the CONTROL must be scored ON THE PATH THE
    # MUTANTS RUN (the temp-dir copy).  rev3 called `_run(None, None, env)` and
    # then THREW THE RESULT AWAY, re-running in place instead -- so if the
    # temp-dir copy were broken every mutation would print "FAILS (good)" and
    # 36/36 would pass vacuously.  Injection-checked below by the caller.
    bad_control = False
    for entry in CONTROL_ENTRIES:
        r = _run(None, None, env, entry=entry)
        ok = r.returncode == 0
        print("CONTROL unmutated %-24s(temp-dir copy) %s" % (
            entry, "PASSES (good)" if ok else
            "FAILS  <<< harness broken: " +
            (r.stderr.strip().splitlines() or ["?"])[-1]))
        bad_control |= not ok
    if bad_control:
        return 2

    escapes = []
    for name, (mod, a, b) in MUTATIONS.items():
        if a not in src[mod]:
            print("%s  SKIPPED <<< mutation target text not found in %s" % (name, mod))
            escapes.append(name.strip())
            continue
        r = _run(mod, src[mod].replace(a, b, 1), env)
        blocked = r.returncode != 0
        print("%s [%-17s] %s" % (name, mod[8:-3], "FAILS (good)" if blocked
                                 else "PASSES  <<< ESCAPE"))
        if not blocked:
            escapes.append(name.strip())
    print("\nescapes: %s" % (escapes if escapes else
                             "none (%d/%d blocked)" % (len(MUTATIONS), len(MUTATIONS))))
    return 1 if escapes else 0


if __name__ == "__main__":
    sys.exit(main())
