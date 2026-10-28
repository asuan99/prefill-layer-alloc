#!/usr/bin/env python3
"""Mutation check of the policy harness (lesson 53: a repair/guard is verified only if the reverted rule makes a test FAIL).
Each mutant copies the harness to a temp dir, reverts ONE registered rule by an exact source substitution (the substituted
text must occur exactly once, otherwise the mutant itself is broken and counts as a survivor), and runs the unit tests with
PH_DIR pointing at the copy.  A mutant is killed iff that run fails.  `run_all()` returns the survivors ([] = all killed)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
TESTS = HERE.parents[2] / "tests" / "test_policy_harness.py"

MUTANTS = [
    ("M01 goodput ITL leg = mean, not request p95", "ph_verdict.py", "percentile(itl, 0.95) <= itl_slo_ms", "(sum(itl) / len(itl)) <= itl_slo_ms"),
    ("M02 duration max() instead of sum (gate 7)", "ph_verdict.py", "dur += float(s[\"duration_s\"])", "dur = max(dur, float(s[\"duration_s\"]))"),
    ("M03 CI uses z=1.96 instead of t(n-1)", "ph_verdict.py", "h = T975[n - 1] * sd / math.sqrt(n)", "h = 1.96 * sd / math.sqrt(n)"),
    ("M04 t table value df=3 wrong", "ph_verdict.py", "3: 3.182", "3: 2.353"),
    ("M05 cliff band 5% -> 50%", "ph_verdict.py", "CLIFF_BAND, CLIFF_FRAC = 0.05, 0.10", "CLIFF_BAND, CLIFF_FRAC = 0.50, 0.10"),
    ("M06 cliff fraction threshold 10% -> 90%", "ph_verdict.py", "CLIFF_BAND, CLIFF_FRAC = 0.05, 0.10", "CLIFF_BAND, CLIFF_FRAC = 0.05, 0.90"),
    ("M07 HE0 cliff check removed", "ph_verdict.py", "elif any(cliff[f\"{a}:{ph}\"] for a in involved for ph in (\"LO\", \"HI\")):", "elif False:"),
    ("M08 ABOVE needs only the a-priori D44", "ph_verdict.py", "elif res[\"a\"][\"lo\"] > MARGIN and res[\"b\"][\"lo\"] > MARGIN:", "elif res[\"a\"][\"lo\"] > MARGIN:"),
    ("M09 n>=4 floor removed", "ph_verdict.py", "if len(seeds) < min_n:", "if False:"),
    ("M10 ABSENT without all registered q", "ph_verdict.py", "if set(qs_required) <= set(qs_run):", "if True:"),
    ("M11 G_best chosen on selection seeds", "ph_verdict.py", "statistics.fmean(tab[D][s][\"total\"] for s in ev))", "statistics.fmean(tab[D][s][\"total\"] for s in sel))"),
    ("M12 phase optimum chosen on evaluation seeds (no cross-fit)", "ph_verdict.py", "statistics.fmean(tab[D][s][w] for s in sel)", "statistics.fmean(tab[D][s][w] for s in ev)"),
    ("M13 narrowest instead of widest role CI", "ph_verdict.py", "\"lo\": min(r[\"lo\"] for r in roles)", "\"lo\": max(r[\"lo\"] for r in roles)"),
    ("M14 H-D cliff rule ignored", "ph_verdict.py", "if any(cliff_by_len.get(p) for p in lens):", "if False:"),
    ("M15 capacity: 90% rule loosened", "ph_verdict.py", "CAP_ATTAIN = 0.90", "CAP_ATTAIN = 0.40"),
    ("M16 tune tie-break by switches removed", "ph_verdict.py", "return min(near, key=lambda c: (sw[c], c))", "return max(near, key=lambda c: mean[c])"),
    ("M17 LO/HI ratio 4x -> 2x", "ph_spec.py", "HE0_RATIO = 4.0", "HE0_RATIO = 2.0"),
    ("M18 tuning seeds overlap evaluation seeds", "ph_spec.py", "TUNE_SEEDS = (101, 102)", "TUNE_SEEDS = (1, 2)"),
    ("M19 dynamic arm gets sticky partition", "ph_spec.py", "return {\"PDMUX_ENGINE_MODE\": \"pdmux\", \"PDMUX_SLO_SCHED\": \"1\"", "return {\"PDMUX_STICKY_PARTITION\": \"1\", \"PDMUX_ENGINE_MODE\": \"pdmux\", \"PDMUX_SLO_SCHED\": \"1\""),
    ("M20 odd/even start phase swapped", "ph_spec.py", "start_first = \"W1\" if k % 2 == 1 else \"W3\"", "start_first = \"W1\" if k % 2 == 0 else \"W3\""),
    ("M21 H-D rate q=1.0 -> 0.8", "ph_spec.py", "HD_Q = 1.0", "HD_Q = 0.8"),
    ("M22 W3 forced output (ignore_eos True)", "ph_trace.py", "\"ignore_eos\": False, \"doc_id\": did", "\"ignore_eos\": True, \"doc_id\": did"),
    ("M23 continuous variant: no phase offset", "ph_trace.py", "t_offset=j * phase_len_s", "t_offset=0"),
    ("M24 phase requests depend on the arm/tag", "ph_trace.py", "sk = (\"hd\", seed, w, phase_len_s)", "sk = (\"hd\", seed, w, phase_len_s, tag)"),
    ("M25 boot env keeps PDMUX_* (instrumentation leak)", "ph_runner.py", "not (k.startswith(\"PDMUX_\") and k not in ALLOW_PDMUX)", "True"),
    ("M26 dwell bookkeeping drops the last interval", "ph_runner.py", "dwell[cur] = dwell.get(cur, 0.0) + (t_last - t_prev)", "pass"),
    ("M27 resume re-runs finished runs", "ph_runner.py", "            if f.exists():\n                continue\n", ""),
    ("M29 natural-output wrapper does not flip ignore_eos", "ph_loadgen.py", 'payload["sampling_params"]["ignore_eos"] = False', "pass"),
    ("M30 abs_mono flag ignored by the wrapper", "ph_loadgen.py", "inner = AM.replay_abs_mono if abs_mono else TL.replay", "inner = TL.replay"),
    ("M31 natural set = every request", "ph_loadgen.py", 'if r.get("ignore_eos") is False}', 'if True}'),
    ("M28 non-empty raw dir allowed without --resume", "ph_runner.py", "and not resume:", "and False:"),
]


def _unittest(d, td):
    env = {**os.environ, "PH_DIR": str(d), "PH_IN_MUTANT": "1"}
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(TESTS.parent), "-p", TESTS.name],
                          env=env, capture_output=True, text=True, timeout=300, cwd=td)


def _copy(td):
    """Mirror engine-port/ so relative lookups (ct_spec -> scripts/costmodel, benchmarks/) keep working; only the policy
    harness is a real copy (the thing that gets mutated)."""
    eng = Path(td) / "engine-port"
    cc = eng / "results" / "cap_char"
    cc.mkdir(parents=True)
    d = cc / "policy_harness"
    shutil.copytree(HERE, d, ignore=shutil.ignore_patterns("__pycache__", "HARNESS_DIGEST.txt"))
    for rel in ("results/cap_char/costterms_harness", "scripts", "benchmarks"):
        (eng / rel).symlink_to(HERE.parents[2] / rel)
    return d


def control():
    """The UNMUTATED copy must pass; otherwise 'killed' would only mean 'the harness could not even load'."""
    with tempfile.TemporaryDirectory() as td:
        r = _unittest(_copy(td), td)
        if r.returncode != 0:
            raise RuntimeError("control run failed:\n" + r.stderr[-2000:])


def run_one(name, fname, old, new):
    with tempfile.TemporaryDirectory() as td:
        d = _copy(td)
        src = (d / fname).read_text()
        if src.count(old) != 1:
            return f"{name}: BROKEN MUTANT (pattern occurs {src.count(old)}x)"
        (d / fname).write_text(src.replace(old, new))
        r = _unittest(d, td)
        # killed for the RIGHT reason: a failing TEST, not an import/syntax error of the mutant
        if r.returncode != 0 and ("FAIL:" not in r.stderr and "ERROR:" in r.stderr and "_FailedTest" in r.stderr):
            return f"{name}: BROKEN MUTANT (module did not load)"
        return None if r.returncode != 0 else f"{name}: SURVIVED"


def run_all():
    control()
    return [x for x in (run_one(*m) for m in MUTANTS) if x]


if __name__ == "__main__":
    s = run_all()
    print(f"{len(MUTANTS) - len(s)}/{len(MUTANTS)} mutants killed")
    for x in s:
        print("  ", x)
    sys.exit(1 if s else 0)
