#!/usr/bin/env python3
"""Mutation check of the stock-baseline harness (lesson 53: a rule is guarded only if reverting it makes a test FAIL).
Each mutant copies results/stock_baseline to a temp mirror of engine-port/, reverts ONE registered rule by an exact source substitution
(the pattern must occur exactly once, else the mutant is BROKEN = a survivor), and runs tests/test_stock_baseline_harness.py with SB_DIR
pointing at the copy.  Killed iff that run fails on a test (not on an import error).  `run_all()` -> survivors ([] = all killed)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
TESTS = ENG / "tests" / "test_stock_baseline_harness.py"

MUTANTS = [
    # --- task list (all six required) ---
    ("M01 goodput duration max() instead of sum (gate 7)", "sb_analyze.py", 'dur += float(s["duration_s"])', 'dur = max(dur, float(s["duration_s"]))'),
    ("M02 r_ser: records outside the load window mixed in", "sb_analyze.py", "t_window=win)", "t_window=None)"),
    ("M03 r_span: chunk window filter dropped", "sb_analyze.py",
     'if x.get("event") != "prefill_chunk" or not inw(x.get("t_issue_start", x.get("timestamp_monotonic_s"))):',
     'if x.get("event") != "prefill_chunk":'),
    ("M04 r_span: decode window filter dropped", "sb_analyze.py", "            and inw(x.get(\"t_host_start\")):", "            and True:"),
    ("M05 pair across Jobs/GPUs allowed", "sb_analyze.py", "    if len(st) > 1:\n        why.append(\"CROSS_JOB", "    if False:\n        why.append(\"CROSS_JOB"),
    ("M06 engine_tree mismatch allowed", "sb_analyze.py", 'if st.get("engine_tree") != want or run.get("engine_tree") != want:', "if False:"),
    ("M07 pristine manifest not compared with the v0.5.10 reference", "sb_analyze.py",
     'if st.get("engine_tree_manifest_sha256") != SP.PRISTINE_REF_SHA256 or st.get("engine_tree_ref_sha256") != SP.PRISTINE_REF_SHA256:', "if False:"),
    ("M08 pristine marker check removed", "sb_analyze.py", 'if st.get("engine_tree_markers_found") != []:', "if False:"),
    ("M09 patched manifests may differ within a Job", "sb_analyze.py", 'return [] if len(shas) <= 1 else', "return [] if True else"),
    ("M10 lambda* tie -> higher rate", "sb_runner.py", "pts = sorted(points, key=lambda x: (x[0], x[1]))", "pts = sorted(points, key=lambda x: (x[0], -x[1]))"),
    ("M11 placebo |mean| threshold loosened to 2x", "sb_analyze.py", 'return abs(pl["mean"]) >= MARGIN or', 'return abs(pl["mean"]) >= 2 * MARGIN or'),
    ("M12 placebo CI-excludes-0 rule removed", "sb_analyze.py", ' or pl["lo"] > 0 or pl["hi"] < 0', ""),
    ("M13 placebo n<4 rule removed", "sb_analyze.py", 'if pl is None or pl["n"] < 4 or', "if pl is None or"),
    ("M14 placebo failure does not force BOOT_NOISE", "sb_analyze.py", "        if pbad:\n            final = \"BOOT_NOISE\"", "        if False:\n            final = \"BOOT_NOISE\""),
    # --- other registered rules ---
    ("M15 ABOVE without the 3 % margin", "sb_analyze.py", "if lo > 0 and m >= MARGIN:", "if lo > 0:"),
    ("M16 label allowed at n < 4", "sb_analyze.py", "if n < 4 or any(isinstance(x, float) and math.isnan(x) for x in (m, lo, hi)):",
     "if any(isinstance(x, float) and math.isnan(x) for x in (m, lo, hi)):"),
    ("M17 cliff check removed", "sb_analyze.py", "if reqs and V.is_cliff(reqs, ttft_slo):", "if False:"),
    ("M18 engine pairs need no definition smoke", "sb_analyze.py", 'and smoke.get("status") != "CLIENT_DEF_OK":', "and False:"),
    ("M19 trace-hash identity not checked", "sb_analyze.py", 'if ra[s]["trace_hash"] != rb[s]["trace_hash"]:', "if False:"),
    ("M20 request failures ignored", "sb_analyze.py", 'if any(q.get("error") for r in arms[x].values() for s in r["segments"] for q in s["requests"]):', "if False:"),
    ("M21 CELL_OUT_OF_BAND 'all > 0.9' rule removed", "sb_analyze.py", "oob = all(x > SP.OUT_OF_BAND_ALL_ABOVE for x in att.values()) or", "oob = False or"),
    ("M22 substrate shift threshold 10 %p -> 50 %p", "sb_spec.py", "SUBSTRATE_SHIFT_PP = 0.10", "SUBSTRATE_SHIFT_PP = 0.50"),
    ("M23 MIX firing threshold 50 -> 0", "sb_spec.py", "MIX_MIN_FORWARDS = 50", "MIX_MIN_FORWARDS = 0"),
    ("M24 lambda*_ref = max over arms", "sb_analyze.py", 'add["lambda_star_ref"] = min(vals.values())', 'add["lambda_star_ref"] = max(vals.values())'),
    ("M25 ABBA round 2 not reversed", "sb_spec.py", "(2, list(reversed(order)))", "(2, list(order))"),
    ("M26 sustained 0.5 point never runs", "sb_runner.py", "                    if not any(e[\"pass\"] for e in evs):\n                        pending.append",
     "                    if False:\n                        pending.append"),
    ("M27 sustained completion bar 0.95 -> 0.5", "sb_spec.py", "SUSTAIN_COMPLETION_MIN, SUSTAIN_TTFT_GROWTH_MAX = 0.95, 2.0", "SUSTAIN_COMPLETION_MIN, SUSTAIN_TTFT_GROWTH_MAX = 0.5, 2.0"),
    ("M28 pristine boot keeps PDMUX_* knobs", "sb_runner.py", 'if not k.startswith("PDMUX_") or k in INFRA_PASS}', "if True}"),
    ("M29 SG-stock planned on the patched tree", "sb_spec.py", 'PRISTINE_ARMS = ("SG-stock", "SG-rep", "SG-mix")', 'PRISTINE_ARMS = ("SG-rep", "SG-mix")'),
    ("M30 instrumentation env allowed in a measurement boot", "sb_spec.py", 'if b["role"] == "meas" and a["env"].get("PDMUX_DECODE_PROBE_PATH"):', "if False:"),
    ("M31 placebo arm may be absent", "sb_spec.py", 'if c != "R3" and "SG-rep" not in by:', "if False:"),
    ("M32 job target accepts any engine digest", "sb_job.sh", '[[ "$(tr -d \' \\n\' < "${edigest}")" != "${expected_digest}" ]]; then', "false; then"),
    ("M33 dead boot exits 0", "sb_runner.py", 'return 17 if any(b["status"] != "done" for b in res) else 0', "return 0"),
    ("M34 SG-off argv differs from SG-stock (mixed chunk on)", "sb_spec.py", '(["--enable-mixed-chunk"] if name == "SG-mix" else [])', '(["--enable-mixed-chunk"] if name in ("SG-mix", "SG-off") else [])'),
    ("M35 R2 = 0.5 lambda*_ref", "sb_spec.py", 'STEADY = {"R1": 0.5, "R2": 0.8}', 'STEADY = {"R1": 0.5, "R2": 0.5}'),
    ("M36 trace HI/LO = 0.8/0.4", "sb_spec.py", "TRACE_LO, TRACE_HI, TRACE_NP, TRACE_ROUNDS = 0.2, 0.8, 200, 3", "TRACE_LO, TRACE_HI, TRACE_NP, TRACE_ROUNDS = 0.4, 0.8, 200, 3"),
    ("M37 smoke tolerance 5 ms -> 50 ms", "sb_spec.py", "CLIENT_DEF_TOL_MS = 5.0", "CLIENT_DEF_TOL_MS = 50.0"),
    ("M38 smoke server TTFT without count-delta check", "sb_loadgen.py", "and (c1 - c0) == 1 and", "and"),
    ("M39 openai client counts every chunk as a token", "sb_loadgen.py",
     'ct = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else self.last_completion_tokens + 1',
     "ct = self.last_completion_tokens + 1"),
    ("M40 resume guard removed", "sb_runner.py", "and not resume:", "and False:"),
    ("M41 R3 rate filled from an unbracketed W3 ladder", "sb_runner.py", 'if lam is not None and st == "BRACKETED" else None', "if lam is not None else None"),
    ("M42 probe window not mapped through the boot clock", "sb_analyze.py", 'return (b["t_host"] + (t0_epoch - b["t_realtime"]), b["t_host"] + (t1_epoch - b["t_realtime"]))',
     "return (t0_epoch, t1_epoch)"),
]


def _unittest(d, td):
    env = {**os.environ, "SB_DIR": str(d), "SB_IN_MUTANT": "1"}
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(TESTS.parent), "-p", TESTS.name],
                          env=env, capture_output=True, text=True, timeout=600, cwd=td)


def _copy(td):
    """Mirror engine-port/: results/stock_baseline is a real copy (the mutated thing); everything it reads is a symlink."""
    eng = Path(td) / "engine-port"
    (eng / "results").mkdir(parents=True)
    d = eng / "results" / "stock_baseline"
    shutil.copytree(HERE, d, ignore=shutil.ignore_patterns("__pycache__", "HARNESS_DIGEST.txt", "runs"))
    for rel in ("results/cap_char", "scripts", "benchmarks", "src", "env"):
        (eng / rel).symlink_to(ENG / rel)
    return d


def control():
    with tempfile.TemporaryDirectory() as td:
        r = _unittest(_copy(td), td)
        if r.returncode != 0:
            raise RuntimeError("control run failed:\n" + r.stderr[-3000:])


def run_one(name, fname, old, new):
    with tempfile.TemporaryDirectory() as td:
        d = _copy(td)
        src = (d / fname).read_text()
        if src.count(old) != 1:
            return f"{name}: BROKEN MUTANT (pattern occurs {src.count(old)}x)"
        (d / fname).write_text(src.replace(old, new))
        r = _unittest(d, td)
        if r.returncode != 0 and "FAIL:" not in r.stderr and "_FailedTest" in r.stderr:
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
