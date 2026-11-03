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
    ("M18 engine pairs need no definition smoke", "sb_analyze.py", '        if smoke.get("status") != "CLIENT_DEF_OK":', "        if False:"),
    ("M19 trace-hash identity not checked", "sb_analyze.py", 'if ra[s]["trace_hash"] != rb[s]["trace_hash"]:', "if False:"),
    ("M20 request failures ignored", "sb_analyze.py", 'if any(q.get("error") for r in arms[x].values() for s in r["segments"] for q in s["requests"]):', "if False:"),
    ("M21 CELL_OUT_OF_BAND 'all > 0.9' rule removed", "sb_analyze.py", "oob = all(x > SP.OUT_OF_BAND_ALL_ABOVE for x in att.values()) or", "oob = False or"),
    ("M22 substrate shift threshold 10 %p -> 50 %p", "sb_spec.py", "SUBSTRATE_SHIFT_PP = 0.10", "SUBSTRATE_SHIFT_PP = 0.50"),
    ("M24 lambda*_ref = max over arms", "sb_analyze.py", 'add["lambda_star_ref"] = min(vals.values())', 'add["lambda_star_ref"] = max(vals.values())'),
    ("M25 ABBA round 2 not reversed", "sb_spec.py", "(2, list(reversed(order)))", "(2, list(order))"),
    ("M26 sustained 0.5 point never runs", "sb_runner.py", "                    if not any(e[\"pass\"] for e in evs):\n                        pending.append",
     "                    if False:\n                        pending.append"),
    ("M27 sustained completion bar 0.95 -> 0.5", "sb_spec.py", "SUSTAIN_COMPLETION_MIN, SUSTAIN_TTFT_GROWTH_MAX = 0.95, 2.0", "SUSTAIN_COMPLETION_MIN, SUSTAIN_TTFT_GROWTH_MAX = 0.5, 2.0"),
    ("M28 pristine boot keeps PDMUX_* knobs", "sb_runner.py", 'if not k.startswith("PDMUX_") or k in keep}', "if True}"),
    ("M29 SG-stock planned on the patched tree", "sb_spec.py", 'PRISTINE_ARMS = ("SG-stock", "SG-rep", "SG-mix")', 'PRISTINE_ARMS = ("SG-rep", "SG-mix")'),
    ("M30 instrumentation env allowed in a measurement boot", "sb_spec.py", 'if b["role"] == "meas" and a["env"].get("PDMUX_DECODE_PROBE_PATH"):', "if False:"),
    ("M31 placebo arm may be absent", "sb_spec.py", 'if c != "R3" and "SG-rep" not in by:', "if False:"),
    ("M32 job target accepts any engine digest", "sb_job.sh", '[[ "$(tr -d \' \\n\' < "${edigest}")" != "${expected_digest}" ]]; then', "false; then"),
    ("M33 dead boot exits 0", "sb_runner.py", 'return 17 if any(b["status"] != "done" for b in res) else 0', "return 0"),
    ("M34 SG-off argv differs from SG-stock (mixed chunk on)", "sb_spec.py", '(["--enable-mixed-chunk"] if name == "SG-mix" else [])', '(["--enable-mixed-chunk"] if name in ("SG-mix", "SG-off") else [])'),
    ("M35 R2 = 0.5 lambda*_ref", "sb_spec.py", 'STEADY = {"R1": 0.5, "R2": 0.8}', 'STEADY = {"R1": 0.5, "R2": 0.5}'),
    ("M36 R3 cell = 0.5 lambda*_W3", "sb_spec.py", "W3_LADDER_N, W3_R3_FACTOR, W3_R3_N = P.CAP_CELL_REQUESTS[\"W3\"], 0.8, 96",
     "W3_LADDER_N, W3_R3_FACTOR, W3_R3_N = P.CAP_CELL_REQUESTS[\"W3\"], 0.5, 96"),
    ("M37 smoke tolerance 5 ms -> 50 ms", "sb_spec.py", "CLIENT_DEF_TOL_MS = 5.0", "CLIENT_DEF_TOL_MS = 50.0"),
    ("M38 smoke server TTFT without count-delta check", "sb_loadgen.py", "and (c1 - c0) == 1 and", "and"),
    ("M39 openai client counts every chunk as a token", "sb_loadgen.py",
     'ct = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else self.last_completion_tokens + 1',
     "ct = self.last_completion_tokens + 1"),
    ("M40 resume guard removed", "sb_runner.py", "and not resume:", "and False:"),
    ("M41 R3 rate filled from an unbracketed W3 ladder", "sb_runner.py", 'if lam is not None and st == "BRACKETED" else None', "if lam is not None else None"),
    ("M42 probe window not mapped through the boot clock", "sb_analyze.py", 'return (b["t_host"] + (t0_epoch - b["t_realtime"]), b["t_host"] + (t1_epoch - b["t_realtime"]))',
     "return (t0_epoch, t1_epoch)"),
    # --- rev1 (user decision 2026-11-02): no change trace, vLLM venv in the one image, two-step TTFT time base, sustain denominator ---
    ("M43 W1 runs re-enter J3 (trace/W1 cells back in the W3 Job)", "sb_spec.py",
     'return [cap_run(f"{job}_{name}_R3_s{s}", "W3", None,', 'return [cap_run(f"{job}_{name}_R3_s{s}", "W1", None,'),
    ("M44 plan check: J3 W3-only rule removed", "sb_spec.py", 'if plan["kind"] == "W3" and any(r.get("workload") != "W3"', 'if False and any(r.get("workload") != "W3"'),
    ("M45 analyzer accepts a rev0 TRACE_W3 J3 tree", "sb_analyze.py", '"J1b": "SUSTAIN", "J2": "STEADY", "J3": "W3"}',
     '"J1b": "SUSTAIN", "J2": "STEADY", "J3": "TRACE_W3"}'),
    ("M46 CELL_OUT_OF_BAND not judged on R2", "sb_analyze.py", 'OOB_CELLS = ("R2", "R3")', 'OOB_CELLS = ("R3",)'),
    ("M47 runner: vLLM venv freeze check removed", "sb_runner.py", 'if engine == "vllm" and not ic.get("vllm_ok"):', "if False:"),
    ("M48 runner: SGLang freeze check removed", "sb_runner.py", '    if not ic.get("sglang_ok"):\n', "    if False:\n"),
    ("M49 runner: recorded venv sha not compared", "sb_runner.py", 'vt == SP.VLLM_VENV_FREEZE_SHA256 and vf == SP.VLLM_VENV_FREEZE_SHA256', "vt == SP.VLLM_VENV_FREEZE_SHA256"),
    ("M50 analyzer: VL venv freeze not checked", "sb_analyze.py", 'if st.get("vllm_venv_freeze_sha256") != SP.VLLM_VENV_FREEZE_SHA256:', "if False:"),
    ("M51 analyzer: VL boots of a Job may differ", "sb_analyze.py", 'return [] if len(keys) <= 1 else', "return [] if True else"),
    ("M52 vLLM env keeps PDMUX_* plumbing", "sb_runner.py", 'keep = INFRA_PASS if a["engine"] == "sglang" else ()', "keep = INFRA_PASS"),
    ("M53 vLLM env keeps PYTHONPATH", "sb_runner.py", '        env.pop("PYTHONPATH", None)\n', "        pass\n"),
    ("M54 TTFT offset tag removed (engine pair stays generic INVALID)", "sb_analyze.py", 'elif engine_pair and "TTFT_OFFSET_SHIFT" in why:', "elif False:"),
    ("M55 TTFT offset not part of engine-pair validity", "sb_analyze.py", 'if ost != "TTFT_OFFSET_OK":', "if False:"),
    ("M56 TTFT offset 20 ms -> 200 ms", "sb_spec.py", "TTFT_OFFSET_SHIFT_MS = 20.0 ", "TTFT_OFFSET_SHIFT_MS = 200.0 "),
    ("M58 TTFT offset without the count-delta check", "sb_analyze.py", "if not tt or dc != len(tt):", "if not tt or dc <= 0:"),
    ("M59 no /metrics scrape around vLLM runs", "sb_runner.py", 'return a["engine"] == "vllm" or b["role"] == "instr_metrics"', 'return b["role"] == "instr_metrics"'),
    ("M60 sustain completion ratio over the NOMINAL arrivals", "sb_runner.py", '"completion_ratio": comp[w] / arr if arr else 0.0',
     '"completion_ratio": comp[w] / (rate * window_s)'),
    ("M61 Job preflight accepts any image", "sb_spec.py", 'return isinstance(image_ref, str) and image_ref.endswith("@" + IMAGE_SBASE_DIGEST)', "return True"),
    # --- rev2 (harness audit r1, 2026-11-02: blocking fixes 1-5); M23 (proxy >= 50 firing bar) retired with the proxy rule ---
    ("M62 sustain numerator back to rev1 'completed inside the window'", "sb_runner.py",
     "comp = [sum(1 for r in by_arr.get(w, []) if sustain_done(r)) for w in range(nwin)]",
     'comp = [sum(1 for r in reqs if not r.get("error") and r.get("completion_s") is not None and int(r["completion_s"] // window_s) == w) '
     "for w in range(nwin)]"),
    ("M63 sustain E2E <= 60 s bound dropped", "sb_runner.py", "return float(r[\"completion_s\"]) - float(t0) <= e2e_max_s", "return True"),
    ("M64 sustain arrivals >= 600 s clamped into the last window (rev1)", "sb_runner.py",
     '        if float(r["scheduled_arrival_s"]) >= SP.SUSTAIN_S:\n            n_late += 1\n            continue\n'
     '        by_arr.setdefault(int(float(r["scheduled_arrival_s"]) // window_s), []).append(r)',
     '        by_arr.setdefault(min(nwin - 1, int(float(r["scheduled_arrival_s"]) // window_s)), []).append(r)'),
    ("M65 sustain TTFT 2x growth rule dropped", "sb_runner.py", " and growth <= SP.SUSTAIN_TTFT_GROWTH_MAX", ""),
    ("M66 BOTH_SATURATED rule removed", "sb_analyze.py", '        elif sat:\n            final = "BOTH_SATURATED"', '        elif False:\n            final = "BOTH_SATURATED"'),
    ("M67 saturation bar 0.98 -> 0.999", "sb_spec.py", "SATURATED_ATTAIN = 0.98 ", "SATURATED_ATTAIN = 0.999 "),
    ("M68 latency co-metrics omitted", "sb_spec.py", 'LATENCY_PAIRS = ("Q3", "Q5", "Q8")', "LATENCY_PAIRS = ()"),
    ("M69 latency labels ignore the validity chain", "sb_analyze.py", 'lfin = "BOOT_NOISE" if pbad else "INVALID" if why else', 'lfin = "BOOT_NOISE" if pbad else'),
    ("M70 SG-mix firing ignores server_args", "sb_analyze.py", "fired = [RU.mix_fired_from_server_args(run_server_args(r, job)) for r in arms[a].values()]",
     "fired = [True for r in arms[a].values()]"),
    ("M71 SG-mix firing without the chunked-prefill condition", "sb_runner.py",
     'return f.get("enable_mixed_chunk") == "True" and cps is not None and cps > 0', 'return f.get("enable_mixed_chunk") == "True"'),
    ("M72 mixing opportunities not computed for SG-stock (negative control)", "sb_spec.py", 'MIX_OPPORTUNITY_ARMS = ("SG-mix", "SG-stock")', 'MIX_OPPORTUNITY_ARMS = ("SG-mix",)'),
    ("M73 Q6 grid keeps 8k", "sb_spec.py", "(512, 2048), 0.25", "(512, 2048, 8192), 0.25"),
    ("M74 P-4 scored at n < 10", "sb_analyze.py", 'elif not b2 or b2["n"] < SP.Q6_MIN_PER_L:', "elif not b2:"),
    ("M75 other-stream extend does not invalidate Q6", "sb_analyze.py", '    if n_other:\n        q6 = "Q6_INVALID', '    if False:\n        q6 = "Q6_INVALID'),
    # --- synthetic single-request prefill window (user decision 2026-11-02, PREREG §5-1 / §6) ---
    ("M82 synthetic window pooled with the W1 window", "sb_analyze.py",
     'probe_window(records, run["t_load_start_epoch"], run["t_load_end_epoch"])',
     'probe_window(records, run["t_load_start_epoch"], run.get("t_synth_end_epoch") or run["t_load_end_epoch"])'),
    ("M83 multi-request prefill counted as single-request", "sb_analyze.py",
     'return g.get("r_ser") is not None and g.get("extend_n_reqs") == 1', 'return g.get("r_ser") is not None and g.get("extend_n_reqs") >= 1'),
    ("M84 runner W1 window runs over the synthetic segment", "sb_runner.py",
     'load = [x for x in rec["segments"] if x["phase"] != SP.SYNTH_PHASE]', 'load = list(rec["segments"])'),
    ("M85 P-4 scored from the W1 window", "sb_analyze.py",
     'syn = (sgi.get("windows") or {}).get("SYNTH") or {}', 'syn = (sgi.get("windows") or {}).get("W1") or {}'),
    ("M76 smoke per-engine dispersion check removed", "sb_analyze.py",
     'st = "CLIENT_DEF_OK" if all(med[e]["sd_ms"] <= SP.CLIENT_DEF_TOL_MS for e in ("sglang", "vllm")) else', 'st = "CLIENT_DEF_OK" if True else'),
    ("M77 smoke cross-engine |SG - VL| <= 5 ms requirement re-added", "sb_analyze.py",
     'st = "CLIENT_DEF_OK" if all(med[e]["sd_ms"] <= SP.CLIENT_DEF_TOL_MS for e in ("sglang", "vllm")) else',
     'st = "CLIENT_DEF_OK" if all(med[e]["sd_ms"] <= SP.CLIENT_DEF_TOL_MS for e in ("sglang", "vllm")) and '
     'abs(med["sglang"]["median_ms"] - med["vllm"]["median_ms"]) <= SP.CLIENT_DEF_TOL_MS else'),
    ("M78 Q1-Q8 rows not forced", "sb_analyze.py", "return {q: rows[q] for q in SP.QUESTIONS}",
     'return {q: r for q, r in rows.items() if r["status"] != "NOT_MEASURED"}'),
    ("M79 DONE.json status not reported", "sb_analyze.py", "        out += boot_status_rows(name, j)", "        pass"),
    ("M80 reference-capacity None reason not reported", "sb_analyze.py", "    out += ref_capacity_rows(add)", "    pass"),
    ("M81 J3 R3 budget guard removed", "sb_runner.py", 'ctx["r3_budget_skip"] = ctx["r3_budget"]["skip"]', 'ctx["r3_budget_skip"] = False'),
    # --- rev2.1 (re-audit r2 (a), user decision 2026-11-02) ---
    ("M86 P-3 sign-scored in a BOTH_SATURATED 80 % cell", "sb_analyze.py",
     'sat3 = bool(r2) and (r2.get("both_saturated") is True or r2.get("label") == "BOTH_SATURATED")', "sat3 = False"),
    # --- rev3 (user decision 2026-11-03, (C) = (A)) ---
    ("M87 reference capacity includes the SGLang-default arms", "sb_spec.py", 'REF_ARMS = ("VL-stock", "PM-ref")', "REF_ARMS = LADDER_ARMS"),
    ("M88 J1-down rungs shifted (k = 0..3: the failed 5.516 rung re-run, 3.154 dropped)", "sb_spec.py",
     "LADDER_DOWN_K = (1, 2, 3, 4) ", "LADDER_DOWN_K = (0, 1, 2, 3) "),
    ("M89 refcell rate = the largest burst lambda* (VL-stock) instead of the PM-ref rung", "sb_analyze.py",
     'add["refcell_rate"] = add["lambda_star_burst"].get(SP.REFCELL_ARM)',
     'add["refcell_rate"] = max(add["lambda_star_burst"].values(), default=None)'),
    ("M90 refcell arm back to SG-stock (rev2.1)", "sb_spec.py", 'REFCELL_ARM, REFCELL_N, REFCELL_SEED = "PM-ref", 120, 201',
     'REFCELL_ARM, REFCELL_N, REFCELL_SEED = "SG-stock", 120, 201'),
    ("M91 sign-only rule removed (Q3/Q5/Q8 magnitudes labelled)", "sb_spec.py", 'SIGN_ONLY_QUESTIONS = ("Q3", "Q5", "Q8")', "SIGN_ONLY_QUESTIONS = ()"),
    ("M92 Q3 latency co-metrics dropped", "sb_spec.py", 'LATENCY_PAIRS = ("Q3", "Q5", "Q8")', 'LATENCY_PAIRS = ("Q5", "Q8")'),
    ("M93 second bottom failure keeps the plain LADDER_BOTTOM_FAIL label", "sb_analyze.py",
     '    if L["status"] == "LADDER_BOTTOM_FAIL":\n        L = dict(L, status=SP.LADDER_DOWN_FAIL', '    if False:\n        L = dict(L, status=SP.LADDER_DOWN_FAIL'),
    ("M94 J1-down points pooled into the J1b input (lambda_star_burst)", "sb_analyze.py",
     'L = ladder_lambda(runs, a)\n', 'L = ladder_lambda(runs + (list(j1down["runs"]) if j1down else []), a, cells=("LADDER", "LADDER_DOWN"))\n'),
    ("M95 sb_job.sh refuses J1-down", "sb_job.sh", "  J1|J1-ext|J1-down|J1b|J2|J3) ;;", "  J1|J1-ext|J1b|J2|J3) ;;"),
    ("M96 J3 floor measured on the refcell arm (PM-ref) instead of SG-stock", "sb_spec.py",
     'boots = [refcell_boot(job, "start", ref_rate), boot(f"{job}_w3lad_SG-stock", arm("SG-stock"), w3l, role="w3ladder", floor=True)]',
     'boots = [refcell_boot(job, "start", ref_rate, floor=True), boot(f"{job}_w3lad_SG-stock", arm("SG-stock"), w3l, role="w3ladder")]'),
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
