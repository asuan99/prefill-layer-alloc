"""N2(3-a) Job-PROFILE HARNESS mutants (CPU only) -- lesson 53 for the rev17 Job harness (n2_plan, n2_job,
n2_capacity.sh, n2_window, n2_analyze, n2_realized_mix, Job extensions of n2_marks / n2_loadgen). Same format as
n2_harness_mutants.HARNESS_MUTANTS: (id, description, expect, [(file, old, new), ...]); expect "kill" / "equivalent".
HARNESS-SEAL file. The smoke-harness ledger H01-H32 stays in n2_harness_mutants.py / `n2_mutants.py --set harness`
(rule-sealed runner, unchanged); THIS ledger is the Job harness's own (JH01…).

Run (sequential, alone):  PYTHONDONTWRITEBYTECODE=1 python3 n2_job_mutants.py [IDs...]
Each mutant: a fresh temp tree <tmp>/workspace/engine-port/results/n2_static_grid/ holding copies of every file of
this directory (rule-sealed files copied unmodified), with results/n3_capacity, results/r2_eval, benchmarks, src,
scripts symlinked read-only to the real tree; the mutation is applied to the copy; anchors must match exactly once
(else BROKEN-MUTANT); bytecode disabled (.pyc checked); a no-mutation control (all suites) runs first and a
CONTROL-FAILED row aborts the run (lesson 254); 900 s per mutant (TIMEOUT row, never a kill).
Suites: mutants of n2_marks.py / n2_loadgen.py run test_n2_harness + test_n2_job; all others run test_n2_job +
n2_reach_tests.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", ".."))
PL, JB, SH, WN, AN, MX, MK, LG = ("n2_plan.py", "n2_job.py", "n2_capacity.sh", "n2_window.py", "n2_analyze.py",
                                  "n2_realized_mix.py", "n2_marks.py", "n2_loadgen.py")
SUITES_ALL = ("test_n2_harness", "test_n2_job", "n2_reach_tests")

JOB_MUTANTS = [
    # ---- n2_plan: ABORT_REG_SOURCE (H14-2 / H15-5) and single source
    ("JH01", "ABORT_REG_SOURCE: file-byte sha256 check removed", "kill",
     [(PL, "    if got != expected_sha256:\n        raise RegSourceError", "    if False:\n        raise RegSourceError")]),
    ("JH02", "aux: K_matches_derivation not checked", "kill",
     [(PL, '    if reg.get("K_matches_derivation") is not True:', '    if False:')]),
    ("JH03", "aux: calculator_sha256 not checked", "kill",
     [(PL, '    if reg.get("calculator_sha256") != calculator_sha256:', '    if False:')]),
    ("JH04", "aux: limits.ok not checked", "kill",
     [(PL, '    if (reg.get("limits") or {}).get("ok") is not True:', '    if False:')]),
    ("JH05", "aux: K != derive_fixed_k not checked", "kill",
     [(PL, '    if reg.get("K") != reg.get("derive_fixed_k"):', '    if False:')]),
    ("JH06", "cell constants read from the sealed file, not the loaded registration (hard-wired source)", "kill",
     [(PL, '                    p = der["profile"][f"{alone_d},{b},{c}"]',
       '                    p = json.load(open(REGISTERED_JSON))["derived"]["profile"][f"{alone_d},{b},{c}"]')]),
    ("JH07", "cell-section cap hard-coded", "kill",
     [(PL, 'return dict(column=col, cap_s=float(b["cap"]),', 'return dict(column=col, cap_s=36700.0,')]),
    ("JH08", "admission uses the NOMINAL per-boot cost", "kill",
     [(PL, '    return float(budget(job, reg)["per_boot_worst"][tag])', '    return float(budget(job, reg)["per_boot_nom"][tag])')]),
    ("JH09", "U-1 column (a) instead of (b)", "kill", [(PL, 'U1 = "b" ', 'U1 = "a" ')]),
    ("JH10", "cell order: c outer, b inner", "kill",
     [(PL, "    return [(b, c) for b in bs for c in cs]", "    return [(b, c) for c in cs for b in bs]")]),
    ("JH11", "cell order: c not alternated in r3", "kill",
     [(PL, "    cs = list(C_GRID) if r in (1, 3) else list(reversed(C_GRID))",
       "    cs = list(C_GRID) if r in (1, 2, 3) else list(reversed(C_GRID))")]),
    ("JH12", "OBS cells use the 108 c_in", "kill",
     [(PL, '                alone_d = 108 if kind == "OFF" else d', '                alone_d = 108 if kind in ("OFF", "OBS") else d')]),
    ("JH13", "greedy probe not on OFF16 r1", "kill",
     [(PL, '"r1_ON44", "r1_OFF16"}', '"r1_ON44"}')]),
    ("JH14", "sticky set on every boot (OFF included)", "kill",
     [(PL, '    if unit["sticky"]:\n        env["PDMUX_STICKY_PARTITION"] = "1"', '    if True:\n        env["PDMUX_STICKY_PARTITION"] = "1"')]),
    ("JH15", "PHASE_EVENTS on for OBS", "kill",
     [(PL, '    if unit["phase_events"]:\n        env["PDMUX_PHASE_EVENTS"] = "1"', '    if True:\n        env["PDMUX_PHASE_EVENTS"] = "1"')]),
    ("JH16", "inherited PDMUX_* knobs reach the server", "kill",
     [(PL, '    env = {k: v for k, v in base_env.items() if not (k.startswith("PDMUX_") and k not in keep)}',
       '    env = dict(base_env)')]),
    ("JH17", "green: sticky_target_index not checked", "kill",
     [(PL, '    if green.get("sticky_target_index") != want_idx:', '    if False:')]),
    ("JH18", "green: plain-group green context not checked", "kill",
     [(PL, '            if not division and not rec.get("green_ctx_is_null", True):', '            if False:')]),
    ("JH19", "wiring: ENABLED line 'at least once' instead of exactly once", "kill",
     [(PL, '        if n_any != 1 or hits != [str(STICKY_IDX[d])]:', '        if str(STICKY_IDX[d]) not in hits:')]),
    ("JH20", "wiring: ENABLED line on an OFF boot ignored", "kill",
     [(PL, '    return [f"sticky ENABLED line present on a sticky-OFF boot ({n_any})"] if n_any else []', '    return []')]),
    ("JH21", "P3 tolerance 1e-2", "kill", [(PL, "P3_REL_TOL = 48, 48 * 16384, 1e-4", "P3_REL_TOL = 48, 48 * 16384, 1e-2")]),
    ("JH22", "capture-bs check drops b = 48", "kill",
     [(PL, "CAPTURE_BS_REQUIRED = (1, 8, 32, 48)", "CAPTURE_BS_REQUIRED = (1, 8, 32)")]),
    ("JH23", "config: duplicate-D detection removed", "kill",
     [(PL, "        if hits != [STICKY_IDX[d]]:", "        if STICKY_IDX[d] not in hits:")]),
    ("JH24", "rule seal: per-file sha loop removed", "kill",
     [(PL, '        if got != want:\n            bad.append(f"{fn}: {got} != sealed {want}")\n    if not bad and',
       '        pass\n    if not bad and')]),
    ("JH25", "harness seal: rule digest of the section not compared", "kill",
     [(PL, "    if rd != RULE_SEAL_DIGEST:\n", "    if False:\n")]),
    # ---- n2_job: boot protocol / aborts / STOP rule
    ("JH26", "warm-up not wrapped in warmup_start/_end (scope times kept, marks not written)", "kill",
     [(JB, "                with mw.warmup(b) as ws:                                   # G-3 (a), outside every cell",
       '                with n2_marks._WarmupScope(type("NoMarks", (), dict(now=staticmethod(time.monotonic), '
       'warmup_start=lambda *a: None, warmup_end=lambda *a: None))(), b) as ws:')]),
    ("JH27", "boot_plan records the alone D (108 for OFF)", "kill",
     [(JB, '    scope = mw.boot({unit["D"]: len(unit["cells"])}, classify)', '    scope = mw.boot({unit["alone_D"]: len(unit["cells"])}, classify)')]),
    ("JH28", "F-6 trailing request removed", "kill",
     [(JB, '                await lg.simple_request(session, base, lg.seed_tokens(lg.seed_of(f"n2|F6|{b}"), F6_INPUT),',
       '                0 and await lg.simple_request(session, base, lg.seed_tokens(lg.seed_of(f"n2|F6|{b}"), F6_INPUT),')]),
    ("JH29", "§11-1 wiring failure not enforced", "kill",
     [(JB, "                if wbad or gbad:\n", "                if False:\n")]),
    ("JH30", "§11-1 green failure ignored (log only)", "kill",
     [(JB, "                gbad = n2_plan.check_green(green, unit[\"sticky\"], unit[\"D\"])", "                gbad = []")]),
    ("JH31", "P3 not enforced", "kill", [(JB, "                if not ok:\n                    job_abort = JobAbort(8,", "                if False:\n                    job_abort = JobAbort(8,")]),
    ("JH32", "CG_BS not enforced", "kill", [(JB, "                if cgb:\n", "                if False:\n")]),
    ("JH33", "empty greedy output not enforced", "kill",
     [(JB, '                    if any(not (o.get("text") or o.get("output_ids")) for o in outs):', '                    if False:')]),
    ("JH34", "G-10 Job (§18-2-d #7): a VALUE_DEPENDENT abort stops the Job (pre-rev17 literal reading)", "kill",
     [(JB, "            # G-10 (§18-2-d #7, user decision 2026-10-11): any other boot abort (VI or VD) -> the next boot runs",
       '            elif res["aborted"] is not None and res["abort_class"] != RED.VALUE_INDEPENDENT:\n'
       '                stopped = "vd_abort"')]),
    ("JH35", "G-10: a VALUE_INDEPENDENT boot death stops the Job", "kill",
     [(JB, "            # G-10 (§18-2-d #7, user decision 2026-10-11): any other boot abort (VI or VD) -> the next boot runs",
       '            elif res["aborted"] is not None:\n                stopped = "boot_abort"')]),
    ("JH36", "§14-3: a refused boot is skipped (later cheaper boots admitted)", "kill",
     [(JB, '            if not n2_plan.admission(remaining, need):\n                stopped = "budget"',
       '            if not n2_plan.admission(remaining, need):\n                stopped = None')]),
    ("JH37", "SIGTERM does not cancel the running cell", "kill",
     [(JB, "        if cur is not None and not cur.done():\n            cur.cancel()", "        pass")]),
    ("JH38", "runner_stop not treated as the end of the cell section", "kill",
     [(JB, '            if res["aborted"] == "runner_stop":\n', '            if False:\n')]),
    ("JH39", "H-5 flush wait removed", "kill",
     [(JB, '                    await asyncio.sleep(2 * opts.get("flush_interval_s", FLUSH_INTERVAL_S))   # H-5',
       '                    await asyncio.sleep(0)')]),
    # ---- n2_loadgen (Job PROFILE cell) / n2_marks (warm-up)
    ("JH40", "PROFILE cancel at the lower median >= c (not c + 71)", "kill",
     [(LG, "spec.c_in + lower_median([q.n for q in reqs]) >= spec.c + TRIGGER_EXTRA:",
       "spec.c_in + lower_median([q.n for q in reqs]) >= spec.c:")]),
    ("JH41", "PROFILE cancel never fires (streams run to O)", "kill",
     [(LG, '            st["t_cancel"] = clock()\n            st["stop"].set()', '            st["t_cancel"] = clock()')]),
    ("JH42", "warmup_end only on success (not in finally)", "kill",
     [(MK, "    def __exit__(self, et, ev, tb):\n        self.t1 = self.mw.now()\n        self.mw.warmup_end(self.boot_id, self.t1)",
       "    def __exit__(self, et, ev, tb):\n        self.t1 = self.mw.now()\n        if ev is None:\n            self.mw.warmup_end(self.boot_id, self.t1)")]),
    ("JH43", "co-shape warm-up sends all three co requests at once", "kill",
     [(LG, "            for group in ((0,), (1, 2)):", "            for group in ((0, 1, 2),):")]),
    # ---- n2_window
    ("JH44", "Δdropped from the last record INSIDE the cell", "kill",
     [(WN, "    after = [e for e in records if _ts(e) > t1]", "    after = [e for e in records if t0 <= _ts(e) <= t1]")]),
    ("JH45", "F-6: no record after the cell counts as 0", "kill",
     [(WN, "    if not after:\n        return None", "    if not after:\n        return 0")]),
    ("JH46", "boot R7 marker ignored", "kill", [(WN, "    return R7_MARK in boot[\"log\"]", "    return False")]),
    ("JH47", "CG evidence counts Prefill lines too", "kill",
     [(WN, 'and txt.startswith("Decode batch")]', 'and "batch" in txt]')]),
    ("JH48", "CG range keeps the start boundary second", "kill",
     [(WN, "    lo, hi = math.ceil(to_wall(boot, cell_id, bd[0])), math.floor(to_wall(boot, cell_id, bd[1])) - 1",
       "    lo, hi = math.floor(to_wall(boot, cell_id, bd[0])), math.floor(to_wall(boot, cell_id, bd[1])) - 1")]),
    ("JH49", "RETRACT ignored", "kill", [(WN, "    if retract_count(boot, cid) > 0:", "    if False:")]),
    ("JH50", "missing prompt_tokens passes bench validity", "kill",
     [(WN, '        if q.get("prompt_tokens") is None or int(q["prompt_tokens"]) != cell["c_in"]:',
       '        if q.get("prompt_tokens") is not None and int(q["prompt_tokens"]) != cell["c_in"]:')]),
    ("JH51", "bench validity does not require the cancel rule to have fired", "kill",
     [(WN, '        if not (finished or (client.get("cancel_fired") and not q.get("done"))):',
       '        if not (finished or not q.get("done")):')]),
    ("JH52", "alone windows judged at the boot's FIXED pair (OFF: not (0,108))", "kill",
     [(WN, '    pi_alone = PAIR[unit["alone_D"]]\n    out = dict(boot=boot["name"], unit=unit["name"], cell_id=cell["cell_id"], kind=cell["kind"],',
       '    pi_alone = PAIR[unit["D"]]\n    out = dict(boot=boot["name"], unit=unit["name"], cell_id=cell["cell_id"], kind=cell["kind"],')]),
    ("JH53", "OBS client window: >= 128 intervals not required", "kill",
     [(WN, "        if len(iv) < WIN:\n            return None, nmin", "        if not iv:\n            return None, nmin")]),
    ("JH75", "stall count uses a 20x threshold (descriptive stall count no longer 10x)", "kill",
     [(WN, "STALL_MULTS = (5.0, 10.0) ", "STALL_MULTS = (5.0, 20.0) ")]),
    ("JH76", "N2C-75 endpoint-prefill count ignores the window end (last t_sync)", "kill",
     [(WN, '        ends = (w[0]["t_launch"], w[-1].get("t_sync"))', '        ends = (w[0]["t_launch"],)')]),
    # ---- n2_analyze
    ("JH54", "108 point from OFF16 boots only", "kill",
     [(AN, '                rs = [r for r in reps if r["kind"] == "PROFILE" and _point_key(r) == (D, b, c)]',
       '                rs = [r for r in reps if r["kind"] == "PROFILE" and _point_key(r) == (D, b, c) and (D != 108 or r["D"] == 16)]')]),
    ("JH55", "point label n >= 3", "kill",
     [(AN, 'label="POINT_REALIZED" if len(valid) >= MIN_N else "UNRESOLVED_N_BELOW_4")',
       'label="POINT_REALIZED" if len(valid) >= 3 else "UNRESOLVED_N_BELOW_4")')]),
    ("JH56", "D label: NOT_ATTEMPTED_BUDGET counted in the attempted set", "kill",
     [(AN, 'r["alone_D"] == D and r["status"] != "NOT_ATTEMPTED_BUDGET"]', 'r["alone_D"] == D]')]),
    ("JH57", "NEG_CONTROL population = VALID windows only", "kill",
     [(AN, '           and r["status"] in ("VALID", "D_UNREALIZED") and r.get("window_len") == nw.WIN]',
       '           and r["status"] == "VALID" and r.get("window_len") == nw.WIN]')]),
    ("JH58", "NEG_CONTROL empty population -> HELD", "kill",
     [(AN, '    if not pop:\n        return dict(label="NEG_CONTROL_UNRESOLVED", n=0)',
       '    if not pop:\n        return dict(label="NEG_CONTROL_HELD", n=0, min_pe_idx5=None)')]),
    ("JH59", "NEG_CONTROL ON-side clause revived (ON windows in the population)", "kill",
     [(AN, '    pop = [r for r in reps if r["kind"] == "PROFILE" and r["boot_kind"] == "OFF"',
       '    pop = [r for r in reps if r["kind"] == "PROFILE" and r["boot_kind"] in ("OFF", "ON")')]),
    ("JH60", "PROFILE label ignores NEG_CONTROL", "kill",
     [(AN, '    if neg["label"] != "NEG_CONTROL_HELD" or n_real == 0:', '    if n_real == 0:')]),
    ("JH61", "D-sensitivity: 3/4 majority instead of 4/4", "kill",
     [(AN, "    if all(x > LN3 for x in xs):\n        return \"D_SENS_GE3\"",
       "    if sum(x > LN3 for x in xs) >= 3:\n        return \"D_SENS_GE3\"")]),
    ("JH62", "D-sensitivity: unpaired rounds accepted", "kill",
     [(AN, "    if set(si) != set(sj):\n        raise ValueError", "    if False:\n        raise ValueError")]),
    ("JH63", "D = 108 round value from the first valid boot only (not pooled)", "kill",
     [(AN, "        if D == 108:\n            pooled = [v for x in vs for v in x[\"stats\"][\"intervals\"]]",
       "        if D == 108:\n            pooled = list(vs[0][\"stats\"][\"intervals\"])")]),
    ("JH64", "OBSERVER: CI-contains-0 condition dropped", "kill",
     [(AN, '        ok = (len(xs) >= MIN_N and abs(statistics.fmean(xs)) < LN3 and ci is not None\n              and ci["ci95_low"] <= 0.0 <= ci["ci95_high"])',
       '        ok = (len(xs) >= MIN_N and abs(statistics.fmean(xs)) < LN3)')]),
    ("JH65", "OBSERVER: n >= 3", "kill",
     [(AN, "        ok = (len(xs) >= MIN_N and", "        ok = (len(xs) >= 3 and"),
      (AN, "    elif any(v[\"n\"] < MIN_N for v in cells.values()):", "    elif any(v[\"n\"] < 3 for v in cells.values()):")]),
    ("JH66", "STICKY_CORRECTNESS compares text only", "kill",
     [(AN, 'x.get("text") == y.get("text") and x.get("output_ids") == y.get("output_ids")', 'x.get("text") == y.get("text")')]),
    ("JH67", "NOT_ATTEMPTED_BUDGET boots reported as BENCH_INVALID", "kill",
     [(AN, '                if st == "NOT_ATTEMPTED_BUDGET":\n                    reps.append(dict(base, status="NOT_ATTEMPTED_BUDGET"',
       '                if False:\n                    reps.append(dict(base, status="NOT_ATTEMPTED_BUDGET"')]),
    # ---- n2_realized_mix
    ("JH68", "H15-4 blind refusal removed", "kill",
     [(MX, '    if scope != "window" and job == "PROFILE" and not blind_exception:', '    if False:')]),
    ("JH69", "E-time weighted by the gap to the PREVIOUS snapshot", "kill",
     [(MX, '        gap = snaps[i + 1]["timestamp_monotonic_s"] - snaps[i]["timestamp_monotonic_s"]',
       '        gap = (snaps[i]["timestamp_monotonic_s"] - snaps[i - 1]["timestamp_monotonic_s"]) if i else 0.0')]),
    ("JH70", "window NA (< 10 busy snapshots) removed", "kill",
     [(MX, "            if res[\"busy_n\"] < MIN_BUSY_FOR_WINDOW:", "            if False:")]),
    # ---- n2_capacity.sh
    ("JH71", "knob allowlist loop removed", "kill",
     [(SH, '  case "$ALLOW" in *" $v "*) ;; *) fail "ABORT_ENV unregistered knob $v is set" 2;; esac', '  :')]),
    ("JH72", "test knobs accepted inside a VESSL Job", "kill",
     [(SH, '  if [ -n "${!v:-}" ] && [ -n "${PDMUX_COMMIT:-}" ]; then', '  if false; then')]),
    ("JH73", "ABORT_REG_SOURCE result ignored", "kill",
     [(SH, '[ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_REG_SOURCE (see REG_CHECK.txt)" 3', ':')]),
    ("JH74", "Job-EA harness-seal check (PROFILE + EA sections) ignored by the entry script (H17-6)", "kill",
     [(SH, '  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_HARNESS_UNSEALED_EA $(cat "$OUT/HARNESS_SEAL_CHECK_EA.txt")" 3',
       '  :')]),
    # ---- rev17 harness-audit repairs (VERDICT_n2_3a_rev17_delta_harness_2026-10-11.md; PREREG §18-2-d #4/#10/#16,
    #      §18-2 two-stage harness seal)
    ("JH77", "H17-1: output_ids presence not required (None == None / text-only -> BYTE_IDENTICAL)", "kill",
     [(AN, "        if not all(_has_ids(o) for o in list(a) + list(b)):", "        if False:")]),
    ("JH78", "H17-1: an EMPTY output_ids list counts as present", "kill",
     [(AN, "    return isinstance(ids, list) and len(ids) > 0", "    return isinstance(ids, list)")]),
    ("JH79", "H17-1: runner does not record greedy output_ids absence", "kill",
     [(JB, '                    rec["greedy_output_ids_absent"] = n_noids', '                    rec["greedy_output_ids_absent"] = 0')]),
    ("JH80", "H17-2: F-N2-7P ignores a non-budget stop (stopped / job_abort)", "kill",
     [(AN, '    f["F-N2-7P"] = ("UNRESOLVED" if ledger_full is None or non_budget_stop',
       '    f["F-N2-7P"] = ("UNRESOLVED" if ledger_full is None')]),
    ("JH81", "H17-2: F-N2-7P ignores the top-level job_abort field", "kill",
     [(AN, ' or led.get("job_abort") is not None', '')]),
    ("JH82", "H17-2: runner leaves JOB_LEDGER.stopped None after a Job abort", "kill",
     [(JB, '        stopped = "job_abort"\n', '        pass\n')]),
    ("JH83", "H17-3 (#16): CG canary not enforced", "kill",
     [(JB, "                if not ok_cg:                                              # §18-2-d #16", "                if False:")]),
    ("JH84", "H17-3: CG canary passes with zero Decode batch lines in the warm-up", "kill",
     [(WN, "    return (bool(canary) and n_bad == 0), len(canary), n_bad, lo, hi",
       "    return (n_bad == 0), len(canary), n_bad, lo, hi")]),
    ("JH85", "H17-3: CG canary passes when ANY warm-up line is cuda graph True (not ALL)", "kill",
     [(WN, '    n_bad = sum(1 for txt in canary if "cuda graph: True" not in txt)',
       '    n_bad = 0 if any("cuda graph: True" in t for t in canary) else len(canary)')]),
    ("JH86", "H17-3: CG canary reads Decode batch lines outside the warm-up seconds", "kill",
     [(WN, 'if txt.startswith("Decode batch") and lo <= ts <= hi]', 'if txt.startswith("Decode batch")]')]),
    ("JH87", "H17-3: CG canary keeps the warm-up start boundary second (floor, not ceil)", "kill",
     [(WN, "    lo, hi = math.ceil(w0), math.floor(w1) - 1", "    lo, hi = math.floor(w0), math.floor(w1) - 1")]),
    ("JH88", "H17-6: a Job-EA needs only the EA section (PROFILE section not required)", "kill",
     [(PL, 'REQUIRED_SCOPES = {"PROFILE": ("PROFILE",), "EA": ("PROFILE", "EA")}',
       'REQUIRED_SCOPES = {"PROFILE": ("PROFILE",), "EA": ("EA",)}')]),
    ("JH89", "H17-6: a scope sealed twice is accepted", "kill",
     [(PL, "        if len(lst) != 1:\n", "        if False:\n")]),
    ("JH90", "H17-6: an untagged section is read as scope PROFILE (legacy format accepted)", "kill",
     [(PL, "            probs.append(f\"section needs exactly one 'scope' line (got {sc})\")",
       '            sc = sc[:1] or ["PROFILE"]')]),
    ("JH91", "H17-6: an unknown scope tag is accepted", "kill",
     [(PL, "        elif sc[0] not in HARNESS_SCOPES:", "        elif False:")]),
    ("JH92", "H17-6: a Job-PROFILE's harness-seal check asks for the wrong scope", "kill",
     [(SH, 'python3 "$PLAN" --verify-harness-seal --scope PROFILE | tee "$OUT/HARNESS_SEAL_CHECK.txt"',
       'python3 "$PLAN" --verify-harness-seal --scope EA | tee "$OUT/HARNESS_SEAL_CHECK.txt"')]),
    ("JH93", "H17-6: a sealed Job-EA falls through into the PROFILE flow (no ABORT_EA_NOT_IMPLEMENTED)", "kill",
     [(SH, '  fail "ABORT_EA_NOT_IMPLEMENTED Job-EA runner is not implemented (H15-3)" 3', '  :')]),
    ("JH94", "H17-8: runner crash leaves stopped=None (F-N2-7P false FALSIFIED)", "kill",
     [(JB, '        stopped = stopped or "crash"', '        pass')]),
    ("JH95", "H17-10: n2_plan.ENG ignores PDMUX_PROJECT_ROOT (file-relative root in a symlinked Job layout)", "kill",
     [(PL, '       if os.environ.get("PDMUX_PROJECT_ROOT") else', '       if False else')]),
    ("JH96", "H17-10: n2_realized_mix.ENG ignores PDMUX_PROJECT_ROOT", "kill",
     [(MX, '       if os.environ.get("PDMUX_PROJECT_ROOT") else', '       if False else')]),
    ("JH97", "H17-10: preflight does not override N2_N3_DIR (empty tracked N3 dir fails the positive control)", "kill",
     [(SH, '(cd "$HERE" && N2_N3_DIR="$OUT/N3_RAW_NOT_IN_JOB" timeout', '(cd "$HERE" && timeout')]),
]

def _tree(tmp, files):
    res = os.path.join(tmp, "workspace", "engine-port", "results")
    d = os.path.join(res, "n2_static_grid")
    os.makedirs(d)
    for fn, txt in files.items():
        with open(os.path.join(d, fn), "w") as fh:
            fh.write(txt)
    if os.path.exists(os.path.join(d, SH)):
        os.chmod(os.path.join(d, SH), 0o755)
    for sub in ("n3_capacity", "r2_eval"):
        os.symlink(os.path.join(ENG, "results", sub), os.path.join(res, sub))
    for sub in ("benchmarks", "src", "scripts", "tests"):
        os.symlink(os.path.join(ENG, sub), os.path.join(tmp, "workspace", "engine-port", sub))
    return d


def _sources():
    out = {}
    for fn in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and not fn.endswith(".pyc") and not fn.startswith("n2p_"):
            with open(p) as fh:
                out[fn] = fh.read()
    return out


def run(selected=None):
    src = _sources()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    rows = []
    muts = [m for m in JOB_MUTANTS if not selected or m[0] in selected]
    for mid, desc, expect, reps in [("CTRL", "no mutation", "pass", [])] + muts:
        files = dict(src)
        broken = False
        for fn, old, new in reps:
            if files[fn].count(old) != 1:
                rows.append((mid, desc, "BROKEN-MUTANT", f"anchor count {files[fn].count(old)} in {fn}"))
                broken = True
                break
            files[fn] = files[fn].replace(old, new)
        if not broken:
            tmp = tempfile.mkdtemp(prefix=f"n2jmut_{mid}_")
            try:
                d = _tree(tmp, files)
                if mid == "CTRL":
                    suites = SUITES_ALL
                elif any(fn in (MK, LG) for fn, _o, _n in reps):
                    suites = ("test_n2_harness", "test_n2_job")
                else:
                    suites = ("test_n2_job", "n2_reach_tests")
                try:
                    r = subprocess.run([sys.executable, "-m", "unittest", *suites], cwd=d, env=env,
                                       capture_output=True, text=True, timeout=900)
                except subprocess.TimeoutExpired:
                    rows.append((mid, desc, "TIMEOUT", "900 s"))
                    print(f"{mid:5s} {desc:70s} {'TIMEOUT':20s} 900 s", flush=True)
                    continue
                pyc = any(x.endswith(".pyc") for _, _, fs in os.walk(d) for x in fs)
                tail = [ln for ln in r.stderr.strip().split("\n") if ln.startswith(("OK", "FAILED"))]
                tail = tail[-1] if tail else r.stderr[-200:].replace("\n", " ")
                failed = r.returncode != 0
                if expect == "pass":
                    status = "CONTROL-OK" if not failed else "CONTROL-FAILED"
                elif expect == "kill":
                    status = "KILLED" if failed else "SURVIVED"
                else:
                    status = "EQUIVALENT-SURVIVED" if not failed else "EQUIVALENT-KILLED"
                rows.append((mid, desc, status, f"{tail}{' pyc!' if pyc else ''}"))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        print(f"{rows[-1][0]:5s} {rows[-1][1]:70s} {rows[-1][2]:20s} {rows[-1][3]}", flush=True)
        if rows[-1][2] == "CONTROL-FAILED":
            raise SystemExit("CONTROL-FAILED: the unmutated suites do not pass in the mutant tree -- aborting")
    return rows


if __name__ == "__main__":
    run(set(sys.argv[1:]) or None)
