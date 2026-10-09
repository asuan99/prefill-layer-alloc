"""N2(3-a) registered mutant runner (CPU only) -- PREREG_N2_3A rev15 ((A') fixed K: judgment-path group J + theta-provenance group T + smoke group S; R8-5, R9-3, R9-8, R10-4, R11-6, G-6/8/9/15, harness H01-H32; N2C-47 issue: every registered mutant
must be re-runnable by an auditor).

Run:  PYTHONDONTWRITEBYTECODE=1 python3 n2_mutants.py [N3_RAW_DIR]
rev15 (R14-9): three groups are printed per row -- J (judgment path), T (theta_rho provenance: theta sample
composition of the shared span-pairing implementation), S (smoke tool chain). A CONTROL-FAILED row aborts the run
(lesson 254: a failing unmutated suite voids every later row). N2_N3_SH is resolved location-independently
(env override, git top level, HERE-relative) and must exist.
Requirements (R9-8): run SEQUENTIALLY and alone -- each mutant runs the full suite in a subprocess, and the N3 positive
control loads ~2.0 GB of JSONL (36 files) into one process; concurrent heavy processes caused swap and 900 s timeouts
in the rev9 audit. A timeout is recorded as a TIMEOUT row (never counted as a kill) and the run continues.
Each mutant is applied to a fresh copy of n2_feasibility.py / n2_probe_reduce.py in its own temp directory together
with both test files; the full suite runs with bytecode disabled (no .pyc is checked); a no-mutation control runs
first. Output: one line per mutant, KILLED / SURVIVED / EQUIVALENT-SURVIVED (expected) / UNEXPECTED, with FAIL counts.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FE, RD = "n2_feasibility.py", "n2_probe_reduce.py"
TESTS = ("test_n2_feasibility.py", "test_n2_probe_reduce.py")
# rev13: the U-2 probe harness is part of the registered tool set. Every mutant directory gets all of these files;
# a mutant of a CORE file (calculator / reducer) runs the core suites, a mutant of a HARNESS file runs the harness
# suite (the core suites import no harness module, so they cannot kill a harness mutant); the control runs all three.
MK, LG, RN = "n2_marks.py", "n2_loadgen.py", "n2_probe_run.py"
HARNESS_SUPPORT = (MK, LG, RN, "fake_n2_engine.py")
HARNESS_TESTS = ("test_n2_harness.py",)

# (id, description, expect, [(file, old, new), ...]); expect = "kill" or "equivalent"
MUTANTS = [
    ("R01", 'K_16 = 32 (registered constant off by one, rev14 re-pin)', "kill", [(FE, 'K_CO = {16: 33, 44: 15}', 'K_CO = {16: 32, 44: 15}')]),
    ("R02", 'K_44 = 16 (registered constant off by one, rev14 re-pin)', "kill", [(FE, 'K_CO = {16: 33, 44: 15}', 'K_CO = {16: 33, 44: 16}')]),
    ("R03", 'L_LAG = 0 (rev14 re-pin)', "kill", [(FE, 'L_LAG = 128 ', 'L_LAG = 0 ')]),
    ("R04", "ealpha_c_in -> c_in(108)", "kill", [(FE, 'use c_in(d)."""\n    return c_in_of(d, b, c)', 'use c_in(d)."""\n    return c_in_of(108, b, c)')]),
    ("R05", "choose_k returns K+1", "kill", [(FE, "        if k_satisfies(d, K):\n            return K", "        if k_satisfies(d, K):\n            return K + 1")]),
    ("R06", "kappa_DW fixed 1.6", "kill", [(FE, 'return k * KAPPA_DW_OVER_DN if dcorner == "DW" else k', 'return 1.6 if dcorner == "DW" else k')]),
    ("R07", "theta rho max->min across p", "kill", [(FE, 'new["rho"]["theta"] = max(med.values())', 'new["rho"]["theta"] = min(med.values())')]),
    ("R08", 'kappa* requirement 2.0 (rev14 re-pin from 1.6)', "kill", [(FE, 'RHO_MAX_REQUIRED = 3.0 ', 'RHO_MAX_REQUIRED = 2.0 ')]),
    ("R09", 'O_EA drops the delta allowance (rev14 retarget)', "kill", [(FE, '+ L_LAG + n_co_max(d, b, c) + TRIM + WIN + DELTA_FIXED + s_hat(d, b, c)', '+ L_LAG + n_co_max(d, b, c) + TRIM + WIN + s_hat(d, b, c)')]),
    ("R10", "cell_time_ealpha cin = c_in(Dalone)", "kill", [(FE, "    cin = ealpha_c_in(d, b, c)\n    k = max(1", "    cin = c_in_of(Dalone, b, c)\n    k = max(1")]),
    ("R11", 'call-site delta uses d (rev14: equivalent -- delta is the fixed 64 for every window)', "equivalent", [(FE, 'delta_tokens(Dalone, b, c, dc)', 'delta_tokens(d, b, c, dc)')]),
    ("R12", "tau accepts D not in 16,44", "kill", [(FE, 'D, b, c = _check_cell(cell, "tau", (16, 44))', 'D, b, c = _check_cell(cell, "tau", (16, 44, 24, 34, 108))'), (FE, "        ratios[D].append(ms / model)", "        ratios.setdefault(D, []).append(ms / model)")]),
    ("R13", "tau rho uses min", "kill", [(FE, "hi = {D: max(r) for D, r in ratios.items()}", "hi = {D: min(r) for D, r in ratios.items()}")]),
    ("R14", "delete W_co condition", "kill", [(FE, "                if min(w_co(d, b, c, pc, dc) for pc, dc in CORNERS) < C_WINDOW_MIN:\n                    return False\n", "")]),
    ("R15", "kappa rho median", "kill", [(FE, 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)', 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = statistics.median(ks)')]),
    ("R16", "delete L_LAG in E-alpha pre", "kill", [(FE, "    pre = (c + 71 - start) + L_LAG", "    pre = (c + 71 - start)")]),
    ("R17", "s_hat rho direction", "kill", [(FE, 'return max(burst_spread(D, b, c, pc, dc, "env") for pc, dc in CORNERS)', 'return max(burst_spread(D, b, c, pc, dc, "rho") for pc, dc in CORNERS)')]),
    ("R18", 'N_co,max uses the rho (fast) theta (rev14 retarget)', "kill", [(FE, 'co_prefill_time(Dsp, "PW", "budget") / (tau_alone_lower(b, c) * KAPPA_FLOOR)', 'co_prefill_time(Dsp, "PW", "rho") / (tau_alone_lower(b, c) * KAPPA_FLOOR)')]),
    ("R19", "theta median -> min (D-9)", "kill", [(FE, 'med[p] = statistics.median(_scalar(v, "theta") for v in lst) * 64.0 / p', 'med[p] = min(_scalar(v, "theta") for v in lst) * 64.0 / p')]),
    ("R20", "accept n < 20", "kill", [(FE, "len(lst) < THETA_MIN_SPANS", "len(lst) < 1")]),
    ("R21", "accept iteration-level raw", "kill", [(FE, "    if isinstance(x, (list, tuple, dict)):\n        raise ProbeRejected", "    if isinstance(x, (list, tuple, dict)):\n        return float(max(x))\n        raise ProbeRejected")]),
    ("R22", "restore K_SEARCH_MAX = 64", "kill", [(FE, "    K = 1\n    while True:", "    K = 1\n    while K <= 64:"), (FE, "        K += 1\n", "        K += 1\n    return None\n")]),
    ("R23", 'smoke gate skips limits_check', "kill", [(FE, '    lim = limits_check()\n    if not lim["ok"]:\n        reasons.append(f"limits_check failed', '    lim = limits_check()\n    if False:\n        reasons.append(f"limits_check failed')]),
    ("A01", "budget tau -> lo", "kill", [(FE, 'for dirn, s in (("rho", hi), ("budget", hi), ("env", lo)):', 'for dirn, s in (("rho", hi), ("budget", lo), ("env", lo)):')]),
    ("A02", "budget kappa -> min", "kill", [(FE, 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)\n    new["env"]["kappa_dn"] = min(ks)', 'new["rho"]["kappa_dn"] = max(ks)\n    new["budget"]["kappa_dn"] = min(ks)\n    new["env"]["kappa_dn"] = min(ks)')]),
    ("A04", "E-alpha cost co_prefill rho", "kill", [(FE, '    tco = co_prefill_time(d, pc, "budget")', '    tco = co_prefill_time(d, pc, "rho")')]),
    ("A05", "PROFILE cost theta rho", "kill", [(FE, '    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[D], pc, "budget")\n    s = burst_spread(D, b, cin, pc, dc, "budget")\n    start = cin + 1 + s / 2\n    steps = (c + 63', '    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[D], pc, "rho")\n    s = burst_spread(D, b, cin, pc, dc, "budget")\n    start = cin + 1 + s / 2\n    steps = (c + 63')]),
    ("A06", 'drop tau binding-cell check', "kill", [(FE, '    for cell in BINDING_EA_CELLS:\n        if cell not in tc:\n            raise ProbeRejected(f"tau cells must include binding cell {cell}", value_independent=True,\n                                code="TAU_BINDING_MISSING")\n', '')]),
    ("A07", "drop A_on window check", "kill", [(FE, ' \\\n                or int(v.get("a_on_window_iters", 0)) < WINDOW_ITERS:', ":")]),
    ("A08", "required p (92,) only", "kill", [(FE, "THETA_REQUIRED_P = (64, 92)", "THETA_REQUIRED_P = (92,)")]),
    ("A10", "probe sha from calculator file", "kill", [(FE, 'rec["probe_sha256"] = _sha256_file(probe_path)', 'rec["probe_sha256"] = _sha256_file(__file__)')]),
    ("A11", "CLI rc always 0", "kill", [(FE, 'return 0 if rec["seal_allowed"] else 2', "return 0")]),
    ("A13", 'delta_exec returns DELTA_MIN (rev14 retarget)', "kill", [(FE, '    """rev14: execution cancel margin = DELTA_FIXED (kept as a function so the harness plan keeps its field)."""\n    return DELTA_FIXED', '    """rev14: execution cancel margin = DELTA_FIXED (kept as a function so the harness plan keeps its field)."""\n    return DELTA_MIN')]),
    ("A09", "drop KV limit", "equivalent", [(FE, "                if L > CONTEXT_LEN or b * L > KV_TOKENS:", "                if L > CONTEXT_LEN:")]),
    ("A12", 'smoke gate drops the registered-K kappa* check (rev14: now a real check)', "kill", [(FE, '        if not k_satisfies(d, K_CO[d]):', '        if False:')]),
    ("A16", "search starts at K = 2", "equivalent", [(FE, "    K = 1\n    while True:", "    K = 2\n    while True:")]),
    ("G01", 'smoke gate diagnostic K hard-coded', "kill", [(FE, '    diag = {d: choose_k(d) for d in EA_D}\n    reasons = []', '    diag = {16: 18, 44: 8}\n    reasons = []')]),
    ("G02", 'smoke gate omits derived constants', "kill", [(FE, '               derived=derived_constants(), budget=budget_record())', '               derived={}, budget=budget_record())')]),
    ("G03", "CLI accepts duplicate keys", "kill", [(FE, "raw = json.load(fh, object_pairs_hook=_no_duplicate_keys)", "raw = json.load(fh)")]),
    ("C09", 'derived delta budget direction (rev14: equivalent -- delta_tokens == delta_exec == DELTA_FIXED)', "equivalent", [(FE, 'delta_on_A2=delta_exec(d, b, c), delta_off_A2=delta_exec(108, b, c)', 'delta_on_A2=delta_tokens(d, b, c), delta_off_A2=delta_tokens(108, b, c)')]),
    ("M01", "budget record FIX_S hard-coded", "kill", [(FE, '    out = {"FIX_S": dict(FIX_S),', '    out = {"FIX_S": {"nom": 22.48, "worst": 35.5},')]),
    ("M02", 'reducer sha check removed', "kill", [(FE, '        if not rec["reducer_sha256"] or rec["reducer_sha256"] != local_red:', '        if False:')]),
    ("P01", "reducer no first-span exclusion", "kill", [(RD, "        if k in first:\n            excluded", "        if False:\n            excluded")]),
    ("P02", "reducer only boot-first excluded", "kill", [(RD, "    for a, b in load_calls:\n        inside", "    for a, b in []:\n        inside")]),
    ("P03", "pooled median -> median of medians", "kill", [(RD, "    pooled = []\n    for w in windows:\n        pooled += s_per_intervals(w)\n    return statistics.median(pooled)", "    return statistics.median(statistics.median(s_per_intervals(w)) for w in windows)")]),
    ("P04", "reducer accepts duplicate keys", "kill", [(RD, '        raise ReduceRejected("duplicate JSON keys")', "        pass")]),
    ("P05", "reducer drops decode-load filter", "kill", [(RD, '        if max(e["bs"] for e in its) > BIND_BS:', "        if False:")]),
    ("P06", "reducer drops realized-split filter", "kill", [(RD, '        split = [e for e in its if 0 < e["prefill_sms"] < 108 and e["decode_sms"] > 0]', '        split = [e for e in its if e["prefill_sms"] < 108 and e["decode_sms"] >= 0] or its')]),
    ("P07", "reducer accepts short window (2)", "kill", [(RD, "    if len(window_iters) < WINDOW_ITERS:", "    if len(window_iters) < 2:")]),
    ("B01", "realized split = ALL iterations", "kill", [(RD, "        if not split:\n", "        if not split or len(split) != len(its):\n")]),
    ("B02", "mixed-p check deleted", "kill", [(RD, "        if len(ps) != 1:", "        if False:")]),
    ("B03", "decode load from split iterations only", "kill", [(RD, '        if max(e["bs"] for e in its) > BIND_BS:', '        if max(e["bs"] for e in split) > BIND_BS:')]),
    ("B04", "load-call membership by span END", "kill", [(RD, "        inside = [s for s in spans if a <= s[0] <= b]", "        inside = [s for s in spans if a <= s[1] <= b]")]),
    ("B06", "S-per over sorted neighbours (not iter_seq)", "kill", [(RD, "        n = by_seq.get(q + 1)", "        n = by_seq.get(min((k for k in by_seq if k > q), default=None))")]),
    ("B16", "window 127 accepted", "kill", [(RD, "    if len(window_iters) < WINDOW_ITERS:", "    if len(window_iters) < WINDOW_ITERS - 1:")]),
    ("B18", "kappa A_on from the C window", "kill", [(RD, '    a1 = statistics.median(s_per_intervals(win["W"]))\n    a2 = statistics.median(s_per_intervals(win["A2"]))', '    a1 = statistics.median(s_per_intervals(win["C"]))\n    a2 = statistics.median(s_per_intervals(win["C"]))')]),
    ("B19", "kappa C from the A_on window", "kill", [(RD, '    c_med = statistics.median(s_per_intervals(win["C"]))', '    c_med = statistics.median(s_per_intervals(win["W"]))')]),
    ("B21", "to_json key format", "kill", [(RD, "        return json.dumps(list(k)) if isinstance(k, tuple) else str(k)", "        return str(k)")]),
    ("C14", 'C window contamination (bs filter only)', "kill", [(RD, '    call = [e for e in inflt if _pair(e) == pi_coloc]', '    call = list(wco)')]),
    ("M03", "contaminated snapshots not excluded", "kill", [(RD, '                   if s_.get("active_decode_sequences") == b)', "                   if True)")]),
    ("M04", "A_on pooled instead of interpolated", "kill", [(RD, '    return c_med, a_on_interp(a1, m["A1"], a2, m["A2"], m["C"])', '    return c_med, statistics.median(s_per_intervals(win["W"]) + s_per_intervals(win["A2"]))')]),
    # ---- rev10 (R9-3): audit rev9 X mutants adapted to the rev10 code, and new rev10 rules (M05-M12) ----
    ("X01", "kappa length-weighted over measurements (pooled-C analogue)", "kill", [(RD, '            kap.setdefault(cell, []).append(out["c_med"] / out["a_on"])', '            kap.setdefault(cell, []).extend([out["c_med"] / out["a_on"]] * len(win["C"]))')]),
    ("X02", "kappa cell = mean of kappa_i", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": statistics.mean(ks),')]),
    ("X04", "reduce_probe silently drops failing measurements", "kill", [(RD, "            if issues:\n                worst = max(issues, key=lambda x: SEVERITY[x[0]])\n", "            if issues:\n                continue\n                worst = max(issues, key=lambda x: SEVERITY[x[0]])\n")]),
    ("X05", 'C filter drops prefill_in_flight only (pair kept)', "kill", [(RD, '    call = [e for e in inflt if _pair(e) == pi_coloc]', '    call = [e for e in bs_eq if _pair(e) == pi_coloc]')]),
    ("X07", "TRIM = 0", "kill", [(RD, "TRIM = 16", "TRIM = 0")]),
    ("X08", 'A2 = last 128 instead of first', "kill", [(RD, '    a2 = seg2[:WINDOW_ITERS]', '    a2 = seg2[-WINDOW_ITERS:]')]),
    ("X09", "co span: drop extend_tokens == n_new*8192 check", "kill", [(RD, 'sp[0] >= t_co and sp[2]["extend_tokens"] == sp[2]["n_new"] * CO_INPUT]', 'sp[0] >= t_co]')]),
    ("X12", "A_on containment guard removed", "kill", [(RD, "    if not min(x1, x2) <= x <= max(x1, x2):\n", "    if False:\n")]),
    ("X13", "raw file sha replaced by a constant", "kill", [(RD, "            raw_files[bt[k]] = sha256_file(bt[k])", "            raw_files[bt[k]] = \"0\" * 64")]),
    ("X14", "alone windows drop the not-in-flight filter", "kill", [(RD, 'e["bs"] == b and not e["prefill_in_flight"]]', 'e["bs"] == b]')]),
    ("X15", "theta: load-call exclusion skips the first candidate", "kill", [(RD, "        inside = [s for s in spans if a <= s[0] <= b]", "        inside = [s for s in spans if a <= s[0] <= b][1:]")]),
    ("M06", "probe status priority: value-dependent before censored", "kill", [(RD, "(KAPPA_CENSORED, VALUE_DEPENDENT, VALUE_INDEPENDENT)", "(VALUE_DEPENDENT, KAPPA_CENSORED, VALUE_INDEPENDENT)")]),
    ("M07", 'CLI allows re-probe after a value-dependent reject', "kill", [(FE, 'reprobe_allowed=(status == "VALUE_INDEPENDENT"), reason_codes=', 'reprobe_allowed=(status in ("VALUE_INDEPENDENT", "VALUE_DEPENDENT")), reason_codes=')]),
    ("M08", "CLI freeze-manifest check removed", "kill", [(FE, '        bad = check_freeze_manifest(freeze_path, rec["calculator_sha256"], local_red)', "        bad = []")]),
    ("M09", "tau not taken from the kappa A1 window (R9-2)", "kill", [(RD, "            tau_w.setdefault(cell, []).append(win[\"W\"])", "            pass")]),
    ("M10", 'C < 128 classified VALUE_DEPENDENT instead of KAPPA_CENSORED (smoke wrapper)', "kill", [(RD, '("C", "CO_WINDOW_SHORT"): (KAPPA_CENSORED, "C_CENSORED"),', '("C", "CO_WINDOW_SHORT"): (VALUE_DEPENDENT, "C_CENSORED"),')]),
    ("M11", 'n < 20 not value-independent', "kill", [(FE, 'value_independent=True, code="THETA_N_SHORT")', 'value_independent=False, code="THETA_N_SHORT")')]),
    ("M12", "negative L_lag accepted", "kill", [(FE, "    if min(lags) < 0:", "    if False:")]),
    # ---- rev11 (R10-4): audit rev10 Y mutants adapted to the rev11 code ----
    ("Y01", "status priority VI before VD", "kill", [(RD, "    status = next((s for s in (KAPPA_CENSORED, VALUE_DEPENDENT, VALUE_INDEPENDENT)", "    status = next((s for s in (KAPPA_CENSORED, VALUE_INDEPENDENT, VALUE_DEPENDENT)")]),
    ("Y02", "kappa cell = median_low (even n)", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": statistics.median_low(ks),')]),
    ("Y03", "kappa cell = max kappa_i", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": max(ks),')]),
    ("Y04", 'censoring judged on W_co not the C value window', "kill", [(RD, '    elif len(cwin) < WINDOW_ITERS:\n        out["status"]["C"] = "CO_WINDOW_SHORT"', '    elif len(wco) < WINDOW_ITERS:\n        out["status"]["C"] = "CO_WINDOW_SHORT"')]),
    ("Y05", 'missing delta_l/fix_s -> VALUE_DEPENDENT', "kill", [(RD, 'issues.append((VALUE_INDEPENDENT, "SCALAR_MARK_MISSING", f"{key} mark missing"))', 'issues.append((VALUE_DEPENDENT, "SCALAR_MARK_MISSING", f"{key} mark missing"))')]),
    ("Y06", "PROBE_D check removed", "kill", [(RD, "    if cell[0] not in PROBE_D:\n        issues.append((VALUE_INDEPENDENT", "    if False:\n        issues.append((VALUE_INDEPENDENT")]),
    ("Y07", 'malformed mark lines ignored', "kill", [(RD, 'issues.append((VALUE_INDEPENDENT, "MARK_MALFORMED", f"malformed mark: {line}")); continue', 'continue')]),
    ("Y08", "freeze manifest: reducer sha not compared", "kill", [(FE, ' or by_name.get("n2_probe_reduce.py") != reducer_sha:', ":")]),
    ("Y09", 'missing binding kappa cell not value-independent', "kill", [(FE, 'raise ProbeRejected(f"kappa cells must include binding cell {cell}", value_independent=True,', 'raise ProbeRejected(f"kappa cells must include binding cell {cell}", value_independent=False,')]),
    ("Y10", 'C censor threshold 127', "kill", [(RD, '    elif len(cwin) < WINDOW_ITERS:\n        out["status"]["C"] = "CO_WINDOW_SHORT"', '    elif len(cwin) < WINDOW_ITERS - 1:\n        out["status"]["C"] = "CO_WINDOW_SHORT"')]),
    ("Y11", "l_lag always 0", "kill", [(RD, 'out["l_lag_iters"] = sum(1 for e in iters if t_co <= e["t_launch"] < t_co0)', 'out["l_lag_iters"] = 0')]),
    ("Y12", "tau from A2 instead of A1", "kill", [(RD, '            tau_w.setdefault(cell, []).append(win["W"])', '            tau_w.setdefault(cell, []).append(win["A2"])')]),
    ("Y13", 'calc reject always allows re-probe', "kill", [(FE, 'reprobe_allowed=e.value_independent, reason_codes=[e.code])', 'reprobe_allowed=True, reason_codes=[e.code])')]),
    ("Y14", 'VI probe status -> re-probe not allowed', "kill", [(FE, 'reprobe_allowed=(status == "VALUE_INDEPENDENT"), reason_codes=', 'reprobe_allowed=False, reason_codes=')]),
    # ---- rev11 new rules (R10-1/2/3, H-1/H-2, R10-5) ----
    ("N01", 'unknown cell_abort reason -> VALUE_INDEPENDENT', "kill", [(RD, 'issues.append((ABORT_REASONS.get(reason, VALUE_DEPENDENT), abort_code(reason)', 'issues.append((ABORT_REASONS.get(reason, VALUE_INDEPENDENT), abort_code(reason)')]),
    ("N02", 'missing cell_end/coload_sent without abort -> VI', "kill", [(RD, 'issues.append((VALUE_DEPENDENT, code, f"{key} mark missing without cell_abort"))', 'issues.append((VALUE_INDEPENDENT, code, f"{key} mark missing without cell_abort"))')]),
    ("N03", "orphan raw co spans not checked", "kill", [(RD, "                if n_in != 1:", "                if False:")]),
    ("N04", 'duplicate cell id marks silently skipped', "kill", [(RD, 'issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", f"duplicate {tag} for cell id {rest[0]}"))\n                    cm["dup"] = True', 'pass')]),
    ("N05", 'per-boot excess not checked', "kill", [(RD, '            if n_started > n_plan:', '            if False:')]),
    ("N06", "shortfall after this boot's death -> VD", "kill", [(RD, '                if death_t is not None and last <= death_t:', '                if False:')]),
    ("N07", "load-call wrapping not checked", "kill", [(RD, "        if n_calls != 1:", "        if False:")]),
    ("N08", "truncated last telemetry line treated as corruption", "kill", [(RD, "            if k == len(lines) - 1:", "            if False:")]),
    ("N09", "CLI re-reduction removed", "kill", [(FE, "        irreproducible = reproduce_probe(probe_path, raw)", "        irreproducible = None")]),
    ("N10", "scalar marks checked first (short-circuit)", "kill", [(RD, "    if cell[0] not in PROBE_D:\n", "    if \"fix_s\" not in cm or \"delta_l\" not in cm:\n        return [(VALUE_INDEPENDENT, \"scalar mark missing\")], None\n    if cell[0] not in PROBE_D:\n")]),
    ("N11", "manifest files not verified", "kill", [(FE, "        if not os.path.exists(path) or _sha256_file(path) != sha:", "        if False:")]),
    ("N12", "harness mark writer not required in manifest", "kill", [(FE, "    if not isinstance(writers, list) or not writers or any(w not in files for w in writers):", "    if False:")]),
    ("N13", "registered call count 13 at d = 44", "kill", [(RD, "PROBE_CALLS = {16: 7, 44: 14}", "PROBE_CALLS = {16: 7, 44: 13}")]),
    ("N14", 'censor detail not recorded', "kill", [(RD, '**({"censor_detail": worst[3]} if len(worst) > 3 else {})', '**{}')]),
    ("N15", 'VI boot_abort not counted as boot death', "kill", [(RD, '                death_t = t_ab if death_t is None else min(death_t, t_ab)', '                pass')]),
    # ---- rev12 (R11-6): audit rev11 Z mutants adapted to the rev12 code, and rev12 rules (N05b/c, N16-N22) ----
    ("Z01", 'coload_payload_invalid -> VI', "kill", [(RD, '"coload_payload_invalid": VALUE_DEPENDENT', '"coload_payload_invalid": VALUE_INDEPENDENT')]),
    ("Z02", 'stream_cancel_failed -> VI', "kill", [(RD, '"stream_cancel_failed": VALUE_DEPENDENT,', '"stream_cancel_failed": VALUE_INDEPENDENT,')]),
    ("Z03", 'cell_timeout -> VI', "kill", [(RD, '"cell_timeout": VALUE_DEPENDENT,', '"cell_timeout": VALUE_INDEPENDENT,')]),
    ("Z04", 'co span inside >1 cell accepted', "kill", [(RD, '                if n_in != 1:', '                if n_in == 0:')]),
    ("Z05", 'coload_sent inside >=1 call accepted', "kill", [(RD, '        if n_calls != 1:', '        if n_calls < 1:')]),
    ("Z06", "excess + this boot's death -> VI", "kill", [(RD, 'boot_issues.append(_issue(bid, VALUE_DEPENDENT, "CELL_COUNT_EXCESS",', 'boot_issues.append(_issue(bid, VALUE_INDEPENDENT if death_t is not None else VALUE_DEPENDENT, "CELL_COUNT_EXCESS",')]),
    ("Z07", 'calculator re-reduces with the recorded expected_calls', "kill", [(FE, '        again = red.to_json(red.reduce_probe(boots))', "        _e = (raw.get('provenance') or {}).get('expected_calls')\n        again = red.to_json(red.reduce_probe(boots, {int(k): v for k, v in _e.items()} if _e else None))")]),
    ("Z08", 're-reduction compares probe_status only', "kill", [(FE, '        if fh.read() != again:', "        if json.loads(fh.read())['probe_status'] != json.loads(again)['probe_status']:")]),
    ("Z09", 'irreproducible probe -> re-probe allowed', "kill", [(FE, 'rec = dict(K=None, seal_allowed=False, reasons=[irreproducible[1]], reprobe_allowed=False,', 'rec = dict(K=None, seal_allowed=False, reasons=[irreproducible[1]], reprobe_allowed=True,')]),
    ("Z10", 'frozen-file dirty check removed', "kill", [(FE, '        if rc != 0 or dirty:', '        if False:')]),
    ("Z11", '_measure duplicate-id issue removed', "kill", [(RD, '    if cm.get("dup"):\n        issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", "duplicate cell id"))', '    if False:\n        issues.append((VALUE_DEPENDENT, "MARK_DUPLICATE", "duplicate cell id"))')]),
    ("Z12", 'last 3 unparsable lines = truncation', "kill", [(RD, '            if k == len(lines) - 1:', '            if k >= len(lines) - 3:')]),
    ("Z13", 'censor detail losses swapped', "kill", [(RD, 'loss_not_in_flight=len(bs_eq) - len(inflt), loss_pair_mismatch=len(inflt) - len(call))', 'loss_not_in_flight=len(inflt) - len(call), loss_pair_mismatch=len(bs_eq) - len(inflt))')]),
    ("Z14", 'any cell_abort -> VI', "kill", [(RD, 'issues.append((ABORT_REASONS.get(reason, VALUE_DEPENDENT), abort_code(reason), f"cell_abort {reason}"))', 'issues.append((VALUE_INDEPENDENT, abort_code(reason), f"cell_abort {reason}"))')]),
    ("Z15", 'unknown boot_abort reason -> VI', "kill", [(RD, '            cls = ABORT_REASONS.get(ab, VALUE_DEPENDENT)', '            cls = ABORT_REASONS.get(ab, VALUE_INDEPENDENT)')]),
    ("Z16", 'reducer failure -> VI', "kill", [(RD, '_issue(bid, VALUE_DEPENDENT, "REDUCER_FAILURE"', '_issue(bid, VALUE_INDEPENDENT, "REDUCER_FAILURE"')]),
    ("Z17", 'git HEAD == manifest.commit check removed', "kill", [(FE, '        if rc != 0 or not man.get("commit") or head != man.get("commit"):', '        if False:')]),
    ("Z18", 'window of a cell without cell_end closes at its start', "kill", [(RD, 'wins = [(cm["cell_start"][1], cm.get("cell_end", float("inf")))', 'wins = [(cm["cell_start"][1], cm.get("cell_end", cm["cell_start"][1]))')]),
    ("Z19", 'truncation not a boot death', "kill", [(RD, '            death_t = max((_ts(e) for e in records), default=float("-inf"))', '            pass')]),
    ("Z20", 'raw absence -> VALUE_DEPENDENT', "kill", [(RD, '_issue(bid, VALUE_INDEPENDENT, "RAW_MISSING"', '_issue(bid, VALUE_DEPENDENT, "RAW_MISSING"')]),
    ("N05b", 'boot plan sum not checked', "kill", [(RD, '            if plan_total[d] != expected.get(d, 0):', '            if False:')]),
    ("N05c", 'per-boot shortfall not checked', "kill", [(RD, '            elif n_started < n_plan:', '            elif False:')]),
    ("N16", 'reducer re-imported via sys.modules (not by file path)', "kill", [(FE, '        red = _load_registered_reducer()', '        import n2_probe_reduce as red')]),
    ("N17", 'raw path absence not reported separately', "kill", [(FE, '    if missing:\n', '    if False:\n')]),
    ("N18", 'K_PROBE d=44 9', "kill", [(RD, 'K_PROBE = {16: 18, 44: 8}', 'K_PROBE = {16: 18, 44: 9}')]),
    ("N19", 'BIND_CELL (32, 2048)', "kill", [(RD, 'BIND_CELL = (32, 8192)', 'BIND_CELL = (32, 2048)')]),
    ("N20", 'aborted-cell censoring promotion removed', "kill", [(RD, '            if n_c < WINDOW_ITERS:', '            if False:')]),
    ("N21", 'promotion even if aborted before the co-load ended', "kill", [(RD, 'all(e is not None and e["timestamp_monotonic_s"] <= t_ab for _s, e in co)', 'True')]),
    ("N22", 'CLI drops the probe reason_codes', "kill", [(FE, 'reason_codes=list(raw.get("reason_codes", [])))', 'reason_codes=[])')]),
    # ---- rev13 (G-6, G-8, G-9) ----
    ("N23", 'G-6: probe fix_s feeds FIX_S', "kill", [(FE, '    COEF.clear(); COEF.update(new)\n    return dict(l_lag_observed_max=int(math.ceil(max(lags))))', '    COEF.clear(); COEF.update(new)\n    _fx = [float(x) for x in probe.get("fix_s", [22.48])]\n    FIX_S["nom"], FIX_S["worst"] = statistics.median(_fx), max(_fx)\n    return dict(l_lag_observed_max=int(math.ceil(max(lags))))')]),
    ("N24", 'G-6: fix_s required again', "kill", [(FE, 'REQUIRED_PROBE_KEYS = ("theta_spans", "tau_cells", "kappa_cells", "l_lag_iters", "delta_l_s")', 'REQUIRED_PROBE_KEYS = ("theta_spans", "tau_cells", "kappa_cells", "l_lag_iters", "delta_l_s", "fix_s")')]),
    ("N25", 'G-8: boot_health_timeout VI even after a cell started', "kill", [(RD, '            if ab in BOOT_ONLY_VI_REASONS and not any_cell:', '            if ab in BOOT_ONLY_VI_REASONS:')]),
    ("N26", 'G-8: boot_health_timeout VI as a cell abort', "kill", [(RD, '                 "boot_health_timeout": VALUE_DEPENDENT}', '                 "boot_health_timeout": VALUE_INDEPENDENT}')]),
    ("N27", 'G-8: client_exception -> VI', "kill", [(RD, '"client_exception": VALUE_DEPENDENT,', '"client_exception": VALUE_INDEPENDENT,')]),
    # ---- rev15 (R14-9): audit rev14 Q mutants, H14-2 verify, and the generalized window paths (R14-8) ----
    ("Q01", 'KAPPA_FLOOR 0.9 -> 1.0', "kill", [(FE, 'KAPPA_FLOOR = 0.9 ', 'KAPPA_FLOOR = 1.0 ')]),
    ("Q02", 'tau_alone_lower uses DW x1.5 (weaker floor)', "kill", [(FE, '    return TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] / 1000.0\n', '    return TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] * 1.5 / 1000.0\n')]),
    ("Q03", 'C window skips the first co-located iteration', "kill", [(RD, '    cwin = call[:WINDOW_ITERS]  ', '    cwin = call[1:WINDOW_ITERS + 1]  ')]),
    ("Q04", 'ctx_mid(C) computed on all co-located iterations', "kill", [(RD, '        out["ctx_mid"]["C"] = _ctx_mid(ctx, cwin)', '        out["ctx_mid"]["C"] = _ctx_mid(ctx, call)')]),
    ("Q05", 'kappa_parts C median over C_all', "kill", [(RD, 'c_med = statistics.median(s_per_intervals(win["C"]))', 'c_med = statistics.median(s_per_intervals(win["C_all"]))')]),
    ("Q06", 'K_matches_derivation hard-coded True (self-report identity)', "kill", [(FE, 'rec["K_matches_derivation"] = rec["derive_fixed_k"] == K_CO', 'rec["K_matches_derivation"] = True')]),
    ("Q07", '--registered rc always 0', "kill", [(FE, 'return 0 if rec["K_matches_derivation"] and rec["limits"]["ok"] else 2', 'return 0')]),
    ("Q08", 'registered record drops budget', "kill", [(FE, 'kappa_star=kstar, limits=limits_check(), derived=derived_constants(), budget=budget_record())', 'kappa_star=kstar, limits=limits_check(), derived=derived_constants(), budget={})')]),
    ("Q09", 'derive_fixed_k returns K_CO (identity)', "kill", [(FE, '        return {d: choose_k(d) for d in EA_D}\n    finally:', '        return dict(K_CO)\n    finally:')]),
    ("Q10", 'kappa* requirement uses DN corner only', "kill", [(FE, 'if min(rho_upper(d, b, c, pc, dc) for pc, dc in CORNERS) < RHO_MAX_REQUIRED', 'if min(rho_upper(d, b, c, pc, "DN") for pc, dc in CORNERS) < RHO_MAX_REQUIRED')]),
    ("Q11", 'RHO_MAX_REQUIRED 3.0 -> 2.9', "kill", [(FE, 'RHO_MAX_REQUIRED = 3.0 ', 'RHO_MAX_REQUIRED = 2.9 ')]),
    ("M13", 'verify_registered skips the byte comparison (H14-2)', "kill", [(FE, '    if data != registered_bytes(registered_record()):', '    if False:')]),
    ("M14", 'verify_registered ignores the sealed digest (H14-2)', "kill", [(FE, '    if digest is not None and hashlib.sha256(data).hexdigest() != digest:', '    if False:')]),
    ("J14", 'alone pi hard-coded to PAIR[d] (OFF alone windows at (0,108) misjudged)', "kill", [(RD, '        pe = _pe(w, pi_alone)', '        pe = _pe(w, PAIR[d])')]),
    ("J15", 'PROFILE cell (no co-load) requires a coload_sent mark', "kill", [(RD, '    if coload and t_co is None:', '    if t_co is None:')]),
    ("J16", 'recorder continuity judged on the FILTERED C instead of pre-filter W_co (R14-2)', "kill", [(RD, '    elif wco and _iter_seq_gap(iters, wco[0]["iter_seq"], wco[-1]["iter_seq"]):', '    elif call and any(call[k + 1]["iter_seq"] != call[k]["iter_seq"] + 1 for k in range(len(call) - 1)):')]),
    ("J17", 'B_FINISHED check dropped', "kill", [(RD, '    elif any(e["bs"] < b for e in wco if e["t_launch"] < t_co1):', '    elif False:')]),
    ("J18", 'COLOC_UNREALIZED (PE_coloc 0.95) check dropped', "kill", [(RD, '    elif len(call) / len(bs_eq) < PE_COLOC_MIN:', '    elif False:')]),
    ("J19", 'CO_WINDOW_SHORT loses the filter-loss breakdown (H14-4)', "kill", [(RD, 'loss_not_in_flight=len(bs_eq) - len(inflt), loss_pair_mismatch=len(inflt) - len(call))', 'loss_not_in_flight=0, loss_pair_mismatch=0)')]),
    # ---- rev14 (A') judgment-path constants and the C value-window truncation ----
    ("J13", 'smoke mapping assigns L_LAG from the probe (rev14: allowance must stay fixed)', "kill", [(FE, '    COEF.clear(); COEF.update(new)\n    return dict(l_lag_observed_max=int(math.ceil(max(lags))))', '    global L_LAG\n    COEF.clear(); COEF.update(new)\n    L_LAG = int(math.ceil(max(lags)))\n    return dict(l_lag_observed_max=int(math.ceil(max(lags))))')]),
    ("J01", 'theta_rho fixed input 5,693 instead of 5,705', "kill", [(FE, 'THETA_RHO_FIXED = 5705.0', 'THETA_RHO_FIXED = 5693.0')]),
    ("J02", 'kappa_floor 1.3 (lower-side guarantee weakened)', "kill", [(FE, 'KAPPA_FLOOR = 0.9 ', 'KAPPA_FLOOR = 1.3 ')]),
    ("J03", 'N_co,max lower bound uses d_factor (not D44 coefficients)', "kill", [(FE, '    return TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] / 1000.0\n', '    return TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] * 1.6 / 1000.0\n')]),
    ("J04", 'N_co,max uses PN theta instead of PW', "kill", [(FE, 'co_prefill_time(Dsp, "PW", "budget") / (tau_alone_lower(b, c) * KAPPA_FLOOR)', 'co_prefill_time(Dsp, "PN", "budget") / (tau_alone_lower(b, c) * KAPPA_FLOOR)')]),
    ("J05", 'L_lag allowance 32', "kill", [(FE, 'L_LAG = 128 ', 'L_LAG = 32 ')]),
    ("J06", 'delta allowance 8', "kill", [(FE, 'DELTA_FIXED = 64 ', 'DELTA_FIXED = 8 ')]),
    ("J07", 'C value window = all co-located iterations (no truncation)', "kill", [(RD, '    cwin = call[:WINDOW_ITERS]  ', '    cwin = call  ')]),
    ("J08", 'C value window = last 128 co-located iterations', "kill", [(RD, '    cwin = call[:WINDOW_ITERS]  ', '    cwin = call[-WINDOW_ITERS:]  ')]),
    ("J09", 'registered_constants K from choose_k (live coefficients)', "kill", [(FE, '    return dict(K=dict(K_CO), K_derivation=', '    return dict(K={d: choose_k(d) for d in EA_D}, K_derivation=')]),
    ("J10", 'smoke gate writes the diagnostic K into K_CO', "kill", [(FE, '    diag = {d: choose_k(d) for d in EA_D}\n    reasons = []', '    diag = {d: choose_k(d) for d in EA_D}\n    K_CO.update({d: k for d, k in diag.items() if k})\n    reasons = []')]),
    ("J11", 'rho-direction theta = design 4,500 (K derivation input drift)', "kill", [(FE, 'theta=(THETA_RHO_FIXED if d == "rho" else THETA_DESIGN)', 'theta=THETA_DESIGN')]),
    ("J12", 'derive_fixed_k uses the live rho set (not the fixed inputs)', "kill", [(FE, '    COEF["rho"].update(theta=THETA_RHO_FIXED, d_factor=dict(D_FACTOR_DESIGN), kappa_dn=KAPPA_DN_DESIGN)\n', '')]),
    ("X06", 'C filter drops the pair condition', "kill", [(RD, '    call = [e for e in inflt if _pair(e) == pi_coloc]', '    call = list(inflt)')]),
    ("X10", "alone PE threshold 0.5", "kill", [(RD, "ALONE_PE_MIN = 0.99", "ALONE_PE_MIN = 0.5")]),
    ("X11", "centered window accepts >= 128", "equivalent", [(RD, "    if len(w) != WINDOW_ITERS:", "    if len(w) < WINDOW_ITERS:")]),
    ("X16", "ctx: earliest valid snapshot instead of latest", "equivalent", [(RD, "        while k + 1 < len(snaps) and snaps[k + 1][0] <= e[\"iter_seq\"]:\n            k += 1", "        while k < 0 and k + 1 < len(snaps) and snaps[k + 1][0] <= e[\"iter_seq\"]:\n            k += 1")]),
    ("M05", "reducer-internal no-drop check removed", "equivalent", [(RD, "    if len(measurements) != n_cells:", "    if False:")]),
    ("B09", "extend_tokens >= 16384", "equivalent", [(RD, 'starts[k].get("extend_tokens") == SPAN_TOKENS', 'starts[k].get("extend_tokens", 0) >= SPAN_TOKENS')]),
    ("B11", "N3 positive control ignores bench window", "equivalent", [(RD, '[(float(marks["bench_start"]), float(marks["bench_end"]))]', "[]")]),
]


# rev14: which mutants bind the JUDGMENT PATH (registered constants, windows, budgets, limits, S-per / A_on / theta
# positive control shared with n2_analyze). Everything else binds the optional smoke tool chain (probe reducer
# classes, provenance, gate, harness) -- retained and run, but §15 counts them separately.
JUDGMENT_IDS = frozenset({"R01", "R02", "R03", "R04", "R06", "R08", "R09", "R10", "R11", "R14", "R16", "R18", "A13",
                          "A01", "A02", "A04", "A05", "A09", "A16", "M01",
                          "B06", "B16", "B18", "B19", "C14", "M03", "M04", "X05", "X06", "X07", "X08", "X10", "X11", "X14",
                          "X16", "B09", "B11", "M05", "N13", "Y04", "Y10", "Y11", "Y12",
                          "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09", "Q10", "Q11", "M13", "M14"}
                         | {f"J{k:02d}" for k in range(1, 20)})
# rev15 (R14-9): theta_rho PROVENANCE group -- theta sample composition of the shared span-pairing implementation
# (reproduces the registered theta_rho source); theta_rho itself is a hard-coded constant, so these bind the
# reproduction path, not the judgment path. A12 is a smoke-gate check (S group).
THETA_IDS = frozenset({"P01", "P02", "P03", "P04", "P05", "P06", "P07", "B01", "B02", "B03", "B04", "X15"})


def n3_sh_path():
    """R14-9: location-independent -- env override, else the git top level of HERE, else HERE-relative; must exist."""
    import subprocess
    cands = [os.environ.get("N2_N3_SH")]
    try:
        top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
        if top:
            cands.append(os.path.join(top, "workspace", "engine-port", "results", "n3_capacity", "n3_capacity.sh"))
    except OSError:
        pass
    cands.append(os.path.join(HERE, "..", "n3_capacity", "n3_capacity.sh"))
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise SystemExit("N2_N3_SH: n3_capacity.sh not found (set N2_N3_SH)")


def _harness_mutants():
    """H01-H30 live in n2_harness_mutants.py (implementer's list); H31-H32 are the rev13 additions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_n2_harness_mutants", os.path.join(HERE, "n2_harness_mutants.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.HARNESS_MUTANTS) + [
        ("H31", "G-15: stop rule reads GATE_PREVIEW (official_only default False)", "kill",
         [(RN, "def gate_result(attempt_dir, official_only=True):", "def gate_result(attempt_dir, official_only=False):")]),
        ("H32", "G-8: runner ignores the no-cell boot_health_timeout VI rule", "kill",
         [(RN, "    if reason in red.BOOT_ONLY_VI_REASONS and n_cells == 0:", "    if False:")]),
    ]


def run(n3_dir=None, only=None):
    names = (FE, RD) + HARNESS_SUPPORT
    src = {fn: open(os.path.join(HERE, fn)).read() for fn in names}
    tests = {fn: open(os.path.join(HERE, fn)).read() for fn in TESTS + HARNESS_TESTS}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "N2_N3_SH": n3_sh_path()}
    if n3_dir:
        env["N2_N3_DIR"] = os.path.abspath(n3_dir)
    rows = []
    allm = [("CTRL", "no mutation", "pass", [])] + MUTANTS + _harness_mutants()
    for mid, desc, expect, reps in [m for m in allm if not only or m[0] in only or m[0] == "CTRL"]:
        files = dict(src)
        if mid == "CTRL":
            suites = TESTS + HARNESS_TESTS
        elif all(fn in (FE, RD) for fn, _o, _n in reps):
            suites = TESTS
        else:
            suites = HARNESS_TESTS
        for fn, old, new in reps:
            if files[fn].count(old) != 1:
                rows.append((mid, desc, "BROKEN-MUTANT", f"anchor count {files[fn].count(old)}"))
                break
            files[fn] = files[fn].replace(old, new)
        else:
            d = tempfile.mkdtemp(prefix=f"n2mut_{mid}_")
            try:
                for fn, c in {**files, **tests}.items():
                    with open(os.path.join(d, fn), "w") as fh:
                        fh.write(c)
                try:
                    r = subprocess.run([sys.executable, "-m", "unittest", *[t[:-3] for t in suites]], cwd=d, env=env,
                                       capture_output=True, text=True, timeout=900)
                except subprocess.TimeoutExpired:                 # R9-8: a timeout is a row, never a kill
                    rows.append((mid, desc, "TIMEOUT", "900 s"))
                    print(f"{mid:5s} {desc:45s} {'TIMEOUT':20s} 900 s", flush=True)
                    continue
                pyc = any(x.endswith(".pyc") for _, _, fs in os.walk(d) for x in fs)
                tail = [ln for ln in r.stderr.strip().split("\n") if ln.startswith(("OK", "FAILED"))]
                tail = tail[-1] if tail else r.stderr[-200:]
                failed = r.returncode != 0
                if expect == "pass":
                    status = "CONTROL-OK" if not failed else "CONTROL-FAILED"
                elif expect == "kill":
                    status = "KILLED" if failed else "SURVIVED"
                else:
                    status = "EQUIVALENT-SURVIVED" if not failed else "EQUIVALENT-KILLED"
                rows.append((mid, desc, status, f"{tail}{' pyc!' if pyc else ''}"))
            finally:
                shutil.rmtree(d, ignore_errors=True)
        grp = "J" if rows[-1][0] in JUDGMENT_IDS else ("T" if rows[-1][0] in THETA_IDS else ("-" if rows[-1][0] == "CTRL" else "S"))
        print(f"{rows[-1][0]:5s} {grp} {rows[-1][1]:45s} {rows[-1][2]:20s} {rows[-1][3]}", flush=True)
        if rows[-1][2] == "CONTROL-FAILED":                # R14-9 (lesson 254): a failed control voids every later row
            raise SystemExit("CONTROL-FAILED: the unmutated suite does not pass in this environment -- aborting")
    return rows


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None, set(sys.argv[2:]) or None)
