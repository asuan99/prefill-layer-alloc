"""N2(3-a) registered mutant runner (CPU only) -- PREREG_N2_3A rev11 (R8-5, R9-3, R9-8, R10-4; N2C-47 issue: every registered mutant
must be re-runnable by an auditor).

Run:  PYTHONDONTWRITEBYTECODE=1 python3 n2_mutants.py [N3_RAW_DIR]
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

# (id, description, expect, [(file, old, new), ...]); expect = "kill" or "equivalent"
MUTANTS = [
    ("R01", "K_16 = 16", "kill", [(FE, "K_CO = {16: 18, 44: 8}", "K_CO = {16: 16, 44: 8}")]),
    ("R02", "K_16 = 20", "kill", [(FE, "K_CO = {16: 18, 44: 8}", "K_CO = {16: 20, 44: 8}")]),
    ("R03", "L_LAG = 0", "kill", [(FE, "L_LAG = 32 ", "L_LAG = 0 ")]),
    ("R04", "ealpha_c_in -> c_in(108)", "kill", [(FE, 'use c_in(d)."""\n    return c_in_of(d, b, c)', 'use c_in(d)."""\n    return c_in_of(108, b, c)')]),
    ("R05", "choose_k returns K+1", "kill", [(FE, "        if k_satisfies(d, K):\n            return K", "        if k_satisfies(d, K):\n            return K + 1")]),
    ("R06", "kappa_DW fixed 1.6", "kill", [(FE, 'return k * KAPPA_DW_OVER_DN if dcorner == "DW" else k', 'return 1.6 if dcorner == "DW" else k')]),
    ("R07", "theta rho max->min across p", "kill", [(FE, 'new["rho"]["theta"] = max(med.values())', 'new["rho"]["theta"] = min(med.values())')]),
    ("R08", "rho requirement 1.6", "kill", [(FE, "RHO_MAX_REQUIRED = 2.0 ", "RHO_MAX_REQUIRED = 1.6 ")]),
    ("R09", "delta tau of 16 not window", "kill", [(FE, "/ tau_alone(Dwin, b, c, dc, dirn)", "/ tau_alone(16, b, c, dc, dirn)")]),
    ("R10", "cell_time_ealpha cin = c_in(Dalone)", "kill", [(FE, "    cin = ealpha_c_in(d, b, c)\n    k = max(1", "    cin = c_in_of(Dalone, b, c)\n    k = max(1")]),
    ("R11", "call-site delta uses d", "kill", [(FE, "delta_tokens(Dalone, b, c, dc)", "delta_tokens(d, b, c, dc)")]),
    ("R12", "tau accepts D not in 16,44", "kill", [(FE, 'D, b, c = _check_cell(cell, "tau", (16, 44))', 'D, b, c = _check_cell(cell, "tau", (16, 44, 24, 34, 108))'), (FE, "        ratios[D].append(ms / model)", "        ratios.setdefault(D, []).append(ms / model)")]),
    ("R13", "tau rho uses min", "kill", [(FE, "hi = {D: max(r) for D, r in ratios.items()}", "hi = {D: min(r) for D, r in ratios.items()}")]),
    ("R14", "delete W_co condition", "kill", [(FE, "                if min(w_co(d, b, c, pc, dc) for pc, dc in CORNERS) < C_WINDOW_MIN:\n                    return False\n", "")]),
    ("R15", "kappa rho median", "kill", [(FE, 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)', 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = statistics.median(ks)')]),
    ("R16", "delete L_LAG in E-alpha pre", "kill", [(FE, "    pre = (c + 71 - start) + L_LAG", "    pre = (c + 71 - start)")]),
    ("R17", "s_hat rho direction", "kill", [(FE, 'return max(burst_spread(D, b, c, pc, dc, "env") for pc, dc in CORNERS)', 'return max(burst_spread(D, b, c, pc, dc, "rho") for pc, dc in CORNERS)')]),
    ("R18", "N_co,max rho direction", "kill", [(FE, 'return math.ceil(max(w_co(Dsp, b, c, pc, dc, "env") for pc, dc in CORNERS))', 'return math.ceil(max(w_co(Dsp, b, c, pc, dc, "rho") for pc, dc in CORNERS))')]),
    ("R19", "theta median -> min (D-9)", "kill", [(FE, 'med[p] = statistics.median(_scalar(v, "theta") for v in lst) * 64.0 / p', 'med[p] = min(_scalar(v, "theta") for v in lst) * 64.0 / p')]),
    ("R20", "accept n < 20", "kill", [(FE, "len(lst) < THETA_MIN_SPANS", "len(lst) < 1")]),
    ("R21", "accept iteration-level raw", "kill", [(FE, "    if isinstance(x, (list, tuple, dict)):\n        raise ProbeRejected", "    if isinstance(x, (list, tuple, dict)):\n        return float(max(x))\n        raise ProbeRejected")]),
    ("R22", "restore K_SEARCH_MAX = 64", "kill", [(FE, "    K = 1\n    while True:", "    K = 1\n    while K <= 64:"), (FE, "        K += 1\n", "        K += 1\n    return None\n")]),
    ("R23", "gate skips limits_check", "kill", [(FE, '        if not lim["ok"]:', "        if False:")]),
    ("A01", "budget tau -> lo", "kill", [(FE, 'for dirn, s in (("rho", hi), ("budget", hi), ("env", lo)):', 'for dirn, s in (("rho", hi), ("budget", lo), ("env", lo)):')]),
    ("A02", "budget kappa -> min", "kill", [(FE, 'new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)\n    new["env"]["kappa_dn"] = min(ks)', 'new["rho"]["kappa_dn"] = max(ks)\n    new["budget"]["kappa_dn"] = min(ks)\n    new["env"]["kappa_dn"] = min(ks)')]),
    ("A04", "E-alpha cost co_prefill rho", "kill", [(FE, '    tco = co_prefill_time(d, pc, "budget")', '    tco = co_prefill_time(d, pc, "rho")')]),
    ("A05", "PROFILE cost theta rho", "kill", [(FE, '    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[D], pc, "budget")\n    s = burst_spread(D, b, cin, pc, dc, "budget")\n    start = cin + 1 + s / 2\n    steps = (c + 63', '    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[D], pc, "rho")\n    s = burst_spread(D, b, cin, pc, dc, "budget")\n    start = cin + 1 + s / 2\n    steps = (c + 63')]),
    ("A06", "drop tau binding-cell check", "kill", [(FE, '    for cell in BINDING_EA_CELLS:\n        if cell not in tc:\n            raise ProbeRejected(f"tau cells must include binding cell {cell}", value_independent=True)\n', "")]),
    ("A07", "drop A_on window check", "kill", [(FE, ' \\\n                or int(v.get("a_on_window_iters", 0)) < WINDOW_ITERS:', ":")]),
    ("A08", "required p (92,) only", "kill", [(FE, "THETA_REQUIRED_P = (64, 92)", "THETA_REQUIRED_P = (92,)")]),
    ("A10", "probe sha from calculator file", "kill", [(FE, 'rec["probe_sha256"] = _sha256_file(probe_path)', 'rec["probe_sha256"] = _sha256_file(__file__)')]),
    ("A11", "CLI rc always 0", "kill", [(FE, 'return 0 if rec["seal_allowed"] else 2', "return 0")]),
    ("A13", "exec delta budget direction", "kill", [(FE, '    return delta_tokens(Dwin, b, c, "DN", "env")', '    return delta_tokens(Dwin, b, c, "DN", "budget")')]),
    ("A09", "drop KV limit", "equivalent", [(FE, "                if L > CONTEXT_LEN or b * L > KV_TOKENS:", "                if L > CONTEXT_LEN:")]),
    ("A12", "gate drops rho table recheck", "equivalent", [(FE, '        if not all(r["passes"] for r in rho_table()):', "        if False:")]),
    ("A16", "search starts at K = 2", "equivalent", [(FE, "    K = 1\n    while True:", "    K = 2\n    while True:")]),
    ("G01", "gate K hard-coded", "kill", [(FE, "    ks = {d: choose_k(d) for d in EA_D}", "    ks = {16: 18, 44: 8}")]),
    ("G02", "derived via design coefficients", "kill", [(FE, '    if not reasons:\n        rec["derived"] = derived_constants()', '    if not reasons:\n        _c = copy.deepcopy(COEF); COEF.clear(); COEF.update({d: dict(theta=THETA_DESIGN, d_factor=dict(D_FACTOR_DESIGN), kappa_dn=KAPPA_DN_DESIGN) for d in DIRECTIONS})\n        rec["derived"] = derived_constants(); COEF.clear(); COEF.update(_c)')]),
    ("G03", "CLI accepts duplicate keys", "kill", [(FE, "raw = json.load(fh, object_pairs_hook=_no_duplicate_keys)", "raw = json.load(fh)")]),
    ("C09", "derived delta budget direction", "kill", [(FE, "delta_on_A2=delta_exec(d, b, c), delta_off_A2=delta_exec(108, b, c)", "delta_on_A2=delta_tokens(d, b, c), delta_off_A2=delta_tokens(108, b, c)")]),
    ("M01", "budget record FIX_S hard-coded", "kill", [(FE, '    out = {"FIX_S": dict(FIX_S),', '    out = {"FIX_S": {"nom": 22.48, "worst": 35.5},')]),
    ("M02", "reducer sha check removed (subsumed by R10-3 re-reduction)", "equivalent", [(FE, '        if not rec["reducer_sha256"] or rec["reducer_sha256"] != local_red:', "        if False:")]),
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
    ("C14", "C window contamination (bs filter only)", "kill", [(RD, '    cwin = [e for e in wco if e["bs"] == b and e["prefill_in_flight"] and _pair(e) == pair]', '    cwin = [e for e in wco if e["bs"] == b]')]),
    ("M03", "contaminated snapshots not excluded", "kill", [(RD, '                   if s_.get("active_decode_sequences") == b)', "                   if True)")]),
    ("M04", "A_on pooled instead of interpolated", "kill", [(RD, '    return c_med, a_on_interp(a1, m["A1"], a2, m["A2"], m["C"])', '    return c_med, statistics.median(s_per_intervals(win["W"]) + s_per_intervals(win["A2"]))')]),
    # ---- rev10 (R9-3): audit rev9 X mutants adapted to the rev10 code, and new rev10 rules (M05-M12) ----
    ("X01", "kappa length-weighted over measurements (pooled-C analogue)", "kill", [(RD, '            kap.setdefault(cell, []).append(out["c_med"] / out["a_on"])', '            kap.setdefault(cell, []).extend([out["c_med"] / out["a_on"]] * len(win["C"]))')]),
    ("X02", "kappa cell = mean of kappa_i", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": statistics.mean(ks),')]),
    ("X04", "reduce_probe silently drops failing measurements", "kill", [(RD, "            if issues:\n                worst = max(issues, key=lambda x: SEVERITY[x[0]])\n", "            if issues:\n                continue\n                worst = max(issues, key=lambda x: SEVERITY[x[0]])\n")]),
    ("X05", "C filter drops prefill_in_flight only (pair kept)", "kill", [(RD, 'e["bs"] == b and e["prefill_in_flight"] and _pair(e) == pair]', 'e["bs"] == b and _pair(e) == pair]')]),
    ("X07", "TRIM = 0", "kill", [(RD, "TRIM = 16", "TRIM = 0")]),
    ("X08", "A2 = last 128 instead of first", "kill", [(RD, '_alone(iters, b, t_co1, float("inf"), pair)[:WINDOW_ITERS]', '_alone(iters, b, t_co1, float("inf"), pair)[-WINDOW_ITERS:]')]),
    ("X09", "co span: drop extend_tokens == n_new*8192 check", "kill", [(RD, 'sp[0] >= t_co and sp[2]["extend_tokens"] == sp[2]["n_new"] * CO_INPUT]', 'sp[0] >= t_co]')]),
    ("X12", "A_on containment guard removed", "kill", [(RD, "    if not min(x1, x2) <= x <= max(x1, x2):\n", "    if False:\n")]),
    ("X13", "raw file sha replaced by a constant", "kill", [(RD, "            raw_files[bt[k]] = sha256_file(bt[k])", "            raw_files[bt[k]] = \"0\" * 64")]),
    ("X14", "alone windows drop the not-in-flight filter", "kill", [(RD, 'e["bs"] == b and not e["prefill_in_flight"]]', 'e["bs"] == b]')]),
    ("X15", "theta: load-call exclusion skips the first candidate", "kill", [(RD, "        inside = [s for s in spans if a <= s[0] <= b]", "        inside = [s for s in spans if a <= s[0] <= b][1:]")]),
    ("M06", "probe status priority: value-dependent before censored", "kill", [(RD, "(KAPPA_CENSORED, VALUE_DEPENDENT, VALUE_INDEPENDENT)", "(VALUE_DEPENDENT, KAPPA_CENSORED, VALUE_INDEPENDENT)")]),
    ("M07", "CLI allows re-probe after a value-dependent reject", "kill", [(FE, 'reprobe_allowed=(status == "VALUE_INDEPENDENT"))', 'reprobe_allowed=(status in ("VALUE_INDEPENDENT", "VALUE_DEPENDENT")))')]),
    ("M08", "CLI freeze-manifest check removed", "kill", [(FE, '        bad = check_freeze_manifest(freeze_path, rec["calculator_sha256"], local_red)', "        bad = []")]),
    ("M09", "tau not taken from the kappa A1 window (R9-2)", "kill", [(RD, "            tau_w.setdefault(cell, []).append(win[\"W\"])", "            pass")]),
    ("M10", "C < 128 classified VALUE_DEPENDENT instead of KAPPA_CENSORED", "kill", [(RD, "raise MeasurementRejected(KAPPA_CENSORED, f\"C has", "raise MeasurementRejected(VALUE_DEPENDENT, f\"C has")]),
    ("M11", "n < 20 not value-independent", "kill", [(FE, '(counted after exclusion)",\n                                value_independent=True)', '(counted after exclusion)",\n                                value_independent=False)')]),
    ("M12", "negative L_lag accepted", "kill", [(FE, "    if min(lags) < 0:", "    if False:")]),
    # ---- rev11 (R10-4): audit rev10 Y mutants adapted to the rev11 code ----
    ("Y01", "status priority VI before VD", "kill", [(RD, "    status = next((s for s in (KAPPA_CENSORED, VALUE_DEPENDENT, VALUE_INDEPENDENT)", "    status = next((s for s in (KAPPA_CENSORED, VALUE_INDEPENDENT, VALUE_DEPENDENT)")]),
    ("Y02", "kappa cell = median_low (even n)", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": statistics.median_low(ks),')]),
    ("Y03", "kappa cell = max kappa_i", "kill", [(RD, '{"kappa_median": statistics.median(ks),', '{"kappa_median": max(ks),')]),
    ("Y04", "censoring judged on W_co not C", "kill", [(RD, "    if len(cwin) < WINDOW_ITERS:\n        bs_eq", "    if len(wco) < WINDOW_ITERS:\n        bs_eq")]),
    ("Y05", "missing delta_l/fix_s -> VALUE_DEPENDENT", "kill", [(RD, 'issues.append((VALUE_INDEPENDENT, f"{key} mark missing"))', 'issues.append((VALUE_DEPENDENT, f"{key} mark missing"))')]),
    ("Y06", "PROBE_D check removed", "kill", [(RD, "    if cell[0] not in PROBE_D:\n        issues.append((VALUE_INDEPENDENT", "    if False:\n        issues.append((VALUE_INDEPENDENT")]),
    ("Y07", "malformed mark lines ignored", "kill", [(RD, 'issues.append((VALUE_INDEPENDENT, f"malformed mark: {line}")); continue', "continue")]),
    ("Y08", "freeze manifest: reducer sha not compared", "kill", [(FE, ' or by_name.get("n2_probe_reduce.py") != reducer_sha:', ":")]),
    ("Y09", "missing binding kappa cell not value-independent", "kill", [(FE, 'raise ProbeRejected(f"kappa cells must include binding cell {cell}", value_independent=True)', 'raise ProbeRejected(f"kappa cells must include binding cell {cell}", value_independent=False)')]),
    ("Y10", "C censor threshold 127", "kill", [(RD, "    if len(cwin) < WINDOW_ITERS:", "    if len(cwin) < WINDOW_ITERS - 1:")]),
    ("Y11", "l_lag always 0", "kill", [(RD, 'out["l_lag_iters"] = sum(1 for e in iters if t_co <= e["t_launch"] < t_co0)', 'out["l_lag_iters"] = 0')]),
    ("Y12", "tau from A2 instead of A1", "kill", [(RD, '            tau_w.setdefault(cell, []).append(win["W"])', '            tau_w.setdefault(cell, []).append(win["A2"])')]),
    ("Y13", "calc reject always allows re-probe", "kill", [(FE, "reprobe_allowed=e.value_independent)", "reprobe_allowed=True)")]),
    ("Y14", "VI probe status -> re-probe not allowed", "kill", [(FE, 'reprobe_allowed=(status == "VALUE_INDEPENDENT"))', 'reprobe_allowed=False)')]),
    # ---- rev11 new rules (R10-1/2/3, H-1/H-2, R10-5) ----
    ("N01", "unknown cell_abort reason -> VALUE_INDEPENDENT", "kill", [(RD, "ABORT_REASONS.get(abort, VALUE_DEPENDENT)", "ABORT_REASONS.get(abort, VALUE_INDEPENDENT)")]),
    ("N02", "missing cell_end/coload_sent without abort -> VI", "kill", [(RD, 'issues.append((VALUE_DEPENDENT, f"{key} mark missing without cell_abort"))', 'issues.append((VALUE_INDEPENDENT, f"{key} mark missing without cell_abort"))')]),
    ("N03", "orphan raw co spans not checked", "kill", [(RD, "                if n_in != 1:", "                if False:")]),
    ("N04", "duplicate cell id marks silently skipped", "kill", [(RD, 'issues.append((VALUE_DEPENDENT, f"duplicate {tag} for cell id {rest[0]}")); cm.setdefault("dup", True)', 'pass')]),
    ("N05", "exact call counts not checked", "kill", [(RD, "        if counts[d] != expected.get(d, 0):", "        if False:")]),
    ("N06", "shortfall after boot death -> VD", "kill", [(RD, "VALUE_INDEPENDENT if (short and dead) else VALUE_DEPENDENT", "VALUE_DEPENDENT")]),
    ("N07", "load-call wrapping not checked", "kill", [(RD, "        if n_calls != 1:", "        if False:")]),
    ("N08", "truncated last telemetry line treated as corruption", "kill", [(RD, "            if k == len(lines) - 1:", "            if False:")]),
    ("N09", "CLI re-reduction removed", "kill", [(FE, "        irreproducible = reproduce_probe(probe_path, raw)", "        irreproducible = None")]),
    ("N10", "scalar marks checked first (short-circuit)", "kill", [(RD, "    if cell[0] not in PROBE_D:\n", "    if \"fix_s\" not in cm or \"delta_l\" not in cm:\n        return [(VALUE_INDEPENDENT, \"scalar mark missing\")], None\n    if cell[0] not in PROBE_D:\n")]),
    ("N11", "manifest files not verified", "kill", [(FE, "        if not os.path.exists(path) or _sha256_file(path) != sha:", "        if False:")]),
    ("N12", "harness mark writer not required in manifest", "kill", [(FE, "    if not isinstance(writers, list) or not writers or any(w not in files for w in writers):", "    if False:")]),
    ("N13", "registered call count 13 at d = 44", "kill", [(RD, "PROBE_CALLS = {16: 7, 44: 14}", "PROBE_CALLS = {16: 7, 44: 13}")]),
    ("N14", "censor detail not recorded", "kill", [(RD, '**({"censor_detail": worst[2]} if len(worst) > 2 else {})', '**{}')]),
    ("N15", "VI boot_abort not counted as boot death", "kill", [(RD, "            dead = dead or cls == VALUE_INDEPENDENT", "            pass")]),
    ("X06", "C filter drops the pair condition", "kill", [(RD, 'e["bs"] == b and e["prefill_in_flight"] and _pair(e) == pair]', 'e["bs"] == b and e["prefill_in_flight"]]')]),
    ("X10", "alone PE threshold 0.5", "kill", [(RD, "ALONE_PE_MIN = 0.99", "ALONE_PE_MIN = 0.5")]),
    ("X11", "centered window accepts >= 128", "equivalent", [(RD, "    if len(w) != WINDOW_ITERS:", "    if len(w) < WINDOW_ITERS:")]),
    ("X16", "ctx: earliest valid snapshot instead of latest", "equivalent", [(RD, "        while k + 1 < len(snaps) and snaps[k + 1][0] <= e[\"iter_seq\"]:\n            k += 1", "        while k < 0 and k + 1 < len(snaps) and snaps[k + 1][0] <= e[\"iter_seq\"]:\n            k += 1")]),
    ("M05", "reducer-internal no-drop check removed", "equivalent", [(RD, "    if len(measurements) != n_cells:", "    if False:")]),
    ("B09", "extend_tokens >= 16384", "equivalent", [(RD, 'starts[k].get("extend_tokens") == SPAN_TOKENS', 'starts[k].get("extend_tokens", 0) >= SPAN_TOKENS')]),
    ("B11", "N3 positive control ignores bench window", "equivalent", [(RD, '[(float(marks["bench_start"]), float(marks["bench_end"]))]', "[]")]),
]


def run(n3_dir=None):
    src = {fn: open(os.path.join(HERE, fn)).read() for fn in (FE, RD)}
    tests = {fn: open(os.path.join(HERE, fn)).read() for fn in TESTS}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if n3_dir:
        env["N2_N3_DIR"] = os.path.abspath(n3_dir)
    rows = []
    for mid, desc, expect, reps in [("CTRL", "no mutation", "pass", [])] + MUTANTS:
        files = dict(src)
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
                    r = subprocess.run([sys.executable, "-m", "unittest", *[t[:-3] for t in TESTS]], cwd=d, env=env,
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
        print(f"{rows[-1][0]:5s} {rows[-1][1]:45s} {rows[-1][2]:20s} {rows[-1][3]}", flush=True)
    return rows


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
