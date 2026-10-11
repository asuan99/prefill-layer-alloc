#!/usr/bin/env python3
"""N2(3-a) Job-PROFILE analyzer (PREREG_N2_3A rev16 §6-3, §9-1…§9-6, §11-2…§11-4, §12, §13, §16-1; §15
`n2_analyze.py`). CPU only. HARNESS-SEAL file.

Labels are produced MECHANICALLY by the registered rules (no label reads a descriptive value):
  replicate status (§9-1, via n2_window: measurement_ok + the shared cell_windows W status; OBS = client window),
  point label (§9-2; D = 108 pools OFF16 + OFF44 UNCONDITIONALLY, n <= 8), D label + STICKY_WIRED_D{d} (§9-3),
  profile label (§9-4), NEG_CONTROL_* (§11-3: OFF-boot windows past measurement_ok with (R) and |W| = 128, regardless
  of VALID), STICKY_CORRECTNESS_* (§11-2), OBSERVER_* (§11-4), D-sensitivity (§9-6: same-round D pairs, 4/4 rule),
  forecasts F-N2-4/5/6/7P/D1/D2/D3 (§13: HELD / FALSIFIED / UNRESOLVED).
Descriptives (§12, never label inputs): point S-per p50/p95/p99, residual_p95, S-dev, ctx, stalls, 108 FIXED-wise
values, PE-iter/PE-time ranges, A2-pair-W + W-idx per window (n2_realized_mix scope 'window'), split_transition
counts, C-itl/S-per ratio, ITL-threshold d table (§12-5), cpu.stat / clock exposure, missing boots with cost rank.
Paired CI = benchmarks/pdmux_eval/analyze.paired_bootstrap_ci(baseline, proposed, samples=2000, seed=1) (loaded by
path, real signature).
BLIND (§5-4 (ii)(나)): this analyzer computes NO co-located statistic from Job-PROFILE raw data -- only the alone
window W of each cell is read (the output carries `blind.coloc_statistics_computed = false`).
Job-EA parts (rho_off assembly H15-3, COLOC_* / DS3_* over real E-alpha replicates) are NOT implemented: coloc_label
and ds3_summary exist as pure functions; `rho_off_assembly` raises NotImplementedError.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n2_plan                   # noqa: E402
import n2_realized_mix as mix    # noqa: E402
import n2_window as nw           # noqa: E402

ENG = n2_plan.ENG
LN3 = math.log(1.03)
B_GRID, C_GRID = n2_plan.B_GRID, n2_plan.C_GRID
D_ALL = (16, 24, 34, 44, 108)
DSENS_PAIRS = ((16, 24), (24, 34), (34, 44), (16, 44), (44, 108))
ADJACENT = ((16, 24), (24, 34), (34, 44))
T_LADDER = (40.0, 50.0, 60.0, 65.0)
BOOT_SAMPLES, BOOT_SEED = 2000, 1
MIN_N = 4
PE_MIN = 0.99
DROP = ("_window", "_records", "stats.intervals")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ANA = _load(os.path.join(ENG, n2_plan.ANALYZE_REL), "_n2_analyze_py")
PROF = _load(os.path.join(ENG, n2_plan.PROFILE_PY_REL), "_n2_profile_py")


def paired_ci(baseline, proposed):
    try:
        return ANA.paired_bootstrap_ci(baseline, proposed, samples=BOOT_SAMPLES, seed=BOOT_SEED)
    except ValueError:
        return None


# ===================================================================================================== collect
def _ledger(out):
    p = os.path.join(out, "JOB_LEDGER.json")
    if not os.path.exists(p):
        return {}, None
    with open(p) as fh:
        led = json.load(fh)
    return {r["unit"]: r for r in led.get("ledger", []) if "unit" in r}, led


def collect(out, units):
    """Every registered replicate of the Job, with its §9-1 status. A boot the ledger marks NOT_ATTEMPTED_BUDGET
    gives NOT_ATTEMPTED_BUDGET replicates; a boot never started for another reason (Job abort) gives
    BENCH_INVALID 'not_attempted_*' replicates (HARNESS READING, reported)."""
    rows, ledger_full = _ledger(out)
    reps, boots = [], {}
    for u in units:
        st = (rows.get(u["name"]) or {}).get("status")
        started = st == "ADMITTED" and os.path.exists(os.path.join(out, f"marks_{u['name']}.txt"))
        boot = nw.load_boot(out, u["name"]) if started else None
        if boot is not None:
            boots[u["name"]] = boot
        for cell in u["cells"]:
            if boot is None:
                base = dict(boot=u["name"], unit=u["name"], cell_id=cell["cell_id"], kind=cell["kind"], D=u["D"],
                            alone_D=u["alone_D"], b=cell["b"], c=cell["c"], round=u["round"], boot_kind=u["kind"])
                if st == "NOT_ATTEMPTED_BUDGET":
                    reps.append(dict(base, status="NOT_ATTEMPTED_BUDGET", reason="stop rule (§14-3)"))
                else:
                    reps.append(dict(base, status="BENCH_INVALID", reason=f"not_attempted ({st or 'no ledger row'})"))
                continue
            fn = nw.obs_replicate if cell["kind"] == "OBS" else nw.profile_replicate
            rep = fn(boot, u, cell)
            rep["_cell"] = cell
            reps.append(rep)
    return reps, boots, ledger_full


def _point_key(r):
    return (r["alone_D"], r["b"], r["c"])


# ===================================================================================================== §9-2
def points(reps):
    out = {}
    for D in D_ALL:
        for b in B_GRID:
            for c in C_GRID:
                rs = [r for r in reps if r["kind"] == "PROFILE" and _point_key(r) == (D, b, c)]
                valid = [r for r in rs if r["status"] == "VALID" and "stats" in r]
                hist = {}
                for r in rs:
                    hist[r["status"]] = hist.get(r["status"], 0) + 1
                pt = dict(D=D, b=b, c=c, n=len(valid), attempted=len(rs), status_hist=hist,
                          label="POINT_REALIZED" if len(valid) >= MIN_N else "UNRESOLVED_N_BELOW_4")
                if valid:
                    p50 = [r["stats"]["s_per_p50"] for r in valid]
                    p95 = [r["stats"]["s_per_p95"] for r in valid]
                    p99 = [r["stats"]["s_per_p99"] for r in valid]
                    pt.update(itl_p50_ms=statistics.median(p50), itl_p95_ms=statistics.median(p95),
                              itl_p99_ms=statistics.median(p99),
                              residual_p95_ms=max(p95) - statistics.median(p95),
                              measured_steps=sum(r["window_len"] for r in valid),
                              ctx_mid_mean=statistics.fmean(r["ctx_mid"] for r in valid),
                              ctx_spread_p95_max=max((r["stats"]["ctx_spread_p95_max"] or 0) for r in valid),
                              pe_iter_range=[min(r["pe_iter"] for r in valid), max(r["pe_iter"] for r in valid)],
                              pe_time_range=[min(r["pe_time"] for r in valid), max(r["pe_time"] for r in valid)],
                              s_dev_median_ms=statistics.median(r["stats"]["s_dev_median_ms"] for r in valid),
                              stalls_gt5=sum(r["stats"]["stalls_gt5"] for r in valid),
                              stalls_gt10=sum(r["stats"]["stalls_gt10"] for r in valid),
                              replicate_p95=p95, replicate_boots=[r["boot"] for r in valid])
                    if D == 108:
                        for fx in (16, 44):
                            sub = [r["stats"]["s_per_p95"] for r in valid if r["D"] == fx]
                            pt[f"fixed{fx}_p95_median_ms"] = statistics.median(sub) if sub else None
                            pt[f"fixed{fx}_n"] = len(sub)
                out[(D, b, c)] = pt
    return out


# ===================================================================================================== §9-3 / §9-4
def d_labels(pts, reps):
    out = {}
    for D in D_ALL:
        A = [r for r in reps if r["kind"] == "PROFILE" and r["alone_D"] == D and r["status"] != "NOT_ATTEMPTED_BUDGET"]
        n_unreal = sum(1 for r in A if r["status"] == "D_UNREALIZED")
        if all(pts[(D, b, c)]["label"] == "POINT_REALIZED" for b in B_GRID for c in C_GRID):
            lab = "D_PROFILE_COMPLETE"
        elif A and n_unreal / len(A) >= 0.5:
            lab = "D_REALIZATION_FAILED"
        else:
            lab = "D_PROFILE_PARTIAL"
        out[D] = dict(label=lab, attempted=len(A), d_unrealized=n_unreal,
                      realized_points=sorted([b, c] for b in B_GRID for c in C_GRID
                                             if pts[(D, b, c)]["label"] == "POINT_REALIZED"))
    return out


def sticky_wired(boots, reps, units):
    """STICKY_WIRED_D{d} := every ON boot of D passed §11-1 (a §11-1 failure aborts the Job, exit 6) ∧ D has >= 1
    VALID window. ON boots that never reached /health were not checked (listed)."""
    out = {}
    for D in n2_plan.D_ON:
        ons = [u["name"] for u in units if u["kind"] == "ON" and u["D"] == D]
        checked = [n for n in ons if n in boots and "wiring" in boots[n]["boot"]]
        passed = [n for n in checked if not (boots[n]["boot"]["wiring"]["log_problems"]
                                             or boots[n]["boot"]["wiring"]["green_problems"])]
        n_valid = sum(1 for r in reps if r["kind"] == "PROFILE" and r["boot_kind"] == "ON" and r["D"] == D
                      and r["status"] == "VALID")
        ok = bool(checked) and len(passed) == len(checked) and n_valid >= 1
        out[D] = dict(label=f"STICKY_WIRED_D{D}" if ok else f"STICKY_WIRING_NOT_ESTABLISHED_D{D}",
                      boots_checked=checked, boots_not_checked=[n for n in ons if n not in checked], valid_windows=n_valid)
    return out


def neg_control(reps):
    """§11-3 (Job-PROFILE): population = OFF-boot PROFILE windows past measurement_ok whose (R) passed and |W| = 128
    (status VALID or D_UNREALIZED), VALID or not. Empty -> UNRESOLVED; all PE-iter(W, (0,108)) >= 0.99 -> HELD."""
    pop = [r for r in reps if r["kind"] == "PROFILE" and r["boot_kind"] == "OFF"
           and r["status"] in ("VALID", "D_UNREALIZED") and r.get("window_len") == nw.WIN]
    if not pop:
        return dict(label="NEG_CONTROL_UNRESOLVED", n=0)
    low = [r["cell_id"] for r in pop if r["pe_iter_idx5"] < PE_MIN]
    return dict(label="NEG_CONTROL_FAILED" if low else "NEG_CONTROL_HELD", n=len(pop), below=low,
                min_pe_idx5=min(r["pe_iter_idx5"] for r in pop))


def profile_label(dl, neg, pts):
    n_real = sum(1 for p in pts.values() if p["label"] == "POINT_REALIZED")
    if neg["label"] != "NEG_CONTROL_HELD" or n_real == 0:
        return "PROFILE_NOT_BUILT"
    if all(dl[D]["label"] == "D_PROFILE_COMPLETE" for D in D_ALL):
        return "PROFILE_COMPLETE"
    return "PROFILE_PARTIAL"


# ===================================================================================================== §11-2 / §11-4
def _has_ids(o):
    """§18-2-d #4 (rev17 H17-1): output_ids must be a NON-EMPTY list -- None == None / [] == [] never pass."""
    ids = o.get("output_ids")
    return isinstance(ids, list) and len(ids) > 0


def sticky_correctness(boots):
    """§11-2 / §18-2-d #4: BYTE_IDENTICAL only when text AND output_ids are equal for all 6 prompts, and only when
    every output on BOTH sides carries non-empty output_ids; otherwise (ids absent) UNRESOLVED -- a text-only match
    is not the registered comparison."""
    ref = (boots.get("r1_OFF16") or {}).get("greedy")
    out = {}
    for D in n2_plan.D_ON:
        g = (boots.get(f"r1_ON{D}") or {}).get("greedy")
        if not g or not ref:
            out[D] = dict(label=f"STICKY_CORRECTNESS_UNRESOLVED_D{D}", why="greedy record missing (boot not run)")
            continue
        a, b = g["outputs"], ref["outputs"]
        if not all(_has_ids(o) for o in list(a) + list(b)):
            out[D] = dict(label=f"STICKY_CORRECTNESS_UNRESOLVED_D{D}", why="output_ids absent", n=len(a))
            continue
        same = len(a) == len(b) == 6 and all(x.get("text") == y.get("text") and x.get("output_ids") == y.get("output_ids")
                                             for x, y in zip(a, b))
        out[D] = dict(label=(f"STICKY_CORRECTNESS_BYTE_IDENTICAL_D{D}" if same
                             else f"STICKY_CORRECTNESS_DIVERGED_D{D}"), n=len(a))
    return out


def observer(reps, boots):
    """§11-4: per cell (1,2048), (32,2048) and round r: x_r = ln(C-itl ON44 r / C-itl OBS44 r), both by the client
    window, both past measurement_ok. Label (literal): OK iff both cells n >= 4 ∧ |mean| < ln 1.03 ∧ CI ∋ 0;
    otherwise any cell n < 4 -> UNRESOLVED_N_BELOW_4; otherwise FAIL."""
    cells = {}
    for b, c in n2_plan.OBS_CELLS:
        base, prop = {}, {}
        for r in reps:
            if (r["b"], r["c"]) != (b, c) or r["D"] != 44:
                continue
            if r["kind"] == "OBS" and r["status"] == "VALID":
                base[r["round"]] = math.log(r["c_itl_ms"])
            elif r["kind"] == "PROFILE" and r["boot_kind"] == "ON" and r["boot"] in boots:
                m, _ = nw.measurement_ok(boots[r["boot"]], r["_cell"])
                citl, _n = nw.obs_client_window(boots[r["boot"]]["client"].get(r["cell_id"]))
                if m is None and citl is not None:
                    prop[r["round"]] = math.log(citl)
        rounds = sorted(set(base) & set(prop))
        xs = [prop[k] - base[k] for k in rounds]
        ci = paired_ci({k: base[k] for k in rounds}, {k: prop[k] for k in rounds})
        ok = (len(xs) >= MIN_N and abs(statistics.fmean(xs)) < LN3 and ci is not None
              and ci["ci95_low"] <= 0.0 <= ci["ci95_high"])
        cells[f"{b},{c}"] = dict(n=len(xs), x=xs, rounds=rounds, mean=statistics.fmean(xs) if xs else None, ci=ci,
                                 ok=ok)
    if all(v["ok"] for v in cells.values()):
        lab = "OBSERVER_OK"
    elif any(v["n"] < MIN_N for v in cells.values()):
        lab = "OBSERVER_UNRESOLVED_N_BELOW_4"
    else:
        lab = "OBSERVER_FAIL"
    return dict(verdict=lab, cells=cells)


# ===================================================================================================== §9-6 D-sensitivity
def dsens_label(xs):
    if len(xs) < MIN_N:
        return "UNRESOLVED_N_BELOW_4"
    if all(x > LN3 for x in xs):
        return "D_SENS_GE3"
    if all(x < -LN3 for x in xs):
        return "D_SENS_INVERTED_GE3"
    if all(abs(x) < LN3 for x in xs):
        return "D_SENS_WITHIN_3"
    return "D_SENS_UNRESOLVED"


def dsens_x(si, sj):
    """Same-round pairs only: si, sj = {round: S-per p50}. Unpaired (different round sets) input is REFUSED --
    callers must intersect explicitly (§9-6 'round 간·비짝 비교는 판정량이 아니다')."""
    if set(si) != set(sj):
        raise ValueError(f"unpaired rounds {sorted(si)} vs {sorted(sj)}")
    return {r: math.log(si[r] / sj[r]) for r in sorted(si)}


def round_p50(reps, D, b, c):
    """{round: S-per p50} of the VALID windows of D at (b, c). D = 108: the OFF16 + OFF44 valid windows of that round
    are POOLED (intervals concatenated, linear p50) -- one valid window alone suffices (§9-6)."""
    out = {}
    for r in (1, 2, 3, 4):
        vs = [x for x in reps if x["kind"] == "PROFILE" and x["alone_D"] == D and (x["b"], x["c"]) == (b, c)
              and x["round"] == r and x["status"] == "VALID" and "stats" in x]
        if not vs:
            continue
        if D == 108:
            pooled = [v for x in vs for v in x["stats"]["intervals"]]
            out[r] = nw.quantile(pooled, 0.5)
        else:
            out[r] = vs[0]["stats"]["s_per_p50"]
    return out


def _lsq_slope(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def dsens(reps):
    rows, mean_x = [], {}
    for b in B_GRID:
        for c in C_GRID:
            for di, dj in DSENS_PAIRS:
                si, sj = round_p50(reps, di, b, c), round_p50(reps, dj, b, c)
                common = sorted(set(si) & set(sj))
                x = dsens_x({k: si[k] for k in common}, {k: sj[k] for k in common})
                xs = [x[k] for k in common]
                ci = paired_ci({k: math.log(sj[k]) for k in common}, {k: math.log(si[k]) for k in common})
                rows.append(dict(b=b, c=c, pair=[di, dj], rounds=common, x=xs, n=len(xs), label=dsens_label(xs),
                                 ci=ci))
                mean_x[(di, dj, b, c)] = statistics.fmean(xs) if xs else None
    slopes_b, slopes_c = {}, {}
    for di, dj in DSENS_PAIRS:
        for c in C_GRID:
            pts = [(math.log(b), mean_x[(di, dj, b, c)]) for b in B_GRID if mean_x[(di, dj, b, c)] is not None]
            slopes_b[f"{di},{dj},c{c}"] = _lsq_slope([p[0] for p in pts], [p[1] for p in pts])
        for b in B_GRID:
            pts = [(math.log(c), mean_x[(di, dj, b, c)]) for c in C_GRID if mean_x[(di, dj, b, c)] is not None]
            slopes_c[f"{di},{dj},b{b}"] = _lsq_slope([p[0] for p in pts], [p[1] for p in pts])
    sd = {}
    for D in D_ALL:
        for b in B_GRID:
            for c in C_GRID:
                v = round_p50(reps, D, b, c)
                sd[f"{D},{b},{c}"] = statistics.stdev(math.log(x) for x in v.values()) if len(v) >= 2 else None
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return dict(rows=rows, mean_x={f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in mean_x.items()},
                slopes_b=slopes_b, slopes_c=slopes_c, round_sd_ln_p50=sd, label_counts=counts,
                n_labels=len(rows))


# ===================================================================================================== §13 forecasts
def forecasts(pts, reps, ds, ledger_full, n_units):
    f = {}
    # F-N2-4: realized ctx relative spread (p95 - p50) / c <= 25 % for every point
    vals = [(p["ctx_spread_p95_max"] / p["c"]) for p in pts.values() if p.get("ctx_spread_p95_max") is not None]
    f["F-N2-4"] = "UNRESOLVED" if not vals else ("FALSIFIED" if max(vals) > 0.25 else "HELD")
    # F-N2-5: point p50 increases with c at fixed (D, b)
    bad, seen = 0, 0
    for D in D_ALL:
        for b in B_GRID:
            v = [pts[(D, b, c)].get("itl_p50_ms") for c in C_GRID if pts[(D, b, c)]["label"] == "POINT_REALIZED"]
            if len(v) >= 2:
                seen += 1
                bad += any(y < x for x, y in zip(v, v[1:]))
    f["F-N2-5"] = "UNRESOLVED" if not seen else ("FALSIFIED" if bad else "HELD")
    # F-N2-6: 108 OFF16 vs OFF44 p95 within 3 % on all 12 cells
    diffs, miss = [], 0
    for b in B_GRID:
        for c in C_GRID:
            p = pts[(108, b, c)]
            a, z = p.get("fixed16_p95_median_ms"), p.get("fixed44_p95_median_ms")
            if a is None or z is None:
                miss += 1
            else:
                diffs.append(abs(math.log(a / z)))
    f["F-N2-6"] = ("FALSIFIED" if any(d >= LN3 for d in diffs) else "UNRESOLVED" if miss else "HELD")
    # F-N2-7P: 28/28 boots admitted -- UNRESOLVED when the Job stopped for a non-budget reason (§18-2-d #10, H17-2)
    led = ledger_full or {}
    adm = sum(1 for r in led.get("ledger", []) if r.get("status") == "ADMITTED")
    non_budget_stop = (led.get("stopped") not in (None, "budget") or led.get("job_abort") is not None
                       or any("job_abort" in r for r in led.get("ledger", [])))
    f["F-N2-7P"] = ("UNRESOLVED" if ledger_full is None or non_budget_stop
                    else "HELD" if adm == n_units else "FALSIFIED")
    mx = ds["mean_x"]
    # F-N2-D1: D16 vs D44 mean x at b = 48 > b = 1 for every c
    v = [(mx.get(f"16,44,48,{c}"), mx.get(f"16,44,1,{c}")) for c in C_GRID]
    f["F-N2-D1"] = ("UNRESOLVED" if any(a is None or z is None for a, z in v)
                    else "HELD" if all(a > z for a, z in v) else "FALSIFIED")
    # F-N2-D2: for every b, |mean x(8192) - mean x(512)| < ln 1.03 (D16 vs D44)
    v = [(mx.get(f"16,44,{b},8192"), mx.get(f"16,44,{b},512")) for b in B_GRID]
    f["F-N2-D2"] = ("UNRESOLVED" if any(a is None or z is None for a, z in v)
                    else "HELD" if all(abs(a - z) < LN3 for a, z in v) else "FALSIFIED")
    # F-N2-D3: no adjacent-pair label is D_SENS_INVERTED_GE3
    adj = [r for r in ds["rows"] if tuple(r["pair"]) in ADJACENT]
    f["F-N2-D3"] = ("FALSIFIED" if any(r["label"] == "D_SENS_INVERTED_GE3" for r in adj)
                    else "UNRESOLVED" if any(r["label"] == "UNRESOLVED_N_BELOW_4" for r in adj) else "HELD")
    return f


# ===================================================================================================== §12-5 d table
def d_table(pts):
    realized = [p for p in pts.values() if p["label"] == "POINT_REALIZED"]
    if not realized:
        return dict(note="no POINT_REALIZED point", rows=[])
    env = PROF.RuntimeEnvironment(engine_commit="", gpu_name="", gpu_sm_count=108, cuda_driver="",
                                  attention_backend="flashinfer", cuda_graph=True)
    prof = PROF.HybridModelProfileV1(
        model_id="n2-descriptive", model_revision="", model_config_hash="n2-descriptive", environment=env,
        attention_layers=1, ssm_layers=1, attention_kind="MIXED", num_attention_heads=1, num_kv_heads=1,
        parameter_count=1, heldout_residual_p95_ms=0.0,
        points=[PROF.DecodeLatencyPoint(decode_sms=p["D"], batch_size=p["b"], context_tokens=p["c"],
                                        itl_p50_ms=p["itl_p50_ms"], itl_p95_ms=p["itl_p95_ms"],
                                        itl_p99_ms=p["itl_p99_ms"], repeats=p["n"], measured_steps=p["measured_steps"],
                                        residual_p95_ms=max(0.0, p["residual_p95_ms"])) for p in realized])
    est = PROF.ConservativeDecodeFloorEstimator(prof, allowed_decode_sms=D_ALL)
    rows = []
    for b in B_GRID:
        for c in C_GRID:
            for T in T_LADDER:
                target = T - max(0.1 * T, prof.heldout_residual_p95_ms)
                ups, best = {}, None
                for D in D_ALL:
                    try:
                        _pred, up = est.predict(D, b, c)
                    except KeyError:
                        continue
                    ups[D] = up
                    if best is None and up <= target:
                        best = D
                rows.append(dict(b=b, c=c, T_ms=T, target_ms=target, min_D=best if best is not None
                                 else "108에서도 불충족", upper_ms=ups,
                                 cliff=any(abs(u - target) <= 1.0 for u in ups.values())))
    return dict(note="§12-5 기술 d 표 — 'decode에 필요한 SM'·'운영 d_min'·'static 권고'로 쓰지 않는다(N2C-15)",
                rows=rows)


# ===================================================================================================== descriptives
def window_descriptives(reps, boots, clk_rows=None):
    out = []
    for r in reps:
        if r["kind"] != "PROFILE" or "stats" not in r:
            continue
        st = r["stats"]
        pi = nw.PAIR[r["alone_D"]]
        rec = dict(cell_id=r["cell_id"], status=r["status"], pe_iter=r.get("pe_iter"), pe_time=r.get("pe_time"),
                   split_transitions=r.get("split_transitions"),
                   a2_pair_w=mix.estimators(r["_records"], pi, "window", st["t_lo"], st["t_hi"]),
                   cpu=nw.cpu_delta(boots[r["boot"]], r["cell_id"]))
        citl = nw.c_itl_in_window(boots[r["boot"]]["client"].get(r["cell_id"]), st["t_lo"], st["t_hi"])
        rec["c_itl_over_s_per"] = (citl / st["s_per_p50"]) if citl and st["s_per_p50"] else None
        if clk_rows is not None:
            wr = nw.window_wall_range(boots[r["boot"]], r["cell_id"], st["t_lo"], st["t_hi"])
            rec["clock"] = clock_exposure(clk_rows, *wr) if wr else None
        out.append(rec)
    return out


def read_clk(path, gmtoff):
    import calendar
    import time as _t
    rows = []
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()[1:]
    except OSError:
        return None
    for ln in lines:
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 5:
            continue
        try:
            ts, frac = f[0].split(".")
            ep = calendar.timegm(_t.strptime(ts, "%Y/%m/%d %H:%M:%S")) - gmtoff + float("0." + frac)
            rows.append((ep, float(f[1].split()[0]), int(f[4], 16)))
        except (ValueError, IndexError):
            continue
    return rows


def clock_exposure(rows, lo, hi):
    s = [r for r in rows if lo <= r[0] <= hi]
    if not s:
        return dict(samples=0)
    return dict(samples=len(s), sw_power_cap_pct=100.0 * sum(1 for r in s if r[2] & 0x4) / len(s),
                sm_mhz_mean=statistics.fmean(r[1] for r in s))


# ===================================================================================================== Job-EA stubs
def coloc_label(rhos):
    """§10-3 (pure; Job-EA)."""
    if len(rhos) < MIN_N:
        return "UNRESOLVED_N_BELOW_4"
    if all(x > LN3 for x in rhos):
        return "COLOC_SLOWER_GE3"
    if all(x < -LN3 for x in rhos):
        return "COLOC_FASTER_GE3"
    if all(abs(x) < LN3 for x in rhos):
        return "COLOC_WITHIN_3"
    return "COLOC_UNRESOLVED"


def ds3_summary(cell_labels):
    """§10-4 (pure; Job-EA): cell_labels = the 8 rho_off cell labels."""
    k = sum(1 for x in cell_labels if x in ("COLOC_SLOWER_GE3", "COLOC_FASTER_GE3"))
    if k >= 1:
        return f"DS3_MISMATCH_OBSERVED_{k}"
    if len(cell_labels) == 8 and all(x == "COLOC_WITHIN_3" for x in cell_labels):
        return "DS3_WITHIN_3_ALL"
    return "DS3_UNRESOLVED"


def rho_off_assembly(*_a, **_k):
    """H15-3 (harness-seal precondition, Job-EA): C_off median / ON replicate A_on(ctx_mid(C_off)) with the
    cross-boot containment guard (AON_CONTAINMENT into the missing histogram) and CTX_MISMATCH. NOT IMPLEMENTED."""
    raise NotImplementedError("rho_off assembly (H15-3) is Job-EA work, not implemented in the PROFILE harness")


# ===================================================================================================== output
def profile_csv(pts, path):
    cols = ["decode_sms", "batch_size", "context_tokens", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms", "repeats",
            "measured_steps", "residual_p95_ms", "ctx_p50_realized_mean"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (D, b, c), p in sorted(pts.items()):
            if p["label"] != "POINT_REALIZED":
                continue
            w.writerow([D, b, c, f"{p['itl_p50_ms']:.6f}", f"{p['itl_p95_ms']:.6f}", f"{p['itl_p99_ms']:.6f}", p["n"],
                        p["measured_steps"], f"{max(0.0, p['residual_p95_ms']):.6f}", f"{p['ctx_mid_mean']:.3f}"])


def render_text(res):
    """§16-1 허용 문안 틀 filled mechanically (label literal + numbers); every quote must carry §16-2 병기."""
    L = [f"# N2 Job-PROFILE labels (mechanical; job {res['job_name']}, GPU {res['gpu_uuid']}, rule seal "
         f"{res['rule_seal_digest'][:16]}…) — 성능 판정 아님", ""]
    for key, p in sorted(res["points"].items(), key=lambda kv: tuple(int(x) for x in kv[0].split(","))):
        D, b, c = p["D"], p["b"], p["c"]
        if p["label"] == "POINT_REALIZED":
            what = ("sticky OFF·prefill 없음 단독(비분할 `(0,108)`)의 decode step 시간이다" if D == 108 else
                    "legacy·cudagraph ON·sticky ON·A-agn config·cap 48 조건의 단독 decode step 시간(엔진측 루프 주기)이며 "
                    "PD-mux 운영점(sticky OFF 동거) 측정이 아니다(N2C-1)")
            L.append(f"점 (D = {D}, b = {b}, c = {c})는 `POINT_REALIZED`(유효 복제 n = {p['n']}"
                     f"{', OFF16·OFF44 boot 합산 n ≤ 8' if D == 108 else ''}, PE-iter {p['pe_iter_range']}, "
                     f"PE-time {p['pe_time_range']}, 실현 ctx_p50 창 평균 {p['ctx_mid_mean']:.1f}, 셀 구간 Δdropped 0)이며 "
                     f"S-per p95 중앙값 {p['itl_p95_ms']:.3f} ms였다. 이 값은 {what}.")
        else:
            L.append(f"점 (D = {D}, b = {b}, c = {c}): `{p['label']}` (유효 n = {p['n']}, 사유 {p['status_hist']}) — "
                     f"측정 부족이지 발견이 아니다(N2C-11).")
    L.append("")
    for D, d in sorted(res["D_labels"].items(), key=lambda kv: int(kv[0])):
        L.append(f"D = {D}: `{d['label']}` (시도 {d['attempted']}, D_UNREALIZED {d['d_unrealized']}; 실현된 점 "
                 f"{d['realized_points']})")
    L.append(f"NEG_CONTROL: `{res['neg_control']['label']}` (n = {res['neg_control']['n']})")
    L.append(f"PROFILE: `{res['profile_label']}`")
    L.append(f"OBSERVER: `{res['observer']['verdict']}`" + (" — 모든 프로파일 인용에 'PHASE_EVENTS ON 조건에서'를 붙인다(§9-4)"
                                                          if res['observer']['verdict'] != "OBSERVER_OK" else ""))
    for D, s in sorted(res["sticky_correctness"].items(), key=lambda kv: int(kv[0])):
        L.append(f"`{s['label']}` (무부하·동시성 1에서만, N2C-16)")
    for D, s in sorted(res["sticky_wired"].items(), key=lambda kv: int(kv[0])):
        L.append(f"`{s['label']}`")
    L.append("D-민감도 라벨 개수(요약은 개수만, §9-6): " + json.dumps(res["dsens"]["label_counts"], ensure_ascii=False))
    L.append("예보: " + json.dumps(res["forecasts"], ensure_ascii=False))
    L.append("필수 병기(§16-2): N2C-1…16, DS-1·DS-3·DS-4, E2VR-6, 구속 설정, 기판, OBSERVER 라벨, 결측 boot 목록, "
             "D-민감도 인용 때 N2C-7·N2C-14·§16-3 26·27, Job 이름과 GPU UUID.")
    return "\n".join(L) + "\n"


def _strip(r):
    r = {k: v for k, v in r.items() if k not in ("_window", "_records", "_cell")}
    if "stats" in r:
        r["stats"] = {k: v for k, v in r["stats"].items() if k != "intervals"}
    return r


def analyze(out, job="PROFILE", reg=None, clk=None, substrate=None):
    reg = reg or n2_plan.load_registered()
    units = n2_plan.units(job, reg)
    reps, boots, ledger_full = collect(out, units)
    pts = points(reps)
    dl = d_labels(pts, reps)
    neg = neg_control(reps)
    ds = dsens(reps)
    gm = next((b["tz_gmtoff_s"] for b in boots.values()), 0)
    clk_rows = read_clk(clk, gm) if clk else None
    sub = {}
    if substrate and os.path.exists(substrate):
        with open(substrate) as fh:
            sub = json.load(fh)
    led_rows = (ledger_full or {}).get("ledger", [])
    missing = [dict(unit=r["unit"], status=r["status"], need_s=r["need_s"]) for r in led_rows
               if r.get("status", "").startswith("NOT_ATTEMPTED")]
    rank = sorted({m["need_s"] for m in missing}, reverse=True)
    for m in missing:
        m["cost_rank"] = rank.index(m["need_s"]) + 1
    res = dict(
        job=job, job_name=os.environ.get("PDMUX_JOB_NAME", os.path.basename(os.path.abspath(out))),
        gpu_uuid=(sub.get("gpu") or {}).get("uuid", "UNKNOWN"), rule_seal_digest=n2_plan.RULE_SEAL_DIGEST,
        registered_sha256=n2_plan.REGISTERED_JSON_SHA256,
        replicates=[_strip(r) for r in reps],
        points={f"{k[0]},{k[1]},{k[2]}": v for k, v in pts.items()},
        D_labels={str(k): v for k, v in dl.items()}, neg_control=neg, profile_label=profile_label(dl, neg, pts),
        sticky_wired={str(k): v for k, v in sticky_wired(boots, reps, units).items()},
        sticky_correctness={str(k): v for k, v in sticky_correctness(boots).items()},
        observer=observer(reps, boots), dsens=ds,
        forecasts=forecasts(pts, reps, ds, ledger_full, len(units)),
        d_table=d_table(pts), windows=window_descriptives(reps, boots, clk_rows),
        missing_boots=missing, ledger=ledger_full,
        replicate_class_counts=dict(
            measurement_failure=sum(1 for r in reps if r["status"] in nw.MEASUREMENT_FAILURE),
            predicate_failure=sum(1 for r in reps if r["status"] in nw.PREDICATE_FAILURE),
            valid=sum(1 for r in reps if r["status"] == "VALID")),
        blind=dict(coloc_statistics_computed=False,
                   note="Job-PROFILE analysis reads only alone windows W (§5-4 (ii)(나)); cell/boot-scope A2-pair "
                        "estimators are blinded (H15-4)"),
        n2c_75_report=dict(
            windows_checked=sum(1 for r in reps if "n2c75_zero_weight" in r),
            zero_weight_windows=sum(1 for r in reps if r.get("n2c75_zero_weight")),
            endpoint_prefill_events=sum(r.get("n2c75_endpoint_prefill_events", 0) for r in reps),
            note="N2C-75: both must be 0 (reported from the Job raw data) before writing that the two unbound "
                 "conventions did not touch a judgment"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--job", default="PROFILE")
    ap.add_argument("--clk", default=None)
    ap.add_argument("--substrate", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    if a.job != "PROFILE":
        print("ABORT_EA_NOT_IMPLEMENTED")
        return 3
    if not os.path.isdir(a.out_dir):
        print(f"ANALYZE_NO_OUT_DIR {a.out_dir}")
        return 2
    res = analyze(a.out_dir, a.job, clk=a.clk, substrate=a.substrate)
    jp = a.json_out or os.path.join(a.out_dir, "N2_PROFILE_RESULT.json")
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    profile_csv({tuple(int(x) for x in k.split(",")): v for k, v in res["points"].items()},
                os.path.join(a.out_dir, "N2_PROFILE_POINTS.csv"))
    with open(os.path.join(a.out_dir, "N2_PROFILE_LABELS.txt"), "w") as fh:
        fh.write(render_text(res))
    print(f"PROFILE={res['profile_label']} NEG={res['neg_control']['label']} OBS={res['observer']['verdict']} "
          f"points_realized={sum(1 for p in res['points'].values() if p['label'] == 'POINT_REALIZED')}/60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
