#!/usr/bin/env python3
"""N3 analyzer -- k-wise SLO capacity of the target-D44 sticky-OFF B1 arm (PREREG_N3_CAPACITY rev1).

DECISION PATH (registered, P-3; constants in n3_plan.py):
  request predicate  = analyze.RequestResult.passes(T_k_ms, ITL_SLO_MS)
                       i.e. TTFT <= T_k  AND  request-internal token-ITL p95 <= 50 ms   (canonical)
  T_k                = read from T_K.json (k x floor, floor measured in THIS Job before any arm
                       cell; seconds, frozen; never re-derived here -- P-2)
  attainment(r, k)   = pooled pass fraction over the VALID seeds of rate r (equal N per seed)
  capacity(k)        = the largest registered rate r* such that attainment >= X_ATTAIN at r* and
                       at every registered rate below it (first-failure rule)
  labels (per k)     = CAPACITY_BRACKETED_k        r* exists and the next rate fails
                       CAPACITY_BELOW_GRID_k       the lowest registered rate already fails
                       CAPACITY_ABOVE_GRID_k       every registered rate passes
                       UNRESOLVED_N_BELOW_4_k      a rate the rule must read has < 4 valid seeds
                                                   (measurement shortfall, not a finding; gate #21)
  CI                 = seed-paired bootstrap (resample seeds jointly across rates) of capacity(k)
                       and of attainment(r, k); n=4 => 35 distinct multisets (coarse; disclosed)
EVERYTHING ELSE IS DESCRIPTIVE: k in [1,25] curve, 99 % attainment, ITL ladder {40,50,60,65},
binding-predicate split, ITL p95 in [45,55] share, cliff zone Z(k), stationarity S1-S3 (sizes only
for STATIONARY cells), observer-effect gate (PHASE_EVENTS ON vs OFF), config bridge vs λ0 VESSL,
realised partition (n3_realized_mix), clocks / cpu.stat, registered forecasts.

N3 has ONE arm.  There is no arm comparison, no Holm step, no "best static" (clause 10).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import os
import sys
from pathlib import Path

# rev5 (2026-10-06): NOT .resolve() -- inside a VESSL Job job_entry.sh replaces engine-port/results/
# with a symlink to /data/runs/<job>/results, and resolving through it moved ENG off the clone
# (benchmarks/ and src/ "MISSING"; Job pdmux-n3-capacity-a4b0c2e-20261006064114 ABORT_SELFTEST).
# abspath keeps the path the runner invoked ($ROOT/workspace/engine-port/results/...), symlinks intact.
HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENG / "benchmarks" / "pdmux_eval"))
import analyze as A  # noqa: E402  (canonical predicate; digest pinned in n3_plan.registered_files)
import n3_plan as P  # noqa: E402
import n3_realized_mix as M  # noqa: E402

RETRACT_RE = re.compile(r"#retracted_reqs:\s*(\d+)")
PE_DISABLED = "PDMUX_PHASE_EVENTS disabled after failure"
_BOOT_SAMPLES = P.BOOTSTRAP_SAMPLES     # the selftest lowers this for speed only


# ------------------------------------------------------------------------------- loading
def load_bench(path: Path):
    """-> (requests, record) or (None, reason)."""
    p = Path(path)
    if not p.exists():
        return None, "bench artefact absent"
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        return None, f"{len(lines)} bench records (registered: 1)"
    rec = json.loads(lines[0])
    reqs, _dur = A.load_bench_serving_rounds(p)
    return reqs, rec


def cell_validity(u: dict, rec: dict, out_dir: Path) -> list:
    bad = []
    errs = rec.get("errors") or []
    n_err = sum(1 for e in errs if e) if isinstance(errs, list) else int(errs or 0)
    if n_err:
        bad.append(f"errors={n_err}")
    if int(rec.get("completed", -1)) != u["np"]:
        bad.append(f"completed {rec.get('completed')} != {u['np']}")
    if int(rec.get("total_input_tokens", -1)) != u["np"] * u["input_len"]:
        bad.append("input length not exact")
    ol = rec.get("output_lens") or []
    if not ol or any(int(v) != u["output_len"] for v in ol):
        bad.append("output length not exact")
    if float(rec.get("random_range_ratio", -1)) != 1.0:
        bad.append("random_range_ratio != 1.0")
    if u["rate"] is not None and abs(float(rec.get("request_rate", -1)) - u["rate"]) > 1e-9:
        bad.append(f"request_rate {rec.get('request_rate')} != {u['rate']}")
    if (out_dir / f"budget_killed_{u['name']}").exists():
        bad.append("cut by the budget enforcer")
    n_ret = retract_count(out_dir / f"srv_{u['name']}.log")
    if n_ret:
        bad.append(f"retracted requests {n_ret} (X-10: per-request exclusion impossible -> cell excluded)")
    ai = arm_identity_of(out_dir, u["name"])
    # D1 (rev3): no evidence = unverifiable = invalid.  The decode leg is required when the unit
    # ran with PHASE_EVENTS ON and no R7 mid-boot-disable marker; thresholds are 0 only.
    expect_dec = u["phase_events"] == 1 and not phase_events_disabled(out_dir, u["name"])
    unv = []
    if ai is not None:
        if ai.get("busy_snapshots_seen", 0) == 0:
            unv.append("no decode-busy snapshot")
        if expect_dec and ai.get("decode_iterations_seen", 0) == 0:
            unv.append("PHASE_EVENTS ON but no decode_iteration")
    if ai is None:
        bad.append("arm identity unverifiable (no telemetry) -- R3")
    elif unv:
        bad.append(f"arm identity unverifiable (D1: {'; '.join(unv)}) -- R3")
    elif ai["decode_iteration_on_nontarget"] or ai["busy_snapshot_on_nontarget"]:
        bad.append(f"arm identity violated (R3): {ai['decode_iteration_on_nontarget']} decode iterations, "
                   f"{ai['busy_snapshot_on_nontarget']} decode-busy snapshots on {ai['forbidden_pairs']}")
    return bad


def arm_identity_of(out_dir: Path, name: str):
    """R3 -- from mix_<name>.json if the runner wrote it, else recomputed from the telemetry."""
    m = mix_of(out_dir, name)
    if m and "arm_identity" in m:
        return m["arm_identity"]
    tel = out_dir / f"tel_{name}.jsonl"
    if not tel.exists():
        return None
    return M.arm_identity(M.read_events(tel), P.NONTARGET_DIVISIONS)


def phase_events_disabled(out_dir: Path, name: str) -> bool:
    """R7 -- the runner's marker, or the server log line itself."""
    if (out_dir / f"phase_events_disabled_{name}").exists():
        return True
    srv = out_dir / f"srv_{name}.log"
    return srv.exists() and PE_DISABLED in srv.read_text(errors="replace")


def retract_count(srv_log: Path) -> int:
    if not Path(srv_log).exists():
        return 0
    return sum(int(m.group(1)) for m in RETRACT_RE.finditer(Path(srv_log).read_text(errors="replace")))


def itl_p95(req) -> float:
    return A.percentile(req.token_itl_ms, 0.95)


# ------------------------------------------------------------------------------- stationarity
def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def s1_fe_ratio(reqs) -> float:
    n = len(reqs)
    if n < 3:
        return float("nan")
    first = [r.ttft_ms for r in reqs[: n // 3]]
    last = [r.ttft_ms for r in reqs[n - n // 3:]]
    return A.percentile(last, 0.5) / A.percentile(first, 0.5)


def s2_gap(reqs, t_ms) -> float:
    n = len(reqs)
    h = n // 2
    if h == 0:
        return float("nan")
    v1 = 1.0 - sum(r.passes(t_ms, P.ITL_SLO_MS) for r in reqs[:h]) / h
    v2 = 1.0 - sum(r.passes(t_ms, P.ITL_SLO_MS) for r in reqs[h:]) / (n - h)
    return v2 - v1


def s3_rho(tel: Path, marks: Path) -> float:
    try:
        import n3_realized_mix as M
    except ImportError:
        return float("nan")
    ev = M.read_events(tel)
    mk = M.read_marks(marks)
    snaps = [e for e in ev if e.get("event") == "runtime_snapshot" and e.get("phase") != "startup"]
    if "bench_start" in mk and "bench_end" in mk:
        snaps = [e for e in snaps if mk["bench_start"] <= e["timestamp_monotonic_s"] <= mk["bench_end"]]
    if len(snaps) < P.S3_MIN_SNAPSHOTS:
        return float("nan")
    q = [float(e.get("prefill_queue_depth") or 0) for e in snaps]
    return spearman(list(range(len(q))), q)


def _nanmean(xs):
    xs = [x for x in xs if x == x]
    return statistics.fmean(xs) if xs else float("nan")


# ------------------------------------------------------------------------------- capacity
def capacity_rule(rates, att, nvalid, x=P.X_ATTAIN, nmin=P.N_SEEDS_MIN):
    """First-failure rule.  `att[r]` pooled attainment, `nvalid[r]` valid seeds."""
    r_cap, r_fail = None, None
    for r in rates:
        if nvalid.get(r, 0) < nmin:
            return {"label": "UNRESOLVED_N_BELOW_4", "missing_at": r}
        if att[r] >= x:
            r_cap = r
        else:
            r_fail = r
            break
    nonmono = bool(r_fail is not None and any(
        att.get(r, -1) >= x for r in rates if r > r_fail and nvalid.get(r, 0) >= 1))
    if r_fail is None:
        return {"label": "CAPACITY_ABOVE_GRID", "capacity": rates[-1], "bracket": [rates[-1], None],
                "non_monotone": False}
    if r_cap is None:
        return {"label": "CAPACITY_BELOW_GRID", "capacity": None, "bracket": [None, rates[0]],
                "non_monotone": nonmono}
    return {"label": "CAPACITY_BRACKETED", "capacity": r_cap, "bracket": [r_cap, r_fail], "non_monotone": nonmono}


def pooled(cells, rate, fn):
    num = den = 0
    for c in cells.get(rate, []):
        num += sum(1 for r in c["reqs"] if fn(r))
        den += len(c["reqs"])
    return (num / den) if den else float("nan")


def bootstrap_capacity(rates, cells, t_ms, x, samples=None, seed=P.BOOTSTRAP_SEED):
    samples = samples or _BOOT_SAMPLES
    seeds = sorted(set.intersection(*[set(c["seed"] for c in cells.get(r, [])) for r in rates])) if rates else []
    if len(seeds) < 2:
        return {"n_paired_seeds": len(seeds), "ci": None}
    by = {(r, c["seed"]): c for r in rates for c in cells.get(r, [])}
    rng = random.Random(seed)
    caps, atts = [], {r: [] for r in rates}
    for _ in range(samples):
        pick = [rng.choice(seeds) for _ in seeds]
        att = {}
        for r in rates:
            num = sum(sum(1 for q in by[(r, s)]["reqs"] if q.passes(t_ms, P.ITL_SLO_MS)) for s in pick)
            den = sum(len(by[(r, s)]["reqs"]) for s in pick)
            att[r] = num / den
            atts[r].append(att[r])
        res = capacity_rule(rates, att, {r: len(pick) for r in rates}, x, 0)
        caps.append(-1.0 if res["label"] == "CAPACITY_BELOW_GRID" else res["capacity"])
    caps.sort()
    return {"n_paired_seeds": len(seeds), "samples": samples,
            "capacity_ci95": [A.percentile(caps, 0.025), A.percentile(caps, 0.975)],
            "capacity_values_note": "-1 = below grid; grid rates otherwise; n=4 seeds => 35 distinct multisets",
            "attainment_ci95": {f"{r:.2f}": [A.percentile(sorted(v), 0.025), A.percentile(sorted(v), 0.975)]
                                for r, v in atts.items()}}


# ------------------------------------------------------------------------------- campaign
def collect(out_dir: Path, shape: str):
    units = [P.parse_row(r) for r in P.units(shape)]
    cells, invalid, by_name = {}, {}, {}
    for u in units:
        if u["kind"] not in ("grid", "obs_on", "obs_off", "bridge", "floor_tail"):
            continue
        reqs, rec = load_bench(out_dir / f"bench_{u['name']}.jsonl")
        if reqs is None:
            invalid[u["name"]] = [rec]
            continue
        bad = cell_validity(u, rec, out_dir) if u["kind"] != "floor_tail" else []
        if bad:
            invalid[u["name"]] = bad
            continue
        c = {"unit": u, "reqs": reqs, "rec": rec, "seed": u["seed"], "name": u["name"]}
        by_name[u["name"]] = c
        if u["kind"] in ("grid", "obs_on"):
            cells.setdefault(u["rate"], []).append(c)
    return units, cells, invalid, by_name


def mix_of(out_dir: Path, name: str):
    p = out_dir / f"mix_{name}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def analyze(out_dir: Path, shape: str, clk_csv: Path = None) -> dict:
    out_dir = Path(out_dir)
    tkp = P.tk_file_problems(out_dir / "T_K.json")
    if tkp:   # R2: invalid, dry-run placeholder, or not a recorded floor -> nothing is scored
        return {"STATUS": "NO_VALID_FLOOR -- no arm cell may be scored (P-2, R2)", "T_K_problems": tkp}
    tk = json.loads((out_dir / "T_K.json").read_text())
    floor_s = float(tk["floor_s"])
    t_ms = {k: 1000.0 * float(tk["T_k_s"][str(k)]) for k in P.K_DECISION}
    s = P.SHAPES[shape]
    rates = list(s["rates"])
    units, cells, invalid, by_name = collect(out_dir, shape)
    nvalid = {r: len(cells.get(r, [])) for r in rates}
    res = {"POLICY": "N3 = one static arm (target-D44 sticky-OFF B1).  No arm comparison, no 'best "
                     "static' (clause 10).  Name: 'target-D44 sticky-OFF B1의 λ*_SLO' -- never 'D44 용량'.",
           "shape": shape, "floor_s": floor_s, "T_k_ms": {str(k): v for k, v in t_ms.items()},
           "x_attain": P.X_ATTAIN, "itl_slo_ms": P.ITL_SLO_MS, "rates_req_s": rates,
           "n_valid_seeds": {f"{r:.2f}": nvalid[r] for r in rates}, "invalid_cells": invalid}
    # ---- decision (per k)
    dec = {}
    for k in P.K_DECISION:
        att = {r: pooled(cells, r, lambda q, t=t_ms[k]: q.passes(t, P.ITL_SLO_MS)) for r in rates}
        rule = capacity_rule(rates, att, nvalid)
        lab = rule["label"] + f"_{k}"
        d = {"label": lab, **{kk: v for kk, v in rule.items() if kk != "label"},
             "attainment": {f"{r:.2f}": att[r] for r in rates},
             "T_k_s": t_ms[k] / 1000.0}
        if not rule["label"].startswith("UNRESOLVED"):
            d["bootstrap"] = bootstrap_capacity(rates, cells, t_ms[k], P.X_ATTAIN)
        dec[str(k)] = d
    res["decision"] = dec
    # ---- stationarity (per rate; S2 per k)
    st = {}
    for r in rates:
        cs = cells.get(r, [])
        s1 = _nanmean([s1_fe_ratio(c["reqs"]) for c in cs])
        s3 = _nanmean([s3_rho(out_dir / f"tel_{c['name']}.jsonl", out_dir / f"marks_{c['name']}.txt") for c in cs])
        row = {"S1_fe_ratio_mean": s1, "S3_rho_mean": s3, "S1_fires": bool(s1 == s1 and s1 > P.S1_FE_RATIO),
               "S3_fires": bool(s3 == s3 and s3 >= P.S3_RHO), "per_k": {}}
        for k in P.K_DECISION:
            s2 = _nanmean([s2_gap(c["reqs"], t_ms[k]) for c in cs])
            fires = bool(s2 == s2 and s2 > P.S2_VIOL_GAP)
            row["per_k"][str(k)] = {"S2_gap_mean": s2, "S2_fires": fires,
                                    "label": "UNSTABLE" if (fires or row["S1_fires"] or row["S3_fires"]) else "STATIONARY"}
        st[f"{r:.2f}"] = row
    res["stationarity"] = st
    for k in P.K_DECISION:
        d = dec[str(k)]
        if d.get("capacity") is not None:
            lab = st[f"{d['capacity']:.2f}"]["per_k"][str(k)]["label"]
            d["capacity_cell_stationarity"] = lab
            d["magnitude_citable"] = lab == "STATIONARY"
    # ---- descriptive
    res["descriptive"] = descriptive(rates, cells, floor_s, t_ms)
    # R4: ITL_BOUND_k -- the registered label/capacity differs from the ITL-65 re-scoring of the
    # same requests (existing ladder, no new constant).  Then the label is a JOINT-predicate
    # capacity and is never written as a 'TTFT capacity' (N3C-5).
    for k in P.K_DECISION:
        d = dec[str(k)]
        alt = res["descriptive"]["itl_ladder_capacity"][f"k{k}_itl65"]
        bound = (alt.get("label"), alt.get("capacity")) != (d["label"][: -len(f"_{k}")], d.get("capacity"))
        d["ITL_BOUND"] = bound
        d["flags"] = ([f"ITL_BOUND_{k}"] if bound else []) + (["NON_MONOTONE"] if d.get("non_monotone") else [])
        d["itl65_label_capacity"] = [alt.get("label"), alt.get("capacity")]
    res["cliff_zone"] = {str(k): [f"{r:.2f}" for r in rates
                                  if P.CLIFF_ATT_LO <= dec[str(k)]["attainment"][f"{r:.2f}"] <= P.CLIFF_ATT_HI]
                         for k in P.K_DECISION}
    res["observer"] = observer(by_name, shape, out_dir)
    res["bridge"] = bridge(by_name, out_dir, shape)
    res["realized_partition"] = {u["name"]: summarize_mix(mix_of(out_dir, u["name"]),
                                                          phase_events_disabled(out_dir, u["name"]))
                                 for u in units if u["kind"] in ("grid", "obs_on", "obs_off", "bridge")}
    res["phase_events_disabled_cells"] = sorted(u["name"] for u in units if phase_events_disabled(out_dir, u["name"]))
    res["forecasts"] = forecasts(shape, rates, cells, out_dir, t_ms)
    res["cg2_all_boots"] = cg2_all(out_dir, units)
    res["binding_settings"] = binding_settings(out_dir, units, by_name)
    res["substrate"] = substrate(out_dir, units, clk_csv)
    tail = out_dir / "T_K_TAIL.json"
    if tail.exists():
        tt = json.loads(tail.read_text())
        res["floor_tail_diagnostic"] = {"valid": tt.get("valid"), "floor_s": tt.get("floor_s"),
                                        "ratio_to_head": (tt["floor_s"] / floor_s) if tt.get("valid") else None,
                                        "policy": "DIAGNOSTIC ONLY -- never re-scores (P-2)"}
    return res


def descriptive(rates, cells, floor_s, t_ms) -> dict:
    out = {"curve_k": list(P.K_CURVE), "per_rate": {}, "capacity_curve": {}, "itl_ladder_capacity": {},
           "x99_capacity": {}}
    for r in rates:
        reqs = [q for c in cells.get(r, []) for q in c["reqs"]]
        if not reqs:
            continue
        fl_ms = 1000.0 * floor_s
        ratio_itl_ok = sorted(q.ttft_ms / fl_ms for q in reqs if itl_p95(q) <= P.ITL_SLO_MS)
        row = {"n_requests": len(reqs),
               "attainment_curve": [sum(q.passes(k * fl_ms, P.ITL_SLO_MS) for q in reqs) / len(reqs) for k in P.K_CURVE],
               "ttft_over_floor_itl_pass": {f"p{int(p * 100)}": A.percentile(ratio_itl_ok, p) for p in (0.5, 0.9, 0.95, 0.99)}
               if ratio_itl_ok else None,
               "itl_p95_in_45_55_frac": sum(1 for q in reqs if 45.0 <= itl_p95(q) <= 55.0) / len(reqs),
               "itl_p95_max_ms": max(itl_p95(q) for q in reqs),
               "ttft_p50_ms": A.percentile([q.ttft_ms for q in reqs], 0.5),
               "ttft_p95_ms": A.percentile([q.ttft_ms for q in reqs], 0.95),
               "expected_failures_at_1pct": 0.01 * len(reqs), "per_k": {}}
        for k in P.K_DECISION:
            t = t_ms[k]
            tt = [q.ttft_ms <= t for q in reqs]
            it = [itl_p95(q) <= P.ITL_SLO_MS for q in reqs]
            row["per_k"][str(k)] = {
                "fail_ttft_only": sum(1 for a, b in zip(tt, it) if not a and b) / len(reqs),
                "fail_itl_only": sum(1 for a, b in zip(tt, it) if a and not b) / len(reqs),
                "fail_both": sum(1 for a, b in zip(tt, it) if not a and not b) / len(reqs),
                "itl_ladder": {f"{L:g}": sum(q.passes(t, L) for q in reqs) / len(reqs) for L in P.ITL_LADDER_MS}}
        out["per_rate"][f"{r:.2f}"] = row
    nval = {r: len(cells.get(r, [])) for r in rates}
    for k in P.K_CURVE:
        att = {r: pooled(cells, r, lambda q, t=k * 1000.0 * floor_s: q.passes(t, P.ITL_SLO_MS)) for r in rates}
        out["capacity_curve"][f"{k:g}"] = capacity_rule(rates, att, nval)
    for k in P.K_DECISION:
        for L in P.ITL_LADDER_MS:
            att = {r: pooled(cells, r, lambda q, t=t_ms[k], l=L: q.passes(t, l)) for r in rates}
            out["itl_ladder_capacity"][f"k{k}_itl{L:g}"] = capacity_rule(rates, att, nval)
        att = {r: pooled(cells, r, lambda q, t=t_ms[k]: q.passes(t, P.ITL_SLO_MS)) for r in rates}
        out["x99_capacity"][str(k)] = capacity_rule(rates, att, nval, x=P.X_TAIL)
    out["POLICY"] = "descriptive only -- never a decision input (P-3/P-5)"
    return out


def observer(by_name, shape, out_dir=None) -> dict:
    s = P.SHAPES[shape]
    obs = s["obs_rate"]
    base, prop = {"ttft": {}, "itl": {}}, {"ttft": {}, "itl": {}}
    excluded = []
    for sd in s["seeds"]:
        on, off = by_name.get(f"g_{P.rtag(obs)}_{sd}"), by_name.get(f"off_{P.rtag(obs)}_{sd}")
        if not on or not off:
            continue
        if out_dir is not None and phase_events_disabled(out_dir, on["name"]):
            excluded.append(on["name"])     # R7: not ON for the whole run -> not a symmetric pair
            continue
        for tag, c, store in (("on", on, prop), ("off", off, base)):
            store["ttft"][sd] = math.log(A.percentile([q.ttft_ms for q in c["reqs"]], 0.5))
            store["itl"][sd] = math.log(statistics.median(itl_p95(q) for q in c["reqs"]))
    out = {"rate": obs, "metrics": {}, "definition": "ln(ON/OFF) per seed pair; OK <=> n>=4, |mean| < ln 1.03, "
                                                     "95% paired-bootstrap CI contains 0, for BOTH metrics"}
    verdict = "OBSERVER_OK"
    for m in ("ttft", "itl"):
        n = len(set(base[m]) & set(prop[m]))
        if n < P.N_SEEDS_MIN:
            out["metrics"][m] = {"n": n}
            verdict = "OBSERVER_UNRESOLVED_N_BELOW_4"
            continue
        ci = A.paired_bootstrap_ci({str(k): v for k, v in base[m].items()},
                                   {str(k): v for k, v in prop[m].items()},
                                   samples=P.BOOTSTRAP_SAMPLES, seed=P.BOOTSTRAP_SEED)
        ok = abs(ci["mean_effect"]) < P.OBS_TOL and ci["ci95_low"] <= 0.0 <= ci["ci95_high"]
        out["metrics"][m] = {"n": n, "mean_ln_ratio": ci["mean_effect"], "ci95": [ci["ci95_low"], ci["ci95_high"]],
                             "pct": 100.0 * (math.exp(ci["mean_effect"]) - 1.0), "ok": ok}
        if not ok and verdict == "OBSERVER_OK":
            verdict = "OBSERVER_FAIL"
    out["excluded_pairs_phase_events_disabled"] = excluded
    out["verdict"] = verdict
    out["consequence"] = ("labels are computed either way (all scored cells carry PHASE_EVENTS ON, symmetric); "
                          "on FAIL/UNRESOLVED every N3 citation adds 'PHASE_EVENTS ON 조건에서' and the "
                          "realised-partition numbers are 'observer-effect-unverified descriptive values'")
    return out


def bridge(by_name, out_dir, shape) -> dict:
    s = P.SHAPES[shape]
    br = s["bridge"]
    rows = []
    for sd in br["seeds"]:
        c = by_name.get(f"br_{br['name']}_{sd}")
        mx = summarize_mix(mix_of(out_dir, f"br_{br['name']}_{sd}"))
        rows.append({"seed": sd, "achieved_req_s": (c["rec"]["completed"] / c["rec"]["duration"]) if c else None,
                     "duration_s": c["rec"]["duration"] if c else None, "mix": mx})
    return {"POLICY": "DESCRIPTIVE side-by-side only.  NOT a 'same arm' verdict; the λ0 §4-4 rule "
                      "(3 % and range non-overlap) is NOT reused -- it is not an N3 estimand (D3).",
            "cell": br["name"], "offered": br["rate"], "n3_new_config_phase_events_off": rows,
            "lambda0_vessl_homog5_n4_achieved": list(s["lambda0_bridge_achieved"]),
            "lambda0_vessl_homog5_mix_pct": s["lambda0_bridge_mix"]}


def summarize_mix(m, pe_disabled=False):
    if not m:
        return None
    sn, ph = m.get("snapshot", {}), m.get("phase", {})
    if pe_disabled:     # R7: PHASE_EVENTS switched off mid-boot -> every phase-based value is void
        ph = {}
        m = {k: v for k, v in m.items() if k != "x_req"}
    return {"E_cnt": sn.get("e_cnt", {}).get("pct"), "E_time": sn.get("e_time", {}).get("pct"),
            "E_iter": sn.get("e_iter", {}).get("pct"), "E_qcond": sn.get("e_qcond", {}).get("pct"),
            "W_dec": ph.get("w_dec", {}).get("target_pct"), "W_idx": ph.get("w_idx", {}).get("share"),
            "X_tok": ph.get("x_tok", {}).get("pct"), "CO_a_wall": ph.get("coresidency_a_wall"),
            "cap_bound_admission_frac": ph.get("cap_bound_admission_frac"),
            "X_req": {k: m.get("x_req", {}).get(k) for k in ("status", "mean", "p10", "p50", "p90", "frac_ge_half")},
            "dropped_events_max": max(sn.get("dropped_events_max", 0) or 0, ph.get("dropped_events_max", 0) or 0),
            "phase_events_voided": bool(pe_disabled), "arm_identity": m.get("arm_identity")}


def forecasts(shape, rates, cells, out_dir, t_ms) -> dict:
    """Registered falsifiable forecasts (PREREG §9).  HELD / FALSIFIED / UNRESOLVED -- never a
    capacity input."""
    def etime(r):
        v = [summarize_mix(mix_of(out_dir, c["name"])) for c in cells.get(r, [])]
        v = [x["E_time"] for x in v if x and x.get("E_time") is not None]
        return statistics.fmean(v) if len(v) >= P.N_SEEDS_MIN else None
    out = {}
    if shape == "B":
        low = [r for r in rates if r <= 0.45 + 1e-9]
        e = [etime(r) for r in low]
        if any(x is None for x in e):
            out["F-B1_low_load_dilution"] = "UNRESOLVED"
        else:
            rho = spearman(low, e)
            out["F-B1_low_load_dilution"] = {"verdict": "HELD" if (rho >= 0.9 and e[0] <= 40.0) else "FALSIFIED",
                                             "spearman": rho, "E_time_at_0.10": e[0], "E_time_seed_means": e}
        itl_fail = [1 for r in rates for c in cells.get(r, []) for q in c["reqs"] if itl_p95(q) > P.ITL_SLO_MS]
        out["F-B2_itl_never_binding"] = "HELD" if not itl_fail else {"verdict": "FALSIFIED", "n_requests": len(itl_fail)}
    else:
        e = [etime(r) for r in (1.35, 1.55, 1.80)]
        if any(x is None for x in e):
            out["F-A1_beta_exposure"] = "UNRESOLVED"
        else:
            ok = all(3.0 <= x <= 15.0 for x in e[:2]) and e[0] <= e[1] <= e[2]
            out["F-A1_beta_exposure"] = {"verdict": "HELD" if ok else "FALSIFIED", "E_time_seed_means": e}
        hit = []
        for r in (2.70, 3.00):
            for c in cells.get(r, []):
                if any(itl_p95(q) > P.ITL_SLO_MS for q in c["reqs"]):
                    hit.append((r, c["seed"]))
        if not cells.get(2.70) and not cells.get(3.00):
            out["F-A2_itl_binds_near_saturation"] = "UNRESOLVED"
        else:
            out["F-A2_itl_binds_near_saturation"] = {"verdict": "HELD" if hit else "FALSIFIED", "cells": hit}
    return out


def cg2_all(out_dir, units) -> dict:
    """Descriptive extension of the in-Job CG2 gate: distinct greedy warm-up output tuples (same 8
    prompts, concurrency 1) among PHASE_EVENTS ON boots and among OFF boots."""
    seen = {"on": set(), "off": set()}
    n = {"on": 0, "off": 0}
    for u in units:
        p = out_dir / f"warmup_{u['name']}.jsonl"
        if not p.exists():
            continue
        try:
            t = json.loads(p.read_text().splitlines()[0]).get("generated_texts")
        except (ValueError, IndexError):
            continue
        k = "on" if u["phase_events"] == 1 else "off"
        n[k] += 1
        seen[k].add(json.dumps(t))
    return {"n_boots": n, "distinct_on": len(seen["on"]), "distinct_off": len(seen["off"]),
            "on_equals_off": bool(seen["on"]) and seen["on"] == seen["off"] and len(seen["on"]) == 1,
            "policy": "descriptive; the gate is the in-Job CG2 (floor_head vs first OFF boot)"}


def binding_settings(out_dir, units, by_name) -> dict:
    mm, mt = set(), set()
    for u in units:
        b = out_dir / f"banner_{u['name']}.txt"
        if b.exists():
            t = b.read_text()
            m1 = re.search(r"max_mamba_cache_size[:=]\s*(\d+)", t)
            m2 = re.search(r"max_total_num_tokens[:=]\s*(\d+)", t)
            if m1:
                mm.add(int(m1.group(1)))
            if m2:
                mt.add(int(m2.group(1)))
    mpt = {c["rec"].get("server_info", {}).get("max_prefill_tokens") for c in by_name.values()}
    return {"max_running_requests": P.MAX_RUNNING, "max_mamba_cache_size": sorted(mm),
            "max_total_num_tokens": sorted(mt), "max_prefill_tokens": sorted(x for x in mpt if x is not None),
            "context_length": P.CTX, "note": "cap 48 = mamba 48 (CONSENSUS §3 item 23); B is bound by "
                                             "max_prefill_tokens 16384 per prefill batch (clause 8)"}


def substrate(out_dir, units, clk_csv) -> dict:
    out = {"cpu_stat_delta": {}, "clk": None}
    for u in units:
        p = out_dir / f"cpu_{u['name']}.txt"
        if not p.exists():
            continue
        blocks, cur = [], {}
        for ln in p.read_text().splitlines():
            if ln.startswith("#"):
                cur = {}
                blocks.append(cur)
                continue
            f = ln.split()
            if len(f) == 2 and f[1].isdigit():
                cur[f[0]] = int(f[1])
        if len(blocks) >= 2:
            out["cpu_stat_delta"][u["name"]] = {k: blocks[1].get(k, 0) - blocks[0].get(k, 0)
                                               for k in ("nr_throttled", "throttled_usec") if k in blocks[0]}
    if clk_csv and Path(clk_csv).exists() and (out_dir / "TIMING.tsv").exists():
        out["clk"] = clk_windows(Path(clk_csv), out_dir / "TIMING.tsv")
    return out


def clk_windows(clk_csv: Path, timing: Path) -> dict:
    import csv
    import datetime as dt
    rows = []
    with open(clk_csv) as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            try:
                t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
                rows.append((t, int(r[1].strip().split()[0]), int(r[4].strip(), 16)))
            except (ValueError, IndexError):
                continue
    out = {}
    lines = timing.read_text().splitlines()
    hdr = lines[0].split("\t") if lines else []
    for ln in lines[1:]:
        f = dict(zip(hdr, ln.split("\t")))
        try:
            a, b = float(f["bench_start_epoch"]), float(f["bench_end_epoch"])
        except (KeyError, ValueError):
            continue
        w = [x for x in rows if a <= x[0] <= b]
        if w:
            out[f["unit"]] = {"n": len(w), "sw_power_cap_pct": 100.0 * sum(1 for x in w if x[2] & 0x4) / len(w),
                              "sm_mhz_mean": statistics.fmean(x[1] for x in w)}
    return out


# ------------------------------------------------------------------------------- selftest
def _synth(td: Path, shape: str, ttft_fn, itl_ms=20.0, drop=None, obs_shift=0.0, floor_s=1.0):
    """Synthetic campaign: per (rate, seed) a bench record whose TTFTs come from ttft_fn(rate, i)."""
    td.mkdir(parents=True, exist_ok=True)
    (td / "T_K.json").write_text(json.dumps({"valid": True, "floor_s": floor_s, "ttfts_s": [floor_s] * P.FLOOR_N,
                                             "T_k_s": {str(k): k * floor_s for k in P.K_DECISION}}))
    clean = json.dumps({"event": "runtime_snapshot", "phase": "benchmark", "decode_running_batch_size": 2,
                        "prefill_sms": 64, "decode_sms": 44, "timestamp_monotonic_s": 1.0}) + "\n" + \
        json.dumps({"event": "decode_iteration", "t_launch": 1.0, "t_sync": 1.01, "bs": 2,
                    "prefill_sms": 64, "decode_sms": 44}) + "\n"     # D1: both legs carry evidence
    for row in P.units(shape):
        u = P.parse_row(row)
        if u["kind"] not in ("grid", "obs_on", "obs_off", "bridge"):
            continue
        if drop and u["name"] in drop:
            continue
        n = u["np"]
        rng = random.Random(u["seed"] * 7 + n)
        sh = (1.0 + obs_shift) if u["kind"] == "obs_on" else 1.0
        rate = u["rate"]
        ttfts = [ttft_fn(rate, i, n, rng) * sh for i in range(n)]
        itls = [[itl_ms * sh / 1000.0] * (u["output_len"] - 1) for _ in range(n)]   # bench stores seconds
        rec = {"completed": n, "errors": [""] * n, "total_input_tokens": n * u["input_len"],
               "output_lens": [u["output_len"]] * n, "random_range_ratio": 1.0, "request_rate": rate,
               "duration": n / rate, "ttfts": ttfts, "itls": itls, "server_info": {"max_prefill_tokens": 16384}}
        (td / f"bench_{u['name']}.jsonl").write_text(json.dumps(rec) + "\n")
        (td / f"tel_{u['name']}.jsonl").write_text(clean)


def selftest() -> None:
    import tempfile
    global _BOOT_SAMPLES
    _BOOT_SAMPLES = 200
    sh = "B"
    rates = P.SHAPES[sh]["rates"]

    def cliff_at(rc):
        # TTFT ~ floor below rc; ~10x floor above -> every k in {3,5,10} fails above rc
        return lambda r, i, n, rng: (1.0 + 0.2 * rng.random()) if r <= rc + 1e-9 else (20.0 + rng.random())

    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        _synth(td / "a", sh, cliff_at(0.38))
        r = analyze(td / "a", sh)
        for k in P.K_DECISION:
            d = r["decision"][str(k)]
            assert d["label"] == f"CAPACITY_BRACKETED_{k}" and d["capacity"] == 0.38 and d["bracket"] == [0.38, 0.45], d
            assert d["bootstrap"]["capacity_ci95"] == [0.38, 0.38], d["bootstrap"]
        assert r["observer"]["verdict"] == "OBSERVER_OK", r["observer"]
        # control 1: the threshold is live -- a k-dependent cliff separates the three k
        _synth(td / "b", sh, lambda r_, i, n, rng: 1.0 + 12.0 * r_ * (i / n))   # pass iff r <~ (k-1)/10.8
        r = analyze(td / "b", sh)
        caps = [r["decision"][str(k)]["capacity"] for k in P.K_DECISION]
        assert caps == [0.15, 0.30, 0.77], caps
        # control 2: lowest rate fails -> BELOW; nothing fails -> ABOVE
        _synth(td / "c", sh, cliff_at(0.0))
        assert analyze(td / "c", sh)["decision"]["3"]["label"] == "CAPACITY_BELOW_GRID_3"
        _synth(td / "d", sh, cliff_at(10.0))
        assert analyze(td / "d", sh)["decision"]["10"]["label"] == "CAPACITY_ABOVE_GRID_10"
        # control 3: a missing seed at a rate the rule must read -> UNRESOLVED, not a capacity
        seeds = P.SHAPES[sh]["seeds"]
        _synth(td / "e", sh, cliff_at(0.38), drop={f"g_r022_{seeds[0]}"})
        lab = analyze(td / "e", sh)["decision"]["5"]["label"]
        assert lab == "UNRESOLVED_N_BELOW_4_5", lab
        # ... but a missing seed ABOVE the first failure does not block the label
        _synth(td / "f", sh, cliff_at(0.38), drop={f"g_r077_{seeds[0]}"})
        assert analyze(td / "f", sh)["decision"]["5"]["label"] == "CAPACITY_BRACKETED_5"
        # control 4: the ITL leg is live -- ITL p95 60 ms fails every request at the 50 ms predicate
        _synth(td / "g", sh, cliff_at(10.0), itl_ms=60.0)
        r = analyze(td / "g", sh)
        assert r["decision"]["10"]["label"] == "CAPACITY_BELOW_GRID_10", r["decision"]["10"]
        assert r["descriptive"]["itl_ladder_capacity"]["k10_itl65"]["label"] == "CAPACITY_ABOVE_GRID"
        assert r["forecasts"]["F-B2_itl_never_binding"]["verdict"] == "FALSIFIED"
        # control 5: observer effect 6 % on both metrics -> FAIL
        _synth(td / "h", sh, cliff_at(0.38), obs_shift=0.06)
        assert analyze(td / "h", sh)["observer"]["verdict"] == "OBSERVER_FAIL"
        # control 6: S2 queue growth (second half violates) -> UNSTABLE at that rate
        _synth(td / "i", sh, lambda r_, i, n, rng: 1.0 if (r_ < 0.3 or i < n // 2) else 30.0)
        st = analyze(td / "i", sh)["stationarity"]
        assert st["0.30"]["per_k"]["3"]["label"] == "UNSTABLE" and st["0.22"]["per_k"]["3"]["label"] == "STATIONARY", st["0.30"]
        # R4: default synthetic is TTFT-bound (ITL 20 ms) -> no ITL_BOUND; the ITL 60 ms campaign "g"
        # is BELOW at ITL 50 but ABOVE at ITL 65 -> ITL_BOUND_10
        ra = analyze(td / "a", sh)
        assert not any(ra["decision"][str(k)]["ITL_BOUND"] for k in P.K_DECISION)
        rg = analyze(td / "g", sh)
        assert rg["decision"]["10"]["ITL_BOUND"] and "ITL_BOUND_10" in rg["decision"]["10"]["flags"]
        # R3: inject ONE decode iteration on (84,24) into a scored cell below the cliff -> that cell is
        # invalid -> n=3 at 0.22 -> UNRESOLVED (control: the clean campaign "a" is BRACKETED)
        bad = json.dumps({"event": "decode_iteration", "t_launch": 1.0, "t_sync": 1.01, "bs": 3,
                          "prefill_sms": 84, "decode_sms": 24}) + "\n"
        p = td / "a" / f"tel_g_r022_{seeds[1]}.jsonl"
        p.write_text(p.read_text() + bad)
        r3 = analyze(td / "a", sh)
        assert r3["decision"]["5"]["label"] == "UNRESOLVED_N_BELOW_4_5", r3["decision"]["5"]
        assert "arm identity violated" in " ".join(r3["invalid_cells"][f"g_r022_{seeds[1]}"])
        (td / "a" / f"tel_g_r022_{seeds[1]}.jsonl").unlink()     # no telemetry -> unverifiable -> invalid
        assert "unverifiable" in " ".join(analyze(td / "a", sh)["invalid_cells"][f"g_r022_{seeds[1]}"])
        # D1 variant 1: an EMPTY telemetry file (the runner pre-creates it) -> unverifiable -> invalid
        (td / "a" / f"tel_g_r022_{seeds[1]}.jsonl").write_text("")
        r = analyze(td / "a", sh)
        assert "D1: no decode-busy snapshot" in " ".join(r["invalid_cells"][f"g_r022_{seeds[1]}"]), r["invalid_cells"]
        assert r["decision"]["5"]["label"] == "UNRESOLVED_N_BELOW_4_5", r["decision"]["5"]
        # D1 variant 2: snapshots only, no decode_iteration, PHASE_EVENTS ON cell -> invalid ...
        snap_only = json.dumps({"event": "runtime_snapshot", "phase": "benchmark", "decode_running_batch_size": 2,
                                "prefill_sms": 64, "decode_sms": 44, "timestamp_monotonic_s": 1.0}) + "\n"
        (td / "a" / f"tel_g_r022_{seeds[1]}.jsonl").write_text(snap_only)
        r = analyze(td / "a", sh)
        assert "PHASE_EVENTS ON but no decode_iteration" in " ".join(r["invalid_cells"][f"g_r022_{seeds[1]}"])
        # ... control: the same file under an R7 marker is valid on the snapshot leg alone
        (td / "a" / f"phase_events_disabled_g_r022_{seeds[1]}").write_text("x\n")
        assert f"g_r022_{seeds[1]}" not in analyze(td / "a", sh)["invalid_cells"]
        # ... control: an OFF observer cell with snapshots only is valid
        (td / "a" / f"tel_off_r030_{seeds[0]}.jsonl").write_text(snap_only)
        assert f"off_r030_{seeds[0]}" not in analyze(td / "a", sh)["invalid_cells"]
        # R7: a mid-boot PHASE_EVENTS disable on one observer ON cell drops that pair -> n=3 -> UNRESOLVED
        _synth(td / "j", sh, cliff_at(0.38))
        (td / "j" / f"srv_g_r030_{seeds[0]}.log").write_text("WARNING " + PE_DISABLED + ": boom\n")
        rj = analyze(td / "j", sh)
        assert rj["observer"]["verdict"] == "OBSERVER_UNRESOLVED_N_BELOW_4", rj["observer"]
        assert rj["phase_events_disabled_cells"] == [f"g_r030_{seeds[0]}"]
        assert rj["decision"]["5"]["label"] == "CAPACITY_BRACKETED_5"   # scoring itself unaffected
        # R2 + control 7: invalid floor, dry-run placeholder, or no ttfts_s -> nothing scored
        for tk in ({"valid": False},
                   {"valid": True, "dryrun": True, "floor_s": 1.0, "T_k_s": {"3": 3.0, "5": 5.0, "10": 10.0}},
                   {"valid": True, "floor_s": 1.0, "T_k_s": {"3": 3.0, "5": 5.0, "10": 10.0}}):
            (td / "j" / "T_K.json").write_text(json.dumps(tk))
            assert analyze(td / "j", sh)["STATUS"].startswith("NO_VALID_FLOOR"), tk
    # capacity rule non-monotone flag
    rr = [1, 2, 3, 4]
    x = capacity_rule(rr, {1: .95, 2: .80, 3: .95, 4: .1}, {r: 4 for r in rr})
    assert x["label"] == "CAPACITY_BRACKETED" and x["capacity"] == 1 and x["non_monotone"], x
    print("N3 ANALYZE SELFTEST OK (bracketed + CI; k-dependent cliff orders capacities; BELOW/ABOVE; "
          "n<4 below the first failure -> UNRESOLVED, above it -> not blocking; ITL leg live; observer 6% -> FAIL; "
          "S2 -> UNSTABLE; invalid/dry-run/no-ttfts floor -> nothing scored (R2); non-monotone flag; ITL_BOUND (R4); arm-identity injection -> cell invalid (R3); D1 empty-telemetry / ON-without-decode-events -> invalid, R7/OFF snapshot-leg controls valid; PHASE_EVENTS mid-boot disable -> pair dropped (R7))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--shape", choices=("A", "B"))
    ap.add_argument("--clk")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.out_dir or not a.shape:
        ap.error("--out-dir and --shape required")
    res = analyze(Path(a.out_dir), a.shape, Path(a.clk) if a.clk else None)
    txt = json.dumps(res, indent=2, sort_keys=True, default=str)
    if a.json_out:
        Path(a.json_out).write_text(txt + "\n")
    for k, d in (res.get("decision") or {}).items():
        print(f"k={k}: {d['label']} capacity={d.get('capacity')} bracket={d.get('bracket')} "
              f"stationarity={d.get('capacity_cell_stationarity')}")
    print(f"observer: {res.get('observer', {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
