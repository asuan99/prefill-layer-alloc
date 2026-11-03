#!/usr/bin/env python3
"""Analysis of PREREG_STOCK_BASE rev3 (2026-11-03; rev2.1 sealed 2026-11-02).  Pure functions over the raw trees of the Jobs; CPU; no server.

  sb_analyze.py addendum --j1 RAW_J1 [--j1ext RAW] [--j1down RAW] [--j1b RAW_J1b] --out ADD.json   # rates for the next Job
  sb_analyze.py report   --j1 RAW [--j1down RAW] --j1b RAW --j2 RAW --j3 RAW [--out ANALYSIS.json]  # labels, predictions, "not measured"

Registered rules implemented here (each guarded by a mutant in sb_mutants.py):
  * good request = no error AND TTFT <= SLO AND request-internal token-ITL p95 <= SLO (gate 4)
  * goodput = sum(good) / sum(segment durations) (gate 7: durations ADD, never max)
  * lambda* = pick rule (sb_runner.lambda_pick; tie -> lower rate); sustained lambda* = largest sustained point passing (§3)
  * lambda*_ref = min over the SUSTAINED capacity of SP.REF_ARMS present (rev3: VL-stock, PM-ref only; the SGLang-default arms are
    excluded -- first-Job ladder bottom failure, ITL-axis cliff; the scope of that min is recorded)
  * rev3: J1-down (SG 3 arms, 5.516 / 1.15^k, k = 1..4) -> pick rule over J1 + J1-down points; bottom failure again -> LADDER_BOTTOM_FAIL_2
    (no value); feeds Q1 (and the PREREG §8 Q1-SG row) only, never lambda*_ref.  refcell rate = burst lambda* of SP.REFCELL_ARM (PM-ref)
  * rev3: pairs of SP.SIGN_ONLY_QUESTIONS (Q3, Q5, Q8) carry a goodput SIGN label only (A_ABOVE_B_SIGN iff CI lo > 0, A_BELOW_B_SIGN iff
    CI hi < 0, else SIGN_UNRESOLVED) + BOTH_SATURATED / INVALID / BOOT_NOISE / MIX_NOT_FIRED + latency co-metrics; tag MAGNITUDE_NOT_CITABLE
  * pair label (§4): A_ABOVE_B iff lo > 0 and mean >= ln1.03; A_BELOW_B iff hi < 0 and mean <= -ln1.03; EQUIV_WITHIN_3PCT iff
    |mean| < ln1.03 and the t(3) CI is inside (-ln1.03, ln1.03); otherwise UNRESOLVED_INTERVAL (also n < 4)
  * K3 placebo (SG-rep vs SG-stock): invalid iff n < 4 or |mean| >= ln1.03 or CI excludes 0 -> every label of the cell BOOT_NOISE
  * validity chain: trace hash per seed, cliff (TTFT within SLO +-5 % > 10 %) in neither arm, one Job and one GPU per pair, engine-tree
    stamps per PREREG §1 (VL-stock: engine_tree "vllm", registered venv + SGLang freeze sha, all VL boots of the Job identical), placebo,
    definition smoke (engine pairs), zero request failures
  * TTFT time base (§5-0, two steps; engine pairs only): (a) smoke offset = median(client - server) of sequential requests; (b) in-load
    offset = run mean client TTFT - server histogram mean (delta sum / delta count around the run) of the J2 INSTR_METRICS run (SGLang)
    and of every VL-stock R1 measurement run.  |b - a| > 20 ms (absolute only, PREREG rev1.1) -> TTFT_OFFSET_SHIFT -> engine pairs
    "INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT"; (b) not computable -> TTFT_OFFSET_UNCHECKED -> engine pairs INVALID (fail-closed)
  * tags: CELL_OUT_OF_BAND (steady-state cell R2 (W1) / R3 (W3) attainment: all arms > 0.9 or best < 0.5), SUBSTRATE_SHIFT (refcell vs the identical Job-1 cell
    differs by >= 10 %p; SUBSTRATE_UNCHECKED when Job 1 holds no identical cell), MIX_NOT_FIRED (rev2: SG-mix server_args lack
    enable_mixed_chunk=True with chunked_prefill_size > 0 on any run; the rev1 "proxy < 50" rule is retired)
  * rev2 (audit r1): BOTH_SATURATED (both arms >= 0.98 attainment on all 4 seeds -> no goodput label, every cell); latency co-metrics for
    Q5/Q8 (seed-level paired ln TPOT mean / TTFT p50 / median request-internal ITL p95, same 3 % + t(3) rule, suffix _TPOT/_TTFT/_ITL);
    SG-mix firing from server_args; Q6 grid {512, 2k}, P-4 UNSCORABLE at n < 10, Q6 INVALID on any other-stream extend; forced Q1-Q8 rows
  * rev2.1 (re-audit r2 (a), 2026-11-02): P-3 = UNSCORABLE (why BOTH_SATURATED) when the 80 % cell PM-ref vs SG-stock pair is saturated
  * r_ser / r_mix: decode_probe.pacing_ratios on the load window ONLY (t_load_start/end mapped to the probe clock through the probe's
    boot record), and SEPARATELY on the synthetic single-request window of the SG probe run (t_synth_start/end; 10 x 512 + 10 x 2k prompts,
    2026-11-02): the two windows are never pooled; P-4 = single-request (n_reqs == 1) 2k spans of the synthetic window only (512 descriptive);
    PM-ref r_span = implfix pc_reduce definition + the same window filter (new function, implfix file untouched)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
import sb_spec as SP  # noqa: E402
import sb_runner as RU  # noqa: E402
import ph_verdict as V  # noqa: E402

MARGIN = SP.MARGIN
FORECAST_CSV = HERE / "R_SER_FORECAST_NANO.csv"
PC_BAND, PC_BACKLOG_R, PC_OVERSPLIT_R = (0.75, 1.25), 1.5, 0.5       # implfix pi_spec values (copied, reference only)
_DP = {}


def decode_probe_module():
    if "m" not in _DP:
        spec = importlib.util.spec_from_file_location("_sb_decode_probe", SP.ENG / "src" / "multiplex" / "decode_probe.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["_sb_decode_probe"] = m
        spec.loader.exec_module(m)
        _DP["m"] = m
    return _DP["m"]


# ---- loading -----------------------------------------------------------------------------------------------------------
def _j(p):
    return json.loads(Path(p).read_text())


def _lines(path):
    p = Path(path)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    if not p.exists():
        return []
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
        return [json.loads(x) for x in f if x.strip()]


def load_job(raw):
    if raw is None or not Path(raw).exists():
        return None
    raw = Path(raw)
    runs = [_j(p) for p in sorted((raw / "runs").glob("*.json"))]
    boots = {}
    for p in sorted((raw / "boots").glob("*/boot.json")):
        b = _j(p)
        b["dir"] = str(p.parent)
        boots[b["boot_id"]] = b
    done = {}
    for p in sorted((raw / "boots").glob("*/DONE.json")):
        d = _j(p)
        done[d.get("boot_id") or p.parent.name] = d
    return {"raw": str(raw), "plan": _j(raw / "plan.json") if (raw / "plan.json").exists() else None, "runs": runs, "boots": boots,
            "done": done,
            "floor": _j(raw / "floor.json") if (raw / "floor.json").exists() else None,
            "w3": _j(raw / "W3_LAMBDA.json") if (raw / "W3_LAMBDA.json").exists() else None}


# ---- basics ------------------------------------------------------------------------------------------------------------
def goodput(run, ttft_slo, itl_slo=SP.ITL_MS, phase=None):
    good = dur = 0.0
    for s in run["segments"]:
        if phase is not None and s["phase"] != phase:
            continue
        good += sum(1 for r in s["requests"] if V.req_good(r, ttft_slo, itl_slo))
        dur += float(s["duration_s"])
    if dur <= 0:
        raise ValueError(f"{run['run_id']}: zero duration")
    return good / dur


def attainment(run, ttft_slo, itl_slo=SP.ITL_MS, phase=None):
    reqs = [r for s in run["segments"] if phase is None or s["phase"] == phase for r in s["requests"]]
    return sum(1 for r in reqs if V.req_good(r, ttft_slo, itl_slo)) / len(reqs) if reqs else None


def label(ci, tokens):
    m, lo, hi, n = ci
    a, b = tokens
    if n < 4 or any(isinstance(x, float) and math.isnan(x) for x in (m, lo, hi)):
        return "UNRESOLVED_INTERVAL"
    if lo > 0 and m >= MARGIN:
        return f"{a}_ABOVE_{b}"
    if hi < 0 and m <= -MARGIN:
        return f"{a}_BELOW_{b}"
    if abs(m) < MARGIN and lo > -MARGIN and hi < MARGIN:
        return "EQUIV_WITHIN_3PCT"
    return "UNRESOLVED_INTERVAL"


def sign_label(ci, tokens):
    """rev3 (SIGN_ONLY_QUESTIONS): the goodput sign only -- no 3 % margin, no EQUIV (a magnitude statement)."""
    m, lo, hi, n = ci
    a, b = tokens
    if n < 4 or not all(isinstance(x, (int, float)) and x == x for x in (m, lo, hi)):      # n < 4 or NaN -> no sign
        return "SIGN_UNRESOLVED"
    if lo > 0:
        return f"{a}_ABOVE_{b}_SIGN"
    if hi < 0:
        return f"{a}_BELOW_{b}_SIGN"
    return "SIGN_UNRESOLVED"


def paired(runs_a, runs_b, ttft_slo):
    """runs_x: {seed: run}.  seed-level paired ln(goodput A / goodput B), t(n-1) 95 %."""
    seeds = sorted(set(runs_a) & set(runs_b))
    d = []
    for s in seeds:
        ga, gb = goodput(runs_a[s], ttft_slo), goodput(runs_b[s], ttft_slo)
        d.append(math.log(ga / gb) if ga > 0 and gb > 0 else math.nan)
    ok = [x for x in d if not math.isnan(x)]
    ci = V.paired_t_ci(ok) if ok else (math.nan, math.nan, math.nan, 0)
    sd = statistics.stdev(ok) if len(ok) >= 2 else math.nan
    return {"seeds": seeds, "ln_ratio": d, "mean": ci[0], "lo": ci[1], "hi": ci[2], "n": ci[3], "sd": sd}


def placebo_invalid(pl):
    if pl is None or pl["n"] < 4 or any(isinstance(pl[k], float) and math.isnan(pl[k]) for k in ("mean", "lo", "hi")):
        return True
    return abs(pl["mean"]) >= MARGIN or pl["lo"] > 0 or pl["hi"] < 0


# ---- rev2: saturation and latency co-metrics (audit r1 B) -------------------------------------------------------------------
def run_reqs(run):
    return [q for s in run["segments"] for q in s["requests"]]


def latency_of(run, metric):
    """Seed-level latency statistic of one run: TPOT = mean over requests (>= 2 tokens, no error) of the mean token ITL; TTFT = p50 of the
    run's TTFTs; ITL = median over requests of the request-internal token-ITL p95."""
    reqs = [q for q in run_reqs(run) if not q.get("error")]
    if metric == "TPOT":
        xs = [statistics.fmean(q["token_itl_ms"]) for q in reqs if q.get("token_itl_ms")]
        return statistics.fmean(xs) if xs else None
    if metric == "TTFT":
        xs = [q["ttft_ms"] for q in reqs if q.get("ttft_ms") is not None]
        return V.percentile(xs, 0.5) if xs else None
    if metric == "ITL":
        xs = [V.percentile(q["token_itl_ms"], 0.95) for q in reqs if q.get("token_itl_ms")]
        return statistics.median(xs) if xs else None
    raise ValueError(metric)


def paired_latency(runs_a, runs_b, metric):
    seeds = sorted(set(runs_a) & set(runs_b))
    d = []
    for s in seeds:
        xa, xb = latency_of(runs_a[s], metric), latency_of(runs_b[s], metric)
        d.append(math.log(xa / xb) if xa and xb and xa > 0 and xb > 0 else math.nan)
    ok = [x for x in d if not math.isnan(x)]
    ci = V.paired_t_ci(ok) if ok else (math.nan, math.nan, math.nan, 0)
    sd = statistics.stdev(ok) if len(ok) >= 2 else math.nan
    return {"metric": metric, "seeds": seeds, "ln_ratio": d, "mean": ci[0], "lo": ci[1], "hi": ci[2], "n": ci[3], "sd": sd}


def seed_attainments(runs, ttft_slo):
    return {s: attainment(r, ttft_slo) for s, r in runs.items()}


def both_saturated(runs_a, runs_b, ttft_slo):
    """BOTH_SATURATED: both arms reach attainment >= 0.98 on every evaluation seed -> goodput = n / duration for both, the ln ratio is a
    drain difference only (audit r1 B), so no goodput label (EQUIV forbidden)."""
    for runs in (runs_a, runs_b):
        if set(runs) != set(SP.EVAL_SEEDS):
            return False
        if any(x is None or x < SP.SATURATED_ATTAIN for x in seed_attainments(runs, ttft_slo).values()):
            return False
    return True


# ---- capacity ----------------------------------------------------------------------------------------------------------
def ladder_lambda(job_runs, arm, ttft_slo=SP.W1_TTFT_MS, cells=("LADDER",)):
    pts = [(r["spec"]["rate"], attainment(r, ttft_slo)) for r in job_runs if r["arm"] == arm and r.get("cell") in cells]
    lam, st = RU.lambda_pick(pts)
    return {"lambda_star": lam if st == "BRACKETED" else None, "lambda_star_raw": lam, "status": st, "points": sorted(pts)}


def sustained_lambda(job_runs, arm):
    rows = []
    for r in job_runs:
        if r["arm"] == arm and r.get("cell") == "SUSTAIN":
            e = RU.sustain_eval(r)
            rows.append({"rate": r["spec"]["rate"], "factor": r["spec"].get("factor"), "pass": e["pass"], "ttft_growth": e["ttft_growth"],
                         "min_completion_ratio": min(w["completion_ratio"] for w in e["windows"]),
                         "first_token_ratio": [w["first_token_ratio"] for w in e["windows"]]})
    ok = [x["rate"] for x in rows if x["pass"]]
    return {"lambda_star_sustained": max(ok) if ok else None, "status": "OK" if ok else ("NO_POINTS" if not rows else "NONE_PASS"),
            "points": sorted(rows, key=lambda x: x["rate"])}


def ladder_down(j1_runs, j1down, arm, ttft_slo=SP.W1_TTFT_MS):
    """rev3: the same pick rule over the first-Job ladder points + the J1-down points of `arm`.  A failing bottom again (the lowest
    J1-down rung) -> LADDER_BOTTOM_FAIL_2 and no value."""
    if j1down is None or not any(r["arm"] == arm and r.get("cell") == "LADDER_DOWN" for r in j1down["runs"]):
        return {"lambda_star": None, "status": "NOT_RUN", "points": []}
    L = ladder_lambda(list(j1_runs) + list(j1down["runs"]), arm, ttft_slo, cells=("LADDER", "LADDER_DOWN"))
    if L["status"] == "LADDER_BOTTOM_FAIL":
        L = dict(L, status=SP.LADDER_DOWN_FAIL, lambda_star=None)
    return L


def build_addendum(j1, j1ext=None, j1b=None, w1_sha=None, j1down=None):
    runs = list(j1["runs"]) + (list(j1ext["runs"]) if j1ext else [])
    add = {"lambda_star_burst": {}, "ladder": {}, "stamps": {}, "w1_lengths_sha256": w1_sha}
    for a in SP.LADDER_ARMS:
        if not any(r["arm"] == a for r in runs):
            add["ladder"][a] = {"status": "NOT_RUN"}
            continue
        L = ladder_lambda(runs, a)
        add["ladder"][a] = L
        add["lambda_star_burst"][a] = L["lambda_star"]
    add["lambda_star_burst"] = {k: v for k, v in add["lambda_star_burst"].items() if v is not None}
    add["refcell_arm"] = SP.REFCELL_ARM
    add["refcell_rate"] = add["lambda_star_burst"].get(SP.REFCELL_ARM)
    add["refcell_rule"] = (f"rev3: refcell = {SP.REFCELL_ARM} at its burst lambda* rung (a first-Job ladder rung: the identical cell exists "
                           "in Job 1); rev2.1 used SG-stock, which has no burst lambda*")
    add["stamps"]["J1"] = sorted({(r.get("job_name"), r.get("gpu_uuid")) for r in j1["runs"]})
    if j1down is not None:                                  # rev3: Q1 only -- kept apart from lambda_star_burst (J1b input)
        add["ladder_down"] = {a: ladder_down(runs, j1down, a) for a in SP.LADDER_DOWN_ARMS}
        add["lambda_star_burst_down"] = {a: v["lambda_star"] for a, v in add["ladder_down"].items()}
        add["stamps"]["J1-down"] = sorted({(r.get("job_name"), r.get("gpu_uuid")) for r in j1down["runs"]})
    add["lambda_star_ref_arms"] = list(SP.REF_ARMS)
    add["lambda_star_ref_excluded"] = {a: "rev3: excluded from the reference capacity (first-Job ladder bottom failure, ITL-axis cliff)"
                                       for a in SP.LADDER_DOWN_ARMS}
    if j1b:
        add["lambda_star_sustained"] = {}
        for a in SP.LADDER_ARMS:
            if any(r["arm"] == a for r in j1b["runs"]):
                add["lambda_star_sustained"][a] = sustained_lambda(j1b["runs"], a)
        vals = {a: v["lambda_star_sustained"] for a, v in add["lambda_star_sustained"].items() if a in SP.REF_ARMS}
        add["lambda_star_ref"] = min(vals.values()) if vals and all(v is not None for v in vals.values()) else None
        add["lambda_star_ref_scope"] = sorted(vals)
        add["lambda_star_ref_missing_arms"] = sorted(set(SP.REF_ARMS) - set(vals))
        if any(v is None for v in vals.values()):
            add["lambda_star_ref_why_none"] = sorted(a for a, v in vals.items() if v is None)
    return add


# ---- stamps / engine tree ------------------------------------------------------------------------------------------------
def tree_failures(run, job):
    """PREREG §1 stamp check of the boot that served `run`."""
    b = job["boots"].get(run.get("boot_id")) or {}
    st = b.get("tree_stamp") or run.get("tree_stamp") or {}
    arm = run["arm"]
    why = []
    if arm == "VL-stock":
        if b.get("engine") != "vllm":
            why.append(f"ENGINE_MISMATCH:{arm}")
        if st.get("engine_tree") != "vllm" or run.get("engine_tree") != "vllm":
            why.append(f"TREE_MISMATCH:{arm}:{st.get('engine_tree')}")
        if st.get("vllm_venv_freeze_sha256") != SP.VLLM_VENV_FREEZE_SHA256:
            why.append(f"VLLM_VENV_FREEZE:{arm}")
        if st.get("sglang_freeze_sha256") != SP.SGLANG_FREEZE_SHA256:
            why.append(f"SGLANG_FREEZE:{arm}")
        return why
    want = "pristine" if arm in SP.PRISTINE_ARMS else "patched"
    if st.get("engine_tree") != want or run.get("engine_tree") != want:
        why.append(f"TREE_MISMATCH:{arm}:{st.get('engine_tree')}")
    argv = st.get("argv") or []
    if want == "pristine":
        if st.get("engine_tree_manifest_sha256") != SP.PRISTINE_REF_SHA256 or st.get("engine_tree_ref_sha256") != SP.PRISTINE_REF_SHA256:
            why.append(f"PRISTINE_MANIFEST:{arm}")
        if st.get("engine_tree_ref_overridden") is not False:
            why.append(f"PRISTINE_REF_OVERRIDDEN:{arm}")
        if st.get("engine_tree_markers_found") != []:
            why.append(f"PRISTINE_MARKERS:{arm}")
        if any(x.split("=", 1)[0] == "--enable-pdmux" for x in argv):
            why.append(f"PRISTINE_PDMUX_ARGV:{arm}")
    else:
        if not set(SP.REQUIRED_PATCHED_MARKERS) <= set(st.get("engine_tree_markers_found") or []):
            why.append(f"PATCHED_MARKERS:{arm}")
    return why


def patched_manifest_failures(job):
    shas = {b.get("tree_stamp", {}).get("engine_tree_manifest_sha256") for b in job["boots"].values()
            if b.get("engine_tree") == "patched" and b.get("status") == "ok"}
    return [] if len(shas) <= 1 else [f"PATCHED_MANIFEST_DIFFERS_IN_JOB:{len(shas)}"]


def vllm_stamp_failures(job):
    """All vLLM boots of one Job (smoke, measurement) carry the same image / interpreter / backend / CUDAGraphMode stamp."""
    keys = {tuple((k, json.dumps((b.get("tree_stamp") or {}).get(k), sort_keys=True)) for k in SP.VL_STAMP_KEYS)
            for b in job["boots"].values() if b.get("engine") == "vllm" and b.get("status") == "ok"}
    return [] if len(keys) <= 1 else [f"VLLM_STAMP_DIFFERS_IN_JOB:{len(keys)}"]


def stamp_failures(runs):
    why, st = [], set()
    for r in runs:
        j, g = r.get("job_name"), r.get("gpu_uuid")
        if not j or not g or g == "UNKNOWN":
            why.append(f"STAMP_MISSING:{r['run_id']}")
        else:
            st.add((j, g))
    if len(st) > 1:
        why.append("CROSS_JOB_OR_GPU:" + ",".join(sorted(f"{j}|{g}" for j, g in st)))
    return why


# ---- definition smoke (§5-0) ---------------------------------------------------------------------------------------------
def smoke_verdict(j1):
    """§5-0 (a), PREREG rev2 wording: per engine, the offsets (client TTFT - server TTFT) of the sequential requests must be a small
    constant -- their standard deviation <= 5 ms (CLIENT_DEF_TOL_MS).  The two engines' constants need NOT be equal (the server-side
    start points differ); rev1 code required |median_SG - median_VL| <= 5 ms, which the text did not register (audit r1 D) -- removed."""
    if j1 is None:
        return {"status": "CLIENT_DEF_UNCHECKED", "why": "no Job 1"}
    med = {}
    for r in j1["runs"]:
        if r.get("kind") == "smoke":
            d = [x["client_ttft_ms"] - x["server_ttft_ms"] for x in r["rows"] if x.get("ok")]
            med[r["engine"]] = {"median_ms": statistics.median(d) if d else None, "sd_ms": statistics.pstdev(d) if len(d) >= 2 else None,
                                "n_ok": len(d), "n": len(r["rows"])}
    if not {"sglang", "vllm"} <= set(med):
        return {"status": "CLIENT_DEF_UNCHECKED", "why": f"smoke engines present: {sorted(med)}", "per_engine": med}
    if any(med[e]["sd_ms"] is None for e in ("sglang", "vllm")):
        return {"status": "CLIENT_DEF_MISMATCH", "why": "fewer than 2 usable smoke requests", "per_engine": med}
    st = "CLIENT_DEF_OK" if all(med[e]["sd_ms"] <= SP.CLIENT_DEF_TOL_MS for e in ("sglang", "vllm")) else "CLIENT_DEF_MISMATCH"
    return {"status": st, "diff_ms": med["sglang"]["median_ms"] - med["vllm"]["median_ms"], "per_engine": med}


def run_ttft_offset(run):
    """(b) in-load offset of one run = mean client TTFT - mean server TTFT, the server mean being delta sum / delta count of the TTFT
    histogram scraped right before and after the run.  The count delta must equal the number of requests with a first token."""
    m = run.get("server_ttft_metric") or {}
    b, a = m.get("before") or {}, m.get("after") or {}
    tt = [q["ttft_ms"] for s in run["segments"] for q in s["requests"] if q.get("ttft_ms") is not None]
    if None in (b.get("sum_s"), b.get("count"), a.get("sum_s"), a.get("count")):
        return None, "NO_METRIC"
    dc = a["count"] - b["count"]
    if not tt or dc != len(tt):
        return None, f"COUNT_MISMATCH:{dc}!={len(tt)}"
    return statistics.fmean(tt) - (a["sum_s"] - b["sum_s"]) / dc * 1000.0, "OK"


def offset_shift(load_ms, smoke_ms):
    d = abs(load_ms - smoke_ms)
    return d > SP.TTFT_OFFSET_SHIFT_MS


def ttft_offset_verdict(smoke, j2):
    """§5-0 (b): compare the in-load offsets of J2 (SGLang: the INSTR_METRICS run; vLLM: every VL-stock R1 measurement run) with the smoke
    offset of the same engine.  -> {status: TTFT_OFFSET_OK | TTFT_OFFSET_SHIFT | TTFT_OFFSET_UNCHECKED, rows}."""
    per = (smoke or {}).get("per_engine") or {}
    if j2 is None:
        return {"status": "TTFT_OFFSET_UNCHECKED", "why": "no J2", "rows": []}
    rows = []
    for r in j2["runs"]:
        if r.get("cell") == "INSTR_METRICS":
            eng = "sglang"
        elif r.get("arm") == "VL-stock" and r.get("cell") == SP.METRICS_CELL and r.get("boot_role") == "meas":
            eng = "vllm"
        else:
            continue
        off, st = run_ttft_offset(r)
        sm = (per.get(eng) or {}).get("median_ms")
        rows.append({"run_id": r["run_id"], "engine": eng, "load_offset_ms": off, "smoke_offset_ms": sm, "status": st,
                     "shift": None if off is None or sm is None else offset_shift(off, sm)})
    engines = {x["engine"] for x in rows}
    if any(x["shift"] for x in rows):
        return {"status": "TTFT_OFFSET_SHIFT", "rows": rows}
    if engines != {"sglang", "vllm"} or any(x["shift"] is None for x in rows):
        return {"status": "TTFT_OFFSET_UNCHECKED", "why": f"engines {sorted(engines)}; unusable rows "
                f"{[x['run_id'] for x in rows if x['shift'] is None]}", "rows": rows}
    return {"status": "TTFT_OFFSET_OK", "rows": rows}


# ---- substrate (refcell) -------------------------------------------------------------------------------------------------
def substrate(job, j1, ttft_slo=SP.W1_TTFT_MS):
    refs = [r for r in job["runs"] if r.get("cell") == "REFCELL"]
    base = None
    if j1 is not None:
        for r in refs:
            m = [x for x in j1["runs"] if x["arm"] == SP.REFCELL_ARM and x.get("trace_hash") == r["trace_hash"]]
            if m:
                base = attainment(m[0], ttft_slo)
                break
    rows = {r["spec"].get("tag"): attainment(r, ttft_slo) for r in refs}
    if base is None:
        return {"tag": "SUBSTRATE_UNCHECKED", "refcells": rows, "job1_baseline": None}
    shift = any(v is not None and abs(v - base) >= SP.SUBSTRATE_SHIFT_PP for v in rows.values())
    return {"tag": "SUBSTRATE_SHIFT" if shift else "SUBSTRATE_OK", "refcells": rows, "job1_baseline": base}


# ---- cells / pairs ---------------------------------------------------------------------------------------------------------
def cell_runs(job, cell):
    out = {}
    for r in job["runs"]:
        if r.get("cell") == cell and r.get("boot_role") == "meas":
            out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def phases_of(cell):
    return ("W3",) if cell == "R3" else ("W1",)


OOB_CELLS = ("R2", "R3")                     # the 80 % steady-state cells (W1, W3); rev1 has no change trace


def out_of_band(arms_runs, ttft_slo, phase):
    att = {a: attainment_pooled(v, ttft_slo, phase) for a, v in arms_runs.items()}
    att = {a: x for a, x in att.items() if x is not None}
    if not att:
        return None, att
    oob = all(x > SP.OUT_OF_BAND_ALL_ABOVE for x in att.values()) or max(att.values()) < SP.OUT_OF_BAND_BEST_BELOW
    return oob, att


def attainment_pooled(runs, ttft_slo, phase):
    reqs = [q for r in runs.values() for s in r["segments"] if s["phase"] == phase for q in s["requests"]]
    return sum(1 for q in reqs if V.req_good(q, ttft_slo, SP.ITL_MS)) / len(reqs) if reqs else None


def pair_validity(a, b, arms, job, ttft_slo, cell, smoke, placebo_bad, offset=None):
    why = []
    ra, rb = arms[a], arms[b]
    for s in sorted(set(ra) & set(rb)):
        if ra[s]["trace_hash"] != rb[s]["trace_hash"]:
            why.append(f"TRACE_HASH_DIFFERS:s{s}")
    for x in (a, b):
        for ph in phases_of(cell):
            reqs = V.arm_requests(list(arms[x].values()), ph)
            if reqs and V.is_cliff(reqs, ttft_slo):
                why.append(f"CLIFF_ADJACENT:{x}:{ph}")
        if any(q.get("error") for r in arms[x].values() for s in r["segments"] for q in s["requests"]):
            why.append(f"REQUEST_FAILURES:{x}")
        for r in arms[x].values():
            why += tree_failures(r, job)
    why += stamp_failures(list(ra.values()) + list(rb.values()))
    why += patched_manifest_failures(job)
    if set(ra) != set(SP.EVAL_SEEDS) or set(rb) != set(SP.EVAL_SEEDS):
        why.append(f"SEEDS_MISMATCH:{a}={sorted(ra)}:{b}={sorted(rb)}")
    if a in SP.ENGINE_PAIR_ARMS or b in SP.ENGINE_PAIR_ARMS:
        if smoke.get("status") != "CLIENT_DEF_OK":
            why.append(f"{smoke.get('status')}")
        ost = (offset or {}).get("status", "TTFT_OFFSET_UNCHECKED")
        if ost != "TTFT_OFFSET_OK":
            why.append(ost)
        why += vllm_stamp_failures(job)
    return sorted(set(why))


def run_server_args(run, job):
    b = (job.get("boots") or {}).get(run.get("boot_id")) or {}
    return run.get("server_args") or b.get("server_args") or {}


def analyze_cell(job, cell, j1, smoke, ttft_slo, offset=None):
    arms = cell_runs(job, cell)
    out = {"cell": cell, "arms": sorted(arms), "pairs": {}, "goodput_mean": {}}
    for a, v in arms.items():
        out["goodput_mean"][a] = statistics.fmean(goodput(r, ttft_slo) for r in v.values())
    pl = paired(arms[SP.PLACEBO[0]], arms[SP.PLACEBO[1]], ttft_slo) if all(x in arms for x in SP.PLACEBO) else None
    out["placebo"] = pl
    pbad = placebo_invalid(pl)
    out["placebo_invalid"] = pbad
    if pl and not math.isnan(pl["sd"]) and pl["sd"] >= SP.PLACEBO_SD_EQUIV_MAX:
        out["placebo_note"] = "placebo sd >= 0.0186: EQUIV not reachable at n=4 (UNRESOLVED_INTERVAL is the normal result)"
    oob, cell_att = out_of_band(arms, ttft_slo, phases_of(cell)[0]) if cell in OOB_CELLS else (None, {})
    out["band_attainment"], out["cell_out_of_band"] = cell_att, oob
    out["ttft_offset"] = (offset or {}).get("status", "TTFT_OFFSET_UNCHECKED")
    sub = substrate(job, j1)
    out["substrate"] = sub
    for q, a, b, tok in SP.PAIRS:
        key = f"{q}:{a}_vs_{b}"
        if a not in arms or b not in arms:
            out["pairs"][key] = {"label": "NOT_RUN", "why": sorted({a, b} - set(arms))}
            continue
        p = paired(arms[a], arms[b], ttft_slo)
        raw = label((p["mean"], p["lo"], p["hi"], p["n"]), tok)
        why = pair_validity(a, b, arms, job, ttft_slo, cell, smoke, pbad, offset)
        engine_pair = a in SP.ENGINE_PAIR_ARMS or b in SP.ENGINE_PAIR_ARMS
        sat = both_saturated(arms[a], arms[b], ttft_slo)
        if pbad:
            final = "BOOT_NOISE"
        elif engine_pair and "TTFT_OFFSET_SHIFT" in why:
            final = "INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT"
        elif why:
            final = "INVALID"
        elif sat:
            final = "BOTH_SATURATED"
        else:
            final = raw
        sign_only = q in SP.SIGN_ONLY_QUESTIONS
        if sign_only and final == raw:          # rev3: an SGLang-default arm sits above its own capacity -> sign only, no magnitude
            final = sign_label((p["mean"], p["lo"], p["hi"], p["n"]), tok)
        tags = ["TTFT_OFFSET_SHIFT"] if engine_pair and "TTFT_OFFSET_SHIFT" in why else []
        if sign_only:
            tags.append("MAGNITUDE_NOT_CITABLE")
        if oob:
            tags.append("CELL_OUT_OF_BAND")
        if sub["tag"] != "SUBSTRATE_OK":
            tags.append(sub["tag"])
        mix_bad = False
        if a == "SG-mix":
            fired = [RU.mix_fired_from_server_args(run_server_args(r, job)) for r in arms[a].values()]
            if not fired or not all(fired):
                mix_bad = True
                tags.append("MIX_NOT_FIRED")
                final = "MIX_NOT_FIRED" if final not in ("BOOT_NOISE", "INVALID") else final
        lat = {}
        if q in SP.LATENCY_PAIRS:
            for m in SP.LATENCY_METRICS:
                pm = paired_latency(arms[a], arms[b], m)
                lraw = label((pm["mean"], pm["lo"], pm["hi"], pm["n"]), tok) + f"_{m}"     # ABOVE = A has the LARGER latency
                lfin = "BOOT_NOISE" if pbad else "INVALID" if why else "MIX_NOT_FIRED" if mix_bad else lraw
                lat[m] = dict(pm, raw_label=lraw, label=lfin)
        if (a in SP.ENGINE_PAIR_ARMS or b in SP.ENGINE_PAIR_ARMS) and not math.isnan(p["sd"]) and p["sd"] > SP.ENGINE_PAIR_SD_MAX:
            tags.append("RERUN_SEED8_REGISTERED")
        out["pairs"][key] = dict(p, raw_label=raw, label=final, validity=why, tags=tags, question=q, both_saturated=sat, sign_only=sign_only,
                                 seed_attainment={a: seed_attainments(arms[a], ttft_slo), b: seed_attainments(arms[b], ttft_slo)},
                                 latency=lat)
    out["mixing_opportunities"] = {x: {s: r.get("mixing_opportunities") for s, r in arms[x].items()} for x in SP.MIX_OPPORTUNITY_ARMS
                                   if x in arms}
    out["mix_server_args"] = {x: {s: run_server_args(r, job) for s, r in arms[x].items()} for x in SP.MIX_OPPORTUNITY_ARMS if x in arms}
    out["gate5"] = {a: V.gate5_row(list(v.values()), ttft_slo) for a, v in arms.items()}
    return out


# ---- instrumentation: r_ser / r_mix (SG) and r_span (PM) ------------------------------------------------------------------
def probe_window(records, t0_epoch, t1_epoch):
    """Client load window (epoch) -> probe host clock, through the probe boot record (t_realtime, t_host) -- PREREG §5-1."""
    boots = [r for r in records if r.get("event") == "decode_probe_boot"]
    if not boots or boots[0].get("t_realtime") is None or boots[0].get("t_host") is None:
        raise ValueError("probe boot record without (t_realtime, t_host): cannot map the load window")
    b = boots[0]
    return (b["t_host"] + (t0_epoch - b["t_realtime"]), b["t_host"] + (t1_epoch - b["t_realtime"]))


def _is_single(g):
    """A gap whose same-stream extend work is exactly ONE request (one EXTEND forward, n_reqs = 1) and that yields an r_ser."""
    return g.get("r_ser") is not None and g.get("extend_n_reqs") == 1


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def _window_stats(dp, records, win):
    """pacing_ratios on ONE window (probe clock) -> r_ser / r_mix distribution, single-request spans and the {512, 2k} bins of them."""
    res = dp.pacing_ratios(records, min_bs=dp.PACING_MIN_BS, t_window=win)
    single = [g for g in res["gaps"] if _is_single(g)]
    by_L = {}
    for g in single:
        for L in SP.Q6_L_GRID:
            if abs(g["extend_tokens"] - L) <= SP.Q6_L_BAND * L:
                by_L.setdefault(L, []).append(g)
    other = [g for g in res["gaps"] if (g.get("other_stream_ms") or 0.0) > 0.0]
    return {"window_probe_clock": win, "n_r_ser": len(res["r_ser"]), "n_r_mix": len(res["r_mix"]),
            "r_ser_p50": statistics.median(res["r_ser"]) if res["r_ser"] else None,
            "r_ser_p10": _q(res["r_ser"], 0.1), "r_ser_p90": _q(res["r_ser"], 0.9),
            "r_mix_p50": statistics.median(res["r_mix"]) if res["r_mix"] else None,
            "n_single_request_spans": len(single), "n_other_stream_gaps": len(other),
            "by_L": {L: {"n": len(v), "r_ser_p50": statistics.median(x["r_ser"] for x in v), "bs_p50": statistics.median(x["bs"] for x in v)}
                     for L, v in sorted(by_L.items()) if v},
            "excluded": {k: res[k] for k in ("n_first", "n_incomplete", "n_trailing_extend", "n_fallback")}}


def r_ser_from_instr(records, run):
    """§5-1: the W1 load window and the synthetic single-request window are reduced SEPARATELY (never one pooled window).
    W1 window = [t_load_start, t_load_end] (r_ser / r_mix distribution, descriptive); synthetic window = [t_synth_start, t_synth_end]
    (the P-4 source: single-request 2k spans; 512 descriptive).  Q6_OK iff no other-stream extend in either window, the synthetic window
    holds >= 10 single-request spans for every L of the grid, and the two windows together hold >= 30 single-request spans (counted per
    window, reported per window)."""
    dp = decode_probe_module()
    w1 = _window_stats(dp, records, probe_window(records, run["t_load_start_epoch"], run["t_load_end_epoch"]))
    syn = None
    if run.get("t_synth_start_epoch") is not None and run.get("t_synth_end_epoch") is not None:
        syn = _window_stats(dp, records, probe_window(records, run["t_synth_start_epoch"], run["t_synth_end_epoch"]))
    n_other = w1["n_other_stream_gaps"] + (syn["n_other_stream_gaps"] if syn else 0)
    n_single = {"W1": w1["n_single_request_spans"], "SYNTH": syn["n_single_request_spans"] if syn else 0}
    syn_L = (syn or {}).get("by_L") or {}
    if n_other:
        q6 = "Q6_INVALID:OTHER_STREAM_EXTEND"          # rev2 (audit r1): a non-stock stream shape, fail-closed
    elif syn is None:
        q6 = "Q6_NO_SYNTH_WINDOW"
    elif sum(n_single.values()) >= SP.Q6_MIN_SPANS and all(syn_L.get(L, {}).get("n", 0) >= SP.Q6_MIN_PER_L for L in SP.Q6_L_GRID):
        q6 = "Q6_OK"
    else:
        q6 = "Q6_FEW_SPANS"
    return {"q6_status": q6, "n_other_stream_gaps": n_other, "n_single_request_spans": n_single, "windows": {"W1": w1, "SYNTH": syn}}


def pc_reduce_windowed(tel_lines, probe_lines, window):
    """implfix pc_reduce (r_span = max chunk_gpu_ms of a span / median decode_fwd_gpu_ms of its decode-paired chunks) restricted to records
    whose server-clock time lies in `window` (prefill_chunk: t_issue_start, else timestamp_monotonic_s; decode_forward: t_host_start)."""
    lo, hi = window
    inw = lambda t: t is not None and lo <= float(t) <= hi
    dec = {}
    for x in probe_lines:
        if x.get("event") == "decode_forward" and x.get("decode_fwd_gpu_ms") is not None and x.get("decode_iter_seq") is not None \
                and inw(x.get("t_host_start")):
            dec.setdefault(int(x["decode_iter_seq"]), float(x["decode_fwd_gpu_ms"]))
    spans, n_chunks = {}, 0
    for x in tel_lines:
        if x.get("event") != "prefill_chunk" or not inw(x.get("t_issue_start", x.get("timestamp_monotonic_s"))):
            continue
        n_chunks += 1
        g, di = x.get("chunk_gpu_ms"), x.get("decode_iter_seq")
        if g is None or not x.get("decode_launched") or di is None or int(di) not in dec:
            continue
        s = spans.setdefault(x.get("span_seq"), {"g": [], "d": []})
        s["g"].append(float(g))
        s["d"].append(dec[int(di)])
    rs = sorted(max(s["g"]) / statistics.median(s["d"]) for s in spans.values() if statistics.median(s["d"]) > 0)
    q = (lambda p: rs[min(len(rs) - 1, int(p * len(rs)))]) if rs else (lambda p: None)
    return {"n_chunks_in_window": n_chunks, "n_spans": len(rs), "r_span_p50": q(0.5), "r_span_p90": q(0.9),
            "share_backlog": (sum(1 for r in rs if r >= PC_BACKLOG_R) / len(rs)) if rs else None,
            "share_oversplit": (sum(1 for r in rs if r < PC_OVERSPLIT_R) / len(rs)) if rs else None}


def instr_summary(j2):
    out = {}
    if j2 is None:
        return out
    for r in j2["runs"]:
        if r.get("cell") != "INSTR":
            continue
        b = j2["boots"].get(r["boot_id"]) or {}
        d = Path(b.get("dir", ""))
        probe = _lines(d / "decode_probe.jsonl")
        try:
            if r["arm"] == "INSTR-SG-stock":
                out["SG"] = r_ser_from_instr(probe, r)
            elif r["arm"] == "INSTR-PM-ref":
                win = probe_window(probe, r["t_load_start_epoch"], r["t_load_end_epoch"])
                out["PM"] = pc_reduce_windowed(_lines(d / "tel.jsonl"), probe, win)
        except (ValueError, KeyError) as e:
            out[r["arm"]] = {"error": str(e)}
    return out


# ---- predictions P-1 .. P-5 (§8; scored, no overall verdict) ---------------------------------------------------------------
def forecast_r_ser(L, bs, ctx=2048, path=FORECAST_CSV):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    bss = sorted({int(r["bs"]) for r in rows})
    b = min(bss, key=lambda x: abs(x - bs))
    for r in rows:
        if int(r["L"]) == L and int(r["bs"]) == b and int(r["ctx"]) == ctx:
            return float(r["r_ser_hat"]), b
    return None, b


def score_predictions(add, cells, instr):
    P = {}
    lb = (add or {}).get("lambda_star_burst", {})
    sg, vl = lb.get("SG-stock"), lb.get("VL-stock")
    P["P-1"] = {"value": sg, "band": [6, 10], "score": None if sg is None else ("HIT" if 6 <= sg <= 10 else "MISS")}
    P["P-2"] = {"value": (vl / sg) if sg and vl else None, "band": [0.8, 1.25],
                "score": None if not (sg and vl) else ("HIT" if 0.8 <= vl / sg <= 1.25 else "MISS")}
    r2 = (cells.get("R2") or {}).get("pairs", {}).get("Q2:PM-ref_vs_SG-stock")
    sat3 = bool(r2) and (r2.get("both_saturated") is True or r2.get("label") == "BOTH_SATURATED")
    if sat3:        # rev2.1 (re-audit r2 (a)): saturated pair -> the ln ratio is a drain difference only, no sign score (prediction doc §5)
        sc = "UNSCORABLE"
    elif r2 and str(r2.get("label", "")).split(":")[0] in ("INVALID", "INVALID/UNRESOLVED", "BOOT_NOISE"):
        sc = "UNSCORABLE"                                   # seal fix: an invalid or boot-noise pair carries no sign evidence either
    elif r2 and r2.get("n", 0) >= 4:
        sc = "HIT(+)" if r2["lo"] > 0 else "MISS(-)" if r2["hi"] < 0 else "NEITHER"
    else:
        sc = None
    P["P-3"] = {"value": None if not r2 else r2.get("mean"), "predicted": "+", "score": sc, "label": None if not r2 else r2.get("label")}
    if sat3:
        P["P-3"]["why"] = "BOTH_SATURATED"
    elif sc == "UNSCORABLE":
        P["P-3"]["why"] = "PAIR_INVALID"
    sgi = (instr or {}).get("SG") or {}
    syn = (sgi.get("windows") or {}).get("SYNTH") or {}      # P-4 reads the synthetic single-request window ONLY (§6, 2026-11-02)
    b2 = (syn.get("by_L") or {}).get(2048)
    if str(sgi.get("q6_status", "")).startswith("Q6_INVALID"):
        P["P-4"] = {"value": None, "score": "UNSCORABLE", "why": sgi.get("q6_status")}
    elif not b2 or b2["n"] < SP.Q6_MIN_PER_L:
        P["P-4"] = {"value": None, "n": 0 if not b2 else b2["n"], "score": "UNSCORABLE",
                    "why": f"single-request L~2k spans of the synthetic window n < {SP.Q6_MIN_PER_L} (rev2: no score below n = 10)"}
    else:
        f, bsel = forecast_r_ser(2048, b2["bs_p50"])
        P["P-4"] = {"value": b2["r_ser_p50"], "forecast": f, "forecast_cell": {"L": 2048, "bs": bsel, "ctx": 2048}, "n": b2["n"],
                    "source": "SYNTH", "score": None if f is None else ("HIT" if abs(b2["r_ser_p50"] - f) <= 0.30 * f else "MISS")}
    b5 = (syn.get("by_L") or {}).get(512)
    if b5 and not str(sgi.get("q6_status", "")).startswith("Q6_INVALID"):
        f5, bs5 = forecast_r_ser(512, b5["bs_p50"])
        P["P-4"]["descriptive_512"] = {"value": b5["r_ser_p50"], "n": b5["n"], "forecast": f5,
                                       "forecast_cell": {"L": 512, "bs": bs5, "ctx": 2048}, "score": "NOT_SCORED"}
    v1, w1 = q1_burst(add)                                 # rev3 PREREG §8 row Q1-SG (registered before the J1-down data)
    lo1, hi1 = SP.Q1_SG_BAND
    P["Q1-SG"] = {"value": v1, "band": [lo1, hi1], "source": w1,
                  "score": "UNSCORABLE" if v1 is None else ("HIT" if lo1 <= v1 <= hi1 else "MISS")}
    if v1 is None:
        P["Q1-SG"]["why"] = w1
    g = (cells.get("R2") or {}).get("gate5", {})
    if "SG-mix" in g and "SG-stock" in g:
        a, b = g["SG-mix"]["itl_ms"]["p95"], g["SG-stock"]["itl_ms"]["p95"]
        P["P-5"] = {"value": {"SG-mix": a, "SG-stock": b}, "score": "HIT" if a < b else "MISS"}
    else:
        P["P-5"] = {"value": None, "score": None}
    return P


# ---- "not measured" (auto) -------------------------------------------------------------------------------------------------
PREREG_LOST = ("조율 fused", "Llama-8B", "동적 PD-mux", "best D 재결정", "W3 사다리(SG-stock 외 arm)", "과부하 구간",
               "부하 변화(LO↔HI) 반응 — 변화 trace 미적용(고정 설정 비교, 게이트 2는 컨트롤러 비교 한정; 결정 2026-11-02)",
               "prefix caching ON", "TP>1")
GPU_UNVERIFIED = ("EXTEND 이벤트 실측", "overlap 경로의 단일 forward stream(stream_key 상수)", "EXTEND ON boot 출력 불변", "관측자 비용",
                  "pristine 보존(PRESERVE_STATUS) GPU 이미지", "SG-mix 발화 판정의 server_args 로그 패턴(enable_mixed_chunk=·chunked_prefill_size=)", "vLLM 로그 백엔드/CUDAGraphMode 패턴",
                  "server TTFT 히스토그램 이름(smoke·부하 중 오프셋)", "/opt/vllm-venv boot(CUDA graph 캡처·flashinfer JIT sm80) on A100")


Q_CELL_PAIRS = {"Q2": ("R1", "R2"), "Q3": ("R1", "R2"), "Q4": ("R1", "R2"), "Q5": ("R1", "R2"), "Q8": ("R1", "R2"), "Q7": ("R3",)}
Q7_KEYS = ("Q2:PM-ref_vs_SG-stock", "Q3:VL-stock_vs_SG-stock")     # Q7 = the signs of Q2 / Q3 on W3 (R3 cell)
LABEL_NOT_MEASURED = ("NOT_RUN", "INVALID", "BOOT_NOISE", "MIX_NOT_FIRED", "INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT")


def q1_burst(add):
    """rev3: SG-stock burst lambda* = the first-Job ladder value if bracketed there, else the J1-down value (same pick rule over both).
    -> (value, source-or-reason)."""
    add = add or {}
    v = (add.get("lambda_star_burst") or {}).get(SP.Q1_ARM)
    if v is not None:
        return v, "J1 ladder"
    st1 = ((add.get("ladder") or {}).get(SP.Q1_ARM) or {}).get("status", "Job 1 없음")
    if "ladder_down" not in add:
        return None, f"J1 {st1}; 아래쪽 사다리(J1-down) raw 없음"
    d = add["ladder_down"].get(SP.Q1_ARM) or {}
    if d.get("lambda_star") is None:
        return None, f"J1 {st1}; J1-down {d.get('status', 'NOT_RUN')}"
    return d["lambda_star"], "J1 + J1-down ladder"


def question_rows(add, cells, instr):
    """rev2 (audit r1 ⑦): one row per question Q1-Q8, ALWAYS (a question with no data is NOT_MEASURED with its reason)."""
    rows = {}
    ls = ((add or {}).get("lambda_star_sustained") or {}).get(SP.Q1_ARM) or {}
    (v_b, src), v_s = q1_burst(add), ls.get("lambda_star_sustained")
    why = []
    if v_b is None:
        why.append(f"{SP.Q1_ARM} 버스트 λ* 없음({src})")
    if v_s is None:
        why.append(f"{SP.Q1_ARM} 지속 λ* 없음(rev3: SGLang 기본 구성 3 arm은 지속 셀을 돌리지 않는다 — 기준 용량에서 제외)")
    rows["Q1"] = {"status": "MEASURED" if not why else ("PARTIAL" if (v_b is not None or v_s is not None) else "NOT_MEASURED"),
                  "detail": {"burst": v_b, "burst_source": src if v_b is not None else None, "sustained": v_s}, "why": why}
    for q, cs in Q_CELL_PAIRS.items():
        got, miss = {}, []
        for c in cs:
            allp = (cells.get(c) or {}).get("pairs") or {}
            ps = {k: v for k, v in allp.items() if (k in Q7_KEYS if q == "Q7" else k.startswith(q + ":"))}
            if not ps:
                miss.append(f"{c}: 셀 없음")
                continue
            for k, v in ps.items():
                got[f"{c} {k}"] = v.get("label")
                if v.get("label") in LABEL_NOT_MEASURED:
                    miss.append(f"{c} {k}: {v.get('label')}")
        ok = [k for k, v in got.items() if v not in LABEL_NOT_MEASURED]
        rows[q] = {"status": "MEASURED" if ok and not miss else "PARTIAL" if ok else "NOT_MEASURED", "detail": got, "why": miss}
    sg = (instr or {}).get("SG")
    if not sg:
        rows["Q6"] = {"status": "NOT_MEASURED", "detail": {}, "why": ["SG 계측 요약 없음"]}
    else:
        st = sg.get("q6_status") or sg.get("error")
        wins = sg.get("windows") or {}
        rows["Q6"] = {"status": "MEASURED" if st == "Q6_OK" else "NOT_MEASURED", "detail": {"q6_status": st,
                      "by_L_n": {w: {L: v["n"] for L, v in ((wins.get(w) or {}).get("by_L") or {}).items()} for w in ("W1", "SYNTH")}},
                      "why": [] if st == "Q6_OK" else [str(st)]}
    return {q: rows[q] for q in SP.QUESTIONS}


def boot_status_rows(name, j):
    """Every planned boot whose DONE.json status is not "done" (skip / correctness failure / image refusal / R3_SKIPPED_BUDGET), and every
    planned boot without a DONE.json at all (killed by the hard cap or never reached)."""
    out = []
    done = j.get("done") or {}
    planned = [b["boot_id"] for b in ((j.get("plan") or {}).get("boots") or [])]
    for bid in planned or sorted(set(done) | set(j["boots"])):
        d = done.get(bid)
        if d is None:
            b = j["boots"].get(bid)
            out.append(f"{name}: boot {bid} DONE.json 없음" + (f" (boot status {b.get('status')})" if b else " (시작 안 됨 / hard cap)"))
        elif d.get("status") != "done":
            out.append(f"{name}: boot {bid} {d.get('status')}" + (f" — {d.get('why')}" if d.get("why") else ""))
    return out


def ladder_down_rows(add, j1down_present):
    """rev3: the downward ladder status of every SGLang-default arm (Q1 only)."""
    if not j1down_present and "ladder_down" not in (add or {}):
        return ["J1-down(아래쪽 사다리): raw 없음 → Q1 버스트 λ*는 첫 번째 Job 사다리 결과만"]
    out = []
    for a, v in sorted(((add or {}).get("ladder_down") or {}).items()):
        if v.get("lambda_star") is None:
            out.append(f"J1-down {a}: {v.get('status')} (값 없음)")
    return out


def ref_capacity_rows(add):
    if add is None:
        return ["기준 용량(λ*_ref): addendum 없음(Job 1 raw 없음)"]
    if "lambda_star_ref" not in add:
        return ["기준 용량(λ*_ref): 지속 Job raw 없음 → 미정"]
    out = []
    if add.get("lambda_star_ref") is None:
        out.append("기준 용량(λ*_ref) None: 지속 λ* 없는 arm " + str(add.get("lambda_star_ref_why_none") or []) +
                   (", 지속 셀 없는 arm " + str(add.get("lambda_star_ref_missing_arms")) if add.get("lambda_star_ref_missing_arms") else ""))
    elif add.get("lambda_star_ref_missing_arms"):
        out.append(f"기준 용량(λ*_ref) 범위 축소: {add['lambda_star_ref_missing_arms']} 지속 셀 없음 (min은 {add.get('lambda_star_ref_scope')}만)")
    return out


def not_measured(jobs, cells, smoke, instr, offset=None, add=None):
    out = [f"{q} {r['status']}: " + ("; ".join(r["why"]) if r["why"] else "—") for q, r in question_rows(add, cells, instr).items()]
    out += [f"PREREG §10: {x}" for x in PREREG_LOST]
    out += ladder_down_rows(add, jobs.get("J1-down") is not None)
    out += ref_capacity_rows(add)
    out.append(f"rev3: 기준 용량 = {list(SP.REF_ARMS)}의 지속 용량 최소; {list(SP.LADDER_DOWN_ARMS)}는 제외(첫 번째 Job 사다리 바닥 실패 — "
               "ITL 축 절벽, 도착당 ~42 ms 정지) — 이 3 arm의 지속 용량은 측정하지 않는다")
    for cname, c in cells.items():
        for k, v in c.get("pairs", {}).items():
            if v.get("sign_only"):
                out.append(f"{cname} {k}: rev3 부호·라벨 + 지연 공동 지표만 (goodput % 크기 인용 금지)")
    for name, j in jobs.items():
        if j is None:
            out.append(f"{name}: Job raw tree absent")
            continue
        for p in (j["plan"] or {}).get("pending", []):
            out.append(f"{name}: {p} pending")
        out += boot_status_rows(name, j)
    for cname, c in cells.items():
        for k, v in c.get("pairs", {}).items():
            if v.get("label") in LABEL_NOT_MEASURED:
                out.append(f"{cname} {k}: {v['label']} {v.get('why') or v.get('validity') or ''}")
            if v.get("label") == "BOTH_SATURATED":
                out.append(f"{cname} {k}: BOTH_SATURATED (goodput 라벨 없음; 두 arm 모두 4 seed attainment >= {SP.SATURATED_ATTAIN})")
            if "CELL_OUT_OF_BAND" in v.get("tags", []):
                out.append(f"{cname} {k}: CELL_OUT_OF_BAND (label only, magnitude not citable)")
    if smoke.get("status") != "CLIENT_DEF_OK":
        out.append(f"§5-0 definition smoke: {smoke.get('status')} -> every engine pair INVALID")
    if (offset or {}).get("status") != "TTFT_OFFSET_OK":
        out.append(f"§5-0 (b) in-load TTFT offset: {(offset or {}).get('status', 'TTFT_OFFSET_UNCHECKED')} -> every engine pair INVALID")
    if "SG" not in instr:
        out.append("Q6 r_ser/r_mix: no SG instrumentation summary")
    elif instr["SG"].get("q6_status") != "Q6_OK":
        out.append(f"Q6: {instr['SG'].get('q6_status')} (single-request spans {instr['SG'].get('n_single_request_spans')})")
    out.append("§5-2 VL-stock profiler r_mix: not implemented (runner has no profiler window)")
    out.append(f"D-1: {SP.INSTR_TREE_NOTE}")
    out += [f"GPU 미검증: {x}" for x in GPU_UNVERIFIED]
    return out


def render_not_measured(items):
    return "## 측정하지 않은 것 (sb_analyze 자동 생성)\n\n" + "\n".join(f"- {x}" for x in items) + "\n"


EXPECTED_KIND = {"J1": "LADDER", "J1-down": "LADDER_DOWN", "J1b": "SUSTAIN", "J2": "STEADY", "J3": "W3"}


def kind_problems(jobs):
    """Each raw tree's plan kind must be the rev1 kind of its Job (J3 = W3 only; a rev0 TRACE_W3 tree is refused)."""
    return {n: f"plan kind {(j.get('plan') or {}).get('kind')} != {EXPECTED_KIND[n]}" for n, j in jobs.items()
            if j is not None and (j.get("plan") or {}).get("kind") != EXPECTED_KIND[n]}


def report(j1, j1b, j2, j3, add=None, j1down=None):
    jobs = {"J1": j1, "J1-down": j1down, "J1b": j1b, "J2": j2, "J3": j3}
    kp = kind_problems(jobs)
    j2u, j3u = (None if "J2" in kp else j2), (None if "J3" in kp else j3)
    smoke = smoke_verdict(j1)
    offset = ttft_offset_verdict(smoke, j2u)
    cells = {}
    if j2u is not None:
        for c in ("R1", "R2"):
            if cell_runs(j2u, c):
                cells[c] = analyze_cell(j2u, c, j1, smoke, SP.W1_TTFT_MS, offset)
    if j3u is not None and j3u.get("floor") and cell_runs(j3u, "R3"):
        cells["R3"] = analyze_cell(j3u, "R3", j1, smoke, SP.W3_TTFT_K * j3u["floor"]["floor_ms"], offset)
    instr = instr_summary(j2u)
    j1du = None if "J1-down" in kp else j1down
    add = add or (build_addendum(j1, None, j1b, j1down=j1du) if j1 else None)
    preds = score_predictions(add, cells, instr)
    nm = [f"{n}: REFUSED ({w})" for n, w in sorted(kp.items())] + not_measured(jobs, cells, smoke, instr, offset, add)
    return {"smoke": smoke, "ttft_offset": offset, "kind_problems": kp, "addendum": add, "cells": cells, "instr": instr,
            "questions": question_rows(add, cells, instr),
            "predictions": preds, "not_measured": nm, "not_measured_md": render_not_measured(nm)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("addendum")
    p.add_argument("--j1", required=True)
    p.add_argument("--j1ext")
    p.add_argument("--j1down")
    p.add_argument("--j1b")
    p.add_argument("--w1-lengths-sha256")
    p.add_argument("--out", required=True)
    q = sub.add_parser("report")
    for k in ("--j1", "--j1down", "--j1b", "--j2", "--j3", "--out", "--addendum"):
        q.add_argument(k)
    a = ap.parse_args(argv)
    if a.cmd == "addendum":
        add = build_addendum(load_job(a.j1), load_job(a.j1ext) if a.j1ext else None, load_job(a.j1b) if a.j1b else None,
                             a.w1_lengths_sha256, load_job(a.j1down) if a.j1down else None)
        Path(a.out).write_text(json.dumps(add, indent=1, sort_keys=True, default=str) + "\n")
        print(json.dumps({"lambda_star_burst": add["lambda_star_burst"], "refcell_arm": add["refcell_arm"], "refcell_rate": add["refcell_rate"],
                          "lambda_star_burst_down": add.get("lambda_star_burst_down"), "lambda_star_ref": add.get("lambda_star_ref")}))
        return 0
    ld = lambda x: load_job(x) if x else None
    res = report(ld(a.j1), ld(a.j1b), ld(a.j2), ld(a.j3), json.load(open(a.addendum)) if a.addendum else None, ld(a.j1down))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    (Path(a.out).write_text(txt + "\n") if a.out else print(txt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
