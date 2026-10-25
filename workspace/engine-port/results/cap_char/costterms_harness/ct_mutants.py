#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- mutation ledger (lesson 53: a check is only evidence if a reverted / broken variant
FAILS it).  Each mutant re-creates one PREREG §11-2 / §12 candidate defect by patching ONE harness function or
constant; each oracle encodes the expected behaviour INDEPENDENTLY of the patched code (fake-engine ground truth,
raw log text, the PREREG's own pre-data table).  tests/test_costterms_harness.py asserts: oracle passes on the
unmutated harness and fails under every mutant.

stage: 'unit' (synthetic inputs), 'analysis' (re-analyse a stored virtual run), 'run' (re-run a small virtual plan).
Stage 2 (M27-M38): the three candidates deferred in stage 1 (TP2 anchor cap, stall median in the co-residence window,
sigma' x sigma) + T3 (sigma from client TTFT, sigma' / rho paired across boots, n_it_obs by conversion, T3 window from
the first decode, no ramp / co budget in the decode O) + Job entry (smoke marker, resume, raw data exported).
Decision 30 (M39-M41b): long boot mid-Job, earlier results deleted on SERVER_DEATH, 96k tried at context 73,728 (plan
and run time).
Decision 33 (M42a-M48): the five restored §11-4 points -- J-C bs 48 row (M42a/b), J-A T1 context 131,072 so (8, 64k) runs
(M43a-c), J-H1 Llama + real-a4 long boots (M44a-e), J-C cap-32 anchor with its own O (M45a-d), J-C TP2 108u long boot
(M46a-c), the generalised "long boots = contiguous last block" plan rule (M47), a death in one long boot must not skip the
next (M48).
"""
from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
from pathlib import Path
from unittest import mock

import ct_spec as S
import ct_extract as X
import ct_fit as F
import ct_runner as R
import ct_client as C

DEFERRED = {}          # stage 1's three deferred candidates are M27-M29

# ---------------------------------------------------------------- scenarios (virtual runs; truth = fake knobs)
FAST_PF = {"pf_u": 0.5, "pf_v": 1e-5, "pf_w4": 0.0}
SCENARIOS = {
    "JH1": {"job": "J-H1", "truth": {}},
    "JH1F": {"job": "J-H1", "truth": {"weight_dependent": 0.05, "client_drift": 0.05, "client_drift_bs": 32,
                                      "i4_unresolved_every": 40}, "boots": ["jh1_dummy_a4", "jh1_real_a4"]},
    "JAF": {"job": "J-A", "truth": dict(FAST_PF, realize={"34": 36, "24": 16})},
    "JB2": {"job": "J-B2", "truth": {}},
    "CG": {"job": "J-TEST", "truth": {"capture_max": 48}},        # plan: cg_plan() (one cap-128 boot)
    "JA0": {"job": "J-A", "truth": {}, "boots": ["ja1_D44", "ja1_108u", "ja1_108v"]},   # default prefill: A-1 ramp
    "T3S": {"job": "J-A", "truth": {"stall_every": 13, "pf_coex": 0.004},       # stalls 6 x step, sigma' > 1
            "boots": ["ja1_t3_D44", "ja1_t3_D24"]},
    "JC": {"job": "J-C", "truth": dict(FAST_PF), "boots": ["jc1_tp2_D44", "jc1_n1_D44", "jc1_n1c32_D44", "jc_tp2_t4"]},
    # decision 30 long points: the server dies on the 126,976 input (after 65,536 and 98,304 were served)
    "LONGDEATH": {"job": "J-A", "truth": dict(FAST_PF, die_on_input_ge=124 * 1024),
                  "boots": ["ja1_off_108v", "ja1_long_108v"]},
    # the server does not honour --context-length 131,072 and logs / enforces 73,728
    "LONGCTX": {"job": "J-A", "truth": dict(FAST_PF, context_clamp=73728), "boots": ["ja1_long_108v"]},
    # decision 33: both TP2 long boots die on the 126,976 input (each boot has its own server): the second must still run
    "LONGDEATH2": {"job": "J-C", "truth": dict(FAST_PF, die_on_input_ge=124 * 1024),
                   "boots": ["jc_tp2_long_108v", "jc_tp2_long_108u"]},
    # decision 33 L-7: one J-A-style T1 boot at context 131,072 holding the (8, 64k) cell (job name J-X: no J-A cell-set rule)
    "JAL7": {"job": "J-X", "truth": dict(FAST_PF), "plan_fn": lambda: l7_plan()},
}


def l7_plan():
    b = R._boot("J-X", 1, "x1_108v_l7", "T1T2", "nano", "vanilla", S.CAP_MAIN, t1=[(8, 2 * S.K), (8, 64 * S.K)],
                context=S.JA_CONTEXT, o_family="JA")
    return {"job": "J-X", "deferred": [], "boots": [b]}


def plan_for(sc, sub):
    """The plan a run-stage mutant scenario uses (test dispatch)."""
    if sc == "CG":
        return cg_plan()
    if SCENARIOS.get(sc, {}).get("plan_fn"):
        return SCENARIOS[sc]["plan_fn"]()
    return sub_plan(*sub)


def run_scenario(name, out, variant_src, plan=None, truth=None):
    sc = SCENARIOS[name]
    tr = sc["truth"] if truth is None else truth
    if plan is None and sc.get("plan_fn"):
        plan = sc["plan_fn"]()
    if plan is None and sc.get("boots"):
        plan = sub_plan(sc["job"], sc["boots"])
    if Path(out).exists():
        shutil.rmtree(out)
    R.run_job(sc["job"], out, 1, variant_src, Path(out) / "variants",
              make_server=lambda b, a, e: R.VirtualServer(b, a, e, tr), plan=plan)
    return R.analyze(out)


def sub_plan(job, boot_ids):
    p = R.make_plan(job, 1)
    return {"job": p["job"], "deferred": [], "boots": [b for b in p["boots"] if b["boot_id"] in boot_ids]}


# ---------------------------------------------------------------- independent oracle helpers (raw text / truth)
def raw_log(run, bid):
    return (Path(run) / "boots" / bid / "server.log").read_text(errors="replace")


def raw_cap(run, bid):
    m = re.findall(r"max_running_requests=([0-9]+), context_len", raw_log(run, bid))
    return int(m[-1]) if m else None


def raw_kv_tokens(run, bid):
    m = re.findall(r"#tokens: *([0-9]+)", raw_log(run, bid))
    return int(m[-1]) if m else None


def raw_is_llama(run, bid):
    return "type=LlamaForCausalLM" in raw_log(run, bid)


def raw_is_dummy(run, bid):
    return "load_format='dummy'" in raw_log(run, bid)


_RAW = {}


def raw_seng(run, bid, seq_range):
    key = (str(run), bid)
    if key not in _RAW:
        recs = [json.loads(x) for x in (Path(run) / "boots" / bid / "decode_probe.jsonl").read_text().splitlines()
                if '"decode_forward"' in x]
        _RAW[key] = {e["seq"]: e for e in recs}
    by = _RAW[key]
    w = [by[s] for s in range(seq_range[0], seq_range[1] + 1) if s in by]
    d = [(b["t_host_start"] - a["t_host_start"]) * 1000.0 for a, b in zip(w, w[1:])]
    return X.median(d)


def synth_window(c, c_in, bs=4, n=S.O_BASE, step=0.01):
    """Synthetic zero-ramp burst: record k at t=k*step (bs_real = bs from record 1 = first reach), every request has
    k-1 tokens at record k (ctx_k = c_in+k-1); the window keeps records 1+TRIM .. 1+TRIM+WINDOW and averages ctx over
    the WINDOW interval-start records, so ctx_mean = c_in + TRIM + (WINDOW-1)/2."""
    recs = [{"event": "decode_forward", "record_schema": X.DECODE_SCHEMA, "seq": k, "mode": "decode", "bs_real": bs,
             "bs_padded": bs, "graph": "hit", "t_host_start": k * step, "decode_fwd_gpu_ms": 9.0,
             "decode_fwd_host_ms": 0.5, "dropped_events": 0} for k in range(1, n)]
    reqs = [{"rid": f"q{i}", "chunks": [[k * step + 0.0001, k] for k in range(1, n + 1)], "prompt_tokens": c_in,
             "completion_tokens": n, "finish": "length", "error": None} for i in range(bs)]
    cell = {"cell_id": "syn", "bs": bs, "c": c, "c_in": c_in, "O": n, "t_first_send": 0.0, "t_end": (n + 1) * step,
            "requests": reqs}
    return recs, cell


# ---------------------------------------------------------------- oracles (return a list of failures)
def o_cap48(run, A):
    v = A["boots"].get("ja1_capinv_D44", {})
    bad = []
    if not v.get("usable") or v.get("problems"):
        bad.append(f"cap-48 boot unusable / problems {v.get('problems')}")
    elif v["identity"]["cap"] != raw_cap(run, "ja1_capinv_D44") or raw_cap(run, "ja1_capinv_D44") != 48:
        bad.append(f"recorded cap {v['identity']['cap']} != logged {raw_cap(run, 'ja1_capinv_D44')}")
    return bad


def o_capture_feasibility(_run=None, _A=None):
    f = {"kv_cache_tokens": 10 ** 7, "kv_k_gib": 1.0, "kv_v_gib": 1.0, "capture_bs": [1, 2, 4, 8, 12, 16, 24, 32, 40, 48],
         "max_mamba_cache_size": None}
    rows = X.feasibility([(64, 2048), (8, 2048)], f, 128, {(64, 2048): S.O_BASE, (8, 2048): S.O_BASE})
    bad = []
    if rows[0]["run"] or rows[0]["label"] != "INFEASIBLE_BY_RULE":
        bad.append("bs 64 (not in the capture list) was allowed to run")
    if not rows[1]["run"]:
        bad.append("bs 8 (in the list) refused")
    return bad


def o_cg_abort(run, A):
    v = next(iter(A["boots"].values()))
    return [] if any(p.startswith("ABORT_CG_BS") for p in v["problems"]) and not v.get("usable") else \
        ["cap-128 boot without bs 64/96 in the capture list was not aborted (ABORT_CG_BS)"]


CTX_MEAN_OFFSET = 16 + (300 - 1) / 2       # independent restatement of the window: trim 16, 300 intervals


def o_ctx_band(_run=None, _A=None):
    """rev3.4: the band is a LABEL (CTX_OFF_NOMINAL, |actual mean - c| > 64), never an exclusion."""
    bad = []
    c = 8192
    for delta, off in ((64, False), (-64, False), (65, True), (-65, True)):
        recs, cell = synth_window(c, c + delta - CTX_MEAN_OFFSET)
        w = X.t1_window(recs, cell, [1, 2, 4, 8])
        if w["status"] != "VALID" or ("CTX_OFF_NOMINAL" in w["labels"]) != off:
            bad.append(f"ctx mean {w.get('ctx_mean')} (c{delta:+d}) -> {w['status']} {w['labels']}")
    return bad


def o_infeasible_rows(run, A):
    bad = []
    for k, v in A["tables"]["t1"].items():
        if k.split("|")[0] == "J-H1" and v["n_registered"] != len(S.JH_T1_CELLS):
            bad.append(f"{k}: {v['n_registered']} feasibility rows != {len(S.JH_T1_CELLS)} registered cells")
    ll = [v for k, v in A["tables"]["t1"].items() if "|llama|" in k]
    if ll:
        kv = raw_kv_tokens(run, "jh1_llama")
        o_van = 332 + 64                                    # PREREG rev3.4 §3-1: vanilla O = O_BASE + 64
        want = sum(1 for b, c in S.JH_T1_CELLS if b * (c - 166 + o_van) > 0.95 * kv)
        if ll[0]["n_infeasible"] != want:
            bad.append(f"llama infeasible rows {ll[0]['n_infeasible']} != {want}")
    else:
        bad.append("llama T1 table missing")
    return bad


def o_t2_warmup(run, A):
    bad = []
    for bid, v in A["boots"].items():
        for x in v.get("t2", []) + v.get("t2p", []):
            if x["status"] == "VALID" and (x["n_used"] != S.T2_REPS - 1 or 0 in x["used_reps"]):
                bad.append(f"{bid} L{x['L']}: used reps {x['used_reps']}")
    return bad or ([] if any(v.get("t2") for v in A["boots"].values()) else ["no T2 values"])


def o_model_labels(run, A):
    bad = []
    for bid, v in A["boots"].items():
        if not v.get("identity"):
            continue
        if (v["identity"]["tag"] == "llama") != raw_is_llama(run, bid):
            bad.append(f"{bid}: tag {v['identity']['tag']} vs log llama={raw_is_llama(run, bid)}")
        if v["identity"]["tag"] != "llama" and v["identity"]["tag"].startswith("dummy") != raw_is_dummy(run, bid):
            bad.append(f"{bid}: tag {v['identity']['tag']} vs log dummy={raw_is_dummy(run, bid)}")
    return bad


def o_modes_separate(run, A):
    keys = list(A["tables"]["t1"])
    bad = []
    if not any("|108u|" in k for k in keys) or not any("|108v|" in k for k in keys):
        bad.append(f"108u / 108v not separate: {[k for k in keys if '108' in k]}")
    fu = A["fits"]["t1"].get("J-A1|r1|real_a4|108u|cap128")
    fv = A["fits"]["t1"].get("J-A1|r1|real_a4|108v|cap128")
    if not (fu and fv and fu["chosen"] and fv["chosen"]):
        bad.append("108u / 108v fits missing")
    elif abs(fv["chosen"]["coef"][0] / fu["chosen"]["coef"][0] - 1.03) > 0.005:     # fake truth: vanilla x 1.03
        bad.append("108v / 108u intercept ratio != fake truth 1.03")
    return bad


def o_seng_is_engine(run, A):
    _RAW.clear()
    bad, n = [], 0
    for bid, v in A["boots"].items():
        for w in v.get("t1_windows", []):
            if w.get("status") == "VALID" and w.get("seq_range"):
                n += 1
                ref = raw_seng(run, bid, w["seq_range"])
                if abs(w["S_eng_ms"] / ref - 1) > 1e-9:
                    bad.append(f"{bid} {w['cell_id']}: S_eng {w['S_eng_ms']} != decode_forward {ref}")
    return bad[:5] or ([] if n else ["no VALID engine windows"])


def o_gpu_field(run, A):
    bad, n = [], 0
    for bid, v in A["boots"].items():
        for w in v.get("t1_windows", []):
            if w.get("status") == "VALID" and w.get("S_eng_ms"):
                n += 1
                if w.get("gpu_ms") is None or abs(w["gpu_ms"] / w["S_eng_ms"] - 0.9) > 0.02:   # fake gpu_frac 0.9
                    bad.append(f"{bid} {w['cell_id']}: gpu/S_eng = {w.get('gpu_ms')}/{w['S_eng_ms']}")
                if "I4_UNRESOLVED" not in w.get("labels", []):                                 # 1 in 40 unresolved
                    bad.append(f"{bid} {w['cell_id']}: I4_UNRESOLVED missing (resolved {w.get('i4_resolved')})")
    return bad[:5] or ([] if n else ["no VALID engine windows"])


def o_excluded_boot(run, A):
    bad = [k for k in A["fits"]["t1"] if "|D24|" in k]
    if "SM_NOT_REALIZED_D24" not in A["boots"]["ja1_D24"]["labels"]:
        bad.append("D24 not labelled SM_NOT_REALIZED_D24")
    return [f"excluded boot in fits: {bad}"] if bad else []


def o_n_real(run, A):
    bad = []
    for k, x in A["tables"]["t4"].items():
        bid, L = k.split("|L")
        L = int(L)
        kv, cap = raw_kv_tokens(run, bid), raw_cap(run, bid)
        R_ = S.T4_R.get(L, S.JH_T4_R)
        truth = min(cap, kv // (L + R_))           # fake admission: FIFO, full reservation, cap = slots
        if x["n_real"] != truth:
            bad.append(f"{k}: n_real {x['n_real']} != engine truth {truth}")
    return bad or ([] if A["tables"]["t4"] else ["no T4 fills"])


def o_prereg_table(_run=None, _A=None):
    bad = []
    for L, (ca, cb, nb_ca, na48, nb48) in S.PREREG_T4_TABLE.items():
        R_ = S.T4_R[L]
        p = X.cmem_predictions(S.PREREG_POOL_BYTES, S.PREREG_S_BYTES, S.PREREG_K_BYTES, ca, L, R_)
        if (p["C_A"], p["C_B"], p["n_B"], p["n_A"]) != (ca, cb, nb_ca, ca):
            bad.append(f"L{L} cap=C_A: {(p['C_A'], p['C_B'], p['n_B'], p['n_A'])} != {(ca, cb, nb_ca, ca)}")
        q = X.cmem_predictions(S.PREREG_POOL_BYTES, S.PREREG_S_BYTES, S.PREREG_K_BYTES, 48, L, R_)
        if (q["n_A"], q["n_B"]) != (na48, nb48):
            bad.append(f"L{L} cap 48: {(q['n_A'], q['n_B'])} != {(na48, nb48)}")
    return bad


def o_mismatch_excluded(run, A):
    ag = A["labels"]["agreement"]
    mm = [k for k, v in ag.items() if "MISMATCH" in v["label"]]
    bad = [] if any(k.endswith("|32") for k in mm) else [f"no MISMATCH_bs32 under client drift: {mm}"]
    for k in A["tables"]["t1_client_secondary"]:
        m, mode, bs, _ = k.split("|")
        if f"{m}|{mode}|{bs}" in mm:
            bad.append(f"secondary table keeps MISMATCH row {k}")
    return bad[:5]


def o_realized_axis(run, A):
    bad = []
    for bid, want in (("ja1_D34", 36),):
        g = json.loads((Path(run) / "boots" / bid / "green.json").read_text())
        got = next(r for r in g["groups"] if r["stream_index"] == S.STICKY_IDX[34])["decode"]["green_sm"]["smCount"]
        sm = A["boots"][bid]["sm"]
        if sm["D_real"] != got or got != want:
            bad.append(f"{bid}: recorded D {sm['D_real']} != green read-out {got}")
    if "SM_NOT_REALIZED_D24" not in A["boots"]["ja1_D24"]["labels"]:
        bad.append("D24 realized 16 SM but not labelled")
    return bad


def o_weight_sign(run, A):
    wi = A["jh1"]["w_indep"]["T1"]
    lns = list(wi["ln"].values())
    bad = []
    if wi["label"] != "WEIGHT_DEPENDENT" or not lns or not all(x < 0 for x in lns):   # truth: real 5 % slower
        bad.append(f"W-INDEP T1 {wi['label']} ln {lns[:2]} (expected WEIGHT_DEPENDENT, ln(dummy/real) < 0)")
    return bad + o_model_labels(run, A)


def o_off_clean(run, A):
    bad = []
    for bid, v in A["boots"].items():
        if v["kind"] == "OFF":
            if v["problems"]:
                bad.append(f"{bid}: {v['problems']}")
            if (Path(run) / "boots" / bid / "decode_probe.jsonl").exists():
                bad.append(f"{bid}: OFF boot wrote decode_probe.jsonl")
    return bad


def o_off_not_applied(run, A):
    tmp = Path(str(run) + "_noOFF")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(run, tmp)
    for d in (tmp / "boots").iterdir():
        if "_off_" in d.name:
            shutil.rmtree(d)
    B = R.analyze(tmp)
    shutil.rmtree(tmp)
    return [] if B["tables"]["t1"] == A["tables"]["t1"] else ["T1 table changes when OFF diagnostic boots are present"]


def o_t3_rejected(_run=None, _A=None):
    p = R.make_plan("J-A", 1)
    p["boots"][0]["t3"] = [[8, 8192]]
    return [] if any("T3" in x for x in R.plan_problems(p)) else ["T3 cell in a TEL-OFF boot accepted"]


def o_sealed(run, A):
    bad = []
    for bid, v in A["boots"].items():
        if v.get("t4_fills"):
            if v.get("t4_prediction_order") != "sealed before fill" or any("T4_PREDICTION" in p for p in v["problems"]):
                bad.append(f"{bid}: {v.get('t4_prediction_order')} {v['problems']}")
    return bad or ([] if any(v.get("t4_fills") for v in A["boots"].values()) else ["no fills"])


def _raw_dec(run, bid):
    return sorted((json.loads(x) for x in (Path(run) / "boots" / bid / "decode_probe.jsonl").read_text().splitlines()
                   if '"decode_forward"' in x), key=lambda e: e["seq"])


def _raw_cells(run, bid):
    p = Path(run) / "boots" / bid / "cells.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def o_window_reach(run, A):
    """Decision 28 A-1, from the raw records: every feasible T1 cell of the PD-mux boots of JA0 is VALID and its window
    starts TRIM records after the cell's first record with bs_real == bs (ramp-aware), with bs_real == bs throughout."""
    bad = []
    for bid in ("ja1_D44", "ja1_108u"):
        v = A["boots"].get(bid) or {}
        raw = _raw_dec(run, bid)
        feas = [r for r in v.get("feasibility", []) if r["run"]]
        cells = {(c["bs"], c["c"]): c for c in _raw_cells(run, bid) if c["kind"] == "t1"}
        wins = {(w["bs"], w["c"]): w for w in v.get("t1_windows", [])}
        if not feas or len(wins) != len(feas):
            bad.append(f"{bid}: {len(wins)} windows for {len(feas)} feasible cells")
        for key, w in wins.items():
            if w.get("status") != "VALID":
                bad.append(f"{bid} {key}: {w.get('status')}")
                continue
            c = cells[key]
            inside = [e for e in raw if c["t_first_send"] <= e["t_host_start"] <= c["t_end"]]
            reach = next(e["seq"] for e in inside if e["bs_real"] == key[0])
            lo, hi = w["seq_range"]
            if lo != reach + 16 or hi != reach + 316:
                bad.append(f"{bid} {key}: window {lo}-{hi}, raw first reach {reach}")
            if any(e["bs_real"] != key[0] for e in inside if lo <= e["seq"] <= hi):
                bad.append(f"{bid} {key}: bs_real != bs inside the window")
    return bad[:6]


def o_fit_truth(run, A):
    """108u decodes on 108 SM when no span is in flight: the F1 fit must return the fake's (a108, b108, c108)."""
    t = dict(__import__("ct_fake_engine").DEFAULT_TRUTH)
    fit = (A["fits"]["t1"].get("J-A1|r1|real_a4|108u|cap128") or {}).get("chosen")
    if not fit:
        return ["108u fit missing"]
    bad = []
    for got, want, nm in zip(fit["coef"], (t["a108"], t["b108"], t["c108"]), "abc"):
        if abs(got / want - 1) > 1e-3:
            bad.append(f"108u {nm} = {got} vs truth {want}")
    return bad


def o_no_residual(run, A):
    """Raw records: inside every T1 cell interval the engine never decodes more requests than the cell sent."""
    bad = []
    for bid in A["boots"]:
        raw = _raw_dec(run, bid) if (Path(run) / "boots" / bid / "decode_probe.jsonl").exists() else []
        for c in _raw_cells(run, bid):
            if c["kind"] != "t1":
                continue
            mx = max((e["bs_real"] for e in raw if c["t_first_send"] <= e["t_host_start"] <= c["t_end"]), default=0)
            if mx > c["bs"]:
                bad.append(f"{bid} {c['cell_id']}: bs_real {mx} > {c['bs']} (residual requests)")
    return bad[:5] or ([] if any(_raw_cells(run, b) for b in A["boots"]) else ["no cells"])


def o_t2_p108(_run=None, _A=None):
    """Decision 28 A-2: T2 only in 108u / 108v; a sticky-D boot with T2 is refused by the plan check."""
    bad = []
    p = R.make_plan("J-A", 1)
    if any(b["t2"] for b in p["boots"] if b["engine"] == "pdmux_sticky"):
        bad.append("registered J-A plan has T2 in a sticky-D boot")
    b = R._boot("J-A1", 1, "x", "T1T2", "nano", "pdmux_sticky", S.CAP_MAIN, D=44, t2=[4096])
    if not any("T2 in a D44 boot" in x for x in R.plan_problems({"boots": [b]})):
        bad.append("T2 in a sticky-D boot accepted (would be labelled P = 108 - D)")
    return bad


def o_t2p_confirm(run, A):
    """Vanilla prefills first (the decode is not executing during the prefill), PD-mux decodes concurrently."""
    bad = []
    for bid, want in (("ja1_D44", "CONFIRMED"), ("ja1_108v", "UNCONFIRMED")):
        for x in A["boots"][bid].get("t2p", []):
            if x["L"] >= 4096 and want not in (x.get("coexist_engine") or {}).get("label", ""):
                bad.append(f"{bid} L{x['L']}: {(x.get('coexist_engine') or {}).get('label')} (expected {want})")
    return bad or ([] if A["boots"]["ja1_D44"].get("t2p") else ["no T2'"])


# ---------------------------------------------------------------- mutants
def _wrap(obj, name, fn):
    orig = getattr(obj, name)
    return mock.patch.object(obj, name, lambda *a, **k: fn(orig, *a, **k))


def _identity_ignoring_arch(orig, b, f):
    out = orig(b, f)
    if out["tag"] == "llama":
        out = dict(out, model="nano", tag="real_a4")
    return out


def _identity_merge108(orig, b, f):
    out = orig(b, f)
    return dict(out, mode="108") if out["mode"] in ("108u", "108v") else out


def _identity_plan_loadformat(orig, b, f):
    out = orig(b, f)
    if out["tag"] != "llama":
        out = dict(out, tag=f"{'real' if b['dummy'] else 'dummy'}_a{out['n_attn']}")   # swapped plan label
    return out


def _t1_client_as_seng(orig, *a, **k):
    w = orig(*a, **k)
    if w.get("client_gap_ms") is not None:
        w["S_eng_ms"] = w["client_gap_ms"]
    return w


def _drop_infeasible(orig, *a, **k):
    return [r for r in orig(*a, **k) if r["run"]]


def _t2_with_warmup(orig, rec, warmup=None):
    return orig(rec, warmup=0)


def _usable_always(orig, bdir):
    v = orig(bdir)
    if v.get("identity"):
        v["usable"] = True
    return v


def _waiting_running(orig, fill, records, admission=None):
    out = orig(fill, records, admission)
    out["n_real"] = fill["N"]                     # every sent, not-yet-finished request counted as running
    out["labels"] = [l for l in out["labels"] if l != "FILL_INCOMPLETE"]
    return out


def _r_zero(orig, pool, s, k, cap, L, R_):
    p = orig(pool, s, k, cap, L, R_)
    p["n_B"] = min(cap, int(math.floor((pool - cap * s) / (k * L))))
    return p


def _na_asym(orig, pool, s, k, cap, L, R_):
    p = orig(pool, s, k, cap, L, R_)
    p["n_A"] = min(cap, p["C_A"])
    return p


def _keep_mismatch(orig, cells, agree):
    return orig(cells, {k: v for k, v in agree.items() if "MISMATCH" not in v["label"]})


def _requested_sm(orig, boot, green):
    out = orig(boot, green)
    if boot["engine"] == "pdmux_sticky":
        out = dict(out, D_real=boot["D"], P_real=S.TOTAL_SM - boot["D"], ok=True, label=None)
    return out


def _env_leak(orig, b, bdir, base=None, run_id="local"):
    env = orig(b, bdir, base, run_id)
    env["PDMUX_DECODE_PROBE_PATH"] = str(Path(bdir) / "decode_probe.jsonl")
    return env


def _off_correction(orig, bdir):
    v = orig(bdir)
    if v.get("kind") == "T1T2" and (Path(bdir).parent / v["boot_id"].replace("ja1_", "ja1_off_")).exists():
        for w in v.get("t1_windows", []):
            if w.get("S_eng_ms"):
                w["S_eng_ms"] *= 0.98          # "instrumentation correction" from the OFF ratio
    return v


def _no_t3_check(orig, plan):
    return [p for p in orig(plan) if "T3" not in p]


def _late_prediction(orig, b, out, variant_dirs, make_server, run_id="local"):
    res = orig(b, out, variant_dirs, make_server, run_id)
    pf = Path(out) / "boots" / b["boot_id"] / "t4_prediction.json"
    if pf.exists():                         # rewritten after the fill (with knowledge of the outcome)
        d = json.loads(pf.read_text())
        for p in d:
            p["n_B"] += 1
        pf.write_text(json.dumps(d))
    return res


def _reach_first_decode(orig, recs, bs):
    return 0 if recs else None


def _nominal_ctx(orig, w):
    return w["c"]


def _no_cancel(orig, tr, hs):
    return None


def _t2_axis_d(orig, mode):
    return S.TOTAL_SM - int(mode[1:]) if mode.startswith("D") else orig(mode)


def _always_confirmed(orig, *a, **k):
    out = orig(*a, **k)
    if out.get("label") != "NOT_COLLECTED":
        out = dict(out, confirmed=True, label=out["label"].replace("UNCONFIRMED", "CONFIRMED"))
    return out


def _o_base_only(orig, tr, cell_id, bs, c, rnd, O, **k):
    return orig(tr, cell_id, bs, c, rnd, S.O_BASE, **k)


# ---------------------------------------------------------------- stage 2 oracles (raw TEL / decode records / cells)
def _raw_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _raw_spans(run, bid):
    ev = _raw_jsonl(Path(run) / "boots" / bid / "tel.jsonl")
    st = {e["span_seq"]: e for e in ev if e.get("event") == "prefill_span_start"}
    en = {e["span_seq"]: e for e in ev if e.get("event") == "prefill_span_end"}
    its = sorted(e["t_launch"] for e in ev if e.get("event") == "decode_iteration")
    return [(st[q]["timestamp_monotonic_s"], en[q]["timestamp_monotonic_s"]) for q in sorted(st) if q in en], its, st


def _span_of(spans, t0, t1):
    c = [sp for sp in spans if t0 <= sp[0] <= t1]
    return c[0] if len(c) == 1 else None


def _t3_reps(A):
    for key, tab in (A["tables"].get("t3") or {}).items():
        for ck, c in tab["cells"].items():
            for w in c["reps"]:
                yield key, ck, w


def _bid_of(A, key):
    job, rnd, tag, mode = key.split("|")[:4]
    return next(b for b, v in A["boots"].items() if v.get("identity") and v["job"] == job and f"r{v['round']}" == rnd
                and v["identity"]["tag"] == tag and v["identity"]["mode"] == mode and v["kind"] == "T3")


def o_t3_sigma_engine(run, A):
    """sigma' from the RAW engine spans: the cell's own prefills (send .. first token) over the reference spans of the
    SAME boot and L (reps >= 1) -- medians, compared to 1e-9 (never client TTFT, never another boot, never x sigma)."""
    bad, n = [], 0
    for key, ck, w in _t3_reps(A):
        if w.get("status") != "VALID":
            continue
        bid = _bid_of(A, key)
        spans, _, _ = _raw_spans(run, bid)
        cells = _raw_cells(run, bid)
        c = next(x for x in cells if x["kind"] == "t3" and x["cell_id"] == w["cell_id"])
        ref = next(x for x in cells if x["kind"] == "t3ref" and x["L"] == c["L"])
        co = [_span_of(spans, p["t_send"], p["t_first"]) for p in c["prefills"]]
        rf = [_span_of(spans, r["t_send"], r["t_first"]) for r in ref["reps"] if r["rep"] >= 1]
        want = X.median([(b - a) * 1000 for a, b in co]) / X.median([(b - a) * 1000 for a, b in rf])
        n += 1
        if w.get("sigma_prime") is None or abs(w["sigma_prime"] / want - 1) > 1e-9:
            bad.append(f"{key} {ck} rep {w.get('rep')}: sigma' {w.get('sigma_prime')} != raw {want}")
    return bad[:4] or ([] if n else ["no VALID T3 rep"])


def o_t3_rho_stall(run, A):
    """rho = median(co) / median(solo) of the cell's OWN raw intervals (t3_iterations.json); a stall = co interval >
    5 x the SOLO median; n_it_obs = raw decode_iteration count with t_launch inside the qualifying spans."""
    raw = json.loads((Path(run) / "t3_iterations.json").read_text())
    bad, n, stalls = [], 0, 0
    for key, ck, w in _t3_reps(A):
        if w.get("status") != "VALID":
            continue
        r = raw[key][ck][str(w["rep"])]
        rho = X.median(r["co_ms"]) / X.median(r["solo_ms"])
        thr = 5.0 * X.median(r["solo_ms"])
        ns = sum(1 for x in r["co_ms"] if x > thr)
        stalls += ns
        n += 1
        if abs(w["rho"] / rho - 1) > 1e-9:
            bad.append(f"{key} {ck}: rho {w['rho']} != raw {rho}")
        if abs(w["stall_co"]["threshold_ms"] / thr - 1) > 1e-9 or w["stall_co"]["n_stall"] != ns:
            bad.append(f"{key} {ck}: stall thr {w['stall_co']['threshold_ms']} n {w['stall_co']['n_stall']} != {thr} {ns}")
        bid = _bid_of(A, key)
        spans, its, _ = _raw_spans(run, bid)
        c = next(x for x in _raw_cells(run, bid) if x["kind"] == "t3" and x["cell_id"] == w["cell_id"])
        cnt = 0
        for p in c["prefills"]:
            sp = _span_of(spans, p["t_send"], p["t_first"])
            cnt += sum(1 for t in its if sp[0] <= t <= sp[1])
        if w.get("n_it_obs") != cnt:
            bad.append(f"{key} {ck}: n_it_obs {w.get('n_it_obs')} != raw count {cnt}")
    if n and not stalls:
        bad.append("scenario has no stall (oracle cannot see a threshold defect)")
    return bad[:4] or ([] if n else ["no VALID T3 rep"])


def o_t3_window(run, A):
    """Every T3 cell of the scenario is VALID and its pre solo window starts TRIM records after the raw first record of
    the cell with bs_real == bs (ramp-aware, as T1); fake truth rho = kappa 1.25 (+- ctx drift 1 %)."""
    bad, n = [], 0
    for key, ck, w in _t3_reps(A):
        n += 1
        if w.get("status") != "VALID":
            bad.append(f"{key} {ck}: {w.get('status')} {w.get('detail')}")
            continue
        bid = _bid_of(A, key)
        c = next(x for x in _raw_cells(run, bid) if x["kind"] == "t3" and x["cell_id"] == w["cell_id"])
        reach = next(e["seq"] for e in _raw_dec(run, bid) if c["t_first_send"] <= e["t_host_start"] <= c["t_end"]
                     and e["bs_real"] == c["bs"])
        if w["pre_seq"][0] != reach + 16:
            bad.append(f"{key} {ck}: pre window starts at {w['pre_seq'][0]}, raw reach {reach}")
        if abs(w["rho"] / 1.25 - 1) > 0.01:
            bad.append(f"{key} {ck}: rho {w['rho']} vs fake kappa 1.25")
    return bad[:4] or ([] if n else ["no T3 rep"])


def o_jc_anchor(_run=None, _A=None):
    """§5-2: the N=1 anchor boots at cap 48 (PREREG literal) on GPU 0 only; the plan check refuses a cap-128 anchor."""
    bad = []
    p = R.make_plan("J-C")
    for b in p["boots"]:
        if b["tp"] == 1 and ((b["cap"] != 48 and "n1c32" not in b["boot_id"]) or b["gpus"] != "0"):
            bad.append(f"{b['boot_id']}: anchor cap {b['cap']} gpus {b['gpus']}")
    x = R._boot("J-C", 1, "x", "T1T2", "nano", "pdmux_sticky", 128, D=44, t1=[(8, 2048)], o_family="JC|44")
    if not any("anchor cap" in q for q in R.plan_problems({"boots": [x]})):
        bad.append("plan check accepted a cap-128 N=1 anchor")
    return bad


def o_jc_ratios(run, A):
    """TP2 / N=1 decode-step ratio of the same cell = fake tp_dec_mult 0.62; F1 / F2 held from both rank banners."""
    jc = A.get("jc") or {}
    bad = []
    rs = [x["ratio"] for r in jc.get("ratios_t1", {}).values() for x in r.values()]
    if not rs or any(abs(x / 0.62 - 1) > 1e-6 for x in rs):
        bad.append(f"TP2/N1 ratios {rs[:3]} != 0.62")
    for bid, f in (jc.get("F") or {}).items():
        if not (f and f["F1"]["held"] and f["F2"]["held"] and f["F4"]["label"] == "F4_STRIDE_DERIVED"):
            bad.append(f"{bid}: F1/F2/F4 {f and (f['F1']['held'], f['F2']['held'], f['F4']['label'])}")
    return bad


def _smoke_run(root):
    plan = sub_plan("J-A", ["ja1_D44"])
    out = Path(root) / "smoke"
    R.run_job("J-A", out, 1, None, None, make_server=lambda b, a, e: R.VirtualServer(b, a, e, dict(FAST_PF)),
              plan=plan, smoke=True, select=["ja1_D44:t1:8x2048", "ja1_D44:t2p:4096"])
    A = R.analyze(out, Path(root) / "summary")
    return out, A


def o_smoke(_run=None, _A=None):
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="ct_smoke_"))
    try:
        out, A = _smoke_run(root)
        bad = []
        if not (out / "SMOKE_UNSEALED").exists() or not (root / "summary" / "SMOKE_UNSEALED").exists():
            bad.append("SMOKE_UNSEALED marker missing (raw or summary)")
        if not A.get("smoke"):
            bad.append("analysis not flagged smoke")
        w = A["boots"]["ja1_D44"]["t1_windows"]
        if [(x["bs"], x["c"]) for x in w] != [(8, 2048)] or "ramp_steps_ub" not in w[0] or "ramp_records" not in w[0]:
            bad.append(f"smoke cell set / ramp fields wrong: {[(x['bs'], x['c']) for x in w]}")
        return bad
    finally:
        shutil.rmtree(root, True)


def o_resume(_run=None, _A=None):
    """Kill a boot after 2 finished cells, resume: the 2 stay (PARTIAL), the rest run once in '<boot>@s1', nothing runs
    twice, RESUME_LOG.json and the analysis record it."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="ct_resume_"))
    plan = sub_plan("J-A", ["ja1_D44"])
    out = root / "raw"
    mk = lambda b, a, e: R.VirtualServer(b, a, e, dict(FAST_PF))  # noqa: E731
    n = {"k": 0}
    orig = R._run_cell

    def dying(*a, **k):
        n["k"] += 1
        if n["k"] > 2:
            raise KeyboardInterrupt("job stopped")
        return orig(*a, **k)
    try:
        with mock.patch.object(R, "_run_cell", dying):
            try:
                R.run_job("J-A", out, 1, None, None, make_server=mk, plan=plan)
            except KeyboardInterrupt:
                pass
        R.run_job("J-A", out, 1, None, None, make_server=mk, plan=plan, resume=True)
        A = R.analyze(out)
        bad = []
        b0 = out / "boots" / "ja1_D44"
        if not (b0 / "PARTIAL").exists():
            bad.append("partial boot not kept")
        keys = [json.loads(x)["cell_key"] for d in (b0, out / "boots" / "ja1_D44@s1") if (d / "cells.jsonl").exists()
                for x in (d / "cells.jsonl").read_text().splitlines() if x.strip()]
        reg = {R.cell_key(*c) for c in R.registered_cells(plan["boots"][0],
                                                          json.loads((b0 / "feasibility.json").read_text()))}
        if len(keys) != len(set(keys)) or set(keys) != reg:
            bad.append(f"cells run {len(keys)} (unique {len(set(keys))}) vs registered feasible {len(reg)}")
        if not json.loads((out / "RESUME_LOG.json").read_text() or "[]"):
            bad.append("RESUME_LOG.json empty")
        if "RESUMED_SEGMENT" not in A["boots"].get("ja1_D44@s1", {}).get("labels", []):
            bad.append("segment boot not labelled RESUMED_SEGMENT")
        return bad
    except Exception as exc:  # noqa: BLE001
        return [f"resume failed: {exc!r}"]
    finally:
        shutil.rmtree(root, True)


def o_raw_not_exported(run, A):
    import tempfile
    sd = Path(tempfile.mkdtemp(prefix="ct_sum_"))
    try:
        R.export_summary(run, sd)
        heavy = [p.name for p in sd.rglob("*") if p.is_file() and (p.name in ("server.log", "cells.jsonl")
                 or p.name.startswith(("decode_probe", "tel", "admission")))]
        bad = [f"raw files exported to results/: {heavy[:4]}"] if heavy else []
        if not (sd / "analysis.json").exists() or not list((sd / "boot_manifests").glob("*.DONE")):
            bad.append("summary misses analysis.json / boot DONE manifests")
        return bad
    finally:
        shutil.rmtree(sd, True)


# ---------------------------------------------------------------- stage 2 mutants
def _stall_co_median(orig, seq_vals, thr):
    co = [x for r in seq_vals for x in r]
    return orig(seq_vals, S.STALL_MULT * X.median(co)) if co else orig(seq_vals, thr)


def _sigma_times_sigma(orig, v, boots):
    out = orig(v, boots)
    for w in out:
        if w.get("sigma_prime") and w.get("span_obs_lag_bound_rel") is not None:
            w["sigma_prime"] *= 1.0 + 0.5 * w["span_obs_lag_bound_rel"] + 0.01   # "x sigma (vs T2)" combined
    return out


def _anchor_cap128(orig):
    p = orig()
    for b in p["boots"]:
        if b["tp"] == 1:
            b["cap"] = 128
    return p


def _sigma_client(orig, cell_out, ref_out):
    sg, why = orig(cell_out, ref_out)
    if sg is not None:
        sg *= 1.0 + 1e-3        # client TTFT includes the queue wait: never equal to the engine span ratio
    return sg, why


def _sigma_other_boot(orig, v, boots):
    others = [b for b in boots.values() if b is not v and b.get("t3_refs")]
    if others:
        v = dict(v, t3_refs=others[0]["t3_refs"])
    return orig(v, boots)


def _rho_other_boot(orig, v, boots):
    out = orig(v, boots)
    other = next((b for b in boots.values() if b is not v and b.get("t3_cells")), None)
    if other:
        solo = {(w["bs"], w["L"]): w.get("solo_median_ms") for w in other["t3_cells"]}
        for w in out:
            if w.get("rho") and solo.get((w["bs"], w["L"])):
                w["rho"] = w["co_median_ms"] / solo[(w["bs"], w["L"])]
    return out


def _n_it_converted(orig, its, t_lo, t_hi):
    ts = sorted(e["t_launch"] for e in its)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return int(round((t_hi - t_lo) / X.median(gaps))) if gaps else 0


def _t3_first_decode(orig, *a, **k):
    with mock.patch.object(X, "first_reach_index", lambda recs, bs: 0 if recs else None):
        return orig(*a, **k)


def _t3_o_small(orig, tr, cell_id, bs, L, m, rnd, O_dec, **k):
    return orig(tr, cell_id, bs, L, m, rnd, S.T3_PRE_TOKENS + S.T3_POST_TOKENS + S.MARGIN_ITERS, **k)


def _no_smoke_marker(orig, *a, **k):
    with mock.patch.object(S, "SMOKE_MARKER", "SMOKE_MARKER_DISABLED_BY_MUTANT"):
        return orig(*a, **k)


def _no_resume(orig, plan, out):
    for b in plan["boots"]:
        d = Path(out) / "boots" / b["boot_id"]
        if d.exists() and not (d / "DONE").exists():
            shutil.rmtree(d)                     # rev3.4 behaviour: redo the partial boot from scratch
    return plan, []


def _export_all(orig, run_dir, summary_dir):
    shutil.copytree(run_dir, summary_dir, dirs_exist_ok=True)


# ---------------------------------------------------------------- decision 30 mutants (long points)
def _long_mid_job(orig, rnd):
    p = orig(rnd)
    lb = [b for b in p["boots"] if b["long"]]
    if lb:
        p["boots"].remove(lb[0])
        p["boots"].insert(len(p["boots"]) // 2, lb[0])
    return p


def _long_in_grid_boot(orig, rnd):
    """the 96k / 124k points appended to the 73,728-context 108v grid boot instead of a separate long boot"""
    p = orig(rnd)
    lb = [b for b in p["boots"] if b["long"]]
    if lb:
        p["boots"].remove(lb[0])
        g = next(b for b in p["boots"] if b["mode"] == "108v" and b["kind"] == "T1T2")
        g["long"] = list(S.T2EXT_L)
    return p


def _wipe_on_death(orig, b, out, *a, **k):
    res = orig(b, out, *a, **k)
    if "SERVER_DEATH" in res.get("labels", []):
        for d in (Path(out) / "boots").iterdir():
            if d.name != b["boot_id"]:
                shutil.rmtree(d, True)
        (Path(out) / "boots" / b["boot_id"] / "cells.jsonl").write_text("")
    return res


def _no_context_guard(orig, b, f):
    return [dict(r, run=True, label=None, reasons=[]) for r in orig(b, f)]


@contextlib.contextmanager
def _no_runtime_context_check():
    """both run-time context checks off: the boot-level ABORT_CONTEXT (logged context != plan) and the per-L
    feasible_long check against the logged context_len"""
    with _wrap(R, "long_feasibility", _no_context_guard), \
            _wrap(X, "label_problems", lambda orig, *a, **k: [q for q in orig(*a, **k) if not q.startswith("ABORT_CONTEXT")]):
        yield


def _raw_long_cells(run, bid):
    p = Path(run) / "boots" / bid / "cells.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip() and json.loads(x).get("kind") == "long"] \
        if p.exists() else []


LONG_EXPECT = {("J-A", 1): ["ja1_long_108v"], ("J-A", 2): [], ("J-H1", 1): ["jh1_long_llama", "jh1_long_real_a4"],
               ("J-C", 1): ["jc_tp2_long_108v", "jc_tp2_long_108u"]}      # independent of S.LONG_BOOTS (decisions 30, 33)


def o_long_placement(_run=None, _A=None):
    """Decisions 30 / 33: the long boots of a Job (J-A1: 1, J-H1: 2, J-C: 2, J-A2: none) form its LAST contiguous block,
    each long-only, context 131,072 > L + 1, L = 65,536 / 98,304 / 126,976 ascending, P = 108 engine (vanilla, or the
    PD-mux engine unpartitioned at TP2); the plan check refuses a long boot with a measurement boot after it."""
    bad = []
    for (job, rnd), ids in LONG_EXPECT.items():
        p = R.make_plan(job, rnd)
        lb = [b["boot_id"] for b in p["boots"] if b["long"]]
        if lb != ids:
            bad.append(f"{job} r{rnd}: long boots {lb} != {ids}")
            continue
        if not ids:
            continue
        if [b["boot_id"] for b in p["boots"][-len(ids):]] != ids:
            bad.append(f"{job} r{rnd}: long boots are not the last {len(ids)} boots, in order")
        for b in p["boots"]:
            if b["long"]:
                if b["long"] != [65536, 98304, 126976] or b["context"] != 131072 or any(L + 1 >= b["context"] for L in b["long"]):
                    bad.append(f"{b['boot_id']}: long {b['long']} context {b['context']}")
                if b["t1"] or b["t2"] or b["t2p"] or b["t4"] or b.get("t3") or b["engine"] not in ("vanilla", "pdmux_108u"):
                    bad.append(f"{b['boot_id']}: long boot not a separate P = 108 long-only boot")
            elif any(L >= 73728 for L in list(b["t2"]) + list(b["t2p"])):
                bad.append(f"{b['boot_id']}: L >= 73,728 in a 73,728-context boot")
        x = R.make_plan(job, rnd)
        x["boots"].insert(0, x["boots"].pop(-len(ids)))        # the first long boot moved to the front: mid-Job
        if not any("LAST" in q for q in R.plan_problems(x)):
            bad.append(f"{job}: plan check accepted a mid-Job long boot")
    return bad


def o_long_death(run, A):
    """A server death at 126,976 keeps everything before it: the earlier boot (DONE, cells), the 65,536 / 98,304 rows
    (served, VALID T2 value), and labels 126,976 SERVER_DEATH -- from the raw directory and the analysis."""
    bad = []
    off = Path(run) / "boots" / "ja1_off_108v"
    if not (off / "DONE").exists() or not (off / "cells.jsonl").exists() or not (off / "cells.jsonl").read_text().strip():
        bad.append("earlier boot ja1_off_108v lost after the long boot's death")
    cells = {c["L"]: c for c in _raw_long_cells(run, "ja1_long_108v")}
    for L in (65536, 98304):
        c = cells.get(L)
        if c is None or not all(r.get("error") is None and r.get("completion_tokens") == 1 for r in c["reps"]):
            bad.append(f"raw long cell {L} missing / not served before the death")
    res = json.loads((Path(run) / "boots" / "ja1_long_108v" / "boot_result.json").read_text())
    if "SERVER_DEATH" not in res.get("labels", []):
        bad.append("long boot death not labelled SERVER_DEATH")
    rows = {r["L"]: r for x in (A or {}).get("tables", {}).get("t2_long", {}).values() for r in x["rows"]}
    for L in (65536, 98304):
        if rows.get(L, {}).get("status") != "VALID" or f"LONG_SERVED_L{L}" not in rows.get(L, {}).get("labels", []):
            bad.append(f"analysis row {L}: {rows.get(L)}")
    if "SERVER_DEATH" not in rows.get(126976, {}).get("labels", []):
        bad.append(f"analysis row 126976 not SERVER_DEATH: {rows.get(126976)}")
    if not (A or {}).get("boots", {}).get("ja1_off_108v", {}).get("usable"):
        bad.append("earlier boot not usable in the analysis")
    return bad


def o_long_ctx_guard(run, A):
    """No long request with L + 1 >= the LOGGED context_len is sent; such L are ABORT_CONTEXT_TOO_SHORT rows."""
    bad = []
    m = re.findall(r"context_len=([0-9]+)", raw_log(run, "ja1_long_108v"))
    ctx = int(m[-1]) if m else None
    for c in _raw_long_cells(run, "ja1_long_108v"):
        if ctx is None or c["L"] + 1 >= ctx:
            bad.append(f"long L {c['L']} sent to a server with context_len {ctx}")
    rows = {r["L"]: r for x in (A or {}).get("tables", {}).get("t2_long", {}).values() for r in x["rows"]}
    for L in (98304, 126976):
        if "ABORT_CONTEXT_TOO_SHORT" not in rows.get(L, {}).get("labels", []):
            bad.append(f"row {L} not ABORT_CONTEXT_TOO_SHORT: {rows.get(L)}")
    return bad



# ---------------------------------------------------------------- decision 33 mutants (restored §11-4 points)
def _strip_bs48(p, only_anchor):
    for b in p["boots"]:
        if b["job"] == "J-C" and not b["long"] and (b["tp"] == 1 or not only_anchor):
            b["t1"] = [c for c in b["t1"] if c[0] != 48]
            b["t1_O"] = {k: v for k, v in b["t1_O"].items() if not k.startswith("48x")}
    return p


def _bs48_tp2_only(orig):
    return _strip_bs48(orig(), True)


def _bs48_nowhere(orig):
    return _strip_bs48(orig(), False)


def _ja_ctx_73728(orig, rnd):
    p = orig(rnd)
    for b in p["boots"]:
        if not b["long"]:
            b["context"] = 73728
    return p


def _drop_8x64k_cell(orig, b, feas=None, feas3=None, feasL=None):
    return [c for c in orig(b, feas, feas3, feasL) if not (c[0] == "t1" and tuple(c[1]) == (8, 65536))]


def _feas_default_context(orig, bs, c, O, kv, cap, capt, context=None):
    return orig(bs, c, O, kv, cap, capt, None)


def _drop_boot_noargs(boot_id):
    def f(orig):
        p = orig()
        p["boots"] = [b for b in p["boots"] if b["boot_id"] != boot_id]
        return p
    return f


def _jh1_long_mid(orig):
    p = orig()
    lb = [b for b in p["boots"] if b["long"]]
    p["boots"] = [b for b in p["boots"] if not b["long"]]
    p["boots"][3:3] = lb
    return p


def _drop_anchor32(orig):
    p = orig()
    p["boots"] = [b for b in p["boots"] if "n1c32" not in b["boot_id"]]
    return p


def _anchor32_tp2_O(orig):
    p = orig()
    for b in p["boots"]:
        if "n1c32" in b["boot_id"]:
            b["o_family"] = "JC|44"
            b["t1_O"] = {f"{bs}x{c}": S.o_tokens(bs, c, "JC|44") for bs, c in b["t1"]}
    return p


def _jc_108u_mid(orig):
    p = orig()
    lb = [b for b in p["boots"] if b["boot_id"] == "jc_tp2_long_108u"]
    p["boots"] = [b for b in p["boots"] if b["boot_id"] != "jc_tp2_long_108u"]
    p["boots"].insert(3, lb[0])
    return p


def _jc_108u_sticky(orig):
    p = orig()
    for b in p["boots"]:
        if b["boot_id"] == "jc_tp2_long_108u":
            b.update(engine="pdmux_sticky", mode="D44", D=44)
    return p


def _plan_check_last_boot_only(orig, plan):
    """the pre-decision-33 style check regressed into 'only the very last boot is looked at': a long boot followed by a
    measurement boot is not refused unless that long boot IS the last boot"""
    return [q for q in orig(plan) if "LAST contiguous" not in q]


def _anchor32_not_special(orig, v):
    return False


def _long_join_ignores_tag(orig, g, v):
    gi, vi = dict(g.get("identity") or {}), dict(v.get("identity") or {})
    return orig(dict(g, identity=dict(gi, tag=vi.get("tag"))), v)


def _jh1_include_long(orig, usable):
    """the long boots join the composition set AFTER the grid boots (plan order: they are last), so they override the grid
    boots of the same identity tag ('real_a4', 'llama') in the by-tag table"""
    jh = [(k, v) for k, v in usable.items() if v["job"] == "J-H1"]
    return dict([x for x in jh if not x[1].get("long_plan")] + [x for x in jh if x[1].get("long_plan")])


def _stop_after_long_death(orig, b, out, *a, **k):
    flag = Path(out) / ".stop_after_death"
    if flag.exists():
        return {"boot_id": b["boot_id"], "labels": ["SKIPPED_AFTER_DEATH"], "problems": [], "start": "skipped"}
    res = orig(b, out, *a, **k)
    if "SERVER_DEATH" in res.get("labels", []):
        flag.write_text("x")
    return res


def o_l15_bs48(_run=None, _A=None):
    """Decision 33 L-15: every J-C T1 grid boot (TP2 and the cap-48 anchor, D44 and 108v, both rounds = 8 boots) registers
    (48, 2k) and (48, 8k) with O 2,728 / 11,453 [코드 prior]; (48, 32k) is NOT registered; plan check clean."""
    bad = []
    p = R.make_plan("J-C")
    grid = [b for b in p["boots"] if b["kind"] == "T1T2" and not b["long"] and "n1c32" not in b["boot_id"]]
    if len(grid) != 8:
        bad.append(f"{len(grid)} J-C T1 grid boots != 8")
    for b in grid:
        cells = [tuple(c) for c in b["t1"]]
        for c, O in (((48, 2048), 2728), ((48, 8192), 11453)) if b["o_family"] == "JC|44" else (((48, 2048), 396), ((48, 8192), 396)):
            if c not in cells:
                bad.append(f"{b['boot_id']}: {c} missing")
            elif b["t1_O"].get(f"{c[0]}x{c[1]}") != O:
                bad.append(f"{b['boot_id']}: O{c} {b['t1_O'].get(f'{c[0]}x{c[1]}')} != {O}")
        if (48, 32768) in cells:
            bad.append(f"{b['boot_id']}: (48, 32k) registered (anchor pool cannot hold it; disclosed loss)")
        if len(cells) != 8:
            bad.append(f"{b['boot_id']}: {len(cells)} T1 cells != 8")
    bad += [f"plan: {q}" for q in R.plan_problems(p)]
    return bad


def o_l7_context(_run=None, _A=None):
    """Decision 33 L-7: the J-A boots with T1 / T2 / OFF cells (7 mode boots, cap-invariance, 2 OFF) have context 131,072,
    the T3 boots 73,728; every mode boot registers (8, 64k) and c_in + O = 65,370 + 14,089 = 79,459 < context [코드 prior]."""
    bad = []
    for rnd in (1, 2):
        p = R.make_plan("J-A", rnd)
        for b in p["boots"]:
            if b["long"]:
                continue
            want = 73728 if b["kind"] == "T3" else 131072
            if b["context"] != want:
                bad.append(f"{b['boot_id']}: context {b['context']} != {want}")
            if b["kind"] == "T1T2" and b["cap"] == 128 and b["t1"]:
                if [8, 65536] not in [list(c) for c in b["t1"]]:
                    bad.append(f"{b['boot_id']}: (8, 64k) not registered")
                elif b["t1_O"].get("8x65536") != 14089 or 65536 - 166 + 14089 >= b["context"]:
                    bad.append(f"{b['boot_id']}: (8, 64k) O {b['t1_O'].get('8x65536')} / c_in + O vs context {b['context']}")
        bad += [f"plan r{rnd}: {q}" for q in R.plan_problems(p)]
    return bad


def o_l7_run(run, A):
    """(8, 64k) at context 131,072 is feasible AND measured (a window exists) in the virtual run."""
    bad = []
    w = [x for x in (A["boots"]["x1_108v_l7"].get("t1_windows") or []) if (x["bs"], x["c"]) == (8, 65536)]
    f = [x for x in A["boots"]["x1_108v_l7"].get("feasibility", []) if (x["bs"], x["c"]) == (8, 65536)]
    if not f or not f[0]["run"]:
        bad.append(f"(8, 64k) feasibility row {f}")
    if not w or w[0].get("status") != "VALID":
        bad.append(f"(8, 64k) window {w and w[0].get('status')}")
    return bad


def o_l8_jh1_plan(_run=None, _A=None):
    """Decision 33 L-8: J-H1 ends with jh1_long_llama then jh1_long_real_a4 (vanilla 108v, cap 48, context 131,072, real
    weights, ascending 65,536 / 98,304 / 126,976), 7 measurement boots before them; Llama and Nano are in the SAME Job."""
    bad = []
    p = R.make_plan("J-H1")
    last = p["boots"][-2:]
    if [b["boot_id"] for b in last] != ["jh1_long_llama", "jh1_long_real_a4"]:
        bad.append(f"J-H1 tail {[b['boot_id'] for b in p['boots'][-3:]]}")
        return bad
    for b, model in zip(last, ("llama", "nano")):
        got = (b["model"], b["dummy"], b["engine"], b["mode"], b["cap"], b["context"], b["long"], b["tp"])
        want = (model, False, "vanilla", "108v", 48, 131072, [65536, 98304, 126976], 1)
        if got != want:
            bad.append(f"{b['boot_id']}: {got} != {want}")
    if sum(1 for b in p["boots"] if not b["long"]) != 7:
        bad.append("J-H1 measurement boots != 7")
    bad += [f"plan: {q}" for q in R.plan_problems(p)] + [f"registry: {q}" for q in R.registry_problems(p)]
    return bad


def o_l8_jh1_join(run, A):
    """Analysis: each J-H1 long boot joins only the grid boot of its own model tag; the composition tests use the grid boots
    only (one real_a4 and one llama entry); the Llama / Nano long contrast has the 3 L from the SAME Job."""
    bad = []
    t2l = A["tables"]["t2_long"]
    ll = next((e for k, e in t2l.items() if e.get("tag") == "llama"), None)
    nn = next((e for k, e in t2l.items() if e.get("tag") == "real_a4"), None)
    if not ll or not nn:
        return [f"long tables {sorted(t2l)}"]
    if sorted(ll["joins"]) != ["jh1_llama"]:
        bad.append(f"Llama long joins {sorted(ll['joins'])} != ['jh1_llama']")
    if sorted(nn["joins"]) != ["jh1_real_a4"]:
        bad.append(f"real-a4 long joins {sorted(nn['joins'])} != ['jh1_real_a4']")
    for e in (ll, nn):
        if [r["status"] for r in e["rows"]] != ["VALID"] * 3:
            bad.append(f"{e['tag']} long rows {[r['status'] for r in e['rows']]}")
    jh = A.get("jh1") or {}
    if sorted(jh.get("boots", [])) != sorted(["dummy_a4", "dummy_a13", "dummy_a22", "dummy_a31", "real_a4", "llama"]):
        bad.append(f"J-H1 composition boots {jh.get('boots')}")
    wi = jh.get("w_indep") or {}
    if [(wi.get(t) or {}).get("label") for t in ("T1", "T2")] != ["WEIGHT_INDEP_HELD"] * 2:
        bad.append(f"w_indep (dummy vs real a4, grid boots) {wi}")
    lc = (jh.get("long_contrast") or {}).get("ratio_llama_over_nano")
    if not lc or sorted(lc) != ["126976", "65536", "98304"]:
        bad.append(f"long contrast {lc}")
    return bad


def o_l14_plan(_run=None, _A=None):
    """Decision 33 L-14: one cap-32 N=1 anchor per round: GPU 0 only, sticky D44, ONLY (32, 32k), 2 windows, its own O
    15,502 [코드 prior] (the TP2 boots keep 30,592), pool (cap 32) holds it, the cap-48 anchor does not."""
    bad = []
    p = R.make_plan("J-C")
    c32 = {b["boot_id"]: b for b in p["boots"] if b["tp"] == 1 and b["cap"] == 32}
    if sorted(c32) != ["jc1_n1c32_D44", "jc2_n1c32_D44"]:
        return [f"cap-32 anchors {sorted(c32)}"]
    for k, b in c32.items():
        got = (b["gpus"], b["engine"], b["D"], [list(c) for c in b["t1"]], b["t1_reps"], b["t1_O"].get("32x32768"), b["round"])
        want = ("0", "pdmux_sticky", 44, [[32, 32768]], 2, 15502, int(k[2]))
        if got != want:
            bad.append(f"{k}: {got} != {want}")
    for b in p["boots"]:
        if b["tp"] == 2 and b["mode"] == "D44" and not b["long"] and b["kind"] == "T1T2":
            if b["t1_O"].get("32x32768") != 30592:
                bad.append(f"{b['boot_id']}: TP2 O {b['t1_O'].get('32x32768')} != 30592")
    pool32 = 1728096
    if 32 * (32768 - 166 + 15502) > 0.95 * pool32:
        bad.append("independent check: cap-32 pool cannot hold the cell")
    if not S.feasible_cell(32, 32768, 15502, 1588000, 48, True):
        bad.append("independent check: cap-48 anchor should NOT hold (32, 32k)")
    bad += [f"plan: {q}" for q in R.plan_problems(p)] + [f"registry: {q}" for q in R.registry_problems(p)]
    return bad


def o_l14_analysis(run, A):
    """The cap-48 pairs keep their ratios (fake tp_dec_mult 0.62 over bs 8/32/48); the cap-32 anchor has its OWN table with
    both actual ctx and both O (30,592 vs 15,502) and is never the anchor of the pairs."""
    bad = []
    jc = A["jc"]
    rs = jc["ratios_t1"].get("r1|D44") or {}
    if not rs or any(not (abs(x["ratio"] / 0.62 - 1) < 1e-6) for x in rs.values()):
        bad.append(f"cap-48 pair ratios r1|D44 {sorted(rs)[:4]}")
    if sorted(rs) != sorted(["8x2048", "8x8192", "8x32768", "32x2048", "32x8192", "48x2048", "48x8192"]):
        bad.append(f"cap-48 pair cells {sorted(rs)} (7: the 8-cell grid minus the anchor-infeasible (32, 32k))")
    a32 = (jc.get("ratios_t1_anchor32") or {}).get("r1|D44", {}).get("32x32768")
    if not a32 or a32["O_tp2"] != 30592 or a32["O_n1"] != 15502 or not a32["ctx_tp2"] or not a32["ctx_n1"]:
        bad.append(f"anchor32 row {a32}")
    return bad


def o_l46_plan(_run=None, _A=None):
    """Decision 33 (TP2 PD-mux path): J-C ends with jc_tp2_long_108v then jc_tp2_long_108u (PD-mux engine, sticky OFF, TP2 on
    GPUs 0,1, cap 48, context 131,072, same L); no other TP2 long boot; +registry clean."""
    bad = []
    p = R.make_plan("J-C")
    last = p["boots"][-2:]
    if [b["boot_id"] for b in last] != ["jc_tp2_long_108v", "jc_tp2_long_108u"]:
        return [f"J-C tail {[b['boot_id'] for b in p['boots'][-3:]]}"]
    got = [(b["engine"], b["mode"], b["D"], b["tp"], b["gpus"], b["cap"], b["context"], b["long"]) for b in last]
    want = [("vanilla", "108v", None, 2, "0,1", 48, 131072, [65536, 98304, 126976]),
            ("pdmux_108u", "108u", None, 2, "0,1", 48, 131072, [65536, 98304, 126976])]
    if got != want:
        bad.append(f"{got} != {want}")
    bad += [f"plan: {q}" for q in R.plan_problems(p)] + [f"registry: {q}" for q in R.registry_problems(p)]
    return bad


def o_long_suffix_rule(_run=None, _A=None):
    """The generalised plan rule (A-47): a measurement boot between / after long boots is refused (both with several long
    boots and with a long boot far from the end), while the registered suffix passes."""
    bad = []
    for job in ("J-C", "J-H1"):
        p = R.make_plan(job)
        if R.plan_problems(p):
            bad.append(f"{job}: registered plan refused: {R.plan_problems(p)[:1]}")
        q = R.make_plan(job)
        n = sum(1 for b in q["boots"] if b["long"])
        meas = next(b for b in q["boots"] if not b["long"])
        q["boots"].remove(meas)
        q["boots"].insert(len(q["boots"]) - n + 1, meas)         # a measurement boot BETWEEN the long boots
        if not any("LAST" in x for x in R.plan_problems(q)):
            bad.append(f"{job}: plan check accepted a measurement boot between long boots")
        q = R.make_plan(job)
        first_long = next(i for i, b in enumerate(q["boots"]) if b["long"])
        q["boots"].insert(first_long - 3, q["boots"].pop(first_long))      # a long boot mid-Job
        if not any("LAST" in x for x in R.plan_problems(q)):
            bad.append(f"{job}: plan check accepted a mid-Job long boot")
    return bad


def o_long_death2(run, A):
    """A death in the first TP2 long boot does not skip the second: both boots ran, both keep 65,536 / 98,304 VALID rows and
    label 126,976 SERVER_DEATH."""
    bad = []
    for bid in ("jc_tp2_long_108v", "jc_tp2_long_108u"):
        d = Path(run) / "boots" / bid
        if not (d / "DONE").exists() or not (d / "cells.jsonl").exists():
            bad.append(f"{bid}: not run (DONE / cells missing)")
            continue
        rows = {r["L"]: r for r in (A["tables"]["t2_long"].get(bid) or {}).get("rows", [])}
        if [rows.get(L, {}).get("status") for L in (65536, 98304)] != ["VALID", "VALID"]:
            bad.append(f"{bid}: rows {[rows.get(L, {}).get('status') for L in (65536, 98304)]}")
        if "SERVER_DEATH" not in rows.get(126976, {}).get("labels", []):
            bad.append(f"{bid}: 126,976 not SERVER_DEATH")
    return bad


MUTANTS = [
    # id, PREREG candidate (§11-2 / §12 text), stage, scenario, patch factory, oracle, plan subset
    ("M01", "cap 48 부팅인데 128로 기록", "run", "JAF", lambda: mock.patch.object(X, "logged_cap", lambda f: S.CAP_MAIN),
     o_cap48, ("J-A", ["ja1_capinv_D44", "ja1_D44"])),
    ("M02a", "캡처 목록 밖 bs를 통과 (실행 가능성)", "unit", None,
     lambda: mock.patch.object(X, "bs_capturable", lambda bs, cl: True), o_capture_feasibility, None),
    ("M02b", "캡처 목록 밖 bs를 통과 (ABORT_CG_BS)", "run", "CG", lambda: mock.patch.object(X, "capture_problems", lambda *a, **k: []),
     o_cg_abort, None),
    ("M03a", "ctx 띠 off-by-one (좁음)", "unit", None, lambda: mock.patch.object(S, "CTX_BAND", S.CTX_BAND - 1), o_ctx_band, None),
    ("M03b", "ctx 띠 off-by-one (넓음)", "unit", None, lambda: mock.patch.object(S, "CTX_BAND", S.CTX_BAND + 1), o_ctx_band, None),
    ("M04", "INFEASIBLE_BY_RULE 셀을 조용히 누락", "run", "JH1", lambda: _wrap(X, "feasibility", _drop_infeasible),
     o_infeasible_rows, ("J-H1", ["jh1_llama"])),
    ("M05", "T2에 첫 요청 워밍업 포함", "analysis", "JH1", lambda: _wrap(X, "t2_value", _t2_with_warmup), o_t2_warmup, None),
    ("M06", "Llama 셀을 Nano로 라벨", "analysis", "JH1", lambda: _wrap(R, "boot_identity", _identity_ignoring_arch),
     o_model_labels, None),
    ("M07", "108u와 108v 합침", "analysis", "JAF", lambda: _wrap(R, "boot_identity", _identity_merge108), o_modes_separate, None),
    ("M08", "클라이언트 토큰 간격을 엔진 S_eng로 표기", "analysis", "JH1F", lambda: _wrap(X, "t1_window", _t1_client_as_seng),
     o_seng_is_engine, None),
    ("M09", "decode_fwd_host_ms를 decode_fwd_gpu_ms로 대체", "analysis", "JH1F",
     lambda: mock.patch.object(X, "I4_FIELD", "decode_fwd_host_ms"), o_gpu_field, None),
    ("M10", "제외 round(SM 미실현 boot)를 적합에 사용", "analysis", "JAF", lambda: _wrap(R, "analyze_boot", _usable_always),
     o_excluded_boot, None),
    ("M11", "대기 요청을 running으로 셈(T4)", "analysis", "JB2", lambda: _wrap(X, "t4_fill", _waiting_running), o_n_real, None),
    ("M12", "T4 예약 R을 0으로 둠(원식 회귀)", "unit", None, lambda: _wrap(X, "cmem_predictions", _r_zero), o_prereg_table, None),
    ("M13", "n_A를 비대칭 형식으로 계산", "unit", None, lambda: _wrap(X, "cmem_predictions", _na_asym), o_prereg_table, None),
    ("M14", "불일치 bs를 제외하지 않음(§1-7)", "analysis", "JH1F", lambda: _wrap(F, "client_secondary_table", _keep_mismatch),
     o_mismatch_excluded, None),
    ("M15", "요청 SM을 실현 SM으로 기록", "run", "JAF", lambda: _wrap(X, "realized_sm", _requested_sm), o_realized_axis,
     ("J-A", ["ja1_D34", "ja1_D24"])),
    ("M16", "dummy·real 라벨 뒤바꿈", "analysis", "JH1F", lambda: _wrap(R, "boot_identity", _identity_plan_loadformat),
     o_weight_sign, None),
    ("M17", "OFF 진단 boot에 계측 env가 남음", "run", "JAF", lambda: _wrap(R, "boot_env", _env_leak), o_off_clean,
     ("J-A", ["ja1_off_108v"])),
    ("M18", "OFF 진단 비로 T1 값을 보정", "analysis", "JAF", lambda: _wrap(R, "analyze_boot", _off_correction),
     o_off_not_applied, None),
    ("M19", "T3 셀을 T1 boot에서 실행(TEL 비대칭)", "unit", None, lambda: _wrap(R, "plan_problems", _no_t3_check),
     o_t3_rejected, None),
    ("M20", "T4 산식 예측을 채우기 뒤에 기록(봉인 순서)", "run", "JB2", lambda: _wrap(R, "run_boot", _late_prediction), o_sealed,
     ("J-B2", ["jb2_cap48"])),
    # decision 28 (rev3.4)
    ("M21", "창을 첫 decode부터 시작(ramp 미인식)", "analysis", "JA0", lambda: _wrap(X, "first_reach_index", _reach_first_decode),
     o_window_reach, None),
    ("M22", "명목 c로 적합", "analysis", "JA0", lambda: _wrap(R, "fit_ctx", _nominal_ctx), o_fit_truth, None),
    ("M23", "취소 누락으로 다음 셀에 잔여 요청", "run", "JA0", lambda: _wrap(C, "cancel_burst", _no_cancel), o_no_residual,
     ("J-A", ["ja1_D44"])),
    ("M24", "T2를 sticky-D boot에서 P=108−D로 라벨", "unit", None, lambda: _wrap(R, "t2_axis", _t2_axis_d), o_t2_p108, None),
    ("M25", "T2′ 동거 미확인을 확인으로 라벨", "analysis", "JA0", lambda: _wrap(X, "t2p_coexist", _always_confirmed),
     o_t2p_confirm, None),
    ("M26", "ramp 상한 없이 O = 332 고정(rev3.3 창)", "run", "JA0",
     lambda: _wrap(C, "run_t1_cell", _o_base_only), o_window_reach, ("J-A", ["ja1_D44", "ja1_108u"])),
    # stage 2 (rev3.5 draft): stage 1's deferred three + T3 + J-C + Job entry
    ("M27", "정지(>5× 중앙값) 계산에서 중앙값을 동거 창으로 잡음", "analysis", "T3S",
     lambda: _wrap(X, "stall_stats", _stall_co_median), o_t3_rho_stall, None),
    ("M28", "σ′와 σ를 함께 곱함", "analysis", "T3S", lambda: _wrap(R, "t3_join", _sigma_times_sigma), o_t3_sigma_engine, None),
    ("M29", "TP2 앵커를 cap 128로 부팅", "unit", None, lambda: _wrap(R, "plan_jc", _anchor_cap128), o_jc_anchor, None),
    ("M30", "σ를 클라이언트 TTFT로 계산", "analysis", "T3S", lambda: _wrap(X, "sigma_prime", _sigma_client),
     o_t3_sigma_engine, None),
    ("M31", "σ′의 동거 창과 단독(기준) 창을 다른 boot에서 가져옴", "analysis", "T3S",
     lambda: _wrap(R, "t3_join", _sigma_other_boot), o_t3_sigma_engine, None),
    ("M32", "ρ의 동거 창과 단독 창을 다른 boot에서 가져옴", "analysis", "T3S",
     lambda: _wrap(R, "t3_join", _rho_other_boot), o_t3_rho_stall, None),
    ("M33", "n_it_obs를 환산 추정(T_span/T_iter)으로 대체", "analysis", "T3S", lambda: _wrap(X, "n_it_in", _n_it_converted),
     o_t3_rho_stall, None),
    ("M34", "T3 단독 창을 첫 decode부터 시작(합류 확인 없음)", "analysis", "T3S", lambda: _wrap(X, "t3_cell", _t3_first_decode),
     o_t3_window, None),
    ("M35", "T3 decode 출력 길이에 ramp·동거 예산 없음", "run", "T3S", lambda: _wrap(C, "run_t3_cell", _t3_o_small),
     o_t3_window, ("J-A", ["ja1_t3_D44"])),
    ("M36", "smoke 출력에 SMOKE_UNSEALED 표지 누락", "unit", None, lambda: _wrap(R, "run_job", _no_smoke_marker), o_smoke, None),
    ("M37", "재개 시 완료 셀을 버리고 boot 전체 재실행(재개 미기록)", "unit", None,
     lambda: _wrap(R, "resume_segments", _no_resume), o_resume, None),
    ("M38", "원자료를 results/로 export", "run", "JC", lambda: _wrap(R, "export_summary", _export_all),
     o_raw_not_exported, ("J-C", ["jc_tp2_t4"])),
    # decision 30 (rev3.5): long points restored in separate last boots
    ("M39", "장문 boot를 Job 중간에 배치", "unit", None, lambda: _wrap(R, "plan_ja", _long_mid_job), o_long_placement, None),
    ("M40", "SERVER_DEATH 시 앞 결과 삭제", "run", "LONGDEATH", lambda: _wrap(R, "run_boot", _wipe_on_death), o_long_death,
     ("J-A", ["ja1_off_108v", "ja1_long_108v"])),
    ("M41a", "context 73,728 boot에서 96k 시도(계획: grid boot에 장문 점)", "unit", None,
     lambda: _wrap(R, "plan_ja", _long_in_grid_boot), o_long_placement, None),
    ("M41b", "context 73,728 boot에서 96k 시도(실행: 로그 context 검사 생략)", "run", "LONGCTX",
     _no_runtime_context_check, o_long_ctx_guard, ("J-A", ["ja1_long_108v"])),
    # decision 33 (rev3.7): the five restored §11-4 points (the oracles' expected values are written down independently)
    ("M42a", "J-C bs 48 행을 TP2에만 등록(앵커 누락)", "unit", None, lambda: _wrap(R, "plan_jc", _bs48_tp2_only), o_l15_bs48, None),
    ("M42b", "J-C bs 48 행을 조용히 뺌", "unit", None, lambda: _wrap(R, "plan_jc", _bs48_nowhere), o_l15_bs48, None),
    ("M43a", "J-A T1 boot context를 73,728로 되돌림(8, 64k 불가)", "unit", None, lambda: _wrap(R, "plan_ja", _ja_ctx_73728),
     o_l7_context, None),
    ("M43b", "(8, 64k) 셀을 실행에서 조용히 누락", "run", "JAL7", lambda: _wrap(R, "registered_cells", _drop_8x64k_cell),
     o_l7_run, None),
    ("M43c", "실행 가능성 재평가가 boot context를 무시(기본 73,728로 판정)", "run", "JAL7",
     lambda: _wrap(S, "feasible_cell", _feas_default_context), o_l7_run, None),
    ("M44a", "J-H1 Llama 장문 boot 누락", "unit", None, lambda: _wrap(R, "plan_jh1", _drop_boot_noargs("jh1_long_llama")),
     o_l8_jh1_plan, None),
    ("M44b", "J-H1 real-a4 장문 boot 누락(Llama만: ④ 대비 불가)", "unit", None,
     lambda: _wrap(R, "plan_jh1", _drop_boot_noargs("jh1_long_real_a4")), o_l8_jh1_plan, None),
    ("M44c", "J-H1 장문 boot를 Job 중간에 배치", "unit", None, lambda: _wrap(R, "plan_jh1", _jh1_long_mid), o_l8_jh1_plan, None),
    ("M44d", "장문 boot를 모델 태그 무관하게 격자 boot에 연결", "analysis", "JH1", lambda: _wrap(R, "long_join_ok", _long_join_ignores_tag),
     o_l8_jh1_join, None),
    ("M44e", "J-H1 조성 검정에 장문 boot를 포함", "analysis", "JH1", lambda: _wrap(R, "jh1_grid_boots", _jh1_include_long),
     o_l8_jh1_join, None),
    ("M45a", "J-C cap 32 앵커 boot 누락", "unit", None, lambda: _wrap(R, "plan_jc", _drop_anchor32), o_l14_plan, None),
    ("M45b", "cap 32 앵커가 TP2 공통 O(30,592)를 씀(자체 O 아님)", "unit", None, lambda: _wrap(R, "plan_jc", _anchor32_tp2_O),
     o_l14_plan, None),
    ("M45c", "cap 32 앵커를 cap 48 쌍의 앵커로 덮어씀", "analysis", "JC", lambda: _wrap(R, "is_anchor32", _anchor32_not_special),
     o_l14_analysis, None),
    ("M46a", "TP2 108u 장문 boot 누락", "unit", None, lambda: _wrap(R, "plan_jc", _drop_boot_noargs("jc_tp2_long_108u")),
     o_l46_plan, None),
    ("M46b", "TP2 108u 장문 boot를 Job 중간에 배치", "unit", None, lambda: _wrap(R, "plan_jc", _jc_108u_mid), o_l46_plan, None),
    ("M46c", "TP2 108u 장문 boot를 sticky D44로 라벨", "unit", None, lambda: _wrap(R, "plan_jc", _jc_108u_sticky), o_l46_plan, None),
    ("M47", "장문 boot 묶음 검사가 마지막 boot만 봄(묶음 사이·앞쪽 측정 boot 통과)", "unit", None,
     lambda: _wrap(R, "plan_problems", _plan_check_last_boot_only), o_long_suffix_rule, None),
    ("M48", "한 장문 boot의 SERVER_DEATH 뒤 다음 장문 boot를 건너뜀", "run", "LONGDEATH2",
     lambda: _wrap(R, "run_boot", _stop_after_long_death), o_long_death2, ("J-C", ["jc_tp2_long_108v", "jc_tp2_long_108u"])),
]


@contextlib.contextmanager
def applied(factory):
    with factory():
        yield


def cg_plan():
    b = R._boot("J-TEST", 1, "cg_D44", "T1T2", "nano", "pdmux_sticky", S.CAP_MAIN, D=44, t1=[(8, 2048)])
    return {"job": "J-TEST", "deferred": [], "boots": [b]}
