#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- mutation ledger (lesson 53: a check is only evidence if a reverted / broken variant
FAILS it).  Each mutant re-creates one PREREG §11-2 / §12 candidate defect by patching ONE harness function or
constant; each oracle encodes the expected behaviour INDEPENDENTLY of the patched code (fake-engine ground truth,
raw log text, the PREREG's own pre-data table).  tests/test_costterms_harness.py asserts: oracle passes on the
unmutated harness and fails under every mutant.

stage: 'unit' (synthetic inputs), 'analysis' (re-analyse a stored virtual run), 'run' (re-run a small virtual plan).
Stage-2 candidates (TP2 anchor cap, stall median in the co-residence window, sigma' x sigma) are listed in DEFERRED.
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

DEFERRED = {
    "TP2 앵커를 cap 128로 부팅": "J-C (stage 2)",
    "정지(>5x 중앙값) 계산에서 중앙값을 동거 창으로 잡음": "T3 (stage 2)",
    "σ′와 σ를 함께 곱함": "T3 (stage 2)",
}

# ---------------------------------------------------------------- scenarios (virtual runs; truth = fake knobs)
FAST_PF = {"pf_u": 0.5, "pf_v": 1e-5, "pf_w4": 0.0}
SCENARIOS = {
    "JH1": {"job": "J-H1", "truth": {}},
    "JH1F": {"job": "J-H1", "truth": {"weight_dependent": 0.05, "client_drift": 0.05, "client_drift_bs": 32,
                                      "i4_unresolved_every": 40}, "boots": ["jh1_dummy_a4", "jh1_real_a4"]},
    "JAF": {"job": "J-A", "truth": dict(FAST_PF, realize={"34": 36, "24": 16})},
    "JB2": {"job": "J-B2", "truth": {}},
    "CG": {"job": "J-TEST", "truth": {"capture_max": 48}},        # plan: cg_plan() (one cap-128 boot)
    "JA0": {"job": "J-A", "truth": {}, "boots": ["ja1_D44", "ja1_108v"]},   # default prefill speed: A-1 ramp
}


def run_scenario(name, out, variant_src, plan=None, truth=None):
    sc = SCENARIOS[name]
    tr = sc["truth"] if truth is None else truth
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


def synth_window(c, c_in, bs=4, n=S.O_TOKENS, step=0.01):
    """Synthetic all-running burst: record k at t=k*step, every request has k-1 tokens at record k (ctx_k = c_in+k-1);
    the extractor's window keeps k = TRIM+1 .. TRIM+WINDOW+1 so ctx p50 = c_in + TRIM + WINDOW/2."""
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
    rows = X.feasibility([(64, 2048), (8, 2048)], f, 128)
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


def o_ctx_band(_run=None, _A=None):
    bad = []
    c = 8192
    for delta, want in ((64, "VALID"), (-64, "VALID"), (65, "CTX_UNREALIZED"), (-65, "CTX_UNREALIZED")):
        c_in = c + delta - (S.TRIM_ITERS + S.WINDOW_ITERS / 2)
        recs, cell = synth_window(c, int(c_in) if float(c_in).is_integer() else c_in)
        w = X.t1_window(recs, cell, [1, 2, 4, 8])
        p50 = w.get("ctx_p50")
        if w["status"] != want:
            bad.append(f"ctx p50 {p50} (c{delta:+d}) -> {w['status']} != {want}")
    return bad


def o_infeasible_rows(run, A):
    bad = []
    for k, v in A["tables"]["t1"].items():
        if k.split("|")[0] == "J-H1" and v["n_registered"] != len(S.JH_T1_CELLS):
            bad.append(f"{k}: {v['n_registered']} feasibility rows != {len(S.JH_T1_CELLS)} registered cells")
    ll = [v for k, v in A["tables"]["t1"].items() if "|llama|" in k]
    if ll:
        kv = raw_kv_tokens(run, "jh1_llama")
        want = sum(1 for b, c in S.JH_T1_CELLS if b * (c + S.O_PLUS_X) > 0.95 * kv)
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
]


@contextlib.contextmanager
def applied(factory):
    with factory():
        yield


def cg_plan():
    b = R._boot("J-TEST", 1, "cg_D44", "T1T2", "nano", "pdmux_sticky", S.CAP_MAIN, D=44, t1=[(8, 2048)])
    return {"job": "J-TEST", "deferred": [], "boots": [b]}
