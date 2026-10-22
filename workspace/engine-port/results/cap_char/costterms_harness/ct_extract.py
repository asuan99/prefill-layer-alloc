#!/usr/bin/env python3
"""Cost-term harness, stage 1 -- PURE extractors (no I/O except the small readers at the top, no clock, no GPU).

  boot log  -> facts (KV pool, mamba slots/bytes, weights, capture list, server args, sticky index, probe banners,
               new_token_ratio [derived], split_forward_token_budget [config file])
  facts     -> feasibility of every registered cell (INFEASIBLE_BY_RULE is a row, never a silent omission)
  green     -> realized SM check (§1-8)
  facts     -> T4 predictions A and B (written and hashed BEFORE the fill, ct_runner)
  decode_forward records + client cell record -> T1 window, S_eng, client gap, ctx band, I-4 resolution
  client    -> T2 / T2' TTFT, T4 fill (n_real, FILL_INCOMPLETE, reason shares)

Window rule (PREREG rev3.4 §1-1, decision 28 A-1, ramp-aware): r0 = the first decode_forward record of the cell with
bs_real == bs; the window = records r0 + TRIM_ITERS .. r0 + TRIM_ITERS + WINDOW_ITERS (301 records, consecutive seq =
WINDOW_ITERS intervals).  Interval k (record k -> k + 1) is EXCLUDED and counted when record k has bs_real != bs, mode
!= decode (mixed = prefill tokens in the forward) or fewer than bs cell requests had their first token by t_host_start
(client-side join confirmation -- the only co-residence evidence with TEL OFF; in PD-mux a request enters the decode
batch only after its split-prefill span ends, so a joined cell cannot have one of its own prefills in flight).  S_eng =
median of the included t_host_start differences (ms).  Actual ctx = mean over the included interval-start records of
the batch-mean (c_in + tokens the client had received by t_host_start); it is the fit input, the nominal c is the
cell label.  Foreign requests (a previous cell not cancelled) = a decode_forward whose bs_real exceeds min(bs, cell
requests joined by t_host_start + JOIN_SLACK_S) -- e.g. a decode step before the cell's first token.
"""
from __future__ import annotations

import ast
import bisect
import json
import math
import re
from pathlib import Path

import ct_spec as S

GIB = 1024 ** 3
DECODE_SCHEMA = "pdmux.decode_probe/v1"
ADMISSION_SCHEMA = "pdmux.admission/v1"
I4_FIELD = "decode_fwd_gpu_ms"          # I-4 GPU time (decode_probe.I4_FIELD)
I4_HOST_FIELD = "decode_fwd_host_ms"    # host bracket of the same call -- never a substitute for I4_FIELD


# ============================================================================ small readers / stats
def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append({"event": "_unparsable"})
    return out


def quantile(xs, q):
    """numpy 'linear' quantile (N2 convention)."""
    s = sorted(xs)
    if not s:
        return None
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs):
    return quantile(xs, 0.5)


# ============================================================================ boot log
_SA = {  # server_args field -> regex (the ServerArgs(...) repr line)
    "model_path": r"model_path='([^']*)'", "load_format": r"load_format='([^']*)'",
    "revision": r"revision='([^']*)'", "max_running_requests": r"max_running_requests=([0-9]+)",
    "max_mamba_cache_size": r"max_mamba_cache_size=([0-9]+)", "chunked_prefill_size": r"chunked_prefill_size=(-?[0-9]+)",
    "max_prefill_tokens": r"max_prefill_tokens=([0-9]+)", "schedule_conservativeness": r"schedule_conservativeness=([0-9.]+)",
    "mem_fraction_static": r"mem_fraction_static=([0-9.]+)", "context_length": r"context_length=([0-9]+)",
    "enable_pdmux": r"enable_pdmux=(True|False)", "pdmux_config_path": r"pdmux_config_path='([^']*)'",
    "cuda_graph_max_bs": r"cuda_graph_max_bs=([0-9]+)", "disable_radix_cache": r"disable_radix_cache=(True|False)",
    "attention_backend": r"attention_backend='([^']*)'", "disable_cuda_graph": r"disable_cuda_graph=(True|False)",
    "random_seed": r"random_seed=([0-9]+)", "tp_size": r"tp_size=([0-9]+)",
}


def _conv(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_boot_log(text, env=None, config_text=None):
    """Facts from one boot log.  `env` (the boot env actually passed) supplies SGLANG_INIT_NEW_TOKEN_RATIO; the PD-mux
    config text supplies split_forward_token_budget (not logged).  Missing lines give None, never a default."""
    text = text or ""
    f = {}
    sa_line = next((ln for ln in text.splitlines() if "server_args=ServerArgs(" in ln), "")
    f["server_args"] = {k: (_conv(m.group(1)) if (m := re.search(p, sa_line)) else None) for k, p in _SA.items()}
    m = re.findall(r"max_total_num_tokens=([0-9]+), chunked_prefill_size=(-?[0-9]+), max_prefill_tokens=([0-9]+), "
                   r"max_running_requests=([0-9]+), context_len=([0-9]+)", text)
    if m:
        a = m[-1]
        f["final"] = {"max_total_num_tokens": int(a[0]), "chunked_prefill_size": int(a[1]),
                      "max_prefill_tokens": int(a[2]), "max_running_requests": int(a[3]), "context_len": int(a[4])}
    else:
        f["final"] = None
    m = re.findall(r"KV Cache is allocated\. #tokens: *([0-9]+), *K size: *([0-9.]+) *GB, *V size: *([0-9.]+) *GB", text)
    f["kv_cache_tokens"], f["kv_k_gib"], f["kv_v_gib"] = (int(m[-1][0]), float(m[-1][1]), float(m[-1][2])) if m else (None, None, None)
    m = re.findall(r"Mamba Cache is allocated\. *max_mamba_cache_size: *([0-9]+), *conv_state size: *([0-9.]+) *GB"
                   r"(?:, *ssm_state size: *([0-9.]+) *GB)?", text)
    if m:
        f["max_mamba_cache_size"], f["mamba_conv_gib"] = int(m[-1][0]), float(m[-1][1])
        f["mamba_ssm_gib"] = float(m[-1][2]) if m[-1][2] else None
    else:
        f["max_mamba_cache_size"] = f["mamba_conv_gib"] = f["mamba_ssm_gib"] = None
    m = re.findall(r"Load weight end\..*?type=([A-Za-z0-9_]+).*?mem usage=([0-9.]+) *GB", text)
    f["model_arch"], f["weight_gib"] = (m[-1][0], float(m[-1][1])) if m else (None, None)
    m = re.findall(r"Capture cuda graph bs \[([0-9, ]*)\]", text)
    f["capture_bs"] = [int(x) for x in re.findall(r"[0-9]+", m[-1])] if m else None
    m = re.findall(r"PD-Multiplexing enabled with ([0-9]+) stream groups, sm_counts \(prefill_sm, decode_sm\): (\[.*?\])\s*$",
                   text, re.M)
    f["pdmux_sm_counts"] = [tuple(x) for x in ast.literal_eval(m[-1][1])] if m else None
    hits = re.findall(r"PD-mux sticky partition ENABLED.*?\(fixed target index=([0-9]+|None)\)", text)
    f["sticky_enabled_lines"] = len(re.findall(r"PD-mux sticky partition ENABLED", text))
    f["sticky_index"] = (None if hits[-1] == "None" else int(hits[-1])) if hits else None
    f["decode_probe_enabled"] = "decode probe ENABLED" in text
    f["admission_probe_enabled"] = "admission probe ENABLED" in text
    m = re.findall(r"PD-mux green read-out: (\{.*\})\s*$", text, re.M)
    try:
        f["green_from_log"] = ast.literal_eval(m[-1]) if m else None
    except (ValueError, SyntaxError):
        f["green_from_log"] = None
    # new_token_ratio: NOT logged by SGLang v0.5.10; scheduler.py:960 init_new_token_ratio =
    # min(SGLANG_INIT_NEW_TOKEN_RATIO (default 0.7) * schedule_conservativeness, 1.0)  [derived, ambiguity A-4]
    env = env or {}
    cons = f["server_args"].get("schedule_conservativeness")
    base = float(env.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7))
    f["init_new_token_ratio_derived"] = min(base * cons, 1.0) if cons is not None else None
    f["init_new_token_ratio_source"] = "derived: min(SGLANG_INIT_NEW_TOKEN_RATIO x schedule_conservativeness, 1)"
    sfb = None
    if config_text:
        m = re.search(r"^split_forward_token_budget:\s*([0-9]+)", config_text, re.M)
        sfb = int(m.group(1)) if m else None
    f["split_forward_token_budget"] = sfb
    f["split_forward_token_budget_source"] = "pdmux config file" if config_text else None
    f["retract_total"] = sum(int(x) for x in re.findall(r"#retracted_reqs: *([0-9]+)", text))
    return f


def mem_terms(f):
    """(pool_bytes, s, k, slots, kv_tokens) from banner facts.  s = mamba bytes / (cap + 1 padding slot); s = 0 when
    the model has no mamba pool (Llama: no banner) or the banner says 0.00 GB (a31).  None if the KV banner is missing."""
    if not f.get("kv_cache_tokens") or f.get("kv_k_gib") is None:
        return None
    kv_bytes = (f["kv_k_gib"] + f["kv_v_gib"]) * GIB
    k = kv_bytes / f["kv_cache_tokens"]
    mamba, s, slots = 0.0, 0.0, None
    if f.get("max_mamba_cache_size") is not None:
        slots = f["max_mamba_cache_size"] + 1
        mamba = ((f.get("mamba_conv_gib") or 0.0) + (f.get("mamba_ssm_gib") or 0.0)) * GIB
        s = mamba / slots
    return {"pool_bytes": kv_bytes + mamba, "s": s, "k": k, "slots": slots, "kv_tokens": f["kv_cache_tokens"],
            "kv_bytes": kv_bytes, "mamba_bytes": mamba}


def logged_cap(f):
    """The cap the server actually ran with (final line max_running_requests; server_args as fallback)."""
    if f.get("final"):
        return f["final"]["max_running_requests"]
    return f["server_args"].get("max_running_requests")


# ============================================================================ boot checks
def capture_problems(capture, required=S.CAPTURE_BS_REQUIRED):
    """ABORT_CG_BS (§3-4) for cap-128 boots: every required bs must be in the logged capture list."""
    if capture is None:
        return ["no 'Capture cuda graph bs [...]' line"]
    miss = [b for b in required if b not in capture]
    return [f"required bs {miss} not in capture list {capture}"] if miss else []


def padded_bs(capture, bs):
    """§3-4 bs_padded rule: smallest list element >= bs (bisect_left); above the max -> eager (bs)."""
    if not capture:
        return bs
    i = bisect.bisect_left(capture, bs)
    return capture[i] if i < len(capture) else bs


def label_problems(boot, f):
    """Plan label vs boot log: model architecture, load format (dummy/real), cap, PD-mux on/off, variant layer counts.
    Any problem aborts the boot (the harness never relabels from the plan)."""
    bad = []
    want_arch = S.ARCH_BY_MODEL[boot["model"]]
    if f.get("model_arch") != want_arch:
        bad.append(f"ABORT_MODEL_LABEL: log arch {f.get('model_arch')} != plan {want_arch}")
    lf = f["server_args"].get("load_format")
    want_lf = "dummy" if boot.get("dummy") else "auto"
    if lf != want_lf:
        bad.append(f"ABORT_LOAD_FORMAT_LABEL: log load_format {lf!r} != plan {want_lf!r}")
    cap = logged_cap(f)
    if cap != boot["cap"]:
        bad.append(f"ABORT_CAP_MISMATCH: log cap {cap} != plan {boot['cap']}")
    if boot["model"] == "nano" and f.get("max_mamba_cache_size") not in (None, boot["cap"]):
        bad.append(f"ABORT_CAP_MISMATCH: mamba slots {f.get('max_mamba_cache_size')} != plan {boot['cap']}")
    pd = f["server_args"].get("enable_pdmux")
    if bool(pd) != boot["engine"].startswith("pdmux"):
        bad.append(f"ABORT_ENGINE_MODE: enable_pdmux={pd} for plan engine {boot['engine']}")
    if boot["engine"] == "pdmux_sticky":
        if f.get("sticky_enabled_lines") != 1 or f.get("sticky_index") != S.STICKY_IDX[boot["D"]]:
            bad.append(f"STICKY_MODE_MISMATCH: sticky lines {f.get('sticky_enabled_lines')} index "
                       f"{f.get('sticky_index')} != {S.STICKY_IDX[boot['D']]}")
    elif f.get("sticky_enabled_lines"):
        bad.append("STICKY_MODE_MISMATCH: sticky ENABLED on a non-sticky boot")
    mt = mem_terms(f)
    if boot["model"] == "nano" and mt is not None:
        implied = round(mt["k"] / (2 * 8 * 128 * 2))           # attention layers implied by the KV banner
        if implied != boot.get("variant"):
            bad.append(f"ABORT_VARIANT_LABEL: banner implies {implied} attention layers, plan a{boot.get('variant')}")
    if f["server_args"].get("context_length") != S.CONTEXT_LENGTH:
        bad.append(f"ABORT_CONTEXT: context_length {f['server_args'].get('context_length')} != {S.CONTEXT_LENGTH}")
    return bad


def instrumentation_problems(kind, env, f, files):
    """§2-2' (1) matrix: env keys and their effects (log banner, files written) must match the boot kind."""
    i1, i46, i5, tel = S.INSTR[kind]
    bad = []
    want = {"I4I6": i46, "I5": i5, "TEL": tel}
    for name, on in want.items():
        for key in S.PROBE_ENV[name]:
            if bool(env.get(key)) != on:
                bad.append(f"{kind} boot: env {key} {'missing' if on else 'present'} (matrix {name}={'ON' if on else 'OFF'})")
    if f is not None:
        if bool(f.get("decode_probe_enabled")) != i46:
            bad.append(f"{kind} boot: 'decode probe ENABLED' log {'absent' if i46 else 'present'}")
        if bool(f.get("admission_probe_enabled")) != i5:
            bad.append(f"{kind} boot: 'admission probe ENABLED' log {'absent' if i5 else 'present'}")
    if files is not None:
        for name, on in (("decode_probe", i46), ("admission", i5), ("tel", tel)):
            if files.get(name) and not on:
                bad.append(f"{kind} boot wrote a {name} file")
    return bad


def realized_sm(boot, green):
    """§1-8.  -> dict(D_req, P_req, D_real, P_real, ok, label, detail).  The table axis uses D_real / P_real."""
    eng = boot["engine"]
    if eng == "vanilla":
        return dict(D_req=108, P_req=108, D_real=108, P_real=108, ok=True, label=None, detail="vanilla: no partition")
    if green is None:
        lab = f"SM_NOT_REALIZED_D{boot['D'] if eng == 'pdmux_sticky' else '108u'}"
        return dict(D_req=boot.get("D"), P_req=None, D_real=None, P_real=None, ok=False, label=lab,
                    detail="no green read-out (PDMUX_GREEN_READOUT)")
    groups = {g.get("stream_index"): g for g in green.get("groups", [])}
    if eng == "pdmux_108u":
        idx = len(S.STREAM_GROUPS) - 1
        g = groups.get(idx) or {}
        plain = (g.get("decode") or {}).get("green_ctx_is_null", False) and (g.get("prefill") or {}).get("green_ctx_is_null", False)
        tgt = (g.get("target_prefill_sm"), g.get("target_decode_sm"))
        ok = plain and tgt == (0, 108) and green.get("sticky_target_index") is None
        return dict(D_req=108, P_req=0, D_real=108 if ok else None, P_real=0 if ok else None, ok=ok,
                    label=None if ok else "SM_NOT_REALIZED_D108u", detail=f"plain group {idx}: {tgt}, green null={plain}")
    d = boot["D"]
    idx = S.STICKY_IDX[d]
    g = groups.get(idx) or {}
    dr = ((g.get("decode") or {}).get("green_sm") or {}).get("smCount")
    pr = ((g.get("prefill") or {}).get("green_sm") or {}).get("smCount")
    p_req = S.TOTAL_SM - d
    ok = (green.get("sticky_target_index") == idx and dr is not None and pr is not None
          and abs(dr - d) <= S.SM_TOL and abs(pr - p_req) <= S.SM_TOL and dr + pr <= S.TOTAL_SM)
    return dict(D_req=d, P_req=p_req, D_real=dr, P_real=pr, ok=ok, label=None if ok else f"SM_NOT_REALIZED_D{d}",
                detail=f"group {idx} decode {dr} prefill {pr} sticky_target {green.get('sticky_target_index')}")


# ============================================================================ feasibility (§3-1)
def bs_capturable(bs, cap_list):
    """§3-1: a cell runs only if its bs is IN the logged capture list (no padding, no eager)."""
    return cap_list is not None and bs in cap_list


def feasibility(cells, f, cap, o_map):
    """Every registered T1 cell -> {cell, O, run: bool, label, reasons}.  Inputs: the boot log and the registered
    per-cell O (`o_map[(bs, c)]`, ct_spec.o_tokens) only."""
    mt = mem_terms(f)
    kv = mt["kv_tokens"] if mt else None
    cap_list = f.get("capture_bs")
    out = []
    for bs, c in cells:
        O = o_map[(bs, c)]
        reasons = S.feasible_cell(bs, c, O, kv, cap, bs_capturable(bs, cap_list))
        out.append({"bs": bs, "c": c, "O": O, "run": not reasons, "label": None if not reasons else "INFEASIBLE_BY_RULE",
                    "reasons": reasons})
    return out


def t1_cells_for_mode(mode):
    """Registered T1 cells of one J-A mode (§1-1 grid + 16k column + rev2 extra + O-12)."""
    return S.ja_cells_for_mode(mode)


# ============================================================================ T4 predictions (§1-4)
def cmem_predictions(pool, s, k, cap, L, R):
    """Both registered forms, same (mamba static reservation) shape; they differ only by R.  Integers by floor."""
    def fl(x):
        return int(math.floor(x))
    den_a, den_b = s + k * L, s + k * (L + R)
    return {"L": L, "R": R, "cap": cap, "pool_bytes": pool, "s": s, "k": k,
            "C_A": fl(pool / den_a), "C_B": fl(pool / den_b),
            "n_A": min(cap, fl((pool - cap * s) / (k * L))),
            "n_B": min(cap, fl((pool - cap * s) / (k * (L + R))))}


# ============================================================================ client helpers
def tokens_by(chunks, t):
    """Tokens a request had received by time t (chunks = [(t, cumulative)])."""
    ts = [c[0] for c in chunks]
    i = bisect.bisect_right(ts, t)
    return chunks[i - 1][1] if i else 0


def client_gaps(requests, t_lo, t_hi):
    """Per-token gaps (s) of every request whose both chunk ends are inside [t_lo, t_hi]."""
    out = []
    for r in requests:
        ch = r.get("chunks") or []
        for (t0, c0), (t1, c1) in zip(ch, ch[1:]):
            if c1 > c0 and t_lo <= t0 and t1 <= t_hi:
                out.append((t1 - t0) / (c1 - c0))
    return out


def request_ok(r, n_in, n_out):
    return (r.get("error") is None and r.get("prompt_tokens") == n_in and r.get("completion_tokens") == n_out
            and r.get("finish") == "length")


def burst_request_ok(r, n_in, O):
    """T1 burst request (rev3.4): no error, exact input, at least ABORT_AT_TOKENS tokens, ended by the harness abort
    or (zero-ramp cells) by length; never more than O."""
    n = r.get("completion_tokens") or 0
    return (r.get("error") is None and r.get("prompt_tokens") == n_in and S.ABORT_AT_TOKENS <= n <= O
            and r.get("finish") in ("abort", "length"))


# ============================================================================ T1 window
def decode_records(records):
    return sorted((e for e in records if e.get("event") == "decode_forward"), key=lambda e: e["seq"])


def schema_problems(records):
    bad = [e for e in records if e.get("event", "").startswith("decode_") and e.get("record_schema") != DECODE_SCHEMA]
    return [f"{len(bad)} decode records with an unknown schema"] if bad else []


def first_reach_index(recs, bs):
    """Index of the first record with bs_real == bs (decode or mixed); None if never reached."""
    return next((i for i, e in enumerate(recs) if e.get("bs_real") == bs), None)


def t1_window(records, cell, capture, log_slice="", presorted=False):
    """One T1 window (rev3.4).  `records` = the boot's decode_forward records, `cell` = the client cell record.
    Status priority: BENCH_INVALID > FOREIGN_RESIDUAL > RETRACT > RECORDER_UNPROVEN > WINDOW_SHORT >
    WINDOW_CONTAMINATED > CG_UNPROVEN > PAD_MISMATCH > VALID.  Labels (never exclusions): I4_UNRESOLVED,
    CTX_OFF_NOMINAL, RAMP_EXCEEDED_BOUND."""
    bs, c, c_in = cell["bs"], cell["c"], cell["c_in"]
    reqs = cell["requests"]
    O = cell.get("O", S.O_BASE)
    out = {"cell_id": cell["cell_id"], "bs": bs, "c": c, "O": O, "status": None, "labels": [], "detail": ""}
    if len(reqs) != bs or not all(burst_request_ok(r, c_in, O) for r in reqs):
        out.update(status="BENCH_INVALID", detail="request error / input tokens / fewer than ABORT_AT_TOKENS tokens")
        return out
    t0, t1 = cell["t_first_send"], cell["t_end"]
    src = records if presorted else decode_records(records)
    lo = bisect.bisect_left([e["t_host_start"] for e in src], t0) if presorted else 0
    recs = []
    for e in src[lo:]:
        if e["t_host_start"] > t1 and presorted:
            break
        if t0 <= e["t_host_start"] <= t1:
            recs.append(e)
    firsts = sorted(r["chunks"][0][0] for r in reqs)

    def joined(t):
        return bisect.bisect_right(firsts, t + S.JOIN_SLACK_S)
    foreign = [e for e in recs if (e.get("bs_real") or 0) > min(bs, joined(e["t_host_start"]))]
    if foreign:
        out.update(status="FOREIGN_RESIDUAL", n_foreign_records=len(foreign),
                   detail=f"{len(foreign)} decode records with bs_real above the joined cell requests (or > {bs})")
        return out
    if re.search(r"#retracted_reqs: *[1-9]", log_slice or ""):
        out.update(status="RETRACT", detail="retract line in the cell's log slice")
        return out
    i0 = first_reach_index(recs, bs)
    out["ramp_records"] = i0
    out["ramp_steps_ub"] = O - S.O_BASE
    if i0 is not None and i0 > O - S.O_BASE:
        out["labels"].append("RAMP_EXCEEDED_BOUND")
    need = S.TRIM_ITERS + S.WINDOW_ITERS + 1
    win = recs[i0 + S.TRIM_ITERS:i0 + need] if i0 is not None else []
    if len(win) < S.WINDOW_ITERS + 1:
        out.update(status="WINDOW_SHORT", detail=f"{'bs never reached' if i0 is None else f'{len(win)} records after reach + trim'} < {S.WINDOW_ITERS + 1}")
        return out
    if any(b["seq"] != a["seq"] + 1 for a, b in zip(win, win[1:])):
        out.update(status="RECORDER_UNPROVEN", detail="seq gap inside the window")
        return out
    drops = [e.get("dropped_events") for e in win if e.get("dropped_events") is not None]
    if drops and max(drops) != min(drops):
        out.update(status="RECORDER_UNPROVEN", detail="dropped_events grew inside the window")
        return out
    ts = [e["t_host_start"] for e in win]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        out.update(status="RECORDER_UNPROVEN", detail="non-increasing t_host_start")
        return out
    per_req = [([ch[0] for ch in r["chunks"]], [ch[1] for ch in r["chunks"]]) for r in reqs]

    def received(tl, cl, t):
        i = bisect.bisect_right(tl, t)
        return cl[i - 1] if i else 0
    excl = {"bs_change": 0, "mixed": 0, "join_unconfirmed": 0}
    s_eng, ctx = [], []
    for e, nxt in zip(win, win[1:]):
        t = e["t_host_start"]
        why = None
        if e.get("bs_real") != bs:
            why = "bs_change"
        elif e.get("mode") != "decode":
            why = "mixed"
        elif joined(t) != bs:
            why = "join_unconfirmed"
        if why:
            excl[why] += 1
            continue
        s_eng.append((nxt["t_host_start"] - t) * 1000.0)
        ctx.append(sum(c_in + received(tl, cl, t) for tl, cl in per_req) / bs)
    n_ex = sum(excl.values())
    gpu = [e.get(I4_FIELD) for e in win]
    host = [e.get(I4_HOST_FIELD) for e in win]
    resolved = sum(1 for g in gpu if g is not None) / len(gpu)
    gaps = client_gaps(reqs, ts[0], ts[-1])
    ctx_mean = sum(ctx) / len(ctx) if ctx else None
    out.update(S_eng_ms=median(s_eng) if s_eng else None, n_intervals=len(s_eng), n_excluded=n_ex, excluded=excl,
               ctx_mean=ctx_mean, ctx_p50=median(ctx) if ctx else None,
               gpu_ms=median([g for g in gpu if g is not None]) if resolved else None,
               host_ms=median([h for h in host if h is not None]),
               i4_resolved=resolved, client_gap_ms=median(gaps) * 1000.0 if gaps else None, n_client_gaps=len(gaps),
               seq_range=(win[0]["seq"], win[-1]["seq"]), reach_seq=recs[i0]["seq"], t_range=(ts[0], ts[-1]),
               bs_padded=sorted({e.get("bs_padded") for e in win}), graph=sorted({str(e.get("graph")) for e in win}),
               s_eng_source="decode_forward.t_host_start", raw_s_eng_ms=s_eng)
    if resolved < S.I4_RESOLVED_MIN:
        out["labels"].append("I4_UNRESOLVED")
    if ctx_mean is not None and abs(ctx_mean - c) > S.CTX_BAND:
        out["labels"].append("CTX_OFF_NOMINAL")
    if n_ex > S.WINDOW_EXCL_MAX or not s_eng:
        out.update(status="WINDOW_CONTAMINATED", detail=f"{n_ex} excluded intervals > {S.WINDOW_EXCL_MAX}: {excl}")
        return out
    if any(e.get("graph") != "hit" for e in win):
        out.update(status="CG_UNPROVEN", detail="eager forward inside the window")
        return out
    exp = padded_bs(capture, bs)
    if any(e.get("bs_padded") != exp for e in win if e.get("bs_real") == bs):
        out.update(status="PAD_MISMATCH", detail=f"bs_padded {out['bs_padded']} != expected {exp}")
        return out
    out["status"] = "VALID"
    return out


def client_only_window(cell):
    """OFF diagnostic boots (no decode_forward): the same ramp-aware rule on the client side.  Reach = the last
    request's first token (all bs joined); TRIM_ITERS tokens of that request are skipped and WINDOW_ITERS more must
    arrive; the window must close before any request ended by length (abort ends them all afterwards)."""
    reqs = cell["requests"]
    O = cell.get("O", S.O_BASE)
    base = {"cell_id": cell["cell_id"], "bs": cell["bs"], "c": cell["c"], "O": O, "labels": []}
    if len(reqs) != cell["bs"] or not all(burst_request_ok(r, cell["c_in"], O) for r in reqs):
        return dict(base, status="BENCH_INVALID")
    ended = [r["chunks"][-1][0] for r in reqs if r.get("finish") == "length"]
    t_end_ok = min(ended) if ended else float("inf")
    last = max(reqs, key=lambda r: r["chunks"][0][0])
    ch = last["chunks"]
    i0, i1 = S.TRIM_ITERS, S.TRIM_ITERS + S.WINDOW_ITERS
    if len(ch) <= i1 or ch[i1][0] >= t_end_ok:
        return dict(base, status="WINDOW_SHORT")
    t_lo, t_hi = ch[i0][0], ch[i1][0]
    gaps = client_gaps(reqs, t_lo, t_hi)
    pts = [ch[k][0] for k in range(i0, i1)]
    ctx = [sum(cell["c_in"] + tokens_by(r["chunks"], t) for r in reqs) / cell["bs"] for t in pts]
    ctx_mean = sum(ctx) / len(ctx)
    if abs(ctx_mean - cell["c"]) > S.CTX_BAND:
        base["labels"].append("CTX_OFF_NOMINAL")
    return dict(base, status="VALID", client_gap_ms=median(gaps) * 1000.0, n_client_gaps=len(gaps), ctx_mean=ctx_mean)


# ============================================================================ T2 / T2'
def t2_value(seq_rec, warmup=S.T2_WARMUP_REPS):
    """seq_rec = {L, reps: [{rep, ttft_s, ok, coexist?}]} -> median TTFT (ms) of reps >= warmup."""
    used = [r for r in seq_rec["reps"] if r["rep"] >= warmup]
    bad = [r for r in used if not r.get("ok")]
    if seq_rec.get("kind") == "t2p":
        bad += [r for r in used if r.get("ok") and not r.get("coexist")]
    if bad or len(used) != S.T2_REPS - warmup:
        return {"L": seq_rec["L"], "status": "INVALID", "n_used": len(used), "n_bad": len(bad)}
    return {"L": seq_rec["L"], "status": "VALID", "ttft_ms": median([r["ttft_s"] * 1000.0 for r in used]),
            "n_used": len(used), "used_reps": [r["rep"] for r in used]}


def t2p_coexist(seq_rec, dec_records, warmup=S.T2_WARMUP_REPS):
    """Decision 28 A-2: was the bs=1 decode actually executing while each used prefill ran?  Engine evidence = a
    decode_forward with bs_real >= 1 whose t_host_start lies in the middle of the TTFT (T2P_CONFIRM_WINDOW).
    -> {label, n_dec_by_rep}; NOT_COLLECTED without engine records (OFF boots)."""
    if not dec_records:
        return {"label": "NOT_COLLECTED", "n_dec_by_rep": None}
    ts = [e["t_host_start"] for e in dec_records]
    n_by = {}
    for r in seq_rec["reps"]:
        if r["rep"] < warmup or r.get("t_first") is None:
            continue
        ttft = r["t_first"] - r["t_send"]
        lo, hi = (r["t_send"] + f * ttft for f in S.T2P_CONFIRM_WINDOW)
        i, j = bisect.bisect_left(ts, lo), bisect.bisect_right(ts, hi)
        n_by[r["rep"]] = sum(1 for e in dec_records[i:j] if (e.get("bs_real") or 0) >= 1)
    ok = bool(n_by) and all(v >= 1 for v in n_by.values())
    return {"label": f"T2P_COEXIST_{'CONFIRMED' if ok else 'UNCONFIRMED'}_L{seq_rec['L']}", "n_dec_by_rep": n_by,
            "confirmed": ok}


# ============================================================================ T4 fill
def t4_fill(fill, records, admission=None):
    """fill = client record {L, R, N, t_first_send, t_end, requests}.  n_real = max engine bs_real (I-6, mode decode)
    inside the fill; FILL_INCOMPLETE per ambiguity A-13; reason shares from I-5 windows (descriptive)."""
    reqs = fill["requests"]
    t0, t1 = fill["t_first_send"], fill["t_end"]
    recs = [e for e in records if e.get("event") == "decode_forward" and t0 <= e["t_host_start"] <= t1
            and e.get("mode") == "decode"]
    n_real = max((e["bs_real"] for e in recs), default=None)
    out = {"L": fill["L"], "R": fill["R"], "N": fill["N"], "n_real": n_real, "n_real_source": "decode_forward.bs_real",
           "labels": []}
    ok = all(request_ok(r, fill["L"], fill["R"]) for r in reqs) and len(reqs) == fill["N"]
    out["bench_ok"] = ok
    firsts = sorted(r["chunks"][0][0] for r in reqs if r.get("chunks"))
    dones = [r["chunks"][-1][0] for r in reqs if r.get("chunks") and r.get("completion_tokens") == fill["R"]]
    if not ok or n_real is None or not dones or len(firsts) < n_real:
        out["labels"].append("FILL_INCOMPLETE")
        out["detail"] = "bench invalid or no engine records"
    elif firsts[n_real - 1] >= min(dones):
        out["labels"].append("FILL_INCOMPLETE")
        out["detail"] = "the n_real-th first token came after the earliest completion"
    if admission is not None:
        wins = [w for w in admission if w.get("event") == "admission_window"
                and w.get("t_mono_end", -1) >= t0 and w.get("t_mono_start", 1e18) <= t1]
        hist_max = max((int(k) for w in wins for k, v in (w.get("decode_bs_hist") or {}).items() if v), default=None)
        out["n_real_admission_hist"] = hist_max
        tot = {}
        tw = {}
        dur = 0.0
        for w in wins:
            cats = w.get("terminal_by_category") or {}
            n = sum(cats.values())
            d = max(0.0, w["t_mono_end"] - w["t_mono_start"])
            for k_, v in cats.items():
                tot[k_] = tot.get(k_, 0) + v
                if n:
                    tw[k_] = tw.get(k_, 0.0) + d * v / n
            dur += d if n else 0.0
        n_all = sum(tot.values())
        out["reason_share_iter"] = {k_: v / n_all for k_, v in tot.items()} if n_all else None
        out["reason_share_time"] = {k_: v / dur for k_, v in tw.items()} if dur else None
    else:
        out["reason_share_iter"] = out["reason_share_time"] = "NOT_COLLECTED"
    return out
