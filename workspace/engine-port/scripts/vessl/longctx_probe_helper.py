#!/usr/bin/env python3
"""longctx_probe_helper.py -- mechanics for scripts/vessl/longctx_probe.sh (128k bring-up probe).

Plan: results/cap_char/PROBE_128k_PLAN_2026-10-20.md (go/no-go items G1-G7).
Decision: results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md (decisions 10-12').

NO PERFORMANCE JUDGMENT lives here.  Statuses are mechanical (PASS = every request in the stage
completed with the forced token counts and no OOM signature; FAIL = it did not; UNRESOLVED =
the measurement condition was missing; SKIPPED).  TTFT / ITL values are descriptive probe values:
not citable, budget inputs only (plan sec 3).

Pure functions (unit-tested on CPU, tests/test_longctx_probe.py):
  splitmix64 / synth_text / synth_ids   fixed-seed reproducible synthetic input of EXACTLY L tokens
  check_budget                          n_in + n_out <= context_length
  parse_banner / c_mem / cmem_table     boot-log memory facts and C_mem(L) = Pool / (s + k*L)
  log_signatures                        OOM / traceback counts in a server log
  parse_sse_line / stream_generate      streamed /generate client (per-token timestamps)
  itl_summary / lstsq_slope             descriptive time-series reductions
  validate_result / score_*             result json schema + mechanical stage status

Subcommands (probe commands write incrementally so a `timeout` kill keeps partial results):
  pre-models  boot-facts  probe-single  probe-g4  probe-g7  probe-conc
  score-single  score-g4  score-g7  score-conc  score-g6  summary
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import threading
import time

SCHEMA = "longctx_probe/v1"
LLAMA_ID = "meta-llama/Llama-3.1-8B"
LLAMA_REV = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LB1_ZIP_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"  # LONGCTX_STATS header
CONTEXT_LENGTH = 131072
K = 1024
# 124k point = 124*1024 (input) so that 126976 + 64 forced output <= 131072 with room for a 4k output
L_GRID = [16 * K, 32 * K, 64 * K, 96 * K, 124 * K]
OUT_TOKENS = 64
GIB = 1024 ** 3
BASE_SEED = 20261020
J1_FLOOR_MS_PER_TOKEN = 0.111   # reference value printed beside G6, never compared here
STAGE_ORDER = ["PRE", "G1", "G2", "G4", "G6", "G7", "TD-nano_pdmux", "G3", "TD-nano_fused",
               "G5", "G5-conc", "TD-llama", "G3-chunked", "TD-nano_chunked"]
PASS, FAIL, UNRES, SKIP = "PASS", "FAIL", "UNRESOLVED", "SKIPPED"

SYNTH_RULE = {
    "name": "splitmix64_pseudoword_text_v1",
    "text": ("space-joined pseudo-words (2-4 syllables from a fixed 70-syllable table), '. ' after every "
             "12th word, newline after every 80th word; all choices from a splitmix64 stream seeded by "
             "seed_for(L, rep)"),
    "length": ("text is tokenized (add_special_tokens=False) in 2048-word chunks and concatenated; the "
               "first L ids are sent as input_ids, so prompt_tokens == L exactly (no retokenization)"),
    "seed": "seed_for(L, rep) = splitmix64(BASE_SEED ^ (L << 8) ^ rep); BASE_SEED = %d" % BASE_SEED,
    "output": "max_new_tokens forced with ignore_eos=true, temperature 0",
}


# ===========================================================================
# small io helpers
# ===========================================================================
def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def write_status(status_dir, stage, status, reason, facts=None):
    os.makedirs(status_dir, exist_ok=True)
    _dump({"stage": stage, "status": status, "reason": reason, "facts": facts or {}},
          os.path.join(status_dir, f"{stage}.json"))
    print(f"{stage}: {status} -- {reason}")


# ===========================================================================
# synthetic input (pure)
# ===========================================================================
_M64 = (1 << 64) - 1
_SYL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]       # 70 syllables


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def splitmix64(x):
    """One splitmix64 step on state x (also used as an integer hash)."""
    return _mix((x + 0x9E3779B97F4A7C15) & _M64)


class _Stream:
    def __init__(self, seed):
        self.s = seed & _M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & _M64
        return _mix(self.s)

    def below(self, n):
        return self.next() % n


def seed_for(L, rep, base=BASE_SEED):
    return splitmix64((base ^ (int(L) << 8) ^ int(rep)) & _M64)


_DRAWS_PER_WORD = 5


def synth_text(seed, n_words, start_word=0):
    """Deterministic pseudo-text of n_words words (5 stream draws per word, so a chunk starting at
    start_word is position-addressable).  Same (seed, n_words, start_word) -> same text."""
    st = _Stream(seed)
    st.s = (st.s + 0x9E3779B97F4A7C15 * start_word * _DRAWS_PER_WORD) & _M64   # jump ahead
    out = []
    for i in range(start_word, start_word + n_words):
        n_syl = 2 + st.below(3)                       # 2..4 syllables
        d = [st.below(len(_SYL)) for _ in range(4)]
        w = "".join(_SYL[x] for x in d[:n_syl])
        if (i + 1) % 80 == 0:
            out.append(w + "\n")
        elif (i + 1) % 12 == 0:
            out.append(w + ". ")
        else:
            out.append(w + " ")
    return "".join(out)


def synth_ids(encode, L, seed, chunk_words=2048):
    """Exactly L token ids from fixed-seed pseudo-text.  encode(text) -> list[int] (no special tokens)."""
    if L <= 0:
        raise ValueError("L must be positive")
    ids, w = [], 0
    while len(ids) < L:
        piece = encode(synth_text(seed, chunk_words, start_word=w))
        if not piece:
            raise ValueError("encoder returned no tokens for a non-empty text")
        ids.extend(piece)
        w += chunk_words
    ids = ids[:L]
    assert len(ids) == L
    return ids


def ids_sha256(ids):
    h = hashlib.sha256()
    h.update((",".join(str(i) for i in ids)).encode())
    return h.hexdigest()


def check_budget(n_in, n_out, ctx=CONTEXT_LENGTH):
    """input + output must fit the context window (plan: assert, not clip)."""
    if n_in <= 0 or n_out <= 0:
        raise ValueError(f"non-positive length n_in={n_in} n_out={n_out}")
    if n_in + n_out > ctx:
        raise ValueError(f"n_in {n_in} + n_out {n_out} = {n_in + n_out} > context_length {ctx}")
    return n_in + n_out


# ===========================================================================
# boot-log facts and C_mem (pure)
# ===========================================================================
def _gib(m):
    return float(m) * GIB


def parse_banner(text):
    """Memory facts from an SGLang boot log.  Sizes in the banner are GiB (memory_pool.py GB=1024**3)."""
    text = text or ""
    f = {"max_total_num_tokens": None, "kv_cache_tokens": None, "kv_k_gib": None, "kv_v_gib": None,
         "max_mamba_cache_size": None, "mamba_conv_gib": None, "mamba_ssm_gib": None,
         "context_len_logged": None, "available_gpu_mem_gib": None}
    m = re.findall(r"max_total_num_tokens=([0-9]+)", text)
    f["max_total_num_tokens"] = int(m[-1]) if m else None
    m = re.findall(r"context_len=([0-9]+)", text)
    f["context_len_logged"] = int(m[-1]) if m else None
    m = re.findall(r"available_gpu_mem=([0-9.]+) *GB", text)
    f["available_gpu_mem_gib"] = float(m[-1]) if m else None
    m = re.findall(r"KV Cache is allocated\. #tokens: *([0-9]+), *K size: *([0-9.]+) *GB, *V size: *([0-9.]+) *GB", text)
    if m:
        f["kv_cache_tokens"], f["kv_k_gib"], f["kv_v_gib"] = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
    m = re.findall(r"Mamba Cache is allocated\. *max_mamba_cache_size: *([0-9]+), *conv_state size: *([0-9.]+) *GB"
                   r"(?:, *ssm_state size: *([0-9.]+) *GB)?", text)
    if m:
        f["max_mamba_cache_size"] = int(m[-1][0])
        f["mamba_conv_gib"] = float(m[-1][1])
        f["mamba_ssm_gib"] = float(m[-1][2]) if m[-1][2] else None
    return f


def mem_terms(facts):
    """(pool_bytes, slot_bytes s, kv_bytes_per_token k) from banner facts; None if KV banner missing.
    Pool = KV pool + Mamba pool; s = Mamba pool / (max_mamba_cache_size + 1 slots) (0 for attention-only)."""
    if not facts.get("kv_cache_tokens") or facts.get("kv_k_gib") is None:
        return None
    kv_bytes = _gib(facts["kv_k_gib"] + facts["kv_v_gib"])
    k = kv_bytes / facts["kv_cache_tokens"]
    mamba = 0.0
    s = 0.0
    if facts.get("max_mamba_cache_size") is not None:
        mamba = _gib((facts.get("mamba_conv_gib") or 0.0) + (facts.get("mamba_ssm_gib") or 0.0))
        s = mamba / (facts["max_mamba_cache_size"] + 1)
    return kv_bytes + mamba, s, k


def c_mem(pool_bytes, slot_bytes, kv_bytes_per_token, L):
    """C_mem(L) = floor(Pool / (s + k*L))  (DESIGN_cap_longctx_v1 sec 1; derived, not a promise)."""
    den = slot_bytes + kv_bytes_per_token * L
    if den <= 0 or L <= 0:
        raise ValueError("bad C_mem inputs")
    return int(pool_bytes // den)


def cmem_table(facts, Ls):
    t = mem_terms(facts)
    if t is None:
        return None
    pool, s, k = t
    return {"pool_bytes": pool, "slot_bytes": s, "kv_bytes_per_token": k,
            "formula": "floor(pool_bytes / (slot_bytes + kv_bytes_per_token * L))",
            "by_L": {str(L): {"c_mem": c_mem(pool, s, k, L),
                              "kv_token_bound": (facts["kv_cache_tokens"] // L)} for L in Ls}}


_OOM_PATS = {"oom": r"CUDA out of memory|OutOfMemoryError|out of memory",
             "traceback": r"Traceback \(most recent call last\)",
             "scheduler_exception": r"Scheduler hit an exception|SIGQUIT|Received sigquit",
             "cuda_graph_capture_failed": r"Capture cuda graph failed",
             "eager_fallback": r"[Ff]all(?:ing)? back to eager|disable_cuda_graph=True"}


def log_signatures(text):
    text = text or ""
    return {k: len(re.findall(p, text)) for k, p in _OOM_PATS.items()}


def sig_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


# ===========================================================================
# streamed /generate client
# ===========================================================================
def parse_sse_line(raw):
    """-> None (ignore) | 'DONE' | dict.  raw is bytes or str."""
    s = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    s = s.strip()
    if not s.startswith("data:"):
        return None
    s = s[5:].strip()
    if s == "[DONE]":
        return "DONE"
    try:
        return json.loads(s)
    except ValueError:
        return None


def stream_generate(port, input_ids, max_new, timeout, host="127.0.0.1", t_ref=None, live=None):
    """One streamed request.  Times are seconds relative to t_ref (default: this call's send time).
    live: optional dict updated with {"n": chunks so far} (lets another thread see progress)."""
    t_ref = time.perf_counter() if t_ref is None else t_ref
    rec = {"t_send": time.perf_counter() - t_ref, "chunks": [], "ttft_s": None, "e2e_s": None,
           "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "error": None}
    body = json.dumps({"input_ids": list(input_ids),
                       "sampling_params": {"max_new_tokens": int(max_new), "temperature": 0.0, "ignore_eos": True},
                       "stream": True})
    conn = None
    t_send = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/generate", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["error"] = f"HTTP {resp.status}: {resp.read(300).decode('utf-8', 'replace')}"
            return rec
        while True:
            line = resp.readline()
            if not line:
                break
            ev = parse_sse_line(line)
            if ev is None:
                continue
            if ev == "DONE":
                break
            now = time.perf_counter()
            mi = ev.get("meta_info") or {}
            ct = mi.get("completion_tokens")
            if ct is not None:
                rec["chunks"].append((round(now - t_ref, 6), int(ct)))
                if live is not None:
                    live["n"] = len(rec["chunks"])
                if rec["ttft_s"] is None and int(ct) >= 1:
                    rec["ttft_s"] = now - t_send
            if mi.get("prompt_tokens") is not None:
                rec["prompt_tokens"] = mi["prompt_tokens"]
            if mi.get("completion_tokens") is not None:
                rec["completion_tokens"] = mi["completion_tokens"]
            if mi.get("finish_reason") is not None:
                rec["finish_reason"] = mi["finish_reason"]
            if "error" in ev:
                rec["error"] = str(ev["error"])[:300]
        rec["e2e_s"] = time.perf_counter() - t_send
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:300]
    finally:
        if conn is not None:
            conn.close()
    return rec


def request_ok(rec, n_in, n_out):
    """Mechanical completion test: no error, token counts as forced, finish by length."""
    fr = rec.get("finish_reason")
    ftype = fr.get("type") if isinstance(fr, dict) else fr
    return (rec.get("error") is None and rec.get("prompt_tokens") == n_in
            and rec.get("completion_tokens") == n_out and ftype == "length")


def server_alive(port, host="127.0.0.1", timeout=5):
    try:
        c = http.client.HTTPConnection(host, port, timeout=timeout)
        c.request("GET", "/health")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except Exception:  # noqa: BLE001
        return False


# ===========================================================================
# descriptive reductions (pure)
# ===========================================================================
def token_gaps(chunks):
    """[(t_end, gap_per_token_s)] from (t, cumulative_completion_tokens) chunks."""
    out = []
    for (t0, c0), (t1, c1) in zip(chunks, chunks[1:]):
        if c1 > c0:
            out.append((t1, (t1 - t0) / (c1 - c0)))
    return out


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def itl_summary(chunks, t_lo, t_hi):
    """Descriptive: token gaps inside [t_lo, t_hi] vs before t_lo.  No threshold, no verdict."""
    g = token_gaps(chunks)
    inw = [d for t, d in g if t_lo <= t <= t_hi]
    pre = [d for t, d in g if t < t_lo]
    return {"n_in_window": len(inw), "max_gap_in_window_s": max(inw) if inw else None,
            "p50_in_window_s": _pct(inw, 0.5), "p95_in_window_s": _pct(inw, 0.95),
            "n_before": len(pre), "p50_before_s": _pct(pre, 0.5), "p95_before_s": _pct(pre, 0.95),
            "last_chunk_t": chunks[-1][0] if chunks else None}


def lstsq_slope(xs, ys):
    """Least-squares slope and intercept; None if < 2 distinct x."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return {"slope": b, "intercept": my - b * mx, "n": n}


# ===========================================================================
# result schema + scoring (pure)
# ===========================================================================
REQ_KEYS = {"kind", "stage", "arm", "model", "L", "rep", "n_in", "max_new", "seed", "ids_sha256", "ok",
            "ttft_s", "e2e_s", "prompt_tokens", "completion_tokens", "finish_reason", "error",
            "server_alive_after", "log_delta", "t_send_utc"}
TOP_KEYS = {"schema", "kind", "stage", "arm", "model", "context_length", "out_tokens", "synth_rule",
            "utc_start", "records"}


def validate_result(obj):
    """-> list of problems (empty = schema OK)."""
    bad = []
    for k in sorted(TOP_KEYS - set(obj)):
        bad.append(f"missing top-level key {k}")
    if obj.get("schema") != SCHEMA:
        bad.append(f"schema != {SCHEMA}")
    for i, r in enumerate(obj.get("records") or []):
        for k in sorted(REQ_KEYS - set(r)):
            bad.append(f"records[{i}] missing {k}")
        if r.get("n_in") is not None and r.get("max_new") is not None:
            try:
                check_budget(r["n_in"], r["max_new"], obj.get("context_length", CONTEXT_LENGTH))
            except ValueError as e:
                bad.append(f"records[{i}] {e}")
    return bad


def new_result(stage, arm, model, kind="single"):
    return {"schema": SCHEMA, "kind": kind, "stage": stage, "arm": arm, "model": model,
            "context_length": CONTEXT_LENGTH, "out_tokens": OUT_TOKENS, "synth_rule": SYNTH_RULE,
            "utc_start": _utc(), "records": []}


def make_record(stage, arm, model, L, rep, ids, seed, max_new, rec, alive, delta, kind="single", extra=None):
    r = {"kind": kind, "stage": stage, "arm": arm, "model": model, "L": L, "rep": rep, "n_in": len(ids),
         "max_new": max_new, "seed": seed, "ids_sha256": ids_sha256(ids),
         "ok": request_ok(rec, len(ids), max_new), "ttft_s": rec.get("ttft_s"), "e2e_s": rec.get("e2e_s"),
         "prompt_tokens": rec.get("prompt_tokens"), "completion_tokens": rec.get("completion_tokens"),
         "finish_reason": rec.get("finish_reason"), "error": rec.get("error"),
         "server_alive_after": alive, "log_delta": delta, "t_send_utc": _utc()}
    if extra:
        r.update(extra)
    return r


def score_single(res, expect_n, stage):
    """-> (status, reason, facts).  PASS only if every expected request is present and ok."""
    if res is None:
        return UNRES, "no result file (probe did not start or was killed before the first request)", {}
    recs = res.get("records") or []
    bad = validate_result(res)
    per_L = {}
    for r in recs:
        d = per_L.setdefault(str(r["L"]), {"ok": 0, "fail": 0, "errors": []})
        if r["ok"]:
            d["ok"] += 1
        else:
            d["fail"] += 1
            d["errors"].append((r.get("error") or f"completion {r.get('completion_tokens')}/{r.get('max_new')} "
                                f"prompt {r.get('prompt_tokens')}/{r.get('n_in')} finish {r.get('finish_reason')}")[:160])
    oom = sum((r.get("log_delta") or {}).get("oom", 0) for r in recs)
    facts = {"per_L": per_L, "n_records": len(recs), "n_expected": expect_n, "oom_log_hits": oom,
             "schema_problems": bad}
    if bad:
        return FAIL, f"result schema problems: {bad[:2]}", facts
    nfail = sum(1 for r in recs if not r["ok"])
    if len(recs) < expect_n:
        return (FAIL if nfail else UNRES), (f"{len(recs)}/{expect_n} requests recorded, {nfail} not ok"
                                            " (probe stopped early: timeout/budget/server death)"), facts
    if nfail or oom:
        failL = sorted({r["L"] for r in recs if not r["ok"]})
        return FAIL, f"{nfail} not-ok requests (L={failL}), oom log hits {oom}", facts
    return PASS, f"{len(recs)}/{expect_n} requests completed with forced token counts, 0 OOM log hits", facts


def score_g7(res):
    if res is None:
        return UNRES, "no result file", {}
    recs = res.get("records") or []
    facts = {"docs": res.get("docs"), "pairs": []}
    if not recs:
        return UNRES, "no G7 requests recorded", facts
    by = {}
    for r in recs:
        by.setdefault(r.get("doc_id"), {"real": [], "synth": []})[r.get("input_kind", "?")].append(r)
    for doc, d in by.items():
        t = lambda xs: [x["ttft_s"] for x in xs if x["ok"] and x["ttft_s"] is not None]  # noqa: E731
        tr, ts = t(d["real"]), t(d["synth"])
        mr = sum(tr) / len(tr) if tr else None
        ms = sum(ts) / len(ts) if ts else None
        facts["pairs"].append({"doc_id": doc, "n_in": (d["real"] or d["synth"])[0]["n_in"],
                               "ttft_real_s": tr, "ttft_synth_s": ts,
                               "rel_diff_real_over_synth": (mr / ms - 1.0) if mr and ms else None})
    nfail = sum(1 for r in recs if not r["ok"])
    exp = 4 * len(by)
    if nfail:
        return FAIL, f"{nfail} not-ok G7 requests", facts
    if len(recs) < exp:
        return UNRES, f"{len(recs)}/{exp} G7 requests recorded", facts
    return PASS, f"{len(recs)} requests completed (real vs synthetic ABBA); values are descriptive only", facts


# ===========================================================================
# tokenizer / model resolution (needs transformers; lazy)
# ===========================================================================
def snapshot_dir(hf_home, model_id):
    ref = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))
    rev = _read(os.path.join(ref, "refs", "main")).strip()
    return ref, rev, (os.path.join(ref, "snapshots", rev) if rev else None)


def load_encoder(tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True, local_files_only=True)
    return (lambda text: tok.encode(text, add_special_tokens=False)), tok


# ===========================================================================
# subcommands
# ===========================================================================
def cmd_pre_models(a):
    """Model snapshots present under HF_HOME (offline); Llama pinned to LLAMA_REV."""
    out = {"kind": "longctx_probe_models", "utc": _utc(), "models": {}}
    ok, why = True, []
    for mid in (a.nano, a.llama):
        ref, rev, snap = snapshot_dir(a.hf_home, mid)
        present = bool(snap) and os.path.isdir(snap)
        names = sorted(os.listdir(snap)) if present else []
        tj = os.path.join(snap, "tokenizer.json") if present else None
        out["models"][mid] = {"refs_main": rev or None, "snapshot_present": present, "files": names[:40],
                              "n_safetensors": sum(n.endswith(".safetensors") for n in names),
                              "tokenizer_json_sha256": hashlib.sha256(open(tj, "rb").read()).hexdigest()
                              if tj and os.path.exists(tj) else None}
        if not present:
            ok = False
            why.append(f"{mid}: snapshot missing under {ref}")
        elif out["models"][mid]["n_safetensors"] == 0 or "config.json" not in names:
            ok = False
            why.append(f"{mid}: no safetensors/config.json in snapshot")
    llama_rev = out["models"][a.llama]["refs_main"]
    if llama_rev != LLAMA_REV:
        ok = False
        why.append(f"{a.llama}: refs/main {llama_rev!r} != pinned {LLAMA_REV}")
    out["ok"], out["problems"] = ok, why
    _dump(out, a.out)
    print(json.dumps({"ok": ok, "problems": why}))
    return 0 if ok else 3


def cmd_boot_facts(a):
    """Boot-log memory facts + C_mem table (G1 'cap/C_mem' and G5/G6)."""
    text = _read(a.srvlog)
    f = parse_banner(text)
    out = {"kind": "longctx_probe_boot_facts", "utc": _utc(), "arm": a.arm, "model": a.model,
           "cap": a.cap, "banner": f, "signatures": log_signatures(text),
           "cmem": cmem_table(f, L_GRID), "context_length_requested": CONTEXT_LENGTH,
           "context_len_matches": f["context_len_logged"] == CONTEXT_LENGTH}
    _dump(out, a.out)
    print(json.dumps({"max_total_num_tokens": f["max_total_num_tokens"], "cmem_ok": out["cmem"] is not None,
                      "context_len_logged": f["context_len_logged"]}))
    return 0


def _ls(arg):
    return [int(x) for x in arg.split(",") if x]


def _run_one(a, res, stage, L, rep, encode, extra_ids=None):
    seed = seed_for(L, rep)
    ids = extra_ids if extra_ids is not None else synth_ids(encode, L, seed)
    check_budget(len(ids), a.out_tokens, CONTEXT_LENGTH)
    before = log_signatures(_read(a.srvlog))
    rec = stream_generate(a.port, ids, a.out_tokens, a.req_timeout)
    alive = server_alive(a.port)
    delta = sig_delta(before, log_signatures(_read(a.srvlog)))
    r = make_record(stage, a.arm, a.model, L, rep, ids, seed, a.out_tokens, rec, alive, delta)
    res["records"].append(r)
    _dump(res, a.out)
    print(f"[{stage}] L={L} rep={rep} ok={r['ok']} alive={alive} err={r['error']}")
    return r


def cmd_probe_single(a):
    """Single synthetic requests, L ascending x reps, plus an un-scored 512-token warm-up request."""
    encode, _ = load_encoder(a.tokenizer)
    res = new_result(a.stage, a.arm, a.model)
    res["synth_ids_check"] = {}
    _dump(res, a.out)
    w = stream_generate(a.port, synth_ids(encode, 512, seed_for(512, 99)), 8, a.req_timeout)
    res["warmup"] = {"ttft_s": w.get("ttft_s"), "error": w.get("error"), "note": "not scored, not an L point"}
    for L in _ls(a.Ls):
        for rep in range(a.reps):
            r = _run_one(a, res, a.stage, L, rep, encode)
            if not r["server_alive_after"]:
                print("server dead -- stopping this probe")
                return 0
    return 0


def cmd_probe_conc(a):
    """C_mem(L) simultaneous requests for each L (Llama G5).  n from the boot-facts json."""
    encode, _ = load_encoder(a.tokenizer)
    bf = _load(a.boot_facts) or {}
    res = new_result(a.stage, a.arm, a.model, kind="conc")
    res["groups"] = []
    for L in _ls(a.Ls):
        cm = ((bf.get("cmem") or {}).get("by_L") or {}).get(str(L))
        if not cm:
            res["groups"].append({"L": L, "n": 0, "note": "no C_mem (boot banner missing)"})
            continue
        n = max(1, min(cm["c_mem"], a.max_conc))
        seeds = [seed_for(L, 100 + i) for i in range(n)]
        idl = [synth_ids(encode, L, s) for s in seeds]
        for ids in idl:
            check_budget(len(ids), a.out_tokens, CONTEXT_LENGTH)
        before = log_signatures(_read(a.srvlog))
        t_ref = time.perf_counter()
        outs = [None] * n

        def work(i):
            outs[i] = stream_generate(a.port, idl[i], a.out_tokens, a.req_timeout, t_ref=t_ref)
        th = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        alive = server_alive(a.port)
        delta = sig_delta(before, log_signatures(_read(a.srvlog)))
        for i in range(n):
            res["records"].append(make_record(a.stage, a.arm, a.model, L, i, idl[i], seeds[i], a.out_tokens,
                                              outs[i], alive, delta, kind="conc",
                                              extra={"group_n": n, "c_mem": cm["c_mem"]}))
        res["groups"].append({"L": L, "n": n, "c_mem": cm["c_mem"], "kv_token_bound": cm["kv_token_bound"],
                              "wall_s": max((o["e2e_s"] or 0) for o in outs)})
        _dump(res, a.out)
        print(f"[{a.stage}] L={L} n={n} ok={sum(r['ok'] for r in res['records'] if r['L'] == L)} alive={alive}")
        if not alive:
            break
    return 0


def cmd_probe_g4(a):
    """4 short-input long-output decode streams running, then 1 long prefill is submitted; the per-token
    time series of all streams is recorded.  Descriptive only -- no threshold, no verdict."""
    encode, _ = load_encoder(a.tokenizer)
    res = new_result("G4", a.arm, a.model, kind="g4")
    res["scenarios"] = []
    _dump(res, a.out)
    for L in _ls(a.Ls):
        seeds = [seed_for(256, 200 + i) for i in range(a.n_decode)]
        dec = [synth_ids(encode, a.decode_in, s) for s in seeds]
        long_seed = seed_for(L, 300)
        lids = synth_ids(encode, L, long_seed)
        for ids in dec:
            check_budget(len(ids), a.decode_out, CONTEXT_LENGTH)
        check_budget(len(lids), a.out_tokens, CONTEXT_LENGTH)
        before = log_signatures(_read(a.srvlog))
        t_ref = time.perf_counter()
        dres = [None] * a.n_decode
        lres = [None]
        live = [{"n": 0} for _ in range(a.n_decode)]

        def wd(i):
            dres[i] = stream_generate(a.port, dec[i], a.decode_out, a.req_timeout, t_ref=t_ref, live=live[i])

        def wl():
            lres[0] = stream_generate(a.port, lids, a.out_tokens, a.req_timeout, t_ref=t_ref)
        dth = [threading.Thread(target=wd, args=(i,)) for i in range(a.n_decode)]
        [t.start() for t in dth]
        t_lead_start = time.perf_counter()
        lead_ready = False
        while time.perf_counter() - t_lead_start < a.lead_timeout_s:   # every decode stream running first
            if all(x["n"] >= a.lead_tokens for x in live):
                lead_ready = True
                break
            time.sleep(0.2)
        t_submit = time.perf_counter() - t_ref
        lth = threading.Thread(target=wl)
        lth.start()
        lth.join()
        [t.join() for t in dth]
        alive = server_alive(a.port)
        delta = sig_delta(before, log_signatures(_read(a.srvlog)))
        lr = lres[0]
        t_first = (t_submit + lr["ttft_s"]) if lr and lr.get("ttft_s") else None
        win_hi = t_first if t_first is not None else (lr["chunks"][-1][0] if lr and lr["chunks"] else t_submit)
        sc = {"L": L, "decode_in": a.decode_in, "decode_out": a.decode_out, "n_decode": a.n_decode,
              "t_submit_long_s": t_submit, "t_long_first_token_s": t_first, "decode_lead_ready": lead_ready,
              "long": make_record("G4", a.arm, a.model, L, 0, lids, long_seed, a.out_tokens, lr, alive, delta,
                                  kind="g4_long"),
              "decode": [], "server_alive_after": alive, "log_delta": delta}
        for i, d in enumerate(dres):
            sc["decode"].append({"i": i, "ok": request_ok(d, a.decode_in, a.decode_out), "error": d.get("error"),
                                 "completion_tokens": d.get("completion_tokens"), "chunks": d["chunks"],
                                 "itl_window": itl_summary(d["chunks"], t_submit, win_hi)})
        res["scenarios"].append(sc)
        _dump(res, a.out)
        print(f"[G4] L={L} long_ok={sc['long']['ok']} decode_ok={[d['ok'] for d in sc['decode']]} alive={alive}")
        if not alive:
            break
    return 0


def load_lb_candidates(zip_path, top_k=60, max_chars=400_000):
    """LongBench v1 original records (English, non-_e tasks), longest by characters first.
    INPUT = context + '\\n' + input exactly like longctx_measure.py (LONGCTX_STATS header)."""
    import zipfile
    cand = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith(".jsonl") or n.endswith("_e.jsonl"):
                continue
            for line in z.open(n):
                r = json.loads(line)
                if r.get("language") != "en":
                    continue
                txt = (r.get("context") or "") + "\n" + (r.get("input") or "")
                if len(txt) <= max_chars:
                    cand.append((len(txt), r.get("dataset"), r.get("_id"), txt))
    cand.sort(key=lambda t: -t[0])
    return cand[:top_k]


def pick_docs(tokenized, max_tokens, n_docs=2):
    """tokenized: [(n_tokens, task, _id, ids)].  Longest doc <= max_tokens, then the longest of a
    different task (>= 8k tokens preferred by ordering only)."""
    ok = sorted([t for t in tokenized if t[0] <= max_tokens], key=lambda t: -t[0])
    chosen, tasks = [], set()
    for t in ok:
        if t[1] in tasks:
            continue
        chosen.append(t)
        tasks.add(t[1])
        if len(chosen) == n_docs:
            break
    return chosen


def cmd_probe_g7(a):
    """Same boot: real LongBench v1 doc vs synthetic input of the SAME token count, ABBA, TTFT only."""
    encode, _ = load_encoder(a.tokenizer)
    raw = _read_bytes_sha(a.lb_zip)
    res = new_result("G7", a.arm, a.model, kind="g7")
    res["lb_zip"] = {"path": a.lb_zip, "sha256": raw, "expected_sha256": LB1_ZIP_SHA256}
    if raw != LB1_ZIP_SHA256:
        res["problem"] = "LongBench v1 zip sha256 mismatch"
        _dump(res, a.out)
        return 4
    cands = load_lb_candidates(a.lb_zip)
    tok = [(len(ids), task, did, ids) for (_, task, did, txt) in cands for ids in [encode(txt)]]
    docs = pick_docs(tok, a.max_tokens, a.n_docs)
    res["docs"] = [{"doc_id": f"{t}:{d}", "task": t, "_id": d, "n_in": n, "ids_sha256": ids_sha256(ids)}
                   for (n, t, d, ids) in docs]
    res["n_candidates_tokenized"] = len(tok)
    res["selection_rule"] = (f"English non-_e LongBench v1 records, INPUT=context+'\\n'+input, {len(cands)} longest by "
                             f"chars tokenized, longest <= {a.max_tokens} tokens, distinct tasks; original text "
                             "is sent unmodified (no truncation)")
    _dump(res, a.out)
    for (n, t, d, ids) in docs:
        did = f"{t}:{d}"
        sids = synth_ids(encode, n, seed_for(n, 400))
        sids2 = synth_ids(encode, n, seed_for(n, 401))
        for kind, rep, ids_ in (("real", 0, ids), ("synth", 0, sids), ("synth", 1, sids2), ("real", 1, ids)):
            check_budget(len(ids_), a.out_tokens, CONTEXT_LENGTH)
            before = log_signatures(_read(a.srvlog))
            rr = stream_generate(a.port, ids_, a.out_tokens, a.req_timeout)
            alive = server_alive(a.port)
            delta = sig_delta(before, log_signatures(_read(a.srvlog)))
            res["records"].append(make_record("G7", a.arm, a.model, n, rep, ids_, seed_for(n, 400 + rep)
                                              if kind == "synth" else 0, a.out_tokens, rr, alive, delta,
                                              kind="g7", extra={"doc_id": did, "input_kind": kind}))
            _dump(res, a.out)
            print(f"[G7] {did} n_in={n} {kind}{rep} ok={res['records'][-1]['ok']} alive={alive}")
            if not alive:
                return 0
    return 0


def _read_bytes_sha(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def cmd_score_single(a):
    st, why, facts = score_single(_load(a.inp), a.expect_n, a.stage)
    write_status(a.status_dir, a.stage, st, why, facts)
    return 0


def cmd_score_conc(a):
    res = _load(a.inp)
    n_exp = a.expect_n or len((res or {}).get("records") or [])
    st, why, facts = score_single(res, n_exp, a.stage) if res else (UNRES, "no result file", {})
    if res and not res.get("records"):
        st, why = UNRES, "no concurrent-request records (probe did not start or was killed)"
    if res:
        facts["groups"] = res.get("groups")
        if any(g.get("n") == 0 for g in res.get("groups") or []):
            st, why = UNRES, "a group had no C_mem (boot banner missing)"
    write_status(a.status_dir, a.stage, st, why, facts)
    return 0


def cmd_score_g4(a):
    res = _load(a.inp)
    if not res or not res.get("scenarios"):
        write_status(a.status_dir, "G4", UNRES, "no scenarios recorded", {})
        return 0
    n_exp = len(_ls(a.Ls))
    bad = [s["L"] for s in res["scenarios"]
           if not s["long"]["ok"] or not all(d["ok"] for d in s["decode"]) or not s["server_alive_after"]]
    facts = {"scenarios": [{"L": s["L"], "t_submit_long_s": s["t_submit_long_s"],
                            "t_long_first_token_s": s["t_long_first_token_s"],
                            "itl_window": [d["itl_window"] for d in s["decode"]]} for s in res["scenarios"]],
             "note": "descriptive ITL values; no threshold applied here"}
    if bad:
        write_status(a.status_dir, "G4", FAIL, f"scenario(s) with a not-ok request or dead server: L={bad}", facts)
    elif len(res["scenarios"]) < n_exp:
        write_status(a.status_dir, "G4", UNRES, f"{len(res['scenarios'])}/{n_exp} scenarios recorded", facts)
    else:
        write_status(a.status_dir, "G4", PASS, f"{n_exp} scenarios completed; time series in {os.path.basename(a.inp)}", facts)
    return 0


def cmd_score_g7(a):
    st, why, facts = score_g7(_load(a.inp))
    write_status(a.status_dir, "G7", st, why, facts)
    return 0


def mean_ttft_by_L(res):
    d = {}
    for r in (res or {}).get("records") or []:
        if r.get("ok") and r.get("ttft_s") is not None:
            d.setdefault(r["L"], []).append(r["ttft_s"])
    return {L: sum(v) / len(v) for L, v in sorted(d.items())}


def cmd_score_g6(a):
    """G6 = the L -> TTFT table and its least-squares slope (descriptive).  PASS = >= 3 distinct L points
    were measured; the J1 floor slope is printed beside it, never compared."""
    facts = {"j1_floor_ms_per_token_reference": J1_FLOOR_MS_PER_TOKEN, "by_input": {}}
    n_pts = 0
    for name, path in (("nano_pdmux_G2", a.nano), ("llama_G5", a.llama)):
        res = _load(path) if path else None
        if not res:
            facts["by_input"][name] = None
            continue
        m = mean_ttft_by_L(res)
        sl = lstsq_slope([float(L) for L in m], list(m.values()))
        facts["by_input"][name] = {"mean_ttft_s_by_L": {str(L): v for L, v in m.items()},
                                   "slope_ms_per_token": sl["slope"] * 1000 if sl else None, "n_points": len(m)}
        if name == "nano_pdmux_G2":
            n_pts = len(m)
    if n_pts >= 3:
        write_status(a.status_dir, "G6", PASS, f"{n_pts} distinct L points measured (descriptive table in facts)", facts)
    else:
        write_status(a.status_dir, "G6", UNRES, f"only {n_pts} L points with ok requests", facts)
    return 0


def cmd_summary(a):
    rd = a.rundir
    st = {}
    sd = os.path.join(rd, "status")
    for n in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
        v = _load(os.path.join(sd, n))
        if v:
            st[v["stage"]] = v
    meta = _load(os.path.join(rd, "meta", "run.json")) or {}
    _dump({"kind": "longctx_probe_summary", "utc": _utc(), "meta": meta, "stages": st},
          os.path.join(rd, "summary.json"))
    L = [f"# Longctx 128k probe SUMMARY — {os.path.basename(rd)}", "",
         "> 기계적 사실만(PASS/FAIL/UNRESOLVED/SKIPPED). TTFT·ITL 값은 기술값이고 인용 금지, Job 예산 입력용이다.",
         "> 원천: `status/*.json`, 요청별 시계열은 `g*/…json`. 성능 판정 없음 -> result-analyst.", "",
         f"- mode: {meta.get('mode')} · commit: `{meta.get('commit')}` (dirty={meta.get('git_dirty')}) · host: {meta.get('host')}",
         f"- budget_min {meta.get('budget_min')} · keep_going {meta.get('keep_going')} · start {meta.get('start_utc')} · end {_utc()}", "",
         "| stage | status | reason |", "|---|---|---|"]
    for s in STAGE_ORDER:
        v = st.get(s)
        if v:
            L.append(f"| {s} | **{v['status']}** | {str(v.get('reason', '')).replace('|', '/')[:300]} |")
    for s in ("G2", "G3", "G3-chunked", "G5"):
        v = st.get(s)
        if v and (v.get("facts") or {}).get("per_L"):
            L.append("")
            L.append(f"- {s} per-L ok/fail: " + ", ".join(f"{k}: {d['ok']}/{d['fail']}" for k, d in
                                                           sorted(v["facts"]["per_L"].items(), key=lambda kv: int(kv[0]))))
    L.append("")
    with open(os.path.join(rd, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s, tok=True):
        s.add_argument("--port", type=int, required=True)
        s.add_argument("--srvlog", required=True)
        s.add_argument("--arm", required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--req-timeout", type=float, default=900.0)
        s.add_argument("--out-tokens", type=int, default=OUT_TOKENS)
        if tok:
            s.add_argument("--tokenizer", required=True)

    s = sub.add_parser("pre-models")
    s.add_argument("--hf-home", default="/data/hf")
    s.add_argument("--nano", required=True)
    s.add_argument("--llama", default=LLAMA_ID)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_pre_models)
    s = sub.add_parser("boot-facts")
    s.add_argument("--srvlog", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--cap", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_boot_facts)
    s = sub.add_parser("probe-single")
    common(s)
    s.add_argument("--stage", required=True)
    s.add_argument("--Ls", default=",".join(str(x) for x in L_GRID))
    s.add_argument("--reps", type=int, default=2)
    s.set_defaults(fn=cmd_probe_single)
    s = sub.add_parser("probe-conc")
    common(s)
    s.add_argument("--stage", default="G5-conc")
    s.add_argument("--Ls", default=f"{64 * K},{124 * K}")
    s.add_argument("--boot-facts", required=True)
    s.add_argument("--max-conc", type=int, default=32)
    s.set_defaults(fn=cmd_probe_conc)
    s = sub.add_parser("probe-g4")
    common(s)
    s.add_argument("--Ls", default=f"{96 * K},{124 * K}")
    s.add_argument("--n-decode", type=int, default=4)
    s.add_argument("--decode-in", type=int, default=256)
    s.add_argument("--decode-out", type=int, default=1024)
    s.add_argument("--lead-tokens", type=int, default=16)
    s.add_argument("--lead-timeout-s", type=float, default=60.0)
    s.set_defaults(fn=cmd_probe_g4)
    s = sub.add_parser("probe-g7")
    common(s)
    s.add_argument("--lb-zip", required=True)
    s.add_argument("--max-tokens", type=int, default=64 * K)
    s.add_argument("--n-docs", type=int, default=2)
    s.set_defaults(fn=cmd_probe_g7)
    for name, fn in (("score-single", cmd_score_single), ("score-conc", cmd_score_conc)):
        s = sub.add_parser(name)
        s.add_argument("--in", dest="inp", required=True)
        s.add_argument("--stage", required=True)
        s.add_argument("--expect-n", type=int, required=True)
        s.add_argument("--status-dir", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("score-g4")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--Ls", default=f"{96 * K},{124 * K}")
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_score_g4)
    s = sub.add_parser("score-g7")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_score_g7)
    s = sub.add_parser("score-g6")
    s.add_argument("--nano")
    s.add_argument("--llama")
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_score_g6)
    s = sub.add_parser("summary")
    s.add_argument("--rundir", required=True)
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
