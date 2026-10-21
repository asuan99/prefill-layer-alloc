#!/usr/bin/env python3
"""compo_probe_helper.py -- mechanics for scripts/vessl/compo_probe.sh (composition-variant boot probe).

Decision 24 (results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md): before the same-skeleton
composition sweep (J-H) is designed, check that Nemotron-Nano-9B-v2-Base variants whose
`hybrid_override_pattern` turns Mamba layers into attention boot under SGLang v0.5.10 with
`--load-format dummy`.  NON-MEASUREMENT probe: every number here is a descriptive probe value, not
citable; there is no performance judgment anywhere in this file.

Statuses (mechanical): PASS | FAIL | UNRESOLVED | SKIPPED.
  BOOT-<b>  /health reached, load_format as intended, context_len as intended, cudagraph capture logged
            with no failure/eager-fallback signature, PD-mux stream-group line present iff PD-mux boot.
  REQ-<b>   8 concurrent synthetic requests (2048 input ids, 64 forced output tokens, greedy, ignore_eos)
            complete with exact token counts and finish=length, no new server error signature, and one
            extra request with return_logprob has 64 finite output logprobs (NaN/inf guard for dummy
            weights; orjson renders NaN/inf as null, so null counts as non-finite).
  PRED-<b>  banner vs the variant's [유도] prediction: weight bytes, KV bytes/token k, Mamba state/slot s
            each within +-2 % (exact-zero rule when the prediction is 0), and the layer counts implied by
            the banner equal the variant's n_attn / n_mamba (evidence that the variant config was read).
  TEL-<b>   (TEL ON boots only: dummy_a13, dummy_a31, dummy_a31_retry; decision 26) the PREREG rev3.3 sec 4'
            J-H2 T3 sigma -- PD-mux `prefill_span_start`/`prefill_span_end` pairs (PDMUX_PHASE_EVENTS=1 +
            PDMUX_TELEMETRY_PATH, src/multiplex/multiplexing_mixin.py `_phase_on_prefill_start/_end`) -- is
            emitted on a dummy variant: 8 decode requests (2048 in, 1024 forced out) run, then ONE 8192-token
            prefill joins them.  PASS iff tel.jsonl holds >=1 start/end pair with the same span_seq, a paired
            span with extend_tokens == 8192, >=1 decode_iteration record, all requests completed, and the server
            log has no "PDMUX_PHASE_EVENTS disabled after failure" / "telemetry disabled after write failure"
            line.  UNRESOLVED if all that holds but no decode_iteration with prefill_in_flight=true was launched
            inside the 8192 span (co-residence not evidenced).  Existence check only, not a measurement.

Pure functions are unit-tested on CPU (tests/test_compo_variant.py).
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LH = _load_mod("longctx_probe_helper", os.path.join(HERE, "longctx_probe_helper.py"))      # banner, synth, io
MV = _load_mod("make_variant", os.path.join(HERE, "..", "compo", "make_variant.py"))

SCHEMA = "compo_probe/v1"
PASS, FAIL, UNRES, SKIP = LH.PASS, LH.FAIL, LH.UNRES, LH.SKIP
GIB = 1024 ** 3
CTX = 73728
IN_LEN = 2048
OUT_TOKENS = 64
N_REQ = 8
TOL = 0.02
BASE_SEED = 20261021
BOOTS = ["real_orig", "dummy_a4", "dummy_a13", "dummy_a22", "dummy_a31", "dummy_a31_retry", "dummy_a4_vanilla"]
TEL_BOOTS = ["dummy_a13", "dummy_a31", "dummy_a31_retry"]       # decision 26; must equal compo_probe.sh TEL_BOOTS
TEL_L = 8192                     # the co-resident prefill
TEL_N_DEC = 8                    # decode batch
TEL_DEC_IN = 2048                # decode context
TEL_DEC_OUT = 1024               # forced decode length (ignore_eos): long enough to outlive the 8192 prefill
TEL_PF_OUT = 8
TEL_SEED = BASE_SEED + 26
PHASE_EVENTS = ("prefill_span_start", "prefill_span_end", "decode_iteration")   # multiplexing_mixin PHASE_EVENT_TYPES
TEL_FAIL_LOG = ("PDMUX_PHASE_EVENTS disabled after failure", "telemetry disabled after write failure")
BOOT_VARIANT = {"real_orig": 4, "dummy_a4": 4, "dummy_a13": 13, "dummy_a22": 22, "dummy_a31": 31,
                "dummy_a31_retry": 31, "dummy_a4_vanilla": 4}
STAGE_ORDER = ["PRE", "VARIANTS"] + [f"{p}-{b}" for b in BOOTS
                                     for p in (("BOOT", "REQ", "TEL", "PRED", "TD") if b in TEL_BOOTS
                                               else ("BOOT", "REQ", "PRED", "TD"))]
SYNTH_RULE = dict(LH.SYNTH_RULE, seed=f"request j uses splitmix64(({BASE_SEED} ^ ({IN_LEN} << 8) ^ j)); "
                                      f"j = 0..{N_REQ - 1} plain, j = {N_REQ} logprob request")

_ERR_PATS = {"nan_logits": r"NaN in the logits|probability tensor contains either|nan or inf",
             "device_assert": r"device-side assert|CUDA error"}


# ===========================================================================
# boot-log facts (pure)
# ===========================================================================
def parse_boot_log(text):
    text = text or ""
    f = LH.parse_banner(text)
    m = re.findall(r"Load weight end\..*?mem usage=([0-9.]+) *GB", text)
    f["weight_mem_usage_gib"] = float(m[-1]) if m else None
    f["cuda_graph_bs_lists"] = [[int(x) for x in re.findall(r"[0-9]+", g)]
                                for g in re.findall(r"Capture cuda graph bs \[([0-9, ]*)\]", text)]
    f["cuda_graph_capture_end"] = len(re.findall(r"Capture cuda graph end", text))
    m = re.findall(r"PD-Multiplexing enabled with ([0-9]+) stream groups, sm_counts \(prefill_sm, decode_sm\): (\[.*?\])\s*$",
                   text, re.M)
    f["pdmux_stream_groups"] = int(m[-1][0]) if m else None
    f["pdmux_sm_counts"] = m[-1][1] if m else None
    m = re.findall(r"load_format='([^']*)'", text)
    f["load_format_logged"] = m[-1] if m else None
    m = re.findall(r"model_path='([^']*)'", text)
    f["model_path_logged"] = m[-1] if m else None
    return f


def signatures(text):
    sig = LH.log_signatures(text)
    sig.update({k: len(re.findall(p, text or "")) for k, p in _ERR_PATS.items()})
    return sig


def banner_terms(f):
    """Banner-derived (w_bytes, k, s, slots) -- any may be None if its banner line is missing."""
    w = f["weight_mem_usage_gib"] * GIB if f.get("weight_mem_usage_gib") is not None else None
    k = None
    if f.get("kv_cache_tokens") and f.get("kv_k_gib") is not None:
        k = (f["kv_k_gib"] + f["kv_v_gib"]) * GIB / f["kv_cache_tokens"]
    s, slots = None, None
    if f.get("max_mamba_cache_size") is not None:
        slots = f["max_mamba_cache_size"] + 1          # one padding slot (verified: 48 -> 49 slots, 6.46 GiB)
        s = ((f.get("mamba_conv_gib") or 0.0) + (f.get("mamba_ssm_gib") or 0.0)) * GIB / slots
    return w, k, s, slots


def rel_check(pred, obs, tol=TOL, zero_abs=None):
    """-> (verdict, rel_err).  verdict in {PASS, FAIL, UNRESOLVED}.  pred == 0 uses an absolute bound."""
    if obs is None:
        return UNRES, None
    if pred == 0:
        bound = zero_abs if zero_abs is not None else 0.0
        return (PASS if abs(obs) <= bound else FAIL), None
    r = (obs - pred) / pred
    return (PASS if abs(r) <= tol else FAIL), r


def compare_prediction(pred, facts, kv_per_layer_token, ssm_layer_bytes):
    """PRED stage: banner vs [유도] prediction; also layer counts implied by the banner."""
    w, k, s, slots = banner_terms(facts)
    out = {"tolerance": TOL, "banner": {"weight_bytes": w, "k_bytes_per_token": k, "s_bytes_per_slot": s,
                                        "mamba_slots": slots},
           "predicted": {"weight_bytes": pred["weight_bytes"], "k_bytes_per_token": pred["kv_bytes_per_token_k"],
                         "s_bytes_per_req": pred["state_bytes_per_req_s"]}, "checks": {}}
    v, r = rel_check(pred["weight_bytes"], w)
    out["checks"]["weight"] = {"verdict": v, "rel_err": r}
    # banner GB values carry 2 decimals: an exact-zero prediction tolerates <= 0.01 GiB over all slots
    v, r = rel_check(pred["kv_bytes_per_token_k"], k)
    out["checks"]["k"] = {"verdict": v, "rel_err": r}
    if pred["state_bytes_per_req_s"] == 0 and s is None:
        out["checks"]["s"] = {"verdict": PASS, "rel_err": None, "note": "prediction 0 and no Mamba pool banner"}
    else:
        v, r = rel_check(pred["state_bytes_per_req_s"], s,
                         zero_abs=(0.01 * GIB / slots) if slots else 0.0)
        out["checks"]["s"] = {"verdict": v, "rel_err": r}
    implied = {"n_attn": round(k / kv_per_layer_token) if k is not None else None,
               "n_mamba": (round(facts["mamba_ssm_gib"] * GIB / (slots * ssm_layer_bytes))
                           if slots and facts.get("mamba_ssm_gib") is not None else (0 if s is None else None))}
    out["implied_layer_counts"] = implied
    lc = UNRES if None in implied.values() else (
        PASS if implied["n_attn"] == pred["n_attn"] and implied["n_mamba"] == pred["n_mamba"] else FAIL)
    out["checks"]["layer_counts"] = {"verdict": lc, "expected": {"n_attn": pred["n_attn"], "n_mamba": pred["n_mamba"]}}
    vs = [c["verdict"] for c in out["checks"].values()]
    out["verdict"] = FAIL if FAIL in vs else (PASS if all(x == PASS for x in vs) else UNRES)
    return out


def boot_verdict(facts, sig, healthy, want_dummy, want_pdmux, ctx=CTX):
    """-> (status, reason)."""
    if not healthy:
        return FAIL, "server did not reach /health (died, boot timeout or budget)"
    if sig.get("cuda_graph_capture_failed") or sig.get("eager_fallback"):
        return FAIL, "cudagraph capture failure / eager fallback signature in the boot log"
    lf = facts.get("load_format_logged")
    if lf is None:
        return UNRES, "server_args line with load_format not found in the log"
    if (lf == "dummy") != bool(want_dummy):
        return FAIL, f"load_format={lf!r} but this boot wants {'dummy' if want_dummy else 'real weights'}"
    if facts.get("context_len_logged") != ctx:
        return FAIL, f"context_len logged {facts.get('context_len_logged')} != {ctx}"
    if not facts.get("cuda_graph_bs_lists") or not facts.get("cuda_graph_capture_end"):
        return UNRES, "no 'Capture cuda graph bs' / 'Capture cuda graph end' line (cudagraph-ON not evidenced)"
    has = facts.get("pdmux_stream_groups") is not None
    if has != bool(want_pdmux):
        return FAIL, f"PD-mux stream-group line present={has} but PD-mux wanted={want_pdmux}"
    return PASS, (f"booted, load_format={lf}, cudagraph bs max {max(facts['cuda_graph_bs_lists'][-1] or [0])}, "
                  f"pdmux groups={facts.get('pdmux_stream_groups')}")


# ===========================================================================
# requests
# ===========================================================================
def post_generate(port, input_ids, max_new, timeout, logprob=False, host="127.0.0.1"):
    body = {"input_ids": list(input_ids),
            "sampling_params": {"max_new_tokens": int(max_new), "temperature": 0.0, "ignore_eos": True}}
    if logprob:
        body.update({"return_logprob": True, "logprob_start_len": -1})
    rec = {"error": None, "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
           "e2e_s": None, "output_ids_sha256": None, "logprobs": None}
    t0 = time.perf_counter()
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/generate", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            rec["error"] = f"HTTP {resp.status}: {data[:300].decode('utf-8', 'replace')}"
            return rec
        obj = json.loads(data)
        if isinstance(obj, list):
            obj = obj[0]
        mi = obj.get("meta_info") or {}
        rec["prompt_tokens"] = mi.get("prompt_tokens")
        rec["completion_tokens"] = mi.get("completion_tokens")
        rec["finish_reason"] = mi.get("finish_reason")
        oid = obj.get("output_ids")
        if oid is not None:
            rec["output_ids_sha256"] = LH.ids_sha256(oid)
        if logprob:
            rec["logprobs"] = [e[0] if isinstance(e, (list, tuple)) and e else e
                               for e in (mi.get("output_token_logprobs") or [])]
        if "error" in obj:
            rec["error"] = str(obj["error"])[:300]
    except Exception as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:300]
    finally:
        rec["e2e_s"] = time.perf_counter() - t0
        if conn is not None:
            conn.close()
    return rec


def logprob_finite(vals, n_out=OUT_TOKENS):
    """-> (status, n_nonfinite).  None/NaN/inf all count as non-finite; a short list is UNRESOLVED."""
    if vals is None or len(vals) < n_out:
        return UNRES, None
    bad = sum(1 for v in vals if v is None or not isinstance(v, (int, float)) or not math.isfinite(v))
    return (PASS if bad == 0 else FAIL), bad


def req_verdict(res, sig_delta):
    plain = res["plain"]
    ok = [LH.request_ok(r, IN_LEN, OUT_TOKENS) for r in plain]
    bad_sig = {k: v for k, v in sig_delta.items()
               if v > 0 and k not in ("cuda_graph_capture_failed", "eager_fallback")}
    if len(plain) != N_REQ or not all(ok):
        return FAIL, f"{sum(ok)}/{N_REQ} plain requests completed with {IN_LEN} in / {OUT_TOKENS} out / finish=length"
    if bad_sig:
        return FAIL, f"server log gained error signatures during the requests: {bad_sig}"
    lp = res.get("logprob")
    st, nbad = logprob_finite(lp.get("logprobs") if lp else None)
    if lp is None or lp.get("error") is not None or st == UNRES:
        return UNRES, (f"8/8 plain requests ok, no error signature; logprob NaN/inf check unresolved "
                       f"({(lp or {}).get('error')})")
    if st == FAIL:
        return FAIL, f"8/8 plain requests ok but {nbad}/{OUT_TOKENS} output logprobs non-finite or null (NaN/inf)"
    return PASS, f"8/8 plain requests ok, no error signature, {OUT_TOKENS}/{OUT_TOKENS} output logprobs finite"


# ===========================================================================
# TEL ON check (decision 26) -- pure parts
# ===========================================================================
def read_jsonl(path):
    """-> (records or None if the file is missing, n_unparsable_lines).  A torn last line is counted, not fatal."""
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, 0
    recs, bad = [], 0
    for ln in lines:
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            recs.append(r)
        else:
            bad += 1
    return recs, bad


def tel_facts(records, l_prefill=TEL_L):
    """Pair prefill spans by span_seq and locate the l_prefill span (records = flat telemetry dicts)."""
    starts, ends = {}, {}
    n_dec, dec = 0, []
    for r in records:
        ev = r.get("event")
        if ev == "prefill_span_start" and r.get("span_seq") is not None:
            starts.setdefault(r["span_seq"], r)
        elif ev == "prefill_span_end" and r.get("span_seq") is not None:
            ends.setdefault(r["span_seq"], r)
        elif ev == "decode_iteration":
            n_dec += 1
            dec.append(r)
    pairs = sorted(set(starts) & set(ends))
    target = [q for q in pairs if starts[q].get("extend_tokens") == l_prefill]
    co = 0
    for q in target:
        t0, t1 = starts[q].get("timestamp_monotonic_s"), ends[q].get("timestamp_monotonic_s")
        if t0 is None or t1 is None:
            continue
        co += sum(1 for d in dec if d.get("prefill_in_flight") is True
                  and d.get("t_launch") is not None and t0 <= d["t_launch"] <= t1)
    return {"n_records": len(records), "n_span_start": len(starts), "n_span_end": len(ends),
            "n_pairs": len(pairs), "target_span_seq": target,
            "target_extend_tokens": [starts[q].get("extend_tokens") for q in target],
            "target_n_new": [starts[q].get("n_new") for q in target],
            "extend_tokens_seen": sorted({starts[q].get("extend_tokens") for q in pairs},
                                         key=lambda x: (x is None, x)),
            "n_decode_iteration": n_dec, "max_decode_bs": max((d.get("bs") or 0 for d in dec), default=None),
            "n_coresident_decode_iter": co}


def tel_verdict(records, n_bad, srvlog_text, req_errors=(), l_prefill=TEL_L):
    """-> (status, reason, facts).  records None = tel.jsonl missing."""
    hits = [p for p in TEL_FAIL_LOG if p in (srvlog_text or "")]
    if records is None:
        return FAIL, "tel.jsonl missing (PDMUX_TELEMETRY_PATH not honoured?)", {"server_log_failure_lines": hits}
    f = tel_facts(records, l_prefill)
    f.update({"n_unparsable_lines": n_bad, "server_log_failure_lines": hits, "request_errors": list(req_errors)})
    why = []
    if hits:
        why.append(f"server log: {hits}")
    if req_errors:
        why.append(f"{len(req_errors)} TEL request(s) did not complete as forced: {list(req_errors)[:3]}")
    if f["n_pairs"] < 1:
        why.append(f"no prefill_span_start/end pair with the same span_seq ({f['n_span_start']} starts, "
                   f"{f['n_span_end']} ends)")
    elif not f["target_span_seq"]:
        why.append(f"no paired span with extend_tokens == {l_prefill} (seen {f['extend_tokens_seen'][:8]})")
    if f["n_decode_iteration"] < 1:
        why.append("no decode_iteration record")
    if why:
        return FAIL, "; ".join(why), f
    base = (f"{f['n_pairs']} span pair(s), span_seq {f['target_span_seq']} has extend_tokens {l_prefill}, "
            f"{f['n_decode_iteration']} decode_iteration records (max bs {f['max_decode_bs']})")
    if f["n_coresident_decode_iter"] < 1:
        return UNRES, base + "; but no decode_iteration with prefill_in_flight launched inside that span", f
    return PASS, base + f", {f['n_coresident_decode_iter']} co-resident decode iterations in it", f


# ===========================================================================
# subcommands
# ===========================================================================
def cmd_variants(a):
    """Create the variants from the original snapshot and record SGLang's own reading of each config."""
    ref, rev, snap = LH.snapshot_dir(a.hf_home, MV.NANO_ID)
    out = {"kind": "compo_variants", "utc": LH._utc(), "snapshot": snap, "refs_main": rev,
           "revision_expected": MV.NANO_REV, "variants": {}}
    status, why = PASS, []
    if rev != MV.NANO_REV or not snap or not os.path.isdir(snap):
        status, why = FAIL, [f"refs/main={rev!r} != pinned {MV.NANO_REV} or snapshot missing"]
    else:
        names = os.listdir(snap)
        out["n_safetensors_in_snapshot"] = sum(n.endswith(".safetensors") for n in names)
        if out["n_safetensors_in_snapshot"] == 0:
            status, why = FAIL, ["original snapshot has no safetensors (real_orig boot impossible)"]
        for n_attn in sorted(set(int(x) for x in a.n_attn.split(","))):
            d = os.path.join(a.out_root, f"a{n_attn}")
            try:
                rec = MV.build_variant(snap, d, n_attn, force=True, source_meta={"revision_observed": rev})
                v = {"dir": d, "pattern": rec["pattern"], "n_attn": rec["n_attn"], "n_mamba": rec["n_mamba"],
                     "changed_layers": rec["changed_layers"], "config_sha256": rec["config_sha256"],
                     "weights_in_dir": [n for n in os.listdir(d) if MV.is_weight_file(n)]}
                if v["weights_in_dir"]:
                    status, why = FAIL, why + [f"a{n_attn}: weight files copied"]
                v["sglang_config"] = sglang_reads(d) if a.sglang_check else None
                sc = v["sglang_config"]
                if sc is not None and (sc.get("error") or sc["pattern"] != rec["pattern"]
                                       or sc["n_mamba_ids"] != rec["n_mamba"] or sc["n_attn_ids"] != rec["n_attn"]
                                       or sc["mamba_cache_per_req"] != rec["prediction"]["state_bytes_per_req_s"]):
                    status, why = FAIL, why + [f"a{n_attn}: SGLang config read disagrees ({sc})"]
                out["variants"][f"a{n_attn}"] = v
            except Exception as exc:  # noqa: BLE001
                status, why = FAIL, why + [f"a{n_attn}: {exc!r}"]
    LH._dump(out, a.out)
    reason = "; ".join(why) if why else (
        f"{len(out['variants'])} variants built from {MV.NANO_REV[:12]}, no weight files, "
        + ("SGLang get_config (trust_remote_code) reads each pattern and Mamba state size as predicted"
           if a.sglang_check else "SGLang config read NOT checked"))
    LH.write_status(a.status_dir, "VARIANTS", status, reason, out)
    return 0 if status == PASS else 1


def sglang_reads(d):
    """SGLang's own config path (AutoConfig with trust_remote_code -> _CONFIG_REGISTRY class)."""
    try:
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype
        from sglang.srt.utils.hf_transformers_utils import get_config
        c = get_config(d, trust_remote_code=True)
        sh = Mamba2StateShape.create(tp_world_size=1, intermediate_size=c.mamba_num_heads * c.mamba_head_dim,
                                     n_groups=c.n_groups, num_heads=c.mamba_num_heads, head_dim=c.mamba_head_dim,
                                     state_size=c.ssm_state_size, conv_kernel=c.conv_kernel)
        cp = Mamba2CacheParams(shape=sh, layers=c.mamba_layer_ids, dtype=mamba2_state_dtype(c))
        return {"config_class": type(c).__module__ + "." + type(c).__name__, "pattern": c.hybrid_override_pattern,
                "n_mamba_ids": len(c.mamba_layer_ids), "n_attn_ids": len(c.full_attention_layer_ids),
                "mamba_cache_per_req": int(cp.mamba_cache_per_req), "state_dtype": str(cp.dtype), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:400], "pattern": None, "n_mamba_ids": None, "n_attn_ids": None,
                "mamba_cache_per_req": None}


def cmd_boot_facts(a):
    text = LH._read(a.srvlog)
    facts = parse_boot_log(text)
    sig = signatures(text)
    st, why = boot_verdict(facts, sig, a.healthy == 1, a.want_dummy == 1, a.want_pdmux == 1)
    rec = {"kind": "compo_boot_facts", "boot": a.boot, "utc": LH._utc(), "facts": facts, "signatures": sig,
           "healthy": a.healthy == 1, "want_dummy": a.want_dummy == 1, "want_pdmux": a.want_pdmux == 1}
    LH._dump(rec, a.out)
    LH.write_status(a.status_dir, f"BOOT-{a.boot}", st, why, rec)
    return 0


def cmd_requests(a):
    encode, _ = LH.load_encoder(a.tokenizer)
    sig0 = signatures(LH._read(a.srvlog))
    ids = [LH.synth_ids(encode, IN_LEN, LH.seed_for(IN_LEN, j, base=BASE_SEED)) for j in range(N_REQ + 1)]
    res = {"schema": SCHEMA, "kind": "compo_requests", "boot": a.boot, "utc": LH._utc(), "in_len": IN_LEN,
           "out_tokens": OUT_TOKENS, "n_req": N_REQ, "synth_rule": SYNTH_RULE,
           "ids_sha256": [LH.ids_sha256(x) for x in ids], "plain": [None] * N_REQ, "logprob": None}
    for x in ids:
        LH.check_budget(len(x), OUT_TOKENS, CTX)

    def one(j):
        res["plain"][j] = post_generate(a.port, ids[j], OUT_TOKENS, a.req_timeout)

    th = [threading.Thread(target=one, args=(j,)) for j in range(N_REQ)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    res["plain"] = [r if r is not None else {"error": "thread produced no record"} for r in res["plain"]]
    LH._dump(res, a.out)
    res["logprob"] = post_generate(a.port, ids[N_REQ], OUT_TOKENS, a.req_timeout, logprob=True)
    time.sleep(2)                                   # let the server flush its log lines
    sig1 = signatures(LH._read(a.srvlog))
    res["sig_delta"] = LH.sig_delta(sig0, sig1)
    res["server_alive_after"] = LH.server_alive(a.port)
    LH._dump(res, a.out)
    st, why = req_verdict(res, res["sig_delta"])
    LH.write_status(a.status_dir, f"REQ-{a.boot}", st, why,
                    {"sig_delta": res["sig_delta"], "server_alive_after": res["server_alive_after"],
                     "plain_ok": [LH.request_ok(r, IN_LEN, OUT_TOKENS) for r in res["plain"]],
                     "logprob_nonfinite": logprob_finite((res["logprob"] or {}).get("logprobs"))[1]})
    return 0


def cmd_tel(a):
    """Decode batch running, then one TEL_L prefill joins it; judge tel.jsonl (no timing judgment)."""
    encode, _ = LH.load_encoder(a.tokenizer)
    dec_ids = [LH.synth_ids(encode, TEL_DEC_IN, LH.seed_for(TEL_DEC_IN, j, base=TEL_SEED)) for j in range(TEL_N_DEC)]
    pf_ids = LH.synth_ids(encode, TEL_L, LH.seed_for(TEL_L, 0, base=TEL_SEED))
    for x in dec_ids:
        LH.check_budget(len(x), TEL_DEC_OUT, CTX)
    LH.check_budget(len(pf_ids), TEL_PF_OUT, CTX)
    res = {"schema": SCHEMA, "kind": "compo_tel", "boot": a.boot, "utc": LH._utc(), "tel_path": a.tel,
           "l_prefill": TEL_L, "n_dec": TEL_N_DEC, "dec_in": TEL_DEC_IN, "dec_out": TEL_DEC_OUT,
           "dec_ids_sha256": [LH.ids_sha256(x) for x in dec_ids], "pf_ids_sha256": LH.ids_sha256(pf_ids),
           "decode": [None] * TEL_N_DEC, "prefill": None, "waited_for_decode_s": None, "decode_seen_bs": None}

    def one(j):
        res["decode"][j] = post_generate(a.port, dec_ids[j], TEL_DEC_OUT, a.req_timeout)

    # REQ-<b> already wrote bs-8 decode_iteration records into the same file: only look past them
    time.sleep(2)
    n0 = len(read_jsonl(a.tel)[0] or [])
    res["n_records_before"] = n0
    th = [threading.Thread(target=one, args=(j,)) for j in range(TEL_N_DEC)]
    for t in th:
        t.start()
    # start the prefill only once the full decode batch is iterating (telemetry flushes at least every 1 s)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.wait_decode_s:
        recs, _ = read_jsonl(a.tel)
        bs = [r.get("bs") or 0 for r in (recs or [])[n0:] if r.get("event") == "decode_iteration"]
        if bs and max(bs[-50:]) >= TEL_N_DEC:
            res["decode_seen_bs"] = max(bs[-50:])
            break
        time.sleep(0.5)
    res["waited_for_decode_s"] = time.perf_counter() - t0
    res["prefill"] = post_generate(a.port, pf_ids, TEL_PF_OUT, a.req_timeout)
    for t in th:
        t.join()
    res["decode"] = [r if r is not None else {"error": "thread produced no record"} for r in res["decode"]]
    time.sleep(3)                                   # telemetry writer flushes every 1 s when idle
    errs = [f"decode[{j}]" for j, r in enumerate(res["decode"]) if not LH.request_ok(r, TEL_DEC_IN, TEL_DEC_OUT)]
    if not LH.request_ok(res["prefill"], TEL_L, TEL_PF_OUT):
        errs.append("prefill")
    recs, nbad = read_jsonl(a.tel)
    st, why, facts = tel_verdict(recs, nbad, LH._read(a.srvlog), errs)
    if res["decode_seen_bs"] is None:
        why += f" [prefill sent after {res['waited_for_decode_s']:.0f} s without seeing decode bs {TEL_N_DEC}]"
    res["facts"] = facts
    res["server_alive_after"] = LH.server_alive(a.port)
    LH._dump(res, a.out)
    LH.write_status(a.status_dir, f"TEL-{a.boot}", st, why, facts)
    return 0


def cmd_pred(a):
    var = LH._load(a.variant_json)
    bf = LH._load(a.boot_facts)
    if var is None or bf is None:
        LH.write_status(a.status_dir, f"PRED-{a.boot}", UNRES, "variant.json or boot_facts.json missing")
        return 0
    pred = var["prediction"]
    cfg = LH._load(os.path.join(os.path.dirname(a.variant_json), "config.json")) or {}
    kv_layer = int(cfg.get("num_key_value_heads", 8)) * int(cfg.get("head_dim", 128)) * 2 * MV.KV_ITEMSIZE
    ssm_layer = pred["ssm_state_bytes_per_layer"]
    cmp_ = compare_prediction(pred, bf["facts"], kv_layer, ssm_layer)
    cmp_.update({"boot": a.boot, "variant": os.path.dirname(a.variant_json), "note": "probe value, not citable"})
    LH._dump(cmp_, a.out)
    c = cmp_["checks"]
    fmt = lambda x: "NA" if x is None else f"{x:+.3%}"   # noqa: E731
    why = (f"weight {fmt(c['weight']['rel_err'])}, k {fmt(c['k']['rel_err'])}, s {fmt(c['s']['rel_err'])} "
           f"(tol +-{TOL:.0%}); implied n_attn/n_mamba {cmp_['implied_layer_counts']['n_attn']}/"
           f"{cmp_['implied_layer_counts']['n_mamba']} vs {pred['n_attn']}/{pred['n_mamba']}")
    LH.write_status(a.status_dir, f"PRED-{a.boot}", cmp_["verdict"], why, cmp_)
    return 0


def cmd_summary(a):
    rd = a.rundir
    st = {}
    for s in STAGE_ORDER:
        v = LH._load(os.path.join(rd, "status", f"{s}.json"))
        if v:
            st[s] = v
    meta = LH._load(os.path.join(rd, "meta", "run.json")) or {}
    LH._dump({"schema": SCHEMA, "run_id": meta.get("run_id"), "statuses": {k: v["status"] for k, v in st.items()}},
             os.path.join(rd, "summary.json"))
    L = [f"# compo_probe {meta.get('run_id')}", "",
         "> **비측정 부팅 프로브(결정 24).** 아래 수치(가중치 GB·KV 풀 토큰·mamba 슬롯·cudagraph bs·요청 시간)는 "
         "모두 **인용 불가 프로브 값**이다. 성능 판정 없음. dummy 가중치 출력은 무의미하며 정확성 검사는 원본 실가중치에서만 한다.",
         "> 원천: `status/*.json`, boot별 `<boot>/boot_facts.json`·`requests.json`·`pred.json`, 변형 `variants/a*/variant.json`.",
         "", f"- mode: {meta.get('mode')} · commit: `{meta.get('commit')}` (dirty={meta.get('git_dirty')}) · "
         f"host: {meta.get('host')} · budget_min {meta.get('budget_min')} · cap {meta.get('cap')} · ctx {meta.get('context_length')}",
         "", "| stage | status | reason |", "|---|---|---|"]
    for s in STAGE_ORDER:
        v = st.get(s)
        if v:
            L.append(f"| {s} | **{v['status']}** | {str(v.get('reason', '')).replace('|', '/')[:300]} |")
    L += ["", "| boot | weight GiB (banner) | KV tokens | mamba slots | s/slot B | k B/tok | cudagraph bs max | PD-mux groups |",
          "|---|---|---|---|---|---|---|---|"]
    for b in BOOTS:
        bf = LH._load(os.path.join(rd, b, "boot_facts.json"))
        if not bf:
            continue
        f = bf["facts"]
        w, k, s, slots = banner_terms(f)
        bs = max(f["cuda_graph_bs_lists"][-1]) if f.get("cuda_graph_bs_lists") and f["cuda_graph_bs_lists"][-1] else None
        L.append(f"| {b} | {f.get('weight_mem_usage_gib')} | {f.get('kv_cache_tokens')} | {f.get('max_mamba_cache_size')} | "
                 f"{'NA' if s is None else round(s)} | {'NA' if k is None else round(k)} | {bs} | {f.get('pdmux_stream_groups')} |")
    L += ["", "TEL ON 확인(결정 26, 존재 검사 — 측정 아님). 나머지 boot는 계측 OFF(결정 22).", "",
          "| boot | span pairs | extend_tokens=8192 span_seq | decode_iteration | co-resident decode iter | log failure lines |",
          "|---|---|---|---|---|---|"]
    for b in TEL_BOOTS:
        tf = LH._load(os.path.join(rd, b, "tel_check.json"))
        if not tf:
            continue
        f = tf.get("facts") or {}
        L.append(f"| {b} | {f.get('n_pairs')} | {f.get('target_span_seq')} | {f.get('n_decode_iteration')} | "
                 f"{f.get('n_coresident_decode_iter')} | {len(f.get('server_log_failure_lines') or [])} |")
    L.append("")
    with open(os.path.join(rd, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("variants")
    s.add_argument("--hf-home", default="/data/hf")
    s.add_argument("--n-attn", default="4,13,22,31")
    s.add_argument("--out-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--status-dir", required=True)
    s.add_argument("--sglang-check", type=int, default=1)
    s.set_defaults(fn=cmd_variants)
    s = sub.add_parser("boot-facts")
    s.add_argument("--srvlog", required=True)
    s.add_argument("--boot", required=True)
    s.add_argument("--healthy", type=int, required=True)
    s.add_argument("--want-dummy", type=int, required=True)
    s.add_argument("--want-pdmux", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_boot_facts)
    s = sub.add_parser("requests")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--srvlog", required=True)
    s.add_argument("--boot", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--req-timeout", type=float, default=300.0)
    s.add_argument("--out", required=True)
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_requests)
    s = sub.add_parser("tel")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--srvlog", required=True)
    s.add_argument("--tel", required=True)
    s.add_argument("--boot", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--req-timeout", type=float, default=240.0)
    s.add_argument("--wait-decode-s", type=float, default=60.0)
    s.add_argument("--out", required=True)
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_tel)
    s = sub.add_parser("pred")
    s.add_argument("--boot", required=True)
    s.add_argument("--variant-json", required=True)
    s.add_argument("--boot-facts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--status-dir", required=True)
    s.set_defaults(fn=cmd_pred)
    s = sub.add_parser("summary")
    s.add_argument("--rundir", required=True)
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
