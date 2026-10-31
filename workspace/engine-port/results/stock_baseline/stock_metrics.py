#!/usr/bin/env python3
"""Common per-request metric extraction for stock-serving baselines (PLAN_stock_baseline_metrics_2026-11-01 §0).

One definition for every source, so stock SGLang / vLLM / PD-mux rows can sit in one table:

  duration      sum of the rounds' durations (gate 7: never max())
  req/s         completed requests / duration
  out tok/s     output tokens of completed requests / duration
  total tok/s   (input + output) tokens of completed requests / duration
  E2E           per-request end - send (only when the source records it; otherwise
                reconstructed as TTFT + sum(ITL) and flagged e2e_source="ttft+itl")
  TTFT          per-request first token - send
  TPOT          per-request mean ITL (= (E2E - TTFT)/(out - 1) when ITL count = out - 1);
                +inf when a completed request carries no ITL (p1op_analyze convention)
  ITL p95(req)  request-internal token-ITL p95 (gate 4 goodput predicate)
  goodput       #(TTFT <= slo_ttft AND ITL p95(req) <= slo_itl) / duration, only when SLOs are given

Percentiles use the same linear interpolation as p1_opint/p1op_analyze.py `pctl`.
This module computes measurements only; it makes no performance judgement.
"""
import json
import math
from pathlib import Path


def pctl(vals, p):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c or s[f] == s[c]:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def request(ttft_s, itls_s, in_tokens, out_tokens, e2e_s=None, ok=True):
    """Normalised request record (all times in seconds)."""
    return {"ttft_s": ttft_s, "itls_s": list(itls_s or []), "in_tokens": in_tokens,
            "out_tokens": out_tokens, "e2e_s": e2e_s, "ok": bool(ok)}


def per_request(r):
    il = r["itls_s"]
    # no ITL recorded for a completed request -> +inf (same convention as p1op_analyze.cell_metrics:
    # such a request can never satisfy an ITL SLO, and percentiles stay conservative)
    tpot = (sum(il) / len(il)) if il else float("inf")
    itl95 = pctl(il, 0.95) if il else float("inf")
    if r["e2e_s"] is not None:
        e2e, src = r["e2e_s"], "measured"
    elif r["ttft_s"] is not None:
        e2e, src = r["ttft_s"] + sum(il), "ttft+itl"
    else:
        e2e, src = float("nan"), "none"
    return tpot, itl95, e2e, src


def metrics(rounds, slo_ttft_s=None, slo_itl_s=None):
    """rounds: list of (duration_s, [request, ...]). Rounds are pooled; duration is summed (gate 7)."""
    dur = sum(d for d, _ in rounds)
    reqs = [r for _, rs in rounds for r in rs]
    ok = [r for r in reqs if r["ok"] and r["ttft_s"] is not None]
    tp, i95, e2e, srcs = [], [], [], set()
    good = 0
    for r in ok:
        t, i, e, s = per_request(r)
        tp.append(t)
        i95.append(i)
        e2e.append(e)
        srcs.add(s)
        if slo_ttft_s is not None and slo_itl_s is not None:
            if r["ttft_s"] <= slo_ttft_s and i <= slo_itl_s:
                good += 1
    ttft = [r["ttft_s"] for r in ok]
    out_tok = sum(r["out_tokens"] or 0 for r in ok)
    in_tok = sum(r["in_tokens"] or 0 for r in ok)
    q = lambda xs, p: pctl(xs, p)
    m = {
        "n_rounds": len(rounds), "n_sent": len(reqs), "n_ok": len(ok), "duration_s": dur,
        "req_per_s": len(ok) / dur if dur else float("nan"),
        "out_tok_per_s": out_tok / dur if dur else float("nan"),
        "total_tok_per_s": (in_tok + out_tok) / dur if dur else float("nan"),
        "e2e_source": "+".join(sorted(srcs)) if srcs else "none",
    }
    for name, xs in (("e2e", e2e), ("ttft", ttft), ("tpot", tp)):
        for p in (50, 95, 99):
            m[f"{name}_p{p}_s"] = q(xs, p / 100)
    m["itl_req_p95_median_s"] = q(i95, 0.5)
    m["goodput_req_per_s"] = (good / dur if dur else float("nan")) if slo_ttft_s is not None else None
    return m


# ---------------------------------------------------------------- loaders

def from_bench_serving_row(row):
    """SGLang bench_serving --output-details row (P1-opint, Gate 2 bench jsonl). Times are seconds.
    No per-request send/finish timestamps: E2E is reconstructed (flagged)."""
    ttfts = row.get("ttfts") or []
    itls = row.get("itls") or []
    errs = row.get("errors") or []
    ins = row.get("input_lens") or []
    outs = row.get("output_lens") or []
    reqs = []
    for i in range(len(ttfts)):
        err = errs[i] if i < len(errs) else ""
        reqs.append(request(ttfts[i], itls[i] if i < len(itls) else [],
                            ins[i] if i < len(ins) else None, outs[i] if i < len(outs) else None,
                            e2e_s=None, ok=not err))
    return float(row.get("duration") or 0.0), reqs


def from_m3_cell(cell_dir):
    """M3 multigpu cell dir (cell.json + summary.json + requests.jsonl). Times in ms in the source."""
    cell_dir = Path(cell_dir)
    meta = json.loads((cell_dir / "cell.json").read_text())
    summ = json.loads((cell_dir / "summary.json").read_text())
    reqs = []
    for ln in (cell_dir / "requests.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        bad = bool(r.get("error")) or r.get("ttft_ms") is None
        ttft = None if r.get("ttft_ms") is None else r["ttft_ms"] / 1e3
        e2e = None
        if r.get("completion_s") is not None and r.get("sent_s") is not None:
            e2e = float(r["completion_s"]) - float(r["sent_s"])
        reqs.append(request(ttft, [x / 1e3 for x in (r.get("token_itl_ms") or [])],
                            r.get("input_tokens"), r.get("observed_output_tokens"), e2e_s=e2e, ok=not bad))
    return meta, float(summ["duration_s"]), reqs
