#!/usr/bin/env python3
"""C3 -- PD-mux prefill span regime census (result-analyst, 2026-10-28, CPU only, descriptive -- no verdict).

Question: per prefill batch, which regime does the PD-mux loop run in?
  SINGLE (미발화): forward_count >= num_hidden_layers -> the whole prefill is one span.
  PILEUP (적체):   >= 2 spans and some span's GPU time > decode step time (host issues spans faster than they run).
  PACED  (박자):   >= 2 spans and every span's GPU time <= decode step time.
Engine fact (src/multiplex/multiplexing_mixin.py @3858f129 :1416-1443, :1477-1480, :1517):
  forward_count = max(1, split_forward_token_budget // extend_num_tokens) (num_hidden_layers if extend 0),
  one chunk issued per loop iteration regardless of whether the previous chunk finished; only the decode
  stream is synchronized per iteration; wait_prefill_kernel_done is set only after the final chunk.
  extend_num_tokens is the BATCH total (sum over admitted requests), not one prompt.

Evidence levels used (label in output):
  measured  -- server log "Prefill batch, #new-token" (= batch extend tokens), PHASE_EVENTS span tables
               (lockstep_diag_values.json: n_chunk / n_wait iterations), client ITL, N2 PROFILE solo step,
               cost-term T2/T2' prefill times.
  derived   -- span time model: W(L, P) = T2'(32k, P) * T2(L) / T2(32k)  (T2 = P108 quadratic fit, T2' = measured
               single prefill with 1 decode co-running), split across chunks in proportion to the canonical
               per-layer FLOPs from scripts/costmodel/analytic_terms.py (implied efficiency eta is reported).
  lower bound (Zamba2, not in the canonical formula set): FLOPs >= 2 * (non-embedding params) * tokens at eta = 1,
               A100 bf16 dense peak 312 TFLOP/s scaled by P/108.
Assumptions are listed in ASSUMPTIONS and copied to the JSON.

Run:  python3 workspace/engine-port/results/impl_defects/c3_regimes.py \
          --out workspace/engine-port/results/impl_defects/c3_regimes.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EP = HERE.parents[1]                      # workspace/engine-port
RES = EP / "results"
REPO = EP.parents[1]
sys.path.insert(0, str(EP / "benchmarks"))
sys.path.insert(0, str(EP / "scripts" / "costmodel"))
from pdmux_eval.analyze import percentile  # noqa: E402  canonical percentile
import analytic_terms as AT                  # noqa: E402  canonical formulas (analytic-v1)

BUDGET = 65536          # every config used by the listed campaigns (grep; engine default pdmux_context.py:20 also 65536)
PEAK = 312e12           # A100 bf16 dense peak [cited spec]
RHO = (1.13, 1.16, 1.19)  # co-residence decode slowdown, cost-term T3 / N2-EA [measured range, RESULT_costterms §5]
BAND = 1.5              # regime band: r <= 1/BAND -> PACED, r >= BAND -> PILEUP, else BORDER [assumption]

ASSUMPTIONS = {
    "A1": "T2(L) shape at P=108 (quadratic fit J-A1 108u) transfers to P<108: W(L,P)=T2'(32k,P)*T2(L)/T2(32k)",
    "A2": "chunk GPU time proportional to its canonical FLOPs (analytic-v1 per-layer split); layer-kind efficiency equal",
    "A3": "T2'(32k,P)*P is constant (measured 407k-414k ms*SM for P 44-92) -> extrapolated to P=16",
    "A4": "decode step during a span = N2 PROFILE sticky-ON solo p50 at (D,b,c) x rho (1.13-1.19); D92 uses the "
          "108-SM (sticky OFF) solo value as a lower bound",
    "A5": "regime band x1.5 either side of r = span/step = 1 is BORDER (model error of A1-A4 not measured)",
    "A6": "diag decode batch by Little's law: 2k cells b~8 (0.5 req/s x ~13 s), 8k cells b~1 (0.1 req/s x ~12 s)",
    "A8": "Zamba2 eta transferred from Nano (0.45) for the non-firm estimate; Zamba2 layer FLOPs not in analytic-v1",
    "A9": "Granite params = log weight GiB / 2 B minus 0.9e9 embedding+head allowance (config not local)",
    "A7": "Zamba2 lower bound: params from safetensors index total_size/2, minus 2 x embedding (tied or not), eta=1",
}

# ------------------------------------------------------------------ measured inputs (cited files)
# RESULT_costterms_2026-10-26.md §3 (T2, P=108, J-A1 108u) and §4 (T2', r1 values)
T2_FIT = dict(u=11.98, v=0.1075, w=1.871e-7)                                   # ms, ms/tok, ms/tok^2
T2P_32K = {92: 4484.0, 84: 4933.0, 74: 5531.0, 64: 6363.0, 44: 9284.0}        # ms at L=32768, r1
# N2 PROFILE solo decode interval p50 (lockstep_diag_values.json, dataset N2PROFILE, class SOLO), read at runtime.


def T2(L):
    return T2_FIT["u"] + T2_FIT["v"] * L + T2_FIT["w"] * L * L


def t2p_32k(P):
    if P >= 108:
        return T2(32768)
    if P in T2P_32K:
        return T2P_32K[P]
    k = statistics.fmean(p * t for p, t in T2P_32K.items())                  # A3
    return k / P


def W_model(L, P):
    return t2p_32k(P) * T2(L) / T2(32768)                                     # A1


# ------------------------------------------------------------------ canonical per-layer FLOPs (Nemotron-H family)
def layer_flops(a: AT.Arch, kind: str, lens):
    T = sum(lens)
    if kind == "M":
        return T * (AT.flops_mamba_proj_per_token(a) + AT.flops_mamba_conv_per_token(a) + AT.flops_ssd_prefill_per_token(a))
    if kind == "-":
        return T * AT.flops_mlp_per_token(a)
    if kind == "*":
        return T * AT.flops_attn_proj_per_token(a) + sum(AT.flops_attn_core_prefill(a, L) for L in lens)
    raise ValueError(kind)


def chunk_times(a: AT.Arch, pattern: str, lens, P, W_override=None):
    """Return list of per-chunk ms, forward_count, n_chunks, implied eta."""
    nl = len(pattern)
    ext = sum(lens)
    fc = max(1, BUDGET_CUR // ext) if ext > 0 else nl
    per = [layer_flops(a, ch, lens) for ch in pattern]
    per[-1] += len(lens) * AT.flops_lm_head_per_token(a)
    F = sum(per)
    W = W_override if W_override is not None else W_model(ext if len(lens) == 1 else ext, P)
    chunks, i = [], 0
    while i < nl:
        j = min(nl, i + fc)
        chunks.append(W * sum(per[i:j]) / F)
        i = j
    eta = F / (W / 1e3 * PEAK * min(P, 108) / 108)
    return chunks, fc, len(chunks), eta


BUDGET_CUR = BUDGET


def regime_from_ratio(r):
    if r <= 1 / BAND:
        return "PACED"
    if r >= BAND:
        return "PILEUP"
    return "BORDER"


# ------------------------------------------------------------------ helpers
PB = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+)")


def prefill_batches(log):
    out = []
    with open(log, errors="replace") as fh:
        for line in fh:
            m = PB.search(line)
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
    return out


def n_spans(ext, nl, budget=BUDGET):
    fc = max(1, budget // ext) if ext > 0 else nl
    return -(-nl // fc)


def qs(v):
    return {"n": len(v), "p50": percentile(v, .5), "p90": percentile(v, .9), "p99": percentile(v, .99),
            "max": max(v) if v else None} if v else {"n": 0}


def client_itl_p50(paths):
    vals = []
    for p in paths:
        d = None
        for line in open(p):
            d = json.loads(line)
            vals.extend(x * 1e3 for req in d.get("itls", []) for x in req)
    return percentile(vals, .5) if vals else None


# ------------------------------------------------------------------ A. HE0 (KISTI, Zamba2-2.7B)
def zamba2_nonembed_params():
    snap = glob.glob(str(REPO / "hf_cache/hub/models--Zyphra--Zamba2-2.7B/snapshots/*/config.json"))[0].replace("config.json", "")
    cfg = json.load(open(snap + "config.json"))
    tot = json.load(open(snap + "model.safetensors.index.json"))["metadata"]["total_size"] / 2
    emb = cfg["vocab_size"] * cfg["hidden_size"]
    return tot - 2 * emb, cfg["num_hidden_layers"], tot


def lb_span_ms(params_ne, ext, n_sp, P):
    """Lower bound on the longest chunk: total/n_spans at eta=1."""
    return 2 * params_ne * ext / (PEAK * P / 108) * 1e3 / n_sp


ETA_T = 0.45   # eta transferred from Nano T2/T2' (implied 0.43-0.46, this script) [assumption A8]


def classify_lb(exts, NL, params_ne, P, step):
    """Multi-span batches: firm PILEUP if the eta=1 lower bound of the average chunk exceeds the step; otherwise the
    eta-transferred estimate (average chunk / ETA_T) is banded.  Average chunk <= longest chunk, so PACED_est is
    biased toward PACED (optimistic) and PILEUP_firm is conservative."""
    c = Counter()
    for e in exts:
        n = n_spans(e, NL)
        if n == 1:
            c["SINGLE"] += 1
            continue
        lb = lb_span_ms(params_ne, e, n, P)
        if lb > step:
            c["PILEUP_firm"] += 1
        else:
            c[regime_from_ratio((lb / ETA_T) / step) + "_est"] += 1
    tot = sum(c.values())
    return {k: v / tot for k, v in sorted(c.items())}


def he0():
    sys.path.insert(0, str(RES / "he0_contamination_2026-10-01"))
    import he0_realized as H  # noqa: E402  (HE0 membership, CONSENSUS 1-7 / 1-13)
    params_ne, NL, tot = zamba2_nonembed_params()
    src = RES / "slo_sched"
    out = {"model": "Zyphra/Zamba2-2.7B", "num_hidden_layers": NL, "budget": BUDGET,
           "budget_source": "results/slo_sched/pdmux_d{16,24,34,44,slo}.yml (9/9 = 65536)",
           "params_total": tot, "params_nonembed_lb": params_ne, "arms": {}}
    all_ext = []
    for arm, jobs in H.HE0_JOBS.items():
        ext, nsp, logs, lo, hi = [], [], 0, [], []
        for j in jobs:
            g = glob.glob(str(src / f"sgptvsrv_*_{j}.log"))
            if not g:
                continue
            logs += 1
            for n_new, e in prefill_batches(g[0]):
                ext.append(e)
                nsp.append(n_spans(e, NL))
            lo += glob.glob(str(src / f"sgptvLo_*_{j}.jsonl"))
            hi += glob.glob(str(src / f"sgptvHi_*_{j}.jsonl"))
        all_ext += ext
        D = H.STATIC_D.get(arm)
        itl_lo, itl_hi = client_itl_p50(lo), client_itl_p50(hi)
        step = max(x for x in (itl_lo, itl_hi) if x is not None)
        P = 108 - D if D else 92                       # dynamic arms: most prefill-heavy grid point (smallest LB)
        multi = [(e, s) for e, s in zip(ext, nsp) if s >= 2]
        pile_lb = sum(1 for e, s in multi if lb_span_ms(params_ne, e, s, P) > step)
        c = Counter(nsp)
        out["arms"][arm] = dict(
            runs=logs, batches=len(ext), ext=qs(ext),
            share_single=c[1] / len(ext), share_multi=1 - c[1] / len(ext),
            spans_hist={str(k): v for k, v in sorted(c.items())},
            itl_p50_lo_ms=itl_lo, itl_p50_hi_ms=itl_hi, step_used_ms=step, P_for_LB=P,
            multi_pileup_by_lower_bound=pile_lb, multi_unresolved=len(multi) - pile_lb,
            share_pileup_lb=pile_lb / len(ext), share_unresolved=(len(multi) - pile_lb) / len(ext),
            lb_span_ms_at_threshold=lb_span_ms(params_ne, 65536 // NL + 1, 2, P),
            regimes_vs_LO_step=classify_lb(ext, NL, params_ne, P, itl_lo),
            regimes_vs_HI_step=classify_lb(ext, NL, params_ne, P, itl_hi))
    out["pooled_ext"] = qs(all_ext)
    out["threshold_ext_single"] = BUDGET // NL
    return out


# ------------------------------------------------------------------ B. P1-opint (KISTI, Zamba2 / Granite, cudagraph ON)
def p1opint():
    out = {"budget_source": "results/p1_opint/pdmux_a100_smoke.yml (65536)", "models": {}}
    params_ne_z, _, _ = zamba2_nonembed_params()
    for tag, NL, model in (("zamba2-27b", 54, "Zamba2-2.7B"), ("granite-40-h-micro-base", 40, "Granite-4.0-h-micro")):
        ext, nsp = [], []
        for g in sorted(glob.glob(str(RES / "p1_opint" / f"p1op_{tag}_srv_agnostic_rep*_*.log"))):
            for _, e in prefill_batches(g):
                if e < 100:      # warm-up / health probes (curl generate, 1-7 tokens)
                    continue
                ext.append(e)
                nsp.append(n_spans(e, NL))
        c = Counter(nsp)
        itl = {}
        for g in sorted(glob.glob(str(RES / "p1_opint" / f"p1op_{tag}_agnostic_rep*_*.jsonl"))):
            for line in open(g):
                d = json.loads(line)
                itl.setdefault(d["request_rate"], []).extend(x * 1e3 for r in d["itls"] for x in r)
        itl50 = {str(k): percentile(v, .5) for k, v in sorted(itl.items())}
        rec = dict(model=model, num_hidden_layers=NL, batches=len(ext), ext=qs(ext),
                   share_single=c[1] / len(ext) if ext else None, spans_hist={str(k): v for k, v in sorted(c.items())},
                   client_itl_p50_by_rate_ms=itl50)
        if tag.startswith("granite"):
            # params from the server log "Load weight end ... mem usage=X GB" (GiB, cross-checked on Zamba2:
            # 5.08 GiB vs safetensors 5.378e9 B) minus a generous embedding+head allowance 0.9e9 [assumption A9]
            g0 = sorted(glob.glob(str(RES / "p1_opint" / f"p1op_{tag}_srv_agnostic_rep1_*.log")))[0]
            gib = float(re.search(r"Load weight end.*mem usage=([0-9.]+) GB", open(g0, errors="replace").read()).group(1))
            pne = gib * 2**30 / 2 - 0.9e9
            rec["params_from_log"] = gib * 2**30 / 2
            rec["params_nonembed_lb"] = pne
            lo_s, hi_s = min(itl50.values()), max(itl50.values())
            rec["regimes"] = {f"P{P}_step{st:.1f}": classify_lb(ext, NL, pne, P, st)
                              for P in (54, 108) for st in (lo_s, hi_s)}
            rec["lb_span_ms_ext2000_P108"] = lb_span_ms(pne, 2000, n_spans(2000, NL), 108)
            rec["lb_span_ms_ext2000_P64"] = lb_span_ms(pne, 2000, n_spans(2000, NL), 64)
        if tag.startswith("zamba"):
            lo_s, hi_s = min(itl50.values()), max(itl50.values())
            rec["regimes"] = {f"P{P}_step{st:.1f}": classify_lb(ext, NL, params_ne_z, P, st)
                              for P in (54, 108) for st in (lo_s, hi_s)}
            # lower bound at P=108 (auto grid: realized P per batch not logged) for the modal 1-request batch
            rec["lb_span_ms_ext2000_P108"] = lb_span_ms(params_ne_z, 2000, n_spans(2000, NL), 108)
            rec["lb_span_ms_ext2000_P64"] = lb_span_ms(params_ne_z, 2000, n_spans(2000, NL), 64)
        out["models"][tag] = rec
    return out


# ------------------------------------------------------------------ C/D. N3, N2-EA measured span tables
def measured_spans():
    d = json.load(open(RES / "lockstep_diag" / "lockstep_diag_values.json"))
    out = {}
    for ds in ("N3", "N2EA"):
        c = Counter()
        ratio, nwait = [], []
        for cell in d["results"]:
            if cell["dataset"] != ds:
                continue
            step = (cell["by_class"].get("CHUNK") or {}).get("p50")
            for s in cell["spans"]:
                if s["n_chunks_expected"] == 1:
                    c["SINGLE"] += 1
                elif s["n_chunk_raw"] == 0 and s["n_wait_it"] == 0:
                    c["NO_DECODE"] += 1          # decode batch empty: no pacing reference (span runs at P=108)
                elif s["n_wait_it"] >= 2:
                    c["PILEUP"] += 1
                    nwait.append(s["n_wait_it"])
                else:
                    c["PACED_or_LAG"] += 1
                if s["n_chunk_raw"] and step and s["n_chunks_expected"]:
                    ratio.append((s["W_ms"] / s["n_chunks_expected"]) / step)
        tot = sum(c.values())
        out[ds] = dict(spans=tot, counts=dict(c), shares={k: v / tot for k, v in c.items()},
                       wait_iters_per_pileup_span=qs(nwait), W_over_nchunks_over_CHUNKp50=qs(ratio))
    t3 = defaultdict(list)
    for x in d["costterms_t3"]:
        var, D = x["key"].split("|")[2], x["key"].split("|")[3]
        t3[(var, D, x["cell"])].append(x["co_p50"])
    rows = {}
    for (var, D, cell), v in sorted(t3.items()):
        b, L = map(int, cell.split("x"))
        P = 108 - int(D[1:])
        nch = n_spans(L, 56)
        span = W_model(L, P) / nch
        rows[f"{var}|{D}|{cell}"] = dict(L=L, P=P, n_chunks=nch, span_ms_model=span, co_p50_ms=statistics.fmean(v),
                                         r=span / statistics.fmean(v), regime=regime_from_ratio(span / statistics.fmean(v)))
    out["T3"] = rows
    return out


# ------------------------------------------------------------------ E. Nano model cells (diag PREREG grid, policy PREREG)
def n2_solo_steps():
    d = json.load(open(RES / "lockstep_diag" / "lockstep_diag_values.json"))
    agg = defaultdict(list)
    for x in d["results"]:
        if x["dataset"] != "N2PROFILE":
            continue
        m = re.match(r"r\d+_(ON|OFF)(\d+)-b(\d+)-c(\d+)", x["cell"])
        s = x["by_class"].get("SOLO")
        if m and s:
            agg[(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))].append(s["p50"])
    return {k: statistics.median(v) for k, v in agg.items()}


def solo_step(steps, D, b, c):
    if D in (16, 24, 34, 44):
        return steps[("ON", D, b, c)], "N2 PROFILE ON"
    return steps[("OFF", 44, b, c)], "N2 PROFILE OFF (108 SM) lower bound"   # D92 / D108


def nano_cell(a, pattern, L, budget, D, b, c, steps):
    global BUDGET_CUR
    BUDGET_CUR = budget
    P = 108 - D
    ch, fc, n, eta = chunk_times(a, pattern, [L], P)
    BUDGET_CUR = BUDGET
    st, src = solo_step(steps, D, b, c)
    rmax = {k: max(ch) / (st * rho) for k, rho in zip(("rho_lo", "rho_mid", "rho_hi"), RHO)}
    regime = "SINGLE" if n == 1 else regime_from_ratio(rmax["rho_mid"])
    share_over = sum(1 for t in ch if t > st * RHO[1]) / n
    return dict(L=L, budget=budget, D=D, P=P, forward_count=fc, n_spans=n, W_ms=W_model(L, P), eta_implied=eta,
                span_ms_max=max(ch), span_ms_med=statistics.median(ch), span_ms_min=min(ch),
                decode_solo_ms=st, decode_src=src, decode_b=b, decode_c=c,
                r_max=rmax, share_spans_over_step=share_over, regime=regime,
                regime_rho_range=sorted({("SINGLE" if n == 1 else regime_from_ratio(v)) for v in rmax.values()}))


def nano_section():
    a = AT.nano9b_v2()
    cfg = json.load(open(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/*/config.json"))[0]))
    pat = cfg["hybrid_override_pattern"]
    assert len(pat) == cfg["num_hidden_layers"] == 56, (len(pat), cfg["num_hidden_layers"])
    steps = n2_solo_steps()
    diag = []
    for L, b, c in ((2048, 8, 2048), (8192, 1, 8192)):
        for budget in (8192, 65536):
            for D in (24, 44, 92):
                diag.append(nano_cell(a, pat, L, budget, D, b, c, steps))
    extra = nano_cell(a, pat, 2048, 512, 44, 8, 2048, steps)
    # validation of the span model against measured N3 W (8k, P64 co-resident 1,463-1,816 ms, median ~1.49 s)
    val = dict(W_model_8k_P64=W_model(8192, 64), W_meas_8k_P64="1463-1816 ms (median ~1490) [N3, DIAG §3-3]",
               W_model_8k_P108=W_model(8192, 108), W_meas_8k_P108="870-891 ms [N3 floor, DIAG §3-3]")
    # policy PREREG W3 (arXiv-Sum p50 6,484, filter <= 16,384) and N3 / N2-EA / T3 cross-check at D44 / D16
    pol = []
    for L in (1171, 2000, 6484, 16384):
        pol.append(nano_cell(a, pat, L, 65536, 16, 8, 2048, steps))
        pol.append(nano_cell(a, pat, L, 65536, 44, 8, 2048, steps))
        pol.append(_d64(a, pat, L, steps))
    return dict(pattern=pat, num_hidden_layers=56, threshold_ext_single=BUDGET // 56,
                diag_grid=diag, extra_2k_b512_D44=extra, validation=val, policy=pol)


def _d64(a, pat, L, steps):
    # D64 decode step not in N2 PROFILE (D16/24/34/44 only): use D44 ON as an upper... it is faster with more
    # decode SMs, so D44 ON is an UPPER bound on the D64 step -> PILEUP calls are conservative, PACED calls are not.
    r = nano_cell(a, pat, L, 65536, 44, 8, 2048, steps)
    global BUDGET_CUR
    ch, fc, n, eta = chunk_times(a, pat, [L], 44)
    r.update(D=64, P=44, W_ms=W_model(L, 44), span_ms_max=max(ch), span_ms_med=statistics.median(ch),
             decode_src="D44 ON (upper bound for D64)")
    rm = max(ch) / (r["decode_solo_ms"] * RHO[1])
    r["r_max"] = {"rho_mid": rm}
    r["regime"] = "SINGLE" if n == 1 else regime_from_ratio(rm)
    return r


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = dict(version="c3_regimes 2026-10-28", analytic=AT.VERSION, budget=BUDGET, peak=PEAK, rho=RHO, band=BAND,
               assumptions=ASSUMPTIONS, he0=he0(), p1opint=p1opint(), measured=measured_spans(), nano=nano_section())
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    # ---- console summary
    h = res["he0"]
    print(f"[HE0] Zamba2 L={h['num_hidden_layers']} single-span if ext<={h['threshold_ext_single']}  pooled ext {h['pooled_ext']}")
    for arm, r in h["arms"].items():
        print(f"  {arm:12s} runs={r['runs']} batches={r['batches']} ext p50/p90/p99/max={r['ext']['p50']:.0f}/"
              f"{r['ext']['p90']:.0f}/{r['ext']['p99']:.0f}/{r['ext']['max']}  single={r['share_single']:.3f} "
              f"pile(LB)={r['share_pileup_lb']:.3f} unres={r['share_unresolved']:.3f} spans={r['spans_hist']} "
              f"ITL p50 LO/HI={r['itl_p50_lo_ms']:.1f}/{r['itl_p50_hi_ms']:.1f} P={r['P_for_LB']}")
        print(f"      vsLO {r['regimes_vs_LO_step']}\n      vsHI {r['regimes_vs_HI_step']}")
    for tag, r in res["p1opint"]["models"].items():
        print(f"[P1-opint] {tag} L={r['num_hidden_layers']} batches={r['batches']} ext={r['ext']} single={r['share_single']}"
              f" spans={r['spans_hist']} ITL50={ {k: round(v, 1) for k, v in r['client_itl_p50_by_rate_ms'].items()} }"
              + (f" regimes={r['regimes']}" if 'regimes' in r else "")
              + (f" LB span(2000,P108/P64)={r['lb_span_ms_ext2000_P108']:.1f}/{r['lb_span_ms_ext2000_P64']:.1f}" if 'lb_span_ms_ext2000_P108' in r else ""))
    for ds in ("N3", "N2EA"):
        m = res["measured"][ds]
        print(f"[{ds}] spans={m['spans']} counts={m['counts']} wait_it/pileup={m['wait_iters_per_pileup_span']}"
              f" (W/n)/CHUNKp50={m['W_over_nchunks_over_CHUNKp50']}")
    for k, r in res["measured"]["T3"].items():
        print(f"[T3] {k:28s} n_ch={r['n_chunks']:2d} span~{r['span_ms_model']:.0f}ms co_p50={r['co_p50_ms']:.1f} r={r['r']:.1f} {r['regime']}")
    n = res["nano"]
    print(f"[Nano] validation {n['validation']}")
    for c in n["diag_grid"] + [n["extra_2k_b512_D44"]]:
        print(f"  diag L={c['L']:5d} B={c['budget']:5d} D{c['D']:2d} P={c['P']:3d} fc={c['forward_count']:2d} spans={c['n_spans']:2d} "
              f"W={c['W_ms']:.0f} span max/med={c['span_ms_max']:.1f}/{c['span_ms_med']:.1f} step_solo={c['decode_solo_ms']:.1f} "
              f"r_max(mid)={c['r_max']['rho_mid']:.2f} over={c['share_spans_over_step']:.2f} eta={c['eta_implied']:.2f} "
              f"-> {c['regime']} {c['regime_rho_range']}")
    for c in n["policy"]:
        print(f"  policy L={c['L']:5d} D{c['D']:2d} spans={c['n_spans']:2d} span max={c['span_ms_max']:.1f} "
              f"step={c['decode_solo_ms']:.1f} r={c['r_max']['rho_mid']:.2f} -> {c['regime']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
