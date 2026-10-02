#!/usr/bin/env python3
"""HE0 co-residency: independent estimator from slow decode iterations (GPU 0, 2026-10-02, result-analyst).

Question (AUDIT_RESULT_VERDICT_2026-10-02.md section 1 #7, section 4 item 1): the HI time-weighted
co-residency of ``he0_realized.py`` is 31 / 39 / 57 % under the +1 iter / pt / earliest conventions (and
~0-1 % under the log-only latest-start convention).  These are convention points, not bounds, and the
lower ones depend entirely on the running-count matching model.  Can an estimator that does NOT use the
running-count matching narrow this?

Estimator (decode-side signature): while a split prefill is in flight the decode stream runs on fewer SMs
(and, as seen in the real token timelines, the iteration at admission is a ~59-75 ms stall), so decode
iterations are slow.  All running requests receive their token of an iteration within a few ms of each
other (checked on real data), so decode iterations are recoverable as clusters of token receipts, and
"prefill in flight" ~= union of slow iterations.  Uses: client token receipt times (``tok_times`` of the
reconstruction = fitted send time + TTFT + cumulative ITL) only.  Does NOT use: #running-req, s_pt, the
start brackets, or T_log.  Shared dependence (stated): the per-request send-offset fit (which sets the
relative placement of requests; its accuracy shows up as cluster spread, reported) and the round window.

=========================================================================================
PRE-REGISTERED (fixed before any real-data run of this file; do not edit after results exist)
=========================================================================================
K_GRID           = (1.25, 1.5, 2.0, 3.0)    slow iff iteration duration > k * baseline
BASELINE         = rolling 20th percentile of iteration durations within +-1.5 s (non-artifact clusters);
                   fallback round-level 20th percentile if < 10 clusters in the window
CLUSTER_GAP_S    = 0.003   consecutive decode-token receipts closer than this belong to one iteration
iteration interval = [median over members of previous receipt, median over members of this receipt]
CLIENT/ARTIFACT separation (applied identically to synthetic and real data):
  A1 gap      : iteration duration > 0.5 s  (no prefill-induced slow iteration is near this: real HI slow
                iterations are 30-90 ms)  -> artifact
  A2 bunching : a request appears >= 2 times in one cluster (receipt bunching = client-side buffering,
                e.g. the 856892 HI2 3.9 s client gap followed by a flush) -> artifact
  excluded time X = union of [artifact interval start - 0.5 s, end + 0.5 s]; X is removed from BOTH the
  numerator and the denominator of every estimator and every convention (common support).
Estimand compared: time fraction of the round window (minus X) that is
  est_k : covered by slow iterations
  conv_x: covered by CO_x of the reconstruction, x in {lo_plus1iter, pt, earliest, latest}
Per-batch start test (secondary): for each batch k (e_k in window, not in X) take the slow run
  (slow intervals merged across gaps <= 2 ms) that contains e_k or ends within 25 ms of it;
  s_hat_k = max(run start, e_{k-1}).  For each convention err_x,k = s_x,k - s_hat_k.  Report median err,
  median |err|, fraction |err| <= 25 ms, and the fraction of batches with no slow signature.
Synthetic validation (truth known; engine from he0_selftest_v2.synth_engine, delay=real):
  regimes phase x D in {LO,HI} x {44,16}; scenarios per regime: no stall / client SEND stall 3 s@100 /
  client RECEIVE stall 3.9 s at mid-window (bunching).  Reported-only stress: no admission-stall iteration.
  k qualifies for a regime iff |est_k - truth| <= 0.03 in every (non-stress) scenario of that regime AND
  the receive-stall window is >= 90 % covered by X.
  k* = smallest k in K_GRID qualifying for BOTH HI regimes.  If none: verdict INCONCLUSIVE (estimator
  not validated), real-data numbers reported as descriptive only.
Real-data decision (HI phase, per arm mean over runs, all rounds; sensitivity = drop he0 quality-fail rounds):
  band = est_k* +- max(0.03, synthetic max |est_k* - truth| over HI scenarios)
  convention x "consistent" in an arm iff conv_x in band.
  REAL (range narrowed)  : in >= 6/7 arms earliest AND latest are outside the band while >= 1 of
                           {lo_plus1iter, pt} is inside;
  NOT-REAL               : in >= 6/7 arms no convention is inside the band (conventions do not bracket);
  INCONCLUSIVE           : otherwise.
=========================================================================================

AMENDMENT 1 (2026-10-02, after the synthetic validation of the estimator above, BEFORE any real-data run
of any estimator in this file).  Disclosure of prior real-data looks: before writing this file the analyst
inspected the token timelines of 4 real rounds (856890 / 859008, LO0 and HI0) to calibrate the synthetic
engine shape (admission-stall iteration, slowed iterations); after the v1 failure, the cluster spread and
bunching COUNTS (not any co-residency estimate) of 24 real rounds (856890, 859008, 857051, 856892) were
inspected to diagnose the failure.
  * v1 (cluster estimator, above) FAILED synthetic validation: no k qualifies (HI max |err| 0.15-0.33).
    Cause: the reconstruction aligns requests relative to each other only to tens of ms (per-request
    send-drift fit error x request index); clusters smear (synthetic HI spread p95 17-125 ms), A2
    "bunching" fires on 72-139 clusters per HI round and X removes 30-80 % of the window.  Real HI rounds
    show the same smear (spread p95 8-100 ms, 0-185 bunched clusters per round) -> v1 is not usable on
    real data either.  v1 is kept and its synthetic numbers are reported as a failed estimator.
  * v2 (per-request, alignment-free numerator) is the estimator used for the real-data decision:
      tokens          : every decode token (j >= 1) of every request; its interval [t_{j-1}, t_j] and ITL d
                        come from that request's own client ITL sequence (no cross-request alignment).
      baseline        : 0.25 s bins; baseline for a token = 20th percentile of ITLs ending within its bin
                        +-6 bins (+-1.5 s) after removing ITL > 0.5 s; fallback round 20th pct if < 10.
      slow            : d > k * baseline, k in K_GRID (unchanged).
      artifacts       : A1 d > 0.5 s; A2 (per request) d_j > 1.25*base AND d_{j+1}, d_{j+2} < 0.3*base
                        (receipt bunching); X = union(artifact interval +- 0.5 s), clipped to the window.
      ESTIMAND (changed): request-time-weighted co share  RTW = sum over kept tokens of time inside
                        co-residency / sum of kept ITL time  (a token is kept iff its interval misses X).
                        est_k = sum d*[slow] / sum d ;  conv_x = sum overlap(interval, P_x) / sum d.
                        Window-time fractions (the 31/39/57 % numbers) are reported alongside for mapping,
                        but the decision uses RTW because only RTW is alignment-free on the estimator side.
      synthetic truth : sum over the same kept tokens of overlap(SERVER interval [g_{j-1}, g_j], P_true)
                        / sum of server ITL time.
      per-batch start test: DROPPED (it needs cross-request alignment at the 10 ms scale, which the
                        reconstruction does not have).
    Qualification, k*, band and the REAL / NOT-REAL / INCONCLUSIVE rules: unchanged, applied to RTW.

EXPLORATORY (post-hoc, added after v2 also failed qualification; NOT used for any decision):
  "split-exposure" truth = share of decode ITL time produced by iterations LAUNCHED while a prefill was in
  flight (the split is chosen at iteration launch, so such an iteration runs on the reduced partition even
  after the prefill ends).  This differs from the he0 estimand (in-flight ∩ decode-active wall time) when
  a slowed iteration outlasts the prefill.  Reported to show what the slow-token estimator actually tracks.

Limits: reconstruction-based (client receipt frame); the reconstruction's alignment to the server log is
validated only to ~0.3-1 s, but this estimator and the CO_x it is compared with live in the same client
frame, so their comparison does not depend on that alignment except through e_k/s_x placement.  The
estimator measures decode-SLOWED time; it misses the part of an in-flight interval before the first
iteration boundary after admission (<= 1 iteration) and any part where the split does not slow decode
by k (e.g. LO d44, factor ~1.05 at small batch).  A slow iteration with no prefill in flight (other
server-side hiccup) is counted as co-residency.  100 ms-scale statements about the real engine remain
model-dependent.

Usage: python3 he0_coresidency_indep.py [--output he0_coresidency_indep_2026-10-02.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import he0_realized as H  # noqa: E402
import he0_followup as F  # noqa: E402
import he0_selftest_v2 as ST  # noqa: E402
from he0_realized import clip, intersect, length, minus, union  # noqa: E402

K_GRID = (1.25, 1.5, 2.0, 3.0)
BASE_Q = 0.20
BASE_HALF_S = 1.5
BASE_MIN_N = 10
CLUSTER_GAP_S = 0.003
GAP_ART_S = 0.5
GUARD_S = 0.5
RUN_MERGE_S = 0.002
EDGE_TOL_S = 0.025
QUAL_TOL = 0.03
RECV_COVER_MIN = 0.90
QUALITY_FAIL = {(856892, "HI", 0), (856892, "HI", 2), (859006, "HI", 2)}


def q(xs, p):
    return float(np.quantile(np.asarray(xs), p, method="linear"))


def iterations(tok_times: List[List[float]]):
    """Cluster decode-token receipts (j >= 1) of all requests into iterations."""
    pts = sorted((tt[j], i, tt[j - 1]) for i, tt in enumerate(tok_times) for j in range(1, len(tt)))
    cl: List[List[Tuple[float, int, float]]] = []
    for p in pts:
        if cl and p[0] - cl[-1][-1][0] < CLUSTER_GAP_S:
            cl[-1].append(p)
        else:
            cl.append([p])
    its = []
    for c in cl:
        ids = [x[1] for x in c]
        a = statistics.median(x[2] for x in c)
        b = statistics.median(x[0] for x in c)
        its.append({"a": a, "b": b, "d": b - a, "n": len(c), "spread": c[-1][0] - c[0][0],
                    "bunched": len(ids) != len(set(ids))})
    return its


def estimate(rec, extra=None):
    """Return dict with est_k, conventions, per-batch start test on common support."""
    win = rec["window"]
    its = iterations(rec["tok_times"])
    art = [it for it in its if it["d"] > GAP_ART_S or it["bunched"]]
    X = clip(union([(it["a"] - GUARD_S, it["b"] + GUARD_S) for it in art]), *win)
    good = [it for it in its if not (it["d"] > GAP_ART_S or it["bunched"])]
    gb = np.array([it["b"] for it in good]); gd = np.array([it["d"] for it in good])
    order = np.argsort(gb); gb = gb[order]; gd = gd[order]
    round_base = q(gd, BASE_Q) if len(gd) else float("nan")
    for it in good:
        lo_i = np.searchsorted(gb, it["b"] - BASE_HALF_S); hi_i = np.searchsorted(gb, it["b"] + BASE_HALF_S, side="right")
        w = gd[lo_i:hi_i]
        it["base"] = q(w, BASE_Q) if len(w) >= BASE_MIN_N else round_base
    W = length(minus([win], X))
    sup = minus([win], X)
    out = {"W_support_s": W, "W_s": win[1] - win[0], "excluded_s": length(X), "n_artifact_iter": len(art),
           "n_iter": len(its), "cluster_spread_ms_p95": 1000 * q([it["spread"] for it in its if it["n"] > 1], 0.95) if its else None,
           "baseline_ms_p50": 1000 * statistics.median(it["base"] for it in good) if good else None,
           "est": {}, "conv": {}, "start_test": {}}
    if W <= 0:
        return out, X, {}
    slow_sets = {}
    for k in K_GRID:
        S = union([(it["a"], it["b"]) for it in good if it["d"] > k * it["base"]])
        slow_sets[k] = S
        out["est"][str(k)] = length(intersect(S, sup)) / W
    D = rec["D"]
    convP = {"lo_plus1iter": rec["P_lo"], "pt": rec["P_pt"], "earliest": rec["P_hi"],
             "latest": clip(union(list(zip(rec["s_hi"], rec["e"]))), *win)}
    for x, P in convP.items():
        out["conv"][x] = length(intersect(intersect(P, D), sup)) / W
    # per-batch start test
    sx = {"lo_plus1iter": None, "pt": rec["s_pt"], "earliest": rec["s_lo"], "latest": rec["s_hi"]}
    all_itl = []
    for tt in rec["tok_times"]:
        all_itl += [tt[j] - tt[j - 1] for j in range(1, len(tt))]
    step = statistics.median(all_itl) if all_itl else 0.0
    sx["lo_plus1iter"] = [min(rec["e"][k], rec["s_pt"][k] + step) for k in range(len(rec["e"]))]
    for k_ in K_GRID:
        runs = []
        for a, b in slow_sets[k_]:
            if runs and a - runs[-1][1] <= RUN_MERGE_S:
                runs[-1][1] = max(runs[-1][1], b)
            else:
                runs.append([a, b])
        starts = [r[0] for r in runs]
        errs = {x: [] for x in sx}
        nosig = tot = 0
        for kb, e in enumerate(rec["e"]):
            if not (win[0] <= e < win[1]) or any(a <= e <= b for a, b in X):
                continue
            tot += 1
            j = int(np.searchsorted(starts, e + EDGE_TOL_S, side="right")) - 1
            run = None
            while j >= 0:
                r = runs[j]
                if r[0] <= e + EDGE_TOL_S and r[1] >= e - EDGE_TOL_S:
                    run = r; break
                if r[1] < e - EDGE_TOL_S:
                    break
                j -= 1
            if run is None:
                nosig += 1; continue
            sh = max(run[0], rec["e"][kb - 1] if kb else -math.inf)
            for x in sx:
                errs[x].append(sx[x][kb] - sh)
        st = {"batches": tot, "no_signature_frac": nosig / tot if tot else None}
        for x, v in errs.items():
            if v:
                st[x] = {"median_err_ms": 1000 * statistics.median(v), "median_abs_err_ms": 1000 * statistics.median(abs(z) for z in v),
                         "frac_abs_le_25ms": sum(1 for z in v if abs(z) <= EDGE_TOL_S) / len(v)}
        out["start_test"][str(k_)] = st
    return out, X, slow_sets


# ----------------------------------------------------------------------------------- v2 per-request estimator
BIN_S = 0.25
BIN_HALF = 6
BUNCH_LO = 0.3
BUNCH_HI = 1.25


def tokens_of(tok_times):
    """(end, start, d, req, j) for every decode token j >= 1."""
    return [(tt[j], tt[j - 1], tt[j] - tt[j - 1], i, j) for i, tt in enumerate(tok_times) for j in range(1, len(tt))]


def baselines(toks, w0):
    ends = np.array([t[0] for t in toks]); d = np.array([t[2] for t in toks])
    okm = d <= GAP_ART_S
    b = np.floor((ends - w0) / BIN_S).astype(int)
    rb = q(d[okm], BASE_Q) if okm.any() else float("nan")
    nb = b.max() + 1 if len(b) else 0
    by = [[] for _ in range(max(nb, 0) + 1)]
    for bi, di, o in zip(b, d, okm):
        if o and 0 <= bi < len(by):
            by[bi].append(di)
    cache = {}
    out = np.empty(len(toks))
    for k, bi in enumerate(b):
        if bi not in cache:
            vals = [x for bb in range(max(0, bi - BIN_HALF), min(len(by), bi + BIN_HALF + 1)) for x in by[bb]] if 0 <= bi < len(by) else []
            cache[bi] = q(vals, BASE_Q) if len(vals) >= BASE_MIN_N else rb
        out[k] = cache[bi]
    return out


def estimate_v2(rec, truth_ctx=None):
    """Per-request, alignment-free numerator.  truth_ctx (synthetic only): (gen, P_true)."""
    win = rec["window"]
    tok_times = rec["tok_times"]
    toks = tokens_of(tok_times)
    base = baselines(toks, win[0])
    idx = {(t[3], t[4]): k for k, t in enumerate(toks)}
    art = []
    for k, (e, s, d, i, j) in enumerate(toks):
        if d > GAP_ART_S:
            art.append((s, e)); continue
        k1, k2 = idx.get((i, j + 1)), idx.get((i, j + 2))
        if d > BUNCH_HI * base[k] and k1 is not None and k2 is not None and \
                toks[k1][2] < BUNCH_LO * base[k1] and toks[k2][2] < BUNCH_LO * base[k2]:
            art.append((s, toks[k2][0]))
    X = clip(union([(a - GUARD_S, b + GUARD_S) for a, b in art]), *win)
    Xs = [x[0] for x in X]
    keep = []
    for k, (e, s, d, i, j) in enumerate(toks):
        if not (win[0] <= s and e <= win[1]):
            continue
        if X and H.overlap_with(s, e, X, Xs) > 0:
            continue
        keep.append(k)
    out = {"n_tok": len(toks), "n_kept": len(keep), "n_artifact": len(art), "excluded_s": length(X),
           "W_s": win[1] - win[0], "baseline_ms_p50": round(1000 * float(np.median(base)), 1) if len(base) else None,  # 0.1 ms: diagnostic only
           "est": {}, "conv": {}, "conv_window_support": {}}
    den = sum(toks[k][2] for k in keep)
    if den <= 0:
        return out, X
    for kk in K_GRID:
        out["est"][str(kk)] = sum(toks[k][2] for k in keep if toks[k][2] > kk * base[k]) / den
    convP = {"lo_plus1iter": rec["P_lo"], "pt": rec["P_pt"], "earliest": rec["P_hi"],
             "latest": clip(union(list(zip(rec["s_hi"], rec["e"]))), *win)}
    for x, P in convP.items():
        Ps = [p[0] for p in P]
        out["conv"][x] = sum(H.overlap_with(toks[k][1], toks[k][0], P, Ps) for k in keep) / den
    sup = minus([win], X); Wsup = length(sup)
    for x, P in convP.items():
        out["conv_window_support"][x] = length(intersect(intersect(P, rec["D"]), sup)) / Wsup if Wsup > 0 else None
    out["support_s"] = Wsup
    out["kept_itl_time_s"] = den
    if truth_ctx is not None:
        gen, Ptrue = truth_ctx
        Pts = [p[0] for p in Ptrue]
        num = dsum = 0.0
        for k in keep:
            i, j = toks[k][3], toks[k][4]
            a, b = gen[i][j - 1], gen[i][j]
            num += H.overlap_with(a, b, Ptrue, Pts); dsum += b - a
        out["truth_rtw"] = num / dsum if dsum > 0 else None
    return out, X


# ----------------------------------------------------------------------------------- synthetic validation
SYN_SCEN = (("no_stall", {}, False), ("send_stall_3s@100", {"send_stall": (100, 3.0)}, False),
            ("recv_stall_3.9s@mid", "RECV", False), ("STRESS_no_admission_stall_iter", {"stall_iter": False}, True))


def synth_validate():
    rows = []
    for ph in ("LO", "HI"):
        for D in (44, 16):
            base = ST.synth_engine(ph, D, "real")
            mid = 0.5 * (base["win"][0] + base["win"][1])
            for tag, kw, stress in SYN_SCEN:
                if kw == "RECV":
                    kw = {"recv_stall": (mid, 3.9)}
                sy = ST.synth_engine(ph, D, "real", **kw)
                rec = H.reconstruct_round(sy["batches"], sy["A_rel"], sy["ttft"], sy["itls"], sy["duration"])
                # v1 (cluster; failed estimator, reported)
                r1, X1, _ = estimate(rec)
                tw = minus([sy["win"]], X1); Wt = length(tw)
                truth1 = length(intersect(intersect(sy["P_true"], sy["D_true"]), tw)) / Wt if Wt > 0 else None
                # v2 (decision estimator)
                r2, X2 = estimate_v2(rec, truth_ctx=(sy["gen"], sy["P_true"]))
                row = {"phase": ph, "D": D, "scenario": tag, "stress": stress, "truth_window_full": sy["truth"],
                       "v1_cluster": {"truth_window_support": truth1, "est": r1["est"], "excluded_s": r1["excluded_s"],
                                      "cluster_spread_ms_p95": r1["cluster_spread_ms_p95"]},
                       "v2": {"truth_rtw": r2.get("truth_rtw"), "est": r2["est"], "conv": r2["conv"],
                              "conv_window_support": r2["conv_window_support"], "excluded_s": r2["excluded_s"],
                              "n_artifact": r2["n_artifact"]}}
                if "recv_stall" in kw:
                    r0, rr1 = kw["recv_stall"][0], kw["recv_stall"][0] + kw["recv_stall"][1]
                    row["v1_cluster"]["recv_stall_cover_by_X"] = length(intersect([(r0, rr1)], X1)) / (rr1 - r0)
                    row["v2"]["recv_stall_cover_by_X"] = length(intersect([(r0, rr1)], X2)) / (rr1 - r0)
                # exploratory split-exposure truth (post-hoc)
                start_of = {te: t0 for t0, te in sy["iters"]}
                SP = union(list(zip(sy["s_true"], sy["e_true"])))
                SPs = [x[0] for x in SP]
                toks = tokens_of(rec["tok_times"])
                Xs2 = [x[0] for x in X2]
                num = den = 0.0
                for (e_, s_, d_, i_, j_) in toks:
                    if not (rec["window"][0] <= s_ and e_ <= rec["window"][1]):
                        continue
                    if X2 and H.overlap_with(s_, e_, X2, Xs2) > 0:
                        continue
                    g1 = sy["gen"][i_][j_]; g0 = sy["gen"][i_][j_ - 1]
                    t0 = start_of.get(g1)
                    flag = t0 is not None and any(a <= t0 < b for a, b in SP[max(0, __import__("bisect").bisect_right(SPs, t0) - 1):][:1])
                    num += (g1 - g0) * flag; den += g1 - g0
                row["v2"]["truth_split_exposure_EXPLORATORY"] = num / den if den else None
                rows.append(row)
                v = row["v2"]
                print(f"[synth {ph} D{D} {tag:31s}] v2 truthRTW={v['truth_rtw']:.4f} est " +
                      " ".join(f"k{k}={v['est'][str(k)]:.4f}" for k in K_GRID) +
                      " | convRTW lo/pt/earl/lat=" + "/".join(f"{v['conv'][x]:.4f}" for x in ("lo_plus1iter", "pt", "earliest", "latest")) +
                      (f" recvcover={v['recv_stall_cover_by_X']:.3f}" if "recv_stall_cover_by_X" in v else "") +
                      f" excl={v['excluded_s']:.2f}s  [expl. split-exposure truth={v['truth_split_exposure_EXPLORATORY']:.4f}]")
    qual = {"v1_cluster": {}, "v2": {}}
    for ph in ("LO", "HI"):
        for D in (44, 16):
            rs = [r for r in rows if r["phase"] == ph and r["D"] == D and not r["stress"]]
            for k in K_GRID:
                e1 = [r["v1_cluster"]["est"][str(k)] - r["v1_cluster"]["truth_window_support"] for r in rs]
                c1 = all(r["v1_cluster"].get("recv_stall_cover_by_X", 1.0) >= RECV_COVER_MIN for r in rs)
                qual["v1_cluster"][f"{ph}_D{D}_k{k}"] = {"max_abs_err": max(abs(e) for e in e1), "qualifies": max(abs(e) for e in e1) <= QUAL_TOL and c1}
                e2 = [r["v2"]["est"][str(k)] - r["v2"]["truth_rtw"] for r in rs]
                c2 = all(r["v2"].get("recv_stall_cover_by_X", 1.0) >= RECV_COVER_MIN for r in rs)
                qual["v2"][f"{ph}_D{D}_k{k}"] = {"max_abs_err": max(abs(e) for e in e2), "errs": e2,
                                                 "recv_cover_ok": c2, "qualifies": max(abs(e) for e in e2) <= QUAL_TOL and c2}
            # convention errors vs truth in synthetic (descriptive)
    conv_err = {}
    for r in rows:
        if r["stress"]:
            continue
        key = f"{r['phase']}_D{r['D']}"
        for x in ("lo_plus1iter", "pt", "earliest", "latest"):
            conv_err.setdefault(key, {}).setdefault(x, []).append(r["v2"]["conv"][x] - r["v2"]["truth_rtw"])
    qual["v2_vs_split_exposure_EXPLORATORY"] = {}
    for ph in ("LO", "HI"):
        for D in (44, 16):
            rs = [r for r in rows if r["phase"] == ph and r["D"] == D and not r["stress"]]
            for k in K_GRID:
                e3 = [r["v2"]["est"][str(k)] - r["v2"]["truth_split_exposure_EXPLORATORY"] for r in rs]
                qual["v2_vs_split_exposure_EXPLORATORY"][f"{ph}_D{D}_k{k}"] = {"max_abs_err": max(abs(e) for e in e3), "errs": e3}
    kstar = next((k for k in K_GRID if qual["v2"][f"HI_D44_k{k}"]["qualifies"] and qual["v2"][f"HI_D16_k{k}"]["qualifies"]), None)
    return rows, qual, kstar, conv_err


# ----------------------------------------------------------------------------------- real data
def real():
    per = []
    for arm, jobs in H.HE0_JOBS.items():
        for job in jobs:
            R = F.recon(job)
            for rd in R["rounds"]:
                rec = rd["rec"]
                r2, X = estimate_v2(rec)
                W = rec["window"][1] - rec["window"][0]
                full = {"pt": length(rec["CO_pt"]) / W, "lo_plus1iter": length(rec["CO_lo"]) / W, "earliest": length(rec["CO_hi"]) / W}
                per.append({"arm": arm, "job": job, "phase": rd["phase"], "round": rd["round"],
                            "quality_fail": (job, rd["phase"], rd["round"]) in QUALITY_FAIL,
                            "W_s": W, "excluded_s": r2["excluded_s"], "n_artifact": r2["n_artifact"],
                            "kept_itl_time_s": r2.get("kept_itl_time_s", 0.0), "baseline_ms_p50": r2["baseline_ms_p50"],
                            "est": r2["est"], "conv": r2["conv"], "conv_window_support": r2["conv_window_support"],
                            "conv_window_full_he0": full})
            print(f"ok {arm} {job}", file=sys.stderr)
    return per


def arm_table(per, phase, drop_quality=False):
    out = {}
    for arm in H.HE0_JOBS:
        rows = [r for r in per if r["arm"] == arm and r["phase"] == phase and not (drop_quality and r["quality_fail"])]
        jobs = sorted(set(r["job"] for r in rows))

        def runval(job, f, wkey="kept_itl_time_s"):
            rr = [r for r in rows if r["job"] == job and r[wkey] > 0 and f(r) is not None]
            S = sum(r[wkey] for r in rr)
            return sum(f(r) * r[wkey] for r in rr) / S if S else None
        a = {"n_runs": len(jobs), "est_rtw": {}, "conv_rtw": {}, "conv_window_full_he0": {}}
        for k in K_GRID:
            a["est_rtw"][str(k)] = F.msd([runval(j, lambda r: r["est"].get(str(k))) for j in jobs])
        for x in ("lo_plus1iter", "pt", "earliest", "latest"):
            a["conv_rtw"][x] = F.msd([runval(j, lambda r: r["conv"].get(x)) for j in jobs])
        for x in ("lo_plus1iter", "pt", "earliest"):
            a["conv_window_full_he0"][x] = F.msd([runval(j, lambda r: r["conv_window_full_he0"][x], "W_s") for j in jobs])
        a["excluded_frac"] = sum(r["excluded_s"] for r in rows) / sum(r["W_s"] for r in rows)
        a["n_artifact"] = sum(r["n_artifact"] for r in rows)
        out[arm] = a
    return out


def decide(tab, kstar, band_half):
    rows, n_real, n_notreal = {}, 0, 0
    for arm, a in tab.items():
        est = a["est_rtw"][str(kstar)]["mean"]
        inside = {x: abs(a["conv_rtw"][x]["mean"] - est) <= band_half for x in ("lo_plus1iter", "pt", "earliest", "latest")}
        narrowed = (not inside["earliest"]) and (not inside["latest"]) and (inside["lo_plus1iter"] or inside["pt"])
        none_in = not any(inside.values())
        n_real += narrowed; n_notreal += none_in
        rows[arm] = {"est": est, "band": [est - band_half, est + band_half], "inside": inside, "narrowed": narrowed, "none_inside": none_in}
    verdict = "REAL" if n_real >= 6 else ("NOT-REAL" if n_notreal >= 6 else "INCONCLUSIVE")
    return {"per_arm": rows, "arms_narrowed": n_real, "arms_none_inside": n_notreal, "verdict": verdict, "band_half": band_half}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "he0_coresidency_indep_2026-10-02.json"))
    ap.add_argument("--synthetic-only", action="store_true")
    args = ap.parse_args()
    rows, qual, kstar, conv_err = synth_validate()
    for ver in ("v1_cluster", "v2", "v2_vs_split_exposure_EXPLORATORY"):
        for kk, v in qual[ver].items():
            print(f"  qual {ver:10s} {kk:16s} max|err|={v['max_abs_err']:.4f} qualifies={v.get('qualifies', 'n/a(exploratory)')}")
    for key, d in conv_err.items():
        print(f"  synthetic convention err vs truth RTW {key}: " + " ".join(f"{x}=[{min(v):+.3f},{max(v):+.3f}]" for x, v in d.items()))
    print("k* (v2) =", kstar)
    out = {"generated": "2026-10-02", "script": Path(__file__).name,
           "prereg": {"K_GRID": K_GRID, "BASE_Q": BASE_Q, "BASE_HALF_S": BASE_HALF_S, "CLUSTER_GAP_S": CLUSTER_GAP_S,
                      "GAP_ART_S": GAP_ART_S, "GUARD_S": GUARD_S, "QUAL_TOL": QUAL_TOL, "BIN_S": BIN_S, "BIN_HALF": BIN_HALF,
                      "BUNCH_LO": BUNCH_LO, "BUNCH_HI": BUNCH_HI, "amendment": 1},
           "synthetic": rows, "qualification": qual, "synthetic_convention_err_rtw": conv_err, "kstar_v2": kstar}
    if not args.synthetic_only:
        per = real()
        out["real_per_round"] = per
        out["real_arm_table"] = {ph: arm_table(per, ph) for ph in ("LO", "HI")}
        out["real_arm_table_drop_quality_fail"] = {"HI": arm_table(per, "HI", drop_quality=True)}
        if kstar is not None:
            band_half = max(QUAL_TOL, max(qual["v2"][f"HI_D{D}_k{kstar}"]["max_abs_err"] for D in (44, 16)))
            out["decision"] = decide(out["real_arm_table"]["HI"], kstar, band_half)
            out["decision_drop_quality_fail"] = decide(out["real_arm_table_drop_quality_fail"]["HI"], kstar, band_half)
        else:
            out["decision"] = {"verdict": "INCONCLUSIVE", "reason": "no k qualifies on synthetic HI regimes (v2)"}
        for ph in ("LO", "HI"):
            print(f"== real {ph} (RTW)")
            for arm, a in out["real_arm_table"][ph].items():
                print(f"  {arm:12s} n={a['n_runs']} excl={a['excluded_frac']:.4f} est " +
                      " ".join(f"k{k}={a['est_rtw'][str(k)]['mean']:.4f}±{a['est_rtw'][str(k)]['sd']:.4f}" for k in K_GRID) +
                      " | convRTW lo/pt/earl/lat=" + "/".join(f"{a['conv_rtw'][x]['mean']:.4f}" for x in ("lo_plus1iter", "pt", "earliest", "latest")) +
                      " | he0 window lo/pt/earl=" + "/".join(f"{a['conv_window_full_he0'][x]['mean']:.4f}" for x in ("lo_plus1iter", "pt", "earliest")))
        print("decision:", json.dumps(F.rnd(out["decision"])))
        if "decision_drop_quality_fail" in out:
            print("decision (drop quality-fail rounds):", out["decision_drop_quality_fail"]["verdict"])
    Path(args.output).write_text(json.dumps(F.rnd(out), indent=1))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
