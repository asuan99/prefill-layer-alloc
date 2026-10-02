#!/usr/bin/env python3
"""HE0 realized-partition / treatment-exposure re-aggregation (GPU 0, 2026-10-01).

Question: in the HE0 change-trace campaign (ShareGPT rate 3<->12, L3H12, jobs
852340-859059, 2026-07-15..18) how much of the run did the arm label actually
act on?  This engine (``adjust_stream_groups``, identical at the HE0 commit
fdbf5d1 and the 2026-07-31 commit on which CONSENSUS 1-25 measured it) applies
the labelled decode split ONLY while a split-prefill batch is in flight AND the
decode running batch is non-empty ("co-residency").  Otherwise:
decode-only -> (0,108), prefill-only / idle -> (108,0) for EVERY arm.

The HE0 runs have NO per-iteration telemetry (no decode_sms/stream idx field).
This script therefore RECONSTRUCTS the timeline from two independent sources
and cross-validates them:

  * server log  : one "Prefill batch" line per admitted prefill batch
                  (1-second timestamp, #new-seq, #new-token, #running-req);
                  "Decode batch" line every 40 decode iterations;
                  controller lines SLO-BIND / SLO-SCHED / SLO-FEAS.
  * client jsonl: per request ttft + token ITLs (bench_serving --output-details),
                  one JSON object per round (3 LO + 3 HI rounds per job).
  * arrival schedule: bench_serving seeds np.random with --seed (default 1) and
                  draws np.random.exponential(1/rate) per request => the schedule
                  is deterministic up to a per-round clock offset (fitted).

Reconstruction (per round):
  1. partition the 200 requests (FCFS, no radix) into the logged prefill batches;
     asserted EXACTLY by sum(input_lens) == #new-token for every batch.
  2. fit clock offset (+ linear send-drift) so that every batch satisfies
     arrival <= admission in [T_log, T_log+1) and admission(k+1) >= end(k).
  3. prefill in-flight interval k = [s_k, e_k]; e_k = first-token receipt of the
     batch; s_k bracketed by [max(T_k, e_{k-1}, last arrival), min(T_k+1, e_k)],
     point estimate = earliest instant in the bracket where the reconstructed
     running count equals the logged #running-req (else bracket low end).
  4. decode-active = union over requests of [e_batch(i), last-token receipt).
  5. co-residency = prefill-in-flight AND decode-active.

All time fractions are reported as a RANGE over (a) prefill start = point
estimate +1 decode iteration ("lo"), point estimate ("pt"), earliest feasible
start ("hi"), and (b) for dynamic arms, earliest vs latest placement of each
1-second controller transition.  Count-weighted (per prefill batch / per prefill
token, log-only, no reconstruction) and token-weighted variants are reported
alongside — weighting convention changes the number, so no single value is
quoted (lesson 252 / E2C-8').

Reuses ``pdmux_eval.analyze`` (load_bench_serving_rounds, summarize_requests,
controller_summary, percentile, unpaired_bootstrap_ci).  Telemetry-based realized
analyzers (smsplit_realized, s2_sticky, p1_gates) are NOT reusable: they consume
runtime_snapshot telemetry which these runs never produced — the timeline
reconstruction here is new code (stated in RESULT).

Usage:
  python3 he0_realized.py            # full run -> he0_realized_2026-10-01.json
  python3 he0_realized.py --self-test
"""
from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # .../prefill-layer-alloc
SRC = HERE.parent / "slo_sched"
sys.path.insert(0, str(HERE.parents[1] / "benchmarks"))
from pdmux_eval import analyze as A  # noqa: E402

TTFT_SLO_MS = 3000.0
ITL_SLO_MS = 60.0
MAX_RUNNING = 48
SM_TOTAL = 108

# HE0 membership: CONSENSUS 1-7 / bench_noise_root_cause.md 4 (n per arm) and
# CONSENSUS 1-13 footnote E (d16/d24/slo n=4 = rep1 + rep44-46).  Gate vs
# no-gate is not in the server log env dump; classification evidence is the
# presence of SLO-FEAS lines (gate code path) and date (gate implemented
# 2026-07-16, commit 81aab84) — recorded per run in the output.
HE0_JOBS: Dict[str, List[int]] = {
    "d44": [852342, 856889, 856890, 856891],
    "d34": [856892, 856917, 856918, 856929],
    "d24": [852341, 859005, 859006, 859007],
    "d16": [852340, 859008, 859025, 859026],
    "bind_gate": [855250, 855254, 856930, 856941, 856942, 857051, 857052, 857111, 857112],
    "bind_nogate": [852435, 856963, 856964, 856975],
    "slo": [852343, 859037, 859038, 859059],
}
STATIC_D = {"d44": 44, "d34": 34, "d24": 24, "d16": 16}
DYNAMIC = ("bind_gate", "bind_nogate", "slo")

RE_TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]"
RE_PREFILL = re.compile(RE_TS + r" Prefill batch, #new-seq: (\d+), #new-token: (\d+), .*?#running-req: (\d+), #queue-req: (\d+)")
RE_DECODE = re.compile(RE_TS + r" Decode batch, #running-req: (\d+),")
RE_GROUPS = re.compile(r"sm_counts \(prefill_sm, decode_sm\): (\[.*\])")
RE_BIND = re.compile(RE_TS + r" SLO-(BIND|SCHED) (\d+)->(\d+) ")
RE_FEAS = re.compile(RE_TS + r" SLO-FEAS refused idx=(\d+)->(\d+) ")
RE_ANCHOR = re.compile(r"anchor=(\d+)")


def ts(s: str) -> float:
    return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


# --------------------------------------------------------------------------- intervals
def union(iv: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted((a, b) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def intersect(x: Sequence[Tuple[float, float]], y: Sequence[Tuple[float, float]], _mut: str = "") -> List[Tuple[float, float]]:
    if _mut == "intersect_is_union":  # self-test mutant
        return union(list(x) + list(y))
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a = max(x[i][0], y[j][0]); b = min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(x, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in x if min(b, hi) > max(a, lo)]


def length(x) -> float:
    return sum(b - a for a, b in x)


def minus(x, y):
    return _sub(union(x), union(y))


def _sub(x, y):
    out = []
    j = 0
    for a, b in x:
        cur = a
        while j < len(y) and y[j][1] <= cur:
            j += 1
        k = j
        while k < len(y) and y[k][0] < b:
            if y[k][0] > cur:
                out.append((cur, y[k][0]))
            cur = max(cur, y[k][1])
            k += 1
        if cur < b:
            out.append((cur, b))
    return out


def overlap_with(a: float, b: float, iv: List[Tuple[float, float]], starts: List[float]) -> float:
    if b <= a:
        return 0.0
    k = max(0, bisect.bisect_right(starts, a) - 1)
    tot = 0.0
    while k < len(iv) and iv[k][0] < b:
        tot += max(0.0, min(b, iv[k][1]) - max(a, iv[k][0]))
        k += 1
    return tot


# --------------------------------------------------------------------------- parsing
def parse_server(path: Path) -> Dict[str, object]:
    pb, dec, ctl, feas = [], [], [], []
    groups = None
    anchor = None
    for ln in path.open(errors="replace"):
        m = RE_PREFILL.match(ln)
        if m:
            pb.append((ts(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
            continue
        m = RE_DECODE.match(ln)
        if m:
            dec.append((ts(m.group(1)), int(m.group(2))))
            continue
        m = RE_BIND.match(ln)
        if m:
            ctl.append((ts(m.group(1)), int(m.group(3)), int(m.group(4))))
            ma = RE_ANCHOR.search(ln)
            if ma and anchor is None:
                anchor = int(ma.group(1))
            continue
        m = RE_FEAS.match(ln)
        if m:
            feas.append((ts(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        m = RE_GROUPS.search(ln)
        if m and groups is None:
            groups = [tuple(x) for x in json.loads(m.group(1).replace("(", "[").replace(")", "]"))]
    return {"prefill": pb, "decode": dec, "ctl": ctl, "feas": feas, "groups": groups, "anchor": anchor}


def arrival_unit(n: int, seed: int = 1) -> np.ndarray:
    iv = np.random.RandomState(seed).exponential(1.0, n)
    return np.concatenate([[0.0], np.cumsum(iv[:-1])])


# --------------------------------------------------------------------------- reconstruction
def _batch_bounds(batches, A, ttft, c):
    lo, hi = [], []
    for k, (T, S, _r, _q) in enumerate(batches):
        fmin = min(A[i] + ttft[i] + c * i for i in S)
        amax = max(A[i] + c * i for i in S)
        l = T - fmin
        h = T + 1 - amax
        if k + 1 < len(batches):
            h = min(h, batches[k + 1][0] + 1 - fmin)
        lo.append(l); hi.append(h)
    return lo, hi


def fit_offset_piecewise(batches, A, ttft, _mut: str = "", tol: float = 0.0):
    """Piecewise-linear clock offset: offset_seg + c_seg * i per request.

    bench_serving sleeps RELATIVE exponential intervals, so client send lateness
    accumulates (slow drift ~0.1-1.3 ms/request) and jumps when the client event
    loop stalls.  Greedy: extend a segment while SOME drift c on the grid keeps
    every batch constraint feasible (width >= -tol); otherwise break.  Each
    segment uses its max-width c and the midpoint offset.  Returns per-request
    offsets, per-segment widths, segment count.
    """
    grid = np.linspace(-0.0005, 0.003, 36)
    bounds = [_batch_bounds(batches, A, ttft, c) for c in grid]
    nb = len(batches)
    segs = []
    k0 = 0
    while k0 < nb:
        L = [-math.inf] * len(grid); H = [math.inf] * len(grid)
        best = None
        k = k0
        while k < nb:
            nL = [max(L[g], bounds[g][0][k]) for g in range(len(grid))]
            nH = [min(H[g], bounds[g][1][k]) for g in range(len(grid))]
            ws = [nH[g] - nL[g] for g in range(len(grid))]
            if max(ws) < -tol and k > k0:
                break
            L, H = nL, nH
            g = max(range(len(grid)), key=lambda g: (ws[g], -abs(grid[g])))
            best = (g, ws[g], 0.5 * (nL[g] + nH[g]))
            k += 1
        segs.append((k0, k, grid[best[0]], best[2], best[1]))
        k0 = k
    xreq = {}
    for k0, k1, c, off, w in segs:
        for k in range(k0, k1):
            for i in batches[k][1]:
                xreq[i] = off + c * i
    if _mut == "offset_plus_1s":
        xreq = {i: v + 1.0 for i, v in xreq.items()}
    return xreq, segs


def reconstruct_round(batches, A_rel, ttft, itls, duration, _mut: str = ""):
    """batches: list of (T_log, [req idx], R_logged, Q_logged). Times in server clock."""
    n = len(ttft)
    xreq, segs = fit_offset_piecewise(batches, A_rel, ttft, _mut=_mut)
    off = xreq[0] + 0.0
    c = segs[0][2]
    width = min(sg[4] for sg in segs)
    a = [xreq[i] + A_rel[i] for i in range(n)]
    first = [a[i] + ttft[i] for i in range(n)]
    tok_times = []
    for i in range(n):
        t = first[i]
        tt = [t]
        for d in itls[i]:
            t += d
            tt.append(t)
        tok_times.append(tt)
    fin = [tt[-1] for tt in tok_times]
    bidx = [None] * n
    e, spread = [], []
    for k, (T, S, R, Q) in enumerate(batches):
        fs = [first[i] for i in S]
        e.append(min(fs))
        spread.append(max(fs) - min(fs))
        for i in S:
            bidx[i] = k
    # running-count step function (request in running batch from e_batch to finish)
    ev = []
    for i in range(n):
        ev.append((e[bidx[i]], +1))
        ev.append((max(fin[i], e[bidx[i]]), -1))
    ev.sort()
    times, cnt = [], []
    cur = 0
    for t, d in ev:
        cur += d
        times.append(t)
        cnt.append(cur)

    def running(t: float) -> int:
        k = bisect.bisect_right(times, t) - 1
        return cnt[k] if k >= 0 else 0

    s_lo, s_hi, s_pt, match = [], [], [], {"exact": 0, "pm1": 0, "none": 0, "bracket_inverted": 0}
    for k, (T, S, R, Q) in enumerate(batches):
        lo = max(T, e[k - 1] if k else -math.inf, max(a[i] for i in S))
        hi = min(T + 1.0, e[k])
        if lo > hi:
            match["bracket_inverted"] += 1
            lo, hi = min(lo, hi), min(lo, hi)
        s_lo.append(lo); s_hi.append(hi)
        cands = [lo] + [t for t in times if lo < t <= hi]
        pt = None
        for t in cands:
            if running(t) == R:
                pt = t; match["exact"] += 1; break
        if pt is None:
            for t in cands:
                if abs(running(t) - R) <= 1:
                    pt = t; match["pm1"] += 1; break
        if pt is None:
            pt = lo; match["none"] += 1
        s_pt.append(pt)
    win = (off, off + duration)
    D = clip(union([(e[bidx[i]], fin[i]) for i in range(n)]), *win)
    out = {"offset": off, "drift_s_per_req": c, "offset_width_s": width, "window": win,
           "offset_segments": len(segs), "offset_seg_widths_s": [sg[4] for sg in segs],
           "offset_seg_bounds": [(sg[0], sg[1]) for sg in segs],
           "lateness_total_s": (xreq[n - 1] - xreq[0]),
           "e": e, "s_lo": s_lo, "s_hi": s_hi, "s_pt": s_pt, "spread_first_token_s": spread,
           "R_match": match, "shift_step_s": None, "a": a, "first": first, "fin": fin, "bidx": bidx,
           "tok_times": tok_times, "D": D, "running": running}
    # "shift" = point estimate delayed by one decode iteration (median token ITL of
    # the round): engine loop filters a finished request one iteration after it
    # finishes and admits the next prefill after that, so s_pt (client-side drop of
    # the running count) is early by ~1 iteration; measured lag to first slow token
    # p50 1-14 ms (RESULT §Q5).  "hi" = earliest feasible start (upper bound).
    all_itl = [d for i in range(n) for d in itls[i]]
    step = statistics.median(all_itl) if all_itl else 0.0
    s_shift = [min(e[k], s_pt[k] + step) for k in range(len(e))]
    out_step = step
    for tag, s in (("pt", s_pt), ("lo", s_shift), ("hi", s_lo)):
        P = clip(union(list(zip(s, e))), *win)
        out["P_" + tag] = P
        out["CO_" + tag] = intersect(P, D, _mut=_mut)
    out["shift_step_s"] = out_step
    return out


def controller_timeline(ctl, feas, co: List[Tuple[float, float]], start_idx: int, place: str):
    """Place each 1-s controller transition at earliest/latest co-res instant in [T,T+1)."""
    starts = [x[0] for x in co]
    pts, chain_ok, unplaced = [], True, 0
    cur = start_idx
    for T, a, b in ctl:
        if a != cur:
            chain_ok = False
        cur = b
        k = max(0, bisect.bisect_right(starts, T) - 1)
        inst = []
        while k < len(co) and co[k][0] < T + 1:
            lo, hi = max(co[k][0], T), min(co[k][1], T + 1)
            if hi > lo:
                inst.append((lo, hi))
            k += 1
        if not inst:
            unplaced += 1
            t = T + 0.5
        else:
            t = inst[0][0] if place == "earliest" else inst[-1][1] - 1e-6
        pts.append((t, b))
    # monotone placement
    fixed, last = [], -math.inf
    for t, b in pts:
        t = max(t, last)
        fixed.append((t, b)); last = t
    def idx_at(t):
        k = bisect.bisect_right([p[0] for p in fixed], t) - 1
        return fixed[k][1] if k >= 0 else start_idx
    fe_ok = fe_tot = 0
    for T, x, _y in feas:
        fe_tot += 1
        # state at any co-res instant inside [T,T+1)
        k = max(0, bisect.bisect_right(starts, T) - 1)
        vals = set()
        while k < len(co) and co[k][0] < T + 1:
            lo, hi = max(co[k][0], T), min(co[k][1], T + 1)
            if hi > lo:
                vals.add(idx_at((lo + hi) / 2))
            k += 1
        if x in vals or (not vals and idx_at(T + 0.5) == x):
            fe_ok += 1
    return fixed, idx_at, {"chain_consistent": chain_ok, "unplaced_transitions": unplaced,
                           "feas_state_agree": fe_ok, "feas_lines": fe_tot}


def sm_split_of_co(co, fixed, groups, start_idx):
    """Split co-res intervals by controller idx -> {decode_sm: seconds}."""
    bounds = [p[0] for p in fixed]
    res: Dict[int, float] = {}
    for a, b in co:
        cuts = [a] + [t for t in bounds if a < t < b] + [b]
        for x, y in zip(cuts, cuts[1:]):
            k = bisect.bisect_right(bounds, (x + y) / 2) - 1
            idx = fixed[k][1] if k >= 0 else start_idx
            d = int(groups[idx][1])
            res[d] = res.get(d, 0.0) + (y - x)
    return res


# --------------------------------------------------------------------------- per run
def run_one(arm: str, job: int, _mut: str = "") -> Dict[str, object]:
    srv = list(SRC.glob(f"sgptvsrv_*_{job}.log"))[0]
    tag = srv.name[len("sgptvsrv_"):-4]
    lo_f, hi_f = SRC / f"sgptvLo_{tag}.jsonl", SRC / f"sgptvHi_{tag}.jsonl"
    S = parse_server(srv)
    lo = [json.loads(x) for x in lo_f.open() if x.strip()]
    hi = [json.loads(x) for x in hi_f.open() if x.strip()]
    assert len(lo) == 3 and len(hi) == 3, (job, len(lo), len(hi))
    rounds = [("LO", 0, lo[0]), ("HI", 0, hi[0]), ("LO", 1, lo[1]), ("HI", 1, hi[1]), ("LO", 2, lo[2]), ("HI", 2, hi[2])]
    pb = S["prefill"]
    k = 0
    while k < len(pb) and not (pb[k][1] == 1 and pb[k][2] == rounds[0][2]["input_lens"][0] and pb[k][3] == 0):
        k += 1  # skip server-internal warmups
    groups = S["groups"]
    is_dyn = arm in DYNAMIC
    if arm.startswith("bind"):
        start_idx = S["ctl"][0][1] if S["ctl"] else (S["feas"][0][1] if S["feas"] else 2)
    elif arm == "slo":
        start_idx = S["ctl"][0][1] if S["ctl"] else (1 + len(groups) - 2) // 2
    else:
        start_idx = 1
    node = None
    outp = SRC / f"sgptv_{job}.out"
    out_combined = None
    if outp.exists():
        txt = outp.read_text(errors="replace")
        mm = re.search(r"SLURM_NODELIST = (\S+)", txt); node = mm.group(1) if mm else None
        mm = re.search(r"COMBINED=([0-9.]+)", txt); out_combined = float(mm.group(1)) if mm else None
    recs = []
    for phase, ri, o in rounds:
        L = o["input_lens"]; n = len(L)
        rate = float(o["request_rate"])
        assert pb[k][1] == 1 and pb[k][2] == L[0], ("warmup expected", job, phase, ri, pb[k])
        k += 1
        batches, i = [], 0
        while i < n:
            T, ns, nt, R, Q = pb[k]
            Sx = list(range(i, i + ns))
            assert sum(L[j] for j in Sx) == nt, ("partition token-sum mismatch", job, phase, ri, k)
            batches.append((T, Sx, R, Q)); i += ns; k += 1
        A_rel = arrival_unit(n, seed=(2 if _mut == "arrival_seed" else 1)) / rate
        ttft = [float(x) for x in o["ttfts"]]
        itls = [[float(v) for v in x] for x in o["itls"]]
        rec = reconstruct_round(batches, A_rel, ttft, itls, float(o["duration"]), _mut=_mut)
        recs.append((phase, ri, o, batches, ttft, itls, rec))
    # controller timeline over the WHOLE run (state persists across rounds)
    timelines = {}
    ctl_check = {}
    if is_dyn:
        cot = union(sum((rc[6]["CO_pt"] for rc in recs), []))
        for place in ("earliest", "latest"):
            fixed, idx_at, info = controller_timeline(S["ctl"], S["feas"], cot, start_idx, place)
            timelines[place] = fixed
            ctl_check[place] = info
    rr, reqrows = [], []
    for phase, ri, o, batches, ttft, itls, rec in recs:
        L = o["input_lens"]; n = len(L); rate = float(o["request_rate"])
        win = rec["window"]; W = win[1] - win[0]
        D = rec["D"]
        frac = {}
        for v in ("pt", "lo", "hi"):
            P, CO = rec["P_" + v], rec["CO_" + v]
            frac[v] = {"co": length(CO) / W, "prefill_only": length(minus(P, D)) / W,
                       "decode_only": length(minus(D, P)) / W,
                       "idle": 1 - length(union(P + D)) / W,
                       "co_s": length(CO), "decode_only_s": length(minus(D, P)),
                       "prefill_only_s": length(minus(P, D)), "W_s": W, "D_s": length(D)}
        # independent validation: "Decode batch" log lines (never used in fitting)
        dv = {"lines": 0, "hit": 0, "absdiff": []}
        for Td, nlog in S["decode"]:
            if not (win[0] <= Td < win[1]):
                continue
            vals = [rec["running"](Td + u / 50.0) for u in range(50)]
            dv["lines"] += 1
            dv["hit"] += int(nlog in vals)
            dv["absdiff"].append(min(abs(v - nlog) for v in vals))
        dval = {"lines": dv["lines"], "hit_rate": dv["hit"] / dv["lines"] if dv["lines"] else None,
                "absdiff_p95": A.percentile(dv["absdiff"], 0.95) if dv["absdiff"] else None}
        rmatch = rec["R_match"]
        nmatch = rmatch["exact"] + rmatch["pm1"] + rmatch["none"]
        quality_ok = (rmatch["exact"] / nmatch >= 0.9) and (rec["lateness_total_s"] > -0.1) and (rec["offset_width_s"] > -0.005)
        nb = len(batches)
        cw = {"batches": nb, "batches_R_gt0": sum(1 for b in batches if b[2] > 0),
              "tokens": sum(L), "tokens_R_gt0": sum(sum(L[j] for j in b[1]) for b in batches if b[2] > 0)}
        sm = {}
        if is_dyn:
            for place, fixed in timelines.items():
                sm[place] = sm_split_of_co(rec["CO_pt"], fixed, groups, start_idx)
        else:
            sm["static"] = {STATIC_D[arm]: length(rec["CO_pt"])}
        P = rec["P_pt"]; Pst = [x[0] for x in P]
        for i in range(n):
            b = rec["bidx"][i]
            tt = rec["tok_times"][i]
            dec_time = tt[-1] - tt[0]
            ov = [overlap_with(tt[j - 1], tt[j], P, Pst) for j in range(1, len(tt))]
            dl = [tt[j] - tt[j - 1] for j in range(1, len(tt))]
            mid = [(tt[j - 1] + tt[j]) / 2 for j in range(1, len(tt))]
            isco = [o_ > 0.5 * d_ for o_, d_ in zip(ov, dl)]
            itl_ms = [1000.0 * v for v in itls[i]]
            p95 = A.percentile(itl_ms, 0.95) if itl_ms else None
            reqrows.append({
                "phase": phase, "round": ri, "i": i,
                "ttft_ms": 1000.0 * ttft[i],
                "queue_ms": 1000.0 * (rec["s_pt"][b] - rec["a"][i]),
                "exec_ms": 1000.0 * (rec["e"][b] - rec["s_pt"][b]),
                "prefill_cores_logged": batches[b][2] > 0,
                "decode_exposure": (sum(ov) / dec_time) if dec_time > 0 else None,
                "itl_p95_ms": p95,
                "pass": (1000.0 * ttft[i] <= TTFT_SLO_MS) and (p95 is None or p95 <= ITL_SLO_MS),
                "pass_ttft": 1000.0 * ttft[i] <= TTFT_SLO_MS,
                "pass_itl": (p95 is None or p95 <= ITL_SLO_MS),
                "_tok_co": [1000.0 * d_ for d_, c_ in zip(dl, isco) if c_],
                "_tok_do": [1000.0 * d_ for d_, c_ in zip(dl, isco) if not c_],
                "_bs_co": [rec["running"](m_) for m_, c_ in zip(mid, isco) if c_],
                "_bs_do": [rec["running"](m_) for m_, c_ in zip(mid, isco) if not c_],
            })
        rr.append({"phase": phase, "round": ri, "rate": rate, "n_batches": nb,
                   "offset_fit_width_s": rec["offset_width_s"], "offset_segments": rec["offset_segments"],
                   "offset_seg_widths_s": rec["offset_seg_widths_s"], "offset_seg_bounds": rec["offset_seg_bounds"],
                   "client_send_lateness_total_s": rec["lateness_total_s"],
                   "R_match": rec["R_match"],
                   "first_token_spread_ms_p95": 1000 * A.percentile(rec["spread_first_token_s"], 0.95),
                   "decode_line_validation": dval, "quality_ok": quality_ok,
                   "frac": frac, "count_weighted": cw, "co_split_s": sm, "duration_s": float(o["duration"])})
    assert k == len(pb), ("unconsumed prefill lines", job, k, len(pb))
    target_sum = None
    if is_dyn:
        cot = union(sum((rc[6]["CO_pt"] for rc in recs), []))
        fixed = timelines["earliest"]
        t0 = cot[0][0] if cot else 0.0
        t1 = max([cot[-1][1]] + [t for t, _ in fixed]) if cot else 0.0
        ev = [{"event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": t0, "decode_sms": int(groups[start_idx][1])}]
        ev += [{"event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": t, "decode_sms": int(groups[b][1])} for t, b in fixed]
        ev.append({"event": "runtime_snapshot", "phase": "benchmark", "timestamp_monotonic_s": t1,
                   "decode_sms": int(groups[fixed[-1][1]][1]) if fixed else int(groups[start_idx][1])})
        cs = A.controller_summary(ev)
        target_sum = {"note": "controller TARGET state, wall-time weighted from first to last co-res instant (NOT realized split)",
                      "split_transitions": cs["split_transitions"], "residency_fraction": cs["residency_fraction"],
                      "dwell_s_p50": A.percentile(cs["dwell_s"], 0.5), "dwell_s_p95": A.percentile(cs["dwell_s"], 0.95),
                      "n_dwell": len(cs["dwell_s"])}
    return {"arm": arm, "job": job, "tag": tag, "node": node, "groups": groups, "start_idx": start_idx,
            "n_ctl_transitions": len(S["ctl"]), "n_feas_lines": len(S["feas"]),
            "out_combined_reported": out_combined, "rounds": rr, "controller_check": ctl_check,
            "controller_target_summary": target_sum, "_req": reqrows,
            "client_files": [lo_f.name, hi_f.name], "server_log": srv.name}


# --------------------------------------------------------------------------- aggregation
def msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None, "n": len(xs),
            "min": min(xs), "max": max(xs)}


def goodput_rows(run):
    tag = run["tag"]
    out = {}
    for ph, f in (("LO", f"sgptvLo_{tag}.jsonl"), ("HI", f"sgptvHi_{tag}.jsonl")):
        reqs, dur = A.load_bench_serving_rounds(SRC / f)
        s = A.summarize_requests(reqs, 0.0, dur, TTFT_SLO_MS, ITL_SLO_MS)
        out[ph] = s
    reqsL, dL = A.load_bench_serving_rounds(SRC / f"sgptvLo_{tag}.jsonl")
    reqsH, dH = A.load_bench_serving_rounds(SRC / f"sgptvHi_{tag}.jsonl")
    out["COMBINED"] = A.summarize_requests(reqsL + reqsH, 0.0, dL + dH, TTFT_SLO_MS, ITL_SLO_MS)
    return out


def aggregate(runs, quality_only: bool = False):
    if quality_only:
        runs = [dict(r, rounds=[x for x in r["rounds"] if x["quality_ok"]]) for r in runs]
    agg = {}
    for arm in HE0_JOBS:
        rs = [r for r in runs if r["arm"] == arm]
        a = {"n_runs": len(rs), "jobs": [r["job"] for r in rs], "nodes": [r["node"] for r in rs]}
        for ph in ("LO", "HI"):
            per = {}
            for v in ("pt", "lo", "hi"):
                for key in ("co", "decode_only", "prefill_only", "idle", "co_over_decode_active"):
                    vals = []
                    for r in rs:
                        rows = [x for x in r["rounds"] if x["phase"] == ph]
                        if not rows:
                            continue
                        Wsum = sum(x["frac"][v]["W_s"] for x in rows)
                        if key == "co_over_decode_active":
                            vals.append(sum(x["frac"][v]["co_s"] for x in rows) / sum(x["frac"][v]["D_s"] for x in rows))
                        else:
                            vals.append(sum(x["frac"][v][key] * x["frac"][v]["W_s"] for x in rows) / Wsum)
                    per.setdefault(key, {})[v] = msd(vals)
            # count-weighted
            cwb, cwt = [], []
            for r in rs:
                rows = [x for x in r["rounds"] if x["phase"] == ph]
                if not rows:
                    continue
                cwb.append(sum(x["count_weighted"]["batches_R_gt0"] for x in rows) / sum(x["count_weighted"]["batches"] for x in rows))
                cwt.append(sum(x["count_weighted"]["tokens_R_gt0"] for x in rows) / sum(x["count_weighted"]["tokens"] for x in rows))
            per["count_w_batches_cores"] = msd(cwb)
            per["count_w_prefill_tokens_cores"] = msd(cwt)
            # token-weighted decode exposure
            tw = []
            for r in rs:
                rq = [q for q in r["_req"] if q["phase"] == ph]
                nco = sum(len(q["_tok_co"]) for q in rq); ndo = sum(len(q["_tok_do"]) for q in rq)
                tw.append(nco / max(1, nco + ndo))
            per["token_w_decode_tokens_in_prefill_inflight"] = msd(tw)
            # realized decode SM over co-res time
            smd = {}
            for place in (("static",) if arm in STATIC_D else ("earliest", "latest")):
                tot: Dict[int, float] = {}
                means = []
                for r in rs:
                    rt: Dict[int, float] = {}
                    for x in r["rounds"]:
                        if x["phase"] != ph:
                            continue
                        for d, s in x["co_split_s"][place].items():
                            tot[d] = tot.get(d, 0.0) + s
                            rt[d] = rt.get(d, 0.0) + s
                    ssum = sum(rt.values())
                    means.append(sum(d * s for d, s in rt.items()) / ssum if ssum else None)
                T = sum(tot.values())
                smd[place] = {"fraction_by_decode_sm": {str(d): s / T for d, s in sorted(tot.items())} if T else {},
                              "mean_decode_sm_over_cores_time_per_run": msd(means)}
            per["cores_decode_sm"] = smd
            # quality checks
            per["offset_fit_width_s_min"] = min(x["offset_fit_width_s"] for r in rs for x in r["rounds"] if x["phase"] == ph)
            per["rounds_used"] = sum(1 for r in rs for x in r["rounds"] if x["phase"] == ph)
            dl = [x["decode_line_validation"] for r in rs for x in r["rounds"] if x["phase"] == ph]
            per["decode_line_validation"] = {"lines": sum(z["lines"] for z in dl),
                                             "hit_rate": sum(z["hit_rate"] * z["lines"] for z in dl if z["lines"]) / max(1, sum(z["lines"] for z in dl))}
            rm = {"exact": 0, "pm1": 0, "none": 0, "bracket_inverted": 0}
            for r in rs:
                for x in r["rounds"]:
                    if x["phase"] == ph:
                        for kk in rm:
                            rm[kk] += x["R_match"][kk]
            per["R_match"] = rm
            a[ph] = per
        agg[arm] = a
    return agg


def q4(runs):
    """Per-request exposure vs outcome; arm comparisons are descriptive (exposure is endogenous)."""
    out = {}
    for arm in HE0_JOBS:
        rq = [q for r in runs if r["arm"] == arm for q in r["_req"]]
        o = {}
        for ph in ("LO", "HI"):
            x = [q for q in rq if q["phase"] == ph]
            co_tok = [v for q in x for v in q["_tok_co"]]
            do_tok = [v for q in x for v in q["_tok_do"]]
            bins = {"exp<0.1": [q for q in x if q["decode_exposure"] is not None and q["decode_exposure"] < 0.1],
                    "0.1<=exp<0.3": [q for q in x if q["decode_exposure"] is not None and 0.1 <= q["decode_exposure"] < 0.3],
                    "exp>=0.3": [q for q in x if q["decode_exposure"] is not None and q["decode_exposure"] >= 0.3]}
            o[ph] = {
                "n_req": len(x),
                "pass_rate": statistics.fmean(q["pass"] for q in x),
                "pass_ttft_rate": statistics.fmean(q["pass_ttft"] for q in x),
                "pass_itl_rate": statistics.fmean(q["pass_itl"] for q in x),
                "ttft_mean_ms": statistics.fmean(q["ttft_ms"] for q in x),
                "queue_mean_ms": statistics.fmean(q["queue_ms"] for q in x),
                "exec_mean_ms": statistics.fmean(q["exec_ms"] for q in x),
                "exec_mean_ms_prefill_cores": msd([q["exec_ms"] for q in x if q["prefill_cores_logged"]]),
                "exec_mean_ms_prefill_alone": msd([q["exec_ms"] for q in x if not q["prefill_cores_logged"]]),
                "frac_req_prefill_cores_logged": statistics.fmean(q["prefill_cores_logged"] for q in x),
                "decode_exposure_mean": statistics.fmean(q["decode_exposure"] for q in x if q["decode_exposure"] is not None),
                "token_itl_ms_cores": {p: A.percentile(co_tok, p / 100) for p in (50, 95, 99)},
                "token_itl_ms_decode_only": {p: A.percentile(do_tok, p / 100) for p in (50, 95, 99)},
                "n_tok_cores": len(co_tok), "n_tok_decode_only": len(do_tok),
                "by_exposure_bin": {b: {"n": len(v), "pass_itl_rate": statistics.fmean(q["pass_itl"] for q in v) if v else None,
                                        "itl_p95_median_ms": A.percentile([q["itl_p95_ms"] for q in v if q["itl_p95_ms"] is not None], 0.5) if v else None}
                                    for b, v in bins.items()},
            }
            # decode-only ITL at matched running batch size (Q3 consistency test)
            bsb = {}
            for q in x:
                for v, bs in zip(q["_tok_do"], q["_bs_do"]):
                    bsb.setdefault(min(48, (bs // 8) * 8), []).append(v)
            o[ph]["decode_only_itl_p50_by_bs8"] = {str(k): (A.percentile(v, 0.5), len(v)) for k, v in sorted(bsb.items())}
            bsc = {}
            for q in x:
                for v, bs in zip(q["_tok_co"], q["_bs_co"]):
                    bsc.setdefault(min(48, (bs // 8) * 8), []).append(v)
            o[ph]["cores_itl_p50_by_bs8"] = {str(k): (A.percentile(v, 0.5), len(v)) for k, v in sorted(bsc.items())}
        out[arm] = o
    return out


def goodput_table(runs):
    t = {}
    for arm in HE0_JOBS:
        rs = [r for r in runs if r["arm"] == arm]
        g = [goodput_rows(r) for r in rs]
        t[arm] = {ph: {"slo_goodput_req_s": msd([x[ph]["slo_goodput_req_s"] for x in g]),
                       "legacy_mean_itl_goodput_req_s": msd([x[ph]["legacy_mean_itl_goodput_req_s"] for x in g]),
                       "ttft_p50_ms": msd([x[ph]["ttft_p50_ms"] for x in g]),
                       "ttft_p95_ms": msd([x[ph]["ttft_p95_ms"] for x in g]),
                       "ttft_p99_ms": msd([x[ph]["ttft_p99_ms"] for x in g]),
                       "token_itl_p50_ms": msd([x[ph]["token_itl_p50_ms"] for x in g]),
                       "token_itl_p95_ms": msd([x[ph]["token_itl_p95_ms"] for x in g]),
                       "token_itl_p99_ms": msd([x[ph]["token_itl_p99_ms"] for x in g]),
                       "throughput_req_s": msd([x[ph]["throughput_req_s"] for x in g])}
                  for ph in ("LO", "HI", "COMBINED")}
        t[arm]["_per_run_combined_slo"] = [x["COMBINED"]["slo_goodput_req_s"] for x in g]
        t[arm]["_per_run_combined_legacy"] = [x["COMBINED"]["legacy_mean_itl_goodput_req_s"] for x in g]
        t[arm]["_out_reported_combined"] = [r["out_combined_reported"] for r in rs]
    return t


# --------------------------------------------------------------------------- self-test
def _synth(seed=7, n=60, rate=6.0, D_label=44, stall=None):
    """Tiny discrete-event FCFS engine with known ground-truth co-residency."""
    rng = random.Random(seed)
    A_rel = arrival_unit(n) / rate
    L = [rng.randint(20, 900) for _ in range(n)]
    out_len = [rng.randint(2, 60) for _ in range(n)]
    off = 1_000_000.25
    t = off
    i = 0
    queue_ptr = 0
    running: Dict[int, float] = {}  # req -> finish time
    batches, e_true, s_true = [], [], []
    first = [None] * n; fin = [None] * n; itls = [None] * n
    pf_iv, dec_reqs = [], []
    lat = [0.0] * n
    if stall:
        for j in range(stall[0], n):
            lat[j] = stall[1]
    while queue_ptr < n:
        arr = off + A_rel[queue_ptr] + lat[queue_ptr]
        t = max(t, arr)
        running = {r: f for r, f in running.items() if f > t}
        slots = MAX_RUNNING - len(running)
        if slots <= 0:
            t = min(running.values()); continue
        S = [j for j in range(queue_ptr, n) if off + A_rel[j] + lat[j] <= t][: min(slots, 8)]
        R = len(running)
        dur = 0.004 + 0.00004 * sum(L[j] for j in S) * (108 / (108 - D_label) if R else 1.0)
        s, e = t, t + dur
        batches.append((math.floor(s), S, R, 0)); s_true.append(s); e_true.append(e)
        pf_iv.append((s, e))
        for j in S:
            first[j] = e
            step = 0.02 + 0.0004 * len(running)
            itls[j] = [step] * (out_len[j] - 1)
            fin[j] = e + step * (out_len[j] - 1)
            running[j] = fin[j]
        queue_ptr += len(S)
        t = e + 0.001
    ttft = [first[j] - (off + A_rel[j] + lat[j]) for j in range(n)]
    duration = max(fin) - off
    D = clip(union([(first[j], fin[j]) for j in range(n)]), off, off + duration)
    P = clip(union(pf_iv), off, off + duration)
    truth = length(intersect(P, D)) / duration
    return batches, A_rel, ttft, itls, duration, truth, L


def self_test() -> int:
    ok = True
    tol = 0.02
    for case, stall in (("no-stall", None), ("client-stall 5s@req30", (30, 5.0))):
        batches, A_rel, ttft, itls, duration, truth, L = _synth(stall=stall)
        def frac(mut=""):
            A_use = arrival_unit(len(ttft), seed=(2 if mut == "arrival_seed" else 1)) / 6.0
            rec = reconstruct_round(batches, A_use, ttft, itls, duration, _mut=mut)
            return length(rec["CO_pt"]) / duration, rec["offset_width_s"], rec["offset_segments"]
        f0, w0, ns0 = frac("")
        ctrl = abs(f0 - truth) <= tol and w0 > -0.005 and ns0 <= (1 if stall is None else 3)
        print(f"[{case}] [control]  truth co-res={truth:.4f} recon={f0:.4f} min_width={w0:.4f}s segments={ns0} -> {'PASS' if ctrl else 'FAIL'}")
        ok &= ctrl
        for mut in ("intersect_is_union", "arrival_seed", "offset_plus_1s"):
            f1, w1, ns1 = frac(mut)
            caught = (abs(f1 - truth) > tol) or (w1 < -0.005) or (ns1 > (1 if stall is None else 3))
            print(f"[{case}] [mutant {mut:18s}] recon={f1:.4f} min_width={w1:.4f}s segments={ns1} -> {'KILLED (expected)' if caught else 'SURVIVED (BAD)'}")
            ok &= caught
    assert union([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert intersect([(0, 2), (3, 4)], [(1, 3.5)]) == [(1, 2), (3, 3.5)]
    assert minus([(0, 4)], [(1, 2), (3, 5)]) == [(0, 1), (2, 3)]
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- main
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default=str(HERE / "he0_realized_2026-10-01.json"))
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    runs = []
    for arm, jobs in HE0_JOBS.items():
        for j in jobs:
            runs.append(run_one(arm, j))
            print(f"ok {arm} {j}", file=sys.stderr)
    agg = aggregate(runs)
    agg_q = aggregate(runs, quality_only=True)
    q = q4(runs)
    gp = goodput_table(runs)
    inputs = {}
    for r in runs:
        for f in r["client_files"] + [r["server_log"]]:
            inputs[f] = sha(SRC / f)
    slim = []
    for r in runs:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        slim.append(rr)
    out = {"generated": "2026-10-01", "script": Path(__file__).name, "slo": {"ttft_ms": TTFT_SLO_MS, "itl_p95_ms": ITL_SLO_MS},
           "aggregate": agg, "aggregate_quality_rounds_only": agg_q, "q4": q, "goodput": gp, "runs": slim, "input_sha256": inputs}
    # d44-vs-arm unpaired bootstrap (arms have no natural rep pairing; analyze.py convention)
    gp_runs: Dict[str, list] = {}
    for r in runs:
        gp_runs.setdefault(r["arm"], []).append(goodput_rows(r))
    ci = {}
    for comp in ("d34", "bind_gate", "bind_nogate", "slo", "d24", "d16"):
        for ph in ("COMBINED", "HI"):
            for key in ("slo_goodput_req_s", "legacy_mean_itl_goodput_req_s"):
                ci[f"{comp}_vs_d44_{ph}_{key}"] = A.unpaired_bootstrap_ci(
                    [g[ph][key] for g in gp_runs["d44"]], [g[ph][key] for g in gp_runs[comp]])
    (HERE / "he0_headline_unpaired_ci_2026-10-01.json").write_text(json.dumps(ci, indent=1))
    Path(args.output).write_text(json.dumps(out, indent=1, default=str))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
