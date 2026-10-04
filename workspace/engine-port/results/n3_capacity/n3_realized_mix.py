#!/usr/bin/env python3
"""N3 realized-partition estimators, keyed by the (prefill_sms, decode_sms) PAIR, not by index.

WHY A NEW FILE.  `e2_sticky_prereg/e2_realized_mix.py` hard-codes `D44_STREAM_INDEX = 2`
(pdmux_homog5.yml).  The N3 config (pdmux_n3_A_agn.yml) puts (64,44) at index 4 and (0,108) at
index 5, so the E2 tool would silently count the wrong group.  E2 files are sealed (E2 Job
running); this file re-implements the SAME four estimators with the target selected by the
(prefill_sms, decode_sms) pair carried on every snapshot.

DESCRIPTIVE ONLY.  Nothing here enters the N3 capacity decision (n3_analyze.py scores client
TTFT/ITL only).  Every quantity is reported with its estimator name (clause 3).

A. Snapshot estimators -- E2 ADDENDUM A2 / E2C-8' conventions, LITERALLY:
   population   = `runtime_snapshot` records with phase != "startup", WHOLE FILE (warm-up
                  traffic included, E2C-16; no probe in N3, so ADDENDUM A3 probe_end = None)
   decode-busy  = `decode_running_batch_size` > 0   (ADDENDUM A2 literal)
   E-cnt        = busy snapshots at the target pair / busy snapshots
   E-time       = busy snapshots weighted by the gap to the NEXT non-startup snapshot (uncapped;
                  last snapshot dropped)                                    (rev2 verdict F1')
   E-iter       = adjacent pairs with decode_iterations delta > 0 and LEFT snapshot busy, weight
                  = delta, attributed to the LEFT snapshot
   E-qcond      = busy AND prefill_queue_depth > 0, uniform
   E-pact       = busy AND prefill_active_batch_size > 0 (construction identity guard, E2C-4)

B. PHASE_EVENTS estimators (new in N3; need PDMUX_PHASE_EVENTS=1; restricted to the bench window
   [first work event, last work event] inside the runner's monotonic marks, as E-1 §5-5):
   W-dec        = decode wall time (sum of t_sync - t_launch) per pair / total decode wall time
   W-idx        = wall-clock index occupancy, EVENT-HELD approximation: the pair carried by each
                  work event (prefill_span_start, decode_iteration@t_launch) is held until the
                  next work event; resolution = one loop iteration
   X-tok        = token-weighted exposure: sum(bs * [pair == target]) / sum(bs) over decode
                  iterations (each decode iteration emits bs tokens)
   CO-a         = co-residency definition (a): |PF n DEC| / |window|, PF = U[span_start,
                  span_end], DEC = U[t_launch, t_sync] (host-observed intervals)
   cap-bound    = fraction of prefill_span_start with running_bs + n_new >= cap and queue_len > 0
   pin          = fraction of co-resident decode iterations (prefill_in_flight and not
                  kernel_done_wait) at the target pair (Stage-0 type check: target vs realised)

C. X-req, per-request exposure (RECONSTRUCTION-BASED -- clause 4):
   The bench client records no absolute times, so request i's send time is rebuilt as
   a_i = t0 + sum_{j<i} Exp(1/rate) from np.random.seed(seed) (bench_serving.py:948, the same
   stream λ0 `ebar` replicates) with t0 = the SECOND prefill_span_start in the window (the first
   is bench_serving's own warm-up request, followed by a 1 s sleep, bench_serving.py:1284).  Its
   decode token times are a_i + ttft_i + cumsum(itl_i); each is matched to the latest decode
   iteration with t_sync <= token time.  exposure_i = share of request i's decode tokens matched
   to a target-pair iteration.  Anchor quality is reported (gap check + first-token residual);
   values are "재구성 기반 기술값", never a measurement of per-request SM.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from pathlib import Path

TARGET_PAIR = (64, 44)
# R3 (rules verdict rev1 §4): the non-target divisions of pdmux_n3_A_agn.yml (= n3_plan.NONTARGET_DIVISIONS)
NONTARGET_DIVISIONS = ((92, 16), (84, 24), (74, 34))
WORK_EVENTS = ("prefill_span_start", "prefill_span_end", "decode_iteration")
E_PACT_ABORT_BELOW = 95.0
ANCHOR_MIN_GAP_S = 1.0          # bench_serving sleeps 1.0 s between its warm-up and request 0

HERE = Path(__file__).resolve().parent
ENG = HERE.parents[1]
LAM0V_ARCHIVE = ENG / "results/r2_eval/lambda0_vessl/lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945"
E2_TOOL = ENG / "results/r2_eval/e2_sticky_prereg/e2_realized_mix.py"
# positive-control cells = the E2 tool's own selftest cell list (its EXPECTED keys)
SELFTEST_CELLS = ("a_r0", "a_r2", "a_r4", "a_r4_s2", "b_r0", "b_r3", "b_r3_s2")
# EXTERNAL expectation (gate #9): VERDICT_result_lambda0_vessl_2026-10-04.md §1 / C9, 2 decimals
EXTERNAL_PCT = {"a_r4": (8.66, 13.89, 8.76, 18.33), "b_r3": (90.48, 97.38, 91.68, 100.00)}


# ------------------------------------------------------------------------------- io
def read_events(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue          # a SIGKILLed writer may leave one torn last line
    return out


def read_marks(path):
    """Runner marks file: lines `bench_start <mono>` / `bench_end <mono>` (CLOCK_MONOTONIC)."""
    m = {}
    p = Path(path) if path else None
    if p and p.exists():
        for ln in p.read_text().splitlines():
            f = ln.split()
            if len(f) >= 2:
                try:
                    m[f[0]] = float(f[1])
                except ValueError:
                    pass
    return m


# ------------------------------------------------------------------------------- A. snapshots
def _is_snapshot(rec):
    return rec.get("event") == "runtime_snapshot" and rec.get("phase") != "startup"


def _is_decode_busy(rec):
    return (rec.get("decode_running_batch_size") or 0) > 0


def _pair(rec):
    return (rec.get("prefill_sms"), rec.get("decode_sms"))


def snapshot_estimators(events, target=TARGET_PAIR):
    snaps = [r for r in events if _is_snapshot(r)]
    busy = [s for s in snaps if _is_decode_busy(s)]
    out = {"snapshots": len(snaps), "busy_n": len(busy), "target_pair": list(target),
           "convention": "E2 ADDENDUM A2 decode-busy literal; whole file, phase != startup (E2C-16); pair-keyed"}
    if not busy:
        out["empty"] = True
        return out

    def _frac(sample):
        hit = sum(1 for s in sample if _pair(s) == target)
        return {"num": hit, "den": len(sample), "pct": (100.0 * hit / len(sample)) if sample else float("nan")}

    out["e_cnt"] = _frac(busy)
    num_s = den_s = 0.0
    neg = 0
    for i in range(len(snaps) - 1):
        gap = snaps[i + 1]["timestamp_monotonic_s"] - snaps[i]["timestamp_monotonic_s"]
        if gap < 0:
            neg += 1
        if not _is_decode_busy(snaps[i]):
            continue
        den_s += gap
        if _pair(snaps[i]) == target:
            num_s += gap
    out["e_time"] = {"num_s": num_s, "den_s": den_s, "pct": (100.0 * num_s / den_s) if den_s else float("nan")}
    out["negative_gaps"] = neg
    num_i = den_i = 0
    for left, right in zip(snaps, snaps[1:]):
        delta = (right.get("decode_iterations") or 0) - (left.get("decode_iterations") or 0)
        if delta <= 0 or not _is_decode_busy(left):
            continue
        den_i += delta
        if _pair(left) == target:
            num_i += delta
    out["e_iter"] = {"num": num_i, "den": den_i, "pct": (100.0 * num_i / den_i) if den_i else float("nan")}
    out["e_qcond"] = _frac([s for s in busy if (s.get("prefill_queue_depth") or 0) > 0])
    out["e_pact"] = _frac([s for s in busy if (s.get("prefill_active_batch_size") or 0) > 0])
    hist = {}
    for s in busy:
        k = f"{s.get('prefill_sms')},{s.get('decode_sms')}"
        hist[k] = hist.get(k, 0) + 1
    out["pair_hist_busy"] = hist
    out["dropped_events_max"] = max([int(r.get("dropped_events", 0) or 0) for r in events] or [0])
    return out


# ------------------------------------------------------------------------------- interval helpers
def union(iv):
    out = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def measure(iv):
    return sum(b - a for a, b in union(iv))


def intersect(x, y):
    x, y = union(x), union(y)
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if min(b, hi) > max(a, lo)]


# ------------------------------------------------------------------------------- B. phase events
def _t(e):
    return float(e["t_launch"]) if e.get("event") == "decode_iteration" else float(e["timestamp_monotonic_s"])


def bench_window(events, marks):
    """[first work event, last work event] inside the runner's monotonic bench marks; without
    marks, the whole file's work events (recorded as `window_source`)."""
    work = sorted(_t(e) for e in events if e.get("event") in WORK_EVENTS)
    if not work:
        return None, "no_work_events"
    if "bench_start" in marks and "bench_end" in marks:
        inside = [t for t in work if marks["bench_start"] <= t <= marks["bench_end"]]
        if not inside:
            return None, "no_work_events_in_marks"
        return (inside[0], inside[-1]), "marks"
    return (work[0], work[-1]), "whole_file"


def phase_estimators(events, marks=None, target=TARGET_PAIR, cap=None):
    marks = marks or {}
    win, src = bench_window(events, marks)
    out = {"window_source": src, "target_pair": list(target)}
    if win is None:
        out["empty"] = True
        return out
    lo, hi = win
    spans, open_s, starts = [], None, []
    dec = []
    for e in events:
        k = e.get("event")
        if k == "prefill_span_start":
            open_s = e
            if lo <= e["timestamp_monotonic_s"] <= hi:
                starts.append(e)
        elif k == "prefill_span_end" and open_s is not None:
            spans.append((open_s["timestamp_monotonic_s"], e["timestamp_monotonic_s"]))
            open_s = None
        elif k == "decode_iteration":
            if lo <= float(e["t_launch"]) <= hi:
                dec.append(e)
    wlen = hi - lo
    out["window_s"] = wlen
    out["n_decode_iterations"] = len(dec)
    out["n_prefill_spans"] = len(starts)
    # W-dec
    share, tot = {}, 0.0
    for e in dec:
        d = float(e["t_sync"]) - float(e["t_launch"])
        key = f"{e.get('prefill_sms')},{e.get('decode_sms')}"
        share[key] = share.get(key, 0.0) + d
        tot += d
    out["w_dec"] = {"share": {k: v / tot for k, v in sorted(share.items())} if tot else {},
                    "target_pct": 100.0 * share.get(f"{target[0]},{target[1]}", 0.0) / tot if tot else float("nan"),
                    "decode_wall_s": tot}
    # W-idx (event-held)
    marks_ev = sorted([(float(e["timestamp_monotonic_s"]), (e.get("prefill_sms"), e.get("decode_sms")))
                       for e in starts] +
                      [(float(e["t_launch"]), (e.get("prefill_sms"), e.get("decode_sms"))) for e in dec])
    held = {}
    for (t, p), (t2, _) in zip(marks_ev, marks_ev[1:] + [(hi, None)]):
        key = f"{p[0]},{p[1]}"
        held[key] = held.get(key, 0.0) + max(0.0, min(t2, hi) - t)
    htot = sum(held.values())
    out["w_idx"] = {"share": {k: v / htot for k, v in sorted(held.items())} if htot else {},
                    "target_pct": 100.0 * held.get(f"{target[0]},{target[1]}", 0.0) / htot if htot else float("nan"),
                    "convention": "event-held pair of the last work event; loop-iteration resolution"}
    # X-tok
    bs_all = sum(int(e.get("bs", 0)) for e in dec)
    bs_t = sum(int(e.get("bs", 0)) for e in dec if (e.get("prefill_sms"), e.get("decode_sms")) == target)
    out["x_tok"] = {"num": bs_t, "den": bs_all, "pct": (100.0 * bs_t / bs_all) if bs_all else float("nan")}
    # CO-a
    pf = clip(spans, lo, hi)
    dv = clip([(float(e["t_launch"]), float(e["t_sync"])) for e in dec], lo, hi)
    co = measure(intersect(pf, dv))
    dm = measure(dv)
    out["coresidency_a_wall"] = co / wlen if wlen else float("nan")
    out["coresidency_a_decode"] = co / dm if dm else float("nan")
    out["decode_active_frac"] = dm / wlen if wlen else float("nan")
    # cap-bound admissions
    if cap is None:
        cap = max([int(s.get("cap", 0) or 0) for s in starts] or [0])
    bound = [int(s.get("running_bs", 0)) + int(s.get("n_new", 0)) >= cap and int(s.get("queue_len", 0)) > 0
             for s in starts]
    out["cap_bound_admission_frac"] = (sum(bound) / len(bound)) if bound else float("nan")
    # pin check on co-resident decode iterations
    co_it = [e for e in dec if e.get("prefill_in_flight") and not e.get("prefill_kernel_done_wait")]
    out["pin"] = {"n": len(co_it),
                  "target_frac": (sum(1 for e in co_it if (e.get("prefill_sms"), e.get("decode_sms")) == target)
                                  / len(co_it)) if co_it else float("nan")}
    out["dropped_events_max"] = max([int(e.get("dropped_events", 0) or 0) for e in events] or [0])
    return out


# ------------------------------------------------------------------------------- C. X-req
def arrival_offsets(seed, rate, n):
    """bench_serving.py:913-950 + :1706: np.random.seed(seed); one Exp(1/rate) AFTER each request."""
    import numpy as np
    np.random.seed(int(seed))
    off, t = [0.0], 0.0
    for _ in range(int(n) - 1):
        t += float(np.random.exponential(1.0 / float(rate)))
        off.append(t)
    return off


def per_request_exposure(events, bench_rec, seed, rate, marks=None, target=TARGET_PAIR):
    marks = marks or {}
    win, src = bench_window(events, marks)
    out = {"convention": "reconstruction-based (arrival RNG replay + client ttft/itl); descriptive",
           "window_source": src}
    ttfts = [float(x) for x in (bench_rec.get("ttfts") or [])]
    itls = bench_rec.get("itls") or []
    if win is None or not ttfts or rate in (None, "inf") or not math.isfinite(float(rate)):
        out["status"] = "NOT_APPLICABLE"
        return out
    lo, hi = win
    starts = sorted(float(e["timestamp_monotonic_s"]) for e in events
                    if e.get("event") == "prefill_span_start" and lo <= e["timestamp_monotonic_s"] <= hi)
    ends = sorted(float(e["timestamp_monotonic_s"]) for e in events if e.get("event") == "prefill_span_end")
    dec = sorted(((float(e["t_sync"]), (e.get("prefill_sms"), e.get("decode_sms")))
                  for e in events if e.get("event") == "decode_iteration"), key=lambda x: x[0])
    if len(starts) < 2 or not dec:
        out["status"] = "UNANCHORED"
        return out
    gap = starts[1] - starts[0]
    out["anchor_gap_s"] = gap
    if gap < ANCHOR_MIN_GAP_S:
        out["status"] = "UNANCHORED"
        return out
    t0 = starts[1]
    offs = arrival_offsets(seed, rate, len(ttfts))
    syncs = [d[0] for d in dec]
    exp, resid, n_tok, n_tok_t = [], [], 0, 0
    for i, tt in enumerate(ttfts):
        ft = t0 + offs[i] + tt
        j = bisect.bisect_right(ends, ft) - 1
        if j >= 0:
            resid.append(1000.0 * (ft - ends[j]))
        cur, hit, tot = ft, 0, 0
        for d in (itls[i] if i < len(itls) else []):
            cur += float(d)
            k = bisect.bisect_right(syncs, cur) - 1
            if k < 0:
                continue
            tot += 1
            if dec[k][1] == target:
                hit += 1
        n_tok += tot
        n_tok_t += hit
        if tot:
            exp.append(hit / tot)
    if not exp:
        out["status"] = "NO_DECODE_TOKENS"
        return out
    se = sorted(exp)

    def q(p):
        return se[min(len(se) - 1, int(round(p * (len(se) - 1))))]

    resid.sort()
    out.update(status="OK", n_requests=len(exp), mean=sum(exp) / len(exp), p10=q(0.10), p50=q(0.50),
               p90=q(0.90), frac_ge_half=sum(1 for x in exp if x >= 0.5) / len(exp),
               token_weighted_pct=100.0 * n_tok_t / n_tok if n_tok else float("nan"),
               first_token_residual_ms={"p50": resid[len(resid) // 2] if resid else None,
                                        "p95": resid[min(len(resid) - 1, int(0.95 * (len(resid) - 1)))] if resid else None})
    return out


def arm_identity(events, forbidden=NONTARGET_DIVISIONS):
    """R3 arm-identity condition (a DECISION-GATING validity check, not a descriptive): zero
    decode iterations (pair = post-switch index, emitted at decode sync) and zero decode-busy
    snapshots on a non-target division.  `decode_iterations_seen` is 0 when PHASE_EVENTS is off
    or was disabled mid-boot; then only the snapshot leg is evidence (recorded)."""
    fb = {tuple(p) for p in forbidden}
    dec = [e for e in events if e.get("event") == "decode_iteration"]
    busy = [e for e in events if _is_snapshot(e) and _is_decode_busy(e)]
    n_dec = sum(1 for e in dec if (e.get("prefill_sms"), e.get("decode_sms")) in fb)
    n_snap = sum(1 for e in busy if _pair(e) in fb)
    return {"forbidden_pairs": [list(p) for p in sorted(fb)], "decode_iterations_seen": len(dec),
            "busy_snapshots_seen": len(busy), "decode_iteration_on_nontarget": n_dec,
            "busy_snapshot_on_nontarget": n_snap, "ok": n_dec == 0 and n_snap == 0}


def compute(tel, bench=None, seed=None, rate=None, marks_path=None, target=TARGET_PAIR, cap=None):
    ev = read_events(tel)
    marks = read_marks(marks_path)
    res = {"snapshot": snapshot_estimators(ev, target), "phase": phase_estimators(ev, marks, target, cap),
           "arm_identity": arm_identity(ev)}
    if bench and Path(bench).exists() and seed is not None and rate not in (None, "inf"):
        rec = json.loads(Path(bench).read_text().splitlines()[0])
        res["x_req"] = per_request_exposure(ev, rec, seed, float(rate), marks, target)
    return res


# ------------------------------------------------------------------------------- selftest
def _load_e2_tool():
    import importlib.util
    sys.dont_write_bytecode = True          # E2 tree is sealed: never drop a __pycache__ there
    spec = importlib.util.spec_from_file_location("e2_realized_mix_ro", E2_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp_e2(e2, mine):
    keys = [("busy_n", e2["busy_n"], mine["busy_n"]),
            ("e_cnt.num", e2["e_cnt"]["num"], mine["e_cnt"]["num"]),
            ("e_cnt.den", e2["e_cnt"]["den"], mine["e_cnt"]["den"]),
            ("e_time.num_s", e2["e_time"]["num_s"], mine["e_time"]["num_s"]),
            ("e_time.den_s", e2["e_time"]["den_s"], mine["e_time"]["den_s"]),
            ("e_iter.num", e2["e_iter"]["num"], mine["e_iter"]["num"]),
            ("e_iter.den", e2["e_iter"]["den"], mine["e_iter"]["den"]),
            ("e_qcond.num", e2["e_qcond"]["num"], mine["e_qcond"]["num"]),
            ("e_qcond.den", e2["e_qcond"]["den"], mine["e_qcond"]["den"]),
            ("e_pact.pct", e2["e_pact"]["pct"], mine["e_pact"]["pct"])]
    return [f"{k}: e2={a!r} n3={b!r}" for k, a, b in keys if a != b]


def selftest_archive(archive=LAM0V_ARCHIVE):
    """POSITIVE CONTROL: on the λ0 VESSL archive (homog5: (64,44) = index 2) the pair-keyed
    estimators must equal the sealed index-keyed E2 tool on all 7 cells (bit-for-bit), and two
    cells must equal the EXTERNAL result-audit values.  NEGATIVE CONTROLS: (i) target (92,16)
    must NOT reproduce; (ii) an index remap 2->4, 4->5 (the A-agn layout) leaves n3 unchanged
    while the E2 tool's E-cnt moves -- the reason this file exists."""
    import tempfile
    missing = [c for c in SELFTEST_CELLS if not (Path(archive) / f"tel_{c}.jsonl").exists()]
    if missing:
        print(f"SELFTEST_ARCHIVE_MISSING {archive} lacks {missing} (git-ignored; not an estimator failure)")
        return 2
    e2 = _load_e2_tool()
    fails, wrong_ok = [], 0
    for c in SELFTEST_CELLS:
        p = Path(archive) / f"tel_{c}.jsonl"
        ref = e2.compute(str(p))
        ev = read_events(p)
        mine = snapshot_estimators(ev, (64, 44))
        d = _cmp_e2(ref, mine)
        fails += [f"{c}.{x}" for x in d]
        if not _cmp_e2(ref, snapshot_estimators(ev, (92, 16))):
            wrong_ok += 1
        if not arm_identity(ev, ((92, 16), (16, 92)))["ok"]:
            fails.append(f"{c}: λ0 archive has decode-busy snapshots on a homog5 non-target division (R3 prior)")
        if c in EXTERNAL_PCT:
            got = tuple(round(mine[k]["pct"], 2) for k in ("e_cnt", "e_time", "e_iter", "e_qcond"))
            if got != EXTERNAL_PCT[c]:
                fails.append(f"{c}: external {EXTERNAL_PCT[c]} != {got}")
        print(f"  {c:8s} busy={mine['busy_n']:5d} E-cnt={mine['e_cnt']['pct']:6.2f} E-time={mine['e_time']['pct']:6.2f} "
              f"E-iter={mine['e_iter']['pct']:6.2f} E-qcond={mine['e_qcond']['pct']:6.2f}  vs E2 tool: "
              f"{'IDENTICAL' if not d else 'DIFF'}")
    if wrong_ok:
        fails.append(f"negative control: target (92,16) reproduced the E2 table on {wrong_ok} cell(s)")
    # remap control on a_r4
    src = Path(archive) / "tel_a_r4.jsonl"
    remap = {2: 4, 4: 5, 3: 6}
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "tel_remap.jsonl"
        with open(src) as fi, open(dst, "w") as fo:
            for ln in fi:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "stream_index" in r and r["stream_index"] in remap:
                    r["stream_index"] = remap[r["stream_index"]]
                fo.write(json.dumps(r) + "\n")
        a = snapshot_estimators(read_events(src))
        b = snapshot_estimators(read_events(dst))
        if (a["e_cnt"], a["e_time"]["pct"]) != (b["e_cnt"], b["e_time"]["pct"]):
            fails.append("remap control: pair-keyed estimator moved under an index remap")
        e2b = e2.compute(str(dst))
        if e2b["e_cnt"]["num"] == a["e_cnt"]["num"]:
            fails.append("remap control vacuous: E2 tool did not move under the remap")
    if fails:
        print("SELFTEST_FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print(f"SELFTEST_OK 7/7 cells identical to the E2 tool (pair (64,44) == index 2 on homog5); "
          f"external a_r4/b_r3 reproduced; (92,16) control fails 7/7; index-remap control: n3 invariant, E2 moves")
    return 0


def selftest_synthetic():
    """Phase-event estimators and X-req on a constructed timeline with known answers."""
    import tempfile
    fails = []
    ev = []
    # window 0..10 s; prefill spans [1,2] at (64,44) start, [5,6]; decode iterations 0.5 s each
    ev.append({"event": "prefill_span_start", "timestamp_monotonic_s": 0.0, "prefill_sms": 108, "decode_sms": 0,
               "running_bs": 0, "n_new": 1, "queue_len": 0, "cap": 48})
    ev.append({"event": "prefill_span_end", "timestamp_monotonic_s": 0.2})
    ev.append({"event": "prefill_span_start", "timestamp_monotonic_s": 1.0, "prefill_sms": 64, "decode_sms": 44,
               "running_bs": 47, "n_new": 1, "queue_len": 3, "cap": 48})
    ev.append({"event": "prefill_span_end", "timestamp_monotonic_s": 2.0})
    t = 0.5
    while t < 9.5:
        inflight = 1.0 <= t < 2.0
        pair = (64, 44) if inflight else (0, 108)
        ev.append({"event": "decode_iteration", "t_launch": t, "t_sync": t + 0.4, "bs": 2,
                   "prefill_sms": pair[0], "decode_sms": pair[1], "prefill_in_flight": inflight,
                   "prefill_kernel_done_wait": False, "timestamp_monotonic_s": t + 0.4})
        t += 0.5
    ph = phase_estimators(ev, {}, (64, 44))
    # decode iterations launched at 1.0 and 1.5 are at (64,44): 2 of 18
    if abs(ph["w_dec"]["target_pct"] - 100.0 * 2 / 18) > 1e-9:
        fails.append(f"W-dec {ph['w_dec']['target_pct']}")
    if abs(ph["x_tok"]["pct"] - 100.0 * 2 / 18) > 1e-9:
        fails.append(f"X-tok {ph['x_tok']['pct']}")
    # CO-a: PF=[0,0.2]u[1,2]; DEC within [1,2] = [1,1.4]u[1.5,1.9] = 0.8 s; window = first work
    # event 0.0 .. last work event (last t_launch 9.0) = 9.0 s
    if abs(ph["coresidency_a_wall"] - 0.8 / 9.0) > 1e-9:
        fails.append(f"CO-a {ph['coresidency_a_wall']}")
    if abs(ph["cap_bound_admission_frac"] - 0.5) > 1e-12:
        fails.append(f"cap-bound {ph['cap_bound_admission_frac']}")
    if ph["pin"]["target_frac"] != 1.0:
        fails.append(f"pin {ph['pin']}")
    # X-req: one bench warm-up span at 100.0, request 0 span at 102.0; rate 1/s, seed 7
    offs = arrival_offsets(7, 1.0, 3)
    ev2 = [{"event": "prefill_span_start", "timestamp_monotonic_s": 100.0, "prefill_sms": 108, "decode_sms": 0},
           {"event": "prefill_span_end", "timestamp_monotonic_s": 100.1}]
    t0 = 102.0
    for i, o in enumerate(offs):
        ev2.append({"event": "prefill_span_start", "timestamp_monotonic_s": t0 + o, "prefill_sms": 108, "decode_sms": 0})
        ev2.append({"event": "prefill_span_end", "timestamp_monotonic_s": t0 + o + 0.05})
    # decode iterations every 10 ms from 102 to 106; (64,44) for t_sync in [103, 104), else (0,108)
    k = 0
    while 102.0 + 0.01 * k < 106.0:
        ts = 102.0 + 0.01 * k + 0.009
        pair = (64, 44) if 103.0 <= ts < 104.0 else (0, 108)
        ev2.append({"event": "decode_iteration", "t_launch": ts - 0.009, "t_sync": ts, "bs": 1,
                    "prefill_sms": pair[0], "decode_sms": pair[1], "prefill_in_flight": False,
                    "prefill_kernel_done_wait": False, "timestamp_monotonic_s": ts})
        k += 1
    bench = {"ttfts": [0.06, 0.06, 0.06], "itls": [[0.0105] * 20, [0.0105] * 20, [0.0105] * 20]}
    xr = per_request_exposure(ev2, bench, 7, 1.0, {}, (64, 44))
    want = []
    for o in offs:
        ft = t0 + o + 0.06
        times = [ft + 0.0105 * (m + 1) for m in range(20)]
        want.append(sum(1 for x in times if 103.0 <= x < 104.0 + 0.0105) / 20.0)
    if xr.get("status") != "OK":
        fails.append(f"X-req status {xr}")
    elif abs(xr["mean"] - sum(want) / 3) > 0.051:
        fails.append(f"X-req mean {xr['mean']} vs constructed {sum(want) / 3}")
    # anchor control: remove the bench warm-up span -> gap < 1 s or wrong anchor must be flagged
    ev3 = [e for e in ev2 if not (e.get("timestamp_monotonic_s") in (100.0, 100.1))]
    xr3 = per_request_exposure(ev3, bench, 7, 1.0, {}, (64, 44))
    if xr3.get("status") == "OK" and abs(xr3["mean"] - xr.get("mean", -1)) < 1e-12:
        fails.append("anchor control: a missing warm-up span went unnoticed")
    # R3: clean timeline passes; one injected decode iteration / busy snapshot on (84,24) fails
    if not arm_identity(ev)["ok"]:
        fails.append("arm identity: clean synthetic timeline flagged")
    bad = ev + [dict(ev[-1], prefill_sms=84, decode_sms=24)]
    if arm_identity(bad)["ok"] or arm_identity(bad)["decode_iteration_on_nontarget"] != 1:
        fails.append("arm identity: injected (84,24) decode iteration not caught")
    snap = {"event": "runtime_snapshot", "phase": "benchmark", "decode_running_batch_size": 3,
            "prefill_sms": 74, "decode_sms": 34, "timestamp_monotonic_s": 5.0}
    if arm_identity(ev + [snap])["ok"]:
        fails.append("arm identity: injected (74,34) busy snapshot not caught")
    if not arm_identity(ev + [dict(snap, decode_running_batch_size=0)])["ok"]:
        fails.append("arm identity: an IDLE snapshot on a division must not count")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in ev) + "\n{torn")
        if len(read_events(p)) != len(ev):
            fails.append("torn last line not tolerated")
    if fails:
        print("SYNTHETIC_SELFTEST_FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print("SYNTHETIC_SELFTEST_OK (W-dec, X-tok, CO-a, cap-bound, pin, X-req vs constructed timeline; anchor control; R3 arm identity + 3 injection controls)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("telemetry", nargs="?")
    ap.add_argument("--selftest", action="store_true", help="synthetic always; archive part if present")
    ap.add_argument("--require-archive", action="store_true", help="missing archive = failure (local submission gate)")
    ap.add_argument("--archive", default=str(LAM0V_ARCHIVE))
    ap.add_argument("--bench"), ap.add_argument("--seed", type=int), ap.add_argument("--rate")
    ap.add_argument("--marks"), ap.add_argument("--cell"), ap.add_argument("--cap", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        rc = selftest_synthetic()
        arc = selftest_archive(Path(a.archive))
        if arc == 2 and not a.require_archive:
            arc = 0
        return rc or arc
    if not a.telemetry:
        ap.error("telemetry path required unless --selftest")
    res = compute(a.telemetry, a.bench, a.seed, a.rate, a.marks, TARGET_PAIR, a.cap)
    res["cell"] = a.cell
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    s, p = res["snapshot"], res["phase"]
    print(json.dumps({"cell": a.cell, "busy_n": s.get("busy_n"),
                      **{k: s.get(k, {}).get("pct") for k in ("e_cnt", "e_time", "e_iter", "e_qcond")},
                      "w_dec": p.get("w_dec", {}).get("target_pct"), "x_tok": p.get("x_tok", {}).get("pct"),
                      "co_a": p.get("coresidency_a_wall"), "x_req": res.get("x_req", {}).get("mean")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
