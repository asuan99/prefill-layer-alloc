#!/usr/bin/env python3
"""HE0 reconstruction self-test v2 (GPU 0, 2026-10-02, result-analyst).

Why v2 (AUDIT_RESULT_VERDICT_2026-10-02.md section 1 #10, section 4 item 2): the v1 self-test in
``he0_realized.py`` is a partial identity -- its synthetic engine generates the running count with the
SAME rule the reconstruction inverts, has zero client receipt delay and few-ms prefills, so the point
estimate ``pt`` and the earliest-start convention give the same number (0.1878 vs 0.1878) and the
mutant "s_pt := earliest" SURVIVES.

This file does NOT modify ``he0_realized.py``.  It imports it and tests ``H.reconstruct_round``
unchanged.  Mutants are built by single-token source substitution of ``reconstruct_round`` (each
substitution asserted to match exactly once); the unsubstituted copy is checked to reproduce
``H.reconstruct_round`` bit-for-bit, so the mutation harness itself is not a re-implementation.

=========================================================================================
PRE-REGISTERED (fixed before the first run of this file; do not edit after results exist)
=========================================================================================
Synthetic engine (``synth_engine``) -- generation rules deliberately DIFFERENT from the reconstruction:
  * server clock, iteration-level PD-mux loop: at each decode-iteration boundary
      (1) requests whose last token was produced at the previous iteration end LEAVE the running set,
      (2) requests whose prefill finished (server time e_srv <= boundary) JOIN the running set,
      (3) if no prefill is in flight and the FCFS queue is non-empty and slots (cap 48) remain, a prefill
          batch is admitted at the boundary (token budget 8192); logged (floor(s), ids, R=len(running), Q).
    => the logged #running-req is the SERVER running set at admission (join at boundary after e_srv,
       leave at the boundary after the last token); the reconstruction instead uses CLIENT receipt times
       [first receipt, last receipt).  Not an identity.
  * prefill duration = (0.03 + 1e-4 * tokens) * (108/(108-D) if decode running else 1)
    -> ~100 ms scale (real HE0 d16-d44 in-flight 96-110 ms).
  * decode iteration = (0.0095 + 0.00022*bs) * f(D, bs) [f only while a prefill is in flight]
    + STALL (0.045 s) on the iteration at which a prefill is admitted.  f interpolates linearly in bs
    between (bs 5, bs 45): D44 1.05 -> 1.6, D16 2.8 -> 3.75.  These shapes are copied from the real
    token timelines (856890 d44, 859008 d16: one ~59/75 ms iteration at admission, then slowed iterations).
  * client receipt delay (detok + HTTP): receipt = g + c(t) + |N(0, 0.4 ms)|, c(t) = 12 ms + 0.4 ms*bs
    + AR(1) noise (phi 0.9 per 0.1 s, sigma 3 ms), clipped >= 3 ms -> ~12-35 ms ("delay=real");
    "delay=0" variant for contrast.  Per-request receipts forced monotone.
  * client send lateness 0.4 ms/request drift; optional client SEND stall (3 s step at request 100).
  * real lengths: input_lens and (len(itls)+1) output lengths of round 0 of job 856890 (LO and HI files).
  * truth (server clock) co-residency = |U[s_k, e_srv_k] ∩ U[decode iterations] ∩ window| / duration.

Scenarios: phase {LO rate 3, HI rate 12} x D {44, 16} x delay {0, real} x send-stall {none, 3 s@100}
           = 16, synth seed 1 (delay noise / jitter).

Criteria:
  T1 (aggregate; same tolerance as v1): |CO_pt - truth| <= 0.02.
  T2 (interval placement; new, frame-invariant): median over batches of
      (e_rec_k - s_pt_k) - (e_srv_k - s_true_k)  within +-0.015 s.
  Scenario usable iff the CONTROL passes T1 and T2.  Mutant KILLED in a scenario iff it fails T1 or T2
  there.  Overall: control must pass in >= 1 scenario of each phase; a mutant is "KILLED" overall iff
  killed in every usable scenario, "PARTIAL" if killed in some, "SURVIVED" if killed in none.  If the
  control fails a scenario, that scenario is reported as "control FAIL (cannot adjudicate)" together with
  its bias -- it is NOT silently dropped.
Mutants:
  v1-carried: intersect_is_union, arrival_seed, offset_plus_1s (via the existing ``_mut`` hook)
  (b) s_pt := earliest   : ``s_pt.append(pt)`` -> ``s_pt.append(lo)``
  (c) e + 30 ms          : ``e.append(min(fs))`` -> ``e.append(min(fs) + 0.030)``
Also reported (descriptive, no criterion): per-scenario convention fractions
  lo(+1 iter) / pt / earliest / latest  vs truth.
=========================================================================================

Limits stated up front: the synthetic engine is a model; passing here shows the reconstruction recovers a
world shaped like this model, not that the real engine behaves like it.  Real alignment of the
reconstruction is validated only to ~0.3-1 s (audit #11); nothing here licenses 100 ms statements about
real runs.

Usage: python3 he0_selftest_v2.py [--output he0_selftest_v2_2026-10-02.json]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import he0_realized as H  # noqa: E402
from he0_realized import SRC, clip, intersect, length, union  # noqa: E402

CAP = 48
TOKEN_BUDGET = 8192
STALL_S = 0.045
F_TABLE = {44: (1.05, 1.6), 16: (2.8, 3.75)}
T1_TOL = 0.02
T2_TOL_S = 0.015
LENGTH_JOB = 856890

# ----------------------------------------------------------------------------------- mutation harness
_RR_SRC = inspect.getsource(H.reconstruct_round)


def make_variant(subs):
    src = _RR_SRC
    for old, new in subs:
        assert src.count(old) == 1, ("substitution must match exactly once", old, src.count(old))
        src = src.replace(old, new)
    ns = dict(H.__dict__)
    exec(compile(src, "<reconstruct_round variant>", "exec"), ns)
    return ns["reconstruct_round"]


MUTANTS = {
    "b_s_pt_is_earliest": [("s_pt.append(pt)", "s_pt.append(lo)")],
    "c_e_plus_30ms": [("e.append(min(fs))", "e.append(min(fs) + 0.030)")],
}


# ----------------------------------------------------------------------------------- synthetic engine
def real_lengths(phase: str):
    f = list(SRC.glob(f"sgptv{'Lo' if phase == 'LO' else 'Hi'}_*_{LENGTH_JOB}.jsonl"))[0]
    o = json.loads([x for x in f.open() if x.strip()][0])
    return list(o["input_lens"]), [len(x) + 1 for x in o["itls"]], float(o["request_rate"])


def _f(D, bs):
    a, b = F_TABLE[D]
    w = min(1.0, max(0.0, (bs - 5) / 40.0))
    return a + (b - a) * w


def synth_engine(phase="HI", D=44, delay="real", send_stall=None, recv_stall=None, stall_iter=True, seed=1,
                 server_stall=None):
    """server_stall (Task C only; default off): (t0 relative to first send, duration, "wide"|"sched").
    The scheduler is frozen in [t0, t0+dur): no iteration starts, in-flight prefill end is pushed back.
    "wide": the HTTP process is silent too (requests sent during the stall are received/logged at its end);
    "sched": HTTP keeps receiving/logging.  Also emits synthetic server log lines (Decode every 40 iterations,
    POST at receipt, Prefill at admission) with 1-s timestamps."""
    L, OL, rate = real_lengths(phase)
    n = len(L)
    rng = np.random.RandomState(seed)
    A_rel = H.arrival_unit(n) / rate
    off = 1_700_000_000.37 + 1000.0 * seed
    send = off + A_rel + 0.0004 * np.arange(n)
    if send_stall:
        send[send_stall[0]:] += send_stall[1]
    avail = send + 0.002
    st0 = st1 = None
    if server_stall:
        st0 = float(send[0]) + server_stall[0]; st1 = st0 + server_stall[1]
        if server_stall[2] == "wide":
            avail = np.where((send >= st0) & (send < st1), st1 + 0.002, avail)
    post = [float(math.floor(x)) for x in avail]
    dec_lines, eng_lines = [], []
    n_it = 0; t_last_line = None
    rem = [None] * n
    gen: List[List[float]] = [[] for _ in range(n)]
    bs_at: List[List[int]] = [[] for _ in range(n)]
    running: List[int] = []
    pending = None  # (s, e, ids)
    batches, s_true, e_true, iters = [], [], [], []
    qptr = 0
    done = 0
    t = float(avail[0])
    while done < n:
        if st0 is not None and st0 <= t < st1:
            if pending is not None and pending[1] > t:
                pending = (pending[0], pending[1] + (st1 - t), pending[2])
            t = st1
        # (1) leave
        keep = []
        for r in running:
            if rem[r] > 0:
                keep.append(r)
            else:
                done += 1
        running = keep
        # (2) join
        if pending is not None and pending[1] <= t:
            for j in pending[2]:
                if rem[j] > 0:
                    running.append(j)
                else:
                    done += 1
            pending = None
        # (3) admit
        admitted = False
        if pending is None and qptr < n and avail[qptr] <= t and len(running) < CAP:
            slots = CAP - len(running)
            S, tok = [], 0
            j = qptr
            while j < n and avail[j] <= t and len(S) < slots and (not S or tok + L[j] <= TOKEN_BUDGET):
                S.append(j); tok += L[j]; j += 1
            Q = sum(1 for x in range(j, n) if avail[x] <= t)
            pd = (0.03 + 1e-4 * tok) * (108.0 / (108 - D) if running else 1.0)
            pending = (t, t + pd, S)
            batches.append((math.floor(t), S, len(running), Q))
            eng_lines.append(float(math.floor(t)))
            s_true.append(t); e_true.append(t + pd)
            for x in S:
                gen[x].append(t + pd); bs_at[x].append(len(running))
                rem[x] = OL[x] - 1
            qptr = j
            admitted = True
        if running:
            bs = len(running)
            dur = (0.0095 + 0.00022 * bs) * (_f(D, bs) if (pending is not None and pending[0] <= t < pending[1]) else 1.0)
            if admitted and stall_iter:
                dur += STALL_S
            te = t + dur
            iters.append((t, te))
            for r in running:
                gen[r].append(te); bs_at[r].append(bs); rem[r] -= 1
            n_it += 1
            if t_last_line is None:
                t_last_line = t
            if n_it % 40 == 0:
                qn = sum(1 for x in range(qptr, n) if avail[x] <= te)
                dec_lines.append((float(math.floor(te)), bs, 40 * bs / max(1e-9, te - t_last_line), qn))
                eng_lines.append(float(math.floor(te)))
                t_last_line = te
            t = te
        else:
            cands = []
            if pending is not None:
                cands.append(pending[1])
            elif qptr < n:
                cands.append(float(avail[qptr]))
            if not cands:
                break
            t = max(t, min(cands))
    # client receipt
    grid0 = off - 1.0
    tmax = max(g[-1] for g in gen) + 2.0
    ng = int((tmax - grid0) / 0.1) + 2
    ar = np.zeros(ng)
    for k in range(1, ng):
        ar[k] = 0.9 * ar[k - 1] + rng.normal(0, 0.003 * math.sqrt(1 - 0.81))
    rec_t = []
    for i in range(n):
        out, prev = [], -math.inf
        for g, b in zip(gen[i], bs_at[i]):
            if delay == "real":
                c = 0.012 + 0.0004 * b + ar[int((g - grid0) / 0.1)]
                r = g + max(0.003, c) + abs(rng.normal(0, 0.0004))
            else:
                r = g
            if recv_stall and recv_stall[0] <= r < recv_stall[0] + recv_stall[1]:
                r = recv_stall[0] + recv_stall[1]
            r = max(r, prev + 0.00005)
            out.append(r); prev = r
        rec_t.append(out)
    ttft = [rec_t[i][0] - send[i] for i in range(n)]
    itls = [[rec_t[i][j] - rec_t[i][j - 1] for j in range(1, len(rec_t[i]))] for i in range(n)]
    duration = max(x[-1] for x in rec_t) - send[0]
    win = (float(send[0]), float(send[0]) + duration)
    P = clip(union(list(zip(s_true, e_true))), *win)
    Dt = clip(union(iters), *win)
    truth = length(intersect(P, Dt)) / duration
    return {"batches": batches, "A_rel": A_rel, "rate": rate, "ttft": ttft, "itls": itls, "duration": duration,
            "truth": truth, "s_true": s_true, "e_true": e_true, "P_true": P, "D_true": Dt, "win": win,
            "send": send, "receipts": rec_t, "gen": gen, "iters": iters, "n": n,
            "decode_lines": dec_lines, "post": sorted(post), "eng": sorted(eng_lines), "server_stall_abs": (st0, st1),
            "max_running": max((b[2] for b in batches), default=0)}


# ----------------------------------------------------------------------------------- scoring
def conventions(rec, duration):
    win = rec["window"]
    D = rec["D"]
    out = {}
    for tag, key in (("lo_plus1iter", "CO_lo"), ("pt", "CO_pt"), ("earliest", "CO_hi")):
        out[tag] = length(rec[key]) / duration
    Pl = clip(union(list(zip(rec["s_hi"], rec["e"]))), *win)
    out["latest"] = length(intersect(Pl, D)) / duration
    return out


def score(rec, sy):
    co = length(rec["CO_pt"]) / sy["duration"]
    dl = [(rec["e"][k] - rec["s_pt"][k]) - (sy["e_true"][k] - sy["s_true"][k]) for k in range(len(rec["e"]))]
    med = float(np.median(dl))
    t1 = abs(co - sy["truth"]) <= T1_TOL
    t2 = abs(med) <= T2_TOL_S
    return {"co_pt": co, "err_co": co - sy["truth"], "median_len_err_s": med, "T1": t1, "T2": t2, "pass": t1 and t2}


def run_scenario(phase, D, delay, send_stall):
    sy = synth_engine(phase, D, delay, send_stall=send_stall)
    ctl_fn = make_variant([])
    rec0 = H.reconstruct_round(sy["batches"], sy["A_rel"], sy["ttft"], sy["itls"], sy["duration"])
    rec1 = ctl_fn(sy["batches"], sy["A_rel"], sy["ttft"], sy["itls"], sy["duration"])
    harness_identical = (rec0["s_pt"] == rec1["s_pt"] and rec0["e"] == rec1["e"] and rec0["CO_pt"] == rec1["CO_pt"])
    ctl = score(rec0, sy)
    res = {"phase": phase, "D": D, "delay": delay, "send_stall": send_stall, "truth": sy["truth"],
           "max_running_logged": sy["max_running"], "n_batches": len(sy["batches"]),
           "true_inflight_ms_p50": 1000 * float(np.median(np.array(sy["e_true"]) - np.array(sy["s_true"]))),
           "rec_inflight_pt_ms_p50": 1000 * float(np.median(np.array(rec0["e"]) - np.array(rec0["s_pt"]))),
           "R_match": rec0["R_match"], "offset_segments": rec0["offset_segments"],
           "harness_copy_identical": harness_identical, "control": ctl,
           "conventions": conventions(rec0, sy["duration"]), "mutants": {}}
    for m in ("intersect_is_union", "arrival_seed", "offset_plus_1s"):
        A_use = H.arrival_unit(sy["n"], seed=(2 if m == "arrival_seed" else 1)) / sy["rate"]
        r = H.reconstruct_round(sy["batches"], A_use, sy["ttft"], sy["itls"], sy["duration"], _mut=m)
        res["mutants"][m] = score(r, sy)
    for m, subs in MUTANTS.items():
        r = make_variant(subs)(sy["batches"], sy["A_rel"], sy["ttft"], sy["itls"], sy["duration"])
        res["mutants"][m] = score(r, sy)
    return res, sy, rec0


def rnd(x, nd=4):
    if type(x).__module__ == "numpy":
        x = x.item()
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        if abs(x) >= 1e6:
            return round(x, 2)
        return float(f"{x:.{nd}g}") if abs(x) >= 1 else round(x, nd)
    if isinstance(x, dict):
        return {str(k): rnd(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    return x


SCENARIOS = [(ph, D, dl, st) for ph in ("LO", "HI") for D in (44, 16) for dl in ("0", "real") for st in (None, (100, 3.0))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "he0_selftest_v2_2026-10-02.json"))
    args = ap.parse_args()
    rows = []
    for ph, D, dl, st in SCENARIOS:
        r, _sy, _rec = run_scenario(ph, D, dl, st)
        rows.append(r)
        c = r["control"]
        print(f"[{ph} D{D} delay={dl:4s} stall={'3s@100' if st else 'none':6s}] truth={r['truth']:.4f} "
              f"conv lo/pt/earl/latest={r['conventions']['lo_plus1iter']:.4f}/{r['conventions']['pt']:.4f}/"
              f"{r['conventions']['earliest']:.4f}/{r['conventions']['latest']:.4f}  ctl err={c['err_co']:+.4f} "
              f"lenerr={1000*c['median_len_err_s']:+.1f}ms -> {'PASS' if c['pass'] else 'FAIL'}  copy_identical={r['harness_copy_identical']}")
        for m, s in r["mutants"].items():
            tag = ("KILLED" if not s["pass"] else "SURVIVED") if c["pass"] else ("(control FAIL) " + ("differs" if s["pass"] != c["pass"] or abs(s["err_co"] - c["err_co"]) > T1_TOL else "same"))
            print(f"      {m:20s} err={s['err_co']:+.4f} lenerr={1000*s['median_len_err_s']:+.1f}ms -> {tag}")
    summary = {}
    for m in rows[0]["mutants"]:
        usable = [r for r in rows if r["control"]["pass"]]
        killed = [r for r in usable if not r["mutants"][m]["pass"]]
        summary[m] = {"usable_scenarios": len(usable), "killed_in": len(killed),
                      "verdict": ("KILLED" if usable and len(killed) == len(usable) else
                                  "PARTIAL" if killed else ("SURVIVED" if usable else "NO_USABLE_SCENARIO")),
                      "killed_by_phase": {ph: f"{sum(1 for r in killed if r['phase'] == ph)}/{sum(1 for r in usable if r['phase'] == ph)}" for ph in ("LO", "HI")}}
    ctl_ok = {ph: sum(1 for r in rows if r["phase"] == ph and r["control"]["pass"]) for ph in ("LO", "HI")}
    print("control pass per phase:", ctl_ok)
    for m, s in summary.items():
        print(f"MUTANT {m:20s} {s['verdict']}  killed {s['killed_in']}/{s['usable_scenarios']} usable  by phase {s['killed_by_phase']}")
    out = {"generated": "2026-10-02", "script": Path(__file__).name, "criteria": {"T1_tol": T1_TOL, "T2_tol_s": T2_TOL_S},
           "control_pass_per_phase": ctl_ok, "mutant_summary": summary, "scenarios": rows}
    Path(args.output).write_text(json.dumps(rnd(out), indent=1))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
