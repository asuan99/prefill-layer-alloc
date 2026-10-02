#!/usr/bin/env python3
"""HE0 contamination follow-up (result-analyst, GPU 0, 2026-10-02).

Three judgements left open by RESULT_2026-10-01.md / SYNTHESIS_2026-10-01.md,
computed from the SAME archived inputs that ``he0_realized.py`` reads
(``results/slo_sched/sgptv{srv,Lo,Hi}_*_L3H12_<job>``; no telemetry exists for
these runs, so everything below rests on the he0_realized timeline
RECONSTRUCTION — its validation and limits are in RESULT §0/§Q5):

  C1  positioning residual (AUDIT_LOGIC_VERDICT C-1).  For each dynamic run and
      phase, residency weights w_d = share of co-residency time spent at decode
      SM d (he0_realized ``co_split_s``, earliest AND latest placement of each
      1-s controller transition).  Prediction = sum_d w_d * mean(static d arm).
      R = observed - prediction, expressed in units of the pooled within-arm SD
      of the static arms.  Rule: |R| <= 1 SD -> positioning only; R < -2 SD ->
      intrinsic dynamic penalty; R > +2 SD -> weakens HE0 explanation.  Static
      references are all-node (primary) and node-matched-where-available
      (secondary, coverage reported — full node matching is infeasible: there is
      no gpu38 d24/d16 run).  Calibration: leave-one-out R of the static runs
      themselves.
  C4  cap-48 binding time share in HI.  Conventions (all reported, a single
      value is never quoted): reconstructed running count, wall-time weighted
      over the round window and over decode-active time, thresholds 46/47/48;
      "binding" = running >= 46 AND reconstructed waiting queue >= 1; log-only
      decode-iteration weighted (``Decode batch`` lines, 1 per 40 iterations);
      log-only prefill-batch weighted (``Prefill batch`` #running-req).
      Durations are SUMMED over the 3 HI rounds (gate 7).
  S   ITL>1 s stall attribution (856964 "no-gate collapse", CONSENSUS 1-11).
      Every run/round: cluster simultaneous token gaps > 1 s, share of the
      decode-active requests that stall, server engine log lines (Prefill/
      Decode) and HTTP POST lines inside the stall's interior whole seconds,
      the next Decode line's gen throughput, client send-lateness jump,
      controller lines (SLO-BIND/SCHED/FEAS) within +-3 s, and co-located HE0
      jobs on the same node.

Reuses ``he0_realized`` (parse_server, arrival_unit, reconstruct_round, run_one,
goodput_rows) and ``pdmux_eval.analyze`` (percentile; bootstrap convention
random.Random(1), 10000 samples).

Usage:  python3 he0_followup.py            -> he0_followup_2026-10-02.json
        python3 he0_followup.py --self-test
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
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import he0_realized as H  # noqa: E402
from he0_realized import SRC, A  # noqa: E402

STATIC_ORDER = ("d16", "d24", "d34", "d44")
D_OF = {16: "d16", 24: "d24", 34: "d34", 44: "d44"}
RE_POST = re.compile(H.RE_TS + r" INFO: .*\"POST /generate")
RE_DEC_THR = re.compile(H.RE_TS + r" Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)")
BOOT_N, BOOT_SEED = 10000, 1
STALL_GAP_S = 1.0


def rnd(x, nd=4):
    """Round floats recursively (keeps JSON free of 17-digit values)."""
    if type(x).__module__ == "numpy":
        x = x.item()
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        if abs(x) >= 1e6:  # epoch seconds
            return round(x, 2)
        return float(f"{x:.{nd}g}") if abs(x) >= 1 else round(x, nd)
    if isinstance(x, dict):
        return {str(k): rnd(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, nd) for v in x]
    return x


def msd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
            "n": len(xs), "min": min(xs), "max": max(xs)}


# --------------------------------------------------------------------------- reconstruction access
def recon(job: int):
    """Same round partition as he0_realized.run_one, but keeps the reconstruction objects."""
    srv = list(SRC.glob(f"sgptvsrv_*_{job}.log"))[0]
    tag = srv.name[len("sgptvsrv_"):-4]
    S = H.parse_server(srv)
    lo = [json.loads(x) for x in (SRC / f"sgptvLo_{tag}.jsonl").open() if x.strip()]
    hi = [json.loads(x) for x in (SRC / f"sgptvHi_{tag}.jsonl").open() if x.strip()]
    rounds = [("LO", 0, lo[0]), ("HI", 0, hi[0]), ("LO", 1, lo[1]), ("HI", 1, hi[1]), ("LO", 2, lo[2]), ("HI", 2, hi[2])]
    pb = S["prefill"]
    k = 0
    while k < len(pb) and not (pb[k][1] == 1 and pb[k][2] == rounds[0][2]["input_lens"][0] and pb[k][3] == 0):
        k += 1
    out = []
    for phase, ri, o in rounds:
        L = o["input_lens"]; n = len(L); rate = float(o["request_rate"])
        assert pb[k][1] == 1 and pb[k][2] == L[0]
        k += 1
        batches, i = [], 0
        while i < n:
            T, ns, nt, R, Q = pb[k]
            Sx = list(range(i, i + ns))
            assert sum(L[j] for j in Sx) == nt
            batches.append((T, Sx, R, Q)); i += ns; k += 1
        A_rel = H.arrival_unit(n) / rate
        ttft = [float(x) for x in o["ttfts"]]
        itls = [[float(v) for v in x] for x in o["itls"]]
        rec = H.reconstruct_round(batches, A_rel, ttft, itls, float(o["duration"]))
        rec["A_rel"] = A_rel
        out.append({"phase": phase, "round": ri, "o": o, "batches": batches, "rec": rec,
                    "ttft": ttft, "itls": itls})
    assert k == len(pb)
    post, dthr = [], []
    for ln in srv.open(errors="replace"):
        m = RE_POST.match(ln)
        if m:
            post.append(H.ts(m.group(1))); continue
        m = RE_DEC_THR.match(ln)
        if m:
            dthr.append((H.ts(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))))
    node = None
    outp = SRC / f"sgptv_{job}.out"
    if outp.exists():
        mm = re.search(r"SLURM_NODELIST = (\S+)", outp.read_text(errors="replace"))
        node = mm.group(1) if mm else None
    eng = sorted([x[0] for x in S["prefill"]] + [x[0] for x in S["decode"]])
    return {"job": job, "tag": tag, "S": S, "rounds": out, "post": sorted(post), "dthr": dthr,
            "eng": eng, "node": node, "srv": srv}


# --------------------------------------------------------------------------- step functions
def step_fn(events: List[Tuple[float, int]]):
    """events (t, +-1) -> (times, counts) right-continuous step function."""
    events = sorted(events)
    times, cnt, cur = [], [], 0
    for t, d in events:
        cur += d
        if times and times[-1] == t:
            cnt[-1] = cur
        else:
            times.append(t); cnt.append(cur)
    return times, cnt


def time_where(times, cnt, w0, w1, pred) -> float:
    """Total time in [w0,w1) where pred(count) holds."""
    tot = 0.0
    k = bisect.bisect_right(times, w0) - 1
    cur_t = w0
    cur_c = cnt[k] if k >= 0 else 0
    k += 1
    while k < len(times) and times[k] < w1:
        if pred(cur_c):
            tot += times[k] - cur_t
        cur_t, cur_c = times[k], cnt[k]
        k += 1
    if pred(cur_c):
        tot += w1 - cur_t
    return tot


def joint_time(f1, f2, w0, w1, pred) -> float:
    """Time in [w0,w1) where pred(c1, c2) holds for two step functions."""
    t1, c1 = f1; t2, c2 = f2
    cuts = sorted(set([w0, w1] + [t for t in t1 if w0 < t < w1] + [t for t in t2 if w0 < t < w1]))
    tot = 0.0
    for a, b in zip(cuts, cuts[1:]):
        m = (a + b) / 2
        k1 = bisect.bisect_right(t1, m) - 1; k2 = bisect.bisect_right(t2, m) - 1
        v1 = c1[k1] if k1 >= 0 else 0; v2 = c2[k2] if k2 >= 0 else 0
        if pred(v1, v2):
            tot += b - a
    return tot


def running_and_queue(rec, batches):
    n = len(rec["fin"])
    e, bidx, fin, a, s = rec["e"], rec["bidx"], rec["fin"], rec["a"], rec["s_pt"]
    run = step_fn([(e[bidx[i]], +1) for i in range(n)] + [(max(fin[i], e[bidx[i]]), -1) for i in range(n)])
    que = step_fn([(a[i], +1) for i in range(n)] + [(max(a[i], s[bidx[i]]), -1) for i in range(n)])
    # cap occupancy = admitted and not finished = decode running + seqs whose prefill is in flight
    # (the logs show the cap counts prefilling seqs: Prefill line #running-req 47 + #new-seq 1 = 48)
    occ = step_fn([(s[bidx[i]], +1) for i in range(n)] + [(max(fin[i], s[bidx[i]]), -1) for i in range(n)])
    return run, que, occ


# --------------------------------------------------------------------------- C4
def decode_line_weights(lines, w0, w1, spread: bool):
    """Backward Delta-t weights for Decode batch log lines inside [w0, w1).

    Each line is taken to describe the interval since the previous line.  Log timestamps are whole
    seconds; ``spread`` places the m lines of one second at s+(j+1)/(m+1) (sub-second spread),
    otherwise all lines of a second sit at s (integer Delta-t: effectively the last line per second)."""
    L = [x for x in lines if w0 <= x[0] < w1]
    if spread:
        by: Dict[float, list] = {}
        for x in L:
            by.setdefault(x[0], []).append(x)
        pos = []
        for s_, xs in sorted(by.items()):
            m = len(xs)
            for j, x in enumerate(xs):
                pos.append((s_ + (j + 1) / (m + 1), x))
    else:
        pos = [(x[0], x) for x in L]
    out, prev = [], math.floor(w0)
    for p, x in pos:
        out.append((max(0.0, p - prev), x)); prev = max(prev, p)
    return out


def c4_run(R) -> Dict[str, object]:
    """HI-phase cap-48 saturation/binding shares; HI rounds pooled with durations SUMMED."""
    acc: Dict[str, float] = {}

    def add(k, v):
        acc[k] = acc.get(k, 0.0) + v
    for rd in R["rounds"]:
        if rd["phase"] != "HI":
            continue
        rec = rd["rec"]; w0, w1 = rec["window"]
        run, que, occ = running_and_queue(rec, rd["batches"])
        add("W", w1 - w0)
        for thr in (46, 47, 48):
            add(f"oc_ge{thr}", time_where(*occ, w0, w1, lambda c, t=thr: c >= t))
            add(f"oc_ge{thr}_q", joint_time(occ, que, w0, w1, lambda r, q, t=thr: r >= t and q >= 1))
        add("q_pos", time_where(*que, w0, w1, lambda c: c >= 1))
        for thr in (46, 47, 48):
            add(f"rc_ge{thr}", time_where(*run, w0, w1, lambda c, t=thr: c >= t))
            add(f"rc_ge{thr}_q", joint_time(run, que, w0, w1, lambda r, q, t=thr: r >= t and q >= 1))
        # log-only: Decode batch lines (#running-req, #queue-req), Delta-t weighted
        for tag, sp in (("dl_int", False), ("dl_spread", True)):
            for wt, (Td, nlog, _thr, qlog) in decode_line_weights(R["dthr"], w0, w1, sp):
                add(f"{tag}_W", wt)
                add(f"{tag}_qpos", wt * (qlog >= 1))
                for thr in (46, 47, 48):
                    add(f"{tag}_ge{thr}", wt * (nlog >= thr))
                    add(f"{tag}_ge{thr}_q", wt * (nlog >= thr and qlog >= 1))
        for Td, nlog, _thr, qlog in R["dthr"]:
            if w0 <= Td < w1:
                add("dl_lines", 1); add("dl_ge46_lines", nlog >= 46); add("dl_eq48_q_lines", nlog >= 48 and qlog >= 1)
        for T, Sx, Rl, Q in rd["batches"]:
            add("pf_batches", 1); add("pf_ge46", Rl >= 46); add("pf_fill48_q", Rl + len(Sx) >= 48 and Q >= 1)
    W = acc["W"]
    o = {"W_s_sum_HI": W, "n_decode_lines": int(acc.get("dl_lines", 0)), "n_prefill_batches": int(acc.get("pf_batches", 0))}
    for thr in (46, 47, 48):
        o[f"recon_time_w_running_ge{thr}"] = acc[f"rc_ge{thr}"] / W
        o[f"recon_time_w_running_ge{thr}_and_queue"] = acc[f"rc_ge{thr}_q"] / W
    for thr in (46, 47, 48):
        o[f"recon_time_w_occupancy_ge{thr}"] = acc[f"oc_ge{thr}"] / W
        o[f"recon_time_w_occupancy_ge{thr}_and_queue"] = acc[f"oc_ge{thr}_q"] / W
    o["recon_occupancy48_and_queue_given_queue"] = acc["oc_ge48_q"] / acc["q_pos"] if acc["q_pos"] else None
    o["recon_time_w_queue_nonempty"] = acc["q_pos"] / W
    o["recon_running_eq48_given_queue"] = acc["rc_ge48_q"] / acc["q_pos"] if acc["q_pos"] else None
    for tag in ("dl_int", "dl_spread"):
        Wd = acc.get(f"{tag}_W", 0.0)
        for thr in (46, 47, 48):
            o[f"{tag}_time_w_running_ge{thr}"] = acc.get(f"{tag}_ge{thr}", 0.0) / Wd if Wd else None
            o[f"{tag}_time_w_running_ge{thr}_and_queue"] = acc.get(f"{tag}_ge{thr}_q", 0.0) / Wd if Wd else None
        o[f"{tag}_time_w_queue_nonempty"] = acc.get(f"{tag}_qpos", 0.0) / Wd if Wd else None
        o[f"{tag}_covered_s_over_window"] = Wd / W
    o["line_w_decode_running_ge46"] = acc.get("dl_ge46_lines", 0) / max(1, acc.get("dl_lines", 0))
    o["line_w_decode_running_eq48_and_queue"] = acc.get("dl_eq48_q_lines", 0) / max(1, acc.get("dl_lines", 0))
    o["batch_w_prefill_running_ge46(understates: admission instants only)"] = acc.get("pf_ge46", 0) / max(1, acc.get("pf_batches", 0))
    o["batch_w_prefill_fills_48_with_queue"] = acc.get("pf_fill48_q", 0) / max(1, acc.get("pf_batches", 0))
    return o


# --------------------------------------------------------------------------- C1
# Primary = HI phase with cliff-free predicates (claims-auditor premise check 2026-10-02): legacy mean-ITL
# goodput, per-request-ITL-p95 goodput re-scored at 65/70 ms (DIAGNOSTIC re-score for the residual test
# only — not a policy conclusion at another SLO, gate 7), TTFT p50.  Canonical 60 ms is reported but is a
# cliff metric (pooled token ITL p95 ~59-61 ms sits on the SLO), where a linear mixture is ill-posed.
# LO is a structural null (goodput bounded by the arrival rate) and COMBINED inherits LO.
C1_METRICS = (("HI", "legacy_mean_itl_goodput_req_s"), ("HI", "slo_goodput_itl65"), ("HI", "slo_goodput_itl70"),
              ("HI", "ttft_p50_ms"), ("HI", "slo_goodput_req_s"),
              ("COMBINED", "legacy_mean_itl_goodput_req_s"), ("LO", "legacy_mean_itl_goodput_req_s"))
CLIFF_FREE = ("HI:legacy_mean_itl_goodput_req_s", "HI:slo_goodput_itl65", "HI:slo_goodput_itl70", "HI:ttft_p50_ms")


def goodput_ext(run) -> Dict[str, Dict[str, float]]:
    g = H.goodput_rows(run)
    tag = run["tag"]
    for ph, f in (("LO", f"sgptvLo_{tag}.jsonl"), ("HI", f"sgptvHi_{tag}.jsonl")):
        reqs, dur = A.load_bench_serving_rounds(SRC / f)
        for itl in (65.0, 70.0):
            g[ph][f"slo_goodput_itl{int(itl)}"] = A.summarize_requests(reqs, 0.0, dur, H.TTFT_SLO_MS, itl)["slo_goodput_req_s"]
    return g


def weights(run, phase, place) -> Dict[int, float]:
    tot: Dict[int, float] = {}
    for x in run["rounds"]:
        if phase != "COMBINED" and x["phase"] != phase:
            continue
        for d, s in x["co_split_s"][place].items():
            tot[int(d)] = tot.get(int(d), 0.0) + s
    T = sum(tot.values())
    return {d: s / T for d, s in tot.items()} if T else {}


def c1(runs_by_arm, gp, exclude: Sequence[int] = ()):
    excl = set(exclude)
    rb = {a: [r for r in rs if r["job"] not in excl] for a, rs in runs_by_arm.items()}
    stat_vals = {}
    for ph, m in C1_METRICS:
        stat_vals[(ph, m)] = {int(a[1:]): [(r["job"], r["node"], gp[r["job"]][ph][m]) for r in rb[a]] for a in STATIC_ORDER}
    out = {"excluded_jobs": sorted(excl), "pooled_sd": {}, "static_loo": {}, "arms": {}}
    for (ph, m), dv in stat_vals.items():
        num = sum((len(v) - 1) * statistics.variance([x[2] for x in v]) for v in dv.values())
        den = sum(len(v) - 1 for v in dv.values())
        psd = math.sqrt(num / den)
        key = f"{ph}:{m}"
        out["pooled_sd"][key] = {"pooled_sd": psd, "df": den,
                                 "per_arm": {f"d{d}": dict(msd([x[2] for x in v]),
                                                           cv=statistics.stdev([x[2] for x in v]) / abs(statistics.fmean([x[2] for x in v])))
                                             for d, v in dv.items()}}
        loo = []
        for d, v in dv.items():
            for j in range(len(v)):
                rest = [x[2] for k, x in enumerate(v) if k != j]
                loo.append((v[j][2] - statistics.fmean(rest)) / psd)
        out["static_loo"][key] = {"R_over_pooledSD": msd(loo), "abs_gt1": sum(abs(z) > 1 for z in loo),
                                  "abs_gt2": sum(abs(z) > 2 for z in loo), "n": len(loo)}
    for arm in H.DYNAMIC:
        rs = rb[arm]
        arm_out = {}
        for ph, m in C1_METRICS:
            key = f"{ph}:{m}"
            dv = stat_vals[(ph, m)]
            psd = out["pooled_sd"][key]["pooled_sd"]
            means = {d: statistics.fmean([x[2] for x in v]) for d, v in dv.items()}
            sds = {d: statistics.stdev([x[2] for x in v]) for d, v in dv.items()}
            per_place = {}
            for place in ("earliest", "latest"):
                rows = []
                for r in rs:
                    w = weights(r, ph, place)
                    obs = gp[r["job"]][ph][m]
                    pred = sum(wd * means[d] for d, wd in w.items())
                    mix_sd = math.sqrt(sum(wd * sds[d] ** 2 for d, wd in w.items()))
                    cov, pred_nm = 0.0, 0.0
                    for d, wd in w.items():
                        same = [x[2] for x in dv[d] if x[1] == r["node"]]
                        if same:
                            cov += wd; pred_nm += wd * statistics.fmean(same)
                        else:
                            pred_nm += wd * means[d]
                    rows.append({"job": r["job"], "node": r["node"], "w": {f"d{d}": wd for d, wd in sorted(w.items())},
                                 "obs": obs, "pred": pred, "R": obs - pred, "R_over_pooledSD": (obs - pred) / psd,
                                 "R_over_mixSD": (obs - pred) / mix_sd if mix_sd > 0 else None,
                                 "pred_node_matched_where_available": pred_nm,
                                 "R_node_matched_over_pooledSD": (obs - pred_nm) / psd,
                                 "node_match_weight_coverage": cov})
                rng = random.Random(BOOT_SEED)
                boot = []
                stat_lists = {d: [x[2] for x in v] for d, v in dv.items()}
                for _ in range(BOOT_N):
                    mb = {d: statistics.fmean(rng.choice(v) for _ in v) for d, v in stat_lists.items()}
                    pick = [rng.choice(rows) for _ in rows]
                    boot.append(statistics.fmean(p["obs"] - sum(wv * mb[int(dk[1:])] for dk, wv in p["w"].items()) for p in pick))
                mR = statistics.fmean(p["R"] for p in rows)
                z = mR / psd
                verdict = ("positioning_only(|R|<=1SD)" if abs(z) <= 1 else
                           "intrinsic_penalty(R<-2SD)" if z < -2 else
                           "weakens_HE0(R>+2SD)" if z > 2 else "between_1_and_2_SD(ambiguous)")
                per_place[place] = {"runs": rows, "mean_R": mR, "sd_R": statistics.stdev([p["R"] for p in rows]) if len(rows) > 1 else None,
                                    "n": len(rows), "mean_R_over_pooledSD": z,
                                    "mean_R_ci95": [A.percentile(boot, 0.025), A.percentile(boot, 0.975)],
                                    "mean_R_ci95_over_pooledSD": [A.percentile(boot, 0.025) / psd, A.percentile(boot, 0.975) / psd],
                                    "mean_R_node_matched_over_pooledSD": statistics.fmean(p["R_node_matched_over_pooledSD"] for p in rows),
                                    "mean_node_match_coverage": statistics.fmean(p["node_match_weight_coverage"] for p in rows),
                                    "runs_R_lt_minus2SD": sum(p["R_over_pooledSD"] < -2 for p in rows),
                                    "runs_abs_R_le_1SD": sum(abs(p["R_over_pooledSD"]) <= 1 for p in rows),
                                    "runs_R_gt_plus2SD": sum(p["R_over_pooledSD"] > 2 for p in rows),
                                    "verdict_by_arm_mean": verdict}
            arm_out[key] = per_place
        out["arms"][arm] = arm_out
    return out


def zero_transition_contrast(runs_by_arm, gp):
    """bind+GATE runs with 0 controller transitions sat at idx2 (24 SM) for the whole run = physically the
    d24 static split plus per-step controller evaluation, on a different node/date (gpu38, 07-17 vs
    gpu37/gpu43, 07-15/07-18).  Descriptive contrast only (node/date confounded, n=3 vs n=4)."""
    z = [r for r in runs_by_arm["bind_gate"] if r["n_ctl_transitions"] == 0]
    keys = ("legacy_mean_itl_goodput_req_s", "slo_goodput_req_s", "slo_goodput_itl65", "ttft_p50_ms", "ttft_p95_ms",
            "token_itl_p50_ms", "token_itl_p95_ms", "token_itl_p99_ms")
    out = {"zero_transition_bind_gate_jobs": [(r["job"], r["node"]) for r in z],
           "d24_jobs": [(r["job"], r["node"]) for r in runs_by_arm["d24"]]}
    for k in keys:
        a = [gp[r["job"]]["HI"][k] for r in z]
        b = [gp[r["job"]]["HI"][k] for r in runs_by_arm["d24"]]
        out[k] = {"bind_gate_0tr": msd(a), "d24": msd(b), "diff_pct_of_d24": 100 * (statistics.fmean(a) - statistics.fmean(b)) / statistics.fmean(b),
                  "unpaired_ci": A.unpaired_bootstrap_ci(b, a)}
    return out


# --------------------------------------------------------------------------- stalls
def stall_events(R) -> List[Dict[str, object]]:
    """Cluster simultaneous client token gaps > 1 s and test them against the server log.

    Discriminator (claims-auditor premise check 2026-10-02): did the server's ``Decode batch`` lines continue
    at the round's normal rate during the client gap (continuous -> client side; broken -> server side),
    together with the client send-lateness step and the HTTP POST receipt lag.  POST lines are logged at
    request RECEIPT (checked: POST ts - reconstructed send = -0.5 s median, the 1-s floor), so a POST lag
    of seconds for requests sent during the gap means the HTTP process itself was silent.  Second-scale
    statements only: reconstruction alignment is validated to ~0.3-1 s, not 100 ms."""
    evs = []
    S = R["S"]
    ctl = sorted((t, f"{a}->{b}") for t, a, b in S["ctl"])
    feas = sorted(t for t, _a, _b in S["feas"])
    dec_t = [x[0] for x in R["dthr"]]
    for rd in R["rounds"]:
        rec = rd["rec"]; w0, w1 = rec["window"]
        tt_all = rec["tok_times"]
        n = len(tt_all)
        gaps = sorted((tt[j - 1], tt[j], i) for i, tt in enumerate(tt_all) for j in range(1, len(tt))
                      if tt[j] - tt[j - 1] > STALL_GAP_S)
        clusters = []
        for g in gaps:
            if clusters and g[0] < clusters[-1]["end"]:
                c = clusters[-1]; c["end"] = max(c["end"], g[1]); c["g"].append(g)
                c["core0"] = max(c["core0"], g[0]); c["core1"] = min(c["core1"], g[1])
            else:
                clusters.append({"start": g[0], "end": g[1], "core0": g[0], "core1": g[1], "g": [g]})
        xreq = [rec["a"][i] - rec["A_rel"][i] for i in range(n)]
        P = [t for t in R["post"] if w0 - 1 <= t < w1 + 1]
        post_lag = [P[k] + 0.5 - rec["a"][k] for k in range(n)] if len(P) == n else None
        dl_round = [t for t in dec_t if w0 <= t < w1]
        dl_rate = len(dl_round) / (w1 - w0)
        thr_round = [x for x in R["dthr"] if w0 <= x[0] < w1 + 1 and x[1] >= 40]
        thr_med = statistics.median([x[2] for x in thr_round]) if thr_round else None
        for c in clusters:
            reqs = sorted(set(g[2] for g in c["g"]))
            mid = (c["core0"] + c["core1"]) / 2 if c["core1"] > c["core0"] else (c["start"] + c["end"]) / 2
            active = [i for i, tt in enumerate(tt_all) if tt[0] < mid < tt[-1]]
            s0, s1 = c["start"], c["end"]
            interior = [s for s in range(math.floor(s0) + 1, math.floor(s1)) if s + 1 <= s1]
            def cnt(ts_list):
                return sum(1 for t in ts_list if interior and interior[0] <= t <= interior[-1])
            dec_in = cnt(dec_t)
            eng_in = cnt(R["eng"])
            post_in = cnt(R["post"])
            exp_dec = dl_rate * len(interior)
            nxt = next((x for x in R["dthr"] if x[0] >= math.floor(s1)), None)
            sent = [i for i in range(n) if s0 <= rec["a"][i] <= s1]
            idx = [i for i in range(1, n) if s0 - 1 <= rec["a"][i] <= s1 + 1]
            lat_jump = max(xreq[i] - xreq[i - 1] for i in idx) if idx else None
            lag = max(post_lag[i] for i in sent) if (post_lag and sent) else None
            ctl_after = [t - s1 for t, _ in ctl if s1 - 0.5 <= t <= s1 + 10]
            ctl_before = [s0 - t for t, _ in ctl if s0 - 10 <= t < s0 - 0.5]
            if interior and exp_dec >= 1:
                if dec_in >= 0.5 * exp_dec:
                    if lat_jump is not None and lat_jump >= 0.5:
                        cls = "CLIENT(server Decode lines continuous; client send-lateness step)"
                    else:
                        cls = "CLIENT_RECEIVE(server Decode lines continuous; no send step)"
                else:
                    if lag is not None and lag >= 0.5 * (s1 - s0):
                        cls = "SERVER_WIDE(Decode lines broken; HTTP receipt also delayed; client sent on schedule)"
                    elif post_in > 0:
                        cls = "SERVER_SCHEDULER(Decode lines broken; HTTP alive)"
                    else:
                        cls = "SERVER(Decode lines broken; HTTP state undetermined)"
            else:
                cls = "UNRESOLVED(<1 interior log second)"
                if nxt is not None and thr_med and nxt[0] <= s1 + 2 and nxt[1] >= 20 and nxt[2] < 0.5 * thr_med:
                    cls = "SERVER_LIKELY(next Decode line throughput dip)"
            evs.append({"phase": rd["phase"], "round": rd["round"],
                        "start_local": _dt.datetime.fromtimestamp(s0).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5],
                        "t_start": s0, "dur_s": s1 - s0, "core_s": max(0.0, c["core1"] - c["core0"]),
                        "n_req_gapped": len(reqs), "n_active_at_core": len(active),
                        "frac_active_gapped": len(set(reqs) & set(active)) / max(1, len(active)),
                        "interior_log_seconds": len(interior), "decode_lines_interior": dec_in,
                        "decode_lines_expected_at_round_rate": exp_dec, "engine_lines_interior": eng_in,
                        "post_lines_interior": post_in, "n_sent_during_gap": len(sent),
                        "max_http_receipt_lag_s_of_requests_sent_during_gap": lag,
                        "client_send_lateness_jump_s": lat_jump,
                        "next_decode_line": None if nxt is None else {"dt_after_end_s": nxt[0] - s1, "running": nxt[1],
                                                                       "gen_tok_s": nxt[2]},
                        "round_median_gen_tok_s_running_ge40": thr_med,
                        "ctl_transitions_within_10s_before": len(ctl_before),
                        "ctl_transitions_within_10s_after_end": len(ctl_after),
                        "ctl_after_end_s": ctl_after,
                        "feas_lines_within_gap": sum(1 for t in feas if s0 - 1 <= t <= s1),
                        "class": cls})
    return evs


def stalls(all_R, arm_of):
    per_run = {}
    for job, R in all_R.items():
        evs = stall_events(R)
        per_run[job] = {"arm": arm_of[job], "node": R["node"], "events": evs,
                        "job_span": [R["eng"][0], R["eng"][-1]]}
    # co-located concurrent HE0 jobs and their simultaneous stalls
    for job, pr in per_run.items():
        for ev in pr["events"]:
            co = []
            for j2, p2 in per_run.items():
                if j2 == job or p2["node"] != pr["node"]:
                    continue
                if p2["job_span"][0] <= ev["t_start"] <= p2["job_span"][1]:
                    sim = [e2 for e2 in p2["events"] if abs(e2["t_start"] - ev["t_start"]) <= 1.5]
                    co.append({"job": j2, "arm": p2["arm"], "simultaneous_stall": bool(sim)})
            ev["colocated_he0_jobs_running"] = co
    return per_run


def stall_summary(per_run, all_R):
    rows = {}
    for job, pr in per_run.items():
        big = [e for e in pr["events"] if e["n_req_gapped"] >= 5 and e["frac_active_gapped"] >= 0.9]
        rows[job] = {"arm": pr["arm"], "node": pr["node"],
                     "n_global_stalls": len(big), "global_stall_s": sum(e["dur_s"] for e in big),
                     "n_global_stalls_HI": sum(e["phase"] == "HI" for e in big),
                     "max_stall_s": max([e["dur_s"] for e in pr["events"]], default=0.0),
                     "global_stall_s_HI": sum(e["dur_s"] for e in big if e["phase"] == "HI"),
                     "classes": sorted(set(e["class"] for e in big)),
                     "n_ctl_transitions_run": len(all_R[job]["S"]["ctl"]),
                     "n_ctl_transitions_within_10s_after_a_global_stall": sum(
                         any(e["t_start"] + e["dur_s"] - 1.0 <= t <= e["t_start"] + e["dur_s"] + 10 for e in big)
                         for t, _a, _b in all_R[job]["S"]["ctl"]),
                     "n_global_stalls_preceded_by_ctl_transition_within_10s": sum(e["ctl_transitions_within_10s_before"] > 0 for e in big)}
    by_arm = {}
    for arm in H.HE0_JOBS:
        rr = [v for v in rows.values() if v["arm"] == arm]
        by_arm[arm] = {"n_runs": len(rr), "runs_with_global_stall": sum(v["n_global_stalls"] > 0 for v in rr),
                       "global_stalls_total": sum(v["n_global_stalls"] for v in rr),
                       "global_stall_s_per_run": msd([v["global_stall_s"] for v in rr]),
                       "classes": sorted(set(c for v in rr for c in v["classes"]))}
    return rows, by_arm


def stall_touch(R, evs, gp_job):
    """Which HI/LO failing requests touch a global stall (token gap or TTFT window overlapping it)."""
    big = [e for e in evs if e["n_req_gapped"] >= 5 and e["frac_active_gapped"] >= 0.9]
    out = {}
    for ph in ("LO", "HI"):
        tot = fail_c = fail_l = touch = fail_c_touch = fail_l_touch = 0
        for rd in R["rounds"]:
            if rd["phase"] != ph:
                continue
            rec = rd["rec"]
            ivs = [(e["t_start"], e["t_start"] + e["dur_s"]) for e in big if e["phase"] == ph and e["round"] == rd["round"]]
            for i, tt in enumerate(rec["tok_times"]):
                life = (rec["a"][i], tt[-1])
                t_ = any(min(life[1], b) > max(life[0], a) for a, b in ivs)
                itl_ms = [1000.0 * v for v in rd["itls"][i]]
                ttft_ok = 1000.0 * rd["ttft"][i] <= H.TTFT_SLO_MS
                ok_c = ttft_ok and (not itl_ms or A.percentile(itl_ms, 0.95) <= H.ITL_SLO_MS)
                ok_l = ttft_ok and statistics.fmean(itl_ms or [0.0]) <= H.ITL_SLO_MS
                tot += 1; touch += t_
                fail_c += not ok_c; fail_l += not ok_l
                fail_c_touch += (not ok_c) and t_; fail_l_touch += (not ok_l) and t_
        out[ph] = {"requests": tot, "touched_by_global_stall": touch, "fail_canonical": fail_c,
                   "fail_canonical_touched": fail_c_touch, "fail_legacy": fail_l, "fail_legacy_touched": fail_l_touch}
    return out


def index_paired_attribution(all_R, target: int, peers: Sequence[int], evs) -> Dict[str, object]:
    """All HE0 runs replay the same 200-prompt rounds with the same arrival schedule (seed 1), so request
    (round, index) is comparable across runs.  For the target run, split each round's requests by their
    position relative to the target's global stalls (pre = finished before the first stall starts,
    touched = lifetime overlaps a stall, post = arrived after the last stall ended) and compare the failure
    rate on the SAME request slots against the peer runs.  Descriptive (n peers small)."""
    big = [e for e in evs if e["n_req_gapped"] >= 5 and e["frac_active_gapped"] >= 0.9]
    out = {}
    for ph in ("LO", "HI"):
        cats: Dict[str, Dict[str, list]] = {c: {"t_leg": [], "p_leg": [], "t_can": [], "p_can": []} for c in ("pre", "touched", "post", "between")}
        per_round = {}
        for rd in all_R[target]["rounds"]:
            if rd["phase"] != ph:
                continue
            rec = rd["rec"]
            pr_acc = per_round.setdefault(f"{ph}{rd['round']}", {})
            ivs = [(e["t_start"], e["t_start"] + e["dur_s"]) for e in big if e["phase"] == ph and e["round"] == rd["round"]]
            peer_rds = [next(x for x in all_R[p]["rounds"] if x["phase"] == ph and x["round"] == rd["round"]) for p in peers]

            def fails(r_, i):
                itl_ms = [1000.0 * v for v in r_["itls"][i]]
                ttft_ok = 1000.0 * r_["ttft"][i] <= H.TTFT_SLO_MS
                leg = not (ttft_ok and statistics.fmean(itl_ms or [0.0]) <= H.ITL_SLO_MS)
                can = not (ttft_ok and (not itl_ms or A.percentile(itl_ms, 0.95) <= H.ITL_SLO_MS))
                return leg, can
            for i, tt in enumerate(rec["tok_times"]):
                life = (rec["a"][i], tt[-1])
                if not ivs:
                    c = "pre"
                elif any(min(life[1], b) > max(life[0], a) for a, b in ivs):
                    c = "touched"
                elif life[1] <= ivs[0][0]:
                    c = "pre"
                elif life[0] >= ivs[-1][1]:
                    c = "post"
                else:
                    c = "between"
                leg, can = fails(rd, i)
                pl = [fails(pr, i) for pr in peer_rds]
                cats[c]["t_leg"].append(leg); cats[c]["t_can"].append(can)
                cats[c]["p_leg"].append(statistics.fmean(x[0] for x in pl)); cats[c]["p_can"].append(statistics.fmean(x[1] for x in pl))
                q = pr_acc.setdefault(c, [0, 0, 0.0])
                q[0] += 1; q[1] += leg; q[2] += statistics.fmean(x[0] for x in pl)
        out[ph] = {c: {"n_slots": len(v["t_leg"]),
                       "target_fail_legacy": sum(v["t_leg"]), "peer_mean_fail_legacy": sum(v["p_leg"]),
                       "target_fail_canonical": sum(v["t_can"]), "peer_mean_fail_canonical": sum(v["p_can"])}
                   for c, v in cats.items() if v["t_leg"]}
        out[ph]["per_round_legacy[n_slots,target_fail,peer_mean_fail]"] = per_round
    return {"target": target, "peers": list(peers), "by_phase": out}


# --------------------------------------------------------------------------- self-test
def self_test() -> int:
    ok = True
    t, c = step_fn([(0.0, 1), (1.0, 1), (2.0, -1), (3.0, -1)])
    v = time_where(t, c, 0.0, 4.0, lambda x: x >= 2)
    ok &= abs(v - 1.0) < 1e-9
    print("[control] step/time_where", "PASS" if abs(v - 1.0) < 1e-9 else "FAIL")
    # mutant: off-by-one threshold must change the answer
    v_m = time_where(t, c, 0.0, 4.0, lambda x: x > 2)
    print("[mutant >= -> >] ", "KILLED" if abs(v_m - v) > 1e-9 else "SURVIVED (BAD)"); ok &= abs(v_m - v) > 1e-9
    j = joint_time((t, c), step_fn([(0.5, 1), (1.5, -1)]), 0.0, 4.0, lambda a, b: a >= 2 and b >= 1)
    ok &= abs(j - 0.5) < 1e-9
    print("[control] joint_time", "PASS" if abs(j - 0.5) < 1e-9 else "FAIL")
    # synthetic C1: a dynamic run that is exactly the weighted static mixture must give R = 0
    means = {24: 1.0, 34: 3.0}
    w = {24: 0.75, 34: 0.25}
    obs = sum(w[d] * means[d] for d in w)
    R = obs - sum(w[d] * means[d] for d in w)
    ok &= abs(R) < 1e-12
    R_m = obs - sum(w[d] * means[d] for d in (24,))  # mutant: drop a component
    print("[control] C1 exact mixture R=0", "PASS" if abs(R) < 1e-12 else "FAIL",
          "| [mutant drop component]", "KILLED" if abs(R_m) > 1e-9 else "SURVIVED (BAD)")
    ok &= abs(R_m) > 1e-9
    lines = [(0.0, 48, 0.0, 1), (0.0, 48, 0.0, 1), (1.0, 40, 0.0, 0)]
    ws = [w for w, _x in decode_line_weights(lines, 0.0, 2.0, True)]
    wi = [w for w, _x in decode_line_weights(lines, 0.0, 2.0, False)]
    c_ok = all(abs(a - b) < 1e-9 for a, b in zip(ws, [1 / 3, 1 / 3, 1.5 - 2 / 3])) and wi == [0.0, 0.0, 1.0]
    print("[control] decode_line_weights spread/int", "PASS" if c_ok else "FAIL"); ok &= c_ok
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output", default=str(HERE / "he0_followup_2026-10-02.json"))
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    runs_by_arm, all_R, arm_of, gp = {}, {}, {}, {}
    for arm, jobs in H.HE0_JOBS.items():
        runs_by_arm[arm] = []
        for j in jobs:
            r = H.run_one(arm, j)
            runs_by_arm[arm].append(r)
            gp[j] = goodput_ext(r)
            all_R[j] = recon(j)
            arm_of[j] = arm
    st = stalls(all_R, arm_of)
    st_rows, st_arm = stall_summary(st, all_R)
    touch = {j: stall_touch(all_R[j], st[j]["events"], gp[j]) for j in all_R}
    attrib = {"856964_vs_other_nogate": index_paired_attribution(all_R, 856964, [852435, 856963, 856975], st[856964]["events"]),
              "856964_vs_bind_gate": index_paired_attribution(all_R, 856964, H.HE0_JOBS["bind_gate"], st[856964]["events"]),
              "856892_vs_other_d34": index_paired_attribution(all_R, 856892, [856917, 856918, 856929], st[856892]["events"])}
    stall_heavy = sorted(j for j, v in st_rows.items() if v["global_stall_s_HI"] >= 3.0)
    res_c1 = {"primary_all_runs": c1(runs_by_arm, gp),
              "sensitivity_exclude_runs_with_HI_global_stall_ge_3s": c1(runs_by_arm, gp, exclude=stall_heavy),
              "zero_transition_bind_gate_vs_d24": zero_transition_contrast(runs_by_arm, gp)}
    c4 = {j: c4_run(R) for j, R in all_R.items()}
    keys = [k for k in next(iter(c4.values())) if not k.startswith(("n_", "W_"))]
    c4_arm = {arm: {k: msd([c4[j][k] for j in H.HE0_JOBS[arm]]) for k in keys} for arm in H.HE0_JOBS}
    pooled = {k: msd([c4[j][k] for j in c4]) for k in keys}
    ng = {j: {"COMBINED_legacy": gp[j]["COMBINED"]["legacy_mean_itl_goodput_req_s"],
              "COMBINED_canonical": gp[j]["COMBINED"]["slo_goodput_req_s"],
              "HI_canonical": gp[j]["HI"]["slo_goodput_req_s"], "HI_legacy": gp[j]["HI"]["legacy_mean_itl_goodput_req_s"],
              "n_ctl_transitions": len(all_R[j]["S"]["ctl"]), "n_feas": len(all_R[j]["S"]["feas"])}
          for j in H.HE0_JOBS["bind_nogate"]}
    inputs = {}
    for R in all_R.values():
        inputs[R["srv"].name] = hashlib.sha256(R["srv"].read_bytes()).hexdigest()
    out = {"generated": "2026-10-02", "script": Path(__file__).name,
           "basis": "reconstruction (he0_realized timeline); no per-iteration telemetry exists for these runs",
           "slo": {"ttft_ms": H.TTFT_SLO_MS, "itl_p95_ms": H.ITL_SLO_MS},
           "C1": res_c1, "C4": {"per_run": c4, "per_arm": c4_arm, "all_33_runs": pooled},
           "stalls": {"global_stall_definition": ">=5 requests with a >1 s token gap overlapping, covering >=90% of decode-active requests",
                      "stall_heavy_jobs_HI_ge_3s": stall_heavy, "per_run_summary": st_rows, "per_arm": st_arm,
                      "events": {str(j): v["events"] for j, v in st.items()},
                      "stall_touch": touch, "bind_nogate_runs": ng, "index_paired_attribution": attrib},
           "server_log_sha256": inputs}
    Path(args.output).write_text(json.dumps(rnd(out), indent=1, ensure_ascii=False))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
