#!/usr/bin/env python3
"""Cost-term harness -- Job time model of PREREG §7-1 / §7-4' [유도, 측정 아님; Job sizing only, every constant is the
PREREG's arbitrary model value].  `python3 ct_timemodel.py` prints hours per Job and the derived hard caps (x 1.2).

Model (§7-4'): T1 cell = sum of the burst's prefill batches (batch 1 at 108 SM, the rest at the mode's concurrent
prefill SM P, TTFT table x (108/P)^0.8; vanilla all at 108) + 318 steps x (12 + 0.45 bs)(108/D)^0.5 ms + 15.5 s;
T2 = 8 TTFT(L) + 15 s; T2' = 8 TTFT(L)(108/P)^0.8 + 16 s; T3 cell = decode-batch ramp + m(L) TTFT(L, P)(108/P)^0.8
+ 300 steps (+ 15.5 s); T3 reference = 8 TTFT(L, P) + 16 s; long point (decision 30) = 8 TTFT(L) + 15 s; T4 fill
6 min; boot + warm-up 0.072 h (TP2 boot 400 s);
TP2 prefill 1.7x faster, TP2 decode step taken equal to N=1 (no model value given -- conservative).
Feasibility is evaluated at the derived cap-128 / cap-48 pools (or a given pool), so INFEASIBLE cells cost nothing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct_spec as S  # noqa: E402

TTFT_S = {1024: 0.12, 2048: 0.24, 4096: 0.46, 8192: 0.92, 16384: 1.84, 32768: 4.0, 65536: 8.4,   # §7-1
          98304: 14.0, 126976: 18.0}   # decision 30 long points: 128k probe descriptive values (GATE_SPLIT_DESIGN
#                                        time basis "98k 약 14 s, 124k 약 18 s"), budget input only
BOOT_H = 0.072
TP2_BOOT_H = (400 + 60) / 3600
TP2_PF = 1.7
CELL_OVH_S = 15.5
FILL_H = 0.1
POOL_CAP128 = 895781          # §3-1 [유도]
POOL_CAP48 = 1588000          # §5-2 [유도] (33.04 GB - 49 x 143.5 MB) / 16,379 B  [CS-OK] 33.04는 풀 GB 값(우연 일치)
POOL_TP2_CAP128 = 3720000     # [유도] per-rank pool ~ 39.7 GB at k/2, s/2 (§5-1 F3)


def ttft(L):
    ks = sorted(TTFT_S)
    if L <= ks[0]:
        return TTFT_S[ks[0]] * L / ks[0]
    if L >= ks[-1]:
        return TTFT_S[ks[-1]] * L / ks[-1]
    for a, b in zip(ks, ks[1:]):
        if a <= L <= b:
            f = math.log(L / a) / math.log(b / a)
            return math.exp(math.log(TTFT_S[a]) + f * math.log(TTFT_S[b] / TTFT_S[a]))


def step_s(bs, D):
    return (12 + 0.45 * bs) * (108 / D) ** 0.5 / 1000.0


def burst_prefill_s(bs, c, P, vanilla, tp=1):
    c_in = c - S.X_TOKENS
    t = 0.0
    for j, n in enumerate(S.span_batches(bs, c_in)):
        x = ttft(n * c_in)
        if j and not vanilla:
            x *= (108 / P) ** 0.8
        t += x
    return t / (TP2_PF if tp > 1 else 1)


def boot_hours(b, pool=None):
    tp = b.get("tp", 1)
    van = b["engine"] == "vanilla"
    D = b["D"] or 108
    P = 108 - b["D"] if b["engine"] == "pdmux_sticky" else (S.PDMUX_108U_P if b["engine"] == "pdmux_108u" else 108)
    pool = pool or (POOL_TP2_CAP128 if tp > 1 else POOL_CAP128 if b["cap"] == 128 else POOL_CAP48)
    caps = S.default_capture_list(b["cap"])
    pf = 1 / TP2_PF if tp > 1 else 1
    t = 0.0
    for bs, c in [tuple(x) for x in b["t1"]]:
        O = b["t1_O"][f"{bs}x{c}"]
        if S.feasible_cell(bs, c, O, pool, b["cap"], bs in caps, b.get("context")):
            continue
        t += b["t1_reps"] * (burst_prefill_s(bs, c, P, van, tp) + 318 * step_s(bs, D) + CELL_OVH_S)
    t += sum(8 * ttft(L) * pf + 15 for L in b["t2"])
    t += sum(8 * ttft(L) * pf * (108 / P) ** 0.8 + 16 for L in b["t2p"])
    for bs, L in [tuple(x) for x in b.get("t3", [])]:
        if S.feasible_t3(bs, L, b["t3_O"][f"{bs}x{L}"], pool, b["cap"], bs in caps):
            continue
        t += b["t3_reps"] * (burst_prefill_s(bs, S.T3_C, P, False, tp) + S.T3_M[L] * ttft(L) * pf * (108 / P) ** 0.8
                             + 300 * step_s(bs, D) + CELL_OVH_S)
    t += sum(S.T3_REF_REPS * ttft(L) * pf * (108 / P) ** 0.8 + 16 for L in b.get("t3ref", []))
    t += sum(S.LONG_REPS * ttft(L) * pf + 15 for L in b.get("long", []))     # decision 30: T2 rule, 8 reps
    h = t / 3600 + FILL_H * len(b["t4"]) + (TP2_BOOT_H if tp > 1 else BOOT_H)
    return h


def job_hours(plan, pools=None):
    pools = pools or {}
    return sum(boot_hours(b, pools.get(b["boot_id"])) for b in plan["boots"])


def main():
    import ct_runner as R
    rows = {}
    for job, rnd in (("J-A", 1), ("J-A", 2), ("J-H1", 1), ("J-H2", 1), ("J-B2", 1), ("J-C", 1)):
        p = R.make_plan(job, rnd)
        pools = {}
        if job == "J-H2":     # probe pools [인용 불가 설계 입력], compo_probe_20261021T131621Z
            pp = {4: 886804, 13: 419117, 22: 334083, 31: 298407}
            pools = {b["boot_id"]: pp[b["variant"]] for b in p["boots"]}
        h = job_hours(p, pools)
        t3 = sum(boot_hours(b, pools.get(b["boot_id"])) for b in p["boots"] if b["kind"] == "T3")
        rows[f"{job}|{rnd}"] = (h, t3)
        ngpu = 2 if job == "J-C" else 1
        print(f"{job} r{rnd}: {h:.2f} h (T3 boots {t3:.2f} h), hard cap x1.2 = {1.2 * h:.2f} h = "
              f"{math.ceil(1.2 * h * 60)} min, reserved GPU-h {h * ngpu:.2f}")
    return rows


if __name__ == "__main__":
    main()
