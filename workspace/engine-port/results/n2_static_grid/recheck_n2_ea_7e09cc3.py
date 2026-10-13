#!/usr/bin/env python3
"""Independent re-aggregation of N2 Job-EA (pdmux-n2-static-grid-7e09cc3-20261013022149).

result-analyst, 2026-10-13. Reads ONLY the Job-EA output directory and its vessl/<job>/meta.
Does NOT import n2_probe_reduce / n2_window / n2_analyze (independent re-implementation of the
registered window definitions, PREREG rev18 §6-1, §6-2, §7-2, §10-2) and does NOT touch Job-PROFILE.

Outputs (stdout + JSON next to this script):
  * per pair: ln rho_off, ln rho_on (recomputed) vs runner N2_EA_RESULT.json
  * per window (A1, C, A2): S-per median, S-dev median (t_sync - t_launch), host gap median
    (t_launch[i+1] - t_sync[i]), kernel_done_wait fraction, SM clock mean, SW-power-cap fraction,
    mean GPU power (20 ms sampler)
  * C decomposition by prefill_kernel_done_wait (issue iterations vs wait iterations)
  * clock-matched sensitivity: rho recomputed using only C iterations whose nearest clk.csv sample
    is 1410 MHz (no cap) -- descriptive only, not a registered quantity.

Usage: python3 -I recheck_n2_ea_7e09cc3.py <ea_dir> <vessl_meta_dir> <out_json>
"""
import bisect
import datetime as dt
import json
import math
import statistics as st
import sys
from collections import defaultdict

TRIM, WIN, CO_IN = 16, 128, 8192
PAIR = {16: (92, 16), 44: (64, 44), 108: (0, 108)}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def marks(path):
    m = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] in ("cell_start", "cell_end", "coload_sent"):
            m[p[1]][p[0]] = float(p[-1])
    return m


def walloff(path):
    off = {}
    for line in open(path):
        p = line.split()
        if p[0] == "cell_start":
            off[p[1]] = float(p[2]) - float(p[3])
    return off


def ts(e):
    return e.get("timestamp_monotonic_s", e.get("t_launch"))


def sper(win):
    by = {e["iter_seq"]: e for e in win}
    return [(by[q + 1]["t_launch"] - e["t_launch"]) * 1e3 for q, e in sorted(by.items()) if q + 1 in by]


def hostgap(win):
    by = {e["iter_seq"]: e for e in win}
    return [(by[q + 1]["t_launch"] - e["t_sync"]) * 1e3 for q, e in sorted(by.items()) if q + 1 in by]


def sdev(win):
    return [(e["t_sync"] - e["t_launch"]) * 1e3 for e in win]


def windows(recs, t0, t1, tco, d, b, c, pi_alone):
    rs = [e for e in recs if t0 <= ts(e) <= t1]
    it = sorted((e for e in rs if e["event"] == "decode_iteration"), key=lambda e: e["iter_seq"])
    sn = sorted((s["decode_iterations"], s["context_p50"]) for s in rs
                if s["event"] == "runtime_snapshot" and s.get("active_decode_sequences") == b)
    S = {e["span_seq"]: e for e in rs if e["event"] == "prefill_span_start"}
    E = {e["span_seq"]: e for e in rs if e["event"] == "prefill_span_end"}
    spans = [(S[k]["timestamp_monotonic_s"], E[k]["timestamp_monotonic_s"], S[k], E[k]) for k in S if k in E]
    co = [s for s in spans if s[0] >= tco and s[2]["extend_tokens"] == s[2]["n_new"] * CO_IN]
    burst_end = max((s[1] for s in spans if s[0] < tco), default=-1e18)
    tc0, tc1 = min(s[0] for s in co), max(s[1] for s in co)
    ctx, k = {}, -1
    for e in it:
        while k + 1 < len(sn) and sn[k + 1][0] <= e["iter_seq"]:
            k += 1
        if k >= 0:
            ctx[e["iter_seq"]] = sn[k][1] + e["iter_seq"] - sn[k][0]
    seg = [e for e in it if burst_end < e["t_launch"] < tc0 and e["bs"] == b and not e["prefill_in_flight"]][TRIM:]
    a1 = [e for e in seg if e["iter_seq"] in ctx and c - 64 <= ctx[e["iter_seq"]] <= c + 63]
    wco = [e for e in it if tc0 <= e["t_launch"] <= tc1]
    call = [e for e in wco if e["bs"] == b and e["prefill_in_flight"] and (e["prefill_sms"], e["decode_sms"]) == PAIR[d]]
    cw = call[:WIN]
    a2 = [e for e in it if e["t_launch"] > tc1 and e["bs"] == b and not e["prefill_in_flight"]][TRIM:TRIM + WIN]
    cm = lambda w: st.mean(ctx[e["iter_seq"]] for e in w if e["iter_seq"] in ctx)
    pe_a = lambda w: sum((e["prefill_sms"], e["decode_sms"]) == pi_alone for e in w) / len(w)
    info = dict(n_a1=len(a1), n_c=len(cw), n_a2=len(a2), n_call=len(call), w_co=len(wco),
                bs_ne_b=sum(e["bs"] != b for e in wco), bs_gt_b=sum(e["bs"] > b for e in wco),
                pe_a1=pe_a(a1), pe_a2=pe_a(a2),
                pe_coloc=len(call) / max(1, sum(e["bs"] == b for e in wco)),
                co_spans=len(co), n_new=[s[2]["n_new"] for s in sorted(co, key=lambda s: s[0])],
                s3_running_bs=[s[2]["running_bs"] for s in sorted(co, key=lambda s: s[0])],
                s3_prev_done=[s[3]["n_done"] for s in sorted(co, key=lambda s: s[0])])
    return a1, cw, a2, call, wco, cm, info


def interp(y1, x1, y2, x2, x):
    return (y1 + y2) / 2 if x1 == x2 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)


class Clock:
    def __init__(self, meta):
        self.t, self.mhz, self.cap, self.pw = [], [], [], []
        for i, line in enumerate(open(f"{meta}/clk.csv")):
            if i == 0:
                continue
            p = [x.strip() for x in line.split(",")]
            t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            self.t.append(t); self.mhz.append(float(p[1].split()[0]))
            self.cap.append(int(p[4], 16) & 0x4 != 0); self.pw.append(float(p[2].split()[0]))
        self.pt, self.pv = [], []
        for i, line in enumerate(open(f"{meta}/power_samples.csv")):
            if i == 0:
                continue
            a, v = line.split(",")
            self.pt.append(int(a) / 1e6); self.pv.append(int(v) / 1e3)

    def window(self, w0, w1):
        i, j = bisect.bisect_left(self.t, w0), bisect.bisect_right(self.t, w1)
        pi, pj = bisect.bisect_left(self.pt, w0), bisect.bisect_right(self.pt, w1)
        n = j - i
        return dict(n=n, mhz=st.mean(self.mhz[i:j]) if n else None, cap=sum(self.cap[i:j]) / n if n else None,
                    power_w=st.mean(self.pv[pi:pj]) if pj > pi else None)

    def nearest_mhz(self, w):
        i = bisect.bisect_left(self.t, w)
        cands = [k for k in (i - 1, i) if 0 <= k < len(self.t)]
        k = min(cands, key=lambda k: abs(self.t[k] - w))
        return self.mhz[k]


def wstat(w, clk, off):
    sp = sper(w)
    out = dict(n=len(w), s_per_med=st.median(sp), s_per_p95=sorted(sp)[int(0.95 * (len(sp) - 1))],
               s_dev_med=st.median(sdev(w)), host_gap_med=st.median(hostgap(w)),
               kdw=sum(e["prefill_kernel_done_wait"] for e in w) / len(w))
    out.update({"clk_" + k: v for k, v in clk.window(w[0]["t_launch"] + off, w[-1]["t_sync"] + off).items()})
    return out


def main(ea, meta, out_json):
    runner = json.load(open(f"{ea}/N2_EA_RESULT.json"))
    clk = Clock(meta)
    cells = [(d, b, c) for d in (16, 44) for b in (8, 32) for c in (2048, 8192)]
    res = {}
    for r in (1, 2, 3, 4):
        for d in (16, 44):
            per = {}
            for kind in ("ON", "OFF"):
                boot = f"r{r}_{kind}{d}"
                recs = load(f"{ea}/tel_{boot}.jsonl"); mk = marks(f"{ea}/marks_{boot}.txt"); off = walloff(f"{ea}/wall_{boot}.txt")
                for (dd, b, c) in cells:
                    if dd != d:
                        continue
                    cid = f"{boot}-ea-b{b}-c{c}"
                    m = mk[cid]
                    a1, cw, a2, call, wco, cm, info = windows(recs, m["cell_start"], m["cell_end"], m["coload_sent"], d, b, c,
                                                            PAIR[d] if kind == "ON" else PAIR[108])
                    o = off[cid]
                    rec = dict(info=info, A1=wstat(a1, clk, o), C=wstat(cw, clk, o), A2=wstat(a2, clk, o),
                               ctx=dict(A1=cm(a1), C=cm(cw), A2=cm(a2)))
                    iss = [e for e in cw if not e["prefill_kernel_done_wait"]]; wt = [e for e in cw if e["prefill_kernel_done_wait"]]
                    by = {e["iter_seq"]: e for e in cw}
                    sp_iss = [(by[e["iter_seq"] + 1]["t_launch"] - e["t_launch"]) * 1e3 for e in iss if e["iter_seq"] + 1 in by]
                    sp_wt = [(by[e["iter_seq"] + 1]["t_launch"] - e["t_launch"]) * 1e3 for e in wt if e["iter_seq"] + 1 in by]
                    rec["C_issue"] = dict(n=len(sp_iss), s_per_med=st.median(sp_iss) if sp_iss else None,
                                          s_dev_med=st.median(sdev(iss)) if iss else None)
                    rec["C_wait"] = dict(n=len(sp_wt), s_per_med=st.median(sp_wt) if sp_wt else None,
                                         s_dev_med=st.median(sdev(wt)) if wt else None)
                    # descriptive: C intervals whose start iteration sits at an uncapped (1410 MHz) clk sample
                    sp_full = [(by[e["iter_seq"] + 1]["t_launch"] - e["t_launch"]) * 1e3 for e in cw
                               if e["iter_seq"] + 1 in by and clk.nearest_mhz(e["t_launch"] + o) >= 1410]
                    rec["C_uncapped"] = dict(n=len(sp_full), s_per_med=st.median(sp_full) if len(sp_full) >= 10 else None)
                    # descriptive: whole C_all
                    rec["C_all_s_per_med"] = st.median(sper(call))
                    per[(kind, b, c)] = rec
            for (dd, b, c) in cells:
                if dd != d:
                    continue
                on, offr = per[("ON", b, c)], per[("OFF", b, c)]
                a_on = lambda x: interp(on["A1"]["s_per_med"], on["ctx"]["A1"], on["A2"]["s_per_med"], on["ctx"]["A2"], x)
                key = f"{d},{b},{c}"
                lr_on = math.log(on["C"]["s_per_med"] / a_on(on["ctx"]["C"]))
                lr_off = math.log(offr["C"]["s_per_med"] / a_on(offr["ctx"]["C"]))
                # S-dev based ratio (descriptive): device-side drain time, C vs A_on (interp of S-dev medians)
                sdev_on = lambda x: interp(on["A1"]["s_dev_med"], on["ctx"]["A1"], on["A2"]["s_dev_med"], on["ctx"]["A2"], x)
                lr_on_dev = math.log(on["C"]["s_dev_med"] / sdev_on(on["ctx"]["C"]))
                lr_off_dev = math.log(offr["C"]["s_dev_med"] / sdev_on(offr["ctx"]["C"]))
                unc = lambda rr: (math.log(rr["C_uncapped"]["s_per_med"] / a_on(rr["ctx"]["C"]))
                                  if rr["C_uncapped"]["s_per_med"] else None)
                rr = runner["cells"][key]
                res.setdefault(key, {})[r] = dict(
                    ln_rho_on=lr_on, ln_rho_off=lr_off,
                    runner_ln_rho_on=rr["rho_on"]["pairs"][str(r)]["ln_rho"],
                    runner_ln_rho_off=rr["rho_off"]["pairs"][str(r)]["ln_rho"],
                    ln_rho_on_sdev=lr_on_dev, ln_rho_off_sdev=lr_off_dev,
                    ln_rho_on_uncapped=unc(on), ln_rho_off_uncapped=unc(offr),
                    contained_on=min(on["ctx"]["A1"], on["ctx"]["A2"]) <= on["ctx"]["C"] <= max(on["ctx"]["A1"], on["ctx"]["A2"]),
                    contained_off=min(on["ctx"]["A1"], on["ctx"]["A2"]) <= offr["ctx"]["C"] <= max(on["ctx"]["A1"], on["ctx"]["A2"]),
                    on=on, off=offr)
    json.dump(res, open(out_json, "w"), indent=1, default=str)
    # ---- compact printout
    print("cell r | lnrho_on (runner) | lnrho_off (runner) | sdev-ratio on/off | uncapped on/off")
    for key in sorted(res, key=lambda k: tuple(map(int, k.split(",")))):
        for r, v in sorted(res[key].items()):
            f = lambda x: "  NA " if x is None else f"{x:.4f}"
            print(key, r, f(v["ln_rho_on"]), f(v["runner_ln_rho_on"]), f(v["ln_rho_off"]), f(v["runner_ln_rho_off"]),
                  f(v["ln_rho_on_sdev"]), f(v["ln_rho_off_sdev"]), f(v["ln_rho_on_uncapped"]), f(v["ln_rho_off_uncapped"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
