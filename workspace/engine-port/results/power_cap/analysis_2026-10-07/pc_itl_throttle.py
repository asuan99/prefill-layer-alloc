#!/usr/bin/env python3
"""Rung A(3): is a decode step slower while the SW power cap (0x4) is active, holding batch size and
prefill co-residency fixed?  EXPLORATORY - not citable in canon.

Data: N3 r3 units with PHASE_EVENTS=1 (grid/floor): `decode_iteration` events (t_launch, t_sync in
CLOCK_MONOTONIC seconds; bs; prefill_in_flight) joined to `meta/clk.csv` (wall clock, 100 ms).

Alignment (monotonic -> epoch), per unit: SGLang logs "Decode batch ..." every DECODE_LOG_INTERVAL (=40)
decode forward passes with a wall-clock [YYYY-mm-dd HH:MM:SS] stamp (second resolution); the k-th line is
paired with telemetry iteration iter_seq = 40k (t_sync, CLOCK_MONOTONIC).  With O = epoch - mono constant
over one unit and the stamp truncating to whole seconds, every est_k = T_k - t_sync_k lies in
(O - 1 + d, O + d] (d = log latency, ms), so  O in [max_k est_k, min_k est_k + 1)  and the bracket is a
few 10s of ms wide once ~40 lines are paired.  A unit whose est spread is >= 1 s (pairing broken) is
dropped.  The runner's marks/T_BS pair (n3_capacity.sh:453-454) is only used as a coarse cross-check:
T_BS - mono_start - O must fall in (-1, 1) s (second truncation + python start-up latency of `mono`).

Strata: bs bin x prefill_in_flight.  Within each stratum: median step time (t_sync - t_launch) and
median iteration period with vs without 0x4.  Iterations within `guard` s of a 0x4 state change are
dropped (sample-alignment edge).  The co-residency contrast (prefill_in_flight T vs F at 0x4 == 0) is
reported alongside as the known SM-channel reference.

    python3 pc_itl_throttle.py --all [--guard 0.3] [--cache-dir DIR] | --selftest
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import re
import json
import math
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_common as C  # noqa: E402

BS_BINS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 6)]
SM_BINS = [(1400, 10 ** 6), (1300, 1399), (0, 1299)]
MIN_N = 30
FIELDS = ("t_launch", "t_sync", "bs", "prefill_in_flight", "prefill_kernel_done_wait", "prefill_sms", "decode_sms",
          "queue_len", "iter_seq")


def bs_bin(bs):
    for lo, hi in BS_BINS:
        if lo <= bs <= hi:
            return f"bs{lo}-{hi if hi < 10**6 else '+'}"
    return "bs?"


def sm_bin(sm):
    for lo, hi in SM_BINS:
        if lo <= sm <= hi:
            return f"sm{lo}-{hi if hi < 10**6 else '+'}"
    return "sm?"


# ---------------------------------------------------------------- alignment
def read_marks(path: Path):
    out = {}
    for ln in open(path):
        p = ln.split()
        if len(p) == 2:
            out[p[0]] = float(p[1])
    return out


DECODE_LOG_INTERVAL = 40
SRV_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")


def srv_decode_lines(path: Path):
    out = []
    for line in open(path, errors="replace"):
        if "Decode batch" in line:
            m = SRV_TS.match(line)
            if m:
                out.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
    return out


def unit_offset(srv_log: Path, iters, marks: dict, timing_row: dict):
    """Per-unit O = epoch - mono from srv.log decode lines paired with iter_seq = 40k."""
    lines = srv_decode_lines(srv_log)
    byseq = {int(e["iter_seq"]): float(e["t_sync"]) for e in iters if e.get("iter_seq") is not None}
    ests = []
    for k, T in enumerate(lines, start=1):
        t = byseq.get(k * DECODE_LOG_INTERVAL)
        if t is not None:
            ests.append(T - t)
    if len(ests) < 5:
        return {"ok": False, "reason": f"only {len(ests)} paired lines", "n_lines": len(lines)}
    lo, hi = max(ests), min(ests) + 1.0
    rec = {"ok": lo < hi, "n_paired": len(ests), "n_lines": len(lines), "O_lower": lo, "O_upper": hi,
           "O_hat": 0.5 * (lo + hi), "half_width_s": 0.5 * (hi - lo), "spread_s": max(ests) - min(ests)}
    if "bench_start" in marks and "a" in timing_row:
        rec["marks_check_s"] = timing_row["a"] - marks["bench_start"] - rec["O_hat"]   # expect (-1, 1): floor + mono() latency
    if not rec["ok"]:
        rec["reason"] = "est spread >= 1 s (pairing broken)"
    return rec


# ---------------------------------------------------------------- extraction (cached)
def extract_unit(tel: Path, marks: dict, cache: Path):
    if cache.exists():
        return [json.loads(l) for l in open(cache)]
    lo, hi = marks.get("bench_start", -math.inf), marks.get("bench_end", math.inf)
    out = []
    with open(tel) as f:
        for line in f:
            if '"decode_iteration"' not in line:
                continue
            e = json.loads(line)
            if e.get("event") != "decode_iteration":
                continue
            t = float(e["t_launch"])
            if lo <= t <= hi:
                out.append({k: e.get(k) for k in FIELDS})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(o) + "\n" for o in out))
    return out


# ---------------------------------------------------------------- join + strata
def join(iters, clk, O, guard, max_gap=0.2):
    ts = [r[0] for r in clk]
    changes = []
    for i in range(1, len(clk)):
        if (clk[i][4] & C.BIT_SW_POWER_CAP) != (clk[i - 1][4] & C.BIT_SW_POWER_CAP):
            changes.append(clk[i][0])
    joined, dropped_gap, dropped_edge = [], 0, 0
    prev_launch = None
    for e in iters:
        tl, tsy = float(e["t_launch"]), float(e["t_sync"])
        period = (tl - prev_launch) if prev_launch is not None else None
        prev_launch = tl
        tm = 0.5 * (tl + tsy) + O
        i = bisect.bisect_left(ts, tm)
        cands = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        if not cands:
            dropped_gap += 1; continue
        j = min(cands, key=lambda k: abs(ts[k] - tm))
        if abs(ts[j] - tm) > max_gap:
            dropped_gap += 1; continue
        k = bisect.bisect_left(changes, tm)
        near = min((abs(changes[q] - tm) for q in (k - 1, k) if 0 <= q < len(changes)), default=math.inf)
        if near < guard:
            dropped_edge += 1; continue
        joined.append({"step_ms": 1000.0 * (tsy - tl), "period_ms": 1000.0 * period if period else None,
                       "bs": int(e.get("bs") or 0), "co": bool(e.get("prefill_in_flight")),
                       "thr": bool(clk[j][4] & C.BIT_SW_POWER_CAP), "sm": clk[j][1], "power_W": clk[j][2]})
        # note: period_ms (launch-to-launch) includes scheduler gaps and prefill spans; kept in the join only
    return joined, {"n_in": len(iters), "n_joined": len(joined), "dropped_gap": dropped_gap, "dropped_edge": dropped_edge,
                    "n_0x4_changes": len(changes)}


def strata_table(joined):
    g = defaultdict(list)
    for r in joined:
        g[(bs_bin(r["bs"]), r["co"], r["thr"])].append(r)
    rows = []
    for (b, co) in sorted({(k[0], k[1]) for k in g}, key=lambda x: (BS_BINS.index(next(bb for bb in BS_BINS if bs_bin(bb[0]) == x[0])), x[1])):
        a, t = g.get((b, co, False), []), g.get((b, co, True), [])
        row = {"bs_bin": b, "co_resident": co, "n_thr0": len(a), "n_thr1": len(t)}
        if a:
            row["step_med_thr0"] = statistics.median(x["step_ms"] for x in a)
            row["sm_mean_thr0"] = statistics.fmean(x["sm"] for x in a)
        if t:
            row["step_med_thr1"] = statistics.median(x["step_ms"] for x in t)
            row["sm_mean_thr1"] = statistics.fmean(x["sm"] for x in t)
        if len(a) >= MIN_N and len(t) >= MIN_N:
            row["ratio_step"] = row["step_med_thr1"] / row["step_med_thr0"]
        rows.append(row)
    return rows


def pooled_ratio(rows, key="ratio_step"):
    num = den = 0.0
    for r in rows:
        if key in r:
            w = 2.0 / (1.0 / r["n_thr0"] + 1.0 / r["n_thr1"])
            num += w * math.log(r[key]); den += w
    return math.exp(num / den) if den else None


def coresidency_reference(joined):
    """Known SM-channel reference: co-resident vs not, at thr == 0, same bs bins."""
    g = defaultdict(list)
    for r in joined:
        if not r["thr"]:
            g[(bs_bin(r["bs"]), r["co"])].append(r["step_ms"])
    rows = []
    for b in {k[0] for k in g}:
        a, c = g.get((b, False), []), g.get((b, True), [])
        if len(a) >= MIN_N and len(c) >= MIN_N:
            rows.append({"bs_bin": b, "n_co0": len(a), "n_co1": len(c), "ratio_step": statistics.median(c) / statistics.median(a),
                         "n_thr0": len(a), "n_thr1": len(c)})
    return rows


def sm_conditioned(joined):
    g = defaultdict(list)
    for r in joined:
        g[(bs_bin(r["bs"]), r["co"], sm_bin(r["sm"]))].append(r["step_ms"])
    rows = []
    for (b, co, s), v in sorted(g.items()):
        if len(v) >= MIN_N:
            rows.append({"bs_bin": b, "co_resident": co, "sm_bin": s, "n": len(v), "step_med_ms": statistics.median(v)})
    return rows


def analyze_job(out_dir: Path, clk_path: Path, cache_dir: Path, guard: float, units_filter=None):
    timing = C.read_timing(out_dir / "TIMING.tsv", "unit")
    clk = C.parse_clk(clk_path)
    per_unit, all_joined, offsets, skipped = {}, [], {}, {}
    for u, w in timing.items():
        if units_filter and u not in units_filter:
            continue
        tel, mp, srv = out_dir / f"tel_{u}.jsonl", out_dir / f"marks_{u}.txt", out_dir / f"srv_{u}.log"
        if not (tel.exists() and mp.exists() and srv.exists()):
            continue
        marks = read_marks(mp)
        iters = extract_unit(tel, marks, cache_dir / f"dec_{u}.jsonl")
        if not iters:
            skipped[u] = "no decode_iteration events (PHASE_EVENTS off)"; continue
        off = unit_offset(srv, iters, marks, w)
        offsets[u] = off
        if not off["ok"]:
            skipped[u] = off["reason"]; continue
        guard_eff = guard + off["half_width_s"]
        joined, js = join(iters, clk, off["O_hat"], guard_eff)
        rows = strata_table(joined)
        meta = C.unit_meta("n3_r3", u)
        per_unit[u] = {"meta": meta, "offset": off, "guard_s": guard_eff, "join": js,
                       "thr_frac": (sum(r["thr"] for r in joined) / len(joined)) if joined else None,
                       "pooled_ratio_step": pooled_ratio(rows),
                       "strata": rows}
        for r in joined:
            r["rate"] = meta.get("rate"); r["unit"] = u
        all_joined += joined
    by_rate = {}
    for rate in sorted({r["rate"] for r in all_joined if r["rate"] is not None}):
        sub = [r for r in all_joined if r["rate"] == rate]
        rows = strata_table(sub)
        by_rate[f"{rate:.2f}"] = {"n": len(sub), "thr_frac": sum(r["thr"] for r in sub) / len(sub),
                                 "pooled_ratio_step": pooled_ratio(rows),
                                 "strata": rows}
    rows_all = strata_table(all_joined)
    ohats = [o["O_hat"] for o in offsets.values() if o.get("ok")]
    hws = [o["half_width_s"] for o in offsets.values() if o.get("ok")]
    mc = [o["marks_check_s"] for o in offsets.values() if o.get("ok") and "marks_check_s" in o]
    return {"offset": {"method": "srv.log 'Decode batch' lines x iter_seq=40k, per unit", "n_units_ok": len(ohats),
                       "O_hat_min": min(ohats) if ohats else None, "O_hat_max": max(ohats) if ohats else None,
                       "half_width_s_max": max(hws) if hws else None,
                       "marks_check_s_min": min(mc) if mc else None, "marks_check_s_max": max(mc) if mc else None},
            "guard_base_s": guard, "skipped": skipped, "n_units": len(per_unit), "n_joined": len(all_joined),
            "thr_frac_all": (sum(r["thr"] for r in all_joined) / len(all_joined)) if all_joined else None,
            "pooled_ratio_step_all": pooled_ratio(rows_all),
            "strata_all": rows_all, "coresidency_reference_thr0": coresidency_reference(all_joined),
            "sm_conditioned": sm_conditioned(all_joined), "by_rate": by_rate, "per_unit": per_unit}


def to_md(res: dict) -> str:
    o = res["offset"]
    out = ["# pc_itl_throttle — decode step time under SW power cap, bs x co-residency strata (EXPLORATORY)", "",
           f"alignment: {o['method']}; units aligned {o['n_units_ok']}; O_hat range {o['O_hat_min']}..{o['O_hat_max']} "
           f"(drift {((o['O_hat_max'] or 0) - (o['O_hat_min'] or 0)):.3f} s over the Job); max half-width "
           f"{(o['half_width_s_max'] or 0)*1000:.0f} ms; marks cross-check (T_BS - mono - O) in "
           f"[{o['marks_check_s_min']}, {o['marks_check_s_max']}] (expect (-1, 1)); base edge guard {res['guard_base_s']:.2f} s "
           f"(+ per-unit half-width); skipped units: {res['skipped']}", "",
           f"joined iterations {res['n_joined']} over {res['n_units']} units; 0x4-active fraction {res['thr_frac_all']:.3f}; "
           f"pooled ratio step(thr1/thr0) = {res['pooled_ratio_step_all']:.4f}", "",
           "## all units — strata"]
    out.append(C.md_table(res["strata_all"], ["bs_bin", "co_resident", "n_thr0", "n_thr1", "step_med_thr0", "step_med_thr1",
                                               "ratio_step", "sm_mean_thr0", "sm_mean_thr1"],
                          {"step_med_thr0": "{:.2f}", "step_med_thr1": "{:.2f}", "ratio_step": "{:.3f}",
                           "sm_mean_thr0": "{:.0f}", "sm_mean_thr1": "{:.0f}"}))
    out.append("\n## reference: co-resident vs not at thr=0 (known SM channel)")
    out.append(C.md_table(res["coresidency_reference_thr0"], ["bs_bin", "n_co0", "n_co1", "ratio_step"], {"ratio_step": "{:.3f}"}))
    out.append("\n## SM-clock-conditioned step time")
    out.append(C.md_table(res["sm_conditioned"], ["bs_bin", "co_resident", "sm_bin", "n", "step_med_ms"], {"step_med_ms": "{:.2f}"}))
    out.append("\n## by offered rate")
    rows = [{"rate": k, **{kk: v[kk] for kk in ("n", "thr_frac", "pooled_ratio_step")}} for k, v in res["by_rate"].items()]
    out.append(C.md_table(rows, ["rate", "n", "thr_frac", "pooled_ratio_step"], {"thr_frac": "{:.3f}", "pooled_ratio_step": "{:.3f}"}))
    out.append("\n## per unit")
    rows = [{"unit": u, "rate": v["meta"].get("rate"), "O_hw_ms": 1000 * v["offset"]["half_width_s"], "n_joined": v["join"]["n_joined"], "dropped_edge": v["join"]["dropped_edge"],
             "thr_frac": v["thr_frac"], "ratio_step": v["pooled_ratio_step"]} for u, v in res["per_unit"].items()]
    out.append(C.md_table(rows, ["unit", "rate", "O_hw_ms", "n_joined", "dropped_edge", "thr_frac", "ratio_step"],
                          {"O_hw_ms": "{:.0f}", "thr_frac": "{:.3f}", "ratio_step": "{:.3f}"}))
    out.append("\nCaveats: observational (0x4 is endogenous to load and to prefill activity); strata control bs and prefill_in_flight "
               "only; NVML reason bits are instantaneous but power is 1 s-averaged; not a decision input; not citable in canon.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- selftest
def _synth_job(td: Path, effect: bool, O_true: float, seed=1, n_units=8, episode_s=5.0):
    import datetime as dt
    rng = random.Random(seed)
    out = td / "out"; out.mkdir(); meta = td / "meta"; meta.mkdir()
    mono0 = 1000.0
    epoch_rows = []
    timing = ["unit\tbench_start_epoch\tbench_end_epoch"]
    unit_len = 60.0
    for k in range(n_units):
        u = f"g_r0{30 + k}_{1000 + k}"
        ms = mono0 + k * (unit_len + 10.0) + rng.random()     # real marks have random fractional parts
        me = ms + unit_len
        frac_s, frac_e = rng.random() * 0.9, rng.random() * 0.9
        T_BS = math.floor(ms + O_true + frac_s * 0.01 + 0.0)      # date taken 0-9 ms after the mark
        T_BE = math.floor(me + O_true - frac_e * 0.01)             # date taken 0-9 ms before the mark
        (out / f"marks_{u}.txt").write_text(f"bench_start {ms}\nbench_end {me}\n")
        timing.append(f"{u}\t{T_BS}\t{T_BE}")
        lines = []
        t = ms + 0.5
        seq = 0
        while t < me - 0.5:
            bs = rng.choice([1, 2, 3, 6, 10, 20, 30, 40])
            co = rng.random() < 0.3
            thr = int(((t + O_true) // episode_s) % 2) == 1
            base = (10.0 + 0.2 * bs) * (1.15 if co else 1.0)
            step = base * (1.25 if (thr and effect) else 1.0) * (1 + rng.uniform(-0.03, 0.03))
            seq += 1
            lines.append(json.dumps({"event": "decode_iteration", "t_launch": t, "t_sync": t + step / 1000.0, "bs": bs,
                                     "prefill_in_flight": co, "prefill_kernel_done_wait": False, "prefill_sms": 64,
                                     "decode_sms": 44, "queue_len": 0, "iter_seq": seq, "timestamp_monotonic_s": t + step / 1000.0}))
            lines.append(json.dumps({"event": "runtime_snapshot", "timestamp_monotonic_s": t}))
            t += step / 1000.0 + 0.002
        (out / f"tel_{u}.jsonl").write_text("\n".join(lines) + "\n")
        srv = []
        for ln in lines:
            e = json.loads(ln)
            if e.get("event") == "decode_iteration" and e["iter_seq"] % DECODE_LOG_INTERVAL == 0:
                T = math.floor(e["t_sync"] + O_true + 0.002)
                srv.append(f"[{dt.datetime.fromtimestamp(T, dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Decode batch, #running-req: 1")
        (out / f"srv_{u}.log").write_text("\n".join(srv) + "\n")
    (out / "TIMING.tsv").write_text("\n".join(timing) + "\n")
    t0 = mono0 + O_true - 5.0
    tend = mono0 + n_units * (unit_len + 10.0) + O_true + 5.0
    clk = ["timestamp, clocks.current.sm [MHz], power.draw [W], temperature.gpu, clocks_event_reasons.active"]
    t = t0
    while t < tend:
        thr = int((t // episode_s) % 2) == 1
        s = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        clk.append(f"{s}, {1300 if thr else 1410} MHz, {400.0 if thr else 300.0:.2f} W, 50, {'0x0000000000000004' if thr else '0x0000000000000000'}")
        t += 0.1
    (meta / "clk.csv").write_text("\n".join(clk) + "\n")
    return out, meta / "clk.csv"


def selftest() -> int:
    fails = []
    O_true = 1791000000.3217
    with tempfile.TemporaryDirectory() as d:
        for effect, lo, hi in ((True, 1.18, 1.32), (False, 0.97, 1.03)):
            td = Path(d) / ("eff" if effect else "null"); td.mkdir()
            out, clk = _synth_job(td, effect, O_true)
            res = analyze_job(out, clk, td / "cache", guard=0.15)
            o = res["offset"]
            if not (o["n_units_ok"] == 8 and abs(o["O_hat_min"] - O_true) < 0.03 and abs(o["O_hat_max"] - O_true) < 0.03
                    and o["half_width_s_max"] < 0.05 and -1.0 < o["marks_check_s_min"] and o["marks_check_s_max"] < 1.0):
                fails.append(f"offset {o} vs true {O_true}")
            r = res["pooled_ratio_step_all"]
            if r is None or not (lo <= r <= hi):
                fails.append(f"effect={effect}: pooled ratio {r} not in [{lo},{hi}]")
            if effect:
                ref = res["coresidency_reference_thr0"]
                pr = pooled_ratio(ref)
                if pr is None or not (1.08 <= pr <= 1.22):
                    fails.append(f"co-residency reference pooled {pr} not ~1.15")
                # mutation: a 5 s misalignment (one full episode -> labels inverted) must INVERT the ratio
                clkrows = C.parse_clk(clk)
                tim = C.read_timing(out / "TIMING.tsv", "unit")
                allj = []
                for u in tim:
                    it = extract_unit(out / f"tel_{u}.jsonl", read_marks(out / f"marks_{u}.txt"), td / "cache" / f"dec_{u}.jsonl")
                    j, _ = join(it, clkrows, res["per_unit"][u]["offset"]["O_hat"] + 5.0, 0.3)
                    allj += j
                rm = pooled_ratio(strata_table(allj))
                if rm is None or not (0.75 <= rm <= 0.9):
                    fails.append(f"misaligned (inverted-label) mutant ratio {rm} not in [0.75,0.9]")
    print("SELFTEST", "PASS" if not fails else "FAIL", fails)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--guard", type=float, default=0.15)
    ap.add_argument("--units", nargs="*")
    ap.add_argument("--cache-dir", default=str(C.HERE / "cache"))
    ap.add_argument("--out-dir", default=str(C.HERE / "out"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    cfg = C.VESSL["n3_r3"]
    cache = Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    (cache / ".gitignore").write_text("*\n")
    res = analyze_job(cfg["out"], cfg["clk"], cache, a.guard, set(a.units) if a.units else None)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    C.dump(res, od / "pc_itl_throttle.json")
    (od / "pc_itl_throttle.md").write_text(to_md(res))
    print(json.dumps({k: res[k] for k in ("offset", "skipped", "n_units", "n_joined", "thr_frac_all",
                                          "pooled_ratio_step_all")}, indent=1))


if __name__ == "__main__":
    main()
