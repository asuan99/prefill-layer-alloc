#!/usr/bin/env python3
"""Decode-only power/energy per N2 cell (D, b, c) -- power_cap track path A (CPU only, DESCRIPTIVE).

N2 (3-a) holds decode ALONE on a realized partition D in {16, 24, 34, 44} (sticky ON) or unpartitioned 108, at running
batch b and context c, with PHASE_EVENTS=1.  job_entry.sh (commit 2196642) records board power alongside.  This tool
joins them per cell.  It is NOT an N2 decision input (N2C-61: no same-Job sampler on/off contrast -> no sampler-effect claim).

Inputs (per boot, N2 harness layout -- n2_probe_run.py / n2_marks.py):
  marks_<boot>.txt  `cell_start <id> <d> <b> <c> <t_mono>` ... `cell_end <id> <t_mono>` (CLOCK_MONOTONIC)
  wall_<boot>.txt   `<tag> <id> <time.time()> <time.monotonic()>` pairs -> epoch = mono + offset (median of pairs)
  tel_<boot>.jsonl  decode_iteration events (t_launch, t_sync, bs, decode_sms, prefill_in_flight, iter_seq)
Job meta (job_entry.sh): energy.csv (epoch_s, energy_mJ; 100 ms), power_samples.csv (t_us epoch, power_mW; ~20 ms).

Windows (preferred): N2's OWN registered windows, `n2_probe_reduce.cell_windows` loaded read-only by file path --
  W  = 128 decode-alone iterations before co-load, C = first 128 iterations co-located with the co-load prefill on the SAME
  partition pair, A2 = 128 decode-alone iterations after; each with N2's status (only VALID windows are scored).
  C gives "decode + prefill co-resident at a fixed (P, D)" -- the exclusion contrast W vs C at one partition.
  Fallback (reducer not importable / PROFILE cells without co-load): whole cell, or pre-`coload_sent` part (window=pre_coload).
Per window: P_mean from 20 ms buffered samples (shifted by SENSOR_LAG_S) is PRIMARY (windows are ~1-3 s; the 100 ms counter
  has up to ~one update of boundary error per edge), E_buf = P_mean * T; the counter difference E_ctr is a cross-check.
  E_win  = energy-counter difference (linear interpolation, no extrapolation)            [J]
  P_mean = mean of 20 ms buffered samples in the window shifted by SENSOR_LAG_S (~20 ms)  [W]
  n_steps, step_ms p50/p95, realized bs (mean), realized decode_sms (mode), prefill_in_flight fraction
  J_per_step = E_win / n_steps ; J_per_token = J_per_step / bs_mean
  J_per_step_above_idle = (E_win - P_idle * T_win) / n_steps, P_idle = median buffered power outside every cell window
Caveats: board-level power (no per-partition split); counter boundary error <= ~one 100 ms update per edge (cells are
seconds long); cells with prefill_in_flight > 1 % are flagged NOT_DECODE_ONLY and kept out of the J tables.

    python3 pc_n2_decode_power.py --run-dir <dir with marks/wall/tel> --meta <job meta dir> [--out <dir>]
    python3 pc_n2_decode_power.py --selftest
"""
from __future__ import annotations

import argparse, bisect, glob, json, math, os, random, statistics as st, sys, tempfile
from pathlib import Path

SENSOR_LAG_S = 0.02
N2_REDUCER = Path(__file__).resolve().parents[2] / "n2_static_grid" / "n2_probe_reduce.py"


def load_reducer(path=N2_REDUCER):
    """Read-only use of N2's registered window code (same file N2 loads by path); None if unavailable."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_pc_n2_reducer", str(path))
        m = importlib.util.module_from_spec(spec); sys.path.insert(0, str(Path(path).parent)); spec.loader.exec_module(m)
        return m if hasattr(m, "cell_windows") and hasattr(m, "parse_marks") else None
    except Exception:  # noqa: BLE001
        return None
PREFILL_TOL = 0.01


def read_marks(p):
    cells, cur = {}, None
    for ln in open(p):
        f = ln.split()
        if not f:
            continue
        if f[0] == "cell_start":
            cells[f[1]] = {"id": f[1], "d": int(f[2]), "b": int(f[3]), "c": int(f[4]), "t0": float(f[5])}
        elif f[0] == "cell_end" and f[1] in cells:
            cells[f[1]]["t1"] = float(f[2])
        elif f[0] == "cell_abort" and f[1] in cells:
            cells[f[1]]["aborted"] = f[2]
        elif f[0] == "coload_sent" and f[1] in cells:
            cells[f[1]]["coload_sent"] = float(f[2])
    return [c for c in cells.values() if "t1" in c]


def wall_offset(p):
    """epoch - monotonic from (time.time, time.monotonic) pairs; spread reported (should be << 1 ms)."""
    offs = []
    for ln in open(p):
        f = ln.split()
        if len(f) >= 4:
            offs.append(float(f[2]) - float(f[3]))
    if not offs:
        return None, None
    return st.median(offs), max(offs) - min(offs)


def read_tel(p):
    out = []
    with open(p) as fh:
        for ln in fh:
            if '"decode_iteration"' not in ln:
                continue
            e = json.loads(ln)
            if e.get("event") == "decode_iteration":
                out.append((float(e["t_launch"]), float(e["t_sync"]), int(e.get("bs") or 0), int(e.get("decode_sms") or 0),
                            bool(e.get("prefill_in_flight"))))
    out.sort()
    return out


def read_energy(p):
    rows = []
    for ln in open(p):
        f = ln.strip().split(",")
        if len(f) == 2 and f[0][:1].isdigit():
            rows.append((float(f[0]), int(f[1])))
    return sorted(rows)


def read_samples(p):
    rows = []
    for ln in open(p):
        f = ln.strip().split(",")
        if len(f) == 2 and f[0][:1].isdigit():
            rows.append((int(f[0]) / 1e6, int(f[1]) / 1000.0))
    return sorted(rows)


def ctr_at(E, x):
    ts = [r[0] for r in E]
    i = bisect.bisect_left(ts, x)
    if i == 0 or i >= len(E):
        return None
    (t0, e0), (t1, e1) = E[i - 1], E[i]
    return e0 + (e1 - e0) * (x - t0) / (t1 - t0) if t1 > t0 else float(e0)


def score_window(iters, off, E, S, st_):
    """iters: list of (t_launch, t_sync, bs, decode_sms, prefill_in_flight) in monotonic seconds."""
    a, b = iters[0][0] + off, iters[-1][1] + off
    T = b - a
    si, sj = bisect.bisect_left(st_, a + SENSOR_LAG_S), bisect.bisect_right(st_, b + SENSOR_LAG_S)
    P = st.fmean(r[1] for r in S[si:sj]) if sj > si else None
    ea, eb = ctr_at(E, a), ctr_at(E, b)
    steps = sorted(1000 * (e[1] - e[0]) for e in iters)
    dsm = [e[3] for e in iters]
    out = {"n_steps": len(iters), "T_s": T, "P_mean_W": P, "n_power_samples": sj - si,
           "E_buf_J": P * T if P is not None else None, "E_ctr_J": (eb - ea) / 1000 if ea is not None and eb is not None else None,
           "step_ms_p50": steps[len(steps) // 2], "step_ms_p95": steps[int(.95 * (len(steps) - 1))],
           "bs_mean": st.fmean(e[2] for e in iters), "decode_sms_mode": max(set(dsm), key=dsm.count),
           "prefill_in_flight_frac": sum(e[4] for e in iters) / len(iters)}
    out["ctr_vs_buf"] = out["E_ctr_J"] / out["E_buf_J"] if out["E_ctr_J"] and out["E_buf_J"] else None
    return out, (a, b)


def _it(e):
    return (float(e["t_launch"]), float(e["t_sync"]), int(e.get("bs") or 0), int(e.get("decode_sms") or 0), bool(e.get("prefill_in_flight")))


def analyze_registered(run_dir: Path, meta: Path, R):
    E = read_energy(meta / "energy.csv"); S = read_samples(meta / "power_samples.csv"); st_ = [r[0] for r in S]
    rows, windows = [], []
    for mp in sorted(glob.glob(str(run_dir / "marks_*.txt"))):
        boot = Path(mp).stem[len("marks_"):]
        off, spread = wall_offset(run_dir / f"wall_{boot}.txt")
        pm = R.parse_marks(open(mp).read().splitlines())[0]
        recs = [json.loads(l) for l in open(run_dir / f"tel_{boot}.jsonl") if l.strip()]
        for cid, cm in pm.items():
            if "cell_start" not in cm or "cell_end" not in cm:
                continue
            (d, b, c), t0 = cm["cell_start"]; t1 = cm["cell_end"]
            cut = [r for r in recs if t0 <= float(r.get("timestamp_monotonic_s", -1)) <= t1]
            coload = "coload_sent" in cm
            base = {"boot": boot, "id": cid, "d": d, "b": b, "c": c, "offset_spread_ms": None if spread is None else 1000 * spread}
            if off is None:
                rows.append(dict(base, window="-", status="NO_WALL_PAIRS")); continue
            try:
                w = R.cell_windows(cut, cm, (d, b, c), R.PAIR[d], R.PAIR[d] if coload else None, coload=coload)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(base, window="-", status=f"WINDOW_ERROR:{type(exc).__name__}")); continue
            for name in ("W", "C", "A2"):
                if name not in w or w[name] is None or name not in w["status"]:
                    continue
                its = [_it(e) for e in w[name] if e.get("event") == "decode_iteration"]
                stt = w["status"][name]
                if stt != "VALID" or len(its) < 10:
                    rows.append(dict(base, window=name, status=f"N2_{stt}")); continue
                sc, ab = score_window(its, off, E, S, st_); windows.append(ab)
                rows.append(dict(base, window=name, status="OK" if sc["P_mean_W"] is not None else "NO_POWER_COVERAGE", **sc))
    return rows, windows, S


def analyze(run_dir: Path, meta: Path, use_registered=True):
    R = load_reducer() if use_registered else None
    if R is not None:
        rows, windows, S = analyze_registered(run_dir, meta, R)
        return _finish(rows, windows, S, source=f"n2_probe_reduce.cell_windows ({N2_REDUCER.name})")
    return analyze_whole(run_dir, meta)


def _finish(rows, windows, S, source):
    win = sorted(windows)
    def inside(x):
        k = bisect.bisect_right([w[0] for w in win], x + 1.0) - 1
        return k >= 0 and win[k][0] - 1.0 <= x <= win[k][1] + 1.0
    idle = [p for t, p in S if not inside(t)]
    P_idle = st.median(idle) if idle else None
    for r in rows:
        if r.get("status") == "OK":
            r["J_per_step"] = r["E_buf_J"] / r["n_steps"]
            r["J_per_token"] = r["J_per_step"] / r["bs_mean"] if r["bs_mean"] else None
            if P_idle is not None:
                r["J_per_step_above_idle"] = (r["P_mean_W"] - P_idle) * r["T_s"] / r["n_steps"]
    # exclusion contrast per cell: C (co-resident) vs mean of W/A2 (decode alone) at the same partition pair
    by = {}
    for r in rows:
        if r.get("status") == "OK":
            by.setdefault((r["boot"], r["id"]), {})[r["window"]] = r
    contrast = []
    for (boot, cid), ws in by.items():
        alone = [ws[k]["P_mean_W"] for k in ("W", "A2") if k in ws]
        if "C" in ws and alone:
            r0 = ws["C"]; pa = st.fmean(alone)
            contrast.append({"boot": boot, "id": cid, "d": r0["d"], "b": r0["b"], "c": r0["c"], "P_alone_W": pa,
                             "P_coresident_W": r0["P_mean_W"], "dP_prefill_W": r0["P_mean_W"] - pa,
                             "step_alone_ms": st.fmean(ws[k]["step_ms_p50"] for k in ("W", "A2") if k in ws),
                             "step_coresident_ms": r0["step_ms_p50"]})
    return {"window_source": source, "P_idle_W": P_idle, "n_idle_samples": len(idle), "cells": rows, "coresidency_contrast": contrast}


def analyze_whole(run_dir: Path, meta: Path):
    E = read_energy(meta / "energy.csv")
    S = read_samples(meta / "power_samples.csv")
    st_ = [r[0] for r in S]
    cells_out, windows = [], []
    for mp in sorted(glob.glob(str(run_dir / "marks_*.txt"))):
        boot = Path(mp).stem[len("marks_"):]
        off, spread = wall_offset(run_dir / f"wall_{boot}.txt")
        tel = read_tel(run_dir / f"tel_{boot}.jsonl")
        tl = [e[0] for e in tel]
        for c in read_marks(mp):
            rec = {"boot": boot, **{k: c[k] for k in ("id", "d", "b", "c")}, "offset_spread_ms": None if spread is None else 1000 * spread}
            if off is None or c.get("aborted"):
                rec["status"] = "NO_WALL_PAIRS" if off is None else f"ABORTED:{c['aborted']}"; cells_out.append(rec); continue
            t_end = c.get("coload_sent", c["t1"])          # E-alpha cell: only the decode-ALONE part before co-load
            rec["window"] = "pre_coload" if "coload_sent" in c else "whole_cell"
            i, j = bisect.bisect_left(tl, c["t0"]), bisect.bisect_right(tl, t_end)
            it = [e for e in tel[i:j] if e[1] <= t_end]
            if len(it) < 10:
                rec["status"] = "TOO_FEW_STEPS"; cells_out.append(rec); continue
            a, b = it[0][0] + off, it[-1][1] + off
            windows.append((a, b))
            T = b - a
            ea, eb = ctr_at(E, a), ctr_at(E, b)
            si, sj = bisect.bisect_left(st_, a + SENSOR_LAG_S), bisect.bisect_right(st_, b + SENSOR_LAG_S)
            pf = sum(e[4] for e in it) / len(it)
            steps = sorted(1000 * (e[1] - e[0]) for e in it)
            dsm = [e[3] for e in it]
            rec.update({"n_steps": len(it), "T_s": T, "E_J": (eb - ea) / 1000 if ea is not None and eb is not None else None,
                        "E_buf_J": None,
                        "P_mean_W": st.fmean(r[1] for r in S[si:sj]) if sj > si else None, "n_power_samples": sj - si,
                        "step_ms_p50": steps[len(steps) // 2], "step_ms_p95": steps[int(.95 * (len(steps) - 1))],
                        "bs_mean": st.fmean(e[2] for e in it), "decode_sms_mode": max(set(dsm), key=dsm.count),
                        "prefill_in_flight_frac": pf})
            rec["status"] = "NOT_DECODE_ONLY" if pf > PREFILL_TOL else ("NO_ENERGY_COVERAGE" if rec["E_J"] is None else "OK")
            cells_out.append(rec)
    # idle power: buffered samples outside every cell window (and >1 s away from them)
    win = sorted(windows)
    def inside(x):
        k = bisect.bisect_right([w[0] for w in win], x + 1.0) - 1
        return k >= 0 and win[k][0] - 1.0 <= x <= win[k][1] + 1.0
    idle = [p for t, p in S if not inside(t)]
    P_idle = st.median(idle) if idle else None
    for r in cells_out:
        if r.get("status") == "OK":
            r["J_per_step"] = r["E_J"] / r["n_steps"]
            r["J_per_token"] = r["J_per_step"] / r["bs_mean"] if r["bs_mean"] else None
            if P_idle is not None:
                r["J_per_step_above_idle"] = (r["E_J"] - P_idle * r["T_s"]) / r["n_steps"]
    return {"P_idle_W": P_idle, "n_idle_samples": len(idle), "cells": cells_out}


def to_md(res):
    cols = ["d", "b", "c", "window", "status", "decode_sms_mode", "bs_mean", "n_steps", "step_ms_p50", "P_mean_W", "J_per_step",
            "J_per_step_above_idle", "J_per_token", "prefill_in_flight_frac"]
    lines = ["# N2 decode-only power per cell (power_cap track, descriptive; not an N2 decision input)", "",
             f"window source: {res.get('window_source', 'whole cell')}", "",
             f"P_idle = {res['P_idle_W']} W (median of {res['n_idle_samples']} buffered samples outside cell windows)", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in sorted(res["cells"], key=lambda r: (r["d"], r["b"], r["c"])):
        def f(k):
            v = r.get(k)
            return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(f(k) for k in cols) + " |")
    if res.get("coresidency_contrast"):
        lines += ["", "## decode alone (W, A2) vs co-resident with prefill (C), same partition pair", "",
                  "| d | b | c | P_alone_W | P_coresident_W | dP_prefill_W | step_alone_ms | step_coresident_ms |", "|---|---|---|---|---|---|---|---|"]
        for c in sorted(res["coresidency_contrast"], key=lambda r: (r["d"], r["b"], r["c"])):
            lines.append(f"| {c['d']} | {c['b']} | {c['c']} | {c['P_alone_W']:.1f} | {c['P_coresident_W']:.1f} | {c['dP_prefill_W']:+.1f} | "
                         f"{c['step_alone_ms']:.2f} | {c['step_coresident_ms']:.2f} |")
    return "\n".join(lines) + "\n"


def plot(res, out: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "font.size": 10})
    ok = [r for r in res["cells"] if r.get("status") == "OK" and r.get("window", "W") in ("W", "A2", "whole_cell", "pre_coload")]
    if not ok:
        return None
    cs = sorted({r["c"] for r in ok}); bs = sorted({r["b"] for r in ok})
    fig, axes = plt.subplots(1, len(cs), figsize=(4.2 * len(cs), 3.8), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cs):
        for k, b in enumerate(bs):
            pts = sorted((r["decode_sms_mode"], r["J_per_step"]) for r in ok if r["c"] == c and r["b"] == b)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2, color=pal[k % len(pal)], label=f"b={b}")
        ax.set_title(f"context c={c}", loc="left", fontsize=10); ax.set_xlabel("realized decode SMs"); ax.grid(axis="y", color=GRID)
    axes[0][0].set_ylabel("board energy per decode step (J)"); axes[0][-1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); p = out / "n2_J_per_step.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ------------------------------------------------------------------ selftest
def _synth(td: Path, offset=1.79e9, idle=85.0, swap_offset=False):
    """Two boots; per cell (d, b): step time and board power follow a known model, so J/step is known exactly."""
    rng = random.Random(3); run = td / "run"; meta = td / "meta"; run.mkdir(); meta.mkdir()
    truth, energy_rows, samples = {}, [], []
    t_mono = 1000.0; acc = 0.0
    def emit_power(t_a, t_b, P):
        nonlocal acc
        t = t_a
        while t < t_b:
            samples.append((t + offset, P)); t += 0.02
    for bi, ds in enumerate(([16, 44], [108])):
        boot = f"b{bi}"
        mk, wl, tel = [], [], []
        if bi == 0:
            emit_power(t_mono - 5, t_mono, idle)   # lead-in idle only before the first boot (no overlap)
        emit_power(t_mono, t_mono + 0.5, idle); t_mono += 0.5
        for d in ds:
            for b in (1, 8):
                cid = f"{boot}_d{d}_b{b}"
                step = 0.010 + 0.0008 * b + 0.15 / d          # seconds
                P = idle + 2.0 * d + 6.0 * b                  # W during decode
                t0 = t_mono; mk.append(f"cell_start {cid} {d} {b} 2048 {t0!r}"); wl.append(f"cell_start {cid} {t0 + offset!r} {t0!r}")
                t = t0 + 0.2; n = 0
                while n < 300:
                    tel.append(json.dumps({"event": "decode_iteration", "t_launch": t, "t_sync": t + step, "bs": b, "decode_sms": d,
                                           "prefill_in_flight": False, "iter_seq": n})); n += 1; t += step
                emit_power(t0, t0 + 0.2, idle); emit_power(t0 + 0.2, t, P)
                truth[(d, b)] = P * step
                t1 = t + 0.1; emit_power(t, t1, idle)
                mk.append(f"cell_end {cid} {t1!r}"); wl.append(f"cell_end {cid} {t1 + offset!r} {t1!r}")
                t_mono = t1 + 1.5; emit_power(t1, t_mono, idle)
        (run / f"marks_{boot}.txt").write_text("\n".join(mk) + "\n")
        (run / f"wall_{boot}.txt").write_text("\n".join(wl) + "\n")
        (run / f"tel_{boot}.jsonl").write_text("\n".join(tel) + "\n")
    samples.sort()
    # energy counter: exact integral of the piecewise-constant power, sampled every 100 ms
    e = 0.0; k = 0; tE = samples[0][0]; rows = []
    while tE < samples[-1][0]:
        while k + 1 < len(samples) and samples[k + 1][0] <= tE:
            k += 1
        rows.append((tE, int(e * 1000))); e += samples[k][1] * 0.1; tE += 0.1
    (meta / "energy.csv").write_text("epoch_s,energy_mJ\n" + "".join(f"{t:.3f},{v}\n" for t, v in rows))
    (meta / "power_samples.csv").write_text("t_us,power_mW\n" + "".join(f"{int(t * 1e6)},{int(p * 1000)}\n" for t, p in samples))
    if swap_offset:   # mutant: wall pairs written with the wrong sign
        for w in run.glob("wall_*.txt"):
            w.write_text("".join(" ".join(l.split()[:2] + [l.split()[3], l.split()[2]]) + "\n" for l in w.read_text().splitlines()))
    return run, meta, truth, idle


def selftest():
    fails = []
    with tempfile.TemporaryDirectory() as d:
        td = Path(d) / "ok"; td.mkdir()
        run, meta, truth, idle = _synth(td)
        res = analyze(run, meta, use_registered=False)
        ok = {(r["d"], r["b"]): r for r in res["cells"] if r.get("status") == "OK"}
        if set(ok) != set(truth):
            fails.append(f"cells OK {sorted(ok)} vs {sorted(truth)}")
        for k, v in truth.items():
            if k in ok and abs(ok[k]["J_per_step"] / v - 1) > 0.03:
                fails.append(f"J/step {k} {ok[k]['J_per_step']:.4f} vs {v:.4f}")
        if res["P_idle_W"] is None or abs(res["P_idle_W"] - idle) > 1:
            fails.append(f"idle {res['P_idle_W']}")
        # ordering control: J/step must rise with b at fixed d (built into the model) -- detects swapped columns
        for d_ in (16, 44):
            if (d_, 1) in ok and (d_, 8) in ok and not ok[(d_, 8)]["J_per_step"] > ok[(d_, 1)]["J_per_step"]:
                fails.append(f"b ordering at d={d_}")
        # mutant: wrong-sign wall pairs -> windows land outside the power trace -> no OK cells
        tm = Path(d) / "mut"; tm.mkdir()
        run2, meta2, _, _ = _synth(tm, swap_offset=True)
        res2 = analyze(run2, meta2, use_registered=False)
        if any(r.get("status") == "OK" for r in res2["cells"]):
            fails.append("wrong-sign clock offset not detected")
        # NOT_DECODE_ONLY flag
        tp = Path(d) / "pf"; tp.mkdir()
        run3, meta3, _, _ = _synth(tp)
        t3 = run3 / "tel_b0.jsonl"
        t3.write_text("\n".join(l.replace('"prefill_in_flight": false', '"prefill_in_flight": true') for l in t3.read_text().splitlines()) + "\n")
        res3 = analyze(run3, meta3, use_registered=False)
        if not all(r["status"] == "NOT_DECODE_ONLY" for r in res3["cells"] if r["boot"] == "b0"):
            fails.append("prefill_in_flight flag not applied")
        fails += _selftest_registered(Path(d) / "reg")
    print("SELFTEST", "PASS" if not fails else "FAIL", fails)
    return 0 if not fails else 1


def _selftest_registered(td: Path):
    """Fake N2 attempt through the REAL harness (fake engine), synthetic power: decode alone 250 W, prefill in flight
    330 W, idle 85 W.  Expect: W/A2 windows ~250 W, C ~330 W (dP ~ +80 W), clock-pair spread < 1 ms."""
    fails = []
    R = load_reducer()
    if R is None:
        return ["N2 reducer not importable (registered path untested)"]
    n2 = N2_REDUCER.parent
    sys.path.insert(0, str(n2))
    try:
        import asyncio, contextlib, io
        import test_n2_harness as T
        td.mkdir(parents=True)
        with contextlib.redirect_stderr(io.StringIO()):
            adir, _ = T.attempt(str(td), {44: 1, 16: 1})
    except Exception as exc:  # noqa: BLE001
        return [f"fake attempt failed: {exc!r}"]
    adir = Path(adir)
    walls = [ln.split() for w in adir.glob("wall_*.txt") for ln in w.read_text().splitlines()]
    off = st.median(float(f[2]) - float(f[3]) for f in walls)
    iters = sorted((float(e["t_launch"]) + off, float(e["t_sync"]) + off, bool(e.get("prefill_in_flight")))
                   for tp in adir.glob("tel_*.jsonl") for e in map(json.loads, tp.read_text().splitlines())
                   if e.get("event") == "decode_iteration")
    starts = [x[0] for x in iters]
    t0 = min(float(f[2]) for f in walls) - 3; t1 = max(float(f[2]) for f in walls) + 3
    meta = td / "meta"; meta.mkdir()
    ps, es = ["t_us,power_mW"], ["epoch_s,energy_mJ"]; e = 0.0; t = t0; k = 0
    while t < t1:
        j = bisect.bisect_right(starts, t) - 1
        p = (330.0 if iters[j][2] else 250.0) if j >= 0 and iters[j][1] >= t else 85.0
        ps.append(f"{int(t * 1e6)},{int(p * 1000)}")
        if k % 5 == 0:
            es.append(f"{t:.4f},{int(e * 1000)}")
        e += p * 0.002; t += 0.002; k += 1          # 2 ms synthetic resolution (finer than the real 20 ms)
    (meta / "power_samples.csv").write_text("\n".join(ps) + "\n"); (meta / "energy.csv").write_text("\n".join(es) + "\n")
    res = analyze(adir, meta)
    if not res["window_source"].startswith("n2_probe_reduce"):
        fails.append("registered path not used")
    okw = [r for r in res["cells"] if r.get("status") == "OK"]
    if {r["window"] for r in okw} != {"W", "C", "A2"}:
        fails.append(f"windows scored {sorted({r['window'] for r in okw})}")
    for r in okw:
        if r["window"] in ("W", "A2") and not (235 <= r["P_mean_W"] <= 255):
            fails.append(f"alone window {r['id']} {r['window']} P {r['P_mean_W']:.1f}")
        if (r["offset_spread_ms"] or 0) > 1.0:
            fails.append("clock-pair spread > 1 ms")
    if not res["coresidency_contrast"] or not all(40 <= c["dP_prefill_W"] <= 90 for c in res["coresidency_contrast"]):
        fails.append(f"co-residency contrast {[round(c['dP_prefill_W'], 1) for c in res['coresidency_contrast']]}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir"); ap.add_argument("--meta"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    out = Path(a.out or Path(__file__).resolve().parent / "out"); out.mkdir(parents=True, exist_ok=True)
    res = analyze(Path(a.run_dir), Path(a.meta))
    (out / "n2_decode_power.json").write_text(json.dumps(res, indent=1))
    (out / "n2_decode_power.md").write_text(to_md(res))
    p = plot(res, out)
    print("cells", len(res["cells"]), "ok", sum(r.get("status") == "OK" for r in res["cells"]), "figure", p)


if __name__ == "__main__":
    main()
