#!/usr/bin/env python3
"""Cell / boot plan, M2g plan, budget dry-run and M2G_PASS_TABLE handling (PREREG_MULTIGPU rev6 §3, §4-1, §4-2, §6).

Pure planning: no GPU, no network.  The Job drivers (mg_drive.py) turn the specs into mg_boot.py runs.

  python3 mg_plan.py m3-plan                 print the static M3 boot/cell skeleton (counts)
  python3 mg_plan.py m3-budget [--ratio R]   per-stage time model -> total vs the 30 h hard cap
  python3 mg_plan.py m2g-plan | m2g-budget
  python3 mg_plan.py m3-forecast             F15: scenario x calibration table (30 h cut, trend domain) + k-Job alternative
  python3 mg_plan.py check-pass-table PATH SHA256 [COMMITTED_SHA256_FILE]     ABORT_M2G_NOT_PASSED logic (F17: 3-way digest)
  python3 mg_plan.py --selftest

TIME MODEL (an ESTIMATE, not a registered rule; rev5 audit "boot 4 min, n/lambda, drain" method):
  cell   = max(n/x + drain, n/lambda)  (+ gap)           lambda = ratio * prior(N, shape)   [lambda0_plan.cell_duration_at]
  boot   = EST_BOOT_S + warm-up per shape + sum(cells)
`ratio` is the true-capacity / prior ratio (1.0 = the audit's "lambda = prior" case, 0.7 = a TP case with lambda < prior).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mg_const as C  # noqa: E402

# ----------------------------------------------------------------------------- configs
def config_spec(cfg):
    """(mode, N, tp, dp, gpu_ids).  TP4X = the NUMA-crossing sensitivity subset (§4-6)."""
    if cfg == "N1":
        return dict(config="N1", mode="N1", N=1, tp=1, dp=1, gpu_ids=C.GPU_SUBSETS["N1"])
    if cfg == "TP4X":
        return dict(config="TP4X", mode="TP", N=4, tp=4, dp=1, gpu_ids=C.NUMA_SUBSET_TP4)
    mode, n = cfg[:2], int(cfg[2:])
    return dict(config=cfg, mode=mode, N=n, tp=n if mode == "TP" else 1, dp=n if mode == "DP" else 1,
                gpu_ids=C.GPU_SUBSETS[cfg])


def round10(n):
    return int(round(n / 10.0) * 10)


def n_req_for(x):
    """§4-2: clamp(round10(x * 170 s), 40, 4000)."""
    return max(C.N_REQ_CLAMP[0], min(C.N_REQ_CLAMP[1], round10(x * C.WINDOW_S)))


def prior(N, shape):
    return N * C.PRIOR_PER_GPU[shape]


def hi_rung(N, shape, ext=0):
    x = C.HI_MULT * prior(N, shape) * (C.EXT_MULT ** ext)
    return {"offered": round(x, 6), "n_req": n_req_for(x)}


def lo_rung(fused_hi_median_achieved):
    x = C.LO_MULT * fused_hi_median_achieved
    return {"offered": round(x, 6), "n_req": n_req_for(x)}


def arm_order(cfg):
    """Random permutation of the 4 arms per config, seed 20261012 (§4-1).  Deterministic."""
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}")
    a = list(C.ARMS)
    rng.shuffle(a)
    return a


def seed_order(cfg, arm, tag, seeds):
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}:{arm}:{tag}")
    s = list(seeds)
    rng.shuffle(s)
    return s


def shape_order(cfg, arm, tag):
    rng = random.Random(f"{C.ARM_ORDER_SEED}:{cfg}:{arm}:{tag}:shape")
    s = ["A", "B"]
    rng.shuffle(s)
    return s


def _boot(cfg, arm, kind, steps, block="main", suffix="", **extra):
    cs = config_spec(cfg)
    spec = dict(cs, arm=arm, kind=kind, block=block, boot_id=f"m3_{cfg}_{arm}_{kind}{suffix}", steps=steps)
    spec.update(extra)
    return spec


def _cell(shape, rung, seed, N, ext=0, rnd=1, block="main", offered=None, n_req=None, **kw):
    r = hi_rung(N, shape, ext) if rung == "HI" else {"offered": offered, "n_req": n_req}
    if offered is not None and rung == "HI":
        r = {"offered": offered, "n_req": n_req}
    d = dict(type="cell", shape=shape, rung=rung, seed=seed, ext=ext, round=rnd, block=block, **r)
    d.update(kw)
    return d


def hi_steps(cfg, arm, shapes, ext=0, rnd=1, block="main"):
    N = config_spec(cfg)["N"]
    steps = []
    for sh in shape_order(cfg, arm, f"hi{ext}{rnd}"):
        if sh not in shapes:
            continue
        steps.append({"type": "warmup", "shape": sh})
        for sd in seed_order(cfg, arm, f"hi{ext}{rnd}{sh}", C.HI_SEEDS):
            steps.append(_cell(sh, "HI", sd, N, ext, rnd, block))
    return steps


def lo_steps(cfg, arm, x_lo_by_shape, ext=0, rnd=1):
    """x_lo_by_shape: shape -> {offered, n_req} (SAME for the four arms of a (config, shape), MGP-9)."""
    steps = []
    ext_of = (lambda sh: ext.get(sh, 0)) if isinstance(ext, dict) else (lambda sh: ext)    # extension is per (config, shape)
    for sh in shape_order(cfg, arm, f"lo{ext_of('A')}{ext_of('B')}{rnd}"):
        if sh not in x_lo_by_shape:
            continue
        steps.append({"type": "warmup", "shape": sh})
        for sd in seed_order(cfg, arm, f"lo{ext_of(sh)}{rnd}{sh}", C.LO_SEEDS):
            steps.append(_cell(sh, "LO", sd, config_spec(cfg)["N"], ext_of(sh), rnd, "main", **x_lo_by_shape[sh]))
    return steps


def boots_gpu_dev():
    """§4-6: N=1 fused shape A HI seed 4386 on GPU 0..7 (boot 8x).  Runs FIRST in the M3 Job."""
    out = []
    for g in range(C.EXPECTED_GPUS):
        spec = dict(config_spec("N1"), gpu_ids=str(g))
        N = 1
        st = [{"type": "warmup", "shape": "A"}, _cell("A", "HI", C.GPU_DEV_SEED, N, 0, 1, "gpu_dev", gpu=g)]
        out.append(dict(spec, arm="fused", kind="gpudev", block="gpu_dev", boot_id=f"m3_gpudev_gpu{g}", steps=st))
    return out


def boots_boot1(cfg, arms=None, exclude=()):
    return [_boot(cfg, a, "boot1", hi_steps(cfg, a, ("A", "B"))) for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_ext(cfg, shapes, arms=None, exclude=()):
    return [_boot(cfg, a, "boot1ext", hi_steps(cfg, a, shapes, ext=1), suffix="") for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_boot2(cfg, x_lo_by_shape, ext=0, arms=None, exclude=()):
    return [_boot(cfg, a, "boot2", lo_steps(cfg, a, x_lo_by_shape, ext)) for a in (arms or arm_order(cfg))
            if f"{cfg}:{a}" not in exclude]


def boots_numa():
    out = []
    for a in ("fused", "agnostic"):
        out.append(_boot("TP4X", a, "numa", hi_steps("TP4X", a, ("A",), block="numa"), block="numa"))
    return out


def boots_nsys():
    N = 8
    x = hi_rung(N, "A")["offered"]
    n30 = max(C.N_REQ_CLAMP[0], round10(x * C.G4_NSYS_S))
    st = [{"type": "warmup", "shape": "A"},
          {"type": "cmd", "name": "nsys_start", "argv": ["nsys", "start", "-o", "{BOOT}/nsys_tp8_d44", "--force-overwrite=true"]},
          _cell("A", "HI", C.GPU_DEV_SEED, N, 0, 1, "nsys", offered=x, n_req=n30),
          {"type": "cmd", "name": "nsys_stop", "argv": ["nsys", "stop"]}]
    return [_boot("TP8", "d44", "nsys", st, block="nsys", nsys=True)]


def boots_anchor2():
    """§4-5: N=1 round 2 -- fused, fused_sym, agnostic shape A HI (4 seeds)."""
    return [_boot("N1", a, "anchor2", hi_steps("N1", a, ("A",), ext=0, rnd=2, block="anchor"), block="anchor")
            for a in ("fused", "fused_sym", "agnostic")]


# ----------------------------------------------------------------------------- M2G pass table
def table_key(cfg, arm):
    return f"{cfg}:{arm}"


def load_pass_table(path, expected_sha256, committed_sha_path=None):
    """Returns (table, abort_code).  F17: the table is accepted only if THREE digests agree --
       (1) PDMUX_M2G_PASS_SHA256 (env / CLI `expected_sha256`), (2) the COMMITTED results/multigpu/M2G_PASS_TABLE.json.sha256
       (`committed_sha_path`; a missing file is a failure when the argument is given), (3) sha256 of the file bytes.
    Missing file / any mismatch / unparsable -> ABORT_M2G_NOT_PASSED."""
    p = Path(path)
    if not p.exists():
        return None, "ABORT_M2G_NOT_PASSED"
    raw = p.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not expected_sha256 or actual != expected_sha256:
        return None, "ABORT_M2G_NOT_PASSED"
    if committed_sha_path:
        cp = Path(committed_sha_path)
        if not cp.exists() or cp.read_text().strip().split()[:1] != [actual]:
            return None, "ABORT_M2G_NOT_PASSED"
    try:
        t = json.loads(raw)
        assert isinstance(t.get("cells"), dict)
    except (ValueError, AssertionError):
        return None, "ABORT_M2G_NOT_PASSED"
    return t, None


def excluded_set(table):
    """'cfg:arm' keys excluded from M3 (included == false)."""
    return {k for k, v in table["cells"].items() if not v.get("included", True)}


def runnable_configs(table):
    """Configs whose fused arm is included (N1 too: the table carries its G0(i)/G4/contamination rows)."""
    ex = excluded_set(table)
    return [c for c in C.CONFIG_ORDER if table_key(c, "fused") not in ex]


# ----------------------------------------------------------------------------- budget model
CALIBS = {                     # F15: (boot_extra_s per boot, cell_gap_s per cell)
    "audited": dict(boot_extra_s=0.0, cell_gap_s=C.EST_CELL_GAP_S),                     # the model behind the audited 29.96 h
    "vessl_boot": dict(boot_extra_s=C.EST_VESSL_BOOT_EXTRA_S, cell_gap_s=C.EST_CELL_GAP_S),   # N3 r3 +49 s attributed per boot
    "vessl_boot_cell": dict(boot_extra_s=C.EST_VESSL_BOOT_EXTRA_S, cell_gap_s=C.EST_VESSL_CELL_GAP_S),   # + 24 s per cell (upper)
}


def cell_seconds(shape, x, n, lam, gap=None):
    span = n / x
    drain = C.EST_DRAIN_S[shape]
    gap = C.EST_CELL_GAP_S if gap is None else gap
    return max(span + drain, n / lam if lam > 0 else span + drain) + gap


def boot_seconds(spec, lam_of, x_lo_of=None, calib=None):
    """lam_of(N, shape) -> true capacity assumed for the arm.  calib = CALIBS entry (default: the audited model)."""
    calib = calib or CALIBS["audited"]
    t = C.EST_BOOT_S + calib["boot_extra_s"]
    N = spec["N"]
    for st in spec["steps"]:
        if st["type"] == "warmup":
            t += C.EST_WARMUP_S[st["shape"]]
        elif st["type"] == "cell":
            t += cell_seconds(st["shape"], st["offered"], st["n_req"], lam_of(N, st["shape"]), calib["cell_gap_s"])
        elif st["type"] == "cmd":
            t += 20.0
    return t


def _arm_ratio(ratio_of, ratio_arm_of, cfg, arm):
    return ratio_arm_of(cfg, arm) if ratio_arm_of else ratio_of(cfg)


def ext_fires(ratio_of, ratio_arm_of, cfg, assume_ext=None):
    """Joint extension (§4-2): ANY arm above EXT_TRIGGER_RATIO x prior (audit §3) -- the four arms extend together."""
    if assume_ext is not None:
        return 1 if assume_ext else 0
    return 1 if any(_arm_ratio(ratio_of, ratio_arm_of, cfg, a) > C.EXT_TRIGGER_RATIO for a in C.ARMS) else 0


def m3_static_boots(ratio_of=None, assume_ext=None, lo_given=True, ratio_arm_of=None):
    """The full M3 boot list for a given assumed capacity ratio (ratio_of(cfg) -> lambda/prior, or ratio_arm_of(cfg, arm)).
    x_lo uses the assumed FUSED HI achieved = ratio * prior.  Extension is assumed iff any arm's ratio > 1.42 (audit)."""
    ratio_of = ratio_of or (lambda cfg: 1.0)
    stages = [("gpu_dev", boots_gpu_dev())]
    for cfg in C.CONFIG_ORDER:
        N = config_spec(cfg)["N"]
        ext = ext_fires(ratio_of, ratio_arm_of, cfg, assume_ext)
        r = _arm_ratio(ratio_of, ratio_arm_of, cfg, "fused")
        lam = {sh: r * prior(N, sh) for sh in ("A", "B")}
        stages.append((f"{cfg}.boot1", boots_boot1(cfg)))
        if ext:
            stages.append((f"{cfg}.boot1ext", boots_ext(cfg, ("A", "B"))))
        xlo = {sh: lo_rung(lam[sh]) for sh in ("A", "B")}
        stages.append((f"{cfg}.boot2", boots_boot2(cfg, xlo, ext)))
    stages += [("numa", boots_numa()), ("nsys", boots_nsys()), ("anchor2", boots_anchor2())]
    return stages


def _boot_true_seconds(b, ratio_of, ratio_arm_of, calib):
    lam_of = lambda N, sh, b=b: _arm_ratio(ratio_of, ratio_arm_of, b["config"], b["arm"]) * prior(N, sh)
    return boot_seconds(b, lam_of, calib=calib)


def m3_budget(ratio_of=None, assume_ext=None, ratio_arm_of=None, calib=None):
    ratio_of = ratio_of or (lambda cfg: 1.0)
    rows, total, n_boots, n_cells = [], 0.0, 0, 0
    for name, boots in m3_static_boots(ratio_of, assume_ext, ratio_arm_of=ratio_arm_of):
        t = 0.0
        for b in boots:
            t += _boot_true_seconds(b, ratio_of, ratio_arm_of, calib)
            n_cells += sum(1 for s in b["steps"] if s["type"] == "cell")
        n_boots += len(boots)
        rows.append((name, len(boots), t))
        total += t
    return {"stages": rows, "total_s": total, "total_h": total / 3600.0, "boots": n_boots, "cells": n_cells,
            "cap_h": C.HARD_CAP_S["m3"] / 3600.0, "over_cap": total > C.HARD_CAP_S["m3"]}


def boot_need_s(spec, ratio=0.5):
    """Admission need (worst corner, like N3 §8-3): ratio of prior capacity.  F15: includes the VESSL-measured per-boot
    overhead (+49 s) -- N3 r3 admitted on a need that was 40 % too small."""
    return boot_seconds(spec, lambda N, sh: ratio * prior(N, sh), calib=CALIBS["vessl_boot"])


# ----------------------------------------------------------------------------- F15 forecast (30 h cut, trend domain)
SCENARIOS = {                  # name -> ratio_arm_of(cfg, arm)  (lambda_true / prior)
    "S1 lambda=prior (all)": lambda cfg, arm: 1.0,
    "S2 TP lambda=0.7 prior": lambda cfg, arm: 0.7 if cfg.startswith("TP") else 1.0,
    "S3 extension: agnostic 1.48": lambda cfg, arm: 1.48 if arm == "agnostic" else 1.0,
}


def m3_timeline(ratio_arm_of, calib, cap_s=None, preflight_s=0.0, drop_stages=()):
    """Sequential simulation of the driver (§4-1 order, admission against the remaining time, F10 rules).
    A boot is ADMITTED iff boot_need_s(spec, 1.0) <= remaining (mg_drive.run_boot); it then takes its scenario duration and
    is PARTIAL if that crosses the cap (the runner kills the running cell).  Returns the boot records."""
    cap_s = C.HARD_CAP_S["m3"] if cap_s is None else cap_s
    ratio_of = lambda cfg: 1.0
    t, recs = float(preflight_s), []

    def run(spec, stage):
        nonlocal t
        need = boot_need_s(spec, 1.0)
        rec = {"stage": stage, "config": spec["config"], "arm": spec["arm"], "block": spec.get("block", "main"),
               "start_s": t, "need_s": need}
        if cap_s - t < need:
            rec.update(status="NOT_ATTEMPTED", end_s=t)
        else:
            d = _boot_true_seconds(spec, ratio_of, ratio_arm_of, calib)
            if t + d > cap_s:
                rec.update(status="PARTIAL", end_s=cap_s)
                t = cap_s
            else:
                rec.update(status="DONE", end_s=t + d)
                t += d
        recs.append(rec)
        return rec["status"] == "DONE"

    for spec in boots_gpu_dev():
        run(spec, "gpu_dev")
    for cfg in C.CONFIG_ORDER:
        N = config_spec(cfg)["N"]
        b1 = [run(s, f"{cfg}.boot1") for s in boots_boot1(cfg)]
        ext = ext_fires(ratio_of, ratio_arm_of, cfg)
        ext_ok = True
        if ext:
            ext_ok = all([run(s, f"{cfg}.boot1ext") for s in boots_ext(cfg, ("A", "B"))])
        fused_i = arm_order(cfg).index("fused")
        fused_ok = b1[fused_i] and (ext_ok if ext else True)
        if not ext_ok:                                           # F10: an incomplete extension forbids entering boot2 (LO)
            for s in boots_boot2(cfg, {sh: lo_rung(prior(N, sh)) for sh in ("A", "B")}, ext):
                recs.append({"stage": f"{cfg}.boot2", "config": cfg, "arm": s["arm"], "block": "main", "start_s": t,
                             "end_s": t, "need_s": 0.0, "status": "NOT_ATTEMPTED_EXT_INCOMPLETE"})
        elif fused_ok:
            lam = {sh: _arm_ratio(ratio_of, ratio_arm_of, cfg, "fused") * prior(N, sh) for sh in ("A", "B")}
            for s in boots_boot2(cfg, {sh: lo_rung(lam[sh]) for sh in ("A", "B")}, ext):
                run(s, f"{cfg}.boot2")
        else:                                                    # fused HI undefined -> x_lo undefined -> LO skipped (MGH-11)
            for s in boots_boot2(cfg, {sh: lo_rung(prior(N, sh)) for sh in ("A", "B")}, ext):
                recs.append({"stage": f"{cfg}.boot2", "config": cfg, "arm": s["arm"], "block": "main", "start_s": t,
                             "end_s": t, "need_s": 0.0, "status": "NOT_ATTEMPTED_FUSED_HI_UNDEFINED"})
    for stage, boots in (("numa", boots_numa()), ("nsys", boots_nsys()), ("anchor2", boots_anchor2())):
        for s in boots:
            run(s, stage)
    return recs


def forecast_summary(recs, ratio_arm_of=None):
    """Which stages are cut and which G_sym points / primary trends survive (all shapes alike in the static model)."""
    done = lambda cfg, arm, stage_suffix: any(r["config"] == cfg and r["arm"] == arm and r["stage"].endswith(stage_suffix)
                                              and r["status"] == "DONE" for r in recs)
    ext_of = lambda cfg: any(r["stage"] == f"{cfg}.boot1ext" for r in recs)
    defined = []
    for cfg in C.CONFIG_ORDER:
        ok = all(done(cfg, a, ".boot1") and (not ext_of(cfg) or done(cfg, a, ".boot1ext")) and done(cfg, a, ".boot2")
                 for a in ("agnostic", "fused_sym"))
        if ok:
            defined.append(cfg)
    Ns = lambda mode: sorted(([1] if "N1" in defined else []) + [int(c[2:]) for c in defined if c.startswith(mode)])
    cut = {}
    for r in recs:
        if r["status"] != "DONE":
            cut.setdefault(r["stage"], []).append(f"{r['arm']}:{r['status']}")
    lab = lambda ns: "TREND_UNDEFINED" if len(ns) < C.TREND_MIN_POINTS else "defined"
    anchor_done = [r for r in recs if r["stage"] == "anchor2"]
    return {"end_h": max((r["end_s"] for r in recs), default=0.0) / 3600.0,
            "n_done": sum(1 for r in recs if r["status"] == "DONE"), "n_boots": len(recs),
            "cut": cut, "N_defined_TP": Ns("TP"), "N_defined_DP": Ns("DP"),
            "trend_TP": lab(Ns("TP")), "trend_DP": lab(Ns("DP")),
            "anchor_boots_done": sum(1 for r in anchor_done if r["status"] == "DONE"), "anchor_boots": len(anchor_done)}


def m3_forecast_table(cap_h=None, preflight_s=0.0):
    """F15 decision table: scenario x calibration -> total (model, uncapped), cut stages at the cap, surviving trend domain."""
    rows = []
    cap_s = (cap_h * 3600.0) if cap_h else None
    for sc, fn in SCENARIOS.items():
        for cname, calib in CALIBS.items():
            tot = m3_budget(ratio_arm_of=fn, calib=calib)["total_h"]
            recs = m3_timeline(fn, calib, cap_s, preflight_s)
            rows.append({"scenario": sc, "calib": cname, "total_h_uncapped": tot, **forecast_summary(recs)})
    return rows


def block_plan_table(nblocks_list=(2, 3), preflight_s=None):
    """F15 alternative: split the campaign into `k` Jobs (blocks).  Contiguous partition of the §4-1 sequence (gpu_dev, N1, TP8,
    DP8, TP2, DP2, TP4, DP4) balanced on model time; NUMA + nsys ride in the LAST block; EVERY block ends with its own anchor
    (N=1 fused / fused_sym / agnostic shape A HI, §4-5 C-9) and pays one preflight.  Total = single-Job work + k anchors + k preflights.
    NOT audited (blocks need a new harness + audit)."""
    preflight_s = C.EST_PREFLIGHT_S if preflight_s is None else preflight_s
    out = []
    for sc, fn in SCENARIOS.items():
        for cname, calib in CALIBS.items():
            ratio_of = lambda cfg: 1.0
            unit = []                                    # (label, seconds) in §4-1 order
            gd = sum(_boot_true_seconds(b, ratio_of, fn, calib) for b in boots_gpu_dev())
            unit.append(("gpu_dev", gd))
            stages = m3_static_boots(ratio_of, ratio_arm_of=fn)
            for cfg in C.CONFIG_ORDER:
                unit.append((cfg, sum(_boot_true_seconds(b, ratio_of, fn, calib)
                                      for nm, bs in stages if nm.startswith(cfg + ".") for b in bs)))
            tail = sum(_boot_true_seconds(b, ratio_of, fn, calib) for nm, bs in stages if nm in ("numa", "nsys") for b in bs)
            anchor = sum(_boot_true_seconds(b, ratio_of, fn, calib) for b in boots_anchor2())
            import itertools
            for k in nblocks_list:
                best = None
                for cuts in itertools.combinations(range(1, len(unit)), k - 1):      # min-max contiguous partition (brute force)
                    edges = (0,) + cuts + (len(unit),)
                    parts = [unit[edges[i]:edges[i + 1]] for i in range(k)]
                    ts = [sum(u[1] for u in pt) for pt in parts]
                    ts[-1] += tail
                    if best is None or max(ts) < best[0]:
                        best = (max(ts), parts, ts)
                _, parts, ts = best
                blocks = [([u[0] for u in pt] + (["numa+nsys"] if i == k - 1 else []), ts[i]) for i, pt in enumerate(parts)]
                hs = [(b[0], (b[1] + anchor + preflight_s) / 3600.0) for b in blocks]
                out.append({"scenario": sc, "calib": cname, "k": k, "blocks": hs,
                            "total_h": sum(h for _, h in hs), "max_block_h": max(h for _, h in hs),
                            "anchor_h_each": anchor / 3600.0})
    return out


def fmt_forecast(rows, blocks):
    L = ["== F15 forecast (model estimates, NOT measurements; MGH-6/MGH-7) ==",
         f"cap {C.HARD_CAP_S['m3'] / 3600:.0f} h; calibrations: audited = model of the audited 29.96 h; vessl_boot = +{C.EST_VESSL_BOOT_EXTRA_S:.0f} s/boot "
         f"(N3 r3: 170 s vs 121 s); vessl_boot_cell = also +{C.EST_VESSL_CELL_GAP_S - C.EST_CELL_GAP_S:.0f} s/cell (upper)",
         f"{'scenario':30s} {'calib':16s} {'total_h':>8s} {'end_h':>6s} {'boots done':>10s} {'N def TP':>10s} {'N def DP':>10s} "
         f"{'trendTP':>16s} {'trendDP':>16s} {'anchor':>7s}  cut"]
    for r in rows:
        cut = "; ".join(f"{k}[{len(v)}]" for k, v in r["cut"].items()) or "-"
        L.append(f"{r['scenario']:30s} {r['calib']:16s} {r['total_h_uncapped']:8.2f} {r['end_h']:6.2f} "
                 f"{r['n_done']:4d}/{r['n_boots']:<5d} {str(r['N_defined_TP']):>10s} {str(r['N_defined_DP']):>10s} "
                 f"{r['trend_TP']:>16s} {r['trend_DP']:>16s} {r['anchor_boots_done']}/{r['anchor_boots']:<5d} {cut}")
    L.append("")
    L.append("== F15 alternative: k Jobs (blocks), each with its own anchor + preflight (not audited) ==")
    L.append(f"{'scenario':30s} {'calib':16s} {'k':>2s} {'total_h':>8s} {'max_block_h':>11s}  blocks (h each)")
    for b in blocks:
        L.append(f"{b['scenario']:30s} {b['calib']:16s} {b['k']:2d} {b['total_h']:8.2f} {b['max_block_h']:11.2f}  "
                 + " | ".join(f"{'+'.join(lab)}={h:.1f}" for lab, h in b["blocks"]))
    return "\n".join(L)


def sockread_warnings(ratio):
    """MGP-35: an arm with lambda < 0.353 prior (0.706 after extension; N=8 A 0.237) loses requests to sock_read."""
    w = []
    if ratio < 0.353:
        w.append("lambda < 0.353 x prior: HI waits can exceed sock_read=600 s -> CLIENT_ERROR (MGP-35)")
    return w


# ----------------------------------------------------------------------------- M2g plan
M2G_CFGS = ("TP2", "TP4", "TP8", "DP2", "DP4", "DP8")


def m2g_boots(dummy_profile="<DUMMY_PROFILE.json>"):
    """§3.  One boot per (config, arm) carries G0(i) health + G1 prompts + (TP agnostic/d44) G3b stress + (fused) G2
    teacher-forced; the N1 boots are the G1-ctl / G2 reference.  Refusal boots G0(ii)/(iii); G4 nsys boots."""
    boots = []
    g1_cmd = lambda tag, cfg: {"type": "cmd", "name": f"g1_{tag}",
                               "argv": ["python3", "workspace/engine-port/scripts/multigpu/m2g_correctness_gate.py",
                                        "serve-g1", "--endpoint", "{ENDPOINT}", "--out", "{OUT}/gates",
                                        "--config", cfg, "--arm", tag]}
    for cfg in ("N1",) + M2G_CFGS:
        cs = config_spec(cfg)
        for arm in arm_order(cfg):
            steps = [g1_cmd(arm, cfg)]
            if arm == "fused":
                steps.append({"type": "cmd", "name": "g2", "argv": [
                    "python3", "workspace/engine-port/scripts/multigpu/m2g_correctness_gate.py", "serve-g2",
                    "--endpoint", "{ENDPOINT}", "--out", "{OUT}/gates", "--config", cfg, "--repeats", "2" if cfg == "N1" else "1"]})
            if cs["mode"] == "TP" and arm in C.G3B_ARMS:
                x = C.G3B_OFFER_MULT * prior(cs["N"], "A")
                steps += [{"type": "warmup", "shape": "A"},
                          dict(type="cell", shape="A", rung="G3B", seed=C.HI_SEEDS[0], ext=0, round=1,
                               block="m2g_g3b", offered=round(x, 6), n_req=int(round(x * C.G3B_DURATION_S)))]
            boots.append(dict(cs, arm=arm, kind="g0", block="m2g", boot_id=f"m2g_{cfg}_{arm}", steps=steps))
    # F4: G5(a) GPU descriptive boot (§3 G5(a), MGP-5): N=1 agnostic under the AUTO-grid config; its decode-busy stream-index
    # histogram is recorded beside the manual run's (mg_drive.g5a_gpu_descriptive).  No exclusion rides on it.
    x5 = prior(1, "A") * C.HI_MULT
    boots.append(dict(config_spec("N1"), arm="agnostic", kind="g5a", block="m2g", boot_id="m2g_g5a_auto_N1",
                      config_rel=C.AUTO_GRID_CONFIG_REL,
                      steps=[{"type": "warmup", "shape": "A"},
                             dict(type="cell", shape="A", rung="G5A", seed=C.HI_SEEDS[0], ext=0, round=1, block="m2g_g5a",
                                  offered=round(x5, 6), n_req=int(round(x5 * C.G5A_AUTO_S)))]))
    for env_k, env_v in C.G0_REFUSED_ENVS:
        extra = {env_k: env_v}
        if (env_k, env_v) == ("PDMUX_R2_POLICY", "hybrid"):
            extra["PDMUX_MODEL_PROFILE"] = dummy_profile
        boots.append(dict(config_spec("TP2"), arm="agnostic", kind="g0ii", block="m2g",
                          boot_id=f"m2g_refuse_{env_k}_{env_v}".replace("=", "_"), steps=[], extra_env=extra,
                          expect_refusal="PD-mux with tp_size="))
    boots.append(dict(config_spec("TP2"), arm="agnostic", kind="g0iii", block="m2g", boot_id="m2g_autogrid_TP2",
                      steps=[], config_rel=C.AUTO_GRID_CONFIG_REL, expect_refusal="PD-mux auto grid (divide_sm)"))
    for n in C.G4_NSYS_TPS:
        cs = config_spec(f"TP{n}")
        x = prior(n, "A") * C.HI_MULT
        st = [{"type": "warmup", "shape": "A"},
              {"type": "cmd", "name": "nsys_start", "argv": ["nsys", "start", "-o", "{BOOT}/nsys", "--force-overwrite=true"]},
              dict(type="cell", shape="A", rung="G4", seed=C.HI_SEEDS[0], ext=0, round=1, block="m2g_g4",
                   offered=round(x, 6), n_req=int(round(x * C.G4_NSYS_S))),
              {"type": "cmd", "name": "nsys_stop", "argv": ["nsys", "stop"]}]
        boots.append(dict(cs, arm="d44", kind="g4", block="m2g", boot_id=f"m2g_g4_TP{n}_d44", steps=st, nsys=True))
    return boots


M2G_G3_COMM_S = 600.0         # estimate: 4 combos x R=500 reverse-order all-reduces + watchdog 120 s, per TP
M2G_G1_S = 150.0              # estimate: 64 prompts x 128 tokens, concurrency 32, + client start
M2G_G2_S = 240.0


def m2g_budget():
    total, rows = 0.0, []
    for b in m2g_boots():
        t = C.EST_BOOT_S if not b.get("expect_refusal") else 120.0
        for st in b["steps"]:
            if st["type"] == "cmd":
                t += {"g1": M2G_G1_S, "g2": M2G_G2_S}.get(st["name"][:2], 20.0)
            elif st["type"] == "warmup":
                t += C.EST_WARMUP_S[st["shape"]]
            elif st["type"] == "cell":
                t += cell_seconds(st["shape"], st["offered"], st["n_req"], prior(b["N"], "A")) \
                     if st["rung"] in ("G4", "G5A") else (st["n_req"] / st["offered"]) + C.EST_DRAIN_S["A"] + C.EST_CELL_GAP_S
        rows.append((b["boot_id"], t))
        total += t
    comm = len(("TP2", "TP4", "TP8")) * M2G_G3_COMM_S
    total += comm
    return {"boots": len(rows), "rows": rows, "g3_comm_s": comm, "total_s": total, "total_h": total / 3600.0,
            "cap_h": C.HARD_CAP_S["m2g"] / 3600.0, "over_cap": total > C.HARD_CAP_S["m2g"]}


# ----------------------------------------------------------------------------- printing / CLI
def fmt_budget(b, title):
    L = [f"== {title} =="]
    for name, nb, t in b["stages"]:
        L.append(f"  {name:16s} boots={nb:3d}  {t / 3600.0:6.2f} h")
    L.append(f"  TOTAL boots={b['boots']} cells={b['cells']}  {b['total_h']:.2f} h   hard cap {b['cap_h']:.0f} h   "
             f"{'OVER CAP' if b['over_cap'] else 'within cap'}")
    return "\n".join(L)


def selftest():
    assert n_req_for(1.6 * 2.99) == 810 and n_req_for(1.6 * 0.69) == 190
    assert n_req_for(1.6 * 8 * 2.99) == 4000 and n_req_for(0.01) == 40
    assert len(arm_order("TP8")) == 4 and arm_order("TP8") == arm_order("TP8")
    assert hi_rung(8, "A", 1)["offered"] == round(2 * 1.6 * 8 * 2.99, 6)
    b = m3_budget()
    assert b["boots"] == 8 + 7 * 8 + 2 + 1 + 3, b["boots"]
    t = {"cells": {"TP2:agnostic": {"included": False}}}
    assert "TP2:agnostic" in excluded_set(t) and runnable_configs(t) == list(C.CONFIG_ORDER)
    assert "TP2" not in runnable_configs({"cells": {"TP2:fused": {"included": False}}})
    print("mg_plan selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=("m3-plan", "m3-budget", "m3-forecast", "m2g-plan", "m2g-budget", "check-pass-table"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--ratio", type=float, default=None, help="lambda/prior for every config")
    ap.add_argument("--ratio-tp", type=float, default=None)
    ap.add_argument("--ratio-dp", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "m3-plan":
        stages = m3_static_boots()
        for name, boots in stages:
            cells = sum(1 for b in boots for s in b["steps"] if s["type"] == "cell")
            print(f"{name:16s} boots={len(boots):2d} cells={cells:3d}  arms={[b['arm'] for b in boots]}")
        return 0
    if a.cmd == "m3-forecast":
        print(fmt_forecast(m3_forecast_table(), block_plan_table()))
        return 0
    if a.cmd == "m3-budget":
        def ratio_of(cfg, a=a):
            if a.ratio is not None:
                return a.ratio
            if cfg.startswith("TP") and a.ratio_tp is not None:
                return a.ratio_tp
            if cfg.startswith("DP") and a.ratio_dp is not None:
                return a.ratio_dp
            return 1.0
        print(fmt_budget(m3_budget(ratio_of), f"M3 budget dry-run (ratio TP/DP/N1 = {ratio_of('TP2')}/{ratio_of('DP2')}/{ratio_of('N1')})"))
        return 0
    if a.cmd == "m2g-plan":
        for b in m2g_boots():
            print(b["boot_id"], b["arm"], [s.get("name", s["type"]) for s in b["steps"]])
        return 0
    if a.cmd == "m2g-budget":
        r = m2g_budget()
        print(f"M2g: boots={r['boots']} g3_comm={r['g3_comm_s'] / 3600:.2f} h total={r['total_h']:.2f} h cap={r['cap_h']:.0f} h "
              f"{'OVER CAP' if r['over_cap'] else 'within cap'}")
        return 0
    if a.cmd == "check-pass-table":
        t, abort = load_pass_table(a.rest[0], a.rest[1], a.rest[2] if len(a.rest) > 2 else None)
        print(abort or "PASS_TABLE_OK")
        return 0 if t else 3
    ap.error("command required")


if __name__ == "__main__":
    sys.exit(main())
