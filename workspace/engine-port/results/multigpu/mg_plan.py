#!/usr/bin/env python3
"""Cell / boot plan, M2g plan, budget dry-run and M2G_PASS_TABLE handling (PREREG_MULTIGPU rev6 §3, §4-1, §4-2, §6).

Pure planning: no GPU, no network.  The Job drivers (mg_drive.py) turn the specs into mg_boot.py runs.

  python3 mg_plan.py m3-plan                 print the static M3 boot/cell skeleton (counts)
  python3 mg_plan.py m3-budget [--ratio R]   per-stage time model -> total vs the 30 h hard cap
  python3 mg_plan.py m2g-plan | m2g-budget
  python3 mg_plan.py check-pass-table PATH SHA256     ABORT_M2G_NOT_PASSED logic
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


def load_pass_table(path, expected_sha256):
    """Returns (table, abort_code).  Missing file / digest mismatch / unparsable -> ABORT_M2G_NOT_PASSED."""
    p = Path(path)
    if not p.exists():
        return None, "ABORT_M2G_NOT_PASSED"
    raw = p.read_bytes()
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
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
def cell_seconds(shape, x, n, lam):
    span = n / x
    drain = C.EST_DRAIN_S[shape]
    return max(span + drain, n / lam if lam > 0 else span + drain) + C.EST_CELL_GAP_S


def boot_seconds(spec, lam_of, x_lo_of=None):
    """lam_of(N, shape) -> true capacity assumed for the arm."""
    t = C.EST_BOOT_S
    N = spec["N"]
    for st in spec["steps"]:
        if st["type"] == "warmup":
            t += C.EST_WARMUP_S[st["shape"]]
        elif st["type"] == "cell":
            t += cell_seconds(st["shape"], st["offered"], st["n_req"], lam_of(N, st["shape"]))
        elif st["type"] == "cmd":
            t += 20.0
    return t


def m3_static_boots(ratio_of=None, assume_ext=None, lo_given=True):
    """The full M3 boot list for a given assumed capacity ratio (ratio_of(cfg) -> lambda/prior).
    x_lo uses the assumed fused HI achieved = ratio * prior.  Extension is assumed iff ratio > 1.42 (audit)."""
    ratio_of = ratio_of or (lambda cfg: 1.0)
    stages = [("gpu_dev", boots_gpu_dev())]
    for cfg in C.CONFIG_ORDER:
        r = ratio_of(cfg)
        N = config_spec(cfg)["N"]
        ext = 1 if (assume_ext if assume_ext is not None else r > 1.42) else 0
        lam = {sh: r * prior(N, sh) for sh in ("A", "B")}
        stages.append((f"{cfg}.boot1", boots_boot1(cfg)))
        if ext:
            stages.append((f"{cfg}.boot1ext", boots_ext(cfg, ("A", "B"))))
        xlo = {sh: lo_rung(lam[sh]) for sh in ("A", "B")}
        stages.append((f"{cfg}.boot2", boots_boot2(cfg, xlo, ext)))
    stages += [("numa", boots_numa()), ("nsys", boots_nsys()), ("anchor2", boots_anchor2())]
    return stages


def m3_budget(ratio_of=None, assume_ext=None):
    ratio_of = ratio_of or (lambda cfg: 1.0)
    rows, total, n_boots, n_cells = [], 0.0, 0, 0
    for name, boots in m3_static_boots(ratio_of, assume_ext):
        t = 0.0
        for b in boots:
            lam_of = lambda N, sh, b=b: ratio_of(b["config"]) * prior(N, sh)
            t += boot_seconds(b, lam_of)
            n_cells += sum(1 for s in b["steps"] if s["type"] == "cell")
        n_boots += len(boots)
        rows.append((name, len(boots), t))
        total += t
    return {"stages": rows, "total_s": total, "total_h": total / 3600.0, "boots": n_boots, "cells": n_cells,
            "cap_h": C.HARD_CAP_S["m3"] / 3600.0, "over_cap": total > C.HARD_CAP_S["m3"]}


def boot_need_s(spec, ratio=0.5):
    """Admission need (worst corner, like N3 §8-3): ratio 0.5 of prior capacity."""
    return boot_seconds(spec, lambda N, sh: ratio * prior(N, sh))


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
                     if st["rung"] == "G4" else (st["n_req"] / st["offered"]) + C.EST_DRAIN_S["A"] + C.EST_CELL_GAP_S
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
    ap.add_argument("cmd", nargs="?", choices=("m3-plan", "m3-budget", "m2g-plan", "m2g-budget", "check-pass-table"))
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
        t, abort = load_pass_table(a.rest[0], a.rest[1])
        print(abort or "PASS_TABLE_OK")
        return 0 if t else 3
    ap.error("command required")


if __name__ == "__main__":
    sys.exit(main())
