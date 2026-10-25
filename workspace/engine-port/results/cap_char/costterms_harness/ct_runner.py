#!/usr/bin/env python3
"""Cost-term harness, stages 1 + 2 -- Job runner and analyzer.  STATUS: CPU-tested against the fake engine only.
NOTHING HERE HAS RUN ON A GPU; writing it is not a PASS of anything and makes no performance claim.

  ct_runner.py plan  --job J-H1|J-A|J-B2|J-H2|J-C [--round 1|2]  print the boot plan (order, argv, env keys, cells)
  ct_runner.py run   --job ... --out RAW [--round N] [--variant-src DIR] [--variant-root DIR] [--resume]
                     [--smoke [--boots a,b] [--cells-per-boot N | --select BOOT:KIND:SPEC,...]]
  ct_runner.py analyze RAW [--summary DIR] [--clk clk.csv]       -> RAW/analysis.json (+ summary copy, CPU)
  ct_runner.py spec-sha                                          ct_spec digest (seal input)

Job entry (experiment-runner): costterms_job.sh sets PDMUX_COSTTERMS_JOB / _ROUND and calls `run`; it refuses to run
unless the harness is sealed (PDMUX_COSTTERMS_SEALED=1) or PDMUX_COSTTERMS_TEST_MODE=1.

Per boot (PREREG §2-2): boot -> facts from the log (label / cap / engine / sticky / instrumentation / capture checks)
-> realized SM (§1-8) -> feasibility of EVERY registered cell (per-cell O, rev3.4 §3-1) written BEFORE any cell -> T4
predictions A and B written
and hashed into events.jsonl BEFORE the first fill -> warm-up + correctness -> cells in seeded random order (T1 bursts
are aborted after the ramp-aware window, decision 28 A-1) -> trailer +
idle 2 x flush -> stop -> DONE (resume skips boots with DONE).  The analyzer re-derives every label from the logs; it
never takes model / load format / cap / D from the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_spec as S  # noqa: E402
import ct_extract as X  # noqa: E402
import ct_fit as F  # noqa: E402
import ct_client as C  # noqa: E402

ENG = HERE.parents[2]
CONFIG_PATH = HERE / S.CONFIG_CANDIDATE
KEEP_ENV = ("PDMUX_ROOT", "PDMUX_PROJECT_ROOT", "PDMUX_DATA", "PDMUX_IO", "PDMUX_COMMIT", "PDMUX_JOB_NAME",
            "PDMUX_ENERGY_SAMPLER")


def _test_mode():
    return os.environ.get("PDMUX_COSTTERMS_TEST_MODE") == "1"


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ plans
def _boot(job, rnd, boot_id, kind, model, engine, cap, *, D=None, variant=4, dummy=False, t1=(), t1_reps=1, t2=(),
          t2p=(), t4=(), o_family="JA", t3=(), t3_reps=S.T3_REPS, t3ref=(), tp=1, gpus="0", context=None, long=()):
    mode = (f"D{D}" if engine == "pdmux_sticky" else "108u" if engine == "pdmux_108u" else "108v")
    arch = S.variant_arch(variant) if model == "nano" else None
    b = {"job": job, "round": rnd, "boot_id": boot_id, "kind": kind, "model": model, "engine": engine, "mode": mode,
         "D": D if engine == "pdmux_sticky" else None, "cap": cap, "variant": variant if model == "nano" else None,
         "dummy": dummy, "t1": [list(c) for c in t1], "t1_reps": t1_reps, "t2": list(t2), "t2p": list(t2p),
         "t4": [list(f) for f in t4], "o_family": o_family,
         "t1_O": {f"{bs}x{c}": S.o_tokens(bs, c, o_family) for bs, c in t1},
         "t3": [list(c) for c in t3], "t3_reps": t3_reps, "t3ref": list(t3ref), "long": list(long),
         "tp": tp, "gpus": gpus, "context": context or S.CONTEXT_LENGTH}
    if t3 or t3ref:
        b["t3_O"] = {f"{bs}x{L}": S.t3_o_tokens(bs, L, D, arch, tp) for bs, L in t3}
        b["t3ref_O"] = {str(L): S.t3_ref_o_tokens(L, D, arch, tp) for L in t3ref}
    return b


def t2_axis(mode):
    """Decision 28 A-2: T2 ("no other load") has a known prefill SM only where prefill runs on 108 SM -- 108u / 108v.
    In a sticky-D boot an empty decode batch sends prefill to group 0 (108 SM) as well, so a T2 there would be a 108-SM
    value under a D label; the plan refuses it (None)."""
    return S.TOTAL_SM if mode in S.T2_MODES else None


def _t3_boot(job, rnd, boot_id, D, variant=4, dummy=False, t1=(), t1_reps=1, o_family="JA"):
    cells = [(bs, L) for bs in S.T3_BS for L in S.T3_L]
    return _boot(job, rnd, boot_id, "T3", "nano", "pdmux_sticky", S.CAP_MAIN, D=D, variant=variant, dummy=dummy,
                 t1=t1, t1_reps=t1_reps, o_family=o_family, t3=cells, t3ref=S.T3_L)


def plan_ja(rnd):
    """J-A round rnd: 7 mode boots (T1 + T2' everywhere, T2 only in 108u / 108v -- decision 28 A-2), T3 boots D24 /
    D44 (TEL ON, §2-2'), cap-invariance boot (round 1 only), OFF diagnostic boots (D44 PD-mux with T2', 108v with T2)."""
    job = f"J-A{rnd}"
    boots = []
    for d in S.D_GRID:
        boots.append(_boot(job, rnd, f"ja{rnd}_D{d}", "T1T2", "nano", "pdmux_sticky", S.CAP_MAIN, D=d,
                           t1=X.t1_cells_for_mode(f"D{d}"), t2p=S.T2P_L_GRID, context=S.JA_CONTEXT))
    boots.append(_boot(job, rnd, f"ja{rnd}_108u", "T1T2", "nano", "pdmux_108u", S.CAP_MAIN,
                       t1=X.t1_cells_for_mode("108u"), t2=S.T2_L_GRID, t2p=S.T2P_L_GRID, context=S.JA_CONTEXT))
    boots.append(_boot(job, rnd, f"ja{rnd}_108v", "T1T2", "nano", "vanilla", S.CAP_MAIN,
                       t1=X.t1_cells_for_mode("108v"), t2=S.T2_L_GRID, t2p=S.T2P_L_GRID, context=S.JA_CONTEXT))
    for d in S.T3_D:
        boots.append(_t3_boot(job, rnd, f"ja{rnd}_t3_D{d}", d))
    boots.append(_boot(job, 1, "ja1_capinv_D44", "T1T2", "nano", "pdmux_sticky", S.CAP_SMALL, D=44, t1=S.CAP_INV_CELLS,
                       context=S.JA_CONTEXT))     # same context as ja*_D44: the cap-invariance pair differs in cap only (A-42)
    boots.append(_boot(job, rnd, f"ja{rnd}_off_D44", "OFF", "nano", "pdmux_sticky", S.CAP_MAIN, D=44,
                       t1=S.OFF_T1_CELLS_JA, t2p=S.OFF_T2_L_JA, context=S.JA_CONTEXT))
    boots.append(_boot(job, rnd, f"ja{rnd}_off_108v", "OFF", "nano", "vanilla", S.CAP_MAIN,
                       t1=S.OFF_T1_CELLS_JA, t2=S.OFF_T2_L_JA, context=S.JA_CONTEXT))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-A|boots")).shuffle(order)
    ordered = [boots[i] for i in order]
    if rnd == 2:      # ABBA: round 2 = round 1 reversed, without the cap-invariance boot
        ordered = [b for b in reversed(ordered) if b["boot_id"] != "ja1_capinv_D44"]
        for b in ordered:
            b["boot_id"] = b["boot_id"].replace("ja1_", "ja2_")
    else:             # decision 30: N=1 T2-ext in its own vanilla boot, context 131,072, the LAST boot of J-A1 only
        ordered += long_boots("J-A|1")
    return {"job": job, "boots": ordered, "deferred": ["J-A1 cap-invariance boot only in round 1",
                                                       "N=1 T2-ext long boot only in J-A1, last (decision 30)"]}


def long_boot(job, rnd, boot_id, tp, gpus, model="nano", engine="vanilla", dummy=False, variant=4):
    """Decision 30 / 33 long boot: P = 108 (vanilla 108v, or the PD-mux engine unpartitioned 108u at TP2 -- decision 33),
    cap LONG_CAP, context LONG_CONTEXT, only the long points (bridge 64k + T2-ext 96k / 124k, ascending), nothing else -- a
    death in it cannot cost any other registered cell.  model 'llama' (J-H1, decision 33 L-8) or 'nano'."""
    return _boot(job, rnd, boot_id, "T1T2", model, engine, S.LONG_CAP, tp=tp, gpus=gpus, variant=variant, dummy=dummy,
                 context=S.LONG_CONTEXT, long=list(S.LONG_L), o_family="VANILLA")


def long_boots(key):
    """The registered long boots of a Job (S.LONG_BOOTS), in order -- always the contiguous suffix of the plan (A-47)."""
    job = key.split("|")[0]
    rnd = int(key.split("|")[1]) if "|" in key else 1
    job = f"J-A{rnd}" if job == "J-A" else job              # J-A boots carry the round in the job name (plan_ja)
    return [long_boot(job, rnd, bid, tp=tp, gpus="0,1" if tp > 1 else "0", model=model, engine=eng, dummy=dummy)
            for bid, model, dummy, eng, tp in S.LONG_BOOTS[key]]


def plan_jh2():
    """J-H2 (§4', decision 25): per dummy variant, PD-mux sticky D24 and D44 boots (TEL ON, T3 boot rule): T1 (32, 2k)
    and (32, 8k) + T3 bs {8, 32} x L {8k, 32k} x 2 + sigma' reference spans.  O from the variant's priors (§4')."""
    boots = []
    for n in S.VARIANTS:
        for d in S.JH2_D:
            boots.append(_t3_boot("J-H2", 1, f"jh2_a{n}_D{d}", d, variant=n, dummy=True, t1=S.JH2_T1_CELLS,
                                  t1_reps=S.JH2_T1_REPS, o_family=f"JH2|{n}|{d}"))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-H2|boots")).shuffle(order)
    return {"job": "J-H2", "boots": [boots[i] for i in order], "deferred": []}


def plan_jc():
    """J-C (§5, rev3.5 draft, O-19): TP2 (both GPUs) and the N=1 anchor (GPU 0 only, the other idle -- O-6 sequential)
    for D44 (T1 + T2') and 108v (T1 + T2), 2 rounds ABBA; TP2 T4 boot (F3); TP2 OFF diagnostic (O-14); optional 96k
    boot LAST (JC_96K_MODE)."""
    boots = []
    for rnd in S.JC_ROUNDS:
        for tp, cap, gpus, tag in ((S.JC_TP, S.JC_TP2_CAP, "0,1", "tp2"), (1, S.JC_ANCHOR_CAP, "0", "n1")):
            boots.append(_boot("J-C", rnd, f"jc{rnd}_{tag}_D44", "T1T2", "nano", "pdmux_sticky", cap, D=44,
                               t1=S.JC_T1_CELLS, t1_reps=S.JC_T1_REPS, t2p=S.JC_T2_L, o_family="JC|44", tp=tp,
                               gpus=gpus))
            boots.append(_boot("J-C", rnd, f"jc{rnd}_{tag}_108v", "T1T2", "nano", "vanilla", cap,
                               t1=S.JC_T1_CELLS, t1_reps=S.JC_T1_REPS, t2=S.JC_T2_L, o_family="VANILLA", tp=tp,
                               gpus=gpus))
    for rnd in S.JC_ROUNDS:        # decision 33 (L-14): cap-32 N=1 anchor for (32, 32k) D44, own O, one per round
        boots.append(_boot("J-C", rnd, f"jc{rnd}_n1c32_D44", "T1T2", "nano", "pdmux_sticky", S.JC_ANCHOR32_CAP,
                           D=S.JC_ANCHOR32_MODE_D, t1=S.JC_ANCHOR32_CELLS, t1_reps=S.JC_T1_REPS, o_family="JC1|44", tp=1,
                           gpus="0"))
    boots.append(_boot("J-C", 1, "jc_tp2_t4", "T4", "nano", "pdmux_sticky", S.JC_TP2_CAP, D=44, t4=S.JC_T4_FILLS,
                       tp=S.JC_TP, gpus="0,1"))
    boots.append(_boot("J-C", 1, "jc_tp2_off_D44", "OFF", "nano", "pdmux_sticky", S.JC_TP2_CAP, D=44,
                       t1=S.JC_OFF_T1_CELLS, o_family="JC|44", tp=S.JC_TP, gpus="0,1"))
    r1 = [b for b in boots if b["round"] == 1]
    order = list(range(len(r1)))
    random.Random(S.order_seed("J-C|boots")).shuffle(order)
    r1 = [r1[i] for i in order]
    r2 = {b["boot_id"]: b for b in boots if b["round"] == 2}
    ordered = r1 + [r2[b["boot_id"].replace("jc1_", "jc2_")] for b in reversed(r1) if b["boot_id"].startswith("jc1_")]
    if S.JC_96K_MODE == "included":       # decision 30: TP2 long boot LAST (96k required, 124k if feasible); decision 33: + 108u
        ordered += long_boots("J-C")
    elif S.JC_96K_MODE != "deleted":
        raise ValueError(S.JC_96K_MODE)
    return {"job": "J-C", "boots": ordered, "deferred": [] if S.JC_96K_MODE != "deleted"
            else ["TP2 long point deleted (NOT the decision-30 default; disclose the lost TP2 long serving point)"]}


def plan_jh1():
    """J-H1 (§4'): boot A per dummy variant (vanilla 108v, cap 128) + real a4 + Llama + OFF diagnostic (dummy a4)."""
    boots = []
    for n in S.VARIANTS:
        boots.append(_boot("J-H1", 1, f"jh1_dummy_a{n}", "T1T2", "nano", "vanilla", S.CAP_MAIN, variant=n, dummy=True,
                           t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L,
                           t4=[(L, S.JH_T4_R) for L in S.JH_T4_L], o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_real_a4", "T1T2", "nano", "vanilla", S.CAP_MAIN, variant=4, dummy=False,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, t4=[(L, S.JH_T4_R) for L in S.JH_T4_L],
                       o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_llama", "T1T2", "llama", "vanilla", S.CAP_MAIN,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, t4=[(L, S.JH_T4_R) for L in S.JH_LLAMA_T4_L],
                       o_family="VANILLA"))
    boots.append(_boot("J-H1", 1, "jh1_off_dummy_a4", "OFF", "nano", "vanilla", S.CAP_MAIN, variant=4, dummy=True,
                       t1=S.JH_T1_CELLS, t1_reps=S.JH_T1_REPS, t2=S.JH_T2_L, o_family="VANILLA"))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-H1|boots")).shuffle(order)
    # decision 33 (L-8): Llama + real-a4 long boots LAST (vanilla 108v, context 131,072, cap 48): PREREG §4 (4) contrast
    return {"job": "J-H1", "boots": [boots[i] for i in order] + long_boots("J-H1"),
            "deferred": ["J-H2 (B/C boots, T3) -- stage 2"]}


def plan_jb2():
    """J-B2 (§1-4, O-15 (b), O-8): Nano T4 9 boots (sticky D44 PD-mux) + 1 fused_sym boot."""
    boots = [_boot("J-B2", 1, "jb2_cap48", "T4", "nano", "pdmux_sticky", S.CAP_SMALL, D=S.T4_ARM_D,
                   t4=[(L, S.T4_R[L]) for L in S.T4_L])]
    for L in S.T4_L:
        boots.append(_boot("J-B2", 1, f"jb2_capCA_L{L}", "T4", "nano", "pdmux_sticky", S.T4_CAP_CA[L], D=S.T4_ARM_D,
                           t4=[(L, S.T4_R[L])]))
        boots.append(_boot("J-B2", 1, f"jb2_capCB_L{L}", "T4", "nano", "pdmux_sticky", S.T4_CAP_CB[L], D=S.T4_ARM_D,
                           t4=[(L, S.T4_R[L])]))
    boots.append(_boot("J-B2", 1, "jb2_fused_cap48", "T4", "nano", "vanilla", S.CAP_SMALL,
                       t4=[(L, S.T4_R[L]) for L in S.T4_L]))
    order = list(range(len(boots)))
    random.Random(S.order_seed("J-B2|boots")).shuffle(order)
    return {"job": "J-B2", "boots": [boots[i] for i in order], "deferred": []}


def make_plan(job, rnd=1):
    if job == "J-A":
        return plan_ja(rnd)
    if job == "J-H1":
        return plan_jh1()
    if job == "J-B2":
        return plan_jb2()
    if job == "J-H2":
        return plan_jh2()
    if job == "J-C":
        return plan_jc()
    raise ValueError(job)


def plan_problems(plan):
    """Structural checks of a plan (matrix §2-2', caps, T3 placement, registered cell sets)."""
    bad = []
    for b in plan["boots"]:
        if b["kind"] not in S.INSTR:
            bad.append(f"{b['boot_id']}: unknown kind {b['kind']}")
            continue
        if b.get("t3") and not S.INSTR[b["kind"]][3]:
            bad.append(f"{b['boot_id']}: T3 cells in a TEL-OFF boot (matrix §2-2')")
        if b["t4"] and b["kind"] == "OFF":
            bad.append(f"{b['boot_id']}: T4 fill in an OFF diagnostic boot")
        anchor32 = (b["job"] == "J-C" and b["tp"] == 1 and b["cap"] == S.JC_ANCHOR32_CAP)
        if b["kind"] == "T1T2" and b["t1"] and b["cap"] not in (S.CAP_MAIN, S.CAP_SMALL) and not anchor32:
            bad.append(f"{b['boot_id']}: T1 boot cap {b['cap']}")
        if b["t2"] and t2_axis(b["mode"]) is None:
            bad.append(f"{b['boot_id']}: T2 in a {b['mode']} boot (decision 28 A-2: T2 only at P = 108; use T2')")
        if any(b["t1_O"].get(f"{bs}x{c}") != S.o_tokens(bs, c, b["o_family"]) for bs, c in b["t1"]):
            bad.append(f"{b['boot_id']}: T1 output length differs from the registered O rule")
        if b["job"].startswith("J-A") and b["kind"] == "T1T2" and b["cap"] == S.CAP_MAIN:
            want = sorted(map(list, X.t1_cells_for_mode(b["mode"])))
            if sorted(b["t1"]) != want:
                bad.append(f"{b['boot_id']}: T1 cell set != registered ({len(b['t1'])} vs {len(want)})")
        # ---- stage 2
        if b["kind"] == "T3":
            if b["engine"] != "pdmux_sticky" or b["D"] not in S.T3_D:
                bad.append(f"{b['boot_id']}: T3 boot must be PD-mux sticky D in {S.T3_D}")
            arch = S.variant_arch(b["variant"])
            if any(b["t3_O"].get(f"{bs}x{L}") != S.t3_o_tokens(bs, L, b["D"], arch, b["tp"]) for bs, L in b["t3"]):
                bad.append(f"{b['boot_id']}: T3 decode output length differs from the registered rule")
            if sorted(map(int, b.get("t3ref_O", {}))) != sorted({L for _, L in b["t3"]}):
                bad.append(f"{b['boot_id']}: sigma' reference L set != the T3 L set (same boot, same L)")
        if b["job"].startswith("J-A") and b["kind"] in ("T1T2", "OFF") and not b["long"] and b["context"] != S.JA_CONTEXT:
            bad.append(f"{b['boot_id']}: J-A T1 boot context {b['context']} != {S.JA_CONTEXT} (decision 33 L-7: (8, 64k) "
                       f"needs c_in + O = 79,459 < context)")
        if anchor32:            # decision 33 L-14: the cap-32 anchor carries exactly the (32, 32k) D44 cell with its own O
            if (b["engine"], b["D"], sorted(map(list, b["t1"])), b["o_family"], b["gpus"]) != \
                    ("pdmux_sticky", S.JC_ANCHOR32_MODE_D, sorted(map(list, S.JC_ANCHOR32_CELLS)), "JC1|44", "0"):
                bad.append(f"{b['boot_id']}: cap-32 anchor must be sticky D{S.JC_ANCHOR32_MODE_D} on GPU 0 with only "
                           f"{S.JC_ANCHOR32_CELLS} and its own O family JC1|44")
        elif b["job"] == "J-C" and b["tp"] == 1 and b["cap"] != S.JC_ANCHOR_CAP:
            bad.append(f"{b['boot_id']}: N=1 anchor cap {b['cap']} != {S.JC_ANCHOR_CAP} (§5-2)")
        if b["job"] == "J-C":
            if b["tp"] == 1 and b["gpus"] != "0":
                bad.append(f"{b['boot_id']}: N=1 anchor must use GPU 0 only (O-6 sequential)")
            if b["tp"] == 1 and not anchor32 and not b["long"] and \
                    sorted(map(list, b["t1"])) != sorted(map(list, S.JC_T1_CELLS)):
                bad.append(f"{b['boot_id']}: N=1 anchor T1 cell set != registered J-C grid (decision 33 L-15 bs 48 row)")
            if b["tp"] == S.JC_TP and not b["long"] and (b["cap"] != S.JC_TP2_CAP or b["gpus"] != "0,1"):
                bad.append(f"{b['boot_id']}: TP2 boot cap / GPUs {b['cap']} {b['gpus']}")
            if b["tp"] == S.JC_TP and b["kind"] == "T1T2" and b["t1"] and not b["long"] and \
                    sorted(map(list, b["t1"])) != sorted(map(list, S.JC_T1_CELLS)):
                bad.append(f"{b['boot_id']}: TP2 T1 cell set != registered J-C grid (decision 33 L-15 bs 48 row)")
            if b["tp"] == S.JC_TP and b["long"] and b["gpus"] != "0,1":
                bad.append(f"{b['boot_id']}: TP2 long boot GPUs {b['gpus']}")
        for L in list(b["t2"]) + list(b["t2p"]):
            if L + 1 >= b["context"]:
                bad.append(f"{b['boot_id']}: L {L} + 1 > context_length {b['context']}")
        # ---- decision 30 long boots: separate, last, long-only, context 131,072, ascending L, P = 108
        if b["long"]:
            i = plan["boots"].index(b)
            if any(not x["long"] for x in plan["boots"][i + 1:]):       # decision 33 (A-47): the long boots are a suffix
                bad.append(f"{b['boot_id']}: a long boot must be in the LAST contiguous block of long boots of the Job "
                           f"(decision 30 / 33: a measurement boot after it would be exposed to its death)")
            if b["t1"] or b["t2"] or b["t2p"] or b.get("t3") or b["t4"]:
                bad.append(f"{b['boot_id']}: a long boot carries only long points (decision 30: separate boot)")
            if b["context"] != S.LONG_CONTEXT:
                bad.append(f"{b['boot_id']}: long boot context {b['context']} != {S.LONG_CONTEXT}")
            if list(b["long"]) != sorted(b["long"]):
                bad.append(f"{b['boot_id']}: long L not ascending (a death at a larger L must keep the smaller)")
            if t2_axis(b["mode"]) is None:
                bad.append(f"{b['boot_id']}: long points in a {b['mode']} boot (T2 needs P = 108, decision 28 A-2)")
            for L in b["long"]:
                if L + S.LONG_MAX_NEW >= b["context"]:
                    bad.append(f"{b['boot_id']}: ABORT_CONTEXT_TOO_SHORT L {L} + {S.LONG_MAX_NEW} >= context_length "
                               f"{b['context']}")
    return bad


def registry_problems(plan):
    """Decision 33 (A-47): a FULL Job plan carries exactly the long boots S.LONG_BOOTS registers for it (id, model, load
    format, engine, tp), as its last boots in that order, and every boot of the registered J-A / J-C / J-H1 sets; a
    sub-plan (smoke / tests) is not checked.  `plan_problems` checks the structure, this checks the count."""
    job = plan["job"]
    if job.startswith("J-A") and len(job) == 4:
        key = f"J-A|{job[3]}"
    else:
        key = job
    want = S.LONG_BOOTS.get(key, ())
    if key == "J-C" and S.JC_96K_MODE == "deleted":
        want = ()
    got = [(b["boot_id"], b["model"], b["dummy"], b["engine"], b.get("tp", 1)) for b in plan["boots"] if b["long"]]
    bad = []
    if got != [tuple(w) for w in want]:
        bad.append(f"long boots {got} != registered {list(want)} (decision 33)")
    tail = [b["boot_id"] for b in plan["boots"]][len(plan["boots"]) - len(want):] if want else []
    if want and tail != [w[0] for w in want]:
        bad.append(f"the registered long boots are not the last boots in order: tail {tail}")
    if job == "J-C":
        n32 = sorted(b["boot_id"] for b in plan["boots"] if b["tp"] == 1 and b["cap"] == S.JC_ANCHOR32_CAP)
        if n32 != [f"jc{r}_n1c32_D44" for r in S.JC_ROUNDS]:
            bad.append(f"cap-32 anchor boots {n32} != one per round (decision 33 L-14)")
    return bad


# ============================================================================ server argv / env
def model_args(b, variant_dirs):
    if b["model"] == "llama":
        return ["--model-path", S.LLAMA_ID, "--revision", S.LLAMA_REV]
    if b["dummy"]:
        return ["--model-path", str(variant_dirs[b["variant"]]), "--load-format", "dummy"]
    return ["--model-path", S.NANO_ID, "--revision", S.NANO_REV]


def server_argv(b, variant_dirs):
    a = model_args(b, variant_dirs) + [
        "--trust-remote-code", "--dtype", "bfloat16", "--attention-backend", S.ATTENTION_BACKEND,
        "--disable-radix-cache", "--mem-fraction-static", str(S.MEM_FRACTION), "--max-running-requests", str(b["cap"]),
        "--context-length", str(b.get("context", S.CONTEXT_LENGTH)), "--random-seed", str(S.SERVER_SEED), "--disable-piecewise-cuda-graph",
        "--chunked-prefill-size", "-1", "--disable-overlap-schedule", "--max-prefill-tokens", str(S.MAX_PREFILL_TOKENS),
        "--host", "127.0.0.1"]
    if b["model"] == "nano":
        a += ["--max-mamba-cache-size", str(b["cap"])]
    if b.get("tp", 1) > 1:
        a += ["--tp-size", str(b["tp"])]
    if b["engine"].startswith("pdmux"):
        a += ["--enable-pdmux", "--pdmux-config-path", str(CONFIG_PATH)]
    return a


def boot_env(b, bdir, base=None, run_id="local"):
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items() if not (k.startswith("PDMUX_") and k not in KEEP_ENV)}
    env.pop("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", None)
    env.update(PDMUX_RUN_ID=run_id, PDMUX_WORKLOAD_ID=f"costterms_{b['boot_id']}", PYTHONUNBUFFERED="1", **S.HEARTBEAT_ENV)
    env["CUDA_VISIBLE_DEVICES"] = b.get("gpus", "0")         # J-C: TP2 "0,1", anchor "0" (O-6); single-GPU Jobs "0"
    for k in ("FAKE_CT_TRUTH",):
        if k in base:
            env[k] = base[k]
    i1, i46, i5, tel = S.INSTR[b["kind"]]
    if i46:
        env["PDMUX_DECODE_PROBE_PATH"] = str(bdir / "decode_probe.jsonl")
    if i5:
        env["PDMUX_ADMISSION_PATH"] = str(bdir / "admission.jsonl")
    if tel and b["engine"].startswith("pdmux"):
        env["PDMUX_TELEMETRY_PATH"] = str(bdir / "tel.jsonl")
        env["PDMUX_PHASE_EVENTS"] = "1"
    if b["engine"].startswith("pdmux"):
        env.update(PDMUX_ENGINE_MODE="pdmux", PDMUX_R2_POLICY="fixed",
                   PDMUX_R2_FIXED_DSM=str(b["D"] if b["engine"] == "pdmux_sticky" else S.FIXED_DSM_108U),
                   PDMUX_GREEN_READOUT="1", PDMUX_GREEN_READOUT_PATH=str(bdir / "green.json"))
        if b["engine"] == "pdmux_sticky":
            env["PDMUX_STICKY_PARTITION"] = "1"
    else:
        env["PDMUX_ENGINE_MODE"] = "vanilla"
    return env


# ============================================================================ servers
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class ProcessServer:
    """Real `python -m sglang.launch_server` (or, in test mode only, PDMUX_CT_SERVER_CMD)."""

    def __init__(self, bdir, argv, env, boot_timeout=1200):
        self.bdir, self.argv, self.env, self.boot_timeout = bdir, argv, env, boot_timeout
        self.log = bdir / "server.log"
        self.proc = None

    def start(self):
        self.port = free_port()
        cmd = [sys.executable, "-m", "sglang.launch_server"]
        if _test_mode() and os.environ.get("PDMUX_CT_SERVER_CMD"):
            cmd = shlex.split(os.environ["PDMUX_CT_SERVER_CMD"])
        self.lf = open(self.log, "wb")
        self.proc = subprocess.Popen(cmd + self.argv + ["--port", str(self.port)], stdout=self.lf,
                                     stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.tr = C.HttpTransport(self.port)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.boot_timeout:
            if self.proc.poll() is not None:
                return "died"
            if self.tr.health():
                return "ok"
            time.sleep(0.1 if _test_mode() else 2)
        return "timeout"

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_size(self):
        try:
            return self.log.stat().st_size
        except OSError:
            return 0

    def stop(self):
        if self.proc is not None:
            for sig, wait in ((signal.SIGTERM, 3 if _test_mode() else 20), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                t0 = time.monotonic()
                while time.monotonic() - t0 < wait and self.proc.poll() is None:
                    time.sleep(0.1)
                if self.proc.poll() is not None:
                    break
            self.lf.close()
        return {"returncode": self.proc.poll() if self.proc else None}


class VirtualServer:
    """The fake engine core in-process on a virtual clock (deterministic CPU e2e)."""

    def __init__(self, bdir, argv, env, truth=None):
        import ct_fake_engine as FE
        self.FE, self.bdir, self.argv, self.env, self.truth = FE, bdir, argv, env, truth
        self.log = bdir / "server.log"

    def start(self):
        self.lf = open(self.log, "a", buffering=1)
        try:
            self.core = self.FE.EngineCore(self.argv, self.env, self.truth, self.lf.write)
        except RuntimeError as exc:
            self.lf.write(f"Traceback (most recent call last):\nRuntimeError: {exc}\n")
            self.core = None
            return "died"
        for ln in self.core.boot_lines():
            self.lf.write(f"[boot] {ln}\n")
        self.tr = self.FE.VirtualTransport(self.core)
        return "ok"

    def alive(self):
        return self.core is not None and not self.core.dead

    def log_size(self):
        self.lf.flush()
        return self.log.stat().st_size

    def stop(self):
        if self.core is not None:
            if not self.core.dead:
                self.core.flush(self.tr.now())
            self.core.close()
        self.lf.close()
        return {"returncode": 0}


# ============================================================================ variants
def smoke_engine_digest(out):
    """Audit r1 m13: the Workspace smoke does not pass costterms_job.sh, so the installed engine tree was never compared with
    ENGINE_DIGEST.txt -- yet smoke values decide sealed inputs (O-22 caps, RAMP_SIGMA).  Record the check (or why there is
    none) in SMOKE_ENGINE_DIGEST.json next to the SMOKE_UNSEALED marker."""
    data = os.environ.get("PDMUX_DATA", "/data")
    manifest = os.environ.get("PDMUX_COSTTERMS_MANIFEST") or f"{data}/runs/{os.environ.get('PDMUX_JOB_NAME', '')}/meta/runtime_source_manifest.sha256"
    rec = {"manifest": manifest, "sealed_file": str(HERE / "ENGINE_DIGEST.txt")}
    if not Path(manifest).exists():
        rec.update(status="NOT_CHECKED", why="no runtime_source_manifest.sha256 (smoke outside job_entry / install_runtime)")
    else:
        r = subprocess.run([sys.executable, str(ENG / "scripts" / "bootstrap" / "engine_digest.py"), "check", "--manifest", manifest,
                            "--sealed-file", str(HERE / "ENGINE_DIGEST.txt")], capture_output=True, text=True)
        rec.update(status="MATCH" if r.returncode == 0 else "MISMATCH", rc=r.returncode, output=(r.stdout + r.stderr)[-400:])
    dump(rec, Path(out) / "SMOKE_ENGINE_DIGEST.json")
    return rec


def gpu_snapshot(path):
    """Audit r1 m8: nvidia-smi memory / utilisation / clocks / throttle reasons before and after every boot (best effort, 20 s
    cap; 'unavailable' where there is no nvidia-smi, e.g. the CPU tests).  Descriptive, never used by an analysis rule."""
    import shutil as _sh
    exe = _sh.which("nvidia-smi")
    if exe is None or _test_mode():
        Path(path).write_text("nvidia-smi unavailable (CPU test / no GPU)\n")
        return
    try:
        r = subprocess.run([exe, "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,clocks.sm,"
                            "clocks_throttle_reasons.active", "--format=csv"], capture_output=True, text=True, timeout=20)
        Path(path).write_text(f"t_wall={time.time()}\n{r.stdout}{r.stderr}")
    except Exception as exc:  # noqa: BLE001 - never let a diagnostic stop a measurement
        Path(path).write_text(f"nvidia-smi failed: {exc!r}\n")


def prepare_variants(src, root, plan):
    """Build the dummy variant directories with scripts/compo/make_variant.py (config + tokenizer, no weights)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ct_make_variant", ENG / "scripts" / "compo" / "make_variant.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    out = {}
    for n in sorted({b["variant"] for b in plan["boots"] if b["model"] == "nano" and b["dummy"]}):
        d = Path(root) / f"a{n}"
        if not (d / "config.json").exists():
            mv.build_variant(str(src), str(d), n, force=True)
        out[n] = d
    return out


# ============================================================================ one boot
class Ledger:
    def __init__(self, path):
        self.path = path

    def __call__(self, event, what="", **kw):
        with open(self.path, "a") as f:
            f.write(json.dumps({"event": event, "what": what, "t_wall": time.time(), **kw}, default=str) + "\n")


def t4_predict(f, cap, fills):
    mt = X.mem_terms(f)
    if mt is None:
        return None
    return [X.cmem_predictions(mt["pool_bytes"], mt["s"], mt["k"], cap, L, R) for L, R in fills]


def rank_name(name, tp, r=0):
    """multigpu_guard.rank_suffixed_path: 'decode_probe.jsonl' -> 'decode_probe.tp0.jsonl' when tp > 1."""
    if tp <= 1:
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}.tp{r}.{ext}"


def cell_key(kind, what, k=0):
    """Cell identity used by selection (smoke) and cell-level resume: 't1|8x2048|0', 't3|32x8192|1', 't2p|4096|0'."""
    if isinstance(what, (tuple, list)):
        what = "x".join(map(str, what))
    return f"{kind}|{what}|{k}"


def registered_cells(b, feas=None, feas3=None, feasL=None):
    """Every cell of a boot in REGISTERED order (before the seeded shuffle): t1 (feasible, x reps), t2, t2p, t3 (feasible,
    x reps), t3ref (for the L of feasible t3 cells), t4, long.  feas / feas3 None -> no feasibility filter (selection)."""
    out = []
    for c_ in [tuple(x) for x in b["t1"]]:
        fe = next((x for x in feas if (x["bs"], x["c"]) == c_), None) if feas is not None else {"run": True}
        if fe is not None and fe["run"]:
            out += [("t1", c_, k) for k in range(b["t1_reps"])]
    out += [("t2", L, 0) for L in b["t2"]] + [("t2p", L, 0) for L in b["t2p"]]
    ok3 = []
    for c_ in [tuple(x) for x in b.get("t3", [])]:
        fe = next((x for x in feas3 if (x["bs"], x["L"]) == c_), None) if feas3 is not None else {"run": True}
        if fe is not None and fe["run"]:
            ok3.append(c_)
            out += [("t3", c_, k) for k in range(b["t3_reps"])]
    out += [("t3ref", L, 0) for L in b.get("t3ref", []) if feas3 is None or any(L == c_[1] for c_ in ok3)]
    out += [("t4", tuple(x), 0) for x in b["t4"]]
    out += [("long", L, 0) for L in b.get("long", []) if feasL is None or any(r["L"] == L and r["run"] for r in feasL)]
    return out


def long_feasibility(b, f):
    """Decision 30: each long L against the boot log -- per-rank KV token pool and the LOGGED context_len (the server
    may clamp or reject a context; the plan value is not trusted).  A context failure is ABORT_CONTEXT_TOO_SHORT."""
    mt = X.mem_terms(f)
    kv = mt["kv_tokens"] if mt else None
    ctx = X.logged_context(f)
    rows = []
    for L in b.get("long", []):
        r = S.feasible_long(L, kv, ctx)
        lab = None if not r else ("ABORT_CONTEXT_TOO_SHORT" if any(x.startswith("ABORT_CONTEXT_TOO_SHORT") for x in r)
                                  else "INFEASIBLE_BY_RULE")
        rows.append({"L": L, "run": not r, "label": lab, "reasons": r, "kv_tokens": kv, "context_logged": ctx,
                     "required": L in S.JC_LONG_REQUIRED and b.get("tp", 1) > 1})
    return rows


def select_cells(b, cells_per_boot=None, select=None):
    """Smoke / resume selection -> sorted list of cell keys kept for this boot (None = all).  `select` = list of
    'BOOT:KIND:SPEC' (exact registered cells; an unregistered request raises); `cells_per_boot` N = first N cells of
    each kind (t1 sorted by (O, -bs, c): the cheapest bursts with bs > 1 first, others in registered order)."""
    if not select and cells_per_boot is None:
        return None
    reg = registered_cells(b)
    keys = {cell_key(*c) for c in reg}
    keep = set()
    if select:
        for item in select:
            bid, kind, spec = item.split(":", 2)
            if bid != b["boot_id"]:
                continue
            k = cell_key(kind, spec, 0)
            if k not in keys:
                raise SystemExit(f"ABORT_SELECT: {item} is not a registered cell of {bid} (registered: {sorted(keys)[:12]} ...)")
            keep.add(k)
    if cells_per_boot is not None and not select:
        by = {}
        for c in reg:
            by.setdefault(c[0], []).append(c)
        for kind, cs in by.items():
            if kind == "t1":
                cs = sorted(cs, key=lambda c: (b["t1_O"][f"{c[1][0]}x{c[1][1]}"], -c[1][0], c[1][1], c[2]))
            keep |= {cell_key(*c) for c in cs[:cells_per_boot]}
    return sorted(keep)


def t3_feasibility(b, f):
    """§3-1 extended to T3 cells and the sigma' reference (feasible_t3), from the boot log only."""
    mt = X.mem_terms(f)
    kv = mt["kv_tokens"] if mt else None
    cap_list = f.get("capture_bs")
    cap = X.logged_cap(f)
    rows = []
    for bs, L in [tuple(x) for x in b.get("t3", [])]:
        O = b["t3_O"][f"{bs}x{L}"]
        r1 = S.feasible_t3(bs, L, O, kv, cap, X.bs_capturable(bs, cap_list), b.get("context"))
        r2 = S.feasible_t3(S.T3_REF_BS, L, b["t3ref_O"][str(L)], kv, cap, X.bs_capturable(S.T3_REF_BS, cap_list),
                           b.get("context"))
        reasons = r1 + [f"reference: {x}" for x in r2]
        rows.append({"bs": bs, "L": L, "O": O, "O_ref": b["t3ref_O"][str(L)], "run": not reasons,
                     "label": None if not reasons else "INFEASIBLE_BY_RULE", "reasons": reasons})
    return rows


def _run_cell(tr, b, kind, what, k, f, o_map, led):
    if kind == "t1":
        bs, c = what
        return C.run_t1_cell(tr, f"t1_b{bs}_c{c}_k{k}", bs, c, b["round"], o_map[(bs, c)], mark=led)
    if kind == "t2":
        return C.run_t2(tr, f"t2_L{what}", what, b["round"], mark=led)
    if kind == "t2p":
        return C.run_t2prime(tr, f"t2p_L{what}", what, b["round"], mark=led)
    if kind == "t3":
        bs, L = what
        return C.run_t3_cell(tr, f"t3_b{bs}_L{L}_k{k}", bs, L, S.T3_M[L], b["round"], b["t3_O"][f"{bs}x{L}"],
                             mark=led, rep=k)
    if kind == "t3ref":
        return C.run_t3_ref(tr, f"t3ref_L{what}", what, b["round"], b["t3ref_O"][str(what)], mark=led)
    if kind == "long":
        return C.run_long_point(tr, f"long_L{what}", what, b["round"], mark=led)
    L, R = what
    return C.run_t4_fill(tr, f"t4_L{L}", L, R, X.logged_cap(f) + S.T4_FILL_EXTRA, mark=led)


def run_boot(b, out, variant_dirs, make_server, run_id="local"):
    bdir = Path(out) / "boots" / b["boot_id"]
    if (bdir / "DONE").exists():
        return load(bdir / "boot_result.json", {})
    bdir.mkdir(parents=True, exist_ok=True)
    for p in bdir.iterdir():                     # a partial boot directory is redone from scratch (resume: run_job)
        if p.is_file():
            p.unlink()
    tp = b.get("tp", 1)
    led = Ledger(bdir / "events.jsonl")
    gpu_snapshot(bdir / "gpu_before.txt")
    argv = server_argv(b, variant_dirs)
    env = boot_env(b, bdir, run_id=run_id)
    dump({"plan": b, "argv": argv, "pdmux_env": {k: v for k, v in sorted(env.items()) if k.startswith("PDMUX_")},
          "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "spec_sha256": S.spec_sha256()}, bdir / "boot.json")
    led("boot_start", b["boot_id"])
    srv = make_server(bdir, argv, env)
    st = srv.start()
    res = {"boot_id": b["boot_id"], "start": st, "problems": [], "labels": []}
    if b.get("after_prior_death"):
        res["labels"].append("AFTER_PRIOR_DEATH")
    if st != "ok":
        res["labels"].append("SERVER_DEATH")
        if tp > 1:
            res["labels"].append("TP2_NOT_BOOTABLE")
        res["problems"].append(f"boot {st}")
        srv.stop()
        dump(res, bdir / "boot_result.json")
        (bdir / "DONE").write_text("boot failed\n")
        return res
    text = srv.log.read_text(errors="replace")
    cfg_text = CONFIG_PATH.read_text() if b["engine"].startswith("pdmux") else None
    f = X.parse_boot_log(text, env, cfg_text)
    if tp > 1:
        f["rank_banners"] = X.rank_banners(text)
    dump(f, bdir / "facts.json")
    led("facts", "boot log parsed")
    probs = X.label_problems(b, f) + X.instrumentation_problems(b["kind"], env, f, None)
    if b["kind"] != "T4" and b["cap"] == S.CAP_MAIN and (b["t1"] or b.get("t3")):
        cg = X.capture_problems(f.get("capture_bs"))
        if cg:
            probs += [f"ABORT_CG_BS: {p}" for p in cg]
    res["problems"] = probs
    aborting = [p for p in probs if p.startswith("ABORT_")]
    green = load(bdir / rank_name("green.json", tp)) if b["engine"].startswith("pdmux") else None
    sm = X.realized_sm(b, green)
    if tp > 1 and b["engine"].startswith("pdmux"):
        sm_r = [X.realized_sm(b, load(bdir / rank_name("green.json", tp, r))) for r in range(tp)]
        sm["per_rank"] = sm_r
        if not all(x["ok"] for x in sm_r):
            sm.update(ok=False, label=sm["label"] or next(x["label"] for x in sm_r if not x["ok"]))
    dump(sm, bdir / "sm.json")
    if aborting:
        res["labels"].append("ABORTED")
        srv.stop()
        dump(res, bdir / "boot_result.json")
        (bdir / "DONE").write_text("aborted\n")
        led("boot_abort", "; ".join(aborting))
        return res
    o_map = {(bs, c): b["t1_O"][f"{bs}x{c}"] for bs, c in b["t1"]}
    feas = X.feasibility([tuple(c) for c in b["t1"]], f, X.logged_cap(f), o_map, X.logged_context(f))   # audit r1 m5: LOGGED context
    dump(feas, bdir / "feasibility.json")
    feas3 = t3_feasibility(b, f) if b.get("t3") else None
    if feas3 is not None:
        dump(feas3, bdir / "feasibility_t3.json")
    feasL = long_feasibility(b, f) if b.get("long") else None
    if feasL is not None:
        dump(feasL, bdir / "feasibility_long.json")
        res["problems"] += [f"{r['label']}: L {r['L']} not sent ({'; '.join(r['reasons'])})" for r in feasL if not r["run"]]
    led("feasibility_written", sha256=sha256_file(bdir / "feasibility.json"))
    if b["t4"]:
        pred = t4_predict(f, X.logged_cap(f), [tuple(x) for x in b["t4"]])
        dump(pred, bdir / "t4_prediction.json")
        led("t4_prediction_sealed", sha256=sha256_file(bdir / "t4_prediction.json"))
    tr = srv.tr
    wu = C.run_warmup(tr, b["boot_id"])
    tr.sleep(S.PROBE_WINDOW_S * 1.2)
    recs = X.read_jsonl(bdir / rank_name("decode_probe.jsonl", tp))
    cg_hit = any(e.get("event") == "decode_forward" and e.get("graph") == "hit" for e in recs) if recs else \
        ("cuda graph: True" in srv.log.read_text(errors="replace"))
    wu["cg_canary"] = cg_hit
    if not (wu["warmup_ok"] and wu["greedy_ok"] and wu["cg_canary"]):
        res["problems"].append(f"CORRECTNESS: warmup {wu['warmup_ok']} greedy {wu['greedy_ok']} cg_canary {wu['cg_canary']}")
    dump(wu, bdir / "correctness.json")
    led("correctness", json.dumps(wu))
    for c_ in [tuple(x) for x in b["t1"]]:
        if next((x for x in feas if (x["bs"], x["c"]) == c_), None) is None:   # every registered cell has a row
            res["problems"].append(f"FEASIBILITY_ROW_MISSING: {c_}")
    cells = registered_cells(b, feas, feas3, feasL)
    if b.get("only_cells") is not None:                      # smoke selection / resume segment (cell keys)
        keep = set(b["only_cells"])
        cells = [c for c in cells if cell_key(*c) in keep]
    rng = random.Random(S.order_seed(f"{b['job']}|{b['boot_id'].split('@')[0]}|cells"))
    rng.shuffle(cells)
    # A-36: the sigma' reference runs FIRST (its rep 0 is the boot's first prefill of that L = the warm-up), so no T3
    # cell carries a first-of-L span; the rest stays in seeded random order
    # decision 30 (A-41): long points last, ascending L, never shuffled (a death at a larger L keeps the smaller ones)
    longs = sorted((c for c in cells if c[0] == "long"), key=lambda c: c[1])
    cells = sorted(c for c in cells if c[0] == "t3ref") + [c for c in cells if c[0] not in ("t3ref", "long")] + longs
    dump([list(map(str, c)) for c in cells], bdir / "cell_order.json")
    with open(bdir / "cells.jsonl", "a") as fh:
        for kind, what, k in cells:
            if not srv.alive():
                res["labels"].append("SERVER_DEATH")
                break
            off0 = srv.log_size()
            cp0 = tr.clock_pair() if hasattr(tr, "clock_pair") else None       # audit r1 m10: a (wall, monotonic) pair at EVERY cell start
            rec = _run_cell(tr, b, kind, what, k, f, o_map, led)
            if rec.get("clock_pair") is None:
                rec["clock_pair"] = cp0
            rec["log_offsets"] = [off0, srv.log_size()]
            rec["server_alive_after"] = srv.alive()           # decision 30: the cell a death happened in
            rec["cell_key"] = cell_key(kind, what, k)
            fh.write(json.dumps(rec) + "\n")                  # one line = one finished cell (cell-level DONE, H-4)
            fh.flush()
            if rec["server_alive_after"]:
                tr.sleep(0.5)                               # settle between cells
    if not srv.alive() and "SERVER_DEATH" not in res["labels"]:   # a death in the LAST cell (e.g. the 124k point)
        res["labels"].append("SERVER_DEATH")
    res["trailer_ok"] = C.run_trailer(tr, b["boot_id"]) if srv.alive() else False
    res["stop"] = srv.stop()
    gpu_snapshot(bdir / "gpu_after.txt")
    files = {"decode_probe": any(bdir.glob("decode_probe*.jsonl")), "admission": any(bdir.glob("admission*.jsonl")),
             "tel": any(bdir.glob("tel*.jsonl"))}
    res["problems"] += X.instrumentation_problems(b["kind"], env, None, files)
    dump(res, bdir / "boot_result.json")
    led("boot_end", b["boot_id"])
    (bdir / "DONE").write_text(json.dumps({"files": {p.name: sha256_file(p) for p in sorted(bdir.iterdir())
                                                     if p.is_file() and p.name != "DONE"}}, indent=1))
    return res


def _plan_fingerprint(plan):
    return hashlib.sha256(json.dumps({"spec": S.spec_sha256(), "boots": [
        {k: v for k, v in b.items() if k not in ("only_cells",)} for b in plan["boots"]]}, sort_keys=True,
        default=str).encode()).hexdigest()


def segment_ref_cells(rest):
    """Audit r1 m7: the sigma' reference spans of a T3 boot must come from the SAME server lifetime as the T3 cells they
    normalise (A-29).  A resume segment holding t3 cells therefore also holds the t3ref cell of every L those cells use
    (re-run, even if the earlier segment finished it)."""
    out = set()
    for ck in rest:
        kind, what, _ = ck.split("|")
        if kind == "t3":
            out.add(cell_key("t3ref", what.split("x")[1], 0))
    return out


def _died_with_cells(bdir, b):
    """DONE boot (not a long boot) whose server died after at least one finished cell: eligible for a resume segment."""
    if b.get("long") or not (bdir / "boot_result.json").exists() or not (bdir / "cells.jsonl").exists():
        return False
    return "SERVER_DEATH" in load(bdir / "boot_result.json", {}).get("labels", []) and \
        bool((bdir / "cells.jsonl").read_text().strip())


def resume_segments(plan, out):
    """H-4 (A-35): a boot directory without DONE that holds finished cells (one cells.jsonl line each) is closed as
    PARTIAL (its cells stay, DONE written) and a new boot '<id>@s<k>' with the REMAINING cells follows it; a directory
    with no finished cell is simply redone.  A different server lifetime is a different boot.  -> (plan, log)."""
    new, log = [], []
    for b in plan["boots"]:
        base = set(b["only_cells"]) if b.get("only_cells") is not None else {cell_key(*c) for c in registered_cells(b)}
        done_all, cur, k = set(), b, 0
        while True:
            bdir = Path(out) / "boots" / cur["boot_id"]
            if (bdir / "PARTIAL").exists():
                done_all |= set(json.loads((bdir / "PARTIAL").read_text())["done_cells"])
            elif (bdir / "DONE").exists() and _died_with_cells(bdir, cur):
                # audit r1 m2: a boot that DIED (not a clean DONE) keeps every cell it finished (including the cell the death
                # happened in -- recorded, never overwritten) and its remaining cells run in the next segment
                fin = {json.loads(x).get("cell_key") for x in (bdir / "cells.jsonl").read_text().splitlines()
                       if x.strip()} - {None}
                (bdir / "PARTIAL").write_text(json.dumps({"done_cells": sorted(fin), "t_wall": time.time(), "died": True}))
                log.append({"boot": cur["boot_id"], "kept_cells": len(fin), "died": True})
                done_all |= fin
            elif (bdir / "DONE").exists() or not bdir.exists():
                new.append(cur)
                break
            else:
                fin = {json.loads(x).get("cell_key") for x in (bdir / "cells.jsonl").read_text().splitlines()
                       if x.strip()} - {None} if (bdir / "cells.jsonl").exists() else set()
                if not fin:
                    new.append(cur)
                    break
                (bdir / "PARTIAL").write_text(json.dumps({"done_cells": sorted(fin), "t_wall": time.time()}))
                (bdir / "DONE").write_text("partial: job stopped, remaining cells in the next segment\n")
                log.append({"boot": cur["boot_id"], "kept_cells": len(fin)})
                done_all |= fin
            new.append(cur)
            rest = base - done_all
            if not rest:
                break
            rest |= segment_ref_cells(rest)         # audit r1 m7: a T3 segment re-runs its own sigma' reference cells
            k += 1
            cur = dict(b, boot_id=f"{b['boot_id']}@s{k}", segment=k, resumed_from=cur["boot_id"], only_cells=sorted(rest))
    return dict(plan, boots=new), log


def run_job(job, out, rnd=1, variant_src=None, variant_root=None, make_server=None, run_id=None, plan=None,
            smoke=False, boots=None, cells_per_boot=None, select=None, resume=None):
    """`plan` overrides the registered plan -- CPU tests only (refused outside PDMUX_COSTTERMS_TEST_MODE).
    smoke (H-1): no seal needed, SMOKE_UNSEALED marker forced, (boot, cell) subset via boots / cells_per_boot / select.
    resume (H-4): an existing out directory is resumed only with resume=True (or PDMUX_COSTTERMS_RESUME=1) and the
    same plan fingerprint; the fact is written to RESUME_LOG.json."""
    if plan is not None and not _test_mode():
        raise SystemExit("ABORT_PLAN_OVERRIDE: a plan override is a test hook")
    full = plan is None and not boots and cells_per_boot is None and not select
    plan = plan or make_plan(job, rnd)
    if boots and not smoke:
        raise SystemExit("ABORT_SELECT: boot subsets are a smoke-only option (--smoke)")
    if boots:
        missing = set(boots) - {b["boot_id"] for b in plan["boots"]}
        if missing:
            raise SystemExit(f"ABORT_SELECT: unknown boots {sorted(missing)}")
        plan = dict(plan, boots=[b for b in plan["boots"] if b["boot_id"] in boots])
    if cells_per_boot is not None or select:
        if not smoke:
            raise SystemExit("ABORT_SELECT: cell subsets are a smoke-only option (--smoke)")
        sel_boots = {x.split(":", 1)[0] for x in (select or [])}
        kept = []
        for b in plan["boots"]:
            if select and b["boot_id"] not in sel_boots:
                continue
            kept.append(dict(b, only_cells=select_cells(b, cells_per_boot, select)))
        plan = dict(plan, boots=kept)
    if smoke:
        plan = dict(plan, smoke=True)
    probs = plan_problems(plan) + (registry_problems(plan) if full else [])
    out = Path(out)
    resume = resume if resume is not None else os.environ.get("PDMUX_COSTTERMS_RESUME") == "1"
    prev = load(out / "plan.json")
    rlog = load(out / "RESUME_LOG.json", [])
    if prev is not None:
        if prev.get("fingerprint") != _plan_fingerprint(plan):
            raise SystemExit("ABORT_RESUME_MISMATCH: the out directory holds a different plan / spec")
        if not resume and not _test_mode():
            raise SystemExit("ABORT_OUT_NOT_EMPTY: set PDMUX_COSTTERMS_RESUME=1 to resume this directory")
    out.mkdir(parents=True, exist_ok=True)
    if smoke:
        (out / S.SMOKE_MARKER).write_text("Workspace smoke, NOT a measurement, numbers not citable; harness unsealed\n")
        smoke_engine_digest(out)
    fp = _plan_fingerprint(plan)
    dump({"plan": plan, "problems": probs, "spec_sha256": S.spec_sha256(), "prereg": S.PREREG_REL, "fingerprint": fp,
          "smoke": bool(smoke)}, out / "plan.json")
    if probs:
        raise SystemExit(f"ABORT_PLAN: {probs}")
    run_plan = plan
    if prev is not None:
        run_plan, seg_log = resume_segments(plan, out)
        rlog.append({"t_wall": time.time(), "resumed": True, "done_boots": sorted(
            b["boot_id"] for b in plan["boots"] if (out / "boots" / b["boot_id"] / "DONE").exists()),
            "segments": seg_log, "job_name": os.environ.get("PDMUX_JOB_NAME")})
        dump(rlog, out / "RESUME_LOG.json")
    vdirs = {}
    if any(b["dummy"] for b in run_plan["boots"]):
        if variant_src is None:
            raise SystemExit("ABORT_VARIANT_SRC: dummy variants need --variant-src (Nano snapshot dir)")
        vdirs = prepare_variants(variant_src, variant_root or (out / "variants"), run_plan)
    make_server = make_server or (lambda bdir, argv, env: ProcessServer(bdir, argv, env))
    results = []
    for b in run_plan["boots"]:
        # audit r1 m8: a boot that follows a SERVER_DEATH of an earlier boot of this Job starts on a GPU that may hold the
        # dead server's memory / a wedged context -> AFTER_PRIOR_DEATH on its result (descriptive; the GPU snapshot says more)
        after = any("SERVER_DEATH" in (r or {}).get("labels", []) for r in results)
        results.append(run_boot(dict(b, after_prior_death=True) if after else b, out, vdirs, make_server,
                                run_id or os.environ.get("PDMUX_JOB_NAME", "local")))
    dump(results, out / "boots_result.json")
    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "RUN_DONE"):
            sums.append(f"{sha256_file(p)}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "RUN_DONE").write_text("ok\n")
    return results


# ============================================================================ analysis
def _cells(bdir):
    return [json.loads(x) for x in (bdir / "cells.jsonl").read_text().splitlines() if x.strip()] \
        if (bdir / "cells.jsonl").exists() else []


def _log_slice(text_bytes, offs):
    return text_bytes[offs[0]:offs[1]].decode("utf-8", "replace")


def boot_identity(b, f):
    """Labels FROM THE LOG: model, load format, cap, engine mode, variant (implied attention layers)."""
    arch = f.get("model_arch")
    model = {v: k for k, v in S.ARCH_BY_MODEL.items()}.get(arch)
    mt = X.mem_terms(f)
    n_attn = round(mt["k"] * (f["server_args"].get("tp_size") or 1) / (2 * 8 * 128 * 2)) if (mt and model == "nano") else None
    lf = f["server_args"].get("load_format")
    pd = f["server_args"].get("enable_pdmux")
    sticky = f.get("sticky_index")
    mode = "108v" if not pd else (f"D{b['D']}" if sticky is not None else "108u")
    tag = "llama" if model == "llama" else f"{'dummy' if lf == 'dummy' else 'real'}_a{n_attn}"
    return {"model": model, "load_format": lf, "cap": X.logged_cap(f), "mode": mode, "tag": tag, "n_attn": n_attn}


def t2p_prefill_sm(b, green, sm):
    """Prefill SM of a T2' prefill (decode busy): sticky D -> realized prefill half of the sticky division; 108u ->
    realized prefill half of the FIXED_DSM_108U division; 108v -> 108 (vanilla prefills first, not concurrent)."""
    if b["engine"] == "vanilla":
        return S.TOTAL_SM
    if b["engine"] == "pdmux_sticky":
        return (sm or {}).get("P_real")
    g = next((r for r in (green or {}).get("groups", []) if r.get("target_decode_sm") == S.FIXED_DSM_108U), None)
    return (((g or {}).get("prefill") or {}).get("green_sm") or {}).get("smCount")


def fit_ctx(w):
    """The ctx a T1 fit uses for one window: the ACTUAL window mean (decision 28 A-1), never the nominal c."""
    return w.get("ctx_mean")


def ledger_order_ok(bdir):
    """T4 predictions must be sealed (hash in the ledger) before the first fill starts, and the file must still match."""
    ev = X.read_jsonl(bdir / "events.jsonl")
    i_seal = next((i for i, e in enumerate(ev) if e.get("event") == "t4_prediction_sealed"), None)
    i_fill = next((i for i, e in enumerate(ev) if e.get("event") == "cell_start" and str(e.get("what", "")).startswith("t4_")), None)
    if i_fill is None:
        return True, "no fill"
    if i_seal is None or i_seal > i_fill:
        return False, "prediction not sealed before the first fill"
    if not (bdir / "t4_prediction.json").exists() or sha256_file(bdir / "t4_prediction.json") != ev[i_seal].get("sha256"):
        return False, "prediction file changed after sealing"
    return True, "sealed before fill"


def instr_violation(out):
    return any(q.startswith("INSTR_MATRIX_VIOLATION") for q in out["problems"])


def plan_tag(b):
    """The identity tag a plan implies ('llama', 'real_a4', 'dummy_aN'); the analyzer joins long boots to grid boots with
    the tag FROM THE LOG (boot_identity) and falls back to this one only when the long boot never produced a log."""
    return "llama" if b["model"] == "llama" else f"{'dummy' if b['dummy'] else 'real'}_a{b['variant']}"


def analyze_boot(bdir):
    meta = load(bdir / "boot.json")
    b = meta["plan"]
    f = load(bdir / "facts.json")
    res = load(bdir / "boot_result.json", {})
    if (bdir / "PARTIAL").exists() and not res:
        res = {"start": "ok", "problems": [], "labels": ["PARTIAL_SEGMENT"]}
    out = {"boot_id": b["boot_id"], "kind": b["kind"], "job": b["job"], "round": b["round"], "plan_mode": b["mode"],
           "problems": list(res.get("problems", [])), "labels": list(res.get("labels", [])),
           "long_plan": list(b.get("long", [])), "tp": b.get("tp", 1), "context_plan": b.get("context"),
           "plan_tag": plan_tag(b), "cap_plan": b.get("cap")}
    if f is None or "ABORTED" in out["labels"] or res.get("start") != "ok":
        out["usable"] = False
        return out
    ident = boot_identity(b, f)
    out["identity"] = ident
    sm = load(bdir / "sm.json")
    out["sm"] = sm
    if sm and sm.get("label"):
        out["labels"].append(sm["label"])
    if instr_violation(out):          # audit r1 m6: a boot whose instrumentation differs from the §2-2' matrix is not a measurement
        out["labels"].append("INSTR_MATRIX_VIOLATION")
    dead = "SERVER_DEATH" in out["labels"] and not b.get("long")
    if dead and S.PRE_DEATH_IN_FIT:
        out["labels"].append("PRE_DEATH_IN_FIT")
    out["usable"] = not any(l.startswith("SM_NOT_REALIZED") or (l.startswith("SERVER_DEATH") and not (dead and S.PRE_DEATH_IN_FIT))
                            for l in out["labels"]) and \
        not any(p.startswith("STICKY_MODE_MISMATCH") for p in out["problems"]) and "INSTR_MATRIX_VIOLATION" not in out["labels"]
    out["facts_summary"] = {k: f.get(k) for k in ("capture_bs", "kv_cache_tokens", "weight_gib", "max_mamba_cache_size",
                                                  "init_new_token_ratio_derived", "split_forward_token_budget")}
    out["facts_summary"]["max_prefill_tokens"] = (f.get("final") or {}).get("max_prefill_tokens")
    out["mem"] = X.mem_terms(f)
    out["feasibility"] = load(bdir / "feasibility.json", [])
    out["feasibility_t3"] = load(bdir / "feasibility_t3.json")
    tp = b.get("tp", 1)
    out["tp"] = tp
    if b.get("segment"):
        out["segment"] = b["segment"]
        out["labels"].append("RESUMED_SEGMENT")
    recs = X.read_jsonl(bdir / rank_name("decode_probe.jsonl", tp))
    out["schema_problems"] = X.schema_problems(recs)
    admp = bdir / rank_name("admission.jsonl", tp)
    adm = X.read_jsonl(admp) if admp.exists() else None
    if tp > 1:
        out["tp_facts"] = X.tp_facts(f.get("rank_banners") or {}, tp)
        out["tp_rank_consistency"] = rank_consistency(bdir, tp)
    logb = (bdir / "server.log").read_bytes()
    capture = f.get("capture_bs")
    t1, t2, t2p, t4, t3c, t3r, longp = [], [], [], [], [], [], []
    tel = X.read_jsonl(bdir / rank_name("tel.jsonl", tp))
    spans, its = X.tel_phase(tel)
    dec_seq = X.decode_records(recs)
    p_exp = S.STREAM_GROUPS[S.STICKY_IDX[b["D"]]][0] if b["engine"] == "pdmux_sticky" else None
    have_engine = any(e.get("event") == "decode_forward" for e in recs)
    dec = X.decode_records(recs)
    if any(b_["t_host_start"] < a_["t_host_start"] for a_, b_ in zip(dec, dec[1:])):
        dec, presorted = recs, False          # seq order != time order: let t1_window sort and filter itself
    else:
        presorted = True
    for c in _cells(bdir):
        if c["kind"] == "t1":
            if b["kind"] == "OFF" or not have_engine:
                w = X.client_only_window(c)
                w["engine_records"] = "NOT_COLLECTED"
            else:
                w = X.t1_window(dec, c, capture, _log_slice(logb, c["log_offsets"]), presorted=presorted)
            w.pop("raw_s_eng_ms", None)
            w.update(bs=c["bs"], c=c["c"])
            t1.append(w)
        elif c["kind"] == "t2":
            t2.append(X.t2_value_checked(c, dec if have_engine and b["kind"] != "OFF" else [], 0))
        elif c["kind"] == "t2p":
            v_ = X.t2_value_checked(c, dec if have_engine and b["kind"] != "OFF" else [], 1)
            v_["coexist_engine"] = X.t2p_coexist(c, dec if have_engine and b["kind"] != "OFF" else [])
            t2p.append(v_)
        elif c["kind"] == "t4":
            t4.append(X.t4_fill(c, recs, adm))
        elif c["kind"] == "t3":
            w = X.t3_cell(dec_seq, spans, its, c, p_exp, S.variant_arch(b["variant"]) if b["model"] == "nano" else None, tp)
            w["clock_pair"] = c.get("clock_pair")
            w["t_range"] = (c["t_first_send"], c["t_end"])
            t3c.append(w)
        elif c["kind"] == "t3ref":
            t3r.append(X.t3_ref(dec_seq, spans, c, p_exp))
        elif c["kind"] == "long":
            longp.append(X.long_row(c, _log_slice(logb, c["log_offsets"]),
                                    dec if have_engine and b["kind"] != "OFF" else []))
    if b.get("long"):
        longp = X.long_rows_complete(b["long"], longp, load(bdir / "feasibility_long.json", []), out["labels"])
    out["not_run_after_death"] = not_run_rows(b, out, bdir) if dead else []
    for row in out["not_run_after_death"]:
        if row["kind"] == "t1":
            t1.append(dict(row, bs=row["what"][0], c=row["what"][1]))
        elif row["kind"] == "t2":
            t2.append(dict(row, L=row["what"]))
        elif row["kind"] == "t2p":
            t2p.append(dict(row, L=row["what"]))
    out.update(t1_windows=t1, t2=t2, t2p=t2p, t4_fills=t4, t3_cells=t3c, t3_refs=t3r, long_points=longp)
    if b["kind"] == "T3" and not tel:
        out["labels"].append("SIGMA_NOT_COLLECTED")
    out["t2p_P"] = t2p_prefill_sm(b, load(bdir / rank_name("green.json", tp)), sm)
    if t4:
        ok, why = ledger_order_ok(bdir)
        out["t4_prediction_order"] = why
        if not ok:
            out["problems"].append(f"T4_PREDICTION_NOT_SEALED: {why}")
        preds = load(bdir / "t4_prediction.json", [])
        out["t4"] = [F.t4_labels(fl, next(p for p in preds if p["L"] == fl["L"])) for fl in t4]
    return out


def rank_consistency(bdir, tp):
    """A-33: rank 0 is the measurement; the other ranks must log the same decode_forward sequence (count, bs_real)."""
    seqs = []
    for r in range(tp):
        d = X.decode_records(X.read_jsonl(bdir / rank_name("decode_probe.jsonl", tp, r)))
        seqs.append([(e["seq"], e.get("bs_real")) for e in d])
    same = all(x == seqs[0] for x in seqs[1:])
    return {"n_records": [len(x) for x in seqs], "bs_sequence_equal": same,
            "label": None if same else "TP_RANK_RECORDS_DIFFER"}


def t3_join(v, boots):
    """rho is inside each cell; sigma' pairs each VALID cell with the reference of the SAME boot and L (A-29).
    `boots` (all analysed boots) is unused by design: pairing never crosses a boot (mutant M31 uses it)."""
    refs = {r["L"]: r for r in v.get("t3_refs", [])}
    out = []
    for w in v.get("t3_cells", []):
        sg, why = X.sigma_prime(w, refs.get(w["L"]))
        out.append(dict(w, sigma_prime=sg, sigma_prime_reason=why,
                        sigma_prime_def="median T_span_eng(co, this cell) / median T_span_eng(1-decode reference, "
                                        "same boot, L, P)"))
    return out


def _cell_values(t1, key="S_eng_ms"):
    """(bs, c) -> median over VALID windows of `key` (unit = (boot, cell), A-10).  key may be a callable."""
    acc = {}
    for w in t1:
        v = key(w) if callable(key) else w.get(key)
        if w.get("status") == "VALID" and v is not None:
            acc.setdefault((w["bs"], w["c"]), []).append(v)
    return {k: X.median(v) for k, v in acc.items()}


def _fit_points(t1, cells=None):
    """[(bs, actual ctx, S_eng, nominal c)] per (boot, cell): the fit input of rev3.4 §1-1."""
    vals, ctx = _cell_values(t1), _cell_values(t1, fit_ctx)
    return [(b_, ctx[(b_, c_)], t, c_) for (b_, c_), t in sorted(vals.items())
            if (b_, c_) in ctx and (cells is None or (b_, c_) in cells)]


def _tkey(v, ident):
    k = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}|cap{ident['cap']}"
    if v.get("tp", 1) > 1:
        k += f"|tp{v['tp']}"
    if v.get("segment"):
        k += f"|s{v['segment']}"
    return k


def _segment_key(k):
    last = k.rsplit("|", 1)[-1]
    return last[:1] == "s" and last[1:].isdigit()


def analyze(run_dir, summary_dir=None, clk_path=None):
    run_dir = Path(run_dir)
    boots = {}
    for bdir in sorted((run_dir / "boots").iterdir()):
        if (bdir / "boot.json").exists():
            boots[bdir.name] = analyze_boot(bdir)
    A = {"spec_sha256": S.spec_sha256(), "boots": boots, "tables": {}, "labels": {}, "priors": {}}
    usable = {k: v for k, v in boots.items() if v.get("usable")}
    # ---- §1-7 agreement over every VALID engine window
    wins = []
    for v in usable.values():
        for w in v.get("t1_windows", []):
            if w.get("status") == "VALID" and w.get("engine_records") != "NOT_COLLECTED" and v["kind"] != "OFF":
                wins.append({"model": v["identity"]["tag"], "mode": v["identity"]["mode"], "bs": w["bs"], "c": w["c"],
                             "cap": v["identity"]["cap"], "tp": v.get("tp", 1), "boot_key": _tkey(v, v["identity"]),
                             "S_eng_ms": w.get("S_eng_ms"), "client_gap_ms": w.get("client_gap_ms")})
    agree = F.agreement_labels(wins)
    A["labels"]["agreement"] = {"|".join(map(str, k)): v for k, v in agree.items()}
    A["tables"]["t1_client_secondary"] = {"|".join(map(str, k)): v for k, v in F.client_secondary_table(wins, agree).items()}
    # ---- T1 tables and fits per (job, round, tag, mode)
    t1_tab, fits = {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t1_windows"):
            continue
        ident = v["identity"]
        key = _tkey(v, ident)
        vals = _cell_values(v["t1_windows"])
        t1_tab[key] = {"D_real": v["sm"].get("D_real"), "cells": {f"{b_}x{c_}": t for (b_, c_), t in sorted(vals.items())},
                       "ctx_actual": {f"{b_}x{c_}": x for (b_, c_), x in sorted(_cell_values(v["t1_windows"], fit_ctx).items())},
                       "status_counts": _status_counts(v["t1_windows"]),
                       "gpu_ms": {f"{b_}x{c_}": t for (b_, c_), t in sorted(_cell_values(v["t1_windows"], "gpu_ms").items())},
                       "feasibility": v["feasibility"], "n_registered": len(v["feasibility"]),
                       "n_infeasible": sum(1 for x in v["feasibility"] if x["label"] == "INFEASIBLE_BY_RULE"),
                       "labels": sorted({l for w in v["t1_windows"] for l in w.get("labels", [])})}
        pts = _fit_points(v["t1_windows"])
        if len(pts) >= 4 and (ident["cap"] == S.CAP_MAIN or v["job"] == "J-C"):
            fit = F.fit_t1(pts, ident["mode"])
            fits[key] = {"chosen": _strip(fit["chosen"]), "F1": _strip(fit["F1"]), "label": fit["label"]}
    A["tables"]["t1"] = t1_tab
    A["fits"] = {"t1": fits}
    # ---- gamma_j (J-A, per round)
    gam = {}
    for rkey in sorted({k.split("|")[0] + "|" + k.split("|")[1] for k in fits if k.startswith("J-A")}):
        base_ks = [k for k in fits if k.startswith(rkey) and not _segment_key(k)]    # audit r1 m7: segments never pooled
        anchor = next((fits[k]["F1"]["coef"] for k in base_ks if k.split("|")[3] == "108u"), None)
        by_d = {t1_tab[k]["D_real"]: tuple(fits[k]["F1"]["coef"]) for k in base_ks
                if k.split("|")[3].startswith("D") and t1_tab[k]["D_real"]}
        if anchor and by_d:
            gam[rkey] = F.gamma_fit(by_d, anchor)
    A["fits"]["gamma_j"] = gam
    # ---- cap invariance (J-A1)
    capinv = next((v for v in usable.values() if v["identity"]["cap"] == S.CAP_SMALL and v["kind"] == "T1T2"
                   and v.get("t1_windows") and v["job"].startswith("J-A")), None)
    if capinv:
        main44 = next((v for v in usable.values() if v["job"] == capinv["job"] and v["round"] == capinv["round"]
                       and v["identity"]["mode"] == "D44" and v["identity"]["cap"] == S.CAP_MAIN and v["kind"] == "T1T2"), None)
        if main44:
            A["labels"]["cap_invariance"] = F.cap_invariance(_cell_values(capinv["t1_windows"]),
                                                             {k: x for k, x in _cell_values(main44["t1_windows"]).items()
                                                              if k in S.CAP_INV_CELLS})
    # ---- T2 (P = 108 boots only, decision 28 A-2) / T2' (P axis)
    t2tab, t2fit, qs = {}, {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t2"):
            continue
        ident = v["identity"]
        if t2_axis(ident["mode"]) is None:
            v["problems"].append("T2_NOT_AT_P108: T2 data in a sticky-D boot ignored (decision 28 A-2)")
            continue
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}"
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2"] if x["status"] == "VALID"]
        t2tab[key] = {"points": {str(L): t for L, t in pts}, "P": S.TOTAL_SM, "P_evidence": "vanilla/unpartitioned"}
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        qi = F.q_index(pts, ft)
        slope, slab = F.j1_slope_label(pts)
        t2fit[key] = {"coef": ft["coef"] if ft else None, "loo_internal": ft["loo_internal"] if ft else None,
                      "accepted": bool(ft and ft["accepted"]),
                      "label": None if (ft and ft["accepted"]) else "FORM_INADEQUATE_T2_P108",
                      "q": qi, "j1_slope": slope, "j1_label": slab}
        qs.setdefault(f"{ident['tag']}|{ident['mode']}", []).append(qi.get("q"))
    A["tables"]["t2"] = t2tab
    A["fits"]["t2"] = t2fit
    A["labels"]["q"] = {k: F.q_label(v) for k, v in qs.items()}
    t2p_tab, t2p_fit = {}, {}
    for v in usable.values():
        if v["kind"] == "OFF" or not v.get("t2p"):
            continue
        ident = v["identity"]
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}"
        conf = {str(x["L"]): (x.get("coexist_engine") or {}).get("label") for x in v["t2p"]}
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2p"] if x["status"] == "VALID"
               and (x.get("coexist_engine") or {}).get("confirmed")]
        t2p_tab[key] = {"points": {str(x["L"]): x.get("ttft_ms") for x in v["t2p"] if x["status"] == "VALID"},
                        "P": v.get("t2p_P"), "coexist": conf, "points_confirmed": {str(L): t for L, t in pts},
                        "estimand": "single prefill TTFT with one bs=1 decode request co-resident (decision 28 A-2)"}
        ft = F.fit_t2(pts) if len(pts) >= 3 else None
        t2p_fit[key] = {"coef": ft["coef"] if ft else None, "accepted": bool(ft and ft["accepted"]),
                        "label": None if (ft and ft["accepted"]) else f"FORM_INADEQUATE_T2P_P{v.get('t2p_P')}",
                        "q": F.q_index(pts, ft)}
    A["tables"]["t2prime"] = t2p_tab
    A["fits"]["t2prime"] = t2p_fit
    A["fits"]["theta_prime"] = theta_prime(t2p_tab)
    # ---- decision 30 long points (every boot, usable or not: a death keeps its earlier rows)
    A["tables"]["t2_long"] = long_table(boots, usable)
    A["tables"]["pre_death"] = pre_death_table(boots)
    # ---- T4
    t4 = {}
    pools = {}
    for v in usable.values():
        for x in v.get("t4", []):
            t4[f"{v['boot_id']}|L{x['L']}"] = x
        if v["kind"] == "T4" and v.get("mem"):
            pools[v["boot_id"]] = v["mem"]["pool_bytes"]
    A["tables"]["t4"] = t4
    if pools:
        A["labels"]["c_mem_drift"] = F.c_mem_drift(pools)
    # ---- J-H1
    jh = jh1_grid_boots(usable)
    if jh:
        A["jh1"] = analyze_jh1(jh)
    if any(v["job"] == "J-H1" and v.get("long_plan") for v in boots.values()):
        A.setdefault("jh1", {})["long_contrast"] = long_contrast(A["tables"]["t2_long"])
    # ---- OFF diagnostic (descriptive)
    A["instr_effect"] = instr_effects(usable)
    # ---- stage 2: T3, J-H2, J-C
    # audit r1 m10: a resumed run keeps the clk.csv of EVERY submission (costterms_job.sh copies each one to RAW/clk/<job name>.csv)
    clk_files = sorted((run_dir / "clk").glob("*.csv")) if (run_dir / "clk").exists() else []
    clk = [r for f_ in clk_files for r in X.read_clk(f_)] if clk_files else (X.read_clk(clk_path) if clk_path else [])
    clk.sort()
    A["tables"]["t3"], raw3 = t3_tables(usable, clk)
    jh2 = {k: v for k, v in usable.items() if v["job"] == "J-H2"}
    if jh2:
        A["jh2"] = analyze_jh2(jh2, A["tables"]["t1"], A["tables"]["t3"])
    jc = {k: v for k, v in usable.items() if v["job"] == "J-C"}
    if jc or any(v["job"] == "J-C" for v in boots.values()):
        A["jc"] = analyze_jc(jc, boots, A)
    A["resume_log"] = load(run_dir / "RESUME_LOG.json")
    A["smoke"] = (run_dir / S.SMOKE_MARKER).exists()
    if A["smoke"]:
        A["labels"]["smoke"] = S.SMOKE_MARKER
    # ---- priors (comparison columns)
    A["priors"] = {f"a{n}": F.priors(n) for n in S.VARIANTS}
    dump(A, run_dir / "analysis.json")
    if raw3:
        dump(raw3, run_dir / "t3_iterations.json")
    if summary_dir is not None:
        export_summary(run_dir, summary_dir)
    return A


def not_run_rows(b, out, bdir):
    """Audit r1 m2: every registered cell of a dead boot that has no finished line gets a NOT_RUN_AFTER_SERVER_DEATH row."""
    fin_keys = {c.get("cell_key") for c in _cells(bdir)}
    keep = set(b["only_cells"]) if b.get("only_cells") is not None else None
    rows = []
    for kind, what, k in registered_cells(b, out["feasibility"], out.get("feasibility_t3")):
        ck = cell_key(kind, what, k)
        if ck in fin_keys or (keep is not None and ck not in keep):
            continue
        rows.append({"cell_key": ck, "kind": kind, "what": list(what) if isinstance(what, tuple) else what,
                     "status": "NOT_RUN", "labels": ["NOT_RUN_AFTER_SERVER_DEATH"]})
    return rows


def pre_death_table(boots):
    """Audit r1 m2 (A-50): per non-long boot with a SERVER_DEATH, the cells that finished VALID before it (PRE_DEATH, not in any
    fit unless S.PRE_DEATH_IN_FIT) and the NOT_RUN_AFTER_SERVER_DEATH rows.  A resumed run's later segments are separate boots."""
    out = {}
    for k, v in boots.items():
        if "SERVER_DEATH" not in v.get("labels", []) or v.get("long_plan") or "t1_windows" not in v:
            continue
        out[k] = {"label": "PRE_DEATH", "in_fit": bool(v.get("usable")), "job": v["job"], "round": v["round"],
                  "t1": {f"{b_}x{c_}": t for (b_, c_), t in sorted(_cell_values(v["t1_windows"]).items())},
                  "t2": {str(x["L"]): x["ttft_ms"] for x in v["t2"] if x.get("status") == "VALID"},
                  "t2p": {str(x["L"]): x["ttft_ms"] for x in v["t2p"] if x.get("status") == "VALID"},
                  "not_run": [r["cell_key"] for r in v.get("not_run_after_death", [])]}
    return out


def long_join_ok(g, v):
    """May the grid boot g be joined to the long boot v?  Same Job, same engine mode, same tp, SAME MODEL TAG (decision 33:
    J-H1 has two long boots -- Llama and real a4 -- and each joins only its own grid boot), g not itself a long / OFF boot
    and with a T2 grid."""
    gi = g.get("identity") or {}
    return not (g["job"] != v["job"] or g.get("long_plan") or g["kind"] == "OFF" or not g.get("t2")
                or gi.get("mode") != v["plan_mode"] or g.get("tp", 1) != v.get("tp", 1)
                or gi.get("tag") != ((v.get("identity") or {}).get("tag") or v.get("plan_tag")))


def long_table(boots, usable):
    """Decision 30: per long boot the rows (T2 value + serving check + death labels) and, descriptively, the join with
    the same-Job, same-mode, same-tp, same-model T2 grid boot(s) (context 73,728, or 131,072 in J-A -- recorded per join): bridge ratio at 65,536 and the grid-only quadratic
    extrapolated to 96k / 124k vs the long values, plus the joint fit (T2_LONG_ACROSS_BOOTS: cap / context differ)."""
    out = {}
    for k, v in boots.items():
        if not v.get("long_plan"):
            continue
        rows = v.get("long_points")
        if rows is None:                                    # the long boot itself failed to start / aborted
            ctx_abort = any(q.startswith("ABORT_CONTEXT") for q in v.get("problems", []))
            rows = [{"L": L, "status": "NOT_RUN", "labels": ["ABORT_CONTEXT_TOO_SHORT" if ctx_abort else "NOT_RUN_BOOT_FAILED"]
                     + (["SERVER_DEATH"] if "SERVER_DEATH" in v["labels"] else [])} for L in v["long_plan"]]
        ent = {"job": v["job"], "round": v["round"], "tp": v.get("tp", 1), "mode": v["plan_mode"],
               "tag": (v.get("identity") or {}).get("tag") or v.get("plan_tag"),
               "context_plan": v.get("context_plan"), "cap": (v.get("identity") or {}).get("cap"),
               "boot_labels": v["labels"], "rows": rows, "joins": {},
               "scope": "P = 108 single-request prefill TTFT (T2 rule); separate last boot, context 131,072 (decision 30)"}
        vals = {r["L"]: r["ttft_ms"] for r in rows if r.get("status") == "VALID"}
        for g in usable.values():
            gi = g.get("identity") or {}
            if not long_join_ok(g, v):
                continue
            pts = [(x["L"], x["ttft_ms"]) for x in g["t2"] if x["status"] == "VALID"]
            if len(pts) < 3:
                continue
            grid = dict(pts)
            ft = F.fit_t2(pts)
            j = {"bridge_L": S.LONG_BRIDGE_L, "grid_context": g.get("context_plan"), "long_context": v.get("context_plan"),
                 "grid_cap": gi.get("cap"), "long_cap": (v.get("identity") or {}).get("cap"),
                 "bridge_ratio_long_over_grid": (vals[S.LONG_BRIDGE_L] / grid[S.LONG_BRIDGE_L])
                 if S.LONG_BRIDGE_L in vals and S.LONG_BRIDGE_L in grid else None,
                 "grid_fit_extrapolation_rel_err": {str(L): (ft["pred"](L) - vals[L]) / vals[L]
                                                    for L in S.T2EXT_L if L in vals and ft} if ft else None,
                 "label": "T2_LONG_ACROSS_BOOTS"}
            both = pts + [(L, vals[L]) for L in S.T2EXT_L if L in vals]
            fj = F.fit_t2(both) if len(both) > len(pts) else None
            j["joint_fit_coef"] = fj["coef"] if fj else None
            j["joint_fit_loo_internal"] = {str(a): b_ for a, b_ in fj["loo_internal"].items()} if fj else None
            ent["joins"][g["boot_id"]] = j
        out[k] = ent
    # decision 33 (A-46): long boots of the same Job and tp that differ in ENGINE MODE only (J-C TP2 108u vs 108v): the
    # per-L ratio of the two long boots' T2 values, descriptive (same Job, same node, same context / cap / L)
    for k, e in out.items():
        e["siblings"] = {}
        for k2, e2 in out.items():
            if k2 == k or e2["job"] != e["job"] or e2["tp"] != e["tp"] or e2["tag"] != e["tag"] or e2["mode"] == e["mode"]:
                continue
            a = {r["L"]: r["ttft_ms"] for r in e["rows"] if r.get("status") == "VALID"}
            b_ = {r["L"]: r["ttft_ms"] for r in e2["rows"] if r.get("status") == "VALID"}
            e["siblings"][k2] = {"mode": e2["mode"], "ratio_this_over_sibling": {str(L): a[L] / b_[L] for L in sorted(a) if L in b_},
                                 "label": "SAME_JOB_LONG_SIBLING (descriptive; the engine mode is the only difference)"}
    return out


def long_contrast(t2_long):
    """Decision 33 (L-8): J-H1 Llama vs real-a4 long boots -- PREREG §4 (4) contrast at L = 65,536 / 98,304 / 126,976, both
    from the SAME Job (rows VALID in both).  Descriptive per-L ratio Llama / Nano; no interpretation here."""
    ll = next((e for e in t2_long.values() if e["job"] == "J-H1" and e.get("tag") == "llama"), None)
    nn = next((e for e in t2_long.values() if e["job"] == "J-H1" and e.get("tag") == "real_a4"), None)
    if not ll or not nn:
        return {"label": "LONG_CONTRAST_NOT_COLLECTED", "ratio_llama_over_nano": None}
    a = {r["L"]: r["ttft_ms"] for r in ll["rows"] if r.get("status") == "VALID"}
    b = {r["L"]: r["ttft_ms"] for r in nn["rows"] if r.get("status") == "VALID"}
    return {"label": "SAME_JOB_LONG_CONTRAST (descriptive; models differ, cap / context / mode equal)",
            "ratio_llama_over_nano": {str(L): a[L] / b[L] for L in sorted(a) if L in b}}


def export_summary(run_dir, summary_dir):
    """H-3: what goes under results/ -- analysis, plan, markers, sums, the T3 raw-value file and one DONE manifest per
    boot; server logs / telemetry / decode records stay in the raw directory (cluster volume)."""
    run_dir, sd = Path(run_dir), Path(summary_dir)
    sd.mkdir(parents=True, exist_ok=True)
    for name in S.SUMMARY_FILES:
        if (run_dir / name).exists():
            (sd / name).write_bytes((run_dir / name).read_bytes())
    man = sd / "boot_manifests"
    man.mkdir(exist_ok=True)
    for d in sorted((run_dir / "boots").iterdir()) if (run_dir / "boots").exists() else []:
        if (d / "DONE").exists():
            (man / f"{d.name}.DONE").write_bytes((d / "DONE").read_bytes())
    (sd / "RAW_LOCATION.txt").write_text(f"{run_dir.resolve()}\n")


def t3_tables(usable, clk):
    """Per (job, round, tag, D[, segment]) T3 boot: per cell the reps (rho, sigma', n_it_obs, stalls, distributions,
    clock covariates) and the cell means of ln rho / ln sigma' over VALID reps (§1-3 맞춤).  Raw interval values go to
    t3_iterations.json, not analysis.json."""
    tab, raw = {}, {}
    for v in usable.values():
        if not v.get("t3_cells"):
            continue
        ident = v["identity"]
        key = f"{v['job']}|r{v['round']}|{ident['tag']}|{ident['mode']}" + (f"|s{v['segment']}" if v.get("segment") else "")
        joined = t3_join(v, usable)
        cells = {}
        for w in joined:
            ck = f"{w['bs']}x{w['L']}"
            raw.setdefault(key, {}).setdefault(ck, {})[str(w.get("rep"))] = {
                "co_ms": w.pop("raw_co_ms", None), "solo_ms": w.pop("raw_solo_ms", None)}
            if w.get("status") == "VALID":
                w["clock_co"] = X.clock_covariates(clk, w.get("clock_pair"), w.get("co_ranges") or [])
                w["clock_solo"] = X.clock_covariates(clk, w.get("clock_pair"), w.get("solo_ranges") or [])
            for k_ in ("co_ranges", "solo_ranges", "span_ms"):
                w.pop(k_, None)
            cells.setdefault(ck, []).append(w)
        summ = {}
        for ck, ws in cells.items():
            ok = [w for w in ws if w.get("status") == "VALID"]
            lr = [math.log(w["rho"]) for w in ok if w.get("rho")]
            ls = [math.log(w["sigma_prime"]) for w in ok if w.get("sigma_prime")]
            summ[ck] = {"reps": ws, "n_valid": len(ok), "ln_rho_mean": sum(lr) / len(lr) if lr else None,
                        "ln_sigma_prime_mean": sum(ls) / len(ls) if ls else None,
                        "sigma_label": None if ls else "SIGMA_NOT_COLLECTED",
                        "q_stall_mean": (sum(w["stall_co"]["q_stall"] for w in ok) / len(ok)) if ok else None,
                        "max_stall_run": max((w["stall_co"]["max_run"] for w in ok), default=None),
                        "n_it_obs": [w.get("n_it_obs") for w in ws],
                        "labels": sorted({l for w in ws for l in w.get("labels", [])})}
        for w in v["t3_cells"]:                       # raw interval arrays live only in t3_iterations.json
            w.pop("raw_co_ms", None)
            w.pop("raw_solo_ms", None)
        tab[key] = {"D_real": (v.get("sm") or {}).get("D_real"), "P_real": (v.get("sm") or {}).get("P_real"),
                    "cells": summ, "refs": v.get("t3_refs"), "feasibility": v.get("feasibility_t3"),
                    "scope": "TEL ON boot; rho / sigma' = total effect incl. the power-limited clock (§1-3)",
                    "a31_note": "attention-only decode: sigma' reads differently (§4')" if ident.get("n_attn") == 31 else None}
    return tab, raw


def analyze_jh2(jh2, t1tab, t3tab):
    """J-H2 (descriptive): per variant the D24 / D44 decode-step ratio of the T1 cells (SM sensitivity of the
    composition, no prediction registered) and the T3 rows of that variant."""
    out = {}
    for n in S.VARIANTS:
        tag = f"dummy_a{n}"
        rows = {d: next((t for k, t in t1tab.items() if k.startswith("J-H2|") and f"|{tag}|D{d}|" in k), None)
                for d in S.JH2_D}
        ratio = {}
        if all(rows.values()):
            for ck in rows[24]["cells"]:
                if ck in rows[44]["cells"]:
                    ratio[ck] = {"S_eng_D24_over_D44": rows[24]["cells"][ck] / rows[44]["cells"][ck],
                                 "ctx_actual": {d: rows[d]["ctx_actual"].get(ck) for d in S.JH2_D}}
        out[tag] = {"t1_sm_ratio": ratio, "t3_keys": sorted(k for k in t3tab if f"|{tag}|" in k),
                    "note": "descriptive; variant weights / power state differ -- not an interference size (§4')"}
    return out


def is_anchor32(v):
    """The decision-33 cap-32 N=1 anchor (identity cap from the LOG): not the anchor of the cap-48 TP2 / N=1 pairs."""
    return v.get("tp", 1) == 1 and v["identity"]["cap"] == S.JC_ANCHOR32_CAP


def v_O(v, bs, c):
    """The registered output length O of a T1 cell of an analysed boot (from its feasibility rows)."""
    r = next((x for x in v.get("feasibility", []) if (x["bs"], x["c"]) == (bs, c)), None)
    return r["O"] if r else None


def analyze_jc(jc, boots, A):
    """J-C (§5, rev3.5 draft): TP2 / N=1 ratios of the same cell in the same round, F1-F5, T4 (F3), 96k point."""
    out = {"ratios_t1": {}, "ratios_t2": {}, "ratios_t2p": {}, "F": {}, "t4": {}, "long": None, "rank": {}}
    by, a32 = {}, {}
    for v in jc.values():
        if v["kind"] == "T1T2" and not v.get("long_plan"):
            if is_anchor32(v):
                a32[v["round"]] = v            # decision 33 L-14: its own table, never the cap-48 anchor of the pair
                continue
            by[(v["round"], v["identity"]["mode"], v.get("tp", 1), v.get("segment", 0))] = v
    for (rnd, mode, tp, seg), v in sorted(by.items()):
        if tp == 1:
            continue
        a = by.get((rnd, mode, 1, seg))             # audit r1 m7: a segment pairs only with the same segment
        if a is None:
            continue
        t_tp, t_n1 = _cell_values(v["t1_windows"]), _cell_values(a["t1_windows"])
        c_tp, c_n1 = _cell_values(v["t1_windows"], fit_ctx), _cell_values(a["t1_windows"], fit_ctx)
        out["ratios_t1"][f"r{rnd}|{mode}" + (f"|s{seg}" if seg else "")] = {f"{b_}x{c_}": {"ratio": t_tp[(b_, c_)] / t_n1[(b_, c_)],
                                                              "ctx_tp2": c_tp.get((b_, c_)), "ctx_n1": c_n1.get((b_, c_))}
                                              for (b_, c_) in sorted(t_tp) if (b_, c_) in t_n1}
        x2 = {x["L"]: x["ttft_ms"] for x in v["t2"] if x["status"] == "VALID"}
        y2 = {x["L"]: x["ttft_ms"] for x in a["t2"] if x["status"] == "VALID"}
        if x2:
            out["ratios_t2"][f"r{rnd}|{mode}" + (f"|s{seg}" if seg else "")] = {str(L): x2[L] / y2[L] for L in sorted(x2) if L in y2}
        xp = {x["L"]: x["ttft_ms"] for x in v["t2p"] if x["status"] == "VALID" and (x.get("coexist_engine") or {}).get("confirmed")}
        yp = {x["L"]: x["ttft_ms"] for x in a["t2p"] if x["status"] == "VALID" and (x.get("coexist_engine") or {}).get("confirmed")}
        if xp:
            out["ratios_t2p"][f"r{rnd}|{mode}" + (f"|s{seg}" if seg else "")] = {str(L): xp[L] / yp[L] for L in sorted(xp) if L in yp}
    out["ratios_t1_anchor32"] = {}
    for rnd, a in sorted(a32.items()):         # (32, 32k) D44: TP2 value / cap-32 anchor value, ACTUAL ctx of both reported
        v = by.get((rnd, "D44", 2, 0))
        if v is None:
            continue
        t_tp, t_n1 = _cell_values(v["t1_windows"]), _cell_values(a["t1_windows"])
        c_tp, c_n1 = _cell_values(v["t1_windows"], fit_ctx), _cell_values(a["t1_windows"], fit_ctx)
        out["ratios_t1_anchor32"][f"r{rnd}|D44"] = {
            f"{b_}x{c_}": {"ratio": t_tp[(b_, c_)] / t_n1[(b_, c_)], "ctx_tp2": c_tp.get((b_, c_)), "ctx_n1": c_n1.get((b_, c_)),
                           "O_tp2": v_O(v, b_, c_), "O_n1": v_O(a, b_, c_),
                           "label": "ANCHOR32_OWN_O (anchor cap 32 and O differ from TP2; compare at the actual ctx)"}
            for (b_, c_) in sorted(t_tp) if (b_, c_) in t_n1}
    tps = [v for v in jc.values() if v.get("tp", 1) > 1]
    for v in tps:
        out["F"][v["boot_id"]] = v.get("tp_facts")
        out["rank"][v["boot_id"]] = v.get("tp_rank_consistency")
        if v.get("t4"):
            out["t4"][v["boot_id"]] = v["t4"]
    out["long"] = {k: x for k, x in A["tables"].get("t2_long", {}).items() if x["job"] == "J-C"} or None
    f_ok = [x for x in out["F"].values() if x]
    out["F5_label"] = None if f_ok and all(x["F5"]["held"] for x in f_ok) else "TP2_NOT_BOOTABLE"
    dead = [k for k, v in boots.items() if v["job"] == "J-C" and v.get("tp", 1) > 1 and "TP2_NOT_BOOTABLE" in v["labels"]
            and not v.get("long_plan")]            # a failed long boot is a long-point result, not F5 (decision 30)
    if dead:
        out["F5_label"] = "TP2_NOT_BOOTABLE"
        out["tp2_dead_boots"] = dead
    out["note"] = ("total-time ratios only; all-reduce not decomposed (§5-2); anchor cap 48 (and cap 32 for (32, 32k) D44, "
                   "own O) vs TP2 cap 128 (disclosed)")
    return out


def _status_counts(ws):
    out = {}
    for w in ws:
        out[w.get("status")] = out.get(w.get("status"), 0) + 1
    return out


def _strip(fit):
    if fit is None:
        return None
    return {k: v for k, v in fit.items() if k != "pred"}


def theta_prime(t2p_tab):
    """Descriptive (§1-2 rev3.4): per (job, round) over the sticky-D boots, through-origin LS of ln(T2'(L, P) /
    T2'(L, P_max)) on ln(P_max / P), pooled over the confirmed L.  No label reads it."""
    out = {}
    groups = {}
    for k, v in t2p_tab.items():
        job, rnd, tag, mode = k.split("|")
        if mode.startswith("D") and v.get("P"):
            groups.setdefault(f"{job}|{rnd}|{tag}", []).append(v)
    for g, vs in groups.items():
        top = max(vs, key=lambda v: v["P"])
        xs, ys = [], []
        for v in vs:
            if v is top:
                continue
            for L, t in v["points_confirmed"].items():
                t0 = top["points_confirmed"].get(L)
                if t0 and t:
                    xs.append(math.log(top["P"] / v["P"]))
                    ys.append(math.log(t / t0))
        den = sum(x * x for x in xs)
        out[g] = {"theta_prime": (sum(x * y for x, y in zip(xs, ys)) / den) if den > 0 else None, "P_ref": top["P"],
                  "n_points": len(xs)}
    return out


def jh1_grid_boots(usable):
    """The J-H1 boots the composition tests (w_indep, P1-P4) use: the grid boots only.  The decision-33 long boots (same
    identity tags 'real_a4' / 'llama' as two grid boots, context 131,072, cap 48) are a separate table (long_contrast)."""
    return {k: v for k, v in usable.items() if v["job"] == "J-H1" and not v.get("long_plan")}


def analyze_jh1(jh):
    by_tag = {v["identity"]["tag"]: v for v in jh.values() if v["kind"] != "OFF"}
    out = {"boots": sorted(by_tag)}
    d4, r4 = by_tag.get("dummy_a4"), by_tag.get("real_a4")
    wi = {}
    if d4 and r4:
        common = set(S.JH_COMMON_CELLS)
        wi["T1"] = F.w_indep({k: x for k, x in _cell_values(d4["t1_windows"]).items() if k in common},
                             {k: x for k, x in _cell_values(r4["t1_windows"]).items() if k in common})
        wi["T2"] = F.w_indep({x["L"]: x["ttft_ms"] for x in d4["t2"] if x["status"] == "VALID"},
                             {x["L"]: x["ttft_ms"] for x in r4["t2"] if x["status"] == "VALID"})
    out["w_indep"] = wi
    p1, p2, cs, qn = {}, {}, {}, {}
    for n in S.VARIANTS:
        v = by_tag.get(f"dummy_a{n}")
        if not v:
            continue
        mt = v["mem"]
        p1[f"a{n}"] = F.compo_mem(n, mt["s"] if mt else None, mt["k"] if mt else None, mt["slots"] if mt else None)
        p2[f"a{n}"] = F.cmem_variant(n, v.get("t4", []))
        pts = _fit_points(v["t1_windows"], set(S.JH_COMMON_CELLS))
        if len(pts) == len(S.JH_COMMON_CELLS):
            ft = F.fit_form(pts, "F1")
            cs[n] = ft["coef"][2] if ft else None
        pts = [(x["L"], x["ttft_ms"]) for x in v["t2"] if x["status"] == "VALID"]
        ft2 = F.fit_t2(pts) if len(pts) >= 3 else None
        qn[n] = F.q_index(pts, ft2).get("q")
    out["P1_compo_mem"] = p1
    out["P2_cmem"] = p2
    out["P3_ctx_slope"] = F.ctx_slope_prop({n: c for n, c in cs.items() if c}, (wi.get("T1") or {}).get("label"))
    out["P4_q"] = F.q_monotone(qn, (wi.get("T2") or {}).get("label"))
    out["c_by_variant"] = cs
    out["q_by_variant"] = qn
    return out


def instr_effects(usable):
    out = {}
    for v in usable.values():
        if v["kind"] != "OFF":
            continue
        ident = v["identity"]
        on = next((w for w in usable.values() if w["kind"] == "T1T2" and w["job"] == v["job"] and w["round"] == v["round"]
                   and w["identity"]["tag"] == ident["tag"] and w["identity"]["mode"] == ident["mode"]
                   and w["identity"]["cap"] == ident["cap"] and w.get("tp", 1) == v.get("tp", 1)
                   and w.get("segment") == v.get("segment")), None)
        if on is None:
            continue
        off_t1 = _cell_values(v["t1_windows"], "client_gap_ms")
        on_t1 = {k: x for k, x in _cell_values(on["t1_windows"], "client_gap_ms").items() if k in off_t1}
        off_t2 = {x["L"]: x["ttft_ms"] for x in v["t2"] + v.get("t2p", []) if x["status"] == "VALID"}
        on_t2 = {x["L"]: x["ttft_ms"] for x in (on["t2"] if v["t2"] else on.get("t2p", []))
                 if x["status"] == "VALID" and x["L"] in off_t2}
        reg1 = [(x["bs"], x["c"]) for x in v.get("feasibility", [])]
        reg2 = sorted({x["L"] for x in v["t2"] + v.get("t2p", [])})
        out[v["boot_id"]] = {"T1": F.instrument_effect(on_t1, off_t1, reg1), "T2": F.instrument_effect(on_t2, off_t2, reg2),
                             "note": "descriptive; never applied to T1-T4 (decision 22)"}
    return out


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--job", required=True)
    p.add_argument("--round", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--job", required=True)
    r.add_argument("--round", type=int, default=1)
    r.add_argument("--out", required=True, help="RAW output directory (cluster volume in a Job, H-3)")
    r.add_argument("--variant-src")
    r.add_argument("--variant-root")
    r.add_argument("--smoke", action="store_true", help="H-1: unsealed Workspace smoke, SMOKE_UNSEALED marker forced")
    r.add_argument("--boots", help="comma-separated boot ids of the registered plan (subset)")
    r.add_argument("--cells-per-boot", type=int, help="smoke: first N cells of each kind per boot")
    r.add_argument("--select", help="smoke: comma-separated BOOT:KIND:SPEC, e.g. ja1_D44:t1:8x2048,ja1_D44:t2p:4096")
    r.add_argument("--resume", action="store_true", help="H-4: resume an existing out directory (same plan)")
    a = sub.add_parser("analyze")
    a.add_argument("dir")
    a.add_argument("--summary", help="H-3: copy the summary files (analysis, plan, markers, sums, manifests) here")
    a.add_argument("--clk", help="nvidia-smi clk.csv for the T3 clock covariates (A-34)")
    sub.add_parser("spec-sha")
    ns = ap.parse_args(argv)
    if ns.cmd == "spec-sha":
        print(S.spec_sha256())
        return 0
    if ns.cmd == "plan":
        plan = make_plan(ns.job, ns.round)
        for b in plan["boots"]:
            print(json.dumps({"boot_id": b["boot_id"], "kind": b["kind"], "mode": b["mode"], "cap": b["cap"],
                              "tp": b["tp"], "n_t1": len(b["t1"]), "t2": b["t2"], "t2p": b["t2p"], "t3": b["t3"],
                              "t4": b["t4"], "long": b["long"]}))
        print(json.dumps({"problems": plan_problems(plan), "deferred": plan["deferred"]}))
        return 0
    if ns.cmd == "run":
        if os.environ.get("PDMUX_COSTTERMS_SEALED") != "1" and not _test_mode() and not ns.smoke:
            print("ABORT_UNSEALED: the cost-term harness is not sealed (PREREG §12 item 13); --smoke for a Workspace "
                  "smoke (not a measurement)", file=sys.stderr)
            return 16
        run_job(ns.job, ns.out, ns.round, ns.variant_src, ns.variant_root, smoke=ns.smoke,
                boots=ns.boots.split(",") if ns.boots else None, cells_per_boot=ns.cells_per_boot,
                select=ns.select.split(",") if ns.select else None, resume=True if ns.resume else None)
        return 0
    if ns.cmd == "analyze":
        analyze(ns.dir, ns.summary, ns.clk)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
