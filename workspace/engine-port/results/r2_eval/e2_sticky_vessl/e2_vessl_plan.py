#!/usr/bin/env python3
"""E2 (sticky partition contrast) VESSL substrate port -- registered constants and checks.

WHAT THIS FILE IS.  The CPU-side companion of `e2_sticky_vessl.sh`, registered by the
substrate-port addendum `PREREG_E2_VESSL_A13_2026-10-04.md` (A13).  It re-implements NO
E2 rule and NO estimator: the five estimators are `../e2_sticky_prereg/e2_realized_mix.py`
and the rules R1-R6 are `../e2_sticky_prereg/e2_label.py`, both CALLED by the runner,
digest-enforced here, never copied.  What this file adds is only what the substrate port
needs and rev3 + ADDENDUM did not have:

  1. REGISTERED DIGESTS.  The E2 decision path (rev3 + ADDENDUM A1-A12, both rule files,
     every prior verdict) must be the registered bytes (commit b917bfc for the rule files and
     rev1-3/ADDENDUM; per-file provenance at REGISTERED_DIGESTS).  A13 itself is digest-pinned here.
     `--verify-digests` exits 3 on any drift.
  2. SELFTEST ARCHIVE.  The rev3 sec 5-3 estimator selftest reads seven git-ignored
     telemetry files of job 908623.  They reach the Job through the object volume and
     must be byte-identical to the local archive (`--verify-archive`).
  3. ENGINE TREE.  The installed tree must differ from job 908623's in EXACTLY one file,
     `multiplex/multiplexing_mixin.py`, and that file must be HEAD's registered bytes
     (`PDMUX_PHASE_EVENTS`, default OFF, same-line hooks; A13 sec 1-3).  `--engine-diff`.
  4. SUBSTRATE (B0).  A100 80GB, cc 8.0, MIG off, driver >= 550, 108 SM (`--check-substrate`).
  5. P7 T_cap from the VESSL lambda0 cells (A13 sec 2-2), P14 on a registered in-script
     hard cap (A13 sec 5), and the VESSL per-boot budget (A13 sec 5-1).
  6. DESCRIPTIVE: one green-context read-out summary per Job (E2C-38), never a decision input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "e2_sticky_prereg"
LAM0_KISTI = HERE.parent / "lambda0_prereg" / "lam0_908623"
LAM0V_DESC = HERE.parent / "lambda0_vessl" / "analysis_2026-10-04" / "lam0v_descriptive.json"

# --------------------------------------------------------------------- (1) registration
# sha256 of the E2 decision path and its audit trail.  Provenance per file (`git log`, checked
# 2026-10-04; corrected after VERDICT_e2_vessl_rules_2026-10-04 (ii) -- the earlier comment said
# "b917bfc is the only commit touching them", which was false for two files):
#   * the two rule files, the KISTI runner, rev1-3 + ADDENDUM and the five rule/harness verdicts (rules x3, harness x2):
#     added in b917bfc (the commit the 2026-09-15 OVERRIDE submitted), never changed since;
#   * OVERRIDE_E2_SUBMIT_2026-09-15.md: added in b917bfc, last changed in dff7bb7 (sec 7-1 submission
#     record, 2026-09-15) -- the registered bytes are dff7bb7's;
#   * VERDICT_e2_substrate_portability_2026-09-18.md: NOT in b917bfc -- added in 40a7be2, last
#     changed in b330ae8 (2026-09-24 scope correction) -- the registered bytes are b330ae8's.
# Relative to ../e2_sticky_prereg/.
REGISTERED_DIGESTS = {
    "e2_realized_mix.py": "ecb0f6fce76f82c3a3b816ebc542885c4b9be4ac247529b51efc6f5032ab0b78",
    "e2_label.py": "feabd5a7ce393c6de24fbfa5bbf030796f714b5e1597c2aa6c1074b22c25868c",
    "e2_sticky.sbatch": "e30df9f4f61cdfefa50f0bbca489d69ba307929f48d3aedac9628c10b3c99f95",
    "PREREG_E2_STICKY_2026-09-15.md": "b09c93a716d53e3b6f2efb93fb4f2d8dc7653c4abba8610c7650d0b26fe88454",
    "PREREG_E2_STICKY_REV2_2026-09-15.md": "d87461a9d59ec48e724be4261b1b4717d86947b158c8eaaf681bee3a5a0f767b",
    "PREREG_E2_STICKY_REV3_2026-09-15.md": "8998d53277abf660bb42bffc388fc92aa05c8de61e991c241783d9c87e1b29a4",
    "PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md": "a9c872ab00fcbf280c90c0df88482526f70190f7f64d559cc363ab74f5b6bed8",
    "VERDICT_e2_rules_2026-09-15.md": "145b786ecb50c249e02bccf7c706eb2185eca153d5bf6c159b714bbfdc295595",
    "VERDICT_e2_rules_rev2_2026-09-15.md": "3386c8076d734e184630c2d3b9b6b747969e7409181f676d4bafd26eac4d9078",
    "VERDICT_e2_rules_rev3_2026-09-15.md": "396927b7d5f73e2617bc1ff5fe16f1a0080a99eaf063cfad9ebce10eb28fe962",
    "VERDICT_e2_harness_2026-09-15.md": "0971cef7ade95f7a47b2a84478eb21b1a67a5e9b87f4538ebc7ce1ef4b13426c",
    "VERDICT_e2_harness_rev2_2026-09-15.md": "9768ec7855c1f13dde445ccaf331f5a2cf81200f8531d987580cc12488e38ee3",
    "VERDICT_e2_substrate_portability_2026-09-18.md": "6b715c879db81b167c4a5a474689f8c212f44516b34b6018af11c2df0c3ffaed",
    "OVERRIDE_E2_SUBMIT_2026-09-15.md": "0e5833e377b7f962d5f627ec3aa286bd0e88794057d3fb6e2791f588061ef213",
}
# A13 (this port's addendum) is pinned too: editing A13 after audit requires editing this
# constant in the same commit, and tests/test_e2_vessl.py fails until both agree.
A13_NAME = "PREREG_E2_VESSL_A13_2026-10-04.md"
A13_DIGEST = "82055fe17f1d438a80b48d9412a21e4d7d300d5f2ce1ed15b284762247a719c0"
CONFIG_REL = "results/longctx_conflict/probes/pdmux_homog5.yml"
CONFIG_DIGEST = "be6dd6473ff8884f429b4d0d825e9cbf0b42b59da0ad4b136b542105c533625c"
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"
# the trace the λ0 VESSL Job read (SUBSTRATE_LAM0V.txt) = e1_plan.SHAREGPT_SHA256
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"

# --------------------------------------------------------------------- (2) selftest archive
# rev3 sec 5-3 / ADDENDUM A1: the seven 908623 telemetry files e2_realized_mix --selftest reads.
ARCHIVE_DIGESTS = {
    "tel_a_r0.jsonl": "736e110d3f633d122dc3de600e61c5b25c86b26dea7ebe5aba6d653c5cdfe6f3",
    "tel_a_r2.jsonl": "9395b41f2311f4ca2731f082d040a09b3c5d0b30f933a9181a00c6518a94328b",
    "tel_a_r4.jsonl": "6d5ca10167028884bc1a62e269dce97e537bc818c094faccc8242fc5aeceadec",
    "tel_a_r4_s2.jsonl": "a837b87c7022846936686b9ee5a20064ddaf46a748bcd926fc1c1b9fc5fa801b",
    "tel_b_r0.jsonl": "ae5154223f0add230f3608d97bf5344eca30a43aa213e8c93ee088143850b52f",
    "tel_b_r3.jsonl": "bb2f41e8882206c0c25dc1fcb7fe262b3fea81a3d55225424d706a708d50518b",
    "tel_b_r3_s2.jsonl": "3197c7d57f3276cd82997c66e363e9ed3d6d9a6784ff4955b861f0ca134f60c8",
}

# --------------------------------------------------------------------- (3) engine tree
# job 908623's manifest vs this commit's installed tree: the ONLY permitted difference.
ENGINE_ALLOWED_DIFF = {
    "sglang/srt/multiplex/multiplexing_mixin.py": {
        "ref_908623": "e2a97b423f93ff6d09f08b1e649596dad010285775845261ed27233c0346f243",
        "this_head": "ba5835aa9f935652860f73c2e38fdb550705f3d462b7ed25015c57d28d2aff8b",
    },
}
ENGINE_MIN_ENTRIES = 24

# --------------------------------------------------------------------- (4) substrate (B0)
SUBSTRATE_SM_COUNT = 108
SUBSTRATE_CC = "8.0"
SUBSTRATE_MIN_MEM_MIB = 79000
SUBSTRATE_MIN_DRIVER_MAJOR = 550        # CUDA 12.4-capable driver branch

# --------------------------------------------------------------------- arm / cells (rev3, unchanged)
FIXED_DSM = 44
CELLS = {   # rev3 sec 2: shape, in, out, offered, --num-prompts
    "a_r4": ("A", 256, 512, "8.00", 400),
    "a_r2": ("A", 256, 512, "3.00", 400),
    "b_r3": ("B", 8192, 64, "1.15", 200),
}
SEEDS_A = ("4386", "4162", "251", "2630")
SEEDS_B = ("4386", "4162")
ARMS = ("OFF", "ON")

# --------------------------------------------------------------------- (5) P7 T_cap
# rev3 sec 7 P7: T_cap = 3 x (OFF measured duration).  KISTI: e2_sticky.sbatch carries
# 393 / 428 / 861 = ceil(3 x the 908623 s1 duration rounded to 0.1 s, i.e. the rev3 sec 8-1
# table values 131.0 / 142.5 / 286.9) -- NOT ceil(3 x the raw duration), which would give 394
# for a_r4.  The same rounding is applied to the VESSL durations (identical result either way).  VESSL (A13 sec 2-2):
# the SAME cells, SAME seed s1, SAME (sticky OFF) arm, measured by the λ0 VESSL Job
# pdmux-lambda0-vessl-1d47d3e-20261004084945 (lam0v_descriptive.json vessl.<cell>.cell.duration_s).
T_CAP_MULT = 3.0
KISTI_OFF_DURATION_S = {"a_r4": 131.01310490909964, "a_r2": 142.45440463908017,
                        "b_r3": 286.8451705798507}
VESSL_OFF_DURATION_S = {"a_r4": 133.61504419101402, "a_r2": 142.90154187195003,
                        "b_r3": 289.66677306126803}
KISTI_T_CAP_S = {"a_r4": 393, "a_r2": 428, "b_r3": 861}       # e2_sticky.sbatch literal


def t_cap(duration_s: float) -> int:
    return int(math.ceil(round(T_CAP_MULT * round(duration_s, 1), 6)))


T_CAP_S = {c: t_cap(d) for c, d in VESSL_OFF_DURATION_S.items()}   # 401 / 429 / 870

# --------------------------------------------------------------------- (5) hard cap / P14 / budget
RATE_USD_H = 1.48
HARD_CAP_S = 14400                 # 4.0 h from job_entry start (user approval "≈4 h", 2026-10-04)
TAIL_RESERVE_S = 600               # P13 verdict + label + green summary after the last boot
P14_MARGIN_S = 200                 # rev3 sec 7 P14 literal, unchanged
# VESSL per-boot overhead = cell wall-clock (difference of consecutive `admit ... remaining_s`
# in the λ0 VESSL BUDGET.txt) minus bench duration; steady-state maxima over the cells of the
# same shape (A: the six N=400 cells a_r2/a_r3/a_r4/a_r4_s2/a_r4_s3/a_r4_s4 = 159.11-160.10 s,
# excluding a_r1 [N=1260, 163.44 s] and a_r0 [first boot]; B: all seven = 118.55-120.93 s).  Includes
# admission, boot, warm-up, post-bench analysis, teardown (5 + 10 s).  KISTI: 145.5 / 106.5 s.
OVERHEAD_S = {"A": 160.1, "B": 121.0}
COLD_FIRST_BOOT_EXTRA_S = 92.1     # λ0 VESSL a_r0 (first boot of the Job) 251.44 s - a_r4 159.38 s
P13_PROBE_ALLOW_S = 30.0           # per arm, first boot only; UNMEASURED allowance (6 x 48 tokens)
PREFLIGHT_ALLOW_S = 600.0          # job_entry start -> first boot; UNMEASURED for E2 (λ0: 732 s
                                   # incl. its ~10 min mutation harness, which E2 does not run)
WORST_BENCH_MULT = {"A": 2.5, "B": 1.5}   # rev3 sec 8-1 registered worst corner (ON arm only)


def units():
    """The 20 boots in registered order (rev3 sec 2): seed-major, OFF then ON adjacent."""
    out = []
    for cell, seeds in (("a_r4", SEEDS_A), ("a_r2", SEEDS_A), ("b_r3", SEEDS_B)):
        for s in seeds:
            for arm in ARMS:
                out.append((cell, s, arm))
    return out


def unit_rows():
    rows = []
    for cell, s, arm in units():
        shape, inl, outl, rate, np_ = CELLS[cell]
        rows.append(f"{cell} {s} {arm} {shape} {inl} {outl} {rate} {np_} {T_CAP_S[cell]}")
    return rows


def expect_pairs():
    seen, out = set(), []
    for cell, s, _ in units():
        if (cell, s) not in seen:
            seen.add((cell, s))
            out.append(f"{cell}/{s}")
    return ",".join(out)


def boot_cost_s(cell: str, arm: str, corner: str) -> float:
    shape = CELLS[cell][0]
    bench = VESSL_OFF_DURATION_S[cell]
    if corner == "worst" and arm == "ON":
        bench *= WORST_BENCH_MULT[shape]
    elif corner == "capped":
        bench = T_CAP_S[cell]
    return bench + OVERHEAD_S[shape]


def simulate(corner: str):
    """Walk the 20 boots with the registered P14 rule against the registered hard cap.
    Returns (admitted boots, total seconds incl. preflight/cold/probe, first refused boot)."""
    t = PREFLIGHT_ALLOW_S
    deadline = HARD_CAP_S - TAIL_RESERVE_S
    admitted, refused = [], None
    probed = set()
    for i, (cell, s, arm) in enumerate(units()):
        if deadline - t < T_CAP_S[cell] + P14_MARGIN_S:
            refused = f"{cell}/{s}/{arm}"
            break
        t += boot_cost_s(cell, arm, corner)
        if i == 0:
            t += COLD_FIRST_BOOT_EXTRA_S
        if arm not in probed:
            probed.add(arm)
            t += P13_PROBE_ALLOW_S
        admitted.append(f"{cell}/{s}/{arm}")
    return admitted, t, refused


def budget_table() -> dict:
    out = {"hard_cap_s": HARD_CAP_S, "hard_cap_gpu_h": HARD_CAP_S / 3600.0,
           "hard_cap_usd": round(HARD_CAP_S / 3600.0 * RATE_USD_H, 2),
           "tail_reserve_s": TAIL_RESERVE_S, "p14_margin_s": P14_MARGIN_S,
           "rate_usd_h": RATE_USD_H, "t_cap_s": T_CAP_S, "overhead_s": OVERHEAD_S,
           "cold_first_boot_extra_s": COLD_FIRST_BOOT_EXTRA_S,
           "p13_probe_allow_s": P13_PROBE_ALLOW_S, "preflight_allow_s": PREFLIGHT_ALLOW_S,
           "scenarios": {}}
    for corner in ("normal", "worst", "capped"):
        adm, t, refused = simulate(corner)
        out["scenarios"][corner] = {
            "boots_admitted": len(adm), "first_refused": refused,
            "container_s_to_last_boot_end": round(t, 1),
            "gpu_h": round(t / 3600.0, 3), "usd": round(t / 3600.0 * RATE_USD_H, 2)}
    # per-boot slack before the registered P14 rule would refuse the LAST boot (b_r3/4162/ON)
    # at the worst corner -- the VESSL analogue of the portability verdict's "+98 s/boot".
    adm, t, _ = simulate("worst")
    last = units()[-1]
    t_before_last = t - boot_cost_s(last[0], last[2], "worst")
    slack = (HARD_CAP_S - TAIL_RESERVE_S) - t_before_last - (T_CAP_S[last[0]] + P14_MARGIN_S)
    out["worst_corner_per_boot_slack_s"] = round(slack / (len(units()) - 1), 1)
    return out


def need_s(cell: str) -> int:
    """P14 (rev3 sec 7, ADDENDUM A4 re-anchored by A13 sec 5): remaining >= T_cap + 200 s."""
    return T_CAP_S[cell] + P14_MARGIN_S


# --------------------------------------------------------------------- checks
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_digests(prereg_dir: Path, here_dir: Path, engine_root: Path):
    bad = []
    for name, want in sorted(REGISTERED_DIGESTS.items()):
        p = Path(prereg_dir) / name
        got = sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"e2 {name}: {got} != registered {want}")
    p = Path(here_dir) / A13_NAME
    got = sha256(p) if p.exists() else "MISSING"
    if got != A13_DIGEST:
        bad.append(f"A13 {A13_NAME}: {got} != registered {A13_DIGEST}")
    p = Path(engine_root) / CONFIG_REL
    got = sha256(p) if p.exists() else "MISSING"
    if got != CONFIG_DIGEST:
        bad.append(f"config {CONFIG_REL}: {got} != registered {CONFIG_DIGEST}")
    return bad


def verify_archive(archive_dir: Path):
    bad = []
    for name, want in sorted(ARCHIVE_DIGESTS.items()):
        p = Path(archive_dir) / name
        got = sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"archive {name}: {got} != registered {want}")
    return bad


def manifest_entries(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        key = p.split("/python/", 1)[1] if "/python/" in p else p
        out[key] = h
    return out


def engine_diff(this_manifest: Path, ref_manifest: Path):
    """Return (problems, differing_keys).  Problems empty <=> the trees differ in exactly the
    registered file(s), each side carrying exactly the registered bytes."""
    a, b = manifest_entries(this_manifest), manifest_entries(ref_manifest)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    probs = []
    if len(a) < ENGINE_MIN_ENTRIES:
        probs.append(f"this manifest has {len(a)} entries < {ENGINE_MIN_ENTRIES}")
    if set(diff) != set(ENGINE_ALLOWED_DIFF):
        probs.append(f"differing files {diff} != registered {sorted(ENGINE_ALLOWED_DIFF)}")
    for k, want in ENGINE_ALLOWED_DIFF.items():
        if a.get(k) != want["this_head"]:
            probs.append(f"{k}: this={a.get(k, 'ABSENT')} != registered HEAD {want['this_head']}")
        if b.get(k) != want["ref_908623"]:
            probs.append(f"{k}: 908623={b.get(k, 'ABSENT')} != registered {want['ref_908623']}")
    return probs, diff


def check_substrate(csv_line: str, sm_count) -> list:
    """csv_line = `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,
    mig.mode.current --format=csv,noheader,nounits` (first GPU)."""
    f = [x.strip() for x in csv_line.split(",")]
    if len(f) < 5:
        return [f"unparseable nvidia-smi line: {csv_line!r}"]
    name, cc, mem, drv, mig = f[:5]
    bad = []
    if "A100" not in name or "80GB" not in name:
        bad.append(f"gpu name {name!r} is not an A100 80GB")
    if cc != SUBSTRATE_CC:
        bad.append(f"compute_cap {cc} != {SUBSTRATE_CC}")
    try:
        if float(mem) < SUBSTRATE_MIN_MEM_MIB:
            bad.append(f"memory.total {mem} MiB < {SUBSTRATE_MIN_MEM_MIB}")
    except ValueError:
        bad.append(f"memory.total {mem!r} unparseable")
    try:
        if int(drv.split(".")[0]) < SUBSTRATE_MIN_DRIVER_MAJOR:
            bad.append(f"driver {drv} < {SUBSTRATE_MIN_DRIVER_MAJOR}")
    except ValueError:
        bad.append(f"driver {drv!r} unparseable")
    if mig not in ("Disabled", "[N/A]", "N/A"):
        bad.append(f"MIG mode {mig!r} (full GPU required)")
    try:
        if int(sm_count) != SUBSTRATE_SM_COUNT:
            bad.append(f"SM count {sm_count} != {SUBSTRATE_SM_COUNT}")
    except (TypeError, ValueError):
        bad.append(f"SM count {sm_count!r} unparseable")
    return bad


def green_summary(out_dir: Path) -> dict:
    """E2C-38 (descriptive).  For every boot: does stream group 2 carry a (64,44) green
    context, and which group does the server name as the sticky target?  NEVER a decision
    input (portability verdict: wiring it to a rule would create a new estimand)."""
    res = {"POLICY": "DESCRIPTIVE ONLY -- not an input to e2_label.py (A13 sec 3, E2C-38)",
           "boots": {}}
    for p in sorted(Path(out_dir).glob("green_*.json")):
        name = p.stem[len("green_"):]
        try:
            g = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            res["boots"][name] = {"error": repr(exc)}
            continue
        row = {"sticky_target_index": g.get("sticky_target_index")}
        for grp in g.get("groups", []):
            if grp.get("stream_index") != 2:
                continue
            row["target"] = [grp.get("target_prefill_sm"), grp.get("target_decode_sm")]
            row["green"] = [((grp.get(side) or {}).get("green_sm") or {}).get("smCount")
                            for side in ("prefill", "decode")]
            row["is_sticky_target"] = grp.get("is_sticky_target")
        row["idx2_green_is_64_44"] = row.get("green") == [64, 44]
        res["boots"][name] = row
    res["n_boots"] = len(res["boots"])
    res["n_idx2_green_64_44"] = sum(1 for r in res["boots"].values() if r.get("idx2_green_is_64_44"))
    return res


# --------------------------------------------------------------------- selftest
def selftest() -> None:
    # rev3 cell set / order: 20 boots, seed-major, OFF before ON
    u = units()
    assert len(u) == 20, len(u)
    assert [x[2] for x in u] == ["OFF", "ON"] * 10
    assert expect_pairs() == ("a_r4/4386,a_r4/4162,a_r4/251,a_r4/2630,a_r2/4386,a_r2/4162,"
                              "a_r2/251,a_r2/2630,b_r3/4386,b_r3/4162")
    # the T_cap formula reproduces the KISTI literals from the KISTI durations (formula identity)
    for c, d in KISTI_OFF_DURATION_S.items():
        assert t_cap(d) == KISTI_T_CAP_S[c], (c, t_cap(d), KISTI_T_CAP_S[c])
    assert T_CAP_S == {"a_r4": 401, "a_r2": 429, "b_r3": 870}, T_CAP_S
    # the VESSL durations are the λ0 VESSL Job's cells (tracked descriptive file), if present
    if LAM0V_DESC.exists():
        d = json.loads(LAM0V_DESC.read_text())["vessl"]
        for c, v in VESSL_OFF_DURATION_S.items():
            assert d[c]["cell"]["duration_s"] == v, (c, d[c]["cell"]["duration_s"], v)
            assert d[c]["cell"]["client_seed"] == 4386, c
    # registered budget corners admit every boot (R4'' survives); capped corner is cut by P14
    b = budget_table()
    assert b["scenarios"]["normal"]["boots_admitted"] == 20, b
    assert b["scenarios"]["worst"]["boots_admitted"] == 20, b
    assert b["scenarios"]["normal"]["container_s_to_last_boot_end"] <= HARD_CAP_S - TAIL_RESERVE_S
    assert b["scenarios"]["worst"]["container_s_to_last_boot_end"] <= HARD_CAP_S - TAIL_RESERVE_S
    # the P14 margin is rev3's literal and the hard cap is within the approved ~4 h
    assert P14_MARGIN_S == 200 and HARD_CAP_S <= 4 * 3600
    # substrate checker: the λ0 VESSL substrate passes, a 40GB / H100 / MIG device fails
    ok = "NVIDIA A100-SXM4-80GB, 8.0, 81920, 580.105.08, Disabled"
    assert check_substrate(ok, 108) == [], check_substrate(ok, 108)
    assert check_substrate("NVIDIA A100-SXM4-40GB, 8.0, 40960, 580.105.08, Disabled", 108)
    assert check_substrate("NVIDIA H100 80GB HBM3, 9.0, 81559, 580.105.08, Disabled", 132)
    assert check_substrate("NVIDIA A100-SXM4-80GB, 8.0, 81920, 580.105.08, Enabled", 108)
    assert check_substrate(ok, 98)
    # engine diff: the registered one-file difference passes; zero or two differences fail
    ref = LAM0_KISTI / "runtime_source_manifest.sha256"
    if ref.exists():
        lines = ref.read_text().splitlines()
        head = [ln.replace(ENGINE_ALLOWED_DIFF["sglang/srt/multiplex/multiplexing_mixin.py"]["ref_908623"],
                           ENGINE_ALLOWED_DIFF["sglang/srt/multiplex/multiplexing_mixin.py"]["this_head"])
                for ln in lines]
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "head.sha256"
            hp.write_text("\n".join(head) + "\n")
            assert engine_diff(hp, ref)[0] == [], engine_diff(hp, ref)
            assert engine_diff(ref, ref)[0], "identical tree must FAIL (A13 sec 1-3)"
            two = list(head)
            two[0] = "0" * 64 + two[0][64:]
            hp.write_text("\n".join(two) + "\n")
            assert engine_diff(hp, ref)[0], "a second differing file must FAIL"
    print("E2V PLAN SELFTEST OK (20 boots in rev3 order; T_cap formula reproduces KISTI 393/428/861 "
          "and gives VESSL 401/429/870; normal+worst corners admit all 20 boots under the 4.0 h cap; "
          "substrate + engine-diff checkers discriminate)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-digests", nargs=3, metavar=("PREREG_DIR", "HERE_DIR", "ENGINE_ROOT"))
    ap.add_argument("--verify-archive", metavar="ARCHIVE_DIR")
    ap.add_argument("--engine-diff", nargs=2, metavar=("THIS_MANIFEST", "REF_MANIFEST"))
    ap.add_argument("--check-substrate", nargs=2, metavar=("NVSMI_CSV_LINE", "SM_COUNT"))
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--expect", action="store_true")
    ap.add_argument("--need", metavar="CELL")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--green-summary", metavar="OUT_DIR")
    ap.add_argument("--const", metavar="NAME")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.verify_digests:
        bad = verify_digests(*(Path(x) for x in a.verify_digests))
        for b in bad:
            print("DRIFT " + b)
        print(f"DIGESTS {'OK' if not bad else 'DRIFT'} ({len(REGISTERED_DIGESTS)} E2 files, A13, 1 config)")
        return 3 if bad else 0
    if a.verify_archive:
        bad = verify_archive(Path(a.verify_archive))
        for b in bad:
            print("DRIFT " + b)
        print(f"ARCHIVE {'OK' if not bad else 'DRIFT'} ({len(ARCHIVE_DIGESTS)} job-908623 telemetry files)")
        return 3 if bad else 0
    if a.engine_diff:
        probs, diff = engine_diff(Path(a.engine_diff[0]), Path(a.engine_diff[1]))
        for k in diff:
            print("ENGINE_DIFF " + k)
        for p in probs:
            print("ENGINE_PROBLEM " + p)
        print(f"ENGINE_TREE {'AS_REGISTERED' if not probs else 'DRIFT'} vs 908623 "
              f"({len(diff)} file(s) differ; registered: {sorted(ENGINE_ALLOWED_DIFF)})")
        return 3 if probs else 0
    if a.check_substrate:
        bad = check_substrate(*a.check_substrate)
        for b in bad:
            print("SUBSTRATE_PROBLEM " + b)
        print(f"SUBSTRATE {'OK' if not bad else 'MISMATCH'}")
        return 3 if bad else 0
    if a.units:
        print("\n".join(unit_rows()))
        return 0
    if a.expect:
        print(expect_pairs())
        return 0
    if a.need:
        print(need_s(a.need))
        return 0
    if a.budget:
        print(json.dumps(budget_table(), indent=2, sort_keys=True))
        return 0
    if a.green_summary:
        print(json.dumps(green_summary(Path(a.green_summary)), indent=2, sort_keys=True))
        return 0
    if a.const:
        print(globals()[a.const])
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
