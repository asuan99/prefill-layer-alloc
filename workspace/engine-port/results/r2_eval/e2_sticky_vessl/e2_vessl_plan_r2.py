#!/usr/bin/env python3
"""E2 VESSL re-run (A14, runner r2) -- the constants and checks A14 ADDS.  Nothing else.

WHAT THIS FILE IS.  `e2_vessl_plan.py` is the first Job's sealed helper (Job
pdmux-e2-sticky-vessl-e6f424e-20261004135228, VERDICT_e2_vessl_rules_rev2 sec 5) and is NOT
edited.  The r2 runner `e2_sticky_vessl_r2.sh` still calls it for everything it did before
(rev3/A13 digests, archive, engine tree, substrate, units, T_cap, hard cap, green summary).
This file adds only what `PREREG_E2_VESSL_A14_2026-10-05.md` (A14) registers:

  1. DIGESTS.  A14 and the r2 runner are pinned here; the first Job's sealed files must still
     be their sealed bytes (A14 sec 0: kept as the record of the first Job).  `--verify-digests-r2`
     exits 3 on any drift.
  2. P3 option (ii) (A14 sec 3, user decision 2026-10-05): mamba == 48 exactly on every boot,
     every max_total_num_tokens >= 48 x 16384 = 786,432, (max - min) / max <= 1e-4.
     `--p3-verdict` (exit 0 PASS / 8 FAIL).
  3. BUDGET (A14 sec 5): the registered corners re-walked (identity with e2_vessl_plan.simulate
     is a selftest assertion), plus the observed-ON corner of the first Job and the settle worst
     corner (O4 runs BEFORE P14 admission in r2, so its time is visible to admission).

It re-implements NO E2 rule and NO estimator (`../e2_sticky_prereg/e2_{realized_mix,label}.py`
are still called by the runner and digest-enforced by e2_vessl_plan.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import e2_vessl_plan as V1  # noqa: E402  (sealed first-Job helper, imported, never copied)

# --------------------------------------------------------------------- (1) digests
# A14 is pinned: editing A14 after audit requires editing this constant in the same commit,
# and tests/test_e2_vessl_r2.py fails until both agree.  Same for the r2 runner.
A14_NAME = "PREREG_E2_VESSL_A14_2026-10-05.md"
A14_DIGEST = "524a08ea7b19ca238e4b69197d61f088a2b003a87f7f75df558ce21f907b2c5a"
RUNNER_R2_NAME = "e2_sticky_vessl_r2.sh"
RUNNER_R2_DIGEST = "d4a146a9f8df65fe22470fe2a689e054d4a5b7161dd58b75c2e2a51b89999fb9"
# The first Job's sealed bytes (VERDICT_e2_vessl_rules_rev2_2026-10-04.md sec 5, harness seal;
# the OVERRIDE row is its consumed-record bytes of commit 9a29f84, which the first Job's
# REGISTRATION does NOT carry -- it carried 46c85d71..., see A14 sec 4-1).  Relative to
# workspace/engine-port/.  They are kept unchanged as the record of that Job.
SEALED_R1 = {
    "results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh":
        "921dfd2df4c532ee2e6e0452533ca5a0cb09fd541daca0dcd4cda927af00a6c4",
    "results/r2_eval/e2_sticky_vessl/e2_vessl_plan.py":
        "57220a46bc8e57d60f470d45accabdbdba5e99135435a0e19c06eb3c5354700f",
    "results/r2_eval/e2_sticky_vessl/PREREG_E2_VESSL_A13_2026-10-04.md":
        "82055fe17f1d438a80b48d9412a21e4d7d300d5f2ce1ed15b284762247a719c0",
    "results/r2_eval/e2_sticky_vessl/stage_e2_inputs.sh":
        "4b27bb0ab53149bf8fb9cd0179da71e5b86b4569fe849e79f7c5a034642d54cb",
    "results/r2_eval/e2_sticky_vessl/OVERRIDE_E2_VESSL_2026-10-04.md":
        "131abfff249221476efdd03d3a995e6a75818d3862adc4b98dc561243f8281b7",
    "tests/test_e2_vessl.py":
        "d163d7020cbe4d737904036b906d63be548529c452e84f924bca415b5a99207c",
}

# --------------------------------------------------------------------- (2) P3 option (ii)
P3_MAX_RUNNING = 48          # = e2_sticky.sbatch MAX_RUNNING (rev3 sec 3)
P3_CTX = 16384               # = e2_sticky.sbatch CTX
P3_MAMBA_EXPECT = 48         # --max-running-requests 48 + --disable-radix-cache (E2C-10)
P3_TOKEN_FLOOR = P3_MAX_RUNNING * P3_CTX     # 786,432: KV can never bind admission above this
P3_REL_TOL = 1e-4


def p3_verdict(rows, max_running: int, ctx: int):
    """rows = P3_KEYS.txt lines split on whitespace (`<mamba> <tokens>`, one per boot so far).
    Returns (ok, line).  Fail-closed on anything unparseable."""
    if int(max_running) != P3_MAX_RUNNING or int(ctx) != P3_CTX:
        return False, (f"FAIL registered_constants max_running={max_running} ctx={ctx} "
                       f"!= {P3_MAX_RUNNING} x {P3_CTX}")
    rows = [r for r in rows if r]
    if not rows:
        return False, "FAIL no_keys"
    if any(len(r) != 2 or not r[0].isdecimal() or not r[1].isdecimal() for r in rows):
        return False, "FAIL banner_missing " + ";".join(" ".join(r) for r in rows if not (
            len(r) == 2 and r[0].isdecimal() and r[1].isdecimal()))
    mamba = sorted({int(r[0]) for r in rows})
    tok = [int(r[1]) for r in rows]
    if mamba != [P3_MAMBA_EXPECT]:
        return False, f"FAIL mamba={mamba} != [{P3_MAMBA_EXPECT}]"
    if min(tok) < P3_TOKEN_FLOOR:
        return False, f"FAIL tokens_min={min(tok)} < floor={P3_TOKEN_FLOOR}"
    rel = (max(tok) - min(tok)) / max(tok)
    ok = rel <= P3_REL_TOL
    return ok, (("PASS" if ok else "FAIL") + f" mamba={P3_MAMBA_EXPECT} tokens_min={min(tok)} "
                f"tokens_max={max(tok)} rel={rel:.3e} tol={P3_REL_TOL:g} floor={P3_TOKEN_FLOOR} "
                f"n_boots={len(rows)} distinct_tokens={len(set(tok))}")


def p3_verdict_file(path, max_running, ctx):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return p3_verdict(rows, max_running, ctx)


# --------------------------------------------------------------------- (3) budget
# Observed in the first Job (TIMING.tsv end_epoch - admit_epoch): non-first a_r4 ON boots
# 373 / 374 / 373 s (first ON boot 391 s incl. its P13 probe); the registered "normal" model
# gives a_r4 ON 293.7 s (bench = OFF bench).  ON bench_serving duration 197.2-197.7 s vs OFF
# 138.0-139.5 s (DESCRIPTIVE_e2v_20261004135228 sec 1-a).  a_r2 ON was never observed (boot
# failed); it is ASSUMED to take the same 375 s.  B ON uses the registered worst corner (x1.5).
OBS_A_ON_BOOT_S = 375.0
SETTLE_MAX_S = 180.0         # O4: 36 polls x 5 s
CORNERS = ("normal", "observed", "worst", "worst_settle", "capped")


def boot_cost_r2(cell: str, arm: str, corner: str) -> float:
    shape = V1.CELLS[cell][0]
    if corner == "observed":
        if arm == "ON":
            return OBS_A_ON_BOOT_S if shape == "A" else V1.boot_cost_s(cell, arm, "worst")
        return V1.boot_cost_s(cell, arm, "normal")
    if corner == "worst_settle":
        return V1.boot_cost_s(cell, arm, "worst")
    return V1.boot_cost_s(cell, arm, corner)


def simulate_r2(corner: str):
    """V1.simulate's walk, with the r2 order: settle (corner worst_settle only: SETTLE_MAX_S)
    -> P14 admission -> boot.  For the other corners settle is 0 and the walk is V1.simulate's."""
    t = V1.PREFLIGHT_ALLOW_S
    deadline = V1.HARD_CAP_S - V1.TAIL_RESERVE_S
    admitted, refused, probed = [], None, set()
    for i, (cell, s, arm) in enumerate(V1.units()):
        if corner == "worst_settle":
            t += SETTLE_MAX_S
        if deadline - t < V1.T_CAP_S[cell] + V1.P14_MARGIN_S:
            refused = f"{cell}/{s}/{arm}"
            break
        t += boot_cost_r2(cell, arm, corner)
        if i == 0:
            t += V1.COLD_FIRST_BOOT_EXTRA_S
        if arm not in probed:
            probed.add(arm)
            t += V1.P13_PROBE_ALLOW_S
        admitted.append(f"{cell}/{s}/{arm}")
    return admitted, t, refused


def budget_table() -> dict:
    out = {"source": "e2_vessl_plan_r2 (A14 sec 5); registered constants from e2_vessl_plan",
           "hard_cap_s": V1.HARD_CAP_S, "hard_cap_gpu_h": V1.HARD_CAP_S / 3600.0,
           "hard_cap_usd": round(V1.HARD_CAP_S / 3600.0 * V1.RATE_USD_H, 2),
           "rate_usd_h": V1.RATE_USD_H, "obs_a_on_boot_s": OBS_A_ON_BOOT_S,
           "settle_max_s": SETTLE_MAX_S, "scenarios": {}}
    for corner in CORNERS:
        adm, t, refused = simulate_r2(corner)
        out["scenarios"][corner] = {
            "boots_admitted": len(adm), "first_refused": refused,
            "container_s_to_last_boot_end": round(t, 1),
            "gpu_h": round(t / 3600.0, 3), "usd": round(t / 3600.0 * V1.RATE_USD_H, 2)}
    return out


# --------------------------------------------------------------------- checks
def verify_r2(here_dir: Path, engine_root: Path):
    bad = []
    for name, want in ((A14_NAME, A14_DIGEST), (RUNNER_R2_NAME, RUNNER_R2_DIGEST)):
        p = Path(here_dir) / name
        got = V1.sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"r2 {name}: {got} != registered {want}")
    for rel, want in sorted(SEALED_R1.items()):
        p = Path(engine_root) / rel
        got = V1.sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"sealed-r1 {rel}: {got} != sealed {want}")
    return bad


# --------------------------------------------------------------------- selftest
JOB1_KEYS_OK = [["48", "2722025"]] * 9 + [["48", "2722022"]]     # first Job, boots 1-9 + 11


def selftest() -> None:
    ok, line = p3_verdict(JOB1_KEYS_OK, 48, 16384)
    assert ok and line.startswith("PASS"), line
    cases = {   # every one of these MUST fail (lesson 53: the gate must discriminate)
        "mamba 47/48": [["48", "2722025"], ["47", "2722025"]],
        "mamba all 47": [["47", "2722025"], ["47", "2722025"]],
        "tokens 0.8%": [["48", "2722025"], ["48", str(int(2722025 * 0.992))]],
        "below floor": [["48", "786431"], ["48", "786431"]],
        "banner missing": [["48", "2722025"], ["NONE", "2722025"]],
        "tokens missing": [["48", "2722025"], ["48", "NONE"]],
        "no keys": [],
    }
    for name, rows in cases.items():
        ok, line = p3_verdict(rows, 48, 16384)
        assert not ok and line.startswith("FAIL"), (name, line)
    assert not p3_verdict(JOB1_KEYS_OK, 47, 16384)[0], "unregistered MAX_RUNNING must fail"
    assert not p3_verdict(JOB1_KEYS_OK, 48, 8192)[0], "unregistered CTX must fail"
    # boundaries: floor itself passes; rel == 1e-4 passes, just above fails
    assert p3_verdict([["48", "786432"]], 48, 16384)[0]
    assert p3_verdict([["48", "1000000"], ["48", "999900"]], 48, 16384)[0]
    assert not p3_verdict([["48", "1000000"], ["48", "999899"]], 48, 16384)[0]
    # budget: the registered corners re-walked here equal e2_vessl_plan.simulate (identity)
    for c in ("normal", "worst", "capped"):
        a1, t1, r1 = V1.simulate(c)
        a2, t2, r2 = simulate_r2(c)
        assert (a1, round(t1, 6), r1) == (a2, round(t2, 6), r2), c
    b = budget_table()["scenarios"]
    assert b["observed"]["boots_admitted"] == 20 and b["worst"]["boots_admitted"] == 20, b
    assert b["worst_settle"]["boots_admitted"] == 20, b
    assert b["normal"]["gpu_h"] < b["observed"]["gpu_h"] < b["worst"]["gpu_h"], b
    print("E2V r2 SELFTEST OK (P3 (ii): first-Job keys PASS; mamba 47/48, all-47, 0.8% token "
          "drift, below 786432, banner/tokens missing, no keys, unregistered constants FAIL; "
          "boundaries hold; registered budget corners == e2_vessl_plan.simulate; observed/worst/"
          "worst+settle admit all 20 boots)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-digests-r2", nargs=2, metavar=("HERE_DIR", "ENGINE_ROOT"))
    ap.add_argument("--p3-verdict", nargs=3, metavar=("P3_KEYS", "MAX_RUNNING", "CTX"))
    ap.add_argument("--budget", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.verify_digests_r2:
        bad = verify_r2(*(Path(x) for x in a.verify_digests_r2))
        for b in bad:
            print("DRIFT " + b)
        print(f"DIGESTS_R2 {'OK' if not bad else 'DRIFT'} (A14, runner r2, {len(SEALED_R1)} first-Job sealed files)")
        return 3 if bad else 0
    if a.p3_verdict:
        ok, line = p3_verdict_file(*a.p3_verdict)
        print(line)
        return 0 if ok else 8
    if a.budget:
        print(json.dumps(budget_table(), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
