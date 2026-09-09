#!/usr/bin/env python3
"""Registered decision rules for PREREG_CAPACITY_2026-09-09.md (probe C).

Applies ONLY the rules written down in the prereg, mechanically, over the
per-cell JSONs produced by `c_capacity_analyze.py`.  No rule here may be added
or retuned after seeing the data (methodology gate #34: the decision rule is
audited before the campaign, not after).

Rules (prereg sec 4; ceilings pre-computed in prereg sec 3):

  R1 KNEE_BRACKETED[arm]   -- out=96 cells only.  Exists a tested rate with
                              achieved/offered >= 0.95 AND a strictly higher
                              tested rate with achieved/offered <= 0.90.
                              Existence, both sides attainable (P1 d44 already
                              observed 0.663 at the upper side).
  R2 P_C1 (out=96)         -- registered prediction: max L_decode_exact over
                              all out=96 cells is < 3.0.
  R3 P_C2 (out=384)        -- registered prediction: max L_decode_exact over
                              all out=384 cells is >= 3.0.
  R4 MODEL_HOLDS           -- for every SATURATED cell (achieved/offered
                              <= 0.90), |L_decode_obs/pred - 1| <= 0.15.
  R5 MU_P_D16              -- measurement only, NO threshold: the saturated
                              throughput of d16, previously never measured.

A cell whose JSON is missing (boot failure / bench failure) is UNRESOLVED and
is reported as such -- a measurement failure is NOT relabelled as a rule
failure (methodology gate #21).

Usage: python3 c_capacity_label.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = (16, 44, 92)
RATE_KEYS_96 = ("r0", "r1", "r2")   # ascending offered rate, out=96
CELL_96 = [f"d{d}_{r}_o96" for d in ARMS for r in RATE_KEYS_96]
CELL_384 = [f"d{d}_r1_o384" for d in ARMS]

L_DECODE_THRESHOLD = 3.0
ACH_HI = 0.95
ACH_LO = 0.90
MODEL_TOL = 0.15


def load_cells(out_dir):
    cells, missing = {}, []
    for name in CELL_96 + CELL_384:
        p = Path(out_dir) / f"cell_{name}.json"
        if p.exists():
            try:
                cells[name] = json.loads(p.read_text())
            except Exception as exc:
                missing.append(f"{name} (unparseable: {exc})")
        else:
            missing.append(name)
    return cells, missing


def rule_knee(cells):
    out = {}
    for d in ARMS:
        seq = []
        for r in RATE_KEYS_96:
            c = cells.get(f"d{d}_{r}_o96")
            if c:
                seq.append((c["offered_rate"], c["achieved_over_offered"]))
        seq.sort()
        bracketed = any(
            seq[i][1] >= ACH_HI and any(seq[j][1] <= ACH_LO for j in range(i + 1, len(seq)))
            for i in range(len(seq))
        )
        out[f"d{d}"] = {
            "verdict": "KNEE_BRACKETED" if bracketed else (
                "KNEE_NOT_BRACKETED" if len(seq) == len(RATE_KEYS_96) else "UNRESOLVED"),
            "n_cells_present": len(seq),
            "offered_vs_ach_over_off": seq,
        }
    return out


def rule_ldecode(cells, names, label, direction):
    vals = {n: cells[n]["L_decode_exact"] for n in names if n in cells}
    if len(vals) < len(names):
        return {"verdict": "UNRESOLVED", "reason": "not all cells present",
                "L_decode_exact": vals}
    m = max(vals.values())
    if direction == "lt":
        ok = m < L_DECODE_THRESHOLD
    else:
        ok = m >= L_DECODE_THRESHOLD
    return {
        "verdict": f"{label}_SUPPORTED" if ok else f"{label}_REFUTED",
        "max_L_decode_exact": m,
        "threshold": L_DECODE_THRESHOLD,
        "direction": direction,
        "L_decode_exact": vals,
    }


def rule_model(cells):
    checked, viol = {}, []
    for name, c in cells.items():
        if c["achieved_over_offered"] > ACH_LO:
            continue  # not saturated -> the closed form does not apply
        ratio = c.get("L_decode_obs_over_pred")
        checked[name] = ratio
        if ratio is None or ratio != ratio or abs(ratio - 1.0) > MODEL_TOL:
            viol.append(name)
    if not checked:
        return {"verdict": "UNRESOLVED", "reason": "no saturated cell"}
    return {
        "verdict": "MODEL_HOLDS" if not viol else "MODEL_VIOLATED",
        "tolerance": MODEL_TOL,
        "obs_over_pred_by_saturated_cell": checked,
        "violations": viol,
    }


def rule_mu_d16(cells):
    seq = []
    for r in RATE_KEYS_96:
        c = cells.get(f"d16_{r}_o96")
        if c:
            seq.append({"offered": c["offered_rate"], "achieved": c["achieved_rate"],
                        "ach_over_off": c["achieved_over_offered"]})
    sat = [s["achieved"] for s in seq if s["ach_over_off"] <= ACH_LO]
    return {
        "verdict": "MEASURED" if sat else ("NO_SATURATED_CELL" if seq else "UNRESOLVED"),
        "note": "measurement only, no registered threshold",
        "mu_p_d16_saturated_req_per_s": max(sat) if sat else None,
        "prior_extrapolation_req_per_s": 0.94,
        "cells": seq,
    }


def main():
    out_dir = sys.argv[1]
    cells, missing = load_cells(out_dir)
    result = {
        "probe": "C_CAPACITY",
        "prereg": "PREREG_CAPACITY_2026-09-09.md",
        "n_cells_present": len(cells),
        "n_cells_expected": len(CELL_96) + len(CELL_384),
        "missing_cells": missing,
        "R1_knee_bracketed": rule_knee(cells),
        "R2_P_C1_out96_lt3": rule_ldecode(cells, CELL_96, "P_C1", "lt"),
        "R3_P_C2_out384_ge3": rule_ldecode(cells, CELL_384, "P_C2", "ge"),
        "R4_model_holds": rule_model(cells),
        "R5_mu_p_d16": rule_mu_d16(cells),
        "crosscheck_rel_err_max": max(
            (c.get("crosscheck_rel_err", 0) for c in cells.values()), default=None),
    }
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(text)
    Path(out_dir, "C_LABEL.json").write_text(text)


if __name__ == "__main__":
    main()
