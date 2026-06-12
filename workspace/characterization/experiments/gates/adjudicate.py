"""
adjudicate.py — automatic G0 / G1 adjudication → verdict JSON.

G0 (from E1 component sweep): is the SSM scan actually a dominant component?
    max scan_share_pct < 30  → G0 = WEAK   (the memory-bound-scan premise is thin)
    else                     → G0 = OK

G1 (from E2 batch-swept saturation): is prefill SM saturation bandwidth-bound or
grid-bound? This is the single failure point of the whole v2 framing.
    sm_sat_ssm batch-invariant (CIs overlap across batch) AND
        BW util at saturation ≥ ~85%                  → G1 = BW_MECHANISM
            (framing holds → E4 may proceed)
    sm_sat_ssm climbs toward total_sm as batch grows,
        tracking E0's grid_sat_sm prediction          → G1 = GRID_MECHANISM
            (framing rejected → fall back to spatial-only / donor decode)
    otherwise                                         → G1 = INCONCLUSIVE
            (recommend more batch points)

Outputs → results_v2/verdicts/{g0_verdict.json, g1_verdict.json}. Each carries the
verdict, the evidence numbers it was based on, and a recommended next action.
Evidence keys are measured/derived/metadata-labelled like every other v2 output.

This is an automatic aid; the final call is still a human's (the v1 gate said the
same). adjudicate never measures — it only reads E1/E2 CSVs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import sweep_spec as ss
from experiments.common.labels import Label, read_labeled_csv, write_labeled_json
from experiments.common.stats import ci_overlap

# thresholds (named so the verdict can cite them)
G0_SCAN_SHARE_MIN = 30.0      # % — below this G0 is WEAK
G1_BW_UTIL_MIN = 85.0         # % — saturation BW util to call it bandwidth-bound
G1_GRID_FRAC = 0.90           # sat_sm ≥ this × total_sm counts as "approaching total"


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# G0
# ---------------------------------------------------------------------------

def adjudicate_g0(e1_dir: Path) -> dict:
    files = sorted(e1_dir.glob("component_sweep_*.csv"))
    per_model = {}
    for fp in files:
        rows, _ = read_labeled_csv(fp)
        shares = [_f(r.get("scan_share_pct")) for r in rows]
        shares = [s for s in shares if s is not None]
        model = rows[0]["model"] if rows else fp.stem
        if shares:
            mx = max(shares)
            per_model[model] = {
                "max_scan_share_pct": round(mx, 3),
                "n_points": len(shares),
                "g0": "WEAK" if mx < G0_SCAN_SHARE_MIN else "OK",
            }
        else:
            per_model[model] = {"max_scan_share_pct": None, "n_points": 0, "g0": "NO_DATA"}

    if not per_model:
        verdict = "NO_DATA"
        action = "Run E1 (run_component_sweep.py) first."
    elif any(v["g0"] == "WEAK" for v in per_model.values()):
        verdict = "WEAK"
        action = ("Scan is not a dominant component for ≥1 model: reconsider the "
                  "memory-bound-scan premise before investing in E2/E4.")
    elif all(v["g0"] == "OK" for v in per_model.values()):
        verdict = "OK"
        action = "Scan share adequate → proceed to E2."
    else:
        verdict = "PARTIAL"
        action = "Some models lack data — complete E1 before G1."

    return {
        "gate": "G0",
        "verdict": verdict,
        "threshold_scan_share_pct": G0_SCAN_SHARE_MIN,
        "per_model": per_model,
        "recommended_action": action,
        "source": "E1 component_sweep",
    }


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------

def _load_ci_rows(e2_dir: Path):
    out = []
    for fp in sorted(e2_dir.glob("saturation_ci_*.csv")):
        rows, _ = read_labeled_csv(fp)
        out.extend(rows)
    return out


def adjudicate_g1_model(ci_rows: list[dict], model: str, layer_type: str,
                        total_sm: int) -> dict:
    """Decide BW vs GRID for one model's SSM saturation across batch."""
    rows = [r for r in ci_rows
            if r["model"] == model and r["layer_type"] == layer_type]
    # group by (chunk, context); within a group, vary batch.
    groups: dict = {}
    for r in rows:
        key = (r.get("chunk_granularity"), r.get("context_len"))
        groups.setdefault(key, []).append(r)

    group_verdicts = []
    for key, grp in groups.items():
        pts = []
        for r in grp:
            sat = _f(r.get("sat_sm_point"))
            lo = _f(r.get("sat_sm_ci_low"))
            hi = _f(r.get("sat_sm_ci_high"))
            if sat is None or lo is None or hi is None:
                continue
            pts.append({
                "batch": int(_f(r.get("batch")) or 0),
                "sat_sm": sat, "ci": (lo, hi),
                "bw_util_at_sat": _f(r.get("bw_util_pct_at_sat")),
                "grid_sat_e0": _f(r.get("grid_sat_sm_e0")),
            })
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p["batch"])

        base_ci = pts[0]["ci"]
        batch_invariant = all(ci_overlap(base_ci, p["ci"]) for p in pts)
        sat_vals = [p["sat_sm"] for p in pts]
        monotone_up = all(b >= a - 1e-9 for a, b in zip(sat_vals, sat_vals[1:]))
        approaches_total = sat_vals[-1] >= G1_GRID_FRAC * total_sm
        bw_utils = [p["bw_util_at_sat"] for p in pts if p["bw_util_at_sat"] is not None]
        bw_high = bool(bw_utils) and (min(bw_utils) >= G1_BW_UTIL_MIN)

        if batch_invariant and bw_high:
            gv = "BW_MECHANISM"
        elif monotone_up and approaches_total:
            gv = "GRID_MECHANISM"
        else:
            gv = "INCONCLUSIVE"

        group_verdicts.append({
            "chunk_granularity": key[0], "context_len": key[1],
            "verdict": gv,
            "batch_invariant": batch_invariant,
            "monotone_up": monotone_up,
            "approaches_total": approaches_total,
            "bw_high": bw_high,
            "sat_sm_by_batch": {p["batch"]: p["sat_sm"] for p in pts},
            "ci_by_batch": {p["batch"]: list(p["ci"]) for p in pts},
            "bw_util_by_batch": {p["batch"]: p["bw_util_at_sat"] for p in pts},
            "grid_sat_e0_by_batch": {p["batch"]: p["grid_sat_e0"] for p in pts},
        })

    # roll up group verdicts to a model verdict
    vs = [g["verdict"] for g in group_verdicts]
    if not vs:
        model_v = "NO_DATA"
    elif vs.count("BW_MECHANISM") and not vs.count("GRID_MECHANISM"):
        model_v = "BW_MECHANISM"
    elif vs.count("GRID_MECHANISM") and not vs.count("BW_MECHANISM"):
        model_v = "GRID_MECHANISM"
    elif vs.count("BW_MECHANISM") and vs.count("GRID_MECHANISM"):
        model_v = "MIXED"
    else:
        model_v = "INCONCLUSIVE"

    return {"model": model, "layer_type": layer_type,
            "verdict": model_v, "groups": group_verdicts}


def adjudicate_g1(e2_dir: Path, layer_type: str, total_sm: int) -> dict:
    ci_rows = _load_ci_rows(e2_dir)
    models = sorted({r["model"] for r in ci_rows}) if ci_rows else []
    per_model = [adjudicate_g1_model(ci_rows, m, layer_type, total_sm) for m in models]

    verdicts = [m["verdict"] for m in per_model]
    if not verdicts:
        overall, action = "NO_DATA", "Run E2 (run_batch_swept_sweep.py) first."
    elif all(v == "BW_MECHANISM" for v in verdicts):
        overall = "BW_MECHANISM"
        action = ("Framing holds: SSM saturation is bandwidth-bound and "
                  "batch-invariant → proceed to E4 concurrent.")
    elif any(v == "GRID_MECHANISM" for v in verdicts):
        overall = "GRID_MECHANISM"
        action = ("Framing rejected: SSM saturation tracks the grid (sat_sm → "
                  "total_sm with batch). Recommend fallback (spatial-only Falcon "
                  "split / donor decode); do NOT run E4 as the headline result.")
    else:
        overall = "INCONCLUSIVE"
        action = ("Inconclusive: add batch points between the diverging ones and "
                  "re-run E2 before deciding.")

    return {
        "gate": "G1",
        "verdict": overall,
        "layer_type": layer_type,
        "thresholds": {"bw_util_min_pct": G1_BW_UTIL_MIN,
                       "grid_frac_of_total": G1_GRID_FRAC, "total_sm": total_sm},
        "per_model": per_model,
        "recommended_action": action,
        "source": "E2 saturation_ci",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_G0_LABELS = {
    "gate": Label.METADATA, "verdict": Label.DERIVED,
    "threshold_scan_share_pct": Label.METADATA, "per_model": Label.DERIVED,
    "recommended_action": Label.METADATA, "source": Label.METADATA,
}
_G1_LABELS = {
    "gate": Label.METADATA, "verdict": Label.DERIVED, "layer_type": Label.METADATA,
    "thresholds": Label.METADATA, "per_model": Label.DERIVED,
    "recommended_action": Label.METADATA, "source": Label.METADATA,
}


def parse_args():
    p = argparse.ArgumentParser(description="G0/G1 adjudication → verdict JSON")
    p.add_argument("--e1-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e1")
    p.add_argument("--e2-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e2")
    p.add_argument("--out-dir", type=Path, default=Path(_CHAR) / "results_v2" / "verdicts")
    p.add_argument("--g1-layer-type", default="ssm",
                   help="SSM layer used for the G1 saturation test (default: ssm)")
    return p.parse_args()


def main():
    args = parse_args()
    total_sm = ss.total_sm()

    g0 = adjudicate_g0(args.e1_dir)
    g1 = adjudicate_g1(args.e2_dir, args.g1_layer_type, total_sm)

    write_labeled_json(args.out_dir / "g0_verdict.json", g0, _G0_LABELS)
    write_labeled_json(args.out_dir / "g1_verdict.json", g1, _G1_LABELS)

    print("=" * 64)
    print(f"  G0 = {g0['verdict']:14s}  ({g0['recommended_action'][:60]})")
    print(f"  G1 = {g1['verdict']:14s}  ({g1['recommended_action'][:60]})")
    print("=" * 64)
    print(f"  wrote {args.out_dir/'g0_verdict.json'}")
    print(f"  wrote {args.out_dir/'g1_verdict.json'}")
    print("  [human] the final gate decision is still yours.")


if __name__ == "__main__":
    main()
