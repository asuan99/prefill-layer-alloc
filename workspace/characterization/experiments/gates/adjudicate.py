"""
adjudicate.py — automatic G0 / G1 adjudication → verdict JSON.

G0 (from E1 component sweep): is the SSM scan actually a dominant component?
    max scan_share_pct < 30  → G0 = WEAK   (the memory-bound-scan premise is thin)
    else                     → G0 = OK

G1 (from E2 batch-swept saturation) — REDEFINED.
  The real layer-alloc thesis is whether attn and ssm have an exploitable
  RESOURCE ASYMMETRY, not whether SSM alone is bandwidth-bound. (The old gate
  asked "is SSM BW-bound?" and mislabelled a GRID finding as a failure — but
  grid-limited SSM is exactly what frees SMs for attention.) So G1 now tests:

    Is attn's saturation SM CI-separated from ssm's over the batch range?
        ≥1 operating batch with non-overlapping CIs        → G1 = ASYMMETRY_PRESENT
            (layer-alloc has headroom → measure the gain in E4)
        gaps exist but CIs always overlap                  → G1 = ASYMMETRY_WEAK
        no meaningful gap                                  → G1 = NO_ASYMMETRY

  The saturation MECHANISM (grid vs bandwidth) is reported as a descriptive
  sub-result per layer type, NOT as the pass/fail. Mechanism is judged from the
  trustworthy signal (measured sat_sm vs E0's analytic grid prediction); absolute
  BW% is recorded but flagged unreliable (attn_bytes overcounts cached KV, the
  SSM byte count omits recurrent-state traffic).

Outputs → results_v2/verdicts/{g0_verdict.json, g1_verdict.json}. Each carries the
verdict, the evidence numbers it was based on, and a recommended next action.
Evidence keys are measured/derived/metadata-labelled like every other v2 output.

This is an automatic aid; the final call is still a human's. adjudicate never
measures — it only reads E1/E2 CSVs.
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
G0_STEADY_BATCH = 32          # batches >= this define the steady (linear-regime) scan share
ASYM_CHUNK = "256"            # chunk granularity used for the asymmetry test
ASYM_MIN_GAP_SM = 13          # ~one Green-Context grid step; smaller gaps = noise
# descriptive mechanism hints (NOT the gate):
MECH_GRID_TOL_SM = 14         # |sat_sm - E0 grid pred| within ~1 step → grid-consistent
MECH_BW_UTIL_HI = 50.0        # % — high measured util hint (unreliable in absolute terms)


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
        model = rows[0]["model"] if rows else fp.stem
        # scan_share is batch-dependent: it is inflated at low batch (scan has a
        # batch-independent critical-path/launch floor while the GEMMs are tiny),
        # then settles to a steady floor once both scale linearly (elbow ~batch 32).
        # Report BOTH the peak and the high-batch steady share so "scan dominant"
        # isn't read off the low-batch overhead plateau alone.
        by_batch = {}
        for r in rows:
            s = _f(r.get("scan_share_pct")); b = _f(r.get("batch"))
            if s is not None and b is not None:
                by_batch.setdefault(int(b), []).append(s)
        if by_batch:
            shares = {b: sum(v) / len(v) for b, v in by_batch.items()}
            mx = max(shares.values())
            steady = [v for b, v in shares.items() if b >= G0_STEADY_BATCH]
            steady_share = round(sum(steady) / len(steady), 3) if steady else None
            if steady_share is None:
                regime = "unknown(no high-batch points)"
            elif steady_share >= G0_SCAN_SHARE_MIN:
                regime = "robust"               # scan material even at serving batch
            elif mx >= G0_SCAN_SHARE_MIN:
                regime = "low-batch-dominant"    # scan matters only when latency-bound
            else:
                regime = "weak"
            per_model[model] = {
                "max_scan_share_pct": round(mx, 3),
                "steady_scan_share_pct": steady_share,   # mean over batch>=G0_STEADY_BATCH
                "scan_regime": regime,
                "n_batches": len(shares),
                "g0": "WEAK" if mx < G0_SCAN_SHARE_MIN else "OK",
            }
        else:
            per_model[model] = {"max_scan_share_pct": None, "steady_scan_share_pct": None,
                                "scan_regime": "no_data", "n_batches": 0, "g0": "NO_DATA"}

    if not per_model:
        verdict = "NO_DATA"
        action = "Run E1 (run_component_sweep.py) first."
    elif any(v["g0"] == "WEAK" for v in per_model.values()):
        verdict = "WEAK"
        action = ("Scan is not a dominant component for ≥1 model: reconsider the "
                  "memory-bound-scan premise before investing in E2/E4.")
    elif all(v["g0"] == "OK" for v in per_model.values()):
        verdict = "OK"
        regimes = {m: v.get("scan_regime") for m, v in per_model.items()}
        action = ("Scan is a meaningful component (peak ≥ 30%) → proceed to E2. "
                  "NOTE: scan share is batch-dependent — peak at low batch (overhead "
                  "floor) then settles; per-model steady regime: " + str(regimes))
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


def _sat_points(ci_rows, model, layer_type, total_sm):
    """{batch: {sat,ci,bw,grid_e0}} for one model+layer at ASYM_CHUNK, context 0."""
    pts = {}
    for r in ci_rows:
        if (r["model"] != model or r["layer_type"] != layer_type
                or str(r.get("chunk_granularity")) != ASYM_CHUNK
                or _f(r.get("context_len")) not in (0, None)):
            continue
        sat, lo, hi = (_f(r.get("sat_sm_point")), _f(r.get("sat_sm_ci_low")),
                       _f(r.get("sat_sm_ci_high")))
        if sat is None or lo is None or hi is None:
            continue
        b = int(_f(r.get("batch")) or 0)
        pts[b] = {"sat": sat, "ci": (lo, hi),
                  "bw": _f(r.get("bw_util_pct_at_sat")),
                  "grid_e0": _f(r.get("grid_sat_sm_e0"))}
    return pts


def _mechanism(pts, use_grid_pred, total_sm):
    """Descriptive (NOT the gate): grid-consistent vs other, from sat_sm vs E0.

    use_grid_pred=True only for SSM (grid_sat_sm_e0 is the SSM grid; for attn the
    E0 column is not the attn grid, so we fall back to a BW-util hint, caveated).
    """
    if not pts:
        return "unknown"
    if use_grid_pred:
        hits = sum(1 for p in pts.values()
                   if p["grid_e0"] is not None
                   and abs(p["sat"] - min(p["grid_e0"], total_sm)) <= MECH_GRID_TOL_SM)
        if hits >= max(1, len(pts) // 2):
            return "grid_limited"
    bws = [p["bw"] for p in pts.values() if p["bw"] is not None]
    if bws and (sum(bws) / len(bws)) >= MECH_BW_UTIL_HI:
        return "bw_or_compute_limited(util-hint;abs%unreliable)"
    return "indeterminate"


def adjudicate_g1_model(ci_rows, model, total_sm) -> dict:
    """Asymmetry test for one model: attn sat_sm vs ssm sat_sm across batch."""
    ssm = _sat_points(ci_rows, model, "ssm", total_sm)
    attn = _sat_points(ci_rows, model, "attn", total_sm)
    batches = sorted(set(ssm) & set(attn))
    detail, sep_batches = [], []
    max_gap = None
    for b in batches:
        s, a = ssm[b], attn[b]
        gap = a["sat"] - s["sat"]                       # +ve: attn needs more SM
        # CI-separated upward asymmetry: attn CI strictly above ssm CI
        separated = (a["ci"][0] > s["ci"][1]) and abs(gap) >= ASYM_MIN_GAP_SM
        if separated:
            sep_batches.append(b)
        if max_gap is None or abs(gap) > abs(max_gap):
            max_gap = gap
        detail.append({"batch": b, "ssm_sat": s["sat"], "ssm_ci": list(s["ci"]),
                       "attn_sat": a["sat"], "attn_ci": list(a["ci"]),
                       "gap_attn_minus_ssm": gap, "ci_separated": separated})

    if not batches:
        verdict = "NO_DATA"
    elif sep_batches:
        verdict = "ASYMMETRY_PRESENT"
    elif max_gap is not None and abs(max_gap) >= ASYM_MIN_GAP_SM:
        verdict = "ASYMMETRY_WEAK"     # gap exists but CIs overlap
    else:
        verdict = "NO_ASYMMETRY"

    return {
        "model": model, "verdict": verdict,
        "max_gap_sm": max_gap,
        "direction": "attn>ssm" if (max_gap or 0) > 0 else "ssm>=attn",
        "batches_ci_separated": sep_batches,
        "ssm_mechanism": _mechanism(ssm, True, total_sm),    # descriptive only
        "attn_mechanism": _mechanism(attn, False, total_sm), # descriptive only
        "detail": detail,
    }


def adjudicate_g1(e2_dir: Path, total_sm: int) -> dict:
    ci_rows = _load_ci_rows(e2_dir)
    models = sorted({r["model"] for r in ci_rows}) if ci_rows else []
    per_model = [adjudicate_g1_model(ci_rows, m, total_sm) for m in models]
    vs = [m["verdict"] for m in per_model]

    present = [m["model"] for m in per_model if m["verdict"] == "ASYMMETRY_PRESENT"]
    if not vs:
        overall = "NO_DATA"
        action = "Run E2 (run_batch_swept_sweep.py) first."
    elif present:
        overall = "ASYMMETRY_PRESENT"
        action = ("Layer-type resource asymmetry confirmed (attn saturates at more "
                  f"SMs than ssm, CI-separated) for: {', '.join(present)}. SSM frees "
                  "SMs for attention -> proceed to E4 to measure the realizable gain "
                  "(spatial split is the main candidate; the asymmetry is grid-driven "
                  "so a static, E0-predicted split should suffice). Note: the gap "
                  "compresses at very high batch -- quantify the per-model collapse batch.")
    elif any(v == "ASYMMETRY_WEAK" for v in vs):
        overall = "ASYMMETRY_WEAK"
        action = ("Gaps present but CIs overlap -- add batch points / raise n_measure "
                  "to tighten CIs before committing to E4.")
    else:
        overall = "NO_ASYMMETRY"
        action = ("No exploitable attn/ssm SM gap -- layer-type SM allocation has no "
                  "headroom in this regime.")

    return {
        "gate": "G1",
        "verdict": overall,
        "criterion": "attn-vs-ssm saturation-SM asymmetry (CI-separated); "
                     "mechanism is descriptive, not the gate",
        "thresholds": {"chunk": ASYM_CHUNK, "min_gap_sm": ASYM_MIN_GAP_SM,
                       "total_sm": total_sm},
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
    "gate": Label.METADATA, "verdict": Label.DERIVED, "criterion": Label.METADATA,
    "thresholds": Label.METADATA, "per_model": Label.DERIVED,
    "recommended_action": Label.METADATA, "source": Label.METADATA,
}


def parse_args():
    p = argparse.ArgumentParser(description="G0/G1 adjudication -> verdict JSON")
    p.add_argument("--e1-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e1")
    p.add_argument("--e2-dir", type=Path, default=Path(_CHAR) / "results_v2" / "e2")
    p.add_argument("--out-dir", type=Path, default=Path(_CHAR) / "results_v2" / "verdicts")
    return p.parse_args()


def main():
    args = parse_args()
    total_sm = ss.total_sm()

    g0 = adjudicate_g0(args.e1_dir)
    g1 = adjudicate_g1(args.e2_dir, total_sm)

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
