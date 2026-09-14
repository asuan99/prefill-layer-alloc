#!/usr/bin/env python3
"""E2 -- registered reachability of EVERY verdict, not just the low-side ceiling.

WHY THIS FILE EXISTS
--------------------
rev2's audit made the plan pre-compute the low-side ceiling `kappa`.  rev3 did
that faithfully -- and the rev3 audit then found that the same repair had made
`KNEE_BRACKETED[shape B]` UNREACHABLE, because the drain guard attached to the
low-side certification also bit the HIGH side, where a large drain is precisely
the evidence a bracket needs.  Computing attainability for one threshold and not
for the others is how both kill causes happened.

So rev4 registers the whole map: for a grid of TRUE lambda* values, what verdict
does the SHIPPED decision path emit?  `lambda0_plan` -> synthetic cell -> the
shipped `lambda0_analyze.analyze` -> the shipped `lambda0_label.rule`.  Nothing
is hand-built, and the assertion is that every registered verdict has a
non-empty domain.

THE PHYSICAL MODEL (the only external input; registered, not tuned)
-------------------------------------------------------------------
    duration(cell) = max(span + drain_uncontended, N / lambda*)
    drain_uncontended = TTFT_floor(shape) + (out-1) * ITL(Bbar)
    ITL(B) = a' + b'*B,  a' = ITL_SOLO_S = 13.06 ms
    b' fixed by self-consistency: at B = MAX_RUNNING the engine delivers
    lambda*, i.e. ITL(48) = 48 / ((out-1) * lambda*)
    Bbar solves Bbar = x*(out-1)*ITL(Bbar)          (Little)

`a'` is a MEASUREMENT, not a guess, and it has now been confirmed three ways:
the registered constant 13.06 ms (P2 job 905712, concurrency 1); the rev3
audit's independent read of the last, uncontended request of the three archived
r0 cells (13.48 / 13.27 / 13.07 ms at D16/D44/D92 -- arm independent); and
job 907959's I2 probe on THIS model and shape, ITL(B=1) median **12.96 ms**.

★HONEST LIMIT, registered: what the drain model needs is the ITL at the LOADED
batch (Bbar ~ 6-20 on shape A's low rungs), and that is still unmeasured.  I2
measured B=1 only.  So this map is "reachability under the registered model",
never a prediction of the run.  The run is protected against the model being
wrong by `drain_model_ok` -- which after E1 disqualifies a cell from the LOW
side instead of voiding the shape.

usage:
  lambda0_reachability.py                 # the table embedded in the prereg
  lambda0_reachability.py --selftest
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lambda0_plan as PLAN          # noqa: E402
from lambda0_analyze import analyze  # noqa: E402
from lambda0_label import rule       # noqa: E402

# Registered sweep: true lambda* as a multiple of the measured lambda_inf.
# Spans well below the ladder (LADDER_TOO_HIGH territory) to well above it
# (LADDER_TOO_LOW), so every registered verdict gets a chance to appear.
LAMBDA_RATIO_GRID = (0.10, 0.20, 0.25, 0.35, 0.50, 0.70, 0.85,
                     1.00, 1.20, 1.45, 1.80, 2.50)
ITL_SOLO_S = PLAN.ITL_SOLO_S
TTFT_FLOOR = {s: PLAN.FLOOR_REF_8192_S * PLAN.SHAPE_IO[s][0] / 8192.0
              * PLAN.TTFT_LOAD_FACTOR for s in ("A", "B")}


def _itl_slope(out_len: int, lam_star: float) -> float:
    """b' such that the model saturates at exactly lambda* with B = 48."""
    itl_at_cap = PLAN.MAX_RUNNING / ((out_len - 1) * lam_star)
    return max(0.0, (itl_at_cap - ITL_SOLO_S) / PLAN.MAX_RUNNING)


def cell_physics(shape: str, x: float, n: int, lam_star: float) -> tuple:
    """(duration, drain_uncontended, itl) under the registered model."""
    in_len, out_len = PLAN.SHAPE_IO[shape]
    b = _itl_slope(out_len, lam_star)
    den = 1.0 - b * x * (out_len - 1)
    itl = ITL_SOLO_S / den if den > 0 else float("inf")
    if itl == float("inf") or x * (out_len - 1) * itl > PLAN.MAX_RUNNING:
        itl = PLAN.MAX_RUNNING / ((out_len - 1) * lam_star)      # capped
    drain = TTFT_FLOOR[shape] + (out_len - 1) * itl
    return drain, itl


def emit(lam_inf_a: float, lam_inf_b: float, ratio: float, ladders=None,
         tmpdir=None) -> dict:
    """Run the SHIPPED decision path at a given true lambda*/lambda_inf."""
    p = PLAN.plan(lam_inf_a, lam_inf_b, ladders=ladders)
    out = {}
    td = tmpdir or tempfile.mkdtemp()
    for shape in ("A", "B"):
        lam_star = ratio * p["lambda_inf"][shape]
        recs = []
        for c in p["shapes"][shape]["cells"]:
            n, x, seed = c["num_prompts"], c["offered_nominal"], p["seed1"]
            np.random.seed(seed)
            iv = np.random.exponential(1.0 / x, size=n - 1)
            span = float(iv.sum())
            drain, itl = cell_physics(shape, x, n, lam_star)
            duration = max(span + drain, n / lam_star)
            bp = Path(td, f"rb_{shape}_{c['name']}.jsonl")
            in_len, out_len = PLAN.SHAPE_IO[shape]
            bp.write_text(json.dumps({
                "errors": [""] * n, "ttfts": [TTFT_FLOOR[shape]] * n,
                "itls": [[itl] * (out_len - 1)] * n,
                "duration": duration, "completed": n, "request_rate": x,
                "request_throughput": n / duration,
                "random_input_len": in_len, "random_output_len": out_len,
                "random_range_ratio": 1.0,
                "total_input_tokens": n * in_len,
                "total_output_tokens": n * out_len,
                "concurrency": float("nan")}))
            recs.append(analyze(bp, shape, c["name"], n, seed,
                                kappa_pred=c["kappa_pred"],
                                drain_pred_s=c["drain_pred_s"],
                                low_side_candidate=c["low_side_candidate"],
                                t_measure_s=c["T_measure_s"]))
        r = rule({shape: recs})[shape]
        lam = r["lambda_star"]
        out[shape] = {
            "true_lambda_star": lam_star, "verdict": r["verdict"],
            "reported_lambda_star": lam,
            "rel_err": None if lam is None else (lam - lam_star) / lam_star,
            "n_usable_low_side": r.get("n_usable_low_side"),
            "n_low_side_candidates": r.get("n_low_side_candidates"),
        }
    return out


SCENARIOS = PLAN.REGISTERED_GRID + (
    ("FALLBACK literal ladders", 2.10, 0.675),
)


def table() -> tuple:
    """The registered reachability map + the per-verdict domain census."""
    lines, seen = [], {"A": {}, "B": {}}
    with tempfile.TemporaryDirectory() as td:
        for title, la, lb in SCENARIOS:
            lad = ({k: tuple(v) for k, v in PLAN.FALLBACK_LADDER.items()}
                   if title.startswith("FALLBACK") else None)
            lines.append(f"### {title}   lambda_inf=({la}, {lb})"
                         + ("  [FALLBACK ladders]" if lad else ""))
            lines.append("    lam*/lam_inf |            shape A            |"
                         "            shape B")
            for ratio in LAMBDA_RATIO_GRID:
                r = emit(la, lb, ratio, ladders=lad, tmpdir=td)
                cells = []
                for s in ("A", "B"):
                    d = r[s]
                    err = ("    --  " if d["rel_err"] is None
                           else f"{100*d['rel_err']:+6.1f}%")
                    cells.append(f"{d['verdict']:<18s} lam*={d['true_lambda_star']:6.3f}"
                                 f" err={err} low={d['n_usable_low_side']}/"
                                 f"{d['n_low_side_candidates']}")
                    seen[s].setdefault(d["verdict"], []).append(
                        (title, round(ratio, 2)))
                lines.append(f"    {ratio:12.2f} | {cells[0]} | {cells[1]}")
            lines.append("")
    return "\n".join(lines), seen


def selftest() -> None:
    txt, seen = table()
    # ★E2: every registered verdict the SWEEP can produce must have a non-empty
    # domain, for BOTH shapes.  rev3 failed exactly here: KNEE_BRACKETED[B] and
    # LADDER_TOO_HIGH[A,B] were empty and nothing in the registration noticed.
    required = ("KNEE_BRACKETED", "KNEE_NOT_BRACKETED",
                "LADDER_TOO_HIGH", "LADDER_TOO_LOW")
    for s in ("A", "B"):
        for v in required:
            assert seen[s].get(v), (
                f"verdict {v} has an EMPTY domain for shape {s} -- the "
                f"registration cannot emit it under its own model", sorted(seen[s]))

    # ...and the bracket must land on lambda* when it fires.  A bracket that
    # reports the wrong number is worse than no bracket.
    with tempfile.TemporaryDirectory() as td:
        for title, la, lb in SCENARIOS[:1]:
            for ratio in (0.35, 0.50, 0.70, 0.85, 1.00):
                r = emit(la, lb, ratio, tmpdir=td)
                for s in ("A", "B"):
                    if r[s]["verdict"] == "KNEE_BRACKETED":
                        assert abs(r[s]["rel_err"]) < 0.15, (title, ratio, s, r[s])

    # ★NEGATIVE CONTROL (lesson 53): with rev3's rule -- the drain guard back in
    # R0, voiding the shape -- shape B must go UNRESOLVED at the anchor.  If
    # this passes, E1 changed nothing.
    import lambda0_label as LBL
    saved = LBL._usable_low, LBL._cell_invalid

    def rev3_usable_low(c):
        return bool(c.get("low_side_candidate", False))

    def rev3_cell_invalid(c):
        r = saved[1](c)
        if r:
            return r
        if c.get("low_side_candidate", False) and not c.get("drain_model_ok", False):
            return "rev3: drain past prediction voids the SHAPE"
        return None
    LBL._usable_low, LBL._cell_invalid = rev3_usable_low, rev3_cell_invalid
    try:
        with tempfile.TemporaryDirectory() as td:
            got = [emit(2.10, 0.675, r, tmpdir=td)["B"]["verdict"]
                   for r in (0.50, 0.70, 1.00)]
        assert all(v == "UNRESOLVED" for v in got), (
            "the rev3 rule must still fail here; if it does not, this control "
            "is not exercising the kill cause", got)
    finally:
        LBL._usable_low, LBL._cell_invalid = saved

    print("REACHABILITY SELFTEST OK (shipped analyze->rule over %d true-lambda* "
          "points x %d scenarios x 2 shapes; all FOUR registered "
          "verdicts have non-empty domains for both shapes; bracket "
          "reports lambda* within 15%%; NEGATIVE CONTROL: the rev3 rule still "
          "returns UNRESOLVED for shape B at the anchor)"
          % (len(LAMBDA_RATIO_GRID), len(SCENARIOS)))


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return 0
    if len(sys.argv) != 1:
        print(__doc__)
        return 2
    txt, seen = table()
    print(txt)
    print("### verdict domain census (must be non-empty for each registered verdict)")
    for s in ("A", "B"):
        for v, where in sorted(seen[s].items()):
            print(f"    shape {s}  {v:<20s} {len(where):3d} grid points  "
                  f"e.g. {where[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
