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

`a'` is a MEASUREMENT, not a guess, but it has NOT been "confirmed three
ways" (F3-(4) of VERDICT_lambda0_rev4_2026-09-14.md; R4C-6).  All three
numbers below are the SAME quantity measured three times -- every one of
them is at B=1, and the engine only uses D44 when prefill and decode are
concurrent, so all three are UNSPLIT 108-SM values (N-7).  They are:
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


_PLAN_CACHE = {}


def _plan(lam_inf_a, lam_inf_b, ladders):
    """`PLAN.plan` memoised across the ratio sweep.

    The plan does not depend on the true lambda* being swept, only on
    (lambda_inf, ladders), and each call spends ~0.5 s searching 5000 seeds.
    rev5 routes plan and analyze mutations through this map as well (F5-1), so
    the map is now run ~20x more often and the 84 redundant plan() calls per
    sweep mattered.  Purity is asserted separately (`lambda0_plan` selftest
    item 6), so caching cannot hide a mutation.
    """
    key = (lam_inf_a, lam_inf_b,
           None if ladders is None else tuple(sorted(
               (k, tuple(v)) for k, v in ladders.items())))
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = PLAN.plan(lam_inf_a, lam_inf_b, ladders=ladders)
    return _PLAN_CACHE[key]


def emit(lam_inf_a: float, lam_inf_b: float, ratio: float, ladders=None,
         tmpdir=None) -> dict:
    """Run the SHIPPED decision path at a given true lambda*/lambda_inf."""
    p = _plan(lam_inf_a, lam_inf_b, ladders)
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

VERDICT_CODE = {"KNEE_BRACKETED": "B", "KNEE_NOT_BRACKETED": "N",
                "LADDER_TOO_HIGH": "H", "LADDER_TOO_LOW": "L",
                "UNRESOLVED": "U"}

# ★rev5/F5-1+2 -- THE REGISTERED MAP, not just its census.
#
# One code per point of `LAMBDA_RATIO_GRID`, in grid order, per scenario and
# shape.  The rev4 audit's escapes Z1 (`MULT["B"]` 1.70 -> 1.16) and Z4
# (`lambda0_analyze.DRAIN_MODEL_TOL` 2.0 -> 3.0) each MOVE exactly one label on
# this map, and the old `assert seen[s][verdict]` was blind to them: a pooled
# non-emptiness test cannot see a point change from one non-empty verdict to
# another.  What this registration publishes is the whole map, so the whole map
# is what is asserted.  Any decision-path change that moves a single point now
# fails here and must be re-registered.
#
# ★Read `LADDER_TOO_LOW[A]` in the FALLBACK row: there is NO "L" there.  The
# scenario that actually runs after F1 cannot emit that verdict for shape A
# under this model (audit V4 / R4C-5) -- the pooled census hides it, this table
# does not.
REGISTERED_MAP = {
    "MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)":
        ("HHNBBBBBBBBL", "HHHHNBBBBBLL"),
    "point estimate lower  (lambda*(A)=2.1 regression extrapolation)":
        ("HHHBBBBBBBBL", "HHHHNBBBBBLL"),
    "point estimate upper  (lambda*(A)=2.5 ctx-768 KV-term variant)":
        ("HHHBBBBBBBBL", "HHHHNBBBBBLL"),
    "absolute lower bound  (A=1.08 worst observed step, B=0.35)":
        ("HHHBBBBBBBBL", "HHHHNBBBBBLL"),
    "absolute upper bound  (A=7.15 step(B=1) floor, B=1.20)":
        ("HHBBBBBBBBBL", "HHHHNBBBBBLL"),
    "B=6 plateau variant   (A=3.91, B=1.00)":
        ("HHNBBBBBBBBL", "HHHHNBBBBBLL"),
    "FALLBACK literal ladders":
        ("HHHHNBBBBBBB", "HHHHHNBBBBLL"),
}


def table() -> tuple:
    """The registered reachability map + the per-verdict domain census.

    Returns `(text, census, codes)`; `codes[title] = (shape-A string, shape-B
    string)` is the compact form `REGISTERED_MAP` pins (F5-1).
    """
    lines, seen, codes = [], {"A": {}, "B": {}}, {}
    with tempfile.TemporaryDirectory() as td:
        for title, la, lb in SCENARIOS:
            lad = ({k: tuple(v) for k, v in PLAN.FALLBACK_LADDER.items()}
                   if title.startswith("FALLBACK") else None)
            lines.append(f"### {title}   lambda_inf=({la}, {lb})"
                         + ("  [FALLBACK ladders]" if lad else ""))
            lines.append("    lam*/lam_inf |            shape A            |"
                         "            shape B")
            row = {"A": "", "B": ""}
            for ratio in LAMBDA_RATIO_GRID:
                r = emit(la, lb, ratio, ladders=lad, tmpdir=td)
                cells = []
                for s in ("A", "B"):
                    row[s] += VERDICT_CODE[r[s]["verdict"]]
                    d = r[s]
                    err = ("    --  " if d["rel_err"] is None
                           else f"{100*d['rel_err']:+6.1f}%")
                    cells.append(f"{d['verdict']:<18s} lam*={d['true_lambda_star']:6.3f}"
                                 f" err={err} low={d['n_usable_low_side']}/"
                                 f"{d['n_low_side_candidates']}")
                    seen[s].setdefault(d["verdict"], []).append(
                        (title, round(ratio, 2)))
                lines.append(f"    {ratio:12.2f} | {cells[0]} | {cells[1]}")
            codes[title] = (row["A"], row["B"])
            lines.append("")
    return "\n".join(lines), seen, codes


def selftest() -> None:
    txt, seen, codes = table()
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

    # ★rev5/F5-1: and the WHOLE MAP must be the registered one.  The assert
    # above is POOLED over scenarios and blind to a label MOVING between two
    # non-empty verdicts -- which is exactly what the rev4 audit's Z1 and Z4 do
    # (one point each).  `REGISTERED_MAP` is published in the pre-registration,
    # so a decision-path change that moves any point fails here.
    assert set(codes) == set(REGISTERED_MAP), (
        "the scenario set changed; the registered map no longer covers it",
        sorted(set(codes) ^ set(REGISTERED_MAP)))
    for title, got in codes.items():
        want = REGISTERED_MAP[title]
        assert got == want, (
            "the reachability map MOVED and is no longer the registered one",
            title, "registered A/B", want, "got A/B", got,
            "grid", LAMBDA_RATIO_GRID)
    assert all(len(c) == len(LAMBDA_RATIO_GRID) for cc in codes.values()
               for c in cc), codes

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
          "verdicts have non-empty domains for both shapes; ★F5-1 all %d map "
          "points equal REGISTERED_MAP, so a moved label is a failure and not "
          "merely a non-empty census; bracket "
          "reports lambda* within 15%%; NEGATIVE CONTROL: the rev3 rule still "
          "returns UNRESOLVED for shape B at the anchor)"
          % (len(LAMBDA_RATIO_GRID), len(SCENARIOS),
             2 * len(SCENARIOS) * len(LAMBDA_RATIO_GRID)))


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return 0
    if len(sys.argv) != 1:
        print(__doc__)
        return 2
    txt, seen, codes = table()
    print(txt)
    print("### verdict domain census (must be non-empty for each registered verdict)")
    for s in ("A", "B"):
        for v, where in sorted(seen[s].items()):
            print(f"    shape {s}  {v:<20s} {len(where):3d} grid points  "
                  f"e.g. {where[0]}")
    print("")
    print("### ★the REGISTERED MAP (F5-1), one code per lambda*/lambda_inf grid "
          "point, in grid order")
    print("###   B=KNEE_BRACKETED  N=KNEE_NOT_BRACKETED  H=LADDER_TOO_HIGH  "
          "L=LADDER_TOO_LOW  U=UNRESOLVED")
    print("###   grid = " + " ".join(f"{r:g}" for r in LAMBDA_RATIO_GRID))
    for title, _, _ in SCENARIOS:
        a, b = codes[title]
        flag = "  <== matches REGISTERED_MAP" \
            if REGISTERED_MAP.get(title) == (a, b) else "  <== ★DRIFTED"
        print(f"    A {a}   B {b}   {title}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
