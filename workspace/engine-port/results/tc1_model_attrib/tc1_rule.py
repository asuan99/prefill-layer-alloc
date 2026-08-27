#!/usr/bin/env python3
"""TC1 (model attribution) -- THE REGISTERED DECISION RULE, fixed in code.

RULE-AS-CODE citation note: stage0ppp_a0_rule.py attributes this discipline to
"gate #66".  VERIFIED 2026-08-27 against PROJECT_STATUS "방법론 게이트": #66 is
"포크는 ..." (S-6 rev1-rev4, 2026-08-23) -- a DIFFERENT gate.  The rules-as-code
discipline is 교훈 항목66 in CONSENSUS §3, which is not gate #66.  This file
therefore cites the discipline BY NAME and carries no gate number.

RULE_REV = 1.  Rule layer only; gate #34 stage 1.  NOT AUDITED.

WHAT THIS FILE IS NOT: not a measurement, not a harness, not evidence.  It fixes
the label of every reachable world BEFORE any TC1 job is submitted, so that no
choice remains to be made after seeing data (the failure mode named in the NSL
rev2 audit as G1: "MATTERS and INERT are not disjoint and the priority is
unregistered, so the author gets to choose once the data are in").

THE QUESTION.  Does the sign of (dynamic - best-static) differ between a hybrid
(Zamba2-2.7B) and a size-matched pure Transformer (Qwen2.5-3B) on the SAME
green context, the SAME conjunctive-SLO goodput, and the SAME ShareGPT varying
trace?  A difference in sign attributes the project's central negative to the
MODEL.  No difference re-frames it as a property of the metric plus the
deployment primitive.

Run:  python3 tc1_rule.py            # full self-test, writes tc1_rule_rev1.json
"""
import hashlib, json, math, os, sys
from itertools import product

RULE_REV = 1

# ── registered labels ────────────────────────────────────────────────────────
ABSENT   = "MEASUREMENT_ABSENT"          # a boot / bench leg did not produce data
NOBAND   = "NO_COMMON_BAND"              # no off-cliff rate pair serves both models
CLIFF    = "CLIFF_STRADDLE"              # TTFT p90 straddles the SLO -> goodput ill-posed
GRIDEDGE = "GRID_EDGE_UNRESOLVED"        # a model's static argmax sits at a grid endpoint
DEGEN    = "CONTROLLER_DEGENERATE"       # the controller could not move on some model
SCOREDEP = "SCORING_DEPENDENT"           # p95 and mean scorings disagree on the verdict
TIE      = "INCONCLUSIVE_TIE"            # a per-model verdict is neither win nor lose
UNDERPWR = "UNDERPOWERED_NULL"           # a load-bearing null lacks power at delta
FLIP     = "FLIP_MODEL_ATTRIBUTED"       # T wins, H loses  -> branch A
REVFLIP  = "REVERSE_FLIP"                # H wins, T loses  -> model-attributed, opposite sign
BOTHWIN  = "NO_FLIP_BOTH_WIN"            # both win -> the central negative itself is shaken
BOTHLOSE = "NO_FLIP_BOTH_LOSE"           # neither wins -> branch B

MEASUREMENT = {ABSENT, NOBAND, CLIFF}          # gate #21: NOT gate failures
DESIGN      = {GRIDEDGE, DEGEN}                # reachable-in-principle, blocked by design
SCORING     = {SCOREDEP}
SUBSTANTIVE = {TIE, UNDERPWR, FLIP, REVFLIP, BOTHWIN, BOTHLOSE}
LABELS      = MEASUREMENT | DESIGN | SCORING | SUBSTANTIVE

# ── registered constants ─────────────────────────────────────────────────────
N_REPS       = 4        # CLAUDE.md 게이트 #3 (n>=4)
DELTA_REL    = 0.03     # CLAUDE.md 게이트 #3 -- under 3% is not a headline
ALPHA        = 0.05     # two-sided
T_CRIT_N4    = 3.182    # t_{.975, 3}; gate #14 (통계 방법 층 정정) -- t-CI, NOT
                        # percentile bootstrap: n=4 coverage is 0.798, one-sided ~10%
POWER_TARGET = 0.80
# POWER_ADEQUATE iff the 80%-power MDE is within this multiple of delta.
MDE_SLACK    = 2.0
# Design-time prior only.  Stage 1 REPLACES it with the measured per-model SD.
SD_PRIOR     = {"H": 0.023, "T": None}   # H from canon d44 +-0.013 / bind+GATE +-0.019
GOODPUT_BASE = {"H": 3.220, "T": None}   # H from CONSENSUS s1-7; T unmeasured by design

# ── world axes (exhaustive enumeration -- rules-as-code discipline) ──────────────────────────
AXES = {
    "boot":     ("ok", "failed"),
    "band":     ("exists", "empty"),
    "cliff":    ("clean", "straddle"),
    "grid_H":   ("interior", "edge_lo", "edge_hi"),
    "grid_T":   ("interior", "edge_lo", "edge_hi"),
    "deg_H":    ("ok", "degenerate"),
    "deg_T":    ("ok", "degenerate"),
    "scoring":  ("agree", "disagree"),
    "sign_H":   ("win", "lose", "tie"),
    "sign_T":   ("win", "lose", "tie"),
    "power_H":  ("adequate", "inadequate"),
    "power_T":  ("adequate", "inadequate"),
}
AXIS_ORDER = tuple(AXES)


def worlds():
    for combo in product(*(AXES[a] for a in AXIS_ORDER)):
        yield dict(zip(AXIS_ORDER, combo))


# ── THE RULE ─────────────────────────────────────────────────────────────────
# The guard ORDER is itself registered (A0 rev4 finding N1: an unguarded branch
# order moved thousands of labels).  M5 below measures how much it carries.
GUARD_ORDER = ("boot", "band", "cliff", "grid", "degen", "scoring")


def label(w, skip=(), order=GUARD_ORDER):
    for g in order:
        if g in skip:
            continue
        if g == "boot"  and w["boot"]  == "failed":    return ABSENT
        if g == "band"  and w["band"]  == "empty":     return NOBAND
        if g == "cliff" and w["cliff"] == "straddle":  return CLIFF
        if g == "grid"  and (w["grid_H"] != "interior" or w["grid_T"] != "interior"):
            return GRIDEDGE
        if g == "degen" and (w["deg_H"] == "degenerate" or w["deg_T"] == "degenerate"):
            return DEGEN
        if g == "scoring" and w["scoring"] == "disagree": return SCOREDEP

    sH, sT = w["sign_H"], w["sign_T"]
    if sH == "tie" or sT == "tie":
        return TIE
    # Power gates ONLY the labels whose reading rests on a null.  A win is a
    # positive finding and needs no power argument; the "lose" half of a flip
    # IS a null and does.
    if sT == "win" and sH == "lose":
        return FLIP if ("power" in skip or w["power_H"] == "adequate") else UNDERPWR
    if sT == "lose" and sH == "win":
        return REVFLIP if ("power" in skip or w["power_T"] == "adequate") else UNDERPWR
    if sT == "win" and sH == "win":
        return BOTHWIN
    # both lose: two nulls, both load-bearing
    if "power" in skip or (w["power_H"] == "adequate" and w["power_T"] == "adequate"):
        return BOTHLOSE
    return UNDERPWR


# ── registered power arithmetic (design-time; the analyzer runs the full t-test)
def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _z_at(p):                      # inverse normal, Acklam-free bisection (stdlib only)
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _phi(mid) < p: lo = mid
        else:             hi = mid
    return (lo + hi) / 2.0


def win_threshold(sd, n=N_REPS, delta_abs=None, t_crit=T_CRIT_N4):
    """A win fires iff  d_hat >= delta_abs  AND  the t-CI excludes 0.
    Both are thresholds on d_hat, so the binding one is their max."""
    se = sd / math.sqrt(n)
    return max(delta_abs, t_crit * se), se


def p_win(true_delta, sd, delta_abs, n=N_REPS):
    thr, se = win_threshold(sd, n, delta_abs)
    return 1.0 - _phi((thr - true_delta) / se)


def mde(sd, delta_abs, n=N_REPS, power=POWER_TARGET):
    thr, se = win_threshold(sd, n, delta_abs)
    return thr - _z_at(1.0 - power) * se


def power_adequate(sd, base_goodput, n=N_REPS):
    """REGISTERED: power is adequate iff the 80%-power MDE is within
    MDE_SLACK x delta.  Stage 1 supplies sd and base_goodput per model."""
    delta_abs = DELTA_REL * base_goodput
    return mde(sd, delta_abs, n) <= MDE_SLACK * delta_abs


# ── self-tests ───────────────────────────────────────────────────────────────
def run_tests():
    W = list(worlds())
    L = [label(w) for w in W]
    res, fail = {}, []

    def check(name, ok, detail=""):
        res[name] = {"pass": bool(ok), "detail": detail}
        if not ok: fail.append(name)

    # T1 totality: every world gets exactly one registered label
    check("T1_totality", all(x in LABELS for x in L), f"{len(W)} worlds")

    # T2 reachability: every registered label is reachable
    marg = {k: L.count(k) for k in sorted(LABELS)}
    unreach = [k for k, v in marg.items() if v == 0]
    check("T2_reachable", not unreach, f"unreachable={unreach}")

    # T3 measurement labels never carry a substantive reading
    check("T3_measurement_is_terminal",
          all(label(w) in MEASUREMENT for w in W
              if w["boot"] == "failed" or w["band"] == "empty" or w["cliff"] == "straddle"),
          "gate #21 (측정 실패를 게이트 실패로 라벨링 마라)")

    # T4 FLIP exclusivity, BOTH directions
    fwd = all(w["sign_T"] == "win" and w["sign_H"] == "lose"
              for w, x in zip(W, L) if x == FLIP)
    rev_ok = any(x == FLIP for w, x in zip(W, L)
                 if w["sign_T"] == "win" and w["sign_H"] == "lose")
    check("T4_flip_exclusive", fwd and rev_ok)

    # T5 NEGATIVE CONTROL: identical per-model signs can never read as a flip
    nc = [x for w, x in zip(W, L) if w["sign_H"] == w["sign_T"]]
    check("T5_negative_control", FLIP not in nc and REVFLIP not in nc,
          f"{len(nc)} same-sign worlds, none attributed")

    # T6 power gates EXACTLY the null-bearing labels -- discriminating in both
    # directions.  (rev1 first wrote this as `bool(p_bad) or True`, i.e. an
    # identity -- the gate-#9 family, caught by the author before submission.)
    def combos(lbl):
        return {(w["power_H"], w["power_T"]) for w, x in zip(W, L) if x == lbl}
    all4 = {("adequate", "adequate"), ("adequate", "inadequate"),
            ("inadequate", "adequate"), ("inadequate", "inadequate")}
    t6a = combos(BOTHWIN) == all4                       # a win needs no power argument
    t6b = combos(FLIP) == {("adequate", "adequate"), ("adequate", "inadequate")}
    t6c = combos(REVFLIP) == {("adequate", "adequate"), ("inadequate", "adequate")}
    t6d = combos(BOTHLOSE) == {("adequate", "adequate")}
    check("T6_power_gates_nulls_only", t6a and t6b and t6c and t6d,
          f"BOTHWIN ungated({len(combos(BOTHWIN))}/4) | FLIP gated by power_H only "
          f"| REVFLIP by power_T only | BOTHLOSE by both")

    # M1-M4 hollowness: each guard must actually move labels when removed
    for g, nm in (("grid", "M1_grid_not_hollow"), ("degen", "M2_degen_not_hollow"),
                  ("scoring", "M3_scoring_not_hollow"), ("power", "M4_power_not_hollow")):
        moved = sum(1 for w, x in zip(W, L) if label(w, skip=(g,)) != x)
        check(nm, moved > 0, f"{moved} worlds change")

    # M5 the guard ORDER is a real degree of freedom -> it is registered, not free
    swapped = ("boot", "band", "cliff", "degen", "grid", "scoring")
    moved = sum(1 for w, x in zip(W, L) if label(w, order=swapped) != x)
    check("M5_order_is_load_bearing", moved > 0,
          f"grid<->degen swap moves {moved} labels; order is REGISTERED as {GUARD_ORDER}")

    # T7 substantive labels are separated, and the separating DISTANCE is itself
    # registered.  rev1 first demanded single-axis adjacency; that is the wrong
    # proxy -- FLIP and REVERSE_FLIP differ by TWO coordinated sign changes,
    # which is exactly what "a flip" means.  The meaningful assertion is that
    # the distance matrix matches the rule's intent.
    sub = sorted(SUBSTANTIVE)
    reps = {}
    for w, x in zip(W, L):
        if x in SUBSTANTIVE:
            reps.setdefault(x, []).append(tuple(w[a] for a in AXIS_ORDER))

    def dmin(a, b):
        return min(sum(1 for i in range(len(AXIS_ORDER)) if u[i] != v[i])
                   for u in reps[a] for v in reps[b])

    dist = {a: {b: dmin(a, b) for b in sub if b != a} for a in sub}
    check("T7_all_substantive_separated",
          all(dist[a][b] >= 1 for a in sub for b in dist[a]),
          "distinct labels never share a world")
    check("T7b_flip_needs_two_signs",
          dist[FLIP][REVFLIP] == 2 and dist[BOTHWIN][BOTHLOSE] == 2,
          f"d(FLIP,REVFLIP)={dist[FLIP][REVFLIP]}  d(BOTHWIN,BOTHLOSE)={dist[BOTHWIN][BOTHLOSE]}"
          " -- a flip is a JOINT move of both model signs, not one")
    check("T7c_flip_one_sign_from_bothwin",
          dist[FLIP][BOTHWIN] == 1 and dist[REVFLIP][BOTHLOSE] == 1,
          "a single per-model sign separates a flip from a no-flip")

    # T11 no inert axis: every registered axis must move at least one label
    inert = []
    idx = {tuple(w[a] for a in AXIS_ORDER): x for w, x in zip(W, L)}
    for i, a in enumerate(AXIS_ORDER):
        moved = False
        for w, x in zip(W, L):
            key = tuple(w[ax] for ax in AXIS_ORDER)
            for v in AXES[a]:
                if v == w[a]:
                    continue
                if idx[key[:i] + (v,) + key[i + 1:]] != x:
                    moved = True
                    break
            if moved:
                break
        if not moved:
            inert.append(a)
    check("T11_no_inert_axis", not inert, f"inert axes={inert}")

    # T8 design blockers dominate scoring disagreement (registered precedence)
    check("T8_design_before_scoring",
          all(label(w) in DESIGN for w in W
              if w["boot"] == "ok" and w["band"] == "exists" and w["cliff"] == "clean"
              and (w["grid_H"] != "interior" or w["deg_H"] == "degenerate")))

    # T9 design-time power arithmetic on the ONLY model with a measured prior
    sdH, baseH = SD_PRIOR["H"], GOODPUT_BASE["H"]
    dH = DELTA_REL * baseH
    thrH, seH = win_threshold(sdH, delta_abs=dH)
    p0 = p_win(0.0, sdH, dH)
    m80 = mde(sdH, dH)
    check("T9_null_rate_bounded", p0 <= 0.01,
          f"P(win | delta=0) = {p0:.2e}; delta binds ({dH:.4f}) over t.se ({T_CRIT_N4*seH:.4f})")
    check("T10_H_power_adequate", power_adequate(sdH, baseH),
          f"MDE(80%) = {m80:.4f} = {m80/baseH*100:.2f}% vs delta {dH:.4f}")

    out = {
        "rule_rev": RULE_REV,
        "n_worlds": len(W),
        "axes": {k: list(v) for k, v in AXES.items()},
        "guard_order": list(GUARD_ORDER),
        "labels": {"MEASUREMENT": sorted(MEASUREMENT), "DESIGN": sorted(DESIGN),
                   "SCORING": sorted(SCORING), "SUBSTANTIVE": sorted(SUBSTANTIVE)},
        "marginals": marg,
        "separating_distance": dist,
        "constants": {"N_REPS": N_REPS, "DELTA_REL": DELTA_REL, "ALPHA": ALPHA,
                      "T_CRIT_N4": T_CRIT_N4, "POWER_TARGET": POWER_TARGET,
                      "MDE_SLACK": MDE_SLACK, "SD_PRIOR": SD_PRIOR,
                      "GOODPUT_BASE": GOODPUT_BASE},
        "design_power_H": {"sd": sdH, "delta_abs": dH, "win_threshold": thrH,
                           "se": seH, "p_win_at_null": p0, "mde_80": m80},
        "tests": res,
        "all_pass": not fail,
        "failed": fail,
    }
    return out


if __name__ == "__main__":
    out = run_tests()
    src = open(os.path.abspath(__file__), "rb").read()
    out["rule_sha256"] = hashlib.sha256(src).hexdigest()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tc1_rule_rev1.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"RULE_REV={RULE_REV}  worlds={out['n_worlds']}  sha256={out['rule_sha256'][:16]}")
    for k, v in out["tests"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v['detail']}")
    print("  marginals:", json.dumps(out["marginals"], ensure_ascii=False))
    print("ALL PASS" if out["all_pass"] else f"FAILED: {out['failed']}")
    sys.exit(0 if out["all_pass"] else 1)
