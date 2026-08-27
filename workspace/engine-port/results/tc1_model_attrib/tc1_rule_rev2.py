#!/usr/bin/env python3
"""TC1 rev2 -- THE REGISTERED DECISION RULE, fixed in code.  NOT AUDITED.

RULE_REV = 2.  rev1 was audited NO-GO (audit_tc1_rules_2026-08-27, F1-F4 fatal,
B1-B18 blocking).  This revision answers every one of them; the map from finding
to change is in PREREG_TC1_RULES_REV2_2026-08-27.md sec 1.

RULES-AS-CODE citation: the discipline is CONSENSUS sec3 item 81 = PROJECT_STATUS
"방법론 게이트" #61.  rev1 wrote "item 66" and stamped it VERIFIED; item 66 is
"사다리 해상도" (G16, 2026-08-17).  Verified 2026-08-27 by direct comparison and
reported in the prereg sec 0.  There are FOUR numbering systems in this repo, not
two: CLAUDE.md #1-8 / PROJECT_STATUS #1-80 / CONSENSUS sec3 #1-100 / memory topic
#1-80.  Every citation below names the gate as well as numbering it.

WHAT CHANGED (fatal findings only; blockers in the prereg):
  F1  sticky is no longer required.  It was a category error: TC1's estimand is
      end-to-end goodput and does not condition on the partition, and the engine
      forbids sticky+SLO_SCHED outright (multiplexing_mixin.py:298-303).
      Realized residency becomes a COMPARABILITY covariate with its own axis.
  F2  the anchor is a model-independent constant, decoupled from Stage 1, and a
      new OBSERVABLE axis `visits_argmax` decides reachability instead of the
      uninstrumentable predicate rev1 registered.
  F3  the estimand is named: COMBINED goodput, durations summed.
  F4  "the sign is invariant to monotone rescaling" is RETRACTED -- goodput is a
      threshold indicator and the threshold is applied AFTER rescaling.  A
      `discrim` axis now requires the predicate to discriminate in both arms.

Run:  python3 tc1_rule_rev2.py
"""
import hashlib, json, math, os, sys
from itertools import product

RULE_REV = 2

# ── labels ───────────────────────────────────────────────────────────────────
BOOTF    = "BOOT_FAILED"
CAPUN    = "CAPACITY_UNMEASURABLE"
CLIFF    = "CLIFF_STRADDLE"
REALMM   = "REALIZED_MISMATCH"          # F1/B1: arms did not share a substrate
TOKSKEW  = "TOKENIZER_SKEW"             # B16: same text, too-different token loads
DEGEN    = "CONTROLLER_DEGENERATE"      # F2/B9: observable now
NODISC   = "NOT_DISCRIMINATING"         # F4: predicate is all-pass or all-fail
GRIDEDGE = "GRID_EDGE_UNRESOLVED"
SCOREDEP = "SCORING_DEPENDENT"
FLIP     = "FLIP_MODEL_ATTRIBUTED"
REVFLIP  = "REVERSE_FLIP"
BOTHWIN  = "NO_FLIP_BOTH_WIN"
BOTHLOSE = "NO_FLIP_BOTH_LOSE"
UNDERPWR = "UNDERPOWERED_NULL"
INCONC   = "INCONCLUSIVE"

MEASUREMENT   = {BOOTF, CAPUN, CLIFF}
COMPARABILITY = {REALMM, TOKSKEW}
DESIGN        = {DEGEN, NODISC, GRIDEDGE}
SCORING       = {SCOREDEP}
SUBSTANTIVE   = {FLIP, REVFLIP, BOTHWIN, BOTHLOSE, UNDERPWR, INCONC}
LABELS = MEASUREMENT | COMPARABILITY | DESIGN | SCORING | SUBSTANTIVE

# ── constants ────────────────────────────────────────────────────────────────
N_REPS      = 4          # CLAUDE.md 게이트 #3 (n>=4)
DELTA_REL   = 0.03       # CLAUDE.md 게이트 #3 (under 3% is not a headline)
ALPHA       = 0.05
POWER_TARGET = 0.80
# B7: MDE_SLACK=2.0 was an ungrounded constant.  Replaced by a DERIVED criterion --
# the design must detect an effect of 2*delta ("clearly material") at 80% power.
POWER_AT    = 2.0        # multiples of delta the design must be able to detect
# B8: arms are NOT paired (separate boots).  Welch two-sample; the critical value
# is computed from the realised df, never hardcoded.
PAIRED      = False
REALIZED_MAX_GAP = 0.15  # F1/B1: |residency_H - residency_T| above this blocks
TOKEN_MAX_GAP    = 0.20  # B16: |median input_len ratio - 1| above this blocks
ANCHOR_IDX_SOURCE = "canonical constant, model-independent (sharegpt_vary_bench.sbatch:40)"
ESTIMAND    = "COMBINED goodput, round durations SUMMED (CLAUDE.md 게이트 #7)"
SCORING_PRIMARY = "p95 in-request ITL (CLAUDE.md 게이트 #4); mean kept as bridge only"
SIGN_SCORING    = "p95"  # B12: which scoring sign_* refers to

# ── axes ─────────────────────────────────────────────────────────────────────
AXES = {
    "boot_H":   ("ok", "failed"),          # B2: per-model, not global
    "boot_T":   ("ok", "failed"),
    "cliff":    ("clean", "straddle", "capacity_unmeasurable"),   # B13 absorbs `band`
    "compar":   ("ok", "realized_mismatch", "tokenizer_skew"),    # F1/B1/B16
    "discrim_H":("ok", "degenerate"),      # F4
    "discrim_T":("ok", "degenerate"),
    "grid_H":   ("interior", "edge"),
    "grid_T":   ("interior", "edge"),
    "visits_H": ("yes", "no"),             # F2/B9: observable from telemetry stream_index
    "visits_T": ("yes", "no"),
    "scoring_H":("agree", "disagree"),     # B12: per-model
    "scoring_T":("agree", "disagree"),
    "sign_H":   ("win", "lose_sig", "equiv", "undecided"),   # B6: lose != not-win
    "sign_T":   ("win", "lose_sig", "equiv", "undecided"),
    "power_H":  ("adequate", "inadequate"),
    "power_T":  ("adequate", "inadequate"),
}
AXIS_ORDER = tuple(AXES)
IX = {a: i for i, a in enumerate(AXIS_ORDER)}
GUARD_ORDER = ("boot", "cliff", "compar", "discrim", "grid", "visits", "scoring")
NEGATIVE = {"lose_sig", "equiv"}          # both are "dynamic did not win", with evidence


def worlds():
    return product(*(AXES[a] for a in AXIS_ORDER))


def label(w, skip=(), order=GUARD_ORDER):
    g = lambda a: w[IX[a]]
    for gd in order:
        if gd in skip:
            continue
        if gd == "boot" and (g("boot_H") == "failed" or g("boot_T") == "failed"):
            return BOOTF
        if gd == "cliff":
            if g("cliff") == "capacity_unmeasurable": return CAPUN
            if g("cliff") == "straddle":              return CLIFF
        if gd == "compar":
            if g("compar") == "realized_mismatch": return REALMM
            if g("compar") == "tokenizer_skew":    return TOKSKEW
        if gd == "discrim" and (g("discrim_H") == "degenerate" or g("discrim_T") == "degenerate"):
            return NODISC
        if gd == "grid" and (g("grid_H") == "edge" or g("grid_T") == "edge"):
            return GRIDEDGE
        if gd == "visits" and (g("visits_H") == "no" or g("visits_T") == "no"):
            return DEGEN
        if gd == "scoring" and (g("scoring_H") == "disagree" or g("scoring_T") == "disagree"):
            return SCOREDEP
    sH, sT, pH, pT = g("sign_H"), g("sign_T"), g("power_H"), g("power_T")
    if sH == "undecided" or sT == "undecided":
        return INCONC
    free = "power" in skip
    if sT == "win" and sH in NEGATIVE:
        return FLIP if (free or pH == "adequate") else UNDERPWR
    if sH == "win" and sT in NEGATIVE:
        return REVFLIP if (free or pT == "adequate") else UNDERPWR
    if sT == "win" and sH == "win":
        return BOTHWIN
    if free or (pH == "adequate" and pT == "adequate"):
        return BOTHLOSE
    return UNDERPWR


# ── power arithmetic (design time; the analyser runs the full Welch test) ─────
def _phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def _z(p):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        m = (lo + hi) / 2.0
        if _phi(m) < p: lo = m
        else:           hi = m
    return (lo + hi) / 2.0

def t_crit(df, alpha=ALPHA):
    """Two-sided critical value.  B8: derived from the realised df, not hardcoded."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131}
    if df in table: return table[df]
    k = max(k for k in table if k <= df) if df >= 1 else 1
    return table[k]

def win_threshold(sd_diff, n=N_REPS, delta_abs=0.0, df=None):
    se = sd_diff / math.sqrt(n)
    df = df if df is not None else (2 * n - 2)
    return max(delta_abs, t_crit(df) * se), se

def p_win(true_delta, sd_diff, delta_abs, n=N_REPS, df=None):
    thr, se = win_threshold(sd_diff, n, delta_abs, df)
    return 1.0 - _phi((thr - true_delta) / se)

def power_adequate(sd_diff, base, n=N_REPS, df=None):
    """REGISTERED (B7): adequate iff an effect of POWER_AT * delta is detected
    with probability >= POWER_TARGET.  Derived from the decision threshold, not
    an ungrounded slack constant."""
    d = DELTA_REL * base
    return p_win(POWER_AT * d, sd_diff, d, n, df) >= POWER_TARGET


# ── named mutants (B14): each must be killed by a NAMED discriminating test ───
MUTANTS = {
    "drop_visits_guard":   dict(skip=("visits",)),
    "drop_discrim_guard":  dict(skip=("discrim",)),
    "drop_compar_guard":   dict(skip=("compar",)),
    "drop_grid_guard":     dict(skip=("grid",)),
    "drop_scoring_guard":  dict(skip=("scoring",)),
    "drop_boot_guard":     dict(skip=("boot",)),
    "drop_cliff_guard":    dict(skip=("cliff",)),
    "drop_power_gate":     dict(skip=("power",)),
    "reorder_grid_visits": dict(order=("boot","cliff","compar","discrim","visits","grid","scoring")),
}


def run_tests():
    W = list(worlds()); L = [label(w) for w in W]
    res, fail = {}, []
    def check(n, ok, d=""):
        res[n] = {"pass": bool(ok), "detail": d}
        if not ok: fail.append(n)

    check("T1_totality", all(x in LABELS for x in L), f"{len(W):,} worlds")
    marg = {k: L.count(k) for k in sorted(LABELS)}
    check("T2_reachable_in_grid", all(v > 0 for v in marg.values()),
          f"unreachable={[k for k,v in marg.items() if v==0]}")

    # B6: a negative verdict requires EVIDENCE, never mere absence of a win
    check("T3_negative_needs_evidence",
          all(w[IX["sign_H"]] in NEGATIVE and w[IX["sign_T"]] in NEGATIVE
              for w, x in zip(W, L) if x == BOTHLOSE),
          "BOTHLOSE requires lose_sig or equiv in BOTH arms, not 'not win'")
    check("T4_undecided_is_its_own_label",
          all(x == INCONC for w, x in zip(W, L)
              if (w[IX["sign_H"]] == "undecided" or w[IX["sign_T"]] == "undecided")
              and label(w, skip=GUARD_ORDER) == x),
          "an undecided arm can never be read as a flip or a null")

    # B5: UNDERPOWERED_NULL must be reachable and must not overlap INCONCLUSIVE
    up = [w for w, x in zip(W, L) if x == UNDERPWR]
    check("T5_underpowered_reachable_and_disjoint",
          len(up) > 0 and all(w[IX["sign_H"]] != "undecided" and w[IX["sign_T"]] != "undecided"
                              for w in up),
          f"{len(up):,} worlds; never overlaps an undecided arm")

    # F2/B9: the observable degeneracy axis actually blocks
    check("T6_visits_blocks",
          all(label(w) == DEGEN for w in W
              if w[IX["boot_H"]]=="ok" and w[IX["boot_T"]]=="ok" and w[IX["cliff"]]=="clean"
              and w[IX["compar"]]=="ok" and w[IX["discrim_H"]]=="ok" and w[IX["discrim_T"]]=="ok"
              and w[IX["grid_H"]]=="interior" and w[IX["grid_T"]]=="interior"
              and (w[IX["visits_H"]]=="no" or w[IX["visits_T"]]=="no")))

    # B14: mutant coverage -- every mutant killed by a NAMED test, none uncovered
    killed, uncovered = {}, []
    for name, kw in MUTANTS.items():
        n = sum(1 for w, x in zip(W, L) if label(w, **kw) != x)
        killed[name] = n
        if n == 0: uncovered.append(name)
    check("T7_mutant_coverage", not uncovered, f"uncovered_mutants={uncovered}")
    check("T8_every_guard_has_a_mutant",
          all(f"drop_{g}_guard" in MUTANTS for g in GUARD_ORDER),
          "each registered guard has a named mutant")

    # B15: state the order effect honestly -- it renames blocks, it does not
    # change whether a SUBSTANTIVE verdict is reached
    reord = MUTANTS["reorder_grid_visits"]
    same_reach = all((label(w, **reord) in SUBSTANTIVE) == (x in SUBSTANTIVE)
                     for w, x in zip(W, L))
    check("T9_order_renames_only",
          same_reach and killed["reorder_grid_visits"] > 0,
          f"reorder moves {killed['reorder_grid_visits']:,} labels but never changes "
          "whether a substantive verdict is reached")

    # B12: the scoring the sign refers to is registered
    check("T10_sign_scoring_registered", SIGN_SCORING == "p95",
          f"sign_* refers to {SIGN_SCORING}; disagreement blocks via SCORING_DEPENDENT")

    # B7: the derived power criterion, on the only arm with a prior
    sdH, baseH = 0.023, 3.220
    dH = DELTA_REL * baseH
    thr, se = win_threshold(sdH, delta_abs=dH)
    check("T11_null_rate_bounded", p_win(0.0, sdH, dH) <= 0.01,
          f"P(win|0)={p_win(0.0, sdH, dH):.2e}; thr={thr:.4f} se={se:.4f} df={2*N_REPS-2}")
    check("T12_power_criterion_derived", power_adequate(sdH, baseH),
          f"P(win|2delta)={p_win(2*dH, sdH, dH):.3f} >= {POWER_TARGET}")
    # and it must FAIL for a noisy arm -- the criterion has teeth
    check("T13_power_criterion_has_teeth", not power_adequate(0.03 * 3.220 * 2, baseH),
          "a CV-6% arm is correctly refused")

    out = {"rule_rev": RULE_REV, "n_worlds": len(W),
           "axes": {k: list(v) for k, v in AXES.items()},
           "guard_order": list(GUARD_ORDER),
           "labels": {"MEASUREMENT": sorted(MEASUREMENT), "COMPARABILITY": sorted(COMPARABILITY),
                      "DESIGN": sorted(DESIGN), "SCORING": sorted(SCORING),
                      "SUBSTANTIVE": sorted(SUBSTANTIVE)},
           "marginals": marg, "mutants_killed": killed,
           "constants": {"N_REPS": N_REPS, "DELTA_REL": DELTA_REL, "ALPHA": ALPHA,
                         "POWER_TARGET": POWER_TARGET, "POWER_AT": POWER_AT,
                         "PAIRED": PAIRED, "REALIZED_MAX_GAP": REALIZED_MAX_GAP,
                         "TOKEN_MAX_GAP": TOKEN_MAX_GAP, "ESTIMAND": ESTIMAND,
                         "SCORING_PRIMARY": SCORING_PRIMARY, "SIGN_SCORING": SIGN_SCORING,
                         "ANCHOR_IDX_SOURCE": ANCHOR_IDX_SOURCE},
           "tests": res, "all_pass": not fail, "failed": fail}
    return out


if __name__ == "__main__":
    out = run_tests()
    out["rule_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "tc1_rule_rev2.json"), "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"RULE_REV={RULE_REV} worlds={out['n_worlds']:,} sha={out['rule_sha256'][:16]}")
    for k, v in out["tests"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v['detail']}")
    print("  marginals:", json.dumps(out["marginals"], ensure_ascii=False))
    print("  mutants  :", json.dumps(out["mutants_killed"]))
    print("ALL PASS" if out["all_pass"] else f"FAILED: {out['failed']}")
    sys.exit(0 if out["all_pass"] else 1)
