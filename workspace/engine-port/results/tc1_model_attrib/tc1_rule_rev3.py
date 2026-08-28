#!/usr/bin/env python3
"""TC1 rev3 -- THE REGISTERED DECISION RULE, fixed in code.  NOT AUDITED.

RULE_REV = 3.  rev2 was audited NO-GO (F5-F7 fatal, B19-B30 blocking) and then
FAILED ITS OWN design-layer reachability check: under the argmax = d44 scenario
the canon supports twice, NOTHING_PURCHASABLE -- no substantive label reachable.
rev3 exists to fix that, and the fix is not cosmetic: two of the three changes
replace the estimand's shape, not its constants.

RULES-AS-CODE citation: CONSENSUS sec3 item 81 = PROJECT_STATUS gate #61.  (rev1
wrote "gate #66", rev2's correction wrote "item 66"; both were wrong and both are
recorded in the prereg sec 0.  Four numbering systems exist in this repo.)

WHAT CHANGED

F5  `visits_argmax` did not remove rev1's forced verdict, it renamed it: at the
    canonical anchor the controller demonstrably never reaches d44, so `no` was
    decided by already-purchased data.  Worse, it folded three different things
    into one blocking label.  Job 896565 settles which: BIND 0 but
    **SLO-FEAS refused 152** -- the controller was ALIVE and BLOCKED, and being
    blocked is HE0's research object, not a measurement failure.
    => the axis becomes DEAD / BLOCKED / MISPOSITIONED / REACHES and **only DEAD
    blocks**.  The other three are reported covariates.

F6  the four-state per-arm sign put the known canonical effect (-0.088 against
    delta 0.0966) in an `undecided` dead zone with P ~ 0.66.  rev3 stops asking
    each arm for a resolvable sign and asks the question the flip always was:
    an INTERACTION.  theta_M = (dyn - static)/static, within-model relative, so
    F3's cross-model-contamination defence is preserved; the estimand is
    theta_T - theta_H.  A dead zone cannot form, because equivalence at
    +-DELTA_INT covers the null explicitly.

F7  the `0` injection was the real data under another name, so "must not fire"
    pre-registered the headline as a control failure.  Dropped.  The permutation
    control's own NON-DEGENERACY becomes a registered, checked condition
    (CONTROL_DEGENERATE) instead of an assumption -- that is the repair for the
    empty-signature family, which has now bitten this track three times.

Run:  python3 tc1_rule_rev3.py
"""
import hashlib, json, math, os, sys
from itertools import product

RULE_REV = 3

# ── labels ───────────────────────────────────────────────────────────────────
BOOTF   = "BOOT_FAILED"
CAPUN   = "CAPACITY_UNMEASURABLE"
CLIFF   = "CLIFF_STRADDLE"
REALMM  = "REALIZED_MISMATCH"
TOKSKEW = "TOKENIZER_SKEW"
NODISC  = "NOT_DISCRIMINATING"
CTRLDEAD= "CONTROLLER_DEAD"          # F5: the ONLY blocking controller state
SCOREDEP= "SCORING_DEPENDENT"
CTLDEG  = "CONTROL_DEGENERATE"       # F7: the permutation null cannot fire
FLIP    = "FLIP_MODEL_ATTRIBUTED"    # interaction significant AND signs oppose
MAGDIFF = "MAGNITUDE_DIFFERS_NO_FLIP"# interaction significant, same sign
NOINT   = "NO_INTERACTION"           # equivalence at +-DELTA_INT
UNDERPWR= "UNDERPOWERED_NULL"
INCONC  = "INCONCLUSIVE"

MEASUREMENT = {BOOTF, CAPUN, CLIFF}
COMPARABIL  = {REALMM, TOKSKEW}
DESIGN      = {NODISC, CTRLDEAD, CTLDEG}
SCORING     = {SCOREDEP}
SUBSTANTIVE = {FLIP, MAGDIFF, NOINT, UNDERPWR, INCONC}
LABELS = MEASUREMENT | COMPARABIL | DESIGN | SCORING | SUBSTANTIVE

# ── constants ────────────────────────────────────────────────────────────────
N_REPS      = 4                 # CLAUDE.md 게이트 #3 (n>=4)
DELTA_REL   = 0.03              # CLAUDE.md 게이트 #3 -- per-arm materiality
# A sign flip with BOTH sides material implies the interaction clears 2*DELTA_REL.
# Derived from DELTA_REL, not chosen: rev2's B7 killed the free-constant habit.
DELTA_INT   = 2 * DELTA_REL     # 0.06 = 6 percentage points of relative arm effect
POWER_TARGET= 0.80
POWER_AT    = 2.0               # detect 2*DELTA_INT at 80% power
PAIRED      = False             # separate boots; Welch, df from data
CTRL_NULL_MAX = 0.10            # permutation control: null firing rate ceiling
CTRL_MIN_FIRE = 0.05            # ...and it must be able to fire at all
REALIZED_MAX_GAP = 0.15
TOKEN_MAX_GAP    = 0.20         # ★ measured 1.136-1.163 (audit B21) -- see prereg
ESTIMAND = ("theta_M = (goodput_dynamic,M - goodput_static,M) / goodput_static,M ; "
            "interaction = theta_T - theta_H ; goodput = COMBINED, durations SUMMED")
SIGN_SCORING = "p95"

# ── axes ─────────────────────────────────────────────────────────────────────
AXES = {
    "boot_H":   ("ok", "failed"),
    "boot_T":   ("ok", "failed"),
    "cliff":    ("clean", "straddle", "capacity_unmeasurable"),
    "compar":   ("ok", "realized_mismatch", "tokenizer_skew"),
    "discrim_H":("ok", "degenerate"),
    "discrim_T":("ok", "degenerate"),
    # F5: four states, ONE of which blocks
    "ctrl_H":   ("reaches", "mispositioned", "blocked", "dead"),
    "ctrl_T":   ("reaches", "mispositioned", "blocked", "dead"),
    "scoring_H":("agree", "disagree"),
    "scoring_T":("agree", "disagree"),
    "control":  ("ok", "degenerate"),                 # F7
    "inter":    ("ge_dint", "between", "le_neg_dint"),# |interaction| vs DELTA_INT
    "int_ci":   ("excludes_zero", "includes_zero"),
    "signs":    ("oppose", "same"),                   # do the two theta CIs oppose?
    "equiv":    ("passes", "fails"),                  # TOST at +-DELTA_INT
    "power":    ("adequate", "inadequate"),
}
AXIS_ORDER = tuple(AXES)
IX = {a: i for i, a in enumerate(AXIS_ORDER)}
GUARD_ORDER = ("boot", "cliff", "compar", "discrim", "ctrl", "scoring", "control")


def worlds(): return product(*(AXES[a] for a in AXIS_ORDER))


def label(w, skip=(), order=GUARD_ORDER):
    g = lambda a: w[IX[a]]
    for gd in order:
        if gd in skip: continue
        if gd == "boot" and (g("boot_H") == "failed" or g("boot_T") == "failed"): return BOOTF
        if gd == "cliff":
            if g("cliff") == "capacity_unmeasurable": return CAPUN
            if g("cliff") == "straddle":              return CLIFF
        if gd == "compar":
            if g("compar") == "realized_mismatch": return REALMM
            if g("compar") == "tokenizer_skew":    return TOKSKEW
        if gd == "discrim" and (g("discrim_H") == "degenerate" or g("discrim_T") == "degenerate"):
            return NODISC
        # ★F5: only `dead` blocks.  blocked / mispositioned / reaches are OBSERVED
        # states of the research object and are reported, not used to refuse.
        if gd == "ctrl" and (g("ctrl_H") == "dead" or g("ctrl_T") == "dead"): return CTRLDEAD
        if gd == "scoring" and (g("scoring_H") == "disagree" or g("scoring_T") == "disagree"):
            return SCOREDEP
        if gd == "control" and g("control") == "degenerate": return CTLDEG

    inter, ci, signs, eq, pw = g("inter"), g("int_ci"), g("signs"), g("equiv"), g("power")
    free = "power" in skip
    if ci == "excludes_zero" and inter in ("ge_dint", "le_neg_dint"):
        return FLIP if signs == "oppose" else MAGDIFF
    if eq == "passes" and ci == "includes_zero":
        return NOINT if (free or pw == "adequate") else UNDERPWR
    return INCONC


# ── power arithmetic ─────────────────────────────────────────────────────────
def _phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
_T = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,
      9:2.262,10:2.228,12:2.179,15:2.131}
def t_crit(df): return _T[max(k for k in _T if k <= max(df,1))]
def p_win(true_d, sd, thr_d, n=N_REPS, df=None):
    se = sd/math.sqrt(n); df = df if df is not None else 2*n-2
    return 1.0-_phi((max(thr_d, t_crit(df)*se)-true_d)/se)
def power_adequate(sd_int, n=N_REPS, df=None):
    return p_win(POWER_AT*DELTA_INT, sd_int, DELTA_INT, n, df) >= POWER_TARGET


MUTANTS = {f"drop_{g}": dict(skip=(g,)) for g in GUARD_ORDER}
MUTANTS["drop_power"] = dict(skip=("power",))
MUTANTS["reorder_ctrl_discrim"] = dict(
    order=("boot","cliff","compar","ctrl","discrim","scoring","control"))


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

    # ★F5: the three non-dead controller states must NOT block
    nonblock = [s for s in ("reaches","mispositioned","blocked")
                if any(label(w) in SUBSTANTIVE for w in W if w[IX["ctrl_H"]] == s)]
    check("T3_only_dead_blocks", sorted(nonblock) == ["blocked","mispositioned","reaches"],
          "blocked / mispositioned / reaches all reach substantive labels; only dead blocks")

    # ★F6: no dead zone -- every interaction/CI/equiv combination lands somewhere,
    # and the null has its own label rather than falling through
    check("T4_no_dead_zone",
          all(label(w) in SUBSTANTIVE for w in W
              if all(w[IX[a]] == b for a, b in (("boot_H","ok"),("boot_T","ok"),("cliff","clean"),
                     ("compar","ok"),("discrim_H","ok"),("discrim_T","ok"),
                     ("ctrl_H","blocked"),("ctrl_T","blocked"),("scoring_H","agree"),
                     ("scoring_T","agree"),("control","ok")))),
          "with all guards clear, every remaining world has a substantive label")
    check("T5_null_has_its_own_label",
          any(x == NOINT for x in L) and
          all(w[IX["equiv"]] == "passes" and w[IX["int_ci"]] == "includes_zero"
              for w, x in zip(W, L) if x == NOINT),
          "NO_INTERACTION requires equivalence, never mere absence of significance")
    check("T6_flip_needs_opposing_signs",
          all(w[IX["signs"]] == "oppose" for w, x in zip(W, L) if x == FLIP) and
          any(x == MAGDIFF for x in L),
          "a significant interaction with the SAME sign is MAGNITUDE_DIFFERS, not a flip")

    # ★F7: the control's own degeneracy is a label, and it blocks
    check("T7_control_degeneracy_blocks",
          all(label(w) == CTLDEG for w in W
              if w[IX["boot_H"]]=="ok" and w[IX["boot_T"]]=="ok" and w[IX["cliff"]]=="clean"
              and w[IX["compar"]]=="ok" and w[IX["discrim_H"]]=="ok" and w[IX["discrim_T"]]=="ok"
              and w[IX["ctrl_H"]]!="dead" and w[IX["ctrl_T"]]!="dead"
              and w[IX["scoring_H"]]=="agree" and w[IX["scoring_T"]]=="agree"
              and w[IX["control"]]=="degenerate"))

    killed = {n: sum(1 for w, x in zip(W, L) if label(w, **kw) != x) for n, kw in MUTANTS.items()}
    check("T8_mutant_coverage", all(v > 0 for v in killed.values()),
          f"uncovered={[k for k,v in killed.items() if v==0]}")
    check("T9_every_guard_has_mutant", all(f"drop_{g}" in MUTANTS for g in GUARD_ORDER))
    check("T10_delta_int_is_derived", abs(DELTA_INT - 2*DELTA_REL) < 1e-12,
          f"DELTA_INT={DELTA_INT} = 2*DELTA_REL; B7 killed the free-constant habit")
    check("T11_power_has_teeth",
          power_adequate(0.02) and not power_adequate(0.20),
          "a tight interaction SD passes, a loose one is refused")

    out = {"rule_rev": RULE_REV, "n_worlds": len(W),
           "axes": {k: list(v) for k, v in AXES.items()},
           "guard_order": list(GUARD_ORDER),
           "labels": {"MEASUREMENT": sorted(MEASUREMENT), "COMPARABILITY": sorted(COMPARABIL),
                      "DESIGN": sorted(DESIGN), "SCORING": sorted(SCORING),
                      "SUBSTANTIVE": sorted(SUBSTANTIVE)},
           "marginals": marg, "mutants_killed": killed,
           "constants": {"N_REPS": N_REPS, "DELTA_REL": DELTA_REL, "DELTA_INT": DELTA_INT,
                         "POWER_TARGET": POWER_TARGET, "POWER_AT": POWER_AT, "PAIRED": PAIRED,
                         "CTRL_NULL_MAX": CTRL_NULL_MAX, "CTRL_MIN_FIRE": CTRL_MIN_FIRE,
                         "REALIZED_MAX_GAP": REALIZED_MAX_GAP, "TOKEN_MAX_GAP": TOKEN_MAX_GAP,
                         "ESTIMAND": ESTIMAND, "SIGN_SCORING": SIGN_SCORING},
           "tests": res, "all_pass": not fail, "failed": fail}
    return out


if __name__ == "__main__":
    out = run_tests()
    out["rule_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "tc1_rule_rev3.json"), "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"RULE_REV={RULE_REV} worlds={out['n_worlds']:,} sha={out['rule_sha256'][:16]}")
    for k, v in out["tests"].items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v['detail']}")
    print("  marginals:", json.dumps(out["marginals"], ensure_ascii=False))
    print("ALL PASS" if out["all_pass"] else f"FAILED: {out['failed']}")
    sys.exit(0 if out["all_pass"] else 1)
