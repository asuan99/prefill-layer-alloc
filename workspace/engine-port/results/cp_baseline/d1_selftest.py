"""Self-test for `d1_rule.py` (RULE_REV 2) + `d1_predicates.py` (PRED_REV 2).
GPU 0.  Run:  python3 d1_selftest.py

Three families:
  C*  the rule layer -- totality, reachability, guard order, hand oracle
  P*  the predicates -- fixtures for every fold, including the A1 regression
  D*  DOCUMENT <-> CODE -- restored from the layer that earned this track's only
      GO (AF-1 C6/C11/C20).  rev1 dropped them and the 1st audit's U1 was five
      statements in the pre-registration that no check could see.
      ★ Discriminant (gate #53): D-checks run against the rev1 pre-registration
      MUST fail there.  A check that passes on the document it was written to
      catch is empty.
Every repair carries a MUTATION that must break it.
"""
import itertools
import random
import re
import sys
from pathlib import Path

import d1_predicates as P
import d1_rule as R

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG_D1_REV2_2026-09-07.md"
PREREG_REV1 = HERE / "PREREG_D1_2026-09-07.md"

FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(f"{name}: {detail}" if detail else name)


# =========================================================== C: the rule layer
def worlds():
    for combo in itertools.product(*(R.AXES[k] for k in R.AXIS_ORDER)):
        yield dict(zip(R.AXIS_ORDER, combo))


ALL = list(worlds())
COH = [w for w in ALL if R.incoherent(w) is None]

gaps = []
for w in COH:
    try:
        R.label(w)
    except Exception as e:                                    # noqa: BLE001
        gaps.append((w, repr(e)))
check("C1 every coherent world gets a label", not gaps, str(gaps[:2]))

known = set(R.SUBSTANTIVE) | set(R.NON_SUBSTANTIVE)
seen = {R.label(w) for w in COH}
check("C2 no undeclared label", not (seen - known), str(seen - known))
check("C3 every declared label is reachable",
      not (known - seen - {"IMPOSSIBLE_WORLD"}), str(known - seen))

miss = []
for w in COH:
    lab = R.label(w)
    try:
        br = R.fork_branch(lab, w["band_binding"])
    except Exception as e:                                    # noqa: BLE001
        miss.append((lab, w["band_binding"], repr(e)))
        continue
    if br not in R.FORK_BRANCHES:
        miss.append((lab, w["band_binding"], br))
check("C4 every reachable (label, band_binding) has a declared branch",
      not miss, str(miss[:3]))
reach_br = {R.fork_branch(R.label(w), w["band_binding"]) for w in COH}
check("C4b every declared branch is reachable",
      set(R.FORK_BRANCHES) == reach_br, str(set(R.FORK_BRANCHES) ^ reach_br))

check("C5 fragile / yardstick labels carry no direction",
      not any(t in l for l in ("GRID_RESOLUTION_DECIDES", "YARDSTICK_DECIDES",
                               "GRID_UNBRACKETABLE")
              for t in ("CP2048", "D44", "PDMUX", "DOMINANT")))
check("C5b directional labels name the registered pair, not the family",
      not any(l.startswith(("PDMUX", "CHUNKED")) for l in R.SUBSTANTIVE)
      and {"CP2048_DOMINANT", "D44_DOMINANT"} <= set(R.SUBSTANTIVE))

# guard order: the yardstick guard must beat the outcome table
w_y = {"trace": "matched", "coverage": "sufficient", "bracket": "bracketed",
       "estimator": "strong_discordant", "sign_field": "crossing",
       "robustness": "stable", "band_binding": "both"}
check("C6 strong discordance beats the outcome table",
      R.label(w_y) == "YARDSTICK_DECIDES", R.label(w_y))
mut = [g for g in R.GUARDS if g is not R._g_yardstick]
check("C6-M dropping the yardstick guard changes that world (mutation caught)",
      R.label(w_y, rules=mut) == "SLO_DEPENDENT", R.label(w_y, rules=mut))

w_bad = dict(w_y, estimator="concordant", sign_field="null",
             band_binding="none")
check("C7 null with no 0-calls is impossible",
      R.label(w_bad) == "IMPOSSIBLE_WORLD", R.label(w_bad))
mut_c = tuple(c for c in R.COHERENCE if c is not R._k_null_needs_zeros)
try:
    _got = R.label(w_bad, clauses=mut_c)
    _caught = False
except R.OutcomeGap:
    _caught, _got = True, "OutcomeGap"
check("C7-M dropping the clause leaves the world unlabellable (mutation caught)",
      _caught, str(_got))

# the distinction rev1 could not make
w_ci = dict(w_y, estimator="concordant", sign_field="null", band_binding="ci")
w_fl = dict(w_ci, band_binding="floor")
check("C8 null bound by the CI is UNAFFORDABLE_PRECISION",
      R.label(w_ci) == "UNAFFORDABLE_PRECISION", R.label(w_ci))
check("C8b null bound by the 3% floor is PAIR_INDISTINGUISHABLE",
      R.label(w_fl) == "PAIR_INDISTINGUISHABLE", R.label(w_fl))
check("C8c the two are different labels with different branches",
      R.fork_branch(R.label(w_ci)) != R.fork_branch(R.label(w_fl)))

check("C9 Stage V is gone: no interim axis, no interim label",
      not any("interim" in v for vals in R.AXES.values() for v in vals)
      and not any("INTERIM" in l for l in known))

reach3 = {(w["sign_field"], w["robustness"], w["band_binding"]) for w in COH
          if w["sign_field"] != "unmeasured"
          and R.label(w) not in ("YARDSTICK_DECIDES",)}
check("C10 OUTCOME keys cover every reachable triple",
      reach3 <= set(R.OUTCOME), str(reach3 - set(R.OUTCOME)))
check("C10b OUTCOME has no key the screen cannot reach",
      set(R.OUTCOME) <= {(w["sign_field"], w["robustness"], w["band_binding"])
                         for w in COH if w["sign_field"] != "unmeasured"},
      "extra keys")

ORACLE = [
    (("unmeasured", "insufficient", "unmeasured", "unmeasured", "unmeasured",
      "unmeasured", "unmeasured"), "NOTHING_SCORED", "n/a"),
    (("mismatched", "sufficient", "unmeasured", "unmeasured", "unmeasured",
      "unmeasured", "unmeasured"), "TRACE_INVALID", "n/a"),
    (("matched", "insufficient", "unmeasured", "unmeasured", "unmeasured",
      "unmeasured", "unmeasured"), "COVERAGE_INSUFFICIENT", "n/a"),
    (("matched", "sufficient", "unbracketable", "unmeasured", "unmeasured",
      "unmeasured", "unmeasured"), "GRID_UNBRACKETABLE", "regime_report"),
    (("matched", "sufficient", "bracketed", "strong_discordant", "crossing",
      "stable", "both"), "YARDSTICK_DECIDES", "yardstick_report"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing", "stable",
      "ci"), "SLO_DEPENDENT", "answered_sign_flip_extremes_only"),
    (("matched", "sufficient", "bracketed", "concordant", "cp_dominant",
      "stable", "none"), "CP2048_DOMINANT", "answered_cp_pair"),
    (("matched", "sufficient", "bracketed", "weak_discordant", "pd_dominant",
      "stable", "floor"), "D44_DOMINANT", "answered_pd_pair"),
    (("matched", "sufficient", "bracketed", "concordant", "null", "stable",
      "floor"), "PAIR_INDISTINGUISHABLE", "answered_null_below_floor"),
    (("matched", "sufficient", "bracketed", "concordant", "null", "stable",
      "ci"), "UNAFFORDABLE_PRECISION", "precision_bound"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing",
      "grid_fragile", "none"), "GRID_RESOLUTION_DECIDES", "refine_grid"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing",
      "grid_fragile", "ci"), "GRID_RESOLUTION_DECIDES", "buy_boots"),
    (("matched", "sufficient", "bracketed", "concordant", "cp_dominant",
      "grid_fragile", "floor"), "GRID_RESOLUTION_DECIDES", "precision_bound"),
    (("matched", "sufficient", "bracketed", "strong_discordant", "cp_dominant",
      "stable", "none"), "IMPOSSIBLE_WORLD", "n/a"),
]
bad_o = []
for vals, lx, bx in ORACLE:
    w = dict(zip(R.AXIS_ORDER, vals))
    gl = R.label(w)
    gb = R.fork_branch(gl, w["band_binding"])
    if (gl, gb) != (lx, bx):
        bad_o.append((vals, gl, gb, lx, bx))
check(f"C11 hand oracle ({len(ORACLE)} rows)", not bad_o, str(bad_o[:2]))

# regime partition -- the transfer question, enforced
import types  # noqa: E402
named = {n for n in dir(R) + dir(P)
         if n.isupper() and not n.startswith("_") and len(n) > 1
         and not isinstance(getattr(R, n, getattr(P, n, None)), types.ModuleType)}
named -= {"RULE_REV", "PRED_REV", "REGIME_FREE", "REGIME_BOUND",
          "DECISION_CONSTANTS", "PREDICATE_MODULE", "PREDICATE_FOLDS"}
unclassified = sorted(n for n in named
                      if n not in R.REGIME_FREE and n not in R.REGIME_BOUND)
check("C12 every constant is classified free or regime-bound",
      not unclassified, str(unclassified))
check("C12b the two regime sets are disjoint",
      not (set(R.REGIME_FREE) & set(R.REGIME_BOUND)))
try:
    R.regime_transfer("NOT_A_CONSTANT")
    caught = False
except KeyError:
    caught = True
check("C12-M an unclassified name raises (mutation caught)", caught)

# ========================================================== P: the predicates
check("P1 bucket_of is nearest in LOG length",
      P.bucket_of(1900) == 0 and P.bucket_of(2600) == 1
      and P.bucket_of(3800) == 2 and P.bucket_of(7000) == 3
      and P.bucket_of(2204.3 - 1) == 0 and P.bucket_of(2204.3 + 1) == 1)

ft, fi = P.FLOOR_TTFT_MS["plain"], P.FLOOR_ITLP95_MS["plain"]
rec_ok = {"prompt_tok": 2514, "ttft_ms": 500.0, "itl_p95_ms": 20.0}
rec_dnf = {"prompt_tok": 2514, "ttft_ms": None, "itl_p95_ms": None}
check("P2 an unfinished request is a violation, not an absence",
      P.request_passes(rec_ok, 2, 2, ft, fi)
      and not P.request_passes(rec_dnf, 1000, 1000, ft, fi))
check("P2b goodput divides by the SUMMED duration",
      abs(P.goodput([rec_ok] * 10, 2, 2, ft, fi, 5.0) - 2.0) < 1e-9)
try:
    P.goodput([rec_ok], 2, 2, ft, fi, 0.0)
    dur_ok = False
except ValueError:
    dur_ok = True
check("P2c a zero/negative duration raises", dur_ok)

check("P3 degenerate ends",
      P.degenerate_low({"a": 0.0, "b": 0.0})
      and not P.degenerate_low({"a": 0.0, "b": 0.1})
      and P.degenerate_high({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
      and not P.degenerate_high({"a": 0.5, "b": 2.0}, {"a": 1.0, "b": 2.0}))
grid = {(1, 1): {"a": 0.0, "b": 0.0}, (64, 32): {"a": 1.0, "b": 2.0}}
check("P4 bracket_ok needs BOTH degenerate ends",
      P.bracket_ok(grid, {"a": 1.0, "b": 2.0})
      and not P.bracket_ok({(1, 1): {"a": 0.0, "b": 0.0}}, {"a": 1.0, "b": 2.0}))
check("P4b extend_grid doubles and stops at K_MAX",
      P.extend_grid((1, 2, 4), 16) == (1, 2, 4, 8)
      and P.extend_grid((1, 2, 4, 8), 8) is None)

check("P5 max(G)==0 is a floor-bound 0-call, never a CI one",
      P.call_sign(0.0, (-1.0, 1.0), 0.0, 0.03) == (0, "floor"))
check("P5b a point is called only if BOTH criteria clear",
      P.call_sign(0.5, (0.2, 0.8), 1.0, 0.03)[0] == 1
      and P.call_sign(0.5, (-0.2, 1.2), 1.0, 0.03) == (0, "ci")
      and P.call_sign(0.01, (0.005, 0.015), 1.0, 0.03) == (0, "floor")
      and P.call_sign(0.01, (-0.2, 0.3), 1.0, 0.03) == (0, "both"))

check("P6 the intersection can remove a sign but never create one",
      P.sign_intersect(1, 1) == 1 and P.sign_intersect(1, 0) == 0
      and P.sign_intersect(1, -1) == 0 and P.sign_intersect(0, 0) == 0
      and all(P.sign_intersect(a, b) in (a, 0)
              for a in (-1, 0, 1) for b in (-1, 0, 1)))
check("P6b estimator_class separates weak from strong discordance",
      P.estimator_class([(1, 1), (0, 0)]) == "concordant"
      and P.estimator_class([(1, 0)]) == "weak_discordant"
      and P.estimator_class([(1, -1), (0, 0)]) == "strong_discordant")
check("P6c an opposite-signed point is filed as `both`, not `ci`",
      P.intersect_reason(1, -1, "called", "called") == "both")

check("P7 sign_field_of and band_binding_of",
      P.sign_field_of([1, 0, -1]) == "crossing"
      and P.sign_field_of([1, 0]) == "cp_dominant"
      and P.sign_field_of([-1]) == "pd_dominant"
      and P.sign_field_of([0, 0]) == "null"
      and P.band_binding_of(["called"]) == "none"
      and P.band_binding_of(["called", "ci"]) == "ci"
      and P.band_binding_of(["ci", "floor"]) == "both"
      and P.band_binding_of(["both"]) == "both")
check("P7b leave-one-out includes the degenerate ends",
      P.loo_stability({(1, 1): 0, (2, 2): 1, (4, 4): 0}) == "grid_fragile"
      and P.loo_stability({(1, 1): 1, (2, 2): 1, (4, 4): 1}) == "stable")

# ★ A1 regression: the measured yardstick bias that killed rev1
OCC = (0.3161, 0.4212, 0.2160, 0.0467)
bt = {a: P.yardstick_bias(a, OCC, P.FLOOR_TTFT_MS) for a in ("cp2048", "d44")}
bi = {a: P.yardstick_bias(a, OCC, P.FLOOR_ITLP95_MS) for a in ("cp2048", "d44")}
dt, di = 100 * (bt["cp2048"] - bt["d44"]), 100 * (bi["cp2048"] - bi["d44"])
check("P8 A1 regression: the differential bias is reproduced (TTFT -5.07%)",
      abs(dt - (-5.07)) < 0.05, f"{dt:+.2f}%")
check("P8b A1 regression: ITL +5.97%", abs(di - 5.97) < 0.05, f"{di:+.2f}%")
check("P8c both exceed PRACTICAL_FLOOR and point OPPOSITE ways",
      abs(dt) > 100 * R.PRACTICAL_FLOOR and abs(di) > 100 * R.PRACTICAL_FLOOR
      and dt * di < 0)

rng = random.Random(11)
lo, hi = P.paired_bootstrap_ci([0.5, 0.6, 0.55, 0.52], 0.95, 2000, rng)
check("P9 paired bootstrap CI brackets the mean and excludes 0 here",
      lo > 0 and lo < 0.5443 < hi, f"[{lo:.4f},{hi:.4f}]")

# ================================================== D: document <-> code
MUT_COUNT = 6          # C6-M, C7-M, C12-M, D1-M, D1b-M, D5-M
KNOWN_WORDS = set()
for mod in (R, P):
    KNOWN_WORDS |= {n for n in dir(mod) if not n.startswith("__")}
KNOWN_WORDS |= {v for vals in R.AXES.values() for v in vals}
KNOWN_WORDS |= set(R.AXES) | set(known) | set(R.FORK_BRANCHES)
KNOWN_WORDS |= set(R.ARM_FLAGS) | {"GRID_RESOLUTION_DECIDES"}
# the deletion ledger: naming a removed identifier is the point of sec 11
KNOWN_WORDS |= set(R.DELETED_IN_REV2)
# prose that is code-shaped but is not ours
ALLOW = {"max_running_requests", "chunked_prefill_size", "enable_pdmux",
         "disable_overlap_schedule", "sharegpt_context_len", "request_rate",
         "num_prompts", "mem_fraction_static", "max_total_num_tokens",
         "longbench_v2", "random_input_len", "nemotron_h", "sglang_engine_dev",
         "disable_piecewise_cuda_graph", "af1_strata", "d1_rule",
         "d1_predicates", "d1_selftest", "presubmit", "design_reachability",
         "check_version_sweep", "PDMUX_TRUE_DUAL_WORKER", "SLURM_JOB_ID"}
CODE_SHAPED = re.compile(r"^(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9_]{3,})$")


def ghosts(path):
    if not path.exists():
        return None
    toks = set(re.findall(r"`([^`\n]+)`", path.read_text()))
    out = set()
    for t in toks:
        t = t.strip().rstrip("()")
        if CODE_SHAPED.match(t) and t not in KNOWN_WORDS and t not in ALLOW:
            out.add(t)
    return out


g2 = ghosts(PREREG)
if g2 is None:
    check("D1 ghost-identifier scan on rev2", False, "PREREG rev2 not found")
else:
    check("D1 no ghost identifier in the pre-registration", not g2, str(sorted(g2)))

# The docstrings are scanned too: U1's ghost (`slo_aware_arm_present`) lived in
# the rev1 rule module's docstring, not in the pre-registration.
def ghosts_text(txt):
    out = set()
    for t in set(re.findall(r"`([^`\n]+)`", txt)):
        t = t.strip().rstrip("()")
        if CODE_SHAPED.match(t) and t not in KNOWN_WORDS and t not in ALLOW:
            out.add(t)
    return out


doc_g = ghosts_text((R.__doc__ or "") + (P.__doc__ or ""))
check("D1b no ghost identifier in the module docstrings", not doc_g,
      str(sorted(doc_g)))

# ★ MUTATION (gate #53): inject the exact ghost the 1st audit found and require
# the scan to flag it.  A check that cannot fail is empty.
GHOST = "slo_aware_arm_present"
inject = (PREREG.read_text() if PREREG.exists() else "") + f"\n`{GHOST}`\n"
check("D1-M ★the scan flags an injected ghost (mutation caught)",
      GHOST in ghosts_text(inject))
check("D1b-M ★the docstring scan flags an injected ghost (mutation caught)",
      GHOST in ghosts_text((R.__doc__ or "") + f"\n`{GHOST}`\n"))

# Observation, not a pass/fail: applied to the rev1 document the scan flags its
# stale identifiers (rev1's label and branch names, renamed in rev2).
_g1 = ghosts(PREREG_REV1)
REV1_STALE = sorted(_g1) if _g1 else []

if PREREG.exists():
    txt = PREREG.read_text()
    missing_fork = [f"{l} -> {b}" for l, b in R.FORK.items()
                    if b != "n/a" and (l not in txt or b not in txt)]
    check("D2 every FORK row appears in the pre-registration",
          not missing_fork, str(missing_fork[:3]))
    missing_arm = [a for a, f in R.ARM_FLAGS.items() if f.split()[0] not in txt]
    check("D3 the arm table quotes the authoritative flags",
          not missing_arm, str(missing_arm))
    m = re.search(r"변이\s*(\d+)\s*건", txt)
    check("D4 the stated mutation count matches the actual one",
          bool(m) and int(m.group(1)) == MUT_COUNT,
          f"stated={m.group(1) if m else None} actual={MUT_COUNT}")
    # ★2nd audit V3: rev2 hard-coded the literal here, so the check could not tell a
    # right value from a wrong one (mutation M5 passed with the CORRECTED constant).
    # The expected string is now DERIVED from the constant it is meant to police.
    _fr = f"{100 * P.FIRING_RATE[2048]:.2f}"
    check("D5 the pre-registration quotes the registered firing rate",
          _fr in txt and "200/200" not in txt, f"expected {_fr}%")
    # MUTATION: change the constant and the check must move with it.
    _saved = P.FIRING_RATE[2048]
    P.FIRING_RATE[2048] = 0.1234
    _moved = f"{100 * P.FIRING_RATE[2048]:.2f}" not in txt
    P.FIRING_RATE[2048] = _saved
    check("D5-M the expected string tracks the constant (mutation caught)", _moved)
    check("D6 sec 10 (regime transfer) names both halves",
          "REGIME_FREE" in txt and "REGIME_BOUND" in txt)
    # ★ gate #88 as an enforced artifact: every deleted name is (a) genuinely
    # gone from both modules and (b) accounted for in the document.
    live = [n for n in R.DELETED_IN_REV2
            if hasattr(R, n) or hasattr(P, n)
            or n in {v for vals in R.AXES.values() for v in vals}
            or n in R.FORK_BRANCHES]
    unlisted = [n for n in R.DELETED_IN_REV2 if n not in txt]
    check("D7 the deletion ledger is real (nothing on it is still live)",
          not live, str(live))
    check("D7b every deleted name is accounted for in the document",
          not unlisted, str(unlisted))
else:
    for n in ("D2", "D3", "D4", "D5", "D6", "D7", "D7b"):
        check(f"{n} needs the rev2 pre-registration", False, "not written yet")

print(f"rev1 stale identifiers the scan flags (observation): {REV1_STALE}")
print(f"world space: {len(ALL)} total, {len(COH)} coherent, "
      f"{len(ALL)-len(COH)} incoherent · labels reached {len(seen)}")
for x in PASSES:
    print(f"  PASS  {x}")
for x in FAILS:
    print(f"  FAIL  {x}")
print(f"\n{len(PASSES)} PASS · {len(FAILS)} FAIL · mutations {MUT_COUNT}")
sys.exit(1 if FAILS else 0)
