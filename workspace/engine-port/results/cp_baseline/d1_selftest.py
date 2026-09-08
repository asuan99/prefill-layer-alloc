"""Self-test for `d1_rule.py` (RULE_REV 3) + `d1_predicates.py` (PRED_REV 3).
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
PREREG = HERE / "PREREG_D1_REV3_2026-09-08.md"
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
       "robustness": "stable", "band_binding": "imprecise"}
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
w_ci = dict(w_y, estimator="concordant", sign_field="null",
            band_binding="imprecise")
w_fl = dict(w_ci, band_binding="precise")
check("C8 null with an unresolved point is UNAFFORDABLE_PRECISION",
      R.label(w_ci) == "UNAFFORDABLE_PRECISION", R.label(w_ci))
check("C8b null with every point precise is PAIR_INDISTINGUISHABLE",
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
      "stable", "imprecise"), "YARDSTICK_DECIDES", "yardstick_report"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing", "stable",
      "precise"), "SLO_DEPENDENT", "answered_sign_flip_extremes_only"),
    (("matched", "sufficient", "bracketed", "concordant", "cp_dominant",
      "stable", "none"), "CP2048_DOMINANT", "answered_cp_pair"),
    (("matched", "sufficient", "bracketed", "weak_discordant", "pd_dominant",
      "stable", "precise"), "D44_DOMINANT", "answered_pd_pair"),
    (("matched", "sufficient", "bracketed", "concordant", "null", "stable",
      "precise"), "PAIR_INDISTINGUISHABLE", "answered_null_below_floor"),
    (("matched", "sufficient", "bracketed", "concordant", "null", "stable",
      "imprecise"), "UNAFFORDABLE_PRECISION", "precision_bound"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing",
      "grid_fragile", "none"), "GRID_RESOLUTION_DECIDES", "refine_grid"),
    (("matched", "sufficient", "bracketed", "concordant", "crossing",
      "grid_fragile", "imprecise"), "GRID_RESOLUTION_DECIDES", "buy_boots"),
    (("matched", "sufficient", "bracketed", "concordant", "cp_dominant",
      "grid_fragile", "precise"), "GRID_RESOLUTION_DECIDES", "refine_grid"),
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

# ---- regime partition: the transfer question, ENFORCED (2nd audit V9) ----
# rev2's C12 read `n.isupper()`, so a lowercase module constant or a threshold
# buried in a function body walked straight past it, and nothing pinned WHICH
# side a name sat on.  All three are mutations below.
import ast          # noqa: E402
import types        # noqa: E402

STRUCTURAL = {"RULE_REV", "PRED_REV", "REGIME_FREE", "REGIME_BOUND",
              "DECISION_CONSTANTS", "PREDICATE_MODULE", "PREDICATE_FOLDS",
              "DELETED_IN_REV2", "DELETED_IN_REV3", "AXES", "AXIS_ORDER",
              "OUTCOME", "FORK", "FORK_BRANCHES", "FORK_GRID_FRAGILE",
              "GUARDS", "COHERENCE", "SUBSTANTIVE", "NON_SUBSTANTIVE",
              "OutcomeGap", "NUMERICAL_TOLERANCES"}


def module_constants(mod):
    """Every module-level value that is not a function, class, module or
    dunder -- IN EITHER CASE.  This is the check rev2 did not have."""
    out = set()
    for n in dir(mod):
        if n.startswith("_"):          # private structural helpers
            continue
        v = getattr(mod, n)
        if isinstance(v, (types.ModuleType, types.FunctionType, type)):
            continue
        out.add(n)
    return out


named = (module_constants(R) | module_constants(P)) - STRUCTURAL
unclassified = sorted(n for n in named
                      if n not in R.REGIME_FREE and n not in R.REGIME_BOUND)
check("C12 every module constant (either case) is classified",
      not unclassified, str(unclassified))
check("C12b the two regime sets are disjoint",
      not (set(R.REGIME_FREE) & set(R.REGIME_BOUND)))

# ★ the side each name sits on is PINNED here by hand, so moving one fails.
EXPECTED_SIDE = {
    "MIN_BOOTS": "free", "CI_LEVEL": "free", "ALPHA": "free",
    "T_CRIT_UNCORRECTED": "free",
    "N_BOOTS": "free", "PRACTICAL_FLOOR": "free",
    "K_MAX": "bound", "K_T_START": "bound", "K_I_START": "bound",
    "PRIMARY_PAIR": "bound", "DIAGNOSTIC_ARMS": "bound", "ARM_FLAGS": "bound",
    "STRATA_TOK": "bound", "FLOOR_TTFT_MS": "bound",
    "FLOOR_ITLP95_MS": "bound", "FIRING_RATE": "bound",
    "YARDSTICK_ARM": "bound", "DEGEN_HIGH_TOL": "bound",
}
side_bad = [n for n, side in EXPECTED_SIDE.items()
            if R.regime_transfer(n) != side]
check("C12c each constant sits on the PINNED side of the regime split",
      not side_bad and set(EXPECTED_SIDE) == named,
      f"moved={side_bad} unpinned={sorted(named - set(EXPECTED_SIDE))}")
# MUTATION: move one across.
_free = list(R.REGIME_FREE)
_saved_free, _saved_bound = R.REGIME_FREE, R.REGIME_BOUND
R.REGIME_FREE = tuple(x for x in _free if x != "MIN_BOOTS")
R.REGIME_BOUND = R.REGIME_BOUND + ("MIN_BOOTS",)
_moved_caught = R.regime_transfer("MIN_BOOTS") != EXPECTED_SIDE["MIN_BOOTS"]
R.REGIME_FREE, R.REGIME_BOUND = _saved_free, _saved_bound
check("C12b-M moving a constant across the split is caught (mutation caught)",
      _moved_caught)

# ★ thresholds buried in function bodies are constants too (mutation M3).
ALLOWED_LITERALS = {0, 1, 2, -1, 0.0, 1.0, 2.0, 1e-15, 1e-6, 0.5, 100.0}
buried = []
for mod_path in ("d1_rule.py", "d1_predicates.py"):
    tree = ast.parse((HERE / mod_path).read_text())
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool)
                    and node.value not in ALLOWED_LITERALS):
                buried.append(f"{mod_path}:{node.lineno}={node.value}")
check("C12d no unnamed numeric threshold hides in a function body",
      not buried, str(buried))
_mut = ast.parse("def f(x):\n    return x > 0.25\n")
_hit = [n for n in ast.walk(_mut) if isinstance(n, ast.Constant)
        and n.value not in ALLOWED_LITERALS]
check("C12d-M a buried threshold is caught (mutation caught)", bool(_hit))
# ★ and a private numeric constant may not hide either: the declared tuple must
# cover exactly the private numeric module constants.
priv = {n for n in dir(P)
        if n.startswith("_") and not n.startswith("__")
        and isinstance(getattr(P, n), (int, float))
        and not isinstance(getattr(P, n), bool)}
check("C12e NUMERICAL_TOLERANCES covers exactly the private numeric constants",
      priv == set(P.NUMERICAL_TOLERANCES),
      str(priv ^ set(P.NUMERICAL_TOLERANCES)))

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

L, TC = R.CI_LEVEL, R.T_CRIT_UNCORRECTED
BIG = [0.5, 0.55, 0.6, 0.52, 0.58, 0.54]        # clear effect, tight
TINY = [0.001, 0.0012, 0.0011, 0.0009, 0.001, 0.0011]   # sub-floor, tight
NOISY = [0.1, -0.12, 0.08, -0.09, 0.11, -0.1]   # null, wide
check("P5 max(G)==0 is a `precise` 0-call (no boots change an identical zero)",
      P.call_sign([0.0] * 6, 0.0, 0.03, L, TC) == (0, "precise"))
check("P5b called / precise / imprecise are the three outcomes",
      P.call_sign(BIG, 1.0, 0.03, L, TC) == (1, "called")
      and P.call_sign(TINY, 1.0, 0.03, L, TC) == (0, "precise")
      and P.call_sign(NOISY, 1.0, 0.03, L, TC) == (0, "imprecise"))
check("P5c ★gate #14: the label-bearing interval is the t-CI, not the bootstrap",
      "rng" not in P.call_sign.__code__.co_varnames
      and "paired_t_ci" in P.call_sign.__code__.co_names
      and "paired_bootstrap_ci" not in P.call_sign.__code__.co_names)
# MUTATION: feed call_sign the bootstrap interval's width and the null world
# above must stop being `imprecise` -- i.e. the choice of interval is load-bearing.
_rng = random.Random(2)
_blo, _bhi = P.paired_bootstrap_ci(NOISY, L, 2000, _rng)
_tlo, _thi = P.paired_t_ci(NOISY, L, TC)
check("P5c-M the bootstrap interval is materially narrower than the t-CI",
      (_bhi - _blo) < 0.75 * (_thi - _tlo),
      f"boot {_bhi-_blo:.4f} vs t {_thi-_tlo:.4f}")
# ★ multiplicity: the correction, its price, and the fact that the
# distribution-free route is closed at this n over this grid.
_M = len(R.K_T_START) * len(R.K_I_START) - 2      # interior, both ends removed
_tb = P.t_crit_for(R.ALPHA / _M, R.MIN_BOOTS - 1)
check("P5e t_crit_for reproduces the table (t(0.05, df=5) = 2.571)",
      abs(P.t_crit_for(0.05, 5) - 2.571) < 0.001, f"{P.t_crit_for(0.05, 5):.4f}")
check("P5f the Bonferroni correction is real and priced",
      _tb > 2.5 * R.T_CRIT_UNCORRECTED, f"m={_M} t_bonf={_tb:.3f}")
check("P5g Holm is step-down (thresholds a/m, a/(m-1), ...) and rejects "
      "nothing in a null-only family",
      P.holm_calls([0.001, 0.02, 0.3], 0.05) == (True, True, False)
      and P.holm_calls([0.06, 0.3], 0.05) == (False, False)
      and P.holm_calls([0.2, 0.3, 0.4], 0.05) == (False, False, False))
check("P5h ★the distribution-free route is CLOSED at this n over this grid "
      "(and the n that would open it is registered)",
      2 / 2 ** R.MIN_BOOTS > R.ALPHA / _M
      and 2 / 2 ** 11 < R.ALPHA / _M)
check("P5d ★the exact sign-flip floor is why MIN_BOOTS is 6",
      abs(P.permutation_p([1] * 6) - 2 / 2 ** 6) < 1e-12
      and 2 / 2 ** R.MIN_BOOTS < 0.05
      and 2 / 2 ** (R.MIN_BOOTS - 1) > 0.05)

check("P6 the intersection can remove a sign but never create one",
      P.sign_intersect(1, 1) == 1 and P.sign_intersect(1, 0) == 0
      and P.sign_intersect(1, -1) == 0 and P.sign_intersect(0, 0) == 0
      and all(P.sign_intersect(a, b) in (a, 0)
              for a in (-1, 0, 1) for b in (-1, 0, 1)))
check("P6b estimator_class separates weak from strong discordance",
      P.estimator_class([(1, 1), (0, 0)]) == "concordant"
      and P.estimator_class([(1, 0)]) == "weak_discordant"
      and P.estimator_class([(1, -1), (0, 0)]) == "strong_discordant")
check("P6c an opposite-signed point is filed as `imprecise`",
      P.intersect_reason(1, -1, "called", "called") == "imprecise")

check("P7 sign_field_of and band_binding_of",
      P.sign_field_of([1, 0, -1]) == "crossing"
      and P.sign_field_of([1, 0]) == "cp_dominant"
      and P.sign_field_of([-1]) == "pd_dominant"
      and P.sign_field_of([0, 0]) == "null"
      and P.band_binding_of(["called"]) == "none"
      and P.band_binding_of(["called", "precise"]) == "precise"
      and P.band_binding_of(["precise", "imprecise"]) == "imprecise")
# ★V1: the degenerate ends are witnesses, not comparison points
GS = {(1, 1): {"a": 0.0, "b": 0.0}, (4, 4): {"a": 0.3, "b": 0.5},
      (64, 32): {"a": 1.0, "b": 2.0}}
ACH = {"a": 1.0, "b": 2.0}
check("P7c interior_points drops both degenerate ends",
      P.interior_points(GS, ACH) == ((4, 4),))
check("P7d the satisfying end's throughput difference is reported, not hidden",
      abs(P.throughput_end_delta(GS, ACH, "a", "b") - (-1.0)) < 1e-12)
check("P7e bracketing still requires both ends",
      P.bracket_ok(GS, ACH))
check("P7b leave-one-out runs over the interior points",
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
MUT_COUNT = 9          # C6-M C7-M C12b-M C12d-M P5c-M D1-M D1b-M D3-M D5-M
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
         "check_version_sweep", "PDMUX_TRUE_DUAL_WORKER", "SLURM_JOB_ID",
         "RESULT_W1", "RESULT_AF1", "PROJECT_STATUS", "CONSENSUS",
         "code_inline", "unpaired_bootstrap_ci", "paired_bootstrap_ci"}
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
    # ★2nd audit V4: rev2 compared `f.split()[0]` only, so it was blind to the
    # flag's VALUE -- the very statement it existed to catch.
    def arm_flag_misses(doc):
        bad = []
        for a, f in R.ARM_FLAGS.items():
            toks = f.split()
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    want = f"{toks[i]} {toks[i + 1]}"   # flag WITH its value
                    i += 2
                else:
                    want = toks[i]                       # bare flag
                    i += 1
                if want not in doc:
                    bad.append(f"{a}:{want}")
        return bad
    check("D3 the arm table quotes the authoritative flags AND their values",
          not arm_flag_misses(txt), str(arm_flag_misses(txt)[:3]))
    _m3 = txt.replace("--chunked-prefill-size -1", "--chunked-prefill-size 8192")
    check("D3-M a wrong flag VALUE in the document is caught (mutation caught)",
          bool(arm_flag_misses(_m3)))

    def const_count_claim(doc):
        mm = re.search(r"결정 상수[^0-9]{0,12}(\d+)\s*개", doc)
        return None if not mm else int(mm.group(1))
    _cc = const_count_claim(txt)
    check("D8 a `결정 상수 N개` claim matches DECISION_CONSTANTS",
          _cc is None or _cc == len(R.DECISION_CONSTANTS),
          f"claimed={_cc} actual={len(R.DECISION_CONSTANTS)}")

    def dead_terms_outside_ledger(doc):
        # ★ whole-token match: `answered_cp` must not fire on `answered_cp_pair`,
        # and the one-word band values (`ci`/`floor`/`both`) are ordinary prose
        # words, so only names long enough to be unambiguous are scanned.
        head = doc.split("## 11")[0]
        terms = [n for n in R.DELETED_IN_REV2 + R.DELETED_IN_REV3 if len(n) > 6]
        hits = [n for n in terms
                if re.search(r"(?<![A-Za-z0-9_])" + re.escape(n)
                             + r"(?![A-Za-z0-9_])", head)]
        if re.search(r"Stage\s*V(?![A-Za-z0-9_])", head):
            hits.append("Stage V")
        return hits
    check("D9 a deleted thing is named only in the deletion ledger",
          not dead_terms_outside_ledger(txt),
          str(dead_terms_outside_ledger(txt)))
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
    # ★ DISCRIMINANT (2nd audit V4).  rev1's rule module was overwritten in
    # place, so three of the five statements it was written to catch no longer
    # exist on disk -- the honest form of the test is a FIXTURE carrying all
    # five, which does not depend on a lost file.
    FIXTURE = ("`slo_aware_arm_present` is a hard guard.  "
               "arm `plain`(엔진 기본값 `--chunked-prefill-size 8192`).  "
               "자기검사는 변이 2건을 포함한다.  "
               "Stage V는 실질 라벨을 낼 수 없다.  "
               "결정 상수 99개가 전부다.\n## 11 원장\n")
    caught = {
        "ghost": bool(ghosts_text(FIXTURE)),
        "flag_value": bool(arm_flag_misses(FIXTURE)),
        "mut_count": (re.search(r"변이\s*(\d+)\s*건", FIXTURE)
                      and int(re.search(r"변이\s*(\d+)\s*건", FIXTURE).group(1))
                      != MUT_COUNT),
        "dead_term": bool(dead_terms_outside_ledger(FIXTURE)),
        "const_count": const_count_claim(FIXTURE) != len(R.DECISION_CONSTANTS),
    }
    check(f"D-DISC ★discriminant: the checks catch "
          f"{sum(bool(v) for v in caught.values())}/5 of the statements they "
          f"were written for (requirement: >=4)",
          sum(bool(v) for v in caught.values()) >= 4,
          str({k: bool(v) for k, v in caught.items()}))

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
    for n in ("D2","D3","D4","D5","D6","D7","D7b","D8","D9","D-DISC"):
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
