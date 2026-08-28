#!/usr/bin/env python3
"""Design-layer reachability -- the meta-check the 2026-08-28 audits demanded.

TOOL_REV = 2.  rev1 of this tool FAILED ITS OWN REQUESTER: the TC1 rev3 audit
showed that rev1 passed a spec whose restrictions did nothing at all.  With
ctrl_H pinned to any of blocked / mispositioned / reaches, the rule's label
function never distinguishes them, so `design_sub` came out as EXACTLY
`grid_sub / 3` -- a uniform rescaling.  rev1 reported `unreachable_by_design: []`
and DISCRIMINATING, which reads as "nothing was lost" when it meant "the
restriction was inert".  Three repairs, all named by that audit:

  (a) RESTRICTIONS_INERT -- a non-empty restriction that only rescales the
      substantive distribution is reported as doing nothing, not as passing.
  (b) supersedes / dropped_restrictions -- a spec that drops an axis its
      predecessor restricted must say why.  TC1 rev3's spec silently dropped
      sign_H, the one restriction that had decided rev2's verdict.
  (c) priors -- a spec must register the prior it assigns each substantive
      label.  Reachable-in-principle and reachable-in-practice are different
      claims, and rev3's headline carried prior ~.000 under two of three priors
      the auditor tried.

A rule file's own `T2_reachable` asks whether every label appears somewhere in
the enumerated GRID -- a property of the axis lattice, not of the experiment.
This tool asks the other question: given what the REGISTERED DESIGN and ALREADY
MEASURED DATA can produce, which substantive labels remain reachable?  Every
restriction carries a provenance string; one without it is refused.

★KNOWN LIMIT, stated because the audit found it: this tool sees the CATEGORICAL
lattice only.  rev2's forcing happened to be categorical (visits = no) so it was
caught; rev3's forcing lives in the CONTINUOUS layer (se, thresholds, TOST
margin, priors) and it was NOT.  A DISCRIMINATING verdict here does not mean the
design can produce a verdict -- only that the categories do not forbid it.

Usage:  python3 design_reachability.py <spec.json>
"""
import importlib.util, json, os, sys
from itertools import product


def load_rule(path):
    spec = importlib.util.spec_from_file_location("rule_under_test", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rule_under_test"] = m
    spec.loader.exec_module(m)
    return m


def reachable(mod, restrict, as_tuple):
    """restrict: axis -> {"values": [...], "why": "..."}"""
    axes = mod.AXES
    order = mod.AXIS_ORDER
    dom = {}
    for a in order:
        if a in restrict:
            vals = restrict[a]["values"]
            bad = [v for v in vals if v not in axes[a]]
            if bad:
                raise SystemExit(f"axis {a}: unknown values {bad}")
            dom[a] = vals
        else:
            dom[a] = list(axes[a])
    seen = {}
    for combo in product(*(dom[a] for a in order)):
        w = combo if as_tuple else dict(zip(order, combo))
        lab = mod.label(w)
        seen.setdefault(lab, 0)
        seen[lab] += 1
    return seen, dom


def _inert(grid_sub, live_sub, tol=1e-9):
    """(a) did the restriction only RESCALE the substantive distribution?"""
    if not live_sub or set(live_sub) != set(grid_sub):
        return False
    ratios = [live_sub[k] / grid_sub[k] for k in live_sub]
    return max(ratios) - min(ratios) < tol


def main(spec_path):
    spec = json.load(open(spec_path))
    mod = load_rule(spec["rule_path"])
    as_tuple = spec.get("world_is_tuple", True)
    restrict = spec["restrict"]
    for a, r in restrict.items():
        if not r.get("why"):
            raise SystemExit(f"axis {a}: a restriction without provenance is refused")

    findings = []

    # (b) a spec that narrows a predecessor must account for every axis it dropped
    prev_path = spec.get("supersedes")
    if prev_path:
        pp = prev_path if os.path.isabs(prev_path) else os.path.join(
            os.path.dirname(os.path.abspath(spec_path)), prev_path)
        prev = json.load(open(pp))
        dropped = set(prev.get("restrict", {})) - set(restrict)
        explained = set(spec.get("dropped_restrictions", {}))
        unexplained = sorted(dropped - explained)
        if unexplained:
            findings.append(("RESTRICTION_DROPPED_UNEXPLAINED",
                             f"axes restricted by {os.path.basename(pp)} and dropped here "
                             f"without a reason: {unexplained}"))

    grid, _ = reachable(mod, {}, as_tuple)          # what the lattice allows
    live, dom = reachable(mod, restrict, as_tuple)  # what the design can produce

    SUB = set(mod.SUBSTANTIVE)
    grid_sub = {k: v for k, v in grid.items() if k in SUB}
    live_sub = {k: v for k, v in live.items() if k in SUB}
    lost = sorted(set(grid_sub) - set(live_sub))

    # (a) inert restrictions
    if restrict and _inert(grid_sub, live_sub):
        findings.append(("RESTRICTIONS_INERT",
                         "every substantive label survived at the SAME ratio "
                         f"({list(live_sub.values())[0]}/{list(grid_sub.values())[0]}); the "
                         "restriction rescaled the lattice without excluding anything"))

    # (c) reachable-in-principle is not reachable-in-practice
    priors = spec.get("priors")
    if not priors:
        findings.append(("PRIOR_UNREGISTERED",
                         "no per-label prior registered; a label reachable in the lattice "
                         "may still have prior ~0 under the design's own measured scale"))
    else:
        missing = sorted(set(live_sub) - set(priors))
        if missing:
            findings.append(("PRIOR_UNREGISTERED", f"no prior for {missing}"))

    out = {
        "tool_rev": 2, "findings": [{"code": c, "detail": d} for c, d in findings],
        "rule_path": spec["rule_path"], "rule_rev": getattr(mod, "RULE_REV", None),
        "label": spec.get("label", os.path.basename(spec_path)),
        "restrictions": {a: {"values": r["values"], "why": r["why"]} for a, r in restrict.items()},
        "grid_substantive": grid_sub, "design_substantive": live_sub,
        "unreachable_by_design": lost,
        "design_labels_all": live,
        "verdict": ("NOTHING_PURCHASABLE" if not live_sub else
                    "SINGLE_LABEL_FORCED" if len(live_sub) == 1 else
                    findings[0][0] if findings else
                    "DISCRIMINATING"),
    }
    out["reading"] = {
        "RESTRICTIONS_INERT": "The restrictions excluded nothing. This run could not have "
                              "failed, so its DISCRIMINATING would have carried no evidence.",
        "RESTRICTION_DROPPED_UNEXPLAINED": "A predecessor spec restricted an axis this one "
                                           "drops without saying why. Dropping the restriction "
                                           "that decided the previous verdict is how a design "
                                           "passes by forgetting.",
        "PRIOR_UNREGISTERED": "Reachable in the lattice is not reachable in practice. Register "
                              "the prior each substantive label carries at the design's own "
                              "measured scale.",
        "NOTHING_PURCHASABLE": "No substantive label is reachable. Running the campaign "
                               "cannot return a verdict; the spend buys nothing.",
        "SINGLE_LABEL_FORCED": "Exactly one substantive label is reachable, so the verdict "
                               "is fixed before the data arrive. This is the gate-#40 "
                               "failure at the design layer rather than the algebra layer.",
        "DISCRIMINATING": "More than one substantive label is reachable; the design can "
                          "still be informative, subject to the other audit findings.",
    }[out["verdict"]]
    p = spec.get("out", os.path.splitext(spec_path)[0] + "_reachability.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False, sort_keys=True)

    print(f"=== {out['label']}  (RULE_REV={out['rule_rev']}) ===")
    for a, r in out["restrictions"].items():
        print(f"  restrict {a:12} -> {r['values']}")
        print(f"           {'':12}    ({r['why']})")
    print(f"  grid   substantive: {grid_sub}")
    print(f"  DESIGN substantive: {live_sub or '{} — none'}")
    if lost: print(f"  unreachable by design: {lost}")
    for c, d in findings:
        print(f"  FINDING  {c}: {d}")
    print(f"  VERDICT: {out['verdict']}\n  {out['reading']}")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
