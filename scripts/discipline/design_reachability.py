#!/usr/bin/env python3
"""Design-layer reachability -- the meta-check both 2026-08-28 audits demanded.

A rule file's own `T2_reachable` asks: "does every label appear somewhere in the
enumerated GRID?"  That is a property of the axis lattice, not of the experiment.
Both audits landed the same blow on it:

  TC1 rev1 B13  "T2_reachable / T11_no_inert_axis 는 격자 내부 성질이라 이런
                 설계층 도달불가를 원리적으로 못 잡는다."
  M4R rev2 F3′  "T9는 증상만 없앴고 설계층 도달가능성 메타검사는 미채택.
                 지금 넣으면 rev2가 자기 자신을 FAIL시킨다."

This tool asks the other question: given what the REGISTERED DESIGN and ALREADY
MEASURED DATA can actually produce, which substantive labels remain reachable?
Every restriction carries a provenance string; a restriction without one is
refused, because that is how a design gets narrowed by assumption instead of by
evidence.

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


def main(spec_path):
    spec = json.load(open(spec_path))
    mod = load_rule(spec["rule_path"])
    as_tuple = spec.get("world_is_tuple", True)
    restrict = spec["restrict"]
    for a, r in restrict.items():
        if not r.get("why"):
            raise SystemExit(f"axis {a}: a restriction without provenance is refused")

    grid, _ = reachable(mod, {}, as_tuple)          # what the lattice allows
    live, dom = reachable(mod, restrict, as_tuple)  # what the design can produce

    SUB = set(mod.SUBSTANTIVE)
    grid_sub = {k: v for k, v in grid.items() if k in SUB}
    live_sub = {k: v for k, v in live.items() if k in SUB}
    lost = sorted(set(grid_sub) - set(live_sub))

    out = {
        "rule_path": spec["rule_path"], "rule_rev": getattr(mod, "RULE_REV", None),
        "label": spec.get("label", os.path.basename(spec_path)),
        "restrictions": {a: {"values": r["values"], "why": r["why"]} for a, r in restrict.items()},
        "grid_substantive": grid_sub, "design_substantive": live_sub,
        "unreachable_by_design": lost,
        "design_labels_all": live,
        "verdict": ("NOTHING_PURCHASABLE" if not live_sub else
                    "SINGLE_LABEL_FORCED" if len(live_sub) == 1 else
                    "DISCRIMINATING"),
    }
    out["reading"] = {
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
    print(f"  VERDICT: {out['verdict']}\n  {out['reading']}")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
