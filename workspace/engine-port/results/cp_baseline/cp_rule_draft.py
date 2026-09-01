"""SUPERSEDED (2026-08-28) -- DO NOT PRE-REGISTER, DO NOT CITE.  History only.

claims-auditor returned NO-GO on this rule layer (9 kill-shots, 12 local defects):
`audit_cp_rules_2026-08-28/VERDICT.md`.  The live rule is `cp_rule.py` (RULE_REV=2).

The three defects that mattered most, all confirmed independently:
  D1  `ci_shape` is claimed below to be "exhaustive and mutually exclusive by
      construction".  It is neither: the axis values are prose predicates with no
      data->axis map, and a CI of the shape [+0.068, +0.108] against delta=0.0966
      (this project's most-cited effect, CONSENSUS 1-7) falls through every value.
  D2  the decision quantity contains a post-hoc argmax over three arms that is not
      an axis, so the reachability certificate cannot see it.
  D7  the CONSENSUS citation is wrong in both the section and the value
      (1-33 / 2.15-2.36x; canon is 1-32 / 2.15-2.35x).
"""

"""CP (chunked-prefill) baseline campaign -- DRAFT decision rule, rules layer only.

STATUS: RULE_REV=1, **NOT a pre-registration**.  Methodology gate #34 requires the
rules layer to pass an adversarial audit BEFORE any harness is written or any job is
submitted.  This file exists so that (a) the decision rule is fixed in code rather
than prose (gate #66) and (b) `scripts/discipline/design_reachability.py` can be run
on it now, at zero GPU cost, as the REACHABILITY_FINDING_2026-08-28 prescription #2
demands.

Decision quantity
-----------------
    D = goodput(best CP arm) - goodput(d44)      paired by rep, canonical
        varying trace, conjunctive goodput (TTFT <= SLO AND intra-request
        token-ITL p95 <= SLO), duration SUMMED over rounds (gate #7).
    delta = 0.03 * goodput(d44)                  project convention (gate #3)

`ci_shape` is the position of the paired CI of D relative to {0, +/-delta}.  It is
exhaustive and mutually exclusive by construction, so no world falls through.

Two positive-control channels, because `--chunked-prefill-size N` acts on two
different things and the per-request channel dies first:
    req_split     -- fraction of requests split into >=2 chunks >= 1% (registered
                     minimum).  Measured offline for the arm-set choice
                     (`sharegpt_prompt_lens.json`) and at runtime in Stage CP-0.
    batch_budget  -- mean extend tokens per prefill forward differs from the fused
                     arm by >= the registered margin, i.e. the batch token cap binds.
An arm where BOTH are absent is not a chunked-prefill arm at all; it is the fused
arm under another name, and scoring it would be scoring an identity.
"""

RULE_REV = 1

AXES = {
    # positive control channel 1: does any single request get split?
    "req_split": ["yes", "no"],
    # positive control channel 2: does the per-batch token cap bind?
    "batch_budget": ["binds", "slack"],
    # offered HI rate vs the arm's own MEASURED achieved capacity (gate #6).
    "capacity": ["above", "below"],
    # paired CI of D relative to 0 and +/-delta.
    "ci_shape": [
        "pos_ge_delta",   # CI entirely above +delta   -> CP better by >= delta
        "pos_lt_delta",   # CI excludes 0, upper < +delta
        "within_delta",   # CI contains 0 and CI subset of (-delta, +delta) -> equivalence
        "neg_lt_delta",   # CI excludes 0, lower > -delta
        "neg_ge_delta",   # CI entirely below -delta   -> CP worse by >= delta
        "straddles",      # CI wider than +/-delta     -> cannot separate
    ],
    # sign of D held across the full SLO ladder TTFT{1.0,3.0}s x ITLp95{50,60,80}ms
    "ladder": ["stable", "flips"],
}

AXIS_ORDER = ["req_split", "batch_budget", "capacity", "ci_shape", "ladder"]

SUBSTANTIVE = [
    "CP_WINS_RANK_ONLY", "CP_WINS_SIZED",
    "CP_LOSES_RANK_ONLY", "CP_LOSES_SIZED",
    "CP_EQUIV_RANK_ONLY", "CP_EQUIV_SIZED",
    "LADDER_UNSTABLE",
]


def label(w):
    # 1. Arm identity first.  No chunking on either channel => not a CP arm.
    if w["req_split"] == "no" and w["batch_budget"] == "slack":
        return "DEGENERATE_ARM"

    # 2. Metric stability before policy verdict.  A sign that flips across the
    #    ladder is a statement about the SLO threshold, not about the policy --
    #    and it is exactly what an arm-dependent ITL cliff would produce
    #    (CONSENSUS §3 item 89: intra-request ITL p95 is bimodal at 58-60 ms
    #    under monolithic prefill, and CP is expected to move that mode).
    if w["ladder"] == "flips":
        return "LADDER_UNSTABLE"

    # 3. Power.  A CI wider than +/-delta separates nothing.  Reported as its own
    #    label rather than folded into an equivalence claim, so that noise cannot
    #    be cashed in as "no difference" (gate #20, null-acceptance).
    if w["ci_shape"] == "straddles":
        return "UNDERPOWERED"

    if w["ci_shape"] in ("pos_ge_delta", "pos_lt_delta"):
        rank = "CP_WINS"
    elif w["ci_shape"] in ("neg_ge_delta", "neg_lt_delta"):
        rank = "CP_LOSES"
    else:
        rank = "CP_EQUIV"

    # 4. Above the arm's achieved capacity the threshold-goodput MAGNITUDE is
    #    ill-posed (gate #6; CONSENSUS §1-33 records the canonical HI phase at
    #    2.15-2.36x achieved capacity).  The ORDER survives -- canon cites the
    #    HE0 ranking from exactly that operating point -- so capacity gates the
    #    size claim only, never the whole verdict.  Letting it gate everything
    #    would repeat the NSL D4 failure (an axis removal that pre-decides the
    #    answer).
    return rank + ("_RANK_ONLY" if w["capacity"] == "above" else "_SIZED")
