"""CP-0 capacity-probe rule.  RULE_REV = 2.

The second thing CP-0 buys.  The CP-1 rule needs a `capacity` value per contrast,
and the 2nd audit's K2 showed that leaving it unmeasured while treating the axis
as free turns ignorance into a size headline.  This rule does not estimate
capacity; it decides WHETHER THE REGISTERED PROBE RATES BRACKET SATURATION, which
is the precondition for reporting a capacity at all.

★rev2 (3rd audit F1).  rev1 registered `saturated == achieved < offered` with no
tolerance.  That is a NOISE INDICATOR: under Poisson pacing the benchmark duration
spans the ramp and drain tail, so it fires far below capacity -- on this repository's
own capacity scan it fires at offered rate 1.  The axis value now comes from
`cp0_predicates.rate_axis`, a coded total function on the ATTAINMENT RATIO
achieved/offered, which is monotone decreasing in every arm of that scan (achieved
throughput itself is NOT: `plain` measures 4.39 / 4.30 / 4.23 at rates 8 / 10 / 12,
and a plateau-difference predicate inverts on it).  The threshold 0.80 and the probe
grid {2, 4, 8} are derived there from measurements that already exist.  A bracket needs one
unsaturated rate below and one saturated rate above; anything else says the probe
rates were wrong, which is itself the useful answer at this stage.

★Deliberately NO restriction is registered in the accompanying spec.
★CORRECTION (4th audit L-b): an earlier draft said capacity was "entirely unmeasured
for every arm".  That was false, and the same docstring cited the measurements that
falsify it.  The repository holds non-pdmux capacity runs for three of the six arms
(`p1_gates/gate2/g2ea_*capscan_*.jsonl`: cps=-1 109 runs, cps=512 108, cps=8192 108,
9 rates x 3 seeds) -- on a DIFFERENT workload (random-ids in2000/out96, N=60), which
is why they choose the threshold and grid but cannot be transferred as capacities.
What is unmeasured is THIS workload's capacity for these arms, so there is still no
provenanced restriction to narrow the lattice with, and a restriction without
provenance is refused by the tool.  This is the
honest form of the 2nd audit's own lesson ("an unmeasured axis read as free
translates ignorance into capability"): that lesson bites a DECISION stage that
claims a verdict it cannot reach.  Here the unmeasured axis IS the deliverable --
CP-0 exists to find out which of these four outcomes holds.
"""

RULE_REV = 2

LABEL_CONSTANTS = ()          # label() is categorical; the data->axis constants
                              # live in cp0_predicates and are mutation-tested there
PREDICATE_MODULE = "cp0_predicates.py"
PREDICATE_CONSTANTS = ("ATTAINMENT_THRESHOLD",)
CAMPAIGN_PARAMS = ()

AXES = {
    "boot": ["ok", "failed"],
    # values produced by cp0_predicates.rate_axis(achieved, offered)
    "lo_rate": ["tracking", "saturated", "unmeasured"],
    "hi_rate": ["tracking", "saturated", "unmeasured"],
}

AXIS_ORDER = ["boot", "lo_rate", "hi_rate"]

SUBSTANTIVE = [
    "CAPACITY_BRACKETED",        # lo unsaturated, hi saturated -- probe was right
    "SATURATED_AT_FLOOR",        # even the lowest probe rate saturates -- rates too high
    "NO_SATURATION_IN_RANGE",    # even the highest does not -- rates too low
    "NONMONOTONE_PROBE",         # lo saturated but hi not -- anomaly, not a capacity
]


def incoherent(w):
    if w["boot"] == "failed" and not (w["lo_rate"] == "unmeasured" and w["hi_rate"] == "unmeasured"):
        return "boot failed: no probe point can exist"
    return None


RULES = [
    # Own label -- see the note in cp0_arm_rule.py RULES; the exhaustive mutation
    # test proved the shared-label version was a dead branch.
    ("boot_failed", lambda w: w["boot"] == "failed", "BOOT_FAILED"),
    ("lo_unmeasured", lambda w: w["lo_rate"] == "unmeasured", "CAPACITY_UNMEASURED"),
    ("hi_unmeasured", lambda w: w["hi_rate"] == "unmeasured", "CAPACITY_UNMEASURED"),
]

OUTCOME = {
    ("tracking", "saturated"): "CAPACITY_BRACKETED",
    ("saturated", "saturated"): "SATURATED_AT_FLOOR",
    ("tracking", "tracking"): "NO_SATURATION_IN_RANGE",
    # The attainment ratio is monotone decreasing in rate across every arm of the
    # reference scan, so a low rate saturated while a high rate tracks contradicts
    # the invariant the axis is built on: an anomaly, not a capacity.
    ("saturated", "tracking"): "NONMONOTONE_PROBE",
}


def label(w):
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in RULES:
        if pred(w):
            return lab
    return OUTCOME[(w["lo_rate"], w["hi_rate"])]
