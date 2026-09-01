"""G1 probe decision rule -- is the registered `saturated` axis IDENTIFIABLE?  RULE_REV = 1.

The 4th audit's G1 killed CP-0's capacity axis on an arithmetic point, not an
opinion: the knee band from which `ATTAINMENT_THRESHOLD = 0.80` was derived is
**0.09 wide**, and the no-load seed variability of that same statistic is
**SD 0.108** (offered rate 1, 60/60 completed, three seeds: attainment 1.075 /
0.957 / 0.859 -- and one of them EXCEEDS 1).  A threshold whose derivation band is
narrower than its own noise does not identify a knee.

Two further facts made the axis unbuyable rather than merely noisy:
  * the derivation came from a DIFFERENT workload (`random-ids in2000/out96`,
    ~2096 tokens per request) while CP-0 runs ShareGPT (~578), so the grid {2,4,8}
    is an absolute-req/s transfer across a 3.6x lighter load (audit G2);
  * the canonical harness fixes `--seed 1`, so CP-0's three reps are three copies
    of ONE arrival realization and a paired CI across them cannot contain this
    variance at all.

This rule scores the probe that buys the missing numbers.  It decides ONE thing --
whether the axis is identifiable on THIS workload -- and no policy question.  It
ranks no arm, produces no goodput, and its PASS licenses nothing about chunked
prefill (see the pre-registration's forbidden sentences).

Gate citations give number + FILE + phrase (PROJECT_STATUS.md and CLAUDE.md run
two disjoint schemes; PROJECT_STATUS.md:5819).
"""

RULE_REV = 1

# ---- what the probe must produce for the axes to be readable ---------------
N_SEEDS = 3                  # arrival realizations, NOT reps of one realization
SEEDS = (1, 7, 17)           # 1 is the canonical harness seed; 7/17 follow the
                             # repository's own capscan seed convention
RATES = (2, 3, 4, 6, 8)      # brackets the registered grid {2,4,8} and adds the
                             # two points where the reference scan's knees sat
ARMS = ("fused_default", "cp512")
CAMPAIGN_PARAMS = ("N_SEEDS", "SEEDS", "RATES", "ARMS")
LABEL_CONSTANTS = ()

AXES = {
    # did every (arm, rate) cell get all N_SEEDS realizations?
    "coverage": ["complete", "partial", "absent"],
    # width of the knee band -- the attainment gap between the last rate that is
    # still climbing and the first rate at the plateau -- against the measured
    # per-cell seed SD of attainment.  This is the 4th audit's arithmetic, applied
    # to this workload instead of borrowed from another one.
    "band_vs_sd": ["band_wider", "band_narrower", "no_knee_in_range"],
    # does the REGISTERED grid {2,4,8} still bracket the knee on this workload?
    "grid_brackets": ["yes", "no"],
}

AXIS_ORDER = ["coverage", "band_vs_sd", "grid_brackets"]

SUBSTANTIVE = [
    "AXIS_IDENTIFIED",          # band wider than noise AND the registered grid brackets
    "AXIS_IDENTIFIED_REGRID",   # band wider than noise but the grid must move
    "AXIS_NOT_IDENTIFIED",      # band narrower than noise -- the 4th audit's prediction
    "NO_KNEE_IN_RANGE",         # capacity sits outside the swept rates entirely
]

RULES = [
    ("coverage_absent", lambda w: w["coverage"] == "absent", "NOT_MEASURED"),
    ("coverage_partial", lambda w: w["coverage"] == "partial", "COVERAGE_INCOMPLETE"),
]

OUTCOME = {
    ("band_wider", "yes"): "AXIS_IDENTIFIED",
    ("band_wider", "no"): "AXIS_IDENTIFIED_REGRID",
    ("band_narrower", "yes"): "AXIS_NOT_IDENTIFIED",
    ("band_narrower", "no"): "AXIS_NOT_IDENTIFIED",
    ("no_knee_in_range", "yes"): "NO_KNEE_IN_RANGE",
    ("no_knee_in_range", "no"): "NO_KNEE_IN_RANGE",
}


def incoherent(w):
    """`no_knee_in_range` means no knee was located, so 'does the grid bracket the
    knee' has no referent.  Registering the pair as impossible keeps the grid from
    counting worlds the probe cannot produce."""
    if w["band_vs_sd"] == "no_knee_in_range" and w["grid_brackets"] == "yes":
        return "no knee was located, so the grid cannot be said to bracket it"
    return None


def label(w):
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in RULES:
        if pred(w):
            return lab
    return OUTCOME[(w["band_vs_sd"], w["grid_brackets"])]
