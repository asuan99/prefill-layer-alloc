"""CP-0 arm-distinctness rule.  RULE_REV = 3.

CP-0 is a MEASUREMENT stage: it returns no policy verdict, no goodput comparison,
no arm ranking.  What it decides is narrower and prior to all of that -- for each
candidate arm, IS THIS ARM MECHANISTICALLY DISTINCT FROM `fused_mono`, and on
which channel.  An arm that chunks on neither channel is `fused_mono` under
another name, and a later campaign that scored it would be scoring an identity.

Why this rule exists as its own pre-registration (2nd audit, K2/K3/L3/L8)
------------------------------------------------------------------------
The rev2 CP-1 rule could not be fixed at the rule layer, because four of its
kill-shots were unresolvable WITHOUT MEASUREMENTS THAT DO NOT EXIST YET:
  K2  `capacity` per arm is unmeasured, so leaving the axis free read ignorance
      as capability and produced a size headline the design cannot support.
  K3  `REQ_SPLIT_MIN_N = ceil(delta*N)` equated a goodput ratio with a request
      count, and the arm set turned on that mis-derivation.
  L3  the batch-budget channel is observable only at runtime, yet the arm set
      was decided without it.
  L8  the batch-channel gate demanded a paired CI with no n registered.
CP-0 buys exactly those four facts.  ★It therefore DROPS the threshold
altogether: **no arm is excluded here.**  All six boots are measured and the
arm-set decision moves to the CP-1 pre-registration, which will be written
against measured numbers instead of a derivation.

`label()` is an ordered guard list (RULES) followed by a total 2x2 outcome table,
so `cp0_selftest.py` can enumerate and mutate EVERY branch mechanically.  That is
the 2nd audit's K4: rev2's mutation list was authored by the self-test itself, and
9 of 14 externally-supplied mutants survived.

Gate citations give number + FILE + opening phrase, because this repository runs
two disjoint gate numbering schemes and rev2 attributed six CLAUDE.md gates to
PROJECT_STATUS (2nd audit K6).  PROJECT_STATUS.md:5819 states the two are
disjoint: "`CLAUDE.md`의 기존 8개 게이트에 **추가로**, 이 정본에 등재한다."
Every citation below was grepped in this session.
"""

RULE_REV = 3

# ---- constants that AFFECT the label map (the self-test must prove each is load-bearing)
LABEL_CONSTANTS = ()          # this rule has none: every branch is categorical.

# ---- campaign parameters, declared NOT to affect the label map ----------------
# Registered here so nobody mistakes the self-test for a guard on them; the
# self-test verifies that perturbing each leaves the label map identical.
N_BOOTS_PER_ARM = 3           # reps; each rep = one job = one node = all arms
CAMPAIGN_PARAMS = ("N_BOOTS_PER_ARM",)
# ★rev2 (3rd audit F4): BATCH_MARGIN_REL used to sit here as a CAMPAIGN_PARAM whose
# "does not touch the label map" check was an IDENTITY -- nothing read it, because
# the data->axis map was prose in the pre-registration.  It now lives in
# cp0_predicates.batch_axis and is mutation-tested against registered vectors there.
PREDICATE_MODULE = "cp0_predicates.py"
PREDICATE_CONSTANTS = ("BATCH_MARGIN_REL",)
PREDICATE_FOLDS = ("req_axis_fold", "batch_axis_fold")

AXES = {
    # did the server boot and serve the smoke workload?
    "boot": ["ok", "failed"],
    # realized --chunked-prefill-size vs requested.  NOTE: this is a CONFIG ECHO,
    # not a third witness -- /server_info, internal_states and the HOLB probe all
    # read the same `server_args` field (`holb_probe.py:574`).  1st audit L6.
    "echo": ["match", "mismatch", "unmeasured"],
    # channel 1: did ANY served request get split into >= 2 chunks?
    "req_channel": ["active", "inactive", "unmeasured"],
    # channel 2: does the per-batch token cap bind?  mean extend tokens per
    # prefill forward vs `fused_mono`, paired by rep.
    # produced by cp0_predicates.batch_axis(...) per probe rate, then folded by
    # cp0_predicates.batch_axis_fold(...).  ★rev3 (4th audit G6): `mixed` is a real
    # outcome -- "the batch cap binds only under backlog" is what the mechanism
    # predicts -- and collapsing it into `slack` would erase the finding.
    "batch_channel": ["binds", "mixed", "slack", "unmeasured"],
}

AXIS_ORDER = ["boot", "echo", "req_channel", "batch_channel"]

SUBSTANTIVE = [
    "ARM_DISTINCT_BOTH",
    "ARM_DISTINCT_BOTH_RATE_DEPENDENT",
    "ARM_DISTINCT_REQUEST_ONLY",
    "ARM_DISTINCT_BATCH_ONLY",
    "ARM_DISTINCT_BATCH_RATE_DEPENDENT",
    "ARM_INDISTINGUISHABLE",
]

# 2x2 outcome table over the two channels.  Total on the product of the measured
# values, so no world falls through.
OUTCOME = {
    ("active", "binds"): "ARM_DISTINCT_BOTH",
    ("active", "mixed"): "ARM_DISTINCT_BOTH_RATE_DEPENDENT",
    ("active", "slack"): "ARM_DISTINCT_REQUEST_ONLY",
    ("inactive", "binds"): "ARM_DISTINCT_BATCH_ONLY",
    ("inactive", "mixed"): "ARM_DISTINCT_BATCH_RATE_DEPENDENT",
    ("inactive", "slack"): "ARM_INDISTINGUISHABLE",
}


def incoherent(w):
    """A failed boot cannot have produced any downstream measurement."""
    if w["boot"] == "failed" and not (
        w["echo"] == "unmeasured"
        and w["req_channel"] == "unmeasured"
        and w["batch_channel"] == "unmeasured"
    ):
        return "boot failed: no downstream observable can exist"
    return None


# Ordered guards.  Each is (name, predicate, label).  Keeping them as data is what
# lets the self-test drop and negate each one mechanically (K4).
RULES = [
    # PROJECT_STATUS.md "방법론 게이트" #21 ("측정 실패를 게이트 실패로 라벨링하지
    # 마라 — 이 프로젝트의 서명 오류"): a boot that did not happen is not a finding
    # about the arm.  Both of these are NON-substantive on purpose.
    # ★A failed boot gets its OWN label, not the generic one.  The exhaustive
    # mutation test caught this: with a shared label the guard was dead code
    # (`incoherent()` already forces every downstream axis to `unmeasured`, so
    # `echo_unmeasured` caught the same worlds).  Distinguishing them is also
    # what PROJECT_STATUS.md "방법론 게이트" #21 asks for -- "the server never
    # started" and "it served but the extractor found nothing" are different
    # measurement failures and must not share a bucket.
    ("boot_failed", lambda w: w["boot"] == "failed", "BOOT_FAILED"),
    ("echo_mismatch", lambda w: w["echo"] == "mismatch", "ARM_UNUSABLE"),
    ("echo_unmeasured", lambda w: w["echo"] == "unmeasured", "ARM_UNUSABLE"),
    ("req_unmeasured", lambda w: w["req_channel"] == "unmeasured", "CHANNELS_UNMEASURED"),
    ("batch_unmeasured", lambda w: w["batch_channel"] == "unmeasured", "CHANNELS_UNMEASURED"),
]


def label(w):
    if incoherent(w):
        return "IMPOSSIBLE_WORLD"
    for _name, pred, lab in RULES:
        if pred(w):
            return lab
    return OUTCOME[(w["req_channel"], w["batch_channel"])]
