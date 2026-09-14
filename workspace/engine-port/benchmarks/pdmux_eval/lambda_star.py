"""Per-SHAPE lambda* (sustainable request rate) for the R2 campaign.

WHY THIS MODULE EXISTS
----------------------
Every campaign workload is defined as a FRACTION of lambda* (W1 0.60, W3 0.80,
W8 0.90, W9 1.10, W2 burst 0.80, W4 0.80 per phase).  Until 2026-09-13 the whole
campaign took ONE scalar -- `PDMUX_SUSTAINABLE_RATE`, default 4 req/s, never
measured on any model -- and applied it to all nine workloads, whose request
shapes differ by 32x in input length.  Two consequences, both established, not
re-derived here:

  (1) Gate #6 ("measure capacity first") was violated by construction: with the
      default, "W8 = near saturation" and "W9 = overload" were assertions.
  (2) W4 became SELF-CONTRADICTORY.  Its two phases are (8192, 64) and
      (256, 512), whose capacities differ ~5x, so NO single scalar puts both at
      0.80x: lambda*=0.675 gives prefill 0.79x but decode 0.21-0.25x;
      lambda*=2.1 gives decode 0.79x but prefill 2.47x.
      (`results/r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`
      section 3(1a).)

The user decision of 2026-09-13 on `reports/paper/EXPERIMENT_ROADMAP.md` "P2"
was option (a): revise the W4 definition to use an INDEPENDENT lambda* PER
PHASE.  This module is that revision's input: a table keyed by request shape.

WHAT A VALUE IN THIS TABLE IS AND IS NOT
----------------------------------------
* It is NOT produced here.  This module only loads, validates and records.
* `source` is mandatory per shape and is the whole point of the file.  A shape
  whose lambda* has not been measured for THIS (model, backend, context,
  operating point) must say `"unmeasured"`, and a campaign built on an
  unmeasured shape is refused unless the caller opts in explicitly.
* `definition` is mandatory because the two candidate definitions differ by
  1.7-3.4x and the project canon and the only existing measurement disagree
  about which one is meant:
    - `slo_sustainable`     -- the canon's "B1 sustainable SLO rate"
                               (EXPERIMENT_ROADMAP.md "common method").  NOT
                               MEASURED for any shape as of 2026-09-13.
    - `throughput_saturation` -- the knee of achieved vs offered rate.  This is
                               what job 905835 measured, and re-scoring its d44
                               cells with the canonical goodput predicate gives
                               53.8% at 0.59x, so lambda*_SLO < 0.59 x
                               lambda*_throughput.  Declaring
                               `throughput_saturation` therefore does NOT close
                               gate #6.

NAMING CAVEAT (verdict citation-stop Q4, carried verbatim into the schema)
--------------------------------------------------------------------------
lambda* is an ARM-SPECIFIC quantity: job 905835 measured 0.933 / 0.675 / 0.187
req/s at D16 / D44 / D92 -- a 5x spread.  Workload names ("W8 near saturation",
"W9 overload") are true at most for the arm whose lambda* was injected (B1).
The SAME trace may be 0.2x capacity for another arm.  Nothing in this module
makes those names true for any other arm, and this file must not be cited as if
it did.

THIS FILE MAKES NO PERFORMANCE CLAIM AND NO MEASUREMENT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Shape = Tuple[int, int]

SCHEMA = "pdmux.lambda_star/v1"
DEFINITIONS = ("slo_sustainable", "throughput_saturation")
SOURCES = ("measured", "unmeasured")

# Every rule that turns per-shape lambda* into an arrival rate.  Recorded next
# to the numbers so a finished run says HOW the numbers were combined, not only
# what they were.
RATE_DERIVATIONS = (
    # one shape, one Poisson stream: rate = fraction x lambda*(that shape)
    "single_shape",
    # W4: each phase gets fraction x lambda*(ITS OWN shape).  This is the
    # 2026-09-13 revision.
    "per_phase",
    # W5/W6 interleave two shapes inside ONE Poisson stream, so no per-shape
    # rate exists for them.  `mixture_rate` below applies a work-conservation
    # DEFINITION (not a measurement, and not pre-registered) to get one stream
    # rate.  It is reported as a caveat so it can never be used silently.
    "interleaved_mixture_harmonic",
)

FAIL_CLOSED_MESSAGE = (
    "no per-shape lambda* table was supplied.  lambda* is measured in campaign "
    "stage 0 (lambda*는 캠페인 0단계에서 측정된다) and has NOT been measured for "
    "this model/backend/context; the campaign generator therefore refuses to "
    "invent one.  Pass --lambda-star-table <file> (or PDMUX_LAMBDA_STAR_TABLE) "
    "with a table of the form documented in "
    "benchmarks/pdmux_eval/lambda_star.py.  The pre-2026-09-13 single scalar "
    "PDMUX_SUSTAINABLE_RATE (default 4 req/s, never measured) is gone on "
    "purpose: it violated gate #6 and made W4 self-contradictory."
)


def shape_key(shape: Shape) -> str:
    return f"{int(shape[0])}x{int(shape[1])}"


def parse_shape_key(text: str) -> Shape:
    parts = str(text).split("x")
    if len(parts) != 2:
        raise ValueError(f"shape key must be '<input>x<output>', got {text!r}")
    inp, out = (int(part) for part in parts)
    if inp <= 0 or out <= 0:
        raise ValueError(f"shape {text!r} must have positive token counts")
    return (inp, out)


@dataclass(frozen=True)
class ShapeLambda:
    """One measured-or-not sustainable rate for one (input, output) shape."""

    req_per_s: float
    source: str
    evidence: str

    def as_json(self) -> Dict[str, object]:
        return {
            "req_per_s": self.req_per_s,
            "source": self.source,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class LambdaStar:
    definition: str
    measured_by: str
    entries: Mapping[Shape, ShapeLambda]

    # -- reads -------------------------------------------------------------
    def rate(self, shape: Shape) -> float:
        entry = self.entries.get(tuple(shape))  # type: ignore[arg-type]
        if entry is None:
            raise KeyError(
                f"no lambda* for shape {shape_key(shape)}; the table declares "
                f"{sorted(shape_key(s) for s in self.entries)}.  "
                + FAIL_CLOSED_MESSAGE
            )
        return entry.req_per_s

    def source(self, shape: Shape) -> str:
        self.rate(shape)  # raises the same fail-closed error when absent
        return self.entries[tuple(shape)].source  # type: ignore[index]

    def mixture_rate(self, shapes: Sequence[Shape]) -> float:
        """Stream rate for a request sequence that interleaves shapes.

        DEFINITION, NOT A MEASUREMENT.  Serving one request of shape s consumes
        1/lambda*(s) of the arm's capacity-seconds, so a stream whose requests
        are shape s with probability p_s saturates at 1 / sum(p_s/lambda*(s)).
        With a single shape this reduces exactly to lambda*(that shape), which
        is why W1/W3/W7/W8/W9 are unaffected by it.  It has NOT been through
        pre-registration or a rules-layer audit, so every workload that needs it
        carries the `undecided_rate_derivation` caveat below.
        """
        if not shapes:
            raise ValueError("mixture_rate needs at least one shape")
        demand = sum(1.0 / self.rate(shape) for shape in shapes) / len(shapes)
        return 1.0 / demand

    # -- provenance --------------------------------------------------------
    def caveats(self, shapes: Iterable[Shape], rate_derivation: str) -> list:
        """Every reason this parameterisation is not a measured one."""
        if rate_derivation not in RATE_DERIVATIONS:
            raise ValueError(f"unknown rate derivation {rate_derivation!r}")
        reasons = []
        for shape in sorted(set(tuple(s) for s in shapes)):
            if self.source(shape) != "measured":
                reasons.append(f"unmeasured_lambda_star:{shape_key(shape)}")
        if self.definition != "slo_sustainable":
            # Gate #6 asks for capacity against the metric cliff; a throughput
            # knee does not locate it (verdict section 3(1b)).
            reasons.append(f"lambda_star_definition:{self.definition}")
        if rate_derivation == "interleaved_mixture_harmonic":
            reasons.append(f"undecided_rate_derivation:{rate_derivation}")
        return reasons

    def record(self, shapes: Iterable[Shape], rate_derivation: str) -> Dict[str, object]:
        """The block stamped into a trace header and into campaign.json.

        `source` is the ROLLUP the runners gate on: "measured" only when there
        is nothing to caveat.  The reasons stay in `caveats` so that refusing
        and disclosing are the same act.
        """
        used = sorted(set(tuple(s) for s in shapes))
        reasons = self.caveats(used, rate_derivation)
        return {
            "schema": SCHEMA,
            "definition": self.definition,
            "measured_by": self.measured_by,
            "rate_derivation": rate_derivation,
            "source": "measured" if not reasons else "unmeasured",
            "caveats": reasons,
            "shapes": {
                shape_key(shape): self.entries[shape].as_json() for shape in used
            },
        }


def coerce(value: object) -> LambdaStar:
    """Accept a LambdaStar; refuse the pre-2026-09-13 scalar, loudly."""
    if isinstance(value, LambdaStar):
        return value
    if value is None or isinstance(value, (int, float)):
        raise TypeError(
            f"a single sustainable rate ({value!r}) no longer parameterises the "
            "campaign: the two W4 phases (8192,64) and (256,512) differ ~5x in "
            "capacity, so no scalar puts both at 0.80x.  " + FAIL_CLOSED_MESSAGE
        )
    raise TypeError(f"expected a LambdaStar table, got {type(value).__name__}")


#: JSON has no comments, so exactly one documentation key is tolerated and
#: ignored.  Anything else unknown is an error -- a typo must not fall back to a
#: default, because that is how an unmeasured rate would pass as measured.
COMMENT_KEY = "_comment"


def _require_keys(mapping: Mapping[str, object], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed) - {COMMENT_KEY})
    if unknown:
        # A typo'd key must not fall back to a default -- that is how an
        # unmeasured rate would pass as measured.
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed {list(allowed)}")
    missing = sorted(set(allowed) - set(mapping))
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")


def parse(document: Mapping[str, object], where: str = "<table>") -> LambdaStar:
    _require_keys(document, ("schema", "definition", "measured_by", "shapes"), where)
    if document["schema"] != SCHEMA:
        raise ValueError(f"{where}: schema must be {SCHEMA}, got {document['schema']!r}")
    definition = document["definition"]
    if definition not in DEFINITIONS:
        raise ValueError(
            f"{where}: definition must be one of {list(DEFINITIONS)}, got {definition!r}"
        )
    shapes = document["shapes"]
    if not isinstance(shapes, Mapping) or not shapes:
        raise ValueError(f"{where}: 'shapes' must be a non-empty object")
    entries: Dict[Shape, ShapeLambda] = {}
    for key, raw in shapes.items():
        shape = parse_shape_key(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: shape {key} must map to an object")
        _require_keys(raw, ("req_per_s", "source", "evidence"), f"{where}: shape {key}")
        rate = float(raw["req_per_s"])
        if not rate > 0:
            raise ValueError(f"{where}: shape {key} needs req_per_s > 0, got {rate}")
        if raw["source"] not in SOURCES:
            raise ValueError(
                f"{where}: shape {key} source must be one of {list(SOURCES)}, "
                f"got {raw['source']!r}"
            )
        evidence = str(raw["evidence"]).strip()
        if not evidence:
            raise ValueError(
                f"{where}: shape {key} needs non-empty 'evidence' (which job / "
                "pre-registration / analyzer produced this number, or why it is "
                "unmeasured)"
            )
        entries[shape] = ShapeLambda(rate, str(raw["source"]), evidence)
    return LambdaStar(str(definition), str(document["measured_by"]), entries)


def load(path: Path) -> LambdaStar:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no lambda* table at {path}.  " + FAIL_CLOSED_MESSAGE)
    return parse(json.loads(path.read_text(encoding="utf-8")), str(path))
