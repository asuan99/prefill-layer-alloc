"""Versioned offline profiles and conservative decode-floor estimation.

The profile is deliberately independent of SGLang so that profiles can be
validated and evaluated on login/CI nodes.  Measurements are full-model decode
latencies collected at the same CUDA-Graph/backend operating point as serving;
isolated layer timings are metadata only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROFILE_SCHEMA = "pdmux.hybrid-model-profile/v1"


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Operating point a profile was measured at / is being used at.

    `engine_commit` is PROVENANCE ONLY (repo HEAD of the driving checkout); it
    is deliberately NOT a compatibility axis, because a docs-only commit moves
    it without changing a single instruction the decode step executes.  The
    compatibility axis is `engine_source_hash` -- a content hash of the engine
    sources that run a decode step (see `engine_source_hash()` at the end of
    this module).  Empty means "unknown", which `is_compatible` treats as
    INCOMPATIBLE (fail-closed), never as a match.
    """

    engine_commit: str
    gpu_name: str
    gpu_sm_count: int
    cuda_driver: str
    attention_backend: str
    cuda_graph: bool
    piecewise_cuda_graph: bool = False
    engine_source_hash: str = ""


@dataclass(frozen=True)
class DecodeLatencyPoint:
    decode_sms: int
    batch_size: int
    context_tokens: int
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    repeats: int
    measured_steps: int
    residual_p95_ms: float = 0.0

    def validate(self) -> None:
        if min(self.decode_sms, self.batch_size, self.context_tokens) <= 0:
            raise ValueError("decode_sms, batch_size and context_tokens must be positive")
        if self.repeats < 1 or self.measured_steps < 1:
            raise ValueError("profile points require at least one repeat and step")
        if not 0 <= self.itl_p50_ms <= self.itl_p95_ms <= self.itl_p99_ms:
            raise ValueError("ITL percentiles must be non-negative and ordered")


@dataclass
class HybridModelProfileV1:
    model_id: str
    model_revision: str
    model_config_hash: str
    environment: RuntimeEnvironment
    attention_layers: int
    ssm_layers: int
    attention_kind: str
    num_attention_heads: int
    num_kv_heads: int
    parameter_count: int
    points: List[DecodeLatencyPoint]
    attention_decode_curve: List[Dict[str, float]] = field(default_factory=list)
    ssm_decode_curve: List[Dict[str, float]] = field(default_factory=list)
    heldout_residual_p95_ms: float = 0.0
    notes: str = ""
    schema: str = PROFILE_SCHEMA

    @property
    def layer_count(self) -> int:
        return self.attention_layers + self.ssm_layers

    @property
    def attention_ratio(self) -> float:
        return self.attention_layers / max(1, self.layer_count)

    @property
    def kv_head_ratio(self) -> float:
        return self.num_kv_heads / max(1, self.num_attention_heads)

    def validate(self) -> None:
        if self.schema != PROFILE_SCHEMA:
            raise ValueError(f"unsupported profile schema: {self.schema!r}")
        if not self.model_id or not self.model_config_hash:
            raise ValueError("model_id and model_config_hash are required")
        if self.layer_count <= 0:
            raise ValueError("profile must describe at least one model layer")
        if self.attention_kind not in {"GQA", "MQA", "MHA", "NONE", "MIXED"}:
            raise ValueError(f"invalid attention_kind: {self.attention_kind}")
        if not self.points:
            raise ValueError("profile contains no latency measurements")
        for point in self.points:
            point.validate()

    def is_compatible(self, runtime: RuntimeEnvironment) -> Tuple[bool, List[str]]:
        """Return strict compatibility and human-readable mismatch reasons.

        Engine identity is compared on `engine_source_hash` (engine SOURCE
        CONTENT), not on `engine_commit` (repo HEAD): a docs-only commit must
        not invalidate a profile, and a one-byte change in an engine source
        must.  `engine_commit` stays in the record as provenance.

        FAIL-CLOSED.  A profile written before the hash axis existed
        (2026-09-12) carries an empty hash; so does a runtime where the hash
        could not be computed.  Either one is reported as a mismatch with an
        explicit reason, so the estimator falls back
        (`fallback=True`, `upper_bound_itl_ms=inf`) instead of silently
        treating "unknown == unknown" as a match.
        """
        reasons: List[str] = []
        for name in (
            "gpu_name",
            "gpu_sm_count",
            "attention_backend",
            "cuda_graph",
            "piecewise_cuda_graph",
        ):
            if getattr(self.environment, name) != getattr(runtime, name):
                reasons.append(
                    f"{name}: profile={getattr(self.environment, name)!r}, "
                    f"runtime={getattr(runtime, name)!r}"
                )
        profile_hash = (self.environment.engine_source_hash or "").strip()
        runtime_hash = (getattr(runtime, "engine_source_hash", "") or "").strip()
        if not profile_hash:
            reasons.append(
                "engine_source_hash: missing in profile (written before the "
                "engine-source axis existed, or built without it) -- "
                "fail-closed; rebuild the profile on the engine it measures"
            )
        if not runtime_hash:
            reasons.append(
                "engine_source_hash: could not be computed for the running "
                f"engine (modules {', '.join(ENGINE_SOURCE_MODULES)}) -- "
                "fail-closed"
            )
        if profile_hash and runtime_hash and profile_hash != runtime_hash:
            reasons.append(
                f"engine_source_hash: profile={profile_hash}, "
                f"runtime={runtime_hash}"
            )
        return not reasons, reasons

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HybridModelProfileV1":
        data = dict(raw)
        data["environment"] = RuntimeEnvironment(**data["environment"])
        data["points"] = [DecodeLatencyPoint(**point) for point in data["points"]]
        profile = cls(**data)
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: Path) -> "HybridModelProfileV1":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DecodeFloorEstimate:
    decode_sms: int
    predicted_itl_ms: float
    upper_bound_itl_ms: float
    safety_margin_ms: float
    confidence: float
    fallback: bool
    reason: str


class ConservativeDecodeFloorEstimator:
    """Piecewise-linear, conservative estimator over an operational profile.

    For each decode-SM state, the estimator interpolates p95 ITL over batch and
    context.  It never extrapolates below the nearest measured coordinate:
    out-of-range high load is clamped to the highest measured load, while
    out-of-range low load is clamped to the lowest.  The confidence bound adds
    the larger of the profile-wide held-out residual and neighboring point
    residuals.
    """

    def __init__(
        self,
        profile: HybridModelProfileV1,
        allowed_decode_sms: Sequence[int] = (16, 24, 34, 44),
        emergency_decode_sms: int = 108,
        fallback_decode_sms: int = 44,
    ):
        profile.validate()
        self.profile = profile
        self.allowed_decode_sms = tuple(sorted({int(x) for x in allowed_decode_sms}))
        self.emergency_decode_sms = int(emergency_decode_sms)
        self.fallback_decode_sms = int(fallback_decode_sms)
        if not self.allowed_decode_sms:
            raise ValueError("at least one steady decode-SM state is required")

    @staticmethod
    def _bounds(values: Iterable[int], target: int) -> Tuple[int, int]:
        ordered = sorted(set(values))
        if not ordered:
            raise ValueError("cannot interpolate an empty axis")
        lower = max((value for value in ordered if value <= target), default=ordered[0])
        upper = min((value for value in ordered if value >= target), default=ordered[-1])
        return lower, upper

    @staticmethod
    def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
        if x0 == x1:
            return max(y0, y1)
        fraction = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        return y0 + fraction * (y1 - y0)

    def _point_value(self, decode_sms: int, batch: int, context: int) -> Tuple[float, float]:
        points = [p for p in self.profile.points if p.decode_sms == decode_sms]
        if not points:
            raise KeyError(f"profile has no measurements for D{decode_sms}")
        batch_lo, batch_hi = self._bounds((p.batch_size for p in points), batch)
        context_lo, context_hi = self._bounds(
            (p.context_tokens for p in points), context
        )

        def nearest(b: int, c: int) -> DecodeLatencyPoint:
            candidates = [p for p in points if p.batch_size == b and p.context_tokens == c]
            if candidates:
                return max(candidates, key=lambda p: p.itl_p95_ms)
            return min(
                points,
                key=lambda p: (
                    abs(math.log2(max(1, p.batch_size) / max(1, b))),
                    abs(math.log2(max(1, p.context_tokens) / max(1, c))),
                ),
            )

        corners = {
            (b, c): nearest(b, c)
            for b in (batch_lo, batch_hi)
            for c in (context_lo, context_hi)
        }
        by_batch: Dict[int, float] = {}
        residual = self.profile.heldout_residual_p95_ms
        for b in (batch_lo, batch_hi):
            p0, p1 = corners[(b, context_lo)], corners[(b, context_hi)]
            context_low_value = p0.itl_p95_ms
            context_high_value = max(p1.itl_p95_ms, context_low_value)
            by_batch[b] = self._lerp(
                context,
                context_lo,
                context_hi,
                context_low_value,
                context_high_value,
            )
            residual = max(residual, p0.residual_p95_ms, p1.residual_p95_ms)
        by_batch[batch_hi] = max(by_batch[batch_hi], by_batch[batch_lo])
        value = self._lerp(
            batch, batch_lo, batch_hi, by_batch[batch_lo], by_batch[batch_hi]
        )
        return value, residual

    def predict(self, decode_sms: int, batch_size: int, context_tokens: int) -> Tuple[float, float]:
        decode_sms = int(decode_sms)
        # Conservative monotone projection: latency at D must not be below a
        # noisier measurement at any larger decode allocation.
        available = sorted({point.decode_sms for point in self.profile.points})
        candidates = [state for state in available if state >= decode_sms]
        if not candidates:
            raise KeyError(f"profile has no measurements at or above D{decode_sms}")
        projected = [
            self._point_value(
                state, max(1, int(batch_size)), max(1, int(context_tokens))
            )
            for state in candidates
        ]
        predicted = max(value for value, _residual in projected)
        residual = max(residual for _value, residual in projected)
        return predicted, predicted + max(0.0, residual)

    def estimate(
        self,
        batch_size: int,
        context_tokens: int,
        itl_slo_ms: float,
        runtime: Optional[RuntimeEnvironment] = None,
    ) -> DecodeFloorEstimate:
        margin = max(
            0.1 * float(itl_slo_ms),
            float(self.profile.heldout_residual_p95_ms),
        )
        if runtime is not None:
            compatible, reasons = self.profile.is_compatible(runtime)
            if not compatible:
                return DecodeFloorEstimate(
                    decode_sms=self.fallback_decode_sms,
                    predicted_itl_ms=math.nan,
                    upper_bound_itl_ms=math.inf,
                    safety_margin_ms=margin,
                    confidence=0.0,
                    fallback=True,
                    reason="incompatible_profile: " + "; ".join(reasons),
                )

        target = float(itl_slo_ms) - margin
        for decode_sms in self.allowed_decode_sms:
            try:
                predicted, upper = self.predict(
                    decode_sms, batch_size, context_tokens
                )
            except KeyError:
                continue
            if upper <= target:
                confidence = max(0.0, min(1.0, 1.0 - (upper - predicted) / max(1.0, target)))
                return DecodeFloorEstimate(
                    decode_sms=decode_sms,
                    predicted_itl_ms=predicted,
                    upper_bound_itl_ms=upper,
                    safety_margin_ms=margin,
                    confidence=confidence,
                    fallback=False,
                    reason="profile_floor",
                )

        predicted = math.nan
        upper = math.inf
        try:
            predicted, upper = self.predict(
                self.emergency_decode_sms, batch_size, context_tokens
            )
        except KeyError:
            pass
        return DecodeFloorEstimate(
            decode_sms=self.emergency_decode_sms,
            predicted_itl_ms=predicted,
            upper_bound_itl_ms=upper,
            safety_margin_ms=margin,
            confidence=0.0 if math.isinf(upper) else 0.95,
            fallback=True,
            reason="no_steady_state_meets_slo",
        )


# ---------------------------------------------------------------------------
# Engine source identity (2026-09-12)
# ---------------------------------------------------------------------------
# `RuntimeEnvironment.engine_source_hash` is the compatibility axis that
# replaced `engine_commit` in `HybridModelProfileV1.is_compatible`.  It answers
# "is the engine that is about to use this profile the same CODE as the engine
# the profile was measured on?".
#
# WHY A CONTENT HASH OF THE LOADED MODULES, AND NOT THE SYNC MANIFEST.
# `scripts/bootstrap/sync_engine_tree.sh` already writes
# `results/runtime_source_manifest.sha256` (a sha256sum listing of the tracked
# engine files), and reading it would have been less code.  It is not used as
# the axis because the manifest is a CLAIM MADE AT INSTALL TIME about a
# repo-relative path, not a statement about the process that is running:
#   * one dev tree is shared by SLURM array tasks and a later sync by another
#     job rewrites the manifest under a running server (the sync script takes a
#     flock precisely because concurrent installs happen);
#   * the manifest path lives in the repo checkout, while the serving process
#     may be started with a different SGLANG_ENGINE_DEV, or from an installed
#     tree with no manifest at all -- in both cases a manifest read would
#     describe sources the process never imported;
#   * "same manifest" can therefore be True while the loaded bytes differ,
#     which is exactly the failure mode (a proxy standing in for the thing)
#     that moving off `engine_commit` is meant to remove.
# Hashing the files of the modules THIS process imported is self-validating:
# the hash moves if and only if the engine source the process actually runs
# moves, and it is computed identically on a GPU node and a login node.  The
# manifest stays useful as the install-time record (it also covers model/config
# files, which this axis deliberately does not).
#
# SCOPE.  The modules below are the ones that execute or schedule a PD-mux
# decode step, i.e. the code whose content can change measured ITL.  The
# default-OFF probe modules (holb_probe, chunk_probe, green_readout) are
# excluded on purpose: their hooks collapse to `is not None` checks when their
# env switches are unset, so including them would invalidate every profile
# whenever a diagnostic is edited.  Model/config files are out of scope too --
# `model_id`/`model_revision`/`model_config_hash` already pin the model.
ENGINE_SOURCE_MODULES: Tuple[str, ...] = (
    "sglang.srt.distributed.parallel_state",
    "sglang.srt.managers.scheduler",
    "sglang.srt.multiplex.controller",
    "sglang.srt.multiplex.dual_worker",
    "sglang.srt.multiplex.multiplexing_mixin",
    "sglang.srt.multiplex.pdmux_context",
    "sglang.srt.multiplex.profile",
    "sglang.srt.multiplex.telemetry",
)
ENGINE_SOURCE_HASH_SCHEME = "pdmux.engine-source/v1"
ENGINE_SOURCE_HASH_UNAVAILABLE = ""


def hash_engine_source_files(sources: Mapping[str, Any]) -> str:
    """Hash a `{module name: source path}` mapping deterministically.

    Only NAMES and FILE CONTENT enter the digest, never absolute paths: the
    same sources installed under a different dev-tree root must hash equal, or
    a profile could not be reused across checkouts of identical code.  Names
    are sorted, so iteration order of the mapping cannot change the result.
    """
    digest = hashlib.sha256()
    digest.update(ENGINE_SOURCE_HASH_SCHEME.encode("utf-8") + b"\n")
    for name in sorted(sources):
        data = Path(sources[name]).read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).hexdigest().encode("ascii") + b"\n")
    return digest.hexdigest()


def resolve_engine_source_files(
    module_names: Sequence[str] = ENGINE_SOURCE_MODULES,
) -> Optional[Dict[str, Path]]:
    """Map engine module names to their source files, or None if incomplete.

    Already-imported modules are read from `sys.modules` (the bytes this
    process ran); anything not imported -- the offline profile builder imports
    no engine -- is located with `importlib.util.find_spec`.  Any failure
    returns None so the caller can fail closed instead of hashing a subset.
    """
    resolved: Dict[str, Path] = {}
    for name in module_names:
        module = sys.modules.get(name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin is None:
            origin = getattr(module, "__file__", None)
        if origin is None:
            try:
                spec = importlib.util.find_spec(name)
            except Exception:  # ImportError, ValueError, partially built pkgs
                return None
            origin = spec.origin if spec is not None else None
        if not origin:
            return None
        path = Path(origin)
        if not path.is_file():
            return None
        resolved[name] = path
    return resolved


def engine_source_hash(
    module_names: Sequence[str] = ENGINE_SOURCE_MODULES,
) -> str:
    """Content hash of the engine sources, or "" when it cannot be computed.

    "" is the fail-closed sentinel: `is_compatible` reports it as an explicit
    mismatch reason, so an un-hashable engine gets the conservative fallback
    rather than a silent match.  Never raises -- a profile-compatibility check
    must not be able to kill a server.
    """
    try:
        sources = resolve_engine_source_files(module_names)
        if sources is None:
            return ENGINE_SOURCE_HASH_UNAVAILABLE
        return hash_engine_source_files(sources)
    except Exception:  # unreadable file, permissions, races
        return ENGINE_SOURCE_HASH_UNAVAILABLE
