"""Versioned offline profiles and conservative decode-floor estimation.

The profile is deliberately independent of SGLang so that profiles can be
validated and evaluated on login/CI nodes.  Measurements are full-model decode
latencies collected at the same CUDA-Graph/backend operating point as serving;
isolated layer timings are metadata only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROFILE_SCHEMA = "pdmux.hybrid-model-profile/v1"


@dataclass(frozen=True)
class RuntimeEnvironment:
    engine_commit: str
    gpu_name: str
    gpu_sm_count: int
    cuda_driver: str
    attention_backend: str
    cuda_graph: bool
    piecewise_cuda_graph: bool = False


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
        """Return strict compatibility and human-readable mismatch reasons."""
        reasons: List[str] = []
        for name in (
            "engine_commit",
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
