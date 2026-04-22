"""
Stage 3 Policy B: Step-level SM adaptation based on model SSM fraction.

SM ratio is chosen once per decode step based on the model's overall
SSM-layer fraction. SSM-heavy models get more SMs for prefill; attention-
heavy models get fewer.

No reconfiguration occurs within a prefill pass (no per-layer changes).
This tests whether step-level model-aware allocation improves over the
fixed baseline without incurring per-layer reconfiguration overhead.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from pathlib import Path

if TYPE_CHECKING:
    from src.smctrl.libsmctrl_wrapper import SMController

# Per-model SM ratios based on SSM layer fraction
# Zamba2: SSM fraction ~0.889 → higher prefill SM share (SSM is parallelizable)
# Falcon-H1: all hybrid layers (SSM+Attn parallel) → intermediate
MODEL_PREFILL_RATIOS = {
    "zamba2": 0.40,    # Zamba2 has periodic attention; SSM is fast, don't over-allocate
    "falcon_h1": 0.70, # Falcon-H1 SSM branch benefits from more SMs at larger seq_lens
}
DEFAULT_PREFILL_RATIO = 0.50
DEFAULT_DECODE_RATIO = 0.50


@dataclass
class PolicyConfig:
    name: str = "step_adaptive"
    model_name: str = "zamba2"


class PolicyStepAdaptive:
    """Policy B: step-level model-characteristic SM adaptation.

    SM ratio is fixed per step based on:
      - model SSM layer fraction (from configs/models.yaml)
      - No per-layer reconfiguration (layer boundaries are ignored)

    The logic is:
      prefill_ratio = MODEL_PREFILL_RATIOS.get(model_name, DEFAULT_PREFILL_RATIO)
      decode_ratio = 1.0 - prefill_ratio
    """

    def __init__(self, smctrl: SMController, config: PolicyConfig = None):
        self.smctrl = smctrl
        self.config = config or PolicyConfig()
        self._prefill_ratio = MODEL_PREFILL_RATIOS.get(
            self.config.model_name, DEFAULT_PREFILL_RATIO
        )
        self._decode_ratio = 1.0 - self._prefill_ratio
        self._current_prefill_ratio = self._prefill_ratio

    def on_step_start(self, decode_batch_size: int, prefill_seq_len: int) -> None:
        """Choose SM ratio for this step. Could also adapt on batch/seq_len here."""
        # Policy B keeps ratio constant per model; extend here for more adaptation
        self._current_prefill_ratio = self._prefill_ratio

    def on_decode(self) -> None:
        """Set SM ratio for decode phase."""
        self.smctrl.set_sm_ratio(self._decode_ratio)

    def on_prefill_layer_start(self, layer_idx: int, layer_type: str) -> None:
        """Set SM ratio for prefill phase (same for all layer types)."""
        self.smctrl.set_sm_ratio(self._current_prefill_ratio)

    def on_prefill_layer_end(self, layer_idx: int, layer_type: str) -> None:
        """No-op for Policy B."""
        pass

    def get_prefill_ratio(self, layer_type: str) -> float:
        return self._current_prefill_ratio

    def get_decode_ratio(self) -> float:
        return self._decode_ratio

    def describe(self) -> dict:
        return {
            "policy": "B_step_adaptive",
            "model_name": self.config.model_name,
            "prefill_sm_ratio": self._prefill_ratio,
            "decode_sm_ratio": self._decode_ratio,
            "dynamic_reconfig": False,
            "granularity": "step",
        }
