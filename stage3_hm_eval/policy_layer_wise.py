"""
Stage 3 Policy C: Layer-boundary SM reconfiguration.

SM ratio changes at every prefill layer boundary based on layer type:
  - SSM layer  → prefill gets 70% SM (SSM is memory-bound, benefits from BW)
  - Attn layer → prefill gets 40% SM (attention is compute-bound, needs fewer)

This is the most aggressive strategy and is only executed when Stage 2
determines overhead_ratio < 0.05 (i.e., smctrl overhead < 5% of layer time).
If overhead is too high, this policy degrades performance vs. Policy A.

Conditional execution check:
    from stage3_hm_eval.policy_layer_wise import should_run_policy_c
    if should_run_policy_c(decision_matrix_path):
        run_with_policy_c(...)
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.smctrl.libsmctrl_wrapper import SMController

# Per-layer-type SM ratios for prefill
SSM_PREFILL_RATIO = 0.70   # SSM is memory-BW bound; benefits from SM parallelism
ATTN_PREFILL_RATIO = 0.40  # Attention is compute-bound; fewer SMs sufficient
MLP_PREFILL_RATIO = 0.50   # MLP intermediate


def should_run_policy_c(decision_matrix_path: Path) -> bool:
    """Check Stage 2 decision matrix to see if Policy C is warranted.

    Returns True only if dominant_strategy == 'layer_wise'.
    """
    if not decision_matrix_path.exists():
        print(f"WARNING: decision_matrix.json not found at {decision_matrix_path}")
        print("  Defaulting to NOT running Policy C (run Stage 2 first).")
        return False

    with open(decision_matrix_path) as f:
        matrix = json.load(f)

    dominant = matrix.get("dominant_strategy", "fixed")
    return dominant == "layer_wise"


@dataclass
class PolicyConfig:
    name: str = "layer_wise"
    ssm_prefill_ratio: float = SSM_PREFILL_RATIO
    attn_prefill_ratio: float = ATTN_PREFILL_RATIO
    mlp_prefill_ratio: float = MLP_PREFILL_RATIO


class PolicyLayerWise:
    """Policy C: layer-boundary SM reconfiguration.

    At each prefill layer start, reads the current layer type and
    calls smctrl.set_sm_ratio() to adjust SM allocation:
      ssm  → prefill_ratio = 0.70, decode gets 0.30
      attn → prefill_ratio = 0.40, decode gets 0.60
      mlp  → prefill_ratio = 0.50, decode gets 0.50

    This is the same granularity as BulletServe/MuxWise layer interleaving:
    each "layer step" in the interleaved loop calls set_sm_ratio before
    running the layer kernel. The overhead of set_sm_ratio is what Stage 2
    quantifies.
    """

    def __init__(self, smctrl: SMController, config: PolicyConfig = None):
        self.smctrl = smctrl
        self.config = config or PolicyConfig()
        self._reconfig_count = 0
        self._reconfig_overhead_us = 0.0  # accumulated (if tracked)

    def on_step_start(self, decode_batch_size: int, prefill_seq_len: int) -> None:
        """No step-level reconfiguration — happens at layer boundaries."""
        pass

    def on_decode(self) -> None:
        """SM ratio for decode is the complement of the last prefill layer's ratio."""
        # Decode always gets what's left; we set a safe default here
        # (actual decode SM share is determined by the last on_prefill_layer_start call)
        self.smctrl.set_sm_ratio(1.0 - ATTN_PREFILL_RATIO)

    def on_prefill_layer_start(self, layer_idx: int, layer_type: str) -> None:
        """Reconfigure SM mask based on layer type before executing the layer.

        This is where the overhead measured in Stage 2 is incurred.
        """
        ratio = self.get_prefill_ratio(layer_type)
        self.smctrl.set_sm_ratio(ratio)
        self._reconfig_count += 1

    def on_prefill_layer_end(self, layer_idx: int, layer_type: str) -> None:
        """No-op — next call to on_decode or on_prefill_layer_start reconfigures."""
        pass

    def get_prefill_ratio(self, layer_type: str) -> float:
        if layer_type == "ssm":
            return self.config.ssm_prefill_ratio
        elif layer_type in ("attn", "attention"):
            return self.config.attn_prefill_ratio
        else:
            return self.config.mlp_prefill_ratio

    def get_decode_ratio(self) -> float:
        # Decode ratio varies with layer type; return the mean for reporting
        return 1.0 - (self.config.ssm_prefill_ratio + self.config.attn_prefill_ratio) / 2

    def get_reconfig_count(self) -> int:
        return self._reconfig_count

    def describe(self) -> dict:
        return {
            "policy": "C_layer_wise",
            "ssm_prefill_ratio": self.config.ssm_prefill_ratio,
            "attn_prefill_ratio": self.config.attn_prefill_ratio,
            "mlp_prefill_ratio": self.config.mlp_prefill_ratio,
            "dynamic_reconfig": True,
            "granularity": "layer",
        }
