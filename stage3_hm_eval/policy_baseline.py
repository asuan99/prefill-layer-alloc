"""
Stage 3 Policy A: Fixed SM split, no layer-boundary reconfiguration.

SM allocation is fixed for the entire evaluation:
  - Prefill: PREFILL_RATIO (default 40%) of total SMs
  - Decode: DECODE_RATIO (default 60%) of total SMs

Layer type transitions within prefill do NOT trigger SM changes.
This is the baseline that Policies B and C are compared against.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.smctrl.libsmctrl_wrapper import SMController

PREFILL_RATIO = 0.40
DECODE_RATIO = 0.60


@dataclass
class PolicyConfig:
    name: str = "baseline"
    prefill_sm_ratio: float = PREFILL_RATIO
    decode_sm_ratio: float = DECODE_RATIO


class PolicyBaseline:
    """Policy A: fixed SM split, no dynamic reconfiguration.

    Implements the MuxWise-style layer-interleaved prefill+decode,
    but with a constant SM allocation throughout. This serves as the
    baseline for measuring the benefit of dynamic SM reallocation.

    Execution pattern (per decode step):
      1. Set SM → decode_ratio (once at init, never changed)
      2. run_decode_step()
      3. If prefill pending: set SM → prefill_ratio, run 1 prefill layer
      4. (no SM change at layer boundaries within prefill)

    Note: "Simultaneous" execution is layer-level interleaving on a
    single GPU, not true kernel-level parallelism. This matches the
    BulletServe/MuxWise architecture described in the paper.
    """

    def __init__(self, smctrl: SMController, config: PolicyConfig = None):
        self.smctrl = smctrl
        self.config = config or PolicyConfig()

    def on_step_start(self, decode_batch_size: int, prefill_seq_len: int) -> None:
        """Called at the beginning of each decode step. No-op for Policy A."""
        pass

    def on_decode(self) -> None:
        """Set SM ratio for decode phase."""
        self.smctrl.set_sm_ratio(self.config.decode_sm_ratio)

    def on_prefill_layer_start(self, layer_idx: int, layer_type: str) -> None:
        """Set SM ratio for prefill phase (fixed, ignores layer_type)."""
        self.smctrl.set_sm_ratio(self.config.prefill_sm_ratio)

    def on_prefill_layer_end(self, layer_idx: int, layer_type: str) -> None:
        """No-op for Policy A."""
        pass

    def get_prefill_ratio(self, layer_type: str) -> float:
        return self.config.prefill_sm_ratio

    def get_decode_ratio(self) -> float:
        return self.config.decode_sm_ratio

    def describe(self) -> dict:
        return {
            "policy": "A_baseline",
            "prefill_sm_ratio": self.config.prefill_sm_ratio,
            "decode_sm_ratio": self.config.decode_sm_ratio,
            "dynamic_reconfig": False,
        }
