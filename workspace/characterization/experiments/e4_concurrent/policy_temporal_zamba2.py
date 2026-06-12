"""
policy_temporal_zamba2.py — layer-boundary SM reconfig for Zamba2 (prefill only).

Zamba2 interleaves pure-Mamba (SSM) layers with sparse hybrid (Attn) layers in
time, so a temporal policy reconfigures the SM allocation at each prefill layer
boundary: more SMs to the SSM layers, fewer to the (rarer) attention layers.

What changed from v1 Policy C (decode REMOVED)
---------------------------------------------
v1 reconfigured SM *within decode* on a per-step basis (Policy B/C). That is
rejected in v2: decode is a donor/fallback, not a thing whose SM budget we tune.
``on_decode`` therefore never reconfigures — it raises if asked to, so a misuse
cannot silently resurrect the decode-step adaptation that v2 killed.

Each reconfig costs ~7.8 µs (v1 Stage-2 measurement, reused in E0's overhead
budget). reconfig_count() lets the runner price the policy against the spatial
(reconfig-free) one.
"""

from __future__ import annotations


class PolicyTemporalZamba2:
    def __init__(self, smctrl, ssm_ratio: float = 0.7, attn_ratio: float = 0.4):
        """smctrl: an src.smctrl.SMController (passed in; not imported here)."""
        self.smctrl = smctrl
        self.ssm_ratio = ssm_ratio
        self.attn_ratio = attn_ratio
        self._reconfigs = 0
        self._last_ratio = None

    def on_prefill_layer(self, layer_type: str) -> None:
        """Reconfigure SM ratio at a prefill layer boundary, by layer type."""
        ratio = self.ssm_ratio if layer_type != "attn" else self.attn_ratio
        if ratio != self._last_ratio:
            self.smctrl.set_sm_ratio(ratio)
            self._last_ratio = ratio
            self._reconfigs += 1

    def on_decode(self, *args, **kwargs):
        """Decode is donor/fallback — NEVER reconfigured within decode in v2."""
        raise RuntimeError(
            "PolicyTemporalZamba2.on_decode is intentionally unsupported: v2 "
            "rejects decode-step SM reconfiguration. Decode donates its SMs to "
            "prefill via a fixed split; do not tune it per step."
        )

    def reconfig_count(self) -> int:
        return self._reconfigs

    def describe(self) -> dict:
        return {
            "policy": "temporal_zamba2",
            "model_family": "zamba2",
            "ssm_ratio": self.ssm_ratio,
            "attn_ratio": self.attn_ratio,
            "reconfig_per_layer": True,
            "reconfig_count": self._reconfigs,
            "decode_reconfig": False,    # removed in v2
            "swap_cost_us_each": 7.8,    # v1 Stage-2 (archive), see E0 budget
            "note": "per-layer-type SM reconfig in prefill only; decode is donor",
        }
