"""
policy_spatial_falcon.py — static spatial SM split for Falcon-H1 (E4 entry point).

Falcon-H1 runs an SSM branch and an Attention branch IN PARALLEL in every layer.
A static spatial split assigns a fixed fraction of SMs to each branch ONCE and
never reconfigures: the two Green Context partitions live for the whole run, so
there is zero per-layer swap overhead and the schedule is CUDA-Graph friendly.

This is the **minimum-cost entry point** for E4 — implement and validate this
before the temporal policy. If even the reconfig-free spatial split shows no
concurrent benefit, the temporal policy (which pays swap overhead) cannot.

Reconfig-free is the whole point, so reconfig_count() is identically 0; that is
asserted, not assumed.
"""

from __future__ import annotations


class PolicySpatialFalcon:
    def __init__(self, ssm_sm_fraction: float = 0.5):
        if not 0.0 < ssm_sm_fraction < 1.0:
            raise ValueError("ssm_sm_fraction must be in (0,1)")
        self.ssm_sm_fraction = ssm_sm_fraction
        self._reconfigs = 0
        self._partitions = None      # (ssm_stream, attn_stream, info)

    # -- setup: create the two persistent partitions ONCE ------------------
    def setup(self, total_sm: int):
        """Create the two spatial partitions a single time. Returns info dict.

        Uses experiments.e4_concurrent._green_ctx.create_two_partitions (which is
        built on src primitives). The SSM branch gets ``ssm_sm_fraction`` of SMs;
        the Attention branch gets the remainder.
        """
        from experiments.e4_concurrent._green_ctx import create_two_partitions
        n_ssm = max(1, round(self.ssm_sm_fraction * total_sm))
        s_ssm, s_attn, info = create_two_partitions(n_first_sm=n_ssm)
        self._partitions = (s_ssm, s_attn, info)
        return info

    def streams(self):
        if self._partitions is None:
            raise RuntimeError("call setup(total_sm) first")
        s_ssm, s_attn, _ = self._partitions
        return {"ssm": s_ssm, "attn": s_attn}

    # -- the contract: no reconfiguration during the run -------------------
    def on_prefill_layer(self, layer_type: str) -> None:
        """No-op by design: the spatial split is fixed for the whole run."""
        # intentionally does NOT touch SM allocation — that is the point.
        return None

    def reconfig_count(self) -> int:
        return self._reconfigs  # always 0

    def describe(self) -> dict:
        return {
            "policy": "spatial_falcon",
            "model_family": "falcon_h1",
            "ssm_sm_fraction": self.ssm_sm_fraction,
            "reconfig_per_layer": False,
            "reconfig_count": self._reconfigs,
            "cuda_graph_friendly": True,
            "note": "static split of parallel SSM/Attn branches; zero swap overhead",
        }


def assert_reconfig_free(policy: PolicySpatialFalcon) -> None:
    """Guard used by the runner: the spatial policy must never reconfigure."""
    assert policy.reconfig_count() == 0, (
        f"spatial policy reconfigured {policy.reconfig_count()} times — it must be 0"
    )
