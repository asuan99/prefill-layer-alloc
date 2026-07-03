# SPDX-License-Identifier: Apache-2.0
"""SGLang config wrapper for Zamba2 (temporal Mamba2 + shared-attention hybrid).

Subclasses HF `transformers.Zamba2Config` and adds the properties SGLang's
hybrid memory pools + MambaMixer2 need (mamba2_cache_params, mamba_layer_ids,
full_attention_layer_ids). Every Zamba2 layer has a Mamba2 mixer; the 9
`hybrid_layer_ids` positions additionally run a *shared* attention block, so:
  - mamba_layer_ids          = all layers            (mamba state pool)
  - full_attention_layer_ids = hybrid_layer_ids      (attention KV pool)
These two pools are indexed independently, so a hybrid layer's global index
appears in both — which is correct for Zamba2.
"""
from transformers import Zamba2Config as HFZamba2Config

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)


class Zamba2Config(HFZamba2Config):
    model_type = "zamba2"

    @property
    def layers_block_type_list(self):
        # HF stores per-layer type in `layers_block_type` (list of "mamba"/"hybrid").
        return list(self.layers_block_type)

    @property
    def mamba_chunk_size(self):
        # SGLang's mamba2 path reads `mamba_chunk_size`; HF Zamba2 names it `chunk_size`.
        return self.chunk_size

    @property
    def mamba_layer_ids(self):
        # Every Zamba2 layer carries a Mamba2 mixer.
        return list(range(self.num_hidden_layers))

    @property
    def full_attention_layer_ids(self):
        # Attention (shared block) runs only at hybrid positions.
        return list(self.hybrid_layer_ids)

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        intermediate_size = self.mamba_expand * self.hidden_size
        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=intermediate_size,
            n_groups=self.mamba_ngroups,
            num_heads=self.n_mamba_heads,
            head_dim=self.mamba_headdim,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )
        return Mamba2CacheParams(
            shape=shape, layers=self.mamba_layer_ids, dtype=mamba2_state_dtype(self)
        )
