# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pure Mamba2 (state-spaces/mamba2-*) model configuration.

Engine-port (prefill-layer-alloc, Stage 0 negative-control arm): a pure-SSM
model (no attention, no MLP) used as the decode SM-insensitivity lower-bound
anchor for the green-ctx PD-mux SM-partition study.

Mamba2Config subclasses NemotronHConfig so that the hybrid mamba runtime wiring
(model_runner.mamba2_config / mambaish_config isinstance checks, MambaPool,
HybridLinearAttnBackend, mamba2_cache_params) engages WITHOUT any change to the
model_runner. The only difference vs. NemotronH is that layers_block_type is
ALL "mamba" (zero attention / zero mlp / zero moe layers).

The native state-spaces checkpoint config.json is NOT HF-formatted (no
`architectures`, no `model_type`, fields d_model/n_layer/ssm_cfg). It is adapted
to this config by scripts/models/convert_mamba2_native.py, which writes an
HF-format wrapper config.json with model_type="mamba2". This class also accepts
the native fields directly for robustness.
"""

from transformers.utils import logging

from sglang.srt.configs.nemotron_h import NemotronHConfig

logger = logging.get_logger(__name__)


class Mamba2Config(NemotronHConfig):
    """Pure Mamba2 config. All layers are Mamba2 mixers."""

    # NOTE: transformers v5 ships a built-in "mamba2" model_type whose config
    # class has incompatible field semantics (validates hidden_size*expand ==
    # num_heads*head_dim). Use a distinct model_type so AutoConfig.register does
    # not collide (the sglang config registry uses suppress(ValueError), which
    # would otherwise silently keep HF's class). The wrapper config.json emitted
    # by convert_mamba2_native.py carries this same model_type.
    model_type = "mamba2_ssm"

    def __init__(
        self,
        # --- canonical (HF-format wrapper) fields ---
        vocab_size: int = 50288,
        hidden_size: int = 2560,
        num_hidden_layers: int = 64,
        tie_word_embeddings: bool = True,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        # mamba dims (mamba2-2.7b: d_inner=5120=80*64, ngroups=1, d_state=128)
        mamba_num_heads: int = 80,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        mamba_n_groups: int = 1,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_hidden_act: str = "silu",
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        mamba_chunk_size: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        **kwargs,
    ):
        # --- native mamba_ssm field aliases (best-effort adapter) ---
        # If a raw state-spaces config.json is ever loaded through this class
        # directly (rather than through the converter), translate its fields.
        if "d_model" in kwargs:
            hidden_size = kwargs.pop("d_model")
        if "n_layer" in kwargs:
            num_hidden_layers = kwargs.pop("n_layer")
        if "tie_embeddings" in kwargs:
            tie_word_embeddings = kwargs.pop("tie_embeddings")
        if "rms_norm_eps" in kwargs:
            layer_norm_epsilon = kwargs.pop("rms_norm_eps")
        # unused native fields we must not forward as PretrainedConfig kwargs
        for _dead in ("ssm_cfg", "attn_cfg", "attn_layer_idx", "d_intermediate",
                      "fused_add_norm", "rms_norm", "pad_vocab_size_multiple"):
            kwargs.pop(_dead, None)

        # Force a pure-mamba layer stack.
        layers_block_type = ["mamba"] * int(num_hidden_layers)

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            layers_block_type=layers_block_type,
            layer_norm_epsilon=layer_norm_epsilon,
            residual_in_fp32=residual_in_fp32,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            ssm_state_size=ssm_state_size,
            mamba_n_groups=mamba_n_groups,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_hidden_act=mamba_hidden_act,
            mamba_conv_bias=mamba_conv_bias,
            mamba_proj_bias=mamba_proj_bias,
            mamba_chunk_size=mamba_chunk_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # NemotronHConfig.mamba2_cache_params reads self.n_groups (NOT
        # self.mamba_n_groups). NemotronH real checkpoints carry `n_groups`
        # in their JSON kwargs; pure-mamba wrapper does not, so set it here.
        self.n_groups = mamba_n_groups
