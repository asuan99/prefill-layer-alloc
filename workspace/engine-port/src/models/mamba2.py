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
"""Inference-only pure Mamba2 model (state-spaces/mamba2-*).

Engine-port (prefill-layer-alloc, Stage 0 negative-control arm). Reuses the
NemotronH hybrid implementation wholesale (NemotronHModel with an all-"mamba"
layers_block_type => 64x NemotronHMambaDecoderLayer => MambaMixer2 + MambaPool +
HybridLinearAttnBackend + the piecewise/green-ctx cudagraph mamba split-op). The
ONLY model-side difference is load_weights, which maps the NATIVE mamba_ssm
checkpoint key layout onto the NemotronH parameter names:

  native (pytorch_model.bin)                 -> sglang param
  backbone.embedding.weight                  -> model.embed_tokens.weight
  backbone.layers.{i}.norm.weight            -> model.layers.{i}.norm.weight
  backbone.layers.{i}.mixer.in_proj.weight   -> model.layers.{i}.mixer.in_proj.weight
  backbone.layers.{i}.mixer.conv1d.{w,b}     -> model.layers.{i}.mixer.conv1d.{w,b}
  backbone.layers.{i}.mixer.A_log            -> model.layers.{i}.mixer.A   (A=-exp(A_log))
  backbone.layers.{i}.mixer.D                -> model.layers.{i}.mixer.D
  backbone.layers.{i}.mixer.dt_bias          -> model.layers.{i}.mixer.dt_bias
  backbone.layers.{i}.mixer.norm.weight      -> model.layers.{i}.mixer.norm.weight
  backbone.layers.{i}.mixer.out_proj.weight  -> model.layers.{i}.mixer.out_proj.weight
  backbone.norm_f.weight                     -> model.norm_f.weight
  lm_head.weight                             -> tied to embed_tokens (skipped)

Key-mapping coverage was verified against the real 2.7b checkpoint (579 keys,
0 unmapped, 0 uncovered).
"""

from collections.abc import Iterable
from typing import Optional

import torch

from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.utils import logger


class Mamba2ForCausalLM(NemotronHForCausalLM):
    """Pure Mamba2 for causal LM. Inherits all NemotronH runtime behaviour
    (forward, forward_split_prefill, cuda-graph mamba hooks); overrides only the
    native checkpoint weight loader."""

    # No stacked/packed mappings: pure mamba has no fused qkv / gate-up.
    stacked_params_mapping = []
    packed_modules_mapping = {}

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        params_dict = dict(self.named_parameters())
        tie = bool(getattr(self.config, "tie_word_embeddings", True))

        for name, loaded_weight in weights:
            # --- native mamba_ssm -> sglang key remap ---
            if name.startswith("backbone."):
                name = "model." + name[len("backbone.") :]
            # state-spaces native uses `backbone.embedding.`; the transformers
            # Mamba2ForCausalLM export (mistralai/Mamba-Codestral-7B-v0.1, the
            # 7B pure-SSM arm) uses `backbone.embeddings.`. Without the plural
            # form the tensor is silently dropped -> random embeddings.
            name = name.replace(".embeddings.", ".embed_tokens.")
            name = name.replace(".embedding.", ".embed_tokens.")
            name = name.replace("mixer.A_log", "mixer.A")

            # lm_head is tied to the input embedding (tie_embeddings=true). The
            # tied module makes lm_head.weight an alias of embed_tokens.weight
            # (already loaded); skip if the standalone param is absent.
            if name == "lm_head.weight":
                if tie or name not in params_dict:
                    continue

            # PP shard filtering (single-GPU: no-op, kept for parity).
            if "embed_tokens" in name and not self.pp_group.is_first_rank:
                continue
            if (
                "norm_f" in name or "lm_head" in name
            ) and not self.pp_group.is_last_rank:
                continue

            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in params_dict")
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [Mamba2ForCausalLM]
