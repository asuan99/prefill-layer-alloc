# SGLang v0.5.10 dev-tree edits for Zamba2 (re-apply if `sglang_engine_dev` rebuilt)

Dev tree: `/scratch/$USER/whlee/sglang_engine_dev/python/sglang/srt/`. New files are
copied under `workspace/engine-port/src/`; the edits to *existing* sglang files:

1. **NEW** `configs/zamba2.py`  ← copy from `src/configs/zamba2.py`.
   `Zamba2Config(HFZamba2Config)` + props: `mamba2_cache_params`, `mamba_layer_ids`
   (=all layers), `full_attention_layer_ids` (=hybrid_layer_ids), `mamba_chunk_size`
   (=chunk_size), `layers_block_type_list`.

2. **NEW** `models/zamba2.py`  ← copy from `src/models/zamba2.py`. `EntryClass=[Zamba2ForCausalLM]`.

3. `utils/hf_transformers_utils.py` — after the `AutoConfig.register` loop, add:
   ```python
   from sglang.srt.configs.zamba2 import Zamba2Config as _SGLZamba2Config
   with contextlib.suppress(ValueError):
       AutoConfig.register("zamba2", _SGLZamba2Config, exist_ok=True)
   ```
   (HF already registers "zamba2"; the suppress-only loop won't override → need exist_ok.)

4. `configs/__init__.py` — `from sglang.srt.configs.zamba2 import Zamba2Config` + add
   `"Zamba2Config"` to `__all__`.

5. `model_executor/model_runner.py` —
   (a) add `Zamba2Config` to the `from sglang.srt.configs import (...)` block;
   (b) in the `mamba2_config` property, add `| Zamba2Config` to the isinstance gate
       (so Zamba2 uses `HybridLinearAttnBackend`);
   (c) `maybe_update_ngram_token_table`: `ngram_embedding_info =
       getattr(forward_batch, "ngram_embedding_info", None)` (pdmux-path fix, P1.0).

Boot: `--dtype bfloat16` required (mamba conv-dtype default bf16 vs model fp16).

## P1.3 (NemotronH pdmux) — additional dev-tree edit
6. `models/nemotron_h.py` — add `forward_split_prefill` to `NemotronHForCausalLM`
   (enables pdmux on hybrid models). Full method: `src/patches/nemotron_h_forward_split_prefill.patch`.
   Boot NemotronH needs `--disable-piecewise-cuda-graph` on py3.14 (inductor/Union).

## P1.4+ (Zamba2 fast attn) — additional dev-tree edit
7. `layers/attention/triton_backend.py` — v_head_dim init: broaden the hybrid
   condition from `hybrid_gdn_config` to `mambaish_config` (covers mamba2 hybrids
   NemotronH/Zamba2), so triton doesn't query layer-0 KV on models where layer 0
   isn't attention. Enables `--attention-backend triton` for Zamba2 (fast, correct;
   flashinfer NaNs on head_dim=160). Patch: `src/patches/triton_backend_mambaish_vheaddim.patch`.
