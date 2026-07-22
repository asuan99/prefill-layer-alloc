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

## P1.5 (Zamba2 pdmux port) — all inside `models/zamba2.py` (item 2 covers re-apply)
Verified: plain + layer_aware pdmux boot OK ("Paris" correct, NO_CRASH under
concurrent long-ctx load; jobs 831123/831135). Additions to `Zamba2ForCausalLM` /
`Zamba2Model`:
- `Zamba2ForCausalLM.forward_split_prefill(input_ids, positions, forward_batch,
  split_interval, input_embeds=None)` — enables pdmux SPLIT_PREFILL on the hybrid.
  Threads `forward_batch.hidden_states` (+ `forward_batch.zamba_original` = cloned
  embeddings) across split windows; forward_batch persists across split calls.
- `Zamba2Model.forward` decode loop rewritten race-safe (per-layer target stream +
  `wait_stream` at each switch = data-dep + safety vs default stream). Modes read from
  `PDMUX_FIXED_DECODE_SM_FILE`: numeric `<sm>@<ctx>` → pin all decode layers to N SM
  (sweep); `layer_aware` → Zamba2HybridLayer (9, no-GQA attn, SM-sensitive) stay on
  base/reserved, ~45 pure-mamba layers drop to `PDMUX_LA_FLOOR_SM` (default 16).
- Fix: `_fixed` parse guards `_sm_part.isdigit()` (else `int("layer_aware")` crashes).
Serving env for layer_aware: `PDMUX_FIXED_DECODE_SM_FILE=<file with "layer_aware">` +
`PDMUX_LA_FLOOR_SM=16` (no `PDMUX_LA_RESERVE` — Zamba2 reserve is by layer type, not
data-driven like NemotronH). Boot: `--attention-backend triton` (head_dim 160).

## P1.6 (agnostic_v2 policy) — inside `models/{nemotron_h,zamba2}.py` (items 2/6 cover re-apply)
New mode `agnostic_v2`: uniform decode-SM reservation **sized to the (cheap, SM-
insensitive) attention layer** = pin ALL decode layers to a low floor
`PDMUX_AGN2_SM` (default 16), maximizing prefill reclaim but NOT protecting the
SM-sensitive layers. Contrast:
- **agnostic (v1)** = reserve for the SM-hungry layer (mamba) → high uniform N, protective.
- **agnostic_v2** = reserve for the attention layer → low uniform N, max reclaim, unprotected.
- **layer_aware** = per-layer: reclaim on insensitive, protect on sensitive.
Both models parse `_sm_part == "agnostic_v2"` → `_fixed = PDMUX_AGN2_SM` (reuses the
fixed-N `_tgt` path; add `"agnostic_v2"` to NemotronH's non-numeric exclusion list —
Zamba2's `.isdigit()` guard already excludes it). Serving env: mode file
`agnostic_v2` + `PDMUX_AGN2_SM=16`. Harness: `p1_4_serving.sbatch`/`p1_4_zb_serving.sbatch`
now run 4 policies; boot check `p1_4_zb_pdmux_check.sbatch agnostic_v2` /
`p1_4_la_check.sbatch agnostic_v2`.

## P1.7 (additional hybrids) — edits to existing sglang model files (mirror in src/models/)
8. `models/falcon_h1.py` — add `FalconH1ForCausalLM.forward_split_prefill` (enables
   pdmux on the SPATIAL hybrid; threads (hidden, residual) + embedding_multiplier +
   final_layernorm). Boot flashinfer (head_dim 128); `--disable-piecewise-cuda-graph`.
   Copy: `src/models/falcon_h1.py`.
9. `models/granitemoehybrid.py` — add `GraniteMoeHybridModel._get_gctx_decode_stream`
   + green-ctx decode-SM pin & per-layer-type CUDA-event timing in `.forward` (env-gated
   `SGLANG_GRANITE_TIMING`/`PDMUX_FIXED_DECODE_SM_FILE`, dormant by default; emits `GMHLT`
   log). For measuring Granite Mamba2-SSD decode SM-sensitivity. Boot flashinfer + bf16 +
   `--disable-piecewise-cuda-graph`. Copy: `src/models/granitemoehybrid.py`.

Cached weights used (P1.7): tiiuae/Falcon-H1-3B-Base, Zyphra/Zamba2-{1.2B,7B-Instruct};
downloaded ibm-granite/granite-4.0-h-micro-base (login-node internet). Harness:
`p1_7_zb_smsens.sbatch <model>`, `p1_7_granite_smsens.sbatch`, `p1_7_bench_one.sbatch
<policy> <model> <backend>`, `p1_7_fh1_check.sbatch <plain|pdmux> [model]`.

## R0b (graduated per-type layer-aware) — inside `models/zamba2.py` (item 2 covers re-apply)
10. `models/zamba2.py` `Zamba2Model.forward` `_tgt`: env `PDMUX_LA_SM_MAP="mamba:54,attn:96"`
    pins each layer TYPE to its own decode-SM green-ctx (`_get_gctx_decode_stream(N)`),
    generalizing the 2-level binary la (full/`PDMUX_LA_FLOOR_SM`) into graduated per-type
    allocation. Backward-compatible: unset → legacy binary behavior. Added to test whether
    per-type allocation beats uniform agnostic (it does not — see `results/r0b/`). Copy:
    `src/models/zamba2.py`. Harness `results/r0b/r0b_graduated_bench.sbatch <agn|la_bin|g_<mamba>_<attn>> <rep>`.

## R1 (Hybrid LLM dual-worker queue ownership)

11. **NEW** `multiplex/dual_worker.py` — copied from
`src/multiplex/dual_worker.py`. Defines `PrefillWorker`,
`DecodeWorker`, `SharedGpuArbiter`, `PhaseCoordinator`, and their composed
`DualWorkerState`. It has no CUDA/model imports and is safe to unit-test on a
CPU-only node.

12. `multiplex/multiplexing_mixin.py` — imports the dual-worker state and
connects it to `init_pdmux`, `update_split_prefill_batch`, and the ordinary
`event_loop_pdmux` path. Enable with `PDMUX_DUAL_WORKER=1`; unset or `0` keeps
the established baseline behavior. `event_loop_pdmux_coord` and all
per-layer switching paths are intentionally unchanged.

The external dev tree used for runtime validation is
`/scratch/ehmoon/whlee/sglang_engine_dev/python`. The tracked copies under
`workspace/engine-port/src/` are the re-application source of truth. The
corresponding design and experiment boundary are documented in
`reports/dual_worker_design.md`.

13. Optional queue telemetry is enabled only when both
`PDMUX_DUAL_WORKER=1` and `PDMUX_DUAL_WORKER_TRACE=<jsonl path>` are set.
`PDMUX_DUAL_WORKER_TRACE_EVERY` controls the scheduler-sync sampling interval
(default `32`). Unset trace variables leave the runtime path unchanged.
