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

## R1 observer와 R2 true dual-worker

R1은 독립 queue/thread가 아니라 observer였으므로 architecture result로 사용하지
않는다. R2는 `src/multiplex/{profile,controller,telemetry}.py`, 두 host-thread
runtime, `src/patches/pdmux_thread_local_role.patch`를 추가한다. 수동 복사 대신
`scripts/bootstrap/sync_engine_tree.sh`를 사용하며 source hash를 run manifest에
보존한다. `PDMUX_TRUE_DUAL_WORKER=1`은 thread-local role capability가 없으면
시작을 거부한다.

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
`reports/dual_worker_design.md`. Submit the current R1 launcher from
`scripts/r1_dual_worker/`; job reports and raw artifacts belong under
`reports/r1_dual_worker/` and `results/r1_dual_worker/`, respectively.

## Stage 0 negative-control — pure Mamba2 (state-spaces/mamba2-2.7b), 2026-07-25

Goal: boot a PURE-SSM model (no attention/MLP) under the pdmux green-ctx
cudagraph path as the decode SM-insensitivity lower-bound anchor. Native
mamba_ssm checkpoint (single `pytorch_model.bin`, `backbone.*` keys, NO
`architectures`/`model_type`/tokenizer in config.json).

13. **NEW** `configs/mamba2.py` ← `src/configs/mamba2.py`. `Mamba2Config`
    **subclasses NemotronHConfig** (so `model_runner.mamba2_config` /
    `mambaish_config` isinstance gates engage with ZERO model_runner change),
    `model_type="mamba2_ssm"` (NOT "mamba2" — transformers v5 ships a built-in
    incompatible "mamba2" config; a distinct type avoids the suppress(ValueError)
    registration collision). Forces `layers_block_type=["mamba"]*n_layer` and
    sets `self.n_groups` (NemotronH.mamba2_cache_params reads it; real NemotronH
    JSON carries it, pure-mamba wrapper does not).

14. **NEW** `models/mamba2.py` ← `src/models/mamba2.py`. `Mamba2ForCausalLM`
    **subclasses NemotronHForCausalLM**; reuses forward / forward_split_prefill /
    cuda-graph mamba hooks. Overrides only `load_weights` for native keys:
    `backbone.`→`model.`, `.embedding.`→`.embed_tokens.`, `mixer.A_log`→`mixer.A`
    (a_weight_loader computes A=-exp(A_log)); `lm_head.weight` skipped (tied).
    `EntryClass=[Mamba2ForCausalLM]`. Key-map verified vs real 2.7b .bin
    (579 keys, 0 unmapped, 0 uncovered).

15. Tracked patch `src/patches/mamba2_pure_ssm_arch.patch` (4 existing files):
    (a) `configs/__init__.py` — import + `__all__` add `Mamba2Config`.
    (b) `utils/hf_transformers_utils.py` — add `Mamba2Config` to the
        `from sglang.srt.configs import (...)` block AND to `_CONFIG_REGISTRY`
        list (so AutoConfig routes model_type "mamba2_ssm" → Mamba2Config).
    (c) `server_args.py` — new `elif model_arch in ["Mamba2ForCausalLM"]` branch
        calling `_handle_mamba_radix_cache(support_mamba_cache=True,
        support_mamba_cache_extra_buffer=False, sm100_default_attention_backend=
        "triton")`. Unlike NemotronH it does NOT forbid the triton attn backend
        (there are zero attention layers; the full-attn sub-backend is created
        but never dispatched to).
    (d) `model_executor/model_runner_kv_cache_mixin.py` — **cell_size==0 guard**
        in `profile_max_num_token`: pure-SSM has 0 attention layers ⇒
        num_layers=0 ⇒ cell_size=0 ⇒ ZeroDivisionError at
        `int(rest_memory*(1<<30)) // cell_size`. Guard sizes the token pool by
        `max_mamba_cache_size * context_len` instead (per-request cost is the
        fixed mamba state, not per-token KV).

`sync_engine_tree.sh` installs files 13–14 and applies patch 15 (grep-guarded on
the server_args branch); all four runtime files added to the SHA-256 manifest.

**Converter** `scripts/models/convert_mamba2_native.py`: writes an HF-format
wrapper dir (config.json model_type=mamba2_ssm + arch=Mamba2ForCausalLM, dims
read from the actual .bin shapes) + symlinks `pytorch_model.bin` (NO rewrite) +
copies the GPT-NeoX-20B tokenizer if resolvable. Built wrapper:
`hf_cache/mamba2-2.7b-sglang/`.

**Boot smoke (drop-in to stage0_smoke.sbatch):**
`MODEL=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/mamba2-2.7b-sglang`
plus `--tokenizer-path EleutherAI/gpt-neox-20b` (native repo ships no tokenizer;
NOT in offline cache — must be provided). `--dtype float16` (checkpoint is fp16).

**UNVERIFIED (GPU-only, experiment-runner):** actual boot + green-ctx cudagraph
capture + decode replay with ZERO attention layers is not yet validated. Known
risks beyond the cell_size guard: the full-attn sub-backend init + whole-decode
cudagraph capture path have never run with an empty attention-layer set. If boot
fails, fallback = NemotronH mamba-dominant config (mostly-M hybrid_override).

## Stage 0 negative-control — pure Mamba2 zero-attention crash #2 (2026-07-25)

Second zero-attention boot crash (after cell_size==0). `TritonAttnBackend.__init__`
(triton_backend.py:103) probes `token_to_kv_pool.get_v_head_dim()` unconditionally
for mambaish models; for pure-Mamba2 the `HybridLinearKVPool.full_kv_pool` has an
empty `v_buffer` (0 full-attention layers) => `get_value_buffer(0)` IndexErrors in
`init_attention_backend()`, before green-ctx/cudagraph capture.

16. `mem_cache/memory_pool.py` — `HybridLinearKVPool.get_v_head_dim` (memory_pool.py
    :1412) guarded: `if self.full_layer_nums == 0: return self.head_dim`. The
    attention backend is still fully constructed (and the pdmux
    `decode_attn_backend_group` too) but never dispatched to (zero RadixAttention
    layers); the returned head_dim only sizes unused decode scratch buffers
    (self.v_head_dim at triton_backend.py:290/450). NemotronH/Zamba2
    (full_layer_nums>=1) take the unchanged branch => no hybrid regression. Only
    other `get_v_head_dim` caller is aiter_backend.py (AMD, unused on A100).
    Appended to `src/patches/mamba2_pure_ssm_arch.patch`; memory_pool.py added to
    the sync SHA-256 manifest.

## P5 instrumentation rework — Zamba2 per-layer-type timing (2026-08-04)

Audit of job 858811 (`results/r0c/decode_knee_vs_ctx.*`) confirmed three
**instrumentation** defects. **Model numerics unchanged** — this edit touches
measurement only.

17. `models/zamba2.py` — instrumentation rewritten (item 2 still covers re-apply,
    and the file is now **installed by `sync_engine_tree.sh` and in the SHA-256
    manifest**, so it is no longer a manual copy):
    - **Buckets symmetrised.** `_ZT_BUCKETS = ("attn","mlp","other","mamba")`,
      disjoint and covering the layer loop. `attn` = the WHOLE
      `Zamba2Attention.forward` (qkv_proj + adapters + core + o_proj), wrapped
      from `Zamba2AttentionDecoderLayer.forward`; `mlp` = the whole
      `Zamba2MLP.forward`; `other` = layernorms, the concat, the hybrid
      `.linear`, the residual adds and the output alloc; `mamba` = the whole
      `MambaMixer2` call (**definition unchanged** => mamba numbers stay
      comparable with 858811). Pre-fix, `attn` covered only the RadixAttention
      core and everything else was unbucketed.
    - **`_ZT_DIAG = ("attn_core",)`** reproduces the pre-fix `attn` definition,
      NESTED inside `attn` and EXCLUDED from the closure sum, so old and new runs
      can be compared directly.
    - **Closure gate.** A separate event pair times the whole layer loop;
      `closure = sum(_ZT_BUCKETS)/loop` is emitted per block and per mode, and a
      `ZBLT_CLOSURE_FAIL` warning fires below `SGLANG_ZAMBA_CLOSURE_MIN`
      (default 0.95). Deliberately NOT a timeline partition (that would be an
      identity, cf. methodology gate on identity-valued gates).
    - **Block accumulator.** `self._zt_blk` is reset at EVERY emit; the legacy
      running mean (`self._zt_acc`, reset only on mode change) is KEPT and
      emitted alongside, so pre-fix reports stay reproducible.
    - **Shape guard.** `(bs, ntok)` is read every timed forward; a block whose
      shape changes is emitted with `dirty=1`.
    - **`SGLANG_ZAMBA_TIMING_EVERY`** overrides the emit period (legacy default
      30 decode / 4 prefill forwards).
    - **Emit tag is now `ZBLT2` / `ZBPT2`.** The `attn` bucket means something
      different, so a pre-2026-08-04 `grep "ZBLT mode="` must MISS rather than
      silently read a redefined quantity. Consequence: these existing harnesses
      produce EMPTY result lines if rerun and need their grep updated first —
      `results/r0c/decode_knee_vs_ctx.sbatch` (superseded, banner added),
      `results/r0c/r0c_knee_isolate.sbatch`, `triage/p1_4_zamba2_ctxsens.sbatch`,
      `triage/p1_7_zb_smsens.sbatch`, `results/prefill_knee/prefill_knee.sbatch`,
      `results/prefill_knee/knee2d.sbatch`, `results/prefill_knee/knee2d_wide.sbatch`
      (+ `analyze_knee2d_wide.py`).
    - Instrumentation-OFF path short-circuits on a single `_ZT["on"]` dict read
      per layer (no lambda allocation, statement sequence unchanged); CUDA events
      are pooled/recycled; the layer loop is wrapped in `try/finally` so a raising
      layer cannot leave the global timing flag set.

    Harness: `results/r0c/decode_knee_vs_ctx_v2.sbatch` (randomised SM-arm order
    from a recorded seed, one discarded warm-up round per arm, all block emits
    kept, per-cell closure + O(1)-in-ctx negative-control record, 4 ctx x 5 SM x
    n=3). Analyzer `results/r0c/analyze_decode_knee_vs_ctx_v2.py` refuses to print
    a knee unless both gates pass. GPU correctness gate (prepared, unsubmitted):
    `results/r0c/zamba2_timing_smoke.sbatch`. CPU regression:
    `tests/test_zamba2_instrumentation.py`.

## Gate 2 direct head-of-line-blocking probe (2026-08-06)

18. **NEW** `multiplex/holb_probe.py` ← `src/multiplex/holb_probe.py`, installed by
    `sync_engine_tree.sh` and hashed in the manifest.
19. Tracked patch `src/patches/holb_probe_scheduler_hooks.patch` (1 existing file,
    `managers/scheduler.py`; grep-guarded on `maybe_create_holb_probe`, idempotent):
    - import `maybe_create_holb_probe`;
    - class-level `Scheduler.holb_probe = None` (so `run_batch` can read the
      attribute unconditionally);
    - `self.holb_probe = maybe_create_holb_probe(self)` at the end of `__init__`;
    - three `begin/end` brackets inside `run_batch`, one per forward-dispatch
      branch: overlap generation, pdmux `forward_batch_split_prefill`, non-overlap
      generation. All three use the ambient `torch.cuda.current_stream()`, so the
      hook has **no arm-specific branch** -- it is the same code for
      `event_loop_overlap`, `event_loop_normal` and `event_loop_pdmux`.
    - **DEFAULT OFF.** Without `PDMUX_HOLB_PATH` the three sites are two
      `is not None` checks per forward and no probe object exists.
    - `managers/scheduler.py` enters the SHA-256 manifest for the first time with
      this change: manifests written before 2026-08-06 simply lack that line, so
      line-for-line comparison of the pre-existing entries (e.g. against jobs
      873944/873945) is unaffected.
    Design + arm-symmetry argument + known biases:
    `results/p1_gates/gate2/DIRECT_BLOCKING_DESIGN.md`.
    CPU regression: `tests/test_holb_probe.py`.

## CP-0 prerequisite P1: chunked-prefill instrumentation (2026-08-28)

20. **NEW** `multiplex/chunk_probe.py` <- `src/multiplex/chunk_probe.py`, installed
    by `sync_engine_tree.sh` and hashed in the manifest. Produces the two counters
    `PREREG_CP0_2026-08-28.md` section 2.1 registers (`n_requests_chunked`,
    `n_chunk_events`) plus per-forward extend tokens, for **every** arm. Before
    this the tree had no chunked-prefill observation outside the pdmux mixin
    (`grep -rn "num_chunked|chunk_count|chunked_prefill_count" sglang/srt/` -> 0).
21. Tracked patch `src/patches/chunk_probe_scheduler_hook.patch` (1 existing file,
    `managers/scheduler.py`; grep-guarded on `maybe_install_chunk_probe`,
    idempotent). **ONE hunk, two statements**, anchored on
    `self.init_deterministic_inference_config()`:
    - a function-local `from sglang.srt.multiplex.chunk_probe import
      maybe_install_chunk_probe`;
    - `self.chunk_probe = maybe_install_chunk_probe(self)`.
    - **DEFAULT OFF, and stronger than the holb precedent**: without
      `PDMUX_CHUNK_PROBE_PATH` the function returns `None` *before installing
      anything*, so `PrefillAdder.add_one_req` / `add_one_req_ignore_eos` /
      `add_chunked_req` and `Scheduler.run_batch` remain the pristine engine
      functions -- there is not even an `is not None` test on the admission path.
      When ON, the instrumentation is five method wrappers installed at scheduler
      construction; `uninstall_wrappers` restores the originals (used by the CPU
      tests, not by the engine).
    - **WHY THIS ANCHOR.** The first draft inserted next to the holb hook and
      thereby broke `test_holb_probe.py::TestSchedulerHooksInstalled::
      test_patch_reapplies_the_hooks_and_nothing_else`: an insertion inside a
      mirrored patch's context window costs that patch its reversibility. The
      existing test caught it; the anchor is now outside both holb windows and
      `tests/test_chunk_probe.py::TestSchedulerHookInstalled` pins the separation
      and re-applies both patches together.
    - **WHY THE SCHEDULER LAYER.** `server_args.py:1254-1259` derives
      `piecewise_cuda_graph_max_tokens` from `chunked_prefill_size`, `:1397-1415`
      keeps only capture sizes `<= max_tokens` (empty list for cps -1), and
      `model_runner.py:2486-2490` then disables piecewise CUDA graph. The negative
      control (`--chunked-prefill-size -1`) is therefore the one arm whose prefill
      runs eager, so a counter inside a graph-captured prefill path would let it
      pass for the wrong reason (acceptance test P1-g).
    Mechanism list, detection argument and the counted/not-counted enumeration:
    the module docstring of `src/multiplex/chunk_probe.py`.
    CPU regression: `tests/test_chunk_probe.py` (drives the **real**
    `PrefillAdder` with stub caches), `tests/test_cp0_p1_tools.py`.
    Offline tools + acceptance harness: `results/cp_baseline/`
    (`analyze_chunk_probe.py`, `check_p1_acceptance.py`, `p1_accept_workload.py`,
    `p1e_expectations.json`, `p1_accept.sbatch` -- **written, not submitted**).

## R2 admission-limit latch fix (2026-09-11)

22. `multiplex/multiplexing_mixin.py` <- `src/multiplex/multiplexing_mixin.py`
    (installed by `sync_engine_tree.sh`; manifest hash `25d170e0...` ->
    `fdea4c32...`). Fixes the stale-True `r2_admission_limited` latch
    (`reports/r2_decoupling_review_2026-07-24.md` item 4), whose fix had been on
    hold until the 2026-09-11 return to the R2 track.
    - **Defect.** The latch is written only by `_r2_decide_idx`, which runs only
      while a split prefill batch is in flight, and read only by
      `update_split_prefill_batch`, which reaches the check only when no split
      prefill batch is in flight. Once the last in-span decision said "limited",
      admission stopped for the rest of the run.
    - **Change.** Two same-line edits plus two helpers appended at the end of the
      class (no earlier line moves, so `scripts/discipline/line_citations.json`
      stays valid: `--check --all` 50 compared, 0 violations):
      the R2 gate in `event_loop_pdmux` also fires on decode-only iterations
      while the latch is set (`_r2_admission_recheck`), and
      `update_split_prefill_batch` releases the latch when the running batch is
      empty (`_r2_admission_holds`). New telemetry events, emitted only when the
      latch is set: `r2_admission_recheck`, `r2_admission_released`.
    - **Scope.** `FixedPolicy` never sets `admission_limited`, so
      `PDMUX_R2_POLICY` unset/fixed runs are unchanged (latch stays False, no
      new events). Only generic/hybrid (`CoarseGrainedController`) are affected.
      cudagraph: no new stream group, capture shape or eager path. For
      generic/hybrid a decode-only R2 decision may now switch the partition
      (drain + index change, the same code as an in-span switch); decode then
      replays the graph captured for that stream index (`cuda_graph_runner`
      key `f"{stream_idx}_{bs}"`).
    - **Rejected alternative.** Clear-on-drain: every read happens after a drain,
      so it would make the latch False at every read, i.e. delete the admission
      limit.
    CPU regression: `tests/test_r2_admission_latch.py` (drives the real
    `event_loop_pdmux` on CPU; `PDMUX_MIXIN_UNDER_TEST=<file>` runs it against a
    variant file without touching this tree).
    GPU correctness harness (legacy vs `PDMUX_TRUE_DUAL_WORKER=1`, fixed D44,
    cudagraph ON): `results/r2_correctness/` -- **written, not submitted**.

## True-dual split-prefill ownership fix (2026-09-11, GPU job 907032)

23. `multiplex/multiplexing_mixin.py` <- `src/multiplex/multiplexing_mixin.py`
    (manifest hash `fdea4c32...` -> `0b88c07c...`). Fixes the crash that killed
    both `PDMUX_TRUE_DUAL_WORKER=1` boots of `results/r2_correctness/job_907032`
    on their first split prefill (the server's own warm-up request):
    `AttributeError: 'NoneType' object has no attribute 'forward_mode'`.
    - **Defect.** `event_loop_pdmux` submitted the prefill chunk to the prefill
      worker thread and then, without waiting, advanced the same ScheduleBatch
      (`split_prefill_finished`, `split_index = next_split_index`). The worker's
      `run_batch` -> `tp_worker.forward_batch_split_prefill` builds
      `split_forward_batch` only when it reads `split_index == 0`; the scheduler
      thread's stores won the race, so the first chunk read a non-zero index and
      passed `split_forward_batch=None` to the model forward. The legacy loop
      never raced (synchronous `run_batch`: read, then advance). Present since
      the R2 true-dual wiring; independent of item 22 (reproduced on CPU on both
      `25d170e0` and `fdea4c32`).
    - **Change.** The advance is applied only after `prefill_future.result()` on
      the true-dual path; the legacy statements and their order are unchanged
      (they now sit under `if prefill_future is None`). No GPU-side change, no
      new stream group, capture shape or eager path.
    CPU regression: `tests/test_true_dual_prefill_ownership.py` (deterministic
    eager/deferred task interleavings + real worker threads + legacy
    bookkeeping equivalence); loop fakes shared with `test_r2_admission_latch.py`
    via `tests/pdmux_loop_fakes.py`.

## Hybrid model files brought under sync + manifest (2026-09-13, GPU 0)

24. `models/{nemotron_h,falcon_h1,granitemoehybrid}.py` — items 6, 8, 9 above
    were **manual copies** and were **absent from the hash manifest**, so any
    campaign served by NemotronH / Falcon-H1 / Granite-4 recorded no provenance
    for the model implementation it ran — including the `forward_split_prefill`
    methods that are the reason PD-mux SPLIT_PREFILL works on those models at
    all. `scripts/bootstrap/sync_engine_tree.sh` now installs all three from
    `src/models/` and hashes them.
    - Verified **byte-identical** before the change
      (`sha256 src/models/X.py == sha256 <dev tree>/sglang/srt/models/X.py` for
      all three: nemotron_h `713333e8…`, falcon_h1 `3fb851ad…`,
      granitemoehybrid `b97f8312…`), so the install is a no-op on the current
      tree and a repair on any rebuilt one. No dev-tree bytes changed.
    - Also added as **hash-only** (pristine upstream, NOT installed):
      `configs/{nemotron_h,falcon_h1,granitemoehybrid}.py` and
      `configs/mamba_utils.py`. These decide the served geometry —
      `configs/nemotron_h.py` maps `hybrid_override_pattern` to
      `layers_block_type` (which layer index is attention vs mamba) and
      `mamba_utils.Mamba2StateShape` turns that into `mamba_cache_per_req`, i.e.
      how much state one in-flight request holds, which is what a capacity
      (lambda*) label rests on.
    - **Manifest: 17 -> 24 entries, appended at the end.** The first 17 lines and
      their order are unchanged, so `sha256sum -c` on a pre-2026-09-13 manifest
      (e.g. `results/r2_correctness/job_907100/runtime_source_manifest.sha256`)
      checks exactly the same 17 files as before. Cross-job comparisons must now
      read as "the original 17 still agree **and** 7 new entries exist", not
      "17/17 identical".
