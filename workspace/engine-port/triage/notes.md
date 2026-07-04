# P0 Triage — running measured-facts log

> Evidence trail for `reports/p0_triage.md`. Every line labeled `[measured]` (from code/exec) or `[derived]` (inference).

## T1 — Codebases acquired (login node glogin01, network OK)

- `[measured]` MuxWise github fork `ykcombat/sglang`, branch `slo_config`, HEAD `eeac5148f "upload configuration"`. Clone: `external/muxwise` (full history).
- `[measured]` `slo_config` branch DOES exist (prompt was right); repo has many `pdmux_*` branches (`pdmux_scheduler`, `pdmux_cuda_graph`, `slo_scheduler`, `greenctx_stream`, `test_green_ctx`, …). `feat-pdmux-scheduler` is repo HEAD.
- `[measured]` Zenodo record 18062118 = single `muxwise.zip` (10 MB, CC-BY-4.0, title "MuxWise: Artifact"). Extracts to `sglang-slo_config/`. `python/sglang` tree is **byte-identical** to github `slo_config` HEAD (diff -rq = 0). → github clone is canonical (adds git history). `external/muxwise-zenodo/`.
- `[measured]` Bullet `zejia-lin/BulletServe`: `external/bullet`, main HEAD `445afae`, branch `asplos26ae` fetched (ASPLOS'26 artifact-eval branch).
- `[measured]` Latest SGLang `sgl-project/sglang`: `external/sglang-latest`, HEAD `926140d789`.
- `[measured]` LICENSES: all three Apache-2.0 (Bullet & MuxWise inherit SGLang's Apache-2.0). MuxWise adds no separate license file beyond SGLang's.
- `[measured]` Docker `combathhhhhh/pdmux:sglpr_torch2.6_bench` — NOT pulled. Tag says torch 2.6; consistent with MuxWise commit `3bf0ca434` warning "perf issues on torch > 2.6.x". But MuxWise `pyproject.toml` pins **torch==2.8.0** (not 2.6). → torch 2.6 is the *recommended perf* base, 2.8 is the *build* pin.

## T2 — MuxWise deep analysis

### T2-1 Base version
- `[measured]` MuxWise = SGLang **0.5.3rc0** (`python/sglang/version.py`, `pyproject.toml`). Pins: `torch==2.8.0`, `flashinfer_python==0.4.0rc1`, `torchao==0.9.0`.
- `[measured]` merge-base(slo_config, upstream main) = `608854821` dated **2025-09-25** ("Update CODEOWNERS…").
- `[measured]` slo_config = **47 commits** on top of that base.
- `[measured]` latest SGLang is **9209 commits** ahead of the merge-base (~9 months). Latest pins `torch==2.11.0`, `flashinfer_python==0.6.12`. → LARGE base divergence 0.5.3rc0→(latest ~0.5.x/gateway line), torch 2.8→2.11, flashinfer 0.4→0.6.

### T2-2 Hybrid-mamba infra in MuxWise (base 0.5.3rc0)
- `[measured]` PARTIAL scaffolding present: `HybridReqToTokenPool`(4 files), `HybridLinearKVPool`(2), `MambaPool`(1), `conv_state`/`ssm_state`. Files: `srt/mem_cache/memory_pool.py`, `srt/managers/schedule_batch.py`, `srt/model_executor/model_runner.py`, `srt/layers/attention/hybrid_linear_attn_backend.py`.
- `[measured]` mamba layer dir has ONLY: `causal_conv1d.py`, `causal_conv1d_triton.py`, `mamba.py`. `mamba.py` = just `mamba_v2_sharded_weight_loader` helper (NOT a Mamba2 SSD mixer). **No `ops/` SSD dir, no `MambaMixer2`.**
- `[measured]` Only hybrid MODEL = `Qwen3Next` (7 files) — **gated-deltanet linear attention**, not Mamba2 SSD. `NemotronH`=0, `FalconH1`=0, `Zamba2`=0.
- `[derived]` Zamba2 uses **Mamba2 (SSD selective scan)**; MuxWise base has the hybrid pools/conv1d but NOT the Mamba2 SSD kernel stack. So strategy A must backport the SSD ops + Mamba2 mixer (from a newer SGLang) *in addition to* writing Zamba2.

### T2-3 GreenContext layer  ★ pivotal
- `[measured]` SM control uses compiled `sgl_kernel.spatial` (NOT repo's ctypes plan): `spatial.get_sm_available(gpu_id)`, `spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)`. Files: `sgl-kernel/csrc/spatial/greenctx_stream.{cu,h}`, `spatial_extension.cc`, `python/sgl_kernel/spatial.py`.
- `[measured]` **`sgl_kernel.spatial` exists in BOTH MuxWise AND latest SGLang, and `greenctx_stream.cu` + `spatial.py` are BYTE-IDENTICAL.** It's UPSTREAM: landed via SGLang PRs #7649 "CUDA Green Context Support", #8701 (cuda<12.4 fix), #9231 (optional extension). → SM-control mechanism is fully upstream & maintained. Strategy B needs ZERO green-ctx porting. Obsoletes repo's planned `green_ctx_controller.py` (ctypes).
- `[measured]` **SM count is NOT H200-hardcoded.** `pdmux_context.py`: `total_sm_count = spatial.get_sm_available(gpu_id)` (device query). `get_arch_constraints(cc)`: major==8 (A100)→(min_per_part=4, multiple=2); major==9 (Hopper)→(8,8); explicitly supports A100. `divide_sm()` computes partitions from queried SM count.
- `[measured]` A100-adaptation point (config, not code): shipped `sharegpt.yml`/`loogle.yml` `manual_divisions` are **H200-tuned** (every entry sums to 132 SM, e.g. `[112,20], [104,28]…`). For A100 (108 SM) → re-profile divisions OR drop `manual_divisions` and use auto `divide_sm()`.

### T2-4 3-module boundaries
- `[measured]` **Engine (bubble-less multiplex)**: `srt/multiplex/multiplexing.py::SchedulerMultiplexMixin.event_loop_pdmux()` (NEW, +195). Runs prefill_stream ∥ decode_stream (green-ctx pair). Prefill is **layer-wise split** via `ForwardMode.SPLIT_PREFILL` (forward_batch_info.py:90), advancing `split_index` by `split_forward_count = split_forward_token_budget // extend_num_tokens` layers/iter over `num_hidden_layers`. `Scheduler`(scheduler.py:223) inherits the mixin (line 230); dispatched at scheduler.py:2868 `event_loop_pdmux()`.
- `[measured]` **Dispatcher (SLO-aware)**: `adjust_stream_groups()` — picks `stream_idx` (=SM partition) from `decode_bs`: manual `decode_bs_threshold` table OR `stream_idx = decode_bs*(N-2)//decode_bs_divisor`. Marked `TODO: temporary demo`. Also calls `model_runner.update_decode_attn_backend(stream_idx)` (model_runner.py:1871) — per-partition attn backend/workspace.
- `[measured]` **Estimator (contention-tolerant)**: NO online predictor code in slo_config (grep contention|estimator|predictor|roofline|solo_run = 0 hits). Realized as **offline `manual_divisions`** in YAML (`PDMuxConfig`, pdmux_context.py). → paper's "estimator" = offline profiling, not runtime module in this branch.
- `[measured]` New module `srt/multiplex/` (multiplexing.py + pdmux_context.py) is self-contained; core hooks are small edits to scheduler.py(+39), model_runner.py(+34), tp_worker.py(+26), parallel_state.py(+32, `set_pdmux_status`), pynccl.py(+38), schedule_batch.py(+13), flashinfer_backend.py(+10), server_args.py(+37).

### T2-5 CUDA graph × partition
- `[measured]` Decode CUDA graphs captured **per (stream_idx, batch_size)**: graph key `f"{current_stream_idx}_{cuda_graph_bs}"` (cuda_graph_runner.py, +161). Capture loops over each partition's decode stream (`decode_stream_groups`). → graph memory scales × N_partitions.

### T2-6 Change scope
- `[measured]` `git diff --stat 608854821..slo_config` = **31 files, +2396 / −91**. Core-engine footprint ≈ 11 files; the rest is benchmark scripts (`benchmark/pdmux/bench_serving.py` +1055), docs, yml, shell, 1 PNG. → MuxWise's engine change is **localized & bounded**.

### T2 — forward-port (B) conflict flags
- `[measured]` `--enable-pdmux` asserts incompatibility with: **overlap schedule**, chunked prefill, disaggregation, pp_size>1 (server_args.py:2746+). Latest SGLang defaults to overlap scheduler → real forward-port conflict (must re-target `event_loop_pdmux` against latest's overlap/scheduler refactor).
- `[measured]` torch>2.6.x perf-degradation warning for pdmux; latest uses torch 2.11 → perf risk for green-ctx streams (needs measurement, ties to T5).

## T3 — Bullet analysis (design-borrow only; Bullet REJECTED as base)

- `[measured]` Bullet = SGLang fork; enabled via `--enable-bullet-engine`; needs **libsmctrl (CUDA ≤ 12.6)** + **MPS** (`scripts/start_mps.sh`). → unusable on CUDA-13 cluster (confirms rejection).
- `[measured]` **(1) libsmctrl surface**: 20 files touch smctrl; python swap-surface = `srt/bullet/sm_controller.py` (`_LibSMCtrl` ctypes: `set_global_mask`/`set_stream_mask` at **TPC granularity**, 2 SM/TPC; `ScheduleBudget(prefill_ratio, decode_ratio)`) + `csrc/src/libsmctrl*.c` + one ref in `tp_worker.py`. In our world this whole layer is *replaced by* MuxWise's `sgl_kernel.spatial` green-ctx (NOT borrowed) — so borrow the design above it, not this.
- `[measured]` **(2) 2-process MPS disaggregation**: `srt/bullet/` package — `launchers.py`/`loop_forever.py` (autonomous prefill & decode processes), IPC via `rpc_server.py`/`req_rpc.py`/`radix_cache_rpc.py`/`memory_pool_rpc_v2.py` + shared-mem `shared_nparray.py`/`shared_mng.py`/`ring_array.py`/`smem_mutex.py`. Metadata IPC = RPC + shared numpy arrays + mutex.
- `[measured]` **(3a) SRM estimator**: `srt/bullet/observability.py` — `PredictorInfoEntry(phase, prefill_len, decode_bs, decode_tokens, prefill_tpc, decode_tpc)` + `PredictorInfos` = SM-scaling roofline inputs (latency vs TPC/SM). Model timing instrumentation: `models/llama_timing.py`, `models/qwen2_timing.py`, `bullet/timing.py`, `bullet/model_monkey_patch.py`.
- `[measured]` **(3b) request timing / (ES,PS,RS)-style state**: `observability.py::ReqTimingState` (arrive/ttft/tpot/decode_duration) + `ReqTimingStateDict`. Async scheduling + reorder/delayed-decode signals appear in `managers/scheduler_output_processor_mixin.py` and `schedule_batch.py` (bullet-tagged), plus per-process `loop_forever.py`.
- `[measured]` **(4) ablation**: `artifact_evaluation/run_all.sh` compares `--enable-bullet-engine` (full) vs vanilla sglang `--chunked-prefill-size 1024` (baseline) over request-rate sweep 10..25, sharegpt, Llama-3.1-8B. `plot.py` renders. (Finer Naive/wPartition/wScheduler ablation is the paper's; the AE script ships the full-vs-chunked comparison.)
- `[derived]` **Borrow list (design, not code)**: (i) SRM roofline predictor shape from `observability.PredictorInfoEntry` to derive our layer-type-aware SM floors; (ii) 2-engine async P∥D decoupling as the mental model (but MuxWise already realizes P∥D via green-ctx streams in ONE process — simpler, so we keep MuxWise's single-process split-prefill loop, not Bullet's 2-process MPS); (iii) request reordering/delayed-decode as an optional dispatcher refinement. Nothing from Bullet's libsmctrl/MPS layer is portable to CUDA-13.

## T4 — Latest SGLang analysis (strategy-B base)  ★★ decisive

### T4-1/T4-3 PD-multiplexing is ALREADY UPSTREAM (MuxWise mechanism merged into SGLang mainline)
- `[measured]` Latest SGLang HEAD `926140d789`; `git describe`=`gateway-v0.3.1-6011-g926140d789`; pins torch==2.11.0, flashinfer 0.6.12.
- `[measured]` Latest already contains the full MuxWise PD-mux engine: `srt/multiplex/multiplexing_mixin.py::SchedulerMultiplexMixin` (`event_loop_pdmux` L96, `adjust_stream_groups` L49) + `srt/multiplex/pdmux_context.py` (`PDMuxConfig`, `manual_divisions`, `divide_sm`, `get_arch_constraints`, `spatial.get_sm_available`, `spatial.create_greenctx_stream_by_value`). Same design as MuxWise `slo_config`.
- `[measured]` `Scheduler` inherits the mixin (scheduler.py:301) and dispatches `event_loop_pdmux()` (scheduler.py:4173) under `enable_pdmux`; `SPLIT_PREFILL` mode (forward_batch_info.py:99, split_index L477); pdmux split handled in run_batch (scheduler.py:3294); `enable_pdmux`/`pdmux_config_path` in server_args (L2416).
- `[measured]` Provenance: introduced upstream via PRs **#11592** "[Feature] PD-Multiplexing Context and Scheduler" and **#12275** (lazy import spatial). Kept compatible through later scheduler refactors (#25609/#25610 request-ingress; type-hint mixin #15916). Even integrated with eagle speculative cuda-graph runners (pdmux refs in `speculative/*_cuda_graph_runner.py`).
- `[measured]` Green-ctx kernel `sgl_kernel.spatial` upstream (PRs #7649/#8701/#9231); `greenctx_stream.cu`+`spatial.py` byte-identical to MuxWise (see T2-3). PR #8701 fixed cuda<12.4 incompatibility → CUDA 12.4+ supported (cluster has CUDA 13.0 ✓).
- `[derived]` ⇒ **Strategy B's "forward-port MuxWise 3 modules" cost ≈ 0**: the modules already live in latest, maintained against latest scheduler/overlap/speculative code. The prompt's B framing (manual forward-port) is superseded by "adopt upstream pdmux". The overlap-scheduler conflict flagged in T2 is already resolved upstream (pdmux coexists as its own event loop; server_args still asserts overlap-incompat, i.e. pdmux mode disables overlap by design).

### T4-2 Zamba2 port surface on latest (reuse vs new-write)
- `[measured]` REUSE (exists in latest): `MambaMixer2` (layers/attention/mamba/mamba.py:191) = full Mamba2 SSD mixer; `Mamba2AttnBackend`; full SSD `ops/` (ssd_chunk_scan/combined/state_passing/bmm/chunk_state); `RadixAttention`; `HybridLinearKVPool`/hybrid pool + `MambaPool` wiring; **`hybrid_override_pattern` per-index layer-class selection** (NemotronHModel:760, `ALL_DECODER_LAYER_TYPES[pattern[idx]]`) — this is exactly Zamba2's ABAB temporal-interleave scaffold.
- `[measured]` Closest template = **NemotronH** (temporal hybrid: `NemotronHMambaDecoderLayer` uses `MambaMixer2`; `NemotronHAttentionDecoderLayer` uses `NemotronHAttention`+`RadixAttention`; layers picked per-index from pattern). FalconH1 = spatial/parallel (attn+mamba same layer) — the layer-aware "not applicable" control.
- `[derived]` NEW-WRITE for Zamba2 specifics (NOT in NemotronH/FalconH1): (i) **shared attention+MLP block** reused across positions (weight sharing; NemotronH has per-layer independent weights); (ii) **per-invocation LoRA** on the shared block; (iii) Zamba2 `hybrid_override_pattern`/config class; (iv) ABAB schedule wiring to the shared block. Reference implementation exists in **vLLM** (`vllm/model_executor/models/zamba2.py`) — repo already runs vLLM Zamba2 (vllm_bench/). Port = translate vLLM Zamba2 logic into SGLang model API atop reused `MambaMixer2`+NemotronH scaffold.
- `[derived]` Strategy A equivalent on MuxWise 0.5.3rc0 would additionally require backporting MambaMixer2 + SSD ops/ + Mamba2AttnBackend + NemotronH-style scaffold (all ABSENT at 0.5.3rc0, T2-2) BEFORE writing Zamba2 → strictly more work than B.

### T4 — research-delta note (not a P0 gate item)
- `[derived]` Our layer-type-aware contribution hooks the **decode path**: pdmux currently sets one decode SM partition per scheduler iteration (whole model). Layer-type-aware = switch green-ctx partition per layer-type window (reserve floor on attn-decode layers, release to prefill on ssm-decode layers). The green-ctx stream infra + per-(stream_idx,bs) CUDA-graph structure exist; the extension = per-layer-type stream switching on decode, which interacts with mamba2 layers that ONLY latest has → further favors B.

## T5 — A100 green-ctx smoke (job 824014, EXIT 0, SUCCESS)

- `[measured]` Scoping: instead of full server boot (torch-2.8/flashinfer-0.4 on CUDA13 = high build friction, would eat 2h budget), JIT-compiled the REAL unmodified upstream `greenctx_stream.cu` (sha256 de20703f) with cluster torch — zero installs, TORCH_EXTENSIONS_DIR isolated to scratch. Harness: triage/t5_greenctx/{test_greenctx.py, spatial_reg.cc, run_t5.sbatch, t5_result.json}.
- `[measured]` Env: A100-SXM4-80GB, cc 8.0, driver 580.105.08, CUDA 13.0, torch 2.9.1+cu130, nvcc 13.0.88.
- `[measured]` check1 get_sm_available=108 ✓ (pure-python path). check2 JIT build 75.9s ✓; create_greenctx_stream_by_value(64,44,0) → actual smA=64, smB=44 (exact 108 split), streams non-null, NO fallback warning ⇒ direct cuGreenCtxStreamCreate (cuda≥12.5) path. check3 matmul on both partition streams ✓.
- `[measured]` Pre-existing corroboration: repo results/stage2/ctx_switch_overhead_a100-sxm4-80gb.json already shows green_ctx backend, 108 SM, partitions 14..108, cpu_swap ~0.45µs on this exact A100.
- `[derived]` ⇒ green-ctx SM-control layer (the novel mechanism risk) is EMPIRICALLY VALIDATED on A100/CUDA13. Deferred to P1: full `--enable-pdmux` server boot (needs full sglang+sgl-kernel+flashinfer build; separate effort).

## P1.0 — latest-SGLang pdmux base on cluster (2026-07-02)

Base decision: **sglang v0.5.10** = last release on torch==2.9.1 (matches cluster), pins `sglang-kernel==0.4.1`, has pdmux + Mamba2/NemotronH. (v0.5.11+ bump to torch 2.11.) HEAD (torch 2.11 + kernel 0.4.4) avoided.
- `[measured]` sgl-kernel PyPI 0.3.21 wheel is **CUDA-12** (libnvrtc.so.12) → unusable on CUDA13. cu13 wheels only exist for `sglang_kernel` 0.4.x at github.com/sgl-project/whl (`+cu130` abi3). Kernel 0.4.0 lacks `awq_marlin_moe_repack` (API drift) → must match sglang↔kernel: **v0.5.10 ↔ kernel 0.4.1+cu130** (aligned).
- `[measured]` Isolated overlay venv `/scratch/ehmoon/whlee/sglang_engine_venv` (--system-site-packages, reuses conda torch 2.9.1+cu130 / flashinfer 0.6.10 / mamba_ssm 2.3.1 / cuda-python 13.2). sglang v0.5.10 + kernel 0.4.1+cu130 install (--no-deps), pure-python deps via constraints (freeze CUDA-13 stack). `import http_server` = OK.
- `[measured]` **GREEN-CTX PDMUX INIT SUCCESS on A100 (job 824469)**: `PD-Multiplexing enabled with 4 stream groups, sm_counts (prefill_sm, decode_sm): [(108,0),(74,34),(54,54),(0,108)]` — device-queried 108 SM (NOT H200 132), partitions correct. Confirms pdmux+green-ctx mechanism works on A100/CUDA13. (Extends T5: now the full pdmux init path, not just the kernel.)
- `[measured]` **BLOCKER for full serve: cluster conda is Python 3.14.** (a) torch.compile raises "not supported on Python 3.14+" (worked around via sitecustomize no-op for boot). (b) Triton 3.5.1 kernel compile dies: `AttributeError: module 'ast' has no attribute 'Num'` (ast.Num removed in py3.12). ⇒ py3.14 env cannot run sglang kernels end-to-end. **Need Python ≤3.13 env with CUDA-13 torch stack** for perf work.
- `[derived]` The torch>2.6 green-ctx perf caveat (report §8) is ALSO surfaced by v0.5.10's own warning at boot — still an open perf risk on torch 2.9.1.

## P1.0 — RESULT: PDMUX_BOOT_OK (job 826832) ✅

- `[measured]` sglang **v0.5.10** + **sglang_kernel 0.4.1+cu130** + cluster torch 2.9.1+cu130/flashinfer 0.6.10/mamba_ssm 2.3.1, py3.14. Server booted healthy (40s), `PD-Multiplexing enabled with 4 stream groups, sm_counts [(108,0),(74,34),(54,54),(0,108)]` (A100 108 SM, device-queried), served a /generate end-to-end (e2e 51ms, 16 tok; gibberish text = dummy weights, pipeline OK). Evidence: `p1_0_SUCCESS_evidence/`.
- **Patches applied (tracked; move to editable dev tree at P1.2):**
  1. `venv .../sitecustomize.py` — py3.14 shims: restore `ast.Num`→`ast.Constant` (triton 3.5.1 code_generator.py:1172/1174; faithful to py≤3.13) + `torch.compile` no-op (sglang defaults enable_torch_compile=False → perf-neutral). Both valid for perf runs.
  2. `venv .../sglang/srt/model_executor/model_runner.py:2402` — `getattr(forward_batch,"ngram_embedding_info",None)`: pdmux path passes ModelWorkerBatch (lacks attr) to `maybe_update_ngram_token_table` (typed ForwardBatch). No-op when ngram unused. **A genuine pdmux-path bug in v0.5.10** — candidate upstream fix.
  3. sbatch: `PATH=$HOME/.local/bin:$PATH` (ninja for flashinfer/triton JIT); `--chunked-prefill-size -1 --disable-overlap-schedule` (pdmux requires).
- `[measured]` Extra deps installed into venv (--no-deps + constraints freezing CUDA-13 stack): pybase64, IPython stack, pydantic(+core 2.46.4), orjson, uvicorn/uvloop, fastapi/starlette, pyzmq, openai, setproctitle, partial-json-parser, dill, sentencepiece, python-multipart, compressed-tensors, gguf, msgspec, jsonschema, xgrammar 0.2.3. (openai_harmony missing = non-fatal optional OpenAIServingResponses warning.)
- `[derived]` py3.14 is workable via the 2 faithful shims → **no py3.12 rebuild needed**. (A py3.13 env would avoid shims if ever preferred.)

## Temporal-hybrid model survey (sglang v0.5.10) — for layer-aware generalization (2026-07-03)
- `[measured]` Present + temporal + Mamba2 SSD (MambaMixer2): **NemotronH** (hybrid_override_pattern), **GraniteMoeHybrid** (layer_types). Present + temporal + non-SSD: **Qwen3Next** (gated-deltanet, layers_block_type).
- `[measured]` Present + **spatial** (attn∥mamba same layer, layer-aware N/A = control): **FalconH1**.
- `[measured]` ABSENT (would need port): **Zamba2** (target; Mamba2 + shared-attn+LoRA), Jamba (Mamba1+MoE), Bamba (Mamba2).
- `[derived]` ⇒ layer-aware SM-reservation (P1.3) is a *class* mechanism over temporal hybrids. **Build/validate it on NemotronH first (already runs)**; Zamba2 = headline (project's models) once ported; FalconH1 = spatial control. Broadens eval set beyond Zamba2 and de-risks P1.3 from the Zamba2 port.

## P1.2 — Zamba2 SGLang port: ZAMBA2_BOOT_OK (dummy weights, job 826938) ✅
- `[measured]` Wrote `sglang/srt/models/zamba2.py` (~330 lines) + `configs/zamba2.py` (Zamba2Config subclass w/ mamba2_cache_params, mamba_layer_ids=all, full_attention_layer_ids=hybrid_layer_ids, mamba_chunk_size alias). Registered: AutoConfig override (exist_ok) for model_type "zamba2"; EntryClass=[Zamba2ForCausalLM]; added Zamba2Config to configs/__init__ + model_runner `mamba2_config` gate.
- `[measured]` Dummy-boot debug loop (4 fixes): (1) hybrid backend not selected → added Zamba2Config to model_runner.py `mamba2_config` isinstance gate; (2) `mamba_chunk_size` missing → alias to `chunk_size`; (3) SSM triton kernel dtype mismatch bf16/fp16 → `--dtype bfloat16` (conv dtype default bf16); → **healthy 90s, served /generate e2e 0.38s through 54 layers (shared-attn+LoRA+Mamba2 SSD)**. Gibberish text = dummy weights; **full structure/pools/forward validated**.
- Next (P1.2c): real weights (hf_cache Zamba2-2.7B safetensors present) → test load_weights mapping → logit-parity vs vLLM.
- Patches now live in dev tree (`sglang_engine_dev`): model_runner.py (mamba2_config gate + import), configs/__init__.py, hf_transformers_utils.py (AutoConfig override), + new zamba2.py/configs/zamba2.py.

## P1.2c — Zamba2 real-weights: ALL WEIGHTS LOAD; forward-numerics bug remains (2026-07-03)
- `[measured]` Real-weights boot (job 826942/826959): server healthy + serves, but output = **all token 0** (empty text). Diagnostic (`ZAMBA2LOAD in=531 params=527 loaded=527 skipped=0 missing=0`) → **weight mapping is COMPLETE** (531 ckpt → 527 params via q/k/v→qkv fusion; every ckpt weight homed, every param loaded). My model's param names already match HF: shared blocks register at `layers.6/12.shared_transformer.*` (num_mem_blocks=2), pure mamba `layers.N.mamba.*`, hybrid mamba `layers.N.mamba_decoder.mamba.*`, per-position LoRA `...gate_up_proj_adapter_list.{block_idx}.{0→A,1→B}`.
- `[measured]` No NaN/inf in log; attention scale matches vLLM `(head_dim/2)**-0.5`. Dummy weights → varied tokens; real weights → token 0. ⇒ **semantic forward bug** (not weights, not crash).
- `[derived]` Candidates for the degenerate-logits bug: (1) mamba `A` param interpretation (A_log vs -exp(A_log)) — NemotronH uses same A_log→A rename so likely OK but verify sglang MambaMixer2 A weight_loader; (2) gated mamba `norm` inside mixer; (3) shared-attn KV routing / block_idx→layer_id per-position; (4) residual/concat flow in sglang flat-token layout. **Next: activation-parity harness** — feed one prompt to my sglang model vs vLLM/HF, compare embed → layer0 out → … to localize first divergence.
- **Milestone**: Zamba2 SGLang port RUNS end-to-end (structure/pools/forward/weight-load all validated); only forward-numerics parity remains.
- Repo copies (tracked): `workspace/engine-port/src/{models,configs}/zamba2.py`. Live/editable in `sglang_engine_dev`.

## P1.3 — pdmux on hybrid models: NemotronH (2026-07-03) ✅
- `[measured]` NemotronH-8B (24M/24-/4* temporal, head_dim 128): base boots+correct("Paris"); needs `--disable-piecewise-cuda-graph` on py3.14 (piecewise→torch._inductor→torch.ao `EdgeOrNode.__module__=` on typing.Union fails).
- `[measured]` **pdmux-hybrid gap**: `forward_split_prefill` implemented by 14 DENSE models only; hybrids (NemotronH/Zamba2) lack it → pdmux `AttributeError`. **Fixed by adding `forward_split_prefill` to NemotronHForCausalLM** (patch in src/patches/). ⇒ **pdmux runs on NemotronH** (job 827062): green-ctx 4 groups, split-prefill through mamba/attn/mlp, "Paris" correct. First pdmux-on-hybrid.
- Design for layer-aware (P1.3d) in `reports/p1_3_nemotronh_layer_aware.md`: split DECODE by layer-type window + per-window green-ctx partition switch (attn→floor, ssm→reclaim to prefill); eager decode on py3.14 makes per-layer stream switch easy. Policies fused/agnostic/layer_aware, NemotronH attn ids from hybrid_override_pattern.
- Open: Zamba2 needs forward_split_prefill too; regular cuda graph on py3.14 (inductor); triton hybrid layer_id=0 bug.

## P1.4 — layer-aware 정책 구현 + 4-정책 평가: 전제 반증 (2026-07-03) ★
- `[measured]` NemotronH decode per-layer-type SM 민감도(green-ctx로 decode를 N SM 고정, 배치32, 210스텝): **mamba/층 0.589→1.547ms(108→16SM, 2.6× = SM-민감·compute-bound SSD)**, attn/층 ~0.2ms 평탄(SM-둔감·GQA memory-bound), mlp 44SM↑ 둔감. step 구성=mamba62%+mlp32%+attn5%.
- `[derived]` ⇒ **sim의 layer-aware 전제(attn=비싼 SM-민감 희소층) 반증**: NemotronH는 mamba가 SM-민감·다수, attn은 둔감·희소. 환원가능 둔감층=attn 4/52뿐 → **layer-aware 순이득≈0, agnostic(균일예약)이 동등/우위**. 전제는 no-GQA attn 가정 의존 → 모델 아키텍처 의존 실증.
- 구현: NemotronHModel.forward에 per-layer green-ctx 스트림 전환(`_get_gctx_decode_stream`, sgl_kernel.spatial) + 데이터-구동 예약(`PDMUX_LA_RESERVE`) + per-layer-type 타이밍 계측(env-gated). 정책 스윕 harness `triage/p1_4_nh_smsens.sbatch`(1서버 파일-스윕; bare `wait`가 서버 대기하는 버그 수정=curl PID만 wait). 원자료 `p1_4_nh_smsens_827583.txt`.
- 한국어 보고서: `reports/p1_4_layer_aware_평가_kr.md`. 미측: 완전-충실 layer-aware 서빙(event_loop 대수술)·fused/co_schedule 동시경합 goodput·Zamba2 민감도(no-GQA attn 후보).

## P1.4 후속 — 시퀀스 길이 의존성 (2026-07-03) ★★
- `[measured]` **NemotronH(GQA/flashinfer) 컨텍스트×SM 스윕(job 827617)**: attn/층 full-SM 0.303→0.337→0.449ms (ctx≈350/2140/5100, +48%=O(L)). **attn SM-민감도 뒤바뀜**: ctx350 16SM=0.237(둔감/faster) → ctx5100 16SM=0.640 vs full 0.449(+43%, SM-민감). mamba/층 ~0.553 컨텍스트 불변(O(1))·항상 SM-민감(1.05@16). ⇒ 사용자 가설 실증: attn "싸고 SM-둔감"은 단문 국한; 장문서 무거워지고 SM-민감해짐. 단 GQA는 장문서도 mamba 지배(mamba13.3 vs attn1.8ms@ctx5100).
- `[measured]` **Zamba2(no-GQA/torch_native) 컨텍스트×SM 스윕(job 827601, 불완전)**: mamba/층 ~0.56(NemotronH와 일치✓), attn/층 ~1.5(torch_native 교란이라 백엔드-혼재, no-GQA 순효과 분리 불가). ctx3000서 CUDA illegal memory access(torch_native+green-ctx 장문 불안정). ⇒ Zamba2는 fast head_dim-160 백엔드 확보 전까지 청정 측정 불가(open item).
- `[derived]` **layer-aware 이득 = (GQA/no-GQA)×(컨텍스트) 2D**. 유리 구간 = 긴 컨텍스트 + no-GQA(비싼 attn) + 희소 attn-층. 보고서 §3.5 갱신.
- harness: `triage/p1_4_nh_ctxsens.sbatch`(NemotronH ctx×SM), `p1_4_zamba2_ctxsens.sbatch`(Zamba2). NHLT/ZBLT 로그에 ctxlen 추가. 원자료 `p1_4_nh_ctxsens_827617.txt`, `p1_4_zb_ctxsens_827601.txt`.

## P1.4++ — Zamba2 fast backend (triton fix) + clean no-GQA context sweep (2026-07-04)
- `[measured]` **triton backend hybrid 버그 수정**: triton_backend.py v_head_dim init `hybrid_gdn_config`→`mambaish_config` → Zamba2가 triton서 정확("Paris")+빠름(torch_native 탈출). NemotronH도 이제 triton 가능.
- `[measured]` **Zamba2 no-GQA 청정 측정(triton, job 830945)**: attn/층 full-SM 0.159(ctx348)→0.607(ctx2140)=~4× (GQA는 ~10%). attn SM-민감 이미 ctx348서 2.6×(0.159→0.407@16SM) vs GQA는 ctx≫2000까지 둔감. ⇒ **no-GQA는 attn이 훨씬 짧은 컨텍스트서 비싸지고 SM-민감** = layer-aware 유리 체제. ctx2140서 attn 9×0.607=5.5ms(step ~17%, 증가중)·희소(9/54).
- `[measured]` 장문(ctx≥2048) reduced-SM(44/16) 점은 green-ctx+triton서 **device-side assert crash**(open). full-SM 컨텍스트 스케일링은 확보.
- 보고서 §3.5/§3.6 갱신.

## P1.4+++ — Zamba2 long-ctx COMPLETE (green-ctx race fixed) → layer-aware confirmed (2026-07-04)
- `[measured]` **green-ctx race 원인 규명**: 장문+batch>1서 device-side assert = decode를 green-ctx 스트림서 실행하는데 embed/clone/sampling은 default 스트림 → cross-stream 미동기화 race(CUDA_LAUNCH_BLOCKING서 사라짐=race 확증). **수정**: Zamba2Model.forward에 `_gstream.wait_stream(cur)` / `cur.wait_stream(_gstream)` 핸드셰이크. 전 그리드 crash 없이 완료.
- `[measured]` **Zamba2 no-GQA 완전 데이터**: attn/층 full 0.159→1.044(ctx348→4092=~7×); attn SM-민감 16SM대비 full ctx348 2.5×→ctx4092 **4.9×**(1.04→5.13). **mamba SM-둔감/inverse**(ctx4092 full 0.453 vs 16SM 0.300). ⇒ **layer-aware 두 조건 충족**(attn 민감·희소·비쌈 + mamba 둔감·다수). 계산: layer_aware decode 25.6ms+92SM환원 vs agnostic@108 33.8ms+0환원 = **layer-aware 압승**.
- `[measured]` **모델크기 의존**: NemotronH(8B) mamba 민감(compute-bound) vs Zamba2(2.7B) mamba 둔감(memory-bound). layer-aware 유리=(no-GQA)×(장문)×(mamba 둔감한 소형/memory-bound).
- 보고서 §3.5 완성. 원자료 `p1_4_zb_ctxsens_831012.txt`.

## P1.4 Task2 — 서빙 goodput 실측 (fused vs agnostic, NemotronH-8B, job 831023)
- `[measured]` in=2000/out=96, SLO TTFT≤3s·TPOT≤60ms. **agnostic(pdmux) TPOT 반감**(25 vs 53ms)·p99 극안정(26 vs ~100). goodput: rate3 agnostic 2.57(60/60) vs fused 2.18(44/60); rate6 2.34 vs 1.43; rate10 agnostic TTFT 폭발(5071ms)→0.62 vs fused 1.43. ⇒ pdmux가 저중부하 우세, 고부하선 prefill SM 굶주림으로 TTFT 폭발=layer-aware가 값할 지점.
- `[derived]` NemotronH는 mamba 민감→환원 SM 없어 layer-aware≈agnostic. Zamba2(mamba 둔감)면 환원으로 고부하 TTFT 개선 예측.
- harness: `bench_client.py`(Poisson+streaming TTFT/TPOT+goodput@SLO), `p1_4_serving.sbatch`. 원자료 `p1_4_serving_831023.txt`. 보고서 §3.7.
- **미완(Task2 완전판)**: layer_aware 서빙 = decode layer-type window ∥ prefill 상보 파티션 교차 event_loop(대수술). fused/agnostic는 실측 완료, layer_aware는 decode-side 근거로 예측.

## P1.4 Task2b — layer_aware 서빙 실증 + event_loop 정리 (2026-07-04)
- `[measured]` **event_loop 정리**: NemotronHModel.forward decode 루프를 race-safe로 재작성 — per-layer target 스트림(fixed-N / layer_aware reserve) + 스트림 전환마다 `wait_stream`(레이어간 data dependency + default 스트림 대비 안전). pdmux 동시 prefill서 검증(job 831067: concurrent 장문 부하 NO_CRASH, "Paris").
- `[measured]` **3정책 서빙 실증(NemotronH, job 831070)**: 셋 다 정상. **고부하 rate10서 layer_aware 최고 goodput 1.07(15/60) vs agnostic 0.50(7/60) vs fused 0.29(4/60)**, TTFT_med 최저(1659 vs 2680 vs 2958). 기전=attn(둔감) SM 환원→prefill 병목 완화, 포화근처 TTFT가 prefill SM에 초민감. 대가: layer_aware TPOT↑(67 vs 49; per-switch wait_stream 오버헤드 포함). 저중부하 유사.
- `[derived]` run-to-run 분산 큼(fused rate10 1.43→0.29 요동) → 1-run 강신호이나 다중run 필요. NemotronH(mamba민감)서도 고부하 이득=예상밖 긍정; Zamba2(mamba둔감)면 더 클 것.
- harness: `p1_4_serving.sbatch`(3정책), `p1_4_la_check.sbatch`(pdmux+layer_aware 검증), `bench_client.py`. 원자료 `p1_4_serving_831070.txt`. 보고서 §3.7.
- 미완: 완전-충실(decode window ∥ prefill 상보파티션 교차) — 현 구현은 decode per-layer 환원(prefill은 pdmux 스트림, 부분 조율). Zamba2 pdmux(forward_split_prefill+layer_aware) 포트.

## P1.5 — Zamba2 pdmux 포팅 완료 (2026-07-04)
- `[measured]` **Zamba2 pdmux 부팅 OK**: `Zamba2ForCausalLM.forward_split_prefill` 추가(SPLIT_PREFILL 활성; `forward_batch.hidden_states`+`zamba_original`=clone(embed)을 split 윈도우 간 스레딩, forward_batch가 split 호출 간 지속됨을 이용) → plain pdmux "Paris" 정확·NO_CRASH(job 831123).
- `[measured]` **Zamba2 layer_aware OK**: `Zamba2Model.forward` decode 루프 race-safe per-layer 전환 — Zamba2HybridLayer 9개(no-GQA attn=SM민감)는 base 유지(예약), 순수-mamba ~45개(둔감)는 `PDMUX_LA_FLOOR_SM`(기본16)으로 강등(prefill에 환원). 동시 장문 부하서 "Paris"·NO_CRASH(job 831135). **환원 비율 45/54(83%) >> NemotronH 4/52(8%)** → 더 큰 goodput 이득 기대.
- `[fix]` `_fixed` 파싱에 `_sm_part.isdigit()` 가드(안 하면 `int("layer_aware")` crash). 스윕 모드 `<sm>@<ctx>`는 여전히 파싱됨.
- layer_aware 서빙 env: `PDMUX_FIXED_DECODE_SM_FILE=<"layer_aware" 파일>`+`PDMUX_LA_FLOOR_SM=16`(NemotronH와 달리 `PDMUX_LA_RESERVE` 불요=레이어타입으로 예약). 부팅 `--attention-backend triton`.
- harness: `p1_4_zb_pdmux_check.sbatch {plain|layer_aware}`, `p1_4_zb_serving.sbatch`(3정책, in3000/out96, rates 6/12/20).
- **진행중**: Zamba2 3정책 서빙(job 831139) + NemotronH 다중run(831140/831141) — 대기중.

## P1.5 방법론 버그 — 포트 충돌로 첫 다중-run 오염 (2026-07-04)
- `[measured]` **포트 충돌**: `p1_4_serving.sbatch` PORT=31099 하드코딩. 두 NemotronH run(831140/831141)이 같은 노드(gpu39)에 co-schedule → 둘 다 127.0.0.1:31099 bind 시도 → 두번째가 `[Errno 98] address already in use` → 벤치가 한 서버에 이중 부하 = 교차오염. run3 layer_aware rate10 `n_ok=11/60, tpot=0.0`(서버 죽음 아님=충돌 아티팩트).
- `[fix]` PORT를 job-id 파생으로: NemotronH `$((31200+JOBID%600))`, Zamba2 `$((32000+JOBID%500))`(disjoint 범위). --gres=gpu:1이라 GPU는 분리되나(노드당 8개) 포트는 노드 네트워크 공유 → 파생 필수.
- `[note]` **오직 solo run(831070)만 clean**. 831140/831141은 폐기. 831172/831173로 재측정(포트 파생). Zamba2 serving도 solo(831146/831166)라 clean.

## P1.5 서빙 청정 다중-run 결과 (2026-07-05) — 이전 결론 정정
- `[measured]` **NemotronH 3청정run**(831070 solo, 831172/831173 포트파생; bind-error 없음 확인). goodput@SLO 평균: rate3 fused2.48/agn2.72/la2.62, rate6 1.61/1.67/1.34, **rate10 0.36/1.16/1.03**. ⇒ **고부하서 pdmux(agn·la)≫fused ~3×**, **la≈agn(우위 없음)**. run별 la-vs-agn 승자 뒤바뀜(run1 la승·run5 agn승) = 무승부.
- `[measured]` **★정정**: 이전 831070 단독 "rate10 la 1.07 vs agn 0.50 vs fused 0.29 → la 압승" 해석은 **오염**. 831140/841 동시-run 포트충돌이 agn을 억눌렀던 것. 청정선 agn 강함(1.16). NemotronH서 layer_aware는 agnostic 대비 추가 이득 **없음**(둔감층 4/52). = §2/§3 decode-side 예측대로.
- `[measured]` **Zamba2 prefill-bound**(831166, in3600/out32, tpot_slo120ms, triton). TTFT_med: rate6 fused1953/agn5002/la3846, rate12 6134/6995/5792, rate18 7196/7092/6595. ⇒ **la가 agn보다 TTFT −7~23% 일관 낮음**(45 mamba SM 환원→prefill 가속) = **기전 확증**. 대조: NH(둔감 4/52)=la≈agn vs ZB(둔감 45/54)=la<agn ⇒ **"la 이득 ∝ 환원가능 레이어 비중"**.
- `[measured]` **단 goodput 미전환**: ZB tpot_med 170-205ms≫SLO(triton no-cudagraph=decode-bound) → 3정책 goodput≈0(fused r6 0.50만). 작은 모델은 fused(전 108SM prefill 이미 빠름)가 최선(TTFT 1953<pdmux). **전이조건=prefill-bound + cudagraph(빠른 decode)**. la TPOT>agn(182vs171): mamba 16SM 잔여민감+per-switch wait_stream 오버헤드(스텝당 45전환).
- `[measured]` decode-bound 참고(out96, 831146): 동일 정성(fused 최선, tpot 76-106ms). 원자료 `p1_4_zb_serving_831146.txt`/`_831166.txt`, NH `p1_4_serving_831172/173.txt`.

## P1.6 — agnostic_v2 정책 추가 (사용자 요청, 2026-07-05)
- **정의**: agnostic_v2 = 균일 예약을 **attention 레이어 기준**(싸고 SM-둔감)으로 = 전 decode 레이어를 낮은 floor `PDMUX_AGN2_SM`(기본16)에 고정. prefill 환원 최대화하나 **SM-민감 레이어 미보호**. 대조: agnostic(v1)=SM-많이쓰는 mamba 기준(높은 N, 보호적)·layer_aware=레이어별(민감 보호+둔감 환원).
- **구현**: `models/{nemotron_h,zamba2}.py` decode forward에 `_sm_part=="agnostic_v2"`→`_fixed=PDMUX_AGN2_SM`(fixed-N `_tgt` 경로 재사용). NemotronH는 non-numeric 제외목록에 "agnostic_v2" 추가, Zamba2는 `.isdigit()` 가드가 이미 제외. `[measured]` 부팅 검증 OK: 양 모델 "Paris"·NO_CRASH(job 831534 ZB / 831535 NH).
- **decode-side 예측(기존 스윕 데이터 재활용)**:
  - `[measured→derived]` **NemotronH**(mamba 민감): agnostic_v2=N16→decode step **53.3ms**(vs fused 22.5·v1@64 22.9·layer_aware ~23.1). prefill 환원 **92 SM**(최대, 균일) but mamba 굶주림. **단 53.3<60 TPOT SLO** → 고부하 prefill-bound서 최저 TTFT로 **이길 가능성 있음**(불확실→실측필요).
  - `[derived]` **Zamba2 장문**(mamba 둔감·attn 민감): agnostic_v2=N16→attn 굶주림(5.13ms/층×9)+mamba(0.30×45)= **~59.7ms** decode(vs layer_aware 22.9·fused 29.8). = **layer_aware 대비 2.6× 느림** — 민감 attn 미보호가 치명적. layer_aware에 확실히 열위.
- **핵심 통찰**: agnostic_v2는 "환원 최대화하나 무차별" → **SM-민감 레이어가 있으면 그걸 굶긴다**. layer_aware의 가치=단순 "더 환원"이 아니라 "**선택적** 환원"임을 입증하는 대조군. agnostic_v2가 유리한 유일 체제=**모든 레이어 SM-둔감**(희귀). 긴장구도: v2=최대 prefill환원(최저 TTFT)·최악 decode(최악 TPOT) → TTFT vs TPOT 중 뭐가 binding이냐로 승부.
- 4정책 서빙 진행중: NemotronH job 831540(port 31740), Zamba2 831541(port 32041).

## P1.6 서빙 실측 — agnostic_v2 4정책 (2026-07-05)
- `[measured]` **NemotronH 4정책 3청정run**(831540/831554/831555, 포트파생 clean). goodput@SLO 평균: rate3 fused2.63/agn2.59/**agn2 2.92**/la2.45, rate6 1.05/1.54/1.86/**la 2.10**, rate10 fused1.21/agn0.84/**agn2 1.29**/la0.84. ⇒ **agnostic_v2가 rate3·rate10 평균 최고, rate6 2위 — 어느 부하서도 최악 아님**. 기전: mamba@16 decode 53ms<60 TPOT SLO → 페널티 흡수+최대 prefill환원(92)로 TTFT 방어. 사용자 직관("attn기준=최대환원") 실증. rate6 순서(la>agn2>agn>fused) 가장 안정, rate10 분산 큼(agn2 0.36~1.80).
- `[measured]` **Zamba2 4정책**(831541, prefill-bound). rate6 TTFT_med: fused4782/agn3928/**agn2 4427**/la1619(goodput 1.09 유일). ⇒ **agnostic_v2 열위** — no-GQA attn@16 굶주림이 decode 느리게(≈60ms)→prefill 밀림→TTFT 악화(4427>agn v1 3928, "더 환원했는데 TTFT 나빠짐"). layer_aware가 attn 보호+mamba 환원으로 최저 TTFT.
- `[derived]` **핵심**: agnostic_v2="attn 기준 균일"=최대 prefill환원(최저 TTFT 잠재)·최악 decode(최악 TPOT). **민감층이 floor(16 SM)를 견디면(SLO 여유 흡수) v2 승(NemotronH), 못 견디면 decode·TTFT로 번져 패(Zamba2 no-GQA attn)**. layer_aware는 그 판단을 층별로 → 양 모델 안전(never worst). **⇒ layer_aware 가치=단순 "더 환원"이 아니라 "선택적 환원".**
- 구현: `models/{nemotron_h,zamba2}.py` mode "agnostic_v2"→전층 `PDMUX_AGN2_SM`(16). 4정책 sbatch `p1_4_serving.sbatch`/`p1_4_zb_serving.sbatch`. 아티팩트(claude.ai artifact) §05에 시각화.

## P1.6b — 측정 신뢰성 문제 진단 (사용자 지적, 2026-07-05)
- `[measured]` **고레이트 goodput/TTFT 신뢰불가 근본원인 3가지**: ① **서버 포화점 ~4.3 req/s**(out_tok/s가 rate6·rate10서 동일 ~410 tok/s ÷96out) → rate6/rate10은 둘 다 포화 넘은 **과부하**(부하-지연 곡선 아님, burst-drain 노이즈). 증거: fused TTFT 비단조(run831540 r6=2378 > r10=1900ms=불가능). ② **hand-rolled `bench_client.py`가 약한 부하생성기**: 60 파이썬 스레드+blocking urllib 스트리밍 → GIL 경합이 TTFT/TPOT 측정에 클라이언트측 지연 주입. ③ **goodput@SLO=절벽 지표**(TTFT≤3s 이진): 포화근처 TTFT~2-3.4s라 미세노이즈가 다수 요청 flip → goodput 폭요동. 증거: agnostic_v2 r10 goodput 3run서 **1.71/0.36/1.80**(한 run서 최악), within-run 승자도 run별 모순(540→agn2, 554→la, 555→fused). ⇒ **rate10 정책순서 무의미. "agnostic_v2 고부하 우위"는 과대해석**(사용자 지적 타당). TTFT 공식 자체는 정상(t0=send, ttft=first_token−t0).
- `[fix]` **`sglang.bench_serving`(공식 async 부하생성기)로 재측정**: sub-saturation rate 1-4(+6 과부하), num-prompts 120, `--random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0`, `--output-details`로 per-request dump→goodput 계산. 정책당 별도 GPU job(OOM carryover 회피). 부팅 `datasets` 패키지 필요(설치, numpy 핀). harness `p1_6_bench_one.sbatch <policy>`.

## P1.6c — async 청정 재측정 결과 (2026-07-05) — agnostic_v2 결론 역전
- harness: `sglang.bench_serving`(async, `--dataset-name random-ids` offline, `--random-input-len 2000 --random-output-len 96 --random-range-ratio 1.0`, num-prompts 120, sub-saturation rate 1-4+6), 정책당 별도 GPU job(831609 fused/831610 agn/831611 agn2/831612 la). goodput은 per-request dump(`--output-details`)로 계산.
- `[measured]` **NemotronH goodput@SLO(TTFT≤3s,TPOT≤60ms)**: rate1 0.97/0.96/0.93/0.96, rate2 1.89/1.88/**0.32**/1.88, rate3 2.22/**2.76**/**0.00**/**2.76**, rate4 1.40/**2.25**/**0.00**/1.58 (fused/agn/agn2/la).
- `[measured]` **Median TPOT(ms)=판별자**: rate2 32/22/**72**/26, rate3 46/24/**119**/35, rate4 63/25/**191**/50. agnostic_v2 TPOT가 rate2부터 60 SLO 돌파(decode 16SM 굶주림). agn v1 평탄 22-25(decode SM 충분). §2 decode-side(16SM=53ms)를 서빙이 확증.
- `[measured]` **Median TTFT(ms)**: rate4 684/1699/**6705**/1529. agnostic_v2 TTFT도 최악=굶긴 decode가 밀려 prefill 스케줄 막음(nominal 환원 SM이 실질 도움 안 됨).
- `[derived]` **청정 순위: agnostic(v1)≈layer_aware>fused>agnostic_v2**. **agnostic_v2=최악**(두 모델 공히; 지배적 decode층이 SM-민감이라 굶기면 파멸). layer_aware≈agnostic(NH선 안전환원 대상 4/52뿐, 고부하 전환오버헤드로 소폭 열위). fused 저부하 양호·부하시 decode-prefill 결합으로 열위.
- **정정 요약**: 초기 "agnostic_v2 고부하 우위"(bench_client.py 60스레드 urllib×포화초과 rate6/10×절벽 goodput)는 오측 → async·sub-saturation로 역전. **layer_aware 가치=단순 환원 아니라 민감층 보호.** 원자료 `p1_6_bench_one_831609~831612.out`.
