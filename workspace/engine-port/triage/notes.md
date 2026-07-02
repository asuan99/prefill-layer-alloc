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
