# P1.7 — Additional hybrid models: scan + experiment plan (scheduled 2026-07-05 06:00)

Extend the existing NemotronH/Zamba2 experiment suite to more hybrid SSM+Attention
models. Prep scan done 05:1x; execution scheduled for 06:00 (off-peak).

## Environment facts (measured during prep)
- **Login node HAS internet** (curl huggingface.co → 200) → can download new weights
  into hf_cache, then run compute jobs offline (HF_HUB_OFFLINE=1).
- **Cached weights** (`hf_cache/hub/`): NemotronH-8B, Zamba2-{1.2B,2.7B,7B}, **Falcon-H1-{3B-Base,7B-Instruct}**, Qwen2.5-0.5B.
- **sglang v0.5.10 already registers** these hybrids (no porting): FalconH1ForCausalLM,
  GraniteMoeHybridForCausalLM, JetNemotronForCausalLM, Qwen3NextForCausalLM,
  NemotronHForCausalLM, Zamba2ForCausalLM (mine).

## Candidate priority (feasibility × value)
1. **Zamba2-1.2B & Zamba2-7B** (cached; zamba2.py already instrumented w/ per-layer
   timing + green-ctx + 4 policies). LOWEST risk = just swap `--model-path` in existing
   sbatches. Value: extends the **model-size axis** of the real-engine sensitivity
   finding (2.7B mamba was SM-insensitive/memory-bound → is 1.2B also? is 7B sensitive
   like NemotronH-8B?). Boot needs `--attention-backend triton` (head_dim 160) + bf16.
2. **Falcon-H1-3B-Base** (cached; **SPATIAL hybrid** = mamba ∥ attn in every layer,
   summed; 32 layers, GQA 10:2, head_dim 128 → flashinfer OK). Value: **boundary test**
   — spatial hybrids have NO layer types, so **layer-aware ≡ agnostic** (nothing to
   reserve per-type). The interesting axis becomes intra-layer (mamba vs attn share the
   layer). Needs per-layer timing instrumentation added to falcon_h1.py (copy the
   NemotronHModel.forward pattern: CUDA-event timing + optional green-ctx pin).
3. **GraniteMoeHybrid (Granite 4.0 small)** — DOWNLOAD at 06:00 (find smallest h-hybrid,
   e.g. ibm-granite/granite-4.0-h-micro ~3B, or -h-tiny). Mamba2 SSD **temporal** hybrid
   **+ MoE** → closest NemotronH analogue with an MoE twist. STRETCH (download + boot).
4. (skip unless time) Qwen3-Next (gated-deltanet, smallest is 80B-A3B = too big),
   JetNemotron.

## Experiment protocol (identical to NemotronH/Zamba2 — §2/§3.5/§3.9)
For each model, in order; stop at first blocker and record it:
- **E0 boot+correctness**: launch_server, `"The capital of France is"` → coherent
  (Paris). Determine backend (flashinfer/triton) + flags (`--disable-piecewise-cuda-graph`
  on py3.14, `--dtype bfloat16`). Job pattern: p1_4_zb_pdmux_check.sbatch style.
- **E1 pdmux boot**: `--enable-pdmux`. Hybrids need `forward_split_prefill`; check if
  present (temporal ones do via my ports; Falcon-H1/Granite = verify, add if missing).
- **E2 per-layer(-type) SM sensitivity**: green-ctx pin decode to N∈{108,64,44,32,16},
  measure ms/layer(-type). Temporal → per type (mamba/attn/mlp). Spatial (Falcon-H1) →
  single layer type (measures if the fused mamba∥attn layer is SM-sensitive). Needs the
  timing instrumentation in the model's forward (present for NemotronH/Zamba2; ADD for
  Falcon-H1/Granite).
- **E3 context sweep**: ms/layer(-type) at ctx≈{350,2140,4092} (attn O(L)? mamba O(1)?).
- **E4 serving (async `sglang.bench_serving`, SUB-SATURATION)**: sweep request-rate below
  the model's ~throughput/ (in2000/out96, random-ids, num-prompts 120), 1 policy per GPU
  job (avoid OOM carryover), job-id-derived ports. Policies: fused, agnostic, agnostic_v2
  (, layer_aware where layer types exist). Report Median/P99 TTFT, Median TPOT (the
  discriminator), goodput@SLO from per-request dump. ⚠️ NEVER use the old bench_client.py
  (GIL-bound) and NEVER measure past saturation — that was the P1.6c measurement bug.
- **검수 (verify)**: outputs coherent, NO_CRASH, numbers internally consistent (TPOT vs
  goodput), compare against framework prediction (layer-aware benefit ∝ SM-insensitive
  time-dominant layer fraction; spatial → layer-aware≡agnostic).

## Deliverables
- Extend report `p1_4_layer_aware_평가_kr.md` with a §3.10 (or new report) covering the
  new models; update the visualization (add the spatial-vs-temporal contrast + size axis).
- Update notes.md / RESUME.md / memory. Commit (engine-port P1.7), do NOT push.

## Reusable harness
- Boot check: `p1_4_zb_pdmux_check.sbatch <mode>` (adapt --model-path/--attention-backend).
- Serving (async, clean): `p1_6_bench_one.sbatch <policy>` (adapt --model-path; ONE policy/GPU).
- Sensitivity sweep template: `p1_4_nh_smsens.sbatch` (green-ctx N-sweep via mode file).
- Env: module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0; venv
  sglang_engine_venv; PATH ~/.local/bin; HF_HOME=.../hf_cache; TRITON_CACHE_DIR set.
- SLURM: partition amd_a100nv_8, --gres=gpu:1.
