# engine-port — Resume / Handoff (updated 2026-07-04)

Strategy **B** confirmed (see [reports/p0_triage.md](reports/p0_triage.md)). Through P1.5: both hybrids (NemotronH, Zamba2) run under pdmux with a working layer_aware policy; serving goodput measured.

## State
| Item | Status |
|---|---|
| P0 triage | ✅ committed (`5d02a75`) — recommend B |
| P1.0 pdmux base on A100/CUDA13 | ✅ boots + serves (green-ctx 108 SM), job 826832 |
| P1.pre editable dev tree + env | ✅ set up |
| P1.2 Zamba2 port (correctness) | ✅ "Paris" correct via `--attention-backend triton` (flashinfer NaN on head_dim=160). Report `reports/zamba2_troubleshooting.md` |
| P1.3 NemotronH pdmux (hybrid first) | ✅ `forward_split_prefill` added → pdmux runs on hybrids |
| P1.4 layer-aware impl + per-layer SM sensitivity + serving | ✅ Korean report `reports/p1_4_layer_aware_평가_kr.md`. NemotronH 3-policy (job 831070): layer_aware wins goodput at high load |
| P1.5 Zamba2 pdmux port | ✅ `forward_split_prefill` + layer_aware (reserve 9 hybrid / release 45 mamba). plain+layer_aware boot OK (jobs 831123/831135). Serving: **running (job 831139)** |
| Multi-run NemotronH serving (variance) | ⏳ running (jobs 831140/831141) |

## Environment (outside the repo; not in git)
- **Isolated venv**: `/scratch/$USER/whlee/sglang_engine_venv` (`python -m venv --system-site-packages` on the py3.14 conda; reuses cluster torch 2.9.1+cu130 / flashinfer 0.6.10 / mamba_ssm 2.3.1 / cuda-python 13.2).
- **Editable sglang dev tree**: `/scratch/$USER/whlee/sglang_engine_dev/python` — **sglang v0.5.10** (git-archive of `external/sglang-latest@v0.5.10`), `pip install -e`.
- **Kernel**: `sglang_kernel 0.4.1+cu130` (from github.com/sgl-project/whl; PyPI `sgl-kernel 0.3.21` is cu12 → unusable on CUDA 13). v0.5.10 ↔ kernel 0.4.1 is the API-aligned pair on torch 2.9.1.
- **Base version rationale**: v0.5.10 = last sglang on torch 2.9.1 (matches cluster); HEAD needs torch 2.11. Has pdmux + Mamba2/NemotronH + green-ctx.

## Applied patches (re-apply if venv/dev tree rebuilt)
1. **py3.14 shims** → `venv/.../site-packages/sitecustomize.py` (vendored copy: [env/sitecustomize.py](env/sitecustomize.py)). Restores `ast.Num`=`ast.Constant` (Triton 3.5.1) + no-ops `torch.compile`. **Perf-neutral & faithful** to py≤3.13 (sglang defaults enable_torch_compile=False).
2. **pdmux bug fix** → `sglang_engine_dev/python/sglang/srt/model_executor/model_runner.py` `maybe_update_ngram_token_table`: `getattr(forward_batch,"ngram_embedding_info",None)` (pdmux passes ModelWorkerBatch, not ForwardBatch). Candidate upstream PR.
3. **ninja on PATH**: `export PATH="$HOME/.local/bin:$PATH"` (flashinfer/triton JIT).
4. Extra pure-python deps installed `--no-deps` w/ constraints freezing the CUDA-13 stack (see [triage/notes.md](triage/notes.md) P1.0 for the list; xgrammar, fastapi, uvicorn, pydantic 2.13.4/core 2.46.4, …).
5. **`datasets 5.0.0`** (+ pyarrow/dill/multiprocess) for `sglang.bench_serving` (proper async load generator). Installed pinning `numpy==2.3.5` so the CUDA stack is untouched (torch 2.9.1 unchanged). Needed because my hand-rolled `bench_client.py` (60-thread urllib) is a GIL-bottlenecked load generator — see §measurement below.

## Reproduce a pdmux boot (validated)
```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/$USER/whlee/sglang_engine_venv/bin/activate
# on an A100 node (SLURM: partition amd_a100nv_8, --gres=gpu:1):
sbatch workspace/engine-port/triage/p1_0_pdmux_boot.sbatch   # -> PDMUX_BOOT_OK
```
pdmux flags: `--enable-pdmux --pdmux-config-path <yml> --chunked-prefill-size -1 --disable-overlap-schedule`.

## Zamba2 — STATUS: correct + pdmux + layer_aware all working
Model+config written & wired (see [env/dev_tree_edits.md](env/dev_tree_edits.md); tracked copies in `src/{models,configs}/zamba2.py`).
- ✅ correctness: `--attention-backend triton --dtype bfloat16` → "Paris" (flashinfer NaNs on head_dim=160; root cause in `reports/zamba2_troubleshooting.md`).
- ✅ pdmux: `forward_split_prefill` on `Zamba2ForCausalLM` enables SPLIT_PREFILL. plain boot OK (job 831123).
- ✅ layer_aware: per-layer green-ctx switch — reserve 9 hybrid (no-GQA attn, SM-sensitive), release 45 mamba (insensitive) to `PDMUX_LA_FLOOR_SM`. Boot OK, NO_CRASH (job 831135).

## Serving results (2026-07-05) — clean, port-collision corrected
Report §3.7/§3.8/§0 updated. Key clean findings:
- **NemotronH** (3 clean runs 831070/831172/831173): high load rate10 **agnostic≈layer_aware ≫ fused** (goodput mean 1.16·1.03 vs 0.36 ≈3×). **layer_aware has NO edge over agnostic** (only 4/52 insensitive layers). The earlier "la 1.07 vs agn 0.50 win" was a **port-collision artifact** (831140/831141 both bound port 31099 → cross-contaminated; discarded).
- **Zamba2** (831166, prefill-bound in3600/out32): **layer_aware beats agnostic on TTFT by −7~23%** (releasing 45/54 insensitive mamba layers speeds prefill) → **mechanism confirmed** (la-edge ∝ released-fraction: NH 4/52=no edge vs ZB 45/54=edge). BUT triton-no-cudagraph decode floors TPOT at 170-205ms ≫ SLO (decode-bound), so the TTFT edge doesn't convert to goodput; small model → fused wins outright.
- **Methodology**: serving sbatch ports must be job-id-derived (fixed).

## P1.7 (2026-07-05) — generalized across 4 hybrids (see reports/p1_7_hybrid_generalization_kr.md)
Optimal SM-reservation policy = f(which layer type is SM-sensitive). Measured taxonomy:
NemotronH-8B (mamba sensitive → agnostic; agn_v2 worst) · Zamba2 1.2/2.7/7B (no-GQA attn
sensitive at long ctx → layer-aware) · Granite-4-h-micro (nothing sensitive → agnostic_v2
optimal) · Falcon-H1-3B (spatial, one layer type → layer-aware N/A, agnostic ≫ fused).
**mamba sensitivity = SSD config (compute vs memory-bound), NOT size** — NemotronH is the
lone compute-bound outlier; corrects the earlier "size-dependent" claim. Instrumented
falcon_h1.py (forward_split_prefill) + granitemoehybrid.py (green-ctx+timing); dev_tree §8/9.

## Resume point / next
- `[open]` **Granite agnostic_v2 serving** (predicted optimal — everything releasable); Falcon-H1 intra-layer attn/mamba split timing; Qwen3-Next (gated-deltanet).
- `[open]` **Zamba2 goodput transfer**: needs a cudagraph-capable env (py3.12 rebuild, or a sglang version supporting pdmux+mamba cudagraph) so decode is fast enough that TTFT (not TPOT) binds → then Zamba2 layer_aware goodput win should materialize.
- `[open]` **Fully-coordinated layer-aware serving** (decode layer-window ∥ prefill complementary partition) — event_loop surgery.
- Boot tests: `sbatch triage/p1_4_zb_pdmux_check.sbatch {plain|layer_aware}` (Zamba2 pdmux), `p1_4_la_check.sbatch` (NemotronH). Serving: `p1_4_serving.sbatch` (NH), `p1_4_zb_serving.sbatch` (ZB).
