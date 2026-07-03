# engine-port — Resume / Handoff (paused 2026-07-03)

Strategy **B** confirmed (see [reports/p0_triage.md](reports/p0_triage.md)). P1.0 done, P1.2 (Zamba2 port) is the resume point.

## State at pause
| Item | Status |
|---|---|
| P0 triage | ✅ committed (`5d02a75`) — recommend B |
| P1.0 pdmux base on A100/CUDA13 | ✅ boots + serves (green-ctx 108 SM), job 826832 |
| P1.pre editable dev tree + env | ✅ set up |
| P1.2 Zamba2 port | 📝 designed ([reports/zamba2_port_plan.md](reports/zamba2_port_plan.md)); **code not yet written** |

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

## Reproduce a pdmux boot (validated)
```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/$USER/whlee/sglang_engine_venv/bin/activate
# on an A100 node (SLURM: partition amd_a100nv_8, --gres=gpu:1):
sbatch workspace/engine-port/triage/p1_0_pdmux_boot.sbatch   # -> PDMUX_BOOT_OK
```
pdmux flags: `--enable-pdmux --pdmux-config-path <yml> --chunked-prefill-size -1 --disable-overlap-schedule`.

## Zamba2 port — STATUS: runs, all weights load; forward-parity bug remains
Model+config written & wired (see [env/dev_tree_edits.md](env/dev_tree_edits.md); tracked copies in `src/{models,configs}/zamba2.py`). Validated:
- ✅ dummy boot OK (job 826938): serves through 54 layers (shared-attn+LoRA+Mamba2 SSD).
- ✅ real weights: **all 531 ckpt weights load** (loaded=527, skipped=0, missing=0); server serves.
- ❌ output = all token 0 (empty). No NaN; scale matches vLLM. ⇒ **semantic forward bug**, not weights.

**Resume point — logit-parity debug (P1.2c):** build an activation-parity harness — same prompt through the sglang model vs vLLM (`vllm_venv`)/HF, compare embed → each layer out → logits to localize first divergence. Candidates (notes.md P1.2c): mamba `A` (A_log vs -exp) interpretation, gated mamba `norm`, shared-attn block_idx→layer_id KV routing, residual/concat in flat-token layout. Boot with `--dtype bfloat16`.
Then: P1.3 layer-aware pdmux (build/validate on NemotronH first — already runs), P1.4 eval.

Boot test: `sbatch workspace/engine-port/triage/p1_2_zamba2_real.sbatch` (real weights) or `p1_2_zamba2_boot.sbatch` (dummy).
