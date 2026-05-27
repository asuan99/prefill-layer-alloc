# Task: Fix SSM Measurement Target — Scan Kernel Only

## Background

The current Stage 1 SM scaling sweep measures the **entire SSM layer**
(in_proj GEMM + mamba_chunk_scan_combined + out_proj GEMM) via
`LayerRunner.run_ssm_layer()`. This causes a fundamental measurement error:

- `in_proj` + `out_proj` GEMMs dominate latency (~65% of total)
- GEMMs are compute-bound (AI ≈ 1927 FLOPs/Byte >> ridge 156)
- `mamba_chunk_scan_combined` is only ~0.4% of total latency
- `mamba_chunk_scan_combined` is memory-bound (AI ≈ 31 FLOPs/Byte < ridge 156)

Result: the sweep reports compute-bound behavior with no SM saturation,
contradicting the literature. The correct measurement target is the
**scan kernel only**, which is memory-bound and should exhibit SM saturation
when HBM bandwidth is saturated.

The correct approach already exists in `stage1_sm_scaling/_ncu_target.py`
(the `ssm` branch), which calls `mamba_chunk_scan_combined` directly with
pre-allocated input tensors, bypassing in_proj/out_proj.

---

## Files to Modify

### 1. `src/profiling/metrics.py` — `BandwidthEstimator.ssm_bytes()`

Replace the current formula (which only counts in_proj/out_proj weight +
activation) with the actual HBM bytes accessed by `mamba_chunk_scan_combined`.

**Current signature:**
```python
@staticmethod
def ssm_bytes(batch, seq_len, hidden_size, n_heads, head_dim, d_state,
              weight_bytes, bytes_per_elem=2) -> tuple[int, int]:
```

**New signature and implementation:**
```python
@staticmethod
def ssm_scan_bytes(batch, seq_len, n_heads, head_dim, d_state, n_groups,
                   bytes_per_elem=2) -> tuple[int, int]:
    """HBM bytes for mamba_chunk_scan_combined scan kernel only.

    Inputs read from HBM:
      x:      (batch, seq_len, n_heads, head_dim)
      dt:     (batch, seq_len, n_heads)
      B:      (batch, seq_len, n_groups, d_state)
      C:      (batch, seq_len, n_groups, d_state)
      A, D, dt_bias: (n_heads,) each — negligible but included

    Output written to HBM:
      y:      (batch, seq_len, n_heads, head_dim)
    """
    bpe = bytes_per_elem
    x_bytes      = batch * seq_len * n_heads * head_dim * bpe
    dt_bytes     = batch * seq_len * n_heads * bpe
    B_bytes      = batch * seq_len * n_groups * d_state * bpe
    C_bytes      = batch * seq_len * n_groups * d_state * bpe
    param_bytes  = n_heads * 3 * bpe   # A, D, dt_bias
    y_bytes      = batch * seq_len * n_heads * head_dim * bpe

    read_bytes  = x_bytes + dt_bytes + B_bytes + C_bytes + param_bytes
    write_bytes = y_bytes
    return read_bytes, write_bytes
```

Keep `ssm_bytes()` as-is for backward compatibility (used by MLP/Attn paths).

---

### 2. `src/models/layer_runner.py` — `run_ssm_layer()`

Replace the current implementation that calls `layer(hidden_states)` (full
layer forward) with a direct call to `mamba_chunk_scan_combined` using
pre-allocated scan-shaped input tensors.

**Key changes:**

- Cache key changes from `(model, batch, seq, use_fallback, force_pytorch_scan)`
  to `(model, batch, seq)` — no fallback needed since we call the kernel directly
- Input tensors: `x (B,L,H,D)`, `dt (B,L,H)`, `B (B,L,G,N)`, `C (B,L,G,N)`
- Use `BandwidthEstimator.ssm_scan_bytes()` instead of `ssm_bytes()`
- Load model config from `configs/models.yaml` via `_get_extractor(model_name).get_model_config()`

**New `_run` closure inside `run_ssm_layer()`:**
```python
def _run():
    with torch.no_grad():
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        mamba_chunk_scan_combined(
            x_cache, dt_cache, A_cache, B_cache, C_cache,
            chunk_size=chunk_size,
            D=D_cache,
            dt_bias=dt_bias_cache,
            dt_softplus=True,
        )
```

Where all `*_cache` tensors are pre-allocated once per `(model, batch, seq_len)`
key and stored in `self._ssm_cache`.

**Tensor shapes (Zamba2 example: n_heads=112, head_dim=64, d_state=64, n_groups=2):**
```
x:      (batch, seq_len, n_heads, head_dim)   bf16
dt:     (batch, seq_len, n_heads)              bf16, filled with 0.1
A:      (n_heads,)                             bf16, filled with -1.0
B:      (batch, seq_len, n_groups, d_state)    bf16, randn
C:      (batch, seq_len, n_groups, d_state)    bf16, randn
D:      (n_heads,)                             bf16, ones
dt_bias:(n_heads,)                             bf16, zeros
```

Remove the `use_fallback_kernel` and `force_pytorch_scan` parameters from
`run_ssm_layer()` — they are no longer needed since we call the kernel directly.
Add a `use_pytorch_fallback: bool = False` parameter for the `--force-pytorch-scan`
path (keep backward compat with `run_ssm_prefill_sweep.py`).

For the pytorch fallback path (`use_pytorch_fallback=True`), keep calling
`FallbackSSMKernel._pytorch_fallback()` with a dummy hidden_states input,
but note in a docstring that this measures a different kernel.

---

### 3. `stage1_sm_scaling/chunked_ssm_runner.py` — `run_chunked_ssm_sweep()`

Same fix: remove `in_proj_weight`, `A_log`, `D_param`, `dt_bias` as learned
parameters. Pre-allocate scan-shaped tensors directly.

**Current (wrong):**
```python
in_proj_weight = torch.randn(2 * inner_dim, d_model, ...)
xz = torch.nn.functional.linear(chunk, in_proj_weight)
x, z = xz.chunk(2, dim=-1)
x = x.view(batch, chunk_tokens, n_heads, head_dim)
_, ssm_state = call_fn(chunk, ssm_state, in_proj_weight, A_log, D_param, dt_bias)
```

**New (correct):**
```python
# Pre-allocate scan-shaped tensors once
x_chunk   = torch.randn(batch, chunk_tokens, n_heads, head_dim, dtype=torch.bfloat16, device=device)
dt_chunk  = torch.ones(batch, chunk_tokens, n_heads, dtype=torch.bfloat16, device=device) * 0.1
A         = -torch.ones(n_heads, dtype=torch.bfloat16, device=device)
B_chunk   = torch.randn(batch, chunk_tokens, n_groups, d_state, dtype=torch.bfloat16, device=device)
C_chunk   = torch.randn(batch, chunk_tokens, n_groups, d_state, dtype=torch.bfloat16, device=device)
D         = torch.ones(n_heads, dtype=torch.bfloat16, device=device)
dt_bias   = torch.zeros(n_heads, dtype=torch.bfloat16, device=device)

# In measurement loop:
y_chunk, new_state = mamba_chunk_scan_combined(
    x_chunk, dt_chunk, A, B_chunk, C_chunk,
    chunk_size=ssd_chunk,
    D=D, dt_bias=dt_bias, dt_softplus=True,
    initial_states=ssm_state, return_final_states=True,
)
```

Also update `_make_ssm_call_fn()` accordingly or remove it and inline the call.

Update the BW estimation call to use `ssm_scan_bytes()` with chunk-level sizes:
```python
read_bytes, write_bytes = BandwidthEstimator.ssm_scan_bytes(
    batch=batch_size,
    seq_len=prefill_chunk_tokens,   # per-call bytes
    n_heads=n_heads,
    head_dim=head_dim,
    d_state=d_state,
    n_groups=n_groups,
)
```

---

### 4. `stage1_sm_scaling/run_ssm_prefill_sweep.py`

No structural changes needed. The fix flows through `LayerRunner.run_ssm_layer()`.

However, update the docstring / comments to clarify that the sweep now measures
`mamba_chunk_scan_combined` only (not the full SSM layer), and update the
`_TOKENS_PER_BLOCK` comment to reflect the correct grid formula:

```python
# mamba_chunk_scan_combined grid:
#   nchunks = seq_len // chunk_size
#   n_blocks ≈ batch × nchunks × n_heads
#   = batch × (seq_len // 256) × n_heads
# For Zamba2 (n_heads=112): n_blocks = batch × seq_len × 112 / 256
#   = batch × seq_len × 0.4375
# For Falcon-H1 (n_heads=24): n_blocks = batch × seq_len × 24 / 256
#   = batch × seq_len × 0.09375
#
# The simplified formula batch × seq_len // 4 was incorrect.
# Use model-specific formula via configs/models.yaml n_heads and chunk_size.
```

Update `_n_blocks()` to be model-aware:
```python
def _n_blocks(batch: int, seq_len: int, n_heads: int, chunk_size: int = 256) -> int:
    nchunks = max(1, seq_len // chunk_size)
    return max(1, batch * nchunks * n_heads)
```

Pass `n_heads` and `chunk_size` from `model_cfg` wherever `_n_blocks()` is called.
Load model config at the top of `run_sweep()`:

```python
cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
with open(cfg_path) as f:
    model_cfg = yaml.safe_load(f)[model_name]
ssm_cfg = model_cfg.get("ssm", {})
n_heads_ssm   = ssm_cfg["n_heads"]
chunk_size_ssm = ssm_cfg["chunk_size"]
```

---

## What NOT to Change

- `_ncu_target.py` — already correct (calls `mamba_chunk_scan_combined` directly)
- `stage1_sm_scaling/run_attn_prefill_sweep.py` — correct as-is
- `stage1_sm_scaling/run_mlp_prefill_sweep.py` — correct as-is
- `src/models/zamba2.py`, `src/models/falcon_h1.py` — not used in scan-only path
- `stage2_overhead/`, `stage3_hm_eval/` — not affected by this change
- `src/smctrl/` — not affected

---

## Expected Outcome After Fix

With scan-kernel-only measurement:

| Metric | Before (full layer) | After (scan only) |
|--------|--------------------|--------------------|
| Arithmetic Intensity | ~1927 FLOPs/Byte | ~31 FLOPs/Byte |
| Bound regime | compute-bound | **memory-bound** ✓ |
| SM saturation | not observed | **observed** ✓ |
| BW utilization | 1~5% | ~10~30% range |
| Consistency with literature | ✗ | ✓ |

The SM scaling curve should now show a saturation point where adding more SMs
no longer improves latency because HBM bandwidth becomes the ceiling — which
is the expected behavior described in BulletServe and related work.

---

## Testing After Change

Run a quick smoke test to verify the new measurement makes sense:

```bash
python -c "
import torch; torch.cuda.init()
from src.models.layer_runner import LayerRunner
runner = LayerRunner(device='cuda')
result = runner.run_ssm_layer(
    model_name='zamba2',
    batch_size=4,
    seq_len=4096,
    sm_count=108,
    n_warmup=3,
    n_measure=10,
    skip_sm_control=True,
)
print(f'latency: {result[\"latency_ms\"]:.3f}ms')
print(f'BW util: {result[\"bw_utilization_pct\"]:.1f}%')
print(f'AI expected ~31 FLOPs/Byte (memory-bound)')
# Expected: latency << 12.6ms (full layer), BW util >> 5%
"
```

Verify SM scaling shows saturation:
```bash
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model zamba2 --device a100_80gb \
    --seq-lens 4096 --batch-sizes 4 \
    --n-warmup 5 --n-measure 20
# SM scaling curve should flatten at higher SM counts
```
