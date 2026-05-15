"""
Two-pass SSD (Selective State Space Duality) algorithm.

Implements the mamba_chunk_scan_combined computation without cooperative grid
barriers (grid.sync()), enabling safe execution under CUDA Green Context SM
restrictions where the original Triton SSD kernel deadlocks.

Algorithm — three phases, two loops per chunk:

  Phase 1 (local scan): Sequential per-token loop (chunk_size iterations).
                        h_local[t] = A[t]*h_local[t-1] + x[t]⊗B[t], h_local[-1]=0.
                        Produces y_local and h_local_final.

  Phase 2 (prefix update): Single tensor op at chunk boundary.
                           prefix_h = A_cumulative * prefix_h + h_local_final

  Phase 3 (correction): Vectorized over all K tokens in the chunk (d_state
                        iterations, each processes K tokens at once — no token loop).
                        correction[t] = C[t]^T · (fp[t] · prefix_h)
                        where fp[t] = A[0]*...*A[t] (cumulative within chunk).

Phase 1 requires a sequential token loop (data dependency). Phase 3 has no
token-level data dependency and is fully vectorized over the chunk: d_state
Python iterations vs chunk_size token iterations, each processing all K tokens
simultaneously. This cuts Phase 3 Python overhead by chunk_size/d_state (4× for
Zamba2's chunk_size=256, d_state=64).

Memory cost: O(n_heads × head_dim × d_state) for prefix_h, O(K × n_heads ×
head_dim) temporary per chunk for Phase 3 (approximately batch×K×n_heads×head_dim
× 4 bytes ≈ 230 MB for Zamba2 batch=32).

This file provides:
  ssd_chunk_scan_twopass()              — core algorithm, takes pre-computed A_bar
  mamba_chunk_scan_combined_two_pass()  — same interface as mamba_chunk_scan_combined
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 1 compiled scan
# ---------------------------------------------------------------------------

_phase1_scan_compiled: Optional[object] = None


def _phase1_scan_impl(
    A_c: torch.Tensor,   # (batch, K, n_heads)
    x_c: torch.Tensor,   # (batch, K, n_heads, head_dim) float32
    B_c: torch.Tensor,   # (batch, K, n_heads, d_state)  float32
    C_c: torch.Tensor,   # (batch, K, n_heads, d_state)  float32
    D_f32: Optional[torch.Tensor],  # (n_heads,) or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential per-token scan for one chunk.  Returns (y_local, h_final)."""
    batch, K, n_heads, head_dim = x_c.shape
    d_state = B_c.shape[-1]
    h = torch.zeros(batch, n_heads, head_dim, d_state, device=x_c.device, dtype=torch.float32)
    y = torch.empty(batch, K, n_heads, head_dim, device=x_c.device, dtype=torch.float32)
    for t in range(K):
        A_t = A_c[:, t, :]              # (batch, n_heads)
        x_t = x_c[:, t, :]             # (batch, n_heads, head_dim)
        B_t = B_c[:, t, :]             # (batch, n_heads, d_state)
        C_t = C_c[:, t, :]             # (batch, n_heads, d_state)
        h = A_t[:, :, None, None] * h + x_t[:, :, :, None] * B_t[:, :, None, :]
        y_t = (h * C_t[:, :, None, :]).sum(-1)
        if D_f32 is not None:
            y_t = y_t + D_f32[None, :, None] * x_t
        y[:, t] = y_t
    return y, h


def _get_phase1_scan():
    """Return compiled phase-1 scan (lazy; falls back to eager if compile fails)."""
    global _phase1_scan_compiled
    if _phase1_scan_compiled is None:
        try:
            _phase1_scan_compiled = torch.compile(
                _phase1_scan_impl, dynamic=True, fullgraph=False
            )
        except Exception:
            _phase1_scan_compiled = _phase1_scan_impl
    return _phase1_scan_compiled


def ssd_chunk_scan_twopass(
    x: torch.Tensor,                            # (batch, seq_len, n_heads, head_dim)
    A_bar: torch.Tensor,                        # (batch, seq_len, n_heads)  float32
    B_exp: torch.Tensor,                        # (batch, seq_len, n_heads, d_state)
    C_exp: torch.Tensor,                        # (batch, seq_len, n_heads, d_state)
    chunk_size: int,
    D: Optional[torch.Tensor] = None,           # (n_heads,)
    initial_states: Optional[torch.Tensor] = None,  # (batch, n_heads, head_dim, d_state)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three-phase two-pass SSD chunk scan.

    No inter-block barriers; safe under Green Context SM restrictions.

    Phase 1 (sequential, chunk_size iterations per chunk) and Phase 3
    (vectorized, d_state iterations per chunk) are separated so Phase 3
    does not contribute to the hot token loop.

    Returns:
        y:            (batch, seq_len, n_heads, head_dim)  — same dtype as x
        final_states: (batch, n_heads, head_dim, d_state)  — float32
    """
    batch, seq_len, n_heads, head_dim = x.shape
    d_state = B_exp.shape[-1]
    dev = x.device

    n_chunks = math.ceil(seq_len / chunk_size)

    if initial_states is not None:
        prefix_h = initial_states.float()
    else:
        prefix_h = torch.zeros(batch, n_heads, head_dim, d_state, device=dev, dtype=torch.float32)

    y_final = torch.zeros(batch, seq_len, n_heads, head_dim, device=dev, dtype=x.dtype)
    D_f32 = D.float() if D is not None else None

    for c in range(n_chunks):
        cs = c * chunk_size
        ce = min(cs + chunk_size, seq_len)
        K = ce - cs

        A_c = A_bar[:, cs:ce, :]      # (batch, K, n_heads)
        x_c = x[:, cs:ce].float()    # (batch, K, n_heads, head_dim)
        B_c = B_exp[:, cs:ce].float() # (batch, K, n_heads, d_state)
        C_c = C_exp[:, cs:ce].float() # (batch, K, n_heads, d_state)

        # Cumulative A product for the chunk — used by Phase 3 and Phase 2.
        # fp[t] = A[0]*...*A[t] within the chunk (not global).
        fp = A_c.cumprod(dim=1)       # (batch, K, n_heads)
        fp_full = fp[:, -1, :]        # (batch, n_heads) — full-chunk decay

        # ── Phase 1: sequential local scan (chunk_size iterations) ──────────
        # h_local starts at 0; Phase 3 will correct for the actual prefix.
        # torch.compile eliminates Python dispatch overhead for the token loop.
        y_local, h_local = _get_phase1_scan()(A_c, x_c, B_c, C_c, D_f32)

        # ── Phase 3: vectorized prefix correction (d_state iterations) ──────
        # correction[t] = sum_d C[t,d] * fp[t] * prefix_h[:,:,:,d]
        # Vectorized over all K tokens at once; d_state << chunk_size for Zamba2.
        y_correction = torch.zeros(batch, K, n_heads, head_dim, device=dev, dtype=torch.float32)

        for ds in range(d_state):
            # C_fp_d[t] = C[t, ds] * fp[t]    (batch, K, n_heads)
            C_fp_d = C_c[:, :, :, ds] * fp
            # prefix_h_d: (batch, n_heads, head_dim)
            prefix_h_d = prefix_h[:, :, :, ds]
            # broadcast: (batch, K, n_heads, 1) * (batch, 1, n_heads, head_dim)
            y_correction += C_fp_d[:, :, :, None] * prefix_h_d[:, None, :, :]

        y_final[:, cs:ce] = (y_local + y_correction).to(x.dtype)

        # ── Phase 2: update running prefix for the next chunk ────────────────
        # Actual final state = fp_full * prefix_h + h_local_final
        prefix_h = fp_full[:, :, None, None] * prefix_h + h_local

    return y_final, prefix_h


def mamba_chunk_scan_combined_two_pass(
    x: torch.Tensor,                            # (batch, seq_len, n_heads, head_dim)
    dt: torch.Tensor,                           # (batch, seq_len, n_heads)
    A: torch.Tensor,                            # (n_heads,) — negative values
    B: torch.Tensor,                            # (batch, seq_len, n_groups, d_state)
    C: torch.Tensor,                            # (batch, seq_len, n_groups, d_state)
    chunk_size: int,
    D: Optional[torch.Tensor] = None,           # (n_heads,)
    z: Optional[torch.Tensor] = None,           # unused (gate applied in caller)
    dt_bias: Optional[torch.Tensor] = None,     # (n_heads,)
    dt_softplus: bool = True,
    seq_idx: Optional[torch.Tensor] = None,     # unused
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
) -> torch.Tensor:
    """Same interface as mamba_ssm mamba_chunk_scan_combined.

    Computes the same SSD result but decomposes the cooperative kernel into
    three sequential phases with no inter-block grid barriers.  Safe to run
    under CUDA Green Context SM restrictions (sm_count < n_blocks).

    A is expected to be negative (A_bar = exp(dt_eff * A) is the decay factor).
    """
    n_groups = B.shape[2]
    heads_per_group = x.shape[2] // n_groups

    # Pre-compute A_bar = exp(softplus(dt + dt_bias) * A)
    dt_eff = dt.float()
    if dt_bias is not None:
        dt_eff = dt_eff + dt_bias.float()[None, None, :]
    if dt_softplus:
        dt_eff = F.softplus(dt_eff)
    A_bar = torch.exp(dt_eff * A.float()[None, None, :])  # (batch, seq_len, n_heads)

    # Expand B and C from n_groups to n_heads
    if heads_per_group > 1:
        B_exp = B.repeat_interleave(heads_per_group, dim=2)  # (batch, seq_len, n_heads, d_state)
        C_exp = C.repeat_interleave(heads_per_group, dim=2)
    else:
        B_exp = B
        C_exp = C

    y, final_states = ssd_chunk_scan_twopass(
        x=x,
        A_bar=A_bar,
        B_exp=B_exp,
        C_exp=C_exp,
        chunk_size=chunk_size,
        D=D,
        initial_states=initial_states,
    )

    if return_final_states:
        return y, final_states
    return y
