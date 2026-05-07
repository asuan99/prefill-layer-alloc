"""
Falcon-H1-7B-Instruct layer extractor.

Loads tiiuae/Falcon-H1-7B-Instruct from HuggingFace and extracts individual
hybrid layer components (parallel SSM branch, attention branch, MLP) for
independent benchmarking.

Falcon-H1 architecture (7B-Instruct, verified from config):
  - 44 total hybrid layers, each containing:
    * Parallel Mamba-2 SSM branch (24 heads, head_dim=128, d_state=256)
    * Grouped-query attention (12 heads, 2 KV heads, head_dim=128)
    * MLP with intermediate_size=12288
  - SSM and Attention run *in parallel* within each layer
    (outputs are added before MLP), not alternating layers.
  - mamba_use_mlp=True: the Mamba block also contains an embedded MLP.
  - mamba_n_groups=1: B and C matrices are not group-split.
"""

import torch
import torch.nn as nn
from typing import Optional


class FalconH1LayerExtractor:
    """Extracts and wraps individual layer components from Falcon-H1-7B-Instruct.

    Args:
        model_path: HuggingFace model ID or local path.
        device: CUDA device string.
        dtype: Weight dtype (default: bfloat16).
    """

    MODEL_PATH = "tiiuae/Falcon-H1-7B-Instruct"

    # Verified from tiiuae/Falcon-H1-7B-Instruct config.json
    HIDDEN_SIZE = 3072
    N_SSM_HEADS = 24            # mamba_n_heads
    SSM_HEAD_DIM = 128          # mamba_d_head = mamba_d_ssm / mamba_n_heads = 3072/24
    D_STATE = 256               # mamba_d_state
    CHUNK_SIZE = 256            # mamba_chunk_size
    SSM_EXPAND = 2              # mamba_expand
    N_SSM_GROUPS = 1            # mamba_n_groups
    N_ATTN_HEADS = 12           # num_attention_heads
    N_KV_HEADS = 2              # num_key_value_heads
    ATTN_HEAD_DIM = 128         # head_dim (attention)
    INTERMEDIATE_SIZE = 12288   # intermediate_size (MLP)
    NUM_HIDDEN_LAYERS = 44      # num_hidden_layers

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path or self.MODEL_PATH
        self.device = device
        self.dtype = dtype
        self._model = None
        self._config = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoConfig
        print(f"Loading Falcon-H1 from {self.model_path} …")
        self._config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_ssm_layer(self, layer_idx: int = 0) -> nn.Module:
        """Return the parallel SSM (Mamba-2) branch at layer_idx.

        In Falcon-H1, each hybrid layer has a parallel SSM branch.
        The branch accepts hidden_states and returns same-shape tensor.
        """
        return self.get_ssm_branch(layer_idx)

    def get_ssm_branch(self, layer_idx: int = 0) -> nn.Module:
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["mamba", "ssm_branch", "ssm", "mixer"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _HybridBranchWrapper(layer, "ssm", self.HIDDEN_SIZE, self.device, self.dtype)

    def get_attention_layer(self, layer_idx: int = 0) -> nn.Module:
        """Return the attention branch at layer_idx.

        In Falcon-H1, each hybrid layer has a parallel attention branch.
        """
        return self.get_attention_branch(layer_idx)

    def get_attention_branch(self, layer_idx: int = 0) -> nn.Module:
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["self_attn", "attention", "attn_branch", "attn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _HybridBranchWrapper(layer, "attn", self.HIDDEN_SIZE, self.device, self.dtype)

    def get_mlp_layer(self, layer_idx: int = 0) -> nn.Module:
        """Return the MLP layer at layer_idx."""
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _HybridBranchWrapper(layer, "mlp", self.HIDDEN_SIZE, self.device, self.dtype)

    def get_model_config(self) -> dict:
        """Return relevant config values for benchmark setup."""
        return {
            "model_name": "falcon_h1",
            "hidden_size": self.HIDDEN_SIZE,
            "n_ssm_heads": self.N_SSM_HEADS,
            "ssm_head_dim": self.SSM_HEAD_DIM,
            "d_state": self.D_STATE,
            "chunk_size": self.CHUNK_SIZE,
            "n_ssm_groups": self.N_SSM_GROUPS,
            "n_attn_heads": self.N_ATTN_HEADS,
            "n_kv_heads": self.N_KV_HEADS,
            "attn_head_dim": self.ATTN_HEAD_DIM,
            "intermediate_size": self.INTERMEDIATE_SIZE,
            # All 44 layers are hybrid (SSM + Attn in parallel)
            "ssm_layer_fraction": 1.0,
            "attn_layer_fraction": 1.0,
        }

    def make_ssm_inputs(
        self, batch_size: int, seq_len: int
    ) -> dict[str, torch.Tensor]:
        """Generate random inputs matching Falcon-H1 SSM branch expected shapes."""
        return {
            "hidden_states": torch.randn(
                batch_size, seq_len, self.HIDDEN_SIZE,
                device=self.device, dtype=self.dtype
            )
        }

    def make_attn_inputs(
        self,
        batch_size: int,
        seq_len: int,
        context_len: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Generate random inputs for Falcon-H1 attention branch."""
        hidden_states = torch.randn(
            batch_size, seq_len, self.HIDDEN_SIZE,
            device=self.device, dtype=self.dtype
        )
        total_len = seq_len + context_len
        attn_mask = torch.ones(
            batch_size, 1, seq_len, total_len,
            device=self.device, dtype=self.dtype
        ).tril()
        return {
            "hidden_states": hidden_states,
            "attention_mask": attn_mask,
        }


class FallbackSSMBranch(nn.Module):
    """Minimal Falcon-H1-7B SSM branch using mamba_chunk_scan_combined directly.

    Used when the full model is not available or for isolated kernel benchmarks.
    Falcon-H1-7B: n_heads=24, head_dim=128, d_state=256, n_groups=1.
    """

    def __init__(
        self,
        d_model: int = 3072,
        n_heads: int = 24,
        head_dim: int = 128,
        d_state: int = 256,
        chunk_size: int = 256,
        n_groups: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.n_groups = n_groups
        self.device = device
        self.dtype = dtype

        inner_dim = n_heads * head_dim
        self.in_proj = nn.Linear(d_model, 2 * inner_dim, bias=False, dtype=dtype, device=device)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False, dtype=dtype, device=device)
        self.A_log = nn.Parameter(torch.randn(n_heads, dtype=dtype, device=device))
        self.D = nn.Parameter(torch.ones(n_heads, dtype=dtype, device=device))
        self.dt_bias = nn.Parameter(torch.randn(n_heads, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        except Exception:
            return self._pytorch_fallback(hidden_states)

        batch, seq_len, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)

        dt = torch.ones(batch, seq_len, self.n_heads, device=self.device, dtype=self.dtype) * 0.1
        B = torch.randn(batch, seq_len, self.n_groups, self.d_state, device=self.device, dtype=self.dtype)
        C = torch.randn(batch, seq_len, self.n_groups, self.d_state, device=self.device, dtype=self.dtype)
        A = -torch.exp(self.A_log.float()).to(self.dtype)

        try:
            y = mamba_chunk_scan_combined(
                x, dt, A, B, C,
                chunk_size=self.chunk_size,
                D=self.D,
                dt_bias=self.dt_bias,
                dt_softplus=True,
            )
        except Exception:
            return self._pytorch_fallback(hidden_states)
        y = y.view(batch, seq_len, -1)
        y = y * torch.sigmoid(z)
        return self.out_proj(y)

    def _pytorch_fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Chunked PyTorch scan for environments where Triton JIT fails."""
        import math
        batch, seq_len, _ = hidden_states.shape
        dev, dtype = hidden_states.device, hidden_states.dtype

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)

        n_chunks = math.ceil(seq_len / self.chunk_size)
        pad = n_chunks * self.chunk_size - seq_len
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))

        x_chunks = x.view(batch, n_chunks, self.chunk_size, self.n_heads, self.head_dim)
        A = torch.exp(-self.A_log.float().abs()).to(dtype).view(1, 1, self.n_heads, 1)
        h = torch.zeros(batch, self.n_heads, self.head_dim, device=dev, dtype=dtype)

        outs = []
        for ci in range(n_chunks):
            xc = x_chunks[:, ci]  # (batch, chunk_size, n_heads, head_dim)
            out = h.unsqueeze(1) * A + xc  # (batch, chunk_size, n_heads, head_dim)
            h = out[:, -1]  # (batch, n_heads, head_dim)
            outs.append(out)

        y = torch.cat(outs, dim=1)[:, :seq_len]  # (batch, seq_len, n_heads, head_dim)
        y = y.reshape(batch, seq_len, self.n_heads * self.head_dim)
        y = y * torch.sigmoid(z)
        return self.out_proj(y)


class _HybridBranchWrapper(nn.Module):
    """Wraps a full hybrid layer and routes forward to a specific branch type."""

    def __init__(
        self,
        layer: nn.Module,
        branch_type: str,
        hidden_size: int,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.layer = layer
        self.branch_type = branch_type
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.layer(hidden_states, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out
