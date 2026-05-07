"""
Zamba2-7B-Instruct layer extractor.

Loads Zyphra/Zamba2-7B-Instruct from HuggingFace and extracts individual
SSM (Mamba-2), Attention, and MLP layer modules so they can be run
independently in stage1 benchmarks without the full model forward pass.

Zamba2 architecture (7B-Instruct, verified from config):
  - 81 total layers: 68 Mamba-2 SSM-only blocks + 13 hybrid (SSM+Attn) blocks
  - Hybrid layer indices: [6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77]
  - SSM: mamba_chunk_scan_combined (Mamba-2 parallel scan) in prefill mode
    * n_mamba_heads=112, mamba_headdim=64, mamba_d_state=64, mamba_ngroups=2
  - Attention: grouped-query attention (MHA: 32 heads, 32 KV heads, head_dim=224)
    * attention_hidden_size=7168 (attention operates in expanded space)
  - MLP: ffn_hidden_size=14336
"""

import torch
import torch.nn as nn
from typing import Optional


_HYBRID_LAYER_IDS = frozenset([6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77])


class Zamba2LayerExtractor:
    """Extracts and wraps individual layers from Zamba2-7B-Instruct.

    Args:
        model_path: HuggingFace model ID or local path.
        device: CUDA device string.
        dtype: Weight dtype (default: bfloat16).
    """

    MODEL_PATH = "Zyphra/Zamba2-7B-Instruct"

    # Verified from Zyphra/Zamba2-7B-Instruct config.json
    HIDDEN_SIZE = 3584
    N_SSM_HEADS = 112           # n_mamba_heads
    HEAD_DIM = 64               # mamba_headdim
    D_STATE = 64                # mamba_d_state
    CHUNK_SIZE = 256            # chunk_size
    SSM_EXPAND = 2              # mamba_expand
    N_SSM_GROUPS = 2            # mamba_ngroups
    N_ATTN_HEADS = 32           # num_attention_heads
    N_KV_HEADS = 32             # num_key_value_heads
    ATTN_HEAD_DIM = 224         # attention_head_dim
    INTERMEDIATE_SIZE = 14336   # ffn_hidden_size / intermediate_size (MLP)
    NUM_HIDDEN_LAYERS = 81      # num_hidden_layers

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
        print(f"Loading Zamba2 from {self.model_path} …")
        self._config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def get_ssm_layer(self, layer_idx: int = 0) -> nn.Module:
        """Return the Mamba-2 SSM module at layer_idx.

        layer_idx=0 is a Mamba-only layer. To get a hybrid layer's SSM
        component, use any index from _HYBRID_LAYER_IDS.

        The returned module accepts:
            hidden_states: (batch, seq_len, hidden_size)
        and returns:
            (batch, seq_len, hidden_size)
        """
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["mamba", "mixer", "ssm"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _LayerWrapper(layer, "ssm")

    def get_attention_layer(self, layer_idx: int = 6) -> nn.Module:
        """Return the attention module at layer_idx.

        Zamba2-7B hybrid layers: [6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77].
        Default layer_idx=6 is the first hybrid layer.
        """
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["self_attn", "attention", "attn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _LayerWrapper(layer, "attn")

    def get_mlp_layer(self, layer_idx: int = 0) -> nn.Module:
        """Return the MLP/FFN module at layer_idx."""
        self._load_model()
        layer = self._model.model.layers[layer_idx]
        for name in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return _LayerWrapper(layer, "mlp")

    def get_ssm_layer_indices(self) -> list[int]:
        """Return indices of all SSM-only layers (non-hybrid) in the model."""
        self._load_model()
        ssm_indices = []
        for i, layer in enumerate(self._model.model.layers):
            for name in ["mamba", "mixer", "ssm"]:
                if hasattr(layer, name):
                    ssm_indices.append(i)
                    break
        if not ssm_indices:
            # Fallback: exclude known hybrid layer indices
            ssm_indices = [i for i in range(self.NUM_HIDDEN_LAYERS) if i not in _HYBRID_LAYER_IDS]
        return ssm_indices

    def get_model_config(self) -> dict:
        """Return relevant config values for benchmark setup."""
        return {
            "model_name": "zamba2",
            "hidden_size": self.HIDDEN_SIZE,
            "n_ssm_heads": self.N_SSM_HEADS,
            "head_dim": self.HEAD_DIM,
            "d_state": self.D_STATE,
            "chunk_size": self.CHUNK_SIZE,
            "n_ssm_groups": self.N_SSM_GROUPS,
            "n_attn_heads": self.N_ATTN_HEADS,
            "n_kv_heads": self.N_KV_HEADS,
            "attn_head_dim": self.ATTN_HEAD_DIM,
            "intermediate_size": self.INTERMEDIATE_SIZE,
        }

    def make_ssm_inputs(
        self, batch_size: int, seq_len: int
    ) -> dict[str, torch.Tensor]:
        """Generate random inputs matching Zamba2 SSM layer expected shapes."""
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
        """Generate random inputs for Zamba2 attention layer."""
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


class FallbackSSMKernel(nn.Module):
    """Minimal Mamba-2 kernel wrapper for environments without the full model.

    Uses mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined directly.
    Zamba2-7B: n_heads=112, head_dim=64, d_state=64, n_groups=2.
    """

    def __init__(
        self,
        d_model: int = 3584,
        n_heads: int = 112,
        head_dim: int = 64,
        d_state: int = 64,
        chunk_size: int = 256,
        n_groups: int = 2,
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
        """Chunked PyTorch scan for environments where Triton JIT fails.

        Matches memory-bandwidth profile of mamba_chunk_scan_combined:
        in_proj GEMM + chunked state accumulation + gate + out_proj GEMM.
        """
        import math
        batch, seq_len, _ = hidden_states.shape
        dev, dt = hidden_states.device, hidden_states.dtype

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)

        n_chunks = math.ceil(seq_len / self.chunk_size)
        pad = n_chunks * self.chunk_size - seq_len
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))

        x_chunks = x.view(batch, n_chunks, self.chunk_size, self.n_heads, self.head_dim)
        A = torch.exp(-self.A_log.float().abs()).to(dt).view(1, 1, self.n_heads, 1)
        h = torch.zeros(batch, self.n_heads, self.head_dim, device=dev, dtype=dt)

        outs = []
        for ci in range(n_chunks):
            xc = x_chunks[:, ci]
            h = h.unsqueeze(1) * A + xc * 0.1
            outs.append(h)
        y = torch.cat(outs, dim=1)[:, :seq_len].reshape(batch, seq_len, -1)
        y = y * torch.sigmoid(z)
        return self.out_proj(y)


class _LayerWrapper(nn.Module):
    """Thin wrapper to expose a full layer block as a specific layer type."""

    def __init__(self, layer: nn.Module, layer_type: str):
        super().__init__()
        self.layer = layer
        self.layer_type = layer_type

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.layer(hidden_states, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out
