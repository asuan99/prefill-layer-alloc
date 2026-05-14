"""
chunked_ssm_runner.py

기존 run_ssm_prefill_sweep.py의 Green Context 직접 측정 대체.
시퀀스를 prefill_chunk_tokens 크기로 나눠 mamba_chunk_scan_combined를 청크 단위로
호출해 cooperative inter-block barrier 우회.

동기:
  mamba_chunk_scan_combined (Triton SSD fused kernel)은 청크 간 hidden state
  전파를 위해 grid.sync() cooperative barrier를 사용한다.  Green Context로 SM을
  극소수로 제한하면 n_blocks >> active_SM이 되어 barrier에서 deadlock → CUDA
  illegal memory access → context 오염 → 연쇄 실패가 발생한다.

  시퀀스를 kernel 호출 경계에서 분할하면 cooperative 제약을 벗어난다.
  청크 간 state 전달은 grid.sync()가 아닌 host-side kernel launch 경계에서
  이루어지기 때문이다.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import statistics
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.smctrl.green_ctx_controller import SMController


# ---------------------------------------------------------------------------
# Model config loader
# ---------------------------------------------------------------------------

def _load_model_config(model_name: str) -> dict:
    """Load SSM params from configs/models.yaml.

    Returns flat dict:
      d_model, n_heads, head_dim, d_state, chunk_size, n_groups
    """
    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    if model_name not in raw:
        raise ValueError(
            f"Unknown model {model_name!r}.  "
            f"Available: {list(raw.keys())}"
        )

    m = raw[model_name]
    ssm = m.get("ssm", {})
    return {
        "d_model":    m["hidden_size"],
        "n_heads":    ssm["n_heads"],
        "head_dim":   ssm["head_dim"],
        "d_state":    ssm["d_state"],
        "chunk_size": ssm["chunk_size"],   # SSD internal chunk_size (fixed 256)
        "n_groups":   ssm.get("n_groups", ssm.get("expand", 2)),
    }


# ---------------------------------------------------------------------------
# mamba_chunk_scan_combined wrapper with initial_states support
# ---------------------------------------------------------------------------

_MAMBA_SUPPORTS_INITIAL_STATES: Optional[bool] = None  # lazy check


def _check_initial_states_support() -> bool:
    global _MAMBA_SUPPORTS_INITIAL_STATES
    if _MAMBA_SUPPORTS_INITIAL_STATES is not None:
        return _MAMBA_SUPPORTS_INITIAL_STATES
    try:
        import inspect
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        sig = inspect.signature(mamba_chunk_scan_combined)
        params = set(sig.parameters.keys())
        _MAMBA_SUPPORTS_INITIAL_STATES = bool(
            params & {"initial_states", "ssm_initial_states"}
        )
    except Exception:
        _MAMBA_SUPPORTS_INITIAL_STATES = False
    return _MAMBA_SUPPORTS_INITIAL_STATES


def _get_initial_states_kwarg() -> str:
    """Return the correct kwarg name for mamba_chunk_scan_combined initial states."""
    try:
        import inspect
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        params = set(inspect.signature(mamba_chunk_scan_combined).parameters.keys())
        if "initial_states" in params:
            return "initial_states"
        if "ssm_initial_states" in params:
            return "ssm_initial_states"
    except Exception:
        pass
    return "initial_states"  # default guess


# ---------------------------------------------------------------------------
# Per-chunk SSM kernel call
# ---------------------------------------------------------------------------

def _make_ssm_call_fn(model_cfg: dict, device: str):
    """Return a closure that runs one SSM kernel call with state passing.

    The returned function has signature:
        (chunk: Tensor, state: Tensor, in_proj_weight: Tensor, A_log, D, dt_bias)
            -> (y_chunk: Tensor, new_state: Tensor)

    Where:
        chunk:          (batch, chunk_tokens, d_model) bf16
        state:          (batch, n_heads, head_dim, d_state) float32
        in_proj_weight: (2*inner_dim, d_model) bf16

    If mamba_ssm does not support initial_states/return_final_states, the state
    is computed analytically (zeros propagation) and a warning is printed once.
    In that case cooperative_safe analysis is still valid — the kernel grid is
    the same; only the state value is wrong (but measurement is latency-only).
    """
    n_heads   = model_cfg["n_heads"]
    head_dim  = model_cfg["head_dim"]
    d_state   = model_cfg["d_state"]
    n_groups  = model_cfg["n_groups"]
    ssd_chunk = model_cfg["chunk_size"]
    inner_dim = n_heads * head_dim

    supports_state = _check_initial_states_support()
    state_kwarg    = _get_initial_states_kwarg() if supports_state else None

    _warned_no_state = [False]

    def call_fn(
        chunk: torch.Tensor,
        state: torch.Tensor,
        in_proj_weight: torch.Tensor,
        A_log: torch.Tensor,
        D_param: torch.Tensor,
        dt_bias: torch.Tensor,
    ):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        batch, chunk_tokens, _ = chunk.shape
        dtype  = chunk.dtype
        dev    = chunk.device

        # in_proj: (d_model → 2*inner_dim)
        xz = torch.nn.functional.linear(chunk, in_proj_weight)
        x  = xz[:, :, :inner_dim]           # (batch, chunk_tokens, inner_dim)
        # z is not used in the timed kernel (no gating in pure SSM pass)

        x  = x.view(batch, chunk_tokens, n_heads, head_dim)

        # Random dt, B, C — values don't affect latency or grid shape
        dt = torch.ones(batch, chunk_tokens, n_heads, device=dev, dtype=dtype) * 0.1
        B  = torch.randn(batch, chunk_tokens, n_groups, d_state, device=dev, dtype=dtype)
        C  = torch.randn(batch, chunk_tokens, n_groups, d_state, device=dev, dtype=dtype)
        A  = -torch.exp(A_log.float()).to(dtype)

        if supports_state:
            # state shape expected by mamba_chunk_scan_combined: (batch, n_heads, head_dim, d_state)
            kwargs = {
                state_kwarg:        state,
                "return_final_states": True,
            }
            result = mamba_chunk_scan_combined(
                x, dt, A, B, C,
                chunk_size=ssd_chunk,
                D=D_param,
                dt_bias=dt_bias,
                dt_softplus=True,
                **kwargs,
            )
            # result is (y, new_state) when return_final_states=True
            if isinstance(result, tuple):
                y, new_state = result
            else:
                # Fallback: some versions return only y even with return_final_states
                y = result
                new_state = state
        else:
            if not _warned_no_state[0]:
                print(
                    "[chunked_ssm_runner] WARNING: mamba_chunk_scan_combined does not "
                    "support initial_states/return_final_states.  State is not passed "
                    "across chunks.  Latency measurement is still valid."
                )
                _warned_no_state[0] = True
            y = mamba_chunk_scan_combined(
                x, dt, A, B, C,
                chunk_size=ssd_chunk,
                D=D_param,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            new_state = state  # unchanged (no state passing)

        y = y.view(batch, chunk_tokens, inner_dim)
        return y, new_state

    return call_fn


# ---------------------------------------------------------------------------
# cooperative_safe check
# ---------------------------------------------------------------------------

def _cooperative_safe(
    batch: int,
    prefill_chunk_tokens: int,
    n_heads: int,
    ssd_chunk: int,
    sm_count: int,
) -> bool:
    """Check whether each kernel call's block count fits within the active SMs.

    Triton SSD kernel grid (dominant kernel): (nchunks, batch, n_heads).
    A100: heavy register usage → conservatively assume max_blocks_per_sm = 1.
    """
    nchunks_per_call  = max(1, prefill_chunk_tokens // ssd_chunk)
    n_blocks_per_call = batch * nchunks_per_call * n_heads
    return n_blocks_per_call <= sm_count


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_chunked_ssm_sweep(
    model_name: str,
    seq_len: int,
    batch_size: int,
    prefill_chunk_tokens: int,
    sm_count: int,
    smctrl: SMController,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: str = "cuda",
) -> dict:
    """Run chunked SSM prefill and measure latency under Green Context.

    시퀀스를 prefill_chunk_tokens 크기의 청크로 나눠 mamba_chunk_scan_combined를
    반복 호출하고, 각 kernel call 사이에 SSM hidden state를 명시적으로 전달한다.

    Args:
        model_name:           모델 이름 (configs/models.yaml 키)
        seq_len:              전체 시퀀스 길이 (처리 대상)
        batch_size:           배치 크기
        prefill_chunk_tokens: 청크 크기 (kernel 호출당 토큰 수).
                              prefill_chunk_tokens // ssd_chunk_size = nchunks_per_call.
                              너무 크면 n_blocks_per_call이 sm_count를 초과해
                              cooperative deadlock 위험.
                              권장: sm_count × ssd_chunk_size 이하.
        sm_count:             Green Context에 적용할 SM 수
        smctrl:               초기화된 SMController 인스턴스
        n_warmup:             워밍업 반복 횟수
        n_measure:            측정 반복 횟수
        device:               CUDA device string

    Returns:
        {
            "latency_ms":           float,   # median latency (전체 시퀀스)
            "latency_std_ms":       float,
            "latency_list_ms":      list[float],
            "n_kernel_calls":       int,     # seq_len // prefill_chunk_tokens (ceil)
            "n_blocks_per_call":    int,     # batch × (pct//ssd_chunk) × n_heads
            "cooperative_safe":     bool,
            "sm_count":             int,
            "prefill_chunk_tokens": int,
            "seq_len":              int,
            "batch_size":           int,
        }
    """
    model_cfg = _load_model_config(model_name)
    d_model   = model_cfg["d_model"]
    n_heads   = model_cfg["n_heads"]
    head_dim  = model_cfg["head_dim"]
    d_state   = model_cfg["d_state"]
    ssd_chunk = model_cfg["chunk_size"]
    inner_dim = n_heads * head_dim

    # cooperative 안전성 체크
    nchunks_per_call  = max(1, prefill_chunk_tokens // ssd_chunk)
    n_blocks_per_call = batch_size * nchunks_per_call * n_heads
    coop_safe = _cooperative_safe(
        batch_size, prefill_chunk_tokens, n_heads, ssd_chunk, sm_count
    )

    import math
    n_calls = math.ceil(seq_len / prefill_chunk_tokens)

    # 전체 시퀀스 입력 (고정)
    x_full = torch.randn(
        batch_size, seq_len, d_model,
        dtype=torch.bfloat16, device=device,
    )

    # SSM 가중치 (측정 목적이므로 random 초기화; kernel shape에만 영향)
    in_proj_weight = torch.randn(
        2 * inner_dim, d_model, dtype=torch.bfloat16, device=device
    )
    A_log   = torch.randn(n_heads, dtype=torch.bfloat16, device=device)
    D_param = torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(n_heads, dtype=torch.bfloat16, device=device)

    call_fn = _make_ssm_call_fn(model_cfg, device)

    smctrl.set_sm_count(sm_count)
    stream = smctrl.get_stream()

    def _run_once() -> None:
        ssm_state = torch.zeros(
            batch_size, n_heads, head_dim, d_state,
            dtype=torch.float32, device=device,
        )
        with torch.cuda.stream(stream):
            for start in range(0, seq_len, prefill_chunk_tokens):
                end   = min(start + prefill_chunk_tokens, seq_len)
                chunk = x_full[:, start:end, :]
                _, ssm_state = call_fn(
                    chunk, ssm_state, in_proj_weight, A_log, D_param, dt_bias
                )
        torch.cuda.synchronize()

    # 워밍업
    for _ in range(n_warmup):
        _run_once()

    # 측정 (CUDA event 기반)
    latencies: list[float] = []
    for _ in range(n_measure):
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record(stream)
        _run_once()
        t_end.record(stream)
        torch.cuda.synchronize()
        latencies.append(t_start.elapsed_time(t_end))

    smctrl.reset()

    return {
        "latency_ms":           statistics.median(latencies),
        "latency_std_ms":       statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_list_ms":      latencies,
        "n_kernel_calls":       n_calls,
        "n_blocks_per_call":    n_blocks_per_call,
        "cooperative_safe":     coop_safe,
        "sm_count":             sm_count,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "seq_len":              seq_len,
        "batch_size":           batch_size,
    }
