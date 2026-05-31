"""
chunked_ssm_runner.py — **PRIMARY SSM measurement path** (single source of truth).

이 모듈이 SSM SM-scaling 측정의 유일한 실측 기준이다.  다른 경로(analytical
wave model, PyTorch fallback scan, two-pass)는 cross-validation 보조 또는
deprecated 상태이며, 여기서 얻은 ``ssm_chunked_*.csv``가 plot/decision-matrix의
primary 입력이 된다.

동기:
  mamba_chunk_scan_combined (Triton SSD fused kernel)은 청크 간 hidden state
  전파를 위해 grid.sync() cooperative barrier를 사용한다.  Green Context로 SM을
  극소수로 제한하면 n_blocks >> active_SM이 되어 barrier에서 deadlock → CUDA
  illegal memory access → context 오염 → 연쇄 실패가 발생한다.

  시퀀스를 kernel 호출 경계에서 분할하면 cooperative 제약을 벗어난다.
  청크 간 state 전달은 grid.sync()가 아닌 host-side kernel launch 경계에서
  이루어지기 때문이다.

  이 방식은 프로덕션 Triton kernel (mamba_chunk_scan_combined)을 그대로 사용하고
  Falcon-H1 CP/Marconi와 정합하므로 다른 우회책보다 우선된다.
"""

import sys
import os
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "../.."))  # workspace/
sys.path.insert(0, os.path.join(_here, ".."))     # characterization/

import math
import statistics
from pathlib import Path
from typing import Optional

import torch

from shared.loaders import get_model_config
from src.smctrl.green_ctx_controller import SMController
from src.profiling.metrics import BandwidthEstimator


# ---------------------------------------------------------------------------
# Model config loader
# ---------------------------------------------------------------------------

def _load_model_config(model_name: str) -> dict:
    """Load SSM params via shared.loaders.get_model_config.

    Returns flat dict:
      d_model, n_heads, head_dim, d_state, chunk_size, n_groups
    """
    raw = get_model_config(model_name)
    ssm = raw.get("ssm", {})
    return {
        "d_model":    raw["hidden_size"],
        "n_heads":    ssm["n_heads"],
        "head_dim":   ssm["head_dim"],
        "d_state":    ssm["d_state"],
        "chunk_size": ssm["chunk_size"],
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
    model_cfg   = _load_model_config(model_name)
    n_heads     = model_cfg["n_heads"]
    head_dim    = model_cfg["head_dim"]
    d_state     = model_cfg["d_state"]
    n_groups    = model_cfg["n_groups"]
    ssd_chunk   = model_cfg["chunk_size"]
    chunk_tokens = prefill_chunk_tokens

    nchunks_per_call  = max(1, chunk_tokens // ssd_chunk)
    n_blocks_per_call = batch_size * nchunks_per_call * n_heads
    coop_safe = _cooperative_safe(batch_size, chunk_tokens, n_heads, ssd_chunk, sm_count)
    n_calls   = math.ceil(seq_len / chunk_tokens)

    # Pre-allocate scan-shaped tensors once (reused across all chunk iterations).
    # Values don't affect latency or kernel grid — only shapes matter.
    x_chunk   = torch.randn(batch_size, chunk_tokens, n_heads, head_dim,
                             dtype=torch.bfloat16, device=device)
    dt_chunk  = torch.full((batch_size, chunk_tokens, n_heads), 0.1,
                            dtype=torch.bfloat16, device=device)
    A         = -torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    B_chunk   = torch.randn(batch_size, chunk_tokens, n_groups, d_state,
                             dtype=torch.bfloat16, device=device)
    C_chunk   = torch.randn(batch_size, chunk_tokens, n_groups, d_state,
                             dtype=torch.bfloat16, device=device)
    D         = torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    dt_bias   = torch.zeros(n_heads, dtype=torch.bfloat16, device=device)

    # HBM bytes for one mamba_chunk_scan_combined call (scan kernel only)
    read_bytes_per_call, write_bytes_per_call = BandwidthEstimator.ssm_scan_bytes(
        batch=batch_size,
        seq_len=chunk_tokens,
        n_heads=n_heads,
        head_dim=head_dim,
        d_state=d_state,
        n_groups=n_groups,
    )

    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    supports_state = _check_initial_states_support()
    state_kwarg    = _get_initial_states_kwarg() if supports_state else None
    _warned_no_state = [False]

    smctrl.set_sm_count(sm_count)
    stream = smctrl.get_stream()

    def _run_once() -> None:
        ssm_state = torch.zeros(
            batch_size, n_heads, head_dim, d_state,
            dtype=torch.float32, device=device,
        )
        with torch.cuda.stream(stream):
            for _ in range(n_calls):
                if supports_state:
                    result = mamba_chunk_scan_combined(
                        x_chunk, dt_chunk, A, B_chunk, C_chunk,
                        chunk_size=ssd_chunk,
                        D=D,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                        **{state_kwarg: ssm_state, "return_final_states": True},
                    )
                    if isinstance(result, tuple):
                        _, ssm_state = result
                else:
                    if not _warned_no_state[0]:
                        print(
                            "[chunked_ssm_runner] WARNING: mamba_chunk_scan_combined does not "
                            "support initial_states/return_final_states.  State is not passed "
                            "across chunks.  Latency measurement is still valid."
                        )
                        _warned_no_state[0] = True
                    mamba_chunk_scan_combined(
                        x_chunk, dt_chunk, A, B_chunk, C_chunk,
                        chunk_size=ssd_chunk,
                        D=D,
                        dt_bias=dt_bias,
                        dt_softplus=True,
                    )
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        _run_once()

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
        "latency_ms":              statistics.median(latencies),
        "latency_std_ms":          statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_list_ms":         latencies,
        "n_kernel_calls":          n_calls,
        "n_blocks_per_call":       n_blocks_per_call,
        "cooperative_safe":        coop_safe,
        "sm_count":                sm_count,
        "prefill_chunk_tokens":    chunk_tokens,
        "seq_len":                 seq_len,
        "batch_size":              batch_size,
        "read_bytes_per_call":     read_bytes_per_call,
        "write_bytes_per_call":    write_bytes_per_call,
    }
