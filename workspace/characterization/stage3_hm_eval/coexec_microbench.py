"""
coexec_microbench.py — 진짜 동시 실행(concurrent execution) overlap 측정.

stage3_hm_eval/deprecated/run_concurrent_eval.py 의 순차 시뮬레이션을 대체한다.
단일 스트림 interleaving 이 아니라, 두 Green Context partition 에서 prefill kernel 과
decode kernel 을 실제로 동시에 enqueue 하여 spatial sharing 을 직접 측정한다.

측정 구조:
  solo_prefill  — prefill kernel 단독, 전체 SM
  solo_decode   — decode kernel 단독, 전체 SM
  concurrent_gc — Green Context 두 partition: prefill stream + decode stream 동시 enqueue
                  (spatial SM partition: 진짜 동시 실행)
  concurrent_ts — 일반 CUDA stream 두 개: 파티션 없이 동시 enqueue
                  (temporal sharing baseline; MPS 와 유사한 거동)

overlap_ratio = concurrent_time / (solo_prefill + solo_decode)
  → 0.5 에 가까울수록 완전 overlap (2× 이득)
  → 1.0 에 가까울수록 overlap 없음 (순차 실행과 동일)

Prefill kernel: chunked SSM (cooperative barrier deadlock 회피)
  - mamba_chunk_scan_combined 을 prefill_chunk_tokens 크기로 반복 호출
  - 각 kernel call 의 n_blocks ≤ sm_count (cooperative safe 보장)

Decode kernel: SSM decode (seq_len=1) + Attn KV read
  - SSM: mamba_chunk_scan_combined with L=1, chunk_size=1
  - Attn: F.scaled_dot_product_attention (Q: (B,1,H,D), KV: (B,context_len,H,D))

MPS 비교 주의:
  concurrent_ts ("two-stream") 는 MPS 와 유사하지만 진짜 MPS 가 아니다.
  진짜 MPS 비교는 `nvidia-cuda-mps-server` + 별도 프로세스가 필요하다.
  이 벤치마크에서 "two_stream" 은 Green Context 없이 두 스트림을 동시에 launch 한
  하드웨어 스케줄러의 거동을 관찰하는 것으로, spatial partition 이 없는 baseline 이다.

Hardware guard: A100 에서만 결론이 유효. 다른 GPU 에서는 PIPELINE VALIDATION ONLY.

Usage:
    python stage3_hm_eval/coexec_microbench.py --model zamba2
    python stage3_hm_eval/coexec_microbench.py --model zamba2 \\
        --sm-splits 0.3 0.4 0.5 0.6 0.7 \\
        --seq-lens 1024 2048 4096 --batch-sizes 1 4 --context-lens 1024 4096

Output:
    results/stage3/coexec_{model}_{tag}.csv   — overlap_ratio per config
    results/stage3/coexec_{model}_{tag}.png   — overlap_ratio vs sm_split figure
"""

from __future__ import annotations

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "../.."))   # workspace/
sys.path.insert(0, os.path.join(_here, ".."))      # characterization/

import argparse
import csv
import ctypes
import math
import statistics
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from shared.loaders import get_model_config, get_hardware_config, device_tag
from src.smctrl.green_ctx_controller import (
    SMController,
    _CUdevResource,
    _load_driver_lib,
    _CU_DEV_RESOURCE_TYPE_SM,
    _CU_GREEN_CTX_DEFAULT_STREAM,
    _CU_STREAM_NON_BLOCKING,
)


# ---------------------------------------------------------------------------
# Hardware guard
# ---------------------------------------------------------------------------

_A100_KW = frozenset({"a100", "a800"})


def _is_a100() -> bool:
    return torch.cuda.is_available() and any(
        kw in torch.cuda.get_device_name(0).lower() for kw in _A100_KW
    )


def _hw_guard() -> None:
    if not _is_a100():
        print("=" * 70)
        print("  WARNING: Non-A100 GPU detected.")
        print("  PIPELINE VALIDATION ONLY — numeric overlap conclusions not valid.")
        print("  Re-run on A100 for paper-quality results.")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Model config helper
# ---------------------------------------------------------------------------

def _load_ssm_cfg(model_name: str) -> dict:
    raw = get_model_config(model_name)
    ssm = raw.get("ssm", {})
    attn = raw.get("attention", {})
    return {
        "d_model":    raw["hidden_size"],
        "n_heads":    ssm["n_heads"],
        "head_dim":   ssm["head_dim"],
        "d_state":    ssm["d_state"],
        "chunk_size": ssm["chunk_size"],
        "n_groups":   ssm.get("n_groups", ssm.get("expand", 2)),
        # attention (for decode)
        "n_attn_heads":    attn.get("num_heads", 32),
        "n_attn_kv_heads": attn.get("num_kv_heads", attn.get("num_heads", 32)),
        "attn_head_dim":   attn.get("head_dim", 128),
    }


# ---------------------------------------------------------------------------
# Green Context two-partition creation
# ---------------------------------------------------------------------------

def _make_green_ctx_stream(
    lib: ctypes.CDLL,
    device_id: int,
    res: _CUdevResource,
) -> Optional[torch.cuda.ExternalStream]:
    """Create a Green Context + stream from an existing CUdevResource."""
    desc = ctypes.c_void_p()
    rc = lib.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(res), 1)
    if rc != 0:
        return None

    green_ctx = ctypes.c_void_p()
    rc = lib.cuGreenCtxCreate(
        ctypes.byref(green_ctx), desc, device_id, _CU_GREEN_CTX_DEFAULT_STREAM
    )
    if rc != 0:
        return None

    stream_ptr = ctypes.c_void_p()
    rc = lib.cuGreenCtxStreamCreate(
        ctypes.byref(stream_ptr), green_ctx, _CU_STREAM_NON_BLOCKING, 0
    )
    if rc != 0:
        lib.cuGreenCtxDestroy(green_ctx)
        return None

    return torch.cuda.ExternalStream(stream_ptr.value, device=device_id)


def create_two_partitions(
    n_prefill_sm: int,
    device_id: int = 0,
) -> tuple[Optional[torch.cuda.Stream], Optional[torch.cuda.Stream], dict]:
    """Create two non-overlapping Green Context partitions.

    Returns (stream_prefill, stream_decode, info_dict).
    On failure (Green Context not supported), returns (None, None, {"error": ...}).

    Partition layout:
      stream_prefill — gets n_prefill_sm SMs (rounded up to GPC granularity)
      stream_decode  — gets all remaining SMs after the prefill split

    This uses cuDevSmResourceSplitByCount with a non-null remaining pointer,
    ensuring the two partitions are spatially exclusive.
    """
    lib = _load_driver_lib()
    if lib is None:
        return None, None, {"error": "libcuda.so not available"}

    torch.cuda.init()

    base_res = _CUdevResource()
    rc = lib.cuDeviceGetDevResource(device_id, ctypes.byref(base_res), _CU_DEV_RESOURCE_TYPE_SM)
    if rc != 0:
        return None, None, {"error": f"cuDeviceGetDevResource rc={rc}"}

    # Split base_res: prefill partition gets n_prefill_sm SMs; decode gets the remainder.
    part1_res = _CUdevResource()
    remaining_res = _CUdevResource()
    nb_groups = ctypes.c_uint(1)
    rc = lib.cuDevSmResourceSplitByCount(
        ctypes.byref(part1_res),
        ctypes.byref(nb_groups),
        ctypes.byref(base_res),
        ctypes.byref(remaining_res),   # capture leftover SMs for decode
        0,                             # useFlags
        ctypes.c_uint(n_prefill_sm),   # minCount per group
    )
    if rc != 0 or nb_groups.value == 0:
        return None, None, {"error": f"cuDevSmResourceSplitByCount rc={rc}"}

    actual_prefill_sm = int(part1_res._impl.smCount)
    actual_decode_sm  = int(remaining_res._impl.smCount)

    stream_prefill = _make_green_ctx_stream(lib, device_id, part1_res)
    stream_decode  = _make_green_ctx_stream(lib, device_id, remaining_res)

    if stream_prefill is None or stream_decode is None:
        return None, None, {"error": "Green Context / stream creation failed"}

    return stream_prefill, stream_decode, {
        "actual_prefill_sm": actual_prefill_sm,
        "actual_decode_sm":  actual_decode_sm,
    }


# ---------------------------------------------------------------------------
# Kernel builders (return zero-arg callables that run on current stream)
# ---------------------------------------------------------------------------

def _build_prefill_fn(
    model_cfg: dict,
    seq_len: int,
    batch_size: int,
    prefill_chunk_tokens: int,
    device: str,
) -> tuple[callable, dict]:
    """Build chunked SSM prefill callable. Returns (fn, meta)."""
    n_heads    = model_cfg["n_heads"]
    head_dim   = model_cfg["head_dim"]
    d_state    = model_cfg["d_state"]
    n_groups   = model_cfg["n_groups"]
    ssd_chunk  = model_cfg["chunk_size"]
    chunk_tok  = prefill_chunk_tokens
    n_calls    = math.ceil(seq_len / chunk_tok)

    x_chunk  = torch.randn(batch_size, chunk_tok, n_heads, head_dim,
                           dtype=torch.bfloat16, device=device)
    dt_chunk = torch.full((batch_size, chunk_tok, n_heads), 0.1,
                          dtype=torch.bfloat16, device=device)
    A        = -torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    B_chunk  = torch.randn(batch_size, chunk_tok, n_groups, d_state,
                           dtype=torch.bfloat16, device=device)
    C_chunk  = torch.randn(batch_size, chunk_tok, n_groups, d_state,
                           dtype=torch.bfloat16, device=device)
    D        = torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    dt_bias  = torch.zeros(n_heads, dtype=torch.bfloat16, device=device)

    try:
        import inspect
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        params = set(inspect.signature(mamba_chunk_scan_combined).parameters)
        state_kwarg = (
            "initial_states" if "initial_states" in params
            else "ssm_initial_states" if "ssm_initial_states" in params
            else None
        )
    except ImportError:
        mamba_chunk_scan_combined = None
        state_kwarg = None

    def fn():
        if mamba_chunk_scan_combined is None:
            # Fallback: simulate with matmul if mamba_ssm not installed
            _ssm_matmul_fallback(x_chunk, n_calls)
            return

        ssm_state = torch.zeros(
            batch_size, n_heads, head_dim, d_state,
            dtype=torch.float32, device=device,
        )
        for _ in range(n_calls):
            if state_kwarg is not None:
                result = mamba_chunk_scan_combined(
                    x_chunk, dt_chunk, A, B_chunk, C_chunk,
                    chunk_size=ssd_chunk, D=D, dt_bias=dt_bias, dt_softplus=True,
                    **{state_kwarg: ssm_state, "return_final_states": True},
                )
                if isinstance(result, tuple):
                    _, ssm_state = result
            else:
                mamba_chunk_scan_combined(
                    x_chunk, dt_chunk, A, B_chunk, C_chunk,
                    chunk_size=ssd_chunk, D=D, dt_bias=dt_bias, dt_softplus=True,
                )

    n_blocks = batch_size * max(1, chunk_tok // ssd_chunk) * n_heads
    meta = {
        "n_calls":         n_calls,
        "n_blocks_per_call": n_blocks,
        "chunk_tok":       chunk_tok,
    }
    return fn, meta


def _ssm_matmul_fallback(x: torch.Tensor, n_reps: int) -> None:
    """Rough SSM memory-bandwidth proxy via matmul when mamba_ssm unavailable."""
    w = torch.ones(x.shape[-1], x.shape[-1], dtype=x.dtype, device=x.device) * 0.01
    for _ in range(n_reps):
        _ = torch.matmul(x.reshape(-1, x.shape[-1]), w)


def _build_decode_fn(
    model_cfg: dict,
    batch_size: int,
    context_len: int,
    device: str,
) -> callable:
    """Build combined decode callable: SSM (seq=1) + Attn KV read."""
    n_heads    = model_cfg["n_heads"]
    head_dim   = model_cfg["head_dim"]
    d_state    = model_cfg["d_state"]
    n_groups   = model_cfg["n_groups"]
    ssd_chunk  = model_cfg["chunk_size"]

    n_attn_heads    = model_cfg["n_attn_heads"]
    n_attn_kv_heads = model_cfg["n_attn_kv_heads"]
    attn_head_dim   = model_cfg["attn_head_dim"]

    # SSM decode tensors (seq_len=1, chunk_size=1)
    x1   = torch.randn(batch_size, 1, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    dt1  = torch.full((batch_size, 1, n_heads), 0.1, dtype=torch.bfloat16, device=device)
    A1   = -torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    B1   = torch.randn(batch_size, 1, n_groups, d_state, dtype=torch.bfloat16, device=device)
    C1   = torch.randn(batch_size, 1, n_groups, d_state, dtype=torch.bfloat16, device=device)
    D1   = torch.ones(n_heads, dtype=torch.bfloat16, device=device)
    dtb1 = torch.zeros(n_heads, dtype=torch.bfloat16, device=device)

    # Attn decode tensors
    n_kv  = n_attn_kv_heads
    Q = torch.randn(batch_size, n_attn_heads, 1, attn_head_dim,
                    dtype=torch.bfloat16, device=device)
    K = torch.randn(batch_size, n_kv, context_len, attn_head_dim,
                    dtype=torch.bfloat16, device=device)
    V = torch.randn(batch_size, n_kv, context_len, attn_head_dim,
                    dtype=torch.bfloat16, device=device)

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        mamba_chunk_scan_combined = None

    def fn():
        if mamba_chunk_scan_combined is not None:
            mamba_chunk_scan_combined(
                x1, dt1, A1, B1, C1, chunk_size=1,
                D=D1, dt_bias=dtb1, dt_softplus=True,
            )
        else:
            # Fallback: single small matmul to approximate decode memory access
            w = torch.ones(n_heads * head_dim, n_heads * head_dim,
                           dtype=torch.bfloat16, device=device) * 0.01
            torch.matmul(x1.reshape(batch_size, -1), w)

        # Attn KV read: the primary memory-BW cost of decode
        F.scaled_dot_product_attention(Q, K, V)

    return fn


# ---------------------------------------------------------------------------
# CUDA-event timing helper
# ---------------------------------------------------------------------------

def _time_fn_on_stream(
    fn: callable,
    stream: torch.cuda.Stream,
    n_warmup: int,
    n_measure: int,
) -> tuple[float, float]:
    """Return (median_ms, std_ms) for fn running on stream (solo)."""
    for _ in range(n_warmup):
        with torch.cuda.stream(stream):
            fn()
    torch.cuda.synchronize()

    lats = []
    for _ in range(n_measure):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end   = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            ev_start.record(stream)
            fn()
            ev_end.record(stream)
        torch.cuda.synchronize()
        lats.append(ev_start.elapsed_time(ev_end))

    return statistics.median(lats), (statistics.stdev(lats) if len(lats) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Concurrent measurement
# ---------------------------------------------------------------------------

def measure_concurrent(
    prefill_fn: callable,
    decode_fn: callable,
    stream_prefill: torch.cuda.Stream,
    stream_decode: torch.cuda.Stream,
    n_warmup: int,
    n_measure: int,
) -> dict:
    """Enqueue prefill + decode on their streams simultaneously; measure overlap.

    Both stream launch blocks are issued back-to-back in the host Python thread
    (no additional OS thread synchronization). The GPU hardware decides whether
    to spatially co-schedule them based on the stream/context type:
      - Green Context streams: hardware enforces SM partitioning → true co-exec
      - Regular CUDA streams:  hardware-best-effort co-scheduling (no guarantee)

    Returns per-stream elapsed times and the concurrent wall-clock time.
    """
    # Warmup
    for _ in range(n_warmup):
        with torch.cuda.stream(stream_prefill):
            prefill_fn()
        with torch.cuda.stream(stream_decode):
            decode_fn()
        torch.cuda.synchronize()

    prefill_times, decode_times, concurrent_times = [], [], []

    for _ in range(n_measure):
        ev_p0 = torch.cuda.Event(enable_timing=True)
        ev_p1 = torch.cuda.Event(enable_timing=True)
        ev_d0 = torch.cuda.Event(enable_timing=True)
        ev_d1 = torch.cuda.Event(enable_timing=True)

        # Issue both enqueues before any synchronize — this is the key:
        # the GPU sees both streams queued and can truly overlap them.
        with torch.cuda.stream(stream_prefill):
            ev_p0.record(stream_prefill)
            prefill_fn()
            ev_p1.record(stream_prefill)

        with torch.cuda.stream(stream_decode):
            ev_d0.record(stream_decode)
            decode_fn()
            ev_d1.record(stream_decode)

        torch.cuda.synchronize()

        t_p = ev_p0.elapsed_time(ev_p1)
        t_d = ev_d0.elapsed_time(ev_d1)
        concurrent_times.append(max(t_p, t_d))
        prefill_times.append(t_p)
        decode_times.append(t_d)

    return {
        "concurrent_ms":      statistics.median(concurrent_times),
        "prefill_stream_ms":  statistics.median(prefill_times),
        "decode_stream_ms":   statistics.median(decode_times),
    }


# ---------------------------------------------------------------------------
# Cooperative-safe check for chunked SSM
# ---------------------------------------------------------------------------

def _cooperative_safe(batch, chunk_tok, n_heads, ssd_chunk, sm_count) -> bool:
    nchunks = max(1, chunk_tok // ssd_chunk)
    return batch * nchunks * n_heads <= sm_count


def _choose_chunk_tokens(model_cfg: dict, sm_count: int, batch: int) -> int:
    """Choose the largest prefill_chunk_tokens that keeps cooperative-safe."""
    ssd_chunk = model_cfg["chunk_size"]
    n_heads   = model_cfg["n_heads"]
    # max nchunks_per_call = sm_count // (batch * n_heads)
    max_nchunks = max(1, sm_count // max(1, batch * n_heads))
    return max_nchunks * ssd_chunk


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    sm_splits: list[float],
    seq_lens: list[int],
    batch_sizes: list[int],
    context_lens: list[int],
    total_sm: int,
    n_warmup: int,
    n_measure: int,
    device: str,
    output_dir: Path,
    hw_cfg: dict,
) -> list[dict]:
    model_cfg = _load_ssm_cfg(model_name)
    tag = device_tag(hw_cfg)
    default_stream = torch.cuda.current_stream()

    all_rows = []

    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for context_len in context_lens:
                chunk_tok = _choose_chunk_tokens(model_cfg, total_sm, batch_size)
                chunk_tok = max(model_cfg["chunk_size"], chunk_tok)

                decode_fn   = _build_decode_fn(model_cfg, batch_size, context_len, device)
                prefill_fn, prefill_meta = _build_prefill_fn(
                    model_cfg, seq_len, batch_size, chunk_tok, device
                )

                coop_safe = _cooperative_safe(
                    batch_size, chunk_tok, model_cfg["n_heads"],
                    model_cfg["chunk_size"], total_sm
                )
                if not coop_safe:
                    print(
                        f"  [WARN] seq={seq_len} bs={batch_size} chunk={chunk_tok}: "
                        f"n_blocks={prefill_meta['n_blocks_per_call']} > sm={total_sm} "
                        f"— cooperative deadlock risk. Skipping."
                    )
                    continue

                print(
                    f"\n  seq={seq_len} bs={batch_size} ctx={context_len} "
                    f"chunk={chunk_tok} n_calls={prefill_meta['n_calls']}"
                )

                # --- Solo measurements (full SM, default stream) ---
                t_prefill_solo, _ = _time_fn_on_stream(
                    prefill_fn, default_stream, n_warmup, n_measure
                )
                t_decode_solo, _ = _time_fn_on_stream(
                    decode_fn, default_stream, n_warmup, n_measure
                )
                t_sum = t_prefill_solo + t_decode_solo
                print(
                    f"    solo: prefill={t_prefill_solo:.2f}ms  "
                    f"decode={t_decode_solo:.2f}ms  sum={t_sum:.2f}ms"
                )

                for sm_split in sm_splits:
                    n_prefill_sm = max(1, round(sm_split * total_sm))
                    n_decode_sm  = total_sm - n_prefill_sm

                    # --- Backend 1: Green Context spatial partition ---
                    stream_pf_gc, stream_dec_gc, gc_info = create_two_partitions(
                        n_prefill_sm=n_prefill_sm, device_id=0
                    )
                    gc_available = stream_pf_gc is not None

                    row_base = {
                        "model":         model_name,
                        "seq_len":       seq_len,
                        "batch_size":    batch_size,
                        "context_len":   context_len,
                        "chunk_tok":     chunk_tok,
                        "n_calls":       prefill_meta["n_calls"],
                        "total_sm":      total_sm,
                        "sm_split":      sm_split,
                        "n_prefill_sm":  n_prefill_sm,
                        "n_decode_sm":   n_decode_sm,
                        "solo_prefill_ms": round(t_prefill_solo, 3),
                        "solo_decode_ms":  round(t_decode_solo, 3),
                        "solo_sum_ms":     round(t_sum, 3),
                    }

                    if gc_available:
                        gc_result = measure_concurrent(
                            prefill_fn, decode_fn,
                            stream_pf_gc, stream_dec_gc,
                            n_warmup, n_measure,
                        )
                        overlap_ratio_gc = gc_result["concurrent_ms"] / max(t_sum, 1e-9)
                        row_gc = {
                            **row_base,
                            "backend":           "green_ctx",
                            "actual_prefill_sm": gc_info.get("actual_prefill_sm", n_prefill_sm),
                            "actual_decode_sm":  gc_info.get("actual_decode_sm", n_decode_sm),
                            "concurrent_ms":     round(gc_result["concurrent_ms"], 3),
                            "prefill_stream_ms": round(gc_result["prefill_stream_ms"], 3),
                            "decode_stream_ms":  round(gc_result["decode_stream_ms"], 3),
                            "overlap_ratio":     round(overlap_ratio_gc, 4),
                        }
                        all_rows.append(row_gc)
                        print(
                            f"    gc   split={sm_split:.0%}: "
                            f"conc={gc_result['concurrent_ms']:.2f}ms  "
                            f"overlap={overlap_ratio_gc:.3f}"
                        )
                    else:
                        print(
                            f"    gc   split={sm_split:.0%}: Green Context unavailable — "
                            f"{gc_info.get('error', 'unknown')}"
                        )

                    # --- Backend 2: Two regular streams (no SM partition) ---
                    # Hardware-best-effort co-scheduling baseline.
                    # Labelled "two_stream" — see module docstring for MPS note.
                    stream_pf_ts  = torch.cuda.Stream()
                    stream_dec_ts = torch.cuda.Stream()
                    ts_result = measure_concurrent(
                        prefill_fn, decode_fn,
                        stream_pf_ts, stream_dec_ts,
                        n_warmup, n_measure,
                    )
                    overlap_ratio_ts = ts_result["concurrent_ms"] / max(t_sum, 1e-9)
                    row_ts = {
                        **row_base,
                        "backend":           "two_stream",
                        "actual_prefill_sm": total_sm,  # no partition
                        "actual_decode_sm":  total_sm,
                        "concurrent_ms":     round(ts_result["concurrent_ms"], 3),
                        "prefill_stream_ms": round(ts_result["prefill_stream_ms"], 3),
                        "decode_stream_ms":  round(ts_result["decode_stream_ms"], 3),
                        "overlap_ratio":     round(overlap_ratio_ts, 4),
                    }
                    all_rows.append(row_ts)
                    print(
                        f"    2str split={sm_split:.0%}: "
                        f"conc={ts_result['concurrent_ms']:.2f}ms  "
                        f"overlap={overlap_ratio_ts:.3f}"
                    )

    return all_rows


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "model", "seq_len", "batch_size", "context_len", "chunk_tok", "n_calls",
    "total_sm", "sm_split", "n_prefill_sm", "n_decode_sm",
    "actual_prefill_sm", "actual_decode_sm",
    "backend",
    "solo_prefill_ms", "solo_decode_ms", "solo_sum_ms",
    "concurrent_ms", "prefill_stream_ms", "decode_stream_ms",
    "overlap_ratio",
]


def save_csv(rows: list[dict], output_dir: Path, model_name: str, tag: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"coexec_{model_name}_{tag}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {out}  ({len(rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_overlap(rows: list[dict], output_dir: Path, model_name: str, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figure")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return

    backends = df["backend"].unique()
    seq_lens = sorted(df["seq_len"].unique())
    colors   = {"green_ctx": "#4CAF50", "two_stream": "#9E9E9E"}
    ls_map   = {"green_ctx": "-", "two_stream": "--"}
    label_map = {
        "green_ctx":  "Green Context (spatial partition)",
        "two_stream": "Two-stream (no partition, MPS-like baseline)",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for backend in sorted(backends):
        sub = df[df["backend"] == backend]
        color = colors.get(backend, "#000000")
        ls    = ls_map.get(backend, "-")
        for sl in seq_lens:
            g = sub[sub["seq_len"] == sl].sort_values("sm_split")
            if g.empty:
                continue
            ax.plot(
                g["sm_split"] * 100, g["overlap_ratio"],
                color=color, linestyle=ls, marker="o", markersize=5,
                label=f"{label_map.get(backend, backend)} (seq={sl})" if sl == seq_lens[0] else "_nolegend_",
            )

    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2, label="Perfect overlap (0.5)")
    ax.axhline(1.0, color="red",   linestyle=":", linewidth=1.0, label="No overlap (1.0 = sequential)")

    ax.set_xlabel("Prefill SM allocation (% of total)")
    ax.set_ylabel("Overlap ratio  (concurrent / solo_sum)")
    ax.set_title(
        f"Concurrent Execution Overlap — {model_name}\n"
        "overlap_ratio: 0.5 = perfect overlap, 1.0 = sequential",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    out = output_dir / f"coexec_{model_name}_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concurrent prefill+decode overlap microbenchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="zamba2",
                   choices=["zamba2", "nemotron_h", "falcon_h1"])
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--sm-splits", nargs="+", type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Prefill SM fraction of total (e.g. 0.5 = 50%% of SMs for prefill)",
    )
    p.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    p.add_argument("--context-lens", nargs="+", type=int, default=[1024, 4096],
                   help="KV context length for Attn decode")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=20)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "results" / "stage3",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    _hw_guard()

    hw_cfg  = get_hardware_config(args.device)
    tag     = device_tag(hw_cfg)
    total_sm = hw_cfg["sm_count"]

    print(f"\n  Model:      {args.model}")
    print(f"  GPU:        {hw_cfg.get('name', 'unknown')}  ({total_sm} SMs)")
    print(f"  SM splits:  {[f'{s:.0%}' for s in args.sm_splits]}")
    print(f"  seq_lens:   {args.seq_lens}")
    print(f"  batch_sizes:{args.batch_sizes}")
    print(f"  ctx_lens:   {args.context_lens}")
    print(f"  warmup/measure: {args.n_warmup}/{args.n_measure}")

    # Check Green Context availability
    smctrl = SMController(total_sm_count=total_sm)
    if not smctrl.is_available():
        print(
            "\n  NOTE: Green Context not available on this GPU/driver.\n"
            "  Only 'two_stream' backend will be measured.\n"
            "  Green Context requires CUDA driver 550+ and Hopper/Ampere datacenter GPU."
        )

    rows = run_sweep(
        model_name=args.model,
        sm_splits=args.sm_splits,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        context_lens=args.context_lens,
        total_sm=total_sm,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        device="cuda",
        output_dir=args.output_dir,
        hw_cfg=hw_cfg,
    )

    if not rows:
        print("  No rows produced — check cooperative_safe conditions.")
        return

    csv_path = save_csv(rows, args.output_dir, args.model, tag)
    plot_overlap(rows, args.output_dir, args.model, tag)

    print(
        f"\n  coexec_microbench complete.\n"
        f"  CSV:    {csv_path}\n"
        f"  Next:   python stage3_hm_eval/projection_analysis.py "
        f"--coexec-csv {csv_path}"
    )


if __name__ == "__main__":
    main()
