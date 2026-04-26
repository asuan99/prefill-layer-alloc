"""
NVTX range markers for Nsight Systems (nsys) correlation.

When the sweep runs under `nsys profile`, NVTX ranges appear in the timeline
as colored bands that can be filtered/selected to isolate specific
(sm_count, seq_len, batch_size) configurations.

Usage — running a sweep under nsys:
    nsys profile \\
        --trace=cuda,nvtx \\
        --capture-range=nvtx \\
        --nvtx-capture="sweep" \\
        --output=results/stage1/nsys_ssm_zamba2 \\
        python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2

    # Open the .nsys-rep file in Nsight Systems GUI, or:
    nsys stats --report gputrace results/stage1/nsys_ssm_zamba2.nsys-rep

Usage — in code (automatic when NVTXMarker is instantiated):
    with NVTXMarker.range(f"sm{sm_count}_seq{seq_len}_bs{batch_size}"):
        runner.run_ssm_layer(...)

The markers use three nested levels:
  Level 0 (color=green):  "sweep"           — entire benchmark run
  Level 1 (color=blue):   "sm{N}_seq{L}"    — one (sm_count, seq_len) combo
  Level 2 (color=yellow): "measure"         — actual measurement window
  Level 2 (color=gray):   "warmup"          — warm-up window

When libNvToolsExt is unavailable, all calls are no-ops (graceful degradation).
"""

import ctypes
import os
import sys
from contextlib import contextmanager
from typing import Optional


# NVTX color constants (ARGB)
_COLOR_GREEN  = 0xFF00C853
_COLOR_BLUE   = 0xFF1565C0
_COLOR_YELLOW = 0xFFFDD835
_COLOR_GRAY   = 0xFF90A4AE
_COLOR_RED    = 0xFFB71C1C
_COLOR_ORANGE = 0xFFE65100

_LAYER_COLORS = {
    "ssm":  _COLOR_BLUE,
    "attn": _COLOR_ORANGE,
    "mlp":  _COLOR_GREEN,
}


def _load_nvtx_lib() -> Optional[ctypes.CDLL]:
    """Try to load libnvToolsExt (bundled with CUDA toolkit)."""
    candidates = [
        "libnvToolsExt.so.1",
        "libnvToolsExt.so",
        # Windows
        "nvToolsExt64_1.dll",
    ]
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    if cuda_home:
        candidates = [
            os.path.join(cuda_home, "lib64", "libnvToolsExt.so.1"),
            os.path.join(cuda_home, "lib", "libnvToolsExt.so.1"),
        ] + candidates

    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            # Verify core symbols
            _ = lib.nvtxRangePushA
            _ = lib.nvtxRangePop
            lib.nvtxRangePushA.restype = ctypes.c_int
            lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
            lib.nvtxRangePop.restype = ctypes.c_int
            lib.nvtxRangePop.argtypes = []
            # nvtxMarkA for point events
            lib.nvtxMarkA.restype = None
            lib.nvtxMarkA.argtypes = [ctypes.c_char_p]
            return lib
        except (OSError, AttributeError):
            continue
    return None


_NVTX_LIB = _load_nvtx_lib()
_NVTX_AVAILABLE = _NVTX_LIB is not None

# Try nvtx Python package as alternative backend
if not _NVTX_AVAILABLE:
    try:
        import nvtx as _nvtx_pkg
        _NVTX_PKG = _nvtx_pkg
        _NVTX_AVAILABLE = True
    except ImportError:
        _NVTX_PKG = None
else:
    _NVTX_PKG = None


def is_available() -> bool:
    """Return True if NVTX is available (either ctypes or nvtx package)."""
    return _NVTX_AVAILABLE


def is_running_under_nsys() -> bool:
    """Heuristic: check if the process was launched under nsys profile."""
    # nsys sets NSYS_PROFILING_SESSION_ID or injects into LD_PRELOAD
    return (
        "NSYS_PROFILING_SESSION_ID" in os.environ
        or any("nsys" in s.lower() for s in os.environ.get("LD_PRELOAD", "").split(":"))
    )


class NVTXMarker:
    """NVTX range/mark manager.

    Provides static methods for push/pop ranges and point markers.
    Degrades gracefully when NVTX is unavailable (all calls are no-ops).
    """

    @staticmethod
    def push(label: str, color: int = _COLOR_BLUE) -> None:
        """Push a named NVTX range onto the stack."""
        if not _NVTX_AVAILABLE:
            return
        if _NVTX_LIB is not None:
            _NVTX_LIB.nvtxRangePushA(label.encode())
        elif _NVTX_PKG is not None:
            _NVTX_PKG.push(label)

    @staticmethod
    def pop() -> None:
        """Pop the innermost NVTX range."""
        if not _NVTX_AVAILABLE:
            return
        if _NVTX_LIB is not None:
            _NVTX_LIB.nvtxRangePop()
        elif _NVTX_PKG is not None:
            _NVTX_PKG.pop()

    @staticmethod
    def mark(label: str) -> None:
        """Insert a point event marker."""
        if not _NVTX_AVAILABLE:
            return
        if _NVTX_LIB is not None:
            _NVTX_LIB.nvtxMarkA(label.encode())
        elif _NVTX_PKG is not None:
            _NVTX_PKG.mark(label)

    @staticmethod
    @contextmanager
    def range(label: str, color: int = _COLOR_BLUE):
        """Context manager for a named NVTX range.

        Example:
            with NVTXMarker.range("ssm_sm27_seq1024_bs4", color=_COLOR_BLUE):
                measure_ssm_layer(...)
        """
        NVTXMarker.push(label, color)
        try:
            yield
        finally:
            NVTXMarker.pop()

    @staticmethod
    @contextmanager
    def config_range(
        layer_type: str,
        sm_count: int,
        seq_len: int,
        batch_size: int,
        total_sm: int,
    ):
        """Standard range for one (layer_type, sm_count, seq_len, batch_size) config.

        Label format: "ssm/sm27(25%)/seq1024/bs4"
        Color: layer_type specific.
        """
        sm_pct = int(sm_count / total_sm * 100)
        label = f"{layer_type}/sm{sm_count}({sm_pct}%)/seq{seq_len}/bs{batch_size}"
        color = _LAYER_COLORS.get(layer_type, _COLOR_GRAY)
        with NVTXMarker.range(label, color):
            yield

    @staticmethod
    @contextmanager
    def warmup_range():
        """Mark the warm-up phase."""
        with NVTXMarker.range("warmup", _COLOR_GRAY):
            yield

    @staticmethod
    @contextmanager
    def measure_range():
        """Mark the actual measurement phase."""
        with NVTXMarker.range("measure", _COLOR_GREEN):
            yield
