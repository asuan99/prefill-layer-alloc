"""
_mamba_compat.py — make `import mamba_ssm` succeed when its prebuilt CUDA
extensions are ABI-mismatched against the installed torch.

The chunked-prefill SSM measurement uses ONLY the Triton kernel
(mamba_chunk_scan_combined); it never calls the compiled selective_scan / causal_conv1d
ops.  But mamba_ssm/__init__.py eagerly imports them, so a torch upgrade that breaks
those .so files (undefined c10::cuda symbol / missing libcudart.so.13) blocks the whole
package — and with it the Triton path we DO need.

ensure_mamba_importable() first tries the real import; only if it fails with the known
ABI/loader errors does it install harmless stub modules for the compiled ops so the
package import (and the Triton kernel) work.  If the real import already succeeds, nothing
is shimmed.
"""
from __future__ import annotations

import sys
import types


def ensure_mamba_importable() -> bool:
    """Return True once mamba_ssm's Triton kernel is importable.

    Returns:
        True if importable for real, or made importable via stubs.
    Raises:
        the original ImportError if the failure is NOT the known compiled-ext ABI issue.
    """
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
        return True
    except ImportError as e:
        msg = str(e)
        known = (
            "selective_scan_cuda" in msg
            or "causal_conv1d_cuda" in msg
            or "libcudart" in msg
            or "undefined symbol" in msg
        )
        if not known:
            raise
        # Stub the broken compiled ops so the package __init__ completes.
        for name in ("selective_scan_cuda", "causal_conv1d_cuda"):
            sys.modules.setdefault(name, types.ModuleType(name))
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
        print("[_mamba_compat] stubbed broken compiled ext(s); "
              "Triton mamba_chunk_scan_combined is active.", flush=True)
        return True
