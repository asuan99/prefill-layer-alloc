import sys
import ast

# Restore ast.Num/Str/NameConstant for Python 3.12+ (removed, replaced by ast.Constant).
# Triton 3.5.x code_generator.py still calls ast.Num(0)/ast.Num(1) when JIT-compiling
# kernels that contain range() loops — raises AttributeError on Python 3.12+.
if sys.version_info >= (3, 12) and not hasattr(ast, 'Num'):
    ast.Num = ast.Constant
    ast.Str = ast.Constant
    ast.NameConstant = ast.Constant
    ast.Ellipsis = ast.Constant

# Mock selective_scan_cuda (Mamba-1 C extension) if it cannot load.
#
# mamba_ssm/__init__.py imports selective_scan_fn from selective_scan_interface.py,
# which does `import selective_scan_cuda` unconditionally. That .so is compiled
# against a fixed PyTorch ABI and breaks with "undefined symbol" when the installed
# PyTorch version differs. We only use the Mamba-2 pure-Triton kernel
# (mamba_chunk_scan_combined from ssd_combined.py), which does NOT use
# selective_scan_cuda at runtime. Installing a mock before any mamba_ssm import
# lets __init__.py load without error; the mock is never actually called.
if 'selective_scan_cuda' not in sys.modules:
    try:
        import selective_scan_cuda  # noqa: F401
    except Exception:
        from unittest.mock import MagicMock
        sys.modules['selective_scan_cuda'] = MagicMock()
