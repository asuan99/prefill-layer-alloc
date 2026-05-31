"""Backward-compatibility shim — delegates to shared.loaders.

All config loading has moved to ``shared/loaders.py`` (single source of
truth).  Callers that do ``from src.hardware_config import get_hardware_config``
continue to work unchanged; new code should import from ``shared.loaders``
directly.
"""

import sys
from pathlib import Path

# Ensure workspace/ is in sys.path so ``shared`` is importable when this
# module is first loaded (scripts add it before importing src.*, but library
# callers may not have done so yet).
_workspace = Path(__file__).resolve().parent.parent.parent
if str(_workspace) not in sys.path:
    sys.path.insert(0, str(_workspace))

from shared.loaders import get_hardware_config, device_tag  # noqa: F401
