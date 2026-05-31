"""Pytest root conftest — sets sys.path for the workspace layout.

Run tests from the ``workspace/`` directory:
    pytest characterization/tests/ -v -m "not cuda"
"""
import sys
from pathlib import Path

_workspace = Path(__file__).parent
# workspace/ → shared.loaders importable
sys.path.insert(0, str(_workspace))
# workspace/characterization/ → src.* importable
sys.path.insert(0, str(_workspace / "characterization"))
