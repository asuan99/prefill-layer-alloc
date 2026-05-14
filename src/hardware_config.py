"""
Hardware configuration loader.

Wraps configs/hardware.yaml into a dict with sm_count, sm_sweep_steps, etc.
Supports auto-detection via torch device properties.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load_yaml() -> dict:
    cfg_path = _CONFIGS_DIR / "hardware.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _compute_sm_steps(total_sm: int, n_steps: int = 8) -> list[int]:
    steps = []
    for i in range(1, n_steps + 1):
        sm = max(1, round(total_sm * i / n_steps))
        if sm not in steps:
            steps.append(sm)
    return sorted(steps)


def get_hardware_config(device_key: str) -> dict:
    """Return hardware config dict for the given device key.

    Args:
        device_key: Key from configs/hardware.yaml (e.g. 'a100_sxm4_80gb',
            'a100-sxm4-80gb').  Hyphens are normalised to underscores before
            lookup.  Pass 'auto' to detect from the current CUDA device.

    Returns:
        {
            'name':           str,
            'sm_count':       int,
            'sm_sweep_steps': list[int],
            'memory_bw_GBs':  float | None,
            'memory_GB':      int | None,
        }
    """
    # Normalize hyphens (CLI-friendly "a100-sxm4-80gb" → yaml key "a100_sxm4_80gb")
    normalized = device_key.replace("-", "_")

    hw = _load_yaml()

    if device_key != "auto" and normalized in hw:
        cfg = dict(hw[normalized])
        if cfg.get("sm_sweep_steps") is None:
            cfg["sm_sweep_steps"] = _compute_sm_steps(cfg["sm_count"])
        return cfg

    # Auto-detect from current CUDA device
    props = torch.cuda.get_device_properties(0)
    n_sm = props.multi_processor_count
    try:
        mem_bw_GBs = (
            2.0 * props.memory_clock_rate * 1e3 * props.memory_bus_width
        ) / (8.0 * 1e9)
    except Exception:
        mem_bw_GBs = None

    return {
        "name":           torch.cuda.get_device_name(0),
        "sm_count":       n_sm,
        "sm_sweep_steps": _compute_sm_steps(n_sm),
        "memory_bw_GBs":  mem_bw_GBs,
        "memory_GB":      None,
    }
