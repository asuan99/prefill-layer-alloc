"""
Shared config loaders for the hybrid-serving workspace.

Single source of truth for models.yaml and hardware.yaml stored in
``shared/configs/``.  Both ``characterization`` and ``serving-eval``
sub-projects import from here instead of reading YAML directly.

Device-tag enforcement (fixes C-3/C-4):
  ``get_hardware_config(..., device_tag=<expected>)`` raises ``ValueError``
  when the resolved config's normalised name does not match the caller's
  expectation, preventing result mixing across GPU types (C-3) and ensuring
  output files are always tagged with the correct device (C-4).
"""

from pathlib import Path
from typing import Optional

import yaml

_CONFIGS_DIR = Path(__file__).parent / "configs"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sm_steps(total_sm: int, n_steps: int = 8) -> list:
    steps = []
    for i in range(1, n_steps + 1):
        sm = max(1, round(total_sm * i / n_steps))
        if sm not in steps:
            steps.append(sm)
    return sorted(steps)


def _make_tag(name: str) -> str:
    """Normalise a GPU display name to a filesystem-safe device tag."""
    return (
        name.lower()
        .replace("nvidia ", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Canonical size-suffixed aliases for the 7B/8B-scale entries, so the sweep can
# name them consistently with the SLM keys (zamba2_1.2b, falcon_h1_3b, …).
# The targets are verified against the cached HF config.json (see models.yaml).
_MODEL_ALIASES = {
    "zamba2_7b": "zamba2",
    "falcon_h1_7b": "falcon_h1",
    "nemotron_h_8b": "nemotron_h",
}


def get_model_config(name: str) -> dict:
    """Return the full model config dict for *name* from ``models.yaml``.

    Args:
        name: Top-level key in models.yaml (e.g. ``"zamba2_1.2b"``), or a
            size-suffixed alias for the large entries (``"zamba2_7b"``,
            ``"falcon_h1_7b"``, ``"nemotron_h_8b"``).

    Raises:
        KeyError: Model not found in models.yaml.
    """
    name = _MODEL_ALIASES.get(name, name)
    with open(_CONFIGS_DIR / "models.yaml") as f:
        all_models = yaml.safe_load(f)
    if name not in all_models:
        raise KeyError(
            f"Model {name!r} not in models.yaml.  "
            f"Available: {sorted(all_models)} (+ aliases {sorted(_MODEL_ALIASES)})"
        )
    return all_models[name]


def get_all_model_configs() -> dict:
    """Return every model config as ``{name: cfg_dict}`` from ``models.yaml``."""
    with open(_CONFIGS_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def get_hardware_config(key: str, *, device_tag: Optional[str] = None) -> dict:
    """Return hardware config dict for *key* from ``hardware.yaml``.

    Args:
        key:        YAML key (e.g. ``"a100_80gb"``) or ``"auto"`` for runtime
                    detection.  Hyphens are normalised to underscores before
                    lookup (``"a100-sxm4-80gb"`` → ``"a100_sxm4_80gb"``).
        device_tag: When provided, validate that the resolved config's
                    normalised name matches this value.  Prevents C-3 (result
                    mixing across GPU types) and C-4 (untagged output files).

    Returns:
        Dict with keys: ``name``, ``sm_count``, ``sm_sweep_steps``,
        ``memory_bw_GBs``, ``memory_GB``, ``device_tag``.

    Raises:
        ValueError: ``device_tag`` mismatch between config and caller.
    """
    import torch

    normalized = key.replace("-", "_")

    with open(_CONFIGS_DIR / "hardware.yaml") as f:
        hw = yaml.safe_load(f)

    if key != "auto" and normalized in hw:
        cfg = dict(hw[normalized])
        if cfg.get("sm_sweep_steps") is None:
            cfg["sm_sweep_steps"] = _compute_sm_steps(cfg["sm_count"])
        tag = _make_tag(cfg["name"])
        cfg["device_tag"] = tag
        if device_tag is not None and tag != device_tag:
            raise ValueError(
                f"device_tag mismatch: config '{normalized}' normalises to "
                f"{tag!r} but caller expected {device_tag!r}.  "
                f"Check --device argument."
            )
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

    detected_name = torch.cuda.get_device_name(0)
    tag = _make_tag(detected_name)

    if device_tag is not None and tag != device_tag:
        raise ValueError(
            f"device_tag mismatch: detected GPU is {detected_name!r} "
            f"(normalised: {tag!r}) but caller expected {device_tag!r}."
        )

    return {
        "name":           detected_name,
        "sm_count":       n_sm,
        "sm_sweep_steps": _compute_sm_steps(n_sm),
        "memory_bw_GBs":  mem_bw_GBs,
        "memory_GB":      None,
        "device_tag":     tag,
    }


def device_tag(hw_cfg: dict) -> str:
    """Return the filesystem-safe device tag from a hardware config dict.

    Prefers the pre-computed ``device_tag`` field set by
    :func:`get_hardware_config`; falls back to normalising ``name``.
    """
    return hw_cfg.get("device_tag") or _make_tag(hw_cfg["name"])
