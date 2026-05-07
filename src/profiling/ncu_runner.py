"""
Nsight Compute (ncu) subprocess launcher and CSV parser.

ncu cannot be called from within a running Python process because it works
via CUDA API intercept (LD_PRELOAD). This module spawns a child process
under ncu and parses the resulting CSV output.

Architecture:
  NCURunner.profile() → spawns: ncu [metrics] python _ncu_target.py [args]
                      → collects stdout CSV
                      → parses and returns dict of metric values

Metric naming by architecture
──────────────────────────────
Ampere / pre-Blackwell (sm_8x / sm_9x):
  sm__active_cycles_sum          — SM-active cycles
  sm__cycles_elapsed_sum         — elapsed SM cycles
  smsp__warps_active_avg_per_cycle_active
  smsp__maximum_warps_per_active_cycle_pct
  launch__grid_size              — grid dimensions (scalar metric)
  launch__block_size
  dram__bytes_read_sum / dram__bytes_write_sum

Blackwell (sm_10x / sm_12x, ncu ≥ 2025.2):
  sm__cycles_active.sum          — dot-notation suffix
  sm__cycles_elapsed.sum
  smsp__warps_active.sum         — warp count (not avg/pct)
  "Grid Size" / "Block Size"     — CSV columns, format "(x, y, z)"
  dram__bytes_op_read.sum / dram__bytes_op_write.sum
  (launch__grid_size removed entirely)

NCU_METRICS_WAVE: curated set for wave quantization + SM utilization.
NCU_METRICS_FULL: extended set including memory and compute throughput.
"""

import csv
import io
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _get_sm_major() -> int:
    """Return the CUDA compute capability major version of device 0, or 0."""
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return major
    except Exception:
        pass
    return 0


def _get_max_warps_per_sm() -> int:
    """Query max resident warps per SM from CUDA device properties.

    Uses max_threads_per_multi_processor / 32 (warp size).
    Falls back to 64 (Ampere A100 value) if CUDA is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return p.max_threads_per_multi_processor // 32
    except Exception:
        pass
    return 64  # conservative fallback (A100/H100)


def _is_blackwell(sm_major: int) -> bool:
    # Blackwell = sm_10x and sm_12x (GB100 / GB200 / GB20x)
    return sm_major >= 10


# ---------------------------------------------------------------------------
# Metric sets — legacy (pre-Blackwell) and Blackwell
# ---------------------------------------------------------------------------

# --- Legacy (Ampere / Hopper: sm_8x, sm_9x) ---

_LEGACY_METRICS_WAVE = [
    "sm__active_cycles_sum",
    "sm__cycles_elapsed_sum",
    "smsp__warps_active_avg_per_cycle_active",
    "smsp__maximum_warps_per_active_cycle_pct",
    "launch__grid_size",
    "launch__block_size",
    "gpu__time_duration",
]

_LEGACY_METRICS_MEMORY = [
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st_sum",
    "dram__bytes_read_sum",
    "dram__bytes_write_sum",
    "l2__global_load_bytes_sum",
    "l2__global_store_bytes_sum",
]

_LEGACY_METRICS_COMPUTE = [
    "sm__sass_thread_inst_executed_op_fadd_pred_on_sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on_sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on_sum",
]

# --- Blackwell (sm_10x / sm_12x, ncu ≥ 2025.2) ---

_BLACKWELL_METRICS_WAVE = [
    "sm__cycles_active.sum",
    "sm__cycles_elapsed.sum",
    # warp count (no _avg suffix in Blackwell)
    "smsp__warps_active.sum",
    # Grid/Block Size come from CSV columns, not metrics — no launch__ metrics
    "gpu__time_duration.sum",
]

_BLACKWELL_METRICS_MEMORY = [
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
]

_BLACKWELL_METRICS_COMPUTE = [
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
]

# Public constants — resolved to the correct arch at module load time
_SM_MAJOR = _get_sm_major()
_USE_BLACKWELL = _is_blackwell(_SM_MAJOR)

if _USE_BLACKWELL:
    NCU_METRICS_WAVE    = _BLACKWELL_METRICS_WAVE
    NCU_METRICS_MEMORY  = _BLACKWELL_METRICS_MEMORY
    NCU_METRICS_COMPUTE = _BLACKWELL_METRICS_COMPUTE
else:
    NCU_METRICS_WAVE    = _LEGACY_METRICS_WAVE
    NCU_METRICS_MEMORY  = _LEGACY_METRICS_MEMORY
    NCU_METRICS_COMPUTE = _LEGACY_METRICS_COMPUTE

NCU_METRICS_FULL = NCU_METRICS_WAVE + NCU_METRICS_MEMORY + NCU_METRICS_COMPUTE

# Max resident warps per SM — used for occupancy calculation from smsp__warps_active.sum
# Queried from CUDA device properties at import time (max_threads_per_sm / 32).
# Examples: A100=64, RTX 3090=48, RTX 5060 Ti GB206=48
_MAX_WARPS_PER_SM = _get_max_warps_per_sm()


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_grid_size(raw: str) -> int:
    """Parse Grid/Block Size column value '(x, y, z)' → x*y*z."""
    raw = raw.strip().strip("()")
    try:
        parts = [int(p.strip()) for p in raw.split(",")]
        result = 1
        for p in parts:
            result *= p
        return result
    except (ValueError, AttributeError):
        return 0


def _parse_ncu_csv(csv_text: str) -> list[dict]:
    """Parse ncu --csv output into list of dicts, one per kernel launch.

    ncu CSV format (CUDA 12+):
      "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
      "Context","Stream","Grid Size","Block Size",
      "Section Name","Metric Name","Metric Unit","Metric Value"

    "Grid Size" / "Block Size" are direct columns (format "(x, y, z)").
    Multiple rows share the same kernel ID but differ in Metric Name.
    We pivot to one dict per kernel, with metric names as keys.
    """
    all_lines = csv_text.strip().splitlines()

    # Skip ncu status lines (==PROF==, ==WARNING==) and any subprocess stdout
    # that leaks before the CSV header. The CSV header always starts with "ID".
    csv_start = None
    for i, line in enumerate(all_lines):
        stripped = line.lstrip('"').lstrip()
        if stripped.startswith("ID") or line.startswith('"ID"'):
            csv_start = i
            break

    if csv_start is None:
        return []

    lines = all_lines[csv_start:]
    if not lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    kernels: dict[str, dict] = {}

    for row in reader:
        kid = row.get("ID", "0")
        if kid not in kernels:
            grid_raw = row.get("Grid Size", "")
            block_raw = row.get("Block Size", "")
            kernels[kid] = {
                "kernel_id": kid,
                "kernel_name": row.get("Kernel Name", ""),
                "kernel_time_ns": row.get("Kernel Time", ""),
            }
            if grid_raw:
                kernels[kid]["grid_size_raw"] = grid_raw
                kernels[kid]["_grid_size_parsed"] = _parse_grid_size(grid_raw)
            if block_raw:
                kernels[kid]["block_size_raw"] = block_raw
                kernels[kid]["_block_size_parsed"] = _parse_grid_size(block_raw)

        metric_name = row.get("Metric Name", "").strip()
        metric_val  = row.get("Metric Value", "").strip()
        metric_unit = row.get("Metric Unit", "").strip()
        if metric_name:
            kernels[kid][metric_name] = metric_val
            kernels[kid][f"{metric_name}.unit"] = metric_unit

    return list(kernels.values())


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def _safe_float(d: dict, *keys) -> Optional[float]:
    """Try each key in order, return first parseable float or None."""
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            try:
                return float(str(v).replace(",", ""))
            except (ValueError, TypeError):
                pass
    return None


def _derive_sm_util(kernel_dict: dict, sm_count: int, use_blackwell: bool = None) -> dict:
    """Compute derived SM utilization and wave stats from raw ncu counters.

    Handles both legacy (sm__active_cycles_sum) and Blackwell
    (sm__cycles_active.sum) metric names.

    Per-SM utilization:
      sm_util_pct = active_cycles_sum / elapsed_cycles_sum × 100
      (averaged over all SMs — active_sum / elapsed_sum is already the mean)

    Wave stats:
      grid_size comes from:
        1. "_grid_size_parsed" (CSV column "Grid Size")       ← Blackwell
        2. "launch__grid_size" metric                         ← legacy
      wave count = ceil(grid_size / sm_count)
      wave_efficiency = grid_size / (n_waves × sm_count)
    """
    if use_blackwell is None:
        use_blackwell = _USE_BLACKWELL

    results = {}
    try:
        # SM cycle counters — try Blackwell names first, then legacy
        active_sum = _safe_float(
            kernel_dict,
            "sm__cycles_active.sum",
            "sm__active_cycles_sum",
        )
        elapsed_sum = _safe_float(
            kernel_dict,
            "sm__cycles_elapsed.sum",
            "sm__cycles_elapsed_sum",
        )

        if active_sum is not None and elapsed_sum is not None and elapsed_sum > 0:
            results["sm_util_per_sm_pct"] = active_sum / elapsed_sum * 100.0

        # Grid size — CSV column takes priority over launch__grid_size metric
        grid_size = kernel_dict.get("_grid_size_parsed")
        if not grid_size:
            gs_raw = _safe_float(kernel_dict, "launch__grid_size")
            grid_size = int(gs_raw) if gs_raw is not None else 0
        else:
            grid_size = int(grid_size)

        if grid_size > 0 and sm_count > 0:
            n_waves = math.ceil(grid_size / sm_count)
            last_wave = grid_size % sm_count
            results["grid_size"] = grid_size
            results["n_waves"] = n_waves
            results["last_wave_blocks"] = last_wave if last_wave > 0 else sm_count
            results["wave_efficiency_pct"] = grid_size / (n_waves * sm_count) * 100.0

        # Occupancy
        # Legacy (pre-Blackwell): smsp__maximum_warps_per_active_cycle_pct — direct %
        # Blackwell: metric removed; compute from smsp__warps_active.sum:
        #   occupancy = warps_active_sum / (elapsed_sum * smsp_per_sm * max_warps_per_smsp)
        #   For sm_8x–sm_12x consumer: 4 SMSP/SM × 16 warps/SMSP = 64 max warps/SM
        occ = _safe_float(kernel_dict, "smsp__maximum_warps_per_active_cycle_pct")
        if occ is not None:
            results["achieved_occupancy_pct"] = occ
        else:
            # Blackwell path: derive from smsp__warps_active.sum counter
            # smsp__warps_active.sum = total warp-cycles across all SMs
            # sm__cycles_elapsed.sum = total elapsed cycles across all SMs
            # → avg warps per SM per cycle = warps_sum / elapsed_sum
            # → occupancy = avg_warps_per_sm / max_warps_per_sm
            # max_warps_per_sm = max_threads_per_multi_processor / 32 (from device props)
            warps_sum = _safe_float(kernel_dict, "smsp__warps_active.sum")
            if warps_sum is not None and elapsed_sum is not None and elapsed_sum > 0:
                results["achieved_occupancy_pct"] = (
                    warps_sum / (elapsed_sum * _MAX_WARPS_PER_SM) * 100.0
                )

        # Active warps raw counter (stored for reference)
        warps = _safe_float(
            kernel_dict,
            "smsp__warps_active.sum",
            "smsp__warps_active_avg_per_cycle_active",
        )
        if warps is not None:
            results["warps_active"] = warps

        # Kernel duration (ns)
        dur = _safe_float(
            kernel_dict,
            "gpu__time_duration.sum",
            "gpu__time_duration",
        )
        if dur is not None:
            results["kernel_duration_ns"] = dur

    except (ValueError, ZeroDivisionError):
        pass

    return results


# ---------------------------------------------------------------------------
# NCURunner
# ---------------------------------------------------------------------------

class NCURunner:
    """Launch ncu subprocess and return parsed per-kernel metrics.

    Args:
        ncu_path: Path to ncu binary. Auto-detected from PATH if None.
                  Prefers /usr/local/cuda/bin/ncu (matches driver 580 / CUDA 13).
        python_path: Python interpreter for the target script.
        target_script: Path to _ncu_target.py.
    """

    def __init__(
        self,
        ncu_path: Optional[str] = None,
        python_path: Optional[str] = None,
        target_script: Optional[Path] = None,
    ):
        self.ncu_path = ncu_path or self._find_ncu()
        self.python_path = python_path or sys.executable
        self.target_script = target_script or (
            Path(__file__).parent / "_ncu_target.py"
        )
        self.use_blackwell = _USE_BLACKWELL

    def _find_ncu(self) -> str:
        """Find ncu binary.

        Priority order:
          1. /usr/local/cuda/bin/ncu   (CUDA toolkit install, matches driver 580+)
          2. $CUDA_HOME/bin/ncu
          3. ncu on PATH
        """
        preferred = ["/usr/local/cuda/bin/ncu"]
        cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
        if cuda_home:
            preferred.append(os.path.join(cuda_home, "bin", "ncu"))
        preferred += ["ncu", "ncu.exe"]

        for c in preferred:
            try:
                result = subprocess.run(
                    [c, "--version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        raise FileNotFoundError(
            "ncu not found. Install CUDA toolkit 13+ and ensure ncu is on PATH.\n"
            "Expected: /usr/local/cuda/bin/ncu"
        )

    def is_available(self) -> bool:
        try:
            self._find_ncu()
            return True
        except FileNotFoundError:
            return False

    def check_permissions(self) -> tuple[bool, str]:
        """Check whether ncu can access GPU hardware performance counters.

        ncu requires NVreg_RestrictProfilingToAdminUsers=0 (set by cluster admin)
        or elevated privileges. On HPC clusters this is typically only available
        inside a properly configured SLURM job — not in interactive sessions.

        Returns:
            (ok, message) — ok=True if counters are accessible, False with
            a descriptive message explaining the failure and how to fix it.
        """
        import sys
        probe_cmd = [
            self.ncu_path,
            "--metrics", "gpu__time_duration",
            "--csv",
            sys.executable,
            "-c",
            (
                "import torch; "
                "x=torch.ones(64,device='cuda',dtype=torch.float16); "
                "y=x+x; "
                "torch.cuda.synchronize()"
            ),
        ]
        try:
            proc = subprocess.run(
                probe_cmd, capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, str(e)

        combined = proc.stdout + proc.stderr
        if "ERR_NVGPUCTRPERM" in combined:
            return False, (
                "ERR_NVGPUCTRPERM — GPU performance counter access denied.\n"
                "  On HPC clusters this is controlled by the system administrator.\n"
                "  Solutions:\n"
                "    1. Submit via SLURM: sbatch slurm/run_ncu_profile.sh\n"
                "       (SLURM jobs on the gpu partition may have counter access)\n"
                "    2. Ask admin to set: NVreg_RestrictProfilingToAdminUsers=0\n"
                "    3. Interactive access: srun --gres=gpu:1 --comment=pytorch "
                "--pty bash, then retry\n"
                f"  ncu stdout: {proc.stdout[:300]}"
            )
        if proc.returncode != 0:
            return False, (
                f"ncu probe failed (exit {proc.returncode}).\n"
                f"  stdout: {proc.stdout[:300]}\n"
                f"  stderr: {proc.stderr[:200]}"
            )
        return True, "ok"

    def profile(
        self,
        layer_type: str,
        model: str,
        sm_count: int,
        seq_len: int,
        batch_size: int = 1,
        context_len: int = 0,
        metrics: list[str] = None,
        n_warmup: int = 10,
        n_measure: int = 3,
        timeout_s: int = 300,
        extra_ncu_args: list[str] = None,
    ) -> dict:
        """Run ncu on a single (layer_type, model, sm_count, seq_len) config.

        ncu command structure:
          ncu
            --metrics <metric1,metric2,...>
            --csv                               ← machine-readable output
            --target-processes all             ← profile child processes too
            --clock-control none               ← don't lock clocks
            --nvtx --nvtx-include "ncu_measure" ← only kernels inside NVTX range
            python _ncu_target.py --layer-type ssm ...

        _ncu_target.py runs n_warmup passes (outside NVTX), then n_measure
        passes inside torch.cuda.nvtx.range_push("ncu_measure"). ncu captures
        only the latter via NVTX filtering, so no --launch-skip math is needed.

        Returns:
            dict with raw metric values + derived SM utilization / wave stats.
            Keys include: kernel_name, grid_size, n_waves, wave_efficiency_pct,
            sm_util_per_sm_pct, achieved_occupancy_pct, warps_active, ...
        """
        if metrics is None:
            metrics = NCU_METRICS_WAVE

        metrics_str = ",".join(metrics)
        ncu_cmd = [
            self.ncu_path,
            "--metrics", metrics_str,
            "--csv",
            "--target-processes", "all",
            "--clock-control", "none",
            # Profile only kernels inside the "ncu_measure" NVTX range
            # (set in _ncu_target.py after warmup passes complete)
            "--nvtx",
            "--nvtx-include", "ncu_measure",
        ]
        if extra_ncu_args:
            ncu_cmd.extend(extra_ncu_args)

        target_cmd = [
            self.python_path,
            str(self.target_script),
            "--layer-type", layer_type,
            "--model", model,
            "--sm-count", str(sm_count),
            "--seq-len", str(seq_len),
            "--batch-size", str(batch_size),
            "--context-len", str(context_len),
            "--n-warmup", str(n_warmup),
            "--n-measure", str(n_measure),
        ]

        full_cmd = ncu_cmd + target_cmd

        try:
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"ncu timed out after {timeout_s}s", "sm_count": sm_count}
        except FileNotFoundError as e:
            return {"error": str(e), "sm_count": sm_count}

        if proc.returncode != 0:
            # ncu errors (ERR_NVGPUCTRPERM, etc.) go to STDOUT, not stderr.
            # Include both so the caller can diagnose without re-running.
            ncu_out = proc.stdout[:1000]
            ncu_err = proc.stderr[:1000]
            return {
                "error": f"ncu exit code {proc.returncode}",
                "ncu_stdout": ncu_out,
                "stderr": ncu_err,
                "sm_count": sm_count,
                "seq_len": seq_len,
                "batch_size": batch_size,
            }

        kernels = _parse_ncu_csv(proc.stdout)
        if not kernels:
            return {
                "error": "No kernels captured in ncu output",
                "sm_count": sm_count,
                "stdout": proc.stdout[:500],
            }

        # Pick dominant kernel by active cycles (try both name variants)
        def _sort_key(k):
            v = _safe_float(k, "sm__cycles_active.sum", "sm__active_cycles_sum")
            return v if v is not None else 0.0

        dominant = max(kernels, key=_sort_key)
        derived = _derive_sm_util(dominant, sm_count, use_blackwell=self.use_blackwell)

        result = {
            "layer_type": layer_type,
            "model": model,
            "sm_count": sm_count,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_kernels_captured": len(kernels),
            **dominant,
            **derived,
        }
        return result

    def profile_sweep(
        self,
        layer_type: str,
        model: str,
        sm_counts: list[int],
        seq_lens: list[int],
        batch_sizes: list[int],
        metrics: list[str] = None,
        **kwargs,
    ) -> list[dict]:
        """Profile multiple configurations and return list of results.

        NOTE: ncu adds significant overhead (~10-100× slower than bare execution).
        Keep sm_counts / seq_lens small for ncu sweeps; use the regular sweep
        scripts for comprehensive latency measurement.
        """
        if metrics is None:
            metrics = NCU_METRICS_WAVE

        results = []
        total = len(sm_counts) * len(seq_lens) * len(batch_sizes)
        done = 0

        for sm_count in sm_counts:
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    done += 1
                    print(
                        f"  [{done}/{total}] ncu: {layer_type} sm={sm_count} "
                        f"seq={seq_len} bs={batch_size} ...",
                        flush=True,
                    )
                    row = self.profile(
                        layer_type=layer_type,
                        model=model,
                        sm_count=sm_count,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        metrics=metrics,
                        **kwargs,
                    )
                    results.append(row)
                    if "error" not in row:
                        sm_util = row.get("sm_util_per_sm_pct")
                        n_waves = row.get("n_waves")
                        wave_eff = row.get("wave_efficiency_pct")
                        occ = row.get("achieved_occupancy_pct")
                        print(
                            f"    sm_util={sm_util:.1f}%  "
                            f"waves={n_waves}  "
                            f"wave_eff={wave_eff:.1f}%  "
                            f"occupancy={occ if occ is not None else 'N/A'}"
                        )
                    else:
                        print(f"    ERROR: {row['error']}")

        return results
