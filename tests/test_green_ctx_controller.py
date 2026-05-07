"""
Tests for src/smctrl/green_ctx_controller.py

Test categories:
  Unit  — struct layout, library loading, preset logic (no GPU required)
  CUDA  — Green Context creation and SM restriction (GPU + driver 550+ required)
  Compat — public interface matches the original SMController contract

Run all:
    cd prefill-layer-alloc
    pytest tests/test_green_ctx_controller.py -v

Run without GPU:
    pytest tests/test_green_ctx_controller.py -v -m "not cuda"

Run verify only:
    pytest tests/test_green_ctx_controller.py -v -k verify
"""

import ctypes
import time

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.smctrl.green_ctx_controller import (
    SMController,
    _CUdevResource,
    _RESOURCE_PADDING_BYTES,
    _RESOURCE_UNION_BYTES,
    _load_driver_lib,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    return torch.cuda.is_available()


def _green_ctx_supported() -> bool:
    """Return True if this GPU + driver fully support Green Contexts.

    Probes the complete pipeline: resource query → split → desc → context → stream.
    cuDeviceGetDevResource alone succeeds on all GPUs; actual Green Context
    creation may fail (e.g. cuGreenCtxStreamCreate requires CU_STREAM_NON_BLOCKING).
    We use SMController itself as the probe since it exercises the full pipeline.
    """
    if not _gpu_available():
        return False
    try:
        n = torch.cuda.get_device_properties(0).multi_processor_count
        ctrl = SMController(preset_sm_counts=[max(1, n // 2), n])
        return ctrl.is_available()
    except Exception:
        return False


def _total_sm() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


# ---------------------------------------------------------------------------
# Mark definitions
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU required")
needs_green_ctx = pytest.mark.skipif(
    not _green_ctx_supported(),
    reason="Green Contexts not supported (data-center GPU + driver 550+ required)",
)


def _sm_restriction_effective() -> bool:
    """Return True if 25% SM allocation yields measurably fewer SMs than full.

    On A100: request 27 SM (25% of 108) → gets ~27. Ratio = 0.25 → restriction works.
    On RTX 5060 Ti: request 9 SM (25% of 36) → gets 16 (GPC-granular). Ratio = 0.44.
      A 256 MB BW-bound kernel saturates at 16/36 SMs → no observable slowdown.
    Threshold: actual_sm / total_sm < 0.40 indicates meaningful restriction.
    """
    if not _green_ctx_supported():
        return False
    n = _total_sm()
    quarter_sm = max(1, n // 4)
    try:
        ctrl = SMController(preset_sm_counts=[quarter_sm, n])
        ctrl.set_sm_count(quarter_sm)
        actual = ctrl._actual_sm_counts.get(ctrl._current_sm_count, ctrl._current_sm_count)
        return (actual / n) < 0.40
    except Exception:
        return False


needs_sm_restriction = pytest.mark.skipif(
    not _sm_restriction_effective(),
    reason=(
        "GPU GPC structure prevents fine-grained SM restriction at 25%: "
        "allocated SM / total_SM >= 0.40 (BW-bound kernel shows no measurable slowdown). "
        "Run on A100/H100 for meaningful SM isolation."
    ),
)


# ===========================================================================
# Unit tests — no GPU / driver required beyond struct math
# ===========================================================================

class TestStructLayout:
    def test_cudev_resource_size(self):
        """CUdevResource must be exactly 144 bytes to match cuda.h layout."""
        assert ctypes.sizeof(_CUdevResource) == 144

    def test_padding_constants(self):
        assert _RESOURCE_PADDING_BYTES == 92
        assert _RESOURCE_UNION_BYTES == 48

    def test_smcount_field_offset(self):
        """smCount must be at offset 96 = 4 (type) + 92 (padding)."""
        res = _CUdevResource()
        res._impl.smCount = 0xDEADBEEF
        raw = bytes(res)
        val = int.from_bytes(raw[96:100], "little")
        assert val == 0xDEADBEEF, f"smCount at wrong offset, got 0x{val:x}"


class TestLibraryLoading:
    def test_load_returns_none_or_cdll(self):
        lib = _load_driver_lib()
        assert lib is None or isinstance(lib, ctypes.CDLL)

    @needs_cuda
    def test_symbols_bound_when_lib_loaded(self):
        lib = _load_driver_lib()
        if lib is None:
            pytest.skip("libcuda.so not found")
        for sym in (
            "cuDeviceGetDevResource",
            "cuDevSmResourceSplitByCount",
            "cuDevResourceGenerateDesc",
            "cuGreenCtxCreate",
            "cuGreenCtxStreamCreate",
            "cuGreenCtxDestroy",
        ):
            assert hasattr(lib, sym), f"symbol {sym} missing after load"


# ===========================================================================
# Interface / contract tests — GPU required, Green Contexts optional
# ===========================================================================

class TestSMControllerInterface:
    """Verify public interface is present and type-correct regardless of backend."""

    @needs_cuda
    def test_construction_succeeds(self):
        ctrl = SMController()
        assert ctrl.total_sm_count > 0

    @needs_cuda
    def test_total_sm_matches_torch(self):
        ctrl = SMController()
        expected = torch.cuda.get_device_properties(0).multi_processor_count
        assert ctrl.total_sm_count == expected

    @needs_cuda
    def test_is_available_returns_bool(self):
        ctrl = SMController()
        assert isinstance(ctrl.is_available(), bool)

    @needs_cuda
    def test_get_backend_name_is_string(self):
        ctrl = SMController()
        name = ctrl.get_backend_name()
        assert isinstance(name, str)
        assert name in ("green_ctx", "none")

    @needs_cuda
    def test_set_sm_count_clamps(self):
        ctrl = SMController()
        ctrl.set_sm_count(0)          # clamp to 1
        ctrl.set_sm_count(10**9)      # clamp to total_sm_count

    @needs_cuda
    def test_set_sm_ratio_clamps(self):
        ctrl = SMController()
        ctrl.set_sm_ratio(0.0)   # clamp to 0.01
        ctrl.set_sm_ratio(2.0)   # clamp to 1.0

    @needs_cuda
    def test_reset_restores_full_sm(self):
        ctrl = SMController()
        ctrl.set_sm_count(1)
        ctrl.reset()
        assert ctrl._current_sm_count == ctrl.total_sm_count

    @needs_cuda
    def test_get_stream_returns_stream(self):
        ctrl = SMController()
        stream = ctrl.get_stream()
        assert isinstance(stream, torch.cuda.Stream)

    @needs_cuda
    def test_verify_sm_control_returns_bool(self):
        ctrl = SMController()
        result = ctrl.verify_sm_control(verbose=True)
        assert isinstance(result, bool)

    @needs_cuda
    def test_measure_reconfigure_latency_returns_dict(self):
        ctrl = SMController()
        result = ctrl.measure_reconfigure_latency_us(
            from_ratio=0.7, to_ratio=0.4, n_trials=10
        )
        expected_keys = {"mean_us", "p50_us", "p99_us", "min_us", "max_us", "std_us",
                         "n_trials", "from_ratio", "to_ratio"}
        assert expected_keys.issubset(result.keys())
        assert result["n_trials"] == 10
        assert result["from_ratio"] == pytest.approx(0.7)
        assert result["to_ratio"] == pytest.approx(0.4)


# ===========================================================================
# Preset logic tests — GPU required
# ===========================================================================

class TestPresetLogic:
    @needs_cuda
    def test_default_presets_cover_full_sm(self):
        ctrl = SMController()
        assert ctrl.total_sm_count in ctrl._sorted_presets

    @needs_green_ctx
    def test_custom_presets_respected(self):
        n = torch.cuda.get_device_properties(0).multi_processor_count
        custom = [max(1, n // 4), max(1, n // 2), n]
        ctrl = SMController(preset_sm_counts=custom)
        for c in custom:
            assert c in ctrl._contexts, f"preset {c} not in contexts: {list(ctrl._contexts)}"

    @needs_cuda
    def test_set_sm_count_snaps_to_nearest(self):
        n = torch.cuda.get_device_properties(0).multi_processor_count
        # Use values that are always valid: multiples of total
        p1 = max(1, n // 4)
        p2 = max(p1 + 1, n // 2)
        ctrl = SMController(preset_sm_counts=[p1, p2, n])
        # Snap to p1 (below midpoint between p1 and p2)
        ctrl.set_sm_count(p1)
        assert ctrl._current_sm_count == p1
        # Snap to n (above midpoint between p2 and n)
        ctrl.set_sm_count(n)
        assert ctrl._current_sm_count == n

    @needs_cuda
    def test_sorted_presets_are_sorted(self):
        ctrl = SMController()
        assert ctrl._sorted_presets == sorted(ctrl._sorted_presets)

    @needs_cuda
    def test_set_sm_count_then_get_stream_consistent(self):
        n = torch.cuda.get_device_properties(0).multi_processor_count
        ctrl = SMController(preset_sm_counts=[n // 2, n])
        ctrl.set_sm_count(n // 2)
        s1 = ctrl.get_stream()
        ctrl.set_sm_count(n)
        s2 = ctrl.get_stream()
        # Streams should differ (different Green Contexts) when Green Ctx is active
        if ctrl.is_available():
            assert s1 is not s2, "Different SM counts should yield different streams"


# ===========================================================================
# Green Context creation tests — data-center GPU required
# ===========================================================================

class TestGreenContextCreation:
    @needs_green_ctx
    def test_contexts_created_for_all_presets(self):
        ctrl = SMController()
        assert len(ctrl._contexts) > 0
        for sm_count in ctrl._sorted_presets:
            assert sm_count in ctrl._contexts, f"No context for sm_count={sm_count}"

    @needs_green_ctx
    def test_streams_are_external_streams(self):
        """Non-full-SM contexts must use ExternalStream; full-SM may use regular Stream."""
        ctrl = SMController()
        for sm_count, (ctx_ptr, stream) in ctrl._contexts.items():
            if ctx_ptr != 0:  # ctx_ptr=0 is the fallback (full SM regular stream)
                assert isinstance(stream, torch.cuda.ExternalStream), (
                    f"sm_count={sm_count}: expected ExternalStream for Green Context, "
                    f"got {type(stream)}"
                )

    @needs_green_ctx
    def test_context_ptr_nonzero_for_non_full(self):
        ctrl = SMController()
        for sm_count, (ctx_ptr, _) in ctrl._contexts.items():
            if sm_count < ctrl.total_sm_count:
                assert ctx_ptr != 0, (
                    f"sm_count={sm_count}: Green Context pointer is NULL"
                )

    @needs_green_ctx
    def test_custom_a100_sweep_steps(self):
        """Verify all A100 Green Context preset sweep steps can be materialized."""
        # Green Context preset boundaries (hardware.yaml: a100_80gb, a100_80gb_pcie)
        a100_steps = [14, 27, 40, 54, 68, 81, 94, 108]
        n = _total_sm()
        steps = sorted(set(min(s, n) for s in a100_steps))
        ctrl = SMController(preset_sm_counts=steps)
        assert ctrl.is_available(), (
            "Green Contexts unavailable despite _green_ctx_supported() = True"
        )
        for sm in steps:
            assert sm in ctrl._contexts, (
                f"Context missing for sm={sm} (total_sm={n})"
            )

    @needs_green_ctx
    def test_kernel_runs_in_green_ctx_stream(self):
        """Kernel launched in Green Context stream completes without error."""
        ctrl = SMController()
        ctrl.set_sm_ratio(0.5)
        stream = ctrl.get_stream()
        x = torch.randn(1024, 1024, device="cuda")
        with torch.cuda.stream(stream):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
        assert y.shape == (1024, 1024)


# ===========================================================================
# SM restriction effectiveness — data-center GPU required
# ===========================================================================

class TestSMRestrictionEffectiveness:
    """verify_sm_control logic broken out into fine-grained assertions."""

    @needs_sm_restriction
    def test_verify_sm_control_passes(self):
        """Green Contexts must produce measurable slowdown at 25% SM on A100/H100."""
        ctrl = SMController()
        assert ctrl.verify_sm_control(verbose=True), (
            "verify_sm_control failed: 25% SM run is NOT slower than full SM run.\n"
            "Expected slowdown ratio >= 1.20."
        )

    @needs_sm_restriction
    def test_full_sm_faster_than_quarter_sm(self):
        """Full SM latency must be lower than quarter-SM latency."""
        n = _total_sm()
        ctrl = SMController()

        size = 32 * 1024 * 1024
        x = torch.ones(size, device="cuda", dtype=torch.float32)
        y = torch.ones(size, device="cuda", dtype=torch.float32)
        n_iters = 10

        def _lat(sm_count):
            ctrl.set_sm_count(sm_count)
            stream = ctrl.get_stream()
            evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n_iters)]
            with torch.cuda.stream(stream):
                for s, e in evs:
                    s.record(stream)
                    torch.add(x, y, out=x)
                    e.record(stream)
            torch.cuda.synchronize()
            return float(np.median([s.elapsed_time(e) for s, e in evs]))

        full_lat = _lat(n)
        quarter_lat = _lat(max(1, n // 4))
        ctrl.reset()

        ratio = quarter_lat / full_lat
        assert ratio >= 1.10, (
            f"Expected quarter-SM to be ≥1.10× slower than full SM.\n"
            f"full={full_lat:.3f}ms  quarter={quarter_lat:.3f}ms  ratio={ratio:.2f}x"
        )

    @needs_green_ctx
    def test_latency_increases_monotonically_with_fewer_sm(self):
        """Fewer SMs should yield higher latency for a BW-bound kernel."""
        n = _total_sm()
        sm_counts = sorted([max(1, round(n * r)) for r in (0.25, 0.5, 1.0)])
        ctrl = SMController(preset_sm_counts=sm_counts)

        size = 32 * 1024 * 1024
        x = torch.ones(size, device="cuda", dtype=torch.float32)
        y = torch.ones(size, device="cuda", dtype=torch.float32)
        n_iters = 10

        latencies = {}
        for sm in sm_counts:
            ctrl.set_sm_count(sm)
            stream = ctrl.get_stream()
            evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n_iters)]
            with torch.cuda.stream(stream):
                for s, e in evs:
                    s.record(stream)
                    torch.add(x, y, out=x)
                    e.record(stream)
            torch.cuda.synchronize()
            latencies[sm] = float(np.median([s.elapsed_time(e) for s, e in evs]))

        ctrl.reset()

        sm_sorted = sorted(latencies.keys())
        lats = [latencies[sm] for sm in sm_sorted]
        for i in range(1, len(lats)):
            assert lats[i] <= lats[i - 1] * 1.05, (
                f"Expected latency to decrease with more SMs.\n"
                f"sm={sm_sorted[i]} ({lats[i]:.3f}ms) not faster than "
                f"sm={sm_sorted[i-1]} ({lats[i-1]:.3f}ms)"
            )


# ===========================================================================
# Overhead / latency tests — data-center GPU required
# ===========================================================================

class TestReconfigurationOverhead:
    @needs_green_ctx
    def test_stream_switch_is_sub_microsecond(self):
        """Green Context stream switch (no sync) should be < 5 μs CPU-side."""
        ctrl = SMController()
        n_iters = 100

        samples = []
        for _ in range(n_iters):
            ctrl.set_sm_ratio(0.7)
            t0 = time.perf_counter_ns()
            ctrl.set_sm_ratio(0.4)
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1_000.0)

        median_us = float(np.median(samples))
        assert median_us < 5.0, (
            f"Stream switch median overhead {median_us:.2f} μs exceeds 5 μs threshold.\n"
            "Green Context stream switch should be a pure Python dict lookup."
        )

    @needs_green_ctx
    def test_measure_reconfigure_latency_all_positive(self):
        ctrl = SMController()
        result = ctrl.measure_reconfigure_latency_us(
            from_ratio=0.7, to_ratio=0.4, n_trials=50
        )
        assert result["mean_us"] >= 0
        assert result["p99_us"] >= result["p50_us"]
        assert result["min_us"] >= 0

    @needs_green_ctx
    @pytest.mark.parametrize("from_r,to_r,label", [
        (0.7, 0.4, "ssm→attn"),
        (0.4, 0.7, "attn→ssm"),
        (1.0, 0.7, "full→ssm"),
        (0.7, 1.0, "ssm→full"),
    ])
    def test_transition_pairs(self, from_r, to_r, label):
        ctrl = SMController()
        result = ctrl.measure_reconfigure_latency_us(
            from_ratio=from_r, to_ratio=to_r, n_trials=30
        )
        assert result["mean_us"] >= 0, f"Negative latency for {label}"
        print(
            f"\n  [{label}] mean={result['mean_us']:.2f}μs  "
            f"p50={result['p50_us']:.2f}μs  p99={result['p99_us']:.2f}μs"
        )


# ===========================================================================
# Backend compatibility sanity check
# ===========================================================================

class TestBackendCompat:
    """Verify SMController from __init__.py resolves to green_ctx_controller."""

    @needs_cuda
    def test_init_exports_green_ctx_controller(self):
        from src.smctrl import SMController as ExportedCtrl
        from src.smctrl.green_ctx_controller import SMController as GreenCtrl
        assert ExportedCtrl is GreenCtrl, (
            "src/smctrl/__init__.py does not export green_ctx_controller.SMController"
        )

    @needs_cuda
    def test_backend_name_not_libsmctrl(self):
        ctrl = SMController()
        assert ctrl.get_backend_name() != "libsmctrl", (
            "Backend is still 'libsmctrl' — __init__.py may not have been updated"
        )
