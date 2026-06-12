"""
test_dispatch.py — unit test for the B-1 fix (prefill kernel dispatch).

The B-1 bug: the v1 concurrent runner built an SSM prefill kernel for EVERY
config, so an "attn" config silently measured an SSM kernel. v2 routes all
prefill construction through kernels.build_prefill_fn; this test proves an "attn"
request runs the attention builder (not the SSM builder) and vice-versa.

Two layers of testing:
  * torch-free routing test (always runs): monkeypatches the leaf builders and
    asserts build_prefill_fn calls the RIGHT one per layer type. This is the
    regression guard for B-1 and needs no GPU.
  * CUDA kernel-identity test (skipped without CUDA): actually builds the kernels
    and checks meta["kernel"].

Run: `python experiments/e4_concurrent/test_dispatch.py`  (also pytest-compatible)
"""

from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
_WORKSPACE = os.path.abspath(os.path.join(_here, "..", "..", ".."))
for _p in (_WORKSPACE, _CHAR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.common import kernels as K

_calls: list[str] = []


def _stub(name):
    def f(*a, **k):
        _calls.append(name)
        return (lambda: None, 1, 1, {"kernel": name, "n_blocks_per_call": 0})
    return f


def _patch():
    orig = (K.build_ssm_scan_fn, K.build_attn_fn, K.build_in_proj_fn, K.build_out_proj_fn)
    K.build_ssm_scan_fn = _stub("ssm_scan")
    K.build_attn_fn = _stub("attn")
    K.build_in_proj_fn = _stub("in_proj")
    K.build_out_proj_fn = _stub("out_proj")
    return orig


def _unpatch(orig):
    (K.build_ssm_scan_fn, K.build_attn_fn,
     K.build_in_proj_fn, K.build_out_proj_fn) = orig


def test_attn_request_runs_attn_not_ssm():
    orig = _patch()
    try:
        _calls.clear()
        _, _, _, meta = K.build_prefill_fn("attn", {}, batch=1, tokens=256)
        assert _calls == ["attn"], f"attn config ran {_calls} (B-1 regression!)"
        assert meta["kernel"] == "attn"
    finally:
        _unpatch(orig)


def test_ssm_request_runs_ssm():
    orig = _patch()
    try:
        _calls.clear()
        _, _, _, meta = K.build_prefill_fn("ssm", {}, batch=1, tokens=256)
        assert _calls == ["ssm_scan"], f"ssm config ran {_calls}"
        assert meta["kernel"] == "ssm_scan"
    finally:
        _unpatch(orig)


def test_ssm_full_runs_three_components():
    orig = _patch()
    try:
        _calls.clear()
        _, _, _, meta = K.build_prefill_fn("ssm_full", {}, batch=1, tokens=256)
        assert _calls == ["in_proj", "ssm_scan", "out_proj"], _calls
        assert meta["kernel"] == "ssm_full"
    finally:
        _unpatch(orig)


def test_invalid_layer_type_raises():
    try:
        K.build_prefill_fn("not_a_type", {}, batch=1, tokens=256)
    except ValueError:
        return
    raise AssertionError("invalid layer_type must raise ValueError")


def test_cuda_kernel_identity():
    """Real build on GPU: attn config actually produces the attn kernel."""
    if not K._HAS_TORCH:
        print("  [skip] torch unavailable")
        return
    import torch
    if not torch.cuda.is_available():
        print("  [skip] CUDA unavailable")
        return
    cfg = K.load_layer_cfg("falcon_h1_1.5b")
    _, _, _, m_attn = K.build_prefill_fn("attn", cfg, 1, 256)
    _, _, _, m_ssm = K.build_prefill_fn("ssm", cfg, 1, 256)
    assert m_attn["kernel"] == "attn"
    assert m_ssm["kernel"] == "ssm_scan"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
