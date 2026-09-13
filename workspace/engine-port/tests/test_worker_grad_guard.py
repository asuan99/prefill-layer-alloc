"""Gate: a true-dual role-worker task must run with autograd OFF.

THE DEFECT (GPU job 907959, both PDMUX_TRUE_DUAL_WORKER=1 boots).
`event_loop_pdmux` is decorated `@torch.inference_mode()` and that guard is
THREAD-LOCAL.  The legacy loop calls `run_batch` on the decorated thread; the
true-dual loop hands it to `RoleWorkerThread`, a thread the decorator never
touched.  So the model forward ran with `torch.is_grad_enabled() == True` in
one arm and `False` in the other -- an arm asymmetry in a CORRECTNESS gate.

WHY IT BIT ONLY NOW.  SGLang blocks autograd at the parameter level nearly
everywhere (weights created `requires_grad=False`; `RMSNorm.forward_cuda` uses
`self.weight.data`).  The exception is `Mixer2RMSNormGated.forward_native`,
whose last line multiplies a bare `nn.Parameter`, and that branch is taken only
when `n_groups != 1` -- true on NemotronH (`mamba_num_groups=8`), false on
Zamba2-2.7B / Falcon-H1 / Granite-4.  With grad on, the multiply builds
`MulBackward0`, which retains its input; a 56-layer split prefill accumulates.
Both true-dual boots of job 907959 died with `torch.OutOfMemoryError` on the
14-seq / 10,125-token split prefill that both legacy boots completed.

WHAT IS PINNED HERE.  The tests drive the REAL `RoleWorkerThread` through the
REAL `_activate_role_context` and observe grad state from inside the worker
thread.  `test_mutation_*` loads a copy of the installed mixin with the guard
DELETED and requires the same observation to flip -- a repair whose test still
passes on the un-repaired code proves nothing (lesson 53).

NOT PINNED HERE: that the repair fixes the OOM.  That is a GPU measurement and
is pre-registered separately; nothing in this file licenses a memory or speed
claim.
"""

import importlib.util
import os
import re
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# test_profile_controller.py installs src/multiplex/profile.py as
# sys.modules["profile"], shadowing the stdlib module and breaking the sglang
# import chain below (same workaround as test_trace_force_prefill.py).
sys.modules.pop("profile", None)

import torch  # noqa: E402

from sglang.srt.multiplex import multiplexing_mixin as mixin_module  # noqa: E402
from sglang.srt.multiplex.dual_worker import (  # noqa: E402
    ExecutionContext,
    TrueDualWorkerRuntime,
    WorkerRole,
)

MIXIN_SOURCE = Path(mixin_module.__file__)
SM_COUNTS = [(108, 0), (64, 44), (0, 108)]


def _noop_cuda_guards():
    """Patch the two CUDA context managers `_activate_role_context` enters.

    Only these two are faked.  The grad guard under test, `set_pdmux_status`
    and the worker thread itself are the real code.
    """
    from contextlib import nullcontext

    return (
        mock.patch.object(torch.cuda, "device", lambda *_a, **_k: nullcontext()),
        mock.patch.object(torch.cuda, "stream", lambda *_a, **_k: nullcontext()),
    )


class FakeScheduler:
    """The three attributes `_activate_role_context` reads."""

    def __init__(self, module, guard=None, mem_telemetry=True):
        self.gpu_id = 0
        self.pdmux_worker_grad_guard = (
            module.resolve_worker_grad_guard(guard)
            if guard is not None
            else module.resolve_worker_grad_guard()
        )
        self.pdmux_mem_telemetry = mem_telemetry
        self._r2_worker_guard = {
            "prefill_worker_grad_enabled": None,
            "prefill_worker_inference_mode": None,
            "decode_worker_grad_enabled": None,
            "decode_worker_inference_mode": None,
        }
        self._activate_role_context = module.SchedulerMultiplexMixin.\
            _activate_role_context.__get__(self)
        self._r2_record_worker_guard = module.SchedulerMultiplexMixin.\
            _r2_record_worker_guard.__get__(self)


def _probe(_context):
    """Run on the worker thread; reports what autograd actually saw.

    `weight` is a bare `nn.Parameter` multiplied by an activation, i.e. the
    shape of `mixer2_rms_norm_gated.py:97`.  `graph_built` is the property that
    decides whether the activation is retained.
    """
    weight = torch.nn.Parameter(torch.ones(4))
    x = torch.ones(4)
    y = weight * x
    return {
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "tensor_is_inference": x.is_inference(),
        "graph_built": y.grad_fn is not None,
        "thread": threading.current_thread().name,
    }


def _run_on_worker(module, guard=None, role=WorkerRole.PREFILL,
                   mem_telemetry=True, callback=_probe):
    sched = FakeScheduler(module, guard=guard, mem_telemetry=mem_telemetry)
    runtime = TrueDualWorkerRuntime(SM_COUNTS, sched._activate_role_context)
    dev, strm = _noop_cuda_guards()
    dev.start()
    strm.start()
    try:
        runtime.select_partition(1)
        future = runtime.submit(role, callback, stream=object())
        result = future.result(timeout=20)
    finally:
        runtime.close()
        strm.stop()
        dev.stop()
    return sched, result


def _load_mutant(text, name):
    """Import a modified copy of the installed mixin under its own name."""
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"pdmux_mutant_{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"pdmux_mutant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WorkerGradGuardTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("PDMUX_WORKER_GRAD_GUARD", None)

    # ---- the repair -------------------------------------------------

    def test_worker_task_runs_with_autograd_off(self):
        _, seen = _run_on_worker(mixin_module)
        self.assertNotEqual(
            seen["thread"], threading.current_thread().name,
            "the probe did not actually run on a worker thread",
        )
        self.assertFalse(seen["grad_enabled"], f"[{MIXIN_SOURCE}]")
        self.assertTrue(seen["inference_mode"], f"[{MIXIN_SOURCE}]")
        self.assertTrue(seen["tensor_is_inference"], f"[{MIXIN_SOURCE}]")
        self.assertFalse(
            seen["graph_built"],
            "a bare nn.Parameter multiply still built an autograd graph on the "
            f"worker thread: activations are retained [{MIXIN_SOURCE}]",
        )

    def test_both_roles_get_the_guard(self):
        for role in (WorkerRole.PREFILL, WorkerRole.DECODE):
            with self.subTest(role=role):
                _, seen = _run_on_worker(mixin_module, role=role)
                self.assertFalse(seen["grad_enabled"])
                self.assertTrue(seen["inference_mode"])

    def test_two_role_threads_hold_the_guard_at_once(self):
        """InferenceMode is thread-local; concurrent entry must be legal.

        Both workers are held inside the guard simultaneously by a barrier, so
        a shared-state guard object or a global guard would deadlock or raise.
        """
        sched = FakeScheduler(mixin_module)
        runtime = TrueDualWorkerRuntime(SM_COUNTS, sched._activate_role_context)
        barrier = threading.Barrier(3, timeout=20)

        def both(_context):
            state = (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
            barrier.wait()
            return state

        dev, strm = _noop_cuda_guards()
        dev.start()
        strm.start()
        try:
            runtime.select_partition(1)
            f1 = runtime.submit(WorkerRole.PREFILL, both, stream=object())
            f2 = runtime.submit(WorkerRole.DECODE, both, stream=object())
            barrier.wait()
            self.assertEqual(f1.result(timeout=20), (False, True))
            self.assertEqual(f2.result(timeout=20), (False, True))
        finally:
            runtime.close()
            strm.stop()
            dev.stop()

    def test_realised_guard_state_is_recorded_from_inside_the_worker(self):
        sched, _ = _run_on_worker(mixin_module, role=WorkerRole.DECODE)
        self.assertIs(sched._r2_worker_guard["decode_worker_grad_enabled"], False)
        self.assertIs(sched._r2_worker_guard["decode_worker_inference_mode"], True)
        # the role that did not run stays unreported rather than assumed
        self.assertIsNone(sched._r2_worker_guard["prefill_worker_grad_enabled"])

    def test_realised_readout_is_off_with_the_telemetry_flag_off(self):
        sched, seen = _run_on_worker(mixin_module, mem_telemetry=False)
        self.assertFalse(seen["grad_enabled"], "the guard itself must not be gated")
        self.assertIsNone(sched._r2_worker_guard["prefill_worker_grad_enabled"])

    # ---- the knob ---------------------------------------------------

    def test_default_is_inference_mode(self):
        self.assertEqual(
            mixin_module.resolve_worker_grad_guard(), "inference_mode"
        )
        self.assertEqual(mixin_module.DEFAULT_WORKER_GRAD_GUARD, "inference_mode")

    def test_env_override_is_honoured_and_validated(self):
        os.environ["PDMUX_WORKER_GRAD_GUARD"] = "no_grad"
        self.assertEqual(mixin_module.resolve_worker_grad_guard(), "no_grad")
        os.environ["PDMUX_WORKER_GRAD_GUARD"] = "NoNe"
        self.assertEqual(mixin_module.resolve_worker_grad_guard(), "none")
        os.environ["PDMUX_WORKER_GRAD_GUARD"] = "off"
        with self.assertRaises(ValueError):
            mixin_module.resolve_worker_grad_guard()

    def test_no_grad_setting_disables_grad_but_is_not_inference_mode(self):
        """The alternative that was NOT chosen, and the reason it was not.

        `no_grad` also stops the graph, so it would also stop the retention --
        but the worker's tensors stay ordinary while legacy's are inference
        tensors, so the two arms would still run the model under different
        tensor semantics.  Pinned so the distinction cannot be lost.
        """
        _, seen = _run_on_worker(mixin_module, guard="no_grad")
        self.assertFalse(seen["grad_enabled"])
        self.assertFalse(seen["inference_mode"])
        self.assertFalse(seen["tensor_is_inference"])
        self.assertFalse(seen["graph_built"])

    def test_none_setting_reproduces_the_defect(self):
        """Negative control through the knob (weaker than the source mutation
        below, but it pins that `none` really is the pre-repair behaviour)."""
        _, seen = _run_on_worker(mixin_module, guard="none")
        self.assertTrue(seen["grad_enabled"])
        self.assertFalse(seen["inference_mode"])
        self.assertTrue(
            seen["graph_built"],
            "without the guard the parameter multiply must build a graph; if "
            "it does not, this test no longer discriminates",
        )

    # ---- mutation control (lesson 53) --------------------------------

    def test_mutation_removing_the_guard_makes_the_gate_fail(self):
        text = MIXIN_SOURCE.read_text()
        mutated, n = re.subn(
            r"with guard\(\), torch\.cuda\.device", "with torch.cuda.device", text
        )
        self.assertEqual(
            n, 1,
            "the guard is not where this mutation expects it; the mutation "
            "control has stopped covering the repair",
        )
        module = _load_mutant(mutated, "noguard")
        _, seen = _run_on_worker(module)
        self.assertTrue(
            seen["grad_enabled"],
            "the un-repaired mixin still reported grad disabled: the "
            "observation in test_worker_task_runs_with_autograd_off does not "
            "depend on the repair",
        )
        self.assertFalse(seen["inference_mode"])
        self.assertTrue(seen["graph_built"])

    def test_mutation_defaulting_the_knob_to_none_makes_the_gate_fail(self):
        text = MIXIN_SOURCE.read_text()
        mutated, n = re.subn(
            r'DEFAULT_WORKER_GRAD_GUARD = "inference_mode"',
            'DEFAULT_WORKER_GRAD_GUARD = "none"',
            text,
        )
        self.assertEqual(n, 1)
        module = _load_mutant(mutated, "defaultnone")
        self.assertEqual(module.resolve_worker_grad_guard(), "none")
        _, seen = _run_on_worker(module)
        self.assertTrue(seen["grad_enabled"])

    def test_unmutated_copy_passes(self):
        """Control for the control: the mutation harness itself must be sound."""
        module = _load_mutant(MIXIN_SOURCE.read_text(), "control")
        _, seen = _run_on_worker(module)
        self.assertFalse(seen["grad_enabled"])
        self.assertTrue(seen["inference_mode"])

    # ---- installed-tree provenance -----------------------------------

    def test_installed_runtime_carries_the_repair(self):
        """A stale dev tree must fail here rather than at the GPU."""
        text = MIXIN_SOURCE.read_text()
        self.assertIn("with guard(), torch.cuda.device", text)
        self.assertIn('DEFAULT_WORKER_GRAD_GUARD = "inference_mode"', text)


if __name__ == "__main__":
    unittest.main()
