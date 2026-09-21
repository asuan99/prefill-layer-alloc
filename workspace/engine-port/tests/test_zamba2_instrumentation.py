"""Gate for the Zamba2 per-layer-type timing instrumentation (2026-08-04 rework).

WHY THIS EXISTS.  Job 858811 (`results/r0c/decode_knee_vs_ctx.*`) was audited on
2026-08-04 and three instrumentation defects were confirmed:

  1. BUCKET ASYMMETRY -- `_zt("attn", ...)` wrapped only the RadixAttention core
     while `_zt("mamba", ...)` wrapped the whole MambaMixer2 call.  qkv_proj,
     o_proj, the MLP, both layernorms and the hybrid `.linear` were in NO bucket,
     so the two reported buckets were neither comparable nor a decomposition of
     the forward.
  2. RUNNING-MEAN ACCUMULATOR -- `_zt_acc` was reset only on mode change, never
     at emit, and the harness took `tail -1` (largest n), i.e. the sample most
     diluted by cold start.  `full` was always the first arm, so the bias sat in
     the denominator of every ratio.
  3. A physical-invariant violation (mamba decode is O(1) in ctx) that follows
     from 1+2.

These tests pin the properties of the FIX.  They are static/AST checks plus
CPU-only behavioural checks of the accumulator arithmetic -- the timing itself
needs a GPU and is covered by the smoke sbatch, not here.

The single most load-bearing check is `test_bucket_attribution_map`: it asserts
WHICH span each compute statement lives in.  Defect 1 is exactly a violation of
that map, and it went unnoticed for months because nothing asserted it.
"""

import ast
import os
import sys
import unittest
from pathlib import Path

# See test_trace_force_prefill.py / test_sticky_partition.py: test_profile_controller.py
# installs src/multiplex/profile.py as sys.modules["profile"], shadowing the stdlib
# module and breaking the sglang import chain used below.
sys.modules.pop("profile", None)

TRACK_ROOT = Path(__file__).resolve().parents[1]
MIRROR = TRACK_ROOT / "src" / "models" / "zamba2.py"
RUNTIME = (
    Path(os.environ.get("SGLANG_ENGINE_DEV")
         or os.path.join(os.environ.get("PDMUX_ROOT", str(TRACK_ROOT.parents[2])),
                          "sglang_engine_dev", "python"))
    / "sglang" / "srt" / "models" / "zamba2.py"
)

# The file under test is the INSTALLED runtime when it exists (so this also
# checks that the tracked source was actually installed), else the mirror.
UNDER_TEST = RUNTIME if RUNTIME.is_file() else MIRROR
SRC = UNDER_TEST.read_text()
TREE = ast.parse(SRC)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    if isinstance(node, ast.Subscript):  # self.dpa_list[block_idx] -> "self.dpa_list[]"
        base = _dotted(node.value)
        return None if base is None else f"{base}[]"
    return None


def _find_method_fn(fn_name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return node
    raise AssertionError(f"module-level {fn_name}() not found in {UNDER_TEST}")


def _find_method(cls_name, fn_name):
    for cls in ast.walk(TREE):
        if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == fn_name:
                    return fn
    raise AssertionError(f"{cls_name}.{fn_name} not found in {UNDER_TEST}")


def _span_trace(fn_node):
    """Replay a method's compute events in source order against the open span.

    Returns (events, unbucketed) where events = [(name, bucket_or_None)].  These
    methods are straight-line (no loops / no reordering), so source order is
    execution order and a linear replay is exact.
    """
    raw = []
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name is not None:
                raw.append((node.lineno, node.col_offset, name, node))
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            raw.append((node.lineno, node.col_offset, "<binop>", node))
    raw.sort(key=lambda t: (t[0], t[1]))

    open_bucket = None
    events, unbucketed = [], []
    for _ln, _col, name, node in raw:
        if name == "_zt_begin":
            assert open_bucket is None, f"nested _zt_begin in {fn_node.name}"
            arg = node.args[0]
            open_bucket = arg.value if isinstance(arg, ast.Constant) else "?"
            continue
        if name == "_zt_end":
            open_bucket = None
            continue
        events.append((name, open_bucket))
        if open_bucket is None:
            unbucketed.append(name)
    return events, unbucketed


class Zamba2BucketStructureTest(unittest.TestCase):
    """Static gate on WHAT is inside WHICH bucket (defect 1)."""

    # Delegating calls: the callee is itself instrumented, so the caller does not
    # need to wrap it (wrapping it would double-count).
    DELEGATES = {"self.shared_transformer", "self.mamba_decoder"}

    EXPECTED = {
        ("Zamba2AttentionDecoderLayer", "forward"): [
            ("torch.concatenate", "other"),
            ("self.input_layernorm", "other"),
            ("self.self_attn", "attn"),          # WHOLE attn module: qkv+core+o_proj
            ("self.pre_ff_layernorm", "other"),
            ("self.feed_forward", "mlp"),        # MLP is its own bucket, not "attn"
        ],
        ("Zamba2HybridLayer", "forward"): [
            ("self.shared_transformer", None),   # delegate
            ("self.linear", "other"),            # hybrid projection: not attn, not mamba
            ("self.mamba_decoder", None),        # delegate
        ],
        ("Zamba2MambaDecoderLayer", "forward"): [
            ("<binop>", "other"),                # + transformer_hidden_states
            ("self.input_layernorm", "other"),
            ("torch.empty_like", "other"),
            ("attn_backend.linear_attn_backend.forward", "mamba"),  # unchanged defn
            ("<binop>", "other"),                # + residual
        ],
    }

    def test_bucket_attribution_map(self):
        for (cls, fn), expected in self.EXPECTED.items():
            with self.subTest(method=f"{cls}.{fn}"):
                events, _ = _span_trace(_find_method(cls, fn))
                self.assertEqual(expected, events)

    def test_no_compute_outside_a_bucket(self):
        """Static mirror of the runtime closure gate: nothing in a layer forward
        may sit outside every span (that is precisely defect 1)."""
        for cls, fn in self.EXPECTED:
            with self.subTest(method=f"{cls}.{fn}"):
                _events, unbucketed = _span_trace(_find_method(cls, fn))
                leaked = [n for n in unbucketed if n not in self.DELEGATES]
                self.assertEqual([], leaked)

    def test_attn_core_is_diagnostic_only(self):
        """The pre-fix "attn" definition is KEPT (so old and new runs can be
        compared) but it is NESTED inside the accounted "attn" span, hence it must
        be a _ZT_DIAG bucket and excluded from the closure sum."""
        events, _ = _span_trace(_find_method("Zamba2Attention", "forward"))
        self.assertIn(("self.dpa_list[].forward", "attn_core"), events)
        # the projections around it are deliberately NOT in attn_core -- they are
        # covered by the caller's "attn" span (that is the defect-1 fix).
        self.assertIn(("self.qkv_proj", None), events)
        self.assertIn(("self.o_proj", None), events)
        buckets = ast.literal_eval(
            SRC.split("_ZT_BUCKETS = ", 1)[1].split("\n", 1)[0].strip()
        )
        diag = ast.literal_eval(SRC.split("_ZT_DIAG = ", 1)[1].split("\n", 1)[0].strip())
        self.assertEqual({"attn", "mlp", "other", "mamba"}, set(buckets))
        self.assertEqual({"attn_core"}, set(diag))


class Zamba2ClosureGateTest(unittest.TestCase):
    """The closure gate is the deliverable: it is what would have caught defect 1."""

    def test_loop_is_timed_by_an_independent_event_pair(self):
        fn = _find_method("Zamba2Model", "forward")
        src = ast.get_source_segment(SRC, fn)
        self.assertIn("_zt_tot_s.record()", src)
        self.assertIn("_zt_tot_e.record()", src)
        self.assertIn("_zt_tot_s.elapsed_time(_zt_tot_e)", src)

    def test_closure_denominator_is_independent_of_the_buckets(self):
        """A timeline partition (segments that tile the interval) would make the
        ratio identically 1 and the gate vacuous.  The denominator must come from
        the loop's own event pair and never from a sum of bucket spans."""
        upd = ast.get_source_segment(SRC, _find_method_fn("_zt_block_update"))
        self.assertIn('blk["total"] += fwd_total', upd)
        self.assertNotIn('blk["total"] += sum', upd)
        fn_src = ast.get_source_segment(SRC, _find_method("Zamba2Model", "forward"))
        # the only writer of the block total is the loop-pair elapsed time
        self.assertIn("_fwd_total = _zt_tot_s.elapsed_time(_zt_tot_e)", fn_src)
        self.assertIn("_zt_block_update(_blk, _fwd, _fwd_total, (_nseq, _ntok))", fn_src)
        # non-identity is asserted numerically in
        # Zamba2InstrumentationRuntimeTest.test_closure_flags_unbucketed_work

    def test_failure_is_reported(self):
        self.assertIn("ZBLT_CLOSURE_FAIL", SRC)
        self.assertIn("SGLANG_ZAMBA_CLOSURE_MIN", SRC)

    def test_emit_tag_changed_so_old_parsers_fail_loudly(self):
        """The "attn" bucket now means something else; a pre-2026-08-04 grep for
        "ZBLT mode=" must MISS rather than silently read a different quantity."""
        self.assertIn('"ZBPT2" if _pk else "ZBLT2"', SRC)
        self.assertNotIn('"ZBPT" if _pk else "ZBLT"', SRC)


class Zamba2AccumulatorTest(unittest.TestCase):
    """Defect 2: block-scoped accumulation, legacy running mean preserved."""

    def test_block_is_reset_at_emit_and_legacy_running_mean_is_not(self):
        fn_src = ast.get_source_segment(SRC, _find_method("Zamba2Model", "forward"))
        emit = fn_src.split("if self._zt_n % _every == 0:", 1)[1]
        self.assertIn("self._zt_blk = _zt_new_block()", emit)   # block reset at emit
        self.assertNotIn("self._zt_acc = {", emit)              # running mean survives
        # legacy fields still emitted (comparability with pre-fix reports)
        for field in ("attn_total=", "mamba_total=", "per-attn(", "per-mamba("):
            self.assertIn(field, emit)
        # new block fields
        for field in ("blk=", "blk_n=", "dirty=", "closure=", "b_per_mamba="):
            self.assertIn(field, emit)

    def test_running_mean_reset_only_on_mode_change(self):
        fn_src = ast.get_source_segment(SRC, _find_method("Zamba2Model", "forward"))
        head = fn_src.split("if self._zt_n % _every == 0:", 1)[0]
        self.assertIn('if getattr(self, "_zt_mode", None) != _mode:', head)


class Zamba2InstrumentationRuntimeTest(unittest.TestCase):
    """CPU-only behavioural checks of the helpers (no GPU, no model build)."""

    @classmethod
    def setUpClass(cls):
        try:
            import sglang.srt.models.zamba2 as z
        except Exception as exc:  # pragma: no cover - env without sglang deps
            raise unittest.SkipTest(f"sglang zamba2 import failed: {exc}")
        cls.z = z

    def setUp(self):
        self.z._ZT["on"] = False
        self.z._zt_reset()

    def tearDown(self):
        self.z._ZT["on"] = False
        self.z._zt_reset()

    def test_off_path_creates_no_cuda_events(self):
        """Gate: instrumentation OFF must short-circuit before touching CUDA."""
        import torch
        from unittest import mock

        with mock.patch.object(torch.cuda, "Event", side_effect=AssertionError("event!")):
            self.assertIsNone(self.z._zt_begin("attn"))
            self.z._zt_end(None)
            self.assertEqual(7, self.z._zt("attn", lambda: 7))
        self.assertEqual(0, self.z._ZT["pool_n"])
        for spans in self.z._ZT["spans"].values():
            self.assertEqual([], spans)

    def test_on_path_records_and_recycles_events(self):
        import torch
        from unittest import mock

        class FakeEvent:
            def __init__(self, enable_timing=False):
                self.n = 0

            def record(self):
                self.n += 1

            def elapsed_time(self, other):
                return 1.0

        with mock.patch.object(torch.cuda, "Event", FakeEvent):
            self.z._ZT["on"] = True
            tok = self.z._zt_begin("mamba")
            self.z._zt_end(tok)
            self.assertEqual(1, len(self.z._ZT["spans"]["mamba"]))
            first = list(self.z._ZT["pool"])
            self.assertEqual(2, self.z._ZT["pool_n"])
            self.z._zt_reset()
            self.assertEqual(0, self.z._ZT["pool_n"])
            self.assertEqual([], self.z._ZT["spans"]["mamba"])
            tok = self.z._zt_begin("attn")
            self.z._zt_end(tok)
            # events are recycled, not re-allocated (keeps ~400 allocs/step off
            # the measured path)
            self.assertIs(first[0], self.z._ZT["pool"][0])
            self.z._ZT["on"] = False

    def test_closure_excludes_nested_diagnostic_bucket(self):
        blk = self.z._zt_new_block()
        fwd = {"attn": 2.0, "mlp": 1.0, "other": 1.0, "mamba": 5.0, "attn_core": 1.5}
        closure = self.z._zt_block_update(blk, fwd, 10.0, (32, 32))
        self.assertAlmostEqual(0.9, closure)          # (2+1+1+5)/10, attn_core excluded
        self.assertAlmostEqual(1.5, blk["attn_core"])  # still accumulated for reporting
        self.assertEqual(0, blk["dirty"])

    def test_closure_flags_unbucketed_work(self):
        """Reproduces defect 1 numerically: only the attn core + mamba bucketed."""
        blk = self.z._zt_new_block()
        fwd = {"attn": 0.0, "mlp": 0.0, "other": 0.0, "mamba": 5.0, "attn_core": 0.3}
        closure = self.z._zt_block_update(blk, fwd, 20.0, (32, 32))
        self.assertLess(closure, 0.95)

    def test_shape_change_marks_block_dirty(self):
        blk = self.z._zt_new_block()
        fwd = {b: 1.0 for b in self.z._ZT_BUCKETS + self.z._ZT_DIAG}
        self.z._zt_block_update(blk, fwd, 8.0, (32, 32))
        self.assertEqual(0, blk["dirty"])
        self.z._zt_block_update(blk, fwd, 8.0, (17, 17))
        self.assertEqual(1, blk["dirty"])
        self.assertEqual(2, blk["n"])

    def test_new_block_is_zeroed(self):
        blk = self.z._zt_new_block()
        fwd = {b: 3.0 for b in self.z._ZT_BUCKETS + self.z._ZT_DIAG}
        self.z._zt_block_update(blk, fwd, 100.0, (1, 1))
        fresh = self.z._zt_new_block()
        for b in self.z._ZT_BUCKETS + self.z._ZT_DIAG:
            self.assertEqual(0.0, fresh[b])
        self.assertEqual(0, fresh["n"])
        self.assertIsNone(fresh["shape"])
        self.assertEqual(0, fresh["dirty"])

    def test_zero_total_is_nan_not_a_divide_error(self):
        blk = self.z._zt_new_block()
        fwd = {b: 0.0 for b in self.z._ZT_BUCKETS + self.z._ZT_DIAG}
        closure = self.z._zt_block_update(blk, fwd, 0.0, (1, 1))
        self.assertNotEqual(closure, closure)  # NaN
        self.assertEqual(float("inf"), blk["closure_min"])  # NaN never becomes the min


class Zamba2MirrorSyncTest(unittest.TestCase):
    def test_tracked_mirror_matches_installed_runtime(self):
        if not RUNTIME.is_file():
            self.skipTest(f"runtime tree absent: {RUNTIME}")
        self.assertEqual(
            MIRROR.read_text(),
            RUNTIME.read_text(),
            "src/models/zamba2.py and the installed runtime diverged; "
            "re-run scripts/bootstrap/sync_engine_tree.sh",
        )


if __name__ == "__main__":
    unittest.main()
