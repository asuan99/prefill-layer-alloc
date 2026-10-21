"""Multi-GPU M2 engine wiring (2026-10-12) -- CPU gates.

Design: reports/multigpu_scaling_design_2026-10-12.md sec 2-1, 2-5, 4-1;
audit R-3 / R-4 (reports/audit/2026-10-12_multigpu_design/
AUDIT_multigpu_design_rev1_2026-10-12.md).

What is pinned here:

  A. The TP>1 fail-fast predicate (`multigpu_guard.tp_pdmux_violations`).
     Allowed for tp_size > 1 = legacy loop AND PDMUX_R2_POLICY unset/fixed AND
     PDMUX_SLO_SCHED, PDMUX_LA_COORD, PDMUX_TRUE_DUAL_WORKER unset.  The cases
     are a HAND-WRITTEN table (not a re-derivation of the predicate), and the
     mutation block (lesson 53) shows that each registered conjunct is
     load-bearing: a source copy with that conjunct relaxed must FAIL this
     table, while the unmutated copy (control, lesson 254) must pass it.
  B. That `init_pdmux` calls the guard as its FIRST statement (before any
     stream group exists) -- AST check plus a mutant without the call.
  C. Per-process output naming + record fields.  tp=dp=1 is byte-identical to
     the pre-change behaviour: the path string is returned untouched and a
     telemetry record matches a golden line with no new key.
  D. The pdmux_context device assert (R-4): fires only in the divide_sm
     (auto-grid) branch; a manual_divisions config with the same mismatch is
     untouched; the pristine upstream file (negative control) accepts the
     mismatch silently.
  E. sync_engine_tree.sh installs the new module, applies the patch under a
     grep guard that matches the patch text, and APPENDS both manifest lines.

NO GPU.  None of this says TP PD-mux is correct or fast; that is the M2g GPU
gate, which has not run.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
SRC = ENG / "src" / "multiplex"
GUARD_PY = SRC / "multigpu_guard.py"
MIXIN_PY = SRC / "multiplexing_mixin.py"
TELEMETRY_PY = SRC / "telemetry.py"
PATCH = ENG / "src" / "patches" / "pdmux_context_device_assert.patch"
SYNC = ENG / "scripts" / "bootstrap" / "sync_engine_tree.sh"
RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(
        os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
        "sglang_engine_dev",
        "python",
    )
)


def load_source(name, text):
    module = types.ModuleType(name)
    module.__file__ = f"<{name}>"
    sys.modules[name] = module  # @dataclass resolves annotations via sys.modules
    exec(compile(text, module.__file__, "exec"), module.__dict__)
    return module


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


GUARD = load_file("multigpu_guard_under_test", GUARD_PY)

# --------------------------------------------------------------------------
# A. predicate table -- written out by hand from the registered rule
# --------------------------------------------------------------------------
# (tp_size, env, allowed)
CASES = [
    # tp=1: the guard never refuses anything (single-GPU behaviour unchanged)
    (1, {}, True),
    (1, {"PDMUX_R2_POLICY": "hybrid", "PDMUX_TRUE_DUAL_WORKER": "1"}, True),
    (1, {"PDMUX_SLO_SCHED": "1"}, True),
    (1, {"PDMUX_LA_COORD": "1"}, True),
    (1, {"PDMUX_R2_POLICY": "generic"}, True),
    # tp>1, allowed: legacy + (unset | fixed), flags unset
    (2, {}, True),
    (2, {"PDMUX_R2_POLICY": "fixed"}, True),
    (8, {"PDMUX_R2_POLICY": "fixed", "PDMUX_R2_FIXED_DSM": "44"}, True),
    (4, {"PDMUX_R2_POLICY": " Fixed "}, True),  # engine strips/lowers it
    (2, {"PDMUX_R2_POLICY": ""}, True),
    (2, {"PDMUX_SLO_SCHED": ""}, True),  # empty == unset
    (2, {"PDMUX_TRUE_DUAL_WORKER": ""}, True),
    # not part of the five registered conjuncts -> not refused here
    (2, {"PDMUX_R2_POLICY": "fixed", "PDMUX_STICKY_PARTITION": "1"}, True),
    (2, {"PDMUX_DUAL_WORKER": "1"}, True),
    # tp>1, refused: dynamic R2 policies
    (2, {"PDMUX_R2_POLICY": "generic"}, False),
    (2, {"PDMUX_R2_POLICY": "hybrid", "PDMUX_MODEL_PROFILE": "/x.json"}, False),
    (8, {"PDMUX_R2_POLICY": "GENERIC"}, False),
    (2, {"PDMUX_R2_POLICY": "nonsense"}, False),
    # tp>1, refused: single-worker SLO controller
    (2, {"PDMUX_SLO_SCHED": "1"}, False),
    (2, {"PDMUX_SLO_SCHED": "0"}, False),  # engine reads truthiness: "0" is ON
    (2, {"PDMUX_SLO_SCHED": "1", "PDMUX_R2_POLICY": "fixed"}, False),
    # tp>1, refused: coordinated per-type loop (not the legacy loop)
    (2, {"PDMUX_LA_COORD": "1"}, False),
    (4, {"PDMUX_LA_COORD": "1", "PDMUX_R2_POLICY": "fixed"}, False),
    # tp>1, refused: true dual-worker (not the legacy loop), even with fixed
    (2, {"PDMUX_TRUE_DUAL_WORKER": "1"}, False),
    (2, {"PDMUX_TRUE_DUAL_WORKER": "1", "PDMUX_R2_POLICY": "fixed"}, False),
    (2, {"PDMUX_TRUE_DUAL_WORKER": "true"}, False),
    (2, {"PDMUX_TRUE_DUAL_WORKER": "0"}, False),  # literal "unset" rule
]


def table_failures(module):
    """Every way `module` disagrees with CASES (empty == agrees)."""
    out = []
    for tp, env, allowed in CASES:
        got = module.tp_pdmux_violations(tp, env)
        if (got == []) != allowed:
            out.append(f"tp={tp} env={env}: allowed={allowed} but violations={got}")
            continue
        try:
            module.check_tp_pdmux_combination(tp, env)
            raised = False
        except RuntimeError:
            raised = True
        if raised == allowed:
            out.append(f"tp={tp} env={env}: check() raised={raised}, allowed={allowed}")
    return out


GUARD_TEXT = GUARD_PY.read_text(encoding="utf-8")

# Each mutant relaxes (or wrongly tightens) exactly one registered conjunct.
MUTANTS = {
    "drop_SLO_SCHED": ('    "PDMUX_SLO_SCHED",\n', ""),
    "drop_LA_COORD": ('    "PDMUX_LA_COORD",\n', ""),
    "drop_TRUE_DUAL_WORKER": ('    "PDMUX_TRUE_DUAL_WORKER",\n', ""),
    "allow_generic": (
        'TP_ALLOWED_R2_POLICIES = ("", "fixed")',
        'TP_ALLOWED_R2_POLICIES = ("", "fixed", "generic")',
    ),
    "allow_hybrid": (
        'TP_ALLOWED_R2_POLICIES = ("", "fixed")',
        'TP_ALLOWED_R2_POLICIES = ("", "fixed", "hybrid")',
    ),
    "tp2_slips_through": ("    if int(tp_size) <= 1:\n", "    if int(tp_size) <= 2:\n"),
    "zero_counts_as_unset": (
        '    return env.get(key, "") == ""\n',
        '    return env.get(key, "") in ("", "0")\n',
    ),
    "no_policy_normalisation": (
        'policy = str(env.get("PDMUX_R2_POLICY", "")).strip().lower()',
        'policy = str(env.get("PDMUX_R2_POLICY", ""))',
    ),
    "check_never_raises": ("    if bad:\n        raise RuntimeError(", "    if False:\n        raise RuntimeError("),
}


class TestTpPredicate(unittest.TestCase):
    def test_tracked_predicate_matches_the_registered_table(self):
        self.assertEqual(table_failures(GUARD), [])

    def test_unmutated_copy_passes_control(self):
        """Lesson 254: the mutation loader itself must not kill a clean copy."""
        self.assertEqual(table_failures(load_source("guard_control", GUARD_TEXT)), [])

    def test_every_mutant_is_killed(self):
        for name, (old, new) in MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertEqual(GUARD_TEXT.count(old), 1, f"{name}: anchor not unique/absent")
                mutant = load_source(f"guard_{name}", GUARD_TEXT.replace(old, new))
                self.assertTrue(table_failures(mutant), f"mutant {name} SURVIVED")

    def test_refusal_names_every_violated_flag(self):
        env = {
            "PDMUX_R2_POLICY": "hybrid",
            "PDMUX_SLO_SCHED": "1",
            "PDMUX_LA_COORD": "1",
            "PDMUX_TRUE_DUAL_WORKER": "1",
        }
        with self.assertRaises(RuntimeError) as ctx:
            GUARD.check_tp_pdmux_combination(4, env)
        message = str(ctx.exception)
        for key in ("PDMUX_R2_POLICY", "PDMUX_SLO_SCHED", "PDMUX_LA_COORD", "PDMUX_TRUE_DUAL_WORKER"):
            self.assertIn(key, message)
        self.assertIn("tp_size=4", message)

    def test_predicate_is_pure(self):
        env = {"PDMUX_R2_POLICY": "generic"}
        before = dict(env)
        with mock.patch.dict(os.environ, {"PDMUX_R2_POLICY": "fixed"}, clear=False):
            self.assertTrue(GUARD.tp_pdmux_violations(2, env))
        self.assertEqual(env, before)
        with mock.patch.dict(os.environ, {"PDMUX_SLO_SCHED": "1"}, clear=False):
            self.assertEqual(GUARD.tp_pdmux_violations(2, {}), [])


# --------------------------------------------------------------------------
# B. the guard is wired as the first statement of init_pdmux
# --------------------------------------------------------------------------
MIXIN_TEXT = MIXIN_PY.read_text(encoding="utf-8")
WIRING_MARKER = "# Multi-GPU (TP>1 / DP>1) wiring -- M2, 2026-10-12"


def init_pdmux_first_call(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "init_pdmux":
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Call)
                and isinstance(first.value.func, ast.Name)
            ):
                return first.value.func.id, [ast.unparse(a) for a in first.value.args]
            return None
    raise AssertionError("init_pdmux not found")


def wiring_block(text):
    index = text.index(WIRING_MARKER)
    return text[text.rindex("\n#", 0, index) :]


def exec_wiring(text=MIXIN_TEXT):
    """Run the REAL appended mixin helpers against the tracked guard module."""
    namespace = {"os": os, "__name__": "mixin_wiring"}
    exec(compile(wiring_block(text), "<mixin wiring>", "exec"), namespace)
    return namespace


def call_wiring(namespace, fn, *args):
    with mock.patch.dict(sys.modules, {"sglang.srt.multiplex": None, "multigpu_guard": GUARD}):
        return namespace[fn](*args)


class TestMixinWiring(unittest.TestCase):
    def test_guard_is_the_first_statement_of_init_pdmux(self):
        self.assertEqual(init_pdmux_first_call(MIXIN_TEXT), ("_pdmux_multigpu_init", ["self"]))

    def test_mutant_without_the_call_is_detected(self):
        mutant = MIXIN_TEXT.replace(
            "        _pdmux_multigpu_init(self)  # M2", "        pass  # M2", 1
        )
        self.assertNotEqual(mutant, MIXIN_TEXT)
        self.assertNotEqual(init_pdmux_first_call(mutant), ("_pdmux_multigpu_init", ["self"]))

    def test_guard_runs_before_stream_groups_are_created(self):
        body = MIXIN_TEXT[MIXIN_TEXT.index("    def init_pdmux(self: Scheduler):") :]
        self.assertLess(body.index("_pdmux_multigpu_init(self)"), body.index("initialize_stream_groups("))

    def test_wrapper_refuses_tp2_dynamic_and_accepts_tp2_fixed(self):
        ns = exec_wiring()
        sched = SimpleNamespace(tp_size=2, tp_rank=1)
        with mock.patch.dict(os.environ, {"PDMUX_R2_POLICY": "generic"}, clear=False):
            with self.assertRaises(RuntimeError):
                call_wiring(ns, "_pdmux_multigpu_init", sched)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
        env["PDMUX_R2_POLICY"] = "fixed"
        with mock.patch.dict(os.environ, env, clear=True):
            call_wiring(ns, "_pdmux_multigpu_init", sched)

    def test_wrapper_ignores_env_on_a_single_gpu_fake(self):
        """A fake scheduler without tp_size (every existing CPU fake) is tp=1."""
        ns = exec_wiring()
        with mock.patch.dict(os.environ, {"PDMUX_R2_POLICY": "hybrid", "PDMUX_TRUE_DUAL_WORKER": "1"}):
            call_wiring(ns, "_pdmux_multigpu_init", SimpleNamespace())
        self.assertEqual(call_wiring(ns, "_pdmux_rank_fields", SimpleNamespace()), {})
        self.assertEqual(call_wiring(ns, "_pdmux_rank_path", SimpleNamespace(), "/t/tel.jsonl"), "/t/tel.jsonl")

    def test_wrapper_paths_for_tp_and_dp(self):
        ns = exec_wiring()
        tp = SimpleNamespace(tp_size=4, tp_rank=3, dp_size=1, dp_rank=None)
        dp = SimpleNamespace(tp_size=1, tp_rank=0, dp_size=2, dp_rank=1)
        self.assertEqual(call_wiring(ns, "_pdmux_rank_path", tp, "/t/tel.jsonl"), "/t/tel.tp3.jsonl")
        self.assertEqual(call_wiring(ns, "_pdmux_rank_path", dp, "/t/g.json"), "/t/g.dp1.json")
        self.assertEqual(call_wiring(ns, "_pdmux_rank_fields", dp), {"tp_rank": 0, "dp_rank": 1})

    def test_all_three_output_sites_are_routed(self):
        self.assertIn("trace_path = _pdmux_rank_path(self, trace_path)", MIXIN_TEXT)
        self.assertIn("static_fields=_pdmux_rank_fields(self)", MIXIN_TEXT)
        self.assertIn(
            'path = _pdmux_rank_path(self, os.environ.get("PDMUX_GREEN_READOUT_PATH", ""))',
            MIXIN_TEXT,
        )


# --------------------------------------------------------------------------
# C. naming and schema
# --------------------------------------------------------------------------
class TestRankNaming(unittest.TestCase):
    def test_single_gpu_path_is_returned_untouched(self):
        for path in ("/r/tel.jsonl", "./tel.jsonl", "tel", "a.b.jsonl", "/r/dir/", ""):
            with self.subTest(path=path):
                self.assertIs(GUARD.rank_suffixed_path(path, 1, 0, 1, None), path)
                self.assertEqual(GUARD.rank_fields(1, 0, 1, None), {})

    def test_tp_only_matches_the_holb_chunk_precedent(self):
        for raw in ("/r/tel.jsonl", "/r/a.b.jsonl", "/r/noext"):
            p = Path(raw)
            precedent = str(p.with_name(f"{p.stem}.tp3{p.suffix}"))  # holb_probe.py:567 before M2
            self.assertEqual(GUARD.rank_suffixed_path(raw, 4, 3), precedent)

    def test_dp_and_tp_levels(self):
        self.assertEqual(GUARD.rank_suffixed_path("/r/t.jsonl", 1, 0, 2, 1), "/r/t.dp1.jsonl")
        self.assertEqual(GUARD.rank_suffixed_path("/r/t.jsonl", 2, 0, 2, 1), "/r/t.dp1.tp0.jsonl")
        self.assertEqual(GUARD.rank_suffixed_path("/r/t.jsonl", 1, 0, 2, None), "/r/t.dp0.jsonl")

    def test_every_process_of_a_dp2_tp4_launch_gets_its_own_file(self):
        paths = {
            GUARD.rank_suffixed_path("/r/t.jsonl", 4, t, 2, d) for d in range(2) for t in range(4)
        }
        self.assertEqual(len(paths), 8)

    def test_fields_carry_both_coordinates_once_multi_gpu(self):
        self.assertEqual(GUARD.rank_fields(2, 1), {"tp_rank": 1, "dp_rank": 0})
        self.assertEqual(GUARD.rank_fields(1, 0, 4, 3), {"tp_rank": 0, "dp_rank": 3})


# Golden record of the PRE-M2 writer for one event with a pinned clock.  Kept as
# a literal (not regenerated from git) so it cannot become an identity after
# the change is committed.
GOLDEN_LINE = (
    '{"a":1,"dropped_events":0,"event":"ev","phase":"ph","request_id":"",'
    '"run_id":"rid","schema":"pdmux.runtime-event/v1",'
    '"timestamp_monotonic_s":12.5,"workload_id":"wid"}'
)


TS = re.compile(r'"timestamp_monotonic_s":[0-9.eE+-]+')


def write_one(tel_module, **kw):
    """One event, writer stopped without the close record.  The clock value is
    normalised textually (the dataclass default_factory holds its own reference
    to time.perf_counter, so it cannot be patched); every other byte is real."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.jsonl"
        writer = tel_module.AsyncJsonlTelemetry(
            path, run_id="rid", workload_id="wid", flush_every=1, **kw
        )
        writer.emit("ev", "ph", a=1)
        writer._queue.put(None)
        writer._thread.join(5)
        lines = path.read_text(encoding="utf-8").splitlines()
    return [TS.sub('"timestamp_monotonic_s":12.5', line) for line in lines]


class TestTelemetrySchema(unittest.TestCase):
    def setUp(self):
        self.tel = load_file("telemetry_m2_under_test", TELEMETRY_PY)

    def test_default_writer_is_byte_identical_to_the_pre_change_golden(self):
        self.assertEqual(write_one(self.tel), [GOLDEN_LINE])

    def test_empty_static_fields_is_byte_identical(self):
        self.assertEqual(write_one(self.tel, static_fields={}), [GOLDEN_LINE])
        self.assertEqual(write_one(self.tel, static_fields=None), [GOLDEN_LINE])

    def test_rank_fields_are_added_to_every_record(self):
        (line,) = write_one(self.tel, static_fields=GUARD.rank_fields(2, 1))
        record = json.loads(line)
        self.assertEqual((record["tp_rank"], record["dp_rank"]), (1, 0))
        golden = json.loads(GOLDEN_LINE)
        self.assertEqual({k: v for k, v in record.items() if k not in ("tp_rank", "dp_rank")}, golden)

    def test_explicit_field_wins_over_static(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.jsonl"
            writer = self.tel.AsyncJsonlTelemetry(
                path, "rid", "wid", flush_every=1, static_fields={"tp_rank": 1}
            )
            writer.emit("ev", "ph", tp_rank=7)
            writer.close()
            records = [json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(records[0]["tp_rank"], 7)
        self.assertEqual(records[-1]["event"], "telemetry_close")
        self.assertEqual(records[-1]["tp_rank"], 1)


class TestProbeNaming(unittest.TestCase):
    """holb/chunk probes: tp-only naming unchanged, dp suffix added."""

    def _helpers(self, name):
        text = (SRC / name).read_text(encoding="utf-8")
        block = text[text.index("# M2 (2026-10-12): per-process path naming") :]
        ns = {"__name__": f"{name}_tail"}
        with mock.patch.dict(sys.modules, {"sglang.srt.multiplex.multigpu_guard": None, "multigpu_guard": GUARD}):
            exec(compile(block, name, "exec"), ns)
            return {
                "suffix": lambda s: ns["_probe_rank_suffix"](s),
                "tag": lambda s: ns["_probe_dp_tag"](s),
            }, text

    def test_both_probes(self):
        for name in ("holb_probe.py", "chunk_probe.py"):
            with self.subTest(probe=name):
                helpers, text = self._helpers(name)
                with mock.patch.dict(sys.modules, {"sglang.srt.multiplex.multigpu_guard": None, "multigpu_guard": GUARD}):
                    self.assertEqual(helpers["suffix"](SimpleNamespace(tp_size=2, tp_rank=1)), ".tp1")
                    self.assertEqual(helpers["suffix"](SimpleNamespace(tp_size=1, dp_size=2, dp_rank=1)), ".dp1")
                    self.assertEqual(helpers["suffix"](SimpleNamespace()), "")
                    self.assertEqual(helpers["tag"](SimpleNamespace(tp_size=2, tp_rank=1)), {})
                    self.assertEqual(helpers["tag"](SimpleNamespace(dp_size=2, dp_rank=1)), {"dp_rank": 1})
                self.assertIn('if tp_size > 1 or int(getattr(scheduler, "dp_size", 1) or 1) > 1:', text)
                self.assertIn('path = path.with_name(f"{path.stem}{_probe_rank_suffix(scheduler)}{path.suffix}")', text)
                self.assertIn("**_probe_dp_tag(scheduler),", text)


# --------------------------------------------------------------------------
# D. pdmux_context device assert (R-4)
# --------------------------------------------------------------------------
INSTALLED_CTX = RUNTIME / "sglang" / "srt" / "multiplex" / "pdmux_context.py"
ASSERT_MARK = "PD-mux auto grid (divide_sm)"


def ctx_variants():
    """(patched_text, pristine_text) built from whatever tree is installed."""
    if not INSTALLED_CTX.is_file() or shutil.which("patch") is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "sglang" / "srt" / "multiplex" / "pdmux_context.py"
        target.parent.mkdir(parents=True)
        shutil.copy(INSTALLED_CTX, target)
        already = ASSERT_MARK in target.read_text(encoding="utf-8")
        direction = ["-R"] if already else []
        first = subprocess.run(
            ["patch", "--batch", "-p1", "-d", td, *direction],
            input=PATCH.read_bytes(), capture_output=True,
        )
        if first.returncode != 0:
            raise AssertionError(f"tracked patch does not apply: {first.stdout!r}{first.stderr!r}")
        other = target.read_text(encoding="utf-8")
        installed = INSTALLED_CTX.read_text(encoding="utf-8")
    return (installed, other) if already else (other, installed)


class FakeSpatial:
    def __init__(self):
        self.calls = []

    def get_sm_available(self, gpu_id):
        return 108

    def create_greenctx_stream_by_value(self, a, b, gpu_id):
        self.calls.append((a, b, gpu_id))
        return (f"g{a}", f"g{b}")


def run_init(text, *, current_device, gpu_id, manual):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        current_device=lambda: current_device,
        get_device_capability=lambda d: (8, 0),
        Stream=lambda d: f"s{d}",
    )
    spatial = FakeSpatial()
    kernel = types.ModuleType("sgl_kernel")
    kernel.spatial = spatial
    with mock.patch.dict(
        sys.modules,
        {"torch": fake_torch, "yaml": types.ModuleType("yaml"), "sgl_kernel": kernel, "sgl_kernel.spatial": spatial},
    ):
        module = load_source("pdmux_context_under_test", text)
        cfg = module.PDMuxConfig(
            sm_group_num=6,
            manual_divisions=[[92, 16, 0], [84, 24, 0], [74, 34, 0], [64, 44, 0]] if manual else [],
        )
        module.initialize_stream_groups(gpu_id, cfg)
        return module.get_sm_counts()


class TestDeviceAssert(unittest.TestCase):
    def setUp(self):
        variants = ctx_variants()
        if variants is None:
            self.skipTest(f"no installed pdmux_context.py under {RUNTIME}")
        self.patched, self.pristine = variants

    def test_patch_touches_only_the_divide_sm_branch(self):
        tree = ast.parse(self.patched)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "initialize_stream_groups")
        asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
        self.assertEqual(len(asserts), 1)
        branch = next(n for n in fn.body if isinstance(n, ast.If) and "manual_divisions" in ast.unparse(n.test))
        self.assertIn(asserts[0], branch.orelse)
        self.assertEqual(ast.unparse(asserts[0].test), "device == gpu_id")

    def test_patch_is_line_neutral(self):
        self.assertEqual(len(self.patched.splitlines()), len(self.pristine.splitlines()))
        a, b = self.patched.splitlines(), self.pristine.splitlines()
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 2)
        self.assertEqual(a[123:138], b[123:138])  # :124-138 is a snapshotted citation

    def test_auto_grid_mismatch_is_refused(self):
        with self.assertRaises(AssertionError) as ctx:
            run_init(self.patched, current_device=0, gpu_id=3, manual=False)
        self.assertIn("gpu_id=3", str(ctx.exception))

    def test_auto_grid_match_proceeds(self):
        counts = run_init(self.patched, current_device=2, gpu_id=2, manual=False)
        self.assertEqual(counts[0], (108, 0))

    def test_manual_divisions_ignore_the_mismatch(self):
        counts = run_init(self.patched, current_device=0, gpu_id=5, manual=True)
        self.assertEqual(counts, [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)])
        self.assertEqual(counts, run_init(self.pristine, current_device=0, gpu_id=5, manual=True))

    def test_negative_control_pristine_accepts_the_mismatch_silently(self):
        """Without the patch the mismatch passes -- the defect being guarded."""
        counts = run_init(self.pristine, current_device=0, gpu_id=3, manual=False)
        self.assertEqual(counts[0], (108, 0))


# --------------------------------------------------------------------------
# E. sync script
# --------------------------------------------------------------------------
class TestSyncScript(unittest.TestCase):
    def setUp(self):
        self.sync = SYNC.read_text(encoding="utf-8")

    def test_guard_module_is_installed(self):
        # inside the `for source in ... ; do` install loop (2026-10-20: no longer its last item,
        # admission_probe.py and decode_probe.py were appended after it)
        loop = self.sync[self.sync.index("for source in \\\n"):]
        loop = loop[:loop.index('; do')]
        self.assertIn('"${track_root}/src/multiplex/multigpu_guard.py"', loop)

    def test_patch_is_applied_under_a_guard_that_matches_its_text(self):
        self.assertIn("pdmux_context_device_assert.patch", self.sync)
        self.assertIn(f'grep -q "{ASSERT_MARK}"', self.sync)
        self.assertIn(ASSERT_MARK, PATCH.read_text(encoding="utf-8").split("@@", 1)[1])

    def test_manifest_lines_are_appended_after_flashinfer(self):
        fi = self.sync.index('"${runtime_python}/sglang/srt/layers/attention/flashinfer_backend.py" \\')
        ctx = self.sync.index('"${runtime_python}/sglang/srt/multiplex/pdmux_context.py" \\')
        guard = self.sync.index('"${runtime_python}/sglang/srt/multiplex/multigpu_guard.py" \\')
        end = self.sync.index('> "${manifest_tmp}"')
        self.assertLess(fi, ctx)
        self.assertLess(ctx, guard)
        self.assertLess(guard, end)


if __name__ == "__main__":
    unittest.main()
