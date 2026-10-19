"""CPU gate for the I-5 admission-reason probe (SPEC_common_measurement_v0 sec 9).

What is pinned here (NO GPU -- nothing here says the probe is neutral or that
ON/OFF outputs match; those are GPU gates (a)/(b)):

  A. ``classify_add`` against a HAND-WRITTEN table (not a re-derivation).
  B. ``classify_add`` agrees with the REAL ``PrefillAdder.add_one_req`` on
     every call of a scenario set (the probe counts ``classifier_mismatch``
     when it does not; here it must stay 0).
  C. The engine's own ``Scheduler._get_new_batch_prefill_raw`` /
     ``update_running_batch`` source (AST-cut from the installed tree, see
     ``build_engine``) driven through the wrappers on a fake scheduler: cap, KV,
     max_prefill_tokens, sticky inheritance, chunked guard, no_waiting,
     admitted_all, retract, decode-bs histogram, slot-vs-cap shadow.
  D. PD-mux pre-gates through the REAL mixin ``update_split_prefill_batch``:
     prefill seat, R2 latch.
  E. Mutation (gate (c) on CPU, lesson 53): a probe that merges KV into cap,
     one that maps every rejection to one reason, one that computes the slot
     shadow from cap headroom, and one that drops sticky inheritance must each
     FAIL the scenario table that the unmutated probe passes.
  F. Default OFF installs nothing; schema gate (d); boot verdicts.
  G. The scheduler hook patch applies to the installed tree, reverses
     exactly, composes with the holb/chunk patches, and lands inside
     ``run_scheduler_process`` (no line of the Scheduler class moves).

The probe module is loaded from the MIRROR (src/multiplex/admission_probe.py):
it is not yet installed by sync_engine_tree.sh.
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ENG = Path(__file__).resolve().parents[1]
PROBE_PY = ENG / "src" / "multiplex" / "admission_probe.py"
PATCH = ENG / "src" / "patches" / "admission_probe_scheduler_hook.patch"
RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(
        os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
        "sglang_engine_dev",
        "python",
    )
)
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
# test_profile_controller.py may have shadowed the stdlib `profile` module.
sys.modules.pop("profile", None)


def _load_probe():
    spec = importlib.util.spec_from_file_location("admission_probe_under_test", PROBE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


ap = _load_probe()


# ---------------------------------------------------------------------------
# A. pure classifier, hand-written table
# ---------------------------------------------------------------------------

def S(**kw):
    base = dict(added=False, n_before=0, rem_total=10_000, cur_rem=10_000,
                rem_input=8192, rem_chunk=None, extend_input_len=100,
                max_new_tokens=10, n_output=0)
    base.update(kw)
    return ap.AddSnapshot(**base)


TABLE = [
    # (name, snapshot, expected (cat, sub, res))
    ("admit_continue", S(added=True), ("continue", "-", "CONTINUE")),
    ("admit_then_kv_exhausted", S(added=True, rem_total=0), ("kv", "budget_after_add", "NO_TOKEN")),
    ("admit_then_cur_rem_exhausted", S(added=True, cur_rem=-1), ("kv", "budget_after_add", "NO_TOKEN")),
    ("admit_then_mpt_exhausted", S(added=True, rem_input=0), ("max_prefill_tokens", "budget_after_add", "OTHER")),
    ("admit_then_chunk_exhausted", S(added=True, rem_chunk=0), ("other", "chunk_budget", "OTHER")),
    ("kv_reject", S(extend_input_len=100, max_new_tokens=10, rem_total=110), ("kv", "add", "NO_TOKEN")),
    ("kv_just_fits", S(extend_input_len=100, max_new_tokens=10, rem_total=111, rem_chunk=None),
     ("unclassified", "expected_admit", "CONTINUE")),
    ("mpt_reject_second", S(n_before=1, extend_input_len=600, rem_input=500), ("max_prefill_tokens", "add", "OTHER")),
    ("mpt_first_req_never_rejected", S(n_before=0, extend_input_len=600, rem_input=500),
     ("unclassified", "expected_admit", "CONTINUE")),
    ("kv_precedes_mpt", S(n_before=1, extend_input_len=600, rem_input=500, rem_total=50), ("kv", "add", "NO_TOKEN")),
    ("prefill_max_requests", S(n_before=2, prefill_max_requests=2), ("other", "prefill_max_requests", "OTHER")),
    ("cp_single", S(n_before=1, cp_single=True), ("other", "cp_single", "OTHER")),
    ("delayer_first", S(delayer_present=True, rem_total=0), ("other", "prefill_delayer", "OTHER")),
    ("ignore_eos_early", S(ignore_eos_path=True, extend_input_len=200, cur_rem=150), ("kv", "ignore_eos_early", "NO_TOKEN")),
    ("ignore_eos_projection", S(ignore_eos_path=True), ("kv", "ignore_eos_projection", "NO_TOKEN")),
    ("chunk_zero", S(rem_chunk=0, extend_input_len=10), ("other", "chunk_budget", "OTHER")),
    ("trunc_align", S(rem_chunk=64, extend_input_len=100, truncation_align_size=128), ("other", "trunc_align", "OTHER")),
    ("paged_mpt", S(n_before=1, extend_input_len=257, page_size=256, rem_input=512), ("max_prefill_tokens", "add", "OTHER")),
]


def table_failures(classify):
    bad = []
    for name, snap, want in TABLE:
        got = classify(snap)
        if tuple(got) != want:
            bad.append((name, got, want))
    return bad


class TestClassifierTable(unittest.TestCase):
    def test_table(self):
        self.assertEqual(table_failures(ap.classify_add), [])

    def test_mamba_slot_never_terminal(self):
        self.assertNotIn(ap.MAMBA_SLOT, ap.TERMINAL_CATEGORIES)
        for _, snap, _ in TABLE:
            self.assertNotEqual(ap.classify_add(snap)[0], ap.MAMBA_SLOT)


# ---------------------------------------------------------------------------
# engine code under test: the INSTALLED tree's source text, executed with stubs
# ---------------------------------------------------------------------------
# The local CPU host has no sgl_kernel, so `import sglang.srt.managers.*` fails
# (fp8_kernel imports it at module scope).  Instead of mirroring the engine, the
# exact source of the functions under test is cut out of the installed files
# with `ast` and executed in a namespace whose OTHER names are stubs.  The
# control flow under test is therefore byte-for-byte the engine's.  The only
# textual change is a prepended `from __future__ import annotations` (scheduler.py
# lacks it, and its signatures name classes that are stubbed away), which
# affects annotation evaluation only.  TestExtractionFidelity compares the cut
# text with inspect.getsource wherever the real import works (GPU image).

import enum as _enum  # noqa: E402
import logging as _logging  # noqa: E402
import textwrap as _textwrap  # noqa: E402
from contextlib import contextmanager as _contextmanager  # noqa: E402
from typing import List as _List, Optional as _Optional  # noqa: E402

POLICY_PY = RUNTIME / "sglang/srt/managers/schedule_policy.py"
SCHED_PY = RUNTIME / "sglang/srt/managers/scheduler.py"
MIXIN_RT_PY = RUNTIME / "sglang/srt/multiplex/multiplexing_mixin.py"

SCHED_METHODS = ("get_new_batch_prefill", "_get_new_batch_prefill_raw",
                 "get_num_allocatable_reqs", "update_running_batch")
MIXIN_METHODS = ("update_split_prefill_batch", "_r2_admission_holds")


def _node_lines(lines, node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[first - 1: node.end_lineno]) + "\n"


def extract_top(path, names):
    text = Path(path).read_text()
    lines = text.splitlines()
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            out[node.name] = _node_lines(lines, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = _node_lines(lines, node)
    missing = set(names) - set(out)
    if missing:
        raise LookupError(f"{path}: not found {sorted(missing)}")
    return out


def extract_methods(path, cls, names):
    text = Path(path).read_text()
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out = {}
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name in names:
                    out[f.name] = _textwrap.dedent(_node_lines(lines, f))
            missing = set(names) - set(out)
            if missing:
                raise LookupError(f"{path}:{cls}: not found {sorted(missing)}")
            return out
    raise LookupError(f"{path}: class {cls} not found")


class _GlobalArgs(SimpleNamespace):
    pass


GLOBAL_ARGS = _GlobalArgs(
    enable_prefill_context_parallel=False, prefill_cp_mode="none",
    enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode="none",
    pp_max_micro_batch_size=48,
)


class DisaggregationMode(_enum.Enum):  # stub: only identity matters here
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


class _Stub:
    def __init__(self, *a, **k):
        pass


def build_engine():
    import torch  # PrefillAdder references torch only on the host-hit path

    ns = {
        "__name__": "engine_under_test",
        "Enum": _enum.Enum, "auto": _enum.auto, "contextmanager": _contextmanager,
        "Optional": _Optional, "List": _List, "os": os, "torch": torch,
        "logger": _logging.getLogger("engine_under_test"),
        "SWATokenToKVPoolAllocator": type("SWATokenToKVPoolAllocator", (), {}),
        "is_nsa_prefill_cp_in_seq_split": lambda: False,
        "is_prefill_context_parallel_enabled": lambda: False,
        "DecLockRefParams": _Stub, "InitLoadBackParams": _Stub,
        "get_global_server_args": lambda: GLOBAL_ARGS,
        "TEST_RETRACT": False, "TEST_RETRACT_INTERVAL": 3, "TEST_RETRACT_NO_PREFILL_BS": 2,
        "DisaggregationMode": DisaggregationMode,
        "PrefillStats": SimpleNamespace(from_adder=lambda *a, **k: None),
        "set_time_batch": lambda *a, **k: None,
        "AbortReq": _Stub,
        "ForwardMode": SimpleNamespace(SPLIT_PREFILL="SPLIT_PREFILL"),
        "WorkerRole": SimpleNamespace(PREFILL="prefill", DECODE="decode"),
    }
    pol = extract_top(POLICY_PY, ("CLIP_MAX_NEW_TOKENS", "IGNORE_EOS_RESERVE_TOKENS",
                                  "AddReqResult", "PrefillAdder"))
    src = "from __future__ import annotations\n" + "".join(
        pol[k] for k in ("CLIP_MAX_NEW_TOKENS", "IGNORE_EOS_RESERVE_TOKENS", "AddReqResult", "PrefillAdder"))
    exec(compile(src, str(POLICY_PY), "exec"), ns)
    sched = extract_methods(SCHED_PY, "Scheduler", SCHED_METHODS)
    mix = extract_methods(MIXIN_RT_PY, "SchedulerMultiplexMixin", MIXIN_METHODS)
    body = "".join(_textwrap.indent(m, "    ") for m in list(sched.values()) + list(mix.values()))
    exec(compile("from __future__ import annotations\nclass EngineScheduler:\n" + body,
                 "<scheduler.py+multiplexing_mixin.py extracted>", "exec"), ns)
    return ns, pol, sched, mix


ENGINE, _POL_SRC, _SCHED_SRC, _MIX_SRC = build_engine()
PrefillAdder = ENGINE["PrefillAdder"]
AddReqResult = ENGINE["AddReqResult"]
CLIP_MAX_NEW_TOKENS = ENGINE["CLIP_MAX_NEW_TOKENS"]
EngineScheduler = ENGINE["EngineScheduler"]


class FakeTreeCache:
    disable = True  # ChunkCache (radix OFF)

    def supports_mamba(self):
        return False

    def supports_swa(self):
        return False

    def is_tree_cache(self):
        return False

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def inc_lock_ref(self, node):
        return SimpleNamespace(swa_uuid_for_lock=None)

    def dec_lock_ref(self, node, params=None):
        return None


class FakeAllocator:
    def __init__(self, size):
        self.size = size

    def available_size(self):
        return self.size


class FakeReq:
    def __init__(self, rid, n_tokens, max_new_tokens=16, ignore_eos=False):
        self.rid = rid
        self.origin_input_ids = list(range(n_tokens))
        self.output_ids = []
        self.fill_ids = list(self.origin_input_ids)
        self.prefix_indices = []
        self.extend_input_len = n_tokens
        self.host_hit_length = 0
        self.last_node = None
        self.sampling_params = SimpleNamespace(ignore_eos=ignore_eos, max_new_tokens=max_new_tokens)
        self.is_chunked = 0
        self.mamba_pool_idx = None
        self.req_pool_idx = None
        self.lora_id = None

    def set_extend_input_len(self, n):
        self.extend_input_len = n

    def init_next_round_input(self, tree_cache=None):
        self.fill_ids = list(self.origin_input_ids) + list(self.output_ids)
        self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))


class FakeRunning:
    def __init__(self, n, max_new_tokens=16):
        self.reqs = [FakeReq(f"r{i}", 8, max_new_tokens) for i in range(n)]
        self.batch_is_full = False
        self.retract_n = 0

    def is_empty(self):
        return len(self.reqs) == 0

    def batch_size(self):
        return len(self.reqs)

    # update_running_batch surface
    def filter_batch(self, **kw):
        return None

    def check_decode_mem(self, *a, **kw):
        return self.retract_n == 0

    def retract_decode(self, server_args):
        out = [self.reqs.pop() for _ in range(self.retract_n)]
        self.retract_n = 0
        return out, 0.5, []

    def prepare_for_decode(self):
        return None


class FakeNewBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.input_embeds = None
        self.return_logprob = False
        self.forward_mode = SimpleNamespace(is_mixed=lambda: False, name="EXTEND")
        for r in reqs:  # init_new allocates req slots
            if r.req_pool_idx is None:
                r.req_pool_idx = id(r)

    @classmethod
    def init_new(cls, reqs, *a, **kw):
        return cls(list(reqs))

    def prepare_for_extend(self):
        return None

    def is_empty(self):
        return not self.reqs


ENGINE["ScheduleBatch"] = FakeNewBatch


class FakeMambaPool:
    def __init__(self, avail):
        self.avail = avail
        self.size = avail

    def available_size(self):
        return self.avail


class FakeScheduler(EngineScheduler):
    def __init__(self, *, running=0, waiting=(), kv_pool=10**9, cap=48,
                 max_prefill_tokens=16384, mamba_avail=None, chunked_req=None):
        self.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        self.enable_hierarchical_cache = False
        self.enable_priority_preemption = False
        self.enable_priority_scheduling = False
        self.running_batch = FakeRunning(running)
        self.waiting_queue = list(waiting)
        self.chunked_req = chunked_req
        self.pp_size = 1
        self.policy = SimpleNamespace(calc_priority=lambda q, rb: None)
        self.chunked_prefill_size = None
        self.enable_dynamic_chunking = False
        self.page_size = 1
        self.tree_cache = FakeTreeCache()
        self.token_to_kv_pool_allocator = FakeAllocator(kv_pool)
        self.new_token_ratio = 1.0
        self.new_token_ratio_decay = 0.0
        self.min_new_token_ratio = 0.1
        self.max_prefill_tokens = max_prefill_tokens
        self.is_mixed_chunk = False
        self.priority_scheduling_preemption_threshold = 0
        self.max_prefill_bs = 0
        self.max_running_requests = cap
        self.server_args = SimpleNamespace(prefill_max_requests=None)
        self.dllm_config = None
        self.enable_lora = False
        self.disaggregation_mode = DisaggregationMode.NULL
        self.enable_hicache_storage = False
        self.truncation_align_size = None
        self.req_to_token_pool = SimpleNamespace(
            mamba_pool=FakeMambaPool(mamba_avail) if mamba_avail is not None else None,
            size=cap,
        )
        self.model_config = None
        self.enable_overlap = False
        self.spec_algorithm = None
        self.prefill_delayer = None
        self.enable_metrics = False
        self.enable_hisparse = False
        self.forward_ct = 0
        # pdmux surface
        self.split_prefill_batch = None
        self.r2_admission_limited = False
        self.r2_policy = None
        self.dual_worker_enabled = False

    def _add_request_to_queue(self, req, is_retracted=False):
        self.waiting_queue.append(req)

    def _dual_worker_start_prefill(self, batch):
        return None


class EngineFixture(unittest.TestCase):
    CAP = 48

    def setUp(self):
        GLOBAL_ARGS.pp_max_micro_batch_size = self.CAP
        self.probe = ap.AdmissionProbe(
            None, window_s=1e9,
            boot={"pp_max_micro_batch_size": self.CAP},
            clock=lambda: 0.0,
        )
        self.probe.clip_max_new_tokens = CLIP_MAX_NEW_TOKENS
        ap.set_probe(self.probe)
        ap.install_wrappers(PrefillAdder, FakeScheduler, CLIP_MAX_NEW_TOKENS)
        self.addCleanup(self._teardown)

    def _teardown(self):
        ap.uninstall_wrappers()
        ap.set_probe(None)

    def terminal(self):
        return dict(self.probe.acct.terminal)


def run_scenarios(testcase_cls_probe_factory=None):
    """Scenario table over the real scheduler admission function.

    Returns a list of failures (empty = pass).  Used by the main tests and by
    the mutation tests, which must see a NON-empty list.
    """
    fails = []

    def check(name, cond, detail):
        if not cond:
            fails.append((name, detail))

    cap = EngineFixture.CAP

    def fresh(**kw):
        p = ap.AdmissionProbe(None, window_s=1e9, boot={"pp_max_micro_batch_size": cap}, clock=lambda: 0.0)
        p.clip_max_new_tokens = CLIP_MAX_NEW_TOKENS
        ap.set_probe(p)
        return p, FakeScheduler(cap=cap, **kw)

    # 1. cap: one free seat, two waiting -> admit one, cap stops the loop.
    p, s = fresh(running=cap - 1, waiting=[FakeReq("a", 10), FakeReq("b", 10)], mamba_avail=1)
    b = s.get_new_batch_prefill()
    check("cap/admit_one", b is not None and len(b.reqs) == 1, b)
    check("cap/terminal", p.acct.terminal == {"cap/loop": 1}, dict(p.acct.terminal))
    check("cap/slot_identical", p.acct.slot_rel == {"identical": 1}, dict(p.acct.slot_rel))
    check("cap/cobind_has_cap_and_slot", {"cap", "slot"} <= set(p.acct.cobind), dict(p.acct.cobind))
    # next decision on the sticky flag inherits cap
    s.get_new_batch_prefill()
    check("cap/sticky", p.acct.terminal.get("cap/sticky") == 1 and p.acct.inherited.get("cap") == 1,
          (dict(p.acct.terminal), dict(p.acct.inherited)))

    # 2. KV: pool too small for the first request.
    p, s = fresh(running=0, waiting=[FakeReq("a", 100, max_new_tokens=16)], kv_pool=50)
    check("kv/none", s.get_new_batch_prefill() is None, None)
    check("kv/terminal", p.acct.terminal == {"kv/add": 1}, dict(p.acct.terminal))
    s.get_new_batch_prefill()
    check("kv/sticky", p.acct.terminal.get("kv/sticky") == 1, dict(p.acct.terminal))

    # 3. max_prefill_tokens: second request overflows the prefill budget.
    p, s = fresh(running=0, waiting=[FakeReq("a", 300), FakeReq("b", 300)], max_prefill_tokens=500)
    b = s.get_new_batch_prefill()
    check("mpt/admit_one", b is not None and len(b.reqs) == 1, b)
    check("mpt/terminal", p.acct.terminal == {"max_prefill_tokens/add": 1}, dict(p.acct.terminal))
    check("mpt/not_sticky", not s.running_batch.batch_is_full, s.running_batch.batch_is_full)

    # 4. KV budget exhausted by an admitted request (budget_state after add)
    p, s = fresh(running=0, waiting=[FakeReq("a", 100, max_new_tokens=10), FakeReq("b", 5)], kv_pool=111)
    b = s.get_new_batch_prefill()
    check("kvb/admit_one", b is not None and len(b.reqs) == 1, b)
    check("kvb/terminal", p.acct.terminal == {"kv/budget_after_add": 1}, dict(p.acct.terminal))

    # 5. no waiting / admitted_all
    p, s = fresh(running=3, waiting=[])
    s.get_new_batch_prefill()
    check("nowait", p.acct.terminal == {"no_waiting/-": 1}, dict(p.acct.terminal))
    p, s = fresh(running=3, waiting=[FakeReq("a", 10), FakeReq("b", 10)])
    b = s.get_new_batch_prefill()
    check("all/admitted", b is not None and len(b.reqs) == 2, b)
    check("all/terminal", p.acct.terminal == {"admitted_all/-": 1}, dict(p.acct.terminal))
    check("all/extend_bs", p.acct.extend_bs == {3: 1} and p.acct.n_extend_running_nonempty == 1,
          dict(p.acct.extend_bs))

    # 6. slot_looser: more free slots than cap headroom (overlap-loop frees)
    p, s = fresh(running=cap - 1, waiting=[FakeReq("a", 10), FakeReq("b", 10)], mamba_avail=5)
    s.get_new_batch_prefill()
    check("slot/looser", p.acct.slot_rel == {"slot_looser": 1}, dict(p.acct.slot_rel))
    check("slot/looser_no_slot_cobind", "slot" not in p.acct.cobind, dict(p.acct.cobind))
    # slot_tighter: fewer free slots than cap headroom -> invariant event
    p, s = fresh(running=10, waiting=[FakeReq("a", 10)], mamba_avail=3)
    s.get_new_batch_prefill()
    check("slot/tighter", p.acct.slot_rel == {"slot_tighter": 1}, dict(p.acct.slot_rel))
    check("slot/tighter_event", any(r.get("kind") == "slot_tighter" for r in p.records), p.records)

    # 7. KV and cap co-bind: cap stops first, next request would also fail KV
    p, s = fresh(running=cap - 1, waiting=[FakeReq("a", 10, max_new_tokens=4), FakeReq("b", 5000)],
                 kv_pool=5000, mamba_avail=1)
    s.get_new_batch_prefill()
    check("cobind/terminal_cap", p.acct.terminal == {"cap/loop": 1}, dict(p.acct.terminal))
    check("cobind/kv_shadow", p.acct.cobind.get("kv") == 1, dict(p.acct.cobind))

    # 8. retract + decode bs histogram via the real update_running_batch
    p, s = fresh(running=cap)
    out = s.update_running_batch(s.running_batch)
    check("decode/at_cap", p.acct.decode_bs == {cap: 1} and p.acct.n_decode_at_cap == 1,
          dict(p.acct.decode_bs))
    s.running_batch.retract_n = 2
    out = s.update_running_batch(s.running_batch)
    check("retract/count", p.acct.n_retract_events == 2 and p.acct.n_retracted_reqs == 2,
          (p.acct.n_retract_events, p.acct.n_retracted_reqs))
    check("retract/decode_bs", p.acct.decode_bs.get(cap - 2) == 1, dict(p.acct.decode_bs))

    # 9. chunked guard: continuing chunked request with the cap reached
    ch = FakeReq("c", 10)
    ch.req_pool_idx = 7
    ch.is_chunked = 1
    p, s = fresh(running=cap, waiting=[FakeReq("a", 10)], chunked_req=ch)
    check("guard/none", s.get_new_batch_prefill() is None, None)
    check("guard/terminal", p.acct.terminal == {"cap/chunked_guard": 1}, dict(p.acct.terminal))

    # 10. PD-mux pre-gates through the real mixin method
    p, s = fresh(running=4, waiting=[FakeReq("a", 10)])
    s.split_prefill_batch = SimpleNamespace(reqs=[1])
    check("seat/ret", s.update_split_prefill_batch(0) is False, None)
    check("seat/terminal", p.acct.terminal == {"prefill_seat/pdmux": 1}, dict(p.acct.terminal))
    s.split_prefill_batch = None
    s.r2_admission_limited = True
    check("latch/ret", s.update_split_prefill_batch(0) is False, None)
    check("latch/terminal", p.acct.terminal.get("r2_latch/pdmux") == 1, dict(p.acct.terminal))
    s.r2_admission_limited = False
    check("pdmux/admit", s.update_split_prefill_batch(0) is True, None)
    check("pdmux/one_decision", sum(p.acct.terminal.values()) == 3, dict(p.acct.terminal))
    p, s = fresh(running=4, waiting=[])
    s.split_prefill_batch = SimpleNamespace(reqs=[1])
    s.update_split_prefill_batch(0)
    check("seat/vacuous", p.acct.terminal == {"no_waiting/seat_busy": 1}, dict(p.acct.terminal))

    # classifier never disagreed with the engine
    check("no_mismatch", p.acct.classifier_mismatch == 0, p.acct.classifier_mismatch)
    return fails


class TestRealScheduler(EngineFixture):
    def test_scenarios(self):
        self.assertEqual(run_scenarios(), [])

    def test_classifier_agrees_with_engine_on_every_add(self):
        """Every add_one_req result in a randomised sweep is reproduced."""
        import random

        rng = random.Random(5)
        mismatches = 0
        for _ in range(300):
            p = ap.AdmissionProbe(None, window_s=1e9, boot={"pp_max_micro_batch_size": self.CAP}, clock=lambda: 0.0)
            p.clip_max_new_tokens = CLIP_MAX_NEW_TOKENS
            ap.set_probe(p)
            reqs = [FakeReq(f"q{i}", rng.randint(1, 3000), rng.randint(1, 600),
                            ignore_eos=rng.random() < 0.3) for i in range(rng.randint(1, 6))]
            s = FakeScheduler(running=rng.randint(0, self.CAP), waiting=reqs,
                              kv_pool=rng.randint(10, 20000), cap=self.CAP,
                              max_prefill_tokens=rng.randint(100, 8000))
            s.get_new_batch_prefill()
            mismatches += p.acct.classifier_mismatch
            self.assertNotIn("unclassified", {k.split("/")[0] for k in p.acct.terminal},
                             dict(p.acct.terminal))
        self.assertEqual(mismatches, 0)


# ---------------------------------------------------------------------------
# E. mutation: collapsed-reason variants must FAIL the same scenarios
# ---------------------------------------------------------------------------

class TestMutations(EngineFixture):
    def _fails_with(self, name, new):
        with mock.patch.object(ap, name, new):
            return run_scenarios()

    def test_control_passes(self):
        self.assertEqual(run_scenarios(), [])

    def test_kv_merged_into_cap_fails(self):
        orig = ap.classify_add

        def mutant(snap):
            c, s, r = orig(snap)
            return ("cap" if c == "kv" else c), s, r

        self.assertNotEqual(table_failures(mutant), [])
        self.assertNotEqual(self._fails_with("classify_add", mutant), [])

    def test_all_rejections_one_reason_fails(self):
        orig = ap.resolve_terminal

        def mutant(trace, last):
            c, s, i = orig(trace, last)
            return ("cap" if c in ap.REJECTION_CATEGORIES else c), s, i

        self.assertNotEqual(self._fails_with("resolve_terminal", mutant), [])

    def test_slot_shadow_from_cap_headroom_fails(self):
        def mutant(mamba_avail, cap_headroom_before, n_new, n_reuse=0):
            return None if mamba_avail is None else "identical"

        self.assertNotEqual(self._fails_with("slot_vs_cap", mutant), [])

    def test_sticky_inheritance_dropped_fails(self):
        orig = ap.resolve_terminal

        def mutant(trace, last):
            c, s, i = orig(trace, last)
            return (ap.UNCLASSIFIED, "sticky_unknown", True) if s == "sticky" else (c, s, i)

        self.assertNotEqual(self._fails_with("resolve_terminal", mutant), [])


# ---------------------------------------------------------------------------
# F. default OFF / schema / boot verdicts / emission
# ---------------------------------------------------------------------------

class TestDefaultOff(unittest.TestCase):
    def _snap(self):
        return (PrefillAdder.__dict__["add_one_req"], PrefillAdder.__dict__["__init__"],
                dict(EngineScheduler.__dict__), dict(FakeScheduler.__dict__))

    def test_unset_installs_nothing(self):
        before = self._snap()
        for val in ("", "   "):
            with mock.patch.dict(os.environ, {"PDMUX_ADMISSION_PATH": val}):
                self.assertIsNone(ap.maybe_install_admission_probe(SimpleNamespace()))
        self.assertEqual(before, self._snap())
        self.assertIsNone(ap.get_probe())

    def test_uninstall_restores_everything(self):
        before = self._snap()
        ap.install_wrappers(PrefillAdder, FakeScheduler, CLIP_MAX_NEW_TOKENS)
        self.assertIn("update_split_prefill_batch", FakeScheduler.__dict__)
        self.assertIsNot(PrefillAdder.__dict__["add_one_req"], before[0])
        ap.uninstall_wrappers()
        self.assertEqual(before, self._snap())

    def test_off_probe_none_is_passthrough(self):
        """Wrappers installed but no probe bound: same admission outcome."""
        def run():
            s = FakeScheduler(cap=48, running=47, waiting=[FakeReq("a", 10), FakeReq("b", 10)])
            b = s.get_new_batch_prefill()
            return (len(b.reqs), s.running_batch.batch_is_full, len(s.waiting_queue))

        GLOBAL_ARGS.pp_max_micro_batch_size = 48
        plain = run()
        ap.install_wrappers(PrefillAdder, FakeScheduler, CLIP_MAX_NEW_TOKENS)
        try:
            wrapped_off = run()
            p = ap.AdmissionProbe(None, window_s=1e9, boot={"pp_max_micro_batch_size": 48})
            ap.set_probe(p)
            wrapped_on = run()
        finally:
            ap.set_probe(None)
            ap.uninstall_wrappers()
        self.assertEqual(plain, wrapped_off)
        self.assertEqual(plain, wrapped_on)


class TestExtractionFidelity(unittest.TestCase):
    """Where the engine imports (GPU image), the cut text IS the class source."""

    def test_extracted_names_exist(self):
        self.assertTrue(callable(EngineScheduler.get_new_batch_prefill))
        self.assertEqual(set(_SCHED_SRC), set(SCHED_METHODS))
        self.assertEqual(set(_MIX_SRC), set(MIXIN_METHODS))

    def test_matches_inspect_getsource(self):
        import inspect
        try:
            import sglang.srt.managers.schedule_policy as real_pol
            import sglang.srt.managers.scheduler as real_sched
        except Exception as exc:  # local CPU host: no sgl_kernel
            self.skipTest(f"engine not importable here: {type(exc).__name__}")
        self.assertEqual(inspect.getsource(real_pol.PrefillAdder), _POL_SRC["PrefillAdder"])
        for name in SCHED_METHODS:
            self.assertEqual(_textwrap.dedent(inspect.getsource(getattr(real_sched.Scheduler, name))),
                             _SCHED_SRC[name])


class TestSchemaAndVerdicts(unittest.TestCase):
    def test_window_record_validates(self):
        acct = ap.AdmissionAccountant()
        acct.record_decision("cap", "loop", False)
        acct.record_decision("kv", "sticky", True)
        acct.record_decode(48, 48)
        acct.record_decode(10, 48)
        rec = acct.window_record(1.0)
        rec["event"] = "admission_window"
        self.assertEqual(ap.validate_record(rec), [])
        self.assertEqual(rec["terminal_by_category"]["cap"], 1)
        self.assertEqual(ap.binding_rate([rec]), 0.5)
        self.assertIsNone(ap.binding_rate([]))
        bad = dict(rec, record_schema="pdmux.admission/v0")
        self.assertNotEqual(ap.validate_record(bad), [])
        bad2 = dict(rec, terminal_by_category=dict(rec["terminal_by_category"], mamba_slot=1))
        self.assertNotEqual(ap.validate_record(bad2), [])

    def test_schema_constant(self):
        self.assertEqual(ap.ADMISSION_SCHEMA, "pdmux.admission/v1")
        self.assertEqual(ap.ADMISSION_SCHEMA_VERSION, 1)

    def test_cap_slot_relation(self):
        f = ap.cap_slot_relation
        kw = dict(is_hybrid=True, tree_supports_mamba=False, extra_buffer=False)
        self.assertEqual(f(**kw, pp_max_micro_batch_size=48, req_pool_size=48, mamba_pool_size=48), "IDENTICAL")
        self.assertEqual(f(**kw, pp_max_micro_batch_size=48, req_pool_size=48, mamba_pool_size=96), "SLOT_LOOSER")
        self.assertEqual(f(**kw, pp_max_micro_batch_size=48, req_pool_size=48, mamba_pool_size=40), "SLOT_TIGHTER")
        self.assertEqual(f(is_hybrid=True, tree_supports_mamba=True, extra_buffer=False,
                           pp_max_micro_batch_size=48, req_pool_size=48, mamba_pool_size=144), "RADIX_MAMBA")
        self.assertEqual(f(is_hybrid=False, tree_supports_mamba=False, extra_buffer=False,
                           pp_max_micro_batch_size=48, req_pool_size=48, mamba_pool_size=None), "NOT_HYBRID")

    def test_cap_source(self):
        f = ap.cap_source
        self.assertEqual(f(user_max_running_requests=48, dp_size=1, estimated=2048, mamba_cap=48, realized=48),
                         "mamba+user")
        self.assertEqual(f(user_max_running_requests=256, dp_size=1, estimated=2048, mamba_cap=183, realized=183),
                         "mamba")
        self.assertEqual(f(user_max_running_requests=None, dp_size=1, estimated=2048, mamba_cap=None, realized=2048),
                         "estimate")
        self.assertEqual(f(user_max_running_requests=48, dp_size=1, estimated=2048, mamba_cap=96, realized=47),
                         "unexplained")


class TestEmission(unittest.TestCase):
    def test_jsonl_through_async_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adm.jsonl"
            t = [0.0]
            p = ap.AdmissionProbe(path, run_id="r", workload_id="w", window_s=1.0,
                                  boot={"pp_max_micro_batch_size": 4, "cap_slot_relation": "IDENTICAL"},
                                  clock=lambda: t[0])
            p.acct.record_decision("cap", "loop", False)
            t[0] = 2.0
            p.on_decode_batch(SimpleNamespace(reqs=[1, 2, 3, 4]))  # triggers a window flush
            p.close()
            rows = [json.loads(x) for x in path.read_text().splitlines()]
        events = [r["event"] for r in rows]
        self.assertEqual(events[0], "admission_boot")
        self.assertIn("admission_window", events)
        win = next(r for r in rows if r["event"] == "admission_window")
        self.assertEqual(ap.validate_record(win), [])
        self.assertEqual(win["n_decode_iter_at_or_above_cap"], 1)
        self.assertEqual(win["terminal_by_category"]["cap"], 1)
        self.assertEqual(rows[0]["schema_version"], 1)


# ---------------------------------------------------------------------------
# G. scheduler hook patch
# ---------------------------------------------------------------------------

class TestHookPatch(unittest.TestCase):
    def setUp(self):
        self.installed = RUNTIME / "sglang/srt/managers/scheduler.py"
        if not self.installed.is_file():
            self.skipTest(f"runtime tree absent: {self.installed}")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.target = Path(self.tmp) / "sglang/srt/managers/scheduler.py"
        self.target.parent.mkdir(parents=True)
        shutil.copy(self.installed, self.target)

    def _patch(self, pf, direction):
        with open(pf, "rb") as fh:
            r = subprocess.run(["patch", direction, "--batch", "-p1", "-d", self.tmp],
                               stdin=fh, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode() + r.stdout.decode())

    def test_forward_then_reverse_is_identity(self):
        orig = self.target.read_text()
        if "maybe_install_admission_probe" in orig:
            self.skipTest("runtime already carries the hook")
        self._patch(PATCH, "--forward")
        self.assertIn("maybe_install_admission_probe", self.target.read_text())
        self._patch(PATCH, "--reverse")
        self.assertEqual(self.target.read_text(), orig)

    def test_composes_with_holb_and_chunk(self):
        orig = self.target.read_text()
        if "maybe_install_admission_probe" in orig:
            self.skipTest("runtime already carries the hook")
        stack = [ENG / "src/patches/holb_probe_scheduler_hooks.patch",
                 ENG / "src/patches/chunk_probe_scheduler_hook.patch", PATCH]
        self._patch(PATCH, "--forward")
        full = self.target.read_text()
        for pf in reversed(stack):
            self._patch(pf, "--reverse")
        for pf in stack:
            self._patch(pf, "--forward")
        self.assertEqual(self.target.read_text(), full)

    def test_hook_inside_run_scheduler_process_and_class_lines_unmoved(self):
        orig = self.target.read_text()
        if "maybe_install_admission_probe" in orig:
            self.skipTest("runtime already carries the hook")
        self._patch(PATCH, "--forward")
        new = self.target.read_text()
        tree = ast.parse(new)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_scheduler_process")
        self.assertIn("maybe_install_admission_probe", ast.get_source_segment(new, fn))
        cls_old = next(n for n in ast.parse(orig).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        cls_new = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        self.assertEqual((cls_old.lineno, cls_old.end_lineno), (cls_new.lineno, cls_new.end_lineno))
        self.assertEqual(orig.splitlines()[: cls_old.end_lineno], new.splitlines()[: cls_new.end_lineno])


if __name__ == "__main__":
    unittest.main()
