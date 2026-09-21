"""Gate for the chunked-prefill probe (CP-0 prerequisite P1).

WHY THIS EXISTS.  ``PREREG_CP0_2026-08-28.md`` section 2 records that the tree
had *no* observation of chunked prefill outside the pdmux mixin, and that three
rule-layer audits failed to notice.  Section 2.1 then fixes ONE definition for
channel 1 because three candidates were in play and *all three pass the obvious
acceptance tests*:

    (1) number of requests that were split
    (2) number of chunk events
    (3) number of prompts longer than chunked_prefill_size

What is pinned here:

  1. the accountant's arithmetic: distinct-rid semantics, the additive split of
     ``n_chunk_events`` into admission + continuation, the isolation of the
     diffusion-LLM truncation into its own contamination counter, and the
     channel-2 max/mean/n_forwards roll-up;

  2. **mechanism coverage as a closed list, checked against the installed
     runtime** (not against a copy).  ``test_mechanism_list_is_closed`` parses
     the real ``schedule_policy.py`` with ``ast`` and fails if SGLang grows a
     truncation site this probe does not wrap.  Completeness is judged from the
     mechanism list, not from a hand-maintained site list;

  3. that the wrappers fire on the **real** ``PrefillAdder`` -- the tests drive
     the actual engine class (with fake caches/pools), so the control flow under
     test is the engine's, not a mirror of it.  Both admission mechanisms are
     exercised: ``ignore_eos=False`` reaches ``add_one_req``'s own else-branch,
     ``ignore_eos=True`` reaches ``add_one_req_ignore_eos``, and the delegation
     between them must be counted exactly once;

  4. the pre-registered P1-e separation: the hand-computed expectations in
     ``results/cp_baseline/p1e_expectations.json`` are reproduced by the real
     ``PrefillAdder`` for both phases, and the three candidate definitions come
     out to three different numbers;

  5. a **mutation test**: with the continuation wrapper removed, the P1-e
     sequential expectation must FAIL.  A repair that nothing depends on is not
     a repair -- PROJECT_STATUS.md "방법론 게이트" #50 ("자기 수리를 검증하는
     검사 자신이 반증 불가능한 항등식일 수 있다"), whose registered remedy is
     "수리를 검증하는 검사는 그 수리를 되돌린 변이본에서 반드시 실패해야 한다";

  6. that the probe is OFF unless ``PDMUX_CHUNK_PROBE_PATH`` is set, and that
     when OFF **no wrapper is installed at all** -- not merely that the probe
     object is None.

WHAT THIS FILE CANNOT CHECK.  Everything downstream of the scheduler: that the
piecewise CUDA graph really is disabled on the ``--chunked-prefill-size -1``
arm, that the probe is performance neutral, and that the ON/OFF outputs are bit
identical.  Those are P1-c / P1-f / P1-g and need a GPU
(``results/cp_baseline/p1_accept.sbatch``).

The module is loaded from the installed runtime tree (same convention as
test_dual_worker.py / test_holb_probe.py), so this also checks that
sync_engine_tree.sh actually installed it.
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV")
    or os.path.join(os.environ.get("PDMUX_ROOT", str(Path(__file__).resolve().parents[4])),
                     "sglang_engine_dev", "python")
)
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

REPO = Path(__file__).resolve().parents[3]
EXPECT_PATH = REPO / "workspace/engine-port/results/cp_baseline/p1e_expectations.json"


def _install_stub_global_server_args():
    """`PrefillAdder.__init__` reads two global-server-args predicates.

    Both are pure attribute reads (`cp_utils.is_prefill_context_parallel_enabled`,
    `nsa/utils.is_nsa_prefill_cp_in_seq_split`), so a stub is enough to construct
    the real class on a CPU-only host.  Nothing else in these tests touches it.
    """
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        SimpleNamespace(
            enable_prefill_context_parallel=False,
            prefill_cp_mode="none",
            enable_nsa_prefill_context_parallel=False,
            nsa_prefill_cp_mode="none",
        )
    )


_install_stub_global_server_args()

from sglang.srt.managers.schedule_policy import (  # noqa: E402
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.multiplex.chunk_probe import (  # noqa: E402
    CLASS_NONPREFILL,
    CLASS_PREFILL,
    CLASS_UNKNOWN,
    NONPREFILL_FORWARD_MODES,
    PREFILL_FORWARD_MODES,
    classify_forward_mode,
    MECH_ADMISSION_IGNORE_EOS,
    MECH_ADMISSION_STD,
    MECH_CONTINUATION,
    MECH_DLLM,
    ChunkAccountant,
    ChunkProbe,
    install_wrappers,
    maybe_install_chunk_probe,
    set_probe,
    uninstall_wrappers,
)

SCHEDULE_POLICY_SRC = RUNTIME / "sglang/srt/managers/schedule_policy.py"


# ---------------------------------------------------------------------------
# CPU stand-ins.  The *adder* is the real engine class; only the things it
# reaches out to (caches, KV pool, request objects) are stubbed.
# ---------------------------------------------------------------------------


class FakeTreeCache:
    """A ChunkCache stand-in: no prefix matching, `disable` hardcoded True.

    Mirrors sglang/srt/mem_cache/chunk_cache.py, whose `disable` property
    returns True unconditionally -- that is what routes `ignore_eos` requests
    into `add_one_req_ignore_eos`.
    """

    def __init__(self, disable=True):
        self.disable = disable

    def supports_mamba(self):
        return False

    def supports_swa(self):
        return False

    def is_tree_cache(self):
        return False

    def evictable_size(self):
        return 0

    def inc_lock_ref(self, node):
        return SimpleNamespace(swa_uuid_for_lock=None)

    def dec_lock_ref(self, node, params=None):
        return None


class FakeAllocator:
    def __init__(self, size=10**9):
        self._size = size

    def available_size(self):
        return self._size


class FakeReq:
    """The attribute surface `PrefillAdder` touches, plus the chunk bookkeeping.

    `init_next_round_input` reproduces schedule_batch.py:949 + :1021 for the
    `tree_cache is None` call the scheduler makes on a continuing chunked
    request (scheduler.py:2408): fill_ids are restored to the full prompt and
    the extend length becomes what is left after the committed prefix.
    """

    def __init__(self, rid, n_tokens, ignore_eos=True, max_new_tokens=8):
        self.rid = rid
        self.origin_input_ids = list(range(1000, 1000 + n_tokens))
        self.output_ids = []
        self.fill_ids = list(self.origin_input_ids)
        self.prefix_indices = []
        self.extend_input_len = n_tokens
        self.host_hit_length = 0
        self.last_node = None
        self.sampling_params = SimpleNamespace(
            ignore_eos=ignore_eos, max_new_tokens=max_new_tokens
        )
        self.is_chunked = 0
        self.logprob_start_len = -1
        self.return_logprob = False
        self.cache_protected_len = 0
        self.mamba_pool_idx = None
        self.lora_id = None

    def set_extend_input_len(self, n):
        self.extend_input_len = n

    def init_next_round_input(self, tree_cache=None):
        self.fill_ids = list(self.origin_input_ids) + list(self.output_ids)
        self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))


def make_adder(cps, page_size=1, tree_cache=None, pool=10**9):
    return PrefillAdder(
        page_size,
        tree_cache if tree_cache is not None else FakeTreeCache(),
        FakeAllocator(pool),
        SimpleNamespace(reqs=[]),
        1.0,  # new_token_ratio
        1 << 30,  # rem_input_tokens (max_prefill_tokens)
        cps,  # rem_chunk_tokens
    )


def run_prefill_loop(reqs, cps, page_size=1, max_batches=200):
    """CPU model of the scheduler's prefill loop around the REAL PrefillAdder.

    Reproduces only the chunked-request plumbing of
    `Scheduler.get_new_batch_prefill` (scheduler.py:2407-2409 and :2497-2500)
    and the ChunkCache commit that grows `prefix_indices`
    (chunk_cache.py:76-81).  Everything that decides *whether* a request is
    truncated is the real engine code.

    NOTE ON SCOPE: this loop is a mirror of the scheduler, so a defect in the
    scheduler's own plumbing would not show up here.  It is the *PrefillAdder*
    branch structure that is under test, plus the probe wrappers around it.
    """
    queue = list(reqs)
    chunked_req = None
    forwards = []
    for _ in range(max_batches):
        if not queue and chunked_req is None:
            break
        adder = make_adder(cps, page_size=page_size)
        if chunked_req is not None:
            chunked_req.init_next_round_input()
            chunked_req = adder.add_chunked_req(chunked_req)
        while queue:
            req = queue[0]
            req.init_next_round_input()
            res = adder.add_one_req(
                req, has_chunked_req=(chunked_req is not None),
                truncation_align_size=None,
            )
            added = bool(adder.can_run_list) and req is adder.can_run_list[-1]
            if added:
                queue.pop(0)
            if res != AddReqResult.CONTINUE:
                break
        if adder.new_chunked_req is not None:
            chunked_req = adder.new_chunked_req
        if not adder.can_run_list:
            break
        forwards.append(sum(r.extend_input_len for r in adder.can_run_list))
        # ChunkCache.cache_unfinished_req: the committed prefix becomes
        # everything processed so far.
        for r in adder.can_run_list:
            r.prefix_indices = list(range(len(r.fill_ids)))
    return forwards


class ProbeFixture(unittest.TestCase):
    """Installs the wrappers around the real PrefillAdder for one test."""

    def setUp(self):
        self.probe = ChunkProbe(None, arm="test")  # path None -> writer is a no-op
        set_probe(self.probe)
        install_wrappers(PrefillAdder, None)
        self.addCleanup(self._teardown)

    def _teardown(self):
        uninstall_wrappers(PrefillAdder, None)
        set_probe(None)

    @property
    def acct(self):
        return self.probe.acct


# ---------------------------------------------------------------------------
# 1. accountant arithmetic
# ---------------------------------------------------------------------------


class TestAccountantArithmetic(unittest.TestCase):
    def test_distinct_rids_vs_events(self):
        a = ChunkAccountant()
        a.record_chunk("r1", MECH_ADMISSION_IGNORE_EOS, 1500, 512)
        a.record_chunk("r1", MECH_CONTINUATION, 988, 512)
        a.record_chunk("r2", MECH_ADMISSION_IGNORE_EOS, 513, 512)
        self.assertEqual(a.n_requests_chunked, 2)
        self.assertEqual(a.n_chunk_events, 3)
        self.assertEqual(a.n_chunk_events_admission, 2)
        self.assertEqual(a.n_chunk_events_continuation, 1)
        self.assertEqual(
            a.n_chunk_events,
            a.n_chunk_events_admission + a.n_chunk_events_continuation,
        )

    def test_admission_only_makes_the_two_counters_an_identity(self):
        """Documented in chunk_probe.py: without M3 the prereg's (2) >= (1) is
        unreachable, because each admission site fires at most once per rid."""
        a = ChunkAccountant()
        for rid in ("r1", "r2", "r3"):
            a.record_chunk(rid, MECH_ADMISSION_STD, 900, 512)
        self.assertEqual(a.n_chunk_events, a.n_requests_chunked)

    def test_dllm_is_not_folded_into_the_registered_counters(self):
        a = ChunkAccountant()
        a.record_chunk("r1", MECH_DLLM, 100, 32)
        self.assertEqual(a.n_dllm_trunc_events, 1)
        self.assertEqual(a.n_chunk_events, 0)
        self.assertEqual(a.n_requests_chunked, 0)

    def test_non_shrinking_event_is_counted_not_swallowed(self):
        a = ChunkAccountant()
        a.record_chunk("r1", MECH_ADMISSION_STD, 512, 512)
        self.assertEqual(a.n_events_without_shrink, 1)
        self.assertEqual(a.n_chunk_events, 1)

    def test_channel2_rollup(self):
        a = ChunkAccountant()
        for mode, ext in (
            ("EXTEND", 512), ("DECODE", 0), ("EXTEND", 512),
            ("DECODE", None), ("EXTEND", 476),
        ):
            a.record_forward(mode, ext)
        s = a.summary()["extend_tokens_per_forward"]
        self.assertEqual(s["n_forwards"], 3)
        self.assertEqual(s["max"], 512)
        self.assertEqual(s["min"], 476)
        self.assertEqual(s["sum"], 1500)
        self.assertAlmostEqual(s["mean"], 500.0)
        self.assertEqual(a.summary()["n_forwards_total"], 5)
        self.assertEqual(a.summary()["n_nonprefill_forwards"], 2)

    def test_empty_boot_reports_none_not_zero(self):
        s = ChunkAccountant().summary()["extend_tokens_per_forward"]
        self.assertIsNone(s["max"])
        self.assertIsNone(s["mean"])
        self.assertEqual(s["n_forwards"], 0)


class TestForwardModeClassification(unittest.TestCase):
    """Repair 1 (2026-09-01): channel 2 counts PREFILL forwards only.

    On job 899768's cps512 boot the probe wrote 116 ``prefill_forward`` lines,
    101 of which had ``forward_mode == "DECODE"``.  ``ScheduleBatch`` does not
    clear ``extend_num_tokens`` when the object is reused for a decode step, so
    ``record_forward``'s ``ext > 0`` test was reading a STALE field: the DECODE
    records in that file carry the extend count of the preceding prefill
    (``mode=DECODE, extend_num_tokens=7, batch_size=1`` is a verbatim example).
    The registered P1-e numbers are per *prefill* forward, so the counter and
    the expectation were not the same quantity.

    The repair filters on ``forward_mode``.  The non-prefill records are NOT
    dropped: they are written to the raw JSONL as ``nonprefill_forward`` with
    ``is_prefill: false``, because those records are the evidence that the
    defect existed and deleting them would erase the diagnosis.  Only the
    AGGREGATES exclude them.
    """

    def _modes(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return ForwardMode

    def test_the_partition_is_the_engine_predicate_not_a_guess(self):
        """`PREFILL_FORWARD_MODES` must be the engine's own `is_extend`.

        Hand-listing the modes here would be a second source of truth; the
        engine's predicate is the first one.  `include_draft_extend_v2=True`
        is the maximal reading and is deliberate: channel 2 asks "did this
        forward carry extend tokens", and DRAFT_EXTEND_V2 does.  No CP-0 arm
        enables speculative decoding, so the difference is unreachable there.
        """
        ForwardMode = self._modes()
        for m in ForwardMode:
            want = CLASS_PREFILL if m.is_extend(include_draft_extend_v2=True) \
                else CLASS_NONPREFILL
            self.assertEqual(
                classify_forward_mode(m.name), want,
                f"{m.name} is classified against the engine's own predicate",
            )

    def test_the_partition_is_total_over_the_engine_enum(self):
        """No enum member may fall through to `unknown`.

        PROJECT_STATUS.md "방법론 게이트" #66: fix the decision rule in code and
        enumerate the world.  If SGLang grows a forward mode, this fails here on
        CPU instead of silently emptying channel 2 on the GPU.
        """
        ForwardMode = self._modes()
        names = {m.name for m in ForwardMode}
        self.assertEqual(
            names - (PREFILL_FORWARD_MODES | NONPREFILL_FORWARD_MODES), set(),
            "a ForwardMode member is in neither set",
        )
        self.assertEqual(
            PREFILL_FORWARD_MODES & NONPREFILL_FORWARD_MODES, frozenset(),
            "the two sets must be disjoint",
        )
        self.assertEqual(
            (PREFILL_FORWARD_MODES | NONPREFILL_FORWARD_MODES) - names, set(),
            "the probe names a mode the engine does not have",
        )

    def test_a_decode_forward_carrying_a_stale_extend_count_is_not_channel_2(self):
        """The exact shape of job 899768's file."""
        a = ChunkAccountant()
        a.record_forward("EXTEND", 512)
        for _ in range(3):
            a.record_forward("DECODE", 512)   # stale: the same 512, batch of 1
        s = a.summary()
        self.assertEqual(s["extend_tokens_per_forward"]["n_forwards"], 1)
        self.assertEqual(s["extend_tokens_per_forward"]["sum"], 512)
        self.assertEqual(s["extend_tokens_per_forward"]["max"], 512)
        self.assertEqual(s["n_nonprefill_forwards_with_extend_tokens"], 3,
                         "the stale-value forwards must be counted, not ignored")
        self.assertEqual(s["sum_extend_tokens_not_counted"], 1536)

    def test_n_prefill_forwards_is_not_n_forwards_total_when_decode_ran(self):
        """The regression this repair exists for.

        Before 2026-09-01 every DECODE forward with a stale positive
        `extend_num_tokens` was counted as a prefill forward, which made
        `n_prefill_forwards == n_forwards_total` and `n_nonprefill_forwards == 0`
        on a boot that plainly decoded.  If those two numbers coincide again on
        this input, the filter has been removed.
        """
        a = ChunkAccountant()
        for mode, ext in (("EXTEND", 512), ("DECODE", 512), ("DECODE", 512),
                          ("EXTEND", 100), ("DECODE", 100)):
            a.record_forward(mode, ext)
        s = a.summary()
        self.assertEqual(s["n_forwards_total"], 5)
        self.assertEqual(s["extend_tokens_per_forward"]["n_forwards"], 2)
        self.assertLess(s["extend_tokens_per_forward"]["n_forwards"],
                        s["n_forwards_total"])
        self.assertEqual(s["n_nonprefill_forwards"], 3)
        self.assertNotEqual(s["n_nonprefill_forwards"], 0)

    def test_unknown_forward_mode_is_counted_not_dropped(self):
        """A mode the probe cannot classify is a hole in the instrument.

        It must not be silently counted as prefill (it could be a decode
        carrying a stale field) nor silently discarded (that would hide the
        hole).  It gets its own counter, and the reader warns on it.
        """
        a = ChunkAccountant()
        a.record_forward("EXTEND", 512)
        a.record_forward(None, 512)
        a.record_forward("None", 512)
        a.record_forward("A_MODE_THAT_DOES_NOT_EXIST", 512)
        s = a.summary()
        self.assertEqual(s["n_unknown_mode_forwards"], 3)
        self.assertEqual(s["extend_tokens_per_forward"]["n_forwards"], 1)
        self.assertEqual(s["n_forwards_total"], 4)

    def test_every_forward_lands_in_exactly_one_cell(self):
        """mode_class x has-extend-tokens is enumerated, and the cells close."""
        a = ChunkAccountant()
        for mode, ext in (("EXTEND", 512), ("EXTEND", 0), ("DECODE", 7),
                          ("DECODE", None), ("IDLE", 0), ("WAT", 3)):
            a.record_forward(mode, ext)
        s = a.summary()
        cells = s["forward_cells"]
        self.assertEqual(sum(cells.values()), s["n_forwards_total"])
        self.assertEqual(cells["prefill/ext"], 1)
        self.assertEqual(cells["prefill/noext"], 1)
        self.assertEqual(cells["nonprefill/ext"], 1)
        self.assertEqual(cells["nonprefill/noext"], 2)
        self.assertEqual(cells["unknown/ext"], 1)
        self.assertEqual(
            s["extend_tokens_per_forward"]["n_forwards"], cells["prefill/ext"],
            "channel 2 is exactly the prefill-mode-with-extend-tokens cell",
        )

    def test_a_stale_decode_forward_stays_in_the_raw_jsonl(self):
        """Excluded from the aggregates, kept in the record.

        The instruction that produced this repair is explicit: the DECODE lines
        were the evidence, so the fix may not delete them.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk.jsonl"
            probe = ChunkProbe(path, run_id="rid", arm="cps512")
            probe.on_forward("EXTEND", 512, 1, True)
            probe.on_forward("DECODE", 512, 1, False)   # stale extend count
            probe.on_forward("DECODE", 0, 4, False)     # nothing to record
            probe.close()
            rows = [json.loads(l) for l in path.read_text().splitlines()]
        by_event = {}
        for r in rows:
            by_event.setdefault(r["event"], []).append(r)
        self.assertEqual(len(by_event["prefill_forward"]), 1)
        self.assertEqual(len(by_event.get("nonprefill_forward", [])), 1,
                         "the stale DECODE forward must survive in the raw data")
        kept = by_event["nonprefill_forward"][0]
        self.assertFalse(kept["is_prefill"])
        self.assertEqual(kept["forward_mode"], "DECODE")
        self.assertEqual(kept["extend_num_tokens"], 512,
                         "the stale value itself is the evidence; keep it")
        self.assertTrue(by_event["prefill_forward"][0]["is_prefill"])
        summary = by_event["chunk_summary"][-1]
        self.assertEqual(summary["extend_tokens_per_forward"]["n_forwards"], 1)
        self.assertEqual(summary["n_forwards_total"], 3)


# ---------------------------------------------------------------------------
# 2. mechanism coverage, checked against the installed runtime
# ---------------------------------------------------------------------------


class TestMechanismCoverage(unittest.TestCase):
    """Completeness judged from the mechanism list, against the real file.

    If an SGLang bump adds a truncation site inside `PrefillAdder`, this test
    fails.  That is the point: an instrument that silently stops covering the
    engine is worse than no instrument.
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCHEDULE_POLICY_SRC.read_text())
        cls.cls_node = next(
            n for n in ast.walk(cls.tree)
            if isinstance(n, ast.ClassDef) and n.name == "PrefillAdder"
        )

    def _methods_containing(self, predicate):
        found = set()
        for meth in self.cls_node.body:
            if not isinstance(meth, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(meth):
                if predicate(node):
                    found.add(meth.name)
                    break
        return found

    def test_new_chunked_req_is_assigned_only_at_the_two_admission_sites(self):
        """The postcondition the wrappers read must be unique to truncation."""

        def is_assign(node):
            if not isinstance(node, ast.Assign):
                return False
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "new_chunked_req"
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    return True
            return False

        methods = self._methods_containing(is_assign)
        self.assertEqual(
            methods,
            {"__init__", "add_one_req", "add_one_req_ignore_eos"},
            "self.new_chunked_req gained an assignment site; the probe's "
            "admission postcondition may no longer be unique to truncation",
        )

    def test_set_extend_input_len_sites_are_the_enumerated_ones(self):
        def is_call(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_extend_input_len"
            )

        methods = self._methods_containing(is_call)
        self.assertEqual(
            methods,
            {"add_chunked_req", "add_one_req", "add_one_req_ignore_eos"},
            "a new set_extend_input_len site appeared inside PrefillAdder; "
            "re-derive the mechanism list in chunk_probe.py before trusting "
            "any chunk count",
        )

    def test_direct_extend_input_len_writes_are_only_the_dllm_ones(self):
        def is_direct_write(node):
            if not isinstance(node, ast.Assign):
                return False
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "extend_input_len":
                    return True
            return False

        methods = self._methods_containing(is_direct_write)
        self.assertEqual(
            methods,
            {"_add_dllm_req", "add_dllm_staging_req"},
            "extend_input_len is now written directly outside the dLLM paths",
        )

    def test_every_enumerated_mechanism_is_wrapped(self):
        install_wrappers(PrefillAdder, None)
        try:
            got = {
                name: getattr(
                    getattr(PrefillAdder, name), "_chunk_probe_mechanism", None
                )
                for name in (
                    "add_one_req",
                    "add_one_req_ignore_eos",
                    "add_chunked_req",
                    "_add_dllm_req",
                    "add_dllm_staging_req",
                )
            }
        finally:
            uninstall_wrappers(PrefillAdder, None)
        self.assertEqual(
            got,
            {
                "add_one_req": MECH_ADMISSION_STD,
                "add_one_req_ignore_eos": MECH_ADMISSION_IGNORE_EOS,
                "add_chunked_req": MECH_CONTINUATION,
                "_add_dllm_req": MECH_DLLM,
                "add_dllm_staging_req": MECH_DLLM,
            },
        )

    # (file, enclosing qualname) -> classification.  This is the MECHANISM LIST
    # for the whole installed runtime, not just PrefillAdder: every place in
    # sglang/srt that can change a request's extend length, and why the probe
    # does or does not count it.  Discovered by AST, so a new site anywhere in
    # the engine fails this test instead of silently escaping the counters
    # (PROJECT_STATUS.md "방법론 게이트" #78: "계측 완전성은 사이트 목록이
    # 아니라 기전 목록에서 판정하라 -- `batch_is_full`을 세우는 자리와
    # admission을 막는 기전은 다른 집합이다").  NOTE ON THE COORDINATE: #78 is
    # declared at PROJECT_STATUS.md:5778 ("...#78 신설[계측 완전성은 사이트
    # 목록이 아니라 기전 목록에서 판정하라 ... CONSENSUS §3 항목98과 대응]") but
    # the enumerated gate list in that file still stops at #70, so the canonical
    # full text is CONSENSUS.md §3 항목98.
    EXTEND_LEN_SITES = {
        ("managers/schedule_policy.py", "PrefillAdder.add_one_req"):
            "M1 admission truncation (COUNTED) + the hierarchical-cache "
            "load-back recompute, which can only grow the extend length",
        ("managers/schedule_policy.py", "PrefillAdder.add_one_req_ignore_eos"):
            "M2 admission truncation (COUNTED)",
        ("managers/schedule_policy.py", "PrefillAdder.add_chunked_req"):
            "M3 continuation truncation (COUNTED)",
        ("managers/schedule_policy.py", "PrefillAdder._add_dllm_req"):
            "M4 dLLM block truncation (separate contamination counter)",
        ("managers/schedule_policy.py", "PrefillAdder.add_dllm_staging_req"):
            "M4 dLLM staging truncation (separate contamination counter)",
        ("managers/schedule_batch.py", "Req.__init__"):
            "initialisation to 0",
        ("managers/schedule_batch.py", "Req.set_extend_input_len"):
            "the setter itself",
        ("managers/schedule_batch.py", "Req.init_next_round_input"):
            "recompute of what is left after the committed prefix",
        ("managers/schedule_batch.py", "Req.reset_for_retract"):
            "retraction resets to 0; a retracted request is re-admitted later "
            "and its next truncation is counted then",
        ("managers/schedule_batch.py", "ScheduleBatch.mix_with_running"):
            "--enable-mixed-chunk only: sets 1 for each running DECODE request "
            "folded into an extend batch. Not a prefill split; it is why "
            "enable_mixed_chunk is echoed and warned about",
        ("managers/scheduler_pp_mixin.py", "SchedulerPPMixin.profile_and_init_predictor"):
            "pipeline-parallel profiling helper (pp_size > 1)",
        ("disaggregation/decode.py", "DecodePreallocQueue._pre_alloc"):
            "PD-disaggregation decode side; sets the FULL fill length",
    }
    COUNTED = {
        "PrefillAdder.add_one_req",
        "PrefillAdder.add_one_req_ignore_eos",
        "PrefillAdder.add_chunked_req",
    }

    @staticmethod
    def _scan_extend_len_sites():
        found = {}
        for f in sorted((RUNTIME / "sglang" / "srt").rglob("*.py")):
            txt = f.read_text(errors="replace")
            if "extend_input_len" not in txt:
                continue
            try:
                tree = ast.parse(txt)
            except SyntaxError:  # pragma: no cover
                continue
            rel = str(f.relative_to(RUNTIME / "sglang" / "srt"))
            stack = []

            def walk(node):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        stack.append(child.name)
                        walk(child)
                        stack.pop()
                        continue
                    hit = False
                    if (isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Attribute)
                            and child.func.attr == "set_extend_input_len"):
                        hit = True
                    if isinstance(child, (ast.Assign, ast.AugAssign)):
                        targets = (child.targets if isinstance(child, ast.Assign)
                                   else [child.target])
                        for t in targets:
                            if (isinstance(t, ast.Attribute)
                                    and t.attr == "extend_input_len"):
                                hit = True
                    if hit:
                        found.setdefault((rel, ".".join(stack)), 0)
                        found[(rel, ".".join(stack))] += 1
                    walk(child)

            walk(tree)
        return found

    def test_mechanism_list_is_closed_over_the_whole_runtime(self):
        found = set(self._scan_extend_len_sites())
        registered = set(self.EXTEND_LEN_SITES)
        new = found - registered
        gone = registered - found
        self.assertFalse(
            new,
            "SGLang gained a site that can change a request's extend length and "
            "it is not in the mechanism list: "
            + "; ".join(f"{a}:{b}" for a, b in sorted(new))
            + ". Classify it (truncation? recompute?) and update chunk_probe.py "
              "before trusting any chunk count.",
        )
        self.assertFalse(
            gone,
            "a registered site disappeared, so the mechanism list describes a "
            "tree that no longer exists: "
            + "; ".join(f"{a}:{b}" for a, b in sorted(gone)),
        )

    def test_every_counted_site_lives_in_a_wrapped_method(self):
        methods = {q.split(".", 1)[1] for f, q in self.EXTEND_LEN_SITES
                   if q in self.COUNTED}
        install_wrappers(PrefillAdder, None)
        try:
            for name in methods:
                self.assertTrue(
                    hasattr(getattr(PrefillAdder, name), "_chunk_probe_wrapped"),
                    f"{name} holds a counted truncation but is not wrapped",
                )
        finally:
            uninstall_wrappers(PrefillAdder, None)

    def test_uninstall_restores_the_engine_functions(self):
        pristine = PrefillAdder.add_one_req
        install_wrappers(PrefillAdder, None)
        self.assertIsNot(PrefillAdder.add_one_req, pristine)
        uninstall_wrappers(PrefillAdder, None)
        self.assertIs(PrefillAdder.add_one_req, pristine)


# ---------------------------------------------------------------------------
# 3. the wrappers on the real PrefillAdder
# ---------------------------------------------------------------------------


class TestRealPrefillAdder(ProbeFixture):
    def test_M2_ignore_eos_admission_truncation(self):
        adder = make_adder(512)
        req = FakeReq("r1", 1500, ignore_eos=True)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(req.extend_input_len, 512)
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(self.acct.n_chunk_events, 1)
        self.assertEqual(self.acct.n_requests_chunked, 1)
        self.assertEqual(
            self.acct.mech_counts[MECH_ADMISSION_IGNORE_EOS], 1,
            "ignore_eos requests must be attributed to add_one_req_ignore_eos",
        )
        self.assertEqual(self.acct.mech_counts[MECH_ADMISSION_STD], 0)

    def test_M1_standard_admission_truncation(self):
        adder = make_adder(512)
        req = FakeReq("r1", 1500, ignore_eos=False)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(req.extend_input_len, 512)
        self.assertEqual(self.acct.mech_counts[MECH_ADMISSION_STD], 1)
        self.assertEqual(self.acct.mech_counts[MECH_ADMISSION_IGNORE_EOS], 0)

    def test_delegation_is_counted_once_not_twice(self):
        """`add_one_req` calls `add_one_req_ignore_eos`; both are wrapped."""
        adder = make_adder(512)
        adder.add_one_req(
            FakeReq("r1", 1500, ignore_eos=True),
            has_chunked_req=False, truncation_align_size=None,
        )
        self.assertEqual(self.acct.n_chunk_events, 1)
        self.assertEqual(sum(self.acct.mech_counts.values()), 1)

    def test_M3_continuation_truncation(self):
        adder = make_adder(512)
        req = FakeReq("r1", 1500, ignore_eos=True)
        req.prefix_indices = list(range(512))
        req.init_next_round_input()
        self.assertEqual(req.extend_input_len, 988)
        out = adder.add_chunked_req(req)
        self.assertIs(out, req, "988 > 512 so the engine reports it truncated")
        self.assertEqual(self.acct.mech_counts[MECH_CONTINUATION], 1)
        self.assertEqual(self.acct.n_chunk_events_continuation, 1)

    def test_last_chunk_is_not_an_event(self):
        adder = make_adder(512)
        req = FakeReq("r1", 1500, ignore_eos=True)
        req.prefix_indices = list(range(1024))
        req.init_next_round_input()
        self.assertEqual(req.extend_input_len, 476)
        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual(self.acct.n_chunk_events, 0)

    def test_exactly_at_the_budget_is_not_chunked(self):
        """The engine compares with `<=`; 512 tokens at cps 512 is one chunk."""
        adder = make_adder(512)
        req = FakeReq("r1", 512, ignore_eos=True)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(self.acct.n_chunk_events, 0)

    def test_fused_mono_arm_never_chunks(self):
        """`--chunked-prefill-size -1` -> Scheduler maps it to None (P1-b)."""
        forwards = run_prefill_loop(
            [FakeReq(f"r{i}", n) for i, n in enumerate([1500, 512, 513, 100])],
            cps=None,
        )
        self.assertEqual(self.acct.n_chunk_events, 0)
        self.assertEqual(self.acct.n_requests_chunked, 0)
        self.assertGreater(
            max(forwards), 512,
            "P1-b also requires the un-chunked arm to exceed 512 tokens in a "
            "single forward",
        )

    def test_page_alignment_differs_between_the_two_admission_mechanisms(self):
        """M1 aligns trunc_len down to a page, M2 does not.  With page_size 8
        and 115 tokens of budget left, M1 gives 112 and M2 gives 115."""
        for ignore_eos, expected in ((False, 112), (True, 115)):
            with self.subTest(ignore_eos=ignore_eos):
                adder = make_adder(512, page_size=8)
                adder.rem_chunk_tokens = 115
                req = FakeReq("r", 400, ignore_eos=ignore_eos)
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                self.assertEqual(req.extend_input_len, expected)


# ---------------------------------------------------------------------------
# 4. the pre-registered P1-e separation
# ---------------------------------------------------------------------------


def _phase(expectations, phase_id):
    return next(p for p in expectations["phases"] if p["id"] == phase_id)


class TestP1eSeparation(ProbeFixture):
    @classmethod
    def setUpClass(cls):
        cls.exp = json.loads(EXPECT_PATH.read_text())

    def _run_phase(self, phase, ignore_eos=True):
        reqs = [
            FakeReq(f"{phase['id']}_{i}", n, ignore_eos=ignore_eos)
            for i, n in enumerate(phase["prompt_lens"])
        ]
        cps = phase["chunked_prefill_size"]
        if phase["id"] == "S":
            forwards = []
            for r in reqs:  # concurrency 1: one request at a time
                forwards.extend(run_prefill_loop([r], cps))
        else:
            forwards = run_prefill_loop(reqs, cps)  # full backlog
        return forwards

    def _assert_phase(self, phase, forwards):
        e = phase["expected"]
        self.assertEqual(self.acct.n_requests_chunked, e["n_requests_chunked"])
        self.assertEqual(self.acct.n_chunk_events, e["n_chunk_events"])
        self.assertEqual(
            self.acct.n_chunk_events_admission, e["n_chunk_events_admission"]
        )
        self.assertEqual(
            self.acct.n_chunk_events_continuation, e["n_chunk_events_continuation"]
        )
        self.assertEqual(forwards, e["extend_tokens_per_forward_multiset"])
        self.assertEqual(len(forwards), e["n_prefill_forwards"])
        self.assertEqual(sum(forwards), e["sum_extend_tokens"])
        self.assertEqual(max(forwards), e["max_extend_tokens"])
        # channel 2 as the probe itself would report it
        n_long = sum(
            1 for n in phase["prompt_lens"] if n > phase["chunked_prefill_size"]
        )
        self.assertEqual(
            n_long, e["n_prompts_longer_than_cps"],
            "the offline lower bound in the expectations file is wrong",
        )

    def test_phase_S_sequential_matches_hand_calculation(self):
        phase = _phase(self.exp, "S")
        self._assert_phase(phase, self._run_phase(phase))

    def test_phase_S_also_holds_for_the_standard_admission_mechanism(self):
        phase = _phase(self.exp, "S")
        forwards = self._run_phase(phase, ignore_eos=False)
        self._assert_phase(phase, forwards)
        self.assertEqual(
            self.acct.mech_counts[MECH_ADMISSION_STD],
            phase["expected"]["n_chunk_events_admission"],
        )

    def test_phase_B_backlog_matches_hand_calculation(self):
        phase = _phase(self.exp, "B")
        self._assert_phase(phase, self._run_phase(phase))

    def test_three_definitions_take_three_different_values(self):
        """The whole point of P1-e: (1), (2) and (3) must not be confusable."""
        s, b = _phase(self.exp, "S"), _phase(self.exp, "B")
        forwards = self._run_phase(s) + self._run_phase(b)
        self.assertEqual(len(forwards), 11)
        claim = self.exp["separation_claim"]["combined_boot_S_plus_B"]
        d1 = self.acct.n_requests_chunked
        d2 = self.acct.n_chunk_events
        d3 = sum(1 for n in s["prompt_lens"] + b["prompt_lens"] if n > 512)
        self.assertEqual((d1, d2, d3), (
            claim["n_requests_chunked"],
            claim["n_chunk_events"],
            claim["n_prompts_longer_than_cps"],
        ))
        self.assertEqual(len({d1, d2, d3}), 3, "the three definitions collided")


class TestMutationOfTheContinuationSite(ProbeFixture):
    """PROJECT_STATUS.md "방법론 게이트" #50 ("자기 수리를 검증하는 검사 자신이
    반증 불가능한 항등식일 수 있다"), registered remedy: the check must fail on a
    variant that reverts the repair.  Remove the continuation wrapper and the
    registered P1-e expectation must break."""

    def setUp(self):
        super().setUp()
        from sglang.srt.multiplex import chunk_probe as cp

        # revert the M3 coverage only
        PrefillAdder.add_chunked_req = cp._ORIGINALS["add_chunked_req"]

    def test_dropping_M3_breaks_the_registered_expectation(self):
        exp = json.loads(EXPECT_PATH.read_text())
        phase = _phase(exp, "S")
        for i, n in enumerate(phase["prompt_lens"]):
            run_prefill_loop([FakeReq(f"S_{i}", n)], phase["chunked_prefill_size"])
        self.assertEqual(
            self.acct.n_requests_chunked, phase["expected"]["n_requests_chunked"],
            "channel 1's request count does not depend on M3 -- expected",
        )
        self.assertNotEqual(
            self.acct.n_chunk_events, phase["expected"]["n_chunk_events"],
            "dropping the continuation site did NOT change n_chunk_events; the "
            "site is not load-bearing and the mutation test is vacuous",
        )
        self.assertEqual(self.acct.n_chunk_events, self.acct.n_requests_chunked)


# ---------------------------------------------------------------------------
# 5. default OFF, and emission
# ---------------------------------------------------------------------------


class TestDefaultOff(unittest.TestCase):
    def test_unset_env_installs_nothing(self):
        pristine = PrefillAdder.add_one_req
        old = os.environ.pop("PDMUX_CHUNK_PROBE_PATH", None)
        try:
            probe = maybe_install_chunk_probe(
                SimpleNamespace(server_args=SimpleNamespace(), tp_rank=0, tp_size=1)
            )
        finally:
            if old is not None:
                os.environ["PDMUX_CHUNK_PROBE_PATH"] = old
        self.assertIsNone(probe)
        self.assertIs(
            PrefillAdder.add_one_req, pristine,
            "the admission path was wrapped even though the probe is OFF",
        )
        self.assertFalse(hasattr(PrefillAdder.add_one_req, "_chunk_probe_wrapped"))

    def test_run_batch_is_pristine_when_off(self):
        """Channel 2 hooks `Scheduler.run_batch`; with the probe OFF that method
        must be the engine's own function, not a wrapper with a None test."""
        from sglang.srt.managers.scheduler import Scheduler

        self.assertFalse(
            hasattr(Scheduler.run_batch, "_chunk_probe_wrapped"),
            "run_batch carries a chunk-probe wrapper even though the probe is OFF",
        )

    def test_wrappers_install_and_uninstall_on_the_real_scheduler_class(self):
        from sglang.srt.managers.scheduler import Scheduler

        pristine = Scheduler.run_batch
        install_wrappers(PrefillAdder, Scheduler)
        try:
            self.assertTrue(hasattr(Scheduler.run_batch, "_chunk_probe_wrapped"))
        finally:
            uninstall_wrappers(PrefillAdder, Scheduler)
        self.assertIs(Scheduler.run_batch, pristine)


class TestConfigEcho(unittest.TestCase):
    """P1-g plus the two settings that change what channel 2 measures."""

    def _echo(self, **server_args):
        from sglang.srt.multiplex.chunk_probe import build_config_echo

        sa = SimpleNamespace(
            chunked_prefill_size=512, enable_mixed_chunk=False,
            disable_radix_cache=True, disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enforce_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=512,
            piecewise_cuda_graph_compiler="eager",
            piecewise_cuda_graph_tokens=[4, 8, 512],
        )
        for k, v in server_args.items():
            setattr(sa, k, v)
        sched = SimpleNamespace(
            server_args=sa, chunked_prefill_size=512, page_size=1,
            max_prefill_tokens=16384, max_running_requests=48,
            enable_dynamic_chunking=False, truncation_align_size=None,
            tree_cache=FakeTreeCache(), enable_pdmux=False, enable_overlap=True,
            tp_rank=0,
        )
        return build_config_echo(sched)

    def test_p1g_fields_present(self):
        cfg = self._echo()
        self.assertEqual(cfg["piecewise_cuda_graph_tokens_len"], 3)
        self.assertEqual(cfg["piecewise_cuda_graph_tokens_min"], 4)
        self.assertEqual(cfg["piecewise_cuda_graph_tokens_max"], 512)
        self.assertIs(cfg["disable_piecewise_cuda_graph"], False)

    def test_empty_capture_list_is_recorded_as_such(self):
        """The cps -1 arm: `_generate_piecewise_cuda_graph_tokens` filters on
        `s <= piecewise_cuda_graph_max_tokens`, which for -1 is empty."""
        cfg = self._echo(piecewise_cuda_graph_tokens=[],
                         piecewise_cuda_graph_max_tokens=-1)
        self.assertEqual(cfg["piecewise_cuda_graph_tokens_len"], 0)
        self.assertIsNone(cfg["piecewise_cuda_graph_tokens_min"])

    def test_admission_mechanism_precondition_is_recorded(self):
        """M2 needs `tree_cache.disable`; without it the ignore-EOS branch is
        never reached and the mechanism attribution changes."""
        cfg = self._echo()
        self.assertEqual(cfg["tree_cache_class"], "FakeTreeCache")
        self.assertIs(cfg["tree_cache_disable"], True)

    def test_true_dual_worker_is_echoed(self):
        old = os.environ.get("PDMUX_TRUE_DUAL_WORKER")
        try:
            os.environ["PDMUX_TRUE_DUAL_WORKER"] = "1"
            self.assertIs(self._echo()["true_dual_worker"], True)
            os.environ["PDMUX_TRUE_DUAL_WORKER"] = "0"
            self.assertIs(self._echo()["true_dual_worker"], False)
        finally:
            if old is None:
                os.environ.pop("PDMUX_TRUE_DUAL_WORKER", None)
            else:
                os.environ["PDMUX_TRUE_DUAL_WORKER"] = old

    def test_realized_cps_none_means_chunking_off(self):
        from sglang.srt.multiplex.chunk_probe import build_config_echo

        sched = SimpleNamespace(
            server_args=SimpleNamespace(chunked_prefill_size=-1),
            chunked_prefill_size=None, tree_cache=None, tp_rank=0,
        )
        cfg = build_config_echo(sched)
        self.assertEqual(cfg["requested_chunked_prefill_size"], -1)
        self.assertIsNone(cfg["realized_chunked_prefill_size"])


class TestEmission(unittest.TestCase):
    def test_jsonl_records_carry_the_p1g_banner_and_the_counters(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk.jsonl"
            cfg = {
                "requested_chunked_prefill_size": -1,
                "realized_chunked_prefill_size": None,
                "disable_piecewise_cuda_graph": False,
                "piecewise_cuda_graph_tokens_len": 0,
                "piecewise_cuda_graph_tokens_min": None,
                "piecewise_cuda_graph_tokens_max": None,
            }
            probe = ChunkProbe(path, run_id="rid", arm="fused_mono", config=cfg)
            probe.on_chunk("r1", MECH_ADMISSION_IGNORE_EOS, 1500, 512)
            probe.on_forward("EXTEND", 512, 1, True)
            probe.on_forward("DECODE", 0, 4, False)
            probe.close()
            rows = [json.loads(l) for l in path.read_text().splitlines()]
        by_event = {}
        for r in rows:
            by_event.setdefault(r["event"], []).append(r)
        self.assertEqual(len(by_event["chunk_open"]), 1)
        self.assertEqual(len(by_event["chunk_event"]), 1)
        self.assertEqual(
            len(by_event["prefill_forward"]), 1, "decode forwards must not emit"
        )
        summary = by_event["chunk_summary"][-1]
        self.assertEqual(summary["phase"], "shutdown")
        self.assertEqual(summary["arm"], "fused_mono")
        self.assertEqual(summary["n_requests_chunked"], 1)
        self.assertEqual(summary["n_chunk_events"], 1)
        self.assertEqual(summary["extend_tokens_per_forward"]["max"], 512)
        self.assertEqual(summary["extend_tokens_per_forward"]["n_forwards"], 1)
        for k in cfg:
            self.assertIn(k, summary, f"P1-g field {k} missing from the summary")
            self.assertIn(k, by_event["chunk_open"][0])
        self.assertEqual(summary["dropped_events"], 0)

    def test_probe_never_raises_into_the_engine(self):
        probe = ChunkProbe(None, arm="test")
        probe.on_forward("EXTEND", object(), 1, False)  # unusable extend count
        probe.on_chunk(object(), MECH_CONTINUATION, None, None)
        self.assertGreaterEqual(probe.n_errors, 1)


# ---------------------------------------------------------------------------
# 6. the dev-tree edit is mirrored, reversible and does not damage the holb patch
# ---------------------------------------------------------------------------


class TestSchedulerHookInstalled(unittest.TestCase):
    TRACK_ROOT = Path(__file__).resolve().parents[1]

    def _scheduler_path(self):
        import sglang.srt.managers.scheduler as sched

        return Path(sched.__file__)

    def test_hook_present_and_off_by_default(self):
        src = self._scheduler_path().read_text()
        self.assertIn(
            "from sglang.srt.multiplex.chunk_probe import maybe_install_chunk_probe",
            src,
        )
        self.assertEqual(src.count("self.chunk_probe = maybe_install_chunk_probe(self)"), 1)

    def test_hook_is_outside_the_holb_patch_windows(self):
        """An insertion inside the holb patch's context costs that patch its
        reversibility.  Pin the separation rather than rediscovering it."""
        src = self._scheduler_path().read_text()
        hook = src.index("self.chunk_probe = maybe_install_chunk_probe(self)")
        holb_import = src.index(
            "from sglang.srt.multiplex.holb_probe import maybe_create_holb_probe"
        )
        holb_init = src.index("self.holb_probe = maybe_create_holb_probe(self)")
        grammar = src.index("self.grammar_manager = GrammarManager(self)")
        self.assertLess(hook, grammar, "hook must precede the holb hunk 3 window")
        self.assertLess(holb_import, hook)
        self.assertLess(hook, holb_init)

    def _reverse_then_forward(self, patch_files):
        import subprocess
        import tempfile

        installed = self._scheduler_path().read_text()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sglang" / "srt" / "managers"
            target.mkdir(parents=True)
            (target / "scheduler.py").write_text(installed)
            for direction in ("--reverse", "--forward"):
                order = patch_files if direction == "--forward" else patch_files[::-1]
                for pf in order:
                    with pf.open("rb") as fh:
                        r = subprocess.run(
                            ["patch", direction, "--batch", "-p1", "-d", tmp],
                            stdin=fh,
                            capture_output=True,
                        )
                    self.assertEqual(
                        r.returncode, 0,
                        f"{direction} {pf.name}: {r.stderr.decode()}{r.stdout.decode()}",
                    )
            return (target / "scheduler.py").read_text(), installed

    def test_patch_reapplies_exactly(self):
        pf = self.TRACK_ROOT / "src/patches/chunk_probe_scheduler_hook.patch"
        self.assertTrue(pf.is_file(), pf)
        got, installed = self._reverse_then_forward([pf])
        self.assertEqual(got, installed)

    def test_both_scheduler_patches_compose(self):
        """sync_engine_tree.sh applies holb then chunk; both must be reversible
        from the installed tree, in either direction of the stack."""
        patches = [
            self.TRACK_ROOT / "src/patches/holb_probe_scheduler_hooks.patch",
            self.TRACK_ROOT / "src/patches/chunk_probe_scheduler_hook.patch",
        ]
        got, installed = self._reverse_then_forward(patches)
        self.assertEqual(got, installed)


class TestMirrorSync(unittest.TestCase):
    def test_tracked_mirror_matches_installed_runtime(self):
        import sglang.srt.multiplex.chunk_probe as cp

        mirror = (
            Path(__file__).resolve().parents[1] / "src" / "multiplex" / "chunk_probe.py"
        )
        runtime = Path(cp.__file__)
        if not runtime.is_file():
            self.skipTest(f"runtime tree absent: {runtime}")
        self.assertEqual(
            mirror.read_text(),
            runtime.read_text(),
            "src/multiplex/chunk_probe.py and the installed runtime diverged; "
            "re-run scripts/bootstrap/sync_engine_tree.sh",
        )

    def test_sync_script_installs_and_hashes_the_module(self):
        sync = (
            Path(__file__).resolve().parents[1]
            / "scripts/bootstrap/sync_engine_tree.sh"
        ).read_text()
        self.assertIn('"${track_root}/src/multiplex/chunk_probe.py"', sync)
        self.assertIn(
            '"${runtime_python}/sglang/srt/multiplex/chunk_probe.py"', sync,
            "chunk_probe.py is installed but not in the SHA-256 manifest",
        )
        self.assertIn("chunk_probe_scheduler_hook.patch", sync)
        self.assertLess(
            sync.index("holb_probe_scheduler_hooks.patch"),
            sync.index("chunk_probe_scheduler_hook.patch"),
            "the chunk hook patch must be applied after the holb patch",
        )


if __name__ == "__main__":
    unittest.main()
