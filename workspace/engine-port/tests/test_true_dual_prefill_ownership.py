"""Gate: the true-dual prefill task must see the batch as it was submitted.

THE CRASH (GPU job 907032, both PDMUX_TRUE_DUAL_WORKER=1 boots, first split
prefill = the server's own warm-up request).  In `event_loop_pdmux` the
scheduler thread submitted the prefill chunk to the prefill worker and then,
without waiting, advanced the same ScheduleBatch:
    split_prefill_finished = True (final chunk) ; split_index = next_split_index
The worker's `run_batch` -> `tp_worker.forward_batch_split_prefill` reads
`batch.split_index == 0` to decide whether to build `split_forward_batch`
(tp_worker.py:537-540).  The scheduler thread's two stores normally beat the
worker's wake-up, so the first chunk read a non-zero index, skipped the
build, and passed `split_forward_batch=None` to the model forward:
    AttributeError: 'NoneType' object has no attribute 'forward_mode'
The legacy loop never raced because its `run_batch` is synchronous: read,
then advance.

THE INVARIANT PINNED HERE.  Every split-prefill chunk's forward must observe
`ScheduleBatch.split_index` equal to the ForwardBatch's own start index --
i.e. the batch exactly as it stood when the chunk was submitted -- for ANY
legal interleaving of the two threads.  Two extremes are driven
deterministically with the real TrueDualWorkerRuntime's lease/arbiter logic:
  eager    the task runs inside submit()   (the legacy order; passed pre-fix)
  deferred the task runs at result()       (worker starts after the scheduler
                                            thread's post-submit statements --
                                            the order job 907032 hit)
plus one run with the real worker threads (a 20 ms read delay stands in for
wake-up latency; after the fix the outcome cannot depend on it).  Finally the
true-dual chunk/merge bookkeeping must be identical to the legacy loop's.

The mixin is loaded as described in pdmux_loop_fakes.py (installed tree, or
PDMUX_MIXIN_UNDER_TEST=<file> to run against a pre-fix copy).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdmux_loop_fakes import (  # noqa: E402  (after the sys.path insert)
    MIXIN_SOURCE,
    NUM_LAYERS,
    SM_COUNTS,
    ExecutionContext,
    FakeReq,
    LoopScheduler,
    LoopTestBase,
    TrueDualWorkerRuntime,
    run_loop,
)
from sglang.srt.multiplex.controller import FixedPolicy  # noqa: E402


class _ResolvedFuture:
    """The part of concurrent.futures.Future the event loop uses."""

    def __init__(self, run):
        self._run = run
        self._ran = False
        self._value = None
        self._error = None

    def run(self):
        if not self._ran:
            self._ran = True
            try:
                self._value = self._run()
            except BaseException as exc:  # noqa: BLE001 - re-raised at result()
                self._error = exc

    def result(self, timeout=None):
        self.run()
        if self._error is not None:
            raise self._error
        return self._value


class _InlineRuntime(TrueDualWorkerRuntime):
    """TrueDualWorkerRuntime whose tasks run on the calling thread, either
    inside submit() (eager) or at the first result() (deferred).  Lease
    acquisition/release and the arbiter are the real ones."""

    def __init__(self, sm_counts, activate, eager):
        super().__init__(sm_counts, activate)
        self._activate_fn = activate
        self._eager = eager

    def submit(self, role, callback, stream=None, decode_backend=None):
        lease = self.arbiter.acquire(role, strict=True)
        context = ExecutionContext(
            role=role,
            stream_index=lease.stream_index,
            prefill_sms=lease.prefill_sms,
            decode_sms=lease.decode_sms,
            generation=lease.generation,
            stream=stream,
            decode_backend=decode_backend,
        )

        def _run():
            try:
                with self._activate_fn(context):
                    return callback(context)
            finally:
                self.arbiter.release(role)

        future = _ResolvedFuture(_run)
        if self._eager:
            future.run()
        return future


def eager_runtime(sched):
    return _InlineRuntime(SM_COUNTS, sched._activate_role_context, eager=True)


def deferred_runtime(sched):
    return _InlineRuntime(SM_COUNTS, sched._activate_role_context, eager=False)


def threaded_runtime(sched):
    return TrueDualWorkerRuntime(SM_COUNTS, sched._activate_role_context)


# A: single-chunk prefill on an empty server (the request job 907032 died on);
# B: four one-layer chunks overlapping A's decode (R2 fixed switches to D44);
# C: two-layer chunks overlapping decode.
ARRIVALS = lambda: {  # noqa: E731 - fresh request objects per run
    0: [FakeReq("A", max_new=30, prompt_len=10)],
    6: [FakeReq("B", max_new=5, prompt_len=100)],
    9: [FakeReq("C", max_new=5, prompt_len=50)],
}
MAX_ITERS = 40


def make(runtime_factory=None, delay=0.0):
    arrivals = ARRIVALS()
    sched = LoopScheduler(
        FixedPolicy(44), "fixed", arrivals, MAX_ITERS,
        runtime_factory=runtime_factory, prefill_read_delay_s=delay,
    )
    sched.all_reqs = [req for reqs in arrivals.values() for req in reqs]
    return sched


class TrueDualPrefillOwnershipTest(LoopTestBase):
    def _run(self, sched, label):
        try:
            run_loop(self.install(sched))
        except AttributeError as exc:
            self.fail(
                f"[{label}] true-dual split prefill crashed: {exc!r}.  The prefill "
                "task read split_index after the scheduler thread had advanced it, "
                "so split_forward_batch was never built (job 907032) "
                f"[{MIXIN_SOURCE}]"
            )
        return sched

    def _assert_chunks_saw_submitted_state(self, sched, label):
        chunks = sched.prefill_chunks()
        self.assertTrue(chunks, f"[{label}] scenario broken: no prefill chunk ran")
        for _, rids, observed, start, end, _thread in chunks:
            self.assertEqual(
                observed, start,
                f"[{label}] chunk of {rids} saw ScheduleBatch.split_index={observed} "
                f"but its forward started at layer {start}: the batch was mutated "
                f"after submit [{MIXIN_SOURCE}]",
            )
        finished = {rids: end for _, rids, _o, _s, end, _t in chunks}
        for rid in ("A", "B", "C"):
            self.assertEqual(
                finished.get((rid,)), NUM_LAYERS,
                f"[{label}] {rid} was not prefilled through all layers",
            )
            self.assertTrue(sched.admissions(rid), f"[{label}] {rid} never admitted")

    def test_task_run_after_post_submit_statements_sees_submitted_batch(self):
        sched = self._run(make(deferred_runtime), "deferred")
        self._assert_chunks_saw_submitted_state(sched, "deferred")

    def test_task_run_inside_submit_sees_submitted_batch(self):
        sched = self._run(make(eager_runtime), "eager")
        self._assert_chunks_saw_submitted_state(sched, "eager")

    def test_real_worker_threads(self):
        sched = self._run(make(threaded_runtime, delay=0.02), "threads")
        self._assert_chunks_saw_submitted_state(sched, "threads")
        threads = {t for *_rest, t in sched.prefill_chunks()}
        self.assertEqual(
            threads, {"pdmux-prefill-worker"},
            "split prefill must run on the prefill worker thread in true-dual mode",
        )

    def test_true_dual_bookkeeping_matches_legacy(self):
        legacy = self._run(make(None), "legacy")
        dual = self._run(make(deferred_runtime), "deferred")

        def trace(sched):
            chunks = [(rids, observed, start, end)
                      for _, rids, observed, start, end, _t in sched.prefill_chunks()]
            admits = [(e[1], e[2]) for e in sched.log if e[0] == "admit"]
            decisions = [(d[2]["current_decode_sms"], d[2]["target_decode_sms"])
                         for d in sched.decisions()]
            return chunks, admits, decisions

        self.assertEqual(trace(dual), trace(legacy))
        tokens = lambda sched: {r.rid: len(r.output_ids) for r in sched.all_reqs}  # noqa: E731
        self.assertEqual(tokens(dual), tokens(legacy))
        self.assertEqual(tokens(legacy), {"A": 30, "B": 5, "C": 5},
                         "scenario broken: not every request finished")


if __name__ == "__main__":
    unittest.main()
