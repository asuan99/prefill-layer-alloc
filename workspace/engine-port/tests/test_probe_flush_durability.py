"""Gate for probe-output DURABILITY on a low-traffic boot (P1 repair 1).

WHY THIS EXISTS
---------------
Job 899768 (`results/cp_baseline/RESULT_P1_899768_2026-08-28.md`) came back
`P1_BLOCKED_INSTRUMENT` with three `EXTRACTION_EMPTY` labels whose whole cause
was that ``chunk_mono_on.jsonl`` and ``chunk_cps4096_on.jsonl`` were **exactly
zero bytes**.  The counters were fine; the bytes never left the process:

  * ``AsyncJsonlTelemetry`` buffers and only calls ``flush()`` after
    ``flush_every`` (128) records or inside ``close()``;
  * ``close()`` is reached only through ``atexit``;
  * the scheduler process never runs ``atexit`` at shutdown -- SGLang's parent
    reaps it with **SIGKILL** (``sglang/launch_server.py`` ``finally:
    kill_process_tree(os.getpid(), include_parent=False)`` ->
    ``srt/utils/common.py`` ``child.kill()``), and the P1 harness then sent its
    own ``pkill -9`` with no grace period at all.

So any boot whose traffic did not happen to exceed 128 telemetry records wrote
NOTHING -- not even the ``chunk_open`` line that is queued synchronously in
``ChunkProbe.__init__``.  The 4-request ``mono`` boot and the 16-request
``cps4096`` burst are exactly that case; the 200-request P1-f boots were not,
which is why the failure looked selective rather than structural.

WHAT IS PINNED HERE
-------------------
1. a boot that emits far fewer than ``flush_every`` records still has its
   records on disk within a bounded time **without anyone calling close()**;
2. the same fixture with the repair reverted (``flush_interval_s = 0``, the
   pre-repair behaviour) leaves the file empty -- so check (1) is not vacuous.
   PROJECT_STATUS.md "방법론 게이트" #53: *"자기 수리를 검증하는 검사는 그
   수리를 되돌린 변이본에서 반드시 실패해야 한다"*;
3. the durable path is the **default**, because the engine constructs the
   writer with no arguments;
4. a process that is SIGTERMed flushes through the probe's own signal handler,
   and the same process without that handler does not (again: mutation);
5. that handler cannot deadlock the thread it interrupts -- closing the probe
   inline from a signal handler re-enters ``queue.Queue.put`` and hangs the
   process forever if the interrupted frame already held that lock, and the
   mutation (the obvious inline handler) demonstrably does hang;
6. the HOLB probe, which shares the writer, gains the same durability and is
   not regressed by it.

WHAT THIS FILE CANNOT CHECK.  That the *engine* really reaches the handler:
the scheduler is SIGKILLed by its own parent, so on the GPU the load-bearing
mechanism is (1)+(3) -- the interval flush -- and the handler is only the
second line of defence for the case where the scheduler is signalled directly.
Only a GPU boot can show the artefact is non-empty end to end.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(
    os.environ.get("SGLANG_ENGINE_DEV", "/scratch/ehmoon/whlee/sglang_engine_dev/python")
)
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

# The low-traffic boots that failed: 4 and 16 requests.  Nothing here may
# assume more than a handful of records.
LOW_TRAFFIC_RECORDS = 3
FAST_INTERVAL_S = 0.2
POSITIVE_WAIT_S = 5.0
# How long the reverted variant is given to prove it stays empty.  It must be
# comfortably longer than the positive path's latency budget.
MUTANT_WAIT_S = 1.5


def _read_when_nonempty(path, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return path.read_text()
        time.sleep(0.02)
    return path.read_text() if path.exists() else ""


def _events(text):
    return [json.loads(l)["event"] for l in text.splitlines() if l.strip()]


def _wait_ready(proc, timeout_s=60):
    """Read the child's stderr until it announces READY.

    Importing sglang emits warnings on stderr, so a single readline() would
    read one of those instead and the test would fail for the wrong reason.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stderr.readline().decode()
        if not line or "READY" in line:
            return line
    return ""


class TestLowTrafficReachesDisk(unittest.TestCase):
    """(1)(2)(3): the bytes leave the process without close()."""

    def _probe(self, path, **tel_kw):
        """Build the probe exactly the way ``maybe_install_chunk_probe`` does.

        With no ``tel_kw`` this is the ENGINE's construction path verbatim --
        that is what makes the first test a reproduction of job 899768 rather
        than a test of a test-only configuration.
        """
        from sglang.srt.multiplex.chunk_probe import ChunkProbe
        from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

        tel = (
            AsyncJsonlTelemetry(path, run_id="rid", workload_id="wid", **tel_kw)
            if tel_kw
            else None
        )
        probe = ChunkProbe(
            path, run_id="rid", workload_id="wid", arm="lowtraffic",
            telemetry=tel,
            summary_every_s=1e9,   # no periodic summary: chunk_open must carry itself
            config={"requested_chunked_prefill_size": -1},
        )
        self.addCleanup(probe.close)
        return probe

    def test_chunk_open_reaches_disk_without_close(self):
        """★THE REPRODUCTION.  The mono boot's failure mode, in one test.

        Default construction, LOW_TRAFFIC_RECORDS records (job 899768's mono
        boot served 4 requests), no ``close()`` -- because the engine's
        scheduler process never reaches ``atexit``.  Before repair 1 this file
        is 0 bytes forever, exactly as ``chunk_mono_on.jsonl`` was.
        """
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk_mono_on.jsonl"
            probe = self._probe(path)          # no kwargs: the engine's own path
            for i in range(LOW_TRAFFIC_RECORDS):
                probe.on_forward("EXTEND", 100 + i, 1, False)
            text = _read_when_nonempty(path, POSITIVE_WAIT_S)
        self.assertTrue(
            text,
            f"{LOW_TRAFFIC_RECORDS} records and the boot config echo never "
            f"reached disk within {POSITIVE_WAIT_S}s; this is job 899768's "
            "EXTRACTION_EMPTY (chunk_mono_on.jsonl, exactly 0 bytes)",
        )
        self.assertIn(
            "chunk_open", _events(text),
            "the boot configuration echo (P1-g fields) is the one record that "
            "must survive even a boot that served nothing",
        )

    def test_the_same_check_fails_with_the_repair_reverted(self):
        """MUTATION.  flush_interval_s = 0 is the pre-repair writer."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk_mono_on.jsonl"
            probe = self._probe(path, flush_interval_s=0.0)
            for i in range(LOW_TRAFFIC_RECORDS):
                probe.on_forward("EXTEND", 100 + i, 1, False)
            text = _read_when_nonempty(path, MUTANT_WAIT_S)
        self.assertEqual(
            text, "",
            "the reverted variant wrote something, so the durability check "
            "above would pass for a reason other than the repair",
        )

    def test_the_durable_path_is_the_default(self):
        """The engine builds the writer with no arguments, so an opt-in repair
        would leave every real boot exactly as broken as job 899768."""
        from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "default.jsonl"
            tel = AsyncJsonlTelemetry(path, run_id="rid", workload_id="wid")
            self.addCleanup(tel.close)
            self.assertGreater(
                getattr(tel, "flush_interval_s", 0), 0.0,
                "the default writer still only flushes every flush_every "
                "records; a low-traffic boot would write nothing",
            )
            tel.emit("probe_event", "measure", value=1)
            text = _read_when_nonempty(path, POSITIVE_WAIT_S)
        self.assertIn("probe_event", _events(text))

    def test_a_full_batch_still_flushes_on_the_count(self):
        """The interval must not have replaced the count trigger: a busy boot
        is still allowed to flush on flush_every rather than waiting."""
        from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "busy.jsonl"
            tel = AsyncJsonlTelemetry(
                path, run_id="rid", workload_id="wid",
                flush_every=4, flush_interval_s=600.0,
            )
            self.addCleanup(tel.close)
            for i in range(8):
                tel.emit("probe_event", "measure", value=i)
            text = _read_when_nonempty(path, POSITIVE_WAIT_S)
        self.assertGreaterEqual(len(_events(text)), 4)


_CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {runtime!r})
    from sglang.srt.multiplex.chunk_probe import ChunkProbe, install_shutdown_handlers
    from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

    path = sys.argv[1]
    want_handler = sys.argv[2] == "1"
    # The repair-1 interval flush is DISABLED here on purpose: this child tests
    # the signal handler alone, so the interval must not be able to pass it.
    tel = AsyncJsonlTelemetry(path, run_id="rid", workload_id="wid",
                              flush_interval_s=0.0)
    probe = ChunkProbe(path, run_id="rid", arm="sigterm", telemetry=tel,
                       summary_every_s=1e9, config={{}})
    probe.on_forward("EXTEND", 123, 1, False)
    if want_handler:
        install_shutdown_handlers(probe)
    sys.stderr.write("READY\\n")
    sys.stderr.flush()
    time.sleep(60)
    """
)


class TestSigtermFlush(unittest.TestCase):
    """(4): the second line of defence, with its own mutation."""

    def _run_child(self, want_handler):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk.jsonl"
            src = _CHILD.format(runtime=str(RUNTIME))
            proc = subprocess.Popen(
                [sys.executable, "-c", src, str(path), "1" if want_handler else "0"],
                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            )
            try:
                # wait for READY so the SIGTERM cannot land before the handler
                self.assertIn("READY", _wait_ready(proc),
                              "child never became ready")
                proc.terminate()
                proc.wait(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
                proc.stderr.close()
            return path.read_text() if path.exists() else ""

    def test_sigterm_flushes_the_probe(self):
        text = self._run_child(want_handler=True)
        self.assertTrue(text, "SIGTERM left the probe output empty")
        events = _events(text)
        self.assertIn("chunk_open", events)
        self.assertIn("chunk_summary", events)

    def test_without_the_handler_sigterm_loses_everything(self):
        """MUTATION.  Python's default SIGTERM disposition skips atexit."""
        text = self._run_child(want_handler=False)
        self.assertEqual(
            text, "",
            "the child wrote without the handler, so the handler test above "
            "would pass for a reason other than the handler",
        )


_DEADLOCK_CHILD = textwrap.dedent(
    """
    import os, signal, sys, time
    sys.path.insert(0, {runtime!r})
    import sglang.srt.multiplex.chunk_probe as cp
    from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

    path, naive = sys.argv[1], sys.argv[2] == "1"
    cp.CLOSE_ON_SIGNAL_TIMEOUT_S = 0.5
    tel = AsyncJsonlTelemetry(path, run_id="rid", workload_id="wid",
                              flush_interval_s=0.0)
    probe = cp.ChunkProbe(path, run_id="rid", arm="deadlock", telemetry=tel,
                          summary_every_s=1e9, config={{}})
    if naive:
        # MUTANT: the obvious handler -- close() straight from the signal
        signal.signal(signal.SIGTERM, lambda s, f: probe.close())
    else:
        cp.install_shutdown_handlers(probe)
    sys.stderr.write("READY\\n")
    sys.stderr.flush()
    # Stand exactly where a scheduler thread stands when it is interrupted in
    # the middle of emit(): holding the telemetry queue's non-reentrant lock.
    with tel._queue.mutex:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    """
)


class TestSignalHandlerCannotDeadlock(unittest.TestCase):
    """A Python signal handler runs INSIDE the interrupted thread.  Closing the
    probe from it re-enters ``queue.Queue.put``; if the interrupted frame was
    already inside ``put`` it holds a non-reentrant lock and the process hangs
    forever with no timeout.  On the GPU that would look like a boot that never
    dies, and the harness would sit in its grace loop for every boot."""

    def _child(self, naive):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chunk.jsonl"
            src = _DEADLOCK_CHILD.format(runtime=str(RUNTIME))
            proc = subprocess.Popen(
                [sys.executable, "-c", src, str(path), "1" if naive else "0"],
                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            )
            try:
                self.assertIn("READY", _wait_ready(proc),
                              "child never became ready")
                started = time.time()
                try:
                    proc.wait(timeout=6)
                    return time.time() - started
                except subprocess.TimeoutExpired:
                    return None          # hung
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
                proc.stderr.close()

    def test_the_installed_handler_terminates(self):
        elapsed = self._child(naive=False)
        self.assertIsNotNone(
            elapsed,
            "the shutdown handler deadlocked against the thread it interrupted",
        )

    def test_the_naive_handler_hangs(self):
        """MUTATION.  Without the helper thread the same child never exits."""
        self.assertIsNone(
            self._child(naive=True),
            "closing inline from the handler did NOT hang, so the check above "
            "is not discriminating on this platform",
        )


class TestSharedWriterTracks(unittest.TestCase):
    """(5): telemetry.py is shared.  The repair must help, not hurt, HOLB."""

    def test_holb_probe_low_traffic_also_reaches_disk(self):
        from sglang.srt.multiplex.holb_probe import HolbProbe
        from sglang.srt.multiplex.telemetry import AsyncJsonlTelemetry

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "holb.jsonl"
            tel = AsyncJsonlTelemetry(
                path, run_id="rid", workload_id="wid",
                flush_interval_s=FAST_INTERVAL_S,
            )
            probe = HolbProbe(path, run_id="rid", telemetry=tel)
            self.addCleanup(probe.close)
            text = _read_when_nonempty(path, POSITIVE_WAIT_S)
        self.assertTrue(
            text,
            "the HOLB probe shares AsyncJsonlTelemetry and must gain the same "
            "durability; if this is empty the two tracks disagree about the "
            "writer",
        )


class TestMirrorSync(unittest.TestCase):
    def test_tracked_mirror_matches_installed_runtime(self):
        import sglang.srt.multiplex.telemetry as tel

        mirror = TRACK_ROOT / "src" / "multiplex" / "telemetry.py"
        runtime = Path(tel.__file__)
        if not runtime.is_file():
            self.skipTest(f"runtime tree absent: {runtime}")
        self.assertEqual(
            mirror.read_text(), runtime.read_text(),
            "src/multiplex/telemetry.py and the installed runtime diverged; "
            "re-run scripts/bootstrap/sync_engine_tree.sh",
        )


if __name__ == "__main__":
    unittest.main()
