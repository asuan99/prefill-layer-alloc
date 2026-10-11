"""N2(3-a) U-2 probe -- HARNESS MARK WRITER (PREREG_N2_3A rev12 §14-2 "마크 형식 사양", H-1, H-4…H-6).

This is a frozen "harness mark writer" (H-2): every byte of a probe marks file is written through MarkWriter.
The registered reader is n2_probe_reduce.parse_marks; this writer emits ONLY the registered tags with the registered
arity (an unregistered tag is MARK_MALFORMED = VALUE_INDEPENDENT in the reducer, so this writer refuses it instead).

Clock: every time field is time.monotonic() of THIS process, which must run in the SAME container as the engine
(engine: time.perf_counter(); both clock_gettime(CLOCK_MONOTONIC) on Linux CPython -- `clock_contract_ok()`).
File discipline (H-6): the marks file is created with mode 'x' (never overwritten, never appended to from a previous
attempt); one line per mark, single-space fields, flushed per line.
Ids: cell ids and call ids are unique per writer AND per probe (`registry` shared across the boots of one attempt);
reuse raises (the reducer would record MARK_DUPLICATE / CALL_DUPLICATE = VALUE_DEPENDENT).

Cell protocol (H-1, R11-3): `async with mw.cell(cell_id, d, b, c) as cell:` writes `cell_start` on entry; on an exception it
writes `cell_abort <id> <reason> <t>` + `abort_detail <id> <ExceptionClass>` (t = the moment the abort was decided,
reason from `n2_loadgen.classify_exception` -- H-4) and re-raises; `cell_end` is ALWAYS written in `finally` after the
caller-supplied drain (`cell.drain`). Boot protocol (R11-2, H-5): `with mw.boot(plan) as boot:` writes `boot_plan <d> <n>`
for every d before any cell; an exception escaping the boot writes `boot_abort <reason> <t>` in `finally`.
"""
from __future__ import annotations

import os
import re
import time

# Registered tag -> number of fields after the tag (== n2_probe_reduce.parse_marks arity; test_n2_harness pins it).
ARITY = {"boot_plan": 2, "cell_start": 5, "load_call_start": 2, "coload_sent": 2, "cell_abort": 3,
         "abort_detail": 2, "cell_end": 2, "load_call_end": 2, "delta_l": 2, "fix_s": 2, "boot_abort": 2}
# rev16 H14-3 / §14-2 G-3 (b): per-boot warm-up interval marks `warmup_start <boot> <t>` / `warmup_end <boot> <t>`
# (== n2_probe_reduce.WARMUP_ARITY, kept apart from ARITY exactly as the reducer keeps it apart from `arity`;
# test_n2_harness C1 pins ARITY ∪ WARMUP_ARITY).
WARMUP_ARITY = {"warmup_start": 2, "warmup_end": 2}
# Registered abort reason names (§14-2 "중단 사유 → 부류 표"; == n2_probe_reduce.ABORT_REASONS keys, pinned by test).
REGISTERED_REASONS = ("server_down", "http_connection_error", "cell_timeout", "coload_payload_invalid",
                      "stream_cancel_failed", "runner_stop", "client_exception", "wiring_check_failed",
                      "boot_health_timeout")                                      # rev13 G-8 / G-9
# rev13 G-8: every name this harness can emit is now registered; none is left unregistered (a typo is refused).
UNREGISTERED_REASONS = ()
_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def clock_contract_ok():
    """§3-3 ABORT_CLOCK: client time.monotonic and engine time.perf_counter must be the same CLOCK_MONOTONIC."""
    a, b = time.get_clock_info("monotonic"), time.get_clock_info("perf_counter")
    return a.implementation == b.implementation == "clock_gettime(CLOCK_MONOTONIC)"


class MarkError(RuntimeError):
    pass


class IdRegistry:
    """Probe-wide uniqueness of cell ids and call ids (shared by every MarkWriter of one probe attempt)."""

    def __init__(self):
        self.cells, self.calls = set(), set()

    def claim(self, kind, ident):
        if not _ID.match(str(ident)):
            raise MarkError(f"{kind} id {ident!r} is not a single token")
        pool = self.cells if kind == "cell" else self.calls
        if ident in pool:
            raise MarkError(f"duplicate {kind} id {ident!r} (reuse is forbidden, §14-2)")
        pool.add(ident)


def _num(x):
    return repr(float(x))


class MarkWriter:
    def __init__(self, path, registry=None, clock=time.monotonic):
        self.path = path
        self.registry = registry if registry is not None else IdRegistry()
        self.clock = clock
        self._fh = open(path, "x", encoding="utf-8")       # H-6: never overwrite
        self.lines = []

    # -- low level ------------------------------------------------------------------------------------------
    def write(self, tag, *fields):
        arity = ARITY.get(tag, WARMUP_ARITY.get(tag))
        if arity is None:
            raise MarkError(f"unregistered mark tag {tag!r}")
        if len(fields) != arity:
            raise MarkError(f"mark {tag} needs {arity} fields, got {len(fields)}")
        parts = [str(f) for f in fields]
        if any((not p) or any(ch.isspace() for ch in p) for p in parts):
            raise MarkError(f"mark {tag} field with whitespace or empty: {parts}")
        line = " ".join([tag] + parts)
        self._fh.write(line + "\n")
        self._fh.flush()
        self.lines.append(line)
        return line

    def now(self):
        return self.clock()

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    # -- registered marks ---------------------------------------------------------------------------------
    def boot_plan(self, d, n):
        return self.write("boot_plan", int(d), int(n))

    def cell_start(self, cell_id, d, b, c, t=None):
        self.registry.claim("cell", cell_id)
        return self.write("cell_start", cell_id, int(d), int(b), int(c), _num(self.now() if t is None else t))

    def load_call_start(self, call_id, t=None):
        self.registry.claim("call", call_id)
        return self.write("load_call_start", call_id, _num(self.now() if t is None else t))

    def load_call_end(self, call_id, t=None):
        return self.write("load_call_end", call_id, _num(self.now() if t is None else t))

    def coload_sent(self, cell_id, t):
        return self.write("coload_sent", cell_id, _num(t))

    def cell_abort(self, cell_id, reason, t, exc_class="-"):
        if reason not in REGISTERED_REASONS and reason not in UNREGISTERED_REASONS:
            raise MarkError(f"abort reason {reason!r} is neither registered nor a declared unregistered name")
        self.write("cell_abort", cell_id, reason, _num(t))
        self.write("abort_detail", cell_id, exc_class or "-")

    def cell_end(self, cell_id, t=None):
        return self.write("cell_end", cell_id, _num(self.now() if t is None else t))

    def delta_l(self, cell_id, s):
        return self.write("delta_l", cell_id, _num(s))

    def fix_s(self, cell_id, s):
        return self.write("fix_s", cell_id, _num(s))

    def boot_abort(self, reason, t=None):
        if reason not in REGISTERED_REASONS and reason not in UNREGISTERED_REASONS:
            raise MarkError(f"boot_abort reason {reason!r} is neither registered nor a declared unregistered name")
        return self.write("boot_abort", reason, _num(self.now() if t is None else t))

    def warmup_start(self, boot_id, t=None):
        return self.write("warmup_start", boot_id, _num(self.now() if t is None else t))

    def warmup_end(self, boot_id, t=None):
        return self.write("warmup_end", boot_id, _num(self.now() if t is None else t))

    # -- protocols ----------------------------------------------------------------------------------------
    def cell(self, cell_id, d, b, c, classify):
        return _CellScope(self, cell_id, d, b, c, classify)

    def boot(self, plan, classify):
        return _BootScope(self, plan, classify)

    def warmup(self, boot_id):
        """§14-2 G-3: `warmup_start` before, `warmup_end` ALWAYS (finally) after the per-boot warm-up, so the
        interval is paired even when the warm-up raises (an unpaired mark is WARMUP_UNPAIRED in the reducer)."""
        return _WarmupScope(self, boot_id)


class CellAborted(Exception):
    """Raised out of a cell scope after cell_abort/cell_end were written; carries the registered reason."""

    def __init__(self, cell_id, reason, cause):
        super().__init__(f"cell {cell_id} aborted: {reason} ({type(cause).__name__}: {cause})")
        self.cell_id, self.reason, self.cause = cell_id, reason, cause


class _CellScope:
    """ASYNC scope. cell_start on entry; cell_abort(+abort_detail) on exception (reason via `classify`, t = the
    decision time); the caller's drain (`self.drain`, an async callable taking aborted=bool, may be None) runs in
    finally; cell_end is ALWAYS written last (H-1)."""

    def __init__(self, mw, cell_id, d, b, c, classify):
        self.mw, self.cell_id, self.d, self.b, self.c, self.classify = mw, cell_id, d, b, c, classify
        self.drain = None
        self.t_start = None
        self.t_end = None
        self.aborted = None

    async def __aenter__(self):
        self.t_start = self.mw.now()
        self.mw.cell_start(self.cell_id, self.d, self.b, self.c, t=self.t_start)
        return self

    def _abort(self, exc):
        t_ab = self.mw.now()                                  # R11-3: the moment the abort is decided
        self.aborted = self.classify(exc)
        self.mw.cell_abort(self.cell_id, self.aborted, t_ab, type(_root(exc)).__name__)

    async def __aexit__(self, et, ev, tb):
        cause = ev
        try:
            if ev is not None:
                self._abort(ev)
            if self.drain is not None:
                try:
                    await self.drain(aborted=self.aborted is not None)
                except Exception as e:                       # a failing drain of a GOOD cell is itself the abort
                    if self.aborted is None:
                        self._abort(e)
                        cause = e
        finally:
            self.t_end = self.mw.now()
            self.mw.cell_end(self.cell_id, t=self.t_end)
        if cause is not None:
            raise CellAborted(self.cell_id, self.aborted, cause) from cause
        return False


class Classified(Exception):
    """An exception already mapped to a reason (classified while the failure state was still observable)."""

    def __init__(self, reason, cause):
        super().__init__(f"{reason} ({type(cause).__name__}: {cause})")
        self.reason, self.cause = reason, cause


class _WarmupScope:
    def __init__(self, mw, boot_id):
        self.mw, self.boot_id = mw, boot_id
        self.t0 = self.t1 = None

    def __enter__(self):
        self.t0 = self.mw.now()
        self.mw.warmup_start(self.boot_id, self.t0)
        return self

    def __exit__(self, et, ev, tb):
        self.t1 = self.mw.now()
        self.mw.warmup_end(self.boot_id, self.t1)
        return False


class _BootScope:
    def __init__(self, mw, plan, classify):
        self.mw, self.plan, self.classify = mw, dict(plan), classify
        self.aborted = None

    def __enter__(self):
        for d, n in sorted(self.plan.items()):
            self.mw.boot_plan(d, n)
        return self

    def __exit__(self, et, ev, tb):
        if ev is not None:
            reason = ev.reason if isinstance(ev, (CellAborted, Classified)) else self.classify(ev)
            self.aborted = reason
            self.mw.boot_abort(reason)
        self.mw.close()
        return False


def _root(exc):
    """The exception that actually failed (unwrap the harness's RequestFailed wrapper)."""
    while getattr(exc, "inner", None) is not None:
        exc = exc.inner
    return exc
