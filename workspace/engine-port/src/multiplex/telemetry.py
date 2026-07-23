"""Symmetric, non-blocking JSONL telemetry for every PD-mux baseline."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


TELEMETRY_SCHEMA = "pdmux.runtime-event/v1"


@dataclass
class RuntimeEvent:
    run_id: str
    workload_id: str
    phase: str
    event: str
    timestamp_monotonic_s: float = field(default_factory=time.perf_counter)
    request_id: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    schema: str = TELEMETRY_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        fields = raw.pop("fields")
        raw.update(fields)
        return raw


class AsyncJsonlTelemetry:
    """Bounded producer queue with a single long-lived writer thread.

    Emission never performs file I/O on a scheduler/worker thread.  A full
    buffer drops the event and increments ``dropped_events``; the count is
    included in subsequent events and the close record.
    """

    def __init__(
        self,
        path: Optional[Path],
        run_id: str,
        workload_id: str,
        max_events: int = 65536,
        flush_every: int = 128,
    ):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.workload_id = workload_id
        self.flush_every = max(1, int(flush_every))
        self._queue: "queue.Queue[Optional[RuntimeEvent]]" = queue.Queue(
            maxsize=max(1, int(max_events))
        )
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self.dropped_events = 0
        self.writer_error: Optional[str] = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._writer_loop,
                name=f"pdmux-telemetry-{run_id}",
                daemon=True,
            )
            self._thread.start()

    def emit(
        self,
        event: str,
        phase: str,
        request_id: str = "",
        **fields: Any,
    ) -> bool:
        if self._closed or self.path is None:
            return False
        fields.setdefault("dropped_events", self.dropped_events)
        item = RuntimeEvent(
            run_id=self.run_id,
            workload_id=self.workload_id,
            phase=phase,
            event=event,
            request_id=request_id,
            fields=fields,
        )
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped_events += 1
            return False

    def mark_phase(self, phase: str) -> bool:
        return self.emit("phase_marker", phase)

    def _writer_loop(self) -> None:
        assert self.path is not None
        pending = 0
        try:
            with self.path.open("a", encoding="utf-8") as output:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    output.write(
                        json.dumps(item.to_dict(), separators=(",", ":"), sort_keys=True)
                        + "\n"
                    )
                    pending += 1
                    if pending >= self.flush_every:
                        output.flush()
                        pending = 0
                output.flush()
        except OSError as exc:
            self.writer_error = str(exc)

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        if self.path is not None:
            self.emit(
                "telemetry_close",
                "shutdown",
                dropped_events=self.dropped_events,
                writer_error=self.writer_error or "",
            )
            try:
                self._queue.put(None, timeout=max(0.0, timeout_s))
            except queue.Full:
                self.dropped_events += 1
            if self._thread is not None:
                self._thread.join(timeout=max(0.0, timeout_s))
                if self._thread.is_alive():
                    self.writer_error = "writer thread did not stop before timeout"
        self._closed = True

    def __enter__(self) -> "AsyncJsonlTelemetry":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
