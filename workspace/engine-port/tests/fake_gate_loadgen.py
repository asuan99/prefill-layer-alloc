#!/usr/bin/env python3
"""Fake mg_loadgen.py for G-gamma CPU tests: a FIFO single-server queue model, deterministic arrivals.
Env: FAKE_LAMBDA (default 30 req/s), FAKE_ON_SLOW (fractional capacity loss of ON instances), FAKE_ON_BREAK_PAIRS
('1,2': ON instances of those pairs return 10 % server errors), PDMUX_GATE_CELL_ON (set by the driver)."""
import argparse, json, os, random, re
from pathlib import Path
ap = argparse.ArgumentParser()
for k in ("--endpoint", "--trace", "--cell-dir"):
    ap.add_argument(k)
a = ap.parse_args()
cell = Path(a.cell_dir)
tr = [json.loads(l) for l in Path(a.trace).read_text().splitlines()]
n = len(tr); span = tr[-1]["arrival_s"]
rate = (n - 1) / span if span > 0 else 1.0
on = os.environ.get("PDMUX_GATE_CELL_ON") == "1"
lam = float(os.environ.get("FAKE_LAMBDA", "30")) * (1 - (float(os.environ.get("FAKE_ON_SLOW", "0")) if on else 0.0))
rnd = random.Random(str(cell))
lam *= 1 + rnd.uniform(-0.003, 0.003)
m = re.search(r"pair(\d+)_", str(cell)); pair = m.group(1).lstrip("0") if m else ""
brk = bool(on and pair and pair in os.environ.get("FAKE_ON_BREAK_PAIRS", "").split(","))
STALL = os.environ.get("FAKE_STALL") == "1"      # every 20th token ITL 40 ms -> ~5 % > 30 ms = STALL_BOUNDARY
svc, fin, rows = 1.0 / lam, 0.0, []
for i in range(n):
    sent = i / rate
    start = max(sent, fin); fin = start + svc
    err = brk and i % 10 == 0
    rows.append({"request_id": tr[i]["request_id"], "phase": "main", "input_tokens": 256, "requested_output_tokens": 512,
                 "observed_output_tokens": 512, "scheduled_arrival_s": sent, "sent_s": sent, "completion_s": fin,
                 "ttft_ms": None if err else (start - sent) * 1000 + 20.0,
                 "token_itl_ms": [] if err else [(40.0 if (STALL and (i * 8 + j) % 20 == 0) else 20.0) * (1 + rnd.uniform(-0.02, 0.02)) for j in range(8)],
                 "http_status": 500 if err else 200, "error": "HTTP_STATUS: 500" if err else "",
                 "error_kind": "HTTP_STATUS" if err else "", "error_side": "server" if err else ""})
(cell / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
(cell / "summary.json").write_text("{}")
