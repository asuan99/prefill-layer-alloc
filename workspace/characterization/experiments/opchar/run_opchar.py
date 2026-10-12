"""run_opchar.py — opchar driver: one worker subprocess per (phase, SM partition).

Modes (plan ~/.claude/plans/figure2-table2-linear-cherny.md):
  probe  Step 1 — profiling-capability facts (probe.py) + memcpy-vs-SM ceiling +
         a builder smoke (smallest sizes, 108 and one green partition, short timing).
  full   Step 2 — Fig 2 / Table 2 / Fig 3 grid on one Job:
         prefill L × SM, decode B × SM (ctx fixed), memcpy × SM, Kineto pass at every point.

  python3 run_opchar.py --mode probe --out-root DIR
  python3 run_opchar.py --mode full  --out-root DIR
  python3 run_opchar.py --mode full  --out-root DIR --dry-run   # print the plan only (CPU)

Partition grid: VESSL A100 green contexts realize 4-SM minimum, 2-SM steps
(results/vessl_substrate_probe_2026-10-04/.../SUMMARY.md B3-gran). Every point below is
even and >= 16 so request == realization is expected; the worker still records the
driver read-out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
WORKER = _HERE / "worker.py"
PROBE = _HERE / "probe.py"

SM_GRID = [16, 24, 32, 44, 54, 64, 74, 84, 92, 104, 108]
PREFILL_L = [1024, 2048, 4096, 16384, 32768]
DECODE_B = [32, 64, 96, 128]
DECODE_CTX = 2048
MODEL = "nemotron_nano_9b_v2"


def plan(mode: str) -> list:
    """[(label, argv)] — pure function, unit-tested."""
    w = [sys.executable, str(WORKER), "--model", MODEL]
    jobs = []
    if mode == "probe":
        jobs.append(("probe", [sys.executable, str(PROBE), "--green-sm", "44"]))
        for sm in SM_GRID:
            jobs.append((f"memcpy_sm{sm}", w + ["--phase", "memcpy", "--sm", str(sm)]))
        for sm in (108, 44):
            jobs.append((f"smoke_prefill_sm{sm}", w + ["--phase", "prefill", "--sizes", "1024", "--sm", str(sm),
                                                      "--n-measure", "5", "--profile"]))
            jobs.append((f"smoke_decode_sm{sm}", w + ["--phase", "decode", "--sizes", "32", "--ctx",
                                                     str(DECODE_CTX), "--sm", str(sm), "--n-measure", "5",
                                                     "--profile"]))
    elif mode == "full":
        for sm in SM_GRID:
            jobs.append((f"memcpy_sm{sm}", w + ["--phase", "memcpy", "--sm", str(sm)]))
        for sm in SM_GRID:
            jobs.append((f"prefill_sm{sm}", w + ["--phase", "prefill", "--sizes", *map(str, PREFILL_L),
                                                "--sm", str(sm), "--profile"]))
            jobs.append((f"decode_sm{sm}", w + ["--phase", "decode", "--sizes", *map(str, DECODE_B),
                                               "--ctx", str(DECODE_CTX), "--sm", str(sm), "--profile"]))
    else:
        raise ValueError(mode)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("probe", "full"), required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    jobs = plan(a.mode)
    out = Path(a.out_root)
    if a.dry_run:
        for label, cmd in jobs:
            print(label, " ".join(cmd[1:]))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": a.mode, "model": MODEL, "sm_grid": SM_GRID, "prefill_L": PREFILL_L,
                "decode_B": DECODE_B, "decode_ctx": DECODE_CTX, "runs": []}
    rc_all = 0
    for label, cmd in jobs:
        sub = out / ("probe" if label == "probe" else "data")
        full = cmd + ["--out-dir", str(sub)]
        t0 = time.time()
        print(f"[opchar] start {label}", flush=True)
        log = out / "logs" / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log, "w") as f:
                rc = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        manifest["runs"].append({"label": label, "rc": rc, "wall_s": round(time.time() - t0, 1),
                                 "cmd": full[1:]})
        print(f"[opchar] end {label} rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            rc_all = 1
        (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[opchar] DONE mode={a.mode} failures={sum(r['rc'] != 0 for r in manifest['runs'])}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
