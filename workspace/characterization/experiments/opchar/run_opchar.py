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
CTRL_SM = [44, 64, 92, 96, 100, 104, 108]
CTRL_L = [1024, 16384]
CTRL_B = [32, 128]
ORDER_SM = [16, 32, 44, 64, 74, 92, 104]       # ctrl2 shuffled; 108 bracketed first AND last
ORDER_SEED = 20261013


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
                                                      "--n-measure", "5", "--launch-log"]))
            jobs.append((f"smoke_decode_sm{sm}", w + ["--phase", "decode", "--sizes", "32", "--ctx",
                                                     str(DECODE_CTX), "--sm", str(sm), "--n-measure", "5",
                                                     "--launch-log"]))
    elif mode == "full":
        for sm in SM_GRID:
            jobs.append((f"memcpy_sm{sm}", w + ["--phase", "memcpy", "--sm", str(sm)]))
        for sm in SM_GRID:
            jobs.append((f"prefill_sm{sm}", w + ["--phase", "prefill", "--sizes", *map(str, PREFILL_L),
                                                "--sm", str(sm), "--launch-log"]))
            jobs.append((f"decode_sm{sm}", w + ["--phase", "decode", "--sizes", *map(str, DECODE_B),
                                               "--ctx", str(DECODE_CTX), "--sm", str(sm), "--launch-log"]))
    elif mode == "ctrl":
        # caveat control for the full Job (98f3a8c): which kernels/grids does each op launch at
        # 104 green vs 108 plain, and does a fine 92..104 sweep show the decode GEMM step?
        for sm in CTRL_SM:
            jobs.append((f"ctrl_prefill_sm{sm}", w + ["--phase", "prefill", "--sizes", *map(str, CTRL_L),
                                                     "--sm", str(sm), "--graph-census", "--launch-log"]))
            jobs.append((f"ctrl_decode_sm{sm}", w + ["--phase", "decode", "--sizes", *map(str, CTRL_B),
                                                    "--ctx", str(DECODE_CTX), "--sm", str(sm),
                                                    "--graph-census"]))
    elif mode == "ctrl2":
        # audit 2026-10-13 resolving experiments (AUDIT_opchar_2026-10-13.md)
        jobs.append(("ncu_launchstats_only", [sys.executable, str(PROBE), "--ncu-launchstats-only"]))
        jobs.append(("abx_104g_106g_108p", w + ["--phase", "abx", "--sm", "108"]))
        import random
        order = list(ORDER_SM)
        random.Random(ORDER_SEED).shuffle(order)
        for k, sm in enumerate([108] + order + [108]):
            tag = f"o{k:02d}"
            jobs.append((f"order_{tag}_prefill_sm{sm}", w + ["--phase", "prefill", "--sizes", "16384",
                                                            "--sm", str(sm), "--graph-census",
                                                            "--out-subdir", f"order/{tag}"]))
            jobs.append((f"order_{tag}_decode_sm{sm}", w + ["--phase", "decode", "--sizes", "128", "--ctx",
                                                           str(DECODE_CTX), "--sm", str(sm),
                                                           "--out-subdir", f"order/{tag}"]))
        for sm in (108, 104):
            jobs.append((f"cublaslog_sm{sm}", w + ["--phase", "prefill", "--sizes", "1024", "16384", "--sm",
                                                  str(sm), "--n-measure", "3", "--n-warmup", "1",
                                                  "--out-subdir", f"cublaslog/sm{sm}"],
                         {"CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_FILE": f"@OUT@/cublaslog/sm{sm}/cublasLt_%i.log",
                          "CUBLAS_LOGINFO_DBG": "1", "CUBLAS_LOGDEST_DBG": f"@OUT@/cublaslog/sm{sm}/cublas.log"}))
    else:
        raise ValueError(mode)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("probe", "full", "ctrl", "ctrl2"), required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    jobs = plan(a.mode)
    out = Path(a.out_root)
    if a.dry_run:
        for job in jobs:
            print(job[0], " ".join(job[1][1:]), *(job[2:] or []))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": a.mode, "model": MODEL, "sm_grid": SM_GRID, "prefill_L": PREFILL_L,
                "decode_B": DECODE_B, "decode_ctx": DECODE_CTX, "runs": []}
    rc_all = 0
    import os
    from datetime import datetime, timezone
    utc = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for job in jobs:
        label, cmd = job[0], list(job[1])
        env = dict(os.environ)
        for k, v in (job[2] if len(job) > 2 else {}).items():
            env[k] = v.replace("@OUT@", str(out))
            if k.endswith(("_FILE", "_DBG")) and "/" in env[k]:
                Path(env[k]).parent.mkdir(parents=True, exist_ok=True)
        sub = out / ("probe" if label in ("probe", "ncu_launchstats_only") else "data")
        if "--out-subdir" in cmd:
            i = cmd.index("--out-subdir")
            sub = sub / cmd[i + 1]
            del cmd[i:i + 2]
        full = cmd + ["--out-dir", str(sub)]
        t0 = time.time()
        start_utc = utc()
        print(f"[opchar] start {label}", flush=True)
        log = out / "logs" / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log, "w") as f:
                rc = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout_s, env=env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        manifest["runs"].append({"label": label, "rc": rc, "wall_s": round(time.time() - t0, 1),
                                 "start_utc": start_utc, "end_utc": utc(), "cmd": full[1:],
                                 "env_extra": {k: env[k] for k in (job[2] if len(job) > 2 else {})}})
        print(f"[opchar] end {label} rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            rc_all = 1
        (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[opchar] DONE mode={a.mode} failures={sum(r['rc'] != 0 for r in manifest['runs'])}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
