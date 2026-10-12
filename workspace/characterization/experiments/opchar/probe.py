"""probe.py — opchar Step 1: what profiling is possible on THIS substrate (no verdicts).

Records facts only; the campaign design reads them afterwards.
  (a) CUPTI Activity via torch.profiler (Kineto): kernel events + launch config present?
      on the plain stream AND on a green-context stream (unknown on VESSL; KISTI ncu failed
      under green contexts — kernel_mech/p1_probe/P1_VERDICT_2026-08-20.md, not portable).
  (b) Nsight tools: `ncu` / `nsys` presence; ncu with counter-free sections only
      (LaunchStats, Occupancy) and with one counter metric, raw output kept.
  (c) RmProfilingAdminOnly and process capabilities (context for (b)).

  python3 probe.py --out-dir DIR [--green-sm 44]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CHAR = _HERE.parent.parent
_WS = _CHAR.parent
for _p in (str(_WS), str(_CHAR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GEMM_SNIPPET = r"""
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
print("gemm ok")
"""


def kineto_check(stream_kind: str, green_sm: int, out_dir: Path, env_extra=None, tag="") -> dict:
    """Run in a subprocess (green contexts leak); returns the child's JSON."""
    code = f"""
import json, sys
sys.path[:0] = [{str(_WS)!r}, {str(_CHAR)!r}]
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from experiments.opchar import worker as W, trace_parse as TP
rec = {{"stream": {stream_kind!r}}}
try:
    st, info = W.open_partition({green_sm if stream_kind == 'green' else 10**6})
    rec["partition"] = {{k: v for k, v in info.items() if not k.startswith("_")}}
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.cuda.stream(st):
        a @ a
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(st):
            with record_function(TP.RANGE_PREFIX + "probe:gemm"):
                for _ in range(3):
                    a @ a
        torch.cuda.synchronize()
    path = {str(out_dir)!r} + "/kineto_{stream_kind}{tag}.json"
    prof.export_chrome_trace(path)
    s = TP.load_trace(path)
    ks = [k for k in s.kernels if k.cat == "kernel"]
    rec.update(n_kernel_events=len(ks), n_attributed=sum(k.op != "unattributed" for k in ks),
               has_grid=all(k.grid for k in ks) if ks else False,
               has_regs=all(k.regs is not None for k in ks) if ks else False,
               has_smem=all(k.smem is not None for k in ks) if ks else False,
               n_launch_events=sum(1 for e in json.load(open(path))["traceEvents"]
                                   if e.get("cat") in ("cuda_runtime", "cuda_driver")),
               sample=[dict(name=k.name[:120], grid=k.grid, block=k.block, regs=k.regs,
                            smem=k.smem, dur_us=k.dur_us, op=k.op) for k in ks[:3]])
    rec["kineto_available"] = torch.profiler.kineto_available() if hasattr(torch.profiler, "kineto_available") else None
    rec["cupti_mapped"] = sorted(set(l.split()[-1] for l in open("/proc/self/maps") if "cupti" in l.lower()))
except Exception as exc:
    import traceback
    rec["error"] = repr(exc); rec["tb"] = traceback.format_exc()[-3000:]
print("PROBE_JSON " + json.dumps(rec, default=str))
"""
    env = dict(os.environ, KINETO_LOG_LEVEL="0", **(env_extra or {}))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            r = json.loads(line[len("PROBE_JSON "):])
            r["rc"] = p.returncode
            r["env_extra"] = env_extra or {}
            r["stderr_profiler_lines"] = [l[:300] for l in p.stderr.splitlines()
                                          if any(w in l.lower() for w in ("cupti", "kineto", "activity", "profil"))][-40:]
            return r
    return {"stream": stream_kind, "rc": p.returncode, "stdout": p.stdout[-3000:],
            "stderr": p.stderr[-3000:]}


def _run(cmd, timeout=600, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"cmd": cmd, "rc": p.returncode, "stdout": p.stdout[-6000:], "stderr": p.stderr[-6000:]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def nsight_check(out_dir: Path) -> dict:
    rec = {"which": {t: shutil.which(t) for t in ("ncu", "nsys", "nvidia-smi")}}
    snippet = out_dir / "probe_gemm.py"
    snippet.write_text(GEMM_SNIPPET)
    ncu = rec["which"]["ncu"]
    if ncu:
        rec["ncu_version"] = _run([ncu, "--version"], 60)
        rec["ncu_launch_sections"] = _run(
            [ncu, "--section", "LaunchStats", "--section", "Occupancy", "--launch-count", "1",
             sys.executable, str(snippet)])
        rec["ncu_counter_metric"] = _run(
            [ncu, "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
             "--launch-count", "1", sys.executable, str(snippet)])
        for k in ("ncu_launch_sections", "ncu_counter_metric"):
            txt = rec[k].get("stdout", "") + rec[k].get("stderr", "")
            rec[k]["mentions_ERR_NVGPUCTRPERM"] = "ERR_NVGPUCTRPERM" in txt
    nsys = rec["which"]["nsys"]
    if nsys:
        rec["nsys_version"] = _run([nsys, "--version"], 60)
        rec["nsys_trace_cuda"] = _run(
            [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "-f", "true",
             "-o", str(out_dir / "nsys_probe"), sys.executable, str(snippet)])
    return rec


def host_context() -> dict:
    rec = {}
    try:
        for line in Path("/proc/driver/nvidia/params").read_text().splitlines():
            if "Profiling" in line:
                rec.setdefault("params", []).append(line)
    except Exception as exc:  # noqa: BLE001
        rec["params_error"] = repr(exc)
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("CapEff", "Uid")):
                rec[line.split(":")[0]] = line.split(":", 1)[1].strip()
    except Exception as exc:  # noqa: BLE001
        rec["status_error"] = repr(exc)
    return rec


def cupti_inventory() -> dict:
    import glob
    pats = ["/usr/local/cuda*/extras/CUPTI/lib64/libcupti*", "/usr/local/cuda*/lib64/libcupti*",
            "/opt/conda/lib/libcupti*", "/opt/conda/lib/python3*/site-packages/nvidia/*/lib/libcupti*",
            "/opt/conda/lib/python3*/site-packages/torch/lib/libcupti*"]
    found = sorted({f for p in pats for f in glob.glob(p)})
    ld = _run(["/sbin/ldconfig", "-p"], 60)
    return {"files": found, "ldconfig_cupti": [l.strip() for l in ld.get("stdout", "").splitlines() if "cupti" in l],
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--green-sm", type=int, default=44)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = cupti_inventory()
    rec = {"host": host_context(), "cupti_inventory": inv,
           "kineto_plain": kineto_check("plain", a.green_sm, out),
           "kineto_green": kineto_check("green", a.green_sm, out),
           "nsight": nsight_check(out)}
    # retry with each CUPTI directory found first on the loader path (facts only)
    for i, d in enumerate(sorted({os.path.dirname(f) for f in inv["files"]})):
        lp = d + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
        rec[f"kineto_plain_ldpath{i}"] = kineto_check("plain", a.green_sm, out, {"LD_LIBRARY_PATH": lp}, f"_ld{i}")
        rec[f"kineto_green_ldpath{i}"] = kineto_check("green", a.green_sm, out, {"LD_LIBRARY_PATH": lp}, f"_ld{i}")
    (out / "PROBE_FACTS.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    kp, kg = rec["kineto_plain"], rec["kineto_green"]
    print(f"[opchar-probe] kineto plain: kernels={kp.get('n_kernel_events')} grid={kp.get('has_grid')} "
          f"| green: kernels={kg.get('n_kernel_events')} grid={kg.get('has_grid')} "
          f"| ncu={rec['nsight']['which']['ncu']} nsys={rec['nsight']['which']['nsys']}")


if __name__ == "__main__":
    main()
