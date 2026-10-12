#!/usr/bin/env python3
"""Substrate gate of PREREG_MULTIGPU rev10 §16-3 (beside, not instead of, substrate_probe_helper score-b0).

  mg_substrate.py check --job J2 --gpu-csv b0/gpu.csv --topo b0/topo_m.txt [--out b0/MG_SUBSTRATE.json]
  mg_substrate.py --selftest

Rules (any failure -> ABORT_SUBSTRATE before GPU work):
  * GPU count == the Job SPEC's GPU count (mg_const.JOB_EXPECTED_GPUS; J1 = 2 although only GPU 0 is used),
  * every GPU name == "NVIDIA A100-SXM4-80GB" EXACTLY (a substring test passes A100-PCIe and A100-40GB: MGP-44),
  * every power limit == 400 W,
  * Job with >= 2 GPUs: `nvidia-smi topo -m` shows NV# on EVERY GPU pair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENG / "scripts" / "vessl"))
import mg_const as C  # noqa: E402
from substrate_probe_helper import parse_gpu_csv  # noqa: E402
from multigpu_probe_helper import parse_topo  # noqa: E402


def _power_w(txt):
    try:
        return float(str(txt).split()[0])
    except (ValueError, IndexError):
        return None


def judge_substrate(gpu_csv_text, topo_text, job, _mut=None):
    """`_mut` 'v' (tests only) = the retired substring name test (`"A100" in name`)."""
    gpu = parse_gpu_csv(gpu_csv_text)
    if not gpu:
        return {"status": "ABORT_SUBSTRATE", "reasons": ["no GPU row"]}
    rows = gpu.get("rows") or [gpu]
    want = C.JOB_EXPECTED_GPUS[job]
    why = []
    if gpu.get("gpu_count") != want:
        why.append(f"gpu_count {gpu.get('gpu_count')} != {want} (spec of {job})")
    name_ok = (lambda n: "A100" in (n or "")) if _mut == "v" else (lambda n: n == C.EXACT_GPU_NAME)
    bad = sorted({r.get("name") for r in rows if not name_ok(r.get("name"))})
    if bad:
        why.append(f"GPU name {bad} != {C.EXACT_GPU_NAME!r}")
    pl = [_power_w(r.get("power_limit")) for r in rows]
    if any(p != C.POWER_LIMIT_W for p in pl):
        why.append(f"power limit {pl} != {C.POWER_LIMIT_W} W")
    topo = None
    if len(rows) >= 2:
        topo = parse_topo(topo_text or "")
        if topo.get("n_gpu") != len(rows) or not topo.get("all_pairs_nvlink"):
            why.append(f"topology: not NV# on every pair (n_gpu={topo.get('n_gpu')}, classes={topo.get('classes')})")
    return {"status": "ABORT_SUBSTRATE" if why else "PASS", "reasons": why, "job": job, "expected_gpus": want,
            "names": sorted({r.get("name") for r in rows}), "power_limit_w": pl, "topology_classes": (topo or {}).get("classes")}


def _csv(name, n, power="400.00 W"):
    return "\n".join(f"{name}, GPU-{i:04d}, 550.1, 8.0, 81920 MiB, 1410 MHz, {power}, Disabled, Default" for i in range(n)) + "\n"


def _topo(n, link="NV12"):
    hdr = "\t" + "\t".join(f"GPU{i}" for i in range(n)) + "\tCPU Affinity\n"
    body = "".join(f"GPU{i}\t" + "\t".join(" X " if i == j else link for j in range(n)) + "\t0-21\n" for i in range(n))
    return hdr + body


def run_mutation_checks():
    """(v) partial-name gate must be KILLED (accepts A100-PCIe / 40GB), unmutated control must match on all scenarios."""
    sc = {
        "ok_J2": (_csv(C.EXACT_GPU_NAME, 2), _topo(2), "J2", "PASS"),
        "pcie_J2": (_csv("NVIDIA A100-PCIE-40GB", 2), _topo(2), "J2", "ABORT_SUBSTRATE"),
        "sxm_40gb_J2": (_csv("NVIDIA A100-SXM4-40GB", 2), _topo(2), "J2", "ABORT_SUBSTRATE"),
        "power_J4": (_csv(C.EXACT_GPU_NAME, 4, "300.00 W"), _topo(4), "J4", "ABORT_SUBSTRATE"),
        "pcie_topo_J8": (_csv(C.EXACT_GPU_NAME, 8), _topo(8, "PHB"), "J8", "ABORT_SUBSTRATE"),
        "count_J1_needs_2": (_csv(C.EXACT_GPU_NAME, 1), _topo(1), "J1", "ABORT_SUBSTRATE"),
        "ok_J1_two_gpus": (_csv(C.EXACT_GPU_NAME, 2), _topo(2), "J1", "PASS"),
    }
    per = {n: {"expected": e, "CONTROL": judge_substrate(g, t, j)["status"], "v": judge_substrate(g, t, j, _mut="v")["status"]}
           for n, (g, t, j, e) in sc.items()}
    verdict = {m: {"killed_on": [n for n, r in per.items() if r[m] != r["expected"]]} for m in ("v", "CONTROL")}
    for v in verdict.values():
        v["killed"] = bool(v["killed_on"])
    return verdict["v"]["killed"] and not verdict["CONTROL"]["killed"], verdict, per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", choices=("check",))
    ap.add_argument("--job", choices=C.JOBS)
    ap.add_argument("--gpu-csv")
    ap.add_argument("--topo")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        ok, verdict, _ = run_mutation_checks()
        print(json.dumps(verdict, sort_keys=True))
        print("mg_substrate selftest", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if a.cmd != "check" or not (a.job and a.gpu_csv):
        ap.error("check --job J --gpu-csv F --topo F")
    rd = lambda p: Path(p).read_text(errors="replace") if p and Path(p).exists() else ""
    r = judge_substrate(rd(a.gpu_csv), rd(a.topo), a.job)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    print(r["status"], "; ".join(r["reasons"]))
    return 0 if r["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
