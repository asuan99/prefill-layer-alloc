#!/usr/bin/env python3
"""Harness digest for the policy harness (same idea as ct_harness_digest.py, audit r1 MAJOR-2 of the cost-term harness).
`make` runs ONCE at the seal; `check` runs in ph_job.sh when PDMUX_POLICY_SEALED=1 and refuses (rc 13) on any differing byte.
Covered: ph_*.py, ph_job.sh, the PD-mux config candidate and ENGINE_DIGEST.txt (costterms_harness, shared), the cost-term harness
files this harness imports (ct_spec.py, ct_runner.py, ct_client.py, ct_extract.py, ct_fit.py), the benchmark loadgen, and the
structural plan sha (plans built with a CANONICAL addendum -- real rates arrive in the sealed addendum, which is hashed separately)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
ENG = HERE.parents[2]
CT = HERE.parent / "costterms_harness"
CANON = {"cap_ladder": {"W1": {"rmin": 1.0}, "W3": {"rmin": 0.1}}, "lambda_star": {"W1": 3.0, "W3": 0.3},
         "W3_floor_ms": 1000.0, "tuned_config": {"anchor_idx": 4, "dwell": 3}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def covered():
    own = [p for p in sorted(HERE.glob("ph_*.py")) if p.name != "ph_mutants.py"] + [HERE / "ph_job.sh", HERE / "ENGINE_DIGEST.txt"]
    ct = [CT / n for n in ("ct_spec.py", "ct_runner.py", "ct_client.py", "ct_extract.py", "ct_fit.py",
                           "pdmux_costterms_D64_candidate.yml")]
    ext = [ENG / "benchmarks/pdmux_eval/trace_loadgen.py", ENG / "benchmarks/pdmux_eval/analyze.py", ENG / "benchmarks/pdmux_eval/abs_mono_loadgen.py",
           ENG / "scripts/multigpu/mg_x_pretokenize.py", ENG / "scripts/bootstrap/engine_digest.py", ENG / "scripts/vessl/job_entry.sh"]
    return own, ct, ext


def make():
    import ph_spec
    if (HERE / "ENGINE_DIGEST.txt").read_text().strip() == ph_spec.ENGINE_DIGEST_PENDING:
        raise SystemExit("ABORT: ENGINE_DIGEST.txt is PENDING_FINAL_TREE; a harness digest cannot be made before the final tree digest")
    own, ct, ext = covered()
    d = {"files": {str(p.relative_to(ENG)): sha(p) for p in own + ct + ext}, "plans_sha": ph_spec.plans_sha(CANON)}
    d["digest"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def check(path):
    p = Path(path)
    if not p.exists():
        print("ABSENT digest file", file=sys.stderr)
        return 13
    want, got = json.loads(p.read_text()), make()
    bad = [k for k in sorted(set(want["files"]) | set(got["files"])) if want["files"].get(k) != got["files"].get(k)]
    if want.get("plans_sha") != got["plans_sha"]:
        bad.append("plans_sha")
    for b in bad:
        print(f"DIFF {b}", file=sys.stderr)
    return 13 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "check"])
    ap.add_argument("--out")
    ap.add_argument("--file", default=str(HERE / "HARNESS_DIGEST.txt"))
    a = ap.parse_args()
    if a.cmd == "make":
        d = make()
        (Path(a.out).write_text(json.dumps(d, indent=1, sort_keys=True) + "\n") if a.out else print(json.dumps(d, indent=1)))
    else:
        sys.exit(check(a.file))
