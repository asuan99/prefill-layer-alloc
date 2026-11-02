#!/usr/bin/env python3
"""HARNESS_DIGEST of PREREG_STOCK_BASE rev1 (PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md).  `make` runs ONCE at the seal; `check` runs
in sb_job.sh when PDMUX_SB_SEALED=1 (rc 13 on any differing byte).  `make` refuses unless ENGINE_DIGEST.txt holds the registered 30-entry engine digest 2836112b... (the tree with the
decode-probe EXTEND flag, dev_tree_edits #39).
Covered: every sb_* file except sb_mutants.py, sb_job.sh, ENGINE_DIGEST.txt, the A5 forecast R_SER_FORECAST_NANO.csv (§11-5 seal), the
sealed documents SEALED_DOCS (the PREREG, the Phase B prediction document, harness audits r1 and r2 -- rev2.1, re-audit r2 (a)); the reused sealed policy / cost-term harness files (bytes are also pinned by their own HARNESS_DIGEST, here they are re-hashed so a
change cannot slip in between); the loadgens; engine_tree.py + the pristine reference manifest; decode_probe.py (pacing_ratios is an
analysis input); engine_digest.py, job_entry.sh, install_runtime.sh; and the structural plan sha of each Job (canonical addendum)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
ENG = HERE.parents[1]
REUSED = ("results/cap_char/policy_harness/ph_spec.py", "results/cap_char/policy_harness/ph_trace.py",
          "results/cap_char/policy_harness/ph_verdict.py", "results/cap_char/policy_harness/ph_loadgen.py",
          "results/cap_char/policy_harness/ph_runner.py",
          "results/cap_char/costterms_harness/ct_spec.py", "results/cap_char/costterms_harness/ct_runner.py",
          "results/cap_char/costterms_harness/ct_client.py", "results/cap_char/costterms_harness/ct_extract.py",
          "results/cap_char/costterms_harness/ct_fit.py", "results/cap_char/costterms_harness/pdmux_costterms_D64_candidate.yml",
          "benchmarks/pdmux_eval/trace_loadgen.py", "benchmarks/pdmux_eval/abs_mono_loadgen.py", "benchmarks/pdmux_eval/analyze.py",
          "scripts/multigpu/mg_x_pretokenize.py", "scripts/vessl/engine_tree.py", "env/sglang_v0.5.10_python_tree.sha256",
          "src/multiplex/decode_probe.py", "scripts/bootstrap/engine_digest.py", "scripts/vessl/job_entry.sh",
          "scripts/vessl/install_runtime.sh")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def own_files():
    names = sorted(p.name for p in HERE.glob("sb_*.py") if p.name != "sb_mutants.py")
    return [HERE / n for n in names] + [HERE / "sb_job.sh", HERE / "ENGINE_DIGEST.txt", HERE / "R_SER_FORECAST_NANO.csv"] + \
        [HERE / n for n in SEALED_DOCS]


SEALED_DOCS = ("PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md", "PREDICTION_phaseB_expected_effects_2026-11-02.md",
               "AUDIT_harness_stock_base_r1_2026-11-02.md", "AUDIT_harness_stock_base_r2_2026-11-02.md")   # rev2.1: pre-registered docs


def make():
    import sb_spec as SP
    got = (HERE / "ENGINE_DIGEST.txt").read_text().strip()
    if got != SP.ENGINE_DIGEST_EXPECTED:
        raise SystemExit(f"ABORT: ENGINE_DIGEST.txt = {got[:12]}..., registered {SP.ENGINE_DIGEST_EXPECTED[:12]}...")
    d = {"files": {str(p.relative_to(ENG)): sha(p) for p in own_files()},
         "reused": {rel: sha(ENG / rel) for rel in REUSED}, "plans_sha": SP.plans_sha()}
    d["digest"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def check(path):
    p = Path(path)
    if not p.exists():
        print("ABSENT digest file", file=sys.stderr)
        return 13
    want, got = json.loads(p.read_text()), make()
    bad = []
    for sect in ("files", "reused", "plans_sha"):
        for k in sorted(set(want.get(sect, {})) | set(got[sect])):
            if want.get(sect, {}).get(k) != got[sect].get(k):
                bad.append(f"{sect}:{k}")
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
