#!/usr/bin/env python3
"""Harness digest (audit r1 MAJOR-2).  The seal flag PDMUX_COSTTERMS_SEALED=1 alone is a self-declaration; the sealed Job must
also run exactly the harness bytes that were sealed.  `make` is run ONCE at the seal (PREREG §12 item 13) and its output
HARNESS_DIGEST.txt is committed with the seal; `check` is run by costterms_job.sh before anything else when SEALED=1 and
refuses (ABORT_HARNESS_DIGEST, rc 13 family) when the file is absent or any covered byte differs.

  ct_harness_digest.py make  [--dir D] [--out FILE]     print / write the digest JSON  (NOT run before the seal)
  ct_harness_digest.py check [--dir D] [--file FILE]    rc 0 ok, 13 absent / mismatch (one line per difference on stderr)

Covered: every ct_*.py, costterms_job.sh, the PD-mux config candidate, ENGINE_DIGEST.txt, the EXTERNAL files (analytic_terms.py,
make_variant.py, engine_digest.py, job_entry.sh), the canonical-spec sha256 (constants) and the plans sha256 (every Job's
registered plan incl. the COMPUTED per-cell O).  Not covered: tests (they are not executed by the Job)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = "pdmux_costterms_D64_candidate.yml"
EXTRA = ("costterms_job.sh", CONFIG, "ENGINE_DIGEST.txt")


EXTERNAL = ("scripts/costmodel/analytic_terms.py", "scripts/compo/make_variant.py", "scripts/bootstrap/engine_digest.py",
            "scripts/vessl/job_entry.sh")        # relative to workspace/engine-port (= HERE.parents[2]); audit r2 N-1


def covered(d):
    d = Path(d)
    return sorted([p.name for p in d.glob("ct_*.py")] + [n for n in EXTRA if (d / n).exists()])


def eng_of(d):
    return Path(d).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _run(d, *args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=str(d))
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else f"ERROR:{r.returncode}"


def spec_sha(d):
    return _run(d, str(Path(d) / "ct_spec.py"), "--sha256")


def plans_sha(d):
    return _run(d, str(Path(d) / "ct_runner.py"), "plans-sha")


def make(d=HERE):
    d = Path(d)
    eng = eng_of(d)
    return {"format": "harness-digest-v2", "files": {n: sha(d / n) for n in covered(d)},
            "external": {n: (sha(eng / n) if (eng / n).exists() else "ABSENT") for n in EXTERNAL},
            "spec_sha256": spec_sha(d), "plans_sha256": plans_sha(d)}


def check(d=HERE, file=None):
    """-> list of problems ([] = ok).  Covers the harness files, the EXTERNAL files the harness imports / runs (analytic_terms,
    make_variant, engine_digest, job_entry), the canonical spec and the fingerprint of every Job's registered plan."""
    d = Path(d)
    f = Path(file) if file else d / "HARNESS_DIGEST.txt"
    if not f.exists():
        return [f"HARNESS_DIGEST.txt absent ({f})"]
    try:
        want = json.loads(f.read_text())
    except ValueError as e:
        return [f"HARNESS_DIGEST.txt unreadable: {e}"]
    have = make(d)
    bad = []
    for n in sorted(set(want.get("files", {})) | set(have["files"])):
        if want.get("files", {}).get(n) != have["files"].get(n):
            bad.append(f"{n}: sealed {str(want.get('files', {}).get(n))[:12]} != installed {str(have['files'].get(n))[:12]}")
    for n in sorted(set(want.get("external", {})) | set(have["external"])):
        if want.get("external", {}).get(n) != have["external"].get(n):
            bad.append(f"external {n}: sealed {str(want.get('external', {}).get(n))[:12]} != installed {str(have['external'].get(n))[:12]}")
    for k in ("spec_sha256", "plans_sha256"):
        if have[k].startswith("ERROR"):
            bad.append(f"{k} cannot be computed in the installed tree ({have[k]})")
        elif want.get(k) != have[k]:
            bad.append(f"{k}: sealed {str(want.get(k))[:12]} != installed {have[k][:12]}")
    return bad


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0) if a else ""
    opt = {a[i]: a[i + 1] for i in range(0, len(a) - 1, 2)}
    d = opt.get("--dir", str(HERE))
    if cmd == "make":
        dg = make(d)
        errs = [k for k in ("spec_sha256", "plans_sha256") if dg[k].startswith("ERROR")] + \
               [n for n, v in dg["external"].items() if v == "ABSENT"]
        if errs:                                  # audit r2: never write a digest that contains an error value
            print(f"ABORT_HARNESS_DIGEST_MAKE: cannot digest {errs}", file=sys.stderr)
            return 2
        txt = json.dumps(dg, indent=1, sort_keys=True) + "\n"
        if "--out" in opt:
            Path(opt["--out"]).write_text(txt)
        else:
            print(txt, end="")
        return 0
    if cmd == "check":
        bad = check(d, opt.get("--file"))
        for b in bad:
            print(f"ABORT_HARNESS_DIGEST: {b}", file=sys.stderr)
        return 13 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
