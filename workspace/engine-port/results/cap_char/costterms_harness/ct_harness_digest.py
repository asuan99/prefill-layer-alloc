#!/usr/bin/env python3
"""Harness digest (audit r1 MAJOR-2).  The seal flag PDMUX_COSTTERMS_SEALED=1 alone is a self-declaration; the sealed Job must
also run exactly the harness bytes that were sealed.  `make` is run ONCE at the seal (PREREG §12 item 13) and its output
HARNESS_DIGEST.txt is committed with the seal; `check` is run by costterms_job.sh before anything else when SEALED=1 and
refuses (ABORT_HARNESS_DIGEST, rc 13 family) when the file is absent or any covered byte differs.

  ct_harness_digest.py make  [--dir D] [--out FILE]     print / write the digest JSON  (NOT run before the seal)
  ct_harness_digest.py check [--dir D] [--file FILE]    rc 0 ok, 13 absent / mismatch (one line per difference on stderr)

Covered: every ct_*.py, costterms_job.sh, the PD-mux config candidate, ENGINE_DIGEST.txt, and the canonical-spec sha256
(ct_spec.py --sha256, i.e. every registered value).  Not covered: tests (they are not executed by the Job)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = "pdmux_costterms_D64_candidate.yml"
EXTRA = ("costterms_job.sh", CONFIG, "ENGINE_DIGEST.txt")


def covered(d):
    d = Path(d)
    return sorted([p.name for p in d.glob("ct_*.py")] + [n for n in EXTRA if (d / n).exists()])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spec_sha(d):
    r = subprocess.run([sys.executable, str(Path(d) / "ct_spec.py"), "--sha256"], capture_output=True, text=True, cwd=str(d))
    return r.stdout.strip() if r.returncode == 0 else f"ERROR:{r.returncode}"


def make(d=HERE):
    d = Path(d)
    return {"format": "harness-digest-v1", "files": {n: sha(d / n) for n in covered(d)}, "spec_sha256": spec_sha(d)}


def check(d=HERE, file=None):
    """-> list of problems ([] = ok)."""
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
    if have["spec_sha256"].startswith("ERROR") and d.resolve() == HERE:      # a scratch copy cannot import analytic_terms; the real tree must
        bad.append(f"spec_sha256 cannot be computed in the installed tree ({have['spec_sha256']})")
    if want.get("spec_sha256") != have["spec_sha256"]:
        bad.append(f"spec_sha256: sealed {str(want.get('spec_sha256'))[:12]} != installed {have['spec_sha256'][:12]}")
    return bad


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0) if a else ""
    opt = {a[i]: a[i + 1] for i in range(0, len(a) - 1, 2)}
    d = opt.get("--dir", str(HERE))
    if cmd == "make":
        txt = json.dumps(make(d), indent=1, sort_keys=True) + "\n"
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
