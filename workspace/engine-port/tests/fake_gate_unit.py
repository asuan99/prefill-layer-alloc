#!/usr/bin/env python3
"""Fake conv1d_int64_unit.py: same CLI/exit codes (0 = RAN, 1 = STATE_MISMATCH, 2 = input error); behaviour predicted from the kernel
file text (CPU tests only).  Decision 20: the output sha of a correct run depends only on (dim, T) -- NOT on layout/width/kernel -- so
'patched view sha == pristine contig sha' is meaningful; a broken run (pristine view past the int32 wrap) either aborts or yields a
different sha.  FAKE_UNIT_MODE: patched_crash (patched @T95100 aborts) | patched_wrong (patched view @T95100 gives a wrong sha) |
co_state (co_only mutant reports a state mismatch) | co_same (co_only mutant is NOT broken: same sha as the patched kernel) |
every run reports max_abs_diff 0.05 like round 1 (descriptive)."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
for k in ("--kernel-file", "--tokens", "--layout", "--width-inproj", "--dim", "--offset", "--out"):
    ap.add_argument(k)
a = ap.parse_args()
src = Path(a.kernel_file).read_text(); T = int(a.tokens); w = int(a.width_inproj); dim = int(a.dim); off = int(a.offset)
if off + dim > w or T < 4:
    print("CONV1D_UNIT_INPUT_ERROR", file=sys.stderr); sys.exit(2)
mode = os.environ.get("FAKE_UNIT_MODE", "")
patched = "int64: token offsets" in src
co_only = (not patched) and ".to(tl.int64)" in src
good_sha = hashlib.sha256(f"{dim}:{T}".encode()).hexdigest()
diff = 0.05       # round-1 reality: descriptive max diff is ~0.05 at every T
if mode == "patched_crash" and patched and T == 95100:
    os.abort()
if mode == "py_exception" and patched and T == 95100 and a.layout == "view":
    sys.exit(1)        # python exception: rc 1, no output file
wrap = (not patched) and (not co_only) and a.layout == "view" and (T - 1) * w + dim - 1 > 2 ** 31 - 1
sha, rc = good_sha, 0
if wrap:
    if os.environ.get("FAKE_BROKEN", "crash") == "crash":
        os.abort()
    sha = hashlib.sha256(b"wrapped" + good_sha.encode()).hexdigest()
if co_only and a.layout == "view" and T * w > 2 ** 31 - 1:
    if mode == "co_state":
        rc = 1
    elif mode != "co_same":
        sha = hashlib.sha256(b"co" + good_sha.encode()).hexdigest()
if mode == "patched_wrong" and patched and a.layout == "view" and T == 95100:
    sha = hashlib.sha256(b"bad").hexdigest()
doc = {"verdict": "RAN" if rc == 0 else "STATE_MISMATCH", "out_sha256": sha, "max_abs_diff": diff, "state_ok": rc == 0}
Path(a.out).write_text(json.dumps(doc)); sys.exit(rc)
