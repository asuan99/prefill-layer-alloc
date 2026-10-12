#!/usr/bin/env python3
"""Path-free ENGINE_DIGEST (PREREG_MULTIGPU rev6 sec 6-3; harness verdict 2026-10-12 sec 6-1(b), F7(b)).

WHY.  `sync_engine_tree.sh` writes `runtime_source_manifest.sha256` as plain
`sha256sum` output: "<hash>  <ABSOLUTE path>".  The absolute prefix differs
between a local dev tree (`~/Experiments/KISTI/sglang_engine_dev_multigpu/python`)
and the VESSL container (`/opt/pdmux/sglang_engine_dev/python`), so the sha256 of
the manifest FILE can never be sealed locally and then match inside a Job
(verdict sec 4: "로컬 sha 봉인 시 Job ABORT_ENGINE_DIGEST").

DEFINITION (pathless-v1).  Let h_1..h_K be the first-column hashes of the K
registered manifest entries (K = 27 since 2026-10-12; the entry list is PARSED
from sync_engine_tree.sh, not copied, so the two cannot drift).  Then

    digest = sha256( "".join(h + "\\n" for h in sorted(h_1..h_K)) )   (ASCII, lowercase hex)

Sorting makes the serialization independent of path and order; each entry is
also structurally checked (count == K, the i-th path ends with the i-th
registered relative path, hash = 64 lowercase hex), so a manifest from a tree
with a different entry set is an INPUT ERROR (rc 2), never a silent digest.

The same digest is computed (a) from a manifest file (inside the Job, after
install_runtime.sh) and (b) directly from a tree by hashing the K registered
files (locally, before sealing).  Equality of (a) and (b) for the same bytes is
tested in tests/test_engine_digest.py.

CLI (exit codes: 0 ok/match, 1 mismatch, 2 input error -- never a verdict):
  engine_digest.py digest  (--manifest M | --runtime R) [--json]
  engine_digest.py check   (--manifest M | --runtime R) (--sealed HEX | --sealed-file F) [--out JSON]
  engine_digest.py compare-prefix (--manifest M | --runtime R)
                           (--reference REF_MANIFEST | --reference-root RESULTS_DIR) [--n 25] [--out JSON]

`compare-prefix` reports, entry by entry, whether the first n (default 25)
entries equal those of a reference manifest (by default the latest VESSL Job
manifest under RESULTS_DIR/*/vessl/*/meta/, latest = the 14-digit timestamp at
the end of the Job directory name).  It is a REPORT: a differing entry is a
fact about the trees, not a defect.

CPU only.  Reads files, writes only --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
SYNC_SCRIPT = HERE / "sync_engine_tree.sh"
SCHEME = "pathless-v1"
EXPECTED_ENTRIES = 27  # sync_engine_tree.sh since 2026-10-12 (entries 26-27 appended)
PREFIX_N_DEFAULT = 25  # verdict sec 6-1(b): "앞 25개는 직전 VESSL manifest와 대조"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB_TS = re.compile(r"-(\d{14})$")


class DigestInputError(ValueError):
    """The input cannot yield a pathless digest (rc 2, not a mismatch)."""


# ------------------------------------------------------------------ registered entry list
def registered_relpaths(sync_script: Path = SYNC_SCRIPT) -> List[str]:
    """The ordered `${runtime_python}/<rel>` arguments of the `sha256sum` block that writes the manifest."""
    text = Path(sync_script).read_text(encoding="utf-8")
    m = re.search(r'^sha256sum \\\n(.*?)>\s*"\$\{manifest_tmp\}"', text, flags=re.S | re.M)
    if not m:
        raise DigestInputError(f"{sync_script}: manifest sha256sum block not found")
    rels = re.findall(r'"\$\{runtime_python\}/([^"]+)"', m.group(1))
    if len(rels) != EXPECTED_ENTRIES or len(set(rels)) != len(rels):
        raise DigestInputError(f"{sync_script}: {len(rels)} entries (unique {len(set(rels))}) != {EXPECTED_ENTRIES}")
    return rels


# ------------------------------------------------------------------ manifests and trees
def parse_manifest(path: Path) -> List[Tuple[str, str]]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise DigestInputError(f"{path}:{i}: not '<hash>  <path>'")
        h, p = parts[0].lower(), parts[1].strip().lstrip("*")
        if not _HEX64.match(h):
            raise DigestInputError(f"{path}:{i}: first column is not a sha256 hex digest")
        rows.append((h, p))
    return rows


def manifest_entries(path: Path, relpaths: Sequence[str], require_all: bool = True) -> List[Tuple[str, str]]:
    """[(rel, hash)] in registered order.  require_all=False accepts a registered PREFIX
    (older manifests: 17, 24, 25 entries) -- used only for the reference side of compare-prefix."""
    rows = parse_manifest(path)
    if require_all and len(rows) != len(relpaths):
        raise DigestInputError(f"{path}: {len(rows)} entries != registered {len(relpaths)}")
    if len(rows) > len(relpaths):
        raise DigestInputError(f"{path}: {len(rows)} entries > registered {len(relpaths)}")
    out = []
    for i, ((h, p), rel) in enumerate(zip(rows, relpaths)):
        if not (p == rel or p.endswith("/" + rel)):
            raise DigestInputError(f"{path}: entry {i + 1} path {p!r} does not end with registered {rel!r}")
        out.append((rel, h))
    return out


def tree_entries(runtime_python: Path, relpaths: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for rel in relpaths:
        f = Path(runtime_python) / rel
        if not f.is_file():
            raise DigestInputError(f"{f}: registered entry missing from the tree")
        out.append((rel, hashlib.sha256(f.read_bytes()).hexdigest()))
    return out


def digest_of_hashes(hashes: Sequence[str]) -> str:
    hs = [h.lower() for h in hashes]
    bad = [h for h in hs if not _HEX64.match(h)]
    if bad:
        raise DigestInputError(f"{len(bad)} non-sha256 hash values")
    return hashlib.sha256("".join(h + "\n" for h in sorted(hs)).encode("ascii")).hexdigest()


def digest_of_entries(entries: Sequence[Tuple[str, str]]) -> str:
    if len(entries) != EXPECTED_ENTRIES:
        raise DigestInputError(f"{len(entries)} entries != {EXPECTED_ENTRIES}")
    return digest_of_hashes([h for _, h in entries])


def load_entries(manifest: Optional[str] = None, runtime: Optional[str] = None,
                 sync_script: Path = SYNC_SCRIPT) -> List[Tuple[str, str]]:
    if (manifest is None) == (runtime is None):
        raise DigestInputError("exactly one of --manifest / --runtime")
    rels = registered_relpaths(sync_script)
    if manifest is not None:
        if not Path(manifest).is_file():
            raise DigestInputError(f"{manifest}: manifest absent")
        return manifest_entries(Path(manifest), rels)
    return tree_entries(Path(runtime), rels)


def read_sealed(token: Optional[str] = None, path: Optional[str] = None) -> str:
    if path is not None:
        try:
            token = Path(path).read_text(encoding="utf-8")
        except OSError:
            token = "ABSENT"
    return "".join((token or "").split()).lower()


def check(actual_entries: Sequence[Tuple[str, str]], sealed: str) -> Dict:
    actual = digest_of_entries(actual_entries)
    sealed_ok = bool(_HEX64.match(sealed))
    match = sealed_ok and sealed == actual
    return {"scheme": SCHEME, "sealed": sealed, "actual": actual, "match": match,
            "why": None if match else ("sealed token is not a pathless-v1 digest (e.g. UNSEALED)"
                                       if not sealed_ok else "digest differs")}


# ------------------------------------------------------------------ prefix comparison
def latest_vessl_manifest(results_root: Path) -> Path:
    """Latest = largest 14-digit timestamp suffix of the Job directory name (not file mtime)."""
    cands = []
    for p in Path(results_root).glob("*/vessl/*/meta/runtime_source_manifest.sha256"):
        m = _JOB_TS.search(p.parents[1].name)
        if m:
            cands.append((m.group(1), str(p), p))
    if not cands:
        raise DigestInputError(f"{results_root}: no */vessl/<job>-<YYYYmmddHHMMSS>/meta/runtime_source_manifest.sha256")
    cands.sort()
    if len(cands) > 1 and cands[-1][0] == cands[-2][0]:
        raise DigestInputError(f"two Job manifests share the latest timestamp {cands[-1][0]}")
    return cands[-1][2]


def compare_prefix(entries: Sequence[Tuple[str, str]], ref_entries: Sequence[Tuple[str, str]],
                   n: int = PREFIX_N_DEFAULT) -> Dict:
    if len(ref_entries) < n or len(entries) < n:
        raise DigestInputError(f"need >= {n} entries on both sides (have {len(entries)}, ref {len(ref_entries)})")
    rows = []
    for i in range(n):
        (rel, h), (rrel, rh) = entries[i], ref_entries[i]
        if rel != rrel:
            raise DigestInputError(f"entry {i + 1}: {rel} vs reference {rrel}")
        rows.append({"index": i + 1, "rel": rel, "sha256": h, "reference_sha256": rh, "equal": h == rh})
    diff = [r["rel"] for r in rows if not r["equal"]]
    return {"n": n, "n_equal": n - len(diff), "differing": diff, "all_equal": not diff, "entries": rows,
            "beyond_prefix": [{"index": i + 1, "rel": rel, "sha256": h} for i, (rel, h) in enumerate(entries) if i >= n]}


# ------------------------------------------------------------------ CLI
def _src(p):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="runtime_source_manifest.sha256 (sha256sum output, any path prefix)")
    g.add_argument("--runtime", help="<sglang>/python tree to hash directly")


def _emit(doc, out):
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("digest")
    _src(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check")
    _src(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sealed")
    g.add_argument("--sealed-file")
    p.add_argument("--out")
    p = sub.add_parser("compare-prefix")
    _src(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference")
    g.add_argument("--reference-root", help="a results/ directory; latest */vessl/*/meta manifest is used")
    p.add_argument("--n", type=int, default=PREFIX_N_DEFAULT)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        ents = load_entries(a.manifest, a.runtime)
        if a.cmd == "digest":
            d = digest_of_entries(ents)
            print(_emit({"scheme": SCHEME, "digest": d, "entries": [{"rel": r, "sha256": h} for r, h in ents]}, None)
                  if a.json else d, end="" if a.json else "\n")
            return 0
        if a.cmd == "check":
            res = check(ents, read_sealed(a.sealed, a.sealed_file))
            res["source"] = a.manifest or a.runtime
            _emit(res, a.out)
            print(f"sealed={res['sealed'] or 'EMPTY'} actual={res['actual']} scheme={SCHEME} match={res['match']}")
            return 0 if res["match"] else 1
        ref = Path(a.reference) if a.reference else latest_vessl_manifest(Path(a.reference_root))
        rels = registered_relpaths()
        res = compare_prefix(ents, manifest_entries(ref, rels, require_all=False), a.n)
        res.update(source=a.manifest or a.runtime, reference=str(ref), scheme=SCHEME,
                   digest=digest_of_entries(ents))
        _emit(res, a.out)
        print(f"prefix {res['n_equal']}/{res['n']} equal to {ref}; differing: {res['differing'] or 'none'}; "
              f"{SCHEME} digest {res['digest']}")
        return 0 if res["all_equal"] else 1
    except DigestInputError as exc:
        print(f"ENGINE_DIGEST_INPUT_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
