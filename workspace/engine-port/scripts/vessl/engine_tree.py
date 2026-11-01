#!/usr/bin/env python3
"""Pristine vs patched SGLang tree: preserve, verify, and select per server boot (2026-11-01).

WHY.  The public image holds pristine SGLang v0.5.10 at $PDMUX_ROOT/sglang_engine_dev/python
(editable install), but job_entry.sh step 2 (install_runtime.sh) patches that tree IN PLACE with the
PD-mux sync + devtree_manual_edits.patch.  Every "fused/stock/OFF" arm so far therefore ran on the
PATCHED tree with flags off -- not on stock SGLang.  This tool lets one Job boot either tree:

  preserve  install_runtime.sh step 0 (job_entry.sh step 2), BEFORE sync/manual edits: cp -a the image
            tree to <SGLANG_ENGINE_DEV>/../../sglang_pristine/python (image: /opt/pdmux/sglang_pristine),
            verify it against the committed `git archive v0.5.10 python` manifest, write PRESERVE_STATUS.
            Exit 5 on FAIL; the caller decides (install_runtime.sh: fatal only with PDMUX_REQUIRE_PRISTINE=1).
  boot      server boot wrapper.  PDMUX_ENGINE_TREE=patched (default) | pristine.
              patched : env and argv passed through UNCHANGED; only checks/stamps the tree.
              pristine: re-verify the preserved copy, refuse engine PDMUX_* knobs and --enable-pdmux,
                        scrub every PDMUX_* from the server env, prepend the pristine root to
                        PYTHONPATH (dropping entries that point at the patched tree), check that
                        `import sglang` resolves there, then exec.
            Both write a JSON stamp (keys: see STAMP_KEYS) before exec.
  manifest  print the tree manifest of a directory (tests / ad-hoc).
  ref       regenerate the reference manifest from a SGLang git checkout (`git archive <tag> python`).

MANIFEST.  One `<sha256>  <relpath>` line per entry under the python/ dir, sorted by relpath bytes;
relpaths are relative to python/.  Regular file -> sha256 of its bytes; symlink -> sha256 of
b"symlink:" + link target (not followed).  Runtime caches are excluded everywhere: `__pycache__/`
dirs, `*.pyc`, and top-level `*.egg-info/` (written by `pip install -e`).  A preserved pristine tree
is OK iff its manifest restricted to the reference paths is byte-identical to the reference, every
non-reference entry is in GENERATED_ALLOW (setuptools_scm writes sglang/_version.py during the
image's editable install), and no project MARKER occurs in any file.

Stdlib only (install_runtime.sh runs it with `python3 -I -S`; use the same for `boot` -- the sglang
resolution check runs in a child with the server's own interpreter and env).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "pdmux.engine_tree_boot/v1"
TREES = ("patched", "pristine")
REF_REL = "workspace/engine-port/env/sglang_v0.5.10_python_tree.sha256"
GENERATED_ALLOW = ("sglang/_version.py",)
# Strings only the project's patches/sources add (upstream v0.5.10 has none of them; checked against
# `git archive v0.5.10 python`, see tests/test_engine_tree.py).  `\bPDMUX_[A-Z]` = an env-knob name;
# upstream's `_PDMUX_PREFILL_TP_GROUP` / `_ENABLE_PDMUX_P_TP` do not match (no word boundary).
MARKERS = (
    r"pdmux_role_is_thread_local",
    r"maybe_create_holb_probe",
    r"maybe_install_chunk_probe",
    r"maybe_install_admission_probe",
    r"maybe_install_decode_probe",
    r"\bPDMUX_[A-Z]",
)
REQUIRED_PATCHED_MARKERS = MARKERS[:3]          # the three the task names; all must be present on patched
_MARKER_RE = re.compile("|".join(MARKERS).encode())
# PDMUX_* that are Job/launch/harness plumbing, never read by the engine.  A pristine boot accepts them
# (and still scrubs them from the server env); any other PDMUX_* is an engine knob -> fail fast.
INFRA_ENV = frozenset({
    "PDMUX_COMMIT", "PDMUX_JOB_NAME", "PDMUX_CAMPAIGN", "PDMUX_TARGET", "PDMUX_IMAGE", "PDMUX_SPEC",
    "PDMUX_ARRAY_SPEC", "PDMUX_IO", "PDMUX_DATA", "PDMUX_OPT", "PDMUX_ROOT", "PDMUX_PROJECT_ROOT",
    "PDMUX_CPU_ONLY", "PDMUX_ENERGY_SAMPLER", "PDMUX_RUN_INDEX", "PDMUX_RUN_LOG_DIR",
    "PDMUX_TEST_SKIP_SYNC", "PDMUX_ENGINE_TREE", "PDMUX_REQUIRE_PRISTINE", "PDMUX_PRISTINE_REF",
})
# Non-PDMUX env gates that exist ONLY in the patched tree (models/nemotron_h.py, granitemoehybrid.py).
PATCHED_ONLY_GATES = ("SGLANG_NH_LAYER_TIMING", "SGLANG_GRANITE_TIMING")   # recorded in every stamp
PDMUX_ARGS = ("--enable-pdmux", "--pdmux-config-path")
STAMP_KEYS = (
    "engine_tree", "engine_tree_root", "engine_tree_sglang_init", "engine_tree_manifest_sha256",
    "engine_tree_manifest_entries", "engine_tree_ref_sha256", "engine_tree_ref_overridden",
    "engine_tree_markers_found", "engine_tree_extras",
)


class TreeError(RuntimeError):
    pass


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded(rel_parts: tuple[str, ...], is_dir: bool) -> bool:
    if "__pycache__" in rel_parts:
        return True
    if len(rel_parts) >= 1 and rel_parts[0].endswith(".egg-info"):
        return True
    return (not is_dir) and rel_parts[-1].endswith(".pyc")


def scan(root: Path) -> dict[str, str]:
    """relpath -> sha256 for every non-cache entry under root (symlinks not followed)."""
    root = Path(root)
    if not root.is_dir():
        raise TreeError(f"not a directory: {root}")
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        rel_d = d.relative_to(root).parts
        keep = []
        for n in dirnames:
            p = d / n
            if p.is_symlink():               # a symlinked dir is an entry, not a subtree
                filenames.append(n)
            elif not _excluded(rel_d + (n,), True):
                keep.append(n)
        dirnames[:] = keep
        for n in filenames:
            rel = rel_d + (n,)
            if _excluded(rel, False):
                continue
            p = d / n
            key = "/".join(rel)
            if p.is_symlink():
                out[key] = _sha256_bytes(b"symlink:" + os.fsencode(os.readlink(p)))
            elif p.is_file():
                out[key] = _sha256_file(p)
            else:
                raise TreeError(f"unsupported entry type: {p}")
    return out


def render(entries: dict[str, str], keys=None) -> str:
    ks = sorted(entries if keys is None else keys, key=lambda k: k.encode())
    return "".join(f"{entries[k]}  {k}\n" for k in ks)


def parse_manifest(text: str) -> dict[str, str]:
    out = {}
    for ln in text.splitlines():
        if not ln.strip():
            continue
        h, sep, rel = ln.partition("  ")
        if not sep or len(h) != 64:
            raise TreeError(f"bad manifest line: {ln!r}")
        out[rel] = h
    return out


def find_markers(root: Path, rels) -> dict[str, list[str]]:
    """{relpath: [marker, ...]} for regular files under root containing a project marker."""
    hits: dict[str, list[str]] = {}
    for rel in rels:
        p = Path(root) / rel
        if p.is_symlink() or not p.is_file():
            continue
        data = p.read_bytes()
        if _MARKER_RE.search(data):
            hits[rel] = sorted({m for m in MARKERS if re.search(m.encode(), data)})
    return hits


def verify_pristine(root: Path, ref_text: str) -> dict:
    """Return a result dict; result["ok"] is the verdict, result["reasons"] why not."""
    ref = parse_manifest(ref_text)
    entries = scan(root)
    reasons = []
    missing = sorted(set(ref) - set(entries))
    extras = sorted(set(entries) - set(ref))
    bad_extras = [e for e in extras if e not in GENERATED_ALLOW]
    changed = sorted(k for k in set(ref) & set(entries) if ref[k] != entries[k])
    if missing:
        reasons.append(f"missing {len(missing)} reference file(s), e.g. {missing[:3]}")
    if changed:
        reasons.append(f"{len(changed)} file(s) differ from v0.5.10, e.g. {changed[:5]}")
    if bad_extras:
        reasons.append(f"{len(bad_extras)} non-reference file(s), e.g. {bad_extras[:5]}")
    markers = find_markers(root, entries)
    if markers:
        reasons.append(f"project markers in {sorted(markers)[:5]}")
    restricted = render(entries, [k for k in ref if k in entries]) if not missing else ""
    if not missing and restricted != render(ref):
        reasons.append("restricted manifest is not byte-identical to the reference")
    return {
        "ok": not reasons, "reasons": reasons, "entries": entries, "extras": extras,
        "markers": markers, "restricted_manifest": restricted, "changed": changed, "missing": missing,
    }


def _kv_write(path: Path, d: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{k}={v}\n" for k, v in d.items()))
    os.replace(tmp, path)


def _kv_read(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        k, sep, v = ln.partition("=")
        if sep:
            out[k] = v
    return out


def _default_ref(project: str | None) -> Path | None:
    return Path(project) / REF_REL if project else None


# ---------------------------------------------------------------- preserve
def cmd_preserve(a) -> int:
    dst_root = Path(a.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    status_p = dst_root / "PRESERVE_STATUS"
    meta = Path(a.meta) if a.meta else None
    st = {"status": "FAIL", "utc": _utc(), "src": a.src or "", "dst": str(dst_root / "python")}

    def done(rc: int) -> int:
        _kv_write(status_p, st)
        if meta:
            meta.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(status_p, meta / "engine_tree_pristine_status.txt")
            for n in ("tree_manifest.sha256", "tree_extras.sha256"):
                if (dst_root / n).exists():
                    shutil.copyfile(dst_root / n, meta / f"engine_tree_pristine_{n}")
        print("".join(f"{k}={v}\n" for k, v in st.items()), end="")
        return rc

    try:
        ref_p = Path(a.ref)
        ref_text = ref_p.read_text()
        st["ref_path"] = str(ref_p)
        st["ref_sha256"] = _sha256_bytes(ref_text.encode())
        st["ref_overridden"] = "1" if a.ref_overridden else "0"
    except OSError as e:
        st["reason"] = f"reference manifest unreadable: {e}"
        return done(5)
    dst_py = dst_root / "python"

    def finish_ok(res, origin):
        (dst_root / "tree_manifest.sha256").write_text(res["restricted_manifest"])
        (dst_root / "tree_extras.sha256").write_text(render(res["entries"], res["extras"]))
        st.update(status="OK", origin=origin, reason="",
                  tree_manifest_sha256=_sha256_bytes(res["restricted_manifest"].encode()),
                  tree_entries=str(len(res["entries"])), extras=",".join(res["extras"]))
        return done(0)

    if dst_py.exists():                      # Workspace re-run: keep an existing copy only if it verifies
        res = verify_pristine(dst_py, ref_text)
        if res["ok"]:
            return finish_ok(res, "existing")
        shutil.rmtree(dst_py)
    if not a.src:
        st["reason"] = "SGLANG_ENGINE_DEV unset (no image tree to preserve)"
        return done(5)
    src = Path(a.src)
    if not src.is_dir():
        st["reason"] = f"source tree missing: {src}"
        return done(5)
    res = verify_pristine(src, ref_text)
    if not res["ok"]:                        # e.g. a Workspace whose tree install_runtime already patched
        st["reason"] = "source tree is not pristine v0.5.10: " + "; ".join(res["reasons"])
        return done(5)
    tmp = dst_root / "python.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    r = subprocess.run(["cp", "-a", str(src), str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        st["reason"] = f"cp -a failed: {r.stderr.strip()[:300]}"
        return done(5)
    res2 = verify_pristine(tmp, ref_text)
    if not res2["ok"] or res2["restricted_manifest"] != res["restricted_manifest"]:
        st["reason"] = "copy does not verify: " + "; ".join(res2["reasons"] or ["differs from source"])
        shutil.rmtree(tmp, ignore_errors=True)
        return done(5)
    os.replace(tmp, dst_py)
    return finish_ok(res2, "copied")


# ---------------------------------------------------------------- boot
_RESOLVER = (
    "import importlib.util, json, sys\n"
    "pre = 'sglang' in sys.modules\n"
    "s = importlib.util.find_spec('sglang')\n"
    "print(json.dumps({'preloaded': pre, 'origin': getattr(s, 'origin', None),\n"
    "  'locations': list(getattr(s, 'submodule_search_locations', None) or [])}))\n"
)


def _resolve_sglang(interp: str, env: dict) -> dict:
    r = subprocess.run([interp, "-c", _RESOLVER], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise TreeError(f"sglang resolver failed (rc={r.returncode}): {r.stderr.strip()[-400:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise TreeError(f"sglang resolver printed no JSON: {r.stdout[-300:]!r}")
    return json.loads(line[-1])


def _under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def _interp(cmd: list[str]) -> str:
    base = os.path.basename(cmd[0]) if cmd else ""
    if re.fullmatch(r"python(\d+(\.\d+)*)?", base):
        return shutil.which(cmd[0]) or cmd[0]
    return sys.executable


def plan_boot(env: dict, cmd: list[str], allow_env=()) -> tuple[dict, dict]:
    """Return (child_env, stamp).  Raises TreeError on any fail-fast condition."""
    tree = env.get("PDMUX_ENGINE_TREE", "patched") or "patched"
    if tree not in TREES:
        raise TreeError(f"PDMUX_ENGINE_TREE={tree!r}; expected one of {TREES}")
    if not cmd:
        raise TreeError("no server command after --")
    root_dir = env.get("PDMUX_ROOT") or env.get("PDMUX_OPT") or "/opt/pdmux"
    patched_root = Path(env.get("SGLANG_ENGINE_DEV") or f"{root_dir}/sglang_engine_dev/python")
    pristine_dst = patched_root.parent.parent / "sglang_pristine"   # same rule as install_runtime.sh step 0
    project = env.get("PDMUX_PROJECT_ROOT")
    stamp = {"schema": SCHEMA, "utc": _utc(), "engine_tree": tree, "argv": cmd,
             "pdmux_env": {k: env[k] for k in sorted(env) if k.startswith("PDMUX_")},
             "patched_only_gates": {k: env[k] for k in PATCHED_ONLY_GATES if k in env}}
    child = dict(env)
    if tree == "patched":
        root = patched_root
        entries = scan(root)
        markers = find_markers(root, entries)
        found = sorted({m for v in markers.values() for m in v})
        missing = [m for m in REQUIRED_PATCHED_MARKERS if m not in found]
        if missing:
            raise TreeError(f"patched tree {root} lacks project markers {missing}: install_runtime did not run?")
        manifest = render(entries)
        stamp.update(engine_tree_ref_sha256=None, engine_tree_ref_overridden=None, engine_tree_extras=None)
    else:
        bad_args = [x for x in cmd if x.split("=", 1)[0] in PDMUX_ARGS]
        if bad_args:
            raise TreeError(f"pristine boot with PD-mux argument(s) {bad_args}")
        allowed = INFRA_ENV | set(allow_env)
        knobs = sorted(k for k in env if k.startswith("PDMUX_") and k not in allowed)
        if knobs:
            raise TreeError(f"pristine boot with engine PDMUX_* env set: {knobs} "
                            "(declare harness-only names with --allow-env; they are scrubbed anyway)")
        st_p = pristine_dst / "PRESERVE_STATUS"
        if not st_p.exists():
            raise TreeError(f"{st_p} missing: job_entry.sh did not preserve the pristine tree")
        st = _kv_read(st_p)
        if st.get("status") != "OK":
            raise TreeError(f"pristine tree not available: {st.get('reason', '?')}")
        ref_p = Path(st.get("ref_path", ""))
        committed = _default_ref(project)
        if st.get("ref_overridden") != "1":
            if committed is None or not committed.exists():
                raise TreeError("PDMUX_PROJECT_ROOT unset or committed reference manifest missing")
            if _sha256_file(committed) != st.get("ref_sha256"):
                raise TreeError("preserved tree was verified against a different reference than the commit's")
        root = pristine_dst / "python"
        res = verify_pristine(root, ref_p.read_text())
        if not res["ok"]:
            raise TreeError("pristine tree changed since preservation: " + "; ".join(res["reasons"]))
        if res["restricted_manifest"] != (pristine_dst / "tree_manifest.sha256").read_text():
            raise TreeError("pristine tree manifest differs from the one recorded at preservation")
        manifest, entries, markers = res["restricted_manifest"], res["entries"], {}
        scrubbed = sorted(k for k in child if k.startswith("PDMUX_"))
        for k in scrubbed:
            del child[k]
        old = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        kept = [p for p in old if not _under(p, patched_root)]
        child["PYTHONPATH"] = os.pathsep.join([str(root)] + kept)
        stamp.update(engine_tree_ref_sha256=st.get("ref_sha256"),
                     engine_tree_ref_overridden=st.get("ref_overridden") == "1",
                     engine_tree_extras={e: entries[e] for e in res["extras"]},
                     pdmux_env_scrubbed=scrubbed, pythonpath_dropped=[p for p in old if p not in kept],
                     preserve_status=st)
    spec = _resolve_sglang(_interp(cmd), child)
    if spec.get("preloaded"):
        raise TreeError("sglang already imported at interpreter start (sitecustomize?); cannot verify tree")
    if not (_under(spec.get("origin"), root) and spec.get("locations")
            and all(_under(x, root) for x in spec["locations"])):
        raise TreeError(f"`import sglang` resolves to {spec.get('origin')}, not under {root}")
    stamp.update(engine_tree_root=str(root), engine_tree_sglang_init=spec["origin"],
                 engine_tree_manifest_sha256=_sha256_bytes(manifest.encode()),
                 engine_tree_manifest_entries=len(manifest.splitlines()),
                 engine_tree_markers_found=sorted({m for v in markers.values() for m in v}),
                 child_pythonpath=child.get("PYTHONPATH", ""))
    return child, stamp


def cmd_boot(a) -> int:
    cmd = list(a.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    stamp_p = Path(a.stamp)
    stamp_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        child, stamp = plan_boot(dict(os.environ), cmd, a.allow_env or ())
    except TreeError as e:
        fail = {"schema": SCHEMA, "utc": _utc(), "status": "REFUSED", "error": str(e),
                "engine_tree": os.environ.get("PDMUX_ENGINE_TREE", "patched"), "argv": cmd}
        stamp_p.with_name(stamp_p.name + ".refused.json").write_text(json.dumps(fail, indent=2) + "\n")
        print(f"ENGINE_TREE_REFUSED: {e}", file=sys.stderr)
        return 6
    stamp["status"] = "EXEC" if not a.check_only else "CHECK_ONLY"
    tmp = stamp_p.with_name(stamp_p.name + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=2) + "\n")
    os.replace(tmp, stamp_p)
    print(f"[engine_tree] {stamp['engine_tree']} root={stamp['engine_tree_root']} "
          f"manifest={stamp['engine_tree_manifest_sha256'][:12]}", file=sys.stderr, flush=True)
    if a.check_only:
        return 0
    os.execvpe(cmd[0], cmd, child)       # no return
    return 127


# ---------------------------------------------------------------- manifest / ref
def cmd_manifest(a) -> int:
    sys.stdout.write(render(scan(Path(a.root))))
    return 0


def cmd_ref(a) -> int:
    with tempfile.TemporaryDirectory() as td:
        arch = subprocess.run(["git", "-C", a.git, "archive", a.tag, "python"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", td], input=arch.stdout, check=True)
        entries = scan(Path(td) / "python")
        markers = find_markers(Path(td) / "python", entries)
        if markers:
            raise TreeError(f"reference tree contains project markers: {sorted(markers)}")
        text = render(entries)
    commit = subprocess.run(["git", "-C", a.git, "rev-parse", f"{a.tag}^{{commit}}"],
                            capture_output=True, text=True, check=True).stdout.strip()
    out = Path(a.out)
    out.write_text(text)
    out.with_suffix(".source").write_text(
        f"tag={a.tag}\ncommit={commit}\ngenerator=engine_tree.py ref --git <sglang> --tag {a.tag}\n"
        f"command=git archive {a.tag} python | tar -x\nentries={len(entries)}\n"
        f"manifest_sha256={_sha256_bytes(text.encode())}\n")
    print(f"{out}: {len(entries)} entries sha256={_sha256_bytes(text.encode())} (tag {a.tag} = {commit})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("preserve")
    p.add_argument("--src", default="")
    p.add_argument("--dst", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-overridden", action="store_true")
    p.add_argument("--meta")
    p.set_defaults(fn=cmd_preserve)
    p = sub.add_parser("boot")
    p.add_argument("--stamp", required=True)
    p.add_argument("--allow-env", action="append")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_boot)
    p = sub.add_parser("manifest")
    p.add_argument("root")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("ref")
    p.add_argument("--git", required=True)
    p.add_argument("--tag", default="v0.5.10")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_ref)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except TreeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
