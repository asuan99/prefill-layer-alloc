#!/usr/bin/env python3
"""Verify every job script carries the KISTI-mandated --comment directive.

Neuron policy (effective 2026-08-12 18:00 KST): a job is rejected at submit time
unless it declares `--comment="field=<field>;appl=<program>"` with values from
`showappl`. This project's fixed value is:

    #SBATCH --comment="field=efficientai;appl=pytorch"

  field=efficientai  Efficient & Scalable AI Systems -- PD-mux SM-partitioned
                     serving efficiency is exactly this bucket.
  appl=pytorch       SGLang is absent from the showappl program list; it is a
                     PyTorch-based engine, so we declare pytorch. Do NOT use
                     `vllm` -- that names a different engine and corrupts the
                     centre's utilisation statistics.

The check is position-aware, which a plain grep is not: Slurm stops parsing
`#SBATCH` directives at the first non-comment, non-blank line. A directive that
sits below `set -euo pipefail` is present in the file and still ignored by the
scheduler, so the job is rejected anyway. That failure mode is the reason this
script exists.

Discovery is by CONTENT, not by extension: any file whose header block contains
a `#SBATCH` line is a job script. Scoping to `*.sbatch` silently missed 26 job
scripts named `*.sh` in this repo (workspace/characterization/slurm/,
workspace/serving-eval/slurm/, slurm/), every one of which would have been
rejected at submit time. Do not narrow this back to a glob.

Usage:
    check_sbatch_comment.py                 # scan repo for job scripts
    check_sbatch_comment.py a.sbatch b.sh   # explicit paths (no content filter)
    check_sbatch_comment.py --fix           # insert/rewrite the directive

Exit status: 0 all conformant, 1 one or more violations, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import stat
import sys

FIELD = "efficientai"
APPL = "pytorch"
DIRECTIVE = f'#SBATCH --comment="field={FIELD};appl={APPL}"'

OK = "OK"
MISSING = "MISSING"  # no --comment anywhere Slurm will read
UNPARSED = "UNPARSED"  # present, but below the first executable line
WRONG = "WRONG"  # in the block, but not our value


# Allow-list, not a skip-list: a script Slurm executes is .sbatch / .sh / bare.
# A skip-list forced us to open all 30k .py and 14k .cubin files in this tree.
SCRIPT_SUFFIXES = {"", ".sbatch", ".sh", ".bash", ".slurm", ".job"}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "site-packages",
    "hf_cache", ".triton", "triton_cache", ".cache", "lib", "lib64", "include",
}


MAX_SCRIPT_BYTES = 1 << 20  # a job script is small; anything larger is data


def _discover(root: pathlib.Path) -> list[pathlib.Path]:
    """Every job script under root, found by content rather than extension.

    Walks with directory pruning and a size cap: this tree holds ~220k files,
    nearly all of them telemetry, so opening every candidate is not viable.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = pathlib.Path(dirpath) / name
            if path.suffix.lower() not in SCRIPT_SUFFIXES:
                continue  # prose quoting #SBATCH in a fenced block is not a job script
            try:
                st = path.lstat()
                if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_SCRIPT_BYTES:
                    continue
                with path.open("r", errors="strict") as fh:
                    head = [next(fh, "") for _ in range(80)]
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
            if any(line.startswith("#SBATCH") for line in head):
                found.append(path)
    return sorted(found)


def _classify(path: pathlib.Path) -> tuple[str, str]:
    """Return (status, detail) for one job script."""
    lines = path.read_text().splitlines()
    in_block = None  # first --comment Slurm would actually honour
    stop = len(lines)

    for i, line in enumerate(lines):
        if i == 0 and line.startswith("#!"):
            continue
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            if stripped.startswith("#SBATCH") and "--comment" in stripped and in_block is None:
                in_block = stripped
            continue
        stop = i  # first executable line: Slurm stops parsing here
        break

    if in_block == DIRECTIVE:
        return OK, ""
    if in_block is not None:
        return WRONG, f"line reads {in_block!r}"

    # Nothing usable in the block. Distinguish "absent" from "present but dead",
    # because the two need different fixes and the second one looks fine to grep.
    for line in lines[stop:]:
        if line.strip().startswith("#SBATCH") and "--comment" in line:
            return UNPARSED, "directive sits below the first executable line (Slurm ignores it)"
    return MISSING, "no --comment directive"


def _orphans(path: pathlib.Path) -> list[int]:
    """1-indexed lines holding a dead --comment below the parsed block.

    Slurm ignores these, so they are not submit failures -- but they read as if
    they were live config, which is how an UNPARSED file gets written in the
    first place. Reported, never auto-deleted: a `#SBATCH`-shaped line down
    there may be inside a heredoc or a deliberate comment.
    """
    lines = path.read_text().splitlines()
    stop = len(lines)
    for i, line in enumerate(lines):
        if i == 0 and line.startswith("#!"):
            continue
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        stop = i
        break
    return [
        stop + n + 1
        for n, line in enumerate(lines[stop:])
        if line.strip().startswith("#SBATCH") and "--comment" in line
    ]


def _fix(path: pathlib.Path) -> bool:
    """Insert or rewrite the directive in the parsed block. True if changed.

    A legacy `--comment` already inside the block is substituted in place, so a
    conversion shows up as a one-line diff instead of a delete-plus-append that
    reorders the header for no reason.
    """
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    last_sbatch = -1
    replaced = False
    insert_at = None

    for i, line in enumerate(lines):
        if i == 0 and line.startswith("#!"):
            out.append(line)
            continue
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            if stripped.startswith("#SBATCH") and "--comment" in stripped:
                if not replaced:
                    out.append(DIRECTIVE + "\n")  # substitute in place
                    replaced = True
                continue  # drop any further duplicates
            if stripped.startswith("#SBATCH"):
                last_sbatch = len(out)
            out.append(line)
            continue
        insert_at = len(out)
        out.extend(lines[i:])
        break
    else:
        insert_at = len(out)

    if not replaced:
        # Prefer right after the last #SBATCH; otherwise just before the first
        # executable line. Both land inside the block Slurm parses.
        out.insert(last_sbatch + 1 if last_sbatch >= 0 else insert_at, DIRECTIVE + "\n")

    new = "".join(out)
    if new == "".join(lines):
        return False
    path.write_text(new)
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=pathlib.Path, help="job scripts (default: scan by content)")
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."), help="scan root when no paths given")
    ap.add_argument("--fix", action="store_true", help="insert/rewrite the directive in place")
    args = ap.parse_args(argv)

    targets = args.paths or _discover(args.root)
    if not targets:
        print(f"no job scripts (files with #SBATCH) found under {args.root}", file=sys.stderr)
        return 2

    bad = []
    for path in targets:
        if not path.is_file():
            print(f"{path}: not a file", file=sys.stderr)
            return 2
        status, detail = _classify(path)
        if status != OK and args.fix:
            _fix(path)
            status, detail = _classify(path)
            if status == OK:
                print(f"FIXED    {path}")
        if status != OK:
            bad.append((path, status, detail))

    for path, status, detail in bad:
        print(f"{status:<8} {path}: {detail}")

    for path in targets:
        for line_no in _orphans(path):
            print(f"NOTE     {path}:{line_no}: dead --comment below the directive "
                  f"block (Slurm ignores it) -- delete by hand if it is not in a heredoc")

    print(f"\n{len(targets) - len(bad)}/{len(targets)} conformant  ({DIRECTIVE})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
