#!/usr/bin/env python3
"""N1 regression test (rev3 addendum B-5(a)#2 fix, 2026-08-16).

Verifies -- by extracting and SOURCING the ACTUAL bash text out of
`g16_grid.sbatch` (not a reimplementation, so this cannot silently drift from
the real harness) -- that:

  A. A synthetic ARTIFACT-INVALID boot (FIELDS_HI_RC=1) writes a sidecar with
     status="artifact_invalid", NOT "completed".
  B. A synthetic HEALTHY boot (TELEM_RC=FIELDS_LO_RC=FIELDS_HI_RC=0) still
     writes status="completed" (no regression on the passing path).
  C. Re-running the OLD (pre-fix) call order -- `write_sidecar ... completed`
     BEFORE the ARTIFACT_OK judgment -- against the SAME (unmodified)
     write_sidecar() function reproduces the exact defect B-5(a)#2 named:
     status="completed" co-occurring with fields_rc.hi=1. This proves the bug
     was in the CALL SITE/ORDER, not in write_sidecar() itself (which this
     fix left untouched), and gives a concrete "what it used to look like"
     artifact for anyone re-auditing this fix.

Extraction is anchored on two markers this test depends on staying in
`g16_grid.sbatch`:
  - the literal line `write_sidecar() {` (bash function definition)
  - the sentinel comments `G16_N1_BLOCK_START` / `G16_N1_BLOCK_END` around
    the post-fix ARTIFACT_OK-then-write_sidecar control flow.
If either goes missing, this test fails loudly (AssertionError) instead of
silently testing stale text.

GPU 0. No server, no real telemetry/bench data, no sglang/torch import --
write_sidecar()'s python body is stdlib-only (json/os/sys), so this runs
under plain system python3, matching G16_HARNESS_NOTES's existing claim that
that function has "0 torch/GPU deps".
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SBATCH = os.path.join(HERE, "g16_grid.sbatch")


def _extract_write_sidecar(text: str) -> str:
    """Return the exact `write_sidecar() { ... }` function body, correctly
    skipping the `}` characters that appear INSIDE the function's embedded
    python heredoc (e.g. the JSON dict literal's closing `}` at column 0) by
    only accepting a bare `}` line that comes AFTER the heredoc's `PYEOF`
    terminator line has been seen.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "write_sidecar() {")
    seen_heredoc_end = False
    for i in range(start + 1, len(lines)):
        if lines[i] == "PYEOF":
            seen_heredoc_end = True
            continue
        if seen_heredoc_end and lines[i] == "}":
            return "\n".join(lines[start : i + 1])
    raise AssertionError("write_sidecar() closing brace not found after a PYEOF line")


def _extract_n1_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "G16_N1_BLOCK_START" in l)
    # Search strictly AFTER `start` -- guards against the START line's own
    # descriptive comment accidentally containing the END marker's literal
    # text (bit us once during this test's own development: the marker text
    # is deliberately NOT self-referential now, but this keeps the extractor
    # robust even if a future edit reintroduces that).
    end = next(i for i in range(start + 1, len(lines)) if "G16_N1_BLOCK_END" in lines[i])
    return "\n".join(lines[start : end + 1])


def _run_boot(tmpdir: str, *, telem_rc: int, fields_lo_rc: int, fields_hi_rc: int,
              extra_script: str, label: str) -> dict:
    text = open(SBATCH).read()
    write_sidecar_src = _extract_write_sidecar(text)
    runid = "g16_test_" + label

    preamble = "\n".join(
        [
            "set -uo pipefail",
            "HERE=" + shlex.quote(tmpdir),
            "BLOCK_TAG=testblk",
            "NODE_NAME=testnode",
            "GPU_UUID=testgpu",
            "MANIFEST_SHA=deadbeef",
            "GIT_COMMIT=deadbeef",
            "HARNESS_SHA_JSON='{}'",
            "MODE_TAG=full",
            "ARM=d44",
            "BOOT_IDX=1",
            "NP=200",
            "ROUNDS=3",
            "RUNID=" + shlex.quote(runid),
            'JLO="$HERE/${RUNID}_LO.jsonl"',
            'JHI="$HERE/${RUNID}_HI.jsonl"',
            'PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"',
            "TELEM_RC=" + str(telem_rc),
            "FIELDS_LO_RC=" + str(fields_lo_rc),
            "FIELDS_HI_RC=" + str(fields_hi_rc),
            "PIN_RC=0",
            "CTRLSUMMARY_RC=0",
            "BOOTS_FAILED=0",
            'FAILED_ARMS=""',
            # Stub -- H16's rename-on-failure behavior is not what this test
            # is about; a no-op keeps the extracted N1 block runnable as-is.
            "mark_failed_artifacts() { :; }",
        ]
    )
    script = preamble + "\n\n" + write_sidecar_src + "\n\n" + extra_script + "\n"

    r = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"bash exited {r.returncode} for case={label!r}\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
    sidecar_path = os.path.join(tmpdir, f"{runid}_sidecar.json")
    with open(sidecar_path) as fh:
        return json.load(fh)


def main() -> int:
    text = open(SBATCH).read()
    n1_block = _extract_n1_block(text)  # AssertionError if markers are gone
    ok = True

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Case A: POST-FIX, artifact-invalid boot, using the REAL N1
        # block sourced straight out of g16_grid.sbatch. -------------------
        rec = _run_boot(
            tmpdir, telem_rc=0, fields_lo_rc=0, fields_hi_rc=1,
            extra_script=n1_block, label="postfix_invalid",
        )
        status = rec.get("status")
        fields_hi = rec.get("fields_rc", {}).get("hi")
        passed = status == "artifact_invalid" and status != "completed"
        print(
            f"[N1 regression] A (post-fix, artifact-invalid): status={status!r} "
            f"fields_rc.hi={fields_hi!r} -> {'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed

        # --- Case B: POST-FIX, healthy boot -- no regression on the pass
        # path. -------------------------------------------------------------
        rec = _run_boot(
            tmpdir, telem_rc=0, fields_lo_rc=0, fields_hi_rc=0,
            extra_script=n1_block, label="postfix_healthy",
        )
        status = rec.get("status")
        passed = status == "completed"
        print(
            f"[N1 regression] B (post-fix, healthy): status={status!r} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )
        ok = ok and passed

        # --- Case C: PRE-FIX reproduction. Same (unmodified) write_sidecar()
        # function, but the OLD call order -- unconditional
        # `write_sidecar ... completed` BEFORE the ARTIFACT_OK judgment ever
        # runs. This is what B-5(a)#2 reported as already-reproduced
        # synthetically; re-deriving it here (against post-fix
        # write_sidecar(), proving the defect was purely in the caller) is
        # the regression evidence, not a new finding. ----------------------
        prefix_script = "\n".join(
            [
                'write_sidecar "$RUNID" completed',
                "ARTIFACT_OK=1",
                '[ "$TELEM_RC" -eq 0 ] || ARTIFACT_OK=0',
                '[ "$FIELDS_LO_RC" -eq 0 ] || ARTIFACT_OK=0',
                '[ "$FIELDS_HI_RC" -eq 0 ] || ARTIFACT_OK=0',
            ]
        )
        rec = _run_boot(
            tmpdir, telem_rc=0, fields_lo_rc=0, fields_hi_rc=1,
            extra_script=prefix_script, label="prefix_repro",
        )
        status = rec.get("status")
        fields_hi = rec.get("fields_rc", {}).get("hi")
        bug_reproduced = status == "completed" and fields_hi == 1
        print(
            f"[N1 regression] C (pre-fix call order, reproduction): "
            f"status={status!r} fields_rc.hi={fields_hi!r} -> "
            f"{'BUG REPRODUCED (expected -- this is what the fix removed)' if bug_reproduced else 'UNEXPECTED'}"
        )
        ok = ok and bug_reproduced

    print(f"[N1 regression] OVERALL={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
