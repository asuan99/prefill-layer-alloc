#!/usr/bin/env python3
"""N3 dry-run check (rev3 addendum B-5(a)#4 fix, 2026-08-16).

Confirms, against the ACTUAL text sourced out of `g16_grid.sbatch` (via the
`G16_N3_BLOCK_START`/`G16_N3_BLOCK_END` sentinel markers), that the blanket
`unset "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"` loop:

  1. Removes every PDMUX_*/SGLANG_ZAMBA_* var simulated as leaked from a
     `--export=ALL` submitting shell (including ones a stale/wrong VALUE was
     pre-set to, to catch "unset only clears empty vars" mistakes) -- with
     the sole exception of the 3 the campaign re-exports immediately after.
  2. Leaves PDMUX_TELEMETRY_PATH / PDMUX_RUN_ID / PDMUX_WORKLOAD_ID set to
     their FRESH, correctly-derived values (not a stale leaked value the
     bulk-unset failed to clear before the re-export overwrote it, and not
     accidentally unset by the bulk-unset loop itself since it runs BEFORE
     the re-export in the extracted block).

--- 2026-08-16 hardening (post-smoke-884292): checks 3/4/5 --------------------
Checks 1-2 above are, by construction, blind to the defect smoke job 884292
actually hit. They only ever look at names the TEST ITSELF injected into a
synthetic preamble; the block is extracted and run in isolation, so a
HARNESS-INTERNAL variable that is defined ELSEWHERE in `g16_grid.sbatch` and
merely happens to carry a `PDMUX_`/`SGLANG_ZAMBA_` prefix is invisible to
them. That is exactly what happened: `PDMUX_EVAL_DIR` (a `sys.path`
directory, NOT a PD-mux runtime env var) was defined near the top of the
script, wiped by this loop on every arm iteration, and then dereferenced
under `set -u` inside the H3'-b heredoc -- killing that heredoc before python
started, on 3/3 arms, which is why 884292 produced `residency_fraction={}`.

So this file now also asserts the INVARIANT that makes checks 1-2 safe:

  3. (dynamic positive control) the mechanism is live and lethal -- a
     harness-internal-looking `PDMUX_*` variable set BEFORE the block is gone
     after it, and dereferencing it after the block under `set -u` aborts the
     shell with "unbound variable" (the 884292 failure reproduced).
  4. (static, assignment side) `g16_grid.sbatch` assigns NO
     PDMUX_*/SGLANG_ZAMBA_* name outside the 3 the N3 block re-exports.
  5. (static, reference side) `g16_grid.sbatch` dereferences NO
     PDMUX_*/SGLANG_ZAMBA_* name outside those same 3.

Checks 4/5 are static text scans of the WHOLE file (not just the extracted
block), which is the level the bug lived at. Known conservative limitation:
they scan every non-comment line, including the bodies of QUOTED (`<<'PYEOF'`)
heredocs where a literal `$PDMUX_X` would NOT be a bash expansion. There are
none today; if a future edit adds one, this fails loudly (a false FAIL that
is trivially diagnosed) rather than going quiet, which is the intended
direction for this class of check.

Usage:
    g16_n3_dryrun_check.py [--sbatch PATH]

`--sbatch` exists so the hardened checks can be pointed at a PRE-fix copy of
the harness as a positive control (checks 4/5 must FAIL there, on
`PDMUX_EVAL_DIR`) -- a gate that cannot be shown to fail on the defect it was
written for is not evidence (methodology gate #9).

GPU 0, no server/telemetry/torch dependency -- pure bash + text.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SBATCH = HERE / "g16_grid.sbatch"

# A representative leak sample: some vars this campaign explicitly cares
# about not leaking (the ones addendum A-6's original curated list named, ones N3
# itself found missing from that list), plus a couple of arbitrary others
# to confirm the blanket approach is not accidentally scoped to a
# hardcoded name list.
LEAKED_VARS = {
    "PDMUX_STICKY_PARTITION": "1.0",
    "PDMUX_TRUE_DUAL_WORKER": "1",
    "PDMUX_R2_POLICY": "hybrid",
    "SGLANG_ZAMBA_TIMING": "1",
    "SGLANG_ZAMBA_PREFILL_KNEE": "3",
    "PDMUX_FIXED_PREFILL_SM_FILE": "/tmp/stale_prefill_pin.json",
    "PDMUX_HOLB_PATH": "/tmp/stale_holb.jsonl",
    "PDMUX_AGN2_SM": "44",
    "PDMUX_LA_SM_MAP": '{"0":16}',
    "PDMUX_LA_FLOOR_SM": "16",
    "PDMUX_QDEPTH_TARGET": "8",  # arbitrary extra, not on any curated list
    # Stale values for the 3 vars the campaign is SUPPOSED to re-export --
    # this is the sharpest check: if the bulk-unset ran AFTER the re-export
    # (order bug) or didn't run at all, these stale values would survive.
    "PDMUX_TELEMETRY_PATH": "/tmp/STALE_LEAKED_TELEMETRY_PATH.jsonl",
    "PDMUX_RUN_ID": "STALE_LEAKED_RUN_ID",
    "PDMUX_WORKLOAD_ID": "STALE_LEAKED_WORKLOAD_ID",
}

NEEDED = {"PDMUX_TELEMETRY_PATH", "PDMUX_RUN_ID", "PDMUX_WORKLOAD_ID"}

# The collateral sentinel for check 3: shaped like the real casualty
# (`PDMUX_EVAL_DIR` -- a harness-internal path, not a runtime lever).
SENTINEL = "PDMUX_SENTINEL_HARNESS_PATH"

PREFIXES = ("PDMUX_", "SGLANG_ZAMBA_")

_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
# `${NAME}` / `${NAME:-...}` / `$NAME`. `${!PDMUX_@}` does NOT match (the char
# after `${` is `!`, not a name char), which is what we want -- that is the
# hygiene loop itself, not a dereference of a specific variable.
_DEREF_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def _extract_n3_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "G16_N3_BLOCK_START" in l)
    end = next(i for i in range(start + 1, len(lines)) if "G16_N3_BLOCK_END" in lines[i])
    return "\n".join(lines[start : end + 1])


def _n3_block_bounds(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "G16_N3_BLOCK_START" in l)
    end = next(i for i in range(start + 1, len(lines)) if "G16_N3_BLOCK_END" in lines[i])
    return start, end


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _matches_prefix(name: str) -> bool:
    return any(name.startswith(p) for p in PREFIXES)


def _bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        capture_output=True,
        text=True,
    )


def _preamble(extra: list[str] | None = None) -> str:
    lines = ["set -uo pipefail", "HERE=/tmp/g16_n3_dryrun", "RUNID=g16_n3dryrun_arm", "ARM=d44"]
    for k, v in LEAKED_VARS.items():
        lines.append(f"export {k}={v!r}")
    lines.extend(extra or [])
    return "\n".join(lines)


def check_1_2(n3_block: str) -> bool:
    """Original checks: leaked vars cleared, the 3 needed vars re-exported fresh."""
    dump = (
        'for _v in "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"; do '
        'eval "echo AFTER:${_v}=\\$${_v}"; done'
    )
    script = _preamble() + "\n\n" + n3_block + "\n\n" + dump + "\n"
    r = _bash(script)
    if r.returncode != 0:
        print("FAIL: bash exited", r.returncode)
        print("--- stdout ---\n" + r.stdout)
        print("--- stderr ---\n" + r.stderr)
        return False

    after = {}
    for line in r.stdout.splitlines():
        if line.startswith("AFTER:"):
            name, _, val = line[len("AFTER:"):].partition("=")
            after[name] = val

    ok = True
    survivors = set(after) - NEEDED
    if survivors:
        print(f"FAIL: leaked var(s) survived the bulk-unset (should be gone): {sorted(survivors)}")
        ok = False
    else:
        print(f"PASS: every leaked PDMUX_*/SGLANG_ZAMBA_* var except {sorted(NEEDED)} was cleared "
              f"({len(LEAKED_VARS) - len(NEEDED)} leaked vars unset)")

    for name in sorted(NEEDED):
        if name not in after:
            print(f"FAIL: campaign-needed var {name} is MISSING after the block (should be re-exported)")
            ok = False
            continue
        stale = LEAKED_VARS[name]
        got = after[name]
        if got == stale:
            print(f"FAIL: {name} still holds its STALE leaked value {got!r} -- "
                  f"bulk-unset did not run before the re-export, or the re-export didn't fire")
            ok = False
        else:
            print(f"PASS: {name}={got!r} (fresh value, stale leaked value {stale!r} was overwritten)")
    return ok


def check_3_collateral_mechanism(n3_block: str) -> bool:
    """POSITIVE CONTROL for checks 4/5: prove the collateral wipe is real.

    Sets a harness-internal-looking `PDMUX_*` variable BEFORE the block (as
    `g16_grid.sbatch:103` did with `PDMUX_EVAL_DIR`) and shows (a) it is gone
    afterwards and (b) dereferencing it after the block under `set -u` aborts
    the shell with "unbound variable" -- i.e. reproduces smoke 884292's
    `line 673: PDMUX_EVAL_DIR: unbound variable` exactly.

    This check PASSES when the wipe DOES kill the sentinel: it is asserting
    the hazard exists, which is what makes the static invariant (4/5)
    load-bearing rather than decorative.
    """
    ok = True
    # (a) sentinel is wiped
    pre = _preamble([f'{SENTINEL}=/some/harness/internal/dir'])
    probe = f'if [ -z "${{{SENTINEL}+set}}" ]; then echo SENTINEL_GONE; else echo "SENTINEL_SURVIVED=${SENTINEL}"; fi'
    r = _bash(pre + "\n\n" + n3_block + "\n\n" + probe + "\n")
    if "SENTINEL_GONE" in r.stdout:
        print(f"PASS: [check 3a] harness-internal-looking {SENTINEL} set before the block is "
              f"WIPED by the bulk-unset (collateral mechanism confirmed live)")
    else:
        ok = False
        print(f"FAIL: [check 3a] {SENTINEL} survived the bulk-unset -- the hygiene loop is not "
              f"prefix-wide any more; checks 4/5 would then be enforcing a non-existent hazard "
              f"(stdout={r.stdout.strip()!r})")

    # (b) dereferencing it after the block is fatal under `set -u`
    deref = f'echo "value=${SENTINEL}"\necho REACHED_END'
    r = _bash(pre + "\n\n" + n3_block + "\n\n" + deref + "\n")
    fatal = r.returncode != 0 and "unbound variable" in r.stderr and "REACHED_END" not in r.stdout
    if fatal:
        print(f"PASS: [check 3b] dereferencing {SENTINEL} after the block aborts under `set -u` "
              f"(rc={r.returncode}, 'unbound variable') -- this is the smoke-884292 failure mode")
    else:
        ok = False
        print(f"FAIL: [check 3b] expected an 'unbound variable' abort, got rc={r.returncode} "
              f"stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r}")
    return ok


def check_4_5_static_namespace(text: str, sbatch_path: Path) -> bool:
    """Static invariant: the harness owns NO name in the wiped namespace.

    4 = no assignment, 5 = no dereference, of any PDMUX_*/SGLANG_ZAMBA_* name
    outside the 3 the N3 block re-exports. Either violation is a live
    `set -u` landmine or a silently-emptied variable on every arm iteration.
    """
    lines = text.splitlines()
    n3_start, n3_end = _n3_block_bounds(text)

    bad_assign: list[tuple[int, str]] = []
    bad_deref: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if _is_comment(line):
            continue
        m = _ASSIGN_RE.match(line)
        if m and _matches_prefix(m.group(1)) and m.group(1) not in NEEDED:
            bad_assign.append((idx + 1, m.group(1)))
        for name in _DEREF_RE.findall(line):
            if _matches_prefix(name) and name not in NEEDED:
                bad_deref.append((idx + 1, name))

    ok = True
    if bad_assign:
        ok = False
        print(f"FAIL: [check 4] {sbatch_path.name} assigns PDMUX_*/SGLANG_ZAMBA_* name(s) outside "
              f"the 3 re-exported by the N3 block -- the per-arm bulk-unset erases these on every "
              f"iteration: " + ", ".join(f"line {ln}: {nm}" for ln, nm in bad_assign))
        print("       FIX: rename the variable OUT of the runtime namespace (e.g. G16_*). Do NOT "
              "add it to the re-export list -- that would weaken N3's hygiene invariant.")
    else:
        print(f"PASS: [check 4] no PDMUX_*/SGLANG_ZAMBA_* assignment outside {sorted(NEEDED)} "
              f"(N3 block occupies lines {n3_start + 1}-{n3_end + 1})")

    if bad_deref:
        ok = False
        print(f"FAIL: [check 5] {sbatch_path.name} dereferences PDMUX_*/SGLANG_ZAMBA_* name(s) the "
              f"bulk-unset removes -- unbound-variable abort under `set -u`: "
              + ", ".join(f"line {ln}: ${nm}" for ln, nm in bad_deref))
    else:
        print(f"PASS: [check 5] no PDMUX_*/SGLANG_ZAMBA_* dereference outside {sorted(NEEDED)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sbatch", default=str(DEFAULT_SBATCH),
                    help="harness to check (point at a pre-fix copy as a positive control)")
    args = ap.parse_args()

    sbatch = Path(args.sbatch).resolve()
    text = sbatch.read_text()
    n3_block = _extract_n3_block(text)
    print(f"[N3 dry-run] target={sbatch}")

    ok = check_1_2(n3_block)
    ok = check_3_collateral_mechanism(n3_block) and ok
    ok = check_4_5_static_namespace(text, sbatch) and ok

    print(f"[N3 dry-run] OVERALL={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
