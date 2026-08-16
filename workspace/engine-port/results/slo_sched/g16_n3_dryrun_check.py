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

GPU 0, no server/telemetry/torch dependency -- pure bash.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SBATCH = HERE / "g16_grid.sbatch"

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


def _extract_n3_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "G16_N3_BLOCK_START" in l)
    end = next(i for i in range(start + 1, len(lines)) if "G16_N3_BLOCK_END" in lines[i])
    return "\n".join(lines[start : end + 1])


def main() -> int:
    text = SBATCH.read_text()
    n3_block = _extract_n3_block(text)

    preamble_lines = ["set -uo pipefail", "HERE=/tmp/g16_n3_dryrun", "RUNID=g16_n3dryrun_arm", "ARM=d44"]
    for k, v in LEAKED_VARS.items():
        preamble_lines.append(f"export {k}={v!r}")
    preamble = "\n".join(preamble_lines)

    # Dump every PDMUX_*/SGLANG_ZAMBA_* var (name=value) AFTER the block runs.
    dump = (
        'for _v in "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"; do '
        'eval "echo AFTER:${_v}=\\$${_v}"; done'
    )

    script = preamble + "\n\n" + n3_block + "\n\n" + dump + "\n"

    r = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", script],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: bash exited", r.returncode)
        print("--- stdout ---\n" + r.stdout)
        print("--- stderr ---\n" + r.stderr)
        return 1

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

    print(f"[N3 dry-run] OVERALL={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
