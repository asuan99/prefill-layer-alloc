#!/usr/bin/env python3
"""G16 pre-registered arm-order generator (PREREG_G16_RULES_REV3 sec 5, H4).

Rule (rev3 sec5): "arm 순서 = 정방향/역순 쌍" -- block 1 = SIGMA, block 2 =
reverse(SIGMA), block 3 = SIGMA_PRIME, block 4 = reverse(SIGMA_PRIME). For a
7-element sequence, reverse-position(i) = 6 - forward-position(i) (0-indexed),
so every arm's block-{1,2} position average is exactly 3.0 and every arm's
block-{3,4} position average is exactly 3.0 -- hence the 4-block average is
exactly 3.0 for EVERY arm, for ANY choice of SIGMA / SIGMA_PRIME. That
invariant is what "verify" below checks; it is not sensitive to which two
permutations were picked.

SIGMA and SIGMA_PRIME are frozen CONSTANTS, not computed at submit/run time
(H4: "job 인자로 고정, 스크립트가 내부에서 결정하지 마라(사후 지정 금지)"). The only
runtime input is BLOCK_IDX (1-4), supplied as the sbatch positional argument;
this module derives that block's arm order deterministically from it and
nothing else (no `shuf`, no unseeded `random`, no read of prior results).

addendum A-4 (2026-08-16 stage-2/harness audit, methodology gate #9):
`--verify`'s "every arm averages position 3.000000 PASS" output is an
ALGEBRAIC IDENTITY of the forward/reverse pairing construction (reverse-
position(i) = 6 - forward-position(i) for a 7-element sequence) -- it holds
for ANY choice of SIGMA / SIGMA_PRIME, including a badly-chosen one, so it is
NOT independent evidence that this specific design is good. It only confirms
"the code computed the pairing it claims to have computed" (a code-honesty
check, not a design-quality check). A prior draft of the harness report
mis-cited a `--verify` PASS as independent verification of the block design;
the auditor corrected this. Do not cite `VERIFY_OVERALL=PASS` as evidence of
anything beyond "these 4 orderings are genuine forward/reverse-paired
permutations of the 7 arms" -- see PREREG_G16_RULES_REV3_2026-08-16.md
addendum A-4 for the full argument, and note A-3 in the same addendum for
what `--verify` does NOT check (wall-clock / curvature drift balance).

SIGMA_PRIME provenance (frozen 2026-08-16, reproducible one-liner):
    python3 -c "import random; print(random.Random(16).sample(
        ['d16','d24','d34','d44','d54','d64','d74'], 7))"
  -> ['d34', 'd44', 'd64', 'd74', 'd24', 'd16', 'd54']
seed=16 chosen to match the campaign name (G16); the seed and the resulting
list are recorded here verbatim so no later run can silently redraw it.

Usage:
    g16_arm_order.py <block_idx 1-4>   -> prints space-separated arm order
    g16_arm_order.py --verify          -> per-arm average-position check (exit
                                           0 iff every arm averages 3.0 across
                                           the 4 blocks and each block is a
                                           genuine permutation of the 7 arms)
"""
from __future__ import annotations

import sys

ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]

SIGMA = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
SIGMA_PRIME = ["d34", "d44", "d64", "d74", "d24", "d16", "d54"]


def block_order(block_idx: int) -> list[str]:
    if block_idx == 1:
        return list(SIGMA)
    if block_idx == 2:
        return list(reversed(SIGMA))
    if block_idx == 3:
        return list(SIGMA_PRIME)
    if block_idx == 4:
        return list(reversed(SIGMA_PRIME))
    raise ValueError(f"block_idx must be 1-4, got {block_idx}")


def verify() -> bool:
    ok = True
    positions: dict[str, list[int]] = {a: [] for a in ARMS}
    for b in (1, 2, 3, 4):
        order = block_order(b)
        is_perm = sorted(order) == sorted(ARMS) and len(order) == len(ARMS)
        print(f"BLOCK_PERM_CHECK block={b} order={order} is_permutation={is_perm}")
        if not is_perm:
            ok = False
            continue
        for pos, arm in enumerate(order):
            positions[arm].append(pos)
    # reverse-pair check, independent of the average check below
    fwd_rev_pairs = ((1, 2, SIGMA), (3, 4, SIGMA_PRIME))
    for fwd_idx, rev_idx, base in fwd_rev_pairs:
        rev_actual = block_order(rev_idx)
        rev_expect = list(reversed(base))
        pair_ok = rev_actual == rev_expect
        print(
            f"REVERSE_PAIR_CHECK fwd_block={fwd_idx} rev_block={rev_idx} "
            f"result={'PASS' if pair_ok else 'FAIL'}"
        )
        ok = ok and pair_ok
    for arm in ARMS:
        pos_list = positions[arm]
        if len(pos_list) != 4:
            print(f"ARM_AVG_POSITION arm={arm} FAIL missing_blocks n={len(pos_list)}")
            ok = False
            continue
        avg = sum(pos_list) / len(pos_list)
        status = "PASS" if abs(avg - 3.0) < 1e-9 else "FAIL"
        if status == "FAIL":
            ok = False
        print(
            f"ARM_AVG_POSITION arm={arm} positions={pos_list} avg={avg:.6f} "
            f"result={status}"
        )
    print(f"VERIFY_OVERALL={'PASS' if ok else 'FAIL'}")
    print(
        "NOTE (addendum A-4): the above is an ALGEBRAIC IDENTITY of the "
        "forward/reverse pairing (reverse-position = 6 - forward-position); it "
        "holds for any SIGMA/SIGMA_PRIME choice and is not independent evidence "
        "of design quality. It also does not check wall-clock/curvature drift "
        "balance (addendum A-3)."
    )
    return ok


def _main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--verify":
        return 0 if verify() else 1
    if len(sys.argv) != 2:
        print("usage: g16_arm_order.py <block_idx 1-4>|--verify", file=sys.stderr)
        return 2
    try:
        idx = int(sys.argv[1])
        order = block_order(idx)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(" ".join(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
