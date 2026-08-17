#!/usr/bin/env python3
"""Audit (iii): adversarial tests of the C-5(1) tie guard.

T1  teeth of g16_tie_guard_test.py  -- re-run its three cases against a
    PRE-FIX reimplementation of identify_donor; a test with teeth must fail.
T2  monotonicity ("can only lower `identified`, never create one") -- search
    for a counterexample as a function of the number of blocks.
T3  boot_frac denominator: ties are excluded from the numerator but kept in the
    denominator.  Intended conservatism or bug?
NON-DECISION-QUANTITY: synthetic fixtures only.
"""
import math, random, statistics, sys
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched")
import g16_analyze as G


def identify_donor_prefix(grid, key):
    """identify_donor as it stood BEFORE commit 502c5bf (ties = evidence)."""
    blocks = sorted({r.block for r in grid})
    per_block = {}
    for block in blocks:
        subset = [r for r in grid if r.block == block]
        winner, _tied = G._argmin_arm(G._arm_means(subset, key))
        per_block[block] = winner
    counts = {}
    for _b, w in per_block.items():
        if w:
            counts[w] = counts.get(w, 0) + 1
    rank_arm = max(counts, key=lambda a: (counts[a], -G.arm_sm(a))) if counts else None
    rank_ok = bool(rank_arm) and counts.get(rank_arm, 0) >= G.K1_BLOCK_ARGMIN_MIN
    by_block_arm = {b: G._arm_means([r for r in grid if r.block == b], key) for b in blocks}
    rng = random.Random(G.K3_BOOTSTRAP_SEED)
    freq = {}
    for _ in range(G.K3_BOOTSTRAP_SAMPLES):
        picked = [by_block_arm[rng.choice(blocks)] for _ in blocks]
        arms = sorted({a for s in picked for a in s})
        means = {a: (statistics.fmean([s[a] for s in picked if a in s])
                     if any(a in s for s in picked) else math.inf) for a in arms}
        w, _t = G._argmin_arm(means)
        if w:
            freq[w] = freq.get(w, 0) + 1
    boot_arm = max(freq, key=lambda a: (freq[a], -G.arm_sm(a))) if freq else None
    boot_frac = freq.get(boot_arm, 0) / G.K3_BOOTSTRAP_SAMPLES if boot_arm else 0.0
    boot_ok = bool(boot_arm) and boot_frac >= G.K1_BOOTSTRAP_MIN_FRAC
    return {"identified": bool(rank_ok and boot_ok and rank_arm == boot_arm),
            "rank_arm": rank_arm, "rank_ok": rank_ok,
            "boot_arm": boot_arm, "boot_frac": boot_frac}


def mk(vals_by_arm, n_blocks):
    recs = []
    for a, vals in vals_by_arm.items():
        for b in range(n_blocks):
            recs.append(G.BootRecord(
                arm=a, decode_sm=G.arm_sm(a), block=f"blk{b+1}", boot=1, phase="HI",
                path="synthetic", est={"M_itl": vals[b], "M_ttft": 1000.0 + G.arm_sm(a)}))
    return recs


print("=== T1: does g16_tie_guard_test.py FAIL against the pre-fix code? ===")
ARMS = ["d16", "d24", "d34", "d44", "d54", "d64", "d74"]
tie = {a: [50.0] * 4 for a in ARMS}
clean = {a: [60.0 - 2.0 * i + 0.01 * b for b in range(4)] for i, a in enumerate(ARMS)}
part = dict(clean); part["d64"] = list(part["d74"])
for name, fixture, expect_not_identified in (("case1 perfect tie", tie, True),
                                             ("case2 clean", clean, False),
                                             ("case3 tied minimum", part, True)):
    g = mk(fixture, 4)
    post = G.identify_donor(g, "M_itl")
    pre = identify_donor_prefix(g, "M_itl")
    print(f"  {name:20} post-fix identified={post['identified']!s:5} "
          f"pre-fix identified={pre['identified']!s:5} (pre-fix arm={pre['rank_arm']}, "
          f"boot_frac={pre['boot_frac']:.3f})")
    if expect_not_identified:
        print(f"      -> the test case {'FAILS' if pre['identified'] else 'PASSES'} "
              f"against pre-fix code == guard {'HAS' if pre['identified'] else 'HAS NO'} teeth here")

print()
print("=== T2: monotonicity counterexample search (fix must never CREATE an id.) ===")
print("    construction: arms {d16,d74}; in K blocks they are EXACTLY tied at the")
print("    minimum (tie-break -> d16); in the remaining blocks d74 wins outright.")
for n_blocks in (4, 5, 6, 7, 8):
    n_tie = n_blocks // 2
    vals = {"d16": [], "d74": []}
    for b in range(n_blocks):
        if b < n_tie:
            vals["d16"].append(50.0); vals["d74"].append(50.0)      # exact tie
        else:
            vals["d16"].append(50.0); vals["d74"].append(40.0)      # d74 wins
    g = mk(vals, n_blocks)
    post = G.identify_donor(g, "M_itl")
    pre = identify_donor_prefix(g, "M_itl")
    flag = ("  <-- COUNTEREXAMPLE: fix CREATED an identification"
            if post["identified"] and not pre["identified"] else "")
    print(f"  n_blocks={n_blocks} (tied blocks={n_tie}, K1_min={G.K1_BLOCK_ARGMIN_MIN}): "
          f"pre-fix={pre['identified']!s:5} post-fix={post['identified']!s:5}"
          f" post-arm={post['arm']} post-boot={post['bootstrap_rule']['fraction']:.4f}"
          f" tied_frac={post['bootstrap_rule']['tied_fraction']:.4f}{flag}")
print("    NOTE: n_blocks=6 is exactly what K9's ADD_2_BLOCKS produces, and K1's")
print("    '>= 3 blocks' is an ABSOLUTE constant, so at 6 blocks it stops being a")
print("    majority rule (two arms can both reach 3).")

print()
print("=== T3: boot_frac denominator under partial ties ===")
vals = {"d16": [50.0, 50.0, 50.0, 50.0], "d74": [50.0, 50.0, 40.0, 40.0]}
g = mk(vals, 4)
post = G.identify_donor(g, "M_itl")
br = post["bootstrap_rule"]
print(f"  distribution={br['distribution']}  tied_fraction={br['tied_fraction']:.4f}")
tot = sum(br["distribution"].values())
print(f"  sum(distribution) = {tot:.4f}  (= 1 - tied_fraction = {1-br['tied_fraction']:.4f})")
print("  -> ties are excluded from the NUMERATOR but kept in the DENOMINATOR:")
print("     conservative (can only lower `identified`), consistent with C-5(1)'s")
print("     'ties contribute NOTHING'.  The reporting side effect is that")
print("     `distribution` no longer sums to 1; `tied_fraction` recovers it.")

# ---------------------------------------------------------------------------
# Audit (ii): K11 fallback chain, all three branches (addendum C-1 verbatim).
# ---------------------------------------------------------------------------
print()
print("=== T4: K11 fallback chain ===")
def gridof(arms, itl_by_arm, n=4):
    return [G.BootRecord(arm=a, decode_sm=G.arm_sm(a), block=f"blk{b+1}", boot=1,
                         phase="HI", path="synthetic",
                         est={"M_ttft": 100.0 + G.arm_sm(a), "M_itl": itl_by_arm[a] + 0.001*b})
            for a in arms for b in range(n)]
cases = [
    ("7-arm campaign shape (>=2 arms with SM>=44)", ["d16","d24","d34","d44","d54","d64","d74"]),
    ("only ONE arm with SM>=44 -> fallback top-half", ["d16","d24","d34","d44"][:3] + ["d44"][:0] + ["d44"]),
    ("no arm with SM>=44 and <2 in top half", ["d16","d24"]),
]
for name, arms in cases:
    arms = sorted(set(arms), key=G.arm_sm)
    itl = {a: 50.0 + 0.1*i for i, a in enumerate(arms)}
    s = G._saturation_gap(gridof(arms, itl), contenders=[arms[0]])
    print(f"  {name}: arms={arms}")
    print(f"     upper_arms={s['upper_arms']} basis={s['upper_arm_basis']} "
          f"gap_upper={s['gap_upper_ms']}")
