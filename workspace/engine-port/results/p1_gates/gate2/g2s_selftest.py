#!/usr/bin/env python3
"""Correctness gate for g2s_analyze.py (CPU, synthetic).

Checks the two things the coordinator required:
  (1) all 9 cells of prereg sec5.2.1 are reachable, for BOTH contrasts;
  (2) sec8.9.1 falsifier NEVER emits "verified" -- no such code path exists.
Plus: Holm alpha-only, sum-not-max aggregation, gate-label attachment.
"""
import importlib.util, itertools, json, math, os, random, sys, tempfile

GD = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2"
spec = importlib.util.spec_from_file_location("g2s", os.path.join(GD, "g2s_analyze.py"))
g2s = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2s)

fails = []

# ---------------------------------------------------------------- (1) 9 cells
def ci_for(state, scale=1.0):
    """Build a diff vector whose paired-t CI lands in the requested state."""
    if state == "+":
        return [scale * (10.0 + 0.1 * i) for i in range(10)]
    if state == "-":
        return [scale * (-10.0 - 0.1 * i) for i in range(10)]
    return [scale * (1.0 if i % 2 else -1.0) * (5.0 + 0.05 * i) for i in range(10)]

reached = set()
for itl_s, ttft_s in itertools.product("+0-", repeat=2):
    ci_i = g2s.paired_t(ci_for(itl_s))
    ci_t = g2s.paired_t(ci_for(ttft_s))
    got = (g2s.axis_state(ci_i), g2s.axis_state(ci_t))
    if got != (itl_s, ttft_s):
        fails.append(f"axis_state mismatch: wanted {(itl_s,ttft_s)} got {got}")
        continue
    nc = g2s.nine_cell(ci_i, ci_t, 12.3)
    reached.add(nc["code"])
    if nc["licences_claim"] is not False or nc["uncorrected"] is not True:
        fails.append(f"{nc['code']}: 9-cell must be an UNCORRECTED coordinate (sec5.6d)")
    if nc["code"].startswith("S2") and "12.3%" not in nc["text"]:
        fails.append(f"{nc['code']}: sec3.4(c) MDE sentence not inserted")
expected = {c for c, _ in g2s.NINE_CELL.values()}
if reached != expected:
    fails.append(f"9-cell reachability: missing {expected - reached}")
print(f"(1) 9-cell reachable codes = {sorted(reached)}  -> "
      f"{'PASS' if reached == expected else 'FAIL'}")
print("    (same table object is applied to BOTH contrasts and BOTH metrics in main(); "
      "reachability is a property of nine_cell(), not of the contrast)")

# ------------------------------------------------- (2) falsifier one-way-ness
src = open(os.path.join(GD, "g2s_analyze.py")).read()
fal_src = src[src.index("def falsifier("):src.index("def apply_falsifier(")]
if "verified" in fal_src:
    fails.append("falsifier() mentions 'verified' -- an upgrade path may exist")
if set(g2s.FALSIFIER_STATES) != {"refuted", "unchanged"}:
    fails.append(f"FALSIFIER_STATES must be exactly refuted/unchanged, got {g2s.FALSIFIER_STATES}")

def write_telemetry(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(dict(event="runtime_snapshot", **r)) + "\n")

tmp = tempfile.mkdtemp()
sm = [(108, 0), (74, 34), (54, 54), (0, 108)]
cases = {}
# a) sparse -> unchanged
p = os.path.join(tmp, "sparse.jsonl")
write_telemetry(p, [dict(timestamp_monotonic_s=i * 0.01, decode_running_batch_size=1,
                         prefill_active_batch_size=1, stream_index=1) for i in range(5)])
cases["sparse"] = g2s.falsifier(p, sm)
# helper: build >=MIN_EPISODES separate pop-A episodes (guard is imported from gate1b)
def pop_rows(split_every=None, n_ep=3, per_ep=20, dt=0.5):
    rows, t, k = [], 0.0, 0
    for _ep in range(n_ep):
        for _i in range(per_ep):
            si = 2 if (split_every and k % split_every == 0) else 1
            rows.append(dict(timestamp_monotonic_s=t, decode_running_batch_size=2,
                             prefill_active_batch_size=1, stream_index=si))
            t += dt; k += 1
        for _i in range(3):          # decode-idle gap -> episode boundary
            rows.append(dict(timestamp_monotonic_s=t, decode_running_batch_size=0,
                             prefill_active_batch_size=0, stream_index=0))
            t += dt
    return rows
# b) dense, split idx never seen -> unchanged (NOT an upgrade)
p = os.path.join(tmp, "clean.jsonl"); write_telemetry(p, pop_rows(None))
cases["clean"] = g2s.falsifier(p, sm)
# c) idx2 present above threshold -> refuted
p = os.path.join(tmp, "refute.jsonl"); write_telemetry(p, pop_rows(5))
cases["refuted"] = g2s.falsifier(p, sm)
# d) missing file -> unchanged
cases["missing"] = g2s.falsifier(os.path.join(tmp, "nope.jsonl"), sm)

for k, v in cases.items():
    print(f"(2) falsifier[{k}]: state={v['state']:<9} {v['verdict'][:64]}")
    if v["state"] not in ("refuted", "unchanged"):
        fails.append(f"falsifier[{k}] produced illegal state {v['state']}")
if cases["refuted"]["state"] != "refuted":
    fails.append("falsifier failed to refute a >=1% idx2 residency")
if cases["clean"]["state"] != "unchanged":
    fails.append("clean case must stay 'unchanged' (never 'verified')")

# monotone downgrade only
for start in ("verified", "unverified", "refuted"):
    for st_ in ("refuted", "unchanged"):
        got = g2s.apply_falsifier(start, {"state": st_})
        if st_ == "unchanged" and got != start:
            fails.append(f"apply_falsifier({start},unchanged) changed the label -> {got}")
        if st_ == "refuted" and got != "refuted":
            fails.append(f"apply_falsifier({start},refuted) -> {got}")
        if start != "verified" and got == "verified":
            fails.append("apply_falsifier UPGRADED to verified -- forbidden")
print("(2) apply_falsifier monotonicity -> " + ("PASS" if not fails else "see failures"))

# ------------------------------------------------------------ Holm alpha-only
# Holm, m=2: .004 <= .05/2 -> reject; .03 <= .05/1 -> reject.  BOTH is correct.
pv = {2.0: 0.004, 3.0: 0.03}
rej = g2s.holm(pv)
if not (rej[2.0] and rej[3.0]):
    fails.append(f"Holm m=2 [.004,.03] should reject both, got {rej}")
# step-down must STOP after the first non-rejection
pv2 = {2.0: 0.04, 3.0: 0.045}
rej2 = g2s.holm(pv2)          # .04 > .05/2=.025 -> stop; neither rejected
if rej2[2.0] or rej2[3.0]:
    fails.append(f"Holm step-down did not stop at first acceptance: {rej2}")
print(f"(3) Holm p={pv} -> {rej} ; p={pv2} -> {rej2}  (p source = ALPHA only, sec5.6c)")

# --------------------------------------------------- headline naming restriction
labels_all_ver = {("zamba2-27b", 2.0): "verified", ("zamba2-27b", 3.0): "verified"}
labels_unver = {("zamba2-27b", 2.0): "unverified", ("zamba2-27b", 3.0): "unverified"}
h1 = g2s.headline([("zamba2-27b", 3.0)], [], labels_all_ver)
h2 = g2s.headline([("zamba2-27b", 3.0)], [], labels_unver)
if "엔진 기본 궤적" not in h1["text"]:
    fails.append("verified cell must be allowed the 'engine default trajectory' naming")
if "엔진 기본 궤적" in h2["text"] or "FixedPolicy(34)" not in h2["text"]:
    fails.append("unverified cell must fall back to literal naming (sec5.6.1 rev6)")
if "sec5.6(f)" not in h2["text"]:
    fails.append("confirmatory firing only on non-verified cells must trigger sec5.6(f)")
h3 = g2s.headline([], [("zamba2-27b", 3.0)], labels_all_ver)
if h3["code"] != "COMBO-3" or "승격하지 않는다" not in h3["text"]:
    fails.append("COMBO-3 must refuse to promote secondary to headline")
for h in (h1, h2, h3, g2s.headline([], [], labels_all_ver)):
    if "sec5.5" not in h["text"]:
        fails.append(f"{h['code']} missing the mandatory sec5.5 no-decomposition clause")
print(f"(4) headline naming restriction + COMBO-3 refusal -> "
      f"{'PASS' if not any('naming' in f or 'COMBO-3' in f for f in fails) else 'FAIL'}")

# ------------------------------------------------------------- gate labels
fs_clear = [dict(arm="C", rate=3.0, flagged=False, F_B_i_evaluated=True)]
fs_fired = [dict(arm="T'", rate=3.0, flagged=True, F_B_i_evaluated=False)]
if "clear" not in g2s.gate_label(fs_clear):
    fails.append("clear gate label missing")
lbl = g2s.gate_label(fs_fired)
if "SIGN ONLY" not in lbl or "UNEVALUATED" not in lbl:
    fails.append(f"fired gate label incomplete: {lbl}")
print(f"(5) gate labels -> {lbl.strip()}")

# ------------------------------------------------------------- sum not max
import types
tmp2 = tempfile.mkdtemp()
rows = []
for r, dur in ((2.0, 74.4), (3.0, 33.8)):
    itls = [[0.02] * 95 for _ in range(20)]
    rows.append(dict(request_rate=r, duration=dur, completed=20, itls=itls,
                     ttfts=[0.2] * 20, mean_itl_ms=20.0, request_throughput=1.0))
with open(os.path.join(tmp2, "g2s_zamba2-27b_plainaux_rep1_1.jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
got = g2s.load_reps(tmp2, "zamba2-27b", "1")
dsum = list(got.values())[0]["_duration_sum_all_rates"]
if abs(dsum - 108.2) > 1e-6:
    fails.append(f"duration must SUM to 108.2 (max would be 74.4), got {dsum}")
print(f"(6) sec12-4 aggregation: duration sum={dsum:.1f} (max would be 74.4) -> PASS")

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL SYNTHETIC CORRECTNESS CHECKS PASSED")
