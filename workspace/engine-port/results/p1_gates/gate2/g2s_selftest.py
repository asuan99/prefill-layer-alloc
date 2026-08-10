#!/usr/bin/env python3
"""Correctness gate for g2s_analyze.py.

Checks the two things the coordinator required:
  (1) all 9 cells of prereg sec5.2.1 are reachable, for BOTH contrasts;
  (2) sec8.9.1 falsifier NEVER emits "verified" -- no such code path exists.
Plus: Holm alpha-only, sum-not-max aggregation, gate-label attachment.

*** SECTION (0) IS NOT SYNTHETIC. ***  Sections (1)-(6) build their own rows, so
they can only verify the scorer against itself -- exactly the shape of
methodology lesson #14 ("a gate that copies the code it is verifying is close to
an identity; put the control on the PRODUCER").  That blind spot is what let
jobs 877107/877109 run 6.4 GPU-hours and produce rows with no `itls`: every
synthetic row here had `itls` because this file wrote it.

Section (0) therefore reads REAL `sglang.bench_serving` output off disk:
  POSITIVE control -- a run known to carry `--output-details` (job 875344)
                      must satisfy the scorer input contract end-to-end.
  NEGATIVE control -- a run known to OMIT `--output-details` (job 877107)
                      must be REJECTED by the same contract.
Both files are archived producer output; neither was written by this test.
A missing archive is a FAILURE, never a skip (a skipped gate is not a gate).
"""
import glob, importlib.util, itertools, json, math, os, random, shutil, sys, tempfile

GD = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/p1_gates/gate2"
spec = importlib.util.spec_from_file_location("g2s", os.path.join(GD, "g2s_analyze.py"))
g2s = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2s)

fails = []

# ==========================================================================
# (0) REAL harness-output contract -- positive + negative control on archived
#     producer output.  See the module docstring for why this must not be
#     synthetic.
# ==========================================================================
# POSITIVE: real rows from a bench run that DID pass --output-details.
GOOD_ARCHIVE = os.path.join(GD, "g2_zamba2-27b_plainaux_rep1_875344.jsonl")
GOOD_CAPSCAN = os.path.join(GD, "g2_zamba2-27b_capscan_plainaux_seed7_875344.jsonl")
# NEGATIVE: real rows from the Gate 2-S run that did NOT (the defect under test).
BAD_ARCHIVE = os.path.join(GD, "g2s_zamba2-27b_plainaux_rep1_877107.jsonl")

for p in (GOOD_ARCHIVE, GOOD_CAPSCAN, BAD_ARCHIVE):
    if not os.path.exists(p):
        fails.append(f"(0) archive missing: {p} -- this gate must not silently skip")

if os.path.exists(GOOD_ARCHIVE):
    good_rows = [json.loads(l) for l in open(GOOD_ARCHIVE)]
    # a) the contract holds on real producer output
    for i, row in enumerate(good_rows, 1):
        try:
            g2s.check_row_contract(row, f"{os.path.basename(GOOD_ARCHIVE)}:{i}")
        except g2s.InputContractError as e:
            fails.append(f"(0a) REAL --output-details row rejected by contract: {e}")
    # b) the whole read path the scorer uses actually runs on real rows
    try:
        for row in good_rows:
            g2s.assert_units(row)
            m = g2s.metrics(row)
            for k in ("alpha_ITLp95_mean", "beta_pooled_p99", "TTFTp95"):
                if not (isinstance(m[k], float) and math.isfinite(m[k])):
                    fails.append(f"(0b) metrics()[{k}] not finite on a real row: {m[k]}")
            g2s.unstable_frac(row, 0.40)
    except Exception as e:                                    # noqa: BLE001
        fails.append(f"(0b) scorer read path crashed on REAL rows: {type(e).__name__}: {e}")
    # c) load_reps + preflight on a directory of REAL rows (g2s naming)
    tmp0 = tempfile.mkdtemp()
    shutil.copy(GOOD_ARCHIVE, os.path.join(tmp0, "g2s_zamba2-27b_plainaux_rep1_0.jsonl"))
    if os.path.exists(GOOD_CAPSCAN):
        shutil.copy(GOOD_CAPSCAN,
                    os.path.join(tmp0, "g2s_zamba2-27b_capscan_plainaux_seed7_0.jsonl"))
    try:
        got0 = g2s.load_reps(tmp0, "zamba2-27b", "0")
        if not got0:
            fails.append("(0c) load_reps returned nothing for REAL rows")
    except Exception as e:                                    # noqa: BLE001
        fails.append(f"(0c) load_reps crashed on REAL rows: {type(e).__name__}: {e}")
    rc_good = g2s.preflight(tmp0, "zamba2-27b", "0")
    if rc_good != 0:
        fails.append(f"(0c) preflight rc={rc_good} on REAL --output-details rows, want 0")
    # d) every key the scorer source dereferences as row["..."] exists in a real row
    src_all = open(os.path.join(GD, "g2s_analyze.py")).read()
    import re as _re
    deref = sorted(set(_re.findall(r'row\["([a-z_0-9]+)"\]', src_all)))
    present = set(good_rows[0])
    absent = [k for k in deref if k not in present]
    if absent:
        fails.append(f"(0d) scorer dereferences row[...] keys absent from a REAL row: {absent}")
    print(f"(0) POSITIVE control {os.path.basename(GOOD_ARCHIVE)}: "
          f"{len(good_rows)} real rows, {len(present)} keys, "
          f"scorer derefs {deref} -> all present={not absent}")

if os.path.exists(BAD_ARCHIVE):
    bad_rows = [json.loads(l) for l in open(BAD_ARCHIVE)]
    caught = 0
    for i, row in enumerate(bad_rows, 1):
        try:
            g2s.check_row_contract(row, f"{os.path.basename(BAD_ARCHIVE)}:{i}")
        except g2s.InputContractError:
            caught += 1
    if caught != len(bad_rows):
        fails.append(f"(0e) NEGATIVE control not rejected: only {caught}/{len(bad_rows)} "
                     "rows from the no---output-details run raised InputContractError")
    tmp1 = tempfile.mkdtemp()
    shutil.copy(BAD_ARCHIVE, os.path.join(tmp1, "g2s_zamba2-27b_plainaux_rep1_0.jsonl"))
    rc_bad = g2s.preflight(tmp1, "zamba2-27b", "0")
    if rc_bad != 7:
        fails.append(f"(0f) preflight rc={rc_bad} on the REAL broken rows, want 7")
    try:
        g2s.load_reps(tmp1, "zamba2-27b", "0")
        fails.append("(0f) load_reps accepted rows with no `itls` -- contract not enforced")
    except g2s.InputContractError:
        pass
    except Exception as e:                                    # noqa: BLE001
        fails.append(f"(0f) load_reps raised {type(e).__name__} not InputContractError: {e}")
    missing_keys = sorted(set(g2s.REQUIRED_ROW_KEYS) - set(bad_rows[0]))
    print(f"(0) NEGATIVE control {os.path.basename(BAD_ARCHIVE)}: "
          f"{len(bad_rows)} real rows missing {missing_keys}; "
          f"rejected {caught}/{len(bad_rows)}; preflight rc={rc_bad} (want 7)")
    print("    -> this is the exact 877107 defect; the gate now fires on it BEFORE "
          "3.5 GPU-hours are spent")

# ==========================================================================
# (0h) REGRESSION FIXTURE: job 877593 (harness smoke).  This directory is the only
# one that holds bench rep files AND telemetry files side by side, which is exactly
# the condition that broke file selection: the glob `g2s_<tag>_*_rep*_<job>.jsonl`
# matched `..._telemetry_<arm>_rep1_...` and pulled 42k telemetry rows into the
# scored set, producing 42,496 "contract violations" from 5 perfectly good rep
# files -- and the classifier then called that MEASUREMENT_INCOMPLETE (exit 21,
# ~6.4 GPU-hr) when the correct answer was "rescore offline, 0 GPU-hr".
#
# 877593 IS A SMOKE RUN (1 rep, single rate, reduced config).  It is used here as a
# PLUMBING fixture only; no quantity from it is a measurement of anything.
# ==========================================================================
SMOKE_JOB = "877593"
smoke_reps = sorted(glob.glob(os.path.join(GD, f"g2s_zamba2-27b_*_rep*_{SMOKE_JOB}.jsonl")))
if not smoke_reps:
    fails.append(f"(0h) job {SMOKE_JOB} fixture missing from {GD} -- this regression "
                 "must not silently skip")
else:
    acc, exc = g2s.select_rep_files(GD, "zamba2-27b", SMOKE_JOB)
    arms_acc = sorted({a for _f, a, _r in acc})
    if len(acc) != 5:
        fails.append(f"(0h) expected 5 accepted bench rep files, got {len(acc)}: {arms_acc}")
    if sorted(arms_acc) != sorted(g2s.ARMS):
        fails.append(f"(0h) accepted arms {arms_acc} != the sec2 arms {sorted(g2s.ARMS)}")
    if not exc:
        fails.append("(0h) telemetry files were NOT excluded -- the 877593 selection "
                     "bug is back")
    for _b, reason in exc:
        if not reason:
            fails.append("(0h) an exclusion carries no reason (silent skip)")
    if any("telemetry" not in b for b, _ in exc):
        fails.append(f"(0h) unexpected exclusion (should be telemetry only): {exc}")
    # the loader must now see exactly the 5 bench files and nothing else
    try:
        sc = g2s.load_reps(GD, "zamba2-27b", SMOKE_JOB)
        if sorted({k[0] for k in sc}) != sorted(g2s.ARMS):
            fails.append(f"(0h) load_reps arms = {sorted({k[0] for k in sc})}")
    except Exception as e:                                    # noqa: BLE001
        fails.append(f"(0h) load_reps crashed on the 877593 fixture: {type(e).__name__}: {e}")
    rc_smoke = g2s.preflight(GD, "zamba2-27b", SMOKE_JOB)
    if rc_smoke != 0:
        fails.append(f"(0h) preflight rc={rc_smoke} on 877593, want 0 "
                     "(the measurement was intact; 7 here costs 6.4 GPU-hr)")
    print(f"(0h) 877593 fixture: {len(smoke_reps)} glob matches -> accepted {len(acc)} "
          f"({arms_acc}), excluded {len(exc)} telemetry; preflight rc={rc_smoke} (want 0)")

    # (0i) The COST-ASYMMETRY classifier.  Reproduce the 877593 failure with a file
    # that slips past the name allow-list (telemetry copied to an ARMS-shaped name)
    # and require the classifier to say SCORER_SELECTION_DEFECT (8, no GPU) rather
    # than MEASUREMENT_INCOMPLETE (7, 6.4 GPU-hr).
    tmp2 = tempfile.mkdtemp()
    for f, _a, _r in acc:
        shutil.copy(f, os.path.join(tmp2, os.path.basename(f).replace(SMOKE_JOB, "1")))
    tel = glob.glob(os.path.join(GD, f"g2s_zamba2-27b_telemetry_*_rep1_{SMOKE_JOB}.jsonl"))
    if tel:
        with open(tel[0]) as src_fh, \
             open(os.path.join(tmp2, "g2s_zamba2-27b_agnostic_rep9_1.jsonl"), "w") as dst_fh:
            for i, line in enumerate(src_fh):
                if i >= 200:
                    break
                dst_fh.write(line)
        rc_mis = g2s.preflight(tmp2, "zamba2-27b", "1")
        if rc_mis != 8:
            fails.append(f"(0i) classifier returned {rc_mis} for a non-bench file that "
                         "passed the name allow-list; want 8 (SCORER_SELECTION_DEFECT, "
                         "0 GPU-hr). Returning 7 here would cost ~6.4 GPU-hr.")
        print(f"(0i) cost-asymmetry classifier: non-bench file disguised under an "
              f"ARMS-shaped name -> rc={rc_mis} (want 8 = rescore offline, NOT 7 = "
              f"6.4 GPU-hr re-run)")
    # (0j) empty inventory must be INDETERMINATE, never the expensive branch
    rc_empty = g2s.preflight(tempfile.mkdtemp(), "zamba2-27b", "1")
    if rc_empty != 9:
        fails.append(f"(0j) empty inventory -> rc={rc_empty}, want 9 (INDETERMINATE). "
                     "An empty set must never be asserted OK nor rounded up to 7.")
    print(f"(0j) empty inventory -> rc={rc_empty} (want 9 = INDETERMINATE, human judgment)")

# (0g) STATIC check on the harness itself.  The two controls above prove the
# scorer rejects bad rows; this proves the runner cannot PRODUCE them.  Without
# it the loop is still open: a future edit could drop the flag again and nothing
# would notice until after the GPU time was spent.
SBATCH = os.environ.get("G2S_SBATCH", os.path.join(GD, "g2s_run.sbatch"))
# G2S_SBATCH exists so this gate can be mutation-tested against a COPY (prove it
# fires when --output-details is removed) without editing the real runner.
if not os.path.exists(SBATCH):
    fails.append(f"(0g) {SBATCH} missing")
else:
    sb = open(SBATCH).read()
    sb_lines = sb.splitlines()
    inv = [i for i, l in enumerate(sb_lines) if "sglang.bench_serving" in l]
    if not inv:
        fails.append("(0g) no bench_serving invocation found in g2s_run.sbatch")
    for i in inv:
        blob = "\n".join(sb_lines[i:i + 8])          # the continued command
        if "--output-details" not in blob:
            fails.append(f"(0g) bench_serving at {os.path.basename(SBATCH)}:{i+1} has no "
                         "--output-details -> itls/ttfts would be dropped (the 877107 defect)")
    # every `bench <...>` call site must pass all 7 positional args (port rate
    # nprompt seed outfile warmup tag); a short call silently makes $7 unbound.
    import shlex as _shlex
    calls = []
    for i, l in enumerate(sb_lines):
        s = l.strip()
        if not s.startswith("bench ") or s.rstrip().endswith("{") or "()" in s:
            continue                                   # skip the `bench () {` definition
        j, blob = i, s
        while blob.rstrip().endswith("\\"):
            j += 1
            blob = blob.rstrip()[:-1] + " " + sb_lines[j].strip()
        calls.append((i + 1, blob))
    if not calls:
        fails.append("(0g) no `bench` call sites found in g2s_run.sbatch")
    for ln, blob in calls:
        # `$(( a + b ))` is ONE shell word but contains spaces -- mask it first.
        masked = _re.sub(r"\$\(\(.*?\)\)", "ARITH", blob)
        nargs = len(_shlex.split(masked, posix=False)) - 1
        if nargs != 7:
            fails.append(f"(0g) bench call at {os.path.basename(SBATCH)}:{ln} passes {nargs} args, "
                         f"want 7 (port rate nprompt seed outfile warmup tag): {blob}")
    g_fails = [f for f in fails if f.startswith("(0g)")]
    print(f"(0g) harness static check on {os.path.basename(SBATCH)}: "
          f"{len(inv)} bench_serving invocation(s), {len(calls)} `bench` call site(s) "
          f"at lines {[ln for ln, _ in calls]} -> "
          f"{'PASS (all carry --output-details, all pass 7 args)' if not g_fails else f'FAIL ({len(g_fails)})'}")

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

# -------------------------------------------- (7) sec6.5 / gate #20 block manifest
# Structural, no data needed: every declared block must resolve to a computation
# line, and every block that emits magnitudes must resolve to an F-series gate line.
# This is the check that would have caught "9 of 19 blocks were never computed".
slm = g2s.section_line_map()
bad_anchor = {k: v["lines"] for k, v in slm.items() if not v["ok"]}
if bad_anchor:
    fails.append(f"(7) sec12-2 SECTION->LINE anchors unresolved/ambiguous: {bad_anchor}")
print(f"(7) sec12-2 section->line map: {len(slm)} anchors, all resolve uniquely="
      f"{not bad_anchor}")

man = g2s.block_manifest()
if len(g2s.BLOCKS) != 19:
    fails.append(f"(7) sec6.5 declares 19 blocks, BLOCKS has {len(g2s.BLOCKS)}")
not_computed = [n for n, _, _ in g2s.BLOCKS if not man[n]["computed_at_line"]]
ungated = [n for n, mag, _ in g2s.BLOCKS if mag and not man[n]["f_series_gate_line"]]
if not_computed:
    fails.append(f"(7) blocks declared but never computed: {not_computed}")
if ungated:
    fails.append(f"(7) magnitude blocks with no F-series gate (gate #17): {ungated}")
if man["premise_falsifier"]["emits_magnitude"]:
    fails.append("(7) premise_falsifier must not emit magnitudes (sec8.9.1)")
print(f"(7) sec6.5 manifest: {len(g2s.BLOCKS)} blocks, all computed={not not_computed}, "
      f"all magnitude blocks gated={not ungated}")

# ------------------------------------------------------- (8) F-A can actually fire
# Regression for the dead gate at the old :396.  Uses a REAL archived row as the
# base so the shape is the producer's, not this file's.
if os.path.exists(GOOD_ARCHIVE):
    base = json.loads(open(GOOD_ARCHIVE).readline())
    unsat = dict(base)
    sat = dict(base); sat["completed"] = base["completed"] - 7
    scored = {("T", 3.0, 1): unsat}
    fa_unsat = g2s.f_series(scored, {}, "T", 3.0)["F_A"]
    scored = {("T", 3.0, 1): sat}
    fa_sat = g2s.f_series(scored, {}, "T", 3.0)["F_A"]
    if fa_unsat is not False:
        fails.append(f"(8) F-A fired on an unsaturated real row (completed==total): {fa_unsat}")
    if fa_sat is not True:
        fails.append("(8) F-A did NOT fire on a saturated row -- the sec6.4 gate "
                     "`completed/total < 1.0` is dead again")
    if g2s._total_requests(base) != len(base["ttfts"]):
        fails.append("(8) _total_requests must be the attempted count len(ttfts)")
    print(f"(8) F-A on REAL row shape: completed={base['completed']} "
          f"total={g2s._total_requests(base)} -> unsaturated F_A={fa_unsat}; "
          f"completed-7 -> F_A={fa_sat}  (dead-gate regression)")

# ------------------------------- (9) sec4.4 component_share suppression is enforced
# Fieller must REFUSE to emit when the denominator is not separated from 0
# (rev4 sec4.4: "분모가 0 근처면 비율을 아예 싣지 않는다").
num_ok = [10.0 + 0.1 * i for i in range(10)]
den_ok = [20.0 + 0.1 * i for i in range(10)]
den_zero = [(1.0 if i % 2 else -1.0) * (5.0 + 0.05 * i) for i in range(10)]
f_ok, f_bad = g2s.fieller_ratio(num_ok, den_ok), g2s.fieller_ratio(num_ok, den_zero)
if not f_ok.get("ok"):
    fails.append(f"(9) Fieller failed on a well-separated denominator: {f_ok}")
if f_bad.get("ok"):
    fails.append("(9) Fieller emitted a ratio with the denominator straddling 0 -- "
                 "sec4.4 rev4 says do not carry the ratio at all")
if not (f_ok.get("identity") and f_bad.get("identity")):
    fails.append("(9) component_share must always carry identity:true (sec4.4)")
print(f"(9) sec4.4 Fieller: separated denom -> ok={f_ok['ok']} "
      f"[{f_ok['lo']:.3f},{f_ok['hi']:.3f}]; denom straddling 0 -> ok={f_bad['ok']} "
      f"({f_bad['reason'][:52]}...)")

# --------------------------- (10) TOST margin + UNDETERMINED vs measured negative
t_eq = g2s.tost([0.01 * i for i in range(10)], 100.0, g2s.DELTA_PROBE)
t_ne = g2s.tost([20.0 + 0.1 * i for i in range(10)], 100.0, g2s.DELTA_PROBE)
if not t_eq["equivalent"] or t_ne["equivalent"]:
    fails.append(f"(10) TOST wrong: tiny diff eq={t_eq['equivalent']}, "
                 f"large diff eq={t_ne['equivalent']}")
if abs(t_eq["delta"] - 5.0) > 1e-9:
    fails.append(f"(10) delta_probe must be 5% of the reference mean, got {t_eq['delta']}")
print(f"(10) TOST δ={t_eq['delta']:.2f} (5% of 100): tiny→equivalent={t_eq['equivalent']}, "
      f"large→equivalent={t_ne['equivalent']}")

print()
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED -- (0) real archived producer output, positive+negative "
      "control; (0h) job 877593 file-selection regression (rep + telemetry side by "
      "side); (0i) cost-asymmetry classifier -> 8 not 7; (0j) empty inventory -> 9; "
      "(0g) harness static check; (1)-(6) decision rules; (7) sec12-2 anchors + "
      "sec6.5 19-block manifest; (8) F-A dead-gate regression; (9) sec4.4 Fieller "
      "suppression; (10) sec6.2 TOST margin")
