#!/usr/bin/env python3
"""E2 realized-partition estimators (PREREG_E2_STICKY_REV3 sec 5 + ADDENDUM A1-A3, A6).

WHAT THIS COMPUTES.  Per cell/arm/seed, the occupancy of the D44 green-context
division (`stream_index == 2`) in decode execution, under FOUR registered
estimators plus one identity guard.  The four exist because no single one is
safe: a count statistic rides a non-uniform sampling grid (measured: idx2
532 ms vs idx4 208 ms between decode-busy snapshots on a_r4), and a
time-weighted statistic imposes an interval on an instantaneous state
(`CONSENSUS sec 3 item 120`, the `C-R` confound).  R1 therefore requires all
four to agree, which is free on the registered cells.

    *** EVERY SAMPLE PREDICATE AND WEIGHT BELOW IS A REGISTERED LITERAL. ***
    Changing one changes the verdict, so each is spelled out here rather than
    described.  ADDENDUM A2 exists because "decode-busy" was NOT spelled out in
    the pre-registration's estimator table: the neighbouring field
    `decode_batch_size` is >= 1 on idle spin snapshots, and selecting it makes
    the a_r4 sample 555 -> 20,082 records and E-cnt 8.65% -> 0.24%, which FLIPS
    R1's ON clause into a false `STICKY_DOES_NOT_REALIZE_A`.

PROVENANCE OF THE SELFTEST EXPECTATIONS (all EXTERNAL to this file -- gate #9:
a selftest whose expectations come from the code it checks is an identity):
  * E-cnt, E-qcond  -> VERDICT_result_lambda0_908623_2026-09-14.md sec 4-A,
                       columns 1 and 3 (five cells; a_r4_s2/b_r3_s2 rows are
                       absent there and come from the rev2 verdict sec 3 B1).
  * E-iter          -> VERDICT_e2_rules_2026-09-15.md sec 2 B3 gives ONE decimal
                       (4.5 / 8.1 / 8.8 / 8.4 / 45.6 / 92.3 / 92.1); the TWO-decimal
                       values this selftest enforces are in
                       VERDICT_e2_rules_rev2_2026-09-15.md:191-194,225 and
                       VERDICT_e2_rules_rev3_2026-09-15.md:594.  Both external.
  * E-time (seconds)-> VERDICT_e2_rules_rev3_2026-09-15.md sec 3 B1" (the
                       pre-registration's own seconds were back-computed as
                       `pct x den` and are CITATION-BANNED by E2C-15).
  * idx0/idx3/split_transition/busy n -> rev2 verdict sec 2-3, re-derived.

NOT A LABEL.  This file computes quantities only; the decision rules R1-R6 live
in `e2_label.py`.  The split is deliberate: the rev1 audit killed a design whose
estimator menu and decision rule could be varied together.
"""

import argparse
import json
import math
import os
import sys

# --------------------------------------------------------------- registered literals
D44_STREAM_INDEX = 2  # pdmux_homog5.yml sm_counts[(108,0),(92,16),(64,44),(16,92),(0,108)]
D44_SM = (64, 44)  # cross-check; a mismatch with the index is an ABORT (prereg sec 5)
PLAIN_UNPARTITIONED_INDEX = 4  # (0,108) -- the OFF arm's fallback, unreachable when sticky ON
GUARD_ROW_INDEX = 3  # (16,92) -- reachable only through the manual_divisions scan (prereg sec 4-1(a))
E_PACT_ABORT_BELOW = 95.0  # ADDENDUM A6 (was "!= 100 => ABORT", which bet the campaign on one sample)


def _is_snapshot(rec):
    """Registered record predicate (prereg sec 5)."""
    return rec.get("event") == "runtime_snapshot" and rec.get("phase") != "startup"


def _is_decode_busy(rec):
    """ADDENDUM A2 -- the literal definition of 'decode-busy'.

    `decode_batch_size` and `active_decode_sequences` are NOT used.  (The latter
    happens to select the same 11/11 cell samples in the 908623 archive, which
    is exactly why the literal is required rather than inferred.)
    """
    return (rec.get("decode_running_batch_size") or 0) > 0


def load_snapshots(path, probe_end=None):
    """Registered population.

    ADDENDUM A3: records at or before `t_probe_end` (the P13 correctness probe)
    are excluded, because the `phase` marker flips to "benchmark" on the FIRST
    request (`multiplexing_mixin.py:765-776`; `CONSENSUS sec 3 item 27` bans the
    `phase == "benchmark"` filter for this reason), so `phase != "startup"` does
    NOT keep probe traffic out.  Archives without a probe pass `probe_end=None`,
    which leaves the selftest expectations unchanged.
    """
    snaps, split_transitions, bad_sm, run_ids = [], 0, 0, set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "split_transition":
                # ADDENDUM A3 applies to Q4 as well, not only to the five estimators.
                if probe_end is None or rec.get("timestamp_monotonic_s", float("inf")) > probe_end:
                    split_transitions += 1
            if rec.get("run_id"):
                run_ids.add(rec["run_id"])
            if not _is_snapshot(rec):
                continue
            if probe_end is not None and rec["timestamp_monotonic_s"] <= probe_end:
                continue
            idx, sms = rec.get("stream_index"), (rec.get("prefill_sms"), rec.get("decode_sms"))
            if idx == D44_STREAM_INDEX and sms != D44_SM:
                bad_sm += 1
            snaps.append(rec)
    return snaps, split_transitions, bad_sm, run_ids


def compute(path, probe_end=None):
    snaps, split_transitions, bad_sm, run_ids = load_snapshots(path, probe_end)
    busy = [s for s in snaps if _is_decode_busy(s)]
    out = {
        "snapshots": len(snaps),
        "busy_n": len(busy),
        "split_transition": split_transitions,
        "sm_index_mismatch": bad_sm,
        "run_ids": sorted(run_ids),
    }
    # ADDENDUM A3 fail-closed: `t_probe_end` is a CLIENT-side `time.monotonic()`.
    # CLOCK_MONOTONIC is system-wide on Linux, so it is comparable to the server's
    # `timestamp_monotonic_s` -- but if the two ever stop sharing an origin the
    # filter would silently drop everything (or nothing).  Assert the overlap
    # instead of trusting it.
    if probe_end is not None:
        raw, _, _, _ = load_snapshots(path, None)
        if raw:
            lo, hi = raw[0]["timestamp_monotonic_s"], raw[-1]["timestamp_monotonic_s"]
            out["probe_end_in_range"] = bool(lo <= probe_end <= hi)
            out["telemetry_span"] = [lo, hi]
            if not out["probe_end_in_range"]:
                out["ABORT_PROBE_CLOCK_MISMATCH"] = (
                    f"t_probe_end={probe_end} outside telemetry span [{lo}, {hi}]")
            out["dropped_by_probe_filter"] = len(raw) - len(snaps)
    if not busy:
        out["empty"] = True
        return out

    def _frac(sample):
        hit = sum(1 for s in sample if s["stream_index"] == D44_STREAM_INDEX)
        return hit, len(sample), (100.0 * hit / len(sample) if sample else float("nan"))

    # -- E-cnt: decode-busy snapshots, uniform weight.
    hit, tot, pct = _frac(busy)
    out["e_cnt"] = {"num": hit, "den": tot, "pct": pct}

    # -- E-time: decode-busy snapshots, weighted by the gap to the NEXT non-startup
    #    snapshot (busy or not), uncapped; the file's last non-startup snapshot has
    #    no defined weight and is dropped.  (rev2 verdict F1': the busy-subsequence
    #    reading gives 31.54% where this gives 66.18% on b_r0 -- the reading had to
    #    be pinned, not described.)
    num_s = den_s = 0.0
    negative_gaps = 0
    for i in range(len(snaps) - 1):
        gap = snaps[i + 1]["timestamp_monotonic_s"] - snaps[i]["timestamp_monotonic_s"]
        if gap < 0:
            negative_gaps += 1
        if not _is_decode_busy(snaps[i]):
            continue
        den_s += gap
        if snaps[i]["stream_index"] == D44_STREAM_INDEX:
            num_s += gap
    out["e_time"] = {
        "num_s": num_s,
        "den_s": den_s,
        "pct": (100.0 * num_s / den_s) if den_s else float("nan"),
    }
    out["negative_gaps"] = negative_gaps

    # -- E-iter: adjacent non-startup snapshot pairs whose `decode_iterations` delta
    #    is > 0 AND whose LEFT snapshot is decode-busy; attributed to the LEFT
    #    snapshot's index, weighted by the delta.  The left-busy gate is what makes
    #    this reproduce the external expectation (rev2 verdict sec 2 B3) and what
    #    keeps all four estimators on one sample condition (ADDENDUM, prereg sec 5-2).
    num_i = den_i = idx0_i = 0
    for left, right in zip(snaps, snaps[1:]):
        delta = (right.get("decode_iterations") or 0) - (left.get("decode_iterations") or 0)
        if delta <= 0 or not _is_decode_busy(left):
            continue
        den_i += delta
        if left["stream_index"] == D44_STREAM_INDEX:
            num_i += delta
        elif left["stream_index"] == 0:
            idx0_i += delta
    out["e_iter"] = {
        "num": num_i,
        "den": den_i,
        "pct": (100.0 * num_i / den_i) if den_i else float("nan"),
        "idx0_weight_pct": (100.0 * idx0_i / den_i) if den_i else float("nan"),
    }

    # -- E-qcond: decode-busy AND prefill_queue_depth > 0, uniform.
    hit, tot, pct = _frac([s for s in busy if (s.get("prefill_queue_depth") or 0) > 0])
    out["e_qcond"] = {"num": hit, "den": tot, "pct": pct}

    # -- E-pact: decode-busy AND prefill_active_batch_size > 0, uniform.
    #    CONSTRUCTION IDENTITY GUARD, not a measurement: the v7 block installs idx2
    #    on every prefill span, so this is ~100% in BOTH arms (E2C-4 bans citing it
    #    as confirmation of anything).  A drop below 95% means the instrument, not
    #    the engine, changed.
    hit, tot, pct = _frac([s for s in busy if (s.get("prefill_active_batch_size") or 0) > 0])
    out["e_pact"] = {"num": hit, "den": tot, "pct": pct}

    # -- histogram + the two indices the one-knob check needs (prereg sec 6 R3).
    hist = {}
    for s in busy:
        hist[str(s["stream_index"])] = hist.get(str(s["stream_index"]), 0) + 1
    out["index_hist_busy"] = hist
    out["idx0"] = {"num": hist.get("0", 0), "den": len(busy), "pct": 100.0 * hist.get("0", 0) / len(busy)}
    out["idx3"] = {"num": hist.get(str(GUARD_ROW_INDEX), 0), "den": len(busy),
                   "pct": 100.0 * hist.get(str(GUARD_ROW_INDEX), 0) / len(busy)}
    return out


# ------------------------------------------------------------------- bench validity
def bench_validity(bench_path, num_prompts, input_len, output_len):
    """P9-P12 (prereg sec 7).  A cell that silently shortened its own denominator
    reads as a throughput GAIN, so these are gates, not diagnostics."""
    rec = json.load(open(bench_path)) if bench_path.endswith(".json") else None
    if rec is None:
        with open(bench_path) as handle:
            rec = json.loads(handle.readline())
    # `errors` is a LIST of per-request error strings in this bench_serving build
    # (mostly empty strings), NOT a count -- reading it as an int raises, and a
    # bare `or 0` would have silently passed a run full of failures.
    if "errors" not in rec:
        errs, errors_absent = [], True
    else:
        errs, errors_absent = rec["errors"], False
    n_err = sum(1 for e in errs if e) if isinstance(errs, list) else int(errs or 0)
    out_lens = rec.get("output_lens") or []
    checks = {
        "P9_errors_zero": (not errors_absent) and n_err == 0,
        "P10_completed_eq_num_prompts": int(rec.get("completed", -1)) == int(num_prompts),
        "P11_input_len_exact": int(rec.get("total_input_tokens", -1)) == int(num_prompts) * int(input_len),
        "P11b_output_len_exact": bool(out_lens) and all(int(v) == int(output_len) for v in out_lens),
        "P12_range_ratio_one": float(rec.get("random_range_ratio", -1.0)) == 1.0,
    }
    checks["all_pass"] = all(checks.values())
    checks["observed"] = {
        "n_errors": n_err,
        "completed": rec.get("completed"),
        "total_input_tokens": rec.get("total_input_tokens"),
        "random_range_ratio": rec.get("random_range_ratio"),
        "duration": rec.get("duration"),
    }
    if rec.get("duration"):
        checks["achieved_req_s"] = float(rec["completed"]) / float(rec["duration"])
    return checks


# ------------------------------------------------------------------------- selftest
# ONE ROW PER CELL of the 908623 archive.  Byte-exact reproduction of this table is
# the submission gate (prereg sec 5-3).  Every number's provenance is an EXTERNAL
# document -- see the module docstring.
EXPECTED = {
    #            busy   e_cnt        e_time (s)          e_iter  e_qcond      idx0     idx3  split
    "a_r0":     (2677, (120, 2677), (31.76, 667.98),      4.50, (6, 34),     (1, 2677), 0,   604),
    "a_r2":     (610,  (49, 610),   (20.61, 195.24),      8.11, (18, 102),   (0, 610),  0,   301),
    "a_r4":     (555,  (48, 555),   (25.05, 183.61),      8.76, (43, 250),   (2, 555),  0,   204),
    "a_r4_s2":  (557,  (46, 557),   (24.57, 183.44),      8.36, (43, 253),   (0, 557),  0,   169),
    "b_r0":     (313,  (135, 313),  (71.26, 107.68),     45.58, (36, 45),    (13, 313), 0,   43),
    "b_r3":     (441,  (400, 441),  (285.08, 292.00),    92.29, (395, 395),  (1, 441),  0,   100),
    "b_r3_s2":  (441,  (400, 441),  (285.99, 293.23),    92.13, (392, 392),  (0, 441),  0,   100),
}
ARCHIVE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "lambda0_prereg", "lam0_908623"
)


def selftest(archive=None):
    """Exit 1 = a registered value did not reproduce.  Exit 2 = the archive is not
    where we looked.  They used to be the same output, which is how a mutation run
    can read as 'the convention is wrong' when it is really 'the copy moved'."""
    archive = archive or ARCHIVE
    failures = []
    missing = [c for c in EXPECTED if not os.path.exists(os.path.join(archive, f"tel_{c}.jsonl"))]
    if missing:
        print(f"SELFTEST_ARCHIVE_MISSING {os.path.normpath(archive)} lacks {missing}")
        print("  (this is NOT an estimator failure; pass --archive to point at lam0_908623)")
        return 2
    print(f"# selftest against {os.path.normpath(archive)} (read-only, GPU 0)")
    print(f"{'cell':10} {'busy':>5} {'E-cnt':>13} {'E-time s':>18} {'E-iter':>7} {'E-qcond':>12} "
          f"{'idx0':>10} {'idx3':>4} {'split':>5}")
    for cell, exp in EXPECTED.items():
        path = os.path.join(archive, f"tel_{cell}.jsonl")
        got = compute(path)  # probe_end=None: the archive has no P13 probe
        e_busy, e_cnt, e_time, e_iter, e_q, e_i0, e_i3, e_st = exp
        row = [
            ("busy_n", got["busy_n"], e_busy),
            ("e_cnt.num", got["e_cnt"]["num"], e_cnt[0]),
            ("e_cnt.den", got["e_cnt"]["den"], e_cnt[1]),
            ("e_time.num_s", round(got["e_time"]["num_s"], 2), e_time[0]),
            ("e_time.den_s", round(got["e_time"]["den_s"], 2), e_time[1]),
            ("e_iter.pct", round(got["e_iter"]["pct"], 2), e_iter),
            ("e_qcond.num", got["e_qcond"]["num"], e_q[0]),
            ("e_qcond.den", got["e_qcond"]["den"], e_q[1]),
            ("idx0.num", got["idx0"]["num"], e_i0[0]),
            ("idx3.num", got["idx3"]["num"], e_i3),
            ("split_transition", got["split_transition"], e_st),
        ]
        for name, actual, want in row:
            if actual != want:
                failures.append(f"{cell}.{name}: got {actual!r}, registered {want!r}")
        print(f"{cell:10} {got['busy_n']:5d} "
              f"{got['e_cnt']['num']:6d}/{got['e_cnt']['den']:<6d} "
              f"{round(got['e_time']['num_s'],2):8.2f}/{round(got['e_time']['den_s'],2):<9.2f} "
              f"{got['e_iter']['pct']:7.2f} "
              f"{got['e_qcond']['num']:5d}/{got['e_qcond']['den']:<6d} "
              f"{got['idx0']['num']:4d}/{got['idx0']['den']:<5d} "
              f"{got['idx3']['num']:4d} {got['split_transition']:5d}")
        # E-pact is an identity in BOTH arms; assert the archive shows it so a future
        # telemetry change cannot silently void the guard (ADDENDUM A6 threshold).
        if got["e_pact"]["den"] and got["e_pact"]["pct"] < E_PACT_ABORT_BELOW:
            failures.append(f"{cell}.e_pact: {got['e_pact']['pct']:.2f}% < {E_PACT_ABORT_BELOW}")
        if got["negative_gaps"]:
            failures.append(f"{cell}: {got['negative_gaps']} negative inter-snapshot gaps")

    # The estimators must RESOLVE the arm difference they are meant to judge; a
    # table that reproduces but cannot separate anything is an identity (lesson 232).
    spread = EXPECTED["a_r4"][1][0] / EXPECTED["a_r4"][1][1] * 100
    if not (spread < 20.0 < EXPECTED["b_r3"][1][0] / EXPECTED["b_r3"][1][1] * 100):
        failures.append("resolution check: registered cells do not straddle the R1 bands")

    if failures:
        print("\nSELFTEST_FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nSELFTEST_OK 7/7 cells reproduce the registered table")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("telemetry", nargs="?", help="tel_<cell>_<arm>_<seed>.jsonl")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--archive", help="lam0_908623 directory for --selftest")
    ap.add_argument("--bench-rc", type=int, default=None,
                    help="exit status of bench_serving; 124 = P7 CAPPED (timeout)")
    ap.add_argument("--cell"), ap.add_argument("--arm"), ap.add_argument("--seed")
    ap.add_argument("--probe-end", type=float, default=None,
                    help="t_probe_end on the monotonic clock (ADDENDUM A3); omit if no probe ran")
    ap.add_argument("--bench", help="bench_serving --output-file record, for P9-P12")
    ap.add_argument("--num-prompts", type=int), ap.add_argument("--input-len", type=int)
    ap.add_argument("--output-len", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.archive)
    if not args.telemetry:
        ap.error("telemetry path required unless --selftest")

    res = compute(args.telemetry, args.probe_end)
    res["cell"], res["arm"], res["seed"] = args.cell, args.arm, args.seed
    res["probe_end"] = args.probe_end
    if args.bench and os.path.exists(args.bench):
        res["bench"] = bench_validity(args.bench, args.num_prompts, args.input_len, args.output_len)
    else:
        # H-F1: say WHICH failure this was.  Without the rc, a P7 CAPPED cell and a
        # crashed server are indistinguishable in the artefact.
        res["bench"] = {"all_pass": False,
                        "why": f"bench artefact absent (bench_rc={args.bench_rc}; 124 = P7 CAPPED)",
                        "observed": {"bench_rc": args.bench_rc}}
    # P8: compare the arm against telemetry CONTENT (`run_id`, written by the engine),
    # not against the filename -- both the filename and --arm come from the same shell
    # variable, so that comparison could never fail (an identity, lesson 9).
    if args.arm and res.get("run_ids") is not None:
        expected_fragment = f"_{args.arm}_"
        bad = [r for r in res["run_ids"] if expected_fragment not in r]
        if bad or not res["run_ids"]:
            res["P8_arm_label_mismatch"] = True
            res["ABORT_ARM_LABEL_MISMATCH"] = (
                f"P8: telemetry run_ids {res['run_ids']} do not carry arm {args.arm}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("cell", "arm", "seed", "busy_n", "e_cnt", "e_time",
                                          "e_iter", "e_qcond", "e_pact", "idx0", "idx3",
                                          "split_transition") if k in res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
