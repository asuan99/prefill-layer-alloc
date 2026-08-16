#!/usr/bin/env python3
"""G16 harness assertions (PREREG_G16_RULES_REV3 + addendum A H2'/H3'/H6') --
pure functions + thin CLI wrappers, kept separate from GPU execution so they
are testable with synthetic input alone (no GPU, no server, no real
telemetry file).

2026-08-16 stage-2 (harness) audit NO-GO fix. This file replaces the
pre-addendum H2/H3/H6 assertions with H2'/H3'/H6':

  * H2 -> H2': "telemetry file non-empty" (equivalent to "did you export the
    env var") is replaced with "telemetry has >= N events with
    event=='runtime_snapshot' and phase=='benchmark' and 'decode_sms' in
    event" -- the addendum's literal ask, because the old check passed on a
    telemetry file containing e.g. only `controller_decision` events.
  * H3 -> H3': this module no longer parses `realized_pin_check.py` output at
    all (that tool's frac-based PASS/FAIL is addendum A-1's central defect --
    it cannot answer the driver-rounding question `pdmux_context.py:113-134`
    raises, and gating on it at ANY threshold selectively excludes arms along
    the decision variable's own axis, since the fraction is monotone in
    decode SM). The replacement gate is `assert_realized_probe`, which reads
    the JSON artifact of ONE `smsplit_realized_probe.probe_one(108-D, D, dev)`
    call and gates on `exact` alone (K10). The old tool's dwell diagnostic is
    separately replaced by `pdmux_eval.analyze.controller_summary`
    (time-weighted, addendum A-1 H3'-b) -- that call lives in g16_grid.sbatch
    directly (it needs no bespoke assertion logic; it is record-only, K10: it
    never gates anything).
  * H6 -> H6': field-presence-only is replaced with
    `len(ttfts) > 0 and len(itls) == len(ttfts) and duration > 0` and a
    not-null check on all three -- catches "every request in the round
    failed" (`{"ttfts": [], "itls": [], "duration": 0.0}` used to PASS).

Run `python3 g16_assert.py selftest` to exercise every assertion path against
synthetic fixtures (GPU 0), INCLUDING the four cases the 2026-08-16 harness
audit named as currently-mis-PASSing:
  1. {"ttfts": [], "itls": [], "duration": 0.0}
  2. ttfts with 200 entries, itls with 3 entries (length mismatch)
  3. {"ttfts": null, "itls": null, "duration": null}
  4. telemetry with 0 runtime_snapshot(phase=benchmark, has decode_sms) events
All four must FAIL post-fix; the selftest asserts that explicitly instead of
just printing PASS/FAIL and trusting a human to notice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# H2': telemetry must contain >= min_count events that are
# event=="runtime_snapshot" AND phase=="benchmark" AND "decode_sms" in event.
# This is the exact predicate `pdmux_eval.analyze.controller_summary` filters
# on (analyze.py:229-240), so "the harness's own diagnostic tool would see
# something" is literally what this checks -- not merely "the file has bytes
# in it", which is what R1/R2/the pre-addendum H2 all silently accepted (gate
# #16's stated 3rd-time-is-the-charm target).
# ---------------------------------------------------------------------------
def assert_runtime_snapshots(path: str, min_count: int) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, f"telemetry file missing: {path}"
    if p.stat().st_size == 0:
        return False, f"telemetry file empty (0 bytes): {path}"
    n_lines = 0
    n_snapshots = 0
    with p.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n_lines += 1
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(e, dict)
                and e.get("event") == "runtime_snapshot"
                and e.get("phase") == "benchmark"
                and "decode_sms" in e
            ):
                n_snapshots += 1
    if n_snapshots < min_count:
        return False, (
            f"runtime_snapshot(phase=benchmark, has decode_sms) count={n_snapshots} "
            f"< min={min_count} (telemetry lines={n_lines}): {path}"
        )
    return True, (
        f"lines={n_lines} runtime_snapshot_benchmark_count={n_snapshots} "
        f"(min={min_count})"
    )


# ---------------------------------------------------------------------------
# H3'-a: driver-realized coordinate check, parsed from the JSON artifact one
# `smsplit_realized_probe.probe_one(108-D, D, dev)` call produces (invoked
# directly in g16_grid.sbatch; this module only parses its output, mirroring
# how the pre-addendum version only parsed realized_pin_check.py's stdout).
#
# The ONLY gating criterion is `exact` (K10, addendum A-1): co-residency /
# pin-fraction diagnostics, at ANY threshold, must NOT disable an arm --
# `multiplexing_mixin.py:504-511` documents that quantity as monotone in
# decode SM, so any fixed threshold on it selectively excludes arms along the
# decision variable's own axis (a 1st-order confound on the argmin
# estimator). `exact=False` means the arm's own coordinate label is wrong
# (the driver rounded (108-D, D) to something else) -- a structural defect,
# not a load-dependent one, so it is the one thing this function may fail on.
# ---------------------------------------------------------------------------
def assert_realized_probe(path: str) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, f"realized-probe file missing (fail-safe -> arm excluded): {path}"
    try:
        text = p.read_text()
    except OSError as exc:
        return False, f"realized-probe file unreadable (fail-safe): {exc} ({path})"
    try:
        rec = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"realized-probe file not valid JSON (fail-safe): {exc} ({path})"
    if not isinstance(rec, dict):
        return False, f"realized-probe file is not a JSON object (fail-safe): {path}"
    if "error" in rec:
        return False, f"probe_one raised an exception: {rec['error']} ({path})"
    if "exact" not in rec:
        return False, (
            "probe_one result has no 'exact' key (build may not expose realized "
            f"SM counts -- see rec.get('note')): {rec.get('note', rec)} ({path})"
        )
    if rec["exact"] is not True:
        return False, (
            f"DRIVER_ROUNDS_REQUEST: requested=({rec.get('requested_a')},"
            f"{rec.get('requested_b')}) realized=({rec.get('realized_a')},"
            f"{rec.get('realized_b')}) ({path})"
        )
    return True, (
        f"exact requested=({rec.get('requested_a')},{rec.get('requested_b')}) "
        f"== realized=({rec.get('realized_a')},{rec.get('realized_b')})"
    )


# ---------------------------------------------------------------------------
# H6': artifact-field assertions on a per-boot LO/HI jsonl file. Schema is
# the sgptv one (sharegpt_vary_bench.sbatch): one JSON object per
# bench_serving round, with keys ttfts / itls / duration. Round count must
# equal ROUNDS. Beyond key-presence (the pre-addendum H6), this now also
# requires: ttfts non-empty, itls length == ttfts length, duration > 0, and
# none of the three null -- the addendum's literal ask ("현재 전 요청 실패
# 라운드가 PASS한다").
# ---------------------------------------------------------------------------
def assert_fields(path: str, expect_rounds: int) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, f"file missing: {path}"
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != expect_rounds:
        return (
            False,
            f"round count mismatch: got {len(lines)} want {expect_rounds} ({path})",
        )
    for i, ln in enumerate(lines):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as exc:
            return False, f"round {i} not valid JSON: {exc} ({path})"
        if not isinstance(obj, dict):
            return False, f"round {i} is not a JSON object ({path})"
        missing = [k for k in ("ttfts", "itls", "duration") if k not in obj]
        if missing:
            return False, f"round {i} missing key(s) {missing} ({path})"
        ttfts = obj.get("ttfts")
        itls = obj.get("itls")
        duration = obj.get("duration")
        if ttfts is None or itls is None or duration is None:
            return False, (
                f"round {i} has null field(s) among ttfts/itls/duration "
                f"(ttfts={ttfts!r} itls={itls!r} duration={duration!r}) ({path})"
            )
        if not isinstance(ttfts, list) or len(ttfts) == 0:
            got = len(ttfts) if isinstance(ttfts, list) else type(ttfts).__name__
            return False, f"round {i} has empty/invalid ttfts (len/type={got}) ({path})"
        if not isinstance(itls, list) or len(itls) != len(ttfts):
            got = len(itls) if isinstance(itls, list) else type(itls).__name__
            return False, (
                f"round {i} itls length/type {got} != ttfts length {len(ttfts)} ({path})"
            )
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            return False, f"round {i} has non-positive/invalid duration={duration!r} ({path})"
    return True, (
        f"rounds={len(lines)} fields=ttfts,itls,duration all present, non-empty, "
        "length-matched, duration>0"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    rs = sub.add_parser("runtime_snapshots", help="H2' telemetry assert")
    rs.add_argument("path")
    rs.add_argument("min_count", type=int)

    f = sub.add_parser("fields", help="H6' bench-serving artifact assert")
    f.add_argument("path")
    f.add_argument("rounds", type=int)

    rp = sub.add_parser("realized_probe", help="H3'-a driver-realized coordinate assert")
    rp.add_argument("path", help="JSON artifact from smsplit_realized_probe.probe_one")

    sub.add_parser("selftest")

    args = ap.parse_args()

    if args.cmd == "runtime_snapshots":
        ok, detail = assert_runtime_snapshots(args.path, args.min_count)
        print(f"ASSERT_RUNTIME_SNAPSHOTS result={'PASS' if ok else 'FAIL'} detail={detail}")
        return 0 if ok else 1

    if args.cmd == "fields":
        ok, detail = assert_fields(args.path, args.rounds)
        print(f"ASSERT_FIELDS result={'PASS' if ok else 'FAIL'} detail={detail}")
        return 0 if ok else 1

    if args.cmd == "realized_probe":
        ok, detail = assert_realized_probe(args.path)
        print(f"ASSERT_REALIZED_PROBE result={'PASS' if ok else 'FAIL'} detail={detail}")
        return 0 if ok else 1

    if args.cmd == "selftest":
        return 0 if _selftest() else 1

    return 2


def _selftest() -> bool:
    import os
    import tempfile

    ok_all = True
    must_fail_after_fix: list[str] = []

    def check(label: str, got, want, *, is_must_fail_case: bool = False) -> None:
        nonlocal ok_all
        result = "PASS" if got == want else "FAIL"
        if result == "FAIL":
            ok_all = False
        print(f"[selftest] {label}: got={got!r} want={want!r} -> {result}")
        if is_must_fail_case:
            must_fail_after_fix.append(f"{label} -> {result}")

    # === H2': assert_runtime_snapshots ======================================
    ok, detail = assert_runtime_snapshots("/nonexistent/path/g16_does_not_exist.jsonl", 1)
    check("runtime_snapshots missing file -> ok", ok, False)

    fd, empty_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        ok, detail = assert_runtime_snapshots(empty_path, 1)
        check("runtime_snapshots empty file -> ok", ok, False)
    finally:
        os.unlink(empty_path)

    # SYNTHETIC CASE 4 (task-mandated): telemetry with events, but ZERO
    # runtime_snapshot(phase=benchmark, has decode_sms) among them -- the
    # exact shape that "file is non-empty" (pre-addendum H2) would PASS.
    fd, no_snapshot_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"event": "controller_decision", "phase": "benchmark"}) + "\n")
        fh.write(json.dumps({"event": "runtime_snapshot", "phase": "warmup", "decode_sms": 44}) + "\n")
        fh.write(json.dumps({"event": "runtime_snapshot", "phase": "benchmark"}) + "\n")  # no decode_sms
    try:
        ok, detail = assert_runtime_snapshots(no_snapshot_path, 1)
        check(
            "SYNTHETIC#4 runtime_snapshot(phase=benchmark,decode_sms) count=0 -> ok",
            ok,
            False,
            is_must_fail_case=True,
        )
    finally:
        os.unlink(no_snapshot_path)

    fd, enough_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for i in range(5):
            fh.write(
                json.dumps(
                    {"event": "runtime_snapshot", "phase": "benchmark", "decode_sms": 44,
                     "timestamp_monotonic_s": float(i)}
                )
                + "\n"
            )
    try:
        ok, detail = assert_runtime_snapshots(enough_path, 5)
        check("runtime_snapshots count=5 min=5 -> ok", ok, True)
        ok, detail = assert_runtime_snapshots(enough_path, 6)
        check("runtime_snapshots count=5 min=6 -> ok", ok, False)
    finally:
        os.unlink(enough_path)

    # === H3'-a: assert_realized_probe =======================================
    ok, detail = assert_realized_probe("/nonexistent/g16_realprobe_missing.json")
    check("realized_probe missing file -> ok", ok, False)

    fd, exact_true_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(
            {"requested_a": 34, "requested_b": 74, "realized_a": 34, "realized_b": 74,
             "exact": True}, fh)
    try:
        ok, detail = assert_realized_probe(exact_true_path)
        check("realized_probe exact=true -> ok", ok, True)
    finally:
        os.unlink(exact_true_path)

    fd, exact_false_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(
            {"requested_a": 34, "requested_b": 74, "realized_a": 32, "realized_b": 76,
             "exact": False}, fh)
    try:
        ok, detail = assert_realized_probe(exact_false_path)
        check("realized_probe exact=false (driver rounds) -> ok", ok, False)
    finally:
        os.unlink(exact_false_path)

    fd, probe_error_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"requested_a": 34, "requested_b": 74, "error": "RuntimeError('boom')"}, fh)
    try:
        ok, detail = assert_realized_probe(probe_error_path)
        check("realized_probe error key present -> ok", ok, False)
    finally:
        os.unlink(probe_error_path)

    fd, no_exact_key_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(
            {"requested_a": 34, "requested_b": 74, "n_returned": 2,
             "note": "build returns <4 elements; realized counts not exposed"}, fh)
    try:
        ok, detail = assert_realized_probe(no_exact_key_path)
        check("realized_probe no 'exact' key (build doesn't expose) -> ok", ok, False)
    finally:
        os.unlink(no_exact_key_path)

    # === H6': assert_fields ==================================================
    fd, one_round_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"ttfts": [1.0], "itls": [[0.01]], "duration": 10.0}) + "\n")
    try:
        ok, detail = assert_fields(one_round_path, expect_rounds=3)
        check("fields round-count mismatch -> ok", ok, False)
    finally:
        os.unlink(one_round_path)

    fd, three_round_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for _ in range(3):
            fh.write(
                json.dumps({"ttfts": [1.0, 2.0], "itls": [[0.01], [0.02]], "duration": 10.0})
                + "\n"
            )
    try:
        ok, detail = assert_fields(three_round_path, expect_rounds=3)
        check("fields 3-round match -> ok", ok, True)
    finally:
        os.unlink(three_round_path)

    fd, missing_key_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"ttfts": [1.0], "duration": 10.0}) + "\n")  # itls missing
    try:
        ok, detail = assert_fields(missing_key_path, expect_rounds=1)
        check("fields missing key -> ok", ok, False)
    finally:
        os.unlink(missing_key_path)

    # SYNTHETIC CASE 1 (task-mandated): all-requests-failed round.
    fd, empty_ttfts_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"ttfts": [], "itls": [], "duration": 0.0}) + "\n")
    try:
        ok, detail = assert_fields(empty_ttfts_path, expect_rounds=1)
        check(
            'SYNTHETIC#1 {"ttfts":[],"itls":[],"duration":0.0} -> ok',
            ok,
            False,
            is_must_fail_case=True,
        )
    finally:
        os.unlink(empty_ttfts_path)

    # SYNTHETIC CASE 2 (task-mandated): ttfts has 200 entries, itls has 3.
    fd, mismatch_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "ttfts": [float(i) for i in range(200)],
                    "itls": [[0.01], [0.02], [0.03]],
                    "duration": 60.0,
                }
            )
            + "\n"
        )
    try:
        ok, detail = assert_fields(mismatch_path, expect_rounds=1)
        check(
            "SYNTHETIC#2 ttfts=200 itls=3 (length mismatch) -> ok",
            ok,
            False,
            is_must_fail_case=True,
        )
    finally:
        os.unlink(mismatch_path)

    # SYNTHETIC CASE 3 (task-mandated): all three fields present but null.
    fd, all_null_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"ttfts": None, "itls": None, "duration": None}) + "\n")
    try:
        ok, detail = assert_fields(all_null_path, expect_rounds=1)
        check(
            'SYNTHETIC#3 {"ttfts":null,"itls":null,"duration":null} -> ok',
            ok,
            False,
            is_must_fail_case=True,
        )
    finally:
        os.unlink(all_null_path)

    # non-positive duration alone (ttfts/itls otherwise fine) must also FAIL
    fd, zero_dur_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"ttfts": [1.0], "itls": [[0.01]], "duration": 0.0}) + "\n")
    try:
        ok, detail = assert_fields(zero_dur_path, expect_rounds=1)
        check("fields duration=0.0 -> ok", ok, False)
    finally:
        os.unlink(zero_dur_path)

    print(
        "[selftest] TASK-MANDATED SYNTHETIC CASES (must all read '-> FAIL' above):\n  "
        + "\n  ".join(must_fail_after_fix)
    )
    print(f"[selftest] OVERALL={'PASS' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(_cli())
