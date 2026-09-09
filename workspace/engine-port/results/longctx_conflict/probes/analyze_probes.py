#!/usr/bin/env python3
"""Analysis for PREREG_PROBES_2026-09-08.md P1-P4.

★Reuses the canonical `compute_decode_realized` from
`results/s8_frontier/e1_pin_check.py` VERBATIM (imported, not reimplemented --
PROJECT_STATUS methodology gate #9). That function only returns the
TIME-weighted estimator (`f_time` = `decode_realized_frac`), `decode_hist`
and `t_decode_active_s`. The COUNT-weighted estimator (`f_count`), the
`split ∧ prefill_active_batch_size==0` diagnostic and `dropped_events` are
NOT in that function (it predates this probe set) -- they are new,
additive computations over the same row population, not a reimplementation
of anything `compute_decode_realized` already does.

`compute_decode_realized`'s own module does `from scipy.stats import beta
as _beta` at import time; that import is only used inside
`compute_episode_gate` (line ~639), which this script never calls. If scipy
is importable (checked, not assumed) we import the module directly and touch
nothing. Only if the bare import fails do we install a placeholder
`scipy`/`scipy.stats` module in `sys.modules` so the *unused* top-level
import resolves -- `compute_decode_realized`'s own source is never edited or
copied.

"split" (per-row) := a genuine green-context division is the CURRENT stream
group, i.e. `0 < decode_sms < 108` (the plain prefill-only group is
(108,0)->decode_sms=0, the plain decode-only fallback group is
(0,108)->decode_sms=108; neither is a "split" in the PREREG's sense --
`multiplexing_mixin.py:250-266` / `dual_worker.py:608,623`).

Usage:
  python3 analyze_probes.py <telemetry.jsonl> <expect_d> [--out out.json] [--label L]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E1_DIR = HERE.parent.parent / "s8_frontier"


def _import_e1_pin_check():
    try:
        import scipy  # noqa: F401

        scipy_stubbed = False
    except ImportError:
        import types

        scipy_stub = types.ModuleType("scipy")
        stats_stub = types.ModuleType("scipy.stats")
        stats_stub.beta = None  # only compute_episode_gate touches this; unused here
        scipy_stub.stats = stats_stub
        sys.modules.setdefault("scipy", scipy_stub)
        sys.modules.setdefault("scipy.stats", stats_stub)
        scipy_stubbed = True
    sys.path.insert(0, str(E1_DIR))
    import e1_pin_check as E  # noqa: E402

    return E, scipy_stubbed


def load_rows(path):
    """All `event=="runtime_snapshot" and phase=="benchmark"` rows, sorted by
    `timestamp_monotonic_s` -- the exact population `compute_decode_realized`
    iterates (verbatim filter, copied because it must match, not because the
    function itself is reimplemented)."""
    rows = []
    n_lines = 0
    parse_errors = 0
    dropped_events_max = 0
    with open(path) as fh:
        for line in fh:
            n_lines += 1
            try:
                e = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            de = e.get("dropped_events")
            if isinstance(de, (int, float)) and de > dropped_events_max:
                dropped_events_max = de
            if e.get("event") != "runtime_snapshot" or e.get("phase") != "benchmark":
                continue
            rows.append(e)
    rows.sort(key=lambda e: e.get("timestamp_monotonic_s", 0.0))
    audit = dict(
        n_lines=n_lines,
        parse_errors=parse_errors,
        n_snapshot_benchmark_rows=len(rows),
        dropped_events=dropped_events_max,
    )
    return rows, audit


def compute_f_count(rows, expect_d):
    """Snapshot-COUNT-weighted decode-realized fraction, decode-active
    (`decode_running_batch_size>0`) population only -- the count analogue of
    `compute_decode_realized`'s time-weighted `f_time` (PREREG sec 0)."""
    n_active = 0
    n_match = 0
    for e in rows:
        drb = e.get("decode_running_batch_size", 0) or 0
        if drb <= 0:
            continue
        n_active += 1
        if e.get("decode_sms") == expect_d:
            n_match += 1
    f_count = (n_match / n_active) if n_active else float("nan")
    return dict(f_count=f_count, n_decode_active_snapshots=n_active,
                n_decode_active_at_target=n_match)


def compute_split_pab0(rows):
    """`split ∧ prefill_active_batch_size==0` -- PREREG sec 0, record-skew
    death cause R3. `split` := 0 < decode_sms < 108 (see module docstring)."""
    n_total = len(rows)
    n_split = 0
    n_split_pab0 = 0
    for e in rows:
        dsm = e.get("decode_sms")
        is_split = isinstance(dsm, (int, float)) and 0 < dsm < 108
        if not is_split:
            continue
        n_split += 1
        pab = e.get("prefill_active_batch_size", None)
        if pab == 0:
            n_split_pab0 += 1
    return dict(
        n_rows_total=n_total,
        n_split_total=n_split,
        n_split_pab0=n_split_pab0,
        frac_split_pab0_of_split=(n_split_pab0 / n_split) if n_split else float("nan"),
        frac_split_pab0_of_all=(n_split_pab0 / n_total) if n_total else float("nan"),
    )


def analyze_one(telemetry_path, expect_d, t0=None, t1=None):
    E, scipy_stubbed = _import_e1_pin_check()
    rows, audit = load_rows(telemetry_path)
    dr = E.compute_decode_realized(telemetry_path, expect_d, t0, t1)
    fc = compute_f_count(rows, expect_d)
    sk = compute_split_pab0(rows)
    out = dict(
        telemetry_path=str(telemetry_path),
        expect_d=expect_d,
        scipy_stubbed=scipy_stubbed,
        audit=audit,
        f_time=dr["decode_realized_frac"],
        t_decode_active_s=dr["t_decode_active_s"],
        decode_hist=dr["decode_hist"],
        decode_hist_mode=(
            max(dr["decode_hist"], key=dr["decode_hist"].get)
            if dr["decode_hist"] else None
        ),
        **fc,
        **sk,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry_path")
    ap.add_argument("expect_d", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    args = ap.parse_args()

    out = analyze_one(args.telemetry_path, args.expect_d, args.t0, args.t1)
    if args.label:
        out["label"] = args.label
    text = json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
