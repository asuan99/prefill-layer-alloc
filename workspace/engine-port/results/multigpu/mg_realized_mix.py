#!/usr/bin/env python3
"""E-time realized-partition estimator for M2g/M3 (PREREG_MULTIGPU_2026-10-12.md rev6 sec 4-3 P-D2).

REGISTERED LITERALS (sec 4-3 "E-time(P-D2)", quoted):
  * decode-busy        = `decode_running_batch_size` > 0                      (A2 predicate)
  * time weight        = gap to the NEXT non-startup `runtime_snapshot`       (e2 convention,
                         uncapped; the file's last non-startup snapshot has no weight)
  * numerator          = `stream_index` NOT IN {0, real_sm_group_num - 1}     (every division
                         group, not one literal index)
  * arm `sm_counts` are checked against the green read-out; a mismatch makes THAT CELL
    `MIX_FAILED` (no ABORT)
  * TP: rank 0 only.  DP: mean over replicas.
  * `PARTITION_NOT_EXERCISED` iff the median E-time of the PD-mux arm's 4 HI seeds < 0.5 (MGP-6).

`e2_realized_mix.py`'s `D44_STREAM_INDEX = 2` is deliberately NOT used: under
`pdmux_agnostic_manual.yml` decode below decode_bs 36 sits on index 1 and under
`pdmux_r2.yml` + fixed D44 on index 4.  The "idx == 2 literally" mutant is
KILLED on agnostic and r2 synthetic telemetry in tests/test_mg_realized_mix.py.

`real_sm_group_num` is NOT read from telemetry (snapshots do not carry it): it is
len(registered arm sm_counts), and the green read-out must show exactly that
many groups -- that is the cross-check that makes the definition non-circular.

Per-rank files follow M2's `multigpu_guard.rank_suffixed_path` (`.dp{r}` then
`.tp{t}` before the last suffix; unchanged at tp = dp = 1).

QUANTITIES ONLY.  Labels other than MIX_FAILED / PARTITION_NOT_EXERCISED live in
mg_label.py.  CPU only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ENG = Path(__file__).resolve().parents[2]

# sec 2 arm table.  The sm_counts are the engine layout [(total,0)] + divisions + [(0,total)]
# (pdmux_context.initialize_stream_groups); tests assert these equal the parsed config files.
ARM_CONFIGS = {
    "agnostic": "benchmarks/configs/pdmux_agnostic_manual.yml",
    "d44": "benchmarks/configs/pdmux_r2.yml",
}
REGISTERED_SM_COUNTS = {
    "agnostic": [(108, 0), (74, 34), (54, 54), (0, 108)],
    "d44": [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)],
}
PDMUX_ARMS = tuple(REGISTERED_SM_COUNTS)
E_TIME_EXERCISED_MIN = 0.5  # sec 4-3: HI 4-seed median E-time < 0.5 -> PARTITION_NOT_EXERCISED (MGP-6)
TOTAL_SM = 108


# ------------------------------------------------------------------ registered predicates
def is_snapshot(rec: Dict) -> bool:
    return rec.get("event") == "runtime_snapshot" and rec.get("phase") != "startup"


def is_decode_busy(rec: Dict) -> bool:
    return (rec.get("decode_running_batch_size") or 0) > 0


def in_split_group(stream_index: int, real_sm_group_num: int) -> bool:
    return stream_index not in (0, real_sm_group_num - 1)


def snapshot_weights(snaps: Sequence[Dict]) -> List[Tuple[int, float]]:
    """(index, weight) for every snapshot that has a successor: gap to the NEXT one."""
    return [(i, snaps[i + 1]["timestamp_monotonic_s"] - snaps[i]["timestamp_monotonic_s"])
            for i in range(len(snaps) - 1)]


# ------------------------------------------------------------------------------ config
def sm_counts_from_config(path: Path, total_sm: int = TOTAL_SM) -> List[Tuple[int, int]]:
    """Minimal YAML read of manual_divisions (no PyYAML dependency); fail-closed."""
    rows: List[Tuple[int, int]] = []
    in_md = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("manual_divisions:"):
            in_md = True
            continue
        if in_md and line.lstrip().startswith("- ["):
            vals = [int(v) for v in line.split("[", 1)[1].split("]", 1)[0].split(",")]
            rows.append((vals[0], vals[1]))
            continue
        if in_md and not line.startswith(" "):
            in_md = False
    if not rows:
        raise ValueError(f"{path}: no manual_divisions (auto grid is not a registered M3 arm)")
    return [(total_sm, 0)] + rows + [(0, total_sm)]


# ---------------------------------------------------------------------------- loaders
def _guard_module():
    path = ENG / "src" / "multiplex" / "multigpu_guard.py"
    spec = importlib.util.spec_from_file_location("multigpu_guard_mg_mix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rank_files(base: str, mode: str, n: int) -> List[str]:
    """Files the estimator reads: N=1 -> base; TP=n -> rank 0 only; DP=n -> every replica."""
    g = _guard_module()
    if mode == "n1" or n == 1:
        return [base]
    if mode == "tp":
        return [g.rank_suffixed_path(base, n, 0, 1, None)]
    if mode == "dp":
        return [g.rank_suffixed_path(base, 1, None, n, r) for r in range(n)]
    raise ValueError(f"mode {mode!r}")


def load_snapshots(path: str) -> Tuple[List[Dict], int]:
    snaps, bad_lines = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if is_snapshot(rec):
                snaps.append(rec)
    return snaps, bad_lines


# --------------------------------------------------------------------------- green
def check_green(green: Dict, sm_counts: Sequence[Tuple[int, int]]) -> List[str]:
    """Registered sm_counts vs the in-process driver read-out (green_readout.py).

    Division groups (1..n-2) must carry a green context whose smCount equals the target
    on both sides; plain groups (0, n-1) must carry none; sticky is OFF (sec 2)."""
    bad = []
    if green.get("sticky_target_index") is not None:
        bad.append(f"sticky_target_index={green.get('sticky_target_index')!r} (sticky OFF registered)")
    groups = {g.get("stream_index"): g for g in green.get("groups", [])}
    if sorted(groups) != list(range(len(sm_counts))):
        bad.append(f"green groups {sorted(groups)} != 0..{len(sm_counts) - 1}")
    for i, (p, d) in enumerate(sm_counts):
        grp = groups.get(i)
        if grp is None:
            continue
        if (grp.get("target_prefill_sm"), grp.get("target_decode_sm")) != (p, d):
            bad.append(f"group {i} engine targets "
                       f"{(grp.get('target_prefill_sm'), grp.get('target_decode_sm'))} != registered {(p, d)}")
        division = 1 <= i <= len(sm_counts) - 2
        for side, sm in (("prefill", p), ("decode", d)):
            rec = grp.get(side) or {}
            got = (rec.get("green_sm") or {}).get("smCount")
            if division and (rec.get("green_ctx_is_null", True) or got != sm):
                bad.append(f"group {i} {side}: green smCount {got} != {sm}")
            if not division and not rec.get("green_ctx_is_null", True):
                bad.append(f"group {i} {side}: plain group carries a green context")
    return bad


# ------------------------------------------------------------------------- estimator
def e_time_file(path: str, sm_counts: Sequence[Tuple[int, int]],
                window: Optional[Tuple[float, float]] = None) -> Dict:
    """E-time for one per-rank telemetry file.

    `window` = (t0, t1) on the server's monotonic clock: a cell inside a multi-cell boot.
    Snapshots with t0 <= t <= t1 are the population; each keeps its literal weight (gap to
    the next non-startup snapshot of the FILE, uncapped).  Fail-closed on a window that
    does not overlap the telemetry span (clock mismatch, CONSENSUS sec 3 item 27 family).
    """
    snaps, bad_lines = load_snapshots(path)
    real = len(sm_counts)
    out = {"file": path, "snapshots": len(snaps), "bad_json_lines": bad_lines, "problems": []}
    if not snaps:
        out["problems"].append("no non-startup runtime_snapshot")
        return out
    lo, hi = snaps[0]["timestamp_monotonic_s"], snaps[-1]["timestamp_monotonic_s"]
    if window is not None and not (window[0] <= hi and window[1] >= lo):
        out["problems"].append(f"window {window} outside telemetry span [{lo}, {hi}]")
        return out
    num = den = 0.0
    steps_num = steps_den = 0
    neg = sm_bad = 0
    hist: Dict[str, int] = {}
    for i, w in snapshot_weights(snaps):
        s = snaps[i]
        t = s["timestamp_monotonic_s"]
        if window is not None and not (window[0] <= t <= window[1]):
            continue
        idx = s.get("stream_index")
        if not isinstance(idx, int) or not 0 <= idx < real:
            sm_bad += 1
            continue
        if (s.get("prefill_sms"), s.get("decode_sms")) != tuple(sm_counts[idx]):
            sm_bad += 1
        if w < 0:
            neg += 1
        if not is_decode_busy(s):
            continue
        hist[str(idx)] = hist.get(str(idx), 0) + 1
        den += w
        split = in_split_group(idx, real)
        if split:
            num += w
        # MGP-2 exposure (G1 descriptive): decode-iteration delta to the next snapshot,
        # attributed to this (busy) snapshot's group.
        nxt = snaps[i + 1] if i + 1 < len(snaps) else None
        delta = ((nxt.get("decode_iterations") or 0) - (s.get("decode_iterations") or 0)) if nxt else 0
        if delta > 0:
            steps_den += delta
            steps_num += delta if split else 0
    if sm_bad:
        out["problems"].append(f"{sm_bad} snapshots whose (prefill_sms, decode_sms) != registered sm_counts[stream_index]")
    if neg:
        out["problems"].append(f"{neg} negative inter-snapshot gaps")
    out.update({
        "num_s": num, "den_s": den,
        "e_time": (num / den) if den > 0 else None,
        "busy_index_hist": hist,
        "exposure_steps": (steps_num / steps_den) if steps_den else None,
        "steps_den": steps_den,
        "span": [lo, hi],
    })
    if den <= 0:
        out["problems"].append("no decode-busy weight in window")
    return out


def cell_e_time(arm: str, mode: str, n: int, telemetry_base: str, green_base: str,
                window: Optional[Tuple[float, float]] = None,
                sm_counts: Optional[Sequence[Tuple[int, int]]] = None) -> Dict:
    """One cell's E-time.  status OK | MIX_FAILED (reasons listed) | NOT_APPLICABLE (fused arms)."""
    if arm not in PDMUX_ARMS:
        return {"arm": arm, "status": "NOT_APPLICABLE", "e_time": None}
    sm_counts = list(sm_counts or REGISTERED_SM_COUNTS[arm])
    tel_files = rank_files(telemetry_base, mode, n)
    green_files = rank_files(green_base, mode, n)
    reasons, per = [], []
    for tf, gf in zip(tel_files, green_files):
        rec: Dict = {"telemetry": tf, "green": gf}
        try:
            green = json.loads(Path(gf).read_text(encoding="utf-8"))
            gbad = check_green(green, sm_counts)
        except (OSError, ValueError) as exc:
            gbad = [f"green read-out unreadable: {exc!r}"]
        rec["green_problems"] = gbad
        try:
            est = e_time_file(tf, sm_counts, window)
        except OSError as exc:
            est = {"problems": [f"telemetry unreadable: {exc!r}"], "e_time": None}
        rec["estimate"] = est
        reasons += [f"{gf}: {b}" for b in gbad] + [f"{tf}: {b}" for b in est.get("problems", [])]
        per.append(rec)
    vals = [r["estimate"].get("e_time") for r in per]
    if reasons or any(v is None for v in vals):
        return {"arm": arm, "mode": mode, "n": n, "status": "MIX_FAILED", "reasons": reasons,
                "e_time": None, "etime": None, "per_rank": per}
    # TP: one file (rank 0).  DP: unweighted mean over replicas ("복제본 평균").
    e = sum(vals) / len(vals)
    # "etime" duplicates "e_time" under the key mg_label.load_cell reads from etime.json.
    return {"arm": arm, "mode": mode, "n": n, "status": "OK", "reasons": [],
            "e_time": e, "etime": e, "per_rank": per,
            "exposure_steps": _mean([r["estimate"].get("exposure_steps") for r in per])}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def partition_label(hi_seed_e_times: Sequence[Optional[float]]) -> str:
    """sec 4-3 / MGP-6 over the PD-mux arm's HI seeds (4 registered).  Fail-closed."""
    if len(hi_seed_e_times) != 4 or any(v is None or not math.isfinite(v) for v in hi_seed_e_times):
        return "E_TIME_UNDEFINED"
    return ("PARTITION_NOT_EXERCISED" if statistics.median(hi_seed_e_times) < E_TIME_EXERCISED_MIN
            else "PARTITION_EXERCISED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arm", required=True, choices=PDMUX_ARMS)
    ap.add_argument("--mode", required=True, choices=("n1", "tp", "dp"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--telemetry", required=True, help="PDMUX_TELEMETRY_PATH as given to the server (unsuffixed)")
    ap.add_argument("--green", required=True, help="PDMUX_GREEN_READOUT_PATH as given (unsuffixed)")
    ap.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = cell_e_time(a.arm, a.mode, a.n, a.telemetry, a.green, tuple(a.window) if a.window else None)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("arm", "mode", "n", "status", "e_time", "reasons")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
