#!/usr/bin/env python3
"""N2(3-a) realized-partition descriptives (PREREG_N2_3A rev16 §7-3; §15 `n2_realized_mix.py`). CPU only.
DESCRIPTIVE ONLY -- no label reads anything from this module (§7-3: "4추정량은 판정 입력이 아니다").

A2-pair estimators (E2 ADDENDUM A2 / E2C-8' literal, pair-keyed with π): population = `runtime_snapshot` with
phase != "startup"; decode-busy = decode_running_batch_size > 0; E-cnt = busy snapshots at π / busy; E-time = busy
snapshots weighted by the gap to the NEXT non-startup snapshot (uncapped, last dropped); E-iter = adjacent pairs with
Δdecode_iterations > 0 and LEFT busy, weight Δ, attributed to the LEFT snapshot; E-qcond = busy ∧
prefill_queue_depth > 0. W-idx = event-held pair of the last work event (prefill_span_start, decode_iteration at
t_launch), N3 §7-2 definition. PE-iter / PE-time come from the shared reducer (n2_probe_reduce._pe / _pe_time).
This file is an INDEPENDENT implementation (not an import of n3_realized_mix) so its positive control is not an
identity: on the λ0 VESSL archive (7 cells, homog5, (64,44) = index 2) the whole-file A2-pair numbers must equal
n3_realized_mix.snapshot_estimators bit for bit; a different π must not (negative control).

A2-pair-W (new convention, N2 windows only -- never compared with whole-file values): the same four estimators over
the snapshots whose timestamp lies in the window's [first t_launch, last t_sync]; E-time gaps are taken to the next
snapshot INSIDE that set (the last one in the window carries no weight). < 10 busy snapshots -> NA (§7-1 (e)); the
alone-window E-qcond has an empty denominator -> NA, never 0 % / 100 % (§7-3).

BLIND (rev16 H15-4, §5-4 (ii)(나), N2C-73): Job-PROFILE boots contain co-located burst iterations. Cell- and
boot-scope pair-wise estimators (E-time / E-iter by pair) would carry co-located per-iteration time (a κ pilot).
`estimators(scope=...)` therefore REFUSES scope "cell" / "boot" for a Job-PROFILE input unless the caller passes
blind_exception=True, which stamps the output with the N2C-73 caveat; n2_analyze calls only scope "window" (the
alone window W, which by §7-1 (c) contains no prefill). §7-3 (산출 시점, §18-2-d #11): the Job-PROFILE cell / boot
scope is computed only after the Job-EA result seal -- `estimators(..., ea_result_seal=<record>)` lifts the refusal
only for a VERIFIED seal record (n2_analyze.verify_ea_result_seal; `n2_analyze.py --job PROFILE --post-ea-seal`), and
stamps the seal record instead of N2C-73. Job-EA inputs (job="EA") are never blinded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Job layout (lesson 270): workspace/engine-port/results is a SYMLINK to /data/runs/<job>/results, and a module
# imported via the cwd ('' on sys.path, e.g. the preflight `cd $HERE && python3 -m unittest`) gets a __file__ under
# the PHYSICAL run dir (getcwd resolves symlinks) -> HERE/../.. leaves the clone and loses benchmarks/ (Job
# pdmux-n2-static-grid-52a3b97-20261011182211, ABORT_PREFLIGHT).  PDMUX_PROJECT_ROOT (exported by job_entry.sh,
# required by n2_capacity.sh) names the clone; without it (local runs) the file-relative root is used.
ENG = (os.path.join(os.path.abspath(os.environ["PDMUX_PROJECT_ROOT"]), "workspace", "engine-port")
       if os.environ.get("PDMUX_PROJECT_ROOT") else os.path.abspath(os.path.join(HERE, "..", "..")))
LAM0V_ARCHIVE = os.path.join(ENG, "results/r2_eval/lambda0_vessl/lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945")
N3_MIX = os.path.join(ENG, "results/n3_capacity/n3_realized_mix.py")
SELFTEST_CELLS = ("a_r0", "a_r2", "a_r4", "a_r4_s2", "b_r0", "b_r3", "b_r3_s2")
MIN_BUSY_FOR_WINDOW = 10
WORK = ("prefill_span_start", "decode_iteration")
N2C_73 = ("N2C-73: Job-EA 개정 창의 블라인드(§5-4 (ii)(나))는 분석자의 자기 기록과 산출물 검사로만 확인된다 — "
          "기록되지 않은 열람·계산은 검출할 수 없고, §7-3의 셀·boot 단위 실현 분할 4추정량(burst 단계 포함)은 등록 "
          "산출물 자체에 동거 구간 정보를 담는다.")


class BlindedScope(Exception):
    """H15-4: cell / boot scope pair-wise estimators of a Job-PROFILE input are blinded until the Job-EA result seal."""


def read_events(path):
    out = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def _pair(rec):
    return (rec.get("prefill_sms"), rec.get("decode_sms"))


def _busy(rec):
    return (rec.get("decode_running_batch_size") or 0) > 0


def a2_pair(events, pi, lo=None, hi=None):
    """The four A2-pair estimators for pair `pi` over the non-startup snapshots of `events` (optionally restricted
    to timestamps in [lo, hi]). Returns num/den and pct (None when the denominator is 0)."""
    snaps = [e for e in events if e.get("event") == "runtime_snapshot" and e.get("phase") != "startup"
             and (lo is None or lo <= e["timestamp_monotonic_s"] <= hi)]
    busy = [s for s in snaps if _busy(s)]
    out = dict(snapshots=len(snaps), busy_n=len(busy), pi=list(pi))

    def frac(sample):
        hit = sum(1 for s in sample if _pair(s) == tuple(pi))
        return dict(num=hit, den=len(sample), pct=(100.0 * hit / len(sample)) if sample else None)
    out["e_cnt"] = frac(busy)
    num_s = den_s = 0.0
    for i in range(len(snaps) - 1):
        if not _busy(snaps[i]):
            continue
        gap = snaps[i + 1]["timestamp_monotonic_s"] - snaps[i]["timestamp_monotonic_s"]
        den_s += gap
        if _pair(snaps[i]) == tuple(pi):
            num_s += gap
    out["e_time"] = dict(num_s=num_s, den_s=den_s, pct=(100.0 * num_s / den_s) if den_s else None)
    num_i = den_i = 0
    for left, right in zip(snaps, snaps[1:]):
        delta = (right.get("decode_iterations") or 0) - (left.get("decode_iterations") or 0)
        if delta <= 0 or not _busy(left):
            continue
        den_i += delta
        if _pair(left) == tuple(pi):
            num_i += delta
    out["e_iter"] = dict(num=num_i, den=den_i, pct=(100.0 * num_i / den_i) if den_i else None)
    out["e_qcond"] = frac([s for s in busy if (s.get("prefill_queue_depth") or 0) > 0])
    return out


def w_idx(events, lo, hi):
    """Event-held pair share of [lo, hi] (N3 §7-2): each work event's pair is held until the next work event."""
    ev = sorted([(e["timestamp_monotonic_s"], _pair(e)) for e in events
                 if e.get("event") == "prefill_span_start" and lo <= e["timestamp_monotonic_s"] <= hi] +
                [(e["t_launch"], _pair(e)) for e in events
                 if e.get("event") == "decode_iteration" and lo <= e["t_launch"] <= hi])
    held = {}
    for (t, p), (t2, _p) in zip(ev, ev[1:] + [(hi, None)]):
        k = f"{p[0]},{p[1]}"
        held[k] = held.get(k, 0.0) + max(0.0, min(t2, hi) - t)
    tot = sum(held.values())
    return {k: v / tot for k, v in sorted(held.items())} if tot else {}


def ea_seal_lifts_blind(ea_result_seal):
    """§7-3 산출 시점 (rev17 H15-4, §18-2-d #11): the Job-PROFILE cell / boot scope is computed only AFTER the Job-EA
    result seal. `ea_result_seal` = the record returned by n2_analyze.verify_ea_result_seal (the seal file's recorded
    sha256 re-checked against the sealed Job-EA result bytes); only a verified record lifts the blind."""
    return isinstance(ea_result_seal, dict) and ea_result_seal.get("verified") is True and \
        bool(ea_result_seal.get("ea_result_sha256"))


def estimators(events, pi, scope, lo=None, hi=None, job="PROFILE", blind_exception=False, ea_result_seal=None):
    """scope 'window' (A2-pair-W + W-idx over [lo, hi]); scope 'cell' / 'boot' = whole input (A2-pair + W-idx over
    the work-event span) -- refused for Job-PROFILE (H15-4) unless blind_exception=True (then N2C-73 is stamped) or a
    VERIFIED Job-EA result seal is passed (then the seal record is stamped instead -- §7-3 산출 시점)."""
    if scope not in ("window", "cell", "boot"):
        raise ValueError(scope)
    if scope != "window" and job == "PROFILE" and not blind_exception:
        if not ea_seal_lifts_blind(ea_result_seal):
            raise BlindedScope(f"{scope}-scope pair-wise estimators of Job-PROFILE are blinded (§5-4 (ii)(나), H15-4)")
    if scope == "window":
        res = a2_pair(events, pi, lo, hi)
        for k in ("e_cnt", "e_time", "e_iter", "e_qcond"):
            if res["busy_n"] < MIN_BUSY_FOR_WINDOW:
                res[k] = dict(res[k], pct=None, na="NA: < 10 decode-busy snapshots in the window (§7-1 (e))")
        res["e_qcond"]["na"] = res["e_qcond"].get("na") or ("NA: empty denominator (alone window, §7-3)"
                                                             if not res["e_qcond"]["den"] else None)
        res["w_idx"] = w_idx(events, lo, hi)
        res["convention"] = "A2-pair-W (N2 window only; not comparable with whole-file values)"
        return res
    work = [e["t_launch"] if e.get("event") == "decode_iteration" else e["timestamp_monotonic_s"]
            for e in events if e.get("event") in WORK]
    res = a2_pair(events, pi)
    res["w_idx"] = w_idx(events, min(work), max(work)) if work else {}
    res["convention"] = f"A2-pair ({scope} scope)"
    if job == "PROFILE" and blind_exception:
        res["blind_exception"] = N2C_73
    elif job == "PROFILE":
        res["post_ea_result_seal"] = dict(ea_result_seal)
    return res


# --------------------------------------------------------------------------------------- positive control
def _load_n3_mix():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_n3_mix_ro", N3_MIX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp(ref, mine):
    keys = [("busy_n", ref["busy_n"], mine["busy_n"]),
            ("e_cnt.num", ref["e_cnt"]["num"], mine["e_cnt"]["num"]),
            ("e_cnt.den", ref["e_cnt"]["den"], mine["e_cnt"]["den"]),
            ("e_time.num_s", ref["e_time"]["num_s"], mine["e_time"]["num_s"]),
            ("e_time.den_s", ref["e_time"]["den_s"], mine["e_time"]["den_s"]),
            ("e_iter.num", ref["e_iter"]["num"], mine["e_iter"]["num"]),
            ("e_iter.den", ref["e_iter"]["den"], mine["e_iter"]["den"]),
            ("e_qcond.num", ref["e_qcond"]["num"], mine["e_qcond"]["num"]),
            ("e_qcond.den", ref["e_qcond"]["den"], mine["e_qcond"]["den"])]
    return [f"{k}: n3={a!r} n2={b!r}" for k, a, b in keys if a != b]


def positive_control(archive=LAM0V_ARCHIVE, cells=SELFTEST_CELLS):
    """(status, details): 'OK' when every cell is bit-identical to n3_realized_mix at π = (64,44) AND π = (92,16)
    differs on every cell; 'MISSING' when the (git-ignored) archive is absent."""
    missing = [c for c in cells if not os.path.exists(os.path.join(archive, f"tel_{c}.jsonl"))]
    if missing or not os.path.exists(N3_MIX):
        return "MISSING", missing
    n3 = _load_n3_mix()
    fails, neg_ok = [], 0
    for c in cells:
        ev = read_events(os.path.join(archive, f"tel_{c}.jsonl"))
        d = _cmp(n3.snapshot_estimators(ev, (64, 44)), a2_pair(ev, (64, 44)))
        fails += [f"{c}.{x}" for x in d]
        if _cmp(n3.snapshot_estimators(ev, (64, 44)), a2_pair(ev, (92, 16))):
            neg_ok += 1
    if neg_ok != len(cells):
        fails.append(f"negative control: π (92,16) reproduced (64,44) on {len(cells) - neg_ok} cell(s)")
    return ("OK" if not fails else "FAILED"), fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--archive", default=LAM0V_ARCHIVE)
    a = ap.parse_args(argv)
    if a.positive_control:
        st, det = positive_control(a.archive)
        print(st, det)
        return 0 if st == "OK" else (2 if st == "MISSING" else 1)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
