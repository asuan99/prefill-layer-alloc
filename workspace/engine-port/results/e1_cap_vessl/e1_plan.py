#!/usr/bin/env python3
"""E-1 cap campaign helper: registered constants, block order, budget, per-run
analysis, block validity and the registered label.

Pre-registration: PREREG_E1_CAP_VESSL_2026-10-04.md (rev1.1 = rev1 rules + wording /
descriptive-field fixes R1 R2 R3 R5 R7 R9 of VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md,
GO-with-caveats E1C-1..9; decide() is byte-for-byte the rev1 rule).

WHAT E-1 ASKS (AUDIT_LOGIC_VERDICT_2026-10-01.md candidate (7)): is the
"prefill-ward move worsens TTFT in the HI burst" sign of HE0 a product of the
running-batch cap 48 (= mamba state pool 48 on the HE0 command line)?

ONE KNOB AT A TIME.  With --disable-radix-cache and --max-running-requests set,
SGLang v0.5.10 sizes the mamba pool FROM the cap (model_runner_kv_cache_mixin.py
`handle_max_mamba_cache`, elif branch) and then clamps the running cap TO the
pool (`_resolve_max_num_reqs`).  So cap and pool move together unless the pool
is given explicitly, and "cap 192 with pool 48" is not constructible (the pool
would clamp running back to 48).  The registered contrasts are therefore:
    C48  = cap 48,  pool 48 (auto)    -- the HE0 command line
    M    = cap 48,  pool 192 (explicit) -- pool-only change   (C48 -> M)
    C192 = cap 192, pool 192 (auto)   -- cap-only change    (M  -> C192)
The label's cap test is M -> C192 (pool held at 192).  C48 -> M is the pool /
KV-reallocation control.

Everything in this file is CPU-only.  The analyzer CALLS the canonical
`benchmarks/pdmux_eval/analyze.py` (goodput predicates, percentile, the
summed-duration loader of methodology gate 7); it does not copy them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# NOT .resolve(): inside a VESSL Job `workspace/engine-port/results` is a SYMLINK to
# /data/runs/<job>/results (job_entry.sh), so resolving would leave the repo and lose
# `benchmarks/` (caught by the GPU-less container dry run, 2026-10-04).
HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
sys.path.insert(0, str(ENG / "benchmarks"))
from pdmux_eval import analyze as A  # noqa: E402  (canonical analyzer, called not copied)

# =============================================================== registered constants
PREREG_NAME = "PREREG_E1_CAP_VESSL_2026-10-04.md"
MODEL = "Zyphra/Zamba2-2.7B"
MODEL_REVISION = "31afeeac4c66b4851a54290ba57c995a68c87861"   # _migration_2026-09-17/D_misc/hf_hub_revisions.tsv
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
CONFIG_DIR_REL = "results/slo_sched"                            # under workspace/engine-port
CONFIG_DIGESTS = {                                              # unchanged since 2026-07-15 (pre-HE0)
    "pdmux_d24.yml": "b48c34c0659be3066d2bb6a337aec2f51f8a55bc56e67d547d2035d27fa92855",
    "pdmux_d34.yml": "e6526b4325477b3c52146d2bb3915307f118462d2258e33ce1ee17c13bc56801",
    "pdmux_d44.yml": "761dbe246c8f4cb263a8374a50f1e94daed7793ea3101a47613f6ae9bca8ad44",
    "pdmux_slo.yml": "bcb6f0aecb3fc6816bc52f578baf21b5673288fb7862816b7ec8254d698c89d1",
}
# HE0 serving arm (slo_sched/sharegpt_vary_bench.sbatch) -- unchanged
BACKEND = "triton"
CTX = 4096
MEM_FRACTION = 0.82
SERVER_SEED = 1                     # HE0 passed none (random); fixed here, all arms alike
NP = 200
RATE_LO = 3
RATE_HI = 12
ROUNDS = 3
SHAREGPT_CONTEXT_LEN = 4000
# HE0 bind+GATE env (sbatch MODE=bind + PDMUX_SLO_FEAS_GATE inferred from FEAS log lines)
BIND_GATE_ENV = {
    "PDMUX_SLO_SCHED": "1", "PDMUX_SLO_MODE": "binding", "PDMUX_TPOT_SLO_MS": "60",
    "PDMUX_TTFT_SLO_MS": "3000", "PDMUX_SLO_ANCHOR_IDX": "2", "PDMUX_SLO_FEAS_GATE": "1",
}

ARMS: Dict[str, Dict] = {
    # ---- primary (P): static x {C48, C192};  memory control (M): cap 48, pool 192
    "S24_C48":      dict(ctrl="static", dsm=24, cap=48,  pool=None, tel=1, role="P"),
    "S34_C48":      dict(ctrl="static", dsm=34, cap=48,  pool=None, tel=1, role="P"),
    "S44_C48":      dict(ctrl="static", dsm=44, cap=48,  pool=None, tel=1, role="P"),
    "S24_C192":     dict(ctrl="static", dsm=24, cap=192, pool=None, tel=1, role="P"),
    "S34_C192":     dict(ctrl="static", dsm=34, cap=192, pool=None, tel=1, role="P"),
    "S44_C192":     dict(ctrl="static", dsm=44, cap=192, pool=None, tel=1, role="P"),
    "S24_M":        dict(ctrl="static", dsm=24, cap=48,  pool=192,  tel=1, role="M"),
    "S44_M":        dict(ctrl="static", dsm=44, cap=48,  pool=192,  tel=1, role="M"),
    # ---- observer-effect arms (O): same as S34_* with the instrument OFF
    "S34_C48_OFF":  dict(ctrl="static", dsm=34, cap=48,  pool=None, tel=0, role="O"),
    "S34_C192_OFF": dict(ctrl="static", dsm=34, cap=192, pool=None, tel=0, role="O"),
    # ---- dynamic, descriptive only (D)
    "B_C48":        dict(ctrl="bind_gate", dsm=None, cap=48,  pool=None, tel=1, role="D"),
    "B_C192":       dict(ctrl="bind_gate", dsm=None, cap=192, pool=None, tel=1, role="D"),
}
PRIMARY_ARMS = ("S24_C48", "S44_C48", "S24_M", "S44_M", "S24_C192", "S44_C192")
OBSERVER_PAIRS = (("S34_C48", "S34_C48_OFF"), ("S34_C192", "S34_C192_OFF"))

# blocks: one Job = one block; 1..6 main, 7..8 spares (only to replace an INVALID main block)
BLOCK_SEEDS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}   # bench_serving --seed; block 1 == HE0's default seed
MAIN_BLOCKS = (1, 2, 3, 4, 5, 6)
SPARE_BLOCKS = (7, 8)
N_VALID_MIN = 4
N_USE_MAX = 6
ORDER_SALT = "e1cap-2026-10-04"

# decision constants
GATE = math.log(1.03)                # 3% effect-size gate (CLAUDE.md methodology gate 3)
M1_CAP48_BOUND_MIN = 0.50            # HI admissions that fill the cap with a queue left (KISTI 0.70, log-only)
M2_CAP192_BOUND_MAX = 0.10
M3_KV_USAGE_MAX = 0.90
PIN_MIN = 0.99
CG2_PROMPTS = 16
CG2_SEED = 7
CLIFF_BAND_MS = (55.0, 65.0)
STALL_S = 1.0                        # R9: silence >= 1 s inside a HI window (descriptive, never an exclusion rule)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}

# budget (VESSL A100 $1.48/h); ★BOOT_S is a placeholder until the lambda0 VESSL job's
# boot times are fetched (prereg sec 8 "lambda0 dependence")
RATE_USD_H = 1.48
BOOT_S = 300
BENCH_EST_S = 480                    # KISTI HE0: ~420 s first prefill -> last line (3 x (LO+HI))
TEARDOWN_S = 20
ANALYZE_S = 15
RUN_EST_S = BOOT_S + BENCH_EST_S + TEARDOWN_S + ANALYZE_S
RUN_NEED_S = 1500                    # admission: a run starts only with >= this much cell-span budget left
PREFLIGHT_EST_S = 1200               # fast unit tests (~60 s) + CG2 (two boots, 16 prompts each); container dry run 2026-10-04
HARD_CAP_S = 14400                   # 4.0 h cell-span cap per Job, from job_entry start
TAIL_RESERVE_S = 600

# ===================================================================== small helpers
def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_pool(arm: str) -> int:
    a = ARMS[arm]
    return a["pool"] if a["pool"] is not None else a["cap"]


def config_name(arm: str) -> str:
    a = ARMS[arm]
    return "pdmux_slo.yml" if a["ctrl"] == "bind_gate" else f"pdmux_d{a['dsm']}.yml"


def block_order(block: int) -> List[str]:
    if block not in BLOCK_SEEDS:
        raise ValueError(f"block {block} not registered (1..8)")
    arms = sorted(ARMS)
    random.Random(f"{ORDER_SALT}:{block}").shuffle(arms)
    return arms


def verify_configs(eng: Path) -> List[str]:
    bad = []
    for name, want in CONFIG_DIGESTS.items():
        p = eng / CONFIG_DIR_REL / name
        if not p.exists():
            bad.append(f"{name} missing")
        elif sha256(p) != want:
            bad.append(f"{name} sha256 {sha256(p)} != registered {want}")
    return bad


def budget() -> Dict:
    n = len(ARMS)
    job_s = PREFLIGHT_EST_S + n * RUN_EST_S
    worst_job_s = HARD_CAP_S + TAIL_RESERVE_S
    return {
        "arms_per_block": n, "run_est_s": RUN_EST_S, "run_need_s": RUN_NEED_S,
        "job_est_s": job_s, "job_est_gpu_h": round(job_s / 3600, 3),
        "job_est_usd": round(job_s / 3600 * RATE_USD_H, 2),
        "campaign_est_gpu_h_6_jobs": round(6 * job_s / 3600, 2),
        "campaign_est_usd_6_jobs": round(6 * job_s / 3600 * RATE_USD_H, 2),
        "job_cellspan_cap_s": HARD_CAP_S, "job_cap_plus_tail_s": worst_job_s,
        "campaign_ceiling_gpu_h_8_jobs": round(8 * worst_job_s / 3600, 2),
        "campaign_ceiling_usd_8_jobs": round(8 * worst_job_s / 3600 * RATE_USD_H, 2),
        "runs_fit_if_every_run_takes_need_s": (HARD_CAP_S - TAIL_RESERVE_S - PREFLIGHT_EST_S) // RUN_NEED_S,
        "note": "cell-span cap enforced by the runner, NOT a billing cap (LV-13); BOOT_S placeholder",
    }


def mean_ci(xs: Sequence[float]) -> Dict:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": xs[0] if xs else math.nan, "sd": math.nan,
                "ci_lo": math.nan, "ci_hi": math.nan}
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    h = T95[min(n - 1, 10)] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci_lo": m - h, "ci_hi": m + h}


def sig_pos(s: Dict) -> bool:
    return s["n"] >= 2 and s["mean"] > GATE and s["ci_lo"] > 0


def sig_neg(s: Dict) -> bool:
    return s["n"] >= 2 and s["mean"] < -GATE and s["ci_hi"] < 0


def within_gate(s: Dict) -> bool:
    return s["n"] >= 2 and s["ci_lo"] > -GATE and s["ci_hi"] < GATE


# ===================================================================== interval maths
def union(iv: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def measure(iv: Sequence[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in union(iv))


def intersect(x: Sequence[Tuple[float, float]], y: Sequence[Tuple[float, float]]):
    x, y = union(x), union(y)
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(iv, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in iv if min(b, hi) > max(a, lo)]


# ===================================================================== per-run analysis
_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_PF = re.compile(r"Prefill batch.*?#new-seq: (\d+).*?full token usage: ([0-9.]+).*?#running-req: (\d+), #queue-req: (\d+)")
_DC = re.compile(r"Decode batch, #running-req: (\d+).*?full token usage: ([0-9.]+).*?#queue-req: (\d+)")
_BANNER_POOL = re.compile(r"max_mamba_cache_size: (\d+)")
_BANNER_RUN = re.compile(r"max_total_num_tokens=(\d+).*?max_running_requests=(\d+)")


def _naive(stamp: str) -> float:
    # Server-log stamps and the runner's PHASE_MARKS wall columns are BOTH naive
    # local-time strings of the same container ("%Y-%m-%d %H:%M:%S").  Parsing
    # both with the same naive parser makes the comparison independent of the
    # timezone of whoever re-runs this analysis later.
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()


def _wall(line: str) -> Optional[float]:
    m = _TS.match(line)
    return _naive(m.group(1)) if m else None


def read_marks(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith("round"):
            continue
        r, ph, ms, me, ws, we, rc = ln.split("\t")
        rows.append(dict(round=int(r), phase=ph, mono_start=float(ms), mono_end=float(me),
                         wall_start=_naive(ws), wall_end=_naive(we), rc=int(rc)))
    return rows


def parse_server_log(path: Path, marks: List[Dict], cap: int) -> Dict:
    out = {"boot_banner": False, "max_mamba_cache_size": None, "max_running_requests": None,
           "max_total_num_tokens": None, "max_full_token_usage": 0.0, "usage_lines": 0,
           "switch_lines": 0, "feas_refused_lines": 0,
           "log_cap_bound_frac": {}, "log_prefill_lines": {}}
    if not path.exists():
        return out
    per_phase = {"LO": [0, 0], "HI": [0, 0]}
    stamps = []
    for ln in path.read_text(errors="replace").splitlines():
        w0 = _wall(ln)
        if w0 is not None:
            stamps.append(w0)
        m = _BANNER_POOL.search(ln)
        if m:
            out["max_mamba_cache_size"] = int(m.group(1))
        m = _BANNER_RUN.search(ln)
        if m:
            out["boot_banner"] = True
            out["max_total_num_tokens"] = int(m.group(1))
            out["max_running_requests"] = int(m.group(2))
        if "SLO-BIND" in ln or "SLO-SCHED" in ln:
            out["switch_lines"] += 1
        if "SLO-FEAS refused" in ln:
            out["feas_refused_lines"] += 1
        m = _PF.search(ln)
        if m:
            new, use, run, q = int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
            out["max_full_token_usage"] = max(out["max_full_token_usage"], use)
            out["usage_lines"] += 1
            w = _wall(ln)
            for mk in marks:
                if w is not None and mk["wall_start"] <= w <= mk["wall_end"]:
                    c = per_phase[mk["phase"]]
                    c[1] += 1
                    c[0] += int(run + new >= cap and q > 0)
                    break
            continue
        m = _DC.search(ln)
        if m:
            out["max_full_token_usage"] = max(out["max_full_token_usage"], float(m.group(2)))
            out["usage_lines"] += 1
    if out["usage_lines"] == 0:
        out["max_full_token_usage"] = math.nan   # unparsed log must not read as "KV not binding"
    for ph, (k, n) in per_phase.items():
        out["log_prefill_lines"][ph] = n
        out["log_cap_bound_frac"][ph] = (k / n) if n else math.nan
    # R9 (descriptive): server-log silences inside HI windows.  Stamps have 1 s
    # resolution, so consecutive stamps >= 2 s apart == >= STALL_S of silence.
    sil = []
    for mk in marks:
        if mk["phase"] != "HI":
            continue
        ts = sorted(t for t in stamps if mk["wall_start"] <= t <= mk["wall_end"])
        sil += [b - a - 1.0 for a, b in zip(ts, ts[1:]) if b - a - 1.0 >= STALL_S]
    out["log_silences_HI"] = {"count": len(sil), "total_s": sum(sil), "max_s": max(sil, default=0.0)}
    return out


def read_telemetry(path: Path) -> List[Dict]:
    ev = []
    if not path.exists():
        return ev
    for ln in path.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            ev.append(json.loads(ln))
        except json.JSONDecodeError:
            continue   # a SIGKILLed writer can leave one torn last line
    return ev


def phase_windows(events: List[Dict], marks: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    """Per phase: [first work event, last work event] inside each bench mark window."""
    work = sorted(e["timestamp_monotonic_s"] for e in events
                  if e.get("event") in ("prefill_span_start", "prefill_span_end", "decode_iteration"))
    out = {"LO": [], "HI": []}
    for mk in marks:
        inside = [t for t in work if mk["mono_start"] <= t <= mk["mono_end"]]
        if inside:
            out[mk["phase"]].append((inside[0], inside[-1]))
    return out


def telemetry_metrics(events: List[Dict], marks: List[Dict], cap: int, dsm: Optional[int]) -> Dict:
    spans, open_start = [], None
    starts = []
    dec, dec_co = [], []
    dsm_time: Dict[str, Dict[int, float]] = {"LO": {}, "HI": {}}
    dropped = 0
    for e in events:
        dropped = max(dropped, int(e.get("dropped_events", 0) or 0))
        kind = e.get("event")
        if kind == "prefill_span_start":
            open_start = e["timestamp_monotonic_s"]
            starts.append(e)
        elif kind == "prefill_span_end" and open_start is not None:
            spans.append((open_start, e["timestamp_monotonic_s"]))
            open_start = None
        elif kind == "decode_iteration":
            iv = (float(e["t_launch"]), float(e["t_sync"]))
            dec.append((iv, e))
            if e.get("prefill_in_flight") and not e.get("prefill_kernel_done_wait"):
                dec_co.append((iv, e))
    win = phase_windows(events, marks)
    res = {"n_events": len(events), "dropped_events_max": dropped,
           "n_decode_iterations": len(dec), "n_prefill_spans": len(spans), "phase": {}}
    pf_iv = spans
    dec_iv = [iv for iv, _ in dec]
    for ph in ("LO", "HI"):
        ws = win[ph]
        wlen = sum(b - a for a, b in ws)
        pf_ph = [x for a, b in ws for x in clip(pf_iv, a, b)]
        dec_ph = [x for a, b in ws for x in clip(dec_iv, a, b)]
        co = measure(intersect(pf_ph, dec_ph))
        dmeas = measure(dec_ph)
        share: Dict[int, float] = {}
        for iv, e in dec:
            for a, b in ws:
                if iv[0] >= a and iv[1] <= b:
                    share[int(e["decode_sms"])] = share.get(int(e["decode_sms"]), 0.0) + (iv[1] - iv[0])
                    break
        tot = sum(share.values()) or 1.0
        st = [s for s in starts if any(a <= s["timestamp_monotonic_s"] <= b for a, b in ws)]
        bound = [int(s["running_bs"] + s["n_new"] >= s["cap"] and s["queue_len"] > 0) for s in st]
        co_pin = [e for iv, e in dec_co if any(a <= iv[0] <= b for a, b in ws)]
        # R5 / M2b (descriptive, NOT a label input): decode steps with a queue, no prefill
        # in flight and the batch below the cap = admission blocked by something other
        # than the running cap (e.g. token reservation).  Count- and time-weighted.
        dec_w = [(iv, e) for iv, e in dec if any(a <= iv[0] <= b for a, b in ws)]
        nb = [(iv, e) for iv, e in dec_w if e.get("queue_len", 0) > 0
              and not e.get("prefill_in_flight") and int(e["bs"]) < cap]
        dsum = sum(iv[1] - iv[0] for iv, _ in dec_w)
        # R9 (descriptive, NOT an exclusion rule): gaps >= STALL_S between consecutive
        # work events inside the phase windows.
        wt = sorted(e["timestamp_monotonic_s"] for e in events
                    if e.get("event") in ("prefill_span_start", "prefill_span_end", "decode_iteration")
                    and any(a <= e["timestamp_monotonic_s"] <= b for a, b in ws))
        gaps = [y - x for x, y in zip(wt, wt[1:]) if y - x >= STALL_S]
        res["phase"][ph] = {
            "window_s": wlen,
            "coresidency_a_wall": (co / wlen) if wlen else math.nan,
            "coresidency_a_decode": (co / dmeas) if dmeas else math.nan,
            "decode_active_frac": (dmeas / wlen) if wlen else math.nan,
            "decode_sms_time_share": {str(k): v / tot for k, v in sorted(share.items())},
            "admissions": len(st),
            "cap_bound_admission_frac": (sum(bound) / len(bound)) if bound else math.nan,
            "coresident_decode_iterations": len(co_pin),
            "M2b_noncap_block_frac_count": (len(nb) / len(dec_w)) if dec_w else math.nan,
            "M2b_noncap_block_frac_time": (sum(iv[1] - iv[0] for iv, _ in nb) / dsum) if dsum else math.nan,
            "stalls": {"count": len(gaps), "total_s": sum(gaps), "max_s": max(gaps, default=0.0)},
            "pin_frac": (sum(1 for e in co_pin if int(e["decode_sms"]) == dsm) / len(co_pin))
                        if (co_pin and dsm is not None) else math.nan,
        }
    return res


def bench_metrics(path: Path) -> Dict:
    reqs, dur = A.load_bench_serving_rounds(path) if path.exists() else ([], 0.0)
    rounds, completed, errors = 0, [], 0
    if path.exists():
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            rounds += 1
            completed.append(int(rec.get("completed") or 0))
            errors += sum(1 for x in (rec.get("errors") or []) if x)
    if not reqs or dur <= 0:
        return {"rounds": rounds, "completed": completed, "errors": errors, "requests": len(reqs)}
    ttft = [r.ttft_ms for r in reqs]
    rp95 = [A.percentile(r.token_itl_ms, 0.95) for r in reqs]
    rp95f = [x for x in rp95 if not math.isnan(x)]
    toks = [v for r in reqs for v in r.token_itl_ms]
    lo, hi = CLIFF_BAND_MS
    return {
        "rounds": rounds, "completed": completed, "errors": errors, "requests": len(reqs),
        "duration_sum_s": dur,
        "ttft_ms": {"p50": A.percentile(ttft, 0.5), "p95": A.percentile(ttft, 0.95),
                    "p99": A.percentile(ttft, 0.99), "mean": statistics.fmean(ttft)},
        "req_itl_p95_ms": {"p50": A.percentile(rp95f, 0.5), "p95": A.percentile(rp95f, 0.95),
                           "p99": A.percentile(rp95f, 0.99)},
        "token_itl_ms": {"p50": A.percentile(toks, 0.5), "p95": A.percentile(toks, 0.95),
                         "p99": A.percentile(toks, 0.99)},
        "no_itl_requests": sum(1 for r in reqs if not r.token_itl_ms),
        "goodput_req_s": {
            "canonical_ttft3s_reqp95_60": sum(r.passes(3000, 60) for r in reqs) / dur,
            "legacy_ttft3s_mean_60": sum(r.passes_legacy_mean(3000, 60) for r in reqs) / dur,
            "ttft3s_reqp95_65": sum(r.passes(3000, 65) for r in reqs) / dur,
            "ttft3s_only": sum(r.ttft_ms <= 3000 for r in reqs) / dur,
        },
        "cliff_frac_reqp95_55_65": sum(1 for x in rp95f if lo <= x <= hi) / max(1, len(rp95f)),
    }


def analyze_run(rd: Path, arm: str) -> Dict:
    a = ARMS[arm]
    marks = read_marks(rd / f"PHASE_MARKS_{arm}.tsv")
    log = parse_server_log(rd / f"srv_{arm}.log", marks, a["cap"])
    out = {"arm": arm, "spec": a, "expected_pool": expected_pool(arm), "server": log,
           "LO": bench_metrics(rd / f"bench_LO_{arm}.jsonl"),
           "HI": bench_metrics(rd / f"bench_HI_{arm}.jsonl"),
           "marks": marks, "killed_by_budget": (rd / f"KILLED_{arm}").exists()}
    if a["tel"]:
        out["telemetry"] = telemetry_metrics(read_telemetry(rd / f"tel_{arm}.jsonl"), marks,
                                             a["cap"], a["dsm"])
    reasons = []
    if not log["boot_banner"]:
        reasons.append("no boot banner")
    if log["max_running_requests"] != a["cap"]:
        reasons.append(f"max_running_requests {log['max_running_requests']} != {a['cap']}")
    if log["max_mamba_cache_size"] != expected_pool(arm):
        reasons.append(f"max_mamba_cache_size {log['max_mamba_cache_size']} != {expected_pool(arm)}")
    for ph in ("LO", "HI"):
        b = out[ph]
        if b["rounds"] != ROUNDS or b["completed"] != [NP] * ROUNDS or b["errors"]:
            reasons.append(f"{ph}: rounds={b['rounds']} completed={b['completed']} errors={b['errors']}")
    if len(marks) != 2 * ROUNDS or any(m["rc"] != 0 for m in marks):
        reasons.append(f"phase marks {len(marks)} / rc {[m['rc'] for m in marks]}")
    if out["killed_by_budget"]:
        reasons.append("killed by budget enforcer")
    if a["tel"]:
        t = out["telemetry"]
        if t["n_decode_iterations"] == 0:
            reasons.append("telemetry ON but no decode_iteration events")
        if a["ctrl"] == "static":
            pin = t["phase"]["HI"]["pin_frac"]
            if not (pin >= PIN_MIN):
                reasons.append(f"HI co-resident decode on D{a['dsm']} only {pin}")
    out["valid"] = not reasons
    out["invalid_reasons"] = reasons
    return out


# ===================================================================== label
def _metric(run: Dict, key: str) -> float:
    ph, *path = key.split(".")
    v = run[ph]
    for k in path:
        v = v[k]
    return float(v)


def contrast(blocks: List[Dict[str, Dict]], hi_arm: str, lo_arm: str, key: str,
             invert: bool = False) -> Dict:
    xs = []
    for b in blocks:
        a, c = _metric(b[hi_arm], key), _metric(b[lo_arm], key)
        if a > 0 and c > 0:
            xs.append(math.log(c / a) if invert else math.log(a / c))
    return mean_ci(xs)


def manipulation(blocks: List[Dict[str, Dict]]) -> Dict:
    def frac(run):
        t = run.get("telemetry")
        if t and t["dropped_events_max"] == 0:
            return t["phase"]["HI"]["cap_bound_admission_frac"]
        return run["server"]["log_cap_bound_frac"].get("HI", math.nan)
    m1 = [frac(b[a]) for b in blocks for a in ("S24_C48", "S44_C48", "S24_M", "S44_M")]
    m2 = [frac(b[a]) for b in blocks for a in ("S24_C192", "S44_C192")]
    m3 = [b[a]["server"]["max_full_token_usage"] for b in blocks
          for a in ("S24_C192", "S44_C192", "S24_M", "S44_M")]
    med = lambda xs: statistics.median([x for x in xs if not math.isnan(x)]) if any(not math.isnan(x) for x in xs) else math.nan  # noqa: E731
    r = {"M1_cap48_bound_median": med(m1), "M2_cap192_bound_median": med(m2),
         "M3_kv_usage_max": (math.nan if (not m3 or any(math.isnan(x) for x in m3)) else max(m3))}
    # E1C-7 (descriptive): per-arm medians must be quoted with the label; the rule
    # itself still uses the pooled median (R4 = per-arm max is a user-decision option).
    r["M2_per_arm_median"] = {a: med([frac(b[a]) for b in blocks]) for a in ("S24_C192", "S44_C192")}
    r["M1_per_arm_median"] = {a: med([frac(b[a]) for b in blocks]) for a in ("S24_C48", "S44_C48", "S24_M", "S44_M")}
    r["measured"] = not any(math.isnan(r[k]) for k in
                            ("M1_cap48_bound_median", "M2_cap192_bound_median", "M3_kv_usage_max"))
    r["M1_ok"] = r["M1_cap48_bound_median"] >= M1_CAP48_BOUND_MIN
    r["M2_ok"] = r["M2_cap192_bound_median"] <= M2_CAP192_BOUND_MAX
    r["M3_ok"] = r["M3_kv_usage_max"] < M3_KV_USAGE_MAX
    return r


def decide(d48: Dict, dM: Dict, d192: Dict, did_cap: Dict, manip: Dict, n_blocks: int) -> str:
    """Registered decision rule (prereg sec 6).  Evaluated top to bottom."""
    if n_blocks < N_VALID_MIN:
        return "UNRESOLVED_N_BELOW_4"
    if not manip.get("measured", True):
        return "UNRESOLVED_MANIPULATION_UNMEASURED"
    if not manip["M1_ok"]:
        return "UNRESOLVED_CAP48_NOT_BINDING"
    if not manip["M2_ok"]:
        return "UNRESOLVED_CAP192_BINDS"
    if not manip["M3_ok"]:
        return "UNRESOLVED_KV_BINDS"
    if not sig_pos(d48):
        return "UNRESOLVED_BASELINE_REVERSED" if sig_neg(d48) else "UNRESOLVED_BASELINE_NOT_REPRODUCED"
    if not sig_pos(dM):
        return "UNRESOLVED_SIGN_LOST_AT_POOL192"
    if sig_neg(d192):
        return "CAP_BOUND_REVERSED"
    if sig_pos(did_cap):
        return "CAP_BOUND_VANISHED" if within_gate(d192) else "CAP_ATTENUATED"
    if sig_pos(d192):
        return "NOT_CAP_BOUND"
    return "UNRESOLVED_AMBIGUOUS"


PRIMARY_KEY = "HI.ttft_ms.p50"
SECONDARY_KEYS = ("HI.ttft_ms.p95", "HI.ttft_ms.p99", "HI.ttft_ms.mean")
SECONDARY_GOODPUT = ("HI.goodput_req_s.legacy_ttft3s_mean_60", "HI.goodput_req_s.ttft3s_reqp95_65",
                     "HI.goodput_req_s.ttft3s_only", "HI.goodput_req_s.canonical_ttft3s_reqp95_60")


def label_blocks(blocks: List[Dict[str, Dict]]) -> Dict:
    n = len(blocks)
    d48 = contrast(blocks, "S24_C48", "S44_C48", PRIMARY_KEY)
    dM = contrast(blocks, "S24_M", "S44_M", PRIMARY_KEY)
    d192 = contrast(blocks, "S24_C192", "S44_C192", PRIMARY_KEY)
    per = []
    for b in blocks:
        f = lambda a, c: math.log(_metric(b[a], PRIMARY_KEY) / _metric(b[c], PRIMARY_KEY))  # noqa: E731
        per.append({"d48": f("S24_C48", "S44_C48"), "dM": f("S24_M", "S44_M"),
                    "d192": f("S24_C192", "S44_C192")})
    did_cap = mean_ci([p["dM"] - p["d192"] for p in per])
    did_pool = mean_ci([p["d48"] - p["dM"] for p in per])
    manip = manipulation(blocks) if blocks else {"M1_ok": False, "M2_ok": False, "M3_ok": False}
    lab = decide(d48, dM, d192, did_cap, manip, n)
    sec = {}
    for key in SECONDARY_KEYS:
        sec[key] = {c: contrast(blocks, f"S24_{c}", f"S44_{c}", key) for c in ("C48", "M", "C192")}
    for key in SECONDARY_GOODPUT:   # goodput: higher is better -> ln(d44/d24) so + == d24 worse
        sec[key] = {c: contrast(blocks, f"S24_{c}", f"S44_{c}", key, invert=True) for c in ("C48", "M", "C192")}
    obs = {}
    for on, off in OBSERVER_PAIRS:
        ok = [b for b in blocks if b.get(on, {}).get("valid") and b.get(off, {}).get("valid")]
        for key in (PRIMARY_KEY, "HI.goodput_req_s.legacy_ttft3s_mean_60"):
            obs[f"{on}/{key}"] = contrast(ok, on, off, key)
    obs_ok = all(s["n"] >= N_VALID_MIN and abs(s["mean"]) < GATE and s["ci_lo"] <= 0 <= s["ci_hi"]
                 for s in obs.values())
    stalls = []
    for b in blocks:
        row = {}
        for arm in PRIMARY_ARMS:
            t = b[arm].get("telemetry", {}).get("phase", {}).get("HI", {}).get("stalls", {})
            row[arm] = {"tel_count": t.get("count"), "tel_max_s": t.get("max_s"),
                        "log_count": b[arm]["server"].get("log_silences_HI", {}).get("count")}
        stalls.append(row)
    return {"label": lab, "n_blocks": n, "primary_key": PRIMARY_KEY, "gate_ln": GATE,
            "stalls_HI_descriptive_not_exclusion": stalls,
            "registered_caveats": "E1C-1..9 (PREREG sec 15; VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md)",
            "d48": d48, "dM": dM, "d192": d192, "did_cap_M_minus_192": did_cap,
            "did_pool_48_minus_M": did_pool, "per_block": per, "manipulation": manip,
            "observer": obs, "observer_ok": obs_ok, "secondary": sec}


def collect(root: Path) -> Tuple[List[Dict[str, Dict]], Dict]:
    """All block dirs under root/e1v_*/block_<k>/; registered selection rule (prereg sec 7)."""
    seen: Dict[int, List[Path]] = {}
    for bd in sorted(root.glob("e1v_*/block_*")):
        try:
            k = int(bd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        seen.setdefault(k, []).append(bd)
    ledger = {"duplicates": {k: [str(p) for p in v] for k, v in seen.items() if len(v) > 1},
              "unregistered": sorted(k for k in seen if k not in BLOCK_SEEDS)}
    valid = {}
    for k, dirs in seen.items():
        if len(dirs) != 1 or k not in BLOCK_SEEDS:
            continue
        runs = {}
        for arm in ARMS:
            p = dirs[0] / f"run_{arm}.json"
            if p.exists():
                runs[arm] = json.loads(p.read_text())
        if all(runs.get(a, {}).get("valid") for a in PRIMARY_ARMS):
            valid[k] = runs
    ledger["valid_blocks"] = sorted(valid)
    use = [k for k in MAIN_BLOCKS if k in valid]
    n_invalid_main = sum(1 for k in MAIN_BLOCKS if k in seen and k not in valid)
    spares_allowed = min(len(SPARE_BLOCKS), n_invalid_main + sum(1 for k in MAIN_BLOCKS if k not in seen))
    use += [k for k in SPARE_BLOCKS[:spares_allowed] if k in valid]
    use = use[:N_USE_MAX]
    ledger["used_blocks"] = use
    return [valid[k] for k in use], ledger


ALLOWED_TEXT = {
    "CAP_BOUND_REVERSED": "On this VESSL A100 node set, with the mamba pool held at 192, raising the running cap 48->192 reversed the HI-phase TTFT sign of the prefill-ward static (d24 vs d44): d24 had lower HI TTFT p50 at cap 192. Scope: Zamba2-2.7B, HE0 trace, cudagraph ON, chunked prefill off.",
    "CAP_BOUND_VANISHED": "With the pool held at 192, raising the cap 48->192 removed the HI-phase TTFT penalty of d24 vs d44 (paired CI inside +-3%).",
    "CAP_ATTENUATED": "With the pool held at 192, raising the cap 48->192 reduced the HI-phase TTFT p50 penalty of d24 vs d44 by more than 3% (DiD_cap paired CI > 0). Whether a penalty remains at cap 192 is NOT judged unless d192 is itself sig_pos; otherwise write 'the residual penalty at cap 192 was not judged' and quote the d192 CI (E1C-3).",
    "NOT_CAP_BOUND": "At cap 192 (pool 192) the HI-phase TTFT p50 penalty of d24 vs d44 remained above 3% (sign kept; d192 paired CI > 0). Quote the DiD_cap mean and CI; a non-significant DiD_cap is NOT evidence that the penalty did not shrink (E1C-2).",
}
FORBIDDEN_TEXT = [
    "any statement that mixes these VESSL numbers with KISTI HE0 numbers (pooling, ratios, 'reproduced HE0's magnitude')",
    "'HE0 is refuted/confirmed' -- E-1 tests candidate (7) on another substrate; HE0's sign on KISTI is not re-scored",
    "'dynamic control beats/loses to static' from the D arms -- they are descriptive (no label)",
    "'the admission cap explains HE0' from a C48 vs C192 contrast alone (two knobs move; the cap test is M -> C192)",
    "'Bullet's result is reproduced' -- no Bullet-style reordering/decode-delay arm exists here (E-2)",
    "any UNRESOLVED_* label read as a negative result for candidate (7) (gate #21: measurement failure != finding)",
    "co-residency or cap-bound fractions without the run's dropped_events and the observer-effect result",
    "extending any CAP_BOUND_*/NOT_CAP_BOUND label beyond the HI TTFT p50 sign: to goodput, ITL, HE0's ranking, the practical recommendation (decode-heavy static), 'no effect of SM partitioning itself', or 'prefill-ward is better in bursts' (PREREG sec 6-4 item 8, E1C-4)",
]


# ===================================================================== CG2
def compare_cg2(a: Path, b: Path) -> Tuple[bool, str]:
    ra = [json.loads(x) for x in Path(a).read_text().splitlines() if x.strip()]
    rb = [json.loads(x) for x in Path(b).read_text().splitlines() if x.strip()]
    if len(ra) != 1 or len(rb) != 1:
        return False, f"expected one record each, got {len(ra)}/{len(rb)}"
    ta, tb = ra[0].get("generated_texts") or [], rb[0].get("generated_texts") or []
    if len(ta) != CG2_PROMPTS or len(tb) != CG2_PROMPTS:
        return False, f"expected {CG2_PROMPTS} texts, got {len(ta)}/{len(tb)}"
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            return False, f"prompt {i} differs"
    if any(e for e in (ra[0].get("errors") or []) + (rb[0].get("errors") or [])):
        return False, "request errors"
    return True, f"{CG2_PROMPTS}/{CG2_PROMPTS} identical greedy outputs"


# ===================================================================== selftest
def _synth_run(rd: Path, arm: str, ttft_hi_ms: float, cap_bound: float = 0.7,
               kv: float = 0.3, pin_ok: bool = True, seed: int = 0) -> None:
    """Synthetic run directory with the runner's file layout (used by selftest + tests)."""
    rng = random.Random(seed)
    a = ARMS[arm]
    rd.mkdir(parents=True, exist_ok=True)
    t0w, t0m = 1_700_000_000.0, 1000.0
    marks, lines, events = [], [], []
    lines.append(f"[2026-10-04 00:00:00] Mamba Cache is allocated. max_mamba_cache_size: {expected_pool(arm)}, conv_state size: 0.08GB")
    lines.append(f"[2026-10-04 00:00:01] max_total_num_tokens=300000, chunked_prefill_size=-1, max_prefill_tokens=16384, max_running_requests={a['cap']}, context_len=4096")
    tm, tw = t0m, t0w
    for r in range(1, ROUNDS + 1):
        for ph, base in (("LO", 80.0), ("HI", ttft_hi_ms)):
            ms, ws = tm, tw
            rec = {"duration": 30.0, "completed": NP, "errors": [""] * NP,
                   "ttfts": [(base * (0.8 + 0.4 * rng.random())) / 1000 for _ in range(NP)],
                   "itls": [[0.02 + 0.005 * rng.random() for _ in range(20)] for _ in range(NP)]}
            with (rd / f"bench_{ph}_{arm}.jsonl").open("a") as h:
                h.write(json.dumps(rec) + "\n")
            # telemetry: 40 admissions, prefill spans of 0.1 s, decode steps of 0.02 s
            t = ms + 1.0
            for k in range(40):
                bound = (k < int(40 * cap_bound)) if ph == "HI" else False
                run = a["cap"] - 1 if bound else 5
                events.append({"event": "prefill_span_start", "timestamp_monotonic_s": t, "span_seq": k,
                               "n_new": 1, "running_bs": run, "queue_len": 3 if bound else 0,
                               "cap": a["cap"], "dropped_events": 0})
                events.append({"event": "prefill_span_end", "timestamp_monotonic_s": t + 0.1, "dropped_events": 0})
                for j in range(8):
                    tl = t + 0.02 * j
                    co = j < 5
                    d = (a["dsm"] if (pin_ok or not co) else 108) if co else 108
                    events.append({"event": "decode_iteration", "timestamp_monotonic_s": tl + 0.019,
                                   "t_launch": tl, "t_sync": tl + 0.019, "decode_sms": d if a["dsm"] else 24,
                                   "prefill_in_flight": co, "prefill_kernel_done_wait": False,
                                   "bs": run, "queue_len": 3 if bound else 0,
                                   "dropped_events": 0})
                sec = int(ws - t0w + (t - ms))
                lines.append(f"[2026-10-04 00:{(sec // 60) % 60:02d}:{sec % 60:02d}] Prefill batch, #new-seq: 1, #new-token: 100, #cached-token: 0, full token usage: {kv:.2f}, mamba usage: 0.5, #running-req: {run}, #queue-req: {3 if bound else 0}, cuda graph: False")
                t += 0.5
            me = t + 1.0
            we = ws + (me - ms)
            marks.append((r, ph, ms, me, ws, we, 0))
            tm, tw = me + 5.0, we + 5.0
    # the synthetic log's wall clock is 2026-10-04 00:MM:SS; marks use the same naive format
    def fmt(w):
        sec = int(w - t0w)
        return f"2026-10-04 00:{(sec // 60) % 60:02d}:{sec % 60:02d}"
    with (rd / f"PHASE_MARKS_{arm}.tsv").open("w") as h:
        h.write("round\tphase\tmono_start\tmono_end\twall_start\twall_end\trc\n")
        for r, ph, ms, me, ws, we, rc in marks:
            h.write(f"{r}\t{ph}\t{ms}\t{me}\t{fmt(ws)}\t{fmt(we)}\t{rc}\n")
    (rd / f"srv_{arm}.log").write_text("\n".join(lines) + "\n")
    if a["tel"]:
        (rd / f"tel_{arm}.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _synth_campaign(root: Path, effects: Dict[str, float], n_blocks: int = 6, noise: float = 0.01,
                    **kw) -> None:
    """effects: arm -> HI TTFT p50 multiplier vs 1000 ms."""
    for k in range(1, n_blocks + 1):
        bd = root / f"e1v_job{k}" / f"block_{k}"
        rng = random.Random(100 + k)
        for arm in ARMS:
            rd = bd / "raw"
            mult = effects.get(arm, 1.0) * math.exp(rng.gauss(0, noise))
            cap_bound = kw.get("cap_bound48", 0.7) if ARMS[arm]["cap"] == 48 else kw.get("cap_bound192", 0.0)
            _synth_run(rd, arm, 1000.0 * mult, cap_bound=cap_bound, kv=kw.get("kv", 0.3), seed=k)
            res = analyze_run(rd, arm)
            (bd / f"run_{arm}.json").write_text(json.dumps(res))


def selftest() -> None:
    import tempfile
    # interval maths
    assert measure(union([(0, 1), (0.5, 2), (3, 4)])) == 3.0
    assert measure(intersect([(0, 2)], [(1, 3), (1.5, 4)])) == 1.0
    # arms / order
    assert sorted(block_order(1)) == sorted(ARMS) and block_order(1) != block_order(2)
    assert block_order(3) == block_order(3)
    assert expected_pool("S24_M") == 192 and expected_pool("S24_C48") == 48 and expected_pool("S44_C192") == 192
    # decide(): one case per label, built from synthetic stats
    s = lambda m, h: {"n": 6, "mean": m, "sd": 0.0, "ci_lo": m - h, "ci_hi": m + h}  # noqa: E731
    ok = {"M1_ok": True, "M2_ok": True, "M3_ok": True}
    P, Z, N = s(0.10, 0.03), s(0.0, 0.02), s(-0.10, 0.03)
    assert decide(P, P, N, s(0.2, 0.05), ok, 6) == "CAP_BOUND_REVERSED"
    assert decide(P, P, Z, s(0.1, 0.03), ok, 6) == "CAP_BOUND_VANISHED"
    assert decide(P, P, s(0.05, 0.03), s(0.05, 0.03), ok, 6) == "CAP_ATTENUATED"
    assert decide(P, P, P, s(0.0, 0.03), ok, 6) == "NOT_CAP_BOUND"
    assert decide(P, P, s(0.02, 0.05), s(0.08, 0.10), ok, 6) == "UNRESOLVED_AMBIGUOUS"
    assert decide(Z, P, P, Z, ok, 6) == "UNRESOLVED_BASELINE_NOT_REPRODUCED"
    assert decide(N, P, P, Z, ok, 6) == "UNRESOLVED_BASELINE_REVERSED"
    assert decide(P, Z, N, P, ok, 6) == "UNRESOLVED_SIGN_LOST_AT_POOL192"
    assert decide(P, P, N, P, dict(ok, M1_ok=False), 6) == "UNRESOLVED_CAP48_NOT_BINDING"
    assert decide(P, P, N, P, dict(ok, M2_ok=False), 6) == "UNRESOLVED_CAP192_BINDS"
    assert decide(P, P, N, P, dict(ok, M3_ok=False), 6) == "UNRESOLVED_KV_BINDS"
    assert decide(P, P, N, P, ok, 3) == "UNRESOLVED_N_BELOW_4"
    assert decide(P, P, N, P, dict(ok, measured=False), 6) == "UNRESOLVED_MANIPULATION_UNMEASURED"
    # end to end on synthetic run directories (file layout == runner's)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _synth_campaign(root, {"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90})
        blocks, ledger = collect(root)
        assert ledger["used_blocks"] == [1, 2, 3, 4, 5, 6], ledger
        lab = label_blocks(blocks)
        assert lab["label"] == "CAP_BOUND_REVERSED", lab["label"]
        assert abs(lab["d48"]["mean"] - math.log(1.10)) < 0.02, lab["d48"]
        assert lab["manipulation"]["M1_ok"] and lab["manipulation"]["M2_ok"]
        r = json.loads((root / "e1v_job1" / "block_1" / "run_S24_C48.json").read_text())
        assert r["valid"], r["invalid_reasons"]
        hi = r["telemetry"]["phase"]["HI"]
        assert abs(hi["cap_bound_admission_frac"] - 0.7) < 1e-9, hi
        assert abs(r["server"]["log_cap_bound_frac"]["HI"] - 0.7) < 1e-9, r["server"]
        assert hi["pin_frac"] == 1.0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _synth_campaign(root, {"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 1.10})
        assert label_blocks(collect(root)[0])["label"] == "NOT_CAP_BOUND"
    with tempfile.TemporaryDirectory() as td:   # cap 48 not binding on this substrate
        root = Path(td)
        _synth_campaign(root, {"S24_C48": 1.10, "S24_M": 1.10, "S24_C192": 0.90}, cap_bound48=0.2)
        assert label_blocks(collect(root)[0])["label"] == "UNRESOLVED_CAP48_NOT_BINDING"
    with tempfile.TemporaryDirectory() as td:   # pin failure invalidates the run -> block not used
        rd = Path(td)
        _synth_run(rd, "S44_C48", 1000.0, pin_ok=False)
        assert not analyze_run(rd, "S44_C48")["valid"]
    print("E1 PLAN SELFTEST OK (intervals; order; pools; 13 decide() branches; synthetic campaign "
          "-> CAP_BOUND_REVERSED / NOT_CAP_BOUND / UNRESOLVED_CAP48_NOT_BINDING; pin check)")


# ===================================================================== CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--verify-configs", metavar="ENG")
    p.add_argument("--check-block", type=int)
    p.add_argument("--order", type=int, metavar="BLOCK")
    p.add_argument("--seed", type=int, metavar="BLOCK")
    p.add_argument("--arm-spec", metavar="ARM")
    p.add_argument("--arm-env", metavar="ARM")
    p.add_argument("--const", metavar="NAME")
    p.add_argument("--budget", action="store_true")
    p.add_argument("--analyze-run", nargs=2, metavar=("RUN_DIR", "ARM"))
    p.add_argument("--out")
    p.add_argument("--block-summary", metavar="BLOCK_DIR")
    p.add_argument("--label", metavar="CAMPAIGN_ROOT")
    p.add_argument("--compare-cg2", nargs=2, metavar=("ON_JSONL", "OFF_JSONL"))
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.verify_configs:
        bad = verify_configs(Path(a.verify_configs))
        for b in bad:
            print("CONFIG_DRIFT", b)
        if not bad:
            print(f"CONFIGS OK ({len(CONFIG_DIGESTS)})")
        return 1 if bad else 0
    if a.check_block is not None:
        return 0 if a.check_block in BLOCK_SEEDS else 2
    if a.order is not None:
        print("\n".join(block_order(a.order)))
        return 0
    if a.seed is not None:
        print(BLOCK_SEEDS[a.seed])
        return 0
    if a.arm_spec:
        s = ARMS[a.arm_spec]
        print(f"CFG={config_name(a.arm_spec)} CAP={s['cap']} POOL={s['pool'] or 'auto'} "
              f"TEL={s['tel']} CTRL={s['ctrl']} DSM={s['dsm'] or '-'} EXPECT_POOL={expected_pool(a.arm_spec)}")
        return 0
    if a.arm_env:
        if ARMS[a.arm_env]["ctrl"] == "bind_gate":
            print(" ".join(f"{k}={v}" for k, v in BIND_GATE_ENV.items()))
        return 0
    if a.const:
        print(globals()[a.const])
        return 0
    if a.budget:
        print(json.dumps(budget(), indent=2))
        return 0
    if a.analyze_run:
        res = analyze_run(Path(a.analyze_run[0]), a.analyze_run[1])
        txt = json.dumps(res, indent=1, sort_keys=True, default=str)
        Path(a.out).write_text(txt) if a.out else print(txt)
        print(f"RUN {a.analyze_run[1]} valid={res['valid']} {res['invalid_reasons']}", file=sys.stderr)
        return 0
    if a.block_summary:
        bd = Path(a.block_summary)
        rows = {}
        for arm in ARMS:
            f = bd / f"run_{arm}.json"
            rows[arm] = (json.loads(f.read_text())["valid"] if f.exists() else None)
        summ = {"block_dir": str(bd), "runs": rows,
                "primary_valid": all(rows.get(x) for x in PRIMARY_ARMS)}
        print(json.dumps(summ, indent=2))
        return 0
    if a.label:
        blocks, ledger = collect(Path(a.label))
        res = label_blocks(blocks)
        res["ledger"] = ledger
        if ledger["duplicates"] or ledger["unregistered"]:
            res["label"] = "UNRESOLVED_LEDGER_CONFLICT"
        res["allowed_text"] = ALLOWED_TEXT.get(res["label"], "UNRESOLVED: report as a measurement outcome only")
        res["forbidden_text"] = FORBIDDEN_TEXT
        txt = json.dumps(res, indent=1, sort_keys=True, default=str)
        Path(a.out).write_text(txt) if a.out else print(txt)
        return 0
    if a.compare_cg2:
        ok, msg = compare_cg2(Path(a.compare_cg2[0]), Path(a.compare_cg2[1]))
        print(("CG2_OK " if ok else "CG2_MISMATCH ") + msg)
        return 0 if ok else 1
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
