#!/usr/bin/env python3
"""M2g correctness gate for multi-GPU PD-mux -- SKELETON (not run, not registered).

Status (2026-10-12): DRAFT by engine-porter for the M2.5 PREREG.  Nothing here
has executed on a GPU.  `plan` prints the boot/arm matrix; `score-*` are pure
comparators (CPU-tested in tests/test_m2g_skeleton.py); there is NO `execute`
subcommand on purpose -- the Job wiring (VESSL, one Job per comparison, arms in
the same Job) belongs to experiment-runner once the PREREG fixes the numbers
marked REGISTER below.  Design: reports/multigpu_scaling_design_2026-10-12.md
sec 6 M2g; audit C-8 ("PD-mux correctness is judged OFF<->ON at the SAME TP=N;
TP=N<->TP=1 includes reduction-order differences; an M2g PASS does not cover
anything outside the tested scope").

TIERS
=====
G0  fail-fast negative controls (no load needed; boot must FAIL at init_pdmux):
      TP=N with PDMUX_R2_POLICY=generic | hybrid, PDMUX_SLO_SCHED=1,
      PDMUX_LA_COORD=1, PDMUX_TRUE_DUAL_WORKER=1.  PASS = every one of these
      boots dies with the multigpu_guard RuntimeError text in the server log,
      and the allowed control (TP=N, legacy, R2 fixed) boots.  This is the GPU
      half of lesson 53 for the predicate: the CPU mutants prove the function,
      G0 proves the function is on the live init path of every rank.
      Auto-grid control: TP=N with an auto-grid config (no manual_divisions)
      and stock launcher must die on the pdmux_context device assert for
      gpu_id != 0 (confirms the "init_pdmux precedes set_device" reading).

G1  PRIMARY (C-8): same TP=N, PD-mux OFF vs ON, token-id equality.
      Arms in ONE Job, same server flags except --enable-pdmux/--pdmux-config-path
      (both: --disable-overlap-schedule --chunked-prefill-size -1
      --disable-radix-cache, cudagraph ON, greedy, ignore_eos).
      Load driver: results/r2_correctness/r2_correctness_client.py unchanged
      (S tier = equality gate at bs=1; O tier = probe prefill overlapping a
      background decode -> forces the green-context division and therefore
      BOTH TP comms live at once; C tier diagnostic only).
      ★G1-control: the same OFF vs ON comparison at TP=1 in the same Job.
      Bit-equality of OFF vs ON was never established at TP=1 (the R2 gate
      compared legacy-pdmux vs true-dual, both ON).  If TP=1 OFF/ON already
      differs, a TP=N difference cannot be attributed to TP and the G1
      criterion must be re-registered against the TP=1 control, not assumed.
      Why a difference is POSSIBLE even when correct: under PD-mux the prefill
      runs on _PDMUX_PREFILL_TP_GROUP whose pynccl comm is force-enabled
      (parallel_state.py `pynccl_comm.disabled = False`), while the OFF arm's
      eager prefill all-reduce goes through torch.distributed; CA v1 handles
      inputs <= max_size in both.  Same NCCL library and communicator size,
      but the PREREG must say whether bitwise equality or a G2-style tolerance
      is the G1 criterion -- REGISTER.

G2  SECONDARY: TP=N vs TP=1 teacher-forced log-probs (never free-generation
      equality).  For each prompt: reference continuation = TP=1 greedy output;
      both servers then score prompt+continuation with max_new_tokens=1,
      return_logprob=True, logprob_start_len=len(prompt)-1, top_logprobs_num=K.
      Criterion: per-position |lp_N - lp_1| <= ATOL_LP and top-1 agreement on
      every position whose TP=1 top1-top2 margin >= MARGIN.  ATOL_LP, MARGIN,
      K, prompt set: REGISTER (score-g2 refuses to run without them).
      Open engine question to settle in the first G2 boot: whether
      return_logprob is honoured on a SPLIT_PREFILL batch under PD-mux.  If not,
      G2 runs with PD-mux OFF (it is a TP-numerics check; PD-mux correctness
      is G1's job) and says so.

G3  comm stress -- the two TP comms (prefill group / decode group) executing
      concurrently, in different relative orders on different ranks.
      G3a standalone: scripts/multigpu/m2g_comm_reverse_stress.py (torchrun,
          no SGLang server): two NCCL communicators over the same ranks, one
          stream each on a green-context split; even ranks issue A then B,
          odd ranks B then A, REPEATS times, watchdog TIMEOUT_S.  PASS = no
          hang and both reductions exact (integer-valued payloads).  Arms:
          plain streams vs green-context streams (the hang mechanism of
          concern is co-residency/starvation; disjoint SM partitions are the
          thing being tested).
      G3b in-engine: O tier repeated REPEATS times per boot at TP=N with the
          watchdog; PASS = no NCCL timeout / no stuck scheduler, telemetry
          `stream_idx` identical across ranks at every logged decision
          (legacy+fixed is deterministic by construction -- this checks it).
      REPEATS, TIMEOUT_S: REGISTER.  NCCL_DEBUG=INFO plus the NCCL/CA env set
      of design sec 5-2 recorded per boot (C-10).

G4  capture + kernel identity: per stream-group cudagraph capture completes
      on every rank (server log `Capturing batches` per group, no eager
      fallback); which all-reduce path runs in decode replay vs eager prefill
      (CA v1 / pynccl / torch.distributed) identified by one short nsys probe
      boot, NOT the measurement boots (observer effect).  Expected from code
      reading (to be confirmed): decode graphs are captured with
      set_pdmux_status(False) -> get_tp_group() = _TP, so only the decode
      group's CA enters capture mode; the prefill group's CA/pynccl run eager.

G5  DP realized partitions: --dp-size N, each worker's green read-out file
      (`*.dp{r}.json` after M2) shows the expected division on gpu_id r*tp.

FAIL -> that (mode, N) is a non-operating point and is excluded from M3/M4
policy claims (design sec 6).  PASS of any tier is scoped to the tested
model, N, configs and prompt set (C-8).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Sequence

GUARD_REFUSAL_TEXT = "PD-mux with tp_size="  # multigpu_guard.check_tp_pdmux_combination
DEVICE_ASSERT_TEXT = "PD-mux auto grid (divide_sm)"  # pdmux_context patch

G0_REFUSED_ENVS = [
    {"PDMUX_R2_POLICY": "generic"},
    {"PDMUX_R2_POLICY": "hybrid"},  # + PDMUX_MODEL_PROFILE in the real boot
    {"PDMUX_SLO_SCHED": "1"},
    {"PDMUX_LA_COORD": "1"},
    {"PDMUX_TRUE_DUAL_WORKER": "1", "PDMUX_R2_POLICY": "fixed"},
]
G0_ALLOWED_ENV = {"PDMUX_R2_POLICY": "fixed"}
AUTO_GRID_CONFIG = "triage/pdmux_a100_smoke.yml"  # no manual_divisions -> divide_sm


def plan(tp_sizes: Sequence[int], model: str, config: str) -> List[Dict]:
    """Boot matrix (one Job).  Printed by `plan`; nothing is launched."""
    boots: List[Dict] = []
    common = [
        "--model-path", model, "--trust-remote-code", "--dtype", "bfloat16",
        "--disable-overlap-schedule", "--chunked-prefill-size", "-1",
        "--disable-radix-cache",
    ]
    for n in tp_sizes:
        tp = ["--tp-size", str(n)]
        if n > 1:  # the guard never refuses at tp=1
            for env in G0_REFUSED_ENVS:
                boots.append({"tier": "G0", "tp": n, "expect": "refused", "env": env,
                              "args": common + tp + ["--enable-pdmux", "--pdmux-config-path", config]})
            boots.append({"tier": "G0", "tp": n, "expect": "boots", "env": G0_ALLOWED_ENV,
                          "args": common + tp + ["--enable-pdmux", "--pdmux-config-path", config]})
            boots.append({"tier": "G0", "tp": n, "expect": "device_assert", "env": G0_ALLOWED_ENV,
                          "args": common + tp + ["--enable-pdmux", "--pdmux-config-path", AUTO_GRID_CONFIG]})
        for pdmux in (False, True):
            args = common + tp + (["--enable-pdmux", "--pdmux-config-path", config] if pdmux else [])
            boots.append({"tier": "G1" if n > 1 else "G1-control", "tp": n, "pdmux": pdmux,
                          "env": dict(G0_ALLOWED_ENV) if pdmux else {}, "args": args})
    return boots


def score_g1(off: Dict[str, List[int]], on: Dict[str, List[int]]) -> Dict:
    """Token-id equality per request id (S and O tiers only; C is diagnostic).

    Fail-closed: a request present in one arm only, or with an empty output, is
    a failure, not a skip.
    """
    ids = sorted(set(off) | set(on))
    mismatched, first_diff = [], {}
    for rid in ids:
        a, b = off.get(rid), on.get(rid)
        if not a or not b:
            mismatched.append(rid)
            first_diff[rid] = None
            continue
        if a != b:
            mismatched.append(rid)
            first_diff[rid] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"n": len(ids), "mismatched": mismatched, "first_diff": first_diff,
            "verdict": "PASS" if ids and not mismatched else "FAIL"}


def score_g2(
    ref: Dict[str, List[Dict]],
    test: Dict[str, List[Dict]],
    atol_lp: Optional[float],
    margin: Optional[float],
) -> Dict:
    """Teacher-forced comparison.  Each position: {"token": id, "lp": logprob,
    "top": [[id, lp], ...] (top-K, descending)} from the TP=1 (ref) and TP=N
    (test) servers on the SAME prompt+continuation token ids.

    Refuses (ValueError) without registered tolerances -- an unregistered
    tolerance chosen after seeing the data is exactly what C-8 forbids.
    """
    if atol_lp is None or margin is None:
        raise ValueError("ATOL_LP and MARGIN must be registered in the M2.5 PREREG before scoring")
    if not (atol_lp >= 0 and margin >= 0):
        raise ValueError("tolerances must be non-negative")
    worst, failures, n_pos = 0.0, [], 0
    for rid in sorted(set(ref) | set(test)):
        a, b = ref.get(rid), test.get(rid)
        if not a or not b or len(a) != len(b):
            failures.append((rid, None, "length/presence"))
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            n_pos += 1
            if x["token"] != y["token"]:
                failures.append((rid, i, "teacher token differs (inputs not identical)"))
                continue
            if not (math.isfinite(x["lp"]) and math.isfinite(y["lp"])):
                failures.append((rid, i, "non-finite logprob"))
                continue
            d = abs(x["lp"] - y["lp"])
            worst = max(worst, d)
            if d > atol_lp:
                failures.append((rid, i, f"|dlp|={d:.3g}"))
            top = x.get("top") or []
            if len(top) >= 2 and (top[0][1] - top[1][1]) >= margin:
                if (y.get("top") or [[None]])[0][0] != top[0][0]:
                    failures.append((rid, i, "top-1 flip above margin"))
    return {"positions": n_pos, "worst_abs_dlp": worst, "failures": failures,
            "verdict": "PASS" if n_pos and not failures else "FAIL"}


def score_g0(boot_logs: Dict[str, str], expect: Dict[str, str]) -> Dict:
    """expect[name] in {"refused", "boots", "device_assert"}; log text per boot."""
    bad = []
    for name, want in expect.items():
        log = boot_logs.get(name)
        if log is None:
            bad.append((name, "missing log"))
            continue
        refused = GUARD_REFUSAL_TEXT in log
        asserted = DEVICE_ASSERT_TEXT in log
        if want == "refused" and not refused:
            bad.append((name, "expected guard refusal"))
        elif want == "device_assert" and not asserted:
            bad.append((name, "expected device assert"))
        elif want == "boots" and (refused or asserted):
            bad.append((name, "allowed config was refused"))
    return {"bad": bad, "verdict": "PASS" if expect and not bad else "FAIL"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print the boot matrix (nothing is launched)")
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base")  # n2_plan.MODEL
    p.add_argument("--config", default="benchmarks/configs/pdmux_r2.yml")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        json.dump(plan(args.tp, args.model, args.config), sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
