#!/usr/bin/env python3
"""N3 (gate #6, SLO capacity of the target-D44 sticky-OFF B1 arm) -- registration constants,
unit list, budget/admission, preflight checks, floor -> T_K.

PRE-REGISTRATION: PREREG_N3_CAPACITY_2026-10-04.md (rev6).  Every constant below is a
registered literal; the runner reads them from here (never from env) and the structural
tests pin them.

WHAT THIS FILE DOES (CPU only, no GPU, no network):
  1. REGISTERED CONSTANTS: arm, config digest, engine-tree manifest (HEAD), model revision,
     ShareGPT digest, substrate (B0), per-shape grid (ABSOLUTE req/s), N per cell, seeds.
  2. UNITS: the ordered boot list of one shape's Job
       floor_head -> bridge (2) -> observer block (4 ON/OFF pairs, ABBA) -> grid rounds
       (4 seeds, ascending/descending alternation) -> floor_tail (diagnostic).
  3. BUDGET (3 layers: normal / worst corner / hard cap) and the per-unit admission need.
  4. PREFLIGHT CHECKS the runner turns into aborts: digests, config (FIXED_DSM in the
     division set, group count <= --sm-group-num), engine manifest == registered HEAD
     manifest, substrate B0, green read-out (targets realised, sticky_target_index null).
  5. FLOOR -> T_K.json: the registered formula T_k = k x floor, floor = median send->first-
     token TTFT of FLOOR_N concurrency-1 requests whose prefill spans are ALL REALISED at
     (108, 0) (PHASE_EVENTS; rev6: realised = prefill_span_end.stream_idx == FLOOR_STREAM_IDX
     and no decode_iteration inside the span -- NOT the entry-time prefill_span_start label,
     which is written before adjust_stream_groups).  Written BEFORE any arm cell starts.

It never reads a measured arm cell.  Scoring lives in n3_analyze.py, realised-partition
estimators in n3_realized_mix.py.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import os
import sys
from pathlib import Path

# rev5 (2026-10-06): NOT .resolve() -- inside a VESSL Job job_entry.sh replaces engine-port/results/
# with a symlink to /data/runs/<job>/results, and resolving through it moved ENG off the clone
# (benchmarks/ and src/ "MISSING"; Job pdmux-n3-capacity-a4b0c2e-20261006064114 ABORT_SELFTEST).
# abspath keeps the path the runner invoked ($ROOT/workspace/engine-port/results/...), symlinks intact.
HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
LAMBDA0_PREREG = ENG / "results" / "r2_eval" / "lambda0_prereg"
sys.path.insert(0, str(LAMBDA0_PREREG))
import lambda0_plan as LP  # noqa: E402  (λ0 rev5 file, digest-pinned below; only `ebar` and SEED_CANDIDATES are used)

# =============================================================================== (1) arm
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
MODEL_REVISION = "dc0661c829b14e5b9246c05cfa89094a0875e052"
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
BACKEND = "flashinfer"
CTX = 16384
MEM_FRACTION = 0.82
MAX_RUNNING = 48                      # = max_mamba_cache_size under --disable-radix-cache (CONSENSUS §3 item 23)
SERVER_SEED = 1
R2_POLICY = "fixed"
FIXED_DSM = 44
TARGET_PAIR = (64, 44)                # (prefill_sms, decode_sms) of the target division
FLOOR_PAIR = (108, 0)                 # the plain prefill-only group (index 0)
FLOOR_STREAM_IDX = 0                  # its stream index: stream_groups()[0]; check_green pins group 0 -> (108,0) per boot
# R3 arm-identity condition: in a scored cell, NO decode iteration (post-switch index) and NO
# decode-busy snapshot may sit on a non-target division of the config (0 allowed; violation =
# cell invalid).  Prior: λ0 VESSL 358,583 records, 0 on non-target divisions (homog5).
NONTARGET_DIVISIONS = ((92, 16), (84, 24), (74, 34))
SERVER_SM_GROUP_NUM = 8               # --sm-group-num is NOT passed => server default 8 (CONFIG_CANDIDATES C9)
EXPECT_MAMBA_CACHE = 48               # banner `max_mamba_cache_size`
EXPECT_MAX_PREFILL_TOKENS = 16384     # bench server_info `max_prefill_tokens` (binding setting, clause 8)

CONFIG_REL = "results/n3_capacity/configs/pdmux_n3_A_agn.yml"
CONFIG_DIGEST = "01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1"

# λ0 rev5 file imported above (ebar only) -- pinned so a drift cannot move the seeds.
LAMBDA0_PLAN_DIGEST = "2e51280c42e61f0e3ff7fdeccb1e7184a5091e228328d66d9d6d482e2893f86d"
# engine-port/benchmarks/pdmux_eval/analyze.py -- the canonical goodput predicate `passes()`.
ANALYZE_REL = "benchmarks/pdmux_eval/analyze.py"
ANALYZE_DIGEST = "1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2"

# =============================================================================== (2) engine tree
# Registered HEAD manifest: job 908623's 25 entries (= λ0 VESSL run, 25/25) with ONE file replaced
# by this commit's mirror (`multiplexing_mixin.py`, PDMUX_PHASE_EVENTS, commit 0922512).  N3 turns
# PHASE_EVENTS ON, so unlike E2 (one-file-difference rule) the whole manifest must equal THIS.
ENGINE_MANIFEST = {
    "sglang/srt/distributed/parallel_state.py": "167424ad6fcf8b57cd17f19ef46bb0e6ce792bd2fa36bb2b024309d8a5f1f78e",
    "sglang/srt/multiplex/dual_worker.py": "7a5022f8681446de3a04b2b868a475951594c5e766f9dbeef632ba40ade716ab",
    "sglang/srt/multiplex/multiplexing_mixin.py": "ba5835aa9f935652860f73c2e38fdb550705f3d462b7ed25015c57d28d2aff8b",
    "sglang/srt/multiplex/profile.py": "d7b6a807f141dc07e706480c1be88887c00f2e154e8a3a60b6ac26ebaf0a766a",
    "sglang/srt/multiplex/controller.py": "45fd29678b2468bc95a561afc887c0479c9ade806d2b0563858026309845eba7",
    "sglang/srt/multiplex/telemetry.py": "e4399835717ff13a50bf8c8720c40d39b308c5ec8cb7df8aff65b4cfba88b0f1",
    "sglang/srt/multiplex/holb_probe.py": "97cd7392287158523337860a334f88b1924ab272cc1d6eb6f41f083912fd1f3f",
    "sglang/srt/multiplex/chunk_probe.py": "7dfa76a9097244a8b9021dd4fe552046ee1cf3d0dbb8a3eafe9e2ad110b2c4b9",
    "sglang/srt/multiplex/green_readout.py": "84b3e728f40cf5244bf3d460bf3f33a68bdc3f7a84e85a9ce7aa54ee96e58672",
    "sglang/srt/managers/scheduler.py": "023b9f1a70592261484c629c43df88d9dfa0b5c41d35a4b96cd13e1587478321",
    "sglang/srt/configs/mamba2.py": "e0488b2035af9091a32058afabb85d65fcdefd911f118f5a8802f846fbb91501",
    "sglang/srt/models/mamba2.py": "43b2a9e4f02fa65df326fb87735a3d96fde5ac0b39ee49f1e85b03f74fbc3ca7",
    "sglang/srt/configs/zamba2.py": "97e91d40908c956014691f64aa03acaf11aa9ebd0bc84cc129a96c37b5bbe437",
    "sglang/srt/models/zamba2.py": "035a7d758ada757a76af4b76332dd1f8e769ea24195b2169fce8b7b7b7a1e264",
    "sglang/srt/server_args.py": "f2e89a1051d4537d959b48768693a21e6b713a7ecbf29d7315329242b09be388",
    "sglang/srt/model_executor/model_runner_kv_cache_mixin.py": "b673f3fb378791065b97b74f03ca68ab4f5ad0a5e1aad70d7fe101d85a69d514",
    "sglang/srt/mem_cache/memory_pool.py": "0e99909a421f004cf25d7f79ad693882301a0de75395c05566ff859ff103e1b2",
    "sglang/srt/models/nemotron_h.py": "713333e87b4a697ad9004860633a8a6fbc57f98358c9d55f2f7f4333293571b0",
    "sglang/srt/models/falcon_h1.py": "3fb851ad84cdd22b4d37fb0147e4a4c3ed24063fce026ea23e1087751f526794",
    "sglang/srt/models/granitemoehybrid.py": "b97f83121ae215b1f6b8e6771ea0bf8093311a4f40cc79b09c22ed8d58ef5ad3",
    "sglang/srt/configs/nemotron_h.py": "bcdface1473d506208d48adca7bbe21eed0d558498da9d5b389fb941351f2f76",
    "sglang/srt/configs/falcon_h1.py": "8057f56c915654f83eed5df0a8dce720744ebb33723fcf08e4f8c4c76a83e930",
    "sglang/srt/configs/granitemoehybrid.py": "f103f9bc0a893d1bbfb644f6f8137a3b252460acaf8b26803e3fb2943e1b509d",
    "sglang/srt/configs/mamba_utils.py": "58e6f716e7dec0d538777442b3e912b48aac4ae4832d1e0977d8e5676c79451a",
    "sglang/srt/layers/attention/flashinfer_backend.py": "9181648bb265f130bdf72d95bce0c2396288e9212cc2d7321c59fe8d24f86983",
}
# installed copies of tracked mirror files (sync_engine_tree.sh `install`): the mirror bytes must
# equal the registered manifest entry, so a src/ edit after registration is caught on the CPU.
MIRROR_OF = {
    "sglang/srt/multiplex/dual_worker.py": "src/multiplex/dual_worker.py",
    "sglang/srt/multiplex/multiplexing_mixin.py": "src/multiplex/multiplexing_mixin.py",
    "sglang/srt/multiplex/profile.py": "src/multiplex/profile.py",
    "sglang/srt/multiplex/controller.py": "src/multiplex/controller.py",
    "sglang/srt/multiplex/telemetry.py": "src/multiplex/telemetry.py",
    "sglang/srt/multiplex/holb_probe.py": "src/multiplex/holb_probe.py",
    "sglang/srt/multiplex/chunk_probe.py": "src/multiplex/chunk_probe.py",
    "sglang/srt/multiplex/green_readout.py": "src/multiplex/green_readout.py",
    "sglang/srt/models/nemotron_h.py": "src/models/nemotron_h.py",
}

# =============================================================================== (3) substrate B0
SUBSTRATE_SM_COUNT = 108
SUBSTRATE_CC = "8.0"
SUBSTRATE_MIN_MEM_MIB = 79000
SUBSTRATE_MIN_DRIVER_MAJOR = 550

# =============================================================================== (4) SLO scoring
K_DECISION = (3, 5, 10)               # P-3 decision points (no "main k")
X_ATTAIN = 0.90                       # P-3 attainment threshold
X_TAIL = 0.99                         # P-3/P-5 descriptive only
ITL_SLO_MS = 50.0                     # canonical predicate leg (fixed; P-5)
ITL_LADDER_MS = (40.0, 50.0, 60.0, 65.0)   # P-5 descriptive re-scoring of the STATIC arm only
K_CURVE = tuple(round(1.0 + 0.25 * i, 2) for i in range(97))   # k in [1, 25] step 0.25 (descriptive)
N_SEEDS_MIN = 4                       # CLAUDE.md methodology gate 3
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 1
# stationarity (cost_model A9 / ttft_measurement_definition_review §3.1), fixed BEFORE running
S1_FE_RATIO = 1.7                     # mean over seeds of p50(TTFT, last arrival third)/p50(first third)
S2_VIOL_GAP = 0.10                    # mean over seeds of (second-half - first-half) violation rate, per k
S3_RHO = 0.50                         # Spearman(order, prefill_queue_depth) over in-window snapshots
S3_MIN_SNAPSHOTS = 10
# 5-a cliff-guard constants (clause 7): an arm's cliff zone Z_a(k) = {r : ATT_LO <= att_a(r,k) <= ATT_HI}
CLIFF_ATT_LO = 0.50
CLIFF_ATT_HI = 0.97
CLIFF_SPACING_FRAC = 0.10             # grid spacing around a cliff, as a fraction of the reference λ*
MAX_SPACING_FRAC = 0.12               # N3 grid: every consecutive step <= 0.12 x lam_ref (B steps 0.07-0.116x, A 0.067-0.10x)
# observer effect (gate 4): OBSERVER_OK <=> both metrics n>=4, |mean ln(ON/OFF)| < ln 1.03, 95% CI contains 0
OBS_TOL = math.log(1.03)

# =============================================================================== (5) floor
K_FLOOR = K_DECISION                  # T_k = k * floor for these k (seconds, written to T_K.json)
FLOOR_N = 16                          # >= 8 distinct prompts after the warm-up (P-2)
FLOOR_SEED = 11                       # != warm-up seed 7 => different prompts
FLOOR_MIN_SPANS = FLOOR_N
# rev7 W-5 (rev6 delta audit optional item; user decision (iv) 2026-10-07: mode "length", default ON).  Three modes:
#   "off"             no W-5 predicate (descriptive `w5_window_gap_s` is still recorded)
#   "length"          rev7 length predicate: 0 <= bench duration - (last floor span end - first floor span start)
#                     <= W5_SLACK_S.  Job 3 floor_head: duration 28.154 s, window 27.273 s, gap 0.881 s (~ the last
#                     request's 64-token decode tail); Job 2 gap 0.89 s (n = 2, same GPU, shape B; shape A unverified).
#   "contain_length"  the ORIGINAL rev6-audit containment predicate (each of the last FLOOR_N span starts lies in
#                     [bench_start, bench_end] of the runner's CLOCK_MONOTONIC `marks_<unit>.txt`, n3_capacity.sh)
#                     AND the length predicate.
# Correction (rev7 delta audit R-1): the runner DOES write CLOCK_MONOTONIC bench_start/bench_end per unit
# (marks_<unit>.txt, ABORT_CLOCK checks the clock) -- an earlier draft said "no clock anchor": false.  The marks
# window wraps the whole bench_serving process, so containment ALONE would NOT have rejected the 3rd-run
# bench_serving test request (span start 1832012.777 lies in [1831996.564, 1832046.971]); the length predicate
# is a different, stronger predicate (it rejects that case: gap < 0).
W5_MODE = "length"          # user decision (iv) 2026-10-07 "권장 사항대로 진행": length predicate ON
W5_MODES = ("off", "length", "contain_length")
W5_ENABLED = W5_MODE != "off"
W5_SLACK_S = 2.0

# =============================================================================== (6) per-shape grid
# Rates are ABSOLUTE offered req/s (DS-4, clause 7).  `lam_ref` (λ*_VESSL, Job
# pdmux-lambda0-vessl-1d47d3e-20261004084945) is a REFERENCE LABEL used only for budget and the
# spacing check -- never to normalise a load.
SHAPES = {
    "B": {
        "input_len": 8192, "output_len": 64,
        "rates": (0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.53, 0.61, 0.69, 0.77),
        "span_s": 100.0, "n_min": 20,
        "obs_rate": 0.30,
        "seeds": (3704, 2084, 4861, 4589),
        # rev7 (2026-10-07, DRAFT -- before delta audit): overhead / floor / bench-fixed RE-ESTIMATED from
        # the 3rd-round Job's own TIMING.tsv (admit -> next admit; see REV7_DRAFT_NOTES_2026-10-07.md).
        # rev6 values were overhead_s 121.0 (λ0 boot-only), floor_bench_s 40.0, no bench-fixed term.
        "lam_ref": 0.690, "overhead_s": 170.0, "drain_s": 3.0, "floor_bench_s": 50.0, "bench_fixed_s": 23.0,
        "bridge": {"name": "b_r3", "rate": 1.15, "n": 200, "seeds": (4386, 4162), "dur_ref_s": 289.67},
        "lambda0_bridge_achieved": (0.6904485381127837, 0.6901401953661896, 0.6909619721961632, 0.6909626440276821),
        "lambda0_bridge_mix": {"e_cnt": 90.48, "e_time": 97.38, "e_iter": 91.68, "e_qcond": 100.0},
    },
    "A": {
        "input_len": 256, "output_len": 512,
        "rates": (1.35, 1.55, 1.80, 2.10, 2.40, 2.70, 3.00),
        "span_s": 90.0, "n_min": 0,
        "obs_rate": 1.80,
        "seeds": (898, 3801, 903, 4364),
        # rev7: shape A is UNMEASURED on this Job layout; rev6 constants kept, bench-fixed 0.0 (= rev6 model).
        # A budget forecast is therefore NOT re-validated (see notes); re-estimate before any A submission.
        "lam_ref": 2.99, "overhead_s": 160.1, "drain_s": 25.0, "floor_bench_s": 110.0, "bench_fixed_s": 0.0,
        "bridge": {"name": "a_r4", "rate": 8.0, "n": 400, "seeds": (4386, 4162), "dur_ref_s": 133.62},
        "lambda0_bridge_achieved": (2.993674869636432, 3.00990296033788, 2.997402548567481, 3.007111215624897),
        "lambda0_bridge_mix": {"e_cnt": 8.66, "e_time": 13.89, "e_iter": 8.76, "e_qcond": 18.33},
    },
}
EBAR_GUARD = (0.95, 1.05)             # 1/Ebar for every (seed, N) in a shape (λ0 analyzer guard)

# =============================================================================== (7) budget
RATE_USD_H = 1.48
# rev7 (DRAFT, user decision (ii) pending): PER-SHAPE cap (audit R-4).  B: 17100 s = 4.75 h.  rev6 value 14400 (4.0 h)
# cannot admit 48/48 once the measured constants are used (normal-corner forecast needs ~15.9k s; 17100 tolerates a
# uniform ~8 % model slowdown -- NOT a measured Job-to-Job bound, N3C-35).  Alternatives (user): 16200 (+2.0 %),
# 18000 (+14.3 %); the stress-corner fixed point below is pinned PER CAP (R-5).  A: stays 14400 s -- the rev7 cap
# does NOT authorise shape A spend (A needs its own re-estimate + approval, §0/§15).
HARD_CAP_S_B = 17100                  # from job_entry start (cell section; NOT a billing cap, LV-13 form)
HARD_CAP_S_A = 14400


def hard_cap(shape: str) -> int:
    return {"A": HARD_CAP_S_A, "B": HARD_CAP_S_B}[shape]


TAIL_RESERVE_S = 600                  # analysis + summaries after the last boot
PREFLIGHT_ALLOW_S = 300.0             # job_entry start -> first admit; rev7: measured 220 s / 224 s (Jobs 2,3; rev6 value 600.0 was the unmeasured E2 A13 value)
COLD_FIRST_BOOT_EXTRA_S = 92.1        # λ0 VESSL a_r0 251.44 s - a_r4 159.38 s
WORST_OVERHEAD_MULT = 1.25            # worst corner: per-boot overhead +25 %
# rev7 stress corner (every unit at overhead x1.25, capacity x0.8): admitted units of B at each candidate cap
# (output of `simulate`, a regression fixed point NOT a verified property -- N3C-37).  Changing the cap or the
# constants requires re-deriving this table; a cap missing from the table is a selftest failure (KeyError).
REV7_B_WORST_MIN_ADMITTED_BY_CAP = {16200: 41, 17100: 44, 18000: 46}
REV7_B_WORST_MIN_ADMITTED = REV7_B_WORST_MIN_ADMITTED_BY_CAP[HARD_CAP_S_B]
WORST_LAM_MULT = 0.80                 # worst corner: capacity 20 % below lam_ref (longer saturated cells)


# ------------------------------------------------------------------------------- helpers
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rtag(rate: float) -> str:
    return f"r{int(round(rate * 100)):03d}"


def n_prompts(shape: str, rate: float) -> int:
    s = SHAPES[shape]
    return max(int(s["n_min"]), int(rate * s["span_s"] + 0.5))


def grid_ns(shape: str):
    return [n_prompts(shape, r) for r in SHAPES[shape]["rates"]]


def ranked_seeds(shape: str, k: int = 4):
    """Same scorer as λ0 rev5 `choose_seeds` (max |Ebar-1| over the N multiset), here over THIS
    shape's grid N multiset.  Returns the top-k seeds and their scores."""
    ns = grid_ns(shape)
    scored = sorted((max(abs(LP.ebar(s, n) - 1.0) for n in ns), s) for s in LP.SEED_CANDIDATES)
    return [s for _, s in scored[:k]], [sc for sc, _ in scored[:k]]


# ------------------------------------------------------------------------------- units
# Row format (whitespace-separated, consumed by the runner):
#   NAME KIND SHAPE RATE NP SEED INLEN OUTLEN PHASE_EVENTS(0|1) BLOCK
# KIND: floor | bridge | obs_on | obs_off | grid | floor_tail
#   obs_on cells ARE grid cells of the observer rate (scored like grid cells).
def _row(name, kind, shape, rate, np_, seed, phase_events, block):
    s = SHAPES[shape]
    r = "inf" if rate is None else f"{rate:.2f}"
    return f"{name} {kind} {shape} {r} {np_} {seed} {s['input_len']} {s['output_len']} {phase_events} {block}"


def units(shape: str):
    s = SHAPES[shape]
    seeds = s["seeds"]
    out = [_row("floor_head", "floor", shape, None, FLOOR_N, FLOOR_SEED, 1, "floor")]
    br = s["bridge"]
    for sd in br["seeds"]:
        # bridge cells: PHASE_EVENTS OFF so the ONLY differences vs λ0 VESSL are config + Job/node
        out.append(_row(f"br_{br['name']}_{sd}", "bridge", shape, br["rate"], br["n"], sd, 0, "bridge"))
    obs = s["obs_rate"]
    nob = n_prompts(shape, obs)
    for i, sd in enumerate(seeds):          # ABBA: ON,OFF / OFF,ON / ON,OFF / OFF,ON
        on = _row(f"g_{rtag(obs)}_{sd}", "obs_on", shape, obs, nob, sd, 1, "observer")
        off = _row(f"off_{rtag(obs)}_{sd}", "obs_off", shape, obs, nob, sd, 0, "observer")
        out += [on, off] if i % 2 == 0 else [off, on]
    rest = [r for r in s["rates"] if abs(r - obs) > 1e-9]
    # R1 (rules verdict rev1 §4): round 1,3 DESCENDING, round 2,4 ASCENDING.  Drift balance is the
    # same as before, but a budget cut now removes the HIGHEST rate first (the last unit before
    # floor_tail is the top rate of round 4), never the lowest rate the first-failure rule must read.
    for i, sd in enumerate(seeds):
        order = list(reversed(rest)) if i % 2 == 0 else rest
        for r in order:
            out.append(_row(f"g_{rtag(r)}_{sd}", "grid", shape, r, n_prompts(shape, r), sd, 1, f"round{i + 1}"))
    out.append(_row("floor_tail", "floor_tail", shape, None, FLOOR_N, FLOOR_SEED, 1, "tail"))
    return out


def parse_row(row: str) -> dict:
    f = row.split()
    return {"name": f[0], "kind": f[1], "shape": f[2], "rate": None if f[3] == "inf" else float(f[3]),
            "np": int(f[4]), "seed": int(f[5]), "input_len": int(f[6]), "output_len": int(f[7]),
            "phase_events": int(f[8]), "block": f[9]}


def scored_cells(shape: str):
    """(rate, seed) -> unit name of every cell that enters the capacity decision."""
    return {(parse_row(r)["rate"], parse_row(r)["seed"]): parse_row(r)["name"]
            for r in units(shape) if parse_row(r)["kind"] in ("grid", "obs_on")}


# ------------------------------------------------------------------------------- budget
def bench_s(shape: str, u: dict, lam_mult: float = 1.0) -> float:
    s = SHAPES[shape]
    if u["kind"] in ("floor", "floor_tail"):
        return s["floor_bench_s"]
    fixed = s["bench_fixed_s"]       # rev7: bench_serving fixed cost (own warm-up request + sleep + teardown)
    if u["kind"] == "bridge":
        return s["bridge"]["dur_ref_s"] / lam_mult + fixed
    lam = s["lam_ref"] * lam_mult
    span = (u["np"] - 1) / u["rate"]
    sat = u["np"] / lam              # throughput-limited duration when offered >= capacity
    return max(span, sat) + s["drain_s"] + fixed


def unit_cost_s(shape: str, row: str, corner: str = "normal") -> float:
    u = parse_row(row)
    s = SHAPES[shape]
    om = WORST_OVERHEAD_MULT if corner == "worst" else 1.0
    lm = WORST_LAM_MULT if corner == "worst" else 1.0
    return bench_s(shape, u, lm) + s["overhead_s"] * om


def need_s(shape: str, name: str) -> int:
    for r in units(shape):
        if parse_row(r)["name"] == name:
            return int(math.ceil(unit_cost_s(shape, r, "worst")))
    raise KeyError(name)


def simulate(shape: str, corner: str) -> dict:
    t = PREFLIGHT_ALLOW_S + COLD_FIRST_BOOT_EXTRA_S
    limit = hard_cap(shape) - TAIL_RESERVE_S
    admitted, by_block = 0, {}
    for r in units(shape):
        u = parse_row(r)
        if limit - t < need_s(shape, u["name"]):
            break
        c = unit_cost_s(shape, r, corner)
        t += c
        admitted += 1
        by_block[u["block"]] = by_block.get(u["block"], 0.0) + c
    n = len(units(shape))
    return {"corner": corner, "container_s": t, "gpu_h": t / 3600.0, "usd": t / 3600.0 * RATE_USD_H,
            "admitted": admitted, "units": n, "all_admitted": admitted == n, "by_block_s": by_block}


def max_uniform_slowdown(shape: str) -> float:
    """Largest uniform factor f on every unit's normal-corner cost for which ALL units are still admitted
    (model arithmetic only -- not a measured Job-to-Job bound, N3C-35)."""
    lo, hi = 0.5, 2.0
    for _ in range(40):
        f = (lo + hi) / 2
        t = PREFLIGHT_ALLOW_S + COLD_FIRST_BOOT_EXTRA_S
        lim = hard_cap(shape) - TAIL_RESERVE_S
        ok = True
        for r in units(shape):
            if lim - t < need_s(shape, parse_row(r)["name"]):
                ok = False
                break
            t += unit_cost_s(shape, r, "normal") * f
        lo, hi = (f, hi) if ok else (lo, f)
    return lo


def budget_table(shape: str) -> dict:
    return {"shape": shape, "rate_usd_h": RATE_USD_H,
            "normal": simulate(shape, "normal"), "worst": simulate(shape, "worst"),
            "hard_cap": {"cell_section_s": hard_cap(shape), "gpu_h": hard_cap(shape) / 3600.0,
                         "usd": hard_cap(shape) / 3600.0 * RATE_USD_H, "tail_reserve_s": TAIL_RESERVE_S},
            "note": (("rev7 shape B: per-unit overhead 170 s, bench fixed cost 23 s, floor bench 50 s are medians of the "
                      "3rd-round Job pdmux-n3-capacity-38ed3f4-20261006092813 TIMING.tsv (41 warm units / 42 non-floor "
                      "units, one GPU c36b3c0c; NOT validated across Jobs or GPUs); preflight 300 s = 220/224 s (n=2, same "
                      "GPU); cold first-boot extra 92.1 s is the lambda0 VESSL value (this Job measured 59 s, n=1). "
                      if shape == "B" else
                      "shape A: constants are the rev6 lambda0 VESSL measurements (E2 A13 5-1) and are UNMEASURED on the N3 "
                      "layout (not re-validated by rev7); bench fixed cost 0 = rev6 model. ")
                     + "bench durations are a model (span vs N/lam_ref); tail and export are outside the cap "
                       "(NOT a billing cap).")}


# ------------------------------------------------------------------------------- config
def parse_config(path: Path) -> dict:
    """Minimal YAML subset reader for the pdmux config (no PyYAML dependency); cross-checked
    against yaml.safe_load in the selftest when PyYAML is importable."""
    out = {"manual_divisions": []}
    in_div = False
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("manual_divisions:"):
            in_div = True
            continue
        if in_div and line.lstrip().startswith("- ["):
            vals = line.split("[", 1)[1].split("]", 1)[0]
            out["manual_divisions"].append([int(v) for v in vals.split(",")])
            continue
        in_div = False
        k, v = line.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


def check_config(path: Path, fixed_dsm: int = FIXED_DSM, server_groups: int = SERVER_SM_GROUP_NUM):
    """Return a list of problems (empty = OK).  The FIXED_DSM membership check closes the silent
    mis-arm of CONFIG_CANDIDATES §1 implication 3 (sticky-OFF fixed with an absent D keeps the
    current index and only logs a warning)."""
    cfg = parse_config(path)
    bad = []
    divs = cfg.get("manual_divisions", [])
    decodes = [d[1] for d in divs]
    if fixed_dsm not in decodes:
        bad.append(f"PDMUX_R2_FIXED_DSM={fixed_dsm} not in the config's division decode SMs {decodes}")
    if decodes.count(fixed_dsm) > 1:
        bad.append(f"decode SM {fixed_dsm} appears {decodes.count(fixed_dsm)} times (ambiguous target)")
    g = cfg.get("sm_group_num")
    if g is None or g != len(divs) + 2:
        bad.append(f"sm_group_num {g} != len(manual_divisions)+2 = {len(divs) + 2}")
    if g is not None and g > server_groups:
        bad.append(f"stream groups {g} > server --sm-group-num {server_groups}")
    if TARGET_PAIR not in [tuple(d[:2]) for d in divs]:
        bad.append(f"target pair {TARGET_PAIR} absent from manual_divisions")
    if divs and divs[0][2] != 0:
        bad.append("first threshold must be 0 (scan unbound-index crash path)")
    for p, d, _ in divs:
        if p + d != SUBSTRATE_SM_COUNT:
            bad.append(f"division ({p},{d}) does not sum to {SUBSTRATE_SM_COUNT}")
    return bad


def stream_groups(path: Path):
    """Index -> (prefill, decode): [(108,0)] + divisions + [(0,108)] (pdmux_context.py:127)."""
    divs = parse_config(path)["manual_divisions"]
    return [FLOOR_PAIR] + [tuple(d[:2]) for d in divs] + [(0, SUBSTRATE_SM_COUNT)]


# ------------------------------------------------------------------------------- digests
def registered_files(eng: Path) -> dict:
    return {
        CONFIG_REL: CONFIG_DIGEST,
        "results/r2_eval/lambda0_prereg/lambda0_plan.py": LAMBDA0_PLAN_DIGEST,
        ANALYZE_REL: ANALYZE_DIGEST,
    }


def verify_digests(eng: Path):
    bad = []
    for rel, want in sorted(registered_files(eng).items()):
        p = eng / rel
        got = sha256(p) if p.exists() else "MISSING"
        if got != want:
            bad.append(f"{rel}: {got} != registered {want}")
    return bad


def manifest_entries(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        key = p.split("/python/", 1)[1] if "/python/" in p else p
        out[key] = h
    return out


def engine_check(this_manifest: Path):
    a = manifest_entries(this_manifest)
    return [f"{k}: installed={a.get(k, 'ABSENT')[:16]} registered={ENGINE_MANIFEST.get(k, 'ABSENT')[:16]}"
            for k in sorted(set(a) | set(ENGINE_MANIFEST)) if a.get(k) != ENGINE_MANIFEST.get(k)]


def mirror_check(eng: Path):
    return [f"{k}: mirror {rel} = {sha256(eng / rel)[:16]} != registered {ENGINE_MANIFEST[k][:16]}"
            for k, rel in sorted(MIRROR_OF.items()) if sha256(eng / rel) != ENGINE_MANIFEST[k]]


def check_substrate(csv_line: str, sm_count) -> list:
    """Same rule as E2 A13 B0 (copied, not imported: E2 files are sealed)."""
    f = [x.strip() for x in csv_line.split(",")]
    if len(f) < 5:
        return [f"unparseable nvidia-smi line: {csv_line!r}"]
    name, cc, mem, drv, mig = f[:5]
    bad = []
    if "A100" not in name or "80GB" not in name:
        bad.append(f"gpu name {name!r} is not an A100 80GB")
    if cc != SUBSTRATE_CC:
        bad.append(f"compute_cap {cc} != {SUBSTRATE_CC}")
    try:
        if float(mem) < SUBSTRATE_MIN_MEM_MIB:
            bad.append(f"memory.total {mem} MiB < {SUBSTRATE_MIN_MEM_MIB}")
    except ValueError:
        bad.append(f"memory.total {mem!r} unparseable")
    try:
        if int(drv.split(".")[0]) < SUBSTRATE_MIN_DRIVER_MAJOR:
            bad.append(f"driver {drv} < {SUBSTRATE_MIN_DRIVER_MAJOR}")
    except ValueError:
        bad.append(f"driver {drv!r} unparseable")
    if mig not in ("Disabled", "[N/A]", "N/A"):
        bad.append(f"MIG mode {mig!r} (full GPU required)")
    try:
        if int(sm_count) != SUBSTRATE_SM_COUNT:
            bad.append(f"SM count {sm_count} != {SUBSTRATE_SM_COUNT}")
    except (TypeError, ValueError):
        bad.append(f"SM count {sm_count!r} unparseable")
    return bad


# ------------------------------------------------------------------------------- green read-out
def check_green(green_json: Path, cfg_path: Path):
    """Clause 1 + preflight: every division group's green context realises its target smCount,
    the plain groups (index 0 and last) carry no green context, sticky_target_index is null and
    no group is a sticky target.  Returns problems (empty = OK)."""
    try:
        g = json.loads(Path(green_json).read_text())
    except (OSError, ValueError) as exc:
        return [f"green read-out unreadable: {exc!r}"]
    want = stream_groups(cfg_path)
    bad = []
    if g.get("sticky_target_index") is not None:
        bad.append(f"sticky_target_index = {g.get('sticky_target_index')!r} (registered null)")
    groups = {grp.get("stream_index"): grp for grp in g.get("groups", [])}
    if sorted(groups) != list(range(len(want))):
        bad.append(f"stream indices {sorted(groups)} != 0..{len(want) - 1}")
    for i, (p, d) in enumerate(want):
        grp = groups.get(i)
        if grp is None:
            continue
        if grp.get("is_sticky_target"):
            bad.append(f"group {i} is_sticky_target true")
        if (grp.get("target_prefill_sm"), grp.get("target_decode_sm")) != (p, d):
            bad.append(f"group {i} targets {(grp.get('target_prefill_sm'), grp.get('target_decode_sm'))} != {(p, d)}")
        division = 1 <= i <= len(want) - 2
        for side, sm in (("prefill", p), ("decode", d)):
            rec = grp.get(side) or {}
            got = (rec.get("green_sm") or {}).get("smCount")
            if division:
                if rec.get("green_ctx_is_null", True) or got != sm:
                    bad.append(f"group {i} {side}: green smCount {got} != target {sm}")
            elif not rec.get("green_ctx_is_null", True):
                bad.append(f"group {i} {side}: plain group carries a green context")
    return bad


# ------------------------------------------------------------------------------- floor -> T_K
def _read_jsonl(path: Path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def floor_from(bench_path: Path, tel_path: Path, shape: str, w5=None, marks_path=None) -> dict:
    """P-2 literal (rev6, 2026-10-06): the floor is valid iff
      (i)   the bench completed FLOOR_N/FLOOR_N with no error, exact lengths, concurrency 1;
      (ii)  the telemetry holds >= FLOOR_MIN_SPANS `prefill_span_start` events, every one paired
            (same span_seq) with a `prefill_span_end`, and no event reports dropped_events > 0
            (an absence-of-decode argument is not evidential under drops);
      (iii) EVERY span in the boot is REALISED at (108, 0): its `prefill_span_end.stream_idx`
            == FLOOR_STREAM_IDX and no `decode_iteration` event lies inside
            [span_start.timestamp, span_end.timestamp];
      (iv)  the LAST FLOOR_N spans (the floor requests, concurrency 1, in order) have
            n_new == 1 and queue_len == 0 at entry;
      (v)   (rev6 audit R-2) after EACH of the last FLOOR_N spans at least one `decode_iteration`
            is recorded before the next prefill_span_start (file end for the last one): the
            absence argument of (iii) needs the decode recorder shown live (lesson 232);
      (vi)  span_seq is unique among prefill_span_start and among prefill_span_end (audit W-1).
    floor_s = median of the FLOOR_N client TTFTs (send -> first token).  T_k = k * floor_s.
    Descriptives only (audit R-3, never decision inputs): `floor_entry_race_count` (floor spans
    whose ENTRY label != (108,0) or entry running_bs != 0 -- the fingerprint of an admission
    stall that inflates that request's TTFT) and `floor_admission_gap_ms` (each floor span start
    minus the latest decode_iteration before it).

    rev6 rationale: `prefill_span_start` is recorded at admission, BEFORE `update_running_batch`
    removes finished requests and BEFORE `adjust_stream_groups` picks the stream, so its
    (prefill_sms, decode_sms, running_bs) can carry the previous iteration's state (Job
    pdmux-n3-capacity-79999aa-20261006081638: span 6 entry label (0,108)/running_bs 1, realised
    stream 0, 0 decode iterations inside -> rev5 (iii) aborted a valid floor).  Those entry
    labels are kept as descriptives (`entry_*`), never as decision inputs."""
    s = SHAPES[shape]
    # w5: None -> module default W5_MODE; True -> "length"; False -> "off"; or one of W5_MODES
    mode = W5_MODE if w5 is None else ("length" if w5 is True else "off" if w5 is False else w5)
    if mode not in W5_MODES:
        raise ValueError(f"unknown W-5 mode {mode!r}")
    w5 = mode != "off"
    res = {"shape": shape, "valid": False, "problems": [],
           "rule": "rev6" if not w5 else f"rev7+W5:{mode}", "w5_enabled": w5, "w5_mode": mode}
    recs = _read_jsonl(bench_path)
    if len(recs) != 1:
        res["problems"].append(f"bench record count {len(recs)} != 1")
        return res
    b = recs[0]
    errs = b.get("errors") or []
    n_err = sum(1 for e in errs if e) if isinstance(errs, list) else int(errs or 0)
    ttfts = [float(x) for x in (b.get("ttfts") or [])]
    if int(b.get("completed", -1)) != FLOOR_N or n_err or len(ttfts) != FLOOR_N:
        res["problems"].append(f"completed={b.get('completed')} errors={n_err} ttfts={len(ttfts)}")
    if int(b.get("total_input_tokens", -1)) != FLOOR_N * s["input_len"]:
        res["problems"].append(f"total_input_tokens {b.get('total_input_tokens')} != {FLOOR_N * s['input_len']}")
    if b.get("max_concurrency") != 1:
        res["problems"].append(f"max_concurrency {b.get('max_concurrency')} != 1")
    ev = _read_jsonl(tel_path)
    spans = [e for e in ev if e.get("event") == "prefill_span_start"]
    ends, dup_end = {}, []
    for e in ev:
        if e.get("event") == "prefill_span_end":
            if e.get("span_seq") in ends:
                dup_end.append(e.get("span_seq"))
            ends.setdefault(e.get("span_seq"), e)       # first end per span_seq (duplicates invalidate, W-1)
    seqs = [e.get("span_seq") for e in spans]
    dup_start = sorted({q for q in seqs if seqs.count(q) > 1})
    if dup_start or dup_end:
        res["problems"].append(f"span_seq not unique: starts {dup_start[:5]} ends {sorted(set(dup_end))[:5]}")
    decs = sorted(float(e["timestamp_monotonic_s"]) for e in ev
                  if e.get("event") == "decode_iteration" and e.get("timestamp_monotonic_s") is not None)
    res["n_spans"] = len(spans)
    res["n_span_ends"] = len(ends)
    res["n_decode_iterations"] = len(decs)
    if len(spans) < FLOOR_MIN_SPANS:
        res["problems"].append(f"{len(spans)} prefill_span_start events < {FLOOR_MIN_SPANS} (PHASE_EVENTS off?)")
    res["dropped_events_max"] = max([int(e.get("dropped_events", 0) or 0) for e in ev] or [0])
    if res["dropped_events_max"] > 0:
        res["problems"].append(f"dropped_events_max {res['dropped_events_max']} > 0 (decode absence not evidential)")
    # descriptives: entry-time labels (recorded before adjust_stream_groups; NOT decision inputs)
    res["entry_label_off_count"] = sum(1 for e in spans if (e.get("prefill_sms"), e.get("decode_sms")) != FLOOR_PAIR)
    res["entry_running_bs_nonzero_count"] = sum(1 for e in spans if e.get("running_bs") != 0)
    # (iii) realised partition per span
    unpaired, off_stream, dec_inside, bad_ts = [], [], [], []
    for st in spans:
        seq = st.get("span_seq")
        en = ends.get(seq)
        if en is None:
            unpaired.append(seq)
            continue
        if en.get("stream_idx") != FLOOR_STREAM_IDX:
            off_stream.append((seq, en.get("stream_idx")))
        try:
            t0, t1 = float(st["timestamp_monotonic_s"]), float(en["timestamp_monotonic_s"])
        except (KeyError, TypeError, ValueError):
            bad_ts.append(seq)
            continue
        if t1 < t0:
            bad_ts.append(seq)
            continue
        n_in = bisect.bisect_right(decs, t1) - bisect.bisect_left(decs, t0)
        if n_in:
            dec_inside.append((seq, n_in))
    res["end_stream_idx_set"] = sorted({e.get("stream_idx") for e in ends.values()}, key=lambda x: (x is None, x))
    if unpaired:
        res["problems"].append(f"{len(unpaired)} prefill spans without prefill_span_end (span_seq {unpaired[:5]})")
    if bad_ts:
        res["problems"].append(f"{len(bad_ts)} prefill spans with unusable timestamps (span_seq {bad_ts[:5]})")
    if off_stream:
        res["problems"].append(f"{len(off_stream)} prefill spans realised off stream {FLOOR_STREAM_IDX} = {FLOOR_PAIR} "
                               f"(span_seq, stream_idx) {off_stream[:5]}")
    if dec_inside:
        res["problems"].append(f"{len(dec_inside)} prefill spans with decode_iteration inside the span "
                               f"(span_seq, count) {dec_inside[:5]}")
    # (iv) floor requests: single admission, empty queue at entry
    last = spans[-FLOOR_N:]
    busy = [e for e in last if e.get("n_new") != 1 or e.get("queue_len") != 0]
    if busy:
        res["problems"].append(f"{len(busy)} floor spans with n_new/queue_len != 1/0")
    # (rev7 W-5, mode W5_MODE) the last FLOOR_N spans must be the bench's own requests
    try:
        lj = list(range(len(spans)))[-FLOOR_N:]
        w_t0 = float(spans[lj[0]]["timestamp_monotonic_s"])
        w_t1 = float(ends[spans[lj[-1]].get("span_seq")]["timestamp_monotonic_s"])
        gap = float(b.get("duration")) - (w_t1 - w_t0)
        res["w5_window_gap_s"] = round(gap, 3)           # descriptive, always recorded
        if w5 and not (0.0 <= gap <= W5_SLACK_S):
            res["problems"].append(f"W-5: bench duration - floor span window = {gap:.3f} s not in [0, {W5_SLACK_S}] "
                                   "(stray span in the last FLOOR_N, or window outside the bench)")
    except (KeyError, TypeError, ValueError, IndexError):
        res["w5_window_gap_s"] = None
        if w5:
            res["problems"].append("W-5: window/duration not computable (fail-closed when W-5 is enabled)")
    if mode == "contain_length":
        mp = Path(marks_path) if marks_path else Path(bench_path).with_name(Path(bench_path).name.replace("bench_", "marks_", 1)).with_suffix(".txt")
        try:
            mk = dict(ln.split() for ln in Path(mp).read_text().splitlines() if ln.strip())
            m0, m1 = float(mk["bench_start"]), float(mk["bench_end"])
            outside = [e.get("span_seq") for e in spans[-FLOOR_N:]
                       if not (m0 <= float(e["timestamp_monotonic_s"]) <= m1)]
            res["w5_marks_window_s"] = [m0, m1]
            if outside:
                res["problems"].append(f"W-5 containment: {len(outside)} of the last {FLOOR_N} span starts outside "
                                       f"[bench_start, bench_end] (span_seq {outside[:5]})")
        except (OSError, KeyError, TypeError, ValueError):
            res["problems"].append("W-5 containment: marks file missing/unparseable (fail-closed)")
    # (v) R-2: the decode recorder must be shown live after every floor span
    last_idx = list(range(len(spans)))[-FLOOR_N:]
    not_live, gaps, race = [], [], 0
    for j in last_idx:
        st, en = spans[j], ends.get(spans[j].get("span_seq"))
        try:
            t0 = float(st["timestamp_monotonic_s"])
        except (KeyError, TypeError, ValueError):
            continue
        # R-3 descriptive: admission gap = start minus the latest decode_iteration before it
        k = bisect.bisect_left(decs, t0)
        gaps.append(round(1000.0 * (t0 - decs[k - 1]), 3) if k > 0 else None)
        if (st.get("prefill_sms"), st.get("decode_sms")) != FLOOR_PAIR or st.get("running_bs") != 0:
            race += 1
        if en is None:
            continue
        try:
            t1 = float(en["timestamp_monotonic_s"])
            t_next = float(spans[j + 1]["timestamp_monotonic_s"]) if j + 1 < len(spans) else math.inf
        except (KeyError, TypeError, ValueError):
            continue
        if bisect.bisect_right(decs, t_next) - bisect.bisect_right(decs, t1) < 1:
            not_live.append(st.get("span_seq"))
    res["floor_entry_race_count"] = race
    res["floor_admission_gap_ms"] = gaps
    if not_live:
        res["problems"].append(f"{len(not_live)} floor spans with no decode_iteration before the next span "
                               f"-- decode recorder not shown live after floor span (span_seq {not_live[:5]})")
    if not res["problems"]:
        floor_s = statistics.median(ttfts)
        res.update(valid=True, floor_s=floor_s,
                   ttfts_s=ttfts,
                   dispersion_max_over_min=max(ttfts) / min(ttfts),
                   T_k_s={str(k): k * floor_s for k in K_FLOOR},
                   formula="T_k = k * median(send->first-token TTFT of FLOOR_N concurrency-1 requests)")
    return res


def tk_problems(tk: dict) -> list:
    """R2: a T_K.json is scoreable only if it is a REAL floor: valid, no `dryrun` key, FLOOR_N
    client TTFTs recorded, and T_k recomputable from floor_s.  The dry-run placeholder fails."""
    bad = []
    if not tk.get("valid"):
        bad.append("valid is not true")
    if "dryrun" in tk:
        bad.append("dry-run placeholder (key `dryrun`)")
    tt = tk.get("ttfts_s")
    if not isinstance(tt, list) or len(tt) != FLOOR_N:
        bad.append(f"ttfts_s missing or not {FLOOR_N} values")
    else:
        try:
            if abs(statistics.median(float(x) for x in tt) - float(tk.get("floor_s"))) > 1e-9:
                bad.append("floor_s != median(ttfts_s)")
            for k in K_FLOOR:
                if abs(float(tk["T_k_s"][str(k)]) - k * float(tk["floor_s"])) > 1e-9:
                    bad.append(f"T_{k} != {k} x floor_s")
        except (TypeError, ValueError, KeyError) as exc:
            bad.append(f"unparseable: {exc!r}")
    return bad


def tk_file_problems(path) -> list:
    try:
        return tk_problems(json.loads(Path(path).read_text()))
    except (OSError, ValueError) as exc:
        return [f"unreadable: {exc!r}"]


# ------------------------------------------------------------------------------- P3 option (ii)
# rev4 (prereg §2-3 R6, user decision 2026-10-05): the SAME rule and constants as E2 A14 §3-1
# (`e2_vessl_plan_r2.p3_verdict`).  Over all boots of THIS Job (resumed units are other boots and
# never enter P3_KEYS.txt):
#   (a) max_mamba_cache_size == 48 on every boot, exact;
#   (b) every max_total_num_tokens >= MAX_RUNNING x CTX = 786,432 (KV can never bind admission);
#   (c) (max - min) / max of max_total_num_tokens <= 1e-4.
# 1e-4 is a USER-DECIDED constant, not a data fit.  N3 runs ONE 6-group config, so (c) only ever
# compares boots of that config (6-group 2,721,290 vs 5-group 2,722,025 = rel 2.70e-4 is a config
# difference that cannot occur inside an N3 Job; CONFIG_DIGEST aborts first).
P3_MAX_RUNNING = MAX_RUNNING              # 48
P3_CTX = CTX                              # 16384
P3_MAMBA_EXPECT = EXPECT_MAMBA_CACHE      # 48
P3_TOKEN_FLOOR = P3_MAX_RUNNING * P3_CTX  # 786,432
P3_REL_TOL = 1e-4


def p3_verdict(rows, max_running, ctx):
    """rows = P3_KEYS.txt lines split on whitespace (`<mamba> <tokens>`, one per boot so far).
    Returns (ok, line); line starts with "PASS " or "FAIL ".  Fail-closed on anything unparseable."""
    try:
        mr, cx = int(max_running), int(ctx)
    except (TypeError, ValueError):
        return False, f"FAIL registered_constants unparseable max_running={max_running!r} ctx={ctx!r}"
    if mr != P3_MAX_RUNNING or cx != P3_CTX:
        return False, (f"FAIL registered_constants max_running={max_running} ctx={ctx} "
                       f"!= {P3_MAX_RUNNING} x {P3_CTX}")
    rows = [r for r in rows if r]
    if not rows:
        return False, "FAIL no_keys"
    badr = [r for r in rows if len(r) != 2 or not r[0].isdecimal() or not r[1].isdecimal()]
    if badr:
        return False, "FAIL banner_missing " + ";".join(" ".join(r) for r in badr)
    mamba = sorted({int(r[0]) for r in rows})
    tok = [int(r[1]) for r in rows]
    if mamba != [P3_MAMBA_EXPECT]:
        return False, f"FAIL mamba={mamba} != [{P3_MAMBA_EXPECT}]"
    if min(tok) < P3_TOKEN_FLOOR:
        return False, f"FAIL tokens_min={min(tok)} < floor={P3_TOKEN_FLOOR}"
    rel = (max(tok) - min(tok)) / max(tok)
    ok = rel <= P3_REL_TOL
    return ok, (("PASS" if ok else "FAIL") + f" mamba={P3_MAMBA_EXPECT} tokens_min={min(tok)} "
                f"tokens_max={max(tok)} rel={rel:.3e} tol={P3_REL_TOL:g} floor={P3_TOKEN_FLOOR} "
                f"n_boots={len(rows)} distinct_tokens={len(set(tok))}")


def p3_verdict_file(path, max_running, ctx):
    try:
        rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    except OSError as exc:
        return False, f"FAIL unreadable {exc!r}"
    return p3_verdict(rows, max_running, ctx)


# E2 first VESSL Job keys (boots 1-9 + 11, homog5): the precedent the (ii) rule must PASS.
P3_E2_JOB1_KEYS = [["48", "2722025"]] * 9 + [["48", "2722022"]]
P3_PROBE_6GROUP_TOKENS = 2721290          # probe_6group_2026-10-05 A1/A2 (6-group A-agn, VESSL)
P3_PROBE_5GROUP_TOKENS = 2722025          # probe_6group_2026-10-05 H5 (homog5) = λ0/KISTI value


def p3_selftest() -> None:
    ok, line = p3_verdict(P3_E2_JOB1_KEYS, 48, 16384)
    assert ok and line.startswith("PASS ") and "rel=1.102e-06" in line, line
    six = [["48", str(P3_PROBE_6GROUP_TOKENS)]] * 48
    assert p3_verdict(six, 48, 16384)[0], "48 identical 6-group boots must PASS"
    cases = {   # every one MUST fail (lesson 53)
        "mamba 47/48": [["48", "2721290"], ["47", "2721290"]],
        "mamba all 47": [["47", "2721290"]] * 3,
        "tokens 0.8%": [["48", "2721290"], ["48", str(round(2721290 * 0.992))]],
        "below floor": [["48", "786431"]] * 2,
        "banner missing": [["48", "2721290"], ["NONE", "NONE"]],
        "tokens missing": [["48", "2721290"], ["48", "NONE"]],
        "no keys": [],
        # documents the config-difference fact: 5-group vs 6-group pool (rel 2.70e-4) is caught
        "5g vs 6g pool": [["48", str(P3_PROBE_6GROUP_TOKENS)], ["48", str(P3_PROBE_5GROUP_TOKENS)]],
    }
    for name, rows in cases.items():
        ok, line = p3_verdict(rows, 48, 16384)
        assert not ok and line.startswith("FAIL "), (name, line)
    assert not p3_verdict(P3_E2_JOB1_KEYS, 47, 16384)[0], "unregistered MAX_RUNNING must fail"
    assert not p3_verdict(P3_E2_JOB1_KEYS, 48, 8192)[0], "unregistered CTX must fail"
    assert p3_verdict([["48", "786432"]], 48, 16384)[0], "floor itself passes"
    assert p3_verdict([["48", "1000000"], ["48", "999900"]], 48, 16384)[0], "rel == 1e-4 passes"
    assert not p3_verdict([["48", "1000000"], ["48", "999899"]], 48, 16384)[0], "rel > 1e-4 fails"


# ------------------------------------------------------------------------------- selftest
def selftest(eng: Path = ENG) -> None:
    import random
    for shape, s in SHAPES.items():
        top, sc = ranked_seeds(shape)
        assert tuple(top) == s["seeds"], (shape, top, s["seeds"])
        for sd in s["seeds"]:
            for n in grid_ns(shape):
                eb = LP.ebar(sd, n)
                assert EBAR_GUARD[0] <= 1.0 / eb <= EBAR_GUARD[1], (shape, sd, n, eb)
        assert s["obs_rate"] in s["rates"]
        us = [parse_row(r) for r in units(shape)]
        names = [u["name"] for u in us]
        assert len(names) == len(set(names)), "duplicate unit names"
        assert us[0]["kind"] == "floor" and us[-1]["kind"] == "floor_tail"
        # every (rate, seed) of the grid is scored exactly once, all with PHASE_EVENTS ON
        sc_cells = scored_cells(shape)
        assert len(sc_cells) == len(s["rates"]) * len(s["seeds"]), len(sc_cells)
        assert all(u["phase_events"] == 1 for u in us if u["kind"] in ("grid", "obs_on", "floor", "floor_tail"))
        assert all(u["phase_events"] == 0 for u in us if u["kind"] in ("obs_off", "bridge"))
        # observer pairs adjacent, ABBA
        ob = [u for u in us if u["block"] == "observer"]
        assert [u["kind"] for u in ob] == ["obs_on", "obs_off", "obs_off", "obs_on",
                                           "obs_on", "obs_off", "obs_off", "obs_on"], ob
        # R1: the last unit before floor_tail is the highest rate; rounds 1,3 descend, 2,4 ascend
        grid = [u for u in us if u["kind"] == "grid"]
        assert grid[-1]["rate"] == max(s["rates"]), grid[-1]
        for i in range(4):
            rr = [u["rate"] for u in us if u["block"] == f"round{i + 1}"]
            assert rr == sorted(rr, reverse=(i % 2 == 0)), (i, rr)
        # spacing: consecutive rates <= MAX_SPACING_FRAC x lam_ref (clause 5: ~0.1x around the cliff)
        rs = s["rates"]
        assert all(b > a for a, b in zip(rs, rs[1:])), "rates must be ascending"
        assert all((b - a) <= MAX_SPACING_FRAC * s["lam_ref"] + 1e-9 for a, b in zip(rs, rs[1:])), rs
        # budget: the normal corner admits every unit; the worst corner admits every SCORED/bridge/
        # observer unit (only the diagnostic floor_tail may be refused by admission)
        assert simulate(shape, "normal")["all_admitted"], simulate(shape, "normal")
        w = simulate(shape, "worst")
        if shape == "B":
            # rev7: with the measured constants the all-units-at-worst stress corner no longer admits
            # units-1 under a <= 5 h cap; pinned lower bound = the value computed for the registered cap
            assert w["admitted"] >= REV7_B_WORST_MIN_ADMITTED, w
        else:
            assert w["admitted"] >= w["units"] - 1, w
    cfg = eng / CONFIG_REL
    assert check_config(cfg) == [], check_config(cfg)
    assert stream_groups(cfg) == [(108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (0, 108)]
    # negative controls: FIXED_DSM absent / too many groups are reported
    assert check_config(cfg, fixed_dsm=54), "control: D54 absent must be reported"
    assert check_config(cfg, server_groups=5), "control: 6 groups > 5 must be reported"
    try:
        import yaml  # noqa: F401
        y = yaml.safe_load(cfg.read_text())
        assert y["manual_divisions"] == parse_config(cfg)["manual_divisions"]
        assert y["sm_group_num"] == parse_config(cfg)["sm_group_num"]
    except ImportError:
        pass
    assert verify_digests(eng) == [], verify_digests(eng)
    assert mirror_check(eng) == [], mirror_check(eng)
    # floor formula on a synthetic boot (rev6): valid; then controls on the REALISED evidence
    # (span_end stream, decode inside the span, unpaired end, dropped events, n_new) -> invalid;
    # and the Job-79999aa entry-label race (start label (0,108)/running_bs 1, realised stream 0,
    # no decode inside) -> VALID under rev6 while the rev5 predicate would have rejected it.
    import tempfile
    rnd = random.Random(0)
    with tempfile.TemporaryDirectory() as td:
        bp, tp = Path(td) / "b.jsonl", Path(td) / "t.jsonl"
        tt = [0.9 + 0.01 * rnd.random() for _ in range(FLOOR_N)]
        bp.write_text(json.dumps({"completed": FLOOR_N, "errors": [""] * FLOOR_N, "ttfts": tt,
                                  "total_input_tokens": FLOOR_N * 8192, "max_concurrency": 1,
                                  "duration": 2.0 * (FLOOR_N - 1) + 0.9 + 0.5}) + "\n")   # W-5: span window 30.9 s + 0.5 s decode tail

        def boot(n=FLOOR_N + 9):
            evs = []
            for i in range(n):
                t0 = 100.0 + 2.0 * i
                evs.append({"event": "prefill_span_start", "span_seq": i + 1, "prefill_sms": 108, "decode_sms": 0,
                            "stream_idx": 0, "running_bs": 0, "n_new": 1, "queue_len": 0,
                            "timestamp_monotonic_s": t0, "dropped_events": 0})
                evs.append({"event": "prefill_span_end", "span_seq": i + 1, "stream_idx": 0, "running_bs": 0,
                            "n_done": 1, "queue_len": 0, "timestamp_monotonic_s": t0 + 0.9, "dropped_events": 0})
                for j in range(5):   # the request's decode, after its prefill, on the plain decode group
                    evs.append({"event": "decode_iteration", "iter_seq": 5 * i + j, "stream_idx": 5,
                                "prefill_sms": 0, "decode_sms": 108, "bs": 1,
                                "timestamp_monotonic_s": t0 + 1.0 + 0.1 * j, "dropped_events": 0})
            return evs

        def write(evs):
            tp.write_text("\n".join(json.dumps(x) for x in evs) + "\n")

        def starts(evs):
            return [e for e in evs if e["event"] == "prefill_span_start"]

        def ends(evs):
            return [e for e in evs if e["event"] == "prefill_span_end"]

        evs = boot()
        write(evs)
        f = floor_from(bp, tp, "B")
        assert f["valid"] and abs(f["floor_s"] - statistics.median(tt)) < 1e-12, f
        assert f["entry_label_off_count"] == 0 and f["n_decode_iterations"] == 5 * (FLOOR_N + 9), f
        assert tk_problems(f) == [], tk_problems(f)
        assert tk_problems(dict(f, dryrun=True)), "control: dry-run key must be refused"
        assert tk_problems({k: v for k, v in f.items() if k != "ttfts_s"}), "control: no ttfts_s refused"
        assert abs(f["T_k_s"]["5"] - 5 * f["floor_s"]) < 1e-12
        # control 1: a span REALISED on a division stream (span_end.stream_idx != 0) -> invalid
        evs = boot(); ends(evs)[-3]["stream_idx"] = 4; write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("realised off stream" in p for p in f["problems"]), f
        # control 2: a decode iteration inside a span (decode co-ran with the floor prefill) -> invalid
        evs = boot(); st = starts(evs)[-1]
        evs.append({"event": "decode_iteration", "iter_seq": 9999, "stream_idx": 3, "prefill_sms": 64,
                    "decode_sms": 44, "bs": 1, "timestamp_monotonic_s": st["timestamp_monotonic_s"] + 0.4,
                    "dropped_events": 0})
        write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("decode_iteration inside" in p for p in f["problems"]), f
        # control 3: a span without its prefill_span_end -> invalid (no realised evidence)
        evs = boot(); evs.remove(ends(evs)[-1]); write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("without prefill_span_end" in p for p in f["problems"]), f
        # control 4: dropped events -> invalid (absence of decode_iteration is not evidential)
        evs = boot(); evs[7]["dropped_events"] = 2; write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("dropped_events_max" in p for p in f["problems"]), f
        # control 5: a floor request admitted with another (n_new 2) / a queue -> invalid
        evs = boot(); starts(evs)[-2]["n_new"] = 2; write(evs)
        assert not floor_from(bp, tp, "B")["valid"], "control: n_new != 1 must invalidate"
        evs = boot(); starts(evs)[-1]["queue_len"] = 1; write(evs)
        assert not floor_from(bp, tp, "B")["valid"], "control: queue_len != 0 must invalidate"
        # control 6: fewer than FLOOR_MIN_SPANS spans -> invalid
        evs = boot(FLOOR_N - 1); write(evs)
        assert not floor_from(bp, tp, "B")["valid"], "control: too few spans must invalidate"
        # positive control (the Job-79999aa race): entry label (0,108)/stream 5/running_bs 1 on a
        # warm-up span, realised stream 0, no decode inside -> VALID under rev6; the rev5 predicate
        # (every prefill_span_start at (108,0)) rejects the same boot.
        evs = boot(); st = starts(evs)[3]; st.update(prefill_sms=0, decode_sms=108, stream_idx=5, running_bs=1)
        write(evs)
        f = floor_from(bp, tp, "B")
        assert f["valid"] and f["entry_label_off_count"] == 1 and f["entry_running_bs_nonzero_count"] == 1, f
        assert any((e["prefill_sms"], e["decode_sms"]) != FLOOR_PAIR for e in starts(evs)), "rev5 predicate must fire"
        # the same race on a FLOOR request (last FLOOR_N) is also admissible: entry running_bs is not an input,
        # but it is counted (R-3 descriptive) and the admission gap is reported per floor span
        evs = boot(); starts(evs)[-1].update(prefill_sms=0, decode_sms=108, stream_idx=5, running_bs=1); write(evs)
        f = floor_from(bp, tp, "B")
        assert f["valid"] and f["floor_entry_race_count"] == 1, f
        assert len(f["floor_admission_gap_ms"]) == FLOOR_N and abs(f["floor_admission_gap_ms"][-1] - 600.0) < 1e-6, f
        # control 7 (audit R-2): decode recorder not shown live after a floor span -> invalid
        evs = boot(); st = starts(evs)[-1]
        evs = [e for e in evs if not (e["event"] == "decode_iteration" and e["timestamp_monotonic_s"] > st["timestamp_monotonic_s"])]
        write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("not shown live" in p for p in f["problems"]), f
        evs = [e for e in boot() if e["event"] != "decode_iteration"]; write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and f["n_decode_iterations"] == 0 and any("not shown live" in p for p in f["problems"]), f
        # control 8 (audit W-1): duplicate span_seq among starts / among ends -> invalid
        evs = boot(); starts(evs)[-1]["span_seq"] = starts(evs)[-2]["span_seq"]; write(evs)
        assert not floor_from(bp, tp, "B")["valid"], "control: duplicate start span_seq must invalidate"
        evs = boot(); evs.append(dict(ends(evs)[-1])); write(evs)
        assert not floor_from(bp, tp, "B")["valid"], "control: duplicate end span_seq must invalidate"
        # control 9 (audit W-2): closed-interval boundaries -- a decode_iteration exactly at start / at end -> invalid
        for key in ("start", "end"):
            evs = boot(); ref = (starts(evs) if key == "start" else ends(evs))[-1]
            evs.append({"event": "decode_iteration", "iter_seq": 9998, "stream_idx": 5, "prefill_sms": 0, "decode_sms": 108,
                        "bs": 1, "timestamp_monotonic_s": ref["timestamp_monotonic_s"], "dropped_events": 0})
            write(evs)
            assert not floor_from(bp, tp, "B")["valid"], f"control: decode_iteration at span {key} boundary must invalidate"
        # control 10 (audit W-3): end before start -> invalid (unusable timestamps)
        evs = boot(); ends(evs)[-1]["timestamp_monotonic_s"] = starts(evs)[-1]["timestamp_monotonic_s"] - 0.1; write(evs)
        f = floor_from(bp, tp, "B")
        assert not f["valid"] and any("unusable timestamps" in p for p in f["problems"]), f
    p3_selftest()
    print("N3 PLAN SELFTEST OK (seeds = scorer top-4 per shape, Ebar guard clean; units/ABBA/phase-events "
          "pinned; spacing <= 0.12x lam_ref; normal corner admits every unit, worst (stress corner) pinned per shape, under the registered cap (rev7: B stress corner pinned >= 44/48); config A-agn passes, D54 and "
          "5-group controls fail; registered digests + mirror match; "
          "floor formula (rev6 realised evidence) + 10 invalidating controls (incl. audit R-2 recorder-live, W-1 span_seq, W-2 boundaries, W-3 timestamps) + entry-label race admitted with R-3 descriptives; "
          "P3 (ii): E2 first-Job keys and 48 x 6-group PASS, mamba 47/48 / all-47 / 0.8% / 786431 / banner or "
          "tokens missing / no keys / 5g-vs-6g pool / unregistered constants FAIL, boundaries hold)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--const", metavar="NAME")
    ap.add_argument("--shape-const", nargs=2, metavar=("SHAPE", "KEY"))
    ap.add_argument("--units", metavar="SHAPE")
    ap.add_argument("--need", nargs=2, metavar=("SHAPE", "NAME"))
    ap.add_argument("--budget", metavar="SHAPE")
    ap.add_argument("--verify-digests", metavar="ENG")
    ap.add_argument("--check-config", metavar="CONFIG")
    ap.add_argument("--engine-check", metavar="MANIFEST")
    ap.add_argument("--check-substrate", nargs=2, metavar=("CSV", "SM"))
    ap.add_argument("--check-green", nargs=2, metavar=("GREEN_JSON", "CONFIG"))
    ap.add_argument("--floor", nargs=4, metavar=("SHAPE", "BENCH", "TEL", "OUT"))
    ap.add_argument("--tk-valid", metavar="T_K_JSON")
    ap.add_argument("--p3-verdict", nargs=3, metavar=("P3_KEYS", "MAX_RUNNING", "CTX"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.const:
        v = globals()[a.const]
        print(" ".join(map(str, v)) if isinstance(v, (tuple, list)) else v)
        return 0
    if a.shape_const:
        v = SHAPES[a.shape_const[0]][a.shape_const[1]]
        print(" ".join(map(str, v)) if isinstance(v, (tuple, list)) else v)
        return 0
    if a.units:
        for r in units(a.units):
            print(r)
        return 0
    if a.need:
        print(need_s(*a.need))
        return 0
    if a.budget:
        print(json.dumps(budget_table(a.budget), indent=2, sort_keys=True))
        return 0
    if a.verify_digests:
        bad = verify_digests(Path(a.verify_digests))
        bad += mirror_check(Path(a.verify_digests))
        for b in bad:
            print("DRIFT " + b)
        print(f"DIGESTS {'OK' if not bad else 'DRIFT'} ({len(registered_files(ENG))} files + {len(MIRROR_OF)} mirror entries)")
        return 3 if bad else 0
    if a.check_config:
        bad = check_config(Path(a.check_config))
        for b in bad:
            print("CONFIG " + b)
        print(f"CONFIG {'OK' if not bad else 'BAD'} groups={stream_groups(Path(a.check_config))}")
        return 3 if bad else 0
    if a.engine_check:
        d = engine_check(Path(a.engine_check))
        for x in d:
            print("ENGINE_DIFF " + x)
        print(f"ENGINE_TREE {'== registered HEAD manifest' if not d else 'DIFFERS'} ({len(d)} entries differ)")
        return 3 if d else 0
    if a.check_substrate:
        bad = check_substrate(*a.check_substrate)
        for b in bad:
            print("SUBSTRATE " + b)
        print(f"SUBSTRATE {'OK' if not bad else 'BAD'}")
        return 3 if bad else 0
    if a.check_green:
        bad = check_green(Path(a.check_green[0]), Path(a.check_green[1]))
        for b in bad:
            print("GREEN " + b)
        print(f"GREEN {'OK' if not bad else 'BAD'}")
        return 3 if bad else 0
    if a.tk_valid:
        bad = tk_file_problems(a.tk_valid)
        for b in bad:
            print("T_K " + b)
        return 1 if bad else 0
    if a.p3_verdict:
        ok, line = p3_verdict_file(*a.p3_verdict)
        print(line)
        return 0 if ok else 8
    if a.floor:
        shape, bench, tel, out = a.floor
        res = floor_from(Path(bench), Path(tel), shape)
        Path(out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: res.get(k) for k in ("valid", "floor_s", "T_k_s", "n_spans", "problems")}))
        return 0 if res["valid"] else 3
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
