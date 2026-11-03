#!/usr/bin/env python3
"""Registered constants, arms and Job plans of PREREG_STOCK_BASE rev3 (2026-11-03, user decision (C) = (A); rev2.1 was SEALED 2026-11-02;
rev3 is re-sealed by a new HARNESS_DIGEST.txt).  CPU only; starts no server.

rev3 (after the first Job: SG-stock / SG-off / SG-mix LADDER_BOTTOM_FAIL at 5.516 req/s, ITL-axis cliff, ~42 ms stall per arrival):
  * reference capacity (lambda*_ref) = min of the SUSTAINED capacity of REF_ARMS = (VL-stock, PM-ref) only; the three SGLang-default arms
    are excluded from the min (PREREG §3) and get no sustained cell (J1b runs REF_ARMS only).
  * J1-down: SG 3 arms x rungs 5.516 / 1.15^k (k = 1..4) x seeds {201, 202} x 120 requests; pick rule over J1 + J1-down points; a failing
    bottom again -> LADDER_BOTTOM_FAIL_2 (no value).  Feeds Q1 only, never lambda*_ref; independent of J1b (any order).
  * refcell = PM-ref at its burst lambda* rung (7.29491 in the first Job; the identical cell exists there) -- SG-stock has no burst lambda*.
  * pairs with an SGLang-default arm against a reference arm or each other (Q3, Q5, Q8; NOT Q2): sign / BOTH_SATURATED / INVALID labels +
    latency co-metrics only (SIGN_ONLY_QUESTIONS); Q3 gains the latency co-metrics.

Arms (PREREG §2): SG-stock, SG-rep, SG-mix, SG-off, VL-stock, PM-ref, PM-ref34.  Each arm = (engine, tree, image, server argv, env).
  * ONE image for every arm (rev1, user decision 2026-11-02 option b'): PDMUX_VESSL_IMAGE_SBASE = sglang-runtime@1e0d335b (SGLang side
    byte-unchanged) + /opt/vllm-venv (pristine vLLM 0.31.0).  Selected at submit time WITHOUT changing launch.sh:
    PDMUX_VESSL_ENV=scripts/vessl/vessl.sbase.env scripts/vessl/launch.sh ...  Every boot first checks the two pip-freeze sha256 of the image
    (sb_runner.image_env_check); a mismatch refuses the boot and is stamped.
  * SGLang arms boot through `scripts/vessl/engine_tree.py boot --stamp <json> -- <server cmd>` with PDMUX_ENGINE_TREE chosen per arm
    (pristine for SG-stock/SG-rep/SG-mix; patched for SG-off/PM-ref/PM-ref34 and for the probe instrumentation boots -- the decode probe
    exists only in the patched tree, see INSTR_TREE_NOTE).
  * VL-stock = `/opt/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server` defaults + --max-num-seqs 48 + prefix caching OFF + --port,
    env CUDA_HOME / TORCH_CUDA_ARCH_LIST only (no PYTHONPATH, no PDMUX_*), in the same Job and on the same GPU as the SGLang arms.
Jobs (PREREG §7): J1 ladder (+ definition smoke), J1-down (rev3, SG 3 arms below the grid), J1b sustained cells (rev3: REF_ARMS only), J2 steady-state R1/R2 (+ instrumentation boots + the in-load TTFT
offset boot), J3 = W3 only (time model ~254 min > the 4 h rule: NOT split, disclosed -- user decision 2026-11-02).  The W1 change trace (LO<->HI) is NOT used: every arm is a fixed configuration, so there is no controller whose
reaction a load change could show (gate 2 applies to controller comparisons only) -- user decision 2026-11-02, PREREG rev1 §3.
Rates that depend on earlier Jobs come from an ADDENDUM json (written by sb_analyze.py addendum after the earlier Job); plans that need
one refuse with ABORT_NO_ADDENDUM.  Path rule: os.path.abspath, never Path.resolve() (job_entry swaps results/ for a volume symlink).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ENG = HERE.parents[1]
PH_DIR = HERE.parent / "cap_char" / "policy_harness"
if str(PH_DIR) not in sys.path:
    sys.path.insert(0, str(PH_DIR))
import ph_spec as P  # noqa: E402  (sealed S7 policy harness constants: SLO, seeds, W3 cap, ladder(), static_arm(), arm_env())

S = P.S                                         # ct_spec (NANO_ID, NANO_REV, ...)
PREREG = "results/stock_baseline/PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md"
PLAN_SEED = 20261101                            # ordering seed of every plan (boot order, cell order, ABBA seed halves)
ENGINE_TREE_PY = ENG / "scripts" / "vessl" / "engine_tree.py"
VESSL_ENV = ENG / "scripts" / "vessl" / "vessl.env"
ENGINE_DIGEST_FILE = HERE / "ENGINE_DIGEST.txt"
ENGINE_DIGEST_EXPECTED = "2836112bd710a712a26a31fb516a685a25c6d893308b2058bb7b5590deda5ed7"   # 30-entry pathless-v1 (dev_tree_edits #39)
PRISTINE_REF_SHA256 = "1bd84d884e53468ccfd65f38806ea02cd855cc08d6ac18d31f88bac48a1c3dd3"      # env/sglang_v0.5.10_python_tree.source
REQUIRED_PATCHED_MARKERS = ("pdmux_role_is_thread_local", "maybe_create_holb_probe", "maybe_install_chunk_probe")

# ---- SLO / decision margins (PREREG §3, §4) ----------------------------------------------------------------------------
W1_TTFT_MS, ITL_MS, W3_TTFT_K = P.W1_TTFT_MS, P.ITL_MS, P.W3_TTFT_K       # 3000 ms, 60 ms, 3 x floor
MARGIN = math.log(1.03)
CAP_ATTAIN = 0.90
PLACEBO_SD_EQUIV_MAX = 0.0186                   # n=4, t(3): EQUIV possible only if sd(ln) < this (PREREG §4)
ENGINE_PAIR_SD_MAX = 0.02                       # SG vs VL pairs: sd above this -> registered seed-8 rerun flag
CLIENT_DEF_TOL_MS = 5.0                         # §5-0 (a) rev2: per-engine sd of the smoke offsets (no cross-engine rule)
OUT_OF_BAND_ALL_ABOVE, OUT_OF_BAND_BEST_BELOW = 0.90, 0.50   # §3 CELL_OUT_OF_BAND on the steady-state cells R2 (W1) and R3 (W3)
SUBSTRATE_SHIFT_PP = 0.10                       # refcell attainment difference vs Job 1 (>= 10 %p)

# ---- arms (PREREG §2) --------------------------------------------------------------------------------------------------
NANO_ARGS = ["--model-path", S.NANO_ID, "--revision", S.NANO_REV]
# Loading identity (+ the PREREG §2 "common" flags).  Everything else = SGLang v0.5.10 default (chunked prefill 8192, overlap ON, mixed OFF,
# cudagraph ON, attention backend auto, mem fraction auto, random seed default); the backend / mem fraction actually chosen are read from
# the server log and recorded (sb_runner.parse_sglang_log).
SG_COMMON = ["--trust-remote-code", "--dtype", "bfloat16", "--disable-radix-cache", "--disable-piecewise-cuda-graph",
             "--context-length", "20480", "--max-prefill-tokens", "16384", "--max-running-requests", "48", "--host", "127.0.0.1"]
VL_COMMON = ["--trust-remote-code", "--dtype", "bfloat16", "--max-model-len", "20480", "--max-num-seqs", "48",
             "--no-enable-prefix-caching", "--host", "127.0.0.1"]
PM_D = {"PM-ref": 44, "PM-ref34": 34}
ARM_NAMES = ("SG-stock", "SG-rep", "SG-mix", "SG-off", "VL-stock", "PM-ref", "PM-ref34")
SG_ARMS = tuple(a for a in ARM_NAMES if a != "VL-stock")
PRISTINE_ARMS = ("SG-stock", "SG-rep", "SG-mix")
PATCHED_ARMS = ("SG-off", "PM-ref", "PM-ref34")
INSTR_TREE_NOTE = ("decode_probe (PDMUX_PROBE_EXTEND_TIME) lives only in the PATCHED tree; a pristine boot refuses PDMUX_* knobs. The 'SG-stock "
                   "instrumentation boot' of PREREG §5-1 therefore runs SG-stock argv on the patched tree (= SG-off config + probe). Deviation D-1.")

# vLLM placement (user decision 2026-11-02, option b'): a venv inside the one combined public image; every arm shares IMAGE_KEY.
VLLM_PLACEMENT = "same_image_venv"
IMAGE_KEY = "PDMUX_VESSL_IMAGE_SBASE"
IMAGE_SBASE_DIGEST = "sha256:a5f9bf0f742de31f0de3f1cc2bc9fb352eec537dd4c746d9753d6d104087ea2b"
VLLM_VENV = "/opt/vllm-venv"
VLLM_PY = VLLM_VENV + "/bin/python"
VLLM_VENV_FREEZE_SHA256 = "adff2ff0828cf3658f9a7d3ca79aab087a2c523871edf4050fcc10a12285e12e"   # /opt/vllm-venv/pip_freeze.txt (BRINGUP)
SGLANG_FREEZE_SHA256 = "45c39de25df56274e5e541281e04c2d477edaa7d3f5f2d1dcb344e19a0fa2324"      # /opt/pdmux/pip_freeze.txt of sglang-runtime@1e0d335b
VL_ENV = {"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0"}
VL_STAMP_KEYS = ("engine_tree", "python", "vllm_venv_freeze_sha256", "sglang_freeze_sha256", "attention_backend", "cudagraph_mode")
TTFT_OFFSET_SHIFT_MS = 20.0                                   # §5-0 (b), PREREG rev1.1: |in-load offset - smoke offset| > 20 ms only


def image_is_sbase(image_ref):
    """sb_job.sh preflight: the Job's image (PDMUX_IMAGE from launch.sh) must be the combined image pinned by its registered digest."""
    return isinstance(image_ref, str) and image_ref.endswith("@" + IMAGE_SBASE_DIGEST)


def image_digests(path=VESSL_ENV):
    """{'PDMUX_VESSL_IMAGE_SBASE': ...} (+ the two single-engine keys, recorded only) parsed from vessl.env (values only, nothing secret)."""
    out = {}
    try:
        for ln in Path(path).read_text().splitlines():
            for k in ("PDMUX_VESSL_IMAGE", "PDMUX_VESSL_IMAGE_VLLM", IMAGE_KEY):
                if ln.startswith(k + "="):
                    out[k] = ln.split("=", 1)[1].split("#")[0].strip().strip('"')
    except OSError:
        pass
    return out


def pm_argv(D):
    import ph_runner as PR                       # the sealed policy-harness argv (bridge arm: identical bytes to CAP/TUNE/implfix boots)
    # the PD-mux config path is absolute on each machine; the plan stores it relative to engine-port ("{ENG}/...") so the plan sha is
    # machine-independent; sb_runner.resolve() substitutes the real root at boot time
    return [x.replace(str(ENG), "{ENG}") for x in PR.server_argv(P.static_arm(D))]


def arm(name):
    """Static definition of one arm: engine, tree, image key, argv (without --port), engine env (harness labels added by the runner)."""
    if name not in ARM_NAMES:
        raise ValueError(f"unknown arm {name}")
    if name == "VL-stock":
        return {"name": name, "engine": "vllm", "tree": "vllm", "image_key": IMAGE_KEY,
                "cmd": [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server", "--model", S.NANO_ID, "--revision", S.NANO_REV] + VL_COMMON,
                "env": dict(VL_ENV), "api": "openai", "placement": VLLM_PLACEMENT}
    if name in PM_D:
        D = PM_D[name]
        return {"name": name, "engine": "sglang", "tree": "patched", "image_key": IMAGE_KEY, "D": D,
                "cmd": ["-m", "sglang.launch_server"] + pm_argv(D), "env": dict(P.arm_env(P.static_arm(D))), "api": "sglang",
                "sticky": P.STATIC_STICKY, "inherited": "D, cap 48, budget 65536 = KISTI 승계 (PREREG §2)"}
    argv = NANO_ARGS + SG_COMMON + (["--enable-mixed-chunk"] if name == "SG-mix" else [])
    tree = "pristine" if name in PRISTINE_ARMS else "patched"
    return {"name": name, "engine": "sglang", "tree": tree, "image_key": IMAGE_KEY,
            "cmd": ["-m", "sglang.launch_server"] + argv, "env": {}, "api": "sglang"}


def smoke_arm(engine):
    """Definition smoke (§5-0, not a measurement): same arms + the server TTFT histogram (SGLang needs --enable-metrics; vLLM has /metrics)."""
    a = arm("SG-stock" if engine == "sglang" else "VL-stock")
    a = json.loads(json.dumps(a))
    a["name"] = f"SMOKE-{a['name']}"
    if engine == "sglang":
        a["cmd"] = a["cmd"] + ["--enable-metrics"]
    a["role"] = "smoke"
    return a


def instr_arm(kind, bdir="{dir}"):
    """Instrumentation boots (§5, C5: never a goodput arm).  SG = SG-stock argv on the PATCHED tree + decode probe EXTEND timing (D-1);
    PM = PM-ref + the implfix PC instrumentation env; METRICS = SG-stock on the PRISTINE tree + --enable-metrics (§5-0 (b): in-load TTFT
    offset; the server histogram is the only stock observable).  The §5-2 VL profiler window is not implemented (VL_PROFILER_PENDING)."""
    if kind == "SG":
        a = json.loads(json.dumps(arm("SG-stock")))
        a.update(name="INSTR-SG-stock", tree="patched", role="instr", deviation="D-1",
                 env={"PDMUX_DECODE_PROBE_PATH": f"{bdir}/decode_probe.jsonl", "PDMUX_PROBE_EXTEND_TIME": "1"})
        return a
    if kind == "PM":
        a = json.loads(json.dumps(arm("PM-ref")))
        a["env"].update({"PDMUX_TELEMETRY_PATH": f"{bdir}/tel.jsonl", "PDMUX_PHASE_EVENTS": "1", "PDMUX_PHASE_DIAG": "1",
                         "PDMUX_DECODE_PROBE_PATH": f"{bdir}/decode_probe.jsonl"})
        a.update(name="INSTR-PM-ref", role="instr")
        return a
    if kind == "METRICS":
        a = json.loads(json.dumps(arm("SG-stock")))
        a["cmd"] = a["cmd"] + ["--enable-metrics"]
        a.update(name="INSTR-METRICS-SG-stock", role="instr_metrics")
        return a
    raise ValueError(kind)


VL_PROFILER_PENDING = "INSTR-VL-stock (§5-2 torch-profiler window: not implemented in the runner)"


# ---- cells (PREREG §3) -------------------------------------------------------------------------------------------------
LADDER_RMIN, LADDER_FACTOR, LADDER_N = 5.516, 1.15, 10      # CAP1-R grid (k = 0..9)
LADDER_EXT_K = (10, 11, 12, 13)                             # registered single extension on LADDER_TOP_PASS
LADDER_ARMS = ("SG-stock", "SG-mix", "SG-off", "VL-stock", "PM-ref")   # §7 (PM-ref34 inherits CAP1-R D34 lambda* 6.343, no ladder)
# rev3 (user decision 2026-11-03): the reference capacity is the min over these arms' SUSTAINED capacity only; the SGLang-default arms
# (LADDER_BOTTOM_FAIL in the first Job, ITL-axis cliff) are excluded and get no sustained cell.
REF_ARMS = ("VL-stock", "PM-ref")
# rev3: downward ladder for the SGLang-default arms (Q1 only, never part of lambda*_ref); same pick rule over J1 + J1-down points
LADDER_DOWN_ARMS = ("SG-stock", "SG-mix", "SG-off")
LADDER_DOWN_K = (1, 2, 3, 4)                                 # 5.516 / 1.15^k -> 4.796522, 4.170888, 3.62686, 3.153791
LADDER_DOWN_FAIL = "LADDER_BOTTOM_FAIL_2"
Q1_SG_BAND = (2.5, 4.8)                                      # rev3 PREREG §8 row Q1-SG (value from the down ladder; none -> UNSCORABLE)
PM_REF34_LAMBDA_INHERITED = 6.343
CAP_SEEDS, CAP_N = P.CAP_SEEDS, 120                          # (201, 202)
SUSTAIN_S, SUSTAIN_SEED, SUSTAIN_WINDOW_S = 600.0, 201, 120.0
SUSTAIN_FACTORS, SUSTAIN_EXTRA = (1.0, 0.8), 0.5
SUSTAIN_COMPLETION_MIN, SUSTAIN_TTFT_GROWTH_MAX = 0.95, 2.0
SUSTAIN_E2E_MAX_S = 60.0                                     # rev2 (audit r1 A): per ARRIVAL window, share of arrivals (< 600 s) done with E2E <= 60 s
SATURATED_ATTAIN = 0.98                                      # rev2 (audit r1 B): both arms >= this on all 4 seeds -> BOTH_SATURATED
LATENCY_PAIRS = ("Q3", "Q5", "Q8")                           # rev2 (audit r1 B): Q5/Q8; rev3: + Q3 (VL-stock vs SG-stock)
# rev3: questions whose pairs contain an SGLang-default arm placed above its own capacity (no SLO burst capacity >= 5.516 req/s):
# goodput sign / BOTH_SATURATED / INVALID labels + latency co-metrics only, no magnitude (Q2 is deliberately NOT in this set -- user text)
SIGN_ONLY_QUESTIONS = ("Q3", "Q5", "Q8")
LATENCY_METRICS = ("TPOT", "TTFT", "ITL")                    # seed-level ln(TPOT mean), ln(TTFT p50), ln(median request-internal ITL p95)
QUESTIONS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8")
STEADY = {"R1": 0.5, "R2": 0.8}
STEADY_N, EVAL_SEEDS = 200, P.HE0_SEEDS                      # (1, 2, 3, 4)
W3_LADDER = (0.1, 1.5, 6)
W3_LADDER_N, W3_R3_FACTOR, W3_R3_N = P.CAP_CELL_REQUESTS["W3"], 0.8, 96
W3_ARMS = ("SG-stock", "SG-rep", "PM-ref", "VL-stock")      # §3 "arm 4개" -- the set is not named in the PREREG (open item O-3)
# rev3: refcell = PM-ref at its burst lambda* rung (the identical cell exists in the first Job); rev2.1 used SG-stock, which has no value
REFCELL_ARM, REFCELL_N, REFCELL_SEED = "PM-ref", 120, 201
Q1_ARM = "SG-stock"                                           # Q1 = capacity of the untouched SGLang (no longer the refcell arm)
SMOKE_N, SMOKE_SEED, SMOKE_MAX_OUT = 20, 9, 64
INSTR_N, INSTR_SEED = 60, 301
METRICS_CELL, METRICS_N = "R1", 200                         # §5-0 (b): R1 load, 200 requests, seed INSTR_SEED
MEAS_ARMS = SG_ARMS + ("VL-stock",)
# §6 rev2 (audit r1 C): SG-mix firing = server_args enable_mixed_chunk=True AND chunked prefill active (chunked_prefill_size > 0) in the
# server log of the boot; the "Prefill batch with #running-req >= 1" count is only the number of MIXING OPPORTUNITIES (printed with or
# without --enable-mixed-chunk, v0.5.10 scheduler_metrics_mixin.py:389-394) and is reported for SG-mix AND SG-stock (negative control).
MIX_OPPORTUNITY_ARMS = ("SG-mix", "SG-stock")
Q6_MIN_SPANS, Q6_MIN_PER_L, Q6_L_GRID, Q6_L_BAND = 30, 10, (512, 2048), 0.25   # rev2: 8k removed (W1 has ~0 % of its requests there)
# §5-1 synthetic single-request prefill window (user decision 2026-11-02): the SG instrumentation run appends, AFTER its W1 segment has
# finished, one lead-in request + 10 x 512 + 10 x 2048 random-id prompts (output 64, ignore_eos), one every 2 s, so each of them is prefilled
# ALONE (one EXTEND forward with n_reqs = 1).  The lead-in (64 tokens, outside both length bands) exists only so that the first counted prompt
# has a decode before it inside the synthetic window (pacing_ratios drops the first gap of a window).  The window is recorded apart from the
# W1 load window (t_synth_start/end); the instrumentation boot is never a goodput arm, so goodput is unaffected.
SYNTH_PHASE = "SYNTH"
SYNTH_LENGTHS, SYNTH_PER_LENGTH, SYNTH_GAP_S, SYNTH_OUT = (512, 2048), 10, 2.0, 64
SYNTH_LEAD_IN_TOKENS = 64
SYNTH_TOKEN_RANGE = (1000, 32000)                   # random token ids, [lo, hi): inside every Nano / Llama-class vocabulary, above specials
SYNTH_DRAIN_MIN = 0.3                            # tail after the last prompt (64 tokens at bs 1 ~ 1 s) + slack


def synth_spec():
    return {"lengths": list(SYNTH_LENGTHS), "per_length": SYNTH_PER_LENGTH, "gap_s": SYNTH_GAP_S, "output_tokens": SYNTH_OUT,
            "lead_in_tokens": SYNTH_LEAD_IN_TOKENS, "token_range": list(SYNTH_TOKEN_RANGE), "phase": SYNTH_PHASE}


def synth_requests(seed, prefix, spec=None):
    """Deterministic synthetic segment (seed -> length order and token ids).  Request 0 = lead-in; then one prompt every gap_s.  Each request
    carries its own input_ids and their sha256 (input_ids are excluded from the trace hash, the sha is not)."""
    spec = spec or synth_spec()
    rng = _rng("synth", seed)
    order = [L for L in spec["lengths"] for _ in range(spec["per_length"])]
    rng.shuffle(order)
    lo, hi = spec["token_range"]
    out = []
    for k, L in enumerate([spec["lead_in_tokens"]] + order):
        ids = [rng.randrange(lo, hi) for _ in range(L)]
        out.append({"record": "request", "request_id": f"{prefix}-synth-{k:05d}", "phase": spec["phase"], "arrival_s": round(k * spec["gap_s"], 6),
                    "input_tokens": L, "output_tokens": spec["output_tokens"], "ignore_eos": True, "input_ids": ids,
                    "input_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                    "synth_role": "lead_in" if k == 0 else "prompt"})
    return out


def synth_est_min(spec):
    return (len(spec["lengths"]) * spec["per_length"]) * spec["gap_s"] / 60.0 + SYNTH_DRAIN_MIN

# ---- pairs / labels (PREREG §0, §4) ------------------------------------------------------------------------------------
# (question, A, B, label tokens (A, B)); label = A_ABOVE_B / EQUIV_WITHIN_3PCT / A_BELOW_B / UNRESOLVED_INTERVAL
PAIRS = (("Q2", "PM-ref", "SG-stock", ("PM", "SG")), ("Q2", "PM-ref34", "SG-stock", ("PM34", "SG")),
         ("Q3", "VL-stock", "SG-stock", ("VL", "SG")),
         ("Q4", "PM-ref", "VL-stock", ("PM", "VL")), ("Q4", "PM-ref34", "VL-stock", ("PM34", "VL")),
         ("Q5", "SG-mix", "SG-stock", ("MIX", "SG")),
         ("Q8", "SG-off", "SG-stock", ("OFF", "STOCK")))
PLACEBO = ("SG-rep", "SG-stock")
ENGINE_PAIR_ARMS = ("VL-stock",)

# ---- time model (planning only; [유도] PREREG §7 + rate-based run lengths) ----------------------------------------------
BOOT_OH_MIN = 6.0                    # boot ~3 + warm-up 1 + check + slack (pi_spec)
CAP_CELL_MIN = 1.6                   # 120-request ladder cell (PREREG §7)
DRAIN_MIN = 0.6                      # tail of a fixed-rate segment (W1 output ~200 tok x ~30 ms + slack)
W3_DRAIN_MIN = 1.2                   # 1024-token natural tail
SMOKE_RUN_MIN, FLOOR_MIN = 2.0, 1.5
HARD_CAP_FACTOR = 1.25
R3_BUDGET_STATUS = "R3_SKIPPED_BUDGET"                       # rev2 (audit r1 ⑨): J3 skips ALL R3 boots if the time left < their estimate
PLAN_LAMBDA = {"burst": 6.0, "ref": 5.0, "W3": 0.3}          # planning guesses (P-1 band low edge; sustain x0.8); NOT registered values


def stable(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def sha(o):
    return hashlib.sha256(stable(o).encode()).hexdigest()


def _rng(*salt):
    return random.Random(int(hashlib.sha256("|".join(map(str, (PLAN_SEED,) + salt)).encode()).hexdigest()[:12], 16))


def ladder_rates(ext=False, down=False):
    if down:
        return [round(LADDER_RMIN / LADDER_FACTOR ** k, 6) for k in LADDER_DOWN_K]
    if ext:
        return [round(LADDER_RMIN * LADDER_FACTOR ** k, 6) for k in LADDER_EXT_K]
    return P.ladder(LADDER_RMIN, LADDER_FACTOR, LADDER_N)


def w3_ladder_rates():
    return P.ladder(*W3_LADDER)


def need(addendum, *keys):
    if not addendum:
        raise RuntimeError("ABORT_NO_ADDENDUM: this Job needs the addendum of the earlier Job(s) (sb_analyze.py addendum)")
    cur = addendum
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise RuntimeError(f"ABORT_NO_ADDENDUM: missing {'.'.join(keys)}")
        cur = cur[k]
    return cur


# ---- run constructors --------------------------------------------------------------------------------------------------
def cap_run(rid, workload, rate, seed, n, kind="cap", **kw):
    return {"run_id": rid, "kind": kind, "workload": workload, "rate": rate, "seed": seed, "n_requests": n, **kw}


def boot(bid, a, runs, role="meas", **kw):
    return {"boot_id": bid, "arm": a["name"], "arm_def": a, "role": role, "runs": runs, **kw}


def refcell_boot(job, tag, rate, floor=False):
    r = cap_run(f"{job}_REF_{tag}", "W1", rate, REFCELL_SEED, REFCELL_N, kind="refcell", cell="REFCELL", tag=tag)
    return boot(f"{job}_ref_{tag}", arm(REFCELL_ARM), [r], role="refcell", floor=floor)


def abba(job, cell_key, arm_names, runs_for):
    """ABBA boot order: round 1 = random permutation of the arms, round 2 = round 1 reversed; the seeds are split in two halves (random,
    the SAME halves for every arm of the cell, so every pair's seed s sits in the same round).  runs_for(arm_name, seeds, round) -> runs."""
    rng = _rng(job, cell_key)
    seeds = list(EVAL_SEEDS)
    rng.shuffle(seeds)
    halves = (sorted(seeds[:2]), sorted(seeds[2:]))
    order = list(arm_names)
    rng.shuffle(order)
    boots = []
    for rd, names in ((1, order), (2, list(reversed(order)))):
        for name in names:
            runs = runs_for(name, halves[rd - 1], rd)
            rng.shuffle(runs)
            boots.append(boot(f"{job}_{cell_key}_r{rd}_{name}", arm(name), runs, abba_round=rd))
    return boots, halves


# ---- Job plans (PREREG §7) -----------------------------------------------------------------------------------------------
def plan_J1(addendum=None, ext=False):
    """Ladder (all ladder arms in this Job, one boot per arm, cells in random order) + definition smoke first."""
    job = "J1"
    rng = _rng(job)
    rates = ladder_rates(ext)
    smoke = [boot(f"{job}_smoke_{e}", smoke_arm(e), [{"run_id": f"{job}_SMOKE_{e}", "kind": "smoke", "seed": SMOKE_SEED,
                                                       "n_requests": SMOKE_N, "cell": "SMOKE"}], role="smoke") for e in ("sglang", "vllm")]
    order = list(LADDER_ARMS)
    rng.shuffle(order)
    boots = list(smoke)
    for name in order:
        cells = [cap_run(f"{job}_{name}_lad_r{r}_s{s}", "W1", r, s, CAP_N, cell="LADDER") for r in rates for s in CAP_SEEDS]
        rng.shuffle(cells)
        boots.append(boot(f"{job}_lad_{name}", arm(name), cells))
    return {"job": job, "kind": "LADDER", "ladder": rates, "ext": ext, "boots": boots, "pending": []}


def plan_J1_down():
    """rev3: downward ladder of the SGLang-default arms (one boot per arm, cells in random order, no smoke); Q1 only."""
    job = "J1-down"
    rng = _rng(job)
    rates = ladder_rates(down=True)
    order = list(LADDER_DOWN_ARMS)
    rng.shuffle(order)
    boots = []
    for name in order:
        cells = [cap_run(f"{job}_{name}_lad_r{r}_s{s}", "W1", r, s, CAP_N, cell="LADDER_DOWN") for r in rates for s in CAP_SEEDS]
        rng.shuffle(cells)
        boots.append(boot(f"{job}_lad_{name}", arm(name), cells))
    return {"job": job, "kind": "LADDER_DOWN", "ladder": rates, "boots": boots, "pending": []}


def plan_J1b(addendum):
    """Sustained cells: {lambda*_arm, 0.8 lambda*_arm} x 600 s x seed 201 per REFERENCE arm (rev3: VL-stock, PM-ref only; + registered
    0.5 point, decided in-boot); refcell (PM-ref at its burst lambda* rung) at Job start and end.  Needs no J1-down."""
    job = "J1b"
    rng = _rng(job)
    lam = need(addendum, "lambda_star_burst")
    ref_rate = need(addendum, "refcell_rate")
    order = list(REF_ARMS)
    rng.shuffle(order)
    boots = [refcell_boot(job, "start", ref_rate)]
    for name in order:
        l = need(addendum, "lambda_star_burst", name)
        pts = [round(f * l, 6) for f in SUSTAIN_FACTORS]
        rng.shuffle(pts)
        runs = [cap_run(f"{job}_{name}_sus_r{r}", "W1", r, SUSTAIN_SEED, int(round(r * SUSTAIN_S)), kind="sustain", cell="SUSTAIN",
                        factor=round(r / l, 6)) for r in pts]
        boots.append(boot(f"{job}_sus_{name}", arm(name), runs, lambda_burst=l,
                          extra_run=cap_run(f"{job}_{name}_sus_r{round(SUSTAIN_EXTRA * l, 6)}", "W1", round(SUSTAIN_EXTRA * l, 6),
                                            SUSTAIN_SEED, int(round(SUSTAIN_EXTRA * l * SUSTAIN_S)), kind="sustain", cell="SUSTAIN",
                                            factor=SUSTAIN_EXTRA, conditional=True)))
    boots.append(refcell_boot(job, "end", ref_rate))
    return {"job": job, "kind": "SUSTAIN", "lambda_star_burst": lam, "boots": boots, "pending": []}


def plan_J2(addendum):
    """Steady-state R1/R2 x 4 seeds, 7 arms (ABBA), refcells, then the instrumentation boots (separate from every measurement arm, C5):
    probe SG, PC PM, and the §5-0 (b) in-load TTFT offset boot (SG-stock pristine + --enable-metrics, R1 load, 200 requests, seed 301).
    VL-stock needs no extra boot for (b): vLLM's /metrics is on by default, the runner scrapes it around every VL run."""
    job = "J2"
    lam = need(addendum, "lambda_star_ref")
    ref_rate = need(addendum, "refcell_rate")
    names = list(MEAS_ARMS)
    rates = {c: round(f * lam, 6) for c, f in STEADY.items()}

    def runs_for(name, seeds, rd):
        return [cap_run(f"{job}_{name}_{c}_s{s}", "W1", rates[c], s, STEADY_N, kind="steady", cell=c) for c in STEADY for s in seeds]
    meas, halves = abba(job, "STEADY", names, runs_for)
    instr = []
    for k in ("SG", "PM"):
        a = instr_arm(k)
        kw = {"synth": synth_spec()} if k == "SG" else {}        # §5-1: synthetic single-request prefill window, SG probe boot only
        instr.append(boot(f"{job}_instr_{k}", a, [cap_run(f"{job}_INSTR_{k}", "W1", rates["R2"], INSTR_SEED, INSTR_N, kind="instr",
                                                          cell="INSTR", **kw)], role="instr"))
    instr.append(boot(f"{job}_instr_METRICS", instr_arm("METRICS"),
                      [cap_run(f"{job}_INSTR_METRICS", "W1", rates[METRICS_CELL], INSTR_SEED, METRICS_N, kind="instr", cell="INSTR_METRICS")],
                      role="instr_metrics"))
    boots = [refcell_boot(job, "start", ref_rate)] + meas + instr + [refcell_boot(job, "end", ref_rate)]
    return {"job": job, "kind": "STEADY", "lambda_star_ref": lam, "rates": rates, "seed_halves": halves, "boots": boots,
            "pending": [VL_PROFILER_PENDING]}


def plan_J3(addendum):
    """W3 only (rev1): floor (rev3: in the SG-stock W3 ladder boot, before its rungs), SG-stock W3 ladder (6 rungs x 2 seeds), then R3 = 0.8 lambda*_W3 x 96 requests x
    4 seeds x W3 arms (ABBA, rates filled in-Job), refcell end.  The W1 change trace is not applied: every arm is a fixed configuration
    (no controller to react to a load change; gate 2 is limited to controller comparisons) -- user decision 2026-11-02."""
    job = "J3"
    lam = need(addendum, "lambda_star_ref")
    ref_rate = need(addendum, "refcell_rate")
    rng = _rng(job, "W3")
    w3l = [cap_run(f"{job}_SG-stock_W3lad_r{r}_s{s}", "W3", r, s, W3_LADDER_N, cell="W3LADDER") for r in w3_ladder_rates() for s in CAP_SEEDS]
    rng.shuffle(w3l)
    def w3_for(name, seeds, rd):
        return [cap_run(f"{job}_{name}_R3_s{s}", "W3", None, s, W3_R3_N, kind="steady", cell="R3", rate_key="R3") for s in seeds]
    w3, w3halves = abba(job, "R3", list(W3_ARMS), w3_for)
    # rev3: the refcell arm is PM-ref, so the W3 floor moves to the SG-stock W3 ladder boot (floor arm stays SG-stock, as in rev2.1)
    boots = [refcell_boot(job, "start", ref_rate), boot(f"{job}_w3lad_SG-stock", arm("SG-stock"), w3l, role="w3ladder", floor=True)]
    boots += [dict(b, stage_gated="R3") for b in w3] + [refcell_boot(job, "end", ref_rate)]
    return {"job": job, "kind": "W3", "lambda_star_ref": lam, "seed_halves": {"R3": w3halves}, "boots": boots, "pending": []}


JOBS = ("J1", "J1-down", "J1b", "J2", "J3")
PLAN_KINDS = ("LADDER", "LADDER_DOWN", "SUSTAIN", "STEADY", "W3")


def make_plan(job, addendum=None):
    if job == "J1":
        return finish(plan_J1(addendum))
    if job == "J1-ext":
        return finish(dict(plan_J1(addendum, ext=True), job="J1-ext"))
    if job == "J1-down":
        return finish(plan_J1_down())
    if job == "J1b":
        return finish(plan_J1b(addendum))
    if job == "J2":
        return finish(plan_J2(addendum))
    if job == "J3":
        return finish(plan_J3(addendum))
    raise ValueError(f"unknown job {job}")


# ---- time model ----------------------------------------------------------------------------------------------------------
def run_est_min(run, lam=PLAN_LAMBDA):
    k = run["kind"]
    if k == "smoke":
        return SMOKE_RUN_MIN
    if k in ("cap", "refcell") and run["workload"] == "W1":
        return CAP_CELL_MIN
    rate = run.get("rate") or (W3_R3_FACTOR * lam["W3"])
    drain = W3_DRAIN_MIN if run["workload"] == "W3" else DRAIN_MIN
    return run["n_requests"] / rate / 60.0 + drain + (synth_est_min(run["synth"]) if run.get("synth") else 0.0)


def boot_est_min(b):
    m = BOOT_OH_MIN + sum(run_est_min(r) for r in b["runs"]) + (FLOOR_MIN if b.get("floor") else 0.0)
    if b.get("extra_run"):
        m += run_est_min(b["extra_run"])            # worst case: the conditional 0.5 point runs
    return round(m, 1)


def finish(plan):
    plan["est_min"] = round(sum(boot_est_min(b) for b in plan["boots"]), 1)
    plan["hard_cap_min"] = int(-(-plan["est_min"] * HARD_CAP_FACTOR // 5) * 5)
    plan["engine_trees"] = sorted({b["arm_def"]["tree"] for b in plan["boots"]})
    plan["image_key"] = IMAGE_KEY
    plan["vllm_placement"] = VLLM_PLACEMENT
    return plan


CANON = {"lambda_star_burst": {a: 6.0 + 0.1 * i for i, a in enumerate(LADDER_ARMS)}, "lambda_star_ref": 5.0, "refcell_rate": 6.0}


def plans_sha(addendum=CANON):
    return {j: sha(make_plan(j, addendum)) for j in JOBS}


# ---- plan checks (runner refuses a plan with any problem) -----------------------------------------------------------------
def plan_problems(plan):
    out = []
    ids = [r["run_id"] for b in plan["boots"] for r in b["runs"]] + [b["extra_run"]["run_id"] for b in plan["boots"] if b.get("extra_run")]
    if len(ids) != len(set(ids)):
        out.append("duplicate run_id")
    bids = [b["boot_id"] for b in plan["boots"]]
    if len(bids) != len(set(bids)):
        out.append("duplicate boot_id")
    if set(EVAL_SEEDS) & (set(CAP_SEEDS) | {SUSTAIN_SEED, INSTR_SEED, SMOKE_SEED}):
        out.append("evaluation seeds overlap capacity/instrumentation seeds")
    for b in plan["boots"]:
        a = b["arm_def"]
        if a["image_key"] != IMAGE_KEY:
            out.append(f"{b['boot_id']}: image key {a['image_key']} != {IMAGE_KEY} (one image per Job)")
        if a["engine"] == "vllm" and (a["cmd"][0] != VLLM_PY or any(k.startswith("PDMUX_") or k == "PYTHONPATH" for k in a["env"])):
            out.append(f"{b['boot_id']}: vLLM boot must use {VLLM_PY} with no PDMUX_*/PYTHONPATH env")
        if a["engine"] == "sglang":
            if a["name"] in PRISTINE_ARMS + ("SMOKE-SG-stock", "INSTR-METRICS-SG-stock") and a["tree"] != "pristine":
                out.append(f"{b['boot_id']}: {a['name']} must boot the pristine tree")
            if a["tree"] == "pristine" and (a["env"] or any(x in ("--enable-pdmux", "--pdmux-config-path") for x in a["cmd"])):
                out.append(f"{b['boot_id']}: pristine boot with PD-mux env/argv")
            if a["name"] in PM_D and a.get("sticky") is not False:
                out.append(f"{b['boot_id']}: static PD-mux arm must be non-sticky (decision 49)")
        if b["role"] == "meas" and a["env"].get("PDMUX_DECODE_PROBE_PATH"):
            out.append(f"{b['boot_id']}: instrumentation env in a measurement boot (C5)")
        if any(r.get("synth") for r in b["runs"]) and not (b["role"] == "instr" and a["env"].get("PDMUX_PROBE_EXTEND_TIME") == "1"):
            out.append(f"{b['boot_id']}: synthetic prefill window outside the EXTEND-timing instrumentation boot (§5-1)")
    if plan["kind"] not in PLAN_KINDS:
        out.append(f"plan kind {plan['kind']} not in {PLAN_KINDS}")
    meas = [b for b in plan["boots"] if b["role"] == "meas" and plan["kind"] in ("STEADY", "W3")]
    cells = {}
    for b in meas:
        for r in b["runs"]:
            cells.setdefault(r["cell"], {}).setdefault(b["arm"], []).append(r["seed"])
    for c, by in cells.items():
        for a, seeds in by.items():
            if sorted(seeds) != sorted(EVAL_SEEDS):
                out.append(f"cell {c}: arm {a} seeds {sorted(seeds)} != {list(EVAL_SEEDS)}")
        if c != "R3" and "SG-rep" not in by:
            out.append(f"cell {c}: placebo arm SG-rep missing")
    if plan["kind"] == "W3" and any(r.get("workload") != "W3" for b in plan["boots"] if b["role"] in ("meas", "w3ladder") for r in b["runs"]):
        out.append("J3 is W3 only (rev1: no W1 change trace)")
    if plan["kind"] == "LADDER_DOWN":                     # rev3: SG 3 arms x registered down rungs x CAP_SEEDS x CAP_N, nothing else
        if sorted(b["arm"] for b in plan["boots"]) != sorted(LADDER_DOWN_ARMS) or any(b["role"] != "meas" for b in plan["boots"]):
            out.append(f"J1-down boots must be one measurement boot per {LADDER_DOWN_ARMS}")
        want = sorted((r, s) for r in ladder_rates(down=True) for s in CAP_SEEDS)
        for b in plan["boots"]:
            if sorted((r["rate"], r["seed"]) for r in b["runs"]) != want or any(r["n_requests"] != CAP_N or r["cell"] != "LADDER_DOWN"
                                                                                 for r in b["runs"]):
                out.append(f"{b['boot_id']}: J1-down cells != registered down rungs x seeds {list(CAP_SEEDS)} x {CAP_N}")
            if any(r["rate"] >= LADDER_RMIN for r in b["runs"]):
                out.append(f"{b['boot_id']}: J1-down rung at or above the first-Job bottom {LADDER_RMIN}")
    if plan["kind"] == "SUSTAIN":                         # rev3: sustained cells for the reference arms only
        sus = sorted(b["arm"] for b in plan["boots"] if b["role"] == "meas")
        if sus != sorted(REF_ARMS):
            out.append(f"J1b sustained arms {sus} != REF_ARMS {sorted(REF_ARMS)}")
    if plan["kind"] in ("SUSTAIN", "STEADY", "W3"):
        if any(b["arm"] != REFCELL_ARM for b in plan["boots"] if b["role"] == "refcell"):
            out.append(f"refcell boot arm must be {REFCELL_ARM}")
    if plan["kind"] in ("STEADY", "W3"):
        refs = [b for b in plan["boots"] if b["role"] == "refcell"]
        if [b["runs"][0]["tag"] for b in refs] != ["start", "end"] or plan["boots"][0]["role"] != "refcell" or plan["boots"][-1]["role"] != "refcell":
            out.append("refcell must be the first and the last boot")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=JOBS + ("J1-ext",))
    ap.add_argument("--addendum")
    a = ap.parse_args()
    add = json.load(open(a.addendum)) if a.addendum else (CANON if a.job not in ("J1", "J1-down") else None)
    p = make_plan(a.job, add)
    print(json.dumps({"job": p["job"], "boots": len(p["boots"]), "est_min": p["est_min"], "hard_cap_min": p["hard_cap_min"],
                      "pending": p["pending"], "problems": plan_problems(p), "plan_sha": sha(p),
                      "canonical_addendum": a.addendum is None and a.job not in ("J1", "J1-down"),
                      "arms": sorted({b["arm"] for b in p["boots"]}),
                      "rates": sorted({r["rate"] for b in p["boots"] for r in b["runs"] if r.get("rate") is not None}),
                      "refcell_rate": (add or {}).get("refcell_rate")}, indent=1))
