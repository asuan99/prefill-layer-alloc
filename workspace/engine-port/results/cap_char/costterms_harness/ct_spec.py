#!/usr/bin/env python3
"""Cost-term measurement harness, stage 1 -- REGISTERED VALUES (single source; seal candidate, NOT sealed).

Every number the stage-1 harness uses comes from this file.  Source of each value:
PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md (rev3.4, unsealed) section given in the comment.  Values the PREREG
leaves open are marked HARNESS_READING (a harness choice that the user must confirm before the seal; listed in
AMBIGUITIES below and in the hand-off report).  Nothing here is a measurement.

Stage 1 = T1, T2, T2', T4 and the boots of J-H1 (vanilla boot A), J-A (T1/T2/T2' part, cap-invariance and OFF
diagnostic boots) and J-B2 (T4).  Stage 2 (T3 co-residence, J-H2, J-C TP2) has only its interface here
(`STAGE2_INTERFACE`): the plan checker already rejects a T3 cell in a non-TEL boot.

`python3 ct_spec.py --json` prints the canonical JSON; `--sha256` its digest (the seal will pin that digest).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "costmodel"))
import analytic_terms as _AT  # noqa: E402  (underscore: kept out of the spec digest)

SPEC_VERSION = "costterms-harness-spec/stage1-draft-2026-10-22b (decision 28)"
PREREG_REL = "workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md"
PREREG_REV = "rev3.4 (unsealed)"

K = 1024
# ---------------------------------------------------------------- substrate / server constants (§0, §1-1, §3-3)
TOTAL_SM = 108
MEM_FRACTION = 0.6                    # decision 13
CONTEXT_LENGTH = 73728                # §1-1 fixed knob
CAP_MAIN = 128                        # §1-1, §3-3 (T1, T2, T3)
CAP_SMALL = 48                        # §1-1 cap-invariance boot, §1-4 T4 boot (i)
MAX_PREFILL_TOKENS = 16384            # §3-3 "명시"
SPLIT_FORWARD_TOKEN_BUDGET = 65536    # §1-6 (config value; recorded per boot, interaction not interpreted here)
ATTENTION_BACKEND = "flashinfer"      # §3-3
SERVER_SEED = 1                       # §2-3 "동일 server seed" -- value HARNESS_READING (engine gate precedent)
HEARTBEAT_ENV = {"PDMUX_HEARTBEAT_S": "0"}   # §3-3

NANO_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
NANO_REV = "dc0661c829b14e5b9246c05cfa89094a0875e052"
LLAMA_ID = "meta-llama/Llama-3.1-8B"
LLAMA_REV = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
ARCH_BY_MODEL = {"nano": "NemotronHForCausalLM", "llama": "LlamaForCausalLM"}

# ---------------------------------------------------------------- cudagraph capture list (§3-4)
CAPTURE_BS_REQUIRED = (1, 8, 32, 48, 64, 96)          # cap-128 boots must capture these, else ABORT_CG_BS


def default_capture_list(max_bs):
    """SGLang v0.5.10 default (server_args.py:1354-1359, non-spec): [1, 2, 4, 8, 12] + range(16, 257, 8), cut at
    cuda_graph_max_bs, max appended if absent.  Used only by the FAKE engine and as a prediction; the harness reads
    the real list from the boot log."""
    base = [1, 2, 4, 8, 12] + list(range(16, 257, 8))
    out = [b for b in base if b <= max_bs]
    if max_bs not in out:
        out.append(max_bs)
    return out


# ---------------------------------------------------------------- T1 grid and window (§1-1, §3-1)
D_GRID = (16, 24, 34, 44, 64)                 # sticky ON PD-mux boots
MODES_108 = ("108u", "108v")                  # 108u = PD-mux engine unpartitioned (sticky OFF), 108v = vanilla
ALL_T1_MODES = tuple(f"D{d}" for d in D_GRID) + MODES_108
BS_GRID = (1, 8, 32, 48, 96)
C_GRID = (2 * K, 8 * K, 32 * K, 64 * K)
C16K = 16 * K
C16K_MODES = ("D24", "D34", "D44", "108u", "108v")   # 16k column only for these
EXTRA_CELLS = ((64, 8 * K), (64, 12 * K))          # rev2 shape-C cells, same 5 modes
EXTRA_MODES = C16K_MODES
# decision 18 closed O-12 as (b): bs=88 row added to J-A1 and J-A2 for the same 5 modes.  The PREREG text names
# (88, 8k) and (88, 9k) and allows (80, 9k) as the replacement if (88, 9k) is INFEASIBLE_BY_RULE.  HARNESS_READING
# A-3: we register (88, 8k) and (88, 9k) and let the feasibility rule label (88, 9k); the (80, 9k) replacement is
# NOT run unless the user decides so (O12_ALT_ENABLED).
O12_CELLS = ((88, 8 * K), (88, 9 * K))
O12_ALT_CELLS = ((80, 9 * K),)
O12_ALT_ENABLED = False

WINDOW_ITERS = 300                  # §1-1: 300 S_eng intervals (301 decode_forward records)
TRIM_ITERS = 16                     # §1-1 (rev3.4): records skipped AFTER the first record with bs_real == bs
MARGIN_ITERS = 16                   # §3-1 "여유 16"
O_BASE = TRIM_ITERS + WINDOW_ITERS + MARGIN_ITERS      # 332 = the rev3.3 O; rev3.4 adds the ramp bound per cell
X_TOKENS = 166                      # §3-1 X = O_BASE/2, c_in = c - X (unchanged; actual ctx is recorded, decision 28)
O_PLUS_X = O_BASE + X_TOKENS        # 498 (rev3.3 rule; rev3.4 feasibility uses the per-cell O, see feasible_cell)
# decision 28 A-1 (rev3.4 §1-1): ramp-aware window.  The client aborts the burst once EVERY request has received
# ABORT_AT_TOKENS tokens: token 1 comes from prefill, decode record r0 (first bs_real == bs) yields token 2, the last
# window record r0 + TRIM + WINDOW yields token TRIM + WINDOW + 2, one more token proves that record was logged.
ABORT_AT_TOKENS = 1 + TRIM_ITERS + WINDOW_ITERS + 2    # 319 (< O_BASE: 13 tokens of client-lag slack)
JOIN_SLACK_S = 0.25                 # [제안] client stream lag allowed when the client-side join count is compared with an
#                                     engine record (a request joins the decode batch before its first token arrives)
WINDOW_EXCL_MAX = 30                # [제안] excluded intervals (bs change / mixed / join unconfirmed) allowed per window
# ramp upper bound (rev3.4 §1-1, decision 28): spans of the burst = FIFO prefill batches that close after the request
# crossing max_prefill_tokens; decode runs during spans 2..m.  Steps <= sum_j ceil(T_span_ub(j) / T_step_lb(j)) +
# RAMP_INTERSPAN_STEPS per span.  T_span_ub = (CM2_FIXED_MS + CM2_SLOPE_MS * sum L_i (1 + gamma L_i / alpha))
# * (108 / P)^RAMP_THETA * RAMP_SIGMA, T_step_lb = HBM roofline a0 + b0 bs_run + c0 bs_run c_in (analytic_terms).
CM2_FIXED_MS = 9.0                  # CM-2 floor (J1, L <= 16k, mem 0.82) [측정, CONFIRMED]
CM2_SLOPE_MS = 0.111                # ms / token, same source
RAMP_THETA = 1.0                    # [제안] prefill time ~ (108/P)^1 (linear SM scaling; T2'/theta is measured later)
RAMP_SIGMA = 1.0                    # [제안] no co-residence slowdown in the bound (T3 sigma unmeasured); sensitivity
#                                     1.25 / 1.5 is reported in the PREREG, exceeding the bound is labelled
RAMP_INTERSPAN_STEPS = 2            # [제안] decode steps allowed between two spans
VANILLA_RAMP_STEPS = 64             # [제안] vanilla (event_loop_normal) prefills first; allowance for staggered arrival
PDMUX_108U_P = 108 - 44             # 108u: with decode busy prefill uses the FIXED_DSM_108U division (P = 64)
CTX_BAND = 64                       # rev3.4: label CTX_OFF_NOMINAL if |actual mean ctx - c| > 64 (no exclusion)
FEASIBLE_KV_FRAC = 0.95             # §3-1
I4_RESOLVED_MIN = 0.99              # §3-4: resolution < 0.99 -> I4_UNRESOLVED (label, not exclusion)
AGREE_LN = 0.029558802241544403     # ln(1.03), §1-7
SM_TOL = 4                          # §1-8: |realized - requested| <= 4 and realized decode + prefill <= 108
CAP_INV_CELLS = ((8, 2 * K), (32, 2 * K), (8, 8 * K), (32, 8 * K))   # §1-1 cap-invariance boot (D=44, cap 48)
CAP_INV_LN = 0.029558802241544403   # "차 <= 3%" read as |ln ratio| <= ln 1.03 (HARNESS_READING A-12)

# T1 fit acceptance (§1-1, O-9' [제안] sealed as-is)
T1_LOO_MEDIAN_MAX = 0.03
T1_LOO_P90_MAX = 0.08
T1_EXTRAP_MAX = 0.10
T1_EXTRAP_CTX_ORDER = (32 * K, 8 * K)      # drop 32k first, then 8k (HARNESS_READING A-9: fallback order)
T1_EXTRAP_BS_ORDER = (96, 32)
T1_FORMS = ("F1", "F2", "F3")              # F1 a+b bs+c bs ctx; F2 + d bs^2; F3 a + b bs^beta + c bs ctx
F3_BETA_GRID = tuple(round(0.30 + 0.01 * i, 2) for i in range(171))   # 0.30 .. 2.00

# ---------------------------------------------------------------- T2 (§1-2), T2' (§1-8)
T2_L_GRID = (1 * K, 4 * K, 8800, 16 * K, 32 * K, 64 * K)
T2_REPS = 8                  # per (boot, L)
T2_WARMUP_REPS = 1           # the first of the 8 is excluded
T2_LOO_MAX = 0.05
Q_L_HI, Q_L_LO = 64 * K, 8800
Q_THRESHOLD = 1.10
J1_SLOPE_MS_PER_TOKEN = 0.111   # §1-2 J1 floor slope (mem 0.82), compared descriptively only
J1_SLOPE_L_MAX = 16 * K
J1_SLOPE_DEV = 0.05
# T2-ext (96k, 124k) deleted (decision 28 A-6): context 73,728 cannot hold it; int64 correctness is G-beta's.
T2_MODES = ("108u", "108v")     # decision 28 A-2: T2 only where prefill runs on 108 SM (vanilla / unpartitioned)
T2P_L_GRID = T2_L_GRID          # decision 28 A-2: T2' (bs=1 decode co-resident) is the P-axis measurement, same L
T2P_CONFIRM_WINDOW = (0.25, 0.75)   # [제안] co-residence confirmed iff a decode_forward (bs_real >= 1) starts in
#                                     the middle half [t_send + 0.25 TTFT, t_send + 0.75 TTFT] of every used rep (the
#                                     edges hold the queue wait and the step that follows the prefill in vanilla)
T2P_DECODE_CTX = 2 * K          # bs=1 decode request "ctx 2k" -- c_in = 2048 (HARNESS_READING A-7)
T2P_DECODE_MAX_NEW = 60000      # outlives 8 x 64k prefills at P = 44 [유도 ~ 141 s / 8 ms]; aborted afterwards
T2P_DECODE_MIN_TOKENS = 16      # the decode must have produced this many tokens before the first prefill is sent

# ---------------------------------------------------------------- T4 (§1-4)
T4_L = (4 * K, 8800, 16 * K, 64 * K)
T4_R = {4 * K: 4096, 8800: 4096, 16 * K: 4096, 64 * K: 8000}     # sealed fill output length == reservation R
T4_FILL_EXTRA = 8                  # N = cap + 8
T4_CAP_CA = {4 * K: 156, 8800: 114, 16 * K: 80, 64 * K: 27}     # (ii) cap = C_A(L) boots
T4_CAP_CB = {4 * K: 118, 8800: 93, 16 * K: 68, 64 * K: 24}      # (iii) cap = C_B(L, R) boots
T4_ARM_D = 44                      # static D=44 PD-mux, sticky ON
CMEM_HELD_LO, CMEM_HELD_HI = 0.90, 1.10
CMEM_ORIG_OVER = 0.90
C_MEM_DRIFT_TOL = 0.02
# PREREG pre-data table (§1-4), derived from pool / s / k; reproduced by ct_extract.cmem_predictions in tests
PREREG_POOL_BYTES = 33_040_000_000        # the PREREG's derived pool value (bytes)
PREREG_S_BYTES = 143_500_000
PREREG_K_BYTES = 16_379
PREREG_T4_TABLE = {   # L: (C_A, C_B, n_B at cap=C_A, n_A at cap 48, n_B at cap 48)
    4 * K: (156, 118, 79, 48, 48), 8800: (114, 93, 78, 48, 48), 16 * K: (80, 68, 64, 48, 48),
    64 * K: (27, 24, 24, 24, 21)}
LLAMA_T4_L = (4 * K, 16 * K)       # §1-4 Llama fill (J-B1, H contrast); J-H1 Llama boot A fills L 4k only

# ---------------------------------------------------------------- composition sweep J-H1 (§4')
VARIANTS = (4, 13, 22, 31)                      # attention layers; a31 = same-skeleton Transformer
VARIANT_MAMBA = {4: 27, 13: 18, 22: 9, 31: 0}
JH_T1_CELLS = ((8, 2 * K), (8, 8 * K), (8, 32 * K), (32, 2 * K), (32, 8 * K), (32, 32 * K))
JH_COMMON_CELLS = ((8, 2 * K), (8, 8 * K), (8, 32 * K), (32, 2 * K), (32, 8 * K))   # sealed comparison grid
JH_T2_L = (4 * K, 16 * K, 32 * K, 64 * K)
JH_T4_L = (4 * K, 16 * K)
JH_T4_R = 4096
JH_LLAMA_T4_L = (4 * K,)
JH_T1_REPS = 4                                  # "셀당 반복 4" read as 4 windows per cell in boot A (A-10)
W_INDEP_LN = 0.019802627296179712               # ln(1.02)
COMPO_MEM_TOL = 0.02                            # P1
CTX_SLOPE_TOL = 0.25                            # P3: c(n)/c(4) within (n/4) +- 25 %

# ---------------------------------------------------------------- OFF diagnostic boots (§2-2' (2))
OFF_T1_CELLS_JA = ((8, 2 * K), (32, 2 * K), (32, 8 * K), (64, 8 * K), (96, 2 * K), (16, 16 * K))
OFF_T2_L_JA = (4 * K, 16 * K)                   # T2 in the 108v OFF boot, T2' in the D44 OFF boot (A-2)
INSTR_SMALL_LN = 0.014888612493750654           # ln(1.015)

# ---------------------------------------------------------------- instrumentation matrix (§2-2' (1)) [sealed]
INSTR = {   # boot kind -> (I1, I4I6, I5, TEL)
    "T1T2": (True, True, False, False),
    "T3": (True, True, False, True),       # stage 2 only
    "T4": (True, True, True, False),
    "OFF": (True, False, False, False),
}
PROBE_ENV = {"I4I6": ("PDMUX_DECODE_PROBE_PATH",), "I5": ("PDMUX_ADMISSION_PATH",),
             "TEL": ("PDMUX_TELEMETRY_PATH", "PDMUX_PHASE_EVENTS")}

# ---------------------------------------------------------------- warm-up / correctness (§2-2)
WARMUP_N = 8
WARMUP_IN = 2 * K - X_TOKENS        # "같은 형상" read as a 2k-context, 64-token shape (HARNESS_READING)
WARMUP_OUT = 64
GREEDY_IN, GREEDY_OUT = 256, 16
IDLE_FLUSHES = 2                    # "idle 2 x flush" = 2 x the probe window (1.0 s)
PROBE_WINDOW_S = 1.0

# ---------------------------------------------------------------- PD-mux config candidate (§1-1: + D=64 index)
CONFIG_CANDIDATE = "pdmux_costterms_D64_candidate.yml"
STREAM_GROUPS = ((108, 0), (92, 16), (84, 24), (74, 34), (64, 44), (44, 64), (0, 108))
STICKY_IDX = {16: 1, 24: 2, 34: 3, 44: 4, 64: 5}
FIXED_DSM_108U = 44     # 108u is sticky OFF; PDMUX_R2_FIXED_DSM still needs a value (HARNESS_READING A-5)

# ---------------------------------------------------------------- seeds (§2-3)
TOKEN_LO, TOKEN_HI = 1000, 100000


def prompt_seed(term, bs, c, rnd):
    """§2-3: seed = SHA-256("cost|<term>|<bs>|<c>|<round>"), first 8 bytes big-endian (N2 convention)."""
    return int.from_bytes(hashlib.sha256(f"cost|{term}|{bs}|{c}|{rnd}".encode()).digest()[:8], "big")


def prompt_ids(term, bs, c, rnd, n):
    r = random.Random(prompt_seed(term, bs, c, rnd))
    return [r.randrange(TOKEN_LO, TOKEN_HI) for _ in range(int(n))]


def order_seed(label):
    """Boot / cell order seeds (HARNESS_READING A-8: the PREREG says 'registered seed' without a value)."""
    return int.from_bytes(hashlib.sha256(f"cost|order|{label}".encode()).digest()[:8], "big")


# ---------------------------------------------------------------- analytic priors (decision 27 single source)
HBM_BYTES_PER_S = 2.039e12      # the bandwidth ANALYTIC_COST_FORMULAS.md §6 plugs in; analytic_terms takes it as input

# ---------------------------------------------------------------- per-cell output length O (rev3.4 §1-1, §3-1)
def ja_cells_for_mode(mode):
    """Registered J-A T1 cells of one mode (§1-1 grid + 16k column + rev2 extra + O-12)."""
    cells = [(bs, c) for bs in BS_GRID for c in C_GRID]
    if mode in C16K_MODES:
        cells += [(bs, C16K) for bs in BS_GRID]
    if mode in EXTRA_MODES:
        cells += list(EXTRA_CELLS) + list(O12_CELLS) + (list(O12_ALT_CELLS) if O12_ALT_ENABLED else [])
    return cells


def prefill_sm_of_mode(mode):
    """Prefill SM while the decode batch is busy (the ramp condition): sticky D -> 108 - D, 108u -> division of
    FIXED_DSM_108U, 108v -> None (vanilla: no concurrent prefill)."""
    if mode == "108v":
        return None
    if mode == "108u":
        return PDMUX_108U_P
    return TOTAL_SM - int(mode[1:])


def ja_pmin(bs, c):
    """Smallest concurrent prefill SM among the J-A PD-mux modes that register (bs, c); 44 (= 108 - 64) for a cell no
    mode registers (OFF diagnostic (16, 16k)).  The same O is used in every J-A mode incl. 108v (identical requests
    across the D ladder)."""
    ps = [prefill_sm_of_mode(m) for m in ALL_T1_MODES if m != "108v" and (bs, c) in ja_cells_for_mode(m)]
    return min(ps) if ps else TOTAL_SM - max(D_GRID)


def span_batches(bs, c_in, max_prefill=MAX_PREFILL_TOKENS):
    """FIFO prefill batches of a burst: a batch closes after the request that makes it reach max_prefill_tokens
    (schedule_policy budget_state; chunked prefill off)."""
    out, left = [], bs
    while left > 0:
        n, toks = 0, 0
        while left > 0 and (n == 0 or toks < max_prefill):
            n, toks, left = n + 1, toks + c_in, left - 1
        out.append(n)
    return out


def ramp_steps_ub(bs, c, p_min, arch=None, sigma=None, theta=None):
    """Upper bound on decode steps before bs_real reaches bs (PD-mux burst) [유도, 측정 아님]."""
    sigma = RAMP_SIGMA if sigma is None else sigma
    theta = RAMP_THETA if theta is None else theta
    a = arch or _AT.nano9b_v2()
    dc = _AT.decode_coefficients(a, HBM_BYTES_PER_S)
    pf = _AT.prefill_flops(a, 1)
    alpha, gam = pf["linear_per_token"], pf["quad_coeff"]
    c_in = c - X_TOKENS
    batches = span_batches(bs, c_in)
    steps, done = 0, batches[0]
    for n in batches[1:]:
        t_span = (CM2_FIXED_MS + CM2_SLOPE_MS * n * c_in * (1.0 + gam * c_in / alpha)) \
            * (TOTAL_SM / p_min) ** theta * sigma
        t_step = (dc["a0_s"] + dc["b0_s_per_req"] * done + dc["c0_s_per_req_token"] * done * c_in) * 1000.0
        steps += math.ceil(t_span / t_step) + RAMP_INTERSPAN_STEPS
        done += n
    return steps


def o_tokens(bs, c, family):
    """Registered max_new_tokens of a T1 burst cell.  family 'JA' (PD-mux ladder, uniform across its 7 modes),
    'VANILLA' (J-H1: every boot vanilla)."""
    if family == "JA":
        return O_BASE + ramp_steps_ub(bs, c, ja_pmin(bs, c))
    if family == "VANILLA":
        return O_BASE + VANILLA_RAMP_STEPS
    raise ValueError(family)


def feasible_cell(bs, c, O, kv_tokens, cap, capturable):
    """§3-1 (rev3.4): the ignore_eos admission check reserves input + max_new for every request in the worst case
    (no request has generated a token yet), so the timing-free sufficient condition is bs (c_in + O) <= 0.95 KV."""
    c_in = c - X_TOKENS
    reasons = []
    if bs > cap:
        reasons.append(f"bs {bs} > cap {cap}")
    if not capturable:
        reasons.append(f"bs {bs} not in capture list")
    if kv_tokens is None:
        reasons.append("no KV pool banner")
    elif bs * (c_in + O) > FEASIBLE_KV_FRAC * kv_tokens:
        reasons.append(f"bs*(c_in+O) = {bs * (c_in + O)} > {FEASIBLE_KV_FRAC} x {kv_tokens}")
    if c_in + O > CONTEXT_LENGTH:
        reasons.append(f"c_in + O = {c_in + O} > context_length")
    return reasons


STAGE2_INTERFACE = {
    "T3": "boot kind 'T3' (TEL ON), cells kind 't3' -- not implemented in stage 1; plan checker enforces TEL ON",
    "J-H2": "variant x D {24, 44} PD-mux boots with T1 2 cells + T3 4 cells -- plan builder raises NotImplementedError",
    "J-C": "TP2 boots (resourcespec-a100x2) -- plan builder raises NotImplementedError",
}

AMBIGUITIES = {
    "A-1": "RESOLVED decision 28: ramp-aware window (first bs_real == bs record, trim 16, 300 intervals), per-cell O = "
           "O_BASE + ramp bound, abort after the window, fit on the actual window ctx",
    "A-2": "RESOLVED decision 28: T2 only in 108u / 108v (P = 108); sticky-D P axis = T2' over the T2 L grid, "
           "co-residence confirmed from decode_forward",
    "A-3": "O-12 (b) cells: (88, 8k)+(88, 9k) vs (88, 8k)+(80, 9k)",
    "A-4": "new_token_ratio is not in the boot log; recorded as min(SGLANG_INIT_NEW_TOKEN_RATIO(0.7) x "
           "schedule_conservativeness, 1) from server_args; split_forward_token_budget comes from the config file",
    "A-5": "108u needs a PDMUX_R2_FIXED_DSM value although sticky is OFF",
    "A-6": "RESOLVED decision 28: T2-ext deleted",
    "A-7": "T2' decode c_in = 2048 and its max_new",
    "A-8": "boot / cell order seed values",
    "A-9": "extrapolation test: fallback order vs both levels; NA treated as not failing",
    "A-10": "J-H1 'repeat 4' read as 4 windows per T1 cell inside boot A; cell value = median of window medians",
    "A-11": "RESOLVED decision 28: J-H T4 reason shares NOT_COLLECTED accepted (P2 needs only the concurrency)",
    "A-12": "label aggregation: AGREE needs every window of (model, mode, bs); W-INDEP needs every common cell / L; "
            "CAP_INVARIANT reads 3 % as |ln| <= ln 1.03",
    "A-13": "FILL_INCOMPLETE operationalised: the n_real-th earliest first-token time must precede the earliest "
            "completion; n_real = max engine bs_real (I-6) during the fill",
    "A-14": "q needs T(8,800); J-H T2 grid has no 8,800 -> q from the fitted quadratic (Q_FROM_FIT)",
    "A-15": "Q_UNRESOLVED interval = [min, max] of per-boot q (no bootstrap registered)",
    "A-16": "fused_sym T4 boot (O-8) cap and L set: mirrors the cap-48 boot, all four L",
    "A-17": "gamma_j anchored at 108u (PD-mux engine) over the realized D axis",
    "A-18": "'window 300' = 300 S_eng intervals = 301 decode_forward records after first reach + trim 16",
    "A-19": "P1 / P2 name only the HELD label; a miss is reported as held=False without an invented label",
    "A-20": "realized SM = the one-shot startup green read-out (PDMUX_GREEN_READOUT); PREREG says 'first request "
            "before/after'",
    "A-21": "a failed correctness step (warm-up / greedy / CG canary) is recorded as a problem, not an abort",
    "A-22": "CLOCK_SENSITIVE (I-1 per window) is not computed in stage 1: clk.csv is wall-clock, windows are monotonic",
    "A-23": "T2' decode request: ctx drifts upward while the 8 prefills run (2k + generated tokens)",
    "A-24": "co-residence inside the T1 window with TEL OFF: decode_forward mode == mixed, bs_real != bs, or a client "
            "count of joined (first token received) cell requests != bs at the record; foreign requests = any "
            "decode_forward before the cell's first token or bs_real > bs",
    "A-26": "actual ctx = c_in + tokens the CLIENT had received by t_host_start; it can lag the engine by the one "
            "in-flight token (bias <= 1 token, i.e. <= 1 x c in the fitted b)",
    "A-25": "T2' engine confirmation: a decode_forward (bs_real >= 1) starting in the middle half of every used TTFT",
}


def spec_dict():
    g = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    return json.loads(json.dumps(g, sort_keys=True, default=lambda o: list(o) if isinstance(o, (tuple, set)) else str(o)))


def spec_json():
    return json.dumps(spec_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_sha256():
    return hashlib.sha256(spec_json().encode()).hexdigest()


if __name__ == "__main__":
    if "--sha256" in sys.argv:
        print(spec_sha256())
    else:
        print(json.dumps(spec_dict(), sort_keys=True, indent=1, ensure_ascii=False))
