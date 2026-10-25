#!/usr/bin/env python3
"""Cost-term measurement harness, stage 1 -- REGISTERED VALUES (single source; seal candidate, NOT sealed).

Every number the stage-1 harness uses comes from this file.  Source of each value:
PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md (rev3.4, unsealed) section given in the comment.  Values the PREREG
leaves open are marked HARNESS_READING (a harness choice that the user must confirm before the seal; listed in
AMBIGUITIES below and in the hand-off report).  Nothing here is a measurement.

Stage 1 = T1, T2, T2', T4 and the boots of J-H1 (vanilla boot A), J-A (T1/T2/T2' part, cap-invariance and OFF
diagnostic boots) and J-B2 (T4).  Stage 2 (this revision) = T3 co-residence (J-A T3 boots, J-H2 B/C boots) and J-C
(TP2 + N=1 anchor).  The ONE knob the Workspace smoke may re-set before the seal (decision 29 O-18, H-6) is
RAMP_SIGMA: every registered output length (o_tokens, t3_o_tokens, t3_ref_o_tokens) derives from it.

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

SPEC_VERSION = "costterms-harness-spec/stage2-2026-10-26 (decisions 28-30, 33; rev3.9 decision 34, unsealed)"
PREREG_REL = "workspace/engine-port/results/cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md"
PREREG_REV = "rev3.9 (decision 34, unsealed)"

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
# Audit r1 m9 (A-55): knobs the harness SETS (or leaves at a default) that the PREREG's fixed-knob list does not name.  They are
# identical in every arm of every Job, so they cannot confound a comparison, but they are part of "this stack" and every
# citation scope inherits them.  A test compares them against the argv the runner builds (a new flag must be listed here).
REGISTERED_FLAGS = ("--model-path", "--revision", "--load-format", "--trust-remote-code", "--attention-backend",
                    "--disable-radix-cache", "--mem-fraction-static", "--max-running-requests", "--context-length",
                    "--random-seed", "--chunked-prefill-size", "--max-prefill-tokens", "--host", "--max-mamba-cache-size",
                    "--tp-size", "--enable-pdmux", "--pdmux-config-path")
UNREGISTERED_KNOBS = {
    "--dtype": "bfloat16 (model default; also set by every M3 / N2 harness)",
    "--disable-piecewise-cuda-graph": "piecewise cuda graph off (the decode cuda graph stays ON)",
    "--disable-overlap-schedule": "overlap scheduler off: the engine loop is event_loop_normal-like in every boot",
    "cuda_graph_max_bs": "SGLang default capture list for the cap (default_capture_list), recorded from the boot log",
    "burst_prompts": "the requests of one T1 burst share one seeded prompt id stream per (bs, c, round) -- identical across arms",
    "inter_cell_settle_s": "0.5 s sleep between cells (virtual / wall), not a measured idle",
}

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
# Decision 34 (A-49 add): the OFF diagnostic boots run (16, 16k); the ON D44 and 108v boots get that cell too so the
# instrumentation-effect comparison has an ON pair for every feasible OFF cell (compared cells 4 -> 5 of 6; (64, 8k) stays
# INFEASIBLE_BY_RULE in both).  One window per cell like every J-A cell.
ON_PAIR_CELLS = ((16, 16 * K),)
ON_PAIR_MODES = ("D44", "108v")
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
# T2-ext RESTORED (decision 30, revising decision 28 A-6 within this scope): the long-context points are a cost
# measurement (the gamma quadratic interval of T2, hybrid long-context motive, TP2 long serving), which G-beta's kernel
# cases do not replace (they replace only the boundary-shift test).  They run in SEPARATE boots placed LAST in their Job
# (a server death there cannot touch earlier cells) with context_length LONG_CONTEXT; the other boots keep 73,728.
LONG_CONTEXT = 131072
T2EXT_L = (96 * K, 124 * K)          # 98,304 and 126,976 (decision 30; the G-beta serving L)
LONG_BRIDGE_L = 64 * K               # [제안] same-Job bridge point: the top L of the 73,728-context T2 grid, re-measured in
#                                      the long boot so the across-boot join (cap / context differ) is visible
LONG_L = (LONG_BRIDGE_L,) + T2EXT_L  # run in THIS order (ascending): a death at a larger L keeps the smaller ones
# Audit r1 m2 (A-50): cells of a NON-long boot that finished (VALID) BEFORE a server death.  Default = EXCLUDED from every
# fit / table of usable boots (conservative: a server that later died may already have been degrading); they are listed under
# tables.pre_death with the PRE_DEATH label.  Flipping this is a user decision (PREREG §11-3 A-50).
PRE_DEATH_IN_FIT = False
LONG_REPS = T2_REPS                  # "반복은 T2와 같은 규칙": 8 per (boot, L), the first excluded (T2_WARMUP_REPS)
LONG_MAX_NEW = 1                     # output 1 token (T2 estimand); the serving check needs prompt == L, completion == 1
LONG_CAP = CAP_SMALL                 # [제안, 근거 기록] cap 48: the exact knob set (vanilla, mem 0.6, context 131,072,
#   cap 48, chunked -1, radix off) under which G-beta beta-1 served 98,304 and 126,976 (2/2 each, no OOM / illegal;
#   pdmux-engine-gate-beta-726de92-20261020112000/BETA1_fused) with NVML peak 79,852 MiB of 81,920 -- the activation
#   headroom is the binding resource, and a cap-128 boot captures cuda graphs up to bs 128 (more graph memory, unmeasured
#   at 126,976).  A single request uses one slot, so the cap only moves the KV / mamba split of the static pool.
LONG_MODE_N1 = "108v"                # decision 30: N=1 T2-ext in a vanilla boot (P = 108, decision 28 A-2)
LONG_MODE_TP2 = "108v"               # [제안, A-39] TP2 long boot: vanilla too -- T2 is defined only at P = 108 (108u / 108v);
#   a sticky-D44 label on a no-decode prefill would be mutant M24's defect; 108v extends J-C's TP2 108v T2 grid
LONG_MODE_TP2_PDMUX = "108u"         # decision 33 (L-8 follow-up, A-46): the PD-mux engine path (non-partitioned) at TP2 > 64k, a
#   SECOND TP2 long boot placed after the 108v one -- same L, same rules, same cap / context (the only change is the engine)
# Decision 33: long boots are a CONTIGUOUS SUFFIX of a Job (A-47) and their number is fixed per Job by this registry:
#   key -> ((boot_id, model, dummy, engine, tp), ...) in plan order.  Plan builders read it; registry_problems() checks a
#   full plan against it (a silently dropped long boot is a plan error, not a smaller Job).
LONG_BOOTS = {
    "J-A|1": (("ja1_long_108v", "nano", False, "vanilla", 1),),                      # decision 30 (L-1)
    "J-H1": (("jh1_long_llama", "llama", False, "vanilla", 1),                      # decision 33 (L-8): Llama + real a4 in
             ("jh1_long_real_a4", "nano", False, "vanilla", 1)),                     #   the SAME Job (PREREG §4 (4) contrast)
    "J-C": (("jc_tp2_long_108v", "nano", False, "vanilla", 2),                      # decision 30 (L-2)
            ("jc_tp2_long_108u", "nano", False, "pdmux_108u", 2)),                  # decision 33 (TP2 PD-mux path)
}
# Decision 33 (L-7, alternative (a)): the J-A T1 boots (the 7 mode boots, the cap-invariance boot and the OFF diagnostic
# boots; NOT the T3 boots, which have no T1 cell) boot with context_length 131,072 so (8, 64k) (c_in + O = 79,459 >
# 73,728) becomes feasible.  Cost: req_to_token pool ~ cap x context x 4 B grows, the KV pool shrinks by ~0.2 % [유도].
JA_CONTEXT = LONG_CONTEXT
REQ_TO_TOKEN_BYTES = 4               # [유도] int32 per (slot, token) entry of SGLang's req_to_token pool


def pool_with_context(pool_tokens, cap, context, k_bytes=None):
    """KV token pool of a boot whose context_length is `context` instead of CONTEXT_LENGTH [유도, design arithmetic]:
    the larger req_to_token table (cap x delta-context x 4 B) comes out of the same static pool.  The boot log's KV banner
    replaces this at run time; it is used by the time model and by the pre-data feasibility tables only."""
    k = PREREG_K_BYTES if k_bytes is None else k_bytes
    return int(pool_tokens - math.ceil(cap * (context - CONTEXT_LENGTH) * REQ_TO_TOKEN_BYTES / k))


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
    if mode in ON_PAIR_MODES:
        cells += [c for c in ON_PAIR_CELLS if c not in cells]
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


def variant_arch(n_attn=4):
    """analytic_terms Arch of a composition variant (same skeleton, n_attn attention / VARIANT_MAMBA[n] Mamba layers)."""
    return _AT.nano9b_v2(n_attn, VARIANT_MAMBA[n_attn])


def _bound_terms(arch):
    a = arch or _AT.nano9b_v2()
    dc = _AT.decode_coefficients(a, HBM_BYTES_PER_S)
    pf = _AT.prefill_flops(a, 1)
    return dc, pf["linear_per_token"], pf["quad_coeff"]


def span_ms_ub(L_tokens, p, arch=None, sigma=None, theta=None, n=1):
    """Upper bound on one split-prefill span of n requests of L tokens on p prefill SM [유도]: CM-2 floor + analytic
    attention quadratic term, (108/p)^theta SM scaling, x RAMP_SIGMA (the single smoke-adjustable knob, H-6)."""
    sigma = RAMP_SIGMA if sigma is None else sigma
    theta = RAMP_THETA if theta is None else theta
    _, alpha, gam = _bound_terms(arch)
    return (CM2_FIXED_MS + CM2_SLOPE_MS * n * L_tokens * (1.0 + gam * L_tokens / alpha)) * (TOTAL_SM / p) ** theta * sigma


def step_ms_lb(bs_run, ctx, arch=None, tp=1):
    """HBM-roofline lower bound on one decode step [유도]; with TP the per-rank traffic is 1/tp of it."""
    dc, _, _ = _bound_terms(arch)
    return (dc["a0_s"] + dc["b0_s_per_req"] * bs_run + dc["c0_s_per_req_token"] * bs_run * ctx) * 1000.0 / tp


def ramp_steps_ub(bs, c, p_min, arch=None, sigma=None, theta=None, tp=1):
    """Upper bound on decode steps before bs_real reaches bs (PD-mux burst) [유도, 측정 아님].  tp > 1 (J-C TP2): the
    step lower bound is the per-rank roofline (1/tp); the span bound stays the N=1 CM-2 floor (assumed not faster than
    TP2 prefill -- a violation shows as RAMP_EXCEEDED_BOUND)."""
    c_in = c - X_TOKENS
    batches = span_batches(bs, c_in)
    steps, done = 0, batches[0]
    for n in batches[1:]:
        t_span = span_ms_ub(c_in, p_min, arch, sigma, theta, n=n)
        steps += math.ceil(t_span / step_ms_lb(done, c_in, arch, tp)) + RAMP_INTERSPAN_STEPS
        done += n
    return steps


def o_tokens(bs, c, family):
    """Registered max_new_tokens of a T1 burst cell.  family 'JA' (PD-mux ladder, uniform across its 7 modes),
    'VANILLA' (every vanilla boot outside J-A), 'JH2|<n_attn>|<D>' (J-H2 PD-mux boot: variant priors, P = 108 - D),
    'JC|<D>' (J-C PD-mux: uniform over TP2 and the N=1 anchor = the larger, i.e. the TP2 per-rank step bound)."""
    if family == "JA":
        return O_BASE + ramp_steps_ub(bs, c, ja_pmin(bs, c))
    if family == "VANILLA":
        return O_BASE + VANILLA_RAMP_STEPS
    parts = family.split("|")
    if parts[0] == "JH2":
        n, d = int(parts[1]), int(parts[2])
        return O_BASE + ramp_steps_ub(bs, c, TOTAL_SM - d, variant_arch(n))
    if parts[0] == "JC":
        d = int(parts[1])
        return O_BASE + max(ramp_steps_ub(bs, c, TOTAL_SM - d, tp=t) for t in (1, JC_TP))
    if parts[0] == "JC1":          # decision 33 L-14: the cap-32 N=1 anchor's own O (N=1 step bound only, no TP2 maximum)
        return O_BASE + ramp_steps_ub(bs, c, TOTAL_SM - int(parts[1]), tp=1)
    raise ValueError(family)


# ---------------------------------------------------------------- T3 (§1-3, rev3.5 draft; J-A T3 boots and J-H2)
T3_D = (24, 44)                     # sticky D of the T3 boots (TEL ON, §2-2')
T3_BS = (8, 32)
T3_L = (8 * K, 32 * K)
T3_C = 8 * K                        # decode-batch ctx [제안] (c_in = T3_C - X_TOKENS)
T3_M = {8 * K: 8, 32 * K: 3}        # m(L) [제안, 값 봉인]: prefill inputs per cell, sent ONE AT A TIME (A-28)
T3_REPS = 2                         # per boot (= per round); J-A: x 2 rounds = 4 per cell
T3_MIN_ITERS = 128                  # n_it_obs below this -> T3_WINDOW_SHORT
T3_SOLO_PRE = 150                   # A-27: solo window = 150 intervals before the first prefill + 150 after the last
T3_SOLO_POST = 150                  #       span (ABA; the ctx / clock drift of the co phase cancels to first order)
T3_POST_TRIM = 2                    # records skipped after the last span end before the post solo window
T3_PRE_TOKENS = 1 + TRIM_ITERS + T3_SOLO_PRE + 2       # 169: client waits for this before the first prefill
T3_POST_TOKENS = T3_POST_TRIM + T3_SOLO_POST + 2       # 154: tokens after the last prefill's first token
STALL_MULT = 5.0                    # §1-3 (b): a stall = an interval > 5 x the SOLO window median (never the co median)
T3_REF_REPS = 8                     # sigma reference spans per (boot, L): 8 sequential prefills, the first excluded
T3_REF_WARMUP = 1
T3_REF_BS = 1                       # reference decode = ONE request at the T3 ctx (prefill on the same P as the cell)
T3_REF_MIN_TOKENS = 16              # the reference decode has produced this many tokens before the first prefill


def t3_o_tokens(bs, L, D, arch=None, tp=1):
    """max_new of the T3 decode batch [유도]: ramp + pre window + m(L) spans + post window + margin.  Every term is an
    upper bound from the same priors as o_tokens (RAMP_SIGMA is the only knob)."""
    p, c_in = TOTAL_SM - D, T3_C - X_TOKENS
    per_span = math.ceil(span_ms_ub(L, p, arch) / step_ms_lb(bs, c_in, arch, tp)) + RAMP_INTERSPAN_STEPS
    return ramp_steps_ub(bs, T3_C, p, arch, tp=tp) + T3_PRE_TOKENS + T3_M[L] * per_span + T3_POST_TOKENS + MARGIN_ITERS


def t3_ref_o_tokens(L, D, arch=None, tp=1):
    p, c_in = TOTAL_SM - D, T3_C - X_TOKENS
    per_span = math.ceil(span_ms_ub(L, p, arch) / step_ms_lb(T3_REF_BS, c_in, arch, tp)) + RAMP_INTERSPAN_STEPS
    return T3_REF_MIN_TOKENS + T3_REF_REPS * per_span + MARGIN_ITERS


def feasible_t3(bs, L, O_dec, kv_tokens, cap, capturable, context=None):
    """§3-1 rule extended to T3: the decode batch's worst-case reservation plus the ONE in-flight prefill (L + 1)."""
    context = CONTEXT_LENGTH if context is None else context
    c_in = T3_C - X_TOKENS
    reasons = []
    if bs > cap:
        reasons.append(f"bs {bs} > cap {cap}")
    if not capturable:
        reasons.append(f"bs {bs} not in capture list")
    need = bs * (c_in + O_dec) + L + 1
    if kv_tokens is None:
        reasons.append("no KV pool banner")
    elif need > FEASIBLE_KV_FRAC * kv_tokens:
        reasons.append(f"bs*(c_in+O)+L+1 = {need} > {FEASIBLE_KV_FRAC} x {kv_tokens}")
    if c_in + O_dec >= context or L + 1 >= context:      # same SGLang rule (input + max_new >= context_len is rejected)
        reasons.append(f"c_in + O = {c_in + O_dec} or L + 1 = {L + 1} >= context_length {context}")
    return reasons


# ---------------------------------------------------------------- J-H2 (§4', rev3.5 draft)
JH2_D = (24, 44)
JH2_T1_CELLS = ((32, 2 * K), (32, 8 * K))
JH2_T1_REPS = 2                     # A-31: "반복 2" read for the T1 cells as well (2 windows per cell per boot)

# ---------------------------------------------------------------- J-C TP2 (§5, rev3.5 draft, O-19)
JC_TP = 2
JC_TP2_CAP = CAP_MAIN               # TP2 boots: cap 128 (§1-1 fixed knob)
JC_ANCHOR_CAP = CAP_SMALL           # N=1 anchor: cap 48 (§5-2; TP2 앵커 cap 48 -- mutant "anchor at 128")
JC_MODES = ("D44", "108v")
# decision 33 (L-14): per round ONE extra N=1 anchor boot at cap 32 (GPU 0 only) for (32, 32k) D44, the cell the cap-48
# anchor pool cannot hold (A-32).  Its O is its OWN N=1 bound (family 'JC1|44'), smaller than the TP2-uniform O, so the cell's
# ACTUAL ctx is reported next to the TP2 one (A-45).  Pool (cap 32) = (PREREG pool - 33 s) / k ~ 1.73M [유도].
JC_ANCHOR32_CAP = 32
JC_ANCHOR32_MODE_D = 44
JC_ANCHOR32_CELLS = ((32, 32 * K),)
JC_T1_CELLS_BS8_32 = ((8, 2 * K), (8, 8 * K), (8, 32 * K), (32, 2 * K), (32, 8 * K), (32, 32 * K))   # rev3.5 grid
# decision 33 (L-15): the bs 48 row (48, 2k), (48, 8k) -- feasible on the cap-48 anchor pool and on the TP2 pool, so there was
# no ground to delete them; (48, 32k) stays unregistered (anchor pool, a (C) loss kept disclosed, PREREG §11-4 L-15, A-43)
JC_BS48_CELLS = ((48, 2 * K), (48, 8 * K))
JC_T1_CELLS = JC_T1_CELLS_BS8_32 + JC_BS48_CELLS
JC_T1_REPS = 2
JC_T2_L = (4 * K, 16 * K, 32 * K, 64 * K)     # T2 in 108v, T2' in D44 (decision 28 A-2 applied to §5-2)
JC_T4_FILLS = ((16 * K, 4096), (64 * K, 8000))
JC_OFF_T1_CELLS = ((8, 2 * K), (32, 2 * K), (32, 8 * K))   # O-14 OFF diagnostic (TP2 D44), 3 cells
JC_ROUNDS = (1, 2)
# O-19 96k point: options 'deleted' (recommended: G-beta already ran the stride-11,328 kernel cases -- pristine view
# 189,000 MATCH / 190,500 CRASH -- and a patched serving 96k point cannot test the boundary) or 'separate_boot' (one
# TP2 boot with context JC_LONG_CONTEXT, the LAST boot, a single T2 of JC_LONG_L).  User decision.
JC_96K_MODE = "included"            # decision 30: default INCLUDED ('deleted' kept only as the disclosed alternative)
JC_LONG_L = LONG_L                  # 96k required, 124k when the rank pool / context admit it (feasible_long, at run time)
JC_LONG_REQUIRED = (96 * K,)
JC_LONG_CONTEXT = LONG_CONTEXT
# F1-F5 (§5-1) predictions [유도, config arithmetic]
JC_F_TOL = 0.02
JC_CONV_DIM_N1 = 12288              # d_inner 10,240 + 2 x 8 x 128
JC_INPROJ_N1 = 22656                # 2 x 10,240 + 2 x 1,024 + 128 heads
JC_D_CONV = 4
JC_STRIDE_TOL = 0.03                # conv_state banner has 2 decimals (~1.5 % at 0.13 GB)


def feasible_cell(bs, c, O, kv_tokens, cap, capturable, context=None):
    """§3-1 (rev3.4): the ignore_eos admission check reserves input + max_new for every request in the worst case
    (no request has generated a token yet), so the timing-free sufficient condition is bs (c_in + O) <= 0.95 KV."""
    context = CONTEXT_LENGTH if context is None else context
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
    if c_in + O >= context:        # SGLang tokenizer_manager rejects input + max_new_tokens >= context_len (audit r1 m5)
        reasons.append(f"c_in + O = {c_in + O} >= context_length")
    return reasons


def feasible_long(L, kv_tokens, context, max_new=LONG_MAX_NEW):
    """Decision 30 long point: one request of L input tokens + max_new output must (a) fit the KV pool of the boot
    (per rank for TP2: every rank holds all L tokens with 1/tp of the KV heads, so the per-rank TOKEN pool is the
    bound) under the §3-1 margin and (b) stay strictly below context_length (SGLang rejects input + max_new >=
    context_len).  -> [] or reasons; a context failure is ABORT_CONTEXT_TOO_SHORT (the request is not sent)."""
    reasons = []
    if kv_tokens is None:
        reasons.append("no KV pool banner")
    elif L + max_new > FEASIBLE_KV_FRAC * kv_tokens:
        reasons.append(f"L + {max_new} = {L + max_new} > {FEASIBLE_KV_FRAC} x {kv_tokens}")
    if context is None or L + max_new >= context:
        reasons.append(f"ABORT_CONTEXT_TOO_SHORT: L + {max_new} = {L + max_new} >= context_length {context}")
    return reasons


# ---------------------------------------------------------------- Job entry (H-2..H-4)
ENGINE_DIGEST_EXPECTED = "99bbef91fe680b4d4b8a8fd0a08b8cc65426518c99baca0b3e18e4fa627f4325"   # pathless-v1, 30 entries
#   (G-alpha / G-beta PASS tree, PREREG §12 item 5); written to ENGINE_DIGEST.txt, checked by costterms_job.sh (rc 13)
SUMMARY_FILES = ("analysis.json", "plan.json", "RUN_DONE", "SHA256SUMS", "RESUME_LOG.json", "SMOKE_UNSEALED",
                 "t3_iterations.json", "SMOKE_ENGINE_DIGEST.json")          # what goes under results/ (raw data stays on the cluster volume, H-3)
SMOKE_MARKER = "SMOKE_UNSEALED"
HARD_CAP_MIN = {"J-H1": 271, "J-A|1": 275, "J-A|2": 253, "J-B2": 78, "J-H2": 126, "J-C": 307}
#   minutes, self-enforced by costterms_job.sh; = ceil(1.2 x ct_timemodel hours) [유도, rev3.7 = decision 33]: J-A r1 3.81 h,
#   J-A r2 3.51 h (L-7 context 131,072: +0.22 h each; decision 34 A-49 ON pair (16, 16k): +0.04 h each), J-H1 3.75 h (L-8 Llama + real-a4 long boots: +0.35 h), J-C 4.26 h on
#   2 GPUs (L-15 bs 48 row +0.50 h, L-14 cap-32 anchors +0.39 h, TP2 108u long boot +0.20 h).  All Jobs <= 6 h incl. the cap.
#   J-B2 stays 78 until the smoke S-B fill time re-derives it (decision 30: O-22 (a), PREREG §12 seal checklist) -- the time
#   model alone says 2.32 h / 168 min.
JOB_GPUS = {"J-H1": 1, "J-A": 1, "J-B2": 1, "J-H2": 1, "J-C": 2}

AMBIGUITIES = {
    "A-1": "RESOLVED decision 28: ramp-aware window (first bs_real == bs record, trim 16, 300 intervals), per-cell O = "
           "O_BASE + ramp bound, abort after the window, fit on the actual window ctx",
    "A-2": "RESOLVED decision 28: T2 only in 108u / 108v (P = 108); sticky-D P axis = T2' over the T2 L grid, "
           "co-residence confirmed from decode_forward",
    "A-3": "O-12 (b) cells: (88, 8k)+(88, 9k) vs (88, 8k)+(80, 9k)",
    "A-4": "new_token_ratio is not in the boot log; recorded as min(SGLANG_INIT_NEW_TOKEN_RATIO(0.7) x "
           "schedule_conservativeness, 1) from server_args; split_forward_token_budget comes from the config file",
    "A-5": "108u needs a PDMUX_R2_FIXED_DSM value although sticky is OFF",
    "A-6": "RESOLVED decision 28: T2-ext deleted -> REVISED decision 30: restored in separate LAST boots (context 131,072)",
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
    "A-27": "T3 solo window = 150 intervals after reach + trim 16 (before the first prefill) + 150 intervals starting "
            "T3_POST_TRIM records after the last span end (ABA); rho = median(co) / median(both solo halves)",
    "A-28": "T3 prefills are sent one at a time (the next on the previous one's first token): FIFO would pack two "
            "8,192-token inputs into one span (max_prefill_tokens 16,384) and §1-3 keeps only n_new = 1, "
            "extend_tokens = L spans; the co window is the union of the qualifying spans (gaps are not co-residence)",
    "A-29": "sigma' reference (rev3.5 draft §1-8): spans of the SAME T3 boot, same L, same prefill_sms, with ONE "
            "decode request at the T3 ctx running (8 sequential prefills, first excluded) -- replaces the rev3.4 "
            "denominator T2'(L, P) (client TTFT of another boot, TEL OFF)",
    "A-30": "co-window membership: a decode_forward whose t_host_start lies in [span_start, span_end) of a qualifying "
            "span; its interval runs to the next record (consecutive seq); n_it_obs = decode_iteration records with "
            "t_launch in [span_start, span_end]",
    "A-31": "J-H2 T1 cells: 2 windows per cell per boot (the '반복 2' of §4' read for T1 as well)",
    "A-32": "J-C O: one O per (cell, mode) for TP2 and the N=1 anchor = the larger bound (TP2 per-rank step); "
            "anchor (32, 32k) D44 then fails §3-1 at the cap-48 pool (O-19 option)",
    "A-33": "TP2 records: rank-0 files (decode_probe.tp0 / tel.tp0 / green.tp0) are the measurement, rank 1 is a "
            "consistency check (record count, bs_real sequence, green read-out); F4 stride = DERIVED from the "
            "per-rank conv_state banner (no engine stride log exists; adding one changes ENGINE_DIGEST)",
    "A-34": "I-1 clock covariates of T3 windows: clk.csv (nvidia-smi wall-clock) aligned by the (time.time, "
            "perf_counter) pair the client records at each cell start; SW power cap = bit 0x4 of the event reasons",
    "A-36": "cell order in a T3 boot: the sigma' reference cells first (rep 0 = the boot's first prefill of that L, "
            "excluded as warm-up), then every other cell in seeded random order",
    "A-37": "T_span_eng includes up to one decode interval of observation lag (prefill_span_end is emitted when the "
            "loop observes completion); recorded per cell as span_obs_lag_bound_rel = LONGEST co interval / median span "
            "(an upper bound per span), not corrected",
    "A-35": "resume (H-4): a boot without DONE keeps its completed cells (one cells.jsonl line = one cell DONE) and the "
            "remaining cells run in a NEW boot directory '<boot>@s<k>' (a different server lifetime is a different "
            "boot; fits and sigma never pair across segments); requires PDMUX_COSTTERMS_RESUME=1 and the same plan",
    "A-38": "decision 30 long boots (J-A1 N=1, J-C TP2): cap 48 (LONG_CAP) -- the knob set under which G-beta served "
            "98,304 / 126,976 at mem 0.6 / context 131,072; a single request needs one slot (cap moves only the "
            "KV / mamba split); the TP2 long boot is exempt from the TP2 cap-128 rule",
    "A-39": "TP2 long boot mode = 108v (T2 needs P = 108, decision 28 A-2; extends J-C's TP2 108v T2 grid); the PD-mux "
            "TP2 path at > 64k is exercised by the decision-33 108u long boot (A-46), not by this one",
    "A-40": "long boots re-measure L = 65,536 as a same-Job bridge to the 73,728-context grid boot of the same mode; "
            "the across-boot T2 extension (grid-only quadratic extrapolated to 96k / 124k, and the joint fit) is "
            "descriptive and labelled T2_LONG_ACROSS_BOOTS (cap 128 vs 48, context 73,728 vs 131,072 differ)",
    "A-41": "long cells run in ascending L, never shuffled, always last in their boot; a server death labels the cell "
            "it happened in SERVER_DEATH and every later L NOT_RUN_AFTER_SERVER_DEATH; nothing earlier is deleted; "
            "feasibility (feasible_long) is re-evaluated from the boot log (pool, logged context_len) before sending",
    "A-42": "decision 33 L-7 (alt. (a)): context_length 131,072 for every J-A boot that carries T1 / T2 / OFF cells (the 7 mode "
            "boots, the cap-invariance boot, the 2 OFF diagnostic boots) so a pair (cap-invariance, ON / OFF) differs in one "
            "knob only; the T3 boots (no T1 cell) stay at 73,728; the 'J-A T1 boot' of the decision is read as this set. "
            "KV pool ~ -0.2 % (req_to_token cap x 57,344 x 4 B ~ +30 MB at cap 128) is disclosed and replaced by the boot "
            "banner at run time; J-A results are 'context 131,072' knob results (grid boots of the other Jobs keep 73,728)",
    "A-43": "decision 33 L-15: bs 48 row (48, 2k), (48, 8k) in TP2 and anchor x {D44, 108v} x 2 rounds x 2 windows; (48, 32k) is "
            "NOT registered even though TP2 could hold it (the pair needs the cap-48 anchor, whose pool cannot): a (C) loss "
            "kept disclosed (PREREG §11-4 L-15)",
    "A-44": "decision 33 L-8: J-H1 long boots jh1_long_llama then jh1_long_real_a4 (vanilla 108v, cap 48, context 131,072, "
            "L = 65,536 / 98,304 / 126,976 ascending, 8 reps, same rules as the Nano long boot); order Llama first (any "
            "order is a choice); the J-H1 composition tests (w_indep, P1-P4) exclude them; the Llama / real-a4 per-L ratio is "
            "descriptive (SAME_JOB_LONG_CONTRAST); Llama loads real weights by id + revision",
    "A-45": "decision 33 L-14: per round one cap-32 N=1 D44 anchor boot (jc{r}_n1c32_D44, GPU 0 only, only (32, 32k), 2 windows) "
            "with its OWN output length O (family JC1|44 = O_BASE + N=1 ramp bound, 15,502) instead of the TP2-uniform 30,592; "
            "it is never the 'anchor' of the cap-48 pairs; the TP2 / cap-32 ratio of that single cell is reported with BOTH "
            "actual ctx values and both O (ratios_t1_anchor32) -- a ctx difference, not a clean paired ratio",
    "A-46": "decision 33 (TP2 PD-mux path): jc_tp2_long_108u = PD-mux engine, sticky OFF (PDMUX_R2_FIXED_DSM 44, value unused "
            "at P = 108), TP2, cap 48, context 131,072, same L / rules as jc_tp2_long_108v, placed AFTER it (a PD-mux death "
            "cannot touch the 108v rows); the per-L 108u / 108v ratio of the two long boots is descriptive "
            "(SAME_JOB_LONG_SIBLING); 'PD-mux serves > 64k' is claimable only for the L whose rows are LONG_SERVED",
    "A-47": "plan check generalised: long boots must form the contiguous LAST block of a Job (a measurement boot after any long "
            "boot is ABORT_PLAN -- M39's intent); the old 'one long boot per Job' cap is replaced by a registry (LONG_BOOTS) "
            "that fixes their number / identity per Job and is checked on full plans (registry_problems)",
    "A-48": "pool arithmetic for the time model / pre-data tables: pool_with_context() = pool - ceil(cap x (context - 73,728) x 4 "
            "B / k); the 4 B per (slot, token) req_to_token entry and the cap-32 pool (PREREG pool - 33 s) / k are [유도]",
    "A-49": "audit r1 m1 + decision 34: OFF diagnostic comparison covers the cells present in BOTH ON and OFF (J-A: 5 of 6 after "
            "the (16, 16k) ON pair was added to D44 and 108v, ON_PAIR_CELLS; (64, 8k) is INFEASIBLE in both); cells_compared / "
            "cells_registered / cells_missing / scope are reported, INSTR_EFFECT_SMALL is never bare; the O of (16, 16k) now uses "
            "the ON mode's P (64)",
    "A-50": "audit r1 m2: dead grid boot -> NOT_RUN_AFTER_SERVER_DEATH row per registered cell; VALID cells before the death "
            "are PRE_DEATH (tables.pre_death) and EXCLUDED from fits (PRE_DEATH_IN_FIT False; inclusion is a USER decision); "
            "resume runs the remaining cells in '<boot>@s<k>' without overwriting finished cells (long boots exempt)",
    "A-51": "audit r1 m4: T2 / long (0 engine requests) and T2' (exactly the one decode request) are INVALID when a decode_forward "
            "with another bs_real lies inside a used rep's [t_send, t_first] (T2_ENGINE_RESIDUAL / T2P_ENGINE_BS_NOT_ONE)",
    "A-52": "audit r1 m5: context checks use the FINAL logged context_len (logged_context); feasibility uses '>=' like SGLang "
            "(input + max_new >= context_len is rejected); the fake engine echoes the argv in ServerArgs and writes the clamp "
            "to the final line only",
    "A-53": "audit r1 m6: INSTR_MATRIX_VIOLATION problem -> label + usable False",
    "A-54": "audit r1 m3 / m7: T3 ramp bound from the boot's variant structure and tp; a resume segment with T3 cells re-runs "
            "the sigma' reference cell of each L; segments are excluded from gamma_j fits and pair only with the same segment",
    "A-55": "audit r1 m9: unregistered knobs (REGISTERED_FLAGS / UNREGISTERED_KNOBS): --dtype, --disable-piecewise-cuda-graph, "
            "--disable-overlap-schedule, default cuda_graph_max_bs, shared burst prompt stream, 0.5 s inter-cell settle",
    "A-56": "audit r1 m8 / m10 / m13: gpu_before / gpu_after snapshots + AFTER_PRIOR_DEATH; clock_pair at every cell start and "
            "RAW/clk/<job>.csv per submission; SMOKE_ENGINE_DIGEST.json in smoke output",
    "A-57": "audit r1 MAJOR-1 / 2: client_secondary_table keyed by (_tkey(boot), bs, c) with the window MEDIAN, agreement per "
            "(model, mode, cap, tp, bs); HARNESS_DIGEST.txt made at the seal (ct_harness_digest.py), checked by "
            "costterms_job.sh when SEALED=1 (ABORT_HARNESS_DIGEST rc 13)",
    "A-58": "audit r1 m11 / m12: dry-run shows hardcap + 30 min setup allowance (<= 360 min principle); raw-data archive step "
            "(sha256sum -c after fetch, local copy, re-verify) is DESIGN only",
    "A-59": "audit r2 N-1: HARNESS_DIGEST v2 = harness files + EXTERNAL files (analytic_terms.py, make_variant.py, engine_digest.py, "
            "job_entry.sh) + spec_sha256 (constants) + plans_sha256 (canonical JSON of every Job's make_plan, i.e. the COMPUTED O); "
            "check treats ERROR as a violation, make refuses ERROR / ABSENT values",
    "A-60": "audit r2 N-2: a vanilla boot with no 'decode probe ENABLED' log is not an INSTR_MATRIX_VIOLATION: ENGINE_PROBE_NOT_COLLECTED, "
            "boot usable, S_eng NOT_COLLECTED -> CLIENT_ONLY_UNVERIFIED rows (the §1-7 fallback is reachable)",
    "A-61": "audit r2 N-3: not_run_after_death rows of a resumed dead boot carry resumed_in + RESUMED_IN for cells a '<boot>@s<k>' segment ran"
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
