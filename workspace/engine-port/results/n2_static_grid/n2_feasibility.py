"""N2(3-a) design-time feasibility & budget calculator (CPU only, no GPU).

Status: DESIGN CALCULATOR for PREREG_N2_3A rev15 (unsealed, (A') fixed K; rule-seal candidate). Every number it prints is a design
forecast from the coefficients below -- NOT a measurement and NOT citable (N2C-12, N2C-18).

Coefficient DIRECTIONS (rev7, R6-5). One probe yields three coefficient sets, each conservative for
the quantity that uses it. Before the probe all three are identical (design values).
  "rho"    : table 8-6-rho requirement and choose_k  -- theta MAX, tau_alone scale MAX, kappa MAX
             (fast prefill shrinks W_co; large tau_alone lowers the rho bound; large kappa shrinks W_co)
  "env"    : s_hat (burst spread) and N_co,max      -- theta MIN, tau scale MIN, kappa MIN
             (slow prefill / fast co-decode maximise spread and W_co)
  "budget" : Job time tables                       -- theta MIN, tau scale MAX, kappa MAX (longest time)
The four-corner multipliers (PW theta x0.6, DW tau x1.5, kappa_DW = kappa_DN x 1.6/1.3) are applied on
top of every set.

Single tau worst-corner rule:
  DN: tau_alone = base, tau_co = tau_alone * kappa_DN ; DW: tau_alone = base * 1.5, tau_co = tau_alone * kappa_DW
  PN: theta = THETA * p/64 ; PW: theta * 0.6
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass

# ---- registered design coefficients ----------------------------------------------------------
B = (1, 8, 32, 48)
C = (512, 2048, 8192)
D_ON = (16, 24, 34, 44)
EA_D = (16, 44)
EA_B = (8, 32)
EA_C = (2048, 8192)
BINDING_EA_CELLS = ((16, 32, 8192), (44, 32, 8192))   # probe cells must include these (D-9)
PREFILL_SMS = {16: 92, 24: 84, 34: 74, 44: 64, 108: 64}
TAU_D44_ALONE_MS = {1: 16.0, 8: 18.0, 32: 28.0, 48: 36.0}
CTX_FACTOR = {512: 0.9, 2048: 1.0, 8192: 1.25}
D_FACTOR_DESIGN = {16: 1.6, 24: 1.35, 34: 1.15, 44: 1.0, 108: 0.75}
THETA_DESIGN = 4500.0        # tok/s at p = 64 -- SOURCE: none (design assumption, N2C-33); env / budget directions
THETA_RHO_FIXED = 5705.0     # rev14 (A'): registered rho-direction theta (tok/s @ p = 64). ESTIMATOR (rev15 R14-5):
                             # N3 r3 / r4 per-Job MEDIANS of realized-split 16,384-token span throughput (run-first spans
                             # excluded), the LARGER of the two = r4 median 5,705.4 (kept 357 / excluded 30 / dropped 0;
                             # r3 median 5,693.5). Reproduce: n2_probe_reduce.n3_theta_positive_control(<r4 dir>).
                             # Design input, cite-forbidden -- used ONCE by derive_fixed_k() to fix K.
KAPPA_FLOOR = 0.9            # rev14 (A'): lower-side kappa for N_co,max (O_EA must cover a FAST co-located decode)
KAPPA_DN_DESIGN = 1.3
KAPPA_DW_OVER_DN = 1.6 / 1.3
THETA_WORST_MULT = 0.6
TAU_WORST_MULT = 1.5
DIRECTIONS = ("rho", "env", "budget")
COEF = {d: dict(theta=(THETA_RHO_FIXED if d == "rho" else THETA_DESIGN), d_factor=dict(D_FACTOR_DESIGN),
                kappa_dn=KAPPA_DN_DESIGN) for d in DIRECTIONS}

MAX_PREFILL_TOKENS = 16384
CONTEXT_LEN = 16384
KV_TOKENS = 2_721_290
CAP = 48
# rev14 (A'): K_16, K_44 are REGISTRATION-TIME INTEGER CONSTANTS, computed once by derive_fixed_k() (theta_rho =
# THETA_RHO_FIXED, design tau incl. the DW corner, kappa*_worst >= RHO_MAX_REQUIRED, window 128) and pinned here.
# They are the same in ON/OFF boots and every round and are NEVER updated at runtime or after any probe/smoke.
K_CO = {16: 33, 44: 15}
CO_INPUT = 8192
L_LAG = 128                  # rev14: fixed generous allowance (iterations trigger -> first co span); overflow fail-closed
X_CONST = 145
TRIM, WIN = 16, 128
C_WINDOW_MIN = 128           # C value window = FIRST 128 co-located iterations (rev14 (A') truncation)
RHO_MAX_REQUIRED = 3.0       # rev14: registered kappa*_worst requirement for the fixed K (review 2026-10-08 §9)
DELTA_MIN = 8
DELTA_FIXED = 64             # rev14: fixed cancel margin delta (tokens) for every A2 window; overflow fail-closed
DELTA_L_S = 0.0              # descriptive only (smoke); no longer feeds delta
FIX_S = {"nom": 22.48, "worst": 35.5}
BOOT_OVERHEAD_S = {"nom": 170.5, "worst": 229.0}   # Job-measured; the probe never changes it
PREFLIGHT_S = {"nom": 224.0, "worst": 600.0}
COLD_S = 60.0
ROUNDS = 4
CAP_FACTOR = 1.1
THETA_MIN_SPANS = 20
THETA_REQUIRED_P = (64, 92)
THETA_ALLOWED_P = (64, 74, 84, 92)     # R7-6: prefill SMs of the A-agn splits
REQUIRED_PROBE_KEYS = ("theta_spans", "tau_cells", "kappa_cells", "l_lag_iters", "delta_l_s")
# rev13 G-6: the probe's per-cell fixed cost (fix_s) is DESCRIPTIVE ONLY -- the probe's in-process loadgen is not the
# Job runner's invocation, so FIX_S (budget) keeps its registered N3-derived values and is never fed by the probe.
WINDOW_ITERS = 128

CORNERS = [(p, d) for p in ("PN", "PW") for d in ("DN", "DW")]


# ---- primitives ----------------------------------------------------------------------------------
def theta(p, pcorner, dirn="rho"):
    return COEF[dirn]["theta"] * p / 64 * (THETA_WORST_MULT if pcorner == "PW" else 1.0)


def _kb(b):
    return min(TAU_D44_ALONE_MS, key=lambda x: abs(x - b))


def tau_alone(D, b, c, dcorner, dirn="rho"):
    t = TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] * COEF[dirn]["d_factor"][D] / 1000.0
    return t * (TAU_WORST_MULT if dcorner == "DW" else 1.0)


def kappa(dcorner, dirn="rho"):
    k = COEF[dirn]["kappa_dn"]
    return k * KAPPA_DW_OVER_DN if dcorner == "DW" else k


def tau_co(D, b, c, dcorner, dirn="rho"):
    return tau_alone(D, b, c, dcorner, dirn) * kappa(dcorner, dirn)


# ---- spread / input length (env direction) -------------------------------------------------------
def burst_spread(D, b, c_in, pc, dc, dirn="env"):
    if b == 1:
        return 0.0
    k = max(1, MAX_PREFILL_TOKENS // c_in)
    n = math.ceil(b / k)
    tspan = min(b, k) * c_in / theta(PREFILL_SMS[D], pc, dirn)
    c_ref = 2048 if c_in < 4096 else 8192
    return (n - 1) * tspan / tau_co(D, max(1, b // 2), c_ref, dc, dirn)


def s_hat(D, b, c):
    """four-corner envelope with the env-direction coefficients (R6-5)"""
    return max(burst_spread(D, b, c, pc, dc, "env") for pc, dc in CORNERS)


def c_in_of(D, b, c):
    return max(256, int(c - (s_hat(D, b, c) + X_CONST)))


def ealpha_c_in(d, b, c):
    """R5-2: both boots of an E-alpha pair use c_in(d)."""
    return c_in_of(d, b, c)


# ---- co-load window --------------------------------------------------------------------------------
def co_prefill_time(Dsp, pc, dirn="rho"):
    return K_CO[Dsp] * CO_INPUT / theta(PREFILL_SMS[Dsp], pc, dirn)


def w_co(Dsp, b, c, pc, dc, dirn="rho"):
    return co_prefill_time(Dsp, pc, dirn) / tau_co(Dsp, b, c, dc, dirn)


def tau_alone_lower(b, c):
    """rev14 lower-side bound on the alone step: D44 coefficients for every d (a d = 16 decode may be as fast as
    D44), DN corner, design ctx factor -- the FASTEST decode O_EA must still cover."""
    return TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] / 1000.0


def n_co_max(Dsp, b, c):
    """rev14 (A') lower-side guarantee: the LONGEST possible W_co = co-load prefill at the PW (slowest, design) theta
    divided by the FASTEST co-located step tau_alone_lower * KAPPA_FLOOR. O_EA built on this keeps bs == b through the
    co-located window even when co-location barely slows decode (kappa -> 0.9)."""
    return math.ceil(co_prefill_time(Dsp, "PW", "budget") / (tau_alone_lower(b, c) * KAPPA_FLOOR))


def rho_upper(Dsp, b, c, pc, dc_alone, dirn="rho"):
    return co_prefill_time(Dsp, pc, dirn) / (C_WINDOW_MIN * tau_alone(Dsp, b, c, dc_alone, dirn))


def rho_table():
    rows = []
    for d in EA_D:
        for b in EA_B:
            for c in EA_C:
                ub = {(pc, dc): rho_upper(d, b, c, pc, dc) for pc, dc in CORNERS}
                wco = {(pc, dc): w_co(d, b, c, pc, dc) for pc, dc in CORNERS}
                rows.append(dict(cell=(d, b, c), K=K_CO[d], rho_ub=ub, w_co=wco,
                                 rho_ub_worst=min(ub.values()), w_co_min=min(wco.values()),
                                 passes=min(ub.values()) >= RHO_MAX_REQUIRED and min(wco.values()) >= C_WINDOW_MIN))
    return rows


# ---- output lengths ---------------------------------------------------------------------------------
def o_profile(D, b, c):
    return (c + 63 - c_in_of(D, b, c)) + s_hat(D, b, c) + 64


def o_ealpha(d, b, c):
    """O_EA = (c + 71 - c_in) + L_lag + N_co,max + trim + window + delta + s_hat (rev14: L_lag, delta fixed)."""
    return (c + 71 - ealpha_c_in(d, b, c)) + L_LAG + n_co_max(d, b, c) + TRIM + WIN + DELTA_FIXED + s_hat(d, b, c)


def delta_tokens(Dwin, b, c, dc="DN", dirn="budget"):
    """rev14: the cancel margin is the FIXED allowance DELTA_FIXED for every window (no measured Delta-l path)."""
    return DELTA_FIXED


def delta_exec(Dwin, b, c):
    """rev14: execution cancel margin = DELTA_FIXED (kept as a function so the harness plan keeps its field)."""
    return DELTA_FIXED


# ---- time model (budget direction) ----------------------------------------------------------------
def _corner(level):
    return ("PW", "DW") if level == "worst" else ("PN", "DN")


def cell_time_profile(D, b, c, level, cancel):
    pc, dc = _corner(level)
    cin = c_in_of(D, b, c)
    k = max(1, MAX_PREFILL_TOKENS // cin)
    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[D], pc, "budget")
    s = burst_spread(D, b, cin, pc, dc, "budget")
    start = cin + 1 + s / 2
    steps = (c + 63 - start) + 8 if cancel else o_profile(D, b, c)
    return tpre + steps * tau_alone(D, b, c, dc, "budget")


def cell_time_ealpha(Dalone, d, b, c, level, cancel):
    """Dalone = d (ON) or 108 (OFF); length/burst/co-load are those of d (R5-2)."""
    pc, dc = _corner(level)
    cin = ealpha_c_in(d, b, c)
    k = max(1, MAX_PREFILL_TOKENS // cin)
    tpre = math.ceil(b / k) * min(b, k) * cin / theta(PREFILL_SMS[d], pc, "budget")
    s = burst_spread(d, b, cin, pc, dc, "budget")
    start = cin + 1 + s / 2
    pre = (c + 71 - start) + L_LAG
    tco = co_prefill_time(d, pc, "budget")
    wc = tco / tau_co(d, b, c, dc, "budget")
    ta = tau_alone(Dalone, b, c, dc, "budget")
    if cancel:
        return tpre + pre * ta + tco + (TRIM + WIN + delta_tokens(Dalone, b, c, dc)) * ta
    return tpre + pre * ta + tco + max(0.0, o_ealpha(d, b, c) - pre - wc) * ta


@dataclass(frozen=True)
class Boot:
    name: str
    kind: str
    D: int
    cells: str


def boots_profile_job():
    out = [Boot(f"ON{d}", "ON", d, "PROFILE") for d in D_ON]
    out += [Boot("OFF16", "OFF", 16, "PROFILE"), Boot("OFF44", "OFF", 44, "PROFILE"), Boot("OBS44", "OBS", 44, "OBS")]
    return out


def boots_ea_job():
    return [Boot("ON16", "ON", 16, "EALPHA"), Boot("ON44", "ON", 44, "EALPHA"),
            Boot("OFF16", "OFF", 16, "EALPHA"), Boot("OFF44", "OFF", 44, "EALPHA")]


def boot_cost(bt, level, cancel):
    if bt.cells == "OBS":
        cells = [(44, b, 2048) for b in (1, 32)]
        t = sum(cell_time_profile(D, b, c, level, cancel) for D, b, c in cells)
        n = 2
    elif bt.cells == "PROFILE":
        Dm = 108 if bt.kind == "OFF" else bt.D
        t = sum(cell_time_profile(Dm, b, c, level, cancel) for b in B for c in C)
        n = len(B) * len(C)
    else:
        Dalone = 108 if bt.kind == "OFF" else bt.D
        t = sum(cell_time_ealpha(Dalone, bt.D, b, c, level, cancel) for b in EA_B for c in EA_C)
        n = len(EA_B) * len(EA_C)
    return BOOT_OVERHEAD_S[level] + t + n * FIX_S[level], n


def job_budget(boots, cancel, n_firstboot_probes):
    res = {}
    for level in ("nom", "worst"):
        per = {bt.name: boot_cost(bt, level, cancel)[0] for bt in boots}
        section = ROUNDS * sum(per.values()) + COLD_S + 30.0 * n_firstboot_probes
        res[level] = dict(per_boot=per, section=section, container=section + PREFLIGHT_S[level],
                          cells=ROUNDS * sum(boot_cost(bt, level, cancel)[1] for bt in boots))
    res["cap"] = math.ceil(CAP_FACTOR * res["worst"]["section"] / 100.0) * 100
    return res


# ---- limits ------------------------------------------------------------------------------------------
def limits_check():
    """context / KV / slots for ALL registered cells (PROFILE + E-alpha). ok=False => no seal (R6-7)."""
    worst_len, worst_kv, where = 0, 0, None
    for D in D_ON + (108,):
        for b in B:
            for c in C:
                L = c_in_of(D, b, c) + o_profile(D, b, c)
                if L > worst_len:
                    worst_len, where = L, ("PROFILE", D, b, c)
                worst_kv = max(worst_kv, b * L)
    span_checks, slots_ok = {}, True
    for d in EA_D:
        K = K_CO[d]
        span_checks[d] = dict(K=K, n_new_max=2, spans_min=math.ceil(K / 2), spans_max=K, sum_n_new=K)
        for b in EA_B:
            slots_ok &= (b <= 44 and b + 2 <= CAP)
            for c in EA_C:
                L = ealpha_c_in(d, b, c) + o_ealpha(d, b, c)
                if L > worst_len:
                    worst_len, where = L, ("EALPHA", d, b, c)
                worst_kv = max(worst_kv, b * L)
    ok = worst_len <= CONTEXT_LEN and worst_kv <= KV_TOKENS and slots_ok
    return dict(max_len=worst_len, max_len_cell=where, max_kv=worst_kv, span_checks=span_checks, ok=ok)


# ---- D-9 / R5-3 / R6-3: probe -> coefficient mapping ------------------------------------------------
class ProbeRejected(ValueError):
    """value_independent=True only for reasons that do not depend on measured values (R9-1): missing keys,
    missing binding cells / required p, theta n < 20. Only those allow a re-probe (R7-3 as amended by R9-1)."""
    def __init__(self, msg, value_independent=False, code="INPUT_INVALID"):
        super().__init__(msg)
        self.value_independent = value_independent
        self.code = code


def _scalar(x, what, positive=True):
    if isinstance(x, (list, tuple, dict)):
        raise ProbeRejected(f"iteration-level raw input rejected for {what}: give the registered per-cell median")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ProbeRejected(f"non-numeric value for {what}")
    if not math.isfinite(v):
        raise ProbeRejected(f"non-finite value for {what}")
    if positive and v <= 0:
        raise ProbeRejected(f"value <= 0 for {what}")
    return v


def _check_cell(cell, what, d_allowed):
    try:
        D, b, c = cell
    except (TypeError, ValueError):
        raise ProbeRejected(f"{what}: cell key must be (D, b, c)")
    if D not in d_allowed:
        raise ProbeRejected(f"{what} probe cell D={D} not allowed (only {d_allowed})")
    if b not in B or c not in C:
        raise ProbeRejected(f"{what} probe cell (b={b}, c={c}) outside the registered grid")
    return D, b, c


def apply_probe(probe):
    """Registered reduction (no discretion). Input schema:
      theta_spans: {p: [tok/s of each realized-split 16,384-token span at p]}  -- n >= 20 per p, p >= {64, 92}
                   theta_meas(p) := MEDIAN over spans; then cross-p: rho = MAX_p, env/budget = MIN_p of theta_meas(p)*64/p
      tau_cells:   {(D, b, c): {"s_per_median_ms": scalar, "window_iters": int}}  -- D in {16, 44} only;
                   s_per_median = median S-per over the 128-iteration alone window (§6-1);
                   per-cell ratio r = s_per_median / model tau_alone(DN, design);
                   cross-cell within D: rho/budget = MAX_r, env = MIN_r; D24/D34/108 get MAX(s16, s44) (rho/budget)
                   or MIN(s16, s44) (env)
      kappa_cells: {(d, b, c): {"kappa_median": scalar, "window_iters": int, "a_on_window_iters": int}}
                   kappa_median = EQUAL-WEIGHT median over the cell's measurements of kappa_i = median S-per(C_i) /
                   A_on_i(ctx_mid(C_i)) (rev10, rho_on form); cross-cell: rho/budget = MAX, env = MIN ;
                   kappa_DW = kappa_DN*1.6/1.3
      l_lag_iters: [scalars] -> MAX ; delta_l_s: [scalars] -> MAX ; fix_s: descriptive only, NOT mapped (rev13 G-6)
    Required: tau_cells and kappa_cells each include BINDING_EA_CELLS. s_hat / N_co,max are never given -- they are
    recomputed from the env set. Boot overhead is never changed."""
    for key in REQUIRED_PROBE_KEYS:                                   # R7-6: no silent design fallback
        if key not in probe:
            raise ProbeRejected(f"probe key missing: {key}", value_independent=True, code="KEY_MISSING")
    new = copy.deepcopy(COEF)
    # theta
    spans = probe["theta_spans"]
    for p in THETA_REQUIRED_P:
        if p not in spans:
            raise ProbeRejected(f"theta spans missing for p={p}", value_independent=True, code="THETA_P_MISSING")
    med = {}
    for p, lst in spans.items():
        if p not in THETA_ALLOWED_P:
            raise ProbeRejected(f"theta: p={p} not a registered prefill split")
        if not isinstance(lst, (list, tuple)) or len(lst) < THETA_MIN_SPANS:
            raise ProbeRejected(f"theta: need >= {THETA_MIN_SPANS} realized-split spans at p={p} (counted after exclusion)",
                                value_independent=True, code="THETA_N_SHORT")
        med[p] = statistics.median(_scalar(v, "theta") for v in lst) * 64.0 / p
    new["rho"]["theta"] = max(med.values())
    new["env"]["theta"] = new["budget"]["theta"] = min(med.values())
    # tau
    tc = probe["tau_cells"]
    for cell in BINDING_EA_CELLS:
        if cell not in tc:
            raise ProbeRejected(f"tau cells must include binding cell {cell}", value_independent=True,
                                code="TAU_BINDING_MISSING")
    ratios = {16: [], 44: []}
    for cell, v in tc.items():
        D, b, c = _check_cell(cell, "tau", (16, 44))
        if not isinstance(v, dict):
            raise ProbeRejected("tau cell must be a dict with s_per_median_ms / window_iters")
        if int(v.get("window_iters", 0)) < WINDOW_ITERS:
            raise ProbeRejected("tau cell window shorter than 128 iterations")
        ms = _scalar(v["s_per_median_ms"], "tau")
        model = TAU_D44_ALONE_MS[_kb(b)] * CTX_FACTOR[c] * D_FACTOR_DESIGN[D]
        ratios[D].append(ms / model)
    for D in (16, 44):
        if not ratios[D]:
            raise ProbeRejected(f"no tau cell for D={D}")
    hi = {D: max(r) for D, r in ratios.items()}
    lo = {D: min(r) for D, r in ratios.items()}
    for dirn, s in (("rho", hi), ("budget", hi), ("env", lo)):
        other = max(s.values()) if dirn != "env" else min(s.values())
        for D in D_FACTOR_DESIGN:
            new[dirn]["d_factor"][D] = D_FACTOR_DESIGN[D] * s.get(D, other)
    # kappa
    kc = probe["kappa_cells"]
    for cell in BINDING_EA_CELLS:
        if cell not in kc:
            raise ProbeRejected(f"kappa cells must include binding cell {cell}", value_independent=True,
                                code="KAPPA_BINDING_MISSING")
    ks = []
    for cell, v in kc.items():
        _check_cell(cell, "kappa", (16, 44))
        if not isinstance(v, dict) or int(v.get("window_iters", 0)) < WINDOW_ITERS \
                or int(v.get("a_on_window_iters", 0)) < WINDOW_ITERS:
            raise ProbeRejected("kappa cell needs kappa_median / window_iters >= 128 / a_on_window_iters >= 128")
        ks.append(_scalar(v.get("kappa_median"), "kappa"))
    new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)
    new["env"]["kappa_dn"] = min(ks)
    def _nonempty(key):
        lst = probe[key]
        if not isinstance(lst, (list, tuple)) or not lst:
            raise ProbeRejected(f"{key} must be a non-empty list")
        return lst
    lags = [_scalar(x, "l_lag", positive=False) for x in _nonempty("l_lag_iters")]
    if min(lags) < 0:                       # rev10: L_lag is an iteration COUNT from raw data; 0 is a valid observation
        raise ProbeRejected("l_lag < 0")
    # delta-l is a SIGNED difference l_d - l_co: finite required (descriptive)
    [_scalar(x, "delta_l", positive=False) for x in _nonempty("delta_l_s")]
    # rev14: L_LAG and the cancel margin are FIXED allowances -- the smoke's l_lag / delta_l are descriptive only and
    # are reported back in the record, never assigned (a mutant assigning them must fail).
    COEF.clear(); COEF.update(new)
    return dict(l_lag_observed_max=int(math.ceil(max(lags))))


# ---- U-9 rule (R5-4), R6-2 (no artificial search cap), R6-7 (seal gate) ------------------------------
def k_satisfies(d, K):
    old = K_CO[d]
    K_CO[d] = K
    try:
        for b in EA_B:
            for c in EA_C:
                if min(rho_upper(d, b, c, pc, dc) for pc, dc in CORNERS) < RHO_MAX_REQUIRED:
                    return False
                if min(w_co(d, b, c, pc, dc) for pc, dc in CORNERS) < C_WINDOW_MIN:
                    return False
        return True
    finally:
        K_CO[d] = old


def k_within_limits(d, K):
    old = K_CO[d]
    K_CO[d] = K
    try:
        for b in EA_B:
            if b > 44 or b + 2 > CAP:
                return False
            for c in EA_C:
                L = ealpha_c_in(d, b, c) + o_ealpha(d, b, c)
                if L > CONTEXT_LEN or b * L > KV_TOKENS:
                    return False
        return True
    finally:
        K_CO[d] = old


def choose_k(d):
    """smallest K satisfying the requirement; search runs to the limits_check boundary (no other cap).
    E-alpha length grows strictly with K (N_co,max ~ K), so the loop always terminates.
    rev14: NOT a judgment input at runtime -- used ONCE by derive_fixed_k() to fix K_CO at registration, and
    afterwards only as a design diagnostic (post_probe_gate smoke)."""
    K = 1
    while True:
        if not k_within_limits(d, K):
            return None
        if k_satisfies(d, K):
            return K
        K += 1


def derive_fixed_k():
    """rev14 (A'): the ONE registration-time computation of K. Fixed inputs: COEF['rho'] = (THETA_RHO_FIXED, design
    d_factor, design kappa_dn incl. the DW corner via kappa()), requirement RHO_MAX_REQUIRED, window C_WINDOW_MIN,
    limits via k_within_limits (N_co,max lower-side guarantee). Returns {d: K}; K_CO must equal it (test-pinned).
    The result is a DESIGN VALUE (cite-forbidden); it is never recomputed from measurements."""
    saved = copy.deepcopy(COEF["rho"])
    COEF["rho"].update(theta=THETA_RHO_FIXED, d_factor=dict(D_FACTOR_DESIGN), kappa_dn=KAPPA_DN_DESIGN)
    try:
        return {d: choose_k(d) for d in EA_D}
    finally:
        COEF["rho"].clear(); COEF["rho"].update(saved)


def registered_constants():
    """rev14: the SINGLE SOURCE of the judgment-path constants for the harness/analyzer (n2_plan reads this record,
    never K_POST_PROBE.json): K, allowances, requirement, per-cell lengths, kappa* table, limits, budgets."""
    kstar = {f"{r['cell'][0]},{r['cell'][1]},{r['cell'][2]}": dict(worst=r["rho_ub_worst"], w_co_min=r["w_co_min"],
                                                                   corners={f"{pc}{dc}": v for (pc, dc), v in r["rho_ub"].items()})
             for r in rho_table()}
    return dict(K=dict(K_CO), K_derivation=dict(theta_rho=THETA_RHO_FIXED, kappa_star_required=RHO_MAX_REQUIRED,
                                                window=C_WINDOW_MIN, tau="design incl. DW corner",
                                                theta_rho_estimator="larger of the N3 r3/r4 per-Job medians of realized "
                                                                    "16,384-token span throughput (run-first spans excluded) "
                                                                    "= r4 median 5,705.4 (kept 357/excl 30/drop 0); "
                                                                    "n2_probe_reduce.n3_theta_positive_control(<r4 dir>)"),
                L_LAG=L_LAG, DELTA_FIXED=DELTA_FIXED, KAPPA_FLOOR=KAPPA_FLOOR, X_CONST=X_CONST, TRIM=TRIM, WIN=WIN,
                kappa_star=kstar, limits=limits_check(), derived=derived_constants(), budget=budget_record())


def is_minimal_k(d, K):
    return K is not None and k_satisfies(d, K) and k_within_limits(d, K) and (K == 1 or not k_satisfies(d, K - 1))


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ProbeRejected("duplicate JSON keys in probe file")
    return dict(pairs)


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def post_probe_gate(probe):
    """rev14: NOT A JUDGMENT INPUT. Retained as the smoke / design diagnostic of the optional post-seal harness
    smoke (U-2 demoted): applies the probe mapping to the coefficient sets, re-runs choose_k as a DIAGNOSTIC and
    reports whether the REGISTERED K_CO would still satisfy the requirement under the smoke's coefficients.
    K_CO is never modified here; 'K' in the record is the registered constant (the harness must not read any other)."""
    observed = apply_probe(probe)
    diag = {d: choose_k(d) for d in EA_D}
    reasons = []
    for d in EA_D:
        if not k_satisfies(d, K_CO[d]):
            reasons.append(f"registered K_CO[{d}]={K_CO[d]} would not meet the kappa* requirement under smoke coefficients")
    lim = limits_check()
    if not lim["ok"]:
        reasons.append(f"limits_check failed: {lim['max_len_cell']} len {lim['max_len']:.0f}")
    rec = dict(K=dict(K_CO), K_choose_k_diagnostic=diag, judgment_input=False, seal_allowed=not reasons,
               reasons=reasons, coef=copy.deepcopy(COEF), l_lag=L_LAG, delta_fixed=DELTA_FIXED, fix_s=dict(FIX_S),
               smoke_observed=observed,
               derived=derived_constants(), budget=budget_record())      # informational under smoke coefficients
    return rec


def budget_record():
    """R8-4: the ONLY budget source for the harness (stop rule, watchdog, re-approval): per-boot normal/worst cost,
    section, container, hard cap and the FIX_S in force, for both Jobs and both U-1 modes."""
    out = {"FIX_S": dict(FIX_S), "BOOT_OVERHEAD_S": dict(BOOT_OVERHEAD_S)}
    for job, boots, probes in (("PROFILE", boots_profile_job(), 5), ("EA", boots_ea_job(), 0)):
        for mode, cancel in (("a", False), ("b", True)):
            r = job_budget(boots, cancel, probes)
            out[f"{job}_{mode}"] = dict(per_boot_nom=r["nom"]["per_boot"], per_boot_worst=r["worst"]["per_boot"],
                                        section_nom=r["nom"]["section"], section_worst=r["worst"]["section"],
                                        container_nom=r["nom"]["container"], container_worst=r["worst"]["container"],
                                        cap=r["cap"])
    return out


def derived_constants():
    """R7-2: every per-cell constant the harness needs, computed HERE from the post-probe state. n2_plan must read
    these from K_POST_PROBE.json and never recompute them (it would get design coefficients)."""
    prof = {}
    for D in D_ON + (108,):
        for b in B:
            for c in C:
                prof[f"{D},{b},{c}"] = dict(c_in=c_in_of(D, b, c), s_hat=s_hat(D, b, c), O_P=o_profile(D, b, c))
    ea = {}
    for d in EA_D:
        for b in EA_B:
            for c in EA_C:
                ea[f"{d},{b},{c}"] = dict(K=K_CO[d], c_in=ealpha_c_in(d, b, c), s_hat=s_hat(d, b, c),
                                          O_EA=o_ealpha(d, b, c), N_co_max=n_co_max(d, b, c),
                                          delta_on_A2=delta_exec(d, b, c), delta_off_A2=delta_exec(108, b, c))
    return dict(profile=prof, ealpha=ea, L_LAG=L_LAG)


def k_sensitivity(k16s=(18, 19, 20, 21, 22), k44s=(8, 9, 10)):
    rows, old = [], dict(K_CO)
    try:
        for k16 in k16s:
            for k44 in k44s:
                K_CO[16], K_CO[44] = k16, k44
                rb, ra = job_budget(boots_ea_job(), True, 0), job_budget(boots_ea_job(), False, 0)
                rho = {d: min(min(rho_upper(d, b, c, pc, dc) for pc, dc in CORNERS) for b in EA_B for c in EA_C)
                       for d in EA_D}
                rows.append(dict(k16=k16, k44=k44, rho16=rho[16], rho44=rho[44], lim=limits_check()["max_len"],
                                 b_nom=rb["nom"]["container"], b_worst=rb["worst"]["container"], b_cap=rb["cap"],
                                 a_nom=ra["nom"]["container"], a_worst=ra["worst"]["container"], a_cap=ra["cap"]))
    finally:
        K_CO.clear(); K_CO.update(old)
    return rows


def _git(base, *args):
    import subprocess
    r = subprocess.run(["git", *args], cwd=base, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_freeze_manifest(freeze_path, calculator_sha, reducer_sha):
    """H-2 / R9-4 / R11-5: the manifest written for the pre-probe freeze commit (FREEZE_N2_PROBE.json):
    {"commit": <git sha of the freeze commit>, "files": {<path relative to the manifest>: <sha256>},
     "harness_mark_writers": [<path>, ...]}. EVERY listed file must exist with that sha256; the calculator and reducer
    entries must match the running tools; at least one harness mark writer must be listed; the git repository that
    holds the manifest must be checked out AT manifest.commit (HEAD == commit) with no uncommitted change in any
    listed file (untracked outputs such as probe.json / K_POST_PROBE.json / the manifest itself are not listed).
    This is a self-check by the running gate: it detects drift; it cannot prove that the gate code was not edited
    together with this check (lesson 9) -- the closing defence is the audit's K recomputation (R10-3 text)."""
    if freeze_path is None:
        return ["no pre-probe freeze manifest given (--freeze)"]
    with open(freeze_path) as fh:
        man = json.load(fh, object_pairs_hook=_no_duplicate_keys)
    base = os.path.dirname(os.path.abspath(freeze_path))
    files = man.get("files") if isinstance(man.get("files"), dict) else {}
    bad = []
    for rel, sha in files.items():
        path = os.path.join(base, rel)
        if not os.path.exists(path) or _sha256_file(path) != sha:
            bad.append(f"freeze manifest: {rel} missing or sha256 differs")
    by_name = {os.path.basename(rel): sha for rel, sha in files.items()}
    if by_name.get("n2_feasibility.py") != calculator_sha or by_name.get("n2_probe_reduce.py") != reducer_sha:
        bad.append("freeze manifest: calculator/reducer sha256 missing or differs from the running tools")
    writers = man.get("harness_mark_writers")
    if not isinstance(writers, list) or not writers or any(w not in files for w in writers):
        bad.append("freeze manifest: harness mark writer(s) missing from the frozen file list")
    try:                                                                      # R11-5: commit binding
        rc, head = _git(base, "rev-parse", "HEAD")
        if rc != 0 or not man.get("commit") or head != man.get("commit"):
            bad.append("freeze manifest: git HEAD is not manifest.commit (gate must run on the freeze checkout)")
        rc, dirty = _git(base, "status", "--porcelain", "--untracked-files=no", "--", *files.keys())
        if rc != 0 or dirty:
            bad.append("freeze manifest: frozen files have uncommitted changes in the freeze checkout")
    except OSError:
        bad.append("freeze manifest: git not available")
    return bad


def _load_registered_reducer():
    """R11-4: load the reducer that sits next to this calculator BY FILE PATH under a private module name, so a
    different 'n2_probe_reduce' already imported into the process can never be used; assert it is that file."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2_probe_reduce.py")
    spec = importlib.util.spec_from_file_location("_n2_registered_reducer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert os.path.samefile(mod.__file__, path)
    return mod


def reproduce_probe(probe_path, raw):
    """R10-3 / R11-4: re-reduce from provenance.boots with the registered reducer (loaded by file path) and compare
    the to_json bytes with the probe file. Returns None if identical, else (reason_code, text)."""
    boots = (raw.get("provenance") or {}).get("boots")
    if not isinstance(boots, list) or not boots:
        return ("PROVENANCE_MISSING", "probe.json has no provenance.boots to re-reduce")
    missing = [b.get(k) for b in boots for k in ("telemetry", "marks") if not os.path.exists(str(b.get(k)))]
    if missing:
        return ("RAW_PATH_MISSING", f"recorded raw path(s) do not open at the gate location: {missing[:3]}")
    try:
        red = _load_registered_reducer()
        again = red.to_json(red.reduce_probe(boots))
    except Exception as e:
        return ("REDUCTION_FAILED", f"re-reduction failed: {e!r}")
    with open(probe_path) as fh:
        if fh.read() != again:
            return ("PROBE_BYTES_MISMATCH", "probe.json differs from the re-reduction of its raw files")
    return None


def registered_record():
    """The exact record `--registered` writes (so verify_registered can regenerate it byte for byte)."""
    rec = registered_constants()
    rec["calculator_sha256"] = _sha256_file(__file__)
    rec["derive_fixed_k"] = derive_fixed_k()
    rec["K_matches_derivation"] = rec["derive_fixed_k"] == K_CO
    return rec


def registered_bytes(rec):
    return (json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n").encode("utf-8")


def verify_registered(path, digest=None):
    """H14-2: a registered-constants file is accepted ONLY if its bytes equal the record regenerated by THIS
    calculator (so a hand-edited K / budget with intact self-report fields is rejected) and, when a sealed digest is
    given, its sha256 equals that digest. Returns a list of problems (empty = accepted). Self-check caveat: this
    is run by the calculator itself (N2C-69) -- the sealed digest and the audit's clean-checkout regeneration are the
    external evidence."""
    problems = []
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        return [f"registered file unreadable: {e}"]
    if data != registered_bytes(registered_record()):
        problems.append("registered file bytes differ from the record regenerated by this calculator")
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        problems.append("registered file sha256 differs from the sealed digest")
    return problems


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--verify-registered"]:
        # rev15 H14-2: python3 n2_feasibility.py --verify-registered N2_REGISTERED.json [<sealed sha256>]
        problems = verify_registered(argv[1], argv[2] if len(argv) > 2 else None)
        print(json.dumps({"ok": not problems, "problems": problems}))
        return 0 if not problems else 2
    if argv[:1] == ["--registered"]:
        # rev14: python3 n2_feasibility.py --registered N2_REGISTERED.json  -- the harness's single constants source
        rec = registered_record()
        with open(argv[1], "wb") as fh:
            fh.write(registered_bytes(rec))
        print(json.dumps({"K": rec["K"], "K_matches_derivation": rec["K_matches_derivation"],
                          "limits_ok": rec["limits"]["ok"]}))
        return 0 if rec["K_matches_derivation"] and rec["limits"]["ok"] else 2
    if argv[:1] == ["--probe"]:
        # post-probe CLI: python3 n2_feasibility.py --probe probe.json --out K_POST_PROBE.json --freeze FREEZE.json
        probe_path, out_path = argv[1], argv[3]
        freeze_path = argv[5] if argv[4:5] == ["--freeze"] else None
        with open(probe_path) as fh:
            raw = json.load(fh, object_pairs_hook=_no_duplicate_keys)
        probe = {k: ({tuple(json.loads(kk)) if kk.startswith("[") else int(kk): vv for kk, vv in v.items()}
                     if isinstance(v, dict) else v) for k, v in raw.items() if k in REQUIRED_PROBE_KEYS}
        status = raw.get("probe_status")
        irreproducible = reproduce_probe(probe_path, raw)
        if irreproducible is not None:
            # R10-3: a probe.json that its own raw files do not reproduce is not evidence -> no K, no re-probe
            rec = dict(K=None, seal_allowed=False, reasons=[irreproducible[1]], reprobe_allowed=False,
                       reason_codes=[irreproducible[0]])
        elif status != "OK":
            # R9-1: KAPPA_CENSORED / VALUE_DEPENDENT -> no seal and NO re-probe (user decision); VALUE_INDEPENDENT ->
            # re-probe allowed. The gate is not run on a non-OK probe (no K is produced from it).
            rec = dict(K=None, seal_allowed=False, reasons=[f"probe_status {status}"],
                       reprobe_allowed=(status == "VALUE_INDEPENDENT"), reason_codes=list(raw.get("reason_codes", [])))
        else:
            try:
                rec = post_probe_gate(probe)
                rec["reprobe_allowed"] = False
                rec["reason_codes"] = []
            except ProbeRejected as e:
                rec = dict(K=None, seal_allowed=False, reasons=[f"ProbeRejected: {e}"],
                           reprobe_allowed=e.value_independent, reason_codes=[e.code])
        rec["probe_status"] = status
        # R8-3: reducer provenance -- probe.json must carry the sha256 of the registered reducer next to this file
        prov = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
        red_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n2_probe_reduce.py")
        local_red = _sha256_file(red_path) if os.path.exists(red_path) else None
        rec["reducer_sha256"] = prov.get("reducer_sha256")
        rec["reducer_sha256_local"] = local_red
        rec["raw_files"] = prov.get("raw_files", {})
        if not rec["reducer_sha256"] or rec["reducer_sha256"] != local_red:
            rec["seal_allowed"] = False
            rec["reasons"] = list(rec.get("reasons", [])) + ["reducer sha256 missing or differs from registered reducer"]
        rec["probe_sha256"] = _sha256_file(probe_path)
        rec["calculator_sha256"] = _sha256_file(__file__)
        # R9-4 / H-2: every frozen file (incl. the harness mark writers) must match the pre-probe freeze manifest
        bad = check_freeze_manifest(freeze_path, rec["calculator_sha256"], local_red)
        rec["freeze_problems"] = bad
        if bad:
            rec["seal_allowed"] = False
            rec["reasons"] = list(rec["reasons"]) + bad
        with open(out_path, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True, default=str)
        print(json.dumps({"K": rec["K"], "seal_allowed": rec["seal_allowed"], "reasons": rec["reasons"]}))
        return 0 if rec["seal_allowed"] else 2
    print("== rho upper-bound table ==")
    for r in rho_table():
        ub = " ".join(f"{pc}{dc}={v:.2f}" for (pc, dc), v in r["rho_ub"].items())
        print(f"cell{r['cell']} K={r['K']} {ub} | worst={r['rho_ub_worst']:.2f} W_co_min={r['w_co_min']:.0f} "
              f"{'PASS' if r['passes'] else 'FAIL'}")
    print("== limits ==", limits_check())
    for cancel in (False, True):
        tag = "(b) cancel" if cancel else "(a) no cancel"
        for name, boots, probes in (("Job-PROFILE", boots_profile_job(), 5), ("Job-EA", boots_ea_job(), 0)):
            r = job_budget(boots, cancel, probes)
            print(f"{name} {tag}: nom section {r['nom']['section']:.0f} container {r['nom']['container']:.0f} | "
                  f"worst section {r['worst']['section']:.0f} container {r['worst']['container']:.0f} | cap {r['cap']}")
            print("   per-boot nom", {k: round(v) for k, v in r['nom']['per_boot'].items()},
                  "worst", {k: round(v) for k, v in r['worst']['per_boot'].items()})
    print("== rev14 fixed K (derive_fixed_k, registration-time) ==", derive_fixed_k(), "registered K_CO", K_CO)
    print("== N_co,max (lower-side, kappa_floor) / O_EA per E-alpha cell ==",
          {f"{d},{b},{c}": (n_co_max(d, b, c), o_ealpha(d, b, c)) for d in EA_D for b in EA_B for c in EA_C})
    return 0


if __name__ == "__main__":
    sys.exit(main())
