"""N2(3-a) design-time feasibility & budget calculator (CPU only, no GPU).

Status: DESIGN CALCULATOR for PREREG_N2_3A rev8 (unsealed). Every number it prints is a design
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
THETA_DESIGN = 4500.0        # tok/s at p = 64 -- SOURCE: none (design assumption, N2C-33)
KAPPA_DN_DESIGN = 1.3
KAPPA_DW_OVER_DN = 1.6 / 1.3
THETA_WORST_MULT = 0.6
TAU_WORST_MULT = 1.5
DIRECTIONS = ("rho", "env", "budget")
COEF = {d: dict(theta=THETA_DESIGN, d_factor=dict(D_FACTOR_DESIGN), kappa_dn=KAPPA_DN_DESIGN) for d in DIRECTIONS}

MAX_PREFILL_TOKENS = 16384
CONTEXT_LEN = 16384
KV_TOKENS = 2_721_290
CAP = 48
K_CO = {16: 18, 44: 8}       # PRE-PROBE registered values; post-probe K comes ONLY from post_probe_gate()
CO_INPUT = 8192
L_LAG = 32                   # unmeasured allowance; probe -> max
X_CONST = 145
TRIM, WIN = 16, 128
C_WINDOW_MIN = 128
RHO_MAX_REQUIRED = 2.0       # registered design requirement (main session, 2026-10-07)
DELTA_MIN = 8
DELTA_L_S = 0.0              # client latency asymmetry (s); probe -> max
FIX_S = {"nom": 22.48, "worst": 35.5}
BOOT_OVERHEAD_S = {"nom": 170.5, "worst": 229.0}   # Job-measured; the probe never changes it
PREFLIGHT_S = {"nom": 224.0, "worst": 600.0}
COLD_S = 60.0
ROUNDS = 4
CAP_FACTOR = 1.1
THETA_MIN_SPANS = 20
THETA_REQUIRED_P = (64, 92)
THETA_ALLOWED_P = (64, 74, 84, 92)     # R7-6: prefill SMs of the A-agn splits
REQUIRED_PROBE_KEYS = ("theta_spans", "tau_cells", "kappa_cells", "l_lag_iters", "delta_l_s", "fix_s")
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


def n_co_max(Dsp, b, c):
    """worst-corner (largest) W_co with env-direction coefficients (R6-5)"""
    return math.ceil(max(w_co(Dsp, b, c, pc, dc, "env") for pc, dc in CORNERS))


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
    return (c + 71 - ealpha_c_in(d, b, c)) + L_LAG + n_co_max(d, b, c) + TRIM + WIN + s_hat(d, b, c) + 64


def delta_tokens(Dwin, b, c, dc="DN", dirn="budget"):
    """R5-5: delta per window, tau of THAT window (ON A2: d, OFF A2: 108).
    dirn="budget" for time tables; the EXECUTION delta (cancel margin) uses delta_exec (env, R7-5)."""
    if DELTA_L_S <= 0:
        return DELTA_MIN
    return max(DELTA_MIN, math.ceil(DELTA_L_S / tau_alone(Dwin, b, c, dc, dirn)))


def delta_exec(Dwin, b, c):
    """R7-5: execution cancel margin -- tau_win from the env (MIN) set, DN corner (fastest decode => largest delta)."""
    return delta_tokens(Dwin, b, c, "DN", "env")


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
    pass


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
      kappa_cells: {(d, b, c): {"c_median_ms": scalar, "a_on_ms": scalar, "window_iters": int}}  (rho_on form)
                   kappa_cell = c_median / a_on ; cross-cell: rho/budget = MAX, env = MIN ; kappa_DW = kappa_DN*1.6/1.3
      l_lag_iters: [scalars] -> MAX ; delta_l_s: [scalars] -> MAX ; fix_s: [scalars] -> nom MEDIAN, worst MAX
    Required: tau_cells and kappa_cells each include BINDING_EA_CELLS. s_hat / N_co,max are never given -- they are
    recomputed from the env set. Boot overhead is never changed."""
    global L_LAG, DELTA_L_S
    for key in REQUIRED_PROBE_KEYS:                                   # R7-6: no silent design fallback
        if key not in probe:
            raise ProbeRejected(f"probe key missing: {key}")
    new = copy.deepcopy(COEF)
    # theta
    spans = probe["theta_spans"]
    for p in THETA_REQUIRED_P:
        if p not in spans:
            raise ProbeRejected(f"theta spans missing for p={p}")
    med = {}
    for p, lst in spans.items():
        if p not in THETA_ALLOWED_P:
            raise ProbeRejected(f"theta: p={p} not a registered prefill split")
        if not isinstance(lst, (list, tuple)) or len(lst) < THETA_MIN_SPANS:
            raise ProbeRejected(f"theta: need >= {THETA_MIN_SPANS} realized-split spans at p={p} (counted after exclusion)")
        med[p] = statistics.median(_scalar(v, "theta") for v in lst) * 64.0 / p
    new["rho"]["theta"] = max(med.values())
    new["env"]["theta"] = new["budget"]["theta"] = min(med.values())
    # tau
    tc = probe["tau_cells"]
    for cell in BINDING_EA_CELLS:
        if cell not in tc:
            raise ProbeRejected(f"tau cells must include binding cell {cell}")
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
            raise ProbeRejected(f"kappa cells must include binding cell {cell}")
    ks = []
    for cell, v in kc.items():
        _check_cell(cell, "kappa", (16, 44))
        if not isinstance(v, dict) or int(v.get("window_iters", 0)) < WINDOW_ITERS \
                or int(v.get("a_on_window_iters", 0)) < WINDOW_ITERS:
            raise ProbeRejected("kappa cell needs c_median_ms / a_on_ms / window_iters >= 128 / a_on_window_iters >= 128")
        ks.append(_scalar(v["c_median_ms"], "kappa") / _scalar(v["a_on_ms"], "kappa"))
    new["rho"]["kappa_dn"] = new["budget"]["kappa_dn"] = max(ks)
    new["env"]["kappa_dn"] = min(ks)
    def _nonempty(key):
        lst = probe[key]
        if not isinstance(lst, (list, tuple)) or not lst:
            raise ProbeRejected(f"{key} must be a non-empty list")
        return lst
    l_lag = int(math.ceil(max(_scalar(x, "l_lag") for x in _nonempty("l_lag_iters"))))
    # delta-l is a SIGNED difference l_d - l_co: finite required; <= 0 means no disadvantage (clipped to 0 by delta)
    dl = max(_scalar(x, "delta_l", positive=False) for x in _nonempty("delta_l_s"))
    xs = [_scalar(x, "fix") for x in _nonempty("fix_s")]
    COEF.clear(); COEF.update(new)
    L_LAG, DELTA_L_S = l_lag, max(0.0, dl)
    FIX_S["nom"], FIX_S["worst"] = statistics.median(xs), max(xs)


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
    E-alpha length grows strictly with K (N_co,max ~ K), so the loop always terminates."""
    K = 1
    while True:
        if not k_within_limits(d, K):
            return None
        if k_satisfies(d, K):
            return K
        K += 1


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
    """The ONLY post-probe K source (R6-6). Applies the mapping, then choose_k unconditionally (R6-7),
    then limits_check over ALL cells. seal_allowed is False if any K is None or limits fail."""
    apply_probe(probe)
    ks = {d: choose_k(d) for d in EA_D}
    reasons = [f"no K within limits for d={d}" for d, k in ks.items() if k is None]
    if not reasons:
        K_CO.update(ks)
        lim = limits_check()
        if not lim["ok"]:
            reasons.append(f"limits_check failed: {lim['max_len_cell']} len {lim['max_len']:.0f}")
        if not all(r["passes"] for r in rho_table()):
            reasons.append("rho table requirement failed after K choice")
    rec = dict(K=ks, seal_allowed=not reasons, reasons=reasons,
               coef=copy.deepcopy(COEF), l_lag=L_LAG, delta_l_s=DELTA_L_S, fix_s=dict(FIX_S))
    if not reasons:
        rec["derived"] = derived_constants()
    return rec


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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--probe"]:
        # post-probe CLI: python3 n2_feasibility.py --probe probe.json --out K_POST_PROBE.json
        probe_path, out_path = argv[1], argv[3]
        with open(probe_path) as fh:
            raw = json.load(fh, object_pairs_hook=_no_duplicate_keys)
        probe = {k: ({tuple(json.loads(kk)) if kk.startswith("[") else int(kk): vv for kk, vv in v.items()}
                     if isinstance(v, dict) else v) for k, v in raw.items() if k != "theta_excluded"}
        try:
            rec = post_probe_gate(probe)
        except ProbeRejected as e:
            rec = dict(K=None, seal_allowed=False, reasons=[f"ProbeRejected: {e}"])
        rec["probe_sha256"] = _sha256_file(probe_path)
        rec["calculator_sha256"] = _sha256_file(__file__)
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
    print("== U-9 rule under design coefficients ==", {d: choose_k(d) for d in EA_D})
    return 0


if __name__ == "__main__":
    sys.exit(main())
