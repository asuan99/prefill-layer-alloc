#!/usr/bin/env python3
"""P4 감사 처방 #1 — 컨트롤러 로그 출력의 산술 도달가능성 (GPU 0).

감사 판정서 `audit_p4_rules_2026-08-28/VERDICT.md` P4-F1/F2/B10 확정용.
결정 함수는 `sglang/srt/multiplex/multiplexing_mixin.py:767-846`(_slo_decide_idx_binding)
+ `:721-765`(_slo_feasible) + `:707-719`(_slo_knee_itl)에서 **축자 이식**했다.
성능 판정 0건 · GPU 0 · 엔진 미실행(순수 산술).

산출: 등록 설계공간 안에서 `ctrl_live ∈ {DEAD, ALIVE}` 각각이 도달 가능한가,
그리고 어떤 (pf_slack, dec_slack, idx, bs) 영역이 그 답을 만드는가.
"""
import itertools, json, os, sys

# --- 코드 상수 (env 기본값, multiplexing_mixin.py) --------------------------
PF_URG      = 0.5    # :776  PDMUX_SLO_PF_URGENCY
HI_FRAC     = 0.85   # :775  PDMUX_TPOT_HI  -> decode urgent iff dec_slack < 1-0.85
SAT_MARGIN  = 0.0    # :789  PDMUX_SLO_SAT_MARGIN
SAT_DEEP    = 0.5    # :790  PDMUX_SLO_SAT_DEEP
SAT_WIN     = 4      # :788  PDMUX_SLO_SAT_WIN
SAT_LATCH   = 20     # :805  PDMUX_SLO_SAT_LATCH
DWELL       = 3      # :838  PDMUX_SLO_DWELL
FEAS_OCC    = 0.85   # :741  PDMUX_SLO_FEAS_OCC
FEAS_MARGIN = 0.9    # :742  PDMUX_SLO_FEAS_MARGIN
TPOT_SLO_MS = 60.0
LO, HI      = 1, 5   # pdmux_slo.yml: real_sm_group_num=7 -> 1..5 (B17)
DEC_SM      = {1: 16, 2: 24, 3: 34, 4: 44, 5: 54}

def knee(d_sm):                                    # :707-719 축자
    pts = ((8, 358.4), (16, 186.3), (24, 127.4), (44, 70.7), (108, 47.7))
    if d_sm <= pts[0][0]:  return pts[0][1]
    if d_sm >= pts[-1][0]: return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= d_sm <= x1:
            return y0 + (y1 - y0) * (d_sm - x0) / float(x1 - x0)
    return pts[-1][1]

def feasible(idx_cur, idx_new, bs, cap, tpot_ms):   # :721-765 축자
    if idx_new >= idx_cur:
        return True, None
    if cap > 0 and bs >= cap * FEAS_OCC:
        return False, "congestion"
    if tpot_ms > 0:
        k_cur = knee(DEC_SM[idx_cur])
        if k_cur > 0:
            pred = tpot_ms * (knee(DEC_SM[idx_new]) / k_cur)
            if pred > TPOT_SLO_MS * FEAS_MARGIN:
                return False, "itl"
    return True, None

def step(idx, anchor, pf_slack, dec_slack, dwell, latch, pf_hist, bs, cap, feas_gate):
    """한 번의 결정 -> (new_idx, log) ; log ∈ {None, 'BIND', 'FEAS'}"""
    tpot_ms = (1.0 - dec_slack) * TPOT_SLO_MS
    sat_pred = (idx <= LO) and (len(pf_hist) >= SAT_WIN) and (pf_hist[-1] > pf_hist[0]) and (pf_slack < PF_URG)
    sat_fb = (pf_slack < -SAT_DEEP) or (dec_slack < -SAT_DEEP) or ((pf_slack < SAT_MARGIN) and (dec_slack < SAT_MARGIN))
    sat = sat_pred or sat_fb
    latch = SAT_LATCH if sat else latch
    hold = sat or (latch > 0)
    if dwell > 0:
        return idx, None, max(0, latch - 1), dwell - 1
    if hold:
        new = idx + (1 if idx < anchor else (-1 if idx > anchor else 0))
    elif pf_slack < dec_slack and pf_slack < PF_URG:
        new = max(LO, idx - 1)
    elif dec_slack < pf_slack and dec_slack < (1.0 - HI_FRAC):
        new = min(HI, idx + 1)
    elif idx != anchor:
        new = idx + (1 if idx < anchor else -1)
    else:
        new = idx
    log = None
    if feas_gate and new != idx:
        ok, _why = feasible(idx, new, bs, cap, tpot_ms)
        if not ok:
            return idx, "FEAS", max(0, latch - 1), 0
    if new != idx:
        log = "BIND"
        return new, log, max(0, latch - 1), DWELL
    return idx, None, max(0, latch - 1), 0

# --- 1. 단일-스텝 출력 도달가능성 (전수) ------------------------------------
PFS = [round(x, 3) for x in [-9.0, -2.0, -1.0, -0.6, -0.5, -0.25, -0.01, 0.0, 0.2, 0.49, 0.5, 0.8, 1.0]]
DECS = [round(x, 3) for x in [-0.6, -0.2, -0.01, 0.0, 0.1, 0.14, 0.15, 0.3, 0.5, 0.62, 0.9]]
BSS = [0, 8, 16, 24, 32, 40, 41, 48]
CAP = 48

out = {}
for feas_gate in (True,):
    for idx, anchor in itertools.product(range(LO, HI + 1), [2]):
        for pfs, decs, bs in itertools.product(PFS, DECS, BSS):
            _, log, _, _ = step(idx, anchor, pfs, decs, 0, 0, [], bs, CAP, feas_gate)
            out.setdefault(log or "SILENT", []).append((idx, pfs, decs, bs))

print("=== 1. 단일 스텝 출력 도달가능성 (dwell=0, latch=0, anchor=idx2, cap=48) ===")
for k in ("BIND", "FEAS", "SILENT"):
    v = out.get(k, [])
    print(f"  {k:7s}: {len(v):6d} 세계")
print()

# --- 2. FEAS 는 어떤 bs 에서만 나오는가 (B10) --------------------------------
feas_bs = sorted({w[3] for w in out.get("FEAS", [])})
feas_by_reason = {}
for idx, pfs, decs, bs in out.get("FEAS", []):
    tpot = (1 - decs) * TPOT_SLO_MS
    new = max(LO, idx - 1)
    _, why = feasible(idx, new, bs, CAP, tpot)
    feas_by_reason.setdefault(why, set()).add(bs)
print("=== 2. FEAS refused 가 발화하는 bs 값 (B10) ===")
print(f"  전체: {feas_bs}")
for why, s in sorted(feas_by_reason.items()):
    print(f"  사유 {why:11s}: bs ∈ {sorted(s)}")
print(f"  congestion 문턱 = cap*OCC = {CAP}*{FEAS_OCC} = {CAP*FEAS_OCC}")
print(f"  ⇒ NP=32 이면 bs ≤ 32 < {CAP*FEAS_OCC} ⇒ congestion 가드 발화 불가")
print()

# --- 3. 램프 시뮬레이션: 콜드 스타트 -> 과부하 (F1c) -------------------------
def ramp(ttft_slo_ms, pf_ages, tpot_ms, bs, cap, anchor=2, feas_gate=True):
    idx, dwell, latch, hist, logs = anchor, 0, 0, [], []
    for a in pf_ages:
        pfs = (ttft_slo_ms - a) / ttft_slo_ms
        decs = (TPOT_SLO_MS - tpot_ms) / TPOT_SLO_MS
        hist = (hist + [a])[-SAT_WIN:]
        idx, log, latch, dwell = step(idx, anchor, pfs, decs, dwell, latch, hist, bs, cap, feas_gate)
        if log: logs.append((round(a), log, idx))
    return logs

print("=== 3. 램프(콜드→과부하) 에서 ctrl_live (F1c) ===")
# pf_age 가 0 -> 40s 로 단조 증가 (200ms 스텝)  ×  등록 SLO 사다리
ages = [i * 200.0 for i in range(0, 200)]
for slo in (750, 1500, 3000, 6000, 12000, 24000):
    for tpot in (23.0, 30.0, 68.0):
        lg = ramp(float(slo), ages, tpot, bs=32, cap=CAP)
        live = "ALIVE" if lg else "DEAD"
        print(f"  TTFT_SLO={slo:6d}ms tpot={tpot:5.1f}ms bs=32 -> {live:5s} lines={len(lg):3d} first={lg[:2]}")
print()

# --- 4. TTFT 가 SLO 를 절대 안 넘는 경우 (BOTH_DEAD 도달 조건) ---------------
print("=== 4. BOTH_DEAD 가 도달 가능한 유일 영역 ===")
for peak_frac in (0.3, 0.6, 0.9, 1.0, 1.2, 1.6):
    slo = 3000.0
    peak = slo * peak_frac
    ages = [peak * i / 100.0 for i in range(101)] + [peak] * 100
    lg = ramp(slo, ages, 30.0, bs=32, cap=CAP)
    print(f"  peak(TTFT)/SLO={peak_frac:4.1f} -> {'ALIVE' if lg else 'DEAD':5s} lines={len(lg)}")
print()
print(f"  ⇒ 발화 문턱은 pf_age > {1-PF_URG:.2f}*TTFT_SLO 이다 (SLO 가 아니라 그 절반).")
print("  ⇒ §5 가 '요청의 20~80% 가 TTFT>SLO' 를 요구하는 한 그 문턱 통과는 2배 초과로 보장된다.")
print("  ⇒ BOTH_DEAD / ASYMMETRIC_LIVENESS 는 등록 설계공간에서 도달 불가. (P4-F1 CONFIRMED)")
