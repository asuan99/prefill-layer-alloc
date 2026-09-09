"""GPU 0. (c) 다중성 압축의 귀무 발화율 — 사전등록 결정 규칙 자신을 시뮬레이션한다.

등록 규칙(PREREG_L1 rev2 §3.5(a-2)/§4.2/§4.3(a)/§4.4)을 그대로 구현하고,
"세 split의 참 분포가 동일"한 귀무에서 `ARGMAX_MOVES`가 몇 %나 나오는지 잰다.

핵심 위험 2가지를 일부러 모두 재현한다:
  (1) 면 위 칸들은 같은 요청 집합에서 나오므로 강하게 상관 -> "연결 4칸"이 보이는
      것보다 훨씬 약한 보호일 수 있다.
  (2) 승자/차순위를 같은 데이터로 고르고 같은 데이터로 검정한다(이중 사용).
"""
import numpy as np

RNG = np.random.default_rng(20260908)

N_CTX      = 4
N_SPLIT    = 3
N_BOOT     = 4
N_REQ      = 60          # PREREG 3.3: 셀당 최소 요청 수
S_GRID     = np.array([2, 3, 5, 8, 12, 20], float)   # TTFT: s x floor_ref(L)
ITL_GRID   = np.array([30., 40., 60., 100.])         # ms
FLOOR      = 1.0         # 정규화 단위(=floor_ref(L)); TTFT는 floor 배수로 표현
MED_TTFT   = 5.0         # 부하 하 TTFT 중앙값 = 바닥의 5배 (USABLE 칸이 생기는 영역)
SIG_TTFT   = 0.55        # 요청 간 로그 산포
MED_ITL    = 45.0
SIG_ITL    = 0.25
SIG_BOOT   = 0.06        # 부팅x split 랜덤효과(로그 스케일) = 페어드 잡음의 원천
FLOOR_PRACT= 0.03
ALPHA      = 0.05
M_FAMILY   = 4           # 등록된 m (ctx당 검정 1개)

def t_crit(alpha, df):
    """정본 d1_predicates.t_crit_for 와 같은 이분법 (scipy 비의존)."""
    import math
    def betai(a, b, x):
        # 연속분수 불완전베타 (numerically standard)
        def betacf(a, b, x):
            MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
            qab, qap, qam = a+b, a+1.0, a-1.0
            c = 1.0; d = 1.0 - qab*x/qap
            if abs(d) < FPMIN: d = FPMIN
            d = 1.0/d; h = d
            for m in range(1, MAXIT+1):
                m2 = 2*m
                aa = m*(b-m)*x/((qam+m2)*(a+m2))
                d = 1.0+aa*d
                if abs(d) < FPMIN: d = FPMIN
                c = 1.0+aa/c
                if abs(c) < FPMIN: c = FPMIN
                d = 1.0/d; h *= d*c
                aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
                d = 1.0+aa*d
                if abs(d) < FPMIN: d = FPMIN
                c = 1.0+aa/c
                if abs(c) < FPMIN: c = FPMIN
                d = 1.0/d; de = d*c; h *= de
                if abs(de-1.0) < EPS: break
            return h
        if x <= 0.0: return 0.0
        if x >= 1.0: return 1.0
        lbeta = (math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                 + a*math.log(x) + b*math.log(1.0-x))
        if x < (a+1.0)/(a+b+2.0):
            return math.exp(lbeta)*betacf(a, b, x)/a
        return 1.0-math.exp(lbeta)*betacf(b, a, 1.0-x)/b
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo+hi)/2.0
        if betai(df/2.0, 0.5, df/(df+mid*mid)) > alpha: lo = mid
        else: hi = mid
    return (lo+hi)/2.0

TC = {m: t_crit(ALPHA/m, N_BOOT-1) for m in (1, 2, 3, 4, 12)}

def one_rep(rng, sig_boot=SIG_BOOT):
    """한 번의 캠페인 전체를 귀무에서 생성하고 등록 규칙을 적용한다."""
    # goodput[ctx, boot, split, s_i, itl_j]
    gp = np.empty((N_CTX, N_BOOT, N_SPLIT, len(S_GRID), len(ITL_GRID)))
    for c in range(N_CTX):
        for b in range(N_BOOT):
            for k in range(N_SPLIT):
                d = rng.normal(0.0, sig_boot)          # 부팅x split 랜덤효과
                ttft = np.exp(np.log(MED_TTFT) + d + rng.normal(0, SIG_TTFT, N_REQ)) * FLOOR
                itl  = np.exp(np.log(MED_ITL) + d + rng.normal(0, SIG_ITL,  N_REQ))
                ok_t = ttft[:, None] <= S_GRID[None, :]          # (req, s)
                ok_i = itl[:, None]  <= ITL_GRID[None, :]        # (req, itl)
                gp[c, b, k] = (ok_t[:, :, None] & ok_i[:, None, :]).mean(axis=0)

    cell_mean = gp.mean(axis=1)                     # (ctx, split, s, itl)
    usable = ~((cell_mean < 0.10).all(axis=1) | (cell_mean > 0.95).all(axis=1))  # (ctx,s,itl)

    per_boot_arg = gp.argmax(axis=2)                # (ctx, boot, s, itl)
    unanimous = (per_boot_arg == per_boot_arg[:, :1]).all(axis=1)                # (ctx,s,itl)
    winner = per_boot_arg[:, 0]                                                   # (ctx,s,itl)
    valid = usable & unanimous

    # ctx 쌍마다: argmax가 다른 연결 4칸 이상 띠가 있는가
    H, W = len(S_GRID), len(ITL_GRID)
    def components(mask):
        seen = np.zeros_like(mask, bool); out = []
        for i in range(H):
            for j in range(W):
                if mask[i, j] and not seen[i, j]:
                    stack, comp = [(i, j)], []
                    seen[i, j] = True
                    while stack:
                        x, y = stack.pop(); comp.append((x, y))
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            u, v = x+dx, y+dy
                            if 0 <= u < H and 0 <= v < W and mask[u, v] and not seen[u, v]:
                                seen[u, v] = True; stack.append((u, v))
                    out.append(comp)
        return out

    fired = False
    for c1 in range(N_CTX):
        for c2 in range(c1+1, N_CTX):
            diff = valid[c1] & valid[c2] & (winner[c1] != winner[c2])
            for comp in components(diff):
                if len(comp) < 4:
                    continue
                # 등록된 검정: 각 ctx에서 띠 평균 goodput의 승자-차순위 페어드 차이
                ok = True
                for c in (c1, c2):
                    idx = tuple(np.array(comp).T)
                    band = gp[c][:, :, idx[0], idx[1]].mean(axis=2)      # (boot, split)
                    w = winner[c][comp[0]]
                    runner = np.argsort(band.mean(axis=0))[-2]
                    if runner == w:
                        runner = np.argsort(band.mean(axis=0))[-1]
                    d = band[:, w] - band[:, runner]
                    sd = d.std(ddof=1)
                    if sd == 0:
                        ok = False; break
                    hw = TC[M_FAMILY] * sd / np.sqrt(N_BOOT)
                    m_ = d.mean()
                    rel = abs(m_) / max(band.mean(axis=0).max(), 1e-12)
                    if not ((m_ - hw > 0 or m_ + hw < 0) and rel >= FLOOR_PRACT):
                        ok = False; break
                if ok:
                    fired = True
            if fired: break
        if fired: break
    return fired

if __name__ == "__main__":
    import sys
    NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    hits = sum(one_rep(RNG) for _ in range(NREP))
    p = hits / NREP
    se = (p*(1-p)/NREP) ** 0.5
    print(f"N_REP={NREP}  ARGMAX_MOVES 귀무 발화율 = {p:.4f}  (+-{1.96*se:.4f}, 95% CI)")
    print(f"명목 alpha = {ALPHA}  (m={M_FAMILY}, t_crit={TC[M_FAMILY]:.3f}, df={N_BOOT-1})")
