#!/usr/bin/env python3
"""요약통계 선택이 decode-SM 축 탐지력에 미치는 영향 (GPU 0).

통제: G16 4블록 전 데이터 -- 모델(Zamba2-2.7B)·워크로드(ShareGPT)·빌드·arm 집합 전부 동일.
변인: **요약통계만**.  같은 token ITL에서 서로 다른 요약을 뽑아 (a) 효과 크기,
      (b) 블록 쌍대응 잡음, (c) 효과/잡음, (d) K1 도너 식별 여부를 비교한다.

★이것은 G16 사전등록 판정의 재판정이 **아니다**.  사전등록 estimand는 p95이고 규칙은
그대로 유효하다.  여기서 묻는 것은 **측정 성질**뿐이다: "split이 ITL에 영향을 주는지
*탐지*하는 데 어느 요약이 축에 반응하는가."

게이트 #40 사전 점검(항등식):
  - token ITL이 이봉(stall~58.5 / fast~20.3)임은 **이미 측정된 사실**이다.  그 조건에서
    "질량이 이동하면 mean은 움직이고 p95는 안 움직인다"는 **부분적으로 구조적**이다.
    ⇒ 결론을 "mean이 더 낫다"로 쓰면 항등식을 증거로 쓰는 것이다.  쓸 수 있는 결론은
    **"분포가 이봉이므로 요약 선택이 탐지력을 결정한다"** 까지다.
  - 반면 **효과/잡음 비**는 항등식이 아니다 -- 민감한 요약이 더 시끄러울 수 있고 그건
    데이터가 정한다.  그래서 1차 보고량을 효과가 아니라 **효과/잡음**으로 둔다.
"""
import json, glob, os, statistics as st, random, math

SL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARMS = ["d16","d24","d34","d44","d54","d64","d74"]
U    = ["d44","d54","d64","d74"]
BLK  = ["blk1","blk2","blk3","blk4"]

def pct(s, q):
    if not s: return float("nan")
    s = sorted(s); k = (len(s)-1)*q
    f, c = int(k), min(int(k)+1, len(s)-1)
    return s[f] + (s[c]-s[f])*(k-f)

def load(arm, blk, phase):
    pat = os.path.join(SL, f"g16_{blk}_{arm}_boot1_*_{phase}.jsonl")
    reqs = []
    for f in [x for x in glob.glob(pat) if "smoke" not in x]:
        for line in open(f):
            e = json.loads(line)
            for itl in e.get("itls", []):
                if itl: reqs.append([1000.0*x for x in itl])
    return reqs

# 후보 요약통계: 요청별 값을 만든 뒤 요청 간 중앙값 (사전등록 M_itl과 같은 2단 구조)
SUMS = {
    "p95 (사전등록)":  lambda r: pct(r, 0.95),
    "요청별 mean":     lambda r: st.fmean(r),
    "요청별 p50":      lambda r: pct(r, 0.50),
    "요청별 p99":      lambda r: pct(r, 0.99),
    "stall 분율(>45ms)": lambda r: sum(1 for x in r if x > 45)/len(r),
}

def cell_values(phase):
    """(summary, arm, blk) -> 값"""
    out = {}
    for blk in BLK:
        for arm in ARMS:
            reqs = load(arm, blk, phase)
            if not reqs: continue
            for name, fn in SUMS.items():
                out[(name, arm, blk)] = st.median([fn(r) for r in reqs])
    return out

def k1_identified(vals, arms, lower_is_better=True):
    """K1: 블록 argmin >= 3/4 AND 블록 부트스트랩 >= 0.80 (사전등록 규칙 그대로)."""
    per = {}
    for blk in BLK:
        cand = {a: vals[(a, blk)] for a in arms if (a, blk) in vals}
        if not cand: continue
        per[blk] = (min if lower_is_better else max)(cand, key=lambda a: cand[a])
    if not per: return False, None, 0.0
    cnt = {}
    for w in per.values(): cnt[w] = cnt.get(w, 0)+1
    rank_arm = max(cnt, key=lambda a: cnt[a]); rank_ok = cnt[rank_arm] >= 3
    rng = random.Random(1); freq = {}
    for _ in range(10000):
        pick = [rng.choice(BLK) for _ in BLK]
        m = {}
        for a in arms:
            v = [vals[(a,b)] for b in pick if (a,b) in vals]
            if v: m[a] = st.fmean(v)
        if not m: continue
        w = (min if lower_is_better else max)(m, key=lambda a: m[a])
        freq[w] = freq.get(w,0)+1
    barm = max(freq, key=lambda a: freq[a]); bfrac = freq[barm]/10000
    return (rank_ok and bfrac >= 0.80 and rank_arm == barm), rank_arm, bfrac

for phase in ("HI","LO"):
    cv = cell_values(phase)
    print(f"\n{'='*96}\n[{phase}]  통제: 모델·워크로드·빌드·arm 동일 / 변인: 요약통계만\n{'='*96}")
    print(f"{'요약통계':>18} {'U 효과(max-min)':>16} {'쌍대응 잡음SD':>14} {'효과/잡음':>10} "
          f"{'K1 식별':>8} {'도너':>6} {'boot':>6}")
    for name in SUMS:
        vals = {(a,b): cv[(name,a,b)] for a in ARMS for b in BLK if (name,a,b) in cv}
        arm_mean = {a: st.fmean([vals[(a,b)] for b in BLK if (a,b) in vals]) for a in ARMS}
        eff = max(arm_mean[a] for a in U) - min(arm_mean[a] for a in U)
        # 블록 쌍대응 잡음: 같은 블록 안 U-arm 편차의 블록 간 SD (블록 주효과 제거)
        dev = []
        for b in BLK:
            base = st.fmean([vals[(a,b)] for a in U if (a,b) in vals])
            for a in U:
                if (a,b) in vals: dev.append(vals[(a,b)] - base - (arm_mean[a] - st.fmean([arm_mean[x] for x in U])))
        noise = st.stdev(dev) if len(dev) > 1 else float("nan")
        lower_better = (name != "stall 분율(>45ms)") or True   # 전부 낮을수록 좋음
        ident, donor, bf = k1_identified(vals, U, lower_is_better=True)
        r = eff/noise if noise and not math.isnan(noise) else float("nan")
        print(f"{name:>18} {eff:16.4f} {noise:14.4f} {r:10.2f} {str(ident):>8} {donor or '-':>6} {bf:6.3f}")
