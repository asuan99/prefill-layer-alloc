# Fragment C — sgptv 구 4-arm 격자 독립 재계산 (C6 표적 + ORACLE §3-2 교차검증)
#
# PROVENANCE: claims-auditor 독립 재구현, 2026-08-16. 원문 그대로 — 리팩터링 금지.
# RUN: cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched \
#      && python3 audit_g16_independent_2026-08-16/fragC_sgptv_recompute.py
#      (상대 글롭이므로 반드시 slo_sched에서 실행)
# ★교차검증: 이 스크립트의 HI 블록 TTFTpass%/ITLpass% 열이
#   ORACLE_REANALYSIS_2026-08-16.md §3-2의 57.083/66.375/68.917/70.708 ·
#   18.292/46.167/92.125/93.458 을 소수 셋째 자리까지 재현한다 —
#   이것이 이 독립 구현의 정당성 근거다(pdmux_eval 미사용).
# 규약: 줄 단위 루프(다중 라운드 처리) + duration 합산(게이트 #7) + 빈 ITL = inf(정본 정합).
# 소비처: PREREG_G16_RULES_REV3_2026-08-16.md §7 PC-E 표적(M_ttft·M_itl) · §4(1) 처리량 대조

import json,glob,statistics
# INDEPENDENT reimplementation (does not import pdmux_eval); linear-interp percentile
def pctl(v,q):
    o=sorted(float(x) for x in v)
    if not o: return float('nan')
    p=(len(o)-1)*q; lo=int(p//1); hi=min(lo+1,len(o)-1)
    return o[lo]+(o[hi]-o[lo])*(p-lo)
def boot(fn):
    tt=[];p95=[];dur=0.0
    for ln in open(fn):
        ln=ln.strip()
        if not ln: continue
        r=json.loads(ln); dur+=float(r.get("duration") or 0)
        T=r.get("ttfts") or []; I=r.get("itls") or []
        for i,t in enumerate(T):
            tok=[1000.0*x for x in (I[i] if i<len(I) else [])]
            tt.append(1000.0*t); p95.append(pctl(tok,0.95) if tok else float('inf'))
    n=len(tt)
    return dict(M_ttft=statistics.median(tt), M_itl=statistics.median(p95), n=n, dur=dur,
                thr=n/dur, tpass=100*sum(1 for x in tt if x<=3000)/n,
                ipass=100*sum(1 for x in p95 if x<=60)/n)
for ph,tag in (("LO","sgptvLo"),("HI","sgptvHi")):
    print(f"=== {ph} (independent recompute, n=4 boots/arm) ===")
    print(f"{'arm':>4} {'M_ttft ms (med TTFT)':>26} {'M_itl ms (med per-req ITLp95)':>32} {'thr req/s':>12} {'TTFTpass%':>10} {'ITLpass%':>9}")
    for arm in ("d16","d24","d34","d44"):
        fs=sorted(glob.glob(f"{tag}_{arm}_rep*_L3H12_*.jsonl"))
        rs=[boot(f) for f in fs]
        f=lambda k:(statistics.mean(r[k] for r in rs), statistics.stdev([r[k] for r in rs]))
        a=f('M_ttft'); b=f('M_itl'); c=f('thr'); d=f('tpass'); e=f('ipass')
        print(f"{arm:>4} {a[0]:>16.1f} ± {a[1]:<7.1f} {b[0]:>20.3f} ± {b[1]:<9.3f} {c[0]:>7.3f}±{c[1]:.3f} {d[0]:>9.3f} {e[0]:>8.3f}   (files={len(fs)})")
