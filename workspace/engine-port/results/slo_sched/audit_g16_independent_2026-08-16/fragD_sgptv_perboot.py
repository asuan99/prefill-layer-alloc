# Fragment D — sgptv per-boot 1차 결정량 (검정력 근거)
#
# PROVENANCE: claims-auditor 독립 재구현, 2026-08-16. 원문 그대로 — 리팩터링 금지.
# RUN: cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched \
#      && python3 audit_g16_independent_2026-08-16/fragD_sgptv_perboot.py
# 산출: HI d34 [58.94, 59.22, 57.65, 59.59] vs d44 [57.24, 58.75, 59.34, 59.26]
#       = 완전 중첩. d34→d44 격차 0.20 ms 대 부팅 내 산포 1.5–2.1 ms.
# 소비처: PREREG_G16_RULES_REV3_2026-08-16.md §6 검정력·δ(K7)·ITL_SATURATED 판정의 근거

import json,glob,statistics
def pctl(v,q):
    o=sorted(float(x) for x in v)
    p=(len(o)-1)*q; lo=int(p//1); hi=min(lo+1,len(o)-1)
    return o[lo]+(o[hi]-o[lo])*(p-lo)
def M(fn):
    tt=[];p95=[]
    for ln in open(fn):
        ln=ln.strip()
        if not ln: continue
        r=json.loads(ln); T=r.get("ttfts") or []; I=r.get("itls") or []
        for i,t in enumerate(T):
            tok=[1000.0*x for x in (I[i] if i<len(I) else [])]
            tt.append(1000.0*t); p95.append(pctl(tok,0.95) if tok else float('inf'))
    return statistics.median(tt), statistics.median(p95)
for ph,tag in (("LO","sgptvLo"),("HI","sgptvHi")):
    print(f"--- {ph}: per-boot M_itl (median per-request ITL p95, ms) / M_ttft")
    for arm in ("d16","d24","d34","d44"):
        vals=[M(f) for f in sorted(glob.glob(f"{tag}_{arm}_rep*_L3H12_*.jsonl"))]
        print(f"  {arm}: M_itl {[round(v[1],2) for v in vals]}   M_ttft {[round(v[0],0) for v in vals]}")
