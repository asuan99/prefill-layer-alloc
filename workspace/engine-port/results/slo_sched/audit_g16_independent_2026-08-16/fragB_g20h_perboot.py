# Fragment B — g2_0_hard 확장격자 per-boot (±SD)
#
# PROVENANCE: claims-auditor 독립 재구현, 2026-08-16. 원문 그대로 — 리팩터링 금지.
# RUN: cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/g2_0_hard \
#      && python3 <이 파일이 있는 경로>/fragB_g20h_perboot.py
#      (상대 글롭 `D="."` 이므로 반드시 g2_0_hard에서 실행)
# ⚠️ A와 동일하게 `json.load()` — ROUNDS=1 전용.
# ⚠️ 빈 ITL 요청을 **드롭**한다(A=9e9, C/D/E=inf와 규약 불일치).
#    정본 `RequestResult.passes`는 percentile(())=nan -> 실패이므로 inf(C/D/E)가 정본 정합이다.
#    이 데이터셋에서는 결과가 바뀌지 않았다(A와 B의 ITLp95med 일치)나, 규약 불일치를 명시한다.
# 소비처: PREREG_G16_RULES_REV3_2026-08-16.md §3 기준1 표(±SD)

import json,glob,statistics
D="."
def pctl(a,q):
    a=sorted(a); k=(len(a)-1)*q; f=int(k); c=min(f+1,len(a)-1)
    return a[f]+(a[c]-a[f])*(k-f)
def s(fn):
    o=json.load(open(fn)); tt=[t*1000 for t in o["ttfts"]]; itl=o["itls"]
    p95=[pctl([x*1000 for x in I],0.95) for I in itl if I]
    return statistics.median(tt), statistics.median(p95), len(o["ttfts"])/o["duration"]
for ph in ("A","B"):
    print(f"--- phase {ph}: per-boot (TTFTmed ms, ITLp95med ms, thr)")
    for arm in ("d34","d44","d54","d64","d74"):
        rows=[s(f) for f in sorted(glob.glob(f"g20h{ph}_{arm}_rep*_rA5B4_OB1024_*.jsonl"))]
        tt=[r[0] for r in rows]; il=[r[1] for r in rows]; th=[r[2] for r in rows]
        print(f"  {arm}: TTFTmed {statistics.mean(tt):8.0f}±{statistics.stdev(tt):7.0f} | ITLp95med {statistics.mean(il):6.2f}±{statistics.stdev(il):5.2f} | thr {statistics.mean(th):.3f}±{statistics.stdev(th):.3f}")
