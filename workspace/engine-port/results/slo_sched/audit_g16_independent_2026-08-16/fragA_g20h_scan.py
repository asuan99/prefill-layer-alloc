# Fragment A — g2_0_hard 확장격자 스캔 (arm 평균 + pass율)
#
# PROVENANCE: claims-auditor 독립 재구현, 2026-08-16 (G16 사전등록 rev1 규칙 감사).
#   원문 그대로 보존 — 리팩터링 금지. `pdmux_eval`을 import하지 않는 것이 존재 이유다
#   (양성대조의 독립성). `percentile`은 정본과 같은 선형보간을 독립 구현한 것.
# RUN: python3 fragA_g20h_scan.py          (경로 절대, cwd 무관)
# ⚠️ `json.load()`(파일당 JSON 객체 1개)라 ROUNDS=1 아티팩트 전용 —
#    sgptv(3라운드/파일)에 걸면 깨진다. 빈 ITL 요청 규약 = 9e9 (C/D/E는 inf).
# 소비처: PREREG_G16_RULES_REV3_2026-08-16.md §3 기준1 표 · §7 PC-D 표적(pass율·joint%)

import json,glob,statistics,re,os
D="/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/g2_0_hard"
def pctl(a,q):
    a=sorted(a)
    if not a: return float('nan')
    k=(len(a)-1)*q; f=int(k); c=min(f+1,len(a)-1)
    return a[f]+(a[c]-a[f])*(k-f)
def summarize(fn):
    o=json.load(open(fn))
    tt=o["ttfts"]; itl=o["itls"]; dur=o["duration"]; n=len(tt)
    ttft_ms=[t*1000 for t in tt]
    p95=[pctl([x*1000 for x in I],0.95) if I else 9e9 for I in itl]
    meanitl=[ (sum(I)/len(I))*1000 if I else 9e9 for I in itl]
    tpass=sum(1 for t in ttft_ms if t<=3000)/n
    ipass=sum(1 for v in p95 if v<=60)/n
    joint=sum(1 for t,v in zip(ttft_ms,p95) if t<=3000 and v<=60)/n
    return dict(n=n,dur=dur,thr=n/dur,ttft_med=statistics.median(ttft_ms),
                tpass=tpass*100,ipass=ipass*100,joint=joint*100,
                itlp95_med=statistics.median(p95), meanitl_med=statistics.median(meanitl))
for phase in ("A","B"):
    print(f"===== g2_0_hard phase {phase} (rA5B4 OB1024; A=in2048/o32@5, B=in2048/o1024@4) =====")
    print(f"{'arm':>5} {'n_boot':>6} {'thr req/s':>10} {'TTFTmed ms':>11} {'TTFTpass%':>10} {'ITLp95med':>10} {'ITLpass%':>9} {'joint%':>7}")
    for arm in ("d34","d44","d54","d64","d74"):
        fs=sorted(glob.glob(f"{D}/g20h{phase}_{arm}_rep*_rA5B4_OB1024_*.jsonl"))
        if not fs: continue
        rows=[summarize(f) for f in fs]
        m=lambda k: statistics.mean(r[k] for r in rows)
        sd=lambda k: statistics.stdev([r[k] for r in rows]) if len(rows)>1 else 0.0
        print(f"{arm:>5} {len(fs):>6} {m('thr'):>10.3f} {m('ttft_med'):>11.0f} {m('tpass'):>10.1f} {m('itlp95_med'):>10.1f} {m('ipass'):>9.1f} {m('joint'):>7.1f}   (sd thr {sd('thr'):.3f}, sd ttftmed {sd('ttft_med'):.0f})")
