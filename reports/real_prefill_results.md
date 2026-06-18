# Real-Prefill 결과 — micro vs full 비교 분석 (A2 close-out)

작성일: 2026-06-18 · 브랜치 `exp/e5-real-prefill` · 측정 job **775529** (E5 full, 4모델)
설정: `--prefill-mode full --prefill-tokens 4096` → prefill = **GEMM-inclusive(ssm_full/attn_full) × 16 chunk(16×256)**, prefill_batch=1. **sweep 축은 v2 widened와 동일**(40셀/모델, decode_batch {1..512}, ctx 4096, fracs {0.5,0.7}). 비교: `compare_micro_vs_full.py` → `results_v2/e5/compare_micro_full_*.csv`.
관련: [설계](real_prefill_experiment_design.md) · [widened 검증](widened_sweep_validation.md) · [closure §5.1](project_closure_report.md) · [검수 A2](review_checklist.md)

---

## TL;DR — A2가 두 결론을 바꾼다

real prefill(prefill solo ~0.5ms→**8.5–9ms**, ~16×)로 재면:

1. **양성 결론(overlap ~2×)은 microbench 산물이었다 — *operating regime이 뒤집힌다*.** 저배치에서 2.0×→**1.05×로 붕괴**, 대신 **고배치에서 1.6–1.76×로 *되살아난다*.** v2 §4.6-C "window는 decode batch로 닫힌다"는 microbench 한정이고, real prefill에선 **window가 고배치서 *열린다*.** 메커니즘은 v2가 말한 **duration matching**이 정확히 맞음 — 2× 봉우리가 prefill 크기에 따라 *이동*할 뿐.
2. **음성 결론(분할 무용)에 진짜 예외 발생.** 3/4 모델은 green_ctx 여전히 패(falcon1.5b·zamba1.2b 0셀, falcon3b 1셀 +0.1% 노이즈). **그러나 zamba2_2.7b에서 green_ctx가 two_stream을 +12.5% 앞선다**(pf=attn×dec=ssm×db256). 노이즈 아님 — 대형모델+real prefill+고배치 = closure §5.1이 미검증으로 남긴 MuxWise/Bullet regime이 *실제로 나타났다.* **단 이 승리는 throughput뿐**(decode_inflation 80.9% vs two_stream 4% → SLO로는 패).

---

## 1. C2/C3 — overlap: 2×는 microbench 산물, 봉우리가 고배치로 이동

window 곡선 (pf=ssm×dec=ssm, speedup_vs_seq), **micro → full**:

| db | 1 | 8 | 32 | 64 | 128 | 256 | **512** |
|---|--|--|--|--|--|--|--|
| falcon_h1_1.5b | 2.04→**1.05** | 2.01→1.05 | 1.89→1.05 | 1.61→1.11 | 1.35→1.18 | 1.18→1.31 | 1.09→**1.63** |
| falcon_h1_3b | 2.02→**1.06** | 1.99→1.06 | 1.89→1.06 | 1.54→1.11 | 1.31→1.17 | 1.16→1.33 | 1.08→**1.68** |
| zamba2_1.2b | 2.01→**1.05** | 1.96→1.07 | 1.74→1.08 | 1.53→1.12 | 1.30→1.21 | 1.16→1.38 | 1.08→**1.76** |
| zamba2_2.7b | 2.05→**1.04** | 1.97→1.06 | 1.72→1.09 | 1.55→1.12 | 1.32→1.18 | 1.16→1.35 | 1.08→**1.64** |

**duration matching 직접 증거:** full prefill solo ≈ 8.5–9.0ms (고정). decode solo: db1 ≈ 0.5ms → db512 ≈ 6ms.
- **db1:** prefill(8.5) ≫ decode(0.5) → 불균형 → decode가 trivial하게 숨어 speedup ≈ 1.05.
- **db512:** prefill(8.5) ≈ decode(6) → 균형 → 거의 완전 overlap → speedup ≈ 1.6–1.76.

→ 2× 봉우리는 *사라진 게 아니라* prefill 크기에 맞는 batch로 **이동**했다. microbench(작은 prefill)는 저배치에서, real prefill(큰 prefill)은 고배치에서 봉우리. **v2의 "window closes" 결론은 prefill 크기에 종속된 artifact였다.**

## 2. C1 — partition: 3/4 유지, zamba2_2.7b에서 진짜 예외

`max(two_stream_ms/green_ctx_ms)` (≥1.0 = green_ctx 승), micro → full:

| model | micro max(≥1셀) | full max(≥1셀) | 판정 |
|---|--|--|---|
| falcon_h1_1.5b | 0.995 (0) | 0.998 (0) | green_ctx 패 유지 |
| zamba2_1.2b | 1.002 (1, 노이즈) | 0.992 (0) | green_ctx 패 (오히려 깨끗) |
| falcon_h1_3b | 1.002 (1) | 1.001 (1, +0.1% 노이즈) | 사실상 패 |
| **zamba2_2.7b** | 1.004 (1, 노이즈) | **1.125 (2)** | **green_ctx 승 — 진짜 예외** |

zamba2_2.7b 승리 셀 상세 (full):
- **pf=attn dec=ssm db256:** seq 6.67 / two_stream 6.38 / **green_ctx@f=0.5 5.67ms (green +12.5%)** / green@f=0.7 8.92ms(패).
- **pf=attn dec=attn db64:** two_stream 7.33 / **green_ctx@f=0.5 7.10ms (green +3.2%)** / green@f=0.7 11.5ms(패).

**메커니즘:** 둘 다 **pf=attn**(GEMM/compute-heavy, SM-포화)이 **고배치 decode**와 동시 실행될 때다. 동적 공유(two_stream)는 두 큰 커널이 **파괴적으로 경쟁** → 균형 정적 분할(54/54)이 격리해 회수 → 12.5% 승. 이게 closure §3.1이 말한 *"동적 공유가 회수 못 하는 slack"(술어 b)의 양성 사례*이고, §5.1이 미검증으로 남긴 영역(대형·real prefill·고배치)에서 **실제로 발생**했다.

**그러나 — throughput 승, SLO 패 (결정적 단서):** 그 12.5% 승리 셀에서 green_ctx@f=0.5의 `decode_inflation`은 **80.9%** (two_stream 4.0%). 즉 합산 벽시간은 이기지만 **decode 지연을 80% 악화**시킨다. SLO 제약 serving이라면 이건 *패*다. f=0.7(prefill 우대)은 벽시간조차 패(0.75×)하고 decode 185% 악화. → "green_ctx가 이긴다"는 *metric 의존*이며, MuxWise/Bullet식 **decode-보호** 분할과는 방향이 다르다(여기 균형 분할은 decode를 *희생*해 throughput을 삼).

## 3. A5 — green_ctx는 real prefill에서도 decode를 굶긴다

decode_inflation_pct mean, micro → full:

| model | two_stream | green_ctx |
|---|--|--|
| falcon_h1_1.5b | 2.7→5.3 | 64.3→**68.9** |
| falcon_h1_3b | 3.0→3.6 | 64.4→**66.3** |
| zamba2_1.2b | 0.9→3.8 | 83.1→**85.6** |
| zamba2_2.7b | 1.5→4.5 | 83.0→**87.7** |

two_stream은 decode를 거의 안 건드림(3.6–5.3%), green_ctx는 real prefill에서도 **66–88% 굶김**(widened §1.3 재현). → green_ctx가 벽시간을 이기는 드문 셀조차 decode latency는 망가뜨린다.

## 4. 해석 — A2가 닫히며 두 결론을 정정

- **(overlap)** "co-schedule이 ~2× 공짜"는 *microbench·저배치* 한정. real prefill에선 저배치 overlap이 거의 사라지고(1.05×), 봉우리가 고배치(1.6–1.76×)로 이동. **운영 함의 정정:** hybrid serving에서 overlap 이득은 *prefill과 decode의 duration이 맞는 batch*에서 최대 — real prefill(긴)에선 그게 **고 decode batch**다. v2의 "고배치서 닫힘"은 뒤집힌다.
- **(partition)** "분할 무용"은 *SLM·microbench·throughput* 한정으로 더 좁혀야 한다. real prefill·대형모델(2.7b)·attn-prefill·고배치에서 **균형 정적 분할이 throughput을 최대 12.5% 회수** — closure §5.1·검수 A2가 정확히 예측한 regime. 단 그 회수는 **decode latency 80% 희생** 위에서이므로, *SLO 목적함수로는 여전히 분할이 불리*하다. 즉 음성 헤드라인은 **"SLO/decode-latency 기준으로 분할은 불리"**로 재정의하면 real prefill에서도 산다 — throughput-only 기준으로는 대형모델서 예외가 생긴다.

## 5. 다른 보고서에 반영할 것

1. **closure §2 표 / §4.2 / widened §1.2:** "window는 decode batch로 닫힘(2.04→1.15)"에 **"단 microbench 한정 — real prefill(job 775529)에선 저배치 1.05×로 붕괴·고배치 1.6–1.76×로 재출현, 봉우리가 이동"** 추가.
2. **closure §1/§2 / 검수 A1·A2:** "분할 의미있는 승리 0"에 **"단 microbench 한정 — real prefill·zamba2_2.7b서 green_ctx +12.5%(throughput) 발생, 그러나 decode 80% 희생이라 SLO로는 여전히 패"** 추가. A2 = **통과(조건부)**: 정성 결론은 *재정의 후* 유지, throughput-only 헤드라인엔 대형모델 예외.
3. **closure §5.1 강화:** MuxWise/Bullet regime이 *우리 데이터에서 실측으로 재현*됨(파괴적 간섭 회수). 단 우리 green_ctx는 decode-희생형이라 그들의 decode-보호형과 방향이 반대 — 둘 다 "real prefill·고배치에서 분할이 의미를 가진다"는 같은 결론.
4. **추가가치 Path 1/3:** 7B 미실행이지만, 2.7b에서 이미 예외가 나왔으므로 **크기-스케일링이 실재**(분할 이득이 모델 크기와 함께 커짐) — Path 1(7B)이 더 결정적이 됨.

## 6. 한계 (정직성)

- **multi-chunk는 timing proxy** — 16회 반복 호출이라 SSM state passing·attn intra-prefill KV 누적 미반영(설계 §5). prefill 절대 latency·SM 점유는 충실하나 chunk 간 의존 효과는 미포착.
- **prefill_batch=1**(full-C 미실행). prefill 배칭을 더하면 분할 예외가 더 커질 수 있음.
- **green_ctx f 격자 거침**(0.5/0.7뿐). decode-보호형 f(decode_sm≥floor) 미스윕 — SLO 관점 분할의 *상한*은 미측정(검수 A5).
- **A100-SXM4·4모델·ctx4096 고정.** 7B(Path 1)·다중 context 미검증.
