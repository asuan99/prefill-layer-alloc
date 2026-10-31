# Phase A 독립 검증 (result-analyst, 2026-11-01, CPU only)

독립 코드(numpy 백분위, 자체 분할; 기존 스크립트 미사용)로 재계산. 기존 파일은 수정하지 않았음.

## 1. P1-opint granite plain, 5 rep pooled, duration 합산 — 일치
| rate | dur s | req/s | out tok/s | TTFT p50/p99 s | TPOT p50 ms | goodput (3 s / 60 ms) | 표 |
|---|---|---|---|---|---|---|---|
| 4 | 151.688 | 3.9555 | 379.73 | 0.15805 / 0.39437 | 12.530 | 3.1116 | 전 항목 일치 |
| 6 | 106.607 | 5.6282 | 540.30 | 0.20260 / 0.99938 | 21.787 | 2.0824 | 전 항목 일치 |
dur = rep 길이 5개의 합(max 아님). 두 셀 모두 ITL 없는 요청은 0건.

## 2. M3 J1 fused B LOx (s4162+s4386) — 일치
n=220, dur 331.217 s(summary.json duration_s 합), req/s 0.66422, out 42.51 tok/s, E2E p50 4.1002 s(측정값),
TTFT p50/p95 1.0597/2.9597 s, TPOT p50/p95 40.56/152.75 ms, ITL p95(요청) 중앙값 15.82 ms. 표 pooled 행과 일치.

## 3. r_ser g2ea granite plain r3 — 일치, 분할 방식은 별도 지적 있음
자체 분할(bench duration 기준으로 끝에서부터 경계를 잡고, 예상 시점 ±30% 안에서 가장 큰 host gap을 고름): n=972,
p50 15.438, p95 42.199, frac>1 = 1.0 → SUMMARY와 동일. 구간 길이/bench duration = 1.02–1.05(20/20).
분모 대안(구간 전체 T_dec 중앙값): p50 15.48 → plain에서는 분모 선택에 둔감함.
**chunk512:** 내 방식은 g2ea granite 20/20 구간을 0.96–1.04로 맞춘다(전 chunk512 그룹: 0.75–1.04).
FAIL은 run 안 decode 침묵(rep3 r4: 10.14 s)이 rate 사이 휴지(10.04 s)보다 길어서 생긴다. 따라서 "decode 시각만으로는
경계를 못 찾는다"는 bench duration을 쓰면 성립하지 않는다. FAIL 행은 SUMMARY 풀에 **그대로 포함**되어 있다
(r3 p50 57.75 vs 수정 분할 55.22; r4 p50 67.60 vs 75.12, p95 493 vs 477). chunk512는 분모에도 민감하다
(g2 granite r6: bs 매칭 213 vs 전역 130). RESULT는 chunk512 rate별 값을 이미 인용 금지로 두었으므로 headline 영향 없음.

## 4. 정의 점검
- duration 합산(게이트 7): P1-opint·M3 둘 다 확인.
- TPOT=+inf: 완료 9600건 중 24건이 ITL 없음(out_len 96, TTFT 1.1–2.7 s ≈ 한 번에 도착한 E2E). 그 TTFT는 사실상 E2E라서
  TTFT p99가 조금 부풀고 goodput에서는 bad로 센다(보수적). +inf 행 12개 = 해당 rep의 TPOT p99. 검증한 셀에는 영향 없음.
- E2E 재구성: 각 bench row에서 TTFT+ΣITL 중앙값 = bench median_e2e_latency_ms(상대 편차 <1e-4). 재구성이 원천과 맞음.
- 사소한 오류: RESULT §3 "973 → 133건"은 캠페인을 섞었다(973은 g2, g2ea plain r3는 972, 133은 g2ea).
- metric cliff(ITL 축): granite plain r4/r6의 ITL p95(요청) 중앙값은 52/69 ms로 SLO 60 ms 근처다. goodput/req_s = 0.79/0.37.
  goodput 열은 ITL 임계값 섭동에 민감하다. TPOT p95(20.9 ms)로는 이 점이 드러나지 않는다.

## cost model 기여
CM-1/CM-2(prefill 정지 ÷ decode step) — KISTI granite stock plain에서 r_ser p50 = 15.4(n=972)를 독립 분할로 재확인했고,
분모 정의(bs 매칭 vs 전역)에 대해 강건하다(±0.3%). 깨진 가정: chunk512 r_ser는 분할·분모 정의에 따라 ±10–60% 흔들려서,
"chunk 크기 → 정지 분할" 항의 계수로는 아직 쓸 수 없다. 범위: KISTI A100, in2000/out96, cudagraph ON.
