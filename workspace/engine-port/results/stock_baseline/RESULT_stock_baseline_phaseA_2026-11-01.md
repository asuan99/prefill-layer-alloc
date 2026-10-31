# 기본 서빙 구조 기준선 — Phase A 결과 (CPU 재집계, 2026-11-01)

> **지위**: 기존 원자료의 CPU 재집계. GPU 0 · 새 측정 0 · **성능 판정 0**. 아래 수치는 모두 기술값이다.
> 감사: claims-auditor r1 GO-with-edits(2026-11-01) — 요구 수정 7건 반영. 수치 독립 재계산: result-analyst `VERIFY_phaseA_2026-11-01.md` — 3건 일치(REAL), 지적 반영.
> 계획: `PLAN_stock_baseline_metrics_2026-11-01.md`(결정 62, 사용자 "진행" 2026-11-01: §6 권고 4건 채택).
> 재현: `python3 build_stock_table.py` · `python3 r_ser_holb.py` · `python3 r_ser_forecast.py` (이 디렉터리),
> 테스트 `tests/test_stock_metrics.py` 11건. 원자료는 전부 로컬 untracked(P1-opint jsonl 24 · HOLB 853 · M3 J1 17 GB).

## 0. 산출물

| 파일 | 내용 |
|---|---|
| `STOCK_METRICS_TABLE.csv` | 244행: P1-opint(plain = stock 기본, agnostic = PD-mux 대조) rep별 + pooled · M3 J1 4 arm × 셀 seed별 + pooled · vLLM 0.22.1 이력 12행. 열 = 시간(duration 합산, E2E p50/95/99)·req/s·out/total tok/s·TTFT·TPOT p50/95/99·요청 내부 ITL p95 중앙값·goodput·인용 상태 |
| `REPRO_P1OPINT.txt` | A2 재현 대조 16셀 **전부 일치** |
| `R_SER_HOLB.csv` / `R_SER_HOLB_SUMMARY.csv` | A3 r_ser·r_mix(파일×rate 구간 355행 / 묶음 37행). 구간별 CSV는 `r_ser_holb.py`로 재생성하는 로컬 파일(미추적: 숫자 부분열이 인용 정지 패턴과 우연히 겹침), SUMMARY만 추적 |
| `R_SER_FORECAST_NANO.csv` | A5 Nano stock SGLang r_ser 사전 예보 격자 |

정의는 `stock_metrics.py` 머리말 하나로 통일했다. TPOT = 요청별 평균 ITL이고, ITL이 없는 완료 요청은 +∞로 둔다
(p1op_analyze 규약 — 이걸 따라야 원문과 일치). goodput = TTFT ≤ SLO AND 요청 내부 ITL p95 ≤ SLO(게이트 4).

## 1. A2 — P1-opint 재현 (KISTI, cudagraph ON, random 2000/96, n=5)

같은 원자료를 새 추출기로 다시 계산해 원문 `P1_OPINT_RESULT_2026-08-05.md` §3.1과 대조했다:
req/s·TTFT p50/95/99·TPOT p50/95(rep 평균) **16셀 모두 반올림 범위 안에서 일치**하고, goodput(게이트 4 술어)도
원문 "정본 p95" 열과 같다. 따라서 이 추출기는 기존 결과와 같은 숫자를 낸다.

새로 붙은 열(원문 요약에 없던 것, pooled = 5 rep 합산·duration 합산):

| 모델 · arm | rate | duration s | req/s | out tok/s | total tok/s | E2E p50 / p99 s | TTFT p50 / p99 s | TPOT p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| Granite · plain(stock) | 2 | 312 | 1.92 | 185 | 4,029 | 0.95 / 1.65 | 0.123 / 0.295 | 8.2 / 13.8 |
| Granite · plain | 3 | 206 | 2.92 | 280 | 6,109 | 1.10 / 2.34 | 0.126 / 0.371 | 10.0 / 19.0 |
| Granite · plain | 4 | 152 | 3.96 | 380 | 8,291 | 1.37 / 2.61 | 0.158 / 0.394 | 12.5 / 20.9 |
| Granite · plain | 6 | 107 | 5.63 | 540 | 11,797 | 2.30 / 5.88 | 0.203 / 0.999 | 21.8 / 51.3 |

- E2E는 이 원천에 요청별 send/finish 시각이 없어 **TTFT + ΣITL 재구성**이다(`e2e_source = ttft+itl`; bench `median_e2e_latency_ms`와 상대 편차 < 1e-4).
- **ITL 축 metric cliff**: plain rate 4·6의 요청 내부 ITL p95 중앙값이 52·69 ms로 SLO 60 ms 근처다 — goodput 열은 이 임계값에 민감하다(TPOT p95로는 안 보임). goodput 인용 시 병기.
- TPOT = +∞ 요청은 완료 9,600건 중 24건(96토큰이 한 번에 도착) — 해당 rep의 TPOT p99가 ∞.
- Zamba2 행도 표에 있으나 결정 46(Zamba 배제)으로 **인용만**, plain r4·r6과 agnostic r6은 절벽 셀이라 인용 금지.
- 이 기판(KISTI)·워크로드 행을 VESSL·Nano 행과 섞어 비교하지 않는다. fused는 **미조율**이다.

## 2. A4 — M3 J1 (VESSL A100-SXM, Nano-9B-v2-Base, flashinfer, cudagraph ON, N=1 Job)

현 기판·모델에서 **기본 구조(fused = chunk 8192·overlap ON; fused_sym = chunk −1·overlap OFF)의 유일한 서빙 원자료**다.
LO 셀만 옮긴다(HI 셀 TTFT는 인용 금지 C:8792, X shape는 `X_CORRECTNESS_FAIL`로 비 미정의).

| arm | shape · rung | req/s | out tok/s | E2E p50 s | TTFT p50 / p95 s | TPOT p50 / p95 ms | ITL p95(요청) 중앙 ms |
|---|---|---|---|---|---|---|---|
| fused | A(256/512) LO | 1.88 | 963 | 9.93 | 0.078 / 0.123 | 19.3 / 22.5 | 19.6 |
| fused_sym | A LO | 1.87 | 959 | 10.82 | 0.059 / 0.083 | 21.1 / 24.3 | 21.1 |
| agnostic | A LO | 1.87 | 956 | 10.93 | 0.096 / 0.153 | 21.2 / 24.7 | 35.0 |
| d44 | A LO | 1.87 | 958 | 10.77 | 0.091 / 0.152 | 20.9 / 23.7 | 31.3 |
| fused | B(8192/64) LOx | 0.66 | 43 | 4.10 | 1.06 / 2.96 | 40.6 / 152.8 | 15.8 |
| fused_sym | B LOx | 0.66 | 42 | 4.43 | 1.11 / 3.65 | 42.5 / 154.2 | 16.0 |
| agnostic | B LOx | 0.66 | 42 | 4.56 | 2.98 / 7.45 | 25.0 / 25.7 | 27.1 |
| d44 | B LOx | 0.66 | 42 | 6.54 | 5.25 / 11.65 | 20.9 / 21.0 | 23.3 |

- 셀당 seed 2개, Job 1개. CI·검정력 없음 — **arm 순위 서술 금지**, 설계 입력용 기술값.
- 관찰(기술): shape B에서 fused 계열은 **TPOT(평균 ITL) p50 ≈ 41 ms인데 요청 내부 ITL p95 중앙은 ≈ 16 ms**다.
  출력 64토큰 요청에서 8k prefill 몇 번이 만든 큰 정지가 평균에는 들어가고 p95 밖으로는 빠지기 때문이다.
  **TPOT와 ITL p95를 서로 대신 쓰면 안 된다**는 사례다. 산술상 출력 64토큰(ITL 63개)에서 요청 내부 p95는 상위 약 3개를 버리므로, **게이트 4 술어는 요청당 정지 3회 이하를 크기와 무관하게 보지 못한다**. 기전(정지가 8k prefill이라는 것)은 미검증, N=1 Job·seed 2.

## 3. A3 — stock SGLang의 r_ser·r_mix (Gate 2 HOLB, KISTI, cudagraph ON, in2000/out96)

정의(계획 §0-1): r_ser = 두 decode 사이 같은 stream의 prefill forward GPU 시간 합 ÷ 같은 구간·같은 bs의 순수 decode
forward GPU 시간 중앙값. r_mix = (MIXED forward − T_dec) ÷ T_dec. **PD-mux r_span과 다른 양 — 같은 열에 합치지 않는다.**

| 모델 · arm (설정) | rate | T_dec 중앙 ms | 정지 1회 prefill ms(p50) · forward 수 | r_ser p50 / p95 | r_mix p50 / p95 |
|---|---|---|---|---|---|
| Granite · plain (chunk 8192, mixed OFF, overlap ON = **SGLang 기본**) | 3 / 4 / 6 | 6.8 / 6.9 / 7.5 | 105 · 1 | 15.4 / 42.5 · 15.1 / 43.5 · 14.1 / 75.8 | — |
| Granite · plainaux (chunk −1, overlap OFF) | 3 / 4 / 6 | 7.3 / 7.8 / 8.0 | 105 · 1 | 14.5 / 50.5 · 14.4 / 61.2 · 14.8 / 97.6 | — |
| Granite · plainmix (chunk 8192, mixed ON) | 3 / 4 | 6.8 / 7.3 | 105 · 1 | 16.3 / 20.7 · 16.3 / 18.3 (정지 133·53건) | 18.4 / 35.0 · 19.7 / 63.4 |
| Granite · chunk512mix | 3 / 4 | 6.8 / 7.1 | 250 · 5 | 39 / 58 · 36 / 39 (29·21건) | 33.3 / 37.2 · 32.1 / 37.3 |

(plain r3·r4·plainmix·chunk512mix = g2ea, plainaux·plain r6 = g2, 파일 10개씩. g2·g2holb 값은 r_ser p50 기준 ±0.4 안이며 p95는 최대 ±3 차이 — `R_SER_HOLB_SUMMARY.csv`.)

- **in2000 고정 길이 워크로드에서** prefill이 끼는 정지는 전부 r_ser > 1이다(prompt 1개 prefill ≈105 ms ≫ T_dec ≈7 ms의 산술 귀결이지 구조 성질이 아니다). p95가 rate와 함께 커지는 원인(한 EXTEND 배치의 다중 prompt 또는 연속 forward)은 이 집계로 구분되지 않는다 — 미검증.
- HOLB 프로브 관측자 효과는 미결정(G5 UNDETERMINED, C:3125-3140)이고 `same_stream_fw_ms`는 설계상 진단 전용이다 — T_dec·prefill ms 절대값을 비계측 런(P1-opint 등)과 비교하지 않는다.
- mixed ON은 정지를 MIXED forward 안으로 옮긴다: 따로 서는 정지는 크게 줄고(Granite g2ea 972 → 133건) 대신 MIXED decode
  step이 T_dec의 약 19배가 된다. 이 경우 r_mix ≈ r_ser이다(18–20 대 16). 이 정지가 client ITL에 어떻게 실리는지는 client ITL로 확인하지 않았다.
- **chunk512(mixed OFF)는 현 분할(decode 이벤트 시각의 큰 공백)로 rate 구간 배정이 실패**한다(seg_check FAIL 26/355가 전부 chunk512): run 안 decode 침묵이 rate 사이 휴지보다 길 수 있다. bench duration 기준으로 끝에서 경계를 잡으면 맞는다(result-analyst 검증). FAIL 구간은 SUMMARY 풀에서 뺐고, chunk512 r_ser는 분할·분모 정의에 따라 10–60% 흔들려 rate별 값을 인용하지 않는다.
- Zamba2 행(plain r_ser p50 ≈ 15, prefill 142 ms / T_dec 8.6–9.7 ms)은 표에 있으나 인용만(결정 46).

## 4. A5 — Nano stock SGLang r_ser 사전 예보 (Phase B 봉인용)

r̂_ser(L, bs, ctx) = T2(L; 108) ÷ T1(bs, ctx; 108v), 계수는 비용 항 캠페인 vanilla `108v`(T1 F1 CONFIRMED(scoped),
T2 q 라벨 CONFIRMED). 예: L = 2k에서 bs 1 → 17.5, bs 32·ctx 16k → 8.7; L = 16k에서 bs 1 → 137, bs 48·ctx 16k → 54.
**편향**: T2는 클라이언트 TTFT(호스트 u≈12 ms 포함), T1은 엔진 step 시간(S_eng)이고 HOLB r_ser는 CUDA-event GPU 시간이다 — 순 편향 방향은 미정. 계수는 context 131,072 boot 값(cost-terms §2 승계 문장). L < 1k는 T2 격자 밖 외삽. 이 표는 Phase B PREREG에 봉인한 뒤 채점하며, 그 전에는 인용하지 않는다.

## 5. 남은 공백 (Phase B가 채울 것)

1. VESSL·Nano·W1/W3·변화 trace에서 stock SGLang 서빙 측정 n ≥ 4 — 없음(M3 J1은 N=1·다른 shape).
2. 현 기판의 stock vLLM — 없음(0.22.1 이력 12행은 `HISTORY ONLY`).
3. Nano에서 측정한 r_ser·r_mix — 없음. `decode_probe`는 EXTEND forward 시간을 재지 않으므로 기본 OFF 플래그 추가 필요(결정 62 ④).
4. 요청별 send/finish 시각은 M3 계열에만 있다. 새 측정은 처음부터 기록한다.

## 6. cost model 기여

- KISTI Granite(in2000, 미조율 fused, HOLB 계측)에서 r_ser p50 ≈ 15였다[기술값]. r_ser는 L과 스케줄러의 chunk 연속 실행에 의존하므로 '구조적 고정'이 아니며, PD-mux r_span과 크기 비교하지 않는다.
- Nano r_ser 예보(§4)는 Phase B에서 같은 GPU-event 단위로 T1·T2를 함께 기록해 채점한다(감사 r1 권고: stock plain L ∈ {512, 2k, 8k} × 용량 아래 rate, n ≥ 4, HOLB/probe ON vs OFF 짝으로 관측자 효과 TOST ±3%와 r_ser(L)를 함께 측정).

## 7. 인용 규칙

- 허용: "이 원자료를 같은 추출기로 다시 계산하면 원문 §3.1과 일치한다", "KISTI Granite stock SGLang(in2000)에서 prefill
  정지는 decode step 약 15개분이었다(r_ser p50, HOLB, 기술값)".
- 금지: KISTI 행과 VESSL 행 수치 비교 · M3 J1 arm 순위 · chunk512 rate별 r_ser · r_ser/r_mix를 r_span과 같은 축에 놓는 서술 ·
  vLLM 0.22.1 행을 대조군 근거로 사용 · A5 예보를 측정값처럼 인용 · in2000 결과를 'stock 구조에서 r ≫ 1은 고정'으로 일반화 ·
  HOLB T_dec 절대값을 비계측 런과 비교 · ITL-p95 goodput이 out=64에서 stall을 본다는 서술.
