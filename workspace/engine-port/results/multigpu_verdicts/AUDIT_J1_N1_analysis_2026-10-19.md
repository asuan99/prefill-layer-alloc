# AUDIT — M3 J1 N=1 분석 (`ANALYSIS_J1_N1_2026-10-19.md`) claims-auditor 1회차 (2026-10-19, 원시 재계산 포함, 읽기 전용 판정의 기록)

> 이 폴더는 `results/multigpu/` 봉인 glob 밖이다(VERDICT_* 금지 규칙은 `results/multigpu/`에만 적용). 재감사는 이 판정 이후 diff만, 최대 1회.

| # | 문장 | 판정 |
|---|---|---|
| 1 | CM-6 "B·X에서 처리율(fused>PD-mux)과 ITL p99(PD-mux 25–37 ms vs fused 1.7–14.8 s) 역순" | **REFUTED(쓰인 그대로)** — HI 확장 rung, achieved 상이(fused 1.06–1.11 vs PD-mux 0.70–0.86) 과부하 비교 + 토큰 합산 p99(멈춤 꼬리). 게이트 4 지표로 achieved가 같은 LO rung(0.64–0.68): 요청 ITL p95≤50ms 통과율 fused_sym 0.99–1.00 = PD-mux 1.00, 요청 p95 중앙 fused_sym 16 ms < PD-mux 24–29 ms, A′ attainment fused_sym 0.83–0.94 vs PD-mux 0.16–0.50. fused p99 0.88 s(B LO) ≈ 8192토큰 prefill 시간 → 미조율 fused(chunk 8192, mixed-chunk off) 노브 효과(confound 2, §1-1 동류). X는 P9-a. ms·s 수치는 "M3 크기 인용" 금지 위반 |
| 2 | CM-2 무경합 prefill ≈ 9.0k tok/s, TTFT_floor 선형 | **CONFIRMED(범위 한정)** — 기울기 0.1108 ms/토큰, 절편 ≈ 9 ms, 16k floor 1.820–1.841 s(Job 4 × boot 8, ±0.6%) |
| 3 | CM-4 "클럭 가정이 B에서 반대" | **REFUTED(쓰인 그대로)** — 평균 클럭 비 B agnostic/fused_sym 0.998(반대 아님). 뒤집힌 것은 SW cap 노출률(B PD-mux 0.47–0.64 vs fused 0.77–0.93), achieved 부하·전력 매개, 시각 교락 식별 불가 |
| 4 | xcheck 2693 = 수치 비결정성, 규칙 (b) 설계 결함 | 원인 **PLAUSIBLE**(logprob 마진 없어 정확성 버그 가능성 잔존), 규칙 (b) 결함 **CONFIRMED** |
| 5 | throttle 공시 | **CONFIRMED(수치 정정)** — X는 HI 80–99%(LO 포함 46–99%), B HI 47–93%(LO 포함 41–93%), A 0–6%. 등록 throttle 공변량(max 0.0012)은 0x4를 못 잡음. "시간 교락 없음" → "반복 쌍 드리프트 미검출(N=1)" |

G0 설계 입력(arm별 prior·확장 발화 기본 시나리오)은 타당. 단 B·X Job 간 분산 미측정이라 prior 상수 고정 금지, 1번 때문에 G0에 튜닝된(mixed-chunk) fused 대조 arm 포함 검토.

## 정본 반영 허용 문안
1. (CM-6 대체) "J1(N=1)의 B shape에서, fused 계열은 prefill step 동안 decode가 멈추는 꼬리(ITL 상위 꼬리 ≈ 입력 하나의 prefill 시간)를 보이고 PD-mux arm은 보이지 않는다. 다만 achieved가 같은 rung에서 요청 ITL p95 SLO 통과율은 fused_sym과 PD-mux가 같으며, 이 꼬리는 미조율 fused(chunk 8192, mixed-chunk off)의 노브 효과일 수 있다. 처리율과 SLO의 역순은 미확립(G0 몫)이다."
2. (CM-2) "무경합 TTFT floor ≈ 9 ms + 0.111 ms/토큰 × L (L 1k–16k, 기울기 기준 ≈ 9.0k tok/s; Nano-9B-v2, 108 SM, chunked-prefill −1, VESSL A100, Job 4개·boot 8개에서 16k 산포 ±0.6%)."
3. (CM-4) "J1에서 SW power cap 노출률의 arm 간 순서는 shape에 따라 달랐고(B에서는 PD-mux가 낮고 X에서는 높음) 부하·전력이 매개한다. 'PD-mux가 클럭을 낮춘다'는 일반 가정으로 쓸 수 없다."
4. (xcheck) "doc 2693의 불일치는 기준 arm의 동시 패스에서만, 그리고 prefill 배치 구성이 다를 때만 나타났다(batch-invariance 부재로 추정, logprob 마진 미측정). 등록 규칙 (b)는 기준 잡음(D0)으로 보정하지 않는 설계 결함이 있으며, X 라벨은 rev13 전까지 인용 불가다."
5. (F-1) "J1 GPU0 SW power cap(0x4) 노출 60.6%, 셀별 노출은 A 0–6%·B 47–93%·X 80–99%(HI 셀), 400 W 한도. 등록 throttle 공변량은 이것을 잡지 못한다. B·X의 arm 비는 400 W 한도 아래 클럭 하락을 포함한 총효과로만 보고하며, 매개와 교란은 분리하지 않는다."

해소 실험: (1) G0 LO·중간 rung achieved 맞춤, arm {fused_sym, fused+mixed-chunk(2048/8192), agnostic}, Job ≥4 × seed 4, 요청 ITL p95 통과율·TTFT 짝 CI. (4) fused_sym·agnostic 각 {2693 단독, 2693+4079 강제 동반} ×5, index 14 top-2 logprob 마진.
