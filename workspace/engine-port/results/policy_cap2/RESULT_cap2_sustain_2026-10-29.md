# CAP2(W3 용량) + λ* 지속성 블록 결과 (2026-10-29) — 기술값, 미감사

Job `pdmux-policy-cap2-e90e233-20261029023336` (job-sg0dtebc3a0i), PREREG rev1.5 §20. stage1 CAP2 rc 0, stage2 ph_sustain rc 0. **아래 수치는 측정 기술값이며 성능 판정이 아니다.** CAP1-R 감사 caveat 1–5 승계.

## 1. 무결성
`RAW_SHA256SUMS` 158줄 전부 OK, `RAW_EXPORT_OK`. run 60개(5 D × 6 rate × 2 seed). trace 동일성((rate, seed)별 D 간 trace_hash) 위반 0. correctness 5 boot n_bad=0(n 99–120). `switch_count` 전부 0. 입력은 CAP2 run만(지속성 run은 `sustain/`에 분리, run id `SUSTAIN_…`; 섞이면 `ABORT_MIXED_CAP_JOBS`). raw는 `cap_char/policy_runs/*/raw/` gitignore.

## 2. W3 SLO floor 상수 (PREREG §13-1·20-3 규칙)
floor = 같은 Job 첫 boot(D24)의 단일 요청 8개 순차 TTFT 중앙값 = (893.50 + 983.06)/2 = **938.28 ms**(재계산 일치, `floor.json`), SLO TTFT = 3 × floor = **2814.84 ms**, ITL p95 ≤ 60 ms. 8개 중 최소 199 ms·최대 1504 ms로 산포가 크다(8점 중앙값; 이 상수가 이후 H-D의 SLO가 됨).

## 3. W3 용량 (attainment ≥ 0.90, 2 seed 합산, 96 req/rate)
| D | 마지막 통과 | 첫 실패 | 통과 rate attainment |
|---|---|---|---|
| 16 | 0.3375 | 0.50625 | 0.906 |
| 24 | 0.225 | 0.3375 | 0.969 |
| 34 | 0.225 | 0.3375 | 0.969 |
| 44 | 0.15 | 0.225 | 0.917(0.225는 0.896) |
| 64 | 없음 | 0.1 | 0.1에서도 0.802 |

- **`CAPACITY_BRACKETED`**, 비단조 없음, 재실행 불필요. argmax D = 16, 동률 없음(동률 규칙 발동 안 함). **λ\*_W3(best static) = 0.3375 req/s**.
- 주의: D16의 0.3375 통과는 0.906(96 req 중 87)으로 경계 직상(6요청 차)이다. W1과 반대로 W3는 작은 D가 큰 용량(D가 커질수록 감소 단조).
- 정상성(용량 셀 D16, 0.3375): seed 201 S1 1.17·S2 0.19, seed 202 S1 0.75·S2 −0.06 → 두 seed **STATIONARY**(CAP1-R과 달리). 전 60 run 중 UNSTABLE 18(대부분 첫 실패 rate 이상).
- CLIFF_ADJACENT: 30셀 중 0(최대 8.3%; 용량 셀 D16 4.2%).
- throttle(SW power cap 0x4, SM clk 중앙값 1410 MHz; boot 창은 clk 유휴 구간으로 복원, 마지막 boot 끝은 추정): D16 17.9%, D24 19.1%, D34 21.5%, D44 24.1%, D64 15.2% (`throttle_by_D.json`). CAP1-R(8.6–17.7%)보다 높고 D 간 차 최대 9 pp — 공변량 병기, 인과 서술 없음. boot 순서: D24, D16, D64, D44, D34.

## 4. λ* 지속성 블록 (기술값, 라벨·판정 없음)
W1, 7.29491 req/s 고정, 4,377 req(도착 약 600 s), 60 s 도착 창, 비sticky static D44·D64 × seed {201,202}.
| run | 첫 창 attainment | 이후 창 | 창별 TTFT p50 (첫→끝, s) | S1 / S2 | throttle(창, 범위) |
|---|---|---|---|---|---|
| D44 s201 | 0.33 | 모두 0.00 | 6.2 → 233.6 | 6.41 / 0.096 | 3–16% |
| D44 s202 | 0.30 | 모두 0.00 | 9.3 → 239.5 | 4.77 / 0.086 | 6–20% |
| D64 s201 | 0.36 | 모두 0.00 | 5.8 → 217.0 | 6.52 / 0.106 | 11–26% |
| D64 s202 | 0.28 | 모두 0.00 | 9.3 → 223.5 | 4.61 / 0.081 | 3–18% |
- 서술: 600 s 지속 부하에서 attainment는 첫 60 s 창 이후 0이고 TTFT는 창당 약 20–25 s씩 선형으로 증가해 큐가 해소되지 않는다(마지막 창 p50 217–240 s). 4개 run 모두 같은 양상(D·seed 무관). 단일 120-요청 셀(용량 셀)에서 0.97–0.98이던 같은 rate가 지속 구간에서는 유지되지 않았다 — 즉 CAP1-R의 λ*_W1은 caveat 1의 "버스트 용량"에 해당하는 값으로 기술된다. 창 TTFT p50 기울기 0.35–0.44 s/s에서 역산한 겉보기 처리율은 약 5.1–5.4 req/s(대략치, 선형 가정). throttle은 창별로 3–26%이고 D·시간과의 관계는 서술하지 않음.
- 함의(판정 아님): H-D의 q=1.0 W1 phase(120/600 s)에서 rate = 7.29491은 이 기판·이 시점 기준 지속 가능하다는 근거가 없으며, 오히려 반대 방향의 기술값이다. λ\*를 바꾸지 않는다는 등록(§20-4)에 따라 값은 그대로 두고 사용자 결정으로 넘긴다.

## 5. W1/W3 λ 실측 비, H-D W3 비중 (감사 caveat 6 갱신용)
λ\*_W1/λ\*_W3 = 7.29491/0.3375 = **21.6**. 요청률 가중 W3 비중 = λ3/(λ1+λ3) = **4.4%** (이전 가정 9%). 3% 판정 여유를 W3 phase만으로 만들려면 필요한 W3 상대 효과 ≈ 0.03/0.0442 = **약 68%** (이전 33%). 단 λ\*_W1은 §4의 지속성 기술값에 따르면 버스트 용량이고 λ\*_W3 통과는 경계 직상이다. 두 λ\*는 서로 다른 best D(44/64 vs 16)에서 나온 값.

## 6. addendum 초안 (미봉인·미제출) `addendum_cap2_result_DRAFT.json`
- HE0: LO = 2.7356, HI = 10.9424 req/s (h=1.5, 비 4, W1 λ\*).
- H-D(q=1.0): rate_W1 = 7.29491, rate_W3 = 0.3375, W3 SLO TTFT = 2814.84 ms(floor 상수).
- 위 caveat 3종(버스트 용량, 경계 통과, best D 불일치)을 파일에 병기.

## 7. 산출물
`cap2_analysis.json`, `throttle_by_D.json`, `cap2_capacity.png`(λ\*_D 막대+bracket, rate별 attainment), `cap2_sustain.png`(창별 attainment·TTFT p50/p95), `analyze_cap2.py`·`plot_cap2.py`·`throttle_by_D.py`.

## cost model 기여
W3(arXiv-Sum 16k, Nano 9B)의 static D별 용량은 W1과 반대로 D가 작을수록 크다(D16 0.3375 > D24·34 0.225 > D44 0.15 > D64 <0.1): prefill 비중이 큰 W3는 prefill SM 할당이 클수록 유리한 단조 곡선, W1은 decode 쪽 D가 클수록 유리 — 두 workload의 최적 D가 어긋난다는 기술값. 두 용량 비 21.6. 지속 부하(W1, 600 s)에서는 겉보기 처리율이 5.1–5.4 req/s 수준으로 버스트 용량 7.29와 차이가 있어, 비용 모델이 용량을 말할 때 셀 길이(버스트 vs 지속)를 구분해야 함을 시사.
