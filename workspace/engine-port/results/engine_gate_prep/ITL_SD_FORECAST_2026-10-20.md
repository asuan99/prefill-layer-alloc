# G-γ 관측자 TOST 짝 수 예보: ITL p95·λ SD 재집계 (2026-10-20, result-analyst)

> 지위: **예보용 입력**이다. 판정에 쓰지 않는다. gate 감사 r1 §3 수리 ②(engsync `results/engine_gate/AUDIT_gate_split_design_r1_2026-10-20.md:16-19`)에 대응한다.
> 스크립트는 `itl_sd_forecast.py`(같은 폴더), 파생 데이터는 `itl_sd_forecast.json`, 콘솔 출력은 `run.log`이다. CPU만 썼고 MC seed는 1이다.
> 원자료는 main `results/multigpu/manual_retrieval/`의 J1(2106f42), J2(e559c64), J4(213df1b), J8(3167f74)다. 원시 `requests.jsonl`에서 다시 계산했고, achieved는 정본 `mg_label.load_cell`을 썼다.

## 0. 기판 주의 (먼저 읽을 것)
- J1–J8은 mem 0.82와 구 엔진 digest로 돌았고 J1에는 power cap 노출이 있다. gate는 mem 0.6, 새 digest(I-4/5/6 계측 포함), 단일 GPU Job이다. **그러니 이 숫자는 짝 수를 정하는 사전 입력일 뿐 판정 근거가 아니다.**
- 네 Job은 이미지 digest(`1e0d335b…`), 드라이버 580.105.08, A100-SXM4-80GB, 400 W가 같고 **호스트는 4개 모두 다르다**. trace sha는 seed마다 Job 간에 같다(`675b5ecb…` 등).
- J1의 LO offered는 1.958이다. gate LO는 1.794로 0.916배다. HI는 둘 다 4.784로 같다.

## 1. 가용 데이터와 빈칸
| 성분 | 구할 수 있는 곳 | gate 짝과의 관계 |
|---|---|---|
| 같은 GPU에서 boot만 바뀐 경우(gate 짝이 겪는 분산) | **없음** | — |
| GPU+boot 간(같은 Job·호스트) | J8 `gpu_dev`: **arm fused**(fused_sym 아님), HI, s4386, GPU 8장 | 상한 쪽. GPU 고정효과까지 들어 있다 |
| Job(호스트+boot) 간 | J1/J2/J4/J8 N1 HI, seed 4개 × Job 4개 (arm마다) | 상한. 호스트 효과는 단일 Job 짝에서 상쇄된다 |
| 2원 잔차(seed+Job 오프셋 제거) | 위 4×4 행렬, df 9 | 셀 수준 잡음. 하한 쪽 |
| seed 간(같은 boot) | HI 4개, LO 2개 | 짝 안에서 seed를 공유하므로 상쇄된다. 참고용 |
| **LO boot 간** | **없음.** N1 LO는 J1 boot2 하나뿐이다(seed 2개) | gate ITL 지표가 LO이므로 **핵심 빈칸이다** |
| 요청 bootstrap(셀 안) | 전 셀 | **쓸 수 없는 proxy다.** HI에서 0.03–0.2%인데 실제 Job·GPU 간은 1.2–4.3%로 10–80배 과소추정한다 |

## 2. log 값 SD (rep = 한 instance의 SD, 짝 차이 SD ≈ √2·rep로 가정)
가정: ON과 OFF는 분산이 같은 독립 반복이고 계측이 분산을 바꾸지 않으며 seed 효과는 짝 안에서 완전히 상쇄된다. ABBA가 선형 drift를 지운다고 본다.

| 지표 | arm | Job 간 (df 12) | 2원 잔차 (df 9) | GPU+boot (J8, fused, n=8) | seed 간 (같은 boot) |
|---|---|---|---|---|---|
| λ = HI achieved | agnostic | 0.85% | 0.61% | — | 0.83% |
| λ = HI achieved | fused_sym | 1.17% | 0.72% | 0.15% (fused) | 0.63% |
| ITL 풀 p95, **HI** | agnostic | 4.28% | 3.53% | — | 3.56% |
| ITL 풀 p95, **HI** | fused_sym | 2.26% | 0.80% | 1.21% (fused) | 0.84% |
| 요청 내부 p95 중앙값, HI | agnostic | 4.31% | 3.50% | — | 3.51% |
| 요청 내부 p95 중앙값, HI | fused_sym | 3.06% | 1.64% | 1.23% (fused) | 1.69% |
| ITL 풀 p95, **LO**(gate 지표) | agnostic / fused_sym | **미측정** | 미측정 | 미측정 | 14.2% / 7.2% (seed 2개, 실현 부하 1.69 대 2.09로 교락) |
| 요청 내부 p95 중앙값, LO | agnostic / fused_sym | 미측정 | 미측정 | 미측정 | 11.0% / 6.1% |

- Job 오프셋: HI ITL p95는 J1·J2가 +2%대, J4·J8이 −1~−4%로 **두 arm 모두 같은 방향**이다. 호스트 수준 효과이며 단일 Job 짝에서는 상쇄된다.
- agnostic HI ITL p95는 셀마다 **이산 수준**(약 41.4 / 44.5 / 47.6 ms)을 오가고 같은 boot 안에서도 바뀐다(J4 s251 47.6 대 나머지 41.4). 가우스 잡음이 아니라 상태 전환형이다.
- J2 fused_sym은 이상치를 낸다(s4162 >30 ms 비율 13.4%, 다른 Job은 6.2%. s4386 요청 p95 중앙값 74.6, 다른 Job은 67–71). fused_sym의 Job 간 SD 상당 부분이 여기서 나온다.

## 3. 반폭 예보와 효과 0일 때 통과 확률
95% 양측 t-CI(α 분할 look)를 쓰고 n은 짝 수, δ = ln 1.015 = 1.49%다. 통과 확률은 MC 40,000회로 구했다. 2단계 = n=10 look에서 PASS면 종료, CI가 0을 배제하면 OBSERVER_EFFECT로 종료, 그 밖이면 n=16으로 확장한다.

| proxy (rep SD) | 반폭 n=8 / 10 / 16 | P(PASS) n=8 / 10 / 16 | 2단계 P(PASS) | 2단계 n=10에서 가짜 OBSERVER_EFFECT |
|---|---|---|---|---|
| 0.61% (λ agnostic 잔차) | 0.72 / 0.61 / 0.46% | 0.97 / 1.00 / 1.00 | 0.998 | 0.2% |
| 0.72% (λ fused_sym 잔차) | 0.85 / 0.73 / 0.54% | 0.89 / 0.97 / 1.00 | 0.99 | 0.8% |
| 0.80% (ITL HI fused_sym 잔차) | 0.95 / 0.81 / 0.60% | 0.78 / 0.91 / 1.00 | 0.98 | 1.6% |
| 0.85% (λ agnostic Job 간) | 1.00 / 0.86 / 0.64% | 0.71 / 0.87 / 0.99 | 0.975 | 2.1% |
| 1.17% (λ fused_sym Job 간) | 1.38 / 1.18 / 0.88% | 0.29 / 0.46 / 0.84 | 0.83 | 4.3% |
| 1.21% (ITL GPU+boot, fused) | 1.43 / 1.23 / 0.91% | 0.25 / 0.41 / 0.80 | 0.80 | 4.4% |
| 1.64% (요청 p95 중앙값 fused_sym 잔차) | 1.94 / 1.66 / 1.24% | 0.06 / 0.10 / 0.36 | 0.38 | 5.1% |
| 2.26% (ITL fused_sym Job 간) | 2.67 / 2.28 / 1.70% | 0.01 / 0.01 / 0.05 | 0.06 | 5% |
| ≥3.5% (ITL agnostic HI, 모든 proxy) | ≥4.1 / ≥3.5 / ≥2.6% | 0 / 0 / 0 | 0 | 5% |

효과 0에서 P(PASS) ≥ 0.80이 되는 rep SD의 상한은 95% CI 기준 n=8에서 0.79%, n=10에서 0.91%, n=16에서 1.21%다(90% CI 기준이면 0.91, 1.05, 1.38%).

## 4. LO ITL p95의 cliff (gate 지표의 구조적 위험)
shape A LO의 token ITL 분포는 **두 봉우리**다. 하나는 decode step 봉우리(약 20 ms), 다른 하나는 prefill이 decode를 막는 stall 봉우리다(fused는 약 60–65 ms, agnostic은 약 30–45 ms). p95가 어느 봉우리에 놓이는지는 stall 비율(>30 ms)이 5%를 넘는지로 정해진다.

| 셀 (J1, offered 1.958) | 실현 λ | stall 비율 | 5%까지 여유 | q95 위치 | q95 근처 기울기 |
|---|---|---|---|---|---|
| fused_sym s4162 | 1.70 | 2.99% | −2.0 pt | 아래 봉우리 20.8 ms (gap 위 63 ms) | 0.7%/pt |
| fused_sym s4386 | 2.09 | 4.01% | **−1.0 pt** | 아래 봉우리 23.0 ms (q96 = 59.7 ms) | 1.9%/pt |
| agnostic s4162 | 1.69 | 6.49% | **+1.5 pt** | 위 봉우리 34.5 ms (gap 아래 22 ms) | 1.9%/pt |
| agnostic s4386 | 2.08 | 9.15% | +4.2 pt | 위 봉우리 42.1 ms | 8.9%/pt |

- stall 비율이 λ에 비례한다고 놓고 [유도] gate LO 1.794(0.916배)로 옮기면 fused_sym은 2.7–3.7%(여유 1.3–2.3 pt), agnostic은 5.9–8.4%(여유 0.9–3.4 pt)다. seed가 같아도 실현 λ는 offered의 0.87–1.07배로 흩어진다. 그래서 gate seed 10개 가운데 **어떤 seed는 여유가 1 pt 미만일 수 있다.**
- 그런 seed에서 ON/OFF 섭동이 stall 비율을 여유만큼 옮기면 그 짝의 log 비는 ±0.4(agnostic 34→22 ms)에서 ±0.95(fused 23→60 ms) 사이로 뛴다. 짝 1개로 TOST가 끝난다. 이것은 분산이 아니라 **지표의 불연속**이다(게이트 4 metric cliff의 percentile판).
- 반대로 cliff에서 멀면 보통의 stall 비율 재현성(HI에서 같은 seed의 Job 간 ±0.05 pt, J2 같은 이상치 제외)과 위 기울기로 계산한 p95 변동은 0.05–0.45%로 작다. 이 경우 지배항은 봉우리 자체의 시간 수준(decode step 시간)이고, 그 proxy는 0.8–1.2%(HI 잔차, GPU+boot)다.

## 5. 판단 입력
1. **λ(HI achieved)**: rep SD 0.6–1.2%다. n=10 + 등록된 확장 16이면 2단계 통과 확률이 0.83–0.998이다. 등록된 설계로 충분하며, n=8 단일 look은 fused_sym에 부족하다(0.29–0.89).
2. **ITL p95(LO)**: SD가 **미측정**이라 예보 구간이 넓다. 같은 GPU 짝의 rep SD가 0.8%면 n=10에서 0.91, 1.2%면 n=16에서야 0.80이고, Job 간 수준(≥2.3%)이면 n=16으로도 0.05 이하다. **권고: n=10 + 확장 16을 유지한다(2단계에서 rep SD 1.2%까지 P≥0.80).** n=8 단일 look은 rep SD 0.79%를 넘으면 0.8 아래로 떨어지므로 권하지 않는다.
3. **원리적으로 통과 불가한 지표**: HI ITL p95(agnostic)는 rep SD가 3.5% 이상이고 이산 수준을 오가므로 n≤16에서 통과 확률이 0이다. gate는 이 지표를 쓰지 않지만 HI 셀 ITL로 대체하면 안 된다. LO ITL p95는 SD 때문이 아니라 §4의 cliff 때문에 seed에 따라 통과 불가가 될 수 있다.
4. **대안 지표**: 요청 내부 p95 중앙값은 모든 proxy에서 풀 p95와 같거나 더 크다(HI fused_sym 잔차 1.64% 대 0.80%, LO 요청 bootstrap 0.33–0.61% 대 0.15–0.22%). 요청 하나가 511개 간격에 stall 약 18개를 가지므로 요청 단위 p95도 같은 cliff 위에 놓인다. **대안으로서 이득이 없다.** cliff를 피하려면 지표를 [decode 봉우리 수준(예: p50 또는 하위 봉우리 중앙값) + stall 비율] 두 개로 나누는 방법이 있으나, 이것은 지표 변경이므로 사전 등록 개정이 필요하다(사용자 결정이며 이 문서는 권고만 한다).
5. **사전 조치(제출 전, 권장)**: gate LO 1.794에서 seed 10개(+확장 6개)의 실현 λ를 trace로 CPU 계산하고, stall 비율 여유가 1 pt 미만으로 예상되는 seed를 교체하거나 공시한다. 또는 3분기 판정의 `UNRESOLVED_POWER` 외에 "짝 log 비 |x|>0.3 = cliff 이탈" 진단 라벨을 등록한다.
6. **A/A 짝(선택)**: 같은 GPU에서 OFF/OFF 짝 2–3개를 두면 이 문서의 빈칸(같은 GPU boot 간 LO SD)을 gate 안에서 직접 잴 수 있다. 비용은 instance 4–6개, 약 0.6–0.9 GPU-h다 [유도].

## 6. cost model 기여
- CM-4(동거 간섭 δ(p,d)): 함수 형태. shape A LO에서 token ITL은 [decode step 봉우리] + [prefill stall 봉우리]의 혼합이고, stall 질량 ≈ λ × (동시 decode bs) / 출력 길이다(J1에서 fused_sym 3.0–4.0%, agnostic 6.5–9.2%). 꼬리 백분위 지표는 이 질량이 1−q를 지날 때 불연속이 된다. 기술 사실이며 계수는 미추정이다.
- 측정 축: ITL p95(HI)에 호스트 효과가 약 ±2%(J1·J2 대 J4·J8, arm 공통) 있고, λ의 호스트 효과는 ±1% 이내다. Job 간 ITL 비교는 이 크기 아래를 가를 수 없다.
