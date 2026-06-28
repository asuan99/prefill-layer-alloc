# 글 흐름 × 그림 배치 × 형태 적합성 (논문용)

목적: (1) Results의 **내용 흐름(storyline)** 정리, (2) 각 plot이 그 흐름의 *어디·왜*에 쓰이는지, (3) **그래프 형태가 해당 지표를 설명하기에 적합한지** 비판적 점검.

---

## 1. Results 흐름 (storyline)

> 논지: *"하이브리드의 attn/ssm 비대칭을 자원배분으로 활용 가능한가?"* → 순서대로 답이 쌓인다.

| § | 내용 단계 | 핵심 메시지 | 주 그림 |
|---|---|---|---|
| **S1** | 비대칭의 *측정* | attn-decode=메모리바운드·O(L)·비쌈 vs ssm-decode=O(1)·BW~0%·쌈 | (E2/E3/E5 특성화) |
| **S2** | 순진한 활용은 *실패* | 정적 layer-type 공간 분할은 co-schedule 못 이김 | **(공백 — §4 참조)** |
| **S3** | *재정식화*: layer-type-aware decode **예약** | 예약을 비싼 attn 레이어에만 → goodput↑ + 메커니즘 | `layer_aware_result` |
| **S4** | *일반성*: 크기 1.2B–7B | 전 크기 이득(1.37/2.02/1.82×) | `layer_aware_size_trend` |
| **S5** | *적용 범위*: temporal-only | 두 전제(분리+비싼 no-GQA attn) | `temporal_vs_spatial` (+`prefill_sm_sensitivity`) |
| **S6** | *프로덕션 대비*: vs fused(vLLM) | fused는 decode 결합→ITL 높음; 예약이 bound | `framework_comparison` |
| **S7** | *검증·한계* | vLLM이 decode 모델 검증(상대), 절대는 directional | `vllm_calibration` |
| **S0** | (선택) 개요 teaser | 1장 요약 | `summary_dashboard` |

**흐름 한 줄:** 비대칭 측정(S1) → 순진한 분할 실패(S2) → 예약으로 재정식화·작동(S3) → 크기 일반화(S4) → 적용범위 한정(S5) → 프로덕션 대비(S6) → 검증·한계(S7).

---

## 2. 그림별 — 내러티브 역할 × 형태 적합성

| 그림 | 흐름 위치 | 보여주는 지표 | 차트 형태 | **형태 적합성 (비판)** |
|---|---|---|---|---|
| `layer_aware_result` (A) goodput-vs-SLO | S3 주결과 | 연속 SLO 스윕의 goodput | **선그래프 + 우위구간 음영** | ✅ **최적.** 연속 독립변수(SLO)→선이 맞고, 음영으로 우위 영역 명시. 논문 main figure감. |
| `layer_aware_result` (B) tradeoff | S3 보조 | throughput·TTFT·ITL | 막대+쌍축 2선 | ⚠ **과적재.** 한 패널에 3지표(단위 다름: tok/s·s·ms)를 막대+쌍축으로 → 가독성 낮음. **분리하거나 정규화 권고.** |
| `layer_aware_result` (C) 메커니즘 | S3 원리 | 레이어별 SM 배분 | 도식(레이어 막대) | ✅ 메커니즘 설명엔 도식이 적합. |
| `layer_aware_size_trend` | S4 | la/agnostic 비(이산 3모델)·정책별 goodput | 막대 / 그룹막대 | ✅ 이산 모델 비교엔 막대 적합. △ 3점으로 "peak" 주장은 약함 → 캡션에 *추세 아닌 3점*임을 명시. |
| `temporal_vs_spatial` | S5 핵심 | per-layer decode 구조(분리성) | **레이어별 막대** | ✅ **우수.** "비싼 attn이 *어디* 있나(분리 가능?)"를 레이어축 막대로 직접 시각화 → 적용범위 논거에 정확. |
| `prefill_sm_sensitivity` | S5 보조(전제②) | SM별 prefill 지연비(연속) | **선그래프(3모델 중첩)** | ✅ **최적.** 연속 SM 스윕→선, 3모델 중첩으로 "겹침(크기-불변)"을 직접 보임. (역방향 x축은 사소한 가독성 흠.) |
| `framework_comparison` | S6 | 정책별 decode ITL(이산) | 그룹막대+vLLM 앵커선 | ✅ 이산 정책 비교엔 막대 적합·앵커 좋음. ⚠ **fused가 y축 지배(0–400)** → 예약 정책 간 차(26 vs 30)가 안 보임. **log축 또는 broken-axis 권고.** |
| `vllm_calibration` (A) | S7 | sim vs 실측 ITL(이산) | 그룹막대+과대비 | ✅ 적합. |
| `vllm_calibration` (B) | S7 핵심 | 크기-스케일링(정규화) | **선그래프(2선 중첩)** | ✅ **우수.** 두 추세(sim·실측)가 겹침을 선중첩으로 보임 = "상대 추세 일치"에 정확. |
| `summary_dashboard` | S0 | 3개 독립 메시지 | 3-panel 혼합 | △ **슬라이드형.** 개요엔 좋으나 논문 본문 figure로는 서로 다른 3주제를 한 그림에 → 분리 추천(또는 발표/teaser 한정). |
| `layer_aware_applicability` | S5(중복) | 레이어 배치 도식 | 만화형 블록 | ⚠ **`temporal_vs_spatial`과 중복**(같은 메시지를 *정성* 만화로). 정량인 `temporal_vs_spatial`이 상위호환 → **본문에선 1개만**(applicability는 부록/발표용). |

### 형태 적합성 원칙 (점검 기준)
- **연속 독립변수**(SLO·SM수·크기-스케일) → **선그래프** ⇒ goodput-vs-SLO·prefill-sensitivity·calibration(B) 모두 정확.
- **이산 범주 비교**(정책·모델) → **막대/그룹막대** ⇒ framework·size_trend·calibration(A) 적합.
- **구조·메커니즘** → **도식** ⇒ mechanism·temporal_vs_spatial 적합.

---

## 3. 권고 (논문 figure 셋)

**본문 main (4–5장):**
1. `layer_aware_result(A)` — 주결과(goodput vs SLO) ★
2. `layer_aware_result(C)` 또는 별도 메커니즘 도식 — 작동 원리
3. `layer_aware_size_trend` — 크기 일반성
4. `temporal_vs_spatial` — 적용범위(구조 근거) ★
5. `framework_comparison`(축 개선) 또는 `vllm_calibration` — 프로덕션 대비/검증

**개선 작업:**
- (B) tradeoff 패널 → **분리/정규화** (또는 throughput만 막대 + 별도 ITL-vs-SLO).
- `framework_comparison` → **log y축**(fused 지배로 가려진 예약-정책 차 노출).
- `summary_dashboard`·`layer_aware_applicability` → **본문 제외**(teaser/부록).

**채워야 할 공백:**
- **S2(정적 분할 실패) 헤드라인 그림 부재.** 현재 특성화(`e5_spatial_specific_gain` 등)에 흩어져 있음 → *green_ctx가 two_stream을 전 sweep서 못 이김*을 한 장으로(예: green/two_stream goodput 산점 또는 sweep 히트맵) 만들면 "순진한 분할=실패→예약으로 재정식화" 흐름이 강해진다.
- 주결과 goodput-vs-SLO가 **2.7B 단일** → 다모델(1.2/2.7/7B) SLO 스윕 중첩 선그래프면 S3+S4를 한 장에 통합 가능.
