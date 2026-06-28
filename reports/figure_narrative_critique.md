# 글 흐름 × 그림 배치 × 형태 적합성 (논문용)

목적: (1) Results의 **내용 흐름(storyline)** 정리, (2) 각 plot이 그 흐름의 *어디·왜*에 쓰이는지, (3) **그래프 형태가 해당 지표를 설명하기에 적합한지** 비판적 점검.

---

## 1. Results 흐름 (storyline) — *재정식화된 시스템 중심*

> 논문의 기여 = **layer-type-aware decode reservation 시스템**. "정적 분할 실패→재정식화"는 *서사가 아니다*. 흐름은 *제안 시스템을 제시·평가*하는 순서로 간다. (정적 분할이 아닌 *예약*을 택한 이유는 설계 근거 한 줄로 충분 — §아래 메모.)

| § | 내용 단계 | 핵심 메시지 | 주 그림 |
|---|---|---|---|
| **S1** | *관찰/동기*: per-layer decode 비대칭 | attn-decode=비쌈(메모리바운드·O(L)) vs ssm-decode=쌈(O(1)·BW~0%) → 레이어별로 *보호 가치가 다르다* | `temporal_vs_spatial`(per-layer 구조) |
| **S2** | *시스템*: layer-type-aware **예약** | 비싼 attn 레이어에만 decode SM 예약, 싼 ssm은 prefill 환원 (메커니즘) | `layer_aware_result(C)` 메커니즘 |
| **S3** | *주결과*: goodput@SLO | baseline 대비 우위(연속 SLO 스윕) | `layer_aware_result(A)` |
| **S4** | *일반성*: 크기 1.2B–7B | 전 크기 이득(1.37/2.02/1.82×) | `layer_aware_size_trend` |
| **S5** | *적용 범위*: temporal-only | 두 전제(분리+비싼 no-GQA attn) | `temporal_vs_spatial` (+`prefill_sm_sensitivity`) |
| **S6** | *프로덕션 대비*: vs fused(vLLM) | fused는 decode 결합→ITL 높음; 예약이 bound | `framework_comparison` |
| **S7** | *검증·한계* | vLLM이 decode 모델 검증(상대), 절대는 directional | `vllm_calibration` |
| **S0** | (선택) 개요 teaser | 1장 요약 | `summary_dashboard` |

**흐름 한 줄:** 비대칭 관찰(S1) → 예약 시스템 제시·작동(S2–S3) → 크기 일반화(S4) → 적용범위 한정(S5) → 프로덕션 대비(S6) → 검증·한계(S7).

> **설계 근거 메모(별도 그림 불요):** *공간 분할*이 아니라 *시간적 예약*을 택한 이유 = 하이브리드에서 attn/ssm 자원 비대칭은 *한 forward 내 SM 분할의 lever가 아니며*(compute⊥memory), 레이어가 시간축에서 타입별로 다르게 나타나는 점을 이용해 *예약*하는 것이 자연스럽다. → Design 절의 1–2문장(필요시 특성화 인용)으로 처리, results 비트·헤드라인 그림 아님.

---

## 2. 그림별 — 내러티브 역할 × 형태 적합성

| 그림 | 흐름 위치 | 보여주는 지표 | 차트 형태 | **형태 적합성 (비판)** |
|---|---|---|---|---|
| `layer_aware_result` (A) goodput-vs-SLO | S3 주결과 | 연속 SLO 스윕의 goodput | **선그래프 + 우위구간 음영** | ✅ **최적.** 연속 독립변수(SLO)→선이 맞고, 음영으로 우위 영역 명시. 논문 main figure감. |
| `layer_aware_result` (B) tradeoff | S3 보조 | throughput·TTFT·ITL | 막대+쌍축 2선 | ⚠ **과적재.** 한 패널에 3지표(단위 다름: tok/s·s·ms)를 막대+쌍축으로 → 가독성 낮음. **분리하거나 정규화 권고.** |
| `layer_aware_result` (C) 메커니즘 | **S2 시스템** | 레이어별 SM 배분 | 도식(레이어 막대) | ✅ 메커니즘 설명엔 도식이 적합. 시스템 절의 핵심 도식. |
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
- 주결과 goodput-vs-SLO가 **2.7B 단일** → 다모델(1.2/2.7/7B) SLO 스윕 중첩 선그래프면 **S3(주결과)+S4(크기 일반화)를 한 장에 통합** 가능. 시스템 평가의 핵심 그림으로 권장.
- (정적 분할 실패 그림은 *불요* — 본 논문은 재정식화된 *예약 시스템*을 제시하는 것이지 실패→재정식화 서사가 아님. 분할 대신 예약을 택한 근거는 Design 절 1–2문장으로 충분, 그림 없음.)
