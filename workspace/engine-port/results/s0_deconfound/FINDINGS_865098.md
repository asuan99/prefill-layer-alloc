# De-confound run 865098 — 결과와 미해결 쟁점 (정본 아님)

2026-07-27. jobs `865098_{0,1,2}` (M/H/T), 전 cell PIN_CHECK **PASS**, errors 0.
**정본 수정 금지 상태** — DESIGN.md §5의 사전등록 규칙(H1이면 claims-auditor 선행)에
따라 여기까지만 기록한다.

## 1. 1차 결과 (raw, prefill 16 SM 고정)

client 측 steady-state ITL p50, n=4:

| arm | D16 | D24 | D44 | D92 | D16/D92 |
|---|---|---|---|---|---|
| M (음성대조) | 22.96 ±0.02 | 13.23 ±0.01 | 9.13 ±0.01 | 7.54 ±0.01 | **3.05×** |
| H (타깃) | 113.55 ±10.56 | 70.19 ±0.84 | 26.87 ±0.69 | 13.22 ±0.50 | **8.59×** |
| T (양성대조) | 17.71 ±0.01 | 12.92 ±0.00 | 9.62 ±0.00 | 8.55 ±0.00 | **2.07×** |

**사전등록 예측 H0(평탄)은 성립하지 않았다.** 단 아래 §2 때문에 H1 채택도 불가.

## 2. 남은 confound — decode batch가 D와 같이 움직였다

client의 closed-loop는 **in-flight 요청 수**를 16으로 고정했을 뿐 **decode batch**를
고정하지 못했다. prefill이 16 SM에 고정된 데다 keepalive 2개가 4096-token prefill을
계속 밀어넣어 prefill이 병목이 됐고, decode-SM이 커질수록 decode 구간이 빨리 비어
**동시 decode 수가 줄었다**. 엔진 telemetry가 독립 확인:

| arm | D16 | D24 | D44 | D92 |
|---|---|---|---|---|
| M | 5.45 | 3.54 | 3.21 | 3.18 |
| H | 8.44 | 8.53 | 5.44 | 3.28 |
| T | 8.34 | 6.42 | 4.78 | 3.82 |

(`decode_running_batch_size` mean.) ITL은 batch에 강하게 의존하므로 §1의 기울기는
decode-SM 효과와 batch 효과가 섞인 값이다. ★**계측해 둔 realized occupancy가 이걸
잡았다** — "고정했다"고 주장만 했으면 그대로 통과했을 confound다.

## 3. matched-batch 재분석 — 기울기는 살아남는다

telemetry의 `measured_itl_ewma_ms`를 `decode_running_batch_size`로 층화해 **같은 batch
끼리만** 비교(각 bin n≥30):

| arm | batch | D16 | D24 | D44 | D92 | D16/D92 |
|---|---|---|---|---|---|---|
| M | 1 | 15.4 | 11.6 | 7.5 | 6.4 | 2.39× |
| M | 2 | 15.8 | 11.7 | 8.1 | 6.8 | 2.32× |
| M | 4 | 18.0 | 13.2 | 9.2 | 7.6 | 2.37× |
| H | 1 | 25.7 | 19.4 | 12.2 | 9.6 | 2.67× |
| T | 5 | 16.5 | 12.8 | 9.6 | 8.7 | 1.89× |
| T | 7 | 17.9 | 12.9 | 9.7 | 8.8 | 2.04× |

⇒ **batch를 통제해도 decode-SM 기울기는 남는다(≈2–2.7×).** 단 일부 bin은 역전
(M batch=3 0.95×, T batch=4 0.87×) — EWMA는 창 평균이라 순간 batch와의 페어링이
근사이고, 저-D arm은 bin당 표본이 적다. **suggestive이지 확정 아님.**

## 4. 핵심 쟁점 — M(음성대조)이 왜 기울었나

DESIGN.md §3은 "M에 기울기가 나오면 잔여 confound의 지표"라고 사전등록했다. 그러나
matched-batch에서도 M이 2.3–2.4×로 기울었고, 여기엔 **서로 배타적인 두 해석**이 있다.

- **(a) 잔여 confound**: 아직 통제 못 한 변수가 있다(prefill 대기, KV/스케줄 상호작용,
  §5의 clock). 이 경우 §1·§3 전부 무효.
- **(b) 음성대조 전제 자체가 틀렸다**: "pure Mamba2 decode는 O(1)이라 **SM-bound 불가**"는
  Stage 0가 세운 전제인데, **O(1)은 context-length에 대한 것이지 SM 수에 대한 것이 아니다.**
  Mamba2 decode step도 d_model×d_state matmul을 하므로 16 SM과 92 SM에서 다를 수 있다.
  (b)가 맞다면 M은 애초에 이 축의 음성대조가 아니며, 기울기는 **실재**한다.

**(b)가 맞다면 Stage 0의 헤드라인(D16≡D108=1.00±0.01, decode SM-무감각)이 아티팩트**가
된다. 가능한 기전: Stage 0의 sub-108 decode 파티션은 **prefill이 in-flight일 때만
활성**인데, PIN_CHECK는 policy의 *target*만 봤을 뿐 **파티션이 실제로 활성이던 시간
비율은 보지 않았다.** Stage 0의 D16이 상당 시간 full-108 스트림에서 돌았다면 D108과
같아 보이는 것이 당연하다. 본 실험은 prefill을 16 SM에 고정해 prefill을 상시 in-flight로
만들었으므로 파티션 활성 시간이 훨씬 길다 — 두 결과의 차이와 정합적인 설명이다.

★ 단, 이는 **가설**이다. 확인하려면 Stage 0 telemetry에서 "decode step 중 sub-108
파티션이 실제 활성이던 비율"을 재집계해야 한다(미실행).

## 5. 이 실행의 알려진 한계

- decode batch 미통제(§2). matched-batch는 **사후 층화**이지 설계상 통제가 아니다.
- `decode_last_tpot_ms`는 telemetry에 항상 0 — 대신 EWMA를 썼다(창 평균, 근사).
- clock: arm마다 활성 SM 총량이 다르다(32 vs 108). 저-D arm이 boost를 더 받으면
  기울기를 **과소평가**하는 방향이라 관측된 기울기의 하한 성격은 유지되나, 정량은 미보정.
- H의 D16 SD가 10.56ms로 유독 크다(다른 셀은 ≤1). 포화 근처일 가능성 — H D16만
  regime이 다를 수 있다.

## 6. 다음 수순 (제안, 미실행)

1. **claims-auditor 적대적 반증** — §4의 (a) vs (b)를 가르는 것이 전부다. 사전등록대로
   정본 수정 전 필수.
2. **Stage 0 파티션 활성시간 재집계** — 기존 `stage0_xctrl/*_telemetry.jsonl`만으로
   가능(신규 GPU 불요). §4 가설의 직접 검증.
3. **batch 설계상 통제 재측정** — keepalive 프롬프트를 짧게(파티션만 활성화, prefill
   병목 제거) + out_tokens 확대 → occupancy가 arm 간 일치하는지 realized occupancy로
   게이트. 클라이언트는 이미 `--keepalive-prompt-file`을 지원한다.
