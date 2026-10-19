# DESIGN v0 — cap 특성화 Job (R16-3 · CM-5 검증 · Claim E′) — 초안, PREREG 아님 (2026-10-19)

> ★**대체됨(2026-10-19)**: `DESIGN_cap_longctx_v1_2026-10-19.md`가 이 초안을 대체한다(결정 8′). 이력용.

> 지위: main 트랙 설계 초안. 측정 0, 감사 0. 이 문서의 수치는 [유도]·[제안]이며 판정 규칙은 PREREG에서 확정한다.
> 근거: 사용자 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` §1–9 · 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`
> §9(I-5·I-6)·§12(R16-1–R16-3) · 비용 메모 `../handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md` · 감사 `../anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`
> · J1 분석 `../multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`.

## 0. 이 Job이 답하는 질문 세 가지

| ID | 질문 | 문장 종류 | 연결 |
|---|---|---|---|
| Q-A | 어느 자원이 실제로 admission·decode bs를 묶는가(KV / cap=슬롯)? 그 경계 ctx는 어디인가? | 특성화(기술값 + 경계 구간) | Q19, X-6, E′ anchor(후보1)의 "[실측 전환점]" |
| Q-B | cap을 바꾸면 SLO goodput(λ\*\_SLO)이 어떻게 변하는가, 그리고 그 변화가 hybrid와 attention 대조 모델에서 다른가? | 특성화 + 대조 비교 | E′ 본문(후보2), "왜 hybrid인가"(결정 4) |
| Q-C | G0에 쓸 cap 값 | 사전등록 결정 규칙의 출력 | R16-1·D-17 |

Claim E 원안(decode floor 예측)은 이 Job에서 시험하지 않는다(결정 7: 열린 질문).

## 1. 왜 hybrid에서 cap이 다른가 [코드 사실 + 유도]

- radix OFF에서 `max_mamba_cache_size = cap`(§1-23)이다. 그래서 cap 하나가 **admission 상한 + decode bs 상한 + state pool 예약(cap × 0.1435 GB, 사용 여부 무관)**을 동시에 정한다. 순수 attention 모델에서는 cap이 admission만 정하고 메모리 비용이 0이다.
- ⇒ hybrid에서 cap을 올리면 KV 풀이 줄어든다(48 → 6.9 GB 예약 [유도]). attention 대조 모델에서는 cap을 올려도 KV 풀이 그대로다. **이 비대칭이 Q-B 대조의 처치 대비다.**
- 엔진은 state를 예약하고 보고한다(SGLang·vLLM). 따라서 "엔진이 state를 무시한다"는 허수아비이고, 이 Job은 **결합의 비용**을 잰다(결정 8 금지 문안).

## 2. 요인 [제안]

| 요인 | 수준 | 메모 |
|---|---|---|
| 모델 | Nemotron-Nano-9B-v2(주) · 순수 attention 대조(예: Qwen2.5-7B, 같은 백엔드) | 결정 4. 대조 모델의 확정은 G0 PREREG와 같이 한다(한 번만 고른다). |
| cap | {48, C\_mem/2, C\_mem} | C\_mem은 **부팅 로그 실측**(R16-3)이다. (모델, ctx 수준)마다 달라지므로 부팅 프로브 단계를 둔다. |
| ctx 수준 | (i) arXiv-Sum 16k 표준 셀 (ii) shape C 표준 셀(데이터셋 미정, 결정 5) (iii) 경계 횡단 고정 길이 {32k, 64k, 96k} [제안] | (iii)은 Q-A 경계용이다. ~52k는 유도값이라 경계를 **사이에 두는** 격자로 잡는다. |
| arm | fused\_sym · agnostic (G0 arm과 같은 노브) | Q-C 규칙이 한쪽 arm만 보면 arm 비대칭 선택 경로가 생긴다(SPEC :242). 그래서 두 arm을 모두 둔다. (iii)은 fused\_sym만 둘지 결정한다(§6 열린 질문 3). |
| seed | 4 | n≥4 게이트. |

고정: radix OFF(F8), cudagraph ON, mem-fraction 0.82, overlap 설정은 G0 arm 정의를 따른다, `--max-mamba-cache-size = cap` 명시. 한 비교의 arm은 한 Job 안에서 돈다.

## 3. 계측 (전부 G0와 공용)

- **I-5 거절 사유**(필수, 엔진 패치 + correctness gate 선행): 스케줄 결정마다 사유 ∈ {KV, cap, mamba 슬롯, prefill 자리, `max_prefill_tokens`, R2 latch}와 retract를 기록한다. 현재 engine-porter가 조사·구현 중이다.
  - ★ 확인할 점: radix OFF에서 슬롯 수 = cap이면 "cap" 사유와 "슬롯" 사유가 **항등적으로 같을 수 있다**. 그러면 Q-A의 이분은 "KV 대 cap(=슬롯)"이 된다. engine-porter 판정을 받아 이 문서를 고친다.
- **I-6 cudagraph 패딩**(cap을 바꾸면 캡처 bs 목록이 바뀐다, R16-3): 필수.
- **I-1 클럭·전력**: 필수. J1에서 B·X shape의 throttle 노출률이 47–98%였다. 그래서 cap 간 비는 "400 W 한도의 총효과"로만 보고한다(F-1 문안).
- **부팅 로그**: `max_total_num_tokens`, mamba 슬롯 수, 실제 KV 풀 바이트. 셀마다 기록한다.
- 셀 단위 실점유: KV 사용 토큰과 state 슬롯 사용 수의 시계열(I-5 필드와 같이 낸다).

## 4. estimand와 판정 규칙 초안 [제안 — PREREG에서 확정]

- **Q-A 경계**: (모델, cap, ctx)마다 거절 사유 비율을 구한다. 대상 rung은 λ\*\_SLO 교차 rung과 HI다. HI는 포화라 항등식에 가깝기 때문에 따로 표시한다. "구속 자원 = 비율 ≥ 50%인 사유"로 정하고, 경계는 그 라벨이 바뀌는 인접 ctx 두 점 **사이의 구간**으로 보고한다. 점추정은 하지 않는다.
- **Q-B 결합 비용**: hybrid와 대조 모델 각각에서 λ\*\_SLO(cap)를 구한다. 1차 대비는 이중차분 Δ = [log λ\*(C\_mem) − log λ\*(48)]\_hybrid − [같은 값]\_attn이다. Job 안 짝 bootstrap CI를 쓰고, |Δ| < 3%이면 방향을 인용하지 않는다(게이트 3).
  - 주의: 두 모델의 C\_mem이 다르다. 비교 축은 cap의 절대값이 아니라 "48 대비 C\_mem"이라는 상대 수준이다. 이 선택은 사전에 고정한다.
- **Q-C G0 cap 규칙**: "두 표준 셀 × 두 arm 모두에서 R16-2 `CAP_BINDING`(10%, 짝 규칙)이 아닌 최소 cap; 없으면 C\_mem". 규칙 입력은 두 arm 모두이며 대칭이다.
- 공통: SLO는 G0의 A′ SLO, goodput 정의는 게이트 4, duration 합산(게이트 7), metric cliff 표시(게이트 6).

## 5. 규모 감 [유도 — PREREG에서 J1 셀 시간 모형으로 다시 낸다]

- 표준 셀 (i)·(ii): 모델 2 × cap 3 × 셀 2 × arm 2 = 24 조합. 각 조합에 rung 사다리 + seed 4를 곱한다.
- 경계 (iii): 모델 2 × cap 3 × ctx 3 × arm 1–2 = 18–36 조합.
- J1 교훈: 확장 단계를 기본 시나리오에 넣는다. B·X 계열은 arm별 prior를 쓴다(fused 계열 ≈1.10, PD-mux 0.70–0.85 [측정, J1]). 단일 Job 16 h 캡 안에 들어가는지는 PREREG의 GPU 사용률 계산으로 판단한다. 2 Job 분할(모델별)도 가능하나, 그러면 Q-B 이중차분이 Job 사이 비교가 된다. 그래서 **모델 둘을 한 Job에 넣는 쪽을 우선**한다.

## 6. 열린 질문 (사용자 결정 또는 PREREG 전 확인)

1. shape C 데이터셋(결정 5)이 이 Job 전에 정해져야 한다. 아니면 (ii)를 빼고 시작할지 정한다.
2. 경계 횡단 ctx {32k, 64k, 96k}에는 `context_length` 상향(현재 20480)과 **128k 프로브**(기존 결정 대기)가 필요하다. 대조 모델의 최대 길이도 확인해야 한다(Qwen2.5-7B 기본 32k, YaRN 확장 여부).
3. (iii)을 fused\_sym만으로 할지 정한다. 경계는 admission 성질이라 arm 무관이라는 가정이 있는데, 이 가정 자체가 미검증이다.
4. 후보2의 비교 arm "엔진 기본 비율 모드(`mamba_full_memory_ratio`)"는 radix ON에서만 존재한다[코드, `model_runner_kv_cache_mixin.py:218-246`]. F8(radix OFF)과 충돌하므로 **제외 [제안]**하고 범위 문장으로 처리한다.
5. Q-B의 대조 모델 효과는 아키텍처 외에도 모델 크기·가중치 메모리가 다르다는 교락을 갖는다. 그래서 "hybrid 고유" 문장은 이중차분의 **부호까지만** 쓴다(claims-auditor 검토 대상).

## 7. 금지 문안 (결정 8·VERDICT에서 승계)
"generic 스케줄러는 mamba state를 무시한다" · "Bullet/MuxWise가 C\_mem을 오예측한다" · "~52k에서 구속 전환"(실측 전) · "Claim E 반증" · J1 비의 크기 문장.

## 8. 다음 단계
I-5 구현 + GPU correctness gate(engine-porter) → §6 결정 → PREREG 작성 → 규칙층·하네스층 감사 → 봉인 → CPU preflight → `launch.sh` dry-run → 승인 → 제출.
