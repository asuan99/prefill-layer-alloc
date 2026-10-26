# 규칙층 감사 r2(최종) — PLAN_DP_routing rev1 (diff `00f5b763..e47b6bf9`), 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 감사 상한 도달 — 3회차 규칙층 감사 없음.

## 전체 판정
**`GO-with-caveats` — PREREG화 가능.** r1 死因 3건 해소, 새 死因 0. 조건: N-1…N-6 문장을 PREREG에 그대로 넣고 하네스층 감사에서 확인. 사용자 결정 1건(U-1) 잔존 — r1 처방 문안 1(Llama R2·R3)의 실현가능성 결함.

## r1 해소
해소: 1a·1b·3(τ)·3(봉인 2단계)·2a·2b·5a·4a·5d·6b·7(9.5 h, 세 곳 정정 일치)·8 Job 분할·MINOR 대부분. 부분 해소: 4b(GIL 문턱 [제안], 계측 OFF 충돌 미확인 → 하네스층), 6a(변화 trace 확인 판정 규칙 없음 → N-1), 6f(O-D3 결정 30 표 미작성 → PREREG).

## 새 문제
- N-1 (MAJOR) 변화 trace: 확인 판정 규칙·라벨·CI·불일치 처리 없음; q = 0.8에서 같은 offered rate면 goodput 비 ≤ 1/att\_den(0.8λ\*) — a = 0.90이면 이득 상한 +11.1%, att\_den ≥ 0.971이면 3% 문턱 도달 불가(H-D LO 절단과 같은 함정); "분모 arm λ\*"가 어느 혼합·arm인지 미정.
- N-2 (MAJOR) 분모 셀별 max(R0, R1, R1′)의 winner's curse — 분모 상향·G 하향(G < 1 주장에 반보수적), 선택 단위·추정량 미기술.
- N-3 (MAJOR) decode floor `T1 × ρ ≤ (1 − m)·θ_ITL`: (a) 프로젝트 예측기 ITL 식(rev2 §6-1 `max(ρ·T_dec, σ·T_span_iter)`)과 다름(span 항 없음) (b) T3 ρ 격자(D {24, 44}·bs {8, 32}·L {8,192, 32,768}) 밖은 클램프 — 짧은 집합 L̄이 8,192로 클램프돼 ρ 과대 → 큰 D 쪽 편향 (c) T1은 step 중앙값, 술어는 요청 내부 p95 — m = 0.1 근거 없음 (d) 만족 D가 없을 때 규칙 없음. ρ를 D 선택에 쓰는 것은 cm\_predict 선례가 있어 rev3 §1-3 금지 문구와 충돌하지 않는다고 판단.
- N-4 (MAJOR) n\_long 순환 정의(서비스 비중 → 엔진·D 의존 → D̂\_LEN은 n\_long 의존), 기준 엔진 미정.
- N-5 (MINOR) R4-D 집합별 D가 R3 cap 아래인지, D̂\_LEN = max(집합별 최소 적격 D) 미명시.
- N-6 (MINOR) 클래스 정의를 전 arm·Llama·변화 trace에 같게 적용한다는 문장 없음.
- N-7 (caveat, 예보) 혼합 30%·자연 출력 셀: f\_long ≈ 0.3 × 0.31 ≈ 0.094 < 0.1 → `ROUTING_DEGENERATE`(ARXSUM\_STATS 입력 > 8,760 = 30.0%); 60% 혼합 0.19, shape C 30% 혼합 약 0.22 통과.

## PREREG 작성 조건(원문)
- N-1: "변화 trace의 rate = 각 phase 혼합에서 분모 arm(N-2 규칙으로 정한 arm)의 λ*_SLO × q, q ∈ {1.0} [1차], {0.8} [기술값]. 확인 판정 = phase 가중 goodput의 seed 짝 ln 비(arm/분모)에 t(3) 95% CI. 1차 사다리 라벨과 부호가 같고 CI가 0을 배제하면 `CHANGE_TRACE_CONSISTENT`, CI가 0을 포함하면 `CHANGE_TRACE_UNRESOLVED`, 부호가 반대이면 `CHANGE_TRACE_CONTRADICTS`(이 경우 1차 라벨은 인용 금지). q = 0.8 비는 상한 1/att_den 때문에 3% 문턱 판정에 쓰지 않는다."
- N-2: "분모 arm은 교차적합으로 고른다. seed {1, 2}의 풀링 λ*로 R0·R1·R1′ 중 하나를 고르고 seed {3, 4}로 평가한 뒤, 역할을 바꿔 평균한다. 동률 폭은 3%이고, 동률이면 R1′ > R1 > R0 순으로 고른다. 선택된 분모 arm의 seed별 λ*로 짝 CI를 계산한다. seed마다 max를 고르는 것은 금지한다."
- N-3: "decode floor는 cm_predict의 ITL 식(prefill 창 안 `max(ρ·T_dec, σ·T_span_iter)`, rev2 §6-1)으로 판정한다. ρ·σ가 T3 격자 밖이면 `RHO_CLAMPED`를 집합별로 공시한다. 만족하는 D가 없으면 격자 최대 D를 쓰고 `FLOOR_INFEASIBLE`로 표시한다. m = 0.1은 중앙값과 p95의 차를 덮는다는 근거가 없는 [제안]이다(caveat)."
- N-4: "n_long의 서비스 비중은 vanilla `108v` 예측(T1 `108v` + T2, 엔진·D와 독립)으로 한 번 계산해 R1′·R3·R4-D·R4-DC가 공유한다. 이를 계산한 뒤에 D̂_LEN과 집합별 D를 고른다."
- N-5: "R4-D의 집합별 D는 R3의 cap 아래에서 고른다. D̂_LEN = max(집합별 최소 적격 D)."
- N-6: "클래스(τ 기준, Nano 요청 ID)와 클래스별 attainment ≥ a의 정의는 R0–R4-DC 전 arm, Llama 참조점, 변화 trace에 같게 적용한다."

## 승계 caveat 6–9
6. "혼합 30%·자연 출력 셀은 f_long ≈ 0.09 [예보]로 `ROUTING_DEGENERATE`가 될 수 있고, 그 경우 라우팅 판정이 없다."
7. "decode floor의 여유 m과 클라이언트 포화 문턱은 [제안] 값이며, floor는 step 시간 중앙값 기반이라 p95 술어 통과를 보장하지 않는다."
8. "N = 2의 R3·R4는 1:1 분할이 강제라 분할 규칙이 아니라 정수 제약 아래의 결과다."
9. "이 캠페인은 어느 결과로도 hybrid 고유성을 판정하지 않으며, `GO-with-caveats`는 게이트를 닫았다는 뜻이 아니다."

## 사용자 결정 U-1 — Llama 참조점 arm
Llama는 이 엔진의 PD-mux 경로 정확성 게이트를 통과한 적이 없고(G-α는 fused_sym만, `engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md:22-24`), 비용 항에 Llama D 축이 없다(rev3 §4 ⑥) → R2·R3는 미조율 static. (a) [권고] Llama 참조점을 R1·R1′(fused `108v`, Transformer에서 길이 라우팅이 HoL을 줄이는지)로 교체 / (b) R2·R3 유지 + Llama PD-mux 정확성 게이트 추가 + "D = Nano D̂ 이식, 미조율" 공시 / (c) Llama 제외(결정 30 형식 공시).
