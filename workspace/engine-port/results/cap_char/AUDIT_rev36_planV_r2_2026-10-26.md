# 규칙층 감사 r2(최종) — r1 수리 diff `70ed6ca5`→`846a4ce0`, 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 감사 상한 2회 도달 — 이후는 사용자 결정.

## 최종 판정
- **PREREG rev3.6′ §1-9(AJ-1…AJ-4): 봉인 가능 `GO-with-caveats`.** r1 死因 3·MINOR 2 전부 해소, 새 死因 없음. caveat는 §1-9 "해석 제한 [봉인]"·"금지 [봉인]"에 이미 있음. AJ-3 실현 가능성 [유도]: L=32k·D16 기준점 lag\_share ≈ 2.5%.
- **PLAN\_V rev1: 사용자 결정 1건 선결 후 PREREG화 가능.** r1 死因 4 중 3 해소, V-O-2·H-D-2는 부분 해소(교차적합으로 선택 편향 제거, CI 추정량 문제 잔존).

## 수리가 새로 만든 문제(처방 문안 → PLAN rev2에 반영)
- N-1 (MAJOR, H-D): q = 1.0이면 best arm이 정의상 TTFT≈SLO 기울기 위. → "어느 phase 종류의 선택 arm 또는 G\_best arm이 `CLIFF_ADJACENT`인 q에서는 라벨이 `HEADROOM_UNRESOLVED`이고, 그 q는 ABSENT의 '모든 q' 조건에서 빠지지 않는다."
- N-2 (MINOR): O-H4 축소안(q=1.0 먼저)과 "ABSENT는 모든 q" 충돌 → 축소안 아래에서는 q=1.0만으로 ABSENT 선언 불가. W1 λ\*\_best-static 용량 측정 비용 미산정.
- N-3 (MINOR, V-O (β)): "`ABOVE_LADDER` arm의 하한이 해결된 arm들의 최대 λ\*를 넘으면 그 셀의 축 O는 `ORACLE_UNRESOLVED`."
- N-4 (MINOR): 축별 라벨 순서 — O: IN > NEAR > FAR > UNRESOLVED, N: BEATS > EQUALS = TIES > WORSE.
- N-5 (MINOR): PRESENT("어느 한 q") 대 ABSENT("모든 q") 다중성 비대칭 공시.

## 사용자 결정 필요 — CI 추정량(r1 처방 자신의 결함)
- seed 4개의 percentile bootstrap 95% CI는 보정되지 않음: 시뮬레이션 피복률 n=4에서 0.806, 교차적합 fold 단위(n=2)에서 0.493(정본 §3 항목56(A)가 금지한 경우). fold 단위 t(1)은 배수 12.7이라 PRESENT·BEATS에 사실상 도달 불가.
- 선택지: (a) seed 8(H-D·V-O 비용 약 2배) + seed 수준 짝차 t(n−1) / (b) seed 4 + t(n−1), `HEADROOM_UNRESOLVED`·`CM_TIES_DEFAULT`가 기본 결과임을 데이터 전 공시 / (c) 축 N은 R(N2) − R(D̂) = ln(λ\*(D̂)/λ\*(N2))로 oracle 항이 상쇄되므로 교차적합 없이 전체 seed 짝 t(3).
- **감사 권고: 축 N = (c), H-D = (a).**
