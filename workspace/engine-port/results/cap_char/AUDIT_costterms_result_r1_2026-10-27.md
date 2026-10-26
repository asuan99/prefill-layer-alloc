# 결과 감사 r1 — RESULT_costterms_2026-10-26.md (+ supplement, AJ_full), 2026-10-27

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 같은 대상 최대 2회 중 1회차. 결정 38: 새 측정 처방 없음.

## 전체 판정
**PLAUSIBLE(조건부)** — 원자료 대조로 라벨·수치 전부 재현, 라벨 오류 0, 반증 라벨 0. 문장 수정 F-1…F-10 반영 전 promote 금지. MAJOR 3: F-1(T4 "9/10" 프레이밍·ratio_A 예측 이탈 누락), F-3(S-6을 "등록 규칙"으로 오기재·"실효 47% 피크"), F-4(CM-2 "재현"·"124k까지 연장"). 보충 S-1·S-2·S-3·S-5·S-7은 등록 규칙을 하네스 함수 그대로 적용(반전 여지 없음), S-6만 fetch 뒤 미등록 기술 비(걸린 라벨 없음).

## 확인된 문제
- AJ-2 J-H1: `aj_analysis.aj2_jh1`(:351-372)이 하네스 형 라벨을 보지 않음 → b(a22)·b(a31)=0.077·기울기 0.0070은 §1-9 공통 규칙(PREREG:205)상 `NA_FORM`. b(a4)·b(a13)은 개별 기술값만.
- J-C 키 충돌 범위: t2·t2prime(TP2가 앵커 덮음, fits.t2prime v 0.097 = TP2)·theta_prime(J-C D44만, 등록 대상 아님)·**labels.q**(job·round·tp 없음, TP2+앵커 4 boot 묶음, q 1.065–1.087 모두 < 1.10 → 라벨 불변). 영향 없음: T1·t1_client_secondary·agreement·t2_long·ratios·J-A·J-H1·AJ-3.
- T4: 판정 HELD 9셀 중 8셀은 cap 구속(n_A = n_B = cap)이라 A·B를 가르지 않음(교훈 9). cap 비구속 점 2개: fused cap 48 L 65,536 n_real 21 = n_B 21(HELD), C_A 114 L 8,800 n_real 92(n_B 78, n_A 114 사이, MISS). **봉인 ratio_A 예측 0.68 ± 0.05(PREREG:156,164) → 실측 0.81 이탈 미기재**.
- "다른 노드"는 확인 불가(hostname = pod 이름) — 다른 GPU UUID·pod·시각의 Job 효과 한 번의 실현값.
- 감사 승계 문장 2개(PREREG:605) 본문 누락.
- T3 ρ 합성 성분에 "부하 의존 관측자 비용(TEL ON, 미분리)" 누락(CONSENSUS:8798, PREREG:395); "D24 > D44 소폭"은 rep 2·차 < 3%·D24 동거 클럭 더 낮음 → 분리 불가.

## 반전 시험(감사자 계산): 반전 없음
J-H1 q 분모(이차식/log-log/선형 보간: real a4 1.097/1.085/1.089) · P3 추정량(모형 없는 ctx 차분: bs 8 3.11·5.36·7.32, bs 32 3.39·5.79·8.05, 모두 ±25% 안) · CLOCK_SENSITIVE(노출 20% 초과 창 0, 최대 2.4%) · J-C q 풀링(TP2 1.065–1.071 / 앵커 1.084–1.087).

## 수정 문안 F-1…F-10 (요지 — 원문은 감사 보고 전사본 기준으로 RESULT에 반영)
F-1 T4 프레이밍(cap 비구속 2점, ratio_A 0.68 → 0.81) · F-2 AJ-2를 예측 대비 잔차 표에서 빼거나 "등록 예측 아님" · F-3 S-6 = 미등록 기술 비, "실효 47% 피크" 삭제, b/b₀·c/c₀ 해석 제한, T2 계수 CLOCK_SENSITIVE NOT_COMPUTED · F-4 J1 기울기 대비는 "mem 0.82 결과, 기술값"으로, "124k까지 연장"은 §1-2′ 허용 틀(Nano −0.9…+1.2%, 연결비 0.991–1.000, Llama −1.8·−3.1%) · F-5 "다른 노드" → 다른 GPU UUID·pod · F-6 승계 문장 2개 삽입 · F-7 ρ 합성 성분 + "D24 > D44" 삭제/단서 · F-8 키 충돌 범위(labels.q 포함) · F-9 "bs 96 한 셀(ctx 2k) 외삽", 볼록껍질, γ_j·AJ-1은 기각된 F1 계수 · F-10 J-C F1 통과와 J-A bs 96 해석 엮는 문장 삭제.

## promote 후보(등급)
1 CONFIRMED CM-5 부팅 로그 사실(s·k ±0.05%, 풀 합 2% 안) · 2 CONFIRMED(scoped) CM-1 F1은 P = 108만, sticky-D는 bs 96 한 셀 외삽 13.6–18.3%로 F2/F3 · 3 CONFIRMED J-A T1 CLOCK_SENSITIVE 없음 / NYS T2·T2′·T4 전력 상태 · 4 CONFIRMED(라벨) T2 q 라벨 · 5 CONFIRMED(scoped) 장문 LONG_SERVED, TP2 PD-mux는 108u 한정 124k · 6 CONFIRMED T3 power cap 노출 대비·ρ 총효과 / NYS 간섭 크기·D 순서 · 7 NYS 산식 B 재현(F-1 문안만) · 8 CONFIRMED(J-H1 한 Job) 조성 P1·P2·P4, P3 조건부 · 9 CONFIRMED TP2 F1·F2·F5 / NYS F4·F3 · 10 CONFIRMED(라벨) AJ-1·AJ-3 · 11 인용 금지 AJ-2 J-H1 · 12 PLAUSIBLE 두 Job T1 F1 계수 차 ≤ 2.7%.
promote 금지: "산식 B 9/10", "mem 0.6에서 J1 재현", "실효 47% 피크", "KV 항 HBM 하한 근접", "124k까지 함수 형태 연장", "ρ D24 > D44", J-H2 조성별 SM 민감도 단조.
필요(처방 아님): T2·T2′ CLOCK_SENSITIVE·T4 MISS 원인은 클러스터 볼륨 원자료 필요.
