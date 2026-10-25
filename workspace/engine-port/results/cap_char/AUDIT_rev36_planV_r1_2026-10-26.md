# 규칙층 감사 r1 — PREREG rev3.6 §1-9(A-1…A-4)·PLAN_V rev0(V-O·H-D), 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 대상 커밋 `70ed6ca5`. 같은 대상 최대 2회 중 1회차. 2회차는 수리 diff만 본다.

## 전체 판정
| 대상 | 등급 | 요지 |
|---|---|---|
| rev3.6 §1-9 | 수리 전 NO-GO, 수리 후 봉인 가능 | 死因 3(A-1·A-2·A-3), A-4 MINOR. 측정 절(§1-1…§1-8) 영향 없음 |
| PLAN_V V-O | 수리 후 PREREG화 가능 | 死因 2(라벨 순서가 기본 규칙 비교를 가림, winner's curse), MAJOR 4 |
| PLAN_V H-D | 수리 후 PREREG화 가능 | 死因 2("상한" 전제 거짓, null에서 거짓 PRESENT), MAJOR 3 |

감사 제안: §1-9는 fetch 뒤 분석 규칙이므로 §1-9를 떼어 측정만 먼저 봉인하고 수리본을 fetch 전에 따로 커밋해도 된다(사용자 몫).

## 항목별
| ID | 판정 | 요지 [근거 수치는 analytic_terms 기반 유도 또는 KISTI 예보] |
|---|---|---|
| A-1 BW(D) | 死因 | a\_D = t0 + W/BW(D) → ω 모형이 참이어도 t0 > 0이면 r\_a < 4(t0 3.70 ms에서 3.2 미만). D ≤ 64 격자로는 D\_sat ∈ {64, 80, 108} 구분 불가 |
| A-2 시간 knee | 死因 | 변별력 없음(교훈 9): non-state 요청당 비용 0.099 ms면 state 0에서도 비 0.56–0.70으로 HELD |
| A-3 θ′ | 死因 | SM 감소·decode 동거 부하·lockstep 관측 지연이 D와 함께 움직임. 참 θ=1에서도 L=1,024 θ′ 0.55–0.87. L 집계 규칙 없음 |
| A-4 조성 회귀 | MINOR | 입력 셀 목록 불일치, 최소 점 수 규칙, Δ\_ideal 함수 미지정 |
| A-공통 | MINOR | ID가 결정 28·§11-3 A-1…A-37과 충돌 → AJ-1…AJ-4 |
| V-O-1 라벨 순서 | 死因 | D̂와 N2가 모두 oracle 집합이면 CM\_HITS\_ORACLE가 먼저 걸려 "기본 규칙보다 낫지 않다" 표현 불가 |
| V-O-2 oracle 선택 | 死因 | 같은 데이터로 argmax·후회 평가(winner's curse). null σ 2.5%에서 HIT/NEAR/둘 다 아님 = 0.47/0.24/0.28 |
| V-O-3 사다리 | MAJOR | arm 중심 = λ\*\_pred → 모델이 가장 틀린 arm에서 oracle censoring |
| V-O-4 fused 포함 | MAJOR | fused\_default는 미조율 분모 |
| V-O-5 N2 = d44 | MAJOR | HE0(KISTI·Zamba2) 상수이지 정본 규칙의 Nano 인스턴스가 아님 |
| V-O-6 비용 | MINOR | 26.1 산술은 맞으나 앵커·관측자 비비례, 사다리 확장 미포함 |
| V-O-7 rev2 정합 | MINOR | 소거 v4·v5 누락, D̂ 동률·셀 간 집계 규칙 없음 |
| V-O-8 O-V1 (a)(c) | MAJOR | 예산 사유 삭제는 결정 30 허용 사유 아님 → §11-4 형식 공시 + 사용자 결정 |
| V-O-9 순서 | MINOR | shape C 출력 길이가 G0 의존인데 V-O를 G0 전에 봉인 |
| H-D-1 "상한" | 死因 | phase 이월 양방향, phase 내부 적응 미포함, fused↔PD-mux 전환 실현 불가 → 상한 아님 |
| H-D-2 문턱 | 死因 | null(arm 동일)에서 phase SD 5%면 P(H ≥ ln 1.03) 23.7%, "rep SD" 미정의 |
| H-D-3 rate | MAJOR | 용량 이하이면 H=0이 정의상 성립(정본 §1-19) |
| H-D-4 선택 단위 | MAJOR | phase 인스턴스별/종류별 미등록 |
| H-D-5 정본 범위 | MAJOR | "어떤 동적 정책도 여지 없음"은 정본 §3 항목57·§1-17 배너(천장 미확정)와 충돌, §1-19/§1-20 선례(+2.1%) 관계 없음 |
| H-D-6 게이트 5 | MINOR | 가상 정책의 전환 수·체류 병기 필요 |

## 수리 처방(요약 — 원문 문안은 PREREG rev3.6′·PLAN rev1에 반영)
- AJ-1: `a_D = t0 + κ/D` 적합(D ≤ 64 1/D 양립 라벨) + t0 무관 꺾임 지표 `s_108 = (a_64 − a_108u)/(a_16 − a_64)`, 사전값 0/0.067/0.136, 두 Job·두 추정량 불일치 라벨. r\_a·BW\_eff는 기술값.
- AJ-2: 라벨 폐지, 기술값으로 강등(+a31 b 참조, 계수 불안정 꼬리표).
- AJ-3: L ∈ {32k, 64k} CONFIRMED 풀링, lag\_share 상한 3%, 라벨명 `T2P_COEXIST_EXP_*`, 원고 §3.2를 닫지 않음.
- AJ-4: 공통 실행 가능 5셀, 항별 W-INDEP, 4점 필수, Δ\_ideal은 커밋된 분석 함수, decode 수준차는 H 근거 아님.
- V-O: 두 축(oracle 축 PD-mux만 / 기본 규칙 축 교차적합 CI), seed 교차적합 oracle, 사다리 비의존화, fused\_default 금지 문장, N2 문구 정정, O-V1 §11-4 형식, 비용 표기·v4·v5·동률·집계 규칙, shape C 길이 선결.
- H-D: 깨끗한 시작 변형 + 전환 가능 arm 한정, phase 종류별 선택·교차적합 CI 라벨, rate = q × λ\*\_best-static(W) q ∈ {0.8, 1.0}, §0 문장 교체·§1-19 선례 문장, 게이트 5 병기·fused\_tuned 미튜닝 공시.

## 등록 caveat(승계)
- AJ-1은 D ≤ 64의 1/D 양립과 64–108 꺾임 여부까지만. "SM 80개에서 포화"·"HBM 포화" 금지.
- AJ-2 c\_knee는 요청당 항 전체의 비이며 state 항이 decode 시간에 보인다는 증거가 아니다.
- AJ-3 θ′는 decode 1개 동거와 SM 감소의 합성 지수(L ≥ 32k)이며 원고 "prefill 처리율 ∝ P"를 닫지 못한다.
- AJ-4 β1은 층 교체(가중치 바이트 변화 포함)의 총효과, 4점·자유도 2 기술값.
- V-O 후회는 두 셀·Nano 한 모델·이 Job들의 값. N2(D=44)는 KISTI HE0에서 옮긴 상수.
- H-D의 ABSENT는 이 trace·rate·phase 경계 전환 정책 한정이며 정본 §3 항목57("HE0 ⇏ 천장 소멸")을 해소하지 않는다.
