# 설계 초안 rev3 — V(조립 검증 + oracle 격차)와 동적 여지 검사(static arm만), 2026-10-26

> **지위**: 설계 초안(PREREG 아님, 미봉인). 측정 0, GPU 0, 성능 판정 0. 결정 31(`../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`)과 대조표 `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md` §4 C-1·C-4에서 나왔다.
> **감사 이력**: 규칙층 r1 `AUDIT_rev36_planV_r1_2026-10-26.md`(V-O 死因 2·MAJOR 4, H-D 死因 2·MAJOR 3) → rev1 = 수리본 → 규칙층 r2 `AUDIT_rev36_planV_r2_2026-10-26.md`(최종, 새 문제 N-1 MAJOR·N-2…N-5 MINOR) → rev2 = N-1…N-5 반영본 → **rev3 = 결정 32(O-S1: 축 N (c), H-D (a)) 반영본**. 감사 상한 도달 — 이후 변경은 사용자 결정. rev0 대비 바뀐 곳은 각 항목 끝 `[r1 수리: …]`로 표시했다.
> **승계**: V의 예측기·셀·사다리·판정표·내부 앵커·귀무 예측기는 `PREREG_COSTMODEL_TERMS_rev2_2026-10-20.md` §6(이하 "rev2 §6")의 설계를 출발점으로 쓴다. 이 문서는 바뀌는 부분과 새로 더하는 부분만 적는다.
> 수치 라벨: [유도] · [예보](타 모델·타 기판 값, 근거 아님) · [제안] · [결정] · [미정](사용자 결정 또는 G0 PREREG 몫).

## 0. 왜 이 두 실험인가

비용 모델이 goodput으로 이어지는 사슬(대조표 §1) 중 ③ "모델이 고른 설정이 oracle에 가까운가"와 ⑤의 전제 "phase 경계 전환으로 이길 여지가 있는가"를 닫는다.

- **V-O(oracle 격차)**: rev2 §6의 V는 예측 λ\*가 큰 상위 2개 D만 돌려 "예측이 놓친 세 번째 D가 더 좋았을 가능성"을 caveat로 남겼다(rev2 §6-2, 등록 caveat 3). V-O는 D 격자 전체를 돌려 oracle을 실측하고 예측 선택의 후회(regret)를 잰다. 논문 방향 ②(비용 모델 기반 static 설정)의 핵심 증거다.
- **H-D(phase 경계 여지)**: 정본 HE0은 기존 동적 제어기가 best static을 못 넘었다고 확정했고, 그 범위를 "위치선정의 증거이지 동적 SM 공급의 천장의 증거가 아니다"로 줄였다(정본 §3 항목57·§1-17 배너: 달성 가능한 천장 미확정). H-D는 그 천장 중 **phase 경계에서만 전환하고 전환 비용이 0인 정책**의 여지를 static arm만으로 잰다. phase-oracle 여지가 없으면, 이 기판·이 trace·이 rate에서 phase 경계 static 전환으로는 이길 여지가 없다. **phase 내부 적응 정책의 여지는 닫지 않는다.** 정본 §1-19의 per-static oracle +2.1%(KISTI, n=1~2, overload-only, 강등)와 결합·비교하지 않는다. [r1 수리: H-D-5]

## 1. V-O — oracle 격차

### 1-1. rev2 §6에서 바뀌는 것
| 항목 | rev2 §6 | V-O |
|---|---|---|
| PD-mux arm | 예측 λ\* 상위 2 D (D ∈ {24, 34, 44}) | **D 격자 전체** {16, 24, 34, 44, 64} [제안] — T1 격자와 같은 D(실현 SM 기준, §1-8) |
| fused arm | `fused_tuned`(G0 튜닝 argmax) | 같음 + `fused_default`(SGLang 기본, 미조율 대조) [제안] |
| 판정 | λ\*\_pred 대 λ\*\_meas(오차 e) | e 판정(rev2 §6-3 승계) + **두 축 후회 판정**(1-3) |
| 귀무 | N0(G=1), N1(T1 단독) | N0·N1 + **N2·N2′**(1-2) |
| 셀 | arXiv-Sum 자연 출력, shape C(강제 출력 4,096) | 같음 [결정 5·12]. shape C 출력 {1k, 4k}는 **V-O 봉인 전에** G0 PREREG에서 정한다 [r1 수리: V-O-9] |
| 소거 변형 | v1–v5 | v1(ctx 항 0)·v2(KV 전용 C\_mem, s = 0)·v3(ρ = 1)·v4(외삽 허용)·v5(F-whole) **전부 승계**, 각 변형의 D̂\_v와 후회를 기술값으로 병기 [r1 수리: V-O-7] |
| D̂ 동률 | "큰 D" | 예측 λ\* 차 < 3%면 큰 D(rev2 승계) [r1 수리: V-O-7] |

`fused_default`는 미조율 분모이므로 어떤 문장에서도 "비용 모델/PD-mux가 fused보다 낫다"의 근거로 쓰지 않는다(confound 2, 정본 §1-1 철회 전례). [r1 수리: V-O-4]

### 1-2. 귀무 N2·N2′ — "비용 모델 없이도 고를 수 있었나"
- **N2** = "HE0(KISTI A100·HE0 모델)에서 최적이었던 D=44를 옮긴 상수 — 정본 규칙('peak decode 부하 기준 decode-heavy static')의 Nano 인스턴스가 아니다." [r1 수리: V-O-5]
- **N2′** = 격자에서 가장 decode-heavy한 D(64).
- 비용 모델 선택 D̂ = `cm_predict`의 λ\*\_pred argmax(rev2 §6-1 예측기, D 격자 전체로 확장, 동률 규칙 1-1).

### 1-3. 후회 판정 — 두 축, 교차적합 [r1 수리: V-O-1·V-O-2]
- **교차적합**: pass 1(seed 1, 2)로 oracle 집합을 고르고 pass 2(seed 3, 4)로 후회를 평가한 뒤, 역할을 바꿔 두 값을 평균한다. oracle 동률 폭 = max(3%, 선택 pass의 R9-6 해상도). seed 분할로 해상도가 √2배 거칠어짐을 공시한다. 평가 pass 해상도 > ln 1.05인 셀은 데이터 전에 `CM_NEAR_UNREACHABLE_RESOLUTION`으로 선언한다(`V_REG`).
- 후회 `R(x) = ln(λ*_meas(oracle) / λ*_meas(x))`(교차적합 평균).
- **축 O**(PD-mux D arm만의 oracle, R\_PD): `ORACLE_UNRESOLVED`(λ\* 미해결 또는 oracle 집합이 해상도 안에서 안 갈림) / `CM_IN_ORACLE`(D̂ ∈ oracle 집합) / `CM_NEAR_ORACLE`(R\_PD(D̂) ≤ ln 1.05) / `CM_FAR_FROM_ORACLE`.
- **축 N**(기본 규칙 대비, N2와 N2′ 각각, 결정 32 (c)): `R(N2) − R(D̂) = ln(λ*(D̂)/λ*(N2))`이므로 oracle 항이 상쇄된다 → **교차적합 없이** 전체 seed 4의 seed 수준 짝차 `d_s = ln(λ*_s(D̂)/λ*_s(N2))`(seed s마다 같은 seed의 두 arm λ\*)에 t(3) 95% CI. D̂ = N2이면 `CM_EQUALS_DEFAULT`. 아니면 CI 하한 > 0 ∧ 점추정 ≥ ln 1.03 → `CM_BEATS_DEFAULT`, CI가 0을 포함 → `CM_TIES_DEFAULT`, CI 상한 < 0 → `CM_WORSE_THAN_DEFAULT`. seed별 λ\*가 해결되지 않은 seed가 있으면 그 셀의 축 N은 `CM_TIES_DEFAULT`가 아니라 `ORACLE_UNRESOLVED`와 같은 취급(`DEFAULT_UNRESOLVED`)으로 둔다.
- **헤드라인은 두 축을 반드시 함께 인용한다.** fused를 포함한 전체 oracle 대비 후회 R\_all은 기술값이다(D̂는 D 격자 안에서만 고르므로 R\_all은 PD-mux 대 fused 격차를 섞는다). [r1 수리: V-O-4]
- **셀 간 집계**: 축마다 두 셀 중 낮은 라벨을 전체 라벨로 한다. 라벨 순서 — 축 O: `CM_IN_ORACLE` > `CM_NEAR_ORACLE` > `CM_FAR_FROM_ORACLE` > `ORACLE_UNRESOLVED`, 축 N: `CM_BEATS_DEFAULT` > `CM_EQUALS_DEFAULT` = `CM_TIES_DEFAULT` > `CM_WORSE_THAN_DEFAULT` > `DEFAULT_UNRESOLVED`. [r1 수리: V-O-7, r2 N-4]
- **CI 추정량 [결정 32]**: 축 N은 위 (c). 축 O는 문턱 라벨이라 CI를 쓰지 않는다. percentile bootstrap은 쓰지 않는다(정본 §3 항목56(A)).
- 허용 문장 틀: "이 Job의 (셀)에서 비용 모델이 고른 static D는 PD-mux D 격자 oracle 대비 후회 …%(축 O: …)였고, HE0에서 옮긴 D=44 대비 …(축 N: …)였다." 금지: "비용 모델이 최적 설정을 찾는다"(두 셀·한 모델 범위 밖), "hybrid 고유 항 덕분"(V-O는 귀속을 판정하지 않음 — 소거 변형 후회는 기술값).

### 1-4. 사다리 — 예측기 비의존 [r1 수리: V-O-3]
oracle 측정 사다리가 `cm_predict`에 의존하면 모델이 가장 크게 틀린 arm에서 oracle이 censoring된다. 둘 중 하나를 봉인한다 [미정, O-V4]: (α) arm 공통 기하 사다리(인접 비 1.2)가 [min λ\*\_pred × 0.5, max λ\*\_pred × 2]를 덮게 한다 / (β) arm 중심 사다리(rev2 §6-2)를 유지하되 교차가 사다리 밖이면 같은 Job 안에서 그 방향으로 2 rung을 추가하는 봉인 규칙(추가는 최대 1회, 그래도 밖이면 `BELOW/ABOVE_LADDER`로 그 arm을 oracle 후보에서 "하한 정보만"으로 둔다; `ABOVE_LADDER` arm의 하한이 해결된 arm들의 최대 λ\*를 넘으면 그 셀의 축 O는 `ORACLE_UNRESOLVED`이다 [r2 N-3]).

### 1-5. 비용 [유도, Job 크기 결정용 — 측정 아님]
rev2 V(3 arm/셀) = J-V1 3.8 h + J-V2a 3.7 + J-V2b 3.7 ≈ 11.2 GPU-h(rev2 §7). arm 셀당 3 → 7이면 **arm 비례 외삽으로 약 26 GPU-h** — 앵커·관측자 블록은 arm 수에 비례하지 않아 과대 쪽, 사다리 확장(1-4)은 미포함이라 과소 쪽이다. [r1 수리: V-O-6] Job당 ≤ 6 h 원칙으로 셀·pass별 6 Job 안팎.

복구 비용 유도: arm·셀 단위 26 / 14 ≈ 1.86 GPU-h를 곱했다 — (a) shape C 3 arm ≈ 5.6, (b) 비후보 5 arm × 2셀의 사다리 2/5 ≈ 26 × (5/7) × (2/5) ≈ 7.4, (c) 2셀 × 1 arm ≈ 3.7(모두 arm 비례 외삽).

줄이는 선택지 [미정, O-V1] — **결정 30 측정점 삭제 원칙 형식**(예산은 허용 사유 "물리적 불가 / 더 강한 증거"가 아니다): [r1 수리: V-O-8]

| 안 | 삭제·축소 | 사유 분류 | 잃는 범위 | 복구 비용 |
|---|---|---|---|---|
| (a) | shape C에서 D 2개(예측 하위)와 fused\_default 제외 | 예산(허용 사유 아님) | shape C 축 O가 "상위 3 D 안의 oracle"로 축소 | 약 +5.6 GPU-h [유도] |
| (b) | 1단계 사다리 3 rung(j = −1, 0, +1), oracle 후보 2개만 2단계 5 rung | 예산(허용 사유 아님) | caveat 문자 그대로: "oracle은 1단계 후보 집합 안의 oracle" | 약 +7.4 GPU-h [유도] |
| (c) | `fused_default` 제외 | 예산(허용 사유 아님) | R\_all의 미조율 대조 없음(헤드라인 영향 없음, 1-3 축 O·N은 PD-mux D만) | 약 +3.7 GPU-h [유도] |

## 2. H-D — phase 경계 여지(static arm만)

### 2-1. 무엇을 재는가
- trace: phase 교대 — **W1**(짧은 대화, ShareGPT 계열)과 **W3**(긴 입력, arXiv-Sum 또는 shape C) 블록을 번갈아 둔다. phase 길이 2종 [제안: 120 s, 600 s].
- **rate** [r1 수리: H-D-3]: 용량 측정 뒤 `rate_W = q × λ*_best-static(W)`, q ∈ {0.8, 1.0}으로 데이터 전에 봉인한다. 각 q에서 TTFT가 SLO ± 5% 안에 드는 요청 비율 > 10%인 arm·phase는 `CLIFF_ADJACENT`(게이트 6). 용량 이하에서는 H = 0이 정의상 성립한다(정본 §1-19 "용량 이하 = 전부 통과") — `ABSENT`는 모든 q에서 성립할 때만 쓴다.
- **두 변형** [r1 수리: H-D-1]: (i) **연속 변형** — phase가 이어 붙는다(이월 있음). (ii) **깨끗한 시작 변형** — 각 phase 블록 앞에 drain 구간(도착 0, running + queue = 0까지)을 둔다. phase-oracle의 상한 성질은 (ii)에서만, "phase 경계에서만 전환하고 전환 비용이 0인 정책"에 대해, **"backlog가 많을수록 phase goodput이 낮다"는 단조성 가정** 아래 성립한다(공시). (i)의 phase-oracle은 이월이 arm마다 달라 양방향으로 편향되므로 **상한이 아니라 기술값**이다.
- arm: **런타임 전환이 실현 가능한 arm = 같은 PD-mux 엔진의 D 집합 {16, 24, 34, 44, 64}**. fused\_tuned는 함께 돌리되 phase-oracle 후보에 넣은 값은 기술값이다(fused↔PD-mux 런타임 전환은 실현 불가). [r1 수리: H-D-1] fused\_tuned는 이 교대 trace에서 튜닝되지 않았음을 공시한다(게이트 8). [r1 수리: H-D-6]
- **전환 가능 구성 대조 arm**: 동적 정책이 실제로 쓸 다중 division 부팅에서 D\* 하나에 고정한 arm 1개 — "전환 가능 구성 자체의 비용"을 잰다. 실현 가능성은 engine-porter 확인 [미정, O-H3].
- 지표: phase별 goodput(그 phase에 **도착한** 요청 기준, 정본 술어: TTFT ≤ SLO ∧ 요청 내부 ITL p95 ≤ SLO), 전체 goodput. 게이트 5에 따라 TTFT/ITL p50/p95/p99, **가상 phase-oracle 정책의 전환 수·arm 체류 분포** 병기. [r1 수리: H-D-6]

### 2-2. 판정 [r1 수리: H-D-2·H-D-4]
- **선택 단위 = phase 종류(W1, W3)**: phase 종류마다 arm 하나를 고르고 그 종류의 모든 인스턴스·seed를 풀링한다(인스턴스별 argmax 금지).
- **seed 8과 교차적합 [결정 32 (a)]**: seed 8개를 선택(seed 1–4)과 평가(seed 5–8)로 나누고 역할을 바꿔 평균한다. CI는 평가 쪽 seed 수준 짝차 `h_s = ln(G_phase-oracle,s / G_best,s)`(같은 seed의 trace에서 계산)에 t(n−1) 95%를 쓰고, 두 역할의 CI 중 **더 넓은 것**을 라벨에 쓴다. percentile bootstrap은 쓰지 않는다.
- `G_best` = 전체 goodput이 가장 큰 단일 static D arm(평가 pass), `G_phase-oracle` = phase 종류별 선택 arm의 평가 pass phase goodput을 phase 길이로 가중 합산, `H = ln(G_phase-oracle / G_best)`.

| 라벨 | 조건 |
|---|---|
| `DYNAMIC_HEADROOM_PRESENT` | 깨끗한 시작 변형에서 교차적합 H ≥ ln 1.03 ∧ t(n−1) 95% CI 하한 > 0, **두 phase 길이 모두**, 어느 한 q에서 |
| `DYNAMIC_HEADROOM_ABSENT` | 깨끗한 시작 변형에서 **두 phase 길이·모든 q** CI 상한 < ln 1.03 |
| `HEADROOM_UNRESOLVED` | 그 외(한 길이에서만 성립 포함) |

연속 변형의 H는 같은 규칙으로 계산하되 기술값이다. "rep SD" 문구는 쓰지 않는다.

- **cliff 규칙 [r2 N-1]**: 어느 phase 종류의 선택 arm 또는 G\_best arm이 `CLIFF_ADJACENT`인 q에서는 라벨이 `HEADROOM_UNRESOLVED`이고, 그 q는 ABSENT의 "모든 q" 조건에서 빠지지 않는다(q = 1.0은 ABSENT를 막을 수만 있고 만들어 내지 못한다).
- **다중성 비대칭 공시 [r2 N-5]**: PRESENT는 "어느 한 q", ABSENT는 "모든 q"라서 PRESENT 쪽이 다중성으로 유리하다. PRESENT는 HyMux PREREG 착수를 허가할 뿐이며 성능 주장이 아니므로 이 비대칭을 허용하고 결과 보고에 공시한다.

- 결과의 쓰임(결정 31): `ABSENT`면 HyMux 구현을 보류하고 논문은 static 설정(방향 ②)으로 간다. `PRESENT`일 때만 HyMux X1–X10(검증 계획 문서) PREREG를 쓴다. `UNRESOLVED`면 사용자 결정.
- 금지: H를 "동적 정책이 얻을 이득"으로 인용(상한 성질은 깨끗한 시작 변형·단조성 가정 아래에서만), `ABSENT`를 "동적 제어는 원리적으로 불가"·"HE0 천장이 닫혔다"로 확장(이 기판·이 trace·이 rate·phase 경계 전환 한정, 정본 §3 항목57 미해소).

### 2-3. 함정(정본에서 결론을 뒤집은 적 있는 것)
- phase 경계 이월: 연속 변형에서는 phase 첫 구간(warm 창 [제안: phase 길이의 10%])을 제외한 값과 제외 전 값을 둘 다 보고한다. 판정은 깨끗한 시작 변형.
- metric cliff(게이트 6): 2-1 rate 규칙.
- duration 합산(게이트 7): 여러 라운드·phase 합산 시 duration은 합산.
- LO phase 절단: 저부하 phase는 도착률이 goodput을 묶어 arm 간 차가 0이 된다(HE0 범위 축소 근거). 그래서 rate를 용량 기준 q로 둔다.

### 2-4. 비용 [유도]
static D arm 5 + fused\_tuned 1 + 전환 가능 구성 대조 1 = 7 arm × phase 길이 2 × q 2 × 변형 2 × **seed 8**(결정 32). rev0의 8–12 GPU-h(arm 6 × 길이 2 × seed 4)에서 q·변형 축으로 약 4배, seed로 2배가 되어 **약 64–96 GPU-h**(깨끗한 시작 변형의 drain 시간·용량 측정 미산정, arm 비례 외삽). **이 규모는 운영 규약상 축소안 결정(O-H4)이 사실상 선결이다.** — 축소안 [미정, O-H4]: q = 1.0만으로 1차 → `UNRESOLVED`일 때만 q = 0.8 추가(결정 30 형식 공시 필요; **이 축소안 아래에서는 q = 1.0 결과만으로 ABSENT를 선언할 수 없다** [r2 N-2]), 연속 변형은 phase 길이 1개만. W1·W3의 λ\*\_best-static 용량 측정 비용은 위 합계에 **미포함**이다 [r2 N-2]. V-O와 arm·config·boot를 공유하므로 같은 Job에 블록으로 넣으면 부팅 비용을 아낀다 [제안 — 같은 비교의 arm은 한 Job 안].

## 3. 순서와 선행 조건
1. 비용 항 Job(J-A1·J-A2·J-B·J-C·J-H) 봉인·제출 — 진행 중 경로(smoke → 하네스층 감사 → 봉인).
2. G0 PREREG에서 fused\_tuned 튜닝·shape C 출력 길이·workload 정의를 먼저 정한다(결정 11·15). [r1 수리: V-O-9]
3. **V-O 구조 봉인 — 항 Job fetch 전** [권고]: 예측기 코드·D 격자·두 축 후회 판정·N2/N2′·소거 변형·사다리 규칙. 계수에서 나온 예측값만 fetch 뒤 `V_REG` addendum으로 봉인(rev2 §6-1 "봉인 시점" 승계).
4. 항 Job 결과 분석(rev3.6′ §1-9 AJ-1…AJ-4 포함) → `V_REG` → V-O·H-D Job 제출 승인.
5. H-D 결과로 HyMux 구현 여부 결정.

## 4. 열린 결정 [미정]
- O-V1 V-O 비용 축소안 (a)/(b)/(c)/없음 — 1-5 표(결정 30 형식).
- O-V2 shape C 출력 길이 1k/4k(G0 PREREG와 함께, V-O 봉인 전).
- O-V3 N2 규칙: N2(D=44 이식)·N2′(D=64) 둘 다(초안) / 하나만.
- O-V4 사다리 (α) 공통 기하 / (β) arm 중심 + 적응 2 rung.
- O-H1 phase 길이 {120 s, 600 s}.
- O-H2 H-D를 V-O Job에 블록으로 합칠지(초안 권고: 합침).
- O-H3 전환 가능 구성 대조 arm(다중 division 부팅)의 실현 가능성 — engine-porter 확인.
- O-H4 H-D 비용 축소안.
- ~~O-S1 CI 추정량~~ → **결정 32로 닫힘**: 축 N (c), H-D (a).

## 5. 등록 caveat(감사 r1 승계, 인용 금지 문장)
- "V-O의 후회는 두 셀·Nano 한 모델·이 Job들의 값이다. N2(D=44)는 KISTI HE0에서 옮긴 상수다."
- "H-D의 ABSENT는 이 trace·rate·phase 경계 전환 정책에 한정되며, 정본 §3 항목57('HE0 ⇏ 천장 소멸')을 해소하지 않는다."
