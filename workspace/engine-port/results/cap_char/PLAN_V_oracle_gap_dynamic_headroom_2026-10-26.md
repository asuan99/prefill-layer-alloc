# 설계 초안 rev0 — V(조립 검증 + oracle 격차)와 동적 여지 검사(static arm만), 2026-10-26

> **지위**: 설계 초안(PREREG 아님, 미봉인, 감사 전). 측정 0, GPU 0, 성능 판정 0. 결정 31(`../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md`)과 대조표 `reports/hymux_review/EXPERIMENT_CROSSWALK_2026-10-26.md` §4 C-1·C-4에서 나왔다.
> **승계**: V의 예측기·셀·사다리·판정표·내부 앵커·귀무 예측기는 `PREREG_COSTMODEL_TERMS_rev2_2026-10-20.md` §6(이하 "rev2 §6")의 설계를 출발점으로 쓴다. 이 문서는 **바뀌는 부분과 새로 더하는 부분만** 적는다.
> 수치 라벨: [유도] · [제안] · [결정] · [미정](사용자 결정 또는 G0 PREREG 몫).

## 0. 왜 이 두 실험인가

비용 모델이 goodput으로 이어지는 사슬(대조표 §1) 중 ③ "모델이 고른 설정이 oracle에 가까운가"와 ⑤의 전제 "동적으로 바꿀 여지가 있는가"를 닫는다.

- **V-O(oracle 격차)**: rev2 §6의 V는 예측 λ\*가 큰 **상위 2개 D**만 돌려 "예측이 놓친 세 번째 D가 더 좋았을 가능성"을 caveat로 남겼다(rev2 §6-2, 등록 caveat 3). 그래서 V를 통과해도 "비용 모델이 최적 설정을 고른다"는 말할 수 없었다. V-O는 D 격자 전체를 돌려 oracle을 실측하고, 예측 선택의 **후회(regret)**를 잰다. 논문 방향 ②(비용 모델 기반 static 설정)의 핵심 증거다.
- **H-D(동적 여지)**: 정본 HE0은 기존 동적 제어기가 best static을 못 넘었다고 확정했고, 그 범위를 "위치선정의 증거이지 동적 SM 공급의 천장의 증거가 아니다"로 줄였다. 천장 자체를 static arm만으로 잴 수 있다 — 부하 구간(phase)마다 가장 좋은 static을 사후에 골라 이어 붙인 값(phase oracle)이 전체 best static보다 낫지 않으면, 어떤 동적 정책(HyMux 포함)도 이 기판·이 부하에서 이길 여지가 없다.

## 1. V-O — oracle 격차

### 1-1. rev2 §6에서 바뀌는 것
| 항목 | rev2 §6 | V-O 초안 |
|---|---|---|
| PD-mux arm | 예측 λ\* 상위 2 D (D ∈ {24, 34, 44}) | **D 격자 전체** {16, 24, 34, 44, 64} [제안] — T1 격자와 같은 D(실현 SM 기준, §1-8) |
| fused arm | `fused_tuned`(G0 튜닝 argmax) | 같음 + **`fused_default`**(SGLang 기본, 미조율 대조) [제안] |
| 판정 | λ\*\_pred 대 λ\*\_meas(오차 e) | e 판정(rev2 §6-3 승계) **+ 후회 판정**(아래 1-3) |
| 귀무 | N0(G=1), N1(T1 단독) | N0·N1 + **N2 = 정본 기본 규칙**(아래 1-2) |
| 셀 | arXiv-Sum 자연 출력, shape C(강제 출력 4,096) | 같음 [결정 5·12] — shape C 출력 {1k, 4k} 중 어느 쪽인지 [미정, G0 PREREG] |

### 1-2. 귀무 N2 — "비용 모델 없이도 고를 수 있었나"
정본 실무 기본값은 "peak decode 부하 기준 decode-heavy static 고정"이다. 비용 모델이 쓸모 있으려면 이 규칙보다 후회가 작아야 한다.
- **N2 선택 규칙 [제안, 데이터 전 봉인]**: 셀마다 D ∈ 격자 중 T1 계수 없이 정할 수 있는 규칙 — "HE0 최적 static(d44)을 그대로 쓴다". 대안 N2′ = "가장 decode-heavy한 D(64)". 둘 다 봉인하고 둘 다 보고한다.
- 비용 모델 선택 `D̂` = `cm_predict`의 λ\*\_pred argmax(rev2 §6-1 예측기, D 격자 전체로 확장).

### 1-3. 후회 판정 [제안 — 문턱은 감사 대상]
- oracle `D*` = 셀마다 실측 λ\*\_meas argmax(fused 두 arm 포함한 전체 arm 중). 동률(차 < 3%, 게이트 3)이면 그 집합 전체를 oracle 집합으로 둔다.
- 후회 `R(x) = ln(λ*_meas(D*) / λ*_meas(x))`, x ∈ {D̂, N2, N2′}.

| 라벨 | 조건(위에서 첫 만족) |
|---|---|
| `ORACLE_UNRESOLVED` | 어느 arm의 λ\*가 해결 안 됨(rev2 §6-3 P1 실패) 또는 oracle 집합이 해상도(R9-6) 안에서 갈리지 않음 |
| `CM_HITS_ORACLE` | `D̂` ∈ oracle 집합 |
| `CM_NEAR_ORACLE` | `R(D̂) ≤ ln 1.05` |
| `CM_BEATS_DEFAULT` | `R(D̂) < R(N2)` ∧ `R(D̂) < R(N2′)`, 차가 rep SD를 넘음 |
| `CM_NO_BETTER_THAN_DEFAULT` | 위 어느 것도 아님 |

- 허용 문장 틀: "이 Job의 (셀)에서 비용 모델이 고른 static D는 실측 oracle 대비 후회 …%였고, 정본 기본 규칙(d44)의 후회는 …%였다." 금지: "비용 모델이 최적 설정을 찾는다"(두 셀·한 모델 범위 밖), "hybrid 고유 항 덕분"(V-O는 귀속을 판정하지 않음 — 귀속은 소거 변형 v1·v2, rev2 §6-1).
- **소거 변형의 후회(기술값)**: rev2 §6-1 소거 변형 v1(ctx 항 0)·v2(KV 전용 C\_mem, s = 0)·v3(ρ = 1)로도 `D̂_v`를 뽑아 그 후회를 병기한다. 원고 §7.5 "hybrid 비용 항 제거 시 goodput 0.94"의 정적 대응물이다(동적 정책 없이 "state 항이 설정 선택을 바꾸는가"만 묻는다).

### 1-4. 비용 [유도, Job 크기 결정용 — 측정 아님]
rev2 V(3 arm/셀) = J-V1 3.8 h + J-V2a 3.7 + J-V2b 3.7 ≈ 11.2 GPU-h(rev2 §7). arm이 셀당 3 → 7(PD-mux 5 + fused 2)이면 arm 비례로 약 2.3배 → **약 26 GPU-h**. Job당 ≤ 6 h 원칙(사용자 메모 "GPU Job 적정 크기")으로 셀·pass별 6 Job 안팎. 줄이는 선택지 [미정]:
- (a) 자연 셀만 D 격자 전체, shape C는 상위 3 D + fused 2(약 −30%)
- (b) 1차로 사다리 3 rung(j = −1, 0, +1)만, oracle 후보 2개에만 5 rung(2단계 적응, 단계 규칙은 데이터 전 봉인)
- (c) `fused_default` 제외(−1 arm, G의 "미조율 분모" 공시만)

## 2. H-D — 동적 여지(static arm만)

### 2-1. 무엇을 재는가
- trace: phase 교대 — **W1**(짧은 대화, ShareGPT 계열, HE0 trace의 저·고 rate 구간)과 **W3**(긴 입력, arXiv-Sum 또는 shape C) 블록을 번갈아 둔다. phase 길이 2종 [제안: 120 s, 600 s] — 짧은 쪽은 큐가 phase 안에서 수렴하지 못하는 경우의 대조.
- arm: V-O와 **같은 static arm 집합**(D 격자 5 + fused\_tuned). 동적 arm은 없다.
- 지표: phase별 goodput(그 phase에 **도착한** 요청 기준, 정본 술어: TTFT ≤ SLO ∧ 요청 내부 ITL p95 ≤ SLO), 전체 goodput. 게이트 5에 따라 TTFT/ITL p50/p95/p99 병기.

### 2-2. 판정 [제안]
- `G_best` = 전체 goodput이 가장 큰 단일 static arm의 값.
- `G_phase-oracle` = phase마다 그 phase goodput이 가장 큰 arm을 골라 phase 길이로 가중 합산한 값(사후 oracle, 전환 비용 0·phase 간 이월 무시 → **동적 정책 이득의 낙관적 상한**).
- 여지 `H = ln(G_phase-oracle / G_best)`.

| 라벨 | 조건 |
|---|---|
| `DYNAMIC_HEADROOM_ABSENT` | `H` ≤ rep SD(n ≥ 4) 또는 `H < ln 1.03`(게이트 3) |
| `DYNAMIC_HEADROOM_PRESENT` | `H ≥ ln 1.03` ∧ rep SD 밖, **두 phase 길이 모두**에서 |
| `DYNAMIC_HEADROOM_LENGTH_DEPENDENT` | 한 phase 길이에서만 PRESENT |

- 결과의 쓰임(결정 31): `ABSENT`면 HyMux 구현을 보류하고 논문은 static 설정(방향 ②)으로 간다. `PRESENT`일 때만 HyMux X1–X10(검증 계획 문서) PREREG를 쓴다. 이때 HyMux의 목표치는 `H`(상한)와 실측 동적 이득의 비로 보고한다.
- 금지: `H`를 "동적 정책이 얻을 이득"으로 인용(상한이다), `ABSENT`를 "동적 제어는 원리적으로 불가"로 확장(이 기판·이 trace·이 arm 집합 범위).

### 2-3. 함정(정본에서 결론을 뒤집은 적 있는 것)
- phase 경계 이월: 앞 phase의 큐·running batch가 다음 phase 요청을 오염시킨다. phase 첫 구간(warm 창 [제안: phase 길이의 10%])을 phase goodput에서 제외하고, 제외 전후 둘 다 보고.
- metric cliff(게이트 6): phase rate는 용량 측정 뒤 TTFT ≈ SLO 경계를 피해 정한다.
- duration 합산(게이트 7): 여러 라운드·phase 합산 시 duration은 합산.
- LO phase 절단: HE0처럼 저부하 phase는 도착률이 goodput을 묶어 arm 간 차가 0이 된다(정본 HE0 범위 축소 근거). 저부하 phase만으로 `ABSENT`를 판정하지 않는다 — 판정은 W3·고부하 phase를 포함한 trace에서만.

### 2-4. 비용 [유도]
static arm 6 × phase 길이 2 × seed 4, trace당 약 20–30분이면 약 8–12 GPU-h. V-O와 arm·config·boot를 공유하므로 **같은 Job에 블록으로 넣으면** 부팅 비용을 아낀다 [제안 — 같은 비교의 arm은 한 Job 안, 운영 규약].

## 3. 순서와 선행 조건
1. 비용 항 Job(J-A1·J-A2·J-B·J-C·J-H) 봉인·제출 — 진행 중 경로(smoke → 하네스층 감사 → 봉인).
2. **V-O 구조 봉인 — 항 Job fetch 전** [권고]. 예측기 코드·D 격자·후회 판정·N2 규칙·소거 변형을 먼저 봉인하고, 계수에서 나온 예측값만 fetch 뒤 `V_REG` addendum으로 봉인(rev2 §6-1 "봉인 시점" 승계). 결정 17의 "데이터 뒤 구조 선택" 위험을 없앤다.
3. G0 PREREG(fused\_tuned 튜닝·shape C 출력 길이·workload 정의) — V-O·H-D의 fused arm과 셀이 여기에 의존한다(결정 11·15).
4. 항 Job 결과 분석(rev3.6 §1-9 A-1…A-4 포함) → `V_REG` → V-O·H-D Job 제출 승인.
5. H-D 결과로 HyMux 구현 여부 결정.

## 4. 열린 결정 [미정]
- O-V1 V-O 비용 축소안 (a)/(b)/(c) 중 무엇을 쓸지(1-4).
- O-V2 shape C 출력 길이 1k/4k(G0 PREREG와 함께).
- O-V3 N2 규칙: d44 고정 / 가장 decode-heavy D / 둘 다(초안은 둘 다).
- O-H1 H-D phase 길이 {120 s, 600 s}와 W1·W3 rate(용량 측정 뒤).
- O-H2 H-D를 V-O Job에 블록으로 합칠지(초안 권고: 합침).
- 감사: 이 초안을 PREREG로 옮길 때 claims-auditor 규칙층 감사(최대 2회).
