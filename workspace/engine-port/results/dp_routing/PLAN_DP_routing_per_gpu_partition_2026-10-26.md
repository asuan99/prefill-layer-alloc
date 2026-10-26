# 설계 초안 rev0 — 복제본별 SM 분할 + 길이 기반 라우팅 DP (2026-10-26)

> **지위**: 설계 초안(PREREG 아님, 미봉인, 감사 전). 측정 0, GPU 0, 성능 판정 0. 사용자 지시("SM 분할은 GPU별로 하면서 DP", "진행", 2026-10-26)와 multi-GPU 증거 검토 `reports/hymux_review/MULTIGPU_EVIDENCE_REVIEW_2026-10-26.md`에서 나왔다.
> 수치 라벨: [코드](저장소·엔진 트리 확인) · [측정](출처 명시) · [유도] · [제안] · [미정](사용자 결정).

## 0. 왜 이 형태인가
- 단순 DP(`--dp-size`, round_robin)는 복제본마다 같은 PD-mux가 독립 GPU에서 돌아 결과가 단일 GPU의 N배 재현이다(설계 감사 F-1 항등식; 등록 M3도 DP를 `NO_TREND_DETECTED` 예측의 하네스 대조로만 둠).
- hybrid는 요청 길이에 따라 구속 자원이 바뀐다(c\* = s/k ≈ 8.8k에서 메모리, 2c\* ≈ 17.5k에서 decode 바이트의 지배 항이 state → KV, analytic-v1 [유도]). 그래서 **어느 요청을 어느 GPU로 보내고, 그 GPU의 분할을 무엇으로 두느냐**가 DP에서만 생기는 hybrid 고유 변수다.
- 각 GPU는 **정적 분할**이고 바뀌는 것은 요청 배치뿐이다 → 정본 "단일 GPU 동적 제어는 best static을 못 넘음"과 충돌하지 않는다. 복제본마다 단일 GPU 비용 모델(비용 항 PREREG T1–T4)이 그대로 적용된다.

## 1. 선행 확인(GPU 0, 2026-10-26)
| 항목 | 결과 |
|---|---|
| `--dp-size`에서 GPU별 분할 | **된다** [코드]: dp rank마다 scheduler 프로세스가 자기 GPU에서 PD-mux 초기화(`sglang_engine_dev_multigpu/python/sglang/srt/multiplex/multiplexing_mixin.py:154`), telemetry `.dp{rank}`. M2g J2 DP2 agnostic·d44가 정확성 게이트 통과 + 분할 단계 노출(`results/multigpu/M2G_PASS_TABLE_J2.json` `g1_exposure_split_steps`) — 정확성이지 성능 아님 |
| `--dp-size`에서 복제본별 다른 분할 | **안 된다** [코드]: `--pdmux-config-path`(`manual_divisions`)와 `PDMUX_R2_FIXED_DSM`이 서버 전체에 하나 |
| 독립 서버 N개(GPU당 1개)의 green context | 각 서버가 `CUDA_VISIBLE_DEVICES=i`로 device 0만 보므로 **N=1 경로와 같은 코드 경로** [코드]. `manual_divisions` 경로는 auto grid의 GPU id 단언(`pdmux_context.py:112-118`)을 타지 않는다. 한 노드 N개 동시 실측은 smoke 필요 |
| 라우터 | 이미지에 `sglang_router` 없음 [코드: `env/venv_packages_2026-09-17.txt`]. 엔진 내장 DP 분배는 `ROUND_ROBIN`·`TOTAL_REQUESTS`·`TOTAL_TOKENS`·`FOLLOW_BOOTSTRAP_ROOM`뿐이고 `--dp-size` 안에서만(`data_parallel_controller.py:70-76`). `trace_loadgen.py`는 단일 endpoint → **클라이언트 측 라우팅으로 확장**(아래 2-2) |
| M3 J2·J4·J8 9.5 h 정지 | **같은 순간 외부 종료**: `completedAt` 21:14:02.986Z(J2)·21:14:03.027Z(J4)·21:14:03.065Z(J8) — 80 ms 안, 시작 시각이 달라 경과 시간 상한이 아님(9h30m55 s·45 s·40 s), 하네스 캡 16 h 미도달. 종료 사유 필드 없음, Job 로그는 preflight 직후에서 끊김. 원인(수동·API 일괄 종료 또는 플랫폼 사건)은 VESSL 쪽 확인 필요. **설계 함의**: 플랫폼 경과 시간 상한의 증거는 아니지만, 원인 미확인이므로 Job ≤ 6 h 원칙 유지 |

## 2. 설계

### 2-1. 실행 형태 [제안]
- **N개 독립 SGLang 서버**(GPU당 1개, 각자 config·env·cap·D), 한 Job·한 노드(`resourcespec-a100x{N}`). 엔진 수정 0, 엔진 digest `99bbef91…` 유지.
- 등록 규칙 승계: M3 PREREG C-7("한 캠페인 안에서 `--dp-size`와 N-독립-서버를 섞지 않는다") → 이 캠페인은 **독립 서버만**, 새 PREREG로 등록. M3 rev12 DP 자료와 결합·비교하지 않는다(엔진 digest `82a5fc5b…` 대 `99bbef91…`, telemetry 비대칭).
- 서버별 기록 = 비용 항 하네스의 단일 GPU boot 기록과 같은 스키마(재사용).

### 2-2. 라우팅 — 클라이언트 측 [제안]
- `trace_loadgen.py`에 endpoint 목록과 라우팅 함수(요청 → 서버 index)를 넣는다. 라우팅 결정은 송신 시점, 서버 상태는 클라이언트가 아는 것(송신·완료 기록)만 쓴다.
- 공시: 클라이언트 측 라우팅은 **오버헤드 0의 이상적 라우터**에 해당한다(별도 프록시 계층 지연·직렬화 없음). 라우팅 결정 시간은 기록한다.
- 클라이언트 프로세스 하나가 N 서버를 몰므로 GIL·이벤트 루프 포화(confound 7)를 `FRONTEND_SATURATED` 계열 라벨로 검사한다(송신 지연 분포, 클라이언트 CPU).

### 2-3. arm (같은 N, 같은 Job 안, 블록 교대) [제안]
| arm | 각 GPU 엔진 | 라우팅 |
|---|---|---|
| R0 `fused_tuned·RR` | tuned fused(mixed-chunk, chunk 격자 사전등록, 같은 규칙으로 N=1에서 선택) | round_robin |
| R1 `fused_tuned·LL` | 같음 | least-load(미완료 토큰 합 최소, 엔진 `TOTAL_TOKENS` 대응) |
| R2 `pdmux_static·LL` | PD-mux 단일 static D(비용 모델 선택 D̂, 전 GPU 동일) | least-load |
| R3 `pdmux_static·LEN` | 같음 | 길이 기반: 입력 + 기대 출력 ≥ 문턱 τ인 요청은 "긴 쪽" 복제본 집합, 나머지는 "짧은 쪽" 집합, 집합 안에서 least-load |
| R4 `pdmux_hetero·LEN` | 복제본 집합별로 다른 static D·cap(비용 모델이 regime별로 선택) | R3과 같음 |
- τ와 집합 크기(긴 쪽 복제본 수)는 **비용 모델로 데이터 전에** 정한다(c\* 기반 후보 τ ∈ {c\*, 2c\*}, 집합 크기는 워크로드 길이 분포의 KV·state 수요 비) — 사후 튜닝 금지(게이트 8). 대안 τ는 민감도 기술값.
- R4의 regime별 D·cap: 비용 항 T1(D·bs·ctx)·T4(C\_mem) 적합에서 각 집합의 기대 (bs, ctx)에 대해 decode floor를 지키는 최소 D와 C\_mem cap — 데이터 전 봉인(V-O와 같은 "구조 먼저 봉인").

### 2-4. 지표 [제안]
- **G\_SLO(N) = λ\*\_SLO(arm) / λ\*\_SLO(R1)**(같은 N·같은 Job), 정본 술어(TTFT ≤ SLO ∧ 요청 내부 ITL p95 ≤ SLO), SLO = 절대 A′(TTFT ≤ k·floor\_{N=1}(L), ITL p95 ≤ 50 ms) — 조밀 사다리 직접 측정(사후 재채점 금지), 목표 attainment 양쪽 rung ≥ 2, 해상도 ≤ 3%.
- 분해: 라우팅 이득 = R3/R2, 분할 이질화 이득 = R4/R3, PD-mux 이득 = R2/R1. N 배율·GPU당 goodput은 기술값만(F-1).
- 게이트 5: arm별 복제본 점유 분포, 복제본별 TTFT/ITL p50/p95/p99, 라우팅 결정 분포(긴 쪽 비율).

### 2-5. 대조·범위 [제안]
- 모델: Nano-9B-v2(주) + 순수 attention 대조 Llama-3.1-8B(같은 N 격자). Transformer에는 regime 전환이 없으므로 R3·R4의 이득이 작거나 없다는 것이 등록 예측 — 차이가 hybrid 귀속의 근거.
- N ∈ {1(앵커, 라우팅 변수 없음), 2, 4}, 가능하면 8. 각 Job에 N=1 앵커.
- 워크로드: 길이 혼합(ShareGPT 계열 짧은 요청 + arXiv-Sum/shape C 긴 요청), 긴 요청 비율 2점 [제안: 30%, 60%]. 변화 trace 여부 [미정](라우팅은 정상 상태에서도 의미가 있으나 게이트 2는 정책 비교에 변화 trace를 요구).
- 반복: seed ≥ 4, boot ≥ 2(ABBA), 짝 CI 0 배제 ∧ ≥ 3%. 전 arm 계측 OFF(결정 22). GPU별 전력·클럭 기록, 노드 전력 공변 공시.

### 2-6. 비용 [유도 — multi-GPU 증거 검토 §4의 손 비례 환산을 옮김, 측정 아님]
config 하나(arm 5 × 혼합 2 × rung 5 × seed 4 ≈ 200셀, 셀 ≈ 175 s) ≈ 9.7 h → Job ≤ 6 h 원칙상 2 Job 이상. 모델 하나 N {2, 4}에서 x2·x4 장비 시간 합으로 약 60–90 GPU-h, 대조 모델 포함 약 120–180 GPU-h. arm·혼합 축소안(예: R1·R2·R3·R4만, 혼합 1점)은 [미정, O-D3]. 정식 값은 하네스 시간 모형 뒤 dry-run.

## 3. 순서·선행 조건
1. 비용 항 Job 봉인·제출·분석(R4의 regime별 D·cap과 τ가 T1·T4 적합에 의존).
2. G0 PREREG(tuned fused 선택 규칙·shape C) — R0·R1 분모 공유.
3. 이 설계를 PREREG로 옮김 → 규칙층 감사(최대 2회) → 하네스(loadgen 다중 endpoint·라우팅 함수·N 서버 기동) → 하네스층 감사 → Workspace smoke(x2: 독립 서버 2개 동시 green context, 라우팅 경로) → 봉인 → 제출 승인.
4. V-O·H-D와 arm·config를 공유할 수 있는지 검토(R2 = V-O의 D̂ arm의 N>1판).

## 4. 열린 결정 [미정]
- O-D1 실행 형태: 독립 서버(권고) / 엔진 패치로 `--dp-size` rank별 config(엔진 digest 변경 → G-α·G-β 재실행).
- O-D2 라우팅 위치: 클라이언트 측(권고, 이상적 라우터 공시) / 별도 프록시.
- O-D3 arm·혼합 축소안과 N 범위(8 포함 여부).
- O-D4 변화 trace 여부.
- O-D5 원고 반영: §7.8을 이 실험 계획으로 교체(측정 전 수치 없음).

## 5. 등록 caveat(초안)
- "이 DP는 GPU당 독립 서버 N개와 클라이언트 측 라우팅이며, 별도 라우터 계층의 비용은 포함하지 않는다."
- "N 배율·GPU당 goodput은 복제 처리율의 재표현이라 PD-mux·라우팅 이득의 근거가 아니다."
- "M3 rev12(`--dp-size`) DP 자료와 결합하지 않는다."
