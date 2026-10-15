# 공통 측정 명세 v0.2 — 단일 GPU·multi-GPU 공용 (2군 Q4–Q12 + 3군 Q13·Q14·Q16 명세분) (2026-10-15)

> **v0.2 — 사용자 결정(2026-10-15: D-C′ (a)·D-6·D-15·D-7) + 감사 2회차 반영판.** 2회차 판정서 `VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md`(이하 V2, 명세 v0.1 NO-GO → 사용자 결정 필요, 死因 D-C′)의 §3 B-9(K7)·§7 승계 caveat K1–K9를 반영했다. v0.1(381줄) sha256 `8c967fa8ffb3b918…`(미커밋).
> **v0.1 — 감사 1회차 수리판.** 1회차 판정서 `VERDICT_spec_v0_and_closure_2026-10-15.md`(이하 V, 명세 NO-GO, 死因 D-A·D-B·D-C)의 §6 명세 수리 R-1–R-16과 §4 지적을 반영했다.
> 수리 전 v0 = 이 파일 293줄판, sha256 `cc582ff867b38e8c…`(미커밋; 커밋 9a2a679의 286줄판과 다름).
> **지위: DRAFT v0.2. 규칙층 감사 2회 완료(상한, §14). 정본이 아니다.** 측정 0, 새 성능 판정 0.
> 정본화 경로는 DISCUSSION_INDEX Q22: 2군 사용자 결정 → 규칙층 감사 → main `promote`. 트랙(G0·M3)은 링크만 건다.
> 출발점: `DISCUSSION_INDEX_2026-10-15.md`(이하 IDX) §1 2군·3군 표, §5. 약어: A = `AGENDA_methodology`, D = `DESIGN_span_and_integration`,
> K = `COSTMODEL_running_cap`, R = `REVIEW_handoff_prompt`, C = `COMPARE_span_sync_muxwise_bullet`, CK = `CHECK_mixed_chunk_semantics`(모두 같은 폴더),
> MG = `../prefill-layer-alloc-multigpu/workspace/engine-port/results/multigpu/PREREG_MULTIGPU_2026-10-12.md`(rev11, 읽기만),
> SLO-R = `reports/slo_definition_rationale_2026-10-04.md`(A′ 본문).
> 수치 라벨: **[측정]**(출처 명시) · **[유도]**(식·코드 대입, 측정 아님) · **[코드]**(소스 사실) · **[결정]**(사용자 결정) · **[제안]**(이 초안이 고른 값, 근거 약함 → 감사 대상).
> 이 문서의 수치는 전부 VESSL A100 SXM 80GB(108 SM) 기판 범위다. KISTI 시기 수치와 섞지 않는다.

**수리 색인 (V §6 → 이 판의 절)**

| 수리 | 내용 | 고친 곳 |
|---|---|---|
| R-1 (死因 D-A) | λ\*\_SLO = seed 합산 attainment 0.90 교차점의 log-rate 선형보간, rung 라벨 기술값, estimand별 e\_j, 공통 격자, 양자화 값 t 금지, N3 rev7 dry-run | §8 R9-1·R9-1b·R9-5, §2 R5-3 |
| R-2 | Holm 가족(k 3개), J별 임계표, 1차 estimand 지정 | §2 R5-3·R5-3b, §6 R8-3 |
| R-3 (死因 D-B) | 오류 서버/클라이언트 분류, 짝 무효 + `UNRESOLVED`, 건너뛰기 금지, client timeout 하한 | §4 R7-3 |
| R-4 (死因 D-C) | G\_mixed 분모 = 합집합 9 config 최대, 동률 규칙, gate 불통과 대체 headline | §6 R8-2d·R8-3·R8-5 (v0.2에서 11 config·D-16으로 대체) |
| R-5 | "((ii) 대조 전)" 삭제, "chunked prefill 대비" 문안 교체 | §6 R8-3 표·R8-5 |
| R-6 | 혼합 실현 지표 정의 | §6 R8-4 |
| R-7 | 튜닝 arm 재채점 = 재스코어 | §4 R7-2 |
| R-8 | `CLIFF_ADJACENT` 이항 SE 정의, 부호 인용 금지 조건 | §8 R9-2 |
| R-9 | 관측자 TOST ±1.5%, 짝 n ≥ 8 계열별, 계측 최소 수준 정의 | §9 gate (b)·관측자 블록 |
| R-10 | R13-3 전제 정정, D-6 재결정 대기, P4 등록 | §11 R13-3·R13-4, §13 D-6 (v0.2에서 D-6 결정) |
| R-11 | R10-2 같은 Job 앵커 floor 판정 | §5 R10-2 |
| R-12 | `CAP_BINDING` 시간 비율·λ\*\_SLO 셀·LO, `CAP_BINDING_UNRESOLVED` | §12 R16-2 |
| R-13 | R5-3 근거 문장 정정(Samuelson 함축) | §2 근거 |
| R-14 | 셀당 요청 수·이항 해상도 등록, 해상도 > 3%면 크기 문장 금지 | §8 R9-6 |
| R-15 | G\_def 분해 호칭 "경로 분해(회계 항등식)" | §6 R8-3·비판 1 |
| R-16 | MG rev11 M3의 R11-2 미충족 범위 문장 | §7 multi-GPU 차이 |
| V §4 | Job 안 ABBA, duration·attainment 합산 등록, G\_mixed 꼬리표 3개 추가, A-2·A-3·A-5·A-13 조건 | §2 R5-1·R5-6, §6 R8-2c, §3 R6-4·R6-5, §7 R11-2, §8 multi-GPU |
| S-1–S-8 | 승계 caveat 문자 그대로 | 각 규칙 옆(S-1 §6 R8-2c, S-2 §6 R8-2, S-3·S-8 §2, S-4 §6 R8-3, S-5 §7 R11-3, S-6 §11 R13-4, S-7 §3 R6-5) |
| **v0.2** D-C′ → D-16 (a) | 동률 띠·λ\*\_thr 동률 기준 삭제, 엄격 argmax λ\*\_SLO + 정확 동률 시 결정적 순서 | §6 R8-2d, §13 D-16 |
| **v0.2** D-15 추가 | mixed-chunk 예산 5점 {512…8192}, 합집합 11 config, 작은 예산 prefill 몫 공시, S-2 주석 | §6 R8-2·R8-2b·R8-2c·R8-2d·R8-3·R8-5, §13 D-3·D-15 |
| **v0.2** D-6 확정 | span 예산 격자 {8k, 16k, 32k, 65k}, G0-S 셀 4/3배, Job 시간 예산은 G0 PREREG 몫 | §11 R13-2·R13-3, §13 D-6 |
| **v0.2** D-7 축소 확인 | 1% 문턱은 클라이언트 측 오류에만(사용자 확인 2026-10-15) | §4 R7-3, §13 D-7 |
| **v0.2** V2 B-9 (K7) | R7-2를 게이트 8에 맞게 되돌림: SLO를 읽는 정책·튜닝 arm 재채점은 기술값으로도 싣지 않음 | §4 R7-2 |
| **v0.2** V2 §7 K1–K9 | 승계 caveat 문자 그대로(K1 §8 R9-2, K2 R9-1, K3 R9-6, K4 §9 gate (b), K5 §2 R5-3b, K6 §4 R7-3, K7 R7-2, K8 §11 R13-4, K9 §6 R8-4) | 각 규칙 옆 |
| **v0.2** V2 B-7 (K8) | P4 "≈ 0" 동등성 여유 = G0 PREREG 등록 필수 [미등록] | §11 R13-4, §15 |

## 0. 범위·전제·상태

**범위**: goodput·용량·SLO·arm·반복(n)·검정·기판 공변·계측을 정의한다. G0(단일 GPU, main)와 M3/M4(multi-GPU) PREREG가 이 명세를 인용한다.
rate 사다리 값·seed 목록·Job 하드캡·셀 수는 각 PREREG의 몫이다(이 명세는 그 값이 지켜야 할 조건만 정한다).

**확정 전제** [결정]: F2 주 모델 = Nemotron-Nano-9B-v2(multi-GPU는 `-Base` revision `dc0661c…`, MG §1) · F3 true dual worker = AMD 전용 ·
F4 lockstep PD-mux가 multi-GPU로 이양, 단일 GPU 결과의 범위는 "lockstep 기판" · F5 A′ · F6 multi-GPU arm 4종·multi-GPU 1차 estimand G\_sym · F7 TP cap 48·N · F8 radix OFF ·
Q2 포지션 = P-C(혼합) · Q3 G0·G2 표준 셀 = arXiv-Sum 16k [결정 2026-10-15, §13 머리].

**미결 전제(바뀌면 표시한 절을 다시 연다)**:

| 전제 | 가정값 | 바뀌면 영향받는 절 |
|---|---|---|
| Q3 5-a | Mooncake Conv + OpenThoughts + ShareGPT | §1 실 trace rate 정의, §3 128k floor, §4 짧은 출력 규약 |
| Q1 N2 종료 | (a) 종료 + F-2를 §9 I-1로 흡수 | §9 I-1 선행 일정, §7 공변 검사의 크기 분리 범위 |
| (해소) D-6·D-15 | 2026-10-15 결정됨(§13): 격자 {8k, 16k, 32k, 65k}, 혼합 예산 5점 | — |

**공통 기판 규칙** (모든 절에 앞선다):
- 한 비교의 arm은 한 Job 안에 두고, **같은 Job 안 공통 rate 격자**(절대 req/s, 데이터 보기 전 등록)에서 잰다(운영 모델 · #262 · R-1). 다른 Job의 arm 값은 비교하지 않는다.
- 기판 gate는 MG §16-3과 같다: GPU 이름 정확히 "NVIDIA A100-SXM4-80GB", power limit 400 W, 기대 GPU 수 일치, 어긋나면 `ABORT_SUBSTRATE`.
- 엔진 트리 digest를 셀마다 기록한다. [2026-10-15 갱신] `multigpu-scaling`은 main `dc06c870`에 merge됐다(조건부 가, 감사 2회차 §6). merge 트리의 ENGINE_DIGEST는 `82a5fc5b…`(multigpu 세션 재대조)이고 merge 이후 main 쪽 엔진 src 변경은 0이다 [측정, 트랙 기록]. G0와 M3 N=1 앵커는 **같은 ENGINE_DIGEST일 때만** 같은 바이너리로 본다. 그래도 Job이 다르면 수준 비교는 금지다(운영 모델·#262). merge 전 바이너리로 잰 봉인 캠페인(N2/N3/E2) 수치와 비교 금지(N2 금지 41, N3 #262).
- **측정 Job 로그 knob(2026-10-15 추가, multigpu 이양 요청 ②)**: merge 트리의 `job_entry.sh`는 기본값으로 로그 tee 미러(`PDMUX_LOG_MIRROR=1`)와 300 s heartbeat(`PDMUX_HEARTBEAT_S=300`, 주기적 `find`)를 측정 중에 띄운다 [코드, `job_entry.sh:51-52, 299`]. 측정 Job은 **`PDMUX_HEARTBEAT_S=0`**으로 고정하고, `PDMUX_LOG_MIRROR` 값은 `meta/log_mirror.txt`에 기록·공시한다 [결정 대기 아님 — 관측자 상주 프로세스를 줄이는 보수 방향, PREREG마다 값 등록]. bring-up·프로브 Job은 기본값 허용.
- 고정 노브: cudagraph ON, radix OFF, `mem_fraction_static 0.82`, `max_running_requests` = cap, **`max_mamba_cache_size` = cap 명시**(§1-23), attention backend `flashinfer`, `mamba_ssm_dtype` 명시 고정(R §3.1: dtype이 B\_max 순위를 바꾼다).

---

## 1. Q4 — 게이트 2의 적용 범위

**규칙 R4-1**: 게이트 2(변화 trace 필수)는 **동적 정책 비교**에만 적용한다. 정적 시스템 비교(fused 계열 vs PD-mux static arm, static D 간 비교)는
고정-rate Poisson 사다리를 쓸 수 있다. 단 결과 문장에 "정적 시스템 비교, 정상상태 고정 rate"를 범위로 적고, 그 결과를 동적 정책의 근거로 쓰지 않는다.
**R4-2**: 정상성 검사는 N3 S1–S3(큐 성장 포함)를 적용하고, 비정상 셀은 용량 판정에서 뺀다.
**R4-3**: timestamp가 있는 실 trace(5-a Mooncake)의 rate 축 = **시간축 압축 배율 c**(도착 간격 ÷ c, 순서·길이 보존). 이때 S1–S3를 적용하지 않고,
판정 규칙은 5-a PREREG에서 따로 등록한다(이 명세는 "c 사다리 + attainment-vs-c 곡선"까지만 정한다).

- 근거: A §1 표("정책 비교 아님 → 해당 없음" 관행, `PREREG_N2_3A:170` 등), A §5-1, R §4.1 게이트 2.
- 대안·기각: (b) G0에도 변화 trace 추가 → 정적 arm 두 개 비교에서 변화 trace는 두 정상상태의 혼합이라 해석이 더 어렵고, 비용이 Job 수에 곱해진다. 기각(5-a·M4가 맡음).
- multi-GPU 차이: M3는 고정 rate HI/LO로 이미 부합한다. M4의 변화 trace(ρ\_LO 0.30 ↔ ρ\_HI 0.85 × λ\*\_fused, 3라운드, MG §5)는 게이트 2 형식을 그대로 따른다.
- 예상 비판 1: "실서비스는 bursty하다." → 대응: 범위 문장 의무화 + 5-a(실 trace)·M4(변화 trace)가 따로 다룬다. G0 결과는 운영 주장으로 쓰지 않는다.
- 예상 비판 2: "게이트 2는 static조차 ±1.3 흔들려서 생긴 규칙인데?" → 대응: 분산 문제는 §2(Job ≥3 반복, Job 수준 판정)로 처리한다. 고정 rate 허용은 분산 규칙과 묶여 있을 때만 유효하다.

## 2. Q5 — n의 층위와 판정 검정

**규칙 R5-1 (설계)**: 판정 셀은 **seed ≥4 × Job ≥3**. Job 안에서 seed마다 boot를 새로 한다(seed = boot, L2·L3 합쳐짐을 공시).
arm 순서는 **Job 안 ABBA 균형**으로 둔다: 등록 순열 π(Job마다 등록 seed로 무작위화)를 seed 블록 1–4에 π, π^R, π^R, π 순서로 쓴다(seed > 4면 이 4블록을 반복) [제안]. 블록 무작위화만으로는 J=3에서 순서·열 효과 균형이 약하다(V §4).
**R5-2 (기록)**: Job마다 hostname, GPU UUID, 드라이버, `cpu.max`, 노드 종류를 기록한다. 두 Job이 같은 GPU UUID면 독립 Job으로 세지 않는다(`JOB_NOT_INDEPENDENT`).
**R5-3 (판정, R-1·R-2)**: Job j의 효과 e\_j는 **estimand별로** 정의한다(log 비, 두 arm은 같은 Job·같은 공통 격자):
- λ\*\_SLO 기반 G: e\_j = log(λ\*\_A,j / λ\*\_B,j), λ\*는 R9-1의 **Job별 보간값**(seed 합산 attainment에서 Job당 한 값). "seed 평균 e\_j"는 쓰지 않는다.
- λ\*\_thr 기반 G: e\_j = seed별 log(achieved\_A,s / achieved\_B,s)의 seed 평균(R9-1의 HI achieved).
- **rung 라벨(첫 실패 rung 바로 아래)과 그 밖의 양자화 값으로는 e\_j·t를 계산하지 않는다.** 라벨은 기술값.
- 판정: Job 수준 t(자유도 J\_eff − 1)를 아래 R5-3b Holm 임계표로 판정한다. J\_eff = e\_j가 `UNRESOLVED`가 아닌 Job 수. J\_eff < 3이면 `UNRESOLVED_J`(판정 불가). 크기 문장은 추가로 |평균 효과| ≥ 3%(게이트 3)와 R9-6 해상도 조건.
- 부호 일치(모든 e\_j 같은 부호)는 공시만 한다. **S-3** "R5-3의 부호 일치 조건은 J ≤ 4에서 t-CI 조건에 수학적으로 포함된다. 독립 증거로 세지 않는다."
**R5-3b (Holm 가족·임계표, R-2)**: 한 가족 = 한 estimand의 k ∈ {3, 5, 10} 세 가설. Holm step-down: |t|가 큰 순서로 1번째는 α/3, 2번째는 α/2, 3번째는 α(양측, α = 0.05)와 비교하고 처음 기각 실패에서 멈춘다.

| J\_eff (자유도) | 1번째 (α/3) | 2번째 (α/2) | 3번째 (α) |
|---|---|---|---|
| 3 (2) | 7.65 | 6.21 | 4.30 |
| 4 (3) | 4.86 | 4.18 | 3.18 |

[유도, `scipy.stats.t.ppf(1 − a/2, J−1)`]. J\_eff ≥ 5는 같은 식으로 G0 PREREG에 표를 등록한다.
**K5** "R5-3b는 \|t\| 순서 step-down이다. J_eff가 k마다 다르면 p 순서 Holm과 라벨이 다를 수 있다."
- **1차 가족(확증)** = G\_mixed(합집합판, §6 R8-3). fused\_mixed gate (i) 불통과 시 같은 자리를 G\_tuned(비혼합 6 config)가 잇는다(R8-2d, 데이터 보기 전 결정되는 전환).
- **2차 가족** = G\_sym(k = 3, 5, 10), 별도 Holm 가족. 1차 판정에 합산하지 않는다.
- 그 밖(S, G\_def, 혼합·비혼합 가족별 값, λ\*\_thr 기반 G, rung 라벨)은 기술값이며 판정 문장에 쓰지 않는다.
**R5-4 (범위)**: Job이 1개뿐인 결과는 "이 Job에서"로만 쓴다. "VESSL A100에서"는 R5-3을 통과한 것만.
**R5-5 (공시)**: Job 안 seed 짝 bootstrap은 기술값이며, n=4면 서로 다른 재표집이 35개뿐임을 적는다(N3VR-11).
**R5-6 (합산 방식, 게이트 7)**: seed·라운드를 합칠 때 duration은 **합산**한다(처리율 = Σ완료 / Σduration, `max()` 금지). attainment 합산은 **합의 비**(Σ합격 / Σ요청)이고, seed별 비의 평균은 기술값으로만 병기한다.
**S-8** "J=3의 '판정 불가'는 '효과 없음'이 아니다."

- 근거(R-13 정정): 반복 Job이 필요한 이유는 **Job × arm 상호작용**(미측정)이다. Job 간 수준 이동(+4% [측정, E2 r1/r2, E2VR-3], 공통 격자 36/36 일방향 Job 효과 [측정, #262])은 같은 Job 안 짝 차분에서 상쇄되므로 그 자체로는 반복의 근거가 아니다.
  3% 효과가 J=3 무보정 임계 4.30을 넘으려면 상호작용 σ ≲ 3%·√3/4.30 ≈ 1.2%가 필요하다 [유도, 근사]. 상호작용 σ는 아직 측정되지 않았다.
- 결정 규칙의 성질 [유도]: t > J−1이면 모든 e\_j가 같은 부호다(Samuelson 부등식). J ≤ 4에서 임계 t가 모두 J−1보다 크므로 (ii)가 (i)을 함축한다 → 부호 일치는 판정 조건에서 빼고 공시만 한다(교훈 9).
- 대안·기각: (a) 현행(Job 1, 4/4 부호) → Job 효과와 구분 불가. (c) seed 6개 단일 Job → Job 층 변동을 전혀 안 잡는다. 둘 다 기각.
- multi-GPU 차이: M3는 config마다 Job 1개(J\_N)이고 seed가 같은 boot 안이다(MGP-7). 그래서 M3의 G\_sym 판정은 **Job 안 판정**이며, 이 명세 기준으로는 R5-4의 "이 Job에서" 범위다.
  M3의 N=1 앵커 + 3% Job 효과 검사(MG §15-3, §16-1)는 Job 효과의 **검출**이지 반복이 아니다. multi-GPU 1차 estimand는 F6대로 G\_sym이다(단일 GPU 1차와 다름을 결과에 적는다).
- 예상 비판 1: "Job 3개, 자유도 2의 t는 검정력이 거의 없다." → 대응: 맞다(감수 비용, D-2). Holm 1번째 임계 7.65 [유도]는 더 엄격하다. 검정력 부족은 S-8대로 쓴다.
- 예상 비판 2: "VESSL이 같은 노드에 다시 배치하면 Job이 독립이 아니다." → 대응: R5-2의 UUID 규칙. 같은 노드·다른 GPU는 독립으로 세되 노드를 공변량으로 공시한다.

## 3. Q6 — A′의 혼합 길이 적용

**규칙 R6-1**: 요청별 TTFT 임계 T\_i = k·floor(L\_i). 결정점 k ∈ {3, 5, 10}(R5-3b Holm 가족), k ∈ [1, 25] 곡선은 기술값. **곱셈형 유지**. T는 처치가 아니라 판독 눈금이다(SLO-R §3–4).
**R6-2 (floor 측정)**: 각 Job의 **첫 블록**에서, 등록 기준 config로 무경합 단일 요청 TTFT를 입력 길이 {1k, 2k, 4k, 8k, 12k, 16k} [제안] 각 16회 재고 중앙값을 낸다. 값은 초 단위로 동결한 뒤 같은 Job 셀을 채점한다.
**R6-3 (보간)**: 측정점 사이는 log–log 구간 선형 보간. 측정 범위 밖 외삽 금지. L < 1k 요청은 floor(1k)로 고정하고 그 비율을 공시한다
(arXiv-Sum Nano 토크나이저 유지분 중 L ≤ 2048은 5.0% [측정, ARXSUM_STATS], 1k 미만은 그보다 적다).
**R6-4 (기준 config)**: floor는 구성 의존이므로(SLO-R §5) **arm 하나의 config로 고정**해 모든 arm에 같은 T를 쓴다. 기준 = fused\_sym(D-1 [결정]). 다른 arm의 floor는 기술값으로 같이 잰다.
조건(V A-2): (a) arm별 자기 floor ÷ 기준 floor 비를 길이점마다 병기한다. (b) PD-mux 무경합 단일 요청이 (0,108)로 도는지와 split span 실현을 W-idx·I-3으로 확인한다(`multiplexing_mixin.py:453` [코드]).
(c) floor 차를 분해하지 않고 "작은 k에서 PD-mux가 짐 → 경합 탓"이라 쓰지 않는다.
**R6-5 (병기)**: (가) 길이 3분위별 attainment, (나) 가산형 대조 T′\_i = floor(L\_i) + (k−1)·floor(L̄) 채점을 **기술값**으로 병기한다. L̄ = 그 셀 표본 요청 입력 길이(토큰)의 산술평균. 판정은 곱셈형만.
**S-7** "가산형 T′와 곱셈형 T의 부호 갈림은 기술값이며 판정이나 결과 문장이 아니다."

- 근거: A §3.2–3.3, SLO-R §3-6. N3 shape B floor 0.907 s → T 2.72/4.53/9.07 s [측정, A §3.2].
- 대안·기각: 고정 절대 SLO(4–15 s) → 긴 요청에 체계적으로 불리(R §4.1). 정규화 TTFT(ms/token) → 짧은 요청을 과벌. 가산형 판정 → 자유 표면이 늘어남. 셋 다 판정용으로 기각, 절대 초 값은 환산 좌표로만 병기.
- multi-GPU 차이: §5(Q10). shape A/B는 단일 길이라 floor 1점이면 된다. arXiv-Sum shape를 M3에 넣으면(Q23) R6-2를 N=1 앵커 블록에서 잰다.
- 5-a 차이: Mooncake 꼬리(p99 85k, 최대 126k)는 측정점 {32k, 64k, 128k}이 추가로 필요하다. 128k 단일 요청 프로브(main inbox 결정)와 합친다.
- 예상 비판 1: "곱셈형은 긴 요청에 절대 여유를 더 줘서 PD-mux의 장문 HOL을 숨긴다." → 대응: R6-5 (가)(나)를 기술값으로 병기해 실패가 어느 길이 구간에서 오는지 보인다(S-7).
- 예상 비판 2: "Job마다 floor가 달라 T가 Job마다 다르다." → 대응: 판정은 Job 안 arm 차분이라 같은 Job 안에서는 T가 같다. Job 간 floor 산포는 Job 효과 진단값으로 공시한다.

## 4. Q7 — ITL 술어 세부

**규칙 R7-1**: 주 술어 = 요청 내부 token-ITL(첫 토큰 제외)의 P95 ≤ 50 ms. P95 계산법을 등록한다(`numpy.percentile`, method `linear`) [제안].
mean-ITL 술어는 secondary이고 **같은 문장·표 칸에 섞지 않는다**(Granite 부호 반전 선례, R §4.1 게이트 4).
**R7-2 (재채점 범위, R-7 → v0.2 V2 B-9로 되돌림)**: SLO를 읽지 않고 정해진 stock static arm(`fused`·`fused_sym`·`agnostic` stock 임계표·`d44` fixed)만 ITL 사다리 {40, 50, 60, 65} ms·k 곡선 재채점을 기술값으로 병기한다(SLO-R §3-5).
**SLO를 입력으로 읽는 정책 arm과 튜닝 arm(`fused_tuned`·`fused_mixed_tuned`·`d_tuned`, A′·ITL 50 ms로 고른 config)의 ITL 사다리·k 곡선 재채점은 재스코어(게이트 8)이므로 기술값으로도 싣지 않는다.** 다른 SLO에서의 값이 필요하면 그 SLO로 재튜닝해 직접 잰다. 정본(게이트 8)이 이 명세보다 우선한다.
**K7** "SLO를 입력으로 읽는 정책 arm의 ITL 사다리·k 곡선 재채점은 기술값으로도 싣지 않는다(게이트 8)."
**R7-3 (요청 처리, R-3)**: 출력 길이는 데이터셋 값으로 고정(`ignore_eos`), 출력 < 2토큰 요청은 표본 추출 단계에서 제외하고 건수 공시. 오류는 원인 쪽으로 나눈다:
- **서버 측**(서버가 요청을 수락한 뒤의 timeout, HTTP 5xx, abort) = attainment **불합격**. 셀 무효 사유가 아니다.
- **클라이언트 측**(송신 전 연결 실패, loadgen 예외, 예정 발송 시각 대비 발송 지연 > 0.5 s [제안]) 비율이 한 rung에서 어느 arm이든 > 1%면, 그 rung의 **비교 두 arm 짝을 함께 무효**로 하고 그 Job의 해당 λ\* = `UNRESOLVED`(R5-3 J\_eff에서 빠짐).
  무효 rung을 **건너뛰고 다음 rung으로 보간·첫 실패를 정하지 않는다**. 1% 문턱은 클라이언트 측 오류에만 적용한다 [결정, D-7 범위 축소 — 사용자 확인 2026-10-15].
**K6** "R7-3의 '해당 λ\*'는 교차 rung 위쪽의 무효 rung에도 적용된다(보수적으로 읽는다)."
- client timeout ≥ 25·floor(16k)(R6-1 곡선 상한 k = 25) [유도 규칙]. 이보다 짧은 timeout은 등록 불가.
- 분류 기록: 요청마다 (수락 여부, 오류 유형, 예정·실제 발송 시각)을 남긴다. 분류가 불가능한 오류는 클라이언트 측으로 센다(보수적 무효화).
**R7-4 (표본 수 공시)**: 셀마다 요청당 ITL 표본 수 n\_ITL의 p5/p50을 적는다. 워크로드 n\_ITL 중앙값 < 20이면 "P95 ≈ 최댓값" 경고를 결과에 붙인다.
**R7-5 (구속 술어)**: 셀마다 실패를 {TTFT만, ITL만, 둘 다}로 분해해 적는다(SLO-R §5, MG §4-6 실패 분해와 같은 형식).

- 근거: A §3.1(`analyze.py:34-38`: ITL 빈 요청은 NaN → 불합격), A §3.3-2·3, SLO-R §3-5. V A-4·D-B(셀 무효가 arm 한쪽에만 걸리면 λ\* 사다리가 fail-open).
- 표본 수 [유도]: arXiv-Sum 출력 mean 203·p50 195 [측정, ARXSUM_STATS] → n\_ITL ≈ 194. Mooncake toolagent 출력 p50 30 → n\_ITL ≈ 29, LooGLE 출력 ~15 → ~14(A §3.3).
- 대안·기각: (a) "오류 > 0이면 셀 무효" → 표본이 선택적으로 빠진다. (b) "모든 오류 = 불합격만" → 하네스 문제와 시스템 문제를 못 가른다. (c) v0의 "오류 > 1% 셀 무효(arm별)" → 서버 측 실패가 많은 arm의 나쁜 rung이 지워져 그 arm에 유리(fail-open). 셋 다 기각.
- multi-GPU 차이: 없음. MG의 기술값 "TTFT 3 s"만 §5 규칙으로 바꾼다. MG §4-3 `CLIENT_ERROR` 문턱(1%)은 클라이언트 측 오류에만 적용된다고 Q23 개정에서 맞춘다.
- 예상 비판 1: "50 ms는 chat 관행값이다." → 대응: SLO-R §5 공시 그대로 + stock arm 사다리 병기. lockstep 기판에서 ITL 술어는 사실상 "토큰의 몇 %가 prefill 창에 걸리는가"를 잰다 — 이 해석을 결과에 명시한다(D §2.2–2.4).
- 예상 비판 2: "짧은 출력에서 P95는 최댓값이다." → 대응: R7-4 공시. 5-a 짧은 출력 별도 규약은 5-a PREREG 결정(D-11).

## 5. Q10 — N 간 SLO

**규칙 R10-1**: multi-GPU 판정용 TTFT 임계 = **N=1 floor 기준 절대 T**: T\_i = k·floor\_{N=1}(L\_i)를 모든 N·mode에 같은 초 값으로 적용한다.
**R10-2 (R-11)**: 판정은 **같은 Job(J\_N) 안 N=1 앵커 블록 floor**로 채점한다. J1 floor와 Job 간 floor 산포는 기술값이다. J\_N 앵커 floor가 J1과 3% 넘게 다르면 `JOB_EFFECT`(MG §15-3) 공시에 포함한다.
"같은 사용자 SLO"는 Job마다 앵커 floor가 다르므로 **Job 효과 범위 안의 근사**라고 결과에 공시한다.
**R10-3**: 기술값으로 T\_i^(N) = k·floor\_N(L\_i)(N별 상대 SLO) 채점을 병기한다. ITL 50 ms는 N 무관 절대값.

- 근거: D §3.2-4, IDX Q10, V A-9(같은 Job 원칙 우선).
- 대안·기각: N별 상대 SLO 판정 → TP가 floor를 낮춘 이득이 눈금에 흡수된다. J1 floor 하나로 판정 → 다른 Job의 값으로 채점하게 되어 같은 Job 원칙 위반. 둘 다 판정용으로 기각.
- 단일 GPU 차이: N=1뿐이므로 R6이 곧 R10이다.
- MG 정합 항목: MG §4-6 SLO 실패 분해(TTFT 3 s)와 M4 SLO(TTFT ≤ 3 s, MG §5)는 A′가 아니다. M4를 A′로 바꾸려면 PREREG 개정 + 재감사가 필요하다(M3는 SLO 무관, MG §201 항목 9). (참고 [유도]: shape B에서 3 s ≈ 3.3 × floor 0.907 s이나 A′로 간주하지 않는다.)
- 예상 비판 1: "절대 T는 큰 N에 유리하다." → 대응: 그것이 사용자 관점의 실제 이득이다. floor 감소분과 용량 증가분은 R10-3과 같이 본다.
- 예상 비판 2: "N=1 floor를 앵커 GPU 하나에서 재면 그 GPU 편차가 들어간다." → 대응: MG §4-6 GPU 편차 블록(J8) 공시, `GPU_HETEROGENEOUS`면 "GPU0 기준"으로 적는다.

## 6. Q8 — arm 집합, fused 베이스라인, 묶음 처치 공시

**규칙 R8-1 (공통 arm)**: 단일·multi GPU 공통 = {`fused`, `fused_sym`, `agnostic`, `d44`}. 정의는 MG §2를 **바이트 단위로** 따른다:
- `fused` = PD-mux off, 엔진 기본 스케줄(overlap ON, chunked prefill 8192) + `--disable-piecewise-cuda-graph`(MG rev7 사유 = py3.14 기동 실패. 단일 GPU 이미지에서도 같은 플래그를 쓰고 사유를 공시).
- `fused_sym` = PD-mux off + `--chunked-prefill-size -1 --disable-overlap-schedule`(PD-mux arm과 같은 스케줄 노브).
- `agnostic` = PD-mux on, R2 정책 없음, `pdmux_agnostic_manual.yml`(manual 등가표 [[74,34,0],[54,54,36]], divisor 36).
- `d44` = PD-mux on, `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`. sticky OFF(D-SEM [결정] 2026-10-04).
**R8-2 (단일 GPU 선택 arm)**: `fused_tuned(k)` = 등록된 유한 E-D 스윕에서 k마다 λ\*\_SLO(k)가 가장 큰 fused config. 비혼합 후보 [제안]:
`chunked-prefill-size` {2048, 4096, 8192} × `num-continuous-decode-steps` {1, 4}(cap 48 고정) = 6 config. 이 6개는 **SGLang 기본 스케줄러의 비혼합 구조**(`enable_mixed_chunk` 기본 False [코드, `server_args.py:630`])의 노브 값만 바꾼 것이다.
바꾸지 않는 것: `max-running-requests` 48, radix off, overlap on, mem-fraction 0.82. `prefill-max-requests`·schedule-conservativeness는 미시험 노브로 공시한다.
선택은 **별도 튜닝 Job**에서 하고, 판정 Job에서 선택된 config를 다시 잰다(승자의 저주 회피). 같은 SLO(A′, ITL 50 ms)로 튜닝한다(게이트 8).
대칭으로 PD-mux 쪽 튜닝 = static D ∈ {24, 34, 44} [제안] 중 k별 최선(`d_tuned(k)`). **양쪽 튜닝 config 수를 결과에 같이 적는다.**
**S-2** "튜닝 공간은 fused 9 config 대 PD-mux 3 config로 비대칭이다. 'PD-mux가 fused보다 못하다'는 PD-mux 쪽 미튜닝(span 예산·임계표 고정)을 정책의 본질로 귀속하는 문장으로 쓰지 않는다."
[2026-10-15 사용자 D-15 결정으로 fused 11 config — 비대칭 확대, S-2의 금지 문장은 그대로 적용]
**R8-2b (구조 축 — mixed-chunk)** [제안 → 예산 격자는 결정, D-15]: 비혼합 튜닝은 stall-free batching 계열 구조(한 반복 안에서 decode와 prefill chunk를 토큰 예산으로 섞음)를 대표하지 못한다. 그래서 같은 엔진 안의 구조 arm `fused_mixed` = `--enable-mixed-chunk` + `chunked-prefill-size` {512, 1024, 2048, 4096, 8192} [결정, D-15]를 둔다
(혼합 켜짐 `scheduler.py:906-909`, 예산 차감 `scheduler.py:2411` + `schedule_policy.py:398, 405-408`, 혼합 조건 `scheduler.py:2550-2568`, `schedule_batch.py:1873` [코드, CK §1]). 튜닝 fused 공간 = {비혼합 6} ∪ {혼합 5} = **11 config** [결정, D-15].
선행 조건: (i) Nano-9B-v2(mamba state)에서 혼합 배치의 출력 정확성 — engine-porter correctness gate(CK §4 G-M1–G-M4, 미실행). (ii) 의미 대조 — 완료(CK). 예산 의미는 decode 토큰 포함 반복당 총 예산이나, 혼합이 조건부이고(logprob·input\_embeds면 꺼짐) 혼합 반복은 decode 메모리 검사·retract를 건너뛰므로 **호칭은 "SGLang mixed-chunk"만** 쓴다. "Sarathi와 같은 예산 의미"는 괄호 설명으로만.
**R8-2c (mixed-chunk 운용 규약)**:
- 모든 요청은 logprob·input\_embeds 없이 보낸다(혼합 유효 조건).
- 예산 C 중 prefill 몫은 C − 실행 중 decode 수(최대 48)다. 튜닝 공간은 {512, 1024, 2048, 4096, 8192} 5점으로 등록한다 [결정, D-15 2026-10-15].
- 작은 예산 공시 [유도]: 실행 중 decode가 cap 48이면 반복당 prefill 몫은 C=512에서 512 − 48 = 464토큰(예산의 90.6%), C=1024에서 1024 − 48 = 976토큰(95.3%)이다. arXiv-Sum 유지분 입력 p50 6,484토큰 [측정, §10]을 이 몫으로 나누면 C=512에서 한 요청의 prefill이 여러 반복에 걸친다. 이것이 TTFT·λ\*에 주는 영향은 **예측하지 않고 공시만** 한다.
- PD-mux arm에서 `enable_mixed_chunk`가 켜지지 않도록 launch 단계에서 assert한다(상호 배제 검사 없음, PD-mux loop도 같은 함수 호출 [코드, `multiplexing_mixin.py:1227`]). radix ON·mamba extra\_buffer와 결합하지 않는다(`schedule_batch.py:2291`).
- G\_mixed 결과 **의무 꼬리표**: (1) "혼합 반복의 decode 토큰은 mamba prefill 커널(chunk-scan)로, piecewise cudagraph off에서 eager로 처리됨" [코드, `mamba2_metadata.py:194-196`, `forward_batch_info.py:167-173`] (2) "logprob 없음" (3) "혼합 반복은 retract 검사 생략" (4) "piecewise 불가 이미지".
- `fused_mixed`는 (i) gate 통과 전에는 튜닝·판정 Job에 넣지 않는다. (i) 판정 기준은 완화형(G-M2: off 대 off 기준선 대비 완전일치 비율 하락 ≤ 1요청, 첫 32토큰 100% 일치, 크래시·NaN 0)이다. 혼합 실현은 holb\_probe의 MIXED forward 수로 확인한다(G-M1).
**S-1** "G0의 fused 계열은 이 이미지(py3.14)에서 piecewise CUDA graph를 쓸 수 없어 혼합·prefill 반복이 eager로 돈다. G_def·G_tuned·G_mixed는 'piecewise 불가 이미지의 SGLang' 대비이며 SGLang 일반·Sarathi-Serve·vLLM·stall-free batching 일반 대비가 아니다."
vLLM 별도 arm은 판정에 쓰지 않는다(커널·cudagraph·메모리 관리가 달라 원인 분리 불가, 게이트 1의 연장). 넣더라도 참고 열(기술값)로만 둔다(D-14).
**R8-2d (G\_mixed 분모·동률·대체, R-4 → v0.2 D-16 (a))**:
- 분모 = 튜닝 Job에서 **합집합 11 config 중 k별 R9-1 보간 λ\*\_SLO(k)가 가장 큰 config(엄격 argmax)**(`fused_best11(k)`)를 판정 Job에서 다시 잰 값. 혼합 5·비혼합 6 가족별 최선값은 기술값.
- 동률 [결정, D-16 (a) 2026-10-15]: 등록 보간 코드가 낸 λ\*\_SLO(k) 값이 **정확히 같을 때만**(같은 부동소수점 값) 동률이다. 해상도 띠·CI 겹침 같은 데이터 의존 띠는 쓰지 않고, **λ\*\_thr는 동률 기준으로 쓰지 않는다**. 정확 동률일 때만 다음 결정적 순서로 하나를 고른다:
  (1) 비혼합 우선 → (2) `chunked-prefill-size`가 큰 쪽 → (3) `num-continuous-decode-steps`가 작은 쪽(1 → 4). 이 순서는 11 config 위의 전순서다. `d_tuned(k)`는 D가 큰 쪽.
  v0.1의 띠(R9-6 해상도)·1차 기준(λ\*\_thr) 규칙은 삭제했다(V2 死因 D-C′: 띠가 데이터 의존이고 λ\*\_thr가 chunk 축에서 λ\*\_SLO와 반대로 움직여 분모가 처리율 최선 config로 돌아감).
- fused\_mixed gate (i) 불통과 시(성능 데이터를 보기 전 확정되는 사건): 1차 가족 = G\_tuned(비혼합 6 config), headline 범위 문장 = "SGLang 기본 스케줄러(비혼합 chunked prefill)를 등록된 6개 config 안에서 같은 SLO로 튜닝한 것 대비"(S-1·S-2 병기). 이 대체는 지금 등록한다.
**R8-3 (estimand와 문장 종류의 고정 대응)**:

| estimand | 정의 | 지위 | 허용 문장 | 금지 문장 |
|---|---|---|---|---|
| S | λ\*\_fused\_sym / λ\*\_fused | 기술값 | "스케줄 노브(chunk −1, overlap off) 효과" | PD-mux 효과로 귀속 |
| G\_sym | λ\*\_agnostic / λ\*\_fused\_sym | 단일 GPU 2차 가족, multi-GPU 1차(F6) | "같은 스케줄 노브에서 PD-mux 기능(SM 분할 + split prefill + 입장 기계) 전체의 효과" | "SM 분할 효과", "PD 분리 효과"(MGP-29) |
| G\_def | λ\*\_agnostic / λ\*\_fused | 기술값 | "미조율 기본 fused 대비 PD-mux 모드(묶음 처치)" — 선행 연구 정렬용 | 원인 귀속, 조율 공간 소진(T2) |
| G\_tuned | λ\*\_d\_tuned(k) / λ\*\_fused\_tuned6(k) | 기술값(gate 불통과 시 1차) | "등록된 비혼합 6 config 안에서 양쪽을 같은 SLO로 튜닝했을 때" | "fused를 어떻게 조율해도" |
| **G\_mixed(합집합판)** | λ\*\_d\_tuned(k) / λ\*\_fused\_best11(k) (R8-2d) | **단일 GPU 1차 가족** | "SGLang mixed-chunk(예산 512–8192, logprob 없음, piecewise 불가 이미지)와 비혼합 6 config를 합친 등록 11 config 튜닝 대비" | "chunked prefill 일반 대비", "vLLM·Sarathi 대비", "stall-free batching 일반 대비" |

**1차 estimand 선택 이유(R-2)**: P-C [결정]에서 결과의 headline은 "튜닝한 fused 대비 PD-mux" 문장이고, R8-5대로 그 문장은 튜닝 분모가 있을 때만 쓸 수 있다. 확증 가설이 headline 문장과 같은 estimand여야 headline이 2차 가족이나 기술값에 기대지 않는다.
합집합판은 혼합·비혼합 중 데이터가 고른 쪽을 분모로 두므로 "fused 쪽 최선 구조"를 고르는 자유를 판정 전에 닫는다(가족별 최선을 사후에 고르는 경로 차단). G\_sym은 같은 노브 대조군이라 기전 쪽 2차 가족으로 둔다.
감수 비용: 튜닝 Job 의존, gate (i) 의존(R8-2d 대체로 정의는 항상 존재), 11 대 3 비대칭(S-2와 그 주석).
**S-4** "G_def = S·G_sym은 회계 항등식이다. 기전 분해나 원인 지분('PD-mux 이득의 x%는 스케줄 노브')으로 쓰지 않는다." G\_def = S·G\_sym의 호칭은 **"경로 분해(회계 항등식)"**이며(R-15), 이 식으로 "일관성 확인"을 두 번째 증거로 세지 않는다(#269).
**R8-4 (플래그 공시 규약)**: arm을 비교하는 모든 표·그림에 열 5개를 의무로 둔다 — `pdmux`(on/off), `chunked_prefill_size`, `overlap_schedule`, `piecewise_cuda_graph`, `enable_mixed_chunk` — 와 "요청 logprob 없음" 확인 열.
`fused_mixed` 행에는 **혼합 실현 지표**(R-6) = MIXED forward 수 / (MIXED + running batch가 비어 있지 않은 EXTEND forward 수)를 적는다. 분모는 혼합 기회가 있었던 반복만 세므로 낮은 rate에서 자연히 낮아지는 성분을 뺀다.
holb `mode_counts`로 분모(running 비어 있지 않은 EXTEND)를 낼 수 없으면 I-5에 "EXTEND 반복의 running\_bs" 기록을 추가한다. 그것도 없으면 MIXED/(MIXED+EXTEND)를 쓰고 rate별 기대값(Poisson 도착에서 running 비어 있지 않을 확률 [유도])을 병기한다.
G\_mixed 근거에서 셀을 뺄 때는 이 지표(분모 수정판)의 문턱만 쓴다(문턱은 G0 PREREG). raw MIXED/(MIXED+EXTEND) 문턱으로 셀을 빼지 않는다(낮은 rate 셀을 체계적으로 지움).
**K9** "혼합 실현 지표는 혼합이 일어났음을 확인하는 값이지 혼합 효과의 크기가 아니다."
fused → agnostic 비교에는 "묶음 처치(PROJECT\_STATUS 게이트 #13)" 꼬리표를 붙인다. "SM 분할" 문장은 E-time ≥ 0.5(MG §4-3)일 때도 "SM 분할 포함 기능 전체"까지만.
**R8-5 (기본 fused vs SLO-튜닝 fused)**: 둘 다 두되 **역할을 나눈다**. 기본 fused = 선행 연구·정본 P1-opint와의 정렬(G\_def, 기술값). SLO-튜닝 fused = 공정성 문장(G\_mixed 1차, G\_tuned 기술값/대체).
비혼합 G\_tuned만으로 쓸 수 있는 최대 문장은 "SGLang 기본 스케줄러(비혼합 chunked prefill)를 등록된 6개 config 안에서 튜닝한 것 대비"다. 혼합 쪽은 "SGLang mixed-chunk(예산 512–8192, logprob 없음, piecewise 불가 이미지) 대비"로만 쓴다(R-5).
"PD-mux가 fused보다 낫다"는 형태의 headline은 **1차 가족(G\_mixed 또는 R8-2d 대체 G\_tuned)이 R5-3을 통과했을 때만** 쓰고, 아니면 headline은 "미조율 기본 fused 대비"로 범위를 줄인다(정본 §1-1 문안과 같은 수준).

- 근거: A §5-5, D §3.1–3.2-2, R §1(정본: fused 미조율, T2 주장 금지, 미시험 노브 6개 이상), R §4.1 게이트 8·#13, MG §2·§4-4, CK §2·§5, V A-6·A-7·A-7b·D-C.
- 대안·기각: (a) 기본 fused만 → 미조율 confound 재생산. (b) fused\_sym만 → 대조군이지 베이스라인이 아니다. (d) 무한/적응적 튜닝 → 사전등록 불가. (e) 혼합 5만 분모 → 비혼합 쪽이 더 나은 k에서 분모가 낮아져 PD-mux에 유리(D-C). (f) v0.1 동률 띠 + λ\*\_thr 1차 기준 → 분모가 처리율 최선 config로 돌아가 G\_mixed 부호가 띠 폭에 따라 갈림(V2 D-C′). 모두 기각.
- 비대칭 방향(V A-7): 11 대 3(v0.1의 9 대 3에서 D-15로 확대)은 "PD-mux 우위" 문장에는 보수적, "PD-mux 열위" 문장에는 비보수적이다 → S-2.
- multi-GPU 차이: R8-2(튜닝 스윕)는 비용 때문에 multi-GPU에 넣지 않는다 [제안]. multi-GPU 문장은 G\_sym·G\_def 범위이고 d44는 "1-GPU에서 고른 D44를 N에 그대로 쓴 값"(MGP-15)이다.
- 예상 비판 1: "fused\_sym은 허수아비 대조군이다." → 대응: headline 분모가 아니라 같은 노브 대조군(2차 가족)이다. G\_def = S·G\_sym은 경로 분해(회계 항등식)일 뿐 기전 분해가 아니다(S-4).
- 예상 비판 2: "튜닝 공간 11개도 작다." → 대응: 맞다. 공간을 문장에 명시하고 T2(조율 공간 소진)는 계속 금지한다. 공간 확장은 PREREG 개정으로만.

## 7. Q11 — 전력·클럭: arm × 창 cap 노출 공변 검사

**규칙 R11-1 (수집)**: 모든 Job에 GPU별 SM 클럭·메모리 클럭·throttle reason 비트(SW power cap, HW slowdown 포함)·전력을 수집한다. 주기: 전력 20 ms, 클럭 100 ms(N2 관행 [측정, A §1]).
수집이 0행이면(N2 F-2 `sm_clk_samples.csv` 0행 문제) 그 Job의 셀은 `SUBSTRATE_UNCHECKED` — 크기 인용 금지.
**R11-2 (검사, 판정 전 단계)**: 판정량의 **분자·분모 창마다**(#268) cap 노출률(SW power cap 비트가 켜진 시간 비율)과 SM 클럭 중앙값을 낸다. 비교하는 두 arm 사이에서
노출률 차 > 10 %p 또는 클럭 중앙값 차 > 1% [제안, D-4]이면 그 셀은 `SUBSTRATE_COVARIES`. 창 정렬은 I-2의 시계 쌍으로 하고, **정렬 오차 상한**(시계 쌍 사이 최대 drift + 클럭 표본 주기)을 셀마다 공시한다.
정렬 오차 상한 > 클럭 표본 주기 2개(200 ms) [제안]이면 그 셀은 `SUBSTRATE_UNCHECKED`(V A-5).
**R11-3 (처리)**: `SUBSTRATE_COVARIES` 셀은 부호만 인용하고, 크기 문장은 "클럭 하락 성분을 포함한 값"으로 범위를 붙인다. 기전(스케줄링 구조) 귀속 금지.
이 검사는 매개(PD-mux가 전력을 더 써서 클럭이 떨어짐)와 교란(arm과 무관한 기판 변동)을 **구분하지 못한다**고 결과에 적는다. 분리는 F-1(burner) 같은 별도 등록의 몫이며 그 실현성은 미확인이다.
**S-5** "SUBSTRATE_COVARIES 셀의 부호는 '400 W 한도 기판에서 클럭 하락을 포함한 총효과'다. 스케줄링 구조로 귀속하지 않는다."
**R11-4**: 검사 코드와 문턱은 PREREG 봉인 전에 고정하고, 변이 시험(교훈 53: 창 정렬을 한 칸 밀거나 throttle 비트를 무시한 변이본이 합성 데이터에서 실패해야 함)을 하네스 회귀에 넣는다.
**R11-5**: 문턱 사다리 {5, 10, 20} %p에서 라벨이 바뀌는 셀 수를 기술값으로 병기한다(D-4). 사다리는 인용 범위만 바꾸고 판정 부호를 바꾸지 않는다.

- 근거: A §5-8, #268(N2 Job-EA C 창 64/64 cap 노출 → ρ ≈ 1.16 크기 CONFOUNDED [측정, N2C-102]), λ0 cap 노출 A 27%·B 49% [측정], N3 창별 11.5–47.4% [측정].
- 대안·기각: 전력 고정(`-pl`) → VESSL 컨테이너 불가 [측정, power-cap 트랙]. 클럭 고정(`-lgc`) → rc=4 [측정, 2026-10-07 프로브]. 사후 분석만 → N2에서 결과 뒤에 발견된 실패 그대로. 그래서 "통제 불가 → 검출 + 범위 제한".
- multi-GPU 차이: GPU별로 잰다. TP는 rank 중 최대 노출률, DP는 복제본별 + 평균 [제안]. MG §1 `clk.csv` index 열이 이미 GPU별이다.
  **R-16 범위 문장**: "MG rev11의 M3는 명세 v0 이전에 등록되었고, I-1–I-5 계측 스키마가 M3 봉인 엔진 digest에 없으므로 R11-2를 충족하지 않는 범위다. 이 명세를 M3 rev11 결과에 소급 적용하지 않는다."
- 예상 비판: "PD-mux가 GPU를 더 바쁘게 써서 클럭이 떨어진다면 그건 처치의 일부다." → 대응: 맞다. R11-3은 셀을 버리지 않고 S-5 범위를 붙인다.

## 8. Q9·Q12 — 용량 정의와 headline 위계

**규칙 R9-1 (두 용량 병기)**:
- **λ\*\_thr**(처리율 knee) = MG §4-2·4-3 방식 그대로: HI rung 4 seed 전부 g ≥ 0.10(`HI_SATURATED`) ∧ LO rung g ≤ 0.05(`LO_UNSATURATED`) → `KNEE_BRACKETED`, λ\* = HI achieved 중앙값. g는 클라이언트 측 Q\_c 시간 적분 평균의 3분할 기울기.
- **λ\*\_SLO(k)(판정값, R-1)** = (Job, arm, k)마다 공통 격자의 rung r에서 **seed 합산 attainment** a(r) = Σ합격 / Σ요청(R5-6)을 내고, 오름차순으로 a가 0.90 이상에서 미만으로 처음 내려가는 인접 rung (r\_l, r\_u)에서 **log-rate 선형보간**:
  log λ\* = log r\_l + (a(r\_l) − 0.90) / (a(r\_l) − a(r\_u)) · (log r\_u − log r\_l). 이 보간 방식 하나만 등록한다.
  - 가장 낮은 rung에서 이미 a < 0.90 → `BELOW_LADDER`, 가장 높은 rung까지 a ≥ 0.90 → `ABOVE_LADDER`. 둘 다 λ\* = `UNRESOLVED`(외삽 금지).
  - (r\_l, r\_u) 중 어느 rung이든 R7-3 무효 또는 S1–S3 비정상이면 λ\* = `UNRESOLVED`(건너뛰기 금지).
  - 첫 교차 뒤 a가 다시 0.90 이상으로 오르는 재교차 수를 기술값으로 적는다(판정값은 첫 교차).
  - **K2** "교차 rung이 비정상이면 λ\*_SLO는 `UNRESOLVED`다(N3 rev7에서는 k=5·10이 이에 해당). `UNRESOLVED_J`는 효과 없음이 아니다."
- **rung 라벨**(N3 첫 실패 규칙: 첫 실패 rung 바로 아래 rung)은 **기술값**으로 병기한다. attainment 99%는 기술값 병기(SLO-R §3-5).
**R9-1b (보간 dry-run, 봉인 전 단계)**: 이 보간 코드를 PREREG 봉인 전에 N3 rev7 아카이브(`results/n3_capacity/n3b_pdmux-n3-capacity-93c3cca-20261007064124/`)로 CPU에서 돌려 rung 라벨과 나란히 비교한다(GPU 0).
목적은 코드 경로·`UNRESOLVED` 빈도·재교차 빈도 확인뿐이며, **산출된 보간값은 인용하지 않는다**(사후·비등록). 결과는 dry-run 기록 파일 하나로 남기고 봉인 대상에 그 sha256를 넣는다.
**R9-2 (cliff 경계, R-8)**: λ\*\_SLO 교차 rung(r\_l, r\_u)의 셀에서 |a − 0.90| ≤ 2·√(0.09 / n)(n = 그 셀의 seed 합산 요청 수)이면 `CLIFF_ADJACENT` — 크기 인용 금지, 곡선만 보고.
`CLIFF_ADJACENT` 셀이 있는 비교에서 두 arm의 λ\* 차가 1 rung 이하이거나 보간 CI가 겹치면 **부호도 인용하지 않는다**. 보간 CI [제안] = r\_l·r\_u의 a를 각각 ±2·√(a(1−a)/n)만큼 움직여 보간식에 넣은 λ\*의 최소–최대.
**K1** "arXiv-Sum 곡선이 N3 rev7과 비슷하면 k=3·5의 λ\*_SLO 비교는 `CLIFF_ADJACENT`와 R9-6 해상도 때문에 크기 문장에 도달하지 못하며, 부호도 1 rung을 넘는 차이에서만 인용한다. Holm 확정은 인용 허가가 아니다."
**R9-3 (사다리 조건)**: 용량 측정을 판정보다 먼저 한다(게이트 6). 공통 격자는 비교하는 **모든 arm의** 예상 λ\* 위·아래로 각각 2 rung 이상, 인접 rung 비 ≤ 1.2 [제안]. 사다리는 데이터를 보기 전에 등록한다.
**R9-4**: 두 용량에서 G를 각각 계산한다. 판정은 R5-3b의 가족 지정대로만 한다(λ\*\_thr 기반 G는 단일 GPU에서 기술값). 두 값의 부호·크기가 갈리면 기술값으로 보고한다.
**R9-5 (양자화 금지, R-1)**: rung 라벨, rung 개수 차, "첫 실패 rung 비" 같은 격자 양자화 값으로 t·CI·크기 문장을 만들지 않는다.
**R9-6 (해상도 등록, R-14)**: PREREG에 셀당 seed 합산 요청 수 n과 이항 해상도 √(0.09/n)를 등록한다. λ\* 해상도 = 2·√(0.09/n) / |a(r\_l) − a(r\_u)| · log(r\_u / r\_l)(교차점 log-rate 환산) [유도 정의].
λ\* 해상도 > 3%인 비교는 크기 문장 금지(부호는 R9-2 조건 안에서만). 셀 창 길이(초)도 셀마다 등록·공시한다. 창 길이 의존 검사(같은 Job 안 긴 창·짧은 창 병치)는 이 명세에 없다 → 크기 문장에 "창 T s에서" 범위를 붙인다.
**K3** "R9-6 해상도는 이항 잡음만 반영하고 rung 사이 보간 곡률 편향은 반영하지 않는다."
**C-1(명세 관련분)** "명세 v0가 게이트 #6 확정 조건을 충족한다" 금지. "G0가 게이트 #6을 닫았다"는 게이트 재정의(사용자 결정 + promote) 없이 금지.
**R12-1 (headline 위계)**: ① 1차 가족의 λ\*\_SLO(k=3/5/10) attainment 90% 용량 + λ\*\_thr 병기 → ② 고정 rate의 TTFT/ITL p50/p95/p99와 실패 분해(R7-5, 길이 3분위) → ③ CUDA event 기전 분해(§9 I-3·I-4). ③은 headline이 될 수 없다(게이트 1).
**R12-2**: 실행 전 roofline 예측(B\_max 등)을 기록하되, 엔진 실효 상한(부팅 로그 KV 풀 토큰, mamba 슬롯 = cap)과 **별개 열**로 둔다(A §5-7, Q21).

- 근거: D §3.2-3, A §3.1·§5-6, MG §4-2·4-3, A §1(N3 첫 실패 규칙·cliff), V D-A(3% 문턱이 rung 해상도보다 작아 양자화 λ\*로는 분산 0 → t 퇴화).
- 대안·기각: λ\*\_thr만 → 사용자 관점 지연이 없다. 첫 실패 rung 라벨을 판정값으로 → D-A. 로지스틱 등 모형 적합 → 자유 표면 증가. goodput req/s 곡선 주 지표 → 용량 정보 없음. 모두 기각.
- multi-GPU 차이: M3는 λ\*\_thr가 판정이고 SLO는 기술값이다(D-9 [결정]). λ\*\_SLO를 M3 봉인 하네스에 넣으면 하네스 digest가 바뀐다 → M3에서는 **봉인 밖 분석기의 "비등록 사후 기술값"**으로만 산출한다(V A-13 ②).
- 예상 비판 1: "cap 48이 구속하면 λ\*\_thr가 cap 비율로 수렴해 G → 1로 퇴화한다(MG §8-2)." → 대응: §12 R16-2 공시 + I-5.
- 예상 비판 2: "보간은 rung 사이 attainment가 선형이라고 가정한다." → 대응: 맞다. 그래서 R9-2 CI와 R9-6 해상도로 그 가정의 영향 범위 안에서만 크기를 쓴다.

## 9. 계측 필수 목록 (Q11·Q14) — 단일·multi GPU 공용

★ 표시 = **엔진 패치 → engine-porter correctness gate 선행**. 나머지는 하네스 수정(하네스 회귀 테스트 선행).

| ID | 항목 | 내용 | 용도 | G0 전 필수? |
|---|---|---|---|---|
| I-1 | 클럭·전력·throttle 수집 | R11-1. F-2 0행 문제 수리, 셀 시작 전 수집 생존 검사 | R11 공변 검사 | **필수** |
| I-2 | clk ↔ telemetry 시간 정렬 ★ | telemetry 시작·끝·매 N초에 (CLOCK\_MONOTONIC, CLOCK\_REALTIME) 쌍 기록. 정렬 오차 상한 공시(R11-2) | R11-2 창 정렬, F-1 | **필수** |
| I-3 | span GPU 시간 ★ | prefill stream CUDA event로 span 시작·끝, 층 범위 [split\_index, next), 배치 토큰 | S\_prefill·정지 분해, §11 P3·P4 | **필수** |
| I-4 | decode 커널 GPU 시간 ★ | decode stream CUDA event로 decode forward만. 기존 `t_launch→t_sync`와 별도 필드 | CM-1 vs CM-4 분리 | **필수** |
| I-5 | admission 거절 사유 ★ | 스케줄 결정마다 사유 ∈ {KV, cap, mamba 슬롯, prefill 자리 점유, `max_prefill_tokens`, R2 latch}, retract 이벤트, EXTEND 반복의 running\_bs(R8-4) | §12 cap 구속, X-6, 혼합 실현 분모 | **필수** |
| I-6 | cudagraph 패딩 ★ | decode iteration마다 bs\_real, bs\_padded, graph hit / eager | ITL 분해 | 선택(G0 안 권고) |
| I-7 | iteration Σctx·ctx p95 ★ | 32 sync 표본 → iteration 단위 | CM-1 ctx 항 | 후속 |

- 근거: A §4.4, D §2.4-4, K §3-3, IDX Q14. 번호 모호성(A §4.4 번호 대 IDX 나열 순서)은 합집합(I-1–I-5) 필수로 처리했다(D-5 [결정]).
- **correctness gate 조건**(★ 항목 공통): (a) 계측 ON/OFF 출력 일치 — greedy, 등록 prompt 64개, MG G1과 같은 m·q 지표.
  (b) **관측자 효과(R-9)**: 계측 ON(판정 수준) 대 OFF(계측 최소 수준) 짝에서 λ\*\_thr의 log 비와 ITL p95의 log 비 **둘 다** TOST 동등성 여유 ±1.5% [제안](양측 90% CI가 ±1.5% 안)이면 통과, 아니면 `OBSERVER_EFFECT`. 점추정 문턱은 쓰지 않는다.
  짝 수 n ≥ 8을 **PD-mux 계열(agnostic)과 fused 계열(fused\_sym) 각각** 채운다(Job 합산 가능, Job은 짝 차분의 층으로 둔다). I-5는 fused 스케줄러도 고치므로 PD-mux 쪽만 보면 arm 비대칭 관측자 편향이 숨는다.
  **계측 최소 수준** = 판정량 계산에 필요한 클라이언트 측 기록 + I-1(외부 프로세스 수집)만 켜고, I-2–I-7·holb\_probe·chunk\_probe·PHASE\_EVENTS 계열 env는 끈 상태.
  **K4** "관측자 TOST 통과는 G에 arm 비대칭 편향을 최대 ±3%(log)까지 허용한다. 3–6% 효과의 크기는 관측자 편향과 분리되지 않는다."
  (c) 변이 시험: I-3을 decode stream에 기록한 변이본, I-4가 host 시간을 쓰는 변이본이 "격리 prefill에서 Σspan GPU 시간 ≈ prefill 벽시계 ±5%" 검사에서 실패해야 함 (d) 텔레메트리 스키마 버전 증가.
- 관측자 블록 배치: 판정 Job마다 (b)의 ON/OFF 짝을 두 계열 모두 넣는다(holb\_probe G5 관측자 효과 미결정 선례, D §1.2). 게이트 #6 ④(ON/OFF 짝 n ≥ 8)와 같은 크기이나, G0는 워크로드·엔진 트리가 달라 원 게이트 #6 조건을 대신하지 않는다(C-1).
- multi-GPU 차이: 모든 항목 rank 접미사. TP는 rank 0 판정 + 전 rank 기록, DP는 복제본별. 패치는 merge 뒤 공통 트리 하나에만 넣는다(merge 전 이중 패치 금지 [제안]).
- 예상 비판 1: "CUDA event 계측 자체가 lockstep 동기화를 바꾼다." → 대응: query/sync 위치가 바뀌면 교란이 생긴다. gate (b) TOST와 Job 안 관측자 짝으로 검출한다.
- 예상 비판 2: "계측 패치가 G0를 늦춘다." → 대응: 맞다(감수 비용). I-3·I-4 없이는 정지 vs decode 감속을 가를 수 없다.

## 10. arXiv-Sum 표준 셀 (Q3 [결정])

- 데이터: `ccdv/arxiv-summarization` document/test, sha256 `b34529df…` [측정, ARXSUM_STATS]. 필터 = Nano-9B-v2 토크나이저 입력 ≤ 16,384 → 유지 5,906/6,440 = 91.7% [측정].
- 유지분 입력 mean 7,116 · p50 6,484 · p90 12,369 · p99 15,722, 출력 mean 203 · p50 195 [측정]. 유지분 중 > 8,192 토큰 34.4% [측정], > 8,760 토큰(KV = state 경계) 30.0% [측정 비율, 경계값은 유도].
- 표본 추출: seed별 비복원 추출, 도착 Poisson(R4-1), 출력 길이 = abstract 길이(R7-3). `max_prefill_tokens` 값을 PREREG에 명시 등록(기본 16384, MG 개정 대기 항목).
- 모델 간 비교(G2)는 문서 ID 교집합을 쓴다(D-12 [결정]; 길이 차이 ≤ 3% [측정, R §2.1]이나 유지 문서 집합이 달라지므로).

## 11. span 예산 스윕(Q13)을 G0에 넣는 방식

**규칙 R13-1 (지위)**: span 예산 스윕은 **lockstep 기판 특성화**(F4)이고 정책·arm 판정이 아니다. headline arm은 upstream 기본값 `split_forward_token_budget = 65536`에 고정한다.
예산을 바꾼 PD-mux를 headline에 쓰면 PD-mux 쪽만 튜닝한 것이 되므로 금지(대칭 튜닝은 R8-2 공간에 넣을 때만).
**R13-2 (배치)**: G0 각 Job 끝에 블록 G0-S로 둔다(Job ≥3 반복을 그대로 받음). arm = `d44`, rate = LO(0.6·λ\*\_thr,d44)와 MID(0.85·λ\*\_thr,d44) [제안], seed 4.
G0-S 셀 = 예산 4점(R13-3) × rate 2 = Job당 8셀(각 seed 4) [유도, D-6 결정 반영] — 3점 격자(6셀) 대비 4/3배. **G0-S를 포함한 Job 시간 예산(하드캡·GPU 사용률)은 G0 PREREG 몫**이다(이 명세는 정하지 않는다).
**R13-3 (예산 격자 — 전제 정정, R-10)**: 층 수 = `max(1, budget // extend_num_tokens)` [코드, `multiplexing_mixin.py:1416-1421`].
- "8k ≡ 16k"는 **2요청 배치(≈14k 토큰 [유도])에서만** 참이다. LO rate에서는 단일 요청 배치가 지배적일 것으로 본다 [유도, 도착률 대 서비스 시간에서의 추론, 실현 비율은 I-3로 확인].
  arXiv-Sum 유지분 입력의 65.6%가 ≤ 8,192 토큰이다 [측정, ARXSUM\_STATS의 > 8,192 = 34.4%에서]. 단일 요청 배치에서 입력 4,097–8,192는 예산 8k → 1층, 16k → 2층이고, ≤ 4,096은 16k가 8k의 2배 층 수다 [유도].
  따라서 8k와 16k는 같은 처치가 아니며, 8k는 lockstep 정지를 줄이는 실무 방향의 유일한 격자점이다. v0의 {16k, 32k, 65k} 권고 근거는 이 점에서 성립하지 않는다(V A-10).
- **격자 = {8192, 16384, 32768, 65536}** [결정, D-6 2026-10-15 — 감사 권고 확장안, 셀 비용 4/3배 [유도]]. 65536은 headline 기본값(R13-1)과 같은 점이다.
**R13-4 (등록 예측)**: (P1) prefill 창 안 decode 토큰의 ITL 중앙값은 예산(span 일)에 단조 증가, (P2) TTFT p50은 예산 감소에 단조 증가(반복 수 증가), (P3) I-3 span GPU 시간 ∝ span 일(기울기 기술값, 계측 타당성 확인용).
**(P4, 판별 예측 — 형식만 지금 등록, 여유는 [미등록])**: span 층 수가 같은 span끼리 묶어, prefill 창 안 decode 토큰의 ITL 초과분(창 밖 ITL 중앙값 대비)을 그 span의 I-3 token-layer GPU 시간에 회귀한다.
lockstep-GPU 가설(decode가 span GPU 작업을 기다림)은 기울기 > 0, CPU-launch 가설(launch 오버헤드 지배)은 기울기 ≈ 0을 예측한다.
**"≈ 0"의 동등성 여유와 기울기 CI 방식 = G0 PREREG 등록 필수 항목, 값 [미등록]**(V2 B-7). 등록 전에는 P4를 판별 예측으로 인용하지 않는다.
**K8** "P4는 '≈ 0' 동등성 여유가 등록되기 전에는 판별 예측이 아니다."
P1·P2 지지를 어느 가설의 지지로도 쓰지 않는다. P1이 깨지면 D §2.2 가설을 기각 방향으로 기록하되, §1-24 문안을 고치는 것은 Q15(claims-auditor 회부) 경로로만.
**S-6** "G0-S의 P1·P2 적중은 lockstep 가설과 CPU-launch 가설을 가르지 않는다."

- 근거: D §2.3–2.4, C §4-2(a), K §0, V A-10(층 수 식은 맞고 퇴화 범위가 틀림).
- multi-GPU 차이: M3 개정 시 예산을 등록값으로 적기만 한다(65536). 스윕은 단일 GPU 깊이 항목(D §3.2-1).
- 예상 비판 1: "예산은 PD-mux의 튜닝 노브인데 기본값에 묶으면 PD-mux를 불리하게 둔다." → 대응: R13-1. 튜닝 PD-mux vs 튜닝 fused는 R8-2 대칭 공간에서만.
- 예상 비판 2: "LO·MID 두 rate에서만 재면 예산이 용량을 바꾸는지 모른다." → 대응: 맞다. G0-S는 기전 특성화이고 λ\* 판정이 아니다.

## 12. cap 처리(Q16)를 G0에 넣는 방식

**규칙 R16-1**: G0·M3의 cap = 48(단일·N=1), TP 48·N, DP 복제본당 48 [결정 F7]. `max_mamba_cache_size` = cap을 명시 인자로 준다(§1-23). 부팅 로그의 `max_total_num_tokens`·mamba 슬롯 수를 셀마다 기록.
**R16-2 (구속 공시, R-12)**: `CAP_BINDING` 판정은 **λ\*\_SLO 교차 rung(r\_l, r\_u) 셀과 LO rung에서만**, **running = cap 시간 비율 ≥ 10%** [제안, D-10]로 한다. "cap 사유 거절 > 0" 조건은 쓰지 않는다(HI 포화 rung에서 항등식에 가까움).
시간 비율은 I-5 기록으로 계산한다. I-5가 없거나 0행이면 `CAP_BINDING_UNRESOLVED`. `CAP_BINDING` 셀의 λ\*·G는 "cap 48 조건부"로만 쓴다. I-5 거절 사유별 건수는 모든 셀에서 기술값으로 적는다.
**R16-3 (cap 스윕은 G0 밖)**: {48, C\_mem/2, C\_mem} 스윕은 비용 모델 검증 Job(Q17–Q19)의 몫이다. C\_mem은 roofline 값(~183 [유도, K §2])이 아니라 **부팅 로그 실측**으로 정한다. cap을 바꾸면 cudagraph 캡처 bs 목록도 바뀌므로(K §1) I-6을 켠다.

- 근거: K §1–4, MG §1·§8-2, V A-11. cap 48은 근거 없이 상속된 값 [코드/이력, K §1]. 수요 하한 C ≥ λ·E[out]·E[ITL]이 lockstep 정지 수준 ITL(~200 ms)에서 ~60 > 48 [유도, K §2(3)].
- 대안·기각: (b) 워크로드별 도출 cap 등록 → 미검증 비용 모델 출력을 처치값으로 씀(게이트 1). (c) G0에서 스윕 → G0 질문이 흐려진다. 둘 다 기각.
- 예상 비판 1: "cap 48은 PD-mux에 불리/유리할 수 있다." → 대응: 모든 arm이 같은 cap이다(MG §8-2). 구속 여부를 셀마다 공시한다.
- 예상 비판 2: "hybrid에서 cap은 state 예약(48 × 0.1435 GB ≈ 6.9 GB [유도])까지 정하므로 사실상 처치다." → 대응: 그래서 cap 스윕을 비용 모델 검증의 독립 Job으로 둔다.

---

## 13. 사용자 결정 항목 (2026-10-15 사용자 결정: D-1…D-14 권고안 채택, Q2 = P-C, Q3 = arXiv-Sum 16k. 감사 2회차 뒤 같은 날 D-6·D-7·D-15·D-16 결정됨. 남은 [제안] 상태 = D-10)

| # | 항목 | 선택지 | 상태 |
|---|---|---|---|
| D-1 | floor 기준 config(R6-4) | fused\_sym / fused / PD-mux 무분할 | [결정] **fused\_sym** + R6-4 조건 (a)–(c) |
| D-2 | Job 반복 수(R5-1)와 M3 적용 | J=3 / J=4 / M3 N=1 셀만 J≥3 | [결정] G0 **J=3**(GPU 사용률 계산은 G0 PREREG), J=4는 첫 G0 결과가 판정 불가일 때. M3는 Job 안 판정 범위 |
| D-3 | fused 튜닝 스윕과 headline 분모 | 없음 / 비혼합 + G\_tuned / 비혼합+혼합 + G\_mixed | [결정] **비혼합+혼합 + G\_mixed**(v0.1 9 config → D-15로 **11 config**). 분모 = 합집합 엄격 argmax(R8-2d, D-16), gate 불통과 대체 등록 |
| D-4 | 공변 문턱(R11-2) | 10 %p·1% / 문턱 사다리 병기 | [결정] 10 %p·1% + {5, 10, 20} %p 사다리 기술값 |
| D-5 | G0 전 계측 범위(§9) | A §4.4 1·2·5 / IDX 나열 / 합집합 | [결정] **합집합(I-1–I-5)** |
| **D-6** | span 예산 격자(R13-3) | 유지 {16k, 32k, 65k} / **확장 {8k, 16k, 32k, 65k}(감사 권고)** / 원안 {8k, 16k, 65k} | **결정됨 2026-10-15: 확장 {8k, 16k, 32k, 65k}**. G0-S 셀 4/3배 [유도], Job 시간 예산은 G0 PREREG(R13-2). 이전 근거("8k ≡ 16k")는 V A-10에서 반박됨 |
| D-7 | 오류 요청 처리(R7-3) | 오류 > 0 무효 / 1% 문턱 / 불합격만 | [결정] 1% 문턱 + 불합격 계산 → R-3로 적용 대상을 클라이언트 측 오류로 한정, 짝 무효·`UNRESOLVED`. **결정됨 2026-10-15: 범위 축소 사용자 확인** |
| D-8 | L < 1k 처리(R6-3) | floor(1k) 고정 / 512 측정점 추가 | [결정] floor(1k) 고정 + 비율 공시 |
| D-9 | M3 용량 판정(§8) | λ\*\_thr 유지 / λ\*\_SLO 승격 | [결정] **λ\*\_thr 유지, λ\*\_SLO 봉인 밖 비등록 사후 기술값** |
| D-10 | `CAP_BINDING` 문턱(R16-2) | 10% / 다른 값 | 10% [제안]. 2회차 감사(V2 B-8)에서 R16-2 문구 이상 없음. 값은 사용자 확정 전 [제안] 유지 — G0 PREREG 봉인 전 확정 |
| D-11 | 짧은 출력 ITL 규약(5-a) | P95 유지 + 공시 / 별도 술어 | [결정] 5-a PREREG에서 결정 |
| D-12 | G2 문서 집합(§10) | 모델별 필터 / 교집합 | [결정] 교집합 |
| D-13 | `fused_mixed` 선행 확인 | engine-porter에 맡김 / 생략 | [결정] **맡김**. (i)은 G0 계측 패치 gate Job에 동승(CK §4) |
| D-14 | vLLM 참고 열 | 넣지 않음 / 참고 열만 | [결정] 넣지 않음 |
| **D-15** | mixed-chunk 작은 예산(512/1024) | 추가 안 함 / 추가(3 → 5점) | **결정됨 2026-10-15: 추가**(초안 권고 "추가 안 함"과 반대). 예산 {512, 1024, 2048, 4096, 8192}, 합집합 11 config, 비대칭 11 대 3(S-2 주석), 작은 예산 prefill 몫 공시(R8-2c) |
| **D-16** | G\_mixed 동률 규칙(R8-2d, V2 死因 D-C′) | (a) 엄격 argmax / (b) 상수 띠 ≤ 1% + λ\*\_SLO 점추정 / (c) G\_mixed 기술값 강등, 1차 = G\_sym | **결정됨 2026-10-15: (a)**. 정확 동률만, 결정적 순서 비혼합 → 큰 chunk → 작은 decode-steps(`d_tuned`는 큰 D). λ\*\_thr 동률 기준 금지 |

## 14. 규칙층 감사

**1회차 판정 요약(2026-10-15)**: 명세 v0 **NO-GO**, 死因 3건(D-A λ\*\_SLO 격자 양자화, D-B 오류 셀 fail-open, D-C G\_mixed 분모 미정), 모두 N2. 질문 A-1·A-4·A-6·A-7b·A-8·A-10·A-12 REFUTED, A-2·A-3·A-5·A-7·A-9·A-11·A-13 PLAUSIBLE(조건부·부분).
전문·반전 시험 표·수리 목록·승계 caveat → `VERDICT_spec_v0_and_closure_2026-10-15.md`. 이 판의 수리 대응은 머리의 수리 색인 표.

**2회차 판정 요약(2026-10-15, 마지막)**: 명세 v0.1 **NO-GO → 사용자 결정 필요**, 死因 1건 D-C′(R8-2d 동률 띠 + λ\*\_thr 1차 기준이 D-C를 다시 엶, N2). D-A·D-B 닫힘. B-9: R7-2 완화가 게이트 8과 충돌(K7). 승계 caveat K1–K9. 전문 → `VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md`.
**감사 2회 완료(상한). D-C′는 사용자 결정 (a)로 해소. 이후 규칙 변경은 PREREG 단계 감사로.** v0.2의 R8-2d(D-16)·D-15·D-6 반영분은 G0 PREREG 감사에서 함께 본다(이 명세를 3회차 감사에 다시 올리지 않는다).

**2회차에 넘겼던 질문(이력; 답은 V2 §3)**:
- B-1 (R-1): §8 R9-1 보간 규칙이 D-A 반전 경로(오프셋만 다른 두 격자에서 확정/0으로 갈림)를 닫는가. `BELOW/ABOVE_LADDER`·무효 rung의 `UNRESOLVED`가 새 fail-open을 만들지 않는가. R9-1b dry-run이 봉인 전 단계로 충분히 고정됐는가.
- B-2 (R-2): R5-3b 임계표 값과 Holm 절차가 맞는가. 1차 가족 = G\_mixed 합집합판 + gate 불통과 시 G\_tuned 대체가 "데이터 보기 전 전환"으로 정당한가. multi-GPU 1차(G\_sym, F6)와 다른 점이 문장 충돌을 만드는가.
- B-3 (R-3): R7-3 서버/클라이언트 분류에 남은 fail-open이 있는가(예: 서버 과부하가 클라이언트 연결 실패로 보이는 경우 → 짝 무효화가 한 arm의 실패를 지우는가). 발송 지연 0.5 s [제안]와 "분류 불가 = 클라이언트 측" 규칙의 방향성.
- B-4 (R-4): R8-2d 합집합 최대·동률 규칙(λ\*\_thr → 비혼합 → 큰 chunk)이 D-C 반전을 닫는가. 동률 판정에 R9-6 해상도를 쓰는 것이 순환인가.
- B-5 (R-6·R-8·R-14): 혼합 실현 지표 분모, `CLIFF_ADJACENT` 식, λ\* 해상도 식이 각각 정의 가능하고 항등식이 아닌가.
- B-6 (R-9): TOST ±1.5%·짝 n ≥ 8 계열별·계측 최소 수준 정의가 A-12 결함을 닫는가.
- B-7 (R-10): P4가 lockstep-GPU와 CPU-launch를 실제로 가르는가(같은 층 수 묶음에서 I-3 token-layer 시간의 변동 폭이 충분한가). D-6 미결 상태에서 G0-S 봉인 보류가 다른 절을 막는가.
- B-8 (V §4·R-7·R-11·R-12·R-13·R-15·R-16): ABBA 배치, 합산 방식(R5-6), R7-2 재스코어 범위, R10-2, R16-2, Samuelson 근거, 경로 분해 호칭, R-16 문장이 판정서 문구와 어긋나지 않는가. S-1–S-8이 문자 그대로인가.
- B-9 (범위): 수리 색인 밖에서 판정 의미가 바뀐 곳이 있는가(v0 sha256 `cc582ff8…` 대비 diff).

## 15. 이 초안이 하지 않은 것

- rate 사다리 값, seed 목록, 셀 수, Job 하드캡·시간 예산(G0-S 포함, R13-2), GPU 사용률 계산, P4 "≈ 0" 동등성 여유·기울기 CI(K8, [미등록]), 혼합 실현 문턱, J\_eff ≥ 5 Holm 임계표, 튜닝 Job에서 λ\*\_SLO가 `UNRESOLVED`인 config의 argmax 후보 처리(R8-2d) → G0 PREREG(Q24)와 M3 개정(Q23)의 몫.
- Q15(§1-24 범위 재서술), Q17–Q21(비용 모델 항) → 이 명세는 계측·스윕 배치만 정했다.
- 종료 규약 수리(V §6 C-R1–C-R11)는 이 명세 밖이다(CLOSURE·DECISIONS·MERGE\_TRIAL 문서의 몫).
- 정본(PROJECT\_STATUS·CONSENSUS·COSTMODEL·MEMORY)·트랙 파일·다른 worktree는 수정하지 않았다.
