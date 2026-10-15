# 공통 측정 명세 v0 — 단일 GPU·multi-GPU 공용 (2군 Q4–Q12 + 3군 Q13·Q14·Q16 명세분) (2026-10-15)

> **지위: DRAFT v0. claims-auditor 규칙층 감사 전. 정본이 아니다.** 측정 0, 새 성능 판정 0.
> 정본화 경로는 DISCUSSION_INDEX Q22: 2군 사용자 결정 → 규칙층 감사 → main `promote`. 트랙(G0·M3)은 링크만 건다.
> 출발점: `DISCUSSION_INDEX_2026-10-15.md`(이하 IDX) §1 2군·3군 표, §5. 약어: A = `AGENDA_methodology`, D = `DESIGN_span_and_integration`,
> K = `COSTMODEL_running_cap`, R = `REVIEW_handoff_prompt`, C = `COMPARE_span_sync_muxwise_bullet`(모두 같은 폴더),
> MG = `../prefill-layer-alloc-multigpu/workspace/engine-port/results/multigpu/PREREG_MULTIGPU_2026-10-12.md`(rev11, 읽기만),
> SLO-R = `reports/slo_definition_rationale_2026-10-04.md`(A′ 본문).
> 수치 라벨: **[측정]**(출처 명시) · **[유도]**(식·코드 대입, 측정 아님) · **[코드]**(소스 사실) · **[결정]**(사용자 결정) · **[제안]**(이 초안이 고른 값, 근거 약함 → 감사 대상).
> 이 문서의 수치는 전부 VESSL A100 SXM 80GB(108 SM) 기판 범위다. KISTI 시기 수치와 섞지 않는다.

## 0. 범위·전제·상태

**범위**: goodput·용량·SLO·arm·반복(n)·검정·기판 공변·계측을 정의한다. G0(단일 GPU, main)와 M3/M4(multi-GPU) PREREG가 이 명세를 인용한다.
rate 사다리 값·seed 목록·Job 하드캡·셀 수는 각 PREREG의 몫이다(이 명세는 그 값이 지켜야 할 조건만 정한다).

**확정 전제** [결정]: F2 주 모델 = Nemotron-Nano-9B-v2(multi-GPU는 `-Base` revision `dc0661c…`, MG §1) · F3 true dual worker = AMD 전용 ·
F4 lockstep PD-mux가 multi-GPU로 이양, 단일 GPU 결과의 범위는 "lockstep 기판" · F5 A′ · F6 multi-GPU arm 4종·1차 estimand G\_sym · F7 TP cap 48·N · F8 radix OFF.

**미결 전제(이 초안이 가정한 값 — 바뀌면 표시한 절을 다시 연다)**:

| 전제 | 가정값 | 바뀌면 영향받는 절 |
|---|---|---|
| Q2 포지션 | P-C(혼합) | §6 headline 분모 선택(P-A면 G\_sym 중심, P-B면 G\_tuned 필수), §10 D-3 |
| Q3 G0·G2 표준 셀 | arXiv-Sum 16k | §3 floor 측정점, §4 ITL 표본 수, §11 span 스윕 격자, §12 cap 구속 예상 |
| Q3 5-a | Mooncake Conv + OpenThoughts + ShareGPT | §1 실 trace rate 정의, §3 128k floor, §4 짧은 출력 규약 |
| Q1 N2 종료 | (a) 종료 + F-2를 §9 I-1로 흡수 | §9 I-1 선행 일정, §7 공변 검사의 크기 분리 범위 |

**공통 기판 규칙** (모든 절에 앞선다):
- 한 비교의 arm은 한 Job 안에 둔다(운영 모델 · #262). 다른 Job의 arm 값은 비교하지 않는다.
- 기판 gate는 MG §16-3과 같다: GPU 이름 정확히 "NVIDIA A100-SXM4-80GB", power limit 400 W, 기대 GPU 수 일치, 어긋나면 `ABORT_SUBSTRATE`.
- 엔진 트리 digest를 셀마다 기록한다. `multigpu-scaling` merge 전에는 G0와 M3 N=1 앵커의 바이너리가 달라 **직접 비교 금지**(D §3.2-6).
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
- multi-GPU 차이: M3는 고정 rate HI/LO로 이미 부합한다. M4의 변화 trace(ρ\_LO 0.30 ↔ ρ\_HI 0.85 × λ\*\_fused, 3라운드, MG §5)는 게이트 2 형식을 그대로 따르므로 그대로 둔다.
- 예상 비판 1: "실서비스는 bursty하다. 고정-rate Poisson은 PD-mux에 유리한 정상상태만 본다." → 대응: 범위 문장 의무화 + 5-a(실 trace)·M4(변화 trace)가 그 질문을 따로 다룬다. G0 결과는 운영 주장으로 쓰지 않는다.
- 예상 비판 2: "게이트 2는 static조차 ±1.3 흔들려서 생긴 규칙인데 고정 rate를 다시 허용하나?" → 대응: 그 분산 문제는 §2(Job ≥3 반복, Job 수준 판정)로 정면 처리한다. 고정 rate 허용은 분산 규칙과 묶여 있을 때만 유효하다.

## 2. Q5 — n의 층위와 판정 검정

**규칙 R5-1 (설계)**: 판정 셀은 **seed ≥4 × Job ≥3**. Job 안에서 seed마다 boot를 새로 한다(seed = boot, L2·L3 합쳐짐을 공시). arm 순서는 Job마다 등록 seed로 블록 무작위화한다.
**R5-2 (기록)**: Job마다 hostname, GPU UUID, 드라이버, `cpu.max`, 노드 종류를 기록한다. 두 Job이 같은 GPU UUID면 독립 Job으로 세지 않는다(`JOB_NOT_INDEPENDENT`).
**R5-3 (판정)**: 효과 = Job j 안 arm 짝 차분(또는 log 비)의 seed 평균 e\_j. 판정은 둘 다 만족할 때만:
(i) 모든 Job에서 e\_j 부호 일치, (ii) Job 수준 t-CI(자유도 J−1, 95%)가 0을 배제. 크기 문장은 추가로 |평균 효과| ≥ 3%(게이트 3).
**R5-4 (범위)**: Job이 1개뿐인 결과는 "이 Job에서"로만 쓴다. "VESSL A100에서"는 R5-3을 통과한 것만.
**R5-5 (공시)**: Job 안 seed 짝 bootstrap은 기술값이며, n=4면 서로 다른 재표집이 35개뿐임을 적는다(N3VR-11).

- 근거: A §2.1–2.4. Job 간 수준 이동 +4% [측정, E2 r1/r2, E2VR-3], 공통 격자 36/36 일방향 Job 효과 [측정, #262]. σ\_Job이 3% 문턱과 같은 크기라 Job 1개로는 효과와 Job을 가를 수 없다.
- 결정 규칙의 성질 [유도]: 귀무가설에서 "J개 Job 부호 전부 일치(방향 무관)" 확률 = 2·(1/2)^J → J=3에서 0.25, J=4에서 0.125. 그래서 부호 일치만으로는 판정이 안 되고 (ii)와 묶는다.
  t 임계(양측 95%) = 4.30(J=3), 3.18(J=4). J=3은 검정력이 낮다는 비용을 감수한다.
- 대안·기각: (a) 현행(Job 1, 4/4 부호) → Job 효과와 구분 불가. (c) seed 6개 단일 Job → 부호검정 최소 p 2/64는 닿지만 Job 층 변동을 전혀 안 잡는다. 둘 다 기각.
- multi-GPU 차이: M3는 config마다 Job 1개(J\_N)이고 seed가 같은 boot 안이다(MGP-7). 그래서 M3의 G\_sym 판정은 **Job 안 판정**이며, 이 명세 기준으로는 R5-4의 "이 Job에서" 범위다.
  M3의 N=1 앵커 + 3% Job 효과 검사(MG §15-3, §16-1)는 Job 효과의 **검출**이지 반복이 아니다. M3 개정 시(Q23) "N=1 셀만 Job ≥3"을 둘지는 사용자 결정(§13 D-2).
- 예상 비판 1: "Job 3개, 자유도 2의 t는 검정력이 거의 없다." → 대응: 맞다(감수 비용). J=4로 올리는 선택지를 비용과 함께 D-2에 둔다. 검정력 부족은 "효과 없음"으로 쓰지 않는다(판정 불가 라벨).
- 예상 비판 2: "VESSL이 같은 노드에 다시 배치하면 Job이 독립이 아니다." → 대응: R5-2의 UUID 규칙. 같은 노드·다른 GPU는 독립으로 세되 노드를 공변량으로 공시한다.

## 3. Q6 — A′의 혼합 길이 적용

**규칙 R6-1**: 요청별 TTFT 임계 T\_i = k·floor(L\_i). 결정점 k ∈ {3, 5, 10}(Holm 보정), k ∈ [1, 25] 곡선은 기술값. **곱셈형 유지**. T는 처치가 아니라 판독 눈금이다(SLO-R §3–4).
**R6-2 (floor 측정)**: 각 Job의 **첫 블록**에서, 등록 기준 config로 무경합 단일 요청 TTFT를 입력 길이 {1k, 2k, 4k, 8k, 12k, 16k} [제안] 각 16회 재고 중앙값을 낸다. 값은 초 단위로 동결한 뒤 같은 Job 셀을 채점한다.
**R6-3 (보간)**: 측정점 사이는 log–log 구간 선형 보간. 측정 범위 밖 외삽 금지. L < 1k 요청은 floor(1k)로 고정하고 그 비율을 공시한다
(arXiv-Sum Nano 토크나이저 유지분 중 L ≤ 2048은 5.0% [측정, ARXSUM_STATS], 1k 미만은 그보다 적다).
**R6-4 (기준 config)**: floor는 구성 의존이므로(SLO-R §5) **arm 하나의 config로 고정**해 모든 arm에 같은 T를 쓴다. 다른 arm의 floor는 기술값으로 같이 잰다. 기준 arm은 §13 D-1.
**R6-5 (병기)**: (가) 길이 3분위별 attainment, (나) 가산형 대조 T′\_i = floor(L\_i) + (k−1)·floor(L̄)(평균 여유가 같고 길이와 무관한 여유) 채점을 **기술값**으로 병기한다. 판정은 곱셈형만.

- 근거: A §3.2–3.3, SLO-R §3-6("혼합 길이면 요청별 k·floor(L\_i), 실측점 맞춤(Etalon), 혼합 길이 때 재검토"). N3 shape B floor 0.907 s → T 2.72/4.53/9.07 s [측정, A §3.2].
- 대안·기각: 고정 절대 SLO(4–15 s) → 길이가 5배 퍼지는 워크로드에서 긴 요청에 체계적으로 불리(R §4.1). 정규화 TTFT(ms/token) → 대기 시간이 길이와 무관한데 길이로 나눠 짧은 요청을 과벌. 가산형 판정 → α·β 두 자유도를 새로 정해야 해 자유 표면이 늘어남. 셋 다 판정용으로 기각, 절대 초 값은 환산 좌표로만 병기.
- multi-GPU 차이: §5(Q10). shape A/B는 단일 길이라 floor 1점(그 길이)이면 된다. arXiv-Sum shape를 M3에 넣으면(Q23) R6-2를 N=1 앵커 블록에서 잰다.
- 5-a 차이: Mooncake 꼬리(p99 85k, 최대 126k)는 측정점 {32k, 64k, 128k}이 추가로 필요하다. 128k 단일 요청 프로브(main inbox 결정)와 합친다.
- 예상 비판 1: "곱셈형은 긴 요청에 절대 여유를 더 줘서 PD-mux의 장문 HOL을 숨긴다." → 대응: R6-5 (가)(나)를 병기해 실패가 어느 길이 구간에서 오는지 보인다. 두 채점의 부호가 갈리면 그 자체를 결과로 쓴다.
- 예상 비판 2: "Job마다 floor가 달라 T가 Job마다 다르다." → 대응: 판정은 Job 안 arm 차분이라 같은 Job 안에서는 T가 같다. Job 간 floor 산포는 Job 효과 진단값으로 공시한다.

## 4. Q7 — ITL 술어 세부

**규칙 R7-1**: 주 술어 = 요청 내부 token-ITL(첫 토큰 제외)의 P95 ≤ 50 ms. P95 계산법을 등록한다(`numpy.percentile`, method `linear`) [제안].
mean-ITL 술어는 secondary이고 **같은 문장·표 칸에 섞지 않는다**(Granite 부호 반전 선례, R §4.1 게이트 4).
**R7-2**: G0·M3의 arm은 전부 SLO를 읽지 않는 static arm이다(fused·fused\_sym·agnostic stock 임계표·d44 fixed). 그래서 ITL 사다리 {40, 50, 60, 65} ms 재채점은 재스코어가 아니며 기술값으로 병기한다(SLO-R §3-5).
SLO를 입력으로 읽는 정책 arm이 들어오면 그 arm은 50 ms 고정, 재채점 금지.
**R7-3 (요청 처리 통일)**: 출력 길이는 데이터셋 값으로 고정(`ignore_eos`), 출력 < 2토큰 요청은 표본 추출 단계에서 제외하고 건수 공시.
오류 요청(HTTP 오류·abort·timeout)은 attainment에서 **불합격**으로 센다. 셀 오류율 > 1%면 셀 무효(`CLIENT_ERROR`, MG §4-3과 같은 문턱).
**R7-4 (표본 수 공시)**: 셀마다 요청당 ITL 표본 수 n\_ITL의 p5/p50을 적는다. 워크로드 n\_ITL 중앙값 < 20이면 "P95 ≈ 최댓값" 경고를 결과에 붙인다.
**R7-5 (구속 술어)**: 셀마다 실패를 {TTFT만, ITL만, 둘 다}로 분해해 적는다(SLO-R §5, MG §4-6 실패 분해와 같은 형식).

- 근거: A §3.1(`analyze.py:34-38`: ITL 빈 요청은 NaN → 불합격), A §3.3-2·3, SLO-R §3-5.
- 표본 수 [유도]: arXiv-Sum 출력 mean 203·p50 195 [측정, ARXSUM_STATS] → n\_ITL ≈ 194. Mooncake toolagent 출력 p50 30 → n\_ITL ≈ 29, LooGLE 출력 ~15 → ~14(A §3.3).
- 대안·기각: (a) N3식 "오류 > 0이면 셀 무효" → 장문 셀 하나의 timeout 하나로 셀 전체가 사라져 표본이 선택적으로 빠진다. (b) analyze.py식 "빈 ITL = 불합격만" → 대량 실패가 attainment에 섞여 하네스 문제와 시스템 문제를 못 가른다. 둘을 1% 문턱으로 합쳤다.
- multi-GPU 차이: 없음. MG의 기술값 "TTFT 3 s"만 §5 규칙으로 바꾼다.
- 예상 비판 1: "50 ms는 chat 관행값이다. 요약 사용자는 ITL을 신경 쓰지 않는다." → 대응: SLO-R §5 공시 그대로("관행값이지 사용자 기준 아님") + 사다리 병기. 또 lockstep 기판에서는 prefill 창 안 ITL이 span 시간으로 정해지므로(D §2.2–2.4) ITL 술어는 사실상 "토큰의 몇 %가 prefill 창에 걸리는가"를 잰다 — 이 해석을 결과에 명시한다.
- 예상 비판 2: "짧은 출력에서 P95는 최댓값이라 술어 성질이 워크로드마다 다르다." → 대응: R7-4 공시. 5-a에서 짧은 출력 워크로드의 별도 규약(예: 요청 내부 P95 대신 '정지 > T 토큰 수')은 5-a PREREG 결정(D-11).

## 5. Q10 — N 간 SLO

**규칙 R10-1**: multi-GPU 판정용 TTFT 임계 = **N=1 floor 기준 절대 T**: T\_i = k·floor\_{N=1}(L\_i)를 모든 N·mode에 같은 초 값으로 적용한다("같은 사용자 SLO").
**R10-2**: floor\_{N=1}은 각 J\_N 안 N=1 앵커 블록에서 잰다(같은 Job 원칙). J1 값이 기준이고, J\_N 앵커 floor가 J1과 3% 넘게 다르면 `JOB_EFFECT`(MG §15-3) 공시에 포함한다.
**R10-3**: 기술값으로 T\_i^(N) = k·floor\_N(L\_i)(N별 상대 SLO) 채점을 병기한다. ITL 50 ms는 N 무관 절대값.

- 근거: D §3.2-4(연구 감사 의견: N 간 비교에는 절대 SLO 필요), IDX Q10.
- 대안·기각: N별 상대 SLO를 판정에 쓰면 TP가 floor를 낮춘 이득이 눈금에 흡수되어 "사용자가 겪는 지연"으로 비교되지 않는다. 기각(기술값으로만).
- 단일 GPU 차이: N=1뿐이므로 R6이 곧 R10이다.
- MG 정합 항목: MG §4-6의 SLO 실패 분해(TTFT 3 s)와 M4 SLO(TTFT ≤ 3 s, MG §5)는 A′가 아니다. Q23 개정에서 이 절로 바꾼다. (참고 [유도]: shape B에서 3 s ≈ 3.3 × floor 0.907 s라 k=3 근처지만, 이것을 A′로 간주하지 않는다.)
- 예상 비판 1: "절대 T는 큰 N에 유리하다(floor가 낮아 여유가 커진다)." → 대응: 그것이 사용자 관점의 실제 이득이다. 이득 중 floor 감소분과 용량 증가분을 가르려면 R10-3 상대 SLO 결과를 같이 본다.
- 예상 비판 2: "N=1 floor를 앵커 GPU 하나에서 재면 그 GPU 편차가 모든 N의 눈금에 들어간다." → 대응: MG §4-6 GPU 편차 블록(J8)을 공시로 붙이고, `GPU_HETEROGENEOUS`면 눈금을 "GPU0 기준"으로 적는다.

## 6. Q8 — arm 집합, fused 베이스라인, 묶음 처치 공시

**규칙 R8-1 (공통 arm)**: 단일·multi GPU 공통 = {`fused`, `fused_sym`, `agnostic`, `d44`}. 정의는 MG §2를 **바이트 단위로** 따른다:
- `fused` = PD-mux off, 엔진 기본 스케줄(overlap ON, chunked prefill 8192) + `--disable-piecewise-cuda-graph`(MG rev7 사유. 단일 GPU 이미지에서도 같은 플래그를 쓰고 사유를 공시).
- `fused_sym` = PD-mux off + `--chunked-prefill-size -1 --disable-overlap-schedule`(PD-mux arm과 같은 스케줄 노브).
- `agnostic` = PD-mux on, R2 정책 없음, `pdmux_agnostic_manual.yml`(manual 등가표 [[74,34,0],[54,54,36]], divisor 36).
- `d44` = PD-mux on, `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`. sticky OFF(D-SEM [결정] 2026-10-04).
**R8-2 (단일 GPU 선택 arm)**: `fused_tuned(k)` = 등록된 유한 E-D 스윕에서 k마다 λ\*\_SLO(k)가 가장 큰 fused config. 스윕 후보 [제안]:
`chunked-prefill-size` {2048, 4096, 8192} × `num-continuous-decode-steps` {1, 4}(cap은 §12대로 48 고정이라 스윕하지 않음) = 6 config.
이 6개는 **SGLang 기본 스케줄러의 비혼합 구조**(prefill-chunk 배치와 decode 배치를 번갈아 실행, `enable_mixed_chunk` 기본 False [코드, `server_args.py:630`])의 노브 값만 바꾼 것이다. 바꾸지 않는 것: `max-running-requests` 48, radix off, overlap on, mem-fraction 0.82. 처음 안건(A §5-5)에 있던 `max-running-requests`·`prefill-max-requests`·schedule-conservativeness는 이 공간 밖이므로 미시험 노브로 공시한다.
**R8-2b (구조 축 — mixed-chunk, 2026-10-15 추가)** [제안]: 비혼합 튜닝은 Sarathi-Serve·vLLM V1의 핵심 구조(한 반복 안에서 decode와 prefill chunk를 토큰 예산으로 섞는 stall-free batching)를 대표하지 못한다. 그래서 같은 엔진 안의 구조 arm `fused_mixed` = `--enable-mixed-chunk` + `chunked-prefill-size` {2048, 4096, 8192}를 둔다(SGLang `mix_with_running` 경로 [코드, `scheduler.py:906, 2562`]). 튜닝 fused 공간 = {비혼합 6} ∪ {혼합 3} = 9 config.
선행 조건 2개: (i) Nano-9B-v2(mamba state)에서 혼합 배치의 출력 정확성 — engine-porter correctness gate. 인자 단계에서 hybrid 때문에 꺼지는 분기는 없지만 동작 검증은 0이다. (ii) SGLang mixed chunk의 토큰 예산 의미가 Sarathi(decode 우선 채움)와 같은지 코드 대조. 다르면 문장에서 "Sarathi식"을 쓰지 않고 "SGLang mixed-chunk"로만 쓴다.
vLLM 별도 arm은 판정에 쓰지 않는다(커널·cudagraph·메모리 관리가 달라 원인 분리 불가, 게이트 1의 연장). 넣더라도 참고 열(기술값)로만 둔다(D-14).
선택은 **별도 튜닝 Job**에서 하고, 판정 Job에서 선택된 config를 다시 잰다(승자의 저주 회피). 같은 SLO(A′, ITL 50 ms)로 튜닝한다(게이트 8).
대칭으로 PD-mux 쪽 튜닝 = static D ∈ {24, 34, 44} [제안] 중 k별 최선(`d_tuned(k)`). **양쪽 튜닝 config 수를 결과에 같이 적는다.**
**R8-3 (estimand와 문장 종류의 고정 대응)**:

| estimand | 정의 | 허용 문장 | 금지 문장 |
|---|---|---|---|
| S | λ\*\_fused\_sym / λ\*\_fused | "스케줄 노브(chunk −1, overlap off) 효과" | PD-mux 효과로 귀속 |
| **G\_sym**(1차, 단일·multi 공통) | λ\*\_agnostic / λ\*\_fused\_sym | "같은 스케줄 노브에서 PD-mux 기능(SM 분할 + split prefill + 입장 기계) 전체의 효과" | "SM 분할 효과", "PD 분리 효과"(MGP-29) |
| G\_def | λ\*\_agnostic / λ\*\_fused | "미조율 기본 fused 대비 PD-mux 모드(묶음 처치)" — 선행 연구 정렬용 | 원인 귀속, 조율 공간 소진(T2) |
| G\_tuned (단일 GPU, 선택) | λ\*\_d\_tuned(k) / λ\*\_fused\_tuned(k) | "등록된 유한 튜닝 공간 안에서 양쪽을 같은 SLO로 튜닝했을 때" | "fused를 어떻게 조율해도"(공간은 유한) |
| G\_mixed (단일 GPU, 선택, R8-2b) | λ\*\_d\_tuned(k) / λ\*\_fused\_mixed\_tuned(k) | "같은 엔진의 혼합 chunked prefill(SGLang mixed-chunk)을 튜닝한 것 대비" | "chunked prefill 일반 대비", "vLLM·Sarathi 대비"((ii) 대조 전) |

**R8-4 (3-플래그 공시 규약)**: arm을 비교하는 모든 표·그림에 열 5개를 의무로 둔다 — `pdmux`(on/off), `chunked_prefill_size`, `overlap_schedule`, `piecewise_cuda_graph`, `enable_mixed_chunk`.
fused → agnostic 비교에는 "묶음 처치(PROJECT\_STATUS 게이트 #13)" 꼬리표를 붙인다. "SM 분할" 문장은 E-time ≥ 0.5(MG §4-3, `PARTITION_NOT_EXERCISED` 아님)일 때도 "SM 분할 포함 기능 전체"까지만.
**R8-5 (R의 열린 질문 — 기본 fused vs SLO-튜닝 fused)**: (호출 메모의 "R §5 열린 질문 5"는 R §4.2 항목 5와 인계 프롬프트 §7-2에 해당한다.)
답: 둘 다 두되 **역할을 나눈다**. 기본 fused = 선행 연구·정본 P1-opint와의 정렬(G\_def). SLO-튜닝 fused = 공정성 문장(G\_tuned).
비혼합 G\_tuned만으로 쓸 수 있는 최대 문장은 "SGLang 기본 스케줄러(비혼합 chunked prefill)를 등록된 6개 config 안에서 튜닝한 것 대비"다. "chunked prefill 대비"는 G\_mixed가 있을 때만 쓴다.
"PD-mux가 fused보다 낫다"는 형태의 headline은 **G\_tuned가 있을 때만** 쓰고, 없으면 headline은 "미조율 기본 fused 대비"로 범위를 줄인다(정본 §1-1 문안과 같은 수준).

- 근거: A §5-5, D §3.1–3.2-2, R §1(정본: fused 미조율, T2 주장 금지, 미시험 노브 6개 이상), R §4.1 게이트 8·#13, MG §2·§4-4. 정본 허용 문장은 R §1 마지막 문단(2모델 운영점 수치, 기전 귀속 미확립).
- 대안·기각: (a) 기본 fused만 → 미조율 confound 재생산. (b) fused\_sym만 → 아무도 운영하지 않는 설정이라 대조군이지 베이스라인이 아니다. (d) 무한/적응적 튜닝 → 자유 표면이 끝없이 늘고 사전등록 불가.
- multi-GPU 차이: R8-2(튜닝 스윕)는 비용 때문에 multi-GPU에 넣지 않는다 [제안]. 그래서 multi-GPU 문장은 G\_sym·G\_def 범위이고 d44는 "1-GPU에서 고른 D44를 N에 그대로 쓴 값"(MGP-15)이다.
- 예상 비판 1: "fused\_sym은 허수아비 대조군이다." → 대응: headline 분모가 아니라 기전 분해용 대조군이다(S와 G\_sym으로 G\_def를 나눔: G\_def = S·G\_sym [유도, 항등식]). headline 분모는 G\_tuned 또는 G\_def.
- 예상 비판 2: "튜닝 공간 6개는 작다. 더 조율하면 fused가 이길 수도 있다." → 대응: 맞다(감수 비용). G\_tuned 문장에 공간을 명시하고 T2(조율 공간 소진)는 계속 금지한다. 공간 확장은 PREREG 개정으로만.

## 7. Q11 — 전력·클럭: arm × 창 cap 노출 공변 검사

**규칙 R11-1 (수집)**: 모든 Job에 GPU별 SM 클럭·메모리 클럭·throttle reason 비트(SW power cap, HW slowdown 포함)·전력을 수집한다. 주기: 전력 20 ms, 클럭 100 ms(N2 관행 [측정, A §1]).
수집이 0행이면(N2 F-2 `sm_clk_samples.csv` 0행 문제) 그 Job의 셀은 `SUBSTRATE_UNCHECKED` — 크기 인용 금지.
**R11-2 (검사, 판정 전 단계)**: 판정량의 **분자·분모 창마다**(#268) cap 노출률(SW power cap 비트가 켜진 시간 비율)과 SM 클럭 중앙값을 낸다. 비교하는 두 arm 사이에서
노출률 차 > 10 %p 또는 클럭 중앙값 차 > 1% [제안]이면 그 셀은 `SUBSTRATE_COVARIES`.
**R11-3 (처리)**: `SUBSTRATE_COVARIES` 셀은 부호만 인용하고, 크기 문장은 "클럭 하락 성분을 포함한 값"으로 범위를 붙인다. 기전(스케줄링 구조) 귀속 금지.
**R11-4**: 검사 코드와 문턱은 PREREG 봉인 전에 고정하고, 변이 시험(교훈 53: 창 정렬을 한 칸 밀거나 throttle 비트를 무시한 변이본이 합성 데이터에서 실패해야 함)을 하네스 회귀에 넣는다.

- 근거: A §5-8, #268(N2 Job-EA C 창 64/64 cap 노출 → ρ ≈ 1.16 크기 CONFOUNDED [측정, N2C-102]), λ0 cap 노출 A 27%·B 49% [측정], N3 창별 11.5–47.4% [측정].
- 대안·기각: 전력 고정(`-pl`) → VESSL 컨테이너 불가 [측정, power-cap 트랙]. 클럭 고정(`-lgc`) → rc=4 [측정, 2026-10-07 프로브]. 사후 분석만 → N2에서 결과 뒤에 발견된 실패 그대로. 그래서 "통제 불가 → 검출 + 범위 제한"을 고른다.
- multi-GPU 차이: GPU별로 잰다. TP는 rank 중 최대 노출률, DP는 복제본별 + 평균으로 검사한다 [제안]. MG §1 `clk.csv` index 열이 이미 GPU별이다.
- 예상 비판 1: "PD-mux가 GPU를 더 바쁘게 써서 클럭이 떨어진다면 그건 처치의 일부(매개변수)다." → 대응: 맞다. 그래서 R11-3은 셀을 버리지 않고 "클럭 효과 포함"으로 범위를 붙인다. 매개 성분 분리는 F-1(burner) 같은 별도 등록이 필요하다.
- 예상 비판 2: "10 %p·1% 문턱은 임의다." → 대응: [제안] 라벨 그대로 감사에 넘긴다(§14 A-5). 문턱 사다리 {5, 10, 20} %p에서 라벨이 바뀌는 셀 수를 기술값으로 병기하는 안도 있다.

## 8. Q9·Q12 — 용량 정의와 headline 위계

**규칙 R9-1 (두 용량 병기)**:
- **λ\*\_thr**(처리율 knee) = MG §4-2·4-3 방식 그대로: HI rung 4 seed 전부 g ≥ 0.10(`HI_SATURATED`) ∧ LO rung g ≤ 0.05(`LO_UNSATURATED`) → `KNEE_BRACKETED`, λ\* = HI achieved 중앙값. g는 클라이언트 측 Q\_c 시간 적분 평균의 3분할 기울기.
- **λ\*\_SLO(k)** = N3 첫 실패 규칙: attainment(r, k) ≥ 0.90인 등록 rate 중 첫 실패 rate 바로 아래. 정상성 S1–S3 적용. attainment 99%는 기술값 병기(SLO-R §3-5).
**R9-2 (cliff 경계)**: λ\*\_SLO 셀에서 attainment가 0.90 문턱과 **요청 1개 차 이내**이면 `CLIFF_ADJACENT` — 크기 인용 금지, 곡선만 보고(N3 k=3 셀이 이 경우였다 [측정, A §1]).
**R9-3 (사다리 조건)**: 용량 측정을 판정보다 먼저 한다(게이트 6). 판정 rate 사다리는 예상 λ\* 위·아래로 각각 2 rung 이상, 인접 rung 비 ≤ 1.2 [제안]. 사다리는 데이터를 보기 전에 등록한다.
**R9-4**: 두 용량에서 G를 각각 계산한다. 두 값의 부호·크기가 갈리면 그것을 결과로 보고한다(예: 처리율은 같고 꼬리만 개선).
**R12-1 (headline 위계)**: ① λ\*\_SLO(k=3/5/10) attainment 90% 용량 + λ\*\_thr 병기 → ② 고정 rate의 TTFT/ITL p50/p95/p99와 실패 분해(R7-5, 길이 3분위) → ③ CUDA event 기전 분해(§9 I-3·I-4). ③은 headline이 될 수 없다(게이트 1).
**R12-2**: 실행 전 roofline 예측(B\_max 등)을 기록하되, 비교 대상은 엔진 실효 상한(부팅 로그 KV 풀 토큰, mamba 슬롯 = cap)과 **별개 열**로 둔다(A §5-7, Q21).

- 근거: D §3.2-3, A §3.1·§5-6, MG §4-2·4-3, A §1(N3 첫 실패 규칙·cliff).
- 대안·기각: λ\*\_thr만 → 사용자 관점 지연이 없다. λ\*\_SLO만 → k에 따라 움직이고 N 간 비교가 어렵다. goodput req/s 곡선을 주 지표로 → 고정 rate에서는 goodput ≤ offered라 용량 정보가 없다(보조값으로만).
- multi-GPU 차이: M3는 λ\*\_thr가 판정이고 SLO는 기술값이다. 개정(Q23) 시 λ\*\_SLO를 **기술값으로** 추가할지, 판정으로 올릴지는 D-9. 이 초안은 "M3 판정 = λ\*\_thr 유지, λ\*\_SLO 기술값"을 권고(M3 규칙층 감사 11회차를 다시 열지 않기 위해).
- 예상 비판 1: "cap 48이 구속하면 λ\*\_thr가 cap 비율로 수렴해 G → 1로 퇴화한다(MG §8-2)." → 대응: §12 `CAP_BINDING` 공시 + I-5 거절 사유 계측으로 퇴화 여부를 셀마다 보인다.
- 예상 비판 2: "첫 실패 규칙은 rung 격자에 민감하다." → 대응: R9-3 사전 격자 + attainment-vs-rate 곡선과 bootstrap CI를 함께 보고한다.

## 9. 계측 필수 목록 (Q11·Q14) — 단일·multi GPU 공용

★ 표시 = **엔진 패치 → engine-porter correctness gate 선행**. 나머지는 하네스 수정(하네스 회귀 테스트 선행).

| ID | 항목 | 내용 | 용도 | G0 전 필수? |
|---|---|---|---|---|
| I-1 | 클럭·전력·throttle 수집 | R11-1. F-2 0행 문제 수리, 셀 시작 전 수집 생존 검사 | R11 공변 검사 | **필수** |
| I-2 | clk ↔ telemetry 시간 정렬 ★ | telemetry 시작·끝·매 N초에 (CLOCK\_MONOTONIC, CLOCK\_REALTIME) 쌍 기록. 정렬 오차 상한 공시 | R11-2 창 정렬, F-1 | **필수** |
| I-3 | span GPU 시간 ★ | prefill stream CUDA event로 span 시작·끝, 층 범위 [split\_index, next), 배치 토큰 | S\_prefill·정지 분해, §11 예측 검증 | **필수** |
| I-4 | decode 커널 GPU 시간 ★ | decode stream CUDA event로 decode forward만. 기존 `t_launch→t_sync`(prefill launch 섞임)와 별도 필드 | CM-1 vs CM-4 분리 | **필수** |
| I-5 | admission 거절 사유 ★ | 스케줄 결정마다 사유 ∈ {KV, cap(running = max), mamba 슬롯, prefill 자리 점유, `max_prefill_tokens`, R2 latch}, retract 이벤트 | §12 cap 구속, X-6 | **필수** |
| I-6 | cudagraph 패딩 ★ | decode iteration마다 bs\_real, bs\_padded, graph hit / eager | ITL 분해 | 선택(G0 안 권고) |
| I-7 | iteration Σctx·ctx p95 ★ | 32 sync 표본 → iteration 단위 | CM-1 ctx 항 | 후속 |

- 근거: A §4.4(우선순위 1–7), D §2.4-4, K §3-3, IDX Q14.
- **번호 모호성 공시**: IDX Q14의 "G0 전에 1·2·5번"은 A §4.4 번호(1 span, 2 decode, 5 admission)로 읽었다. IDX 표의 나열 순서로 읽으면 5번 = clk 정렬이 된다. 이 초안은 둘의 합집합(I-2·I-3·I-4·I-5)을 필수로 권고한다(D-5).
- **correctness gate 조건 [제안]**(★ 항목 공통): (a) 계측 ON/OFF 출력 일치 — greedy, 등록 prompt 64개, MG G1과 같은 m·q 지표 (b) 관측자 효과: 등록 rung에서 λ\*\_thr 차 ≤ 1%, 초과면 `OBSERVER_EFFECT`
  (c) 변이 시험: I-3을 decode stream에 기록한 변이본, I-4가 host 시간을 쓰는 변이본이 "격리 prefill에서 Σspan GPU 시간 ≈ prefill 벽시계 ±5%" 검사에서 실패해야 함 (d) 텔레메트리 스키마 버전 증가.
- 관측자 효과 블록: 판정 Job마다 한 arm(agnostic)을 계측 최소 수준으로 한 번 더 재서 (b)를 Job 안에서 확인한다(holb\_probe G5 관측자 효과 미결정 선례, D §1.2).
- multi-GPU 차이: 모든 항목 rank 접미사. TP는 rank 0 판정 + 전 rank 기록, DP는 복제본별. 패치는 merge 뒤 공통 트리 하나에만 넣는다(merge 전 이중 패치 금지 [제안] — 두 트리의 계측 구현이 갈리면 공용 명세가 깨진다).
- 예상 비판 1: "CUDA event 계측 자체가 lockstep 동기화를 바꿔 측정 대상을 교란한다." → 대응: event record는 비동기지만 query/sync 위치가 바뀌면 교란이 생긴다. gate (b)와 Job 안 관측자 블록으로 검출한다.
- 예상 비판 2: "계측 패치가 G0를 늦춘다." → 대응: 맞다(감수 비용). I-3·I-4 없이는 G0 결과의 기전(정지 vs decode 감속)을 가를 수 없고, 같은 질문을 위해 Job을 다시 돌려야 한다.

## 10. arXiv-Sum 표준 셀 (Q3 전제에 묶인 부분)

- 데이터: `ccdv/arxiv-summarization` document/test, sha256 `b34529df…` [측정, ARXSUM_STATS]. 필터 = Nano-9B-v2 토크나이저 입력 ≤ 16,384 → 유지 5,906/6,440 = 91.7% [측정].
- 유지분 입력 mean 7,116 · p50 6,484 · p90 12,369 · p99 15,722, 출력 mean 203 · p50 195 [측정]. 유지분 중 > 8,760 토큰(KV = state 경계) 30.0% [측정 비율, 경계값은 유도].
- 표본 추출: seed별 비복원 추출, 도착 Poisson(R4-1), 출력 길이 = abstract 길이(R7-3). `max_prefill_tokens` 값을 PREREG에 명시 등록(기본 16384, MG 개정 대기 항목).
- 모델 간 비교(G2)에는 각 모델 토크나이저로 같은 문서 집합을 다시 거른다. 길이 차이 ≤ 3% [측정, R §2.1]이지만 유지 문서 집합 자체가 달라지므로 문서 ID 교집합 사용을 PREREG에서 정한다(D-12).

## 11. span 예산 스윕(Q13)을 G0에 넣는 방식

**규칙 R13-1 (지위)**: span 예산 스윕은 **lockstep 기판 특성화**(F4)이고 정책·arm 판정이 아니다. headline arm은 upstream 기본값 `split_forward_token_budget = 65536`에 고정한다.
예산을 바꾼 PD-mux를 headline에 쓰면 PD-mux 쪽만 튜닝한 것이 되므로 금지(대칭 튜닝은 R8-2 공간에 넣을 때만).
**R13-2 (배치)**: G0 각 Job 끝에 블록 G0-S로 둔다(Job ≥3 반복을 그대로 받음). arm = `d44`(분할 고정이라 해석이 깨끗함), rate = LO(0.6·λ\*\_thr,d44)와 MID(0.85·λ\*\_thr,d44) [제안], seed 4.
**R13-3 (예산 격자 — 원안 수정 권고)**: 원안 {8k, 16k, 65k}는 arXiv-Sum에서 **일부 퇴화한다** [코드+유도].
- 층 수 = `max(1, budget // extend_num_tokens)` [코드, `multiplexing_mixin.py:1416-1421`]. 배치 토큰이 예산 이상이면 1층/span으로 바닥.
- `max_prefill_tokens` 16384에서 흔한 배치(요청 ~2개, ≈14k 토큰 [유도])는 예산 8k·16k 모두 1층/span → **두 점이 같은 처치**다. 차이는 배치 < 8,192 토큰일 때만 난다.
- 14k 배치 기준 span 일 [유도]: 16k → 1층 ≈14k token-layer, 32k → 2층 ≈28k, 65k → 4층 ≈57k.
- 권고 격자 = **{16384, 32768, 65536}** (D-6). 8k를 유지하려면 `max_prefill_tokens`도 함께 낮춰야 하는데, 그러면 두 노브가 섞인다.
**R13-4 (등록 예측)**: (P1) prefill 창 안 decode 토큰의 ITL 중앙값은 예산(span 일)에 단조 증가, (P2) TTFT p50은 예산 감소에 단조 증가(반복 수 증가), (P3) I-3 span GPU 시간 ∝ span 일(기울기 기술값).
P1이 깨지면 D §2.2의 "host 동기화로 decode가 span을 기다린다" 가설을 기각 방향으로 기록한다. 이 결과로 §1-24 문안을 고치는 것은 Q15(claims-auditor 회부) 경로로만.
- 근거: D §2.3–2.4, C §4-2(a), K §0(lockstep 정지는 특성화 대상).
- multi-GPU 차이: M3 개정 시 예산을 등록값으로 적기만 한다(65536). 스윕은 단일 GPU 깊이 항목(D §3.2-1).
- 예상 비판 1: "예산은 PD-mux의 튜닝 노브인데 기본값에 묶으면 PD-mux를 불리하게 둔다." → 대응: R13-1처럼 headline은 upstream 기본값으로 정의된 시스템이다. 예산 튜닝 이득은 G0-S에서 특성화로 보이고, 튜닝 PD-mux vs 튜닝 fused는 R8-2 대칭 공간에서만.
- 예상 비판 2: "LO·MID 두 rate에서만 재면 예산이 용량을 바꾸는지 모른다." → 대응: 맞다. G0-S는 ITL/TTFT 분포의 기전 특성화이고 λ\* 판정이 아니다. 용량 영향은 별도 등록.

## 12. cap 처리(Q16)를 G0에 넣는 방식

**규칙 R16-1**: G0·M3의 cap = 48(단일·N=1), TP 48·N, DP 복제본당 48 [결정 F7]. `max_mamba_cache_size` = cap을 명시 인자로 준다(§1-23). 부팅 로그의 `max_total_num_tokens`·mamba 슬롯 수를 셀마다 기록.
**R16-2 (구속 공시)**: 판정 셀마다 running = cap 시간 비율, I-5 거절 사유별 건수를 적는다. running = cap 비율 ≥ 10% [제안] 또는 cap 사유 거절 > 0이면 `CAP_BINDING` — 그 셀의 λ\*·G는 "cap 48 조건부"로만 쓴다.
**R16-3 (cap 스윕은 G0 밖)**: {48, C\_mem/2, C\_mem} 스윕은 비용 모델 검증 Job(Q17–Q19)의 몫이다. C\_mem은 roofline 값(arXiv-Sum 평균 길이에서 ~183 [유도, K §2])이 아니라 **부팅 로그 실측**(그 cap에서 KV 풀과 슬롯)으로 정한다. cap을 바꾸면 cudagraph 캡처 bs 목록도 바뀌므로(K §1) I-6을 켠다.

- 근거: K §1–4, MG §1·§8-2. cap 48은 근거 없이 상속된 값 [코드/이력, K §1]. 수요 하한 C ≥ λ·E[out]·E[ITL]이 lockstep 정지 수준 ITL(~200 ms)에서 ~60 > 48 [유도, K §2(3)] → arXiv-Sum G0에서 cap 구속 가능성이 있다.
- 대안·기각: (b) 워크로드별 창에서 도출한 cap 등록 → 아직 검증 안 된 비용 모델 출력을 처치값으로 쓰는 셈(게이트 1). (c) G0에서 스윕 → arm 수 × cap 수로 Job이 커지고 G0 질문(PD-mux vs fused)이 흐려진다. 둘 다 G0에서는 기각.
- 예상 비판 1: "cap 48은 PD-mux에 불리/유리할 수 있다." → 대응: 모든 arm이 같은 cap이다(MG §8-2). 구속 여부를 셀마다 공시하고, 구속 셀은 조건부로만 인용한다.
- 예상 비판 2: "hybrid에서 cap은 state 예약(48 × 0.1435 GB ≈ 6.9 GB [유도])까지 정하므로 사실상 처치다." → 대응: 그래서 cap 스윕을 비용 모델 검증의 독립 Job으로 둔다(Q18 anchor 주장 후보).

---

## 13. 사용자 결정 항목 (2026-10-15 사용자 결정: D-1…D-14 전부 권고안 채택. Q2 = P-C, Q3 = arXiv-Sum 16k 표준 셀 전제 확정)

| # | 항목 | 선택지 | 권고 |
|---|---|---|---|
| D-1 | floor 기준 config(R6-4) | fused\_sym / fused / PD-mux 무분할(N3 방식) | **fused\_sym**: un-chunked·108 SM·PD-mux 기계 없음이라 arm 중립. N3 방식과의 차이는 기술값으로 병기 |
| D-2 | Job 반복 수(R5-1)와 M3 적용 | J=3 / J=4 / M3 N=1 셀만 J≥3 | G0는 **J=3**(GPU 사용률 계산을 G0 PREREG에 제시), J=4는 첫 G0 결과가 판정 불가일 때. M3는 Job 안 판정 범위 유지 + 공시 |
| D-3 | fused 튜닝 스윕(R8-2)과 headline 분모 | 스윕 없음(G\_def 범위) / 비혼합 스윕 + G\_tuned / 비혼합+혼합 스윕 + G\_mixed | P-C 전제에서 **비혼합+혼합(9 config) + G\_mixed**. 혼합을 빼면 headline은 "SGLang 기본(비혼합) 튜닝 대비"까지, 스윕을 안 하면 "미조율 기본 fused 대비"까지 |
| D-4 | 공변 문턱(R11-2) | 10 %p·1% / 문턱 사다리 병기 | 10 %p·1% + {5, 10, 20} %p 사다리 기술값 |
| D-5 | G0 전 계측 범위(§9) | A §4.4 1·2·5 / IDX 나열 1·2·5 / 합집합 | **합집합(I-1–I-5)** |
| D-6 | span 예산 격자(R13-3) | {8k, 16k, 65k} / {16k, 32k, 65k} | **{16k, 32k, 65k}** |
| D-7 | 오류 요청 처리(R7-3) | 오류 > 0 무효 / 1% 문턱 / 불합격만 | **1% 문턱 + 불합격 계산**(MG와 통일) |
| D-8 | L < 1k 처리(R6-3) | floor(1k) 고정 / 512 측정점 추가 | floor(1k) 고정 + 비율 공시 |
| D-9 | M3 용량 판정(§8) | λ\*\_thr 판정 유지 / λ\*\_SLO 승격 | **λ\*\_thr 유지, λ\*\_SLO 기술값 추가** |
| D-10 | `CAP_BINDING` 문턱(R16-2) | 10% / 다른 값 | 10% [제안], 감사 후 확정 |
| D-11 | 짧은 출력 워크로드 ITL 규약(5-a) | P95 유지 + 공시 / 별도 술어 | 5-a PREREG에서 결정, 이 명세는 공시 의무만 |
| D-12 | G2 문서 집합(§10) | 모델별 필터 / 교집합 | 교집합(토크나이저 교락 제거) |
| D-13 | `fused_mixed` 선행 확인(R8-2b (i)(ii)) | engine-porter에 맡김 / 생략(G\_mixed 포기) | **맡김**. (i)은 GPU 확인이라 G0 계측 패치의 correctness gate Job에 같이 싣는다 |
| D-14 | vLLM 참고 열 | 넣지 않음 / 참고 열(기술값)만 | 넣지 않음 [제안]. venue 요구가 확인되면 G0 뒤 별도 Job |

## 14. claims-auditor 규칙층 감사에 넘길 질문

- A-1: R5-3(Job 부호 일치 ∧ t-CI, J=3)의 1종·2종 오류를 공시할 수 있는가. 부호 일치 조건이 t-CI에 더하는 것이 있는가, 아니면 장식인가(교훈 9: 게이트가 항등식일 수 있다).
- A-2: R6-4 floor 기준을 fused\_sym으로 두면 PD-mux arm에 유리·불리한 방향이 생기는가(PD-mux 무경합 TTFT가 decode-empty fallback으로 108 SM을 쓰는지 실현 확인 필요).
- A-3: R6-5 가산형 대조 T′(β = (k−1)·floor(L̄))가 새 자유 표면인가. 판정에 쓰지 않는 기술값이라도 결과 서술에서 판정처럼 쓰일 위험은?
- A-4: R7-3 "오류 = 불합격 + 1% 초과 무효"가 arm 비대칭 오류(예: fused만 timeout)에서 fail-open이 되는 경로가 있는가.
- A-5: R11-2 문턱(10 %p, 1%)과 "부호만 인용" 처리가 #268을 충족하는가. 매개(PD-mux가 전력을 더 씀)와 교란을 이 규칙이 구분하지 못하는 점을 어떻게 서술해야 하는가.
- A-6: R8-3의 G\_def = S·G\_sym 항등식을 근거로 "기전 분해"라고 부르는 것이 과장인가(PROJECT_STATUS 게이트 #269: 같은 관측의 재진술을 두 번째 증거로 세지 않기).
- A-7: R8-2 튜닝 공간(fused 6 config vs d 3 config)의 비대칭이 G\_tuned를 한쪽으로 기울이는가. 튜닝 Job → 판정 Job 재측정이 승자의 저주를 충분히 없애는가.
- A-7b: R8-2b 혼합 arm 추가로 fused 튜닝 공간이 9 config가 되어 PD-mux 쪽(3 config)과의 비대칭이 커진다. 이것이 G\_mixed를 PD-mux에 불리하게만 기울이는가(보수적 방향이면 허용 가능한가).
- A-8: R9-2 `CLIFF_ADJACENT`(요청 1개 차) 정의가 셀 요청 수에 따라 의미가 달라지는가 — 비율 기준이 필요한가.
- A-9: R10-1 절대 SLO에서 J\_N 앵커 floor를 쓰면 "같은 사용자 SLO"가 Job마다 달라진다. J1 floor 하나로 고정하는 편이 나은가(같은 Job 원칙과의 충돌).
- A-10: R13-3 퇴화 지적(8k ≡ 16k on 14k 배치)이 맞는가, 그리고 {16k, 32k, 65k}의 예측 P1–P3가 lockstep 가설과 대안 가설(CPU launch 지배)을 실제로 가르는가.
- A-11: R16-2 `CAP_BINDING`이 λ\*\_thr의 G → 1 퇴화(MG §8-2)를 검출하는 데 충분한가, 아니면 I-5 없이는 항등식인가.
- A-12: §9 correctness gate (b) 관측자 문턱 1%가 3% headline 문턱과 비교해 충분히 작은가, Job 안 관측자 블록 1회로 검출력이 있는가.
- A-13: 이 명세가 MG rev11 규칙(MGP-0…56′)과 충돌하는 지점 전수 — 특히 SLO(TTFT 3 s → A′ 절대 T), λ\*\_SLO 추가, 계측 스키마 변경이 M3 봉인 digest에 주는 영향.

## 15. 이 초안이 하지 않은 것

- rate 사다리 값, seed 목록, 셀 수, Job 하드캡, GPU 사용률 계산 → G0 PREREG(Q24)와 M3 개정(Q23)의 몫.
- Q15(§1-24 "고칠 수 없다" 범위 재서술), Q17–Q21(비용 모델 항) → 이 명세는 계측·스윕 배치만 정했다.
- 정본(PROJECT\_STATUS·CONSENSUS·COSTMODEL·MEMORY)·트랙 파일·다른 worktree는 수정하지 않았다.
