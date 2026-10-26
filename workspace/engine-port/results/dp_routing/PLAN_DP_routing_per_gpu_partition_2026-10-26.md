# 설계 초안 rev1 — 복제본별 SM 분할 + 길이 기반 라우팅 DP (2026-10-26)

> **지위**: 설계 초안(PREREG 아님, 미봉인). 측정 0, GPU 0, 성능 판정 0. 사용자 지시("SM 분할은 GPU별로 하면서 DP", 결정 35)와 multi-GPU 증거 검토 `reports/hymux_review/MULTIGPU_EVIDENCE_REVIEW_2026-10-26.md`에서 나왔다.
> **감사 이력**: 규칙층 r1 `AUDIT_dp_routing_r1_2026-10-26.md`(rev0 NO-GO: 死因 N1 Llama τ 항등식·결정 14·24 충돌·N3 τ = 2c\* 퇴화) → **rev1 = 수리 1–12 반영본**(수리 1·2는 결정 36). r2는 수리 diff만 본다. 수리 위치는 `[r1 수리 n]`.
> 수치 라벨: [코드] · [측정](출처 명시) · [유도] · [제안] · [가설] · [미정](사용자 결정).

## 0. 무엇을 묻는가 (그리고 묻지 않는가)
- 질문: **Nemotron-Nano-9B-v2를 GPU당 독립 서버 N개로 띄울 때, 요청을 길이로 나눠 보내고(라우팅) 복제본마다 그 요청 군에 맞는 정적 SM 분할을 두면 SLO goodput이 오르는가.**
- 근거 [가설]: hybrid는 길이에 따라 구속 자원이 바뀐다는 해석적 근거가 있다(c\* = s/k ≈ 8.8k에서 메모리, 2c\* ≈ 17.5k에서 decode 바이트의 지배 항이 state → KV, analytic-v1 [유도]). 다만 길이 기반 라우팅은 Transformer에서도 HoL·KV 단편화·batch 균질화로 이득을 낼 수 있으므로 **hybrid 고유성은 이 캠페인의 판정 대상이 아니다**(결정 14·24·36). [r1 수리 1]
- 단순 DP(`--dp-size`, round_robin)는 결과가 단일 GPU의 N배 재현이라(설계 감사 F-1) 질문이 되지 못한다 — 그래서 라우팅·분할 이질화를 변수로 둔다.
- 정적/동적 경계: 각 GPU의 분할은 정적이다. 라우팅은 SM 분할이 아닌 별개 레버라 정본 "단일 GPU 동적 제어는 best static을 못 넘음"과 모순은 없으나, **요청별 상태 의존 정책이므로 게이트 2·3을 적용하고 정본을 지지 근거로 인용하지 않는다.** [r1 MINOR 5c]
- 복제본 단위 비용 모델: 비용 항 PREREG T1–T4가 라우팅된 하위 혼합에서도 적용된다는 것은 [가설]이다(T1–T4 측정 0). [r1 MINOR 6g]

## 1. 선행 확인(GPU 0, 2026-10-26)
| 항목 | 결과 |
|---|---|
| `--dp-size`에서 GPU별 분할 | 된다 [코드]: dp rank마다 scheduler가 자기 GPU에서 PD-mux 초기화(`sglang_engine_dev_multigpu/python/sglang/srt/multiplex/multiplexing_mixin.py:154`). M2g J2 DP2 agnostic·d44 정확성 게이트 통과 + 분할 단계 노출(`results/multigpu/M2G_PASS_TABLE_J2.json`) — 정확성이지 성능 아님 |
| `--dp-size`에서 복제본별 다른 분할 | 안 된다 [코드]: `--pdmux-config-path`·`PDMUX_R2_FIXED_DSM`이 서버 전체에 하나 |
| 독립 서버 N개의 green context | 각 서버가 `CUDA_VISIBLE_DEVICES=i`로 device 0만 봐 N=1과 같은 코드 경로 [코드], `manual_divisions`는 auto grid GPU id 단언(`pdmux_context.py:112-118`)을 타지 않음. 동시 실측은 smoke |
| 노드 자원 | GPU당 11 CPU·119 GiB(×2 22/238, ×4 44/477, ×8 88/954 [측정, vesslctl resource spec]), ×8은 NUMA 2(GPU0–3 / GPU4–7). 클라이언트 CPU 고정 위치와 노드 전력 공변 공시 [r1 MINOR 4c] |
| 라우터 | 이미지에 `sglang_router` 없음, 엔진 내장 DP 분배는 `--dp-size` 안에서만(`data_parallel_controller.py:70-76`) → 클라이언트 측 라우팅(2-2) |
| M3 J2·J4·J8 9.5 h 정지 | ★정정(감사 r1 REFUTED): 2026-10-18 21:14:03Z ±40 ms에 함께 종료(`terminated`), 경과 9.511–9.515 h. Job별 정밀 타이머는 배제되지만 주기적으로 집행되는 9.5 h 경과 상한이나 계정 단위 일괄 종료와는 구분되지 않는다(계정 이력에 9.515 h를 넘긴 Job 없음). 원인 미확인, Job ≤ 6 h 유지. [r1 수리 12] |

## 2. 설계

### 2-1. 실행 형태
- N개 독립 SGLang 서버(GPU당 1개, 각자 config·env·cap·D), 한 Job·한 노드. 엔진 수정 0, 엔진 digest `99bbef91…` 유지. M3 C-7 승계 → 독립 서버만, 새 PREREG. M3 rev12(`--dp-size`) DP 자료와 결합·비교하지 않는다.
- 서버별 기록 = 비용 항 하네스 단일 GPU boot 스키마 재사용.

### 2-2. 라우팅 — 클라이언트 측
- `trace_loadgen.py`에 endpoint 목록과 라우팅 함수(요청 → 서버 index)를 넣는다. 결정은 송신 시점, 정보는 클라이언트 송신·수신 기록만.
- **LL 함수 [봉인 대상]**: 클라이언트가 송신했고 완료를 받지 않은 요청의 (입력 토큰 + max\_new\_tokens − 수신 토큰 수) 합이 최소인 서버, 동률은 index 작은 쪽. R1·R1′·R2·R3·R4가 이 함수 하나를 쓴다. 엔진 `TOTAL_TOKENS`(서버 보고 점유 토큰 + 대기 seqlen, `scheduler_metrics_mixin.py:866-883`)와는 다른 양이다. [r1 수리 6]
- **공시 [봉인]**: "라우터는 오버헤드 0이고, `ignore_eos` 강제 출력이라 요청의 실제 출력 길이를 안다(출력 길이 oracle, `trace_loadgen.py:64-66`). 실서비스 라우터의 이득은 이보다 작거나 같을 수 있다." 라우팅 결정 시간은 기록한다. [r1 수리 7]
- **클라이언트 포화 [제안, 하네스층에서 확정]**: 송신 지연(예정 대비 실제) p99 > 5 ms 또는 클라이언트 프로세스 CPU > 85%인 셀은 `FRONTEND_SATURATED`로 판정에서 빼고 건수를 공시한다. 각 boot의 한 rung에서 클라이언트 토큰 간격 p95와 엔진 `decode_forward` 간격 p95의 차가 3%를 넘으면 그 N의 ITL 술어 판정은 `CLIENT_ITL_UNVERIFIED`. 엔진 기록이 계측 OFF(결정 22)에서 남지 않으면 이 대조는 별도 계측 boot 1개로 한정한다. [r1 수리 8]

### 2-3. 분할 규칙 [봉인 대상]
- **τ = c\* = s/k** 단일 값(그 Job 부팅 로그의 s·k로 재계산, 토큰 수 반올림). 기준량 = 요청의 입력 + max\_new\_tokens. 2c\*는 민감도 기술값으로도 쓰지 않는다(자연 출력 arXiv-Sum에서 f\_long = 0 [유도, ARXSUM\_STATS: 유지분 5,906건 최대 입력+출력 16,744 < 17,520]). [r1 수리 3]
- **퇴화 검사**: 봉인 전 CPU에서 각 혼합 셀의 trace로 긴 쪽 비율 f\_long(τ)을 계산해 기록한다. 0.1 ≤ f\_long ≤ 0.9가 아닌 셀은 `ROUTING_DEGENERATE`로 R1′·R3·R4를 돌리지 않는다(결정 30 형식 공시). [r1 수리 3]
- **집합 크기**: 긴 쪽 복제본 수 n\_long = round(N × 긴 쪽 요청의 기대 서비스 비중)(비중 = 비용 모델의 요청당 예측 시간 합의 비), 1 ≤ n\_long ≤ N − 1로 클램프, 동률은 작은 쪽. N = 2에서는 1:1만 가능함을 공시. 집합 안에서는 LL. [r1 수리 4]
- **Llama 참조점**: Llama는 s = 0이라 c\* = 0이므로 τ를 대입하지 않는다 — **같은 trace에서 Nano 토큰 기준으로 정한 요청 ID 분할을 그대로 쓴다.** [r1 수리 1]

### 2-4. arm (같은 N, 같은 Job, 같은 seed 블록 안)
| arm | 각 GPU 엔진 | 라우팅 |
|---|---|---|
| R0 `fused_tuned·RR` | tuned fused(mixed-chunk, chunk 격자 사전등록, N=1에서 같은 규칙으로 선택 — G0 승계) | round_robin |
| R1 `fused_tuned·LL` | 같음 | LL |
| **R1′ `fused_tuned·LEN`** | 같음 | R3과 같은 분할 + 집합 안 LL [r1 수리 5] |
| R2 `pdmux_static·LL` | PD-mux 공통 static D̂(혼합 전체 기준 비용 모델 선택) | LL |
| R3 `pdmux_static·LEN` | PD-mux 공통 static D̂\_LEN(**LEN 분할 아래** 비용 모델이 고른 공통 D; D̂와 다르면 둘 다 기록) [r1 수리 5] | LEN |
| R4-D `pdmux_hetero_D·LEN` | 집합별 D(비용 모델 선택), cap은 R3과 같음 — **1차** [r1 수리 5] | LEN |
| R4-DC `pdmux_hetero_DC·LEN` | 집합별 D + cap(C\_mem) — 묶음 처치, 기술값 [r1 수리 5] | LEN |
- arm 순서: boot마다 정순·역순 교대(R0…R4-DC / R4-DC…R0). [r1 수리 10]
- Llama 참조점: R2·R3 각 1셀(같은 Job), 모델 간 goodput 비·이중차분은 기술값으로도 내지 않는다. [r1 수리 1]

### 2-5. D·cap 선택 규칙 [봉인 2단계, r1 수리 4]
- (i) **비용 항 Job fetch 전** 봉인: τ 규칙, 집합 크기 규칙, LL 함수, D̂·D̂\_LEN·집합별 D의 목적함수 = "decode floor를 지키는 D 중 prefill 처리율 최대", **decode floor 수치 정의** = 그 집합의 기대 (bs, 평균 ctx)에서 `T1(D, bs, ctx) × ρ(D, bs, L̄)` ≤ (1 − m)·θ\_ITL(m = 0.1, θ\_ITL = 50 ms [제안]) — 쓰는 항 T1·T3(ρ)·T4(cap), 동률은 큰 D, 격자 = 비용 항 T1의 D 격자. cap = 그 집합 길이 분포의 C\_mem 산식(T4).
- (ii) **fetch 뒤**: 계수로 계산한 수치(D̂, D̂\_LEN, 집합별 D·cap, n\_long)만 `DP_REG` addendum으로 봉인. (ii) 이후 규칙 변경 금지.

### 2-6. 지표
- **λ\*\_SLO** = 짧은·긴 두 클래스(τ 기준) **각각** attainment ≥ a를 동시에 만족하는 최대 rate(a = G0 PREREG 값 승계). 전체 attainment 기준 λ\*는 기술값. [r1 수리 9]
- 술어: TTFT ≤ SLO ∧ 요청 내부 ITL p95 ≤ SLO, SLO = 절대 A′(TTFT ≤ k·floor\_{N=1}(L), ITL p95 ≤ 50 ms). 조밀 사다리 직접 측정(사후 재채점 금지), 목표 양쪽 rung ≥ 2, 해상도 ≤ 3%.
- **분모** = 셀별 max(λ\*(R0), λ\*(R1), λ\*(R1′))(데이터 전 규칙). G\_SLO(arm) = λ\*\_SLO(arm) / 분모. [r1 수리 5]
- 분해: PD-mux 이득 R2/R1 · 라우팅 이득 R3/R2 · 분할 이질화 이득 R4-D/R3 · **라우팅 × 엔진 상호작용** = ln(R3/R2) − ln(R1′/R1), seed 짝 t(3) 95% CI. [r1 수리 5]
- cliff: TTFT가 SLO ± 5% 안에 드는 요청 비율 > 10%인 (arm, rung)은 `CLIFF_ADJACENT`, λ\* 교차가 그런 rung 사이면 그 셀의 비는 `G_UNRESOLVED`. [r1 수리 10]
- duration: 한 라운드 안 N 서버의 goodput = good 요청 수 합 ÷ 그 라운드의 **공통 벽시계** duration(서버별 duration 합산 금지 — N배 축소 함정), 라운드 간은 duration 합산(게이트 7). [r1 수리 10]
- 게이트 5: arm별 복제본 점유 분포, 복제본별·클래스별 TTFT/ITL p50/p95/p99, 라우팅 결정 분포. N 배율·GPU당 goodput은 기술값만(F-1). N 간 G\_SLO 비교는 Job·노드가 달라 기술값.

### 2-7. 워크로드·반복
- 길이 혼합(ShareGPT 계열 짧은 요청 + arXiv-Sum/shape C 긴 요청), 긴 요청 비율 2점 [제안: 30%, 60%] — 각 셀은 2-3 퇴화 검사 통과 필수.
- **변화 trace 채택(결정 36, 게이트 2)**: 1차 = rung별 정상 Poisson 사다리 λ\*\_SLO, 정책 비교 확인 = 긴 요청 비율 30% ↔ 60% 교대 trace 3라운드(rate는 정상 사다리 결과로 데이터 전 봉인 규칙 = 분모 arm λ\* × q, q ∈ {0.8} [제안]). [r1 수리 2]
- seed ≥ 4, 짝 CI 0 배제 ∧ ≥ 3%. 전 arm 계측 OFF(결정 22). GPU별 전력·클럭 기록.

### 2-8. Job·비용
- **Job 분할 단위 = seed 블록**: 각 Job은 그 seed의 모든 arm·혼합을 포함하고, Job은 짝 CI의 블록 인자다(한 비교의 arm이 다른 Job으로 갈리지 않는다). Job ≤ 6 h. [r1 수리 11]
- 비용 [유도]: rev0의 손 환산(200셀 × 175 s ≈ 9.7 h/config, 셀당 175 s의 출처는 추적되지 않음 — 감사 r1)은 폐기하고, 하네스 시간 모형 코드(`ct_timemodel.py` 계열)로 재산정한다. 상한 쪽에 M3 과소 예보 1.46×(J1 8.60 h 대 5.88 h)를 공시. arm 7개(R1′·R4-DC 추가)로 rev0 대비 약 +40% [유도], 변화 trace 3라운드 추가. 참고 하한: rev0 산술 ×1.4 ≈ 모델 하나 N {2, 4} 약 80–120 GPU-h(Llama 참조점 2셀은 소량).
- N = 8과 축소안은 결정 30 형식 표로 쓴다(사유 = 예산은 허용 사유 아님, 잃는 범위, 복구 비용) [미정, O-D3]. [r1 수리 11]

## 3. 순서·선행 조건
1. 비용 항 Job 봉인·제출·분석(D·cap·n\_long이 T1·T3·T4 적합에 의존). **그 fetch 전에 2-5 (i) 구조 봉인.**
2. G0 PREREG(tuned fused 선택 규칙·a·shape C) — 분모·a 승계.
3. 이 설계를 PREREG로 → 규칙층 감사 r2(최종) → 하네스(loadgen 다중 endpoint·LL·LEN·클래스별 attainment·N 서버 기동) → 하네스층 감사 → Workspace smoke(x2) → 봉인 → 제출 승인.

## 4. 열린 결정
- ~~O-D4 변화 trace~~ → 결정 36: 채택. 귀속 → 결정 36: 포기(a).
- O-D1 독립 서버(권고) / 엔진 패치. O-D2 클라이언트 측 라우팅(권고) / 별도 프록시. O-D3 축소안·N = 8(결정 30 형식). O-D5 원고 §7.8을 이 계획으로 교체.

## 5. 등록 caveat(인용 금지 형태)
1. "이 DP는 GPU당 독립 서버 N개와 클라이언트 측 라우팅이며, 라우터는 오버헤드 0이고 출력 길이를 안다."
2. "이 캠페인은 길이 라우팅 이득의 hybrid 고유성을 판정하지 않는다."
3. "G_SLO의 N 간 비교는 Job·노드가 달라 기술값이다."
4. "N 배율·GPU당 goodput은 복제 처리율의 재표현이라 PD-mux·라우팅 이득의 근거가 아니다."
5. "M3 rev12(`--dp-size`) DP 자료와 결합하지 않는다."
