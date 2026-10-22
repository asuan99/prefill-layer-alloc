# 사전등록 rev3 (MEAS) — hybrid 서빙 비용 모델의 항 측정(T1–T4), 2026-10-20

> **봉인 상태: 미봉인 · 감사 이력: 규칙층 2회(상한), 항 측정 부분 死因 0 · 이 문서는 감사 대상이 아니라 결정 17에 따라 봉인 후보.**
> 지위: `PREREG_COSTMODEL_TERMS_rev2_2026-10-20.md`에서 **항 측정(T1–T4, J-A1·J-A2·J-B·J-C, TP2 항, 계측, 선행 조건, 금지 문안)만** 옮기고 2회차 감사(`VERDICT_prereg_costmodel_rev2_rules_r2_2026-10-20.md`)의 [미등록] 체크리스트 중 항 측정 해당분을 채운 봉인용 문서다. 문서 작업(CPU)만 했다. 제출·GPU·커밋 없음. 측정 0, 성능 판정 0.
> **V 한 줄**: V는 항 데이터 이후 별도 PREREG(새 감사)로 다룬다. 데이터를 본 뒤 구조를 고르는 위험은 그 PREREG에서 공시한다(결정 17).
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·모형 예보, 측정 아님) · [코드] · [결정](사용자 결정 문서 명시) · [제안](이 문서가 고른 값, 봉인 시 사용자 확인) · **[봉인 시 기입]**(봉인 순간에만 알 수 있는 값: digest·경로·gate 통과값).
> 근거: 계획 `PLAN_costmodel_terms_2026-10-20.md` · 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 1–17 · 감사 1회차 `VERDICT_prereg_costmodel_rev1_rules_r1_2026-10-20.md`·2회차 `VERDICT_prereg_costmodel_rev2_rules_r2_2026-10-20.md` · 프로브 `probe_128k/RESULT_probe_128k_2026-10-20.md`·`AUDIT_probe_128k_2026-10-20.md`(허용 문안 1–7)·`ANALYSIS_triton_96k_failure_2026-10-20.md` · J1 감사 `../multigpu_verdicts/AUDIT_J1_N1_analysis_2026-10-19.md` · `../anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md` · 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(이하 SP) · 항 정의 `COSTMODEL.md` CM-1…7 · gate 설계 `workspace/engine-port/results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md`(worktree `prefill-layer-alloc-engsync`; main 경로는 병합 후 [봉인 시 기입]).
> 기판: VESSL A100 SXM 80GB(108 SM) 1장(TP2 Job은 2장), SGLang v0.5.10, Nemotron-Nano-9B-v2-Base(rev `dc0661c…`), cudagraph ON, radix OFF, **`--mem-fraction-static` 0.6**(결정 13). 0.82로 잰 결과(M3 J1, 프로브 1회차)와 섞지 않는다.
> **ENGINE\_DIGEST** = 엔진 묶음 worktree `prefill-layer-alloc-engsync` 브랜치 `engine-sync-i5-int64`(`34f2acc8`, 30항목, **예상 `99bbef91…`** — 커밋 바이트 기준 예상일 뿐 gate 통과값이 아니다). 통과값 = **[봉인 시 기입 — G-α·G-β PASS(engsync, 2026-10-20) + G-γ는 결정 22로 "관측자 효과 특성화 완료" 종결; main 병합 후 같은 digest 재확인]**.
> **rev3.1 개정(2026-10-21, 미봉인 문서 갱신)**: 결정 18(O-항목)을 §10·§7에, 결정 22(계측 정책)를 §2-2′·§8-1·§9-2에 반영하고 §12 봉인 체크리스트를 신설했다. 측정 0, 성능 판정 0. 29항목 digest `2b1ab054…`는 폐기(배포·봉인된 적 없음). gate 안의 `ABORT_ENGINE_DIGEST`가 예상값 불일치를 막는다.
> **rev3.2 개정(2026-10-21, 미봉인)**: 결정 24로 H의 주 근거를 §4′ 같은 골격 조성 스윕(J-H)으로 바꾸고, §4의 Llama 대비는 J-H 안의 참조점 1 boot로 축소했다(J-B1 폐지). §7-4·§10·§12 갱신. 측정 0.
> **rev3.3 개정(2026-10-21, 미봉인)**: 결정 25로 J-H에 동거 간섭(T3)의 조성 의존을 추가하고 J-H를 J-H1(vanilla)·J-H2(PD-mux)로 나눴다. PD-mux boot의 D를 64→44로 바꿔 T3 격자와 공유한다. 측정 0.
> **rev3.4 개정(2026-10-22, 미봉인)**: 결정 28(`../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 끝)을 반영했다 — **A-1** T1 창을 ramp 인식 방식으로(셀별 출력 길이 O = 332 + ramp 상한, `bs_real`이 등록 bs에 처음 도달한 기록 기준 창, 창 뒤 요청 취소, 적합 입력 = 창 안 실제 평균 ctx) §1·§1-1·§3-1·§4′; **A-2** T2는 P=108 boot(`108u`·`108v`)에서만, sticky-D boot의 P 축은 T2′(bs=1 decode 동거)를 주 측정으로 승격하고 L을 T2와 같게 확장 §1-2·§1-8; **A-6** T2-ext 삭제; **A-11** J-H T4 거절 사유 `NOT_COLLECTED` 수용; **A-3…A-26** 하네스 해석 명시 §11-3. 시간·GPU-h 재계산 §7-4, 금지 문안 §9-2, 체크리스트 §12. 측정 0, 판정 0. 수치는 모두 [유도](시간 모형·사전값) 또는 가짜 엔진 값(인용 불가)이다.

## R. rev2 → rev3 변경 대응표

| 대상 | 처리 |
|---|---|
| 빠진 절(V 관련) | rev2 §R(수리표·V 항목), §1-6 T5 조립식, §6 전체(6-1 예측기·6-2 셀/arm/cap_V/사다리·6-3 판정표·6-4 V 문장·V-diag·6-5·6-6·6-7 J-V 앵커·6-8 귀무 N0/N1), §7의 J-V1·J-V2a·J-V2b 행, §8의 V\_REG(S2)·R8-2′, §9의 V 라벨·V 금지 문안, §10의 O-5·O-7·O-9(V 부분)·O-11, §11의 V 쟁점 4·14·1·2·3·6·10. **V 산출물을 위해 남긴 항 측정 산출물**은 유지(T3 σ·q\_stall, T1 (64,8k)·(64,12k)·16k 셀, `bs_padded`, I-4·I-6 필드, T2′, 외삽 시험 수준) |
| 2회차 체크리스트 채움 | 아래 표 |
| 등록 caveat | 3문장 중 1번만 옮김. 2번(V 10%)·3번(상위 2 D)은 V 전용이라 V PREREG로 이월 |

2회차 체크리스트 처리(항 측정 해당분):

| 항목 | 상태 | 위치 |
|---|---|---|
| "KV 먼저 구속" 문장 정정, 83 | 채움(문장 폐기, 83 정정, 60 대 83 해소) | §3-1 |
| n\_A 형식 | 채움(A도 B와 같은 mamba 정적 예약 형식으로 통일; 64k cap 48 칸 27→**24**) | §1-4 |
| I-4·I-6 필드 이름 | 채움(`decode_fwd_gpu_ms`, `bs_real`/`bs_padded`/`graph` 등; engsync 코드 확인) | §3-4 |
| σ 엔진 필드, T\_span\_iter, SPAN\_TOKENS | 채움(`prefill_span_start/end` 쌍, `extend_tokens`) | §1-3, §1-6 |
| σ 기본값 | 채움(NA + 기술값 1.0 한정) | §1-3 |
| σ′ 정의, T2′ L 범위 밖 | 채움 | §1-8 |
| q\_stall 역할 | 채움(기술 표만) | §1-3 |
| 캡처 bs 목록, bs\_padded 규칙 | 채움 | §3-4 |
| 채우기 출력 길이, m(L), 외삽 시험 수준 | 채움 | §1-4, §1-3, §1-1 |
| X\_T1·O | 채움 [제안] | §3-1 |
| 적합 채택 기준, 실현 SM 허용, Llama 짝 셀, Job 하드캡 | 채움 [제안] | §1-1·§1-2, §1-8, §4, §7 |
| "±3% 이동" 문장 | V 쪽 문장이라 함께 빠짐 | — |
| ENGINE\_DIGEST 통과값, config digest, gate 문서 main 경로, 서버 `new_token_ratio` 부팅 로그 값 | **[봉인 시 기입]** | 각 위치 |
| cap\_V, 앵커 보정, N1 클램프, Δa 출처, 3% 대 5%, 부호 있는 중앙값, 소거 변형, V\_REG | V PREREG로 이월 | — |

### 등록 caveat (문자 그대로, 감사서)
- "이 비용 모델은 Nemotron-Nano-9B-v2 한 모델의 것이며 'hybrid 서빙 비용 모델' 일반이 아니다."
- 항 측정 전용 추가 caveat [제안]: 항 값은 **고정 배치·저부하 전력 상태**에서 Job별로 잰 기술값이다. 조립·예측에 쓰이기 전까지 어떤 서빙 성능 주장의 근거도 아니다. T4 산식 대조는 구현 사실 확인이다.


## 0. 프레이밍과 이 문서가 하지 않는 것

- **주 목적(C)** = 비용 모델의 **항 측정**: lockstep PD-mux 비용 구조를 항 T1–T4로 재고(마이크로, 고정 배치·고정 길이) 기술 적합까지 낸다. **(H)** "왜 hybrid인가"는 hybrid 고유 항을 Llama와 **대비**해 답한다. 모델 간 goodput 비교는 하지 않는다.
- 하지 않는 것: 조립 검증(V)·λ\*\_SLO 예측·G0 튜닝·M3 PREREG가 아니다. cap goodput 스윕은 없다(cap = C\_mem 산식, 결정 15). 정책 순위·HE0·Claim E/G 등급을 바꾸지 않는다. layer-type별 분해는 하지 않는다(T1은 전체 모델 시간이라 원리적으로 불가, VERDICT\_claimE 근거 N2 PROFILE 항목).
- **항 T5**(CM-6 대기·TTFT 조립)와 CM-3(전환·경계)·CM-7(radix)은 범위 밖이다.
- 인용 범위(모든 항 공통): 이 기판(VESSL A100-SXM4-80GB, Job별 GPU UUID 공시)·이 스택(SGLang v0.5.10, 봉인될 ENGINE\_DIGEST, Triton causal\_conv1d)·mem 0.6·클럭 비고정. **Job별 값**이다 — 서로 다른 Job의 수준값을 비교·결합하지 않는다(Job 효과 진단 예외: J-A1/J-A2·J-B의 `108v` 차).

## 1. 항별 estimand와 맞춤 형태

공통: 단위 = (boot, cell) 창 중앙값. 창 = **ramp 인식 창(rev3.4, 결정 28 A-1, 정의는 §1-1 "창 정의")** — burst 뒤 `bs_real`이 등록 bs에 처음 도달한 기록부터 16 기록 제외 후 연속 300 간격, 창 안 bs 변화·동거 iteration은 제외·계수, 연속 seq, retract 0, cudagraph hit 정황. rev3.3의 "batch 평균 ctx p50이 목표 c의 ±64 띠 안"은 **제외 규칙에서 내리고** 라벨(`CTX_OFF_NOMINAL`)로만 남긴다(적합 입력은 실제 평균 ctx). 클럭·전력(I-1)은 창마다 공변량으로 기록하고 **제외 규칙으로 쓰지 않는다**(민감도 표만, §1-5).


### 1-1. T1 — CM-1 `T_dec(D, bs, ctx)`

| 항목 | 등록 |
|---|---|
| estimand | 결정 D(green-context로 **실현된** decode SM 수, sticky ON)·실행 batch bs·**창 안 실제 batch 평균 ctx**(rev3.4: 명목 c는 셀 라벨, 적합 입력은 실제값 — 결정 28 A-1)에서 decode step 시간(ms). 1차 = 엔진 측 decode 시간(PD-mux·vanilla **양쪽 경로**, 엔진 `decode_forward` 레코드의 연속 두 레코드 `t_host_start` 차 = `S_eng`, N2 선례 S-per와 같은 창 규칙; GPU 커널 시간 `decode_fwd_gpu_ms`[I-4]는 분해용 공변량이지 `S_eng`의 대체가 아니다 — 필드는 §3-4). 2차 = 클라이언트 steady-state 토큰 간격 중앙값(모델 간 대비의 공통 추정량, §4). 엔진 대 클라이언트 bs별 3% 대조와 불일치 bs 제외는 §1-7. 입력 bs는 실행 bs `bs_real`이 아니라 `bs_padded`(I-6, 규칙 §3-4)로도 기록 |
| 격자 | D ∈ {16, 24, 34, 44, 64} sticky ON + **D=108을 두 모드로**: `108u`(PD-mux 엔진 모드, 비분할 `(0,108)`, N2 관례), `108v`(vanilla 엔진, fused 계열 decode). bs ∈ {1, 8, 32, 48, 96}, c ∈ {2k, 8k, 32k, 64k} + **16k**(D ∈ {24, 34, 44, 108u, 108v}만; shape C 구간 11–20k 보간용) + **(bs, c) = (64, 8k), (64, 12k)**(rev2 신규, 같은 5개 D·모드만; shape C 영역 bs 55–80 보간, 감사 MAJOR 4. 12k 셀은 §3-1 규칙으로 실행 여부 판정, 비실행이면 `INFEASIBLE_BY_RULE`). 실행 가능 판정은 §3-1의 규칙 |
| **창 정의 [봉인, rev3.4]** | (1) burst: bs개 요청을 동시에(`input_ids` 길이 c\_in = c − 166, `ignore_eos`, `max_new_tokens` = 셀별 O, §3-1). (2) r0 = 그 셀 구간의 `decode_forward` 중 `bs_real` = bs인 **첫 기록**. (3) 창 = r0+16 … r0+316의 301 기록(연속 seq, 300 간격; trim 16은 r0 **뒤**에서 센다). (4) 간격 k(기록 k→k+1)는 기록 k가 `bs_real` ≠ bs(bs 변화) · `mode` ≠ decode(`mixed` = forward 안 prefill 토큰) · **클라이언트 합류 미확인**(그 시각 + 0.25 s까지 첫 토큰을 받은 셀 요청 수 ≠ bs) 중 하나면 **제외하고 사유별 개수를 기록**; 제외 > 30이면 `WINDOW_CONTAMINATED`. (5) **동거 판정 근거(TEL OFF)**: T1 boot에는 phase event가 없으므로 prefill 동거는 `decode_forward.mode` = mixed와 클라이언트 측 합류(첫 토큰 수신 시각)로만 판정한다. PD-mux는 split-prefill이 끝난 요청만 decode batch에 들어가므로(코드 사실) 셀 요청이 모두 합류한 뒤 셀 자신의 prefill은 동거할 수 없고, 남는 동거 원인은 **이전 셀의 잔여 요청**뿐이다 → 셀 구간 어디서든 `bs_real` > min(bs, 그 시각 합류한 셀 요청 수)이면 `FOREIGN_RESIDUAL`(셀 무효). (6) S\_eng = 포함 간격의 중앙값. (7) **실제 ctx** = 포함 간격 시작 기록마다 (c\_in + 그 시각까지 클라이언트가 받은 토큰 수)의 요청 평균을 다시 창 평균한 값(클라이언트 수는 엔진보다 진행 중 토큰 1개 늦을 수 있다, 편향 ≤ 1 토큰). (8) 모든 요청이 319 토큰(= 1 + 16 + 300 + 2)을 받으면 클라이언트가 HTTP `/abort_request`로 셀의 모든 요청을 취소하고 스트림 종료를 확인한 뒤 다음 셀로 간다. (9) 관측 ramp(r0 이전 기록 수)가 등록 상한을 넘으면 `RAMP_EXCEEDED_BOUND`(라벨; 창이 깨지면 `WINDOW_SHORT`/`WINDOW_CONTAMINATED`). 상태 우선순위: `BENCH_INVALID` > `FOREIGN_RESIDUAL` > `RETRACT` > `RECORDER_UNPROVEN` > `WINDOW_SHORT` > `WINDOW_CONTAMINATED` > `CG_UNPROVEN` > `PAD_MISMATCH` > VALID. **ctx 띠 결정 [봉인]**: 적합은 실제 평균 ctx, 셀 라벨은 명목 c와 실제값 병기. ±64 띠는 `CTX_OFF_NOMINAL` 라벨(제외 아님). 근거: ramp가 있으면 앞선 요청이 ramp만큼 더 생성해 실제 ctx가 c보다 커지며(가짜 엔진에서 D44 (32, 8k) 약 +650 토큰, 인용 불가), 띠를 제외 규칙으로 두면 ramp 셀이 전부 빠진다; 정본 수식상 decode 비용은 Σctx에 비례하므로 실제값으로 적합해야 c 계수가 편향되지 않는다(명목 c 적합은 가짜 엔진 진리값 대비 a −0.3%·b +1.3%·c +5.6% 편향, 변이 M22) |
| 고정 노브 | cap = 128(V 운용 cap 근방 [유도 99–127], 결정 15), `--max-mamba-cache-size` = cap, mem 0.6, context\_length 73,728, cudagraph ON, 캡처 bs ⊇ {1, 8, 32, 48, **64**, 96}(§3-4 목록; 부팅 로그 검사, 아니면 `ABORT_CG_BS`), 같은 config 바이트(`pdmux_n3_A_agn.yml` 계열 + D=64 index 추가, digest [봉인 시 기입]) |
| 제안 함수형 | `T_dec(D,bs,c) = a_D + b_D·bs + c_D·bs·c` — a = bs·ctx 무관 고정(가중치 sweep·launch), b = 요청당(SSM state·MLP), c = KV 읽기. 대안 형(적합도 실패 때만, 이 순서): F2 = F1 + d\_D·bs², F3 = a + b·bs^β + c·bs·c. 셋 다 실패하면 `FORM_INADEQUATE_T1_D{d}`이고, 이 항을 쓰는 후속 예측기는 **격자 안 이중선형 표 보간(log bs, log c), 외삽 금지**를 써야 한다(예측기는 이 문서 밖) |
| 맞춤 방법 | 상대오차 가중 최소제곱(Σ((T−T̂)/T)² 최소), D·엔진 모드마다 따로. 단위 = (boot, cell) 중앙값, **입력 ctx = 그 셀의 실제 평균 ctx(창별 값의 중앙값)**, 외삽 시험의 "수준"은 명목 c로 정한다(rev3.4). D 간 부속: 계수별 `κ_j(D) = κ_j(108)·(108/D)^γ_j`(j = a, b, c)를 6 D에 맞춰 γ\_j를 **기술값**으로(N2 PROFILE의 계수비는 인용 금지·재측정, 층 종류 귀속 금지) |
| 적합도 보고 | (i) 표본 내 상대 잔차 중앙·최대 (ii) leave-one-cell-out 중앙·p90 (iii) **외삽 시험 [수준 봉인]**: ctx 수준 1개(32k를 먼저, 그다음 8k)와 bs 수준 1개(96을 먼저, 그다음 32)를 통째로 빼고 맞춘 뒤 뺀 수준 예측 오차. 뺀 뒤 남는 서로 다른 수준이 2개 미만인 (D, 모드)는 그 시험 `NA` (iv) 두 round를 각각 맞춘 계수의 차. **적합 채택 기준 [제안, 봉인 시 이 값 고정]**: (ii) 중앙 ≤ 3% ∧ p90 ≤ 8% ∧ (iii) 최대 ≤ 10%. 미충족은 위 라벨 |
| 인용 범위 | 해당 D·엔진 모드·**cap 128 부팅**·이 격자 박스 안(rev3.4: 박스의 ctx 축은 **실제 평균 ctx** 기준)(볼록껍질 밖 조회는 후속 예측기가 `PRED_EXTRAPOLATED`로 다룰 몫 — V PREREG). 박스 밖 외삽 문장 금지. 허용: "이 Job에서 D=…, bs=…, ctx=…의 decode step 중앙값은 …ms이고, 함수형 F1의 표본 내 잔차는 …, 외삽 시험 오차는 …였다" |
| 금지 | N2 PROFILE 계수(2.51/4.63/1.95×, −14~16%)·"shape B ctx 항 1%"·"shape C에서 hybrid 항이 유의하다"(VERDICT\_claimE 금지 문안); γ\_j를 s\_attn/s\_ssm 증거로 읽는 것; D 간 비를 "layer-type 차이"로 읽는 것 |

cap 불변성 점검(J-A1 안, D=44 한 boot): cap 48로 부팅해 bs ∈ {8, 32} × c ∈ {2k, 8k} 4셀을 cap 128 boot의 같은 셀과 비교. 차 ≤ 3%면 `CAP_INVARIANT`, 아니면 `CAP_DEPENDENT`(T1 값에 "cap 128 조건부" 꼬리표, TP2 앵커는 cap 48을 쓰므로 영향 공시).

### 1-2. T2 — CM-2 `T_span(L, SM)`

| 항목 | 등록 |
|---|---|
| estimand | 다른 부하가 없을 때 단일 요청 prefill 시간. 1차 = 클라이언트 TTFT(`input_ids`로 정확히 L 토큰, 출력 1토큰, `ignore_eos`) 중앙값 — J1 floor(CM-2)와 같은 정의라 모든 모델·모드에서 비교 가능. 2차 = 엔진 span 시간(Nano PD-mux만; I-3 없음). **P = 108에서만 정의된다(rev3.4, 결정 28 A-2)**: decode batch가 비면 sticky ON이어도 prefill은 stream group 0 = (108, 0)에서 돈다(코드 사실 `multiplexing_mixin.py` `adjust_stream_groups`) |
| 격자 | L ∈ {1,024, 4,096, **8,800**(≈ s/k, 유도 경계 근방), 16,384, 32,768, 65,536} × **P = 108만**(`108u`·`108v` boot, J-H1 vanilla boot). rev3.3의 "P = 108 − D ∈ {92, 84, 74, 64, 44}(T1 boot의 부산물)"은 **폐기** — sticky-D boot의 단독 prefill은 108 SM에서 돌므로 그 값은 P = 108 − D가 아니다. prefill SM 축은 §1-8의 T2′가 맡는다. 한 (boot, L)당 8회, 앞 1회는 워밍업이라 제외. 계획 검사가 sticky-D boot의 T2를 거부한다 |
| ~~조건부 확장 T2-ext~~ | **삭제(결정 28 A-6)**: context 73,728에서 96k·124k 입력은 불가능하고, int64 패치의 정확성은 G-β(서빙 98,304 통과·되돌린 판 95,100 사망)가 이미 확인했다 |
| 제안 함수형 | `T_span(L,P) = u_P + v_P·L + w_P·L²`(P = 108). P 의존은 T2′에서 `(P_max/P)^θ′`로 기술값(§1-8). **모델-무관 형태 지표** `q = [T(65,536)/65,536] / [T(8,800)/8,800]` (토큰당 비용 증가비) |
| 맞춤 방법 | 상대오차 가중 최소제곱, 단위 = boot 중앙값 |
| 적합도 보고 | leave-one-L-out(내부 L 4개) 상대오차, 두 round 계수 차. 채택 기준 [제안, 봉인 시 고정]: 내부 L 제외 오차 ≤ 5%. 미충족 `FORM_INADEQUATE_T2_P{P}`, 표 보간(log L) 사용 |
| 라벨 | `q ≥ 1.10` → `PREFILL_SUPERLINEAR_SCOPED`, `q < 1.10` → `PREFILL_NEAR_LINEAR_SCOPED`, 구간이 1.10을 걸치면 `Q_UNRESOLVED` [제안]. J1 기울기(0.111 ms/토큰, L ≤ 16k, **mem 0.82 결과**)와 P=108 기울기의 차는 **기술값으로만**(차 > 5%면 `T2_J1_SLOPE_DEVIATES`, 기판·mem 차이 공시 항목이지 결과·보정 근거 아님) |
| 인용 범위 | 이 기판·이 스택·chunk −1(PD-mux 강제 조건과 같음)·**P = 108**에서 L ≤ 64k. "초선형/선형"은 q 라벨까지만 |
| 용도 | 후속 예측기의 SLO floor 입력과 fused 청크 비용 입력(예측기는 V PREREG). 청크 prefill 비용을 `T2(pos+c) − T2(pos)`로 근사하는 가정은 **이 문서가 검증하지 않는다** — 필요하면 V PREREG가 따로 등록 |

### 1-3. T3 — CM-4 ρ (동거 간섭, **총효과**)

| 항목 | 등록 |
|---|---|
| estimand | 같은 boot·같은 셀에서 `ρ = S-per(동거 창) / S-per(단독 창)`. 동거 창 = 고정 decode batch(c = 8,192 [제안])가 돌 때 prefill 입력(L)이 지속 진행되는 iteration들. 처리 = 동거 + 그에 동반된 전력 한도 클럭 하락의 **합성값**(N2 Job-EA 선례: 분리 불가) |
| **기록(rev2 추가, 감사 MAJOR 8)** | 중앙값 ρ만 쓰지 않는다. 셀·창마다 (a) 동거 창 **iteration 시간 분포 전체**(원 값 파일 + p50/p90/p95/p99/max)와 단독 창 분포 (b) **정지 횟수·비율**: 단독 창 중앙값의 5배를 넘는 iteration 수와 연속 정지 길이(SLO가 요청 내부 ITL p95라 중앙값이 정지를 가린다; 프로브 64k 최대 정지 0.8 s가 선례) (c) **span co/solo `σ = T_span_eng(동거) / T_span_eng(단독)`**: 같은 boot·L·P에서 동거 창의 prefill span 시간 대 단독 창의 비. **σ 엔진 필드 [봉인]** = PD-mux 이벤트 `prefill_span_start`/`prefill_span_end`(`PDMUX_PHASE_EVENTS=1`, 같은 `span_seq`의 두 레코드 `timestamp_monotonic_s` 차 = `T_span_eng`; `n_new = 1`·`extend_tokens` = L인 span만; 한 시점에 split-prefill는 하나라 쌍이 엄밀히 교대, `src/multiplex/multiplexing_mixin.py:2186-2224`). 이벤트가 없으면(`PHASE_EVENTS` 비활성·레코드 누락) `SIGMA_NOT_COLLECTED`. 클라이언트 TTFT는 큐잉을 포함하므로 σ로 쓰지 않는다. **`SIGMA_NOT_COLLECTED`의 기본값 [봉인]**: σ 칸은 `NA`로 두고 어떤 값으로도 채우지 않는다. 후속 사용을 위한 기술용 하한값은 σ = 1.0(간섭 없음)이며 쓸 때는 반드시 `SIGMA_NOT_COLLECTED` 꼬리표와 σ ∈ {1, ρ} 민감도를 병기한다 — 이 기본값을 쓰는 것의 승인은 후속 PREREG의 몫이다. (d) span 구간 안의 `decode_iteration` 레코드 수 `n_it_obs`(동거 창 iteration 수 실측) |
| 격자 | D ∈ {24, 44} × bs ∈ {8, 32} × L ∈ {8,192, 32,768} = 8셀. 동거 창이 128 iteration 이상 되도록 prefill 입력 개수 **m(L) [제안, 값 봉인]: m(8,192) = 8, m(32,768) = 3** (근거 [유도, 시간 모형 §7-1]: 최악 셀에서 span 1회가 덮는 decode step 수 ≈ 20–35[L=8,192, D=24·bs 32 / D=44·bs 8]·≈ 88[L=32,768, D=24·bs 32]이므로 8×20 = 160, 3×88 = 264로 128을 넘는다; 근거가 틀려 실측 `n_it_obs` < 128이면 그 셀은 `T3_WINDOW_SHORT` 라벨로 표에 남기고 ρ를 조건부로 표시, 재측정은 새 Job). 반복 = round당 2회 × round 2 = 셀당 4 |
| 맞춤 | `ln ρ`와 `ln σ`의 셀별 평균(표). 격자 보간 규칙은 (log bs, log L) 이중선형, D=34는 ln D 선형(`RHO_D_INTERP`), 격자 밖은 클램프(`RHO_CLAMPED` 횟수 공시) — 후속 사용 시 따를 규칙이다. 정지 비율 `q_stall`(>5× 중앙값)도 셀 표로 둔다. **`q_stall`의 역할 [봉인]**: 기술 표(꼬리 위험의 크기)이며 이 문서에서 어떤 예측식의 입력으로도 쓰지 않는다 |
| 적합도 | 4개 반복의 산포, 셀별 SW power cap 노출률·SM 클럭 중앙값(동거/단독 창 각각) |
| 인용 범위 | **총효과로만**: "400 W 한도 아래 클럭 하락을 포함한 동거 감속". N2 Job-EA 허용 문안 (P)(A)(B)의 틀을 따른다. N2 RESULT :68-72(decode는 prefill 청크를 기다리지 않음)는 이 기판·그 설정의 기술이며 σ·`q_stall` 측정이 이 PREREG의 확인 수단이다 |
| 금지 | "ρ는 동거 간섭의 크기다"(cap 병기 없이)·"전력 무관 간섭"·"ρ는 D·b·c에 무관"·ρ에서 정책/재배분 경계 도출·J1/N2 Job의 ρ와의 비교·중앙값 ρ만으로 "꼬리가 없다" |

### 1-4. T4 — CM-5 C\_mem과 거절 사유 몫 (rev2 전면 개정, 감사 死因 1)

**성격 라벨(공시 의무)**: 이 항의 산식 대조는 **"구현 사실 확인"**이다. 산식 B는 엔진의 입장 회계(admission accounting)를 그대로 옮긴 것이라, 채우기 실측이 B와 맞는 것은 엔진이 코드대로 동작한다는 사실 확인이지 독립 예측의 성공이 아니다(교훈 9: 항등식에 가까운 검사). 산식 대조 결과를 "비용 모델이 C\_mem을 예측했다"로 쓰지 않는다. T4에서 비용 모델에 실제로 기여하는 것은 (a) 부팅 로그 사실(s, k, 풀)과 (c) 거절 사유 몫이다.

| 항목 | 등록 |
|---|---|
| estimand | (a) 부팅 로그 사실: KV 풀 토큰 수, mamba 슬롯 수·바이트, 가중치 바이트 → s = 슬롯당 바이트, k = KV B/토큰, 풀 합 (b) 채우기 실측: 같은 L의 요청 N = cap + 8개를 한꺼번에 보내 **실현 최대 동시 running 수** `n_real` (c) 거절 사유 몫(I-5: KV / cap≡슬롯), iteration 가중과 시간 가중 병기 |
| 예약 회계 [코드] | `ignore_eos`이고 radix OFF이면 입장 시 **입력 + `max_new_tokens` 전량**을 예약한다(`schedule_policy.py` `add_one_req_ignore_eos`: `new_token_ratio = 1.0 if ignore_eos` :647-652, 입장 분기 :752-753). 그 외 요청은 `min(max_new_tokens, 4096[CLIP_MAX_NEW_TOKENS]) × new_token_ratio`를 예약한다. 서버 `new_token_ratio` 값은 부팅 로그 값을 기입 [봉인 시 기입] |
| **산식 두 개(둘 다 봉인)** | **원식 A(rev1)** `C_A(L) = floor(풀 / (s + k·L))` — 출력 예약 없음. **수정식 B(rev2 1차)** `C_B(L, R) = floor(풀 / (s + k·(L + R)))`, 예약 `R` = (`ignore_eos`: `max_new_tokens` 전량; 그 외: `min(max_new_tokens, 4096) × new_token_ratio`). 임의 cap에서의 실현 예측 `n_B(cap) = min(cap, floor((풀 − cap·s) / (k·(L + R))))`(mamba 슬롯 cap개가 정적 예약되어 KV 풀이 cap에 따라 줄므로 풀 합이 아니라 이 형으로 쓴다). 원식의 대응 예측은 **B와 같은 형식(mamba 정적 예약 반영)에서 R만 0으로 둔 `n_A(cap) = min(cap, floor((풀 − cap·s) / (k·L)))`로 통일**한다(감사 2회차 caveat: rev2의 `min(cap, C_A)`는 B와 비대칭이었다). 두 식은 R 항 하나만 다르다. `C_A(L) = floor(풀/(s + k·L))`는 부팅 cap을 정하는 값으로만 남는다. `cap = C_B`이면 `n_B = C_B`로 닫힌다(고정점) |
| Nano 격자 | L ∈ {4,096, 8,800, 16,384, 65,536}, 출력 R = {4,096, 4,096, 4,096, 8,000}(채우기 규칙과 동일, `ignore_eos`). **부팅 9개**: (i) cap 48 1개(네 L 모두) (ii) **cap = C\_A(L)** 4개 = {156, 114, 80, 27} — 원식 A와 수정식 B가 갈리는 판별 부팅 (iii) **cap = C\_B(L, R)** 4개 = {118, 93, 68, 24} — B의 고정점 부팅. 풀이 cap과 독립이라는 가정(KV 토큰 + mamba 바이트 = 상수)은 [유도]이며 **부팅 9개의 풀 합 일치(2% 안, 아니면 `C_MEM_DRIFT`)로 검증** |
| **데이터 이전 예측 표** [유도, 풀 33.04 GB·s 143.5 MB·k 16,379 B 대입; **부팅 로그의 풀로 같은 코드가 재계산한 표를 각 부팅 직후·채우기 전에 봉인**] | 아래 표 | [CS-OK] 33.04는 풀 GB 값(우연 일치)
| 채우기 규칙 | 출력 길이는 마지막 요청의 첫 토큰 시각이 가장 이른 요청의 완료 시각보다 앞서도록 L별로 등록한다(L = 4k–16k는 4,096, 64k는 8,000). 위반하면 `FILL_INCOMPLETE`(그 셀은 산식 대조에서 제외, 건수 공시). **출력을 줄이면 R도 같이 줄어 `C_B`가 커지므로 출력 길이는 R의 입력으로 고정 기록**한다. **채우기 출력 길이 [봉인]**: L ∈ {4,096, 8,800, 16,384} → 4,096, L = 65,536 → 8,000(위 표 R 열과 동일). 변경 금지. arm = static D=44 PD-mux [제안, fused 대칭 점검은 O-8] |
| Llama(대비) | 부팅 1개(cap 128, mem 0.6): 풀 토큰 수(s = 0, k 131 KiB) + 채우기 L ∈ {4,096, 16,384}, R = 4,096(`ignore_eos`). 예측: KV 풀 263,900 토큰 [rev1 §4 ③] 기준 `n_B` = {32, 12}, 원식 A = {64, 16} |
| 판정 형 | 산식 B: `n_real / n_B ∈ [0.90, 1.10]` → `CMEM_ACCT_HELD`, 아니면 `CMEM_ACCT_MISS` [제안]. 원식 A: `n_real / n_A < 0.90`이면 `CMEM_ORIG_OVERPREDICTS`(예측된 결과, 아래 표의 ratio 열을 사전 예측으로 봉인 — ±0.05). 사유 몫은 기술값 |
| 인용 범위 | 부팅 로그 사실은 [측정] 그대로. 산식은 "이 스택의 정적 풀 구현(cap≡슬롯)에서 입장 회계를 확인했다"는 범위. 할당기 종속이라 hybrid 일반 문장 금지(결정 8′). "예측이 맞았다"·"비용 모델이 C\_mem을 검증했다" 금지 |

사전 예측 표 (Nano; `ratio_A` = cap = C\_A 부팅에서의 `n_B / C_A` = 원식의 예측 과대 정도):

| L | R | C\_A | C\_B | C\_B / C\_A | cap = C\_A 부팅의 `n_B` (`ratio_A`) | cap 48 부팅의 `n_A` / `n_B` |
|---|---|---|---|---|---|---|
| 4,096 | 4,096 | 156 | 118 | 0.756 | 79 (**0.51**) | 48 / 48 |
| 8,800 | 4,096 | 114 | 93 | 0.816 | 78 (**0.68**) | 48 / 48 |
| 16,384 | 4,096 | 80 | 68 | 0.850 | 64 (**0.80**) | 48 / 48 |
| 65,536 | 8,000 | 27 | 24 | 0.889 | 24 (**0.89**) | **24 / 21** |

감사서의 비 0.51·0.69·0.80·0.89가 재현된다(8,800은 풀 반올림으로 0.68 vs 0.69; 부팅 로그 풀로 재계산). rev1 설계로 데이터를 모았다면 비가 1이 되는 L은 없고 `CMEM_PRED_MISS`가 설계 단계에서 확정이었다는 뜻이다. rev2는 이 표를 데이터 이전에 낸다. cap 48 부팅의 64k 칸이 A와 B가 갈리는 두 번째 판별점이다(통일 형식 24 대 21; 다른 세 L은 cap 48이 구속해 48 대 48). cap = C\_A 부팅에서는 `n_A = C_A`(통일 형식으로 재계산해도 156/114/80/27 그대로)이고 B와의 비가 위 ratio 열이다.


### 1-5. 공변량·민감도(모든 항)

I-1 클럭·전력·throttle을 창마다 기록한다. SW power cap 노출률 > 20%인 창을 뺀 재적합을 민감도 표로 낸다(계수 변화 > 3%면 `CLOCK_SENSITIVE`, 제외는 하지 않음) [제안]. 클럭 고정은 불가(VESSL). 항 측정은 고정 배치·저부하 전력 상태라 서빙 부하의 전력 상태와 다를 수 있다(등록 caveat 2번째).


### 1-6. T\_span 단위 봉인 (T5 조립은 이 문서 밖)

항 측정이 후속 조립을 위해 남기는 단위만 봉인한다. 엔진이 내는 prefill 쪽 시간은 **span 1건**(split-prefill 배치 입장→완료, `prefill_span_start`/`prefill_span_end` 쌍)이며 이것이 `T_span_eng`이다(§1-3). `SPAN_TOKENS` = `prefill_span_start`의 `extend_tokens`. 요청 1개 전체 prefill 시간은 T2의 `T_span(L, P)`(클라이언트 TTFT)다. 한 span 안의 decode iteration 수는 `n_it_obs`로 실측한다(환산 추정 `T_span/T_span_iter`는 하지 않는다). `split_forward_token_budget`(65,536)·`max_prefill_tokens`(16,384)의 부팅 로그 값은 모든 boot에 기록하고, 둘의 상호작용 해석은 **[봉인 시 기입 — 하네스 로그에서 읽은 값, 해석은 후속]**.

### 1-7. 엔진 S-per 대 클라이언트 간격 대조 (rev2 신설, 감사 MAJOR 7)

배경: rev1 시점에는 `decode_iteration`이 PD-mux mixin에서만 방출되어(`src/multiplex/multiplexing_mixin.py:2245-2270`) Llama·`108v`는 클라이언트 토큰 간격뿐이었고 GIL·클라이언트 과부하(confound 7)에 노출되었다. 결정 16 O-2로 I-4·I-6이 엔진 묶음에 들어가면 **엔진 측 decode 시간이 PD-mux와 vanilla 양쪽 경로에서 기록**된다(필드 `decode_forward`: §3-4). 이것으로 문제가 상당 부분 해소되지만 대조는 유지한다.
- **1차 추정량 = 엔진 측 decode 시간**(양 경로). **2차 = 클라이언트 steady-state 토큰 간격**(모델 간 공통 추정량).
- **대조**: 각 (엔진 모드 `108u`·`108v`·Llama vanilla, bs)마다 같은 창에서 `|ln(client 중앙값 / S_eng 중앙값)| ≤ ln 1.03`(비교 대상은 `decode_fwd_gpu_ms`가 아니라 `S_eng`다 — gate가 I-4 대 간격 비를 fused에서 0.75–1.00으로 허용하므로 커널 시간과의 3% 대조는 의미가 없다)이면 `CLIENT_ENGINE_AGREE_bs{b}`, 아니면 `CLIENT_ENGINE_MISMATCH_bs{b}` — 그 bs의 클라이언트 간격 기반 양은 H 대비(§4 ③)와 T1 2차에서 **제외**하고 엔진 측 양만 쓴다. 대조는 J-A·J-B에서 수집된 모든 창에서 계산한다.
- **fallback**: gate에서 vanilla 경로의 엔진 측 시간 방출이 확인되지 않으면(`NOT_COLLECTED`) 대조는 `108u`에서만 가능하고, `108v`·Llama의 클라이언트 간격은 `108u`에서 얻은 bs별 일치 여부를 **조건**으로 `CLIENT_ONLY_UNVERIFIED` 꼬리표를 달아 쓴다(제외 아님, 주장 약화).
- **원인 문장 금지**: Nano 1개 대 Llama 1개의 기술 차이까지만 쓴다. "hybrid 고유"를 원인으로 쓰지 않는다(§4-2).
### 1-8. PD-mux boot의 sticky 모드와 P 실현 검증 (rev2 신설, 감사 MAJOR B)

- **sticky 모드 고정**: J-A·J-B·J-C의 모든 PD-mux boot는 **sticky ON**(T1과 같은 config 계열 `pdmux_n3_A_agn.yml`, D별 digest [봉인 시 기입])이다. 후속 PD-mux arm도 같다. 모드가 arm·Job 사이에서 다르면 `STICKY_MODE_MISMATCH`로 그 단위를 제외한다.
- **실현 SM 로그 검증**: 요청 SM ≠ 실현 SM(생성 성공 ≠ 요청 수신)이므로 boot마다 첫 요청 처리 전후의 realized decode/prefill SM 로그를 읽어 요청 D·P와 비교한다(**허용 규칙 [제안, 봉인]**: A100 green-context 입도 4 SM이므로 `|실현 − 요청| ≤ 4` ∧ `실현 decode + 실현 prefill ≤ 108`이면 일치; 표의 D·P 축에는 **요청값이 아니라 실현값**을 쓴다(D=34 같은 4의 배수가 아닌 값이 반올림되는 것을 포함). 4 초과 차이면 아래 라벨). 불일치 → `SM_NOT_REALIZED_D{d}`, 그 boot의 T1·T2 값은 해당 D 항에 쓰지 않는다.
- **T2′ = prefill SM 축의 주 측정 (rev3.4, 결정 28 A-2 — rev3.3의 `T2_P_NOT_REALIZED` 조건부 대체를 승격)**: bs=1 decode 요청 1개(입력 2,048, `max_new_tokens` 60,000, `ignore_eos`)가 decode 중인 상태에서 잰 단일 요청 prefill TTFT(L 토큰, 출력 1). decode가 바쁘면 sticky-D boot의 prefill은 그 division의 prefill 절반(P = 108 − D, 실현값은 green read-out)에서 돈다(코드 사실). **L 범위 [봉인]** = T2와 같은 {1,024, 4,096, 8,800, 16,384, 32,768, 65,536}, 한 (boot, L)당 8회, 앞 1회 제외. 수집 boot = J-A의 7개 모드 boot 전부(sticky-D 5개가 P 축, `108u`는 FIXED\_DSM 44 division의 prefill 절반(P 64), `108v`는 vanilla라 prefill이 먼저 돌아 decode가 멈추는 참조점) + D44 OFF 진단 boot(L 4,096·16,384).
- **성립 확인 라벨 [봉인]**: 클라이언트 측(decode 토큰이 prefill 송신 전과 첫 토큰 뒤에 모두 도착)은 rep 유효 조건이고, **엔진 측** = 쓰인 rep마다 TTFT 구간의 가운데 절반 [t\_send + 0.25·TTFT, t\_send + 0.75·TTFT] 안에서 시작한 `decode_forward`(`bs_real` ≥ 1)가 1개 이상이면 `T2P_COEXIST_CONFIRMED_L{L}`, 아니면 `T2P_COEXIST_UNCONFIRMED_L{L}`(구간 양 끝은 대기와 vanilla에서 prefill 직후의 decode step을 담으므로 뺀다). OFF boot는 `NOT_COLLECTED`. 적합·θ′에는 CONFIRMED 점만 쓴다. `108v`는 UNCONFIRMED가 정상(prefill-first)이며 그 값은 "decode 1개가 대기 중"인 참조다.
- **적합(기술값)**: (boot마다) T2와 같은 이차형·채택 기준(내부 L 제외 오차 ≤ 5%, 미충족 `FORM_INADEQUATE_T2P_P{P}`), q 지표. P 의존 `θ′` = 같은 Job·round의 sticky-D boot들에서 ln(T2′(L,P)/T2′(L,P\_max))를 ln(P\_max/P)에 원점 통과 최소제곱(P\_max = D16 boot의 실현 P). 라벨 없음.
- **estimand 변경 공시 [봉인 문장]**: "rev3.4부터 prefill SM 축(P = 108 − D)의 값은 '다른 부하가 없을 때'가 아니라 **bs=1 decode 요청 1개가 동시에 decode 중일 때**의 단일 prefill TTFT(T2′)다. decode가 비면 이 엔진은 sticky 설정과 무관하게 108 SM으로 prefill하므로 '다른 부하 없음'과 'P < 108'은 이 엔진에서 함께 성립하지 않는다. 따라서 T2′와 T2(P = 108)의 차에는 SM 감소와 decode 1개 동거 효과가 섞여 있고, 둘을 분리하지 않는다." 라벨 `T2_FALLBACK_BS1_COEXIST_D{d}`는 폐지(모든 D가 T2′)하고, 인용 시 이 문장을 붙인다.
- **σ′ 정의 [봉인, rev3.4 갱신]**: sticky-D boot의 T3 σ는 `σ′ = T_span_eng(동거, T3 창) / T2′(L, P)`만 쓴다 — 분모가 이미 bs=1 동거를 포함한다. T3의 L 32,768은 이제 T2′ 범위 안이고, 8,192는 4,096·8,800 사이 log L 선형 보간만 허용한다. 범위 밖은 `T2PRIME_L_OUT_OF_RANGE`, σ′ `NA`. σ(대 T2)와 σ′를 곱하거나 병용하지 않는다.


## 2. 측정 방식

### 2-1. 한 줄 정의
고정 배치·고정 길이를 엔진 안에서 만들고(burst로 bs개를 동시에 보내 ctx c까지 도달시킨 뒤 창을 읽는다) 서빙 부하(rate 사다리)는 쓰지 않는다. 서빙 부하(rate 사다리)는 이 문서에서 쓰지 않는다.

### 2-2. 단계 분리
boot마다 warm-up(8요청, 같은 형상) → correctness(sticky 배선 로그 + 짧은 greedy 출력 비공백 + CG canary) → benchmark(셀들) → 후행 비측정 요청 → idle 2×flush → 종료. 계측 수준은 boot 종류별로 §2-2′ 매트릭스에 고정한다(rev3의 "관측자 효과는 gate의 TOST가 묶는다"는 결정 22로 폐기 — G-γ는 PASS가 아니라 특성화로 닫혔다).

### 2-2′. 계측 정책 적용 (결정 22, rev3.1)
결정 22의 "판정 측정은 계측 OFF"는 이 문서에 해당하는 측정이 없다(판정 0, V 이월). 이 문서의 1차 estimand는 계측 레코드로 **정의**되므로(T1 `S_eng` = `decode_forward` 간격, T3 σ = phase event 쌍, T4 사유 몫 = I-5) 계측을 끄면 estimand가 사라진다. 대신 아래 두 규칙을 봉인한다.

**(1) 계측 매트릭스 [봉인]** — 같은 비교 단위 안의 boot는 계측 집합이 같다.

| boot 종류 | I-1 | I-4·I-6 (`PDMUX_DECODE_PROBE_PATH`) | I-5 | TEL (`PDMUX_TELEMETRY_PATH`+`PDMUX_PHASE_EVENTS=1`) |
|---|---|---|---|---|
| T1·T2·T2′ boot (D 16·24·34·44·64, `108u`, `108v`, Llama, TP2·앵커) | ON | ON | OFF | **OFF** |
| T3 전용 boot (D 24·44, PD-mux) | ON | ON | OFF | **ON**(σ에 필요) |
| T4 boot (채우기) | ON | ON | ON | OFF |
| OFF 진단 boot (아래 (2)) | ON | OFF | OFF | OFF |

- rev3에서는 T3 셀이 D 24·44의 T1 boot 안에 있었다. TEL이 λ를 −1.7% 움직이므로(결정 22 근거) 그대로 두면 D 사다리 안에서 D 24·44만 TEL ON이 되어 비대칭이다. 그래서 **T3 셀을 D 24·44의 별도 boot로 분리**한다(J-A당 boot +2, 셀 수 불변).
- TEL은 PD-mux 경로에서만 방출된다. T3 ρ·σ는 같은 boot 안의 동거 대 단독 비라 TEL 효과가 분자·분모에 함께 들어가지만, 상쇄된다고 가정하지 않는다 — "T3 boot는 TEL ON"을 인용 범위에 적는다.
- I-4·I-6은 PD-mux와 vanilla에 같은 래퍼로 걸린다(§3-4, arm 대칭). I-4의 ITL p95 +1.0%[측정, agnostic 분해 Job]는 T1 값 자체에 들어가 있다 — T1은 "I-4 ON 상태의 decode step 시간"이다.

**(2) Job 내 ON/OFF 진단 짝 [봉인, 셀·boot 수는 [제안]]** — 계측 효과를 같은 Job 안에서 재서 공시만 한다(보정 계수로 쓰지 않는다, 결정 22).
- J-A1·J-A2: OFF 진단 boot 2개(D=44 PD-mux, `108v`), 각 셀 6개 = T1 (8, 2k)·(32, 2k)·(32, 8k)·(64, 8k)·(96, 2k)·(16, 16k) + L ∈ {4,096, 16,384}(rev3.4: `108v` OFF는 T2, D44 OFF는 **T2′** — ON D44 boot에 T2가 없으므로). 같은 Job의 ON boot 같은 셀과 비교한다. rev3.4 실행 가능성 규칙(§3-1)에서 (64, 8k)는 `INFEASIBLE_BY_RULE`(ON 짝도 비실행)이라 실제 비교 셀은 5개다. OFF boot의 T1 창은 같은 ramp 인식 규칙을 클라이언트 측으로 적용한다(합류 = 마지막 요청의 첫 토큰, 그 요청 토큰 16개 제외 후 300 토큰).
- J-H(rev3.2, 구 J-B1): OFF 진단 boot 1개(dummy a4 vanilla, 같은 셀). Nano `108v`의 OFF 짝은 J-A에 있다.
- J-C: OFF 진단 boot 1개(TP2, T1 셀 3개). [결정 23 — O-14 포함]
- 산출(기술값): 셀별 `ln(client 간격 중앙값_ON / client 간격 중앙값_OFF)`와 T2 TTFT 비. 2차 estimand(클라이언트 간격)만 비교할 수 있다(OFF boot에는 `S_eng`가 없다). 라벨: 모든 셀에서 |비| ≤ 1.5%면 `INSTR_EFFECT_SMALL`, 아니면 `INSTR_EFFECT_DISCLOSED`(값 표 병기). 어느 쪽이든 T1–T4 값은 고치지 않는다.
- OFF 진단 boot는 Job 안 boot 순서 무작위화(§2-3)에 포함한다.

### 2-3. 무작위화·반복
한 Job 안에서 boot 순서는 등록 seed로 블록 무작위화, round 2는 round 1의 역순(ABBA). boot 안 셀 순서는 boot마다 등록 seed로 무작위. 프롬프트 토큰은 `SHA-256("cost|<term>|<bs>|<c>|<round>")`의 seed로 생성(D·모델과 무관 → D 간 같은 프롬프트). 동일 server seed.

### 2-4. N2 PROFILE(3-a) 하네스 재사용 판정 [코드 독해, 저장소 확인]
**판정: 그대로는 재사용 불가, 함수 단위 차용(복사 + 새 이름 + 새 digest)은 가능.** 봉인 파일은 한 바이트도 바꾸지 않는다.

| 대상 | 판정 | 사유 |
|---|---|---|
| `n2_static_grid/` 전체(`n2_job.py`·`n2_capacity.sh`·`n2_plan.py`·`n2_window.py`) | 불가 | 규칙 봉인(`5ea2a60`, digest `56473d6f…`)·하네스 봉인(`7e09cc3`, `9257845f…`) 대상이고 `ABORT_REG_SOURCE`가 `N2_REGISTERED.json` 바이트를 요구. D 격자가 {16,24,34,44,108}·bs ≤ 48·c ≤ 8,192·context 16,384·cap 48·config `pdmux_n3_A_agn.yml`에 묶임(`c = 15360, b ≥ 32` 구조적 도달 불가 명시). 이 PREREG의 격자(c 64k, bs 96, D=64, cap 128, 두 엔진 모드, Llama)는 전부 범위 밖 |
| `n2_loadgen.py`: `seed_tokens`·`payload`·`stream`·`lower_median`·`classify_exception`·`wait_engine_idle`·`run_profile_cell`(spec 주도) | **차용 가능** | 함수가 spec 인자로 일반화돼 있고 상수는 모듈 머리에 분리됨. `CO_INPUT = 8192`·`TRIGGER_EXTRA = 71`·`CELL_TIMEOUT_S`는 재등록 |
| `n2_loadgen.run_ealpha_cell`·`run_coshape_warmup` | **차용 가능(T3)** | 동거 형상(decode batch + 지속 prefill backlog)을 이미 구현. L {8k, 32k}·D {24, 44}로 재매개화 |
| `n2_marks.py` 마크 형식, 서버 프로세스 관리(`ProcessServer`: killpg, 로그 폐쇄) | 차용 가능 | 형식·정리 규약이 검증된 부분 |
| `n2_window.py`/`n2_probe_reduce.py` 창 추출 | **개념 차용, 코드 신규** | 봉인된 reducer를 파일 경로로 로드하는 구조라 봉인에 종속. 규칙(단독 필터, trim 16, ctx 띠, iter\_seq 연속 쌍, numpy-linear 분위수)만 새로 구현 |
| `n2_mutants.py`·`n2_job_mutants.py`·`fake_n2_engine.py`·`test_n2_*.py` | **템플릿** | 가짜 엔진 + 변이 장부 방식은 그대로 이식(가짜 엔진은 D 무관이라 D 민감도 값은 무의미, 선례의 경고) |
| 기존 N2 PROFILE 데이터(`n2p_*`) | **인용 금지, 재측정** | VERDICT\_claimE가 N2 PROFILE 계수 일체를 금지했고, 경량 확인(`CHECK_profile_a5de2d6`)만 거쳤으며 블라인드 노출 caveat(N2C-87)가 있고 cap 48·context 16k 범위다. 이 PREREG는 그 숫자를 설계에도 근거로 쓰지 않는다(격자 선택의 일부 근거는 N2 격자 관례와 KV 산술이다) |

신규 하네스는 `results/cap_char/costterms_harness/` 같은 새 디렉터리에 둔다(경로 [봉인 시 기입]).

### 2-5. 128k 프로브 스크립트 재사용 판정 [코드 독해]
**판정: 헬퍼 함수는 재사용, 스크립트는 Workspace 전용이라 Job 진입점으로 불가.** 근거: `longctx_probe_helper.py`는 단위 시험된 순수 함수(`tests/test_longctx_probe.py`)를 갖는다.

| 재사용(그대로 import 또는 복사) | 용도 |
|---|---|
| `synth_ids`·`splitmix64`·`seed_for`·`ids_sha256` | 정확히 L 토큰인 합성 입력(재토큰화 없음). 단 생성기 규칙(`splitmix64_pseudoword_text_v1`)은 결정 12′의 bench\_serving random과 다르다 — T1·T2는 **내용 무관 마이크로 측정**이라 `SHA-256` 토큰 id 직접 생성(N2 방식)을 1차로 하고, 프로브 생성기는 T2 보조 대조에만 |
| `parse_banner`·`mem_terms`·`c_mem`·`cmem_table` | T4(a)(b): 부팅 배너 → s, k, 풀 → C\_mem(L) |
| `stream_generate`·`itl_summary`·`token_gaps` | T2 클라이언트 TTFT, T1 2차 추정량(토큰 간격) |
| `log_signatures`·`sig_delta` | OOM·traceback 표지(`SERVER_DEATH` 분류) |
| `write_status`·`validate_result`·결과 json 스키마 | 단계별 PASS/FAIL 기록 형식 |

| 재사용 불가 | 사유 |
|---|---|
| `longctx_probe.sh` | Workspace 전제(수동 `install_runtime`, `/data/runs` 출력, `--budget-min`), mem 0.82 기본값, 4-boot 고정 순서, 성능 판정 없는 bring-up용. 단 **부팅 argv 구성**(arm별 노브)은 읽어서 T2·T4·Llama boot 명령의 출발점으로 쓴다 |
| `cmd_probe_g4`·`cmd_probe_g7` | G4는 감사 5번(요청 내부 p95가 정지를 가림)으로 판정에 못 쓰고 G7은 이 PREREG 범위 밖 |



## 3. 셀 격자, 실행 가능성, 계측

### 3-1. 실행 가능성 규칙(데이터 전에 등록, 부팅 로그로 평가) — rev3.4 개정(결정 28 A-1)

**규칙 [봉인]**: 셀 (bs, c)는 `bs ≤ cap` ∧ `bs ∈ 캡처 bs 목록` ∧ **`bs·(c_in + O_cell) ≤ 0.95 × KV 풀 토큰(부팅 로그)`** ∧ `c_in + O_cell ≤ context_length`일 때만 실행한다. c\_in = c − 166. 아니면 `INFEASIBLE_BY_RULE` 행으로 남긴다(실패 아님, 조용한 누락 금지).
- **근거(코드)**: `ignore_eos` 요청의 입장 검사(`schedule_policy.py` `add_one_req_ignore_eos` :641-695)는 남은 토큰(`max_new_tokens` − 생성 수)을 오름차순으로 놓고 남은 풀이 버티는지 본다. 앞선 요청이 몇 토큰을 생성했는지는 타이밍에 달려 있으므로, 타이밍 없이 성립하는 충분조건은 "아직 아무도 생성하지 않았다"는 최악, 즉 모든 요청이 입력 + `max_new_tokens` 전량을 예약하는 경우다. rev3.3의 `bs·(c + O + X)`(O+X = 498)는 출력 332 고정을 전제한 식이었다.
- **셀별 출력 길이 O\_cell [봉인]** = 332(= 16 + 300 + 16) + **ramp 상한**. ramp 상한 = burst의 prefill batch(FIFO, `max_prefill_tokens` 16,384를 넘기는 요청 뒤에서 닫힘)를 j = 1…m으로 놓고, decode가 도는 batch 2…m에 대해 Σ⌈T\_span\_ub(j) / T\_step\_lb(j)⌉ + 2/batch. T\_span\_ub(j) = (9 ms + 0.111 ms/토큰 × Σ L\_i·(1 + γ·L\_i/α)) × (108/P\_min)^1 [CM-2 floor(J1, CONFIRMED) + analytic\_terms의 attention 이차항 α·γ, prefill SM 선형 비례], T\_step\_lb(j) = a0 + b0·bs\_run + c0·bs\_run·c\_in [analytic\_terms HBM roofline 하한, 2.039 TB/s; bs\_run = 앞선 batch까지 합류한 요청 수]. P\_min = 그 셀을 등록한 J-A PD-mux 모드의 동시 prefill SM 중 최소(기본 격자 44 = 108 − 64, 16k 열·추가 셀 64 = D44·`108u`, 어느 모드도 등록하지 않은 OFF 셀 (16, 16k)은 44). **J-A는 같은 셀에 7개 모드(108v 포함) 모두 같은 O**(D 사다리 안에서 요청이 동일). J-H1(전 boot vanilla)은 O = 332 + 64(vanilla는 prefill-first라 ramp ≈ 0, 도착 지연 여유 64). 상한의 가정(prefill이 CM-2 floor의 SM 선형 비례보다 느리지 않음, σ = 1)이 깨지면 `RAMP_EXCEEDED_BOUND`로 드러난다. 값은 하네스 `ct_spec.o_tokens`가 계산하며 spec digest로 봉인된다 [유도, 측정 아님].
- **O\_cell 표 [유도, cap 128]** (J-A): bs 1 전부·(8, 2k) 332 · (8, 8k) 1,583 · (8, 16k) 2,410 · (8, 32k) 7,334 · (8, 64k) 14,089 · (32, 2k) 1,464 · (32, 8k) 6,173 · (32, 16k) 8,290 · (48, 2k) 2,078 · (48, 8k) 8,428 · (48, 16k) 11,037 · (96, 2k) 3,441 · (96, 8k) 13,300 · (64, 8k) 7,202 · (64, 12k) 10,437 · (88, 8k) 8,807 · (88, 9k) 9,926 · (16, 16k, OFF) 6,691.

cap 128 부팅의 KV 풀 [유도 895,781 토큰 = (풀 33.04 GB − 128×143.5 MB)/16,379 B; 부팅 로그가 대체; 조성 프로브 실측 dummy a4 886,804]: [CS-OK] 33.04는 풀 GB 값(우연 일치)

| bs \ c | 2k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| 1 | ○ | ○ | ○* | ○ | ○ |
| 8 | ○ | ○ | ○* | ○ (320k) | **×** (c\_in + O = 79,459 > context 73,728) |
| 32 | ○ | ○ (454k) | ○* (784k) | × | × |
| 48 | ○ | ○ (790k) | **×** (1.31M) | × | × |
| 96 | ○ (511k) | **×** (2.05M) | × | × | × |
| 64 (추가) | — | **×** (975k) | — | — | — |
| 88 (O-12) | — | **×** (1.48M) | — | — | — |

(\* = D ∈ {24, 34, 44, 108u, 108v}만. 굵은 × = rev3.3에서 ○였다가 rev3.4에서 ×.) (64, 12k) **×** (1.44M), (88, 9k) × (rev3.3에서도 ×). **셀 수 변화**: 모드별 실행 셀 rev3.3 → rev3.4 = D16·D64 14 → **12**, 나머지 5모드 21(18 + 신규 2 + O-12 1) → **15**, 합 133 → **99**(라운드당). 잃는 셀 = (8, 64k)·(96, 8k)·(48, 16k)·(64, 8k)·(64, 12k)·(88, 8k) — **shape C 보간용으로 넣은 bs 48–88 셀 전부**(rev2 감사 MAJOR 4·O-12 (b))가 다시 볼록껍질 밖으로 나간다. 실제 ctx 축은 ramp 때문에 명목보다 커진다(볼록껍질의 ctx 상단이 약간 올라감). **민감도 [유도]**: ramp 상한에 σ = 1.25를 곱하면 92셀(추가로 (48, 8k) ×), σ = 1.5면 87셀(추가로 (32, 16k) ×). 등록값은 σ = 1. J-H1 공통 5셀(bs 8 × {2k, 8k, 32k}, bs 32 × {2k, 8k})은 O = 396에서도 네 변형 모두 ○(최소 풀 a31 약 298k: (8, 32k) 264k ≤ 283k, (32, 8k) 270k)로 **불변**, Llama 칸의 ×도 불변. **이 손실은 이 문서에서 판정하지 않고 §10 O-18(사용자 결정)로 올린다.** shape C 사실 정리(rev3의 ①–③, cap\_V 83/101)는 그대로 유효하다.

### 3-2. 새 하네스 항목(차용 목록은 §2-4, 신규 구현은 §11에 나열)

### 3-3. Job 간 정렬되는 상수
cap 128(T1·T2·T3), 48(cap 불변성 boot·T4 일부), mem 0.6, context 73,728, 같은 `max_prefill_tokens` 16,384 명시, 같은 attention backend(flashinfer, Llama도 명시), `ignore_eos`, radix OFF, logprob 없음, `PDMUX_HEARTBEAT_S=0`.

### 3-4. 계측 (I-1, I-4, I-5, I-6) — 구현 사실 [코드: engsync worktree `34f2acc8` `src/multiplex/decode_probe.py`, 2026-10-20 독해]

| 계측 | 쓰이는 항 | 구현 [코드] |
|---|---|---|
| I-1 클럭·전력·throttle | 전 항(공변량) | `clk.csv`·`substrate.json` 수집은 `job_entry.sh`·`substrate_probe_helper.py`. SP는 "F-2 0행 문제 수리·셀 시작 전 생존 검사"를 요구 → 생존 검사 구현 확인, 0행이면 `SUBSTRATE_UNCHECKED` |
| **I-4 decode 커널 GPU 시간** | T1·T3(host 간격과 커널 시간 분리), §1-7 | 레코드 `decode_forward`(스키마 `pdmux.decode_probe/v1`, env `PDMUX_DECODE_PROBE_PATH`), 필드 **`decode_fwd_gpu_ms`**(`ModelRunner.forward` 전후 CUDA 이벤트 2개 사이, forward가 발행되는 스트림 — PD-mux는 green-ctx decode 스트림), 병기 `decode_fwd_host_ms`·`t_host_start`·`mode`(decode \| mixed). PD-mux와 vanilla(fused, Llama 포함)가 `ModelRunner.forward`에 건 같은 래퍼 하나로 기록된다(arm 대칭). 이벤트 쌍은 비동기 해소라 `decode_fwd_gpu_ms`가 `None`인 레코드가 있을 수 있다(미해소·overflow는 계수) → 해소 비율 < 0.99인 창은 `I4_UNRESOLVED`로 표기. `decode_iteration`의 `t_launch`/`t_sync`는 별개이며 PD-mux 전용 |
| I-5 거절 사유 | T4 | `src/multiplex/admission_probe.py`, 스키마 `pdmux.admission/v1`, env `PDMUX_ADMISSION_PATH`; 엔진 묶음에 포함 |
| **I-6 cudagraph 패딩** | T1(캡처 목록 밖 bs 방지), T4 | 같은 `decode_forward` 레코드의 **`bs_real`**(`forward_batch.batch_size`), **`bs_padded`**(그래프 히트: 재생된 캡처 bs / eager: `bs_real`), **`graph`**(`hit` \| `eager`), `bs_pad_expected`·`pad_check`(ok \| mismatch \| na; 부팅 캡처 목록 `bisect_left`에서 독립 예측), `eager_reason`(graph\_runner\_absent \| bs\_over\_max \| mode\_not\_capturable \| can\_run\_false) |
| I-2·I-3 | — | 이 PREREG에 포함하지 않음. T2는 클라이언트 TTFT가 1차라 I-3 불필요. T3는 총효과로만 쓴다(M3와 같은 범위 문장) |

vanilla 경로에서 `decode_forward` 방출이 확인되지 않으면(`NOT_COLLECTED`, G-α FAIL 조건) §1-7 fallback(`CLIENT_ONLY_UNVERIFIED`)을 쓴다.

**캡처 bs 목록과 `bs_padded` 규칙 [봉인]**: SGLang v0.5.10 기본(`server_args.py:1354-1359`, 비 spec): `[1, 2, 4, 8, 12] + range(16, 257, 8)`(= 16, 24, 32, …, 256) + …, `cuda_graph_max_bs` 이하로 자르고 max가 없으면 덧붙인다. cap 128 부팅의 목록이 `{1, 8, 32, 48, 64, 96}`를 포함해야 하며(기본 목록은 포함) 포함하지 않으면 `ABORT_CG_BS`. `bs_padded` = 목록에서 `bs_real` 이상인 최소 원소(`bisect_left`, I-6의 `bs_pad_expected`와 같은 규칙); 목록 최대 초과는 eager(`bs_padded = bs_real`). 예: 83 → 88, 55 → 56, 80 → 80, 64 → 64. PD-mux는 (stream\_idx, bs)마다 그래프를 캡처한다(`triage/notes.md:42`). 실제 목록 값은 부팅 로그의 `Capture cuda graph bs` 줄에서 읽어 모든 boot에 기록한다 — **[봉인 시 기입 불가: 부팅 값]**.


## 4. (H) hybrid 고유 항 대비 — Llama-3.1-8B (base, rev `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`)

원칙: **Llama는 이 표의 칸을 채울 때만 쓴다.** goodput 비교·모델 간 이중차분 없음. 대비 칸은 **같은 Job**(J-B) 안의 같은 엔진 모드(vanilla fused\_sym, flashinfer, cap 128, mem 0.6)에서 잰다. Nano 쪽 짝은 `108v`이다.

| 대비 칸 | Nano에서 재는 것 | Llama에서 재는 것 | 맨 끝까지 말할 수 있는 것 | 말할 수 없는 것 |
|---|---|---|---|---|
| ① 요청당 메모리 `s + k·L` | 부팅 배너 s, k, 풀(T4a) | 같은 항: s = 0, k | 부팅 로그 사실 두 줄(s·k 값), 산술 경계 `s/k ≈ 8.8k(GB)–9.4k(GiB)`[유도] | "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED, 구간 확장 금지) |
| ② 채우기 동시 수 | 채우기 실현 동시 수(T4b) L ∈ {4k, 16k} 쪽 | 같은 L 두 점(KV 한계) | "이 스택·mem 0.6·이 풀에서 L=…일 때 측정된 최대 동시 요청 수는 Nano …, Llama …이다. 가중치 바이트가 달라 풀 크기가 다르다" — 사실 병기까지 | "hybrid가 긴 L에서 배치 여유가 더 크다"를 일반 문장으로(측정 두 L·두 모델·교락 목록 안에서만) |
| ③ decode ctx 기울기 | `T_dec` 계수 c(`108v`; 엔진 측 시간 1차, 클라이언트 간격은 §1-7 대조 통과 bs만) | 같은 형 c, 짝 셀 9개: bs 1×{2k, 8k, 16k, 32k, 64k}, bs 8×{2k, 8k, 16k}, bs 48×{2k}. bs ≥ 8의 ≥ 32k 점은 Llama KV 풀 263,900 토큰으로 실행 불가(`INFEASIBLE_BY_RULE`) | 같은 (bs, c) 셀에서의 T\_dec 비와, 고정 bs(1, 8)에서 ctx 구간 기울기 비. 방향·형태 서술 | 층 종류 귀속("attention 4/56층이라 작다"는 [가설], 층 수·크기 교락), 크기 일반화 |
| ④ prefill L 의존 | T2 q (P=108, vanilla chunk −1) | 같은 q | `q` 라벨 두 개의 나란히 놓기: "Nano …, Llama … (L ≤ 64k)" (rev3.4: T2-ext 삭제, 64k 초과 점 없음) | "Llama가 64k 넘으면 초선형"(프로브 1–2회 기술값), "hybrid prefill은 선형" 일반화 |
| ⑤ cap 결합·거절 사유 | I-5 사유 몫(T4c): cap≡슬롯 | I-5 사유 몫: KV 위주(cap은 admission만) | 코드 사실(정적 풀, 설정으로 공시)과 사유 몫 기술값 | "슬롯 상한이 hybrid 고유"·"generic 스케줄러는 mamba state를 무시한다"·"Bullet/MuxWise가 C\_mem을 오예측한다" |
| ⑥ decode SM 민감도 | T1의 D 축 | **대비하지 않음**(Llama는 PD-mux 없이 `108v`) | Nano 안의 γ\_j 기술값만 | Llama와의 SM 민감도 비교 |

### 4-1. 교락 목록(공시 의무, 모든 대비 문장에 붙임)
모델 크기(9B vs 8B)·가중치 바이트(풀 크기가 달라짐, Nano 16.6 GiB vs Llama 약 15 GiB [유도])·층 수(56 vs 32)·**attention 층 수(4 vs 32 — 이것이 대비하려는 구성 자체이므로 교락이 아니라 처치와 뒤섞인 차이)**·MLP 형(Nano의 층 구성 vs gated SiLU)·hidden(4,480 vs 4,096)·RoPE(Nano attention 층은 NoPE로 알려짐[미확인]) vs RoPE·토크나이저(어휘·길이 분포; 이 PREREG는 `input_ids` 고정 길이라 길이 분포 영향은 없지만 어휘 크기에 따른 lm\_head 비용은 남음)·attention 백엔드(둘 다 flashinfer 명시)·학습/라이선스(base 동일). **Llama 한 모델 대 Nano 한 모델**이므로 "transformer 대 hybrid" 일반화 불가.

### 4-2. 허용 문장 틀
"이 Job(VESSL A100, mem 0.6, cap 128, vanilla chunk −1)에서 고정 (bs, ctx) 셀의 decode step 시간은 Nano-9B-v2 …ms, Llama-3.1-8B …ms였고, ctx 기울기는 …로 달랐다. 두 모델은 크기·층 수·attention 층 비율·MLP 형이 함께 다르다."
금지: "hybrid 고유"를 원인으로 쓰는 문장(Nano 1개 대 Llama 1개의 기술 차이까지만, §1-7), "~52k가 hybrid 경계", "9k 경계를 실측했다", 모델 간 goodput 비교 문장.


## 4′. (H) 같은 골격 조성 스윕 — 결정 24 (rev3.2, §4를 대체; §4는 Llama 참조점 칸 정의로만 남는다)

**원칙**: 서빙 비용(시간·메모리)은 가중치 값과 무관하다는 전제 아래(Nano는 MoE 아님, 출력은 `ignore_eos`로 고정), Nano-9B-v2 골격(hidden 4480·MLP·56층 고정)에서 Mamba 층 일부를 attention으로 바꾼 변형을 `--load-format dummy`로 띄워 비용 항을 잰다. 바뀌는 변수는 층 조성 하나다. **공시 의무**: 층당 파라미터가 달라(Mamba 약 147.4M, attention 약 55.1M [유도]) 가중치 바이트와 KV 풀이 함께 변한다. 그래서 H의 판정 대상은 **조성에 묶인 항(c, s, k, 동시 수, prefill 비선형성)**이고, 가중치 바이트에 묶인 고정항 a는 H 근거로 쓰지 않는다.

**변형 [봉인 시 기입: 패턴 문자열·바뀐 층 인덱스 = `scripts/compo/make_variant.py`의 variant.json]**

| 변형 | attention / Mamba 층 | 요청당 state s [유도] | KV k [유도] | L\* = s/k [유도] | 가중치 변화 [유도] |
|---|---|---|---|---|---|
| a4 (원본 골격) | 4 / 27 | 143.5 MB | 16,384 B/토큰 | ≈ 8.8k | 0 |
| a13 | 13 / 18 | 95.7 MB | 53,248 B/토큰 | ≈ 1.8k | −1.66 GB |
| a22 | 22 / 9 | 47.8 MB | 90,112 B/토큰 | ≈ 0.53k | −3.33 GB |
| a31 (같은 골격 Transformer) | 31 / 0 | 0 | 126,976 B/토큰 | 0 | −4.99 GB |

(s = Mamba 층 수 × (ssm state 128·80·128·4 B + conv 12,288·3·2 B) ≈ 5.32 MB/층, k = attention 층 수 × 8·128·2·2 B. 원본 값은 부팅 로그 실측 s 143.5 MB·k 16,382 B와 일치.)

**T1 창 규칙 [봉인, rev3.4]**: J-H1의 T1(boot A, vanilla)과 J-H2의 T1도 §1-1 "창 정의"와 같은 ramp 인식 창·실제 평균 ctx 적합·창 뒤 취소를 쓴다. J-H1은 전 boot vanilla라 O = 332 + 64 = 396(§3-1), J-H2(PD-mux)의 셀별 O는 2단계 하네스에서 같은 ramp 상한 식에 변형별 analytic\_terms 사전값(a0·b0·c0·α·γ)과 그 boot의 P(84·64)를 넣어 계산한다. 공통 비교 격자 5셀은 불변(§3-1). W-INDEP(dummy 대 real a4)·P3(c 계수)는 같은 셀의 **실제 ctx로 적합한 계수**를 비교한다.

**Job J-H (단일 GPU, 계측은 §2-2′ 매트릭스)**: 변형마다 boot 3개 — (A) vanilla `108v`: T1 bs {8, 32} × ctx {2k, 8k, 32k} 6셀, T2 L {4k, 16k, 32k, 64k}(P=108), T4 부팅 배너 + 채우기 L {4k, 16k}; (B) PD-mux sticky D=24, (C) D=44 [rev3.3: 64→44]: T1 bs 32 × ctx {2k, 8k} 2셀씩 + **T3(rev3.3, 결정 25)** bs {8, 32} × L {8,192, 32,768} 4셀씩(정의·m(L)·기록 항목은 §1-3과 같음, 셀당 반복 2), TEL ON(§2-2′ T3 전용 boot 규칙 — B·C boot 모두 같은 계측 집합이라 D24 대 D44 비는 대칭). 여기에 **실가중치 원본(real a4) boot A** 1개, **Llama-3.1-8B 참조점 boot A** 1개(§4 ①–④ 칸의 기술값; T4는 배너 + 채우기 L 4k), OFF 진단 boot 1개(dummy a4 vanilla, §2-2′ (2)). 무작위화·반복은 §2-3, 셀당 반복 4.

**gate W-INDEP [봉인]**: 같은 Job의 real a4 대 dummy a4 boot A 같은 셀에서 T1 `S_eng` 중앙값과 T2 TTFT의 `|ln(dummy/real)| ≤ ln 1.02` → `WEIGHT_INDEP_HELD`; 아니면 `WEIGHT_DEPENDENT` — 그 항(T1 또는 T2)은 조성 비교에 쓰지 않고 기술값으로만 남긴다. T4 배너 값(s, k, 풀)은 가중치와 무관한 코드 사실이라 gate 대상이 아니다.

**등록 예측 [유도, 봉인]**: (P1) 배너 s·k가 위 표의 ±2% → `COMPO_MEM_HELD`. (P2) T4 실현 동시 수가 각 변형의 산식 B(§1-4, 그 변형의 s·k·풀)와 [0.90, 1.10] → `CMEM_ACCT_HELD_a{n}`. (P3) decode ctx 기울기 c(n)이 attention 층 수에 비례: `c(n)/c(4)`가 `n/4`의 ±25% 안 → `CTX_SLOPE_PROP_HELD`, 아니면 `CTX_SLOPE_PROP_MISS`(형태를 기술). (P4) prefill 비선형성 q(n)은 n에 단조 증가 → `PREFILL_Q_MONOTONE` / 위반 시 `_NOT_MONOTONE`. **예측 없음(기술값)**: D=24 대 D=64의 decode 시간 비(SM 민감도)의 조성 의존 — Claim E 첫 화살표(층 종류별 SM 스케일링 차이)와 관련되지만 **판정이 아니라 기술값**이며, 등급 변경은 claims-auditor 감사 뒤에만 논의한다.

**프로브 반영(2026-10-21, `compo_probe/RESULT_compo_probe_2026-10-21.md`, 인용 불가 프로브 값 — 설계 입력으로만)**: 변형 a4·a13·a22·a31 모두 dummy·PD-mux·cudagraph·cap 128로 부팅, 배너 s·k·층 수가 위 표와 일치(±0.3% 안), a31도 원 argv로 부팅, TEL phase event 쌍이 a13·a31에서 방출. mem 0.6의 KV 풀 토큰이 조성에 따라 약 887k(a4)→419k→334k→298k(a31)로 줄어든다(k가 커지므로; 풀 바이트는 오히려 늘어남). **§3-1 실행 가능성 규칙(O+X = 498)을 이 풀에 적용하면 T1 6셀 중 (32, 32k)는 네 변형 모두 `INFEASIBLE_BY_RULE`**이므로, 변형 간 비교 격자는 공통 실행 가능 5셀(bs 8 × {2k, 8k, 32k}, bs 32 × {2k, 8k})로 고정한다 [봉인]. 실제 풀 값은 측정 Job의 부팅 로그로 다시 평가한다. 측정 Job은 이미지에 `libnuma1`이 필요하다(프로브 시도 0의 `sgl_kernel` import 실패 원인; 기존 Job 경로에서의 처리 여부를 봉인 전 확인).

**rev3.3 Job 분할**: **J-H1** = A boot 전부(변형 4 + real a4 + Llama + OFF 진단; W-INDEP·P1–P4는 J-H1 안에서 판정), **J-H2** = B·C boot 전부(변형 4 × 2; SM 민감도 기술값과 T3 조성 의존은 J-H2 안에서만 비교). 두 Job 사이의 수준값은 비교하지 않는다(§0).
**T3 조성 의존 [rev3.3, 기술값 — 예측 등록 안 함]**: 변형별 ln ρ·ln σ·q_stall 셀 표. ρ는 동거·전력 한도 클럭·부하 의존 관측자 비용의 합성값이라(CONSENSUS §1-38·§8) **변형 간 차이를 '간섭의 조성 의존 크기'로 쓰지 않는다**. 변형마다 가중치 바이트·전력 상태가 달라 클럭 공변(I-1)을 창 단위로 병기한다. a31(Mamba 0층)은 동거 대상 decode가 attention 전용이라 σ 해석이 다르다는 점을 표에 표시한다.

**허용 문장 틀**: "같은 Nano-9B-v2 골격에서 attention 층을 4→13→22→31개로 바꾸면(무작위 가중치, 가중치 바이트도 함께 변함), 이 Job에서 요청당 고정 state는 …→0 MB, KV는 …→… B/토큰, decode ctx 기울기는 …배, 동시 요청 수는 L=16k에서 …→…였다." **금지**: "hybrid 고유"를 원인으로 쓰는 문장 중 조성 외 변수(가중치 바이트)를 공시하지 않은 것, dummy 변형의 출력 품질·goodput 언급, a31을 "실재 Transformer의 측정값"으로 쓰는 것(실재 참조점은 Llama 칸뿐), W-INDEP 미통과 항의 조성 비교.

## 5. TP2 항 (N 항)

TP2 PD-mux는 M2G J2에서 정확성 gate를 통과했다(G0(ii)·G1·G3 PASS, `G2_NOT_TESTED`) [측정, `RESULT_m2g_rev11_jobs_2026-10-13.md`]. 이 PREREG는 TP2에서 **비용 항 측정**과 **샤딩 사실 확인**을 한다. 성능 비교(G\_sym 등)는 하지 않는다.

### 5-1. 샤딩 사실 확인 항목
| # | 확인 | 예측 [유도, config 산술] | 측정 방법 | 판정 |
|---|---|---|---|---|
| F1 | 슬롯당 mamba state s/N | N=2에서 약 71.8 MB/rank(143.5/2); 128 mamba heads·8 groups가 2로 나누어떨어짐 | TP2 부팅 배너 rank별 슬롯 바이트 | rank 0·1 일치 ∧ 예측의 ±2% |
| F2 | KV B/토큰 k/N | KV head 8개(k = 4 layers×2×8×128×2 B = 16,384 B와 일치) → 4개/rank → 약 8,190 B | 부팅 배너 rank별 | ±2%. (N=4 → 2개, N=8 → 1개, N>8은 복제. **N=2만 측정**) |
| F3 | rank당 풀과 C\_mem(TP2, L) | 산식 B `풀_rank / (s/2 + (k/2)·(L+R))`(R = 채우기 출력 4,096/8,000 `ignore_eos`), 원식 A는 R = 0; 가중치가 절반이라 풀이 약 39.7 GB/rank [유도] → 원식 A의 C\_mem(32k)가 약 117(N=1은 약 49), 산식 B 값은 부팅 로그 풀로 데이터 전에 계산해 봉인 | 부팅 + 채우기(T4, L ∈ {16k, 64k}, cap 128) | `CMEM_ACCT_HELD` 형(§1-4, 구현 사실 확인) |
| F4 | in\_proj 폭과 conv 커널 stride | 폭 22,656 → 11,328/rank(`2×5120 + 2×512 + 64`). int32 오프셋 경계 `2^31 / stride`: N=1 94,786(관측 T ≥ 94,793과 일치) → N=2 **189,572** [산술] | (a) 첫 prefill에서 conv 입력 `x.stride()` 로그(하네스·엔진 로그, [봉인 시 기입: 로그 필드명]) (b) gate의 커널 단위 시험에 stride 11,328 변형 추가 (c) **TP2 서빙 96k 1점(Job 마지막)** | stride 로그가 11,328이면 F4 입력 확인. 96k 점의 의미는 §5-3 |
| F5 | 헤드 분할 가능성 | attention heads 40 → 20, KV 8 → 4, mamba heads 128 → 64 | TP2 부팅 성공 + F1·F2 | 부팅 실패면 이후 항 전부 `TP2_NOT_BOOTABLE` |

### 5-2. 측정 셀 (J-C, `resourcespec-a100x2`, 두 GPU 동시 사용)
| 항 | TP2 셀 | 같은 Job의 N=1 앵커 | boot 수 |
|---|---|---|---|
| T1 | D ∈ {44, 108v} × bs ∈ {8, 32} × c ∈ {2k, 8k, 32k} = 12셀. (bs 48은 N=1 앵커가 (48, 32k)에서 KV 풀 1.59M 토큰 한계라 뺌) | 같은 12셀, cap 48 | TP2 boot 4(D×round 2), 앵커 4 |
| T2 | L ∈ {4k, 16k, 32k, 64k} × P ∈ {64, 108}(TP2/앵커 각각, T1 boot 안) | 같음 | 위 boot 재사용 |
| T4 | 부팅 로그(F1–F3) + 채우기 L ∈ {16k, 64k}, cap 128 | 해당 없음(N=1 사실은 J-A·J-B 몫이며 Job 간 비교 금지, 산식 대조로만; 산식은 §1-4의 **산식 B**(출력 예약 R 반영, rank당 풀·s/2·k/2)) | TP2 boot 1 |
| 마지막 | 96k 1점 [제안: 통과하면 124k 1점 추가] | — | TP2 boot 1(Job의 마지막 boot) |

TP2 항에서 내는 양: (i) 같은 셀의 `T_dec(TP2)/T_dec(N=1)` 비와 계수 `a', b', c'` 대 `a, b, c` (ii) T2의 `T_span(TP2)/T_span(N=1)` (iii) F1–F5. **통신(all-reduce) 시간을 따로 분해하지 않는다**(profiler 불가, ncu는 green-ctx 하에서 exit 9). 총합 비로만 쓴다. rank lockstep 지연(rank 간 launch 시각 차)은 rank별 telemetry가 있을 때만 기술값 [미등록 — 엔진 확인].

### 5-3. 96k 점의 의미(충돌 지점, 사용자 결정 O-4)
결정 15가 int64 패치(`chunk_offset`·`sequence_start_index` 두 곳 — `chunk_offset`만으로는 conv\_state 꼬리 읽기가 남는다)를 기본 digest에 넣으면 **패치가 켜진 TP2 96k 점은 경계 이동 시험이 아니다**(패치가 있으면 N=1에서도 통과해야 한다). 경계 이동(94,786 → 189,572) 예측의 시험 경로는 둘이다.
- (가) **권고**: gate의 커널 단위 시험에서 패치 전 커널 사본(두 곳 모두 패치 전)으로 stride 11,328·T ∈ {95,100, 126,976, 189,000, 190,500}을 돌린다(GPU 1장, 서버 없음, 서버 사망 위험 없음). TP2 서빙 96k 점은 패치 ON의 정상 점(T2 확장, **int64 gate 통과 조건부**, O-3)으로 쓴다. 토글이 필요 없다.
- (나) 엔진에 패치 토글 env를 두어 TP2에서 패치 OFF 96k를 시험한다. 서버가 죽을 수 있다(`SERVER_DEATH`, 가장 마지막). 토글은 digest·정확성 gate를 늘린다.
어느 쪽이든 96k 점은 **Job의 마지막 boot**에 둔다(죽어도 앞 결과가 회수된다). 허용 문안 상한: "TP2에서 stride가 …로 로그되었고 96k prefill이 통과/실패했다. 경계 산술이 맞았는지는 (가) 시험 결과로만 말한다."



## 6. (이월)

V는 항 데이터 이후 별도 PREREG(새 감사)로 다룬다. 데이터를 본 뒤 구조를 고르는 위험은 그 PREREG에서 공시한다(결정 17). (절 번호 6은 rev2와의 대조를 위해 비워 둔다.)

## 7. Job 구성, 시간, GPU-h (항 측정 Job만; Job마다 ≤ 6 h 목표; 사용자 메모 "GPU Job 적정 크기": 동시 사용 GPU 수에 맞춘 스펙과 사용률)

### 7-1. 시간 모형 [유도 — 측정 아님, Job 크기 결정용]

boot 200 s, boot별 warm-up 60 s(=boot당 0.072 h), 셀 오버헤드 15 s, decode 창 300 token, T1 셀실행 평균 약 50 s [임의]. 단일 요청 prefill TTFT [기술값, 프로브·J1]: 1k 0.12 s, 2k 0.24, 4k 0.46, 8k 0.92, 16k 1.84, 32k 4.0, 64k 8.4. prefill SM 감소 배율 `(108/P)^0.8` [임의 중심값, T2 전에 모름]. step 시간 `(12 + 0.45·bs) ms × (108/D)^0.5` [임의]. TP2 prefill 속도 1.7× [임의], TP2 boot 400 s. V 셀 평균: 자연 125 s, shape C 370 s [rev0 모형 승계, 임의]. T4 채우기 1회 약 4–8 min [임의]. **모든 시간은 Job 크기 결정용이며 측정 전 값이다.**

### 7-2. 분할 원칙
- **반복은 여러 Job**(운영 모델): J-A는 **round별 2 Job**으로 나눈다. D 사다리(비교 단위)는 각 Job 안에서 완결되고(모든 D·`108u`·`108v`가 각 Job에 있음) round 간 차는 Job 효과를 포함한다 — `108v`·`108u` 같은 셀의 J-A1/J-A2 차가 Job 효과의 참고치다(§6-7과 공유). round 2는 round 1의 역순(ABBA).
- 목표 상한 6 h: 모든 Job 기대 시간 ≤ 4.4 h(하드캡 ×1.2 ≤ 5.3 h). 정지 선례(J2·J4·J8이 약 9.5 h에서 외부 정지, 원인 미확정)보다 충분히 아래다. 재개 가능(셀 단위 DONE·`vessl_jobs.tsv`·sha).

### 7-3. Job 표 (재확인 2026-10-20, rev2 수치와 같음)
| Job | 내용 | 스펙 | boot | 셀/반복 | 예상 h [유도] | 하드캡(×1.2) [제안] | 예약 GPU-h | 사용 GPU-h(사용률) |
|---|---|---|---|---|---|---|---|---|
| **J-A1** Nano D 사다리 round 1 | T1(D 16·24·34·44·64 + 108u + 108v, round 1) + T2·T2′ + T3(D 24·44, round 1분) + cap 불변성 boot(D=44, cap 48) | `resourcespec-a100x1` | 8 | T1 128 셀실행(고유 셀 128 = 118 + 신규 10) · T2 42 셀 + T2′ 14 · T3 16 실행 | 3.7 | 4.4 | 3.7 | 약 3.4 (92%)† |
| **J-A2** Nano D 사다리 round 2 | 같은 boot 7개, 역순(cap 불변성 boot 없음) | `resourcespec-a100x1` | 7 | T1 128 · T2 42 + T2′ 14 · T3 16 | 3.5 | 4.2 | 3.5 | 약 3.2 (92%)† |
| **J-B** Nano-Llama 대비 + T4 | Nano `108v` 4 round + Llama 4 round(T1+T2) + **T4 Nano 9 boot(12 fill)** + T4 Llama 1 boot(2 fill) | `resourcespec-a100x1` | 18 | T1 Nano 80(신규 (64,·) 셀 8 포함)·Llama 36 셀실행, T2 48 셀 | 4.4 | 5.3 | 4.4 | 약 4.0 (91%)† |
| **J-C** TP2 | 샤딩 확인 F1–F5, T1·T2 TP2/앵커, T4 TP2, 마지막 96k | `resourcespec-a100x2` | 10 (TP2 6 + 앵커 4) | T1 96 셀실행 · T2 32 셀 | 2.5 | 3.0 | 5.0 | 4.0 (81%)‡ |
| **합계(항 측정)** | | | 43 | | | **약 19.9 GPU-h(캡 합)** | **약 16.6 GPU-h** | 약 14.6 |

† 단일 GPU Job: 측정 중 GPU 점유는 거의 100%, 부팅·클라이언트 대기·drain 때문에 평균 약 90–95% [유도]. ‡ TP2 Job: TP2 boot는 2장 모두, N=1 앵커 boot는 1장만 쓴다(`(2×TP2 시간 + 1×앵커 시간)/(2×Job 시간)`). 개선안(O-6): 앵커를 두 GPU에서 독립 서버 두 개로 동시에 돌려 사용률을 약 100%로 올릴 수 있으나 전력·열을 공유해 두 앵커가 서로 교락한다(J1 GPU0 throttle 61% 선례).

**재확인 내역(CPU 산술)**: ① 시간 유도 — J-A1 = rev1 J-A 6.4 h의 반 3.2 + 신규 (64,·) 10 고유 셀 × 50 s ≈ 0.14 + T2′ ≈ 0.15 + cap 48 boot ≈ 0.2 = 3.69 h → 3.7. J-A2 = 3.2 + 0.14 + 0.15 = 3.49 → 3.5. J-B = rev1 3.6 + T4 판별·고정점 부팅 4개 + 채우기(4 × 0.072 + 4 × 약 6 min ≈ 0.7) + 신규 `108v` 셀 0.1 = 4.4 h. J-C 2.5 h × 2장 = 5.0 GPU-h. ② 셀 수 — J-A: 14·14(D 16·64) + 18×5(24·34·44·108u·108v) = 118, 신규 (64,8k)·(64,12k) × 5모드 10 → 128 고유 셀 ✓; T2 6L × 7모드 = 42, T2′ 2L × 7 = 14, T3 8셀 × 2회 = 16 ✓. J-B: `108v` 20셀(18 + 신규 2) × 4 round = 80 ✓, Llama 9셀 × 4 = 36 ✓, T2 6L × 4 round × 2모델 = 48 ✓, boot = 4 + 4 + 9 + 1 = 18 ✓(T4 채우기 12 + 2). J-C: T1 12셀 × (TP2 + 앵커) × 2 round × 2회 = 96 ✓, T2 4L × 2P × 2 × 2 round = 32 ✓, boot 6 + 4 = 10 ✓. ③ 합계 — 예약 3.7 + 3.5 + 4.4 + 5.0 = **16.6** (= rev2 27.8 − V 11.2), 캡 합 4.4 + 4.2 + 5.3 + 3.0×2 = **19.9** (= rev2 33.3 − V 13.4). ④ 이 합계 밖 의존 비용: 분할 GPU gate G-α→G-β→G-γA∥G-γF **약 9.0 GPU-h(하드캡 합 13.0)**(`GATE_SPLIT_DESIGN`), Workspace 하네스 smoke(`pdmux-probe`, 비측정, 상한 약 1.5 h [유도]), O-12 채택 시 J-A당 약 +0.14 GPU-h. ⑤ 모든 시간은 측정 전 임의 모형이므로 gate·smoke에서 boot·셀 시간 실측이 나오면 Job 분할은 **봉인 전에** 재확인한다.

정지 시: 이미 끝난 셀은 회수하고 나머지는 새 job name으로 재실행(실패 셀만 재실행한 사실은 결과 보고에 적는다). J-A를 합치면 약 7.2 h로 6 h 목표를 넘으므로 round별 분할이다. D 사다리는 각 J-A Job 안에서만, H 대비(Nano `108v` 대 Llama)는 J-B 안에서만 성립한다. `108v`가 J-A1·J-A2·J-B에 모두 있어 Job 효과 진단값을 겸한다.

### 7-4. rev3.1 재계산 (결정 18·22 반영) [유도 — §7-1 시간 모형, 측정 아님]

| Job | 추가분 | 예상 h | 하드캡(×1.2) | 예약 GPU-h |
|---|---|---|---|---|
| J-A1 | O-12 셀 10(+0.14) · T3 boot 분리 +2 boot(+0.14) · OFF 진단 boot 2(각 약 0.18, +0.36) | 3.7 → **4.3** | 5.2 | 4.3 |
| J-A2 | 같음(+0.64) | 3.5 → **4.1** | 5.0 | 4.1 |
| ~~J-B1~~ → **J-H1** (결정 24, rev3.3) | A boot 7개(변형 4 + real a4 + Llama + OFF 진단) [유도: boot 0.5 h + 셀 약 0.38 h × 6 + OFF 0.11 h] | **약 2.9** | 3.5 | 2.9 |
| **J-H2** (결정 25, rev3.3) | B·C boot 8개(변형 4 × D {24, 44}): T1 2셀 + T3 4셀 × 반복 2 [유도: boot 0.58 h + T1 0.22 h + T3 약 0.27 h × 4] | **약 1.9** | 2.3 | 1.9 |
| J-B2 (Nano T4 9 boot + O-8 fused\_sym 1 boot) | O-15 (b) 분리, 셋업 1회(+0.07) | **1.1** | 1.3 | 1.1 |
| J-C | OFF 진단 TP2 boot(boot 400 s + 셀 3, +0.15 h × 2장) [O-14] | 2.5 → **2.7** | 3.2 | 5.4 |
| **합계** | | | **약 23.7 GPU-h(캡 합)** | **약 19.7 GPU-h** |

- J-B는 결정 23(O-15 (b))으로 J-B1·J-B2로 나눴다. T4 분량 0.7 h는 §7-3 재확인 ①의 값이다. J-B2의 T4 산식 대조는 boot 단위라 H 대비와 Job이 달라도 영향이 없고, O-8 대칭 점검(fused\_sym 대 PD-mux T4)은 J-B2 안에서 성립한다.
- OFF 진단 boot 시간(0.18 h = boot·warm-up 0.072 + T1 셀 6 × 50 s + T2 2L × 8회)은 §7-1의 임의 모형이다.


### 7-4′. rev3.4 재계산 (결정 28) [유도 — 측정 아님, Job 크기 결정용]

**모형**: §7-1의 값을 셀 단위로 조립했다 — T1 셀 = Σ prefill batch 시간(batch 1은 108 SM, 나머지는 그 모드의 동시 prefill SM P에서 §7-1 TTFT 표 × (108/P)^0.8; vanilla는 전부 108) + 합류 뒤 318 step × `(12 + 0.45·bs)·(108/D)^0.5` ms + 셀 오버헤드 15.5 s; T2 = 8 × TTFT(L) + 15 s; T2′ = 8 × TTFT(L)·(108/P)^0.8 + 16 s; T3 셀 = (decode batch ramp + m(L) × TTFT(L, P) + 300 step) × 반복 2; T4 채우기 6 min/회. 실행 셀은 §3-1 rev3.4 규칙. 가짜 엔진의 ramp는 등록 상한의 약 25–45%였다(가짜 엔진 값, 인용 불가). **비교를 위해 rev3.3 설계를 같은 모형에 넣으면** J-A1 4.69 h·J-A2 4.51 h·J-H1 3.37 h다(§7-4의 4.3·4.1·2.9는 "T1 셀 평균 50 s [임의]" 모형이라 직접 비교하지 않는다). 셀 시간은 ramp 인식 창으로 거의 변하지 않는다 — rev3.3도 ramp 동안 prefill을 기다렸고(그때는 앞선 요청이 먼저 끝나 창이 깨졌을 뿐), 합류 뒤 step 수는 332 → 318이다. J-A가 줄어든 이유는 비싼 셀((96, 8k)·(48, 16k)·(64, ·)·(88, 8k)·(8, 64k))이 실행 불가가 되고 sticky-D boot의 T2(6 L)가 빠진 몫이 T2′ 확장(2 → 6 L)보다 크기 때문이다.

| Job | rev3.4 변경 | 예상 h | 하드캡(×1.2) [제안] | 예약 GPU-h |
|---|---|---|---|---|
| J-A1 | 셀 133 → 99, sticky-D T2 삭제, T2′ 6 L × 7 boot, T3 boot 2개 포함 | **3.4** (1단계 2.88 + T3 0.48) | 4.0 | 3.4 |
| J-A2 | 같음(cap 불변성 boot 없음) | **3.3** (2.76 + 0.48) | 3.9 | 3.3 |
| J-H1 | 창 규칙만(O 396, 셀 불변) | **3.4** | 4.1 | 3.4 |
| J-H2 | T1 2셀·T3의 decode batch ramp 반영(a4 속도로 근사) | **2.2** | 2.6 | 2.2 |
| J-B2 | 변경 없음 | 1.1 | 1.3 | 1.1 |
| J-C | 변경 없음(단, T1 셀 실행 가능성은 2단계에서 rev3.4 규칙으로 재평가 — §10 O-19) | 2.7 | 3.2 | 5.4 |
| **합계** | | | **약 22.3 GPU-h(캡 합)** | **약 18.8 GPU-h** |

모든 Job의 기대 시간이 4.4 h 목표 안이라 **Job 분할은 필요 없다**. `costterms_job.sh`의 하드캡을 J-H1 245 min · J-A r1 243 min · r2 234 min으로 고쳤다(J-B2 78 min 불변). 모형 값이므로 Workspace smoke에서 boot·셀·ramp 시간을 실측하면 봉인 전에 다시 계산한다(§12 항목 11).

## 8. 선행 조건 순서

1. **GPU correctness gate — 분할 gate G-α·G-β PASS + G-γ 특성화 종결** (결정 16, rev3.1에서 결정 22로 개정: G-α PASS 28/28·G-β PASS(engsync, 2026-10-20); G-γA `OBSERVER_EFFECT`·G-γF `UNRESOLVED_POWER`는 PASS가 아니며 "관측자 효과 특성화 완료"로 닫았다. 계측 효과는 §2-2′로 공시한다). 앞 단계가 PASS일 때만 다음 단계를 제출한다. 세부는 `GATE_SPLIT_DESIGN_2026-10-20.md`(재서술하지 않음). 엔진 묶음 = I-5 동기화 + Triton causal\_conv1d **int64 패치(`chunk_offset`·`sequence_start_index` 두 곳)** + I-4·I-6; 브랜치 `engine-sync-i5-int64` `34f2acc8`, 예상 ENGINE\_DIGEST `99bbef91…`(30항목). 사후에 digest를 바꾸면 gate는 G-α부터 다시다. 이 PREREG가 gate에 요구하는 것: ON/OFF 출력 일치, 관측자 TOST(±1.5%, 계열별 — rev3.1: 미충족, 결정 22로 특성화 대체), 변이 시험(I-4 host 시간 변이가 FAIL해야 함), 패치 반증(패치 후 fused\_sym 96k·124k 통과 + 되돌리면 95,100 재실패 + 단위 시험(12288, 95100)·stride 22,656 + 94,784/94,800 경계), stride 11,328 변형(§5-3 (가)), **vanilla 경로(Nano·Llama)에서 `decode_forward` 방출**(§1-7), 실현 SM 로그(§1-8). **96k·124k 확장(T2-ext, TP2 96k)은 int64 gate 통과 조건부(O-3)** — rev3.1: 패치 반증은 G-β PASS로 충족(γ와 무관). rev3.4: T2-ext는 삭제(결정 28 A-6); TP2 96k 점은 context 73,728과 충돌하므로 2단계에서 처리(§10 O-19).
2. **하네스**: 새 하네스(§11) CPU 시험·변이·가짜 엔진 e2e → Workspace smoke(비측정, 수치 인용 금지).
3. **감사**: 항 측정은 결정 17에 따라 추가 규칙층 감사 없이 봉인 후보. 하네스층 감사는 별도(최대 2회).
4. **봉인 S1**: 규칙·하네스·엔진 digest(gate 통과 digest)·config digest. 이 문서의 [봉인 시 기입] 칸을 채운다.
5. **CPU preflight** → `launch.sh` dry-run(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit`. J-A1·J-A2·J-B·J-C는 독립 제출 가능(gate 전부 PASS 뒤). B5 선례(운영 모델): dry-run 때 확인.
6. **fetch**(`fetch.sh`, DONE + sha256) → result-analyst가 T1–T4 맞춤(분석 코드는 fetch 전에 커밋, 맞춤 규칙은 이 문서).

**R16-3′**: SP R16-3의 "후보 {48, C\_mem/2, C\_mem}에서 G0 cap 결정 규칙 [미등록]"은 cap 선택을 goodput 스윕이 아니라 C\_mem 산식(예약 반영)으로 대체하며(결정 16 O-1 승인), **T4가 R16-3(cap 특성화 Job)을 대체한다.** R16-3의 CM-5 검증 기능은 T4의 "구현 사실 확인"(산식 B)과 거절 사유 몫이 맡는다. (R8-2′ 본문과 G0 튜닝은 V PREREG로 이월.)

## 9. 라벨과 금지 문안

### 9-1. 라벨
항: `FORM_INADEQUATE_T1_D{d}`·`FORM_INADEQUATE_T2_P{P}`·`CAP_INVARIANT`/`CAP_DEPENDENT`·`CLOCK_SENSITIVE`·`PREFILL_SUPERLINEAR_SCOPED`/`PREFILL_NEAR_LINEAR_SCOPED`/`Q_UNRESOLVED`·`T2_J1_SLOPE_DEVIATES`·`RHO_D_INTERP`·`RHO_CLAMPED`·`SIGMA_NOT_COLLECTED`·`T3_WINDOW_SHORT`·**`CMEM_ACCT_HELD`/`CMEM_ACCT_MISS`/`CMEM_ORIG_OVERPREDICTS`**·`C_MEM_DRIFT`·`FILL_INCOMPLETE`·`INFEASIBLE_BY_RULE`·`TP2_NOT_BOOTABLE`·`CLIENT_ENGINE_AGREE_bs{b}`/`CLIENT_ENGINE_MISMATCH_bs{b}`/`CLIENT_ONLY_UNVERIFIED`·`SM_NOT_REALIZED_D{d}`/`T2PRIME_L_OUT_OF_RANGE`/`STICKY_MODE_MISMATCH`·`I4_UNRESOLVED`. **rev3.4 추가**: 창 상태 `FOREIGN_RESIDUAL`·`WINDOW_SHORT`·`WINDOW_CONTAMINATED`·`RECORDER_UNPROVEN`·`CG_UNPROVEN`·`PAD_MISMATCH`·`BENCH_INVALID`·`RETRACT`, 창 라벨 `CTX_OFF_NOMINAL`·`RAMP_EXCEEDED_BOUND`, T2′ `T2P_COEXIST_CONFIRMED_L{L}`/`T2P_COEXIST_UNCONFIRMED_L{L}`·`FORM_INADEQUATE_T2P_P{P}`. **rev3.4 폐지**: `T2_P_NOT_REALIZED`·`T2_FALLBACK_BS1_COEXIST_D{d}`(모든 sticky-D boot가 T2′), `CTX_UNREALIZED`(띠는 라벨로). 승계: SP 계열 `SUBSTRATE_WARNING`(드라이버 변경)·`SUBSTRATE_UNCHECKED`·`SERVER_DEATH`(죽은 (arm, cap, L)은 `UNRESOLVED(arm_death)`로 두고 다른 arm 결과를 지우지 않으며 일반화하지 않는다)·`CLIENT_ERROR`·`SERVER_ERROR_HI`·`NOT_COLLECTED`(I-4·I-6, vanilla 경로 미방출 포함).

### 9-2. 금지 문안 (승계 + 이 PREREG)
- "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED, 구간 확장 금지; 긴 L의 방향은 T4 측정 후 ②의 틀로만)
- "~52k 경계"/"~52k에서 구속 전환", "9k 경계를 실측했다"(유도값), "~52k가 hybrid 경계"
- "PD-mux는 64k를 넘을 수 없다"·"hybrid PD-mux는 64k 불가"·"OOM 때문에 64k"(0.6의 95k+ 실패는 OOM이 아님; 64k는 격자 선택, 한계는 chunk −1 Nano arm 공통, 프로브 감사 허용 문안 3–4), "96k 실패는 SSD 커널 오류", "PD-mux 고유 한계", "0.82에서 PD-mux가 먼저 OOM 난 것은 split-prefill 때문", 0.82 결과와 0.6 결과의 혼합
- "Bullet/MuxWise 오예측"·"Bullet/MuxWise가 C\_mem을 오예측한다"·"generic 스케줄러는 mamba state를 무시한다"·"슬롯 상한이 hybrid 고유"·"Claim E 반증"
- **T4: "비용 모델이 C\_mem을 예측했다/검증했다"(산식 B는 엔진 회계의 재현 — 구현 사실 확인)**, 원식 A를 B와 같은 비중의 경쟁 가설로 서술
- J1 크기 문장: J1 B·X arm 비의 크기, CM-6 "처리율↔ITL 역순"을 쓰인 그대로, "PD-mux가 클럭을 낮춘다" 일반 가정, fused 정지 꼬리를 PD-mux 구조 이득으로 귀속
- G4를 "ITL SLO 충족"으로 읽는 것(요청 내부 p95가 정지 1회를 가림), "G7이 합성 생성기 동등성을 보였다"
- N2 PROFILE 계수·"shape B ctx 항 1%"·"shape C에서 hybrid 항이 유의하다"·s\_attn/s\_ssm 증거로서 γ\_j·T1 D 비
- T3: ρ를 간섭 크기로, ρ에서 정책·재배분 경계·굶김 연쇄·HE0 원인 도출, 중앙값 ρ만으로 "꼬리가 없다"
- H 대비: "hybrid 고유"를 원인으로 쓰는 문장(§1-7), 클라이언트 간격만으로 Llama·108v 값을 단정(불일치 bs 제외)
- 선행 gate 전: "I-5가 거절 사유를 측정했다"(동기화·GPU gate 전), "preflight PASS가 이 하네스를 검증했다"


추가 금지(결정 22): "엔진 묶음이 관측자 TOST를 통과했다"·"G-γ PASS"·"계측은 측정에 영향이 없다"·"T1은 계측 없는 decode step 시간이다"·분해 Job 값(λ −1.7%, ITL p95 +1.0%)을 다른 arm·모델·부하의 보정 계수로 쓰는 것.

추가 금지(V 분리에 따른 것): "이 항 값으로 G·λ\*\_SLO·goodput을 설명/예측했다"(조립은 이 문서 밖), "항 측정이 비용 모델을 검증했다".

추가 금지(결정 28, rev3.4): "T1은 명목 ctx(2k·8k·…)에서 잰 값이다"(적합 입력·인용 ctx는 창 안 실제 평균 ctx, 명목 c는 셀 라벨) · 실제 ctx 병기 없이 "(bs, 8k) 셀의 decode step은 …ms" · "T2는 P = 108 − D에서의 단독 prefill 시간이다"/"T2로 prefill SM 민감도를 쟀다"(T2는 P = 108만, SM 축은 T2′) · T2′를 "다른 부하 없는 prefill 시간"으로 서술(bs=1 decode 동거 공시 문장 없이) · T2′ − T2를 "SM 감소 효과"로만 읽기(동거 효과가 섞임) · `T2P_COEXIST_UNCONFIRMED` 점(특히 `108v`)을 동시 실행 상태의 값으로 인용 · "ramp 상한이 실제 ramp를 예측했다"(사전값에서 유도한 보수적 상한일 뿐) · 가짜 엔진의 VALID 셀 수·ramp 비율을 실엔진 기대치로 인용 · rev3.3 설계로 잰 값과 rev3.4 값을 같은 표에 섞기.

## 10. 열린 결정 (항 측정 해당분, 사용자 몫, 권고 병기)
**닫힌 것**: 결정 16 O-1·O-2·O-3, 결정 17(V 분리), **결정 18(2026-10-20): O-4 (가)·O-6 순차·O-8 포함·O-9′ [제안]값 봉인·O-10′ 유지·O-12 (b)·O-13 확장 안 함** — 아래 각 항목은 권고 원문 기록용이며 모두 권고안대로 닫혔다. 결정 22(계측 정책) → §2-2′. V 전용 O-5·O-7·O-9(V 부분)·O-10(k=5·소거 변형)·O-11은 V PREREG로 이월.
- **O-4** TP2 96k 점의 의미: (가) gate 단위 시험(두 곳 패치 변형)으로 경계 이동 + 서빙 96k는 정상 점 / (나) 패치 토글 env. **권고 (가)** — 서버 사망 위험·digest 증가 없음.
- **O-6** TP2 앵커: 순차 1장(사용률 81%) / 두 GPU 동시 독립 앵커(≈100%, 교락 공시). **권고 순차** — 앵커가 TP2 대비 비(분모)이므로 전력·열 교락이 곧 estimand 오염이다. 비용 차는 작다.
- **O-8** T4 arm 대칭 점검(fused\_sym 한 boot 추가, +약 0.3 GPU-h). **권고 포함** — 산식 B가 "엔진 일반"이 아니라 PD-mux 부팅 한 경로의 사실임을 막는 값싼 대칭 점검.
- **O-9′** 항 측정 채택 기준 T1(중앙 ≤ 3%·p90 ≤ 8%·외삽 ≤ 10%)·T2(≤ 5%)·CLOCK\_SENSITIVE(노출 > 20%·계수 변화 > 3%): **권고 [제안] 그대로 봉인** — 라벨만 붙고 어떤 판정도 이 값에 걸려 있지 않다.
- **O-10′** 라운드 수: 기본 2, `108v`·Llama 4. **권고 유지**(H 대비의 판정 하한 n ≥ 4 충족, D 사다리는 기술 적합).
- **O-12 (신규)** shape C 영역 T1 보강: 감사 2회차는 shape C(bs ≈ 83 → `bs_padded` 88, c̄ ≈ 9.1k)가 T1 볼록껍질 밖(bs 96의 c\_max 8,192)이 확실하다고 지적했다. 선택: (a) 보강 없음 — 항 데이터를 본 뒤 필요하면 새 Job(감사 선택지 (b)의 추정 약 +1 GPU-h) / (b) 지금 `bs=88` 줄 (88, 8k)·(88, 9k) × 5모드(24·34·44·108u·108v) 10셀을 J-A1·J-A2에 추가(각 Job 약 +0.14 GPU-h, 봉인 전에만 데이터 영향 없이 가능; (88, 9k)는 O+X = 498에서 88×(9,216+498) = 855k > 852k라 `INFEASIBLE_BY_RULE`이 될 수 있어 (88, 8k)와 (80, 9k)로 대체 가능). **권고 (b)** — 데이터 전 추가는 구조 선택 편향이 없고, 후에 필요해진 뒤에는 Job을 새로 내야 한다.
- **O-14 (rev3.1 신규, 결정 23: 포함)** J-C OFF 진단 boot(+0.3 GPU-h): 포함 / 빼고 "TP2 계측 효과 미측정" 공시. **권고 포함** — 결정 22가 Job 내 짝을 요구하고 TP2 경로의 계측 효과는 N=1과 다를 수 있다.
- **O-15 (rev3.1 신규, 결정 23: (b))** J-B 4.9 h(목표 4.4 초과): (a) 그대로(하드캡 5.9 h < 6 h) / (b) T4 boot 10개를 J-B2로 분리(H 대비와 T4는 서로 다른 Job이 되나 T4 산식 대조는 boot 단위라 영향 없음, Llama T4 1 boot는 H 대비 ⑤ 때문에 J-B1에 남김). **권고 (b)** — 외부 정지 선례(약 9.5 h)보다 아래라도 재개 단위가 작을수록 잃는 셀이 적다; 예약 합계는 거의 같다(boot 공통 오버헤드 없음).
- **O-16 (결정 26: 포함; rev3.2 신규; rev3.3에서 B·C boot는 T3 때문에 필수가 되어 D 축 비용은 T1 2셀분만 남음)** J-H의 D 축: 포함 / 제외. **권고 포함** — Claim E 첫 화살표(층 종류별 SM 스케일링)에 대한 유효 측정이 0건인 상태에서 같은 골격 조성으로 재는 유일한 경로다(기술값, 판정 아님).
- **O-17 (rev3.2 신규, 결정 26: 제외)** 순수 SSM 극한 a0(attention 0 / Mamba 31) 추가(+약 0.5 GPU-h): **권고 제외** — H의 질문은 attention 비율이 커질 때의 변화이며, a0은 Nano 원본보다 attention 쪽 반대편이라 우선순위가 낮다.
- **O-13 (신규)** T2′ L 범위(4k·16k)를 T3의 L(8,192·32,768)까지 확장(+약 0.25 GPU-h): **권고 확장하지 않음** — T2′는 `T2_P_NOT_REALIZED` 조건부 대체이고 현재 규칙(밖은 `NA`)이 정직하다. → **결정 28 A-2로 대체**(T2′가 주 측정, L은 T2 격자 전체).
- **O-18 (rev3.4 신규, 사용자 결정 대기)** ramp 인식 창의 예약 비용으로 shape C 보간 셀(bs 48–88: (48, 16k)·(64, 8k)·(64, 12k)·(88, 8k))과 (96, 8k)·(8, 64k)가 `INFEASIBLE_BY_RULE`이 된다(§3-1). 선택: (a) 그대로 두고 V PREREG에서 볼록껍질 밖 외삽으로 공시 / (b) Workspace smoke에서 실제 ramp를 재고 그 값 × 여유로 상한을 봉인 전에 다시 정한다(실제 step이 HBM roofline 하한보다 길면 상한이 줄어 일부 셀 회복; smoke 값은 설계 입력이지 인용 불가) / (c) 요청별 `max_new_tokens` 차등(먼저 입장하는 요청만 긴 출력) — 예약이 대략 절반으로 줄지만 도착 순서가 보장되지 않아 순서가 어긋나면 창이 깨진다(검출은 됨) / (d) 등록 노브 변경(예: T1 boot의 `--mem-fraction-static`)은 결정 13과 충돌. **권고 (b)** — 데이터 전 설계 보정이고 상한의 보수성(σ = 1에서도 하한 step 가정)이 손실의 주원인이다.
- **O-19 (rev3.4 신규, 2단계)** J-C(TP2)의 T1 셀과 96k 점: TP2 boot의 context가 73,728이면 96k 입력이 불가하고, T1 셀의 실행 가능성도 rev3.4 규칙(TP2 풀)으로 다시 평가해야 한다. 2단계 하네스 작성 때 표와 함께 올린다.

## 11. [봉인 시 기입]과 하네스

**[봉인 시 기입]** (값을 지금 알 수 없다): ENGINE\_DIGEST 통과값(예상 `99bbef91…`) · config digest(D=64 index 추가본, D별) · gate 설계 문서의 main 경로 · 서버 `new_token_ratio` 부팅 로그 값 · 부팅 캡처 bs 목록 실제값 · `split_forward_token_budget`/`max_prefill_tokens` 부팅값 · 신규 하네스 디렉터리 경로(`results/cap_char/costterms_harness/` 후보) · 보고서 파일 이름 · TP2 rank 지연 지표(rank별 telemetry가 있을 때만 기술값, 엔진 확인 후) · 분할 gate 통과 라벨.

감사 쟁점은 두지 않는다(결정 17: 항 측정은 caveat를 붙여 봉인 후보). 단 gate·하네스층 감사와 result-analyst 분석은 별도다.

### 11-2. 하네스층
신규 구현(차용 목록은 §2-4·§2-5): ① 새 디렉터리의 spec 주도 셀 클라이언트(T1 burst + ramp 인식 창 + 창 뒤 HTTP 취소(rev3.4), T3 동거, T2 단일 요청·T2′(동거 확인), T4 채우기) ② 창 추출·`S_eng` 계산(규칙 §1) ③ 클라이언트 토큰 간격 추정량 + **엔진 대 클라이언트 대조기(§1-7)** ④ 실행 가능성 평가(부팅 로그 → `INFEASIBLE_BY_RULE`, O+X = 498) ⑤ 캡처 bs 목록 검사(`ABORT_CG_BS`) + CG canary ⑥ D=64 index를 포함한 config와 실현 SM 배선 검사(`STICKY_WIRED_D64`; `SM_NOT_REALIZED` §1-8) ⑦ 엔진 모드 두 개(`108u`, `108v`)와 Llama 부팅 ⑧ T4 부팅 로그 파서 + 채우기 판정(**산식 A·B 동시 예측 기록, 부팅 직후 봉인**) ⑨ round 역순·boot 무작위 순서 ⑩ 맞춤 코드(상대오차 가중 최소제곱, LOO, 외삽 시험) + 분석기(fetch 전 커밋) ⑪ `SERVER_DEATH` 분류 ⑫ 재개·DONE 마커·sha ⑬ T3 분포·정지·σ·`n_it_obs` 기록기.
변이 시험 후보(rev3.4 추가, 하네스 `ct_mutants.py` M21–M26 전부 검출): "창을 첫 decode부터 시작(ramp 미인식)" · "명목 c로 적합" · "취소 누락으로 다음 셀에 잔여 요청" · "T2를 sticky-D boot에서 P=108−D로 라벨" · "T2′ 동거 미확인을 확인으로 라벨" · "ramp 상한 없이 O = 332 고정". 기존 후보: "cap 48 부팅인데 128로 기록" · "캡처 목록 밖 bs를 통과" · "ctx 띠 off-by-one" · "`INFEASIBLE_BY_RULE` 셀을 조용히 누락" · "T2에 첫 요청 워밍업 포함" · "Llama 셀을 Nano로 라벨" · "`108u`와 `108v` 합침" · "클라이언트 토큰 간격을 엔진 `S_eng`로 표기" · "`decode_fwd_host_ms`를 `decode_fwd_gpu_ms`로 대체" · "제외 round를 적합에 사용" · "대기 요청을 running으로 셈(T4)" · "TP2 앵커를 cap 128로 부팅" · "T4 예약 R을 0으로 둠(원식 회귀)" · "n\_A를 비대칭 형식으로 계산" · "불일치 bs를 제외하지 않음(§1-7)" · "정지(>5× 중앙값) 계산에서 중앙값을 동거 창으로 잡음" · "σ′와 σ를 함께 곱함" · "요청 SM을 실현 SM으로 기록".

### 11-3. 하네스 해석 A-3…A-26 (rev3.4 명시, 결정 28 — 하네스층 감사 대상)

결정 28이 닫은 것: A-1(§1-1 창 정의), A-2(§1-2·§1-8), A-6(T2-ext 삭제), A-11(J-H boot A의 T4 거절 사유는 I-5 OFF라 `NOT_COLLECTED`, P2는 동시 수만 필요). 나머지는 하네스가 택한 읽기이며 봉인 전 하네스층 감사가 검토한다(값은 `ct_spec.py` `AMBIGUITIES`와 같은 문장).
- **A-3** O-12 (b): (88, 8k)·(88, 9k)를 등록하고 규칙이 판정; (80, 9k) 대체는 사용자 결정 없이는 돌리지 않는다. rev3.4에서는 셋 다 `INFEASIBLE_BY_RULE`(§3-1, O-18).
- **A-4** `new_token_ratio`는 v0.5.10 부팅 로그에 없다 → `min(SGLANG_INIT_NEW_TOKEN_RATIO(0.7) × schedule_conservativeness, 1)`로 유도해 기록, `split_forward_token_budget`은 config 파일에서 읽는다.
- **A-5** `108u`(sticky OFF)에도 `PDMUX_R2_FIXED_DSM` 값이 필요 → 44(동시 실행 시 그 division, T2′의 P 64).
- **A-7** T2′ decode 요청: 입력 2,048, `max_new_tokens` 60,000(8 × 64k prefill을 P 44에서 버팀 [유도]), 측정 후 취소. **A-23** 그 decode의 ctx는 8회 prefill 동안 2k + 생성 수로 올라간다.
- **A-8** boot·셀 순서 seed = SHA-256("cost|order|<라벨>") 앞 8바이트.
- **A-9** 외삽 시험: 등록 순서대로 첫 가능한 수준(32k → 8k, 96 → 32), 남는 수준 < 2면 `NA`이며 `NA`는 실패로 세지 않는다. rev3.4: 수준은 명목 c.
- **A-10** J-H1 "셀당 반복 4" = boot A 안에서 T1 셀마다 창 4개, 셀 값 = 창 중앙값들의 중앙값.
- **A-12** 라벨 집계: AGREE는 (모델, 모드, bs)의 모든 창, W-INDEP는 모든 공통 셀·L, `CAP_INVARIANT`의 "3%"는 |ln| ≤ ln 1.03.
- **A-13** `FILL_INCOMPLETE` = n\_real번째로 이른 첫 토큰이 가장 이른 완료보다 늦음, n\_real = 채우기 동안 엔진 `bs_real` 최대.
- **A-14** J-H T2 격자에 8,800이 없어 q는 적합 이차식에서(`Q_FROM_FIT`). **A-15** `Q_UNRESOLVED` 구간 = boot별 q의 [min, max](bootstrap 미등록).
- **A-16** fused\_sym T4 boot(O-8) = cap 48 boot와 같은 cap·네 L.
- **A-17** γ\_j의 기준 = `108u`(PD-mux 엔진), D 축은 실현값.
- **A-18** "창 300" = 300 S\_eng 간격 = r0 + 16 뒤 301 기록.
- **A-19** P1·P2는 HELD 라벨만 등록 → 미충족은 `held = False`로 보고(새 라벨 만들지 않음).
- **A-20** 실현 SM = 시작 시 1회 green read-out(`PDMUX_GREEN_READOUT`); PREREG의 "첫 요청 전후"를 이렇게 읽는다.
- **A-21** warm-up·greedy·CG canary 실패는 abort가 아니라 problem으로 기록.
- **A-22** `CLOCK_SENSITIVE`(창별 I-1)는 1단계에서 계산하지 않는다(clk.csv는 wall-clock, 창은 monotonic) — 2단계 또는 분석기에서 시각 정렬 후.
- **A-24** TEL OFF에서의 창 안 동거 판정 = `mode` mixed · `bs_real` ≠ bs · 클라이언트 합류 수 ≠ bs(+0.25 s 여유); 잔여 요청 = `bs_real` > min(bs, 합류 수) (§1-1 창 정의 (4)–(5)).
- **A-25** T2′ 엔진 확인 = TTFT 가운데 절반 안의 `decode_forward`(§1-8).
- **A-26** 실제 ctx는 클라이언트 수신 토큰으로 계산 → 엔진보다 진행 중 토큰 1개 늦을 수 있다(적합 b에 최대 1 × c, 가짜 엔진에서 확인).
- 상수 [제안, 봉인 시 확인]: 창 제외 상한 30, 합류 여유 0.25 s, ramp 여유 2 step/batch, vanilla ramp 64, σ = 1·θ = 1, 취소 시점 319 토큰.

## 12. 봉인 체크리스트 (rev3.1, 2026-10-21; rev3.4 갱신 2026-10-22)

| # | 항목 | 상태 | 막는 것 |
|---|---|---|---|
| 1 | 결정 18 O-항목 반영 | 완료(§10 닫힘, §7-4 시간) | — |
| 2 | 결정 22 계측 정책 반영 | 완료(§2-2′·§8-1·§9-2) | — |
| 3 | O-14·O-15 | 완료(결정 23: O-14 포함, O-15 (b) J-B1·J-B2, §7-4) | — |
| 4 | engsync → main 병합 | **완료** — squash `ca1f30e7`(2026-10-21), 원자료 271개 약 217 MB 저장소 밖 이동(`engine_gate_prep/EXCLUDED_RAW_engsync_2026-10-21.sha256`), `.gitignore` 보강 | — |
| 5 | ENGINE\_DIGEST 통과값 | **확인** — 병합 후 main 소스로 재구성한 트리(scratch 사본)에서 `99bbef91fe68…4325`, gate Job `digests.json`과 일치. 봉인 S1에 기입 | — |
| 6 | gate 설계 문서 main 경로 | 완료 — `workspace/engine-port/results/engine_gate/GATE_SPLIT_DESIGN_2026-10-20.md` | — |
| 7 | 신규 하네스(§11-2 ①–⑬ + §2-2′ 계측 매트릭스 검사·OFF 진단 boot) 구현·CPU 시험·변이 | **1단계 완료**(`af8c0500`) + **결정 28 반영**(rev3.4: ramp 인식 창·셀별 O·취소·실제 ctx 적합·T2 P=108 한정·T2′ 6 L·동거 확인, 변이 28/28 검출). 2단계(T3·J-H2·J-C) 미착수 | 4 |
| 8 | 하네스 경로·보고서 파일명 | 대기 | 7 |
| 9 | config digest(D=64 index 추가본, D별) | 대기 | 7 |
| 10 | 하네스층 감사(최대 2회) | 대기 | 7 |
| 11 | Workspace smoke(비측정) → boot·셀 시간 실측으로 §7-4 재확인 | 대기 | 7 |
| 12 | 부팅값(`new_token_ratio`·캡처 bs 목록·`split_forward_token_budget`/`max_prefill_tokens`) | 부팅 시 기록(봉인 대상 아님, 기록 규칙만 봉인) | — |
| 13 | 봉인 S1(규칙·하네스·엔진·config digest) | 대기 | 5–11 |
| 14 | CPU preflight → `launch.sh` dry-run → 사용자 승인 → `--submit` | 대기 | 13 |
| 15 | (rev3.2) 조성 변형 부팅 프로브(비측정, `scripts/vessl/compo_probe.sh`) PASS — 변형 a4·a13·a22·a31이 dummy·PD-mux·cudagraph로 부팅, 배너 s·k 대조 | **완료 — 전 단계 PASS**(run `compo_probe_20261021T131621Z`, 커밋 `1d7261c6`) | 사용자 승인(GPU) |
| 16 | (rev3.2) O-16·O-17 | 완료(결정 26: O-16 포함, O-17 제외) | — |
| 17 | (rev3.4) 결정 28 A-1·A-2·A-6·A-11 반영 | 완료(§1-1·§1-2·§1-8·§3-1·§4′·§7-4′·§9·§11-3) | — |
| 18 | (rev3.4) O-18 shape C 셀 손실 처리 | **사용자 결정 대기**(권고 (b) smoke 실측으로 ramp 상한 재보정) | 13 |
| 19 | (rev3.4) A-3…A-26 해석 하네스층 감사 | 대기(항목 10과 함께) | 13 |

하네스 변이 시험 추가(§2-2′): "T3 셀을 T1 boot에서 실행(TEL 비대칭)" · "OFF 진단 boot에 계측 env가 남음" · "OFF 진단 비로 T1 값을 보정".
