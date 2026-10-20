# 사전등록 rev2 (DRAFT) — hybrid 서빙 비용 모델의 항별 측정(T1–T4)과 조립 검증(V), 2026-10-20

> 지위: **DRAFT rev2. 미봉인, 규칙층 감사 1회차 NO-GO(`VERDICT_prereg_costmodel_rev1_rules_r1_2026-10-20.md`)의 수리본 — 2회차 대기, 측정 0, 성능 판정 0.** 문서 작업(CPU)만 했다. 제출·GPU·커밋 없음. rev1(`PREREG_COSTMODEL_TERMS_rev1_2026-10-20.md`)은 이력으로 남긴다. 이 문서는 `PREREG_CAP_LONGCTX_rev0_2026-10-20.md`를 **대체**한다(rev0의 Q-A·Q-B·Q-C·Q-D는 없다. Q-B 이중차분은 처치 방향이 모델마다 반대라 폐기, 결정 14).
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·모형 예보, 측정 아님) · [코드] · [결정](사용자 결정 문서 명시) · [제안](이 초안이 고른 값, 감사 대상) · **[미등록]**(봉인 전에 사용자 또는 addendum이 채울 칸).
> 근거: 계획 `PLAN_costmodel_terms_2026-10-20.md` · 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 1–16 · **감사 `VERDICT_prereg_costmodel_rev1_rules_r1_2026-10-20.md`(수리 대상)** · 프로브 `probe_128k/RESULT_probe_128k_2026-10-20.md`·`AUDIT_probe_128k_2026-10-20.md`(허용 문안 1–7)·`ANALYSIS_triton_96k_failure_2026-10-20.md` · J1 감사 `../multigpu_verdicts/AUDIT_J1_N1_analysis_2026-10-19.md` · 감사 `../anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md` · 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(이하 SP, v0.2+) · 항 정의 `COSTMODEL.md` CM-1…7.
> 기판: VESSL A100 SXM 80GB(108 SM) 1장(TP2 Job은 2장), SGLang v0.5.10, Nemotron-Nano-9B-v2-Base(rev `dc0661c…`), cudagraph ON, radix OFF, **`--mem-fraction-static` 0.6**(결정 13). 0.82로 잰 결과(M3 J1, 프로브 1회차)와 섞지 않는다. **ENGINE\_DIGEST = [gate 통과 digest — 봉인 시점에 기입]**(엔진 묶음 = worktree `prefill-layer-alloc-engsync` 브랜치 `engine-sync-i5-int64`, 현재 `b0f6cfeb`; I-4·I-6 추가로 digest가 다시 바뀌므로 지금 값을 쓰지 않는다).

## R. rev1 → rev2 수리 대응표 (감사 1회차 항목 전부)

| 감사 항목 | 수리 위치 | 수리 내용 |
|---|---|---|
| ① 死因 1 — T4 고정점에 출력 예약 항 없음 | §1-4, §6-1, §6-2 | 원식 A와 수정식 B를 **둘 다 봉인**. L별 예측 비 표를 데이터 이전에 제시(0.51/0.68/0.80/0.89 재현). "구현 사실 확인" 라벨(`CMEM_ACCT_*`)로 낮추고 항등식 성격 공시. V 예측기 동시 수·cap\_V를 같은 예약 식으로 수정. shape C KV 상한 60 vs 83 불일치는 [미등록]으로 올림 |
| ① 死因 2 — V 사다리 중심·폭 | §6-2 | arm별 중심 = 그 arm 예측, ±2 rung(SP R9-3). 오판 구간 산술로 해소 확인(\|e\| ≤ 0.223은 항상 ±0.365 안) |
| ② MAJOR(1) 조립 귀속 불가·항등식 | §6-3, §6-8 | 귀무 예측기 N0·N1을 V 데이터 전에 봉인, 오차가 더 작아야 PASS. V-diag `TERM_DEVIATES` 0개가 `V_PASS` 조건, 아니면 `V_PASS_CANCELLATION` |
| ③ MAJOR(2) 허용 오차·Job 효과 | §6-3, §6-7, §R 등록 caveat | J-V 내부 앵커(T1 4셀 + T2 3L, 108v와 PD-mux D), 3% 초과 `V_JOB_SHIFTED` + 보정 예측 2차. R9-6 해상도 > 5% 셀은 `V_PASS_UNREACHABLE`. 10% 출처 caveat 문장 |
| ④ MAJOR(B) sticky·P 실현 | §1-8, §6-2 | V arm sticky 모드 고정(ON), 실현 SM 로그 검증 `SM_NOT_REALIZED`. `T2_P_NOT_REALIZED`일 때 bs=1 decode 동거 T2′ 대체 측정(J-A boot마다 상시 수집) |
| ⑤ MAJOR(7) 엔진 S-per 대 클라이언트 간격 | §1-1, §1-7, §3-4 | bs별 3% 이내 대조, 불일치 bs 제외. I-4(O-2)로 PD-mux·vanilla 양쪽 엔진 측 시간 확보. "hybrid 고유" 원인 문장 불가 |
| ⑥ MAJOR(8) T3 꼬리·span 측 | §1-3, §1-6 | iteration 분포 전체·정지(>5×중앙값) 횟수·span co/solo σ 기록. `max(T_dec·ρ, T_span)`의 T\_span 단위 봉인 |
| ⑦ MAJOR(4) T1 외삽 | §1-1, §3-1, §6-1 | 셀 (64, 8k)·(64, 12k) 추가(고유 셀 118 → 128). `PRED_EXTRAPOLATED` 처리 봉인(10% 초과 시 `V_PASS_UNREACHABLE`) |
| ⑧ MAJOR(10) R8-2′ + O-1 | §8-2 | SP 개정 문안 R8-2′(O-1 승인, 결정 16). cap\_V 예약 반영 산식. T4가 R16-3 대체. R8-2c 예시 재계산(C=512, 388토큰, 75.8%) |
| ⑨ MINOR (3)(6)(12)(13)(C)(D)(E) | §6-2, §6-1, §1-2, §4-2, §1-1, §6-4, §6-1 | (3) 상위 2 D 범위 문장 (6) 자연 셀 튜닝 편향 G 문장 (12) 합성 대표성 G7 허용 문안 6 (13) 64k·T2-ext 허용 문안 3–4 승계 (C) 예측기 bs\_padded (D) J1 기울기 기술값 공시 (E) 확률 시뮬레이션·`CLIFF_ADJACENT_PRED` |
| OK (5)(11)(F), caveat (G) | §6-3, §6-4, §6-1, §1-5 | n=2 round는 기술 적합·V는 "이 Job에서"(5) / Claim E 함의 한계 유지(11) / duration 합산은 R5-1 승계 명시(F) / 전력 상태 차이 caveat(G) |
| 등록 caveat 3문장 | §R 아래 | 문자 그대로 |
| 결정 16 O-1·O-2·O-3·O-5·O-7, gate 분할 | §8-1, §8-2, §3-4, §5-3, §6-2, §7 | O-1 R8-2′ 승인 / O-2 I-4·I-6 포함 / O-3 96k·124k는 int64 gate 통과 조건부(§1-2, §5-3) / O-5 V PD-mux arm 2개 / O-7 J-V 분할(J-V1·J-V2a·J-V2b) / gate는 분할 G-α → G-β → G-γ |
| 자체 정정(rev1 오류) | §3-1 | rev1 "합 100개 고유 셀"은 14·14·18×5 = 118의 오기. rev2는 118 + 신규 10 = 128 |
| Job 구성 재계산 | §7 | Job마다 ≤ 6 h 목표. J-A를 round별 2 Job으로 분할, J-V 3 Job. 약 27.8 GPU-h |

### 등록 caveat (문자 그대로, 감사서)
- "이 비용 모델은 Nemotron-Nano-9B-v2 한 모델의 것이며 'hybrid 서빙 비용 모델' 일반이 아니다."
- "V는 한 Job·두 셀의 예측 일치이며, 허용 오차 10%는 KISTI 타 모델 G에서 빌린 값이다."
- "agnostic 최선은 예측 상위 2 D 중 최선이다."

(rev2에서 V는 3 Job으로 분할되므로 두 번째 문장은 감사서 원문 그대로 둔다. 분할 사실은 §6-3의 판정 범위 문장에 따로 적는다.)


## 0. 프레이밍과 이 문서가 하지 않는 것

- **주 목적(C)** = 비용 모델: lockstep PD-mux 비용 구조를 항 T1–T4로 재고(마이크로, 고정 배치·고정 길이), 조립해서 λ\*\_SLO를 예측하고(V) 서빙 실측과 대조한다. **(G)** PD-mux 이득은 그 모델이 설명해야 할 **검증 대상**이다. **(H)** "왜 hybrid인가"는 hybrid 고유 항을 Llama와 **대비**해 답한다. 모델 간 goodput 비교는 하지 않는다.
- 하지 않는 것: G0·M3 PREREG가 아니다. cap goodput 스윕(rev0 Q-A/Q-C)은 없다(cap = C\_mem 산식, 결정 15). 정책 순위·HE0·Claim E/G 등급을 바꾸지 않는다. layer-type별 분해는 하지 않는다(T1은 전체 모델 시간이라 원리적으로 불가, VERDICT\_claimE 근거 N2 PROFILE 항목).
- **항 T5**(CM-6 대기·TTFT)는 측정하지 않는다. T1–T4로 조립하고 V에서 대조한다. CM-3(전환·경계)·CM-7(radix)은 범위 밖이다.
- 인용 범위(모든 항 공통): 이 기판(VESSL A100-SXM4-80GB, Job별 GPU UUID 공시)·이 스택(SGLang v0.5.10, 봉인될 ENGINE\_DIGEST, Triton causal\_conv1d)·mem 0.6·클럭 비고정. **Job별 값**이다 — 서로 다른 Job의 수준값을 비교·결합하지 않는다(Job 효과 진단 예외: §6-7의 J-V 내부 앵커와 J-A1/J-A2·J-B의 `108v` 차).

## 1. 항별 estimand와 맞춤 형태

공통: 단위 = (boot, cell) 창 중앙값. 창 = N2 선례의 개념을 차용한다(burst 뒤 단독 decode, 처음 16 iteration 제외, batch 평균 ctx p50이 목표 c의 ±64 띠 안, 연속 iter\_seq 쌍, retract 0, cudagraph hit 정황). 상세 구현은 새 하네스의 몫이다(§3-2). 클럭·전력(I-1)은 창마다 공변량으로 기록하고 **제외 규칙으로 쓰지 않는다**(민감도 표만, §1-5).

### 1-1. T1 — CM-1 `T_dec(D, bs, ctx)`

| 항목 | 등록 |
|---|---|
| estimand | 결정 D(green-context로 **실현된** decode SM 수, sticky ON)·실행 batch bs·batch 평균 ctx c에서 decode step 시간(ms). 1차 = 엔진 측 decode 시간(PD-mux·vanilla **양쪽 경로**, I-4·I-6 포함 — O-2 결정; N2 선례의 연속 iteration 쌍 간격 S-per와 같은 창 규칙). 2차 = 클라이언트 steady-state 토큰 간격 중앙값(모델 간 대비의 공통 추정량, §4). 엔진 대 클라이언트 bs별 3% 대조와 불일치 bs 제외는 §1-7. 입력 bs는 실행 bs가 아니라 `bs_padded`(I-6)로도 기록 |
| 격자 | D ∈ {16, 24, 34, 44, 64} sticky ON + **D=108을 두 모드로**: `108u`(PD-mux 엔진 모드, 비분할 `(0,108)`, N2 관례), `108v`(vanilla 엔진, fused 계열 decode). bs ∈ {1, 8, 32, 48, 96}, c ∈ {2k, 8k, 32k, 64k} + **16k**(D ∈ {24, 34, 44, 108u, 108v}만; shape C 구간 11–20k 보간용) + **(bs, c) = (64, 8k), (64, 12k)**(rev2 신규, 같은 5개 D·모드만; shape C 영역 bs 55–80 보간, 감사 MAJOR 4. 12k 셀은 §3-1 규칙으로 실행 여부 판정, 비실행이면 `INFEASIBLE_BY_RULE`). 실행 가능 판정은 §3-1의 규칙 |
| 고정 노브 | cap = 128(V 운용 cap 근방, 결정 15와 §6-2), `--max-mamba-cache-size` = cap, mem 0.6, context\_length 73,728, cudagraph ON, 캡처 bs ⊇ {1, 8, 32, 48, **64**, 96}(부팅 로그 검사, 아니면 `ABORT_CG_BS`), 같은 config 바이트(`pdmux_n3_A_agn.yml` 계열 + D=64 index 추가, digest [미등록]) |
| 제안 함수형 | `T_dec(D,bs,c) = a_D + b_D·bs + c_D·bs·c` — a = bs·ctx 무관 고정(가중치 sweep·launch), b = 요청당(SSM state·MLP), c = KV 읽기. 대안 형(적합도 실패 때만, 이 순서): F2 = F1 + d\_D·bs², F3 = a + b·bs^β + c·bs·c. 셋 다 실패하면 `FORM_INADEQUATE_T1_D{d}`, V 예측기는 **격자 안 이중선형 표 보간(log bs, log c), 외삽 금지**를 쓴다 |
| 맞춤 방법 | 상대오차 가중 최소제곱(Σ((T−T̂)/T)² 최소), D·엔진 모드마다 따로. 단위 = (boot, cell) 중앙값. D 간 부속: 계수별 `κ_j(D) = κ_j(108)·(108/D)^γ_j`(j = a, b, c)를 6 D에 맞춰 γ\_j를 **기술값**으로(N2 PROFILE의 계수비는 인용 금지·재측정, 층 종류 귀속 금지) |
| 적합도 보고 | (i) 표본 내 상대 잔차 중앙·최대 (ii) leave-one-cell-out 중앙·p90 (iii) **외삽 시험**: ctx 수준 1개(32k, 그다음 8k)와 bs 수준 1개(96, 그다음 32)를 통째로 빼고 맞춘 뒤 뺀 수준 예측 오차 (iv) 두 round를 각각 맞춘 계수의 차. **적합 채택 기준 [제안]**: (ii) 중앙 ≤ 3% ∧ p90 ≤ 8% ∧ (iii) 최대 ≤ 10%. 미충족은 위 라벨 |
| 인용 범위 | 해당 D·엔진 모드·**cap 128 부팅**·이 격자 박스 안(V 예측기가 볼록껍질 밖을 조회하면 `PRED_EXTRAPOLATED`, §6-1). 박스 밖 외삽 문장 금지. 허용: "이 Job에서 D=…, bs=…, ctx=…의 decode step 중앙값은 …ms이고, 함수형 F1의 표본 내 잔차는 …, 외삽 시험 오차는 …였다" |
| 금지 | N2 PROFILE 계수(2.51/4.63/1.95×, −14~16%)·"shape B ctx 항 1%"·"shape C에서 hybrid 항이 유의하다"(VERDICT\_claimE 금지 문안); γ\_j를 s\_attn/s\_ssm 증거로 읽는 것; D 간 비를 "layer-type 차이"로 읽는 것 |

cap 불변성 점검(J-A1 안, D=44 한 boot): cap 48로 부팅해 bs ∈ {8, 32} × c ∈ {2k, 8k} 4셀을 cap 128 boot의 같은 셀과 비교. 차 ≤ 3%면 `CAP_INVARIANT`, 아니면 `CAP_DEPENDENT`(T1 값에 "cap 128 조건부" 꼬리표, TP2 앵커는 cap 48을 쓰므로 영향 공시).

### 1-2. T2 — CM-2 `T_span(L, SM)`

| 항목 | 등록 |
|---|---|
| estimand | 다른 부하가 없을 때 단일 요청 prefill 시간. 1차 = 클라이언트 TTFT(`input_ids`로 정확히 L 토큰, 출력 1토큰, `ignore_eos`) 중앙값 — J1 floor(CM-2)와 같은 정의라 모든 모델·모드에서 비교 가능. 2차 = 엔진 span 시간(Nano PD-mux만; I-3 없음) |
| 격자 | L ∈ {1,024, 4,096, **8,800**(≈ s/k, 유도 경계 근방), 16,384, 32,768, 65,536} × 실현 prefill SM P = 108 − D ∈ {92, 84, 74, 64, 44}(D 격자 5개의 부산물, T1 boot 안에서 측정) + P = 108(`108u`·`108v`). 등록 대표 4점 P ∈ {44, 64, 74, 108}, P = 84·92는 무료 기술값. 한 (boot, L)당 8회, 앞 1회는 워밍업이라 제외 |
| 조건부 확장 T2-ext | **int64 패치(`chunk_offset`·`sequence_start_index` 두 곳)가 분할 gate G-α→G-β→G-γ를 전부 통과하면(§8-1)** P = 108에서 L ∈ {98,304, 126,976}을 두 모델에 더한다(O-3 결정: gate 통과 조건부로 포함). 미통과면 없음. 결정 13의 64k는 격자 선택이며 프로브 감사 허용 문안 3–4를 승계한다 |
| 제안 함수형 | `T_span(L,P) = u_P + v_P·L + w_P·L²`(P마다). P 의존 `(108/P)^θ`는 기술값. **모델-무관 형태 지표** `q = [T(65,536)/65,536] / [T(8,800)/8,800]` (토큰당 비용 증가비) |
| 맞춤 방법 | 상대오차 가중 최소제곱, 단위 = boot 중앙값 |
| 적합도 보고 | leave-one-L-out(내부 L 4개) 상대오차, 두 round 계수 차. 채택 기준 [제안]: 내부 L 제외 오차 ≤ 5%. 미충족 `FORM_INADEQUATE_T2_P{P}`, 표 보간(log L) 사용 |
| 라벨 | `q ≥ 1.10` → `PREFILL_SUPERLINEAR_SCOPED`, `q < 1.10` → `PREFILL_NEAR_LINEAR_SCOPED`, 구간이 1.10을 걸치면 `Q_UNRESOLVED` [제안]. J1 기울기(0.111 ms/토큰, L ≤ 16k, **mem 0.82 결과**)와 P=108 기울기의 차는 **기술값으로만**(차 > 5%면 `T2_J1_SLOPE_DEVIATES`, 기판·mem 차이 공시 항목이지 결과·보정 근거 아님) |
| 인용 범위 | 이 기판·이 스택·chunk −1(PD-mux 강제 조건과 같음)에서 L ≤ 64k. "초선형/선형"은 q 라벨까지만 |
| 용도 | V의 SLO floor 예측과 fused 청크 비용(§6-1). fused\_tuned의 청크 prefill 비용은 `T2(pos+c) − T2(pos)`로 근사한다(위치 의존 가정, 미검증) |


### 1-3. T3 — CM-4 ρ (동거 간섭, **총효과**)

| 항목 | 등록 |
|---|---|
| estimand | 같은 boot·같은 셀에서 `ρ = S-per(동거 창) / S-per(단독 창)`. 동거 창 = 고정 decode batch(c = 8,192 [제안])가 돌 때 prefill 입력(L)이 지속 진행되는 iteration들. 처리 = 동거 + 그에 동반된 전력 한도 클럭 하락의 **합성값**(N2 Job-EA 선례: 분리 불가) |
| **기록(rev2 추가, 감사 MAJOR 8)** | 중앙값 ρ만 쓰지 않는다. 셀·창마다 (a) 동거 창 **iteration 시간 분포 전체**(원 값 파일 + p50/p90/p95/p99/max)와 단독 창 분포 (b) **정지 횟수·비율**: 단독 창 중앙값의 5배를 넘는 iteration 수와 연속 정지 길이(SLO가 요청 내부 ITL p95라 중앙값이 정지를 가린다; 프로브 64k 최대 정지 0.8 s가 선례) (c) **span co/solo `σ = T_span_iter(동거) / T_span_iter(단독)`**: 같은 boot·L·P에서 동거 창의 prefill span 시간 대 단독 창(T2와 같은 span)의 비. 출처 = 엔진 span 시간(Nano PD-mux, 필드 [미등록 — engine-porter 확인]). 없으면 `SIGMA_NOT_COLLECTED`(클라이언트 TTFT는 큐잉을 포함하므로 σ로 쓰지 않는다) |
| 격자 | D ∈ {24, 44} × bs ∈ {8, 32} × L ∈ {8,192, 32,768} = 8셀. 동거 창이 128 iteration 이상 되도록 prefill 입력 개수 m(L)을 등록 상수로 둔다 [미등록 — 하네스 몫, 값은 데이터 전 고정]. 반복 = round당 2회 × round 2 = 셀당 4 |
| 맞춤 | `ln ρ`와 `ln σ`의 셀별 평균(표). 예측기는 (log bs, log L) 이중선형 보간, D=34는 ln D 선형 보간(`RHO_D_INTERP`), 격자 밖은 클램프(`RHO_CLAMPED` 횟수 공시). 정지 비율 `q_stall`(>5× 중앙값)도 셀 표로 둔다 |
| 적합도 | 4개 반복의 산포, 셀별 SW power cap 노출률·SM 클럭 중앙값(동거/단독 창 각각) |
| 인용 범위 | **총효과로만**: "400 W 한도 아래 클럭 하락을 포함한 동거 감속". N2 Job-EA 허용 문안 (P)(A)(B)의 틀을 따른다. N2 RESULT :68-72(decode는 prefill 청크를 기다리지 않음)는 이 기판·그 설정의 기술이며 σ·`q_stall` 측정이 이 PREREG의 확인 수단이다 |
| 금지 | "ρ는 동거 간섭의 크기다"(cap 병기 없이)·"전력 무관 간섭"·"ρ는 D·b·c에 무관"·ρ에서 정책/재배분 경계 도출·J1/N2 Job의 ρ와의 비교·중앙값 ρ만으로 "꼬리가 없다" |


### 1-4. T4 — CM-5 C\_mem과 거절 사유 몫 (rev2 전면 개정, 감사 死因 1)

**성격 라벨(공시 의무)**: 이 항의 산식 대조는 **"구현 사실 확인"**이다. 산식 B는 엔진의 입장 회계(admission accounting)를 그대로 옮긴 것이라, 채우기 실측이 B와 맞는 것은 엔진이 코드대로 동작한다는 사실 확인이지 독립 예측의 성공이 아니다(교훈 9: 항등식에 가까운 검사). 산식 대조 결과를 "비용 모델이 C\_mem을 예측했다"로 쓰지 않는다. T4에서 비용 모델에 실제로 기여하는 것은 (a) 부팅 로그 사실(s, k, 풀)과 (c) 거절 사유 몫이다.

| 항목 | 등록 |
|---|---|
| estimand | (a) 부팅 로그 사실: KV 풀 토큰 수, mamba 슬롯 수·바이트, 가중치 바이트 → s = 슬롯당 바이트, k = KV B/토큰, 풀 합 (b) 채우기 실측: 같은 L의 요청 N = cap + 8개를 한꺼번에 보내 **실현 최대 동시 running 수** `n_real` (c) 거절 사유 몫(I-5: KV / cap≡슬롯), iteration 가중과 시간 가중 병기 |
| 예약 회계 [코드] | `ignore_eos`이고 radix OFF이면 입장 시 **입력 + `max_new_tokens` 전량**을 예약한다(`schedule_policy.py` `add_one_req_ignore_eos`: `new_token_ratio = 1.0 if ignore_eos` :647-652, 입장 분기 :752-753). 그 외 요청은 `min(max_new_tokens, 4096[CLIP_MAX_NEW_TOKENS]) × new_token_ratio`를 예약한다. 서버 `new_token_ratio` 값은 부팅 로그 값을 기입 [미등록] |
| **산식 두 개(둘 다 봉인)** | **원식 A(rev1)** `C_A(L) = floor(풀 / (s + k·L))` — 출력 예약 없음. **수정식 B(rev2 1차)** `C_B(L, R) = floor(풀 / (s + k·(L + R)))`, 예약 `R` = (`ignore_eos`: `max_new_tokens` 전량; 그 외: `min(max_new_tokens, 4096) × new_token_ratio`). 임의 cap에서의 실현 예측 `n_B(cap) = min(cap, floor((풀 − cap·s) / (k·(L + R))))`(mamba 슬롯 cap개가 정적 예약되어 KV 풀이 cap에 따라 줄므로 풀 합이 아니라 이 형으로 쓴다). 원식의 대응 예측 `n_A(cap) = min(cap, C_A(L))`. `cap = C_B`이면 `n_B = C_B`로 닫힌다(고정점) |
| Nano 격자 | L ∈ {4,096, 8,800, 16,384, 65,536}, 출력 R = {4,096, 4,096, 4,096, 8,000}(채우기 규칙과 동일, `ignore_eos`). **부팅 9개**: (i) cap 48 1개(네 L 모두) (ii) **cap = C\_A(L)** 4개 = {156, 114, 80, 27} — 원식 A와 수정식 B가 갈리는 판별 부팅 (iii) **cap = C\_B(L, R)** 4개 = {118, 93, 68, 24} — B의 고정점 부팅. 풀이 cap과 독립이라는 가정(KV 토큰 + mamba 바이트 = 상수)은 [유도]이며 **부팅 9개의 풀 합 일치(2% 안, 아니면 `C_MEM_DRIFT`)로 검증** |
| **데이터 이전 예측 표** [유도, 풀 33.04 GB·s 143.5 MB·k 16,379 B 대입; **부팅 로그의 풀로 같은 코드가 재계산한 표를 각 부팅 직후·채우기 전에 봉인**] | 아래 표 | [CS-OK] 33.04는 풀 GB 값(우연 일치)
| 채우기 규칙 | 출력 길이는 마지막 요청의 첫 토큰 시각이 가장 이른 요청의 완료 시각보다 앞서도록 L별로 등록한다(L = 4k–16k는 4,096, 64k는 8,000). 위반하면 `FILL_INCOMPLETE`(그 셀은 산식 대조에서 제외, 건수 공시). **출력을 줄이면 R도 같이 줄어 `C_B`가 커지므로 출력 길이는 R의 입력으로 고정 기록**한다. arm = static D=44 PD-mux [제안, fused 대칭 점검은 O-8] |
| Llama(대비) | 부팅 1개(cap 128, mem 0.6): 풀 토큰 수(s = 0, k 131 KiB) + 채우기 L ∈ {4,096, 16,384}, R = 4,096(`ignore_eos`). 예측: KV 풀 263,900 토큰 [rev1 §4 ③] 기준 `n_B` = {32, 12}, 원식 A = {64, 16} |
| 판정 형 | 산식 B: `n_real / n_B ∈ [0.90, 1.10]` → `CMEM_ACCT_HELD`, 아니면 `CMEM_ACCT_MISS` [제안]. 원식 A: `n_real / n_A < 0.90`이면 `CMEM_ORIG_OVERPREDICTS`(예측된 결과, 아래 표의 ratio 열을 사전 예측으로 봉인 — ±0.05). 사유 몫은 기술값 |
| 인용 범위 | 부팅 로그 사실은 [측정] 그대로. 산식은 "이 스택의 정적 풀 구현(cap≡슬롯)에서 입장 회계를 확인했다"는 범위. 할당기 종속이라 hybrid 일반 문장 금지(결정 8′). "예측이 맞았다"·"비용 모델이 C\_mem을 검증했다" 금지 |

사전 예측 표 (Nano; `ratio_A` = cap = C\_A 부팅에서의 `n_B / C_A` = 원식의 예측 과대 정도):

| L | R | C\_A | C\_B | C\_B / C\_A | cap = C\_A 부팅의 `n_B` (`ratio_A`) | cap 48 부팅의 `n_A` / `n_B` |
|---|---|---|---|---|---|---|
| 4,096 | 4,096 | 156 | 118 | 0.756 | 79 (**0.51**) | 48 / 48 |
| 8,800 | 4,096 | 114 | 93 | 0.816 | 78 (**0.68**) | 48 / 48 |
| 16,384 | 4,096 | 80 | 68 | 0.850 | 64 (**0.80**) | 48 / 48 |
| 65,536 | 8,000 | 27 | 24 | 0.889 | 24 (**0.89**) | **27 / 21** |

감사서의 비 0.51·0.69·0.80·0.89가 재현된다(8,800은 풀 반올림으로 0.68 vs 0.69; 부팅 로그 풀로 재계산). rev1 설계로 데이터를 모았다면 비가 1이 되는 L은 없고 `CMEM_PRED_MISS`가 설계 단계에서 확정이었다는 뜻이다. rev2는 이 표를 데이터 이전에 낸다. cap 48 부팅의 64k 칸이 A와 B가 갈리는 두 번째 판별점이다(27 대 21).


### 1-5. 공변량·민감도(모든 항)

I-1 클럭·전력·throttle을 창마다 기록한다. SW power cap 노출률 > 20%인 창을 뺀 재적합을 민감도 표로 낸다(계수 변화 > 3%면 `CLOCK_SENSITIVE`, 제외는 하지 않음) [제안]. 클럭 고정은 불가(VESSL). 항 측정과 V의 전력 상태 차이는 §6-4 caveat(G).


### 1-6. T5 — 조립(측정 없음)

반복 하나 = decode step 1개 + prefill span 1개(PLAN §1, CM-4). prefill 창 안의 ITL은 `max(ρ·T_dec, σ·T_span_iter)`. TTFT = W\_queue + n\_it(L)·σ·T\_span\_iter + 통신(TP). 동시 수 상한은 cap과 §1-4의 **예약 회계 산식 B**(요청별 `L_i + R_i`의 합이 풀을 넘지 않는 입장 규칙), 수요 = λ·E[out]·E[ITL].

**T\_span 단위 봉인(감사 MAJOR 8)**: `T_span_iter` = **엔진이 한 iteration에서 prefill 쪽에 할당하는 forward 1회의 시간**(span 1회; 단위 토큰 수 `SPAN_TOKENS`는 T3 로그에서 읽어 기록, `split_forward_token_budget` 65,536·`max_prefill_tokens` 16,384와의 관계는 [미등록 — 하네스 몫]). 요청 1개 전체 prefill 시간은 T2의 `T_span(L, P)`(= 전체)이고 `n_it(L) = T_span(L, P) / T_span_iter`로 환산한다. 조립 1차 = 이 단위(F-iter). 변형 `v5`(소거 변형에 추가, 기술값): 요청 전체를 한 span으로 읽는 `max(ρ·T_dec, σ·T_span(L, P))`(F-whole). 둘 다 V 데이터 전에 예측을 봉인하고, 어느 쪽도 사후 선택하지 않는다. 이 구조의 실현(span 단위, 예산, 청크 fused의 합산 가정)은 §6-1의 예측기 명세가 정한다.

### 1-7. 엔진 S-per 대 클라이언트 간격 대조 (rev2 신설, 감사 MAJOR 7)

배경: rev1 시점에는 `decode_iteration`이 PD-mux mixin에서만 방출되어(`src/multiplex/multiplexing_mixin.py:2245-2270`) Llama·`108v`는 클라이언트 토큰 간격뿐이었고 GIL·클라이언트 과부하(confound 7)에 노출되었다. 결정 16 O-2로 I-4·I-6이 엔진 묶음에 들어가면 **엔진 측 decode 시간이 PD-mux와 vanilla 양쪽 경로에서 기록**된다(필드 정의 [미등록 — engine-porter 문서]). 이것으로 문제가 상당 부분 해소되지만 대조는 유지한다.
- **1차 추정량 = 엔진 측 decode 시간**(양 경로). **2차 = 클라이언트 steady-state 토큰 간격**(모델 간 공통 추정량).
- **대조**: 각 (엔진 모드 `108u`·`108v`·Llama vanilla, bs)마다 같은 창에서 `|ln(client 중앙값 / engine S-per)| ≤ ln 1.03`이면 `CLIENT_ENGINE_AGREE_bs{b}`, 아니면 `CLIENT_ENGINE_MISMATCH_bs{b}` — 그 bs의 클라이언트 간격 기반 양은 H 대비(§4 ③)와 T1 2차에서 **제외**하고 엔진 측 양만 쓴다. 대조는 J-A·J-B에서 수집된 모든 창에서 계산한다.
- **fallback**: gate에서 vanilla 경로의 엔진 측 시간 방출이 확인되지 않으면(`NOT_COLLECTED`) 대조는 `108u`에서만 가능하고, `108v`·Llama의 클라이언트 간격은 `108u`에서 얻은 bs별 일치 여부를 **조건**으로 `CLIENT_ONLY_UNVERIFIED` 꼬리표를 달아 쓴다(제외 아님, 주장 약화).
- **원인 문장 금지**: Nano 1개 대 Llama 1개의 기술 차이까지만 쓴다. "hybrid 고유"를 원인으로 쓰지 않는다(§4-2).

### 1-8. V arm의 sticky 모드와 P 실현 검증 (rev2 신설, 감사 MAJOR B)

- **sticky 모드 고정**: J-A·J-B·J-C·J-V의 모든 PD-mux boot는 **sticky ON**(T1과 같은 config 계열 `pdmux_n3_A_agn.yml`, D별 digest [미등록])이다. V의 PD-mux 2 arm도 같다. 모드가 arm·Job 사이에서 다르면 `STICKY_MODE_MISMATCH`로 그 단위를 제외한다.
- **실현 SM 로그 검증**: 요청 SM ≠ 실현 SM(생성 성공 ≠ 요청 수신)이므로 boot마다 첫 요청 처리 전후의 realized decode/prefill SM 로그를 읽어 요청 D·P와 비교한다(허용 = 4 SM 입도 반올림 규칙 [미등록]). 불일치 → `SM_NOT_REALIZED_D{d}`, 그 boot의 T1·T2 값은 해당 D 항에 쓰지 않고, V에서 그 D를 arm으로 쓰면 그 arm은 `UNRESOLVED(sm_not_realized)`로 둔다.
- **`T2_P_NOT_REALIZED` 대체 측정**: 단일 요청 prefill이 sticky ON에서 `P = 108 − D`로 실제 도는지 로그로 확인한다. 아니면(안 돌면 T2의 SM 축 정의 불가) **T2′ = bs=1 decode 요청(ctx 2k)이 동거하는 상태에서 잰 단일 prefill TTFT**를 쓴다. T2′는 J-A boot마다 L ∈ {4,096, 16,384}에서 **상시 수집**(8회 중 앞 1회 제외)해 두어, 라벨이 붙어도 재부팅 없이 대체한다. 대체한 D는 `T2_FALLBACK_BS1_COEXIST_D{d}`를 달고 V 예측기가 그 D에 한해 T2′를 쓴다(estimand가 "다른 부하 없음"에서 "bs=1 동거"로 바뀐 사실을 인용 범위에 적는다). J-V 내부 앵커(§6-7)도 같은 L에서 T2′를 수집한다.


## 2. 측정 방식

### 2-1. 한 줄 정의
고정 배치·고정 길이를 엔진 안에서 만들고(burst로 bs개를 동시에 보내 ctx c까지 도달시킨 뒤 창을 읽는다) 서빙 부하(rate 사다리)는 쓰지 않는다. rate 사다리는 V에서만 쓴다.

### 2-2. 단계 분리
boot마다 warm-up(8요청, 같은 형상) → correctness(sticky 배선 로그 + 짧은 greedy 출력 비공백 + CG canary) → benchmark(셀들) → 후행 비측정 요청 → idle 2×flush → 종료. 계측 ON 수준은 모든 Job이 같다(관측자 효과는 gate의 TOST가 묶는다, §8-1).

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

신규 하네스는 `results/cap_char/costterms_harness/` 같은 새 디렉터리에 둔다(경로 [미등록]).

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

### 3-1. 실행 가능성 규칙(데이터 전에 등록, 부팅 로그로 평가)
셀 (bs, c)는 `bs ≤ cap` ∧ `bs ∈ 캡처 bs 목록` ∧ `bs·(c + O + X) ≤ 0.95 × KV 풀 토큰(부팅 로그)`일 때만 실행한다. 아닌 셀은 실패가 아니라 `INFEASIBLE_BY_RULE`로 표에 남긴다. cap 128 부팅의 KV 풀 [유도 895,781 토큰 = (풀 33.04 GB − 128×143.5 MB)/16,379 B; 부팅 로그가 대체]: [CS-OK] 33.04는 풀 GB 값(우연 일치)

| bs \ c | 2k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| 1 | ○ | ○ | ○* | ○ | ○ |
| 8 | ○ | ○ | ○* | ○ | ○ (526k) |
| 32 | ○ | ○ | ○* | × (1.06M) | × |
| 48 | ○ | ○ | ○* | × | × |
| 96 | ○ | ○ (820k) | × | × | × |

(\* = D ∈ {24, 34, 44, 108u, 108v}만.) 실행 셀 수/D: 16·64 → 14, 나머지 5모드(24·34·44·108u·108v) → 18, 합 **118**개(rev1의 "100개"는 오기, T1 "240 셀실행"은 118×2 ≈ 236과 맞음) + **신규 (64, 8k)·(64, 12k) × 5모드 = 10** → **128개 고유 셀**(bs 64 줄은 신규 셀 8k·12k만 두며 2k·16k·32k·64k는 두지 않는다 — 16k는 64×16k = 1.05M > 0.95 × 895k = 851k라 ×. 12k는 64 × (12,288 + O + X) ≤ 851k, 즉 O + X ≤ 약 1,000 토큰일 때만 ○ — 이 경계 셀은 규칙 평가에 맡기고 비실행이면 `INFEASIBLE_BY_RULE`). 비어 있는 (32, 32k)·(48, 32k)·(·, 64k at bs ≥ 32)는 **cap 128·mem 0.6에서 KV가 모자라서가 아니라 격자 규칙상** 비운 것이며, 그 영역의 모델 값은 외삽이다(§1-1 인용 범위). V의 shape C는 bs ≈ 55–80, c ≈ 7–20k 영역 [유도]이므로 8k와 16k 점, 그리고 rev2의 (64, 8k)·(64, 12k)가 보간을 받치고 32k는 bs ≤ 8 점만 있다 — 볼록껍질 밖 조회는 `PRED_EXTRAPOLATED`로 처리하고(§6-1) 이 한계는 §11 감사 쟁점 (4)에 올린다.

### 3-2. 새 하네스 항목(차용 목록은 §2-4, 신규 구현은 §11-2에 나열)

### 3-3. Job 간 정렬되는 상수
cap 128(T1·T2·T3), 48(cap 불변성 boot·T4 일부), mem 0.6, context 73,728, 같은 `max_prefill_tokens` 16,384 명시, 같은 attention backend(flashinfer, Llama도 명시), `ignore_eos`, radix OFF, logprob 없음, `PDMUX_HEARTBEAT_S=0`.

### 3-4. 계측 (I-1, I-4, I-5, I-6) — 구현 상태 [저장소 grep, engine-porter 확인 요청]

| 계측 | 쓰이는 항 | 저장소 상태 [코드] | 필요 조치 |
|---|---|---|---|
| I-1 클럭·전력·throttle | 전 항(공변량) | `clk.csv`·`substrate.json` 수집은 `job_entry.sh`·`substrate_probe_helper.py`에 있음. SP는 "F-2 0행 문제 수리·셀 시작 전 생존 검사"를 요구 | 생존 검사 구현 확인. 0행이면 `SUBSTRATE_UNCHECKED` |
| I-4 decode 커널 GPU 시간(엔진 측 decode 시간) | T1·T3(host 간격과 커널 시간 분리), V-diag, §1-7 대조 | rev1 시점 **구현 흔적 없음**(`holb_probe.py` 하나) | **O-2 결정: I-5·int64 엔진 묶음에 포함, 같은 ENGINE\_DIGEST**. PD-mux와 vanilla 양쪽 경로에서 엔진 측 decode 시간이 기록되는 것이 gate의 확인 항목. 미방출이면 `NOT_COLLECTED` + §1-7 fallback |
| I-5 거절 사유 | T4, V | `src/multiplex/admission_probe.py` CPU 검증 완료, **엔진 트리 미동기화**·GPU gate 전 | 결정 15에 이미 포함 |
| I-6 cudagraph 패딩(bs\_real, bs\_padded, hit/eager) | T1(캡처 목록 밖 bs 방지), T4, 예측기의 `bs_padded` | rev1 시점 **구현 흔적 없음**(`bs_padded` 없음) | **O-2 결정: 같은 digest에 포함**. 부팅 로그의 캡처 목록 검사 + CG canary는 보조로 유지 |
| I-2·I-3 | — | 이 PREREG에 포함하지 않음 | T2는 클라이언트 TTFT가 1차라 I-3 불필요. T3는 R11-2 정렬 없이 총효과로만 쓴다(M3와 같은 범위 문장) |


## 4. (H) hybrid 고유 항 대비 — Llama-3.1-8B (base, rev `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`)

원칙: **Llama는 이 표의 칸을 채울 때만 쓴다.** goodput 비교·모델 간 이중차분 없음. 대비 칸은 **같은 Job**(J-B) 안의 같은 엔진 모드(vanilla fused\_sym, flashinfer, cap 128, mem 0.6)에서 잰다. Nano 쪽 짝은 `108v`이다.

| 대비 칸 | Nano에서 재는 것 | Llama에서 재는 것 | 맨 끝까지 말할 수 있는 것 | 말할 수 없는 것 |
|---|---|---|---|---|
| ① 요청당 메모리 `s + k·L` | 부팅 배너 s, k, 풀(T4a) | 같은 항: s = 0, k | 부팅 로그 사실 두 줄(s·k 값), 산술 경계 `s/k ≈ 8.8k(GB)–9.4k(GiB)`[유도] | "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED, 구간 확장 금지) |
| ② 채우기 동시 수 | 채우기 실현 동시 수(T4b) L ∈ {4k, 16k} 쪽 | 같은 L 두 점(KV 한계) | "이 스택·mem 0.6·이 풀에서 L=…일 때 측정된 최대 동시 요청 수는 Nano …, Llama …이다. 가중치 바이트가 달라 풀 크기가 다르다" — 사실 병기까지 | "hybrid가 긴 L에서 배치 여유가 더 크다"를 일반 문장으로(측정 두 L·두 모델·교락 목록 안에서만) |
| ③ decode ctx 기울기 | `T_dec` 계수 c(`108v`; 엔진 측 시간 1차, 클라이언트 간격은 §1-7 대조 통과 bs만) | 같은 형 c, 짝 셀 9개: bs 1×{2k, 8k, 16k, 32k, 64k}, bs 8×{2k, 8k, 16k}, bs 48×{2k}. bs ≥ 8의 ≥ 32k 점은 Llama KV 풀 263,900 토큰으로 실행 불가(`INFEASIBLE_BY_RULE`) | 같은 (bs, c) 셀에서의 T\_dec 비와, 고정 bs(1, 8)에서 ctx 구간 기울기 비. 방향·형태 서술 | 층 종류 귀속("attention 4/56층이라 작다"는 [가설], 층 수·크기 교락), 크기 일반화 |
| ④ prefill L 의존 | T2 q (P=108, vanilla chunk −1) | 같은 q | `q` 라벨 두 개의 나란히 놓기: "Nano …, Llama … (L ≤ 64k)". T2-ext 통과 시 96k–124k 점 | "Llama가 64k 넘으면 초선형"(프로브 1–2회 기술값), "hybrid prefill은 선형" 일반화 |
| ⑤ cap 결합·거절 사유 | I-5 사유 몫(T4c): cap≡슬롯 | I-5 사유 몫: KV 위주(cap은 admission만) | 코드 사실(정적 풀, 설정으로 공시)과 사유 몫 기술값 | "슬롯 상한이 hybrid 고유"·"generic 스케줄러는 mamba state를 무시한다"·"Bullet/MuxWise가 C\_mem을 오예측한다" |
| ⑥ decode SM 민감도 | T1의 D 축 | **대비하지 않음**(Llama는 PD-mux 없이 `108v`) | Nano 안의 γ\_j 기술값만 | Llama와의 SM 민감도 비교 |

### 4-1. 교락 목록(공시 의무, 모든 대비 문장에 붙임)
모델 크기(9B vs 8B)·가중치 바이트(풀 크기가 달라짐, Nano 16.6 GiB vs Llama 약 15 GiB [유도])·층 수(56 vs 32)·**attention 층 수(4 vs 32 — 이것이 대비하려는 구성 자체이므로 교락이 아니라 처치와 뒤섞인 차이)**·MLP 형(Nano의 층 구성 vs gated SiLU)·hidden(4,480 vs 4,096)·RoPE(Nano attention 층은 NoPE로 알려짐[미확인]) vs RoPE·토크나이저(어휘·길이 분포; 이 PREREG는 `input_ids` 고정 길이라 길이 분포 영향은 없지만 어휘 크기에 따른 lm\_head 비용은 남음)·attention 백엔드(둘 다 flashinfer 명시)·학습/라이선스(base 동일). **Llama 한 모델 대 Nano 한 모델**이므로 "transformer 대 hybrid" 일반화 불가.

### 4-2. 허용 문장 틀
"이 Job(VESSL A100, mem 0.6, cap 128, vanilla chunk −1)에서 고정 (bs, ctx) 셀의 decode step 시간은 Nano-9B-v2 …ms, Llama-3.1-8B …ms였고, ctx 기울기는 …로 달랐다. 두 모델은 크기·층 수·attention 층 비율·MLP 형이 함께 다르다."
금지: "hybrid 고유"를 원인으로 쓰는 문장(Nano 1개 대 Llama 1개의 기술 차이까지만, §1-7), "~52k가 hybrid 경계", "9k 경계를 실측했다", 모델 간 goodput 비교 문장.


## 5. TP2 항 (N 항)

TP2 PD-mux는 M2G J2에서 정확성 gate를 통과했다(G0(ii)·G1·G3 PASS, `G2_NOT_TESTED`) [측정, `RESULT_m2g_rev11_jobs_2026-10-13.md`]. 이 PREREG는 TP2에서 **비용 항 측정**과 **샤딩 사실 확인**을 한다. 성능 비교(G\_sym 등)는 하지 않는다.

### 5-1. 샤딩 사실 확인 항목
| # | 확인 | 예측 [유도, config 산술] | 측정 방법 | 판정 |
|---|---|---|---|---|
| F1 | 슬롯당 mamba state s/N | N=2에서 약 71.8 MB/rank(143.5/2); 128 mamba heads·8 groups가 2로 나누어떨어짐 | TP2 부팅 배너 rank별 슬롯 바이트 | rank 0·1 일치 ∧ 예측의 ±2% |
| F2 | KV B/토큰 k/N | KV head 8개(k = 4 layers×2×8×128×2 B = 16,384 B와 일치) → 4개/rank → 약 8,190 B | 부팅 배너 rank별 | ±2%. (N=4 → 2개, N=8 → 1개, N>8은 복제. **N=2만 측정**) |
| F3 | rank당 풀과 C\_mem(TP2, L) | 산식 B `풀_rank / (s/2 + (k/2)·(L+R))`(R = 채우기 출력 4,096/8,000 `ignore_eos`), 원식 A는 R = 0; 가중치가 절반이라 풀이 약 39.7 GB/rank [유도] → 원식 A의 C\_mem(32k)가 약 117(N=1은 약 49), 산식 B 값은 부팅 로그 풀로 데이터 전에 계산해 봉인 | 부팅 + 채우기(T4, L ∈ {16k, 64k}, cap 128) | `CMEM_ACCT_HELD` 형(§1-4, 구현 사실 확인) |
| F4 | in\_proj 폭과 conv 커널 stride | 폭 22,656 → 11,328/rank(`2×5120 + 2×512 + 64`). int32 오프셋 경계 `2^31 / stride`: N=1 94,786(관측 T ≥ 94,793과 일치) → N=2 **189,572** [산술] | (a) 첫 prefill에서 conv 입력 `x.stride()` 로그(하네스·엔진 로그, [미등록]) (b) gate의 커널 단위 시험에 stride 11,328 변형 추가 (c) **TP2 서빙 96k 1점(Job 마지막)** | stride 로그가 11,328이면 F4 입력 확인. 96k 점의 의미는 §5-3 |
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


## 6. 검증 V — 조립 모델로 λ\*\_SLO 예측, 실측과 대조

### 6-1. 예측기(sealed 순수 CPU 프로그램, 이하 `cm_predict`)
- 입력: J-A·J-B에서 나온 맞춤 결과(T1 계수/표, T2, T3 표[ρ, σ, q\_stall], T4 상수), V 셀의 요청 집합(같은 trace hash: 입력/출력 길이열·`max_new_tokens`·`ignore_eos` 플래그), cap\_V, SLO 함수(A′: TTFT T = k·floor(L), ITL 요청 내부 p95 ≤ 50 ms; primary k = 5 [제안], 3·10은 기술값).
- 출력: arm·rate마다 SLO attainment 곡선 → SP R9-1(0.90 교차의 log-rate 선형보간)과 **같은 코드 경로**로 `λ*_pred`.
- 구조: PLAN §1(§1-6). agnostic static D = lockstep(`max(ρ·T_dec, σ·T_span_iter)`, §1-6 단위), fused\_tuned = **반복당 시간 = T\_dec(108v, bs, c̄) + [T2(pos+청크) − T2(pos)]**의 **합산 가정**(혼합 청크가 한 forward라 합산이 과대일 수 있음, 미검증 — 이 가정은 V가 시험한다). 통신 항은 N=1이라 없음.
- **동시 수 상한(rev2, 死因 1 수리)**: 요청 입장은 §1-4의 예약 회계 그대로 — 입장 시 요청별 `s + k·(L_i + R_i)`의 합이 풀을 넘지 않는 한에서만 running에 들이고, `R_i`는 `ignore_eos`이면 `max_new_tokens`, 아니면 `min(max_new_tokens, 4096) × new_token_ratio`. 따라서 상한은 `min(cap_V, 예약 회계가 허용하는 수)`이며 rev1의 `min(cap, C_mem(L))`(예약 없음)을 쓰지 않는다. 교훈 9에 따라 이 입장 규칙은 T4가 확인한 엔진 회계의 재현이라 V의 독립 예측력에 기여하는 부분이 아님을 공시한다.
- **cudagraph 패딩(MINOR C)**: 예측기의 T\_dec 입력 bs는 실행 bs가 아니라 `bs_padded = ceil(캡처 bs 목록)`(I-6 기록과 대조; 캡처 목록 `{1, 8, 32, 48, 64, 96}`과 부팅 로그 값). 패딩 때문에 T1 격자 bs가 실행 bs와 같다고 가정하지 않는다.
- **T1 외삽 처리(MAJOR 4)**: 예측기가 (bs\_padded, c̄) 조회 시 해당 D·모드의 **측정 셀 볼록껍질**(log bs, log c 평면)을 벗어나면 그 iteration에 `PRED_EXTRAPOLATED`를 붙인다. (셀, arm)마다 iteration 가중 비율 `f_ext`를 공시한다. 값은 선택된 함수형(F1/F2/F3 또는 표 보간 — 표 보간이면 외삽 금지 규칙에 따라 가장 가까운 경계 셀 값으로 클램프)이 주되, `f_ext > 10%`인 (셀, arm)은 `V_PASS`에 도달할 수 없다(§6-3 `V_PASS_UNREACHABLE`, 사유 코드 `EXTRAPOLATED`). shape C 영역(bs 55–80, c 11–20k)에서는 (64, 8k)·(64, 12k) 셀이 받치지만 c > 12k·bs > 48의 칸(64×16k는 KV 규칙상 비실행)은 껍질 밖이므로 **데이터 전에 `f_ext`를 계산해 V\_REG에 적는다**. 사전 계산에서 이미 > 10%이면 그 (셀, arm)의 V\_PASS 불가는 데이터 전에 선언된다.
- **metric cliff(MINOR E)**: 예측 attainment 곡선은 점 추정이 아니라 **trace 기반 확률 시뮬레이션**(T2·T1 적합 잔차 분포로 100회 재표집)으로 낸다. 0.90 교차 부근에서 TTFT가 SLO ±5% 안에 있는 요청 비율 > 10%이면 `CLIFF_ADJACENT_PRED`를 붙이고 그 (셀, arm)은 `V_PASS`에서 제외한다(실측 쪽 `CLIFF_ADJACENT`와 별개로 예측 쪽에도 적용).
- **duration 합산(F)**: λ\*\_meas가 라운드·boot를 합칠 때 duration은 **합산**(max 금지, CLAUDE.md 게이트 7). λ\*\_thr를 쓰는 경우 SP R5-1을 승계한다.
- **봉인 시점**: 구조(명세 문서와 코드)는 **J-A·J-B 결과 fetch 전에** 봉인한다(데이터를 보고 구조를 고르지 못하게). 맞춤 계수에서 나온 예측값은 J-A·J-B 분석 후 **V 제출 전에** `V_REG` addendum으로 봉인한다(§8-1).
- 소거 변형(전부 V 데이터 전에 예측 봉인, 기술값): (v1) ctx 항 `c_D = 0` (v2) KV 전용 C\_mem(s = 0으로 두는 **모델 변형**이며 어떤 시스템의 동작 주장이 아님) (v3) ρ = 1 (v4) 외삽 허용 (v5) F-whole(§1-6). 어느 항이 V 오차에 닿는지 가르는 용도다. 변형 이름에 시스템 명을 쓰지 않는다.

### 6-2. 셀·arm·cap·사다리
- 셀 2개: (a) **arXiv-Sum 자연 출력**(유지분 5,906건 ≤ 16,384 토큰, 출력 평균 약 203, SP §10) (b) **shape C** = arXiv-Sum 입력 × 강제 출력 4,096(`ignore_eos`). context\_length 20,480 [제안, Nano 유지분 최대 16,379 + 4,096 = 20,475 ≤ 20,478].
- **cap\_V = floor(풀 / (s + k·(L̄_자연 + R_자연)))** 한 값을 두 셀·모든 arm·G0 튜닝에 공통 적용 [제안]. `R_자연 = min(max_new_tokens, 4096) × new_token_ratio`(자연 셀은 `ignore_eos`가 아니다). 값은 부팅 로그와 T4(산식 B 판정) 후 `V_REG`에 등록 [미등록 — `max_new_tokens`, `new_token_ratio`, `L̄_자연`]. 민감도 [유도, 풀 33.04 GB·s 143.5 MB·k 16,379 B]: `L̄_자연` 7.1k/7.3k/7.5k × `R_자연` 0/203/717/2,867/4,096 → cap\_V = 127/125/124 · 125/124/122 · 121/120/118 · 107/106/105 · 101/100/99. rev1의 "약 124"는 `R = 203`·L̄ 7.3k에 해당한다. [CS-OK] 33.04는 풀 GB 값(우연 일치)
- **shape C KV 상한(死因 1, 같은 식)**: 구속 수 = `floor((풀 − cap_V·s) / (k·(L̄_in + 4,096)))`. 값은 `L̄_in`의 정의에 의존한다 — **불일치 [미등록, 2회차 감사에 제기]**: 감사서의 "약 60"은 `L̄ = 11.4k`를 입력 평균으로 읽고 예약 4,096을 더한 산정(cap 124에서 60)이고, rev1의 "약 81"은 11.4k가 입력 + 출력 합이라는 읽기(cap 124에서 81)다. COMPARE :15의 arXiv-Sum 입력 평균 7.1k를 쓰면 (7.1k + 4,096) 약 83이다. 필터(≤ 16,384) 후 입력 평균을 `ARXSUM_STATS`에서 확정해 `V_REG`에 적는다. 어느 읽기든 shape C에서는 KV가 cap\_V(99–127)보다 먼저 구속하므로 `min(cap, 회계 허용 수)`가 실제로 시험된다.
- arm(모두 같은 Job 안, 셀마다 boot): `fused_tuned` + **agnostic static D 후보 상위 2개**(셀마다 예측 λ\*\_pred가 큰 순서의 D ∈ {24, 34, 44}, 동률이면 큰 D) = 셀당 3 arm [O-5 결정: PD-mux arm 2개]. **"agnostic(최선 static D)"는 이 상위 2개 안의 실측 최선이다** — 예측이 놓친 세 번째 D가 더 좋았을 가능성이 있다(MINOR 3, 등록 caveat 3). D ∈ {16, 64}는 V arm 후보가 아니다. V arm의 sticky 모드는 §1-8(ON, 실현 SM 검증).
- `fused_tuned` = G0 튜닝 argmax config(SP R8-2d `fused_best11(k=5)`, 결정 11·15), **자연 셀에서 튜닝**한 값을 shape C에도 쓴다(범위 문장: "자연 셀 argmax config"; "shape C에서 튜닝된"이라 쓰지 않음, 게이트 8). **자연 셀 튜닝 편향(MINOR 6)은 G 문장에만 박는다**: fused\_tuned가 shape C에서 미튜닝이므로 shape C의 `G`(PD-mux 대 fused)는 PD-mux에 **유리한 쪽(분모 과소)으로 부풀 수 있다** — PD-mux 우위 문장에는 비보수적인 방향이다. shape C의 G는 "자연 셀 argmax config 기준이며 shape C 튜닝 fused는 미측정"이라고만 쓰고, 이 편향을 V 판정(λ\*\_pred 대 λ\*\_meas)의 허용 오차에 흡수시키지 않는다. mixed-chunk가 argmax이면 SP R8-2c 운용 규약과 G-M1–G-M4를 gate에 포함한다.
- **사다리(死因 2 수리, SP R9-3)**: **arm별로** 그 arm 예측 `λ*_pred(arm)`을 중심으로 인접 비 1.2, `j = −2, −1, 0, +1, +2`의 5 rung(`λ*_pred·1.2^j`)을 둔다. 공통 격자·기하평균 중심·폭 확장은 **없다**(rev1의 중심 = arm 기하평균 규칙 폐기). rung 절대값은 `V_REG`에 봉인. **오판 구간 산술**: 각 arm의 사다리 범위는 `×1.2^{−2} … ×1.2^{+2} = ×0.694 … ×1.44`, 즉 `e = ln(λ*_meas / λ*_pred) ∈ [−0.365, +0.365]`. `V_COARSE` 경계 `|e| ≤ ln 1.25 = 0.223 < 0.365`이므로 **25% 안에서 맞춘 어떤 (셀, arm)도 사다리 밖으로 나가지 않는다**. rev1에서는 예측 폭 w = 1.44일 때 `e ∈ (0.182, 0.223]`인 arm이 격자 밖(`V_FAIL_OUT_OF_LADDER`)으로 판정되는 오판 구간이 있었다(중심이 arm 기하평균이라 한 arm의 한쪽 여유가 `0.365 − ln w/2`까지 줄어듦). 이제 그 구간은 비어 있다. 보정 예측(`V_JOB_SHIFTED` 2차)도 ±3% 안에서 이동하므로 같은 사다리 안이다. **Job-0·확장 없음**.
- seed 4(판정 하한), 2 pass(seed {1,2}/{3,4}, pass 2는 arm 역순), n/rung/seed = 자연 60, shape C 45 [제안, rev0 승계]. R9-6 해상도(`2√(0.09/n)/|Δa|·log ρ`)를 (셀, arm)마다 `V_REG`에 공시하고, **해상도 > 5%인 (셀, arm)은 `V_PASS`에 도달할 수 없다**(§6-3 `V_PASS_UNREACHABLE`, 사유 `RESOLUTION`). shape C(n = 45)는 사전 계산에서 > 5%일 수 있고, 그러면 shape C의 `V_PASS`는 데이터 전에 불가로 선언된다(`RESOLUTION_COARSE`와 같은 뜻이며 더 이상 허용 오차 10%와 "별개"로 두지 않는다). warm-up은 전송하고 채점 제외(자연 20 s, shape C 120 s).
- **Job 분할(O-7 결정)**: 자연 셀 = **J-V1**(관측자 ON/OFF 블록 포함), shape C = **J-V2a**(pass 1, seed {1,2}) + **J-V2b**(pass 2, seed {3,4}, arm 역순). 각 Job 안에 그 셀의 arm 3개가 전부 들어 있어 arm 비교는 Job 안에서 완결된다. shape C의 λ\*\_meas는 두 Job의 seed 4개를 합쳐 R9-1로 구하고(Job 효과가 seed 사이에 섞임), 두 Job의 앵커 편차(§6-7)와 Job별 부분 곡선의 차를 기술값으로 병기한다.
- 관측자: SP §9에 따라 ON/OFF 짝 블록(n ≥ 8, PD-mux 계열·fused 계열 각각)을 J-V1에 둔다(자연 셀, 한 rung).

### 6-3. 판정 규칙 (데이터 전 등록, [제안] — 허용 오차 값은 감사 대상)
오차 `e = ln(λ*_meas / λ*_pred)`, 지표 `λ*_meas`는 같은 Job(shape C는 두 Job 합산)의 R9-1 보간, k=5. 단위 = (셀, arm). 판정 범위는 "이 Job들에서"(SP R5-4; Job 효과는 모른다 — §6-7로 일부 진단).

**전제 조건(`V_PASS` 후보 자격, 모두 만족해야 PASS 계열에 들어간다)**:
(P1) 모든 (셀, arm)에서 λ\* 해결(`LAMBDA_RESOLVED`, `CLIFF_ADJACENT` 아님, 예측 쪽 `CLIFF_ADJACENT_PRED` 아님) · (P2) `|e| ≤ ln 1.10` · (P3) 셀 안 arm 순서(λ\*\_agnostic 대 λ\*\_fused\_tuned의 부호)가 예측과 일치(3% 미만 차는 순서 판정 보류, 게이트 3) · (P4) `|ln G_meas − ln G_pred| ≤ ln 1.10` · (P5) **귀무 예측기보다 오차가 작다**(§6-8: 모든 (셀, arm)에서 `|e| < |e_N1|`, 셀마다 `|ln G_meas − ln G_pred| < |ln G_meas − ln G_N0|`) · (P6) 해상도 ≤ 5%이고 `f_ext ≤ 10%`(둘 다 데이터 전에 계산해 `V_REG`에 적은 값).

| 라벨 | 조건 (위에서 아래 순서로 첫 만족) |
|---|---|
| `V_PASS_UNREACHABLE` | (P6) 위반이 데이터 전(`V_REG`)에 확정된 (셀, arm)이 있음 — `V_PASS` 불가, 사유 `RESOLUTION`/`EXTRAPOLATED` 표기. 이 단위는 다른 라벨(`V_COARSE` 이하)로만 판정 |
| `V_JOB_SHIFTED` | 전제 (P1)–(P6) 만족 ∧ 어느 V Job의 내부 앵커 편차 > 3%(§6-7). 보정 예측 `λ*_pred^corr`로 같은 (P2)(P4) 검사를 2차 라벨로 병기(재스코어 아님: 보정 규칙은 §6-7에 데이터 전 봉인) |
| `V_PASS_CANCELLATION` | (P1)–(P6) 만족 ∧ 앵커 편차 ≤ 3% ∧ V-diag에서 `TERM_DEVIATES_{T1..T4}`가 **1개 이상**(항 오차가 상쇄되어 조립이 맞아 보일 수 있음) |
| `V_PASS` | (P1)–(P6) 만족 ∧ 앵커 편차 ≤ 3% ∧ **V-diag `TERM_DEVIATES` 0개** |
| `V_NOT_BETTER_THAN_NULL` | (P1)–(P4)·(P6) 만족 ∧ (P5) 불만족 |
| `V_COARSE` | 위 어느 것도 아님 ∧ 모든 (셀, arm)에서 `|e| ≤ ln 1.25` ∧ 해결 |
| `V_FAIL` | 어느 (셀, arm)이든 `|e| > ln 1.25` 또는 순서 불일치(3% 이상) |
| `V_FAIL_OUT_OF_LADDER` | 어느 (셀, arm)에서 교차가 그 arm의 사다리 밖(`BELOW/ABOVE_LADDER`). 범위가 ×0.694–×1.44이므로 `|e| > 0.365`의 **하한 정보**만 있다(이제 `|e| ≤ 0.223`에서는 발생 불가) |
| `V_UNRESOLVED` | 서버 사망(`SERVER_DEATH`)·`UNRESOLVED_J`·`CLIFF_ADJACENT`·`UNRESOLVED(sm_not_realized)`로 해결 못 한 칸이 있고 나머지가 `V_COARSE` 이내 |

전체 V 라벨은 모든 (셀, arm)·모든 V Job 단위 라벨 중 **가장 낮은 것**(`V_PASS` > `V_PASS_CANCELLATION` > `V_JOB_SHIFTED` > `V_NOT_BETTER_THAN_NULL` > `V_COARSE` > `V_UNRESOLVED` > `V_FAIL_OUT_OF_LADDER` > `V_FAIL`; `V_PASS_UNREACHABLE` 단위는 최대 `V_COARSE`)로 한다.

- 10%의 근거(제안): 관측된 PD-mux 이득 크기가 +27–40%대이고(CONSENSUS §1-1) 예측 오차가 그 1/3을 넘으면 모델이 G를 "설명한다"고 말할 수 없다. 25%는 사다리 범위(±0.365)의 안쪽 경계다. **10%는 KISTI 타 모델의 G 크기에서 빌린 값이며 이 기판·이 모델의 근거가 아니다**(등록 caveat 2). 정당화는 감사 쟁점 (2).
- **n=2 round와 판정 n(OK 5)**: 항 맞춤(T1–T4)은 **기술 적합**이고 게이트 3의 n ≥ 4는 **판정**(V)에 적용된다. V의 seed는 4(shape C는 두 Job에 걸쳐)이며 판정 문장은 "이 Job들에서"로 쓴다.

### 6-4. 실패해도 결과로 쓰는 문장 (사전 등록)
- `V_COARSE`/`V_FAIL`/`V_FAIL_OUT_OF_LADDER`/`V_NOT_BETTER_THAN_NULL`: "봉인된 예측기(T1–T4 + 조립 구조)는 이 Job들의 (셀, arm)에서 λ\*\_SLO를 …% 빗나갔다(부호 …). 항 측정값은 §1의 범위에서 유효하며, **조립이 검증되지 않았다.** 이 PD-mux 이득(G)은 이 모델이 설명하는 것으로 주장하지 않는다." 허용되는 진단: V-diag(아래) 표.
- `V_PASS`: "이 Job들의 두 셀·세 arm에서 λ\*\_SLO가 봉인된 예측의 ±10% 안이었고 귀무 예측기보다 가까웠으며 V-diag 항 이탈이 없었다. 이 기판·이 스택·이 두 workload에서의 예측 일치이며, 다른 모델·trace·기판으로의 일반화나 hybrid 고유성 귀속이 아니다. **허용 오차 10%는 KISTI 타 모델 G에서 빌린 값이다.**"
- `V_PASS_CANCELLATION`: 위 문장에 "V-diag에서 항 `…`이 이탈해 일치가 항 오차의 상쇄일 수 있다"를 덧붙이고 "조립이 검증되었다"로 쓰지 않는다. `V_JOB_SHIFTED`: "이 Job의 내부 앵커가 J-A·J-B 값에서 …% 이동해 항 값이 같은 기판 상수라고 가정할 수 없다. 보정 예측 기준으로는 …"
- 두 경우 모두 예측기 수정은 **새 PREREG·새 Job**이다(재스코어 금지, 게이트 8).
- **J1 기울기 비교(MINOR D)**: J-A의 P = 108 T2 기울기와 J1 floor 기울기(0.111 ms/토큰, L ≤ 16k)의 비교는 J1이 mem 0.82 결과이므로 **기술값**으로만 공시하고 판정·보정에 쓰지 않는다.
- **전력 상태 caveat(G)**: 항 측정(T1–T4)은 고정 배치·저부하의 전력 상태에서, V는 power cap 노출이 높은 서빙 부하에서 잰다. 두 전력 상태의 차이가 `e`에 섞일 수 있다(예측기는 ρ 안의 클럭 하락만 포함). V-diag의 클럭·전력 공변량과 SW power cap 노출률을 V 보고에 병기한다.
- **V-diag**(사후, 기술값, 재스코어 아님): V 텔레메트리로 모델 내부 양을 실현값과 대조한다. (1) 실현 decode bs 분포 대 예측 (2) 실현 step 시간(**엔진 측 I-4**, 클라이언트 간격이 아니라) 대 `T1(bs_padded, c̄)` 비 (3) span 시간 대 T2 (4) 동거 step 비 대 ρ, span 비 대 σ (5) 실현 동시 수 대 예약 회계 허용 수 (6) 같은 Job 실측 floor 대 T2 floor. `|ln 비| > ln 1.10`인 항에 `TERM_DEVIATES_{T1..T4}`를 붙인다 — 원인 후보 정렬용이며 인과 판정이 아니다. **`V_PASS`의 조건으로 쓰인다(위 표)**.

### 6-5. V를 읽는 금지 문안
"G가 비용 모델로 설명됐다/예측됐다"(V\_PASS도 두 셀·한 모델의 예측 일치까지만) · "기전 귀속" · J1 B·X arm 비의 크기 문장 · "PD-mux 우위"(G 크기는 이 PREREG의 estimand가 아님) · "G4(장문 prefill 중 ITL 기술값)는 ITL SLO 충족" · "agnostic이 최선 static D다"(상위 2 D 중 최선만) · "비용 모델이 C\_mem을 예측했다"(구현 사실 확인) · `V_PASS_CANCELLATION`/`V_JOB_SHIFTED`를 `V_PASS`로 인용.

### 6-6. 합성 대표성·64k 범위(MINOR 12·13)
T1·T2의 합성 입력(`input_ids`, 내용 무관)과 V의 실제 arXiv-Sum 입력 사이의 대표성은 G7 허용 문안 6의 범위(**단일 요청 prefill 시간까지**)에서만 주장한다. 64k 격자와 T2-ext(96k·124k, 아래 §1-2·§5-3)는 프로브 감사 허용 문안 3–4를 승계한다("64k는 격자 선택, 한계는 chunk −1 Nano arm 공통, PD-mux 고유 아님").

### 6-7. J-V 내부 앵커 (rev2 신설, 감사 MAJOR 2)
다른 Job의 항 값을 V 예측에 넣으면 Job 효과가 `e`에 섞이므로(§0 자기모순), **각 V Job 안에서 같은 Job의 항을 다시 잰다**.
- 앵커 boot(Job당 2개, 예측 대상 arm 쪽 D 하나와 `108v`): ① **V의 PD-mux arm 중 예측 λ\*가 큰 D**(`D_a`) PD-mux boot ② `108v` vanilla boot. cap 128, 나머지는 J-A의 T1 boot와 같은 config 바이트.
- 측정: **T1 4셀** `(bs, c)` ∈ {(8, 2k), (32, 8k), (48, 8k), (64, 8k)} [제안; 모두 J-A 격자 안] 각 2회, **T2 3L** ∈ {4,096, 16,384, 65,536} × 8회(앞 1회 제외) + T2′(L ∈ {4,096, 16,384}, §1-8).
- 판정: 앵커 셀별 `|ln(J-V 앵커 / J-A 해당 라운드 값)|`의 중앙값을 `δ_Job`로 둔다(J-A 두 Job의 같은 셀 평균, J-A1/J-A2의 차를 Job 간 산포의 참고치로 병기). **`δ_Job > ln 1.03`이면 그 Job에 `V_JOB_SHIFTED`**, 보정 예측 `λ*_pred^corr` = 각 항을 앵커 비(T1·T2별 중앙 비)로 곱한 항으로 `cm_predict`를 다시 돌린 값을 **2차로 병기**한다. 보정 규칙(곱셈 보정, 항별 비)은 이 문서로 봉인하며 앵커 데이터를 본 뒤 바꾸지 못한다.
- 비용: Job당 약 0.4 GPU-h(§7).

### 6-8. 귀무 예측기 (rev2 신설, 감사 MAJOR 1)
V 데이터 전에(`V_REG`) 봉인하고 `V_PASS`는 이보다 작은 오차를 요구한다.
- **N0**: `G_pred = 1`(PD-mux와 fused가 같은 λ\*를 갖는다는 귀무). 비교: `|ln G_meas − ln G_pred| < |ln G_meas|`.
- **N1**: T1 단독 예측기 — prefill·동거·큐잉·메모리 회계 없이 `λ*_N1(arm) = cap_V / (E[out] · T1(arm 모드, bs = cap_V, c̄))`(decode만의 처리 한계; fused는 `108v`). 비교: `|e| < |e_N1|`. 세부 상수(`c̄`, `E[out]`)는 `V_REG` [미등록].
- 목적: 조립(T2·T3·T4와 DES)이 T1 단독보다 정보를 더하는지 확인한다. 통과해도 "조립이 검증되었다"가 아니라 "T1 단독보다 낫다"까지다.


## 7. Job 구성, 시간, GPU-h (Job마다 ≤ 6 h 목표; 사용자 메모 "GPU Job 적정 크기": 동시 사용 GPU 수에 맞춘 스펙과 사용률)

### 7-1. 시간 모형 [유도 — 측정 아님, Job 크기 결정용] (재현은 이 문단의 파라미터로)
boot 200 s, boot별 warm-up 60 s(=boot당 0.072 h), 셀 오버헤드 15 s, decode 창 300 token, T1 셀실행 평균 약 50 s [임의]. 단일 요청 prefill TTFT [기술값, 프로브·J1]: 1k 0.12 s, 2k 0.24, 4k 0.46, 8k 0.92, 16k 1.84, 32k 4.0, 64k 8.4. prefill SM 감소 배율 `(108/P)^0.8` [임의 중심값, T2 전에 모름]. step 시간 `(12 + 0.45·bs) ms × (108/D)^0.5` [임의]. TP2 prefill 속도 1.7× [임의], TP2 boot 400 s. V 셀 평균: 자연 125 s, shape C 370 s [rev0 모형 승계, 임의]. T4 채우기 1회 약 4–8 min [임의]. **모든 시간은 Job 크기 결정용이며 측정 전 값이다.**

### 7-2. 분할 원칙
- **반복은 여러 Job**(운영 모델): J-A는 **round별 2 Job**으로 나눈다. D 사다리(비교 단위)는 각 Job 안에서 완결되고(모든 D·`108u`·`108v`가 각 Job에 있음) round 간 차는 Job 효과를 포함한다 — `108v`·`108u` 같은 셀의 J-A1/J-A2 차가 Job 효과의 참고치다(§6-7과 공유). round 2는 round 1의 역순(ABBA).
- J-V는 **분할**(O-7): 자연 셀 1 Job, shape C는 pass별 2 Job(§6-2). 각 Job에 그 셀의 arm 3개가 전부 있다. 각 Job에 내부 앵커 2 boot(약 0.4 GPU-h).
- 목표 상한 6 h: 모든 Job 기대 시간 ≤ 4.4 h(하드캡 ×1.2 ≤ 5.3 h). 정지 선례(J2·J4·J8이 약 9.5 h에서 외부 정지, 원인 미확정)보다 충분히 아래다. 재개 가능(셀 단위 DONE·`vessl_jobs.tsv`·sha).

### 7-3. Job 표
| Job | 내용 | 스펙 | boot | 셀/반복 | 예상 h [유도] | 하드캡(×1.2) | 예약 GPU-h | 사용 GPU-h(사용률) |
|---|---|---|---|---|---|---|---|---|
| **J-A1** Nano D 사다리 round 1 | T1(D 16·24·34·44·64 + 108u + 108v, round 1) + T2·T2′ + T3(D 24·44, round 1분) + cap 불변성 boot(D=44, cap 48) | `resourcespec-a100x1` | 8 | T1 128 셀실행(고유 셀 128 = 118 + 신규 10) · T2 42 셀 + T2′ 14 · T3 16 실행 | 3.7 | 4.4 | 3.7 | 약 3.4 (92%)† |
| **J-A2** Nano D 사다리 round 2 | 같은 boot 7개, 역순(cap 불변성 boot 없음) | `resourcespec-a100x1` | 7 | T1 128 · T2 42 + T2′ 14 · T3 16 | 3.5 | 4.2 | 3.5 | 약 3.2 (92%)† |
| **J-B** Nano-Llama 대비 + T4 | Nano `108v` 4 round + Llama 4 round(T1+T2) + **T4 Nano 9 boot(12 fill)** + T4 Llama 1 boot(2 fill) | `resourcespec-a100x1` | 18 | T1 Nano 80(신규 (64,·) 셀 8 포함)·Llama 36 셀실행, T2 48 셀 | 4.4 | 5.3 | 4.4 | 약 4.0 (91%)† |
| **J-C** TP2 | 샤딩 확인 F1–F5, T1·T2 TP2/앵커, T4 TP2, 마지막 96k | `resourcespec-a100x2` | 10 (TP2 6 + 앵커 4) | T1 96 셀실행 · T2 32 셀 | 2.5 | 3.0 | 5.0 | 4.0 (81%)‡ |
| **J-V1** 검증 자연 셀 | 3 arm × 5 rung(arm별 중심) × 4 seed + 관측자 ON/OFF 블록 + 내부 앵커 | `resourcespec-a100x1` | 12 (arm 6 + 관측자 4 + 앵커 2) | 60 셀(arm당 20) 3,600 요청 | 3.8 | 4.6 | 3.8 | 약 3.6 (95%)† |
| **J-V2a** 검증 shape C pass 1 | 3 arm × 5 rung × seed {1,2} + 내부 앵커 | `resourcespec-a100x1` | 5 | 30 셀(arm당 10) 1,350 요청 | 3.7 | 4.4 | 3.7 | 약 3.5 (95%)† |
| **J-V2b** 검증 shape C pass 2 | 3 arm × 5 rung × seed {3,4}(arm 역순) + 내부 앵커 | `resourcespec-a100x1` | 5 | 30 셀 1,350 요청 | 3.7 | 4.4 | 3.7 | 약 3.5 (95%)† |
| **합계** | | | | | | **약 33.3 GPU-h(캡 합)** | **약 27.8 GPU-h** | 약 25.2 |

† 단일 GPU Job: 측정 중 GPU 점유는 거의 100%, 부팅·클라이언트 대기·drain 때문에 평균 약 90–95% [유도].
‡ TP2 Job: TP2 boot는 2장 모두, N=1 앵커 boot는 1장만 쓴다(`(2×TP2 시간 + 1×앵커 시간)/(2×Job 시간)`). 개선안(O-6): 앵커를 두 GPU에서 독립 서버 두 개로 동시에 돌려 사용률을 약 100%로 올릴 수 있으나 전력·열을 공유해 두 앵커가 서로 교락한다(J1 GPU0 throttle 61% 선례).

시간 유도(요약): J-A 한 round = rev1 J-A 6.4 h의 반(3.2) + 신규 셀 (64, 8k)·(64, 12k) 5모드 10 고유 셀 × 50 s ≈ 0.14 h + T2′ 수집 ≈ 0.15 h, J-A1은 cap 48 boot(약 0.2 h)이 추가. J-B = rev1 3.6 + T4 판별·고정점 부팅 4개 추가(boot 4 × 0.072 + 채우기 4회 × 약 6 min ≈ 0.7 h) + 신규 108v 셀 0.1 h. J-V1 = 자연 셀 60 × 125 s(2.08 h) + boot 12 × 0.072(0.87) + 관측자 블록(0.6) + 앵커(0.25). J-V2a/b = 30 × 370 s(3.08 h) + boot 5 × 0.072(0.36) + 앵커(0.25).

### 7-4. 합계 재계산(rev1 약 24.5 GPU-h 대비)
- rev1 24.5 = 단일 항 10.0(J-A 6.4 + J-B 3.6) + TP2 5.0 + V 9.5. rev2 = 단일 항 **11.6**(J-A1 3.7 + J-A2 3.5 + J-B 4.4) + TP2 **5.0**(2장 × 2.5 h) + V **11.2**(J-V1 3.8 + J-V2a 3.7 + J-V2b 3.7) = **약 27.8 GPU-h**. 하드캡 합 약 33.3.
- 늘어난 이유: ① 신규 T1 셀 + T2′ 상시 수집(J-A, +0.6) ② T4 판별·고정점 부팅(+0.8) ③ V 분할(boot·앵커 중복, +1.7: 앵커 3 × 0.4 + 분할로 늘어난 boot).
- 선택지(O-5 결정 반영): V의 PD-mux arm 2개는 이미 반영. 하드캡 합(×1.2) 약 33.3.
- **이 합계 밖 의존 비용**: ① 분할 GPU gate G-α → G-β → G-γ(engine-porter 설계 중, 범위 [미등록]) ② G0 튜닝 Job(fused\_tuned 확정, G0 PREREG 몫, SP R8-2: 11 config) ③ Workspace 하네스 smoke(`pdmux-probe`, 비측정, 상한 약 1.5 h [유도]) ④ TP2 검증 셀(V의 TP2 판)은 이 PREREG에 없음.

### 7-5. 16 h 캡과 Job 분할 사유
모든 Job 기대 시간 ≤ 4.4 h로 6 h 목표 안이다. J-A를 합치면 약 7.2 h로 목표를 넘고, 하나로 합친 J-V(rev1 9.5 h)는 정지 선례와 같다. J-A·J-B 분리로 H 대비(Nano `108v` 대 Llama)는 J-B 안에서만, D 사다리는 각 J-A Job 안에서만 성립한다. `108v`가 J-A1·J-A2·J-B에 모두 있어 Job 효과 진단값을 겸한다. 정지 시: 이미 끝난 (셀, arm)은 회수하고 나머지는 새 job name으로 재실행(실패 셀만 재실행한 사실은 결과 보고에 적는다).


## 8. 선행 조건 순서와 R8-2 개정

### 8-1. 순서(각 단계는 앞 단계가 끝나야 시작; 사용자 지정 순서 승계)
1. **GPU correctness gate — 분할 gate G-α → G-β → G-γ(계열별) 전부 PASS** (결정 16): 단일 장시간 Job과 의도적 crash를 섞지 않도록 비교 단위로 분할한다. 각 Job은 자기 비교를 Job 안에 완결하고, **앞 단계가 PASS일 때만 다음 단계를 제출**한다. 세부 설계(단계 내용·계열 구분·시험 목록·GPU-h)는 **engine-porter 문서(설계 중, 경로 [미등록])**를 참조하며 이 PREREG는 재서술하지 않는다. 분할 gate는 **한 번의 ENGINE\_DIGEST 변경**(결정 15·16)을 전제로 한다: 엔진 묶음 = I-5 엔진 동기화 + Triton causal\_conv1d **int64 패치(`chunk_offset`과 `sequence_start_index` 두 곳 — `chunk_offset`만으로는 부족하다: conv\_state 꼬리 읽기가 남는다)** + **I-4(decode 커널 GPU 시간)·I-6(cudagraph 패딩)**(O-2 결정). 현재 worktree `prefill-layer-alloc-engsync` 브랜치 `engine-sync-i5-int64`(`b0f6cfeb`)이며, I-4/I-6 추가로 **예상 digest가 다시 바뀐다**. PREREG의 digest 칸은 **"gate 통과 digest를 봉인 시점에 기입"**하는 자리표시 `ENGINE_DIGEST = [미기입]`이다. gate에 포함되어야 하는 이 PREREG의 요구: ON/OFF 출력 일치, 관측자 TOST(±1.5%, n ≥ 8 짝, PD-mux·fused 각각), 변이 시험, 패치 반증 실험(감사 2번 ①–③: 패치 후 fused\_sym 96k·124k 통과 + 되돌리면 95,100 재실패 + 단위 시험(12288, 95100)·stride 22,656 + 94,784/94,800 경계 2회씩), stride 11,328 변형(§5-3 (가)), vanilla 경로에서 엔진 측 decode 시간 방출 확인(§1-7), 실현 SM 로그(§1-8). 사후에 digest를 바꾸면 이후 단계가 다시 시작이다. **96k·124k 확장(T2-ext, TP2 96k)은 int64 gate(두 곳 패치) 통과를 조건으로 한다(O-3).**
2. **하네스**: 새 하네스(§11-2) CPU 시험·변이·가짜 엔진 e2e → Workspace smoke(비측정, 수치 인용 금지).
3. **G0 튜닝 Job**(별도 PREREG): fused\_tuned config를 cap = cap\_V에서 정한다(R8-2′).
4. **감사**: 규칙층(이 문서 §11 쟁점) → 하네스층. 각 최대 2회(규칙층은 1회차 NO-GO, 이 rev2가 2회차 대상).
5. **봉인 S1**: 규칙·하네스·엔진 digest(gate 통과 digest)·config digest. (V의 예측기 구조 명세·코드·귀무 예측기 구조도 여기서 봉인.)
6. **CPU preflight** → **`launch.sh` dry-run**(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit`. J-A1·J-A2·J-B·J-C는 독립 제출 가능(fused\_tuned 불필요, 단 J-A·J-B는 gate 전부 PASS 뒤). J-V1·J-V2a·J-V2b는 아래 S2 뒤.
7. **fetch**(`fetch.sh`, DONE + sha256) → result-analyst가 T1–T4 맞춤(분석 코드는 fetch 전에 커밋, 맞춤 규칙은 이 문서).
8. **봉인 S2 `V_REG`**: cap\_V, fused\_tuned config, agnostic D 후보 상위 2개, arm별 rung 절대값, 예측값 `λ*_pred`(소거 변형 포함), 귀무 예측기 값 `λ*_N1`·`G_N0`, 해상도·`f_ext` 사전 계산값, 앵커 보정 규칙, `L̄_in` 확정(§6-2 불일치), sha. **J-V 데이터 전에** 커밋.
9. J-V1 → J-V2a → J-V2b dry-run → 승인 → submit → fetch → 판정(§6-3).
B5 선례(운영 모델): Job 1개가 `fetch.sh` 통과까지 간 선례가 있다. dry-run 때 확인한다.

### 8-2. R8-2′ 개정 문안 (사용자 승인 완료: 결정 16 O-1; 규칙층 감사 2회차 대상)
SP R8-2 현행: "…(cap = G0 cap 고정, R16-1)… 바꾸지 않는 것: `max-running-requests` = G0 cap(R16-1), radix off, overlap on, mem-fraction 0.82…".

**R8-2′ (SP 개정 문안)**:
1. **cap**: fused\_tuned 튜닝과 판정은 **같은 cap = cap\_V**에서 한다. `cap_V = floor(풀 / (s + k·(L̄ + R)))`(§6-2, **출력 예약 R 반영**; `R` = 자연 셀은 `min(max_new_tokens, 4096) × new_token_ratio`, `ignore_eos` 셀은 `max_new_tokens` 전량)이며 `풀, s, k`는 부팅 로그, 산식 B는 T4가 구현 사실로 확인한다. cap\_V는 G0 cap 결정 규칙(R16-3)이 아니다.
2. **mem-fraction**은 **0.6**으로 통일한다(결정 13; 0.82는 모든 arm·블록에서 사용하지 않음).
3. **동시 변경 공시**: `mem 0.82 → 0.6`과 `cap = G0 cap → cap_V`는 **두 변수가 동시에** 바뀐다. 이전 0.82 시기의 G0·fused config를 승계하지 않고 새 조건에서 **다시 튜닝**하므로 교락된 비교는 없다(튜닝 결과를 0.82 결과와 비교하지 않는다).
4. 튜닝 workload는 자연 arXiv-Sum 셀이고, shape C의 fused\_tuned는 "자연 셀 argmax config"로 표기한다(편향 방향은 §6-2: PD-mux 우위 문장에 비보수적).
5. 튜닝은 V보다 먼저 별도 Job에서 하고 V에서 같은 config를 다시 잰다(승자의 저주 회피 유지).
6. SLO(A′, ITL 50 ms)·후보 11 config·엄격 argmax·동률 순서·mixed-chunk 운용 규약은 불변.
7. S-2(튜닝 공간 비대칭 11 대 PD-mux 상위 2 D)는 **더 커진다 — 같은 금지 문장 적용**: "PD-mux가 fused보다 못하다"를 PD-mux 미튜닝의 귀속으로 쓰지 않는다.
8. **R8-2c 예시 재계산**: mixed-chunk 운용에서 청크 예산 `C` 중 decode 몫을 cap\_V 슬롯으로 보면 prefill 몫 = `C − cap_V` 토큰이다. rev1 cap\_V = 124 가정에서 **C = 512 → 388 토큰, 388/512 = 75.8%** (SP R8-2c의 옛 예시 c = 48은 cap 48 가정이며 폐기). 이 수치는 rev1의 cap\_V 값이다 — 예약 반영 cap\_V(§6-2 민감도 99–127)가 `V_REG`에서 확정되면 같은 식으로 다시 계산해 공시한다(예: cap\_V = 100이면 412 토큰, 80.5%). "작은 예산 공시" 문구는 이 재계산값으로 쓴다.

**R16-3′**: SP R16-3의 "후보 {48, C\_mem/2, C\_mem}에서 G0 cap 결정 규칙 [미등록]"은 이 PREREG가 cap 선택을 goodput 스윕이 아니라 C\_mem 산식(예약 반영)으로 대체한다. **T4가 R16-3(cap 특성화 Job)을 대체한다**(결정 16 O-1 승인). R16-3의 CM-5 검증 기능은 T4의 "구현 사실 확인"(산식 B)과 거절 사유 몫이 맡는다.


## 9. 라벨과 금지 문안

### 9-1. 라벨
항: `FORM_INADEQUATE_T1_D{d}`·`FORM_INADEQUATE_T2_P{P}`·`CAP_INVARIANT`/`CAP_DEPENDENT`·`CLOCK_SENSITIVE`·`PREFILL_SUPERLINEAR_SCOPED`/`PREFILL_NEAR_LINEAR_SCOPED`/`Q_UNRESOLVED`·`T2_J1_SLOPE_DEVIATES`·`RHO_D_INTERP`·`RHO_CLAMPED`·`SIGMA_NOT_COLLECTED`·**`CMEM_ACCT_HELD`/`CMEM_ACCT_MISS`/`CMEM_ORIG_OVERPREDICTS`**(rev1의 `CMEM_PRED_HELD`/`MISS` 대체)·`C_MEM_DRIFT`·`FILL_INCOMPLETE`·`INFEASIBLE_BY_RULE`·`TP2_NOT_BOOTABLE`·**`CLIENT_ENGINE_AGREE_bs{b}`/`CLIENT_ENGINE_MISMATCH_bs{b}`/`CLIENT_ONLY_UNVERIFIED`**·**`SM_NOT_REALIZED_D{d}`/`T2_P_NOT_REALIZED`/`T2_FALLBACK_BS1_COEXIST_D{d}`/`STICKY_MODE_MISMATCH`**·`PRED_EXTRAPOLATED`. 검증: `V_PASS`/**`V_PASS_CANCELLATION`**/**`V_JOB_SHIFTED`**/**`V_PASS_UNREACHABLE`**/**`V_NOT_BETTER_THAN_NULL`**/`V_COARSE`/`V_FAIL`/`V_FAIL_OUT_OF_LADDER`/`V_UNRESOLVED`·`TERM_DEVIATES_{T1..T4}`·`CLIFF_ADJACENT_PRED`. 승계: SP 계열 `LAMBDA_RESOLVED`·`BELOW/ABOVE_LADDER`·`CLIFF_ADJACENT`·`SUBSTRATE_WARNING`(드라이버 변경)·`SUBSTRATE_UNCHECKED`·`SERVER_DEATH`(rev0 §6-1 정의: 죽은 (arm, cap, L)은 `UNRESOLVED(arm_death)`로 두고 다른 arm 결과를 지우지 않으며 일반화하지 않는다)·`CLIENT_ERROR`·`SERVER_ERROR_HI`·`NOT_COLLECTED`(I-4·I-6, vanilla 경로 미방출 포함).

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
- V: §6-5. `V_PASS_CANCELLATION`·`V_JOB_SHIFTED`를 `V_PASS`로 인용.
- 선행 gate 전: "I-5가 거절 사유를 측정했다"(동기화·GPU gate 전), "preflight PASS가 이 하네스를 검증했다"

## 10. 열린 결정 (사용자 몫)
**결정 16으로 닫힌 것**: O-1(R8-2′·R16-3′ 승인, T4가 R16-3 대체) · O-2(I-4·I-6 엔진 묶음에 포함) · O-3(96k·124k는 int64 gate 통과 조건부로 포함) · O-5(V PD-mux arm 2개) · O-7(J-V 분할). 아래는 **미결**이다.
- **O-4** TP2 96k 점의 의미: (가) 단위 시험으로 경계 이동 + 96k는 정상 점(권고, 단위 시험은 두 곳 패치 변형 포함) / (나) 패치 토글 env 도입.
- **O-6** TP2 앵커: 순차 1장(사용률 81%) / 두 GPU 동시 독립 앵커(≈100%, 교락 공시).
- **O-8** T4 arm 대칭 점검(fused\_sym 한 boot 추가, +약 0.3 h).
- **O-9** 허용 오차 10%/25%와 T1 적합 채택 기준(3%/8%/10%)이 모델 크기에 비해 타당한지(감사 쟁점 2와 같이). 앵커 임계 3%·해상도 5%·`f_ext` 10%·귀무 비교 규칙의 타당성 포함.
- **O-10** primary k = 5, 소거 변형 5개 범위, 라운드 수(기본 2, `108v`·Llama 4).
- **O-11 (rev2 신규)** shape C를 pass별 2 Job(J-V2a/b, 각 3.7 h, 앵커 중복 0.4 GPU-h 추가)으로 나눈 것 대 1 Job(약 7.0 h, 6 h 목표 초과): 기본은 2 Job. seed가 Job에 걸쳐 Job 효과가 seed 산포에 섞이는 비용을 감수.

## 11. [미등록] 목록과 감사 쟁점

### 11-1. 규칙층
[미등록]: cap\_V 값(`max_new_tokens`·`new_token_ratio`·`L̄_자연` 포함) / shape C KV 상한의 `L̄_in` 정의(60 대 83 불일치) / `V_REG`(fused\_tuned config, D 후보 상위 2개, arm별 rung 절대값, 예측값, 귀무 예측값, 해상도·`f_ext` 사전값, 보정 규칙) / 예측기 명세(`SPAN_TOKENS`, `T_span_iter` 필드, 청크 합산 가정, 통신 없음) / I-4·I-6 필드 이름·정의(engine-porter 문서) / 분할 gate G-α/β/γ 세부(engine-porter 문서) / ENGINE\_DIGEST(gate 통과 digest) / X\_T1(c\_in = c − X)와 O(출력 길이) / m(L)(T3 prefill 입력 수) / 채우기 출력 길이 L별 확정값 / T1 외삽 시험 수준 순서 / 소거 변형 정의 세부 / 귀무 예측기 N1 상수(`c̄`, `E[out]`) / 허용 오차(O-9) / 적합 채택 기준 / config digest(D=64 index 추가본) / 실현 SM 허용 규칙 / Llama 대비 짝 셀의 최종 목록(부팅 로그 평가) / 보고서 파일 이름 / 하드캡 값 / fused 대칭 점검(O-8) / TP2 rank 지연 지표 / 서버 `new_token_ratio` 부팅 로그 값.

감사 쟁점 (2회차는 **수리 diff만** 본다 — 범위는 §R 표와 맨 끝 diff 범위 절):
1. **예측-실측 대조가 "비용 모델 검증"인가**: 귀무 예측기(N0·N1)와 `V_PASS_CANCELLATION`이 조립 귀속 불가·상쇄 위험을 충분히 막는가. `V_PASS` 문장이 한계를 담았는가.
2. 허용 오차 10%/25%의 정당화(근거는 G 크기 +27–40%와 사다리 범위), 사후에 바꿀 수 없게 봉인되는가. Job 효과가 내부 앵커 3%로 진단되는가.
3. "agnostic(최선 static D)"가 상위 2개 D 중 최선일 뿐이다 — 범위 문장(등록 caveat 3)이 충분한가.
4. **T1 외삽**: 신규 셀 (64, 8k)·(64, 12k)와 `PRED_EXTRAPOLATED`·`f_ext` 10% 규칙이 shape C 영역을 다루는가.
5. n: T1·T2 D≠108은 round 2(= J-A1/J-A2), H 대비는 round 4, V는 seed 4. 게이트 3은 판정(V)에 적용되고 항 맞춤은 기술 적합이라는 구분(OK 5).
6. fused\_tuned를 자연 셀에서만 튜닝하고 shape C에서 쓰는 비대칭(G 문장에만 편향 방향 서술).
7. Llama 대비의 교락 목록과 허용 문장, 클라이언트 간격 대조(§1-7)의 fallback 문장.
8. T3: 총효과 정의, 전력 한도 교락, 동거 창 128 iteration 조건이 L=8k에서 만족되는지, T\_span 단위(F-iter 대 F-whole) 봉인이 충분한가.
9. T4: 산식 A·B 동시 봉인, 판별 부팅(cap = C\_A) 설계, 항등식 공시, `FILL_INCOMPLETE`, 정적 풀 서술이 결정 8′의 범위 안인가. **shape C KV 상한 60 대 83 불일치**의 해소.
10. R8-2′가 SP R8-2·R16-1·R16-3과 충돌하지 않는가(동시 변경 공시, R8-2c 재계산).
11. 함의 한계: T1의 c\_D는 전체 모델 계수라 Claim E의 첫 화살표(a: s\_attn ≠ s\_ssm)와 무관하다(OK 11).
12. 합성 입력 마이크로 측정과 V의 실제 입력 사이의 대표성(G7 허용 문안 6).
13. 64k 격자 선택과 T2-ext(O-3, int64 gate 조건부)의 서술 범위.
14. (rev2 신규) 사다리 arm별 ±2 rung의 오판 구간 산술(§6-2)이 맞는가, `V_PASS_UNREACHABLE`이 shape C를 사전에 불가로 만드는 것이 의도(정직)인가 과잉인가.

### 11-2. 하네스층
신규 구현(차용 목록은 §2-4·§2-5): ① 새 디렉터리의 spec 주도 셀 클라이언트(T1 burst + 창, T3 동거, T2 단일 요청·T2′, T4 채우기) ② 창 추출·S-per 계산(규칙 §1) ③ 클라이언트 토큰 간격 추정량(모델 공통) + **엔진 대 클라이언트 대조기(§1-7)** ④ 실행 가능성 평가(부팅 로그 → `INFEASIBLE_BY_RULE`) ⑤ 캡처 bs 목록 검사(`ABORT_CG_BS`, 목록에 64 포함) + CG canary ⑥ D=64 index를 포함한 config와 실현 SM 배선 검사(`STICKY_WIRED_D64`; **요청 SM ≠ 실현 SM** 검사 `SM_NOT_REALIZED`, §1-8) ⑦ 엔진 모드 두 개(`108u`, `108v`)와 Llama 부팅 ⑧ T4 부팅 로그 파서 + 채우기 판정(**산식 A·B 동시 예측 기록, 부팅 직후 봉인**) ⑨ round 역순·boot 무작위 순서 ⑩ 맞춤 코드(상대오차 가중 최소제곱, LOO, 외삽 시험) + 분석기(fetch 전 커밋) ⑪ `cm_predict` 예측기와 소거 변형(v1–v5), 귀무 예측기 N0·N1, SP R9-1 보간 재사용, 확률 시뮬레이션·`f_ext` 계산 ⑫ **arm별 V 사다리 생성기**(중심 = arm 예측) ⑬ `SERVER_DEATH` 분류 ⑭ 재개·DONE 마커·sha ⑮ T3 분포·정지·σ 기록기 ⑯ J-V 내부 앵커 러너와 보정 예측 계산.
변이 시험 후보: "cap 48 부팅인데 128로 기록" · "캡처 목록 밖 bs를 통과" · "ctx 띠 off-by-one" · "`INFEASIBLE_BY_RULE` 셀을 조용히 누락" · "T2에 첫 요청 워밍업 포함" · "Llama 셀을 Nano로 라벨" · "`108u`와 `108v` 합침" · "클라이언트 토큰 간격을 엔진 S-per로 표기" · "제외 round를 적합에 사용" · "대기 요청을 running으로 셈(T4)" · "예측값을 데이터 뒤에 봉인" · "사다리 중심이 예측이 아님(arm 기하평균 중심으로 회귀)" · "I-5 필드 누락 시 통과" · "fused 청크 비용을 T2(pos+c)가 아니라 T2(c)로 계산" · "TP2 앵커를 cap 128로 부팅" · "T4 예약 R을 0으로 둠(원식 회귀)" · "`TERM_DEVIATES` 있는데 `V_PASS` 부여" · "앵커 편차 > 3%인데 `V_JOB_SHIFTED` 미부여" · "불일치 bs를 제외하지 않음(§1-7)" · "정지(>5× 중앙값) 계산에서 중앙값을 동거 창으로 잡음".


## 12. 2회차 감사에 넘길 diff 범위 (rev1 → rev2)
- **전면 신규/개정(감사 대상)**: §R(대응표·등록 caveat) · §1-3(T3 기록 추가) · §1-4(T4 전면) · §1-6(T\_span 단위) · §1-7(엔진 대 클라이언트 대조) · §1-8(sticky·P 실현) · §6 전체(예측기 예약 회계·사다리·판정표·앵커 §6-7·귀무 §6-8) · §7 전체(Job 재구성) · §8-1 1번·8-2(R8-2′) · §9–§11.
- **국소 개정**: §0 마지막 줄 · §1-1(격자 bs 64 셀·bs\_padded·1차 추정량) · §1-2(T2-ext 조건부·허용 문안 3–4) · §1-5(전력 caveat 포인터) · §3-1(셀 수 정정 118→128) · §3-4(I-4·I-6 포함) · §4 ③·§4-2(대조 연동) · §5-3(두 곳 패치).
- **변경 없음(감사 비대상)**: §2(측정 방식·하네스 재사용 판정) · §4 표 ①②④⑤⑥ · §5-1·§5-2 표 · §3-2·§3-3.
