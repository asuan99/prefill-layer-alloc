# 사전등록 rev1 (DRAFT) — hybrid 서빙 비용 모델의 항별 측정(T1–T4)과 조립 검증(V), 2026-10-20

> 지위: **DRAFT rev1. 미봉인, 감사 0회, 측정 0, 성능 판정 0.** 문서 작업(CPU)만 했다. 제출·GPU·커밋 없음. 이 문서는 `PREREG_CAP_LONGCTX_rev0_2026-10-20.md`를 **대체**한다(rev0의 Q-A·Q-B·Q-C·Q-D는 이 문서에 없다. Q-B 이중차분은 처치 방향이 모델마다 반대라 폐기, 결정 14).
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·모형 예보, 측정 아님) · [코드] · [결정](사용자 결정 문서 명시) · [제안](이 초안이 고른 값, 감사 대상) · **[미등록]**(봉인 전에 사용자 또는 addendum이 채울 칸).
> 근거: 계획 `PLAN_costmodel_terms_2026-10-20.md` · 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 1–15 · 프로브 `probe_128k/RESULT_probe_128k_2026-10-20.md`·`AUDIT_probe_128k_2026-10-20.md`(허용 문안 1–7)·`ANALYSIS_triton_96k_failure_2026-10-20.md` · J1 감사 `../multigpu_verdicts/AUDIT_J1_N1_analysis_2026-10-19.md` · 감사 `../anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md` · 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`(이하 SP, v0.2+) · 항 정의 `COSTMODEL.md` CM-1…7.
> 기판: VESSL A100 SXM 80GB(108 SM) 1장(TP2 Job은 2장), SGLang v0.5.10, Nemotron-Nano-9B-v2-Base(rev `dc0661c…`), cudagraph ON, radix OFF, **`--mem-fraction-static` 0.6**(결정 13). 0.82로 잰 결과(M3 J1, 프로브 1회차)와 섞지 않는다.

## 0. 프레이밍과 이 문서가 하지 않는 것

- **주 목적(C)** = 비용 모델: lockstep PD-mux 비용 구조를 항 T1–T4로 재고(마이크로, 고정 배치·고정 길이), 조립해서 λ\*\_SLO를 예측하고(V) 서빙 실측과 대조한다. **(G)** PD-mux 이득은 그 모델이 설명해야 할 **검증 대상**이다. **(H)** "왜 hybrid인가"는 hybrid 고유 항을 Llama와 **대비**해 답한다. 모델 간 goodput 비교는 하지 않는다.
- 하지 않는 것: G0·M3 PREREG가 아니다. cap goodput 스윕(rev0 Q-A/Q-C)은 없다(cap = C\_mem 산식, 결정 15). 정책 순위·HE0·Claim E/G 등급을 바꾸지 않는다. layer-type별 분해는 하지 않는다(T1은 전체 모델 시간이라 원리적으로 불가, VERDICT\_claimE 근거 N2 PROFILE 항목).
- **항 T5**(CM-6 대기·TTFT)는 측정하지 않는다. T1–T4로 조립하고 V에서 대조한다. CM-3(전환·경계)·CM-7(radix)은 범위 밖이다.
- 인용 범위(모든 항 공통): 이 기판(VESSL A100-SXM4-80GB, Job별 GPU UUID 공시)·이 스택(SGLang v0.5.10, 봉인될 ENGINE\_DIGEST, Triton causal\_conv1d)·mem 0.6·클럭 비고정. **Job별 값**이다 — 서로 다른 Job의 수준값을 비교·결합하지 않는다(§9의 Job 효과 진단 예외 하나만).

## 1. 항별 estimand와 맞춤 형태

공통: 단위 = (boot, cell) 창 중앙값. 창 = N2 선례의 개념을 차용한다(burst 뒤 단독 decode, 처음 16 iteration 제외, batch 평균 ctx p50이 목표 c의 ±64 띠 안, 연속 iter\_seq 쌍, retract 0, cudagraph hit 정황). 상세 구현은 새 하네스의 몫이다(§3-2). 클럭·전력(I-1)은 창마다 공변량으로 기록하고 **제외 규칙으로 쓰지 않는다**(민감도 표만, §1-5).

### 1-1. T1 — CM-1 `T_dec(D, bs, ctx)`

| 항목 | 등록 |
|---|---|
| estimand | 결정 D(green-context로 **실현된** decode SM 수, sticky ON)·실행 batch bs·batch 평균 ctx c에서 decode step 시간(ms). 1차 = 엔진측 S-per(N2 선례, 연속 iteration 쌍 간격). 2차 = 클라이언트 steady-state 토큰 간격 중앙값(모델 간 대비의 공통 추정량, §4). I-4(decode 커널 GPU 시간)가 있으면 병기(§3-4) |
| 격자 | D ∈ {16, 24, 34, 44, 64} sticky ON + **D=108을 두 모드로**: `108u`(PD-mux 엔진 모드, 비분할 `(0,108)`, N2 관례), `108v`(vanilla 엔진, fused 계열 decode). bs ∈ {1, 8, 32, 48, 96}, c ∈ {2k, 8k, 32k, 64k} + **16k**(D ∈ {24, 34, 44, 108u, 108v}만; shape C 구간 11–20k 보간용). 실행 가능 판정은 §3-1의 규칙 |
| 고정 노브 | cap = 128(V 운용 cap 근방, 결정 15와 §6-2), `--max-mamba-cache-size` = cap, mem 0.6, context\_length 73,728, cudagraph ON, 캡처 bs ⊇ {1, 8, 32, 48, 96}(부팅 로그 검사, 아니면 `ABORT_CG_BS`), 같은 config 바이트(`pdmux_n3_A_agn.yml` 계열 + D=64 index 추가, digest [미등록]) |
| 제안 함수형 | `T_dec(D,bs,c) = a_D + b_D·bs + c_D·bs·c` — a = bs·ctx 무관 고정(가중치 sweep·launch), b = 요청당(SSM state·MLP), c = KV 읽기. 대안 형(적합도 실패 때만, 이 순서): F2 = F1 + d\_D·bs², F3 = a + b·bs^β + c·bs·c. 셋 다 실패하면 `FORM_INADEQUATE_T1_D{d}`, V 예측기는 **격자 안 이중선형 표 보간(log bs, log c), 외삽 금지**를 쓴다 |
| 맞춤 방법 | 상대오차 가중 최소제곱(Σ((T−T̂)/T)² 최소), D·엔진 모드마다 따로. 단위 = (boot, cell) 중앙값. D 간 부속: 계수별 `κ_j(D) = κ_j(108)·(108/D)^γ_j`(j = a, b, c)를 6 D에 맞춰 γ\_j를 **기술값**으로(N2 PROFILE의 계수비는 인용 금지·재측정, 층 종류 귀속 금지) |
| 적합도 보고 | (i) 표본 내 상대 잔차 중앙·최대 (ii) leave-one-cell-out 중앙·p90 (iii) **외삽 시험**: ctx 수준 1개(32k, 그다음 8k)와 bs 수준 1개(96, 그다음 32)를 통째로 빼고 맞춘 뒤 뺀 수준 예측 오차 (iv) 두 round를 각각 맞춘 계수의 차. **적합 채택 기준 [제안]**: (ii) 중앙 ≤ 3% ∧ p90 ≤ 8% ∧ (iii) 최대 ≤ 10%. 미충족은 위 라벨 |
| 인용 범위 | 해당 D·엔진 모드·**cap 128 부팅**·이 격자 박스 안. 박스 밖 외삽 문장 금지. 허용: "이 Job에서 D=…, bs=…, ctx=…의 decode step 중앙값은 …ms이고, 함수형 F1의 표본 내 잔차는 …, 외삽 시험 오차는 …였다" |
| 금지 | N2 PROFILE 계수(2.51/4.63/1.95×, −14~16%)·"shape B ctx 항 1%"·"shape C에서 hybrid 항이 유의하다"(VERDICT\_claimE 금지 문안); γ\_j를 s\_attn/s\_ssm 증거로 읽는 것; D 간 비를 "layer-type 차이"로 읽는 것 |

cap 불변성 점검(J-A 안, D=44 한 boot): cap 48로 부팅해 bs ∈ {8, 32} × c ∈ {2k, 8k} 4셀을 cap 128 boot의 같은 셀과 비교. 차 ≤ 3%면 `CAP_INVARIANT`, 아니면 `CAP_DEPENDENT`(T1 값에 "cap 128 조건부" 꼬리표, TP2 앵커는 cap 48을 쓰므로 영향 공시).

### 1-2. T2 — CM-2 `T_span(L, SM)`

| 항목 | 등록 |
|---|---|
| estimand | 다른 부하가 없을 때 단일 요청 prefill 시간. 1차 = 클라이언트 TTFT(`input_ids`로 정확히 L 토큰, 출력 1토큰, `ignore_eos`) 중앙값 — J1 floor(CM-2)와 같은 정의라 모든 모델·모드에서 비교 가능. 2차 = 엔진 span 시간(Nano PD-mux만; I-3 없음) |
| 격자 | L ∈ {1,024, 4,096, **8,800**(≈ s/k, 유도 경계 근방), 16,384, 32,768, 65,536} × 실현 prefill SM P = 108 − D ∈ {92, 84, 74, 64, 44}(D 격자 5개의 부산물, T1 boot 안에서 측정) + P = 108(`108u`·`108v`). 등록 대표 4점 P ∈ {44, 64, 74, 108}, P = 84·92는 무료 기술값. 한 (boot, L)당 8회, 앞 1회는 워밍업이라 제외 |
| 조건부 확장 T2-ext | int64 패치가 gate에서 통과하면(§8-1) P = 108에서 L ∈ {98,304, 126,976}을 두 모델에 더한다. 결정 13의 64k는 **격자 선택**이므로 확장은 사용자 결정(O-3), 미통과면 없음 |
| 제안 함수형 | `T_span(L,P) = u_P + v_P·L + w_P·L²`(P마다). P 의존 `(108/P)^θ`는 기술값. **모델-무관 형태 지표** `q = [T(65,536)/65,536] / [T(8,800)/8,800]` (토큰당 비용 증가비) |
| 맞춤 방법 | 상대오차 가중 최소제곱, 단위 = boot 중앙값 |
| 적합도 보고 | leave-one-L-out(내부 L 4개) 상대오차, 두 round 계수 차. 채택 기준 [제안]: 내부 L 제외 오차 ≤ 5%. 미충족 `FORM_INADEQUATE_T2_P{P}`, 표 보간(log L) 사용 |
| 라벨 | `q ≥ 1.10` → `PREFILL_SUPERLINEAR_SCOPED`, `q < 1.10` → `PREFILL_NEAR_LINEAR_SCOPED`, 구간이 1.10을 걸치면 `Q_UNRESOLVED` [제안]. J1 기울기(0.111 ms/토큰, L ≤ 16k)와 P=108 기울기의 차는 **기술값**(차 > 5%면 `T2_J1_SLOPE_DEVIATES`, 기판 공시 항목이지 결과 아님) |
| 인용 범위 | 이 기판·이 스택·chunk −1(PD-mux 강제 조건과 같음)에서 L ≤ 64k. "초선형/선형"은 q 라벨까지만 |
| 용도 | V의 SLO floor 예측과 fused 청크 비용(§6-1). fused\_tuned의 청크 prefill 비용은 `T2(pos+c) − T2(pos)`로 근사한다(위치 의존 가정, 미검증) |

### 1-3. T3 — CM-4 ρ (동거 간섭, **총효과**)

| 항목 | 등록 |
|---|---|
| estimand | 같은 boot·같은 셀에서 `ρ = S-per(동거 창) / S-per(단독 창)`. 동거 창 = 고정 decode batch(c = 8,192 [제안])가 돌 때 prefill 입력(L)이 지속 진행되는 iteration들. 처리 = 동거 + 그에 동반된 전력 한도 클럭 하락의 **합성값**(N2 Job-EA 선례: 분리 불가) |
| 격자 | D ∈ {24, 44} × bs ∈ {8, 32} × L ∈ {8,192, 32,768} = 8셀. 동거 창이 128 iteration 이상 되도록 prefill 입력 개수 m(L)을 등록 상수로 둔다 [미등록 — 하네스 몫, 값은 데이터 전 고정]. 반복 = round당 2회 × round 2 = 셀당 4 |
| 맞춤 | `ln ρ`의 셀별 평균(표). 예측기는 (log bs, log L) 이중선형 보간, D=34는 ln D 선형 보간(`RHO_D_INTERP`), 격자 밖은 클램프(`RHO_CLAMPED` 횟수 공시) |
| 적합도 | 4개 반복의 산포, 셀별 SW power cap 노출률·SM 클럭 중앙값(동거/단독 창 각각) |
| 인용 범위 | **총효과로만**: "400 W 한도 아래 클럭 하락을 포함한 동거 감속". N2 Job-EA 허용 문안 (P)(A)(B)의 틀을 따른다 |
| 금지 | "ρ는 동거 간섭의 크기다"(cap 병기 없이)·"전력 무관 간섭"·"ρ는 D·b·c에 무관"·ρ에서 정책/재배분 경계 도출·J1/N2 Job의 ρ와의 비교 |

### 1-4. T4 — CM-5 C\_mem과 거절 사유 몫

| 항목 | 등록 |
|---|---|
| estimand | (a) 부팅 로그 사실: KV 풀 토큰 수, mamba 슬롯 수·바이트, 가중치 바이트 → s = 슬롯당 바이트, k = KV B/토큰, 풀 합 (b) `C_mem(L) = floor(풀 / (s + k·L))` 산식의 예측 대 **채우기 실측**: 같은 L의 요청 N = min(cap, C\_mem) + 8개를 한꺼번에 보내 실현 최대 동시 running 수 (c) 거절 사유 몫(I-5: KV / cap≡슬롯), iteration 가중과 시간 가중 병기 |
| Nano 격자 | L ∈ {4,096, 8,800, 16,384, 65,536}. 부팅 5개: cap 48(네 L 모두) + cap = floor(C\_mem(L)) = {156, 114, 80, 27}(L별 1부팅, 그 L). 풀이 cap과 독립이라는 가정(KV 토큰 + mamba 바이트 = 상수)은 [유도]이며 **부팅 5개의 풀 합 일치(2% 안, 아니면 `C_MEM_DRIFT`)로 검증**한다. cap = C\_mem(L) 부팅인 이유: mamba 슬롯은 cap개가 정적 예약되어 KV 풀이 cap에 따라 줄므로(cap 160으로는 64k의 가용 토큰이 약 615k) 고정점에서만 산식이 시험된다 |
| 채우기 규칙 | 출력 길이는 마지막 요청의 첫 토큰 시각이 가장 이른 요청의 완료 시각보다 앞서도록 L별로 등록한다(L = 4k–16k는 4,096, 64k는 8,000). 위반하면 `FILL_INCOMPLETE`(그 셀은 산식 대조에서 제외, 건수 공시). arm = static D=44 PD-mux [제안, fused 대칭 점검은 O-8] |
| Llama(대비) | 부팅 1개(cap 128, mem 0.6): 풀 토큰 수(s = 0, k 131 KiB) + 채우기 L ∈ {4,096, 16,384} (둘 다 KV 한계 쪽) |
| 판정 형 | `CMEM_PRED_HELD`(실현/예측 비 ∈ [0.90, 1.10]) / `CMEM_PRED_MISS` [제안]. 사유 몫은 기술값 |
| 인용 범위 | 부팅 로그 사실은 [측정] 그대로. 산식은 "이 스택의 정적 풀 구현(cap≡슬롯)에서"라는 범위. 할당기 종속이라 hybrid 일반 문장 금지(결정 8′) |

### 1-5. 공변량·민감도(모든 항)

I-1 클럭·전력·throttle을 창마다 기록한다. SW power cap 노출률 > 20%인 창을 뺀 재적합을 민감도 표로 낸다(계수 변화 > 3%면 `CLOCK_SENSITIVE`, 제외는 하지 않음) [제안]. 클럭 고정은 불가(VESSL).

### 1-6. T5 — 조립(측정 없음)

반복 하나 = decode step 1개 + prefill span 1개(PLAN §1, CM-4). prefill 창 안의 ITL은 `max(T_dec·ρ, T_span)`. TTFT = W\_queue + ΣT\_span + 통신(TP). 동시 수 상한 = min(cap, C\_mem(L)), 수요 = λ·E[out]·E[ITL]. 이 구조의 실현(span 단위, 예산 `split_forward_token_budget` 65536, 청크 fused의 합산 가정)은 §6-1의 예측기 명세가 정한다.

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
셀 (bs, c)는 `bs ≤ cap` ∧ `bs ∈ 캡처 bs 목록` ∧ `bs·(c + O + X) ≤ 0.95 × KV 풀 토큰(부팅 로그)`일 때만 실행한다. 아닌 셀은 실패가 아니라 `INFEASIBLE_BY_RULE`로 표에 남긴다. cap 128 부팅의 KV 풀 [유도 895,781 토큰 = (풀 33.04 GB − 128×143.5 MB)/16,379 B; 부팅 로그가 대체]:

| bs \ c | 2k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| 1 | ○ | ○ | ○* | ○ | ○ |
| 8 | ○ | ○ | ○* | ○ | ○ (526k) |
| 32 | ○ | ○ | ○* | × (1.06M) | × |
| 48 | ○ | ○ | ○* | × | × |
| 96 | ○ | ○ (820k) | × | × | × |

(\* = D ∈ {24, 34, 44, 108u, 108v}만.) 실행 셀 수/D: 16·64 → 14, 나머지 → 18, 합 100개 고유 셀. 비어 있는 (32, 32k)·(48, 32k)·(·, 64k at bs ≥ 32)는 **cap 128·mem 0.6에서 KV가 모자라서가 아니라 격자 규칙상** 비운 것이며, 그 영역의 모델 값은 외삽이다(§1-1 인용 범위). V의 shape C는 bs ≈ 55–80, c ≈ 7–20k 영역 [유도]이므로 8k와 16k 점이 보간을 받치고 32k는 bs ≤ 8 점만 있다 — 이 한계는 §10 감사 쟁점 (4)에 올린다.

### 3-2. 새 하네스 항목(차용 목록은 §2-4, 신규 구현은 §11-2에 나열)

### 3-3. Job 간 정렬되는 상수
cap 128(T1·T2·T3), 48(cap 불변성 boot·T4 일부), mem 0.6, context 73,728, 같은 `max_prefill_tokens` 16,384 명시, 같은 attention backend(flashinfer, Llama도 명시), `ignore_eos`, radix OFF, logprob 없음, `PDMUX_HEARTBEAT_S=0`.

### 3-4. 계측 (I-1, I-4, I-5, I-6) — 구현 상태 [저장소 grep, engine-porter 확인 요청]

| 계측 | 쓰이는 항 | 저장소 상태 [코드] | 필요 조치 |
|---|---|---|---|
| I-1 클럭·전력·throttle | 전 항(공변량) | `clk.csv`·`substrate.json` 수집은 `job_entry.sh`·`substrate_probe_helper.py`에 있음. SP는 "F-2 0행 문제 수리·셀 시작 전 생존 검사"를 요구 | 생존 검사 구현 확인. 0행이면 `SUBSTRATE_UNCHECKED` |
| I-4 decode 커널 GPU 시간 | T1·T3(host 간격과 커널 시간 분리), V-diag | **구현 흔적을 찾지 못함**(`src`·`scripts`·`tests`에서 `I-4`·`decode_kernel`·CUDA event 사용은 `holb_probe.py` 하나) | 결정 15의 digest 묶음에 **포함할지 사용자 결정 O-2**. 미포함이면 T1은 S-per·토큰 간격만, I-4 열은 `NOT_COLLECTED` |
| I-5 거절 사유 | T4, V | `src/multiplex/admission_probe.py` CPU 검증 완료, **엔진 트리 미동기화**·GPU gate 전 | 결정 15에 이미 포함 |
| I-6 cudagraph 패딩(bs\_real, bs\_padded, hit/eager) | T1(캡처 목록 밖 bs 방지), T4(cap이 캡처 목록을 바꿈) | **구현 흔적을 찾지 못함**(`bs_padded` 없음) | O-2와 같이 결정. 미포함이면 부팅 로그의 캡처 목록 검사 + CG canary(`Decode batch … cuda graph: True`)로 대체하고 I-6 열은 `NOT_COLLECTED` |
| I-2·I-3 | — | 이 PREREG에 포함하지 않음 | T2는 클라이언트 TTFT가 1차라 I-3 불필요. T3는 R11-2 정렬 없이 총효과로만 쓴다(M3와 같은 범위 문장) |

## 4. (H) hybrid 고유 항 대비 — Llama-3.1-8B (base, rev `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`)

원칙: **Llama는 이 표의 칸을 채울 때만 쓴다.** goodput 비교·모델 간 이중차분 없음. 대비 칸은 **같은 Job**(J-B) 안의 같은 엔진 모드(vanilla fused\_sym, flashinfer, cap 128, mem 0.6)에서 잰다. Nano 쪽 짝은 `108v`이다.

| 대비 칸 | Nano에서 재는 것 | Llama에서 재는 것 | 맨 끝까지 말할 수 있는 것 | 말할 수 없는 것 |
|---|---|---|---|---|
| ① 요청당 메모리 `s + k·L` | 부팅 배너 s, k, 풀(T4a) | 같은 항: s = 0, k | 부팅 로그 사실 두 줄(s·k 값), 산술 경계 `s/k ≈ 8.8k(GB)–9.4k(GiB)`[유도] | "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED, 구간 확장 금지) |
| ② 채우기 동시 수 | 채우기 실현 동시 수(T4b) L ∈ {4k, 16k} 쪽 | 같은 L 두 점(KV 한계) | "이 스택·mem 0.6·이 풀에서 L=…일 때 측정된 최대 동시 요청 수는 Nano …, Llama …이다. 가중치 바이트가 달라 풀 크기가 다르다" — 사실 병기까지 | "hybrid가 긴 L에서 배치 여유가 더 크다"를 일반 문장으로(측정 두 L·두 모델·교락 목록 안에서만) |
| ③ decode ctx 기울기 | `T_dec` 계수 c(`108v`) | 같은 형 c, 짝 셀 9개: bs 1×{2k, 8k, 16k, 32k, 64k}, bs 8×{2k, 8k, 16k}, bs 48×{2k}. bs ≥ 8의 ≥ 32k 점은 Llama KV 풀 263,900 토큰으로 실행 불가(`INFEASIBLE_BY_RULE`) | 같은 (bs, c) 셀에서의 T\_dec 비와, 고정 bs(1, 8)에서 ctx 구간 기울기 비. 방향·형태 서술 | 층 종류 귀속("attention 4/56층이라 작다"는 [가설], 층 수·크기 교락), 크기 일반화 |
| ④ prefill L 의존 | T2 q (P=108, vanilla chunk −1) | 같은 q | `q` 라벨 두 개의 나란히 놓기: "Nano …, Llama … (L ≤ 64k)". T2-ext 통과 시 96k–124k 점 | "Llama가 64k 넘으면 초선형"(프로브 1–2회 기술값), "hybrid prefill은 선형" 일반화 |
| ⑤ cap 결합·거절 사유 | I-5 사유 몫(T4c): cap≡슬롯 | I-5 사유 몫: KV 위주(cap은 admission만) | 코드 사실(정적 풀, 설정으로 공시)과 사유 몫 기술값 | "슬롯 상한이 hybrid 고유"·"generic 스케줄러는 mamba state를 무시한다"·"Bullet/MuxWise가 C\_mem을 오예측한다" |
| ⑥ decode SM 민감도 | T1의 D 축 | **대비하지 않음**(Llama는 PD-mux 없이 `108v`) | Nano 안의 γ\_j 기술값만 | Llama와의 SM 민감도 비교 |

### 4-1. 교락 목록(공시 의무, 모든 대비 문장에 붙임)
모델 크기(9B vs 8B)·가중치 바이트(풀 크기가 달라짐, Nano 16.6 GiB vs Llama 약 15 GiB [유도])·층 수(56 vs 32)·**attention 층 수(4 vs 32 — 이것이 대비하려는 구성 자체이므로 교락이 아니라 처치와 뒤섞인 차이)**·MLP 형(Nano의 층 구성 vs gated SiLU)·hidden(4,480 vs 4,096)·RoPE(Nano attention 층은 NoPE로 알려짐[미확인]) vs RoPE·토크나이저(어휘·길이 분포; 이 PREREG는 `input_ids` 고정 길이라 길이 분포 영향은 없지만 어휘 크기에 따른 lm\_head 비용은 남음)·attention 백엔드(둘 다 flashinfer 명시)·학습/라이선스(base 동일). **Llama 한 모델 대 Nano 한 모델**이므로 "transformer 대 hybrid" 일반화 불가.

### 4-2. 허용 문장 틀
"이 Job(VESSL A100, mem 0.6, cap 128, vanilla chunk −1)에서 고정 (bs, ctx) 셀의 decode step 시간은 Nano-9B-v2 …ms, Llama-3.1-8B …ms였고, ctx 기울기는 …로 달랐다. 두 모델은 크기·층 수·attention 층 비율·MLP 형이 함께 다르다."
금지: "hybrid 고유"를 원인으로 쓰는 문장, "~52k가 hybrid 경계", "9k 경계를 실측했다", 모델 간 goodput 비교 문장.

## 5. TP2 항 (N 항)

TP2 PD-mux는 M2G J2에서 정확성 gate를 통과했다(G0(ii)·G1·G3 PASS, `G2_NOT_TESTED`) [측정, `RESULT_m2g_rev11_jobs_2026-10-13.md`]. 이 PREREG는 TP2에서 **비용 항 측정**과 **샤딩 사실 확인**을 한다. 성능 비교(G\_sym 등)는 하지 않는다.

### 5-1. 샤딩 사실 확인 항목
| # | 확인 | 예측 [유도, config 산술] | 측정 방법 | 판정 |
|---|---|---|---|---|
| F1 | 슬롯당 mamba state s/N | N=2에서 약 71.8 MB/rank(143.5/2); 128 mamba heads·8 groups가 2로 나누어떨어짐 | TP2 부팅 배너 rank별 슬롯 바이트 | rank 0·1 일치 ∧ 예측의 ±2% |
| F2 | KV B/토큰 k/N | KV head 8개(k = 4 layers×2×8×128×2 B = 16,384 B와 일치) → 4개/rank → 약 8,190 B | 부팅 배너 rank별 | ±2%. (N=4 → 2개, N=8 → 1개, N>8은 복제. **N=2만 측정**) |
| F3 | rank당 풀과 C\_mem(TP2, L) | `풀_rank / (s/2 + (k/2)·L)`; 가중치가 절반이라 풀이 약 39.7 GB/rank [유도] → C\_mem(32k)가 약 117(N=1은 약 49) | 부팅 + 채우기(T4, L ∈ {16k, 64k}, cap 128) | `CMEM_PRED_HELD` 형(§1-4) |
| F4 | in\_proj 폭과 conv 커널 stride | 폭 22,656 → 11,328/rank(`2×5120 + 2×512 + 64`). int32 오프셋 경계 `2^31 / stride`: N=1 94,786(관측 T ≥ 94,793과 일치) → N=2 **189,572** [산술] | (a) 첫 prefill에서 conv 입력 `x.stride()` 로그(하네스·엔진 로그, [미등록]) (b) gate의 커널 단위 시험에 stride 11,328 변형 추가 (c) **TP2 서빙 96k 1점(Job 마지막)** | stride 로그가 11,328이면 F4 입력 확인. 96k 점의 의미는 §5-3 |
| F5 | 헤드 분할 가능성 | attention heads 40 → 20, KV 8 → 4, mamba heads 128 → 64 | TP2 부팅 성공 + F1·F2 | 부팅 실패면 이후 항 전부 `TP2_NOT_BOOTABLE` |

### 5-2. 측정 셀 (J-C, `resourcespec-a100x2`, 두 GPU 동시 사용)
| 항 | TP2 셀 | 같은 Job의 N=1 앵커 | boot 수 |
|---|---|---|---|
| T1 | D ∈ {44, 108v} × bs ∈ {8, 32} × c ∈ {2k, 8k, 32k} = 12셀. (bs 48은 N=1 앵커가 (48, 32k)에서 KV 풀 1.59M 토큰 한계라 뺌) | 같은 12셀, cap 48 | TP2 boot 4(D×round 2), 앵커 4 |
| T2 | L ∈ {4k, 16k, 32k, 64k} × P ∈ {64, 108}(TP2/앵커 각각, T1 boot 안) | 같음 | 위 boot 재사용 |
| T4 | 부팅 로그(F1–F3) + 채우기 L ∈ {16k, 64k}, cap 128 | 해당 없음(N=1 사실은 J-A·J-B 몫이며 Job 간 비교 금지, 산식 대조로만) | TP2 boot 1 |
| 마지막 | 96k 1점 [제안: 통과하면 124k 1점 추가] | — | TP2 boot 1(Job의 마지막 boot) |

TP2 항에서 내는 양: (i) 같은 셀의 `T_dec(TP2)/T_dec(N=1)` 비와 계수 `a', b', c'` 대 `a, b, c` (ii) T2의 `T_span(TP2)/T_span(N=1)` (iii) F1–F5. **통신(all-reduce) 시간을 따로 분해하지 않는다**(profiler 불가, ncu는 green-ctx 하에서 exit 9). 총합 비로만 쓴다. rank lockstep 지연(rank 간 launch 시각 차)은 rank별 telemetry가 있을 때만 기술값 [미등록 — 엔진 확인].

### 5-3. 96k 점의 의미(충돌 지점, 사용자 결정 O-4)
결정 15가 int64 패치를 기본 digest에 넣으면 **패치가 켜진 TP2 96k 점은 경계 이동 시험이 아니다**(패치가 있으면 N=1에서도 통과해야 한다). 경계 이동(94,786 → 189,572) 예측의 시험 경로는 둘이다.
- (가) **권고**: gate의 커널 단위 시험에서 패치 전 커널 사본으로 stride 11,328·T ∈ {95,100, 126,976, 189,000, 190,500}을 돌린다(GPU 1장, 서버 없음, 서버 사망 위험 없음). TP2 서빙 96k 점은 패치 ON의 정상 점(T2 확장)으로 쓴다. 토글이 필요 없다.
- (나) 엔진에 패치 토글 env를 두어 TP2에서 패치 OFF 96k를 시험한다. 서버가 죽을 수 있다(`SERVER_DEATH`, 가장 마지막). 토글은 digest·정확성 gate를 늘린다.
어느 쪽이든 96k 점은 **Job의 마지막 boot**에 둔다(죽어도 앞 결과가 회수된다). 허용 문안 상한: "TP2에서 stride가 …로 로그되었고 96k prefill이 통과/실패했다. 경계 산술이 맞았는지는 (가) 시험 결과로만 말한다."

## 6. 검증 V — 조립 모델로 λ\*\_SLO 예측, 실측과 대조

### 6-1. 예측기(sealed 순수 CPU 프로그램, 이하 `cm_predict`)
- 입력: J-A·J-B에서 나온 맞춤 결과(T1 계수/표, T2, T3 표, T4 상수), V 셀의 요청 집합(같은 trace hash: 입력/출력 길이열), cap\_V, SLO 함수(A′: TTFT T = k·floor(L), ITL 요청 내부 p95 ≤ 50 ms; primary k = 5 [제안], 3·10은 기술값).
- 출력: arm·rate마다 SLO attainment 곡선 → SP R9-1(0.90 교차의 log-rate 선형보간)과 **같은 코드 경로**로 `λ*_pred`.
- 구조: PLAN §1(§1-6). agnostic static D = lockstep(`max(T_dec·ρ, T_span)`), fused\_tuned = **반복당 시간 = T\_dec(108v, bs, c̄) + [T2(pos+청크) − T2(pos)]**의 **합산 가정**(혼합 청크가 한 forward라 합산이 과대일 수 있음, 미검증 — 이 가정은 V가 시험한다). 통신 항은 N=1이라 없음.
- **봉인 시점**: 구조(명세 문서와 코드)는 **J-A·J-B 결과 fetch 전에** 봉인한다(데이터를 보고 구조를 고르지 못하게). 맞춤 계수에서 나온 예측값은 J-A·J-B 분석 후 **V 제출 전에** `V_REG` addendum으로 봉인한다(§8-2).
- 소거 변형(전부 V 데이터 전에 예측 봉인, 기술값): (v1) ctx 항 `c_D = 0` (v2) KV 전용 C\_mem(s = 0으로 두는 **모델 변형**이며 어떤 시스템의 동작 주장이 아님) (v3) ρ = 1 (v4) 외삽 허용. 어느 항이 V 오차에 닿는지 가르는 용도다. 변형 이름에 시스템 명을 쓰지 않는다.

### 6-2. 셀·arm·cap
- 셀 2개: (a) **arXiv-Sum 자연 출력**(유지분 5,906건 ≤ 16,384 토큰, 출력 평균 약 203, SP §10) (b) **shape C** = arXiv-Sum 입력 × 강제 출력 4,096(`ignore_eos`). context\_length 20,480 [제안, rev0 §5-1의 여유 3토큰 위험과 Llama 위반 문서 규칙은 Nano 단독이라 해당 없음, Nano 유지분 최대 16,379 + 4,096 = 20,475 ≤ 20,478].
- **cap\_V = floor(C\_mem(L̄\_자연)) 한 값**을 두 셀·모든 arm·G0 튜닝에 공통 적용 [제안]: 산식 [유도] 약 124 → 부팅 로그와 T4 대조 후 등록(`V_REG`) [미등록 — 값]. shape C에서는 KV 풀(cap 124에서 약 929k 토큰 / L̄ 11.4k ≈ 81)이 먼저 구속하므로 `min(cap, C_mem)`이 실제로 시험된다.
- arm(모두 같은 Job 안, 셀마다 boot): `fused_tuned` + **agnostic static D 후보 상위 2개**(셀마다 예측 λ\*\_pred가 큰 순서의 D ∈ {24, 34, 44}, 동률이면 큰 D) = 셀당 3 arm [제안, 선택지는 O-5]. "agnostic(최선 static D)"는 이 상위 2개 안의 실측 최선이다 — 예측이 놓친 세 번째 D가 더 좋았을 가능성은 범위 문장으로 공시한다.
- `fused_tuned` = G0 튜닝 argmax config(SP R8-2d `fused_best11(k=5)`, 결정 11·15), **자연 셀에서 튜닝**한 값을 shape C에도 쓴다(범위 문장: "자연 셀 argmax config"; "shape C에서 튜닝된"이라 쓰지 않음, 게이트 8). mixed-chunk가 argmax이면 SP R8-2c 운용 규약과 G-M1–G-M4를 gate에 포함한다.
- 사다리: 셀마다 **공통 격자, 인접 비 1.2, 5 rung, 중심 = 그 셀 arm들의 `λ*_pred` 기하평균** [제안]. arm 간 예측 폭이 1.44를 넘으면 SP R9-3대로 rung을 `ceil(log(폭/1.44)/log 1.2)`개 더한다. **Job-0·확장 없음**: 사다리를 예측으로 정하므로 보정 Job이 필요 없고, 교차가 격자 밖이면 §6-3의 라벨이 된다(데이터를 보고 rung을 고르는 자유 제거).
- seed 4(판정 하한), boot 2 pass(seed {1,2}/{3,4}, pass 2는 arm 역순), n/rung/seed = 자연 60, shape C 45 [제안, rev0 승계]. R9-6 해상도(`2√(0.09/n)/|Δa|·log ρ`)를 셀마다 공시(shape C는 `RESOLUTION_COARSE`일 수 있음, 크기 문장 용도가 아닌 V 허용 오차 10%와는 별개). warm-up은 전송하고 채점 제외(자연 20 s, shape C 120 s).
- 관측자: SP §9에 따라 ON/OFF 짝 블록(n ≥ 8, PD-mux 계열·fused 계열 각각)을 같은 Job에 둔다(자연 셀, 한 rung).

### 6-3. 판정 규칙 (데이터 전 등록, [제안] — 허용 오차 값은 감사 대상)
오차 `e = ln(λ*_meas / λ*_pred)`, 지표 `λ*_meas`는 같은 Job의 R9-1 보간, k=5. 단위 = (셀, arm).

| 라벨 | 조건 |
|---|---|
| `V_PASS` | 모든 (셀, arm)에서 λ\* 해결(`LAMBDA_RESOLVED`, `CLIFF_ADJACENT` 아님) ∧ `|e| ≤ ln 1.10` ∧ 셀 안 arm 순서(λ\*\_agnostic 대 λ\*\_fused\_tuned의 부호)가 예측과 일치(3% 미만 차는 순서 판정 보류, 게이트 3) ∧ `|ln G_meas − ln G_pred| ≤ ln 1.10` |
| `V_COARSE` | `V_PASS` 불성립 ∧ 모든 (셀, arm)에서 `|e| ≤ ln 1.25` ∧ 해결 |
| `V_FAIL` | 어느 (셀, arm)이든 `|e| > ln 1.25` 또는 순서 불일치(3% 이상) |
| `V_FAIL_OUT_OF_LADDER` | 어느 (셀, arm)에서 교차가 격자 밖(`BELOW/ABOVE_LADDER`). 격자가 예측 중심 ±2 rung(×0.69–×1.44)이므로 `|e|`가 그 범위를 넘었다는 **하한 정보**만 있다 |
| `V_UNRESOLVED` | 서버 사망(`SERVER_DEATH`)·`UNRESOLVED_J`·`CLIFF_ADJACENT`로 해결 못 한 칸이 있고 나머지가 `V_COARSE` 이내 |

- 10%의 근거(제안): 관측된 PD-mux 이득 크기가 +27–40%대이고(CONSENSUS §1-1) 예측 오차가 그 1/3을 넘으면 모델이 G를 "설명한다"고 말할 수 없다. 25%는 사다리 범위(×0.69–×1.44)의 안쪽 경계다. **정당화는 감사 쟁점 (2)**.
- Job 1개이므로 판정 범위는 "이 Job에서"(SP R5-4). Job 간 분산은 모른다. J-A·J-B·J-C·J-V는 서로 다른 노드·GPU일 수 있으므로 `e`에는 **Job 효과가 섞여 있다**. 그 크기의 하한 참고치로 J-A와 J-B가 모두 갖는 `108v` T1·T2 계수의 Job 간 차를 기술값으로 낸다(§9 예외).

### 6-4. 실패해도 결과로 쓰는 문장 (사전 등록)
- `V_COARSE`/`V_FAIL`/`V_FAIL_OUT_OF_LADDER`: "봉인된 예측기(T1–T4 + 조립 구조)는 이 Job의 (셀, arm)에서 λ\*\_SLO를 …% 빗나갔다(부호 …). 항 측정값은 §1의 범위에서 유효하며, **조립이 검증되지 않았다.** 이 PD-mux 이득(G)은 이 모델이 설명하는 것으로 주장하지 않는다." 허용되는 진단: V-diag(아래) 표.
- `V_PASS`: "이 Job의 두 셀·세 arm에서 λ\*\_SLO가 봉인된 예측의 ±10% 안이었다. 이 기판·이 스택·이 두 workload에서의 예측 일치이며, 다른 모델·trace·기판으로의 일반화나 hybrid 고유성 귀속이 아니다."
- 두 경우 모두 예측기 수정은 **새 PREREG·새 Job**이다(재스코어 금지, 게이트 8).
- **V-diag**(사후, 기술값, 재스코어 아님): V 텔레메트리로 모델 내부 양을 실현값과 대조한다. (1) 실현 decode bs 분포 대 예측 (2) 실현 step 시간 대 `T1(bs, c̄)` 비 (3) span 시간 대 T2 (4) 동거 step 비 대 ρ (5) 실현 동시 수 대 `min(cap, C_mem)` (6) 같은 Job 실측 floor 대 T2 floor. `|ln 비| > ln 1.10`인 항에 `TERM_DEVIATES_{T1..T4}`를 붙인다 — 원인 후보 정렬용이며 인과 판정이 아니다.

### 6-5. V를 읽는 금지 문안
"G가 비용 모델로 설명됐다/예측됐다"(V\_PASS도 두 셀·한 모델의 예측 일치까지만) · "기전 귀속" · J1 B·X arm 비의 크기 문장 · "PD-mux 우위"(G 크기는 이 PREREG의 estimand가 아님) · "G4(장문 prefill 중 ITL 기술값)는 ITL SLO 충족".

## 7. Job 구성, 시간, GPU-h, 16 h 캡 (사용자 메모 "GPU Job 적정 크기": 동시 사용 GPU 수에 맞춘 스펙과 사용률)

### 7-1. 시간 모형 [유도 — 측정 아님, Job 크기 결정용] (`scratchpad/bud.py`, 비영속; 재현은 이 문단의 파라미터로)
boot 200 s, boot별 warm-up 60 s, 셀 오버헤드 15 s, decode 창 300 token. 단일 요청 prefill TTFT [기술값, 프로브·J1]: 1k 0.12 s, 2k 0.24, 4k 0.46, 8k 0.92, 16k 1.84, 32k 4.0, 64k 8.4. prefill SM 감소 배율 `(108/P)^0.8` [임의 중심값, T2 전에 모름]. step 시간 `(12 + 0.45·bs) ms × (108/D)^0.5` [임의]. TP2 prefill 속도 1.7× [임의], TP2 boot 400 s. V 셀 평균: 자연 125 s, shape C 370 s [rev0 모형 승계, 임의]. **모든 시간은 Job 크기 결정용이며 측정 전 값이다.**

### 7-2. Job 표
| Job | 내용 | 스펙 | boot | 셀/반복 | 예상 h [유도] | 하드캡(×1.2) | 예약 GPU-h | 사용 GPU-h(사용률) |
|---|---|---|---|---|---|---|---|---|
| **J-A** Nano D 사다리 | T1(D 5 + 108u + 108v, 각 2 round) + T2(같은 boot) + T3(D 24·44) + cap 불변성 boot(D=44, cap 48) | `resourcespec-a100x1` | 15 | T1 240 셀실행(100 고유 셀 중 round 반복) · T2 84 셀(672 요청) · T3 32 실행 | 6.4 | 7.7 | 6.4 | 약 5.8 (90%)† |
| **J-B** Nano-Llama 대비 + T4 | Nano `108v` 4 round + Llama 4 round(T1+T2) + T4 Nano 5 boot(8 fill) + T4 Llama 1 boot(2 fill) | `resourcespec-a100x1` | 14 | T1 Nano 72·Llama 36 셀실행, T2 48 셀 | 3.6 | 4.4 | 3.6 | 약 3.3 (90%)† |
| **J-C** TP2 | 샤딩 확인 F1–F5, T1·T2 TP2/앵커, T4 TP2, 마지막 96k | `resourcespec-a100x2` | 10 (TP2 6 + 앵커 4) | T1 96 셀실행 · T2 32 셀 | 2.5 | 3.0 | 5.0 | 4.0 (81%)‡ |
| **J-V** 검증 | 자연 + shape C × 3 arm × 5 rung × 4 seed + 관측자 ON/OFF 블록 | `resourcespec-a100x1` | 12+ | 120 셀(자연 60 + shape C 60 \[각 arm당 20\]) 3,600 + 2,700 요청 | 9.5 | 11.4 | 9.5 | 약 9.0 (95%)† |
| **합계(기본)** | | | | | | | **약 24.5 GPU-h** | 약 22 |

† 단일 GPU Job: 측정 중 GPU 점유는 거의 100%, 부팅(가중치 적재)·클라이언트 대기·drain 때문에 평균 약 90–95% [유도].
‡ TP2 Job: TP2 boot는 2장 모두, N=1 앵커 boot는 1장만 쓴다(`(2×TP2 시간 + 1×앵커 시간)/(2×Job 시간)`). 개선안(O-6): 앵커를 두 GPU에서 독립 서버 두 개로 동시에 돌려 사용률을 약 100%로 올릴 수 있으나 전력·열을 공유해 두 앵커가 서로 교락한다(J1 GPU0 throttle 61% 선례).

### 7-3. 합계 재계산(계획의 약 25–33 GPU-h 대비)
- 계획 25 = 단일 항 7 + TP2 6 + V 10–14(확장 포함). 이번 계산: 단일 항 **10.0**(J-A 6.4 + J-B 3.6) + TP2 **5.0**(2장 × 2.5 h) + V **9.5**(3 arm, 확장 없음) = **약 24.5 GPU-h**.
- 선택지(O-5) 영향: V 2 arm 6.5 h → 총 21.5, V 4 arm 12.5 h → 총 27.5. 하드캡 합 (×1.2) 약 29.5(기본).
- 계획보다 단일 항이 큰 이유: ① D 6개 × 엔진 모드 2개(108u·108v)의 boot 비용(boot당 약 18–25분) ② 16k 점 추가 ③ Llama 4 round(H 대비의 n=4) ④ T4 Nano 5 boot. 계획보다 V가 작은 이유: Job-0 없음(사다리를 예측으로), 확장 규칙 없음(교차 밖 = `V_FAIL_OUT_OF_LADDER`).
- **이 합계 밖 의존 비용**: ① GPU correctness gate(결정 15, engine-porter 몫, 범위 [미등록]) ② G0 튜닝 Job(fused\_tuned 확정, G0 PREREG 몫, SP R8-2: 11 config) ③ Workspace 하네스 smoke(`pdmux-probe`, 비측정, 상한 약 1.5 h [유도]) ④ TP2 검증 셀(V의 TP2 판)은 이 PREREG에 없음.

### 7-4. 16 h 캡과 Job 분할 사유
- 모든 Job 기대 시간 ≤ 9.5 h, 하드캡 ≤ 11.4 h < 16 h. J2·J4·J8이 공통으로 약 9.5 h에서 외부 정지된 선례(원인 미확정, `M3_SCOPE_OPTIONS`)가 있으므로 **J-V는 하한 설계 필요**: 기대 9.5 h는 정지 지점과 같다. 대응: 재개 가능(셀 단위 DONE·`vessl_jobs.tsv`·sha), 정지 시 이미 끝난 (셀, arm)은 회수하고 나머지는 새 job name으로 재실행(사실 공시). 사용자 결정 O-7(J-V를 셀 2개 Job으로 쪼개 각 5 h 안쪽으로: 자연 셀 단독 Job과 shape C 단독 Job; 두 셀은 서로 비교하지 않으므로 same-Job 원칙을 어기지 않음. 각 셀의 arm 3개는 한 Job에 남는다).
- J-A와 J-B로 나눈 이유: 하나로 합치면 약 10 h로 정지 선례에 닿는다. 단 H 대비(Nano `108v` 대 Llama)는 J-B 안에서만, D 사다리(108v 포함)는 J-A 안에서만 성립한다. `108v`가 두 Job에 모두 있어 Job 효과 진단값을 겸한다.

## 8. 선행 조건 순서와 R8-2 개정

### 8-1. 순서(각 단계는 앞 단계가 끝나야 시작; 사용자 지정 순서 승계)
1. **gate(결정 15)**: I-5 엔진 동기화 + Triton causal\_conv1d int64 패치를 **한 번의 ENGINE\_DIGEST 변경**으로 묶고 GPU correctness gate 1회. 포함 시험: ON/OFF 출력 일치, 관측자 TOST(±1.5%, n ≥ 8 짝, PD-mux·fused 각각), 변이 시험, 패치 반증 실험(감사 2번 ①–③: 패치 후 fused\_sym 96k·124k 통과 + 되돌리면 95,100 재실패 + 단위 시험(12288, 95100)·stride 22,656 + 94,784/94,800 경계 2회씩) + (이 PREREG 요구) stride 11,328 변형(§5-3 (가)). **O-2**(I-4·I-6을 같은 digest에 포함할지)가 결정되면 같은 gate에 넣는다. 사후에 digest를 바꾸면 이후 단계가 다시 시작이다.
2. **하네스**: 새 하네스(§11-2) CPU 시험·변이·가짜 엔진 e2e → Workspace smoke(비측정, 수치 인용 금지).
3. **G0 튜닝 Job**(별도 PREREG): fused\_tuned config를 cap = cap\_V에서 정한다(R8-2′).
4. **감사**: 규칙층(이 문서 §10 쟁점) → 하네스층. 각 최대 2회.
5. **봉인 S1**: 규칙·하네스·엔진 digest·config digest. (V의 예측기 구조 명세·코드도 여기서 봉인.)
6. **CPU preflight** → **`launch.sh` dry-run**(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit`. J-A·J-B·J-C는 독립 제출 가능(fused\_tuned 불필요). J-V는 아래 S2 뒤.
7. **fetch**(`fetch.sh`, DONE + sha256) → result-analyst가 T1–T4 맞춤(분석 코드는 fetch 전에 커밋, 맞춤 규칙은 이 문서).
8. **봉인 S2 `V_REG`**: cap\_V, fused\_tuned config, agnostic D 후보 상위 2개, 사다리 rung 절대값, 예측값 `λ*_pred`(소거 변형 포함) sha. **J-V 데이터 전에** 커밋.
9. J-V dry-run → 승인 → submit → fetch → 판정(§6-3).
B5 선례(운영 모델): Job 1개가 `fetch.sh` 통과까지 간 선례가 있다. dry-run 때 확인한다.

### 8-2. R8-2 개정 문안(규칙층 감사 대상, 사용자 확인 필요 O-1)
SP R8-2 현행: "…(cap = G0 cap 고정, R16-1)… 바꾸지 않는 것: `max-running-requests` = G0 cap(R16-1), radix off, overlap on, mem-fraction 0.82…".
**R8-2′**: (1) fused\_tuned 튜닝과 판정은 **같은 cap = cap\_V**에서 한다. cap\_V는 G0 cap 결정 규칙(R16-3, [미등록])이 아니라 **C\_mem 산식**(Nano, mem 0.6, 자연 셀 L̄, T4가 검증한 산식)으로 정한다(결정 15). (2) mem-fraction은 **0.6**으로 통일한다(결정 13; 0.82는 모든 arm·블록에서 사용하지 않음). (3) 튜닝 workload는 자연 arXiv-Sum 셀이고, shape C의 fused\_tuned는 "자연 셀 argmax config"로 표기한다. (4) 튜닝은 V보다 먼저 별도 Job에서 하고 V에서 같은 config를 다시 잰다(승자의 저주 회피 유지). (5) SLO(A′, ITL 50 ms)·후보 11 config·엄격 argmax·동률 순서·mixed-chunk 운용 규약은 불변. (6) S-2(튜닝 공간 비대칭 11 대 PD-mux 상위 2 D)는 **더 커진다 — 같은 금지 문장 적용**: "PD-mux가 fused보다 못하다"를 PD-mux 미튜닝의 귀속으로 쓰지 않는다.
**R16-3′ 병기**: SP R16-3의 "후보 {48, C\_mem/2, C\_mem}에서 G0 cap 결정 규칙 [미등록]" 은 이 PREREG가 cap 선택을 goodput 스윕이 아니라 C\_mem 산식으로 대체하므로 문언 개정이 필요하다. R16-3 "cap 특성화 Job"의 CM-5 검증 기능은 T4가 맡는다. 대체 여부는 사용자 결정(O-1).

## 9. 라벨과 금지 문안

### 9-1. 라벨
항: `FORM_INADEQUATE_T1_D{d}`·`FORM_INADEQUATE_T2_P{P}`·`CAP_INVARIANT`/`CAP_DEPENDENT`·`CLOCK_SENSITIVE`·`PREFILL_SUPERLINEAR_SCOPED`/`PREFILL_NEAR_LINEAR_SCOPED`/`Q_UNRESOLVED`·`T2_J1_SLOPE_DEVIATES`·`RHO_D_INTERP`·`RHO_CLAMPED`·`CMEM_PRED_HELD`/`CMEM_PRED_MISS`·`C_MEM_DRIFT`·`FILL_INCOMPLETE`·`INFEASIBLE_BY_RULE`·`TP2_NOT_BOOTABLE`. 검증: `V_PASS`/`V_COARSE`/`V_FAIL`/`V_FAIL_OUT_OF_LADDER`/`V_UNRESOLVED`·`TERM_DEVIATES_{T1..T4}`. 승계: SP 계열 `LAMBDA_RESOLVED`·`BELOW/ABOVE_LADDER`·`CLIFF_ADJACENT`·`SUBSTRATE_WARNING`(드라이버 변경)·`SUBSTRATE_UNCHECKED`·`SERVER_DEATH`(rev0 §6-1 정의: 죽은 (arm, cap, L)은 `UNRESOLVED(arm_death)`로 두고 다른 arm 결과를 지우지 않으며 일반화하지 않는다)·`CLIENT_ERROR`·`SERVER_ERROR_HI`·`NOT_COLLECTED`(I-4·I-6).

### 9-2. 금지 문안 (승계 + 이 PREREG)
- "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED, 구간 확장 금지; 긴 L의 방향은 T4 측정 후 ②의 틀로만)
- "~52k 경계"/"~52k에서 구속 전환", "9k 경계를 실측했다"(유도값), "~52k가 hybrid 경계"
- "PD-mux는 64k를 넘을 수 없다"·"hybrid PD-mux는 64k 불가"·"OOM 때문에 64k"(0.6의 95k+ 실패는 OOM이 아님; 64k는 격자 선택, 한계는 chunk −1 Nano arm 공통, 프로브 감사 허용 문안 3–4), "96k 실패는 SSD 커널 오류", "PD-mux 고유 한계", "0.82에서 PD-mux가 먼저 OOM 난 것은 split-prefill 때문", 0.82 결과와 0.6 결과의 혼합
- "Bullet/MuxWise 오예측"·"Bullet/MuxWise가 C\_mem을 오예측한다"·"generic 스케줄러는 mamba state를 무시한다"·"슬롯 상한이 hybrid 고유"·"Claim E 반증"
- J1 크기 문장: J1 B·X arm 비의 크기, CM-6 "처리율↔ITL 역순"을 쓰인 그대로, "PD-mux가 클럭을 낮춘다" 일반 가정, fused 정지 꼬리를 PD-mux 구조 이득으로 귀속
- G4를 "ITL SLO 충족"으로 읽는 것(요청 내부 p95가 정지 1회를 가림), "G7이 합성 생성기 동등성을 보였다"
- N2 PROFILE 계수·"shape B ctx 항 1%"·"shape C에서 hybrid 항이 유의하다"·s\_attn/s\_ssm 증거로서 γ\_j·T1 D 비
- T3: ρ를 간섭 크기로, ρ에서 정책·재배분 경계·굶김 연쇄·HE0 원인 도출
- V: §6-5
- 선행 gate 전: "I-5가 거절 사유를 측정했다"(동기화·GPU gate 전), "preflight PASS가 이 하네스를 검증했다"

## 10. 열린 결정 (사용자 몫)
- **O-1** R8-2′와 R16-3′ 개정 승인(§8-2). cap 특성화 Job을 이 PREREG의 T4가 대체하는 것을 확인.
- **O-2** I-4·I-6을 결정 15의 ENGINE\_DIGEST 묶음에 넣을지(미포함이면 두 열 `NOT_COLLECTED`).
- **O-3** T2-ext(96k·124k, P=108, 두 모델)를 결정 13의 64k 격자 선택에 더할지(약 0.2 GPU-h, 패치 gate 통과 조건부).
- **O-4** TP2 96k 점의 의미: (가) 단위 시험으로 경계 이동 + 96k는 정상 점(권고) / (나) 패치 토글 env 도입.
- **O-5** V의 agnostic arm: (a) 예측 최선 D 1개(2 arm, 6.5 h) / (b) 상위 2개(3 arm, 9.5 h, 권고) / (c) D 3개(4 arm, 12.5 h, 정지 선례 9.5 h 위험).
- **O-6** TP2 앵커: 순차 1장(사용률 81%) / 두 GPU 동시 독립 앵커(≈100%, 교락 공시).
- **O-7** J-V를 셀 2개 Job으로 분할할지(정지 선례 대응; GPU-h는 부팅 반복 분만큼 +약 0.3).
- **O-8** T4 arm 대칭 점검(fused\_sym 한 boot 추가, +약 0.3 h).
- **O-9** 허용 오차 10%/25%와 T1 적합 채택 기준(3%/8%/10%)이 모델 크기에 비해 타당한지(감사 쟁점 2와 같이).
- **O-10** primary k = 5, 소거 변형 4개 범위, 라운드 수(기본 2, `108v`·Llama 4).

## 11. [미등록] 목록과 감사 쟁점

### 11-1. 규칙층
[미등록]: cap\_V 값 / `V_REG`(fused\_tuned config, D 후보 상위 2개, rung 절대값, 예측값) / 예측기 명세(span 단위, 청크 합산 가정, 통신 없음) / X\_T1(c\_in = c − X)와 O(출력 길이) / m(L)(T3 prefill 입력 수) / 채우기 출력 길이 L별 확정값 / T1 외삽 시험 수준 순서 / 소거 변형 정의 세부 / 허용 오차(O-9) / 적합 채택 기준 / config digest(D=64 index 추가본) / Llama 대비 짝 셀의 최종 목록(부팅 로그 평가) / 보고서 파일 이름 / 하드캡 값 / fused 대칭 점검 / TP2 rank 지연 지표.
감사 쟁점:
1. **예측-실측 대조가 "비용 모델 검증"인가**: 조립 구조(DES, fused 합산 가정)가 항 측정과 따로 미검증이다. 통과해도 오차 상쇄일 수 있고(소거 변형·V-diag가 보조), 실패해도 항 탓인지 조립 탓인지 가를 수 없다. `V_PASS`/`V_FAIL` 문장이 이 한계를 충분히 담았는가.
2. 허용 오차 10%/25%의 정당화(근거는 G 크기 +27–40%와 사다리 범위). 사후에 바꿀 수 없게 봉인되는가. Job 효과(다른 Job·노드의 항 값)가 `e`에 섞이는 문제.
3. "agnostic(최선 static D)"가 상위 2개 D 중 최선일 뿐이다. 범위 문장이 충분한가. 대안(D 3개)의 정지 위험.
4. **T1 격자 외삽**: shape C 영역(bs 55–80, c 7–20k)은 8k·16k 점이 받치지만 (32, 32k) 등 KV로 비운 칸이 많고 bs=96 위쪽이 없다. 함수형 채택이 이 영역을 보장하는가.
5. n: T1·T2 D≠108은 round 2 (CI 의미 제한), H 대비는 round 4, V는 seed 4. 게이트 3의 n ≥ 4는 판정에 적용되며 항 맞춤은 기술 적합이라는 구분이 타당한가. 두 round의 분리 예측 대역(round-split prediction)이 불확실성으로 충분한가.
6. fused\_tuned를 자연 셀에서만 튜닝하고 shape C에서 쓰는 비대칭(게이트 8과 S-2의 방향성: PD-mux 우위 문장에는 보수적, 열위 문장에는 비보수적).
7. Llama 대비의 교락 목록과 허용 문장: 크기·층 수·attention 층 수가 처치와 섞여 "hybrid 고유" 문장이 어디까지 가능한가. Llama 실행 불가 셀(KV 263,900 토큰)이 대비 폭을 좁히는 문제.
8. T3: 총효과 정의, 전력 한도 교락(CM-4), 동거 창 128 iteration 조건이 L=8k에서 만족되는지.
9. T4 산식 시험의 고정점 설계(cap = C\_mem(L) 부팅 4개), `FILL_INCOMPLETE` 규칙, 정적 풀·할당기 종속 서술이 결정 8′의 범위 안인가.
10. R8-2′가 SP R8-2·R16-1·R16-3과 충돌하지 않는가(특히 11 대 2의 비대칭 확대, mem 0.6으로 바뀐 뒤 SP R8-2c의 "작은 예산 공시" 수치 c=48 예가 cap\_V로 다시 계산돼야 함).
11. 함의 한계: T1의 c\_D는 전체 모델 계수라 Claim E의 첫 화살표(a: s\_attn ≠ s\_ssm)와 무관하다. 이 PREREG의 어떤 라벨도 그쪽으로 읽히지 않는지.
12. 합성 입력(내용 무관) 마이크로 측정과 V의 실제 arXiv-Sum 입력 사이의 대표성(G7 허용 문안 6의 범위: 단일 요청 prefill 시간까지).
13. 64k 격자 선택과 T2-ext(O-3)의 서술 범위.

### 11-2. 하네스층
신규 구현(차용 목록은 §2-4·§2-5): ① 새 디렉터리의 spec 주도 셀 클라이언트(T1 burst + 창, T3 동거, T2 단일 요청, T4 채우기) ② 창 추출·S-per 계산(규칙 §1) ③ 클라이언트 토큰 간격 추정량(모델 공통) ④ 실행 가능성 평가(부팅 로그 → `INFEASIBLE_BY_RULE`) ⑤ 캡처 bs 목록 검사(`ABORT_CG_BS`) + CG canary ⑥ D=64 index를 포함한 config와 실현 SM 배선 검사(`STICKY_WIRED_D64`; 요청 SM ≠ 실현 SM 주의) ⑦ 엔진 모드 두 개(`108u`, `108v`)와 Llama 부팅 ⑧ T4 부팅 로그 파서 + 채우기 판정 ⑨ round 역순·boot 무작위 순서 ⑩ 맞춤 코드(상대오차 가중 최소제곱, LOO, 외삽 시험) + 분석기(fetch 전 커밋) ⑪ `cm_predict` 예측기와 소거 변형, SP R9-1 보간 재사용 ⑫ V 사다리 생성기(예측 중심, 폭 규칙) ⑬ `SERVER_DEATH` 분류 ⑭ 재개·DONE 마커·sha. 단일 요청 prefill 시 sticky ON에서 prefill이 실제로 `P = 108 − D`에서 도는지(엔진 확인, 안 돌면 T2의 SM 축 정의 불가 → `T2_P_NOT_REALIZED`).
변이 시험 후보: "cap 48 부팅인데 128로 기록" · "캡처 목록 밖 bs를 통과" · "ctx 띠 off-by-one" · "`INFEASIBLE_BY_RULE` 셀을 조용히 누락" · "T2에 첫 요청 워밍업 포함" · "Llama 셀을 Nano로 라벨" · "`108u`와 `108v` 합침" · "클라이언트 토큰 간격을 엔진 S-per로 표기" · "제외 round를 적합에 사용" · "대기 요청을 running으로 셈(T4)" · "예측값을 데이터 뒤에 봉인" · "사다리 중심이 예측이 아님" · "I-5 필드 누락 시 통과" · "fused 청크 비용을 T2(pos+c)가 아니라 T2(c)로 계산" · "TP2 앵커를 cap 128로 부팅".
