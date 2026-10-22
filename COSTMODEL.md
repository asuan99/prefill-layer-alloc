# COSTMODEL — 지금까지 드러난 비용 구조 (진척판, 2026-10-13 신설 · 2026-10-21 갱신)

> **지위**: 측정 결과를 새로 만들지 않는다. 이미 정본(`reports/CONSENSUS.md` §1, `reports/cost_model_design_space_2026-10-01.md`,
> 감사 판정서)에 있는 사실을 **비용 항(CM-1…CM-7) 축으로 다시 배치**한 장부다. 연구 축은 `reports/system_design_anchor_2026-09-30.md`
> (스케줄러 확장)이며, 이 장부는 그 과정에서 **cost model이 얼마나 성립했는지**를 보여 주는 진척판이다.
> 규칙: 현재 형태만 적는다(이력은 git). 한 항 최대 6줄. 새 근거는 결과 분석의 "cost model 기여" 절에서만 들어온다.
> 상태 어휘: `가설` · `데이터 있음(미감사)` · `형태 확인` · `계수 확정(범위 한정)` · `서빙 검증` · `반증`

## 축 정의 (고정)

`비용 = f(E 엔진/시스템 부하, M 모델 구조, U 워크로드)` — **배치 ctx 분포 = U × E**. M은 배포 시 상수라 단일 모델 안에서는 변수가 아니다
(조성 귀속은 모델 hold-out에서만). 상세 `cost_model_design_space_2026-10-01.md` §1.0. ★**수식 기반(고정, 결정 27, 2026-10-22)**: 층별 FLOPs·상주 메모리·decode 트래픽의 닫힌 식은 `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1, 계산 단일 출처 `workspace/engine-port/scripts/costmodel/analytic_terms.py`)이고, 각 CM 항은 이 식을 사전(prior)으로 두고 측정으로 효율·PD-mux 항을 덧붙인다 — 요약은 파일 끝 절.

## 진척 요약 (한눈에)

| 항 | 무엇 | 상태 | 다음 데이터 |
|---|---|---|---|
| CM-1 | decode step 시간 `T_dec(SM, bs, ctx)` | 형태 일부 확인 / **hybrid 고유 항 미입증**(연결 근거 부재, 반증 아님) | hold-out 서빙(고비용)·opchar(특성화) |
| CM-2 | prefill span 시간 `T_span(L, SM)` | 형태 확인(마이크로벤치·격리 커널) / **서빙 floor 계수 확정(범위 한정: Nano-9B-v2 무경합)** / chunk -1 장문 prefill 상한 ≈94.8k는 Triton conv int32 오프셋(수리 gate 통과, 성능 항 아님) | T2 격자(rev3.1, 미봉인) · A1 span-SM, opchar 후속 |
| CM-3 | 전환·경계 비용 | **계수 확정(범위 한정)** — 무시 가능 | 컨트롤러 구동 경로만 미측정 |
| CM-4 | 동거 간섭 `δ(p,d)` | **존재 확인(N2 Job-EA)** / 크기는 전력 한도와 교락 / J1: cap 노출 순서가 shape 의존이라 "PD-mux가 클럭을 낮춘다" 일반 가정 불가 | N2 후속 F-1…F-4(사용자 결정) |
| CM-5 | 메모리·admission 예산 | **형태 확정(코드 사실)** · s·k 실측(Nano: k 16,382 B/토큰, s 143.5 MB/요청, L\*≈8.8k [유도]) / 두 구간·cap 특성화는 측정 0 | T4 cap×L 격자(rev3.1, 미봉인; I-5는 gate α PASS) |
| CM-6 | 대기열·TTFT | 구조 확인(entanglement 사슬) / 분해 규약 확정 / J1 B에서 처리율↔SLO 역순은 **미확립**(G0 몫) | N3 용량 · G0 LO·중간 rung |
| CM-7 | prefix cache (radix ON) | 가설 | 보류(radix OFF 우선, 2026-09-24 결정) |
| (E) | 컨트롤러·결정 비용 | **계수 확정** — 무시 가능 | — |
| (확장) | N-GPU·TP, 층 배치(AMD), 에너지 | 가설 — 정확성 게이트·캡처 경로·merge preflight(CPU) `PREFLIGHT_PASS`만 통과(실행 바이트 보증 아님). M3는 J1(N=1)만 완주 — 비용 항 판정은 CM-2 floor 외 0, J2·J4·J8 2026-10-18 21:14 UTC 외부 종료 → **G_sym 전 N 미정의**, G0 뒤로 연기(2026-10-19 결정); multigpu worktree 2026-10-21 폐기·main 흡수 | 각 트랙 TRACK.md · G0 PREREG |

## 항별 장부

### CM-1 decode step 시간
- 제안 형태: attn 층 `∝ bs·ctx`, SSM 층 `∝ bs`(ctx 무관), MLP `∝ bs`, 계열별 SM 스케일링 `s_attn(SM)`, `s_ssm(SM)` (`cost_model_design_space` §1.1).
- **확정**: decode는 SM에 둔감하지 않다 — decode ITL SM16→92 **2.36–2.91×, 4 arm 모델-무관(pure Mamba2 포함)** (C2 `CONFIRMED(scoped)`, 레버 존재만). "decode는 memory-bound라 SM 둔감"(X-7)은 이 기판에서 반증. attention 비중·SM 민감도가 ctx에 따라 커지는 방향(§1-5)은 확정, 절대 knee 값은 마이크로벤치라 인용 금지. "hybrid는 KV가 작아 배치 여유가 크다"는 REFUTED(CM-5).
- **미입증(가장 중요한 빈칸)**: "layer-type별 decode SM 스케일링 차이(a)는 유효 측정 0건(결함 job 철회, per-type UNAUDITED)으로 미입증이다." ★정정(2026-10-19, claims-auditor `anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`): 10-12 의견("증거 사슬이 첫 화살표에서 끊긴다")은 반증이 아니라 **연결 근거 부재**다 — Diff B·C2·opchar는 s_attn≈s_ssm의 근거가 아니다. a는 Claim E의 필요조건이 아니다(§6 Claim E는 "미검증 핵심 가설" 그대로).
- "generic 예측기로 충분" 가설은 기각되지 않았다(검증되지도 않았다). 단일 모델 내 예측표 발산은 재매개화 항등식이며, hybrid 귀속은 모델 hold-out(또는 충실한 이식 generic)에서만 시험 가능하다(A4 K2). N2 PROFILE은 인용 금지.
- 연산자 단위(opchar, 특성화 전용·격리 커널, `AUDIT_opchar_2026-10-13.md` Safe wording 범위): decode에서 SSM state update 비중 B 32→128에서 36%→64%(ctx 2048); 메모리 바운드 연산자는 44 SM에서 108 SM 속도의 ≥86%; GEMM 계단 = wave quantization(104→108 점프).

### CM-2 prefill span 시간
- **확정(마이크로벤치, 서빙 아님)**: attn/SSM SM 민감도 비 Diff B는 L≤512에서 열리고 L≥1024에서 ≈1.0(§1-3 WIDE 스윕). 서빙에서 그 차이를 런타임 per-type 분할로 쓰면 granularity 비용이 압도(TPOT 42→124 ms, §1-3/§1-15) ⇒ 층 종류 차이는 **존재하나 레버가 아니다**.
- **서빙 floor 계수(J1 N=1 감사 CONFIRMED·범위 한정, 2026-10-19)**: "무경합 TTFT floor ≈ 9 ms + 0.111 ms/토큰 × L (L 1k–16k, 기울기 기준 ≈ 9.0k tok/s; Nano-9B-v2, 108 SM, chunked-prefill −1, VESSL A100, Job 4개·boot 8개에서 16k 산포 ±0.6%)."
- 연산자 단위(opchar, 같은 범위 제한): 격리 연산자 시간 합 기준 Mamba 층이 prefill의 ~66–71%(L=1k–32k). "모델 시간의 %"로 인용 금지.
- **장문 prefill 상한(사실, 성능 항 아님)**: chunk -1에서 Nano의 Triton `causal_conv1d` int32 오프셋이 약 94.8k(T ≥ 94,793)에서 넘친다 — 서빙 실패 경계이며 PD-mux 고유 아님. int64 패치는 gate G-β PASS(패치본 98,304 서빙 통과·되돌린 판 95,100 사망·재적용 통과)로 정확성만 확인했다. 감사 `cap_char/probe_128k/AUDIT_probe_128k_2026-10-20.md`: 기전 PLAUSIBLE(강함). 64k는 격자 선택이다. "PD-mux는 64k를 넘을 수 없다"·"OOM 때문" 금지.
- 미감사·인용 금지: prefill 축 SM 스윕(TTFT~L 기울기비). 열린 형태 질문: MuxWise `N_PL` 층 균일 가정(X-4)이 비용 가중 층 수로 바뀌어야 하는가.

### CM-3 전환·경계 비용
- **계수 확정(agnostic 경로·2모델·이 config 한정)**: `s ≤ 0.04 ms/전환`, `d ≤ 0.07 ms/경계` (§1-8/§1-34). 진짜 비용은 전환 자체가 아니라 prefill 생애주기 경계.
- 미측정: 컨트롤러 구동 전환 경로, PAUSE 진입/해제, SSM chunk 경계 state 이월.

### CM-4 동거 간섭
- **구조 확인**: ITL 꼬리는 decode step time이 아니라 un-chunked prefill이 decode를 멈춘 시간이 지배하며 D(decode SM)에 단조 증가 — conjunctive goodput의 ITL 항이 decode-SM 레버와 반대 부호 항을 내장(§1-24). PD-mux는 chunked prefill과 비양립(엔진 assert).
- **존재 확인(N2 Job-EA `7e09cc3`, 라벨 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped))**: "창별 실측 클럭 하락을 전부 빼도 64/64 짝의 ln ρ는 ln 1.03을 넘었다 … `COLOC_SLOWER_GE3`의 존재는 클럭 성분만으로 설명되지 않는다. 이 문장은 크기 진술이 아니다."
- **크기는 미분리**: ρ는 동거 + 전력 한도 클럭 하락의 합성값 — "간섭 크기" 인용 금지(N2C-102), "D·b·c 무관" NOT-YET-SUPPORTED. 분리 경로 = F-1·F-3(사용자 결정 대기).
- **J1 허용 문안 3(감사 2026-10-19)**: "J1에서 SW power cap 노출률의 arm 간 순서는 shape에 따라 달랐고(B에서는 PD-mux가 낮고 X에서는 높음) 부하·전력이 매개한다. 'PD-mux가 클럭을 낮춘다'는 일반 가정으로 쓸 수 없다." (원문 "클럭 가정이 B에서 반대"는 REFUTED)
- **J1 허용 문안 5**: "J1 GPU0 SW power cap(0x4) 노출 60.6%, 셀별 노출은 A 0–6%·B 47–93%·X 80–99%(HI 셀), 400 W 한도. 등록 throttle 공변량은 이것을 잡지 못한다. B·X의 arm 비는 400 W 한도 아래 클럭 하락을 포함한 총효과로만 보고하며, 매개와 교란은 분리하지 않는다."

### CM-5 메모리·admission 예산
- **형태 확정(코드 사실)**: `disable_radix_cache ∧ max_running_requests`이면 `max_mamba_cache_size = max_running_requests`, 즉 cap≡mamba 슬롯 (§1-23). 이는 SGLang 정적 풀(부팅 시 할당, 슬롯 고갈 = assert)의 구현 결과로 할당기에 종속이며(통합 풀이면 소멸) **실험 설정으로 공시**한다. 순수 Transformer에선 admission만.
- **확정**: hybrid KV가 오히려 큼(8.6×), 구속 자원은 mamba 슬롯(2026-09-28 감사). slot당 state 비용은 모델마다 다름(§1-23).
- **E′ 문안(사용자 결정 8′, 2026-10-19 — 결정이며 판정 아님·측정 0)**: "hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져, 짧은 ctx(< s/k, Nano-9B-v2 약 8.8k [유도: 실측 k·s에서, GB 단위; GiB로 9.4k])에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다. 이 비용 구조와 goodput 영향을 양 구간에서 특성화한다. SGLang 정적 풀에서는 cap이 admission·decode bs·state 예약을 함께 정한다(실험 설정 공시)."
- **실측(128k 프로브 `cap_char/probe_128k/RESULT_probe_128k_2026-10-20.md`)**: Nano KV 16,382 B/토큰, state 143.5 MB/요청 → s/k ≈ 8.8k(GB). 이것은 부팅 로그 두 줄이며 경계 자체의 실측이 아니다.
- 금지: "~52k가 hybrid 경계"·"9k 경계 실측"(유도값, config 산술). I-5 거절 사유는 KV / cap(≡슬롯) 2분류 — `src/multiplex/admission_probe.py`; 엔진 묶음 gate G-α 28/28 PASS(정확성 gate, 성능 claim 아님).
- 설계 v1(cap {48, C_mem} × 표준 셀 + long-ctx {32k,64k,96k}): `workspace/engine-port/results/cap_char/DESIGN_cap_longctx_v1_2026-10-19.md`. 낡은 문장: `handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md` §3-2의 "generic 예측기가 C_mem 과대평가"는 결정 8′로 폐기됨.

### CM-6 대기열·TTFT
- **구조 확인(방향, n≥4 변화 trace)**: entanglement 사슬 decode 굶김 → ITL↑ → batch 정체 → admission 차단 → TTFT 폭발(§1-4). TTFT 차이의 ≥95%가 공유 대기열(산술 CONFIRMED).
- **확정**: 비대칭 — decode 과소공급은 고부하에서 파국, 과다공급은 (조건 명시 시) 저부하에서 작은 손실(§1-6). ⇒ 최적 split = max(모델 floor, 부하항 ∝ λ×출력길이)(§1-5, 형태만).
- 분해 규약 확정: `TTFT = W_queue + S_prefill + 잔차`, 정상성 검사 3종(`system_design_anchor` A9).
- **J1(N=1) B shape 허용 문안(감사 2026-10-19)**: "J1(N=1)의 B shape에서, fused 계열은 prefill step 동안 decode가 멈추는 꼬리(ITL 상위 꼬리 ≈ 입력 하나의 prefill 시간)를 보이고 PD-mux arm은 보이지 않는다. 다만 achieved가 같은 rung에서 요청 ITL p95 SLO 통과율은 fused_sym과 PD-mux가 같으며, 이 꼬리는 미조율 fused(chunk 8192, mixed-chunk off)의 노브 효과일 수 있다. 처리율과 SLO의 역순은 미확립(G0 몫)이다." (원문 "처리율↔ITL 역순"은 REFUTED, `AUDIT_J1_N1_analysis_2026-10-19.md`)

### CM-7 prefix cache — 가설 (radix ON 확장 때만)

### (E) 결정 비용
- **계수 확정**: 컨트롤러 CPU 오버헤드 mean 32–36 µs/call — decode step 대비 무시 가능(§1-12). ⇒ 동적 제어 실패는 비용이 아니라 positioning(§1-17, HE0).

## cost model이 이미 답한 것 (논문 서술로 이어지는 문장)

1. 단일 GPU PD-mux의 결정 비용(CM-3·E)은 무시 가능하다 → 정책의 성패는 **예측 정확도(CM-1·CM-6)**에 달려 있다.
2. hybrid의 층 종류 차이는 연산자 비용에는 보이지만(CM-1·CM-2, 격리 커널) 런타임 분할 레버로는 쓸 수 없고(CM-2 실행시 비용),
   그 차이가 SM 스케일링 곡선의 차이(=예측기를 틀리게 만드는 항)로 이어진다는 증거는 아직 없다.
3. hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져 짧은 ctx에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다(CM-5, 구간 경계는 유도값). SGLang 정적 풀에서는 cap≡슬롯이 admission·decode bs·state 예약을 함께 정한다 — 할당기 종속이라 "선행 모델이 놓치는 변수"로는 쓰지 않는다(허수아비 점검, 결정 8′).
4. ITL 꼬리는 동거 간섭(CM-4)이 지배하며 decode-SM 레버와 반대 부호다. 분할 후에도 동거 감속은 남고(Job-EA), 그 일부는 전력
   한도를 거친다 — 단일 GPU 비용 모형에 **전력 항**이 필요할 수 있다(크기 미분리).

## 아직 답하지 못한 것 (= 다음 실험이 채울 칸)

- **계측 관측자 비용(측정 방법 항, 2026-10-21)**: PD-mux 쪽 telemetry 묶음(경로+PHASE_EVENTS, 몫 미분리)이 λ를 −1.7%, I-4가 ITL p95를 +1.0% 움직였다(분해 Job, shape A·agnostic·n=4 한정; 보정 계수 아님). 정책: 판정 측정은 전 arm 계측 OFF + Job 내 ON/OFF 짝 공시(결정 22). 영향 범위는 M3 J1만(P1-opint·HE0 불변; CONSENSUS §8).
- 비용 항 측정(T1–T4)은 PREREG rev3.1 미봉인(체크리스트 6/14); 검증(V)은 항 데이터 뒤 별도 PREREG.
- `s_attn(SM)` ≠ `s_ssm(SM)`인가 — 유효 측정 0건(Claim E 판정서 2026-10-19). 이 칸이 비어 있어도 Claim E는 반증되지 않으며, a는 E의 필요조건도 아니다(우회: 모델 hold-out 조성 귀속·memory/admission). 판정에는 hold-out 서빙이 필요하다(고비용).
- `δ(p,d)`의 **전력 무관 크기**와 prefill 종류 의존 (N2 후속 F-1…F-4, 사용자 결정 대기). J1 GPU0 SW power cap 노출 60.6%라 multi-GPU 쪽도 총효과로만 읽힌다.
- 워크로드 축: shape C(긴 입력 + 긴 출력)는 2026-10-19 사용자 결정으로 G0 두 번째 표준 셀로 편입됐으나(구체값은 G0 PREREG) **아직 측정 0**. hybrid ctx 항이 의미를 갖는지는 열린 질문이다.
- hybrid의 고정 state s 대 KV의 두 구간(짧은 ctx에서 s, 긴 ctx에서 KV)과 cap의 goodput 영향 — 설계 v1만 있고 측정 0(I-5 계측·GPU gate 선행).
- ρ의 D·b·c 무관성 (F-4) — 해제는 F-4 격자 등록 동등성 라벨 새 캠페인의 결과 감사로만(C-2). N2·N3·E2는 2026-10-15 종료, 게이트 #6 미닫힘.
- N-GPU·TP(multi-GPU)와 층 배치(AMD)가 새로 만드는 항(통신·핸드오프). M3는 J2·J4·J8 외부 종료로 G_sym이 전 N에서 미정의이며 G0 뒤로 연기됨(2026-10-19 결정); 하네스 시간 모형이 확장 셀을 빠뜨려 rev12 예산(190 GPU-h)이 과소했다(`results/multigpu_verdicts/M3_SCOPE_OPTIONS_2026-10-19.md`).

## 해석적 수식 기반 (고정, 결정 27 — 2026-10-22; 기존 줄 번호 보존을 위해 파일 끝에 둔다)

정본 수식 문서 `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1)의 요약. 수치는 모두 [유도]이며 변경은 그 문서의 변경 규칙을 따른다.
- **상주 메모리**: s = n_M·(n_h·p·N·β_s + d_conv·(K−1)·β), k = n_A·2·n_kv·d_h·β, L\* = s/k. Nano: 143.5 MB · 16,384 B/토큰 · ≈ 8.8k (배너 실측과 일치) → **CM-5**.
- **prefill**: F_pre(L) = α·L + γ·L² (+ 2dV), γ = n_A·2·n_q·d_h. Nano: α = 15.65 GFLOP/토큰, γ = 40,960, 이차항 = 선형항인 L ≈ 382k(64k에서 17%) → **CM-2**. J1 floor 기울기 0.111 ms/토큰은 실효 약 141 TFLOP/s(피크의 약 45%)에 해당.
- **decode**: 메모리 바운드 하한 T_dec ≥ a₀ + b₀·b + c₀·b·c, a₀ = 가중치 바이트/BW(Nano 8.14 ms), b₀ = 2s/BW(0.141 ms/요청, SSM state 읽기·쓰기), c₀ = k/BW(8.0 ns/요청·토큰). state 항 = KV 항인 문맥 c\* = 2s/k ≈ 17.5k → **CM-1**의 함수형 `a + b·bs + c·bs·ctx`의 물리적 하한.
- **조성 스케일링**: k와 c₀는 n_A에 정확히 비례, s는 n_M에 비례, prefill 비선형성은 n_A에 따라 단조 증가(조성 변형 표는 정본 수식 문서 §7) → J-H 예측 P3·P4의 근거.
- **확장 틀**: 측정이 덧붙이는 항 = 층 종류별 효율 η_X(SM), PD-mux SM 분할, lockstep·동기화, 동거 간섭 ρ, cudagraph 패딩, SGLang 정적 두 풀 할당, TP 분할(정본 수식 문서 §8).
