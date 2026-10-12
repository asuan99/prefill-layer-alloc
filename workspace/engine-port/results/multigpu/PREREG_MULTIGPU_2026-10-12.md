# 사전등록 — Multi-GPU(A100 ×1/2/4/8, DP·TP) 정확성 게이트 M2g + 용량비 M3 + 변화 trace M4 (2026-10-12/13, rev9)

> **지위**: 설계 문서 `reports/multigpu_scaling_design_2026-10-12.md`(rev2)의 M2.5. 감사 판정
> `reports/audit/2026-10-12_multigpu_design/AUDIT_multigpu_design_rev1_2026-10-12.md`(`MIXED`)의 MG-0이 요구하는 **결정 규칙 등록**이다.
> **rev2**: rev1 규칙층 감사(`VERDICT_prereg_rules_rev1_2026-10-12.md`, `MIXED`: M2g `GO-with-caveats`, M3 `NO-GO` — 死因 D1·D2)를
> 반영했다. 대조표 §9. MGP-0…MGP-16을 문자 그대로 승계한다. **델타 감사 대상 = D1·D2·MGP-1·MGP-10.**
> **rev3**: rev2 델타 감사(`VERDICT_prereg_rules_rev2_delta_2026-10-12.md`, `MIXED`: M2g `GO-with-caveats` 유지, M3 `NO-GO` — D3 입장 기계 항등식,
> D4 arm 비대칭 표본, D5 메모리 정확 일치, D6 `ps %cpu`)를 반영했다. 대조표 §10. MGP-17…MGP-25 추가 승계.
> **rev4**: rev3 델타 감사(`VERDICT_prereg_rules_rev3_delta_2026-10-12.md`, `MIXED`: M2g `GO-with-caveats` 유지, M3 `NO-GO` — D7 fused prefill 우선으로
> shape B Decode 줄 고갈, D8 NVML memory.used 폴백이 PD-mux 캡처 메모리를 불일치로 판정)를 반영했다. 대조표 §11. MGP-26…MGP-32 추가 승계.
> **rev5**(사용자 결정 2026-10-12): MGP-29(arm 간 스케줄 노브 비대칭) 대응으로 대칭 대조군 `fused_sym` arm을 추가하고 1차 estimand를 G\_sym = agnostic/fused\_sym으로 바꿨다. 대조표 §12.
> **rev9**(2026-10-13): rev8 규칙층 감사(`VERDICT_rules_rev8_2026-10-13.md`, `MIXED` — M3 死因 D9: G(1) 미정의 시 교차 Job 추세 fail-open) 처방 P9-a–d 반영 + 사용자 결정 3건(메모리 규칙 (iii) 폐지[사후 변경 공시], Job별 하드캡 예상치 기준, J1을 x2 노드 GPU 0 단독). 상세 §16 — §16이 §15와 충돌하면 §16이 우선. MGP-42…50 승계. **규칙층 경량 델타 감사 대기.**
> **rev8**(사용자 결정 2026-10-12): **Job을 구성별 적정 크기로 나눈다**(x8 한 Job 순차 실행 폐기 — 예약 GPU 사용률 M2g 실측 59%, M3 예보 약 50%로 크레딧 낭비). rev7 M2g 3회차 Job(`pdmux-multigpu-f515970-20261012142501`)은 실행 40분 만에 사용자 결정으로 취소(산출물 없음, 측정 0). 상세 §15 — §15가 §1 '기판'·'GPU 부분집합'·§4-1 실행 순서·§4-5 앵커·§4-6 기술 블록·§6을 대체한다. **규칙층 델타 감사 대기.**
> **rev7**(사용자 결정 2026-10-12, M2g 2회차 `RESULT_m2g_c1395d0_2026-10-12.md` 대응): fused arm에 `--disable-piecewise-cuda-graph`. 그 결과 fused와 fused\_sym의 차이는 overlap·chunked **두 노브**로 줄어든다(MGP-36의 세 번째 노브 제거 — MGP-36 문장 중 'piecewise' 부분은 rev7 아래에서 해당 없음, S는 여전히 두 노브의 합성). M2g는 이 정의로 **전부 재실행**한다(같은 Job 원칙). 대조표 §14. **규칙층 델타 감사 대기.**
> **rev4+rev5 델타 감사**(`VERDICT_prereg_rules_rev5_delta_2026-10-12.md`): **M2g·M3 모두 `GO-with-caveats`**(死因 0). MGP-0의 'M3 `NO-GO`'는 해제됐다.
> 이 등급은 규칙 실행 허가이지 'M3가 PD-mux 스케일링을 판정할 수 있다'는 뜻이 아니다. **rev6**은 그 판정서 §8-1의 경량 반영(표 행·caveat·고정 조건·예산)이며, 하네스 H1–H10 작성과 하네스층 감사가 다음 gate다. 제출은 사용자 승인. `GO`/`GO-with-caveats` 전에는 M2g·M3 Job을 `--submit`하지 않는다(교훈 34). 하네스(plan·label 스크립트)는
> 규칙 감사 뒤에 작성하고 별도로 하네스층 감사를 받는다.
> 작성 시점: GPU 지출 = M1 프로브 1건뿐, 서빙 측정 0, 성능 판정 0, Claim 변경 0.
> 브랜치 `multigpu-scaling`(worktree `prefill-layer-alloc-multigpu`), 기준 커밋 `c0e200d`. 이 브랜치의 엔진 트리는 N2/N3/E2 봉인 digest와
> 다르다(의도). 이 사전등록의 엔진 digest는 하네스 봉인 때 이 브랜치 트리로 새로 고정한다.
> 사용자 결정(2026-10-12, 설계 §9): 즉시 병행 시작 · M1+V0 묶음(완료) · x8에는 Job GPU-h 한도 미적용 · TP cap = 48·N · x8을 못 잡으면 보류.

## 0. 한 줄 요약

x8 노드 한 대에서 (1) M2g Job으로 TP=2/4/8 PD-mux가 **돌아가기는 하는지**를 판정하고, PASS한 (mode, N)만 대상으로 (2) M3 Job에서
arm별 포화 처리율 λ\*를 재어 **1차 estimand G(N, mode) = λ\*\_agnostic / λ\*\_fused의 N-추세**를 판정한다. (3) M4 변화 trace는
2차 estimand이며 M3 결과를 입력으로 하는 결정적 규칙만 여기서 등록한다.

## 1. 고정 조건 (전 단계 공통)

| 항목 | 등록값 | 근거·강제 |
|---|---|---|
| 기판 | VESSL `resourcespec-a100x8` Job. M2g 1개, M3 1개(필요 시 블록 분할, §6-4) | M1 결과: 전 쌍 NV12, NUMA0 = GPU0–3, NUMA1 = GPU4–7 |
| 기판 gate | `substrate_probe_helper score-b0 --expected-gpus 8`. A100·cc8.0·MIG off·UUID 8개 상이·드라이버 균일 | 실패하면 `ABORT_SUBSTRATE`(GPU 작업 전) |
| 모델 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`, revision `dc0661c829b14e5b9246c05cfa89094a0875e052` | λ0 VESSL과 동일. 불일치하면 `ABORT_MODEL_REVISION` |
| TP 분할 가능성 | attention heads 40, KV heads 8, mamba heads 128, n_groups 8, intermediate 15680, vocab 131072(HF config, 2026-10-12 조회) → TP ∈ {2,4,8} 모두 나누어떨어짐 | 실제 TP 지원 여부는 M2g G0에서 판정 |
| shape | A = (input 256, output 512), B = (input 8192, output 64) — λ0 `SHAPE_IO` 그대로 | `lambda0_plan.py:107` |
| cudagraph | ON | 게이트 운영점 |
| radix cache | OFF(`--disable-radix-cache`, runner 상속) | 설계 §5-2 |
| cap | N=1: 48. TP=N: 48·N. DP=N: 복제본당 48(`--dp-size`, dp-attention 없음) | 사용자 결정. runner가 강제 |
| DP 라우팅 | `--load-balance-method round_robin` | 설계 §3-3 |
| 런처 | `engine_bench_runner.sh --tp-size/--dp-size/--gpu-ids`. 한 캠페인 안에서 N-독립-서버 폴백과 섞지 않는다(C-7) | M2g G0에서 `--dp-size`가 FAIL하면 DP 축 전체를 보류하고 이 문서에 追記한다(폴백은 새 등록) |
| 통신 env | `NCCL_*`, `CUDA_DEVICE_MAX_CONNECTIONS`, `SGLANG_CUSTOM_ALLREDUCE_ALGO` = **unset**(기록만). `SGLANG_USE_JIT_ALL_REDUCE` unset 강제(설정되면 runner rc 3) | C-10. `comm_env.json`에 셀마다 기록 |
| 클라이언트 | `trace_loadgen.py --connector-limit 0`(연결 상한 해제) | M1.5 로컬: 기본 상한 100에서 32 req/s부터 클라이언트 큐잉 |
| GPU 부분집합(주) | N=1 `{0}`, N=2 `{0,1}`, N=4 `{0,1,2,3}`, N=8 `{0..7}`. 부분집합은 **순차** 실행(동시 실행 금지) | C-1. NUMA 교차는 N=8에만 생긴다 → §4-6 민감도 블록 |
| attention backend | `flashinfer`(`PDMUX_ATTENTION_BACKEND`, 셀마다 기록). runner 기본 triton은 NemotronH assert로 거부된다(`server_args.py:1959-1962`) | H4 |
| 클라이언트 timeout·fd | `trace_loadgen` `sock_read` = 600 s 유지(MGP-35 문턱의 전제), 클라이언트 `RLIMIT_NOFILE` soft ≥ 8192로 상향하고 `ulimit -n`을 셀마다 기록 | H5, H6 |
| piecewise CUDA graph | **rev7: fused arm만 `--disable-piecewise-cuda-graph`**(나머지 arm은 chunked −1이라 원래 piecewise 경로에 들어가지 않음). 셀마다 `disable_piecewise_cuda_graph`·`piecewise_cuda_graph_tokens`·캡처 로그 기록 | H8, rev7 |
| 서버 로그 | `--decode-log-interval 40`(SGLang 기본값을 명시 등록), `SGLANG_LOG_MS=1`(밀리초 타임스탬프). 판정에는 `Decode batch` 줄만 쓴다(§4-3) | D3 처방 |
| 기록 | GPU별 clk·throttle(job_entry `clk.csv` index 열), `cpu.stat` 셀 전후, `cell_meta.json`(max_total_tokens·mamba cache·effective cap), realized split telemetry(rank 접미사) | C-1, C-4, C-5 |

## 2. arm 정의

| arm | 정의 | 비고 |
|---|---|---|
| `fused` | `PDMUX_ENGINE_MODE=vanilla`(PD-mux off), 엔진 기본 스케줄(overlap ON, chunked prefill 8192) + **`--disable-piecewise-cuda-graph`**(rev7) | policy_adapter `vanilla:vanilla`. **운영 기본값 대조군**(단 piecewise CUDA graph는 끈다 — 이 이미지의 Python 3.14 + torch 조합에서 piecewise 캡처가 `typing.Union.__module__` 비호환으로 기동 실패, M2g 2회차 7/7. KISTI fused cudagraph-ON 선례와 동일) |
| `fused_sym` | `PDMUX_ENGINE_MODE=vanilla`(PD-mux off) + PD-mux arm과 **같은 스케줄 노브** `--chunked-prefill-size -1 --disable-overlap-schedule`(runner가 PD-mux arm에 붙이는 인자와 바이트 동일) | **대칭 대조군**(사용자 결정 2026-10-12, MGP-29 대응). PD-mux arm과의 차이는 PD-mux 기능 전체(SM 분할 + split prefill + 입장 기계)이며 'SM 분할 단독'은 아니다 |
| `agnostic` | PD-mux on, **R2 정책 없음**(stock 임계표), config = `pdmux_agnostic_manual.yml`: `sm_group_num 4`, `manual_divisions [[74,34,0],[54,54,36]]`, `decode_bs_divisor 36` | A100 자동 격자 `divide_sm(108, (8,0), 2)` = [(74,34),(54,54)], idx = max(1, min(2, decode_bs·2//36))와 **같은 선택 함수**를 manual 표로 옮긴 것. 자동 격자 경로는 TP/DP>1에서 device assert로 거부되므로(M2 발견) manual 등가표를 쓴다. 등가성은 M2g G5에서 N=1 auto vs manual realized 선택 일치로 검증 |
| `d44` | PD-mux on, `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`, config `pdmux_r2.yml` | 1-GPU decode-heavy static 권고의 D. **N마다 재조율하지 않는다** — 이 arm에 대한 결론은 "1-GPU에서 고른 D44를 N에 그대로 쓴 경우"로 범위를 한정한다(C-6) |

- **agnostic 임계표의 mode 의미(C-6)**: 임계값 36은 **엔진 인스턴스의 decode 배치 크기** 기준이다. TP=N에서는 전체 배치이고(GPU당 연산은 ≈1/N), DP=N에서는 복제본 배치(≈전체/N)다. 이 표를 N에 맞춰 스케일하지 않는다. G(N, mode)는 "1-GPU 표를 그대로 쓴 agnostic"의 값이다.
- 동적 정책(R2 generic/hybrid, SLO_SCHED, true-dual)은 이 사전등록에 없다. TP에서는 엔진이 거부한다(`multigpu_guard`).
- `sticky` 플래그는 OFF(기본).

## 3. M2g — 정확성 게이트 (Job 1)

"PASS"는 아래 시험 범위(모델 1개, 등록 prompt 집합, 등록 길이)의 정확성만 뜻한다(C-8, MGP-2…MGP-4).

| 게이트 | 내용 | PASS 조건(등록) |
|---|---|---|
| **G0 boot·거부** | (i) TP∈{2,4,8}, DP∈{2,4,8} × arm{fused, fused\_sym, agnostic, d44} 서버가 기동하고 `/health` 응답. (ii) TP=2에서 금지 env **5종을 하나씩**: `PDMUX_R2_POLICY=generic`, `PDMUX_R2_POLICY=hybrid`(+ 더미 profile), `PDMUX_SLO_SCHED=1`, `PDMUX_LA_COORD=1`, `PDMUX_TRUE_DUAL_WORKER=1`. (iii) TP=2 + 자동 격자 config | (i) 기동 제한시간 1800 s 안에 health(runner 대기 상한을 1800 s로 맞춘다, H2; 그보다 짧은 대기에서 난 실패는 `NOT_TESTED`, MGP-23). (ii) 5종 모두 서버 로그에 `multigpu_guard` 등록 에러 문자열 + 프로세스 비정상 종료. (iii) device assert 문자열 + 비정상 종료 |
| **G1 PD-mux OFF↔ON 출력** | 같은 (mode, N)에서 `fused`↔`agnostic`, `fused`↔`d44`. 등록 prompt 64개(r2_correctness_client S/O tier, seed 고정), greedy, max_new_tokens 128, 동시 요청 32 | **대조군 G1-ctl**(N=1 같은 비교)을 같은 Job에서 먼저 잰다. 지표 ① prompt별 첫 불일치 이전 일치 토큰 비율 m, ② 불일치 prompt 비율 q(m < 1인 prompt의 비율). PASS ⇔ m 중앙값 ≥ G1-ctl m 중앙값 − 0.05 **그리고** q ≤ G1-ctl q + 0.10 **그리고** 출력 길이 0·비정상 종료 0. 노출 비율(분할 group에 있던 decode step 비율, telemetry)을 병기한다(MGP-2). G1-ctl의 m 중앙값 < 0.5이면 G1은 `UNINFORMATIVE`(FAIL 아님, 정확성 문장 금지) |
| **G2 TP=N↔TP=1 teacher-forced** | `fused` arm, 같은 64 prompt, TP=1 greedy 출력 열을 teacher-forced로 TP=N에 넣어 위치별 top-1 logprob 비교 | 바닥 = TP=1 동일 서버 2회 반복의 같은 지표. PASS ⇔ \|Δlogprob\| p99 ≤ max(4 × 바닥 p99, 0.05 nat) 그리고 top-1 불일치율 ≤ 2%. FAIL은 `TP_NUMERIC_DIVERGENT` 표기만 하고 제외하지 않는다(MGP-3) |
| **G3 comm 역순 스트레스** | `m2g_comm_reverse_stress.py`: TP=N마다 두 pynccl comm(prefill/decode)을 짝·홀 rank가 반대 순서로 발행, plain stream과 green stream(16/92, 44/64) 각각 | 반복 R = 500/조합, watchdog 120 s. PASS ⇔ hang 0 **그리고** 정수 합 all-reduce 검산 오류 0 |
| **G3b 서빙 스트레스** | TP=N × arm{agnostic, d44}, shape A, offered = 1.5 × (N × 2.99) req/s, 600 s | 완료 요청 > 0, 서버 생존, 마지막 60 s 안 토큰 진행 |
| **G4 캡처·커널 식별** | stream group별 cudagraph 캡처 로그. 짧은 nsys 프로브(TP=2,8 × d44, 30 s)로 실제 all-reduce 커널 이름 | 캡처 성공 = PASS 조건. 커널 이름은 기술치 |
| **G5 파티션** | (a) agnostic manual 등가표 ↔ 자동 격자 선택 함수의 등가성: **결정적 CPU 시험**(decode_bs 0…512 전수에서 두 경로의 `stream_idx`·`SM_COUNTS` 일치, 하네스 회귀 테스트). GPU에서는 N=1 auto vs manual의 선택 분포를 **기술치**로만 기록(MGP-5). (b) DP=N 복제본별 telemetry의 실현 `(p,d)`가 그 arm의 등록 표 안의 값인지. (c) arm 간 메모리 조건(MGP-12, P-D5): 같은 config 안 네 arm에 대해 `MEMORY_MISMATCH` ⇔ (i) mamba cache 슬롯 수 ≠ effective cap(정확 비교), 또는 (ii) `max_total_num_tokens` < effective cap × (ISL+OSL)(KV 구속: A 36,864·N, B 396,288·N 토큰 — TP는 엔진 인스턴스 cap 48N 기준, DP는 복제본 cap 48 기준), 또는 (iii) arm 간 `max_total_num_tokens` 상대차 > 1%. 그 외 차이는 기술치(MGP-18). 오염 검사(P-D8): (가) 각 boot 전 부분집합 GPU의 NVML `memory.used` < 1 GiB(빈 GPU 확인), (나) NVML `nvmlDeviceGetComputeRunningProcesses` **항목 수**가 GPU당 기대 프로세스 수와 같음(PID 네임스페이스 무관). 항목 수조차 0으로 숨겨지면 (가)만 쓴다. 어느 하나라도 어긋나면 `GPU_CONTAMINATED`(그 셀 판정 제외). arm 간 `memory.used` 비교는 **기술치**(boot 직후·셀 종료 시 기록)이며 `MEMORY_MISMATCH` 조건이 아니다 — PD-mux는 SM group마다 cudagraph를 캡처해 구조적으로 더 쓴다(MGP-32 기판 고지) | (a) CPU 시험 실패 → 하네스 봉인 불가. (b) 실패 → 해당 DP=N 제외. (c) 실패 → 해당 (mode, N)에 `MEMORY_MISMATCH`, M3 G 인용 금지(측정은 진행) |

### 3-1. FAIL 단위와 처리 (MGP-1)
| 결과 | 제외 범위 |
|---|---|
| G0(i) `fused` 기동 실패 | 그 (mode, N) 전체 |
| G0(i) `fused_sym` 기동 실패 | 그 (mode, N)의 fused\_sym arm → G\_sym 미정의(G\_def를 대입하지 않는다, MGP-38). G\_def·G\_d44 계열 보조 라벨은 유지 |
| G0(i) `agnostic` 또는 `d44` 기동 실패 | 그 (mode, N)의 해당 arm만 |
| G0(ii) 5종 중 하나라도 거부 실패 | **TP 축 전체**(guard가 작동하지 않으면 TP 결정 분기 위험이 남는다). DP 축은 영향 없음 |
| G0(iii) 거부 실패 | 제외 없음(M3는 자동 격자를 쓰지 않는다). 발견으로 기록 |
| G1 `fused↔agnostic` FAIL | 그 (mode, N)의 agnostic arm → G 미정의 |
| G1 `fused↔d44` FAIL | 그 (mode, N)의 d44 arm → G\_d44 미정의 |
| G1-ctl `UNINFORMATIVE` | 제외 없음. 정확성 문장 금지 |
| G3 또는 G3b FAIL | 그 TP=N 전체 |
| G4 캡처 실패 | 그 (mode, N)의 해당 arm |
| G5(b) FAIL | 그 DP=N 전체 |
| 하드캡 등으로 **시험하지 못한** 게이트 | 그 게이트가 덮는 범위를 FAIL과 같이 제외하고 `NOT_TESTED`로 표기 |

M2g Job은 `M2G_PASS_TABLE.json`((mode, N, arm) → 포함/제외 + 사유)을 산출한다. M3 Job은 이 표의 sha256을 등록값과 대조하고 **포함된 셀만** 실행한다. 표가 없거나 digest가 다르면 `ABORT_M2G_NOT_PASSED`(Job 전체).

## 4. M3 — 용량과 1차 estimand (Job 2)

### 4-1. 셀 구성
- 구성(config) 7개: N=1, DP={2,4,8}, TP={2,4,8} 중 `M2G_PASS_TABLE`이 포함한 것.
- arm 4개(fused, fused\_sym, agnostic, d44), shape 2개(A, B).
- 서버 기동은 (config, arm)마다 1회. shape A·B와 seed 반복은 같은 boot 안에서 순차로 돈다. **boot 효과와 seed 효과는 분리되지 않는다**(MGP-7).
- 실행 순서(하드캡 절단 시 추세 점이 남도록, MGP-14): GPU 편차 블록 → **N=1(1회차)** → TP8 → DP8 → TP2 → DP2 → TP4 → DP4 → NUMA 블록 → nsys → **N=1(2회차)**. arm 순서는 config마다 seed 20261012 기반 무작위 순열.

### 4-2. 측정 rung (shape·arm마다)
- prior(N, shape) = N × λ\*\_VESSL(shape)(A 2.99, B 0.690 req/s). rung 위치를 정하는 데만 쓴다.
- **HI rung**: offered x\_hi = 1.6 × prior, 요청 수 = clamp(round10(x\_hi × 170 s), 40, 4000), seed 4개(4386, 4162, 251, 2630).
- **LO rung**: offered x\_lo = 0.60 × (그 (config, shape)에서 **fused** HI achieved seed 중앙값), 요청 수 = clamp(round10(x\_lo × 170 s), 40, 4000), seed 2개(4386, 4162). 같은 (config, shape)의 네 arm이 **같은 x\_lo·요청 수**를 쓴다(MGP-9). boot 배치: (config, arm)마다 **boot 1** = shape A·B의 HI(seed 4개씩). 네 arm의 boot 1이 모두 끝나 x\_lo가 정해진 뒤 (config, arm)마다 **boot 2** = shape A·B의 LO(seed 2개씩). 따라서 boot는 (config, arm)당 2회이고, HI와 LO는 서로 다른 boot다(범위 공시).
- **확장 규칙(1회, 공동)**: (config, shape)의 **어느 한 arm이라도** HI에서 g < 0.10(§4-3)이면 그 (config, shape)의 **네 arm 모두** x\_hi를 2배로 올려 HI를 다시 잰다(arm마다 새 boot 1′, x\_lo는 확장 후 fused HI 중앙값으로 정한다). `SATURATION_UNRESOLVED`·`CLIENT_ERROR`는 확장을 발화하지 않고 라벨로 남긴다(교훈 21). 확장 뒤에도 g < 0.10인 arm은 `LADDER_TOO_LOW`.
- ratio(achieved/offered)는 **기술치**이며, 분모는 셀의 **realized 도착률**(요청 수 ÷ 첫 전송~마지막 전송 구간)로 쓴다.

### 4-3. 라벨 (shape·config·arm마다) — 독립 포화 증거 (P-D1, λ0R-4·MGP-8)
- **대기열 원천 = 클라이언트 측 시스템 내 대기 수**(P-D7, arm 대칭): Q\_c(t) = #{요청: sent\_s ≤ t} − #{요청: sent\_s + ttft ≤ t}(첫 토큰을 아직 받지 못한 요청 수). `requests.jsonl`의 `sent_s`·`ttft_ms`로 계산한다. 서버 로그·시계 다리가 필요 없고, 송신 창 안의 사건만 쓴다. TTFT가 없는 오류 요청은 Q\_c에서 빼고 개수를 기록한다. 오류 요청이 셀 요청의 1%를 넘으면 그 셀은 `CLIENT_ERROR`(판정 제외).
- **g**: 송신 창 [첫 sent\_s, 마지막 sent\_s], 길이 T를 시간으로 3등분하고, 각 구간에서 Q\_c의 **시간 적분 평균**을 낸다. g = (마지막 1/3 평균 − 첫 1/3 평균) / ((2T/3) × r), r = n/T(엔진 전체; DP도 엔진 전체로 판정하고 복제본 균형은 서버 로그 기반 기술치). 셀의 송신 창 안 요청 수 < 20이면 `SATURATION_UNRESOLVED`.
- **HI 포화**: 4 seed 모두 g ≥ 0.10이면 `HI_SATURATED`. 이 라벨은 x\_hi/λ\*의 수치 하한을 주지 않는다(MGP-26).
- **LO 비포화**: 2 seed 모두 g ≤ 0.05이면 `LO_UNSATURATED`. 이 라벨은 x\_lo < λ\*를 보증하지 않는다(MGP-27).
- 문턱 0.10/0.05의 출처는 rev2 델타 감사 처방 P-D3/D4다(MGP-28). 대기열 원천은 rev3의 서버 Decode 줄에서 rev4의 클라이언트 Q\_c로 바뀌었고 문턱은 유지했다.
- 서버 로그의 `#running-req`·`#queue-req`(Decode·Prefill 줄)는 **기술치**로만 기록한다(MGP-17). 셀 자신의 마지막 요청 지연으로 drain을 보정하는 계산은 쓰지 않는다(교훈 9).
- 하네스 변이 검사(교훈 53): (a) "서버 Decode 줄만 쓰는 추정량" 변이본이 fused prefill 우선 합성 로그(cap까지 Prefill만 → 63-step burst)에서 KILLED, (b) "표본 평균(시간 적분 아님)" 변이본이 burst 도착 합성 trace에서 KILLED, (c) "오류 요청을 Q\_c에 남김" 변이본 KILLED, (d) "running==cap 술어" 변이본 KILLED.
- λ\*(config, arm, shape) = HI 4 seed achieved 처리율(요청 수 ÷ 첫 전송~마지막 완료)의 **중앙값**, 4값 범위 병기.
- `KNEE_BRACKETED` ⇔ `HI_SATURATED` ∧ `LO_UNSATURATED`. 아니면 `KNEE_NOT_BRACKETED`(λ\* 기록, G 판정 제외).
- `HARNESS_LIMITED` ⇔ 셀 중 하나라도 클라이언트 전송 지연 p99 > 5 ms 또는 셀 구간 cgroup **nr\_throttled 증가분 / nr\_periods 증가분 > 1%**(분모 등록, MGP-31). throttled\_usec/벽시계도 함께 기록하되 판정에는 쓰지 않는다.
- `FRONTEND_SATURATED`(MGP-11, P-D6) ⇔ HI 셀 송신 창에서 launch\_server 주 프로세스(TokenizerManager), detokenizer, DP controller 중 하나라도 `/proc/<pid>/stat` utime+stime의 **1 s 구간 델타**가 0.90 코어 이상인 초의 비율 ≥ 20%. `ps %cpu`(수명 평균)는 쓰지 않는다(MGP-19). 이 표기가 붙은 config의 G·λ\*는 "GPU 확장성"·"PD-mux 스케일링"으로 쓰지 않는다(측정·라벨은 유지).
- **E-time(P-D2)**: decode-busy = `decode_running_batch_size` > 0(A2 술어), 시간 가중 = 다음 non-startup 스냅샷까지의 간격(e2 규약), **분자 = `stream_index` ∉ {0, real_sm_group_num−1}**(분할 group 전체). arm별 `sm_counts`를 green readout과 대조하고, 불일치하면 그 셀만 `MIX_FAILED`(ABORT 안 함). TP는 rank 0, DP는 복제본 평균. PD-mux arm의 HI 4 seed 중앙값 E-time < 0.5이면 `PARTITION_NOT_EXERCISED`(MGP-6). `e2_realized_mix.py`의 `D44_STREAM_INDEX = 2` 하드코딩은 이 등록에 쓰지 않는다 — 새 추정 함수를 하네스에 두고 "idx == 2 문자 그대로" 변이본이 agnostic·r2 합성 telemetry에서 KILLED 됨을 회귀 테스트로 보인다.

### 4-4. 1차 estimand와 판정
- **1차 estimand**: G\_sym(N, mode, shape) = λ\*\_agnostic / λ\*\_fused\_sym(같은 스케줄 노브 대비 PD-mux 기능의 용량비). **2차**: G\_def = λ\*\_agnostic / λ\*\_fused(운영 기본값 대비), G\_d44,sym = λ\*\_d44 / λ\*\_fused\_sym, 기술치 S = λ\*\_fused\_sym / λ\*\_fused(스케줄 노브 효과). 각 비는 두 arm 모두 `KNEE_BRACKETED`, `HARNESS_LIMITED` 아님, config에 `MEMORY_MISMATCH`·`GPU_CONTAMINATED` 없음일 때만 정의된다. 아래 CI·추세·크기·예측 규칙은 G\_sym에 적용하는 것이 판정이고, G\_def·G\_d44,sym에는 같은 계산을 **보조 라벨**로 붙인다(다중 비교: 판정 문장은 G\_sym만). 이하 'G'는 G\_sym을 뜻한다.
- **G(1)은 N=1 1회차 값**이다(MGP-10). 2회차는 드리프트 검사에만 쓴다.
- **CI**: seed 4개를 짝으로(같은 seed의 agnostic/fused\_sym achieved 비; 보조 라벨은 해당 분모) bootstrap 10,000회(seed 1), 95% 백분위 CI. 이 CI는 boot 산포를 포함하지 않는다(MGP-7).
- **추세**: mode마다 log G를 log₂N(정의된 N ∈ {1,2,4,8})에 최소제곱 회귀한 기울기 s. **재표집 단위 = seed 색인**: 한 번의 bootstrap에서 seed 색인 4개를 복원추출하고, 그 색인을 그 mode의 **모든 N과 두 arm에 공동 적용**한다(MGP-10). 정의된 점 < 3이면 `TREND_UNDEFINED`.
  - CI가 0 포함 → `NO_TREND_DETECTED`, CI > 0 → `G_INCREASES_WITH_N`, CI < 0 → `G_DECREASES_WITH_N`.
- **크기 규칙**: 분모 = G(1), 분자 = G(N\_max), N\_max = 그 mode에서 G가 정의된 가장 큰 N(보고에 명시). G(1)이 정의되지 않으면 크기 규칙은 적용하지 않는다(방향 라벨만, `MAGNITUDE_UNDEFINED`). |G(N\_max)/G(1) − 1| < 3%이면 방향 라벨도 방향의 증거로 인용하지 않는다(MGP-7).
- **등록 예측**: P-DP — G\_DP(N) 추세 `NO_TREND_DETECTED` 예측. 깨지면 MGP-13: 새로 등록·감사받은 진단 전까지 TP 라벨 인용 금지. P-TP — 방향 예측 없음(탐색). G(N) < 1도 유효한 결과(MGP-16).
- e(N, mode, arm) = λ\*(N)/(N·λ\*(1))은 기술치로만 보고한다.

### 4-5. 블록 앵커
- N=1 2회차의 λ\*(fused, A), λ\*(fused\_sym, A), G\_sym(1, A) 중 **하나라도** 1회차와 > 3% 다르면 `ANCHOR_DRIFT`를 붙이고 N-추세 라벨에 드리프트 공시를 의무로 붙인다(MGP-10).
- Job을 블록으로 나누면 블록마다 N=1 fused·fused\_sym·agnostic shape A HI(seed 4개)를 앵커로 다시 재고 같은 규칙을 적용한다(C-9 후단).

### 4-6. 기술 블록 (판정 아님)
- **GPU 편차(C-2)**: N=1 `fused` shape A HI rung seed 4386을 GPU 0..7 각각에서 1회(boot 8회). 최대/최소 achieved 차이 > 3%이면 `GPU_HETEROGENEOUS`를 붙이고 N=1 앵커를 "GPU0 값"으로만 쓴다. 이 블록은 M3 Job의 맨 앞에서 실행한다.
- **NUMA 민감도(C-1)**: TP=4를 `{0,1,4,5}`(NUMA 교차)로 `fused`·`agnostic` shape A HI rung seed 4개. 주 부분집합 `{0,1,2,3}`과의 G 차이를 기술치로 보고한다. 판정에는 쓰지 않는다.
- **통신 기전 nsys**: TP=8 `d44` shape A HI rung 30 s 창에서 all-reduce 커널 시간 비율과 그 커널이 돈 스트림(prefill/decode green ctx). 측정 run과 분리된 짧은 boot에서 잰다.
- 모든 셀에서 TTFT·ITL p50/p95/p99, SLO 실패 분해(TTFT 3 s, token-ITL p95 50 ms 기준; `analyze_concurrent.slo_failure_breakdown`)를 기술치로 기록한다.

## 5. M4 — 변화 trace, 2차 estimand (Job 3, M3 결과 뒤)

M4는 M3 결과를 입력으로 하는 다음 결정적 규칙만 등록한다. M4 Job은 M3 결과 감사 뒤 별도로 승인받는다.
- **대상**: G(N)이 정의된 (mode, N)만.
- **정규화**: offered rate = ρ × λ\*\_fused(config, A). ρ\_LO = 0.30, ρ\_HI = 0.85(두 phase 모두 fused 용량 아래 — 과부하 비교 회피, C-3).
- **trace**: LO 120 s → HI 120 s를 3라운드. phase 길이는 **초 단위로 고정**한다(phase당 요청 수는 λ\* 비례). 도착은 Poisson, seed 4개(§4-2와 같은 seed), 각 seed가 1 rep. arm은 `fused`·`fused_sym`·`agnostic`·`d44`, 같은 Job.
- **SLO**: TTFT ≤ 3 s ∧ 요청 내부 token-ITL p95 ≤ 50 ms(게이트 4). duration은 라운드 합산, DP 복제본은 공유 벽시계(`analyze_concurrent`).
- **보고량**: 같은 offered rate에서 arm 간 good 비율 차이(seed 짝 bootstrap CI). goodput/GPU의 N-추세는 e(N)과 분리해서 보고한다. 3% 미만은 headline이 아니다. 셀마다 SLO 실패 분해를 붙인다.
- M4에서 `agnostic > fused`가 나와도 그것을 "multi-GPU 고유 기전"으로 귀속하지 않는다. 기전 귀속은 §4-6 nsys 기술치와 별도 기전 arm 등록이 필요하다.

## 6. 예산·하드캡·중단

### 6-1. 규모(추정, 하네스 dry-run이 확정)
- **追記(2026-10-12, 하네스층 감사 `VERDICT_harness_rev6_2026-10-12.md` §5)**: 아래 rev1 문장의 "셀 ≈290개"는 산술 오류다. rev5의 4 arm 기준 본 셀 = 7 config × 4 arm × 2 shape × (4 HI + 2 LO) = 336, 기술 블록 29(GPU 편차 8 + NUMA 8 + nsys 1 + 앵커 12)를 더해 **365개**, boot 70회다(`mg_plan.py m3-plan`). 판정 규칙은 바뀌지 않는다.
- M2g: 서빙 boot ≈ 6 config × 4 arm + 대조 + 거부 시험 ≈ 25회, G3 마이크로벤치. 벽시계 추정 3–5 h.
- M3: boot 7 config × 4 arm × 2(HI·LO 분리, §4-2) + 앵커 + GPU 편차 8 + NUMA 2 + nsys 1 ≈ 70회. 셀 = (7×4×2 shape × (4 HI + 2 LO)) + 기술 블록 ≈ 290개. 벽시계 추정 15–25 h.
- rev5 감사 예보: λ = prior 가정에서 M3 ≈ 28–31 h, TP에서 λ < prior면 ≈ 31–35 h — 확장 없이도 하드캡 30 h를 넘을 가능성이 높다. 블록 분할 시 §4-5 앵커(fused·fused\_sym·agnostic)가 발동한다.
- **하네스 dry-run 예산이 하드캡 30 h를 넘으면 제출 전에 사용자에게 블록 분할을 묻는다.**
- 추정치는 1-GPU 경험(셀당 3–5 분, boot 3–5 분)에서 외삽한 것이라 신뢰구간이 넓다. TP=8 boot·캡처 시간은 미실측이다.

### 6-2. 하드캡
- 각 Job 스크립트 내부 벽시계 집행기를 둔다(λ0 VESSL 방식). M2g 8 h, M3 30 h. 초과하면 진행 중 셀을 끝내지 않고 중단하고, 잘린 셀은 `TRUNCATED`로 남긴다(판정에서 제외).
- 사용자 결정(C-9)으로 운영 모델의 Job당 GPU-h 한도는 적용하지 않는다. 다만 M3를 블록으로 나눌지는 하네스 dry-run 규모를 보고 사용자에게 다시 묻는다.

### 6-3. 중단 코드(GPU 작업 전)
`ABORT_SUBSTRATE`, `ABORT_MODEL_REVISION`, `ABORT_ENGINE_DIGEST`(하네스 봉인 digest 불일치), `ABORT_COMM_ENV`(JIT AR 설정), `ABORT_M2G_NOT_PASSED`(M3 Job이 `M2G_PASS_TABLE.json`을 못 찾거나 digest 불일치 — Job 전체).

## 7. 금지 서술 (이 등록 아래의 결과 문서 전부)

설계 문서 §8 전부를 승계한다. 추가:
- `KNEE_NOT_BRACKETED`·`HARNESS_LIMITED`·`TRUNCATED` 셀의 λ\*를 크기로 인용하는 것.
- `PARTITION_NOT_EXERCISED` G(N)을 "SM 분할 효과"로 서술하는 것.
- `NO_TREND_DETECTED`를 "N에 무관하다/일정하다"로 쓰는 것.
- P-DP가 깨졌을 때 원인 배제 전에 TP 결과를 PD-mux 스케일링으로 쓰는 것.
- d44 결과를 "N별 최적 static"으로 쓰는 것(N마다 재조율하지 않았다).
- λ\*(N)을 1-GPU λ\*\_VESSL·KISTI λ\*와 풀링하거나 그 값으로 빈 셀을 채우는 것.
- M2g PASS를 "TP PD-mux 정확성 일반"으로 쓰는 것.
- rev1 감사 caveat MGP-0…MGP-16(`VERDICT_prereg_rules_rev1_2026-10-12.md` §7) 전부 — 문자 그대로 승계. 특히 MGP-15(d44는 비조율 arm, 설계 §5-3의 N별 선택에서 이탈)와 MGP-16(x8의 G(1)은 정본 P1.7의 재검정이 아님).
- 원시 ratio(achieved/offered)를 포화·비포화 근거로 쓰는 것(MGP-8).
- rev5 델타 감사 caveat MGP-26′·MGP-27′(수치 갱신본이 MGP-26·27을 대체)·MGP-33…MGP-39(`VERDICT_prereg_rules_rev5_delta_2026-10-12.md` §5) 전부 — 문자 그대로 승계.
- rev3 델타 감사 caveat MGP-26…MGP-32(`VERDICT_prereg_rules_rev3_delta_2026-10-12.md` §5) 전부 — 문자 그대로 승계. 특히 MGP-29: fused arm은 overlap ON·chunked 8192, PD-mux arm은 overlap OFF·chunked −1이므로 G(N)을 'SM 분할 효과'·'PD 분리 효과'로 귀속하지 않는다.
- rev2 델타 감사 caveat MGP-17…MGP-25(`VERDICT_prereg_rules_rev2_delta_2026-10-12.md` §5) 전부 — 문자 그대로 승계.

## 8. 감사자가 봐야 할 자유 표면 (자기 공시)

1. **λ\* 정의**: 포화 처리율(HI rung achieved 중앙값). λ0의 "포화 rung achieved 최댓값"과 다르다. 최댓값은 seed 4개에서 위쪽으로 치우치므로 중앙값을 골랐다. 반전 시험 대상이다.
2. **HI 배수 1.6, LO 배수 0.60, 확장 1회 ×2**: cap 구속 포화(λ0 A: #queue 221)에서 G가 cap 비율로 수렴할 위험. TP cap 48·N vs DP 48/복제본은 같은 N에서 총 cap이 같다(48N). fused와 PD-mux arm은 같은 cap이다. 그래도 cap 구속이면 G → 1로 퇴화할 수 있다(N3 계열 퇴화). E-time gate가 이를 표기하지만 막지는 못한다.
3. **분할 실현 gate 문턱 0.5**와 rank 0 / 복제본 평균 규약.
4. **G1 −0.05, G2 4배·0.05 nat·2%, G3 R=500, G5 TV 0.05**: 근거가 1-GPU 실측이 아닌 손 선택이다. 반전 시험 대상이다.
5. **seed = boot 공유**: seed 반복이 같은 boot 안이라 n=4가 boot 산포를 포함하지 않는다. λ0 VESSL 분산 블록은 boot마다 seed를 재었다.
6. **agnostic manual 등가표**: 선택 함수 등가성은 코드로 유도했고(§2), G5(a)가 실측 검증한다. 다만 auto config는 TP/DP>1에서 거부되므로 N>1에서는 등가성을 직접 확인할 수 없다(N=1로만 확인).
7. **arm 순서 무작위화**는 config 안에서만 한다. config 순서는 고정이라 시간 드리프트가 N과 공변한다. 앵커 2회(§4-5)로 기술만 한다.
8. **shape B의 의미**: B는 prefill 지배(λ0 B: #running 2). PD-mux 분할이 decode를 보호하는 상황이 아니다. B의 G(N)은 "prefill-heavy 대조"로 서술한다.
9. **SLO 수치(3 s, 50 ms)**는 기술치와 M4에만 쓴다. M3 판정은 SLO에 의존하지 않는다.
10. **DP=N 총 cap 48N과 TP=N 총 cap 48N이 같다**는 점이 mode 비교를 공정하게 만든다는 주장은 하지 않는다. KV/mamba 용량은 다르다(C-5 기록).

## 9. rev1 → rev2 대조표

| 감사 항목 | 반영 |
|---|---|
| D1 LO 브래킷 drain 항등식, HI ratio ≤ 0.90 승계 누락 | §4-3 독립 포화/비포화 증거(X 0.80, Y 0.90, 샘플 정의·최소 20줄), ratio는 realized 분모 기술치, §4-2 LO 창·요청 수 등록 |
| D2 E-time 추정량 모순 | §4-3 E-time 문자 정의(분자 = 분할 group 전체), `MIX_FAILED`, idx==2 변이 검사 |
| MGP-1 FAIL 단위·NOT_TESTED·G0(ii) 5종·G3 정확도 | §3 표, §3-1, `M2G_PASS_TABLE.json` |
| MGP-10 G(1) 회차·재표집 단위·크기 분모·드리프트 | §4-4, §4-5 |
| MGP-2 노출 비율·불일치 prompt 비율 | §3 G1 |
| MGP-5 G5(a) 결정적 CPU 시험 | §3 G5(a) |
| MGP-9 공동 확장·공동 x\_lo | §4-2 |
| MGP-11 프론트엔드 지표 | §4-3 `FRONTEND_SATURATED` |
| MGP-12 arm 간 메모리 동등성 | §3 G5(c), §4-4 |
| MGP-14 하드캡 절단 순서 | §4-1 실행 순서 |
| MGP-15·16 | §7 |
| 예산 대조 | §6-1 |

## 10. rev2 → rev3 대조표

| 감사 항목 | 반영 |
|---|---|
| D3 HI 술어가 입장 기계를 잼(Prefill 줄 혼입, running==cap 구멍) | §4-3 g 통계(Decode 줄만, arm 대칭), running==cap은 기술치, 변이 3종 |
| D4 LO 술어 arm 비대칭 표본 | §4-3 LO도 g ≤ 0.05 |
| decode_log_interval·타임스탬프 해상도 미등록 | §1 서버 로그 행(40, `SGLANG_LOG_MS=1`) |
| D5 G5(c) 정확 일치 | §3 G5(c) 3조건 + 1% 허용, 가시성 프로브·NVML 폴백 |
| D6 `ps %cpu` | §4-3 `/proc` 1 s 델타 |
| MGP-17…25 | §7 승계 |

~~g 문턱 0.10/0.05는 감사자 모형 예보를 등록 상수로 승격한 것이 아니라 저자의 유체 근사로 정했다~~ — **rev4 정정(MGP-28)**: g의 정의와 문턱 0.10/0.05는 rev2 델타 감사 처방 P-D3/D4를 문자 그대로 채택한 것이다. 유체 근사 g ≈ 1 − λ\*/x는 사후 정당화이며 0.05를 도출하지 않는다.

## 11. rev3 → rev4 대조표

| 감사 항목 | 반영 |
|---|---|
| D7 fused prefill 우선 → shape B Decode 줄 고갈, 확장 결정적 발화·예산 절단 | §4-3 대기열 원천을 클라이언트 Q\_c(시간 적분 평균)로, §4-2 확장 트리거 = g < 0.10만, 변이 검사 (a)–(d) 갱신, 예산은 하네스 dry-run에서 재예보 |
| D8 NVML memory.used 1% 폴백 | §3 G5(c) 오염 검사 (가)(나), arm 간 memory.used는 기술치 |
| MGP-28 출처 정정 | §10 문단 취소선 + 정정, §4-3 문턱 출처 명시 |
| MGP-31 throttle 분모 | §4-3 `HARNESS_LIMITED` = nr\_throttled/nr\_periods 증가분 |
| MGP-26, 27, 29, 30, 32 | §4-3·§7 승계 |

## 12. rev4 → rev5 대조표 (사용자 결정, 감사 처방 아님)

| 변경 | 위치 | 이유 |
|---|---|---|
| `fused_sym` arm 추가(vanilla + `--chunked-prefill-size -1 --disable-overlap-schedule`) | §2, §3 G0, §4-1, §5, §6-1 | MGP-29: fused(overlap ON·chunked 8192)와 PD-mux(overlap OFF·chunked −1)의 노브 비대칭 |
| 1차 estimand G = agnostic/fused\_sym, G\_def·G\_d44,sym은 보조 라벨, S = fused\_sym/fused 기술치 | §4-4 | 노브를 맞춘 비가 PD-mux 기능 귀속에 더 가깝다. 단 SM 분할 **단독** 귀속은 여전히 아니다(split prefill·입장 기계 포함) |
| x\_lo 기준은 그대로 fused(운영 기본값) HI 중앙값 | §4-2 | 변경 없음 — 네 arm 공동 |

## 13. rev5 → rev6 대조표 (rev5 델타 감사 §8-1 경량 반영)

| 항목 | 반영 |
|---|---|
| H2 health 대기, MGP-23 | §3 G0(i) |
| H4 attention backend, H5 sock_read, H6 fd, H8 piecewise 기록 | §1 고정 조건 3행 |
| MGP-38 fused\_sym 기동 실패 행, 앵커 | §3-1, §4-5 |
| G5(c) 네 arm, §4-4 CI 문장 | §3, §4-4 |
| §6-1 예산 재예보 | §6-1 |
| MGP-26′·27′·33…39 승계 | §7 |
| 선택 처방(arm별 x\_lo) | **채택 안 함**(공동 x\_lo 유지, MGP-34 병기로 처리). 바꾸려면 사용자 결정 + 델타 감사 |

## 14. rev6 → rev7 대조표 (사용자 결정, M2g 2회차 결과 대응)

| 변경 | 위치 | 이유 |
|---|---|---|
| fused = vanilla + `--disable-piecewise-cuda-graph` | §1 piecewise 행, §2 fused 행 | M2g 2회차: piecewise 캡처가 py3.14 `typing.Union.__module__` 비호환으로 fused 7/7 기동 실패 → PASS 표 28칸 전부 제외 |
| MGP-36의 piecewise 노브 | 해당 없음 처리 | fused와 fused\_sym 모두 piecewise 미사용. S = λ\*\_fused\_sym/λ\*\_fused는 overlap·chunked 두 노브의 합성(여전히 'overlap 효과' 단독으로 쓰지 않음) |
| M2g 재실행 | — | 같은 Job 원칙. 2회차 PASS(G3, G0(ii)/(iii), G3b, G5(b))는 재사용하지 않고 다시 잰다 |
| G4 공허 PASS(boot 실패 arm) | 하네스 수정 | 판정 규칙 변경 아님 — boot status ≠ OK면 G4 = NOT_TESTED |

## 15. rev8 — 구성별 적정 크기 Job (사용자 결정 2026-10-12)

### 15-1. Job 배치
| Job | 스펙 | 담는 config | 앵커 |
|---|---|---|---|
| J1 | `resourcespec-a100x1` | N=1 | (자기 자신이 기준) |
| J2 | `resourcespec-a100x2` | TP2, DP2 | N=1 앵커(GPU 0) |
| J4 | `resourcespec-a100x4` | TP4, DP4 | N=1 앵커(GPU 0) |
| J8 | `resourcespec-a100x8` | TP8, DP8 | N=1 앵커(GPU 0) |

- 네 Job은 서로 다른 노드에서 **동시에** 실행해도 된다(벽시계 단축). M2g와 M3 각각 이 4-Job 배치를 쓴다.
- 한 config의 네 arm은 **반드시 같은 Job** 안에 둔다(1차 estimand G\_sym은 같은 Job 안 arm 비율이라 Job 공통 배율 효과가 상쇄된다).
- Job 안 순서: preflight → (M2g) G5(a) CPU·G3·거부 boot / (M3) 앵커 → config(TP 먼저, DP 다음; 순서는 Job마다 seed 20261012로 무작위) → 끝. N=1 2회차 앵커와 NUMA 블록은 **없앤다**.
- 부분집합: J\_N 안에서 TP·DP는 노드의 GPU 0..N−1 전부를 쓴다. 노드 위상(`nvidia-smi topo -m`)을 preflight에서 기록하고, Job 안 GPU 쌍 중 NVLink가 아닌 쌍이 있으면 `TOPOLOGY_NOT_NVLINK` 플래그를 그 Job 결과 전체에 붙인다(차단 아님; M1 프로브는 x8 노드만 확인했다).

### 15-2. M2g 배분
- J\_N에 그 N의 G3(TP=N), G0(i) 4 arm×{TP,DP}, G1·G2·G3b·G4·G5(b)(c)를 둔다. **G1-ctl과 G2 바닥(TP=1 2회)은 각 Job의 N=1 앵커 boot에서 그 Job 안에서 잰다**(같은 Job 원칙 유지).
- G0(ii) 금지 env 5종과 G0(iii) 자동 격자 거부(TP=2 필요)는 J2에 둔다. G0(ii) 실패 시 TP 축 전체 제외 규칙(§3-1)은 J2 결과로 J4·J8의 TP에도 적용한다.
- G5(a) CPU 시험은 Job마다 실행한다.
- PASS 표는 Job마다 하나(`M2G_PASS_TABLE_J{N}.json` + `.sha256`), 커밋 뒤 M3 J\_N이 자기 N의 표와 J2 표(G0(ii) 결과)를 3자 대조로 읽는다(F17·R5 규칙을 Job별로 적용).

### 15-3. M3 배분과 Job 효과 처리
- J\_N(N≥2)은 config 측정 전에 N=1 앵커를 잰다: GPU 0 단독, arm {fused\_sym, agnostic}, shape A HI(seed 4개) — G\_sym,anchor(J\_N) 산출.
- J1의 N=1이 기준 G\_sym(1)이다.
- **Job 효과 검사**: |G\_sym,anchor(J\_N)/G\_sym(1) − 1| > 3%이거나 앵커 λ\*(fused\_sym, A)가 J1과 3% 넘게 다르면 `JOB_EFFECT(J_N)` 플래그. 추세 라벨에 플래그가 하나라도 붙으면 (i) 원 추세와 (ii) 앵커 보정 추세(G\_sym(N)·G\_sym(1)/G\_sym,anchor(J\_N))를 **둘 다** 보고하고, 둘의 라벨이 다르면 `TREND_JOB_SENSITIVE`로 방향 라벨을 인용하지 않는다.
- 앵커가 측정되지 않은 Job(입장 거부·기동 실패)은 `JOB_EFFECT_UNTESTED`이며 그 N의 점은 추세에 들어가되 공시를 의무로 붙인다.
- 기술 블록: GPU 편차 블록은 J8에서 GPU 0..7에 **동시에** 8개 N=1 fused HI(seed 4386)로 잰다(동시 부하 하 기술치로 명시). NUMA 블록 폐지. nsys 블록은 이미지에 nsys가 없으므로(MGH-24) `NOT_RUN`.

### 15-4. 하드캡·예산(예보, 하네스 dry-run이 확정)
- Job별 하드캡: M2g J\_N 각 4 h, M3 J1 8 h·J2/J4/J8 각 14 h.
- 예약 대비 사용률: config 측정 중 100%, 앵커 구간만 1/N. M3 예보 총 약 120 GPU-h(이전 x8 단일 Job 약 240 GPU-h 예약), 벽시계 최장 Job 약 8–9 h.
- 하드캡에 걸린 Job의 잘린 config는 `TRUNCATED`, 다른 Job에 영향 없음.

### 15-5. 이 변경이 바꾸는 caveat
- C-1(노드 부하·부분집합 위상)은 "Job마다 노드가 다르다"로 바뀐다 — N 간 비교에는 Job 효과가 들어가며 §15-3 앵커로만 검사된다. 교훈 274(Job 효과)를 정면으로 받는 설계다.
- MGH-6·MGH-20(앵커 절단)·MGH-7(30 h 단일 Job에서 추세 미정의 예보)은 해당 없음으로 바뀐다(Job별 하드캡). MGP-14(N=8 우선 절단)도 해당 없음.
- 새 금지 서술: "N 간 G 차이는 N 효과다"(Job 효과 검사·앵커 보정 병기 전), "앵커 보정 추세가 원 추세보다 정확하다"(보정은 G 배율 모형 가정).

## 16. rev9 (2026-10-13)

### 16-1. D9 처방
- **P9-a**: G\_sym(1, shape)이 어떤 이유로든 미정의이면 그 shape의 **모든 N-추세 방향 라벨은 인용 불가**(`G1_UNDEFINED_NO_JOB_CHECK`), 모든 J\_N에 `JOB_EFFECT_UNTESTED`, P-DP는 `P_DP_UNTESTED`. §4-4의 "G(1) 미정의면 방향 라벨만(MAGNITUDE_UNDEFINED)" 문장은 rev9에서 **폐기**한다.
- **P9-b**: `JOB_EFFECT`가 하나라도 발화하면 원 추세와 앵커 보정 추세가 **둘 다** 크기 규칙(|G(N_max)/G(1)−1| ≥ 3%)을 통과하고 방향 라벨이 같을 때만 방향을 인용한다. 아니면 `TREND_JOB_SENSITIVE`.
- **P9-c**: DP 추세가 `TREND_JOB_SENSITIVE`·`TREND_UNDEFINED`·G(1) 미정의 중 하나이면 P-DP = `P_DP_UNTESTED`. 이때 MGP-13은 발화하지 않고, 모든 TP 추세에 "P-DP 미검정" 공시를 의무로 붙인다(MGH-8과 같은 처리).
- **P9-d**: J1 하드캡 10 h.
- Job 효과 검사의 기준값은 **J1 boot1(확장 전 1.6× rung)**의 G\_sym(A)·λ\*(fused\_sym, A). 앵커 셀이 `HI_SATURATED`가 아니거나 `HARNESS_LIMITED`·`CLIENT_ERROR`이면 그 Job은 `JOB_EFFECT_UNTESTED`. 보정 추세의 bootstrap은 앵커 seed 색인을 같은 색인으로 함께 재표집한다.

### 16-2. 사용자 결정 (2026-10-13)
1. **메모리 규칙 (iii) 폐지 — 사후 변경**: §3 G5(c)·§4-4의 `MEMORY_MISMATCH` 조건 중 (iii) "arm 간 `max_total_num_tokens` 상대차 > 1%"를 삭제한다. (i) mamba 슬롯 ≠ effective cap, (ii) `max_total_num_tokens` < effective cap × (ISL+OSL)(KV 구속)만 남는다. **공시**: 이 변경은 M2g 2회차(`m2g_pdmux-multigpu-c1395d0-20261012081851`)의 `max_total_num_tokens` 값(TP4 1.34–1.38%, TP8 1.29–1.33%, TP2 0.9989%)을 본 뒤에 이루어졌다. 근거는 KV 풀이 구속되지 않는다는 것(TP4 필요 약 1.59 M 토큰 vs 가용 약 13.7 M)이다. 결과 문서는 이 사후 변경과 (iii) 기준이었다면 미정의였을 셀을 함께 적는다. arm 간 상대차는 기술치로 계속 기록한다.
2. **하드캡**: J1 10 h, J2·J4·J8 각 14 h(예상치 기준). 공동 확장(S3)이 발화하면 DP2·DP8 LO 등이 잘릴 수 있다(MGP-49) — 잘린 셀은 `TRUNCATED`로 공시.
3. **J1 스펙**: `resourcespec-a100x2`에서 GPU 0만 사용(GPU 1 유휴). CPU quota·노드 종류를 J2와 맞추기 위한 것. J1의 GPU 1은 측정에 쓰지 않는다.

### 16-3. 하네스 조건 (rev8 감사 §6 승계)
- config 순서: Job마다 TP·DP 순서를 seed 20261012로 무작위화하고 실행 로그에 기록한다(§15-1의 "TP 먼저" 문장은 폐기).
- 기판 gate: `--expected-gpus` = 그 Job의 스펙 GPU 수(J1은 2), GPU 이름이 정확히 "NVIDIA A100-SXM4-80GB", power limit 400 W, Job 안 GPU가 2장 이상이면 전 쌍 NV#. 하나라도 어긋나면 GPU 작업 전 `ABORT_SUBSTRATE`.
- Job별 `cpu.max`와 셀별 throttle 비율을 공변량으로 결과표에 병기.
- `PREREG_REV = 9`, Job별 `HARD_CAP_S`·`EXPECTED_GPUS`, 계획에서 anchor2·numa·nsys 제거, 앵커 boot 2개(fused\_sym, agnostic) 추가, GPU 편차 블록은 J8 안 맨 앞(분기 없음, 기술치).
- 변이 KILLED: (i) G(1) 미정의인데 방향 인용 (ii) 확장 rung 값과 앵커 비교 (iii) 다른 Job 앵커로 보정 (iv) 보정 추세에서 크기 규칙 제거 (v) gate 이름 부분 일치.
- M3 J\_N은 자기 PASS 표와 J2 표를 3자 대조로 읽는다. J2 표가 없으면 TP는 `NOT_TESTED`, DP만 진행(Job 전체 ABORT 아님).
- M2g의 Job별 N=1 앵커 boot에는 G1-ctl용 fused·agnostic·d44 N=1 arm을 모두 포함한다.
