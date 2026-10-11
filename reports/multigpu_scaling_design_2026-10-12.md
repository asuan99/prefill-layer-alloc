# Multi-GPU 확장(A100 ×1/2/4/8, DP·TP) 설계 검토 — 단일 A100 결과에서 넘어갈 때 추가로 고려할 것 (2026-10-12, rev2)

> **지위**: 설계 **검토 문서**(병행 트랙 제안, M0). **측정 아님** — GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0 ·
> 엔진 코드 변경 0. 실행 순서와 시작 시점은 사용자 결정이다(현재 N2(3-a) 사다리와 GPU 예산이 겹친다).
> **사용자 결정(2026-10-11)**: 확장 축 = **DP 복제와 TP=N 둘 다**, 관측 목적 = **(a) PD-mux 이득의 N-스케일링과
> (b) 멀티 GPU에서만 생기는 새 기전**을 함께 본다.
> **rev2**: rev1 규칙층 감사(`reports/audit/2026-10-12_multigpu_design/AUDIT_multigpu_design_rev1_2026-10-12.md`, **`MIXED`**:
> M1·M2·M2g = `GO-with-caveats`, M3·M4 = `NO-GO`[死因 F-1: estimand 부재·정규화 항등식])를 반영했다. 대조표는 §10.
> **MG-0(문자 그대로 승계)**: "이 설계 문서(rev1)의 규칙으로는 M3·M4를 실행할 수 없다. M3 이전에 PREREG_multigpu와 그 규칙층
> 감사가 선행해야 한다. `GO-with-caveats`는 M1·M2·M2g에만 해당한다." rev2도 M3·M4 규칙의 **방향**만 적는다.
> 등록 규칙은 M2.5 PREREG에서 정하고 감사받는다. 이 판정을 "multi-GPU 트랙 GO"로 요약하지 않는다.
> 줄 번호는 2026-10-12 HEAD(`97a91b5`) 기준이다. 경로 접두 `EP/` = `workspace/engine-port/`, `DEV/` = dev tree
> `../sglang_engine_dev/python/sglang/`이다.

## 0. 요약

1. **저장소에는 multi-GPU 실행 경로가 없다.** VESSL 스펙은 `resourcespec-a100x1`뿐이다(`EP/scripts/vessl/vessl.env:7`).
   `--tp-size`·`--dp-size`·`--gres=gpu:[2-9]`는 저장소 전체에 0건이고(`EP/scripts/r2_eval/engine_bench_runner.sh:85-128`는 기본값
   tp=1·dp=1), TP>1에서 PD-mux를 돌린 적이 없다. 따라서 이 트랙은 **새 기판 + 새 엔진 경로 + 새 사전등록**이다.
2. **1-GPU 결론은 N-GPU로 이식하지 않는다**(게이트 1의 연장). λ\*, C2, HE0, decode-heavy static 권고, cap48 아래의 용량값은
   N·mode마다 다시 잰다. 이전되는 것은 정의와 규율뿐이다(§7).
3. **Job 효과가 N 효과를 가린다.** 교훈 274는 Job(GPU·노드·시간) 묶음이 36/36 셀을 한 방향으로 움직인 사례다. GPU
   단독 귀속은 확립되지 않았다. → **8-GPU 노드 하나에서 N=1/2/4/8을 같은 Job 안의 `CUDA_VISIBLE_DEVICES` 부분집합으로
   돌린다.** 이렇게 해도 노드 부하·부분집합 위상이 N과 함께 변한다(C-1). Job 크기 규약과도 충돌한다(C-9, 사용자 결정).
4. **(a)의 판정량은 arm별 용량비 G(N, mode) = λ\*\_agnostic / λ\*\_fused의 N-추세다**(P-F1). 1-GPU λ\*로 정규화한
   goodput/GPU의 N-추세는 M3가 낸 λ\* 스케일링의 재표현이라(F-1 항등식) 판정량이 될 수 없다.
5. **(b)의 1순위 후보**: green-context 파티션 안에서 통신 커널(custom AR v1의 고정 상한 36 CTA, NCCL 채널 CTA)이 차지하는
   SM 비율, 그리고 PD-mux가 만든 두 벌의 TP comm(prefill/decode)이 동시에 실행되는 것. **둘 다 동작 자체가 아직 검증되지 않았다.**
   correctness gate(M2g)가 성능 측정보다 먼저다.
6. **DP의 새 교락**: 라우팅, 단일 프론트엔드(TokenizerManager·클라이언트) 병목, 동시 실행 집계(§4-2).

## 1. 기판·인프라

| 항목 | 단일 A100에서의 상태 | N>1에서의 문제 | 처리 |
|---|---|---|---|
| 노드 형상 | `resourcespec-a100x1` | **2026-10-12 `vesslctl resource-spec list` 조회**: `resourcespec-a100x{2,4,8}` 존재·USABLE, 전부 cluster-betelgeuse, CPU는 GPU당 11(x8 = 88), **x8 가용성 `low`**(스케줄 지연·실패 가능). NVLink/NVSwitch vs PCIe는 스펙 목록으로 알 수 없음(`reports/amd_design_discussion_2026-10-11.md:153`) | M1 프로브(`EP/scripts/vessl/multigpu_probe.sh` 초안, 미커밋): `topo -m`, nvlink, P2P, all-reduce 대역폭 |
| Job 효과 | 교훈 274 | N별 Job 분리 시 N 효과 = Job 효과 | x8 한 Job 안 N 부분집합. 블록을 나누면 블록마다 N=1 앵커(C-9) |
| **노드 부하 상관** | 해당 없음 | N=8은 노드 전력·열·NVSwitch·호스트 메모리를 쓰고 N=1은 7장이 유휴 → 노드 클럭·throttle이 N과 공변. 부분집합 동시 실행 시 상호 간섭 | 부분집합은 **순차** 실행. GPU별 클럭·throttle 사유를 셀마다 공변량으로 병기(C-1) |
| **부분집합 위상** | 해당 없음 | N=2/4 GPU 집합이 NVLink 위상·NUMA 소켓·CPU 친화도와 교락 | 부분집합 GPU ID를 사전등록하고 위상 기록(C-1) |
| GPU 사이 편차 | 정의상 없음 | N=1 부분집합 선택이 결과를 바꿈 | M1에서 같은 1-GPU 셀을 8장 각각에서 실행. 편차 문턱과 분기 규칙은 PREREG에 등록(예: >3%면 N=1 회전 의무 + GPU 공변량, C-2) |
| CPU·프론트엔드 | cgroup quota 11 CPU, 가시 96(교훈 268) | scheduler N개 + **단일** TokenizerManager·detokenizer·클라이언트. GIL 병목이 GPU 확장성 한계로 보임(confound #7 재발 경로) | `cpu.max`·`cpu.stat` 매 run 기록, throttled 실격 문턱 사전등록. **M1.5 하네스 용량 프로브**(C-4) |
| 기판 기록 | `EP/scripts/vessl/job_entry.sh:120-172` | `gpu.csv`를 `splitlines()[0]`만 파싱 → x8에서 **GPU0만** 기록. `clk.csv`(`nvidia-smi -lms`)는 전 GPU 행을 섞어 기록 | GPU별 기록으로 확장(M2). `job_entry.sh`의 현재 미커밋 수정이 정리된 뒤 편집 |
| 기판 gate | `EP/scripts/vessl/substrate_probe_helper.py:227-257`, `EP/results/n2_static_grid/n2_plan.py:134,473-498`(정확히 1장·108 SM) | N장에서 실패 | N을 인자로 받는 gate. 1장 동작은 회귀 테스트로 보존 |
| 앵커 | KISTI 908623 ↔ VESSL λ0가 "구별되지 않는다" | x8 노드의 N=1은 또 다른 기판 | **N=1-in-node = 새 앵커.** 기존 결론·수치와 섞지 않는다 |

## 2. 엔진 — TP=N (미검증 경로)

1. **랭크별 결정 분기와 허용 조합.** R2 generic/hybrid(`_r2_decide_idx`, `EP/src/multiplex/multiplexing_mixin.py:626`, 호출 `:1327`)와
   SLO_SCHED(`_slo_decide_idx`, `:1095`)는 벽시계·ITL EWMA를 입력으로 쓴다. 랭크마다 다른 파티션이나 admission
   (`:1221`, `get_new_batch_prefill` 직전)을 고르면 collective가 hang하거나 배치가 어긋날 수 있다. 이 결론은 코드 입력
   분석만으로 성립한다. static arm은 `PDMUX_R2_POLICY=fixed` 경로라(`EP/results/n2_static_grid/n2_plan.py:627-637`,
   `EP/scripts/r2_eval/policy_adapter.sh:33-45`) `_r2_decide_idx`를 탄다. `FixedPolicy.decide`(`EP/src/multiplex/controller.py:88-103`)는
   `safe_boundary`만 쓰는데, legacy 루프에서는 상수 True라 **legacy + fixed는 결정적**이다. **true-dual**에서는
   `arbiter.safe_to_switch()`가 타이밍에 의존하므로 fixed도 발산할 수 있다(`:1966` 합의 경로).
   → **M2 fail-fast 술어(열거형)**: TP>1 **허용** = legacy 루프 ∧ (`PDMUX_R2_POLICY` unset ∨ `fixed`) ∧ `PDMUX_SLO_SCHED` unset ∧
   `PDMUX_LA_COORD` unset ∧ `PDMUX_TRUE_DUAL_WORKER` unset. 그 밖은 전부 거부한다. sticky + fixed는 M2g에서 따로 확인한다.
   각 금지 조합을 허용으로 바꾼 변이본에서 테스트가 실패해야 한다(교훈 53). 동적 arm은 TP 축에서 제외한다.
2. **TP all-reduce 경로(R-1 정정).** v0.5.10의 CUDA 기본은 sgl_kernel **v1 custom all-reduce**다
   (`DEV/srt/distributed/device_communicators/custom_all_reduce.py:339-356`). 그리드 상한은 `kMaxBlocks = 36`으로 **SM 수나
   파티션에 의존하지 않는 고정 상한**이다(`sgl-kernel/csrc/allreduce/custom_all_reduce.cuh:20,417,444`). 크기가 `max_size`를
   넘거나 (world_size≠2 ∧ full NVLink 아님)이면 NCCL로 간다(`DEV/srt/distributed/parallel_state.py:585-619`). 그래서 prefill의 큰
   all-reduce는 NCCL, decode의 작은 all-reduce는 CA v1로 갈 가능성이 높다(실측 전 가설). `NUM_CTA = multi_processor_count`
   (`DEV/jit_kernel/all_reduce.py:129`)는 `SGLANG_USE_JIT_ALL_REDUCE=1` **opt-in 경로에서만** 쓰인다.
   (인용원 `handoff-report/ssm_kernel_sm_partition_2026-09-21.md:243`의 "TP>1이면 발화" 서술은 기본 설정에서 거짓이라
   doc-steward 정정 대상으로 회부한다.)
   **살아 있는 (b) 가설**: green ctx 파티션(예: decode 16–44 SM) 안에서 CA v1의 최대 36 CTA와 NCCL 채널 CTA가 차지하는 SM 비율.
   기전 arm = custom AR on/off(`--disable-custom-all-reduce`). 단 **PCIe 노드의 N≥4에서는 full NVLink가 아니라 CA가 원래 꺼지므로
   이 arm은 무처치**다. 위상 결과(M1)로 arm의 존재 여부를 셀마다 확정한다(C-10).
3. **두 벌의 TP comm 동시 실행.** PD-mux를 켜면 `duplicate_tp_group=enable_pdmux`(`DEV/srt/model_executor/model_runner.py:972`)가
   `_PDMUX_PREFILL_TP_GROUP`(`DEV/srt/distributed/parallel_state.py:1806-1823`)을 만든다. 두 그룹 모두 custom AR comm과 pynccl comm을
   갖는다. 그래서 prefill과 decode의 collective가 서로 다른 파티션에서 동시에 돈다. `tp_cpu_group.allreduce(flags) == tp_size` 합의
   (`multiplexing_mixin.py:1525-1526`, `:1966`)는 tp=1에서도 매번 호출되지만 **비만장일치 결과 분기는 실행된 적이 없다**
   (`reports/impl_vs_external_pdmux_2026-08-28.md:157`). 교착 위험이 있다. stream group별 cudagraph capture
   (`DEV/srt/model_executor/cuda_graph_runner.py:806-814`)와 NCCL의 상호작용도 검증한다. 같은 블록의 `server_args.py:6138`은
   green ctx + cudagraph 저하 가능성(torch ≥2.7)을 경고할 뿐, TP/DP 금지 assert는 없다.
4. **모델 분할 가능성.** attention head, KV head, mamba head 수가 TP=2/4/8로 나누어떨어지는지와 SGLang v0.5.10의 해당 모델 TP 지원은
   모델 config로 확인한다(이 문서는 추정하지 않는다). TP 축의 주 모델은 **Nemotron-Nano-9B-v2**다(Nemotron 우선). 나누어떨어지지
   않는 (모델, N)은 셀에서 빼고 그 사실을 기록한다.
5. **capability 조회(R-4).** `DEV/srt/multiplex/pdmux_context.py:109`(`current_device()`)와 `:120`(`get_device_capability`)은
   **`divide_sm`(자동 격자) 분기에서만** 결과에 영향을 준다. `manual_divisions` 캠페인과는 무관하다. `gpu_id`와의 일치를 검사하는 assert는
   자동 격자 config(`pdmux_a100_smoke.yml`)를 N-GPU에 쓸 때 결과를 지킨다.

## 3. 엔진 — DP=N

1. 파티션은 프로세스·디바이스 단위로 만들어진다(`initialize_stream_groups(self.gpu_id, …)`, `multiplexing_mixin.py:155`).
   SGLang `--dp-size` worker마다 PD-mux가 초기화되는지는 M2g에서 확인한다.
2. **런처는 하나로 고정한다(C-7).** `--dp-size`(단일 프론트엔드)와 N개 독립 서버(N개 프론트엔드 + 외부 라우터)는 같은 조건이
   아니다. 한 캠페인 안에서 섞지 않는다. N=1도 같은 런처로 잰다. `--dp-size`가 M2g를 통과하면 그것을 쓰고, 실패하면 캠페인
   전체를 독립 서버 방식으로 바꾼다.
3. **라우팅**: 주 실험은 round_robin으로 고정한다. shortest_queue는 민감도 arm이다. 라우팅 효과를 PD-mux 스케일링으로 귀속하지 않는다.
4. **복제본 기록**: 복제본별 큐 길이, **realized** 파티션(교훈 89·246), goodput(분모는 §4-2).

## 4. 관측·분석 하네스

1. **Telemetry 충돌.** `PDMUX_TELEMETRY_PATH`는 rank 접미사 없이 append된다(`EP/src/multiplex/telemetry.py:126`). green readout은
   `"w"`라 마지막 rank가 덮어쓴다(`multiplexing_mixin.py:351`). → `.tp{rank}`/`.dp{replica}` 접미사와 rank 필드를 추가한다.
   선례는 `holb_probe.py:564-567`, `chunk_probe.py:987-990`이다.
2. **동시 실행 duration 규칙.** 게이트 7(라운드를 합칠 때 duration 합산)은 **순차 라운드**의 규칙이며 그대로 유지한다.
   여기에 다음을 **정의로** 추가한다.
   - **공유 벽시계** = 라운드마다 클라이언트 기준 단일 구간(그 라운드의 첫 요청 전송 시각부터 마지막 요청 완료 시각까지).
     라운드 사이는 계속 합산한다.
   - 집계 goodput과 **복제본별 goodput 모두** 이 공유 벽시계를 분모로 쓴다. 복제본이 자기 구간(자기 첫 요청~마지막 완료)을
     분모로 쓴 뒤 합산하는 방식은 금지한다.
   - 적용 지점: N-독립-서버 런처와 복제본 분해 분석. `--dp-size` 단일 서버에서는 클라이언트 구간이 원래 하나다.
   - 회귀 테스트: (i) 동시 복제본 duration을 합산하는 변이본, (ii) 복제본 자기 구간을 분모로 쓰는 변이본, (iii) 순차 라운드를
     합산하지 않는 변이본이 각각 KILLED 되어야 한다(교훈 53).
3. **기록 지표**: arm별 λ\*(N, mode), 용량비 G(N, mode), 셀별 SLO 실패 분해(TTFT-fail / ITL-fail / 둘 다, C-3), 복제본 불균형,
   TP 랭크 간 `stream_idx` 일치율, switch_count·split 체류분포(게이트 5) 랭크별, `max_total_tokens`·mamba cache 크기(C-5),
   통신 커널 시간 비율. 통신 비율은 측정 run이 아니라 별도의 짧은 nsys 프로브에서 잰다(관측자 효과).

## 5. "동일 조건"의 정의와 estimand (방향 — 등록은 M2.5 PREREG)

### 5-1. estimand (P-F1)
- **1차(목적 a)**: G(N, mode) = λ\*\_agnostic(N, mode) / λ\*\_fused(N, mode). arm별로 KNEE_BRACKETED를 잡고 같은 Job에서 잰다.
  판정량은 N-추세 dG/dN(paired CI)이다. decode-heavy static은 §5-3 규칙으로 고른 D로 같은 비를 잰다.
  예측 출력공간에 "거짓"을 포함한다(예: "dG/dN의 CI가 0을 포함", "G(N) < 1").
- **2차**: 변화 trace의 goodput 비교(M4). offered rate는 **단일 기준 arm(fused)의 λ\*(N, mode)**로 정규화하고 ρ\_lo·ρ\_hi를 등록한다.
  보고량은 같은 offered rate에서 arm 간 good 비율 차이(paired)다. goodput/GPU의 N-추세는 용량 스케일링 효율
  e(N) = λ\*\_fused(N)/(N·λ\*\_fused(1))과 **분리해서** 보고하고, 둘을 서로의 독립 증거로 쓰지 않는다(F-1).
- **(b) 기전**: CA on/off arm(처치가 존재하는 위상·N에서만), 통신 커널 SM 점유(nsys 프로브), prefill comm과 decode comm의 겹침.

### 5-2. 고정 / 조정 / 결정
| 구분 | 내용 |
|---|---|
| **N과 무관하게 고정** | 모델·revision, seed 규약(`EP/benchmarks/pdmux_eval/campaign.py`의 `seed*1000+rep`), SLO(TTFT·token-ITL p95, 게이트 4), cudagraph ON, **radix cache OFF**(runner의 `--disable-radix-cache` 상속, 2026-09-24 radix OFF 우선 결정과 정합), GPU 1장당 SM 분할, **NCCL·통신 env**(`NCCL_ALGO/PROTO/MAX_NCHANNELS/MIN_NCHANNELS`, `CUDA_DEVICE_MAX_CONNECTIONS`, `SGLANG_USE_JIT_ALL_REDUCE` unset, `SGLANG_CUSTOM_ALLREDUCE_ALGO`, NCCL 버전을 고정하고 기록, C-10), GPU 부분집합 ID |
| **N에 맞춰 조정** | offered rate = ρ·λ\*\_fused(N, mode). HE0 trace의 절대 rate(3/12 req/s)는 쓰지 않는다. 1-GPU 수치로 환산하면 HI phase가 과부하(ρ>1)일 수 있다(C-3, 수치 인용 금지) |
| **등록할 것** | 변화 trace 주기의 단위(초 고정이면 phase당 요청 수가 λ\*(N)에 비례하고, 요청 수 고정이면 phase 시간이 달라진다). 정상상태 길이를 시간으로 둘지 요청 수로 둘지. *(rev1의 "요청 수 NP×N이면 duration이 같다"는 duration ≈ NP/(ρ·λ\*(1)·e(N))이라 e(N)=1일 때만 참이므로 철회)* |
| **사용자 결정(확정: 48·N)** | TP의 `--max-running-requests` = 48·N(§9-4). cap과 별개로 TP는 GPU당 가중치가 1/N이라 `--mem-fraction-static` 아래 KV 풀과 mamba 슬롯이 커진다. DP는 1-GPU와 같다 → 셀마다 기록(C-5) |

### 5-3. arm 정의의 N 의존 (C-6)
- "agnostic"은 decode-bs 임계표다(`multiplexing_mixin.py:1180-1188`). TP에서 decode_bs는 전체 배치(GPU당 일은 1/N)이고,
  DP 복제본의 decode_bs는 ≈1/N이다. 같은 표가 mode에 따라 다른 정책이 된다. 1-GPU 표를 그대로 쓸지 스케일할지는 PREREG에서 정하고,
  그 선택을 결과 서술 범위에 적는다.
- static D는 N·mode마다 **M3 데이터로 등록 규칙에 따라 선택**하고, M4는 별도 seed로 잰다(winner's curse 방지).

## 6. 실험 사다리

| 단계 | 내용 | 실행 | 다음 조건 / NO-GO |
|---|---|---|---|
| **M0** | 설계 문서 + 규칙층 감사 | 로컬 | rev1 `MIXED` → rev2(이 문서) |
| **M1** | VESSL 기판 프로브(Workspace): 스펙 목록, x2/x4/x8 존재, `topo -m`, nvlink, P2P, 2/4/8 all-reduce 대역폭·지연, cgroup quota, GPU별 green ctx 생성, NCCL·통신 env 덤프. GPU별 1-GPU 셀 편차는 PREREG 이후 | 사용자 승인 후 | x8 + 위상 확인 → CA 기전 arm의 존재 여부 확정. x8이 없으면 사용자 결정. AMD 트랙 V0와 같은 프로브로 묶을 수 있다 |
| **M1.5** | 하네스 용량 프로브: 등록 최대 offered rate의 ≥1.5배에서 클라이언트 send-lag·TokenizerManager 큐 지연(더미 백엔드 또는 짧은 Job) | 로컬 또는 짧은 Job | FAIL → 다중 프로세스 클라이언트·`tokenizer_worker_num` 조정 후 재시험 |
| **M2** | 엔진·하네스(engine-porter): telemetry rank 접미사, TP fail-fast 열거형 술어 + 변이 검사, capability assert, 런처(DP·TP), GPU별 기판 기록, N장 기판 gate, `analyze.py` 동시 duration 규칙 + 변이 검사 | 로컬 + CPU 회귀 | 회귀·변이 검사 통과 |
| **M2g** | correctness: **1차 = 같은 TP=N에서 PD-mux OFF↔ON 출력 일치**, 2차 = TP=N↔TP=1 **teacher-forced logit** 허용오차(사전명시, 자유 생성 완전 일치 요구 금지), stream group별 cudagraph capture, prefill·decode comm **역순 실행 스트레스**(반복 횟수 등록), 실제 선택된 all-reduce 커널 식별, `--dp-size` worker별 realized 파티션 | 짧은 Job | FAIL인 (mode, N)은 비운영점으로 기록하고 **M3/M4 정책 주장에서 제외** |
| **M2.5** | **PREREG_multigpu + claims-auditor 규칙층 감사**(estimand §5-1, 등록 항목 §5-2, 편차 문턱, throttled 실격 문턱, static D 선택 규칙, 블록 구성, 예측 출력공간) | 로컬 | 감사 `GO`/`GO-with-caveats` 없이는 M3를 시작하지 않는다 |
| **M3** | arm별 λ\*(N, mode). x8 Job에서 N 부분집합을 순차로 돌리고 블록마다 N=1 앵커 | Job | KNEE_BRACKETED. 클라이언트 한계는 `HARNESS_LIMITED`로 `UNSTABLE`과 구분(교훈 21). `UNSTABLE` 셀은 크기 인용 금지 |
| **M4** | 2차 estimand: 변화 trace × arm{fused, agnostic, static D} × n≥4. 기전 arm(CA on/off, 처치가 있는 셀만), 라우팅 민감도(DP) | Job 블록 | result-analyst → claims-auditor 결과 감사. 3% 미만은 headline 아님 |

**Job 크기 규약 충돌(C-9 — 2026-10-12 사용자 결정: x8에는 한도 미적용, §9-3)**: 운영 모델의 한도는 Job당 ≲3–4 **GPU-h**다
(`handoff-report/vessl_operating_model_2026-10-02.md:20`). x8에서는 벽시계로 ≲0.4–0.5 h이고, TP=8 부팅과 stream group별
cudagraph 캡처를 더하면 블록 하나가 이 한도에 들어가지 않을 수 있다. x8 순차 실행에서는 N=1 셀도 GPU 8장분 GPU-h를 쓴다.
선택지: (i) x8용 Job 한도를 개정한다, (ii) 블록을 잘게 나누고 블록마다 N=1 앵커를 둔다(앵커 비용 증가).

**검정력**: 셀을 줄여야 할 때 순서는 ① 라우팅 민감도 arm, ② 2.7B 모델의 TP, ③ cap 상향 arm이다. n을 줄이는 쪽으로는 가지 않는다.

## 7. 이전되는 것 / 이전되지 않는 것

| 이전됨(정의·규율) | 이전 안 됨(재측정 대상) |
|---|---|
| goodput 정의, paired bootstrap, 변화 trace 규약, n≥4, cliff 회피, realized pin 검증, 한 Job 안 arm | λ\*(모든 shape), rate 3↔12 절대값, cap48 아래 용량 |
| telemetry 스키마(rank 필드 추가), residency census, 관측자 판정 규약 | C2 수치, HE0 판정, decode-heavy static 권고, §1-25/§1-26 실현 분할 수치 |
| 4모델 roster(TP는 모델별 분할 가능성 확인 후) | 1-GPU Job 사이 비교 결과(교훈 274) |

## 8. 금지 서술

- "1-GPU λ\*/C2/HE0/decode-heavy static 권고가 N-GPU에서도 성립한다"(측정 전).
- "x8 노드의 N=1 = 기존 앵커."
- "N-GPU가 1-GPU보다 낫다"(자원이 N배다).
- "TP=N에서 PD-mux가 동작한다"(M2g 전). "M2g PASS ⇒ TP PD-mux 정확성 일반."
- "v0.5.10 기본 TP all-reduce가 SM 수(108) 기준 그리드를 쓴다" / "TP>1이면 `jit_kernel/all_reduce.py`가 발화한다."
- "custom all-reduce가 파티션을 침범한다/성능을 깎는다"(M2g·nsys 전에는 가설).
- "custom AR off arm에서 본 차이 = 운영점의 multi-GPU 고유 기전 크기."
- "λ\*(N)/(N·λ\*(1)) 스케일링 효율 = PD-mux 이득의 스케일링."
- "ρ 고정 goodput/GPU의 N-추세는 λ\* 스케일링과 독립된 증거다."
- "DP=N과 TP=N의 PD-mux 이득 순위"를 일반 명제로 쓰는 것.
- "N에 따른 goodput 변화는 PD-mux 효과"(셀별 SLO 실패 분해 전).
- 라우팅 효과를 PD-mux 스케일링으로 귀속하는 것.
- 블록 앵커로 Job 효과를 배제하지 않은 채 서로 다른 Job의 N을 한 곡선으로 잇는 것.
- 동시 DP 복제본의 duration 합산, 복제본 자기 구간 분모.
- "이 트랙의 DP/TP 결과 = AMD 트랙의 대조군/근거"(iso-GPU 공정성 등록 전).
- 감사 판정 `MIXED`를 "multi-GPU 트랙 GO"로 요약하는 것.

## 9. 사용자 결정 (2026-10-12 확정)

1. **시작 시점**: 지금 시작한다. N2(3-a)·N3와는 별도 Job이므로 병행 실행한다.
2. **M1 프로브**: AMD 트랙 V0와 **한 프로브로 묶어** 제출한다. V0 항목으로 GPU 쌍별 소형 메시지 복사 지연(F3, eager 기술치이며 V1
   crossing 마이크로벤치가 아니다)과 RDMA NIC·mooncake 유무(F5)를 추가했다.
3. **C-9**: x8 Job에는 운영 모델의 Job당 GPU-h 한도를 **적용하지 않는다**(사용자 결정). 따라서 x8 한 Job 안 N 부분집합 원칙을
   유지하고 블록 분할은 필요할 때만 한다. 블록을 나눌 경우 블록별 N=1 앵커 규칙(C-9 후단)은 그대로 적용한다.
4. **TP의 `--max-running-requests`**: **48·N**(N배)으로 한다. C-5에 따라 `max_total_tokens`·mamba cache 크기를 셀마다 기록한다.
   DP는 복제본당 48이다. 결과 서술에서 "TP와 DP의 cap 규약이 다르다"는 점을 범위로 명시한다.
5. **x8을 못 잡으면 보류**한다(x2/x4로 축소하지 않는다).

## 10. rev1 → rev2 감사 반영 대조표

| 감사 항목 | 반영 위치 |
|---|---|
| F-1(N1) estimand 부재·정규화 항등식 → P-F1 | §0-4, §5-1, §6 M2.5, §8 |
| R-1 jit AR은 opt-in, 기본 = CA v1(36 CTA)+NCCL, PCIe N≥4 무처치 | §0-5, §2-2, §8. handoff `:243` doc-steward 회부 |
| R-2 anchor:67은 PAUSE 계약 → 근거를 코드 입력 분석으로 이동 | §2-1 |
| R-3 static = fixed 경로, true-dual 발산, 열거형 fail-fast + 변이 검사 | §2-1, §6 M2 |
| R-4 capability assert 스코프 | §2-5 |
| R-5 Job 한도 GPU-h 오인용 → C-9 | §6 |
| 노드 부하 상관·부분집합 위상(C-1), 편차 분기(C-2) | §1 |
| ρ·SLO 영역, 셀별 실패 분해(C-3) | §4-3, §5-2 |
| CPU·GIL, M1.5(C-4) | §1, §6 |
| KV·mamba 용량(C-5) | §4-3, §5-2 |
| agnostic 의미·static D 선택(C-6) | §5-3 |
| DP 런처 혼용 금지(C-7) | §3-2 |
| M2g 비교 대상 교체(C-8), FAIL 셀 제외, 역순 스트레스 | §6 M2g |
| NCCL env 고정·기록(C-10), radix OFF 상속 | §5-2 |
| 게이트 7: 공유 벽시계 정의, 복제본 분모, 적용 지점, 변이 테스트 | §4-2 |
| duration 동일성 문장 철회, trace 주기 단위 | §5-2 |
| `job_entry.sh` GPU0만 기록, `:1966` 누락, `:157` 문구 정밀화 | §1, §2-1, §2-3 |
| §0-3 "GPU가 바뀌면" → Job 묶음 효과 | §0-3 |
