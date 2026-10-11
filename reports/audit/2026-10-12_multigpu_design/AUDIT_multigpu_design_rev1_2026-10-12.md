# 규칙층 감사 판정서: Multi-GPU 확장 설계 검토 rev1 (M0)

> 감사자: claims-auditor(읽기 전용, GPU 0). 2026-10-12. 판정문 원문을 저장한 것이다(저장: 메인 세션).
> 대상: `reports/multigpu_scaling_design_2026-10-12.md` rev1(HEAD `97a91b5` 시점 미추적 파일).

## 0. 판정 — `MIXED`

| 범위 | 등급 | 이유 |
|---|---|---|
| **M1(기판 프로브) · M2(엔진·하네스 구현) · M2g(correctness gate)** | **`GO-with-caveats`** | 死因 0. 정책 판정을 하지 않는 단계다. 필수 정정 R-1…R-4와 caveat 승계를 조건으로 실행 승인 |
| **M3(λ\*(N,mode)) · M4(본 비교)를 rev1 규칙으로 실행** | **`NO-GO`** | **F-1(N1: estimand 부재, 항등식 경로)**. 목적 (a)의 판정량이 정의되지 않았고, §5의 정규화와 goodput/GPU를 조합하면 판정량이 M3 출력의 재표현이 된다. M2g→M3 사이에 사전등록·규칙층 감사 gate가 없다 |

`GO-with-caveats`는 실행 승인일 뿐이며 주장 범위를 좁힌다. 어떤 게이트도 닫지 않는다. M3/M4의 `NO-GO`는 트랙 중단이
아니며, M3 앞에 PREREG와 그 규칙층 감사를 넣으면 해소된다(P-F1).

## 1. 인용 근거 검증

| 인용 | 실제 | 판정 |
|---|---|---|
| `vessl.env:7` | 일치 | OK |
| `engine_bench_runner.sh:85-128` | 실제 경로 `workspace/engine-port/scripts/r2_eval/...`. `--tp-size/--dp-size` 인자 없음, 저장소 전체 0건 | OK(경로 표기) |
| `telemetry.py:126` | `self.path.open("a")`, rank 처리 없음 | OK |
| `multiplexing_mixin.py:351` | 일치 | OK |
| `:155` | 일치 | OK |
| `:626`/`:1327`/`:1095` | 일치 | OK(서술 수위는 R-3) |
| `:1221` | `update_split_prefill_batch` 안, `:1227 get_new_batch_prefill` 직전 | OK |
| `:1525-1526` | 일치. 같은 구조가 `:1966`(true-dual)에도 있음 — rev1 누락 | OK, 누락 |
| `holb_probe.py:564-567` | 일치(`chunk_probe.py:987-990`도 선례) | OK |
| dev `model_runner.py:972` | 일치 | OK |
| dev `parallel_state.py:1806-1823` | 일치. 두 그룹 모두 `_ENABLE_CUSTOM_ALL_REDUCE` 기본값을 받아 **CA comm이 두 벌**, 둘 다 `pynccl_comm.disabled=False` | OK, 추가 사실 |
| dev `cuda_graph_runner.py:806-814` | 일치 | OK |
| dev `server_args.py:6138` | 일치. 같은 블록에 TP/DP 금지 assert **없음**(pp=1, chunked=-1, overlap off만 강제) | OK |
| dev `pdmux_context.py:109,119` | 실제 `:109`·`:120`. `:120`은 `divide_sm` 분기에서만 실행 | 줄 off-by-one, 스코프 누락(R-4) |
| **`jit_kernel/all_reduce.py:129`** | 줄은 일치하나 **v0.5.10 CUDA 기본 TP all-reduce 경로가 아님**(R-1) | **필수 정정** |
| `ssm_kernel_sm_partition_2026-09-21.md:243` | "TP>1이면 발화, NUM_CTA=전체 SM" — **원 문서 서술 자체가 기본 설정에서 거짓** | 인용원 오류 전파 |
| **`system_design_anchor_2026-09-30.md:67`** | §4 **Pause manager 계약**의 금지 조합(미구현 PAUSE 기능 대상) | **스코프 오인용**(R-2) |
| `impl_vs_external_pdmux_2026-08-28.md:157` | "실행된 적 없는 분기"는 비만장일치 결과 분기에 한해서만 참. 합의 allreduce 호출은 tp=1에서도 매번 실행 | 경미 |
| `amd_design_discussion:153` | 일치 | OK |
| `job_entry.sh:120-172` | 일치. 추가: `gpu.csv`를 `splitlines()[0]`만 파싱 → x8에서 **GPU0만** 기록, `clk.csv`는 전 GPU 행 혼합 | 문서보다 나쁨 |
| `substrate_probe_helper.py:227-257` | 일치 | OK |
| `n2_plan.py` | 실제 `results/n2_static_grid/n2_plan.py`, 일치 | OK |
| `vessl_operating_model:20` | 원문 **"Job당 ≲3–4 GPU-h"**. rev1은 벽시계 h로 오인용 | **오인용**(R-5 → C-9) |

### R-1 (필수): jit custom all-reduce는 기본 경로가 아니다
`srt/distributed/device_communicators/custom_all_reduce.py:339-356` `dispatch_custom_allreduce()`: `SGLANG_USE_JIT_ALL_REDUCE`
(기본 false)일 때만 `CustomAllReduceV2`(jit, `NUM_CTA = SM 수`)를 쓰고, 기본은 sgl_kernel v1 `CustomAllreduce`다.
- v1(`sgl-kernel/csrc/allreduce/custom_all_reduce.cuh:20,417,444`) 그리드: `kMaxBlocks = 36`,
  `blocks = min(36, ceil(size/threads))` — **SM 수·파티션 무관 고정 상한 36 CTA**.
- 크기 분기(`parallel_state.py:585-619`): `should_custom_ar`(크기 ≤ `max_size` ∧ (world_size==2 ∨ full NVLink))이면 CA v1,
  아니면 NCCL. prefill의 큰 all-reduce는 NCCL, decode의 작은 all-reduce는 CA v1일 가능성이 높다(실측 전 가설).
- `_SUPPORTED_WORLD_SIZES = [2,4,6,8]`. PCIe 노드 N≥4에서는 full_nvlink 거짓 → CA 꺼짐 → **`--disable-custom-all-reduce`
  arm이 N=4/8에서 무처치**.
- 처방: §2-2를 "기본 = CA v1(36 CTA 고정, 파티션 비인지) + 크기 초과 시 NCCL; `NUM_CTA=SM 수`는 opt-in"으로 정정,
  `SGLANG_USE_JIT_ALL_REDUCE` unset을 기판 기록에 남김. handoff `:243`은 doc-steward 정정 회부.
- 살아 있는 가설: green ctx 파티션(예: 16 SM) 안에서 CA v1의 36 CTA와 NCCL 채널 CTA가 점유하는 SM 비율, prefill comm과
  decode comm의 동시 실행 — (b)의 정확한 1순위 후보.

### R-2 (필수): anchor:67은 PAUSE 계약의 금지 조합
R2 generic/hybrid·SLO_SCHED의 TP>1 랭크별 발산 위험은 `multiplexing_mixin.py:626/1095` 입력 분석으로 독립 성립한다.
근거를 그쪽으로 옮겨라.

### R-3 (필수): "주 비교 arm은 결정적"의 조건
static arm은 `manual_divisions` 임계표가 아니라 **`PDMUX_R2_POLICY=fixed`** 경로다(`n2_plan.py:627-637`,
`policy_adapter.sh:33-45`) → `_r2_decide_idx`를 탄다. `FixedPolicy.decide`(`controller.py:88-103`)는 `safe_boundary`만 쓰고,
legacy 루프에서 상수 True → **legacy + fixed는 결정적**. **true-dual**에서는 `arbiter.safe_to_switch()`가 타이밍 의존이라
fixed도 발산 가능(`:1966` 합의 경로). 처방: TP>1 **허용** = legacy 루프 ∧ (`PDMUX_R2_POLICY` unset ∨ `fixed`) ∧
`PDMUX_SLO_SCHED` unset ∧ `PDMUX_LA_COORD` unset ∧ `PDMUX_TRUE_DUAL_WORKER` unset, 그 밖은 **거부**. sticky+fixed는 M2g에서
따로 확인. 각 금지 조합을 허용으로 바꾼 변이본에서 테스트가 실패해야 한다(교훈 53).

### R-4 (경미): capability 불일치 assert 스코프
영향 지점은 `divide_sm` 경로(`:120`)뿐. assert는 유지하되 무엇을 막는지 정확히 적어라.

### R-5: Job 한도 오인용(GPU-h ↔ h) — C-9로 연결.

## 2. 死因 F-1 (N1: estimand 부재와 정규화 항등식)

- (a) "PD-mux 이득의 N-스케일링"의 수식 정의가 없다. λ\*(N,mode)를 어느 arm에서 재는지도 미정.
- `analyze.py:54-70`: goodput = good/duration, duration ≈ 요청 수/offered rate. §5(offered = ρ·λ\*(N,mode), 요청 수 = NP·N)를
  대입하면 cliff 아래에서 goodput/GPU ≈ f·ρ·λ\*(N,mode)/N = f·ρ·λ\*(1)·e(N), f = good 비율, e(N) = λ\*(N)/(N·λ\*(1)).
  게이트 6이 f≈1을 강제하므로 M4의 "goodput/GPU 스케일링"은 M3의 e(N) 재표현이다.
  - λ\*를 arm별로 재고 ρ 고정 → arm 간 goodput 비 ≈ (λ\*\_agn/λ\*\_fused)·(f비) — 이득 전체가 분모에서 나옴.
  - 한 기준 arm에서만 재면 arm 간 차이는 f비. 단 cliff에서 멀면 f가 포화(N3 계열 퇴화).
- **처방 P-F1**:
  1. (a)의 1차 estimand = **용량비 G(N,mode) = λ\*\_agnostic(N,mode)/λ\*\_fused(N,mode)**(arm별 KNEE_BRACKETED, 같은 Job),
     판정량 = dG/dN. decode-heavy static은 C-6 규칙으로 고른 D로 같은 방식.
  2. M4는 **2차 estimand**: offered rate는 **단일 기준 arm(fused)의 λ\*(N,mode)**로 정규화, ρ\_lo·ρ\_hi 사전등록,
     보고량 = 같은 offered rate에서 arm 간 f 차이(paired). goodput/GPU의 N-추세는 e(N)과 분리 보고.
  3. **M2.5 = PREREG_multigpu + 규칙층 감사** 없이 M3 시작 금지.
  4. 예측 출력공간에 "거짓"(예: dG/dN paired CI가 0 포함, G(N)<1)을 포함.

## 3. 반전 시험 표

N>1 원자료 0건. 1-GPU 수치는 방향 예보에만 사용.

| 자유 표면 | 범위 | 판정 변화 |
|---|---|---|
| λ\* 기준 arm | arm별 ↔ 단일 | 판정량 자체가 바뀜 → F-1 |
| ρ\_lo/ρ\_hi | HE0 3/12 환산 | shape A λ\*로 환산 시 12 req/s ≈ 3.9×λ\* → HI phase 과부하 가능(형상 상이, 수치 인용 금지) → C-3 |
| TP cap | 48 ↔ 384 | 미입증 → C-5 |
| 라우팅 | RR ↔ SQ | 미입증, 주/민감도 구분 적절 |
| agnostic 임계표 TP 의미 | 1-GPU 표 ↔ ×N | TP decode_bs = 전체 배치(GPU당 일 1/N), DP 복제본 decode_bs ≈ 1/N → mode마다 다른 정책(`multiplexing_mixin.py:1180-1188`) → C-6 |
| static D | 고정 ↔ 사후 최적 | winner's curse 또는 미조율 → C-6 |
| M2g logit 허용오차 | 0 ↔ 느슨 | 완전 일치 요구 시 무해한 합산 순서 차이로 FAIL 가능 → C-8 |
| cgroup throttled 문턱 | 0% ↔ ∞ | → C-4 |
| 셀 축소 순서 | ①→③ | 판정 분기 불변 |
| DP 구현 | `--dp-size` ↔ N 독립 서버 | 프론트엔드 차이 → C-7 |

수치 반전 표면 0개. N2 死因 없음.

## 4. "동일 조건" 교락 점검

| 교락 | 감사 |
|---|---|
| Job 효과 | 교훈 274는 Job(GPU·노드·시간) 묶음 효과, GPU 단독 귀속 미확립. 노드 안 편차 문턱·분기 규칙 없음 → C-2 |
| **노드 부하 상관(누락)** | N=8은 노드 전력·열·NVSwitch·호스트 메모리를 쓰고 N=1은 7장 유휴 → 노드 클럭·throttle이 N과 공변. 동시 실행 시 간섭 → C-1, C-9 |
| **부분집합 위상(누락)** | N=2/4 GPU 집합이 NVLink 위상·NUMA·CPU 친화도와 교락 → 부분집합 사전등록·위상 기록(C-1) |
| CPU quota | 문턱 미지정. **누락**: TokenizerManager·detokenizer·클라이언트가 단일 프로세스 → GIL 병목이 GPU 확장성 한계로 오독(confound #7) → C-4 |
| **KV·mamba 용량(누락)** | TP는 GPU당 가중치 1/N → `--mem-fraction-static 0.82` 아래 KV 풀·mamba 슬롯이 N에 따라 증가. DP는 1-GPU와 동일 → C-5 |
| prefix/radix | runner가 `--disable-radix-cache` 강제. 상속 명시·OFF 고정 등록(켜면 RR이 prefix 적중을 쪼갬) |
| TP 통신 | R-1. PCIe N≥4에서 무처치 → 위상별 처치 강도 사전 확인 |
| **NCCL env(누락)** | `NCCL_ALGO/PROTO/MAX_NCHANNELS/MIN_NCHANNELS`, `CUDA_DEVICE_MAX_CONNECTIONS`, NCCL 버전, `SGLANG_USE_JIT_ALL_REDUCE`, `SGLANG_CUSTOM_ALLREDUCE_ALGO`(`.cuh:447`) 고정·기록. 채널 수는 파티션 안 SM 사용을 바꾸는 숨은 노브 → C-10 |
| **ρ와 SLO 영역(누락, 교훈 267)** | TP는 step당 ITL을 줄여 ITL SLO 구속 여부가 N에 따라 바뀜 → 셀별 실패 분해(TTFT-fail/ITL-fail/둘 다) 의무 → C-3 |
| duration 동일 | duration ≈ NP/(ρ·λ\*(1)·e(N)) → e(N)=1일 때만 같음. 문장 정정, 정상상태를 시간으로 등록할지 결정 |
| 변화 trace 주기 | 초 고정 ↔ 요청 수 고정 중 하나 등록 |

λ\* 정규화가 스케일링을 흡수하는 항등식 위험은 **실재** → F-1.

## 5. 게이트 7 해석 — 충돌 없음(반증 실패)

동시 복제본은 시간 창을 공유하므로 합산하면 분모가 N배. 단 문자로 등록할 것:
1. 공유 벽시계 = 라운드별 클라이언트 기준 단일값(첫 전송~마지막 완료), 라운드 사이는 합산. `max()`라는 단어를 쓰지 말고 정의로.
2. 복제본별 goodput도 공유 벽시계를 분모로. 복제본 자기 구간으로 나눈 뒤 합산하는 변이본은 회귀 테스트에서 KILLED.
3. `--dp-size` 단일 서버에서는 클라이언트 duration이 원래 하나 — 적용 지점은 N-독립-서버 폴백과 복제본 분해 분석.

## 6. 사다리·NO-GO 점검

| 단계 | 결함 | 처방 |
|---|---|---|
| M1 | 편차 결과에 따른 분기 없음 | 문턱 등록(예: 노드 안 1-GPU 셀 차이 >3%면 N=1 회전 의무, GPU 공변량), 위상 결과로 CA 기전 arm 존재 여부 확정 |
| M1 | 클라이언트·토크나이저 여유 확인 없음 | **M1.5 하네스 용량 프로브**: 등록 최대 offered rate ×1.5에서 send-lag·TokenizerManager 큐 지연 확인(더미 백엔드 가능). FAIL 시 다중 프로세스 클라이언트·`tokenizer_worker_num` 조정 |
| M2g | 비교 대상 오류 | 1차 = 같은 TP=N에서 PD-mux OFF↔ON, 2차 = TP=N↔TP=1 **teacher-forced logit** 허용오차. 자유 생성 완전 일치 요구 금지 |
| M2g | FAIL 처리 | 그 (mode, N)의 M3/M4 셀은 정책 주장에서 제외 명시 |
| M2g | 교착 검사가 시간 제한뿐 | prefill/decode comm 역순 실행 스트레스 케이스, 반복 횟수 등록 |
| M2→M3 | 사전등록 gate 부재 | M2.5 |
| M3 | arm 미지정 | arm별 λ\*. 클라이언트 한계는 `HARNESS_LIMITED`로 `UNSTABLE`과 구분(교훈 21) |
| M4 | static D 선택 규칙 부재 | M3 데이터로 등록 규칙에 따라 선택, M4는 별도 seed. 동적 arm은 TP 축 제외 |
| 전체 | x8 Job 크기 규약 충돌(C-9) | 한도는 ≲3–4 **GPU-h** → x8 벽시계 ≲0.4–0.5 h, TP=8 부팅+cudagraph 캡처로 블록이 안 들어갈 수 있음. 사용자 결정: (i) x8 한도 개정 (ii) 잘게 분할+블록별 N=1 앵커. "셀당 GPU-h는 N에 비례"는 x8 순차 실행에서 거짓(N=1 셀도 8장분) |

### §8 금지 서술 추가
- "v0.5.10 기본 TP all-reduce가 SM 수(108) 기준 그리드를 쓴다" / "TP>1이면 `jit_kernel/all_reduce.py`가 발화한다".
- "λ\*(N)/(N·λ\*(1)) 스케일링 효율 = PD-mux 이득의 스케일링".
- "ρ 고정 goodput/GPU의 N-추세는 λ\* 스케일링과 독립된 증거다".
- "DP=N과 TP=N의 PD-mux 이득 순위"를 일반 명제로.
- "custom AR off arm에서 본 차이 = 운영점의 multi-GPU 고유 기전 크기".
- "M2g PASS ⇒ TP PD-mux 정확성 일반".
- "이 트랙의 DP/TP 결과 = AMD 트랙의 대조군/근거"(iso-GPU 공정성 등록 전).
- "N에 따른 goodput 변화는 PD-mux 효과"(셀별 실패 분해 전).

## 7. 정본과의 모순

모순 없음(Claim 등급·게이트 #6·HE0·§1-3·§1-4·1-GPU 권고 불변, §7 "이전 안 됨"은 게이트 1 연장·AMD rev2와 정합).
수정 필요: R-1(인용원 handoff `:243` 오류), R-2, R-5, §0-3 "GPU가 바뀌면"(Job 묶음 효과). 이 판정서의 `MIXED`를
"multi-GPU 트랙 GO"로 요약 금지.

## 8. caveat (문자 그대로 승계)

- **MG-0**: "이 설계 문서(rev1)의 규칙으로는 M3·M4를 실행할 수 없다. M3 이전에 PREREG_multigpu와 그 규칙층 감사가 선행해야 한다. `GO-with-caveats`는 M1·M2·M2g에만 해당한다."
- **C-1**: "x8 노드에서 N 부분집합을 순차 또는 동시 실행한 결과는 노드 부하·GPU 부분집합 위상·NUMA와 공변하므로, 부분집합 GPU ID·위상·GPU별 클럭/throttle을 공변량으로 병기하지 않은 N-추세는 인용하지 않는다."
- **C-2**: "M1의 GPU별 편차 측정은 사전등록 문턱과 결과 분기 없이는 N=1 앵커의 대표성을 보증하지 않는다."
- **C-3**: "같은 ρ는 N과 mode에 걸쳐 같은 SLO 영역을 뜻하지 않는다(교훈 267). 셀별 TTFT-fail/ITL-fail 분해 없이 goodput의 N-추세를 PD-mux에 귀속하지 않는다. HE0 3/12 req/s를 ρ로 환산한 HI phase가 과부하(ρ>1)이면 그 비교는 phase 길이 의존 과부하 비교(게이트 #18 계열)다."
- **C-4**: "단일 TokenizerManager·단일 클라이언트 프로세스의 한계는 'GPU 확장성 한계'와 구별되지 않는다. 하네스 용량 프로브(M1.5) PASS 전의 상위 N 용량 수치는 인용하지 않는다. cgroup throttled 실격 문턱은 사전등록 값이다."
- **C-5**: "TP=N과 DP=N은 GPU당 KV 풀과 mamba 상태 슬롯 수가 다르다. `--max-running-requests` 규약(사용자 결정)과 별개로 `max_total_tokens`·mamba cache 크기를 셀마다 기록하지 않은 mode 간 비교는 인용하지 않는다."
- **C-6**: "'agnostic'은 decode-bs 임계표라 TP와 DP에서 GPU당 일 기준으로 다른 정책이다. static D의 N별 선택 규칙이 사전등록되지 않은 static arm은 미조율 arm(confound #2)이다."
- **C-7**: "`--dp-size` 경로와 N-독립-서버 폴백은 프론트엔드가 달라 같은 조건이 아니다. 한 캠페인 안에서 섞지 않으며, N=1도 같은 런처로 잰다."
- **C-8**: "M2g의 TP=N↔TP=1 비교는 수치 순서 차이를 포함한다. PD-mux 정확성은 같은 TP=N에서 PD-mux OFF↔ON으로 판정한다. M2g PASS는 시험 범위 밖 정확성을 보증하지 않는다."
- **C-9**: "운영 모델의 Job 한도는 GPU-h 단위(≲3–4 GPU-h)다. x8 Job 설계는 그 한도 개정(사용자 결정) 또는 블록 분할을 전제한다. 블록 사이 N 비교는 블록별 N=1 앵커로 Job 효과를 배제한 경우에만 하나의 곡선으로 잇는다."
- **C-10**: "NCCL 채널·알고리즘 env, `SGLANG_USE_JIT_ALL_REDUCE`, `SGLANG_CUSTOM_ALLREDUCE_ALGO`를 고정·기록하지 않은 (b) 기전 관측은 인용하지 않는다. PCIe 위상에서 N≥4이면 custom AR on/off arm은 무처치이므로 그 셀에서 '통신 기전 없음'을 결론 내리지 않는다."

## 9. 처방 요약 (rev2)

1. R-1…R-5 정정. handoff `ssm_kernel_sm_partition_2026-09-21.md:243`는 doc-steward 회부.
2. P-F1(1차 estimand = arm별 용량비 G(N,mode)와 N-추세; M4 = 단일 기준 arm 정규화 2차 estimand; 출력공간에 "거짓").
3. 사다리: M1.5, M2.5, M2g 비교 대상 교체, fail-fast 열거형 술어(R-3)와 변이 검사.
4. §5 추가 등록: radix OFF 상속, NCCL env, GPU 부분집합, trace 주기 단위, duration 문장 정정, 셀별 SLO 실패 분해.
5. §4-2 공유 벽시계를 `max()` 없는 정의로, 복제본 자기 구간 분모 변이 테스트.
6. 처방 실현가능성(게이트 #113): P-F1은 M3 틀에 arm 차원 추가, M1.5는 GPU 없이 가능. **C-9만 사용자 결정 의존** — x8 한 Job 원칙과 GPU-h 한도가 정면 충돌, rev2 전에 결정 필요.

반증을 시도한 범위에서 M1/M2/M2g의 死因은 찾지 못했다. 위 caveat 없이 M3/M4의 정책·스케일링 주장은 성립하지 않는다.
