# C-1 — `%smid` 직접 계측: 기술 타당성 평가와 설계 (2026-08-11, engine-porter)

> **문서 지위**: 설계 문서. **구현 아님, 결과 아님, 성능 주장 없음.**
> 이 문서는 `PREREG_GATE2S_2026-08-09.md` §6.2.2·§0.0.B·§11.3·§12-5가
> "병행 별건으로 즉시 등재"라고 지정한 항목의 **타당성 평가 + 설계 초안**이다.
> 사전등록이 아니다(§5의 결정 규칙은 **후보**이며 확정이 아니다).
>
> **공유 dev 트리 무변경.** 이 조사 중 `/scratch/ehmoon/whlee/sglang_engine_dev/`에
> 쓰기 작업은 0건이다. 프로토타입은 전부 세션 스크래치패드에서 컴파일만 했고
> (GPU 미사용), git commit 0건이다.

---

## 0. 읽는 순서 (요약)

1. **§2 검증표를 먼저 읽어라.** 이 문서에서 *확인된 것*과 *추정한 것*은 §2에서
   기계적으로 분리돼 있다. 나머지 절은 그 분류를 참조만 한다.
2. **§8("이 실험으로도 닫히지 않는 것")을 §7(권고)보다 먼저 읽어라.** 방법론
   게이트 #28의 요구다.
3. §6이 "이것이 네 번째 통계 게이트가 아닌 이유"의 논증이다. 이 논증이 무너지면
   설계 전체가 §11.3에 걸린다.

**한 줄 결론**: `%smid` 직접 계측은 **이 트리에서 기술적으로 가능하고, nvcc 빌드도
CUPTI 권한도 필요 없다**(Triton `inline_asm_elementwise`로 CPU에서 sm_80 PTX/cubin
생성까지 확인). 그러나 **그것이 닫는 것은 "전달 질문"뿐이고 §1-1의 귀속은 닫지
않는다.** 권고는 **L0(엔진 무패치 standalone census) → L1(부팅 시 in-server census)**
2단이며, **L2(프로덕션 커널 태깅)는 이번에 사지 않는다.**

---

## 1. 무엇을 닫으려는가 — 그리고 그게 코드 사실인지 추정인지 (방법론 게이트 #28)

게이트 #28: *"실험이 무엇을 풀어주는지가 코드 사실인지 추정인지, 실행 전에 코드로
확인한 뒤 정당화하라."* 아래는 그 확인 결과다.

### 1.1 구멍 A — "무분할"의 물리적 내용이 미측정

`PREREG_GATE2S §1.1`(rev4 하향)·§1.3이 확립한 것:

- `pdmux_context.py:124-137` — `SM_COUNTS`의 idx 0 = `(108, 0)`, idx N-1 = `(0, 108)`은
  **부기 라벨**이고, 그 인덱스의 스트림 쌍은 green context가 아니라 평범한
  `torch.cuda.Stream`이다(`:129-131`, `:136-138`). green ctx는 idx 1..N-2만
  (`:133-135`, `spatial.create_greenctx_stream_by_value`).
- **코드로 검증된 상한은 "명시적 분할이 적용되지 않는다"까지다.** "양쪽 다 전체
  108 SM을 쓴다"는 **미검증 물리 주장**이며 rev4에서 철회됐다.
- 액면 인용은 Stage 0 D108 오류의 **거울상**이다(그때는 target pin과 realized가
  달랐다 — 교훈: pin은 target이 아니라 realized로 검증하라).

**이 구멍이 실제로 코드의 어디를 오염시키는가 (게이트 #28 확인 결과):**

| 소비처 | 코드 | 라벨 의존 형태 | 이 실험이 고치는가 |
|---|---|---|---|
| `internal_epsilon_34_to_108` | `g2s_analyze.py:1488` `eps = log(mt/mc)/log(108.0/34.0)` | **분모가 물리 SM 비를 하드코딩**한다. C가 물리적으로 108이 아니면 이 수는 잘못된 끝점비로 계산된 것이다 | **예** — 이 하나는 실제로 고쳐진다 |
| 사전등록 in-job 반증기 | `g2s_analyze.py:1132` `falsifier(tp, [(108,0),(74,34),(54,54),(0,108)], split_idx=2)` | 기대 라벨 집합이 하드코딩. **라벨 대 라벨** 비교라 물리량과 무관 | **아니오** (항등식성 비교) |
| 크기 인용 자격 | `g2s_analyze.py:1157-1161` `gate_label([...])` = **F-계열 게이트** | SM 라벨과 **무관**. 인용을 막는 건 F-계열이지 `premise`/SM 라벨이 아니다 | **아니오** |
| telemetry `runtime_snapshot.prefill_sms` | `dual_worker.py:608, 622-623` | 순수 부기. `sm_counts[i][0]`을 실제 제약으로 쓰는 코드는 전수 확인상 **없다**(§1.3) | 라벨을 **대체**할 실측치를 준다 |

⇒ **정직한 결론(코드 사실)**: `%smid`가 성공해도 **Gate 2-S의 인용 가능 셀 수는
1개에서 늘어나지 않는다.** 늘어난다고 쓰면 그것이 정확히 G1-c에서 반증된 서술
형태다. 실제로 바뀌는 것은 (a) 부수 통계량 하나(`internal_epsilon`)의 끝점 정당성,
(b) **모든 Δ_split 인용에 의무 병기되는 캐비어트 문장**("잔여 차감은 미측정"),
(c) telemetry 라벨을 실측 집합으로 **교체**할 수 있게 되는 것.

### 1.2 구멍 B — 잔여 차감(residual carve)의 격리 측정 수단 소실

`PREREG_GATE2S §6.2.1` rev3에서 **P-b가 삭제**됐다(처치 = green ctx 개수 3 vs 4가
KV 풀·cudagraph 캡처 메모리와 완전 공선). 남은 P-carve(C vs A2)는 자인된 혼입
(`event_loop_normal` vs `event_loop_pdmux` 루프 오버헤드)이 있고, rev4가 **마이크로
체제 캐비어트를 의무 병기**시킨다(8요청·in128/out128). ⇒ **서빙 동작점의 잔여
차감은 어떤 프로브로도 배제되지 않았다.**

`%smid` census는 이 질문을 **성능 지표가 아니라 SM 집합의 차집합**으로 바꾼다:
같은 프로세스에서 green ctx 생성 **전** 평범 스트림의 SM 집합 `S_before`와 생성
**후** `S_after`를 재고 `S_before \ S_after`를 본다. 이건 통계량 비교가 아니라
집합 연산이다.

### 1.3 구멍 C — (이 조사에서 새로 발견) **cudagraph 경유 전달이 미검증**

문서화된 구멍은 아니었으나 코드 확인 중 나왔다. 전부 **코드 사실**이다:

- `cuda_graph_runner.py:547` — `capture_forward_mode = ForwardMode.DECODE`. ⇒ 운영점
  (cudagraph ON)에서 **decode만 그래프로 캡처되고 prefill은 eager**다.
- `:812-816` — pdmux 분기는 `for i, sg in enumerate(self.stream_groups)`로 돌며
  **`graph_capture(stream=sg[1])`**, 즉 **각 stream group의 decode 스트림 위에서**
  그룹마다 별도 그래프를 캡처한다.
- `:1158-1161` — 재생은 `graph_key = f"{stream_idx}_{bs}"` → `self.graphs[key].replay()`.

⇒ **미검증 물리 질문**: green-ctx 스트림 위에서 **캡처된** CUDA graph를 **재생**할 때
그 green context의 SM 제한이 그대로 전달되는가? 이 프로젝트의 운영점은
cudagraph-ON이고 decode 측 SM 분할은 전부 이 경로를 지난다. **어떤 아티팩트도 이걸
측정한 적이 없다.** `%smid`는 이 질문에 답할 수 있는 유일한 수단이다(§3의 어떤
다른 경로도 SM **정체성**을 주지 않는다).

⚠️ **여기서 방향을 단정하지 않는다.** "전달된다"도 "안 된다"도 현재 근거 없음.
이 문서는 그것이 **측정 가능한 열린 질문**이라고만 말한다.

### 1.4 구멍 D — (동상) `ExternalStream` 경유 전달이 미검증

`sgl_kernel/spatial.py:create_greenctx_stream_by_value`는 green ctx의 CUstream 포인터를
`torch.cuda.streams.ExternalStream`으로 감싸 돌려주고, 엔진은 그것을 primary context
안에서 `with torch.cuda.stream(s)`로 쓴다. CUDA 의미론상 launch가 실행되는 context는
스트림이 결정하지만, **이 조합이 실제로 SM 한정을 전달하는지는 이 저장소에서 측정된
적이 없다.** `%smid`는 이것도 함께 답한다(§5-R1과 같은 판독).

---

## 2. ★ 검증표 — 이 문서에서 **확인한 것 / 추정한 것**

미검증 가정이 이 프로젝트의 결론을 두 번 뒤집었으므로 분리해 둔다.

### 2.A 이 세션에서 **직접 확인**한 것 (재현 명령 포함)

| # | 사실 | 확인 방법 | 결과 |
|---|---|---|---|
| V1 | **Triton 3.5.1이 `%smid`를 PTX로 낸다** | `tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", ...)` → `triton.compile(..., target=GPUTarget("cuda",80,32))` | PTX에 `mov.u32 %r3, %smid;` 1개소. **GPU 없이 컴파일됨** |
| V2 | `%nsmid`·`%globaltimer`도 같은 방식으로 나온다 | 동상 | 각 1·2 mov 사이트 |
| V3 | **`%globaltimer` 스핀 루프가 hoist되지 않는다** | 생성 PTX의 basic block 검사 | `$L__BB0_1` 안에 `mov.u64 %rd17, %globaltimer;` + `@%p2 bra $L__BB0_1` (역방향 분기) — 루프 안에서 재판독 확인 |
| V4 | **cubin까지 CPU에서 생성된다** | `k.asm.keys()` = `['source','ttir','ttgir','llir','ptx','cubin']` | nvcc·GPU 불필요 |
| V5 | **레지스터 델타를 GPU 없이 잴 수 있다** | 동일 커널 baseline/instrumented를 cubin으로 덤프 → `/apps/cuda/13.0.2/bin/cuobjdump -res-usage` | `REG:8`(base) → `REG:10`(+`%smid` 1판독+1스토어), `SHARED:0 LOCAL:0 STACK:0` 양쪽 동일 |
| V6 | **libcuda.so.1이 green ctx 전 API를 export한다** | `nm -D /lib64/libcuda.so.1` | `cuGreenCtxCreate`·`cuGreenCtxGetDevResource`·`cuCtxGetDevResource`·`cuStreamGetGreenCtx`·`cuDevSmResourceSplitByCount`·`cuGreenCtxGetId` 전부 `T` |
| V7 | **`cuStreamGetGreenCtx`는 비-green 스트림에 NULL을 반환한다**(문서) | `/apps/cuda/13.0.2/include/cuda.h:25151-25196` | "Otherwise, `*phCtx` is set to NULL instead." ⇒ §1.1의 "평범 스트림"을 **드라이버 층에서** 검증 가능 |
| V8 | **`CUdevSmResource`에는 SM **개수**만 있고 ID 집합은 없다** | `cuda.h:24770-24777` (`smCount`·`minSmPartitionSize`·`smCoscheduledAlignment`) | ⇒ 경로 3(자기보고)은 **정체성**을 원리적으로 줄 수 없다 |
| V9 | **드라이버가 green ctx 분할의 disjoint를 문서화한다** | `cuda.h:24992-24995` "guarantee a split that will create a **disjoint** set of symmetrical partitions" | 단 **primary context가 영향받는지는 헤더가 말하지 않는다** (= 구멍 B가 실재하는 이유) |
| V10 | **로그인 노드에서 프로파일링이 admin 전용이다** | `cat /proc/driver/nvidia/params \| grep -i restrict` | **`RmProfilingAdminOnly: 1`** |
| V11 | **CUPTI·ncu·nsys는 설치돼 있다** | `/apps/cuda/13.0.2/extras/CUPTI/lib64/`, `bin/{ncu,nsys}` | `libcupti.so.2025.3.1`·`libnvperf_*`·`libpcsamplingutil.so` 존재 |
| V12 | **decode만 cudagraph로 캡처된다** | `cuda_graph_runner.py:547` | `capture_forward_mode = ForwardMode.DECODE` |
| V13 | **pdmux 캡처는 stream group마다 decode 스트림 위에서 일어난다** | `:812-816` | `for i, sg in ...: graph_capture(stream=sg[1])` |
| V14 | **재생은 stream-idx별 그래프를 고른다** | `:1158-1161` | `graph_key = f"{stream_idx}_{bs}"` |
| V15 | **mamba SSU 백엔드 기본값 = triton** | `server_args.py:490` `mamba_backend: str = "triton"`, `ssu_dispatch.py:180` `requested = server_args.mamba_backend or "triton"` | 저장소 전 캠페인에서 `--mamba-backend`는 **한 번도 설정된 적 없음**(`grep -rn "mamba-backend" results/ scripts/ triage/` = 0건) |
| V16 | **SSM 커널은 두 모델 모두 Triton이다** | decode: `mamba/mamba.py:636,671` `selective_state_update(...)` → `ops/ssu_dispatch.py:53` `TritonSSUBackend` (V15로 기본값) → `ops/mamba_ssm.py:67-68` `@triton.jit _selective_scan_update_kernel`. prefill: `mamba/mamba.py:533` `mamba_chunk_scan_combined(...)` → `ops/ssd_combined.py` → `ops/ssd_chunk_scan.py:19-20` `@triton.jit _chunk_scan_fwd_kernel` | ⇒ **모델 대칭인 L2 부착점이 존재한다** |
| V17 | **attention 백엔드는 모델마다 다르다** | `g2s_run.sbatch:54-55` | Zamba2 = `triton`(우리 커널), **Granite = `flashinfer`**(precompiled, 계측 불가) |
| V18 | **`causal_conv1d`는 가능하면 precompiled sgl_kernel을 쓴다** | `mamba/causal_conv1d.py:15-22`; `sgl_kernel`에 `causal_conv1d_fwd`·`causal_conv1d_update` 존재 | ⇒ conv 경로는 계측 불가(필요 없음) |
| V19 | **프로브가 기존 부팅에 얹히는 전례가 있다** | `g2s_run.sbatch:247-249` (채점 벤치 **이전** 실행) | L1의 하네스 패턴은 이미 있다 |
| V20 | **`sm_group_num: 4`, manual 없음** | `results/p1_gates/gate2/pdmux_a100_smoke.yml` | 실현 `sm_counts` = `[(108,0),(74,34),(54,54),(0,108)]` |
| V21 | ★ **CUPTI 13.0.2 헤더 전체에 `smId` 필드가 없다** | `grep -rn "smId" /apps/cuda/13.0.2/extras/CUPTI/include/` → **0건**. 같은 헤더에 `CUpti_PCSamplingPCData`는 3회 등장(구조체는 존재) | ⇒ **CUPTI는 SM 정체성을 원리적으로 주지 않는다.** 권한(V10/E1)과 **무관하게** 경로 2가 우리 질문에 답하지 못한다 |

### 2.B 이 문서가 **추정**한 것 (검증 안 됨 — 프로브가 필요)

| # | 추정 | 왜 지금 확인 못 했나 | 어떻게 확인하나 |
|---|---|---|---|
| E1 | 컴퓨트 노드도 `RmProfilingAdminOnly: 1`이다 | V10은 **glogin01 측정**이다. 커널 모듈 파라미터는 노드별 배포 설정 | §10-P3 (초 단위) |
| E2 | green context 안에서 `%smid`가 **물리** SM id를 준다(가상화되지 않는다) | GPU 실행 필요. 문서화된 보장 없음 | §10-P1 (자기검증형, §5-R0) |
| E3 | A100 80GB에서 `%nsmid`·`%smid` 값역이 조밀한 `0..107`이다 | GPU 실행 필요. GA100은 물리 128 SM 슬롯 중 108 enable ⇒ **id가 희소할 수 있다** | §10-P1 |
| E4 | primary context는 green ctx 생성 후에도 108 SM을 본다 | 헤더가 침묵(V9) | §5-R2/R3 |
| E5 | green-ctx 스트림에서 캡처한 cudagraph 재생이 SM 한정을 전달한다 | 측정된 적 없음(§1.3) | §5-R4 |
| E6 | `ExternalStream` 경유 launch가 SM 한정을 전달한다 | 측정된 적 없음(§1.4) | §5-R1 |
| E7 | census 커널의 블록 수를 키우면 가용 SM 집합이 포화한다 | GPU 실행 필요 | §5의 포화 곡선(진단) |
| E8 | L2 계측 시 실제 프로덕션 커널의 레지스터가 occupancy 경계를 넘지 않는다 | V5는 **장난감 커널**에서 +2 reg. 실제 커널은 각각 재야 한다 | §6.2 (CPU-only, 무료) |
| E9 | 각 비용 수치(§9) | 과거 캠페인 부팅 시간에서 외삽 | 스모크(방법론 게이트 #26) |
| E10 | `ncu`가 `sm__*` 계열을 SM 인스턴스별로 뽑을 수 있다 | 실행하지 않았다(권한 벽 E1 + 직렬화가 측정 대상을 파괴하므로 확인 가치가 낮다고 판단) | 확인하지 않는다 — §3 결론 1에 따라 **E1이 거짓이어도 채택 안 함** |
| E11 | green ctx의 SM 인스턴스 index ↔ `%smid` 물리 id 매핑 | 문서화 없음, 실행 안 함 | 확인하지 않는다(E10과 동반 사망) |

---

## 3. 계측 경로 비교표

측정 대상 · 부착 지점 · 오버헤드 · **licence 범위** · 실패 모드.
"licence"는 *성공했을 때 쓸 수 있게 되는 문장*이다.

| # | 경로 | 실제로 무엇을 재는가 | 부착 지점 (file:line) | 오버헤드(추정 E9) | licence **하는** 것 | licence **못 하는** 것 | 주 실패 모드 |
|---|---|---|---|---|---|---|---|
| **1a** | **인라인 PTX `%smid` — out-of-band census** (권고) | 특정 스트림에 launch된 커널 블록이 **착지한 물리 SM id 집합** `S`. `%globaltimer` 병기 시 블록 체류 구간 | 엔진 무패치(L0): standalone 스크립트가 `sgl_kernel.spatial` 호출. In-server(L1): `multiplexing_mixin.py:64-70` `init_pdmux` 직후 + cudagraph 캡처 후 | 부팅당 1회, 정지 상태에서 수십 ms. **서빙 루프 오버헤드 0** | 파티션이 **물리적으로 전달됐는가**, 두 role SM 집합의 **disjoint 여부**, 잔여 차감의 **SM 단위 크기**, cudagraph 전달 여부 | 서빙 혼합 부하에서의 **동시** 점유, SM **활용도**(warp 수), 어떤 **효과** 귀속도 | E2가 거짓(가상 id)이면 전 판독 해석 불가 → §5-R0에서 즉시 정지 |
| **1b** | **인라인 PTX `%smid` — 프로덕션 커널 태깅** (L2, 이번엔 사지 않음) | 실제 serving 커널이 role별로 점유한 SM 히스토그램(경계 스냅샷) | decode: `mamba/ops/mamba_ssm.py:67-68` `_selective_scan_update_kernel`; prefill: `mamba/ops/ssd_chunk_scan.py:19-20` `_chunk_scan_fwd_kernel`; Zamba2 한정 추가: `triton_ops/decode_attention.py:45`, `triton_ops/extend_attention.py:107/113/219` | 블록당 atomicAdd 1회 + 레지스터 +2~3 | **서빙 동작점**에서의 role별 realized SM 집합·중첩 | Granite의 **attention** 경로(V17: flashinfer), 모든 **GEMM**(cuBLAS), `causal_conv1d`(V18) ⇒ **커널 커버리지가 부분적**이다 | cudagraph가 포인터를 굽는다(V13/V14) → role 태그 배관이 복잡. occupancy 변화(E8) |
| **2** | **CUPTI PM / PC 샘플링** | 커널·SM 카운터. PC 샘플링은 stall 이유 | 외부 프로세스 또는 injection | 커널 직렬화(metric replay) — **동시성을 파괴** | (권한이 있다면) 카운터 기반 SM 활동 | ★ **SM 정체성을 주지 않는다** — **V21**: CUPTI 13.0.2 헤더 전체에 `smId` 필드가 **0건** | **두 겹으로 막힌다**: (i) V21 = 권한과 무관한 **원리적** 불가, (ii) V10 `RmProfilingAdminOnly: 1` (비-admin은 `ERR_NVGPUCTRPERM`), (iii) metric replay가 **측정 대상인 동시성 자체를 파괴** |
| **2b** | **`ncu` instanced metrics** (`sm__cycles_active` 등) | SM 인스턴스별 카운터 | 외부 | 커널 직렬화 | 이론상 per-SM 활동 | 정체성은 인스턴스 index로만, **green ctx 매핑 불명** | 경로 2와 동일 권한 벽 + 직렬화가 **PD 동시성 자체를 없앤다**(측정 대상 파괴) |
| **2c** | **`nsys` 트레이싱**(카운터 아님) | 커널 타임라인·스트림·중첩 | 외부 | 수 % (추정) | prefill/decode 커널이 **시간적으로** 겹치는지 | ★ **공간 정보 0.** SM id 없음 | 권한 벽은 낮으나(트레이싱은 보통 허용) 이 질문에 답하지 않음 |
| **3** | **green ctx API 자기보고** (`cuGreenCtxGetDevResource` / `cuCtxGetDevResource`) | 드라이버가 **할당했다고 보고하는** `smCount` | 무패치, ctypes로 libcuda 직접 호출(V6) | ~0 | "드라이버가 primary ctx에 몇 SM을 배정했다고 보고하는가" | ★★ **realized 점유가 아니라 또 하나의 target이다.** V8: **ID 집합 자체가 API에 없다** ⇒ disjoint 여부·정체성 불가 | **Stage 0 D108 오류의 재발 위험.** 이 값을 realized로 인용하면 그때와 같은 실패다 |
| **3b** | **`cuStreamGetGreenCtx` 스트림 소속 조회** | 스트림이 green ctx에 속하는가(NULL 여부) | 무패치, ctypes(V6/V7) | ~0 | §1.1의 "idx 0·N-1은 평범 스트림"을 **드라이버 층에서** 확인 | SM 개수·정체성 **아무것도** | 사실상 없음. **가장 싼 코드-사실 확인**(§10-P2) |
| **4** | **DCGM / `nvidia-smi`** | 디바이스 단위 SM 점유율(`DCGM_FI_PROF_SM_ACTIVE` 등) | 무패치 | 낮음 | 디바이스 전체 활동률 | ★ **role 구분 불가**(프로세스 하나·컨텍스트 다수), SM 정체성 불가, 시간 해상도 ~ms | **배제**: 우리가 묻는 건 "같은 프로세스 안 두 스트림의 SM 집합이 disjoint인가"인데 이 층엔 그 구분이 존재하지 않는다 |
| **5** | **타이밍 계단(occupancy staircase) 간접 추론** | 블록 수 N을 늘릴 때 실행시간이 `ceil(N/S)`로 계단지는 지점에서 **유효 SM 수** 역산 | 무패치 | 낮음 | 권한 0으로 "유효 SM 수"의 **추정치** | 정체성·disjoint 여부 불가, **추정치이지 판독이 아님** | ★ **`%smid`가 가능하면 채택하지 않는다.** 이건 다시 "두 노이즈 지표의 비교"로 미끄러지는 형태 = §11.3이 금지한 것 |

**표에서 즉시 나오는 결론 3개**

1. **경로 2/2b는 권한 이전에 원리로 죽었다** — **V21**(CUPTI 13.0.2 헤더에 `smId` 필드
   0건)이 확인됐고, metric replay는 동시성을 직렬화해 **측정 대상을 파괴**한다. 두 이유
   모두 권한과 무관하므로, **P3(E1)가 "권한 있음"으로 나와도 경로 2를 되살리지 않는다.**
   V10(`RmProfilingAdminOnly: 1`)은 세 번째 벽일 뿐이다.
2. **경로 3은 target이지 realized가 아니다**(V8). 단 **비대칭 프로브**로서의 가치는
   있다: `cuCtxGetDevResource(primary)`가 108 **미만**을 보고하면 그것만으로 잔여 차감이
   **확정**된다(반증 가능). 108을 보고해도 아무것도 확정되지 않는다. ⇒ **싸고 한
   방향으로만 결정적**. §10-P2에 넣는다.
3. **경로 4는 배제**. 근거는 해상도가 아니라 **범주**다 — role은 프로세스 내부의 스트림
   속성인데 이 층의 관측 단위는 디바이스/프로세스다.

---

## 4. 권고 설계 — 계측 **사다리** L0 → L1 (→ L2는 보류)

핵심 발상: **성능 지표를 비교하지 않는다. SM id 집합을 읽는다.**

### L0 — standalone census (엔진 패치 0줄, 권고 1순위)

한 프로세스 안에서 엔진의 부팅 순서를 **그대로 재현**한다:

```
[t0] 평범 torch.cuda.Stream 2개 생성            → census → S_pre
[t1] initialize_stream_groups()와 동일 호출로
     green ctx pair 2개 생성 (74,34) (54,54)     (pdmux_a100_smoke.yml 재현)
[t2] 평범 스트림 다시 census                     → S_post      (구멍 B)
[t3] green idx1의 prefill/decode 스트림 각각 census → S_p1, S_d1 (구멍 A·D)
[t4] idx1 두 스트림에 census를 **동시** launch    → 시간 겹침 구간의 S_p∩S_d
[t5] idx0·idxN-1 census                          → S_0, S_N-1  (구멍 A)
[t6] cuStreamGetGreenCtx / cuCtxGetDevResource 병기(경로 3b/3)
```

- **census 커널**(Triton, V1–V4로 컴파일 확인): 블록마다 `%smid`·`%nsmid`·`%globaltimer`
  t0/t1을 읽고, `spin_ns` 동안 상주한 뒤 기록. 그리드는 포화까지 스윕
  (`n_blocks ∈ {108, 216, 432, 864, ...}`).
- **배리어를 쓰지 않는다.** 상주 블록 전원 랑데부 배리어는 가용 SM이 가정보다 적으면
  **데드락**한다. 고정 시간 스핀 + 다중 launch union이 같은 커버리지를 안전하게 준다.
- 산출: `S`(집합), `|S|`, SM별 히트 수, 포화 곡선, 블록 체류 구간.

**왜 이게 L1보다 먼저인가**: 구멍 A·B는 **부팅 시 green ctx 생성이 무엇을 하는가**의
질문이고, 그 기전은 엔진 밖에서 문자 그대로 같은 API 호출로 재현된다. 엔진을 건드리기
전에 **E2(가상 id 여부)를 먼저 죽이거나 살려야** 한다 — E2가 거짓이면 L1·L2 전부 무의미.

### L1 — in-server boot census (최소 패치 1개)

- **부착**: `multiplexing_mixin.py:64-70` `init_pdmux`의 `initialize_stream_groups()`
  직후 1회 + cudagraph 캡처 완료 후 1회. env gate `PDMUX_SMID_CENSUS=<path>`,
  **미설정이면 import도 하지 않는다**(holb_probe의 기본-OFF 규율 그대로,
  `src/multiplex/holb_probe.py` 전문 참조).
- **서빙 루프에 코드 0줄.** 프로브는 벤치 시작 **이전**에만 돈다(`g2s_run.sbatch:247-249`
  전례, V19).
- **추가로 답하는 것**: (i) 실제 서버 프로세스(모델 가중치·KV 풀·TP group 존재)에서도
  L0 결과가 성립하는가, (ii) ★**R4 = cudagraph 재생이 SM 한정을 전달하는가**(E5).
  R4는 L0로는 못 한다 — 캡처된 그래프가 엔진의 것이어야 하기 때문.
- 패치 규율: `src/patches/pdmux_smid_census.patch`로 미러, `env/dev_tree_edits.md` 기록,
  `sync_engine_tree.sh` manifest에 반영. **G1-a 대기 중이므로 manifest drift를 만들지
  않도록, 이 패치는 별도 캠페인 매니페스트로 격리하고 기존 캠페인 재현에는 넣지 않는다.**

### L2 — 프로덕션 커널 태깅 (설계는 해 두되 **이번에 사지 않는다**)

- **부착점(모델 대칭, V16)**: decode `_selective_scan_update_kernel`
  (`mamba/ops/mamba_ssm.py:67-68`), prefill `_chunk_scan_fwd_kernel`
  (`mamba/ops/ssd_chunk_scan.py:19-20`). Zamba2 전용 추가: triton attention
  (`triton_ops/decode_attention.py:45`, `triton_ops/extend_attention.py:107/113/219`).
- **기록 자료구조**: 링버퍼가 아니라 **`uint32 counts[n_role][128]` per-SM 원자 카운터**.
  블록당 `atomicAdd` 1회. cudagraph 재생 하에서도 고정 포인터로 동작한다(V13/V14).
  경계마다(`multiplexing_mixin.py:1056-1057`, `:1073-1074`의 드레인 지점) 스냅샷 →
  **구간별 SM 히스토그램**. ⚠️ 스냅샷 케이던스는 이 프로젝트에서 이미 한 번 아티팩트를
  만든 축이다(§0 스냅샷 케이던스 건) ⇒ 케이던스를 아티팩트에 반드시 기록.
- **role 태그 배관 (2안, 둘 다 미검증 설계)**:
  (a) 그래프가 stream-idx별로 별도 캡처되므로(V13) **캡처 시점에 idx별 카운터 포인터를
  굽는다**; prefill은 eager라 인자로 넘기면 된다.
  (b) 그래프 선두에 `set_role` 커널을 함께 캡처해 device global을 세팅.
  → (a)가 단순하지만 모델 forward까지 인자를 관통시켜야 한다. **배관 리스크 = 이 설계에서
  가장 큰 미지수.**
- **커버리지 한계를 미리 자백한다**: GEMM(cuBLAS)·Granite attention(flashinfer)·
  causal_conv1d(sgl_kernel)는 계측 밖이다. ⇒ L2는 "role의 **전체** SM 점유"가 아니라
  "**계측된 커널들의** SM 점유"만 licence한다.

### 차선책 (권고가 막혔을 때)

1. **E2가 거짓(=`%smid`가 green ctx 안에서 가상화)** → 경로 1 전체가 죽는다. 차선은
   **경로 3의 비대칭 사용**: `cuCtxGetDevResource(primary)`가 108 미만이면 잔여 차감
   **확정**, 108이면 **미결**로 남긴다. 그리고 §1.1 캐비어트는 **그대로 유지**한다.
   (경로 5로 내려가지 않는다 — §6.)
2. **L1 패치가 correctness gate를 통과 못 함** → L0만으로 구멍 A·B를 다루고, 구멍 C(R4)는
   **미해결로 명시 등재**한다. 부분 결과를 전체로 승격하지 않는다.
3. **R4가 "전달 안 됨"으로 나옴** → 그건 성능 판정이 아니라 **코드 사실의 정정**이다.
   즉시 claims-auditor·doc-steward로 보내 §1-1·Gate 2-S 인용 문구 재검토를 개시한다.
   engine-porter는 거기서 멈춘다.

---

## 5. 결정량과 결정 규칙 **후보** (사전등록 초안 — 확정 아님)

전부 **물리량의 집합 술어**다. 두 노이즈 지표의 대소 비교가 아니다.

**표기**: `S(x)` = 대상 `x`에 launch한 census의 union SM id 집합. `D` = 이 디바이스에서
어떤 스트림으로도 관측 가능한 SM id의 전집합(R0에서 확립).

| ID | 질문 | 결정량 | 판정 규칙 후보 | 실패 시 |
|---|---|---|---|---|
| **R0** | `%smid`는 green ctx 안에서 **물리** id인가 (E2) | `S(green_p) ∩ S(green_d)`, `\|S(green_p) ∪ S(green_d)\|` | 상보 분할 `(74,34)`에서 **교집합 = ∅** ∧ **합집합 크기 = `\|D\|`** 이면 물리 id로 판정. 겹치거나 둘 다 0에서 시작하면 **가상 id** | ★ **즉시 정지.** R1–R4 전부 해석 불가. §4 차선책 1로 이동 |
| **R1** | green 분할이 전달되는가 (E6) | `\|S(green_p)\|`, `\|S(green_d)\|` | 코드 유래 기대 `(74, 34)`와 **정확히 일치**하는가. 불일치 시 관측치를 그대로 보고 | "공간 분할"이라는 이름 자체가 재검토 대상 |
| **R2** | "무분할"의 물리적 내용 (구멍 A) | `\|S(idx0_p)\|`, `\|S(idxN-1_d)\|` | `= \|D\|` 이면 무차감. `< \|D\|` 이면 차감 **확정**이고 그 크기는 `\|D\| − \|S\|` (SM 단위, **추정 아님**) | — |
| **R3** | 생성 자체가 차감하는가 (구멍 B) | `S_pre \ S_post` (같은 프로세스, 같은 평범 스트림) | 비어 있지 않으면 **차감 확정**. 비어 있으면 "이 조건에서 차감 없음" | — |
| **R4** | cudagraph 재생이 한정을 전달하는가 (구멍 C, E5) | green idx1의 decode 스트림에서 **그래프에 캡처된** census의 `S`, vs **eager** census의 `S` | 두 집합이 **같으면** 전달됨. 그래프 쪽이 `\|D\|` 로 커지면 **재생이 한정을 벗어남** | 운영점 해석 전면 재검토 사유 |
| **R5** | 동시 실행 시 겹치는가 | `%globaltimer` 구간이 겹치는 창에서 `\|S_p(t) ∩ S_d(t)\|` | 전 겹침 창에서 0이면 disjoint. >0이면 **시분할 공유 존재** | — |
| **D1** (진단) | 커버리지가 포화했는가 (E7) | `\|S\|` vs 블록 수 / 반복 수 | 마지막 2 스텝에서 `\|S\|` 불변이면 포화. **비포화면 어떤 "부재" 주장도 금지** | — |
| **D2** (진단) | 희소 히트 SM (게이트 #29의 정신) | SM별 최소 히트 수 | span/합계만 보고하지 말고 **최소 히트**를 병기 | — |

**의도적으로 넣지 않은 것**

- **arm 간 성능 비교 없음.** 위 어디에도 ITL·TTFT·goodput이 없다.
- **등가 검정(TOST) 없음.** 귀무 채택형 게이트는 이 저장소의 서명 오류다(G5·§6.2.0).
  R2/R3의 "차감 없음"은 등가 선언이 아니라 **집합 차가 공집합이라는 관측**이며, 그
  주장력의 한계는 D1(포화)로 **직접** 제한된다.
- **반복 n에 대한 유의성 검정 없음.** 반복은 **재현성 확인**용이다(집합이 부팅마다
  같은가). 집합이 부팅마다 다르면 그 자체를 보고한다.

---

## 6. ★ 이것이 "네 번째 게이트"가 아닌 이유 — 그리고 미끄러지는 지점

`PREREG_GATE2S §11.3`의 요점은 **통계량 교체가 아니라 측정 층 변경**이다. 아래가
그 조건을 만족한다는 논증이고, **어디서 미끄러질 수 있는지도 같이 적는다.**

**만족 근거**
1. **판독이 물리 식별자다.** `%smid`는 하드웨어 특수 레지스터이고, 결정량은 그
   식별자들의 **집합**이다. 두 arm의 지연 분포를 비교하는 구조가 아니다.
2. **기대값이 코드에서 온다.** `74`/`34`/`|D|`는 `pdmux_context.py:124-137`과
   `divide_sm()`의 출력이지 조정 가능한 자유 모수가 아니다.
3. **자기검증이 내장돼 있다.** R0은 외부 앵커 없이 스스로 판독 의미를 검정한다
   (상보 분할의 disjoint ∧ union = 전집합). §0.1이 문제 삼은 "외부 앵커 없는 등가
   마진"과 달리, R0은 **등가가 아니라 항등에 가까운 구조 검사**다.
4. **licence 위치에 게이트가 없다.** R0을 제외하면 어떤 규칙도 "통과해야 인용 가능"의
   자리에 있지 않다. R0은 통과 여부와 무관하게 결과를 **해석 불가로 만들 뿐** 어떤
   성능 주장도 열지 않는다(⇒ §11.3 트리거 1의 "하중 조건"에 걸리지 않는다).

**★ 미끄러지는 지점 (사전 자백)**
- **(S1)** R2에서 `|S| = |D|`가 나왔을 때 "무분할 = 108 확인"이라고 쓰면, 그건 D1
  포화가 뒷받침하는 범위를 넘는 순간 **귀무 채택형 서술**로 되돌아간다. 규칙:
  "포화 곡선이 `n_blocks = X`에서 평평해진 조건에서 관측된 집합 크기는 `|D|`였다"로만
  쓴다.
- **(S2)** L2로 가면 관측자 부하가 실재하므로 **거기서부터는 §6.2의 문제로 돌아온다.**
  L2를 사기 전에 이 문서를 다시 읽어야 한다.
- **(S3)** 경로 5(타이밍 계단)로 후퇴하면 **즉시 네 번째 게이트가 된다.** 채택 금지.

---

## 7. 관측자 부하 — 어떻게 bound하는가

전제: 이 프로젝트의 관측자 효과 게이트 **G5는 UNDETERMINED**(TOST 재채점, 72셀 중 초과
지지 0·등가 29·검정력부족 43)다. ⇒ **"프로브 무해"를 전제할 수 없고, 무해를 *증명*하는
통계 게이트를 새로 만들어서도 안 된다**(그게 G5와 같은 형태다).

### 7.1 L0 — 구조적으로 문제가 성립하지 않는다
프로브가 유일한 GPU 작업이고, 측정 대상은 "이 스트림에 launch된 커널이 어느 SM에
앉는가"다. 프로브는 교란원이 아니라 **트레이서 자신**이다. 엔진 코드 0줄.

### 7.2 L1 — 시간 격리로 0으로 만든다
프로브는 **부팅 시 1회**, 채점 벤치 **이전**에 돈다(V19의 전례). 서빙 루프에 코드가
없으므로 벤치 구간의 관측자 부하는 **정의상 0**이다. 남는 것은 (a) 부팅 시간 증가
(수십 ms, 아티팩트에 기록), (b) 프로브가 남긴 메모리/캐시 상태 — (b)를 없애려면
프로브 버퍼를 즉시 해제하고 `torch.cuda.empty_cache()`를 부르며, **그 호출 자체를
arm 대칭으로** 넣는다(env가 꺼져 있어도 부팅 경로가 같도록).

### 7.3 L2 — 통계가 아니라 **기전**으로 bound한다 (V5로 실현 가능 확인)
1. **컴파일타임 occupancy bound (GPU 0, 통계 0).** 대상 커널의 baseline/instrumented
   cubin을 CPU에서 생성해 `cuobjdump -res-usage`로 `REG/SHARED/LOCAL/STACK`을 비교한다
   (V5에서 장난감 커널로 실측: `REG:8 → REG:10`, SHARED/LOCAL/STACK 불변).
   이론 occupancy(블록/SM)는 `REG·warps·SHARED`의 결정적 함수이므로, **경계를 넘지
   않으면 SM당 상주 블록 수가 동일함이 산술로 확정**된다. 넘으면 그 커널은 계측 대상에서
   **뺀다**(임계를 조정하지 않는다).
2. **명령어 수 상한.** SASS에서 추가 명령 수를 세어 블록당 추가 사이클의 **상한**을
   제시한다(`nvdisasm`, CPU-only).
3. **대칭.** 계측은 **전 arm에 동일하게** 들어간다(telemetry 대칭 규율과 동일). 프로브가
   공통항이면 arm 간 대비에서 소거된다 — 단, **소거된다는 주장 자체를 통계로 증명하지
   않는다**. 소거는 설계상 보장이지 측정 결과가 아니다.
4. **금지**: "on/off paired TOST로 무해 입증" — G5 형태. 하지 않는다.
5. ⚠️ **잔여 자백**: 위 1–3은 *occupancy와 명령 수*를 bound하지 *메모리 시스템 간섭*
   (atomic 트래픽의 L2 압박)을 bound하지 않는다. L2를 살 때는 이 항목을 **미해결로**
   등재해야 한다.

---

## 8. ★★ 이 실험으로도 **닫히지 않는 것** (필수 절)

> 이 프로젝트는 게이트가 세 번 죽었다. 네 번째로 비싼 것을 사기 전에 payoff를 코드·원리로
> 확인하는 것이 방법론 게이트 #28이다. 아래는 그 확인 결과이며 **낙관 없이** 적는다.

1. **§1-1의 "PD 분리 자체 귀속"은 닫히지 않는다.** 이건 감사자가 이미 판정한 것이다
   (§0.0.B): `%smid`는 **전달 질문**("분할이 물리적으로 전달됐는가")에 답하고 §1-1은
   **효과 귀속 질문**이다. 게다가 2026-08-10 귀속 스코프 결론에 따르면 두 pdmux arm의
   성능-층 등가검정으로 그 몫을 닫으려면 **n ≈ 64**가 필요하다(현행 10). `%smid`는 그
   산술을 **한 자리도** 바꾸지 않는다.
2. **"SM 분할 자체"의 효과 크기는 여전히 분리되지 않는다.** 분할이 74/34로 전달됐음을
   보아도, Gate 2-S의 Δ가 SM 수 때문인지 selector·sticky·드레인 때문인지는 그대로 미분리다.
   §1.4가 보인 **정방향 arm 구조적 불가능**은 계측으로 해결되는 종류의 문제가 아니다.
3. **인용 가능 셀은 늘지 않는다** — §1.1 표에서 코드로 확인했다(`g2s_analyze.py:1157-1161`
   F-계열이 크기 인용을 통제하고, SM 라벨은 통제하지 않는다). 이걸 "늘어난다"고 쓰면
   G1-c에서 반증된 서술을 재생산하는 것이다.
4. **SM 집합 ≠ SM 처리량.** `%smid`는 **정체성**만 준다. SM이 집합에 들어 있어도 활용도가
   5%일 수 있다. "74 SM 전달됨"에서 "74/108 만큼의 계산 자원"으로 넘어가는 추론은 **금지**다.
5. **시분할은 부분적으로만 보인다.** green ctx가 공간 보장을 주더라도 primary context가
   무제한이면 같은 SM을 **시간적으로** 공유할 수 있다. R5(타임스탬프 겹침)가 이걸 보지만
   **마이크로 체제에서만**이고, 서빙 동작점으로의 이전은 금지된다(§6.2.1 rev4 캐비어트와
   같은 이유·같은 규칙).
6. **L2 없이는 서빙 동작점의 동시 점유를 말할 수 없다.** L0/L1이 licence하는 것은
   **가용성/전달**이지 **혼합 부하에서의 실제 점유**가 아니다.
7. **L2를 사더라도 커널 커버리지가 부분적이다** — GEMM(cuBLAS)·Granite attention
   (flashinfer, V17)·causal_conv1d(V18)는 계측 밖. "role이 점유한 SM 집합"이 아니라
   "**계측된 커널이** 점유한 SM 집합"이다.
8. **goodput·SLO·fused 대비 우열에 대해 아무것도 말하지 않는다.**
9. **R4가 "전달됨"으로 나와도 새로 열리는 성능 주장은 없다.** 열리는 건 캐비어트 하나가
   닫히는 것뿐이다. 반대로 R4가 "전달 안 됨"이면 그건 **기존 서술을 좁히는** 방향으로만
   작용한다(새 승리 주장이 아니라 정정).
10. **부팅 시점 측정은 정상상태 측정이 아니다.** L1 census는 부팅 직후 정지 상태다.
    장시간 서빙 중 파티션 상태가 표류하는지는 이 설계가 보지 않는다.

**⇒ 정직한 payoff 요약**: 이 실험은 **주장을 새로 열지 않고, 미검증 물리 가정 3–4개를
측정치로 바꾼다.** 그 가치는 "Stage 0 D108 거울상 위험의 제거"와 "cudagraph 전달 여부의
최초 측정"이지, §1-1 전진이 아니다. 이 문장에 동의할 수 없다면 이 실험을 사지 마라.

---

## 9. 비용 산정 (전부 §2.B-E9 = 추정. 스모크로 검증할 것)

| 항목 | 구현 공수 | GPU-hr | 배관 리스크 |
|---|---|---|---|
| **P1–P4 선행 프로브**(§10) | 반나절 | **≈ 0.05–0.15** | 낮음. P3는 초 단위 |
| **L0 standalone census** | 1일 (스크립트 1개, 엔진 0줄) | **≈ 0.2–0.3** (JIT 워밍 포함, 반복 n=5) | 낮음. 유일 미지수 = E2/E3 |
| **L1 in-server boot census** | 1–2일 (패치 ~80줄 + 하네스) | **≈ 0.3–0.6** (2모델 × 구성 4–5 × 부팅 5, 부팅당 2–4분) | 중. cudagraph 캡처 후 훅 시점, manifest 격리(G1-a 대기) |
| **L2 프로덕션 태깅** | **2–4일 + correctness gate 재통과** | **≈ 1–2** (축소 격자) | ★ **높음** — role 태그를 cudagraph에 굽는 배관, occupancy 재검증, Granite 커버리지 결손 |
| 합계 (L0+L1, 권고) | ≈ 2–3일 | **≈ 0.55–1.05** | — |

비교 기준: Gate 2-S 본 캠페인 = 6.35 GPU-hr, G1-c = 0.10 GPU-hr, 배관 스모크 = 0.11 GPU-hr.
⇒ **L0+L1은 Gate 2-S의 1/6 이하**이며, **L2가 비용의 대부분**이다. 그래서 §4는 L2를
분리해 보류한다.

⚠️ **방법론 게이트 #26 적용**: 하네스·스코어러가 신규 작성이고 예상 비용이 ≥1 GPU-hr가
되면(= L2 포함 시) **본 제출 전 n=1 배관 스모크 필수**.

---

## 10. 선행 조건과 **가장 싼 프로브**

순서대로. 앞이 실패하면 뒤는 의미 없다.

| ID | 확인 대상 | 비용 | 프로브 | 실패 시 |
|---|---|---|---|---|
| **P0** | Triton이 `%smid`를 낸다 | **0 (완료)** | V1–V4. 재현: 스크래치패드 `ptx_probe3.py` | — |
| **P0b** | 계측 레지스터 델타를 GPU 없이 잰다 | **0 (완료)** | V5. `cuobjdump -res-usage` | — |
| **P1** | ★ **E2/E3** — green ctx에서 `%smid`가 물리 id인가, `\|D\|`는 얼마인가 | **≈ 5분 GPU** | L0의 R0 단독 실행: `(74,34)` green pair 생성 → 양쪽 census → 교집합·합집합 | **경로 1 전면 사망.** §4 차선책 1 |
| **P2** | 드라이버 층 코드 사실 2건 | **≈ 1분 GPU** (커널 0개) | ctypes로 libcuda 직접 호출(V6): 각 스트림에 `cuStreamGetGreenCtx`(idx0/N-1이 NULL인가, V7) + green ctx 생성 전/후 `cuCtxGetDevResource(primary, SM)`의 `smCount` | `smCount < 108`이면 **차감 확정**(그것만으로 보고 가치 있음) |
| **P3** (선택) | **E1** — 컴퓨트 노드의 프로파일링 권한 | **≈ 초** | `srun -p amd_a100nv_8 -t 1 --gres=gpu:1 cat /proc/driver/nvidia/params \| grep -i restrict` — **제출은 experiment-runner 소관. 이 세션에서 실행하지 않았다** | ⚠️ **결정력 없음**: V21로 경로 2는 이미 원리적으로 죽었다. 기록용으로만 돌린다. **P1/P2를 이것 뒤로 미루지 마라** |
| **P4** | L2 대상 커널의 occupancy 여유 | **0 (CPU)** | 실제 `_selective_scan_update_kernel`/`_chunk_scan_fwd_kernel`을 캠페인 config로 baseline/instrumented 컴파일 → `cuobjdump -res-usage` 비교 | 경계 초과 커널은 계측 대상에서 제외 |
| **P5** | manifest 격리 | **0** | `sync_engine_tree.sh` manifest에 L1 패치를 넣기 전, G1-a 대기 상태 확인. 기존 캠페인 재현 경로와 분리 | drift 발생 시 캠페인 비교가능성 훼손 |

**가장 싼 결정적 프로브 = P1 + P2 (합쳐 GPU 10분 이내).** 이 둘이 §4 전체의 생사를
가른다. 나머지는 그 뒤에 산다.

---

## 11. 실패 모드 (사전 열거)

| # | 실패 | 징후 | 대응 (사전 지정) |
|---|---|---|---|
| F1 | `%smid` 가상화 (E2 거짓) | R0에서 두 green 스트림 집합이 겹치거나 둘 다 0 기점 | **즉시 중단.** 경로 1 사망 보고. 차선책 1 |
| F2 | SM id가 희소/비조밀 (E3) | `max(smid) > 107`, 또는 `\|D\| ≠ 108` | 오류 아님. **`0..107` 조밀 가정을 쓰는 모든 분석 코드를 금지**하고 `\|D\|`를 관측으로 정의 |
| F3 | 커버리지 비포화 (E7) | D1 곡선이 계속 상승 | **어떤 "SM 부재" 주장도 금지.** 블록 수·스핀 시간 상향 후 재측정 |
| F4 | 스핀 커널 데드락/워치독 | 커널 타임아웃, GPU 리셋 | 배리어 미사용(§4 L0) + `spin_ns` 상한(≤2ms) + 워치독. **랑데부 배리어 설계로 되돌아가지 않는다** |
| F5 | 부팅 census가 서버 상태 오염 | 이후 KV 풀·캡처 메모리 변화 | 프로브 버퍼 즉시 해제 + `KV Cache is allocated. #tokens:` 로그를 census on/off에서 **직접 대조**(§2.3-a 전례) |
| F6 | manifest drift | `sync_engine_tree.sh` manifest 변경 | P5. 별도 캠페인 매니페스트로 격리 |
| F7 | L2 role 태그가 그래프에 잘못 구워짐 | 한 role의 카운터에 양쪽 기록 | 양성대조: sticky ON(idx1 고정) 부팅에서 idx1 카운터만 증가해야 함 |
| F8 | 측정 실패를 결과로 라벨링 | 프로브 아티팩트 없음 → "차감 없음" | ★ 방법론 게이트 #21(7회 재발). **아티팩트 부재 = UNDETERMINED**, 절대 "없음"이 아니다. 분석 코드에 이 분기를 **먼저** 넣는다 |
| F9 | 분석 코드가 스스로 거짓 음성 | 빈 입력이 통과 경로로 샘 | 게이트 #21 7번째 재발 형태. `n_total == 0` 분기를 단위 테스트로 고정 |

---

## 12. 하지 말 것 (이 트랙에 고정)

- **공유 dev 트리 무단 수정 금지.** L1 패치는 승인·미러(`src/patches/`)·
  `env/dev_tree_edits.md` 기록·manifest 반영이 전부 끝난 뒤에만.
- **성능 주장 금지.** 이 트랙의 산출물은 SM 집합이지 지연·처리량이 아니다.
- **경로 5(타이밍 계단) 채택 금지** — §6-S3.
- **경로 3의 `smCount`를 realized로 인용 금지** — Stage 0 D108 오류의 재발이다.
  "드라이버 자기보고 target"으로만 표기.
- **on/off 등가검정으로 프로브 무해 입증 금지** — G5 형태(§7.3-4).
- **L2를 GPU correctness gate 통과 전에 기본값 ON 금지.**
- **`%smid` 결과를 §1-1 격차의 성분·분해로 서술 금지**(Gate 2-S 인용 제한과 동일 취지).
- **본 문서를 사전등록으로 인용 금지.** 사전등록은 claims-auditor 설계 감사를 거친
  별도 문서여야 한다.

---

## 13. 다음 액션 (제안 — 승인 필요)

1. **P1+P2 L0 최소본 작성**(engine-porter, 스크래치패드) → GPU 10분 이내 실행 요청.
   **R0 결과가 나오기 전에는 L1 패치를 쓰지 않는다.**
2. R0 통과 시에만: L0 전체(R1–R3, R5) → 결과를 claims-auditor에 **설계 감사**로 회부 →
   그 뒤 L1 사전등록 초안.
3. L2는 **위 2가 끝나고 별도 승인**을 받은 뒤에만 검토한다.
4. (선택) P3 — 기록용. 위 1–3을 지연시키지 않는다.

---

### 부록 A — 이 문서를 만들며 실제로 실행한 명령 (전부 CPU / 읽기)

- `cat /proc/driver/nvidia/params | grep -i restrict` → `RmProfilingAdminOnly: 1` (V10)
- `nm -D /lib64/libcuda.so.1 | grep -i "GreenCtx\|DevResource\|SmResource"` (V6)
- `/apps/cuda/13.0.2/include/cuda.h` 읽기 (V7·V8·V9)
- `triton.compile(..., target=GPUTarget("cuda", 80, 32))` — `%smid`/`%nsmid`/`%globaltimer`
  + 스핀 루프 (V1–V4). 산출물: 세션 스크래치패드
  `.../scratchpad/smid/{ptx_probe.py,ptx_probe3.py,census.ptx,dumpcubin.py}`
- `cuobjdump -res-usage {base,instr}.cubin` (V5)
- dev-tree/저장소 읽기 전용 grep (V12–V20)

**GPU 커널 실행 0건, dev-tree 쓰기 0건, git 조작 0건.**
