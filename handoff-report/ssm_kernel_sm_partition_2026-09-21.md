# SSM 커널 × green-context SM 분할 — 코드 사실 정리 (2026-09-21)

사용자 질문(2026-09-21): "ssm 커널의 경우에 smid로 실행하기 때문에, greencontext에서
SM 파티션을 한다고 하더라도 문제가 발생할 수 있다고 알고 있어. 현재 프로젝트에서는
이를 고려해 수정을 진행한 것 같은데 어떻게 진행된 건지 정리해봐. 그리고 이 부분이
실행에 영향을 미치는 부분들을 정리해"

> **지위**: 코드 사실 확인 + 기존 등재 결과의 재조합. **GPU 지출 0 · 새 측정 0 ·
> 새 성능 판정 0건.** 연구 결론·Claim 등급·게이트 불변. 조사는 2026-09-18에 로컬로
> 재구성한 dev tree(v0.5.10 태그 `1519acf3…` + sync + `devtree_manual_edits.patch`)의
> **실소스**에서 수행했다.

## 1. 결론 먼저

| 질문 | 답 |
|---|---|
| SSM 커널이 `%smid`(또는 SM 수)로 work를 분배하는가? | ★**일반론으로는 그런 구현이 실재하지만, 우리 실행 경로에서는 아니다.** 아래 §2 |
| 그래서 프로젝트가 SSM 커널을 "수정"했는가? | **아니다.** `%smid` 작업은 **커널 수정이 아니라 측정 도구**였다(§3). 실제로 한 모델측 수정은 별건(`forward_split_prefill`, §4) |
| green context SM 분할에 실제로 있는 구멍은 무엇인가? | **스트림 수준 제약이라는 점**이다 — 그 스트림에 launch된 것만 갇힌다(§5) |

## 2. ★"SSM 커널이 SM 수를 읽는다"는 어디서 참이고 어디서 거짓인가

사용자의 우려는 **외부 `mamba_ssm` 패키지에 대해서는 정확히 참이다**. 로컬 설치본
(`mamba_ssm` 2.3.1 계열) 실측:

```
mamba_ssm/ops/triton/ssd_chunk_state.py:825   sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
mamba_ssm/ops/triton/ssd_chunk_scan.py:1446   sm_count = ...
mamba_ssm/ops/triton/ssd_chunk_scan.py:1492   sm_count = ...
mamba_ssm/ops/triton/layernorm_gated.py:309   sm_count = ...
mamba_ssm/ops/triton/layer_norm.py:658        sm_count = ...
```

`torch.cuda.get_device_properties().multi_processor_count`는 **green context 안에서도
물리 전체 SM 수(A100=108)를 반환**한다 — green context는 스트림·컨텍스트 수준 제약이지
device property를 바꾸지 않는다. 따라서 이 코드 경로를 타면 **34 SM만 준 파티션에서도
커널이 108 기준으로 work를 나눠** 과대 분할·비효율이 생길 수 있다. 사용자가 말한
문제는 이것이다.

**그런데 우리는 이 경로를 타지 않는다** (실측, dev tree 전수 grep):

| 확인 | 결과 |
|---|---|
| sglang이 `mamba_ssm`/`causal_conv1d` 패키지를 import하는가 | **0건.** 자체 Triton 포팅(`srt/layers/attention/mamba/ops/`)과 `sgl_kernel`만 쓴다 (`causal_conv1d.py:12-17`) |
| sglang 자체 mamba ops에 `multi_processor_count`/`sm_count`/`num_sms`가 있는가 | ★**0건**(`srt/layers/attention/mamba/` 전체 grep 공백) |
| 그 커널들의 grid 결정 방식 | **데이터 크기 기반**. 예: `ssd_chunk_scan.py:474` `grid = lambda META: (triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(headdim, META["BLOCK_SIZE_N"]), ...)` — SM 수가 들어가지 않는다 |
| 엔진 전역에서 `%smid`를 읽는 코드 | **우리가 넣은 `srt/multiplex/green_readout.py`의 주석뿐.** SSM/attention 커널은 `%smid`를 읽지 않는다 |
| 우리 패치(`src/`)가 `mamba_ssm`을 쓰는가 | **아니다.** `src/configs/mamba2.py:76`·`src/models/mamba2.py:20,67`의 언급은 **native 체크포인트 weight key를 sglang 이름으로 remap**하는 주석이다(로딩 경로, 커널 아님) |
| **vLLM**(대조군 후보)은 어떤가 | 0.12.0 실측 — `model_executor/layers/mamba/`에 SM 수 의존 **0건**, 외부 `mamba_ssm` import **0건**. **자체 구현** |

⇒ **sglang·vLLM 둘 다 자체 Triton 포팅을 쓰고, 그 포팅에는 SM-수 의존이 없다.**
`venv_packages_2026-09-17.txt`에 `mamba_ssm==2.3.1`·`causal_conv1d==1.5.3.post1`이
핀으로 들어 있는 것은 **설치돼 있을 뿐 서빙 경로에서 쓰이지 않는다**(설치 ≠ 사용).

★**~~미확인~~ → 해소(§8, 같은 날)**: `sgl_kernel`의 `causal_conv1d_fwd`/`causal_conv1d_update`
내부는 dev tree clone에 **소스가 함께 들어 있었다**(`sgl-kernel/csrc/mamba/causal_conv1d.cu`).
검토 결과 **SM 수 의존 0건**이다. 상세 §8.

## 3. `%smid` 작업의 정체 — 커널 수정이 아니라 **계측**

프로젝트의 `%smid`는 "green context가 실제로 SM을 가뒀는가"를 **측정하는 도구**다.
설계는 `workspace/engine-port/reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`
(C-1), 실행은 두 단계다.

### R0 — SM census (job 889631, 2026-08-22, 0.0275 GPU-hr)
`results/smid_census/`. 엔진·모델·요청 없는 **별도 프로세스**에서 spin 커널을 띄우고
`%smid`를 읽었다. 판정:
- `%smid`는 green context를 건너는 **`GLOBALLY_CONSISTENT_LABEL`**이다.
- green 쌍이 **서로소**, union = 108 = `|D|`, `sizes=[74,34]` = target과 일치.
- ★★**R2(서술, 3스코프 필수)**: `D \ S_post = ∅` — **green context를 만드는 것 자체가
  primary 스트림이 도달하는 SM id 집합을 줄이지 않는다**(eager·idle 스트림·이 기판 한정).
- ★**라벨 상한**: `%smid`는 **물리 SM 인덱스로 확립되지 않았다.** 라벨 집합의 cardinality를
  컴퓨트 자원 비율로 환산하는 것은 **금지**. 아티팩트 결함 N1·N2로 **`.json`만 인용**.

### P0-A — cudagraph replay가 한정을 보존하는가 (job 890893, 2026-08-23, 0.054 GPU-hr)
`results/bcg_probe/`. 질문: `--enable-pdmux`에서 `cuda_graph_runner.py:811-816`이
**각 stream group의 green 스트림 위에** decode 그래프를 캡처하고
`multiplexing_mixin.py`가 `with torch.cuda.stream(decode_stream)` 안에서 replay하는데,
**green context의 SM 한정이 replay를 통과해 유지되는가**.
- 판정 **`CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`**, claims-auditor
  `CONFIRMED with conditions`(C1–C8).
- `(74,34)` 분할의 decode(34) 스트림에 캡처한 토이 그래프를 같은 스트림에서 replay →
  `%smid` 라벨 집합이 eager census의 34-라벨 집합과 **정확히 일치**(`Δ=∅`,
  33,480 block 관측, 집합 밖 0건, 라벨별 최소 히트 476).
- 부수 사실: **green 스트림 위 CUDA 그래프 캡처·replay가 이 기판에서 가능하다**
  (캡처 75/75) — 단 **단일 커널 노드·`pool=None`·`capture_error_mode="global"`** 한정이고
  **엔진의 캡처 경로에 대한 진술이 아니다**.
- ★**이 판정이 하지 않는 것**: 구멍 C를 닫지 않고, 성능 주장을 허가하지 않으며,
  **cudagraph-ON 서빙 운영점으로 전이되지 않는다**(엔진 없는 L0 토이).
- 서술 레그 2개(`capture_green→replay_plain`=34, `capture_plain→replay_green`=108)는
  **기전 해석 금지** — "한정이 캡처 시점에 박힌다"와 "`replay()`가 ambient 스트림이
  아닌 곳에 launch한다"를 **구분하지 못한다**(두 가설이 같은 두 수를 예측).

## 4. 실제로 한 모델측 수정 — `forward_split_prefill` (별건)

사용자가 "수정"이라고 짚은 것에 해당할 만한 실제 코드 변경은 `%smid`가 아니라 이것이다.
`sync_engine_tree.sh`가 install하는 모델 파일들(`src/models/{zamba2,nemotron_h,
falcon_h1,granitemoehybrid,mamba2}.py`)에 **PD-mux SPLIT_PREFILL을 hybrid 모델에서
가능하게 하는** `forward_split_prefill`이 들어 있다:

```python
# PATCH (engine-port P1.3): enable pdmux split-prefill for hybrid models.
# Mirrors dense forward_split_prefill but threads NemotronH's (hidden, residual)
# pair across split windows. mamba/attn/mlp layers all take the same signature.
```
(`src/models/nemotron_h.py:859-891`)

★핵심: 이 분할은 **레이어 축**(`split_interval = [start, end)` 레이어 구간)이지
**토큰 축이 아니다.** 각 레이어는 여전히 한 번에 실행되므로 **mamba의 conv/ssm state
연속성이 깨지지 않는다** — 윈도우 사이로 넘기는 것은 `(hidden_states, residual)` 쌍뿐이다.
SSM state가 문제가 되는 것은 토큰 축으로 쪼갤 때(chunked prefill)인데 그건 이 패치의
축이 아니다.

부수: NemotronH·Falcon-H1·Granite-4의 모델 파일은 **2026-09-13까지 수동 복사였고 해시
manifest에 없었다** ⇒ 그 모델로 서빙한 캠페인은 자기가 실제로 돌린 모델 구현의
provenance를 기록하지 못했다(`forward_split_prefill` 포함). 그때 sync + manifest로
들어왔다(당시 dev tree와 바이트 동일 확인 ⇒ 기존 트리에는 no-op, 재구성 트리에는 수리).

## 5. ★실행에 영향을 미치는 부분

### (a) green context는 **스트림 수준** 제약이다
가두는 대상은 **그 green 스트림에 launch된 커널**이다. R0 R2가 측정한 대로 **primary
스트림이 도달하는 SM 집합은 green ctx 생성으로 줄지 않는다.** 실행상의 함의:
- PD-mux가 `with torch.cuda.stream(decode_stream)` 안에서 forward를 도는 한, 그 안의
  **SSM 커널도 같은 스트림에 간다** — 별도 조치 없이 한정을 받는다.
- 반대로 **그 컨텍스트 밖에서 도는 것**(초기화, 일부 유틸, CPU 동기화 후 default
  스트림에 떨어지는 launch)은 **가둬지지 않는다**. 이것이 구조적 구멍이다.

### (b) 라벨을 자원량으로 읽으면 안 된다
`%smid` 라벨 집합 크기(34)를 "34/108의 컴퓨트를 받았다"로 환산하는 것은 R0가
명시적으로 금지한 추론이다. 이 프로젝트가 **pin을 target이 아니라 realized로 검증하라**는
교훈을 두 번 값비싸게 배운 지점과 같은 계열이다(Stage 0의 D108이 실은 16 SM,
λ0에서 λ\*(A)의 D44 점유가 실은 소수).

### (c) E2 사전등록은 realized green SM을 재지 않는다
2026-09-18 규칙층 감사가 등재한 **E2C-38**: `e2_sticky.sbatch`는
`PDMUX_GREEN_READOUT`을 설정하지 않으므로, R1이 `STICKY_REALIZES_A`를 내도 그것은
**`stream_index`가 2였다**는 뜻이지 **decode가 44 SM을 실제로 받았다**는 뜻이 아니다.
즉 (a)(b)의 구멍이 **현재 등록된 실험 안에 그대로 남아 있다**.

### (d) 기판을 옮길 때
대여 GPU 확보 후 B3(SM 입도 probe)에서 **`create_greenctx_stream_by_value`가 요청값을
실제로 주는지**를 `green_readout.py`/`%smid` census로 확인해야 한다. 코드 상수
(`get_arch_constraints`의 `(8,8)` 등)를 실측 대신 인용하지 말 것
(`gpu_rental_checklist_2026-09-18.md` §4).

### (e) ★vLLM 대조군에서 새로 볼 것은 없다 — 단 버전 의존이다
vLLM 0.12.0은 자체 mamba 구현이고 SM 수 의존이 **0건**이므로, "대조군 엔진이 SM 수를
잘못 읽어 불리해진다"는 교락은 **현재 버전에서는 성립하지 않는다.** 단 이것은
**버전에 붙은 사실**이다 — 0.28.0(또는 대여 시점 최신)으로 올리면 **같은 grep을 다시
돌려야 한다**(vLLM은 Mamba 경로를 활발히 고치는 중: prefix caching·ReplaySSM·
FlashInfer Mamba SSU). 어차피 vLLM에는 green context SM 분할이 없으므로 이 축이
대조군 비교를 오염시키지는 않는다.

### (f) 남은 미확인
1. `sgl_kernel` 바이너리(`causal_conv1d_fwd` 등) 내부의 SM 수 의존 — **소스 없음**.
2. P0-A는 **L0 토이**다. 엔진의 실제 캡처 경로(공유 memory pool, 다중 커널 노드,
   piecewise 그래프)에서 한정이 보존되는지는 **측정된 적 없다**.
3. 서빙 운영점에서 green 스트림 밖으로 새는 launch가 있는지 전수 확인된 바 없다.

## 6. 출처

**1차(이 세션 실소스 확인)**: dev tree `~/Experiments/KISTI/sglang_engine_dev/python/sglang`
(v0.5.10 `1519acf3…` + sync + patch) — `srt/layers/attention/mamba/**` 전수 grep,
`ops/ssd_chunk_scan.py:474`, `causal_conv1d.py:12-17`, 엔진 전역 `smid` grep ·
로컬 `mamba_ssm/ops/triton/{ssd_chunk_state.py:825, ssd_chunk_scan.py:1446,1492,
layernorm_gated.py:309, layer_norm.py:658}` · 로컬 `vllm 0.12.0`
`model_executor/layers/mamba/` grep · 저장소 `src/models/nemotron_h.py:859-891`,
`scripts/bootstrap/sync_engine_tree.sh`.

**2차(등재 문서 인용)**: `reports/CONSENSUS.md` rev43 문단(R0 판정) ·
`results/smid_census/PREREG_SMID_R0_2026-08-14.md:240-243`(라벨 상한) ·
`results/bcg_probe/{P0A_RESULT_890893_2026-08-23.md, PREREG_P0A_2026-08-22.md,
p0a_graph_sm_confinement.py}` · `reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md` ·
`VERDICT_e2_substrate_portability_2026-09-18.md`(E2C-38).

## 7. ★추가(같은 날) — "Triton이면 안전한가"는 잘못된 축이다

사용자 후속 질문에 답해 grid 정책을 코드 수준으로 분리했다.

**Triton/CUDA는 축이 아니다.** 문제가 있는 `mamba_ssm`도 **Triton 커널**이다
(`mamba_ssm/ops/**triton**/`). 같은 알고리즘의 Triton 구현 둘이 서로 다르게 동작한다.
실제 축은 **(A) grid 산정 정책**과 **(B) forward/backward** 두 개다.

### 7.1 문제가 발생하는 패턴 — persistent / SM-sized grid

**패턴 P1: grid가 곧 SM 수** (`mamba_ssm/ops/triton/layer_norm.py:658-668`, `_layer_norm_bwd`)
```python
sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
_dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)   # SM당 partial 버퍼
rows_per_program = math.ceil(M / sm_count)                                     # SM당 work 할당
grid = (sm_count,)                                                             # ★ grid ≡ SM 수
```
green ctx가 34 SM을 줘도 `get_device_properties`는 **108**을 반환한다 ⇒ 108 CTA가 34 SM에
3–4 wave로 실리고(persistent 가정 붕괴), `_dw`가 `(108,N)`으로 과대 할당되며, 뒤따르는
리덕션도 108행을 훑는다.

**패턴 P2: 분할 정도를 SM 수로 결정** (`mamba_ssm/ops/triton/ssd_chunk_state.py:825-830`,
`_chunk_state_bwd_db`)
```python
sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, ...)   # nsplits가 중간 버퍼 크기
grid_db = lambda META: (..., batch * nchunks, nsplits * ngroups)
...
dB = dB.sum(2)                                                    # nsplits 축 리덕션
```
SM 수를 크게 보면 **과도하게 쪼개고**, 중간 버퍼와 리덕션 비용이 함께 커진다.

### 7.2 문제가 발생하지 않는 패턴 — data-parallel grid

(`sglang/srt/layers/attention/mamba/ops/ssd_chunk_scan.py:474-479`, forward)
```python
grid = lambda META: (
    triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
    batch * nchunks if chunk_offsets is None else len(chunk_offsets),
    nheads,
)
```
같은 파일 계열의 `ssd_chunk_state.py:517-521`도 `triton.cdiv(headdim, …) * triton.cdiv(dstate, …)`,
`layernorm_gated.py:121`은 `grid = (M, ngroups)`. **셋 다 SM 수가 식에 들어가지 않는다.**

안전한 이유: grid가 **문제 크기**로만 정해지므로 CTA 총수가 가용 SM과 무관하게 옳다.
가용 SM이 34로 줄면 같은 CTA들이 더 많은 wave로 실릴 뿐이고, **정확성도 자원 가정도
깨지지 않는다**(느려질 뿐, 그리고 그 느려짐이야말로 우리가 측정하려는 양이다).

### 7.3 우리 경로는 이중으로 비껴간다

| 방어선 | 내용 |
|---|---|
| ① forward-only 포팅 | sglang `mamba/` 아래에 `_bwd`/`backward`/`autograd.Function` **0건**(grep). `mamba_ssm`의 `sm_count` 사용 **5곳 전부 `_bwd`**: `_chunk_state_bwd_db` · `_chunk_scan_bwd_dC` · `_chunk_scan_bwd_dcb` · `_layer_norm_bwd`×2 |
| ② data-parallel grid | 포팅된 forward 커널의 grid가 전부 §7.2 형태 |

⇒ 추론 서빙은 backward를 타지 않으므로, **설령 외부 `mamba_ssm`이 설치돼 있어도
`sm_count` 경로에 도달하지 않는다.** 그리고 sglang이 그 패키지를 import하지도 않는다(§2).

### 7.4 mamba 밖에서 SM 수를 읽는 곳 — 발화 조건 점검

| 위치 | 용도 | 우리 운영 조건에서 |
|---|---|---|
| `jit_kernel/gptq_marlin.py:75` · `moe_wna16_marlin.py:118` | 양자화 GEMM | **미발화** (bf16 비양자화) |
| `jit_kernel/all_reduce.py:129` `NUM_CTA = props.multi_processor_count` | custom all-reduce | **미발화** (단일 GPU, TP=1). ★**TP>1로 확장하면 발화하고, NUM_CTA가 전체 SM이 된다** |
| `srt/batch_overlap/operations_strategy.py:96,175,252` `deep_gemm_num_sms` | DeepSeek MoE + DeepEP + two-batch-overlap | **미발화** (해당 모델·기능 미사용) |
| `srt/multiplex/profile.py:39,131` `gpu_sm_count` | 우리 프로파일 부기 | 라벨 기록용 |

★ 즉 **현재 운영점(단일 GPU·TP=1·bf16·TBO 미사용)에서는 SM-수 의존 경로가 하나도
발화하지 않는다.** 이것은 **조건부 사실**이며, 조건이 바뀌면(특히 TP>1) 다시 봐야 한다.

### 7.5 그래서 일반화하면 안 되는 것

"Triton이니까 green context에서 안전하다"는 **틀린 일반화**다. 정확한 문장은:
**"우리가 타는 forward 경로의 커널들이 data-parallel grid를 쓰기 때문에 SM 수 오판의
영향을 받지 않는다"**이고, 이는 (i) 엔진 버전 (ii) 모델 (iii) 병렬화 설정 (iv) 양자화
여부에 붙은 조건부 사실이다. 엔진·vLLM 버전을 올리면 §2·§7.4의 grep을 다시 돌려야 한다.

## 8. ★추가(같은 날) — `sgl_kernel` 바이너리 커널 소스 검토 (§6 미확인 1번 해소)

`sgl_kernel`의 소스는 **따로 받을 필요가 없었다** — 2026-09-18에 `git clone --depth 1
--branch v0.5.10`로 받은 dev tree에 `sgl-kernel/`(4.0 MB)이 함께 들어 있다. 온라인
조달 불필요.

### 8.1 `causal_conv1d` CUDA 커널 — SM 수 의존 **0건**

`sgl-kernel/csrc/mamba/causal_conv1d.cu` (669행). `multiProcessorCount` ·
`getDeviceProperties` · `%smid` · persistent 패턴 **전부 0건**(grep). launch 2곳:

```cpp
// :500  causal_conv1d_fwd_launch(ConvParamsBase &params, cudaStream_t stream)
dim3 grid(params.batch, params.dim);                                  // :506  순수 데이터
kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);      // :521

// :647  causal_conv1d_update_launch(ConvParamsBase &params, cudaStream_t stream)
dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);    // :648  데이터
kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);              // :652
```

⇒ §7.2의 **data-parallel grid**에 해당한다. 즉 sglang의 mamba 경로는 **Triton 폴백이든
`sgl_kernel` CUDA 커널이든 양쪽 다 SM 수를 읽지 않는다.** 분기는
`mamba/causal_conv1d.py:15-23`의 `try: from sgl_kernel import causal_conv1d_fwd …
except: _HAS_SGL_KERNEL = False`이고, **어느 가지를 타도 결론이 같다.**

★**부수 확인(중요)**: 커널 스트림은 `at::cuda::getCurrentCUDAStream().stream()`
(`:194`, `:287`)에서 온다 = **ambient 스트림을 따른다**. 따라서 PD-mux가
`with torch.cuda.stream(decode_stream)` 안에서 호출하면 이 CUDA 커널도 **green 스트림에
launch되어 한정을 받는다**(§5(a)의 조건이 이 커널에 대해 충족됨을 소스로 확인).

### 8.2 `sgl-kernel` 전체에서 SM 수를 읽는 곳 — 5곳, 발화 조건

```
csrc/expert_specialization/es_sm100_mxfp8_blockscaled_launcher.cuh:106   hw_info.sm_count = …
csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh:393
        dim3 grid(getCurrentDeviceProperties()->multiProcessorCount * max_active_blocks_per_sm, 1, 1);
csrc/cutlass_extensions/gemm/cutlass_gemm_caller.cuh:45                  hw_info.sm_count = …
csrc/moe/fp8_blockwise_moe_kernel.cu:128                                 hw_info.sm_count = …
csrc/gemm/per_token_quant_fp8.cu:252                                     const int sm_count = …
```

| 위치 | 성격 | 우리 운영 조건에서 |
|---|---|---|
| `es_sm100_*` 2곳 | **sm100(Blackwell) 전용** + MXFP8 expert specialization. `:393`은 ★**완전한 persistent grid**(`SM수 × max_active_blocks_per_sm`) = §7.1 P1의 교과서형 | **미발화** — A100은 sm80이고 MXFP8도 아님 |
| `cutlass_gemm_caller.cuh:45` | CUTLASS `KernelHardwareInfo.sm_count` → persistent/stream-K **tile scheduler**에 전달 | **미발화** — 이 헤더를 쓰는 곳은 `fp8_blockwise_gemm_sm90_dispatch.cuh`·`fp8_blockwise_gemm_kernel.cu` **둘뿐**(FP8 blockwise, sm90). bf16 비양자화·A100에서는 도달 불가 |
| `fp8_blockwise_moe_kernel.cu:128` · `per_token_quant_fp8.cu:252` | FP8 양자화 경로 | **미발화** (bf16) |

⇒ **sgl-kernel 쪽에서도 현재 운영점에서 발화하는 SM-수 의존 경로는 0건이다.**
단 전부 **조건부**다 — FP8/MXFP8 양자화를 켜거나 Blackwell로 옮기면 위 경로들이 살아나고,
그때는 §7.1 P1 패턴이 green context와 정면으로 만난다.

### 8.3 ★★ 뿌리 — `get_sm_available()` 자체가 물리값을 반환한다

`sgl-kernel/python/sgl_kernel/spatial.py:44-63`:
```python
def get_sm_available(device_id: int = None) -> int:
    """Get the SMs available on the device."""
    device_props = torch.cuda.get_device_properties(device_id)
    sm_count = device_props.multi_processor_count      # ★ 물리 총수. green ctx를 모른다
    return sm_count
```

이 함수가 `pdmux_context.py:110`의 `total_sm_count = spatial.get_sm_available(gpu_id)`이고,
거기서 `SM_COUNTS`의 idx0 `(total,0)`·마지막 `(0,total)`이 만들어진다.

함의 셋:
1. **이름과 달리 "green context 안에서 쓸 수 있는 SM 수"가 아니다.** 어디서 불러도 108이다.
2. 따라서 `(0,108)` 같은 끝 라벨은 "decode가 108 SM을 받는다"는 **물리 주장이 아니라 부기**다
   — `PREREG_GATE2S §1.1`이 "부기 라벨"이라 못박고 rev4에서 물리 주장을 철회한 지점,
   그리고 Stage 0 D108·λ0 D44 점유 계열 교훈과 **같은 뿌리**다.
3. **"SM 수를 물리값으로 읽는다"는 문제는 우리 엔진의 green-context API 자체에도 있다.**
   다행히 그 값은 **파티션 설계 시점**(`divide_sm`, `SM_COUNTS` 부기)에만 쓰이고
   **커널 grid 산정에는 쓰이지 않는다.** 만약 어떤 커널이 이 값을 grid에 쓴다면 그 순간
   §7.1 P1이 된다.

### 8.4 갱신된 미확인 목록

§6의 1번은 해소됐다. 남은 것:
1. ~~`sgl_kernel` 바이너리 내부~~ → **해소**(§8.1, SM 의존 0건).
2. P0-A는 **L0 토이**다. 엔진의 실제 캡처 경로(공유 memory pool·다중 커널 노드·piecewise
   그래프)에서 한정이 보존되는지는 여전히 **미측정**.
3. 서빙 운영점에서 green 스트림 **밖으로 새는 launch**가 있는지 전수 확인된 바 없다
   (커널 각각은 ambient 스트림을 따르지만, ambient가 항상 green인지는 별개 문제다).
4. FP8/MXFP8·Blackwell·TP>1로 조건이 바뀌면 §7.4·§8.2를 다시 돌려야 한다.
