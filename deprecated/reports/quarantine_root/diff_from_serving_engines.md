# 기존 Serving Engine과 prefill-layer-alloc의 차이점 보고서

> 작성일: 2026-05-12  
> 비교 대상: vLLM v0.6+, SGLang v0.3+ vs prefill-layer-alloc (현재 브랜치)

---

## 목차

1. [개요 및 연구 목적의 차이](#1-개요-및-연구-목적의-차이)
2. [스케줄링 단위 및 실행 모델](#2-스케줄링-단위-및-실행-모델)
3. [SM 자원 관리 방식](#3-sm-자원-관리-방식)
4. [Hybrid 모델 처리 방식](#4-hybrid-모델-처리-방식)
5. [핵심 발견: SSM 커널 SM 분할 불가 문제](#5-핵심-발견-ssm-커널-sm-분할-불가-문제)
6. [캐시 및 상태 관리](#6-캐시-및-상태-관리)
7. [프로파일링 및 계측 인프라](#7-프로파일링-및-계측-인프라)
8. [종합 비교표](#8-종합-비교표)
9. [시사점 및 결론](#9-시사점-및-결론)

---

## 1. 개요 및 연구 목적의 차이

### 1.1 기존 Serving Engine의 목적

vLLM과 SGLang은 **프로덕션 LLM 서빙 시스템**이다. 이들의 설계 목표는 다음과 같다.

- 다수 요청의 동시 처리 (Continuous Batching)
- KV 캐시 메모리 효율화 (PagedAttention, RadixAttention)
- Throughput 최대화 (token/s)
- SLO 준수 (TTFT, TPOT)

두 엔진 모두 **GPU 전체를 하나의 단일 커널 실행 단위**로 보며, 레이어별 SM 할당을 의식적으로 제어하지 않는다. 스케줄러는 어떤 요청을 다음 forward pass에 포함시킬지 결정하지만, 그 forward pass 내부에서 레이어 별로 SM을 재분배하는 로직은 존재하지 않는다.

### 1.2 prefill-layer-alloc의 목적

이 프로젝트는 서빙 시스템이 아니라 **연구 검증 프레임워크**다. 핵심 연구 질문은 다음 세 가지다.

1. **SSM / Attention / MLP prefill 레이어는 각각 몇 개의 SM에서 수확체감이 시작되는가?**  
   → Stage 1: SM scaling curve 측정 (`stage1_sm_scaling/`)

2. **CUDA Green Contexts stream 전환 overhead는 레이어 실행 시간 대비 얼마인가?**  
   → Stage 2: 전환 비용 측정 (`stage2_overhead/`)

3. **Layer-wise / Step-adaptive / Fixed split 중 어느 SM 할당 전략이 최적인가?**  
   → Stage 3: Policy A/B/C 동시 실행 평가 (`stage3_hm_eval/`)

이 질문들은 기존 서빙 엔진이 전혀 다루지 않는 영역이다.

---

## 2. 스케줄링 단위 및 실행 모델

### 2.1 기존 엔진: 요청 단위 스케줄링

```
[vLLM Continuous Batching 루프]

Iteration N:
  scheduler.schedule() → {prefill 요청 X개, decode 요청 Y개}
  model_runner.execute_model(batch)
    └─ for layer in model.layers:
         layer.forward(hidden_states)  ← GPU 전체 SM 사용
  postprocess: KV 캐시 업데이트, 완료 요청 제거
```

- **스케줄링 단위**: 요청 (request)
- **실행 단위**: forward pass 전체 (레이어 루프)
- prefill과 decode가 같은 배치에 섞여 있더라도, **레이어 실행은 순차적** (레이어 0 → 1 → ... → N-1)
- 배치 내 요청 종류(prefill/decode)에 따라 SM 사용량을 달리하는 메커니즘 없음

### 2.2 prefill-layer-alloc: 레이어 경계 단위 인터리빙

```
[run_concurrent_eval.py의 루프 — MuxWise/BulletServe 방식 모사]

Iteration (decode_step):
  ① [Decode Phase]
     policy.on_decode()                    ← SM ratio 설정 (decode용)
     runner.run_ssm_layer(seq_len=1, ...)  ← decode step 1개 실행

  ② [Prefill Phase — 1 layer씩]
     policy.on_prefill_layer_start(layer_idx, layer_type)  ← SM ratio 전환
     runner.run_{ssm|attn}_layer(seq_len=prefill_len, ...) ← prefill 1 layer
     policy.on_prefill_layer_end(layer_idx, layer_type)
```

- **스케줄링 단위**: 레이어 (layer boundary)
- 한 decode step + 한 prefill layer를 번갈아 실행 → **GPU 시간 공유**
- 각 실행 단위마다 SM 비율을 독립적으로 지정 가능
- `PrefillState`가 레이어 인덱스를 추적하며 81개(Zamba2) 또는 44개(Falcon-H1) 레이어를 순차적으로 처리

### 2.3 차이의 의미

| 항목 | vLLM/SGLang | prefill-layer-alloc |
|------|-------------|---------------------|
| "동시 실행"의 의미 | prefill+decode가 같은 배치에서 연산 (같은 forward) | prefill과 decode가 레이어 단위로 번갈아 GPU 점유 |
| GPU idle time | Attention O(L²) 중 decode가 기다림 | decode step은 prefill layer 사이 빈틈에 삽입됨 |
| 레이어 타입 인식 | 스케줄러가 알지 못함 | `MODEL_LAYER_TYPES` 함수로 레이어별 타입 추적 |
| SM 재할당 | 없음 | 레이어 경계마다 `smctrl.set_sm_ratio()` 호출 |

---

## 3. SM 자원 관리 방식

### 3.1 기존 엔진: SM 관리 없음

vLLM과 SGLang은 CUDA 커널을 단순히 `torch.nn.functional` 또는 custom CUDA extension(FlashAttention, PagedAttention 등)을 통해 호출한다. 이 커널들은 **GPU의 모든 유휴 SM을 자동으로 점유**하며, 엔진 레벨에서의 SM 분할 또는 격리는 일어나지 않는다.

```python
# vLLM의 forward pass (개념적)
for layer in self.model.layers:
    hidden_states = layer(hidden_states, kv_cache, ...)
    # 커널은 GPU 전체 SM 사용; 동시 실행 경쟁 없음
```

### 3.2 prefill-layer-alloc: CUDA Green Contexts 기반 SM 파티셔닝

`SMController` (`src/smctrl/green_ctx_controller.py`)가 CUDA driver API를 직접 호출하여 SM 파티션별 Green Context와 ExternalStream을 사전 생성한다.

```python
# SMController 초기화 시 preset SM 수별 Green Context 생성
for n_sm in [14, 27, 40, 54, 68, 81, 94, 108]:   # A100 preset
    split_res = cuDevSmResourceSplitByCount(base_res, n_sm)
    desc      = cuDevResourceGenerateDesc(split_res)
    green_ctx = cuGreenCtxCreate(desc, device_id, flags)
    stream    = cuGreenCtxStreamCreate(green_ctx, CU_STREAM_NON_BLOCKING)
    self._contexts[n_sm] = (green_ctx, ExternalStream(stream))
```

실행 시점의 SM 전환:

```python
# set_sm_count(n) — O(log P) bisect, 드라이버 호출 없음
self._current_sm_count = nearest_preset(n_sm)

# 커널 실행 — SM 제한 적용
with torch.cuda.stream(smctrl.get_stream()):   # ← Green Context stream
    layer(hidden_states)
```

#### Green Contexts vs 기존 방식 비교

| 항목 | vLLM/SGLang | prefill-layer-alloc |
|------|-------------|---------------------|
| SM 할당 단위 | 없음 (전체 GPU) | 1 SM 단위 (GPC 정렬) |
| 동시 격리 | 불가 | 가능 (별도 Green Context + stream) |
| 전환 overhead | 해당 없음 | < 1 μs (CPU-side pointer 교체) |
| 필요 드라이버 | CUDA 임의 버전 | CUDA driver 550+ (CUDA 12.4+) |
| A100 CUDA 13.0 지원 | 해당 없음 | 완전 지원 |

#### 이전 libsmctrl 방식과의 차이 (역사적 맥락)

이 프로젝트는 초기에 `libsmctrl`(비공식 TPC 마스킹 ioctl)을 사용했으나, A100 CUDA 13.0에서 드라이버 내부 구조 변경으로 미지원 판정을 받아 Green Contexts로 전면 교체했다. Green Contexts는 하드웨어 수준 격리를 지원하므로 libsmctrl 대비 격리 신뢰도가 높고, TPC 단위가 아닌 SM 단위 제어가 가능하다.

---

## 4. Hybrid 모델 처리 방식

### 4.1 vLLM/SGLang의 Hybrid 모델 처리

vLLM은 Jamba, Zamba, Falcon-H1 등을 지원한다. 처리 방식은 다음과 같다.

- `JambaCacheManager` / `ZambaCacheManager` 등이 SSM state와 KV cache를 동시에 관리
- 각 레이어가 Attention인지 SSM인지는 모델 forward pass 내부에서 처리
- **스케줄러는 레이어 타입을 알지 못함**: 단순히 요청을 배치에 넣고 forward pass를 실행
- SSM layer이든 Attention layer이든 SM 할당 차이 없이 GPU 전체 사용

```python
# vLLM Zamba 모델 forward (개념적)
for i, layer in enumerate(self.model.layers):
    if i in HYBRID_LAYER_IDS:
        # SSM + Attention 모두 실행 → 두 캐시에 접근
        hidden = layer.ssm_forward(hidden, ssm_state_cache[i])
        hidden = layer.attn_forward(hidden, kv_cache[i])
    else:
        # Pure SSM
        hidden = layer.ssm_forward(hidden, ssm_state_cache[i])
    # ← 어느 레이어든 GPU 전체 SM이 사용됨
```

### 4.2 prefill-layer-alloc의 Hybrid 모델 처리

레이어별 타입을 사전에 인코딩하여 실행 전에 SM 할당을 조정한다.

```python
# configs/models.yaml에서 레이어 구성 명시
# zamba2: hybrid_layer_ids: [6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77]

# run_concurrent_eval.py
MODEL_LAYER_TYPES = {
    "zamba2": lambda i: "attn" if i in _ZAMBA2_HYBRID_IDS else "ssm",
    "falcon_h1": lambda i: "ssm",  # 모든 레이어 SSM+Attn 병렬 → SSM 병목
}

# 레이어 타입에 따라 SM 비율 결정 (Policy C)
SSM_PREFILL_RATIO  = 0.70  # SSM: compute-bound → 더 많은 SM
ATTN_PREFILL_RATIO = 0.40  # Attn: compute-bound, tile 독립 → 적은 SM도 유효
MLP_PREFILL_RATIO  = 0.50
```

Zamba2-7B (81 layers: 68 pure SSM + 13 hybrid) 기준 Policy C 동작:

```
Layer 0–5   (SSM):   SM 70% prefill + 30% decode
Layer 6     (Attn):  SM 40% prefill + 60% decode  ← SM 비율 전환
Layer 7–10  (SSM):   SM 70% prefill + 30% decode  ← 재전환
Layer 11    (Attn):  SM 40% prefill + 60% decode
...
```

### 4.3 레이어 타입 인식 수준 비교

| 측면 | vLLM/SGLang | prefill-layer-alloc |
|------|-------------|---------------------|
| 레이어 타입 메타데이터 | 모델 내부에만 존재 (Python class) | `configs/models.yaml` + `MODEL_LAYER_TYPES` 함수로 스케줄러 계층에 노출 |
| 레이어 타입에 따른 자원 차별화 | 없음 | SM 비율, decode/prefill 분리 비율 |
| 레이어 수준 프로파일링 | 없음 (요청/배치 단위) | SM 수별 latency curve 측정 (`LayerRunner`) |
| Hybrid layer의 SSM/Attn 분리 측정 | 불가 | Zamba2 hybrid layer에서 SSM만, Attn만 독립 측정 가능 |

---

## 5. 핵심 발견: SSM 커널 SM 분할 불가 문제

이 절은 기존 서빙 엔진이 전혀 다루지 않는 `prefill-layer-alloc`의 고유한 발견이다.

### 5.1 mamba_chunk_scan_combined의 Cooperative Barrier 문제

`mamba_chunk_scan_combined` (Triton SSD 커널)은 청크 간 state를 전달하기 위해 `grid.sync()` barrier를 사용한다.

```
[SSD parallel scan 실행 구조]

Chunk 0 처리 Block Group ──┐
Chunk 1 처리 Block Group ──┤── grid.sync() ──→ inter-chunk state 병합
Chunk K 처리 Block Group ──┘
```

`grid.sync()`는 **모든 thread block이 동시에 active해야** 완료된다. Green Context로 SM을 제한하면 일부 block이 SM 부족으로 scheduled-out → 다른 block이 barrier에서 영구 대기 → deadlock 또는 illegal memory access.

### 5.2 실험적 증거

Stage 1 SSM sweep 실행 시 관찰된 결과:

```
sm=14, seq=8192, bs=32 → CUDA error: illegal memory access  ← 첫 발생
sm=14, seq=8192, bs=64 → CUDA error: illegal memory access  ← context 오염
sm=27, seq=512,  bs=1  → CUDA error: illegal memory access  ← 연쇄 실패
```

CUDA context가 오염된 후 단순한 config도 실패하는 패턴은 커널 자체의 버그가 아닌 barrier deadlock으로 인한 context 손상임을 보여준다.

Wave model 검증 (n_blocks = 65,536):

| config | 측정 latency | 이론 비율 | 실측 비율 |
|--------|-------------|---------|---------|
| seq=4096, bs=64 | 2909.5 ms | 1.000 | **1.000** |
| seq=8192, bs=32 | 2908.5 ms | 1.000 | **1.000** |

오차 0.03% — latency가 seq_len/batch 개별 값이 아닌 `n_blocks = batch × seq_len ÷ chunk_size`에만 의존함. 커널이 cooperative wave-serial 구조임을 확인.

### 5.3 Attention 커널과의 대비

FlashAttention (FlashInfer `BatchPrefillWithRaggedKVCacheWrapper`)은 QK^T 행렬을 독립 타일로 분할하며 inter-block 동기화가 없다.

```
Stage 1 Attention sweep 결과:
측정 SM 레벨: [14, 27, 40, 54, 68, 81, 94, 108]
CUDA 에러: 없음 (전 레벨 정상 완료)
```

| SM | seq=4096 bs=4 latency | 개선율 |
|----|----------------------|-------|
| 14 | 197.4 ms | — |
| 27 | 99.6 ms | +49.6% |
| 54 | 52.2 ms | +25.4% |
| 108 | 28.9 ms | +9.9% |

Attention은 SM 수에 따라 단조 개선. Green Context 격리가 정상 동작.

### 5.4 서빙 엔진에 대한 함의

| 커널 | Green Context SM 분할 | 이유 |
|------|----------------------|------|
| mamba_chunk_scan_combined (Triton SSD) | **불가** | cooperative `grid.sync()` — 전체 SM 독점 필요 |
| FlashAttention / FlashInfer | **가능** | 독립 타일, inter-block sync 없음 |
| cuBLAS GEMM (MLP) | **가능** | 표준 GEMM, cooperative 아님 |

**현재 vLLM/SGLang은 이 제약을 인식하지 못한 채 모든 레이어를 동일하게 처리한다.** 미래에 SM-level co-execution을 구현하려면, SSM 레이어에서는 decode와의 SM 공유가 이 방식으로는 불가능하다는 제약을 고려해야 한다.

---

## 6. 캐시 및 상태 관리

### 6.1 vLLM/SGLang의 캐시 관리

| 레이어 타입 | 캐시 | 관리 방식 |
|------------|------|---------|
| Attention | KV Cache (PagedAttention 블록) | `BlockManager` / `RadixAttention` |
| SSM | SSM State (conv_state + ssm_state) | `MambaCacheManager` (슬롯 단위) |
| Hybrid | 두 캐시 동시 관리 | `JambaCacheManager` 등 |

이 캐시들은 서빙 정확성을 위한 것이며, SM 자원 최적화와는 무관하다.

### 6.2 prefill-layer-alloc의 상태 관리

이 프로젝트는 서빙 정확성을 위한 KV/SSM 캐시를 구현하지 않는다. 대신 **레이어 실행 latency 측정**을 위한 캐시 구조를 사용한다.

```python
# LayerRunner의 레이어 입력 캐시 (재사용으로 allocation overhead 제거)
self._ssm_cache: dict  # (model, batch, seq_len) → (layer, hidden_states, bytes)
self._attn_cache: dict  # (model, batch, seq_len, ctx_len) → (attn_fn, bytes)
```

목적이 다르다:
- vLLM/SGLang의 캐시: 이전 forward pass의 계산 결과 저장 (정확성)
- prefill-layer-alloc의 캐시: 반복 측정 시 입력 텐서 재사용 (측정 공정성)

`PrefillState`는 현재 prefill 중인 요청의 레이어 진행 상태를 추적하지만, 실제 SSM state를 레이어 간 전달하지 않는다 — prefill 처리량 및 TTFT 측정이 목적이기 때문이다.

---

## 7. 프로파일링 및 계측 인프라

### 7.1 기존 엔진의 프로파일링

vLLM/SGLang은 운영용 메트릭(요청/s, TTFT p50/p99, TPOT p50/p99)을 제공하지만, **레이어별 SM 활용도**나 **SM saturation point**를 측정하는 기능은 없다.

### 7.2 prefill-layer-alloc의 계측 스택

레이어 수준의 하드웨어 특성 측정에 특화된 도구들을 포함한다.

```
src/profiling/
├── metrics.py          ← LatencyMeter (CUDA event 기반), BandwidthEstimator
├── ncu_runner.py       ← NCU 자동화 (wave quantization, SM occupancy, BW 분석)
├── nvtx_markers.py     ← NVTX range 마킹 (Nsight Systems 연동)
├── wave_estimator.py   ← Wave model 합성 (SM 스케일링 이론값 계산)
├── cupti_monitor.py    ← CUPTI 기반 SM utilization 수집
└── nvml_monitor.py     ← NVMLMonitor (백그라운드 GPU 메트릭)
```

- **`LatencyMeter`**: `torch.cuda.Event`로 커널 수준 latency 측정 (host-side timer보다 정밀)
- **`BandwidthEstimator`**: 레이어별 이론 메모리 접근량(`ssm_bytes`, `attn_bytes`, `mlp_bytes`)으로 HBM 활용률 계산
- **`wave_estimator.py`**: SSM 커널이 SM 분할 불가이므로, `n_waves(sm_k) = ⌈n_blocks / sm_k⌉` 공식으로 SM 스케일링을 이론적으로 합성 → ncu 실측값과 0.03% 오차

#### NCU 프로파일링 항목

```python
# stage1_sm_scaling/run_ncu_profile.py
NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM 처리율
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",    # L2 캐시 라인
    "dram__bytes_read.sum",                              # HBM 읽기 바이트
    "launch__waves_per_multiprocessor",                  # wave 수 (SM당)
]
```

이 수준의 하드웨어 계측은 vLLM/SGLang에 존재하지 않으며, serving 최적화 정책을 설계하기 위한 근거 데이터 수집에 목적이 있다.

---

## 8. 종합 비교표

### 8.1 아키텍처 레벨 비교

| 항목 | vLLM | SGLang | prefill-layer-alloc |
|------|------|--------|---------------------|
| **용도** | 프로덕션 서빙 | 프로덕션 서빙 | 연구 검증 프레임워크 |
| **스케줄링 단위** | 요청 (request) | 요청 (request) | 레이어 경계 (layer boundary) |
| **SM 파티셔닝** | 없음 | 없음 | CUDA Green Contexts |
| **Prefill+Decode 동시성** | 같은 배치 내 sequential | 같은 배치 내 sequential | 레이어 단위 인터리빙 |
| **레이어 타입 인식 (스케줄러)** | 없음 | 없음 | 있음 (SSM/Attn/MLP) |
| **KV/SSM 캐시** | 완전 구현 | 완전 구현 | 미구현 (측정 목적 캐시만) |
| **Hybrid 모델 지원** | Jamba, Zamba, Falcon-H1 등 | Jamba 등 | Zamba2, Falcon-H1 (측정용) |
| **대상 GPU** | 범용 | 범용 | A100/H100 (Green Contexts 필요) |

### 8.2 기술 요소별 비교

| 기술 요소 | vLLM/SGLang | prefill-layer-alloc |
|----------|-------------|---------------------|
| SM 제어 API | 없음 | `cuGreenCtxCreate`, `cuGreenCtxStreamCreate` |
| SM 전환 overhead | 해당 없음 | < 1 μs (stream pointer 교체) |
| SSM prefill 커널 | `mamba_chunk_scan_combined` (전체 SM) | 동일 커널 + SM sweep 측정 (Wave model fallback) |
| Attention prefill 커널 | FlashAttention-2/3 | FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` |
| Decode 커널 | `selective_state_update`, PagedAttention | `selective_state_update` (측정용 seq_len=1) |
| 배치 스케줄링 | Continuous batching (vLLM), RadixAttention (SGLang) | 고정 decode_batch + prefill queue |
| SM 할당 정책 | 없음 | Policy A (고정), B (step-adaptive), C (layer-wise) |
| 프로파일링 | 요청 수준 metrics | NCU + CUPTI + NVTX + NVML 레이어 수준 |

### 8.3 정책별 vLLM/SGLang 대응 관계

| prefill-layer-alloc Policy | vLLM/SGLang 대응 동작 |
|---------------------------|----------------------|
| **Policy A (Baseline)**: 고정 40/60 분할 | vLLM의 현재 동작에 가장 가까움. 단, vLLM은 prefill/decode 비율을 명시적으로 분리하지 않음 |
| **Policy B (Step-adaptive)**: 모델 SSM 비율 기반 step-level 조정 | vLLM/SGLang에 없음. SSM-heavy 모델(Zamba2)에서 prefill SM을 더 많이 확보하는 새로운 전략 |
| **Policy C (Layer-wise)**: 레이어 경계마다 SM 재할당 | vLLM/SGLang에 없음. SSM 레이어에서 SM 분할 불가 제약 때문에 실제로는 Attention 레이어에서만 효과 |

---

## 9. 시사점 및 결론

### 9.1 기존 서빙 엔진이 다루지 않는 문제들

이 프로젝트는 다음 두 가지를 발견 및 정량화했으며, 이는 vLLM/SGLang의 설계에서 전혀 고려되지 않은 영역이다.

**① SSM 커널의 SM 분할 불가능성**

`mamba_chunk_scan_combined`는 cooperative `grid.sync()` barrier를 사용하므로, 어떤 SM 파티셔닝 메커니즘을 사용하든 전체 SM을 독점해야 한다. 이는 SSM 레이어 실행 중에는 decode와의 SM 공유가 근본적으로 불가능함을 의미한다. 미래에 vLLM/SGLang이 SM-level co-scheduling을 구현하려 한다면, SSM 레이어에 대한 특수 처리가 필요하다.

**② SM Saturation이 레이어 타입별로 다름**

- SSM: wave-serial 구조로 SM에 선형 비례 → 포화점 없음, SM 감소 = 선형 slowdown
- Attention: 독립 타일 구조로 SM 증가에 따라 단조 개선되나, 108 SM에서도 ~10% 개선 여지 존재 (포화 미달)
- MLP (GEMM): compute/BW bound 전환점 존재

이 차이를 알면, Hybrid 모델에서 Attention 레이어 실행 시 decode 스트림에 40~60%의 SM을 할당할 수 있다는 근거가 생긴다.

### 9.2 정리

```
┌──────────────────────────────────────────────────────────────────────┐
│                핵심 차이 3줄 요약                                      │
├──────────────────────────────────────────────────────────────────────┤
│ 1. vLLM/SGLang은 GPU를 블랙박스로 보고 레이어별 SM을 제어하지 않는다.  │
│    prefill-layer-alloc은 레이어 경계마다 CUDA Green Contexts로         │
│    SM 비율을 전환하며 prefill과 decode의 GPU 자원을 분리한다.          │
│                                                                      │
│ 2. 이 프로젝트의 핵심 발견은 SSM 커널(Triton SSD)이 cooperative        │
│    grid.sync()로 인해 SM 분할이 불가능하다는 것이다.                   │
│    Attention(FlashInfer)과 MLP(cuBLAS)는 SM 분할이 가능하다.           │
│                                                                      │
│ 3. 따라서 Zamba2 같은 SSM-heavy Hybrid 모델에서 Policy C의 실질적      │
│    이득은 전체 81 레이어 중 13개의 Attention 레이어에서만 발생한다.     │
│    나머지 68개 SSM 레이어에서는 전체 SM을 독점 사용해야 한다.          │
└──────────────────────────────────────────────────────────────────────┘
```

---

*참고 파일: `src/smctrl/green_ctx_controller.py`, `src/models/layer_runner.py`, `stage3_hm_eval/policy_*.py`, `reports/ssm_sm_partitioning_analysis.md`, `green_ctx_migration_report.md`*
