# vLLM · SGLang에서의 Mamba 및 Hybrid 모델 서빙 기술 보고서

> 작성일: 2026-05-12

---

## 목차

1. [배경: Mamba / SSM 아키텍처 개요](#1-배경)
2. [핵심 CUDA 커널](#2-핵심-cuda-커널)
3. [vLLM의 Mamba 서빙](#3-vllm의-mamba-서빙)
4. [SGLang의 Mamba 서빙](#4-sglang의-mamba-서빙)
5. [Hybrid 모델 (Mamba + Attention) 서빙](#5-hybrid-모델-서빙)
6. [Transformer 서빙과의 핵심 차이점 비교](#6-transformer와의-차이점)
7. [성능 특성 및 트레이드오프](#7-성능-특성)
8. [요약](#8-요약)

---

## 1. 배경

### 1.1 SSM (State Space Model) / Mamba 아키텍처

Mamba는 선형 RNN의 일종인 Structured State Space Model(S4)을 기반으로, **입력 의존적(selection mechanism)**인 상태 전이를 추가한 모델이다. 핵심 연산은 다음과 같다.

```
h_t = A(x_t) · h_{t-1} + B(x_t) · x_t     (SSM recurrence)
y_t = C(x_t) · h_t                           (output projection)
```

여기서 `A`, `B`, `C`가 입력 `x_t`에 따라 동적으로 결정된다는 점이 고전 S4와의 차이다.

### 1.2 Mamba1 vs Mamba2

| 항목 | Mamba1 | Mamba2 |
|------|--------|--------|
| 알고리즘 | Selective Scan (SISO per channel) | SSD (State Space Duality, structured masked attention) |
| Head 구조 | 없음 (채널 독립) | Multi-head SSM |
| 병렬화 | 시퀀스 방향 sequential | Chunk 단위 병렬 처리 가능 |
| 주요 커널 | `selective_scan_fwd` | `mamba_chunk_scan_combined` |

### 1.3 Hybrid 모델 현황

순수 Mamba 모델 외에, Attention 레이어와 Mamba 레이어를 혼합한 모델들이 등장하고 있다.

| 모델 | 개발사 | 구성 |
|------|--------|------|
| Jamba | AI21 Labs | Mamba + MoE + Attention (1:7 비율) |
| Zamba | Zyphra | 공유 Attention + Mamba 블록 |
| Falcon-H1 | TII | Mamba2 + GQA Attention 혼합 |
| NVIDIA Hymba | NVIDIA | Hybrid SSM-Attention |

---

## 2. 핵심 CUDA 커널

### 2.1 causal-conv1d 패키지

Mamba 입력 전처리에 사용되는 1D depthwise convolution 커널이다.

```
causal_conv1d_fwd(x, weight, bias, activation)
causal_conv1d_update(x, conv_state, weight, bias, activation)
```

- **`causal_conv1d_fwd`**: Prefill 단계에서 길이 L인 시퀀스 전체를 한 번에 처리. CUDA warp-level 병렬화.
- **`causal_conv1d_update`**: Decode 단계에서 단일 토큰 처리. `conv_state` (슬라이딩 윈도우 버퍼, shape `[B, D, d_conv]`)를 in-place로 갱신.

### 2.2 mamba-ssm 패키지 — Mamba1 커널

```
selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
selective_state_update(ssm_state, x, dt, A, B, C, D, z, dt_bias, dt_softplus)
```

- **`selective_scan_fwd`** (prefill): parallel scan 알고리즘으로 시퀀스 전체를 병렬 처리. Blelloch-style prefix scan을 CUDA 커널 내부에서 수행.
- **`selective_state_update`** (decode): 이미 계산된 `ssm_state`에 단일 토큰 정보를 recurrent하게 합산. 매우 경량 연산 (`O(d_state × d_inner)`).

### 2.3 mamba-ssm 패키지 — Mamba2 커널 (SSD)

```
mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, ...)
chunk_state_varlen(...)
```

- SSD 알고리즘: 시퀀스를 `chunk_size` 단위로 분할하여 chunk 내부는 행렬 연산(Flash Attention과 유사)으로 처리하고, chunk 간 전파는 recurrence로 처리.
- Chunk 크기가 클수록 GPU 활용도가 높아지지만 메모리 사용량 증가.
- Varlen 변형(`chunk_state_varlen`)은 배치 내 서로 다른 시퀀스 길이를 처리하는 데 사용.

### 2.1 Triton 커널 (SGLang 및 커스텀 구현)

SGLang은 일부 Mamba 연산에 Triton 커널을 활용하여 CUDA 커널 대비 유연성을 확보한다.

```python
# SGLang triton-based selective scan (예시)
@triton.jit
def selective_scan_kernel(u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, ...):
    # block 단위 parallel prefix scan
    ...
```

---

## 3. vLLM의 Mamba 서빙

### 3.1 지원 모델 및 관련 파일

| 파일 경로 | 역할 |
|-----------|------|
| `vllm/model_executor/models/mamba.py` | Mamba1 pure 모델 |
| `vllm/model_executor/models/mamba2.py` | Mamba2 pure 모델 |
| `vllm/model_executor/models/jamba.py` | Jamba hybrid 모델 |
| `vllm/model_executor/models/zamba.py` | Zamba hybrid 모델 |
| `vllm/worker/mamba_model_runner.py` | Mamba 전용 모델 러너 |
| `vllm/attention/backends/mamba_attn.py` | Mamba 레이어용 "attention" 백엔드 |

### 3.2 SSM State Cache 관리: `MambaCacheManager`

Transformer의 KV 캐시와 달리, Mamba는 **고정 크기의 SSM state**를 시퀀스별로 유지한다.

```
Conv State:  shape [num_seqs, d_inner, d_conv]      ← 슬라이딩 윈도우
SSM State:   shape [num_seqs, num_heads, d_head, d_state]  ← 압축된 문맥
```

`MambaCacheManager`는 다음 역할을 담당한다:

1. **슬롯 할당 (Slot Allocation)**: 각 sequence_id에 대해 cache 슬롯 인덱스를 발급. PagedAttention의 블록 할당과 유사하지만, 블록이 아닌 단일 슬롯 단위.
2. **상태 복사**: 선점(preemption) 발생 시 CPU로 SSM state를 offload, 재스케줄링 시 GPU로 reload.
3. **배치 인덱싱**: forward pass에서 `cache_indices` 텐서를 통해 배치 내 각 샘플이 자신의 슬롯 상태를 읽고 씀.

```python
# vLLM MambaCacheManager 핵심 로직 (개념적 표현)
class MambaCacheManager:
    def __init__(self, layer_num, d_model_args, max_batch):
        self.conv_states = torch.zeros(
            [layer_num, max_batch, d_inner, d_conv], device='cuda'
        )
        self.ssm_states = torch.zeros(
            [layer_num, max_batch, num_heads, d_head, d_state], device='cuda'
        )
    
    def get_seqlen_agnostic_capture_inputs(self, batch_size):
        # CUDA Graph capture 시 사용할 슬롯 인덱스 반환
        return self.conv_states[:, :batch_size], self.ssm_states[:, :batch_size]
```

### 3.3 Prefill 워크플로우

```
입력 토큰 시퀀스 (길이 L)
        ↓
[임베딩 레이어]
        ↓
[각 Mamba 레이어]
   ├─ Linear projection → x, z, B, C, dt (shape: [B, L, d_model])
   ├─ causal_conv1d_fwd(x)            ← 전체 시퀀스 depthwise conv
   ├─ selective_scan_fwd(...)          ← parallel scan으로 SSM 전파
   │       └─ 최종 hidden state → conv_states, ssm_states에 저장
   └─ gated output projection
        ↓
[LM Head] → logits
```

- 시퀀스 전체를 **parallel scan**으로 한 번에 처리 → 메모리 사용: O(L) (중간 출력용), 상태 저장: O(d_state) (고정)
- 청크 단위 prefill (chunked prefill)도 지원: 각 청크 끝의 SSM state를 다음 청크 시작점으로 전달

### 3.4 Decode 워크플로우

```
새 토큰 1개
        ↓
[임베딩]
        ↓
[각 Mamba 레이어]
   ├─ Linear projection → x, z, B, C, dt (shape: [B, 1, d_model])
   ├─ causal_conv1d_update(x, conv_state)   ← 슬라이딩 윈도우 업데이트
   ├─ selective_state_update(ssm_state, ...) ← O(d_state) recurrent step
   └─ output projection
        ↓
[LM Head] → next token logits
```

- **핵심**: SSM state 크기는 시퀀스 길이와 무관하게 고정 → decode step마다 KV 캐시 크기가 증가하는 Transformer와 근본적 차이
- Decode 연산량: `O(d_inner × d_state)` (상수), Transformer decode: `O(L × d_model)` (선형)

### 3.5 CUDA Graph 지원

vLLM은 decode 단계에서 CUDA Graph를 통해 커널 launch overhead를 제거한다. Mamba는 Transformer보다 CUDA Graph 적용이 단순한데, 이는 decode 입력 shape이 `[batch, 1, d_model]`로 고정되어 있기 때문이다. 다만 `conv_states`와 `ssm_states`가 in-place 업데이트되므로 Graph replay 시 정확한 슬롯 인덱싱이 보장되어야 한다.

---

## 4. SGLang의 Mamba 서빙

### 4.1 아키텍처 접근 방식

SGLang은 **RadixAttention**을 핵심 캐시 메커니즘으로 사용하지만, Mamba 모델에서는 별도의 `MambaCache` 추상화를 도입한다.

### 4.2 주요 컴포넌트

```
sglang/
├── srt/models/mamba.py             ← Mamba1 모델 정의
├── srt/models/mamba2.py            ← Mamba2 모델 정의
├── srt/models/jamba.py             ← Jamba hybrid
├── srt/managers/schedule_batch.py  ← Mamba state를 포함한 배치 스케줄링
└── srt/layers/mamba/              ← Mamba 레이어 구현
    ├── mamba_mixer.py
    └── triton_kernels/
```

### 4.3 Extend (Prefill) vs Decode 모드 분리

SGLang은 내부적으로 `extend` (첫 prefill + 이후 context extension)와 `decode` (autoregressive generation)를 명시적으로 분리하여 다른 커널 경로를 사용한다.

```python
# SGLang Mamba 레이어 (개념적 표현)
class MambaMixer:
    def forward(self, hidden_states, inference_params, is_decode):
        if is_decode:
            # 단일 토큰 recurrent step
            out = self._decode_step(hidden_states, inference_params)
        else:
            # 전체 시퀀스 parallel scan (extend/prefill)
            out = self._prefill_scan(hidden_states, inference_params)
        return out
```

### 4.4 RadixAttention과의 통합

SGLang의 RadixAttention은 prefix caching을 위해 Radix Tree 구조로 KV 캐시를 관리한다. Mamba 모델에서는 다음과 같은 차이가 있다:

- **Transformer**: 동일한 prefix를 공유하는 요청들이 KV 캐시 블록을 재사용
- **Mamba**: SSM state는 전체 이전 시퀀스의 압축 표현이므로, 동일한 prefix에 대한 SSM state도 이론적으로 재사용 가능하나 구현 복잡도가 높음. 현재는 per-request state 유지 방식이 주류.

### 4.5 Chunked Prefill 지원

SGLang은 메모리 효율을 위해 긴 입력을 청크로 나누어 prefill을 수행한다. Mamba에서 청크 경계에서의 SSM state 전달:

```
Chunk 0 [0:512]   → SSM state_0 계산 후 보관
Chunk 1 [512:1024] → state_0을 초기 상태로 SSM 이어서 계산
Chunk 2 [1024:L]  → state_1 → state_2 → 최종 state 캐시에 저장
```

---

## 5. Hybrid 모델 서빙

### 5.1 이중 캐시 구조

Hybrid 모델(예: Jamba)은 레이어별로 다른 캐시가 필요하다.

```
Layer 0  (Mamba)    → SSM state 슬롯
Layer 1  (Mamba)    → SSM state 슬롯
Layer 2  (Attention)→ PagedAttention KV 캐시 블록
Layer 3  (Mamba)    → SSM state 슬롯
...
Layer 7  (Attention)→ PagedAttention KV 캐시 블록
```

vLLM의 Jamba 구현에서는 `JambaCacheManager`가 두 종류의 캐시를 함께 관리한다.

```python
# Jamba 캐시 (개념적 표현)
class JambaCacheManager:
    def __init__(self):
        self.mamba_cache = MambaCacheManager(...)   # SSM states
        self.attn_cache = PagedKVCacheManager(...)  # KV blocks
    
    def allocate(self, seq):
        self.mamba_cache.allocate_slot(seq.id)
        self.attn_cache.allocate_blocks(seq)
```

### 5.2 Hybrid Prefill 워크플로우

```
입력 시퀀스
        ↓
Layer 0 (Mamba):
   selective_scan_fwd → SSM state 저장
        ↓
Layer 1 (Mamba):
   selective_scan_fwd → SSM state 저장
        ↓
Layer 2 (Attention, GQA):
   FlashAttention-2/3 → KV 캐시 블록에 Key, Value 저장
        ↓
...
        ↓
Layer N (Attention):
   FlashAttention → KV 저장
        ↓
[LM Head]
```

### 5.3 Hybrid Decode 워크플로우

```
새 토큰
        ↓
Layer 0 (Mamba):
   causal_conv1d_update + selective_state_update
        ↓
Layer 2 (Attention):
   PagedAttention (cached KV에 대해 attention 계산)
   → 새 KV를 블록에 append
        ↓
...
```

### 5.4 MoE와의 결합 (Jamba 케이스)

Jamba는 Attention 레이어에 MoE(Mixture of Experts)가 결합되어 있다. 이 경우:

- 각 Attention+MoE 레이어에서 Expert 라우팅 → 활성 Expert의 파라미터만 계산
- vLLM의 FusedMoE 커널 (`fused_topk_softmax`, `grouped_gemm`) 활용
- SSM state cache와 KV cache, Expert routing이 동시에 관리됨 → 스케줄러 복잡도 증가

---

## 6. Transformer와의 차이점

### 6.1 캐시 구조 비교

| 항목 | Transformer | Mamba | Hybrid |
|------|-------------|-------|--------|
| 캐시 종류 | KV Cache (Key, Value 텐서) | SSM State (conv_state + ssm_state) | 두 캐시 모두 |
| 캐시 크기 (시퀀스 길이 L) | O(L × num_layers × num_heads × head_dim) | O(num_layers × d_inner × (d_conv + d_state)) | 두 캐시의 합 |
| 시퀀스 의존성 | 길이에 비례하여 증가 | **고정 크기** (길이 무관) | Attention 레이어 수에 비례 |
| Prefix 재사용 | RadixAttention, vLLM prefix cache | 어렵 (상태가 전체 히스토리 압축) | Attention 부분만 prefix 재사용 |

### 6.2 연산 복잡도 비교

| 단계 | Transformer | Mamba |
|------|-------------|-------|
| Prefill | O(L² × d_model) — full attention | O(L × d_inner × d_state) — linear scan |
| Decode (1 step) | O(L × d_model) — attend to all past tokens | O(d_inner × d_state) — 상수 |
| 메모리 (decode) | KV 캐시 O(L) 증가 | SSM state O(1) 고정 |

### 6.3 Continuous Batching 동작 차이

**Transformer (PagedAttention)**:
- 시퀀스 preemption 시 KV 캐시 블록을 CPU로 swap하거나 재계산
- 블록 단위 할당으로 external fragmentation 최소화

**Mamba**:
- Preemption 시 `conv_state` + `ssm_state` 전체를 CPU로 offload (크기가 작아 swap 비용 낮음)
- 하지만 새 시퀀스가 추가될 때 배치 인덱싱 재정렬 필요
- vLLM은 `copy_inputs`/`copy_outputs` 단계를 통해 슬롯 재배치를 처리

```python
# vLLM Mamba 배치 재정렬 (개념적)
# 새 시퀀스 추가 또는 완료 시 슬롯 인덱스 재배치
def _prepare_mamba_cache(self, scheduled_seqs):
    for seq in scheduled_seqs:
        if seq.id not in self.slot_mapping:
            slot = self.free_slots.pop()
            self.slot_mapping[seq.id] = slot
```

### 6.4 FlashAttention vs Selective Scan 커널 비교

| 항목 | FlashAttention (Transformer) | Selective Scan (Mamba) |
|------|------------------------------|----------------------|
| 알고리즘 | Online softmax attention, tiling | Parallel prefix scan |
| IO 복잡도 | O(L²/B) — SRAM tiling으로 HBM 접근 최소화 | O(L) — 순차 의존성으로 tiling 제한 |
| 병렬화 | Head 차원, 시퀀스 블록 병렬 | 채널 차원 병렬 (시퀀스 방향은 sequential) |
| Decode 커널 | PagedAttention (비연속 KV 블록 접근) | selective_state_update (in-place state 업데이트) |
| Decode 메모리 접근 | O(L × num_heads × head_dim) | O(d_inner × d_state) |

### 6.5 Speculative Decoding 호환성

- **Transformer**: Draft 모델 + 검증 → 현재 vLLM, SGLang 모두 완전 지원
- **Mamba**: SSM state가 draft와 target 간에 동기화되어야 하는 추가 요구사항 → 현재 제한적 지원 (구현 복잡)

### 6.6 Scheduler 관점의 차이

```
[Transformer Scheduler]
- 각 시퀀스의 현재 길이에 따라 남은 KV 캐시 블록 추적
- 가용 블록 수가 스케줄링 병목

[Mamba Scheduler]
- 각 시퀀스의 슬롯 인덱스 추적
- max_num_seqs 이상의 동시 요청 불가 (슬롯 고정 풀)
- 슬롯 풀 크기 = 서버 시작 시 고정된 max_batch_size
- OOM 패턴이 다름: KV 메모리 부족이 아닌 슬롯 고갈로 throttle
```

---

## 7. 성능 특성 및 트레이드오프

### 7.1 처리량 (Throughput)

| 시나리오 | Transformer 우세 | Mamba 우세 |
|----------|-----------------|-----------|
| 짧은 시퀀스 (< 512) | Attention overhead 작음, parallel prefill | - |
| 긴 시퀀스 (> 4K) | - | Prefill O(L) vs O(L²), decode 상수 비용 |
| 대규모 배치 | KV 캐시 병목 발생 | SSM state 크기 고정으로 배치 확장 용이 |
| Long-context serving | KV 캐시 메모리 폭증 | 메모리 효율적 |

### 7.2 지연 시간 (Latency)

- **TTFT (Time To First Token, Prefill latency)**:
  - 짧은 시퀀스: Transformer ≈ Mamba (FlashAttention 최적화로 gap 작음)
  - 긴 시퀀스: Mamba 유리 (O(L) vs O(L²))

- **TPOT (Time Per Output Token, Decode latency)**:
  - Mamba가 일반적으로 유리: `selective_state_update`는 매우 경량
  - 단, small batch에서 Attention의 KV 재사용 이점이 사라지면 격차 커짐

### 7.3 메모리 효율

```
예: 7B 모델, 32 레이어, 시퀀스 길이 32K, 배치 크기 32, bf16

[Transformer KV Cache]
32 layers × 2(KV) × 32 heads × 128 head_dim × 32K × 32 batch × 2 bytes
≈ 32 × 2 × 32 × 128 × 32768 × 32 × 2 ≈ ~500 GB  ← 불가능

[Mamba SSM State]  
32 layers × batch 32 × (d_inner 4096 × d_conv 4 + d_inner 4096 × d_state 16) × 2 bytes
≈ 32 × 32 × (16384 + 65536) × 2 ≈ ~160 MB  ← 매우 작음
```

### 7.4 현재 구현 한계

1. **Prefix Caching 미지원 (순수 Mamba)**: SSM state가 전체 시퀀스 히스토리의 손실 압축이라 prefix 재사용 불가
2. **Speculative Decoding 제한**: draft model과 target model 간 SSM state 동기화 비용
3. **Tensor Parallelism**: Attention의 head-parallel 분산과 달리, Mamba의 채널 분산은 all-reduce 패턴이 달라 구현 복잡
4. **Variable-length 배치 효율**: Transformer의 FlashAttention varlen은 잘 최적화되어 있으나, Mamba의 varlen selective scan은 padding 또는 별도 처리 필요

---

## 8. 요약

```
┌─────────────────────────────────────────────────────────────────────┐
│                    핵심 차이점 요약                                   │
├────────────────┬─────────────────────┬──────────────────────────────┤
│ 항목           │ Transformer          │ Mamba                        │
├────────────────┼─────────────────────┼──────────────────────────────┤
│ 캐시 구조      │ KV Cache (O(L) 성장)│ SSM State (O(1) 고정)        │
│ Prefill 커널   │ FlashAttention       │ selective_scan_fwd /         │
│                │                     │ mamba_chunk_scan_combined    │
│ Decode 커널    │ PagedAttention       │ selective_state_update +     │
│                │                     │ causal_conv1d_update          │
│ Decode 복잡도  │ O(L) per token      │ O(1) per token               │
│ 메모리 확장성  │ 시퀀스 길이에 비례  │ 배치 크기에만 비례           │
│ Prefix 재사용  │ 지원 (RadixAttn)    │ 미지원 (손실 압축 특성)      │
│ Scheduler 병목 │ KV 블록 고갈        │ 슬롯 풀 고갈                 │
└────────────────┴─────────────────────┴──────────────────────────────┘
```

**Hybrid 모델**은 두 캐시를 동시에 관리하는 이중 구조를 채택함으로써, Attention 레이어의 문맥 포착 능력과 Mamba 레이어의 메모리 효율성을 결합한다. 서빙 엔진(vLLM, SGLang) 관점에서는 스케줄러가 두 캐시의 가용량을 모두 추적해야 하며, preemption 시 두 종류의 상태를 함께 저장·복원해야 하는 복잡도가 추가된다.

Mamba 서빙의 핵심 가치는 **긴 시퀀스에서의 메모리 효율과 decode 지연 시간 단축**이며, 이는 long-context 추론, 스트리밍 생성, 대규모 배치 serving 시나리오에서 Transformer 대비 실질적인 이점을 제공한다.

---

*참고 코드베이스: vLLM v0.6+, SGLang v0.3+, mamba-ssm v2.x, causal-conv1d v1.4+*
