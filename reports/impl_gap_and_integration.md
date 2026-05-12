# prefill-layer-alloc 구현 공백 및 Serving Engine 통합 요구사항 보고서

> 작성일: 2026-05-12

---

## 목차

1. [전제: 두 시스템의 역할 차이](#1-전제)
2. [Serving Engine에 있지만 이 프로젝트에 없는 것들](#2-구현-공백)
3. [Serving Engine과의 결합을 위해 필요한 것들](#3-통합-요구사항)
4. [통합 시 핵심 제약과 설계 결정](#4-핵심-제약)
5. [우선순위 로드맵](#5-로드맵)

---

## 1. 전제

`prefill-layer-alloc`은 서빙 시스템이 아니라 **SM 재할당 전략의 타당성을 검증하는 연구 프레임워크**다. 따라서 아래 공백들은 "버그"가 아니라 설계상 의도적으로 제외된 것들이다. 단, 이 기법을 실제 서빙 엔진에 이식하려면 반드시 채워야 하는 항목들이다.

---

## 2. 구현 공백

### 2.1 모델 실행 레이어

#### 2.1.1 완전한 Forward Pass 없음

`LayerRunner`는 개별 레이어를 **격리 실행**한다. 실제 모델 forward는 다음 요소들이 빠져 있다.

| 없는 요소 | 관련 코드 | 영향 |
|----------|----------|------|
| 레이어 간 residual connection | `layer_runner.py` — 각 레이어 독립 호출 | latency는 단일 레이어 기준; 실제 end-to-end TTFT와 다름 |
| LayerNorm / RMSNorm 포함 여부 불확실 | `get_ssm_layer()` — HF 모델 내부 서브모듈 추출 | 레이어 경계 정의가 HF 모델 구현에 의존 |
| 임베딩 레이어 | 없음 | 입력은 항상 `torch.randn(..., hidden_size)` 난수 |
| LM Head | 없음 | logit 생성 없음 |
| Softmax / Sampling | 없음 | 다음 토큰 결정 불가 |

현재 TTFT 계산 (`run_concurrent_eval.py:202`):
```python
ttft_ms = step_start_ms - current_prefill.start_time_ms + tpot_ms
```
이는 **wall-clock 기반 근사값**이다. 실제 TTFT는 `n_layers` 개의 prefill 레이어 CUDA 실행 시간 합산이어야 하지만, 각 레이어가 순차 실행되며 CUDA Event로 측정되지 않아 오차가 있다.

#### 2.1.2 레이어 간 SSM State 미전달

Mamba 모델에서 레이어 N의 출력 hidden state는 레이어 N+1의 입력이 되고, SSM state(`conv_state`, `ssm_state`)는 decode 시 다음 step으로 전달된다. 현재 구현:

```python
# run_concurrent_eval.py — _run_prefill_layer
def _run_prefill_layer(runner, model_name, batch_size, seq_len, total_sm, layer_type="ssm"):
    runner.run_ssm_layer(model_name=model_name, batch_size=batch_size,
                         seq_len=seq_len, sm_count=total_sm,
                         n_warmup=0, n_measure=1, skip_sm_control=True)
    # ← 반환값 무시, 다음 레이어에 전달 없음
```

각 레이어는 매번 `_ssm_cache`에 저장된 **동일한 난수 입력**으로 실행된다. 따라서:
- 정확성(correctness) 검증 불가
- 레이어 간 데이터 의존성이 제거되어 latency는 독립 측정

vLLM의 실제 동작:
```python
# vLLM MambaModel.forward (개념적)
hidden_states = self.embeddings(input_ids)
for i, layer in enumerate(self.layers):
    hidden_states = layer(hidden_states,
                          ssm_state=mamba_cache.conv_states[i],  # ← 이전 step 상태
                          kv_cache=kv_cache[i])                   # ← Attn layer용
```

#### 2.1.3 FallbackSSMKernel의 물리적 한계

`FallbackSSMKernel._pytorch_fallback()`은 Triton SSD와 다른 병목을 가진다.

```python
# zamba2.py:285 — PyTorch fallback의 recurrence loop
for ci in range(n_chunks):
    xc = x_chunks[:, ci]
    out = h.unsqueeze(1) * A + xc   # GEMM 없음, elementwise만
    h = out[:, -1]
    outs.append(out)
```

- Triton SSD: compute-bound, SM 포화점 ~70%+
- PyTorch fallback: GEMM-bound (`in_proj`, `out_proj`), SM 포화점 ~30–40%

`force_pytorch_scan=True`로 측정한 SM scaling 데이터를 Decision Matrix에 입력하면 SSM에 과소 SM 할당 → 실제 Triton 커널 실행 시 TPOT SLO 위반. 이 문제는 `reports/ssm_sm_partitioning_analysis.md`에 상세히 기록되어 있으며, 현재는 Wave Model 합성으로 회피 중이다.

---

### 2.2 메모리 및 캐시 관리

#### 2.2.1 KV Cache 없음

vLLM의 PagedAttention과 SGLang의 RadixAttention은 다음을 제공한다.

| 기능 | vLLM | SGLang | prefill-layer-alloc |
|------|------|--------|---------------------|
| KV 블록 할당/해제 | `BlockManager` | `RadixTree` | **없음** |
| 비연속 물리 메모리 접근 | `paged_attn_kernel` | 동일 | **없음** |
| Prefix 공유 | `prefix_cache` | `RadixAttention` | **없음** |
| 선점 시 KV offload | `CpuOffloader` | 없음 | **없음** |

Attention 벤치마크(`run_attn_layer`)에서는 `context_len`만큼의 KV 텐서를 난수로 할당하고, 실제 이전 토큰의 Key/Value가 아님:

```python
# layer_runner.py:409
ctx_k = torch.randn(batch_size, context_len, n_kv_heads, head_dim, ...)
ctx_v = torch.randn(batch_size, context_len, n_kv_heads, head_dim, ...)
```

이 방식은 latency 측정에는 충분하지만, 실제 서빙에서 발생하는 KV 메모리 압박(OOM, 스와핑)을 재현하지 못한다.

#### 2.2.2 SSM State Cache 없음

vLLM `MambaCacheManager`가 관리하는 per-sequence SSM state가 없다.

```python
# vLLM (개념적)
conv_states: Tensor  # shape [n_layers, max_seqs, d_inner, d_conv]
ssm_states:  Tensor  # shape [n_layers, max_seqs, n_heads, head_dim, d_state]
```

현재 `_run_decode_step`은 `seq_len=1`로 SSM 레이어를 실행하지만, 이전 decode step의 conv_state / ssm_state를 이어받지 않으므로 **매 step이 cold start**다. 따라서 측정되는 decode latency는 state 로드 비용을 포함하지 않는다.

#### 2.2.3 GPU 메모리 예산 관리 없음

vLLM은 서버 시작 시 전체 GPU 메모리의 `gpu_memory_utilization`(기본 90%) 내에서 KV 캐시 블록 수를 결정한다. prefill-layer-alloc에는 이 예산 계산이 없으며, OOM은 `torch.cuda.empty_cache()`로만 처리된다 (`run_ssm_prefill_sweep.py:157`).

---

### 2.3 스케줄러

#### 2.3.1 단순화된 요청 주입 모델

```python
# run_concurrent_eval.py:52,164
PREFILL_INTERVAL_STEPS = 4   # 고정 간격으로 요청 주입
N_PREFILL_REQUESTS = 50      # 총 50개 요청 후 종료

if decode_step % PREFILL_INTERVAL_STEPS == 0:
    prefill_queue.append(PrefillState(...))
```

실제 서빙 엔진의 스케줄러가 처리하는 항목들:

| 기능 | vLLM Scheduler | 이 프로젝트 |
|------|---------------|------------|
| 요청 도착 모델 | Poisson process (온라인), 파일 (오프라인) | 고정 간격 |
| 우선순위 | FCFS, priority queue | 단순 deque |
| 선점 (preemption) | KV 블록 부족 시 실행 중 요청 강제 중단 | 없음 |
| 재스케줄링 | swap-in/out + re-queue | 없음 |
| 배치 크기 동적 조정 | 매 iteration마다 가용 메모리 기반 결정 | 고정 `decode_batch_size` |
| 최대 토큰 수 제한 | `max_num_batched_tokens` | 없음 |

#### 2.3.2 Continuous Batching 없음

vLLM의 continuous batching은 decode 중인 요청과 새 prefill 요청을 **같은 forward pass**에 넣고, attention mask로 구분하여 처리한다. prefill-layer-alloc은 decode와 prefill을 **시간적으로 분리**하여 번갈아 실행한다.

```
[vLLM Continuous Batching]
Forward N: [decode req A (token 47), decode req B (token 12), prefill req C (len 512)]
            → 하나의 forward pass, 단일 배치 텐서

[prefill-layer-alloc]
Step N:    decode_step (seq_len=1) → prefill_layer_k (seq_len=prefill_len)
            → 두 개의 독립 LayerRunner 호출
```

---

### 2.4 운영 기능

#### 2.4.1 Tensor Parallelism 없음

vLLM/SGLang은 `tensor_parallel_size > 1`일 때 모델 가중치를 여러 GPU에 분산하고 `all-reduce`로 동기화한다. prefill-layer-alloc은 단일 GPU만 지원한다. Green Contexts는 단일 GPU 내 SM 분할이므로, 멀티-GPU 환경에서는 GPU 간 통신 overhead와 SM 분할 overhead 간의 상호작용을 별도로 측정해야 한다.

#### 2.4.2 CUDA Graph 없음

vLLM decode 단계는 CUDA Graph로 커널 launch overhead를 제거한다.

```python
# vLLM MambaCUDAGraphRunner (개념적)
with torch.cuda.graph(self.graph):
    for layer in model.layers:
        hidden = layer(hidden, ssm_states=..., kv_cache=...)
# replay:
self.graph.replay()
```

CUDA Graph는 실행 시점의 stream, shape, 메모리 주소가 고정된다. Green Context stream을 레이어 경계마다 바꾸는 Policy C는 **CUDA Graph와 호환되지 않는다** — Graph replay는 캡처 시점의 stream 포인터를 재사용하기 때문이다. prefill-layer-alloc은 CUDA Graph를 사용하지 않으므로 이 충돌이 발생하지 않는다.

#### 2.4.3 Chunked Prefill 없음

vLLM은 긴 prompt를 `chunk_size` 토큰 단위로 분할하여 여러 iteration에 걸쳐 prefill한다. 이는 다음 두 가지를 달성한다:
- 단일 배치의 메모리 사용량 상한 제한
- prefill 중에도 decode 요청이 GPU 시간을 배분받을 수 있음 (starvation 방지)

prefill-layer-alloc의 레이어-단위 인터리빙은 conceptually chunked prefill과 유사하지만, 청크 경계에서 SSM state를 실제로 전달하지 않는다.

#### 2.4.4 양자화(Quantization) 없음

vLLM은 AWQ, GPTQ, FP8, INT8 등 다양한 양자화를 지원한다. 양자화된 GEMM 커널은 SM 포화 특성이 달라지므로, 양자화 환경에서의 SM scaling curve는 이 프로젝트의 측정값과 다를 수 있다.

---

### 2.5 측정 정확도의 한계

#### 2.5.1 동시 실행이 아닌 인터리빙

`run_concurrent_eval.py` 주석에 명시:

```python
# This is NOT true kernel-level parallelism but layer-level interleaving
# on a single GPU, matching the execution model of MuxWise/BulletServe.
```

decode_step과 prefill_layer는 같은 CUDA stream에서 순차 실행되지 않고, 각자의 Green Context stream에서 실행되지만 CPU 측에서 동기화하므로 **실질적으로 sequential**이다. 진정한 동시 실행(두 커널이 SM을 물리적으로 동시 점유)은 별도의 multi-stream 동기화가 필요하며, SSM 레이어에서는 불가능하다.

#### 2.5.2 decode_sm_sensitivity 추정 불확실성

`compute_decision_matrix.py`의 `potential_gain` 계산:

```python
potential_gain = free_sm_fraction × ssm_layer_fraction × decode_sm_sensitivity
```

`decode_sm_sensitivity` (기본값 `0.5`)는 "SM 10% 추가당 TPOT 개선율"인데, 실측값이 아닌 상수다. 이 값은 `mixer-alloc` 선행 프로젝트에서 얻어야 하며, 현재 hardcoding되어 있다.

---

## 3. 통합 요구사항

서빙 엔진(vLLM 또는 SGLang)에 이 프로젝트의 SM 재할당 정책을 이식하기 위해 필요한 구현 항목들이다.

### 3.1 SMController를 서빙 엔진에 초기화

Green Contexts는 CUDA primary context 초기화 **이후**에 생성해야 한다.

**vLLM 통합 위치**: `vllm/worker/worker.py`의 `Worker.init_device()` 내부.

```python
# vllm/worker/worker.py (수정 후 개념적)
class Worker:
    def init_device(self):
        torch.cuda.set_device(self.device)
        torch.cuda.init()   # ← primary context 초기화
        
        # 추가: SMController 초기화
        if self.model_config.is_hybrid_ssm_model():
            from src.smctrl import SMController
            self.smctrl = SMController(
                device_id=self.device.index,
                total_sm_count=self.device_config.sm_count,
                preset_sm_counts=self.sm_preset_counts,  # hardware.yaml에서 로드
            )
            assert self.smctrl.verify_sm_control(), \
                "Green Contexts SM control verification failed"
        else:
            self.smctrl = None
```

**초기화 순서 제약**: `cuGreenCtxCreate`는 primary context 활성화 전 호출 시 `CUDA_ERROR_INVALID_CONTEXT`를 반환한다. `torch.cuda.init()` → `SMController()` 순서가 보장되어야 한다.

**MIG 모드 체크**: A100에서 MIG가 활성화된 경우 `CUDA_ERROR_NOT_SUPPORTED` 반환. 서버 시작 시 MIG 비활성화 여부를 `nvidia-smi -q | grep "MIG Mode"` 또는 `pynvml.nvmlDeviceGetMigMode()`로 확인해야 한다.

---

### 3.2 ModelRunner에 레이어 경계 훅 삽입

현재 vLLM의 forward pass는 레이어 루프가 `model.forward()` 내부에 캡슐화되어 있다.

```python
# vllm/model_executor/models/zamba.py (현재)
def forward(self, input_ids, positions, kv_caches, attn_metadata, ...):
    hidden = self.embed_tokens(input_ids)
    for i, layer in enumerate(self.layers):
        hidden = layer(hidden, kv_caches[i], attn_metadata, ...)
    return self.norm(hidden)
```

SM 재할당을 삽입하려면 레이어 루프를 **훅 가능한 구조**로 분리해야 한다.

```python
# 수정 후 (개념적)
def forward(self, input_ids, positions, kv_caches, attn_metadata,
            smctrl=None, sm_policy=None, is_prefill=False):
    hidden = self.embed_tokens(input_ids)
    for i, layer in enumerate(self.layers):
        layer_type = self._get_layer_type(i)   # "ssm" | "attn" | "mlp"
        
        if smctrl is not None and sm_policy is not None and is_prefill:
            sm_policy.on_prefill_layer_start(i, layer_type)
            
        with torch.cuda.stream(smctrl.get_stream() if smctrl else torch.cuda.current_stream()):
            hidden = layer(hidden, kv_caches[i], attn_metadata, ...)
            
        if smctrl is not None and sm_policy is not None and is_prefill:
            sm_policy.on_prefill_layer_end(i, layer_type)
    
    return self.norm(hidden)
```

**필요한 헬퍼**: `_get_layer_type(i)` — 현재 `prefill-layer-alloc`의 `MODEL_LAYER_TYPES` 람다와 동일한 역할. vLLM에서는 모델 config(`hybrid_layer_ids`, `ssm_layer_fraction` 등)에서 동적으로 생성해야 한다.

---

### 3.3 SSM 레이어에서 SM 분할 금지 처리

SSM 레이어 (`mamba_chunk_scan_combined`) 실행 시 Green Context로 SM을 제한하면 cooperative barrier deadlock이 발생한다 (`reports/ssm_sm_partitioning_analysis.md` 참조). 통합 코드는 이를 반드시 처리해야 한다.

```python
# sm_policy.on_prefill_layer_start() 내부에 추가할 로직
def on_prefill_layer_start(self, layer_idx: int, layer_type: str) -> None:
    if layer_type == "ssm":
        # SSM 레이어: SM 분할 불가 → 전체 SM 사용
        # policy는 prefill에 70% 주고 싶지만, Triton SSD cooperative barrier
        # 제약으로 인해 Green Context 사용 불가 → 기본 stream으로 fallback
        self.smctrl.reset()            # 전체 SM
        self._current_stream = torch.cuda.current_stream()
    elif layer_type in ("attn", "mlp"):
        # Attention/MLP: SM 분할 가능 → Green Context stream 사용
        ratio = self.get_prefill_ratio(layer_type)
        self.smctrl.set_sm_ratio(ratio)
        self._current_stream = self.smctrl.get_stream()
```

이 로직이 없으면 Zamba2의 68개 SSM 레이어에서 CUDA context 오염이 발생한다.

---

### 3.4 Decode Stream 분리

prefill이 Green Context stream에서 실행되는 동안 decode가 기본 stream에서 실행된다면, 두 스트림 간에 명시적 동기화가 없으면 데이터 hazard가 발생한다.

```
[현재 prefill-layer-alloc 동작]
CPU: decode_step() → prefill_layer() → decode_step() → ...
     각 호출은 동기적으로 완료 후 다음 실행 (torch.cuda.Event 측정)
     → 실질적으로 sequential, 동기화 문제 없음

[실제 서빙 엔진 통합 시 요구사항]
decode stream (Green Context A: 60% SM) ──→ 연속 실행
prefill stream (Green Context B: 40% SM) ─→ 레이어 경계마다 CPU trigger

필요한 동기화:
- prefill stream이 layer k 완료 → decode stream에 signal (cudaStreamWaitEvent)
- decode stream이 step t 완료 → prefill stream에서 layer k+1 시작 가능
```

두 stream이 진정으로 동시에 실행되려면 `cudaEvent`를 통한 producer-consumer 동기화가 필요하다. 현재 프로젝트에는 이 메커니즘이 없다.

---

### 3.5 CUDA Graph와의 호환성 처리

vLLM decode step은 CUDA Graph로 capture되어 있다. Policy C (layer-wise stream 전환)는 Graph와 호환되지 않는다.

**선택지**:

| 옵션 | 방법 | 비용 |
|------|------|------|
| A: decode에 CUDA Graph 비활성화 | `enforce_eager=True` (vLLM 옵션 존재) | decode latency 증가 (~10–20%) |
| B: SM 전환을 Graph 외부에서 처리 | decode Graph replay 전/후에만 stream 전환 | Policy A/B만 가능 (레이어 경계 불가) |
| C: Graph를 SM 비율별로 별도 capture | 각 SM preset마다 별도 Graph | 초기화 시간/메모리 증가 (preset 수 × model 크기) |
| D: Graph 안에 stream 전환 포함 | CUDA Graph는 ExternalStream launch를 캡처 가능 | 검증 필요; 복잡도 높음 |

옵션 B가 현실적 타협점이다: decode는 CUDA Graph (전체 SM), prefill은 Green Context stream (레이어별 SM 분할). prefill-decode 간 진정한 동시 실행 없이 decode step 완료 후 prefill layer 실행.

---

### 3.6 레이어 타입 메타데이터 테이블 구축

현재 프로젝트:
```python
# run_concurrent_eval.py:63
_ZAMBA2_HYBRID_IDS = frozenset([6, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77])
MODEL_LAYER_TYPES = {
    "zamba2": lambda i: "attn" if i in _ZAMBA2_HYBRID_IDS else "ssm",
    "falcon_h1": lambda i: "ssm",
}
```

서빙 엔진 통합 시 이 정보는 모델 config (`configs/models.yaml`)에서 동적으로 생성해야 한다.

```python
# 서빙 엔진 통합용 헬퍼 (신규 구현 필요)
def build_layer_type_sequence(model_config) -> list[str]:
    """모델 config를 읽어 레이어별 타입 시퀀스를 반환."""
    n_layers = model_config.num_hidden_layers
    hybrid_ids = set(getattr(model_config, "hybrid_layer_ids", []))
    
    types = []
    for i in range(n_layers):
        if i in hybrid_ids:
            types.append("attn")   # SSM+Attn hybrid → Attn이 병목
        else:
            # pure SSM 또는 모든 레이어 SSM+Attn 병렬 (Falcon-H1)
            types.append("ssm")
    return types
```

지원해야 하는 모델별 특수 케이스:
- **Zamba2**: `hybrid_layer_ids`에 있는 레이어만 Attn → SSM/Attn 번갈아
- **Falcon-H1**: 모든 레이어 SSM+Attn 병렬 → SSM 병목 (Attention도 실행되지만 SSM이 compute 지배)
- **Jamba**: SSM + MoE Attention 혼합 → MoE 라우팅 overhead도 고려 필요

---

### 3.7 Decision Matrix를 서빙 엔진 시작 시 로드

현재 `compute_decision_matrix.py`는 오프라인 도구다. 서빙 엔진에서는 시작 시 로드해야 한다.

```python
# 서빙 엔진 Worker 초기화 (개념적)
class Worker:
    def _load_sm_policy(self, model_name: str, device_tag: str) -> Policy:
        dm_path = Path(f"results/stage2/decision_matrix_{device_tag}.json")
        if dm_path.exists():
            with open(dm_path) as f:
                dm = json.load(f)
            dominant = dm["dominant_strategy"]
        else:
            dominant = "fixed"   # 데이터 없으면 안전한 기본값
        
        if dominant == "layer_wise" and self.smctrl.is_available():
            return PolicyLayerWise(self.smctrl)
        elif dominant == "step_adaptive" and self.smctrl.is_available():
            return PolicyStepAdaptive(self.smctrl, model_name=model_name)
        else:
            return PolicyBaseline(self.smctrl)
```

Decision Matrix는 **GPU별, 모델별, seq_len 범위별**로 다르므로, 서빙 엔진이 여러 모델이나 GPU를 지원한다면 매트릭스도 조합별로 필요하다.

---

### 3.8 Variable-Length 배치 처리

현재 `LayerRunner`는 항상 고정 `(batch_size, seq_len)`으로 실행한다.

```python
hidden_states = torch.randn(batch_size, seq_len, hidden_size, ...)
```

실제 서빙에서는 배치 내 각 요청의 seq_len이 다르다. vLLM은 padding 없이 FlashAttention varlen API를 사용한다:

```python
# vLLM FlashAttention varlen (개념적)
flashattn_varlen_func(
    q=q_flat,           # [total_tokens, n_heads, head_dim]
    k=k_flat,
    v=v_flat,
    cu_seqlens_q=...,   # 요청별 누적 token 수
    cu_seqlens_k=...,
    max_seqlen_q=...,
    max_seqlen_k=...,
)
```

SM 재할당 정책이 variable-length 배치에서 동작하려면:
- `on_prefill_layer_start()`에 전달되는 `layer_type`이 현재 배치의 dominant layer type을 대표해야 함
- 배치 내 요청별 seq_len에 따라 SM saturation point가 달라지므로 정책 파라미터도 동적으로 조정 필요

---

### 3.9 SSM State 전달과 SM 전환의 연동

decode 단계에서 `selective_state_update`는 이전 step의 `conv_state`와 `ssm_state`를 읽고 in-place로 업데이트한다.

```python
# vLLM decode 단계 (개념적)
for seq_id, slot in slot_mapping.items():
    conv_state_slice = conv_states[layer_idx, slot]    # shape: [d_inner, d_conv]
    ssm_state_slice  = ssm_states[layer_idx, slot]     # shape: [n_heads, head_dim, d_state]
    
    # SM 전환 후 이 커널 실행
    selective_state_update(ssm_state_slice, x, dt, A, B, C, ...)
```

SM 전환과 state 업데이트 사이에 동기화가 없으면 race condition 가능. Green Context stream 전환 후 state 텐서 접근 전에 stream 동기화(`cudaStreamSynchronize` 또는 `cudaEventWait`)가 필요하다.

---

### 3.10 Preemption 시 Green Context 상태 처리

vLLM은 KV 메모리 부족 시 실행 중인 요청을 선점한다. prefill 중 선점이 발생하면:

1. 현재 prefill 중단
2. KV 블록 → CPU offload 또는 재계산 예약
3. `PrefillState.layer_idx` 초기화 필요

현재 `run_concurrent_eval.py`의 `prefill_queue`에는 선점 로직이 없다. 서빙 엔진 통합 시 `PrefillState`를 확장하여 선점된 요청의 진행 상태를 보존하고, 재스케줄링 시 마지막 완료 레이어부터 재개해야 한다.

단, SSM 레이어는 state가 누적되므로 임의 레이어부터 재개가 불가능하다 — 레이어 0부터 재계산하거나, 중간 레이어의 SSM state를 모두 저장해야 한다.

---

## 4. 핵심 제약과 설계 결정

### 4.1 SSM 레이어 SM 분할 불가 — 피할 수 없는 제약

이 프로젝트의 가장 중요한 발견이자 통합 시 가장 큰 설계 제약이다.

```
Triton SSD (mamba_chunk_scan_combined)
  └── cooperative grid.sync() barrier
        └── SM 분할 시 → deadlock → CUDA context corruption
              └── 해결책: 전체 SM 독점 실행 (Green Context 미사용)
```

결과적으로 SM 재할당의 실질적 이득은:
- **Zamba2 (81 layers)**: 13개 Attention 레이어에서만 → 전체 forward pass의 ~16%
- **Falcon-H1 (44 layers)**: 모든 레이어 SSM+Attn 병렬 → SSM branch가 compute 지배 → Attention SM 분할 이득 불확실

이 제약을 우회하려면 Triton SSD 커널 자체를 cooperative-free 방식으로 재작성해야 한다 (예: warp-level scan without `grid.sync()`). 이는 `mamba-ssm` 패키지 수준의 변경이다.

### 4.2 Green Contexts는 Data Center GPU 전용

`cuGreenCtxCreate`는 A100, H100, H200 등 데이터센터 GPU에서만 동작한다.

```
A100/H100: Green Contexts 지원 ✓
RTX 4090/5060Ti: 지원 안 함 ✗ (SMController.is_available() = False)
```

소비자 GPU에서는 `SMController.get_backend_name()` = "none"이 반환되며, 모든 policy가 자동으로 Policy A (고정 분할, 실질적으로 SM 분할 없음)로 fallback된다. 서빙 엔진 통합 시 GPU 타입에 따른 조건부 활성화가 필요하다.

### 4.3 Preset SM 수 snapping

`set_sm_count(n)`은 `__init__` 시 생성된 preset 중 가장 가까운 값으로 snap한다.

```python
# green_ctx_controller.py:344
idx = bisect.bisect_left(self._sorted_presets, n_sm)
snapped = lo if (n_sm - lo) <= (hi - n_sm) else hi
```

A100 (108 SM), preset `[14, 27, 40, 54, 68, 81, 94, 108]` 기준:

| 요청 SM | snap된 SM | 오차 |
|--------|---------|------|
| 43 SM (40%) | 40 SM | -3 SM (-2.8%) |
| 76 SM (70%) | 81 SM | +5 SM (+4.6%) |

서빙 엔진에서 정밀한 SM 제어가 필요하다면, sweep에서 사용하는 모든 SM 비율에 대응하는 Green Context를 `preset_sm_counts` 인자로 사전 생성해야 한다.

---

## 5. 로드맵

구현 우선순위를 기준으로 정리한다. "필수"는 정확성에 영향, "권장"은 실용성에 영향, "선택"은 고급 기능이다.

### Phase 1: 연구 프레임워크 완성 (현재 프로젝트 내)

| 항목 | 현재 상태 | 필요 작업 | 우선순위 |
|------|----------|----------|---------|
| SSM SM scaling (Wave Model) | 완료 | A100 재측정 확인 | — |
| Attention SM scaling | 완료 | — | — |
| Green Contexts 초기화 | 완료 | — | — |
| decode_sm_sensitivity 실측 | **미완료** | Stage 1에서 decode latency vs SM count 측정 | 필수 |
| Stage 2 A100 재실행 | **미완료** (`a100_migration_report.md` 참조) | `measure_ctx_switch_latency.py` 실행 | 필수 |
| Stage 3 A100 Policy A/B/C 비교 | **미완료** | `run_concurrent_eval.py --device a100_80gb` | 필수 |
| 실제 TTFT 측정 (CUDA Event) | **미완료** | prefill layer 완료 시점을 CUDA Event로 기록 | 권장 |

### Phase 2: vLLM 통합 PoC

| 항목 | 구현 난이도 | 비고 |
|------|-----------|------|
| SMController → `Worker.init_device()` 주입 | 낮음 | import 추가 + 조건부 초기화 |
| 레이어 경계 훅 (`on_prefill_layer_start`) | 중간 | `model.forward()` 분해 필요 |
| SSM 레이어 Green Context 우회 | 낮음 | `layer_type == "ssm"` 시 `smctrl.reset()` |
| Policy 로더 (decision_matrix.json) | 낮음 | 파일 읽기 + Policy 객체 생성 |
| `_get_layer_type()` 헬퍼 | 낮음 | `configs/models.yaml` 기반 생성 |
| CUDA Graph 비활성화 (decode) | 낮음 | `enforce_eager=True` 옵션 사용 |
| decode SSM state 전달 | **높음** | `MambaCacheManager` 연동 |
| Variable-length 배치 지원 | **높음** | `cu_seqlens` 기반 policy 조정 |
| 선점(preemption) 처리 | **높음** | `PrefillState` 상태 저장/복원 |

### Phase 3: SGLang 통합 (vLLM PoC 완료 후)

SGLang은 `extend` / `decode` 모드 분리가 이미 명확하고, Triton 커널을 사용하는 부분이 vLLM보다 접근하기 쉽다. SMController 통합 위치는 `srt/model_executor/model_runner.py`의 `forward()` 내부다.

---

*참고 파일: `src/smctrl/green_ctx_controller.py`, `src/models/layer_runner.py`, `stage3_hm_eval/run_concurrent_eval.py`, `stage3_hm_eval/policy_*.py`, `stage2_overhead/compute_decision_matrix.py`, `reports/ssm_sm_partitioning_analysis.md`, `a100_migration_report.md`*
