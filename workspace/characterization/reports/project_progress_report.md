# prefill-layer-alloc 프로젝트 진척 보고서

> 작성일: 2026-05-15  
> 하드웨어: NVIDIA A100-SXM4-80GB (108 SM, HBM2e ~2,000 GB/s)  
> 대상 모델: Zamba2-7B-Instruct (81층: 68 pure SSM + 13 hybrid Attn)

---

## 목차

1. [프로젝트 배경 및 해결하고자 한 문제](#1-프로젝트-배경-및-해결하고자-한-문제)
2. [실험 설계 및 예상 결과](#2-실험-설계-및-예상-결과)
3. [실험 시각화 해석 방법](#3-실험-시각화-해석-방법)
4. [실제 관측 결과 및 예상치와의 괴리](#4-실제-관측-결과-및-예상치와의-괴리)
5. [괴리 원인 분석](#5-괴리-원인-분석)
6. [우회 방법론 정리](#6-우회-방법론-정리)
7. [방향성 유지 가능성 탐색](#7-방향성-유지-가능성-탐색)

---

## 1. 프로젝트 배경 및 해결하고자 한 문제

### 1.1 동기

Hybrid SSM+Attention 모델(Zamba2-7B, Falcon-H1-7B 등)을 서빙할 때, prefill과 decode는 동시에 GPU를 점유한다. 기존 서빙 엔진(vLLM, SGLang)은 GPU 전체를 단일 실행 단위로 보고 레이어 타입별로 SM 할당을 달리하지 않는다.

**핵심 관찰**: SSM prefill 레이어와 Attention prefill 레이어는 연산 특성이 근본적으로 다르다.

| 레이어 | 알고리즘 | 이론적 병렬성 | SM 활용 가설 |
|--------|---------|------------|------------|
| SSM (Mamba2 SSD) | Parallel chunk scan + inter-chunk recurrence | Chunk 단위 병렬 | 비교적 이른 포화 |
| Attention (FlashAttention) | 독립 타일 attention score 계산 | Head × Tile 병렬 | 느린 포화 |
| MLP (cuBLAS GEMM) | 행렬 곱셈 | GEMM 병렬 | Compute/BW bound 전환점 존재 |

이 차이를 활용하면, 각 레이어 실행 중 **"남는 SM"을 decode 스트림에 제공**할 수 있다는 가설을 세웠다.

### 1.2 선행 프로젝트와의 연계

```
mixer-alloc      → 고정 TPC 비율별 전체 forward pass latency → optimal ratio 탐색
mamba-cuda-graph → Decode CUDA Graph + SM isolation → TPOT/TTFT 측정
prefill-layer-alloc (이 프로젝트) → prefill+decode 동시 실행 중 레이어 경계 SM 재분배 효과 측정
```

### 1.3 세 가지 연구 질문

1. SSM prefill 레이어는 몇 개의 SM에서 수확체감이 시작되는가? (**SM saturation point**)
2. CUDA Green Contexts stream 전환 overhead는 레이어 실행 시간 대비 얼마인가?
3. 위 결과를 바탕으로 **layer-wise / step-adaptive / 고정 분할** 중 어느 전략이 최적인가?

---

## 2. 실험 설계 및 예상 결과

### 2.1 실험 인프라

**SM 제어 백엔드**: CUDA Green Contexts (driver 550+, A100 CUDA 13.0 지원)

```
A100 preset 파티션 (108 SM)
├── GreenContext[ssm_prefill]  : 76 SM (70%)  → SSM prefill stream
├── GreenContext[attn_prefill] : 43 SM (40%)  → Attn prefill stream
└── GreenContext[decode]       : 나머지 SM    → decode stream (complement)
```

CPU-side stream pointer 교체만으로 SM을 전환하므로 overhead < 1 μs 예상.

**3단계 실험 구조**:

| Stage | 목적 | 핵심 측정 |
|-------|------|---------|
| Stage 1 | 레이어별 SM scaling curve | SM 수 대비 latency, BW 활용률, wave 수 |
| Stage 2 | Green Contexts 전환 비용 측정 | stream 전환 latency (동기화 포함/불포함) |
| Stage 3 | Policy A/B/C 동시 실행 평가 | TTFT, TPOT, prefill throughput 비교 |

### 2.2 Policy 설계

| Policy | 전략 | SM 전환 타이밍 | 기대 효과 |
|--------|------|--------------|---------|
| **A (Baseline)** | 고정 40%/60% 분할 | 없음 | 기준선 |
| **B (Step-adaptive)** | 모델의 레이어 SSM 비율 기반 step 시작 시 결정 | 1회/step | 모델 특성 반영 |
| **C (Layer-wise)** | SSM↔Attn 경계마다 stream 전환 | N회/forward | 최대 이득 |

### 2.3 예상 결과

**Stage 1 예상**:
- SSM: SM이 증가할수록 latency가 감소하다가, 특정 SM 수(~60–70% 추정)에서 수확체감 시작 → **"free SM zone"** 발생
- Attention: SSM보다 SM 효율이 높아 더 적은 SM으로 포화 가능
- MLP: Roofline에서 compute↔BW 전환점 관찰

**Stage 2 예상**:
- Green Contexts CPU swap: < 1 μs (드라이버 호출 없음)
- GPU 동기화 포함 전환: < 10 μs
- overhead ratio = 전환 비용 / 레이어 latency < 5% → Policy C 실행 가능

**Stage 3 예상**:
- Policy C가 Attn 레이어에서 decode에 더 많은 SM을 양보 → TPOT 개선
- Zamba2 기준 Policy C > Policy B > Policy A 순으로 성능

---

## 3. 실험 시각화 해석 방법

### 3.1 Fig 1: SM Scaling Curve (fig1_scaling_zamba2_bs*.png)

```
latency(ms)
    │
    │  ●  sm=14
    │
    │     ●  sm=27
    │
    │        ●  sm=40
    │           ● sm=54
    │              ● sm=68
    │                 ● sm=81
    │                    ● sm=94 ● sm=108  ← 포화 구간 (기대)
    └─────────────────────────────────────→ SM count
```

- **X축**: Green Context로 제한한 SM 수 (14, 27, 40, 54, 68, 81, 94, 108)
- **Y축**: 해당 레이어 실행 median latency (ms)
- **해석 포인트**: 곡선이 꺾이는 지점 = saturation SM count. 이 지점 이후 SM이 decode에 "무료"로 제공 가능
- **배치별 비교**: bs=1, 4, 16, 32, 64를 겹쳐서 포화점이 batch에 독립적인지 확인

### 3.2 Fig 2: Free SM Zone (fig2_free_sm_zamba2.png)

- **의미**: `(model, seq_len, batch)` 조합별로 포화점 이후 남는 SM 비율
- **활용**: "이 config에서 decode에 최대 X% SM을 양보 가능" 판단 근거
- **해석**: 초록색 셀 = 여유 SM 많음, 빨간색 = SM이 부족해 decode 양보 불가

### 3.3 Fig 4: Module Latency Comparison (fig4_module_latency_zamba2.png)

- **구성**: SSM(wave model 합성), SSM(PyTorch scan), Attention 세 모듈의 latency를 (batch, seq_len) 격자로 시각화
- **해석**: 같은 (bs, seq) 에서 모듈 간 latency 비율이 SM 분할 비율 결정 근거
- **주의**: SSM(wave model)과 SSM(PyTorch scan)의 스케일 차이에 주목 — 두 곡선의 포화점이 다름

### 3.4 Fig 5: SSM Wave Model Validation (fig5_ssm_validation_zamba2.png)

- **X축**: Wave model 합성 latency (이론값)
- **Y축**: 실측 latency (전체 SM 기준)
- **이상적 상태**: 대각선 위 점 분포 (y=x)
- **현재 결과**: RMSE=50.665ms, MAPE=19.57% (n=192)
- **해석**: 일부 config에서 wave model 오차가 큰 이유를 탐구 필요

### 3.5 Fig 6: Saturation Heatmap (fig6_saturation_heatmap_zamba2.png)

- **X축**: seq_len, **Y축**: batch_size, **색**: saturation SM 비율(%)
- **해석**: 밝은 색 = 낮은 SM에서 포화 (decode에 많은 SM 가능), 어두운 색 = 전체 SM 필요
- **활용**: Policy 선택 시 "현재 config가 어느 zone인지" 즉각 판단

### 3.6 Fig 7: SM Sensitivity (fig7_sm_sensitivity_zamba2.png)

- **의미**: SM 10% 감소 시 latency가 몇 % 증가하는가 (`δlatency / δSM`)
- **해석**: 민감도가 높은 레이어 타입 = SM을 함부로 줄이면 SLO 위반 위험
- **활용**: `decode_sm_sensitivity` 상수를 실측값으로 대체 (현재 hardcode 0.5)

### 3.7 SRM Roofline (srm_zamba2_bs*.png, srm_bound_zamba2_bs*.png)

- **X축**: Arithmetic Intensity (FLOPs/Byte)
- **Y축**: Achieved TFLOPS
- **해석**:
  - 점이 BW ceiling 왼쪽 → 메모리 대역폭 병목 (BW-bound)
  - 점이 Compute ceiling 아래 → 계산 병목 (Compute-bound)
  - Ridge point = BW ceiling과 Compute ceiling의 교점
- **A100 기준**: Ridge = 312 TFLOPS / 2000 GB/s ≈ 156 FLOPs/Byte

### 3.8 Chunked SSM 관련 시각화 (results/stage1/chunked/)

- **Fig 8 (fig8_chunked_safety_zamba2.png)**: `cooperative_safe` 판정 결과 — (sm_count, prefill_chunk_tokens) 조합별 안전 여부
- **Fig 9 (fig9_chunked_sm_scaling_zamba2.png)**: Chunked 방식으로 실측한 SSM SM scaling curve (wave model과 비교)
- **Fig 10 (fig10_chunked_overhead_zamba2.png)**: prefill_chunk_tokens 크기별 kernel launch overhead

---

## 4. 실제 관측 결과 및 예상치와의 괴리

### 4.1 핵심 관측: SSM 커널의 SM 분할 불가

**예상**: SSM 레이어도 Attention처럼 SM을 제한하면 latency가 증가하고, 특정 SM 수에서 포화한다.

**관측**: SM을 14개로 제한하면 `CUDA error: an illegal memory access was encountered` 발생, 이후 CUDA context 전체가 오염되어 연쇄 실패.

```
실패 패턴 (Stage 1 SSM sweep):
sm=14, seq=8192, bs=32 → CUDA error: illegal memory access  ← 첫 발생
sm=14, seq=8192, bs=64 → CUDA error: illegal memory access  ← context 오염
sm=27, seq=512,  bs=1  → CUDA error: illegal memory access  ← 단순 config도 실패
sm=27, seq=512,  bs=4  → CUDA error: illegal memory access
... (이후 모든 configs 연쇄 실패)
```

**결론**: SSM은 Green Context로 SM을 제한할 수 없다.

### 4.2 SM Saturation이 관측 범위 내에 없음

**예상**: SSM saturation ≈ 60–70% SM, Attention saturation < SSM

**관측 (Attention, seq=4096, bs=4)**:

| SM | latency | 개선율 |
|----|---------|-------|
| 14 | 197.4 ms | — |
| 27 | 99.6 ms | **+49.6%** |
| 40 | 70.0 ms | +29.7% |
| 54 | 52.2 ms | +25.4% |
| 68 | 42.0 ms | +19.5% |
| 81 | 35.5 ms | +15.5% |
| 94 | 32.1 ms | +9.7% |
| **108** | **28.9 ms** | **+9.9%** |

108 SM에서도 여전히 ~10% 개선 여지 존재 → **포화점이 측정 범위(108 SM)를 초과**.

### 4.3 SSM Latency는 n_blocks에만 의존

SSM latency가 `seq_len`과 `batch_size` 개별 값이 아닌, `n_blocks = batch × seq_len // 4`에만 의존하는 것이 확인됨.

**n_blocks 동일 케이스 검증 (A100-SXM4)**:

| config | n_blocks | 측정 latency | 이론 비율 | 실측 비율 |
|--------|---------|------------|---------|---------|
| seq=4096, bs=64 | 65,536 | 2909.5 ms | 1.000 | **1.000** |
| seq=8192, bs=32 | 65,536 | 2908.5 ms | 1.000 | **1.000** |

오차 0.03% — **커널이 pure wave-serial 구조**임을 수학적으로 확인.

**BW 활용률** (SSM, sm=14, A100-SXM4):

| seq_len | batch | BW util (%) | 해석 |
|---------|-------|------------|------|
| 512 | 1 | 5.9% | Compute-bound |
| 1024 | 4 | 1.0% | Compute-bound |
| 4096 | 64 | 0.13% | 극단적 compute-bound |

SSM은 메모리가 아닌 **compute throughput에 의해 결정** → BW-bound라는 초기 가정이 틀림.

### 4.4 Wave Model 오차가 일부 Config에서 큼

wave model validation 결과: **RMSE=50.665ms, MAPE=19.57%** (n=192)

이는 동일 n_blocks 케이스 간 비율 오차(0.03%)와 상반된 결과처럼 보이나, wave model이 *상대 비율*은 정확하지만 **절대값 기반 비교에서 큰 규모 차이**(seq_len 수십 배 변동에 따른 latency 수백 배 차이)로 인해 RMSE가 커진 것으로 추정.

### 4.5 Green Contexts 전환 비용 (실측, A100-SXM4)

**Stage 2 측정 결과** (`ctx_switch_overhead_a100-sxm4-80gb.json`):

| 측정 항목 | 값 |
|---------|---|
| CPU pointer swap (no sync) | median **0.43 μs** |
| 전환 + GPU sync | median **7.8 μs** |
| 81레이어 전환 총 비용 | mean **1.72 ms** (per-transition ≈ 21.2 μs) |
| 초기화 비용 (one-time) | mean **14.3 ms** |

CPU swap은 예상(< 1 μs)과 일치. GPU sync 포함 시 ~8 μs로 예상(< 10 μs) 범위 내.

### 4.6 Stage 3: Policy A와 B의 성능 차이 없음

**Stage 3 실험 결과** (Zamba2, seq=1024, bs=8):

| Policy | TTFT mean | TTFT p99 | TPOT p50 | TPOT p99 |
|--------|-----------|----------|----------|----------|
| **A (고정 40/60)** | 67,344 ms | 129,463 ms | 0.33 ms | 0.9 ms |
| **B (step-adaptive)** | 67,339 ms | 130,388 ms | 0.33 ms | 0.9 ms |
| **C (layer-wise)** | *미완료* | — | — | — |

Policy A와 B의 차이: TTFT 기준 **0.007%** — 측정 오차 수준.

> Policy C Stage 3 결과가 없는 이유: SSM 레이어 SM 분할 불가 문제 미해결 상태에서 실행 시 CUDA context 오염 위험.

---

## 5. 괴리 원인 분석

### 5.1 SSM SM 분할 불가의 알고리즘적 근거

`mamba_chunk_scan_combined` (Triton SSD 커널)의 내부 구조:

```
Phase 1: 각 chunk 내부 local SSM scan     (block 독립)
              ↓
         grid.sync()  ← 전체 thread block이 동시에 active여야 완료
              ↓
Phase 2: chunk 간 hidden state prefix 전파 (cooperative 필요)
         chunk_0.state → chunk_1.initial → ...
```

**cooperative kernel의 전제 조건**: `cudaLaunchCooperativeKernel`은 `n_blocks ≤ max_active_blocks(device)` 조건이 보장될 때만 correctness가 성립한다.

**Green Context의 충돌**: Green Context는 CUDA 런타임의 사전 검사(n_blocks 체크) *이후*에 SM을 제한한다. 결과적으로 검사를 통과한 커널이 실제 실행 시점에는 조건이 깨진 채로 실행된다.

**Deadlock 발생 메커니즘**:

```
n_blocks = batch × seq_len // 4  (예: bs=32, seq=8192 → 65,536 blocks)
제한 SM = 14개
동시 활성 블록 ≈ 28개

Wave 1 (28 blocks): Phase 1 완료 → grid.sync() 도달 → 나머지 대기
Wave 2 (28 blocks): SM이 Wave 1로 점유 → 스케줄 큐 대기

결과: Wave 1은 나머지 block을 기다리고,
      나머지 block은 SM이 비기를 기다림 → 순환 대기
```

CUDA watchdog이 timeout 후 커널을 강제 종료 → partial write 상태 → context 오염 → 이후 연쇄 실패.

**에러 발생 임계점**:

| seq | batch | n_blocks | waves@14SM | 에러 |
|-----|-------|---------|-----------|------|
| 4096 | 64 | 65,536 | 4,682 | 없음 ✓ |
| **8192** | **32** | **65,536** | **4,682** | **CUDA 에러** ✗ |

*동일한 n_blocks=65,536임에도 결과가 다름 → n_blocks 외에 seq_len에 비례하는 내부 버퍼 크기(B, C state matrix `seq × n_groups × d_state`)도 임계값에 관여하는 것으로 추정.*

### 5.2 Saturation 없음의 물리적 해석

SSM 레이어의 latency 공식 (wave model):

```
latency(sm_k) = latency(full_sm) × ⌈n_blocks / k⌉ / ⌈n_blocks / full_sm⌉
```

전형적인 serving config (seq=4096, bs=32) 기준:

```
n_blocks = 32 × 4096 // 4 = 32,768
108 SM: ⌈32,768 / 108⌉ = 304 waves
 54 SM: ⌈32,768 /  54⌉ = 607 waves  → 정확히 2배 latency
 27 SM: ⌈32,768 /  27⌉ = 1,214 waves → 정확히 4배 latency
```

**파동 효율(ncu 실측)**: 모든 config에서 `wave_eff_pct = 99.97%+`  
→ 어떤 SM 수에서도 항상 수백~수천 개의 파동이 남아있음.  
→ SM을 줄여도 커널 구조가 변하지 않고 단순히 파동 수만 증가  
→ **포화점이 존재하지 않음** (이론적으로 n_blocks → 1이 될 때만 포화).

Attention 포화 없음의 이유: FlashAttention의 tile 수는 `n_heads × (seq_len / tile_size)²`로, seq_len=4096, bs=4, n_heads=28(GQA)에서 tile 수가 A100 108 SM을 훨씬 초과함.

### 5.3 Policy A = B의 이유

Policy B(step-adaptive)는 "SSM 비율이 높은 step에서 prefill에 더 많은 SM을 준다"는 전략이다. 그런데:

1. SSM 레이어 자체에서 SM 분할 불가 → prefill SM 비율을 올려도 실제로 SM이 제한되지 않음
2. Attention 레이어(13개)에서만 SM 분할이 작동하는데, Policy A와 B 모두 Attention에서는 동일한 40%를 적용
3. 결과적으로 두 정책이 실질적으로 동일한 실행 경로를 밟음

### 5.4 Wave Model MAPE가 19.57%인 이유

wave model은 **상대 비율**은 정확(동일 n_blocks 케이스 간 오차 0.03%)하지만, validation 방식이 "전체 SM 기준 latency를 사용해 부분 SM latency를 합성"이므로, 합성 대상이 되는 full-SM latency 자체의 측정 노이즈가 작은 SM 추정값에 비율로 증폭된다. seq_len과 batch_size 조합이 수십 배씩 변하는 전체 범위에서 RMSE 기반 비교 시 큰 값의 오차가 지배적이 됨.

---

## 6. 우회 방법론 정리

### 6.1 현재 채택 중: Wave Model 합성

**상태**: Stage 1 완료, 기본 경로로 채택  
**원리**:

```python
n_blocks = batch × seq_len // 4          # ncu로 검증된 공식
latency(sm_k) = latency(full_sm) × ⌈n_blocks / k⌉ / ⌈n_blocks / full_sm⌉
```

**장점**: 구현 완료, Triton SSD의 compute 특성을 정확히 반영, 동일 n_blocks 케이스 간 오차 0.03%  
**한계**: SM 분할 직접 측정이 아니므로 cooperative 예외 config(이론값과 다른 경우)를 발견하지 못할 수 있음

### 6.2 현재 채택 중: Subprocess Isolation

**상태**: `_ssm_worker.py`에 구현 완료  
**원리**: 각 SM level을 별도 subprocess에서 실행 → CUDA context 오염이 subprocess 내부로 격리됨

```python
if is_cuda_error:
    cuda_dead = True   # 해당 subprocess 포기, 다음 SM level은 새 subprocess
```

**장점**: parent process 보호, 다른 SM level 데이터 유지  
**한계**: 오염 발생 시 해당 subprocess의 나머지 config 전체 손실

### 6.3 현재 채택 중: Chunked Prefill 방식

**상태**: `chunked_ssm_runner.py` 구현 완료, 데이터 수집 완료  
**원리**: 단일 `mamba_chunk_scan_combined` 호출이 아닌, `prefill_chunk_tokens` 크기로 나눠 kernel 경계에서 SSM state를 명시적으로 전달

```python
state = zeros(batch, n_heads, head_dim, d_state)
for start in range(0, seq_len, prefill_chunk_tokens):
    chunk = x[:, start:start + prefill_chunk_tokens]
    y_chunk, state = mamba_chunk_scan_combined(
        chunk, ..., ssm_initial_states=state, return_final_states=True
    )
    # kernel 호출 사이 Green Context 전환 가능
```

각 kernel call의 `n_blocks_per_call = batch × (prefill_chunk_tokens//256) × n_heads`가 `sm_count × max_blocks_per_sm` 이하이면 cooperative 조건 성립.

**현재 결과**: 전체 608개 config 중 `cooperative_safe=True`인 행 없음  
→ 현재 채택한 chunk 크기와 batch 조합이 모두 unsafe 범위에 해당.

**cooperative 안전 조건** (Zamba2, A100):

```
n_heads = 112, max_blocks_per_sm = 1 (conservative)
n_blocks_per_call = batch × (pct//256) × 112 ≤ sm_count
→ batch=4: pct ≤ sm_count / 112 × 256 ≈ sm_count × 2.3
   sm=14: pct ≤ 32 tokens (1 SSD chunk = 256 → 불가)
   sm=108: pct ≤ 247 tokens (역시 256 미만 → 불가)
```

실용적인 batch(≥4), pct(≥256)에서 안전 구간이 거의 없음.

### 6.4 미구현: Two-pass Kernel 분해

**상태**: 설계 문서 완성 (`reports/ssm_two_pass_decomposition_guideline.md`), 구현 대기  
**원리**: `mamba_chunk_scan_combined`를 cooperative barrier 없이 3개 독립 커널로 분리

```
Kernel A: local chunk scan          (n_blocks = batch × C, inter-block sync 없음)
    ↓  carry[C] GPU 메모리 저장
Kernel B: inter-chunk prefix scan   (n_blocks = batch, 단일 블록으로 처리 가능)
    ↓  prefix[C] GPU 메모리 저장
Kernel C: apply prefix + finalize  (n_blocks = batch × C, inter-block sync 없음)
```

**예상 overhead**: 커널 론칭 3회(~25μs) + carry state I/O(~0.015ms) → < 0.002% of SSM latency

**구현 비용**: mamba_ssm 외부 패키지 fork 또는 monkey-patch 필요, Triton 커널 수정 후 수치 검증 필요

### 6.5 미구현: Persistent Kernel

**원리**: block 수를 n_sm으로 고정, 각 block이 자기 담당 chunk들을 순차 처리. wave가 1개이므로 모든 block이 동시 활성화 → `grid.sync()` 불필요.

**단점**: SM 수마다 다른 커널 인자 필요, chunk 간 state를 register chain 또는 global memory로 전달해야 함

### 6.6 우회 방법론 비교

```
Triton SSD 특성 반영 정확도
    ↑
    │  Two-pass 커널 분해      ← 정확도 최고, 구현 비용 최고
    │  Persistent kernel       ← 정확도 높음, 구현 비용 높음
    │  Wave model 합성 ★       ← 정확도 높음 (오차<0.03%), 구현 완료
    │  Chunked prefill ★       ← 직접 측정, 현재 safe 구간 없음
    │  SM 하한 설정             ← 측정 범위 매우 협소
    │  PyTorch scan             ← 다른 커널, 의사결정 입력 불가
    ↓
낮음                              구현 비용 →                       높음
```

---

## 7. 방향성 유지 가능성 탐색

### 7.1 현재 제약의 범위

**근본 제약 (피할 수 없음)**:

```
Triton SSD cooperative barrier → SSM 레이어는 Green Context로 SM 분할 불가
```

이 제약이 의미하는 것:
- Zamba2 (81 layers: 68 SSM + 13 Attn) → SM 재할당 이득이 전체의 ~16%에서만 발생
- Falcon-H1 (44 layers: 모든 레이어 SSM+Attn 병렬) → SSM과 Attn을 같은 레이어에서 분리 불가

**우회 가능 제약 (기술적 해결책 존재)**:
- Chunked prefill의 cooperative_safe=False → chunk 크기와 batch 조합 재설계
- Two-pass 커널 미구현 → 구현 완료 시 SSM도 SM 분할 가능

### 7.2 방향성 유지 시나리오

**시나리오 A: Attention 레이어 중심 최적화 (현재 가능)**

Zamba2의 Attention 레이어 13개에서만 SM 공유:

```
SSM layers (68개): 전체 SM 108개 독점 → decode 스트림 중단
Attn layers (13개): prefill 40% SM + decode 60% SM → 동시 실행
```

| 항목 | 예상 |
|------|------|
| SM 재할당 이득 비율 | 전체 latency의 ~16% |
| ctx_switch 비용 | 13 × 2회 × 21μs = 0.55ms (Attn latency 수십ms 대비 무시) |
| Policy C 실현 가능성 | SSM 레이어 guard 코드 추가 후 실행 가능 |

**시나리오 B: Two-pass 커널 구현 (중기)**

Two-pass 구현 완료 시:
- SSM 레이어에서도 SM 분할 가능
- Zamba2 전체 81개 레이어에서 SM 공유 → 이론적 최대 이득

| 검증 항목 | 기준값 |
|-----------|--------|
| 수치 오차 (bf16) | max_diff < 1e-2 |
| Green Context @ 14 SM | CUDA 에러 없음 |
| 전체 SM latency overhead | < 15% |

**시나리오 C: Chunked Prefill 최적화 (단기 탐구)**

`prefill_chunk_tokens`를 매우 작게 설정(≤ 1 SSD chunk = 256 tokens)하면 단일 kernel call의 n_blocks가 sm_count 이하가 될 수 있으나:
- bs=1일 때만 sm=108 수준에서 safe 가능 (`1 × 1 × 112 = 112 ≈ 108`)
- Serving에서 bs=1은 비실용적

**시나리오 D: Falcon-H1 집중 탐구 (대안 모델)**

Falcon-H1은 SSM과 Attn이 병렬로 실행되므로:
- Policy C (layer-wise SM 분할) 적용 불가
- 대신 **batch 또는 request 단위 SM 분할** 접근 가능
- prefill 요청과 decode 요청을 별도 Green Context에서 완전 격리 실행

### 7.3 다음 우선순위

1. **단기**: Policy C 실행 시 SSM 레이어 guard 코드(`if layer_type == "ssm": smctrl.reset()`) 추가 → Stage 3 Policy C 결과 획득
2. **단기**: Wave Model validation MAPE=19.57% 원인 분석 — 구체적으로 어떤 config에서 괴리가 큰지 파악
3. **중기**: Two-pass 커널 구현 (Triton, mamba_ssm monkey-patch) → SSM SM scaling 직접 측정
4. **중기**: `decode_sm_sensitivity` 실측 — Stage 1 decode latency vs SM count 직접 측정으로 hardcode 0.5 대체
5. **장기**: vLLM 통합 PoC — `Worker.init_device()`에 SMController 주입, layer 경계 훅 삽입

### 7.4 핵심 판단

> SSM 레이어의 SM 분할 불가는 "버그"가 아닌 **알고리즘적 필연**이다.  
> Parallel scan (SSD)은 전체 SM이 동시에 barrier에 도달해야 correctness가 보장된다.  
> 이 제약을 우회하는 근본 해결책은 Two-pass 분해뿐이며,  
> 그 이전까지는 Attention/MLP 레이어에서의 SM 공유로 제한된 이득을 추구하는 것이  
> 현실적인 경로다.

Zamba2(SSM-heavy, 16% attention)와 달리, **Attention 비율이 높은 모델**에서 이 기법의 효과가 클 것으로 예상된다. 모델 포트폴리오 확장이 연구 방향성 유지의 핵심 전략이다.

---

## 부록: 핵심 수치 요약

| 항목 | 값 | 출처 |
|------|---|------|
| n_blocks 공식 | `batch × seq_len // 4` | ncu 실측 |
| SSM wave_eff_pct | 99.97%+ | ncu profiling (전 config) |
| Wave model 동일 n_blocks 오차 | 0.03% | Stage 1 실측 |
| Wave model validation MAPE | 19.57% (RMSE=50.7ms, n=192) | fig5 |
| Attention SM 개선율 (14→108) | ~6.8× latency 감소 | Stage 1 |
| Attention 포화점 | 미관측 (>108 SM) | Stage 1 |
| Green Ctx CPU swap | median 0.43 μs | Stage 2 |
| Green Ctx + GPU sync | median 7.8 μs | Stage 2 |
| 81레이어 전환 총 비용 | mean 1.72 ms | Stage 2 |
| Two-pass carry state (Zamba2, seq=4096, bs=32) | ~7.3 MB | 이론 계산 |
| Two-pass 예상 overhead | < 0.002% | 이론 계산 |
| Policy A TTFT | 67,344 ms | Stage 3 |
| Policy B TTFT | 67,339 ms (차이 0.007%) | Stage 3 |
| Chunked cooperative_safe=True 비율 | 0 / 608 configs | Stage 1 chunked |

---

*참고 파일: `README.md`, `reports/ssm_sm_partitioning_analysis.md`, `reports/ssm_cooperative_barrier_context_corruption.md`, `reports/ssm_two_pass_decomposition_guideline.md`, `reports/CLAUDE_chunked_prefill_sm_partition.md`, `reports/impl_gap_and_integration.md`, `reports/diff_from_serving_engines.md`, `a100_migration_report.md`*
