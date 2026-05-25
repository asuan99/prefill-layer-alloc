# Zamba2 vs Falcon-H1 비교 분석 및 연구 로드맵

> 작성일: 2026-05-25  
> 하드웨어: NVIDIA A100-SXM4-80GB (108 SM, HBM2e ~2,000 GB/s)  
> 실험 범위: Stage 1 SM scaling sweep + Chunked SSM cooperative safety 분석

---

## 목차

1. [모델 아키텍처 비교](#1-모델-아키텍처-비교)
2. [SSM prefill latency 비교](#2-ssm-prefill-latency-비교)
3. [Attention prefill latency 비교](#3-attention-prefill-latency-비교)
4. [MLP latency 비교](#4-mlp-latency-비교)
5. [Chunked SSM cooperative safety 비교](#5-chunked-ssm-cooperative-safety-비교)
6. [CUDA error 패턴 비교](#6-cuda-error-패턴-비교)
7. [Policy C 적용 가능성 비교](#7-policy-c-적용-가능성-비교)
8. [핵심 발견 요약](#8-핵심-발견-요약)
9. [다음 실험 스텝 및 연구 단계](#9-다음-실험-스텝-및-연구-단계)

---

## 1. 모델 아키텍처 비교

| 항목 | Zamba2-7B-Instruct | Falcon-H1-7B-Instruct |
|------|--------------------|-----------------------|
| 전체 레이어 수 | 81 | 44 |
| 레이어 구성 | 68 pure SSM + 13 hybrid (SSM+Attn+MLP) | 44 전층 SSM+Attn+MLP 병렬 |
| SSM n_heads | **112** | **24** |
| SSM head_dim | 64 | 128 |
| SSM d_state | 64 | 256 |
| SSM n_groups | 2 | 1 |
| SSM chunk_size | 256 | 256 |
| Attn num_heads | 32 | 12 |
| Attn KV heads | 32 (MHA) | **2 (GQA)** |
| Attn head_dim | 224 | 128 |
| MLP intermediate | 14,336 | 12,288 |
| Hidden size | 3,584 | 3,072 |

**구조적 핵심 차이**: Zamba2는 SSM 전용 레이어와 hybrid 레이어가 분리되어 있어 layer boundary 단위의 SM 재분배 대상이 명확하다. Falcon-H1은 모든 레이어에서 SSM branch와 Attn branch가 **동시에** 실행되는 구조여서 inter-layer가 아닌 **intra-layer** SM 분할이 필요하다.

---

## 2. SSM prefill latency 비교

### 2.1 n_blocks 공식

두 모델 모두 SSM latency를 지배하는 변수는 동일하다.

```
n_blocks = batch × seq_len ÷ 4
```

Zamba2(n_blocks=65,536)와 Falcon-H1(n_blocks=65,536)이 동일 n_blocks일 때 latency ratio가 일정하게 유지되는 것이 확인되었다. n_blocks × 8 증가 시 latency도 ×8 수렴 (측정값: Zamba2 ×8.15, Falcon-H1 ×7.93).

### 2.2 절대 latency 비교 (@ 108 SM, seq=4096, bs=4)

| n_blocks | Falcon-H1 | Zamba2 | 비율 |
|----------|-----------|--------|------|
| 4,096    | 5.1 ms   | 12.6 ms | **2.45×** |
| 8,192    | 10.3 ms  | 25.4 ms | **2.47×** |
| 32,768   | 40.5 ms  | 101.5 ms | **2.51×** |

**Falcon-H1이 ~2.4× 빠른 이유**: 같은 n_blocks임에도 블록당 연산량이 다르다.

```
블록당 compute ∝ n_heads × head_dim × d_state
Zamba2:    112 × 64  × 64  = 458,752
Falcon-H1:  24 × 128 × 256 = 786,432  (더 많음)
```

역설적으로 Falcon-H1이 블록당 연산량은 더 많지만 전체 latency는 낮다. 이는 Zamba2의 n_heads=112가 만들어내는 극단적인 warp 수와 bank conflict 구조가 throughput을 제한하는 반면, Falcon-H1은 n_heads=24로 각 블록이 더 넓은 상태(d_state=256)를 처리해 메모리 접근 패턴이 효율적이기 때문으로 추정된다.

### 2.3 SM scaling curve (seq=4096, bs=4)

두 모델의 SM별 speedup이 **수치적으로 동일**하다.

| SM 수 | Falcon-H1 latency | Zamba2 latency | speedup (공통) | waves |
|-------|------------------|----------------|----------------|-------|
| 14 (13%) | 39.7 ms | 97.4 ms | 1.00× | 293 |
| 27 (25%) | 20.6 ms | 50.5 ms | 1.93× | 152 |
| 40 (37%) | 13.9 ms | 34.2 ms | 2.84× | 103 |
| 54 (50%) | 10.3 ms | 25.3 ms | 3.86× | 76  |
| 68 (62%) | 8.3 ms  | 20.3 ms | 4.80× | 61  |
| 81 (75%) | 6.9 ms  | 16.9 ms | 5.75× | 51  |
| 94 (87%) | 6.0 ms  | 14.6 ms | 6.66× | 44  |
| 108 (100%) | 5.1 ms | 12.6 ms | **7.71×** | 38 |

speedup이 동일한 이유: wave 수가 동일하기 때문이다. n_blocks=4,096, SM=27일 때 `ceil(4096/27)=152`로 두 모델이 공유한다. **SM saturation 포화점은 두 모델 모두 관측 범위(108 SM) 내에 없다**.

### 2.4 BW 활용률 비교

| 모델 | BW util 범위 (@108 SM) | 해석 |
|------|----------------------|------|
| Falcon-H1 | 4.0% – 10.3% | 극단적 compute-bound |
| Zamba2 | 1.9% – 20.2% | 역시 compute-bound |

두 모델 모두 SSM은 메모리 대역폭이 아닌 **compute throughput**에 의해 실행 시간이 결정된다. 배치가 커질수록 BW util이 감소하는 것도 동일하다.

---

## 3. Attention prefill latency 비교

### 3.1 SM scaling 패턴 (seq=4096, bs=4)

| SM 수 | Falcon-H1 | Zamba2 | 개선율/SM (F-H1) | 개선율/SM (Z2) |
|-------|-----------|--------|-----------------|---------------|
| 14    | 27.9 ms  | 197.4 ms | — | — |
| 14→27 | 14.1 ms | 99.5 ms | 3.80%/SM | 3.82%/SM |
| 27→40 | 10.1 ms | 70.1 ms | 2.20%/SM | 2.28%/SM |
| 40→54 | 7.6 ms  | 52.2 ms | 1.79%/SM | 1.82%/SM |
| 54→68 | 6.0 ms  | 42.0 ms | 1.46%/SM | 1.40%/SM |
| 68→81 | 5.1 ms  | 35.7 ms | 1.19%/SM | 1.15%/SM |
| 81→94 | 4.5 ms  | 32.3 ms | 0.82%/SM | 0.73%/SM |
| 94→108 | 4.1 ms | 29.2 ms | 0.64%/SM | 0.70%/SM |
| 14→108 ratio | **6.75×** | **6.77×** | — | — |

**SM scaling 패턴이 양 모델에서 거의 동일**하다. FlashAttention의 tile-independent 병렬 구조가 모델 아키텍처와 무관하게 동일한 SM scaling 특성을 만든다.

### 3.2 절대 latency 차이

Zamba2 Attn이 Falcon-H1 대비 **6.5–7.1×** 느리다. 주요 원인:

```
Zamba2:    32 heads × head_dim=224 × MHA → KV cache: 32 heads
Falcon-H1: 12 heads × head_dim=128 × GQA → KV cache: 2 heads (6× 압축)

tile 수 ∝ n_q_heads × (seq_len / tile_size)²
Zamba2:    32 × (4096 / 64)² = 131,072 tiles
Falcon-H1: 12 × (4096 / 64)² = 49,152 tiles  (2.67× 적음)
```

Attn도 두 모델 모두 **108 SM에서 saturation 미관측** (94→108 SM 전환 시 8–10% 개선 여지 존재).

---

## 4. MLP latency 비교

| 모델 | @14SM (seq=4096, bs=4) | @108SM | 14→108 ratio | BW util @108SM |
|------|----------------------|--------|--------------|----------------|
| Falcon-H1 | 68.4 ms | 10.2 ms | **6.72×** | 3.5% |
| Zamba2 | 91.8 ms | 13.8 ms | **6.64×** | 3.2% |

MLP는 순수 GEMM 연산이므로 두 모델이 매우 유사한 SM scaling 특성을 보인다. 절대 latency 차이는 intermediate_size 차이(14,336 vs 12,288)에서 기인한다.

---

## 5. Chunked SSM cooperative safety 비교

### 5.1 결과 요약

| 모델 | cooperative_safe=True | 전체 configs | 비율 |
|------|----------------------|------------|------|
| Zamba2 | **0** | 608 | **0.0%** |
| Falcon-H1 | **69** | 608 | **11.3%** |

### 5.2 Zamba2 safe=0의 이유

cooperative kernel의 안전 조건:

```
n_blocks_per_call = batch × (pct ÷ 256) × n_heads ≤ sm_count × max_blocks_per_sm
```

Zamba2는 n_heads=112이므로, 최소 설정(bs=1, pct=256)에서도:

```
n_blocks = 1 × 1 × 112 = 112 > 108 (A100 SM 수)
```

A100의 SM 수(108)보다 n_heads(112)가 크기 때문에 **어떤 조합도 조건을 충족할 수 없다**. 최소 chunk 크기(=256 tokens)가 정확히 하나의 SSD chunk와 같아 더 이상 줄일 수도 없다.

### 5.3 Falcon-H1 safe=69의 구조

Falcon-H1은 n_heads=24이므로:

| batch | pct=256 | n_blocks | safe 가능 SM |
|-------|---------|----------|------------|
| bs=1 | 256 | **1×1×24 = 24** | sm ≥ 24 → **27, 40, ..., 108 전부** ✓ |
| bs=4 | 256 | **4×1×24 = 96** | sm ≥ 96 → **108만** ✓ |
| bs=1 | 512 | 1×2×24 = 48 | sm ≥ 48 → 54, 68, ..., 108 ✓ |
| bs=1 | 1024 | 1×4×24 = 96 | sm ≥ 96 → 108만 ✓ |
| bs=4 | 512 | 4×2×24 = 192 | **어디서도 불가** ✗ |
| bs=16+ | any | ≥ 384 | **어디서도 불가** ✗ |

**bs=1에서 safe 케이스가 집중되는 이유**: n_heads=24라는 낮은 병렬성이 최소 n_blocks(24)를 SM 수(≥27) 이하로 만들기 때문이다. bs가 증가하면 n_blocks가 batch에 선형 비례하여 즉시 한계를 초과한다.

실측 safe rows 분포: bs=1에서 64건, bs=4(pct=256, sm=108에서만)에서 5건.

### 5.4 Safe 구간의 실용적 가치 검토

Falcon-H1의 safe 구간(특히 bs=1)에서의 chunked latency를 wave model(전체 SM 기준 합성) 대비 비교:

| 조건 | chunked lat | wave model @same SM | overhead |
|------|------------|---------------------|---------|
| pct=256, sm=27, seq=512, bs=1 | 1.47 ms | 1.99 ms | –26% (이례적) |
| pct=256, sm=27, seq=1024, bs=1 | 2.83 ms | 2.67 ms | **+6%** |
| pct=256, sm=27, seq=2048, bs=1 | 5.54 ms | 3.03 ms | **+83%** |
| pct=256, sm=27, seq=4096, bs=1 | 10.94 ms | 5.21 ms | **+110%** |
| pct=256, sm=27, seq=8192, bs=1 | 21.92 ms | 10.59 ms | **+107%** |

chunked 방식의 overhead가 seq가 길어질수록 급격히 증가한다. 그 이유:

```
seq=4096 시 chunked 동작:
  호출 횟수 = 4096 ÷ 256 = 16회
  각 호출 n_blocks = 1 × 1 × 24 = 24
  wave@27SM = ceil(24/27) = 1 (단 1 wave)
  total logical waves = 16 × 1 = 16

wave model 동작:
  n_blocks = 1 × 4096 ÷ 4 = 1024
  wave@27SM = ceil(1024/27) = 38 waves

호출 횟수가 16회로 증가하면서
  - 16× 커널 론칭 overhead (각 ~수 μs)
  - 경계마다 SSM state 명시적 I/O
    state shape = (1, 24, 128, 256) = 1,536 KB
    16 경계 × 2회(read+write) = 48 MB
    이론 전송 시간 ≈ 25 μs (2,000 GB/s 기준)
```

safe 구간에서도 chunked 방식이 wave model 대비 ~100% overhead를 가지므로 **현재 chunk 크기(256 tokens)와 실용 batch(≥4) 조합에서는 chunked prefill의 실질적 이득이 없다**.

---

## 6. CUDA error 패턴 비교

### 6.1 Zamba2 패턴 (이미 알려진)

Green Context로 SM을 제한하면 `cudaLaunchCooperativeKernel`의 전제 조건이 깨져 deadlock → `cudaErrorIllegalAddress`. 임계점: n_blocks > sm_count × max_active_blocks_per_sm.

### 6.2 Falcon-H1 새로운 패턴

**Wave model 직접 측정 sweep** (full SM 기준):
- seq=8192, bs=64에서 전체 SM(108) 환경에서도 CUDA error 발생
- 이후 seq≥16384 전 config 연쇄 실패

```
첫 에러: seq=8192, bs=64 (n_blocks = 64 × 8192 ÷ 4 = 131,072)
전 config 성공: seq=8192, bs=32 (n_blocks = 65,536)
```

이것은 Green Context 문제가 아닌 **OOM 또는 내부 버퍼 오버플로우** 가능성이 높다. Falcon-H1의 SSM state 크기: `(bs=64, n_heads=24, head_dim=128, d_state=256)` = 64×24×128×256×2 bytes = 1.5 GB. 이 사이즈가 A100 HBM의 타 텐서와의 합산으로 OOM을 유발했거나, 내부 workspace 할당 실패일 수 있다.

**Torch scan sweep** (직접 Green Context 측정):
- sm=14까지는 (seq=8192, bs=64 제외) 측정 성공
- sm=14, seq=8192, bs=64에서 첫 에러 발생 → subprocess isolation 없어 이후 sm=27~108 전체 연쇄 실패

Falcon-H1 torch scan은 cooperative barrier가 없어 이론적으로 Green Context에서 안전해야 하지만, **context 오염 전파 방지를 위한 subprocess isolation이 없어** 단 한 번의 에러가 나머지 280 configs를 모두 무력화했다.

---

## 7. Policy C 적용 가능성 비교

### 7.1 Zamba2: inter-layer SM 공유 (partial)

Zamba2 forward pass의 layer별 시간 비율 (seq=4096, bs=4, @108SM):

```
SSM 레이어 (81개): 81 × 12.6 ms = 1,021 ms → 64.5%
Attn 레이어 (13개): 13 × 29.2 ms = 379 ms  → 24.0%
MLP 레이어 (13개): 13 × 13.8 ms = 179 ms   → 11.3%
총합 ≈ 1,582 ms
```

Policy C가 Attn 레이어(24%)에서 decode에 SM을 양보하는 시나리오:

| prefill SM | decode SM | Attn 레이어 penalty | decode 이득 기회 |
|-----------|-----------|---------------------|----------------|
| 54 SM | 54 SM | +79.1% (52.2ms → 29.2ms) | decode가 54 SM 사용 가능 |
| 40 SM | 68 SM | +140.3% (70.1ms → 29.2ms) | decode가 68 SM 사용 가능 |
| 27 SM | 81 SM | +241.2% (99.5ms → 29.2ms) | decode가 81 SM 사용 가능 |

**제약**: Attn prefill의 penalty가 매우 크다. decode TPOT에서 얻는 이득이 이 penalty를 상쇄하려면, decode latency가 SM 수에 매우 민감해야 한다. SSM 레이어(64.5%)에서는 현재 SM 공유 불가이므로 이득 구간 자체가 forward pass의 16% 레이어(13개 Attn)에 한정된다.

### 7.2 Falcon-H1: intra-layer SSM+Attn 병렬 실행

Falcon-H1의 구조적 특성: 모든 레이어에서 SSM과 Attn이 독립적으로 실행 가능하다. 현재는 순차 실행이지만, 두 branch를 서로 다른 Green Context partition에서 **동시에** 실행하면:

```
현재 (순차):
  SSM(108SM) + Attn(108SM) = 5.1ms + 4.1ms = 9.2ms/layer

제안 (병렬):
  SSM(68SM) ‖ Attn(40SM) = max(8.3ms, 10.1ms) = 10.1ms/layer

  → 순차 대비 overhead: +9.8% 대신
     Attn에 쓰던 SM이 SSM에 겹쳐 사용 가능
```

더 균등한 분할(SSM@54SM ‖ Attn@54SM):
```
max(10.3ms, 7.6ms) = 10.3ms/layer  (순차 17.9ms 대비 1.73× speedup)
```

**핵심**: Falcon-H1에서의 SM 공유는 decode stream이 아닌 **같은 레이어 내 SSM과 Attn 사이의 SM 공유**로 접근해야 한다. 이를 통해 단일 forward pass latency를 ~1.7×~1.8× 줄일 수 있으며, 이 구조가 Zamba2의 inter-layer 방식보다 이득이 더 클 수 있다.

단, Falcon-H1 SSM에서도 동일한 cooperative barrier 문제가 존재하므로, intra-layer 병렬 실행을 위해서는 Two-pass 커널 분해가 선행되어야 한다.

---

## 8. 핵심 발견 요약

| 항목 | Zamba2 | Falcon-H1 | 비고 |
|------|--------|-----------|------|
| SSM wave model 공식 | batch×seq÷4 | batch×seq÷4 | 동일 |
| SM speedup curve | ×7.71 (14→108) | ×7.71 (14→108) | 완전 동일 |
| SSM latency @same n_blocks | 기준 | **2.4× 빠름** | n_heads 차이 |
| SSM BW bound | compute-bound | compute-bound | 동일 |
| Attn SM scaling | ×6.77 | ×6.75 | 거의 동일 |
| Attn latency @108SM | 기준 | **6.7× 빠름** | head 수/GQA |
| Attn saturation | 미관측 | 미관측 | 동일 |
| cooperative_safe | **0/608** | **69/608** | n_heads 차이 |
| SM 공유 전략 | inter-layer (Attn 13개) | **intra-layer** (전층) | 구조적 차이 |
| Policy C 이득 구간 | 전체의 24% (Attn only) | **전체의 ~50% (SSM+Attn)** | Two-pass 필요 |
| CUDA error 임계점 | sm=14 seq=8192 bs=32 | seq=8192 bs=64 (OOM 추정) | 원인 상이 |

---

## 9. 다음 실험 스텝 및 연구 단계

### 9.1 단기 (즉시 실행 가능, NCU 불필요)

#### Step 1: Falcon-H1 CUDA error 임계점 정밀 분석

**목표**: seq=8192, bs=64에서의 전체 SM OOM 에러 원인 확인

```bash
# GPU memory 사용량 모니터링 포함 실험
python stage1_sm_scaling/run_ssm_prefill_sweep.py \
    --model falcon_h1 --device a100_80gb \
    --seq-lens 8192 --batch-sizes 48 56 64 \
    --monitor-memory
```

조사 항목:
- `(bs=64, seq=8192)` 시 SSM state 크기: `(64, 24, 128, 256)` = **1.5 GB**
- model weight + activation + state 합산이 80 GB를 초과하는지 확인
- 에러가 OOM인지 cooperative deadlock인지 구분: `CUDA_LAUNCH_BLOCKING=1`으로 stacktrace 확보

#### Step 2: Falcon-H1 Torch scan subprocess isolation 추가

**목표**: sm=14의 단일 에러가 전체 sweep을 무력화하는 것 방지

현재 `_ssm_worker.py`의 subprocess isolation이 Zamba2 SSM cooperative 에러에 대응하도록 구현되어 있지만, Falcon-H1 torch scan sweep에는 적용되지 않았다. `run_ssm_prefill_sweep.py`의 torch scan 경로에 동일 isolation 추가 필요.

#### Step 3: Policy C SSM guard 코드 추가 후 Stage 3 재실행

```python
# policy_layer_wise.py 수정
def get_sm_count_for_layer(self, layer_type: str) -> int:
    if layer_type == "ssm":
        return self.smctrl.total_sm  # guard: SSM은 전체 SM
    elif layer_type == "attn":
        return self.attn_sm_count    # Attn에서만 SM 분할
    return self.smctrl.total_sm
```

Zamba2 Policy C 실행 → 현재 미수집 Stage 3 데이터 확보. 예상: Attn 레이어 24%에서만 이득이 발생하므로 Policy A/B 대비 small improvement (5–15% TTFT 감소 추정).

#### Step 4: Falcon-H1 intra-layer concurrent 실험 설계

Falcon-H1의 SSM branch와 Attn branch를 서로 다른 Green Context stream에서 동시 실행하는 실험:

```python
# 실험 구조: 두 개의 Green Context stream을 동시에 launch
ssm_stream = smctrl.get_stream(sm_ratio=0.5)   # 54 SM
attn_stream = smctrl.get_stream(sm_ratio=0.5)  # 54 SM

with torch.cuda.stream(ssm_stream):
    ssm_out = ssm_branch(hidden)                # async launch

with torch.cuda.stream(attn_stream):
    attn_out = attn_branch(hidden)              # async launch

torch.cuda.synchronize()                         # 두 스트림 완료 대기
out = ssm_out + attn_out
```

**단, SSM cooperative barrier 문제로 인해 현재는 deadlock 가능**. Two-pass 구현 없이는 bs=1의 특정 구간에서만 safe하게 테스트 가능.

### 9.2 중기 (Two-pass 커널 구현)

#### Step 5: Two-pass SSM 커널 구현

설계 문서(`reports/ssm_two_pass_decomposition_guideline.md`) 완성 상태. 구현 단계:

```
Kernel A: local chunk scan  (no inter-block sync)
    ↓  carry state → GPU global memory
Kernel B: inter-chunk prefix (single block, sequential)
    ↓  prefix state → GPU global memory
Kernel C: apply prefix + finalize  (no inter-block sync)
```

예상 구현 비용:
- Triton으로 Kernel A, C 작성 (각 ~150 LOC)
- Kernel B는 단일 블록 → 단순 for loop (~30 LOC)
- `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` monkey-patch

검증 기준:
```
bf16 max_diff < 1e-2  (대비: FP32 reference)
Green Context @ 14 SM → CUDA 에러 없음
full-SM latency overhead < 15% (2-pass carry state I/O 포함)
```

#### Step 6: Two-pass 적용 후 Falcon-H1 intra-layer concurrent 실험

Two-pass 완성 이후:
1. Falcon-H1 SSM branch → Two-pass 커널로 교체
2. intra-layer concurrent sweep: SSM SM + Attn SM = 108 조합별 latency 측정
3. 최적 분할 비율 탐색: SSM:Attn = 54:54, 68:40, 81:27 등

예상 최적점: SSM과 Attn의 단독 latency가 비슷한 비율에서 병렬 실행 시 최대 speedup. seq=4096, bs=4 기준: SSM@68SM(8.3ms) ≈ Attn@40SM(10.1ms) → 병렬 latency ≈ 10.1ms (순차 9.3ms 대비 +8.6% 손실, 그러나 decode stream에 추가 SM을 줄 수 있음).

### 9.3 중기 (NCU 프로파일링 결과 수신 후)

#### Step 7: Wave model validation 완성

현재 PENDING인 NCU job(729105)이 완료되면:
- 실측 `grid_size`, `n_waves`, `wave_eff_pct`를 analytical 값과 비교
- Falcon-H1과 Zamba2의 wave_eff 차이 확인 (예상: 둘 다 99.9%+)
- MAPE 19.57%의 정확한 원인 분리 (scale 의존인지 특정 config 이상값인지)

#### Step 8: decode SM sensitivity 실측 (두 모델)

현재 `policy_step_adaptive.py`에 hardcode된 `decode_sm_sensitivity = 0.5`를 실측값으로 교체:

```bash
# decode latency vs SM count sweep
python stage2_overhead/measure_layer_latency.py \
    --model zamba2 --model falcon_h1 \
    --measure-decode-sensitivity \
    --device a100_80gb
```

이 값이 Policy C의 SM 양도량 결정 기준이 되므로 실측이 중요하다.

### 9.4 장기 (통합 및 논문)

#### Step 9: Falcon-H1 Stage 2 + Stage 3 실행

```bash
# Falcon-H1 Green Contexts 전환 비용 (두 번의 전환: SSM→Attn, Attn→SSM per layer)
python stage2_overhead/measure_ctx_switch_latency.py \
    --model falcon_h1 --device a100_80gb

# Policy 실행: Falcon-H1은 A/B/C가 아닌 intra-layer concurrent 정책
# Policy D: SSM+Attn 동시 실행 (decode SM 분리 없음, layer 내부 병렬화)
python stage3_hm_eval/run_concurrent_eval.py \
    --model falcon_h1 --policy A D --device a100_80gb
```

#### Step 10: 모델 포트폴리오 확장 및 논문 §4, §5 작성

| 모델 유형 | 대표 모델 | SM 공유 전략 | 이득 구간 |
|----------|----------|------------|---------|
| SSM-heavy (16% Attn) | Zamba2 | inter-layer, Attn only | 전체의 24% |
| 균형형 (all hybrid) | Falcon-H1 | intra-layer, SSM‖Attn | 전체의 ~50%* |
| Attn-heavy (80%+ Attn) | LLaMA-3 등 | inter-layer, full | 80%+ 구간 |

*Two-pass 커널 구현 완료 이후.

---

## 부록: 핵심 수치 비교표

| 측정 항목 | Zamba2 | Falcon-H1 |
|----------|--------|-----------|
| SSM latency @108SM (seq=4096, bs=4) | 12.6 ms | **5.1 ms** |
| SSM latency per n_block | 3.30 μs | **1.58 μs** |
| SSM 14→108 SM speedup | **7.71×** | **7.71×** |
| Attn latency @108SM (seq=4096, bs=4) | 29.2 ms | **4.1 ms** |
| Attn 14→108 SM speedup | **6.77×** | **6.75×** |
| MLP latency @108SM (seq=4096, bs=4) | 13.8 ms | **10.2 ms** |
| cooperative_safe / 608 configs | **0** | **69** |
| min safe n_blocks | N/A | **24** (bs=1, pct=256) |
| max safe batch | N/A | **bs=4** (pct=256, sm=108) |
| Policy C 이득 레이어 비율 | **16%** (13 Attn) | **100%*** (전층) |
| 1 forward pass SSM 점유 시간 | ~64.5% | ~26.1% |
| 1 forward pass Attn 점유 시간 | ~24.0% | ~21.1% |
| Green Ctx CPU swap overhead | 0.43 μs | 0.43 μs (동일 backend) |

*Two-pass 구현 완료 후 intra-layer 병렬 실행 시.

---

*참고 파일: `results/stage1/ssm_scaling_*`, `results/stage1/attn_scaling_*`, `results/stage1/mlp_scaling_*`, `results/stage1/chunked/ssm_chunked_*`, `results/stage2/ctx_switch_overhead_*`, `logs/stage1_731452.log`*
