# SSM Cooperative Barrier로 인한 CUDA Context 오염 분석 및 해결 방안

**작성일**: 2026-05-10  
**대상 하드웨어**: NVIDIA A100-SXM4-80GB (108 SM, HBM2e 1,000 GB/s)  
**관련 파일**: `stage1_sm_scaling/run_ssm_prefill_sweep.py`, `stage1_sm_scaling/_ssm_worker.py`, `src/models/layer_runner.py`  

---

## 1. 문제 정의

### 1.1 증상

`run_ssm_prefill_sweep.py`에서 CUDA Green Context로 SM을 제한하여 SSM 레이어의 latency를 직접 측정하려 하면 두 가지 현상이 발생한다.

```
sm=14, seq=8192, bs=32: CUDA error: an illegal memory access was encountered
sm=14, seq=8192, bs=64: CUDA error: illegal memory access          ← context 이미 오염
sm=27, seq=512,  bs=1 : CUDA error: illegal memory access          ← 단순 config도 실패
sm=27, seq=512,  bs=4 : CUDA error: illegal memory access
... (이후 전 configs 연쇄 실패)
```

또는 에러 없이 CPU 97% 상태로 수십 분 hang.

### 1.2 문제의 특이점

Attention과 MLP는 동일한 Green Context 환경에서 SM=14~108 전 레벨 정상 동작하는 반면, SSM만 실패한다. SSM은 O(L) 메모리를 사용하므로 메모리 용량 문제가 아니다.

```
레이어     Green Context 직접 측정   이유
─────────  ──────────────────────   ──────────────────────────────
SSM        불가 (CUDA 에러 / hang)   cooperative grid barrier 존재
Attention  가능 (전 SM 레벨 성공)   독립 타일, inter-block sync 없음
MLP        가능                     표준 cuBLAS GEMM, cooperative 아님
```

---

## 2. 근본 원인: Cooperative Grid Barrier

### 2.1 SSD 커널의 청크 간 state 전파 구조

`mamba_chunk_scan_combined` (Triton SSD 커널)은 시퀀스를 고정 크기 청크(chunk_size=256)로 분할하여 처리한다.

```
seq_len=4096, chunk_size=256  →  C=16 chunks

Phase 1: 각 청크 내부 SSM 병렬 스캔          (블록 독립, GEMM+scan)
         ──────────────────────────────────
         __grid_sync()   ←── 전체 블록 barrier
         ──────────────────────────────────
Phase 2: 청크 간 hidden state prefix 전파    (cooperative 필요)
         chunk_0.state → chunk_1.initial
         chunk_1.state → chunk_2.initial  ...
```

Phase 2의 prefix scan은 **모든 청크 블록이 Phase 1을 완료한 뒤 동시에 활성화된 상태**에서만 correctness가 보장된다. 이를 위해 Triton은 `grid.sync()` (CUDA Cooperative Groups의 `this_grid().sync()`)를 사용한다.

### 2.2 Cooperative Kernel의 전제 조건

CUDA cooperative kernel launch (`cudaLaunchCooperativeKernel`)는 다음을 요구한다.

> **모든 thread block이 동시에 GPU에 상주(resident)해야 한다.**

이를 위해 CUDA 런타임은 launch 시점에 `n_blocks ≤ max_active_blocks(device)` 조건을 사전 검사한다. 조건 불만족 시 launch 자체를 거부한다.

그러나 **Green Context는 이 검사 이후에 SM을 제한**하므로, CUDA 런타임의 사전 검사를 통과한 커널이 실제 실행 시점에는 조건이 깨진 상태로 구동된다.

### 2.3 Deadlock 발생 메커니즘

```
n_blocks = batch × seq_len / 4    (예: batch=32, seq=8192 → 65,536 blocks)
제한된 SM = 14개
SM당 최대 동시 블록 = ~2개 가정  →  동시 활성 블록 ≈ 28개

[상황]
Wave 1 (28 blocks):  Phase 1 완료 → grid.sync() 도달 → 나머지 블록 대기
Wave 2 (28 blocks):  SM이 Wave 1 블록으로 점유됨 → 스케줄 큐 대기
Wave 3, 4, ...:      동일

결과: Wave 1은 나머지 블록을 기다리고
      나머지 블록은 SM이 비기를 기다림 → 순환 대기 (Deadlock)
```

### 2.4 Context 오염으로 이어지는 과정

GPU에는 CPU의 mutex deadlock 해제 메커니즘이 없다. 대신 CUDA 드라이버 watchdog이 일정 시간 응답 없는 커널을 강제 종료한다.

```
1. grid.sync()에서 deadlock 발생
2. CUDA watchdog timeout → 커널 강제 종료
3. 강제 종료 시점에 출력 버퍼와 SSM state 버퍼가 partial write 상태
4. 이후 CUDA 연산이 반쯤 쓰인 포인터를 역참조
   → cudaErrorIllegalAddress (illegal memory access)
5. CUDA context 전체가 invalid 상태로 전환
6. 같은 context의 이후 모든 CUDA 연산 연쇄 실패
```

따라서 illegal memory access는 **메모리 용량 부족이 아니라 deadlock으로 인한 context 오염의 부작용**이다. SSM의 O(L) 메모리 효율성과는 무관하다.

### 2.5 에러 발생 임계점

에러는 `n_blocks`가 충분히 커지는 구간에서 처음 발생한다.

| seq | batch | n_blocks | waves@14SM | 에러 |
|-----|-------|---------|-----------|------|
| 4096 | 32 | 32,768 | 2,341 | 없음 ✓ |
| 4096 | 64 | 65,536 | 4,682 | 없음 ✓ |
| **8192** | **32** | **65,536** | **4,682** | **CUDA 에러** ✗ |
| 8192 | 64 | 131,072 | 9,363 | CUDA 에러 ✗ |

seq=4096,bs=64와 seq=8192,bs=32는 동일한 n_blocks=65,536임에도 결과가 다르다. n_blocks 외에 seq_len에 비례하는 내부 버퍼 크기(B, C state matrix의 `seq × n_groups × d_state`)도 threshold에 관여하는 것으로 추정된다.

---

## 3. 해결 방안 분류

해결 방향은 세 가지로 분류된다.

```
[A] 커널 구조 변경    — cooperative barrier 자체를 제거
[B] 측정 전략 우회    — Triton SSD를 직접 측정하지 않음
[C] 런타임 격리/복구  — 오염 발생 후 피해 범위 제한
```

---

## 4. [A] 커널 구조 변경

### 4.1 Two-pass 커널 분해

Cooperative barrier를 제거하고 동일한 SSD 알고리즘을 두 개의 독립 커널로 분리한다.

```
[현재: 1개의 cooperative kernel]
Kernel: Phase1 → grid.sync() → Phase2

[변경: 3개의 독립 kernel]
Kernel A:  각 청크의 local scan + carry state 계산     (C개 독립 블록)
  ↓ (GPU 메모리에 carry[0..C-1] 저장, 호스트 동기화 없음)
Kernel B:  carry state들에 대한 prefix scan            (C가 작으므로 1 블록으로 충분)
  ↓
Kernel C:  각 청크에 prefix 적용하여 최종 출력 계산    (C개 독립 블록)
```

C (청크 수)는 seq_len / chunk_size이므로 최대 수백 개 수준이다. Kernel B는 단일 블록으로 처리 가능하다. CUB의 `DeviceScan`이 이 방식으로 구현되어 있다.

**장점**
- 알고리즘이 SSD와 동일하므로 compute 특성 보존
- inter-block barrier 완전 제거 → Green Context SM 제한에서 안전
- 커널 론칭 오버헤드 2회 추가이지만 PCIe 이동 없음

**단점**
- Triton 커널 재작성 필요 (개발 비용 높음)
- Phase 간 GPU 메모리에 `C × d_state × d_state` carry state 임시 저장

**구현 고려사항**: Triton에서 `tl.atomic_*` 또는 별도 kernel을 통해 Kernel B를 구현할 수 있다. mamba_ssm 저장소의 `ssd_combined.py` 수정이 필요하다.

### 4.2 Persistent Kernel (블록 수 고정)

커널 론칭 시 block 수를 `n_blocks`가 아닌 `n_sm`에 고정한다.

```python
# 현재: n_blocks = batch × seq_len / 4 개 블록 론칭
# 변경: SM_count개 블록 론칭, 각 블록이 자기 담당 청크들을 순차 처리

# Triton kernel 내부 (pseudo-code)
pid = tl.program_id(0)              # 0 ~ SM_count-1
for chunk_idx in range(pid, C, SM_count):   # stride = SM_count
    state = process_chunk(chunk_idx, prev_state)
    prev_state = state
```

블록 수 = SM 수이면 wave가 1개이므로 모든 블록이 동시 활성화된다. `grid.sync()`가 불필요하다.

**장점**
- Triton SSD와 유사한 구조 유지 (cooperative 제거만)
- block 수와 SM 수를 일치시키면 wave quantization 오버헤드도 없음

**단점**
- SM 수마다 block 수가 달라지므로 `SM_count`를 커널 인자로 받아야 함
- 청크 간 state를 shared memory가 아닌 register chain 또는 global memory로 전달해야 하므로 메모리 접근 패턴 변화
- SM 수가 청크 수보다 많을 경우 일부 블록이 idle

---

## 5. [B] 측정 전략 우회

Triton SSD를 SM 제한 하에서 직접 측정하는 대신, 간접 방법으로 SM scaling 곡선을 구한다.

### 5.1 Wave Model 합성 (현재 구현, 권장)

```python
# stage1_sm_scaling/run_ssm_prefill_sweep.py: _synthesize_sm_scaling()

n_blocks = batch × seq_len // 4          # ncu 검증 공식
latency(sm_k) = latency(full_sm) × ⌈n_blocks / k⌉ / ⌈n_blocks / full_sm⌉
```

이 방식이 유효한 근거:

| 검증 항목 | 결과 |
|-----------|------|
| wave_eff_pct (ncu 측정) | 99.97%+ (전 config) |
| 동일 n_blocks 케이스 latency 비율 | 이론값과 오차 0.03% |
| 모델 독립성 (n_blocks만 의존) | seq=4096,bs=64 vs seq=8192,bs=32 latency 일치 확인 |

**장점**: 구현 완료, Triton SSD의 실제 compute 특성 정확히 반영  
**한계**: cooperative kernel의 실제 SM 포화 동작이 wave model과 다른 예외적 config에서 오차 가능 (현재까지 발견되지 않음)

### 5.2 PyTorch Scan 직접 측정 (`--force-pytorch-scan`)

```python
# stage1_sm_scaling/run_ssm_prefill_sweep.py: _measure_direct_sm()
# src/models/zamba2.py: FallbackSSMKernel(force_pytorch_scan=True)

# Python for-loop으로 청크 간 state를 순차 전달
# → cooperative barrier 없음 → Green Context에서 안전
```

**장점**: Green Context 직접 측정 가능, 구현 완료  
**중요한 한계**: Triton SSD와 근본적으로 다른 커널이다.

| 특성 | Triton SSD | PyTorch Scan |
|------|-----------|--------------|
| 연산 병목 | Compute-bound (recurrent scan) | GEMM-bound (in_proj/out_proj) |
| SM 포화점 (추정) | ~70% SM | ~35% SM |
| inter-block 통신 | `grid.sync()` | 없음 (sequential) |
| BW 활용률 | 0.1–5.9% | 더 높음 |

PyTorch scan의 SM scaling 곡선을 Decision Matrix에 입력하면 SSM에 과소 SM을 할당하게 되어 serving에서 TPOT SLO 위반이 발생한다.  
**Wave model 대비 정확도가 낮으므로 참고 측정 목적으로만 사용할 것.**

---

## 6. [C] 런타임 격리 및 복구

### 6.1 Subprocess Isolation (현재 구현)

```python
# stage1_sm_scaling/_ssm_worker.py
# 각 SM level을 별도 subprocess에서 실행
# → context 오염이 subprocess 내부에 격리됨

is_cuda_error = (
    "CUDA error" in err_str or "illegal memory" in err_str.lower() ...
)
if is_cuda_error:
    cuda_dead = True   # 해당 subprocess 포기, 다음 SM level은 새 subprocess
```

**장점**: parent process 보호, 다른 SM level 데이터 유지  
**한계**: context 오염 발생 시 해당 subprocess의 나머지 config 전체 손실

### 6.2 CUDA Context 재생성

오염된 context를 파기하고 새 context로 교체한다.

```python
import ctypes

libcuda = ctypes.CDLL("libcuda.so.1")

def reset_cuda_context(device_idx: int = 0):
    """context 오염 후 재초기화. 이후 모든 tensor/model 재할당 필요."""
    ctx = ctypes.c_void_p()
    libcuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if ctx:
        libcuda.cuCtxDestroy(ctx)
    libcuda.cuCtxCreate(ctypes.byref(ctx), 0, device_idx)
    return ctx
```

또는 더 단순하게:

```python
torch.cuda.empty_cache()
torch._C._cuda_clearCublasWorkspaces()
# 이후 model/tensor 재할당
```

`cudaDeviceReset()`은 process 내 모든 CUDA 상태를 초기화하므로 subprocess 내에서만 사용해야 한다.

**한계**: context 재생성 후 GPU에 올려둔 model weight를 전부 재전송해야 한다 (A100 기준 ~수 초). 반복 측정 루프에서 사용하기 어렵다.

### 6.3 SM 수 하한 설정으로 Deadlock 조건 차단

`n_blocks ≤ sm_count`이면 wave가 1개이므로 cooperative barrier가 안전하다.

```python
def safe_sm_minimum(batch: int, seq_len: int, tokens_per_block: int = 4) -> int:
    """deadlock 없이 직접 측정 가능한 최소 SM 수."""
    return batch * seq_len // tokens_per_block

# 예: batch=1, seq=512 → min_sm=128 → A100 108SM으로는 항상 안전
#     batch=4, seq=512 → min_sm=512 → 모든 SM level에서 위험
```

A100(108 SM)에서 `n_blocks ≤ 108`을 만족하는 config:

| batch | max safe seq_len |
|-------|----------------|
| 1 | 432 |
| 4 | 108 |
| 16 | 27 |
| 32 | 13 |

현실적인 prefill config (seq≥512, batch≥4)에서는 이 조건을 만족시키기 어렵다. **측정 범위를 크게 제한하므로 실용적이지 않다.**

---

## 7. 방안 비교

### 7.1 정확도 vs 구현 비용

```
정확도 (Triton SSD 특성 반영)
    ↑
    │  Two-pass 커널 분해     ← 정확도 최고, 구현 비용 최고
    │  Persistent kernel      ← 정확도 높음, 구현 비용 높음
    │  Wave model 합성 ★      ← 정확도 높음 (오차<0.03%), 구현 완료
    │  SM 하한 설정            ← 정확도 무관, 측정 범위 매우 협소
    │  PyTorch scan            ← 정확도 낮음 (다른 커널), 구현 완료
    ↓
낮음                           구현 비용 →                         높음
```

### 7.2 방안별 요약표

| 방안 | 분류 | 구현 여부 | 정확도 | 구현 비용 | 비고 |
|------|------|---------|--------|---------|------|
| Wave model 합성 | 측정 우회 | 완료 ★ | 높음 (오차<0.03%) | — | **현재 기본 경로** |
| PyTorch scan | 측정 우회 | 완료 | 낮음 | — | `--force-pytorch-scan`, 참고 측정 전용 |
| Subprocess isolation | 런타임 격리 | 완료 | — | — | context 오염 피해 제한 |
| SM 하한 설정 | 런타임 격리 | 미구현 | — | 낮음 | 현실적 config에서 비실용 |
| CUDA context 재생성 | 런타임 복구 | 미구현 | — | 중간 | subprocess 안에서만 유효 |
| Persistent kernel | 커널 변경 | 미구현 | 높음 | 높음 | Triton 커널 수정 필요 |
| Two-pass 분해 | 커널 변경 | 미구현 | 최고 | 최고 | mamba_ssm 의존성 수정 필요 |

---

## 8. 권고

### 8.1 현재 측정 인프라 목적 (Stage 1)

Wave model 합성이 이미 충분한 정확도(오차 <0.03%)를 제공한다. 추가 구현 없이 현재 경로를 유지한다.

PyTorch scan (`--force-pytorch-scan`)은 cooperative-free 환경에서의 동작 검증, 또는 "Triton SSD와 PyTorch scan의 SM scaling 곡선 비교" 실험에만 사용한다. Decision Matrix 입력으로는 사용하지 않는다.

### 8.2 Triton SSD 직접 측정이 필요할 경우

Two-pass 커널 분해 구현이 유일한 근본 해결책이다. 구현 우선순위 평가 시 고려할 사항:

1. `mamba_ssm` 패키지가 외부 의존성이므로 fork 또는 패치 적용 필요
2. Triton kernel 수정 시 `chunk_size`와 `d_state`에 대한 회귀 검증 필요
3. 구현 완료 후 ncu로 `grid.sync()` 제거 확인 및 wave_eff_pct 재검증 권장

### 8.3 Policy 설계 함의

SSM 레이어는 현재 SM 분할이 불가능하므로 전체 SM을 독점 사용한다. Policy C (layer-boundary reconfiguration)는 Zamba2의 Attention 레이어 13개에서만 SM 공유 이득을 얻을 수 있다. Stage 2 ctx_switch 측정 결과 transition latency가 ~8 μs이므로 레이어 경계마다 SM 구성을 변경하는 overhead는 Attention latency(수십 ms 규모) 대비 무시 가능하다.

---

## 부록: 핵심 수식

**n_blocks 공식 (ncu 검증)**
```
n_blocks = batch × seq_len // 4
```

**Wave model latency 합성**
```
latency(sm_k) = latency(sm_full) × ⌈n_blocks / k⌉ / ⌈n_blocks / sm_full⌉
```

**Deadlock 발생 조건**
```
n_blocks > sm_count × max_blocks_per_sm   (cooperative kernel 제약 위반)
```

**Two-pass carry state 크기**
```
carry_state_bytes = C × n_groups × d_state × d_state × bytes_per_elem
                  = (seq_len / chunk_size) × 2 × 64 × 64 × 2
                  ≈ seq_len × 32 bytes   (Zamba2-7B 기준)
예: seq_len=4096 → ~128 KB  (GPU 메모리 부담 없음)
```
