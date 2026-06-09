# SSM Two-pass Kernel 분해 구현 가이드라인

**작성일**: 2026-05-10  
**대상 하드웨어**: NVIDIA A100-SXM4-80GB (108 SM, HBM2e 1,000 GB/s)  
**대상 모델**: Zamba2-7B-Instruct, Falcon-H1-7B-Instruct  
**관련 보고서**: `reports/ssm_cooperative_barrier_context_corruption.md`  

---

## 1. 목적 및 범위

본 문서는 `mamba_chunk_scan_combined` (Triton SSD 커널)을 cooperative barrier 없이 세 개의 독립 커널로 분해하는 구현 가이드라인이다.

두 가지 사용 목적이 있다.

| 사용 목적 | 달성 목표 |
|-----------|-----------|
| **Stage 1 직접 측정** | Wave model 합성에 의존하지 않고 Triton SSD 커널 자체를 SM 제한 하에서 직접 latency 측정 |
| **Serving SM 분할** | Hybrid LLM prefill 실행 중 SSM 레이어에서도 decode와 SM을 공유할 수 있도록 cooperative barrier 제거 |

현재 구현된 Wave model 합성 경로(`_synthesize_sm_scaling`)의 정확도가 이미 오차 < 0.03%이므로 Stage 1 측정 목적만이라면 이 구현은 선택적이다. Serving SM 분할이 필요하면 필수다.

---

## 2. 배경 요약

`reports/ssm_cooperative_barrier_context_corruption.md`의 핵심만 발췌한다.

```
mamba_chunk_scan_combined 내부 구조

Phase 1 : 청크별 local SSM scan          (블록 독립)
            ↓
          grid.sync()    ← 전체 블록이 동시에 도달해야 완료
            ↓
Phase 2 : 청크 간 hidden state prefix 전파 (cooperative 필요)
```

Green Context로 SM을 제한하면 일부 블록이 스케줄 큐에 대기하면서 `grid.sync()`에서 순환 대기가 발생한다. CUDA watchdog이 커널을 강제 종료하면 partial write 상태로 context가 오염되고 이후 모든 CUDA 연산이 연쇄 실패한다.

**Two-pass 분해는 `grid.sync()`가 필요한 Phase 2를 별도 커널로 분리하고 블록 수를 줄여 cooperative 제약을 제거한다.**

---

## 3. Mamba-2 SSD 수식과 청크 분해 원리

### 3.1 SSM 점화식

Mamba-2 레이어의 토큰 단위 SSM 점화식:

```
h[t] = Ā[t] ⊗ h[t-1]  +  B[t] ⊗ x[t]          ... (1)
y[t] =  C[t]ᵀ ⊗ h[t]  +  D * x[t]               ... (2)

Ā[t] = exp(−dt[t] · A)   (헤드별 선택적 감쇠, scalar per head)
B[t] ∈ R^(n_groups × d_state)
C[t] ∈ R^(n_groups × d_state)
h[t] ∈ R^(n_heads × d_state)   (SSM 숨겨진 상태)
```

### 3.2 청크 내 병렬 스캔

seq_len을 크기 `K = chunk_size`의 C개 청크로 분할한다.

```
C = ceil(seq_len / chunk_size)

청크 c 내 전체 토큰에 대한 cumulative 변환은 행렬 결합법칙으로 표현:
  (M_c, b_c) = (Ā[cK+K-1], ...) ⊗ ... ⊗ (Ā[cK], B[cK]·x[cK])

  여기서 M_c = 청크 c의 전체 state transition matrix
         b_c = 청크 c의 input contribution vector
```

### 3.3 청크 간 연결 (Inter-chunk State Propagation)

청크 c의 초기 상태는 청크 c-1의 최종 상태다.

```
h_initial[c] = M_{c-1} ⊗ h_initial[c-1] + b_{c-1}

prefix state:
  P[0] = (I, 0)                             (첫 청크는 초기 상태 0)
  P[c] = (M_{c-1}, b_{c-1}) ∘ P[c-1]       (결합법칙 적용)

∘ 는 affine 결합: (M_a, b_a) ∘ (M_b, b_b) = (M_a @ M_b, M_a @ b_b + b_a)
```

**이 P[c] 계산이 `grid.sync()`를 요구하는 부분이다.** P[c]는 P[c-1]에 의존하므로 직렬 연산이지만, C는 최대 수십 개 수준이어서 계산량이 극히 작다.

---

## 4. Three-Kernel 설계

### 4.1 데이터 흐름

```
입력: x, dt, A, B, C, D, dt_bias
  │
  ▼ ─────────────────────────────────── Kernel A ──
  │  local chunk scan                (n_blocks = batch × C)
  │  inter-block sync 없음
  │  Green Context 안전
  ├──→ y_local  [batch, seq_len, n_heads, head_dim]
  └──→ carry    [batch, C, n_groups, d_state+1, d_state]
                                      ↑ (M_c, b_c) 결합
  │
  ▼ ─────────────────────────────────── Kernel B ──
  │  carry state prefix scan         (n_blocks = batch)
  │  청크 수 C에 대한 순차 연산
  │  블록 수 최소 → cooperative 제약 無
  └──→ prefix   [batch, C, n_groups, d_state+1, d_state]
                                      ↑ P[c]
  │
  ▼ ─────────────────────────────────── Kernel C ──
  │  apply prefix + compute output   (n_blocks = batch × C)
  │  inter-block sync 없음
  │  Green Context 안전
  └──→ y_final  [batch, seq_len, n_heads, head_dim]
```

세 커널 모두 regular kernel launch (`cudaLaunchKernel`)로 실행된다. `cudaLaunchCooperativeKernel`이 필요 없다.

### 4.2 Kernel A: Local Chunk Scan

**역할**: 각 청크 내부의 SSM 병렬 스캔. 청크 간 의존성 없음.  
**블록 수**: `batch × C`  (예: batch=32, seq=4096, chunk=256 → 32 × 16 = 512 블록)  
**출력**: `y_local` (청크 간 state 전파 전 임시 출력) + `carry` (청크 경계 전환 행렬)

```python
# Triton kernel (pseudo-code)
@triton.jit
def kernel_a_local_chunk_scan(
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, dt_bias_ptr,
    y_local_ptr, carry_M_ptr, carry_b_ptr,
    # strides: batch, chunk, token, head, dim ...
    batch_size, chunk_count, chunk_size,
    n_heads, head_dim, d_state, n_groups,
):
    bid = tl.program_id(0)          # 0 ~ batch*C-1
    b   = bid // chunk_count
    c   = bid % chunk_count

    # 이 청크의 초기 state를 0으로 초기화 (carry 적용은 Kernel C에서)
    h = tl.zeros([n_heads, d_state], dtype=tl.float32)

    # chunk_size 토큰에 대해 SSM 점화식 적용
    for t in range(chunk_size):
        global_t = c * chunk_size + t
        if global_t >= seq_len:
            break

        x_t   = load(x_ptr[b, global_t, :, :])          # (n_heads, head_dim)
        dt_t  = load(dt_ptr[b, global_t, :])             # (n_heads,)
        B_t   = load(B_ptr[b, global_t, :, :])           # (n_groups, d_state)
        C_t   = load(C_ptr[b, global_t, :, :])           # (n_groups, d_state)

        A_bar = tl.exp(-tl.softplus(dt_t + dt_bias) * tl.abs(A))   # (n_heads,)

        # h[head] = A_bar[head] * h[head] + B_t[g(head)] * x_t[head]
        # (n_heads, d_state) 업데이트
        h = A_bar[:, None] * h + B_t[group_of_head, :] * x_t[:, None]

        # y_t = sum_d_state(C_t * h) + D * x_t (gate 제외)
        y_t = tl.sum(C_t[group_of_head, :] * h, axis=1) + D * x_t
        store(y_local_ptr[b, global_t, :, :], y_t)

    # carry = (M_c, b_c): 이 청크의 state transition matrix와 bias
    # M_c = cumulative A_bar product over chunk (diagonal, n_heads)
    # b_c = accumulated input contribution (n_heads, d_state)
    store(carry_M_ptr[b, c, :, :], M_c)
    store(carry_b_ptr[b, c, :, :], h)           # h 자체가 b_c
```

**Carry state 크기** (모델별):

| 모델 | n_heads | d_state | C (seq=4096) | carry 크기 (batch=32) |
|------|---------|---------|--------------|----------------------|
| Zamba2-7B | 112 | 64 | 16 | 32 × 16 × 112 × 64 × 2B ≈ **7.3 MB** |
| Falcon-H1-7B | 24 | 256 | 16 | 32 × 16 × 24 × 256 × 2B ≈ **6.3 MB** |

두 모델 모두 GPU 메모리 부담 없는 수준이다.

### 4.3 Kernel B: Inter-Chunk Prefix Scan

**역할**: carry[c]들에 대한 affine 결합 prefix scan. P[c] = P[c-1] ∘ carry[c].  
**블록 수**: `batch` (최대 64)  
**직렬성**: c에 대한 순차 의존성 있음. C가 최대 128 (seq=32K, chunk=256)으로 작아 GPU에서도 빠름.

```python
@triton.jit
def kernel_b_prefix_scan(
    carry_M_ptr, carry_b_ptr,      # (batch, C, n_heads, d_state) — Kernel A 출력
    prefix_M_ptr, prefix_b_ptr,    # (batch, C, n_heads, d_state) — Kernel B 출력
    batch_size, chunk_count, n_heads, d_state,
):
    b = tl.program_id(0)           # 0 ~ batch-1

    # P[0] = (I, 0): 첫 청크는 이전 state 없음
    prev_M = tl.eye(d_state)       # (d_state, d_state)
    prev_b = tl.zeros([n_heads, d_state])

    store(prefix_M_ptr[b, 0], prev_M)
    store(prefix_b_ptr[b, 0], prev_b)

    for c in range(1, chunk_count):
        cur_M = load(carry_M_ptr[b, c-1, :, :])   # (n_heads, d_state) — diagonal
        cur_b = load(carry_b_ptr[b, c-1, :, :])   # (n_heads, d_state)

        # affine 결합: P[c] = (cur_M, cur_b) ∘ (prev_M, prev_b)
        #   new_M = cur_M * prev_M  (element-wise: diagonal 행렬이므로)
        #   new_b = cur_M * prev_b + cur_b
        new_M = cur_M * prev_M
        new_b = cur_M * prev_b + cur_b

        store(prefix_M_ptr[b, c], new_M)
        store(prefix_b_ptr[b, c], new_b)

        prev_M = new_M
        prev_b = new_b
```

> **주의**: Mamba-2에서 A는 diagonal (헤드별 scalar)이므로 M은 full matrix가 아닌 (n_heads, d_state) 텐서다. 이 경우 matrix multiply가 element-wise multiply로 단순화된다. 일반 SSD에서 A가 full matrix이면 `tl.dot`을 사용해야 한다.

**float32 누적 권장**: bfloat16으로 C=128 단계 누적 시 상대 오차가 ~1%까지 커질 수 있다.

### 4.4 Kernel C: Apply Prefix and Finalize

**역할**: Kernel A의 임시 출력 `y_local`에 청크별 prefix state를 적용하여 최종 출력 계산.  
**블록 수**: `batch × C`  (Kernel A와 동일)  
**inter-block sync 없음**

```python
@triton.jit
def kernel_c_finalize(
    y_local_ptr,                    # (batch, seq_len, n_heads, head_dim) — Kernel A 출력
    prefix_M_ptr, prefix_b_ptr,    # (batch, C, n_heads, d_state) — Kernel B 출력
    C_ptr,                         # (batch, seq_len, n_groups, d_state) — SSM C 행렬
    D_ptr,                         # (n_heads,)
    x_ptr,                         # (batch, seq_len, n_heads, head_dim)
    y_final_ptr,                   # (batch, seq_len, n_heads, head_dim) — 최종 출력
    batch_size, chunk_count, chunk_size, n_heads, head_dim, d_state, n_groups,
):
    bid = tl.program_id(0)
    b   = bid // chunk_count
    c   = bid % chunk_count

    # 이 청크에 적용할 prefix state 로드
    prefix_M = load(prefix_M_ptr[b, c, :, :])   # (n_heads, d_state)
    prefix_b = load(prefix_b_ptr[b, c, :, :])   # (n_heads, d_state)

    for t in range(chunk_size):
        global_t = c * chunk_size + t
        if global_t >= seq_len:
            break

        # y_local[t]는 초기 state = 0으로 계산된 임시값
        # 실제 초기 state = prefix_b (이전 청크들의 누적 state)
        # 보정: y_final[t] = y_local[t] + C[t]ᵀ · (A_bar_cumulative_to_t · prefix_b)

        y_local_t = load(y_local_ptr[b, global_t, :, :])
        C_t       = load(C_ptr[b, global_t, :, :])         # (n_groups, d_state)

        # prefix_b를 이 토큰까지 decay한 후 C와 내적
        A_bar_cum = compute_cumulative_decay(dt, A, c, t)   # (n_heads,)
        prefix_contribution = tl.sum(
            C_t[group_of_head, :] * A_bar_cum[:, None] * prefix_b, axis=1
        )  # (n_heads,)

        y_final_t = y_local_t + prefix_contribution[:, None]
        store(y_final_ptr[b, global_t, :, :], y_final_t)
```

---

## 5. 모델별 파라미터 및 규모

### 5.1 Zamba2-7B-Instruct

| 파라미터 | 값 | 비고 |
|---------|---|------|
| `d_model` | 3584 | |
| `n_heads` | 112 | |
| `head_dim` | 64 | |
| `d_state` | 64 | carry 행렬 dim |
| `chunk_size` | 256 | C = seq_len / 256 |
| `n_groups` | 2 | B, C 행렬 그룹 수 |
| 관련 파일 | `src/models/zamba2.py:195-216` | `FallbackSSMKernel.__init__` |

**seq_len별 Kernel 스케일**:

| seq_len | C (청크 수) | Kernel A 블록 (batch=32) | Kernel B 블록 | carry 메모리 |
|---------|------------|--------------------------|--------------|-------------|
| 512 | 2 | 64 | 32 | 0.9 MB |
| 4096 | 16 | 512 | 32 | 7.3 MB |
| 16384 | 64 | 2,048 | 32 | 29 MB |
| 32768 | 128 | 4,096 | 32 | 58 MB |

### 5.2 Falcon-H1-7B-Instruct

| 파라미터 | 값 | 비고 |
|---------|---|------|
| `d_model` | 3072 | |
| `n_heads` | 24 | |
| `head_dim` | 128 | `d_model / n_heads = 128` |
| `d_state` | 256 | Zamba2 대비 4배 큼 |
| `chunk_size` | 256 | |
| `n_groups` | 1 | B, C 행렬 그룹 없음 |
| 관련 파일 | `src/models/falcon_h1.py:36-47` | |

**Falcon-H1 아키텍처 특이사항**: SSM과 Attention이 **같은 레이어 내에서 병렬**로 실행된다 (Zamba2처럼 교대 배치가 아님). `src/models/falcon_h1.py:11-14`에 명시:

```
SSM branch ──┐
             ├─→ add ─→ MLP
Attn branch ─┘
```

이는 Serving SM 분할 설계에서 중요하다. Falcon-H1은 SSM과 Attention을 같은 레이어에서 SM을 나눌 수 없다 (두 브랜치가 동시에 실행됨).

---

## 6. 기존 코드베이스 통합

### 6.1 수정 파일 목록

```
mamba_ssm/ops/triton/ssd_combined.py       ← 외부 패키지, fork/patch 필요
  + _kernel_a_local_chunk_scan()
  + _kernel_b_prefix_scan()
  + _kernel_c_finalize()
  + mamba_chunk_scan_combined_two_pass()    ← 기존 함수와 동일 인터페이스

src/models/zamba2.py
  FallbackSSMKernel
    + force_two_pass: bool = False 파라미터 추가
    forward() 분기 추가

src/models/falcon_h1.py
  FallbackSSMBranch (또는 해당 SSM 래퍼)
    + force_two_pass 파라미터 추가 (동일 패턴)

src/models/layer_runner.py
  run_ssm_layer()
    + force_two_pass: bool = False 파라미터 추가
    _ssm_cache 키에 force_two_pass 포함

stage1_sm_scaling/run_ssm_prefill_sweep.py
  + --force-two-pass CLI 인자 추가
  run_sweep() → force_two_pass 파라미터 전달
  출력 파일명: ssm_scaling_{model}_{tag}_twopass.csv
```

### 6.2 zamba2.py 통합 포인트

현재 `FallbackSSMKernel.forward()` (`src/models/zamba2.py:225-262`):

```python
# [현재] 우선순위: force_pytorch_scan → triton → pytorch fallback
def forward(self, hidden_states):
    if self.force_pytorch_scan:
        return self._pytorch_fallback(hidden_states)
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        y = mamba_chunk_scan_combined(x, dt, A, B, C, ...)
    except Exception:
        return self._pytorch_fallback(hidden_states)
```

```python
# [수정 후] 우선순위: force_pytorch_scan → force_two_pass → triton → two_pass → pytorch
def forward(self, hidden_states):
    if self.force_pytorch_scan:
        return self._pytorch_fallback(hidden_states)

    if self.force_two_pass:
        return self._two_pass_forward(hidden_states)

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        y = mamba_chunk_scan_combined(x, dt, A, B, C, ...)
    except Exception:
        try:
            return self._two_pass_forward(hidden_states)   # cooperative 실패 시 자동 fallback
        except Exception:
            return self._pytorch_fallback(hidden_states)

def _two_pass_forward(self, hidden_states):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined_two_pass
    # ... 기존 forward와 동일한 전처리 ...
    y = mamba_chunk_scan_combined_two_pass(
        x, dt, A, B, C,
        chunk_size=self.chunk_size,
        D=self.D, dt_bias=self.dt_bias, dt_softplus=True,
    )
    # ... gate, out_proj ...
```

### 6.3 layer_runner.py 통합 포인트

`run_ssm_layer()` (`src/models/layer_runner.py:129-219`):

```python
# [수정] ssm_cache 키에 force_two_pass 추가
ssm_key = (model_name, batch_size, seq_len, use_fallback_kernel,
           force_pytorch_scan, force_two_pass)          # ← 추가

if ssm_key not in self._ssm_cache:
    if use_fallback_kernel or force_pytorch_scan or force_two_pass:
        layer = self._build_fallback_ssm(
            model_name,
            force_pytorch_scan=force_pytorch_scan,
            force_two_pass=force_two_pass,              # ← 추가
        )
```

### 6.4 mamba_ssm 패치 전략

외부 패키지이므로 세 가지 방법 중 하나를 선택한다.

| 방법 | 적합한 경우 | 방법 |
|------|-----------|------|
| **git submodule + patch** | 재현성 중요, CI 환경 | `git submodule add`, `git apply two_pass.patch` |
| **pip editable install** | 개발 단계 | `pip install -e ./mamba_ssm_fork` |
| **런타임 monkey-patch** | 패키지 수정 최소화 | 모듈 import 후 함수 교체 |

런타임 monkey-patch 예시:

```python
# src/models/zamba2.py 상단에서 한 번만 실행
def _apply_two_pass_patch():
    import mamba_ssm.ops.triton.ssd_combined as _ssd
    from ._two_pass_kernels import mamba_chunk_scan_combined_two_pass
    _ssd.mamba_chunk_scan_combined_two_pass = mamba_chunk_scan_combined_two_pass
```

---

## 7. 사용 목적 A: Stage 1 SM Sweep 직접 측정

### 7.1 현재 방식 vs. Two-pass 방식

```
[현재: Wave model 합성]
1. 전체 SM에서 측정 (Green Context 없음)
2. latency(sm_k) = latency(full) × ⌈n_blocks/k⌉ / ⌈n_blocks/full⌉
→ 오차 < 0.03%, 이미 충분한 정확도

[Two-pass: 직접 측정]
1. 각 SM step에서 Two-pass 커널 직접 실행 (Green Context 적용)
2. 실제 latency를 CUDA event로 측정
→ Wave model과의 편차를 직접 검증 가능
→ wave_eff < 99%인 예외 config 발견 가능
```

### 7.2 출력 파일 및 Wave model과의 비교

Two-pass 측정 결과는 `ssm_scaling_{model}_{tag}_twopass.csv`로 저장한다. Wave model 합성 결과 (`ssm_scaling_{model}_{tag}.csv`)와 나란히 비교할 수 있다.

`analytical` 컬럼: Wave model 합성 결과는 `True`, Two-pass 직접 측정은 `False`.

### 7.3 CLI 인자 설계

`run_ssm_prefill_sweep.py`에 추가할 인자:

```python
parser.add_argument(
    "--force-two-pass", action="store_true",
    help=(
        "Use two-pass cooperative-barrier-free kernel decomposition. "
        "Enables direct Green Context measurement of Triton SSD algorithm "
        "(no deadlock). Output filename includes '_twopass' suffix. "
        "Requires mamba_ssm with two-pass patch applied."
    ),
)
```

---

## 8. 사용 목적 B: Serving 환경 SM 분할 설계

### 8.1 현재 Policy C의 한계

`stage3_hm_eval/policy_layer_wise.py:28-30`에서 SM 비율이 하드코딩되어 있다.

```python
SSM_PREFILL_RATIO  = 0.70   # hardcoded
ATTN_PREFILL_RATIO = 0.40
MLP_PREFILL_RATIO  = 0.50
```

Cooperative barrier가 있으면 SSM 레이어에서 SM 분할이 불가능하므로 이 비율은 현재 이론값이다. Two-pass가 구현되면 Stage 1 측정 데이터에서 **SSM saturation SM count**를 실제로 도출하여 이 비율을 자동화할 수 있다.

### 8.2 SSM Saturation 도출 → Policy C 자동화

```python
# stage2_overhead/compute_decision_matrix.py: load_stage1_saturation()
# SSM SM saturation = "latency가 더 이상 linear하게 줄지 않는 SM 수"

def derive_ssm_prefill_ratio(stage1_csv: Path, total_sm: int) -> float:
    """Two-pass 직접 측정 결과에서 SSM prefill에 최적인 SM 비율 도출."""
    df = pd.read_csv(stage1_csv)
    df = df[df["analytical"] == False]   # two-pass 직접 측정만 사용

    # marginal gain < 3%인 첫 SM count를 saturation으로 정의
    grouped = df.groupby("sm_count")["latency_ms"].mean().sort_index()
    for i in range(1, len(grouped)):
        prev_lat = grouped.iloc[i-1]
        cur_lat  = grouped.iloc[i]
        marginal_gain = (prev_lat - cur_lat) / prev_lat
        if marginal_gain < 0.03:
            saturation_sm = grouped.index[i]
            return saturation_sm / total_sm

    return 1.0   # 포화 없음 → 전체 SM 사용
```

도출된 비율을 `policy_layer_wise.py`에 반영:

```python
# [수정 후] 하드코딩 제거, decision matrix에서 로드
class PolicyLayerWise:
    def __init__(self, smctrl, decision_matrix_path):
        matrix = json.load(open(decision_matrix_path))
        self.ssm_prefill_ratio  = matrix.get("ssm_saturation_ratio",  0.70)
        self.attn_prefill_ratio = matrix.get("attn_saturation_ratio",  0.40)
        self.mlp_prefill_ratio  = matrix.get("mlp_saturation_ratio",   0.50)
```

### 8.3 Falcon-H1 아키텍처 주의사항

Falcon-H1은 SSM과 Attention이 같은 레이어에서 병렬 실행된다.

```
Zamba2:  [SSM layer] → [SSM layer] → ... → [Attn layer] → [SSM layer] → ...
         ↑ SM 분할 독립적 설정 가능

Falcon-H1: [SSM + Attn + MLP] → [SSM + Attn + MLP] → ...
           ↑ SSM과 Attn이 동시 실행 → 레이어 내 SM 분할 불가
```

Falcon-H1에서 Policy C는 레이어 단위 SM 분할이 아닌, **배치/요청 단위 SM 분할**로 접근해야 한다. 두 가지 옵션이 있다.

**Option 1: 레이어를 직렬화** — SSM branch와 Attn branch를 순차 실행하되 각각 다른 SM count 적용. Falcon-H1의 설계 의도 (병렬 실행 성능)를 희생한다.

**Option 2: 고정 SM 비율 사용** — SSM과 Attn을 함께 실행하므로 Policy A (고정 비율)만 적용 가능. Two-pass는 측정 목적으로만 사용한다.

---

## 9. 검증 전략

### 9.1 단계별 검증 순서

```
Step 1: 수식 검증 (CPU, 소규모)
Step 2: 커널별 출력 검증 (GPU, 원본 커널과 비교)
Step 3: Green Context 호환성 검증 (SM 제한 하 실행)
Step 4: 성능 오버헤드 측정
Step 5: Stage 1 전체 sweep 실행
```

### 9.2 수식 검증 (Step 1)

Two-pass 구현 전에 Python으로 알고리즘을 검증한다.

```python
def two_pass_reference(x, dt, A, B, C, chunk_size):
    """Two-pass 알고리즘의 Python reference 구현."""
    batch, seq_len, n_heads, head_dim = x.shape
    n_chunks = seq_len // chunk_size

    # Phase 1: local scan
    y_local = torch.zeros_like(x)
    carries_h = []
    for c in range(n_chunks):
        h = torch.zeros(batch, n_heads, d_state)
        for t in range(chunk_size):
            gt = c * chunk_size + t
            A_bar = torch.exp(-F.softplus(dt[:, gt] + dt_bias) * A.abs())
            h = A_bar.unsqueeze(-1) * h + B[:, gt] @ x[:, gt]
            y_local[:, gt] = (C[:, gt] * h).sum(-1)
        carries_h.append(h.clone())

    # Phase 2: prefix scan
    prefix_h = [torch.zeros_like(carries_h[0])]
    for c in range(1, n_chunks):
        prefix_h.append(prefix_h[-1] * A_cumulative[c-1] + carries_h[c-1])

    # Phase 3: finalize
    y_final = y_local.clone()
    for c in range(n_chunks):
        for t in range(chunk_size):
            gt = c * chunk_size + t
            A_bar_cum = A_cumulative_to_token(gt, c * chunk_size)
            correction = (C[:, gt] * A_bar_cum * prefix_h[c]).sum(-1)
            y_final[:, gt] += correction

    return y_final

# 검증: reference vs. original mamba_chunk_scan_combined
assert (two_pass_reference(...) - mamba_chunk_scan_combined(...)).abs().max() < 1e-2
```

### 9.3 GPU 커널 수치 검증 (Step 2)

```python
def test_two_pass_numerical(batch=4, seq_len=4096):
    """Two-pass Triton 커널 vs. 원본 커널 비교. 전체 SM에서 실행."""
    # Zamba2 파라미터
    x   = torch.randn(batch, seq_len, 112, 64, dtype=torch.bfloat16).cuda()
    dt  = torch.rand (batch, seq_len, 112, dtype=torch.bfloat16).cuda() * 0.5
    A   = -torch.rand(112, dtype=torch.bfloat16).cuda()
    B   = torch.randn(batch, seq_len, 2, 64, dtype=torch.bfloat16).cuda()
    C   = torch.randn(batch, seq_len, 2, 64, dtype=torch.bfloat16).cuda()
    D   = torch.ones (112, dtype=torch.bfloat16).cuda()
    dt_bias = torch.zeros(112, dtype=torch.bfloat16).cuda()

    y_orig     = mamba_chunk_scan_combined(x, dt, A, B, C, 256, D, dt_bias, True)
    y_two_pass = mamba_chunk_scan_combined_two_pass(x, dt, A, B, C, 256, D, dt_bias, True)

    max_diff  = (y_orig - y_two_pass).abs().max().item()
    mean_diff = (y_orig - y_two_pass).abs().mean().item()

    # bfloat16 정밀도 고려: 허용 오차 1e-2
    assert max_diff < 1e-2, f"max diff: {max_diff:.4f}"
    print(f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  → PASS")
```

### 9.4 Green Context 호환성 검증 (Step 3)

```python
def test_green_context_compatibility():
    """SM 제한 하에서 deadlock 없이 완료되는지 확인."""
    from src.smctrl import SMController

    smctrl  = SMController(total_sm_count=108)
    sm_targets = [14, 27, 54, 108]          # 기존에 14 SM에서 deadlock 발생

    for sm in sm_targets:
        smctrl.set_sm_count(sm)
        stream = smctrl.get_stream()

        try:
            with torch.cuda.stream(stream):
                y = mamba_chunk_scan_combined_two_pass(
                    x, dt, A, B, C, 256, D, dt_bias, True
                )
            torch.cuda.synchronize()
            print(f"sm={sm:3d}  OK  y.shape={y.shape}")
        except Exception as e:
            print(f"sm={sm:3d}  FAIL  {e}")
        finally:
            smctrl.reset()
```

**합격 기준**: sm=14에서 `CUDA error: illegal memory access` 없이 완료.

### 9.5 기대 수치 검증 요약

| 검증 항목 | 기준값 | 측정 방법 |
|-----------|--------|-----------|
| 수치 오차 (bfloat16) | max_diff < 1e-2 | 원본 커널 대비 element-wise |
| Green Context @ 14 SM | CUDA 에러 없음 | seq=8192, batch=32 — 기존 deadlock 조건 |
| Kernel 론칭 오버헤드 | < 15% | full SM 중앙값 latency 비율 |
| carry state 메모리 | < 100 MB | seq=32K, batch=64 최대 규모 |
| Stage 1 sweep 완료율 | 8/8 SM 레벨 성공 | 전 (seq_len, batch) 조합 |

---

## 10. 예상 성능 오버헤드

### 10.1 커널 론칭 오버헤드

세 번의 커널 론칭 = 약 3 × 5–10 μs = 15–30 μs 추가.  
A100에서 SSM 레이어 latency는 seq=4096, batch=32 기준 ~2,900 ms이므로 **< 0.001%** 수준으로 무시 가능.

### 10.2 carry state 추가 메모리 접근

Kernel A에서 carry를 GPU 메모리에 쓰고 Kernel B에서 읽는 추가 메모리 이동:

```
carry 크기 (Zamba2, seq=4096, batch=32) = 7.3 MB
추가 BW 소비 = 7.3 MB × 2 (write + read) = 14.6 MB
A100 HBM 대역폭 = 1,000 GB/s
추가 시간 = 14.6 MB / 1,000 GB/s ≈ 0.015 ms
```

전체 SSM latency 대비 **< 0.001%** — 무시 가능.

### 10.3 Kernel B 계산량

Zamba2, C=16 청크, n_heads=112, d_state=64:

```
Kernel B 연산 = C × n_heads × d_state × 2 ops
             = 16 × 112 × 64 × 2 = 229,376 FLOPs
A100 FP32 throughput = 19.5 TFLOPS
예상 시간 = 229,376 / 19.5e12 ≈ 12 ns
```

완전히 무시 가능한 수준이다.

### 10.4 총 오버헤드 예상

| 항목 | 추가 시간 | 비율 (seq=4096, bs=32) |
|------|----------|----------------------|
| 커널 론칭 3회 | ~25 μs | < 0.001% |
| carry write/read | ~0.015 ms | ~0.001% |
| Kernel B 연산 | ~0.001 ms | 무시 가능 |
| **합계** | **~0.04 ms** | **< 0.002%** |

원본 mamba_chunk_scan_combined 대비 측정 가능한 오버헤드는 없을 것으로 예상된다. 실제 측정 시 캐시 효과, JIT 컴파일 시간, 스트림 동기화 등에 의해 2–5% 수준의 변동이 있을 수 있다.

---

## 11. 구현 체크리스트

### Phase 1: 수식 및 알고리즘 확정

- [ ] mamba_ssm `ssd_combined.py` 소스 분석 — carry state의 정확한 수학적 정의 확인
- [ ] Python reference 구현 및 원본 커널 대비 수치 검증 (CPU 소규모)
- [ ] Falcon-H1 SSM 수식이 Zamba2와 동일한지 확인 (`d_state=256` 차이 외)
- [ ] Kernel C의 prefix 적용 방식 결정 (직접 보정 vs. 재계산)

### Phase 2: Triton 커널 구현

- [ ] Kernel A: local chunk scan — `_kernel_a_local_chunk_scan()`
- [ ] Kernel B: prefix scan — `_kernel_b_prefix_scan()`
- [ ] Kernel C: finalize — `_kernel_c_finalize()`
- [ ] Wrapper: `mamba_chunk_scan_combined_two_pass()` — 기존 함수와 동일 인터페이스
- [ ] 패딩 처리 (seq_len % chunk_size ≠ 0 케이스)
- [ ] Kernel B의 float32 누적 적용

### Phase 3: 코드베이스 통합

- [ ] `src/models/zamba2.py` — `force_two_pass` 파라미터 추가, `_two_pass_forward()` 구현
- [ ] `src/models/falcon_h1.py` — 동일 패턴 적용
- [ ] `src/models/layer_runner.py` — `run_ssm_layer()` 파라미터 추가, cache 키 업데이트
- [ ] `stage1_sm_scaling/run_ssm_prefill_sweep.py` — `--force-two-pass` CLI 인자, 파일명 suffix `_twopass`
- [ ] `stage1_sm_scaling/_ssm_worker.py` — `--force-two-pass` 인자 추가

### Phase 4: 검증

- [ ] 수치 검증: max_diff < 1e-2 (Zamba2, Falcon-H1 각각)
- [ ] Green Context @ 14 SM 실행: CUDA 에러 없음
- [ ] Green Context @ 27, 54, 108 SM 실행: 모두 성공
- [ ] 전체 SM latency 오버헤드: < 15%
- [ ] Stage 1 SSM sweep 전체 완료 (8 SM 레벨 × 전 config)

### Phase 5: Stage 2/3 통합 (Serving 목적인 경우)

- [ ] `compute_decision_matrix.py` — two-pass 측정 CSV에서 SSM saturation 비율 도출
- [ ] `policy_layer_wise.py` — `SSM_PREFILL_RATIO` 하드코딩 → decision matrix 자동 로드
- [ ] Stage 3 concurrent eval 실행 — Policy C SSM SM 할당이 올바른지 확인
- [ ] Falcon-H1 아키텍처 제약 반영 (병렬 SSM+Attn 레이어 → Option 1/2 선택)

---

## 12. 의존성 및 참고 자료

### 프로젝트 내부

| 파일 | 관련 내용 |
|------|-----------|
| `reports/ssm_cooperative_barrier_context_corruption.md` | 문제 분석, deadlock 메커니즘 |
| `reports/ssm_sm_partitioning_analysis.md` | Wave model 검증 데이터, PyTorch scan 비교 |
| `src/models/zamba2.py:180-301` | FallbackSSMKernel, _pytorch_fallback 패턴 |
| `src/models/falcon_h1.py:1-47` | Falcon-H1 SSM 파라미터 |
| `src/models/layer_runner.py:129-219` | run_ssm_layer, SM 제어 패턴 |
| `stage2_overhead/compute_decision_matrix.py:47-100` | SSM saturation 로드 로직 |
| `stage3_hm_eval/policy_layer_wise.py:27-30` | 하드코딩된 SM 비율 위치 |

### 외부 참고

| 자료 | 목적 |
|------|------|
| `mamba_ssm/ops/triton/ssd_combined.py` | carry state 수식 및 청크 스캔 구현 |
| Mamba-2 논문 (Dao & Gu, 2024) | SSD 알고리즘 수식 — affine 결합 연산자 정의 |
| Triton 공식 문서 | `tl.dot`, shared memory, program_id 사용법 |
| CUB `DeviceScan` | 멀티패스 prefix scan 설계 참고 |
| CUDA Cooperative Groups 문서 | `cudaLaunchCooperativeKernel` 제약 조건 이해 |
