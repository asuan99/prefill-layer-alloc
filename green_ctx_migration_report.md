# libsmctrl → Green Contexts + Stream 분리 마이그레이션 보고서

**작성일**: 2026-05-07  
**대상**: prefill-layer-alloc 프로젝트  
**배경**: A100 서버 CUDA driver 13.0 환경에서 libsmctrl 미지원 → libsmctrl 의존성 완전 제거 후 CUDA Green Contexts + stream 분리 방식으로 교체  

---

## 1. 배경 및 제약

| 항목 | 현황 |
|------|------|
| 서버 GPU | NVIDIA A100 (108 SM, HBM2e) |
| CUDA driver | 13.0 |
| libsmctrl 지원 여부 | **미지원** (driver internal struct 변경으로 ioctl 실패) |
| driver 변경 가능성 | 없음 (관리 서버) |
| CUDA Toolkit 버전 | 13.0 (Green Contexts는 12.4+부터 공식 지원 → 완전 호환) |

RTX 5060 Ti 환경에서도 libsmctrl은 이미 작동하지 않아 MPS fallback으로 동작했으나, MPS fallback은 이미 초기화된 CUDA context에서는 SM 비율 변경이 무효화되는 구조적 한계가 있었다. A100 CUDA 13.0 환경에서는 libsmctrl 자체가 로드되지 않으므로 의존성 자체를 제거해야 한다.

---

## 2. 현재 libsmctrl 의존성 전체 목록

### 2-1. 직접 의존 (import 또는 ctypes load)

| 파일 | 의존 형태 | 사용 방식 |
|------|-----------|-----------|
| `src/smctrl/libsmctrl_wrapper.py` | `ctypes.CDLL("libsmctrl.so")` | `smctrl_set_mask_for_current_ctx` ioctl 호출 |
| `src/smctrl/overhead_timer.py` | `from .libsmctrl_wrapper import SMController` | SMController.set_sm_ratio() 반복 호출 |
| `src/models/layer_runner.py` | `from src.smctrl.libsmctrl_wrapper import SMController` | `set_sm_count()` / `reset()` — 레이어 실행 전후 |
| `stage2_overhead/measure_smctrl_latency.py` | `from src.smctrl.libsmctrl_wrapper import SMController` | 오버헤드 직접 측정 |
| `stage3_hm_eval/run_concurrent_eval.py` | `from src.smctrl.libsmctrl_wrapper import SMController` | Policy 객체에 smctrl 주입 |

### 2-2. 간접 의존 (TYPE_CHECKING 참조)

| 파일 | 참조 방식 | 실제 영향 |
|------|-----------|-----------|
| `stage3_hm_eval/policy_baseline.py` | `TYPE_CHECKING` import | runtime 영향 없음, 타입 힌트만 |
| `stage3_hm_eval/policy_step_adaptive.py` | `TYPE_CHECKING` import | runtime 영향 없음 |
| `stage3_hm_eval/policy_layer_wise.py` | `TYPE_CHECKING` import | runtime 영향 없음 |

### 2-3. 의존성 그래프

```
libsmctrl.so (C shared library)
    └── src/smctrl/libsmctrl_wrapper.py  ← ctypes 바인딩
            ├── src/smctrl/overhead_timer.py
            ├── src/models/layer_runner.py
            │       ├── stage1_sm_scaling/run_*_prefill_sweep.py (3개)
            │       └── stage3_hm_eval/run_concurrent_eval.py
            └── stage2_overhead/measure_smctrl_latency.py
```

---

## 3. CUDA Green Contexts 기술 상세

### 3-1. API 개요

CUDA 12.4(driver 550+)에서 도입된 공식 SM 파티셔닝 API. A100 CUDA 13.0에서 완전 지원.

```c
// 핵심 API (libcuda.so, driver API)
CUresult cuDevSmCount_v1(CUdevice, unsigned int *);
CUresult cuGreenCtxCreate(CUgreenCtx *, CUdevResourceDesc, CUdevice, unsigned int flags);
CUresult cuGreenCtxStreamCreate(CUstream *, CUgreenCtx, unsigned int flags, int priority);
CUresult cuCtxFromGreenCtx(CUcontext *, CUgreenCtx);
CUresult cuGreenCtxDestroy(CUgreenCtx);
CUresult cuDevResourceGenerateDesc(CUdevResourceDesc *, CUdevResource *, unsigned int);
```

### 3-2. libsmctrl vs Green Contexts 비교

| 특성 | libsmctrl | Green Contexts |
|------|-----------|----------------|
| 지원 driver | ~545 이하 | 550+ (CUDA 12.4+) |
| A100 CUDA 13.0 | **미지원** | **지원** |
| SM 제어 단위 | TPC (2 SM/TPC) | SM (1 SM 단위) |
| 변경 타이밍 | 현재 context에 즉시 적용 | context 생성 시 고정 → stream 전환으로 변경 |
| 동시 병렬 격리 | 불가 (단일 context 내 마스킹) | 가능 (별도 context, stream) |
| Python 바인딩 | ctypes + .so | ctypes + libcuda.so (driver API) |
| 오버헤드 | ~1–10 μs (ioctl) | ~수 μs (stream sync + switch) |

### 3-3. A100에서의 SM 파티셔닝 구조

```
A100 총 108 SM
├── GreenContext[prefill_ssm]  : 76 SM (70%)  → ssm prefill stream
├── GreenContext[prefill_attn] : 43 SM (40%)  → attn prefill stream  
└── GreenContext[decode]       : 32~65 SM     → decode stream (complement)
```

**핵심 제약**: Green Context는 생성 시 SM 수가 고정된다. libsmctrl처럼 런타임에 동적으로 SM 비율을 변경하는 방식이 아니라, **SM 비율별 context를 사전 생성**해두고 stream을 전환하는 방식으로 Policy C의 레이어 경계 SM 변경을 구현한다.

---

## 4. 새 아키텍처 설계

### 4-1. 신규 모듈 구조

```
src/smctrl/
├── __init__.py                  ← SMController export 유지 (인터페이스 호환)
├── green_ctx_controller.py      ← 신규: Green Contexts 기반 SMController 구현
├── overhead_timer.py            ← 기존 유지 (SMController 인터페이스 재사용)
└── libsmctrl_wrapper.py         ← 삭제 예정
```

### 4-2. GreenContextController 설계

libsmctrl_wrapper.py의 `SMController`와 **동일한 public 인터페이스**를 유지해 호출부(`layer_runner.py`, `policy_*.py`) 수정을 최소화한다.

```python
# src/smctrl/green_ctx_controller.py

class SMController:
    """Green Contexts 기반 SM 파티셔닝 컨트롤러.
    
    libsmctrl_wrapper.SMController와 동일한 public 인터페이스.
    내부적으로 SM 비율별 GreenContext + stream을 사전 생성.
    
    동작 방식:
      - set_sm_count(n) / set_sm_ratio(r): n_sm에 가장 가까운
        사전 생성 GreenContext의 stream을 반환 (또는 현재 stream 교체)
      - reset(): 전체 SM GreenContext로 복귀
    """

    def __init__(self, device_id=0, total_sm_count=None,
                 preset_ratios=(0.40, 0.50, 0.60, 0.70, 1.0)):
        ...
        self._green_contexts: dict[int, CUgreenCtx] = {}   # n_sm → green ctx
        self._streams: dict[int, CUstream] = {}            # n_sm → stream
        self._current_sm_count: int = self.total_sm_count
        self._lib: ctypes.CDLL = None                      # libcuda.so

        self._load_driver_api()
        self._create_preset_contexts(preset_ratios)

    # public 인터페이스 (libsmctrl_wrapper와 동일)
    def set_sm_count(self, n_sm: int) -> None: ...
    def set_sm_ratio(self, ratio: float) -> None: ...
    def reset(self) -> None: ...
    def get_stream(self) -> CUstream: ...       # 신규: 현재 SM에 대응하는 stream
    def is_available(self) -> bool: ...
    def get_backend_name(self) -> str: ...      # 반환값: "green_ctx"
    def verify_sm_control(self, verbose=True) -> bool: ...
    def measure_reconfigure_latency_us(self, ...) -> dict: ...
```

### 4-3. Stream 분리 방식

Green Contexts의 핵심 이점은 **stream 격리**다. SM 제한을 가진 stream에서 실행되는 커널은 해당 Green Context에 할당된 SM 범위 내에서만 실행된다.

```python
# layer_runner.py 수정 후 사용 패턴
def run_ssm_layer(self, ..., sm_count: int, ...):
    if not skip_sm_control:
        self.smctrl.set_sm_count(sm_count)   # 내부적으로 적절한 stream 선택
        stream = self.smctrl.get_stream()    # GreenContext에 bound된 stream
    else:
        stream = torch.cuda.current_stream() # 기존 stream 유지

    with torch.cuda.stream(stream):
        result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
```

---

## 5. 파일별 변경 사항

### 5-1. 신규 생성: `src/smctrl/green_ctx_controller.py`

libcuda.so를 ctypes로 바인딩해 Green Context API를 호출. 핵심 구현:

```python
import ctypes, os, math, time
import torch
import numpy as np

_LIBCUDA_NAMES = ["libcuda.so", "libcuda.so.1"]

# Green Context 관련 상수
CU_GREEN_CTX_DEFAULT_STREAM = 0x1
CU_DEV_RESOURCE_TYPE_SM = 0x1

class _GreenCtxLib:
    """libcuda.so driver API 바인딩 (Green Contexts 관련 심볼만)."""
    
    def __init__(self):
        self._lib = None
        for name in _LIBCUDA_NAMES:
            try:
                lib = ctypes.CDLL(name)
                # 필수 심볼 존재 확인 (CUDA 12.4+ 에서만 존재)
                _ = lib.cuGreenCtxCreate
                _ = lib.cuGreenCtxStreamCreate
                _ = lib.cuGreenCtxDestroy
                _ = lib.cuDevResourceGenerateDesc
                self._lib = lib
                self._bind_symbols()
                break
            except (OSError, AttributeError):
                continue
    
    def _bind_symbols(self):
        lib = self._lib
        # cuGreenCtxCreate(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned int)
        lib.cuGreenCtxCreate.restype = ctypes.c_int
        lib.cuGreenCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,               # CUdevResourceDesc (opaque struct)
            ctypes.c_int,                  # CUdevice
            ctypes.c_uint,                 # flags
        ]
        lib.cuGreenCtxStreamCreate.restype = ctypes.c_int
        lib.cuGreenCtxStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,               # CUgreenCtx
            ctypes.c_uint, ctypes.c_int,
        ]
        lib.cuGreenCtxDestroy.restype = ctypes.c_int
        lib.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]
    
    @property
    def available(self) -> bool:
        return self._lib is not None
```

**사전 생성 로직**: `preset_ratios = (0.40, 0.50, 0.60, 0.70, 1.0)` 에 대응하는 Green Context를 `__init__` 시 일괄 생성. 각 ratio에 가장 가까운 SM 수로 반올림 후 context 생성.

**`set_sm_count(n)`**: 가장 가까운 preset SM 수를 찾아 `_current_sm_count` 및 active stream을 교체. 비용 = Python dict lookup + stream 포인터 교체 → sub-microsecond.

### 5-2. 수정: `src/smctrl/__init__.py`

```python
# 기존
from .libsmctrl_wrapper import SMController
from .overhead_timer import SMOverheadTimer

# 변경 후
from .green_ctx_controller import SMController
from .overhead_timer import SMOverheadTimer

__all__ = ["SMController", "SMOverheadTimer"]
```

단 한 줄만 변경. 나머지 코드베이스는 `from src.smctrl import SMController` 형태로 import하므로 자동 적용.

### 5-3. 수정: `src/smctrl/overhead_timer.py`

import 문 변경 1줄만 필요. 나머지 로직 (`measure_single_transition`, `measure_n_transitions`, `measure_cold_start_penalty`) 은 `SMController`의 public 인터페이스를 사용하므로 무수정.

```python
# 기존
from .libsmctrl_wrapper import SMController
# 변경 후
from .green_ctx_controller import SMController
```

### 5-4. 수정: `src/models/layer_runner.py`

import 1줄 변경 + stream-aware 실행 패턴 적용.

```python
# 기존 import
from src.smctrl.libsmctrl_wrapper import SMController
# 변경 후
from src.smctrl.green_ctx_controller import SMController
```

각 `run_*_layer()` 메서드의 SM 제어 블록 수정:

```python
# 기존 패턴 (run_ssm_layer, run_attn_layer, run_mlp_layer 공통)
if not skip_sm_control:
    self.smctrl.set_sm_count(sm_count)
try:
    result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
finally:
    if not skip_sm_control:
        self.smctrl.reset()

# 변경 후
if not skip_sm_control:
    self.smctrl.set_sm_count(sm_count)
stream = self.smctrl.get_stream()
try:
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        result = self._measure(_run, n_warmup=n_warmup, n_measure=n_measure)
finally:
    if not skip_sm_control:
        self.smctrl.reset()
```

수정 위치: `layer_runner.py`의 3개 메서드 (`run_ssm_layer:182–188`, `run_attn_layer:268–272`, `run_mlp_layer:347–352`)

### 5-5. 수정: `stage2_overhead/measure_smctrl_latency.py`

스크립트 제목 및 meta 필드를 Green Contexts 기반으로 업데이트. SMController 인터페이스 호출부는 동일하므로 로직 변경 없음.

```python
# 기존
from src.smctrl.libsmctrl_wrapper import SMController
# 변경 후 (또는 __init__.py를 통해)
from src.smctrl import SMController

# meta 필드 변경
results["meta"]["backend"] = smctrl.get_backend_name()  # "green_ctx" 반환
```

파일명도 `measure_smctrl_latency.py` → `measure_ctx_switch_latency.py` 변경 권장 (Stage 2 목적 재정의 반영).

### 5-6. 수정: `stage3_hm_eval/run_concurrent_eval.py`

```python
# 기존
from src.smctrl.libsmctrl_wrapper import SMController
# 변경 후
from src.smctrl import SMController
```

Policy 객체(`PolicyBaseline`, `PolicyStepAdaptive`, `PolicyLayerWise`)에 SMController를 주입하는 방식은 동일. TYPE_CHECKING 참조이므로 policy 파일들 수정 불필요.

### 5-7. 삭제: `src/smctrl/libsmctrl_wrapper.py`

모든 참조가 `green_ctx_controller.py`로 이전 완료 후 삭제.

---

## 6. Stage별 영향 분석

### Stage 1 (SM scaling sweep)

**영향**: 간접. `LayerRunner`가 Green Contexts stream을 사용해 SM-restricted 커널을 실행.

**검증 포인트**: `verify_sm_control()` 통과 여부 (slowdown ratio ≥ 1.3 기대). Green Contexts는 하드웨어 수준 격리이므로 libsmctrl보다 격리 신뢰도가 높다.

**주의**: preset SM 비율의 반올림 오차. `set_sm_count(54)` → 가장 가까운 preset이 `round(108 * 0.5) = 54`이면 정확히 매핑. sm_sweep_steps 중 preset에 없는 값은 최근접 preset으로 snapping된다 → sweep 결과 CSV의 `sm_count` 열이 실제 커널 실행 SM 수와 다를 수 있음.

**해결**: `green_ctx_controller.py`에서 sweep에서 요청하는 모든 SM 수(sm_sweep_steps: `[11, 22, 33, 44, 54, 65, 87, 108]`)에 대응하는 Green Context를 생성. 또는 `__init__`에 `custom_sm_counts` 인자 추가.

### Stage 2 (reconfiguration overhead 측정)

**큰 변경**: 측정 대상이 바뀐다.

| 항목 | 기존 (libsmctrl) | 변경 후 (Green Contexts) |
|------|-----------------|------------------------|
| 측정 대상 | `smctrl_set_mask_for_current_ctx` ioctl 지연 | GreenContext stream 전환 지연 |
| 오버헤드 원천 | kernel ioctl + TPC mask flush | Python dict lookup + `torch.cuda.stream()` context manager |
| 예상 오버헤드 | ~1–10 μs | < 1 μs (stream 전환은 CPU-side pointer 교체) |
| 동기화 필요 | 있음 (mask가 즉시 적용되지 않을 수 있음) | 없음 (커널이 stream에 enqueue될 때 자동 적용) |

RTX 5060 Ti 실측 데이터 (`smctrl_overhead_geforce_rtx_5060_ti.json`) 참조:
- MPS fallback 기준: ssm→attn 전환 mean = **4.7 μs** (sync 포함)
- Green Contexts stream 전환: 예상 **< 0.5 μs** (CPU-side 연산만)

**Stage 2 재측정 범위**: `measure_smctrl_latency.py`를 `measure_ctx_switch_latency.py`로 재작성. stream 전환 오버헤드 + torch stream context 진입 비용 측정.

### Stage 3 (Policy A/B/C 평가)

**Policy A, B**: SM 비율 설정이 step 경계에서만 발생 → stream 전환 1회/step. 영향 최소.

**Policy C (layer_wise)**: 레이어마다 다른 SM 비율 → 레이어마다 stream 전환. 기존 libsmctrl의 ioctl보다 오버헤드가 낮을 것으로 예상 → `should_run_policy_c()` 조건이 더 쉽게 통과될 가능성.

**run_concurrent_eval.py**: `_run_decode_step`, `_run_prefill_layer` 내부에서 `LayerRunner`를 호출하는 방식은 동일. stream 전환은 `LayerRunner.run_*_layer()` 내부에서 처리.

---

## 7. 구현 시 주의사항

### 7-1. Green Context 생성 타이밍

`cuGreenCtxCreate`는 CUDA context가 초기화된 **이후** 호출해야 한다. `SMController.__init__`이 호출되는 시점에 `torch.cuda.is_available()`이 `True`이고 `torch.cuda.current_device()`가 유효해야 한다.

```python
# layer_runner.py 또는 sweep script에서
torch.cuda.init()   # CUDA context 명시적 초기화
smctrl = SMController(total_sm_count=108)  # Green Context 생성
```

### 7-2. torch.cuda.ExternalStream 사용

Green Context stream을 torch와 연동하려면 `torch.cuda.ExternalStream(raw_stream_ptr)`을 사용한다.

```python
import torch

stream_ptr = ctypes.c_void_p()
lib.cuGreenCtxStreamCreate(ctypes.byref(stream_ptr), green_ctx, 0, 0)

# torch에서 사용
ext_stream = torch.cuda.ExternalStream(stream_ptr.value)
with torch.cuda.stream(ext_stream):
    layer(hidden_states)  # 해당 GreenContext SM 범위에서 실행
```

### 7-3. SM 수 snapping 로그

`set_sm_count(n)`이 preset과 다른 값을 받을 경우 경고를 출력해 sweep 결과 해석 오류를 방지:

```python
def set_sm_count(self, n_sm: int) -> None:
    snapped = self._nearest_preset(n_sm)
    if snapped != n_sm:
        # sweep에서 요청한 SM 수와 실제 GreenContext SM 수가 다름을 기록
        pass  # sweep CSV에는 n_sm 원본값 기록, GreenContext는 snapped
    self._current_sm_count = snapped
```

### 7-4. MIG 모드 충돌

A100에서 MIG가 활성화된 경우 `cuGreenCtxCreate`는 `CUDA_ERROR_NOT_SUPPORTED`를 반환한다. MIG 비활성화 확인:

```bash
nvidia-smi mig -e 0  # 또는 관리자에게 요청
nvidia-smi -q | grep "MIG Mode"  # Current: Disabled 이어야 함
```

### 7-5. Python에서 cuDevResourceDesc 구성

`CUdevResourceDesc`는 CUDA driver API의 opaque struct. ctypes로 정확히 레이아웃을 맞춰야 한다. 공식 cuda.h 기준 구조:

```c
typedef struct CUdevResourceDesc_st {
    CUdevResourceType type;    // 4 bytes
    union {
        struct { unsigned int smCount; } sm;  // SM 수
    };
} CUdevResourceDesc;
```

pycuda가 설치된 환경이라면 `pycuda.driver.GreenContext` 래퍼를 활용하는 것도 대안이나, ctypes 직접 바인딩이 의존성을 최소화한다.

---

## 8. 구현 우선순위 및 체크리스트

```text
Phase 1 — 핵심 구현 (Green Contexts 동작 확인)
───────────────────────────────────────────────────────────
[ ] green_ctx_controller.py 작성
    [ ] _GreenCtxLib: libcuda.so 로드 + 심볼 바인딩
    [ ] SMController.__init__: preset SM 수별 GreenContext 생성
    [ ] set_sm_count / set_sm_ratio / reset / get_stream 구현
    [ ] is_available / get_backend_name("green_ctx") 구현
    [ ] verify_sm_control: 25% SM vs full SM 지연 비교
[ ] src/smctrl/__init__.py 교체 (1줄)
[ ] A100에서 verify_sm_control() 통과 확인 (ratio ≥ 1.3)

Phase 2 — 실행 레이어 통합
───────────────────────────────────────────────────────────
[ ] layer_runner.py import 변경 (1줄)
[ ] run_ssm_layer / run_attn_layer / run_mlp_layer stream 적용
    [ ] torch.cuda.ExternalStream 연동 확인
[ ] Stage 1 SSM sweep 재실행 (--device a100_40gb)
    [ ] sm_count별 latency 변화 확인 (SRM curve shape)

Phase 3 — Stage 2 재측정
───────────────────────────────────────────────────────────
[ ] measure_ctx_switch_latency.py 작성 (measure_smctrl_latency.py 대체)
    [ ] stream 전환 오버헤드 측정
    [ ] GreenContext 생성 오버헤드 측정 (일회성)
[ ] compute_decision_matrix.py 재실행 → decision_matrix.json 갱신

Phase 4 — Stage 3 연동 및 정리
───────────────────────────────────────────────────────────
[ ] run_concurrent_eval.py import 변경 (1줄)
[ ] Policy A/B/C 전체 재실행 (A100 기준)
[ ] smctrl_overhead_*.json meta.backend 확인 ("green_ctx" 기록)
[ ] src/smctrl/libsmctrl_wrapper.py 삭제
[ ] requirements.txt 확인 (libsmctrl 별도 패키지 없으면 변경 불필요)
```

---

## 9. 예상 오버헤드 변화

| 측정 항목 | RTX MPS fallback (실측) | 예상 Green Contexts (A100) |
|-----------|------------------------|---------------------------|
| ssm→attn 전환 (sync 포함) | 4.7 μs | < 1 μs |
| attn→ssm 전환 (sync 포함) | 8.2 μs | < 1 μs |
| n=54 레이어 총 전환 overhead | 173 μs | < 50 μs |
| per-layer 전환 overhead | 3.2 μs | < 1 μs |
| cold-start 커널 penalty | ~0 μs (MPS 무효) | ~0 μs (예상) |

Green Contexts stream 전환은 CPU-side pointer 교체이므로 ioctl 라운드트립이 없다. Policy C의 `overhead_ratio < 0.05` 조건이 A100에서 더 쉽게 충족될 것으로 예상.

---

## 10. 리스크 및 미결 사항

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| `cuGreenCtxCreate` rc ≠ 0 (MIG 활성화) | 중간 | MIG 비활성화 확인 후 재시도 |
| `cuDevResourceDesc` ctypes 레이아웃 불일치 | 중간 | cuda.h 직접 참조 또는 nvcuda 헤더 파싱 |
| torch.cuda.ExternalStream이 GreenCtx stream과 동기화 불일치 | 낮음 | 레이어 실행 후 `torch.cuda.synchronize()` 명시 |
| preset SM 수 snapping으로 sweep 분해능 저하 | 낮음 | `custom_sm_counts` 인자로 sweep 전체 SM 값 사전 생성 |
| A100에서 Green Contexts 실제 SM 격리 미보장 | 낮음 | `verify_sm_control()` + ncu SM utilization으로 검증 |
