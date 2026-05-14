# prefill-layer-alloc: Chunked Prefill 기반 SSM SM Partitioning 실험

## 배경 및 핵심 인사이트

이 태스크는 기존 `stage1_sm_scaling/` 실험의 근본적인 설계 문제를 수정한다.

### 기존 방식의 문제

기존 Stage 1은 전체 시퀀스를 단일 `mamba_chunk_scan_combined` 호출로 실행한 뒤
Green Context로 SM을 제한하려 했다.

```python
# 기존 방식 — 실패
y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256)
# seq=4096, bs=32 → kernel 내부 block 수 = batch × nchunks × nheads = 수만 개
# 14 SM으로 제한 → cooperative grid.sync() barrier에서 deadlock
# → CUDA error: illegal memory access → context 오염 → 연쇄 실패
```

`mamba_chunk_scan_combined`는 Triton SSD fused kernel로,
청크 간 hidden state 전파를 위해 `grid.sync()`(cooperative barrier)를 사용한다.
이 barrier는 "현재 kernel launch의 모든 block이 동시에 active"여야 완료된다.
Green Context로 SM을 극소수로 제한하면 block 수 >> active SM이 되어 deadlock이 발생한다.

### 올바른 접근: Chunked Prefill (kernel 경계에서 state 전달)

Falcon-H1 논문(Section 3.3.3, Context Parallelism)과 Marconi 논문에서 동일한 패턴을 사용한다:
시퀀스를 **kernel 호출 수준에서 분할**하고 청크 경계마다 SSM state를 명시적으로 전달한다.

```python
# 올바른 방식 — chunked prefill
prefill_chunk_tokens = 512  # kernel 호출당 토큰 수

state = zeros(batch, n_heads, head_dim, d_state)  # initial state
for start in range(0, seq_len, prefill_chunk_tokens):
    chunk = x[:, start:start + prefill_chunk_tokens]
    # 이 kernel call은 prefill_chunk_tokens 분량만 처리
    # block 수 = batch × (prefill_chunk_tokens // 256) × nheads
    # = 1 × 2 × 112 = 224 (bs=1, Zamba2-7B, chunk=512)
    # → 68 SM에서 ceil(224/68)=4 waves → cooperative 조건 성립
    y_chunk, state = mamba_chunk_scan_combined(
        chunk, ..., ssm_initial_states=state, return_final_states=True
    )
    # kernel 호출 사이에 Green Context 전환 가능
```

각 kernel call 사이에 Green Context를 바꿀 수 있다.
청크 간 state 전달은 `grid.sync()`가 아니라 host-side 동기화(kernel 경계)이므로
cooperative 제약을 벗어난다.

---

## 환경 정보

- **하드웨어**: A100-SXM4-80GB (108 SM, HBM2e 1,000 GB/s)
- **모델**: Zamba2-7B-Instruct (SSM-heavy, 81 layers: 68 SSM + 13 hybrid Attn)
- **기존 프로젝트 경로**: `prefill-layer-alloc/`
- **Green Context 인프라**: `src/smctrl/green_ctx_controller.py` (기존 구현 재사용)
- **Layer runner 인프라**: `src/models/layer_runner.py` (기존 구현 재사용)
- **기존 Stage 1 결과**: `results/stage1/` (Attention scaling은 유효, SSM은 재측정 필요)

---

## 구현 태스크

### Task 1: `chunked_ssm_runner.py` 신규 작성

**경로**: `stage1_sm_scaling/chunked_ssm_runner.py`

**목적**: 전체 시퀀스를 `prefill_chunk_tokens` 크기로 나눠 SSM prefill을 실행하면서
각 kernel call 전에 Green Context로 SM 수를 설정하는 runner.

```python
"""
chunked_ssm_runner.py

기존 run_ssm_prefill_sweep.py의 Green Context 직접 측정 대체.
mamba_chunk_scan_combined를 청크 단위로 호출해 cooperative barrier 우회.
"""

import torch
import time
from pathlib import Path
from typing import Optional
import csv

# 프로젝트 내 기존 인프라 재사용
from src.smctrl.green_ctx_controller import SMController
from src.models.layer_runner import LayerRunner  # 기존 Attn/MLP runner 참고용


def run_chunked_ssm_sweep(
    model_name: str,          # "zamba2"
    seq_len: int,             # 전체 시퀀스 길이 (측정 대상)
    batch_size: int,
    prefill_chunk_tokens: int, # kernel 호출당 토큰 수 (핵심 파라미터)
    sm_count: int,            # Green Context에 적용할 SM 수
    smctrl: SMController,
    n_warmup: int = 3,
    n_measure: int = 10,
    device: str = "cuda",
) -> dict:
    """
    chunked prefill 방식으로 SSM layer를 실행하고 latency를 측정한다.

    Args:
        prefill_chunk_tokens: 청크 크기. 너무 작으면 kernel launch overhead 증가,
                               너무 크면 block 수가 많아져 cooperative 위험.
                               권장: SM 수 × (SSD chunk_size=256) 이하
                               예: 68 SM × 256 = 17,408 → 4,096~8,192가 안전

    Returns:
        {
            "latency_ms": float,          # median latency
            "latency_std_ms": float,
            "n_kernel_calls": int,        # seq_len // prefill_chunk_tokens
            "n_blocks_per_call": int,     # batch × (prefill_chunk_tokens//256) × nheads
            "cooperative_safe": bool,     # n_blocks_per_call <= sm_count * max_blocks_per_sm
            "sm_count": int,
        }
    """
    # 1. 모델 파라미터 로드 (기존 LayerRunner 참고)
    #    Zamba2-7B: d_model=3584, n_heads=112, head_dim=64, d_state=64, chunk_size=256
    #    configs/models.yaml에서 읽어올 것
    model_cfg = load_model_config(model_name)
    d_model    = model_cfg["d_model"]      # 3584
    n_heads    = model_cfg["n_heads"]      # 112
    head_dim   = model_cfg["head_dim"]     # 64
    d_state    = model_cfg["d_state"]      # 64
    ssd_chunk  = model_cfg["chunk_size"]   # 256

    # 2. cooperative 안전성 체크
    nchunks_per_call = prefill_chunk_tokens // ssd_chunk
    # Triton SSD 주요 kernel grid = batch × nchunks × nheads
    n_blocks_per_call = batch_size * nchunks_per_call * n_heads
    # A100: SM당 최대 동시 block ≈ 2 (register/shared mem 점유 기반)
    max_blocks_per_sm = 2
    cooperative_safe = n_blocks_per_call <= sm_count * max_blocks_per_sm

    n_calls = seq_len // prefill_chunk_tokens

    # 3. 입력 텐서 준비 (전체 시퀀스)
    x_full = torch.randn(batch_size, seq_len, d_model,
                         dtype=torch.bfloat16, device=device)

    # 4. Green Context 설정
    smctrl.set_sm_count(sm_count)
    stream = smctrl.get_stream()

    def _run_once():
        """전체 시퀀스를 chunked 방식으로 처리."""
        # 초기 SSM state (zeros)
        # shape: [batch, n_heads, head_dim, d_state]
        ssm_state = torch.zeros(
            batch_size, n_heads, head_dim, d_state,
            dtype=torch.float32, device=device
        )

        with torch.cuda.stream(stream):
            for start in range(0, seq_len, prefill_chunk_tokens):
                end = min(start + prefill_chunk_tokens, seq_len)
                chunk = x_full[:, start:end, :]

                # mamba_chunk_scan_combined with initial_state
                # return_final_states=True로 다음 청크의 init state 획득
                from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
                # 실제 Zamba2 SSM layer forward를 그대로 호출
                # (src/models/zamba2.py의 FallbackSSMKernel.forward 참고)
                y_chunk, ssm_state = _call_ssm_layer(
                    chunk, ssm_state, model_cfg, device
                )
        torch.cuda.synchronize()

    # 5. 워밍업
    for _ in range(n_warmup):
        _run_once()

    # 6. 측정
    latencies = []
    for _ in range(n_measure):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        _run_once()
        end_event.record(stream)
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    import statistics
    return {
        "latency_ms":           statistics.median(latencies),
        "latency_std_ms":       statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_list_ms":      latencies,
        "n_kernel_calls":       n_calls,
        "n_blocks_per_call":    n_blocks_per_call,
        "cooperative_safe":     cooperative_safe,
        "sm_count":             sm_count,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "seq_len":              seq_len,
        "batch_size":           batch_size,
    }
```

**`_call_ssm_layer` 구현 요구사항**:
- `src/models/zamba2.py`의 `FallbackSSMKernel.forward()`를 참고해 작성
- `mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D, dt_bias, dt_softplus, ssm_initial_states=state, return_final_states=True)` 호출
- dt, A, B, C는 linear projection 결과지만 측정 목적이므로 random으로 초기화해도 무방
- `ssm_initial_states`와 `return_final_states` 파라미터 지원 여부를 mamba_ssm 버전에서 확인할 것

---

### Task 2: `run_chunked_ssm_sweep.py` 작성

**경로**: `stage1_sm_scaling/run_chunked_ssm_sweep.py`

**기존 `run_ssm_prefill_sweep.py`를 대체**하는 sweep 스크립트.

```python
"""
run_chunked_ssm_sweep.py

사용법:
    python stage1_sm_scaling/run_chunked_ssm_sweep.py \
        --model zamba2 \
        --device a100-sxm4-80gb \
        --prefill-chunk-tokens 512 1024 2048 4096 \
        --sm-counts 14 27 40 54 68 81 94 108 \
        --seq-lens 512 1024 2048 4096 8192 \
        --batch-sizes 1 4 16 32 \
        --output-dir results/stage1/chunked/
"""

import argparse
import csv
import json
from pathlib import Path
import torch

from src.smctrl.green_ctx_controller import SMController
from src.hardware_config import get_hardware_config
from stage1_sm_scaling.chunked_ssm_runner import run_chunked_ssm_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="zamba2")
    parser.add_argument("--device",    default="a100-sxm4-80gb",
                        help="hardware config tag (hardware.yaml)")
    parser.add_argument("--prefill-chunk-tokens", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096],
                        help="prefill_chunk_tokens sweep (핵심 독립 변수)")
    parser.add_argument("--sm-counts", nargs="+", type=int,
                        default=[14, 27, 40, 54, 68, 81, 94, 108])
    parser.add_argument("--seq-lens",  nargs="+", type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 4, 16])
    parser.add_argument("--n-warmup",  type=int, default=3)
    parser.add_argument("--n-measure", type=int, default=10)
    parser.add_argument("--output-dir", default="results/stage1/chunked/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hw_cfg = get_hardware_config(args.device)
    smctrl = SMController(
        device_id=0,
        total_sm_count=hw_cfg["sm_count"],  # A100: 108
        preset_sm_counts=args.sm_counts,
    )

    # Green Context 유효성 검증 (cooperative barrier 문제 재발 방지)
    if not smctrl.verify_sm_control():
        raise RuntimeError(
            "SMController 검증 실패. Green Context가 실제로 SM을 제한하는지 확인:\n"
            "  latency(14 SM) >> latency(108 SM)이어야 함 (ratio >= 2.0)\n"
            "  현재 flat하다면 driver가 Green Context를 무시하는 환경임"
        )

    # CSV 헤더
    fieldnames = [
        "model", "device", "seq_len", "batch_size", "sm_count", "sm_ratio_pct",
        "prefill_chunk_tokens", "n_kernel_calls", "n_blocks_per_call",
        "cooperative_safe", "latency_ms", "latency_std_ms",
    ]

    output_file = output_dir / f"ssm_chunked_{args.model}_{args.device}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = (len(args.prefill_chunk_tokens) * len(args.sm_counts)
                 * len(args.seq_lens) * len(args.batch_sizes))
        done = 0

        for pct in args.prefill_chunk_tokens:
            for sm in args.sm_counts:
                for seq in args.seq_lens:
                    # seq < pct면 skip (청크 크기가 전체 시퀀스보다 클 수 없음)
                    if seq < pct:
                        done += len(args.batch_sizes)
                        continue
                    for bs in args.batch_sizes:
                        done += 1
                        print(f"[{done}/{total}] pct={pct} sm={sm} seq={seq} bs={bs}")

                        try:
                            result = run_chunked_ssm_sweep(
                                model_name=args.model,
                                seq_len=seq,
                                batch_size=bs,
                                prefill_chunk_tokens=pct,
                                sm_count=sm,
                                smctrl=smctrl,
                                n_warmup=args.n_warmup,
                                n_measure=args.n_measure,
                            )
                        except RuntimeError as e:
                            print(f"  ERROR: {e}")
                            result = {
                                "latency_ms": float("nan"),
                                "latency_std_ms": float("nan"),
                                "n_kernel_calls": seq // pct,
                                "n_blocks_per_call": -1,
                                "cooperative_safe": False,
                            }

                        row = {
                            "model":                args.model,
                            "device":               args.device,
                            "seq_len":              seq,
                            "batch_size":           bs,
                            "sm_count":             sm,
                            "sm_ratio_pct":         round(sm / hw_cfg["sm_count"] * 100, 1),
                            "prefill_chunk_tokens": pct,
                            "n_kernel_calls":       result["n_kernel_calls"],
                            "n_blocks_per_call":    result["n_blocks_per_call"],
                            "cooperative_safe":     result["cooperative_safe"],
                            "latency_ms":           result["latency_ms"],
                            "latency_std_ms":       result["latency_std_ms"],
                        }
                        writer.writerow(row)
                        f.flush()

    print(f"\nDone. 결과 저장: {output_file}")


if __name__ == "__main__":
    main()
```

---

### Task 3: `prefill_chunk_tokens` 최적값 분석 스크립트

**경로**: `stage1_sm_scaling/analyze_chunk_size.py`

```python
"""
analyze_chunk_size.py

run_chunked_ssm_sweep.py 결과를 읽어서:
1. cooperative_safe=True인 구간 확인
2. 같은 (seq_len, batch_size, sm_count) 조건에서 prefill_chunk_tokens별 latency 비교
3. 최적 prefill_chunk_tokens 권장

사용법:
    python stage1_sm_scaling/analyze_chunk_size.py \
        --csv results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print("=== cooperative_safe=False 케이스 ===")
    unsafe = df[~df["cooperative_safe"]]
    if len(unsafe) > 0:
        print(unsafe[["seq_len","batch_size","sm_count","prefill_chunk_tokens",
                       "n_blocks_per_call","latency_ms"]].to_string(index=False))
    else:
        print("없음 (모든 케이스 안전)")

    print("\n=== prefill_chunk_tokens별 overhead 분석 ===")
    print("(seq=2048, bs=4, sm=108 기준, 단일 kernel 대비 overhead)")

    base = df[
        (df["seq_len"] == 2048) &
        (df["batch_size"] == 4) &
        (df["sm_count"] == 108)
    ].set_index("prefill_chunk_tokens")["latency_ms"]

    full_seq_latency = base.get(2048, None)  # pct=seq_len = 단일 kernel과 동등
    if full_seq_latency is not None:
        for pct, lat in base.items():
            overhead_pct = (lat - full_seq_latency) / full_seq_latency * 100
            n_calls = 2048 // pct
            print(f"  pct={pct:5d} n_calls={n_calls:3d} "
                  f"latency={lat:.2f}ms overhead={overhead_pct:+.1f}%")

    print("\n=== SM scaling curve (pct=512, seq=2048, bs=4) ===")
    sub = df[
        (df["prefill_chunk_tokens"] == 512) &
        (df["seq_len"] == 2048) &
        (df["batch_size"] == 4)
    ].sort_values("sm_count")

    full_sm_lat = sub[sub["sm_count"] == 108]["latency_ms"].values
    if len(full_sm_lat) > 0:
        full_sm_lat = full_sm_lat[0]
        print(f"{'sm':>5} {'latency_ms':>12} {'throughput':>12} {'safe':>6}")
        for _, row in sub.iterrows():
            tp = full_sm_lat / row["latency_ms"]  # normalized throughput
            print(f"{int(row['sm_count']):>5} {row['latency_ms']:>12.2f} "
                  f"{tp:>12.3f} {str(row['cooperative_safe']):>6}")

    print("\n=== 권장 prefill_chunk_tokens ===")
    # 안전하고 overhead < 5%인 가장 작은 pct
    safe_df = df[df["cooperative_safe"] == True]
    if len(safe_df) > 0:
        min_pct = safe_df["prefill_chunk_tokens"].min()
        print(f"  최소 안전 pct: {min_pct} tokens")
        print(f"  A100 108 SM 기준 권장: seq_len에 따라 동적 설정")
        print(f"    n_blocks_per_call = batch × (pct//256) × n_heads ≤ sm_count × 2")
        print(f"    Zamba2 bs=4: pct ≤ sm_count × 2 × 256 / (4 × 112) = sm_count × 1.14")
        print(f"    sm=68: pct ≤ 77 tokens → pct=256 (1 SSD chunk) 권장")


if __name__ == "__main__":
    main()
```

---

### Task 4: 기존 `plot_compare_modules.py` 업데이트

기존 시각화가 SSM을 wave model 합성값으로 표시했다면,
chunked 실험 결과를 실측값으로 오버레이하는 옵션을 추가한다.

```python
# plot_compare_modules.py에 추가할 옵션
parser.add_argument("--ssm-chunked-csv",
    help="chunked prefill 방식으로 직접 측정한 SSM CSV (있으면 wave model 대신 사용)")
```

---

## 실행 순서

```bash
# Step 1: cooperative 안전성 확인 (소규모 테스트)
python stage1_sm_scaling/run_chunked_ssm_sweep.py \
    --model zamba2 \
    --device a100-sxm4-80gb \
    --prefill-chunk-tokens 256 512 1024 \
    --sm-counts 14 27 54 108 \
    --seq-lens 1024 2048 \
    --batch-sizes 1 4 \
    --n-warmup 2 --n-measure 5 \
    --output-dir results/stage1/chunked/

# Step 2: 결과 분석
python stage1_sm_scaling/analyze_chunk_size.py \
    --csv results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv

# Step 3: 안전한 pct 확인 후 전체 sweep
python stage1_sm_scaling/run_chunked_ssm_sweep.py \
    --model zamba2 \
    --device a100-sxm4-80gb \
    --prefill-chunk-tokens 512 1024 2048 4096 \  # Step 2 결과 기반으로 선택
    --sm-counts 14 27 40 54 68 81 94 108 \
    --seq-lens 256 512 1024 2048 4096 8192 \
    --batch-sizes 1 4 16 32 \
    --n-warmup 3 --n-measure 10 \
    --output-dir results/stage1/chunked/

# Step 4: 기존 Attn/MLP 결과와 함께 시각화
python stage1_sm_scaling/plot_compare_modules.py \
    --device a100-sxm4-80gb \
    --ssm-chunked-csv results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv
```

---

## 구현 시 주의사항

### 1. mamba_ssm API 확인 필수

`mamba_chunk_scan_combined`의 `ssm_initial_states`와 `return_final_states` 파라미터
지원 여부는 mamba_ssm 버전에 따라 다르다. 먼저 확인:

```python
import inspect
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
print(inspect.signature(mamba_chunk_scan_combined))
```

지원하지 않는다면 `chunk_state_varlen`이나 별도 state passing 로직을 작성해야 한다.

### 2. cooperative_safe 계산 공식

```python
# Triton SSD 주요 kernel의 grid 크기 (실제 코드에서 확인 필요)
# mamba_ssm/ops/triton/ssd_combined.py의 grid lambda 참고
nchunks = prefill_chunk_tokens // 256  # ssd_chunk_size=256
n_blocks_per_call = batch * nchunks * n_heads  # 가장 큰 grid를 갖는 kernel 기준

# A100 SM당 최대 동시 block: register/shared mem 점유에 따라 1~2
# mamba_chunk_scan_combined는 heavy register usage → max_blocks_per_sm=1 가정이 안전
cooperative_safe = n_blocks_per_call <= sm_count * 1
```

### 3. 측정 대상 명확화

`run_chunked_ssm_sweep.py`가 측정하는 것:
- **전체 시퀀스 처리 latency** = n_kernel_calls개의 kernel call 합계
- Green Context로 `sm_count` SM에 제한된 상태에서의 실제 latency

기존 wave model 합성과의 차이:
- Wave model: `latency(sm_k) = latency(full) × ⌈n_blocks/k⌉ / ⌈n_blocks/full⌉`
- 이 실험: 실제 kernel 실행 시간을 CUDA event로 직접 측정

두 결과를 비교해 wave model의 정확도를 검증하는 것이 부가 목표다.

### 4. 기존 인프라 재사용

- `src/smctrl/green_ctx_controller.py`: SMController 그대로 사용
- `src/hardware_config.py` (또는 `configs/hardware.yaml`): A100 preset SM counts
- `configs/models.yaml`: Zamba2 모델 파라미터 (d_model, n_heads 등)
- `src/models/zamba2.py`: `FallbackSSMKernel` 구조 참고 (dt, A, B, C 준비 방식)

새로 작성하는 파일:
- `stage1_sm_scaling/chunked_ssm_runner.py` (핵심)
- `stage1_sm_scaling/run_chunked_ssm_sweep.py`
- `stage1_sm_scaling/analyze_chunk_size.py`

---

## 예상 결과 해석

### 성공 케이스
```
cooperative_safe=True, latency가 sm_count에 반비례 → 실측 SM scaling curve 확보
→ wave model 합성값과 비교 → "wave model 오차 X%" 정량화
→ Free SM Zone을 실측값으로 업데이트
```

### 실패 케이스 (여전히 deadlock)
```
cooperative_safe=True임에도 CUDA error 발생
→ n_blocks_per_call 계산 공식이 틀림 (실제 grid가 더 큰 kernel 존재)
→ ncu로 실제 grid 크기 확인 후 공식 수정
→ pct를 더 줄이거나 sm_count를 높여 재시도
```

### 예상 overhead 패턴
```
pct=256 (1 SSD chunk): 커널 호출 횟수 많음 → launch overhead 누적 → latency +10~20%
pct=1024:              적절한 균형점
pct=seq_len:           단일 kernel과 동등 → cooperative 위험은 여전히 존재
```

---

## 이 실험이 연구에 갖는 의의

기존 Stage 1이 "SSM은 SM 분할 불가"라는 결론을 내린 근거는
단일 fused kernel에서 Green Context로 SM을 극소수(14개)로 제한했을 때 deadlock이 발생한 것이었다.

Chunked prefill 방식으로 실험하면:
1. SSM prefill의 실제 SM scaling curve를 직접 측정 가능
2. 특정 (prefill_chunk_tokens, sm_count) 조합에서 cooperative 조건이 성립하는 구간을 정량화
3. fig2(Free SM Zone)를 wave model 합성이 아닌 실측값으로 업데이트
4. Stage 3에서 실제로 SM을 분할한 concurrent prefill+decode 실험의 근거 데이터 확보

이는 "HM이 가능한 operating range는 어디인가"에 대한 답을 실측으로 제공한다.
