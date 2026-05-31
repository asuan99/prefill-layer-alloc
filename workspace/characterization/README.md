# prefill-layer-alloc

Hybrid SSM+Attention 모델의 serving에서 **prefill 중 layer type 경계마다 SM(Streaming Multiprocessor)을 동적으로 재할당**하는 것이 실질적 이득을 주는가를 검증하는 실험 프레임워크.

## 연구 질문

1. SSM prefill layer는 몇 개의 SM에서 수확체감이 시작되는가? (SM saturation point)
2. CUDA Green Contexts stream 전환 overhead는 prefill layer 실행 시간 대비 얼마인가?
3. 위 두 결과에 따라 layer-wise / step-level / 고정 분할 중 어느 전략이 최적인가?

## 선행 프로젝트

- `mixer-alloc`: 고정 TPC ratio별 전체 forward pass latency → optimal ratio 탐색
- `mamba-cuda-graph`: decode CUDA Graph + SM isolation → TPOT/TTFT 측정
- **이 프로젝트**: prefill+decode 동시 실행 중 layer 경계 SM 재분배 효과 측정

---

## 대상 모델

| 모델 | HF repo | 레이어 구성 | hidden size |
|------|---------|------------|-------------|
| Zamba2-7B-Instruct | `Zyphra/Zamba2-7B-Instruct` | 81층 (68 pure SSM + 13 hybrid SSM+Attn) | 3584 |
| Falcon-H1-7B-Instruct | `tiiuae/Falcon-H1-7B-Instruct` | 44층 (전층 SSM+Attn 병렬) | 3072 |

Zamba2는 pure SSM 레이어가 대다수(84%)를 차지하고 Attn이 13개 레이어에만 존재한다. Falcon-H1은 모든 레이어에 SSM branch와 Attention branch가 병렬로 존재한다.

---

## SM 제어 백엔드

### CUDA Green Contexts (primary, 현재 활성)

- CUDA 12.4+ (driver 550+) 공식 SM 파티셔닝 API
- A100 CUDA driver 13.0에서 완전 지원
- `src/smctrl/green_ctx_controller.py`에 구현
- **동작 방식**: SM 비율별 GreenContext + ExternalStream을 `__init__` 시 사전 생성. `set_sm_count(n)`은 가장 가까운 preset의 stream pointer만 교체 → sub-μs overhead
- 커널은 반드시 `with torch.cuda.stream(smctrl.get_stream()):` 블록 내에서 실행해야 SM 제한이 적용됨. `LayerRunner`가 자동으로 처리

```
A100 Green Context 파티션 예시 (108 SM)
├── GreenContext[ssm_prefill]  : 76 SM (70%)  → SSM prefill stream
├── GreenContext[attn_prefill] : 43 SM (40%)  → Attn prefill stream
└── GreenContext[decode]       : 32~65 SM     → decode stream (complement)
```

#### Green Contexts vs libsmctrl

| 특성 | libsmctrl | Green Contexts |
|------|-----------|----------------|
| 지원 driver | ~545 이하 | 550+ (CUDA 12.4+) |
| A100 CUDA 13.0 | **미지원** | **지원** |
| SM 제어 단위 | TPC (2 SM/TPC) | SM (1 SM 단위) |
| 변경 타이밍 | 현재 context에 즉시 | stream 전환으로 변경 |
| 동시 격리 | 불가 | 가능 (별도 context) |
| overhead | ~1–10 μs (ioctl) | < 1 μs (CPU-side 연산) |

---

## 환경 설정

```bash
pip install -r requirements.txt

# CUDA context 초기화 확인 (Green Contexts는 torch.cuda.init() 이후 생성 가능)
python -c "
import torch; torch.cuda.init()
from src.smctrl import SMController
c = SMController()
print('Backend:', c.get_backend_name())   # 'green_ctx' 이어야 함
print('SM control works:', c.verify_sm_control())
"
```

#### A100 체크리스트

```bash
# MIG 비활성화 확인 (MIG ON이면 cuGreenCtxCreate → CUDA_ERROR_NOT_SUPPORTED)
nvidia-smi -q | grep "MIG Mode"   # Current: Disabled 이어야 함

# ncu 프로파일링 권한
cat /proc/sys/kernel/perf_event_paranoid   # 2 이하이어야 함

# SM 제어 동작 검증 (25% SM 시 ≥ 2× slowdown 확인)
python -c "
import torch; torch.cuda.init()
from src.smctrl import SMController
SMController().verify_sm_control(verbose=True)
"
```

---

## 실행 방법

### Stage 1: SM scaling curve 측정

각 layer type의 SM 수 대비 latency curve를 측정한다.

```bash
# SSM prefill layer SM sweep
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device a100_40gb
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model falcon_h1 --device a100_40gb

# Attention prefill layer SM sweep
python stage1_sm_scaling/run_attn_prefill_sweep.py --model zamba2 --device a100_40gb
python stage1_sm_scaling/run_attn_prefill_sweep.py --model falcon_h1 --device a100_40gb

# MLP layer SM sweep
python stage1_sm_scaling/run_mlp_prefill_sweep.py --model zamba2 --device a100_40gb

# 결과 시각화 (results/stage1/*.png)
python stage1_sm_scaling/plot_saturation.py
python stage1_sm_scaling/plot_srm.py --hw-config a100_40gb   # --hw-config 필수

# NCU 심층 프로파일 (SM occupancy, wave quantization)
python stage1_sm_scaling/run_ncu_profile.py --model zamba2 --device a100_40gb
```

> **주의**: `--device auto` 사용 시 `BandwidthEstimator`가 1000 GB/s fallback으로 기록됨.
> A100에서는 `--device a100_40gb`를 항상 명시적으로 지정한다.

### Stage 2: overhead 측정 및 전략 결정

```bash
# Green Contexts stream 전환 latency 측정
python stage2_overhead/measure_ctx_switch_latency.py --device a100_40gb

# 비교 기준 layer latency 수집 (Stage 1 완료 후 실행)
python stage2_overhead/measure_layer_latency.py --model zamba2 --device a100_40gb
python stage2_overhead/measure_layer_latency.py --model falcon_h1 --device a100_40gb

# 전략 자동 판단 (results/stage2/decision_matrix.{json,html})
python stage2_overhead/compute_decision_matrix.py \
    --stage1-dir results/stage1 --stage2-dir results/stage2
```

> RTX 5060 Ti 실측 기준 MPS fallback overhead: ssm→attn 전환 **4.7 μs**.
> Green Contexts stream 전환 예상: **< 1 μs** (CPU-side pointer 교체).

### Stage 3: 동시 실행 평가

```bash
# Stage 2 완료 후 실행
python stage3_hm_eval/run_concurrent_eval.py --model zamba2 --policy all --device a100_40gb
python stage3_hm_eval/run_concurrent_eval.py --model falcon_h1 --policy A B --device a100_40gb

# 논문용 Figure 생성
python stage3_hm_eval/plot_results.py
```

---

## 프로젝트 구조

```
prefill-layer-alloc/
├── configs/
│   ├── models.yaml          # zamba2-7b, falcon_h1-7b 설정 (layer 수, head 수 등)
│   └── hardware.yaml        # GPU별 SM count, BW (A100, H100, RTX 4090/5060Ti 등)
├── src/
│   ├── models/
│   │   ├── layer_runner.py      # LayerRunner: 단일 layer 독립 실행 + Green Ctx SM 제한
│   │   ├── zamba2.py            # Zamba2-7B layer 추출 + FallbackSSMKernel
│   │   └── falcon_h1.py         # Falcon-H1-7B layer 추출 + FallbackSSMBranch
│   ├── smctrl/
│   │   ├── __init__.py              # SMController export (green_ctx_controller 사용)
│   │   ├── green_ctx_controller.py  # SMController: CUDA Green Contexts 기반
│   │   └── overhead_timer.py        # SMOverheadTimer: stream 전환 latency 정밀 측정
│   └── profiling/
│       ├── cupti_monitor.py     # CUPTI 기반 SM utilization 수집
│       ├── nvml_monitor.py      # NVMLMonitor: 백그라운드 GPU 메트릭 수집
│       ├── ncu_runner.py        # NCU 자동화 실행 (wave 분석용)
│       ├── nvtx_markers.py      # NVTX range 마킹 (Nsight 연동)
│       ├── wave_estimator.py    # SM wave quantization 추정
│       └── metrics.py           # LatencyMeter (CUDA event), BandwidthEstimator
├── stage1_sm_scaling/
│   ├── run_ssm_prefill_sweep.py # SSM layer SM sweep (csv 출력)
│   ├── run_attn_prefill_sweep.py
│   ├── run_mlp_prefill_sweep.py
│   ├── run_ncu_profile.py       # NCU 심층 프로파일 (occupancy, wave)
│   ├── plot_saturation.py       # Figure 1 (scaling curve) + Figure 2 (free SM zone)
│   ├── plot_srm.py              # SRM Roofline 분석 (--hw-config 필수)
│   └── plot_sm_split.py
├── stage2_overhead/
│   ├── measure_ctx_switch_latency.py  # Green Contexts stream 전환 latency 측정
│   ├── measure_layer_latency.py       # 비교 기준 layer latency
│   └── compute_decision_matrix.py     # 전략 판단 (json + html)
├── stage3_hm_eval/
│   ├── policy_baseline.py         # Policy A: 고정 분할 (40/60)
│   ├── policy_step_adaptive.py    # Policy B: step-level 모델 특성 기반
│   ├── policy_layer_wise.py       # Policy C: layer boundary마다 stream 전환
│   ├── run_concurrent_eval.py     # Prefill+Decode 인터리빙 메인 루프
│   └── plot_results.py            # Figure 1–3 (논문용)
├── tests/
│   └── test_green_ctx_controller.py
├── a100_migration_report.md       # RTX→A100 이관 시 알려진 버그 및 체크리스트
├── green_ctx_migration_report.md  # libsmctrl→Green Contexts 교체 설계 문서
└── results/
    ├── stage1/  # ssm_scaling_*.csv, attn_scaling_*.csv, fig*.png
    ├── stage2/  # ctx_switch_overhead_*.json, layer_latency_*.csv, decision_matrix.*
    └── stage3/  # eval_*.csv, sm_timeline_*.csv, fig1–3 *.png
```

---

## Policy 비교

| Policy | 전략 | SM 전환 타이밍 | overhead |
|--------|------|--------------|---------|
| A (Baseline) | 고정 분할 40%/60% | 없음 | 0 |
| B (Step adaptive) | step 시작 시 layer 구성 기반 비율 결정 | 1회/step | < 1 μs |
| C (Layer-wise) | SSM↔Attn 경계마다 stream 전환 | 레이어마다 | < 1 μs × N |

Policy C의 `should_run_policy_c()` 조건: `overhead_ratio < 0.05` (전환 overhead가 layer latency의 5% 미만). Green Contexts는 stream pointer 교체이므로 RTX MPS fallback(4.7 μs)보다 훨씬 낮아 조건 통과 가능성이 높다.

---

## 알려진 이슈 (A100 이관 후)

자세한 내용은 [`a100_migration_report.md`](a100_migration_report.md) 참조.

| ID | 파일 | 증상 | 조치 |
|----|------|------|------|
| C-1 | `plot_srm.py` | SRM BW ceiling 오계산 (하드코딩 상수) | `--hw-config a100_40gb` 명시 |
| C-2 | `metrics.py` | `theoretical_bw_GBs=1000.0` 고정 (속성 누락) | `--device a100_40gb` 명시 |
| C-3 | `compute_decision_matrix.py` | RTX+A100 결과 혼합 → saturation 오판 | `results/` 디렉토리 분리 또는 `--device-tag` 필터 추가 |
| C-4 | Stage 2 전체 | RTX 기반 decision matrix로 A100 Policy 결정 | Stage 2 A100 재실행 필수 |
| C-5 | `hardware.yaml` | `rtx_5060ti` BW/TFLOPS 오기재 | `memory_bw_GBs: 288`, `compute_fp16_tflops: 200` 으로 수정 |
| W-2 | `policy_layer_wise.py` | SM 비율 상수 미검증 (RTX 실측 없음) | Stage 1 A100 결과 기반 재조정 권장 |

---

## 예상 산출물 (논문용)

| Stage | Figure | 내용 |
|-------|--------|------|
| 1 | SM Scaling Curve | SSM vs Attn prefill의 SM 수확체감 지점 비교 |
| 1 | Free SM Zone | (model, seq_len, batch) 조합별 decode에 개방 가능한 SM |
| 1 | SRM Roofline | layer type별 compute vs bandwidth bound 분석 |
| 2 | Decision Matrix | model × device × overhead_ratio × strategy |
| 3 | TTFT–TPOT Trade-off | Policy A/B/C 비교 scatter |
| 3 | SM Utilization Timeline | 동적 재할당의 실제 SM 활용 효과 |
| 3 | Throughput Improvement | Policy B/C의 prefill throughput / Policy A |

논문 §4 (Motivation) + §5 (HM Evaluation) 구성.
