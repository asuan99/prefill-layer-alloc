# prefill-layer-alloc

Hybrid SSM+Attention 모델의 serving에서 **prefill 중 layer type 경계마다 SM(Streaming Multiprocessor)을 동적으로 재할당**하는 것이 실질적 이득을 주는가를 검증하는 실험 프레임워크.

## 연구 질문

1. SSM prefill layer는 몇 개의 SM에서 수확체감이 시작되는가? (SM saturation point)
2. libsmctrl SM 재설정 overhead는 prefill layer 실행 시간 대비 얼마인가?
3. 위 두 결과에 따라 layer-wise / step-level / 고정 분할 중 어느 전략이 최적인가?

## 선행 프로젝트

- `mixer-alloc`: 고정 TPC ratio별 전체 forward pass latency → optimal ratio 탐색
- `mamba-cuda-graph`: decode CUDA Graph + libsmctrl SM isolation → TPOT/TTFT 측정
- **이 프로젝트**: prefill+decode 동시 실행 중 layer 경계 SM 재분배 효과 측정

---

## 실행 방법

### 환경 설정

```bash
pip install -r requirements.txt

# libsmctrl 빌드 (optional — MPS fallback 자동 사용)
git clone https://github.com/msr-fiddle/libsmctrl
cd libsmctrl && make && export LIBSMCTRL_PATH=$(pwd)/libsmctrl.so
```

### Stage 1: SM scaling curve 측정

```bash
# SSM prefill layer SM sweep
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --device auto
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model falcon_h1 --device auto

# Attention prefill layer SM sweep
python stage1_sm_scaling/run_attn_prefill_sweep.py --model zamba2
python stage1_sm_scaling/run_attn_prefill_sweep.py --model falcon_h1

# MLP layer SM sweep
python stage1_sm_scaling/run_mlp_prefill_sweep.py --model zamba2

# 결과 시각화 (results/stage1/*.png)
python stage1_sm_scaling/plot_saturation.py

# HuggingFace 없이 fallback kernel로 실행 (shapes only, 모델 미다운로드)
python stage1_sm_scaling/run_ssm_prefill_sweep.py --model zamba2 --fallback
```

### Stage 2: overhead 측정 및 전략 결정

```bash
# libsmctrl 재설정 latency 측정
python stage2_overhead/measure_smctrl_latency.py

# 비교 기준 layer latency 수집
python stage2_overhead/measure_layer_latency.py --model zamba2
python stage2_overhead/measure_layer_latency.py --model falcon_h1

# 전략 자동 판단 (results/stage2/decision_matrix.{json,html})
python stage2_overhead/compute_decision_matrix.py
```

### Stage 3: 동시 실행 평가

```bash
# Stage 2 완료 후 실행
python stage3_hm_eval/run_concurrent_eval.py --model zamba2 --policy all
python stage3_hm_eval/run_concurrent_eval.py --model falcon_h1 --policy A B

# 논문용 Figure 생성
python stage3_hm_eval/plot_results.py
```

---

## 프로젝트 구조

```
prefill-layer-alloc/
├── configs/
│   ├── models.yaml          # zamba2, falcon_h1 설정 (layer 수, head 수 등)
│   └── hardware.yaml        # GPU별 SM count, BW (A100, H100, RTX 5060Ti 등)
├── src/
│   ├── models/
│   │   ├── layer_runner.py      # LayerRunner: 단일 layer 독립 실행 + SM 제한
│   │   ├── zamba2.py            # Zamba2-1.2B layer 추출 + FallbackSSMKernel
│   │   └── falcon_h1.py         # Falcon-H1-1.5B layer 추출 + FallbackSSMBranch
│   ├── smctrl/
│   │   ├── libsmctrl_wrapper.py # SMController: libsmctrl 또는 MPS fallback
│   │   └── overhead_timer.py    # SMOverheadTimer: 재설정 latency 정밀 측정
│   └── profiling/
│       ├── nvml_monitor.py      # NVMLMonitor: 백그라운드 GPU 메트릭 수집
│       └── metrics.py           # LatencyMeter (CUDA event), BandwidthEstimator
├── stage1_sm_scaling/
│   ├── run_ssm_prefill_sweep.py # SSM layer SM sweep (csv 출력)
│   ├── run_attn_prefill_sweep.py
│   ├── run_mlp_prefill_sweep.py
│   └── plot_saturation.py       # Figure 1 (scaling curve) + Figure 2 (free SM zone)
├── stage2_overhead/
│   ├── measure_smctrl_latency.py  # 재설정 latency (μs) 측정
│   ├── measure_layer_latency.py   # 비교 기준 layer latency
│   └── compute_decision_matrix.py # 전략 판단 (json + html)
├── stage3_hm_eval/
│   ├── policy_baseline.py         # Policy A: 고정 분할 (40/60)
│   ├── policy_step_adaptive.py    # Policy B: step-level 모델 특성 기반
│   ├── policy_layer_wise.py       # Policy C: layer boundary마다 재설정
│   ├── run_concurrent_eval.py     # Prefill+Decode 인터리빙 메인 루프
│   └── plot_results.py            # Figure 1–3 (논문용)
└── results/
    ├── stage1/  # ssm_scaling_*.csv, attn_scaling_*.csv, fig1_*.png, fig2_*.png
    ├── stage2/  # smctrl_overhead_*.json, layer_latency_*.csv, decision_matrix.*
    └── stage3/  # eval_*.csv, sm_timeline_*.csv, fig1–3 *.png
```

---

## SM 제어 백엔드

### libsmctrl (primary)
- kernel-level SM TPC mask via ioctl
- `LIBSMCTRL_PATH` 환경변수 또는 표준 라이브러리 경로에서 로드
- overhead: 수십 μs (단일 전환)

### CUDA MPS (fallback)
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 환경변수
- MPS 데몬이 실행 중일 때 자동 사용
- overhead: libsmctrl보다 크며 Stage 2에서 직접 측정

---

## 예상 산출물 (논문용)

| Stage | Figure | 내용 |
|-------|--------|------|
| 1 | SM Scaling Curve | SSM vs Attn prefill의 SM 수확체감 지점 비교 |
| 1 | Free SM Zone | (model, seq_len, batch) 조합별 decode에 개방 가능한 SM |
| 2 | Decision Matrix | model × device × overhead_ratio × strategy |
| 3 | TTFT–TPOT Trade-off | Policy A/B/C 비교 scatter |
| 3 | SM Utilization Timeline | 동적 재할당의 실제 SM 활용 효과 |
| 3 | Throughput Improvement | Policy B/C의 prefill throughput / Policy A |

논문 §4 (Motivation) + §5 (HM Evaluation) 구성.
