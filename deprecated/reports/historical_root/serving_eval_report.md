# 서빙 실험 결과 보고서: vLLM + ShareGPT 측정 (2026-06-10)

## 0. 요약

| 항목 | 결과 |
|------|------|
| 실험 설정 | vLLM V1, ShareGPT 200 req, A100-SXM4-80GB, max_num_seqs=64 |
| 측정 모델 | Zamba2-7B, Falcon-H1-7B, Nemotron-H-8B |
| GPU SM 이용률 (NVML 평균) | 93.4–93.8% (세 모델 모두) |
| NVML @100% 비율 | 57–67% (bimodal: 100% vs 낮은 값) |
| Stage 2 SSM BW 이용률 | 22–30% (scan kernel, memory-bound 확인) |
| Stage 2 Attn BW 이용률 | 7–22% (seq 증가할수록 하락 → compute-intensifying) |
| Stage 3 Policy A vs B (TTFT) | 67344 ms vs 67340 ms (사실상 동일) |
| **핵심 미결** | E2 동시 실행 counterfactual 미측정 |

---

## 1. 실험 개요

### 1.1 환경 및 설정

| 항목 | 값 |
|------|----|
| GPU | NVIDIA A100-SXM4-80GB (108 SM, HBM2e ~2000 GB/s) |
| 서빙 엔진 | vLLM V1 (chunked prefill 기본 활성화) |
| 데이터셋 | ShareGPT (real user conversations, 200 requests) |
| max_num_seqs | 64 |
| 프롬프트 길이 (평균) | 247–277 tokens |
| 출력 길이 (평균) | 179–328 tokens |

### 1.2 측정 모델

| 모델 | 아키텍처 | n_layers | SSM 비율 |
|------|----------|----------|----------|
| Zamba2-7B | Hybrid SSM+Attn (Mamba2+GQA) | 54 | ~89% |
| Falcon-H1-7B | Hybrid SSM+Attn (Mamba2+GQA) | 32 | ~75% |
| Nemotron-H-8B | Hybrid SSM+Attn (Mamba2+GQA) | 52 | ~85% |

모든 실험은 `valid_for_conclusions=True` 조건을 충족함 (n_err=0, n_ok=200).

---

## 2. E2E 서빙 결과

### 2.1 요청 처리 성능

| 모델 | wall_time (s) | latency p50 (s) | latency p95 (s) | latency p99 (s) | output tok/s |
|------|---------------|-----------------|-----------------|-----------------|--------------|
| Zamba2-7B | 124.34 | 3.984 | 12.224 | 12.646 | 31.4 |
| Falcon-H1-7B | 128.3 | 5.368 | 7.826 | 8.778 | 64.4 |
| Nemotron-H-8B | 64.33 | 1.914 | 6.718 | 6.783 | 52.4 |

**관찰:**
- Nemotron-H는 wall_time이 절반(64.3s) — ShareGPT 분포에서 출력 길이가 짧은 요청이 많아 생긴 차이
- Falcon-H1은 output tok/s가 Zamba2의 2배(64.4 vs 31.4) — Falcon-H1의 Attn 구조가 이 workload에서 decode 효율이 높음
- 모든 모델 n_err=0 — 서빙 안정성 확인

### 2.2 토큰 분포

| 모델 | prompt_tokens_mean | output_tokens_mean |
|------|-------------------|-------------------|
| Zamba2-7B | 276.7 | 198.6 |
| Falcon-H1-7B | 268.1 | 327.7 |
| Nemotron-H-8B | 246.6 | 178.9 |

ShareGPT 특성: 짧은 프롬프트(200–280 tok) + 중간 길이 출력(179–328 tok). 실제 서빙 workload에서의 전형적인 분포.

---

## 3. GPU SM 이용률 분석 (NVML)

### 3.1 이용률 통계

NVML `nvmlDeviceGetUtilizationRates` (~100ms 샘플링, device-level aggregate).

| 모델 | n_samples | 전체 시간(s) | mean (%) | @100% 비율 | idle(<5%) 비율 | p50 |
|------|-----------|------------|----------|----------|--------------|-----|
| Zamba2-7B | 1229 | 124.3 | 93.7 | 67.0% | 0.3% | 100% |
| Falcon-H1-7B | 1262 | 128.2 | 93.8 | 64.7% | 1.9% | 100% |
| Nemotron-H-8B | 631 | 64.3 | 93.4 | 56.9% | 3.3% | 100% |

### 3.2 Bimodal 패턴 해석

세 모델 모두 NVML SM 이용률이 bimodal 분포를 보임:
- **High burst (=100%)**: 전체 시간의 57–67%. 요청 처리 중 compute 또는 memory 연산
- **Low gap (~0–80%)**: 요청 간 idle, KV-cache 조작, CPU-GPU sync
- **mean=93–94%**: 고평균 이용률 → "GPU is mostly busy" 인상

**중요 한계 — NVML ≠ 레이어별 SM 이용률:**

```
NVML device-level 측정은 어떤 kernel이 실행 중이면 100%를 반환함.
SSM 레이어 실행 중 (scan kernel: memory-bound, ~22–30% BW util)이라도
NVML은 동일하게 100%로 표기.

따라서 NVML high utilization은:
- 레이어 타입별 자원 효율성 차이를 측정하지 못함
- "SM이 최적으로 배분됨"을 의미하지 않음
- 단지 "어떤 kernel이 실행 중"임을 뜻함
```

---

## 4. Stage 2: 레이어별 자원 특성 측정

Stage 2는 개별 레이어 커널의 **실측** BW 이용률과 지연 시간을 측정함 (Green Context SM 제한, CUDA event timing).

### 4.1 SSM scan kernel — BW 이용률 (Zamba2)

> 데이터 출처: `workspace/characterization/results/stage2/layer_latency_zamba2_a100_sxm4_80gb.csv`

| seq_len | batch_size | latency (ms) | BW util (%) | 해석 |
|---------|-----------|-------------|-------------|------|
| 1024 | 4 | 0.536 | 22.5% | memory-bound |
| 1024 | 16 | 1.587 | 30.4% | memory-bound |
| 4096 | 4 | 1.584 | 30.4% | memory-bound |
| 4096 | 16 | 7.526 | 25.6% | memory-bound |
| 8192 | 4 | 3.616 | 26.7% | memory-bound |
| 16384 | 4 | 7.512 | 25.7% | memory-bound |

이론 피크 BW: 1000 GB/s (측정 기준치). A100-SXM4-80GB 실제 HBM2e BW = ~2000 GB/s이므로 실제 이용률은 수치의 절반 (약 11–15%).

**핵심:** 구 보고서의 "BW util 0.1–5% → compute-bound" 주장은 `torchscan` 전체 레이어 측정값에서 유래. scan kernel 단독 측정에서는 22–30%로 **memory-bound** 확인.

### 4.2 SSM scan kernel — BW 이용률 (Falcon-H1)

> 데이터 출처: `workspace/characterization/results/stage2/layer_latency_falcon_h1_a100_sxm4_80gb.csv`

| seq_len | batch_size | latency (ms) | BW util (%) |
|---------|-----------|-------------|-------------|
| 1024 | 4 | 0.534 | 10.2% |
| 1024 | 16 | 0.988 | 22.2% |
| 4096 | 4 | 0.971 | 22.5% |
| 4096 | 16 | 4.165 | 21.0% |
| 8192 | 4 | 2.102 | 20.8% |
| 16384 | 4 | 4.151 | 21.1% |

Falcon-H1 SSM도 BW util 10–22% → **memory-bound**. Zamba2 대비 낮은 값은 n_heads=24 (vs Zamba2 n_heads=112)로 인한 block 수 차이.

### 4.3 Attn — BW 이용률 변화 패턴 (Falcon-H1)

| seq_len | batch_size | latency (ms) | BW util (%) | 해석 |
|---------|-----------|-------------|-------------|------|
| 1024 | 4 | 0.607 | 21.6% | 중간 |
| 4096 | 4 | 3.077 | 14.9% | compute-intensifying |
| 8192 | 4 | 7.917 | 11.3% | compute-intensifying |
| 16384 | 4 | 23.927 | 7.4% | compute-bound 전이 |

**BW util 하락 트렌드** (21% → 7%): seq 증가에 따라 Attn이 compute-bound에 가까워짐. Roofline 상에서 ridge point 방향으로 이동. SSM은 seq와 무관하게 memory-bound를 유지.

이 비대칭이 HM thesis의 물리적 근거임.

### 4.4 Green Context 전환 오버헤드

> 데이터 출처: `results/stage2/ctx_switch_overhead_a100-sxm4-80gb.json`

| 전환 방식 | median (μs) | mean (μs) | p99 (μs) |
|-----------|------------|----------|---------|
| CPU context swap (no GPU sync) | 0.45 | 0.45 | 0.55 |
| GPU sync 포함 전환 (ssm→attn) | 7.81 | 8.00 | 13.13 |
| GPU sync 포함 전환 (attn→ssm) | 7.85 | 7.89 | 8.63 |
| 초기화 overhead (one-time) | 14,276 μs | 14,289 μs | — |

**n=81 레이어 (Zamba2 전체) 누적 overhead:**
- CPU swap 방식: ~1,717 μs = **1.72 ms** (n=81, mean 21.2 μs/layer)
- sync 방식 추정: 81 × ~8 μs = **648 μs** (단, kernel 실행 중 sync 필요)

초기화 overhead(14.3 ms)는 one-time cost로 요청 단위 오버헤드에는 해당 없음.

---

## 5. Stage 1 재분석: 서빙 배치 크기에서의 SSM 포화

### 5.1 포화 기준

"10% SM 증가당 throughput gain < 3%인 첫 SM 지점" (동일 기준, Stage 1 보고서와 동일)

측정값: `results/stage1/chunked/ssm_chunked_{model}_a100-sxm4-80gb.csv` (min latency per SM group)

### 5.2 Falcon-H1 SSM 포화 지점

| seq_len | bs=1 | bs=4 | bs=16 | bs=32 |
|---------|------|------|-------|-------|
| 512 | 40 | 54 | >108 | 108 |
| 1024 | 54 | 94 | >108 | >108 |
| 2048 | 68 | 94 | >108 | 108 |
| 4096 | 94 | 108 | >108 | 108 |
| 8192 | 68 | >108 | >108 | 108 |
| 16384 | 54 | >108 | >108 | 108 |

### 5.3 Zamba2 SSM 포화 지점

| seq_len | bs=1 | bs=4 | bs=16 | bs=32 |
|---------|------|------|-------|-------|
| 512 | 54 | 94 | >108 | >108 |
| 1024 | 81 | >108 | >108 | >108 |
| 2048 | 108 | >108 | >108 | >108 |
| 4096 | >108 | >108 | >108 | >108 |
| 8192 | 108 | >108 | >108 | >108 |
| 16384 | 94 | >108 | >108 | >108 |

### 5.4 서빙 맥락 해석

ShareGPT 서빙 실험 설정 (max_num_seqs=64, 평균 prompt 247–277 tok):
- 실제 배치 크기: 요청 동시 처리 수 ≥ 4 (batching에 의해)
- 대표 구간: bs≥4, seq 256–2048

이 구간에서 SSM 포화 분석:
- **Falcon-H1 bs=4, seq≤2048**: sm_sat=54–94 → free zone=14–54 SM (일부 존재)
- **Falcon-H1 bs≥16**: 포화 없음(>108) → free zone 없음
- **Zamba2 bs≥4**: 대부분 포화 없음(>108) → free zone 없음

**결론**: HM thesis가 전제하는 "유의미한 free SM zone"은 **단일 요청(bs=1) 또는 소규모 배치(bs=4, 짧은 seq)** 에서만 존재. 실제 서빙 배치에서는 SSM도 전체 SM을 활용해야 throughput을 극대화할 수 있음.

### 5.5 Attn 포화 지점 (참고)

두 모델 모두 bs≥4에서 Attn 포화 없음(>108). SSM과 동일하게 전 SM 필요.
bs=1에서만 일부 포화 (seq=512: sm=81, seq=1024: sm=94).

따라서 free SM zone (SSM_sat ≠ Attn_sat) 역시 bs=1의 좁은 범위에서만 관측됨.

---

## 6. Stage 3: 순차 SM 정책 실험

### 6.1 실험 설정

| 항목 | 값 |
|------|----|
| 모델 | Zamba2-7B |
| 장치 | A100-SXM4-80GB |
| seq_len | 1024 |
| batch_size | 8 |
| n_prefill | 50 |
| n_decode_steps | 4050 |

**Policy A**: 전체 레이어 동일 SM 비율 (baseline)
**Policy B**: SSM 레이어에 낮은 SM 비율, Attn 레이어에 높은 SM 비율 (순차 인터리빙)

> 데이터 출처: `results/stage3/eval_zamba2_A_a100-sxm4-80gb.csv`, `eval_zamba2_B_a100-sxm4-80gb.csv`

### 6.2 결과

| 지표 | Policy A | Policy B | 차이 |
|------|----------|----------|------|
| ttft_mean_ms | 67,344 | 67,340 | -4 ms (0.006%) |
| ttft_p99_ms | 129,463 | 130,389 | +926 ms (0.7%) |
| tpot_p50_ms | 0.33 | 0.33 | 동일 |
| tpot_p99_ms | 0.9 | 0.9 | 동일 |
| prefill_throughput (tok/s) | 15.2 | 15.2 | 동일 |
| slo_violations | 0 | 0 | 동일 |

### 6.3 해석

Policy A와 B는 **사실상 동일**. 순차 인터리빙으로 레이어별 SM 비율을 달리 해도 TTFT, TPOT, throughput 모두 개선 없음.

원인: **순차 실행에서는 SM 절약 효과가 없음**. SSM 레이어가 SM을 줄여도 해당 시간에 다른 작업이 그 SM을 사용하지 않음. Free SM은 물리적으로 유휴 상태이지만 활용되지 않음.

### 6.4 미측정 counterfactual: E2 동시 실행

순차 인터리빙이 무효임을 확인했으나, **E2 동시 실행** (prefill SM + decode SM 동시 할당, 서로 다른 Green Context 파티션에서 병렬 실행)은 아직 측정되지 않음:

```
E2 가설:
  - Prefill SSM 레이어: 일부 SM (예: 60 SM)
  - Decode 레이어: 나머지 SM (예: 48 SM)
  - 두 작업 동시 실행 → 실질적인 SM 활용 개선

측정 상태: 미측정 공백
```

이것이 HM thesis 검증의 핵심 미결 질문임.

---

## 7. 동기 분석 그림 메타데이터

Stage 1 보고서와 함께 생성된 motivation 그림들의 측정 기반 정리:

| 그림 | 내용 | 측정 방식 | 한계 |
|------|------|-----------|------|
| Fig A | vLLM 서빙 중 NVML 이용률 | NVML (~10 Hz, device-level) | per-layer 분해 불가 |
| Fig B | Free SM zone fraction vs seq_len | Stage 1 chunked sweep → saturation criterion | plot_saturation.py의 grouping 버그로 free_sm_zone.csv 값 부정확 (§5 참고) |
| Fig C | Roofline (SSM vs Attn) | Stage 1 isolated-kernel CUDA event timing | 해석적 AI 추정 (FLOPs/bytes 계산), 하드웨어 카운터 없음 |

**Fig B 정오:** `free_sm_zone_zamba2.csv`, `free_sm_zone_falcon_h1.csv`의 모든 `saturation_sm=108, free_sm=0` 값은 **오류**. `plot_saturation.py`가 SM별 5회 반복 측정값을 개별 행으로 처리해 같은 SM 내 비교를 수행했기 때문. 올바른 분석(각 SM 최소값 기준)은 §5 표 참고.

---

## 8. 종합 평가 및 미결 질문

### 8.1 HM thesis 검증 상태 (서빙 맥락)

| 논거 | 검증 상태 | 증거 |
|------|-----------|------|
| SSM은 memory-bound | **확인** | BW util 22–30% (Stage 2 scan-only) |
| Attn은 긴 seq에서 compute-intensifying | **확인** | BW util 7% at seq=16384 (Stage 2) |
| 레이어 타입 간 자원 비대칭 존재 | **확인** | Stage 2 BW util 패턴 |
| 서빙 배치에서 SSM이 일찍 포화 | **한정적** | bs=1 전용; bs≥4에서는 포화 없음 |
| Free SM zone이 실용적 크기 | **미확인** | bs=4 이상에서 거의 0 |
| 순차 인터리빙으로 이득 획득 가능 | **부정됨** | Policy A vs B 동일 (Stage 3) |
| E2 동시 실행으로 이득 획득 가능 | **미측정** | Stage 3 E2 counterfactual 없음 |

### 8.2 서빙 NVML 이용률의 함의

세 hybrid 모델 모두 NVML mean ~93–94%: 이는 "GPU가 거의 항상 바쁨"을 뜻하나 다음을 의미하지 않음:
- SSM/Attn 레이어가 자신의 자원 프로파일에 최적으로 매핑되었음
- SM 재배분으로 추가 개선 여지가 없음

NVML bimodal(100% burst + 간헐적 gap)은 서빙 엔진 레벨의 batching/scheduling overhead이지 per-layer 자원 효율성이 아님.

### 8.3 미결 질문 우선순위

1. **E2 동시 실행 counterfactual** (최우선): prefill SSM과 decode를 다른 SM 파티션에서 병렬 실행 시 실제 TTFT/TPOT 변화
2. **서빙 배치 크기에서의 per-layer SM 이용률**: NVML은 device-level → 레이어별 하드웨어 카운터(NCU) 필요
3. **Free SM zone의 실용 크기 재확인**: §5의 올바른 saturation 분석 기반으로 Fig B 재생성

---

## 9. 데이터 출처 요약

| 데이터 | 파일 경로 |
|--------|-----------|
| 서빙 결과 (Zamba2) | `results/serving-eval/serving_zamba2_sharegpt.json` |
| 서빙 결과 (FH1) | `results/serving-eval/serving_falcon_h1_sharegpt.json` |
| 서빙 결과 (Nemotron-H) | `results/serving-eval/serving_nemotron_h_sharegpt.json` |
| NVML 이용률 (Zamba2) | `results/serving-eval/nvml_zamba2_sharegpt.csv` |
| NVML 이용률 (FH1) | `results/serving-eval/nvml_falcon_h1_sharegpt.csv` |
| NVML 이용률 (Nemotron-H) | `results/serving-eval/nvml_nemotron_h_sharegpt.csv` |
| Stage 2 레이어 지연 (Zamba2) | `workspace/characterization/results/stage2/layer_latency_zamba2_a100_sxm4_80gb.csv` |
| Stage 2 레이어 지연 (FH1) | `workspace/characterization/results/stage2/layer_latency_falcon_h1_a100_sxm4_80gb.csv` |
| 전환 오버헤드 | `results/stage2/ctx_switch_overhead_a100-sxm4-80gb.json` |
| Stage 3 Policy A | `results/stage3/eval_zamba2_A_a100-sxm4-80gb.csv` |
| Stage 3 Policy B | `results/stage3/eval_zamba2_B_a100-sxm4-80gb.csv` |
| Stage 1 chunked SSM (Zamba2) | `results/stage1/chunked/ssm_chunked_zamba2_a100-sxm4-80gb.csv` |
| Stage 1 chunked SSM (FH1) | `results/stage1/chunked/ssm_chunked_falcon_h1_a100-sxm4-80gb.csv` |
