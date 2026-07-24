# Stage 1 정정 측정 vs HM Thesis 검정 보고서

> 작성일: 2026-06-09  
> 하드웨어: NVIDIA A100-SXM4-80GB (108 SM, HBM2e 2,000 GB/s)  
> 대상 모델: Zamba2-7B-Instruct, Falcon-H1-7B-Instruct  
> 데이터 출처: `results/stage1/chunked/ssm_chunked_*_a100-sxm4-80gb.csv` (primary SSM),  
> `workspace/characterization/results/stage1/attn_scaling_*_a100_sxm4_80gb.csv` (Attn)

---

## 목차

1. [정정 요약 — 무엇이 왜 바뀌었나](#1-정정-요약)
2. [SSM / Attn saturation 실측](#2-ssm--attn-saturation-실측)
3. [비대칭 검정 결론 — HM thesis 성립 / 위험 판정](#3-비대칭-검정-결론)
4. [Marginal curve — dL/dSM 표 및 교차점](#4-marginal-curve)
5. [기존 결론 정정](#5-기존-결론-정정)
6. [남은 공백 — 미측정 항목](#6-남은-공백)

---

## 1. 정정 요약

### 1.1 버그 체인 — 세 층의 오류

기존 보고서(`deprecated_reports/problem_redefinition.md` 등)의 SSM 수치는 아래 세 가지
오류가 겹쳐서 발생했다.

| 층 | 오류 내용 | 영향 |
|----|-----------|------|
| **측정 대상 오류** | scan kernel(`mamba_chunk_scan_combined`) 대신 full SSM layer (in_proj + scan + out_proj GEMM)의 torchscan latency를 기준값으로 사용 | latency 과대 측정 (~6–7×) |
| **n_blocks 공식 오류** | `n_blocks = batch × seq / 4` (model-agnostic) 사용 → Zamba2: 1.75× 과소 추정, Falcon-H1: 2.67× 과대 추정 | wave model 예측 방향이 반대 |
| **실측 없음** | sm=108 torchscan 1회 수치를 base로 잡고 나머지 SM 값은 전부 wave model 예측치로 채움 | "실측 확인" 없이 결론 도달 |

**정정된 측정 방식**: `mamba_chunk_scan_combined` 커널만 직접 호출하는
`run_chunked_ssm_sweep.py`를 사용한 chunked prefill (host-level 분할로 cooperative barrier 우회).

### 1.2 대표 config 비교 (seq=4096, bs=4, sm=108)

| 구분 | 출처 | latency | bw_util | n_blocks |
|------|------|---------|---------|----------|
| Stale "측정값" (실제: torchscan full-layer) | `ssm_scaling_*_torchscan.csv` | **11.91 ms** | 3.27% | 7,168 (실측) |
| Stale 예측 sm=14 (wave model) | analytical 계산 | **~97.1 ms** | — | 4,096 (오류 공식) |
| **Corrected: scan-only @ sm=14** | `ssm_chunked_zamba2_*.csv` | **10.517 ms** | N/A | 7,168 |
| **Corrected: scan-only @ sm=108** | `ssm_chunked_zamba2_*.csv` | **1.820 ms** | N/A | 7,168 |

핵심: stale 보고서가 "sm=108 측정값 12.6ms"로 제시한 수치는 full-layer torchscan의 실측값
(11.913ms)이었으며, scan-only 실측값(1.820ms)과 6.5× 차이가 난다.

### 1.3 n_blocks 공식 비교

| 모델 | 오류 공식 | 정정 공식 | n_blocks @ (seq=4096, bs=4) | 비율 |
|------|----------|----------|------------------------------|------|
| Zamba2 (n_heads=112) | `batch × seq / 4` | `batch × (seq/chunk) × n_heads` | 4,096 → 7,168 | +1.75× |
| Falcon-H1 (n_heads=24) | `batch × seq / 4` | `batch × (seq/chunk) × n_heads` | 4,096 → 1,536 | −0.375× |

- Zamba2: n_blocks를 1.75× 과소 추정 → 실제보다 파도 수가 많아 포화가 더 어려워야 함 (wave model 예측 방향 일치했으나 base latency가 잘못됨)
- Falcon-H1: n_blocks를 2.67× 과대 추정 → stale model은 Falcon-H1이 포화 없다고 예측했으나, 실제는 n_blocks가 훨씬 적어 **sm=81에서 포화** (결론 방향 반전)

### 1.4 BW utilization — compute-bound 주장의 출처

stale 보고서(`ssm_sm_partitioning_analysis.md`)의 "BW util 0.1~5% → compute-bound" 수치는
torchscan full-layer SSM의 실측값에서 도출되었다.

| 출처 | seq | bs | BW util @ sm=14 | 해석 |
|------|-----|----|-----------------|------|
| torchscan (full-layer, 실제 측정) | 512 | 1 | 6.26% | GEMM 작아 memory-bound |
| torchscan (full-layer, 실제 측정) | 4096 | 4 | 0.49% | GEMM 커져 compute-bound |
| torchscan (full-layer, 실제 측정) | 8192 | 32 | 0.31% | GEMM 극단적 compute-bound |

full-layer에서 BW util이 낮은 이유: in_proj / out_proj GEMM이 large matrix에서 compute-bound.
**scan kernel 단독**은 AI ≈ 31 < ridge 156 (A100-SXM4 기준), 즉 memory-bound이며
BW util은 이보다 유의미하게 높아야 한다. chunked CSV에는 bw_utilization_pct 컬럼이 없어
직접 비교 불가 → **미측정 공백** (§6 참조).

---

## 2. SSM / Attn Saturation 실측

### 2.1 데이터 품질

| 파일 | 총 row | cooperative_safe=False | latency=NaN | 분석 대상 |
|------|--------|------------------------|-------------|----------|
| `ssm_chunked_zamba2_a100-sxm4-80gb.csv` | 768 | 768 (all) | 0 | 768 |
| `ssm_chunked_falcon_h1_a100-sxm4-80gb.csv` | 768 | 685 | 0 | 768 |
| `attn_scaling_zamba2_a100_sxm4_80gb.csv` | 192 | N/A | 0 | 184 (8 OOM 제외: seq=16384, bs=32) |
| `attn_scaling_falcon_h1_a100_sxm4_80gb.csv` | 192 | N/A | 0 | 192 |

**cooperative_safe 처리 주의**: chunked CSV는 모든 kernel call을 host-level로 분할하는 방식으로
cooperative barrier를 우회하므로 cooperative_safe=False가 측정 오류를 의미하지 않는다.
chunked 측정에서 이 컬럼은 "해당 config가 단일 cooperative call로도 실행 가능한지"를 나타낼 뿐이다.
Zamba2는 n_heads=112 > 108 SM이므로 전 config에서 False(예상값). Falcon-H1(n_heads=24)은
작은 batch/SM에서만 True(83개).

**chunk 선택**: (seq, bs, sm) 조합당 가장 큰 prefill_chunk_tokens(= kernel call 수 최소화)를
primary 값으로 사용.

### 2.2 포화 판정 기준

> **sm_sat**: "10% SM 증가당 throughput gain < 3%인 첫 SM 값" (`plot_saturation.py`와 동일 기준)
> throughput = base_lat / lat, gain normalized to per-10%-SM-increase

### 2.3 Zamba2 SSM saturation (scan-only, 정정 측정)

| (seq, bs) | sm_sat | lat@14SM | lat@108SM | 14→108 speedup |
|-----------|--------|---------|---------|---------------|
| (512, 1) | **14** | 0.830 ms | 0.600 ms | 1.38× |
| (512, 4) | **54** | 1.645 ms | 0.705 ms | 2.33× |
| (512, 16) | ≥108 | 5.435 ms | 1.143 ms | 4.75× |
| (512, 32) | ≥108 | 10.659 ms | 1.872 ms | 5.69× |
| (1024, 1) | **27** | 1.105 ms | 0.632 ms | 1.75× |
| (1024, 4) | ≥108 | 2.835 ms | 0.850 ms | 3.34× |
| (1024, 16) | ≥108 | 10.575 ms | 1.839 ms | 5.75× |
| (1024, 32) | **94** | 21.180 ms | 3.856 ms | 5.49× |
| (2048, 1) | **68** | 1.649 ms | 0.703 ms | 2.35× |
| (2048, 4) | ≥108 | 5.381 ms | 1.141 ms | 4.72× |
| (2048, 16) | **94** | 21.109 ms | 3.826 ms | 5.52× |
| (2048, 32) | **94** | 42.038 ms | 7.705 ms | 5.46× |
| (4096, 1) | ≥108 | 2.824 ms | 0.852 ms | 3.31× |
| (4096, 4) | ≥108 | 10.517 ms | 1.820 ms | 5.78× |
| (4096, 16) | **94** | 41.961 ms | 7.699 ms | 5.45× |
| (4096, 32) | **94** | 83.787 ms | 15.887 ms | 5.27× |
| (8192, 1) | **81** | 5.401 ms | 1.354 ms | 3.99× |
| (8192, 4) | ≥108 | 20.842 ms | 3.415 ms | 6.10× |
| (8192, 16) | **94** | 83.764 ms | 15.239 ms | 5.50× |
| (8192, 32) | **94** | 167.400 ms | 31.645 ms | 5.29× |
| (16384, 1) | **81** | 10.575 ms | 2.361 ms | 4.48× |
| (16384, 4) | ≥108 | 41.489 ms | 6.584 ms | 6.30× |
| (16384, 16) | **94** | 167.360 ms | 30.306 ms | 5.52× |
| (16384, 32) | **94** | 334.651 ms | 63.135 ms | 5.30× |

패턴: **bs=1은 대부분 조기 포화**(14~68 SM), **bs=4는 대부분 ≥108**(미포화),
**bs≥16은 sm_sat=94** (seq≥1024 이상 대부분 일관).

### 2.4 Zamba2 Attn saturation

| (seq, bs) | sm_sat | lat@14SM | lat@108SM | bw_util@108SM |
|-----------|--------|---------|---------|--------------|
| (512, 1) | **68** | 6.285 ms | 1.276 ms | 30.49% |
| (512, 4) | ≥108 | 24.754 ms | 4.055 ms | 23.17% |
| (512, 16) | **81** | 98.146 ms | 15.108 ms | 20.79% |
| (512, 32) | ≥108 | 196.216 ms | 30.320 ms | 20.04% |
| (1024, 1) | **81** | 10.553 ms | 1.871 ms | 24.32% |
| (1024, 4) | ≥108 | 41.592 ms | 6.329 ms | 19.02% |
| (1024, 16) | ≥108 | 166.094 ms | 24.429 ms | 17.19% |
| (1024, 32) | ≥108 | 331.992 ms | 48.845 ms | 16.77% |
| (2048, 1) | ≥108 | 21.067 ms | 3.312 ms | 17.73% |
| (2048, 4) | ≥108 | 83.246 ms | 12.083 ms | 14.34% |
| (2048, 16) | ≥108 | 332.506 ms | 47.705 ms | 13.23% |
| (2048, 32) | ≥108 | 664.799 ms | 95.289 ms | 13.03% |
| (4096, 1) | ≥108 | 49.502 ms | 7.517 ms | 11.33% |
| (4096, 4) | ≥108 | 197.415 ms | 28.798 ms | 9.69% |
| (4096, 16) | ≥108 | 788.929 ms | 114.996 ms | 9.17% |
| (4096, 32) | ≥108 | 1577.522 ms | 228.543 ms | 9.13% |
| (8192, 1) | ≥108 | 138.003 ms | 20.494 ms | 6.73% |
| (8192, 4) | ≥108 | 551.019 ms | 80.993 ms | 6.05% |
| (8192, 16) | ≥108 | 2203.419 ms | 322.884 ms | 5.88% |
| (8192, 32) | ≥108 | 4328.631 ms | 627.961 ms | 6.02% |
| (16384, 1) | ≥108 | 439.799 ms | 67.103 ms | 3.63% |
| (16384, 4) | ≥108 | 1757.310 ms | 268.487 ms | 3.40% |
| (16384, 16) | ≥108 | 6958.654 ms | 1058.871 ms | 3.39% |

참고: (16384, 32)는 OOM으로 측정 불가 (8개 row 전 SM 제외됨).

### 2.5 Falcon-H1 SSM saturation (scan-only, 정정 측정)

| (seq, bs) | sm_sat | lat@14SM | lat@108SM | 14→108 speedup |
|-----------|--------|---------|---------|---------------|
| (512, 1) | **14** | 0.659 ms | 0.594 ms | 1.11× |
| (512, 4) | **14** | 0.946 ms | 0.620 ms | 1.53× |
| (1024, 4) | **27** | 1.578 ms | 0.667 ms | 2.37× |
| (2048, 4) | **81** | 2.884 ms | 0.773 ms | 3.73× |
| (4096, 4) | **81** | 5.420 ms | 1.217 ms | 4.45× |
| (4096, 16) | **81** | 20.857 ms | 4.394 ms | 4.75× |
| (4096, 32) | **81** | 41.482 ms | 8.509 ms | 4.88× |
| (8192, 4) | **94** | 10.504 ms | 2.191 ms | 4.79× |

Falcon-H1 패턴: n_heads=24로 n_blocks가 Zamba2 대비 0.375× → 전형적 서빙 config에서
**sm=81이 일관된 포화점**. 3×24=72 blocks/seq이므로 108 SMs 대비 wave 수가 적어 조기 포화.

### 2.6 Falcon-H1 Attn saturation

| (seq, bs) | sm_sat | lat@14SM | lat@108SM | bw_util@108SM |
|-----------|--------|---------|---------|--------------|
| (512, 1) | **40** | 0.983 ms | 0.262 ms | 15.20% |
| (512, 4) | ≥108 | 3.458 ms | 0.660 ms | 14.13% |
| (1024, 4) | ≥108 | 5.697 ms | 0.902 ms | 16.39% |
| (2048, 4) | ≥108 | 11.415 ms | 1.766 ms | 14.54% |
| (4096, 4) | ≥108 | 27.948 ms | 4.190 ms | 11.34% |
| (4096, 16) | ≥108 | 110.817 ms | 15.667 ms | 11.71% |
| (4096, 32) | ≥108 | 221.416 ms | 31.298 ms | 11.65% |
| (8192, 4) | ≥108 | 81.361 ms | 11.611 ms | 7.85% |

---

## 3. 비대칭 검정 결론

### 3.1 판정 기준 (HM argument chain §C·D)

- **HM thesis 성립 조건**: `sm_sat_ssm` ≪ `sm_sat_attn` → free SM zone 실재 → 비대칭 배분으로 이득
- **HM thesis 위험 조건**: 두 값이 수렴 (둘 다 낮거나 둘 다 ≥108)

### 3.2 비교 표 — 대표 configs

| 모델 | (seq, bs) | sm_sat_SSM | sm_sat_Attn | 비대칭 gap | 판정 |
|------|-----------|-----------|------------|-----------|------|
| Zamba2 | (4096, 1) | ≥108 | ≥108 | 0 | ✗ thesis 무효 |
| Zamba2 | (4096, 4) | ≥108 | ≥108 | 0 | ✗ thesis 무효 |
| Zamba2 | (4096, 16) | **94** | ≥108 | **≥14 SM** | △ thesis 조건부 성립 |
| Zamba2 | (4096, 32) | **94** | ≥108 | **≥14 SM** | △ thesis 조건부 성립 |
| Falcon-H1 | (4096, 4) | **81** | ≥108 | **≥27 SM** | ✓ thesis 성립 |
| Falcon-H1 | (4096, 16) | **81** | ≥108 | **≥27 SM** | ✓ thesis 성립 |
| Falcon-H1 | (4096, 32) | **81** | ≥108 | **≥27 SM** | ✓ thesis 성립 |

### 3.3 판정 문장

> **Falcon-H1: HM thesis 성립.**  
> scan-only 재측정에서 sm_sat_ssm = 81, sm_sat_attn = ≥108 (전형적 서빙 config seq≥2048, bs≥4 기준).
> 81→108 구간 27 SM(25% of total)에서 SSM은 수확체감 진입, Attn은 여전히 +6–7%/10%SM 이득.
> 이 비대칭은 Attn에 추가 SM을 할당함으로써 전체 prefill latency를 단축할 수 있는
> **구조적 free SM zone**의 실재를 의미한다.

> **Zamba2: HM thesis 조건부 성립 (bs≥16 한정).**  
> bs=4 이하에서는 SSM도 포화 없음(≥108) → Attn과 동률이라 free zone 없음.
> bs≥16에서는 SSM sm_sat=94, Attn sm_sat=≥108 → 14 SM(13%)의 narrow free zone 존재.
> 이 14 SM 구간의 SSM 한계 이득(94→108: +1.8%/10%SM)과 Attn 한계 이득(+7.3%/10%SM)의
> **4× 비대칭**이 재분배 근거가 된다. 단 여백이 좁아 실용적 이득은 Falcon-H1보다 작다.

---

## 4. Marginal Curve

marginal gain = (throughput_curr - throughput_prev) / throughput_prev × 100 / (ΔSM/SM_prev×10),
즉 10% SM 증가당 throughput 증가율 (%). `*SAT` = 3% 임계값 하회(포화 진입).

### 4.1 Zamba2 (seq=4096, bs=4) — free zone 없음

| SM | SSM lat | dTput_SSM/10%SM | Attn lat | dTput_Attn/10%SM |
|----|---------|----------------|---------|-----------------|
| 14 | 10.517 ms | (base) | 197.415 ms | (base) |
| 27 | 5.447 ms | +10.0% | 99.513 ms | +10.6% |
| 40 | 3.931 ms | +8.0% | 69.998 ms | +8.8% |
| 54 | 3.032 ms | +8.5% | 52.184 ms | +9.8% |
| 68 | 2.514 ms | +7.9% | 42.002 ms | +9.3% |
| 81 | 2.180 ms | +8.0% | 35.459 ms | +9.7% |
| 94 | 1.984 ms | +6.1% | 31.895 ms | +7.0% |
| 108 | 1.820 ms | +6.1% | 28.798 ms | +7.2% |

→ SSM marginal gain < Attn marginal gain 전 구간에서 성립하지만, 두 곡선 모두 >3% 유지.
교차점 없음 (SSM이 항상 낮고 Attn이 항상 높으나, 둘 다 미포화).

### 4.2 Zamba2 (seq=4096, bs=16) — narrow free zone at sm=94

| SM | SSM lat | dTput_SSM/10%SM | Attn lat | dTput_Attn/10%SM |
|----|---------|----------------|---------|-----------------|
| 14 | 41.961 ms | (base) | 788.929 ms | (base) |
| 27 | 21.290 ms | +10.5% | 397.419 ms | +10.6% |
| 40 | 15.141 ms | +8.4% | 279.047 ms | +8.8% |
| 54 | 11.525 ms | +9.0% | 207.038 ms | +9.9% |
| 68 | 9.511 ms | +8.2% | 166.695 ms | +9.3% |
| 81 | 8.331 ms | +7.4% | 142.424 ms | +8.9% |
| 94 | 7.911 ms | +3.3% | 127.456 ms | +7.3% |
| 108 | 7.699 ms | **+1.8% \*SAT** | 114.996 ms | +7.3% |

→ **sm=94→108 구간**: SSM +1.8%/10%SM vs Attn +7.3%/10%SM → **4.1× 비대칭**.
SSM이 94에서 포화 진입, Attn는 108에서도 여전히 이득. 이 구간의 SM을 SSM→Attn 재분배하면
전체 bottleneck latency 단축 가능.

### 4.3 Falcon-H1 (seq=4096, bs=4) — clear free zone at sm=81

| SM | SSM lat | dTput_SSM/10%SM | Attn lat | dTput_Attn/10%SM |
|----|---------|----------------|---------|-----------------|
| 14 | 5.420 ms | (base) | 27.948 ms | (base) |
| 27 | 2.875 ms | +9.5% | 14.157 ms | +10.5% |
| 40 | 2.143 ms | +7.1% | 10.103 ms | +8.3% |
| 54 | 1.711 ms | +7.2% | 7.576 ms | +9.5% |
| 68 | 1.467 ms | +6.4% | 6.035 ms | +9.8% |
| 81 | 1.312 ms | +6.2% | 5.110 ms | +9.5% |
| 94 | 1.255 ms | **+2.8% \*SAT** | 4.606 ms | +6.8% |
| 108 | 1.217 ms | **+2.1% \*SAT** | 4.190 ms | +6.7% |

→ **sm=81→94 구간**: SSM +2.8%/10%SM vs Attn +6.8%/10%SM → **2.4× 비대칭**.
→ **sm=94→108 구간**: SSM +2.1%/10%SM vs Attn +6.7%/10%SM → **3.2× 비대칭 (심화)**.
SSM의 marginal 곡선이 81에서 3% 임계 이하로 하락하는 반면, Attn은 108까지 6–7% 유지.
**교차점**: SSM 곡선과 Attn 곡선의 marginal gain이 교차하는 지점은 측정 범위(14–81 SM)
내에서는 확인되지 않음 — 전 구간에서 Attn marginal > SSM marginal. 포화(비선형 꺾임)는
SSM에서만 발생.

### 4.4 비대칭 요약

| 구간 | Zamba2 (4096,4) SSM:Attn | Zamba2 (4096,16) SSM:Attn | FH1 (4096,4) SSM:Attn |
|------|--------------------------|---------------------------|----------------------|
| 81→94 | 7.9% : 7.0% (역전없음) | 3.3% : 7.3% (**2.2×**) | 2.8% : 6.8% (**2.4×**) |
| 94→108 | 6.1% : 7.2% | **1.8% : 7.3% (4.1×)** | **2.1% : 6.7% (3.2×)** |

---

## 5. 기존 결론 정정

### 5.1 `deprecated_reports/problem_redefinition.md` 정정

**기존 결론**: "SSM은 108 SM 범위에서 포화되지 않는다 — Wave model이 정확히 성립."

**정정**: 이 결론은 scan-only latency를 측정하지 않고, torchscan full-layer(GEMM 포함) latency
(11.91ms at sm=108)를 base로 wave model 예측치를 "측정값"으로 제시한 결과다. 또한 n_blocks
공식(batch×seq/4)이 Zamba2에 대해 1.75× 과소 추정이어서 wave 수가 적게 계산됐다.

정정 후:
- Zamba2: bs≤4에서는 ≥108 미포화 유지 (stale 결론과 우연히 일치, 단 근거가 달라짐)
- Zamba2: **bs≥16에서는 sm_sat=94** — stale 보고서의 "포화 없음"은 틀림
- Falcon-H1: **sm_sat=81 (bs≥4)** — stale 보고서의 "Falcon-H1 동일 ×7.71 선형" 주장은 완전히 틀림;
  실제 n_blocks가 2.67× 적어 조기 포화함

**기존 결론**: "SSM과 Attn 사이에 SM 효율성 차이가 뚜렷하게 존재하지 않아, 분할의 이득이 없다."

**정정**: Falcon-H1에서 sm_sat_ssm=81 vs sm_sat_attn=≥108으로 **명확한 비대칭 존재**. 이 비대칭이
layer-type-aware SM 배분(HM thesis)의 근거가 된다. Zamba2도 bs≥16에서 4× marginal 비대칭 확인.

### 5.2 `deprecated_reports/ssm_sm_partitioning_analysis.md` 정정

**기존 결론**: "BW util 0.1~5% → Compute-bound (BW를 전혀 못 씀)"

**정정**: 해당 BW util 수치는 full-layer SSM (in_proj + scan + out_proj GEMM 포함)의 torchscan
측정값이다. large batch에서 GEMMs가 compute-bound이므로 BW util이 낮게 나온 것이지,
scan kernel 자체의 특성이 아니다. scan kernel (`mamba_chunk_scan_combined`)은 AI ≈ 31로
A100 roofline ridge point 156 이하 → **memory-bound**. BW util 직접 측정은 §6 미측정 공백으로 남음.

**기존 결론**: "Triton SSD Kernel의 SM 포화 지점 ~60–80% SM (추정)"

**정정**: Falcon-H1 scan-only 실측에서 sm_sat=81 (75% of 108) — 추정 범위에 우연히 일치.
Zamba2 scan-only: bs=4에서 ≥108, bs=16에서 sm_sat=94 (87%) — 추정보다 높음.
이 값은 이제 추정이 아닌 실측값이다.

### 5.3 `deprecated_reports/project_progress_report.md` 정정

**기존 결론**: "SSM은 SM 수에 선형으로 의존한다 (포화 없음). 레이어별 SM 재분배는 원리적으로 이득이 없다."

**정정**: full-layer 아닌 scan-only 기준으로는 위 2-1~2-3의 포화 패턴이 실재한다.
Falcon-H1의 경우 sm=81 이후 SSM의 한계 이득이 Attn의 한계 이득보다 2–4× 낮아지는
비대칭이 확인되었으며, 이것이 layer-type-aware 재분배의 이득 경로다.

---

## 6. 남은 공백 — 미측정 항목

### 6.1 BW utilization (scan-only, 경로 F)

- **공백**: chunked CSV에 bw_utilization_pct 없음. scan-only가 memory-bound임을 정량 확인 필요.
- **실험 설계**: `src/models/layer_runner.run_ssm_layer(seq=4096, bs=4, sm_count=108, layer_type='ssm')`를
  1회 호출해 achieved_bandwidth_GBs / 2000 GB/s 계산.

### 6.2 E2: Concurrent counterfactual (HM argument chain §E — headline)

- **공백**: 현재 Stage 3는 sequential interleaving이며, **동일 총 SM·workload에서
  phase-level uniform 분할(각 layer type에 108/2=54 SM) vs layer-type-aware 분할(SSM≤81 SM,
  나머지를 Attn)의 합산 TTFT / TPOT / goodput 비교**가 없음.
  이것이 §3의 비대칭 이득을 실제 서빙 metric으로 환산하는 핵심 측정이다.
- **실험 설계**: `run_stage3_concurrent.py`에 policy_aware 모드 추가:
  SSM stream에 sm_count=81(FH1) 또는 94(Zamba2), Attn stream에 나머지 SM 고정 배분 후
  concurrent prefill TTFT/TPOT을 baseline(uniform 54 SM each)과 비교.

### 6.3 Alpha (HBM 경합 완화, 경로 B)

- **공백**: SSM을 적은 SM에 묶을 때 HBM bandwidth 경합이 줄어드는 효과(alpha)는 free variable.
  논리적으로는 SSM memory-bound 특성상 SM 축소 시 HBM traffic 집중 → 다른 stream에 대역폭 양보 가능성이 있으나 미측정.
- **실험 설계**: NSight Systems로 concurrent 실행 시 HBM bandwidth utilization timeline을
  policy별(uniform vs aware)로 비교.

---

*이 보고서는 scan-only 재측정(chunked prefill)으로 HM argument chain의 §C·D 전제를 검정한다.
§E(concurrent counterfactual)는 미측정 공백으로 남아 있어 §3의 비대칭 이득이 실제 서빙 metric으로
전환됨을 아직 확인하지 못했다.*
