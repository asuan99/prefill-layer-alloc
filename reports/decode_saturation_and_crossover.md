# Decode Saturation & KV/Weight Crossover

모든 수치에 **measured** / **derived** 라벨을 명시한다. 미측정 칸은 비워두고 `미측정` 으로 표기한다. 보간·합성값 없음.

- 측정 환경: A100-SXM4-80GB (108 SM, HBM2e 2000 GB/s)
- saturation 기준: throughput(∝1/latency) marginal gain < 3% per 10% SM (Stage 1 과 동일)


## zamba2  (free zone = 14 SM, Stage 1 정정 측정)

### 판정 1 — Q1 (Policy C 생사)

대표 serving config: layer scope, bs=8, L=8192 (decode_attn) / L=0 (decode_ssm).

| cell | sm_sat_decode (derived) | free_zone | 판정 |
|------|------|------|------|
| decode_ssm | 40 | 14 | NOT absorbed — decode가 free zone보다 26 SM 더 요구; 잔여 이득 상한 ≈ 26 SM (sm_sat=40 > 14) |
| decode_attn | 68 | 14 | NOT absorbed — decode가 free zone보다 54 SM 더 요구; 잔여 이득 상한 ≈ 54 SM (sm_sat=68 > 14) |

### 판정 2 — Q3 가정 (a) (decode 행 BW 분리)

measured achieved_bw(decode_attn,kernel)/achieved_bw(decode_ssm,kernel) vs L, bs=8 — Task 0 derived (kv/state) 곡선과 overlay.

| L | achieved_bw_attn (measured, GB/s) | achieved_bw_ssm (measured, GB/s) | ratio measured | ratio derived (Task 0) |
|---|---|---|---|---|
| 512 | 895.3 | 170.9 | 5.240 | 7.749 |
| 2048 | 992.4 | 170.9 | 5.808 | 30.997 |
| 8192 | 1019.7 | 170.9 | 5.968 | 123.987 |
| 32768 | 1027.6 | 170.9 | 6.014 | 495.948 |

> measured ratio 가 derived ratio 의 추세(L 증가 시 단조 증가)를 따르면 분석 모델(가정 a) 신뢰. 큰 괴리는 모델 재검토 필요.

### Sanity (measured)

- decode_ssm kernel sm=108 bs=8: latency=85.92 us (measured) — Stage 1 decode 참조치와 자릿수 비교용
- Stage 1 ssm_chunked reference loaded (768 rows) — decode 자릿수 sanity 참조 (prefill seq_len=1 기준)
- status 분포 (measured): ok=400, OOM=0, CUDA_ERROR=0 — CUDA_ERROR 행이 후속 행을 오염시키지 않음(서브프로세스 격리)
- decode_attn kernel sm=108 bs=1: latency monotonic in L = True (measured)

![fig_decode_sat_zamba2.png](fig_decode_sat_zamba2.png)
![fig_decode_bw_split_zamba2.png](fig_decode_bw_split_zamba2.png)


## falcon_h1  (free zone = 27 SM, Stage 1 정정 측정)

### 판정 1 — Q1 (Policy C 생사)

대표 serving config: layer scope, bs=8, L=8192 (decode_attn) / L=0 (decode_ssm).

| cell | sm_sat_decode (derived) | free_zone | 판정 |
|------|------|------|------|
| decode_ssm | 27 | 27 | absorbed — fixed split이 free zone(27)을 흡수, layer-wise 추가 이득 상한 ≈ 0 (sm_sat=27 ≤ 27) |
| decode_attn | 108 | 27 | NOT absorbed — decode가 free zone보다 81 SM 더 요구; 잔여 이득 상한 ≈ 81 SM (sm_sat=108 > 27) |

### 판정 2 — Q3 가정 (a) (decode 행 BW 분리)

measured achieved_bw(decode_attn,kernel)/achieved_bw(decode_ssm,kernel) vs L, bs=8 — Task 0 derived (kv/state) 곡선과 overlay.

| L | achieved_bw_attn (measured, GB/s) | achieved_bw_ssm (measured, GB/s) | ratio measured | ratio derived (Task 0) |
|---|---|---|---|---|
| 512 | 54.0 | 297.9 | 0.181 | 0.165 |
| 2048 | 64.4 | 297.9 | 0.216 | 0.661 |
| 8192 | 74.2 | 297.9 | 0.249 | 2.643 |
| 32768 | 72.8 | 297.9 | 0.244 | 10.570 |

> measured ratio 가 derived ratio 의 추세(L 증가 시 단조 증가)를 따르면 분석 모델(가정 a) 신뢰. 큰 괴리는 모델 재검토 필요.

### Sanity (measured)

- decode_ssm kernel sm=108 bs=8: latency=84.46 us (measured) — Stage 1 decode 참조치와 자릿수 비교용
- Stage 1 ssm_chunked reference loaded (768 rows) — decode 자릿수 sanity 참조 (prefill seq_len=1 기준)
- status 분포 (measured): ok=400, OOM=0, CUDA_ERROR=0 — CUDA_ERROR 행이 후속 행을 오염시키지 않음(서브프로세스 격리)
- decode_attn kernel sm=108 bs=1: latency monotonic in L = True (measured)

![fig_decode_sat_falcon_h1.png](fig_decode_sat_falcon_h1.png)
![fig_decode_bw_split_falcon_h1.png](fig_decode_bw_split_falcon_h1.png)
