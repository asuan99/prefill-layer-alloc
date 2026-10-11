# AMD 1차 계산표 (2026-10-11, rev2) — 바이트·횟수·전송 시간의 가정 하 가산 비용

> 생성: `python3 -I amd_firstorder_calc.py`. **커널 시간 0건 사용.** 모든 지연은 `L0 + bytes/BW`의 **가산 비용(경계 아님)**이며
> 기판은 미실측(V0 프로브 전) — NVLink/PCIe를 병기한다. L0 65/130 µs는 SQD의 B200 실측을 A100 가정으로 이식한 값[ASSUME].
> 비율 열의 눈금 TPOT 12 ms는 cudagraph ON Zamba2-2.7B 운영점(results/cudagraph_probe)에서 빌린 **모델 공통 환산 눈금**이지
> 모델별 TPOT도 예측도 아니다. 부팅 배너 단위는 GiB, 슬롯 수는 49(size+1).

## 0. 입력 상수

| 모델 | hidden | attn 호출/step | mamba 층 | KV (K+V) B/token | state/req (ssm+conv) | 출처 vs 부팅배너 (ssm+conv)/49 |
|---|---:|---:|---:|---:|---:|---|
| Nemotron-H-8B [YAML] | 4096 | 4 | 24 | 16,384 | 97.5 MiB | 분석식+배너conv 97.46 vs 97.38 MiB |
| Nemotron-Nano-9B-v2 [CANON] | 4480 | 4 | 27 | 16,384 | 136.9 MiB | canon 136.90 vs 136.88 MiB |
| Zamba2-2.7B [YAML] | 2560 | 9 | 54 | 184,320 | 69.2 MiB | 분석식+배너conv 69.17 vs 69.17 MiB |
| Zamba2-7B [YAML] | 3584 | 13 | 81 | 372,736 | 145.1 MiB | 분석식+배너conv 145.09 vs 145.03 MiB |
| Granite-4.0-h-micro [BOOT/WORK] | 2048 | 4 | 36 | 8,192 | 72.9 MiB | 배너 72.93 vs 72.93 MiB |
| Falcon-H1 | — | — | — | — | — | **제외**: 매 층 attention∥mamba 병렬(spatial) — 층 단위 분리 자체가 정의되지 않음 |

## 1. L-B decode step당 crossing (attention 호출마다 왕복 2회, hidden 텐서 1개 기준; residual까지 보내면 바이트 ×2)

### Nemotron-H-8B [YAML]

| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |
|---:|---|---|---:|---:|---:|---:|
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.082 ms | 0.7% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.522 ms | 4.3% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.042 ms | 8.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.083 ms | 0.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.523 ms | 4.4% |
| 8 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.043 ms | 8.7% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.101 ms | 0.8% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.541 ms | 4.5% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.061 ms | 8.8% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.083 ms | 0.7% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.523 ms | 4.4% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.043 ms | 8.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.085 ms | 0.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.525 ms | 4.4% |
| 16 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.045 ms | 8.7% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.122 ms | 1.0% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.562 ms | 4.7% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.082 ms | 9.0% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.087 ms | 0.7% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.527 ms | 4.4% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.047 ms | 8.7% |
| 32 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.090 ms | 0.8% |
| 32 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.530 ms | 4.4% |
| 32 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.050 ms | 8.8% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.164 ms | 1.4% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.604 ms | 5.0% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.124 ms | 9.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.090 ms | 0.8% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.530 ms | 4.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.050 ms | 8.8% |
| 48 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.096 ms | 0.8% |
| 48 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.536 ms | 4.5% |
| 48 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.056 ms | 8.8% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.206 ms | 1.7% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.646 ms | 5.4% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.166 ms | 9.7% |

### Nemotron-Nano-9B-v2 [CANON]

| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |
|---:|---|---|---:|---:|---:|---:|
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 70.0 KiB | 0.082 ms | 0.7% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 70.0 KiB | 0.522 ms | 4.3% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 70.0 KiB | 1.042 ms | 8.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 70.0 KiB | 0.083 ms | 0.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 70.0 KiB | 0.523 ms | 4.4% |
| 8 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 70.0 KiB | 1.043 ms | 8.7% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 70.0 KiB | 0.103 ms | 0.9% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 70.0 KiB | 0.543 ms | 4.5% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 70.0 KiB | 1.063 ms | 8.9% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.084 ms | 0.7% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.524 ms | 4.4% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.044 ms | 8.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.086 ms | 0.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.526 ms | 4.4% |
| 16 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.046 ms | 8.7% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.126 ms | 1.0% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.566 ms | 4.7% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.086 ms | 9.0% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.3 MiB | 0.088 ms | 0.7% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.3 MiB | 0.528 ms | 4.4% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.3 MiB | 1.048 ms | 8.7% |
| 32 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.3 MiB | 0.091 ms | 0.8% |
| 32 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.3 MiB | 0.531 ms | 4.4% |
| 32 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.3 MiB | 1.051 ms | 8.8% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.3 MiB | 0.172 ms | 1.4% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.3 MiB | 0.612 ms | 5.1% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.3 MiB | 1.132 ms | 9.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.091 ms | 0.8% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.531 ms | 4.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.051 ms | 8.8% |
| 48 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.097 ms | 0.8% |
| 48 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.537 ms | 4.5% |
| 48 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.057 ms | 8.8% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.4 MiB | 0.218 ms | 1.8% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.4 MiB | 0.658 ms | 5.5% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.4 MiB | 1.178 ms | 9.8% |

### Zamba2-2.7B [YAML]

| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |
|---:|---|---|---:|---:|---:|---:|
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 18 | 40.0 KiB | 0.182 ms | 1.5% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 40.0 KiB | 1.172 ms | 9.8% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 40.0 KiB | 2.342 ms | 19.5% |
| 8 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 18 | 40.0 KiB | 0.184 ms | 1.5% |
| 8 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 40.0 KiB | 1.174 ms | 9.8% |
| 8 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 40.0 KiB | 2.344 ms | 19.5% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 18 | 40.0 KiB | 0.209 ms | 1.7% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 40.0 KiB | 1.199 ms | 10.0% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 40.0 KiB | 2.369 ms | 19.7% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 18 | 80.0 KiB | 0.185 ms | 1.5% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 80.0 KiB | 1.175 ms | 9.8% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 80.0 KiB | 2.345 ms | 19.5% |
| 16 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 18 | 80.0 KiB | 0.187 ms | 1.6% |
| 16 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 80.0 KiB | 1.177 ms | 9.8% |
| 16 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 80.0 KiB | 2.347 ms | 19.6% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 18 | 80.0 KiB | 0.239 ms | 2.0% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 80.0 KiB | 1.229 ms | 10.2% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 80.0 KiB | 2.399 ms | 20.0% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.190 ms | 1.6% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.180 ms | 9.8% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.350 ms | 19.6% |
| 32 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.195 ms | 1.6% |
| 32 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.185 ms | 9.9% |
| 32 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.355 ms | 19.6% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.298 ms | 2.5% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.288 ms | 10.7% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.458 ms | 20.5% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.195 ms | 1.6% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.185 ms | 9.9% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.355 ms | 19.6% |
| 48 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.202 ms | 1.7% |
| 48 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.192 ms | 9.9% |
| 48 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.362 ms | 19.7% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 18 | 0.2 MiB | 0.357 ms | 3.0% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 18 | 0.2 MiB | 1.347 ms | 11.2% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 18 | 0.2 MiB | 2.517 ms | 21.0% |

### Zamba2-7B [YAML]

| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |
|---:|---|---|---:|---:|---:|---:|
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 26 | 56.0 KiB | 0.265 ms | 2.2% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 56.0 KiB | 1.695 ms | 14.1% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 56.0 KiB | 3.385 ms | 28.2% |
| 8 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 26 | 56.0 KiB | 0.267 ms | 2.2% |
| 8 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 56.0 KiB | 1.697 ms | 14.1% |
| 8 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 56.0 KiB | 3.387 ms | 28.2% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 26 | 56.0 KiB | 0.320 ms | 2.7% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 56.0 KiB | 1.750 ms | 14.6% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 56.0 KiB | 3.440 ms | 28.7% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 26 | 0.1 MiB | 0.270 ms | 2.2% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.1 MiB | 1.700 ms | 14.2% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.1 MiB | 3.390 ms | 28.2% |
| 16 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.1 MiB | 0.275 ms | 2.3% |
| 16 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.1 MiB | 1.705 ms | 14.2% |
| 16 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.1 MiB | 3.395 ms | 28.3% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.1 MiB | 0.379 ms | 3.2% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.1 MiB | 1.809 ms | 15.1% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.1 MiB | 3.499 ms | 29.2% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 26 | 0.2 MiB | 0.280 ms | 2.3% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.2 MiB | 1.710 ms | 14.2% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.2 MiB | 3.400 ms | 28.3% |
| 32 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.2 MiB | 0.290 ms | 2.4% |
| 32 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.2 MiB | 1.720 ms | 14.3% |
| 32 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.2 MiB | 3.410 ms | 28.4% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.2 MiB | 0.499 ms | 4.2% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.2 MiB | 1.929 ms | 16.1% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.2 MiB | 3.619 ms | 30.2% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 26 | 0.3 MiB | 0.290 ms | 2.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.3 MiB | 1.720 ms | 14.3% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.3 MiB | 3.410 ms | 28.4% |
| 48 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.3 MiB | 0.305 ms | 2.5% |
| 48 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.3 MiB | 1.735 ms | 14.5% |
| 48 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.3 MiB | 3.425 ms | 28.5% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 26 | 0.3 MiB | 0.618 ms | 5.1% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 26 | 0.3 MiB | 2.048 ms | 17.1% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 26 | 0.3 MiB | 3.738 ms | 31.1% |

### Granite-4.0-h-micro [BOOT/WORK]

| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |
|---:|---|---|---:|---:|---:|---:|
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 32.0 KiB | 0.081 ms | 0.7% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 32.0 KiB | 0.521 ms | 4.3% |
| 8 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 32.0 KiB | 1.041 ms | 8.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 32.0 KiB | 0.081 ms | 0.7% |
| 8 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 32.0 KiB | 0.521 ms | 4.3% |
| 8 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 32.0 KiB | 1.041 ms | 8.7% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 32.0 KiB | 0.090 ms | 0.8% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 32.0 KiB | 0.530 ms | 4.4% |
| 8 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 32.0 KiB | 1.050 ms | 8.8% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.082 ms | 0.7% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.522 ms | 4.3% |
| 16 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.042 ms | 8.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.083 ms | 0.7% |
| 16 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.523 ms | 4.4% |
| 16 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.043 ms | 8.7% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 64.0 KiB | 0.101 ms | 0.8% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 64.0 KiB | 0.541 ms | 4.5% |
| 16 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 64.0 KiB | 1.061 ms | 8.8% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.083 ms | 0.7% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.523 ms | 4.4% |
| 32 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.043 ms | 8.7% |
| 32 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.085 ms | 0.7% |
| 32 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.525 ms | 4.4% |
| 32 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.045 ms | 8.7% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.1 MiB | 0.122 ms | 1.0% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.1 MiB | 0.562 ms | 4.7% |
| 32 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.1 MiB | 1.082 ms | 9.0% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.085 ms | 0.7% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.525 ms | 4.4% |
| 48 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.045 ms | 8.7% |
| 48 | NVLink(실효 보수 200 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.088 ms | 0.7% |
| 48 | NVLink(실효 보수 200 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.528 ms | 4.4% |
| 48 | NVLink(실효 보수 200 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.048 ms | 8.7% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 10us(커널 p2p 낙관) | 8 | 0.2 MiB | 0.143 ms | 1.2% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 65us[ASSUME: SQD B200 one-sided put 하한 이식] | 8 | 0.2 MiB | 0.583 ms | 4.9% |
| 48 | PCIe Gen4 x16(실효 25 GB/s) | 130us[ASSUME: SQD B200 상한 이식] | 8 | 0.2 MiB | 1.103 ms | 9.2% |

## 2. P→D 핸드오프: KV(attention 층, L-스케일) vs mamba state(L-불변) — L-B는 state만 건너고 PD-disagg 대조군은 둘 다 건넌다

### Nemotron-H-8B [YAML]

| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 16.0 MiB | 97.5 MiB | 0.056 ms | 0.341 ms | 0.397 ms |
| 1024 | NVLink(실효 보수 200 GB/s) | 16.0 MiB | 97.5 MiB | 0.084 ms | 0.511 ms | 0.595 ms |
| 1024 | PCIe Gen4 x16(실효 25 GB/s) | 16.0 MiB | 97.5 MiB | 0.671 ms | 4.088 ms | 4.759 ms |
| 4096 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 64.0 MiB | 97.5 MiB | 0.224 ms | 0.341 ms | 0.564 ms |
| 4096 | NVLink(실효 보수 200 GB/s) | 64.0 MiB | 97.5 MiB | 0.336 ms | 0.511 ms | 0.847 ms |
| 4096 | PCIe Gen4 x16(실효 25 GB/s) | 64.0 MiB | 97.5 MiB | 2.684 ms | 4.088 ms | 6.772 ms |
| 8192 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 128.0 MiB | 97.5 MiB | 0.447 ms | 0.341 ms | 0.788 ms |
| 8192 | NVLink(실효 보수 200 GB/s) | 128.0 MiB | 97.5 MiB | 0.671 ms | 0.511 ms | 1.182 ms |
| 8192 | PCIe Gen4 x16(실효 25 GB/s) | 128.0 MiB | 97.5 MiB | 5.369 ms | 4.088 ms | 9.457 ms |
| 16384 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 256.0 MiB | 97.5 MiB | 0.895 ms | 0.341 ms | 1.235 ms |
| 16384 | NVLink(실효 보수 200 GB/s) | 256.0 MiB | 97.5 MiB | 1.342 ms | 0.511 ms | 1.853 ms |
| 16384 | PCIe Gen4 x16(실효 25 GB/s) | 256.0 MiB | 97.5 MiB | 10.737 ms | 4.088 ms | 14.825 ms |

### Nemotron-Nano-9B-v2 [CANON]

| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 16.0 MiB | 136.9 MiB | 0.056 ms | 0.479 ms | 0.534 ms |
| 1024 | NVLink(실효 보수 200 GB/s) | 16.0 MiB | 136.9 MiB | 0.084 ms | 0.718 ms | 0.802 ms |
| 1024 | PCIe Gen4 x16(실효 25 GB/s) | 16.0 MiB | 136.9 MiB | 0.671 ms | 5.742 ms | 6.413 ms |
| 4096 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 64.0 MiB | 136.9 MiB | 0.224 ms | 0.479 ms | 0.702 ms |
| 4096 | NVLink(실효 보수 200 GB/s) | 64.0 MiB | 136.9 MiB | 0.336 ms | 0.718 ms | 1.053 ms |
| 4096 | PCIe Gen4 x16(실효 25 GB/s) | 64.0 MiB | 136.9 MiB | 2.684 ms | 5.742 ms | 8.426 ms |
| 8192 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 128.0 MiB | 136.9 MiB | 0.447 ms | 0.479 ms | 0.926 ms |
| 8192 | NVLink(실효 보수 200 GB/s) | 128.0 MiB | 136.9 MiB | 0.671 ms | 0.718 ms | 1.389 ms |
| 8192 | PCIe Gen4 x16(실효 25 GB/s) | 128.0 MiB | 136.9 MiB | 5.369 ms | 5.742 ms | 11.111 ms |
| 16384 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 256.0 MiB | 136.9 MiB | 0.895 ms | 0.479 ms | 1.373 ms |
| 16384 | NVLink(실효 보수 200 GB/s) | 256.0 MiB | 136.9 MiB | 1.342 ms | 0.718 ms | 2.060 ms |
| 16384 | PCIe Gen4 x16(실효 25 GB/s) | 256.0 MiB | 136.9 MiB | 10.737 ms | 5.742 ms | 16.479 ms |

### Zamba2-2.7B [YAML]

| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 180.0 MiB | 69.2 MiB | 0.629 ms | 0.242 ms | 0.871 ms |
| 1024 | NVLink(실효 보수 200 GB/s) | 180.0 MiB | 69.2 MiB | 0.944 ms | 0.363 ms | 1.306 ms |
| 1024 | PCIe Gen4 x16(실효 25 GB/s) | 180.0 MiB | 69.2 MiB | 7.550 ms | 2.901 ms | 10.451 ms |
| 4096 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 720.0 MiB | 69.2 MiB | 2.517 ms | 0.242 ms | 2.758 ms |
| 4096 | NVLink(실효 보수 200 GB/s) | 720.0 MiB | 69.2 MiB | 3.775 ms | 0.363 ms | 4.138 ms |
| 4096 | PCIe Gen4 x16(실효 25 GB/s) | 720.0 MiB | 69.2 MiB | 30.199 ms | 2.901 ms | 33.100 ms |
| 8192 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 1440.0 MiB | 69.2 MiB | 5.033 ms | 0.242 ms | 5.275 ms |
| 8192 | NVLink(실효 보수 200 GB/s) | 1440.0 MiB | 69.2 MiB | 7.550 ms | 0.363 ms | 7.912 ms |
| 8192 | PCIe Gen4 x16(실효 25 GB/s) | 1440.0 MiB | 69.2 MiB | 60.398 ms | 2.901 ms | 63.299 ms |
| 16384 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 2880.0 MiB | 69.2 MiB | 10.066 ms | 0.242 ms | 10.308 ms |
| 16384 | NVLink(실효 보수 200 GB/s) | 2880.0 MiB | 69.2 MiB | 15.099 ms | 0.363 ms | 15.462 ms |
| 16384 | PCIe Gen4 x16(실효 25 GB/s) | 2880.0 MiB | 69.2 MiB | 120.796 ms | 2.901 ms | 123.697 ms |

### Zamba2-7B [YAML]

| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 364.0 MiB | 145.1 MiB | 1.272 ms | 0.507 ms | 1.779 ms |
| 1024 | NVLink(실효 보수 200 GB/s) | 364.0 MiB | 145.1 MiB | 1.908 ms | 0.761 ms | 2.669 ms |
| 1024 | PCIe Gen4 x16(실효 25 GB/s) | 364.0 MiB | 145.1 MiB | 15.267 ms | 6.086 ms | 21.353 ms |
| 4096 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 1456.0 MiB | 145.1 MiB | 5.089 ms | 0.507 ms | 5.596 ms |
| 4096 | NVLink(실효 보수 200 GB/s) | 1456.0 MiB | 145.1 MiB | 7.634 ms | 0.761 ms | 8.394 ms |
| 4096 | PCIe Gen4 x16(실효 25 GB/s) | 1456.0 MiB | 145.1 MiB | 61.069 ms | 6.086 ms | 67.155 ms |
| 8192 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 2912.0 MiB | 145.1 MiB | 10.178 ms | 0.507 ms | 10.685 ms |
| 8192 | NVLink(실효 보수 200 GB/s) | 2912.0 MiB | 145.1 MiB | 15.267 ms | 0.761 ms | 16.028 ms |
| 8192 | PCIe Gen4 x16(실효 25 GB/s) | 2912.0 MiB | 145.1 MiB | 122.138 ms | 6.086 ms | 128.224 ms |
| 16384 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 5824.0 MiB | 145.1 MiB | 20.356 ms | 0.507 ms | 20.863 ms |
| 16384 | NVLink(실효 보수 200 GB/s) | 5824.0 MiB | 145.1 MiB | 30.535 ms | 0.761 ms | 31.295 ms |
| 16384 | PCIe Gen4 x16(실효 25 GB/s) | 5824.0 MiB | 145.1 MiB | 244.276 ms | 6.086 ms | 250.362 ms |

### Granite-4.0-h-micro [BOOT/WORK]

| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 8.0 MiB | 72.9 MiB | 0.028 ms | 0.255 ms | 0.283 ms |
| 1024 | NVLink(실효 보수 200 GB/s) | 8.0 MiB | 72.9 MiB | 0.042 ms | 0.382 ms | 0.424 ms |
| 1024 | PCIe Gen4 x16(실효 25 GB/s) | 8.0 MiB | 72.9 MiB | 0.336 ms | 3.059 ms | 3.395 ms |
| 4096 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 32.0 MiB | 72.9 MiB | 0.112 ms | 0.255 ms | 0.367 ms |
| 4096 | NVLink(실효 보수 200 GB/s) | 32.0 MiB | 72.9 MiB | 0.168 ms | 0.382 ms | 0.550 ms |
| 4096 | PCIe Gen4 x16(실효 25 GB/s) | 32.0 MiB | 72.9 MiB | 1.342 ms | 3.059 ms | 4.401 ms |
| 8192 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 64.0 MiB | 72.9 MiB | 0.224 ms | 0.255 ms | 0.479 ms |
| 8192 | NVLink(실효 보수 200 GB/s) | 64.0 MiB | 72.9 MiB | 0.336 ms | 0.382 ms | 0.718 ms |
| 8192 | PCIe Gen4 x16(실효 25 GB/s) | 64.0 MiB | 72.9 MiB | 2.684 ms | 3.059 ms | 5.743 ms |
| 16384 | NVLink(A100 SXM, 단방향 300 GB/s 사양) | 128.0 MiB | 72.9 MiB | 0.447 ms | 0.255 ms | 0.702 ms |
| 16384 | NVLink(실효 보수 200 GB/s) | 128.0 MiB | 72.9 MiB | 0.671 ms | 0.382 ms | 1.053 ms |
| 16384 | PCIe Gen4 x16(실효 25 GB/s) | 128.0 MiB | 72.9 MiB | 5.369 ms | 3.059 ms | 8.428 ms |

## 3. wall-time 합산 비율 R — **SLO 지표 아님**, 뒤집힘 격자

R = (요청당 KV 전송 절감 ms) / (출력 토큰 수 × step당 crossing 가산 ms). KV 절감은 TTFT 쪽 1회, crossing은 ITL 쪽 매 step이라
같은 ms로 더한 R은 goodput과 무관하다. R은 **ctx가 길고·출력이 짧고·기판이 느리고·L0가 작을수록** 1을 넘는다(아래 격자, B=32).
어느 축이 SLO 위에서 이기는지는 V4 직접 측정만 답한다.

| 모델 | ctx | 출력 | 기판 | L0 | KV 절감/req | crossing/step | R |
|---|---:|---:|---|---:|---:|---:|---:|
| Nemotron-H-8B [YAML] | 4096 | 64 | NVLink 200 | 10 | 0.336 ms | 0.090 ms | 0.058 |
| Nemotron-H-8B [YAML] | 4096 | 64 | NVLink 200 | 65 | 0.336 ms | 0.530 ms | 0.010 |
| Nemotron-H-8B [YAML] | 4096 | 64 | PCIe 25 | 10 | 2.684 ms | 0.164 ms | 0.256 |
| Nemotron-H-8B [YAML] | 4096 | 64 | PCIe 25 | 65 | 2.684 ms | 0.604 ms | 0.069 |
| Nemotron-H-8B [YAML] | 4096 | 256 | NVLink 200 | 10 | 0.336 ms | 0.090 ms | 0.014 |
| Nemotron-H-8B [YAML] | 4096 | 256 | NVLink 200 | 65 | 0.336 ms | 0.530 ms | 0.002 |
| Nemotron-H-8B [YAML] | 4096 | 256 | PCIe 25 | 10 | 2.684 ms | 0.164 ms | 0.064 |
| Nemotron-H-8B [YAML] | 4096 | 256 | PCIe 25 | 65 | 2.684 ms | 0.604 ms | 0.017 |
| Nemotron-H-8B [YAML] | 4096 | 1024 | NVLink 200 | 10 | 0.336 ms | 0.090 ms | 0.004 |
| Nemotron-H-8B [YAML] | 4096 | 1024 | NVLink 200 | 65 | 0.336 ms | 0.530 ms | 0.001 |
| Nemotron-H-8B [YAML] | 4096 | 1024 | PCIe 25 | 10 | 2.684 ms | 0.164 ms | 0.016 |
| Nemotron-H-8B [YAML] | 4096 | 1024 | PCIe 25 | 65 | 2.684 ms | 0.604 ms | 0.004 |
| Nemotron-H-8B [YAML] | 8192 | 64 | NVLink 200 | 10 | 0.671 ms | 0.090 ms | 0.116 |
| Nemotron-H-8B [YAML] | 8192 | 64 | NVLink 200 | 65 | 0.671 ms | 0.530 ms | 0.020 |
| Nemotron-H-8B [YAML] | 8192 | 64 | PCIe 25 | 10 | 5.369 ms | 0.164 ms | 0.512 |
| Nemotron-H-8B [YAML] | 8192 | 64 | PCIe 25 | 65 | 5.369 ms | 0.604 ms | 0.139 |
| Nemotron-H-8B [YAML] | 8192 | 256 | NVLink 200 | 10 | 0.671 ms | 0.090 ms | 0.029 |
| Nemotron-H-8B [YAML] | 8192 | 256 | NVLink 200 | 65 | 0.671 ms | 0.530 ms | 0.005 |
| Nemotron-H-8B [YAML] | 8192 | 256 | PCIe 25 | 10 | 5.369 ms | 0.164 ms | 0.128 |
| Nemotron-H-8B [YAML] | 8192 | 256 | PCIe 25 | 65 | 5.369 ms | 0.604 ms | 0.035 |
| Nemotron-H-8B [YAML] | 8192 | 1024 | NVLink 200 | 10 | 0.671 ms | 0.090 ms | 0.007 |
| Nemotron-H-8B [YAML] | 8192 | 1024 | NVLink 200 | 65 | 0.671 ms | 0.530 ms | 0.001 |
| Nemotron-H-8B [YAML] | 8192 | 1024 | PCIe 25 | 10 | 5.369 ms | 0.164 ms | 0.032 |
| Nemotron-H-8B [YAML] | 8192 | 1024 | PCIe 25 | 65 | 5.369 ms | 0.604 ms | 0.009 |
| Nemotron-H-8B [YAML] | 16384 | 64 | NVLink 200 | 10 | 1.342 ms | 0.090 ms | 0.232 |
| Nemotron-H-8B [YAML] | 16384 | 64 | NVLink 200 | 65 | 1.342 ms | 0.530 ms | 0.040 |
| Nemotron-H-8B [YAML] | 16384 | 64 | PCIe 25 | 10 | 10.737 ms | 0.164 ms | **1.024** |
| Nemotron-H-8B [YAML] | 16384 | 64 | PCIe 25 | 65 | 10.737 ms | 0.604 ms | 0.278 |
| Nemotron-H-8B [YAML] | 16384 | 256 | NVLink 200 | 10 | 1.342 ms | 0.090 ms | 0.058 |
| Nemotron-H-8B [YAML] | 16384 | 256 | NVLink 200 | 65 | 1.342 ms | 0.530 ms | 0.010 |
| Nemotron-H-8B [YAML] | 16384 | 256 | PCIe 25 | 10 | 10.737 ms | 0.164 ms | 0.256 |
| Nemotron-H-8B [YAML] | 16384 | 256 | PCIe 25 | 65 | 10.737 ms | 0.604 ms | 0.069 |
| Nemotron-H-8B [YAML] | 16384 | 1024 | NVLink 200 | 10 | 1.342 ms | 0.090 ms | 0.014 |
| Nemotron-H-8B [YAML] | 16384 | 1024 | NVLink 200 | 65 | 1.342 ms | 0.530 ms | 0.002 |
| Nemotron-H-8B [YAML] | 16384 | 1024 | PCIe 25 | 10 | 10.737 ms | 0.164 ms | 0.064 |
| Nemotron-H-8B [YAML] | 16384 | 1024 | PCIe 25 | 65 | 10.737 ms | 0.604 ms | 0.017 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 64 | NVLink 200 | 10 | 0.336 ms | 0.091 ms | 0.057 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 64 | NVLink 200 | 65 | 0.336 ms | 0.531 ms | 0.010 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 64 | PCIe 25 | 10 | 2.684 ms | 0.172 ms | 0.244 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 64 | PCIe 25 | 65 | 2.684 ms | 0.612 ms | 0.069 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 256 | NVLink 200 | 10 | 0.336 ms | 0.091 ms | 0.014 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 256 | NVLink 200 | 65 | 0.336 ms | 0.531 ms | 0.002 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 256 | PCIe 25 | 10 | 2.684 ms | 0.172 ms | 0.061 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 256 | PCIe 25 | 65 | 2.684 ms | 0.612 ms | 0.017 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 1024 | NVLink 200 | 10 | 0.336 ms | 0.091 ms | 0.004 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 1024 | NVLink 200 | 65 | 0.336 ms | 0.531 ms | 0.001 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 1024 | PCIe 25 | 10 | 2.684 ms | 0.172 ms | 0.015 |
| Nemotron-Nano-9B-v2 [CANON] | 4096 | 1024 | PCIe 25 | 65 | 2.684 ms | 0.612 ms | 0.004 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 64 | NVLink 200 | 10 | 0.671 ms | 0.091 ms | 0.115 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 64 | NVLink 200 | 65 | 0.671 ms | 0.531 ms | 0.020 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 64 | PCIe 25 | 10 | 5.369 ms | 0.172 ms | 0.488 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 64 | PCIe 25 | 65 | 5.369 ms | 0.612 ms | 0.137 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 256 | NVLink 200 | 10 | 0.671 ms | 0.091 ms | 0.029 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 256 | NVLink 200 | 65 | 0.671 ms | 0.531 ms | 0.005 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 256 | PCIe 25 | 10 | 5.369 ms | 0.172 ms | 0.122 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 256 | PCIe 25 | 65 | 5.369 ms | 0.612 ms | 0.034 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 1024 | NVLink 200 | 10 | 0.671 ms | 0.091 ms | 0.007 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 1024 | NVLink 200 | 65 | 0.671 ms | 0.531 ms | 0.001 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 1024 | PCIe 25 | 10 | 5.369 ms | 0.172 ms | 0.031 |
| Nemotron-Nano-9B-v2 [CANON] | 8192 | 1024 | PCIe 25 | 65 | 5.369 ms | 0.612 ms | 0.009 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 64 | NVLink 200 | 10 | 1.342 ms | 0.091 ms | 0.229 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 64 | NVLink 200 | 65 | 1.342 ms | 0.531 ms | 0.039 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 64 | PCIe 25 | 10 | 10.737 ms | 0.172 ms | 0.977 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 64 | PCIe 25 | 65 | 10.737 ms | 0.612 ms | 0.274 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 256 | NVLink 200 | 10 | 1.342 ms | 0.091 ms | 0.057 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 256 | NVLink 200 | 65 | 1.342 ms | 0.531 ms | 0.010 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 256 | PCIe 25 | 10 | 10.737 ms | 0.172 ms | 0.244 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 256 | PCIe 25 | 65 | 10.737 ms | 0.612 ms | 0.069 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 1024 | NVLink 200 | 10 | 1.342 ms | 0.091 ms | 0.014 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 1024 | NVLink 200 | 65 | 1.342 ms | 0.531 ms | 0.002 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 1024 | PCIe 25 | 10 | 10.737 ms | 0.172 ms | 0.061 |
| Nemotron-Nano-9B-v2 [CANON] | 16384 | 1024 | PCIe 25 | 65 | 10.737 ms | 0.612 ms | 0.017 |
| Zamba2-2.7B [YAML] | 4096 | 64 | NVLink 200 | 10 | 3.775 ms | 0.195 ms | 0.303 |
| Zamba2-2.7B [YAML] | 4096 | 64 | NVLink 200 | 65 | 3.775 ms | 1.185 ms | 0.050 |
| Zamba2-2.7B [YAML] | 4096 | 64 | PCIe 25 | 10 | 30.199 ms | 0.298 ms | **1.584** |
| Zamba2-2.7B [YAML] | 4096 | 64 | PCIe 25 | 65 | 30.199 ms | 1.288 ms | 0.366 |
| Zamba2-2.7B [YAML] | 4096 | 256 | NVLink 200 | 10 | 3.775 ms | 0.195 ms | 0.076 |
| Zamba2-2.7B [YAML] | 4096 | 256 | NVLink 200 | 65 | 3.775 ms | 1.185 ms | 0.012 |
| Zamba2-2.7B [YAML] | 4096 | 256 | PCIe 25 | 10 | 30.199 ms | 0.298 ms | 0.396 |
| Zamba2-2.7B [YAML] | 4096 | 256 | PCIe 25 | 65 | 30.199 ms | 1.288 ms | 0.092 |
| Zamba2-2.7B [YAML] | 4096 | 1024 | NVLink 200 | 10 | 3.775 ms | 0.195 ms | 0.019 |
| Zamba2-2.7B [YAML] | 4096 | 1024 | NVLink 200 | 65 | 3.775 ms | 1.185 ms | 0.003 |
| Zamba2-2.7B [YAML] | 4096 | 1024 | PCIe 25 | 10 | 30.199 ms | 0.298 ms | 0.099 |
| Zamba2-2.7B [YAML] | 4096 | 1024 | PCIe 25 | 65 | 30.199 ms | 1.288 ms | 0.023 |
| Zamba2-2.7B [YAML] | 8192 | 64 | NVLink 200 | 10 | 7.550 ms | 0.195 ms | 0.606 |
| Zamba2-2.7B [YAML] | 8192 | 64 | NVLink 200 | 65 | 7.550 ms | 1.185 ms | 0.100 |
| Zamba2-2.7B [YAML] | 8192 | 64 | PCIe 25 | 10 | 60.398 ms | 0.298 ms | **3.167** |
| Zamba2-2.7B [YAML] | 8192 | 64 | PCIe 25 | 65 | 60.398 ms | 1.288 ms | 0.733 |
| Zamba2-2.7B [YAML] | 8192 | 256 | NVLink 200 | 10 | 7.550 ms | 0.195 ms | 0.151 |
| Zamba2-2.7B [YAML] | 8192 | 256 | NVLink 200 | 65 | 7.550 ms | 1.185 ms | 0.025 |
| Zamba2-2.7B [YAML] | 8192 | 256 | PCIe 25 | 10 | 60.398 ms | 0.298 ms | 0.792 |
| Zamba2-2.7B [YAML] | 8192 | 256 | PCIe 25 | 65 | 60.398 ms | 1.288 ms | 0.183 |
| Zamba2-2.7B [YAML] | 8192 | 1024 | NVLink 200 | 10 | 7.550 ms | 0.195 ms | 0.038 |
| Zamba2-2.7B [YAML] | 8192 | 1024 | NVLink 200 | 65 | 7.550 ms | 1.185 ms | 0.006 |
| Zamba2-2.7B [YAML] | 8192 | 1024 | PCIe 25 | 10 | 60.398 ms | 0.298 ms | 0.198 |
| Zamba2-2.7B [YAML] | 8192 | 1024 | PCIe 25 | 65 | 60.398 ms | 1.288 ms | 0.046 |
| Zamba2-2.7B [YAML] | 16384 | 64 | NVLink 200 | 10 | 15.099 ms | 0.195 ms | **1.211** |
| Zamba2-2.7B [YAML] | 16384 | 64 | NVLink 200 | 65 | 15.099 ms | 1.185 ms | 0.199 |
| Zamba2-2.7B [YAML] | 16384 | 64 | PCIe 25 | 10 | 120.796 ms | 0.298 ms | **6.334** |
| Zamba2-2.7B [YAML] | 16384 | 64 | PCIe 25 | 65 | 120.796 ms | 1.288 ms | **1.465** |
| Zamba2-2.7B [YAML] | 16384 | 256 | NVLink 200 | 10 | 15.099 ms | 0.195 ms | 0.303 |
| Zamba2-2.7B [YAML] | 16384 | 256 | NVLink 200 | 65 | 15.099 ms | 1.185 ms | 0.050 |
| Zamba2-2.7B [YAML] | 16384 | 256 | PCIe 25 | 10 | 120.796 ms | 0.298 ms | **1.584** |
| Zamba2-2.7B [YAML] | 16384 | 256 | PCIe 25 | 65 | 120.796 ms | 1.288 ms | 0.366 |
| Zamba2-2.7B [YAML] | 16384 | 1024 | NVLink 200 | 10 | 15.099 ms | 0.195 ms | 0.076 |
| Zamba2-2.7B [YAML] | 16384 | 1024 | NVLink 200 | 65 | 15.099 ms | 1.185 ms | 0.012 |
| Zamba2-2.7B [YAML] | 16384 | 1024 | PCIe 25 | 10 | 120.796 ms | 0.298 ms | 0.396 |
| Zamba2-2.7B [YAML] | 16384 | 1024 | PCIe 25 | 65 | 120.796 ms | 1.288 ms | 0.092 |
| Zamba2-7B [YAML] | 4096 | 64 | NVLink 200 | 10 | 7.634 ms | 0.290 ms | 0.412 |
| Zamba2-7B [YAML] | 4096 | 64 | NVLink 200 | 65 | 7.634 ms | 1.720 ms | 0.069 |
| Zamba2-7B [YAML] | 4096 | 64 | PCIe 25 | 10 | 61.069 ms | 0.499 ms | **1.914** |
| Zamba2-7B [YAML] | 4096 | 64 | PCIe 25 | 65 | 61.069 ms | 1.929 ms | 0.495 |
| Zamba2-7B [YAML] | 4096 | 256 | NVLink 200 | 10 | 7.634 ms | 0.290 ms | 0.103 |
| Zamba2-7B [YAML] | 4096 | 256 | NVLink 200 | 65 | 7.634 ms | 1.720 ms | 0.017 |
| Zamba2-7B [YAML] | 4096 | 256 | PCIe 25 | 10 | 61.069 ms | 0.499 ms | 0.478 |
| Zamba2-7B [YAML] | 4096 | 256 | PCIe 25 | 65 | 61.069 ms | 1.929 ms | 0.124 |
| Zamba2-7B [YAML] | 4096 | 1024 | NVLink 200 | 10 | 7.634 ms | 0.290 ms | 0.026 |
| Zamba2-7B [YAML] | 4096 | 1024 | NVLink 200 | 65 | 7.634 ms | 1.720 ms | 0.004 |
| Zamba2-7B [YAML] | 4096 | 1024 | PCIe 25 | 10 | 61.069 ms | 0.499 ms | 0.120 |
| Zamba2-7B [YAML] | 4096 | 1024 | PCIe 25 | 65 | 61.069 ms | 1.929 ms | 0.031 |
| Zamba2-7B [YAML] | 8192 | 64 | NVLink 200 | 10 | 15.267 ms | 0.290 ms | 0.823 |
| Zamba2-7B [YAML] | 8192 | 64 | NVLink 200 | 65 | 15.267 ms | 1.720 ms | 0.139 |
| Zamba2-7B [YAML] | 8192 | 64 | PCIe 25 | 10 | 122.138 ms | 0.499 ms | **3.828** |
| Zamba2-7B [YAML] | 8192 | 64 | PCIe 25 | 65 | 122.138 ms | 1.929 ms | 0.990 |
| Zamba2-7B [YAML] | 8192 | 256 | NVLink 200 | 10 | 15.267 ms | 0.290 ms | 0.206 |
| Zamba2-7B [YAML] | 8192 | 256 | NVLink 200 | 65 | 15.267 ms | 1.720 ms | 0.035 |
| Zamba2-7B [YAML] | 8192 | 256 | PCIe 25 | 10 | 122.138 ms | 0.499 ms | 0.957 |
| Zamba2-7B [YAML] | 8192 | 256 | PCIe 25 | 65 | 122.138 ms | 1.929 ms | 0.247 |
| Zamba2-7B [YAML] | 8192 | 1024 | NVLink 200 | 10 | 15.267 ms | 0.290 ms | 0.051 |
| Zamba2-7B [YAML] | 8192 | 1024 | NVLink 200 | 65 | 15.267 ms | 1.720 ms | 0.009 |
| Zamba2-7B [YAML] | 8192 | 1024 | PCIe 25 | 10 | 122.138 ms | 0.499 ms | 0.239 |
| Zamba2-7B [YAML] | 8192 | 1024 | PCIe 25 | 65 | 122.138 ms | 1.929 ms | 0.062 |
| Zamba2-7B [YAML] | 16384 | 64 | NVLink 200 | 10 | 30.535 ms | 0.290 ms | **1.646** |
| Zamba2-7B [YAML] | 16384 | 64 | NVLink 200 | 65 | 30.535 ms | 1.720 ms | 0.277 |
| Zamba2-7B [YAML] | 16384 | 64 | PCIe 25 | 10 | 244.276 ms | 0.499 ms | **7.656** |
| Zamba2-7B [YAML] | 16384 | 64 | PCIe 25 | 65 | 244.276 ms | 1.929 ms | **1.979** |
| Zamba2-7B [YAML] | 16384 | 256 | NVLink 200 | 10 | 30.535 ms | 0.290 ms | 0.412 |
| Zamba2-7B [YAML] | 16384 | 256 | NVLink 200 | 65 | 30.535 ms | 1.720 ms | 0.069 |
| Zamba2-7B [YAML] | 16384 | 256 | PCIe 25 | 10 | 244.276 ms | 0.499 ms | **1.914** |
| Zamba2-7B [YAML] | 16384 | 256 | PCIe 25 | 65 | 244.276 ms | 1.929 ms | 0.495 |
| Zamba2-7B [YAML] | 16384 | 1024 | NVLink 200 | 10 | 30.535 ms | 0.290 ms | 0.103 |
| Zamba2-7B [YAML] | 16384 | 1024 | NVLink 200 | 65 | 30.535 ms | 1.720 ms | 0.017 |
| Zamba2-7B [YAML] | 16384 | 1024 | PCIe 25 | 10 | 244.276 ms | 0.499 ms | 0.478 |
| Zamba2-7B [YAML] | 16384 | 1024 | PCIe 25 | 65 | 244.276 ms | 1.929 ms | 0.124 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 64 | NVLink 200 | 10 | 0.168 ms | 0.085 ms | 0.031 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 64 | NVLink 200 | 65 | 0.168 ms | 0.525 ms | 0.005 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 64 | PCIe 25 | 10 | 1.342 ms | 0.122 ms | 0.172 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 64 | PCIe 25 | 65 | 1.342 ms | 0.562 ms | 0.037 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 256 | NVLink 200 | 10 | 0.168 ms | 0.085 ms | 0.008 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 256 | NVLink 200 | 65 | 0.168 ms | 0.525 ms | 0.001 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 256 | PCIe 25 | 10 | 1.342 ms | 0.122 ms | 0.043 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 256 | PCIe 25 | 65 | 1.342 ms | 0.562 ms | 0.009 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 1024 | NVLink 200 | 10 | 0.168 ms | 0.085 ms | 0.002 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 1024 | NVLink 200 | 65 | 0.168 ms | 0.525 ms | 0.000 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 1024 | PCIe 25 | 10 | 1.342 ms | 0.122 ms | 0.011 |
| Granite-4.0-h-micro [BOOT/WORK] | 4096 | 1024 | PCIe 25 | 65 | 1.342 ms | 0.562 ms | 0.002 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 64 | NVLink 200 | 10 | 0.336 ms | 0.085 ms | 0.062 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 64 | NVLink 200 | 65 | 0.336 ms | 0.525 ms | 0.010 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 64 | PCIe 25 | 10 | 2.684 ms | 0.122 ms | 0.344 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 64 | PCIe 25 | 65 | 2.684 ms | 0.562 ms | 0.075 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 256 | NVLink 200 | 10 | 0.336 ms | 0.085 ms | 0.015 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 256 | NVLink 200 | 65 | 0.336 ms | 0.525 ms | 0.002 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 256 | PCIe 25 | 10 | 2.684 ms | 0.122 ms | 0.086 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 256 | PCIe 25 | 65 | 2.684 ms | 0.562 ms | 0.019 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 1024 | NVLink 200 | 10 | 0.336 ms | 0.085 ms | 0.004 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 1024 | NVLink 200 | 65 | 0.336 ms | 0.525 ms | 0.001 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 1024 | PCIe 25 | 10 | 2.684 ms | 0.122 ms | 0.021 |
| Granite-4.0-h-micro [BOOT/WORK] | 8192 | 1024 | PCIe 25 | 65 | 2.684 ms | 0.562 ms | 0.005 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 64 | NVLink 200 | 10 | 0.671 ms | 0.085 ms | 0.123 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 64 | NVLink 200 | 65 | 0.671 ms | 0.525 ms | 0.020 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 64 | PCIe 25 | 10 | 5.369 ms | 0.122 ms | 0.688 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 64 | PCIe 25 | 65 | 5.369 ms | 0.562 ms | 0.149 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 256 | NVLink 200 | 10 | 0.671 ms | 0.085 ms | 0.031 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 256 | NVLink 200 | 65 | 0.671 ms | 0.525 ms | 0.005 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 256 | PCIe 25 | 10 | 5.369 ms | 0.122 ms | 0.172 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 256 | PCIe 25 | 65 | 5.369 ms | 0.562 ms | 0.037 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 1024 | NVLink 200 | 10 | 0.671 ms | 0.085 ms | 0.008 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 1024 | NVLink 200 | 65 | 0.671 ms | 0.525 ms | 0.001 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 1024 | PCIe 25 | 10 | 5.369 ms | 0.122 ms | 0.043 |
| Granite-4.0-h-micro [BOOT/WORK] | 16384 | 1024 | PCIe 25 | 65 | 5.369 ms | 0.562 ms | 0.009 |

굵은 R(>1)은 합산 wall-time에서 L-B가 PD-disagg보다 덜 옮기는 셀이다 — 그래도 ITL 축만 보면 §1의 '÷TPOT' 열이 가산 비용이고,
TTFT 축만 보면 §2의 전송 열이 가산 비용이다. 둘은 goodput의 서로 다른 술어에 들어간다.
