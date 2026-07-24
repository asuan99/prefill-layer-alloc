# R1 observer-path A/B 재분석 — job 862512

최종 판정: **모든 차이는 방향성 관측이며 architecture effect가 아니다.**

## 실행 조건과 confounder

| 항목 | 확인된 값/상태 |
|---|---|
| 모델/GPU | Zamba2-2.7B, A100 108 SM |
| 요청 수 | scenario당 48; alternating mixed 96 |
| context/max running | 4096 / 48 |
| 반복 | policy/scenario당 1회 |
| 실행 순서 | 항상 legacy 후 observer |
| request seed | benchmark seed 1 |
| server seed | policy별 상이 |
| CUDA Graph | 두 arm 모두 비활성화 |
| telemetry | observer arm만 동기식 JSONL |
| worker 구현 | 같은 `event_loop_pdmux`; queue/batch alias |
| SLO score | TTFT≤3 s 및 request mean ITL≤60 ms |

이 조건에서는 평균, variance, confidence interval을 계산할 수 없고 observer
bookkeeping과 server randomness를 architecture 차이에서 분리할 수 없다.

## Scenario 정의

| Scenario/phase | 실제 input/output token | Arrival rate | Split | Avg/max concurrency legacy→observer |
|---|---:|---:|---:|---:|
| decode-heavy | 1999/96 | 8 req/s | D24/P84 | 32.54/48→31.45/48 |
| prefill_burst | 3599/32 | 2 req/s | D24/P84 | 5.25/14→5.20/14 |
| alternating prefill | 3599/32 | 2 req/s | D24/P84 | 5.50/15→5.24/15 |
| alternating decode | 1999/96 | 8 req/s | D24/P84 | 30.93/47→31.11/47 |
| low-decode-SM | 1999/96 | 8 req/s | D16/P92 | 34.43/48→34.56/48 |
| overload | 3599/96 | 12 req/s | D24/P84 | 33.62/48→33.62/48 |

동시성은 fixed input이 아니라 arrival와 service time에서 발생한 관측값이다.

## 보존할 원시 결과

| Scenario | Throughput legacy→observer | Good requests | TTFT p50/p95/p99 ms legacy→observer | ITL p50/p95/p99 ms legacy→observer |
|---|---:|---:|---|---|
| decode-heavy | 3.150091→3.464730 | 2→2/48 | 3645/6023/6385→3348/5325/5434 | 71.06/91.31/91.32→64.24/77.31/77.31 |
| prefill_burst | 2.150164→2.150047 | 48→48/48 | 972/2458/2509→932/2389/2509 | 42.23/51.69/51.71→42.03/50.56/50.87 |
| alternating prefill | 2.132611→2.151137 | 48→48/48 | 1025/2614/2943→972/2346/2529 | 41.35/53.86/54.87→42.83/50.74/51.75 |
| alternating decode | 3.649811→3.572735 | 3→6/48 | 3094/4522/4636→3108/4585/4713 | 60.20/76.54/76.54→59.39/79.22/79.23 |
| alternating mixed | 2.692169→2.685399 | 51→54/96 | 1935/4448/4596→1594/4465/4693 | 44.26/76.53/76.54→45.78/79.22/79.23 |
| low-decode-SM | 2.988063→3.004238 | 0→0/48 | 4046/7373/7468→3991/7259/7563 | 83.69/105.03/105.04→88.63/109.40/109.40 |
| overload | 2.052796→2.064987 | 2→2/48 | 8084/15537/15713→8068/15526/15703 | 95.78/118.77/118.77→94.07/117.04/117.05 |

R1 ITL percentile은 token-level ITL distribution이 아니라 request별 mean ITL의
percentile이다. 새 primary metric과 혼용하지 않는다.

## Workload 재분류

- `prefill_burst`는 stationary rate-2 long-prompt workload로 실제 burst가 아니다.
- `alternating`은 prefill phase 1회 후 decode phase 1회이며 반복 alternation이
  아니다.
- `decode_heavy`는 input 약 1999/output 96으로, 후속 W3(input 256/output 512)보다
  약한 decode-heavy 조건이다.
- benchmark가 요청한 2000/3600 input은 tokenizer에서 실제 1999/3599가 됐다.

## 복구된 보조 수치

| Scenario | Avg/max concurrency legacy | Avg/max concurrency observer | Observer queue max/oldest age | Decode batch max |
|---|---:|---:|---:|---:|
| decode-heavy | 32.54/48 | 31.45/48 | 17/3306 ms | 41 |
| prefill_burst | 5.25/14 | 5.20/14 | 4/879 ms | 7 |
| alternating prefill | 5.50/15 | 5.24/15 | phase 미분리 | phase 미분리 |
| alternating decode | 30.93/47 | 31.11/47 | 합산 14/3796 ms | 합산 39 |
| low-decode-SM | 34.43/48 | 34.56/48 | 18/4004 ms | 43 |
| overload | 33.62/48 | 33.62/48 | 36/14268 ms | 26 |

## 미수집

- legacy queue/worker telemetry
- structured KV total/full/mamba occupancy
- worker busy/idle/utilization과 overlap
- exact transition count와 transition cost
- time-weighted split residency와 dwell distribution
- token-level ITL p95
- controller CPU overhead
- run-to-run variance와 CI
- benchmark-only phase telemetry

R1의 `108/0` sample frequency는 correctness/warm-up/idle을 포함한 sample-count
분포이므로 split residency가 아니다. `decode_ready_queue_depth`는 관측 구조상
항상 0이어서 독립 queue의 증거가 아니다.

## 후속 사용 원칙

R1은 correctness fixture와 P1 instrumentation-parity workload의 출발점으로만
사용한다. “4개 개선/0개 열세/3개 혼재”, “decode-heavy에서 dual architecture가
10% 향상”은 인용하지 않는다.
