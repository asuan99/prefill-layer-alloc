# R2 experiment roadmap

최종 갱신: 2026-07-23

## 공통 방법

- 먼저 B1 sustainable SLO rate `lambda*`를 모델별로 측정한다.
- configuration당 최소 5회, paired CI가 0을 교차하거나 variance가 크면 10회
  이상 수행한다.
- 모든 pair는 동일 immutable trace/hash, workload seed, server seed를 사용하고
  node 내 실행 순서를 randomize한다.
- warm-up/correctness/benchmark phase를 분리한다.
- CUDA Graph, backend, GPU clock/power, KV capacity, max-running을 고정한다.
- mean, median, SD, paired bootstrap 95% CI와 percent effect를 보고한다.
- request의 TTFT와 request 내부 token-level ITL p95가 모두 SLO 이하여야 primary
  goodput에 포함한다. 기존 request-mean-ITL score는 secondary로만 보존한다.
- 3% 미만 차이는 headline improvement로 사용하지 않는다.

## 단계와 stop/go gate

### P0 — 상태/R1 정정

`PROJECT_STATUS.md`, claim matrix, R1 재분석, stale banner를 정본으로 반영한다.

### P1 — 계측 observer effect

동일 fixed split에서 다음 네 arm을 paired AB/BA로 비교한다.

1. legacy, telemetry off
2. legacy, symmetric telemetry on
3. R1 observer state on, trace off
4. R1 observer state와 trace on

각 overhead effect가 3% 미만이고 CI가 0을 포함해야 진행한다. 실패 시 buffer,
sampling, writer를 수정하고 architecture 비교를 보류한다.

### P2 — Architecture

- legacy fixed D24/D44
- true dual fixed D24/D44

Decode-heavy/alternating에서 decode progress, ITL 또는 oldest queue age가
유의하게 개선되고 throughput regression이 3% 이하일 때만 Claim D를 채택한다.
그렇지 않으면 true dual은 negative architecture result로 남긴다.

### P3 — Offline profile/estimator

- CUDA Graph on, 실제 full-model decode
- D16/D24/D34/D44/D108
- batch 1/4/8/16/32/48
- context 256/1K/4K/8K, 지원 시 16K
- point당 warm-up 후 30 step 이상, 5회 이상

Profile은 model/revision/config hash, engine commit, GPU/driver/backend/graph,
attention/SSM count와 ratio, GQA metadata, latency percentile/residual을 기록한다.

Acceptance:

- upper-bound empirical coverage ≥95%
- under-reservation epoch ≤1%
- median over-reservation ≤한 state
- unseen/stale profile은 D44 fallback, live violation은 D108/admission limit

### P4 — Baseline와 profile ablation

| ID | Policy |
|---|---|
| B0 | vanilla continuous batching |
| B1 | training workload에서 선택한 one global static |
| B2 | per-workload offline best static oracle |
| B3 | 기존 single-worker dynamic |
| B4 | true dual fixed |
| B5 | true dual generic dynamic |
| B6 | true dual Hybrid-informed dynamic |
| B7 | layer-granular negative baseline |
| B8 | future trace와 transition/dwell cost를 아는 offline oracle |

Profile ladder는 no profile, model-size only, attention-ratio aware, full Hybrid
profile 순으로 동일 held-out point/trace에서 평가한다.

Claim E acceptance:

- target 영역에서 B6가 B1/B5보다 paired CI 기준 유의하고 ≥3% 개선
- non-target 영역에서 >3% regression 없음
- B2/B8은 upper bound로만 보고하며 이를 이긴다고 주장하지 않음

### P5 — Mechanism

대표 target/non-target point에서 Nsight Systems/Compute로 kernel timeline,
GPU idle gap, stream/event overlap, graph replay, SM active, occupancy, Tensor Core,
DRAM/L2를 수집한다.

### P6 — Workload

| ID | 고정 workload |
|---|---|
| W1 | input 2K/output 128, Poisson 0.60 lambda* |
| W2 | input 8K/output 64, 10초 4× burst+30초 drain, 3 cycles |
| W3 | input 256/output 512, 0.80 lambda* |
| W4 | W2형/W3형 30초 phase를 3 cycles |
| W5 | 1K/8K context 교대, 지원 시 16K |
| W6 | input 2K, output 32/512 교대 |
| W7 | 0.20 lambda* |
| W8 | 0.90/0.95 lambda* |
| W9 | 1.10/1.25/1.50 lambda* |

ShareGPT와 현재 cache된 LongBench를 우선 사용한다. coding/agentic trace는
출처·license·전처리 규칙이 확정된 뒤 추가한다.

Applicability map은 average combined demand, peak-minus-average demand,
prefill/decode pressure temporal correlation으로 만들고 B6−B1 goodput effect를
색으로 표시한다. `peak sum>1`, `average≤1`, pressure 교대 영역을 사전 정의한
target으로 사용한다.

## Controller ablation

| Ablation | 검증 claim/mechanism |
|---|---|
| dual worker 제거 | D |
| dynamic 제거/fixed | architecture 대 policy |
| profile 및 feature ladder | A, E |
| runtime context 제거 | A |
| ITL slack 제거 | E의 SLO protection |
| feasibility gate 제거 | unsafe downshift |
| hysteresis/dwell 제거 | oscillation |
| safety margin 제거 | under-reservation |
| static floor | runtime load term |
| layer-level switching | B |
| CUDA Graph off | B와 operating-point sensitivity |
| chunked prefill on/off | orthogonal composability |

## Controller defaults

- steady: D16/D24/D34/D44; emergency D108
- evaluate every `max(4 decode iterations, 100 ms)` or bucket change
- immediate safe-boundary upshift on ITL violation or 85% KV/batch occupancy
- downshift only below 0.75×SLO for 3 epochs
- dwell `max(8 decode steps, 200 ms)`; upshift exempt
- D108 risk 또는 occupancy 90%가 2 epochs 지속되면 admission 제한
