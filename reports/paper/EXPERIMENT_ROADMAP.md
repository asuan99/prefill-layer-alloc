# R2 experiment roadmap

최종 갱신: 2026-07-25

## 벡터1 (disjoint conflict-regime escape hatch) — 별도 트랙, CONFIRMED closure (scoped, 종결)

이 항목은 P0–P6 Claim D/E gate 배선 밖의 별도 트랙이었다(`reports/CONSENSUS.md`
§5-8(c) 추적, `PROJECT_STATUS.md` "벡터1" 절, `reports/paper/CLAIM_EVIDENCE_MATRIX.md`
Claim F 참조). **2026-07-25 CONFIRMED closure(scoped)로 종결** — 아래 5단계
sweep 계열의 최종 판정.

1. **g2_0_full**(2026-07-24, 1차 스윕) — razor-thin disjoint(feasible-A ∩
   feasible-B = ∅) 발견.
2. **g2_0_hard**(2026-07-24/25, hardening 재스윕) — **ILL-POSED at rA5**: 1차
   disjoint가 재현되지 않음(TTFT 3s-cliff bimodality), "disjoint 소멸" 관측은
   별도 ITL-p95 percentile-window 아티팩트로 판명.
3. **g2_0_decliff stage-1**(jobs 863880–863948) — `rA{2,3,3.5,4}×{d16,d44,d54}`
   스캔, `rA=2`만 clean off-cliff(n=6 확증). 유일한 clean 지점에서 static `d54`가
   양 phase 동시 커버(`feasible-A={d16,d44,d54} ∩ feasible-B={d54} = {d54} ≠ ∅`)
   하나 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증 3항목: off-cliff
   에서도 살아있는 split→TTFT gradient·d54 배제 onset 미측정 전이대·
   "binding-A⟺on-cliff" 미증명).
4. **g2_0_rasweep**(120 job, `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`) —
   off-cliff sub-band(rate≤2.75)에서 disjoint 재확인 없음, 전이대를 rate
   3.0–3.5로 좁힘.
5. **g2_0_raconf**(claims-auditor pre-registered 24-job 확증 열, `rate{3.5,3.75}×
   {d44,d54}×n6`) — 결정 규칙(어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
   d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
   d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)를 **companion
   collapse로 판정**: rate3.5 d44 0.953±0.035≈d54 0.948±0.035(d54 failTTFT=0/6);
   rate3.75 d44 0.932±0.042 < **d54 0.948±0.062**(d54가 더 높음). REOPEN 전제
   양쪽 붕괴 — d54는 어느 rate서도 <0.7이 아니고 d44도 어느 rate서도 견고히
   ≥0.95가 아님(웜업성·split-대칭적 TTFT-blowup). d54는 Phase B 유일 feasible
   split이면서 Phase A도 d44와 대등하게 커버 → 단일 split이 양 phase를 시간축
   에서 커버 → **disjoint 없음, 최종 확정**.

★**필수 caveat**: magnitude는 ill-posed(metric cliff, run-length 의존)이나
**순위(d54≈d44, d54 미선-배제)는 견고**. Phase-B d44 0.188은 50ms 경계 바로 위라
magnitude fragile·방향 견고. scope는 {Zamba2-2.7B, ctx4096, Phase A in2048/o32,
Phase B in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON,
A100 108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75}에 한정 — **"hybrid엔 disjoint 없음"으로
일반화 금지**. closure는 얽힘 억제(drain) 조건 관측 = 필요조건 bound이지 hot
varying-trace(Claim C 얽힘) 실증 아님.

이 트랙의 결과는 Claim D/E나 §1-20(spatial coupling-tax)에 영향을 주지 않는다 —
시간적 disjoint와 공간적 coupling-tax는 별개 축. **남은 방향(후속, 미실행)**:
(i) long-context(decode floor가 ctx 상승에 따라 올라가는 영역 — CONSENSUS
§1-5 — 충돌이 발생할 수 있음, 모델/ctx 교체 필요), (ii) §1-20 spatial
decoupling(별도 device pool disaggregation, +16% headroom). 상세 verdict:
`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`,
`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`.

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
