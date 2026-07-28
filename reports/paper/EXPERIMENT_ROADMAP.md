# R2 experiment roadmap

최종 갱신: 2026-07-28(★★★claims-auditor 감사 — P6의 Stage 0/L−2 게이트 판정
[non-binding]을 **철회**(C1 CONFIRMED: D108 앵커가 실은 decode 16 SM). L−2는
"실행 완료"가 아니라 "게이트 미실행"으로 정정. 대신 8B decode-SM 민감도 측정
노트[C2, scoped]와 그 프론티어 게이트 E1–E4를 추가 — 아래 P6 절 갱신). 이전:
2026-07-26(P6 long-context Stage 0/L−2 게이트 실행 완료 — non-binding,
아래 P6 절 갱신 — ★2026-07-28 철회, 위 참조). 2026-07-25(벡터2 재프레이밍 — cross-substrate serving 이식
[XS-series]을 "불필요·부적합"으로 하향 후 새 게이트 "Transformer-control on
green-context"[TC-series]로 교체, positioning 판정)

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

## 벡터2 (substrate-robustness 식별) — Transformer-control on green-context (2026-07-25)

`reports/paper/venue_positioning.md` §0.1(2026-07-25, venue-strategist
prior-art 조사)의 판정: 이 논문의 central negative는 substrate-robustness 축으로
두 갈래다 — **(A) green-context 종속**(layer-aware 死·cudagraph 비양립, Claim
B) vs **(B) mechanism-independent 후보**(lever-weakness=mamba decode
SM-둔감·entanglement·decode 비대칭, Claim A/C).

⚠️★**2026-07-25 하향·재프레이밍(사용자 지적) — cross-substrate serving 이식
(XS0/XS1/XS2)은 불필요·부적합.** 이 절의 초판은 두 번째 substrate(libsmctrl/MPS)
serving 이식을 "make-or-break 필수 게이트"로 걸었으나 **철회**한다. 이식은
불필요할 뿐 아니라 부적합하다:

- **MPS**: SM 파티션이 프로세스별·정적 → 런타임 동적 PD-mux 불가 +
  단일-프로세스 `event_loop_pdmux`의 멀티-프로세스 전면 재구조화 비용이 실익
  초과.
- **libsmctrl**: NVIDIA 비제공 리버스-엔지니어링(per-arch SM 마스킹) →
  하드웨어 세대·드라이버 귀속(driver-580 BLOCKED가 증거). 배포 근거에 비-vendor·
  비-이식 의존성을 들이는 셈.
- ★**green-context = 배포 primitive 방어**: NVIDIA 공식 fine-grained SM
  primitive는 green-context 하나(CUDA Green Contexts 12.4+)뿐. ⇒ "libsmctrl
  쓰면 되잖아"의 답 = "green-context가 배포 가능한 유일 vendor primitive다.
  DuetServe/Bullet의 동적-승은 libsmctrl(세대 귀속·비-vendor) 위에서만 성립 →
  libsmctrl에서 hybrid 동적이 이겨도 이식 불가한 research curiosity이지 배포
  가이드라인의 반례가 아니다." 따라서 (A) layer-aware 死는 green-context-bound로
  정직히 스코프하고, 그 스코프를 libsmctrl 비-이식성이 오히려 받쳐준다.

따라서 XS0/XS1/XS2(별도 substrate serving 이식)는 **실행하지 않는다.** 대신
Risk 2(모델 vs substrate 귀속)를 **기존 green-context 위에서** 닫는 값싼
식별 실험 3수로 교체한다:

- ★**신규 게이트 = Transformer-control on green-context**: 순수 Transformer
  (예: Qwen/Llama)를 기존 pdmux(green-context)에 통과시켜 hybrid와 **같은
  green-context + 같은 conjunctive-SLO**에서 대조한다. drain 비용은 두 모델에
  동일하게 작용 → **상쇄**. 이 벡터의 ID prefix `TC`(Transformer-Control)는
  P4 baseline ID `B0`–`B8`와 별개다.
  - **TC0**: 순수 Transformer 모델(Qwen/Llama류)을 기존 green-context pdmux에
    배선(모델 로딩·correctness gate).
  - **TC1**: hybrid와 동일 워크로드/SLO/split-grid에서 reactive dynamic vs
    decode-heavy static을 측정.
  - **결정 규칙(사전 등록)**: **Transformer 동적-승 ∧ hybrid 동적-패 → flip은
    substrate·메트릭 고정 하에 모델(hybrid) 귀속 확정**(Risk 2 닫힘, 진짜 식별).
    **둘 다 동적-패 → negative는 hybrid가 아니라 메트릭(conjunctive-SLO
    goodput)+배포-primitive(green-context) 탓으로 재프레이밍**(여전히 유효하나
    다른 기여).
- **보강 (실행 불필요·기존 데이터)**: (2) **lever-weakness = roofline
  microbenchmark**(r0c SM-민감도 데이터 보유) — mamba decode SM-둔감은
  연산강도 성질이라 primitive-robust; "libsmctrl이 고친다"는 반론은
  drain(=(A))에만 닿고 lever(Claim A)엔 안 닿음. (3) **헤드라인 HE0는 이미
  entanglement 귀속으로 측정 완료**(`switch_count`≈0·컨트롤러 0.014% 직접 계측
  → 동적-패가 overhead/drain 탓 아님) → drain-아티팩트 반론은 (A)에만 닿고
  헤드라인 무관.

벡터1(disjoint conflict-regime, 시간축 disjoint-feasibility)과는 무관한 별개
트랙. 벡터2는 이제 파티셔닝 primitive 불변성을 **cross-substrate 이식이 아니라
green-context 위 모델-대조(+기존 microbench/telemetry)**로 식별한다.

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

★**long-context(W5 등)의 논문적 역할(2026-07-25 positioning 판정,
`venue_positioning.md` §0.1(4))**: `longcontext_trace_plan.md`의 long-ctx
트랙(Stage 0/L−2 → L3/L3s)은 negative→가이드라인 전환과 "언제 유효한가" 경계
획정, 그리고 granularity 비용의 모델-composition 독립성(ctx-불변 구조적
성질)을 보이는 데 유효하다. **substrate 귀속(Risk 2)은 long-ctx가 아니라 위
벡터2(green-context 위 Transformer-control 대조 + roofline microbench + 기측정
entanglement 귀속)가 닫는다** — 별도 substrate serving 이식은 불필요·부적합으로
철회됐으므로, long-ctx가 "이식의 대체재"일 필요도 없다. 두 트랙은 서로 다른
질문(long-ctx=ctx-regime 경계, 벡터2=primitive/모델 귀속)을 담당한다.

★★**Stage 0(L−2) 게이트 실행(2026-07-26) — non-binding, ★★★2026-07-28 철회
(claims-auditor 감사, C1 CONFIRMED)**
(`../stage0_verdict_2026-07-26.md`, jobs 864230+864601): 2026-07-26엔 운영점서
decode SM-무감각이 hybrid·pure-Transformer·pure-Mamba 전부, ctx≤16k 전부로
확인됐다고 기록했으나, 근거였던 "D108 무경합 앵커"가 실은 decode 16 SM이었음이
3중 독립 증거(코드 기전·telemetry 재집계·클라이언트 서명)로 확인돼 **판정을
철회**한다. **L−2는 SM-binding 여부를 측정한 적이 없다** — "negative→가이드라인
전환" 역할은 실현되지 않았고, `longcontext_trace_plan.md` §6의 L−1 이상은
"게이트 실패로 보류"가 아니라 **"게이트 미실행"**이다.

★★**대신(2026-07-28) 8B decode-SM 민감도 측정 노트 — C2 CONFIRMED(scoped)**
(`../../workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`,
jobs 865289–865533): prefill을 16 SM에 고정한 채 decode-SM만 16→92로 올리면
decode ITL이 **2.36–2.91×**(4 arm, 모델-무관) 개선된다 — Stage 0가 주장하던
"SM-무감각"과 정반대 방향. 단 이는 **decode 측 등량곡선**(예산 제약
`prefill+decode≤108` 없음)이라 **레버 존재만 확립**하며 정책 이득 근거가
아니다. **claims-auditor가 지정한 프론티어 게이트 E1**(`[108−D,D]` 스윕,
D∈{16,24,44,54,92}+best-static 대조, 4 arm, offered-rate 고정, n≥4, 사전등록
파티션 점유율≥0.80·활성률≥0.60 게이트, 결정규칙: best static 대비 conjunctive
goodput ≥3% 개선 & paired CI가 0 배제)가 이 레버가 예산 제약 하 net-positive인지
판정한다 — 병행 게이트 E2(ctx 확장)·E3(duty-cycle, 설계상 종결)·E4(C2b는 통제
불가하므로 주장 폐기). 상세 `../PROJECT_STATUS.md` "8B decode-SM 민감도 측정
노트"·"열린 긴장"·"다음 실험 gate" #8.

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
