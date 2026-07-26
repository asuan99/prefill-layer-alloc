# `prefill-layer-alloc` project status

최종 갱신: 2026-07-26 (Stage 0[long-ctx L−2 게이트]: 운영점 decode SM-무감각을
hybrid·16k ctx까지 확장 확인 — non-binding, long-ctx 충돌 가설 이 regime서 붕괴,
HE0/벡터1 ctx-무관으로 강화). 이전: 2026-07-25 (벡터1[G2.0 short-ctx disjoint
conflict-regime]: CONFIRMED closure, scoped — narrow-rA 확증 sweep g2_0_raconf
완료; 같은 날 논문 positioning 판정[multiplexing 신규성 축 + Transformer-control
게이트, cross-substrate 이식 프레이밍 철회] "다음 실험 gate" #6 추가). 이 문서가
프로젝트 전체의 유일한 현재 상태
정본이다. 이전 문서와 충돌하면 이 문서와
[`reports/paper/`](reports/paper)의
판정을 우선한다.

## 논문 방향

Layer composition을 runtime scheduling boundary로 사용하지 않는다. Hybrid
구조는 coarse-grained decode floor를 예측하는 offline model profile로만
사용한다.

- **H-Architecture:** 역할별 queue, host issue loop, CUDA stream을 실제로
  분리하면 single-worker control-plane coupling을 줄일 수 있다.
- **H-Policy:** model profile과 runtime load로 예측한 decode floor가 시간적으로
  상보적인 near-saturation workload에서 generic dynamic 및 global static보다
  높은 SLO goodput을 제공한다.

두 가설은 아직 검증되지 않았다. 구현 완료와 성능 주장 성립을 구분한다.

## 확정된 결과

1. 여러 Hybrid 모델에서 prefill/decode resource separation은 fused execution보다
   유리한 operating point를 제공한다.
2. 현재 A100/SGLang green-context substrate에서 layer-boundary resource
   switching은 sub-step drain과 synchronization을 일으켜 decode TPOT을 약
   `42→124 ms`로 악화시켰다. 최적화 후에도 약 `85 ms`였다.
3. decode starvation은 active sequence 체류시간과 shared running-batch capacity를
   증가시켜 prefill admission과 TTFT를 악화시킨다. 대표적으로 D16은 D24보다
   prefill SM이 많지만 TTFT가 `7.24 s` 대 `1.21 s`였다.
4. 적절한 static split은 workload와 context/load에 따라 이동한다.
5. 기존 single-worker SLO-aware/binding-first dynamic은 valid varying trace에서
   best static을 넘지 못했다.

KV congestion은 plausible mechanism이지만 기존 artifact에 구조화된 KV occupancy가
없어 아직 독립적인 causal claim이 아니다.

### 벡터1 (G2.0 short-ctx disjoint conflict-regime escape hatch) — CONFIRMED closure (scoped)

6. short-ctx band(아래 scope)에는 동적 제어가 이길 수 있는 disjoint-feasibility
   conflict regime(어떤 static도 두 phase를 동시에 못 커버하는 워크로드)이 **없다**.
   `reports/CONSENSUS.md` §5-8(c) 미결 갈래 (c)를 최종적으로 닫는다.

실험 계열(2026-07-24 실행 시작·2026-07-25 최종 판정): `reports/CONSENSUS.md`
§5-8(c)("충돌 regime 워크로드" — 동적이 이길 disjoint-feasibility escape hatch가
있는가)를 n≥4로 재검증하는 G2.0 short-ctx 스윕(Zamba2-2.7B). 1차 라운드
(g2_0_full/g2_0_hard)는 **ILL-POSED at rA5**로 판정됐다: g2_0_full이 찾은
razor-thin real disjoint(feasible-A={d16,d44} ∩ feasible-B={d54}=∅)는 g2_0_hard
hardening 스윕에서 재현되지 않았고(TTFT 3s-cliff bimodality, n=10 pool 시 d44/d54
둘 다 ~0.86–0.90로 통계적 구분 불가), "disjoint 소멸" 관측은 별도의 ITL-p95
percentile-window 아티팩트였다.

**de-cliff stage-1(jobs 863880–863948) 완료**: `rA{2,3,3.5,4}×{d16,d44,d54}`를
스캔해 `rA=2`만 clean off-cliff임을 확인(`rA≥3`은 전부 여전히 bimodal)하고
`rA=2`를 n=6으로 확증 — 유일한 clean off-cliff 지점에서 static `d54`가 양
phase를 동시 커버, 단 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증
3항목: off-cliff에서도 살아있는 split→TTFT gradient·d54 배제 onset이 미측정
전이대·"binding-A⟺on-cliff" 미증명).

**narrow-rA 확증 sweep 완료 — CONFIRMED로 승격**: `g2_0_rasweep`(120 jobs,
`rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`)이 off-cliff sub-band
(rate≤2.75)에서 disjoint 부재를 재확인해 전이대를 rate 3.0–3.5로 좁혔고,
그 창을 겨눈 **pre-registered 24-job 확증 열 `g2_0_raconf`**(rate{3.5,3.75}×
{d44,d54}×n=6, 결정규칙: 어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)가 **companion
collapse로 판정**:

| rate | split | frac_good mean±SD (n=6) | 비고 |
|---|---|---|---|
| 3.5 | d44 | 0.953 ± 0.035 | |
| 3.5 | d54 | 0.948 ± 0.035 | failTTFT=0/6 (Phase-A도 d44와 통계적 동률) |
| 3.75 | d44 | 0.932 ± 0.042 | |
| 3.75 | d54 | **0.948 ± 0.062** | **d54가 d44보다 높음** |

REOPEN 전제 둘 다 붕괴(d54는 어느 rate서도 <0.7이 아니고, d44도 어느 rate서도
견고히 ≥0.95가 아님: rep 하나가 warm-up성 TTFT-blowup으로 0.844–0.875까지
떨어짐 — 이 blowup은 **split-대칭적**이라 disjoint를 만들지 않음). d54는 Phase B의
유일 feasible split(d44 ITL-p95 50.7ms로 50ms SLO 초과, `frac_good` 0.188;
d54는 44.1ms, `frac_good` 1.000, SD=0)이면서 Phase A도 d44와 대등하게 커버 →
단일 split(d54)이 양 phase를 시간축에서 커버 → **disjoint 없음, 최종 확정**.
상세 per-rep 표·기전·caveat:
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

★**필수 caveat(overclaim 방지)**: **magnitude는 ill-posed, 순위는 견고** —
Phase-A frac_good≈0.95는 웜업성 TTFT/ITL tail-event(metric cliff)가 결정해
run-length 의존이나, "d54≈d44·d54 미선-배제"라는 **순위**는 견고하다. Phase-B
d44 0.188은 50ms 경계 바로 위라 magnitude는 fragile하나 방향(d44는 decode 못
커버)은 견고하다.

**scope 한정(필수)**: {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B
in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON, A100
108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75} — **"hybrid엔 disjoint 없음"으로 일반화
금지**. **drain caveat**: closure는 얽힘 억제(drain) 조건 관측 = 필요조건
bound이지 hot varying-trace(Claim C 얽힘) 실증 아님. Claim D/E와 §1-20(spatial
coupling-tax, 92+24=116>108, disaggregation +16% headroom)에는 영향 없음 —
시간적 disjoint(단일 static이 시간축에서 양 phase를 커버)와 공간적
coupling-tax는 별개 축이며 "단일 static으로 충분 ⟹ coupling tax 없음"으로
새지 않는다.

**남은 방향(벡터1 종결이 열어두는 것)**: (i) **long-context**(decode floor가
ctx 상승에 따라 올라가므로 — CONSENSUS §1-5 — 충돌이 발생할 수 있는 영역).
★**2026-07-26 갱신**: 이 방향의 전제(운영점서 decode floor가 ctx로 상승해
binding해지는가)를 Stage 0 게이트로 검증 — **ctx≤16k에서는 상승하지 않는다**(아래
"Stage 0" 절). (ii) **§1-20 spatial decoupling**(별도 device pool disaggregation,
+16% headroom)은 아직 실행되지 않았다.

상세 verdict:
[`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`](workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md),
[`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

### Stage 0 (long-ctx L−2 게이트, 2026-07-26) — 운영점 decode SM-무감각, hybrid·16k로 확장

7. 운영점(cudagraph-ON, green-context pdmux)에서 decode ITL은 decode-SM(16→108,
   6.75×)에 **무감각(non-binding)**하다 — pure-Transformer(Qwen2.5-3B, 양성
   대조)·pure-Mamba(Mamba2-2.7B, 음성 대조)·hybrid(Zamba2-2.7B, 타깃) **전부**,
   ctx {4k, 8k, **16k**} 전부에서. 유일한 de-confounded 대조 **D16 vs D108(prefill
   경합 0인 두 점) = 1.00 ± 0.01, 3 arm × 3 ctx 전부**.

실험(jobs 864230[H/T]·864601[M], `workspace/engine-port/results/stage0_xctrl/`,
PIN_CHECK 전부 PASS): raw coupled ITL(D) 스윕(D16/D44/D92)은 **CONFOUNDED로
판정**됐다 — D16이 D108과 9셀 전부 ≤0.5% 동일(6.75× SM 증가가 무이득), 최속점 D92는
비단조(prefill을 16 SM으로 굶기는 지점), 그리고 **음성 대조 M**(decode가 O(1)
recurrent라 원리상 SM-bound 불가)이 H와 동형의 "민감도"(2.4×대)를 보이는 것 자체가
그 곡선이 decode-SM이 아니라 prefill 경합/batch-entanglement를 재고 있다는 증거다.
D108 무경합 앵커만이 이 confound를 우회한다.

**판정**: `reports/longcontext_trace_plan.md` §6이 사전 등록한 L−2 게이트의
**"게이트 실패이자 강한 결과"** 분기가 실현됐다 — long-ctx 충돌 가설(H_L4 시간축·
H_L5 공간축)은 이 regime에서 필요한 decode-floor 운영점 상승을 얻지 못해 **붕괴**하고,
**HE0/벡터1(동적이 best-static을 못 넘음)은 ctx-무관으로 강화**된다. CONSENSUS의
HE2(운영점 decode 비-binding)·r0c(no-cudagraph decode-floor micro-측정)와 정합·확장
관계이며, 이는 **아크의 반전이 아니라 기존 결론의 강화**다.

★**필수 scope**: {M/H/T 2.7–3B, triton attn+mamba, cudagraph-ON green-context
pdmux, ctx≤16k, 이 coupled 하네스, one-shot 32-conc burst}. **더 큰 모델·>16k·다른
substrate는 미측정.** coupled 스윕의 "민감도 magnitude"는 confound로 측정 불가하므로
결론은 **방향(non-binding)만**이다. 완전 de-confound 재측정(prefill-SM 고정+
steady-state rate+occupancy 통제)은 **미실행** — 사용자가 현 증거로 결론 확정을
결정했다. ★**방법론 교훈**: coupled ITL(D) 스윕은 decode-SM과 prefill-SM(108−D)을
공변시켜 confounded되기 쉽다 — **음성 대조**(그 축에 binding 불가능한 arm) +
**무경합 앵커**(공변이 0으로 붕괴하는 특수점)의 조합이 이를 식별했다. 상세
[`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md).

## 철회된 가설

- attention/SSM layer별 static resource partition이 보편적으로 유리하다.
- layer-granular switching이 phase-granular allocation보다 유리하다.
- 기존 single-worker dynamic이 best static을 이긴다.
- 과거 simulation/no-CUDA-Graph 결과의 1.37–2.02× layer-aware goodput 향상이
  현재 real-engine 논문 결과다.
- R1 `PDMUX_DUAL_WORKER=1`이 독립 worker architecture를 구현했다.

## R1 판정

`job_862512`는 **R1 observer-path A/B**다. 기존 scheduler queue/batch의 alias와
logical arbiter를 추가했을 뿐 동일 `event_loop_pdmux`에서 실행됐다. dual arm에만
동기식 JSONL bookkeeping이 있었고 CUDA Graph가 꺼졌으며 server seed가 달랐고
각 arm은 한 번만 실행됐다.

따라서 “4 improved / 0 worse / 3 mixed” 판정과 decode-heavy 성능 차이의
dual-worker 인과 귀속을 철회한다. 모든 scenario는 방향성 관측, 통계·인과
미확정이다. 전체 수치는
[`R1_REANALYSIS.md`](reports/paper/R1_REANALYSIS.md)에
보존한다.

## 현재 구현

- `PDMUX_DUAL_WORKER=1`: R1 observer 재현 전용
- `PDMUX_TRUE_DUAL_WORKER=1`: 두 long-lived host issue thread, role task queue,
  immutable execution context, safe-boundary resource lease를 사용하는 R2 경로
- `PDMUX_TELEMETRY_PATH`: architecture와 무관하게 동일한 비동기 telemetry 사용
- versioned `HybridModelProfileV1`, conservative floor estimator, fixed/generic/
  Hybrid controller가 구현되어 `PDMUX_R2_POLICY`로 engine safe-boundary에 연결됨
- W1–W9 deterministic trace, paired campaign manifest, request token-ITL p95
  goodput와 paired bootstrap 분석 도구가 구현됨

True dual 경로는 module-global PD-mux role을 thread-local `ContextVar`로 바꾸는
tracked patch를 필수로 요구하며, 적용되지 않은 runtime에서는 fail-fast한다.
GPU correctness/performance 검증 전에는 production-ready로 분류하지 않는다.
현재 controller의 online ITL p95는 최근 decode-iteration wall-time window의
추정치이며 request token-level p95는 load generator에서 별도로 계산한다.

### 코드 리뷰 스코프 정정 (2026-07-24)

읽기 전용 코드 리뷰([`reports/r2_decoupling_review_2026-07-24.md`](reports/r2_decoupling_review_2026-07-24.md),
engine-porter, file:line 근거)가 확인한 구조: `PDMUX_TRUE_DUAL_WORKER=1`은
**control-plane dual-worker**다 — 두 host issue thread, role별 task queue,
immutable `ExecutionContext`, thread-local role(ContextVar)만 분리한다.
**data/resource plane은 전면 공유**된다: running batch(`max_running_requests`)는
단일 scheduler 속성이고 완료된 prefill을 같은 running batch로 in-place merge하며,
KV/mamba pool도 단일 객체, SM 파티션도 `SharedGpuArbiter`가 하나의
`stream_index`만 추적한다(92+24=116의 별도 device pool이 아니라 ≤108 단일
coupled index). 확정된 결과 §3의 死因 얽힘이 사는 substrate(공유
running-batch+KV)를 이 구현은 **구성상 깰 수 없다** — 관측될 win/loss는
host-thread overlap(control-plane)에 귀속되며, 별도 device pool disaggregation과
hybrid mamba/SSM state transfer가 필요한 headroom에는 도달 불가하다. 이 두
경로는 코드에 **미구현**이다(state-transfer 경로 전무, mamba conv/ssm state
migration 스캐폴딩조차 없음). 따라서 **Claim D는 "control-plane coupling
감소"로 범위를 축소**한다 — "얽힘을 깬다"는 프레이밍으로 쓰지 않는다. R2는 GPU
correctness gate를 통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성,
`architecture=true_dual` telemetry 전무). 부가로 admission
latch(`r2_admission_limited`)에 **known-latent 버그**가 코드 근거로 확인됐다:
split batch가 None으로 배수되면 재평가 경로가 없어 latch가 True로 고착되어
prefill admission을 영구 차단할 수 있다(clear 경로 부재) — **사용자 결정으로
현재 수정하지 않고 보류**한다.

## 증거 수준

| Claim | 상태 |
|---|---|
| A. composition/context/load-dependent decode demand (★2026-07-26 Stage 0: 운영점 decode SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k 전부로 확장 확인 — lever-weakness 강화, 등급 불변) | 부분 지지 |
| B. layer-level reconfiguration의 critical-path 손상 | 강한 지지, 현 substrate 한정 |
| C. decode starvation의 TTFT entanglement | running-batch 경로 강함, KV 경로 부분 |
| D. true dual-worker가 coupling 감소 (★2026-07-24 코드 리뷰로 control-plane 범위로 축소, 위 "코드 리뷰 스코프 정정" 참조) | 미검증 |
| E. Hybrid-informed policy가 generic/static보다 우수 | 미검증 |

## 다음 실험 gate

1. 대칭 telemetry의 observer effect가 3% 미만이고 paired CI가 0을 포함해야 한다.
2. 동일 fixed split에서 true dual이 legacy 대비 decode progress/ITL/queue age를
   개선하며 throughput regression이 3%를 넘지 않아야 Claim D를 채택한다.
3. estimator의 95% upper-bound coverage가 95% 이상, under-reservation epoch가
   1% 이하여야 한다.
4. target applicability 영역에서 proposed가 B1/B5보다 paired CI 기준 유의하고
   effect가 3% 이상이어야 Claim E를 채택한다.
5. 벡터1(disjoint conflict-regime escape hatch, CONSENSUS §5-8(c)): **CONFIRMED
   closure (scoped, 2026-07-25)** — g2_0_full → g2_0_hard → g2_0_decliff →
   g2_0_rasweep → g2_0_raconf(pre-registered 24-job 확증 열, 결정 규칙 충족)로
   short-ctx band(scope는 위 "벡터1" 절 참조)에 견고한 disjoint 없음을 최종
   확정. 더 이상의 게이트 없음(트랙 종결) — 남은 방향은 (i) long-context
   재검증(decode floor 상승 영역, 미실행), (ii) §1-20 spatial coupling-tax/
   decoupled substrate(별도 트랙, 아래 항목 2 참조). 어느 쪽도 아직 실험
   설계·게이트가 없다.
6. **논문 positioning(2026-07-25, venue-strategist prior-art 조사,
   [`reports/paper/venue_positioning.md`](reports/paper/venue_positioning.md)
   §0.1)**: 신규성 축은 disaggregation이 아니라 **co-located multiplexing**
   (DuetServe/MuxWise/SGLang-pdmux/Nexus/Bullet 대조) — 부분적 신규성 실증,
   방어 자산은 DuetServe(libsmctrl·Transformer서 동적 승)와의 정량적 상반.
   negative를 (A) green-context 종속(Claim B, 헤드라인 금지) / (B)
   mechanism-independent 후보(Claim A/C, lever-weakness·entanglement)로 분리.
   ★**green-context = 배포 가능한 유일 vendor primitive(CUDA 12.4+)** →
   "libsmctrl 쓰면 되잖아"는 배포 불가 research curiosity로 반박(약점 아님).
   ⚠️**초판의 "cross-substrate serving 이식 make-or-break" 프레이밍은 철회**
   (MPS=정적·프로세스별, libsmctrl=비-vendor·세대귀속 → 이식 불필요·부적합).
   대신 (B)를 **기존 green-context 위에서** 닫는 vendor-substrate 3수: 새 게이트
   = **Transformer-control 대조**(같은 green-context+SLO, drain 상쇄 → 동적
   flip이 모델서 갈리면 hybrid 귀속 식별) + roofline lever-weakness microbench
   (기존 r0c) + 기측정 entanglement 귀속(switch≈0). long-ctx(위 5번 (i))는
   ctx-regime 경계용(별도 질문). 상세는
   [`reports/paper/EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)
   "벡터2"(TC-series) 절.
7. **long-context Stage 0(L−2) 게이트: 실행 완료, non-binding(2026-07-26)** —
   `reports/longcontext_trace_plan.md` §6이 사전 등록한 make-or-break 게이트.
   운영점서 decode SM-무감각이 hybrid·pure-Transformer·pure-Mamba 전부, ctx≤16k
   전부에서 확인(D16≡D108). **long-ctx L−1 이상(SLO 재정의·모델 교체 baseline·
   시간축/공간축 충돌 스윕)은 이 regime(ctx≤16k, 2.7–3B급)에서는 진행 근거
   없음** — 게이트 설계대로 여기서 멈춘다. 남은 미측정: >16k ctx, 더 큰 모델.
   상세 [`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md).

실험·통계·fallback의 상세 정본은
[`EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)다.
