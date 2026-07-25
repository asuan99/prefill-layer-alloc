# Claim–evidence matrix

최종 갱신: 2026-07-25

| Claim | 현재 판정 | Existing evidence | Missing evidence | Required experiment |
|---|---|---|---|---|
| A. Hybrid composition, context, active load에 따라 decode demand가 변한다 | 부분 지지 | Zamba2 context knee; synthetic/ShareGPT의 best split 이동; 다중 모델 batch/context characterization | 동일 CUDA Graph 운영점의 joint surface, GQA/composition 통제, held-out accuracy | P3 full-model profile, feature ladder, leave-one-workload/model-family-out |
| B. layer-level reconfiguration은 ITL critical path와 CUDA Graph를 훼손한다 | 강한 지지, 현 구현 범위 한정 | coordinated TPOT 약 42→124 ms, 최적화 후 약 85 ms; sub-step drain; graph incompatibility | 다중 모델 반복과 timeline attribution | B7 반복, CUDA Graph on/off, Nsight synchronization timeline |
| C. decode starvation은 TTFT도 악화시킨다 | running-batch 경로 강함; KV 경로 부분 | D16 TTFT 7.24 s/ITL 61.9 ms 대 D24 1.21 s/39.9 ms; admission capacity 관측 | time-aligned KV occupancy와 admission reason | D16/D24 paired replay, structured KV/full/mamba occupancy, mediation timeline |
| D. execution-state separation은 single-worker coupling을 줄인다 (★2026-07-24 코드 리뷰로 scope 축소, 아래 "주장 제한" 참조) | 미검증 | R1은 observer라 해당 증거가 아님; 2026-07-24 읽기 전용 코드 리뷰([`../r2_decoupling_review_2026-07-24.md`](../r2_decoupling_review_2026-07-24.md), file:line 근거)로 `PDMUX_TRUE_DUAL_WORKER=1`의 구조 확인: 두 host issue thread/role별 task queue/immutable `ExecutionContext`/thread-local role(ContextVar)만 분리하는 **control-plane dual-worker**이며, running batch(`max_running_requests`)·KV/mamba pool·SM 파티션(`SharedGpuArbiter` 단일 `stream_index`, ≤108)은 **전면 공유** | 실제 두 host loop에서의 fixed-split 비교(coupled ceiling 내); GPU correctness 동치 테스트(현재 없음); admission latch(`r2_admission_limited`) stale-True 버그 수정; results/r2_eval 캠페인 실행(현재 미생성) | legacy fixed 대 true dual fixed, 동일 telemetry/seed/graph — coupled ceiling(+2%, PROJECT_STATUS/CONSENSUS §1-20) 내에서만 유의미, "얽힘 깨기"로 측정 불가(§1-4 死因의 substrate가 구성상 불변) |
| E. Hybrid-informed decode floor가 generic/global static보다 높은 SLO goodput을 낸다 | 미검증 핵심 가설 | 없음 | architecture control, generic policy, static/oracle, profile generalization | B0–B8, P4 profile ablation, W1–W9 및 real trace |
| F. short-ctx conflict-regime 워크로드에는 동적 제어가 이길 수 있는 disjoint-feasibility escape hatch(어떤 static도 두 phase 동시 SLO를 못 만족하는 워크로드)가 있다 | **강한 지지(범위 한정) — CONFIRMED closure, scoped negative(2026-07-25)**: escape hatch **없음**을 확정 | `g2_0_full`(n=4, razor-thin real disjoint 최초 관측)→`g2_0_hard`(n=6–10, claims-auditor 재채점, ILL-POSED at rA5)→`g2_0_decliff`(rA2 n=6, PLAUSIBLE closure)→`g2_0_rasweep`(120 job, off-cliff band rate≤2.75 disjoint 재확인 없음)→`g2_0_raconf`(pre-registered 24-job 확증 열, rate{3.5,3.75}×{d44,d54}×n6, companion-collapse 결정규칙 충족: rate3.5 d44 0.953±0.035≈d54 0.948±0.035; rate3.75 d54 0.948±0.062>d44 0.932±0.042) | long-context(decode floor 상승 영역, CONSENSUS §1-5) 재검증; hot varying-trace(drain 아닌 entangled 조건)에서의 직접 실증 | long-context G2.0-style disjoint sweep(모델/ctx 교체 필요); §1-20 spatial coupling-tax와 결합한 재검토 |

## 주장 제한

- Claim B는 A100/SGLang green-context implementation에 한정한다.
- Claim C는 KV telemetry 전까지 “shared running-batch/capacity congestion”으로
  표현하고 KV causal chain을 확정하지 않는다.
- Claim D는 true dual fixed가 architecture gate를 통과한 뒤에만 사용한다.
  ★**2026-07-24 코드 리뷰 정정**(engine-porter, 읽기 전용,
  [`../r2_decoupling_review_2026-07-24.md`](../r2_decoupling_review_2026-07-24.md)):
  현재 `PDMUX_TRUE_DUAL_WORKER=1` 구현은 **control-plane dual-worker**다 — 두
  host issue thread, role별 task queue, immutable `ExecutionContext`,
  thread-local role(ContextVar)만 분리한다. running batch(`max_running_requests`,
  완료 prefill을 같은 batch로 in-place merge), KV/mamba pool, SM 파티션
  (`SharedGpuArbiter`의 단일 `stream_index`, ≤108)은 **전면 공유**된다(92 prefill
  SM + 24 decode SM = 116의 별도 device pool이 아니다). Claim C의 死因 얽힘이
  사는 substrate(공유 running-batch+KV)를 이 구현은 **구성상 깰 수 없다** —
  관측되는 win/loss는 host-thread overlap(control-plane)에 귀속되며, coupled
  ceiling(+2%, PROJECT_STATUS/CONSENSUS §1-20)을 초과할 수 없다. 그 위의
  +16% headroom은 별도 device pool disaggregation과 hybrid(mamba conv/ssm)
  state transfer를 요구하며 **둘 다 미구현**(state-transfer 경로 코드에 전무).
  따라서 **Claim D는 "control-plane coupling 감소"로 스코프를 축소**하고,
  "얽힘을 깬다"는 프레이밍으로 쓰지 않는다. R2는 GPU correctness gate를
  통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성, `architecture=true_dual`
  telemetry 전무). admission latch(`r2_admission_limited`)에 known-latent
  stale-True 버그(split batch가 None으로 배수되면 재평가 경로 없어 latch가
  True로 고착, clear 경로 부재)가 file:line 근거로 확인됐다 — **사용자 결정으로
  현재 수정하지 않고 보류**한다. 판정은 여전히 **미검증**이며, 이는 성능 판정이
  아니라 코드-구조 사실이다.
- Claim E는 B6가 B1과 B5를 모두 유의하게 이긴 경우에만 사용한다.
- D/E가 실패하면 A–C의 characterization 및 negative result를 논문의 중심으로
  유지한다.
- Claim F는 {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4,
  triton attn+mamba, disable-radix-cache, cudagraph-ON, A100 108-SM
  green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase drain된
  순차 2-phase, rate_A≤3.75}에 한정한다. **"hybrid엔 disjoint 없음"으로
  일반화하지 않는다.** Phase-A frac_good≈0.95는 metric cliff(웜업성 tail-event)
  가 결정해 magnitude는 run-length 의존·ill-posed이나, 순위(d54≈d44, d54
  미선-배제)는 견고하다. §1-20(spatial coupling-tax, disaggregation +16%
  headroom)과는 별도 축 — "단일 static으로 시간축 커버 ⟹ coupling tax 없음"으로
  새지 않는다. 상세
  [`../../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](../../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).
