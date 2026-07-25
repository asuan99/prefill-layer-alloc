# Session handoff — 2026-07-25

## 이번 세션 요약

사용자 지적("계획된 단일-GPU 작업은 이전 경향을 반복할 것 + decoupling 고려 필요")에서 출발해
두 트랙을 병행: **(1) 현재 decoupling 구현체(R2 true-dual-worker) 코드 리뷰** → control-plane
전용임을 확정하고 Claim D 스코프 축소, **(2) HE0-reopen 벡터1(short-ctx disjoint 충돌-regime)을
서빙 실증으로 종결**(pre→full→hard→de-cliff→rasweep 120job→raconf 24job, claims-auditor 3회).
매 결론 후보마다 감사가 over-read를 잡아(razor-thin→ill-posed→구조논증 반려→bimodality-masking)
**벡터1을 CONFIRMED closure(scoped)로 정직하게 종결**. 이어 **long-context 트랙 설계를 완성**하고
SLO 방법론 문서를 교정. 전부 로컬 커밋(push 안 함). long-ctx 실험 착수는 **pending**(사용자 지시).

## 결정·측정

### R2 decoupling 구현체 리뷰 (engine-porter, 읽기전용)
- `PDMUX_TRUE_DUAL_WORKER=1` = **control-plane dual-worker 뿐**: 두 host thread/role queue/immutable
  ExecutionContext/thread-local role만 분리. **running batch·KV/mamba pool·SM(SharedGpuArbiter 단일
  stream_index ≤108) 전면 공유** → §1-4 얽힘을 **구성상 못 깬다**. state-transfer/hybrid SSM state
  migration **전무**. GPU correctness gate 미통과, `results/r2_eval/` 실제로 비어있음. admission latch
  stale-True 버그 CONFIRMED(미수정·보류).
- 결정: **Claim D를 "control-plane coupling 감소"로 스코프 축소**, 판정은 미검증 유지. "+16% headroom"은
  별도 device pool disaggregation + hybrid state transfer(둘 다 미구현) 요구.
- 보고서: [`reports/r2_decoupling_review_2026-07-24.md`](../reports/r2_decoupling_review_2026-07-24.md)

### 벡터1 (short-ctx disjoint 충돌-regime) — CONFIRMED closure (scoped)
전부 Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4, cudagraph-ON, triton, radix off,
max-running 48, drained 2-phase, **정적 split만(동적/aging/policy 없음)**. canonical goodput = TTFT≤3s ∧
per-req token-ITL-p95≤50ms.

- **G2.0-pre(프로브)**: 절벽/용량 규명 — Phase A rate6서 절벽, Phase B는 d24서 **ITL이 먼저 터짐**
  (decode-starved). 측정점 rA5/rB4 확정. [`results/g2_0_pre/probe_analysis_2026-07-24.md`]
- **G2.0-full(d16–d54×n4)**: razor-thin disjoint → **감사①: disjoint도 소멸도 같은 cliff 노이즈,
  rA5 측정이 ill-posed**(d44/d54 순위 sweep 간 반전). [`results/g2_0_full/disjoint_verdict_2026-07-24.md`]
- **G2.0-hard(OB1024 + d64/d74)**: disjoint "소멸"이 **ITL-p95 percentile-window 아티팩트**(긴 output이
  ITL 중앙값↑·p95↓)임을 감사가 규명. [`results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`]
- **G2.0-decliff(rA{2,3,3.5,4}×{d16,d44,d54})**: rA2만 off-cliff, 거기선 **Phase A non-binding**(전 split
  통과) → disjoint 없음. **감사②: 구조논증만으론 못 닫음**(rA∈(2,3) 격자 구멍, d54 gradient 살아있음
  t≈7.2). [`results/g2_0_decliff/decliff_verdict_2026-07-25.md`]
- **G2.0-rasweep(rA{2.25…3.25}×{d16,d34,d44,d54}×n6, 120job)**: off-cliff band(≤2.75) disjoint 없음 확정.
  **감사③: analyst "d54 p90 1556ms"가 bimodality-masked**(worst-rep 2771ms, cliff-adjacent) → 24-job 확증 열 필수.
- **G2.0-raconf(rA{3.5,3.75}×{d44,d54}×n6, 24job)**: per-rep 판정 = **companion collapse**. rate3.5 d44
  0.953≈d54 0.948(동률), rate3.75 **d54 0.948 > d44 0.932**. d54가 홀로 먼저 배제되지 않음(REOPEN 두 전제
  독립 반증). [`results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`]
- **판정: 벡터1 CONFIRMED closure(scoped)** — magnitude는 tail-determined/ill-posed지만 **순위(d54 미선-배제)는
  견고**. scope: {Zamba2-2.7B, ctx4096, 이 워크로드, drained, cudagraph-ON, rate≤3.75}. "hybrid 일반화" 금지.
  **§1-20 공간적 coupling-tax 방화벽 유지**(별개 축, 열려있음). 정본 = CLAIM_EVIDENCE_MATRIX **Claim F**.

## 코드·문서 변경 (전부 커밋됨)

- `681029a` — R2 리뷰 → Claim D 스코프 축소(PROJECT_STATUS/CONSENSUS/CLAIM_EVIDENCE_MATRIX + r2 review 보고서).
- `7a9196b` — 벡터1 ill-posed(cliff-bound) 기록(de-cliff 전).
- `15281a8` — **벡터1 CONFIRMED closure(scoped) 정본화**(PROJECT_STATUS/CONSENSUS §5-8(c)/CLAIM_EVIDENCE_MATRIX
  Claim F/EXPERIMENT_ROADMAP) + **long-context 설계 갱신**([`reports/longcontext_trace_plan.md`](../reports/longcontext_trace_plan.md):
  §0.5 신설·§4.3 SLO 교정·§6 L−2/L3s·§2 H_L5) + verdict 보고서·sbatch·manifest.
- `0d986ec` — campaign runtime scratch(driver/sentinel/job-map) gitignore, 트리 clean.
- 메모리(repo 밖): `MEMORY.md` + `slo-aware-scheduling-track.md`에 벡터1 종결 반영(doc-steward, 세션 중).

### long-context 설계 완성 (핵심)
- **기전(grounded, r0c ctx16384 micro)**: decode floor가 ctx로 상승 — attn 9층이 16k서 79% 지배·SM 급민감
  (108→16 SM = 6.5× 팽창), mamba 54층 O(1). short-ctx는 decode floor 낮아(54 SM) 충돌 없음(=벡터1 종결 이유)
  → **long-ctx는 그 원인 정조준.**
- **시간축(H_L4, 동적) vs 공간축(H_L5 신설, §1-20 증폭)** 분리. short-ctx §1-20=116>108(≈8초과) →
  **long-ctx ~180>108(≈72초과)** = decoupling headroom 대폭 확대. hybrid state-transfer(mamba O(1)+KV O(L)) 훅.
  **무게중심 = L3s(공간축).**
- **SLO 방법론 교정(venue-strategist)**: 초안 `TTFT_SLO(L)=a+b·L`(ad-hoc) 폐기 → **Primary=절대 class SLO +
  DistServe SLO-scale sweep**, **Secondary=prefill-normalized slowdown/stretch**(=관측/floor, Bansal-Harchol-Balter
  계보; "normalized latency" 명명 금지). **prefill_floor 실측 필수.**

## 열린 항목 / 다음 세션 시작점

- **★다음 실행 후보 = long-context Stage 0 (L−2), 사용자 승인 대기.** decode-only, ctx{4k,8k,16k}×decode-SM
  {16,44,108}, cudagraph-ON, ITL 측정. **make-or-break 게이트**: 운영점서 long-ctx decode가 SM-binding하나?
  (§1-5 6.5×는 triton/no-cudagraph micro, HE2는 운영점서 decode non-binding 관측). **Zamba2 타이밍 재사용 →
  모델 교체·데이터 준비 없이 즉시·값쌈.** 죽으면 long-ctx 전체 死·HE0 ctx-무관 강화; 살면 L−1↓ 확장.
- **설계상 미결 fork**(L−1부터 걸림): ① chunked-prefill(long-ctx는 chunked-on 현실적이나 substrate 변경 →
  재-baseline vs 8k까지 chunked-off), ② L0+ 모델 = granite-4.0-h-micro(131072, rope 외삽 금지), ③ longbench_v2
  데이터 미보유(오프라인 다운로드), mooncake radix 결정.
- **§1-20 공간 decoupling(사용자 관심사)**은 벡터1과 별개로 열려있음 — long-ctx L3s가 그 증폭 시험.

## 미완·주의

- **admission latch stale-True 버그**(multiplexing_mixin.py:758 근처) known-latent, **미수정·보류**(사용자 결정).
  R2/admission 트랙 착수 시 반드시 선수정.
- **MEMORY.md 디스크본**: 이전 세션에 claims-auditor 2026-07-24 문단 누락 발견 이력(이번 범위 밖, 다음에 확인).
- 벡터1 종결의 **magnitude는 metric-cliff상 ill-posed**(rank만 견고) — 재인용 시 이 caveat 유지. "hybrid 일반화" 금지.
- 실행 중 job 없음. 로컬 커밋 4개(681029a/7a9196b/15281a8/0d986ec) **push 안 됨**(원격 URL PAT 노출).
