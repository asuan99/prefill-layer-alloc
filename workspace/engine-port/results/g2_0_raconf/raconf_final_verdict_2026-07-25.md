# 벡터1 (G2.0 short-ctx disjoint) 최종 확증 판정 — 2026-07-25

**판정: CONFIRMED closure.** short-ctx(in2048)에는 rate 3.5·3.75 band에서도 견고한
off-cliff disjoint가 없다. d54-first TTFT 반전(REOPEN)은 반증됨.

Scope: Zamba2-2.7B, ctx4096, in2048 prefill(o32) / in2048 decode(o512) workload, drained,
cudagraph-ON 운영점, rate_A∈{3.5,3.75}, rB=4. n=6, rep별 독립 boot. 이 워크로드 밖으로
일반화 금지. canonical 채점: TTFT≤3000ms AND per-request token-ITL-p95≤50ms
(`pdmux_eval.analyze.percentile` / `RequestResult.passes` 재사용).

## 0. Predecessors / combined evidence chain

이 판정은 5-단계 sweep 계열의 최종 항이다:

1. [`../g2_0_full/disjoint_verdict_2026-07-24.md`](../g2_0_full/disjoint_verdict_2026-07-24.md)
   (2026-07-24, n=4/mode) — 1차 라운드, razor-thin real disjoint 보고
   (`feasible-A={d16,d44} ∩ feasible-B={d54}=∅`, d54 Phase-A 0.852 fragile).
2. [`../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md)
   (2026-07-25, n=6–10/mode, claims-auditor 독립 재채점) — **ILL-POSED at rA5**:
   byte-identical Phase-A서 d44/d54 견고성 순위 완전 반전, pool(n=10) 시 통계적
   구분 불가(TTFT 3s-cliff bimodality); "disjoint 소멸"은 별도 ITL-p95
   percentile-window 아티팩트.
3. [`../g2_0_decliff/decliff_verdict_2026-07-25.md`](../g2_0_decliff/decliff_verdict_2026-07-25.md)
   (2026-07-24/25, jobs 863880–863948) — de-cliff stage-1(capacity only):
   `rA{2,3,3.5,4}×{d16,d44,d54}` 스캔 결과 **`rA=2`만 clean off-cliff**(전 split
   mean p90<2.0s ∧ std<0.15s); `rA≥3`은 여전히 bimodal. `rA=2`(n=6)에서 disjoint
   미발견(static `d54`가 양 phase 동시 커버) — **PLAUSIBLE closure, CONFIRMED
   아님**(claims-auditor 반증 3항목: off-cliff에서도 살아있는 split→TTFT
   gradient·d54 배제 onset 미측정 전이대·"binding-A⟺on-cliff" 미증명).
4. `../g2_0_rasweep/`(2026-07-25, jobs 863958–864128, 120 jobs,
   `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`) — decliff의 pre-registered
   stage-2 narrow-band 원안. **off-cliff sub-band(rate≤2.75)에서는 disjoint가
   확인되지 않음**(rA2 companion-cover 패턴이 rA2.75까지 유지). raw jsonl만
   존재(별도 verdict 문서 미작성 — provenance는 `RASWEEP_DONE`/`manifest.tsv`);
   이 결과는 전이대를 **rate 3.0–3.5**로 좁혔고, 그 좁혀진 창을 정밀 확증하는
   것이 아래 raconf(본 문서)다.
5. **본 문서(g2_0_raconf, 24 jobs, rate{3.5,3.75}×{d44,d54}×n6)** —
   claims-auditor pre-registered 결정 규칙(§2)을 rasweep이 좁힌 전이대 경계
   (rate 3.0–3.5)에서 직접 확증. 축은 rasweep의 4-split×5-rate 격자에서 **결정에
   필요한 두 split(d44=최선 Phase-A 후보, d54=유일 Phase-B feasible)**만 남겨
   검정력을 집중했다.

**결합 근거**: rasweep(off-cliff band≤2.75서 disjoint 없음) + de-cliff(rA2
Phase-A non-binding, d54 양 phase 커버) + raconf(전이대 rate3.5/3.75서 companion
collapse, d54 미선-배제) → 세 sweep이 일관되게 같은 결론(disjoint 부재)을
가리키며, raconf가 pre-registered REOPEN 조건을 직접 반증해 최종 CONFIRMED로
닫는다.

Confound gate: 24 job 전부 model/ctx/in/out 동일, errs=0, single-round dur(A≈10–11s),
runtime manifest **byte-identical to g2_0_rasweep**, cudagraph ON (disable_cuda_graph=False).
single-variable(rate×split) 충족.

---

## 1. 결정 규칙 입력 (Phase A, per-rep, n=6)

frac_good = good/32. failTTFT/failITL = 그 원인으로 실패한 요청 수(중복 가능, failBoth=0 전부).

### rate 3.5
| split | frac_good per-rep | mean±SD | TTFT_p90 ms per-rep | failTTFT(총) | failITL(총) |
|---|---|---|---|---|---|
| d44 | [0.875, 0.969, 0.969, 0.969, 0.969, 0.969] | **0.953 ± 0.035** | [2595,824,766,1029,825,765] | 3 (rep1만) | 6 (rep당 1) |
| d54 | [0.906, 1.000, 0.938, 0.969, 0.969, 0.906] | **0.948 ± 0.035** | [943,955,1206,1073,943,1170] | **0** | 10 |

### rate 3.75
| split | frac_good per-rep | mean±SD | TTFT_p90 ms per-rep | failTTFT(총) | failITL(총) |
|---|---|---|---|---|---|
| d44 | [0.969, 0.938, 0.844, 0.969, 0.938, 0.938] | **0.932 ± 0.042** | [877,916,2670,879,878,1127] | 3 (rep3만) | 10 |
| d54 | [0.969, 0.969, 0.969, 0.812, 1.000, 0.969] | **0.948 ± 0.062** | [1175,1220,1232,3218,1085,1165] | 5 (rep4만) | 6 |

---

## 2. 판정 분기 적용

**REOPEN (d54-first 반전) — 반증(NOT met).** 어느 rate에서도 d54 frac_good이 견고히
<0.7 unimodal이 아니다. d54 mean = 0.948(양 rate), 0.7 근처도 아님. d54 clean-rep TTFT p90
≈ 940–1230ms(≪3s), **rate3.5 d54는 failTTFT=0(6/6 rep)**. 동시에 d44도 견고히 ≥0.95
off-cliff이 아니다(3.5=0.953 경계, 3.75=0.932). 두 전제 모두 실패 → REOPEN 조건 성립 불가.

**CONFIRMED (companion) — 성립.** 두 rate 모두에서 d54가 홀로 먼저 배제되지 않는다.
rate3.5: d44 0.953 ≈ d54 0.948 (차이 0.005 ≪ SD 0.035, 통계적 동일). rate3.75: **d54 0.948 ≥
d44 0.932** (d54가 오히려 높음). d54는 Phase B를 커버하는 유일 split(아래 §4)이면서 Phase A도
≈0.95로 커버 → **단일 split(d54)이 양 phase 커버 → disjoint 없음.** 확정.

---

## 3. companion-cliff 기전 (규칙 #3)

- d54가 "먼저 degrade"하지 않는다. Phase-A 열화는 split-특정이 아니라 **rep마다 드물게
  터지는 stochastic TTFT-blowup**이며 **양 split 모두**에 발생: d44(r3.5 rep1: 3 failTTFT,
  max 3698), d44(r3.75 rep3: 3 failTTFT, max 3702), d54(r3.75 rep4: 5 failTTFT, max 4260).
- 기전 가설(d54=prefill SM 최소 54 → prefill 굶김 → d54-first TTFT 반전)은 **실현되지 않음.**
  d54의 prefill floor(54 SM)는 이 band에서 충분했고, d54의 Phase-A TTFT p90는 d44와 대등.
- fail-cause: d54가 배제되는 유일 이벤트(r3.75 rep4)는 TTFT 원인이 맞으나, 계통적이지 않고
  d44보다 먼저도 아님. 대부분 rep의 frac 상한(0.969)은 **rep당 정확히 1건의 ITL-floor 요청**
  (itl_p95 ≈ 160–300ms, 웜업성 스파이크)이 결정 — 이것도 양 split 공통.

**★bimodal 경고(감사 지적) 확인.** per-rep 분포는 rep-간 bimodal이다: "clean" mode(~0.969,
ITL-floor 1건 제한) + 드문 "TTFT-blowup" mode(~0.81–0.875, 3–5건 3s 초과). mean-over-rep
(~0.93–0.95)은 이를 가린다(지난 "1556ms" 오진 계열). 그러나 이 bimodality는 **split-대칭적
이고 disjoint를 만들지 않는다.** 부수 효과: 이 blowup mode 때문에 d44는 "견고히 ≥0.95
off-cliff"가 아니며(rep3 0.844) → REOPEN 전제를 독립적으로도 무너뜨린다.

---

## 4. Phase B (rB4 고정, 참고 — split 트레이드오프 확정)

| split | frac_good per-rep | mean±SD | itl_p95(중앙/최대) ms | fail-cause |
|---|---|---|---|---|
| d44 | [0.188]×6 | **0.188 ± 0.000** | 50.7 / 51.6 | failITL 26/32 |
| d54 | [1.000]×6 | **1.000 ± 0.000** | 44.1 / 44.3 | 없음 |

SD=0.000 = rock-solid unimodal. **decode phase는 d54를 요구**(d44 ITL-p95 ≈ 50.7ms가 50 SLO를
0.7ms 초과 → 붕괴). d44의 0.188은 metric cliff(50ms 경계) 위라 magnitude는 fragile하나,
방향(d44는 decode 못 커버)은 견고. 이것이 §2 결론의 핵심 축: 양 phase를 한 split으로 덮으려면
d54여야 하고, 그 d54가 Phase A도 ≈0.95로 덮으므로 disjoint가 없다.

---

## 5. 게이트·불확실성·caveat

- n=6 per-rep ✓, fail-cause 분해 ✓, output-불변(o32/o512 고정) ITL 보조 ✓, single-variable ✓.
- **metric cliff 명시**: Phase-A frac_good≈0.95는 tail 이벤트(blowup rep max TTFT 3.7–4.3s,
  1건 ITL 스파이크)가 결정 → 정확한 0.95 값은 run-length 의존/ill-posed. **순위(d54≈d44,
  d54 미선-배제)는 견고.** Phase-B d44 0.188은 50ms 경계 위 → magnitude fragile, 방향 견고.
- reps는 독립 boot이므로 d44↔d54 rep-index paired bootstrap은 인과적 의미 없음 → 미보고
  (규칙대로 per-rep 분포 + mean±SD로 판정).
- §1–20 방화벽: 본 판정은 이 workload/모델/ctx/drained/cudagraph-ON 운영점에 한정.

---

## 결론 한 줄

**CONFIRMED closure — 벡터1을 닫는다.** rate 3.5·3.75 short-ctx band에서 d54(양 phase 유일
커버 split)가 Phase A를 d44와 대등하게(≈0.95) 커버하고 홀로 먼저 배제되지 않음 → 견고한
off-cliff disjoint 부재 최종 확정. 기전 가설(d54-first prefill-starvation TTFT 반전)은 직접
측정으로 반증(d54 rate3.5 failTTFT=0/6, TTFT p90 대등). REOPEN 없음.
