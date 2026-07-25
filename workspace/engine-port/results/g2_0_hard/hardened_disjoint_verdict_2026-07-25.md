# G2.0 HARDENING sweep — vector-1 (disjoint conflict-regime) verdict (2026-07-25)

Experiment run: 2026-07-24 (jobs 863623–863672, `g2_0_hard_bench.sbatch`).
Analysis + adversarial re-check: 2026-07-25 (result-analyst partial + claims-auditor
independent raw-jsonl re-scoring). Predecessor: [`../g2_0_full/disjoint_verdict_2026-07-24.md`](../g2_0_full/disjoint_verdict_2026-07-24.md)
(g2_0_full, 2026-07-24, n=4/mode, first pass — found a "razor-thin real disjoint").

This sweep was designed explicitly to harden that razor-thin result along two axes
(see `g2_0_hard_bench.sbatch` header): **axis 1** widens Phase B (OB 512→1024,
`d64`/`d74` case-arms added) to push feasible-B away from feasible-A; **axis 2** adds
reps at `d44`/`d54` (n≥6 total, `rA5`) plus a sub-cliff rate (`rA4`) to pin whether the
g2_0_full Phase-A fragility at `d54` (0.852, one collapse to 0.531) was a real ~0.85
regime or cliff noise.

## VERDICT (claims-auditor final, canonical — do not overclaim past this)

> **벡터1은 rA5 운영점에서 ILL-POSED로 판명. escape hatch 근거로 "지지 안 됨", 그러나
> "종결"도 아니다.**
>
> - g2_0_full에서 관측된 razor-thin disjoint(feasible-A={d16,d44} ∩ feasible-B={d54}=∅)는
>   **Phase-A TTFT 3s-cliff bimodality에 기댄 것**이다: 동일(byte-identical) Phase-A
>   워크로드(in2048/o32@rA5)에서 d44/d54 견고성 순위가 sweep 간 **완전히 반전**한다
>   (g2_0_full: d44 0.969 견고/d54 0.852 fragile → g2_0_hard OB1024 subset: d44 0.695
>   붕괴/d54 0.938 견고). 정직하게 pool(n=10)하면 **둘 다 ~0.86–0.90**, 통계적으로 구분
>   불가하며 둘 다 견고 ≥0.9 아니다.
> - g2_0_hard 축1의 "feasible-B가 d34로 넓어져 disjoint 소멸"은 **물리적 완화가 아니라
>   per-request ITL-p95 percentile-window 아티팩트**다: OB512→1024에서 ITL **중앙값은
>   상승**(경합↑)하고 **p95만 하락**(고정 초기 스파이크가 더 많은 토큰에 희석). 사용자
>   토큰 경험은 오히려 악화된다.
> - ⇒ **견고한 disjoint도, 견고한 공유 단일-static도 입증되지 않았다. disjoint를
>   escape-hatch 근거에서 제거하나, 그 부재를 입증하지도 않는다.** 판정은 절벽 위
>   측정이라 ill-posed.
> - ★**§1-20 공간적 coupling-tax(116=92+24 > 108, disaggregation +16%)와 별개 — 이
>   벡터1 결과는 §1-20에 영향 없음. §1-20은 여전히 열려 있다.**
> - 제대로 판정하려면 de-cliff 재스윗이 필요(진행 예정): 먼저 rA를 낮춰 Phase-A p90≪3s
>   확보 → output-길이 불변 ITL 지표 → ≥2-SM-step 간극에서 n≥6 paired.

**방법론 교훈(CONSENSUS §3 gate 강화)**: 또 절벽 위에서 측정했다(gate #6 위반 — "메트릭이
임계 지시함수면 절벽을 피해 측정할 것"). g2_0_full 자체가 gate #6을 어긴 벤치였고, 이번
hardening sweep은 gate #6을 지키려 시도했으나(축1=워크로드를 바꿔 절벽에서 밀어내려 함)
**축1 자체가 다른 절벽(ITL percentile-window)에 새로 올라앉는 바람에 실패**했다.
**교훈을 명시적으로 강화**: feasibility 판정 전 반드시 **capacity de-cliff를 선행**해야
하고, de-cliff 여부는 "결과가 바뀌었는가"가 아니라 "메트릭이 여전히 임계/percentile
경계에 앉아 있는가"로 검증해야 한다 — 워크로드를 바꾸는 것만으로는 새 절벽을 만들 수
있다는 점을 이번 sweep이 실증했다.

---

## 1. Sign-flip evidence — d44/d54 Phase-A robustness is not reproducible at rA5

Canonical scoring (identical to g2_0_full): request good iff **TTFT ≤ 3000ms AND
that request's own token-ITL p95 ≤ 50ms**; Phase A = in2048/o32, cudagraph ON,
Zamba2-2.7B, ctx4096, n=32/rep. Phase-A workload bytes are **identical** regardless of
`OB` (Phase-B output length has no bearing on Phase-A's own workload spec, only on
what runs afterward in the same server process).

| campaign | mode | rA | OB | reps | per-rep frac_good | mean | notes |
|---|---|---|---|---|---|---|---|
| g2_0_full (2026-07-24) | d44 | 5 | 512 | 4 | 0.969, 0.969, 0.969, 0.969 | **0.969** | robust, cliff-free (verdict: "feasible-A") |
| g2_0_full (2026-07-24) | d54 | 5 | 512 | 4 | 0.531, 0.969, 0.938, 0.969 | **0.852** | bimodal, one collapse (verdict: "Phase-A-fragile") |
| g2_0_hard (2026-07-24) | d44 | 5 | 1024 | 4 | **0.000**, 0.969, 0.969, 0.844 | **0.695** | flip: one full collapse (job 863655) |
| g2_0_hard (2026-07-24) | d54 | 5 | 1024 | 4 | 0.938, 0.938, 0.938, 0.938 | **0.938** | flip: rock-solid, zero variance |
| g2_0_hard (2026-07-24) | d44 | 5 | 512 (rep5/6 extension) | 2 | 0.969, 0.969 | 0.969 | matches g2_0_full d44 |
| g2_0_hard (2026-07-24) | d54 | 5 | 512 (rep5/6 extension) | 2 | 0.938, 0.938 | 0.938 | matches hard d54 |

Pooled (n=10, all rA5 Phase-A reps regardless of OB — legitimate pool since Phase-A
workload is byte-identical across OB):

| mode | n | mean frac_good | population SD | individual values |
|---|---|---|---|---|
| d44 | 10 | **0.859** | 0.289 | 0.969×4 (full), 0.0, 0.969, 0.969, 0.844, 0.969, 0.969 (hard) |
| d54 | 10 | **0.903** | 0.125 | 0.531, 0.969, 0.938, 0.969 (full), 0.938×6 (hard) |

Both land in ~0.86–0.90 with overlapping spread; d44's outlier collapse (job 863655,
frac=0.000, TTFT p50 3762ms) alone swings its 4-rep mean from 0.969 to 0.695 — the
"robust ≥0.9" claim for either split does not survive a 10-rep pool. d44's fragility
is at least as bad as d54's in the hardened sample, inverting the original verdict's
ranking. This is consistent with genuine cliff-adjacency noise (per gate #6/CONSENSUS
§1-14 "metric cliff" mechanism), not a reproducible physical regime boundary.

Raw data: `g2_0_full/g20A_d44_rep*_rA5B4_*.jsonl`, `g2_0_full/g20A_d54_rep*_rA5B4_*.jsonl`,
`g2_0_hard/g20hA_d44_rep{1-4}_rA5B4_OB1024_*.jsonl`, `g2_0_hard/g20hA_d44_rep{5,6}_rA5B4_OB512_*.jsonl`,
`g2_0_hard/g20hA_d54_rep{1-4}_rA5B4_OB1024_*.jsonl`, `g2_0_hard/g20hA_d54_rep{5,6}_rA5B4_OB512_*.jsonl`.

---

## 2. Axis-1 "widening" evidence — a percentile-window artifact, not physical relief

The hardening sbatch's stated intent for axis 1 (OB 512→1024) was to make Phase B
*heavier* so it needs *more* decode SM, pushing feasible-B away from feasible-A. What
actually happened at every measured split (`d34`, `d44`, `d54`) is the opposite
direction on the metric that gates feasibility: **per-request ITL-p95 fell** while
**per-request ITL-median rose** — i.e. steady-state contention got *worse*, but the
canonical feasibility metric (`ITL-p95 ≤ 50ms`) reads as *better*, because a
fixed-size early-generation spike is diluted by more steady-state tokens when output
length is longer (512→1024 tokens), pulling the per-request p95 down toward (worse)
median while leaving the spike's absolute size unchanged.

Median-of-per-request-p95 and median-of-per-request-median ITL (ms), Phase B
(in2048, cudagraph ON, n=32/rep, mean over reps):

| mode | OB | ITL p95-of-request (ms) | ITL median-of-request (ms) | feasible under 50ms bar? |
|---|---|---|---|---|
| d34 | 512 (g2_0_full) | 59.2–60.3 | 28.9–29.1 | **no** |
| d34 | 1024 (g2_0_hard) | 47.9–48.3 | 33.4–33.6 | **yes (artifact)** |
| d44 | 512 (g2_0_full) | 50.6–51.2 | 27.5–30.0 | borderline no |
| d44 | 1024 (g2_0_hard) | 42.1–42.2 | 33.4–33.5 | yes (artifact) |
| d54 | 512 (g2_0_full) | 44.1–44.2 | 28.2–28.5 | yes | yes (already feasible, real) |
| d54 | 1024 (g2_0_hard) | 38.9–41.2 | 33.2–33.3 | yes | yes (still feasible, but median worse) |

Every split shows the same sign pattern: p95 down (looks better), median up (is
worse). This is the mechanism claims-auditor flagged for `d34` specifically, and it
generalizes to `d44`/`d54` too — the metric, not the workload, moved. The "feasible-B
widened to include d34, killing the disjoint" claim from a naive read of axis-1 is
therefore **not a physical relief of the conflict**; it is a percentile-window
artifact of extending output length under a fixed-percentile SLO metric.

Separately, axis 1's `d64`/`d74` arms (added to push feasible-B *up* past d54, the
originally-intended direction) instead **collapsed Phase A almost entirely**: d64
Phase-A frac_good mean 0.164 (0.312, 0.031, 0.312, 0.000), d74 mean 0.062 (flat
0.062×4, i.e. only 2/32 requests pass per rep). Widening decode SM further does not
create a wider disjoint window with two feasible sides — it just pushes Phase-A off a
cliff into near-total failure while Phase-B feasibility was already saturated at d54.
No wider, cleaner disjoint was found on this axis either.

Raw data: `g2_0_hard/g20hB_{d34,d44,d54}_rep*_rA5B4_OB1024_*.jsonl` (axis-1 Phase-B),
`g2_0_hard/g20hA_{d64,d74}_rep*_rA5B4_OB1024_*.jsonl` (axis-1 Phase-A collapse).

---

## 3. Axis-2 (sub-cliff rate, rA4) — inconclusive, flagged not resolved

The sbatch's axis 2 added `rA4` (below the `rA5` used everywhere else) at `d44`/`d54`
to check whether lowering arrival rate below the apparent cliff stabilizes the
Phase-A frac_good estimate. It does not cleanly resolve either way:

| mode | rA | reps | frac_good values | mean | SD |
|---|---|---|---|---|---|
| d44 | 4 | 6 | 0.938, 0.938, 0.125, 0.625, 0.594, 0.375 | 0.599 | 0.290 |
| d54 | 4 | 6 | 0.594, 0.625, 0.688, 0.938, 0.625, 0.281 | 0.625 | 0.192 |

Counter-intuitively, the lower-rate (`rA4`) arm is *not* more robust than `rA5` for
either split in this sample (`rA4` d44 mean 0.599 < `rA5` pooled d44 mean 0.859) —
which is itself evidence that these runs are still cliff-adjacent / noisy rather than
settled into a clean sub-capacity regime, or that per-launch confounds (separate
server boot per rep) are contributing variance not accounted for by rate alone. This
axis does **not** support a stronger claim in either direction and is left as an open
question for the de-cliff resweep rather than folded into the headline verdict.

Raw data: `g2_0_hard/g20hA_{d44,d54}_rep*_rA4B4_OB512_*.jsonl`.

---

## 4. Firewall — §1-20 spatial coupling-tax is unaffected

CONSENSUS §1-20 ("Oracle 재구성... headroom은 +2%가 아니라 +16% — 단 그건
disaggregation 몫") establishes a **spatial** SM-budget argument: the decoupled oracle
needs 92 prefill SM + 24 decode SM = 116 > 108 available, i.e. a single-GPU coupling
tax independent of any temporal disjoint-feasibility claim. This G2.0 vector-1 sweep
tests a **different, temporal** question (does any single static split serve both
phases of a workload across time) and its ill-posed result **has no bearing on
§1-20's SM-budget arithmetic**. §1-20 remains open per CONSENSUS §5-8(a) (decoupled
substrate unimplemented, headroom unverified on any substrate).

---

## 5. Recommended de-cliff resweep recipe (not yet run)

To convert "ill-posed at rA5" into a real verdict (either direction), per
claims-auditor:

1. **Lower `rA` further** than the `rA4` tried here, until Phase-A TTFT p90 is
   comfortably below the 3s SLO (e.g. p90 < 1.5s, roughly half the SLO) — not just
   "below the median cliff crossing" as `rA4` attempted.
2. **Use an output-length-invariant ITL metric** for Phase-B feasibility instead of
   raw per-request ITL-p95 at a fixed output length — e.g. steady-state ITL computed
   only over the tail (last N tokens, excluding early-generation transient) or a
   fixed-window percentile normalized by output length, so extending `OB` cannot
   mechanically move the metric without moving true contention.
3. **Widen the SM-step granularity being compared** to ≥2 steps (e.g. compare d24
   vs d54/d64 directly rather than adjacent d44/d54) so any true disjoint is not
   sitting exactly on a single-step boundary.
4. **n≥6 paired reps** at each candidate split, both phases, before drawing a
   feasibility-set conclusion (per CONSENSUS §3 gate: "n≥4 없이 정책 결론 금지" — this
   sweep already shows n=4 is not enough at cliff-adjacent points).
5. Do **not** re-score existing OB512/OB1024 data across a new ITL definition (gate
   #8 analog) — re-run with the new metric captured live.

---

## Provenance

- sbatch: `g2_0_hard/g2_0_hard_bench.sbatch` (derived from `g2_0_full/g2_0_full_bench.sbatch`).
- jobs: 863623–863672 (2026-07-24), partition `amd_a100nv_8`, Zamba2-2.7B, ctx4096,
  cudagraph ON, `--attention-backend triton --disable-radix-cache`.
- raw jsonl: `g2_0_hard/g20hA_*.jsonl` (Phase A), `g2_0_hard/g20hB_*.jsonl` (Phase B).
- scoring: canonical `TTFT≤3000ms AND per-request token-ITL p95≤50ms`, matching
  `pdmux_eval.analyze.summarize_requests` percentile definition (linear interpolation).
- predecessor verdict: [`../g2_0_full/disjoint_verdict_2026-07-24.md`](../g2_0_full/disjoint_verdict_2026-07-24.md).
