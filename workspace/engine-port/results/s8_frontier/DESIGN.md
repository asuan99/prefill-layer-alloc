# E1 — 8B frontier campaign: `ITL(D) vs TTFT(108-D)` under budget constraint

Built 2026-07-28 (experiment-runner). **Harness build only — nothing here has
been submitted or run.** `s0_deconfound/DESIGN.md` §5 pre-registration still
governs: this is not a claim job, nothing in this directory enters canon
before claims-auditor review.

## 0. Why this campaign (do not re-derive — see `PROJECT_STATUS.md` "8B
   decode-SM 민감도 측정 노트" / "열린 긴장" for the full trail)

`s8_scaleup/FINDINGS_8B_2026-07-28.md` (**C2, CONFIRMED scoped**) measured a
real decode-SM lever: with prefill pinned at 16 SM, decode ITL improves
2.36–2.91× from SM16→SM92, model-agnostic across 4 arms. But that is an
**equi-load curve** — low-D cells leave most of the GPU idle on purpose. The
real serving question is what happens when prefill and decode compete for
the **same 108-SM budget** (`prefill_SM + decode_SM == 108`), because that is
exactly the constraint HE2 (`CONSENSUS.md`) found flat: at the cudagraph
operating point, the optimal static split did not move and dynamic control
did not beat best-static. **Tension A**: does C2's lever survive that budget
constraint, or is it net-negative once you have to pay for decode SM with
prefill SM? This campaign measures that directly — it is E1 in
`PROJECT_STATUS.md`'s "다음 실험 gate" list (item 8), claims-auditor-specified,
not something this agent invented.

## 1. Arms (identical roster to `s8_scaleup`/`s8p_prefill`)

| id | label | model | family | backend |
|---|---|---|---|---|
| 0 | M8 | `hf_cache/mamba-codestral-7b-sglang` (7.3B) | pure SSM | triton |
| 1 | Ha8 | `Zyphra/Zamba2-7B-Instruct` | additive hybrid | triton |
| 2 | Hs8 | `nvidia/Nemotron-H-8B-Base-8K` | substitutive hybrid | **flashinfer** (server_args.py:1959 bans triton) |
| 3 | T8 | `Qwen/Qwen2.5-7B` | pure Transformer | triton |

Backend asymmetry is unavoidable (same reasoning as the two prior campaigns:
NemotronH cannot run triton, Zamba2-7B's cudagraph capture dies on
flashinfer) — no hybrid arm runs on both backends, cross-arm claims
involving Hs8 carry that caveat.

## 2. Grid — `[108-D, D]`, D ∈ {16, 24, 44, 54, 92}

Configs: `pdmux_e1_d{16,24,44,54,92}.yml`. All are **budget-constrained**
(no idle SM, unlike the two equi-load campaigns this reuses code from):

| cell | prefill_SM | decode_SM | `manual_divisions` rows | guard-satisfier needed? |
|---|---|---|---|---|
| d16 | 92 | 16 | 1 | no (16 ∈ {16,24,34,44}) |
| d24 | 84 | 24 | 1 | no (24 ∈ set) |
| d44 | 64 | 44 | 1 | no (44 ∈ set) |
| d54 | 54 | 54 | 2 | yes (54 ∉ set; guard row `[64,44,48]`) |
| d92 | 16 | 92 | 2 | yes (92 ∉ set; guard row `[64,44,48]`, identical content to `s8_scaleup/pdmux_c_d92.yml`) |

`_build_r2_policy` (`multiplexing_mixin.py:127-142`) requires at least one
`manual_divisions` row whose decode column is in `{16,24,34,44}`; d54/d92
satisfy this with a never-selected second row (`FixedPolicy(D)` matches the
`sm_counts` entry with `decode_sms==D` exactly, verified by reading
`controller.py:80-94`). `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=D` per
cell, same convention as `s8_scaleup`/`s8p_prefill`.

**"Best-static" is not a 6th arm.** The decision rule (§4.4) computes
best-static as the empirical argmax over these 5 measured D cells per arm —
the grid already spans the `prefill_SM+decode_SM<=108` trade-off space the
question is about, so a separately-sourced "best static" (e.g. reusing
d44/d54 numbers from the 2.7B `slo_sched`/`g2_0_*` campaigns) would compare
across model scale and is not needed. This was a deliberate scoping decision,
not an oversight.

## 3. Staging (why not all 4 arms at once)

4 arms × 5 D-cells × REPS≥4, each cell needing a fresh 7-8B server boot
(~3–10 min observed, §6.1), is 20 cells and, per the time budget in §7, on
the order of 6–9 hours of GPU compute even before the capacity pre-scan.
Submitting all 4 arms in one wave risks burning a large, unattended block of
GPU time on a design (workload rates, SLOs) that has not been validated by a
smoke test yet, and this task's instructions explicitly ask for staged
arms with the cut documented, not silent truncation.

**Stage 1 (submit after capacity scan + smoke pass): T8 + Hs8**
(`ARM_IDS="3"` and `ARM_IDS="2"` as two parallel single-arm jobs, or
`ARM_IDS="3 2"` as one sequential job — see §7). Rationale: T8 (pure
Transformer) is the fastest-booting, cleanest-to-interpret positive control
(attention prefill cost is the textbook case for this trade-off); Hs8
(substitutive hybrid, the arm C2 found had the *lowest* absolute latencies
among the hybrids) is the primary scientific target — does the C2 lever
survive the budget constraint for a hybrid architecture specifically.

**Stage 2 (after stage 1 gates pass and results are sane): M8 + Ha8**
(`ARM_IDS="0"` / `ARM_IDS="1"`). M8 is the pure-SSM negative-control mirror
of T8; Ha8 (additive hybrid) is the architecture most implicated in the
2.7B entanglement/HE0 negative-result work — most decisive for connecting
this campaign back to CONSENSUS, but also the arm with known duty-cycle
sensitivity (`FINDINGS_8B_2026-07-28.md` §4-5) and the slowest boot observed
in `s8_scaleup`, so it goes second rather than first.

If GPU time is scarcer than expected, stage 1 alone (T8+Hs8) already answers
tension A for a pure-Transformer/substitutive-hybrid pair; stage 2 extends
model-agnosticity but is not required to get a first citeable answer.

## 4. Workload

### 4.1 Shape — open-loop, ShareGPT, rate alternating LO↔HI (change-trace)

Reuses `sglang.bench_serving --request-rate` (Poisson open-loop; **not**
`s0dc_client.py`/`s8p_client.py`, which are closed-loop and explicitly
forbidden for this campaign per the task spec — closed-loop caused the
865098/865311 decode-batch-covaries-with-D confound in the two equi-load
campaigns this harness otherwise mirrors). Directly mirrors
`slo_sched/sharegpt_vary_bench.sbatch` and `g2_0_raconf`'s two-phase
open-loop pattern, both already the project's validated "change-trace, not
stationary r8" vehicle (methodology gate #2, `CLAUDE.md`). Each rep = one
LO-rate ShareGPT phase (`NP` prompts) + drain + one HI-rate ShareGPT phase
(`NP` prompts), same dataset/content in both phases — only the **arrival
rate** changes, matching the project's established "rate 3↔12" pattern
rather than g2_0's phase-A/phase-B *content*-switch pattern (that pattern
answers a different question — feasible-region disjointness across
workload *types*, not this campaign's ITL/TTFT trade-off under a fixed
mixed workload).

`--context-length` cap = `CTX(4096) + 512 + 256 = 4864` (matches
`s8_scaleup`/`s8p_prefill`), `--sharegpt-context-len 3600` (drops
conversations whose recorded input+output exceeds 3600 tokens, keeping
every admitted request's context comfortably under the 4864 cap and under
Zamba2-7B's native `max_position_embeddings=4096` — see §9 risk 3).

### 4.2 Capacity pre-measurement (methodology gate #6 — required, not optional)

★**Revised 2026-07-28 (coordinator directive, after the §4.3 SLO revision
below) — the scan must output BOTH distributions per cell, not just an
elbow.** `e1_capacity_scan.sbatch` boots each (arm, D-cell) once and fires
short ShareGPT probes at `RATES="1 2 3 4 6 8 12 16"` (req/s, 20s-target
duration each via `NP = max(45, ceil(rate*20))` (`NP_MIN` raised 15→45
2026-07-29, §4.2.3), `PROBE_TARGET_S` tunable),
printing, per (cell, rate): raw **TTFT p50/p95/p99** AND the
**request-internal token-ITL p95 distribution's own p50/p95** (already
implemented in the script's `CAPSCAN` line — this was already computed, the
revision is in how it is *used*, see below). This is not merely an
elbow-location tool any more: it is the direct input to (a) the ITL-ladder
cliff-hazard check (§4.3) and (b) the TTFT-SLO siting rule (§4.3), both of
which need the actual percentile values at candidate rates, not just "is
this side of the elbow." A human (or result-analyst) still uses the TTFT
curve to find the elbow and picks `RATE_LO`/`RATE_HI` **comfortably below
the elbow for every cell in the arm** (same "locate before judge" logic
`g2_0_decliff`/`g2_0_pre` used, `bench_noise_root_cause.md`'s
3%-perturbation→2×-goodput trap is exactly what this exists to avoid) — but
now ALSO runs `e1_analyze.py --capscan-dir ... --capscan-rate <chosen LO or
HI>` (added this revision, see §4.3) against the chosen rate(s) before
finalizing, to get the CLIFF HAZARD / ITL-NONBINDING / TTFT-margin verdicts
for that specific site.

**Key risk this step is designed to surface, not paper over (see also
§9.1): capacity is not one number per arm — it can differ BY CELL.** D16
(decode=16 SM) is the worst-case *decode* capacity cell; D92 (prefill=16 SM)
is the worst-case *prefill* capacity cell. A rate that is comfortably
off-cliff for D44 could already be on-cliff for D16's decode throughput or
D92's prefill throughput. The scan is run across **all 5 cells of an arm**
specifically so the chosen `RATE_LO`/`RATE_HI` can be validated as
off-cliff **simultaneously for every cell**, not just the middle of the
grid. If no such rate exists (i.e. the safe bands for D16 and D92 do not
overlap), that is itself a finding worth recording, not something to route
around by picking different rates per cell (which would confound the D
comparison with a rate comparison) — flag it in the scan write-up and
escalate before submitting `e1_sweep.sbatch` for that arm.

The scan's per-probe settle is a fixed 5s sleep, **not** `e1_sweep.sbatch`'s
full running==0/queued==0 drain-detection loop — acceptable because scan
output is never cited directly (elbow-location only), documented deviation
per this task's "note deviations" instruction.

### 4.2.1 ★★REVISED 2026-07-29 (coordinator directive, smoke 866868) —
    `achieved_rps` was tail-diluted and unreliable; split into
    `arrival_rps` / `completion_rps`, elbow-finding reassigned to TTFT/ITL
    trend

**Diagnosis.** 866868 (T8, d44, rates 2/8, `PROBE_TARGET_S=10` override)
showed `achieved_rps` values (1.50 at rate=2, 3.40 at rate=8) far below
nominal, and — the decisive tell — **TTFT p50 at rate=2 (210ms) was HIGHER
than at rate=8 (99ms)**, a physical impossibility for a monotonic capacity
curve. Root cause: `dur` (the denominator of the old `achieved_rps`) is
bench_serving's own completion-based duration — first-request-dispatch to
last-request-FULL-COMPLETION, including that last request's entire decode
generation. For a SHORT probe (arrival window ~10s target) this "drain
tail" is a large, non-shrinking fraction of the measured window
(rate=2: dur=13.3s for a ~10s arrival window, ~3.3s pure tail; rate=8:
dur=23.6s, ~13.6s tail — WORSE at higher rate because queueing under load
extends the tail further), so `n/dur` conflates "is the server saturated"
with "how much did this specific short probe's trailing generation happen
to cost" — exactly the confound the coordinator flagged (cannot distinguish
saturation from an artifact of too few requests).

**Fix — two numbers, not one, neither used alone for elbow-finding:**
- **`arrival_rps`**: reconstructed via replaying `sglang.bench_serving`'s
  own seeded arrival-time RNG (`np.random.seed(args.seed)` +
  `np.random.exponential(1/rate)` per `get_request()`,
  `bench_serving.py:948`; verified exact for THIS configuration —
  `--dataset-name sharegpt` sampling only calls Python's stdlib
  `random.shuffle` (`sglang/benchmark/datasets/sharegpt.py:98`), never
  `np.random`, so the `np.random` stream is untouched between
  `np.random.seed()` (`bench_serving.py:1706`) and `get_request()`'s first
  draw — see the inline scorer's comment for the FRAGILITY caveat: this
  reconstruction silently desyncs if the dataset/backend changes to one
  that DOES consume `np.random` during prep, or if `--seed` is ever
  overridden per-probe). This answers "did the client actually dispatch
  requests on schedule" — a VALIDITY check, not a capacity signal (an
  open-loop client dispatches unconditionally, so `arrival_rps` will track
  the nominal rate closely whenever the realized Poisson draw isn't
  unusually far from its mean).
- **`completion_rps`** (renamed from `achieved_rps`, same `n/dur`
  computation as before, kept for continuity): still tail-diluted for short
  probes, printed with an explicit caveat, not to be read alone.
- **Elbow-finding is reassigned to the TTFT/ITL percentile TREND across
  rates** (both already printed, and now warmup-corrected per §4.2.2) —
  standard practice for open-loop capacity probing: throughput alone is a
  poor saturation signal in a short, finite-duration open-loop test (a
  saturating queue needs time to visibly build; latency degrades sooner and
  more reliably). This is a change in HOW the already-printed TTFT/ITL
  numbers are used, not a new measurement.
- **Every rate is now probed at ≥2 seeds** (`SEEDS`, default `"1 2"`, §4.5
  item 4) and a `CAPSCAN_SEED_DIVERGENCE` line reports the relative
  difference in TTFT p50 and `arrival_rps` between them, flagged if TTFT
  p50 differs by >25% (a judgment-call threshold, not data-derived) — a
  rate flagged this way is an unstable region for off-cliff siting and
  should not be picked as `RATE_LO`/`RATE_HI` even if its single-seed
  numbers looked clean.

**Retrospective recomputation on the EXISTING 866868 data** (no GPU needed,
per the coordinator's ask) — how far apart do the two rate definitions run:

| rate | NP | dur (s) | arrival_span_s (reconstructed) | `arrival_rps` | `completion_rps` (old `achieved_rps`) |
|---|---|---|---|---|---|
| 2 | 20 | 13.3 | 5.5 | **3.45** | 1.50 |
| 8 | 80 | 23.6 | 8.8 | **8.96** | 3.40 |

`arrival_rps` at rate=8 (8.96) tracks nominal closely (+12%); at rate=2
(3.45) it deviates sharply (+72%) — the MAGNITUDE is explained by realized
Poisson variance at small n (with only 19 usable inter-arrival draws,
NP=20, the relative standard deviation of the arrival span is
`1/sqrt(19)`≈23%, derivation in §4.2.3, so a +72% draw is an unlucky but
plausible ~1.7σ outcome, not evidence of a broken client). **★★But the
coordinator's follow-up (2026-07-29) identified a SEPARATE, more serious
problem this framing alone misses: `--seed` was never overridden
(`e1_capacity_scan.sbatch`/`e1_sweep.sbatch` both call `sglang.bench_serving`
without a `--seed` flag, so its implicit default, `seed=1`, applied to
EVERY probe/rep/cell/arm in the entire campaign) — meaning this +72% draw
is not noise that would average out across reps, it is the SAME FIXED draw
on every rep. §4.5 has the full diagnosis and fix (seed varies by rep,
matched across cells/arms) — this table's numbers, and every prior smoke's
numbers, reflect exactly ONE (non-representative, per this table) workload
realization repeated, not a sample from the realization distribution.**
This independently corroborates requirement (3)'s premise (small NP at low
rate is the dominant noise source WITHIN a realization) using a mechanism
distinct from either the tail-dilution diagnosis above or the §5.2
concurrency-diagnostic bug — but §4.5's finding is the more consequential
one: NP alone cannot fix a problem that is about realization-to-realization
variance never being sampled at all.

### 4.2.2 ★★NEW 2026-07-29 — warmup discard (pre-registered)

866868's TTFT reversal (210ms at rate=2 vs 99ms at rate=8) is explained by
a cold-start transient: the FIRST few real probe requests at a
freshly-booted cell are slower (triton/cudagraph/cache warmup residue not
fully cleared by bench_serving's own built-in single `--warmup-requests 1`
request), and at rate=2's small NP=20 this transient dominates the whole
sample's p50. **Pre-registered fix** (mirrors `s0dc_client.py`'s
`--warmup-s` convention): discard the first `WARMUP_N =
ceil(WARMUP_S * rate)` DISPATCH-order requests (index into the `ttfts`/
`itls` arrays, which `asyncio.gather` returns in dispatch order regardless
of completion order — no RNG replay needed for this specific trim) from
TTFT/ITL percentile computation only (still counted toward `n`/`errs` for
admission/error-rate diagnostics). `WARMUP_S` default **3s**, scaling with
rate exactly like `PROBE_TARGET_S`/`NP` so the discarded FRACTION of a
probe stays roughly constant across rates (≈30% at the default settings —
acceptable for a short scan probe, would be reconsidered for a longer
main-sweep-style measurement).

**Retrospective recomputation, same 866868 data, `WARMUP_S=3`:**

| rate | warmup_n | n_steady | TTFT p50 pre-fix | **TTFT p50 post-warmup-discard** |
|---|---|---|---|---|
| 2 | 6 | 14 | 210ms | **87ms** |
| 8 | 24 | 56 | 99ms | **92ms** |

The reversal is resolved (87ms < 92ms, now monotonic in the expected
direction) — strong confirmation the warmup transient, not a genuine
capacity artifact, was driving the original inversion.

### 4.2.3 ★★NEW 2026-07-29 — `NP_MIN` raised 15→45 (rate-independent
    variance floor)

The relative standard deviation of a Poisson arrival span over `n` draws is
`std/mean = (sqrt(n)/rate) / (n/rate) = 1/sqrt(n)` — **rate cancels out**,
so the sample-count floor needed for a given relative-precision target is
the SAME at every rate, not something `PROBE_TARGET_S` (a duration target)
alone can guarantee at low rates. `1/sqrt(45) ≈ 0.149` (~15% relative std)
was chosen as the target; `NP_MIN` raised from 15 to **45** accordingly.
Only the lowest 1-2 `RATES` values are affected in practice (`NP =
max(NP_MIN, ceil(rate*PROBE_TARGET_S))` — the `PROBE_TARGET_S` term already
dominates once `rate*PROBE_TARGET_S > 45`, e.g. rate≥3 at the default
`PROBE_TARGET_S=20`), so this raises probe cost only where the noise
problem is actually concentrated.

### 4.2.4 ★★NEW 2026-07-29 — per-cell `concurrent_time_frac` varies
    substantially and MUST be reported per cell, not just checked for "low"

866868's d44 cell (prefill=64 SM) measured `concurrent_time_frac=0.0566`
(5.66%, time-weighted per §5.2's fix) — an order of magnitude BELOW d92's
25–40% (§5.2), and comparable to d16's ~1% low end. **This is a real,
cell-dependent asymmetry, not noise**: d44's 64-SM prefill allocation is
fast enough that individual prefill windows are short, giving less
wall-clock opportunity to overlap with a concurrent decode step, the SAME
mechanism §9.7 already established for d16 (92 SM, fastest, ~1%) vs d92
(16 SM, slowest, 25–40%) — d44 sits at an intermediate SM allocation and
(so far, n=1 cell measured at this depth) an intermediate-to-low
concurrency value, consistent with a monotonic SM→concurrency relationship
across the grid, though this is only 3 of 5 cells measured so far (d16,
d44, d92; d24/d54 unmeasured) and should not yet be treated as a confirmed
monotonic trend.

**Consequence for interpretation, not just measurement**: because
`concurrent_time_frac` differs substantially by cell, **cells in the D-grid
are not all "equally multiplexed" when compared against each other** — a
low-D cell (small decode SM, large prefill SM, fast prefill) is
structurally exercised in the co-resident regime for LESS of its own
wall-clock than a high-D cell is. This is not a defect to fix (the whole
D-grid, by construction, spans exactly this SM-allocation range) but it
**must be reported per cell alongside any D-comparison result** — a
citation that compares D44's goodput to D92's without also reporting their
respective `concurrent_time_frac` values would silently compare a
"lightly-multiplexed" cell against a "heavily-multiplexed" one as if they
were measured under equivalent conditions. `e1_analyze.py`'s "MANDATORY
DIAGNOSTIC" section already prints this per (arm, cell) — this subsection
formalizes the requirement that it be read and cited, not skipped, for
every result table in the main sweep's write-up.

### 4.3 SLO selection — ★★REVISED 2026-07-28 (coordinator directive, pre-registered
    BEFORE any E1 data is seen — §4.3.1 below explains why 150ms, this
    design's original placeholder, would have killed the experiment)

**Do not re-derive this — it supersedes the original text of this section**
(preserved reasoning below is the corrected version, not the first draft).

**4.3.1 ITL-p95 SLO — 60ms fixed as the PRIMARY value, {50,60,80}ms as a
pre-registered LADDER.**

The primary ITL-p95 SLO is **60ms**, chosen from prior practice, not from
this campaign's own data: `reports/serving_slo_survey.md`'s chat-class band
(interactive chat TTFT 300ms / ITL 50ms) and `CONSENSUS.md` §1-17's
precedent of retuning the controller to a chat 300/50 SLO. 60ms sits in
that band; **150ms (this design's original placeholder, used only in the
smoke-test command) does not** and must not be used for anything beyond
pipeline-plumbing checks.

**Why this matters mechanically, not just as a convention choice:**
`FINDINGS_8B_2026-07-28.md` §2's batch=12-matched ITL p50 (ms) at SM16/SM92
— M8 58.44/20.09, Hs8 40.63/15.73, T8 30.20/12.77, Ha8(b9) 89.80/31.49 — puts
**every arm, at every D, under 150ms**. If the ITL SLO term never binds,
conjunctive goodput collapses to TTFT-only, and the decision rule (§4.4)
would then trivially favor prefill-heavy D — not because the decode-SM
lever is net-negative, but because **the experiment stopped being able to
ask the question it exists to ask.** That is a design failure, not a null
result, and it would have silently produced a wrong-for-the-wrong-reason
"C2 is net-negative" verdict for every arm.

A single point value is still fragile (M8's SM16 p50 of 58.44ms sits close
enough to 60ms that its p95 could land on either side of the SLO purely
from rep-to-rep noise — see §9.6, "CLIFF HAZARD"). The pre-registered
**ladder {50, 60, 80}ms** is reported alongside the 60ms headline for every
arm/cell as a sensitivity check, implemented in `e1_analyze.py` (default
`--itl-p95-slo-ms 60`, `--itl-ladder-ms "50,60,80"`).

**4.3.2 Gate #8 scope clarification (so this ladder is not misread as a
violation later).** `CLAUDE.md` methodology gate #8 ("SLO를 바꿔 평가할 땐
기존 컨트롤러 재스코어 금지 → 그 SLO로 재튜닝해 직접 측정") is a discipline
**about dynamic controllers**: re-scoring a controller that was tuned for
SLO_A against SLO_B is invalid because the controller's *behavior* (its
switching decisions) would have been different had it been tuned for
SLO_B — you cannot infer what it would have done. **E1 has no such
controller.** Every cell runs `PDMUX_R2_POLICY=fixed` — a static partition
that does not observe or react to the SLO at all. There is nothing to
re-tune, so evaluating the SAME raw per-request records against multiple
**pre-registered** SLO values (the ladder) is not the violation gate #8
describes — it is exactly what gate #8's own alternative ("그 SLO로
재튜닝해 직접 측정") reduces to when there is no controller: direct
measurement at each SLO, from data collected once. This distinction is
recorded here explicitly because a future reader who sees "re-scored the
same jsonl at three different SLOs" without this context could mistake it
for the very violation gate #8 exists to catch.

**4.3.3 TTFT SLO — the symmetric trap, and the pre-registered siting rule.**
If ITL is binding but TTFT never is, the mirror-image failure occurs:
decode-heavy D trivially wins (prefill starvation never costs anything on
the TTFT axis), which is just as uninformative as the ITL-vacuous case
§4.3.1 fixes. **Conjunctive goodput only tests the frontier tension when
BOTH terms can bind.** A 3600-token ShareGPT-capped prompt pushed through a
16-SM prefill partition (the D92 cell's prefill allocation) plausibly takes
seconds, so a 3s TTFT SLO (this project's historical default) is likely to
bind only barely, at the decode-heavy end of the grid — which is itself
squarely inside the gate #6 metric-cliff danger zone, not a safely-off-cliff
regime.

⇒ **TTFT SLO is sited from THIS campaign's own capacity-scan data (§4.2),
but by a pre-registered RULE fixed before looking at the data, not by
picking whatever "looks discriminating":**

> The TTFT SLO is chosen as a value that (i) **binds in at least one
> measured cell** (that cell's TTFT p95 exceeds the candidate SLO), and
> (ii) has **≥15% relative margin** from every cell's measured TTFT p95
> (`|p95 - SLO| / SLO >= 0.15` for every D cell in the arm — this keeps the
> chosen threshold away from every cell's own value, not just away from one
> reference cell). If no value satisfies both conditions simultaneously,
> **that fact itself is reported** — "the TTFT axis is ill-posed for this
> grid" is a valid, citeable (post claims-auditor) outcome, not a failure to
> route around by picking a metric-cliff-adjacent value anyway.

Implemented as `ttft_site_check()` in `e1_analyze.py`, runnable against
either capacity-scan data (`--capscan-dir`, before the main sweep — the
intended use) or the main sweep's own pooled TTFT percentiles (post-hoc
cross-check, `--dir` mode).

**4.3.4 Distribution-level flags (auto-computed, implemented in
`e1_analyze.py`, reused identically whether run against capacity-scan or
main-sweep data):**

- **`CLIFF HAZARD`**: a (arm, D) cell's measured ITL-p95 falls within ±15%
  of the 60ms primary SLO (or of whichever ladder rung is being evaluated).
  Flagged cells report the **full ladder**, not a single collapsed number,
  in every downstream table. **M8 is flagged at risk from the FINDINGS
  data already in hand** (SM16 p50=58.44ms — a p95 sits close enough to the
  60ms boundary that rep noise alone could flip its bind/no-bind status;
  this is exactly why the ladder is pre-registered rather than added after
  seeing which way M8 breaks).
- **`ITL-NONBINDING`**: an arm where the measured ITL-p95 stays comfortably
  (≥15% margin) below the ladder's *lowest* rung (50ms) at **every** D cell.
  T8 is flagged as the likely candidate from the FINDINGS data (SM16
  p50=30.20ms) — if confirmed, T8's ITL term is effectively a constant
  across D and the decision rule degenerates to TTFT-only for that arm.
  **An `ITL-NONBINDING` arm's "no D beat best-static" result must NOT be
  reported as evidence that the decode-SM lever is net-negative** — it is
  evidence that this SLO ladder never tested the lever for that arm. The
  50ms rung is specifically kept in the ladder to give T8-like arms one more
  chance to bind before being written off.
- **TTFT margin violation**: any candidate TTFT SLO that fails the §4.3.3
  siting rule for a given arm — reported per-arm, not silently substituted
  with a "close enough" value.

### 4.3.5 ★★★NEW 2026-07-31 — the symmetric hole, the ITL staircase, and the
    on-cliff cell exclusion rule (PRE-REGISTERED BEFORE THE M8/Ha8/Hs8
    CAPACITY SCANS LAND — jobs 870295/870296/870297 were queued and
    unstarted when this was written)

Three pre-registrations, all forced by claims-auditor's 2026-07-31 audit of
job 867231. **Timing matters for their validity**: no capacity-scan data
exists yet for M8, Ha8 or Hs8, so these rules are fixed in advance for those
arms. T8's data does exist — but T8 is *already* classified `ITL-NONBINDING`
by the pre-existing §4.3.4 flag, and **none of the rules below can change
T8's verdict**, which is what keeps them from being rules fitted to T8.

**(a-1) `ITL-ALWAYS-BINDING` — the missing mirror of `ITL-NONBINDING`.**

§4.3.4 flags the case where every cell's ITL-p95 sits ≥15% *below* the
ladder's lowest rung: the ITL term is constant across D, so conjunctive
goodput degenerates to TTFT-only and a "no D wins" result must not be read as
lever-negative. The mirror case — every cell's ITL-p95 *above* the ladder's
highest rung — has the identical structure (ITL constant across D) and the
identical misreading risk, and was not covered.

The two are **not symmetric in consequence**, which is why a label alone is
insufficient:

| | `ITL-NONBINDING` (all pass) | `ITL-ALWAYS-BINDING` (all fail) |
|---|---|---|
| conjunctive goodput | reduces to TTFT-only | **0 in every cell** |
| a comparison still exists? | yes, degenerate (TTFT ranking) | **no** |
| paired bootstrap | works | 0 vs 0 → zero variance, degenerate CI |
| §4.4's "≥3% margin" | applies | **0/0, undefined** |

⇒ `e1_analyze.py` must carry a **numeric guard**: when best-static's combined
goodput is 0 (or all cells' are equal within float tolerance), the decision
rule does not run and reports `ITL-ALWAYS-BINDING / DEGENERATE` instead. Left
unguarded it would divide by zero or, worse, emit a confident verdict from a
vacuous comparison — the failure mode this campaign has now hit twice in one
day (bug #7's stale-file verdict, and the pre-fix capscan report).

**Is the risk real?** From `FINDINGS_8B_2026-07-28.md` §2's batch-12 ITL p50
scaled by T8's own batch-12→48 factor (×1.66) — an extrapolation, not a
measurement, and the factor is derived from one arm so it is not guaranteed
to transfer:

| arm | ≈d16 | ≈d92 | vs ladder {50,60,80} |
|---|---|---|---|
| T8 | 50 | 21 | all below → `ITL-NONBINDING` (measured, confirmed) |
| Hs8 | 67 | 26 | crosses → decidable |
| M8 | 97 | 33 | crosses → decidable |
| **Ha8** | **149** | ~52 | d16 above every rung, other end near 50 → **borderline** |

Only Ha8 is at risk and probably still crosses. The scans settle it.

**(a-2) Per-rung 4-way classification replaces the two binary flags.**

The deeper finding is that on this substrate the ITL axis is **not a
continuum**. Because `--max-running-requests` truncates the decode batch (see
§4.3.6), each cell's per-request ITL-p95 distribution is a near-deterministic
spike, and the axis is effectively a staircase of ~5 constants — for T8,
{15, 24, 26, 37, 50} ms. Measured consequence at rate 16, d16:
`frac(ITL ≤ 48) = 0.34/0.29` but `frac(ITL ≤ 50) = 0.94/0.99` — **moving the
SLO by 4% triples the ITL goodput.** "Binding" is therefore not a smooth
property; the answer is determined by which constants the rung falls between.

Every rung `r` is classified by one quantity, its relative distance to the
nearest cell, `m = min_cells |ITL_p95(cell) − r| / r`:

| condition | classification |
|---|---|
| every cell ≥15% below `r` | `ITL-NONBINDING` |
| every cell above `r` | `ITL-ALWAYS-BINDING` (new) |
| `m < 15%` | `CLIFF HAZARD` (existing) |
| otherwise | **`DISCRIMINATING`** |

**Pre-registered: a headline result may only be reported from a rung
classified `DISCRIMINATING` for that arm.** All rungs are still reported. This
derives all three flags from a single quantity and removes the freedom to
pick a rung after seeing which one is favourable.

⚠️ **Accepted in advance**: the 60 ms primary rung may be non-`DISCRIMINATING`
for every arm, in which case **E1 produces no headline at its pre-registered
primary SLO.** That is a result, not a failure, and it is exactly the state
§4.3.1 already describes as "a design failure, not a null result" — to be
reported as such rather than repaired by moving the SLO.

**(b) On-cliff cell exclusion (the d92 problem).**

d92 = `[16, 92]` is simultaneously the *most* faithful cell (decode duty cycle
0.52 vs d16's 0.11, §4.3.6) and the *least* comparable one:

| | d92 | others |
|---|---|---|
| low-load TTFT plateau | 181–196 ms | 47–66 ms (ρ 3–4× higher at equal rate) |
| capacity knee | **2.80 req/s** | 12.6–16 |
| max decode batch | **38** (prefill admission binds first) | 48 (the cap) |

The batch difference is decisive: step time depends on batch, so **d92's ITL
is not the same quantity as the other cells' ITL**. Raising
`--max-running-requests` does not fix it — d92 never reaches the cap.

Pre-registered rule:

> A cell whose capacity-scan knee lies **below the chosen operating rate** is
> **excluded from the best-static argmax**. Excluded cells report their full
> percentiles and the exclusion is named in every table in which they appear.
> **The operating rate stays common to all cells** — excluding a cell is not
> the same as giving it its own rate, which §4.2 forbids as a rate-confound.

This is a mechanical function of the measured knee, not a post-hoc choice.
Two consequences are accepted in advance rather than discovered later:

1. **The excluded cell may differ by arm.** T8 loses its prefill-starved end
   (d92, knee 2.80; likely d54, knee 8.45). M8/Ha8, whose decode is 2–3×
   slower, will plausibly lose the *decode*-starved end (d16) instead. Then
   "which D won" is answered on a different grid per arm, so **cross-arm
   comparison of the winning D cannot be a headline.** This is a physical fact
   about differing feasible regions, not a defect of the rule.
2. ★★ **E1 may be structurally unable to answer C2.** If the rule removes the
   decode-rich end for every arm, then E1's feasible operating region and C2's
   lever (established as the D16↔D92 contrast) live in **disjoint regions**.
   The honest verdict is then *"E1 as designed cannot reach this question"* —
   **not** "the lever is net-negative". This is a design-level finding and the
   four queued jobs settle it **before** the main sweep spends GPU time.

### 4.3.6 ★★NEW 2026-07-31 — `--max-running-requests 48` is an unregistered
    constant that sets the ITL axis, and the decode-side duty cycle is
    unmeasured by the pre-registered gates

Recorded here because §4.3.4/§4.3.5's verdicts are all downstream of it.

`--max-running-requests 48` is set in `e1_capacity_scan.sbatch:143` and
`e1_sweep.sbatch:178` and appears **nowhere else in this document** (grep 0
hits before this revision). Evidence that it, not the model, sets the ITL
ceilings (`decode_duty_check.py` on 867231, reproducing the auditor
independently):

- `decode_running_batch_size` truncates at exactly **48** in d16/d24/d44/d54
  (d92 reaches only 38 — prefill admission binds first).
- `kv_occupancy` at that truncation is **0.024** — memory is ~40× from
  binding, so this is a config cap, not a resource limit.
- d16's ITL-p95 is **flat at 50.2–50.6 ms from rate 12 to 32**; an open queue
  would keep climbing with the batch.

⇒ The per-cell "ceilings" {d16 50.6, d24 37.9, d44 26.2, d54 23.9, d92 19.1
ms} are a **harness property**. `batchcap.sbatch` (job 870301) tests this
directly with cap ∈ {48, 96, 192} and a pre-registered three-way decision.
Until it lands, every ITL conclusion in this campaign carries the scope
`max_running_requests=48`.

Separately, the **decode-side realized-partition duty cycle covaries with D**:
time-weighted, cells sit at their labelled decode SM only 0.110 (d16), 0.112
(d24), 0.166 (d44), 0.201 (d54), 0.518 (d92) of decode-active time; the rest
runs unpartitioned at 108 SM (`CONSENSUS.md` §1-22 auto-revert). **The D axis
therefore moves two variables at once** — the SM limit, and the fraction of
time it is in force — and the second increases monotonically with the first.
§5's pre-registered pin gate checks the **prefill** side only and is
structurally blind to this: `D=16 (P92) pin_frac=0.950` is a statement about
prefill sitting at 92 SM. `decode_duty_check.py` reports it per cell.

It is deliberately **not** made cite-blocking: §5 pre-registered exactly two
gates, and adding a third threshold after seeing 867231's data would be
choosing a gate from the data — the very move §4.3.3's siting rule exists to
prevent. It is reported and escalated. (On 867231 a 0.80 threshold would fail
**every** cell including d92, which is itself the finding.) The 11–52% figures
are first-order: they come from the same `runtime_snapshot` sampling the pin
gate uses, so they are themselves a re-measurement target for an engine-side
accumulated per-partition timer. What is **not** sampling-sensitive is the
ordering and its monotone covariation with D — which is what makes it a
confound rather than a rounding error.

None of this changes the workload/rate grid (§4.2) or adds GPU time — it is
purely how the already-collected TTFT/ITL distributions are read, so it has
no effect on the time budget (§7 unchanged; see also §11 for the explicit
before/after note the coordinator asked for).

### 4.3.7 ★★NEW 2026-08-02 — the knee estimator becomes code; the rung table
    must be built at ONE common rate; and the ITL estimand is chosen here,
    before the sweep (M1 + M2, zero GPU time)

Written after the 2026-08-02 claims-auditor referral, which returned
**2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED** on the 2026-08-01 claims.
Two of its findings are defects in *this document's* machinery rather than in
any measurement, and both are fixed here.

**(a) The knee is now `e1_analyze.py:knee_rps()`, with pre-registered
parameters.** §4.3.5(b) calls its exclusion rule "a mechanical function of the
measured knee, fixed before the M8/Ha8/Hs8 scans landed". That was true of the
*rule* and false of the *input*: the knee itself was hand-computed, lived only
in a handoff table, and appeared in no script and in no pre-registration — the
auditor had to reverse-engineer its definition from the 20 published numbers in
order to audit it. A quantity that gates the whole sweep cannot sit outside
version control. Pre-registered here:

    plateau = mean TTFT-p50 over the 3 lowest-arrival_rps probes
    knee    = arrival_rps of the first probe exceeding 2.0 x plateau
    x-axis  = RNG-replay realized arrival_rps, one point per (rate, seed)
    y-axis  = TTFT-p50, nearest-rank (matching the sbatch's own CAPSCAN lines)

`KNEE_PLATEAU_N=3`, `KNEE_MULT=2.0`, grid `(2,3,4) x (1.5,2.0,3.0)`. This is a
*fixing*, not an invention: the codified estimator reproduces all 20 published
cells **exactly**, so nothing was tuned to taste.

Three consequences, all now printed by `--knee-scan`:

1. **The knee's range is the probe grid.** It can only return one of the ~10
   realized rates actually visited, so cross-arm agreement is partly forced by
   the support. "The knee is a common 2.80 across all four arms" is therefore
   **retired as a claim**: over the 9-variant grid the d92 knee set is
   T8 {2.80, 4.20}, M8 {2.05, 2.80, 3.08}, Ha8 {2.05, 2.80, 3.08, 4.10},
   Hs8 {2.05, 2.80, 4.10}. Same failure family as `arrival_rps` and
   `kv_mamba_occupancy` (`PROJECT_STATUS.md` gate #6): a number whose support
   guarantees the agreement.
2. **What the exclusion rule actually consumes is the ORDER**, not the value.
   That part is robust: d92 is the first cell to fall in **9/9** variants for
   T8, M8 and Hs8, and 8/9 for Ha8 (the ninth is a tie with d54, not a
   reversal), with a gap to the next cell of 1.00–5.73x. **Claims must be made
   on the order; the absolute rate may not be quoted without its grid.**
3. **Three knees are FRAGILE** — set by a single probe clearing threshold by
   <5%: Hs8 d16 (2.3%), Hs8 d44 (3.2%), Ha8 d54 (3.4%). ★The 2026-08-01
   handoff attributed Hs8 d16's fragility to the `arr=12.61 seed=1` outlier;
   that is **wrong and the outlier is irrelevant to it** — the knee is set by
   `r8 seed=2` at 172ms against a 169ms threshold, and deleting the outlier
   leaves the knee at 9.09 unchanged.

**(b) The rung table must be built at ONE rate for all arms —
`e1_analyze.py --common-rate`.** §4.3.5(b) already pre-registers "the operating
rate stays COMMON to all cells — ... which §4.2 forbids as a rate-confound",
but the published 4-arm rung table violated it: M8/Ha8/Hs8 were classified at
rate 2 and **T8 at rate 12**. Running `--capscan-rate` once per arm made this
easy and invisible; the new mode takes one rate for all arms and prints it in
every header. **Pre-registered common rate = 2 req/s**, chosen by a rule that
does not look at ITL: it is the highest probe rate at which every non-d92 cell
of every arm is off-cliff under **all 9** knee variants (the binding constraint
is Ha8 d54, min knee 3.08). d92 is excluded at this rate under every variant.

**(c) The ITL estimand — decided now, on data, before the sweep
(`estimand_check.py`).** The auditor argued the registered per-request ITL-p95
is not decode step time but "worst stall a short request sat through". Measured
at the common rate over 40 probes, the picture is **real but localized**, and
two parts of the stated mechanism do not hold:

| | measured |
|---|---|
| `output_len <= 20` | 15.4–25.6% of requests |
| requests whose per-request ITL-p95 **is** their max ITL | **0.0–5.1%** (the interpolating percentile rarely returns the max — this part of the argument does not hold literally) |
| probes with a server-wide stall (>=3 requests sharing one max, that max >5x the probe's typical ITL) | **17/40**, and **0/10 in T8** |
| A/B (registered estimand / per-request-median estimand) | 1.11x – **22.59x** |

The corruption concentrates in **d92** (A/C = 4–19x) and in M8's mid cells at
seed 1 — i.e. **mostly in the cell the on-cliff rule already excludes**. For
the non-excluded cells of T8, Ha8 and Hs8, A and C agree within a few percent,
so Ha8's ITL-ALWAYS-BINDING is **not** a stall artifact.

Decision, pre-registered: **keep A (per-request ITL-p95) as the SLO term** —
it is CLAUDE.md gate #4 and swapping it would be an unforced estimand change —
and additionally report **C (`output_len >= 40` only)** and the **stall counter**
for every cell, with the rule that *any headline whose label differs between A
and C is reported as estimand-sensitive and may not be a headline*. Measured
label instability under the swap: **4/24** (M8 s1 at 50/80ms, M8 s2 at 80ms,
Hs8 s1 at 50ms).

**(d) What survives all three corrections.** Rebuilt at the common rate, with
d92 excluded, under **both** estimands and **both** seeds: **no arm has a
DISCRIMINATING rung at 50, 60 or 80 ms.** `HEADLINE-ELIGIBLE RUNGS = NONE`
therefore stands — but it now stands on a rate-clean, estimand-robust,
code-reproducible basis, whereas the arm x rung *table* published on 2026-08-01
does not survive and is superseded by `--common-rate` output. Note this is a
statement about **where the ladder sits**, not about the lever.

None of this adds GPU time (§7 unchanged).

### 4.3.8 ★★★NEW 2026-08-02 — M4 closed offline (the stalls are monolithic
    prefill, and that is structural); M5 retired as vacuous; M3 re-scoped
    into the Transformer-control contrast. **This is the pre-registration
    for `e1_m3_control.sbatch`.**

#### (a) M4 — CLOSED with zero GPU. The stall is prefill, not a mystery.

Reconstructing absolute token-emission times (arrival replay + TTFT + cumulative
ITL) and locating each probe's single largest ITL shows: in **16 of 17** stall
probes a request was **mid-prefill for the entire stall**, and in almost all of
them that request carried the probe's **longest prompt** (2469–2776 tokens).
The stall size is monotone in D for a fixed prompt — Ha8 seed 1: d24 167.7ms →
d44 225.6 → d54 263.6 → **d92 865.6** — exactly as prefill SM = 108 − D shrinks.

★ The 2026-08-01 handoff's guess (`--chunked-prefill-size -1` lets one long
prompt occupy the prefill window) was **right**, and the 2026-08-02 auditor's
REFUTED verdict on it used the wrong test: it checked the max ITL of the long
request *itself*, but a request being prefilled is not yet decoding, so it
cannot observe its own stall — the already-decoding requests do. Restore the
mechanism; keep the auditor's separate, correct finding that this outlier does
not set the Hs8 d16 knee (§4.3.7(3)).

★★ **It is not fixable and not a bug.** `server_args.py:6130` asserts
`chunked_prefill_size == -1` whenever `enable_pdmux` is set ("PD-Multiplexing
is not compatible with chunked prefill"). Un-chunked prefill is a *precondition*
of the substrate under study, so this belongs in the **(A) green-context-
dependent** bucket of `reports/paper/venue_positioning.md` §0.1 — a cost of the
one deployable vendor primitive, not a property of hybrid models.

**Pre-registered consequence — the ITL decomposition.** Because the blocking
term grows with D while the decode term (C2) shrinks with D, the registered ITL
statistic mixes two opposite-signed effects. From now on every cell reports
both, and the pair is fixed here so it cannot be chosen later:

    A_all   p95 across requests of per-request ITL-p95, all ITLs  [SLO term,
            unchanged -- CLAUDE.md gate #4]
    A_free  same, dropping every ITL whose interval overlaps the prefill
            window (arrival -> first token) of a request with
            input_len >= PREFILL_BLOCK_TOK

`PREFILL_BLOCK_TOK = 1024`, fixed now. **Known limitation, stated in advance:**
at d92 prefill holds only 16 SM, so prompts *below* the threshold also block,
and `A_free` stays contaminated there. `A_free` is therefore interpreted **only
over d16–d54**, never at d92.

#### (b) M5 — RETIRED as an experiment; replaced by an assertion (M6 folded in)

The proposed cap TOST would have been run at the operating rate, where measured
max concurrency is **12–30** (44 at Ha8 d92) against a cap of 48. A cap that
cannot bind cannot produce a non-equivalence, so the test is **guaranteed to
pass** — the same vacuous-statistic failure as `arrival_rps` and
`kv_mamba_occupancy` (gate #6). Spending 0.5 GPU-h on it would buy a foregone
conclusion.

Replaced, at zero GPU cost, by three pre-registered items:

1. `--max-running-requests` **48, kept unchanged** — ★a 96 was drafted and
   rejected. The only requirement on the cap is that it cannot bind, and at the
   common rate measured max concurrency is 12–30 (44 at Ha8 d92, not run in M3),
   which 48 already clears with ~60% headroom. Raising it to 96 would **double
   the SSM state pool on Ha8** (0.141 GB/slot: 6.78 → 13.5 GB), and that memory
   comes out of the attention KV pool — already only 52% subscribed on that arm.
   That is a memory-split confound purchased for no benefit; the fix for the
   double-knob problem is item 2, not a larger cap. **and**
2. `--max-mamba-cache-size` set **explicitly and equal to the cap** — an
   arm-common *rule*, deliberately **not** an arm-common absolute constant:
   per-slot cost is M8 0.255 / Ha8 0.141 / Hs8 0.096 GB, so a shared absolute
   pool would force a *different* memory split on each arm, a new cross-arm
   confound. This also breaks the `kv_mamba_occupancy` identity
   (`model_runner_kv_cache_mixin.py:223-230`) by taking the explicit branch at
   `:218`, so mamba occupancy becomes an independent signal for the first time.
3. A per-probe **assertion**: realized `max_concurrent_requests < 0.8 x cap`,
   and each arm's `#KV tokens` / `kv_full_occupancy p99` printed. Violation is
   an escalation, not a silent footnote.

#### (c) M3 — re-scoped to the Transformer-control contrast (the only GPU job)

The original M3 ("is Ha8's ITL-ALWAYS-BINDING powered?") is already answered by
the CI at the common rate: Ha8's best cell is 90.4ms with a lower bound of
88.0 > 80ms. The question worth GPU is the one M4 exposed. Offline, over
d16→d54 with blocking removed, the decode-SM response splits by arm:

| arm | `A_free`(d16)/`A_free`(d54) | seeds |
|---|---|---|
| T8 | **2.03x** monotone | 1, 2 agree |
| Hs8 | 1.56x | agree |
| M8 | 1.39x / 0.87x | **disagree** |
| Ha8 | **1.03x / 0.89x** | agree — no response |

This is Tension A (HE2 vs C2) at its sharpest: C2 measured 2.36–2.91x on **all
four arms** at batch=1, yet at serving concurrency T8 keeps ~2x and Ha8 shows
none. ★ The two are **not the same configuration** — C2 pinned prefill at 16 SM
and swept decode alone (pure elasticity), whereas every E1 cell is
**complementary** (D + P = 108), so raising D necessarily starves prefill. That
difference is not a flaw in either; it is precisely the content of C2's
pre-registered branch *"the lever exists but is net-negative under the budget
constraint"*.

**Design.** Arms **T8 (positive control) + Ha8**; cells **d16, d24, d44, d54**
(d92 excluded from the argmax by §4.3.5(b) and unreadable for `A_free` by (a),
so not run); **one common rate = 2 req/s** (§4.3.7(b)); **8 blocks x 1 seed**;
`NP_MIN = 200`. Measured cost ≈ **3.5 h** (boot 26 s median / 59 s p90 and probe
tail 15 s median, both measured from the 2026-08-01 logs, not assumed).

- ★★ **A BLOCK, not a seed, is the replication unit — and the first draft of
  this design got that wrong.** `g` pairs d16 with d54, and a static split
  cannot change without a reboot (`multiplex/multiplexing_mixin.py:155`), so the
  two cells of `g` **always come from different boots**: boot-to-boot variance
  does not cancel in `g`, it enters it directly. Seeds sharing a boot are
  therefore not independent replicates of `g`. The draft's "4 seeds in 2
  boot-pairs" had `n_independent = 2`, which tolerates `sd(g) <= 0.021` — it
  could never have fired. This is the same pseudo-replication the auditor found
  in batch-cap, reproduced inside the fix for it.
  A **block** = one fresh boot of every cell, run at one seed (seed = block
  index, so workload and boot vary together); `n_independent = BLOCKS`.
- **How many blocks.** The binding side is the test arm's upper bound against
  `G_FLAT = 1.15` (gap 0.19 from the offline `g = 0.96`):

  | blocks | tolerable `sd(g)` |
  |---|---|
  | 2 | 0.021 |
  | 4 | 0.119 |
  | 6 | 0.181 |
  | **8** | **0.227** |

  Observed `sd(g)` across this campaign (from 2 seeds, so itself unstable):
  T8 0.102, Hs8 0.037, Ha8 0.104, **M8 0.223** — M8's `g` moved 1.387 → 1.071
  between two seeds. **8 blocks is the smallest value covering that worst case.**
- **`NP_MIN` 300 → 200 pays for blocks 2 → 8.** This moves power from *within* a
  cell to *between* independent replicates, and the verdict's CI depends only on
  the latter. 200 still leaves ~194 post-warmup requests — 5x the 39 that made
  the §4.3.7 per-cell labels UNPOWERED — so the per-cell reliability the extra
  100 requests would have bought is not the binding constraint. `NP_MIN`, not
  `PROBE_TARGET_S`, remains the knob: at rate <= 5 the probe size is
  `max(NP_MIN, PROBE_TARGET_S x rate)`, so 8 → 20 would change nothing.
- **Counterbalancing.** Odd blocks run the cells forward, even blocks reversed,
  so cell is crossed with boot/time order rather than confounded with it.
- ★ **The interval is a t-interval, not the percentile bootstrap** used
  elsewhere in this campaign. Simulated coverage of a nominal 95% interval
  (4000 trials): n=4 → bootstrap **79.8%** (width 1.52) vs t 94.5% (2.93);
  n=6 → 85.0% vs 95.0%; n=8 → 89.0% (1.24) vs 95.7% (1.63). At these n the
  percentile bootstrap is about half the width it should be, so "the CIs are
  disjoint" would fire far too easily — on the rule that decides the campaign.
  The bootstrap is still printed, and is not read by the verdict.
- **Guard:** `m3_analyze.py` refuses a verdict below **6** independent blocks,
  because at fewer a null is indistinguishable from insufficient replication.
- **In-run off-cliff assertion, replacing a capacity re-scan.** The knees in
  §4.3.7 were measured at cap 48; this job changes the cap. Rather than re-run
  the scan, each cell asserts in-run that rate 2 is still on the plateau: TTFT-p50
  must be <= 2 x the cell's §4.3.7 plateau. Failure escalates and voids that cell.

**Decision rule, pre-registered** (in code: `m3_analyze.py`, written before the
job runs — §4.3.7(a) is the lesson being applied). Let
`g(arm, block) = A_free(d16)/A_free(d54)`, paired **within block**, with a
**block-clustered t-interval** over the 8 blocks.

| outcome | verdict |
|---|---|
| T8 `g >= 1.5` **and** Ha8 `g <= 1.15`, CIs disjoint | the ITL-lever asymmetry is **attributable to the arm** — the defensive asset against DuetServe's Transformer result. E1 proceeds, reporting per-arm. |
| **both** `g >= 1.5` | the lever is present at serving concurrency in both — proceed to the E1 main sweep **(A)**, ladder re-sited by the measured `A_free` range. |
| **both** `g <= 1.15` | the lever is absent at serving concurrency on this substrate → close E1 as **(B)**, with the correct reason: *the ITL term of conjunctive goodput is not a function of the decode-SM lever here* — **not** "the exclusion rule deletes the lever", which §4.3.7 and the C2 log-range both refute. |
| anything else (incl. seed disagreement, as M8 already shows) | **no verdict**; report and stop. Do not re-cut the thresholds after seeing the data. |

Scope stated in advance: one rate, one context regime, two arms, `A_free` valid
only over d16–d54. **This job cannot and does not decide whether the lever pays
off in goodput** — it decides whether the ITL axis responds to D at all, which
is a precondition for that question being askable.

★**Recorded inconsistency (2026-08-02) — RESOLVED FORWARD-ONLY 2026-08-03.**
The rule above is written on **point estimates** ("T8 `g >= 1.5` and Ha8
`g <= 1.15`, CIs disjoint") and `m3_analyze.py` implemented exactly that. The
blocks-vs-seeds power calculation earlier in this section instead framed the
test-arm side as *upper bound* `< 1.15`. On 872077 the two disagree — Ha8's
t-CI upper bound is 1.190 — so the deciding rule was ambiguous exactly where it
mattered.

**Resolution, pre-registered 2026-08-03 and binding only on jobs submitted
after this revision:**

> **RULE_BOUNDS** — each arm is judged on the CI bound **facing** its
> threshold: control arm needs **lower** bound ≥ `G_LEVER`, test arm needs
> **upper** bound ≤ `G_FLAT`, plus disjoint CIs.

Rationale: the test arm's half is an **acceptance of a null** ("Ha8 does not
respond"), and a null accepted on a point estimate alone is not evidence.
Bound-facing-threshold makes both halves symmetric and makes the null actually
evidenced. ★Note the direction: **RULE_BOUNDS does not fire on 872077** while
RULE_POINT does — the stricter rule is the one that cuts against the earlier
apparent result, which is why it can be adopted honestly and why it binds
**only going forward**.

For 872077 and any other job predating this revision, `m3_analyze.py` prints
**both** readings and, when they disagree, returns **NO VERDICT** — because
"the deciding rule was ambiguous where it mattered" is the truthful verdict for
such a job. Neither reading may be adopted for it after the fact.

#### (f) ★OUTCOME of job 872077 (2026-08-02) — NO VERDICT, and why that is a result

Clean execution: 64/64 probes, zero errors, all three in-run gates PASS,
`n_keep=194`, 3h01m. **The pin gate (e) then voided 19 of 64 cell-blocks, `g`'s
numerator among them** — T8 d16 2/8 PASS, Ha8 d16 5/8; d54 8/8 on both arms.
Per (e) both arms lose `g`, so **there is no verdict**, and on the surviving
pin-passing blocks `n_indep` is 2 (T8) and 5 (Ha8), below the guard of 6.

Two things this exposed, both now fixed or recorded:

1. **The gate was cite-blocking but not machine-readable.** `m3_analyze.py` did
   not read the pin output and printed a confident
   "ASYMMETRY ATTRIBUTABLE TO THE ARM" over voided cells. Now wired
   (`load_pin()`), with **absence of the pin file refusing the verdict** —
   "not run" must never read the same as "passed".
2. **The failure is substantive, not a power artifact.** 15 of the 19 failures
   have `pin_frac < 0.90`; T8 d16 runs as low as 0.61, i.e. **20–40% of
   prefill-active time executed at P108 (no split) instead of the target P92**.
   The pattern is systematic — worse the *larger* the prefill partition
   (d16, d24) and worse on the *faster-prefill* arm (T8) — the same direction
   as `CONSENSUS.md` §1-22's auto-revert asymmetry, though the mechanism is not
   independently verified here. Prefill-active time totals only 1.5–7 s inside
   a ~200 s probe, so episodes are few (6–34) on top of the genuine unpinning.

⇒ **The open question is now realizability, not policy**: is there any
`(D, rate)` region where a *low-D* cell actually holds its target partition? If
not, "d16" is not an operating point on this substrate at this load, which is
the same class of error as Stage 0's D108 (label ≠ realized) — caught by a gate
this time rather than after publication. Measured by `m3r_realizability.sbatch`
(diagnostic only: no SLO, no goodput, no decision rule).

Direction-of-bias note, **not usable to rescue the verdict**: unsplit execution
gives decode 108 SM where the label says 16, so `A_free(d16)` is biased *low*
and `g` therefore *under*-estimated — more so on T8, which pins worse. That
would enlarge the asymmetry, not create it. It remains an untested inference.

#### (d) What is deliberately NOT re-run

- **Capacity scans (867231, 870295–297).** Their knees survive as an *order*
  (§4.3.7(2)), which is all §4.3.5(b) consumes, and the cap change cannot move
  the low-rate end where the binding knee (d92) sits. Covered by the in-run
  assertion above instead.
- **batch-cap (870301).** It was run at rate 16, ~5.7x above the common
  off-cliff band, so it answers a question outside E1's operating region. Its
  claim was already NOT-YET-SUPPORTED/REFUTED (§(b)); it is **retired**, not
  repeated.
- **s8_scaleup C2.** Audited and scoped; the prefill-pinned configuration is a
  *feature* of that measurement, now explicitly contrasted above.

#### (e) The pin check — ★REVISED 2026-08-02: no GPU job needed after all

The first version of this subsection said the pin check "MUST be re-run" as a
**separate short trace-force-ON run**, because job 867298's `PIN_CHECK` crashed
on an argument-order bug and left no pin data. That requirement was wrong on
its own terms, and is replaced by `m3_pin_check.sh` (zero GPU):

1. **The time-weighted pin gate does not need `PDMUX_TRACE_FORCE_PREFILL`.**
   `e1_capacity_scan.sbatch` never sets it (engine default `"0"`,
   `multiplex/multiplexing_mixin.py:104`) and still produced **passing** gates
   on every cell M3 uses — T8 d16/d24/d44 `pin_frac` 0.950–0.985, Ha8
   d16/d24/d44/d54 0.994–1.000, `n_episodes` 31–144. Trace-force only adds
   prefill-active samples; it was never a precondition. So pin data does exist.
2. **Forcing it would inject the exact observer effect §4.7.1 exists to keep
   out of an ITL measurement** (+2.0% [+0.78, +3.21] on d92's `itl_p95`).
3. **But the capacity-scan gates alone do not certify the operating point**:
   they aggregate over *all* probe rates, and `CONSENSUS.md` §1-22 shows
   green-context reverts to no-split when decode is empty — which is *more*
   likely at low rate. What is actually needed is a **rate-2-only** gate.
4. **M3's own telemetry is exactly that.** Each `e1m3_<arm>_<cell>_<block>`
   file holds a single rate-2 probe, so the whole-file gate *is* the
   rate-restricted gate — no window arguments, no episode reconstruction, no
   second run, and trace-force stays OFF.

`m3_pin_check.sh <job>` is **cite-blocking and must be run before reading the
verdict**: any cell failing it is VOIDED exactly like the in-run gates, and if a
voided cell is an arm's d16 or d54 then that arm has no `g`.

#### (g) ★★2026-08-03 — the pin gate's instrument was inadequate, (e) was wrong,
    and M3R rev1 could not answer. Pre-registration for `m3r2_pinforce.sbatch`.

**The correction.** Subsection (e) removed §4.7.1's separate trace-force-ON pin
run, arguing "the gate does not need `PDMUX_TRACE_FORCE_PREFILL` — the capacity
scans passed without it". **That argument was wrong**, and job 872236's own
telemetry shows why: the capacity scans pooled **eight rates** into one file,
while a single-rate probe does not. Measured, trace-force OFF:

| cell | total snapshots | prefill-active |
|---|---|---|
| Ha8 d16 rate 2 | 10,558 | **2** |
| T8 d16 rate 2 | 10,993 | **8** |
| Ha8 d54 rate 2 | 9,906 | 16 |
| T8 d54 rate 2 | 10,921 | 28 |

So the gate that voided M3 (f), and the gate that then "passed" Ha8 d16 at
`lower95=1.000` in rev1, were both computed from **2–28 samples**. Neither job
measured realizability. §4.7.1's original requirement is **restored**: pin
measurement gets trace-force ON, in a job that emits **no ITL number at all**,
which is what makes the +2.0% observer effect irrelevant there.

**Two reporting bugs fixed in `e1_pin_check.py` (2026-08-03).** The
`[UNRELIABLE: only n_episodes=…]` note rode on `reason`, and `reason` prints
only on FAIL — so an unreliable **PASS was silent**. And `n_pa_snapshots`, the
gate's actual sample size, was computed but never printed. Both now always
appear. This is the same "not-measured looks like passed" failure the analyzer
wiring in (e) fixed one layer up.

**rev1's second defect: seed and boot were confounded.** M3 ran
`seed == block == boot`, so "seeds 6,7,8 failed" cannot separate a workload
effect from a server-state effect. rev1 then used a **single seed (1)** — which
had *passed* in M3 on both arms — and so sampled none of the failing region,
while its header asserted the variation was "an engine behaviour, not a
workload effect". M3's own per-block numbers refute that assertion.

**rev2 design.** Arms T8 + Ha8; `(cell,rate)` ∈ {d16:2, d16:6, d24:2, d54:2}
(d16 = the voided cell at the operating rate *and* under rev1's load
hypothesis; d24 = also voided on T8; d54 = the control that never failed);
**3 boots × 4 seeds, the same seeds inside every boot** — crossed, so seed and
boot are separable for the first time. Trace-force ON; one telemetry file per
**boot**, windowed per probe with `--t0/--t1`.
★The window clock is `time.perf_counter()` to match `telemetry.py:23` exactly.
Both resolve to `clock_gettime(CLOCK_MONOTONIC)` here (verified 2026-08-03),
but `traceforce_gate.sbatch` paired `monotonic()` against `perf_counter()` and
its pin checks all crashed, so that pairing was never validated end-to-end.

**Instrument sanity gate, pre-registered:** a probe leaving fewer than
`MIN_PA_SNAPSHOTS = 200` prefill-active snapshots is reported
**UNMEASURABLE** — neither passed nor failed. rev2's whole premise is that
sample size was the binding constraint, so sample size must itself be gated,
or rev2 repeats rev1's mistake with a larger number attached.

**Stated in advance — what rev2 cannot do.** The seeds {1,6,7,8} were chosen to
span M3's observed pass (1) and fail (6,7,8) blocks for Ha8 d16. That is a
**targeted replication, not a random sample**, so rev2 can say whether the
variation is seed-indexed or boot-indexed and whether M3's failures reproduce,
but it **cannot estimate an unconditional failure rate over workloads**. Do not
compute one from it.

**Consequence for M3.** Job 872077's verdict stays void, but the *reason* is
now narrower and more honest: not "d16 does not hold its partition" — that was
never measured — but **"the pin gate had 2–34 samples, so the cells were never
certified either way."** Whether M3 must be re-run depends on rev2's answer.

#### (h) ★★★2026-08-03, claims-auditor — THE PIN GATE WAS AN IDENTITY.
    (e), (f) and (g) are superseded; so is M3R rev2's reason for existing.

**Verified independently over 120 telemetry files and 77,688 prefill-active
snapshots (jobs 872077 + 872236 + 872497), zero violations in either
direction:**

```
prefill_sms != target   <=>   decode_running_batch_size == 0
    off-target & decode-empty : 50,328
    on-target  & decode-busy  : 27,360
    violations (both ways)    :      0
```

`multiplex/multiplexing_mixin.py:773,792-794` drops to the unsplit partition
**by design** when the decode batch is empty, and `CONSENSUS.md` §1-22 already
records that fallback as *"the policy working as designed"*. So
`compute_time_weighted_pin_gate` never measured partition control — its
"pin_frac" is *the share of prefill-in-flight time during which decode happened
to be non-empty*. **Methodology gate #6 (do not use an identity as evidence),
committed by the very gate written to enforce correctness.**

**Corollary, from the same table:** the conditional pin fraction is
`27,360 / 27,360 = 1.000` exactly. **The partition is fully realized wherever
the question is well posed.** Re-scored, job 872077 is **64/64 PASS** and its
`n_indep = 8` on both arms is restored.

**What (e)/(f)/(g) got wrong.**
- (f)'s "19/64 voided, and 15 of them with `pin_frac < 0.90`" — those were
  decode-empty intervals, not unpinned ones.
- (g)'s "the instrument had 2–34 samples" — a real problem for *rev1*, but the
  numbers quoted for M3 were **rev1's** (2–39); M3's own `n_pa_snapshots` are
  **4–130**. A canonical section described M3's invalidity using another job's
  sample sizes.
- (g)'s premise that trace-force was needed at all. The auditor's SCHED-only
  decomposition (keep only `trace_forced != true` records in rev2, i.e. apply
  the OFF estimator to the ON system) puts the **system** effect of trace-force
  at **mean +0.001, sign 4+/2−** — the apparent improvement was the estimator's
  mesh, not the engine. **rev2 answered a question that did not need asking.**
- `MIN_PA_SNAPSHOTS` is **the complement of the estimand**: most prefill-active
  snapshots are the off-target decode-empty spin, so a *well-pinned* probe has
  *fewer* of them and gets flagged UNMEASURABLE. All 20 UNMEASURABLE probes in
  rev2 fell on the two highest-pin seeds; the two lowest-pin seeds had none.
  A gate built to prevent gate #6 violated gate #6 in reverse.

**★The finding that replaces them — the decode axis was never gated at all.**
`E1_DECODE_REALIZED`, time-weighted over decode-**active** time, job 872077
(n=8 blocks/cell, independently recomputed):

| arm | d16 | d24 | d44 | d54 |
|---|---|---|---|---|
| T8 | **0.038** | 0.047 | 0.082 | 0.093 |
| Ha8 | 0.104 | 0.110 | 0.148 | 0.187 |

The cell's decode split is realized over **4–19% of decode work time**; the
other 81–96% runs **unsplit at 108 SM**. Two consequences that must accompany
any reading of `g`:

1. **The E1 grid does not deliver a sustained decode-SM allocation**, so it is
   not measuring the quantity C2 measured (C2 pinned prefill and ran decode at
   D continuously). Tension A cannot be closed by `g` without this factor.
2. **The dilution varies by cell** (T8 0.038 → 0.093, Ha8 0.104 → 0.187), so
   `g = A_free(d16)/A_free(d54)` moves the SM *level* and the *engagement rate*
   together — **a confound inside `g` itself**.

This is the decode-side analogue of Stage 0's D108 error (label ≠ what ran),
and unlike Stage 0 there was **no gate here at all**. §1-22 required realized
distributions to be reported for partition sweeps; that was applied to the
prefill axis only.

**Job 872077's standing, as of 2026-08-02.** All gates pass, `n_indep = 8` on
both arms, `T8 g = 1.837 [1.666, 2.009]`, `Ha8 g = 1.068 [0.947, 1.190]`. The
verdict is **still NO VERDICT — but now for exactly one reason**: RULE_POINT
fires and RULE_BOUNDS does not (Ha8 upper 1.190 > 1.15), and per (c) a job
predating the 2026-08-03 fix cannot have either reading adopted for it.

★★★**SUPERSEDED, 2026-08-03 (same day, continued session) — §4.3.9 below.**
The sentence that used to sit here ("a re-run under RULE_BOUNDS would settle
it, and the open quantity is whether Ha8's upper bound falls below 1.15 with
more blocks") is **now wrong**. It assumed the only defect was CI width. §4.3.9
shows the defect is structural: on this grid "decode ran at D SM" and "prefill
was in flight" are the same event, so no amount of blocking resolves it — more
blocks would still be estimating a quantity that conflates decode-SM
elasticity with prefill-SM elasticity. Do not re-run under the current grid.
Any future re-run must report the decode-realization table beside `g` **and**
run on a sticky-partition substrate (§4.3.9).

**Not re-run, and why:** M3's pin question is answered (conditional pin =
1.000, three jobs). rev2 (872497) is retained as data but its stated purpose is
withdrawn. `MIN_PA_SNAPSHOTS` is withdrawn as a gate.

### 4.3.9 ★★★2026-08-03, same day, continued session — `g` retires on this
    grid: the "dilution attenuation" hypothesis REFUTED, the NO VERDICT reason
    widened to "estimand not identified", `A_free`'s own flaws, and an
    uncontrolled confound between arms. Source discipline: (a)-(d) below are
    **[AUDITED]** (claims-auditor independently re-analyzed the 872077 raw
    data — telemetry 64 files + bench 64 files — and these readings are
    citable in canon). (e) is **[UNAUDITED]** (result-analyst output,
    claims-auditor has not reviewed it; not citable in canon except the one
    item marked otherwise).

#### (a) The "dilution attenuation" argument — REFUTED

The main session, working from §4.3.8(h)'s `E1_DECODE_REALIZED` table (4-19%),
argued: `A_free(dD) = w_D * A(D) + (1 - w_D) * A(108)` is a mixture, and the
dilution attenuates `g` toward 1. Back-solving for Ha8's "true" split-only
response gave a corrected `g` of roughly 1.62-1.70 — which would have made
Ha8's NO VERDICT an engagement artifact rather than a genuine flat response.

**Verdict: REFUTED.** Three independent lines of evidence, run entirely from
872077's existing telemetry (no new GPU time):

1. **Control-arm reductio.** Apply the identical correction formula to T8.
   Corrected `g` comes out to **21-29x** (for `A(108) in {12,14,15}` ms,
   `b in {0,2}` ms), a **10x violation** of the pre-registered C2 measurement
   (2.36-2.91x over the wider 16-92 SM range) on the narrower 16-54 SM range.
   The model requires T8 d16's split-conditional ITL p95 to be 352-360ms; the
   measured value is **30.67ms**.
2. **De-engagement experiment.** Re-sample the split-labeled ITL tokens from
   the same cell's unsplit distribution to lower engagement by a factor `f`
   applied to `w`. Measured change in `A_free`: **1-11% only** (Ha8 d16
   113.51 -> 112.29, -1.1%; Ha8 d54 107.45 -> 95.74, -10.9%; T8 d16
   28.31 -> 26.97, -4.7%; T8 d54 15.38 -> 15.02, -2.3%). At `w = 0` (all
   engagement removed), `g` is Ha8 1.173 / T8 1.796 — the headline survives
   almost unchanged.
3. **The core assumption "`A(108)` is cell-invariant" is violated in the
   data.** Applying the `A_free` decomposition to sub-populations: Ha8 ALL
   1.068 [0.947, 1.190] / **SPLIT-only 0.920 [0.842, 0.998]** (CI excludes 1,
   **opposite sign**) / UNSPLIT-only 1.146 [0.997, 1.296]. T8 ALL 1.837 /
   SPLIT-only 1.847 [1.641, 2.053] / **UNSPLIT-only 1.795 [1.589, 2.001]** —
   the entire T8 headline effect **reproduces in the sub-population where the
   decode-SM contrast is zero by construction**.

Confound type: gate #1 (replacing a direct serving measurement with an
offline arithmetic model) plus **gate #6** (dividing by `w`, a quantity
derived from an identity, as if it were a free nuisance parameter — `w` is
prefill duty cycle itself, see (b)).

**What survives.** Low engagement itself is robust — three independent
instruments agree (snapshot time-share 3.8-18.7%, event-driven
`controller_decision`-based 2.7-8%, token-based). What dies is the
**correction**, not the premise. Aggregation-unit sign is settled (time-share
> token-share, so time-weighted engagement over-estimates: Ha8 d16 0.104 vs
0.092, d54 0.187 vs 0.177; T8 d16 0.038 vs 0.038, d54 0.091 vs 0.087) but is
moot — the model is already dead by (1)-(3).

#### (b) 872077's standing — NO VERDICT reason widened from "rule ambiguity"
    to "estimand not identified"

Code facts (auditor-confirmed): `pdmux_context.py:initialize_stream_groups`
hardcodes `SM_COUNTS = [(108,0)] + divisions + [(0,108)]`, and
`multiplexing_mixin.py:773,792-794` falls back to `real_sm_group_num - 1` =
plain `(0,108)` (not even a green context) whenever prefill is not in flight.
Therefore, on this substrate:

> **"decode ran at D SM" and "prefill was in flight at the same time" are the
> same event.**

No statistic on this grid can separate the decode-SM lever from prefill
interference. Combined with §4.3.8(a)'s M4 finding (ITL tail = monolithic
prefill, size monotone in `108 - D`), `g` was expected in advance to be
**"prefill-SM elasticity wearing a decode-SM-elasticity label"** — and the
measurement matches: the effect reproduces in the UNSPLIT-only population.

**This is not fixable by adding blocks.** Record:

- 872077's NO VERDICT status **stands**, reason widened as above.
- `g = A_free(d16)/A_free(d54)` **retires on this grid** — do not cite until
  the sticky-partition substrate fix lands.
- Expanding blocks 8 -> 12-16 on the current grid is **pre-emptively
  disallowed**. For reference, assuming current mean/sd hold, P(Ha8 upper
  bound <= 1.15) was n=12 43% / 16 55% / 24 75% / 32 87% / 40 93% — but sd
  will change post-fix, so this table is **void in advance**.
- "Ha8 has no decode-SM lever" is **not CONFIRMED** — the current data cannot
  answer that question. **Tension A (HE2 vs C2) is not closed at all.**

#### (c) `A_free`'s own defects (applies across the E1 harness, not just M3)

Reading `e1_m3_control.sbatch:281-306`:

1. **The blocking filter does not work.** `PREFILL_BLOCK_TOK = 1024`, but only
   4-5% of this workload's requests have input >= 1024 tokens, and their share
   of total prefill work is only 23-26% — so **74-77% of prefill work passes
   the filter unblocked** (only ~2% of all ITL entries are actually removed).
   §4.3.8(a)'s monolithic-prefill stall finding therefore **contaminates
   d16-d54, not just d92** (widening §4.3.8(a)'s original "d92 only"
   limitation) — the requests that make up T8 d16's tail have inputs of
   221-804 tokens, all below the threshold.
2. **Double extreme-percentile statistics.** 27.5-29.5% of requests have
   output <= 25 tokens, so their per-request p95 degenerates to essentially
   the max ITL. The linear-mixture identity behind the dilution model does not
   hold over extreme quantiles (demonstrated by the non-monotone response Ha8
   d54 107.45 -> 101.89 -> 102.84 -> 97.75 under (a)'s de-engagement sweep).

`A_free` is therefore **not a blocking-removed statistic** — it is mostly an
extreme-tail statistic made of monolithic-prefill stall. Estimator replacement
is a follow-up roadmap item, not attempted here.

#### (d) An uncontrolled confound between arms

At the common rate 2: T8 concurrency 12.8 / decode batch 4.5 / ITL p50 ~11ms
vs Ha8 concurrency 30.6 / batch 15.8 / ~30ms. Both arms sit on an off-cliff
plateau (0.88-1.07x plateau ratio, not a metric cliff), but decode batch size
is a covariate that determines whether a decode step is memory- or
compute-bound, and it is fully confounded with arm. **"Attributable to the
arm" is not licensed on the current data** — the correct control is a rate
chosen to match realized concurrency/decode batch, not arrival rate. This
confound does not explain the observed *direction* (the larger-batch arm is
the *less* responsive one), so it is recorded as an **unremoved confound**,
not an alternative explanation. Side note: d16 has `sm_group_num: 3`, d54 has
4 (a guard row), so the number of green contexts differs by cell, and d54's
`decode_sms == 44` is never observed in telemetry (the guard row is never
selected) — not a behavioral confound, but a cell-to-cell difference worth
recording.

#### (e) [UNAUDITED] result-analyst's decode-empty diagnosis

Script: `m3_decode_empty.py` (re-runnable). Not claims-auditor reviewed, but
partially convergent with (b) independently, so recorded:

1. Dilution's cause is **prefill absence, not decode-empty**.
   `E1_DECODE_REALIZED` is conditioned on decode-active time, so decode-empty
   never enters numerator or denominator by construction. Measured
   contribution T8 -0.0006+-0.0046 / Ha8 -0.0001+-0.0028 = 0; the
   block-paired (d54 - d16) decomposition attributes 100% of the gradient to
   the prefill-occupancy gradient (residual CI includes 0: T8
   -0.0074+-0.0111, Ha8 -0.0002+-0.0198).
2. Decode-empty within the loaded window is only **0.6-1.3%**; the raw 15-19%
   is a **measurement-window artifact** (client warmup-to-dataset-prep gap of
   14.9-17.7s plus a 5.2-6.2s post-drain tail). Telemetry analysis should
   anchor the loaded window on the client-reported `duration`.
3. Gate #6 field audit — **four dead telemetry fields**, with code citations:
   `decode_ready_queue_depth` is identically 0 (only populated inside the
   dual-worker guard, and 872077 runs `architecture == "legacy"`);
   `active_decode_sequences` is identical to `decode_running_batch_size` (same
   `running_batch.batch_size()`); `decode_idle_ratio`/`prefill_idle_ratio` are
   identically 0.0 (declared but never assigned in `controller.py:55`);
   `running_batch_occupancy` is identical to `min(1, drb/48)`;
   `prefill_admission_blocked` is identical to `(pqd > 0 and pab == 0)`.
4. Aggregation-unit (gate #5) re-confirmed: decode-empty time-share
   0.0056-0.0127 vs count-share 0.333-0.424 (30-60x apart); the decomposition
   ratio (prefill dominates) is robust across all three units.
5. **Realized ceilings are a workload property and differ 3x by arm**:
   sum(TTFT)/decode-busy-time gives T8 0.120-0.154 vs Ha8 0.381-0.558 — a
   client-side quantity, immune to telemetry instrumentation issues, and an
   additional axis of arm comparison confound beside (d)'s decode batch size.
6. **UNDETERMINED**: the *magnitude* of the realization gradient. Snapshot dt
   p95 is 195-805ms, the same order as prefill span itself, so "prefill
   episode length" reconstructed from snapshot gaps is a sampling artifact,
   **not citable as a physical quantity**. Also unresolved: a `t_pa`-based
   per-request prefill-span increase (T8 d16 -> d54, +30.6ms) versus the
   client-measured TTFT increase (mean +7.5ms, median +10.6ms) disagree —
   resolution requires emitting prefill batch start/end events directly and
   re-measuring.
7. ★**This one item converges independently with the auditor and may be cited
   as [AUDITED]**: raising load to increase engagement is **not** supported by
   the data (decode is already ~99% busy under load; raising rate scales
   prefill and decode proportionally, and even the cross-cell difference in
   `w` traces to prefill slowing down at 108-D rather than to arrival rate;
   split-eligible iteration counts are nearly cell-invariant at 194-214).

#### (f) Next gate (planned, not yet run)

Auditor-recommended order: 1-3 = record this section in canon (done, zero
GPU), 4 = implement `PDMUX_STICKY_PARTITION` + correctness gate
(engine-porter, ~0.25h), 5 = one sticky-grid run (872077's design, 8 blocks,
~3.0h; 872077 itself serves as the non-sticky control, ~3.3 GPU-hour total),
6 = block expansion only **after** 4 and 5.

**Pre-registered discriminating prediction (the point of the sticky run):**

- If the tail is prefill-driven (per §4.3.8(a)), the unsplit population
  disappears under sticky partitioning and `g` **falls** to its
  split-conditional value: **T8 ~= 1.85, Ha8 ~= 0.92**.
- If argument (a) above had been right, Ha8's `g` would instead **rise**
  toward **~1.6**.
- The two predictions have non-overlapping CIs (Ha8 0.92 [0.84, 1.00] vs
  1.66), so **8 blocks discriminate**.

Pre-registered gate: `E1_DECODE_REALIZED >= 0.90` (every cell-block) — on the
sticky substrate this is **no longer an identity**, so it becomes a real gate
for the first time. Confirmed not achievable by config alone
(`initialize_stream_groups` unconditionally appends the final unsplit group)
— an engine change is required. engine-porter is implementing this separately;
**implementation completion is not the same as a performance claim**.

### 4.3.10 ★★2026-08-03, same day, second continuation session (doc-steward
    recording claims-auditor's estimand handoff + engine-porter's
    `PDMUX_STICKY_PARTITION`) — a replacement estimator for `A_free`,
    implementation of a genuinely conditional decode partition, and forward
    pre-registration for the first sticky-substrate run. Source discipline:
    (I)(1)-(2) below are **[AUDITED]** (claims-auditor's own re-analysis,
    bit-identical reproduction). **(I)(3) is an exception and is
    [UNAUDITED]** — it is new output the auditor produced *this turn*, so it
    is the auditor auditing itself; do not cite until an independent pass
    confirms it. (II) is an **implementation fact** (correctness-gate class),
    not a performance claim. (III) is a pre-registration, not a result.

#### (I) A conditional per-token estimator to replace `A_free`

Code: `results/s8_frontier/m3_conditional.py` (new, untracked). Every number
below reproduces bit-identically from that one file. Run:
`python3 m3_conditional.py --job 872077 --cache <path>.pkl`; `--only
{align,engagement,conditional,loo,tail,overlap,deengage,thresh}` selects a
section; labeling all 64 probes takes ~4 minutes.

**(1) Definition.** Unit = one ITL interval (one emitted token); no
per-request aggregation. Inclusion: warmup excluded (`i >=
ceil(WARMUP_S*rate)`, index 6, `e1_m3_control.sbatch:265`), `errors[i]==""`.
Label: for interval `[a,b]` (client clock, `a = arr[i]+ttft[i]+cumulative
ITL`), `split_frac(a,b) = (time within [a,b] where decode_sms==D)/(b-a)`
computed from the telemetry step function. **SPLIT := split_frac >= 0.90 /
UNSPLIT := <= 0.10 / AMBIGUOUS := in between and excluded from both** (never
force-classify). Observed AMBIGUOUS is 0.3-1.4%. Per-block-average label
yield: Ha8 d16 SPLIT 3,738 / UNSPLIT 38,657 (amb 0.008); Ha8 d54 7,045/34,287
(0.014); T8 d16 1,390/36,470 (0.003); T8 d54 3,115/34,239 (0.008). Statistic:
per-cell-block **direct per-token quantiles** — primary `p95(SPLIT)`,
secondary `p50(SPLIT)`, **control `p95/p50(UNSPLIT)`**, mean reported
alongside. The UNSPLIT control population is the key safeguard: both cells
sit at 108 SM there, so the decode-SM contrast is zero by construction — if
the ratio there is not 1, the difference was not made by decode SM. Contrast
`r(arm,stat,block) = stat(d16,block)/stat(d54,block)`, paired within block,
8-block block-clustered t-interval (`TCRIT` from `m3_analyze.py:60`, keyed on
**n** not n-1, `TCRIT[8]=2.365`; percentile bootstrap is disallowed per its
known 79.8% coverage at n=8). Motivating measurement: T8 d16 pooled
per-token p95 = **11.60** vs `A_free` = **28.31** (more than 2x further into
the tail); 27.5-29.5% of requests have outlen<=25; `A_free`'s outer p95 rests
on ~9.6 requests, the new estimator on 1,390-7,045 tokens per cell-block.
⚠️This estimator is better-defined than `A_free`, nothing more — it is
**not the pre-registered SLO term**. `A_all` (the registered term) is still
reported alongside per §1-24's corollary.

**(2) Client<->telemetry clock alignment and ALIGN-WEAK handling.** Procedure
(`m3_conditional.py:align`): anchor on the first `phase_marker(phase ==
"benchmark")`, replay `replay_arrivals(seed, rate, n)` bit-identically to
`e1_m3_control.sbatch:284-287` (`np.random.seed` -> `exponential(1/rate)` ->
`cumsum`), build the client in-flight step function, maximize Pearson r
against telemetry `decode_running_batch_size` (coarse `L in [-5,60) step
0.5s`, then fine `+-0.5 step 0.02s`). The lag is large (13.9-35.9s) because
the `benchmark` marker fires on the server's first-seen request — the *warm-up*
request in `bench_serving` (`multiplexing_mixin.py:386-398`) — and dataset
tokenization then further delays the actual probe start. **The marker is a
search anchor, not the probe boundary**, so `load_telemetry` must **not**
filter on `phase == "benchmark"`. Self-correction: the prior report's "one
weakly-aligned probe (T8 d16 b1, r=0.882)" was a 24-probe spot check; the full
64-probe scan finds **two** (adding T8 d16 b6, r=0.930). **All reported
numbers above were already computed over the full 8 blocks, so nothing
changes** — only the description of that count changes. Rule:
`ALIGN_R_MIN=0.95`; below it, flag ALIGN-WEAK and report both the
included and excluded version — never drop silently. Reasons: (a) `r` is
paired within block, so excluding one probe drops the whole block
(`n_indep` 8 -> 7, re-losing exactly what §4.3.8(h) recovered); (b)
misalignment randomizes labels, pulling both sub-populations toward pooled —
it can only weaken separation, not manufacture it (conservative bias); (c)
measured LOO (`sp_p95` ratio): T8 full 1.688 -> drop b1 **1.822**, drop b6
1.640 — dropping the weakest block makes the contrast *larger*, not smaller.
Ha8 full 1.340, LOO range 1.219-1.375 (every Ha8 probe has r >= 0.993).

**(3) [UNAUDITED — the auditor's own new output this turn, self-graded;
requires independent confirmation before canon citation] Blocking-threshold
sweep.** `PREFILL_BLOCK_TOK` 1024 -> 512 -> 256 -> 0
(`keep_frac / pooled-p95 / A_free-form`, ms):

| arm/cell | 1024 (as-run) | 512 | 256 | 0 |
|---|---|---|---|---|
| Ha8 d16 | 0.979/98.66/**113.51** | 0.934/75.41/111.35 | 0.901/34.25/104.38 | 0.827/33.50/**34.63** |
| Ha8 d54 | 0.960/88.01/**107.45** | 0.882/66.46/89.08 | 0.832/34.63/87.42 | 0.756/33.01/**35.11** |
| T8 d16 | 0.994/11.60/**28.31** | 0.979/11.54/21.66 | 0.963/11.49/13.31 | 0.928/11.44/**11.65** |
| T8 d54 | 0.984/14.01/**15.38** | 0.955/12.40/15.11 | 0.926/11.53/14.52 | 0.877/11.38/**11.59** |

There is **no knee**, and at threshold 0 the d16-vs-d54 contrast **collapses
in both arms** (Ha8 0.986, T8 1.005). The threshold is therefore not a free
tuning parameter — it is **the handle that sets the answer** — and post-hoc
selection is disallowed. Threshold 0 redefines the estimand as "ITL during a
moment with no prefill in flight at all" and that selection correlates with
load, i.e. a selection bias, so it is not adoptable as a replacement
threshold. ★This table uses no telemetry and no clock alignment, so it is
immune to the instrumentation objections against the conditional analysis
above — that much of it is robust regardless of audit status.

**Recommendation arising from (1)-(3):** retire `A_free` (not re-tune its
threshold); keep `A_all` alongside. **Primary label = realized partition**
(`decode_sms == D`). Secondary label = prefill overlap
(`prefill_overlap_frac(a,b) = fraction of [a,b] where prefill_active_batch_size
> 0`, BLOCK-FREE := <= 0.0), but its **instrumentation bias must be stated**:
872077 ran with `PDMUX_TRACE_FORCE_PREFILL=0`, so prefill-active is severely
under-sampled (documented case: 4.9% of wall time vs 0.07% of snapshots,
`multiplexing_mixin.py:400-408`). Measured contradiction: the fraction of
SPLIT-labeled tokens that come out "overlap-free" is Ha8 d16/d54 66.3%/38.8%,
T8 76.1%/27.1% — physically impossible if SPLIT implies prefill in flight, so
that gap is exactly the instrumentation shortfall. `decode_sms`, by contrast,
is a **persisted state variable** read on every snapshot even during dense
idle-spin sampling, hence robust — which is why primary uses `decode_sms`.

**(4) Three places a re-implementation would diverge (also pinned in code
comments; keep here too):**

1. `replay_arrivals` is bit-identical to the sbatch (`np.random.seed` +
   `exponential` call order).
2. `load_telemetry` applies **no** `phase == "benchmark"` filter.
3. `report_deengagement` allocates one `rng` for the whole nested loop, so it
   is **loop-order dependent**. Reproducing the reported values (113.51 ->
   112.29, etc.) requires fixing that order. If this is ever absorbed into
   production code, switching to a `(arm,cell,block,f)`-keyed seed is the
   right fix, but **the specific value will move by MC noise** when it does.

### 4.3.11 [Implementation fact, not a performance claim] `PDMUX_STICKY_PARTITION`
    implemented and correctness-gated (engine-porter, 2026-08-03)

All changes in `src/multiplex/multiplexing_mixin.py` (+151/-17, 4 hunks; line
numbers below are post-patch).

**All five fallback paths were audited:**

1. `adjust_stream_groups`:904 `elif not running_batch.is_empty():
   set_current_stream_idx(real_sm_group_num-1)` — the **core fallback** to
   plain `(0,108)` whenever decode is busy but prefill is absent. Guarded to
   `if not running_batch.is_empty() and (split_prefill_batch or
   sticky_partition_enabled)` so sticky never reaches this fallback while
   decode is busy.
2. :906 `else: set_current_stream_idx(0)` (decode **empty** -> plain
   `(108,0)`) — **deliberately left unchanged** (see rationale below).
3. `event_loop_pdmux`:1010-1012's trigger
   (`stream_idx>0 and running_batch.is_empty()`) — unchanged, commented to
   stay consistent with #2.
4. `event_loop_pdmux_coord`:1309-1311 + `set_current_stream_idx(HEAVY)` at
   :1437/1475/1524/1557/1602 + :1327 — the **combination is rejected at
   init** with a `RuntimeError` (no half-sticky state).
5. The SLO branch :872-880 and v7 `_tgt` :1027-1063 — unchanged (only
   reachable during a prefill span; sticky only fills the gaps between
   spans).

**Decode-empty releases to index 0 (does not hold).** Three reasons: (a)
`E1_DECODE_REALIZED` is decode-active-time-weighted, so this interval carries
zero weight and holding could not improve the gate it exists to serve; (b)
there is no decode work to protect, so holding would only strand D SM from
prefill with no offsetting benefit; (c) it keeps fallback path #3 valid,
keeping the two halves of the mechanism consistent. Smoke measurement:
decode-empty snapshots land at `(0,108,0)` in both arms (OFF 10,812 / ON
12,046) — **the two arms differ only in the decode-busy population.**

**OFF is byte-identical to pre-patch.** The only change is
`split_prefill_batch or sticky_partition_enabled`; with the flag OFF the
right-hand disjunct is `False`, so short-circuit evaluation makes the
predicate identical to before the patch. The original selector block moved
inside an `else:` unchanged (byte-for-byte, including a latent
`UnboundLocalError` corner case — deliberately not refactored into a shared
helper). `test_off_matches_pre_patch_selector` asserts equivalence against an
**independently re-implemented pre-patch selector** across the full grid of 3
configs x decode_bs{0,1,4,47,48,96} x {prefill present, prefill absent}.
Telemetry code is unchanged (no new events/fields/emission-cadence changes).

**cudagraph preserved.** `cuda_graph_runner.py` keys captures by
`f"{stream_idx}_{bs}"` and `capture()` captures **every** stream-group index,
so pinning the division index still replays a captured graph — no eager
fallback, the operating point is preserved.

**Mode interactions — sticky ON rejects at init with `RuntimeError` against:**
`PDMUX_LA_COORD`, `PDMUX_SLO_SCHED`, `PDMUX_FIXED_DECODE_SM_FILE`,
`PDMUX_R2_POLICY` not in `{unset, fixed}`, `real_sm_group_num < 3`. Allowed:
`PDMUX_R2_POLICY=fixed` (the target index is resolved at init from
`FixedPolicy.decode_sms`, rejected if it only matches a plain group), no
policy, `PDMUX_DUAL_WORKER`, `PDMUX_TRUE_DUAL_WORKER`. Side effect: under
`fixed`, sticky pins the *resolved index*, not the decode-bs selector — this
matters because `pdmux_e1_d54.yml`'s guard-satisfier row `[64,44,48]` would
otherwise be selected once `decode_bs>=48`, landing on a **different
partition than the cell label**
(`test_on_with_fixed_target_ignores_the_guard_satisfier_row`).

**Code confirmation that telemetry records realized, not target.**
`_dual_worker_sync(stream_idx)` -> `observe_scheduler` ->
`arbiter.select_partition(stream_index)` -> `metrics()` returns
`arbiter.sm_counts[arbiter.stream_index]`. The index passed in is the same
`CURRENT_STREAM_IDX`/loop-local variable that selects the CUDA stream and is
reassigned at every call site — so `decode_sms` is not a label, it is **the
SM count of the green context decode actually ran on** (which is why
pre-patch runs showed D108 92-96% of the time instead of the cell label).
⚠️**Unre-verified residual scope (not a claim):** whether
`create_greenctx_stream_by_value(92,16)` causes the hardware to actually
grant 16 SM is a layer this patch does not touch and was not re-probed here.

**Correctness-gate results:** (1) CPU regression, sticky OFF, **PASS** — 40
tests (28 existing + 12 new), `sync_engine_tree.sh` mirror/dev-tree SHA-256
match, recorded in manifest
(`multiplexing_mixin.py =
59eaafb4ac61cc09ad8d28f663c495cf6a0e850435c15b873547b8a6e5a7d20a`), no sync
script changes needed. (2) 12 sticky unit tests **PASS**
(`tests/test_sticky_partition.py`; loads the mixin from the **installed**
runtime, so it also asserts the sync actually ran). (3) thread-local role
patch nominal. (4) **GPU smoke PASS** — job **872800**,
`amd_a100nv_8`/gpu38, ~9 minutes. Ha8 = Zamba2-7B-Instruct, d16
(`PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=16`), server args copied
verbatim from `e1_m3_control.sbatch`; the two boots differ only in
`PDMUX_STICKY_PARTITION`. 6 fixed greedy prompts (`temperature 0`,
`max_new_tokens 48`) issued sequentially -> **all 6 OFF/ON outputs
byte-identical.**

**★Realized observation (an observation, not a gate).** Probe = 100 ShareGPT
prompts, rate 2, seed 1, `PDMUX_TRACE_FORCE_PREFILL=0`, existing
`e1_pin_check.py:compute_decode_realized`.

| arm | `E1_DECODE_REALIZED` | decode-active histogram |
|---|---|---|
| sticky OFF | **0.0839** | D108 66.7s, D16 6.1s |
| sticky ON | **1.0000** | D16 164.3s (no D108) |

OFF reproduces the existing behavior (872077 Ha8 d16 = 0.104-0.127 by the
same estimator). ON is **1.0000, pre-registration was >=0.90, and nothing
was tuned to hit it.** Snapshot cross-check: decode-busy snapshots land at
`(idx1,92,16)` 139/139 for ON, `(idx2,0,108)` 146 / `(idx1,92,16)` 8 for OFF —
ON never enters index 2. ⚠️**Caution with no interpretation attached:**
decode-active wall time differs between arms (72.8s vs 164.3s) because the
arms run decode on different partitions. **n=1, not repeated, not a
performance measurement** — recorded so the realized fraction is not misread
as a like-for-like denominator. ★**Implementation completion is not the same
as a performance claim.** No statement about throughput, latency, goodput,
or `g` is licensed. What this patch claims is only that
**`E1_DECODE_REALIZED` has stopped being an identity and has become a real
gate.**

**New files:** `tests/test_sticky_partition.py`,
`results/sticky_smoke/sticky_smoke.sbatch`,
`results/sticky_smoke/stksmoke_Ha8_d16_872800_result.txt` + smoke artifacts
(two telemetry.jsonl files, ~14MB each; recommend git-untracked per the
`s8_frontier` convention). `pdmux_context.py` is unchanged (`[(108,0)] +
divisions + [(0,108)]` retained — sticky avoids the trailing unsplit group
rather than removing it, preserving OFF reproducibility).

### 4.3.12 Pre-registration for the first sticky run — record now, leave
    `G_LEVER`/`G_FLAT` undetermined

**(a) 872077's retroactive re-analysis under the new estimator is
DIAGNOSTIC-ONLY, not a verdict.** The estimator in §4.3.10 was **selected
after seeing** 872077, so adopting a verdict from that data would be a
confound (re-score vs re-tune) — the same failure mode canon has already
been burned by once, and the same logic that made §4.3.8(c)'s RULE_BOUNDS
forward-only. The retroactive application is usable only as **supporting
material for the design verdict** ("estimand not identified"), never as a
performance reading. `m3_conditional.py` prints this sentence at the end of
every run.

**(b) One fixed primary**: declare `p95(SPLIT tokens)` ratio as primary;
p50/mean/UNSPLIT are secondary. With six statistics available, going in
without a declared primary opens post-hoc selection (multiplicity).

**(c) Four gates** — re-verify, before registering each, that it measures a
quantity **logically independent** of the condition it is meant to pass
(methodology gates #6/#7; do not repeat `MIN_PA_SNAPSHOTS`'s mistake of
counting the estimand's *complement*):

| Gate | Value | Rationale |
|---|---|---|
| `ALIGN_R_MIN` | 0.95, **flag-only** | exclusion costs `n_indep`; misalignment bias is conservative |
| `E1_DECODE_REALIZED` | **>= 0.90** per cell-block | no longer an identity under sticky, so this is finally a real gate (smoke n=1 observed 1.0000) |
| `AMBIG_FRAC` | pre-registered upper bound (872077 observed 0.003-0.014) | if the ambiguous band eats the sample, the estimator silently changes |
| `MIN_N_SPLIT` | per-cell-block lower bound (872077 observed 1,390-7,045) | should rise under sticky; a fall means sticky did not take effect |

**(d) ★`G_LEVER`/`G_FLAT` = UNDETERMINED. Record "undetermined", not a
number.** The existing 1.5/1.15 thresholds are scaled for `A_free`'s
double-extreme statistic; the new estimator is a per-token quantile on a
**different scale** — porting the old numbers over would itself be post-hoc
adjustment. Live argument: because the new estimator sits on the **same
axis** as C2 (per-token ITL quantiles), anchoring `G_LEVER` to C2's measured
range (**2.36-2.91x**, prefill fixed at 16 SM, SM16->SM92) is *in-principle*
defensible — but E1's D range is narrower (16->54) and **complementary**
(prefill = 108-D moves in lockstep), so C2's value cannot be transplanted
as-is. **Record this as an open item requiring its own pre-registration
before the sticky run is submitted**, together with the reasoning above.

**(e) Two things change under sticky — pre-register now**: (i) the UNSPLIT
sub-population shrinks toward empty or very small (that is the point) — do
**not** require the `un_*` row as a mandatory output; report NaN when
undefined, and base the verdict on the SPLIT series. If UNSPLIT stays large,
sticky did not take effect, and `E1_DECODE_REALIZED` catches that first.
(ii) the **primary population shifts to `SPLIT and BLOCK-FREE`** — pre-sticky
the two labels are effectively redundant, post-sticky that is the **only**
population that isolates decode SM from prefill interference. Pre-registering
this shift now is what keeps it from being a post-hoc selection.

**(f) Discriminating prediction (the final test of argument A)** — carried
forward from §4.3.9 unchanged: if the tail is prefill-driven (§1-24) under
sticky, Ha8 -> ~0.92 / T8 -> ~1.85; if argument A (the refuted dilution
model) had been right, Ha8 would instead rise toward ~1.6. The CIs do not
overlap, so **8 blocks discriminate**. Block expansion is out of scope until
**after** this discriminating run.

**(g) Open design issue (left to a human/claims-auditor decision, not
resolved here).** If the post-sticky primary population moves to `SPLIT and
BLOCK-FREE`, §4.7.1's force-trace prohibition becomes a **binding
constraint** (because the prefill-overlap label's instrumentation shortfall,
§4.3.10(1), depends on it). But §4.3.8(h)'s SCHED-only decomposition already
re-attributed force-trace's *systemic* effect as mean +0.001ms (sign 4+/2-),
so **whether to re-permit trace-force ON is a sticky-specific design
question, not settled by that prior finding.** Recorded open, not decided.

### 4.4 Decision rule (pre-registered — do not change without updating this file)

> For each arm, let best-static = the D cell (of the 5 measured) with the
> highest mean rep-wise **combined conjunctive goodput**
> (`(good_LO + good_HI) / (dur_LO + dur_HI)`, duration **summed** across LO
> and HI phases within a rep, gate #7 — never `max()`). A cell D "wins" if
> its mean combined goodput exceeds best-static's by **≥3%** AND the
> **paired bootstrap** 95% CI of `(D − best_static)` (paired by rep index,
> both server-boot-and-trace-position-matched within the SAME arm) excludes
> 0. If no D wins for an arm, **C2 is confirmed as "the ITL lever exists but
> is net-negative under the prefill+decode≤108 budget" for that arm**; if at
> least one D wins, tension A resolves toward net-positive for that arm.
> Implemented verbatim in `e1_analyze.py`'s "DECISION RULE" section.

Per-arm, not pooled-across-arms — a lever that is net-positive for one
architecture and net-negative for another is itself a finding, not noise to
average away.

### 4.5 ★★NEW 2026-07-29 (coordinator directive, after 867008's `arrival_rps`
    diagnosis) — SEED POLICY: rep = workload realization, cell/arm = matched

**Bug found**: `e1_capacity_scan.sbatch`/`e1_sweep.sbatch` never passed
`--seed` to `sglang.bench_serving`, so every probe/rep/cell/arm in the whole
campaign silently used the SAME implicit default (`seed=1`) — bit-identical
arrival sequences AND bit-identical ShareGPT prompt samples (`random.seed`/
`np.random.seed(args.seed)` govern both, `bench_serving.py:1705-1706`) every
single time. The tell: 866868's reconstructed `arrival_rps` at rate=2 was
+72% off nominal — large enough to matter, but (per gate #3's own logic:
`bench_noise_root_cause.md`) NOT obviously anomalous on its own (a ~1.7σ
draw at n=19). **The problem is that this specific draw is not noise that
averages out across reps — it is the SAME fixed draw on every rep**, so any
rep-level statistic (including the paired-bootstrap CI the decision rule,
§4.4, depends on) was blind to workload variation entirely; it could only
ever reflect run-to-run ENGINE noise (clock jitter, GC pauses, scheduling),
never "would a different but equally-plausible arrival pattern / prompt
sample change the answer." A CI that is structurally too narrow can promote
a result that only holds for one specific (and, per the arrival-rps
diagnosis, possibly non-representative) workload realization into "CI
excludes 0" — exactly the failure mode gate #3 exists to prevent, but
gate #3's own historical justification (`CONSENSUS.md` §1-14: "4 runs, same
workload fingerprint, so noise isn't a workload artifact") is being
**inverted** here: THAT finding was used to rule OUT workload variation as
an explanation for noisy goodput; HERE, because the comparison itself is a
goodput-based SLO indicator function (sensitive to exactly where the
metric-cliff boundary falls, gate #6), an accidentally-fixed single workload
realization is a live risk to the DECISION RULE's own validity, not
reassurance about it.

**Pre-registered fix (implemented in both sbatch scripts, `e1_sweep.sbatch`
§ "SEED POLICY" comment, `e1_capacity_scan.sbatch` `$SEEDS`):**

> **Seed varies BY REP, is FIXED across cells and arms for the same rep
> index.** `e1_sweep.sbatch`: `--seed $((BASE_SEED + rep))` for BOTH the LO
> and HI phase of a rep (`BASE_SEED` default 100). Rep 1 of D16/T8 uses the
> SAME seed as rep 1 of D92/T8 and rep 1 of D16/Hs8 — every cell/arm in the
> campaign sees the identical SET of 4 (or `REPS`) workload realizations,
> just possibly in a different D-cell context — this is what keeps
> cross-cell comparisons workload-MATCHED (paired), the precondition the
> paired bootstrap in §4.4 already assumed but the implementation did not
> actually guarantee before this fix (it accidentally satisfied the pairing
> condition in the DEGENERATE way of using the SAME single realization for
> everything, rather than the intended way of using matched-but-varying
> realizations). `e1_capacity_scan.sbatch`: every rate is probed at
> `SEEDS` (default `"1 2"`, ≥2 required) — §4.2.1 already established
> single-realization elbow-siting is fragile; this makes that check
> mechanical (`CAPSCAN_SEED_DIVERGENCE`, see §4.2.1/item 4 below).

**Consequence for CI interpretation (item 3 — must be read alongside ANY
cited CI from this campaign, present or past)**: post-fix, a rep-wise
paired-bootstrap CI reflects BOTH engine noise AND workload realization
variation. This is the scientifically correct scope for the CI (it is
supposed to answer "would this conclusion hold under a different but
equally valid run", and workload realization is one of the things that
legitimately varies between runs) — but it also means **narrow CIs
observed elsewhere in this project's SIBLING campaigns that did NOT vary
seed across reps should NOT be read as evidence of high confidence**. In
particular: `s8p_prefill`'s reported rep-to-rep spread (sd 0.01–0.9% per
the coordinator's own framing) was very likely measured this same way
(closed-loop `s0dc_client.py`-family clients don't take a `--seed` at all,
a DIFFERENT but related exposure — no seed argument means no run-to-run
workload variation is possible by construction, an even more direct version
of this same category of confound). This DESIGN.md does not audit that
campaign's files (out of scope, avoid duplicating result-analyst's work,
per the 866868 message's own instruction) — flagged here only so E1's own
citations are not read as "E1 is less reliable than its narrower-CI
siblings"; the narrower CIs are more likely UNDER-covering their true
uncertainty, not more precise.

**Validation (item 2, CPU-only, no GPU/server needed)**: called
`sglang.bench_serving.get_request()` directly (dummy inputs, no real
requests) and measured ACTUAL `asyncio`-scheduled wall-clock dispatch times
against this harness's manual RNG-replay reconstruction, for `seed ∈ {1, 2,
7}` × `rate ∈ {2, 8}` (6 combinations): max absolute error 2–70ms out of
several seconds of total span (consistent with ordinary event-loop
scheduling jitter, not a reconstruction error) — confirms the replay
mechanism (§4.2.1) generalizes correctly to non-default seeds, not just the
one value (`seed=1`) every prior probe happened to use. This closes the
"does the replay still work" question the coordinator raised as a
precondition for trusting `arrival_rps` post-fix.

**Item 5 — rate labeling**: any report/plot keyed on "rate" from this
campaign (capacity scan or main sweep) should use the RECONSTRUCTED
`arrival_rps`, not the nominal `--request-rate` value, as the x-axis/label
of record. Nominal rate is kept as a human-readable reference only (what
was REQUESTED, not what was REALIZED) — `e1_capacity_scan.sbatch`'s
`CAPSCAN` line now prints `arrival_rps` first, `nominal_rate` second, per
this policy (§4.2.1).

### 4.6 ★★NEW 2026-07-29 (coordinator directive) — low-rate arrival deviation
    is x-axis scatter, not error; two-stage scan; per-rep elbow-margin
    enforcement

**Reframing (coordinator, superseding §4.2.1's earlier framing of the
+72%/+12% deviation as primarily a "small-n noise" concern to fix with
bigger `NP`)**: `arrival_rps` is a MEASURED quantity — it deviating from
nominal is x-axis scatter, not a measurement error, and inflating `NP`
further (already tried once, §4.2.3) is not the right response to scatter.
Two things make this non-threatening for the MAIN sweep specifically: (i)
§4.5's seed policy means every cell/arm shares the SAME set of workload
realizations for a given rep index, so scatter in the realized arrival
rate is common to all cells being paired-compared in that rep and cancels
in the paired-bootstrap decision rule (§4.4); (ii) the only thing that
actually threatens validity is **picking an off-cliff rate that turns out
not to be off-cliff for some rep's specific realization** — a single,
narrower risk than "arrival scatter" broadly.

**Two-stage capacity scan** (mirrors vector-1's `g2_0_rasweep` precedent —
coarse locate, then densify around the transition, not a single flat
sweep): Stage 1 uses the CURRENT `RATES` ladder (broad, sparse) to locate
where the TTFT/ITL percentile trend visibly bends (or where errors start
appearing) — per §4.2.1, throughput numbers are not the elbow signal,
latency trend is. Stage 2 re-scans ONLY the neighborhood of that
transition at finer rate granularity, the same "coarse-then-dense" pattern
`g2_0_rasweep`/`g2_0_raconf` used to narrow vector-1's transition band from
a broad sweep down to rate 3.0-3.5. §7's time budget already assumed a
single flat scan; §11 (time-budget note) is updated to reflect the 2-stage
cost.

**Per-rep elbow-margin enforcement (pre-registered NOW, before any main-sweep
data exists)**: once a cell/arm's elbow-onset rate is identified from the
2-stage scan, `e1_analyze.py --elbow-onset-rate-lo`/`--elbow-onset-rate-hi`
reconstructs EACH rep's own REALIZED `arrival_rps` (same seeded-RNG replay,
§4.5/§5.3 — a rep's specific seed can, by chance, realize a rate closer to
the elbow than the nominal `RATE_LO`/`RATE_HI` choice would suggest, exactly
the kind of scatter this section is about) and EXCLUDES (reports as-run,
never silently averages in) any rep whose realized rate is not ≥15%
(`ELBOW_MARGIN`) below the onset — this is gate #6 enforced at the
individual-rep level, not just at the one-time `RATE_LO`/`RATE_HI`
selection step. Checked BEFORE the goodput/decision-rule computation, so an
excluded rep never reaches `by_cell` even if it also passed the pin gate.

### 4.7 ★★NEW 2026-07-29 — `PDMUX_TRACE_FORCE_PREFILL` pre-registration
    (engine-porter's patch): uniform across all 5 cells, gated on the
    observer-effect check, asymmetric perturbation direction CORRECTED

**What it is**: engine-porter's patch forces an extra telemetry snapshot
whenever prefill is in flight — directly targeting bug #4/#5's root cause
(ordinary event-loop-iteration subsampling almost never lands a sample
inside a fast cell's brief prefill window, §5.3/§5.4). Default **OFF**,
byte-identical output when off (engine-porter's own CPU regression, 28
tests PASS), manifest `b0c92f2a…`.

**Pre-registered rule: identical across all 5 cells, no partial
application.** `e1_sweep.sbatch` exports `PDMUX_TRACE_FORCE_PREFILL`
globally (once, before the arm/cell loop, `E1_TRACE_FORCE_PREFILL` env
var defaulting to `0`) so every cell in a given sweep invocation gets the
SAME setting — turning it on for only some cells (e.g. only the ones that
need the extra samples) would confound the D-axis comparison with a
measurement-DENSITY difference between cells, on top of whatever genuine
partition difference exists. **Not yet enabled for any real run**: the
default is `0`, and per the coordinator's explicit instruction, this flag
must not be set to `1` for a real `e1_sweep.sbatch` submission until
engine-porter's observer-effect gate (job `867298`, a paired OFF/ON
overhead comparison, `results/e1_traceforce/tfgate_analyze.py`) has passed.

**★Corrected 2026-07-29 — the perturbation asymmetry runs OPPOSITE to
this agent's own earlier expectation, and is NOT a constant across
cells.** This agent's prior framing implicitly assumed a fast-prefill cell
(D16) would see the LARGEST relative increase in emitted telemetry volume
(since it needs the most help). **Corrected direction (coordinator,
engine-porter's own measurement)**: forced-emission VOLUME is proportional
to the number of scheduler-loop SYNC CALLS while prefill is in flight, and
a SLOWER prefill (D92, 16 SM) occupies far MORE such calls per request than
a fast one (D16, 92 SM) — so the perturbation is LARGEST at the
slow-prefill end of the grid, not the fast end. Measured/projected: **D16
+1.2% (~2 emissions/s) · D44 +2.2% · D92 +28.5% (~44/s)**; on a
scheduler-wall-clock basis this is smaller in absolute terms (D16 ≲0.02%,
D92 ≲0.3%, median 33-45µs per emission) but **the asymmetry itself is the
point** — it means the observer effect is NOT a constant across the D-grid,
it runs ALONG the frontier and is LARGEST at exactly the cell (D92) that
already has the most citeable evidence and SMALLEST at the cell (D16) that
most needs the help. This must be read alongside any cross-cell comparison
once the flag is on: a D92-vs-D16 result cannot assume the measurement
apparatus perturbed both cells equally, even though the perturbation is
small in absolute (wall-clock) terms at both ends.

### 4.7.1 ★★★SUPERSEDES the §4.7 application rule (2026-07-31, after job
    867298 landed) — MEASUREMENT AND VERIFICATION ARE SEPARATED; the main
    sweep runs `PDMUX_TRACE_FORCE_PREFILL=0`

**Gate outcome (job `867298`, ABBA boot order OFF/ON/ON/OFF, n=4 per
condition, paired, `tfgate_T8_867298_result.txt` §`=== PAIRED SUMMARY ===`):
CONDITIONAL PASS — an asymmetric perturbation was detected and it lands on
the decision metric.**

| cell | telemetry volume | metrics whose t95 CI excludes 0 |
|---|---|---|
| d16 = `[92,16]` | +9.3% | **none** (all metrics) |
| d92 = `[16,92]` | +41.7% | **`itl_p95` +2.00%** [+0.78, +3.21] · `itl_p99` −1.28% · `itl_mean` +0.56% |

TTFT includes 0 at both cells; `output_throughput` +0.02%.

**The problem is not the magnitude, it is the coincidence of location.** The
one metric that moves significantly is `itl_p95` — the exact quantity the
§4.4 decision rule thresholds against the pre-registered 60 ms ITL SLO — and
it moves at **only one end of the D-grid** (d92, the decode-heavy end),
in the direction predicted by §4.7's corrected asymmetry. A +2.0% shift in
`itl_p95` is small against the ≥3% conjunctive-goodput decision margin, but
it is not small against a conjunctive goodput that is a **threshold
indicator**: near the SLO boundary a 2% shift in the percentile moves
requests across the pass/fail line, which is precisely the metric-cliff
failure mode this project has already been burned by (`CLAUDE.md` gate #6,
`reports/bench_noise_root_cause.md`). The p95 (+2.00%) / p99 (−1.28%) sign
disagreement further suggests a mix of a real micro-effect and
multiple-comparison noise (3 of 18 tests significant), so the safe reading is
"an effect exists at d92 and its size is not reliably estimated."

**Pre-registered resolution — split the two jobs the flag was serving.** A
pin check answers *"does this configuration actually place prefill on the
target partition?"* That is **a property of the configuration, not of a
particular performance run**. It therefore does not have to be measured
inside the run whose numbers decide the campaign.

1. **Main sweep (`e1_sweep.sbatch`): `E1_TRACE_FORCE_PREFILL=0`, unchanged
   default.** No forced emission touches any number that feeds §4.4. This
   supersedes §4.7's "identical across all 5 cells" rule only in the sense
   that the uniform value is now pinned to OFF; the prohibition on *partial*
   application (some cells on, some off) stands unchanged and is in fact
   strengthened, since the asymmetry is now measured rather than projected.
2. **Pin verification: separate short per-cell runs with
   `PDMUX_TRACE_FORCE_PREFILL=1`**, at the **same arm, cell, rate, and seed
   policy** as the main sweep. Same rate matters: §6/`CONSENSUS.md` §1-22
   established that the realized target/auto-revert mixture is itself
   rate-dependent, so a pin run at a different rate does not verify the
   sweep's realized allocation. These runs produce the ≥0.80 time-weighted
   `pin_frac` evidence for the §5 gate; they contribute **no** latency or
   goodput numbers.
3. **The §5 pin gate is evaluated on the verification runs, and the sweep's
   own OFF-mode telemetry is reported alongside as a consistency check** (it
   is the same population, just sparsely sampled — §5.5's `trace_forced !=
   true` filter makes the two directly comparable). A disagreement between
   them is itself a finding and blocks citation.

**Consequence accepted**: the d16 statistical-power problem that motivated
the patch (§9.8 — `n_episodes = 8`, `lower95 = 0.554`) is now solved in the
verification runs rather than in the sweep, so the sweep's own d16 telemetry
will remain sparse. This is the correct trade: sparse pin evidence in the
sweep is a *power* deficiency that the verification run repairs, whereas a
perturbed `itl_p95` in the sweep would be a *bias* on the decision metric,
which nothing downstream can repair.

⚠️ **Job 867298 produced no pin data.** Every `PIN_CHECK` invocation in that
job crashed: `traceforce_gate.sbatch`'s call site carried the pre-bug-#6
positional order (`… "$DSM" "$JOUT" "$SEED" "$RATE" 0.80`) while
`e1_pin_check.py`'s `main()` reads `min_lower95` as `argv[2]`, so every call
died in `float()` on a path string. The paired OFF/ON summary above is
unaffected (it is computed by `tfgate_analyze.py` from the probe records, not
by `e1_pin_check.py`). **Fixed 2026-07-31** — the call site now mirrors
`e1_sweep.sbatch:258`. The gate verdict above therefore rests on the paired
latency/volume comparison only; it does **not** include a pin check, and does
not need one, since the pin question is now handled by item 2 above.

## 5. Pre-registered gates (telemetry-based, auto-judged, cite-blocking) —
   ★★★RE-REVISED 2026-07-30 (coordinator directive, a SIXTH distinct bug
   in this same gate lineage, full investigation in §5.5 — supersedes the
   "PRIMARY = episode gate" framing below, which was itself only just
   fixed in §5.4; history in §5.1/§5.2/§5.3/§5.4)

**★ 2026-07-30 update, read this before the rest of §5**: the PRIMARY,
cite-blocking gate is now `compute_time_weighted_pin_gate`/
`time_weighted_gate_verdict` (episode-cluster-bootstrap 95% lower bound of
the TIME-WEIGHTED pin fraction ≥ 0.80, over ALL prefill-active snapshots,
not just the ones inside a request bracket) — see §5.5 for the full
diagnosis. The episode-based gate described in points 1-2 immediately below
(`compute_episode_gate`/`episode_gate_verdict`, §5.4's fix) is **RETAINED,
UNCHANGED, but RESCOPED to a non-gating, per-request LATENCY-ATTRIBUTION
diagnostic only** — it is no longer what a cell's citeability is decided
on. `compute_concurrency_diagnostic` (the `concurrent_time_frac` mandatory
diagnostic) is unaffected by this revision, it was already time-weighted
(§5.2) and answers a different question again (§5.3 point 4).

Implemented in `e1_pin_check.py` (`compute_time_weighted_pin_gate`/
`time_weighted_gate_verdict` for the PRIMARY gate, `compute_episode_gate`/
`episode_gate_verdict` for the non-gating latency-attribution diagnostic,
`compute_concurrency_diagnostic` for the mandatory concurrency diagnostic —
all imported directly by `e1_analyze.py` so the sbatch's own stdout gate
lines and the analyzer's citeability filter are **the same code path**, not
two re-implementations that could silently drift):

1. **(SUPERSEDED as of §5.5 — kept for the record of §5.4's fix, no
   longer what gates citeability) EPISODE-based realized-partition pin,
   Clopper-Pearson 95% lower bound ≥ 0.80.** §5.1's fix (condition on
   `prefill_active_batch_size > 0` SNAPSHOTS) was itself found wrong at the
   unit level (§5.3): `--chunked-prefill-size -1` means one request's whole
   prefill is ONE event-loop iteration, so it produces at most one telemetry
   snapshot before subsampling — counting snapshots undercounts prefill
   EPISODES regardless of their duration (unlike §5.2's bug, this is not
   about duration-weighting, it is about counting the wrong OBJECT).
   Replaced with EPISODE accounting, **ported from
   `s8p_prefill/s8p_analyze.py`'s interval-bracketing technique** (per
   coordinator instruction, not a new technique): reconstruct each
   request's `[admission, first-token]` interval via the same seeded-RNG
   arrival replay `e1_capacity_scan.sbatch` uses for `arrival_rps`
   (validated for any seed, §4.5), anchored so the reconstructed LAST
   request's completion time lands exactly at the phase's own measured end
   (`T1_LO`/`T1_HI`) — validated against 866066 to ~0.1s out of a ~95s
   window (cross-checked against `bench_serving`'s own reported `duration`
   field). Bisect each interval against the telemetry timeline; ACCEPT only
   if the realized `prefill_sms` immediately before admission equals the
   realized `prefill_sms` immediately after first-token, within a `GUARD`
   (3.0s, matching `s8p_prefill`'s own value) proximity to both endpoints —
   this is the literal ported algorithm. Among accepted episodes, exclude
   **vacuous-idle** brackets (both endpoints show
   `prefill_active_batch_size==0` — internally consistent but zero positive
   evidence the target partition ran for that specific request; an
   empirically NECESSARY refinement found while validating the port, not
   present in the literal s8p_prefill script — see §5.3 for why omitting it
   gives an outright WRONG answer, not merely a conservative one). Among the
   remaining **informative** episodes, `pin_frac` = fraction at the cell's
   target `prefill_sms`. The gate criterion is the **Clopper-Pearson exact
   95% one-sided lower bound of `pin_frac` ≥ 0.80** — this single rule
   replaces BOTH the old `pin_frac >= threshold` check AND the old separate
   `n >= min_n_prefill_active` (magic number 20) floor: a small or noisy
   informative sample now automatically widens the CI and fails the SAME
   lower-bound test a genuinely-wrong-SM sample would fail, so there is no
   longer a distinct `UNDERPOWERED` reason code — a cell with zero
   informative episodes (D16, §5.3) fails with an explicit "no informative
   episodes" message instead.
2. **(also superseded as a gate input, retained as diagnostic)
   `acceptance_rate` (= n_accepted / n_total, matching the literal ported
   check) and `informative_rate` (= n_informative / n_total, after the
   vacuous-idle exclusion) MUST be reported alongside every episode-gate
   attribution** — `s8p_prefill`'s own campaign saw 54-73% acceptance
   depending on L; a low rate here is itself diagnostic, independent of
   whether the accepted/informative episodes hit the target SM.
3. **RETIRED as a hard gate (2026-07-28, unchanged by this revision) —
   "partition activity rate ≥ 0.60."** The original definition conditioned
   on `engaged = prefill_active_batch_size>0 OR decode_running_batch_size>0`,
   which §5.1 found conflates genuine reversion-on-idle with CORRECT,
   BY-DESIGN full-GPU-to-decode behavior. Retired rather than "loosened,"
   replaced by the mandatory diagnostic below.

**MANDATORY DIAGNOSTIC (always reported, NOT gate-blocking on its own — no
principled threshold exists yet, coordinator: "그게 낮으면 D 축 자체가
약해지므로 결과 해석에 필수"): `concurrent_time_frac`** = ★★TIME-WEIGHTED
(2026-07-29 fix — see §5.2, this quantity was itself count-based and badly
biased until this revision, a SEPARATE bug from §5.1's pin-gate fix) share
of the window's wall-clock where prefill AND decode are BOTH simultaneously
active. This answers "how much of the experiment's wall-clock actually
exercised the prefill/decode SM-contention the D-axis is about" — a cell
can pass the pin gate (partition correctly realized whenever prefill
happens to be active) while still having a low `concurrent_time_frac`
(partition rarely mattered because prefill was rarely active at all, e.g.
an open-loop workload at low offered rate). Any citation of a result from a
cell **must** report this number alongside it, per the coordinator's
instruction — see §5.2/§9.7 for the corrected empirical values (866066: 25%
and 40% for d92 LO/HI, NOT the ~1.5% the count-based version had reported).
Also reported for context: `prefill_active_time_frac`/`decode_active_time_frac`
(how much wall-clock time has prefill/decode in flight at all,
time-weighted). This diagnostic is computed by `compute_concurrency_diagnostic()`,
a function SEPARATE from BOTH the (now-primary) time-weighted pin gate
(`compute_time_weighted_pin_gate()`) and the (now-attribution-only)
episode-based check (`compute_episode_gate()`) -- none of the three share
fields or a denominator, by design (§5.3 point 4/§5.5: they answer
genuinely different questions — an occupancy time share, a per-request
event-conditional proportion, and an aggregate time-weighted pin fraction,
respectively).

The PRIMARY (time-weighted) gate is evaluated **per rep, per phase** (LO
window and HI window separately, using the `[t0,t1]` monotonic brackets
`e1_sweep.sbatch` writes to `<RUNID>_rounds.jsonl`) — a rep is only
included in the citeable sample if **both phases pass** it. This is
per-rep filtering, not cell-level averaging, consistent with the "n≥4, no
rep pooling" methodology gate: a bad rep is dropped, not blended into a
cell mean that would hide it. Dropped reps are still written to the
"as-run" section of `e1_analyze.py`'s output for transparency, never
silently discarded.

Diagnostic-only (not gating, but printed and flagged if elevated):
`prefill_admission_blocked` rate + reason histogram, computed over ALL
benchmark-phase snapshots (not the prefill-active population — by
construction a blocked sample has `prefill_active_batch_size==0`,
`dual_worker.py:590-599`, so conditioning on prefill-active would always
read 0% and hide the signal). See §9.2 — the confirmed admission-latch bug
does not apply to this harness's `PDMUX_R2_POLICY=fixed` design, but the
raw `prefill_admission_blocked` telemetry field can still be true for
unrelated, legitimate backpressure reasons and is worth watching,
especially at D92.

### 5.1 Diagnosis of the 865832 smoke gate failure (Hypothesis A CONFIRMED)

**Observation**: smoke job 865832 (T8, cells d16/d92, `SMOKE_CELLS=1
RATE_LO=2 RATE_HI=4 NP=20 REPS=1`) completed end-to-end (`boot_ok=1` both
cells, bench_serving ran cleanly, telemetry non-empty, analyzer ran) but
the (pre-revision) pin gate FAILED for both cells — `E1_ACTIVITY_GATE
D=16: FAIL -> 0.003 < 0.6`, `E1_PIN_GATE D=16(P92): FAIL -> realized pin
0.029 < 0.8`, `E1_REALIZED_hist(engaged n=69): P0/D108:67(97%),
P92/D16:2(3%)` — while the **client-measured TTFT differed 4.5×** between
the two cells (d16 median TTFT 83.60ms vs d92 376.77ms), matching
`s8p_prefill`'s independently-measured prefill-SM sensitivity (3.5–4.8×,
L-dependent) in both size and direction. A partition that was really
unpartitioned 97–99.7% of the time would make the two cells
indistinguishable — they were not, so the partition WAS being applied; the
(pre-revision) gate's population definition was wrong (Hypothesis A), not
the engine reverting under load (Hypothesis B).

**Conditional re-aggregation from the EXISTING 865832 telemetry** (no GPU
needed, per the coordinator's ask), conditioning strictly on
`prefill_active_batch_size > 0` (mirroring `s8p_prefill/prefill_pin_check.py`):

| cell | target (P,D) | n_bench | n_prefill_active | pin_frac (prefill-active-conditioned) | n_both_active | concurrent_frac_of_bench |
|---|---|---|---|---|---|---|
| d16 | (92,16) | 40,806 | **1** | 1.000 (n=1, UNDERPOWERED) | 1 | 0.0000245 |
| d92 | (16,92) | 26,820 | **41** | **0.976** (PASS, n≥20) | 40 | 0.00149 |

★**Retroactive note (added 2026-07-29, §5.2):** the `concurrent_frac_of_bench`
column above is the RETIRED count-based quantity computed at the time of
this diagnosis (2026-07-28) — it is now known to badly understate true
wall-clock concurrency (§5.2). These specific 865832 (NP=20) values were
never re-verified time-weighted; do not read them as "concurrency was
negligible" — by the same mechanism found at 866066, true concurrency here
was very likely tens-of-× higher than shown. Kept as-is for the historical
record of the pin-gate (§5.1's actual subject) investigation, which is
unaffected by this caveat.

For comparison, the ORIGINAL (retired) `engaged = prefill_active OR
decode_active` population for d16 was 130 samples of which only 4 (3%)
matched target — but 125 of those 130 (96%) were legitimately decode-only
`(0,108)` windows (decode running, prefill idle — CORRECT behavior per
`multiplexing_mixin.py`'s stream-selection logic, not a partition failure),
which is exactly what diluted the fraction to a false FAIL. Conditioning on
`prefill_active_batch_size > 0` alone removes those 125 irrelevant samples
entirely; what remains for d16 is a single genuine prefill-active
observation, and that ONE observation is at the target partition (100%).
d92's larger prefill-active population (41, because the D92 cell's 16-SM
prefill windows last longer than D16's 92-SM prefill windows, more likely
to be caught by the ~2ms telemetry sampling grain — see §9.7) gives a
directly interpretable, gate-passing 97.6%.

**Verdict: Hypothesis A confirmed** (gate population was wrong), not
Hypothesis B (genuine reversion under low load) — though d16's single
observation is itself underpowered (§9.7 discusses whether this persists at
main-sweep NP). No evidence of the legacy-revert-under-load failure mode
in this data once the population is corrected.

**File:line comparison of the two pin-check definitions** (as requested):
- `s8p_prefill/prefill_pin_check.py:51` (`if
  e.get("prefill_active_batch_size", 0) <= 0: continue`) → population is
  prefill-active ONLY. Line 67 (`frac = realized[expect] / n`) divides by
  that same population. This is what E1's PRIMARY gate now matches.
- `e1_pin_check.py` (pre-revision) line 109 (`engaged = p_active or
  d_active`) → population was the OR of prefill-active and decode-active,
  diluting the denominator with decode-only windows that are supposed to
  show the unpartitioned stream. This was the bug; fixed in the current
  version of the file (see its module docstring for the full before/after).

Full numeric diagnosis (including the whole-file histograms) is reproduced
in this section from the raw 865832 telemetry; no separate diagnosis file
was kept since the corrected `e1_pin_check.py`/`e1_analyze.py` now
reproduce these exact numbers when re-run against the same telemetry (see
§5 above for the corrected code, and §9.7 for what this implies about the
main-sweep design).

### 5.2 ★★Diagnosis of the 866066 (NP=150) concurrency-diagnostic bug
    (2026-07-29) — a SECOND, DIFFERENT bug in `concurrent_frac_of_bench`,
    and RETRACTION of the "D-axis barely exercised / ill-posed" conclusion

**Result of the §8.1 diagnostic smoke (866066, NP=150).** The §5.1 pin-gate
fix works as intended: d92 now has n=118–242 prefill-active samples with
pin 0.968–1.000 (clean PASS); d16 has n=12, correctly flagged
`UNDERPOWERED` (not a false "wrong SM" — the fast-92-SM-prefill sampling
problem §5.1/§9.7 anticipated is real and did not resolve at NP=150,
addressed further below).

**But the coordinator caught a second, independent bug in
`concurrent_frac_of_bench` itself**, using an arithmetic cross-check: at
T8/d92 (866066, rate≈2 over a ~79s LO window, 150 requests, median TTFT
234.99ms), `sum(median_ttft × n_requests) / window ≈ 44%` — yet the
(then-current) `prefill_active_frac_of_bench` reported only **0.0152**, a
~29× discrepancy. d16 showed a ~240× discrepancy in the same direction. The
mismatch ratio DIFFERING by cell (29× vs 240×, not a constant factor) ruled
out a simple constant-scale bug and pointed at something duration-
dependent.

**Root cause — confirmed via file:line (coordinator's requirement #3):**
`runtime_snapshot` is written from `_dual_worker_sync`
(`multiplexing_mixin.py:356`), called from EXACTLY ONE place inside
`event_loop_pdmux`'s outer `while True:` loop body per pass (lines 790 and
1061 — twice per outer iteration, both unconditional on what work that
iteration did), incrementing `dual_worker_trace_count` each time
(`multiplexing_mixin.py:389`) and writing a trace row only on a
count-based subsample — `dual_worker_trace_count == 1 or
dual_worker_trace_count % dual_worker_trace_every == 0`
(`multiplexing_mixin.py:391-395`), with `dual_worker_trace_every` defaulting
to 32 (`multiplexing_mixin.py:90-95`, `PDMUX_DUAL_WORKER_TRACE_EVERY` env
override). **This subsampling cadence is EVENT-LOOP-ITERATION-based, not
wall-clock-based, and — critically — it is applied by the exact same
counter/modulo rule regardless of whether the iteration was a prefill step
or a decode step** (requirement #3's "같은 규칙" question: yes, same rule,
but that is exactly why the bias exists) — an outer-loop pass that does one
prefill forward (16 SM: ~235ms wall-clock this run; 92 SM: a few ms) and an
outer-loop pass that does one decode step (~11-12ms ITL this run) each
contribute the SAME ~1-2 increments to `dual_worker_trace_count`,
REGARDLESS of how long they took. So "fraction of SAMPLES with property X"
silently measures "fraction of EVENT-LOOP ITERATIONS with X", which
underweights long-duration states (prefill, especially at low SM) relative
to short-duration ones (decode steps) by roughly their duration ratio —
observed empirically as tens-of-× in both directions of the cell asymmetry
above (d92's slower, 16-SM prefill windows are longer, so this
UNDERSTATEMENT is somewhat less severe there than at d16's fast 92-SM
prefill windows, matching the 29× vs 240× asymmetry qualitatively).

**Time-weighted recomputation (requirement #1), from the EXACT SAME
866066 telemetry, using the same `[t0,t1]` windows `e1_sweep.sbatch`
already records in `<RUNID>_rounds.jsonl`** — each snapshot's state is
held from its own `timestamp_monotonic_s` to the next snapshot's (capped to
the window's `t1` for the final in-window row; for an unbounded whole-file
check the trailing duration of the very last row is simply dropped, one
row's negligible weight out of thousands — see `e1_pin_check.py`'s
`compute_gates` for the exact implementation):

| cell/phase | window (s) | n_prefill_active | pin_frac | `prefill_active_time_frac` (NEW, correct) | `concurrent_time_frac` (NEW) | `prefill_active_frac_count_based` (OLD, retired) |
|---|---|---|---|---|---|---|
| d92 LO (rate 2) | 95.06 | 124 | 0.968 | **0.248** | **0.248** | 0.0152 |
| d92 HI (rate 4) | 59.22 | 118 | 1.000 | **0.398** | **0.398** | 0.0154 |
| d16 LO (rate 2) | 113.92 | 10 | (underpowered) | **0.0129** | 0.0102 | 0.0006 |

The time-weighted numbers are **~16–26× higher** than the retired
count-based ones for these three windows, matching the coordinator's
arithmetic mismatch in direction and (given the naive TTFT-sum bound is a
loose UPPER bound expected to overshoot, especially under queueing) in
rough order of magnitude.

**Requirement #2, independent client-side cross-check**, computed
precisely (not just the back-of-envelope in the coordinator's message) from
the `_rep1_{lo,hi}.jsonl` `ttfts`/`duration` fields directly (bench_serving
does not record per-request absolute arrival timestamps, so an exact
interval reconstruction is not possible from this file alone — `sum(ttfts)
/ duration` is used instead, which is a valid but LOOSE upper bound: it is
exact only if no two requests' TTFT-windows overlap, and overlapping
requests inflate the sum above the true union-of-intervals occupancy):

| phase | n | duration (s) | sum(ttft) (s) | naive upper bound = sum(ttft)/duration |
|---|---|---|---|---|
| d92 LO | 150 | 79.06 | 49.54 | **63%** |
| d92 HI | 150 | 43.13 | 110.19 | **255%** (loose — heavy queueing under-rate-4 congestion on a 16-SM prefill cell, TTFTs overlap heavily, sum exceeds the window) |
| d16 LO | 150 | 79.00 | 10.09 | **13%** |

These loose upper bounds and the time-weighted telemetry numbers agree on
the THING THAT MATTERS — true occupancy is **tens of percent, not ~1%** —
even though the two methods do not (and should not be expected to) match
exactly. Three independent signals (telemetry time-weighting, client-side
TTFT-sum bound, and the coordinator's own back-of-envelope) now triangulate
on the same conclusion.

**★★RETRACTION: the previous conclusion ("D-axis barely exercised even
where pin passes, structural tension resembling vector-1's ILL-POSED
finding," written into the earlier version of §9.7) was ITSELF the
artifact, not a real finding.** True prefill/decode concurrency at 866066
was 25–40% for d92 (not ~0.15–1.5%) — a substantial, well-exercised
overlap regime, not a negligible one. d16 remains genuinely low-concurrency
(~1%) even after the fix, but that is a real, mechanistically-understood
asymmetry (92-SM prefill finishes so fast there is little wall-clock in
which to overlap with anything, at this rate) rather than a red flag about
the campaign's viability. See §9.7 for the full corrected risk framing.

**Requirement #4 — does `s8p_prefill/prefill_pin_check.py` have the SAME
bug?** No — **and this is a meaningful distinction, not a technicality.**
`concurrent_frac_of_bench`'s bug was specifically about using a
snapshot-COUNT ratio to answer an ABSOLUTE WALL-CLOCK-SHARE question ("how
much of total time did X happen"). `prefill_pin_check.py`'s PRIMARY output
(`frac = realized[expect] / n`, both counted over the SAME prefill-active
population) is a CONDITIONAL EVENT PROPORTION ("of the times prefill was
active, what fraction were at the target SM") — the semantically correct
unit for that question is REQUESTS/EVENTS, not wall-clock time, since the
downstream client-side latency analysis is itself computed per request, not
per unit time. So `prefill_pin_check.py`'s pin fraction (and E1's own
now-fixed PRIMARY gate, which uses the identical conditional-proportion
design) are NOT vulnerable to the §5.2 bug in the way `concurrent_frac_of_bench`
was.

A **read-only** check of `s8p_prefill`'s own 865973/865974 telemetry (T8
arm, cells p44/p92 — chosen because both FAILed near the 0.665–0.80
boundary the coordinator flagged; no files modified, this does not
duplicate result-analyst's work, it only characterizes bias direction/size
as asked) found something else worth flagging to them: the "wrong SM"
samples are **100% anti-correlated with co-residency** (p44: 0/165 "wrong"
samples had `decode_running_batch_size>0`, vs 619/619 "right" samples that
did; p92: 0/163 vs 490/490) and recur **periodically roughly every ~14s
across the ENTIRE ~335s window** (not clustered at startup) — consistent
with `s8p_prefill`'s own documented risk (`DESIGN.md` §7: "probe 간 짧은
간극에서 `split_prefill_batch`가 일시적으로 비어" / the keepalive
background decode load occasionally, periodically draining to empty across
its `decode_conc` workers, at which point the engine correctly falls back
to a non-target stream for that instant). This looks like a REAL,
recurring keepalive-decode-duty-cycle gap, not a measurement artifact of
either the §5.1 or §5.2 kind — i.e., **the marginal FAILs in
865973/865974 most likely reflect a genuine, periodic partition-reversion
problem tied to their keepalive design, not a gate-definition bug**. One
residual, unresolved nuance worth flagging to result-analyst (not resolved
here, to avoid duplicating their analysis): each ~14s gap produces a
cluster of ~7-8 consecutive "wrong" samples (24 gaps × ~7 ≈ matches the
observed n_wrong almost exactly), so the reported sample-level fraction
(0.79, 0.75, etc.) is better read as "share of PREFILL-ACTIVE WALL-CLOCK
TIME at target" than "share of DISTINCT PROBE REQUESTS at target" — these
could differ somewhat if de-duplicated at the episode level, a possible
but not yet attempted refinement.

### 5.3 ★★★A FOURTH bug in the same lineage (2026-07-29, coordinator
    re-diagnosis) — snapshot-COUNT was the wrong OBJECT for the pin gate,
    not just the wrong population (§5.1) or the wrong weighting (§5.2);
    fixed by porting `s8p_prefill`'s episode bracketing

**Diagnosis.** §5.1's fix conditioned the pin gate on `runtime_snapshot`
rows with `prefill_active_batch_size > 0` — correct POPULATION (bug #1),
but never validated as the correct UNIT. `--chunked-prefill-size -1` means
one request's entire prefill runs as ONE forward pass inside ONE event-loop
iteration (`event_loop_pdmux`'s `while True:` body, `multiplexing_mixin.py`)
— so a prefill EPISODE, however long its wall-clock duration, produces AT
MOST one `_dual_worker_sync` call with `prefill_active_batch_size>0`, and
`dual_worker_trace_every=32` subsampling then discards most of even that
single opportunity. Meanwhile idle/decode-only iterations fire at ~200-280Hz
(measured: 208s → 57,993 snapshots at 866868's scale). Empirically:
**150 real requests produced only ~10-40 `prefill_active_batch_size>0`
snapshots depending on cell** — roughly 1/4 to 1/15 of episodes captured
at all, and NONE of D16's (the fastest-prefill cell) ever showed up this
way even once across NP=150 (§9.7/§9.8). This is a FOURTH distinct
manifestation of the same underlying class of error this session has now
hit repeatedly (target-vs-realized → §5.1's OR-population →
§5.2's count-vs-time → this section's snapshot-vs-episode) — snapshot
COUNTS were never validated as a stand-in for EPISODE counts, and for a
single-forward-pass-per-request engine they systematically are not one.

**Fix — episode accounting, ported (not reinvented) from
`s8p_prefill/s8p_analyze.py`'s interval-bracketing technique**, which
passed on all 16 (arm, cell) combinations in that campaign this same way:

1. **Reconstruct each request's `[admission, first-token]` interval.**
   `sglang.bench_serving`'s output jsonl has no per-request absolute
   timestamps, so this campaign's own §4.5 seeded-RNG arrival replay is
   reused: `arrival_rel[i]` (request i's dispatch time relative to the
   first) from `np.random.exponential(1/rate)` draws, `completion_rel[i] =
   arrival_rel[i] + ttft[i] + sum(itls[i])`. **Anchoring**: unlike
   `e1_capacity_scan.sbatch`'s own use of this replay (which only needs
   RELATIVE timing for `arrival_rps`), the pin gate needs ABSOLUTE times to
   bisect against telemetry — anchored so the reconstructed LAST request's
   completion lands exactly at the phase's own measured end (`T1_LO`/
   `T1_HI`, already recorded in `<RUNID>_rounds.jsonl`): `T0_abs = T1_anchor
   - max(completion_rel)`. **Validated** against 866066's real data: this
   implies a ~16.1s gap between the wrapper's own `T0_LO` (captured just
   before the `bench_serving` subprocess launches) and the reconstructed
   anchor — and `bench_serving`'s OWN reported `duration` field (79.06s)
   vs the wrapper's `T1_LO - T0_LO` (95.06s) independently gives the SAME
   ~16.0s gap (server-readiness-check + `--warmup-requests` overhead) to
   within 0.1s. Two independent computations agreeing to ~0.1% is strong
   evidence the anchor is correct, not an artifact.
2. **Bisect + bracket, verbatim from `s8p_prefill/s8p_analyze.py`**: for
   each request's interval `[a, b]`, find the telemetry snapshot
   immediately before `a` (`ii`) and immediately after `b` (`jj`). Accept
   only if `psm[ii] == psm[jj]` (realized `prefill_sms` agrees at both
   ends) AND both are within `GUARD=3.0s` of `a`/`b` respectively — same
   algorithm, same GUARD value as the ported campaign.
3. **★Necessary addition found during validation, NOT in the literal
   port: exclude "vacuous-idle" brackets.** Applying step 2 literally (no
   further filter) to 866066's D92 LO phase gives `pin_frac=0.233` (FAIL)
   — directly contradicting the independently-established ~0.97-1.00
   realized pin for that exact cell/window (§5.1's own fix, and the
   client-side TTFT evidence in §5.1). Root cause: 78 of D92's 103
   "accepted" brackets have `prefill_active_batch_size==0` at BOTH `ii` and
   `jj` — the bracket is internally consistent (both endpoints agree) but
   neither endpoint ever observed prefill actually running; this happens
   when an episode is short enough to complete entirely between two
   idle-looking snapshots, and it defaults to the non-target
   `prefill_sms=0` fallback value, which the literal algorithm then
   (wrongly) counts as evidence AGAINST the target. Excluding these 78
   brackets (`n_vacuous_idle`) leaves 24 **informative** episodes, all 24
   at the target `prefill_sms=16` — `pin_frac=1.000`, matching §5.1 exactly.
   This exclusion is reported transparently (`n_vacuous_idle`,
   `n_informative`, both histograms) rather than silently applied, per this
   session's established discipline of not hiding a refinement inside an
   unlabeled number.
4. **Gate criterion — Clopper-Pearson 95% one-sided lower bound ≥ 0.80,
   over informative episodes**, replacing the old two-part
   `pin_frac>=threshold AND n>=min_n` check (§5.1's `MIN_N_PREFILL_ACTIVE`
   floor is retired, no magic number 20 any more): `scipy.stats.beta.ppf(0.05,
   k, n-k+1)` for `k` target-matching informative episodes of `n` total. A
   small `n` widens this bound automatically and fails the SAME criterion a
   genuinely-wrong-SM sample would fail — e.g. 866868's T8/D92 LO round
   (n_informative=8, pin_frac=1.000 point estimate) still correctly FAILS at
   `lower95=0.688 < 0.80`, because 8 perfect observations are not yet enough
   absolute evidence at 95% confidence — exactly the intended behavior.

**D16's status, definitively (not merely "underpowered" — a real, telemetry-
invisible-at-this-density cell)**: re-run with the corrected, informative-
only method, D16 (866066, NP=150) has **0 informative episodes out of 138
accepted brackets — ALL 138 were vacuous-idle**. This is stronger and more
specific than §5.1/§9.7's earlier finding (`n_prefill_active=10-12`
snapshot count) — it says the 92-SM prefill window is so brief that, across
150 real requests, NOT ONE bracket-consistent episode ever coincided with a
snapshot showing prefill genuinely active on either side. §9.8 (below)
finalizes what to do about this with the coordinator's own guidance.

★★★**RETRACTED BY §5.4 BELOW — §5.3's own "definitively" framing was
itself premature.** §5.3 read the realized partition at the bracket
ENDPOINTS, which are by construction outside the request's own prefill
window — so of course they never showed the target for a fast cell. §5.4
fixes this (read from INSIDE the bracket instead) and the "0 informative,
definitively telemetry-invisible" conclusion above does not survive it:
D16 does have informative episodes once read correctly, just very few.
Kept here, retracted rather than deleted, as a record of how many times
this same class of error recurred in one session (see §5.4's own history
list).

### 5.4 ★★★A FIFTH bug (2026-07-29, coordinator + engine-porter
    re-diagnosis, while validating `PDMUX_TRACE_FORCE_PREFILL`) — the
    episode gate read the realized partition at the bracket ENDPOINTS,
    which are OUTSIDE the prefill window by construction; fixed by reading
    INSIDE the bracket instead

**Diagnosis.** §5.3's episode gate accepts a bracket via the endpoints
`ii` (last snapshot at/before admission `a`) and `jj` (first snapshot
at/after first-token `b`), and — this is the bug — ALSO read the realized
`prefill_sms` FROM those same two points. Both `ii` and `jj` are, by
construction, outside `(a, b)`: `ii` is whatever the runtime looked like
just BEFORE this request was admitted, `jj` just AFTER it finished. For a
fast-prefill cell, the runtime's auto-partition between requests is the
idle fallback (`(0,108)` or `(108,0)`, §5.1), so the endpoints structurally
can never see the target partition regardless of how well-pinned the
ACTUAL prefill computation was — confirmed directly: 866066 D16's
endpoint histogram over 150 accepted brackets was `P0:143(95%),
P92:4(3%), P108:3(2%)` — essentially never the P92 target, no matter how
much sampling density is added, because the endpoints are the wrong two
instants to look at, not because the samples are too sparse.

A read-only diagnostic engine-porter built while validating their
`PDMUX_TRACE_FORCE_PREFILL` patch (`results/e1_traceforce/tfgate_inside_bracket.py`
— imports `e1_pin_check`, does not modify it) asked the complementary
question: within `(a, b)` itself, are there any `prefill_active_batch_size>0`
snapshots, and what do THEY show? Run against the SAME 866066 telemetry
this DESIGN.md already uses: **D16 — 1 of 150 episodes has an inside-bracket
active snapshot, showing `P108` (NOT the target, pin_frac=0.000 on that
single sample); D92 (control) — 49 of 150, ALL showing `P16` (the target,
pin_frac=1.000, matching §5.1/§5.3's independent evidence exactly)**.
Re-running `tfgate_inside_bracket.py` (verbatim, unmodified) against the
identical files this agent already had on disk reproduced these exact
numbers, confirming the finding is a real property of the endpoint-reading
bug, not an artifact of a different dataset.

**Fix, implemented in `e1_pin_check.compute_episode_gate`**: the bracket
endpoints (`ii`, `jj`, GUARD proximity check, ported from
`s8p_prefill/s8p_analyze.py`) are now used ONLY to confirm the interval's
own boundaries are close to real telemetry (a timing sanity check) — NOT
to read the realized partition. The old `psm[ii]==psm[jj]` consistency
requirement is DROPPED (it was a proxy that only made sense when reads
came from the endpoints themselves). The realized partition is now read
from snapshots STRICTLY INSIDE `(a, b)` with `prefill_active_batch_size>0`
(`lo = bisect.bisect_right(ts, a)`, `hi = bisect.bisect_left(ts, b)`, same
indexing as `tfgate_inside_bracket.py`); a bracket with no such snapshot is
`n_vacuous_idle` (unchanged semantics, re-scoped from "endpoints agree and
are idle" to "nothing found inside"); if multiple inside-active snapshots
disagree on `prefill_sms` (not expected under a static config), the
chronologically first is used and the disagreement is counted
(`n_inconsistent_partition`), never silently resolved.

**Validation (requirement 2 — reproduce the D92 control) — done, with an
honest caveat about the exact count.** Re-running the fixed
`compute_episode_gate` against the SAME 866066 D92 telemetry this
DESIGN.md already used gives **`n_informative=49`, `pin_frac=1.000`**,
EXACTLY matching `tfgate_inside_bracket.py`'s own count on the identical
files (cross-validated by running that unmodified reference script
directly against this agent's own telemetry files, byte-for-byte the same
inputs). `pin_frac=1.000` (the accuracy criterion) is reproduced exactly.
The SPECIFIC counts in the coordinator's message (D16: 2 informative, both
P92; D92: 44 informative) do not match what this agent's own 866066 files
give (D16: 1 informative, P108; D92: 49 informative, all P16) — but since
running the coordinator's OWN unmodified diagnostic script against this
agent's OWN files reproduces this agent's numbers exactly, the discrepancy
is a DATA PROVENANCE mismatch (the coordinator's cited numbers most likely
came from a different telemetry capture — a different rep, phase, or job
— not from `e1_T8_d16_866066_rep1_lo.jsonl`/`e1_T8_d92_866066_rep1_lo.jsonl`
specifically), not a bug in either implementation. **This is flagged
rather than silently reconciled**: it means, on the ONE dataset available
to verify against, D16 STILL fails the pin gate after this fix (n=1,
showing the non-target `P108`) — a materially different, and less
optimistic, outcome than "d16's evidence already exists and is 2/2
correct" for THIS specific data. Whether a genuinely well-pinned D16
episode is discoverable in general remains to be seen once
`PDMUX_TRACE_FORCE_PREFILL` actually multiplies the sampling density in a
NEW run (§4.7) — this fix makes the gate capable of finding it if it is
there; it does not manufacture evidence where none exists in already-collected data.

**`admission_blocked_frac` fix (requirement 3)**: also count-based, and
also biased once `PDMUX_TRACE_FORCE_PREFILL` is on — every FORCED record
is, by construction, `prefill_active_batch_size>0` at the moment it fires,
which is mutually exclusive with `prefill_admission_blocked`
(`dual_worker.py:590-599`). Counting forced records in the denominator
mechanically dilutes this fraction, and does so MORE for cells where
forcing adds proportionally more records (§4.7's asymmetry — D92 far more
than D16), a CELL-DEPENDENT bias that would corrupt any cross-cell
comparison once the flag is on. Fixed: `compute_concurrency_diagnostic`
now computes `admission_blocked_frac` only over samples with
`trace_forced != True` (a no-op with the flag off or on pre-patch
telemetry, since that field is then always absent/false — verified: this
agent's 866066 re-test after the fix reproduced the exact same
`admission_blocked_frac=0.001` as before the fix). The time-weighted
concurrency fractions (`*_time_frac`) are NOT filtered this way — forcing
only adds resolution to a genuinely time-weighted quantity, it does not
mechanically bias it the way a blocked/not-blocked COUNT ratio is biased
by an always-not-blocked-by-construction extra population.

### 5.5 ★★★A SIXTH bug (2026-07-30, coordinator re-diagnosis from raw
    866066 telemetry) — the episode-bracket approach (§5.3/§5.4), even
    fully fixed, answers the wrong question FOR A GATE; replaced the
    PRIMARY gate with a time-weighted estimator over ALL prefill-active
    samples

**Diagnosis.** §5.4 fixed the episode gate to read the realized partition
from INSIDE each request's `(a,b)` bracket rather than at its endpoints —
correct for what that gate is FOR (attributing a specific request's latency
to the partition that was actually in effect while it ran), but the
coordinator's own hand-aggregation of the full 866066 D16 telemetry file
(32,145 total snapshots) found that this is not the same question as "was
the runtime pinned to the target partition." Direct count: D16 had exactly
**12 prefill-active snapshots in the entire file — 7 at the target `P92`,
5 at the idle-fallback `P108`** — count-fraction 7/12 = 0.583, matching
§5.4's episode-gate output on the nose (both are conditioning on the same
tiny population, just accessed two different ways: one per-snapshot, one
per-request-bracket). But **TIME-weighting those same 12 snapshots** (each
weighted by the wall-clock gap to the next snapshot, exactly
`compute_concurrency_diagnostic`'s existing technique from §5.2) flips the
answer: the 5 fallback snapshots are systematically SHORTER-duration than
the 7 target ones (the runtime reverts to full-GPU decode only in the
brief windows between back-to-back prefills, then repartitions as soon as
the next one is admitted), giving **1.719s of 1.719+0.311=2.030s at the
target = 84.7% time-weighted pin**, not 58.3%. D92 control: 47.091s of
47.144s = 99.9%, matching every prior measurement of that cell exactly (it
was never in question).

**Why the episode gate can never answer the aggregate question it was
being asked to answer, even fixed.** A gate's job is "what fraction of
this cell's prefill-serving TIME ran at the target partition" — an
aggregate, denominator-is-wall-clock question. The episode gate's
denominator is "how many REQUESTS have any inside-bracket evidence at
all" — for a fast cell like D16 that is a tiny, request-count-limited
sample by construction (one request's whole prefill is one iteration,
§5.3), and discarding all but a handful of brackets to get a clean
per-request attribution is *correct behavior for attribution* and *the
wrong entry to gate on*, because it throws away exactly the duration
information (which snapshots were long vs short) that decides the
aggregate answer. Two estimators were being asked to do each other's job:
the episode gate (built for per-request latency attribution, correctly
data-starved by design) was being read as if it were the aggregate
occupancy gate, while the real aggregate answer was sitting one time-weighting
step away, in the same 12 samples, the whole time.

**Fix, implemented as `e1_pin_check.compute_time_weighted_pin_gate`
(NEW function, PRIMARY gate) + `time_weighted_gate_verdict`**: build
"prefill episodes" as maximal contiguous runs of
`prefill_active_batch_size>0` telemetry snapshots (not per-request
brackets — a purely engine-observable segmentation, needs no RNG-replay
reconstruction of the workload at all), each contributing
`(t_total_seconds, t_at_target_seconds)` via the same time-weighting as
`compute_concurrency_diagnostic`. The pooled `pin_frac = sum(t_at_target) /
sum(t_total)` over ALL such episodes in the window is the point estimate.
**Uncertainty**: Clopper-Pearson (§5.3/§5.4) is for i.i.d. Bernoulli trials
and is the WRONG tool here — the underlying observations are
autocorrelated durations within episodes, not independent coin flips, so
its lower bound would be systematically too narrow (overconfident). Fixed
per the coordinator's explicit instruction: **episode-level cluster
bootstrap** — resample whole episodes (not individual snapshots) with
replacement, `n_boot=10000`, recompute the pooled time-weighted `pin_frac`
each draw, report the 2.5th percentile as `pin_frac_lower95`. This
correctly treats each PREFILL EPISODE (not each snapshot, not each
request) as the unit of independent evidence, and is reported alongside
**`n_episodes`** (the effective sample size the coordinator asked to have
reported honestly) so a low-n cell's wide bootstrap CI is visible rather
than hidden behind a point estimate. `n_episodes ∈ {0, 1}` is handled as a
special case (0: "no prefill-active time in this window, cannot judge";
1: point estimate reported but flagged `[UNRELIABLE: bootstrap CI not
meaningful]`, since a bootstrap of a single cluster cannot estimate
between-cluster variance).

**Validation against the coordinator's own hand-derived numbers** (whole-file
866066, no LO/HI windowing, i.e. the same population the coordinator
aggregated by hand): `python3 e1_pin_check.py e1_T8_d16_866066_telemetry.jsonl 16 0.80`
→ `pin_frac=0.847, n_episodes=8, pin_frac_lower95=0.554` → **FAILS** the
0.80 gate (0.554 < 0.80) — an honest failure due to LOW STATISTICAL POWER
at only 8 independent prefill episodes in the whole file, not due to a
bad point estimate (0.847 is itself comfortably above 0.80) and not due to
zero evidence (§5.3's now-doubly-retracted framing). `python3
e1_pin_check.py e1_T8_d92_866066_telemetry.jsonl 92 0.80` →
`pin_frac=0.999, n_episodes=75, pin_frac_lower95=0.996` → **PASSES**. Both
numbers match the coordinator's cited 0.847/0.999 point estimates exactly.
(Note: `e1_analyze.py`'s own per-rep table applies this same gate
separately to each LO/HI *sub-window* of a rep, so its printed `n_episodes`
and `pin_frac` for D16 will differ from — and typically be smaller/noisier
than — these whole-file numbers; this is expected windowing, not a
discrepancy.)

**Role of the retained episode-bracket gate (§5.4), rescoped per explicit
instruction not to alter its already-correct fix**: `compute_episode_gate`/
`episode_gate_verdict` remain **exactly as fixed in §5.4** (bracket, GUARD,
inside-bracket read, vacuous-idle exclusion — no code changed) but are no
longer imported by `e1_analyze.py`'s citeability filter. They are now
printed purely as an **informational, non-gating latency-attribution
diagnostic** (labeled `E1_EPISODE_ATTRIBUTION` in `e1_pin_check.py`'s CLI
output) — useful for answering "was THIS SPECIFIC slow/fast request
plausibly served under the target partition," never for deciding whether a
cell/rep is citeable.

**★ Requirement 3 (pre-registered here, do not walk back without updating
this file): D16's non-pin time is a REAL POLICY BEHAVIOR, not a bug, and
must be reported as a first-class result, not hidden.** The runtime's own
scheduler auto-reverts to a full-GPU (`108/0` or `0/108`) partition
whenever decode's running batch drains to empty between the sparse arrivals
a fast-prefill cell produces (this is correct, intended behavior of the
underlying auto-partition fallback, not a defect in the harness or the
engine — see §12/§9.7 for the mechanism). Concretely for D16 (866066,
whole file): **85% of prefill-active wall-clock time ran at the TARGET
`[92,16]` partition, 15% ran un-partitioned at the idle-fallback `P108`**
(`realized_hist = {92: 1.719s, 108: 0.311s}` out of `t_prefill_active_total
= 2.030s`). **The main sweep MUST report, per cell, per rep, per phase,
the full time-weighted `realized_hist` breakdown** (already wired into
`e1_analyze.py`'s per-rep print block as "realized partition split
LO:/HI:") and describe results using "target D=16, realized target
85%/auto-partition 15%" language, never bare "D=16" — a cell label like
`[92,16]` names a TARGET configuration, not a guaranteed realized one.

**★ Requirement 4 (pre-registered): this exposure is asymmetric across the
grid, and the asymmetry is itself a result, not noise.** D92's control
cell has effectively 0% exposure to auto-partition reversion (99.9%/0.1%
in the same file) because decode's running batch rarely empties at that
end of the frontier. The fast-prefill, low-decode-SM end of the grid (D16,
and to a lesser extent D24) is structurally the MOST exposed to this
effect, because that is precisely where inter-arrival gaps are long enough
relative to prefill duration for decode to drain and the runtime to
auto-revert before the next admission. This is the SAME root mechanism as
the concurrency asymmetry already documented in §4.2.4/§9.7 (d16
1.3%/d44 4.9%/d92 25-40% `concurrent_time_frac`) — one root cause
(prefill-rich cells see long idle gaps between rare, fast prefill bursts)
produces two visible symptoms (low concurrency exercise AND exposure to
un-partitioned auto-revert time). **Net effect on interpretation**: the
15% auto-partition exposure at D16 runs at `108/0`, i.e. prefill gets MORE
SM than its 92-SM target during that time, which — if it materially
speeds up D16's prefill — biases D16's measured TTFT to look BETTER than
a "genuinely always-92-SM" D16 would, i.e. it favors the
prefill-SM-rich end of the frontier, not the decode-SM-rich end.
Structurally similar to Stage 0's D108-anchor failure (`stage0-deconfound-contested`
memory entry — a cell label not matching its realized partition) but a
**different root cause**: Stage 0's was a static config-pinning bug,
this is the runtime's own dynamic, by-design idle-fallback policy
reasserting itself between sparse arrivals — it cannot be "fixed" by a
harness change, only measured, reported, and (if it materially affects a
citeable conclusion) mitigated by a workload change (e.g. shorter
inter-arrival times at D16 specifically, discussed as a live option in
§9.8).

## 6. Deviations from the precedent recipes (documented per task instructions)

### 6.1 Single boot per cell, REPS loop inside (not `g2_0_raconf`'s "each rep
a separate server boot")

`g2_0_raconf`'s auditor-specified recipe used a fresh server boot per rep
for a 2.7B model (Zamba2-2.7B boots in tens of seconds). 7-8B models in this
project's own `s8_scaleup` runs took observably 3–10+ minutes per boot
(§7.1) — 4 separate boots per cell would multiply that overhead 4×, making
the full grid infeasible within a reasonable time budget. This harness
instead follows the **`s8_scaleup`/`s8p_prefill` precedent** (which this
task explicitly told this agent to reuse as the skeleton): one boot per
cell, `REPS` sequential rounds inside it, each round's LO/HI phases written
to their own never-overwritten jsonl files so `e1_analyze.py` can still
compute genuinely rep-wise (not pooled) statistics. The independence this
buys is weaker than a fresh boot (shared KV/radix-cache-disabled state,
shared GPU clock history across reps of the same cell) but the project's
own precedent treats this as acceptable for n≥4 rep-wise stats, and
GPU-clock sampling (`clocks.csv`) is still recorded per cell so a
clock-drift confound would be visible if it appeared.

### 6.2 Capacity scan settle = fixed 5s sleep, not full drain-detection

Covered in §4.2 — scan output is elbow-location only, never cited.

## 7. Time budget (estimates — see caveats)

**Boot cost.** From `s8_scaleup` sacct/srv.log timestamps (865311/865312,
2026-07-27): observed cell-to-cell deltas of ~166s (fastest, warm triton
cache) to ~634s (slowest observed, M8 d44) for boot alone; treat 3–10 min/
boot as the working range, arm-dependent.

**Capacity scan** (`e1_capacity_scan.sbatch`, per arm, all 5 cells): boot +
8 rates × (20s probe + 5s settle) ≈ boot + 200s ≈ 6.3–13.3 min/cell →
**≈32–67 min/arm**, all 5 cells sequential in one job. Running this for all
4 arms up front (recommended — it's cheap relative to the main sweep and
de-risks the whole campaign) is **≈2.1–4.5h serial**, or roughly half that
wall-clock with 2 parallel single-arm jobs (QOS run≤2).

**Main sweep** (`e1_sweep.sbatch`, per cell): boot (3–10 min) + REPS=4 ×
(NP/RATE_LO + drain(~10–40s) + NP/RATE_HI). With `NP=150` and rates in the
1–10 req/s range typical of the `sharegpt_vary`/`g2_0` precedent (RATE_LO
~2–4, RATE_HI ~4–12), a rep's bench time is very roughly 30–150s, so REPS=4
≈ 2–10 min bench, cell total **≈5–20 min**. NP is an env override
(`NP=100` if a slow arm's scan says 150 is too expensive at its rates).

- **Stage 1 (T8+Hs8, 5 cells each = 10 cells)**: ≈50–200 min serial (one
  job, `ARM_IDS="3 2"`), or ≈25–100 min wall if submitted as 2 parallel
  single-arm jobs (recommended, uses both QOS run slots).
- **Stage 2 (M8+Ha8)**: same order, submitted after stage 1's gates and
  decision-rule output are sane.
- **Grand total, both stages + capacity scan**: very roughly **4–10 hours
  of GPU compute**, ≈2–6 hours wall-clock with parallel submission,
  **on top of whatever fairshare queue wait applies** (unknown and
  currently non-trivial — the task briefing notes the queue is fairshare-
  limited and `s8p_prefill`'s smoke is PENDING as of this writing, 2026-07-28,
  squeue confirms exactly 1 job, `865738 s8psweep PENDING`).

These are estimates from a structurally similar but not identical harness
(closed-loop keepalive vs open-loop ShareGPT rate-varying) — the capacity
scan itself will produce the first real E1-specific timing data; if actual
rep bench times differ substantially from this estimate, revise `NP`/`REPS`
before the full stage-2 submission rather than after.

## 8. Smoke test (run before either capacity scan or main sweep — NOT
   executed by this agent, commands only)

Per this task's "제출은 하지 마라" instruction, none of the following has
been submitted by this agent. **Note (2026-07-28, coordinator):** a smoke
job (`865832 e1sweep`, PENDING as of the SLO revision) was submitted
directly by the user for pipeline-plumbing verification — it predates and
is independent of the SLO revision in §4.3 (a smoke test only needs SOME
SLO value to exercise `e1_analyze.py`'s code path, not the correct one) and
does not need to be resubmitted. Recommended order for the remaining,
not-yet-submitted steps once the user decides to proceed:

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

# 1) config/harness sanity, single arm (T8, fastest boot), two extreme
#    cells, no rate decisions needed yet (bench_serving default request-rate
#    behavior with a tiny NP is enough to check plumbing, not capacity):
sbatch --export=ALL,ARM_IDS="3",SMOKE_CELLS=1,RATE_LO=2,RATE_HI=4,NP=20,REPS=1 \
  workspace/engine-port/results/s8_frontier/e1_sweep.sbatch

# Checks after it lands (mirrors s8p_prefill/DESIGN.md sec 8's 3-point gate):
#   (i) both cells reach boot_ok=1
#   (ii) e1_pin_check.py prints a PASS/FAIL verdict (either is fine for a
#        smoke -- the point is the pipeline runs end to end)
#   (iii) e1_analyze.py's per-rep table is non-empty (t0_monotonic_s /
#        rounds.jsonl attribution did not silently zero out, the exact
#        failure mode flagged in the task brief and already once observed in
#        s8_scaleup FINDINGS sec 4-4)
python3 workspace/engine-port/results/s8_frontier/e1_analyze.py \
  --dir workspace/engine-port/results/s8_frontier \
  --ttft-slo-ms 3000   # --itl-p95-slo-ms defaults to 60 now (sec 4.3.1);
                        # placeholder TTFT only, smoke run does not need the
                        # real siting rule satisfied, just non-empty output

# 2) capacity scan smoke, one arm one cell, few rates:
sbatch --export=ALL,ARM_IDS="3",CELLS="d44",RATES="2 8",PROBE_TARGET_S=10 \
  workspace/engine-port/results/s8_frontier/e1_capacity_scan.sbatch
```

Only after (1) and (2) both come back clean should the real capacity scan
(§4.2, all 5 cells, all arms) and then the staged main sweep (§3) be
submitted — and only with the user's go-ahead, per this task's constraints.

### 8.1 865832 result (2026-07-28) — pipeline validated, gate bug found and
    fixed, one open question remains

Job **865832 completed** (`boot_ok=1` both cells, `bench_serving` ran
cleanly, telemetry non-empty, analyzer ran end-to-end) — the pipeline
itself is validated. But the (pre-revision) pin gate FAILED both cells
while client TTFT differed 4.5× between them (matching `s8p_prefill`'s
independently-measured prefill-SM sensitivity in size and direction) —
investigated in full in §5.1: **Hypothesis A confirmed** (the gate's
population definition, `engaged = prefill_active OR decode_active`, wrongly
counted decode-only windows — which are SUPPOSED to show the unpartitioned
stream — as pin failures). `e1_pin_check.py`/`e1_analyze.py` are fixed
(§5); re-running the corrected code against the SAME 865832 telemetry (no
resubmission needed for the diagnosis itself) reproduces the numbers in
§5.1's table and confirms the fix.

**One open question §5.1/§9.7 could not resolve from 865832 alone**: does
`n_prefill_active` clear the new `MIN_N_PREFILL_ACTIVE=20` floor for
fast-prefill cells (D16/D24) at the main sweep's actual NP=150, or is this
harness structurally underpowered there regardless of NP? Authorized by
the coordinator ("스모크 규모(NP=20, rate 2–4)가 자체로 아티팩트인지도
판단해라... 더 큰 NP로 스모크를 재실행하는 게 답일 수 있다, 재제출은 해도
된다, 단 스모크 규모로만"), a follow-up diagnostic smoke at NP=150 (same 2
cells, REPS=1, same RATE_LO=2/RATE_HI=4 — still smoke-scale, no capacity
scan or main sweep involved) is the direct way to answer this:

```bash
sbatch --export=ALL,ARM_IDS="3",SMOKE_CELLS=1,RATE_LO=2,RATE_HI=4,NP=150,REPS=1 \
  workspace/engine-port/results/s8_frontier/e1_sweep.sbatch
```

**Submitted 2026-07-28: job `866066` (`e1sweep`, PENDING at submission
time)** — this diagnostic smoke, and only this, was submitted by this
agent per the coordinator's explicit authorization ("재제출은 해도 된다,
단 스모크 규모로만"). Still smoke-scale (1 arm, 2 cells, REPS=1) — not the
capacity scan, not the main sweep, neither of which has been submitted.

Read the result with the corrected gate/diagnostic code:
```bash
python3 workspace/engine-port/results/s8_frontier/e1_analyze.py \
  --dir workspace/engine-port/results/s8_frontier --ttft-slo-ms 3000
```
Check specifically: does `n_prefill_active` for d16 clear 20? If yes, NP
scaling alone resolves it and the main sweep can proceed with NP=150 as
planned. If no (as §9.7's proportional-scaling estimate of ~7 suggests is
likely), remedy 1 (longer/targeted prefill windows) or 3 (CI-based gate,
§9.7) needs to be applied before the main sweep, and this must be resolved
before submitting the real capacity scan for D16/D24-type cells.

### 8.2 866066 result (2026-07-29) — pin-gate fix confirmed working; a
    SECOND, unrelated bug found in the concurrency diagnostic and fixed
    (no new job submitted for this round — diagnosed entirely from existing
    telemetry, per coordinator instruction, GPU not needed)

Job **866066 completed**. The §5.1 pin-gate fix behaves exactly as
intended: d92 (n=118–242 prefill-active samples) PASSES cleanly (pin
0.968–1.000); d16 (n=10–12) is correctly flagged `UNDERPOWERED` — NP=20→150
(7.5×) scaled d16's count roughly proportionally (1→10-12) but did not
clear the 20-sample floor, confirming §9.7's prediction (see §9.7's
"remains a live, unresolved risk" for what this implies going forward).

**But the coordinator independently caught a second bug**, this time in
the `concurrent_frac_of_bench` mandatory diagnostic, via an arithmetic
cross-check against client-side TTFT/duration numbers (a ~29–240×
mismatch, differing by cell, ruling out a simple constant-scale error).
Full investigation in §5.2: **confirmed** — `concurrent_frac_of_bench` was
itself snapshot-COUNT-based (event-loop-iteration counts, not wall-clock
time), and understated true concurrency by ~16–26× at this job's cells.
Fixed by time-weighting (§5.2, `e1_pin_check.py`'s `compute_gates`).
**This required retracting §9.7's prior conclusion** that the D-axis might
be only weakly exercised (structurally resembling vector-1's ILL-POSED
finding) — that conclusion was itself downstream of the now-fixed bug; the
corrected numbers show 25–40% genuine concurrency for d92, a
well-exercised regime. See §9.7 for the retraction and the (separate, still
real) d16 `UNDERPOWERED` risk that survives this correction.

No new smoke was submitted for this round — both the pin-gate re-check and
the concurrency-diagnostic bug diagnosis were done entirely by re-reading
866066's existing telemetry with corrected code, exactly as the
coordinator's "GPU 불요" instruction anticipated.

### 8.3 866868 result (2026-07-29) — capacity-scan smoke: pipeline/gates
    confirmed working (pin PASS frac=0.955 n=22, time-weighted concurrency
    correctly separated from the count-based audit-trail number), but the
    scan's own `achieved_rps`/warmup/NP design had THREE further bugs, now
    fixed in `e1_capacity_scan.sbatch` (§4.2.1–4.2.3) — one more smoke
    submitted to validate the fix end-to-end

Job **866868 completed** (`e1_capacity_scan.sbatch`, T8, cell d44, rates
2/8, `PROBE_TARGET_S=10` override). Confirms the §5/§5.2 gate fixes
generalize beyond the two cells (d16/d92) they were diagnosed on: pin gate
PASS (frac=0.955, n=22, comfortably over the 20-floor), time-weighted
`concurrent_time_frac=0.0566` correctly reported as the mandatory
diagnostic with the count-based `concurrent_frac_count_based=0.0008`
clearly separated as audit-trail-only — exactly the intended behavior.

But the scan's OWN primary output (the rate/throughput numbers meant to
locate the capacity elbow) had three further, distinct problems, all
diagnosed from the existing 866868 artifacts without GPU (§4.2.1–4.2.3):
(1) `achieved_rps` tail-diluted by completion time, confounding "saturated"
with "probe too short"; (2) a cold-start TTFT reversal (rate=2 SLOWER than
rate=4) traced to no warmup discard; (3) `NP` too small at low rates
(rate-independent Poisson relative-variance floor, not fixed by
`PROBE_TARGET_S` alone). All three are fixed in the current
`e1_capacity_scan.sbatch` (`arrival_rps`/`completion_rps` split, warmup
discard, `NP_MIN` 15→45).

**Follow-up smoke submitted** (per coordinator's explicit authorization,
"스모크 규모 재제출은 허용한다") to validate the fix end-to-end with a live
run rather than only the retrospective recomputation:
```bash
sbatch --export=ALL,ARM_IDS="3",CELLS="d44",RATES="2 8",PROBE_TARGET_S=10 \
  workspace/engine-port/results/s8_frontier/e1_capacity_scan.sbatch
```
Same arm/cell/rates as 866868 for direct comparability; `NP_MIN` and
`WARMUP_S` now use the new code's defaults (45 and 3s respectively).
Checks per requirement (4): (i) TTFT trend monotonic non-decreasing with
rate, (ii) `arrival_rps` ≈ nominal rate at the low end (validity check —
`completion_rps` is not expected to equal nominal, that was never the
claim), (iii) `n_prefill_active` clears the 20-floor with more margin than
866868's borderline 22.

**Submitted 2026-07-29: job `867008` (`e1capscan`, PENDING at submission
time)** — this validation smoke, and only this, was submitted by this
agent per the coordinator's explicit authorization. Still smoke-scale
(1 arm, 1 cell, 2 rates) — not the full capacity scan (all cells/arms),
which remains held per the coordinator's "전 arm 본 스캔은 위 (4)가
깨끗해질 때까지 계속 보류" instruction.

### 8.4 867034 validation passed (coordinator-confirmed) → seed-policy and
    d16/low-rate methodology decisions implemented (§4.5/§4.6/§5.3/§9.8) →
    STAGE-1 full capacity scan submitted (T8, all 5 cells)

The coordinator confirmed a follow-on validation job (`867034`) passed:
TTFT monotonicity recovered (57→86ms / 60→70ms across the two rates
checked), the `arrival_rps`-led label works, and `CAPSCAN_SEED_DIVERGENCE`
correctly caught a real instability (rate≈8: arrival rate differed only
2.2% between the two seeds, but TTFT p50 differed 18.3% — flagged per
§4.2.1's >25%(*) threshold discussion, reported as informative even where
not flagged outright). Both open items from the prior round were then
resolved with directions, not left as options (§4.5 seed policy adopted
as pre-registered rule; §4.6/§5.3/§9.8 for the d16/episode-accounting
resolution, including the honest finding that D16 remains uncitable even
under the corrected method).

**Time budget for the requested full capacity scan (item C)**, computed
before submitting, per the coordinator's explicit request:
`RATES="1 2 3 4 6 8 12 16 24 32"` (extended upward from the prior 2-point
smoke specifically to find where TTFT/ITL clearly bend or errors appear,
per instruction — no a priori upper bound was fixed, this ladder is
STAGE 1 of the two-stage protocol, §4.6; if no elbow appears by rate=32
that fact will be reported and the ladder extended further, not silently
capped), `SEEDS="1 2"`, `PROBE_TARGET_S=8` (reduced from the default 20 to
bound `NP` growth at the high end of this wider ladder while still
respecting the `NP_MIN=45` floor at the low end, §4.2.3) → `NP` per rate =
{45,45,45,45,48,64,96,128,192,256}, **1,928 requests per cell (both
seeds), 9,640 total for 5 cells**. Exact wall-clock is unknown until
capacity is known (that is what this scan measures), but bounded by the
job's `--time=05:00:00` override (bumped from the script's default 3h given
the wider ladder and 5-cell scope); boot cost ~3-10 min/cell × 5 cells is
the other major component (§9.4).

**Submitted 2026-07-29: job `867231`** (`ARM_IDS="3"` = T8, all 5 cells,
the above ladder/seeds/PROBE_TARGET_S) — **stage 1 only**, per the
coordinator's explicit two-stage instruction ("T8부터 5셀 전부를 돌려 ...
2단계로 갈 거면 1단계 결과를 보고하고 멈춰라"): this validates the rate
range and the new episode gate at full campaign scale (150-256 requests/
probe vs the smokes' 20-150) before extending to the other 3 arms, which
remain **not submitted** pending this job's results and the user's
go-ahead.

## 9. Risks identified while building this harness

### 9.1 Off-cliff bands may differ by arm AND by cell within an arm (highest risk)

Covered in depth in §4.2. This is the single biggest threat to a clean
result: if `RATE_LO`/`RATE_HI` chosen from an averaged or D44-only scan
turn out to be on-cliff for D16 or D92 specifically, the D-comparison would
be contaminated by a capacity-cliff artifact rather than measuring the
genuine `prefill_SM+decode_SM<=108` trade-off — precisely the
`bench_noise_root_cause.md` failure mode (3% perturbation → 2× goodput
swing at a TTFT≈SLO boundary) this project already burned significant time
diagnosing once. The scan protocol (§4.2) is designed to catch this before
the main sweep runs, but it requires actually reading the per-cell scan
output arm-by-arm, not just picking one rate from the D44 row.

### 9.2 Admission-latch stale-True bug — considered, does NOT apply to this harness's design

`reports/r2_decoupling_review_2026-07-24.md` #4 confirms `r2_admission_limited`
can latch `True` with no clear path, permanently blocking prefill admission.
Traced the mechanism (`controller.py:80-94`, `183`, `207`, `222`): that
latch is only ever set by `CoarseGrainedController.stabilize()`
(`overload_streak >= 2`), which is used by the **generic/hybrid dynamic
policies** — `FixedPolicy.decide()` (what every E1 cell uses,
`PDMUX_R2_POLICY=fixed`) constructs its `SplitDecision` without an
`admission_limited` kwarg, and the dataclass default is `False`
(`controller.py:68`). **This specific confirmed bug cannot fire in E1.** The
task brief's example risk (a decode-starved `[92,16]`-style cell, i.e.
D16's `[92 prefill, 16 decode]`) is real in the sense that decode gets very
few SM there, but the *admission-latch* mechanism specifically is ruled out
by this code read — flagging the correction rather than silently dropping
the concern. The unrelated, still-live mechanism is
`prefill_admission_blocked` from genuine `shared_batch_capacity`/
`scheduler_admission_pending` backpressure (`dual_worker.py:590-599`),
which `e1_pin_check.py` reports as a diagnostic (flagged if >10% of engaged
samples) precisely so this residual risk is visible in the data rather than
assumed away.

### 9.3 Zamba2-7B-Instruct's native context cap is 4096, below this campaign's `--context-length`

`max_position_embeddings=4096` (read directly from the cached
`config.json`), while `e1_sweep.sbatch`/`e1_capacity_scan.sbatch` pass
`--context-length 4864` to every arm including Ha8, requiring
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (same override
`s8_scaleup`/`s8p_prefill` already use without incident). The mitigating
fact: `--sharegpt-context-len 3600` drops any conversation whose recorded
input+output exceeds 3600 tokens, so every ADMITTED request stays under
3600 — comfortably inside Zamba2's native 4096, not just inside the 4864
server cap. Residual risk: `--sharegpt-context-len` filters on the
dataset's *recorded* output length, not the model's actual generated length
under `ignore_eos`-free real sampling — if Ha8 (or any arm) tends to
generate noticeably longer completions than the dataset's recorded length
for the same prompts, a request could still exceed 3600 tokens combined and
hit context-length enforcement mid-generation. Watch server logs for
context-overflow errors per arm during the smoke test (§8) and the capacity
scan, especially for Ha8.

### 9.4 Boot-cost variance makes the time budget (§7) a wide range, not a point estimate

Observed boot deltas in `s8_scaleup` spanned ~4× (166s to 634s) across
cells of the SAME arm in the SAME job — plausibly first-vs-warm triton
kernel cache effects for a given SM-split geometry, not fully diagnosed
there either. This campaign's per-job `TRITON_CACHE_DIR` (§ below, carried
over from `s8p_prefill`) means each job starts with a **cold** cache, so
even "warm" boots within `s8_scaleup`'s shared-cache job may not transfer —
the actual first-cell boot of any E1 job could be on the slow end of the
observed range. This is a time-budget risk, not a correctness risk.

### 9.5 GPU queue contention

Per this task's briefing, the GPU queue is fairshare-limited. At harness-
build time `squeue -u ehmoon` showed exactly one job (`865738 s8psweep
PENDING`); after the §4.3 SLO revision it shows two
(`865738 s8psweep PENDING`, `865832 e1sweep PENDING` — the latter a
pipeline-smoke job the user submitted directly, §8). This agent has **not
submitted anything itself**. When the user is ready to submit the real
capacity scan / staged main sweep, be aware they will compete with
`s8p_prefill`'s continued runs for the same `submit<=4/run<=2` QOS slots on
`amd_a100nv_8`.

### 9.6 CLIFF HAZARD at the 60ms primary ITL SLO — M8 flagged from existing data

`FINDINGS_8B_2026-07-28.md` §2's SM16 batch-12-matched ITL p50 for M8 is
58.44ms — close enough to the 60ms primary SLO (§4.3.1) that its actual
p95 (not yet measured for this campaign's own workload/rate) could land on
either side of the boundary from ordinary rep-to-rep noise, i.e. M8/D16 is
a `CLIFF HAZARD` candidate by the §4.3.4 rule using only data already in
hand, before this campaign has run a single request. `e1_analyze.py`
computes this automatically (`cliff_hazard()`) once real data exists; this
subsection just records that the flag is EXPECTED to fire for M8, so its
absence (or presence) at scan time is itself informative about whether the
prior FINDINGS numbers transfer to this campaign's rate/workload. If M8/D16
is flagged, M8's headline must be reported as the full {50,60,80}ms ladder,
not the 60ms point value, per §4.3.4.

### 9.7 ★★★RETRACTED (2026-07-29): "concurrency tension resembling vector-1's
    ILL-POSED finding" was a MEASUREMENT ARTIFACT, not a real risk —
    superseded by §5.2

**This subsection previously argued** (from 865832's count-based
`concurrent_frac_of_bench`, ~0.0015–0.00002) that the D-axis might be only
weakly exercised even where the pin gate passes, structurally resembling
vector-1's ILL-POSED closure. **That argument is retracted.** §5.2 found
`concurrent_frac_of_bench` was itself count-based (event-loop-iteration
counts, not wall-clock time) and understated true concurrency by ~16–26×;
the corrected, time-weighted `concurrent_time_frac` at 866066 (NP=150) is
**25% (d92 LO) and 40% (d92 HI)** — a substantial, well-exercised overlap
regime, independently corroborated by a client-side TTFT-sum upper bound
(63%/255%-loose) computed directly from the same job's raw request data.
**There is no evidence of an off-cliff-vs-concurrency conflict for d92.**
Retracting this honestly (per the coordinator's explicit instruction) is
the correct call even though it reverses yesterday's own risk assessment —
see §5.2 for the full diagnosis, mechanism (file:line), and cross-checks.

**d16 remains genuinely low-concurrency** (`concurrent_time_frac` ≈0.01,
still small after the fix) — but this is now understood as a REAL,
mechanistically-explained asymmetry (92-SM prefill finishes so fast, a few
ms per request at this workload, that there is little wall-clock in which
to overlap with a concurrent decode step) rather than a red flag requiring
remediation. It is a property of that specific cell (largest prefill SM
allocation in the grid → fastest prefill → least overlap opportunity at a
given rate), not a symptom of the campaign's rate/NP choices being wrong.

**What DOES remain a live, unresolved risk from the original §9.7
(distinct from the retracted concurrency-tension framing): d16's
`n_prefill_active` sample count for the PRIMARY pin gate is still
UNDERPOWERED at NP=150 (n=10–12, still under the 20-sample floor)**, even
though NP scaled 7.5× from the original smoke's n=1. This is the SAME
issue §5.1 first flagged, now confirmed empirically at main-sweep NP rather
than merely estimated. The remediation candidates identified there remain
valid and are NOT retracted:
1. **Longer prompts widen the prefill window itself**, giving each
   individual prefill event more telemetry samples regardless of NP/rate
   (mirroring `s8p_prefill`'s explicit L-controlled probes — switch D16/D24
   specifically to a `random-ids` dataset with a deliberately long fixed
   input length instead of ShareGPT's natural mix, at least for a
   discriminating check).
2. **Increase NP further** — already tried 20→150 (7.5×), d16 went from
   n=1 to n=10–12 (roughly the expected ~7.5× scaling, consistent with
   §9.7's own prediction) but did not clear the 20-sample floor; a further
   increase (e.g. NP=300) is a plausible next step but cost scales
   linearly and this is the SLOWEST cell type to boot/measure per §7.
3. **A confidence-interval-based gate instead of a hard sample-count floor**
   (e.g. Wilson/Clopper-Pearson 95% lower bound ≥ pin threshold) would
   degrade more gracefully at small n, but does not fix the underlying
   small-population problem, only how it is reported.
4. Raising the telemetry sampling rate itself (`dual_worker_trace_every`,
   `multiplexing_mixin.py:90-95`) is an engine-porter-scoped change, out of
   reach for a harness-only build.

**Recommendation going forward**: D16/D24 (the largest-prefill-SM cells)
should be treated as at-risk for `UNDERPOWERED` pin-gate failures in the
main sweep specifically due to brief prefill windows, independent of the
(now-resolved) concurrency-diagnostic question. If the capacity scan or
main sweep reproduces this, remedy 1 (targeted longer-input probes for
those specific cells) is the most promising lever, since remedy 2 (more
NP) has already been shown insufficient by itself at a 7.5× step.

### 9.8 ★★★RESOLVED-AS-DIAGNOSIS, STILL HINGES ON `PDMUX_TRACE_FORCE_PREFILL`
    FOR CITEABILITY (2026-07-30, FOURTH revision, after §5.5's
    time-weighted gate) — D16 is genuinely 85% pinned by TIME, but fails
    the gate on LOW STATISTICAL POWER (n_episodes=8), not on zero or
    wrong-direction evidence

**This section's history is itself informative, once more**: "D16 is
UNDERPOWERED, pick a/b/c" (original) → "0 informative episodes, uncitable"
(§5.3, retracted) → "1 inside-bracket sample, showing the WRONG partition"
(§5.4, also superseded) → **§5.5 shows all three were answering a
per-request attribution question with a request-count-starved estimator,
when the actual gate question is an aggregate time-weighted one that the
SAME underlying telemetry answers much better**: D16's whole-file 866066
evidence is **12 prefill-active snapshots, time-weighted to 85% at the
target `P92`** (`pin_frac=0.847`), from **8 independent prefill episodes**.
The point estimate is good — 0.847 clears the 0.80 bar with room to spare —
but the honest episode-cluster-bootstrap lower95 on only 8 clusters is
**0.554**, which FAILS the 0.80 gate. This is a **materially different, and
more informative, failure mode** than any prior framing: not "no evidence,"
not "evidence points the wrong way," but **"the evidence we have is
plausibly good, we just don't have enough of it to say so at 95%
confidence."**

**Why this is not yet resolved, and what would resolve it**: the root
cause (D16's 92-SM prefill computation is fast enough that ordinary
telemetry subsampling rarely lands ANY sample while prefill is active) is
exactly what engine-porter's `PDMUX_TRACE_FORCE_PREFILL` patch targets
(§4.7/§12) — forcing an extra snapshot whenever prefill is in flight should
multiply D16's available prefill-active evidence roughly 30× per the
projected emission-rate numbers (§4.7/§12), which under the NOW-CORRECT
time-weighted estimator should directly translate into more independent
episodes and a tighter bootstrap CI (unlike the old episode-bracket
approach, where more snapshots only tightened brackets around a
request-count-limited population — see §5.5's "wrong question" diagnosis
for why this matters). **This has not been tested yet** —
`PDMUX_TRACE_FORCE_PREFILL` requires a live GPU run to produce new
telemetry, and per §4.7/§12/the coordinator's explicit instruction, no such
run happens before the observer-effect gate (job 867298) passes. So D16's
citeability is an OPEN EMPIRICAL QUESTION contingent on that gate and a
subsequent forced-sampling run — but it is now a QUANTIFIED open question
(need enough additional independent prefill episodes to shrink the
bootstrap CI from ±0.29 to within 0.05 of the 0.847 point estimate), not a
qualitative one.

**What this means for the main sweep, concretely**:
1. Do not decide D16's fate (cite / exclude / workload-specific remedy)
   until a real `PDMUX_TRACE_FORCE_PREFILL=1` run's D16 telemetry has
   actually been checked against `compute_time_weighted_pin_gate`. If that
   run still gives too few independent prefill episodes to clear the
   bootstrap bound, the old menu's remaining live options
   (workload-specific shorter inter-arrival time for D16 only to produce
   more prefill episodes per unit wall-clock, or accept D16 as
   as-run-only/excluded) apply exactly as previously described.
2. **Independently of whether the gate ultimately passes**, §5.5's
   requirements 3-4 apply regardless: D16's realized-partition split
   (currently 85% target / 15% auto-partition fallback, whole-file 866066)
   MUST be reported as a first-class per-cell, per-rep, per-phase number
   in the main sweep's results, described as "target D=16, realized
   target 85%/auto-partition 15%" (or whatever the actual PDMUX_TRACE_FORCE_PREFILL=1
   run measures), never as bare "D=16" — this is a real, asymmetric,
   frontier-position-dependent property of the runtime's own auto-partition
   fallback policy (§5.5 req. 4), not an artifact that forced-sampling
   removes; forced sampling only lets us MEASURE it with enough power to
   gate on it, it does not change the underlying 85/15 split itself.

## 10. Artifact mapping

- design: this file.
- configs: `pdmux_e1_d{16,24,44,54,92}.yml`.
- harness: `e1_sweep.sbatch` (main grid), `e1_capacity_scan.sbatch`
  (pre-scan). Both derived from `s8_scaleup/s8_sweep.sbatch`'s skeleton
  (bootstrap sync, fail-fast PIPESTATUS, clock sampling) +
  `s8p_prefill/s8p_sweep.sbatch`'s per-job `TRITON_CACHE_DIR` (carried over
  verbatim, triton cache race gotcha) + `slo_sched/sharegpt_vary_bench.sbatch`
  /`g2_0_raconf`'s open-loop ShareGPT LO/HI rate-alternation pattern (client
  = `sglang.bench_serving`, no new client script needed — its
  `--output-details` already emits everything the analyzer needs; the only
  new instrumentation is the `t0/t1` monotonic bracketing this harness
  writes per round for telemetry-window gate attribution, since
  `bench_serving` itself has no such hook).
- gate: `e1_pin_check.py` (dual-axis merge of
  `s8_scaleup/realized_pin_check.py` + `s8p_prefill/prefill_pin_check.py`,
  `compute_gates`/`gate_verdict` importable by the analyzer).
- analysis: `e1_analyze.py` (rep-wise conjunctive goodput, paired bootstrap
  CI, pre-registered decision rule).
- manifest: `runtime_source_manifest.sha256`, written by
  `sync_engine_tree.sh` on first boot of either sbatch (not present yet —
  no job has run).
- results (not yet produced): `e1_<arm>_<job>_result.txt`,
  `e1_<arm>_<cell>_<job>_telemetry.jsonl`,
  `e1_<arm>_<cell>_<job>_rounds.jsonl`,
  `e1_<arm>_<cell>_<job>_rep<r>_{lo,hi}.jsonl`,
  `e1_<arm>_<cell>_<job>_srv.log`, `e1_<arm>_<cell>_<job>_clocks.csv`;
  capacity scan: `e1cap_<arm>_<job>_result.txt`,
  `e1cap_<arm>_<cell>_<job>_telemetry.jsonl`,
  `e1cap_<arm>_<cell>_<job>_r<rate>.jsonl`.

## 11. Time-budget impact of the 2026-07-28 SLO revision (§4.3)

**No change to §7's estimates.** The revision is entirely a change in how
already-collected TTFT/ITL distributions are read (fixed 60ms primary +
{50,60,80}ms ladder instead of a scan-time-chosen point value, plus the
`CLIFF HAZARD`/`ITL-NONBINDING`/TTFT-margin flags) — it does not change:
- the workload shape (still ShareGPT, open-loop, LO/HI rate alternation),
- the RATES grid the capacity scan probes (`1 2 3 4 6 8 12 16`, unchanged —
  it already collected the full ITL-p95 distribution per rate, this
  revision only changes what threshold that distribution gets compared
  against),
- `NP`/`REPS`/the D grid/the arm roster.

The only added cost is **analyst time**, not GPU time: before finalizing
`RATE_LO`/`RATE_HI`, the scan output must now also be run through
`e1_analyze.py --capscan-dir ... --capscan-rate <candidate>` to check the
ladder/margin/hazard flags at the candidate site (§4.2), rather than just
reading the TTFT elbow off a plot. This is a few seconds of local
computation per candidate rate, not a rerun of anything on the GPU.

One second-order, genuinely possible (not yet known) effect on wall-clock:
if the `CLIFF HAZARD` or TTFT-margin-siting check reveals that **no
rate/SLO combination satisfies both the ITL ladder discrimination
requirement and the TTFT siting rule simultaneously** for a given arm (§4.3
flags this as a valid "ill-posed for this grid" outcome rather than forcing
a value), that arm's stage would need to be either reported as ill-posed
(zero additional GPU time — a documented negative scoping result, same
spirit as `g2_0`'s "vector-1 ill-posed" precedent) or would require
revisiting the workload's context-length/output-length parameters (e.g. a
shorter or longer ShareGPT cap to shift where TTFT sits) — which WOULD add
GPU time, but only if the siting check fails, and only for the specific arm
it fails for. This is a risk to watch during the capacity-scan stage, not a
committed cost.

## 12. `PDMUX_TRACE_FORCE_PREFILL` — engine-side fix for the d16 telemetry
   blind spot, and what it does to count-based statistics
   (added 2026-07-29 by engine-porter; engine patch only, harness unchanged)

**The blind spot.** `runtime_snapshot` emission is a pure COUNT subsample of
`_dual_worker_sync` calls (`multiplexing_mixin.py`: emit when
`trace_count == 1 or trace_count % PDMUX_DUAL_WORKER_TRACE_EVERY == 0`,
default 32). The PD-mux event loop keeps spinning when idle at several
kHz, while a prefill-in-flight iteration is a handful of (slow) sync calls,
so the sampled population is dominated by idle iterations. Measured on this
campaign's own telemetry: E1/T8/d44 prefill was active **4.9% of wall time
but 0.07% of sampled snapshots** (a ~70× under-representation), and d16
(`[92,16]`, the fastest prefill in the grid) produced 12 prefill-active
snapshots in 181s — which is why §9.8's episode gate found 138/138 brackets
vacuous. The bias runs in the direction of the experiment: the more SM
prefill gets, the less observable it is, so the frontier's decode-starved
end is the least verifiable end.

**The fix (engine).** With `PDMUX_TRACE_FORCE_PREFILL=1`, a sync whose
`split_prefill_batch is not None` emits regardless of the subsample grid.
Default is **OFF**, so past campaigns and any run that does not opt in keep
writing byte-identical records (the extra `trace_forced` field is written
only when the flag is on). The scheduled grid itself is untouched in both
modes — `dual_worker_trace_count` still advances once per sync — so forced
records are strictly ADDITIONAL, never a shift or a replacement.
(Bookkeeping: this patch shifts the `multiplexing_mixin.py` line numbers
cited in §5.3/§9.8 — the emission gate is now at `:399-437` and the flag is
read at `:97-106`. Installed hash is in this directory's
`runtime_source_manifest.sha256`; the flag is delivered through
`src/multiplex/multiplexing_mixin.py` + `sync_engine_tree.sh`, not through a
separate `src/patches/*.patch`, because that file is one of the tracked
sources the sync script installs wholesale.)

**★ What this does to count-based statistics.** In force mode the emitted
population is deliberately OVER-sampled on prefill-in-flight syncs.
Therefore any statistic of the form "fraction of snapshots" is biased
toward prefill and is **not comparable across the flag**:
- `concurrent_frac_count_based` / `prefill_active_frac_count_based`
  (`e1_pin_check.py`, already retained for audit trail ONLY) — will read
  much higher in force mode, for a reason that has nothing to do with the
  run's actual behaviour.
- `admission_blocked_frac` (same file) is also count-based, and is biased
  the OTHER way: by construction a blocked sample has
  `prefill_active_batch_size == 0`, so every forced record lands in the
  denominator and never the numerator, **diluting** the reported blocked
  fraction. If that number is ever read across a flag boundary, recompute
  it over `trace_forced != true` records only.
- Recovery recipe: filtering to `trace_forced != true` reproduces the
  legacy population EXACTLY (it is precisely `sample_index == 1 or
  sample_index % trace_every == 0`), so any count statistic can be made
  comparable again after the fact. Nothing is lost, but nothing is
  automatically comparable either.

**What is NOT affected.** The two things this campaign actually gates on:
- **Time-weighted** diagnostics (`compute_concurrency_diagnostic` weights
  each snapshot by the gap to the next one) are unbiased under denser
  sampling — a finer partition only makes the Riemann sum of the same
  occupancy function more accurate. Denser sampling can only move the
  time-weighted numbers toward the truth, never away from it.
- **Episode-based** gates (`compute_episode_gate`) count requests, not
  snapshots; extra snapshots only tighten the `[ii, jj]` bracket around
  each episode.
This is why the campaign's 2026-07-29 move off count-based accounting (§5,
§5.3) is what makes this patch safe to use at all.

**Cost and its asymmetry (must be respected when arms are compared).** The
added emission volume is bounded by the number of prefill-in-flight sync
calls, which is a per-cell property. Projected from this campaign's own
pre-patch telemetry (prefill-active snapshots × `trace_every`):
`d16` +1.2% emitted records (~2/s), `d44` +2.2% (~6/s), `d92` +28.5%
(~44/s). Measured cost of one emission on the scheduler thread (CPU
microbench, batch 8–48): median 33–45 µs, mean 47–63 µs, i.e. ≲0.02% of
scheduler wall time at d16 and ≲0.3% at d92. The point to keep in mind is
not the magnitude but the **shape**: the observer load is smallest at the
prefill-rich cells and largest at the prefill-poor ones, i.e. it varies
along the very axis the frontier compares. Keep the flag set to the SAME
value for every cell of a comparison, and treat the measured bound (see
`results/e1_traceforce/`) as the floor of resolvable effects, not as
"zero".
