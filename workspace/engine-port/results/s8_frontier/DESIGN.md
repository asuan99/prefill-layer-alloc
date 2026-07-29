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

None of this changes the workload/rate grid (§4.2) or adds GPU time — it is
purely how the already-collected TTFT/ITL distributions are read, so it has
no effect on the time budget (§7 unchanged; see also §11 for the explicit
before/after note the coordinator asked for).

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

## 5. Pre-registered gates (telemetry-based, auto-judged, cite-blocking) —
   ★★REVISED 2026-07-28 (coordinator directive, after smoke job 865832
   exposed a gate-definition bug — full investigation in §5.1)

Implemented in `e1_pin_check.py` (`compute_gates`/`gate_verdict`, imported
directly by `e1_analyze.py` — the sbatch's own stdout gate lines and the
analyzer's citeability filter are **the same code path**, not two
re-implementations that could silently drift):

1. **PRIMARY, cite-blocking gate — realized partition pin ≥ 0.80, among
   PREFILL-ACTIVE samples only.** Among `runtime_snapshot` samples with
   `prefill_active_batch_size > 0` (decode state irrelevant to this
   condition — mirrors `s8p_prefill/prefill_pin_check.py:51,56-58,67`
   exactly), the fraction whose REALIZED `(prefill_sms, decode_sms)` pair
   equals the cell's target `(108-D, D)` must be ≥ `min_pin_frac` (0.80)
   **and** the prefill-active population itself must have
   ≥ `min_n_prefill_active` samples (default 20 — added 2026-07-28; a tiny
   population, e.g. n=1, can trivially satisfy a fraction threshold at
   either 0% or 100% without being informative, exactly what happened to
   the T8/d16 smoke cell). A cell/rep failing on sample size fails with the
   distinct reason `UNDERPOWERED`, never conflated with `wrong SM`. Judged
   on realized telemetry fields (`dual_worker.py:608/622-623`), never on
   policy target/`controller_decision` — that distinction is exactly what
   made Stage 0's D108 anchor silently invalid
   (`results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md`).
2. **RETIRED as a hard gate (2026-07-28) — "partition activity rate ≥
   0.60."** The original definition conditioned on `engaged =
   prefill_active_batch_size>0 OR decode_running_batch_size>0`, which
   §5.1 found conflates two different things: genuine reversion-on-idle
   (both queues empty) and CORRECT, BY-DESIGN full-GPU-to-decode behavior
   whenever decode has work but prefill does not
   (`multiplexing_mixin.py`'s stream-selection: `elif not
   self.running_batch.is_empty(): set_current_stream_idx(<last group>)` —
   decode-only windows are SUPPOSED to show the unpartitioned `(0,108)`
   stream, not the target split; that is not a failure to fix). This gate
   is retired rather than "loosened" per the coordinator's explicit
   instruction not to relax it into meaninglessness — its replacement is
   the mandatory diagnostic below, which measures the thing that actually
   matters and does not pretend to have a validated pass/fail threshold.

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
Also reported for context: `prefill_active_time_frac` (how much wall-clock
time has prefill in flight at all, time-weighted) and
`co_resident_frac_of_prefill_active` (conditional on the PRIMARY
population — an event-count proportion, deliberately NOT time-weighted,
see §5.2 for why that is correct for THIS specific quantity). The retired
count-based numbers are still computed and printed as
`*_frac_count_based`, for audit-trail only — **do not use them to argue
low concurrency**, see §5.2.

The PRIMARY gate is evaluated **per rep, per phase** (LO window and HI
window separately, using the `[t0,t1]` monotonic brackets
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

### 9.8 ★NEW 2026-07-29 — D16's `n_prefill_active` floor: decision needed
    before the main sweep (options only, no default chosen here — user's
    call, per coordinator instruction)

**Status**: d16's PRIMARY pin gate remains `UNDERPOWERED` at NP=150
(n=10–12 vs floor 20, 866066), confirmed a real, physical property of that
cell (92-SM prefill windows are simply too brief to be caught reliably by
the event-loop-iteration-cadence telemetry, §5.2), not a harness bug. Fixing
this AFTER seeing main-sweep data would be a pre-registration violation
(the coordinator's own framing) — so the choice must be made now, before
the main sweep, even though none of the options is free of tradeoffs.
Three candidates, presented without a recommendation (user's decision):

**(a) Widen the window for D16/D24 specifically** (longer prompts and/or a
higher rate for just those cells, e.g. switching to a `random-ids` dataset
with a deliberately long fixed input length, mirroring `s8p_prefill`'s
L-controlled probe design). *Pro*: attacks the root cause (brief prefill
windows) directly, keeps the gate/floor as-is so D16 is judged by the SAME
standard as every other cell. *Con*: breaks the "single-variable" workload
design (§4.1 — every OTHER cell would still use natural ShareGPT-length
prefill, only D16/D24 would use an artificially lengthened one), so a
result for D16 would carry a workload-composition caveat other cells don't
have; requires new config/prompt plumbing not yet built.

**(b) Redefine the sample-size floor on a time-weighted basis** instead of
a raw prefill-active COUNT (e.g., require the time-weighted
`prefill_active_time_frac` × window duration to exceed some minimum
WALL-CLOCK seconds of prefill-active time, rather than `n_prefill_active
>= 20`). *Pro*: directly measures the thing that actually matters
(how much genuine wall-clock evidence supports the pin_frac estimate) and
would treat a cell with FEW but LONG prefill events as adequately powered
even if the raw event count is low — which is arguably the more honest
statistic given §5.2 established that events, not counts, are the right
unit for time-based claims (though pin_frac itself is deliberately an
EVENT-conditional proportion, not a time-weighted one, per §5.2's discussion
of why that is the correct choice for THAT specific quantity — this option
would need its own careful justification for why a TIME threshold is
appropriate for a COUNT-based proportion's power, not just an assertion).
*Con*: introduces a second, harder-to-intuit threshold family (seconds
instead of a plain sample count) and does not actually fix the low
information content of a genuinely tiny population — a handful of long
events is still a handful of independent observations for the purpose of
estimating a PROPORTION (event count, not their duration, governs the
proportion's own statistical power) — this option's honesty benefit is
more about reporting than about actually resolving underpowering.

**(c) Report D16 as `UNDERPOWERED` and drop it from the cited grid**,
keeping D∈{24,44,54,92} as the main sweep's citeable set, with D16 kept
"as-run" only (per the existing dropped-rep convention, §5) — extending the
same treatment already applied automatically to any individual failing
rep to the entire cell. *Pro*: zero additional harness work, fully
consistent with the existing gate philosophy (discard, don't paper over),
and does not compromise any OTHER cell's result. *Con*: D16 is the
decode-most-starved extreme of the grid (§0/§2) — dropping it narrows the
frontier's span exactly at the end where C2's decode-SM lever is
theoretically most binding, potentially the single most informative point
for testing tension A; losing it weakens the campaign's ability to answer
its own motivating question at one end of the trade-off space.

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
