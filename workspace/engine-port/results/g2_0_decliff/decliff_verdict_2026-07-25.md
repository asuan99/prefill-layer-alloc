# G2.0 de-cliff stage-1 — vector-1 (disjoint conflict-regime) verdict (2026-07-25)

Experiment run: 2026-07-24/25 (jobs 863880–863948, `g2_0_decliff_bench.sbatch`,
two-tier driver `.campaign_drv.sh`). Analysis: result-analyst (canonical scoring
from raw per-request jsonl, constrained from writing this report directly).
Adversarial re-check: claims-auditor. Written up here by doc-steward per the
analyst's approved conclusions below — this file records, it does not re-derive.

Predecessors: [`../g2_0_full/disjoint_verdict_2026-07-24.md`](../g2_0_full/disjoint_verdict_2026-07-24.md)
(first pass, razor-thin real disjoint on the TTFT cliff), [`../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md)
(hardening sweep, sign-flip on the same cliff ⇒ **ILL-POSED at rA5**, de-cliff
resweep recommended as the only way to get a real verdict). This document is
that de-cliff resweep, stage 1 (capacity/de-cliff only).

## Design (stage 1 = capacity, not feasibility)

Two-tier driver, `d16`/`d44`/`d54` × `rA ∈ {2, 3, 3.5, 4}` (Phase A arrival
rate), `rB=4` fixed, `OB=512` fixed (matches `g2_0_full`, not the `g2_0_hard`
axis-1 OB1024 percentile-window artifact).

- **Tier 1a (coarse locate, n=3/cell, 36 jobs)**: per (rate, split) cell, mean
  and stdev of Phase-A TTFT p90 across 3 reps. A rate is "de-cliffed" iff
  **all three splits** have mean p90 < 2.0s **and** stdev p90 < 0.15s (crude
  proxy for "no bimodal collapse", gate #6). Selection picked the **highest**
  de-cliffed rate.
- **Tier 1b (confirm, n=3 more/cell at the chosen rate, 9 jobs)**: brings the
  chosen (rate, split) cells to n=6 total.

Selection result (`STAGE1A_SELECTION.txt`, `DECLIFF_P90_MAX=2.0
DECLIFF_STD_MAX=0.15`):

```
  rate   mode   n  mean_p90  std_p90 vals
     2    d16   3     0.483    0.075 [0.44, 0.44, 0.57]
     2    d44   3     0.567    0.012 [0.58, 0.56, 0.56]
     2    d54   3     0.740    0.095 [0.65, 0.84, 0.73]
  -> rate=2 worst_split_mean_p90=0.740 worst_split_std_p90=0.095 declift=True
     3    d16   3     1.177    0.947 [0.63, 2.27, 0.63]
     3    d44   3     1.483    1.166 [0.81, 2.83, 0.81]
     3    d54   3     2.163    1.080 [2.87, 0.92, 2.7]
  -> rate=3 worst_split_mean_p90=2.163 worst_split_std_p90=1.166 declift=False
   3.5    d16   3     1.257    1.070 [0.58, 2.49, 0.7]
   3.5    d44   3     1.490    1.083 [0.83, 2.74, 0.9]
   3.5    d54   3     2.423    1.287 [0.94, 3.24, 3.09]
  -> rate=3.5 worst_split_mean_p90=2.423 worst_split_std_p90=1.287 declift=False
     4    d16   3     1.597    1.381 [0.64, 0.97, 3.18]
     4    d44   3     1.800    1.524 [3.56, 0.92, 0.92]
     4    d54   3     2.283    1.625 [1.32, 1.37, 4.16]
  -> rate=4 worst_split_mean_p90=2.283 worst_split_std_p90=1.625 declift=False

CHOSEN_RATE=2
```

**Only `rA=2` de-cliffs cleanly.** From `rA=3` up, all three splits are
already bimodal (per-rep p90 swings from ~0.6s to ~2.9s at n=3) — this
reproduces, rather than resolves, the g2_0_full/g2_0_hard cliff. `rA2` is the
only tested rate with a legitimate off-cliff read; it was confirmed to n=6
per split (tier 1b).

## VERDICT (result-analyst canonical scoring + claims-auditor adversarial check)

> **de-cliff stage-1 (`rA{2,3,3.5,4} × {d16,d44,d54}`, `rA2` n=6) complete: no
> robust off-cliff disjoint found = PLAUSIBLE closure, NOT CONFIRMED.**
>
> - At the one clean off-cliff point, `rA2`, there is no disjoint: static
>   `d54` covers both phases simultaneously (Phase-A `frac_good` 0.974±,
>   TTFT p99 ≤ 1028ms / Phase-B ITL-p95 42.3±0.10ms, `frac_good` 1.0, robust
>   under output-length-invariant metrics too). `feasible-A={d16,d44,d54} ∩
>   feasible-B={d54} = {d54} ≠ ∅`.
> - Candidate disjoint only reappears at `rate ≥ 3.5` (all bimodal-cliff,
>   n=3) — i.e. exactly the same cliff regime `g2_0_full`/`g2_0_hard` already
>   flagged as ill-posed, not a newly resolved regime.
> - ★**Not "closed" (claims-auditor rebuttal):**
>   (a) the split→TTFT gradient is **still alive off-cliff**: `rA2` p99
>       `d16` 712±37ms vs `d54` 1028±101ms, `t≈7.2`, monotone across splits.
>       This does not rule out a path where, as rate rises, `d54` crosses the
>       cliff *first* and produces a real disjoint before `d16`/`d44` do.
>   (b) the onset of `d54` exclusion (somewhere ~rate 3.0–3.5) sits exactly
>       in the **unmeasured, n=3, bimodal transition band** — argument alone
>       cannot close this, it needs direct measurement.
>   (c) the load-bearing equating of "`binding-A` ⟺ `on-cliff`" conflates
>       "`d54` excluded" with "`d54` cliff-flicker" — this equivalence is an
>       **unproven empirical claim**, not established by this stage-1 data.
>
> - **Pre-registered stage-2 (required before any direction is adopted)**:
>   narrow `rA{2.25, 2.5, 2.75, 3.0, 3.25} × {d16, d44, d54(+d34)}`, n≥6
>   confirmatory sweep. Falsification target: some rate at which `d54` is
>   robustly `<0.7` (p90>3s, unimodal) **and simultaneously** `d44`/`d16` are
>   robustly `≥0.95` and off-cliff (p90<2s) ⇒ disjoint real ⇒ **vector-1
>   REOPEN**. (In progress — not yet run.)
>
> - **Scope statement (required)**: this result is scoped to {Zamba2-2.7B,
>   ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4, triton
>   attn+mamba, `disable-radix-cache`, cudagraph-ON, A100 108-SM
>   green-context pdmux, SLO = TTFT 3s ∧ per-request ITL-p95 50ms,
>   inter-phase-drained sequential 2-phase}. Does **not** generalize to
>   "hybrid models have no disjoint conflict regime".
>
> - **Drain caveat**: this closure is observed under an entanglement-suppressed
>   (drained) condition — it is a necessary-condition bound, not a
>   demonstration on a hot varying trace (Claim C entanglement).
>
> - ★**§1-20 firewall maintained**: temporal disjoint (a single static
>   covering both phases across time) and spatial coupling-tax (92+24=116>108,
>   disaggregation +16% headroom) are separate axes. "a single static
>   suffices ⟹ no coupling tax" does **not** follow and must not leak into
>   §1-20.

## Supporting tables (rA2, n=6: tier-1a rep1–3 + tier-1b rep4–6)

Phase A (in2048/o32 @ rA2) canonical `TTFTs p50/p90/p95/p99` (s), per-rep,
from `G20DC_PCT ... A` lines (raw per-request jsonl: `g20dcA_<split>_<rep>_rA2B4_OB512_<jobid>.jsonl`):

| split | rep1 p99 | rep2 p99 | rep3 p99 | rep4 p99 | rep5 p99 | rep6 p99 | jobs (rep1..6) |
|---|---|---|---|---|---|---|---|
| d16 | 0.73 | 0.82 | 0.72 | 0.87 | 0.73 | 0.75 | 863887,863915,863935,863944,863946,863948 |
| d44 | 0.86 | 1.09 | 1.05 | 0.92 | 1.01 | — | 863926,863922,863929,863941,863947,863938 |
| d54 | 0.99 | 1.12 | 1.27 | 1.00 | 0.97 | 1.12 | 863919,863913,863916,863939,863945,863940 |

(`d16` p99 mean 0.770s; `d54` p99 mean ~1.078s in this quick per-rep-log read
— result-analyst's canonical `frac_good`/paired-CI numbers in the boxed
verdict above supersede this table for any performance judgment; this table
is provenance/spot-check only, not the scoring source of truth.)

Phase B (in2048/o512 @ rB4, held fixed across all rA) `d54` ITL: printed
`G20DC_PCT ... B` per-rep ITL p95 values 30.9/30.8/30.8/30.7/32.1/30.7ms —
consistent with "clean, saturated, `frac_good≈1.0`" at `d54`; canonical
`ITL-p95 42.3±0.10ms` in the boxed verdict is result-analyst's per-request
(not per-rep-aggregate) computation from the same raw jsonl
(`g20dcB_d54_*_rA2B4_OB512_*.jsonl`) and is the number of record.

Raw data (all rA2 rA3 rA3.5 rA4, all splits, all reps): `g20dcA_*.jsonl`
(Phase A), `g20dcB_*.jsonl` (Phase B), stdout/stderr `g20dc_<jobid>.{out,err}`,
selection log `STAGE1A_SELECTION.txt`, submission log `driver.log`, tier-1a/1b
job-id manifest `DECLIFF_STAGE1_DONE`.

## Next: stage-2 confirmatory sweep (recipe, not yet run)

Per the pre-registered falsification target above:

1. `rA ∈ {2.25, 2.5, 2.75, 3.0, 3.25}` — narrow the gap between the one clean
   off-cliff point (`rA2`) and the first bimodal-cliff point (`rA3`), instead
   of jumping straight to a cliff-adjacent rate.
2. `{d16, d44, d54}` plus `d34` as an intermediate control point.
3. `n≥6` per cell from the start (not a coarse-then-confirm two-tier design) —
   stage-1's own bimodal cells (`rA≥3`) show n=3 is not enough to tell
   "genuinely bimodal" from "one unlucky rep".
4. Score with the same canonical `TTFT≤3000ms ∧ per-request ITL-p95≤50ms`
   definition; also carry the output-length-invariant ITL check (tail-only
   percentile) that stage-1 used to keep `d54`'s Phase-B feasibility number
   honest, per the `g2_0_hard` percentile-window-artifact lesson.
5. Do not re-score stage-1 data under a new rate grid — run live (gate #8
   analog, per `hardened_disjoint_verdict_2026-07-25.md` §5 and CONSENSUS §3).

## Provenance

- sbatch: `g2_0_decliff_bench.sbatch`, driver: `.campaign_drv.sh`.
- jobs: tier-1a 863880–863937 (36 jobs, `rA∈{2,3,3.5,4}×{d16,d44,d54}×3reps`),
  tier-1b 863938–863948 (9 jobs, `rA=2×{d16,d44,d54}×3 more reps`).
  Partition `amd_a100nv_8`, Zamba2-2.7B, ctx4096, cudagraph ON,
  `--attention-backend triton --disable-radix-cache`.
- sentinels: `STAGE1A_SELECTION.txt` (rate selection table), `driver.log`
  (submission/drain timeline), `DECLIFF_STAGE1_DONE` (final marker,
  `chosen_rate:2`).
- scoring: canonical `TTFT≤3000ms AND per-request token-ITL p95≤50ms`,
  matching `pdmux_eval.analyze.summarize_requests` percentile definition.
- predecessor verdicts: [`../g2_0_full/disjoint_verdict_2026-07-24.md`](../g2_0_full/disjoint_verdict_2026-07-24.md),
  [`../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](../g2_0_hard/hardened_disjoint_verdict_2026-07-25.md).
- canon pointers updated alongside this file (2026-07-25): `reports/CONSENSUS.md`
  §5-8(c), `PROJECT_STATUS.md` "HE0-reopen 벡터1", `reports/paper/EXPERIMENT_ROADMAP.md`
  벡터1 stub.
