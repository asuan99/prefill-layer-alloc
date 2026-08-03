# Pre-registration — §0 axis check (GPU 0, offline re-analysis)

**Written before any output was produced.** Target = `DESIGN.md` §4.3.13 §0,
the current top open item. Registered 2026-08-03 (fourth continuation
session). Script: `s0_axis_check.py`.

---

## 1. Why this exists (and what changed the framing)

§4.3.13 §0 records a 2.6x disagreement between two direct measurements of
"decode at 16 SM" per-token ITL p50:

| Source | Condition | value |
|---|---|---|
| C2 job 865493, SPLIT population, **batch bin 5** | prefill 16, decode 16 | 28.79 ms |
| 872077 (E1 grid) d16, **decode batch ~4.5** | prefill 92, decode 16 | 11.09 ms |

Two facts about that table, established before writing this file:

1. **Decode batch was already matched** (bin 5 vs ~4.5). So "the arms differ
   in decode batch size" is **not** an available third explanation — the
   auditor already controlled that axis. Any pre-registration that framed
   this check as "screen out batch mismatch" would be re-running a control
   that already exists.
2. **The E1-side number 11.09 ms has no stored artifact.** `grep` over the
   whole of `results/` finds it only in `DESIGN.md` prose; the C2-side 28.79
   is reproducible from `c2_anchor.py` step [4] (bin-conditional), but
   whether 11.09 is the **bin-5 conditional** or the **pooled p50 over the
   whole SPLIT population** (whose *mean* batch is 4.5, but whose batch
   distribution is not a point mass) is **not recorded anywhere**. A pooled
   quantile over a batch distribution and a bin-conditional quantile are
   different estimands (methodology gate #4/#5 — aggregation unit must be
   fixed and named).

⇒ **This check's purpose is therefore not to test a new hypothesis.** It is
to put both sides of the top open item on **one recorded, bin-conditional
estimand computed by one script**, so that the magnitude the open item
asserts is reproducible. Canon currently rests a top-priority open item on a
number whose aggregation unit is unrecorded.

## 2. Units, fixed now (methodology gates #4/#5)

- **unit**: one ITL interval = one emitted token. No per-request inner
  aggregation. (Identical to §4.3.10(1) and to `c2_anchor.py`.)
- **label**: REALIZED partition, duration-weighted over the interval —
  `split_frac(a,b) = time at decode_sms == D / (b-a)`; SPLIT iff
  `>= 0.90`. Same constant, same rule, on both sides. Never target.
- **bin**: mean in-interval `decode_running_batch_size`, duration-weighted,
  rounded to int — **identical to `c2_anchor.py` step [4]**, which is what
  produced "bin 5". Applied to **both** sides (this is the change).
- **statistic**: per-bin **p50** primary, **p95** secondary, direct
  quantiles of the per-token population, same interpolation rule as
  `c2_anchor.pct`.
  ⚠️ The sticky campaign's declared primary is `p95(SPLIT)`
  (§4.3.12(b)) and **this file does not change that.** p50 is primary *here*
  only because §0's own headline is stated in p50; using p95 would not
  reproduce the disputed number.
- **minimum cell size**: 50 tokens per (job, bin) — same floor as
  `c2_anchor.py` step [4]. Bins below it are printed as UNDERPOWERED, never
  silently dropped.
- **replicate**: C2 = rep index within one boot per cell (**pseudo-
  replication**, understates between-boot variance — recorded, not hidden);
  E1 = block, 8 boots. **No CI is computed that treats C2 reps as
  independent boots.** Cross-job comparison is reported as a ratio of
  pooled per-token quantiles with n on both sides, not as a tested contrast.

## 3. Arms and cells

Primary: **T8** (pure Transformer, the positive control) cell **d16**, both
jobs. Secondary, reported if present: **Ha8** d16. No other cell is used —
this check is about reproducing one number, not a sweep.

## 4. Equivalence gates (must pass before any number is read)

1. **E1 side**: the labelling loop in `s0_axis_check.py` reproduces
   `m3_conditional.label_probe`'s `(itl_ms, split_frac)` records **exactly**
   for the same probe (the only difference permitted is the added mean-batch
   field). Fails => the numbers are not comparable to the audited estimator
   and are void.
2. **C2 side**: the script reproduces `c2_anchor.py` step [4]'s T8 d16
   bin-5 SPLIT p50 = **28.79 ms** to within 0.05 ms. Fails => the C2
   reconstruction differs from the audited one and the numbers are void.
3. **Alignment**: E1 side reports `align_r` per probe; `< 0.95` is
   **flag-only, never a silent exclusion** (§4.3.10(1) — excluding them
   made the contrast *larger*, so exclusion is the self-serving direction).

## 5. Pre-registered readouts

- **R1** — E1 T8 d16 **bin-5** SPLIT p50 is within ±15% of 11.09 ms
  => the §0 table was already bin-conditional on both sides; §0 stands
  **as written** and this check adds only a reproducible artifact.
- **R2** — E1 bin-5 p50 differs from 11.09 by **> ±15%**
  => §0's E1-side number was a differently-aggregated statistic. The §0
  **magnitude** must be restated in `DESIGN.md` with the bin-conditional
  value. This does **not** by itself retract §0 — the disagreement may
  survive at a different size, or may not.
- **R3** — the C2/E1 ratio across all shared bins with n >= 50: if the ratio
  is **flat in bin**, batch composition is not driving it; if it **trends**,
  the single-bin comparison in §0 is not representative of the disagreement
  and the open item should be restated in terms of the trend.

## 6. What this check CANNOT do (registered now so it is not claimed later)

1. **It cannot adjudicate (i) vs (ii).** Whether 872077's `decode_sms == 16`
   is a real 16-SM hardware execution, versus C2's 28-31 ms being a property
   of the `[16,16,76-idle]` + saturated-keepalive cell composition, is
   **untouched by any outcome here**. Both remain open. Only a direct
   measurement (hardware SM-grant probe in engine context, or gate S1)
   separates them.
2. **It cannot verify the §4.3.11 residual layer** (green-context creation
   vs hardware SM grant). Note that job **864669** already established
   `DECODE_SM_HONORED` at the **API layer in isolation** (standalone matmul,
   pairwise green-context creation, no engine, no cudagraph, no concurrent
   prefill: `B(92,16)` eff 17.7, `B(16,44)` 55.7 == `B(64,44)` 55.6). The
   residual §4.3.11 flags is narrower than what 864669 answered — it is the
   **engine context** (3-4 groups created together by
   `initialize_stream_groups`, cudagraph-captured, prefill co-resident).
   Nothing offline reaches that layer.
3. **Non-removable confounds between the two populations**, to be *reported
   as descriptives*, never as controls:
   - prefill SM differs (16 vs 92) and idle SM differs (76 vs 0) — that
     difference **is** hypothesis (ii);
   - selection asymmetry: E1's SPLIT population is 3.8-9.3% of decode-active
     time and, on this substrate, exists **only while prefill is in flight**
     (§1-26(B) identity), whereas C2's is ~99-100% of its load window
     (§4.3.13 claim 1). The script reports `prefill_active` co-residency and
     SPLIT share within each population so the asymmetry is visible;
   - `s8_scaleup`'s keepalive reproducibility defect (§4.3.14 B-1) applies
     to the campaign, not to 865493 itself (which had
     `keepalive_errors = 0`), but any citation of the C2 side must carry it.

## 7. Verdict language

**DIAGNOSTIC ONLY.** No performance claim, no `G_LEVER`/`G_FLAT` value, no
statement about decode-SM elasticity is licensed by this check. Permitted
outputs are exactly: (a) a reproducible bin-conditional restatement of §0's
two numbers, (b) the bin-trend of their ratio, (c) descriptives of the two
populations. The result goes to claims-auditor before any canon edit —
this file was written by the same session that will run the script, so the
prescription is self-audited until an independent pass clears it
(methodology lesson 12).
