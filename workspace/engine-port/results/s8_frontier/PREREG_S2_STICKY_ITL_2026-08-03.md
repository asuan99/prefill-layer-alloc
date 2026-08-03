# Pre-registration — S2: what the first sticky run must show about the label

**Written before the sticky grid is submitted.** Registered 2026-08-03
(fourth continuation session). Companion to, and **does not replace**,
`DESIGN.md` §4.3.12 (the sticky run's own pre-registration for `g`).

## 0. Why this exists

`DESIGN.md` §4.3.13 §0 poses a dichotomy — **(i)** 872077's
`decode_sms == 16` is not a real 16-SM hardware execution, or **(ii)** C2's
28–31 ms is a property of its cell composition. The 2026-08-03 audit
proposes a **third** answer: both jobs sit on the same axis, 872077 *does*
contain genuine D-SM decode at C2's cost, and the label `split_frac ≥ 0.90`
merely fails to isolate it (only ~9% of T8 d16 SPLIT tokens carry the D-SM
cost; the rest run at 108 SM).

That reframing is **UNAUDITED** and is being replicated offline as S0-R.
S2 is the *causal* test, and it is **free**: the `PDMUX_STICKY_PARTITION`
grid is already implemented, correctness-gated (§4.3.11), and pending
submission. Registering a prediction now costs nothing and makes that run
decisive about the label question in addition to whatever it says about `g`.

## 1. Run (no new GPU work beyond the already-planned grid)

T8 **d16** primary, **d54** companion · sticky **ON**
(`PDMUX_STICKY_PARTITION=1`, `PDMUX_R2_POLICY=fixed`) · ShareGPT rate 2 ·
8 blocks · `PDMUX_TRACE_FORCE_PREFILL=0` · server args as
`e1_m3_control.sbatch:212-218`.

## 2. Gates (must pass before the prediction is read)

`E1_DECODE_REALIZED ≥ 0.90` (a **real** gate under sticky, per §1-25/§4.3.11 —
it was an identity before) · `ALIGN_R` flag-only, never a silent drop ·
`AMBIG_FRAC` upper bound and `MIN_N_SPLIT` lower bound as §4.3.12(c).

## 3. The prediction, fixed now

Statistic: **per-token p50 of the SPLIT population**, `a_free_only=True`,
pooled over 8 blocks, per-block scatter reported.

| arm/cell | 872077 (sticky OFF) | **S2 prediction (sticky ON)** | source of the interval |
|---|---|---|---|
| T8 d16 | 11.09 ms | **[28, 34] ms** | C2 865493 T8 d16 SPLIT p50 = 31.12, ±~10% |
| T8 d54 | — | **[13, 16] ms** | C2 d44 14.68 / audit's d54 upper mode 14.12 |

⚠️ **Statistic scope.** p50 is registered **for this prediction only**,
because the disputed quantity (§0's 2.6×) is stated in p50 and because the
audit shows p95 on this substrate is a *realization-rate indicator* on a
bimodal population (2 of 8 blocks give r(p95) ≈ 2.6, six give ≈ 1.0).
**§4.3.12(b)'s `p95(SPLIT)` primary for the `g` verdict is UNCHANGED** —
§4.3.13 claim 2 already REFUTED switching it, and nothing here reopens that.

## 3.1 Revisions after S0-R — made BEFORE submission, per §5.1

S0-R (`S0R_REPLICATION_2026-08-03.md`, independent execution) adjudicated all
five rows. **§3's intervals are NOT revised** — its binding condition was
rows 1–3, and those replicated (row 1 ratio 0.992/0.991/1.013 flat in D;
row 3 fast-mode ≈ UNSPLIT at 1.011). Four corrections and one strengthening
are recorded instead:

1. **Population declaration for the audit's figures (row-2 inconsistency,
   inside the pre-registration, not in the data).** The slow-share series
   8.95 / 13.60 / 23.62 / 29.38 % quoted from the audit is the
   **`a_free_only=False`** population. The `a_free_only=True` population that
   `PREREG_S0R` §2 fixes gives **6.82 / 9.48 / 17.68 / 22.55 %**. Any figure
   cited from the audit must carry its population. Ordering replicates either
   way; the d24−d16 step is **not resolved** (+2.37±3.52 pp, t=1.90 < 2.365).
2. **The mode window's 60 ms ceiling is not arm-portable.** With Ha8's
   p50 ≈ 31.7 ms, `(1.15×p50, 60]` holds **0.16%** of Ha8 d16 SPLIT tokens
   (47) against a `share_above` of 15.7% — Ha8's slow mass, including the
   unexplained ~87 ms spike, sits **above the ceiling by construction**. State
   the ceiling in units of p50 wherever this estimator is reused. S0-R
   correctly declined to change it after the fact.
3. **Row-1's comparand matters at the edge.** Against C2's *batch-matched*
   bin-5 comparand the ratios are 1.072 / 1.060 / 1.079 — **outside**
   [0.95, 1.05] though still flat. §3's interval **[28, 34] ms covers both**
   comparands (bin-5 28.79 and pooled 31.12), so the prediction is unaffected;
   a narrower interval would not have been.
4. **C2's SPLIT label is near-vacuous** (90,848/91,073 = 99.75% SPLIT, **zero**
   UNSPLIT at d16), so row 1 compares an E1 3.7% minority against effectively
   all of C2. Inherits §1-29 B-2 (high residency = keepalive workload device)
   and the §4.3.14 B-1 keepalive defect.

★ **Strengthening — S2 is now MORE decisive, not less.** Row 4 fired: the
slow mode is **present in the UNSPLIT(108 SM) class at every cell**, its
location **tracks the cell** (33.88 → 22.12 → 15.62 → 14.12 ms as D goes
16→24→54), and **88.9% of d16's slow tokens carry the UNSPLIT label**. So
`split_frac ≥ 0.90` is neither pure (~93% of its tokens are fast) nor
complete. **S2 does not depend on that label**: under sticky ON with
`E1_DECODE_REALIZED ≥ 0.90` the partition is held for essentially all
decode-active time, so `p50(SPLIT)` ≈ `p50(all tokens)` and the readout in §4
is a statement about the *run*, not about the label. Report **both**
`p50(SPLIT)` and `p50(all)` and require them to agree within 5% — if they do
not, realization did not actually reach the population the statistic is taken
over, and the run is UNMEASURABLE rather than negative.

## 4. Three-way readout

| outcome | reading |
|---|---|
| **T8 d16 p50 lands in [28, 34] ms** with realization ≥ 0.90 | the partition **is** honored in engine context and sticky realizes it ⇒ §0's 2.6× was the label-dilution artifact ⇒ **§0's dichotomy closes as false**; §4.3.11's residual layer is answered *behaviourally* (not directly) |
| **p50 stays ≈ 11 ms** with realization = 1.00 | the green context is **not** honored for graph-replayed decode in engine context ⇒ **Stage-0-class correction**: a labeled SM count that does not correspond to the SM count granted. Everything built on 872077 and on the sticky grid is affected |
| **intermediate** (p50 ∈ (14, 28) ms, or realization < 0.90) | ambiguous ⇒ escalate to **S3** (per-decode-step logging of the actual stream / green-context handle; audit predicts 15–20% of steps on the D stream) |

## 5. Binding conditions

1. **If S0-R fails to replicate** (`PREREG_S0R_MODE_2026-08-03.md` §3 rows
   1–3), this pre-registration's intervals are derived from a finding that
   did not survive, and §3 **must be revised before submission** — not
   re-scored afterwards (methodology gate #8).
2. **Gate S1 (§4.3.13) is not runnable as written.** Its three-way verdict
   table has **no branch for "partially realized"**, which is exactly the
   outcome this audit points at. It needs a fourth branch before it is
   submitted. Recorded here so S1 is not run in its current form.
3. This file adds a readout to an existing run. It does **not** license any
   statement about throughput, goodput, `g`, `G_LEVER`, `G_FLAT`, or
   decode-SM elasticity.
