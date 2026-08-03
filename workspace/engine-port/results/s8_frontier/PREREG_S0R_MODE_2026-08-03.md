# Pre-registration — S0-R: independent replication of the bimodality finding

**Written before S0-R is run.** GPU 0, offline re-analysis of job **872077**
(and **865493** for the C2 comparands). Registered 2026-08-03 (fourth
continuation session).

## 0. What is being replicated, and why it needs replication

On 2026-08-03 claims-auditor, while auditing `FINDINGS_S0_AXIS_2026-08-03.md`,
produced a **reframing** of `DESIGN.md` §4.3.13 §0: that the E1 SPLIT
per-token ITL population is **bimodal**, that its upper mode reproduces C2's
per-cell SPLIT p50 to within 1–2% across d16/d24/d44, that the lower mode is
statistically identical to the same job's UNSPLIT (108 SM) population, and
therefore that `split_frac ≥ 0.90` **does not identify tokens that executed
on the D partition**. If true, §0's (i)/(ii) dichotomy is false and canon
gains a **second dilution layer inside the already-recorded time-level
dilution** (`E1_DECODE_REALIZED` 4–19%, §1-25).

**It needs replication because the auditor produced diagnosis and reframing
in one turn** — methodology lesson 12, which that same auditor invoked
against the previous turn and then explicitly self-applied here. Nothing
from it may enter canon on its own say-so.

## 1. Independence requirement (binding)

The analyst **must not be** claims-auditor, and **must not be** the session
that wrote `PREREG_S0_AXIS_2026-08-03.md` / `s0_axis_check.py` (the main
session). The analyst **must not reuse** the auditor's scratchpad scripts nor
`s0_axis_check.py`; it writes its own reader over the raw artifacts.
Reusing the *audited* `m3_conditional.py` primitives is permitted **and must
be declared per primitive**, because the disputed label is defined there.

⚠️ **Residual weakness, recorded now:** this pre-registration is written by
the main session, i.e. by a party that is not independent of the object under
replication. The estimator in §2 is **the auditor's proposal**, adopted
verbatim; §3's falsifiers 4 and 5 are the main session's additions. Neither
author is the analyst. This is a partial, not a complete, independence
guarantee — state it wherever the result is cited.

## 2. Estimator, fixed now (auditor's proposal (a), adopted verbatim)

- **unit**: one ITL interval = one emitted token, `a_free_only=True`
  (`m3_conditional.tokens`'s primary population — the population
  `s0_axis_check.py` failed to use).
- **label**: `split_frac ≥ 0.90` (SPLIT) / `≤ 0.10` (UNSPLIT108), as
  `m3_conditional.py:45-48`. Unchanged — the label is the *object* of study,
  not a free parameter.
- **mode estimator**: argmax of a **0.25 ms histogram** over
  ITL ∈ (1.15 × p50, 60] ms. Fixed before running; the 1.15 factor and the
  60 ms ceiling are not to be tuned.
- **aggregation**: pooled over the 8 blocks; per-block values reported
  alongside (block = boot = the replication unit, §4.3.8).

## 3. Pre-registered predictions and falsifiers

| # | prediction | falsifier |
|---|---|---|
| 1 | T8 `E1_slowmode(D) / C2_SPLIT_p50(D) ∈ [0.95, 1.05]` for D ∈ {16, 24, 44}, and **flat in D** | any D outside the band, or a monotone trend in the ratio |
| 2 | T8 slow-mode **share rises monotonically in D** (audit: 8.95 → 13.60 → 23.62 → 29.38%) | non-monotone, or any adjacent pair inverted beyond block-level scatter |
| 3 | T8 **fast mode ≈ same job's UNSPLIT population**, ratio ∈ [0.97, 1.03] (audit: 11.06 vs 10.97 = 1.011) | outside the band |
| 4 ★ | **NEGATIVE CONTROL (new, main session):** the same mode estimator applied to the **UNSPLIT108** population is **unimodal** — no second mode above 5% share in (1.15 × p50, 60] | if UNSPLIT also carries a ~31 ms mode at ~9%, then "slow mode = D-SM execution" collapses: the mode would be a tail present *regardless of partition*, and the whole reframing fails |
| 5 ★ | **Lag-shift companion (auditor's (c), mandatory):** δ ∈ [−0.40, +0.40] s in 0.05 s steps; report argmax of slow-share. Audit found argmax at δ ≈ +0.05–0.10 s with share 13.6–14.9% vs 8.95% at δ=0 | if the argmax share reaches ≥ 0.90, the "impurity" is entirely a clock artifact and `m3.align` — not the label — is the defect |

**Ha8 is NOT out-of-sample.** The auditor has already seen its answer
(~0.80–0.85×, plus an unexplained cell-independent ~87 ms spike, plus a flat
SPLIT p50 of 31.6–32.1 across all four cells = zero SM dependence). It is
registered here as a **consistency check with a known answer**: a replication
result outside [0.75, 0.90] indicates divergence from the audit and must be
reported as such, not smoothed over. **The unexplained ~87 ms Ha8 spike is an
open sub-item either way** — no outcome of S0-R explains it.

## 4. Scope limits (registered now)

1. S0-R is **offline**. It cannot show that the slow mode *is* 16-SM hardware
   execution — only that the E1 population contains a mode whose magnitude
   matches C2's. The causal claim needs S2 (or S3).
2. C2's side is `n_indep = 1` boot (4 reps = pseudo-replication), unchanged.
3. C2 comparands exist only for **T8** and **Hs8** at d16 (865493 has no
   client `t0` anchor for Ha8 d16 or for M8 at all — verified 2026-08-03);
   the *unconditioned* cell-level contrast is available for all arms and is a
   different, weaker estimand.
4. `s8_scaleup`'s keepalive reproducibility defect (§4.3.14 B-1) attaches to
   any citation of the C2 side.

## 5. Verdict language

**DIAGNOSTIC.** Permitted outputs: replicate / fail-to-replicate per row of
§3, with per-block scatter. **No** performance claim, **no** `G_LEVER`/
`G_FLAT` value, **no** adjudication of §0's (i)/(ii), **no** canon edit.
If §3 rows 1–3 replicate and rows 4–5 do not fire, the recommended next
action is S2 (`PREREG_S2_STICKY_ITL_2026-08-03.md`) — which is a *GPU*
measurement, and is where the causal claim gets tested.
