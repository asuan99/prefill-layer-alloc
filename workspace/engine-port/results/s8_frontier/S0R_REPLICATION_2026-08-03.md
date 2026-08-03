# S0-R — independent replication of the bimodality finding [DIAGNOSTIC ONLY]

Pre-registration: [`PREREG_S0R_MODE_2026-08-03.md`](PREREG_S0R_MODE_2026-08-03.md)
· Script: [`S0R_REPLICATION_2026-08-03.py`](S0R_REPLICATION_2026-08-03.py)
· Raw output: [`S0R_REPLICATION_2026-08-03.txt`](S0R_REPLICATION_2026-08-03.txt)
· GPU 0. Offline re-analysis of **872077** (E1) and **865493** (C2).
· Run by **result-analyst**, 2026-08-03 — not claims-auditor, not the session
that wrote the pre-registration.

> **Verdict language is fixed by prereg §5: DIAGNOSTIC.** Nothing below is a
> performance claim, a `G_LEVER`/`G_FLAT` value, an adjudication of `DESIGN.md`
> §4.3.13 §0's (i)/(ii), or a canon edit. Per prereg §1's own recorded
> weakness, the independence here is **partial**: the estimator is the
> auditor's proposal and falsifiers 4–5 are the main session's; only the
> execution is independent. Quote that scope wherever this is cited.

## 0. Independence ledger (prereg §1)

Reused from the audited `m3_conditional.py`, because the disputed label is
defined there and re-implementing it would change the object under study:
`label_probe` (the whole E1 labelling pass — `split_frac`, the `a_free` flag,
the clock alignment), `tokens` (primary population, `a_free_only=True`),
`sel_split`/`sel_unsplit` (bands 0.90/0.10), `pctl`, and transitively
`load_telemetry`, `replay_arrivals`, `align`, `label_probe`'s inner `frac`.

Written fresh: the mode estimator and every share statistic, the **entire C2
reader** (anchor parser, ITL reconstruction, telemetry loader, `wfrac`
integrator, batch weighting), the lag-shift driver, all reporting. Neither
`s0_axis_check.py` nor any scratchpad script was read into the program.
`c2_anchor.py` was read only to learn *which estimand* the number 28.79 names
(a batch-bin-matched p50) so the row-1 comparand would not silently be a
different quantity; no code or value is imported from it — 28.79 is
re-derived here from raw artifacts.

Two prereg under-specifications, resolved **before** running and both reported
with their alternative: **D1** histogram bin origin = global grid anchored at
0.0 ms (so SPLIT and UNSPLIT modes land on one grid); alt-anchor at 1.15·p50
reported. **D2** "slow share" = in-window mass `#(1.15·p50 < ITL ≤ 60)/n`
(primary; the support the estimator and row 4 are stated on), unbounded mass
reported alongside.

## 1. Instrument gates (mine, run first)

| gate | result |
|---|---|
| **G-A** my `wfrac` ≡ `label_probe`'s inner `frac`, element-wise (T8 d16 b1) | **PASS** — 39,557/39,557 records, max abs diff **0.000e+00** |
| **G-B** my C2 ITL reconstruction ≡ `s0dc_client`'s own recorded summary | **PASS** — 20/20 anchored reps, `itl_samples` exact and `itl_ms_p50` exact to 4 dp |
| anchors | Ha8 d16 (4 reps) has **no** `t0_monotonic_s` → excluded, not imputed. M8 has none at all (and is not an E1 arm). |
| E1 align | 2/64 probes `align_r` < 0.95 (T8 d16 b1 = 0.882, b6 = 0.930) — **flagged, never dropped** |

G-B validates the C2 side against the *producer* (`s0dc_client.py`), not
against another analysis script — the circularity the audit flagged in the
previous gate 2 does not apply here.

## 2. Row-by-row verdict

### Row 1 — E1 slow mode / C2 SPLIT p50 ∈ [0.95, 1.05] and flat in D → **REPLICATES (T8), with a comparand caveat**

| cell | E1 slow mode (ms) | C2 SPLIT p50 (ms) | ratio | per-block ratio (n=8) |
|---|---|---|---|---|
| T8 d16 | 30.88 | 31.11 (n=90,848) | **0.992** | 1.041 ± 0.053 [0.976, 1.113] |
| T8 d24 | 21.88 | 22.07 (n=116,908) | **0.991** | 0.990 ± 0.009 [0.968, 1.002] |
| T8 d44 | 14.88 | 14.68 (n=165,794) | **1.013** | 1.013 ± 0.009 [0.996, 1.030] |

All three inside [0.95, 1.05]; spread across D is 2.2 % and non-monotone
(0.992 / 0.991 / 1.013) ⇒ **flat in D**. Alt-anchor bins give 0.992/0.989/1.011.

**Caveat that must travel with this row.** The ratio's band membership depends
on which C2 comparand is used:

- vs **all-batch** C2 SPLIT p50 (the prereg's stated comparand): 0.992/0.991/1.013 — in band.
- vs **batch-matched** C2 SPLIT p50 at the E1 modal decode batch (bin 5, the
  estimand "28.79" names — reproduced here independently as **28.79**/20.64/13.78):
  **1.072 / 1.060 / 1.079** — *out* of band, though still flat in D.

Also: C2's SPLIT label is near-vacuous. Of C2 T8 d16's 91,073 in-window
tokens, **90,848 (99.75 %) are SPLIT-labelled and 0 are UNSPLIT108**; C2's
SPLIT p50 equals its unconditioned p50 to 2 dp. So this row compares an E1
minority mode (3.7 % of tokens) against what is effectively C2's whole
population, and inherits `CONSENSUS` §1-29 B-2 (C2's high residency is the
keepalive workload device, not decode-partition control) plus the
`s8_scaleup` reproducibility defect.

### Row 2 — T8 slow-mode share rises monotonically in D → **REPLICATES in ordering; 2 of 3 steps are inside block scatter; the audit's *levels* come from a different population**

Pre-registered population (`a_free_only=True`), pooled share_win:
**6.82 → 9.48 → 17.68 → 22.55 %** for d16/d24/d44/d54. Monotone, no adjacent
inversion.

Paired-within-block (blocks share the seed across cells, so the block index
pairs; n=8):

| step | Δ share (pp) | blocks positive | t (df=7) |
|---|---|---|---|
| d24 − d16 | **+2.37 ± 3.52** | 6/8 | 1.90 — **not resolved** (t_crit 2.365) |
| d44 − d24 | **+8.85 ± 4.57** | 8/8 | 5.48 — resolved |
| d54 − d44 | **+3.53 ± 4.10** | 6/8 | 2.44 — marginal |

**Divergence located and explained.** The prereg's parenthetical audit numbers
(8.95 / 13.60 / 23.62 / 29.38 %) are **not** from the population the prereg's
§2 estimator fixes. They are reproduced by the `a_free_only=**False**`
population: I get **8.95** / 13.79 / 24.03 / 29.69 % (`share_above`; n = 11,451
/ 14,168 / 23,783 / 27,731), against 6.82 / 9.48 / 17.68 / 22.55 % on the
pre-registered `a_free_only=True` population (n = 11,124 / 13,424 / 21,770 /
24,917). That is an internal inconsistency **inside the pre-registration**
(§2 fixes `a_free_only=True`; §3 row 2 quotes numbers from the other
population), not a data discrepancy. The direction — the object of row 2 —
replicates on both.

### Row 3 — T8 fast mode ≈ same job's UNSPLIT population, ratio ∈ [0.97, 1.03] → **REPLICATES**

| cell | SPLIT fast mode | UNSPLIT mode | ratio | per-block (n=8) | p50/p50 |
|---|---|---|---|---|---|
| T8 d16 | 11.12 | 11.12 | **1.000** | 1.003 ± 0.022 | 1.011 |
| T8 d24 | 11.12 | 10.88 | **1.023** | 1.003 ± 0.015 | 1.008 |
| T8 d44 | 10.88 | 10.88 | **1.000** | 1.003 ± 0.022 | 1.009 |
| T8 d54 | 11.12 | 10.88 | **1.023** | 1.006 ± 0.011 | 1.016 |

All in band; the d16 p50/p50 = **1.011** is exactly the audit's figure. The
0.25 ms grid quantises the ratio to ±0.023 at 11 ms, which is the width of the
band itself — the p50/p50 column is the more discriminating read and agrees.

### Row 4 ★ (negative control) — **letter: does not fire at d16/d24/d44, FIRES at d54. Substance: the second mode is present in UNSPLIT at every cell, at the same location.**

Same estimator on the UNSPLIT108 population:

| cell | UNSPLIT n | UNSPLIT slow mode | UNSPLIT share_win | per-block (n=8) | vs 5 % threshold |
|---|---|---|---|---|---|
| T8 d16 | 291,763 | **33.88** | **2.71 %** | 2.71 ± 0.17 [2.41, 2.94] | does not fire |
| T8 d24 | 288,818 | **22.12** | **3.28 %** | 3.28 ± 0.19 [2.98, 3.66] | does not fire |
| T8 d44 | 278,216 | **15.62** | **4.84 %** | 4.85 ± 0.50 [4.13, 5.56] | does not fire (margin 0.16 pp) |
| T8 d54 | 273,914 | **14.12** | **5.61 %** | 5.57 ± 0.57 [4.97, 6.57] | **FIRES** |

The pre-registered falsifier was "a ~31 ms mode at ~9 %". UNSPLIT carries it at
**2.71 %**, so the falsifier as written does not fire at d16 — but its
*rationale* is partly borne out, and the shape data are the reason:

1. **UNSPLIT is bimodal too, not a decaying tail.** 1 ms histograms (raw
   output, [ROW 4b]): at d16 both classes have **zero mass in bins 12–27** and
   a separated bump at 28–34 ms (SPLIT 0.2/0.7/1.8/1.3/0.9/0.9/0.8 %;
   UNSPLIT 0.1/0.3/0.5/0.5/0.4/0.6/0.3 %). Same structure at d24 (bump 20–26)
   and d44 (bump 12–20).
2. **The UNSPLIT bump's location tracks the cell**: 33.88 → 22.12 → 15.62 →
   14.12 ms as D goes 16 → 24 → 44 → 54, i.e. it moves with the *cell label*
   although every token in that class is labelled as having run at 108 SM. At
   d54 the two classes' slow modes are **identical** (14.12 vs 14.12).
3. **Enrichment, not exclusivity.** Base rate of slow tokens over the whole
   a_free population and the two class rates (cut = 1.15·p50 of SPLIT):

   | cell | base | SPLIT | enrich | UNSPLIT | enrich | slow tokens: SPLIT / all |
   |---|---|---|---|---|---|---|
   | T8 d16 | 2.93 % | 6.82 % | 2.33× | 2.71 % | 0.93× | 759 / 8,893 = **8.5 %** |
   | T8 d24 | 3.63 % | 9.48 % | 2.61× | 3.28 % | 0.90× | 1,272 / 11,017 = 11.5 % |
   | T8 d44 | 5.81 % | 17.68 % | 3.04× | 4.77 % | 0.82× | 3,848 / 17,540 = 21.9 % |
   | T8 d54 | 6.96 % | 22.55 % | 3.24× | 5.41 % | 0.78× | 5,619 / 20,974 = 26.8 % |

   At d16, of the 8,893 slow tokens **7,903 (88.9 %) carry the UNSPLIT label**,
   759 (8.5 %) carry SPLIT and 231 (2.6 %) are AMBIGUOUS. (The enrichment table
   uses one common cut = 1.15·p50(SPLIT) for all sub-populations so the shares
   are commensurable; the row-4 table above uses each population's own cut,
   hence 4.77 vs 4.84 % at d44.)
4. **Not an artifact of the two weak-alignment blocks**: the UNSPLIT slow
   share is 2.71 ± 0.17 pp across all 8 blocks (range 2.41–2.94), i.e. present
   in every block including the six with `align_r` ≥ 0.99.

Two readings survive and **S0-R cannot separate them offline**: (a) the slow
mode is a cell-level phenomenon present regardless of the *realised* partition,
or (b) it is D-execution leaking into the UNSPLIT class through a systematic
clock offset (row 5 shows the labelling is sharply lag-sensitive at exactly
the ±0.1 s scale). Note the prefill-overlap column cannot arbitrate: it is
computed from the *same* shifted clock, so a mislabelled token is mislabelled
on both fields simultaneously (and `CONSENSUS` §1-27 records that this label is
under-sampled in 872077 anyway). Under either reading the consequence for the
label is the same and is *stronger* than the audit's version: `split_frac ≥
0.90` is neither pure (≈93 % of SPLIT tokens sit at the fast mode) nor
complete (it captures 8.5 % of the job's slow tokens at d16).

### Row 5 ★ (lag sweep) — **falsifier does NOT fire; argmax location replicates; the contrast is narrow in δ**

δ ∈ [−0.40, +0.40] s in 0.05 s steps, relabelling with `lag + δ`, pooled over 8 blocks:

| arm/cell | argmax δ | share at argmax | share at δ=0 | share at \|δ\| ≥ 0.25 | fires (≥0.90)? |
|---|---|---|---|---|---|
| T8 d16 | **+0.10 s** | **11.40 %** | 6.82 % | 2.80–3.09 % | **no** |
| T8 d44 | **+0.10 s** | **26.97 %** | 17.68 % | 7.86–8.65 % | **no** |

The audit's argmax location (δ ≈ +0.05…+0.10 s) replicates on both cells. Two
observations the prereg did not ask for but that bear on rows 1–4:

- At \|δ\| ≥ 0.25 s the SPLIT slow share **collapses to 2.80–3.09 %, i.e. to
  the UNSPLIT background (2.71 %) and the population base rate (2.93 %)**. The
  entire SPLIT-vs-background contrast lives inside a ±0.15 s window around the
  estimated lag. That is evidence the label carries *real* timing information
  (a random label would sit at the base rate everywhere), and simultaneously
  that it is fragile at the 0.05 s scale.
- The as-run alignment sits on the **shoulder** of that peak, not on it:
  δ=0 gives 6.82 % where δ=+0.10 gives 11.40 %, and `n_split` at δ=0 (11,124)
  is 10–15 % below its plateau. Even at the optimum, **88.6 % of
  SPLIT-labelled tokens are at the fast mode**.

### Ha8 — consistency check with a known answer, **DIVERGES at d24, agrees at d44**, and the estimator is out of scope for Ha8 at d16/d24

Prereg band [0.75, 0.90]:

| cell | E1 slow mode | C2 SPLIT p50 | ratio | per-block (n=8) | in band? |
|---|---|---|---|---|---|
| Ha8 d16 | 38.12 | — | — | — | C2 anchor missing |
| Ha8 d24 | 55.88 | 80.56 | **0.694** | 0.673 ± 0.096 [0.445, 0.737] | **NO — divergence** |
| Ha8 d44 | 42.88 | 51.16 | **0.838** | 0.886 ± 0.146 [0.667, 1.087] | yes |
| Ha8 d54 | 40.12 | — | — | — | no C2 d54 cell |

**Why Ha8's numbers should not be pushed further:** Ha8's SPLIT p50 is ~31.7 ms,
so the pre-registered window (1.15·p50, **60**] holds almost nothing —
**0.16 %** of Ha8 d16 SPLIT tokens (**47 tokens**) and 1.24 % at d24 (415
tokens), against `share_above` of 15.71 % / 17.90 %. Most of Ha8's slow mass is
**above the 60 ms ceiling**, i.e. outside the estimator's support by
construction. The fixed ceiling was chosen for a population with an 11 ms p50;
on Ha8 it makes the mode an argmax over tens of tokens. As the prereg said, the
unexplained ~87 ms Ha8 spike is an open sub-item that **no S0-R outcome
explains** — and this is the mechanical reason why: the estimator cannot see it.

## 3. Summary table

| # | prediction | verdict |
|---|---|---|
| 1 | ratio ∈ [0.95,1.05], flat in D | **replicates** (0.992/0.991/1.013, flat) — but 1.060–1.079 and out of band against the batch-matched comparand |
| 2 | slow share monotone in D | **replicates in ordering**; d24−d16 (+2.37 ± 3.52 pp, 6/8) and d54−d44 (+3.53 ± 4.10 pp, 6/8) not resolved beyond block scatter; audit's *levels* are from the `a_free_only=False` population |
| 3 | fast mode ≈ UNSPLIT, ∈ [0.97,1.03] | **replicates** (1.000–1.023; p50/p50 1.011 at d16) |
| 4 ★ | UNSPLIT unimodal, no 2nd mode > 5 % | **letter: holds at d16 (2.71 %), d24 (3.28 %), d44 (4.84 %); FIRES at d54 (5.61 %).** Substance: UNSPLIT is bimodal at every cell, bump at the same cell-tracking location, 88.9 % of the job's slow tokens are UNSPLIT-labelled at d16 (SPLIT holds 8.5 %) |
| 5 ★ | argmax slow-share over δ; fires if ≥ 0.90 | **does not fire** (max 11.40 % at δ=+0.10 s, d16; 26.97 %, d44). Argmax location replicates. Contrast vanishes to the base rate by \|δ\| ≥ 0.25 s |
| — | Ha8 consistency, [0.75, 0.90] | d44 0.838 in band; **d24 0.694 diverges**; d16/d54 not reconstructible; estimator support ~empty for Ha8 at d16/d24 |

## 4. Scope limits (prereg §4, plus what this run added)

1. Offline. Nothing here shows the slow mode **is** 16-SM hardware execution —
   only that the E1 population contains a mode whose magnitude matches C2's.
   Rows 4–5 sharpen why: the same mode is present in the class labelled 108 SM,
   and the label's contrast is a narrow function of an estimated clock lag.
   Separating the two readings needs a GPU measurement (S2 sticky).
2. C2 side is `n_indep = 1` boot (4 reps = pseudo-replication) — unchanged.
3. C2 comparands exist only where a client `t0_monotonic_s` anchor exists:
   T8 d16/d24/d44 and Ha8 d24/d44 here. Ha8 d16 (4 reps) and M8 (all) have
   none. Verified, not assumed.
4. `s8_scaleup`'s keepalive reproducibility defect (§4.3.14 B-1) and §1-29 B-2
   (C2 residency = workload device) attach to every C2 citation above.
5. **New:** the pre-registration is internally inconsistent about the
   population behind row 2's quoted levels (§2 says `a_free_only=True`; the
   §3 row-2 numbers are the `False` population). Any future prereg that quotes
   audit figures should quote them **with their population**.
6. **New:** the 60 ms ceiling is not arm-portable. It is a live constant for
   T8 (p50 ≈ 11 ms) and effectively censoring for Ha8 (p50 ≈ 31.7 ms). A
   ceiling stated in units of p50 would have avoided this; changing it now
   would be tuning after the fact, so it is recorded, not changed.

## 5. What this does and does not license

Per prereg §5: rows 1–3 replicate and row 5 does not fire, so the recommended
next action is unchanged — **S2** (`PREREG_S2_STICKY_ITL_2026-08-03.md`), a GPU
measurement, is where the causal claim gets tested. Row 4's split result
(letter passes at 3 of 4 cells, fires at d54, substance shows the mode in both
classes) means the auditor's reframing **may not be adopted as "the slow mode
identifies D-SM execution"**; what is replicated is weaker and quantitative:
the slow mode is **enriched 2.3–3.2× in the SPLIT class, not confined to it**.
No canon edit is made here and none is implied; routing to doc-steward is the
main session's call, and any citation must carry the partial-independence
scope from prereg §1 and the caveats in §2–§4 above.
