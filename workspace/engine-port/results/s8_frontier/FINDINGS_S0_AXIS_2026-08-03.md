# §0 axis check — results [AUDITED 2026-08-03 — PARTLY RETRACTED, do not cite]

> ★★★**RETRACTION BANNER (claims-auditor, 2026-08-03, same day).** Three
> sentences below are withdrawn, and the audit's own re-analysis of the data
> this script loaded points at a conclusion this document does not reach.
>
> **Withdrawn:**
> 1. §1's "**§0 stands as written**" — over-read. What was shown is only that
>    E1's p50 is ≈11 ms under three nearby aggregations.
> 2. §1's "**aggregation-invariant**" — the correct statement is "dominated by
>    a single mode at 11.06 ms, hence insensitive to aggregation." The
>    agreement establishes **mode dominance, not estimand identification**.
> 3. §1's motivating premise "**11.09 has no recorded aggregation unit**" —
>    **false**. Its producer is `m3_conditional.report_conditional` report [3],
>    T8 d16 `sp_p50` = **11.0905** (n=11,124, `a_free_only=True`), documented at
>    `m3_conditional.py:158-161,251-262,316-329`. Only its *stdout* was never
>    saved. One command answered it:
>    `python3 m3_conditional.py --job 872077 --only conditional`.
>
> **Further defects found (see the audit for evidence):** gate 1 is
> near-tautological and never validates the one new quantity (`wmean`); gate 2
> is circular (calls `c2_anchor.collect`, the producer of 28.79); `e1_side`
> omits `a_free_only=True`, so its SPLIT population (n=11,451) is **not** the
> audited estimator's (n=11,124) and the 327 extra records are p50 31.30 ms —
> the disputed slow mode; §5's `align_r` row says one block <0.95, measured is
> **two** (0.882, 0.930); §3's "shares an upper tail but not a body" is the
> wrong frame (C2's p95 **is** its body; E1's p95 is a 9% minority mode) and
> the p95 agreement is a **metric cliff on a bimodal population** carrying one
> bit — "slow share > 5%"; §4's "absent measurement" is too broad (the Ha8 d16
> reps ran — `itl_ms_p50=112.84`, n=5200 — only the telemetry anchor is
> missing) and "T8-only by construction" is too strong (the *unconditioned*
> Ha8 cross-job contrast exists and is **larger**: 112.84 vs 31.76 = 3.55×).
>
> **What survives:** the two numbers reproduce (bin-5 11.00 / pooled 11.10 /
> C2 28.79); batch is dead as an explanation (though for a stronger reason
> than the 3-point OLS — both sides' ITL is flat in batch *and* in KV length);
> §6's 864669 scope reading is accurate; the DIAGNOSTIC-ONLY framing, the
> refusal to adjudicate (i)/(ii), never-drop-UNDERPOWERED, flag-only
> `align_r`, and the self-audit disclosure all held.
>
> **The audit's own finding is itself UNAUDITED** (it produced diagnosis and
> reframing in one turn — methodology lesson 12) and must not enter canon
> without independent replication (proposed gate S0-R). Summary of it is in
> the session record; nothing here or there is canon yet.

Pre-registration: [`PREREG_S0_AXIS_2026-08-03.md`](PREREG_S0_AXIS_2026-08-03.md)
· Script: [`s0_axis_check.py`](s0_axis_check.py)
· Raw output: [`S0_AXIS_CHECK_2026-08-03.txt`](S0_AXIS_CHECK_2026-08-03.txt)
· GPU 0 (offline re-analysis of jobs **865493** and **872077**).
Written by the same session that wrote the pre-registration ⇒ **self-audited
until claims-auditor clears it** (methodology lesson 12). Target =
`DESIGN.md` §4.3.13 §0, the current top open item.

## Gates (PREREG §4)

| gate | result |
|---|---|
| 1. E1 labelling ≡ `m3_conditional.label_probe` `(itl, split_frac)`, element-wise, all 8 blocks × 2 arms | **PASS** (asserted, not eyeballed) |
| 2. C2 T8 d16 bin-5 SPLIT p50 reproduces `c2_anchor.py` [4] = 28.79 ± 0.05 ms | **PASS** (28.79) |
| 3. `align_r` flag-only | T8 0.88–1.00 (one block < 0.95, **flagged, not excluded**), Ha8 1.00–1.00 |

## 1. R1 — §0's E1-side number is reproducible and aggregation-invariant

| statistic | value | vs §0's 11.09 ms |
|---|---|---|
| E1 T8 d16 **bin-5 conditional** SPLIT p50 | **11.00 ms** (n=1720) | −0.8% |
| E1 T8 d16 **pooled** SPLIT p50 (mean batch 6.29) | **11.10 ms** | +0.1% |

⇒ **R1**, not R2. The concern that motivated this check — that 11.09's
aggregation unit was unrecorded and might be a pooled quantile standing in
for a bin-conditional one — is **resolved without changing §0**: on this
population the two forms agree to 1%. §0's magnitude now has a stored
artifact behind it. **§0 stands as written.**

## 2. R3 — the 2.6× does not come from decode batch

Shared bins with n ≥ 50 on both sides (only three exist):

| bin | C2 p50 | E1 p50 | r(p50) | C2 p95 | E1 p95 | r(p95) |
|---|---|---|---|---|---|---|
| 1 | 29.98 | 10.68 | 2.808 | 30.76 | 32.54 | 0.945 |
| 4 | 28.76 | 10.91 | 2.636 | 30.84 | 31.01 | 0.995 |
| 5 | 28.79 | 11.00 | 2.618 | 30.74 | 31.71 | 0.969 |

`r(p50)` OLS slope = **−0.050/bin** — flat. ⇒ batch composition is not
driving the disagreement, consistent with the auditor having already matched
that axis. **Limit**: the two batch distributions barely overlap (C2's mass
is at bin 16 with n=73,126; E1's at bins 4–9), so "flat" is established over
bins 1–5 only, and bins 2/3/6–10 are single-sided.

## 3. ★ Unregistered observation — p50 differs 2.6×, p95 agrees within 5%

This was **not pre-registered** and is therefore **exploratory only**. At all
three shared bins `r(p95)` = 0.945 / 0.995 / 0.969 while `r(p50)` ≈ 2.6–2.8.
Shapes:

- **C2** T8 d16 SPLIT: p50 28.79 → p95 30.74 — nearly a **point mass** at ~30 ms.
- **E1** T8 d16 SPLIT: p50 11.00 → p95 31.71 — **body at ~11 ms, tail to ~32 ms**.

So the two populations **share an upper tail but not a body**. This is
discriminator-shaped with respect to §0's (i)/(ii), which is exactly why it
must not be interpreted here: the obvious reading (E1's SPLIT label covers a
mixture whose slow component coincides with C2's entire population) is one of
several, and this session has no licence to pick one. Registered as a
**question for the auditor**, not a finding.

⚠️ It also touches a live pre-registration: §4.3.12(b) declares
`p95(SPLIT)` the sticky primary, and §4.3.13 claim 2 **REFUTED** the proposed
switch to p50 (T8's positive control collapses to 1.00 at p50). If the p95s
of two substrates that disagree 2.6× at p50 are indistinguishable, the
question of what `p95(SPLIT)` is sensitive to is live — **but nothing here
licenses changing the declared primary**, and no such change is proposed.

## 4. Data availability discovered during the run (not a result)

`c2_anchor.collect` needs a client `t0_monotonic_s` anchor per cell. In job
**865493** it exists for **T8** {d16,d24,d44,d92,np} and **Hs8** {same}, but
for **Ha8** only {d24,d44,d92,np} — **d16 is missing** — and for **M8**
**none at all**.

⇒ (a) §0's cross-job comparison is **T8-only by construction**: C2 lacks the
Ha8 d16 anchor and E1 (872077) has no Hs8 arm. (b) An arm with no anchor
yields zero rows, which is an **absent measurement, not a measured zero** —
the script now labels it so, because `c2_anchor.py`'s own tables show the
same blanks without that distinction. (c) On the C2 side the comparison rests
on **one boot** with 4 reps = pseudo-replication (§4.3.13's `n_indep = 1`,
unchanged).

## 5. Population descriptives (reported, not controls — PREREG §6.3)

| arm | C2 n_tot | C2 SPLIT share | C2 prefill@decode-busy | C2 realized@D | E1 SPLIT share | E1 realized@D |
|---|---|---|---|---|---|---|
| T8 | 90,971 | 0.998 | 0.904 | 0.958 | 0.037 | 0.0380 |
| Ha8 | 0 (absent) | — | 0.825 | 0.865 | 0.090 | 0.1041 |

The engagement asymmetry is as §1-25 recorded (E1 3.8–10.4% vs C2 ~96%).
`E1 prefill|SPLIT` (T8 0.245 / Ha8 0.356) is printed by the script but
**must not be read as physical** — §4.3.10(1) established that 872077's
prefill-overlap label is instrumentation-short because the job ran
`PDMUX_TRACE_FORCE_PREFILL=0`.

## 6. What is still open (unchanged by this check)

§0's **(i)** "872077's `decode_sms == 16` is not a real 16-SM hardware
execution" vs **(ii)** "C2's 28–31 ms is a property of the
`[16,16,76-idle]` + saturated-keepalive cell composition" — **both remain
open**, as pre-registered. Separating them needs a direct measurement.

Relevant prior art for the next step: job **864669** (2026-07-27) already
established **`DECODE_SM_HONORED`** — but only at the **isolated API layer**
(standalone matmul, pairwise green-context creation, no engine, no cudagraph,
no co-resident prefill): `B(92,16)` eff **17.7**, `B(16,44)` 55.7 ≈
`B(64,44)` 55.6, `B(16,92)` 110.7. The residual §4.3.11 flags is narrower —
the **engine context** (3–4 groups created together by
`initialize_stream_groups`, cudagraph-captured, prefill co-resident). So an
engine-context SM-grant probe is **not** a repeat of 864669, and gate S1
(§4.3.13, ~1 GPU-hour, harness extension required) remains the instrument
that separates (i) from (ii) behaviourally.
