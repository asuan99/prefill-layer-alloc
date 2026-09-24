# PART-residency census (2026-09-21)

> **참고용 기술 집계 · 판정 아님 · 어떤 결정 규칙에도 연결되지 않음 · GPU 지출 0 · 새 성능 판정 0건.**
> (Descriptive reference aggregation only. Not a verdict. Not wired to any decision
> rule. No arm ranking, no operating point, no SLO/goodput scoring. Read-only
> re-aggregation of telemetry already on disk.)

This directory answers exactly one **descriptive** question over the existing
archive: *how much of each run actually sat in a partitioned (PART) green-context
state, and how much of that PART time was co-resident with prefill work?*

It exists to put three self-diagnoses that the project already made separately —
E2C-21 (86–89% of the decode-busy weighted time the treatment moves is prefill-idle),
2F9, and the λ0 finding that the "fixed D44" cell realized D44 only a minority of the
time — onto **one common axis with one stated set of conventions**. It introduces no
new estimand, no new claim, and no comparison between arms.

Canonical documents (`PROJECT_STATUS.md`, `reports/CONSENSUS.md`, `reports/paper/*`)
are **not** touched by this work. HE0, the layer-type negative result, the policy
ordering and the Claim D/E grades are unchanged and untouched.

---

## 1. What was read

* Root: `workspace/engine-port/results/`
* Population: every `*.jsonl` larger than 1 KiB whose first 200 KB contain a
  `runtime_snapshot` record → **1077 files, 14 campaigns, 33,115,587 snapshots,
  235,387 s (65.4 h) of summed telemetry span.**
* Exact file list: `file_list.txt` (relative to the results root).
* Parse errors: **0**. Files with zero non-startup snapshots: **3**
  (`r2_correctness/job_907032/tel_TD1.jsonl`, `.../tel_TD2.jsonl`,
  `s0_deconfound/s0dc_M_C4096_P16D44_865006_telemetry.jsonl`).

### Coverage gap, stated plainly: there is **no W1–W9 coverage**

The task asked for a per-workload (W1–W9) breakdown. **No telemetry file in the
archive carries a W1–W9 workload identity.** `grep -rlE '"workload_id":\s*"W[1-9]"'`
over all of `results/` returns **0 files**. The `workload_id` field that telemetry
*does* carry is a **campaign-local cell label** (`pdmux_split34`, `g16_d44`,
`lam0_a_r4`, `c2r_Ha8_C1024_d92_blk3`, …), not the `benchmarks/pdmux_eval/workloads.py`
W1–W9 taxonomy. The single `W1` string found elsewhere
(`cp_baseline/w1_probe_*/w1_analysis.json`) belongs to a different, campaign-local
"W1 probe" whose cells record `"boot_ok": false, "scored": false` and which emits no
`runtime_snapshot` at all.

So the grouping used here is **(campaign, architecture, campaign-local `workload_id`)**,
taken from telemetry *content* (not from filenames). A W1–W9 residency comparison
cannot be produced from this archive and is **not** produced here. Nothing was
substituted to fill the gap.

## 2. Tooling

`residency_census.py`. It **imports** the canonical sample predicates from
`../r2_eval/e2_sticky_prereg/e2_realized_mix.py` rather than copying them:

* `_is_snapshot`    — `event == "runtime_snapshot"` and `phase != "startup"`
* `_is_decode_busy` — `decode_running_batch_size > 0` (that module's ADDENDUM A2
  literal; the neighbouring `decode_batch_size` is ≥ 1 on idle spin, and selecting it
  once flipped an E2 clause — mutation arm **M2** below re-demonstrates it)

The census *generalises* those from "fraction that is `stream_index == 2`" to "PART vs
FULL over the realized `(prefill_sms, decode_sms)` pair", and **streams** instead of
materialising the snapshot list (single archives reach 800 MB).

## 3. Conventions (every number below carries its convention name)

A percentage without its convention name is not citable here — the same λ0 cell reads
13.5 % or 13.64 % depending on convention, and the convention-less value became
citation-banned (E2C-8′). Populations:

* **population A** = all non-startup `runtime_snapshot` records in the file
* **population B** = population A restricted to `_is_decode_busy`

| name | population | weight |
|---|---|---|
| `R-cnt-all`   | A | uniform |
| `R-time-all`  | A | forward gap `t(i+1) − t(i)` on the non-startup sequence; the file's **last** snapshot has no defined weight and is **dropped** |
| `R-cnt-busy`  | B | uniform (same family as e2 `E-cnt`) |
| `R-time-busy` | B | forward gap, numerator **and** denominator restricted to decode-busy *left* snapshots (same family as e2 `E-time`) |
| `R-iter-busy` | B | `Δ(decode_iterations)` between adjacent non-startup snapshots, attributed to the **left** snapshot, gated on the left snapshot being decode-busy (same family as e2 `E-iter`) |

**Why five and not one.** A count statistic rides a **non-uniform sampling grid** (the
E2 authors measured 532 ms vs 208 ms between decode-busy snapshots inside one cell),
and a time-weighted statistic **imposes an interval on an instantaneous state**
(`CONSENSUS §3 항목120`, the C-R confound: 64 instantaneous states × 0.406 s once
inflated a figure to 25.70 s). Neither is safe alone. All five are emitted side by
side and **their disagreement is part of the record, not something to resolve here.**

**State predicate.** `PART ⟺ prefill_sms > 0 AND decode_sms > 0`, read from the
**realized** per-snapshot fields, never from a target/config value (Stage 0's "D108"
anchor was realized 16 SM; the λ0 "fixed D44" cell realized D44 only a minority of the
time). `FULL ⟺` exactly one side is 0, i.e. `108/0` or `0/108`. This is a
**snapshot-level proxy** for the engine's per-step `split_prefill_batch is not None`
condition (`multiplexing_mixin.py:880-893`) and **cannot resolve states shorter than
the sampling interval**; `median_gap_s` / `p95_gap_s` are emitted per file so the
resolution limit stays visible.

**Cohabitation predicates** (uniform weight, among PART-and-decode-busy snapshots):

* `P-act` : `prefill_active_batch_size > 0`
* `P-q`   : `prefill_queue_depth > 0`

E2C-4 caveat carried forward: in builds whose controller installs a split index on
every prefill span, `P-act` is near-100 % **by construction**. It is reported as an
instrument reading, never as evidence about the engine.

**Aggregation.** Denominators and spans are **summed**, never `max()`'d — the `max()`
form was a real harness bug that inflated a duration ~3×. Pinned by
`selftest_sum_not_max`.

**Validity quarantine.** A file whose snapshot timestamps are not monotonically
non-decreasing has no defined forward-gap weight (its `span_s` comes out negative).
Such files are **excluded from the two time conventions** and counted in
`files_time_axis_nonmonotonic`; they still contribute to the count- and
iteration-weighted conventions, for which ordering is irrelevant. **32 files**, all in
`p1_gates/gate2`, are in this state (exactly one negative step each — consistent with
two segments concatenated into one file).

## 4. Self-check (mutation testing with a no-mutation control)

`python3 residency_census.py --selftest` — three tiers, all currently passing.

* **Tier 1**, synthetic fixture, archive-independent. Expectations are hand-derived
  from the convention prose above (arithmetic written out inline in the source), not
  read back from the implementation. Arms:

  | arm | result |
  |---|---|
  | **M0 control — no mutation** | **SURVIVED** (required) |
  | M1 uniform weight replaces forward-gap | KILLED (`R_time_all` 50.0 → 60.0) |
  | M2 decode-busy := `decode_batch_size > 0` | KILLED (`snapshots_busy` 5 → 6) |
  | M3 PART := always true | KILLED (`R_cnt_all` 50.0 → 100.0) |
  | M4 PART := `stream_index != 4` (ignores `108/0`) | KILLED (`R_cnt_all` 50.0 → 83.33) |

  Plus a cross-path identity: the PART numerator must equal the sum of the
  `stream_index` histogram over the indices whose realized SM pair has both sides
  non-zero.
* **Tier 1b**: three identical rounds must aggregate to `span_s_sum = 3 × span`; a
  `max()` implementation would return `1 ×`.
* **Tier 2**: cross-implementation agreement. This second, streaming implementation
  reproduces `e2_realized_mix.EXPECTED` (the registered D44 table, whose own provenance
  is external verdict documents) **byte-exactly on all 7 λ0 cells** — busy `n`, E-cnt
  num/den, E-time num/den, E-iter pct, idx0, idx3, `split_transition` count — plus the
  resolution guard that `a_r4` and `b_r3` straddle the 20 % band.

  Caveat carried from lesson 254: a no-mutation control catches a harness that always
  fails, but **cannot** catch a false survival from an equivalent mutation.

## 5. Outputs

| file | content |
|---|---|
| `census_per_file.json` | one record per telemetry file (1077) — all five conventions, per-`stream_index` count/time/iteration histograms, realized SM histograms, cohabitation, dwell, gap percentiles, `split_transition` count, `run_id`s |
| `census_groups.json` / `.csv` | 406 groups = (campaign, architecture, campaign-local `workload_id`) |
| `census_groups_by_campaign.csv` | 14 campaign rollups |
| `census_groups_realized_sm.json` | realized `(prefill_sms, decode_sms)` shares, count- and time-weighted |
| `file_list.txt` | exact input list |

Reproduce:

```
python3 residency_census.py --selftest
python3 residency_census.py --file-list file_list.txt \
  --scan-root <repo>/workspace/engine-port/results \
  --jobs 24 --out-json census_groups.json --out-csv census_groups.csv \
  --out-files-json census_per_file.json
```

## 6. Campaign rollup (descriptive; no ranking implied, rows are alphabetical)

`PART %` columns are the share of the stated population/weight that was in a
partitioned realized SM pair. `P-act %` / `P-q %` are shares **of the PART-and-busy
snapshots**.

| campaign | files | span Σ (s) | snapshots | busy share % | R-cnt-busy % | R-time-busy % | R-iter-busy % | P-act % | P-q % | mean PART dwell (s, grid-limited) | split_transition events |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e1_traceforce | 8 | 1577.1 | 337,677 | 5.64 | 79.26 | 26.88 | 25.28 | 99.28 | 26.91 | 0.363 | 1531 |
| kernel_mech | 1 | 79.7 | 13,436 | 0.19 | 100.00 | 100.00 | 100.00 | 23.08 | 0.00 | 8.882 | 2 |
| longctx_conflict | 56 | 51,755.1 | 11,813,602 | 0.94 | 56.39 | 63.11 | 41.93 | 99.14 | 74.83 | 6.526 | 4576 |
| p1_gates | 272 | 28,621.8 | 6,906,767 | 2.54 | 47.12 | 49.24 | 41.12 | 71.28 | 25.49 | 1.188 | 9794 |
| r2_correctness | 29 | 1296.0 | 257,972 | 6.91 | 78.88 | 23.49 | 14.57 | 99.84 | 42.22 | 0.817 | 229 |
| r2_eval | 11 | 3885.8 | 299,505 | 3.17 | 20.48 | 35.68 | 20.50 | 89.98 | 69.99 | 2.482 | 3061 |
| s0_deconfound | 24 | 8729.8 | 1,248,674 | 1.65 | 87.95 | 94.39 | 92.51 | 99.76 | 95.96 | 88.193 | 224 |
| s2_sticky | 20 | 2663.3 | 285,094 | 2.89 | 99.89 | 99.92 | 99.93 | 11.80 | 5.24 | 34.859 | 7 |
| s8_frontier | 152 | 47,448.3 | 7,419,236 | 1.78 | 29.92 | 22.30 | 15.45 | 91.82 | 35.75 | 0.822 | 37,707 |
| s8_scaleup | 408 | 64,426.3 | 2,406,179 | 4.06 | 64.29 | 75.73 | 64.60 | 97.13 | 79.23 | 22.717 | 10,969 |
| s8p_prefill | 18 | 5860.5 | 161,816 | 5.39 | 78.35 | 86.39 | 78.69 | 91.08 | 2.99 | 7.515 | 9489 |
| slo_sched | 34 | 12,623.6 | 1,753,105 | 2.33 | 11.51 | 16.01 | 11.57 | 64.78 | 55.27 | 0.488 | 0 — **not instrumented**, see §7 |
| stage0_xctrl | 42 | 6110.9 | 190,976 | 2.50 | 87.31 | 97.98 | 88.78 | 99.69 | 90.57 | 14.458 | 1840 |
| sticky_smoke | 2 | 308.7 | 21,548 | 1.36 | 50.17 | 71.81 | 50.03 | 5.44 | 6.80 | 13.669 | 61 |

`R-cnt-all` (uniform over **all** snapshots, including decode-idle) is **not** in this
table on purpose: see §7.1.

## 7. Instrument-level observations that constrain how these rows may be read

**7.1 Population A is dominated by decode-idle spin.** Only **0.19–6.9 %** of
non-startup snapshots are decode-busy. Consequently `R-cnt-all` is 0.19–5.5 % in every
campaign while `R-time-all` is 11.6–91.1 % in the same campaigns — a 10–60× divergence.
Mechanically: the realized pair `108/0` holds **94–99 % of snapshot *count*** but only
**8–55 % of forward-gap-weighted *time***, i.e. snapshot emission is far denser in
`108/0` than in partitioned states (in `longctx_conflict` the mean forward gap is
0.0029 s in FULL vs 0.275 s in PART, ~95×). Any count-weighted residency figure over
population A is therefore mostly a measurement of spin sampling density. This is the
non-uniform-grid hazard the E2 header warned about, reproduced at archive scale.

**7.2 `R-time-*` vs `R-cnt-*` disagree by a lot, and the direction is not constant.**
e1_traceforce: `R-cnt-busy` 79.3 % vs `R-time-busy` 26.9 % (count > time).
r2_correctness: 78.9 % vs 23.5 %. But s8_scaleup: 64.3 % vs 75.7 % (time > count) and
sticky_smoke 50.2 % vs 71.8 %. **No single convention is adopted here.**

**7.3 `slo_sched` emits no `split_transition` event at all** (0 of 34 files contain the
string). Its `0` in the transition column means **not instrumented in that build**, not
"zero transitions". Coverage of the event across campaigns:
e1_traceforce 8/8, kernel_mech 1/1, longctx_conflict 52/56, p1_gates 202/272,
r2_correctness 27/29, r2_eval 11/11, s0_deconfound 12/24, s2_sticky 5/20,
s8_frontier 152/152, s8_scaleup 408/408, s8p_prefill 18/18, **slo_sched 0/34**,
stage0_xctrl 32/42, sticky_smoke 1/2.

**7.4 Transition aliasing.** `transition_alias_ratio` = `split_transition` events ÷
PART/FULL runs the snapshot grid could resolve. It ranges 0.06 (s2_sticky) to 7.54
(s8p_prefill). Where it exceeds 1, the snapshot grid is **provably** missing
transitions, so the dwell columns are upper bounds on true dwell. Dwell is labelled
`_gridlimited` for this reason.

**7.5 Non-monotonic time axis in 32 `p1_gates/gate2` files** (quarantined from the time
conventions, §3). Each has exactly one backward step; `span_s` computed naively is
negative (e.g. −650.1 s, −779.0 s), which is what a concatenation of two segments into
one file looks like. These files are a candidate for the "multiple rounds appended to
one file" structure that the duration-summation gate exists for.

**7.6 `P-act` behaves as the construction identity E2C-4 predicts in most campaigns**
(89–100 % in 8 of 14) but **not in all**: s2_sticky 11.8 %, sticky_smoke 5.4 %,
kernel_mech 23.1 %, p1_gates/`pdmux_split34` 38.7 %, slo_sched 64.8 %. Where it is far
below 100 %, the census is recording PART-and-decode-busy snapshots with **no resident
prefill batch**. This is reported as an observation about what the two predicates
select, and is **not** interpreted here.

## 8. Group-level rows worth having on the record (still descriptive)

Full table: `census_groups.csv` (406 rows). Some rows that are useful mainly as
**instrument checks**:

* `p1_gates` / `pdmux_nosplit` (74 files, 39,084 busy snapshots): PART = **0.0000 %**
  under *all five* conventions, and no PART-busy snapshots at all. This is the
  census's negative control — the PART predicate resolves an arm that is known by
  construction never to partition.
* `p1_gates` / `pdmux_split34` (74 files): PART 99.68 / 99.72 / 99.69 %
  (cnt-busy / time-busy / iter-busy), `P-act` 38.66 %, `P-q` 14.90 %, mean PART dwell
  6.74 s, 564 `split_transition` events.
* `p1_gates` / `pdmux_split34_nosticky` (54 files): PART 34.83 / 44.59 / 35.65 %,
  `P-act` 92.95 %, mean PART dwell 0.53 s, **8391** `split_transition` events.
* `s2_sticky` (20 cells): PART ≈ 99.8–100 % under all conventions, with `P-act`
  **1.1–33.9 %** and `P-q` **2.2–15.5 %**.
* `r2_eval` / λ0 cells: the `a_*` ladder reads PART (R-cnt-busy) 4.5 / 6.7 / 8.0 / 8.8 /
  8.6 / 8.3 % for `a_r0…a_r4_s2`, while the `b_*` ladder reads 43.1 / 66.9 / 87.0 /
  90.7 / 90.7 %. Mean PART dwell spans 0.28 s (`a_r0`) to 285.98 s (`b_r3_s2`).
* `slo_sched` `g16_d16 → g16_d74`: PART (R-cnt-busy) 6.20 / 7.14 / 8.11 / 9.56 / 11.66 /
  15.67 / 21.46 %, with `P-act` 28.94 → 84.13 % across the same ladder.
* `stage0_xctrl` `*_D108` cells (9 groups, 1 file each): PART 79.6–96.9 % under
  R-cnt-busy and **0** `split_transition` events, and the realized SM pair holding
  most of the time-weight is **`92/16`, not `0/108`** — 48.4 % (`T_C4096_D108`) to
  90.9 % (`H_C16384_D108`), with `108/0` taking 8.7–49.4 % and `0/108` only 0.3–2.2 %.
  This re-expresses, on this census's axis, the target-vs-realized finding already on
  the record for Stage 0 (the "D108 anchor" was realized 16 SM). It is a
  re-observation of an established fact, **not** a new finding, and it changes nothing
  about the Stage 0 verdict, which is already corrected in the canonical documents.

Realized-SM shares (count- and time-weighted, per campaign and per group) are in
`census_groups_realized_sm.json`. In **every** campaign the count-weighted mode is
`108/0`; the time-weighted mode varies (`0/108` in 5 campaigns, `108/0` in 4,
`16/92` in 2, `92/16` in 2, `16/16` in 1).

## 9. 이 집계로 할 수 없는 말 — what this census cannot say

1. **No policy, arm, split, campaign or workload is better, worse, more or less
   favourable than any other.** No ranking was computed and none may be inferred from
   row order (rows are alphabetical).
2. **Nothing about goodput, TTFT, ITL, SLO attainment, throughput or capacity.** No
   request-level metric was read; `analyze.py` was not invoked; no `run.json` was read.
3. **No causal statement.** PART residency is not attributed to sticky mode, to the
   controller, to the SM grid, to the model, or to the load. Arms differ in many
   uncontrolled ways across these archives (build, cudagraph state, model, campaign
   commit, SLO, rate), and none of that was checked, because none of it needs to be
   checked for a descriptive count.
4. **No `n`, no variance, no CI, no effect size.** Groups pool files of unequal length
   with no pairing and no seed structure. Nothing here is powered for any comparison,
   and none is offered.
5. **No claim that PART residency "should" be higher or lower**, and no operating point,
   threshold, band or acceptance criterion. Connecting any number here to a decision
   rule would create a new estimand requiring pre-registration and a rules-layer audit.
6. **No sub-sample-interval resolution.** Where `transition_alias_ratio > 1` the grid
   demonstrably misses transitions; dwell values are upper bounds.
7. **Not a single canonical percentage per cell.** Five conventions are reported
   precisely because they disagree; quoting one without its convention name reproduces
   the E2C-8′ failure.
8. **No W1–W9 statement of any kind** — the archive contains no W1–W9 telemetry (§1).
9. **The 3 zero-snapshot files and the 32 non-monotonic files are excluded as stated**,
   not repaired; no attempt was made to reconstruct their time axes.
10. **Nothing in the canonical hierarchy is modified or superseded by this directory.**
