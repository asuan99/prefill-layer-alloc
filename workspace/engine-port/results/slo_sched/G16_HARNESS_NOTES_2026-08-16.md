# G16 harness notes (2026-08-16) — methodology gate #34, stage 2

**Revision**: this is now the **post-addendum-B(-5(a)) fix** version — a
second pass on top of the post-addendum-A fix described below. The stage-2
(harness) audit's **second** re-audit (rev3's trailing `【addendum B】`)
returned **GO** for smoke submission, conditional on 5 pre-block-1 caveats
listed in **B-5(a)**. This document is updated **in place, again** (same
"no separate vN file" convention as the addendum-A round) to cover BOTH
rounds — the addendum-A crosswalk further down is unchanged and still
accurate; a new "Addendum B-5(a) crosswalk" section below it covers this
round. **Items 2/3/4/5 of B-5(a) (N1/N2/N3/N7) are done by this fix; item 1
(commit) is split — the DETECTION logic (untracked/dirty check) is done here,
the commit ITSELF is out of scope (main session, after user approval, per
this task's own instructions).**

**Scope of this document**: the harness only (artifact production + assert).
Rules are `PREREG_G16_RULES_REV3_2026-08-16.md` **including both
addenda (A and B)**. Estimand computation (rules §4/§6, the `Δ`/`Δ_SLO`
decision rules) is the **next stage's analyzer** using `pdmux_eval`,
explicitly out of scope for this harness.

**GPU spend so far: 0.** Nothing was submitted (`sbatch --test-only` only, and
`--test-only` jobs confirmed absent from `squeue` after each check) —
unchanged across both the addendum-A and addendum-B rounds.

## Files touched by this fix

Addendum-A round (unchanged from before):
- `g16_grid.sbatch` — the harness, rewritten.
- `g16_assert.py` — H2'/H3'-a/H6' assertion logic, rewritten (old H2/H3/H6
  functions removed, not deprecated-in-place — see "why removed, not kept"
  below).
- `g16_arm_order.py` — unchanged logic, one doc/print addition (A-4).

Addendum-B(-5(a)) round (this pass, new):
- `g16_grid.sbatch` — 4 more edits (item 1 detection, N1, N2, N3 — see
  crosswalk below). `g16_assert.py`/`g16_arm_order.py` **untouched** this
  round (none of the 4 items needed assertion-logic or arm-order changes).
- `g16_n1_regression_test.py` — **new**, standalone regression test for N1
  (extracts and sources the real `write_sidecar()`/N1-block bash text out of
  `g16_grid.sbatch`; no reimplementation).
- `g16_n3_dryrun_check.py` — **new**, standalone dry-run check for N3
  (extracts and sources the real bulk-unset block; simulates a leaked
  `--export=ALL` environment).
- This file.

`sharegpt_vary_bench.sbatch` (the R2 reproduction path) remains **untouched**
(both rounds).

---

## Addendum-A defect list (a) → resolution crosswalk

The task that produced this fix listed 10 "반드시 고칠 것" items, itself a
restatement of addendum A's own "감사 (a) 목록". All 10 resolved:

| # | Item | Where resolved |
|---|---|---|
| **1** | ★H3 → H3' tool swap (the #1 NO-GO cause): drop `PIN_GATE=0.80`/`realized_pin_check.py`-frac gating entirely; gate ONLY on `smsplit_realized_probe.probe_one` `exact`; demote dwell diagnostic to `analyze.controller_summary` time-weighted, record-only | `g16_grid.sbatch`: H3'-a block (`REALPROBE_JSON` heredoc + `g16_assert.py realized_probe`), H3'-b block (`CTRLSUMMARY_JSON` heredoc, no gating variable reads its RC). `g16_assert.py`: `assert_realized_probe` replaces `parse_pin_verdict`/the old `pin` CLI subcommand entirely (not kept as dead code — see below). `PIN_CHECK_PY`/`PIN_GATE` variables removed from the sbatch. K10 stated in both files' comments. |
| **2** | H2' — telemetry assert = `event=="runtime_snapshot" ∧ phase=="benchmark" ∧ "decode_sms" in e` count ≥ N; wire `TELEM_RC`/`FIELDS_*_RC` to a real failure path | `g16_assert.py`: `assert_runtime_snapshots` (replaces `assert_telemetry_nonempty`). `g16_grid.sbatch`: called as `g16_assert.py runtime_snapshots "$PDMUX_TELEMETRY_PATH" "$MIN_SNAPSHOTS"`; the `ARTIFACT_OK` block after `G16_ARM_SUMMARY` ANDs `TELEM_RC`/`FIELDS_LO_RC`/`FIELDS_HI_RC` and, if any is nonzero, increments `BOOTS_FAILED`, appends to `FAILED_ARMS`, and calls `mark_failed_artifacts` — previously these RCs were computed and printed but never read by any `if`. |
| **3** | ★H13 per-boot JSON sidecar, all addendum A-6 fields, per-round `(phase, t_start, t_end)` | `g16_grid.sbatch`: `write_sidecar()` function + per-round `T_LO0/T_LO1/T_HI0/T_HI1` timestamps appended to `${RUNID}_rounds.jsonl.tmp`, consumed by the sidecar builder. Called at all 3 boot outcomes (`boot_failed`, `bench_failed`, `completed`) so failed boots get a (partial) sidecar too, not just successful ones. |
| **4** | H6' — `len(ttfts)>0`, `len(itls)==len(ttfts)`, `duration>0`, not-null | `g16_assert.py`: `assert_fields`, rewritten (see diff in file; adds the 4 new checks after the pre-existing key-presence check). |
| **5** | H14 — expand env unset list | `g16_grid.sbatch`: the `unset` line inside the arm loop now lists `PDMUX_STICKY_PARTITION PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER PDMUX_LA_COORD PDMUX_MODEL_PROFILE PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY` in addition to the pre-existing 5. |
| **6** | H15 — FULL mode `NP=200`/`ROUNDS=3` hard assert | `g16_grid.sbatch`: `if [ "$NP" != "200" ] || [ "$ROUNDS" != "3" ]; then ... exit 1; fi` inside the `else` (FULL) branch, right after the `${NP:-200}`/`${ROUNDS:-3}` defaults are applied (which only catch *unset*, not *wrong*, hence the explicit assert). |
| **7** | H16 — remove/rename 0-byte `JLO`/`JHI`/telemetry from failed boots | `g16_grid.sbatch`: `mark_failed_artifacts()`, called from the `BOOT_FAILED`, `BENCH_FAILED`, and `ARTIFACT_INVALID` paths. Verified with synthetic 0-byte/non-empty fixtures (see "Verification performed" below) — only the genuinely-empty file was renamed. |
| **8** | H17 — separate smoke tag to `smoke1` | `g16_grid.sbatch`: `BLOCK_TAG="smoke1"` in the `SMOKE` branch (was `blk1`); `SMOKE4`'s filename check updated to match `*smoke1_d74_boot1*`. This is a **controlled resolution of an apparent §9/addendum wording mismatch**, not a silent rule change — see "Rule tension found and how it was resolved" below. |
| **9** | H18 — remove first-position cold-cache | `g16_grid.sbatch`: **chose the job-local warm-up boot**, not the shared-`TRITON_CACHE_DIR` alternative. Reasoning below ("H18 choice and why"). |
| **10** | Add d64 to the smoke arms | `g16_grid.sbatch`: `ARM_ORDER="d44 d64 d74"` (was `"d44 d74"`), `N_EXPECT_ARMS=3`. A new `SMOKE2B` check (A-7 item 8, second half) verifies d64's `exact` flag the same way `SMOKE2` verifies d74's. |

**All 10 resolved. 0 unresolved from the audit's own (a) list.**

> ⚠️**Row 5 (H14) above is now SUPERSEDED** by this round's N3 fix (see next
> section) — the 9-name curated `unset` list it describes was replaced by a
> blanket `unset "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"`. Left in place here
> as the historical record of what addendum A actually shipped (which is
> exactly the incomplete list N3 found missing 7 names from); do not read
> row 5 as describing the current `g16_grid.sbatch` text.

---

## Addendum B-5(a) defect list → resolution crosswalk (2026-08-16, 2nd pass)

rev3's trailing `【addendum B】` §B-5(a) listed **5** "블록 1 제출 전 필수"
items after the 2nd-round harness re-audit returned **GO**. The task that
produced this pass assigned item 1's commit action to the main session
(post-user-approval) and this session's job to (i) item 1's **detection**
logic only, plus (ii) items 2–5 in full.

| # | Item | Where resolved | Verified how |
|---|---|---|---|
| **1 (detection only)** | Untracked/dirty check for the 3 harness files — the pre-existing `G16_A5_COMMIT_CONDITION_WARNING` only fires when `git rev-parse` fails outright (not a git checkout), which is blind to "is a checkout, but these files are untracked/dirty" | `g16_grid.sbatch`: new block right after the pre-existing rev-parse check, computing `G16_HARNESS_UNCOMMITTED`/`G16_UNCOMMITTED_FILES` via `git ls-files --error-unmatch` (untracked) + `git status --porcelain` (dirty) per file. FULL mode: loud banner, **non-fatal** (no `exit`). SMOKE mode: plain one-line note. Neither mode blocks the run — B-1(ii)'s "스모크는 커밋 없이도 유효하다" judgment is read to extend to FULL too, since `harness_sha256` content-addresses the harness regardless of git state either way. | 4 standalone scenarios run against the extracted block (not `--test-only`, which doesn't execute the script body): (a) real repo, files untracked (this project's actual live state), MODE_TAG=full → `G16_HARNESS_UNCOMMITTED=1` + loud banner + exit 0; (b) same, MODE_TAG=smoke → `=1` + plain note + exit 0; (c) throwaway git repo with the 3 files committed and clean → `G16_HARNESS_UNCOMMITTED=0`, no output; (d) same repo, one file edited after commit → `=1` + `dirty[ M ...]` detail. All 4 correct. |
| **2 (N1)** | `write_sidecar` ran (unconditionally, status="completed") BEFORE the `ARTIFACT_OK` judgment, so an artifact-invalid boot's sidecar still said "completed" | `g16_grid.sbatch`: `write_sidecar` call **moved** to after `ARTIFACT_OK` is computed, **and** a new status value `artifact_invalid` introduced (both remedies the rules text offered, applied together) — `ARTIFACT_OK` block now bracketed by `G16_N1_BLOCK_START`/`G16_N1_BLOCK_END` sentinel comments for test extraction. | `g16_n1_regression_test.py` (new, GPU 0): sources the real `write_sidecar()` + N1 block out of `g16_grid.sbatch` in a synthetic bash subprocess. 3 cases, all PASS: (A) post-fix artifact-invalid boot → `status="artifact_invalid"`, not `"completed"`; (B) post-fix healthy boot → `status="completed"` (no regression); (C) reproduces the pre-fix call order against the SAME write_sidecar() → confirms `status="completed"` + `fields_rc.hi=1` co-occurring (this IS the bug B-5(a)#2 named — proves it lived in the call site, not in `write_sidecar()` itself). |
| **3 (N2)** | Warm-up bench output (`ttfts`/`itls`/`duration` schema, NP=2) had no sidecar/assert and lived in the same directory/namespace as real campaign artifacts | `g16_grid.sbatch`: warm-up server log + bench output moved into a new subdirectory `$WARMUP_DIR="$HERE/g16_warmup_artifacts"` (axis 1: out of the flat directory a non-recursive `$HERE/*_LO.jsonl`/`*_HI.jsonl` glob would scan), **and** the bench output filename gets a `.IGNORE` suffix appended (axis 2: defeats even a hypothetical recursive glob, since the literal suffix no longer ends in `_LO.jsonl`/`_HI.jsonl`). Both applied (either alone satisfies the rule text's "이거나"). | Synthetic-file glob test (scratchpad, not committed): created 1 real-shaped `..._LO.jsonl`/`..._HI.jsonl` pair at `$HERE`-equivalent top level + 1 warm-up srv.log + 1 warm-up `..._bench.jsonl.IGNORE` under `g16_warmup_artifacts/`, then ran `glob.glob("*_LO.jsonl")`/`"*_HI.jsonl"` both flat and recursive (`**`) — the real pair matched both ways, the warm-up files matched **neither**, in **neither** glob style. |
| **4 (N3)** | H14's curated `unset` list (9 names, addendum A) already missed 7 physical-config vars on its first pass (`SGLANG_ZAMBA_TIMING`/`_PREFILL_KNEE`, `PDMUX_FIXED_PREFILL_SM_FILE`, `PDMUX_HOLB_PATH`, `PDMUX_AGN2_SM`, `PDMUX_LA_SM_MAP`, `PDMUX_LA_FLOOR_SM`) | **Chose the recommended bulk approach** (rules text: "권장: `PDMUX_*`/`SGLANG_ZAMBA_*`를 일괄 unset"). `g16_grid.sbatch`: `for _pv in "${!PDMUX_@}" "${!SGLANG_ZAMBA_@}"; do unset "$_pv"; done`, immediately followed by the 3 re-exports the campaign needs (`PDMUX_TELEMETRY_PATH`/`PDMUX_RUN_ID`/`PDMUX_WORKLOAD_ID`) — order matters (bulk-unset must run BEFORE the re-export, or it erases it too). Block bracketed by `G16_N3_BLOCK_START`/`G16_N3_BLOCK_END` for test extraction. **Vars this campaign actually needs** (grep-confirmed, see below): exactly those 3 — no R2 policy, dual-worker, SLO scheduler, or LA/AGN coordination env var is read on this campaign's code path (`manual_divisions` comes from the yml via `--pdmux-config-path`, a CLI flag). | `git grep`-style scan of the full engine tree (`grep -rhoE "PDMUX_[A-Z0-9_]+"`/`"SGLANG_ZAMBA_[A-Z0-9_]+"`, 2026-08-16) found **55 `PDMUX_*` + 4 `SGLANG_ZAMBA_*`** env vars total (full names list below) — confirming a curated list is structurally incapable of staying complete (the addendum-A list named 9 of 59, missed 7 of the ones that mattered for THIS campaign, and would still miss most of the other 50 if extended piecemeal). `g16_n3_dryrun_check.py` (new, GPU 0): sources the real N3 block, pre-sets 11 leaked `PDMUX_*`/`SGLANG_ZAMBA_*` vars (including the 7 N3 itself found missing, plus a few not on any prior list, plus **stale values for the 3 needed vars** to catch an order bug) via a simulated `--export=ALL` leak, runs the block, dumps every `PDMUX_*`/`SGLANG_ZAMBA_*` var afterward. Result: all 11 leaked vars gone; the 3 needed vars present with their FRESH (not stale) values. |
| **5 (N7)** | 2026-07-15/18 sgptv grid ran WITHOUT telemetry (`sharegpt_vary_bench.sbatch` has no `PDMUX_TELEMETRY_PATH`), so rev3 §9's "d16/d24 = drift anchor to that grid" claim conflates build drift with instrumentation-overhead drift | **Doc-only, no code change** (per the task and per B-2 itself, which is a rules-layer scope note, not a harness defect). See "N7 — drift-anchor scope note" section below. | `grep -n "PDMUX_TELEMETRY_PATH" sharegpt_vary_bench.sbatch` → no match (confirmed 2026-08-16, this session). |

**Full `PDMUX_*`/`SGLANG_ZAMBA_*` name list (55 + 4, grepped from
`/scratch/ehmoon/whlee/sglang_engine_dev/python`, 2026-08-16)**, for anyone
auditing the N3 bulk-unset's scope claim:

```
PDMUX_AGN2_SM PDMUX_DUAL_WORKER PDMUX_DUAL_WORKER_TRACE PDMUX_DUAL_WORKER_TRACE_EVERY
PDMUX_ENGINE_COMMIT PDMUX_FIXED_DECODE_SM PDMUX_FIXED_DECODE_SM_FILE
PDMUX_FIXED_PREFILL_SM_FILE PDMUX_HOLB_DRAIN_BUDGET PDMUX_HOLB_EMIT_GAPS
PDMUX_HOLB_MAX_INFLIGHT PDMUX_HOLB_PATH PDMUX_HOLB_RUN_ID PDMUX_HOLB_SUMMARY_EVERY_S
PDMUX_HOLB_WORKLOAD_ID PDMUX_LA_COORD PDMUX_LA_COORD_OPT PDMUX_LA_COORD_PF
PDMUX_LA_COORD_V4 PDMUX_LA_FLOOR_SM PDMUX_LA_RESERVE PDMUX_LA_SM_MAP PDMUX_L_REF_TOK
PDMUX_MODEL_PROFILE PDMUX_PF_CHUNK PDMUX_PREFILL_TP_GROUP PDMUX_P_TP
PDMUX_QDEPTH_TARGET PDMUX_R2_FALLBACK_DSM PDMUX_R2_FIXED_DSM PDMUX_R2_POLICY
PDMUX_RUN_ID PDMUX_SLO_ANCHOR_IDX PDMUX_SLO_DWELL PDMUX_SLO_EMA PDMUX_SLO_FEAS_GATE
PDMUX_SLO_FEAS_MARGIN PDMUX_SLO_FEAS_OCC PDMUX_SLO_LFF PDMUX_SLO_MODE
PDMUX_SLO_PF_URGENCY PDMUX_SLO_PIN PDMUX_SLO_SAT_DEEP PDMUX_SLO_SAT_LATCH
PDMUX_SLO_SAT_MARGIN PDMUX_SLO_SAT_WIN PDMUX_SLO_SCHED PDMUX_SLO_SPAN_TYPE
PDMUX_STICKY_PARTITION PDMUX_TELEMETRY_PATH PDMUX_TPOT_HI PDMUX_TPOT_LO
PDMUX_TPOT_SLO_MS PDMUX_TRACE_FORCE_PREFILL PDMUX_TRUE_DUAL_WORKER
PDMUX_TTFT_SLO_MS PDMUX_WORKLOAD_ID
SGLANG_ZAMBA_CLOSURE_MIN SGLANG_ZAMBA_PREFILL_KNEE SGLANG_ZAMBA_TIMING
SGLANG_ZAMBA_TIMING_EVERY
```

**All 5 resolved** (4 fully by this pass; item 1 split as scoped by the task
— detection done here, the commit action itself remains a human/main-session
step). **0 unresolved from B-5(a)'s own list.**

## N7 — drift-anchor scope note (rules-layer, doc-only)

Per rev3 addendum §B-2: `multiplexing_mixin.py:470-473`→`dual_worker.py:
566-602 observe_scheduler` runs on every scheduler sync when telemetry is
enabled (write-subsampled 1/32, but the read/dispatch path itself is
`O(waiting_queue + batch)` python on the scheduler thread — which is the
**critical path** under `--disable-overlap-schedule`, the flag this
campaign's server invocation uses verbatim from `sharegpt_vary_bench.sbatch`).
**The 2026-07-15/18 sgptv grid ran WITHOUT telemetry** — confirmed this
session: `grep -n "PDMUX_TELEMETRY_PATH" sharegpt_vary_bench.sbatch` returns
no match, and rev3 §B-2 independently confirms 0 telemetry files exist under
`slo_sched/` from that date range. This G16 campaign, by contrast, runs
`PDMUX_TELEMETRY_PATH` on **every** boot (H2/H2').

**Consequence for result interpretation** (not a harness defect — no code in
this campaign changes as a result): rev3 §9's framing of d16/d24 as "the
**only** drift anchor to 2026-07-15/18" is only correct if telemetry
overhead is zero. It is not zero, and — per §B-2 — its cost is monotone in
queue depth, which correlates with the decision axis (decode SM). So:

1. **This campaign's own internal comparisons (its 1st-order decision
   quantities `D_ttft`/`D_itl`/`Δ`/`Δ_SLO`) are unaffected** — every arm in
   every block runs under the identical instrumentation, so within-campaign
   `argmin`/`Δ` comparisons have no telemetry-overhead confound.
2. Any comparison of this campaign's **absolute** `M_ttft`/`M_itl` values
   against the 2026-07-15/18 numbers is confounded: the delta reflects
   **build drift (C2-R-confirmed, manifest 11→15) PLUS instrumentation
   overhead**, not build drift alone — the two cannot be separated from
   this campaign's data by itself.
3. The rules document's own §9 wording ("d16·d24는 07-15/18과의 유일한 드리프트
   앵커") should be read with that combined-cause caveat when the eventual
   result document cites it — flagged here so that citation doesn't silently
   drop the caveat rev3 §B-2 itself already states.

No harness change follows from this — it is purely an interpretation caveat
for whoever writes the result document downstream, which is why this section
is prose, not a diff.

---

## H1–H18 crosswalk (full, post-fix)

| # | Requirement | Where implemented | Verified how |
|---|---|---|---|
| H1 | Boot 1 = measurement 1; filename `blk<N>_<arm>_boot<idx>` (or `smoke1_<arm>_boot<idx>`) | `BOOT_IDX=1` fixed, never incremented; `RUNID` built once per arm-loop iteration | Unchanged from pre-addendum version; `sbatch --test-only` accepted |
| H2' | Telemetry assert = runtime_snapshot(phase=benchmark, has decode_sms) count ≥ N, wired to failure | `g16_assert.py:assert_runtime_snapshots`; `g16_grid.sbatch` `ARTIFACT_OK` block | `python3 g16_assert.py selftest` — 5 branches incl. the task-mandated "0 matching events despite non-empty file" synthetic case, all correct |
| H3'-a | Driver-realized coordinate probe, ONE call per boot, gates on `exact` alone | `smsplit_realized_probe.probe_one` heredoc + `g16_assert.py realized_probe` | selftest exercises exact=true/false/error-key/missing-key/missing-file, all correct; `(44,64)`/`(34,74)` both covered by smoke (item 10) |
| H3'-b | Time-weighted dwell diagnostic, record only, never gates | `pdmux_eval.analyze.controller_summary` heredoc, RC never read by an `if` | Manually verified `controller_summary` importable+callable with 0 torch/GPU deps (stdlib only); dry-run against a 3-event synthetic telemetry list produced the expected `residency_fraction`/`dwell_s` shape |
| H4 | Arm order = forward/reverse pairs, fixed as a job argument | `g16_arm_order.py` (unchanged logic) | `python3 g16_arm_order.py --verify` → `VERIFY_OVERALL=PASS` + **new** A-4 disclaimer printed (this PASS is an algebraic identity, not independent evidence — see A-5 note below on the still-open commit condition) |
| H5 | Per-boot failure isolation + `BOOTS_FAILED` + record which arm/block failed | `BOOT_FAILED`/`BENCH_FAILED`/`ARTIFACT_INVALID` branches, each `continue`s | `bash -n` confirms no unreachable code |
| H6' | `len(ttfts)>0`, `len(itls)==len(ttfts)`, `duration>0`, no nulls, round count, LO/HI separate | `g16_assert.py:assert_fields` | selftest — 3 task-mandated synthetic FAIL cases + 4 pre-existing branches, all correct |
| H7 | `case` pattern covers `d64|d74` | `g16_grid.sbatch` arm-loop `case` (unchanged from pre-addendum) | `sbatch --test-only` with d64/d74 in the order accepted |
| H8 | `--comment` inside `#SBATCH` block | Line 10 | `check_sbatch_comment.py`: this file 1/1, full repo 130/130 conformant (no regression) |
| H9 | Engine manifest sha + git commit per block | `sync_engine_tree.sh` call + `git rev-parse` (unchanged) | Same as pre-addendum, plus now **also carries the harness sha (A-5(ii), see H-crosswalk row below)** |
| H10 | Per-boot co-tenancy snapshot | `cotenancy_snapshot()` (unchanged) | Unchanged, re-verified `bash -n` |
| H11 | 1 job = 7 arms sequential, walltime 03:00:00 | Unchanged structurally; **walltime budget re-checked** given the new warm-up boot (below) | `sbatch --test-only` accepted with 3h walltime |
| H12 | Analyzer's `Δ_SLO` ladder | Out of scope (next-stage analyzer) | N/A |
| **H13** | Per-boot JSON sidecar, all fields | `write_sidecar()` | Dry-run with synthetic env vars + a 2-line rounds.jsonl + a realized-probe JSON fixture produced valid, complete JSON with every addendum-A-6 field present (see "Verification performed") |
| **H2'** | (see above) | | |
| **H6'** | (see above) | | |
| **H14** | Env hygiene, expanded unset list | `unset` line in the arm loop | Grep-diffable against addendum A-6's literal list — all 8 named vars present, plus the 5 pre-existing ones |
| **H15** | FULL mode NP/ROUNDS hard assert | `if [ "$NP" != "200" ] ... exit 1` | Logic traced by hand; `sbatch --test-only g16_grid.sbatch 1` (no NP/ROUNDS override) still accepted, confirming the assert doesn't fire on the default path |
| **H16** | 0-byte artifact cleanup | `mark_failed_artifacts()` | Standalone synthetic test: 0-byte `_LO.jsonl` and `_telemetry.jsonl` renamed to `.FAILED`, non-empty `_HI.jsonl` left untouched (see "Verification performed") |
| **H17** | Smoke tag separated | `BLOCK_TAG="smoke1"` | `sbatch --test-only --export=ALL,G16_SMOKE=1 g16_grid.sbatch` accepted; `SMOKE4`'s case pattern updated in lockstep |
| **H18** | Cold-cache fix at position 0 | Job-local warm-up boot (see below) | `bash -n` confirms the warm-up block is well-formed; `${ARM_ORDER%% *}` extraction tested standalone for both FULL (`d16`) and SMOKE (`d44`) orderings |

---

## H18 choice and why (job-local warm-up, not shared `TRITON_CACHE_DIR`)

Addendum A-6 offered two options: **(a)** revert to the shared warm
`TRITON_CACHE_DIR` used by `sharegpt_vary_bench.sbatch:30`
(`$OUT/../../triage/.triton_cache`, no job-id suffix), or **(b)** a throwaway
warm-up boot at block start, keeping `TRITON_CACHE_DIR` job-unique.

**Chose (b).** Reasoning:

1. This project's own tooling explicitly flags a cache dir **shared across
   concurrently-running SLURM jobs** as a race hazard — that is *why*
   `g16_grid.sbatch`'s pre-existing (pre-addendum) comment already reads
   "병렬 array가 캐시 충돌... TRITON_CACHE_DIR를 job/run별로 분리" and cites the
   `s8_sweep.sbatch`/`batchcap.sbatch` precedent. Reverting to a job-shared
   path reintroduces precisely that hazard for this campaign.
2. rev3 §5 describes the 4 blocks as 4 separate SLURM jobs with node/date as
   a "기록 축 (SLURM 배정)" — i.e. nothing in the rules forces the 4 blocks to
   run sequentially or on the same node. If 2+ blocks are scheduled onto
   different nodes concurrently (plausible on an 8-node partition), a shared
   filesystem-path cache would let two `python -m sglang.launch_server`
   processes JIT-compile into the same Triton cache directory at once.
   `sharegpt_vary_bench.sbatch` invocations are already run this way in
   practice (1 job = 1 arm, historically submitted with staggered timing by a
   human), which is a different concurrency profile than "4 jobs that may all
   be eligible to start within the same scheduling window."
3. The warm-up costs one extra boot-equivalent of GPU time per block
   (roughly the cost of one more arm's boot, before its first `curl` 200 —
   no bench_serving beyond a 2-prompt priming call), which the H11 walltime
   budget already has margin for: rev3 §9's own measured per-boot cost is
   ≤12 min for the *existing* 5-arm grid (d16–d54) and the walltime is fixed
   at 03:00:00 for 7 real arms (≈84 min at that rate) + 1 warm-up
   (≈10–12 min) ≈ 96 min, well under budget even before accounting for the
   new arms' possibly-slower bench rounds (rev3 §9's own d74 1.92× caveat).
4. It equalizes ALL 4 block positions (position 0 in every block, not just
   block 1) with positions 1–6, since every block pays the warm-up
   independently — a shared cache would only remove the cold-cache cost for
   the *first* block to run, not for blocks 2–4 if their first arm happens to
   differ (which it does: block 3 starts with d34, block 4 with d54).

**This is a harness-design judgment call, not a rule reinterpretation** —
addendum A-6 explicitly offered both options and asked for the choice with
reasoning recorded here, which is what this section is.

---

## Rule tension found and how it was resolved (H17 / rev3 §9 condition 4)

rev3's **original** §9 (pre-addendum) pass condition (4) reads "파일명 규약
`blk1_d74_boot1`". Addendum A-6's **H17** explicitly requires separating the
smoke tag to `smoke1` because of the exact ambiguity that literal wording
creates (a real block-1 boot and a smoke boot both writing `blk1_d74_boot1`-
shaped filenames). This is not a silent contradiction: the addendum is later
than, and explicitly targets, that specific §9 wording — the task instructions
governing this fix state that "addendum이 덮어쓰는 부분만 바뀌었고 나머지는
유효하다" (only the parts the addendum overwrites change), and H17 is
addendum text that names this exact defect. **Resolution applied**:
`SMOKE4`'s filename check now matches `*smoke1_d74_boot1*` instead of
`*blk1_d74_boot1*`, tracking the `BLOCK_TAG` rename. This is flagged here
per the "report conflicts, don't silently resolve them" instruction, even
though it is judged to be addendum-authorized rather than a genuine
rule-vs-rule conflict — no other rule tension was found while implementing
addendum A-6/A-7.

---

## Why the old H2/H3 assertion functions were REMOVED, not kept alongside the new ones

`g16_assert.py`'s pre-addendum `assert_telemetry_nonempty` and
`parse_pin_verdict`/the `pin` CLI subcommand were deleted, not marked
deprecated-and-unused. Rationale: addendum A-1 identifies the old H3 gate
(`PIN_GATE=0.80` against `realized_pin_check.py`'s frac) as the harness's
**primary NO-GO cause** — leaving that logic present but merely uncalled
creates a live footgun for a future edit that "helpfully" wires it back in
(e.g. to add a stricter gate), silently reintroducing the exact defect this
fix closes. The old H2 check (`assert_telemetry_nonempty`) is subsumed by the
new H2' check in every respect a caller would want (any file that passes H2'
also passes the old H2, but not vice versa), so keeping both invited exactly
the "which one does the harness actually call" confusion addendum A-1 is
about. Both are fully superseded; `git log`/this notes file is the historical
record if the old logic is ever needed for comparison.

---

## Explicitly out of scope (not a compromise, a task-boundary decision)

- **H12 and all of rev3 §4/§6** (estimand computation, decision-rule
  evaluation, the 9-verdict table, the `Δ_SLO` ladder, and the addendum A-3
  drift-corrected estimator it requires the analyzer to compute) remain the
  next stage's `pdmux_eval`-based analyzer. This harness produces the raw
  artifacts (now including the H13 sidecar, which is what makes the A-3
  drift correction possible downstream) and asserts their structural
  validity, but computes no `M_ttft`/`M_itl`/`Δ`/`Δ_SLO`/drift-corrected
  estimate itself.
- **A-5(i) / B-5(a)#1's commit action itself** ("commit the 3 harness files
  before block-1 submission") is a process condition this script cannot
  perform from inside a job — it is a `git commit` a human runs, not
  something a SLURM job can do to its own submitting repo mid-flight. **The
  DETECTION side of this is now closed** (this pass): `g16_grid.sbatch` now
  runs an actual `git ls-files --error-unmatch` + `git status --porcelain`
  check per harness file (`G16_HARNESS_UNCOMMITTED`), not just the weaker
  rev-parse-failure fallback (`G16_A5_COMMIT_CONDITION_WARNING`, kept
  alongside it). **The commit action itself remains open** until a human
  runs `git commit` on `g16_grid.sbatch`/`g16_assert.py`/`g16_arm_order.py`
  (confirmed live as of this pass: all 3 files show `??` in `git status` —
  i.e. `G16_HARNESS_UNCOMMITTED` would currently read `1` if a block ran
  right now) — per this task's own instructions, the commit is out of scope
  for this fix (main session does it after user approval). Flagged here so
  it is not forgotten before block 1 is ever really submitted. Per rev3
  addendum B-1(ii), this does NOT block submission (smoke was re-audited GO
  without a commit precondition, and FULL mode's warning is non-fatal for
  the same reason — `harness_sha256` content-addresses the harness either
  way).

## Compromises / judgment calls (reported honestly)

1. **H4's "job 인자로 고정"** reading is unchanged from the pre-addendum
   version (A-5 judged the existing implementation sufficient, with the two
   conditions listed above, both now addressed: (i) is open/flagged, (ii) is
   done).
2. **Resource sizing** (`--cpus-per-task=8`, `--mem=80G`) — unchanged,
   matches `sharegpt_vary_bench.sbatch`.
3. **H3 disablement is per-arm, not a hard job abort** — unchanged in spirit
   under H3': a realized-partition mismatch (now: `exact != true`) does not
   stop the block, it excludes that one arm (`PIN_FAILED_ARMS`) while H2'/H6'
   failures now DO count toward `BOOTS_FAILED` (item 2 in this doc's
   defect-list crosswalk) — these are different failure classes on purpose:
   a coordinate mismatch is about *which arm this data belongs to*, while an
   artifact-validity failure is about *whether the data is usable at all*.
4. **`MIN_SNAPSHOTS` thresholds (20 for FULL, 5 for SMOKE) are harness-chosen
   constants, not specified by the rules document.** Addendum A-6 only
   specifies the predicate ("count ≥ N"), not N. Chosen conservatively low
   relative to the expected snapshot volume (default `trace_every=32`
   scheduler-sync sampling; a FULL-mode boot runs ~250s of combined
   LO+HI benchmark time across 3 rounds, a SMOKE-mode boot ~16s) so the
   assert is very unlikely to false-fail on a healthy boot while still
   catching a true "0 snapshots" defect. If the real per-boot snapshot count
   turns out far lower than expected (visible directly in every
   `G16_ARM_SUMMARY`/H13 sidecar `telem_rc` and in the smoke run's item-9
   `runtime_snapshot_benchmark_count` printout), this constant should be
   revisited before FULL-mode submission — flagging explicitly since it is a
   number the rules document did not hand down.
5. **Smoke arm count grew from 2 to 3** (task item 10 / addendum A-7 item 8),
   which raises the smoke GPU-time estimate from rev3 §9's "≈0.2 GPU-hr" to
   roughly **≈0.3 GPU-hr** (proportional, plus the new warm-up boot). Still
   well inside the "≥1 GPU-hr ⇒ pipeline-smoke-required" rule's own budget
   category.
6. **H13's `manifest_sha` field is the sha256 of the manifest FILE** (a
   single fingerprint over the combined engine + harness sha256 list), not a
   nested object of individual file hashes — the per-file granularity is
   already available in `harness_sha256` (harness files) and by dereferencing
   `manifest_sha`'s source file (engine files, unchanged H9 format). Chosen
   for sidecar compactness; if per-file engine hashes are needed downstream,
   they are one `sha256sum -c "$manifest"`-shaped file away, not lost.
7. **`fields_rc` in the H13 sidecar is `{"lo": ..., "hi": ...}`**, not a
   single scalar — the addendum's field list names it singular
   (`fields_rc`) but the harness computes it separately per LO/HI file
   (pre-existing `FIELDS_LO_RC`/`FIELDS_HI_RC` split, unchanged from the
   pre-addendum harness). Read as "the literal field name `fields_rc` holding
   whatever shape preserves the information the harness actually has" rather
   than "must be a bare int" — flagging the reading since the addendum text
   doesn't disambiguate.

## Verification performed — addendum-A round (all GPU-free)

```
bash --noprofile --norc -n g16_grid.sbatch              -> OK
python3 -m py_compile g16_arm_order.py g16_assert.py     -> OK
5 heredoc python blocks extracted from g16_grid.sbatch    -> all compile()
12 inline `python3 -c "..."` snippets extracted           -> all compile()
python3 g16_assert.py selftest                            -> 17/17 checks PASS,
  including all 4 task-mandated synthetic cases (all evaluate to the
  required FAIL/exclude outcome):
    1. {"ttfts":[],"itls":[],"duration":0.0}                  -> FAIL (excluded)
    2. ttfts len=200, itls len=3 (length mismatch)             -> FAIL (excluded)
    3. {"ttfts":null,"itls":null,"duration":null}              -> FAIL (excluded)
    4. telemetry with 0 runtime_snapshot(phase=benchmark,
       decode_sms-present) events despite non-empty file       -> FAIL (excluded)
python3 g16_arm_order.py --verify                         -> VERIFY_OVERALL=PASS
                                                               + A-4 disclaimer printed
write_sidecar()'s python body, dry-run with synthetic env vars/rounds.jsonl/
  realized-probe JSON                                      -> produced valid,
                                                               complete JSON with
                                                               every addendum-A-6
                                                               field populated
mark_failed_artifacts() bash logic, dry-run with a mock
  0-byte LO.jsonl + 0-byte telemetry.jsonl + non-empty HI.jsonl
                                                             -> only the two
                                                                0-byte files
                                                                renamed to
                                                                .FAILED; the
                                                                non-empty file
                                                                untouched
`${ARM_ORDER%% *}` (H18 warm-up-arm extraction), tested for
  both FULL ("d16 d24...") and SMOKE ("d44 d64 d74") orderings -> d16 / d44
  respectively, correct
A-5(ii) harness-sha-append block, dry-run against a synthetic
  manifest file                                             -> appended lines
                                                                are valid
                                                                sha256sum
                                                                format,
                                                                HARNESS_SHA_JSON
                                                                parses as JSON
check_sbatch_comment.py g16_grid.sbatch                    -> 1/1 conformant
check_sbatch_comment.py (full repo)                        -> 130/130 conformant
                                                               (no regression)
sbatch --test-only g16_grid.sbatch 1                        -> accepted (884224)
sbatch --test-only g16_grid.sbatch 2                         -> accepted (884226)
sbatch --test-only --export=ALL,G16_SMOKE=1 g16_grid.sbatch -> accepted (884225,
                                                                 884227 on rerun)
squeue -u $USER checked after each --test-only               -> none of the
                                                                above job ids
                                                                present (confirms
                                                                --test-only did
                                                                not queue anything)
```

No job was submitted; `--test-only` only assigns a hypothetical job id and
schedule estimate without queuing anything.

## Verification performed — addendum B-5(a) round (all GPU-free, this pass)

```
bash --noprofile --norc -n g16_grid.sbatch                -> OK (re-checked
                                                               after every edit)
5 heredoc python blocks extracted from g16_grid.sbatch      -> all compile()
                                                               (unchanged count
                                                               -- this round
                                                               added 0 new
                                                               heredocs)
12 inline `python3 -c "..."` snippets extracted             -> all compile()
                                                               (unchanged count)
python3 -m py_compile g16_arm_order.py g16_assert.py
  g16_n1_regression_test.py g16_n3_dryrun_check.py           -> OK
python3 g16_assert.py selftest                               -> 17/17 PASS
                                                               (unchanged --
                                                               this round did
                                                               not touch
                                                               g16_assert.py)

--- item 1 (untracked/dirty detection), 4 standalone scenarios, extracted
    block run against real/synthetic git states ---
(a) real repo, 3 files untracked (actual live state, 2026-08-16), MODE_TAG=full:
      G16_HARNESS_UNCOMMITTED=1 files=[ g16_grid.sbatch:untracked
        g16_assert.py:untracked g16_arm_order.py:untracked]
      -> loud FULL-mode banner printed, EXIT_CODE_WOULD_BE=0 (non-fatal)   PASS
(b) same repo/files, MODE_TAG=smoke:
      G16_HARNESS_UNCOMMITTED=1 -> plain G16_HARNESS_UNCOMMITTED_NOTE line,
      EXIT_CODE_WOULD_BE=0                                                 PASS
(c) throwaway git repo, 3 files committed clean:
      G16_HARNESS_UNCOMMITTED=0 files=[], no warning/note printed          PASS
(d) same throwaway repo, g16_grid.sbatch edited after commit:
      G16_HARNESS_UNCOMMITTED=1 files=[ g16_grid.sbatch:dirty[ M ...]]     PASS

--- N1 regression (g16_n1_regression_test.py) ---
A. post-fix, artifact-invalid boot (FIELDS_HI_RC=1):
     status='artifact_invalid' fields_rc.hi=1                              PASS
B. post-fix, healthy boot (all RCs 0):
     status='completed'                                                    PASS
C. pre-fix call order reproduced against the SAME (unmodified) write_sidecar():
     status='completed' fields_rc.hi=1 (bug reproduced, as expected --
     this is exactly what B-5(a)#2 reported and what the fix removes)      PASS
   OVERALL=PASS

--- N2 (warm-up namespace isolation), synthetic-file glob test ---
Created (scratchpad, not committed): 1 real-shaped pair
  g16_blk1_d16_boot1_999999_{LO,HI}.jsonl at top level, +
  g16_warmup_artifacts/g16_blk1_warmup_999999_srv.log, +
  g16_warmup_artifacts/g16_blk1_warmup_999999_bench.jsonl.IGNORE
glob.glob("*_LO.jsonl") / glob.glob("*_HI.jsonl")  (flat)      -> only the
                                                                    real pair
                                                                    matched
glob.glob("**/*_LO.jsonl", recursive=True) / same for HI       -> only the
                                                                    real pair
                                                                    matched
                                                                    (warm-up
                                                                    files
                                                                    excluded
                                                                    even
                                                                    recursively,
                                                                    because of
                                                                    the
                                                                    .IGNORE
                                                                    suffix)
                                                                             PASS

--- N3 dry-run (g16_n3_dryrun_check.py) ---
11 leaked PDMUX_*/SGLANG_ZAMBA_* vars simulated (7 named in N3's own defect
report + 3 pre-set to STALE values for PDMUX_TELEMETRY_PATH/RUN_ID/
WORKLOAD_ID + 1 arbitrary extra not on any prior list), against the real
bulk-unset block sourced out of g16_grid.sbatch:
  all 11 leaked vars cleared                                               PASS
  PDMUX_TELEMETRY_PATH/RUN_ID/WORKLOAD_ID all present with FRESH
    (not stale-leaked) values                                              PASS
  OVERALL=PASS

--- regression checks (repo-wide, confirm this round introduced 0 regressions) ---
check_sbatch_comment.py g16_grid.sbatch                    -> 1/1 conformant
check_sbatch_comment.py (full repo)                        -> 130/130 conformant
sbatch --test-only g16_grid.sbatch 1                        -> accepted (884249)
sbatch --test-only --export=ALL,G16_SMOKE=1 g16_grid.sbatch -> accepted (884250)
squeue -u $USER (after both --test-only calls)              -> empty (confirms
                                                                 --test-only did
                                                                 not queue
                                                                 anything)

--- N7 basis check ---
grep -n "PDMUX_TELEMETRY_PATH" sharegpt_vary_bench.sbatch   -> no match
  (confirms the doc-only claim in the N7 scope-note section above)
```

No job was submitted this round either; both `sbatch --test-only` calls
above are new job-id assignments from THIS pass (884249/884250), distinct
from the addendum-A round's 884224–884227 — `squeue` confirmed empty after
both.

## Rule defects found (report only, not fixed here)

**None found in the RULES layer this round either** (rev3 §4/§5/§6/§8/§9 +
addenda A and B, re-checked against B-5(a)'s 5 items specifically). No new
tension found beyond the pre-existing H17 one (addendum-A round, still
resolved as documented, unaffected by this round's changes).

**Addendum-A round's original note, preserved below:** the one apparent
tension (H17 vs rev3's original §9 condition-4 wording) is addressed above
under "Rule tension found and how it was
resolved" and judged to be addendum-authorized, not a genuine rule
contradiction — no rule text was changed to resolve it, only the harness's
tag-naming implementation.
