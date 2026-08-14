# LTSM-P1 — per-layer-type SM response, single-cell admissibility probe
## Pre-registration rev1 (2026-08-14, experiment-runner) — GPU spend to date: 0

> **Status: rev1, first submission, NOT YET claims-auditor reviewed.** Do not
> submit `ltsm_p1_probe.sbatch` until this document has been through the same
> adversarial design-audit process as `PREREG_GATE2S_2026-08-09.md` and
> `PREREG_G1C_2026-08-11.md`. Format follows that trio's standard (integrity
> rules, blind declaration, self-invalidation conditions, decision rules,
> interpretation limits) — scaled down from a 5-arm/6+ GPU-hr campaign to a
> single-boot ≈0.4–0.6 GPU-hr probe.

> ⚠️ **This document is the pre-registration. The design report it is drawn
> from is not.** `workspace/engine-port/reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md`
> states this itself, twice, and this document inherits the prohibition
> rather than repeats it as a courtesy:
> - §0 header: *"사전등록이 아니다(§8의 결정 규칙은 후보이며 확정이 아니다)"*
> - §13 (last bullet, "하지 말 것"): *"본 문서를 사전등록으로 인용 금지. 사전등록은
>   claims-auditor 설계 감사를 거친 별도 문서여야 한다."*
>
> Every decision rule in §5–§6 below that traces to that design doc has been
> **re-derived, corrected where wrong (§3), and fixed here** — this document,
> not the design doc, is what a future job / analyst / auditor cites.

---

## 0. What this buys, in one paragraph (method precedes result)

This is **not** a measurement of whether attention layers and mamba layers
respond differently to SM allocation. It is a **gate that must pass before
that measurement is attempted at all**. The instrumentation this probe would
use has already run once, on a 2.7B model, and **failed its own physical-
invariant check in exactly the corner this line of work needs** (§1). This
probe asks, at 7B scale, on the model and SM values the eventual full grid
would use: is that failure mode still present? If yes, the "eager ladder"
this design doc's §5 describes is dead and the design routes to a different
instrumentation path (nsys) that this document does not cover. If no, a
follow-on pre-registration for the full `(L,B,SM)` grid becomes submittable.
**This probe cannot itself produce a "layer type X is more SM-sensitive"
claim** — see §9.

---

## 1. Standing facts this pre-registration relies on (file:line, re-confirmed
## this session — not inherited from the design doc without recheck)

| # | Fact | Confirmed by (this session) |
|---|---|---|
| F1 | **The instrumentation this probe uses already ran once (job 873783, 2026-08-05) and FAILED GATE N** (the mamba-decode O(1)-in-ctx physical invariant) in the corner this probe re-enters: high-SM, short-ctx. `per_mamba` inflated up to **2.78×** and was **non-monotone in SM** at ctx 256/1024 (full/sm44/sm24 all violated even on the bs=32-only, batch-matched subset). Mechanism (established by direct measurement, not inference): host-paced forwards bill idle GPU time to whichever span is open; multi-launch spans (mamba mixer, `other`) absorb it, the 2-GEMM `mlp` span does not. | Read `results/r0c/P5_GATES_BATCH1_2026-08-05.md` §2 in full this session. |
| F2 | **`cuda_graph_runner.py:547`** sets `capture_forward_mode = ForwardMode.DECODE`, and **`:1161`** replay is `self.graphs[graph_key].replay()` — a CUDA graph replay, which does not invoke the Python `forward()` this instrumentation lives in. ⇒ per-layer-type timing is **structurally unavailable under cudagraph-ON**, and this probe (eager, `--disable-cuda-graph --disable-piecewise-cuda-graph`) is **off-operating-point by construction**, not by a caveat that could be lifted with more care. | Read directly, `sglang_engine_dev/python/sglang/srt/model_executor/cuda_graph_runner.py:540-560,1155-1165`. |
| F3 | **`SGLANG_ZAMBA_TIMING=0` is truthy.** `zamba2.py:544`: `_timing = bool(_os.environ.get("SGLANG_ZAMBA_TIMING")) and _phase` — `os.environ.get(...)` returns the *string* `"0"` when the var is set to `0`, and `bool("0")` is `True` in Python (any non-empty string is truthy). The same pattern is at `nemotron_h.py:646` (`_os.environ.get("SGLANG_NH_LAYER_TIMING") and _is_decode`, no `bool()` even needed to trigger it) and `granitemoehybrid.py:474` (`bool(_os.environ.get("SGLANG_GRANITE_TIMING"))`). **The OFF arm of any TIMING on/off dyad must `unset` the variable, never set it to `"0"`.** Contrast, confirmed correct: `multiplexing_mixin.py:104` reads `PDMUX_TRACE_FORCE_PREFILL` with `os.environ.get(..., "0") in ("1", "true", "True")`, which correctly treats `"0"` as OFF. | Read `src/models/zamba2.py:544`, `nemotron_h.py:646`, `granitemoehybrid.py:474`, `multiplexing_mixin.py:95-105` directly this session. |
| F4 | **`sync_engine_tree.sh` hashes exactly 15 files** (`:99-115`) and this probe's dev-tree footprint is fully inside that set (`models/zamba2.py`, `configs/zamba2.py`, plus the multiplex/scheduler files the boot depends on transitively). `nemotron_h.py`/`granitemoehybrid.py` are **not** installed by sync (manual copies, per the script's own `:73-75` comment) and this probe does not touch them (Zamba2-only). | Read `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh:1-119` in full. |
| F5 | **The existing Gate-1/Gate-2-S reference manifests are mutually byte-identical (15/15 sha, jobs 877756/877757/877974)** and this is the correct trip-wire anchor for a **no-change-to-tree** job — freezing a *new* reference here would be redundant and would fragment the audit trail. | `diff` of the three manifest files' hash columns this session — identical. |
| F6 | **`gate1_run.sbatch:61-66`'s own reference manifest (`reference_manifest_873944.sha256`) is stale** — it has 13 lines (predates `holb_probe.py`/`scheduler.py` entering the 15-file set), so running that harness today would hit its `MANIFEST_MISMATCH_ABORT` exit 3. Not this probe's problem to fix, but it is why this probe reuses the 877974 manifest, not the 873944 one. | `wc -l` + `cat` on `results/p1_gates/gate1/reference_manifest_873944.sha256` this session — 13 lines. |
| F7 | **`test_p5_gate_tools.py:40-41,176-183,237-243` pins `results/r0c/analyze_decode_knee_vs_ctx_v2.py` and `results/r0c/decode_knee_vs_ctx_v2.sbatch` by reading their literal file content** (`ANALYZER.read_text()` / `SBATCH.read_text()` and asserting substrings). An in-place edit of either file would fail this regression, which is itself the last blocking gate on `g2s_run.sbatch:91`'s `unittest discover` call — i.e. it would fail a *different* campaign's (E1-b's) boot precondition. This probe is therefore a **new file** (`ltsm_p1_probe.sbatch`), and reuses the r0c analyzer's Python module **unmodified**, via its existing `--dir`/`--n-attn`/`--n-mamba` CLI flags. | Read `workspace/engine-port/tests/test_p5_gate_tools.py` in full this session; read `results/p1_gates/gate2/g2s_run.sbatch:91` directly. |
| F8 | **`gate_n_conditional`, `host_boundness_rows`, `batch_mixture`, `summarize` are already generic in `(n_attn, n_mamba)`** — passed as CLI flags (`--n-attn`, `--n-mamba`, default 9/54 for Zamba2-2.7B), not hard-coded. A 7B run needs the right numbers, not new code. | Read `results/r0c/analyze_decode_knee_vs_ctx_v2.py:245-266,347-450,453-463` directly this session. |
| F9 | **Zamba2-7B-Instruct: `num_hidden_layers=81`, `len(hybrid_layer_ids)=13`, `max_position_embeddings=4096`.** `zamba2.py:490-491` computes `self._zt_n_attn`/`self._zt_n_mamba` from the *built model's own layers* (not hard-coded), so the emitted `ZBLT2` lines will already carry the right per-arm divisor counts (`per-attn(13)=`, `per-mamba(81)=`) regardless of what this pre-registration passes to the analyzer — but `summarize()`'s `per_mlp`/`per_other` computation (§F8) needs the CLI flags set correctly to match, since `b_mlp`/`b_other` in the raw record are per-forward totals, not pre-divided. | `python3 -c "import json; ..." ` against `hf_cache/hub/models--Zyphra--Zamba2-7B-Instruct/snapshots/*/config.json` this session. |
| F10 | **`--mamba-track-interval`'s checkpoint mask is only ever computed inside `if get_global_server_args().enable_mamba_extra_buffer():`** (`schedule_batch.py:2144-2175`), which requires `--mamba-scheduler-strategy=extra_buffer`; the default `mamba_scheduler_strategy="auto"` resolves to `"no_buffer"` (`server_args.py:964-967`) and this probe's `--disable-radix-cache` never overrides that resolution (the `elif not self.disable_radix_cache:` branch at `:2201` that would touch `no_buffer`'s interaction with radix cache is itself gated on radix cache being *enabled*, i.e. it is skipped here too). **The checkpoint path does not execute in this configuration at all** — there is no counter to read, empirically or otherwise. | Read `sglang_engine_dev/python/sglang/srt/managers/schedule_batch.py:2144-2175` and `server_args.py:964-967,2140-2225` directly this session; traced the `--disable-radix-cache` interaction line by line. |
| F11 | **`disable_radix_cache` interaction sanity**: at `server_args.py:2155`, if `support_mamba_cache` is false for the model architecture the function returns early forcing `disable_radix_cache=True` and never reaches the extra_buffer/no_buffer branch at all — Zamba2 already runs with `--disable-radix-cache` successfully in the existing r0c harness (job 873783 completed), so this is a known-working combination, not new territory. | Cross-referenced against `results/r0c/decode_knee_vs_ctx_v2.sbatch:108`, which already uses `--disable-radix-cache` with Zamba2-2.7B and completed. |
| F12 | **`/server_info` returns `dataclasses.asdict(server_args)`** (`http_server.py:610-618`), which includes `mamba_scheduler_strategy` as a field — a safe, GPU-node-only way to confirm F10's resolution at boot time from the *live* server, without constructing a standalone `ServerArgs()` object (whose `__post_init__` has many entangled side effects that are unsafe to trigger outside a real boot). | Read `sglang_engine_dev/python/sglang/srt/entrypoints/http_server.py:591-618` directly this session. |
| F13 | **This cluster's actual `--cpus-per-task` ceiling for `--gres=gpu:1` is 8, not 16.** `sbatch --test-only` on a draft of this job with `--cpus-per-task=16` was rejected: `Job rejected: requested CPU cores per node (16) exceed the allowed limit (8) for 1 GPU(s)`. `results/r0c/decode_knee_vs_ctx_v2.sbatch` already uses `--cpus-per-task=8`, consistent with this. **This differs from the generic project convention quoted to this session** (`--cpus-per-task=16`); §3 correction 5. | `sbatch --test-only` this session, both before and after the fix (see manifest/appendix — reproducible). |

---

## 2. Self-invalidation conditions — declared before any data exists

This probe answers nothing if any of the following is true, and the
pre-registration commits, in advance, to how each is handled.

### 2.1 ★ The prior GATE N FAIL (F1) is the single most important fact governing
### this probe's design, and its resolution is the probe's *primary* judgement
### target, not a footnote

The 2026-08-05 failure was not a marginal miss — it was **all five SM arms
violating** the as-specified gate, and **3 of 5 arms violating even on the
strict batch-matched subset**, with the worst violation (2.78×) occurring at
the **`full`** arm (no green context at all — ruling out a green-context
artefact as the mechanism) at the **shortest ctx, highest available SM**
combination tested. This probe's own target cell (ctx 1024/2048, SM 44/92)
sits in the **same qualitative corner** (short-to-moderate ctx, high SM) that
failed before, on a **larger** model where per-forward device work is bigger
(more layers, wider hidden dim) — the one lever the P5_GATES report itself
identified as the fix direction (§2.4/§2.5: raise device work above the host
floor). **Whether the corrected A1 (§5.1) passes at 7B in this corner is
therefore the single fact this probe exists to establish; everything else
(A2/E2/E8) is secondary.**

### 2.2 ★★ Off-operating-point ceiling (F2) is a hard ceiling on the claim
### this probe's descendants can ever make, not an interpretation caveat

Because per-layer-type timing cannot execute under cudagraph replay (F2),
**no result from this probe, or any full-grid follow-on it might license,
can ever be read as an operating-point (cudagraph-ON) number.** This is not
"the results should be interpreted cautiously" — it is "the claim space this
line of work can reach is capped below the operating point, permanently,
by the architecture of `CUDAGraphRunner`". Any future write-up citing this
probe's descendants **must** carry the cudagraph-OFF caveat in the same
sentence as any number, not in a footnote (methodology gate: "the gate goes
in every reporting block").

### 2.3 Grid constraint forces a within-model design; no cross-arm synthesis

The design doc's own §6.3 arithmetic establishes that "both layer types
architecture-dominated (≥2/3 non-weight traffic)" is satisfiable for Zamba2
(Ha8) at **exactly one point**, `(L=1024, B≈96)`, itself at 98% of that
point's memory ceiling — and is **not simultaneously satisfiable** for a
second hybrid model (NemotronH/Hs8) at any point this probe touches. This
probe does not attempt to reach that architecture-dominated composition
target at all (§3 correction 2) — it is an **admissibility** probe, not the
composition measurement — but the grid constraint is recorded here because
it bounds what any *follow-on* L2 campaign licensed by this probe passing
can claim: **within-model only, no cross-arm (Zamba2-vs-NemotronH) synthesis,
ever, on this instrumentation.**

### 2.4 Self-invalidation triggers (checked in this order)

1. **Boot failure** (either server) → probe UNDETERMINED, not a negative
   result. Report the server log tail, do not infer anything about A1/A2.
2. **E8 boot-time strategy check (F10/F12) returns anything other than
   `no_buffer`** → the code-trace this pre-registration relies on (F10) is
   wrong, and the sbatch aborts (exit 12) rather than silently mis-scoring
   E8 as "structurally inapplicable" when it might not be.
3. **Zero `ZBLT2 BLOCK` lines emitted** at either SM/ctx cell → instrumentation
   is not firing (env truthy bug variant, or an unrelated regression) — probe
   UNDETERMINED, not "layers are SM-insensitive".
4. **A1 UNSCOREABLE** (realized batch sizes never overlap between ctx=1024
   and ctx=2048 at a given SM arm) → reported as UNSCOREABLE, which does
   **not** pass and does **not** fail — it means this specific probe design
   could not test the invariant, and a redesign (not a threshold change) is
   needed before proceeding. This is a real, anticipated possibility: the
   two cells are only *targeted*, not *guaranteed*, to realize the same
   batch (§4.2).
5. **`per_mlp` (negative control) shows a VIOLATION** → measurement
   corruption beyond the known host-pacing mechanism (P5_GATES found
   `per_mlp` clean to 0.4% in every one of its 20 cells, including the
   failing ones) — treat the whole run as compromised, do not report A1/A2
   numbers as if they were clean.

---

## 3. Corrections made to the design doc's §5-L1 spec (found and fixed this
## session, before any GPU spend — methodology gate #28: verify payoff and
## feasibility by code, not by re-quoting the source document)

1. **`SGLANG_ZAMBA_TIMING=0` is truthy (F3).** The design doc's own §5-L1 text
   ("`plus: 같은 (L=1024,B≈88)에서 SGLANG_ZAMBA_TIMING=0 의 step time`") is
   correct in *intent* but would be silently wrong if implemented literally
   as `export SGLANG_ZAMBA_TIMING=0` — that still turns timing ON. The sbatch
   `unset`s the three `SGLANG_ZAMBA_*` variables for the OFF arm instead
   (`ltsm_p1_probe.sbatch`, `boot_server()`). **Code is not modified** —
   this is a harness-side workaround, and the underlying truthy-bug fix is
   an engine-porter hand-off item, not in scope here.

2. ★★ **The design doc's literal L1 cell spec, `(L=2048, B≈88)`, is
   infeasible by the design doc's OWN §6.2 memory arithmetic.** §6.2's table
   gives Ha8 (Zamba2-7B) `B_max(mem)` at `L=2048` as **57**, not 88 — using
   `B≈88` at `L=2048` would exceed the memory budget the same document
   derived (an OOM risk, design doc's own §12 failure mode F5). This
   pre-registration substitutes **`TARGET_CAP=48`** for **both** cells
   (`L=1024` and `L=2048`), for two reasons: (a) it is safe with margin at
   both points (headroom to 98 at L=1024, headroom to 57 at L=2048, unlike
   the asymmetric 88/57 the literal spec implied), and (b) it is the exact
   value the design doc's own §6.3 "recommended grid" table independently
   settles on for the `L=2048` cell of the eventual full grid — so this is
   not a new number, it is resolving a document-internal inconsistency in
   the design doc's favour of its own more careful later section.
   **Consequence, stated plainly**: this sacrifices the "architecture-
   dominated composition" target that motivated `B≈88–96` in the first
   place (§2.3 above) — at `TARGET_CAP=48`, neither cell reaches the ≥2/3
   architecture-dominated threshold (§6.2's own table: L=1024/B=48 gives
   mamba-state fraction 53.5%, L=2048/B=48 gives 53.5%/88.7%). **This does
   not compromise L1's stated purpose** (A1/A2/E2/E8 are all measurement-
   validity checks, not the composition measurement itself), but it means a
   pass here does **not** license running `(L=1024, B≈96)` blind — that
   specific higher-B cell is untested by this probe and would need its own
   quick admissibility check if a full L2 grid is pursued.

3. **`--mamba-track-interval` is a structural no-op in this configuration
   (F10) and is not passed at all**, rather than being set to "a large
   value" as the design doc's informal spec suggested. E8 is reframed from
   "measure a firing count and require it to be 0" to "confirm the boot
   resolved to `no_buffer` strategy" (F12) — a **boot-time structural
   precondition**, not an empirical measurement. If this precondition is
   violated, the job aborts (§2.4.2) rather than silently reporting a
   measured-zero that was never actually computable.

4. **Manifest trip-wire reuses the 877974 reference, does not freeze a new
   one** (F5/F6), per explicit task instruction and because this probe makes
   zero tree changes — freezing a new baseline here would fragment, not
   strengthen, the audit trail.

5. **`--cpus-per-task=16` (the generic project convention quoted to this
   session) is rejected by this cluster for `--gres=gpu:1`; the real ceiling
   is 8** (F13). The sbatch uses `--cpus-per-task=8`, matching
   `results/r0c/decode_knee_vs_ctx_v2.sbatch`'s existing, working value.

---

## 4. Design

### 4.1 Model, environment, boot

- Model: `Zyphra/Zamba2-7B-Instruct`. Zamba2-only (§2.3 grid constraint) —
  NemotronH is out of scope for this probe (its timing instrumentation is a
  pre-2026-08-04-rework manual copy, not in the sync manifest; bringing it in
  is a separate, larger, engine-porter-gated item per the design doc §5-L2).
- `--trust-remote-code --dtype bfloat16 --attention-backend triton
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache
  --mem-fraction-static 0.80 --max-running-requests 48 --context-length 2560`.
  `triton` backend matches the design doc's V21 finding (Zamba2 requires
  triton for cudagraph capture — moot here since cudagraph is off, but kept
  for consistency with the already-audited r0c harness and to avoid
  introducing a second uncontrolled variable).
- `SGLANG_ZAMBA_TIMING=1 SGLANG_ZAMBA_TIMING_EVERY=8 SGLANG_ZAMBA_CLOSURE_MIN=0.85`
  for the primary boot (Phase 1); all three **unset** for the E2 baseline
  boot (Phase 2) — never set to `"0"` (§3 correction 1).
- `--n-attn 13 --n-mamba 81` passed to the analyzer (F9).

### 4.2 Cells, arms, order

- Two prompt lengths: `L ∈ {1024, 2048}` tokens (approximate, built via the
  same repeated-phrase-text generator as the r0c harness).
- Two SM pins: `SM ∈ {44, 92}` via `PDMUX_FIXED_DECODE_SM_FILE`, independent
  of `--enable-pdmux` (the model's `_get_gctx_decode_stream` builds its own
  green-context pair lazily; the design doc's V7 confirms pdmux is not
  required, and the existing r0c harness ran this way successfully).
- `TARGET_CAP=48` (`--max-running-requests`) at **both** cells (§3
  correction 2). `CONC=64` concurrent client requests fired per round
  (above the cap, to saturate admission), `OUTTOK=32` output tokens,
  `MEAS_ROUNDS=2` measured rounds per (SM,ctx) cell plus 1 discarded warm-up
  round — identical aggregation convention to the audited r0c harness
  (P5_GATES_BATCH1), so `blk_spread_pct`/`n_blocks` numbers remain
  comparable in kind (not in value — different model/SM values) to that
  prior run.
- **Visit order of the 4 (SM,ctx) cells is randomised** from a seed (default
  `20260814`, overridable as the sbatch's `$1`), guarding against the
  "first arm most diluted by cold start" bias the P5_GATES report's defect 2
  lineage identified.
- **realized batch size is NOT forced equal between the two ctx cells** — it
  is *targeted* via the shared `TARGET_CAP`/`CONC`, but admission/completion
  dynamics can still produce different realized `bs` at each cell (the
  P5_GATES report's own data show mixtures like 6×bs19+8×bs13 within a
  single cell). **A1 is scored only on whatever realized-`bs` overlap
  actually occurs** (`gate_n_conditional`'s existing UNSCOREABLE-if-no-
  overlap behaviour, reused unmodified) — this pre-registration does **not**
  guarantee A1 will be scoreable, and treats non-overlap as a real, reported
  outcome (§2.4.4), not a design failure to paper over.

### 4.3 E2 baseline cell (single, designated)

Ctx=1024, SM=92, chosen (before running anything) because it is the
**highest-SM** arm in this probe: device work per decode step is smallest
there, so a fixed host-side instrumentation cost (the `torch.cuda.
synchronize()` + per-span `Event` pair + ~270 `elapsed_time()` read-backs the
design doc's §7 describes) is the **largest relative fraction of total step
time** at this arm — a conservative choice for detecting overhead (harder to
hide it here, not easier). E2 does **not** test SM=44 or ctx=2048; a
generalisation of the measured overhead percentage to those cells is not
licensed by this probe.

### 4.4 Pipeline smoke (methodology gate #26, amended 2026-08-14)

This job's harness code is new relative to the last audited campaign this
instrumentation ran under (job 873783). Per gate #26's amended trigger
("the combined GPU cost of a batch of new/changed-code campaigns submitted
together, not a single campaign in isolation"), this probe is part of a
batch noted in `handoff-report/session_handoff_2026-08-13.md` §2.4–2.5 (LTSM-
P1 + E1-b/E1-c + `%smid` P1/P2) whose **combined** estimated cost (≈1.15–1.65
GPU-hr) crosses the 1 GPU-hr trigger, even though this probe alone
(≈0.4–0.6 GPU-hr) does not. `ltsm_p1_probe.sbatch`'s Phase 0 is this probe's
share of that obligation:

- **What it does**: boots the real model with the real server flags and
  `SGLANG_ZAMBA_TIMING=1` (the real execution path), fires a tiny round
  (4 concurrent requests, 24 output tokens, ctx≈99 tokens — cheap, a few
  minutes including boot), writes output through the **exact same file-
  naming convention and analyzer invocation** the real cells use (under a
  file-name tag distinct from the real job tag, so it cannot contaminate the
  real GATE C/N computation — see the sbatch's Phase 0 comment), and checks
  that the unmodified r0c analyzer runs to completion (no traceback) and
  prints its three expected structural sections (`GATE C`, `GATE N`,
  `HOST-BOUNDNESS OBSERVABLES`).
- **What a PASS here means**: the harness, the instrumentation, and the
  analyzer's file-discovery/parsing contract all work end-to-end for this
  job's naming scheme and this model's layer counts.
- **What a PASS here does NOT mean, and must never be cited as**: anything
  about A1, A2, E2, or E8. A single n=1, 4-request, 24-token smoke round is
  far too small to render any of those verdicts, and — this is the specific
  trap gate #26 exists to name — the smoke's own GATE C/N verdict is
  **expected to be non-PASS or UNSCOREABLE**, because `keep_steady()` drops
  the first block of the (single) round, likely leaving 0–1 kept blocks.
  The smoke check therefore does **not** gate on the analyzer's process exit
  code (`main()` returns 1 by design whenever GATE C/N do not cleanly PASS —
  that is not a crash) — it gates on the *absence* of a Python traceback and
  the *presence* of the expected output sections. A pipeline-smoke PASS
  licenses zero performance or admissibility conclusions.

---

## 5. Decision rules (fixed before any GPU hour is spent — not re-tuned
## against this run's own data, methodology gate #8)

### 5.1 A1 — physical invariant (per SM arm)

**Decision quantity**: `gate_n_conditional(subcells, "per_mamba", 1.15, ["44","92"])`
(the r0c analyzer's existing, unmodified function), scored per SM arm across
the two ctx points, conditional on realized batch size.

- **VIOLATION at either arm → that arm FAILS A1.** The 1.15 threshold is
  pre-registered (it is the r0c analyzer's own `GATE_N_MAX_DEFAULT`,
  `test_p5_gate_tools.py`-pinned) and is **not** re-derived from this run.
- **UNSCOREABLE (no realized-bs overlap between L=1024 and L=2048 at that
  arm) does not pass.** It is reported as UNSCOREABLE and this probe cannot
  license a follow-on L2 grid on the strength of the *other* arm alone —
  see §6.
- **`per_mlp` (complement negative control) must show OK at both arms.** A
  `per_mlp` VIOLATION invalidates the entire run (§2.4.5) — it is not scored
  as an independent A1 failure, it is treated as evidence the measurement
  itself is compromised beyond the known host-pacing mechanism.
- **A3 (monotonicity in SM) is explicitly NOT evaluated here.** Only two SM
  points exist in this probe (44, 92); monotonicity needs ≥3 points to be
  meaningful and is deferred to a full L2 grid (`{44,54,74,92}` per the
  design doc's §6.3 recommended grid, if this probe passes).

### 5.2 A2 — device-bound vs host-paced (at ctx=1024, the two SM arms)

**Decision quantity**: `fwd_ms(SM=44)` vs `fwd_ms(SM=92)`, read from
`host_boundness_rows()`'s existing output (which already computes, per
(ctx,bs), the ratio of each arm's `fwd_ms` to the minimum across arms at
that cell).

- **FAIL (host-paced, cell inadmissible) if EITHER**:
  (a) `fwd_ms(44) < fwd_ms(92)` — the reversed, physically-suspect direction
  (more SM ⇒ slower step), which is exactly the pattern P5_GATES_BATCH1
  flagged as a physical-invariant violation in its own §2.3/§6.2 ("NOTE …
  physically suspect"); **or**
  (b) `fwd_ms(44) / fwd_ms(92) < 1.05` — not distinguishable beyond ~10×
  this harness's demonstrated measurement noise floor (P5_GATES_BATCH1 §5:
  round1/round2 agreement ≤0.5%, cross-position reproducibility ≤0.31%).
- **The 1.05 threshold is reused verbatim from A1's own already-registered
  value** (the design doc's own §8 free-parameter accounting names "A1's
  1.05 and A2's floor margin" as the only two new free parameters this line
  of work introduces) — this pre-registration is the point at which A2's
  value is fixed *before* data exists, per the design doc's own instruction
  that "the margin comes from what P1 measures, not from tuning against P1's
  own outcome". Reusing A1's number, rather than inventing a second one, is
  the more conservative choice available given the source document assigns
  no numeric value of its own.
- **PASS** otherwise (correct direction AND ≥1.05 separation).

### 5.3 E2 — instrumentation self-cost (descriptive, not gated)

**Decision quantity**: mean round wall-clock time with `SGLANG_ZAMBA_TIMING=1`
(from Phase 1's SM=92/ctx=1024 rounds) vs mean round wall-clock time with
`SGLANG_ZAMBA_TIMING` unset (Phase 2, same cell), reported as a percentage:
`(mean_with − mean_without) / mean_without × 100`.

- **No pass/fail threshold.** This is a size-the-overhead measurement (the
  design doc's §7-1: *"이것은 '무해 입증'이 아니라 부하의 크기를 숫자로 적어
  아티팩트에 남기는 것"*), not a hypothesis test — and this project's
  methodology gate #14 (t-CI, not percentile bootstrap, for n≤8) does not
  apply here because **no confidence interval is computed at all**: n=2
  rounds per side, a single wall-clock point comparison, explicitly
  descriptive. If a future revision of this probe wants an inferential E2
  claim, it must use a paired t-CI over ≥5 rounds per side, pre-registered
  before running.
- **≥10% overhead is an ADVISORY flag** for human review before authorising
  any follow-on L2 grid (narrowing the A2 margin, per the design doc's §7-1:
  "결과가 크면 그것은 셀 admissibility를 좁히는 방향으로만 쓴다") — it is
  **not** an auto-fail of this job, and 10% is a round, pre-committed
  number, not derived from this run's own outcome.
- If either boot fails, E2 is **UNDETERMINED**, and this must not be reported
  or treated as "overhead = 0%".

### 5.4 E8 — mamba-track-interval structural check (boot-time precondition)

**Decision quantity**: `mamba_scheduler_strategy` field of the running
server's `/server_info` response.

- **Expected value: `"no_buffer"`** (F10). If confirmed, E8 is reported as
  `STRUCTURALLY_INAPPLICABLE` — the checkpoint-mask code path this check
  concerns does not execute in this configuration at all, so there is
  nothing to measure, and this is explicitly **not** the same claim as "a
  firing count was measured and found to be zero".
- **If the field is anything else, the job ABORTS (exit 12)** rather than
  silently proceeding as if F10 held. This is a hard precondition check, not
  a soft warning, because the entire justification for treating E8 as
  structurally moot rests on F10's code trace being correct.

### 5.5 Statistical discipline (methodology gates #8, #14, #24)

- **No percentile-bootstrap confidence interval is computed anywhere in
  this probe.** n_indep=1 (single boot per arm) for the primary A1/A2
  quantities — there is nothing to bootstrap; the decision rules above are
  deterministic threshold checks on point estimates (medians over kept
  blocks within one boot), not statistical tests. If a future L2 revision
  introduces repeated boots (n≥5) and a between-arm comparison, primary
  inference must use a paired t-CI (gate #14), never percentile bootstrap.
- **No `any()`-over-reps screen is used anywhere in this probe.** Each
  decision quantity (A1 per arm, A2 at one cell, E2 at one cell, E8 at boot
  time) is evaluated exactly once per probe execution — there is no
  multiple-comparison inflation to disclose (gate #24's 1−(1−α)ⁿ null-
  firing-rate concern applies to OR-screens over n≥2 reps of the *same*
  test; this probe has n=1 throughout).

---

## 6. Top-level verdict and what it licenses

**`ELIGIBLE_FOR_L2`** iff **all** of:
1. Both server boots (Phase 1, Phase 2) succeed.
2. E8 boot-time precondition holds (§5.4) — otherwise the job has already
   aborted before this point is reached.
3. GATE C (closure ≥ 0.85, the analyzer's pre-registered, unmodified
   threshold) == PASS for the scored cells.
4. A1 == OK (not VIOLATION, not UNSCOREABLE) for **both** SM=44 and SM=92.
5. `per_mlp` == OK for both arms.
6. A2 == PASS at ctx=1024.

Otherwise: **`NOT_ELIGIBLE_FOR_L2`**. Route to the design doc's own
documented next step for this failure mode (§5 "차선책": if A1/A2 fail, the
eager-instrumentation path this probe belongs to is dead at this hardware
scale, and the alternative is the nsys/`--cuda-graph-trace` path the design
doc's §4 row 2 and §11.2-P4 describe — **out of scope for this
pre-registration**, would need its own design + pre-registration). **Do not
re-run this exact job with adjusted A1/A2 thresholds if it fails** — that is
methodology gate #8 by name (re-tuning a gate against the data it was meant
to gate).

`ELIGIBLE_FOR_L2` licenses **submitting a pre-registration for the design
doc's §6.3 full grid** (Zamba2-only, per §2.3's grid constraint) — it does
**not** itself constitute that pre-registration, and does not license
skipping the claims-auditor design-audit step for that follow-on document.

---

## 7. What this probe does not, and structurally cannot, close (gate #28 —
## payoff stated narrowly, not optimistically)

Applying the same discipline the design doc's own §1.2/§9 apply to itself:

1. **No layer-type SM-sensitivity claim of any kind.** This is an
   admissibility gate, not the composition measurement. A PASS produces
   zero numbers about which layer type is more SM-sensitive; it only
   produces permission to *attempt* measuring that at a larger grid.
2. **No operating-point (cudagraph-ON) claim, ever, from this instrumentation
   path** (§2.2, F2) — structurally, not by caveat.
3. **No goodput/TTFT/ITL/policy claim.** Nothing here touches `PDMUX_LA_*`
   or any layer-aware runtime policy path; per the design doc's own §1.1,
   even a full-grid measurement downstream of this probe passing would not
   revive the layer-type *policy* (dead on exploitation cost, §1-1 of
   `CONSENSUS.md`; per_layer_type_postmortem.md), only the layer-type
   *instrumentation* question.
4. **No cross-model synthesis** (§2.3) — Zamba2-only, and even within
   Zamba2 the design doc's §6.3 matched-cell criterion is not attempted by
   this probe (§3 correction 2).
5. **No deployment relevance.** Even if a follow-on L2 grid runs and
   produces a clean signal, `TARGET_CAP=48` here (and `B≈88–96` in the
   design doc's own L2 grid) is 4–8× the observed serving decode batch
   (9–12, per `FINDINGS_8B_2026-07-28.md`) — this line of work characterises
   an architecture-dominated traffic regime, not the deployment operating
   point, and the design doc's §6.5 "이 격자는 배포 동작점이 아니다" applies
   in full to any descendant of this probe.
6. **This probe does not validate that the SM values requested (`44`,`92`)
   are what the hardware actually delivered.** That is the `%smid`
   (C-1) design's question (`SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`),
   a separate, complementary track — this probe's x-axis (target SM) is
   trusted, not independently verified, exactly as the design doc's §9-5
   flags.

**Honest summary**: passing this probe buys exactly one thing — permission
to spend more GPU hours on the next rung of this ladder. It buys nothing
that can be written into `CONSENSUS.md` or `PROJECT_STATUS.md` as a result.

---

## 8. Artifacts (per-job, under `results/ltsm_probe/`)

- `runtime_source_manifest_ltsmp1_<jobid>.sha256`, `manifest_diff_ltsmp1_<jobid>.txt`
  — trip-wire outputs (§1 F5/F6).
- `deckneectx2_result_C{1024,2048}_r1_<jobid>.txt`,
  `deckneectx2_srv_C{0,1024}_r{1,2}_<jobid>.log` — raw BLOCK/CELL/ARMDONE/HEADER
  records and server logs, r0c-compatible naming (§1 F7 — deliberate reuse of
  the unmodified analyzer's file-discovery glob).
- `deckneectx2_result_C99_r9_<jobid>s.txt`, `deckneectx2_srv_C99_r9_<jobid>s.log`,
  `smoke_analyzer_out_<jobid>.txt` — Phase 0 smoke artefacts, under a
  jobid**s** tag structurally distinct from the real jobid so they cannot be
  globbed into the real analysis (§4.4).
- `e2_overhead_<jobid>.txt`, `phase1_round_wall_<jobid>.txt` — E2 wall-clock
  numbers (§5.3).
- `ltsmp1_analyzer_out_<jobid>.txt` — the sbatch's own convenience run of the
  unmodified r0c analyzer against the real result files. **Not the analysis
  of record** — an independent re-parse of the raw `BLOCK` lines by
  result-analyst is required before any A1/A2/GATE-C/GATE-N number from this
  probe is cited anywhere, matching this project's standing practice for
  every prior gate in this line (Gate 1/1b/1c/2-S all required independent
  re-derivation before citation).

---

## 9. Do-not list (carried from the design doc's §13, plus items specific to
## this pre-registration)

- Do not cite the design doc (`PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md`)
  as a pre-registration — this document is.
- Do not re-tune A1's 1.15 or A2's 1.05 against this run's own outcome.
- Do not report E8 as "measured zero firings" — it is a structural
  precondition, not a measurement (§5.4).
- Do not treat `ELIGIBLE_FOR_L2` as itself licensing GPU spend on a full
  grid — a separate pre-registration, separately audited, is required.
- Do not generalise the E2 overhead percentage from SM=92/ctx=1024 to any
  other cell.
- Do not report any number from this probe as an operating-point
  (cudagraph-ON) claim.
- Do not synthesise across Zamba2 and any other hybrid model on this
  instrumentation.
- Do not treat Phase 0 (pipeline smoke) PASS as informative about A1/A2/E2/E8
  (§4.4).
- Do not submit `ltsm_p1_probe.sbatch` before this document has a
  claims-auditor design-audit verdict recorded in a §0.0 revision-history
  block (this document currently has none — rev1 only).
