# P5 re-measurement, batch 1 (rep1) — instrumentation gate verdict

**STATUS: UNAUDITED.** result-analyst output, 2026-08-05. Not audited by claims-auditor,
not canonical. Do not cite from `PROJECT_STATUS.md` / `CONSENSUS.md` until audited.

**Scope: instrumentation only.** No knee, no composition, no performance/policy claim is
made or permitted from these data (both instrumentation gates are scored below; one fails).
`n_indep = 1` (rep1 only). Block-to-block spread inside a cell is *within one server boot and
one arm activation* — it is **not** reproducibility.

Raw: `results/r0c/deckneectx2_result_C{256,1024,4096,16384}_r1_873783.txt` (+ `_srv_*.log`),
job 873783 array 0-3 (SLURM ids 873847/873848/873850/873783), all COMPLETED 0:0, all on node
**gpu41**, `git_head=6c06da8`, `zamba2_sha256=035a7d75…64`, eager (`--disable-cuda-graph
--disable-piecewise-cuda-graph`), triton attn backend, `--disable-radix-cache`,
`mem-fraction-static 0.80`, `max-running-requests 48`, conc=32, outtok=32.
Analyzer: `analyze_decode_knee_vs_ctx_v2.py` (GATE C 0.85 / GATE N 1.15); every number below
was independently re-parsed from the `BLOCK` lines and matches the analyzer's CELL output.

---

## 0. Aggregation unit (declared first — methodology gate #4)

One `BLOCK` record = a window of `blk_n = 8` consecutive **timed decode forwards** of one
server, one SM arm, one round. Within that window (source: `zamba2.py:658-701`):

| field | exact meaning |
|---|---|
| `fwd_ms` | `Σ_8 (independent event pair around the whole layer loop) / 8` = **ms per decode forward**, layer loop only (excludes embed, final_layernorm, sampler, scheduler, detokenizer) |
| `per_mamba` | `Σ_8 Σ_{54 mamba-mixer spans} / 8 / 54` = **ms per mamba-mixer invocation per forward** |
| `per_attn` | same, `/9` attention modules — **whole** `Zamba2Attention.forward` (new bucket) |
| `per_attn_core` | same, `/9` — RadixAttention core only (= pre-2026-08-04 `attn` definition, `_ZT_DIAG`, nested, excluded from closure) |
| `b_mlp`, `b_other` | per-forward **totals** (not per-layer). Below I divide `mlp` by 9 and `other` by 54 for the complement test |
| `closure` | `(attn+mlp+other+mamba)/fwd_ms` — this is an **identity** given the emitted fields (verified: max deviation 5.2e-05 over 200 blocks); its content is entirely in the *independence* of the denominator event pair |

CELL value = **median over kept blocks**; kept = `dirty==0` AND not the first block of its
round. Kept/raw = 6/8 for ctx 256/1024/4096, 14/16 for ctx 16384. **Zero** blocks were dropped
for `dirty`; the 2 dropped per cell are exactly the first block of each of the 2 rounds.
CELL `bs` = `statistics.mode` over kept blocks (see §2.1 — this hides a mixture).

---

## 1. TABLE A — 20 cells (medians over kept blocks)

```
   ctx    sm  nb    nseq  clo_med  clo_min    fwd_ms  unbkt_ms  per_attn  per_core  a/core  per_mamba  blk spread%  mlp_tot  oth_tot
   256  full   6    [32]   0.9454   0.9427    43.978     2.403    0.5192    0.3276   1.585     0.5546        0.18     1.359    5.597
   256    44   6    [32]   0.9300   0.9286    45.361     3.178    0.8194    0.6131   1.336     0.5186        0.45     1.647    5.173
   256    24   6    [32]   0.9754   0.9731    45.540     1.120    1.3114    1.0308   1.272     0.4976        1.08     2.455    3.310
   256    16   6    [32]   0.9849   0.9845    51.159     0.773    1.8810    1.5251   1.233     0.5305        0.24     3.221    1.574
   256     8   6    [32]   0.9918   0.9915    92.960     0.767    3.6097    2.9534   1.222     0.9598        0.09     6.078    1.801
  1024  full   6    [32]   0.9521   0.9513    44.069     2.111    1.2341    1.0420   1.184     0.4623        0.72     1.365    4.496
  1024    44   6    [32]   0.9578   0.9563    46.054     1.943    2.3053    2.0995   1.098     0.3413        0.40     1.644    3.293
  1024    24   6    [32]   0.9861   0.9853    62.541     0.869    4.0410    3.7949   1.065     0.3900        0.09     2.454    1.774
  1024    16   6    [32]   0.9912   0.9910    87.706     0.772    5.9530    5.5977   1.063     0.5296        0.11     3.214    1.552
  1024     8   6    [32]   0.9952   0.9950   164.819     0.783   11.5719   10.9136   1.060     0.9625        0.12     6.100    1.810
  4096  full   6    [32]   0.9824   0.9819    52.087     0.919    4.1406    3.9817   1.040     0.1995        0.53     1.365    1.757
  4096    44   6    [32]   0.9887   0.9884    92.268     1.047    8.1761    8.0038   1.022     0.2632        0.45     1.648    1.779
  4096    24   6    [32]   0.9946   0.9944   163.837     0.885   15.2946   15.0483   1.016     0.3907        0.37     2.445    1.787
  4096    16   6    [32]   0.9968   0.9967   239.292     0.766   22.7831   22.4262   1.016     0.5312        0.42     3.220    1.555
  4096     8   6    [32]   0.9983   0.9983   456.335     0.776   43.9760   43.3208   1.015     0.9609        0.10     6.074    1.820
 16384  full  14 [13,19]   0.9850   0.9826    62.272     0.931    5.4762    5.3250   1.028     0.1643        7.19     1.308    1.800
 16384    44  14 [13,19]   0.9921   0.9917   134.280     1.061   13.2328   13.0363   1.015     0.1979       10.81     1.544    1.868
 16384    24  14 [13,19]   0.9954   0.9953   235.070     1.081   23.9319   23.6457   1.012     0.2644       14.02     2.325    2.001
 16384    16  14 [13,19]   0.9970   0.9969   340.611     1.039   35.0955   34.6882   1.012     0.3424       16.41     3.053    2.174
 16384     8  14 [13,19]   0.9988   0.9987   662.729     0.795   69.1184   68.3445   1.011     0.5939       18.02     5.747    2.110
```
`unbkt_ms = fwd_ms*(1-closure)`. `blk spread% = (max-min)/median` of `per_mamba` over kept blocks.

---

## 2. G2 (★ GATE N) — **FAIL, and the gate is also mis-specified**

### 2.1 The gate spec is confounded (checked first, methodology gate #7)

`ctx16384` cells are **not single-condition cells**. Kept blocks run at `nseq = 19` for
blk 1-3 and `nseq = 13` for blk 4-7, in *both* rounds and *all five* arms (6 blocks at bs=19,
8 at bs=13 per cell). The CELL line prints `bs=13` because it reports
`statistics.mode` — the 6 bs=19 blocks are hidden from the label but **not** from the median.
This is also the whole of the elevated within-cell spread there (7.2–18.0% vs ≤1.1% elsewhere).

per-mamba is demonstrably batch-sensitive, so the gate as written confounds ctx with bs:

| sm | per_mamba @bs19 | @bs13 | 19/13 | @bs32 (c4096) | 32/13 |
|---|---|---|---|---|---|
| full | 0.1753 | 0.1640 | **1.069** | 0.1995 | 1.217 |
| 44 | 0.2183 | 0.1977 | **1.104** | 0.2632 | 1.331 |
| 24 | 0.3009 | 0.2642 | **1.139** | 0.3907 | 1.479 |
| 16 | 0.3980 | 0.3423 | **1.163** | 0.5312 | 1.552 |
| 8 | 0.6994 | 0.5930 | **1.180** | 0.9609 | 1.621 |

The 19→13 column is a **within-cell, same-ctx, same-server, same-arm** contrast — it is pure
batch, and in 3/5 arms it already exceeds the 1.15 gate threshold on its own. An affine model
`per_mamba = a + b·bs` fitted on (bs13, bs32) predicts the held-out bs19 point to within
0.0–1.4% in every arm. ⇒ **The ctx16384 column cannot be scored by GATE N at all.**

### 2.2 Re-scored on the bs=32 subset only — **the failure survives**

| sm | max/min over ctx {256,1024,4096} (all bs=32) | verdict | c256 / c1024 / c4096 |
|---|---|---|---|
| full | **2.780** | VIOLATION | 0.5546 / 0.4623 / 0.1995 |
| 44 | **1.971** | VIOLATION | 0.5186 / 0.3413 / 0.2632 |
| 24 | **1.276** | VIOLATION | 0.4976 / 0.3900 / 0.3907 |
| 16 | 1.003 | OK | 0.5305 / 0.5296 / 0.5312 |
| 8 | 1.003 | OK | 0.9598 / 0.9625 / 0.9609 |

(As specified, i.e. including ctx16384: **all five arms VIOLATE** — full 3.38, 44 2.62,
24 1.88, 16 1.55, 8 1.62. For sm=16 and sm=8 that failure is entirely the bs confound: their
bs=32 ctx-invariance is exact to 0.3%, and the affine-in-bs fit of §2.1 accounts for the
c16384 point. For full/44/24 it is not.)

### 2.3 Complement / negative control (methodology gate #10)

The same O(1)-in-ctx invariant applies to every non-attention bucket. `attn` is the positive
control that *must* move.

| bucket | bs=32-only max/min: full / 44 / 24 / 16 / 8 |
|---|---|
| `per_mlp` (÷9) | 1.004 / 1.003 / 1.004 / 1.002 / 1.004 — **clean everywhere** |
| `per_other` (÷54) | **3.186 / 2.907 / 1.866** / 1.015 / 1.010 — fails with mamba, same rank order |
| `per_mamba` | **2.780 / 1.971 / 1.276** / 1.003 / 1.003 |
| `per_attn` (positive ctrl) | 7.98 / 9.98 / 11.66 / 12.11 / 12.18 — moves, as it must |

So the estimator is **not** uniformly dead (mlp is invariant to 0.4%, attention responds), and
the failure is **not** mamba-specific: `other` fails in lockstep. This kills the reading
"mamba bucket is broken" and points at a cross-bucket attribution problem.

### 2.4 What the failure is (leading mechanism + what rules the alternatives out)

The violating cells are exactly the cells whose forward is **host-paced, not GPU-paced**:

- `fwd_ms` at fixed ctx across arms whose GPU work differs 2–8×: ctx256 → full **44.0**,
  sm44 **45.4**, sm24 **45.5**, sm16 51.2, sm8 93.0; ctx1024 → full **44.1**, sm44 **46.1**,
  sm24 62.5, sm16 87.7, sm8 164.8. Three/two arms pinned at the *same* ~44–46 ms floor
  despite very different device work = classic "the step time is set by the host issue rate".
  `fwd_ms` comes from an event pair independent of the buckets, so this is not bucket algebra.
- Cross the floor and the inflation disappears, sharply: sm24 c256 (fwd 45.5) inflated 1.276,
  c1024 (fwd 62.5) 1.000; sm44 c256 (45.4) 1.971, c1024 (46.1) 1.297, c4096 (92.3) 1.000;
  full c256 (44.0) 2.78, c1024 (44.1) 2.32, c4096 (**52.1**) 1.000.
- CUDA-event spans measure GPU-timeline elapsed, so when the device starves the idle is billed
  to whichever span is open. Multi-kernel, many-launch spans (mamba mixer, `other`) absorb it;
  the 2-GEMM `mlp` span does not — which is what §2.3 shows.
- Corroboration, coarse: engine-reported decode throughput at ctx256 peaks at 287 tok/s @bs32
  ⇒ ≥111 ms/step end-to-end vs a 44 ms layer loop; this configuration is host-heavy overall.
  (Aggregated over arms in the server log — not a per-arm number.)

Ruled out by data already in hand:
- **green-context artifact**: the *largest* violation (2.78×) is the `full` arm, which uses the
  base stream and no green context at all.
- **JIT / autotune / post-switch transient**: warm-up round discarded, first block of each round
  dropped, and round1/round2 agree to ±0.5% in all 20 cells (§5).
- **DVFS / thermal**: would make long-ctx (heavy) cells *slower*; observed direction is the
  opposite (mamba is *faster* at long ctx).
- **L2/memory pressure from a large KV cache**: same sign argument — would slow long ctx.
- **Node/GPU heterogeneity**: all four tasks ran on gpu41.

Not ruled out, and not separable offline:
- **Co-tenancy**: tasks ran in overlapping pairs on gpu41 (ctx256 ‖ ctx1024, 09:58–10:00;
  ctx4096 ‖ ctx16384, 10:01–10:26). Host CPU pressure differs along the ctx axis and is aligned
  with the violation pattern. It can only set the *height* of the host floor (which cells are
  contaminated), not create the mechanism — bounded by the 0.28–0.31% cross-task reproducibility
  of the GPU-bound arms (§5) — but it should be eliminated or recorded before rep2/rep3.
- **The instrumentation's own host cost**: ~2 event records per span × ~5 spans/layer × 54
  layers, plus a per-forward `torch.cuda.synchronize()` and ~270 `elapsed_time()` read-backs,
  all on the issuing thread. This plausibly *raises* the host floor and therefore *enlarges*
  the contaminated region. Untested here; testable directly (`SGLANG_ZAMBA_TIMING=0` step time).

### 2.5 Should the gate be redefined?

Yes for the **bs confound** — GATE N compares `per_mamba` across cells whose realised batch
differs (32 vs 19/13), and the batch effect alone is 1.07–1.18× (19→13) to 1.62× (13→32).
The invariant "mamba decode is O(1) in ctx" is only testable **at fixed bs**. Redefinition
required: score GATE N per (ctx, **realised bs**) cell, and make the CELL line carry the batch
*mixture*, not its mode.

No for the **threshold**: 1.15 must not be re-tuned against this run, and no host-boundness
cut-off may be picked from these data (methodology gate #8). Any such threshold has to be
pre-registered before batch 2 and audited independently — note also that the diagnosis in §2.4
and any prescription derived from it come from the same turn (methodology lesson 12), so the
prescription below is a **candidate**, not a design decision.

---

## 3. G6 — batch 2/3 submission verdict: **NO-GO as-is (conditional)**

Not "the data are worthless": the run is a *valid* measurement in the GPU-bound subregion
(10 of 15 bs=32 cells, ctx-invariance exact to 0.3%). But:

1. Reps add `n` to a **systematic, sign-consistent** bias; they cannot average it out. Two more
   reps would tighten intervals around a contaminated point estimate in exactly the cells the
   P5 question is about (short ctx × high SM — the corner any "knee vs ctx" reading needs).
2. GATE N as written cannot be scored at ctx16384 (bs confound, §2.1), so 1 of 4 ctx columns
   would come back unscoreable again.
3. Two harness/analyzer defects (§6) would be replicated 8 more times.

Candidate remedies (**unaudited, not pre-registered, listed as options only**): (a) hold the
realised batch equal across ctx so GATE N is scoreable; (b) raise device work per forward above
the host floor (batch/duty) — but this changes the estimand and is impossible at ctx256×full
with bs=32 (device work ≈18 ms vs a ≈44 ms floor); (c) measure at the project's actual operating
point (cudagraph ON), which removes most host pacing — feasibility of per-layer event spans
inside a graph is unverified; (d) pre-register a host-boundness admissibility criterion *and*
accept that it excludes the short-ctx/high-SM corner; (e) serialise array tasks (no co-tenancy)
and record `SGLANG_ZAMBA_TIMING=0` step times to size the instrumentation's own host cost.

**Minimum before resubmission:** fix GATE N's batch confound (§2.5) and pick one of (a)-(d)
with the criterion pre-registered and audited. rep2/rep3 as literal repeats of batch 1 are not
worth the GPU hours.

---

## 4. G1 (GATE C, closure ≥ 0.85) — **20/20 PASS**, and the covariation matches launch-gap

All 20 cells pass on the per-cell median (range **0.9300** (ctx256, sm44) – **0.9988**
(ctx16384, sm8)); worst single-forward `closure_min` over all 200 blocks is 0.9286, also above
0.85. Zero closure failures in any measured round.

Covariation: `spearman(fwd_ms, closure) = +0.991` (n=20) — slow cells sit at 1, fast cells low,
exactly the pattern flagged in the task. It decomposes into two parts:

- **a hard floor**: `unbucketed_ms` = 0.766–0.795 ms/forward in *every* GPU-bound cell
  (sm8 all ctx; sm16 at ctx 256/1024/4096) — a per-forward fixed event/sync cost. This alone
  makes closure ≈ `1 − 0.77/fwd_ms`, i.e. mechanically lower in fast cells.
- **excess above the floor**, present *only* in host-bound cells: +1.6 (256,full), +2.4
  (256,sm44), +1.3 (1024,full), +1.2 (1024,sm44), +0.36 (256,sm24), ≤0.32 everywhere else.
  Correlation with the in-span mamba inflation across the 15 bs=32 cells: **pearson +0.892**
  (both are driven by the same latent host-boundness; they are not the same quantity — idle
  billed *inside* a span produces inflation with no closure deficit — but they are not
  independent evidence either).

**This is a post-hoc validation of the 0.85 calibration, with one important limit.** The
threshold was chosen for defect 1 (bucket asymmetry ⇒ closure ≈0.70), and the observed healthy
population (0.93–0.999) sits clear of it; the closest cell reaches only 0.28–0.47× of its trip
point (a FAIL at (256,full) would need 6.60 ms unbucketed; 2.40 observed). But:

> **GATE C is blind to the defect that actually fired.** Starvation billed *inside* spans leaves
> closure at 0.93–0.95 while inflating `per_mamba` by up to 2.78×. This run is the existence
> proof: **20/20 GATE C PASS with 3–5/5 GATE N VIOLATION**. "Closure passed" must never be
> read as "the per-type numbers are a measurement". The two gates are non-redundant, and
> keeping GATE N is vindicated.

Side note on discriminating power: the one closure failure in the whole job is
`ZBLT_CLOSURE_FAIL mode=44@1024rw blk=0 closure=0.7119` — a **discarded warm-up** block, whose
value (0.71) coincides with the legacy defective recompute (0.6997). So ≈0.70 has at least two
producers (mis-bucketed span set; a one-off host stall). GATE C's power against defect 1 rests
on the deficit being *persistent*, not on the value.

---

## 5. G5 (defect 2, warm-up / arm order) — **fixed, residual ≤0.31%**

- **Randomisation fired.** `sm_order` differs across tasks: ctx256 `16 full 24 8 44`,
  ctx1024 `24 8 full 16 44`, ctx4096 `full 44 8 24 16`, ctx16384 `full 44 8 24 16`
  (3 distinct orders / 4 tasks; `full` first in 2 by chance). Regenerating the same
  seeded shuffle for tasks 4-11 gives, for rep2/rep3: `8 full 16 24 44`, `24 44 full 16 8`,
  `16 8 full 24 44`, `16 44 24 full 8`, `24 44 full 16 8`, `16 24 8 44 full`,
  `24 44 8 16 full`, `8 full 24 16 44` — **`full` is first in 0 of 8**. At `n_indep=1`,
  arm position is still confounded with ctx; only reps break that.
- **Which blocks were dropped, and why.** Exactly 2 per cell = the first block of each of the
  2 measured rounds (post-switch transient). **Zero** blocks were dropped as `dirty` in any
  cell. The separate warm-up round is discarded by construction and never enters the file.
- **Residual position bias inside the kept blocks.** round1/round2 medians agree to
  **±0.5%** in all 20 cells (`per_mamba` 0.998–1.005, `per_attn` 0.998–1.002, `fwd` 0.996–1.003).
  First kept block vs the rest of its round: within **±1.2%** for all ctx 256/1024/4096 cells.
  At ctx16384 that ratio is 1.39–1.49× — **that is the batch trajectory (bs 19→13), not a
  warm-up transient**; the first kept block is a bs=19 block in every ctx16384 cell.
- **Direct cross-position test** (only possible for arms whose estimator is uncontaminated):
  `per_mamba` at sm16 = 0.5305 / 0.5296 / 0.5312 at run positions **1 / 4 / 5** in three
  different tasks (spread **0.31%**); sm8 = 0.9598 / 0.9625 / 0.9609 at positions **4 / 2 / 3**
  (spread **0.28%**). Defect 2's "first-arm bias in every denominator" is gone to <0.31%.

---

## 6. G4 (defect 1 magnitude) + G3 (defect 3) + tooling defects

### 6.1 G4 — `per_attn / per_attn_core`

```
   sm    c256    c1024   c4096   c16384        absolute miss 9*(per_attn-per_core), ms/forward
 full   1.585    1.184   1.040   1.028          1.72   1.73   1.43   1.36
   44   1.336    1.098   1.022   1.015          1.86   1.85   1.55   1.77
   24   1.272    1.065   1.016   1.012          2.53   2.22   2.22   2.58
   16   1.233    1.063   1.016   1.012          3.20   3.20   3.21   3.67
    8   1.222    1.060   1.015   1.011          5.91   5.93   5.90   6.97
```
Read the **absolute** column, not the ratio: what the old bucket missed (qkv+o_proj+the module's
own glue) is **1.4–5.9 ms/forward** and is stable within an arm at fixed bs (sm8: 5.91/5.93/5.90
across three ctx = 0.5% spread), scaling with SM budget as GEMM work should. As a share of the
whole forward it is **1.05%–6.35%**. The *ratio* explodes at short ctx only because its
denominator (attention core) collapses — and at (256,full) part of the numerator is itself
host-starvation inflation. So the 3.02× seen in the single-block smoke and the 1.585× here are
**upper bounds contaminated in host-bound cells**, not the size of defect 1.

### 6.2 G3 — is the physical-invariant violation fixed? **Partly. Answer differs by ctx.**

| ctx | per_mamba full(108 SM) / sm44 | sign |
|---|---|---|
| 256 | **1.069** | violation (108 SM slower) |
| 1024 | **1.354** | violation (108 SM slower) |
| 4096 | 0.758 | OK |
| 16384 | 0.830 | OK |

858811's 2.34× at ctx256 is down to 1.069, and the sign is now correct at ctx 4096/16384 — but
the violation is **not eliminated**, it has retreated to the host-bound cells. Same story for
monotonicity of `per_mamba` in SM count (8→16→24→44→108): **not monotone** at ctx256
(0.9598/0.5305/0.4976/0.5186/**0.5546**) and ctx1024 (0.9625/0.5296/0.3900/0.3413/**0.4623**);
**monotone** at ctx4096 and ctx16384.

### 6.3 Tooling defects found (both should be fixed before batch 2)

1. **Analyzer contradicts itself on GATE C.** It prints `GATE C (closure >= 0.85): PASS` and
   then `GATES: closure=FAIL`, because `gates_ok` counts `ARMDONE closure_fail`, and the
   harness greps `ZBLT_CLOSURE_FAIL mode=$SM@${CTX}r` — which also matches the **discarded
   warm-up** tag `rw` (`decode_knee_vs_ctx_v2.sbatch:152`). The single failure in this job is a
   warm-up block. Scope the count to measured rounds, or report warm-up failures separately.
2. **`mamba_spread` column is identity-valued at n=1.** It is max/min over *reps*, so it prints
   `1.00` for all 20 cells with rep1 only. It must not be read as stability (methodology gate
   #6); the real block-to-block spread is in TABLE A (`blk spread%`).
3. **CELL `bs` = `statistics.mode`** silently labels a 6×bs19 + 8×bs13 mixture as `bs=13`.

---

## 7. One-line verdicts

| gate | verdict |
|---|---|
| G1 GATE C | **PASS 20/20** (0.9300–0.9988); covariation matches launch-gap/host-pacing; 0.85 calibration validated *for defect 1 only* — GATE C is blind to the defect that fired |
| G2 GATE N | **FAIL** — all 5 arms as specified; full 2.78× / sm44 1.97× / sm24 1.28× even on the bs=32-only subset; **and the gate itself is mis-specified** (confounds ctx with realised bs at ctx16384) |
| G3 defect 3 | **partly fixed** — 2.34× → 1.069× (ctx256) / 1.354× (ctx1024); sign now correct at ctx 4096/16384; violation retreated to host-bound cells |
| G4 defect 1 | **fixed and quantified** — old bucket missed 1.4–5.9 ms/forward (1.05–6.35% of the step); the 1.02–1.59× ratio is denominator-driven |
| G5 defect 2 | **fixed** — randomisation fired, round1/round2 ±0.5%, cross-position reproducibility 0.28–0.31% |
| G6 | **NO-GO for rep2/rep3 as literal repeats.** Fix GATE N's batch confound + pre-register a host-boundness criterion (or move off the host-bound operating point) first |
