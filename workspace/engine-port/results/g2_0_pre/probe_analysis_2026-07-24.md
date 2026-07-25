# G2.0-pre capacity/cliff probe — analysis (2026-07-24)

**Nature: n=1 capacity/cliff probe, NOT a policy comparison.** Purpose = locate the
metric cliff and capacity before the full n>=4 d16/d24/d34/d44 sweep (method gate #6).
No statistical policy claim is made or permitted here.

## Setup (verified from sbatch + telemetry)
- Zamba2-2.7B, ctx4096, cudagraph operating point, **d24 static only** (split
  `manual_divisions [84 prefill, 24 decode] SM`, sm_group_num 3).
- Phase A = prefill-heavy in2048/o32; Phase B = decode-heavy long-ctx in2048/o512.
- `ROUNDS=1, NPROMPT=32` per phase (single round -> gate #7 duration-sum N/A here; the
  sbatch scorer nonetheless implements `dur+=d` correctly for the multi-round sweep).
- SLO: **canonical goodput = TTFT<=3000ms AND per-request token-ITL p95<=50ms.**
  sbatch's own `G20_RESULT` uses **mean-ITL<=60ms (legacy/secondary)** — numbers below
  are RE-SCORED with the canonical p95-ITL definition; they differ from the .out files.
- Health: all 7 jobs boot_ok=1, SANITY pass, **32/32 completed, 0 errors, 0 OOM**. The
  `.err` "Killed" line is the end-of-job server teardown (kill $SRV), not a crash.

## 1. Per-rate table (canonical p95-ITL goodput)

### Phase A (prefill-heavy in2048/o32) — TTFT-bound; ITL never binds (p95 29-39ms)
| tag    | rA | dur s | thru r/s | ttft p50 | p90  | p95  | p99  | itl p95 | good r/s@50 | good/off |
|--------|----|-------|----------|----------|------|------|------|---------|-------------|----------|
| rA4B4  | 4  | 9.8   | 3.27     | 362      | 657  | 1396 | 1494 | 28.8    | 3.07        | 0.94 |
| rA5B4  | 5  | 7.9   | 4.04     | 399      | 882  | 1298 | 1457 | 33.5    | 3.53        | 0.88 |
| rA6B4  | 6  | 8.6   | 3.70     | **2923** | 3846 | 3870 | 3906 | 37.0    | 1.97        | 0.53 |
| rA7B4  | 7  | 10.0  | 3.22     | 1933     | 4410 | 4814 | 4855 | 39.4    | 1.91        | 0.59 |
| rA8B4  | 8  | 8.7   | 3.67     | 1907     | 4153 | 4368 | 4510 | 38.7    | 2.53        | 0.69 |
| rA6B3  | 6  | 9.2   | 3.47     | **3507** | 4426 | 4456 | 4487 | 36.7    | 0.65        | 0.19 |
| rA6B5  | 6  | 8.6   | 3.71     | **2917** | 3837 | 3865 | 3898 | 36.9    | 1.85        | 0.50 |

### Phase B (decode-heavy in2048/o512) — ITL-bound; TTFT stays low (<1s) except rA8
| tag    | rB | dur s | thru r/s | ttft p50 | p90  | p95  | p99  | itl p95  | good r/s@50 | good/off |
|--------|----|-------|----------|----------|------|------|------|----------|-------------|----------|
| rA4B4  | 4  | 23.0  | 1.39     | 319      | 674  | 723  | 729  | **66.8** | 0.479       | 0.34 |
| rA5B4  | 4  | 22.9  | 1.39     | 324      | 684  | 732  | 738  | **65.8** | 0.479       | 0.34 |
| rA6B4  | 4  | 23.0  | 1.39     | 318      | 723  | 740  | 786  | **65.2** | 0.479       | 0.34 |
| rA7B4  | 4  | 22.9  | 1.40     | 321      | 731  | 764  | 785  | **66.3** | 0.480       | 0.34 |
| rA8B4  | 4  | 28.0  | 1.14     | 729      | 2741 | 3108 | 3760 | 54.7     | 0.500       | 0.44 |
| rA6B3  | 3  | 24.2  | 1.32     | 282      | 609  | 679  | 724  | **68.5** | 0.414       | 0.31 |
| rA6B5  | 5  | 22.4  | 1.43     | 500      | 817  | 900  | 975  | **64.8** | 0.492       | 0.34 |

(@60ms ITL SLO Phase B is essentially unchanged: p95 still 65-68ms > 60, good/off ~0.34.)

**Phase B rB4 pseudo-replicate (rA4/5/6/7, excl. rA8):** goodput 0.479 r/s SD 0.0004,
ITL-p95 66.0ms SD 0.59, throughput 1.394 r/s SD 0.0012. Extremely low noise — this
decode-bound regime is reproducible (opposite of the deprecated stationary ShareGPT r8).

## 2. Cliff and capacity

- **Phase A TTFT cliff onset = rate 6.** rate<=5: p50 360-400ms, p90<=880ms (huge 3s
  margin). At rate 6 p50 jumps to 2923-3507ms sitting **ON the 3s boundary**; p90 already
  >3000. Textbook metric cliff (gates #4/#6): the rate-6/7/8 rows are non-monotone
  (rate7 p50 1933 < rate6 p50 2923) precisely because n=1 on the cliff is unreliable.
- **Phase A capacity ~= 4 req/s** (throughput plateaus at 3.2-4.0 r/s for offered 4-8;
  offered >=6 exceeds capacity -> backlog -> TTFT cliff). ITL never binds in Phase A.
- **Phase B: ITL terminates first, not TTFT.** TTFT stays ~320ms (p90 ~700) at rB<=5,
  far under 3s. But per-request ITL-p95 median is 71ms and aggregate p95 ~66ms > both
  50 and 60ms SLO at **every** tested rB (3,4,5). Phase B is decode-SM-starved, not
  TTFT-limited.
- **Phase B capacity ~= 1.4 req/s** (throughput flat 1.32-1.43 across rB 3/4/5; even
  rB=3 is over capacity — 32 prompts inject in ~11s but makespan ~24s). No cliff-free
  operating point exists in this grid for the p95-ITL SLO on d24.
- **rA8B4 is contaminated:** its Phase B shows TTFT p50 729 / p90 2741 and *lower* ITL
  p95 (54.7) — the rate-8 Phase A backlog bled into the start of Phase B (phases share
  one server, run sequentially). Treat rA8 Phase B as a sequential-carryover artifact,
  not signal. Full sweep should drain/settle between phases or cap rA sub-cliff.

## 3. Disjoint-structure early signal (d24 only, NOT a claim)

**Signal: PRESENT (weak, n=1).** At the same d24 split and low rate (rA4-5 / rB4), d24
**passes Phase A** (good/off 0.88-0.94, TTFT far under SLO) but **fails Phase B**
(good/off 0.34, per-request ITL-p95 median 71ms >> 50). d24 = 84 prefill / 24 decode SM:
prefill-abundant, decode-starved. This is the expected asymmetry (prefill-heavy fine,
decode-heavy ITL-violating) and is consistent with the entanglement/decode-floor story.
BUT it does NOT prove the split moves the outcome — the decisive test is whether **d44
(44 decode SM) pulls Phase B ITL-p95 under 50ms** while d16 keeps Phase A cliff-free at
higher rate. Cannot be judged from d24 alone. Verdict: **signal present, confirmation
requires the d16/d34/d44 arms.**

## 4. Recommended full n>=4 sweep measurement points

- **Primary: rA5 / rB4.** Phase A sub-cliff with margin (p50 ~400, p90 ~880, throughput
  already at ~4 r/s = near-capacity so prefill-SM split will differentiate: d44's 64
  prefill SM should cliff earlier than d16's 92). Phase B ITL-binding with safe TTFT
  (p90 ~700ms), so decode-SM split is decisive (test: does d44 bring ITL-p95 < 50?).
  Both phases stressed enough to expose the disjoint split conflict, neither on the cliff.
- **Secondary (optional stress): rA6 / rB4** — maximizes Phase-A split differentiation
  but sits ON the TTFT cliff; admit **only with n>=4 AND full TTFT distribution +
  sustainable-rate reporting** (gate #6), since single runs there swing 2x.
- Avoid rB>=5 (pushes Phase B TTFT up, mixes binding constraints) and rA>=7 (Phase A
  over-cliff + contaminates the following Phase B).
- Sweep hygiene: ROUNDS=3, confirm duration **summed** not maxed (sbatch already does
  `dur+=d`); report per-request token-ITL p95 (canonical), not the sbatch mean-ITL; keep
  baseline variance (n>=4) before any d-vs-d claim; do not re-score across SLOs.

## Anomalies
- No OOM / no errors / no degenerate runs; all 32/32 completed.
- sbatch headline scorer uses **mean-ITL<=60ms** (legacy) — will over-count Phase B
  goodput vs canonical p95-ITL (0.34 pass under p95 vs higher under mean). Use canonical.
- rA8B4 Phase B sequential carryover (see 2).
- Phase A cliff-region (rate 6-8) non-monotonicity = expected n=1-on-cliff instability.
