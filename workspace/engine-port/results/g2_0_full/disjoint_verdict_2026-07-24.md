# G2.0 full static sweep — DISJOINT verdict (2026-07-24)

Canonical scoring (`pdmux_eval.analyze.summarize_requests`, percentile def, seed rules):
request good iff **TTFT ≤ 3000ms AND that request's token-ITL p95 ≤ 50ms**;
`slo_goodput_req_s = good / duration`. Zamba2-2.7B, ctx4096, cudagraph ON, rA5/rB4
fixed, n=4 per mode. Split = [prefill,decode] SM: d16[92,16] d24[84,24] d34[74,34]
d44[64,44] d54[54,54]. ROUNDS=1 (gate #5 N/A). All 40 runs 32/32 completed, 0 real
errors. Phase-B first-request TTFT ~164ms every rep ⇒ no Phase-A carryover despite
`drain_ok=0` (poll never caught running==0; server was near-idle at handoff).

## Phase A (prefill-heavy in2048/o32, rA=5) — TTFT-bound
| mode | good r/s (mean±SD) | frac_good | thru r/s | ttft p50/p90/p95/p99 ms | iITLp95 med |
|------|--------------------|-----------|----------|--------------------------|-------------|
| d16  | 3.857±0.239 | 0.953±0.060 | 4.046 | 323/717/1104/1160 | 44.1 |
| d24  | 3.079±1.347 | 0.789±0.279 | 3.747 | 669/1999/2286/2495 | 33.9 |
| d34  | 2.402±1.559 | 0.633±0.353 | 3.605 | 1907/2699/2798/2966 | 25.5 |
| d44  | 3.726±0.032 | 0.969±0.000 | 3.846 | 1142/1435/1600/1750 | 17.1 |
| d54  | 2.798±0.946 | 0.852±0.214 | 3.209 | 2154/3254/3479/3655 | 19.3 |

Robust (SD~0, cliff-free p90<3s): **d16, d44**. d24/d34/d54 are bimodal — TTFT sits
on the 3s cliff (d34 rep1/rep4 t50 3063/3385; d54 rep1 collapses to 0.531, p90 3254 >
3000). Phase A ITL never binds (except d16 marginal 44.1).

## Phase B (decode-heavy in2048/o512, rB=4) — ITL-bound, clean/reproducible
| mode | good r/s | frac_good | thru r/s | ttft p50/p90/p95/p99 ms | iITLp95 med (ms) |
|------|----------|-----------|----------|--------------------------|------------------|
| d16  | 0.552±0.000 | 0.406±0.000 | 1.358 | 305/695/735/875   | 95.6±0.1 |
| d24  | 0.460±0.031 | 0.344±0.000 | 1.339 | 361/782/895/994   | 71.6±0.2 |
| d34  | 0.351±0.000 | 0.250±0.000 | 1.406 | 376/829/930/1023  | 59.8±0.5 |
| d44  | 0.264±0.003 | 0.188±0.000 | 1.409 | 582/1145/1300/1382| 50.8±0.3 |
| d54  | 1.413±0.002 | 1.000±0.000 | 1.413 | 834/1533/1625/1868| 44.2±0.1 |

Per-request ITL-p95 median falls **monotonically** with decode-SM: 95.6→71.6→59.8→
50.8→44.2, crossing the 50ms SLO **between d44 and d54**. SD ≤0.5ms, frac_good SD=0 —
this decode-bound regime is essentially noise-free. No TTFT cliff in Phase B (p99<1.9s).

## Core question: does d44/d54 pull Phase-B ITL-p95 < 50 while failing Phase A?
- **d54 (54 decode SM):** yes — B ITL-p95 med 44.2ms, 32/32 pass every rep (0.188→1.000
  jump vs d44). And it is **fragile in Phase A** (0.852, one collapse to 0.531, p90/p95
  = 3254/3479 > 3000 ⇒ over the TTFT cliff). So d54 clears B decisively but is NOT
  robustly Phase-A-feasible.
- **d44 (44 decode SM):** robustly clears Phase A (0.969±0.000, cliff-free) but its B
  ITL-p95 med is 50.8ms — **one SM-step short**, 6/32 pass. Last prefill-safe split,
  misses the ITL SLO by a hair.

## Feasibility sets
- **frac_good ≥ 0.90 (robust bar):** feasible-A = {d16, d44}; feasible-B = {d54};
  **intersection = ∅ → DISJOINT.**
- **frac_good ≥ 0.50 / median-request bar:** feasible-A = all 5; feasible-B = {d54};
  intersection = {d54}. Under the lax bar d54 is a shared static — but it sits on the
  Phase-A TTFT cliff (p90 3.25s, one collapse), so it is not a *clean* single-static.

## Statistical robustness / cliff hygiene
- Phase B result is rock-solid (monotone across 5 modes, SD≤0.5ms, frac SD=0) — the
  decode-SM→ITL constraint boundary is structural, not noise.
- Phase A robustness only holds for d16/d44; d24/d34/d54 are TTFT-cliff bimodal (gate
  #4). The disjointness therefore hinges on d54 being over the Phase-A cliff, which it
  structurally is (p90>3s), but the margin is thin.
- The disjoint is **real but narrow/cliff-adjacent**: both constraint boundaries cross
  in the single SM-step between d44 and d54. No wide gap.

## VERDICT — (a) DISJOINT is REAL (narrow, monotone, cliff-adjacent)
At the robust frac≥0.9 criterion the phase-feasible split sets are disjoint: the only
split that meets Phase-B ITL (d54) is Phase-A-fragile, and every split that robustly
meets Phase-A TTFT (d16,d44) misses Phase-B ITL (d44 by 0.8ms). Phase A wants decode
≤44 SM; Phase B needs decode ≥54 SM; the boundaries do not overlap.

⇒ **Vector-1 (disjoint conflict-regime) NOT killed.** A single-GPU static split cannot
robustly serve both phases at this operating point ⇒ workload basis for a dynamic /
decoupling escape-hatch is **established for this A/B regime** — but narrowly, and the
upside a dynamic policy could capture over best-fragile-static d54 is bounded (Phase-A
0.85→~0.95; Phase-B already 1.0 at d54).

## To convert "narrow/marginal" → "robust wide disjoint" (recommended next)
The crossing sits exactly between the two adjacent extreme splits, so it is razor-thin.
To widen and de-cliff: (i) make Phase B heavier (longer o or higher rB) so ITL needs
decode ≫54 SM, pushing feasible-B further from feasible-A; and/or (ii) add n≥6 at d54
Phase A (or a slightly sub-cliff rA) to pin whether d54-A is 0.85 or genuinely <0.5 —
current d54-A is bimodal on the cliff. Do NOT re-score across SLOs (gate #8); re-run.
