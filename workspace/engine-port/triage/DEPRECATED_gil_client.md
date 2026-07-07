# ⚠️ DEPRECATED — hand-rolled GIL client (`bench_client.py`) serving outputs

**Marked 2026-07-06 (R-series realignment, R2).** The measurement *client* and the
serving outputs it produced (listed below) are **deprecated**. The data files are kept
in place (reversal-history is preserved; nothing is deleted or moved) but must **not** be
cited as serving evidence. Use the clean async re-measurements instead.

## What is deprecated

- **Client**: `bench_client.py` — a hand-rolled 60-thread `urllib` streaming load
  generator. It is **GIL-bottlenecked**: the client itself cannot drain tokens fast
  enough, so it inflates TPOT (Zamba2 in3600/out32 showed TPOT ~170–205 ms that is a
  *client* artifact — the clean async client measures agnostic TPOT ~40 ms in the same
  regime) and it was run **past saturation** (rates 6/12/18), where goodput at the SLO
  cliff is a coin-flip.
- **Zamba2 serving outputs** (replaced by R0a): `p1_4_zb_serving.sbatch` →
  `p1_4_zb_serving_{831146,831166,831541}.txt`, `p1_4_zbsrv_{831146,831166,831541}_*.log`.
  These are the source of the **old §04 Zamba2 TTFT panel** in `sm_policy_report.html`
  (the "layer-aware beats agnostic on TTFT" reading), which the clean data **refutes**.
- **NemotronH serving outputs** from the same client (`p1_4_serving.sbatch` →
  `p1_4_serving_{831023,831070,831140,831141,831172,831173}.*`,
  `p1_4_srv_*_{fused,agnostic,layer_aware}.log`) were already superseded by the clean
  async NemotronH runs (see below); notably the earlier "agnostic_v2 best at high load"
  ranking was a GIL-client + past-saturation artifact (see `sm_policy_report.html` §05
  "Methodology correction").

## Replacements (authoritative)

- **Client**: `python -m sglang.bench_serving` (official async, `random-ids`,
  sub-saturation sweep).
- **Zamba2 clean async** — in3600/out32 (§04 prefill-bound panel): R0a jobs
  **834914 (agnostic) / 834915 (fused) / 834916 (layer_aware)** + reps
  **834927 / 834929 / 834928**; outputs in
  `workspace/engine-port/results/r0a/` (`r0a_summary.csv`, per-run CSVs).
- **Zamba2 clean async** — in2000/out96 (§07 generalization table): jobs
  **832701 (agnostic) / 832702 (layer_aware)**.
- **NemotronH clean async** — in2000/out96: jobs **831609/831610/831611/831612**;
  `p1_6_clean_async_results.txt`.

## Bottom line (what the clean data shows)

layer-aware ≤ agnostic on every hybrid measured; on Zamba2 in3600/out32 clean async,
layer-aware goodput@SLO is **0 from rate 2** (agnostic sustains ~2.07 req/s at rate 2).
Canonical write-up: `workspace/engine-port/reports/sm_policy_report.html` (§04, §07).
