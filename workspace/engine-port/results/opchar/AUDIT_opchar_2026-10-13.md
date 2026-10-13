# opchar — claims-auditor verdict (2026-10-13, read-only audit of commits a9d55ad..9b004fb)

Scope: VESSL A100, one GPU UUID (GPU-32e4d449…, full + ctrl Jobs), isolated engine kernels, random
weights, n=30 medians, two Jobs on the same GPU (median diff 0.0% at 300 shared points). CHARACTERIZATION
ONLY — no serving claim, no policy implication (layer-type policy death, HE0, gate #13 untouched).

| claim | verdict |
|---|---|
| C1 profiling substrate | PLAUSIBLE (cond.). nsys absent, Kineto empty (cause unknown), ncu ERR_NVGPUCTRPERM for **LaunchStats+Occupancy together** and a counter metric — "blocked even for counter-free sections" is OVERREACH (Occupancy collects counters; LaunchStats alone never run). Triton log and graph census agree on all 14 Triton launch tuples. Not substitutes for: durations, residency b, eager non-Triton launches. |
| C2 graph census, 0/60 change with partition | CONFIRMED (scoped: graph-captured launches, this image/versions/sizes). Eager not directly censused (indirect support: eager prefill steps land at census-predicted SM counts). |
| C3 104→108 decode GEMM jump = wave quantization | PLAUSIBLE (strong). The GEMM is split-K (35×1×3 = 105 CTAs, b=1 exact; reduce kernel 560 blocks, 2 waves at both). Stronger evidence: green-only prefill steps (up_proj 96→100, in_proj 100→104 at predicted wave changes); constant-wave plain-108 = green-104. Residual: one GPU, no 106-green arm. B=32 qkv/up slow down at constant waves — not wave-explained. |
| C4 Fig 2(a) shares | PLAUSIBLE (cond.). Estimand = layer-count-weighted sum of ISOLATED op medians (no embedding/lm_head/overlap). Prefill L=1k exposed to eager launch latency; decode graph replay reuses weights in L2; 108 point throttled; ctx fixed 2048. |
| C5 Table 2 | SSM Triton idle 0–5%: CONFIRMED (derived; grid-driven, b register-bound; equal-weight would give ~19% at 1k/104). GEMM partition dependence (out_proj 0%@108 vs 16.9%@104): CONFIRMED (derived; timing −16% vs −20% predicted). **Attention cells: not citable** (b=2→1 flips 25.9%→1.2%). **REFUTED: full Job `ANALYSIS/table2_idle.md` GEMM cells (assumed tile; census contradicts).** Bullet positive-control test is an identity (tile chosen to fit). |
| C6 Fig 3 | PLAUSIBLE (cond.). Compute-bound ops NOT "~linear": ≈1.13× linear to 64 SM, then −11–13% per-SM efficiency from 74 SM (power-cap throttle + monotone run order confound). SSD scan 84% of linear at 1k. Attention at L=1k strongly sublinear. Memory-bound saturation real (≥86% of 108-SM speed by 44 SM at 16k, isolated, no HBM contention). L=1k small-op points not citable (±76% between Jobs). |

## Safe wording (verbatim, with scope)
"VESSL A100 (single GPU UUID GPU-32e4d449…), isolated engine kernels, random weights, n=30 medians, two
Jobs on the same GPU — characterization only, not serving, no policy implication. (1) Kernel choice and
launch grid of all 60 graph-captured operator configs are identical across green partitions 44–104 SM and
the 108-SM plain stream. (2) Several GEMMs show time steps at exactly the SM counts where the recorded
grid's wave count changes, including between green partitions; the 104→108 decode GEMM drop coincides
with a 2→1 wave change of a 105-CTA split-K GEMM, and 1-wave GEMMs show no 104-green/108-plain difference
(a 106-SM green arm has not been run). (3) In the sum of isolated per-operator times, Mamba layers account
for ~66–71% of prefill (L=1k–32k) and the SSM state update 36%→64% of decode (B=32→128, ctx 2048).
(4) Memory-bound operators reach ≥86% of their 108-SM speed by 44 SMs in isolation."

## Must not cite
"% of model time"; any attention idle cell; full-Job Table 2 GEMM cells / "Bullet-validated" wave model;
"~linear to 108" for GEMM/attn; Fig 2 util bars as hardware utilisation; L=1k small-op Fig 3 points;
"ncu blocked for counter-free sections"; "green contexts have no cost" in a PD-mux/co-running sense (says
nothing about switch cost, gate #13); any partition-sizing or policy inference; any KISTI comparison.
Also: commit fff9b1d "prefill stays smooth" is wrong (L=1k green-only steps exist).

## Resolving experiments (proposed, not run)
106-SM green arm interleaved with 104 green / 108 plain (C3; second GPU UUID); randomized/reversed SM
order with 108 bracketed + per-run UTC stamps (C6 throttle); b via cuOccupancyMaxActiveBlocksPerMultiprocessor
on the census CUfunction (C5 attn); CUBLASLT_LOG_LEVEL=5 eager GEMM choice (C2 eager); ncu LaunchStats alone
(C1); L2-flush decode arm + graph/back-to-back prefill at L=1k + end-to-end forward calibration (C4).

## Resolution run — ctrl2 Job job-h0ro98uvs4gx (050483c, 2026-10-13), GPU UUID GPU-6ca4a276… (a SECOND GPU)
Facts from the run; grading changes below are the run's mechanical outcome against the predictions
registered in commit 050483c, not a new audit.

- **C3 (registered prediction HELD, second GPU).** One process, arms interleaved over 10 shuffled rounds
  (n=100 pooled per arm/op): decode B=128 o_proj 104g 0.0688 / **106g 0.0471** / 108p 0.0468 ms;
  down_proj 0.1744 / **0.1127** / 0.1124; out_proj 0.1197 / **0.0779** / 0.0779; 1-wave control in_proj
  0.1395 / 0.1398 / 0.1400. Census: main split-K GEMM 1 wave at 106 and 108, 2 at 104 (b=1 driver).
  Prefill out_proj (432 CTAs) goes the other way as its waves predict (5 @104/106, 4 @108: 0.478/0.490/0.406).
  → the 104↔108 step is wave quantization; green-vs-plain shows no effect at equal waves (isolated kernels).
- **C2 eager (resolved).** CUBLASLT_LOG_LEVEL=5 on eager prefill (L 1k/16k): every heuristic query carries
  `smCountTarget=108` on the 104-SM green partition too; the chosen algo (id/tile/stages/splitK) is
  identical for all 12 GEMM shapes at 104 and 108.
- **C1 (resolved).** `ncu --section LaunchStats` alone: ERR_NVGPUCTRPERM. ncu is unusable on this image.
- **C5 attention residency.** Driver occupancy b=2 for the FlashInfer ragged prefill kernel at L=16384
  (smem 65568) = calc. The L=1024 variant (smem 69664, the flip case) was not in this run → still not citable.
  All 288 census kernels now carry driver b.
- **C6 order confound (removed; cause open).** SM order shuffled, 108 first and last (drift ≤1.2%).
  Prefill L=16384 per-SM efficiency (speedup ÷ linear) still ~1.12 flat to 64 SM, then 1.10/1.05/1.01/1.00
  at 74/92/104/108. Power-cap events occur only in runs at ≥74 SM regardless of run position (11–18% of
  samples), but the median SM clock in those runs is still 1410 MHz → time order excluded; a power-limit
  cause is NOT established.
- Two-GPU replication now exists for decode GEMM values (o_proj 108p: 0.0453–0.0456 on GPU-32e4…, 0.0468 here).

Safe wording addition (same scope rules): "On a 106-SM green partition the 105-CTA split-K decode GEMMs time
within 0.6% of the 108-SM plain stream and ~1.5× faster than on 104 SM, as their recorded wave count predicts;
cuBLASLt targets 108 SMs (smCountTarget=108) and picks identical algorithms on 104-SM green partitions."
