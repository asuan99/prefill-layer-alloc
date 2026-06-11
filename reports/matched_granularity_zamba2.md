# Matched-granularity analysis — zamba2

device: a100_sxm4_80gb · chunk: 512 · saturation threshold: 0.03
column check: schema diff 0 ✓

## (A) Matched-granularity SSM vs Attn asymmetry

| seq | bs | ssm sm_sat | attn sm_sat | ssm 14→108× | attn 14→108× | verdict |
|----:|---:|:---------:|:----------:|:-----------:|:------------:|:--------|
| 512 | 1 | 54 | 68 | 1.40 | 3.98 | MAINTAINED (attn later) |
| 512 | 4 | 81 | 94 | 2.35 | 6.07 | MAINTAINED (attn later) |
| 512 | 8 | 108 | 108 | 3.40 | 5.48 | REDUCED (equal sat) |
| 512 | 16 | 108 | 54 | 4.80 | 5.88 | ELIMINATED (ssm later) |
| 512 | 32 | 108 | 81 | 5.67 | 6.90 | ELIMINATED (ssm later) |
| 2048 | 1 | 27 | 68 | 1.12 | 4.20 | MAINTAINED (attn later) |
| 2048 | 4 | 54 | 94 | 2.61 | 6.20 | MAINTAINED (attn later) |
| 2048 | 8 | 94 | 108 | 4.62 | 6.54 | MAINTAINED (attn later) |
| 2048 | 16 | 108 | 108 | 5.84 | 6.86 | REDUCED (equal sat) |
| 2048 | 32 | 108 | 108 | 6.16 | 6.94 | REDUCED (equal sat) |
| 4096 | 1 | 27 | 68 | 1.07 | 4.25 | MAINTAINED (attn later) |
| 4096 | 4 | 54 | 94 | 2.68 | 6.22 | MAINTAINED (attn later) |
| 4096 | 8 | 94 | 108 | 5.00 | 6.55 | MAINTAINED (attn later) |
| 4096 | 16 | 108 | 108 | 6.09 | 6.83 | REDUCED (equal sat) |
| 4096 | 32 | 108 | 108 | 6.25 | 6.93 | REDUCED (equal sat) |
| 8192 | 1 | 14 | 68 | 1.03 | 4.14 | MAINTAINED (attn later) |
| 8192 | 4 | 40 | 94 | 2.72 | 6.18 | MAINTAINED (attn later) |
| 8192 | 8 | 94 | 108 | 5.24 | 6.55 | MAINTAINED (attn later) |
| 8192 | 16 | 108 | 108 | 6.22 | 6.83 | REDUCED (equal sat) |
| 8192 | 32 | 108 | 108 | 6.30 | 6.89 | REDUCED (equal sat) |

**Summary:** MAINTAINED=11  REDUCED=7  ELIMINATED=2 (of 20 comparable cells).

## (B) Granularity effect on Attention (chunked vs full-seq)

| seq | bs | full-seq sm_sat | chunked sm_sat | Δ (chunked advances by) |
|----:|---:|:--------------:|:--------------:|:------------------------|
| 512 | 1 | 68 | 68 | +0 SM (same) |
| 512 | 4 | 108 | 94 | +14 SM (earlier) |
| 512 | 16 | 81 | 54 | +27 SM (earlier) |
| 512 | 32 | 108 | 81 | +27 SM (earlier) |
| 2048 | 1 | 108 | 68 | +40 SM (earlier) |
| 2048 | 4 | 108 | 94 | +14 SM (earlier) |
| 2048 | 16 | 108 | 108 | +0 SM (same) |
| 2048 | 32 | 108 | 108 | +0 SM (same) |
| 4096 | 1 | 108 | 68 | +40 SM (earlier) |
| 4096 | 4 | 108 | 94 | +14 SM (earlier) |
| 4096 | 16 | 108 | 108 | +0 SM (same) |
| 4096 | 32 | 108 | 108 | +0 SM (same) |
| 8192 | 1 | 108 | 68 | +40 SM (earlier) |
| 8192 | 4 | 108 | 94 | +14 SM (earlier) |
| 8192 | 16 | 108 | 108 | +0 SM (same) |
| 8192 | 32 | 108 | 108 | +0 SM (same) |
