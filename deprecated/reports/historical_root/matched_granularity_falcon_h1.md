# Matched-granularity analysis — falcon_h1

device: a100_sxm4_80gb · chunk: 512 · saturation threshold: 0.03
column check: schema diff 0 ✓

## (A) Matched-granularity SSM vs Attn asymmetry

| seq | bs | ssm sm_sat | attn sm_sat | ssm 14→108× | attn 14→108× | verdict |
|----:|---:|:---------:|:----------:|:-----------:|:------------:|:--------|
| 512 | 1 | 27 | 40 | 1.14 | 2.69 | MAINTAINED (attn later) |
| 512 | 4 | 54 | 40 | 1.55 | 5.84 | ELIMINATED (ssm later) |
| 512 | 8 | 68 | 40 | 2.37 | 6.68 | ELIMINATED (ssm later) |
| 512 | 16 | 94 | 68 | 3.77 | 5.91 | ELIMINATED (ssm later) |
| 512 | 32 | 108 | 108 | 4.26 | 5.29 | REDUCED (equal sat) |
| 2048 | 1 | 14 | 54 | 1.04 | 2.64 | MAINTAINED (attn later) |
| 2048 | 4 | 27 | 81 | 1.43 | 4.78 | MAINTAINED (attn later) |
| 2048 | 8 | 54 | 108 | 2.58 | 5.18 | MAINTAINED (attn later) |
| 2048 | 16 | 108 | 81 | 4.69 | 5.42 | ELIMINATED (ssm later) |
| 2048 | 32 | 108 | 108 | 4.76 | 6.70 | REDUCED (equal sat) |
| 4096 | 1 | 14 | 68 | 1.01 | 3.08 | MAINTAINED (attn later) |
| 4096 | 4 | 27 | 108 | 1.41 | 5.99 | MAINTAINED (attn later) |
| 4096 | 8 | 54 | 108 | 2.65 | 6.41 | MAINTAINED (attn later) |
| 4096 | 16 | 108 | 108 | 4.91 | 6.66 | REDUCED (equal sat) |
| 4096 | 32 | 108 | 108 | 4.85 | 6.83 | REDUCED (equal sat) |
| 8192 | 1 | 14 | 68 | 1.01 | 3.81 | MAINTAINED (attn later) |
| 8192 | 4 | 27 | 108 | 1.41 | 6.30 | MAINTAINED (attn later) |
| 8192 | 8 | 54 | 108 | 2.69 | 6.65 | MAINTAINED (attn later) |
| 8192 | 16 | 108 | 108 | 5.02 | 6.86 | REDUCED (equal sat) |
| 8192 | 32 | 108 | 108 | 4.90 | 6.99 | REDUCED (equal sat) |

**Summary:** MAINTAINED=10  REDUCED=6  ELIMINATED=4 (of 20 comparable cells).

## (B) Granularity effect on Attention (chunked vs full-seq)

| seq | bs | full-seq sm_sat | chunked sm_sat | Δ (chunked advances by) |
|----:|---:|:--------------:|:--------------:|:------------------------|
| 512 | 1 | 68 | 40 | +28 SM (earlier) |
| 512 | 4 | 108 | 40 | +68 SM (earlier) |
| 512 | 16 | 108 | 68 | +40 SM (earlier) |
| 512 | 32 | 108 | 108 | +0 SM (same) |
| 2048 | 1 | 108 | 54 | +54 SM (earlier) |
| 2048 | 4 | 108 | 81 | +27 SM (earlier) |
| 2048 | 16 | 108 | 81 | +27 SM (earlier) |
| 2048 | 32 | 108 | 108 | +0 SM (same) |
| 4096 | 1 | 108 | 68 | +40 SM (earlier) |
| 4096 | 4 | 108 | 108 | +0 SM (same) |
| 4096 | 16 | 108 | 108 | +0 SM (same) |
| 4096 | 32 | 108 | 108 | +0 SM (same) |
| 8192 | 1 | 108 | 68 | +40 SM (earlier) |
| 8192 | 4 | 108 | 108 | +0 SM (same) |
| 8192 | 16 | 108 | 108 | +0 SM (same) |
| 8192 | 32 | 108 | 108 | +0 SM (same) |
