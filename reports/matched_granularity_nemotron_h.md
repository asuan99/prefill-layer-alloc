# Matched-granularity analysis — nemotron_h

device: a100_sxm4_80gb · chunk: 512 · saturation threshold: 0.03
column check: schema diff 0 ✓

## (A) Matched-granularity SSM vs Attn asymmetry

| seq | bs | ssm sm_sat | attn sm_sat | ssm 14→108× | attn 14→108× | verdict |
|----:|---:|:---------:|:----------:|:-----------:|:------------:|:--------|
| 512 | 1 | 54 | 81 | 1.42 | 5.44 | MAINTAINED (attn later) |
| 512 | 4 | 81 | 68 | 2.84 | 6.96 | ELIMINATED (ssm later) |
| 512 | 8 | 108 | 94 | 4.27 | 6.37 | ELIMINATED (ssm later) |
| 512 | 16 | 108 | 108 | 5.67 | 5.67 | REDUCED (equal sat) |
| 512 | 32 | 108 | 54 | 5.17 | 6.08 | ELIMINATED (ssm later) |
| 2048 | 1 | 27 | 68 | 1.12 | 5.27 | MAINTAINED (attn later) |
| 2048 | 4 | 68 | 81 | 3.33 | 6.12 | MAINTAINED (attn later) |
| 2048 | 8 | 108 | 81 | 5.06 | 6.55 | ELIMINATED (ssm later) |
| 2048 | 16 | 108 | 108 | 5.36 | 6.96 | REDUCED (equal sat) |
| 2048 | 32 | 108 | 108 | 5.46 | 7.23 | REDUCED (equal sat) |
| 4096 | 1 | 27 | 108 | 1.06 | 5.61 | MAINTAINED (attn later) |
| 4096 | 4 | 68 | 108 | 3.45 | 6.37 | MAINTAINED (attn later) |
| 4096 | 8 | 108 | 108 | 5.24 | 6.75 | REDUCED (equal sat) |
| 4096 | 16 | 108 | 108 | 5.46 | 7.04 | REDUCED (equal sat) |
| 4096 | 32 | 108 | 108 | 5.51 | 7.25 | REDUCED (equal sat) |
| 8192 | 1 | 14 | 108 | 1.03 | 5.92 | MAINTAINED (attn later) |
| 8192 | 4 | 68 | 108 | 3.52 | 6.58 | MAINTAINED (attn later) |
| 8192 | 8 | 108 | 108 | 5.33 | 6.90 | REDUCED (equal sat) |
| 8192 | 16 | 108 | 108 | 5.51 | 7.18 | REDUCED (equal sat) |
| 8192 | 32 | 108 | 108 | 5.53 | 7.34 | REDUCED (equal sat) |

**Summary:** MAINTAINED=7  REDUCED=9  ELIMINATED=4 (of 20 comparable cells).

## (B) Granularity effect on Attention (chunked vs full-seq)

_full-seq attn CSV (results/stage1/attn_scaling_*) not found — skipped._
