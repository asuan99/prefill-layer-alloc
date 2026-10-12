# Table 2 analogue — wave-quantization SM idle % (DERIVED), job probe_pdmux-opchar-dd405fe-20261012141151

Triton ops (conv1d, SSD scan): recorded launch grid/regs/smem, segment-time weighted. GEMMs: ASSUMED 128x256 tiles, b=1 (not observed). Attention/norm/elementwise not covered (no CUPTI on this image).

| L | SM | M in_proj | M conv1d | M SSD scan | M out_proj | A qkv | A o_proj | MLP up | MLP down |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 44 | 4.8* | — | — | 18.2* | 15.2* | 18.2* | 6.1* | 18.2* |
| 1024 | 108 | 5.8* | — | — | 33.3* | 30.9* | 33.3* | 8.1* | 33.3* |

`*` = assumed GEMM tile.
