> **REFUTED (claims-auditor 2026-10-13): the GEMM cells (`*`) use an ASSUMED tile and are contradicted by the recorded graph census (e.g. out_proj @108 L=1024: 33.3% assumed vs 0.0% recorded). Do not cite. Use `ctrl_pdmux-opchar-fff9b1d-20261013062641/ANALYSIS/table2_idle.md`. See `../../AUDIT_opchar_2026-10-13.md`.**

# Table 2 analogue — wave-quantization SM idle % (DERIVED), job full_pdmux-opchar-98f3a8c-20261012141951

Triton ops (conv1d, SSD scan): recorded launch grid/regs/smem, segment-time weighted. GEMMs: ASSUMED 128x256 tiles, b=1 (not observed). Attention/norm/elementwise not covered (no CUPTI on this image).

| L | SM | M in_proj | M conv1d | M SSD scan | M out_proj | A qkv | A o_proj | MLP up | MLP down |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 16 | 1.1* | 0.0 | 0.5 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0* |
| 1024 | 44 | 4.8* | 3.0 | 2.0 | 18.2* | 15.2* | 18.2* | 6.1* | 18.2* |
| 1024 | 108 | 5.8* | 5.2 | 3.9 | 33.3* | 30.9* | 33.3* | 8.1* | 33.3* |
| 2048 | 16 | 0.0* | 0.0 | 0.1 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0* |
| 2048 | 44 | 1.9* | 3.0 | 1.1 | 6.5* | 7.4* | 6.5* | 2.0* | 6.5* |
| 2048 | 108 | 5.8* | 5.2 | 2.5 | 11.1* | 17.0* | 11.1* | 8.1* | 11.1* |
| 4096 | 16 | 0.0* | 0.0 | 0.1 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0* |
| 4096 | 44 | 0.4* | 1.0 | 0.5 | 6.5* | 3.0* | 6.5* | 2.0* | 6.5* |
| 4096 | 108 | 2.3* | 0.2 | 1.3 | 11.1* | 7.8* | 11.1* | 3.3* | 11.1* |
| 16384 | 16 | 0.0* | 0.0 | 0.0 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0* |
| 16384 | 44 | 0.0* | 0.4 | 0.2 | 1.2* | 0.7* | 1.2* | 0.4* | 1.2* |
| 16384 | 108 | 0.5* | 0.2 | 0.3 | 3.0* | 2.4* | 3.0* | 0.7* | 3.0* |

`*` = assumed GEMM tile.
