# Table 2 analogue — wave-quantization SM idle % (DERIVED), job ctrl_pdmux-opchar-fff9b1d-20261013062641

Source per cell: no mark = Triton launch log (recorded grid/regs/smem, segment-time weighted); `c` = CUDA-graph census (recorded launch of the single kernel / main GEMM; `c+` = split-K reduce epilogue excluded); `*` = ASSUMED 128x256 GEMM tile, b=1 (not observed). Residency b = occupancy calculator (max smem carveout, upper bound).

| L | SM | M in_proj | M conv1d | M SSD scan | M out_proj | A qkv | A attn | A o_proj | MLP up | MLP down |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 44 | 4.8c | 3.0 | 1.9 | 1.8c | 15.2c | 9.1c | 20.5c | 6.1c | 1.8c |
| 1024 | 104 | 2.2c | 1.5 | 5.2 | 16.9c | 28.2c | 23.1c | 32.7c | 4.6c | 16.9c |
| 1024 | 108 | 5.8c | 5.2 | 4.1 | 0.0c | 30.9c | 25.9c | 35.2c | 8.1c | 0.0c |
| 16384 | 44 | 0.2c | 0.4 | 0.2 | 1.2c | 0.7c | 1.4c | 0.2c | 0.1c | 0.2c |
| 16384 | 104 | 0.1c | 0.3 | 0.6 | 3.7c | 1.5c | 1.5c | 2.1c | 0.4c | 2.1c |
| 16384 | 108 | 0.1c | 0.2 | 0.3 | 3.0c | 2.4c | 1.2c | 1.2c | 0.2c | 1.2c |

