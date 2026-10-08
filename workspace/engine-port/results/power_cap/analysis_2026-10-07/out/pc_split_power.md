# pc_split_power — power/clock vs SM split (descriptive, 0 GPU-h)

KISTI busy mask: utilization.gpu >= 90; near-cap = power >= 380 W (0.95 x 400 W, A100-SXM4-80GB on both substrates). KISTI files have NO timestamps (5 s cadence, ~17-20 rows per file); realised partition is not verified by this script.

## (a) KISTI sweeps — busy-sample power by split
| campaign | mode | arm | ctx | split | n_files | n_busy | busy_mean_W | busy_max_W | busy_min_MHz | frac_near_cap | frac_MHz_lt_1400 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0_deconfound | deconf(prefill pinned) | H | 4096 | P16/D16 | 8 | 132 | 291 | 424 | 1380 | 0.48 | 0.36 |
| s0_deconfound | deconf(prefill pinned) | H | 4096 | P16/D24 | 8 | 132 | 299 | 417 | 1380 | 0.44 | 0.43 |
| s0_deconfound | deconf(prefill pinned) | H | 4096 | P16/D44 | 8 | 132 | 317 | 417 | 1380 | 0.42 | 0.41 |
| s0_deconfound | deconf(prefill pinned) | H | 4096 | P16/D92 | 8 | 132 | 328 | 414 | 1380 | 0.39 | 0.39 |
| s0_deconfound | deconf(prefill pinned) | M | 4096 | P16/D16 | 8 | 132 | 294 | 407 | 1395 | 0.32 | 0.02 |
| s0_deconfound | deconf(prefill pinned) | M | 4096 | P16/D24 | 8 | 132 | 311 | 407 | 1395 | 0.46 | 0.12 |
| s0_deconfound | deconf(prefill pinned) | M | 4096 | P16/D44 | 4 | 68 | 221 | 286 | 1410 | 0.00 | 0.00 |
| s0_deconfound | deconf(prefill pinned) | M | 4096 | P16/D92 | 8 | 132 | 300 | 408 | 1395 | 0.40 | 0.17 |
| s0_deconfound | deconf(prefill pinned) | T | 4096 | P16/D16 | 8 | 115 | 304 | 425 | 1350 | 0.34 | 0.41 |
| s0_deconfound | deconf(prefill pinned) | T | 4096 | P16/D24 | 8 | 120 | 318 | 419 | 1350 | 0.39 | 0.43 |
| s0_deconfound | deconf(prefill pinned) | T | 4096 | P16/D44 | 8 | 112 | 336 | 424 | 1335 | 0.35 | 0.39 |
| s0_deconfound | deconf(prefill pinned) | T | 4096 | P16/D92 | 8 | 112 | 338 | 421 | 1335 | 0.33 | 0.39 |
| s8_frontier | e1 frontier | T8 | - | D16 | 2 | 27 | 294 | 416 | 1320 | 0.07 | 0.19 |
| s8_frontier | e1 frontier | T8 | - | D92 | 2 | 28 | 311 | 356 | 1410 | 0.00 | 0.00 |
| s8_scaleup | c2r(re-measure, prefill pinned 16) | Ha8 | 1024 | P16/D16 | 102 | 1724 | 211 | 314 | 1410 | 0.00 | 0.00 |
| s8_scaleup | c2r(re-measure, prefill pinned 16) | Ha8 | 1024 | P16/D92 | 102 | 1730 | 252 | 387 | 1290 | 0.00 | 0.00 |
| s8_scaleup | c2r(re-measure, prefill pinned 16) | M8 | 1024 | P16/D16 | 56 | 931 | 218 | 418 | 1335 | 0.00 | 0.00 |
| s8_scaleup | c2r(re-measure, prefill pinned 16) | M8 | 1024 | P16/D92 | 54 | 913 | 254 | 391 | 1350 | 0.00 | 0.01 |
| s8_scaleup | coupled | Ha8 | 4096 | P16/D92 | 4 | 68 | 223 | 336 | 1410 | 0.00 | 0.00 |
| s8_scaleup | coupled | Ha8 | 4096 | P64/D44 | 4 | 68 | 351 | 446 | 1350 | 0.26 | 0.41 |
| s8_scaleup | coupled | Ha8 | 4096 | P92/D16 | 4 | 67 | 296 | 480 | 1305 | 0.21 | 0.21 |
| s8_scaleup | coupled | Ha8 | 4096 | np(108, unpartitioned) | 4 | 68 | 353 | 476 | 1305 | 0.34 | 0.34 |
| s8_scaleup | coupled | Hs8 | 4096 | P16/D92 | 4 | 68 | 286 | 355 | 1410 | 0.00 | 0.00 |
| s8_scaleup | coupled | Hs8 | 4096 | P64/D44 | 4 | 68 | 378 | 447 | 1275 | 0.53 | 0.94 |
| s8_scaleup | coupled | Hs8 | 4096 | P92/D16 | 4 | 68 | 366 | 448 | 1275 | 0.60 | 0.93 |
| s8_scaleup | coupled | Hs8 | 4096 | np(108, unpartitioned) | 4 | 65 | 379 | 449 | 1320 | 0.57 | 0.92 |
| s8_scaleup | coupled | M8 | 4096 | P16/D92 | 4 | 68 | 222 | 340 | 1410 | 0.00 | 0.00 |
| s8_scaleup | coupled | M8 | 4096 | P64/D44 | 4 | 68 | 377 | 476 | 1320 | 0.49 | 0.78 |
| s8_scaleup | coupled | M8 | 4096 | P92/D16 | 4 | 68 | 363 | 457 | 1290 | 0.54 | 0.63 |
| s8_scaleup | coupled | M8 | 4096 | np(108, unpartitioned) | 4 | 68 | 366 | 443 | 1320 | 0.34 | 0.41 |
| s8_scaleup | coupled | T8 | 4096 | P16/D92 | 4 | 68 | 312 | 364 | 1410 | 0.00 | 0.00 |
| s8_scaleup | coupled | T8 | 4096 | P64/D44 | 4 | 68 | 386 | 437 | 1245 | 0.71 | 0.94 |
| s8_scaleup | coupled | T8 | 4096 | P92/D16 | 4 | 68 | 361 | 447 | 1230 | 0.56 | 0.91 |
| s8_scaleup | coupled | T8 | 4096 | np(108, unpartitioned) | 4 | 64 | 385 | 445 | 1290 | 0.61 | 0.98 |
| s8_scaleup | deconf | Ha8 | 1024 | P16/D16 | 8 | 136 | 213 | 298 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 1024 | P16/D24 | 8 | 136 | 223 | 298 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 1024 | P16/D44 | 8 | 136 | 235 | 331 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 1024 | P16/D92 | 8 | 136 | 230 | 351 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 1024 | np(108, unpartitioned) | 8 | 132 | 332 | 454 | 1320 | 0.44 | 0.55 |
| s8_scaleup | deconf | Ha8 | 4096 | P16/D16 | 4 | 68 | 191 | 292 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 4096 | P16/D24 | 4 | 68 | 203 | 276 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 4096 | P16/D44 | 4 | 68 | 218 | 318 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 4096 | P16/D92 | 4 | 68 | 231 | 344 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Ha8 | 4096 | np(108, unpartitioned) | 4 | 68 | 359 | 488 | 1305 | 0.51 | 0.50 |
| s8_scaleup | deconf | Hs8 | 1024 | P16/D16 | 12 | 203 | 256 | 341 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 1024 | P16/D24 | 8 | 135 | 264 | 316 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 1024 | P16/D44 | 8 | 136 | 288 | 345 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 1024 | P16/D54 | 6 | 88 | 294 | 340 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 1024 | P16/D92 | 8 | 136 | 284 | 374 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 1024 | np(108, unpartitioned) | 8 | 130 | 357 | 444 | 1320 | 0.42 | 0.58 |
| s8_scaleup | deconf | Hs8 | 4096 | P16/D16 | 4 | 68 | 237 | 256 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 4096 | P16/D24 | 4 | 68 | 271 | 297 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 4096 | P16/D44 | 4 | 68 | 316 | 358 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 4096 | P16/D92 | 4 | 68 | 291 | 364 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | Hs8 | 4096 | np(108, unpartitioned) | 4 | 67 | 378 | 454 | 1290 | 0.49 | 0.93 |
| s8_scaleup | deconf | M8 | 1024 | P16/D16 | 8 | 136 | 216 | 312 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 1024 | P16/D24 | 8 | 136 | 225 | 311 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 1024 | P16/D44 | 8 | 135 | 238 | 422 | 1365 | 0.01 | 0.02 |
| s8_scaleup | deconf | M8 | 1024 | P16/D92 | 8 | 132 | 249 | 346 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 1024 | np(108, unpartitioned) | 8 | 135 | 353 | 416 | 1380 | 0.47 | 0.33 |
| s8_scaleup | deconf | M8 | 4096 | P16/D16 | 4 | 68 | 202 | 246 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 4096 | P16/D24 | 4 | 68 | 214 | 270 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 4096 | P16/D44 | 4 | 68 | 244 | 313 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 4096 | P16/D92 | 4 | 68 | 222 | 338 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | M8 | 4096 | np(108, unpartitioned) | 4 | 68 | 374 | 460 | 1335 | 0.40 | 0.46 |
| s8_scaleup | deconf | T8 | 1024 | P16/D16 | 12 | 200 | 272 | 329 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 1024 | P16/D24 | 8 | 132 | 286 | 318 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 1024 | P16/D44 | 8 | 133 | 327 | 380 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 1024 | P16/D54 | 6 | 92 | 333 | 364 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 1024 | P16/D92 | 8 | 136 | 323 | 380 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 1024 | np(108, unpartitioned) | 8 | 133 | 353 | 442 | 1260 | 0.44 | 0.55 |
| s8_scaleup | deconf | T8 | 4096 | P16/D16 | 4 | 68 | 257 | 278 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 4096 | P16/D24 | 4 | 68 | 295 | 326 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 4096 | P16/D44 | 4 | 68 | 360 | 390 | 1410 | 0.15 | 0.00 |
| s8_scaleup | deconf | T8 | 4096 | P16/D92 | 4 | 68 | 320 | 372 | 1410 | 0.00 | 0.00 |
| s8_scaleup | deconf | T8 | 4096 | np(108, unpartitioned) | 4 | 64 | 388 | 448 | 1260 | 0.67 | 1.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Ha8 | - | P16/D16 | 4 | 68 | 182 | 261 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Ha8 | - | P24/D16 | 4 | 67 | 215 | 270 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Ha8 | - | P44/D16 | 4 | 64 | 281 | 329 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Ha8 | - | P92/D16 | 4 | 68 | 354 | 432 | 1350 | 0.65 | 0.87 |
| s8p_prefill | prefill-axis(decode pinned 16) | Hs8 | - | P16/D16 | 4 | 67 | 230 | 295 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Hs8 | - | P24/D16 | 4 | 68 | 264 | 307 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Hs8 | - | P44/D16 | 4 | 68 | 326 | 387 | 1410 | 0.01 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | Hs8 | - | P92/D16 | 4 | 66 | 388 | 443 | 1305 | 0.79 | 0.97 |
| s8p_prefill | prefill-axis(decode pinned 16) | M8 | - | P16/D16 | 4 | 68 | 194 | 280 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | M8 | - | P24/D16 | 4 | 68 | 230 | 282 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | M8 | - | P44/D16 | 4 | 63 | 279 | 323 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | M8 | - | P92/D16 | 4 | 68 | 367 | 424 | 1365 | 0.72 | 0.79 |
| s8p_prefill | prefill-axis(decode pinned 16) | T8 | - | P16/D16 | 5 | 71 | 253 | 303 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | T8 | - | P24/D16 | 4 | 64 | 290 | 303 | 1410 | 0.00 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | T8 | - | P44/D16 | 4 | 64 | 353 | 392 | 1410 | 0.12 | 0.00 |
| s8p_prefill | prefill-axis(decode pinned 16) | T8 | - | P92/D16 | 5 | 71 | 392 | 455 | 1275 | 0.82 | 0.97 |

unmatched file names: 0

arm legend: M=pure-Mamba2 (s0 negative control); H=Zamba2 hybrid (s0 target); T=Qwen2.5-3B attn (s0 positive control); M8=Mamba-Codestral-7B pure-Mamba2; Ha8=Zamba2-7B additive hybrid; Hs8=Nemotron-H-8B substitutive hybrid; T8=Qwen2.5-7B attention

## (b) VESSL lambda0 cells — bench-window power vs realised D44 time share
| cell | shape | offered | achieved | e_time_D44_pct | idx0_pct | n | mean_W | p50_W | p95_W | cap_pct | sm_MHz | ttft_p95_s | itl_p95_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_r0 | A | 1.1 | 1.072 | 3.2 | 0.0 | 6144 | 330 | 329 | 352 | 1.4 | 1410 | 0.104 | 0.0193 |
| a_r1 | A | 1.8 | 1.754 | 6.1 | 0.0 | 7172 | 343 | 342 | 380 | 10.2 | 1408 | 0.142 | 0.0311 |
| a_r2 | A | 3.0 | 2.799 | 13.1 | 0.0 | 1426 | 354 | 359 | 406 | 28.8 | 1404 | 2.8 | 0.0484 |
| a_r3 | A | 4.9 | 2.974 | 12.4 | 0.0 | 1339 | 355 | 361 | 404 | 30.8 | 1404 | 41.0 | 0.0487 |
| a_r4 | A | 8.0 | 2.994 | 13.9 | 0.2 | 1334 | 353 | 359 | 400 | 27.2 | 1404 | 66.9 | 0.0493 |
| a_r4_s2 | A | 8.0 | 3.010 | 12.9 | 0.4 | 1327 | 354 | 359 | 401 | 25.8 | 1404 | 64.8 | 0.0489 |
| a_r4_s3 | A | 8.0 | 2.997 | 14.2 | 0.0 | 1331 | 352 | 356 | 402 | 26.4 | 1404 | 64.7 | 0.0489 |
| a_r4_s4 | A | 8.0 | 3.007 | 12.9 | 0.0 | 1328 | 356 | 359 | 403 | 27.5 | 1404 | 64.4 | 0.0487 |
| b_r0 | B | 0.45 | 0.437 | 70.1 | 0.3 | 1827 | 271 | 306 | 412 | 41.5 | 1392 | 6.9 | 0.0223 |
| b_r1 | B | 0.62 | 0.602 | 86.1 | 0.0 | 1824 | 326 | 345 | 418 | 52.6 | 1384 | 9.3 | 0.0233 |
| b_r2 | B | 0.85 | 0.689 | 96.4 | 0.0 | 2028 | 347 | 348 | 419 | 49.0 | 1380 | 34.4 | 0.0237 |
| b_r3 | B | 1.1 | 0.690 | 97.4 | 0.0 | 2892 | 345 | 346 | 419 | 49.4 | 1380 | 114.2 | 0.0238 |
| b_r3_s2 | B | 1.1 | 0.690 | 97.4 | 0.7 | 2893 | 346 | 347 | 420 | 49.2 | 1379 | 112.5 | 0.0238 |
| b_r3_s3 | B | 1.1 | 0.691 | 97.4 | 0.0 | 2890 | 345 | 346 | 420 | 49.0 | 1380 | 115.1 | 0.0237 |
| b_r3_s4 | B | 1.1 | 0.691 | 97.5 | 0.0 | 2889 | 344 | 345 | 420 | 49.2 | 1380 | 107.4 | 0.0239 |

## (c) VESSL E2 r2 — sticky OFF vs ON, paired by (cell, seed)
| cell | seed | OFF_mean_W | ON_mean_W | dW_ON_minus_OFF | OFF_cap_pct | ON_cap_pct | OFF_sm_MHz | ON_sm_MHz | OFF_e_time | ON_e_time |
|---|---|---|---|---|---|---|---|---|---|---|
| a_r2 | 251 | 323 | 260 | -63 | 22.7 | 12.6 | 1406 | 1408 | - | - |
| a_r2 | 2630 | 327 | 271 | -55 | 24.0 | 21.3 | 1406 | 1406 | - | - |
| a_r2 | 4162 | 331 | 263 | -68 | 26.7 | 16.1 | 1404 | 1407 | - | - |
| a_r2 | 4386 | 325 | 262 | -63 | 25.1 | 13.8 | 1405 | 1408 | - | - |
| a_r4 | 251 | 328 | 267 | -61 | 24.7 | 17.4 | 1404 | 1406 | - | - |
| a_r4 | 2630 | 331 | 268 | -63 | 25.1 | 16.4 | 1404 | 1406 | - | - |
| a_r4 | 4162 | 328 | 268 | -59 | 23.9 | 16.0 | 1404 | 1406 | - | - |
| a_r4 | 4386 | 335 | 265 | -70 | 25.8 | 17.5 | 1402 | 1407 | - | - |
| b_r3 | 4162 | 332 | 332 | +0 | 46.4 | 46.2 | 1382 | 1382 | - | - |
| b_r3 | 4386 | 335 | 332 | -4 | 46.4 | 46.4 | 1380 | 1381 | - | - |

Forbidden readings: arm-delta attribution to power cap (LVR-2/VC-2/N3C-16); 'SM split partitions power' as a mechanism claim (no per-partition power attribution exists); any policy statement.
