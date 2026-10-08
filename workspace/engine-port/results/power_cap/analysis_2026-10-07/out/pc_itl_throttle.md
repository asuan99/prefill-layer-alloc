# pc_itl_throttle — decode step time under SW power cap, bs x co-residency strata (EXPLORATORY)

alignment: srv.log 'Decode batch' lines x iter_seq=40k, per unit; units aligned 37; O_hat range 1789447353.0188532..1789447353.0727606 (drift 0.054 s over the Job); max half-width 52 ms; marks cross-check (T_BS - mono - O) in [-0.6137776374816895, 0.3513607978820801] (expect (-1, 1)); base edge guard 0.15 s (+ per-unit half-width); skipped units: {'br_b_r3_4386': 'no decode_iteration events (PHASE_EVENTS off)', 'br_b_r3_4162': 'no decode_iteration events (PHASE_EVENTS off)', 'off_r030_3704': 'no decode_iteration events (PHASE_EVENTS off)', 'off_r030_2084': 'no decode_iteration events (PHASE_EVENTS off)', 'off_r030_4861': 'no decode_iteration events (PHASE_EVENTS off)', 'off_r030_4589': 'no decode_iteration events (PHASE_EVENTS off)'}

joined iterations 55793 over 37 units; 0x4-active fraction 0.611; pooled ratio step(thr1/thr0) = 1.0010

## all units — strata
| bs_bin | co_resident | n_thr0 | n_thr1 | step_med_thr0 | step_med_thr1 | ratio_step | sm_mean_thr0 | sm_mean_thr1 |
|---|---|---|---|---|---|---|---|---|
| bs1-1 | False | 20212 | 1035 | 12.10 | 12.12 | 1.002 | 1410 | 1371 |
| bs1-1 | True | 136 | 18734 | 19.20 | 18.79 | 0.978 | 1410 | 1361 |
| bs2-4 | False | 1257 | 0 | 12.47 | - | - | 1410 | - |
| bs2-4 | True | 123 | 14296 | 18.78 | 19.16 | 1.021 | 1410 | 1350 |

## reference: co-resident vs not at thr=0 (known SM channel)
| bs_bin | n_co0 | n_co1 | ratio_step |
|---|---|---|---|
| bs2-4 | 1257 | 123 | 1.506 |
| bs1-1 | 20212 | 136 | 1.587 |

## SM-clock-conditioned step time
| bs_bin | co_resident | sm_bin | n | step_med_ms |
|---|---|---|---|---|
| bs1-1 | False | sm1300-1399 | 1035 | 12.12 |
| bs1-1 | False | sm1400-+ | 20212 | 12.10 |
| bs1-1 | True | sm1300-1399 | 18714 | 18.79 |
| bs1-1 | True | sm1400-+ | 136 | 19.20 |
| bs2-4 | False | sm1400-+ | 1257 | 12.47 |
| bs2-4 | True | sm0-1299 | 218 | 19.21 |
| bs2-4 | True | sm1300-1399 | 14078 | 19.16 |
| bs2-4 | True | sm1400-+ | 123 | 18.78 |

## by offered rate
| rate | n | thr_frac | pooled_ratio_step |
|---|---|---|---|
| 0.10 | 3352 | 0.268 | 1.002 |
| 0.15 | 3468 | 0.311 | 1.001 |
| 0.22 | 3752 | 0.329 | 1.002 |
| 0.30 | 5255 | 0.426 | 1.002 |
| 0.38 | 6598 | 0.505 | 1.002 |
| 0.45 | 7703 | 0.591 | 1.002 |
| 0.53 | 6179 | 0.683 | 1.002 |
| 0.61 | 6565 | 0.767 | 1.002 |
| 0.69 | 6292 | 0.902 | 1.002 |
| 0.77 | 5985 | 0.953 | 1.021 |

## per unit
| unit | rate | O_hw_ms | n_joined | dropped_edge | thr_frac | ratio_step |
|---|---|---|---|---|---|---|
| floor_head | - | 4 | 644 | 395 | 0.149 | 1.001 |
| g_r030_3704 | 0.3 | 32 | 1199 | 533 | 0.385 | 1.002 |
| g_r030_2084 | 0.3 | 8 | 1380 | 478 | 0.378 | 1.002 |
| g_r030_4861 | 0.3 | 13 | 1358 | 437 | 0.454 | 1.002 |
| g_r030_4589 | 0.3 | 12 | 1318 | 477 | 0.484 | 1.001 |
| g_r077_3704 | 0.77 | 20 | 2016 | 598 | 0.936 | - |
| g_r069_3704 | 0.69 | 1 | 2302 | 627 | 0.861 | - |
| g_r061_3704 | 0.61 | 7 | 2204 | 662 | 0.749 | - |
| g_r053_3704 | 0.53 | 21 | 1975 | 639 | 0.694 | - |
| g_r045_3704 | 0.45 | 3 | 1870 | 555 | 0.528 | 1.002 |
| g_r038_3704 | 0.38 | 10 | 1644 | 529 | 0.448 | - |
| g_r022_3704 | 0.22 | 23 | 909 | 382 | 0.352 | - |
| g_r015_3704 | 0.15 | 45 | 844 | 447 | 0.346 | - |
| g_r010_3704 | 0.1 | 39 | 787 | 504 | 0.310 | 1.001 |
| g_r010_2084 | 0.1 | 52 | 774 | 517 | 0.251 | - |
| g_r015_2084 | 0.15 | 21 | 875 | 416 | 0.280 | - |
| g_r022_2084 | 0.22 | 26 | 992 | 425 | 0.320 | - |
| g_r038_2084 | 0.38 | 28 | 1617 | 619 | 0.491 | 1.002 |
| g_r045_2084 | 0.45 | 0 | 1956 | 532 | 0.628 | 1.002 |
| g_r053_2084 | 0.53 | 11 | 2188 | 615 | 0.682 | - |
| g_r061_2084 | 0.61 | 8 | 2093 | 584 | 0.785 | - |
| g_r069_2084 | 0.69 | 8 | 2020 | 657 | 0.895 | - |
| g_r077_2084 | 0.77 | 19 | 2011 | 666 | 0.962 | - |
| g_r077_4861 | 0.77 | 17 | 1958 | 593 | 0.961 | - |
| g_r069_4861 | 0.69 | 6 | 1970 | 581 | 0.957 | - |
| g_r061_4861 | 0.61 | 23 | 2268 | 724 | 0.769 | - |
| g_r053_4861 | 0.53 | 32 | 2016 | 661 | 0.672 | - |
| g_r045_4861 | 0.45 | 6 | 1903 | 648 | 0.611 | 1.002 |
| g_r038_4861 | 0.38 | 14 | 1634 | 539 | 0.583 | 1.002 |
| g_r022_4861 | 0.22 | 10 | 1010 | 407 | 0.314 | 1.002 |
| g_r015_4861 | 0.15 | 5 | 897 | 394 | 0.274 | - |
| g_r010_4861 | 0.1 | 5 | 921 | 370 | 0.282 | 1.002 |
| g_r010_4589 | 0.1 | 14 | 870 | 421 | 0.231 | 1.002 |
| g_r015_4589 | 0.15 | 43 | 852 | 439 | 0.349 | - |
| g_r022_4589 | 0.22 | 49 | 841 | 513 | 0.334 | - |
| g_r038_4589 | 0.38 | 25 | 1703 | 596 | 0.499 | - |
| g_r045_4589 | 0.45 | 12 | 1974 | 640 | 0.593 | 1.001 |

Caveats: observational (0x4 is endogenous to load and to prefill activity); strata control bs and prefill_in_flight only; NVML reason bits are instantaneous but power is 1 s-averaged; not a decision input; not citable in canon.
