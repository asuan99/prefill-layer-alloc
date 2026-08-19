claims-auditor adversarial audit of the four 2026-08-18 GPU-0 re-analysis docs
(SLO_THRESHOLD_DIAGNOSTIC / SLO_DISCRIMINATION_SURVEY / SPLIT_MOVES_GOODPUT /
PLATEAU_WIDTH).  Run date 2026-08-19.  READ-ONLY: no source artifact modified.

All scorers go through the canonical library
  workspace/engine-port/benchmarks/pdmux_eval/analyze.py
  (load_bench_serving_rounds -> summed duration [gate #7];
   percentile(linear interp); RequestResult.passes -> TTFT<=3000ms AND
   per-request token-ITL p95 <= 60ms [gate #4])
i.e. the same calls g16_analyze.boot_estimands makes.  No new scorer.

  python3 p1_valley.py > p1_valley.json   # g2_0_full A/B per-rep goodput, canonical
  python3 p1_stats.py                     # paired/Welch/permutation CI on the d34 "valley"
  python3 p1_replication.py               # g2_0_hard rA5 phase-A independent replication
  python3 b1_channels.py                  # G16 HI "two channels": identity + paired block CI
  python3 s1_cache.py                     # cache per-file request ITL-p95 (2203 files)
  python3 s1_survey.py                    # full-enumeration survey + 12-file resampling
  python3 g1_decliff.py                   # decliff per-cell + arm-pooled band-mass structure
  python3 d1d2_window_stall.py            # theta sweep under both criteria; stall height/frac
  python3 d2_height_vs_frac.py            # which of height/fraction drives the ITL pass
  python3 p2_plateau_sensitivity.py       # plateau width vs 3/5/7/10pp and vs aggregation
  python3 p2_out_only.py                  # the one contrast where `out` moves alone
  python3 ../audit_g17_rules_2026-08-18/a6_estimand_structure.py   # source of 0.154->0.091

Outputs: *.txt / *.json next to each script.  itl95_cache.pkl is a derived cache.
