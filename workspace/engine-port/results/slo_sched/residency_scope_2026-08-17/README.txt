residency_scope_2026-08-17/ -- reproduction path for the addendum A-2
conditional-label SCOPE SIZE measurement (2026-08-17, result-analyst).

SCOPE.  This directory measures ONE thing: how large is the window over which
the G16 arm label ("arm = decode gets D SM") is actually in force.  It does NOT
recompute, re-score or re-decide any G16 decision quantity.  M_itl, gap_upper,
the donor, Delta and Delta_SLO are read from G16_RESULTS_2026-08-17.json as
GIVEN CONSTANTS and are not modified.  This is a scope measurement, not a
performance comparison.

NO NEW ESTIMAND.  Every number produced here is one of the two canonical
residency estimands already in the repository, evaluated on a stated
(denominator, window) pair:
  E-CNT = results/s8_scaleup/realized_pin_check.py  (addendum A-2's own tool)
          sha256 9d46542ffb8f161a987e6f2b8e24d97641364fef4138c30339c40a92c6768bb9
  E-TIM = benchmarks/pdmux_eval/analyze.py :: controller_summary  (H3'-b)
          sha256 1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2

FILES
  residency_scope.py            single pass over all 34 g16 telemetry files
                                (28 campaign + 6 smoke); emits the 2x2
                                (count|time) x (all|decode-active) grid plus the
                                phase-window restriction.  Runs the two canonical
                                tools as controls.   ~4.5 min.
  residency_scope_2026-08-17.json   per-boot output + control ledger.
  residency_tables.py           tables, identity checks (gate #40), placement
                                analysis, task-3 interpretation.  Reads only the
                                JSON above + the campaign report (read-only).
  tables_2026-08-17.txt         captured output of residency_tables.py.

CONTROLS (all executed, see the "controls" key of the JSON)
  PC-1   realized_pin_check.py run as a SUBPROCESS on all 34 files; its printed
         REALIZED_hist / CO_RESIDENT_frac / frac compared to the single-pass
         recomputation.  34/34 match.
  PC-2   controller_summary() imported and run on the loaded events; compared to
         the harness-written *_controller_summary.json produced on the compute
         node at run time.  31 match / 3 artifact_missing.  The 3 missing are
         smoke job 884292, whose harness lost H3'-b (g16_grid.sbatch:553) --
         that is a MISSING ARTIFACT, not a value disagreement (lesson item 21).
  PC-2b  same, compared to residency_fraction_by_boot in the campaign report.
         28/28 match, and the LO and HI views of that field are identical
         (residency is computed per telemetry file, which spans both phases).
  NC-1   the PC-1 comparator re-run against the WRONG arm SM (D+10) must FAIL.
         34/34 correctly detected.
  NC-1b  the PC-2 comparator re-run against a truncated event list must FAIL.
         34/34 correctly detected.
  (NC-1/NC-1b exist so that "all controls passed" is not an empty signature --
   methodology gate #44 / lesson item 47.)

PROVENANCE
  campaign git commit f3fcdbb387906b45a5f9cf07cbf5915bcac2385d
  campaign artifacts: git_commit a02490ad, manifest_sha 79413d03, 28/28 boots,
  4 blocks, n_indep(placement) = 3 (blk1 gpu42 solo | blk2+blk3 gpu41 concurrent
  | blk4 gpu41 solo).  All quoted residency values inherit that placement caveat.
  g16_analyze.py at the time of reading: sha256 2c9626f2... (addendum E-5 final).

UNRESOLVED
  The addendum A-2 baseline ("53-68% of decode-active samples at (0,108)",
  26 telemetry files) and the A-1 baseline ("CO_RESIDENT 0.582 -> 0.818",
  s8 grid) have NO preserved selection script in the repository, so neither
  can be recomputed here.  They are quoted as written and are NOT used as
  comparators for the G16 numbers.
