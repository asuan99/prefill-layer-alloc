audit_g16_results_2026-08-17/ -- adversarial audit of G16_RESULTS_2026-08-17.md
(claims-auditor, 2026-08-17).  READ-ONLY with respect to every campaign
artifact: nothing outside this directory was written or modified.

WHY THESE SCRIPTS EXIST
  Lesson items 39 / 41: an audit that cannot be re-run is not an audit.  Every
  number quoted in the audit verdict comes from one of the four scripts below.
  NONE of them imports g16_analyze.py -- they are an INDEPENDENT re-derivation
  from the raw bench_serving artifacts and telemetry, so agreement with the
  campaign report is evidence, not a shared-code tautology.

FILES (run order)
  indep_perboot.py     re-implements sec4's M_ttft = median(ttft) and
                       M_itl = median(percentile(token_itl,0.95)) directly from
                       g16_blk*_{LO,HI}.jsonl (linear-interp percentile copied
                       from pdmux_eval.analyze.percentile, duration summed per
                       gate #7).  Writes indep_perboot.json (7 arms x 4 blocks
                       x 2 phases).  ~30 s.
                       RESULT: reproduces campaign.{LO,HI}.arm_means to 4 dp.
  sens_delta_slo.py    block-bootstrap (20 000 draws, 4 blocks with
                       replacement) of D_ttft, S_itl(60), Delta_SLO(60) and
                       gap_upper; prints the EXACT S_itl breakpoints.
  sens2.py             per-boot achieved throughput (completed / summed
                       duration) vs the TTFT donor; the reachability-edge band
                       identity (first ladder band == Delta); d74-vs-d64
                       ordering robustness; what the old 4-arm SUBGRID of this
                       same campaign would have concluded.
  sens3_ordering.py    paired within-block contrasts among U = {d44..d74} and
                       the bootstrap probability that the ordering sec2.4 uses
                       to refute the pure-exposure model is reproducible.
  sens4_placement.py   H-7: re-derives the block / placement contrasts from
                       residency_scope_2026-08-17.json, plus the d16->d74
                       imbalance ratio on EVERY reported denominator x window.
  sens5_curvature.py   executes prereg addendum A-3-2's registered primary
                       threat: arm position schedule, sum (pos-3)^2 per arm,
                       within-arm estimate of a curvature (symmetric) position
                       drift, and the curvature that would be needed to erase
                       d44's TTFT advantage.
  sens6_batch.py       independent reconstruction of the state-conditional
                       decode_running_batch_size that sec2.4 quotes without a
                       preserved computation path.  ~6 min for 2 arms.

INPUTS (read-only)
  ../g16_blk*_{LO,HI}.jsonl, ../g16_blk*_sidecar.json,
  ../g16_blk*_telemetry.jsonl, ../G16_RESULTS_2026-08-17.json,
  ../residency_scope_2026-08-17/residency_scope_2026-08-17.json
