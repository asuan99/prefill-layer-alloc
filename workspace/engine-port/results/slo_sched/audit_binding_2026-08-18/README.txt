audit_binding_2026-08-18/ -- adversarial audit of the G16 side-output
"binding constraint" reading (readings R-1..R-4).  READ-ONLY audit; nothing in
the campaign or the analyzer was modified.

Run (module load conda/pytorch_2.9.1_cuda13; venv not required -- stdlib only,
the scripts import pdmux_eval.analyze directly from the engine-port tree):

  python3 binding_2x2.py      > binding_2x2.json       # per-request 2x2 joint of {TTFT fail} x {ITL-p95 fail}
  python3 cliff_and_blocks.py > cliff_and_blocks.json  # metric-cliff ladder + per-block decomposition + paired t(3)
  python3 mechanism.py        > mechanism.json         # TTFT/ITL vs launch decile, loss decomposition, throughput coupling
  python3 shadow_price.py     > shadow_price.json      # d(joint)/d(log threshold) = the shadow price of each SLO
  python3 lo_check.py         > lo_check.json          # LO headroom + LO paired contrasts
  python3 frechet.py          > frechet.txt            # Frechet bounds => power of the "independence" reading

Verification: binding_2x2.py reproduces
G16_RESULTS_2026-08-17.json campaign.{HI,LO}.side_outputs.per_arm
{ttft_pass_pct, itl_p95_pass_pct, joint_pass_pct} to floating-point equality
(0 mismatches, 42 cells).  Predicate = pdmux_eval.analyze.RequestResult.passes
(TTFT <= 3000 ms AND per-request p95 token-ITL <= 60 ms), empty-ITL = FAIL.

Data: jobs 884336/884410/884411/884412 (blk1..blk4), 7 arms x 1 boot x
3 rounds x 200 prompts per phase.  Zamba2-2.7B, ctx4096, ShareGPT,
LO=3 req/s, HI=12 req/s.
