audit_eb1_rules_2026-08-18/ -- rules-layer (gate #34 stage 1) adversarial audit of
DESIGN_EB1_SHADOW_PRICE_2026-08-18.md.  READ-ONLY: nothing in the campaign, the
analyzer, or the design document was modified.

Run (module load conda/pytorch_2.9.1_cuda13; stdlib only -- the scripts import
pdmux_eval.analyze directly from the engine-port tree):

  python3 window_shape.py       > window_shape.json       # theta-sweep: which thresholds are jointly adjudicable
  python3 ceiling.py            > ceiling.json            # ITL(B) from srv.log; max_running_requests=48 cap
  python3 window_exists.py      > window_exists.json      # DECISIVE: can ANY rate make a cell adjudicable?
  python3 window_exists_lobo.py > window_exists_lobo.json # same, leave-one-block-out (design sec2-2)
  python3 estimand_vs_gate.py   > estimand_vs_gate.json   # gate #6 caps the estimand; eps sweep; block CIs
  python3 reachable_price.py    > reachable_price.json    # raw S ratio vs policy-reachable S ratio
  python3 reachability.py       > reachability.json       # linearisation vs observed cross-arm goodput

Data: jobs 884336/884410/884411/884412 (G16 blk1..blk4), 7 arms x 1 boot x
3 rounds x 200 prompts per phase, Zamba2-2.7B, ctx4096, ShareGPT, LO=3 / HI=12
req/s.  Predicate = pdmux_eval.analyze.RequestResult.passes (TTFT <= 3000 ms AND
per-request p95 token-ITL <= 60 ms); band mass = g16_analyze.py:300-304
(+-10% around each threshold).

Cross-check: the pooled {ttft_pass_pct, itl_p95_pass_pct, joint_pass_pct} and the
band masses reproduce G16_RESULTS_2026-08-17.json campaign.{HI,LO}.side_outputs
and audit_binding_2026-08-18/binding_2x2.json (same loader, same predicate).
