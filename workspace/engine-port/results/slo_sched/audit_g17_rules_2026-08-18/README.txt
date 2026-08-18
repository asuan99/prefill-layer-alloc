audit_g17_rules_2026-08-18 -- claims-auditor, RULES-LAYER audit of
DESIGN_G17_PAYOFF_BAND_2026-08-17.md (gate #34 stage 1).  READ-ONLY: nothing
outside this directory was written; no campaign artifact was modified.
GPU spend 0.  No new performance verdict; no grade change.

Run order (all pure-python + numpy, ~15 min total, a6 is the slow one):
  a1_restricted_grid.py       does the proposed S2 grid U={d44..d74} re-create
                              the prereg sec3 FORCED cell (D_ttft = S_min =>
                              Delta >= 0)?   [YES -- both phases]
  a2_block_power.py           the sec2 "tens of blocks" arithmetic, executed on
                              the real 4x7 block table; also shows K1's rank
                              rule is ANTI-consistent (P(identify) falls with N)
  a3_required_amplification.py how large must the conditional contrast be for
                              K1 to identify D_itl on a 4-arm x 4-block ON leg
  a3b_scale_invariance.py     K1 false-identification under exact ties, and the
                              scale-invariance identity created by "re-derive
                              delta_ON from the ON leg's own spread"
  a4_exposure_ceiling.py      structural ceiling of the P1 gate quantity
                              W_tim_all under a PERFECT sticky lever
  a5_state_conditional_probe.py  feasibility: state-conditional decode step time
                              from ALREADY-COLLECTED telemetry (GPU cost 0)
  a6_estimand_structure.py    what M_itl = median_req(p95(token_itl)) actually
                              measures (bimodal ITL; p95 pinned to a ~58.5 ms
                              stall mode that is arm- and load-invariant)
  a7_p2_gate_power.py         can P2 ("M_ttft within +20%") protect the sec1
                              reduction?  flip margin vs gate envelope
  a8_band_sign_power.py       what the payoff band's SIGN costs: it is the
                              d34-vs-d64 ordering, +0.092 ms on paired sd 0.704

Inputs (read-only):
  ../audit_g16_results_2026-08-17/indep_perboot.json   (independent per-boot
      re-derivation of sec4's M_ttft/M_itl; reproduces arm_means to 4 dp)
  ../residency_scope_2026-08-17/residency_scope_2026-08-17.json
  ../g16_blk*_{LO,HI}.jsonl, ../g16_blk*_telemetry.jsonl
  ../G16_RESULTS_2026-08-17.json (knobs only)

None of these scripts imports g16_analyze.py.
