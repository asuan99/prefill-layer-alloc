audit_g16_prereg_c93_2026-08-17 -- preserved reproduction path
==============================================================
claims-auditor re-audit of prereg addendum C-9(3) + D-4, run 2026-08-17,
BEFORE any G16 decision quantity was computed.  GPU spend: 0.
Lessons 39/41: the code an audit relies on must live in the repo.

Files (run from anywhere; each inserts the slo_sched dir on sys.path):
  a1_placement_diag.py         per-boot wall structure, arm order / K5 mean
                               position, blk2<->blk3 overlap, block spans,
                               residency_fraction per boot.
  a2_clock_axis.py             is sidecar `t_boot0` on a common clock origin
                               across gpu41/gpu42 (A-3 drift correction)?
                               Pairs t_boot0 with the absolute UTC stamp of
                               the H10 co-tenancy snapshot.
  a3_tie_guard_adversarial.py  T1 teeth of g16_tie_guard_test.py against a
                               pre-fix reimplementation of identify_donor;
                               T2 monotonicity counterexample search over
                               n_blocks; T3 boot_frac denominator; T4 K11
                               fallback chain (all three branches).
  a4_loader_contract.py        load_campaign_grid filtering contract with the
                               estimands MONKEY-PATCHED TO CONSTANTS, plus a
                               structural pass of the canonical loader over
                               all 56 artifacts (counts only).
  a5_f2_suppression.py         F2 "no new numbers on control failure" and F4
                               analyzer_sha256, with estimands firewalled;
                               also resolves the K11 set on the real arm set.
  a6_ladder_gate_and_reporting.py  Delta_SLO rung verdicts vs the sec6
                               "donor identified" conjunct; residency_fraction
                               _by_arm collapse.
  a7_aliasing_matrix.py        block x node x concurrency design matrix from
                               sidecars + H10 snapshots only (addendum D-4
                               criterion (i): placement, not data).

DATA-ACCESS BOUNDARY held during this audit
-------------------------------------------
NOT read: donors, Delta, Delta_SLO, S_itl, per-arm M_ttft / M_itl of the
G16 campaign.  a4/a5/a6 deliberately monkey-patch `_headline_estimands` to a
constant so the campaign estimands are never evaluated.  Values reported for
the OLD grids (sgptv 4-arm, g2_0_hard) come from --controls and are not
campaign data.

DISCLOSURE (leak, declared rather than concealed): a1 prints per-boot
wall-clock `bench_s` (t_boot1 - t_healthy).  In a closed-budget NP=200 x
ROUNDS=3 benchmark this is close to 1/throughput, and prereg sec3-1 states
that the throughput donor and the TTFT donor coincide by near-structural
force in this regime.  a1's per-arm bench_s is therefore a PROXY for the
primary-A donor.  It was read for the block-level H18 / walltime diagnosis;
it was never aggregated across blocks, never ranked across arms, and no
conclusion in the audit is conditioned on it.  Anyone re-running a1 should
know this before looking.
