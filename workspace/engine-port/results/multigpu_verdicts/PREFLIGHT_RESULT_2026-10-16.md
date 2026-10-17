# PREFLIGHT_RESULT 2026-10-16 (CPU container run) -- label PREFLIGHT_FAIL_NEW

Correctness probe, not a measurement. Not a gate pass: `results/multigpu/PREFLIGHT_PASS.json` was NOT written.
No analysis of cause beyond pointers (-> result-analyst / engine-porter).

- Job `pdmux-preflight-cl3-68f386b-20261016114954` (slug job-pww479muugxs), spec `resourcespec-a100cpu` (CPU only, 3.5 CPU), image `1e0d335b…`, Job HEAD `68f386b6`, PRE `7b15214b`, POST `e43749c91b35…`. Job HEAD vs POST on the 7 gate paths: empty diff.
- State `failed` by design (judge rc != 0); DONE + `sha256sum -c` passed in fetch.sh; 0 merge conflicts. Elapsed in-container 9730 s (suites: pre 32 min, post 48 min, post 48 min, pre 33 min).
- CPU mode (meta/CPU_ONLY.txt): GPU stamp/samplers skipped; `LD_LIBRARY_PATH` + `/usr/local/cuda-13.0/compat`; venv mode `image_python_prefix` (/opt/conda). Preceding probe Job `pdmux-cpu-probe-68f386b-20261016113915`: `import torch, sgl_kernel`, `sgl_kernel.spatial` and `multiplexing_mixin` all OK; 161 tests of the 10 CL-3 import modules OK.
- Judge: pre ran=989 failing=32 | post ran=1487 failing=26; L2 reproduced 9/12; new=2; sig_changed=9; fixed_in_post=17.
- Coverage of the 31 target items: A_import9 9/9, B_holb4 3/4, C_venv17 17/17, D_zamba2 1/1, E_rev12 1/1 = 31 of 32 target items verified; 1 HOLE (the preflight design text says 31 items; the judge counts 32).

## Items needing attention
NEW (fail in POST only, both reps, not in PRE):
1. `test_n3_capacity.TestRegistration.test_plan_selftest` -- `n3_plan.mirror_check`: installed POST tree hashes of `chunk_probe.py`, `holb_probe.py`, `multiplexing_mixin.py`, `telemetry.py` != the n3 registered ENGINE_MANIFEST (PRE matches). Same DRIFT lines also show in `test_resume_and_its_controls`, `test_config_without_fixed_dsm_aborts`, `test_unbroken_copy_runs` (rc 3). Pointer: n3 registration predates the merge's changes to those four engine files.
2. `test_mg_harness.JobScriptsDryRun.test_m3_dry_run_every_job` -- DRYRUN_NOTES lacks `ABORT_MODEL_REVISION` (only ABORT_SUBSTRATE shown first); the test assumes the machine would abort on the model revision. Possibly container-dependent (/data/hf present). Needs reading by result-analyst; may be CPU-substrate-related (the notes show the GPU checks failing, as expected on CPU).
CHANGED signatures (9): test_lambda0_prereg x4, test_lambda_star_per_shape x2, test_n3_capacity.TestBootBranchSnippets x2 (t_h1_sticky, t_h3_floor), TestBudgetEnforcer.test_overrunning_unit_is_cut.
HOLE: `B_holb4:test_holb_probe.TestSchedulerHooksInstalled.test_patch_reapplies_the_hooks_and_nothing_else` -- result line interleaved with a ResourceWarning in both arms, so the judge could not read a status.

## CPU caveat for UNVERIFIABLE/FAIL reading
Tests that need an A100 (e.g. `ABORT_SUBSTRATE`-dependent dry-runs) cannot be exercised as on the target substrate; the run shows the GPU gate checks failing in dry-run notes. Whether items 1-2 reproduce on a GPU container is untested.

Artifacts: `results/preflight_cl3/pdmux-preflight-cl3-68f386b-20261016114954/` (PREFLIGHT_LABEL, judge.json, logs/, meta/), `results/preflight_cl3/vessl/<job>/`, probe: `results/cpu_probe/`.
