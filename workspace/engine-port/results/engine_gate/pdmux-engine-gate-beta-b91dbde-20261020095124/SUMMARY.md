# engine gate beta  -- FAIL

job pdmux-engine-gate-beta-b91dbde-20261020095124  commit b91dbded440b2640ceaae41a5d8dda88f271e35f  elapsed 884.4 s  hardcap 120.0 min

Harness verdict on this Job's data only; no performance statement. Judgement/analysis -> result-analyst; refutation -> claims-auditor.

| check | verdict |
|---|---|
| beta0_cases | FAIL |
| beta0_contig_digest_equal | PASS |
| beta0_green_event_calib | PASS |
| beta1_BETA1_fused | PASS |
| beta1_BETA1_agn | PASS |
| beta2_BETA2_boot1 | FAIL |
| beta2_BETA2_boot2_blocking | PASS |
| beta2_reapplied_ok | PASS |
| digest_3x | PASS |
| instrumentation_level | PASS |

| stage | status | reason |
|---|---|---|
| beta0_patched_view_w22656_T12288_r0 | FAIL | patched view w22656 T=12288: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T94784_r0 | FAIL | patched view w22656 T=94784: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T94784_r1 | FAIL | patched view w22656 T=94784: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T94800_r0 | FAIL | patched view w22656 T=94800: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T94800_r1 | FAIL | patched view w22656 T=94800: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T95100_r0 | FAIL | patched view w22656 T=95100: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T95100_r1 | FAIL | patched view w22656 T=95100: MISMATCH (expected MATCH) |
| beta0_patched_view_w22656_T126976_r0 | FAIL | patched view w22656 T=126976: MISMATCH (expected MATCH) |
| beta0_patched_view_w11328_T95100_r0 | FAIL | patched view w11328 T=95100: ERROR (expected MATCH) |
| beta0_patched_view_w11328_T126976_r0 | FAIL | patched view w11328 T=126976: ERROR (expected MATCH) |
| beta0_patched_view_w11328_T189000_r0 | FAIL | patched view w11328 T=189000: ERROR (expected MATCH) |
| beta0_patched_view_w11328_T190500_r0 | FAIL | patched view w11328 T=190500: ERROR (expected MATCH) |
| beta0_patched_contig_w22656_T95100_r0 | FAIL | patched contig w22656 T=95100: MISMATCH (expected MATCH) |
| beta0_pristine_view_w22656_T12288_r0 | FAIL | pristine view w22656 T=12288: MISMATCH (expected MATCH) |
| beta0_pristine_view_w22656_T94784_r0 | FAIL | pristine view w22656 T=94784: MISMATCH (expected MATCH) |
| beta0_pristine_view_w22656_T94784_r1 | FAIL | pristine view w22656 T=94784: MISMATCH (expected MATCH) |
| beta0_pristine_view_w11328_T95100_r0 | FAIL | pristine view w11328 T=95100: ERROR (expected MATCH) |
| beta0_pristine_view_w11328_T126976_r0 | FAIL | pristine view w11328 T=126976: ERROR (expected MATCH) |
| beta0_pristine_view_w11328_T189000_r0 | FAIL | pristine view w11328 T=189000: ERROR (expected MATCH) |
| beta0_pristine_view_w11328_T190500_r0 | FAIL | pristine view w11328 T=190500: ERROR (expected BROKEN) |
| beta0_pristine_contig_w22656_T95100_r0 | FAIL | pristine contig w22656 T=95100: MISMATCH (expected MATCH) |
| BETA2_boot1_crash95100 | FAIL | pristine tree: server died; signature missing |
