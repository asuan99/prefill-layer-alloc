# engine gate alpha  -- FAIL

job pdmux-engine-gate-alpha-0990828-20261020081434  commit 09908284b3ab08a3cb878385ad792f1586d8bb67  elapsed 1721.1 s  hardcap 120.0 min

Harness verdict on this Job's data only; no performance statement. Judgement/analysis -> result-analyst; refutation -> claims-auditor.

| check | verdict |
|---|---|
| a_spass_agnostic | PASS |
| a_cpass_agnostic | PASS |
| a_spass_fused_sym | PASS |
| a_cpass_fused_sym | PASS |
| a_spass_llama | PASS |
| a_cpass_llama | PASS |
| b_int64_le64k_sanity_agnostic | PASS |
| b_int64_le64k_sanity_fused_sym | PASS |
| emission_B2 | PASS |
| i6_B2 | PASS |
| i4_B2 | PASS |
| i5i6_B2 | FAIL |
| emission_B4 | PASS |
| i6_B4 | PASS |
| i4_B4 | PASS |
| i5i6_B4 | PASS |
| emission_B6 | PASS |
| i6_B6 | PASS |
| i4_B6 | PASS |
| i5i6_B6 | PASS |
| emission_B7 | PASS |
| i6_B7 | PASS |
| schema_all_on | PASS |
| instrumentation_level | PASS |
| cudagraph_on_honoured | PASS |
| survival_all_steps | PASS |
| gpu_memory_released | PASS |
| digest_3x | PASS |

| stage | status | reason |
|---|---|---|
| B2_green | RECORDED | 3 candidate green-context/SM lines (recorded, not a pass criterion) |
