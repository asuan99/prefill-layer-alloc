# Longctx 128k probe SUMMARY — longctx_probe_20261020T040836Z

> 기계적 사실만(PASS/FAIL/UNRESOLVED/SKIPPED). TTFT·ITL 값은 기술값이고 인용 금지, Job 예산 입력용이다.
> 원천: `status/*.json`, 요청별 시계열은 `g*/…json`. 성능 판정 없음 -> result-analyst.

- mode: gpu · commit: `88a73ce469d0c43f4eee1a21f4d7d1a778a3273b` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0
- budget_min 75 · keep_going False · start 2026-10-20T04:08:38Z · end 2026-10-20T04:17:03Z

| stage | status | reason |
|---|---|---|
| PRE | **PASS** | manifest verified (27 entries), install commit == HEAD, models staged (Llama pinned d04e592bb4f6), 4 boot argv parse |
| G1 | **PASS** | booted PD-mux agnostic ctx 131072 cap 48; max_total_num_tokens=1587762, mamba slots=48; C_mem table in nano_pdmux/boot_facts.json |
| G2 | **UNRESOLVED** | 4/10 requests recorded, 0 not ok (probe stopped early: timeout/budget/server death) |
| G4 | **PASS** | 2 scenarios completed; time series in g4_mixed.json |
| G6 | **UNRESOLVED** | only 2 L points with ok requests |
| G7 | **PASS** | 8 requests completed (real vs synthetic ABBA); values are descriptive only |
| TD-nano_pdmux | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| G3 | **FAIL** | 2/5 requests recorded, 1 not ok (probe stopped early: timeout/budget/server death) |
| TD-nano_fused | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| G5 | **SKIPPED** | --only |
| G5-conc | **SKIPPED** | --only |
| TD-llama | **SKIPPED** | --only |
| G3-chunked | **SKIPPED** | --only |
| TD-nano_chunked | **SKIPPED** | --only |

- G2 per-L ok/fail: 32768: 2/0, 65536: 2/0

- G3 per-L ok/fail: 94500: 1/0, 95100: 0/1

