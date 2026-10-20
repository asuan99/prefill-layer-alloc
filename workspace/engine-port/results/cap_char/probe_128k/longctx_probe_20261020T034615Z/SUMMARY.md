# Longctx 128k probe SUMMARY — longctx_probe_20261020T034615Z

> 기계적 사실만(PASS/FAIL/UNRESOLVED/SKIPPED). TTFT·ITL 값은 기술값이고 인용 금지, Job 예산 입력용이다.
> 원천: `status/*.json`, 요청별 시계열은 `g*/…json`. 성능 판정 없음 -> result-analyst.

- mode: gpu · commit: `63de358261cf6688d32906f611142801828b66fd` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0
- budget_min 120 · keep_going False · start 2026-10-20T03:46:17Z · end 2026-10-20T04:00:31Z

| stage | status | reason |
|---|---|---|
| PRE | **PASS** | manifest verified (27 entries), install commit == HEAD, models staged (Llama pinned d04e592bb4f6), 4 boot argv parse |
| G1 | **PASS** | booted PD-mux agnostic ctx 131072 cap 48; max_total_num_tokens=1587762, mamba slots=48; C_mem table in nano_pdmux/boot_facts.json |
| G2 | **FAIL** | 7/10 requests recorded, 1 not ok (probe stopped early: timeout/budget/server death) |
| G4 | **SKIPPED** | server not alive after G2 |
| G6 | **PASS** | 3 distinct L points measured (descriptive table in facts) |
| G7 | **SKIPPED** | server not alive |
| TD-nano_pdmux | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| G3 | **FAIL** | 4/5 requests recorded, 1 not ok (probe stopped early: timeout/budget/server death) |
| TD-nano_fused | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| G5 | **PASS** | 10/10 requests completed with forced token counts, 0 OOM log hits |
| G5-conc | **PASS** | 6/6 requests completed with forced token counts, 0 OOM log hits |
| TD-llama | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| G3-chunked | **PASS** | 5/5 requests completed with forced token counts, 0 OOM log hits |
| TD-nano_chunked | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |

- G2 per-L ok/fail: 16384: 2/0, 32768: 2/0, 65536: 2/0, 98304: 0/1

- G3 per-L ok/fail: 16384: 1/0, 32768: 1/0, 65536: 1/0, 98304: 0/1

- G3-chunked per-L ok/fail: 16384: 1/0, 32768: 1/0, 65536: 1/0, 98304: 1/0, 126976: 1/0

- G5 per-L ok/fail: 16384: 2/0, 32768: 2/0, 65536: 2/0, 98304: 2/0, 126976: 2/0

