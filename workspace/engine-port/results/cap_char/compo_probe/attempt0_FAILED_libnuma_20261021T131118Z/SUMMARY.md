# compo_probe compo_probe_20261021T131118Z

> **비측정 부팅 프로브(결정 24).** 아래 수치(가중치 GB·KV 풀 토큰·mamba 슬롯·cudagraph bs·요청 시간)는 모두 **인용 불가 프로브 값**이다. 성능 판정 없음. dummy 가중치 출력은 무의미하며 정확성 검사는 원본 실가중치에서만 한다.
> 원천: `status/*.json`, boot별 `<boot>/boot_facts.json`·`requests.json`·`pred.json`, 변형 `variants/a*/variant.json`.

- mode: gpu · commit: `170706171c202caf82db432f1a2491fb880c6fd4` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0 · budget_min 96 · cap 128 · ctx 73728

| stage | status | reason |
|---|---|---|
| PRE | **PASS** | manifest verified (30 entries), install commit == HEAD, thread-local role patch, LoadFormat.DUMMY |
| VARIANTS | **PASS** | 4 variants built from dc0661c829b1, no weight files, SGLang get_config (trust_remote_code) reads each pattern and Mamba state size as predicted |
| BOOT-real_orig | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-real_orig | **SKIPPED** | server not healthy |
| PRED-real_orig | **SKIPPED** | server not healthy |
| TD-real_orig | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a4 | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a4 | **SKIPPED** | server not healthy |
| PRED-dummy_a4 | **SKIPPED** | server not healthy |
| TD-dummy_a4 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a13 | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a13 | **SKIPPED** | server not healthy |
| TEL-dummy_a13 | **SKIPPED** | server not healthy |
| PRED-dummy_a13 | **SKIPPED** | server not healthy |
| TD-dummy_a13 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a22 | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a22 | **SKIPPED** | server not healthy |
| PRED-dummy_a22 | **SKIPPED** | server not healthy |
| TD-dummy_a22 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a31 | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a31 | **SKIPPED** | server not healthy |
| TEL-dummy_a31 | **SKIPPED** | server not healthy |
| PRED-dummy_a31 | **SKIPPED** | server not healthy |
| TD-dummy_a31 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a31_retry | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a31_retry | **SKIPPED** | server not healthy |
| TEL-dummy_a31_retry | **SKIPPED** | server not healthy |
| PRED-dummy_a31_retry | **SKIPPED** | server not healthy |
| TD-dummy_a31_retry | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a4_vanilla | **FAIL** | server did not reach /health (died, boot timeout or budget) |
| REQ-dummy_a4_vanilla | **SKIPPED** | server not healthy |
| PRED-dummy_a4_vanilla | **SKIPPED** | server not healthy |
| TD-dummy_a4_vanilla | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |

| boot | weight GiB (banner) | KV tokens | mamba slots | s/slot B | k B/tok | cudagraph bs max | PD-mux groups |
|---|---|---|---|---|---|---|---|
| real_orig | None | None | None | NA | NA | None | None |
| dummy_a4 | None | None | None | NA | NA | None | None |
| dummy_a13 | None | None | None | NA | NA | None | None |
| dummy_a22 | None | None | None | NA | NA | None | None |
| dummy_a31 | None | None | None | NA | NA | None | None |
| dummy_a31_retry | None | None | None | NA | NA | None | None |
| dummy_a4_vanilla | None | None | None | NA | NA | None | None |

TEL ON 확인(결정 26, 존재 검사 — 측정 아님). 나머지 boot는 계측 OFF(결정 22).

| boot | span pairs | extend_tokens=8192 span_seq | decode_iteration | co-resident decode iter | log failure lines |
|---|---|---|---|---|---|

