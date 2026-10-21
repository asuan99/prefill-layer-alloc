# compo_probe compo_probe_20261021T131621Z

> **비측정 부팅 프로브(결정 24).** 아래 수치(가중치 GB·KV 풀 토큰·mamba 슬롯·cudagraph bs·요청 시간)는 모두 **인용 불가 프로브 값**이다. 성능 판정 없음. dummy 가중치 출력은 무의미하며 정확성 검사는 원본 실가중치에서만 한다.
> 원천: `status/*.json`, boot별 `<boot>/boot_facts.json`·`requests.json`·`pred.json`, 변형 `variants/a*/variant.json`.

- mode: gpu · commit: `170706171c202caf82db432f1a2491fb880c6fd4` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0 · budget_min 96 · cap 128 · ctx 73728

| stage | status | reason |
|---|---|---|
| PRE | **PASS** | manifest verified (30 entries), install commit == HEAD, thread-local role patch, LoadFormat.DUMMY |
| VARIANTS | **PASS** | 4 variants built from dc0661c829b1, no weight files, SGLang get_config (trust_remote_code) reads each pattern and Mamba state size as predicted |
| BOOT-real_orig | **PASS** | booted, load_format=auto, cudagraph bs max 128, pdmux groups=4 |
| REQ-real_orig | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| PRED-real_orig | **PASS** | weight +0.147%, k +0.062%, s +0.023% (tol +-2%); implied n_attn/n_mamba 4/27 vs 4/27 |
| TD-real_orig | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a4 | **PASS** | booted, load_format=dummy, cudagraph bs max 128, pdmux groups=4 |
| REQ-dummy_a4 | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| PRED-dummy_a4 | **PASS** | weight +0.147%, k +0.062%, s +0.023% (tol +-2%); implied n_attn/n_mamba 4/27 vs 4/27 |
| TD-dummy_a4 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a13 | **PASS** | booted, load_format=dummy, cudagraph bs max 128, pdmux groups=4 |
| REQ-dummy_a13 | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| TEL-dummy_a13 | **PASS** | 9 span pair(s), span_seq [9] has extend_tokens 8192, 1301 decode_iteration records (max bs 9), 43 co-resident decode iterations in it |
| PRED-dummy_a13 | **PASS** | weight +0.222%, k -0.021%, s +0.023% (tol +-2%); implied n_attn/n_mamba 13/18 vs 13/18 |
| TD-dummy_a13 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a22 | **PASS** | booted, load_format=dummy, cudagraph bs max 128, pdmux groups=4 |
| REQ-dummy_a22 | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| PRED-dummy_a22 | **PASS** | weight +0.239%, k +0.009%, s +0.023% (tol +-2%); implied n_attn/n_mamba 22/9 vs 22/9 |
| TD-dummy_a22 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a31 | **PASS** | booted, load_format=dummy, cudagraph bs max 128, pdmux groups=4 |
| REQ-dummy_a31 | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| TEL-dummy_a31 | **PASS** | 9 span pair(s), span_seq [9] has extend_tokens 8192, 1269 decode_iteration records (max bs 9), 34 co-resident decode iterations in it |
| PRED-dummy_a31 | **PASS** | weight +0.261%, k -0.024%, s NA (tol +-2%); implied n_attn/n_mamba 31/0 vs 31/0 |
| TD-dummy_a31 | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |
| BOOT-dummy_a31_retry | **SKIPPED** | not needed (BOOT-dummy_a31 PASS) or no budget / teardown FAIL |
| REQ-dummy_a31_retry | **SKIPPED** | not needed (BOOT-dummy_a31 PASS) or no budget / teardown FAIL |
| TEL-dummy_a31_retry | **SKIPPED** | not needed (BOOT-dummy_a31 PASS) or no budget / teardown FAIL |
| PRED-dummy_a31_retry | **SKIPPED** | not needed (BOOT-dummy_a31 PASS) or no budget / teardown FAIL |
| TD-dummy_a31_retry | **SKIPPED** | not needed (BOOT-dummy_a31 PASS) or no budget / teardown FAIL |
| BOOT-dummy_a4_vanilla | **PASS** | booted, load_format=dummy, cudagraph bs max 128, pdmux groups=None |
| REQ-dummy_a4_vanilla | **PASS** | 8/8 plain requests ok, no error signature, 64/64 output logprobs finite |
| PRED-dummy_a4_vanilla | **PASS** | weight +0.147%, k +0.008%, s +0.023% (tol +-2%); implied n_attn/n_mamba 4/27 vs 4/27 |
| TD-dummy_a4_vanilla | **PASS** | memory.used 0 MiB < 1024 MiB after teardown |

| boot | weight GiB (banner) | KV tokens | mamba slots | s/slot B | k B/tok | cudagraph bs max | PD-mux groups |
|---|---|---|---|---|---|---|---|
| real_orig | 16.58 | 886804 | 128 | 143581756 | 16394 | 128 | 4 |
| dummy_a4 | 16.58 | 886804 | 128 | 143581756 | 16394 | 128 | 4 |
| dummy_a13 | 15.04 | 419117 | 128 | 95721170 | 53237 | 128 | 4 |
| dummy_a22 | 13.49 | 334083 | 128 | 47860585 | 90120 | 128 | 4 |
| dummy_a31 | 11.94 | 298407 | 128 | 0 | 126946 | 128 | 4 |
| dummy_a4_vanilla | 16.58 | 887290 | 128 | 143581756 | 16385 | 128 | None |

TEL ON 확인(결정 26, 존재 검사 — 측정 아님). 나머지 boot는 계측 OFF(결정 22).

| boot | span pairs | extend_tokens=8192 span_seq | decode_iteration | co-resident decode iter | log failure lines |
|---|---|---|---|---|---|
| dummy_a13 | 9 | [9] | 1301 | 43 | 0 |
| dummy_a31 | 9 | [9] | 1269 | 34 | 0 |

