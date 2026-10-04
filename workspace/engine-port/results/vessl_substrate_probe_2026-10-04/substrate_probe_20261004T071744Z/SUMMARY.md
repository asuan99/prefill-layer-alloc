# Substrate probe SUMMARY — substrate_probe_20261004T071744Z

> 기계적 사실만 기록한다(PASS/FAIL/UNRESOLVED/SKIPPED). 성능 해석·기판 동등성 판정 없음.
> `%smid` 집합 크기는 라벨 집합 카디널리티다(R0 label ceiling) — 계산량으로 환산 금지.
> 판정 원천: `status/*.json`(규칙: `scripts/vessl/substrate_probe_helper.py`), P0-A/R0는 각 도구 자신의 verdict json.

- mode: gpu · commit: `396a9f2d6ca05f292aa715688456f9e0b51ef378` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0
- runtime manifest: `/opt/pdmux/runtime_source_manifest.sha256` sha256 `67f7353942117b77e4a3b07ee8a62477fd08856ef285c3f79476f5781fe63daa`
- start 2026-10-04T07:17:46Z · end 2026-10-04T07:26:08Z

| stage | status | reason | wall s | non-idle clock-reason samples |
|---|---|---|---|---|
| PRE | **PASS** | manifest verified (25 entries), manual edits applied, install commit == HEAD, thread-local patch importable | 7.9 | 0/0 |
| B0 | **PASS** | A100 cc8.0, 108 SM (torch and sgl_kernel), MIG off | 7.9 | 0/74 |
| B3-R0 | **PASS** | R0 verdict = GLOBALLY_CONSISTENT_LABEL | 22.0 | 0/219 |
| B3-plain | **PASS** | /D/=108 == get_sm_available, unchanged after green creation |  | / |
| B3a | **PASS** | 92/16:PASS; 84/24:PASS; 74/34:PASS; 64/44:PASS; 16/92:PASS | 11.6 | 0/115 |
| B3b | **PASS** | 92/16:PASS; 84/24:PASS; 74/34:PASS; 64/44:PASS; 16/92:PASS |  | / |
| B3-gran | **PASS** | 1/107->4/104; 2/106->4/104; 3/105->4/104; 4/104->4/104; 5/103->6/102; 6/102->6/102; 15/93->16/92; 17/91->18/90; 18/90->18/90; 20/88->20/88; 22/86->22/86; 16/17->16/18; 16/19->16/20; 16/21->16/22; 2/2->REJECTED | 76.0 | 0/755 |
| B3c | **PASS** | P0-A verdict = CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY | 224.3 | 0/2238 |
| B4 | **PASS** | boot ok, 12/12 non-empty greedy responses, cuda graph captured (disable_cuda_graph=False), pdmux on, no Traceback | 148.9 | 0/1482 |
| B4-green | **PASS** | 5 groups: driver smCount == target (green), no green context (plain) |  | / |

- B0: NVIDIA A100-SXM4-80GB | cc 8.0 | driver 580.105.08 | MIG Disabled | get_sm_available 108
- B3a: [92, 16]->[92, 16] PASS; [84, 24]->[84, 24] PASS; [74, 34]->[74, 34] PASS; [64, 44]->[64, 44] PASS; [16, 92]->[16, 92] PASS
- B3b: [92, 16]-> PASS; [84, 24]-> PASS; [74, 34]-> PASS; [64, 44]-> PASS; [16, 92]-> PASS

## B3-gran (descriptive: requested -> realized |%smid set|, driver smCount)

| requested p/d | outcome | realized | driver smCount | note |
|---|---|---|---|---|
| 1/107 | ROUNDED | [4, 104] | [4, 104] |  |
| 2/106 | ROUNDED | [4, 104] | [4, 104] |  |
| 3/105 | ROUNDED | [4, 104] | [4, 104] |  |
| 4/104 | EXACT | [4, 104] | [4, 104] |  |
| 5/103 | ROUNDED | [6, 102] | [6, 102] |  |
| 6/102 | EXACT | [6, 102] | [6, 102] |  |
| 15/93 | ROUNDED | [16, 92] | [16, 92] |  |
| 17/91 | ROUNDED | [18, 90] | [18, 90] |  |
| 18/90 | EXACT | [18, 90] | [18, 90] |  |
| 20/88 | EXACT | [20, 88] | [20, 88] |  |
| 22/86 | EXACT | [22, 86] | [22, 86] |  |
| 16/17 | ROUNDED | [16, 18] | [16, 18] |  |
| 16/19 | ROUNDED | [16, 20] | [16, 20] |  |
| 16/21 | ROUNDED | [16, 22] | [16, 22] |  |
| 2/2 | REJECTED |  |  | RuntimeError('ERROR: CUDA DRV call "cuDevResourceGenerateDesc(&desc[1], &resources[1], 1)" in line 81 of file ../csrc/sp |

- B3-R0 vs KISTI (kisti_889631): this run = `GLOBALLY_CONSISTENT_LABEL`, KISTI = `GLOBALLY_CONSISTENT_LABEL` (나란히 기록만)

- B3c vs KISTI (kisti_890893): this run = `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`, KISTI = `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY` (나란히 기록만)

- clock trace (whole run): samples 4883, SM MHz 210–1410, non-idle reason samples 0 []

