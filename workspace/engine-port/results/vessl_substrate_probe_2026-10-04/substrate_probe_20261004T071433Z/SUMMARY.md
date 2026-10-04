# Substrate probe SUMMARY — substrate_probe_20261004T071433Z

> 기계적 사실만 기록한다(PASS/FAIL/UNRESOLVED/SKIPPED). 성능 해석·기판 동등성 판정 없음.
> `%smid` 집합 크기는 라벨 집합 카디널리티다(R0 label ceiling) — 계산량으로 환산 금지.
> 판정 원천: `status/*.json`(규칙: `scripts/vessl/substrate_probe_helper.py`), P0-A/R0는 각 도구 자신의 verdict json.

- mode: gpu · commit: `396a9f2d6ca05f292aa715688456f9e0b51ef378` (dirty=False) · host: main-wsp-7ix1cg4xl1jd-0
- runtime manifest: `/opt/pdmux/runtime_source_manifest.sha256` sha256 `67f7353942117b77e4a3b07ee8a62477fd08856ef285c3f79476f5781fe63daa`
- start 2026-10-04T07:14:34Z · end 2026-10-04T07:16:20Z

| stage | status | reason | wall s | non-idle clock-reason samples |
|---|---|---|---|---|
| PRE | **PASS** | manifest verified (25 entries), manual edits applied, install commit == HEAD, thread-local patch importable | 7.9 | 0/0 |
| B0 | **FAIL** | failed: sgl_kernel_spatial_import,get_sm_available_108 | 7.1 | 0/66 |
| B3-R0 | **SKIPPED** | B0 not PASS | 3.4 | 0/34 |
| B3-plain | **SKIPPED** | B0/B3-R0 not PASS (%smid label premise) |  | / |
| B3a | **SKIPPED** | B0/B3-R0 not PASS (%smid label premise) | 6.9 | 0/68 |
| B3b | **SKIPPED** | B0/B3-R0 not PASS (%smid label premise) |  | / |
| B3-gran | **SKIPPED** | B3-plain not PASS (no reference D) | 3.4 | 0/33 |
| B3c | **SKIPPED** | B0/B3-R0 not PASS (P0-A prereg sec9-2: R0 is its premise) | 3.4 | 0/34 |
| B4 | **SKIPPED** | an earlier gating stage is not PASS | 66.2 | 0/660 |
| B4-green | **SKIPPED** | B4 did not boot a server |  | / |

- B0: NVIDIA A100-SXM4-80GB | cc 8.0 | driver 580.105.08 | MIG Disabled | get_sm_available None

- clock trace (whole run): samples 895, SM MHz 210–1185, non-idle reason samples 0 []

