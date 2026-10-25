# 하네스층 감사 r2(최종) — r1 수리 diff `3077b69c..4430dc37`, 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 감사 상한 2회 도달 — 이후는 사용자 결정. 시험 67/67 OK(241 s) 직접 확인.

## 최종 판정
**봉인 S1 아직 불가 — MAJOR 1건 잔존(N-1).** 死因 0. MAJOR-1·m1·m2·m4·m5·m6 해소, MAJOR-2 부분 해소.

| r1 항목 | 판정 |
|---|---|
| MAJOR-1 | 해소(`agree_key` = model·mode·cap·tp·bs, 2차 표 VALID 창 중앙값 + `n_windows`, M49a·M49b) |
| MAJOR-2 | 부분 해소 → N-1 |
| m1 | 해소(공시: "4 of 6 (PARTIAL)", 짝 추가는 A-49 사용자 결정) |
| m2 | 해소(NOT_RUN 행, pre_death 표 기본 적합 제외 A-50, 재개 `@s<k>`) |
| m4 | 해소(`engine_residual`) |
| m5 | 해소(`logged_context`, `>=`, 가짜 엔진 argv 되풀이 M57) |
| m6 | 해소, 부작용 N-2 |

**수리자 질문(ERROR==ERROR 통과 경로)**: 실제 Job에는 없음 — `costterms_job.sh`가 항상 `--dir "${here}"`를 넘겨 `d.resolve() == HERE`, ERROR는 위반(scratch에서 analytic_terms 제거 → rc 13 확인). `make`가 ERROR를 그대로 쓰는 점은 MINOR(check가 막으므로 안전 쪽).

## 새 문제
- **N-1 (MAJOR, 봉인 차단)**: `ct_harness_digest.covered()`가 ct_*.py·costterms_job.sh·config·ENGINE_DIGEST.txt만 본다. spec digest(`spec_dict`)는 대문자 상수만 담고 `o_tokens`·`t3_o_tokens`가 계산하는 셀별 O를 담지 않는다. 재현: `kv_bytes_per_token` 3배 → O(32, 8k) 6,173→5,393, T3 O 4,523으로 바뀌는데 `check` rc 0. 같은 위치: `scripts/compo/make_variant.py`, `scripts/bootstrap/engine_digest.py`, `job_entry.sh`. PREREG §3-1 "O … spec digest로 봉인된다"도 사실과 다름(plan.json 지문에 O가 있으나 봉인 검사가 대조하지 않음).
- N-2 (MINOR): m6 수리가 §1-7 fallback(`CLIENT_ONLY_UNVERIFIED`)을 막음 — vanilla에서 probe 부재만 매트릭스 위반에서 빼고 `NOT_COLLECTED`로.
- N-3 (MINOR, 표기): 재개된 사망 boot의 `not_run_after_death` 목록에 `@s<k>`에서 실제로 돈 셀이 남음 → RESUME_LOG와 함께 읽기.

## 사용자 결정(감사 상한)
N-1: (a) [권고] analytic_terms.py·make_variant.py·engine_digest.py·job_entry.sh + 전 Job `make_plan` 출력 sha를 digest에 넣고 "analytic_terms 1바이트 변경 → rc 13" 변이 추가, 3회차 감사 없이 그 시험 통과로 봉인 / (b) 수리 뒤 3회차 감사(상한 예외) / (c) 수리 없이 봉인 + caveat(비권고).
A-49·A-50·m12는 봉인을 막지 않는 사용자 결정으로 남음.
