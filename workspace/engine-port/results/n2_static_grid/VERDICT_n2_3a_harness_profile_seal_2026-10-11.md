# 판정서 — N2(3-a) 하네스층 델타 감사: 하네스 봉인(scope PROFILE) 가능 여부 (2026-10-11)

> 감사자: claims-auditor (읽기 전용). 저장: 메인 세션(2026-10-11), 본문은 감사 보고 그대로(서식만 정리).
> 기준: 규칙 봉인 갱신 커밋 `5e29990`(규칙 digest `fc9e6404…4c069`), 직전 판정서 `VERDICT_n2_3a_rev17_redelta_2026-10-11.md` §2-4 선행 조건 1–6. 대상 = 작업 트리 미커밋 하네스 diff 9파일(핵심 H17-7·H17-8).

## 0. 요약

| 항목 | 판정 | 死因 |
|---|---|---|
| H17-7(규칙 digest 상수 갱신) | 정확 — `--verify-registration` → `REG_OK rule_seal fc9e6404…`, `RULE_FILES` 7개 sha·순서 = `REGISTRATION_SHA256.txt` | 0 |
| H17-8(러너 비정상 종료 → `stopped="crash"`) | 정확 — fail-open 잔존 0, 시험·JH94가 사살 | 0 |
| 변이 장부(H 32/32, JH 94/94) | 원본 대조 일치 — 입력 바이트 = 현재 바이트, 모든 mtime < 시작 | 0 |
| **하네스 봉인(PROFILE)** | **현재 바이트로는 봉인 불가(보류) — 필수 수리 H17-9.** H17-9 + §4 조건 충족 시 **`GO-with-caveats`**(N2C-82·N2C-83), 재감사 불필요 | **0** |
| PREREG 문언(57·93) ↔ 실제(58·94) | 규칙 재개봉 없이 하네스 봉인 기록에 흡수 — 타당. 단 N2C-82 승계, 기록 블록 이름 제약(§3) | 0 |

신규 결함 1건(H17-9): 하네스 절을 실제로 써 넣으면 `test_n2_job`의 `test_ea_needs_profile_and_ea_sections`가 실패한다. 이 시험은 Job preflight에서 돌므로 첫 실제 Job이 `ABORT_PREFLIGHT`(exit 4)로 무조건 멈춘다 — fail-closed라 판정 오염은 없으나, 현재 바이트로 봉인하면 실행 불가능한 하네스를 봉인하는 셈이다.

## 1. 구현 정확성

### 1-1. H17-7
`n2_plan.RULE_SEAL_DIGEST = fc9e6404…4c069`; `RULE_FILES`의 PREREG `62fbfff9…`·`test_n2_probe_reduce` `a796980c…`·`n2_mutants` `ffb4d6f3…`가 REGISTRATION 규칙 절과 같고 순서도 같다(digest는 순서 민감). 머리 주석은 서술. 단위 시험 제자리 실측: `test_n2_job` 58/58, `test_n2_harness` 36/36, `n2_reach_tests` 37/37.

### 1-2. H17-8
`run_job`: `except JobAbort`(→ `stopped="job_abort"`, 재발생) 다음 `except BaseException: stopped = stopped or "crash"; raise`, `finally`가 `JOB_LEDGER.json`에 기록; 분석기는 `stopped ∉ {None,"budget"}` → F-N2-7P `UNRESOLVED`.

| 경로 | 결과 | 방향 |
|---|---|---|
| `run_boot` 바깥 try 예외(`server.start`·`settle`·`open(...,"x")`) | `_BootScope.__exit__`가 `boot_abort` 기록 후 예외를 올림 → `"crash"` → `UNRESOLVED` | fail-closed |
| watchdog SIGTERM이 셀 밖에 도착 | `on_term` flag → 다음 셀 진입 시 `RunnerStop` → `"budget"` | 등록과 일치 |
| 3층 SIGKILL, `main()`의 `run_job` 이전 예외 | 원장 부재 → `UNRESOLVED` | fail-closed(N2C-11·N2C-18이 덮음) |
| `"budget"` 뒤 예외 | `"budget"` 유지 — 이미 입장 거부 발생, `FALSIFIED` 정당 | 정확 |
| CancelledError·KeyboardInterrupt | BaseException → `"crash"` | fail-closed |

시험 `test_runner_crash_marks_crash_and_f7p_unresolved` 확인, JH94 KILLED. 남은 fail-open 0.

### 1-3. 변이 장부 대조(scratchpad `r17h/`)
`prerun_sha.txt` 21파일 sha = 현재(diff 없음). 하네스 파일 mtime 최대 19:52:45, PREREG 19:49:15, REGISTRATION 19:51:38 — 모두 시작 19:54:29 이전; `mut_harness.txt` 20:19:14, `mut_job.txt`·`end.txt` 21:37:32(순차). `mut_harness.txt` CTRL `OK (skipped=1)`, H01–H32 32 KILLED; `mut_job.txt` CTRL `OK`, JH01–JH94 94 KILLED, 중복 0. **H17-9 이전 바이트 장부이므로 H17-9 뒤 재실행 필요(§4).**

### 1-4. 하네스 절 형식 (코드: `parse_harness_sections`·`_verify_section`·`verify_harness_seal`)
```
## HARNESS SEAL
scope PROFILE
rule_seal_digest fc9e64046b8f748de1769dc7032757628f5414815e0573fa77772243075c4069
<sha256>  n2_marks.py
<sha256>  n2_loadgen.py
<sha256>  n2_probe_run.py
<sha256>  fake_n2_engine.py
<sha256>  test_n2_harness.py
<sha256>  n2_harness_mutants.py
<sha256>  n2_capacity.sh
<sha256>  n2_plan.py
<sha256>  n2_job.py
<sha256>  n2_window.py
<sha256>  n2_signature.py
<sha256>  n2_realized_mix.py
<sha256>  n2_analyze.py
<sha256>  n2_reach_tests.py
<sha256>  test_n2_job.py
<sha256>  n2_job_mutants.py
harness_digest <sha256>
```
- 파일 목록·순서 = `n2_plan.HARNESS_FILES` 16개(다르면 BROKEN); 각 파일은 현재 바이트 sha와 대조.
- `harness_digest` = 위 16줄(`<sha>␠␠<file>\n`)의 sha256: `sha256sum <16 files in order> | sha256sum` 또는 `n2_plan.harness_listing()` + `sha256_bytes`.
- 형식 함정(실측): `scope PROFILE` 줄 꼬리 공백·주석 → BROKEN; 블록 안 추가 `^<64hex>␠␠<이름>$` 줄 → BROKEN; `## HARNESS SEAL`로 시작하는 다른 헤더도 하네스 절로 파싱(`## HARNESS SEAL LEDGER (PROFILE)` → BROKEN) — 장부 기록은 `## HARNESS LEDGER PROVENANCE (scope PROFILE)` 같은 다른 이름(실측 SEALED); 블록은 다음 `## ` 줄에서 끝남; `rule_seal_digest` 줄은 하나만.
- 이 형식 사본에서 `verify_harness_seal(scope="PROFILE")` → `('SEALED', [])`, EA → `UNSEALED`(실측). 규칙 절 정규식 영향 없음.
- REGISTRATION 머리의 낡은 `# harness seal: NOT YET …` 주석 갱신. 하네스 파일과 REGISTRATION 하네스 절은 같은 커밋(Job은 `PDMUX_COMMIT` checkout 바이트를 검사).

## 2. 신규 결함 H17-9 (필수, 死因 아님)
`TestCapacityScript.test_ea_needs_profile_and_ea_sections`는 Job 레이아웃 사본의 실제 REGISTRATION을 `base`로 읽고 `ea`만 붙였을 때 `ABORT_HARNESS_UNSEALED_EA`를 기대한다. 실제 REGISTRATION에 PROFILE 절이 들어가면 `base + ea`가 SEALED → `ABORT_EA_NOT_IMPLEMENTED`로 시험 실패; `base + prof + ea`도 PROFILE 중복으로 BROKEN. 실측(scratch 트리, REGISTRATION만 바꿔 비교): 봉인본에서만 FAIL; 5 suite 274 tests 중 봉인 유발 차이는 이 1건. 결과: `n2_capacity.sh` preflight가 봉인 직후 모든 Job을 exit 4로 거부(fail-closed).
처방(scratch에서 봉인·미봉인 양쪽 `TestCapacityScript` 5/5 OK 확인):
```python
with open(reg) as fh:
    base = "".join(b for b in re.split(r"(?m)^(?=## )", fh.read()) if not b.startswith("## HARNESS SEAL"))
```
(`import re` 추가). 시험 수 58 불변. 검사는 실패해야 하는 상태에서 확인(교훈 53) — JH 장부는 미봉인 상태라 H17-9 되돌림 변이를 사살할 수 없으므로 검사 수단은 §4-3 봉인 상태 사본 preflight다(수리 전 바이트 실패는 위 실측이 보임).

## 3. PREREG 문언(57·93) ↔ 실제(58·94)
**흡수 타당, 재개봉 불필요.** 시험·변이 수는 판정 입력이 아니다(판정 경로가 숫자를 읽지 않음); PREREG가 H17-8을 이미 등록(§18-2-d #10 "`stopped = "crash"`(H17-8)") — 58·94는 등록된 수리의 귀결; 실제 장부는 상위집합(`test_n2_job` HEAD 48 → −1 +11 = 58, JH01–93 + JH94); 재개봉해도 판정 변화 0(§5). 한계: §21-31 자기검사는 하네스 봉인 장부에 대해 거짓 — 낡은 수치 문언 4회째(R15-2·R17-1·R17-4 다음). 다음 규칙 재개봉(EA 회차) 때 PREREG 하네스 수치를 "`REGISTRATION_SHA256.txt` 하네스 기록이 정본" 포인터로 바꾸기를 권고(이번 봉인 조건 아님). 기록 위치는 하네스 절 밖 별도 블록:
```
## HARNESS LEDGER PROVENANCE (scope PROFILE)
built on rule seal 5e29990 (digest fc9e6404…); files = the HARNESS SEAL scope PROFILE listing
tests in place: test_n2_harness 36 · test_n2_job 58 · n2_reach_tests 37 (rule 143, skip 0)
H01–H32: … CTRL OK (skipped=1), 32/32 KILLED
JH01–JH94: … CTRL OK, 94/94 KILLED
sealed-state preflight: scratch copy with this section, 5 suites pass (skip count …)
counts: PREREG rev17 states test_n2_job 57 / JH 93 (rule-seal-time wording); H17-8 +1 test +JH94; H17-9 edits one test (count unchanged). See N2C-82.
```

## 4. 봉인 절차(H17-9 뒤) — 정확히 충족하면 재감사 없이 `GO-with-caveats`
1. H17-9 diff는 `test_n2_job.py`의 해당 hunk(+ `import re`)만. 다른 하네스·규칙 파일 바이트를 바꾸면(주석 포함) 이 GO는 무효, 재감사.
2. 최종 바이트에서 두 장부 재실행(JH 필수; H는 장부 입력 = 봉인 바이트 동일성 유지용): CTRL `OK (skipped=1)`·`OK`, 32/32·94/94 KILLED, `prerun_sha` 기록, 입력 mtime < 시작.
3. 봉인 상태 사본 preflight: Job 레이아웃 사본 REGISTRATION에 최종 하네스 절을 넣고 `test_n2_feasibility test_n2_probe_reduce test_n2_harness test_n2_job n2_reach_tests` 실패 0; 같은 사본에서 `--verify-harness-seal --scope PROFILE` → `SEALED scope=PROFILE`, `--scope EA` → `UNSEALED`(exit 3).
4. 커밋 뒤 제자리 재확인: `--verify-registration` REG_OK, `--verify-harness-seal --scope PROFILE` SEALED, `test_n2_job` 58/58, `test_n2_harness` 36, `n2_reach_tests` 37.
5. OBSERVER_FAIL 원인 분리(같은 WALL_SCALE에서 HEAD·수정 트리 각 ≥ 2회)는 봉인 조건 아님 — dry-run 전 권고(N2C-83).

## 5. 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 57·93 흡수 ↔ 규칙 재개봉 | 두 끝 | 라벨·부호·순위·예보 반전 0 | 판정 코드가 숫자를 읽지 않음(grep); 차이는 장부 상위집합뿐 |
| H17-8 `stopped`(`None`↔`"crash"`) | 두 끝 | F-N2-7P만: 수리 전 입장 < 28이면 거짓 `FALSIFIED`, 수리 뒤 `UNRESOLVED`; 점·D·dsens 라벨 0 | JH94 KILLED; 예보는 판정 입력 아님(N2C-18) |
| `stopped or "crash"` ↔ 무조건 `"crash"` | 두 끝 | `"budget"` 선행 시만 다름(`FALSIFIED` 정당↔`UNRESOLVED`), 예보만·보수 방향 | 코드 경로 |
| 가짜 모드 `WALL_SCALE` 20↔200 | 두 끝 | 측정 경로 영향 0 — `--fake`·`PDMUX_N2_TEST_*`는 `PDMUX_COMMIT` 있으면 거부(`ABORT_TEST_KNOB_IN_MEASUREMENT`) | `n2_capacity.sh` 차단 루프, `n2_job.main` |
| 하네스 절 형식 변형 4종 | — | 전부 BROKEN 또는 SEALED, fail-open 0 | scratch 실측 |
| 봉인↔미봉인 상태(preflight) | 두 끝 | 판정 반전 없음; Job 전부 exit 4(fail-closed) → H17-9 | 274 tests 중 봉인 유발 실패 1 |

→ N1–N4 해당 없음, 死因 0.

## 6. 신규 caveat (문자 그대로)
- **N2C-82**: "사전등록 rev17 본문의 하네스 시험·변이 수(`test_n2_job` 57, JH 93)는 규칙 봉인 시점의 문언이다. 하네스 봉인(PROFILE) 장부의 정본 수치는 `REGISTRATION_SHA256.txt`의 하네스 기록(`test_n2_job` 58, JH01–JH94, H17-8로 +1·+1)이며, 사전등록 §21-31의 '시험 수가 같다' 자기검사는 하네스 봉인 장부에 대해 성립하지 않는다. 사전등록의 수치를 하네스 장부의 근거로 인용하지 않는다."
- **N2C-83**: "가짜 엔진 e2e의 `OBSERVER_FAIL`/`PASS`는 관측자 경로가 실엔진에서 어떻게 거동하는지에 대한 증거가 아니다. HEAD 사본 1회와 수정 트리 1회가 같은 결과를 낸 것은 '이번 수정의 회귀가 아니다'를 n=1로 지지할 뿐이며, 관측자 판정은 실엔진 Job 산출 `observer.verdict`로만 인용한다."
(N2C-80·81 승계.)

## 7. dry-run 전 확인 사항 (실엔진/VESSL Job에서만)
| # | 항목 | 실패 시 | fail-closed |
|---|---|---|---|
| 1 | Job 안 preflight 5 suite(H17-9 뒤); 기대 skip = 1(λ0 VESSL 아카이브 git-ignored → `test_positive_and_negative_control` skip), skip 수 기록 | exit 4 | 예 |
| 2 | checkout 바이트로 `--verify-registration`, `--verify-harness-seal --scope PROFILE` | exit 3 | 예 |
| 3 | CG 카나리아: VESSL 서버 로그 `Decode batch … cuda graph: True` 형식·local-time 스탬프와 `tz_gmtoff_s` 정합; sticky ON 로그 형식 미확인 | 첫 boot `ABORT_CG_CANARY`(≈170–229 s) | 예(N2C-80) |
| 4 | greedy non-streaming `output_ids`(VESSL 설치 트리 ↔ 로컬 dev `tokenizer_manager.py:1620-1623` 동일성 미확인) | `STICKY_CORRECTNESS_UNRESOLVED_D*`(abort 아님) | 예 |
| 5 | wiring 로그·green JSON, P3 banner, CG capture bs | exit 6 / 8 / 3 | 예 |
| 6 | 엔진 manifest 25항목, thread-local role patch, model `refs/main`, B0 SM 수 | exit 3 | 예 |
| 7 | 포트 창 20000–20799 ↔ ephemeral 범위 | Job에서 exit 5 | 예 |
| 8 | watchdog SIGTERM → `runner_stop` → `"budget"` 실서버 경로(90 s 내 미종료 시 SIGKILL → 원장 부재 → F-N2-7P `UNRESOLVED`) | 예보 결측 | 예 |
| 9 | H17-8 crash 경로는 가짜에서만 시험 | `"crash"` → `UNRESOLVED` | 예 |
| 10 | `settle` 기준선, cgroup `cpu.stat`·quota, 에너지 샘플러 상태 | 서술 | 해당 없음 |

fail-open 항목 없음(반증 실패). #3·#4는 실엔진 Job에서만 닫힌다.

## 8. 결론
- **하네스 봉인(PROFILE): 현재 바이트로는 보류, 死因 0.** H17-7·H17-8 정확, 장부 일치. 봉인 순간 preflight가 모든 Job을 거부하는 H17-9가 남아 있다.
- H17-9 + §4의 1–4를 정확히 충족하면 **`GO-with-caveats`(N2C-82·83, N2C-80·81 승계)**, 재감사 불필요. 그 밖 바이트 변경 시 재감사.
- 이 GO는 하네스 봉인 승인일 뿐 — dry-run 제출 승인 아님(launch.sh dry-run 뒤 사용자 승인 필요), 게이트 닫음도 아님.
