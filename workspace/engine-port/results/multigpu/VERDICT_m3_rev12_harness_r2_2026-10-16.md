# 판정서: M3 rev12 하네스층 감사 2회차(마지막) (2026-10-16)

> claims-auditor(읽기 전용) 최종 보고를 main 세션이 저장(2026-10-16). 하네스층 감사 상한 2회 중 2회차(마지막).
> 범위(1회차 §8): R-H1–R-H6 수리 diff만. 대상 `5dc5fa8b`(하네스 수리), `57a6c01f`(PREREG D1·§19-6 `.git`), `749d134c`(§19-6 (i) 작성 주체·형식). 전수 재감사 아님.
> 저장소 변경 0. scratchpad: `r2/repo/`(`git archive 749d134c`), `r2tools/mut_r2.py`(감사자 독자 변이 16), `r2tools/run_mut_all_copy.py`(engine-porter 실행기 사본), `jl/`(실제 저장소 bundle로 만든 Job 레이아웃), `r2w/`(record 경계 사례). 변이는 매번 원복·바이트 대조. sbatch 구동 시험 미실행.

## 1. 총평

| 대상 | 판정 | 死因 | 차단 결함 |
|---|---|---|---|
| rev12 하네스 `6321c8f4`+`0b541c74`+수리 `5dc5fa8b`(+문서 `57a6c01f`·`749d134c`) | **재봉인 가능** | 0 | 0 — R-H1·R-H2 해소. R-H7 봉인 전제는 남음 |

- R-H1–R-H6 처방대로 구현. 1회차 생존 변이 7개(M1·M12·M13·M20–M23)와 수리 변이(G1·G2·G4·M24) 사본 재실행에서 전부 KILLED, 대조군 통과.
- 감사자 독자 변이 16개: KILLED 13, 대조군 1, 생존 2(A13 등가 변이, A14 수리 이전부터 있던 시험 공백·현 코드 맞음) — 반전 없음, 死因 아님.
- 수리가 만든 자유 표면 6개 반전 시험: fail-open 반전 0, caveat 3개.

## 2. 재실행 결과 (사본, HEAD `749d134c`)

- `tests.test_mg_rev12`: 79 OK, skip 1(`MG_TEST_X_TOKENS` 미설정).
- `test_mg_harness`·`test_m2g_gate`·`test_loadgen_outcome`·`test_mg_symlink_layout`: 294, 실패 1(`test_r5_launch_dry_run…` "not a git repository", archive 사본 환경 아티팩트, 1회차와 같음). 새 실패 0.
- engine-porter 실행기 재실행: M0 통과. KILLED: M1, M6b, M8, M12, M13, M20, M21, M22, M23, G1, G2, G4, M24.
- 감사자 독자 변이(`mut_r2.py`):

| 변이 | 결과 | 죽인 시험 |
|---|---|---|
| A0 대조군 | 통과 | — |
| A1 `committed_matches_head` 바이트 비교 제거 | KILLED | job_layout·G2·uncommitted |
| A2 `rel`을 `realpath`로 | KILLED | job_layout·tool_output 두 레이아웃 |
| A3 record 누락 필드 무시 | KILLED | selftest(R5–R7) |
| A4 label에 'PASS' 포함이면 PASS | KILLED | G4 |
| A5 logprob 검사를 "0개만 FAIL"로 | KILLED | M24 |
| A6 `n_logprobs`를 output_ids 길이로 | KILLED | serve_records_one_logprob |
| A7 gate verdict 검사 제거 | KILLED | G4·tool_output |
| A8 `check_committed` 항상 참 | KILLED | m3_job check_committed |
| A9 POST 기본값 `dc06c870` 복귀 | KILLED | shell_tool |
| A10 POST 40-hex 검사 제거 | KILLED | shell_tool |
| A11 record post sha 검사 제거 | KILLED | selftest R7 |
| A12 record image digest 검사 제거 | KILLED | selftest R5 |
| A13 `cat-file -e` 검사 제거 | **생존 — 등가** | 미추적 파일은 `git show HEAD:<rel>` rc≠0이라 여전히 False |
| A14 gate "PASS 파일 sha256 == .sha256" 검사 제거 | **생존 — 수리 이전 공백** | §5-3 |
| A15 shell 7경로 중 `scripts/multigpu` 제거 | KILLED | shell_tool(7경로 핀) |

## 3. 항목별 판정

- **(a) Job 레이아웃 gate 시험: CONFIRMED.** `job_layout()`은 `job_entry.sh:111-124`를 단계별 재현(bundle(`HEAD`) → `clone --no-checkout` → `checkout <sha>` → results run 볼륨 복사 → 원 results 삭제 → symlink; `cp -a` ≡ `copytree(symlinks=True)`). 결함 자체도 고정(옛 `diff --quiet` rc≠0, 새 검사 `(True, "")`, 미추적 False). 실제 저장소 독립 재현(`jl/`, git 2.34.1, `749d134c` bundle 155 MB): 옛 `git diff --quiet HEAD -- …/M2G_PASS_TABLE_J2.json.sha256` rc 1, `mg_plan.py committed-clean` = `COMMITTED_CLEAN`, run 볼륨 1바이트 변조 → `NOT_COMMITTED_CLEAN … bytes differ from HEAD`, `preflight-gate` → `(i) PREFLIGHT_PASS.json or its .sha256 absent`(커밋된 PASS 없어 기대대로). 평면 레이아웃 J1·J2·J4·J8 `.sha256` 넷 다 `COMMITTED_CLEAN`. G1·G2 KILLED(G2 시험은 출하 `m3_job.sh`에서 `check_committed`를 잘라 bash로 실행). 같은 결함의 다른 사이트: 하네스 git 호출은 `mg_plan.py:678,681,706`과 `mg_job_common.sh:99`(이미 results 제외)뿐 → 남은 사이트 0.
- **(b) `record` 산출물: CONFIRMED.** `test_tool_output_passes_the_gate_in_both_layouts`가 CLI `record` 바이트를 그대로 커밋해 두 레이아웃 `(True, [])` 확인, FAIL_NEW·UNVERIFIABLE은 두 레이아웃 거짓 + `verdict '<label>'` 사유. G4 KILLED. PASS 오기록 경로 없음: `verdict = label if label in LABELS else UNVERIFIABLE`, 누락 필드 하나면 PASS → UNVERIFIABLE, gate가 verdict 재검사(A7 KILLED). 경계(`r2w/`): label 키 없음·job_name 빈 값·대문자 post sha → UNVERIFIABLE. judge.json이 JSON 배열이면 `AttributeError` crash → 파일 미생성 → gate (i) absent → 거짓(fail-closed; 실제 judge는 dict). smoke(`PDMUX_PF_MODULES`)는 `--invalid` → UNVERIFIABLE(`preflight_cl3.sh:272`). POST 필수화(기본값 제거, 40-hex 아니면 exit 2, A9·A10 KILLED, dry-run rc 2·메시지 시험 고정). 7경로 핀(`post_vs_job_head` = `C.PREFLIGHT_GATE_PATHS` 정규식 대조, A15 KILLED), targets에 `E_rev12 module test_loadgen_outcome`.
- **(c) M1·M12·M13·M20–M23·M24: CONFIRMED.** 모두 KILLED, 대조군 통과. `XWiringEndToEnd`는 합성 셀 → `aggregate` → `aggregate_jobs` 끝까지(S1: X 1.2 vs A 1.0, S2: X 앵커 1.0 vs X 기준 1.2). M1 시험은 `n_logprobs` 32 고정으로 R-H5가 M1을 가리지 못하게 함(맞음). M8은 `job_effect()` 계약에 별도 고정, 등가성 판정은 1회차와 같음.
- **(d) 문서: CONFIRMED(D1·D4·D6), 잔여 표기 지연 2건(死因 아님).** D1 §19-2:145·§19-13 U7 → `JOB_EFFECT` 불변(148행·MGH12-3과 정합). D4 `mg_label` 머리 "rev11 규칙 + late-send 합산, 순수 rev11 아님". D6 docstring 정정. §19-6 `.git` 문장 1회차 처방과 실질 동일. `749d134c`가 (i) 작성 주체·필드 확정. 잔여 1: §19-2:147 (c)는 아직 "logprob 전부 유한" — 코드(R-H5)는 "요청마다 정확히 32개 + 유한", 코드가 엄격(fail-closed), 봉인 문서에서 문구 정합. 잔여 2: §19-2:145·§19-16:352 "engine-porter 커밋(미도착)"은 `0b541c74` 뒤 낡은 문구.
- **R-H5 엔진 가정: PLAUSIBLE(코드 읽기, 실측 아님).** `scheduler.py:1902-1915`: `return_logprob`만 주고 start_len 미지정이면 기본값이 `len(origin_input_ids)` — 명시 `len(ids)`와 같은 경로, 1917행은 `>`일 때만 abort. 출력 logprob은 토큰마다 1개: prefill 마지막 chunk에서 `add_logprob_return_values` 1개(`scheduler_output_processor_mixin.py:180-218`, `:843-845`), 비최종 chunk(`is_chunked > 0`) 미부착, decode step마다 1개, 전송 시 `[offset:len(output_ids)]`(`:1089-1094`). PD-mux split prefill도 같음: `tp_worker.forward_batch_split_prefill`(`:536-559`)은 최종 split에서 `logits_processor`(`nemotron_h.py:889`)·`sample`, `run_batch`(`scheduler.py:2785-2791`)가 split 경로에도 `extend_*_per_req`를 채움. 따라서 `ignore_eos`·32토큰이면 32개 기대. 미실행. 어긋나면 xcheck FAIL → 그 config X 미정의(fail-closed).

## 4. 반전 시험 표 (수리가 만든 자유 표면)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| `committed_matches_head` `rel` 계산 | abspath(현행) ↔ realpath, symlink 너머 다른 파일 비교 시도 | fail-open 반전 없음. realpath면 Job에서 항상 거짓(fail-closed). 추적 symlink 경로는 `HEAD:<rel>` 조회 실패 → 거짓. 읽는 바이트는 Job이 쓰는 run 볼륨 사본 | A2 KILLED, `jl/` 1바이트 변조 → `NOT_COMMITTED_CLEAN`, `.gitattributes` 없음(필터·LFS로 blob≠checkout 경로 없음) |
| `record`의 judge.json 신뢰 | label 임의·키 없음·JSON 배열·파일 없음 | PASS 반전 0(정확히 `PREFLIGHT_PASS`일 때만). 배열 → crash → 파일 없음 → gate 거짓 | A4 KILLED, selftest R3·R4, `r2w/` nolabel·nojob → UNVERIFIABLE, 배열 rc 1·파일 없음 |
| PASS 기록 출처(Job 산출물 → 사람이 커밋) | 회수본과 다른 바이트 커밋 | gate는 자기 일관성(파일·.sha256·HEAD)만, 출처 미검사. 등록 매개변수가 아니라 위조 행위라 반전 대상 아님 → **caveat** | `preflight_gate`에 회수 `SHA256SUMS` 대조 없음, 대신 `job_name`·`judge_json_sha256`이 사후 대조 수단 |
| xcheck `n_logprobs == 32` | 엔진이 31·33·0개 | 인용 가능 방향 반전 없음(FAIL → 그 config X 미정의) | M24·A5 KILLED, 엔진 코드 읽기상 32 |
| POST 선택(필수 env) | 더 오래된 POST | (iii)가 POST..HEAD 7경로 diff 빈 출력 요구 → 바뀌었으면 거짓, 안 바뀌었으면 같은 바이트라 PASS 정당 | A9·A10 KILLED, `mg_plan.py:735-741` |
| gate PASS 파일 sha256 == .sha256 | 검사 제거(A14) | 현 코드 맞음. record가 둘을 한 번에 써 정직한 흐름에서 불일치 없음. 반전 수치 없음, 시험 공백 공시 | A14 생존, 수리 이전 검사·R-H 범위 밖 |

## 5. 死因·공시

- **死因 0.** 2회차라 처방 없음. 공시·봉인 문서 정리(재감사 대상 아님): (1) §19-2:147 (c) → "요청마다 정확히 32개 output-token logprob, 전부 유한", §19-2:145·§19-16:352 "(미도착)" 정리 (2) `preflight_judge.pass_record`는 JSON 배열 judge.json에서 crash — fail-closed라 무해 (3) A14 공백: 커밋된 .json·.sha256 상호 불일치를 시험이 고정하지 않음(현 코드는 거부). 봉인 전 시험 추가는 engine-porter 재량, 이 판정서의 조건 아님.
- **R-H7 봉인 전제 그대로**: `-Base` 토크나이저 sha 대조, 토큰 파일 업로드, merge 트리 digest 재실측 → 최종 POST → preflight 1회. 위 (1)·(3)처럼 `tests/`·`scripts/multigpu`(gate 경로)를 건드리는 변경은 반드시 최종 POST 전에. 그 뒤에 바꾸면 (iii)이 거짓.

## 6. 승계 caveat (인용 금지 문장, 문자 그대로)

- **MGH12-7** "M3 제출 gate (i)은 커밋된 `PREFLIGHT_PASS.json`과 `.sha256`이 HEAD 객체와 바이트 단위로 같고 서로 일치하는지만 검사한다. 그 바이트가 회수된 preflight Job 산출물과 같다는 것(출처)은 gate가 아니라 회수 `SHA256SUMS`·`job_name`·`judge_json_sha256` 사후 대조로만 보장된다."
- **MGH12-8** "R-H5의 'logprob 정확히 32개' 요구는 SGLang `/generate`가 출력 토큰마다 logprob 1개를 돌려준다는 엔진 코드 읽기에 기댄다. 실측은 첫 M3 xcheck 단계가 처음이며, 어긋나면 그 config의 X 추정량이 미정의가 될 뿐 판정을 인용 가능 쪽으로 바꾸지 않는다."
- **MGH12-9** "Job 레이아웃 gate 시험은 `job_entry.sh` 1단계(bundle clone → checkout → results 복사 → symlink)를 로컬 git 2.34.1로 재현한 것이다. VESSL 컨테이너의 git 판에서 같은 결과가 나온다는 것은 첫 M3 Job의 `PREFLIGHT_GATE.txt`로 처음 확인된다."
- 1회차 MGH12-1–6, VR1 MG12-1–10(MG12-6 비적용 유지), VR2 MG12-4′·12–15, rev11 MGP·MGH 전부 승계.

## 7. 허용·금지 문안

허용: "rev12 하네스 수리 `5dc5fa8b`는 하네스층 감사 2회차에서 死因 0으로 재봉인 가능 판정을 받았다. R-H1(Job symlink 레이아웃에서 성립하는 커밋 대조)과 R-H2(preflight Job 안에서 `PREFLIGHT_PASS.json`을 직접 작성)는 Job 레이아웃 재현 시험과 변이 G1·G2·G4로 고정됐다." · "1회차 생존 변이 7개(M1·M12·M13·M20–M23)와 logprob 부재 변이 M24는 2회차 재실행에서 모두 KILLED였고, `tests.test_mg_rev12` 79건이 통과했다(skip 1)."

금지: "M3 제출 gate가 VESSL Job에서 동작함을 확인했다"(MGH12-9, 첫 M3 Job `PREFLIGHT_GATE.txt` 전) · "preflight PASS가 M3 실행 바이트를 보증한다"(MGH12-7) · "SGLang이 32개 logprob을 반환함을 확인했다"(MGH12-8) · "M3 rev12 봉인 완료"·"M3 제출 가능"(R-H7·최종 POST·preflight 1회 전) · "하네스층 감사가 결과의 정확성을 보였다"(배선만 다룸). 1회차 금지 문안 중 "하네스층 감사 통과"·"재봉인 가능"은 이 판정으로 해제, 나머지 유지.

## 8. 종결

하네스층 감사 상한 2회 도달로 종결. 이후 바뀐 바이트(봉인 문서 정리, A14 시험 추가 시 그것)는 새 대상 — 필요하면 규칙층 감사나 봉인 diff 확인으로 따로.
