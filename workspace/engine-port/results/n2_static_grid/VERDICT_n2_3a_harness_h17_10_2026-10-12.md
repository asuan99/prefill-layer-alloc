# 판정서 — N2(3-a) 하네스 봉인(PROFILE) 재발행 델타 감사 H17-10 (2026-10-12)

> 감사자: claims-auditor (읽기 전용; 단위 시험 3건만 실행, 변이 러너 미실행). 저장: 메인 세션, 본문은 감사 보고 그대로(서식만 정리).
> 대상: 미커밋 작업 트리(`git diff HEAD -- .`: n2_plan.py, n2_realized_mix.py, n2_capacity.sh, test_n2_job.py, n2_job_mutants.py, REGISTRATION_SHA256.txt). 직전 판정 `VERDICT_n2_3a_harness_profile_seal_2026-10-11.md`(봉인 `97a91b5`).
> 계기: Job `pdmux-n2-static-grid-52a3b97-20261011182211` preflight exit 4 `ABORT_PREFLIGHT`, 셀 측정 0.

## 0. 요약 판정
| 질문 | 판정 | 死因 |
|---|---|---|
| 1. H17-10 정확성·범위 | 정확. 같은 계열 잠복 결함은 Job-PROFILE 경로에 0; Job 경로 밖 3건(비차단); `PDMUX_PROJECT_ROOT`가 테스트 기대를 오염시키는 경로 없음 | 0 |
| 2. N2C-78 문언·규칙 재개봉 | 결론은 이제 사실과 맞으나 "원자료가 git 밖이라"는 기전 서술이 N3 쪽에서 틀렸다 → N2C-84. 규칙 재개봉 불필요(규칙 바이트 불변, 판정 경로 무관, 반전 0) | 0 |
| 3. 하네스 봉인(PROFILE) 재발행 | **`GO-with-caveats`**(N2C-84–86, N2C-78·80–83 승계). §3 조건 1–5 정확 충족 시 재감사 불필요. 필수 코드 수리 없음(H17-11 없음). 조건 4(봉인 상태 종단 preflight)는 필수 봉인 조건 | 0 |
| 4. 첫 실패 Job 기록 | §4 문언 | — |

## 1. H17-10 — 정확성과 범위
### 1-1. 기전 재확인
회수 `logs/target.log`: REG_OK·`SEALED scope=PROFILE`·EXTERNAL_DIGESTS_OK·CONFIG_OK·B0_OK 뒤 preflight 실패 — `test_n2_job.py:19 import n2_analyze` → `n2_analyze.py:59 _load(ENG/benchmarks/pdmux_eval/analyze.py)` → `FileNotFoundError: /data/runs/<job>/benchmarks/...`. 레이아웃은 `job_entry.sh:103-105`(results 전체를 `${RUN}/results`로 복사 후 심볼릭 링크), `PDMUX_PROJECT_ROOT` export(:110). 실측: 스크립트 실행 시 `sys.path[0]`은 물리 경로, `sys.path.insert(0, dirname(abspath(__file__)))` 뒤에는 논리 경로 — `python3 "$HERE/n2_job.py"`·`python3 "$HERE/n2_analyze.py"`는 수리 전에도 무사, `cd "$HERE" && python3 -m unittest`만 물리 경로 import. 진단과 일치.

### 1-2. 같은 계열 잔존 위치(전수)
| 위치 | 성격 | Job-PROFILE 경로? | 판정 |
|---|---|---|---|
| `n2_plan.ENG`, `n2_realized_mix.ENG` | env 우선(H17-10) | 예 | 수리됨 |
| `n2_analyze.ENG = n2_plan.ENG` | 승계 | 예 | 수리됨 |
| `n2_job.py:476/502` cfg = `n2_plan.ENG/CONFIG_REL` | 승계 | 예 | 수리됨; 서버 `Popen` cwd 미지정(`cd "$ROOT"` 상속), 텔레메트리 경로 `$OUT` 절대경로 |
| `n2_window.REDUCER_FILE = HERE/n2_probe_reduce.py` | 같은 디렉터리 | 예 | 물리·논리 같은 바이트, 무해 |
| `HERE/../n3_capacity/{n3_capacity.sh,n3_plan.py,configs}` | 형제 디렉터리 | preflight | git 추적·results 전체 복사로 물리 경로에도 존재, 무해 |
| `n2_probe_run.ENG`(:49) | 미수리 | 아니오(`main()`만, 시험이 부르지 않음; Workspace 스크립트는 논리 경로) | 비차단 잔존 |
| `n2_job_mutants.ENG`(:23) | 파일 상대 | 아니오(로컬 장부 전용) | 비차단 |
| `test_n2_job.REPO`(:28) | 사용처 0 | — | 사문 |
| λ0 `tel_*.jsonl`(git-ignored) | **파일 존재**로 판정 → MISSING → skip | preflight | 올바르게 skip |
| N3 `n3b_…38ed3f4…` | 규칙 시험이 **`isdir`**로 판정(추적 파일 367개로 참) → FAIL | preflight | H17-10 명시 override로 skip; 그 밖에 N3_DIR 읽는 시험 없음 |

결론: Job-PROFILE 경로(preflight → `n2_job.py` → `ProcessServer` → `n2_analyze.py`)의 잔존 위치 0.

### 1-3. `PDMUX_PROJECT_ROOT` ↔ 테스트 기대
`sh()`(:775)와 신규 두 시험은 `PDMUX_*`를 제거 후 명시 주입(Job의 `PDMUX_COMMIT` 누출 없음); :156은 env 전체 상속이나 `--verify-harness-seal`만 호출(ENG 무관). 프로세스 안 `P.ENG` 사용은 `CFG`(:31)·`job_layout` 심볼릭 링크 대상(:770)뿐, HERE 경로와 문자열 비교 0건. 잘못된 root → ENG 측 3파일(config·analyze.py·profile.py)이 `external_digest_problems`(n2_plan:613)로 digest 고정 → `ABORT_REG_DRIFT` 또는 FileNotFound(fail-closed). n2_* 하네스 파일은 HERE에서 읽히고 봉인으로 검사.

### 1-4. 실행한 확인
`env -u PDMUX_PROJECT_ROOT`로 신규 두 시험 + `TestRealizedMix.test_positive_and_negative_control` 3/3 OK(1.9 s); 실행 환경 `PDMUX_*` 0개.

### 1-5. 시험의 약점(조건 4로 메움)
JH97 사살 시험은 `n2_capacity.sh`를 정규식 매칭 후 emulation — preflight 블록 자체를 실행하지 않는다. 이 하네스에서 preflight 블록(SKIP 아님)을 심볼릭 링크 레이아웃에서 종단 실행한 적이 없다(기존 `test_dryrun_in_symlinked_job_layout`은 SKIP_PREFLIGHT, 메인 세션 시뮬레이션은 unittest 직접 호출). 첫 Job은 import 단계에서 죽어 5 suite 어느 것도 컨테이너(Python 3.14)에서 실행된 적이 없다(로컬 3.13.9). fail-closed 위험일 뿐 死因 아님 → 조건 4 + N2C-86.

## 2. N2C-78과 규칙 재개봉
N2C-78은 PREREG rev17 본문(:1190, 규칙 봉인 바이트). "…원자료가 git 밖이라 Job 트리에서 skip된다": λ0는 사실(파일 존재 판정), N3는 수리 전 거짓(`isdir` 참 → FAIL), 수리 후 결과는 참이나 기전은 하네스 명시 override. 효력 부분("`PREFLIGHT_TESTS.txt` 통과를 양성 대조 통과로 인용하지 않는다", "근거는 로컬 봉인 장부(skip 0)뿐")은 그대로 성립. 재개봉 불필요: 규칙 7파일 바이트 불변(REGISTRATION diff는 하네스 절·장부 블록 제거만), override 기제(`N2_N3_DIR` env)는 이미 봉인된 `test_n2_probe_reduce.py:14`에 있음, 집행은 preflight(fail-closed 관문)에만 작용.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| `N2_N3_DIR` override 값 | 부재 ↔ JSONL 있는 실재 ↔ 빈 추적 디렉터리 | 없음(skip / 대조 실행 / FAIL→exit 4, 모두 셀 이전) | `n2_capacity.sh:157-160`, `test_n2_probe_reduce.py:271` |
| ENG 출처(env ↔ 파일 상대 ↔ 다른 체크아웃) | 세 경우 | 없음(다른 바이트 `ABORT_REG_DRIFT`, 부재 FileNotFound) | `n2_plan.py:613-615` |
| preflight 허용 skip 수 | 0 ↔ 2 ↔ 임의 | 조건 4에서 정확히 2건·이름 지정으로 고정(조이는 방향, 교훈 251) | §3 |

반전 0건.
- **N2C-84**: "N2C-78의 '원자료가 git 밖이라 skip된다'는 N3 θ 양성 대조에 대해서는 기전 서술이 틀렸다 — Job 클론에는 N3 원자료 디렉터리가 추적 파일과 함께 존재하므로 규칙 시험은 skip이 아니라 FAIL하며, 하네스 봉인 rev(H17-10)부터는 `n2_capacity.sh`가 `N2_N3_DIR`을 부재 경로로 지정해 **명시적으로** skip시킨다(`PREFLIGHT_ENV.txt`에 기록). λ0 대조의 skip은 원자료 부재 감지에 의한 것이다. 어느 쪽도 Job 안의 양성 대조 통과로 인용하지 않는다."

## 3. 하네스 봉인(PROFILE) 재발행 — `GO-with-caveats`
死因 0(N1 측정 판정량 불변, N2 반전 0, N3 예측·정의역 불변, N4 비운영점 정책 결론 없음). 필수 코드 수리 없음. 조건 1–5 정확 충족 시 재감사 없이 재봉인, 그 밖 바이트 변경 시 재감사.
1. **장부**: 하네스 16파일 = 현 작업 트리 바이트에서 H01–H32 CTRL OK(skipped=1)·32/32 KILLED; JH01–JH97 CTRL OK·97/97 KILLED·미사살 0·TIMEOUT/BROKEN/.pyc 0; 입력 최신 mtime < 시작; prerun sha 재대조 공백; **장부 셸의 `PDMUX_PROJECT_ROOT`·`N2_N3_DIR` 값(또는 unset)을 provenance에 기록**.
2. **제자리 suite**: 5 suite 276/276, skip 0(규칙 143 skip 0 포함), `PDMUX_PROJECT_ROOT` unset.
3. **기록 수치**: provenance에 `test_n2_job` 60(58+2), JH01–JH97. N2C-82의 "정본 수치는 하네스 기록"은 이 값으로 갱신 승계. 규칙 절·rule_seal_digest `fc9e6404…` 불변.
4. **봉인 상태 종단 preflight(필수)**: 봉인 커밋 뒤 그 커밋의 추적 파일만 있는 클론(`git clone`/`worktree`)에서 `workspace/engine-port/results`를 별도 디렉터리로 옮기고 심볼릭 링크로 바꾼 상태로 `PDMUX_PROJECT_ROOT=<클론> PDMUX_N2_JOB=PROFILE PDMUX_N2_DRYRUN=1 /usr/bin/bash .../n2_capacity.sh`를 `PDMUX_N2_TEST_SKIP_PREFLIGHT` **없이** 실행. 기대: exit 0; `HARNESS_SEAL_CHECK.txt`에 `SEALED scope=PROFILE`; `PREFLIGHT_ENV.txt` 존재; `PREFLIGHT_TESTS.txt`가 `OK (skipped=2)`이고 skip이 정확히 `test_positive_control_n3`와 `TestRealizedMix.test_positive_and_negative_control`.
5. 커밋 뒤 `launch.sh` dry-run → 사용자 승인 → `--submit`; Job 커밋의 하네스 바이트 = 봉인 바이트.
- **N2C-85**: "Job `pdmux-n2-static-grid-52a3b97-20261011182211`의 preflight 산출물(REG_OK, SEALED, EXTERNAL_DIGESTS_OK, CONFIG_OK, B0_OK, ENGINE_TREE_CHECK, SUBSTRATE_N2)은 그 Job, 그 노드의 관문 통과 기록일 뿐이다 — 다음 Job의 기판·엔진 트리 근거로 인용하지 않는다."
- **N2C-86**: "로컬 장부와 로컬 Job 레이아웃 시뮬레이션(Python 3.13)의 통과는 컨테이너(Python 3.14, 심볼릭 링크 결과 볼륨)에서의 preflight 통과를 함의하지 않는다 — Job 안 `PREFLIGHT_TESTS.txt` 통과는 하네스가 그 컨테이너에서 import·실행된다는 사실만 보이며, 측정·양성 대조·관측자 거동의 근거로 인용하지 않는다."
- N2C-78(+N2C-84)·N2C-80–83 승계. 비차단 권고(`n2_probe_run.ENG` env 우선 또는 Workspace 전용 주석, `test_n2_job.REPO` 삭제)는 하네스 바이트를 바꾸므로 장부 실행 전에만.
> 메인 세션 주(2026-10-12): 장부는 감사 전에 이미 시작됐으므로(03:40:55 KST) 비차단 권고는 이번 봉인에 반영하지 않는다(다음 하네스 회차로 이월).

## 4. 첫 실패 Job 기록 문언
> Job `pdmux-n2-static-grid-52a3b97-20261011182211`(2026-10-11T18:22:18Z 제출, commit `52a3b97`, 하네스 바이트 = 봉인 `97a91b5`, resourcespec-a100x1): **preflight exit 4 `ABORT_PREFLIGHT`, 셀 측정 0, 서버 boot 0, 약 1분.** 원인은 하네스 결함(H17-10) — `cd $HERE && python3 -m unittest`가 cwd(물리 경로)로 모듈을 import해 `n2_plan.ENG`가 클론 밖(`/data/runs/<job>`)을 가리켰다(교훈 270 재발). 같은 수리 범위에서 두 번째 잠복 결함(N3 원자료 디렉터리의 JSONL 부재 → 규칙 시험 FAIL)도 발견. **이 Job은 측정이 아니고 결과도 아니다.** 어떤 라벨·수치·기판 근거로도 인용하지 않는다(N2C-85). 회수 산출물(`vessl/<job>/` DONE+SHA256SUMS)은 실패 기록으로만 보존. 데이터 0이라 다음 Job과의 선택 문제 없음. 다음 Job을 이 캠페인의 첫 측정 시도로 센다. 이 Job의 GPU 시간은 별도 행으로 적되 측정 예산에 산입하지 않는다.
