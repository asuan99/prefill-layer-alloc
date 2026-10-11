# 판정서 — N2(3-a) rev17 좁은 재감사 (감사 뒤 변경분 (a)–(i) + 하네스 수리 H17-1/2/3/6) (2026-10-11)

> 감사자: claims-auditor (읽기 전용). 저장: 메인 세션(2026-10-11), 본문은 감사 보고 그대로(서식만 정리).
> 대상: `workspace/engine-port/results/n2_static_grid/` 작업 트리(미커밋), PREREG 감사 시점 sha256 = `31150489f461999613b5daad7e1d8cb1f60488fe7d760218d02593244ee1a8b5`. 기준 = HEAD(`c35d9a6`, 규칙 봉인 `fcc0b9e` digest `fbb13b15…` + 하네스 `1037655`)와 직전 판정서 `VERDICT_n2_3a_rev17_delta_harness_2026-10-11.md`(감사 바이트 `0dbdfffb…`, 사본 없음).
> 직접 실행: 규칙 시험 2종, `test_n2_job`, `n2_reach_tests`, `test_n2_harness`, 해시 대조, 코어 장부 재집계. 변이 러너는 돌리지 않았다. 새 성능 판정 0건(등록 감사).

## 0. 결론 요약

| 대상 | 등급 | 死因 |
|---|---|---|
| **규칙 봉인 갱신(rev17)** | **`GO-with-caveats`** — 조건: R17-4·R17-5(문언만)를 봉인 갱신 커밋에 넣는다. 그 커밋의 PREREG diff가 현재 바이트(`31150489…`) 기준으로 §3에 적은 위치만 건드리면 재감사 없이 GO 유지 | **0** |
| **하네스 봉인(PROFILE)** | **아직 불가(보류)** — H17-1/2/3/6은 등록 문언과 일치; 남은 선행 조건은 §2-4(H17-7, 신규 H17-8) | **0** |

- 판정 경로(K·창·라벨·예산·한도) 불변: 계산기·축약기·`N2_REGISTERED.json` sha가 봉인값과 같다.
- 반전 시험: 라벨·부호 반전 0건. 예보 F-N2-7P만 움직이며 N2C-18이 이미 인용을 막는다.
- 새 결함: 낡은 수치 문언 3곳(§15 하네스 행 "35개", §18-2-a "JH(76)", §18-2-d #7 행 중복 열) → §21-31 자기검사가 다시 거짓(R15-2/R17-1 계열 3회째); §14-2 G-10 본문에 Job 적용 지시 없음; N2C-77 미등재 근거 문장 부정확; 러너 비정상 종료 시 `stopped=None` → F-N2-7P 거짓 `FALSIFIED` fail-open(H17-8).

## 1. 규칙 쪽

### 1-1. 바뀐 규칙 봉인 파일 대조
| 파일 | 현재 sha256 | 기대값 | 결과 |
|---|---|---|---|
| `PREREG_N2_3A_2026-10-07.md` | `31150489…` | 변경 허용 | 변경됨 |
| `test_n2_probe_reduce.py` | `a796980c…ca171` | 직전 감사 바이트 `a796980c…` | 같음 |
| `n2_mutants.py` | `ffb4d6f3…a8d` | 직전 감사 바이트 `ffb4d6f3…` | 같음 |
| `n2_feasibility.py` | `1ee3338d…0c60` | 봉인값 | 같음 |
| `n2_probe_reduce.py` | `8ef0c8cc…282d` | 봉인값 | 같음 |
| `N2_REGISTERED.json` | `e7d743c8…87e7` | 봉인값 | 같음 |
| `test_n2_feasibility.py` | git 미변경 | `331c7b0f…` | 같음 |

바뀐 규칙 봉인 파일은 셋뿐(확인). 규칙 시험 실측 143/143 OK, skip 0(N3 포함, 저장소 제자리).

### 1-2. (h) `@@CORE17@@` 채움 ↔ 원본 장부
원본 장부(scratchpad `r17/mut_core.txt` = `mut_core_rev17_final.txt` 바이트 동일) 재집계: 194행(CTRL·EXIT 제외) — J 85 = KILLED 77 + EQUIVALENT 8, T 12 = KILLED 12, S 97 = KILLED 96 + EQUIVALENT 1; 예상 밖 생존·BROKEN·TIMEOUT 0; **J28·J29 KILLED**(각 failures=1). CTRL `OK (skipped=1)`, 모든 변이 행 skipped=1 → R17-3 조건 충족. 시작 = `prerun_sha.txt` mtime 18:03:10.19, 두 파일 mtime 15:41:17.27(앞섬), 종료 19:34:06. `prerun_sha.txt`의 입력 sha는 `a796980c…`·`ffb4d6f3…`·계산기·축약기 봉인값; 그 시점 PREREG sha `997a35ec…`이나 코어 장부는 PREREG를 읽지 않는다(grep 확인) — 영향 없음. → §21-30 "도구·파일" 행 기록은 사실과 일치.

### 1-3. (a)–(i) 문언 대조
| 항목 | 판정 |
|---|---|
| (a) R17-1 | 일치(`test_n2_job` 실측 53/57, 실패 4건이 공시 목록과 같음). 단 §15 하네스 행 셀에 현행형 "35개"가 남아 §21-31 "35는 이력에만"이 여전히 거짓 → R17-4 |
| (b) R17-2 | 직전 처방과 일치(PROFILE 선행 조건에 H17-3(#16) 추가는 사용자 채택 귀결로 정합) |
| (c) R17-3 | 일치 |
| (d) 사용자 결정 | 일치. 단 (i) #7 행이 열 5개(`§14-2 G-10, §14-3` 중복) (ii) §14-2 G-10 본문은 여전히 "VD 중단 뒤에는 더 이상 boot를 돌리지 않는다"이고 Job 적용 예외를 가리키지 않는다 — 단일 출처 원칙(교훈 85·89) 위반 → R17-4 |
| (e) #8·#10·#17 | 코드와 일치. 단 #10 "`stopped ∉ {None, "budget"}`"은 러너 비정상 종료가 `None`을 남기는 경로를 열어 둔다 → R17-5·H17-8 |
| (f) N2C-76·78·79 / N2C-77 | 76·78·79는 직전 판정서와 문자 그대로 같다; N2C-77은 §1-5 |
| (g) Job 57·reach 37·JH 93 | 57·37 실측 일치(reach 37 OK, harness 36 OK). §18-2-a 표 "JH(76)" vs 860행·§21-30 "JH 93" 불일치 → R17-4 |
| (h) §21-30 + 코어 장부 | §1-2대로 일치 |
| (i) n2_plan digest 문언 | `RULE_SEAL_DIGEST` = `fbb13b15…`, `RULE_FILES` = rev16 값 그대로 — 문언과 일치 |

감사 뒤 PREREG 변경은 문언(§18-2·§18-2-a/b/d·§20-1·§21-30/31·§15/§14-2 재현 문장)·하위 사유(#8)·예보 조작화(#10)·abort 규칙(#16)·Job 진행 정책(#7)뿐이다. K·창·술어·점/D/dsens 라벨 규칙·예산·한도 문장은 diff 밖(hunk 대조) — 불변.

### 1-4. 반전 시험 표(감사 뒤 새 자유 표면)
원자료: 등록 단위 `n2_plan --units PROFILE` = 28 boot(7종 × 4 round), `n2_analyze.MIN_N = 4`, boot 부대비용 170.5/229 s, 하드캡 36,700 s; 서버 로그 형식 근거 = N3 4회차 `n3b_pdmux-n3-capacity-93c3cca-20261007064124/srv_*.log`(같은 config·25항목 manifest) — `Decode batch` 줄 3,337개, `cuda graph: False` 0개, 형식 `LOG_TS` 일치; `decode_log_interval` 기본 40.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| G-10 Job 적용(#7: 정지↔계속) | 두 끝 × 결함 빈도 {boot 1개 VD, 모든 boot 첫 셀 VD} | 라벨·부호·순위 반전 0 — round 정확히 4·MIN_N 4라 VD로 잃은 복제가 있으면 그 셀은 `UNRESOLVED`, 선택된 부분집합 경로 없음; "계속"은 "정지"의 VALID 복제 상위집합만 낸다. F-N2-7P만: 모든 boot VD일 때 정지판 `UNRESOLVED`, 계속판 28/28 입장 → `HELD`(공허) — 예보·판정 입력 아님, N2C-18이 막음 → 死因 아님 | 모든 boot 첫 셀 VD면 ≥ 28 × 170.5–229 s = 4,774–6,412 s + preflight, 상한 = 승인 하드캡 36,700 s |
| CG 카나리아 경계(#16) | `[ceil, floor−1]` ±1 s; "전부"↔"하나라도" | 반전 0 — 카나리아는 판정 입력 아님(`cg_evidence`·`CG_UNPROVEN` 불변); 거짓 abort는 "라벨 없음"으로만(fail-closed) | warm-up 8요청 순차 × 64토큰 = decode 512회 ⇒ 로그 줄 ≈12–13, 지속 ≥ ≈6 s; 경계 절삭 ≤ 2 s ⇒ 남는 줄 ≥ ≈8. 형식은 N3 로그로 일치 확인(sticky ON 조건은 미확인) |
| 카나리아 첫 boot abort(형식 불일치) | 일치↔불일치 | rev16 = Job 끝 전 셀 `CG_UNPROVEN`(라벨 0), rev17 = 첫 boot abort(라벨 0) — 판정 같음, 비용만 감소 | 위 |
| #8 output_ids 부재 | 부재↔존재 | `BYTE_IDENTICAL` → `UNRESOLVED` 방향만(fail-closed) | 로컬 dev 트리 `tokenizer_manager.py:1620-1623`은 non-streaming 응답에 `output_ids`를 싣는다(등록 VESSL 트리 동일성 미확인) |
| #17 전 절 현재 바이트 대조 | 요구 절만↔전 절 | fail-closed, 판정 무관 | — |
| #10 F-N2-7P `stopped` 집합 | `{None, budget}`↔러너 예외 `None` | 예보만: `HELD`/`UNRESOLVED` → 거짓 `FALSIFIED`; N2C-18이 막음 → 死因 아님, 수리 R17-5/H17-8 | §2-2 |

→ N1–N4 해당 없음, 死因 0. 새 추정량 없음.

### 1-5. (f) N2C-77 미등재 검증
미등재는 정당(반증 실패), 단 근거 문장이 부정확(R17-5). 원안 두 조항 — (1) 예산 외 정지 시 F-N2-7P를 예산 예보 성패로 인용 금지 (2) 그 정지로 n<4가 된 라벨을 "효과 없음"으로 쓰지 않음 — 의 전제("VD → Job 정지")는 #7로 거짓. (1)은 N2C-18이, (2)는 N2C-11이 덮는다. 그러나 기존 주의 "하드캡·`ABORT_*` 정지는 H17-2가 막는다"는 (i) 하드캡은 예산 사유(감시자 SIGTERM → `runner_stop` → `stopped="budget"` → 정당한 `FALSIFIED` 가능; 3층 `timeout -k` SIGKILL이면 원장 미기록 → `UNRESOLVED`) (ii) H17-2는 `job_abort`만 막고 러너 비정상 종료(`stopped=None`)는 못 막음 — 두 점에서 틀렸다. #7로 새 위험(공허한 `HELD`)이 생겼고 이것도 N2C-18이 덮는다.

## 2. 하네스 쪽

### 2-1. 등록 문언 ↔ 구현
| 등록 | 구현 | 일치 | fail-open |
|---|---|---|---|
| #4/#8 H17-1 | `_has_ids` + `sticky_correctness` 양쪽 전 출력 검사 → `UNRESOLVED`(why "output_ids absent"); 러너 기록, abort 아님; 빈 출력만 exit 9(기존) | 예 | 없음 |
| #7 G-10 계속 | `runner_stop` → `stopped="budget"`, 그 밖 boot abort(VI·VD) 뒤 다음 boot; 남은 셀 `not_attempted_boot_aborted`; JH34·JH35 양방향 | 예 | 없음 |
| #10 H17-2 | `stopped ∉ {None,"budget"}` ∨ top-level `job_abort` ∨ 원장 행 `job_abort` → `UNRESOLVED`; JobAbort 시 `stopped="job_abort"` | 예 | **있음(H17-8)**: `run_job`은 `except JobAbort`만 잡는다; `run_boot` 안쪽 try 밖(`server.start`·`settle`·`open(...,"x")` 등)이나 `run_job` 자체 예외는 `_BootScope.__exit__`가 억제하지 않아 올라가고, `finally`가 `stopped=None, job_abort=None`으로 원장을 쓰며 미시도 unit은 행이 없다; `n2_capacity.sh`는 `RUNNER_EXIT *`에서도 분석 → F-N2-7P 입장 < 28 → 거짓 `FALSIFIED` |
| #16 H17-3 | `warmup_cg_canary`(`pair_to_wall`, `[ceil(w0), floor(w1)−1]`, ≥1 ∧ 전부 True) → `JobAbort(3,"ABORT_CG_CANARY")`; `cg_evidence` 불변; 시험 5 + JH83–87 | 예(TZ = boot 기록 `tz_gmtoff_s`, 러너가 항상 설정) | 없음 |
| #17/H17-6 | `parse_harness_sections`·`verify_harness_seal(scope)`(이중·형식 오류 BROKEN, 요구 절 전부 현재 바이트·현행 규칙 digest, 결측 UNSEALED); `n2_capacity.sh` PROFILE `--scope PROFILE`, EA는 `--scope EA` 검사 뒤 `ABORT_EA_NOT_IMPLEMENTED`; JH88–93 | 예 | 없음 |

`to_wall` → `pair_to_wall` 리팩터는 등가(HEAD 대조). 단위 시험: `test_n2_job` 53/57(실패 4건 = 공시 목록, 원인 `verify_rule_seal` sha 불일치 — 의도), `n2_reach_tests` 37 OK, `test_n2_harness` 36 OK.

### 2-2. 신규 필수 수리 H17-8
`run_job`에 `except BaseException:` 분기(JobAbort 뒤): `stopped = stopped or "crash"` 후 재발생 → F-N2-7P는 등록 문언대로 `UNRESOLVED`. 시험: 가짜 서버 `start()` OSError → `JOB_LEDGER.stopped == "crash"`, F-N2-7P `UNRESOLVED`. 변이: 분기 삭제 → 사살. "crash"는 새 값이므로 R17-5로 #10에 등록.

### 2-3. 가짜 엔진 e2e OBSERVER_FAIL
관측자 경로 코드는 이번 diff에서 바뀌지 않았다(함수 AST 대조: `n2_analyze` 변경 = `_has_ids`·`collect`(docstring)·`forecasts`·`sticky_correctness`; `n2_window` = `pair_to_wall`·`to_wall`(등가)·`warmup_cg_canary`; `observer`·`measurement_ok`·`obs_client_window`·`obs_replicate`·`paired_ci`·`cg_evidence` 불변). 입력 쪽 차이 3가지: (1) 가짜 모드 `PDMUX_N2_TEST_WALL_SCALE` 기본 20 → 200 (2) G-10 계속(HEAD e2e가 전부 VALID면 해당 없음) (3) C-itl은 클라이언트 단조 시각(CPU 타이밍)이라 작은 효과도 FAIL(교훈 275). 확정 실험: 같은 WALL_SCALE로 HEAD 사본과 수정 트리 각 ≥ 2회, `observer.cells[*].x`·`mean`·`ci` 비교 — 반복 사이에서도 뒤집히면 CPU 잡음.
> 메인 세션 주(2026-10-11): HEAD 사본을 WALL_SCALE 200으로 1회 돌린 결과도 `PROFILE_COMPLETE` 60/60·`OBSERVER_FAIL` — 이번 수정의 회귀가 아님. 반복 비교(≥ 2회)는 미실시.

### 2-4. 하네스 봉인(PROFILE) 남은 선행 조건
1. R17-4·R17-5 반영 → 규칙 봉인 갱신 커밋(`REGISTRATION_SHA256.txt` 규칙 절 새 digest).
2. H17-7: `n2_plan.RULE_SEAL_DIGEST`·`RULE_FILES` 갱신 → `test_n2_job` 57/57; `n2_capacity.sh` 머리 주석 "rev16 … fbb13b15…" 갱신(서술).
3. H17-8 구현 + 시험 + JH94.
4. 새 규칙 digest 위에서 재실행: 규칙 시험 143(skip 0), `test_n2_harness` 36, `test_n2_job` 57(+H17-8), `n2_reach_tests` 37, `n2_mutants.py --set harness`(H01–H32), `n2_job_mutants.py` JH 전부(93 + H17-8), CTRL skip 수 기록.
5. §2-3 OBSERVER_FAIL 원인 분리(봉인 조건 아님, dry-run 전 권고).
6. 하네스층 델타 재감사 → `scope PROFILE` 하네스 봉인 커밋.

## 3. 필수 수리·caveat·결론

### 규칙(문언만, 봉인 갱신 커밋에 포함)
- **R17-4**: (i) §15 하네스 행 "35개" → "36개(rev17; rev16 35)" (ii) §18-2-a `n2_job_mutants.py` "JH(76)" → "JH(93)" (iii) #7 행 중복 셀 `| §14-2 G-10, §14-3 |` 하나 삭제 (iv) §14-2 G-10 끝에 "(Job 적용은 §18-2-d #7 — VD 중단 뒤에도 다음 boot 진행; 이 문장의 'VD 뒤 중지'는 smoke 러너에만 적용)" (v) §21-31 시험 수 문장 재대조. (선택) §18-2-a `n2_marks` "확장 예정"은 낡은 표현. 이 위치 밖 바이트를 바꾸면 GO 무효.
- **R17-5**: (i) N2C-77 미등재 주 근거 교체 — "남는 인용 위험은 N2C-18(입장 수·결측 boot 수를 예산 예보 성패로 인용 금지)과 N2C-11(결측 → `UNRESOLVED`는 발견이 아님)이 덮는다. 하드캡 정지는 예산 사유(`stopped = "budget"`)이고 `ABORT_*` 정지는 H17-2가, 러너 비정상 종료는 H17-8이 F-N2-7P를 `UNRESOLVED`로 만든다." (ii) #10 F-N2-7P 끝에 "러너가 `JobAbort` 아닌 예외로 끝나면 `stopped = "crash"`(H17-8)".

### 하네스
- **H17-8**(§2-2), H17-7(직전 판정서 그대로).

### 신규 caveat(문자 그대로)
- **N2C-80**: "`ABORT_CG_CANARY`로 끝난 Job은 cuda graph 불성립의 증거가 아니다 — 카나리아는 warm-up(b = 1, 순차 8요청) 구간의 로그 형식·경계 규약에 의존하는 비용 보호 장치이며, 셀의 cuda graph 성립은 `cg_evidence`로만 판정한다; 반대로 카나리아 통과를 셀의 `cuda graph: True` 증거로 인용하지 않는다."
- **N2C-81**: "§18-2-d #7(VD 중단 뒤에도 Job 계속) 아래에서 F-N2-7P의 `HELD`는 boot가 VD로 일찍 끝나 공허하게 성립할 수 있다 — `HELD`를 등록 예산 모형이 맞았다는 근거로 인용하지 않는다(N2C-18과 같은 금지의 #7 귀결)."

### 사용자 결정 필요
없음 — #7·#16·H17-4/5 미채택은 이미 결정됐고, 반전 시험이 그 결정을 뒤집을 수치를 내지 못했다.

### 결론
- **규칙 봉인 갱신: `GO-with-caveats`, 死因 0.** "판정 경로 불변"은 반증 실패(sha 일치, 판정 코드·판정 문장 diff 0, 반전 0건). 남은 결함은 문언 정합(R17-4)과 근거 문장(R17-5)뿐.
- **하네스 봉인(PROFILE): 보류, 死因 0.** H17-1/2/3/6은 등록과 일치하고 fail-open 없음; H17-2 인접 경로에 러너 비정상 종료 fail-open(H17-8)이 남아 있다.
- 이 GO는 규칙 봉인 갱신 승인이다. 하네스 봉인이나 dry-run 승인이 아니다.

### 확인한 파일
`PREREG_N2_3A_2026-10-07.md`(1–4, 244, 314, 495, 758, 819–823, 854–860, 940–1063, 1096, 1106, 1183–1190, 1668–1696행), `VERDICT_n2_3a_rev17_delta_harness_2026-10-11.md`, `REGISTRATION_SHA256.txt`, `n2_job.py`, `n2_window.py`, `n2_analyze.py`, `n2_plan.py`, `n2_capacity.sh`, `fake_n2_engine.py`, `n2_marks.py`, `n2_job_mutants.py`, `test_n2_job.py`, scratchpad `r17/{mut_core.txt, mut_core_rev17_final.txt, prerun_sha.txt}`, N3 4회차 `srv_*.log`, 로컬 dev 엔진 트리 `tokenizer_manager.py:1600-1645`·`scheduler.py:2620-2629`·`scheduler_metrics_mixin.py:600-660`·`server_args.py:418`(참고 — 등록 VESSL 트리 동일성 미확인).
