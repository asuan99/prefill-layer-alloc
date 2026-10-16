# 판정서: M3 rev12 하네스층 감사 1회차 (2026-10-16)

> claims-auditor(읽기 전용) 최종 보고를 main 세션이 저장(2026-10-16). 하네스층 감사 상한 2회 중 1회차.
> 대상: PREREG r2+(HEAD `9b0e75e`), 하네스 unsealed `6321c8f4` + `0b541c74`, 공용 loadgen `0eca947`, preflight 도구(`results/handoff_review_2026-10-15/merge_trial_tools/`), VESSL 배포(`scripts/vessl/launch.sh`, `job_entry.sh`).
> 저장소 변경 0. 작업은 scratchpad만: `repo/`(`git archive 9b0e75ef`), `mut/run_mut{2,3,4}.py`(변이 실행기, 원복 후 `git show` diff로 깨끗함 확인), `pg/`(Job 레이아웃 gate 재현), `gt/`(git symlink 재현). sbatch 구동 시험 미실행(`M3JobScript`는 `SGLANG_*` 제거 + dry-run이라 실행).

## 1. 총평

| 대상 | 판정 | 판정 반전 死因 | 실행 차단 결함 |
|---|---|---|---|
| rev12 하네스 `6321c8f4`+`0b541c74` | **수리 후 재봉인** | 0 | **2건(R-H1·R-H2)** — 둘 다 fail-closed라 결과 오염은 없으나 이대로면 M3 Job이 VESSL에서 **항상 abort** |

- 사용자 변경 규칙(D-1′ 합산 1%, U7 위치·범위, U12 그룹, LO·발송 지연)은 코드가 문서대로 구현(시험·변이 확인).
- 판정 경로 중 **시험이 고정하지 않은 곳 7곳** — 감사자 변이 생존. 현 코드는 맞으나(기능 재현) 교훈 53에 따라 재봉인 전 시험 필요.
- 반전 시험은 H-2·H-3에서 수행, 반전 못 만들어 caveat로.

## 2. 재실행 결과

- `tests.test_mg_rev12`: 61 OK, skip 1(`MG_TEST_X_TOKENS` 미설정).
- `test_mg_harness`·`test_m2g_gate`·`test_loadgen_outcome`·`test_mg_symlink_layout`: 294, 실패 1(`test_r5_launch_dry_run…` "not a git repository" — archive 사본이라 `.git` 없음, 환경 아티팩트).
- **감사자 변이 23개**(실제 소스 한 곳 치환 + 전체 `test_mg_rev12`; 무변이 M0 통과):
  - KILLED 16: M2 NaN 무시, M3 `NOT_RUN`→PASS, M4 xcheck fit 제거, M5 발송 지연 무시, M6 gate 깨끗함 검사 제거, M7 LO에 rev11 규칙, M9 X KV 한계를 A 값으로, M10 아무 config 기록이나 PASS, M11 동률을 서버로, M14 floor 외삽, M15 이미지 무시, M16 D0 없이 D1만, M17 기록 없는 config가 N1 판정 승계, M18 지연 문턱 5 s, M19 `FLOOR_JOB_EFFECT` 미발화.
  - **생존 7**: M1(출력 32토큰 길이 검사 제거), M12(`DTYPE_UNVERIFIED` 공시 누락), M13(floor 부분 집합 중앙값), M20(`ref_x`를 shape A 셀로), M21(`anchor_x`를 shape A 셀로), M22(X 추세가 shape A Job 효과·앵커를 읽음), M23(X 보정이 A 기준 배열).
  - M8(J1 기준 `x_correctness` 무시)도 생존했으나 **등가 변이**: J1 xcheck 불통과 → N1 X 추정량 미정의 → `g1_def("G_sym","X")` 거짓 → P9-a로 먼저 UNTESTED(`mg_label.py:1362` 부근). 기능 재현 확인.
- M22 기능 재현: X 앵커 G=1.2·A 앵커 G=1.0일 때 `G_sym|DP|X` = `TREND_JOB_SENSITIVE`(FIRED N=2,4,8), `G_sym|DP|A` = `NO_TREND_DETECTED`(CLEAR) — 현 코드 맞음. M22 변이에서는 X가 `NO_TREND_DETECTED`로 **인용 가능**(fail-open)인데 막는 시험이 없음.

## 3. H-1 – H-9

- **H-1 (U7 불통과 → Job 효과): CONFIRMED(코드) / 문서 불일치 1, 死因 아님.** `mg_label.py:867-879`(`x_correctness_reason`·`xcheck_of`, 기록 없음 → `NOT_RUN`), `:1012-1014`(그 config X만 미정의), `:1052`·`:1377`(J1 N1 불통과 → X 기준 불가). 시험 `test_fail_and_not_run_void_x_only`, `test_scope_is_the_failing_mode_only`, X1·X5, 감사자 M3·M10·M17 KILLED. J1 불통과 재현 → `G_sym|DP|X`·`|TP|X` = `G1_UNDEFINED_NO_JOB_CHECK`, A·B 불변. 불일치: PREREG §19-2:145·§19-13 U7은 "J_N non-PASS → `JOB_EFFECT_UNTESTED(J_N, X)` 강제", 코드와 :148(r2++)은 묶지 않음. `trend()`(`:772-809`)에서 UNTESTED는 보정을 끄고 raw만 남기므로 문서 문언대로면 FIRED 보정이 사라져 **덜 보수적** → 코드가 fail-closed 끝, 문서만 수정(R-H6).
- **H-2 (LO 서버 측 오류 = 기술값): PLAUSIBLE → caveat.** `mg_label.py:432-440` 문서대로, `test_lo_cell…`·M7 KILLED. 반전 시험: TTFT 없는 서버 측 오류는 Q_c에서 빠져(`:75-78`) 오류 몰린 arm의 LO g↓ → `KNEE_BRACKETED` 쉬워짐(정의 여부만 fail-open). 반전 못 만든 이유: LO_SEEDS ⊂ HI_SEEDS·같은 π_s prefix라 LO 문서 ⊂ HI 문서, 문서 결정적 오류는 HI 합산 1%에 다시 걸림(n_LO ≈ 0.375·n_HI(λ≈prior)면 LO 오류 ≈2.7% 초과 시 HI 무효), 그 이하면 HI 1% 이하로 VR2 상한(+2.21%) 안, G 크기는 HI achieved. MG12-14에 MGH12-5 보강.
- **H-3 (성공 요청 발송 지연 > 0.5 s = 오류): PLAUSIBLE → caveat, 이중 계산 있으나 무해.** `mg_label.py:417-430`, M5·M18 KILLED. rev11 `HARNESS_LIMITED` = send-lag p99 > **5 ms**(`mg_const.py:109`) — 셀의 1%를 넘는 요청이 > 500 ms 늦으면 p99 ≥ 500 ms라 이미 `HARNESS_LIMITED`(rev11에서도 짝 미정의). 재현(1000요청): 늦은 11건 → `CLIENT_ERROR`/p99 600 ms/limited/rev11 미정의 · 늦은 10건 → OK/6 ms/limited/미정의 · 늦은 5 + 서버 오류 6 → `SERVER_ERROR_HI`/0 ms/not limited/rev11 OK(정의). 새 무효는 지연+오류 합 > 1% 셀에서만, 방향 fail-closed. arm 의존 지연은 대칭 무효라 G 부호 못 만듦. 부수: `CLIENT_ERROR` 셀은 확장 trigger 불가(`:604`), rev11 `HARNESS_LIMITED`(OK)는 가능했음 — 반전 수치 못 만듦, 공시만.
- **H-4 (U12 P10 그룹): CONFIRMED.** `P10_GROUPS`(`:894`), `p_dp_groups`, `P_DP` = AB 그룹 + `groups`, P11(b) 그룹 안 절단(`:1402-1424`). P1·P2·P3 KILLED. "세 shape 일관" 생성기 없음(grep 0) — 결과 문서 금지 문안 몫.
- **H-5 (dtype 전달 공백): PLAUSIBLE — 공백 실재·판정 무해·시험 없음.** `mg_boot.py:329` `check_server_info_knobs`의 `unverified`는 `boot.json`에만 — 인자 echo(F-4)라 실효값 무관. 실효 증거 `ssm_state`는 `kv_dtype_flags`(`mg_label.py:1633`) → `DTYPE_UNVERIFIED` 공시(`:977`). 허용오차(`M2G_SERVER_REFERENCE.json` 실측): N1 `ssm_state` 6.46 GB/48 슬롯 ≈ 0.135 GB/슬롯 = 0.011 GB의 12배 → 슬롯 1개 차도 검출, bf16이면 ≈3.23 GB로 검출, 슬롯 수는 G5(c)(i)가 별도. **M12 생존**. 결과 문서는 MG12-15 상시 게재.
- **H-6 (`.git`·preflight 형식·이미지): REFUTED — 실행 차단, fail-closed.** `.git`은 있음: `launch.sh:104` `git bundle create … HEAD`(전체 이력, 로컬 `is-shallow-repository = false`), `job_entry.sh:114-117` bundle clone·checkout → POST가 HEAD 조상이면 (iii) commit 대 commit diff 성립. **그러나 (i)은 Job 안에서 항상 거짓**: `job_entry.sh:121-124`가 `results/`를 `/data/runs/<job>/results`로 복사 후 **symlink로 교체** → git은 symlink 너머 추적 파일을 삭제(D)로 봄. 재현 1(`gt`, git 2.34.1): `ls-files --error-unmatch` rc 0, `diff --quiet HEAD -- results/…` **rc 1**. 재현 2(`pg`, 실제 `mg_plan.preflight_gate`): 일반 레이아웃 `(True, [])`, Job 레이아웃 `(False, ['(i) PREFLIGHT_PASS.json not tracked/clean', '(i) …sha256 not tracked/clean'])` → `m3_job.sh:37` exit 19 `ABORT_PREFLIGHT_NOT_PASSED`. **같은 결함이 rev11 승계 F17 `check_committed`(`m3_job.sh:24-26`)에도**: M2G 표 `.sha256`이 `results/multigpu`에 있어 exit 15 `ABORT_M2G_NOT_PASSED`. DRYRUN에서 건너뛰고 시험(`PreflightGate._repo`, `tests/test_mg_rev12.py:986`)도 일반 레이아웃만 → 한 번도 실행된 적 없음. 정황: `mg_job_common.sh:99`가 이미 `':!workspace/engine-port/results'`로 제외. **preflight 도구는 등록 형식으로 쓰지 않음**: `preflight_judge.py:105-132`는 `judge.json`에 `label`(≠`verdict`), `post_commit`·`merge_tree_engine_digest`·`image` 없음. `preflight_cl3.sh:30` POST 기본 `dc06c870`, `:136-137` `post_vs_job_head` 5경로, `preflight_cl3_targets.txt`에 `test_loadgen_outcome` 없음. `PREFLIGHT_PASS.json` 작성 주체 미정 → 수기 전사 자유 표면. 이미지 (ii)는 CONFIRMED(digest 없으면 거짓, `mg_plan.py:705-707`, M15 KILLED; `launch.sh:41`이 digest 고정 강제).
- **H-7 (U7 변이): PLAUSIBLE — 생존 1, 공백 1.** "N=1에서만" X4, "D0 없이 D1" M16, "NaN 무시" M2, "fit 누락" M4, "`NOT_RUN`→PASS" M3 KILLED. **"출력 32토큰 미만 PASS" M1 생존**(`mg_xcheck.py:116`; 세 패스 모두 같은 31토큰이면 D0 = D1 = 4, 첫 16 일치 → PASS). **NaN 검사 공백**: `mg_xcheck.py:74-77`은 `n_logprobs` 기록만, logprob 0개 응답은 `nonfinite = 0` → PASS(재현). SGLang dev tree는 `logprob_start_len == len(input)` 허용(`scheduler.py:1917`은 `>`일 때만 abort)이라 실제로는 logprob이 올 가능성이 높으나 검사는 항등식에 가까움(교훈 9). 격리: xcheck 출력은 `boots/<id>/xcheck.json`에만, `cells/` 밖 → 셀·λ\*에 섞이지 않음.
- **H-8 (합산 1%): CONFIRMED(규칙), U14·F-5 배선 시험 공백.** `mg_label.py:417-440`, `analyze_cell :460`, 앵커·J1 기준·확장 셀이 같은 `analyze_cell`(rung "HI"). E1–E4·J3, M5·M11 KILLED, 발송 지연은 합산 분자에(M5). **M20–M23 생존**: XJobEffect 시험이 `make_docs`로 `ref_x`·`anchor_x`를 직접 주입해 `aggregate`(`:1052-1054`)·`aggregate_jobs`(`:1377`, `:1401`) 배선을 고정 못 함.
- **H-9 (floor): PLAUSIBLE / 일부 NOT-YET.** `floor_T`(`:1545`, M14), `job_floor`(`:1563`, F1), `FLOOR_JOB_EFFECT`(M19) KILLED. `floor_summary`(`mg_boot.py:804`)는 fail-closed로 맞으나 **M13 생존**(PREREG §19-4가 [코드] 근거로 인용). §19-9 분석기 미존재 → 대조 시험 불가, 분석기 커밋(fetch 전) 의무로 남음.

## 4. 문서와 다르게 동작하는 곳 (전수)

| # | 위치 | 차이 | 판정 의미 |
|---|---|---|---|
| D1 | PREREG §19-2:145·§19-13 U7 vs 코드·§19-2:148 | J_N non-PASS의 `JOB_EFFECT_UNTESTED` 강제 여부 | 코드가 보수적 끝, 문서 수정 |
| D2 | §19-6 (i) "`.git` 가정" | `.git`은 있으나 `results/` symlink로 (i)·F17 항상 거짓 | 실행 차단(R-H1) |
| D3 | §19-6 (i) 형식 vs preflight 도구 | 형식 불일치·작성 주체 미정 | 실행 차단(R-H2) |
| D4 | `mg_label` 머리 "M2g rung = rev11 규칙" | M2g rung에도 발송 지연 합산 | M2g 재실행 없음 → 무관, 문구 정정 |
| D5 | §19-2 U7 "logprob 전부 유한" | logprob 부재 시 vacuous PASS | 보강(R-H5) |
| D6 | `mg_plan.py:218`, `mg_drive.py:14` docstring | "앵커 boot에 xcheck", 옛 `gates/XCHECK.json` | 표기만 |
| D7 | §19-2 `-Base` 토크나이저 대조, 토큰 파일 업로드, digest 재실측 | 미실행(문서 자인) | 봉인·제출 전 필수 |

그 밖 셀 560, floor 블록 20, xcheck 14, 예산(`job-table` J1 5.88·J2 13.22·J4 13.15·J8 12.40 h — §19-10 r2++와 일치), `ANCHOR_SHAPES`, `CONTEXT_LENGTH` 20480, 노브 3개, `ABORT_LOG_KNOB`(`job_entry.sh:299-301` 형식 일치)은 문서와 같음.

## 5. 필수 수리 (재봉인 전, 死因 아님)

- **R-H1 (차단).** `mg_plan.preflight_gate` (i) "tracked/clean"과 `m3_job.sh` `check_committed`를 `results/` symlink 레이아웃에서도 성립하는 비교로(예: `git cat-file -e HEAD:<rel>` + `git show HEAD:<rel>` sha256 == 파일 바이트 sha256). job_entry 레이아웃(clone → results 복사 → symlink) 시험 추가: 유효 PASS → True, 변조 → False. 변이 "옛 `diff --quiet HEAD`로 되돌림" KILLED.
- **R-H2 (차단, 도구).** preflight 도구가 Job 안에서 `PREFLIGHT_PASS.json` = `{verdict, post_commit, merge_tree_engine_digest, image}`(+권고 `job_name`, `judge_json_sha256`)를 직접 작성. `verdict`는 judge `label`이 `PREFLIGHT_PASS`일 때만, FAIL_NEW·UNVERIFIABLE은 절대 PASS로 안 씀. `image` = `PDMUX_IMAGE`. `merge_tree_engine_digest`는 `ENGINE_DIGEST.txt`와 같은 절차. POST env 필수화(`dc06c870` 기본값 제거), `post_vs_job_head` 7경로, targets에 `test_loadgen_outcome`. selftest: 도구 산출물이 일반·Job 레이아웃 모두에서 `preflight_gate` 통과.
- **R-H3 (시험, U14·F-5 배선).** `aggregate` → `aggregate_jobs` 끝까지 타는 시험으로 M20–M23 KILLED(X 앵커 ≠ A 앵커 값 조건).
- **R-H4 (시험).** M1(출력 32토큰 미만), M12(`DTYPE_UNVERIFIED`가 추정량 공시까지 전달), M13(`floor_summary` 성공 < 16 → `floor_s` None) 시험.
- **R-H5 (xcheck 보강).** 요청마다 `n_logprobs == XC_MAX_NEW` 요구, 아니면 FAIL. 변이 "logprob 부재 PASS" KILLED.
- **R-H6 (문서·표기).** D1(§19-2:145, §19-13 U7을 :148에 맞춤), D4, D6 정정. §19-6 `.git` 문장 → "bundle은 전체 이력이고, symlink 레이아웃 때문에 R-H1 수리가 필요했다".
- **R-H7 (봉인 전제, 문서 자인분).** `-Base` 토크나이저 sha 대조, 토큰 파일 업로드, merge 트리 digest 재실측 → 최종 POST → preflight 1회(F-1). **R-H1·R-H3·R-H4·R-H5는 gate 경로(`tests/`·`scripts/multigpu`)를 건드리므로 반드시 preflight 전에.**

## 6. 승계 caveat (인용 금지 문장, 문자 그대로)

- **MGH12-1** "U7 xcheck는 fused\_sym과 agnostic만 대조한다. d44의 shape X 경로(G\_d44sym의 X)는 정확성 단계 밖이며, U7 PASS를 d44의 16k 정확성으로 쓰지 않는다."
- **MGH12-2** "U7 PASS는 두 arm이 서로 일치하고 같은 boot 안 잡음 기준선 대비 저하가 없음을 보일 뿐이다. 두 arm에 공통인 16k(TP 포함) 오작동은 검출하지 않는다(절대 정답 대조 없음)."
- **MGH12-3** "J\_N의 N=1 X 앵커는 그 Job의 U7 단계에 묶이지 않는다. J\_N xcheck 불통과는 그 config의 X 추정량만 미정의로 만들고 `JOB_EFFECT(J_N, X)` 판정은 바꾸지 않는다."
- **MGH12-4** "성공 요청의 발송 지연 > 0.5 s 규칙은 rev11 `HARNESS_LIMITED`(send-lag p99 > 5 ms)에 대부분 포함된다. rev11 대비 새 무효는 지연과 오류가 합쳐 셀 요청의 1%를 넘는 셀에서만 생기며 방향은 무효 증가뿐이다."
- **MGH12-5** (MG12-14 보강) "LO 셀 서버 측 오류의 상한 해제가 G를 기울이지 않는다는 논증은 LO 문서가 같은 seed의 HI 문서 prefix에 포함되어 문서 결정적 오류가 HI 합산 1%에 다시 걸린다는 구조에 기댄다. 이 논증은 모형 수준이며 부하 의존 오류의 arm 편중은 측정되지 않았다."
- **MGH12-6** "`DTYPE_UNVERIFIED`가 없다는 것은 부팅 로그 `ssm_state` 크기가 M2g와 0.011 GB 안에서 같다는 뜻일 뿐, 실효 dtype의 직접 측정이 아니다."
- VR1 MG12-1–10(MG12-6 비적용 표시 유지), VR2 MG12-4′·12–15, rev11 MGP·MGH 전부 승계.

## 7. 허용·금지 문안

허용: "rev12 하네스 체크포인트(`6321c8f4`+`0b541c74`)는 D-1′(a) 합산 1%, U7 config 단위 범위, U12 P10 그룹, floor fail-closed를 CPU 시험 61건으로 구현했다. 하네스층 감사 1회차에서 감사자 변이 23개 중 16개가 KILLED였고, 생존 7개(등가 변이 1개 별도)는 수리 대상이다." · "M3 제출 gate는 fail-closed로 설계되었으나, 1회차 감사 시점에는 VESSL Job 레이아웃에서 항상 거짓이었다(R-H1)."

금지: "M3 제출 gate가 VESSL Job에서 동작함을 확인했다"(R-H1 수리·Job 레이아웃 시험 전) · "preflight PASS가 rev12 하네스(`results/multigpu/`)를 검증했다"(승계) · "U7 xcheck가 16k TP 경로의 정확성을 보였다"(MGH12-2), "d44의 X 정확성"(MGH12-1) · "발송 지연 규칙이 rev11 A·B 셀의 유효성을 바꾸지 않는다" · "하네스층 감사 통과"/"재봉인 가능"(R-H1–R-H5 전) · 이번 재현 수치(lag p99 600·6·0 ms, 슬롯당 ≈0.135 GB, n_LO ≈ 0.375 n_HI, 2.7%)를 서빙 측정값으로 인용.

## 8. 2회차(마지막) 범위

R-H1–R-H6 수리 커밋 diff만: (a) Job 레이아웃 gate 시험과 그 변이 KILLED (b) 도구 산출물이 두 레이아웃에서 `preflight_gate` 통과 (c) M1·M12·M13·M20–M23·logprob 부재 변이 KILLED + 대조군 통과 (d) 문서 D1·D4·D6 정정. 2회차에도 死因이 남으면 "사용자 결정 필요"로 종료.
