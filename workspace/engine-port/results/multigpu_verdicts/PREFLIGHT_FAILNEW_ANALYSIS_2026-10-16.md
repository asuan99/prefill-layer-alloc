# PREFLIGHT_FAIL_NEW 사유 분류 — `pdmux-preflight-cl3-68f386b-20261016114954` (2026-10-16, result-analyst)

> 지위: 분석 기록, 커밋하지 않음. 측정 아님(정확성 프로브의 사유 분류). **판정은 등록된 대로 `PREFLIGHT_FAIL_NEW`이고 바꾸지 않는다.**
> 이 문서는 라벨을 다시 매기지 않는다. 사유마다 사실 분류만 한다. 판정기·시험·하네스·추적 파일은 수정하지 않았다.
> 근거: 회수본 `results/preflight_cl3/pdmux-preflight-cl3-68f386b-20261016114954/`(logs 4개, judge.json, meta/), `results/preflight_cl3/vessl/<job>/meta/`,
> 판정기 `handoff_review_2026-10-15/merge_trial_tools/{preflight_judge.py,sigcmp.py,preflight_cl3_l2_ids.txt,preflight_cl3_targets.txt,pin_pre.json,pin_post.json}`,
> 로컬 시험 merge 원본 로그(scratchpad `crr1/logs/{pre.main,postA,postB}.log`), git 객체 `7b15214b`(PRE)·`e43749c9`(POST)·`e44e2e09`(로컬 시험 merge 대상).
> 진단용 스크립트는 전부 scratchpad에 있다(`pf/parsed.json`, `pf/lenient_lib.py`, `pf/cf.py`). 판정기를 고친 것이 아니라 읽기 전용으로 import해 돌렸다.

## 0. 한 줄 결론

**신규 실패 2건 가운데 진짜 merge 회귀로 읽을 여지가 남는 것은 없다.** (a)는 등록된 L2 원인과 같다. 판정기가 PRE의 `ok`를 읽지 못해 NEW로 분류됐다.
(b)는 환경 의존 시험이다. 컨테이너의 `HF_HOME`이 시험 env 필터를 통과해 새어 들어갔다. 로컬에서 반사실(counterfactual) 실행으로 양방향 모두 재현했다.
시그니처 변경 9건 중 8건은 마스킹 결함이다. arm 이름(`/pre/`·`/post/`)이 경로에 남았다. 원인은 양쪽 모두 아카이브 부재(FileNotFoundError)로 같다.
나머지 1건은 L2다. PRE 쪽은 컨테이너 타이밍 실패가 가리고 있다. post에서 해소된 17건은 PRE runner가 이미지 conda `activate`에 인자를 넘겨 실패했고, POST runner의 `shift 2`가 그 인자를 우연히 없앴다.
HOLE 1건은 사람이 읽으면 네 반복 모두 `ok`이다.
**단, "회귀 0건"이라고 쓰지 않는다(C-4).** 이 런에서는 L2·타이밍·아카이브 부재가 가려서 전혀 검증되지 않은 시험 묶음이 있다(§5-2).

## 1. 조건 확인 (짝 구조·기판)

| 항목 | 값 | 확인 |
|---|---|---|
| PRE / POST | `7b15214b` / `e43749c9`(multigpu 브랜치 끝, `merge-base --is-ancestor PRE POST` 참) | provenance.txt + 로컬 git |
| 로컬 시험 merge 대상(L2 목록의 출처) | `e44e2e09`(POST보다 57커밋 앞, POST의 조상) | **L2 목록은 POST와 다른 트리에서 나왔다.** 이 런에서는 L2 밖 실패가 전부 설명되므로(§2–§3) 목록을 늘릴 근거는 없다 |
| 반복 | ABBA 2회. pre rep1=rep2, post rep1=rep2(실패 id·서명 일치, `invalid: []`) | judge.json |
| 수집 | pre Ran 989(failures 26, errors 6, skipped 41) / post Ran 1487(failures 20, errors 6, skipped 42) | 헤더 줄 수 = Ran 수(989/1487). 실패 블록 수 = failures+errors(32/26) |
| 기판 | **두 arm이 한 컨테이너**(`resourcespec-a100cpu`, GPU 없음, cgroup `cpu.max=350000/100000` = 3.5 vCPU, nproc 48), 이미지 `1e0d335b…` | arm 사이에는 기판 confound가 없다. 그러나 설계 §1은 이 CPU 스펙을 **기각**하고 A100x1을 채택했었다. 실제 런은 설계와 다른 스펙에서 돌았다(§4) |
| cgroup 스로틀 | 종료 시 `nr_throttled 16498 / nr_periods 93542`(17.6 %), CPU 사용 13,503 s / 경과 9,730 s(평균 ≈1.39코어) | `vessl/<job>/meta/cgroup_cpu.txt` |
| 스위트 소요 | pre 1,940/1,977 s(989건), post 2,862/2,873 s(1487건). 로컬 pre는 970 s(877건) | 건당 시간이 로컬의 약 1.8배 |

## 2. 신규 실패 2건

### 2-(a) `test_n3_capacity.TestRegistration.test_plan_selftest`: L2와 같은 원인이고, NEW 분류는 판정기 파싱 결함이다

- **원인은 L2와 같다(사실).** POST 서명이 로컬 `pin_post.json`의 이 id 서명과 **모든 필드에서 일치**한다(kind FAIL, AssertionError, `mirror_check`, msg/tail 동일).
  실패 위치는 `n3_plan.py:932 assert mirror_check(eng) == []`이다. 4개 엔진 파일이 등록 sha와 다르다: chunk `2d1d8cff…≠7dfa76a9…`, holb `5a58c809…≠97cd7392…`, mixin `c2483a14…≠ba5835aa…`, telemetry `d34ce5d0…≠e4399835…`.
  이 4쌍은 known_failures v2 §2가 L2 원인으로 적은 것과 같다.
- **로컬 시험 merge의 L2 목록에 이미 있었다.** 이 id는 `preflight_cl3_l2_ids.txt` 11번째 줄에 있고, known_failures v2 §2 표 3행, `pin_post.json`에도 있다. 로컬에서도 실제로 돌았다(`pre.main.log:755 ... ok`, `postA.log:1823 ... FAIL`). `pin_pre.json`에는 없다(PRE에서 통과).
  즉 질문의 전제("로컬 L2 목록에 없었다")는 사실이 아니다. 로컬에서 다른 이유로 실패한 것도, 돌지 않은 것도 아니다.
- **NEW로 분류된 이유(사실).** `preflight_judge.judge`는 L2 소속을 인정하려면 `pst.get(tid) == 'ok'`, 즉 PRE 상태가 파싱돼 있어야 한다. PRE 로그 862–863행은 다음과 같다.
  ```
  test_plan_selftest (test_n3_capacity.TestRegistration.test_plan_selftest) ... N3 PLAN SELFTEST OK (seeds = ...)
  ok
  ```
  selftest의 stdout 출력이 `... `와 `ok` 사이에 끼었다. `sigcmp.parse`의 상태 정규식은 상태 토큰이 `... `와 **같은 줄 끝**에 있어야 잡는다(docstring 한 줄만 사이에 허용). 그래서 PRE 상태가 `None`이 됐고 L2 hit에서 빠져 NEW가 됐다.
- **로컬에서는 같은 일이 없었던 이유(사실).** 로컬 실행에서는 stdout이 블록 버퍼링돼 이 출력이 로그 끝(`pre.main.log:1811`)에 몰렸다. 컨테이너는 `job_entry.sh:309`가 `PYTHONUNBUFFERED=1`을 export한다. `preflight_cl3.sh`는 `PDMUX_*`만 걷어 내므로 이 값이 시험까지 상속되고, 출력이 제자리에 끼어든다.
- 같은 구조로 상태를 읽지 못한 시험: pre 21건, post 50건. 관대한 재스캔(다음 헤더 전 첫 상태 토큰)으로 보면 **전부 `ok`**이고, 그중 실패 블록이 있는 것은 0이다. L2 id 중 이 구조에 걸린 것은 `test_plan_selftest` 하나다.
- 잔여 공백: POST에서는 `selftest()`가 932행에서 멈춘다. 그 뒤 단계(floor 공식, 10개 무효화 대조 등)는 POST 트리에서 검증되지 않았다. 이는 L2의 기지 한계다(known_failures §7 (ii)).

### 2-(b) `test_mg_harness.JobScriptsDryRun.test_m3_dry_run_every_job`: 환경 의존이고, PRE에는 없는 시험이다

- **PRE 트리에는 없다(사실).** `7b15214b`에는 `tests/test_mg_harness.py`가 없다(PRE 로그의 `(test_mg_harness.` 0건, POST 176건). 그래서 짝 대조가 불가능하다. POST에서 실패하면 정의상 NEW다.
- **환경 의존(사실, 코드 + 반사실 실행).**
  - 시험은 `test_mg_harness.py:2489`에서 env를 `PDMUX_`·`SGLANG_`·`NCCL_` 접두만 걷어 내고 `PDMUX_DATA=<td>/data`(빈 디렉터리)를 준다. 그리고 `:2510–2511`에서 `ABORT_MODEL_REVISION`이 DRYRUN_NOTES에 **있어야** 한다고 단정한다(주석 "would abort a real run on this machine").
  - `mg_job_common.sh:51`은 `HF_HOME="${HF_HOME:-$DATA/hf}"`다. 상속된 `HF_HOME`이 있으면 그것을 쓴다. `:147–149`는 `$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/refs/main`이 등록 revision(`mg_const MODEL_REVISION = dc0661c8…`)과 같을 때만 노트를 생략한다.
  - 컨테이너는 `job_entry.sh:309`가 `HF_HOME=/data/hf`를 export한다. `/data/hf`에 Nemotron(과 Zamba2)이 캐시돼 있다는 간접 증거가 있다. 로컬에서 skip·실패하던 `test_r2_correctness_ctx.TestCachedModelsResolveAsExpected` 두 건이 컨테이너에서는 양 arm 모두 `ok`였다.
    노트에 `ABORT_MODEL_REVISION`이 없다는 사실 자체가 refs/main이 등록 revision과 일치했다는 직접 증거다.
  - **반사실(로컬 CPU, 이 세션, HEAD `68f386b`; `results/multigpu`와 `tests/`는 POST와 바이트 동일, `git diff --stat e43749c9 HEAD`가 빈 출력):**
    A) `HF_HOME` unset → `Ran 1 test … OK`.
    B) `HF_HOME=<가짜 캐시, refs/main = dc0661c8…>` → `AssertionError: 'ABORT_MODEL_REVISION' not found in 'DRYRUN_NOTE ABORT_SUBSTRATE …'`. 컨테이너 실패와 같은 메시지다.
    GPU 유무와 무관하게 `HF_HOME` 한 변수로 양방향이 재현된다.
  - 로컬 시험 merge(`e44e2e09`)에서는 `postA.log:806 ... ok`였다. 로컬 `HF_HOME`이 unset이라 `$td/data/hf`가 비어 있었기 때문이다. `pin_post.json`에도 없다.
- **CPU 기판 탓이 아니다.** GPU 검사 실패(`ABORT_SUBSTRATE`)는 시험이 오히려 **있어야 한다**고 요구하는 항목이고 실제로 있었다. A100x1 GPU 컨테이너였어도 `HF_HOME=/data/hf`는 같으므로 같은 실패가 났을 것이다(추론. 근거는 반사실 B가 GPU 없이 재현된다는 점).
- 잔여 공백: 시험이 J1 첫 반복의 `assertIn`에서 멈췄으므로, 컨테이너에서는 J1의 나머지 단정과 J2/J4/J8가 확인되지 않았다. 그 부분은 반사실 A(로컬, POST와 같은 바이트)에서 전부 통과했다.

## 3. 시그니처 변경 9건 · post에서 해소된 17건

### 3-1. 시그니처 변경 9건 (pre·post 모두 실패하지만 서명이 다름)

| id | PRE 서명 | POST 서명 | 다른 필드 | 분류 |
|---|---|---|---|---|
| test_lambda0_prereg.TestArrivalRealization.test_analyzer_records_the_drain_decomposition | ERROR FileNotFoundError `…/pf_work/pre/…/results/longctx_conflict/probes/c_N/bench_d44_r2_o96.jsonl` | 같은 예외, `…/pf_work/post/…` | msg·tail의 arm 경로 | 마스킹 결함 + 환경(아카이브 부재) |
| …TestArrivalRealization.test_realized_denominator_removes_the_artifact_on_archived_cells | ERROR FNF `…/pre/…/bench_d16_r0_o96.jsonl` | `…/post/…` 동일 | arm 경로 | 같음 |
| …TestDrainIdentity.test_drain_identity_on_archived_cells | ERROR FNF `…/pre/…/bench_d16_r0_o96.jsonl` | `…/post/…` 동일 | arm 경로 | 같음 |
| …TestShippedSelftests.test_lambda_inf_predicate_selftest | FAIL AssertionError `F1 recount artifact missing: …/pre/…/r2_correctness/job_N/srv_warmup.log` | `…/post/…` 동일 | arm 경로 | 같음 |
| test_lambda_star_per_shape.CliSurfaces.test_workloads_cli_optin_writes_the_caveats_into_the_header | FAIL `1 != 0`, tail FNF `no lambda* table at …/pre/…/lambda_star.example.json` | `…/post/…` 동일 | tail의 arm 경로 | 같음(파일이 `.gitignore *.json`이라 번들에 없음, known_failures §3-2) |
| test_lambda_star_per_shape.FailClosed.test_shipped_template_loads_and_is_entirely_unmeasured | ERROR FNF `…/pre/…/lambda_star.example.json` | `…/post/…` 동일 | arm 경로 | 같음 |
| test_n3_capacity.TestBootBranchSnippets.test_t_h1_sticky | ERROR FNF `…/pre/…/n3_capacity/probe_6group_2026-10-05/srv_A2_warm.log` | `…/post/…` 동일 | arm 경로 | 같음 |
| test_n3_capacity.TestBootBranchSnippets.test_t_h3_floor | ERROR FNF `…/pre/…/probe_6group_2026-10-05/tel_A2_warm.jsonl` | `…/post/…` 동일 | arm 경로 | 같음 |
| test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut | FAIL `Regex didn't match: 'fake_rc=(124\|143\|137) unit=floor_head' not found in ''`(:723, rc·wall 단정은 통과) | FAIL `3 != 0`(:721) | msg·tail | **POST = L2**(pin_post와 서명 동일). **PRE = 컨테이너 타이밍**(로컬 pre는 `ok`, `pre.main.log:717`) |

- 8건의 마스킹 결함(사실): `sigcmp.mask`는 `/(tmp|home)/\S+`만 `<path>`로 바꾼다. 컨테이너의 arm 루트 `/data/runs/<job>/pf_work/{pre,post}/`는 숫자만 `N`으로 바뀌고 `pre`/`post`는 남는다. 그래서 원인이 같아도 arm마다 서명이 달라진다.
  같은 원인으로 양 arm 모두 실패하면서 경로가 메시지에 안 나오는 시험(`test_lambda0_prereg.TestMutationHarness.test_no_escapes`, `test_lambda_star_per_shape.CliSurfaces.test_workloads_cli_refuses_the_unmeasured_template`, `…ShellRefusals.test_generate_campaign_refuses_the_unmeasured_template`)은 "same"으로 흡수됐다.
- 8건의 근본 원인은 환경이다(git-ignored 아카이브가 컨테이너에 없음). 로컬 시험 merge에서는 아카이브를 심볼릭 링크로 심어 전부 통과했다(`pre.main.log` 439/574/708 등). `pin_pre`·`pin_post` 어디에도 없다.

### 3-2. post에서 해소된 17건 (= 대상 집합 `C_venv17` 17 id 전부)

| id 집합 | PRE 서명 | POST | 로컬 pin_pre 서명 | 분류 |
|---|---|---|---|---|
| test_campaign_provenance 10 + test_lambda_star_per_shape.SbatchReadsTheField 3 + test_r2_eval_runner.TestSbatchPathResolution 4 | FAIL AssertionError `2 != 0 : Synced PD-mux runtime: …`. 원문: `ArgumentError: activate does not accept more than one argument: ['…/run.json', '…/run_0']`(pre.rep1에 17회, post 0회) | 전부 `ok` | `… line N: … No such file or directory`(로컬에는 `sglang_engine_venv/bin/activate` 자체가 없었음). **로컬과 원인이 다르다** | 환경(이미지 conda) + 브랜치 변경의 부수 효과. 회귀가 아니다 |

- 기전(사실, 코드): venv 모드가 `image_python_prefix`(`/opt/conda`)라서 runner의 `source "${venv_dir}/bin/activate"`가 conda의 `activate`를 부른다. bash의 `source`는 인자를 주지 않으면 스크립트의 위치 인자를 그대로 넘긴다.
  PRE `engine_bench_runner.sh`에는 `shift`가 없어 `run_record`, `run_dir` 두 인자가 conda `activate`에 전달되고 거부된다(rc 2).
  POST runner는 multi-GPU 옵션 파싱을 위해 7행에 `shift 2`를 넣었으므로 115행 `source` 시점에는 인자가 0이다. **해소는 의도한 수정이 아니라 부수 효과다.**
- 판정상 의미: 17건은 POST에서 실제로 실행·통과했다(대상 항목 검증은 성립한다). 그러나 PRE와의 짝 대조는 불가능하다. PRE가 다른 이유로 전부 실패했기 때문이다. 이 17건의 증거는 "POST 절대 통과"이지 "PRE 대비 불변"이 아니다.
  결론 범위에 venv 모드(`image_python_prefix`)가 붙는다(설계 §3).

### 3-3. L2 재현 9/12의 내역 (judge가 따로 출력하지 않는 사실)

| L2 id | PRE | POST | judge 처리 |
|---|---|---|---|
| 9개(head_mixin, config_without_fixed_dsm, unbroken_copy, equivalent_mutant_watchdog, symlinked_results, r2_dryrun, live_repo_digests, resume, runner_env) | ok | FAIL | L2 hit |
| test_plan_selftest | ok(파싱 실패로 `None`) | FAIL(L2 서명) | **NEW**(§2-(a)) |
| test_overrunning_unit_is_cut | FAIL(타이밍) | FAIL(L2) | **CHANGED** |
| test_control_both_kill_layers_removed_overruns | FAIL `38.3 / 39.7 not >= 60`(타이밍) | FAIL `16.4 / 16.1 not >= 60`(L2 drift abort를 컨테이너 속도로 느리게 통과) | **어느 목록에도 안 나온다.** `N.N` 마스킹이 두 수를 같게 만들어 "양쪽 같은 실패"로 흡수됨 |

- `test_unbroken_copy_runs`의 POST 블록에는 DRIFT 줄이 없다. 시험이 `out[-2000:]`만 싣는데, 하위 python 프로세스마다 찍히는 pynvml FutureWarning(14줄)이 꼬리를 채웠다. 원인을 원문으로 볼 수 없다.
  rc 3과, 같은 트리에서 실행된 `test_resume_and_its_controls`·`test_r2_dryrun…`·`test_inherited…`의 DRIFT 5줄(다이제스트 `6b227c54fb22`, 로컬과 동일)로 미루어 L2(`ABORT_REG_DRIFT`)일 가능성이 높다(추론). L2 소속은 id 단위라 판정에는 영향이 없다(설계 §4·§8-6의 기지 한계).

## 4. HOLE 1건 — `test_holb_probe.TestSchedulerHooksInstalled.test_patch_reapplies_the_hooks_and_nothing_else`

- **사람이 읽은 결과: 네 반복(pre1, post1, post2, pre2) 모두 `ok`.** 실패 블록(`FAIL:`/`ERROR:`)도 없다. 원문 구조는 다음과 같다.
  ```
  test_patch_reapplies_the_hooks_and_nothing_else (test_holb_probe.TestSchedulerHooksInstalled.…)
  The mirrored patch must reproduce the installed tree exactly, so the ... …/test_holb_probe.py:667: ResourceWarning: unclosed file <_io.BufferedReader …>
    r = subprocess.run(
  ResourceWarning: Enable tracemalloc to get the object allocation traceback
  (같은 경고 2회 반복)
  ok
  ```
- 놓친 이유(사실): 정규식은 헤더 뒤에 줄 하나(docstring)만 허용하고 상태가 `... `와 같은 줄 끝에 있어야 한다. 여기서는 docstring 줄 끝에 ResourceWarning이 붙고(stderr 끼어쓰기) `ok`가 4줄 아래에 있다. 상태가 `absent`가 됐고 대상 id 단정(`status == 'ok'`)이 실패했다.
  ResourceWarning의 출처는 시험 코드 `test_holb_probe.py:667–668`이다(`stdin=patch_file.open("rb")`가 닫히지 않음). unittest가 경고 필터를 `default`로 켜서 출력됐다.
- 로컬에서는 이 시험이 `sgl_kernel` import 오류로 아예 돌지 않았다(`pre.main.log:391 ERROR`). 그래서 경고 끼어쓰기가 처음 드러난 것이다. 판정기 selftest의 변이 11건에도 이 형태는 없다.

## 5. CPU 기판 영향과 "진짜 회귀" 가능성

### 5-1. CPU라서(또는 이 컨테이너 사양이라서) 다르게 동작한 시험

| 원인 축 | 시험 | 근거 | 등급 |
|---|---|---|---|
| **CPU 사양 + 시작 비용**(3.5 vCPU quota, 17.6 % 기간 스로틀. 이미지 `sitecustomize.py`가 python을 띄울 때마다 `import torch`) | test_e1_cap_vessl.TestBudgetEnforcer.test_overrunning_run_is_cut_at_the_deadline(wall 60.5/62.9 s, 기준 <50, 양 arm) · …test_equivalent_mutant_one_layer_removed_still_cuts(61.3/63.6 s, 양 arm) · test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut(PRE, FAKE_BENCH 비어 있음) · …test_control_both_kill_layers_removed_overruns(PRE 38–40 s) | 로컬에서는 넷 다 `ok`. 같은 코드 경로(L2 drift abort까지)가 로컬 0.89 s, 컨테이너 16.1–16.4 s(약 18배). 판정기 등의 비-torch 스크립트에서도 pynvml 경고가 찍히므로 sitecustomize의 torch import가 확인된다. "deadline ≈ 20 s" 창을 시작 비용이 잠식했다는 설명은 **추론**(창 안의 시간 분해는 로그에 없음) | 환경(사양). 설계 §8-5가 예견한 위험 |
| **GPU 부재** | test_mg_harness의 dry-run 노트 `ABORT_SUBSTRATE`(nvidia-smi 없음, NVML 라이브러리 없음) · n3 dry-run 노트 | 시험이 이 노트가 **있어야** 한다고 요구하고 실제로 있었다. 결과(ok/FAIL)를 바꾼 시험은 0건 | 영향 없음 |
| GPU 부재 중 `sgl_kernel` import | A_import9 9모듈, B_holb4, D_zamba2 | `sgl_kernel ok /opt/conda/…`(양 arm). CPU 로드 경로로 import됐다. **cc 8.0 GPU에서 로드되는 변형으로 검증된 것은 아니다**(결론 범위 한정) | 범위 주석 |
| CPU 무관(환경) | §3-1의 8건 + test_no_escapes + lambda_star 2건(아카이브 부재), skip 41/42건(전부 "git-ignored … absent" 류, GPU 사유 0), §3-2 17건(conda activate), §2-(b)(HF_HOME), §2-(a)·§4(PYTHONUNBUFFERED·ResourceWarning 끼어쓰기) | 로그 원문 | 환경 |

### 5-2. 신규 실패 가운데 진짜 merge 회귀일 가능성 (핵심 질문)

- **(a) 회귀 가능성 없음.** POST 서명이 로컬 L2 핀과 필드 단위로 같다. PRE 원문은 `ok`다.
- **(b) 회귀 가능성 없음.** 로컬 POST-동일 바이트에서 `HF_HOME` 하나로 통과와 실패를 양방향 재현했다. 시험 자체가 "이 기계에는 모델이 없다"는 가정을 담은 비밀폐(non-hermetic) 시험이다.
- **진단용 반사실 재판정**(scratchpad `pf/cf.py`, 판정기를 고치지 않고 import만 함, **라벨 변경이 아님**):
  | 조건 | 라벨 | NEW | CHANGED | HOLE |
  |---|---|---|---|---|
  | 등록 판정기 | FAIL_NEW | 2 | 9 | 1 |
  | + arm 경로 마스킹 | FAIL_NEW | 2 | 1(overrunning) | 1 |
  | + 관대한 상태 파싱 | FAIL_NEW | 1(mg_harness) | 9 | 0 |
  | + 둘 다 | **FAIL_NEW** | 1(mg_harness) | 1(overrunning) | 0 |
  파서 결함 두 개를 고쳐도 라벨은 FAIL_NEW로 남는다. 남는 2건은 각각 환경 의존(HF_HOME)과 타이밍(CPU 사양)이고, 둘 다 회귀 쪽 증거는 없다. 따라서 이 런을 재판정해 PASS로 만들 수 있는 경로는 없고, 그래야 맞다.
- **이 런이 검증하지 못한 영역**(회귀가 숨어 있어도 이 런으로는 보이지 않는 곳):
  1. test_n3_capacity의 예산 강제·재개·abort 분기 로직. POST에서 L2 drift abort가 먼저 발화하고 PRE에서는 타이밍이 깨서 양쪽 모두 미검증이다.
  2. 아카이브 의존 시험 약 53건(실패 11 + skip 41–42). 양 arm에서 돌지 않았다. 로컬은 `e44e2e09`에서 아카이브를 링크해 통과했지만 POST `e43749c9`에서는 확인한 바 없다.
  3. test_e1_cap_vessl.TestBudgetEnforcer 2건. 양 arm에서 타이밍으로 실패했다.
  4. venv 17건의 PRE 기준선.
  이 중 CL-3 대상(31/32 항목)에 속하는 것은 없다. 대상 항목은 HOLE 1건을 사람이 읽으면 32/32 `ok`다.

## 6. cost model 기여

없음(정확성 점검. CM 항에 계수·함수 형태·가정·범위를 더하지 않는다). 사전 등록된 예측값도 없다.

## 7. 다음 preflight 전에 필요한 수정 후보

표기: **gate 경로** = 설계 §6 대조 경로(`src`, `tests`, `scripts/bootstrap`, `scripts/r2_eval`, `env`, `benchmarks/pdmux_eval`, M3는 +`scripts/multigpu`)를 바꾸는가.
변경하면 POST가 바뀌므로 새 POST로 preflight를 다시 해야 한다. **자유 표면** = 이 런의 데이터를 본 뒤 정한 규칙 변경인가. 그렇다면 사전 선언과 감사(claims-auditor)가 필요하다.
판정기·스크립트는 `results/handoff_review_2026-10-15/merge_trial_tools/`에 있어 gate 경로가 아니다. 그래도 gate의 측정기이므로 바꾸면 도구 sha가 바뀌고 결론 범위가 바뀐다.

| # | 후보 | gate 경로 | 자유 표면 | 비고 |
|---|---|---|---|---|
| F1 | **"L2 목록 보완"은 불필요.** 이 런의 L2 밖 실패는 전부 §2–§3으로 설명된다. 목록에 없는 L2성 실패는 0이다. (a)는 이미 목록에 있다 | — | — | 목록을 늘리면 오히려 데이터를 본 뒤 면제 목록을 넓히는 자유 표면이 된다. 하지 말 것 |
| F2 | `test_mg_harness.JobScriptsDryRun.run_job`(및 같은 필터를 쓰는 `:143, :2534, :2548, :2562`)의 env 필터에 `HF_HOME` 제거 또는 `HF_HOME=<td>/hf` 고정을 추가해 시험을 밀폐한다 | **예**(`tests/`, multigpu 트랙 소유. M3 하네스 봉인과 맞물리는지 multigpu 트랙이 확인) | 낮음. 시험 의도("이 기계에서 abort")를 바꾸지 않고 밀폐성 버그를 고친다. 반사실 A/B가 이미 있다. 단, 단정을 조건부로 약화하는 방식은 금지 | 수정 뒤 반사실 B(가짜 캐시)에서도 통과해야 한다. 이것이 "되돌린 변이에서 실패" 대응 검사다 |
| F3 | 판정기 상태 파싱을 견고하게: (i) 근본 해결로, 정규식 대신 구조화된 결과를 쓴다(unittest `TestResult`를 감싸 id별 상태를 JSON으로 내는 러너를 `preflight_cl3.sh`가 호출). 또는 (ii) `python -m unittest -b`(통과 시험의 stdout/stderr 버퍼링)로 끼어쓰기를 원천 제거한다 | 아니오(merge_trial_tools) | **예.** 이 런을 보고 정한 변경이다. 설계서에 개정으로 선언하고, selftest에 이 런의 실제 로그 형태(stdout 끼어쓰기 PRE `ok`, ResourceWarning 사이 `ok`)를 양방향 변이로 넣은 뒤 감사 | (ii)는 실패 시험의 출력 위치를 바꾼다. 서명 추출(`sigcmp`)이 영향을 받는지 먼저 확인해야 한다. "끼어든 줄 사이의 ok를 인정"하는 느슨한 정규식만으로는 **위험**하다. 다른 시험의 상태를 오인할 수 있으므로 상태 토큰이 다음 헤더 전에 정확히 1개일 때만 인정해야 한다 |
| F4 | ResourceWarning 처리: (i) F3이면 판정기 쪽 처리는 불필요. (ii) 원천은 `test_holb_probe.py:667–668`의 닫히지 않은 `patch_file.open("rb")` | (ii)는 **예**(`tests/`) | (i) 예(F3과 같은 선언). (ii) 낮음(시험 의미 불변, 자원 누수 수정) | 실행 env에 `PYTHONWARNINGS=ignore::ResourceWarning`을 거는 방식은 다른 시험이 경고에 의존하지 않는지 확인 전에는 비권고 |
| F5 | `sigcmp.mask`에 컨테이너 arm 루트 `/data/runs/<job>/pf_work/(pre\|post)/`를 추가한다(지금은 `/tmp\|/home`만) | 아니오 | **예**(데이터 본 뒤). 효과는 동일 원인을 동일 서명으로 만드는 것뿐이다(8건) | selftest에 "arm 경로만 다른 동일 실패 → 같은 서명" 통제군과 "arm 경로 + 다른 파일 → 다른 서명" 변이를 함께 넣는다 |
| F6 | L2 id가 **PRE에서 실패**하면 조용히 흡수하지 말고 `L2_UNVERIFIED`(구멍 → UNVERIFIABLE 쪽)로 올린다. `control_both_kill_layers`처럼 `N.N` 마스킹으로 다른 원인이 같은 서명이 되는 경우를 드러낸다 | 아니오 | 예. 다만 방향은 **더 엄격**(fail-closed)이라 자유 표면 위험이 작다 | 같은 이유로 판정기 출력(judge.json)에 L2별 pre/post 상태 표(§3-3)를 기록하게 하면 사람이 대조하기 쉽다(규칙 아님) |
| F7 | 실행 스펙 결정: 설계 §1의 A100x1으로 돌릴지, CPU 스펙을 유지하고 타이밍 시험(§5-1 4건)을 **다음 런 전에** "CPU-스펙 기지 실패"로 선언할지 | 아니오 | 선언 방식이면 **예**(데이터 본 뒤 목록). 스펙 변경이면 아니오 | 사용자 결정 사항. 어느 쪽이든 결과 해석이 달라지므로 런 전에 고정한다 |
| F8 | 아카이브 의존 시험(§5-2 2번): 필요한 git-ignored 원자료를 Job 볼륨에 올리거나, "컨테이너 아카이브 부재"로 사전 선언한 구멍 목록으로 둔다 | 아니오 | 선언 방식이면 예 | 대상 항목 밖이라 라벨에는 직접 영향이 없다. 범위 공시용 |
| F9 | (선택) pynvml FutureWarning 소음이 `out[-2000:]`류 시험 메시지를 밀어낸다(§3-3) | `env/`(이미지에 `nvidia-ml-py`)이면 **예** | 낮음 | 서명 품질 문제이고 판정 규칙과는 무관하다. 급하지 않다 |

**권고 순서**
1. F7(스펙 결정, 사용자).
2. F2(시험 밀폐, gate 경로이므로 새 POST가 필요. POST가 바뀌는 변경을 먼저 모은다. F4-(ii)를 함께 넣을지도 이때 결정).
3. F3 + F5 + F6을 판정기 개정 한 묶음으로 설계서에 선언하고, 이 런의 로그를 고정 픽스처로 쓴 양방향 selftest 뒤 claims-auditor 감사(최대 2회).
4. F8 범위 공시.
5. 새 POST로 preflight 재실행.

이 런의 라벨 `PREFLIGHT_FAIL_NEW`는 기록 그대로 둔다. 개정된 판정기를 이 런에 소급 적용해 PASS를 만드는 것은 금지한다(§5-2 반사실도 FAIL_NEW).
