# 시험 merge 회귀 — 기지 실패 목록 v2 (2026-10-15) — 감사 1회차 C-R1–C-R4 수리판

> 지위: 초안 v2, 커밋하지 않음. 측정 0, GPU Job 0. D9 (가) "기지 실패 목록 공시"의 후보.
> 대체 대상: v1(같은 파일명, 같은 날). v1은 두 트리 모두 `No module named 'sglang'`·미추적 아카이브 부재 상태에서 잰 것이라
> 변경 모듈 시험이 돌지 않았다(VERDICT §5 N1: "기타 0건"은 항등식). 이 v2가 그것을 정식 조건에서 다시 쟀다.
> 근거: `Q1_N2_CLOSURE_PREP_2026-10-15.md` §2·§6, `VERDICT_spec_v0_and_closure_2026-10-15.md` §5·§6(C-R1–C-R4).
> 스크립트·원본 로그: scratchpad `/tmp/claude-1000/-home-wonho-Experiments-KISTI/f75d12d4-0c3a-4b6e-bd06-788a5356ecdc/scratchpad/crr1/`
> (`logs/`, `sigcmp.py`, `pincheck.py`, `mutants.py`, `cs_fixed.py`, `pin_pre.json`, `pin_post.json`).

## 0. 한 줄 요약 (금지 문장: "회귀 0건")

- **실행된 변경 모듈 시험 326건 중 신규 실패 12건(전부 L2 = 봉인 sha 고정, 설계상 예상) / 변경 모듈 시험 미실행 13 id(약 125 test def, §5)**.
  L2 밖의 신규 실패는 실행된 범위에서는 0건이다. 그러나 미실행 범위가 있으므로 "회귀 0건"이라고 쓰지 않는다(C-4).
- v1 대비 달라진 것: 변경 모듈 시험이 실제로 돌았다(test_holb_probe 35건 통과, test_probe_flush_durability 10, test_e2_vessl 48 중 47, test_n3_capacity 88 중 77, G5(a) 19건 통과 등). 새 skip 25건은 전부 실행돼 통과했다. 환경 실패는 51 → 35로 줄었다.
- 그래도 sgl_kernel가 없는 이 로컬 환경에서 import가 안 되는 9개 모듈 + 4건은 여전히 돌지 않는다(§5). 이 부분은 증거가 아니라 **공백**이다.

## 1. 조건 (C-R1)

- main = `7b15214b2d554ec7f8394485892980af8271734c` (2026-10-15 시험 시점 HEAD; 부모 `9a2a6796`. 9a2a6796 이후 docs 커밋 1건뿐이고 `src/`·`tests/`는 동일).
  merge 대상 = `multigpu-scaling` `e44e2e09f8eaa0c63844f71df2dd054cc6606de9`. 시험 merge는 `--no-ff --no-commit`, 충돌 없음(`job_entry.sh` 자동 병합). merge 후 스테이징 2948 파일.
- **main 작업 트리는 건드리지 않았다.** 방법: scratchpad에 detached worktree 2개(`pre` = 7b15214b, `post` = 같은 커밋 + 위 merge). 각 worktree에 main의 미추적·ignored 항목 12,290개를 **심볼릭 링크**로 심어(`git status --porcelain --ignored`로 열거, 이미 있는 경로는 건너뜀) 아카이브·`hf_cache/`·`runtime_source_manifest.sha256`가 보이게 했다. 링크는 merge 뒤에 만들었다(merge가 덮어쓸 미추적 파일 충돌 방지).
- 환경변수: `PDMUX_ROOT=/home/wonho/Experiments/KISTI`, 인터프리터 miniconda base python 3.13.9(CLAUDE.md 기준선과 같음).
  - 전: `SGLANG_ENGINE_DEV=PYTHONPATH=~/Experiments/KISTI/sglang_engine_dev/python`.
  - 후: `~/Experiments/KISTI/sglang_engine_dev_multigpu/python`을 **복사한 사본**(scratchpad `devsync/`)에 `post` worktree의 `sync_engine_tree.sh`를 돌려 branch의 4개 변경 모듈(holb_probe `5a58c809…`, telemetry `d34ce5d0…`, chunk_probe `2d1d8cff…`, mixin `c2483a14…`)을 설치한 것. 이유는 아래 §1-1.
  - 시험 코드 수정 0, 새 shim 0. `sglang`은 pip 설치가 아니라 `PYTHONPATH`로 dev tree를 가리키는 env 규약으로 import한다(시험이 `from sglang.srt...`를 쓰고 dev tree가 그 설치 형태). 이것만으로 순수 파이썬 모듈(holb_probe 등)은 import된다.
- 명령(각 worktree 루트): `python -m unittest discover -s workspace/engine-port/tests -v`(+ n2_static_grid, characterization/tests, triage/t5_greenctx).
- **결과 요약**

| 실행 | tests | failures | errors | skipped | 실패 id |
|---|---|---|---|---|---|
| 전 (`logs/pre.main.log`) | 877 | 21 | 14 | 2 | 35 |
| 후 최종 (`logs/postA.log` 1176건 + `logs/postB.log` 89건) | 1265 | 33 | 14 | 2 | 47 |
| n2_static_grid 전/후 | 278 / 278 | 0 / 0 | 0 / 0 | – | 0 (정식 트리에서는 `test_positive_control_n3` 통과: 아카이브 링크 때문) |
| characterization/tests 전/후 | 2 / 2 | – | 2 / 2 | – | `No module named 'pytest'` 동일 |
| triage/t5_greenctx 전/후 | 1 / 1 | – | 1 / 1 | – | `RuntimeError: Not enough SMs available…` 동일 |

  (L2 12건 + 환경 35건 = 후 47건. 전 877 → 후 1265 = branch 추가 시험 +388.)

### 1-1. 시험 도구에 관한 중요한 발견 (순서 의존·부작용)

- `test_campaign_provenance`·`test_lambda_star_per_shape`·`test_r2_eval_runner`(sbatch 구동 시험)는 `$PDMUX_ROOT/prefill-layer-alloc`(= main 트리)의 `r2_eval.sbatch`를 호출하고, 그 안의 `sync_engine_tree.sh`가 상속된 `SGLANG_ENGINE_DEV`에 **main의 src 바이트를 덮어 설치한다**. 이 시험들이 알파벳 순으로 `test_holb_probe`보다 앞이라, 후 트리를 그대로 두고 한 번에 돌린 첫 실행(`logs/post.main.log`)에서는 변경 모듈이 도중에 main 바이트로 되돌려져 `test_holb_probe.TestMirrorSync`·`test_probe_flush_durability.TestMirrorSync` 2건이 추가로 실패했다. 이 2건은 merge 회귀가 아니라 **dev tree 상태 문제**다. 최종 실행은 (A) sbatch 구동 3모듈을 제외한 1176건(`devsync` 사본, 시험 후 sha 불변 확인), (B) 그 3모듈 89건(별도 사본 `devsyncB`)으로 갈라 돌렸다.
- **부작용 공시:** 이 시험 도중 sbatch 구동 시험이 실제 `~/Experiments/KISTI/sglang_engine_dev/python`(내용 동일, mtime만 갱신)과 `~/Experiments/KISTI/sglang_engine_dev_multigpu/python`을 main 바이트로 덮어 설치했다. 후자는 시험 전 4개 변경 모듈의 바이트를 확인하지 못했다(시험 후 main 바이트 `97cd…`·`e439…` 등으로 확인). 그래서 시험 뒤 `prefill-layer-alloc-multigpu`(e44e2e09)의 `sync_engine_tree.sh`로 **branch HEAD 바이트로 재설치**해 복구했다(4개 모듈 sha가 branch src와 일치함을 확인). 복구 기준이 "시험 전 상태"가 아니라 "branch HEAD"이므로, multigpu 세션이 그 트리를 다른 커밋 기준으로 쓰고 있었다면 다시 `sync_engine_tree.sh`를 돌려야 한다(미확인).
- 이 시험들이 의존하는 `engine_bench_runner.sh:71`(전)·`:115`(후, branch가 +85줄 고침)의 `sglang_engine_venv/bin/activate`가 이 머신에 없어 17건이 거기서 멈춘다. 즉 **branch가 고친 `engine_bench_runner.sh`는 이 환경에서 시험으로 검증되지 않는다**(§5).

## 2. merge로 새로 생긴 실패 12건 — 전부 L2 (C-R2: id + 서명)

서명 = (종류, 예외 유형, 마스킹한 첫 메시지 줄, 마스킹한 마지막 줄, 핵심어 집합, `DRIFT` 줄 다이제스트). 마스킹: 경로·소수·5자리 이상 수·hex·`line N`. 짧은 정수(rc 3 ≠ 0 등)는 마스킹하지 않는다.
원인은 하나: 브랜치가 `src/multiplex/{multiplexing_mixin,telemetry,holb_probe,chunk_probe}.py`를 고쳐 **추적 중인 src 미러 sha ≠ N3/E2 등록 manifest**(mixin `c2483a14…` ≠ 등록 `ba5835aa…`, telemetry `d34ce5d0…` ≠ `e4399835…`, holb `5a58c809…` ≠ `97cd7392…`, chunk `2d1d8cff…` ≠ `7dfa76a9…`).

| 시험 id (모듈.클래스.메서드) | 전 | 후 | 서명 (예외 / 핵심 메시지) |
|---|---|---|---|
| test_e2_vessl.TestEngineTree.test_head_mixin_is_src_mirror | PASS | FAIL | AssertionError / sha256 문자열 불일치(`'<hex>' != '<hex>'`) |
| test_n3_capacity.TestRegistration.test_live_repo_matches_registered_digests | PASS | FAIL | AssertionError / `Lists differ: ['sglang/srt/multiplex/chunk_probe.py: mirror …'] != []` (mirror_check) |
| test_n3_capacity.TestRegistration.test_plan_selftest | PASS | FAIL | AssertionError / mirror_check 목록 비어야 함 위반 |
| test_n3_capacity.TestJobSymlinkLayout.test_symlinked_results_layout | PASS | FAIL | AssertionError `1 != 0` / n3_plan.py main의 mirror_check |
| test_n3_capacity.TestAbortBranchesBehaviour.test_unbroken_copy_runs | PASS | FAIL | AssertionError `3 != 0` / `DIGESTS DRIFT (3 files + 9 mirror entries)` → `ABORT_REG_DRIFT` |
| test_n3_capacity.TestAbortBranchesBehaviour.test_config_without_fixed_dsm_aborts | PASS | FAIL | AssertionError / 기대 `ABORT_CONFIG` 대신 `ABORT_REG_DRIFT`가 먼저 발화 |
| test_n3_capacity.TestR2R7Runner.test_r2_dryrun_placeholder_never_admits_an_arm_unit | PASS | FAIL | AssertionError `3 != 0` / `ABORT_REG_DRIFT` |
| test_n3_capacity.TestResume.test_resume_and_its_controls | PASS | FAIL | AssertionError `3 != 2` / `ABORT_REG_DRIFT` |
| test_n3_capacity.TestRunnerEnv.test_inherited_phase_events_allowed_and_overwritten | PASS | FAIL | AssertionError `3 != 0` / `ABORT_REG_DRIFT` |
| test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut | PASS | FAIL | AssertionError `3 != 0` (preflight rc=3, drift abort가 예산 강제보다 먼저) |
| test_n3_capacity.TestBudgetEnforcer.test_equivalent_mutant_watchdog_removed_still_cuts | PASS | FAIL | AssertionError `3 != 0` |
| test_n3_capacity.TestBudgetEnforcer.test_control_both_kill_layers_removed_overruns | PASS | FAIL | AssertionError `0.9 not >= 60` "mutant still cut — vacuous" (0.9 s에 drift abort) |

- `DRIFT` 줄 4개(chunk/holb/mixin/telemetry의 `mirror … = <sha16> != registered <sha16>`)의 다이제스트가 `6b227c54fb22`로 4개 시험에서 동일하다(서명 일부).
- **L3: 0건 실패.** `test_e2_vessl_r2`(test_e2_vessl.py를 `d163d702…`로 고정)는 전·후 통과. L2를 고치거나 skip하면 L3가 터진다(= 고치지 말라는 근거).
- v1의 "기타 진짜 회귀 0건" 문장은 폐기한다. 실행된 범위(§0)에서는 L2 밖 신규 실패가 없다는 사실만 남는다.

### 2-1. 서명 대조 (C-R2) — 결과와 변이 시험(교훈 53)

- **전 35개 실패 id vs 후 최종**: 공통 35 id의 서명 변화 **0**, 후에만 생긴 id 12(위 L2), 사라진 id 0. 같은 35 id가 3번의 후 실행(`post.main.log`·`postsync.main.log`·최종)에서 동일 서명으로 재현됐다.
- 도구: `sigcmp.py`(전/후 비교), `pincheck.py`(핀 대조: `pin_post.json` 47건, `pin_pre.json` 35건). 핀 대조는 "실패 id 집합과 모든 서명이 정확히 일치할 때만 rc 0".
- **변이 시험**(`mutants.py`, 최종 후 로그를 변이해 `pin_post.json`과 대조): 통제군(무변이) rc=0. 6개 변이 전부 rc=1로 탐지됨.

| 변이 | 효과 | 결과 |
|---|---|---|
| M1 같은 id, 다른 원인(venv `No such file` → `Permission denied`) | signature_changed=1 | 탐지 |
| M2 예외 유형 `ModuleNotFoundError` → `ImportError` | signature_changed=1 | 탐지 |
| M3 L2 블록 안 DRIFT sha 한 줄 변경 | signature_changed=1 | 탐지 |
| M4 `ABORT_REG_DRIFT` → `ABORT_CONFIG`(한 블록) | signature_changed=1 | 탐지 |
| M5 FAIL → ERROR(한 블록) | signature_changed=1 | 탐지 |
| M6 실패 블록 하나 삭제 | pinned_but_absent=1 | 탐지 |

  (첫 구현은 M3을 놓쳤다 — 변이가 블록 밖 DRIFT 줄에 적용돼서였다. 변이를 블록 안으로 고쳐 재실행했다. 또 첫 마스킹은 모든 숫자를 지워 `rc 3 ≠ 0`과 `rc 4 ≠ 0`이 같은 서명이 됐다. 짧은 정수를 마스킹하지 않도록 고쳤더니 `engine_bench_runner.sh`의 `line 71`/`line 115`가 "다른 서명"으로 드러났고, 원인은 branch가 그 스크립트를 고쳐 줄 번호가 밀린 것 — `line N` 마스킹으로 처리, 아래 §4 참조.)
- 한계: 서명은 텍스트 수준이다. 같은 예외·같은 메시지의 다른 원인은 못 가른다. 핀은 한 번의 실행에서 만들었고 다른 실행 3회와 일치를 확인했다. 마스킹된 `N.N` 소수 비교(타이밍 시험)는 값을 보지 않는다.

## 3. 목록 3분할 (C-R3)

### 3-1. L2 (봉인 sha 고정) — 12건: §2 표. 처리 = 수정·skip 금지, id 단위 고정 목록(`pin_post.json`)으로 공시. 목록 밖 신규 실패는 즉시 회귀로 취급.

### 3-2. 환경 실패 — 전·후 동일 35건 (정식 트리·아카이브·PDMUX_ROOT 기준으로 재집계; v1의 51건에서 감소)

| 건수 | 원인(서명) | 해당 | 변경 모듈 관련 |
|---|---|---|---|
| 13 | `ModuleNotFoundError: No module named 'sgl_kernel'` (`fp8_kernel.py:55 from sgl_kernel import sgl_per_token_quant_fp8` 경로) | 모듈 단위 import 오류 9(test_chunk_probe, test_phase_events, test_mem_telemetry_symmetry, test_sticky_partition, test_true_dual_prefill_ownership, test_worker_grad_guard, test_r2_admission_latch, test_r2_admission_persistence, test_trace_force_prefill) + test_holb_probe.TestSchedulerHooksInstalled 4 | **예 — §5 미검증** |
| 17 | `sglang_engine_venv/bin/activate: No such file or directory` (`engine_bench_runner.sh:71`, 변이본은 `:115`) → `Synced PD-mux runtime: …` 뒤 rc 1 | test_campaign_provenance 10, test_lambda_star_per_shape.SbatchReadsTheField 3, test_r2_eval_runner.TestSbatchPathResolution 4 | 간접(branch가 고친 `engine_bench_runner.sh`가 이 환경에서 닿지 않음) — §5 |
| 4 | `FileNotFoundError: no lambda* table at …/scripts/r2_eval/lambda_star.example.json` | test_lambda_star_per_shape: CliSurfaces 2, FailClosed 1(error), ShellRefusals 1 | 아니오. 이 파일은 `.gitignore` `*.json`에 걸려 **main 작업 트리에도 없다**(아카이브 부재) |
| 1 | `LocalEntryNotFoundError` — `Zyphra/Zamba2-2.7B`가 hf_cache에 없음 | test_r2_correctness_ctx.TestCachedModelsResolveAsExpected.test_zamba2_2_7b_still_derives_4096 | 아니오 |

- n2_static_grid `test_positive_control_n3`: 아카이브(미추적 results)를 링크한 정식 조건에서 **통과**(278/278). v1에서의 실패는 아카이브 부재 탓이었다. 환경 분류에서 제외.
- triage/t5_greenctx(1 error): `test_greenctx` import 시 `sgl_kernel` green context 호출 → `RuntimeError: Not enough SMs available…`(로컬 GPU), 전·후 동일. 회귀 후보에서 제외할 것.
- characterization/tests(2 errors): `No module named 'pytest'`, 전·후 동일. e4_concurrent: 시험 0건.
- skip 2건(전·후 동일): `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base not in hf_cache`, `test_zamba2_instrumentation` setUpClass "sglang zamba2 import failed: No module named 'sgl_kernel'"(후자는 미검증 범주).

### 3-3. 새 skip 25건 (52→77, v1) — 이번에 전부 실행·통과

v1에서 미실행이던 시험 25건은 `SGLANG_ENGINE_DEV`를 주자 skip이 풀려 모두 **ok**: test_mg_agnostic_manual_equiv 18(G5(a) 결정적 13 + 기록 5), test_multigpu_guard 6(설치된 pdmux_context.py 의존), test_m2g_gate 1. 즉 G5(a) 18건은 이제 "runtime tree not found" 없이 실제로 실행됐다. 후 최종의 skip은 위 2건뿐이다.

## 4. 변경 모듈별 시험 실행 현황 (파싱 기반, 근사)

| 시험 파일 | 실행·통과 | 실행·실패 | 미실행 |
|---|---|---|---|
| test_holb_probe | 35 | 0 | 4 (TestSchedulerHooksInstalled, sgl_kernel) |
| test_probe_flush_durability | 10 | 0 | 0 |
| test_e2_vessl | 47 | 1 (L2) | 0 |
| test_n3_capacity | 77 | 11 (L2) | 0 |
| test_mg_agnostic_manual_equiv (G5(a)) | 19 | 0 | 0 |
| test_m2g_gate | 95 | 0 | 0 |
| test_multigpu_guard | 31 | 0 | 0 |
| test_dual_worker | 10 | 0 | 0 |
| **소계(실행)** | **326** | **12 (전부 L2)** | |
| test_chunk_probe(def 49)·phase_events(8)·mem_telemetry_symmetry(12)·sticky_partition(12)·true_dual_prefill_ownership(4)·worker_grad_guard(13)·r2_admission_latch(6)·r2_admission_persistence(12)·trace_force_prefill(5) | – | – | **9개 모듈 전체**(수집 단계 import 오류; def 수 합 121, 근사) |

(수치는 `-v` 출력 파싱이고 로그 끼어쓰기로 1–2건 놓칠 수 있다. 총량 1265와 `Ran` 줄은 일치.)
`engine_bench_runner.sh` 줄 번호 `71`→`115`는 branch 변경의 부수 효과로, 같은 원인(venv 없음)이라 서명 비교에서 `line N`으로 정규화했다.

## 5. 변경 모듈 회귀 미검증 목록 (이 로컬 환경에서 검증 불가) — 공백으로 공시

이유(예외 메시지): `ModuleNotFoundError: No module named 'sgl_kernel'` — base python에는 `sgl_kernel`이 없고, `prefill-alloc` conda env(py3.10)에는 있으나 `common_ops` 로드에서 실패(`No module named 'common_ops'`; 앞선 기준선 문서는 `undefined symbol`). 저장소에 `sgl_kernel` stub/shim은 없다(`env/sitecustomize.py`는 py3.14 shim뿐). 시험 코드 수정이나 임의 stub 주입은 하지 않았다(주입하면 "import는 되지만 검증은 아닌" 항등식을 다시 만들 위험).

1. 9개 모듈(위 §4 하단): 각각 변경된 mixin·telemetry·chunk_probe 등을 import/참조(chunk_probe→chunk_probe·holb_probe, phase_events·mem_telemetry→telemetry, sticky·grad_guard·admission·trace_force→mixin/dual_worker/controller 계열).
2. test_holb_probe.TestSchedulerHooksInstalled 4건(scheduler import).
3. test_zamba2_instrumentation(전체 skip, 같은 사유).
4. 17건(§3-2 venv): branch가 +85줄 고친 `engine_bench_runner.sh`·`multigpu_cell_meta.py` 경로는 이 환경에서 닿지 않는다. 변경 스크립트 회귀 미검증.
5. 이 모든 항목은 sgl_kernel이 정상 빌드된 환경(VESSL pdmux-build Workspace 또는 A100 컨테이너 CPU 모드)에서만 확인 가능하다. 로컬에서 닫을 수 없다(C-R1의 이 부분은 불가). **판단: D9 공시에 "변경 모듈 시험 일부는 로컬에서 미실행"을 명시해야 한다.**

## 6. check_citation_stops — merge 커밋 1회 예외 고정값 (C-R4)

- 도구는 스테이징 diff의 추가 줄만 검사한다. 시험 merge(스테이징 2948 파일)에서 **추가 11,285,275줄, 위반 1230건, rc=1**. 위반 경로는 전부 `workspace/engine-port/results/multigpu/**/*.{csv,json}`(csv·json 각 파일 형식 외, .md/.py 위반 0). pre-commit hook 없음. 전(merge 없음)은 0 added·0 violation·rc=0.
- 패턴별 집계(`cs_check.py`로 분류 + `cs_fixed.py`로 핀 고정; 합계 1230):

| 패턴 | 건수 | 원인 |
|---|---|---|
| `20\.39` | 386 | 로그 타임스탬프 `…08:18:20.393` 안에 임베드 | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
| `19\.49` | 464 | clk.csv 타임스탬프 `…14:19.495` 안에 임베드 | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
| `33\.04` | 289 | clk.csv 타임스탬프 `…33:33.046` 안에 임베드 | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
| `870295\|870296\|870297\|870301` | 35 | 부동소수 숫자열 속 우연(`…0.0009148702956736088`, power_samples `…870296,…`) | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
| `13\.711\|13\.71\s*ms` | 52 | clk.csv 타임스탬프 `…18:13.711` 46(독립 토큰이지만 시각 필드) + energy.csv 임베드 6 | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
| `7\.24\s*s` | 4 | 부팅 로그 `Capture cuda graph end. Time elapsed: 7.24 s`(§1-4의 7.24 s와 다른 양) 2 + tqdm 줄 2(첫 줄이 잘려 매칭 위치 확인 불가) | <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->

  분류 합계: 숫자열 임베드 1174 + 13.711 임베드 6 + clk.csv 시각 초 필드 46 + 부팅 로그 4 = 1230. <!-- [CS-OK] 금지 패턴의 메타 기술(값 인용 아님), 사용자 승인 2026-10-15 -->
- **처리: 이 merge 커밋 1회에 한한 예외.** 원시 파일에 `[CS-OK]` 삽입·경로 영구 면제·도구 수정 없음. 예외 값은 `cs_fixed.py`로 (합계 1230, 위 패턴별 건수, 경로 `^workspace/engine-port/results/multigpu/.*\.(csv|json)$`, 추가 줄 11,285,275)에 고정했고, 이 값에서 벗어나면(건수·패턴·경로 중 하나라도) 예외는 무효다.
- 변이 시험(교훈 53): 통제군(실제 로그) `PINNED_OK` rc=0. (M1) .md 경로 위반 1건 추가, (M2) csv 위반 1건을 .py 경로로, (M3) 한 위반의 패턴을 바꿈, (M4) 위반 1건 삭제 → 4개 전부 `PIN_MISMATCH` rc=1.
- 이 예외는 이 merge 커밋에서만 유효하며, 다음 커밋에는 평소대로 도구를 적용한다.

## 7. 기지 실패가 봉인 캠페인 종료 후 무해한 이유 (v1 §6 유지, 단 한정 추가)

- L2 12건은 전부 "N2/N3/E2 등록 엔진 manifest(25항목, mixin `ba5835aa…`)와 현 src 미러의 일치"를 시험한다. 캠페인 종료 = 그 manifest로 새 Job을 내지 않음이므로 일치는 더 이상 보장할 대상이 아니다. branch의 엔진 변경은 PREREG_MULTIGPU가 "봉인 digest와 다르다(의도)"로 명시했다. 과거 결과는 영향 없음(Job은 `PDMUX_COMMIT`을 체크아웃해 설치).
- 시험 파일은 봉인 바이트(L3)이므로 수정·skip 금지. 처리 = id 단위 고정 목록 공시.
- 무해하지 않은 점: (i) 12건이 상시 빨간불이면 `test_n3_capacity.py`의 새로운 진짜 회귀를 가린다 → `pin_post.json` 대조로 id+서명 단위 감시. (ii) `TestBudgetEnforcer`·`TestResume`은 drift abort에 가려 예산 강제·재개 로직을 더 이상 검증하지 못한다(하네스를 다른 캠페인이 재사용하면 공백). (iii) G0가 같은 `src/multiplex`를 고치면 같은 12건이 이미 빨강이므로 G0 변경으로 인한 L2 추가 실패는 구분 불가 → G0 착수 전 이 핀을 기준선으로 고정. (iv) **§5 미검증 공백**이 있으므로 이 목록만으로 "merge가 변경 모듈을 깨지 않았다"고 말할 수 없다.

## 8. 미확인

- §5의 미실행 시험이 sgl_kernel 환경에서 통과하는지(VESSL에서만 확인 가능).
- multigpu 세션의 `sglang_engine_dev_multigpu` 원래 상태(§1-1; branch HEAD로 복구함).
- 실제 merge 커밋 시 presubmit이 citation-stop rc=1로 막는지(예외 고정값은 위 §6).

## 승계 금지 문안 (감사 1회차 판정서 §7, 문자 그대로)

- **C-4** "시험 merge에서 회귀 0건" 금지(변경 모듈 시험 미실행).
- **C-5** "citation stop 위반 1230건은 무해하므로 results/multigpu는 검사 면제" 금지.

## 대조 도구 보존 (2026-10-15, main 세션)

scratchpad는 세션 전용이라 서명 대조·고정 도구를 `merge_trial_tools/`로 옮겼다: `sigcmp.py`, `pincheck.py`, `mutants.py`, `pin_pre.json`, `pin_post.json`, `cs_fixed.py`. 실제 merge 직전 이 도구로 (id, 서명)·citation stop 고정값을 다시 대조한다. 공유 dev tree `sglang_engine_dev_multigpu`의 변경 모듈 4개 sha가 branch HEAD `e44e2e09`의 src와 일치함을 main 세션이 재확인했다(시험 부작용으로 덮인 뒤 재동기화).
