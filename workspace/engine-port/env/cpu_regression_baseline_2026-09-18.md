# CPU 회귀 기준선 (2026-09-18, 로컬 PC, dev tree 재구성 후)

작성자: experiment-runner (CPU 전용 세션). 성능 판정 0건 — 이 문서는 **배관 상태 기록**이다.

## 0. 배경

`migration_plan_local_dev_remote_gpu_2026-09-18.md` §3-3의 "서버 부트스트랩" 순서 중
"SGLang v0.5.10 소스 → `sync_engine_tree.sh` → `devtree_manual_edits.patch`" 부분을
로컬(GPU 없이)에서 수행해, 테스트가 참조하는
`~/Experiments/KISTI/sglang_engine_dev/python/sglang/...` 가 실재하도록 만들었다.
이 작업 이전 기준선(작업 지시서에 기재된 값, 인터프리터 미상이나 아래 §3에서 동일 인터프리터로
확인): **519 tests / failures=7 / errors=36 / skipped=11 / ~272초**, errors 다수가
`FileNotFoundError: .../sglang_engine_dev/python/sglang/...`.

## 1. 수행한 명령 (실제 실행분)

```bash
# 1) SGLang v0.5.10 소스 확보 (저장소 밖, 형제 디렉터리)
cd ~/Experiments/KISTI
git clone --depth 1 --branch v0.5.10 https://github.com/sgl-project/sglang.git sglang_engine_dev
# -> commit 1519acf37c23f2189adb93f57ca9cd2db1bebf18 (2026-04-05 09:47:12 -0700, tag v0.5.10)

# 2) dev tree 동기화 (SGLANG_ENGINE_DEV 명시 — sync_engine_tree.sh의 PDMUX_ROOT 기반 기본값을 우회)
cd ~/Experiments/KISTI/prefill-layer-alloc
SGLANG_ENGINE_DEV="/home/wonho/Experiments/KISTI/sglang_engine_dev/python" \
  bash workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh
# manifest 인자 미지정 -> 스크립트 기본값 사용:
#   workspace/engine-port/results/runtime_source_manifest.sha256 (gitignored, 25개 항목)
# 출력: "Synced PD-mux runtime: /home/wonho/Experiments/KISTI/sglang_engine_dev/python"

# 3) 수동 편집 patch (5개 파일, dry-run 무충돌 확인 후 적용)
patch -p1 -d /home/wonho/Experiments/KISTI/sglang_engine_dev/python \
  < workspace/engine-port/env/devtree_manual_edits.patch
# patching: configs/__init__.py, layers/attention/triton_backend.py,
#           managers/scheduler.py, model_executor/model_runner.py,
#           utils/hf_transformers_utils.py  (전부 성공, 충돌 0)

# 4) CPU 회귀 재실행 (아래 §2 참조)
```

사용한 manifest: 스크립트 기본 경로
`workspace/engine-port/results/runtime_source_manifest.sha256`
(2번째 인자를 넘기지 않음 — `.sha256` 파일 자체는 `.gitignore:127`로 추적 제외됨).
경로가 절대경로(`/home/wonho/Experiments/KISTI/sglang_engine_dev/python/...`)로
기록되므로 이 세션의 소스가 어느 파일에서 왔는지는 이 매니페스트로 재현 가능.

## 2. CPU 회귀 재실행 — 인터프리터별 결과

### (A) `python3` = miniconda base 3.13.9 (`/home/wonho/miniconda3/bin/python3`) — ★정본 검증 경로 권장

```bash
cd ~/Experiments/KISTI/prefill-layer-alloc
python3 -m unittest discover -s workspace/engine-port/tests -v
```
결과: **Ran 598 tests in 296.422s — FAILED (failures=26, errors=13, skipped=3)**
(추가 env var 없음 — CLAUDE.md/체크리스트가 문서화한 명령 그대로, `python`도 동일 바이너리를 가리킴).

### (B) `prefill-alloc` conda env, python 3.10.20 (`sglang 0.5.10.post1` pip 설치본)

```bash
~/miniconda3/envs/prefill-alloc/bin/python -m unittest discover -s workspace/engine-port/tests -v
```
결과: **Ran 598 tests in 329.969s — FAILED (failures=27, errors=13, skipped=2)**

두 인터프리터의 실패 시그니처는 **거의 동일**(39/41개 signature 일치). 차이는:
- `sgl_kernel` 부재 시 예외 형태만 다름(A: `ModuleNotFoundError`, B: `undefined symbol`
  — 둘 다 §4(a) 범주, sm120 로컬 GPU에서 예상된 실패).
- B에서만 2건 추가 실패 — 이 세션 도중 **다른 에이전트가 GPU로 HF 캐시를 채우던 레이스**로
  보인다(A 실행 시점엔 `Zyphra/Zamba2-2.7B`가 hf_cache에 없어 스킵됐고, B 실행 시점엔
  `_cached()`가 True를 반환했지만 실제 `AutoConfig.from_pretrained` 해석은
  `LocalEntryNotFoundError`로 실패 — 부분 다운로드/미완성 캐시로 추정). §4(c)로 분류,
  **재현 불필요**(외부 요인, 이 세션의 dev-tree 작업과 무관).

### 정본 권장: (A) 미니콘다 base `python3`

이유: (1) CLAUDE.md·`gpu_rental_checklist_2026-09-18.md`가 문서화한 명령을 그대로 쓸 때
자동으로 이 인터프리터가 선택된다(별도 `conda activate` 불필요, 온보딩 마찰 최소).
(2) `sgl_kernel` 부재가 깨끗한 `ModuleNotFoundError`로 나타나 실패 원인이 즉시 분류된다
(B의 `undefined symbol` 형태는 "설치는 됐는데 깨졌다"는 오인을 유발하기 쉽다).
(3) 이번 세션에서 B만 겪은 hf_cache 레이스처럼, B는 실제 GPU 서빙 스택(`sglang` pip
설치본)과 얽혀 있어 CPU 전용 회귀에는 불필요한 잡음을 더 만든다.
**단, B는 "실제 서버에 설치될 패키지 버전(`sglang 0.5.10.post1`)에 대해 우리 코드가
돌아가는가"의 교차검증으로는 유효하므로 완전히 버리지 말 것.**

## 3. 전후 비교 (해소 vs 잔존)

| | 이전 기준선 | 이번 실행 (A, python3, 추가 env 없음) |
|---|---|---|
| tests | 519 | 598 (+79 — 이전엔 import 단계에서 죽어 개별 test method가 아예 수집 안 된 모듈들이 이번엔 로드되어 그 안의 실제 테스트가 실행됨) |
| failures | 7 | 26 |
| errors | 36 | 13 |
| skipped | 11 | 3 |
| 소요 | ~272s | 296.4s |
| pass 수 (역산) | 465 | 556 |

- **`FileNotFoundError: .../sglang_engine_dev/python/sglang/...` (dev tree 부재) — 0건 잔존**
  (`grep -c` 확인). 이번 작업의 목표였던 결함 클래스는 **완전히 해소**.
- errors 36→13: 남은 13건은 전부 `sgl_kernel`(GPU 커널 확장) 부재로 인한 import 실패(§4-a) — dev
  tree 문제가 아니라 **원래도 이 CPU 박스에서 재현 불가능한 부류**. 이전 기준선의 36건 중
  나머지(≈23건 상당)는 dev tree 부재로 가려져 있던 것이 이번에 드러나 **failures 쪽으로
  이동**했다(아래).
- failures 7→26: 증가분(+19)의 실체는 새로운 결함이 아니라 **이전엔 import 자체가 실패해서
  도달조차 못했던 assertion들이 이번엔 실행되어 자신의 실패 사유(주로 §4-c)를 드러낸 것**.
  즉 "더 나빠졌다"가 아니라 "더 많이 보이게 됐다".

## 4. 잔존 실패 분류 (A 실행, 39건 = failures 26 + errors 13)

### (a) `sgl_kernel`/GPU 부재 — 12건, **정상적으로 예상된 실패** (이 세션 범위 밖, 고치지 말 것)

원인: `sglang.srt.layers.quantization.fp8_kernel` (및 `layers/attention/utils.py`)가
`from sgl_kernel import ...`로 컴파일된 CUDA 확장을 요구. 이 박스의 `sgl_kernel`은
sm120(로컬 5060 Ti) 대응 빌드가 없거나(`prefill-alloc` env) 아예 미설치(base env)다.
CLAUDE.md가 이미 경고한 바로 그 결함(★로컬 GPU sm_120 미지원)의 CPU 사이드 이펙트.

```
ERROR: test_chunk_probe (unittest.loader._FailedTest.test_chunk_probe)
ERROR: test_class_level_default_is_none (test_holb_probe.TestSchedulerHooksInstalled...)
ERROR: test_hooks_present_in_all_three_dispatch_branches (test_holb_probe.TestSchedulerHooksInstalled...)
ERROR: test_off_path_is_a_none_check (test_holb_probe.TestSchedulerHooksInstalled...)
ERROR: test_patch_reapplies_the_hooks_and_nothing_else (test_holb_probe.TestSchedulerHooksInstalled...)
ERROR: test_mem_telemetry_symmetry (unittest.loader._FailedTest.test_mem_telemetry_symmetry)
ERROR: test_r2_admission_latch (unittest.loader._FailedTest.test_r2_admission_latch)
ERROR: test_r2_admission_persistence (unittest.loader._FailedTest.test_r2_admission_persistence)
ERROR: test_sticky_partition (unittest.loader._FailedTest.test_sticky_partition)
ERROR: test_trace_force_prefill (unittest.loader._FailedTest.test_trace_force_prefill)
ERROR: test_true_dual_prefill_ownership (unittest.loader._FailedTest.test_true_dual_prefill_ownership)
ERROR: test_worker_grad_guard (unittest.loader._FailedTest.test_worker_grad_guard)
```
GPU가 있어야만 확인되는 부분: 이 12건 + 이들이 커버하는 `dual_worker`/`sticky_partition`/
`chunk_probe`/`holb_probe`/`trace_force_prefill`/`true_dual_prefill_ownership`/
`worker_grad_guard`/`r2_admission_*` 로직 자체의 정합성. **현재 CPU로는 원천적으로 검증
불가** — 대여 GPU 확보 후 `sgl_kernel`이 정상 빌드된 환경에서 재확인 필요.

### (c) 환경변수/로컬 자산 미비 — 27건, **실제 코드결함 아님** (다만 문서·doc-steward 회부 필요)

**(c-1) `PDMUX_ROOT`/`PDMUX_PROJECT_ROOT` 미설정 — 23건.**
`r2_eval.sbatch:51`이 최근(우연히 같은 세션대의 다른 작업으로, **커밋 안 됨**) 다음처럼
바뀌어 있다: `project_root="${PDMUX_ROOT:?...}"` — 두 env var 모두 없으면 **의도적으로
unbound-variable 에러로 죽는다**(fail-closed 설계, 코드 주석에 "Neither set -> project_root
deliberately resolves to an unbound-variable error"라고 명시). 이 스크립트를 subprocess로
호출하는 `test_r2_eval_runner.py`·`test_campaign_provenance.py`·`test_lambda_star_per_shape.py`
의 `_SbatchDriver`/유사 fixture는 `self.env = dict(os.environ)`으로 **호출 프로세스의 환경을
그대로 상속**할 뿐, `PDMUX_ROOT`를 스스로 export하지 않는다. 즉 이 23건은 **`PDMUX_ROOT`를
export하지 않고 테스트를 돌리면 항상 이렇게 실패하도록 설계돼 있다** — 코드 결함이 아니라
"실행자가 작업 루트를 알려줘야 한다"는 요구사항이 테스트 fixture에는 아직 반영 안 된 상태.

`PDMUX_ROOT=/home/wonho/Experiments/KISTI`를 주고 재실행해 확인:
```bash
PDMUX_ROOT=/home/wonho/Experiments/KISTI python3 -m unittest discover -s workspace/engine-port/tests -v
# Ran 598 tests in 345.825s — FAILED (failures=21, errors=13, skipped=2)
```
23건 중 **6건은 이것만으로 통과**(전부 "부팅 전에 거부돼야 한다"는 fail-closed 테스트라
`sglang_engine_venv` 활성화 전에 이미 끝남):
`test_contradicting_env_model_is_refused`, `test_env_contradicting_cuda_graph_{false,true}_is_refused`,
`test_oversized_trace_is_refused_before_any_allocation_is_used`,
`test_record_context_length_above_the_model_limit_is_refused`,
`test_unmeasured_record_is_refused_before_any_boot`.

**(c-2) 나머지 17건은 `PDMUX_ROOT`를 줘도 그 다음 단계에서 막힌다**:
`engine_bench_runner.sh:71`이 `/home/wonho/Experiments/KISTI/sglang_engine_venv/bin/activate`를
요구하는데, 이 로컬 머신에는 **그런 venv가 없다**(conda env만 있음 — 이양 문서가 상정한
"`prefill-layer-alloc/`·`sglang_engine_dev/`·`sglang_engine_venv/` 형제 디렉터리" 레이아웃
중 `sglang_engine_venv`는 이번 작업 지시 범위에 없었으므로 만들지 않았다). `PDMUX_DRY_RUN=1`
이라 GPU는 필요 없지만, 스크립트가 venv 활성화를 무조건 실행해 CPU-only dry-run조차 막힌다.
**이 작업의 지시("GPU를 쓰지 말고 서빙을 띄우지 마라", "venv 우회 금지")에 따라 venv를
만들지 않고 그대로 보고한다.**

```
FAIL: test_a_spool_copy_unrelated_cwd (test_r2_eval_runner.TestSbatchPathResolution...)
FAIL: test_agreeing_env_model_is_accepted (test_campaign_provenance.TestRecordDrivesWhatIsServed...)
FAIL: test_b_login_node_direct_run (test_r2_eval_runner.TestSbatchPathResolution...)
FAIL: test_c_array_task_selects_its_own_run (test_r2_eval_runner.TestSbatchPathResolution...)
FAIL: test_context_length_comes_from_the_record (test_campaign_provenance.TestRecordDrivesWhatIsServed...)
FAIL: test_cuda_graph_false_adds_the_disable_flags (test_campaign_provenance.TestCudaGraphFieldIsRead...)
FAIL: test_cuda_graph_true_means_no_disable_flags (test_campaign_provenance.TestCudaGraphFieldIsRead...)
FAIL: test_measured_record_boots (test_lambda_star_per_shape.SbatchReadsTheField...)
FAIL: test_model_comes_from_the_record (test_campaign_provenance.TestRecordDrivesWhatIsServed...)
FAIL: test_mutant_dropping_context_preflight_reintroduces_16384 (test_r2_eval_runner.TestSbatchPathResolution...)
FAIL: test_mutant_dropping_the_read_makes_the_field_decoration_again (test_lambda_star_per_shape.SbatchReadsTheField...)
FAIL: test_mutant_ignoring_the_field_makes_it_decoration_again (test_campaign_provenance.TestCudaGraphFieldIsRead...)
FAIL: test_null_context_length_derives_the_model_limit (test_campaign_provenance.TestRecordDrivesWhatIsServed...)
FAIL: test_run_json_keeps_the_declared_fields (test_campaign_provenance.TestRunRecordIsCopiedVerbatim...)
FAIL: test_server_args_txt_records_the_realized_tuple (test_campaign_provenance.TestRunRecordIsCopiedVerbatim...)
FAIL: test_unmeasured_record_boots_with_the_explicit_optin (test_lambda_star_per_shape.SbatchReadsTheField...)
FAIL: test_v1_record_still_runs_through_the_env (test_campaign_provenance.TestRecordDrivesWhatIsServed...)
```

**doc-steward/engine-porter 회부 사항**: `workspace/engine-port/scripts/r2_eval/r2_eval.sbatch`,
`scripts/bootstrap/install_engine.sh`, `scripts/r2_eval/engine_bench_runner.sh`,
그리고 이들을 호출하는 테스트 9개 파일(`test_campaign_provenance.py`,
`test_chunk_probe.py`, `test_cp0_p1_tools.py`, `test_lambda_star_per_shape.py`,
`test_probe_flush_durability.py`, `test_r2_correctness_instrument.py`,
`test_r2_eval_runner.py`, `test_sticky_partition.py`, `test_trace_force_prefill.py`,
`test_zamba2_instrumentation.py`)가 **커밋 안 된 작업 트리 상태**로 `PDMUX_ROOT` 기반
경로 해석으로 바뀌어 있다(`git status` 확인, 이 세션이 만든 변경 아님 — 동시에 다른
에이전트가 작업 중인 것으로 보인다). 이 상태로 커밋되면: (i) 로컬 CPU 회귀를 돌리는
누구든 `PDMUX_ROOT`를 export해야 한다는 사실이 CLAUDE.md에 아직 반영 안 됨, (ii) 테스트
fixture가 그 env var를 스스로 세팅하지 않아 "무엇을 export해야 통과하는지"가 트레이스백에만
있다. 정본 문서·테스트 fixture 정합은 이 세션의 범위 밖이라 **고치지 않고 보고만 한다.**

### (b) 실제 코드/데이터 결함 — 4건, **이 세션이 만든 게 아닌 pre-existing gap**

`workspace/engine-port/scripts/r2_eval/lambda_star.example.json`을 참조하지만
**저장소 어디에도 이 파일이 없다**(`git log --all -- '*lambda_star.example.json'` 무출력,
`.gitignore`에도 없음 — 커밋된 적이 없다). `test_lambda_star_per_shape.py`의 `EXAMPLE_TABLE`
상수는 이번 세션의 우발적 변경분(§ (c-1)의 PDMUX_ROOT diff)과 **무관하게 그 줄 자체는
`git diff` 상 불변**이었다 — 즉 이 결함은 이번 세션 이전부터 있었다.

```
ERROR: test_shipped_template_loads_and_is_entirely_unmeasured (test_lambda_star_per_shape.FailClosed...)
FAIL: test_generate_campaign_refuses_the_unmeasured_template (test_lambda_star_per_shape.ShellRefusals...)
FAIL: test_workloads_cli_optin_writes_the_caveats_into_the_header (test_lambda_star_per_shape.CliSurfaces...)
FAIL: test_workloads_cli_refuses_the_unmeasured_template (test_lambda_star_per_shape.CliSurfaces...)
```

**engine-porter 회부**: `benchmarks/pdmux_eval/lambda_star.py`가 문서화하는 형식대로
`scripts/r2_eval/lambda_star.example.json` fixture를 만들어 커밋해야 이 4건이 풀린다
(또는 테스트 쪽에서 픽스처를 스스로 생성하도록 바꾸거나). 내용 자체는 성능 수치가 아니라
스키마 예시이므로 이 세션에서 만들어도 "새 성능 판정"에 해당하지 않지만, **지시("절차가
실패하면 실패를 그대로 보고해라")에 따라 만들지 않고 회부만 한다.**

## 5. GPU가 있어야만 확인되는 것 (분리)

- §4(a)의 12건 전체 — `sgl_kernel` 컴파일 확장이 필요한 import 경로. 대여 GPU에서
  `sgl_kernel`이 정상 빌드된 환경(sm80/sm90, CLAUDE.md §B 기판 기준)을 확보한 뒤
  재실행해야 참/거짓이 갈린다.
- §4(c-2)의 17건 중, `PDMUX_DRY_RUN=1`이 서버 기동 자체는 건너뛰므로 이론상 venv만
  있으면 GPU 없이도 통과 가능해 보인다(`engine_bench_runner.sh`가 venv activate 이후
  실제로 `launch_server`를 실행하는지 여부는 미확인) — **단정하지 않음, venv를
  만들지 않았으므로 이번 세션에서 직접 확인하지 못했다.**
- 서빙 자체(부팅 스모크, correctness gate)는 이번 작업 범위 밖(CLAUDE.md의 로컬 GPU
  사용 금지 규정).

## 6. 새 성능 판정

0건. 이 문서는 테스트 통과/실패 개수와 그 원인 분류만 다룬다.
