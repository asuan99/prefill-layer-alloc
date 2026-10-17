# merge preflight 실행 스펙 평가 — CPU(`resourcespec-a100cpu`) 대 A100x1 (2026-10-17, engine-porter)

> 지위: 분석 기록. 커밋하지 않음. 측정 아님. GPU 실행 0, Job 제출 0, Workspace 기동 0(`pdmux-build`·`pdmux-probe` 모두 켜지 않음).
> 질문: "merge preflight를 CPU가 아니라 GPU에서 돌려야 하는 이유가 있는가?"
> 표기: **[사실]** = 파일·로그·바이트로 직접 확인. **[코드]** = 코드 읽기로 확인(실행 관측 아님). **[추론]** = 근거에서 유도했으나 관측 안 됨.
> 이 문서는 회수본 라벨 `PREFLIGHT_FAIL_NEW`를 바꾸지 않는다.

## 0. 결론

**GPU 고유 경로를 보증하려고 A100에서 preflight를 돌릴 이유는 없다.** 이유는 세 가지다.
1. 이 스위트에는 CUDA 커널을 실제로 띄우는 시험이 하나도 없다. GPU 관련 호출은 전부 fake나 mock이다.
2. `sgl_kernel` 로더는 cc 8.0에서도 CPU와 **같은 `sm100/common_ops` 파일**을 고른다.
3. merge 후 엔진 트리의 GPU 경로는 이미 A100에서 실행됐다. preflight POST arm의 엔진 digest는 M2g PASS 트리와 같은 `82a5fc5b…`이다. M3 Job 안에도 별도의 fail-closed gate(B0·ENGINE_DIGEST·xcheck)가 있다.

**다만 스펙이 판정에 영향을 주는 것은 사실이다.** 그 지점은 GPU 경로가 아니라 **CPU 처리량(타이밍)**이다.
- CPU 스펙에서는 `test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut`이 PRE에서 타이밍 때문에 실패한다. 그래서 판정기의 `CHANGED`(= FAIL_NEW에 합산)로 남는다.
- 분석서 F2·F3·F5를 모두 고쳐도 CPU 스펙에서는 라벨이 PASS가 될 수 없다.
- A100x1 노드(11 vCPU)가 이 문제를 없앤다는 증거는 없다. 오히려 GPU 1회차는 더 느렸고 실패·오류가 더 많았다(원인 미확인, 로그 소실).

**권고: (A) CPU 유지 + 재실행 전 사전 선언.**
- 타이밍 시험 처리 규칙과 결론 범위를 선언한다.
- 타이밍 문제를 규칙이 아니라 측정으로 풀고 싶다면 (C) 소형 A100x1 프로브를 붙일 수 있다(약 20–25분, GPU 사용률 ≈ 0 %). 다만 그 목적은 "11 vCPU에서 타이밍 시험이 통과하는가"이지 GPU 경로 보증이 아니다.
- (B) 전체 A100x1은 권고하지 않는다.

## 1. sgl_kernel 아키텍처별 로드 경로

| 항목 | 내용 | 등급 |
|---|---|---|
| 이미지의 sgl_kernel | `sglang_kernel-0.4.1+cu130` wheel, pin sha256 `9164b8fc…`(`env/docker/Dockerfile.sglang-runtime:25-29`) | [사실] |
| wheel 대조 | 공개 URL의 wheel 전체를 스트리밍해 sha256 `9164b8fc2c16…c5c0`를 얻었다. pin과 일치한다. HTTP range로 `load_utils.py`·`__init__.py`·`spatial.py`만 추출했다(CRC 검증). 셋 다 로컬 `prefill-alloc` env의 0.4.1과 바이트 동일하다. `spatial.py` sha `687898ee…`는 A100 Job의 b0 기록(`multigpu/m2g_pdmux-multigpu-313be0a-…/b0/b0_python.json` `sgl_kernel_spatial_sha256`)과 같다 | [사실] |
| 로더 분기 | `load_utils.py:60-68`: `cc == 90` → `sm90`, `cc is not None`(cc 80 포함) → `sm100`, `cc is None`(CPU) → `sm100`. wheel에는 `sm90/common_ops.abi3.so`·`sm100/common_ops.abi3.so` 두 개만 있다 | [코드] + wheel 목록 [사실] |
| CPU에서 로드된 변종 | `sm100`(cc None 분기). 회수본은 `sgl_kernel ok …/__init__.py`만 남겼다(`meta/arm_{pre,post}/sgl_kernel.txt`). `common_ops.__file__`은 기록되지 않았다 | 경로는 [코드], import 성공은 [사실] |
| A100(cc 8.0)에서 로드되는 변종 | 역시 `sm100`. **CPU와 같은 파일**이다 | [코드] |
| CPU와 A100의 실제 차이 | (i) libcuda 해석: CPU는 `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`의 forward-compat `libcuda.so.580.82.07`(`cpu_probe/…/env.txt`, `vessl/…/meta/CPU_ONLY.txt`), A100은 호스트 드라이버 `580.105.08`(`m2g_…313be0a…/b0/gpu.csv`) (ii) A100에서는 `_get_compute_capability()`가 `torch.cuda.current_device()`를 불러(`load_utils.py:21`) **sgl_kernel을 import하는 프로세스마다 CUDA context가 생긴다**. CPU에서는 생기지 않는다 (iii) `sm100/common_ops` 안에 sm80 커널 이미지가 있는지는 커널을 실행할 때만 드러난다 | (i)(ii) [사실]/[코드], (iii) [추론] |
| `spatial_ops` | `sgl_kernel/__init__.py`는 `spatial`을 지연 import한다(`create_greenctx_stream_by_value`·`get_sm_available` 래퍼). `spatial.py:4-9`는 `spatial_ops` 로드 실패를 **삼킨다**. 그래서 `import sgl_kernel` 성공은 green-context 확장의 로드를 보증하지 않는다. CPU 프로브는 `from sgl_kernel import spatial`까지 통과했다(`cpu_probe/…/import_spatial.txt`). 그러나 오류를 삼키는 구조라 이것도 `spatial_ops` 로드의 증거는 아니다 | [코드] + [사실] |

**질문에 대한 답:** 맞다. CPU PASS는 A100에서의 변종 로드나 드라이버 심볼 문제를 잡지 못하는 구조다. 그러나 **A100x1 preflight도 그것을 잡지 못한다.** 스위트는 `spatial`을 실제로 쓰지 않고 커널도 띄우지 않는다(§2). 이 층은 merge와 무관한 **이미지 속성**이다. sgl_kernel 바이트는 이미지에 있고 엔진 bundle에는 없다. 그리고 이 층은 같은 이미지 `1e0d335b…`의 A100 Job에서 이미 실행됐다.
- 이 이미지로 돈 Job: b5·λ0·E2·N2·N3·M2g x2/x4/x8 등 A100 Job 19개. 각 Job의 `vessl/*/meta/substrate.json` image digest가 같다.
- M2g B0 score: `PASS A100 cc8.0, 108 SM (torch and sgl_kernel), MIG off`(`m2g_…313be0a…/b0/score.txt`). 이 값은 `spatial.get_sm_available(0)` 실호출에서 나왔다(`scripts/vessl/substrate_probe_helper.py:196-199`).

## 2. CUDA 유무에 따라 갈리는 시험 (POST `e43749c9` 기준 전수 grep, PRE도 같은 구조)

grep 대상: `cuda.is_available|skipUnless|skipIf|nvidia-smi|pynvml|green context|cuGreenCtx|sgl_kernel.spatial|create_greenctx|get_device_capability|.cuda(|device=cuda|torch.cuda|SkipTest`. 대상 파일은 `tests/*.py` 52개와 시험이 호출하는 `src/`·`scripts/`·`results/multigpu/*.sh`다.

**`torch.cuda.is_available()`로 skip하거나 분기하는 시험은 0건이다.** skip 조건은 전부 아카이브, 런타임 트리, 모델 캐시, aiohttp, env의 유무다. CUDA 텐서를 만드는 시험(`.cuda()`, `device="cuda"`)도 0건이다 [사실, grep].

| 시험 | GPU 분기 형태 | CPU 회수본 | A100x1이었다면 | 등급 |
|---|---|---|---|---|
| `test_mem_telemetry_symmetry.…test_values_are_honest_when_cuda_is_absent` (CL-3 A_import9) | `if not torch.cuda.is_initialized():` 안에서만 None 단정(`tests/test_mem_telemetry_symmetry.py:215-218`) | `ok`(pre·post). CUDA 미초기화 → **강한 분기 실행** | sgl_kernel import가 CUDA를 초기화하므로(§1 (ii)) 단정 3개를 **건너뛰는 약한 분기**가 된다. 대신 `multiplexing_mixin.py:620` 이하의 실제 allocator 읽기가 돈다. 그 결과값은 단정하지 않는다 | [코드]. 실행 관측 없음 |
| `test_mem_telemetry_symmetry`의 나머지 reset/peak 시험 | `torch.cuda.is_initialized`·`reset_peak_memory_stats`를 mock | `ok` | 같다(mock) | [코드] |
| `test_multigpu_guard.TestDeviceAssert` 6건(merge 신규) | fake torch(`get_device_capability=(8,0)`)·fake `sgl_kernel.spatial`(`tests/test_multigpu_guard.py:451-482`) | post `ok` 6 | 같다(실제 GPU를 보지 않음) | [코드] |
| `test_worker_grad_guard`·`test_r2_admission_*`·`test_sticky_partition` 등 mixin 행동 시험 | `torch.cuda.stream`/`device`/`empty_cache` mock(`tests/pdmux_loop_fakes.py:315-317`, `test_worker_grad_guard.py:68-69`) | `ok` | 같다 | [코드] |
| `test_zamba2_instrumentation` 17건(CL-3 D_zamba2) | `torch.cuda.Event`를 mock·FakeEvent로 대체(`:260, :282`). skip 조건은 sglang zamba2 import 실패(`:244`) | `ok` 17(pre·post), skip 0 | 같다 | [사실]/[코드] |
| `test_green_readout` 15건 | 가짜 드라이버(`cuGreenCtxGetDevResource` fake, `:91`) | `ok` 15 | 같다 | [사실] |
| `test_mg_harness.JobScriptsDryRun` 5건 | `mg_job_common.sh:113-149`: `nvidia-smi`·`score-b0 --expected-gpus N`·`mg_substrate`·NVML 프로브(`assert n == EXPECTED_GPUS`) | 4 `ok`, 1 FAIL(`test_m3_dry_run_every_job`, HF_HOME 원인, 분석서 §2-(b)) | x1에서도 `ABORT_SUBSTRATE`가 난다. J1은 2 GPU를 요구하고 1 ≠ 2이며 NVML assert도 실패한다(`mg_substrate.py:43-46`). 결과는 같다. **x2 이상 A100 SXM 노드에서는 노트가 사라져 실패할 수 있다**(이 시험은 하드웨어에 대해서도 밀폐되어 있지 않다) | [코드] |
| `test_r2_correctness_ctx.TestCachedModelsResolveAsExpected` 2건 | `hf_cache` 유무로 skip | `ok`(모델 캐시 있음) | 같다(같은 cluster volume, §4) | [사실] |
| n3/e1 러너가 부르는 하위 스크립트 | `n3_capacity.sh` 안의 `python3 -c 'import torch; torch.cuda.get_device_properties(0)…'`, pynvml 호출 등 | CPU에서는 빨리 실패하고 `NA`를 쓴다 | GPU에서는 CUDA를 초기화한다(프로세스당 추가 비용) | [코드] |

CPU 회수본의 skip은 pre 41건, post 42건이다. 사유는 아카이브 부재 41건과 `MG_TEST_X_TOKENS` 미지정 1건이고, **GPU 사유는 0건이다**(로그 원문 집계).

**분석서의 "GPU가 없어서 결과(ok/FAIL)가 달라진 시험 0건"은 결과 수준에서는 맞다.** 그러나 경로 수준에서는 보완할 점이 둘 있다.
1. GPU에서 분기가 달라지는 시험은 1건(mem telemetry)이다. 그 방향은 **CPU 쪽이 더 강한 단정**이다.
2. CL-3 대상 32개 가운데 **GPU 경로를 실제로 타는 시험은 CPU에서든 A100에서든 0개**다. 전부 fake 또는 mock 설계다. 따라서 "CPU에서는 GPU 경로를 타지 않은 대상이 있는가?"의 답은 "있다. 그러나 A100에서도 타지 않는다"이다.

**merge가 바꾼 엔진 코드의 GPU 관련 부분** [사실, `meta/src_diff_pre_post.txt` + diff]
- `chunk_probe`·`holb_probe`·`telemetry`·`multiplexing_mixin`의 merge diff에는 새 `torch.cuda`·`sgl_kernel`·stream 호출이 없다. 해당 줄은 주석뿐이다.
- 유일한 GPU 인접 변경은 `pdmux_context_device_assert.patch`다. 자동 격자(`divide_sm`) 분기의 assert 한 줄이고, 기존 `torch.cuda.current_device()`를 재사용한다.
- 신규 `multigpu_guard.py`는 순수 Python이다(rank 이름 붙이기, TP fail-fast).

## 3. 타이밍·자원 의존

**사실**
- CPU 스펙: cgroup `cpu.max=350000/100000`(3.5 vCPU), nproc 48. 종료 시 `nr_throttled 16498 / nr_periods 93542`(`vessl/…/meta/cgroup_cpu.txt`)였다.
- A100x1 스펙: `cpu.max=1100000/100000`(**11 vCPU**), nproc 96. 근거는 N3·E2 A100x1 Job 6개의 `vessl/*/meta/cgroup_cpu.txt`이고 `vesslctl resource-spec list`도 CPU 11.0이다. x2/x4/x8은 22/44/88 vCPU다. CPU 전용 스펙은 3.5 vCPU 하나뿐이고, 다른 클러스터에 3.0이 있다.
- 이미지의 `sitecustomize.py:19-25`는 py3.14에서 **모든 python 프로세스가 `import torch`** 하게 만든다(`env/sitecustomize.py`).
- 벽시계 단정이 있고 컨테이너에서 실제로 도는 시험(아카이브 skip 제외)과 CPU 결과(pre1/post1/post2/pre2):

| 시험 | 단정 | CPU 결과 | 판정기 영향 |
|---|---|---|---|
| `test_e1_cap_vessl.TestBudgetEnforcer.test_overrunning_run_is_cut_at_the_deadline` | `wall < 50`(`:513`) | FAIL×4 (60.7 s) | 양 arm 같은 서명 → 라벨 영향 없음, 검증 공백 |
| `…test_equivalent_mutant_one_layer_removed_still_cuts` | `wall < 50`(`:539`) | FAIL×4 (60.3 s) | 같음 |
| `…test_control_both_kill_layers_removed_overruns` | `wall ≥ 60` | ok×4 | — |
| `test_n3_capacity.TestBudgetEnforcer.test_overrunning_unit_is_cut` (L2) | `wall<50` + FAKE_BENCH 정규식(`:720-724`) | FAIL×4. PRE는 타이밍(FAKE_BENCH 빈 문자열), POST는 L2 | **PRE와 POST 서명이 달라 `CHANGED`가 되고 `fail_new`에 합산된다**(`preflight_judge.py:113` `fail_new = sorted(new + changed)`) |
| `…test_control_both_kill_layers_removed_overruns` (L2) | `wall ≥ 60` | FAIL×4 (PRE 38–40 s, POST 16 s) | `N.N` 마스킹으로 "같음"에 흡수됨(분석서 §3-3) |
| `…test_equivalent_mutant_watchdog_removed_still_cuts` (L2) | `wall < 50` | pre ok, post FAIL | L2 hit |
| `…test_t_h7_*` 2건 | 타임아웃 마커 | ok×4 | — |
| `test_n3_capacity` `:1215-1232`(`wall<20`, `≥6`, `<30`) | | 분석 범위 밖, CPU에서 FAIL 목록에 없음 | — |
| `test_mg_harness` `:2738`(monotonic 차 < 120 s) | | FAIL 목록에 없음 | — |

**가장 중요한 귀결**
- 분석서 §5-2 반사실의 "+둘 다" 행은 NEW 1(mg_harness)과 CHANGED 1(overrunning)이었다.
- F2(HF_HOME 밀폐)로 mg_harness를 고쳐도 **CPU 스펙에서는 overrunning의 PRE 타이밍 실패 하나가 FAIL_NEW를 유지한다** [추론, 판정기 코드 + 4회 결정적 재현].
- F6(PRE에서 실패하는 L2 → `L2_UNVERIFIED` 구멍)을 넣으면 라벨은 UNVERIFIABLE이 된다. 어느 쪽이든 **CPU 스펙에서는 규칙을 바꾸지 않는 한 PASS에 도달할 수 없다**.

**GPU 노드에서 이 문제가 사라지는가: 근거 부족이고, 반대 방향의 단서가 있다**
- [사실] GPU 1회차(`job-qukdn58jv1ph`, A100x1, 같은 이미지, PRE `7b15214`, POST `e43749c`, 같은 도구 바이트): `suite pre rep1 rc=1 FAILED (failures=27, errors=9, skipped=41)`, 09:42:39 → 10:23:41(**2,462 s**). 같은 PRE arm의 CPU 결과는 failures 26, errors 6, skipped 41, **1,940–1,977 s**였다(`vessl/…/logs/target.log:12,18`). **GPU 노드가 약 27 % 느렸고 실패 +1, 오류 +3이었다.**
- [사실] 이 Job의 산출물 디렉터리는 object volume에 비어 있다(`vesslctl volume ls … --prefix results/pdmux-preflight-cl3-e43749c-20261016094057/`의 항목 0). `vesslctl job logs`는 34줄만 돌려준다. 그래서 어떤 시험이 더 실패했는지, 타이밍 시험이 통과했는지는 **알 수 없다**.
- [사실] post rep1은 10:23:41에 시작했고 Job은 11:06:02에 terminated(`vesslctl job show`)되었다. 기본 `PDMUX_PF_SUITE_TIMEOUT`은 3000 s다(`preflight_cl3.sh:49`). GPU Job env에는 이 값을 덮는 설정이 없었다.
- [추론] CPU post 2,862–2,873 s × 1.27 ≈ 3,640 s > 3,000 s. 그대로 두었다면 post가 suite timeout에 걸려 UNVERIFIABLE이 됐을 가능성이 높다.
- [추론] 느려진 원인 후보는 셋이다. (a) sgl_kernel·sglang을 import하는 하위 프로세스마다 CUDA context 생성(§1 (ii)) (b) GPU 모드에서만 도는 `job_entry.sh` 3단계의 clk sampler(`nvidia-smi -lms 100`)와 energy sampler (c) 노드 차이. 세 후보 모두 분해 근거가 없다.
- [추론] n3 러너는 deadline(≈20 s) 안에서 첫 unit 전까지 python 프로세스를 20개 넘게 띄운다(`n3_capacity.sh` 94–378행에 `python3` 28회. 일부는 SKIP_PREFLIGHT로 생략된다). 프로세스마다 sitecustomize의 torch import가 든다. 로컬의 warm torch import는 0.6–0.74 s다(28코어, py3.10 env, 참고값). 11 vCPU 컨테이너에서도 20 s 창에 들어온다는 보장은 없다. GPU 노드에서는 `torch.cuda.get_device_properties` 호출이 오히려 CUDA를 초기화한다.

## 4. 이번 판정 문제와 스펙의 관계

| 문제 | 스펙 의존? | 근거 |
|---|---|---|
| `PYTHONUNBUFFERED=1`로 생긴 출력 끼어쓰기(§2-(a) `test_plan_selftest`가 NEW로 분류됨) | **무관**. GPU 노드에서도 같다 | `job_entry.sh:309`(68f386b)와 `:281`(e43749c)의 `export HF_HOME=… PYTHONUNBUFFERED=1`은 4단계(target 실행)에 있다. cpu_only 분기(`fi # cpu_only`) **밖**이다. GPU Job의 커밋 e43749c에는 분기 자체가 없다 [사실] |
| ResourceWarning 끼어쓰기(§4 HOLE) | 무관 | 시험 코드의 `test_holb_probe.py:667-668`과 unittest 기본 경고 필터에서 나온다 [사실] |
| `HF_HOME` 누출(§2-(b)) | 무관 | 두 Job 모두 `clustervol-havdzigstmv7:/data`를 마운트했다(`vesslctl job show`, `vessl/…/job_show_at_fetch.txt:17`). `HF_HOME=/data/hf`도 같은 줄에 있다 [사실] |
| arm 경로 마스킹 결함(§3-1, 8건) | 무관 | `/data/runs/<job>/pf_work/{pre,post}` 경로는 두 스펙에서 같다 [코드] |
| 아카이브 부재 skip·실패 | 무관 | 번들 내용에서 비롯된다 [사실] |
| PRE conda `activate` 인자 문제(17건) | 무관 | 같은 이미지다 [사실] |
| **CPU에서만 생기는 문제** | e1·n3 타이밍 실패(§3). 판정에 주는 영향은 overrunning의 `CHANGED`이고 이것이 FAIL_NEW를 유지한다 | §3 |
| **GPU에서만 생기거나 바뀌는 문제** | (i) suite 약 27 % 느림 → 기본 3000 s timeout 위험 [사실+추론] (ii) PRE rep1 실패·오류가 CPU보다 1·3건 많음, 원인 미확인 [사실] (iii) mem telemetry 시험이 약한 분기로 바뀜 [코드] (iv) n3 러너 하위 단계가 CUDA를 초기화함 [코드] (v) x2 이상 스펙이면 mg dry-run 시험이 뒤집힘(x1은 무관) [코드] | §2, §3 |

## 5. preflight가 보증해야 하는 것 — 문언 대조

- **감사 판정서**(`handoff_review_2026-10-15/VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md:57`): 조건은 "**sgl_kernel이 import되는 환경**(VESSL Workspace 또는 첫 Job의 측정 전 preflight)"이다. GPU나 cc 8.0을 요구하는 문언은 없다. 대상은 §5 공백 (1)–(3), 즉 import 실패로 로컬에서 돌지 않은 시험이다.
- **known_failures v2 §5-5**(`MERGE_TRIAL_known_failures_2026-10-15.md:135`): "sgl_kernel이 정상 빌드된 환경(VESSL pdmux-build Workspace 또는 **A100 컨테이너 CPU 모드**)에서만 확인 가능". CPU 모드를 명시적으로 허용한다.
- **설계 §1**(`PREFLIGHT_merge_cl3_design_2026-10-15.md:15-17`): A100x1을 고른 이유는 "검증된 Job 경로"와 "시험이 실제로 도는 조건(cc8.0, sgl_kernel 정상 로드)이 G0·M3와 같다"였다. 같은 절에 "GPU가 꼭 필요하진 않다"고도 적었다. CPU 스펙을 기각한 사유는 "job_entry.sh가 GPU 전제라 미시험 경로"였는데, 이는 `68f386b`의 CPU-only 모드로 해소됐다.
  - 이 문서가 바로잡는 점: "cc8.0 조건이 같다"는 이 스위트에서 실질 내용이 없다. 시험이 cc 8.0 경로를 쓰지 않는다(§2). 같은 절의 소요 추정 "≈1 시간"은 실측에서 3배 이상 빗나갔다(§6).
- **M3 rev12**(`multigpu/PREREG_MULTIGPU_rev12_2026-10-15.md:201`)는 "VESSL 컨테이너에서"라고만 적어 GPU를 요구하지 않는다. 금지 문안은 두 개다. "preflight PASS가 M3 실행 바이트를 보증한다"(`VERDICT_m3_rev12_harness_r2_2026-10-16.md:78`, MGH12-7)와 "preflight 통과를 merge 회귀 부재의 근거로"(`PREREG…:214, :271`, C-4)다. **preflight는 처음부터 기판 보증 수단으로 설계되지 않았다.**
- **GPU 고유 경로를 따로 덮는 장치** [사실]
  - (a) M2g PASS 표 J1·J2·J4·J8은 A100에서 측정했다. J1(`m2g_pdmux-multigpu-313be0a-…`)에는 N1 agnostic·d44·fused·fused_sym boot가 있고, `ENGINE_DIGEST_CHECK.txt`가 `82a5fc5b… match=True`다.
  - (b) **preflight POST arm 매니페스트(27행)의 pathless digest = `82a5fc5b…`**다(`engine_digest.py check --manifest meta/arm_post/manifest.sha256 --sealed-file results/multigpu/ENGINE_DIGEST.txt` → `match=True`, 이 세션 실행). preflight가 검사한 엔진 바이트와 M2g가 A100에서 돌린 엔진 바이트가 같다.
  - (c) M3 Job 자체가 셀보다 먼저 `mg_job_common.sh`의 B0 score(`get_sm_available_108`, cc 8.0, `sgl_kernel_spatial_import`), `mg_substrate`, NVML, `ABORT_ENGINE_DIGEST`를 거친다. 하나라도 어긋나면 abort한다(`:113-160`). 그 뒤 xcheck가 fused_sym·agnostic의 GPU 출력 token을 대조한다(rev12 §19-2 U7).
- **조건**
  - 위 (b)는 **최종 POST의 엔진 digest가 `82a5fc5b…`로 유지될 때만** 성립한다. 최종 POST가 `src/`를 바꾸면 새 엔진 바이트의 GPU 경로는 M2g가 덮지 않는다. 이 경우는 rev12 §19-6이 이미 "M2g PASS 유효 범위를 사용자에게 다시 묻는다"로 넘긴다.
  - G0는 계측 패치가 `src/multiplex`를 바꾸므로 이 preflight PASS가 원래 무효다(설계 §6). G0 자신의 GPU correctness gate가 그 몫이다.

## 6. 권고

| 안 | 내용 | 소요(근거) | GPU 사용률 | 판정 |
|---|---|---|---|---|
| **(A) CPU 유지 + 사전 선언** | 지금 경로. 재실행 전 아래 선언을 설계서 개정으로 고정하고 claims-auditor가 감사한다 | 실측 9,730 s ≈ **2.7 h**(ABBA 2회). 분석서 F2/F3/F5/F6 반영 뒤에도 같은 규모 | GPU 0장(해당 없음) | **권고** |
| (B) A100x1 전체 | 같은 스크립트를 x1에서 실행 | [추론] pre 2 × ≈2,460 s + post 2 × ≈3,640 s ≈ 12,200 s + 준비 ≈2 min → **≈3.4 h**. `PDMUX_PF_SUITE_TIMEOUT` ≥ 5,400이 필요하다(기본 3000이면 post timeout) | 1장 할당, **SM 사용 ≈ 0 %**(스위트가 띄우는 커널 0개, 하위 프로세스의 CUDA context 생성뿐). ≈3.4 GPU-h를 거의 놀린다 | 비권고. GPU 경로 보증이 늘지 않는다(§2). 더 느리고, 원인 미확인 실패·오류 +4가 관측됐다. 타이밍 해소 여부도 미지 |
| (C) 혼합 | (A) 전체 + 소형 A100x1 Job. 내용은 POST·PRE arm 트리에서 ① `SGLANG_…`/logging debug로 `common_ops.__file__`·cc 기록 ② `spatial.get_sm_available`과 green-ctx 쌍 1개 생성 ③ `test_mem_telemetry_symmetry`·`test_multigpu_guard`·`test_mg_harness.JobScriptsDryRun`·`test_n3_capacity.TestBudgetEnforcer`·`test_e1_cap_vessl.TestBudgetEnforcer`를 짝으로 2회 | (A) 2.7 h(GPU 없음) + x1 ≈ 1.5 min 준비(GPU 1회차 실측: 시작 → suite 시작 91 s) + 시험 ≈ 15–20 min → **x1 ≈ 20–25 min** [추론] | x1 구간 1장, SM ≈ 0 %(green-ctx 생성 순간뿐), ≈0.4 GPU-h | 조건부. ①②는 M2g·M3 B0가 이미 덮어 **추가 정보가 거의 없다.** ③의 타이밍 시험만 새 정보다("11 vCPU에서 PRE overrunning이 통과하는가"). 판정기 설계상 부분집합 실행은 INVALID라 PASS 산출물이 될 수 없다. 보조 기록으로서의 역할을 미리 선언해야 한다 |

**(A)의 사전 선언에 들어갈 것** (데이터를 본 뒤 정하는 규칙이므로 자유 표면이다. 감사 최대 2회를 거쳐야 한다.)
1. **결론 범위:** "CPU 컨테이너(3.5 vCPU), 이미지 `1e0d335b…`, compat libcuda, sgl_kernel `sm100` 변종(cc None 분기, cc 8.0과 같은 파일), CUDA 커널 실행 0." PASS는 GPU 실행 경로에 대해 말하지 않는다. GPU 경로는 M2g PASS(digest `82a5fc5b…`)와 M3 Job 안의 B0·ENGINE_DIGEST·xcheck가 덮는다. 단 최종 POST의 엔진 digest가 같을 때만이다.
2. **타이밍 집합:** e1 TestBudgetEnforcer 2건과 n3 TestBudgetEnforcer 3건(L2)을 id로 고정하고 처리 규칙을 정한다. 선택지는 둘이다.
   - (가) F6 방향(fail-closed): PRE에서 실패한 L2는 `L2_UNVERIFIED` 구멍으로 처리한다. CPU에서는 PASS에 도달할 수 없음을 받아들이고 UNVERIFIABLE을 사용자에게 올린다.
   - (나) "CPU 스펙 기지 실패"로 판정에서 제외하고 공백으로 공시한다. 이쪽이 더 느슨해서 감사 부담이 크다.
   - 어느 쪽이든 **재실행 전에** 고정한다. CPU 4회 결정적 재현이 이미 있으므로 사후 선택으로 보이지 않게 이 문서의 수치를 근거로 단다.
3. 분석서 F2·F3·F5와 함께 한 번에 개정한다. 이 선언은 gate 경로를 바꾸지 않는다.

**(C)를 고를 조건:** 사용자가 2-(가)·(나)의 규칙 변경 대신 측정으로 풀기를 원할 때다. x1 프로브에서 PRE overrunning이 통과하면 그 시험의 공백은 (B)로만 닫힌다. 그때 (B)를 다시 검토하되 timeout과 원인 미확인 +4를 먼저 해결해야 한다. 통과하지 못하면 (A)의 선언으로 간다.

## 7. 미확인 항목

1. GPU 1회차(PRE rep1)에서 CPU보다 많았던 실패 1건·오류 3건의 정체. 로그가 소실됐다(volume 디렉터리 비어 있음, job logs 34줄).
2. GPU 노드가 27 % 느린 원인((a) CUDA context, (b) 샘플러, (c) 노드 차). 분해하지 않았다.
3. 11 vCPU에서 e1·n3 타이밍 시험의 통과 여부. 관측이 없다.
4. CPU에서 실제로 로드된 `common_ops.__file__`. 로더 코드상 `sm100`이지만 로그에 기록되지 않았다. `spatial_ops.abi3.so`가 CPU에서 로드됐는지도 모른다(오류를 삼키는 구조).
5. `sm100/common_ops.abi3.so` 안에 sm80 SASS·PTX가 있는지. 바이너리를 검사하지 않았다. A100 Job들의 서빙 성공으로 간접 확인될 뿐이다.
6. 최종 POST(재봉인 뒤)의 엔진 digest가 `82a5fc5b…`로 유지되는지. 아직 POST가 정해지지 않았다.
7. CPU의 e1 실패(wall ≈ 60 s = FAKE)가 정확히 어떤 기전인지. 러너는 `CAPT ≥ 1`을 보장한다(`e1_cap_vessl.sh:299`). 그런데도 컷이 일어나지 않은 이유는 분해하지 않았다. 분석서 §5-1의 시작 비용 설명도 추론이다.
