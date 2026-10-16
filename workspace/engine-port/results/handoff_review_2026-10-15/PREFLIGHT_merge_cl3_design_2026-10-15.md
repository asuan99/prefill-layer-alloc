# Merge preflight (CL-3 미검증 시험) 설계 — G0·M3 공용 1회 (2026-10-15, 초안)

> 지위: 설계 + 스크립트 초안. VESSL 제출 0, Workspace 기동 0, GPU 실행 0, 커밋 0. 측정이 아니라 **정확성 프로브**다. 성능 수치·판정 없음.
> 근거: `VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md` §6 조건 1, `MERGE_TRIAL_known_failures_2026-10-15.md`(v2) §3-2·§5, 운영 모델 §1(측정=Job, Workspace=bring-up).
> **개정 r1 (2026-10-16)**: M3 rev12 감사 1회차 R-5(`../multigpu/VERDICT_m3_rev12_draft_r1_2026-10-15.md`(이하 VR1) §5)와 사용자 결정(`DECISIONS_Q1_consolidated_2026-10-15.md` §4)에 맞춰 §2·§3·§5·§6·§7·§8·§9를 고쳤다. 요지: **POST = 공용 loadgen·runner 수정 커밋**(`dc06c870` 고정 아님, PRE `7b15214` 그대로), 공용 조건 대조 경로 5 → 7(+`benchmarks/pdmux_eval`, M3는 +`scripts/multigpu`), venv 후보 (c) 허용. 스크립트(`preflight_cl3.sh`) 갱신은 이 문서 밖(§9).
> **개정 r2 (2026-10-17)**: 사용자 결정(`DECISIONS_Q1_consolidated_2026-10-15.md` §8 "권고대로 진행")에 따라 실행 스펙을 **CPU(`resourcespec-a100cpu`)**로 바꾸고(§1), 판정기를 개정했다(§4·§5·§10: F3 구조화 상태, F5 arm 경로 마스킹, F6 PRE 실패 L2 fail-closed, CPU 스펙 기지 타이밍 5건 사전 선언, 결론 범위 선언). 근거 `../multigpu_verdicts/PREFLIGHT_FAILNEW_ANALYSIS_2026-10-16.md` §7(F1–F9), `../multigpu_verdicts/PREFLIGHT_SPEC_ASSESSMENT_2026-10-17.md`. **런 `pdmux-preflight-cl3-68f386b-20261016114954`의 라벨 `PREFLIGHT_FAIL_NEW`는 기록 그대로이며 개정 판정기로 재판정하지 않는다** — 그 로그는 시험 fixture로만 쓴다(§10-4). r2는 이 런의 데이터를 본 뒤 정한 규칙이라 **자유 표면**이고 claims-auditor 감사(최대 2회)가 필요하다.
> 파일: `merge_trial_tools/preflight_cl3.sh`(진입), `preflight_judge.py`(판정), `preflight_judge_selftest.py`(변이 11건; r2에서 53건), `preflight_cl3_l2_ids.txt`(L2 12), `preflight_cl3_targets.txt`(CL-3 대상 31 항목 = 9 모듈 + 4 id + 17 id + zamba2).

## 1. 실행 수단 — 짧은 Job 1개, Workspace 아님 (r2: 스펙 = `resourcespec-a100cpu`)

**r2 결정 (2026-10-17, 사용자 §8 F7)**: 실행 스펙은 **CPU(`resourcespec-a100cpu`, 3.5 vCPU)**다. 아래 r1 표의 CPU 스펙 기각 사유("`job_entry.sh`가 GPU 전제")는 `68f386b`의 CPU-only 모드로 해소됐고, A100x1 채택 근거였던 "시험이 도는 조건(cc8.0, sgl_kernel 정상 로드)이 G0·M3와 같다"는 이 스위트에서 실질 내용이 없다 — 스위트는 CUDA 커널을 띄우지 않고(GPU 관련 호출은 전부 fake/mock), `sgl_kernel` 로더는 cc None(CPU)과 cc 8.0 모두에서 같은 `sm100/common_ops`를 고른다(평가서 §1·§2). GPU 고유 경로는 M2g PASS(엔진 digest `82a5fc5b…`)와 M3 Job 내부 gate(B0·ENGINE_DIGEST·xcheck)가 덮는다(평가서 §5; 최종 POST 엔진 digest가 같을 때만). 대가: CPU 처리량 때문에 벽시계 단정 시험 5건이 결정적으로 실패한다 → §10-3 사전 선언. 실측 소요는 ABBA 2회 9,730 s(≈2.7 h; 평가서 §6) — §7의 "≈1시간"은 r1 추정이며 빗나갔다. 아래 표는 r1 판단의 기록이다(대체됨).

| 선택지 | 판단 |
|---|---|
| `pdmux-probe` Workspace | 기각. 가변 환경(ssh 세션은 이미지 ENV 미상속, 운영 모델 §9), 산출물이 DONE/SHA256SUMS 경로를 안 탄다, 사람 손이 들어간 환경에서 낸 "PASS"는 감사 근거가 약하다. 이 preflight는 G0·M3 제출의 **게이트**라서 재현 가능한 경로가 필요하다. |
| `pdmux-build` CPU Workspace | 기각(조건부). `sgl_kernel/load_utils.py`는 GPU 없으면 `sm100` 변형을 로드하는 CPU 경로가 있어 import는 될 수 있으나(로컬 설치본 확인), 이미지(`1e0d335b`, libnuma1 포함)에서 CPU만으로 import되는지는 **미실측**이다. 과거 CPU 회귀(B2)의 errors 9는 libnuma 이전 이미지였다. 또 Workspace는 위와 같은 사유로 게이트 근거가 약하다. |
| CPU 스펙 Job(`resourcespec-a100cpu`) | 기각. `job_entry.sh`가 `nvidia-smi` 도장·`clk.csv`·substrate.json을 GPU 전제로 만든다. CPU 스펙에서는 미시험 경로. |
| **A100x1 Job** | **채택.** 검증된 경로(B5 이래 fetch.sh까지 통과), 이미지 digest·커밋 bundle·DONE+sha256로 감사 가능, 시험이 실제로 도는 조건(`cc8.0` 노드, `sgl_kernel` 정상 로드)이 G0·M3 Job과 같다. 부수 이득: merge된 `job_entry.sh`(로그 미러·heartbeat)가 VESSL 컨테이너에서 처음 한 번 돈다(증거가 아니라 부수 관찰). |

GPU가 **꼭** 필요하진 않다(시험 대부분은 CPU 코드, `PDMUX_DRY_RUN=1`로 서버를 띄우지 않는다). GPU 점유는 "검증된 Job 경로를 쓰는 대가"다. 사용률은 ~0 %로 예상(아래 §7).

## 2. 전·후 두 트리 (한 컨테이너 안)

**PRE·POST 정의 (r1)**: PRE = `7b15214b…`(main, merge 전 — 그대로). **POST = 공용 loadgen·runner 수정 커밋**(M3 rev12 R-2·R-3, G0 공용, `<sha 미정>`, engine-porter 구현 중). merge `dc06c870`은 POST의 조상이어야 한다(스크립트의 `merge-base --is-ancestor PRE POST` 검사와 함께). preflight는 그 커밋 전에는 제출하지 않는다(VR1 R-5: runner 변경 전 PASS가 바뀐 파일을 덮지 않는 구멍). 그래서 POST−PRE 차이는 merge분 + loadgen·runner 수정분이며, 두 수정분에서 생긴 L2 밖 새 실패도 이 짝 대조가 잡는다.

`job_entry.sh`가 이미 `/opt/pdmux/sglang_engine_dev`를 **Job 커밋(HEAD)**으로 sync했다. 이 트리를 그대로 두 번 쓰지 않는다. arm마다:

1. `git worktree add --detach $WORK/<arm>/prefill-layer-alloc <sha>` (`WORK=/data/runs/<job>/pf_work`, 종료 시 삭제). `hf_cache -> /data/hf` 링크. `<arm>` = `pre`(`7b15214b…`) / `post`(**loadgen·runner 수정 커밋**, `PDMUX_PF_POST`로 전체 sha 지정; 스크립트 기본값 `dc06c870`은 쓰지 않는다).
2. dev tree = `cp -a` of `dirname($SGLANG_ENGINE_DEV)` → `$WORK/<arm>/sglang_engine_dev/python`.
3. **pre만**: 복사본은 post 성격(HEAD sync)이라 merge-only 효과를 뺀다 — `pdmux_context_device_assert.patch` 역적용 + `multigpu_guard.py` 삭제. 그 뒤 `pdmux_context.py` sha가 로컬 main dev tree(무패치)의 `b4e6ff1a…`와 같고 패치가 다시 정방향으로 적용되는지 확인한다. 다르면 UNVERIFIABLE(이미지 upstream이 로컬과 다름).
4. 각 arm 자신의 `install_runtime.sh`(sync + manual edits) → manifest(pre 25행, post 27행) → `sha256sum -c`.
5. **sbatch 구동 시험의 `SGLANG_ENGINE_DEV` 덮어쓰기 해법**: `test_campaign_provenance`·`test_lambda_star_per_shape`·`test_r2_eval_runner`는 `r2_eval.sbatch`를 호출하고 그 안의 `sync_engine_tree.sh`가 dev tree에 src를 설치한다(known_failures §1-1 부작용). arm별로 **`PDMUX_ROOT=$WORK/<arm>`, `PDMUX_PROJECT_ROOT` unset, `SGLANG_ENGINE_DEV`·`PYTHONPATH`=그 arm 복사본**으로 돌리면 sbatch가 읽는 `project_root`(=`$PDMUX_ROOT/prefill-layer-alloc`)가 그 arm 자신의 worktree이므로 재설치는 같은 바이트의 no-op이다. 환경의 `PDMUX_*`는 `env -u`로 전부 걷어 낸다(`PDMUX_CAMPAIGN`·`PDMUX_PROJECT_ROOT`가 시험 입력을 바꾸지 않게). 시험 전후로 multiplex/*.py + scheduler.py 해시를 비교해 변하면 INVALID.
6. 검사: 두 arm 트리의 `diff -rq`가 `git diff PRE POST -- src`로 설명되는 파일에 한정. POST가 `src`를 건드리지 않으면 이 집합은 `dc06c870` 때와 같은 6개 파일(chunk_probe, holb_probe, multigpu_guard, multiplexing_mixin, telemetry, pdmux_context)이고, 건드리면 허용 목록을 그 `git diff`로 다시 낸다(스크립트가 이미 `meta/src_diff_pre_post.txt`로 낸다). 그 밖이면 INVALID. `import sglang`이 arm 복사본에서 오는지(이미지의 editable 설치가 아니라) 확인. `import sgl_kernel` 실패 시 시험을 돌리지 않고 UNVERIFIABLE.

★로컬 검증 완료(CPU, `--dry-run-trees`, 로컬 `sglang_engine_dev_multigpu`를 기반으로, **POST = `dc06c870` 때의 기록** — 새 POST에서 다시 돌려야 한다): pre 재구성 후 `pdmux_context.py` sha = `b4e6ff1a…` 일치, 정방향 재적용 가능, manifest 25/27행, 두 트리 차이 = 위 6개 파일 정확히, 미설명 차이 0. 남은 INVALID는 로컬에 `sgl_kernel`이 없는 것뿐(의도된 fail-closed).

## 3. venv 17건 — 코드에서 확인한 사실과 처리

- 시험 17건(campaign_provenance 10 / lambda_star SbatchReadsTheField 3 / r2_eval_runner TestSbatchPathResolution 4)은 `engine_bench_runner.sh:115 source "${venv_dir}/bin/activate"`에서 멈춘다. `venv_dir="${PDMUX_VENV:-${pdmux_root}/sglang_engine_venv}"`(:59) — **`PDMUX_VENV`가 공식 env 지정**이고 시험은 `dict(os.environ)`을 그대로 넘기므로 시험 코드 수정 없이 닿는다. `PDMUX_DRY_RUN=1`이라 activate 뒤 서버 인자를 출력하고 멈춘다.
- 이미지에는 `sglang_engine_venv`가 없다(Dockerfile: 이미지 python에 직접 pip). 후보를 `set -euo pipefail`(러너와 같은 셸 옵션)로 **먼저 프로브**해 첫 통과를 쓴다: (a) `PDMUX_VENV`(--env로 준 값) (b) 이미지 python의 `sys.base_prefix`(conda base면 `/opt/conda/bin/activate`) (c) `python -m venv --system-site-packages`로 만든 실제 venv. 결과는 `meta/venv_mode.txt`(`env|image_python_prefix|created_system_site|none`)와 `venv_probe.txt`.
- **(c) 허용 [결정, 사용자 2026-10-16, DECISIONS §4]**: (c)는 "정당한 env 지정"의 경계에 있다(환경 제공이지 시험·코드 변경은 아니지만, 원래의 `sglang_engine_venv` 내용물과 다르다). 사용자가 허용했으므로 스크립트는 (a) → (b) → (c) 순서로 시도하고, 쓰인 모드를 `meta/venv_mode.txt`에 반드시 기록한다(PASS 결론의 범위에 venv 모드가 붙는다). 세 후보가 모두 실패하면 17건은 **검증 불가**로 남는다(만들지 않은 activate 스텁은 쓰지 않는다 — 항등식 위험).
- 판정상 17건은 post에서 전부 `ok`여야 "검증"이다. 전·후 똑같이 다른 이유로 실패하면(예: 러너가 activate 뒤 필요로 하는 모듈 부재) 대조는 일치하지만 **구멍**으로 계산한다 → UNVERIFIABLE.

## 4. 대조 기준 (VESSL 안에서 짝지음, 로컬 pin 미사용)

- 단위 = (시험 id, 서명): `sigcmp.parse_multi`의 서명(예외 유형, 마스킹 메시지, 핵심어, DRIFT 다이제스트). `preflight_judge.py`는 `pin_post.json`을 읽지 않는다.
- **신규 실패** = post에서 실패하나 pre에서 같은 서명으로 실패하지 않고 L2도 아닌 id. 양쪽 실패인데 서명이 다른 id도 신규(같은 id 다른 원인). L2 = "pre ok → post fail" 12 id(목록 파일; id 단위 소속. 서명은 pre/post 짝만 본다 — L2의 원인이 다른 것으로 바뀌는 경우는 이 판정이 못 가른다, 한계).
- 반복: 기본 2회 ABBA(pre, post, post, pre). 같은 arm 반복 간 실패 집합·서명이 다르면 flaky → UNVERIFIABLE.
- 구멍(coverage): 31 대상 항목이 전부 post에서 실행·통과해야 한다. skip(큰따옴표 skip 포함 — `sigcmp`는 단일따옴표만 인식해서 판정기가 원문을 따로 스캔함)·import 오류·실패는 구멍.
- **r2 (§10)**: 상태는 구조화 러너의 JSON이 정본이고 텍스트 상태(엄격 파서)·실패 블록·`Ran N`과 교차 대조한다(불일치 = INVALID). 서명 마스킹에 arm 루트(`…/pf_work/{pre,post}`)를 넣는다. L2 id는 PRE 상태가 정확히 `ok`일 때만 재현으로 센다 — PRE에서 실패·skip·미판독이면 구멍 `L2_UNVERIFIED`(위 "L2 원인이 바뀌는 경우" 한계 중 PRE 실패 쪽을 닫는다). CPU 스펙에서만 사전 선언 5건을 판정에서 뺀다.

## 5. 판정 라벨 (우선순위 순)

| 라벨 | 조건 | rc |
|---|---|---|
| `PREFLIGHT_UNVERIFIABLE` | INVALID(트리 재구성 실패, sgl_kernel import 실패, 로그 잘림/타임아웃, 반복 불일치, L2 0건 재현, 시험 중 트리 변경, smoke 부분집합) | 3 |
| `PREFLIGHT_FAIL_NEW` | INVALID 없음 + L2 밖 신규 실패 ≥1 → G0·M3 제출 중단 | 1 |
| `PREFLIGHT_UNVERIFIABLE` | 신규 0이지만 구멍 ≥1 | 3 |
| `PREFLIGHT_PASS` | 신규 0, INVALID 0, 구멍 0 | 0 |

**r2 추가 INVALID**: 구조화 상태 JSON 부재(`--require-structured`)·판독 불가·JSON/텍스트 상태 불일치·실패 블록과 JSON 실패 집합 불일치·JSON ran ≠ `Ran N`, CPU 플래그인데 GPU가 보임(스펙 모순), 선언 id가 CL-3 대상과 겹침. **r2 추가 구멍**: 텍스트 상태 미판독(JSON이 없을 때), `L2_UNVERIFIED`(PRE에서 `ok`가 아닌 L2 id).

판정기 자체는 합성 로그 변이 11건으로 검증했다(`preflight_judge_selftest.py`: 통제군 PASS, 신규 실패 2종→FAIL_NEW, skip·import 오류·잘린 로그·flaky·양쪽 동일 venv 실패·L2 미재현→UNVERIFIABLE, 신규>구멍 우선순위, L2 1건 부재→PASS). 로컬 실제 로그(pre 877, post 1265)에 돌리면 `PREFLIGHT_UNVERIFIABLE`(구멍 30/31)이 나온다 — 이 로컬 공백을 있는 그대로 재현.
**PASS의 의미**: 이 31 항목과 전체 스위트에서 L2 밖 신규 실패가 없다는 것. "회귀 0건"(C-4)이라고 쓰지 않는다. 결론 범위는 (PRE sha, POST sha, 이미지 digest, 도구 sha, venv 모드)에 묶인다.

## 6. G0·M3 공용 조건 (요청 ③, r1 개정 — VR1 R-5)

한 번의 PASS가 G0·M3를 덮으려면 제출 때 셋 다 참이어야 한다(M3는 rev12 §19-6 gate (i)–(iii)과 같은 식):
(i) `PREFLIGHT_PASS` 산출물·sha256 커밋, 산출물의 merge 트리 digest = 그 PREREG 등록 digest (ii) preflight Job과 제출 Job의 **이미지 digest 동일**
(iii) `git diff --stat <POST> <제출 커밋> -- <대조 경로>`가 **빈 출력**. POST = §2의 loadgen·runner 수정 커밋.

**대조 경로** (경로는 모두 `workspace/engine-port/` 아래):

| 경로 | G0 | M3 | 근거 |
|---|---|---|---|
| `src`, `tests`, `scripts/bootstrap`, `scripts/r2_eval`, `env` | 예 | 예 | r0의 5경로(VR1 R-5 원문) |
| `benchmarks/pdmux_eval` | **예(추가)** | **예(추가)** | runner가 `python -m pdmux_eval.trace_loadgen`으로 부르고(`engine_bench_runner.sh:116, 226` [코드]) R-2 loadgen 수정이 여기 들어간다. 스위트의 여러 시험(`test_benchmark_tools`·`test_campaign_provenance`·`test_lambda_star_per_shape`·`test_n3_capacity` 등)이 이 패키지를 import한다. 빼면 POST 뒤 loadgen 변경이 PASS를 그대로 받는다 — VR1 Q9가 runner에서 찾은 구멍과 같은 구조이며 감사가 놓친 경로다 |
| `scripts/multigpu` | 아니오 | **예(추가)** | `mg_loadgen.py`(trace_loadgen 래퍼)·`mg_runner.sh`가 M3 경로이고 `tests/test_mg_harness.py`가 import·실행한다. G0는 쓰지 않는다 |
| `results/multigpu/`(M3 하네스) | — | **아니오** | preflight **뒤** 재봉인 단계에서 바뀐다(순서 [결정]). 넣으면 gate가 항상 거짓. 그 검증은 하네스층 감사·재봉인 sha의 몫이고 PASS는 이 하네스에 대해 말하지 않는다 |

- 결과적 순서 제약: 제출 전에 대조 경로를 고치는 변경(예: M3 하네스 시험이 `tests/`에, `mg_loadgen.py` 변경이 `scripts/multigpu`에 들어감)은 POST 커밋에 함께 넣거나, 아니면 그 커밋을 새 POST로 preflight를 다시 돈다(fail-closed). 선택은 M3 rev12 U13.
- G0가 `src/multiplex`를 고치면(계측 패치 I-2–I-5) 이 PASS는 G0에 무효이고 G0 변경분은 별도 짝 대조가 필요하다(r0와 같음). G0가 `benchmarks/pdmux_eval`에 분석기를 더해도 (iii)이 거짓이 된다 → 분석기 파일은 대조 경로 밖에 두거나 POST 전에 커밋한다.
- 스크립트의 `meta/post_vs_job_head.txt`(`preflight_cl3.sh:136-137`)는 지금 5경로만 적는다 → 7경로로 늘려야 한다(§9). 기록일 뿐 판정이 아니다(arm은 POST에서 재구성).

## 7. 산출물·회수·소요

- 제출: `launch.sh --campaign preflight_cl3 --target workspace/engine-port/results/handoff_review_2026-10-15/merge_trial_tools/preflight_cl3.sh --env PDMUX_PF_POST=<POST 전체 sha> [--env PDMUX_PF_REPS=2] [--env PDMUX_VENV=...]` dry-run → 승인 → `--submit`. 도구 파일 **커밋 선행 필요**(bundle=HEAD).
- 산출물(`results/preflight_cl3/<job>/` → fetch가 `results/` 원위치로 병합): `PREFLIGHT_LABEL`, `judge.json`, `logs/<arm>.rep<i>.log`·`.rc`, `meta/`{provenance, arm_*/(manifest, install_runtime.log, reverse_patch.log, sglang_path, sgl_kernel), arm_tree_diff*, allowed_tree_diff, venv_mode·venv_probe, judge_selftest, lists_check, post_vs_job_head, elapsed}. Job 메타는 `results/preflight_cl3/vessl/<job>/`(substrate·clk·cost). 회수 = `fetch.sh preflight_cl3 <job>`: DONE + `sha256sum -c` + 병합 충돌 0. 스크립트는 판정 rc로 종료하므로 PASS 외에는 Job이 failed로 보이나 DONE·SHA256SUMS는 항상 쓰인다.
- 소요(추정, 미실측): 부팅·이미지 pull·install 5–10분 + worktree 2개·dev tree 복사 3–5분 + 스위트(로컬 기준선 598건 296초; 현재 1265건, sbatch 구동 시험은 개당 최대 900초 timeout) arm·회당 8–15분 × 4회 = 35–60분 → **총 ≈ 1시간, ≈1 GPU-h**. `PDMUX_PF_REPS=1`이면 ≈ 40분.
- GPU 사용률: 할당 1장, 시험 중 사용 ≈ 0 %(CPU 단일 프로세스, 서버 미기동). 더 작은 GPU 스펙은 없고(x1이 최소), GPU 없는 스펙은 §1처럼 미시험 경로라 x1이 적정하다. **(r1 기록. r2: CPU 스펙, GPU 0장 — §1. 실측 소요 9,730 s.)**

## 8. 미해결 위험

1. **venv 17건**: 이미지 base conda의 `activate`가 `set -u`에서 실패할 수 있다(프로브가 드러냄). (c)는 허용됨(2026-10-16). (a)–(c)가 모두 실패하면 17건 검증 불가.
2. **zamba2_instrumentation 런타임 클래스**: `sgl_kernel`이 풀려도 hf_cache에 Zamba2-2.7B가 없으면(스테이징은 Nemotron-Nano만) 다른 이유로 skip될 수 있다 → 구멍 → UNVERIFIABLE. 가능성 높음.
3. 컨테이너에서 처음 실행: `git worktree`가 `/data` 볼륨에서 도는 속도·용량(트리 2개 ≈ 1.5 GB), sbatch 구동 시험의 컨테이너 내 동작. 첫 Job 실패 시 수정 후 새 job name 재제출(자동 재시도 없음).
4. pre 재구성의 기준 sha(`b4e6ff1a…`)는 로컬 main dev tree 값. 이미지 upstream이 다르면 UNVERIFIABLE(원인 추적 필요).
5. 타이밍 시험(`TestBudgetEnforcer` 등)이 공유 노드 CPU에서 흔들리면 flaky 판정으로 UNVERIFIABLE. ABBA로 감지만 하고 보정하지 않는다. **(r2: CPU 스펙에서 흔들림이 아니라 결정적 실패로 실현됐다 — 4/4 반복. 5건을 §10-3으로 사전 선언. 선언 밖 타이밍 실패는 그대로 잡는다.)**
6. L2 서명 동일성은 강제하지 않는다(§4 한계).
7. 판정 PASS여도 CL-3 밖(예: 이 스위트가 안 닿는 경로)은 말하지 않는다.

## 9. 다음 행동 (모두 사용자 승인 필요)

(1) 도구 6개 커밋(`d138e7f`, 완료) → (2) venv (c) 허용(결정 2026-10-16, 완료) → (3) 공용 loadgen·runner 수정 커밋(= POST, engine-porter) → (4) 스크립트 갱신: `PDMUX_PF_POST` 기본값을 POST로 바꾸거나 제출 때 `--env PDMUX_PF_POST=<POST 전체 sha>`, `post_vs_job_head` 경로 7개, 새 POST에서 `--dry-run-trees` 재실행 → (5) `launch.sh` dry-run 출력 승인 → (6) `--submit` → (7) `fetch.sh` 후 라벨을 `tracks/multigpu.md` §1-b와 SPEC에 반영.

## 10. r2 판정기 개정 (2026-10-17) — F3·F5·F6·사전 선언·결론 범위

> 바뀐 파일(전부 `merge_trial_tools/`, gate 7경로 밖 — POST를 바꾸지 않는다): `preflight_judge.py`, `sigcmp.py`, `preflight_cl3.sh`, `preflight_judge_selftest.py`, 신규 `pf_unittest_runner.py`·`preflight_cl3_cpu_timing_declared.txt`·`preflight_judge_fixture_selftest.py`·`preflight_judge_mutants.py`. 도구 sha가 바뀌므로 다음 런의 `meta/provenance.txt`가 새 sha를 적는다. 판정기는 gate의 측정기다 — 바꾸면 결론 범위가 바뀐다.

### 10-1. F3 상태 파싱 (구조화)
- **원인**: 컨테이너에서 `PYTHONUNBUFFERED=1`(`job_entry.sh` 4단계, CPU/GPU 무관)로 시험 stdout·ResourceWarning이 `<name> (<id>) ... `와 상태 토큰 사이에 끼어, 옛 정규식이 상태를 놓쳤다(pre 21, post 50건; 분석서 §2-(a)·§4).
- **수정**: (a) `pf_unittest_runner.py` — `python -m unittest discover -s … -v`와 같은 `-v` 텍스트를 내고(실패 블록·`Ran N` 불변 → 서명 불변), 끝에 `LOG.status.json`(id별 상태를 TestResult 콜백에서 직접)을 원자적으로 쓴다. id 정규화는 sigcmp 규약(`setUpClass (mod.Cls)` → `mod.Cls`, subTest 실패는 부모 id). `-b`(버퍼링)는 실패 출력이 블록 안으로 옮겨져 서명이 바뀌므로 쓰지 않는다. `sys.path[0]` = cwd(= `python -m` 동작). (b) `sigcmp.parse_status` 엄격 파서 — 시험 구간(헤더 ~ 다음 헤더/구분선/`Ran`) 안에 상태 토큰이 **정확히 1개**일 때만 인정, 0개·2개 이상이면 미판독(분석서 F3의 "느슨한 정규식 위험" 지적 반영). (c) 판정기는 JSON을 정본으로, 엄격 텍스트 상태·실패 블록·`Ran N`을 교차 대조한다. 어긋나면 INVALID. Job은 `--require-structured`(JSON 없으면 INVALID). JSON이 없는 옛 로그는 엄격 텍스트 상태를 쓰고, 미판독은 `ok`가 아니므로 구멍으로만 끝난다(fail-closed).

### 10-2. F5 마스킹 · F6 L2
- **F5**: `sigcmp.mask`가 `…/pf_work/pre|post`(arm 루트)를 `<arm>`으로 바꾸고 **나머지 경로는 남긴다** — 같은 원인은 같은 서명, 다른 파일은 다른 서명.
- **F6**: L2 id는 PRE 상태가 정확히 `ok`일 때만 재현. PRE에서 실패(서명이 같아도)·skip·미판독이면 구멍 `L2_UNVERIFIED`(UNVERIFIABLE 쪽). 더 엄격한 방향(fail-closed). `judge.json`에 L2별 pre/post 상태표(`l2_detail`)를 남긴다(규칙 아님, 대조용).

### 10-3. CPU 스펙 기지 타이밍 실패 — 사전 선언 (사용자 결정 §8 "(나)")
- 파일 `merge_trial_tools/preflight_cl3_cpu_timing_declared.txt`(sha256 `23f60658bccfde1c94aebfaf683506e3ef7a6df23fc237d8efd5d06189fbccaa`), **5 id**:
  `test_e1_cap_vessl.TestBudgetEnforcer.{test_overrunning_run_is_cut_at_the_deadline, test_equivalent_mutant_one_layer_removed_still_cuts}`,
  `test_n3_capacity.TestBudgetEnforcer.{test_overrunning_unit_is_cut, test_control_both_kill_layers_removed_overruns, test_equivalent_mutant_watchdog_removed_still_cuts}`(셋 다 L2이기도 함).
- **적용 조건**: `preflight_judge.py spec`의 규칙 = (`PDMUX_CPU_ONLY=1` 또는 `PDMUX_SPEC=resourcespec-a100cpu`) **그리고** GPU가 보이지 않음(`nvidia-smi -L`·`torch.cuda.is_available()`). 그때만 `--cpu-spec`이 붙고 5 id가 짝 대조·L2·flaky 규칙에서 빠진다(L2는 나머지 9로 판정). CPU 플래그인데 GPU가 보이면 INVALID. GPU 스펙에서는 파일을 읽어 기록하되 **적용하지 않는다**.
- **공시**: `judge.json` `declared_cpu_timing` = {file, sha256, cpu_spec, applied, ids, 반복별 pre/post 상태, `UNVERIFIED (CPU timing)` 문구}. `PREFLIGHT_PASS.json`(schema `preflight_pass/v2`)에 별도 필드 `declared_cpu_timing`(record 단계에서 파일 sha를 다시 계산해 judge.json 값과 대조, 다르면 PASS 거부)와 `conclusion_scope`를 둔다. `record_notes`는 "PASS를 거부한 사유" 목록으로 그대로 둔다(거기 넣으면 PASS가 불가능해진다). M3 gate(`mg_plan.preflight_gate`)는 `verdict`·digest·image·post_commit만 읽으므로 새 필드는 gate 동작을 바꾸지 않는다.
- **이 5건은 미검증이다.** PASS가 나와도 그 시험들(예산 강제·워치독·deadline 컷)이 POST에서 동작한다는 근거가 아니다. 선언 밖 타이밍 실패는 그대로 FAIL_NEW다.
- **자유 표면**: 이 목록은 런 `68f386b`를 본 뒤 정했다. 근거는 평가서 §3(4/4 결정적 재현, 3.5 vCPU)이고, 다음 런 **전에** 고정한다. 감사 대상.

### 10-4. 결론 범위 선언 (record 산출물 `conclusion_scope`와 같은 내용)
- **CPU 스펙 PASS의 범위**: CPU 컨테이너(`resourcespec-a100cpu`, GPU 없음), `sgl_kernel`은 `sm100` 변종(cc None 분기), **CUDA 커널 실행 0**. PASS는 merge 트리의 단위 스위트에 대해서만 말한다. **GPU 실행 경로는 이 PASS 범위 밖**이고 M2g PASS(digest `82a5fc5b…`)와 M3 Job 내부 gate(B0·ENGINE_DIGEST·xcheck)가 덮는다 — **최종 POST의 엔진 digest가 `82a5fc5b…`일 때만**. 사전 선언 5건은 미검증. "회귀 0건"(C-4)·"preflight PASS가 M3 실행 바이트를 보증"(MGH12-7)이라고 쓰지 않는다.
- 런 `68f386b`의 로그는 **fixture로만** 쓴다. 그 런의 라벨은 기록된 `PREFLIGHT_FAIL_NEW` 그대로다(fixture selftest는 비변이 로그의 라벨을 출력하지 않는다).

### 10-5. 검증 (양방향)
- `preflight_judge_selftest.py`(Job 안에서도 실행): 기존 22건 + r2 31건 = **53건** — 컨테이너 형태(stdout 끼어쓰기, ResourceWarning) 통제군 PASS, 2-토큰 모호 구간 → 구멍, JSON 6종 불일치 → INVALID, arm 경로만 다름 → 같은 서명 / 다른 파일 → CHANGED, PRE 실패·skip L2 → 구멍, 선언(CPU 적용·GPU 미적용·선언 밖 타이밍·대상 선언 금지·L2 선언), 스펙 결정표 8행, 러너 자체(ok/FAIL/ERROR/subTest/skip/setUpClass skip/import 오류), record 4건.
- `preflight_judge_fixture_selftest.py`(로컬, 런 `68f386b` 로그 읽기 전용, 39항목): (i) 4개 로그 전 헤더 판독·요약줄 계수 일치·두 컨테이너 형태 `ok` (ii) arm 경로 8건 서명 동일·CHANGED 0 (iii) 주입한 신규 실패 → FAIL_NEW, 다른 파일 → CHANGED (iv) 선언 밖 타이밍 실패 → FAIL_NEW (v) GPU 스펙: 미적용(L2 12, PRE 실패 n3 2건 `L2_UNVERIFIED`, post에서만 실패한 선언 id → NEW) / CPU 스펙: 적용(L2 9, 구멍 0, 같은 id 제외).
- `preflight_judge_mutants.py`: 판정기·sigcmp·러너 변이 21종 전부 KILLED(통제군 통과; 등가 변이 1종은 목록에서 이유와 함께 제외). fixture selftest 단독으로도 F3 텍스트·F5·F6·선언·범위 변이 7종을 잡는다.

### 10-6. 남는 한계
- 구조화 러너는 컨테이너에서 아직 한 번도 돌지 않았다(다음 런이 첫 실행). 러너가 `python -m unittest`와 다른 동작을 하면(`sys.argv`, `__main__`) 시험 결과가 달라질 수 있다 — 로컬 전체 회귀로 대조했다(커밋 메시지·보고).
- 아카이브 부재 시험(F8)·venv 17건 PRE 기준선 부재는 r2가 다루지 않는다(범위 공시만).
- 선언 5건과 n3 예산 강제·재개 분기는 CPU 스펙에서 계속 미검증이다(분석서 §5-2 1·3).
