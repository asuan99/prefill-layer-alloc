# Merge preflight (CL-3 미검증 시험) 설계 — G0·M3 공용 1회 (2026-10-15, 초안)

> 지위: 설계 + 스크립트 초안. VESSL 제출 0, Workspace 기동 0, GPU 실행 0, 커밋 0. 측정이 아니라 **정확성 프로브**다. 성능 수치·판정 없음.
> 근거: `VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md` §6 조건 1, `MERGE_TRIAL_known_failures_2026-10-15.md`(v2) §3-2·§5, 운영 모델 §1(측정=Job, Workspace=bring-up).
> 파일: `merge_trial_tools/preflight_cl3.sh`(진입), `preflight_judge.py`(판정), `preflight_judge_selftest.py`(변이 11건), `preflight_cl3_l2_ids.txt`(L2 12), `preflight_cl3_targets.txt`(CL-3 대상 31 항목 = 9 모듈 + 4 id + 17 id + zamba2).

## 1. 실행 수단 — 짧은 Job 1개 (`resourcespec-a100x1`), Workspace 아님

| 선택지 | 판단 |
|---|---|
| `pdmux-probe` Workspace | 기각. 가변 환경(ssh 세션은 이미지 ENV 미상속, 운영 모델 §9), 산출물이 DONE/SHA256SUMS 경로를 안 탄다, 사람 손이 들어간 환경에서 낸 "PASS"는 감사 근거가 약하다. 이 preflight는 G0·M3 제출의 **게이트**라서 재현 가능한 경로가 필요하다. |
| `pdmux-build` CPU Workspace | 기각(조건부). `sgl_kernel/load_utils.py`는 GPU 없으면 `sm100` 변형을 로드하는 CPU 경로가 있어 import는 될 수 있으나(로컬 설치본 확인), 이미지(`1e0d335b`, libnuma1 포함)에서 CPU만으로 import되는지는 **미실측**이다. 과거 CPU 회귀(B2)의 errors 9는 libnuma 이전 이미지였다. 또 Workspace는 위와 같은 사유로 게이트 근거가 약하다. |
| CPU 스펙 Job(`resourcespec-a100cpu`) | 기각. `job_entry.sh`가 `nvidia-smi` 도장·`clk.csv`·substrate.json을 GPU 전제로 만든다. CPU 스펙에서는 미시험 경로. |
| **A100x1 Job** | **채택.** 검증된 경로(B5 이래 fetch.sh까지 통과), 이미지 digest·커밋 bundle·DONE+sha256로 감사 가능, 시험이 실제로 도는 조건(`cc8.0` 노드, `sgl_kernel` 정상 로드)이 G0·M3 Job과 같다. 부수 이득: merge된 `job_entry.sh`(로그 미러·heartbeat)가 VESSL 컨테이너에서 처음 한 번 돈다(증거가 아니라 부수 관찰). |

GPU가 **꼭** 필요하진 않다(시험 대부분은 CPU 코드, `PDMUX_DRY_RUN=1`로 서버를 띄우지 않는다). GPU 점유는 "검증된 Job 경로를 쓰는 대가"다. 사용률은 ~0 %로 예상(아래 §7).

## 2. 전·후 두 트리 (한 컨테이너 안)

`job_entry.sh`가 이미 `/opt/pdmux/sglang_engine_dev`를 **Job 커밋(HEAD)**으로 sync했다. 이 트리를 그대로 두 번 쓰지 않는다. arm마다:

1. `git worktree add --detach $WORK/<arm>/prefill-layer-alloc <sha>` (`WORK=/data/runs/<job>/pf_work`, 종료 시 삭제). `hf_cache -> /data/hf` 링크. `<arm>` = `pre`(`7b15214b…`) / `post`(`dc06c870…`).
2. dev tree = `cp -a` of `dirname($SGLANG_ENGINE_DEV)` → `$WORK/<arm>/sglang_engine_dev/python`.
3. **pre만**: 복사본은 post 성격(HEAD sync)이라 merge-only 효과를 뺀다 — `pdmux_context_device_assert.patch` 역적용 + `multigpu_guard.py` 삭제. 그 뒤 `pdmux_context.py` sha가 로컬 main dev tree(무패치)의 `b4e6ff1a…`와 같고 패치가 다시 정방향으로 적용되는지 확인한다. 다르면 UNVERIFIABLE(이미지 upstream이 로컬과 다름).
4. 각 arm 자신의 `install_runtime.sh`(sync + manual edits) → manifest(pre 25행, post 27행) → `sha256sum -c`.
5. **sbatch 구동 시험의 `SGLANG_ENGINE_DEV` 덮어쓰기 해법**: `test_campaign_provenance`·`test_lambda_star_per_shape`·`test_r2_eval_runner`는 `r2_eval.sbatch`를 호출하고 그 안의 `sync_engine_tree.sh`가 dev tree에 src를 설치한다(known_failures §1-1 부작용). arm별로 **`PDMUX_ROOT=$WORK/<arm>`, `PDMUX_PROJECT_ROOT` unset, `SGLANG_ENGINE_DEV`·`PYTHONPATH`=그 arm 복사본**으로 돌리면 sbatch가 읽는 `project_root`(=`$PDMUX_ROOT/prefill-layer-alloc`)가 그 arm 자신의 worktree이므로 재설치는 같은 바이트의 no-op이다. 환경의 `PDMUX_*`는 `env -u`로 전부 걷어 낸다(`PDMUX_CAMPAIGN`·`PDMUX_PROJECT_ROOT`가 시험 입력을 바꾸지 않게). 시험 전후로 multiplex/*.py + scheduler.py 해시를 비교해 변하면 INVALID.
6. 검사: 두 arm 트리의 `diff -rq`가 `git diff PRE POST -- src`로 설명되는 6개 파일(chunk_probe, holb_probe, multigpu_guard, multiplexing_mixin, telemetry, pdmux_context)에 한정. 그 밖이면 INVALID. `import sglang`이 arm 복사본에서 오는지(이미지의 editable 설치가 아니라) 확인. `import sgl_kernel` 실패 시 시험을 돌리지 않고 UNVERIFIABLE.

★로컬 검증 완료(CPU, `--dry-run-trees`, 로컬 `sglang_engine_dev_multigpu`를 기반으로): pre 재구성 후 `pdmux_context.py` sha = `b4e6ff1a…` 일치, 정방향 재적용 가능, manifest 25/27행, 두 트리 차이 = 위 6개 파일 정확히, 미설명 차이 0. 남은 INVALID는 로컬에 `sgl_kernel`이 없는 것뿐(의도된 fail-closed).

## 3. venv 17건 — 코드에서 확인한 사실과 처리

- 시험 17건(campaign_provenance 10 / lambda_star SbatchReadsTheField 3 / r2_eval_runner TestSbatchPathResolution 4)은 `engine_bench_runner.sh:115 source "${venv_dir}/bin/activate"`에서 멈춘다. `venv_dir="${PDMUX_VENV:-${pdmux_root}/sglang_engine_venv}"`(:59) — **`PDMUX_VENV`가 공식 env 지정**이고 시험은 `dict(os.environ)`을 그대로 넘기므로 시험 코드 수정 없이 닿는다. `PDMUX_DRY_RUN=1`이라 activate 뒤 서버 인자를 출력하고 멈춘다.
- 이미지에는 `sglang_engine_venv`가 없다(Dockerfile: 이미지 python에 직접 pip). 후보를 `set -euo pipefail`(러너와 같은 셸 옵션)로 **먼저 프로브**해 첫 통과를 쓴다: (a) `PDMUX_VENV`(--env로 준 값) (b) 이미지 python의 `sys.base_prefix`(conda base면 `/opt/conda/bin/activate`) (c) `python -m venv --system-site-packages`로 만든 실제 venv. 결과는 `meta/venv_mode.txt`(`env|image_python_prefix|created_system_site|none`)와 `venv_probe.txt`.
- **판단 요청**: (c)는 "정당한 env 지정"의 경계에 있다(환경 제공이지 시험·코드 변경은 아니지만, 원래의 `sglang_engine_venv` 내용물과 다르다). 기본 스크립트는 (c)까지 시도하되 모드를 기록한다. 사용자가 (c)를 불허하면 (a)(b)만 쓰고, 둘 다 안 되면 17건은 **검증 불가**로 남는다(만들지 않은 activate 스텁은 쓰지 않는다 — 항등식 위험).
- 판정상 17건은 post에서 전부 `ok`여야 "검증"이다. 전·후 똑같이 다른 이유로 실패하면(예: 러너가 activate 뒤 필요로 하는 모듈 부재) 대조는 일치하지만 **구멍**으로 계산한다 → UNVERIFIABLE.

## 4. 대조 기준 (VESSL 안에서 짝지음, 로컬 pin 미사용)

- 단위 = (시험 id, 서명): `sigcmp.parse_multi`의 서명(예외 유형, 마스킹 메시지, 핵심어, DRIFT 다이제스트). `preflight_judge.py`는 `pin_post.json`을 읽지 않는다.
- **신규 실패** = post에서 실패하나 pre에서 같은 서명으로 실패하지 않고 L2도 아닌 id. 양쪽 실패인데 서명이 다른 id도 신규(같은 id 다른 원인). L2 = "pre ok → post fail" 12 id(목록 파일; id 단위 소속. 서명은 pre/post 짝만 본다 — L2의 원인이 다른 것으로 바뀌는 경우는 이 판정이 못 가른다, 한계).
- 반복: 기본 2회 ABBA(pre, post, post, pre). 같은 arm 반복 간 실패 집합·서명이 다르면 flaky → UNVERIFIABLE.
- 구멍(coverage): 31 대상 항목이 전부 post에서 실행·통과해야 한다. skip(큰따옴표 skip 포함 — `sigcmp`는 단일따옴표만 인식해서 판정기가 원문을 따로 스캔함)·import 오류·실패는 구멍.

## 5. 판정 라벨 (우선순위 순)

| 라벨 | 조건 | rc |
|---|---|---|
| `PREFLIGHT_UNVERIFIABLE` | INVALID(트리 재구성 실패, sgl_kernel import 실패, 로그 잘림/타임아웃, 반복 불일치, L2 0건 재현, 시험 중 트리 변경, smoke 부분집합) | 3 |
| `PREFLIGHT_FAIL_NEW` | INVALID 없음 + L2 밖 신규 실패 ≥1 → G0·M3 제출 중단 | 1 |
| `PREFLIGHT_UNVERIFIABLE` | 신규 0이지만 구멍 ≥1 | 3 |
| `PREFLIGHT_PASS` | 신규 0, INVALID 0, 구멍 0 | 0 |

판정기 자체는 합성 로그 변이 11건으로 검증했다(`preflight_judge_selftest.py`: 통제군 PASS, 신규 실패 2종→FAIL_NEW, skip·import 오류·잘린 로그·flaky·양쪽 동일 venv 실패·L2 미재현→UNVERIFIABLE, 신규>구멍 우선순위, L2 1건 부재→PASS). 로컬 실제 로그(pre 877, post 1265)에 돌리면 `PREFLIGHT_UNVERIFIABLE`(구멍 30/31)이 나온다 — 이 로컬 공백을 있는 그대로 재현.
**PASS의 의미**: 이 31 항목과 전체 스위트에서 L2 밖 신규 실패가 없다는 것. "회귀 0건"(C-4)이라고 쓰지 않는다. 결론 범위는 (PRE sha, POST sha, 이미지 digest, 도구 sha)에 묶인다.

## 6. G0·M3 공용 조건 (요청 ③)

한 번의 PASS가 둘 다 덮으려면: (i) 이미지 digest 동일 (ii) G0·M3 제출 커밋의 `workspace/engine-port/{src,tests,scripts/bootstrap,scripts/r2_eval,env}`가 POST(`dc06c870`)와 바이트 동일 — `meta/post_vs_job_head.txt`(preflight Job 커밋 대비)에 기록, 제출 때 같은 `git diff --stat POST <제출커밋> -- <5경로>`가 비어야 한다. G0가 `src/multiplex`를 고치면 이 PASS는 G0에 무효(G0 변경분은 별도 짝 대조 필요). 현재 `dc06c870`→HEAD는 docs만 다르다.

## 7. 산출물·회수·소요

- 제출: `launch.sh --campaign preflight_cl3 --target workspace/engine-port/results/handoff_review_2026-10-15/merge_trial_tools/preflight_cl3.sh [--env PDMUX_PF_REPS=2] [--env PDMUX_VENV=...]` dry-run → 승인 → `--submit`. 도구 파일 **커밋 선행 필요**(bundle=HEAD).
- 산출물(`results/preflight_cl3/<job>/` → fetch가 `results/` 원위치로 병합): `PREFLIGHT_LABEL`, `judge.json`, `logs/<arm>.rep<i>.log`·`.rc`, `meta/`{provenance, arm_*/(manifest, install_runtime.log, reverse_patch.log, sglang_path, sgl_kernel), arm_tree_diff*, allowed_tree_diff, venv_mode·venv_probe, judge_selftest, lists_check, post_vs_job_head, elapsed}. Job 메타는 `results/preflight_cl3/vessl/<job>/`(substrate·clk·cost). 회수 = `fetch.sh preflight_cl3 <job>`: DONE + `sha256sum -c` + 병합 충돌 0. 스크립트는 판정 rc로 종료하므로 PASS 외에는 Job이 failed로 보이나 DONE·SHA256SUMS는 항상 쓰인다.
- 소요(추정, 미실측): 부팅·이미지 pull·install 5–10분 + worktree 2개·dev tree 복사 3–5분 + 스위트(로컬 기준선 598건 296초; 현재 1265건, sbatch 구동 시험은 개당 최대 900초 timeout) arm·회당 8–15분 × 4회 = 35–60분 → **총 ≈ 1시간, ≈1 GPU-h**. `PDMUX_PF_REPS=1`이면 ≈ 40분.
- GPU 사용률: 할당 1장, 시험 중 사용 ≈ 0 %(CPU 단일 프로세스, 서버 미기동). 더 작은 GPU 스펙은 없고(x1이 최소), GPU 없는 스펙은 §1처럼 미시험 경로라 x1이 적정하다.

## 8. 미해결 위험

1. **venv 17건**: 이미지 base conda의 `activate`가 `set -u`에서 실패할 수 있고(프로브가 드러냄), (c) 사용 여부는 사용자 판단. 모두 실패하면 17건 검증 불가.
2. **zamba2_instrumentation 런타임 클래스**: `sgl_kernel`이 풀려도 hf_cache에 Zamba2-2.7B가 없으면(스테이징은 Nemotron-Nano만) 다른 이유로 skip될 수 있다 → 구멍 → UNVERIFIABLE. 가능성 높음.
3. 컨테이너에서 처음 실행: `git worktree`가 `/data` 볼륨에서 도는 속도·용량(트리 2개 ≈ 1.5 GB), sbatch 구동 시험의 컨테이너 내 동작. 첫 Job 실패 시 수정 후 새 job name 재제출(자동 재시도 없음).
4. pre 재구성의 기준 sha(`b4e6ff1a…`)는 로컬 main dev tree 값. 이미지 upstream이 다르면 UNVERIFIABLE(원인 추적 필요).
5. 타이밍 시험(`TestBudgetEnforcer` 등)이 공유 노드 CPU에서 흔들리면 flaky 판정으로 UNVERIFIABLE. ABBA로 감지만 하고 보정하지 않는다.
6. L2 서명 동일성은 강제하지 않는다(§4 한계).
7. 판정 PASS여도 CL-3 밖(예: 이 스위트가 안 닿는 경로)은 말하지 않는다.

## 9. 다음 행동 (모두 사용자 승인 필요)

(1) 도구 6개 커밋 → (2) venv (c) 허용 여부 결정 → (3) `launch.sh` dry-run 출력 승인 → (4) `--submit` → (5) `fetch.sh` 후 라벨을 `tracks/multigpu.md` §1-b와 SPEC에 반영.
