# 사전등록 — λ0 VESSL 기판 이식 追記 (λ0 rev5 기판 재앵커 + 기준선 분산) (2026-10-04)

> **지위**: `lambda0_prereg/PREREG_LAMBDA0_REV5_2026-09-14.md`(sha `8367b6ec…`)와 그 구속 追記
> `PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md`(sha `632c2d78…`)에 대한 **기판 이식 追記**다.
> rev5의 판정 규칙·셀·시드·사다리·라벨은 **한 글자도 바꾸지 않는다**. 바뀌는 것은 실행 기판과
> 환경 배선, 그리고 rev5에 없던 **추가물 4개**(§2-3)뿐이다.
> - 사용자 승인(2026-10-04): VESSL 기판 재앵커 = λ0(용량) 재측정 + 기준선 분산.
> - 이 문서는 **claims-auditor 규칙층 감사 대기**다. `GO` 전에는 `launch.sh --submit` 금지.
> - 작성 시점 GPU 지출 0. 성능 판정 0건. Claim 등급 변경 0건.
> - 작성: engine-porter. 커밋은 메인 세션이 한다.
>
> **파일**(모두 `workspace/engine-port/results/r2_eval/lambda0_vessl/`)
> - `lambda0_vessl.sh` — 러너. `lambda0_prereg/lambda0.sbatch`의 VESSL 이식본.
> - `lambda0_vessl_plan.py` — 등록 상수(digest·분산 시드·예산), 분산 블록 행 생성, 예산·입장 계산,
>   기술 요약. rev5 모듈을 **import**만 한다(복사 없음).
> - `stage_lambda0_inputs.sh` — git 밖 판정 입력 3개를 Object 볼륨에 올리는 로컬 도구(기본 dry-run).
> - 테스트: `workspace/engine-port/tests/test_lambda0_vessl.py`.
>
> **줄 번호 인용 규약**: rev5 파일은 digest로 고정돼 있으므로 `파일:줄`로 인용한다.
> 이 회차의 새 파일은 아직 커밋되지 않았으므로 줄 번호 대신 **코드 텍스트**로 인용한다
> (λ5A-11과 같은 이유 — 미커밋 파일의 줄 번호는 낡는다).

---

## 0. 한 줄 요약

KISTI job 908623과 **같은 바이트의 규칙·계획·셀**을 VESSL A100(108 SM) 위에서 한 Job으로 다시 돌린다.
그 뒤에 rev5 최상단 rung의 seed 반복을 2개 더 붙여(shape당 n=4) λ\* 측정의 기준선 산포를 잰다.
VESSL 수치는 KISTI 908623 수치와 **나란히 놓기만 하고 절대 합치지 않는다**.

---

## 1. (a) rev5와 같은 것 — 전수와 강제 수단

"같다"는 주장은 문서가 아니라 **러너 안의 중단 조건**으로 강제한다.
아래 표의 "강제" 열이 비어 있는 행은 코드 텍스트 동일성만으로 보장된다(§1-2의 구조 테스트가 감시한다).

### 1-1. 판정 경로·입력 (바이트 동일, 위반 시 GPU 0에서 중단)

| 항목 | rev5 출처 | VESSL 이식본 | 강제 |
|---|---|---|---|
| 결정 경로 7파일 + `lambda0_cellprint.py` + `lambda0.sbatch` + rev5 사전등록 + 追記 (11개) | `lam0_908623/REGISTRATION_SHA256.txt` · 追記 §B-2 | 같은 파일을 `$PREREG/…`로 **호출** | `lambda0_vessl_plan.py` `REV5_DIGESTS` ↔ 실제 sha256. 하나라도 다르면 `ABORT_F6_DRIFT`(exit 3) |
| λ_inf 술어 입력 5개(job 907959: `srv_warmup.log`·`I3a/I3b_*.jsonl`·`I_log_offsets.txt`·`I3_max_running_req.txt`) | `lam0_908623/LAMBDA_INF_DECISION.txt` `LAMBDA0_INPUT_SHA256_*` | 앞의 3개는 git 밖이라 `/io/staging/lambda0_inputs/`에서 복사(§2-5) | `DECISION_INPUT_DIGESTS`. 불일치 시 `ABORT_F6_DRIFT` |
| 분지 = `FALLBACK`, `ANCHORED`면 중단 | `lambda0.sbatch:169-236` | 같은 명령열(`--selftest`·`--recount`·`decide()`·`case`) | rev5 그대로(`ABORT_D18`) |
| 계획 `plan.json` | `lambda0.sbatch:269-270` · `--lambda-inf-a 2.10 --lambda-inf-b 0.675 --fallback`(`:235`) | 같은 CLI | `cmp $OUT/plan.json lam0_908623/plan.json` 불일치 시 `ABORT_PLAN_DRIFT`(exit 3). ★로컬(Python 3.13.9·numpy 2.2.6)과 공개 이미지 컨테이너(Python 3.14.6·numpy 2.5.1, GPU 없음) 양쪽에서 **바이트 동일 확인**(§9) |
| 셀 11개·순서·시드 4386/4162 | `lambda0_cells.py`(`rows()`), `lambda0_plan.py:275-280`(`choose_seeds`) | 같은 CLI | `n_cells=11` 아니면 `ABORT_CELL_COUNT` |
| selftest 5종 + reachability + 변이 하네스(rc 분기 포함) | `lambda0.sbatch:185,238-267` | 같은 명령열, 같은 rc 분기(`exit 5`/`exit 6`) | rev5 그대로 |
| 라벨 호출 `--cells-dir $OUT --expect … --repeat …` | `lambda0.sbatch:413-415` | 같은 명령. 분산 셀은 **하위 디렉터리 `variance/`**에 있어 `load()`(`lambda0_label.py:260-283`, `--expect` 이름만 읽음)에 보이지 않는다 | 구조 테스트 |
| D12 manifest(≥24항목, `nemotron_h.py` 포함) | `lambda0.sbatch:147-162` | 같은 sync 호출 | rev5 그대로 + §1-3 엔진 트리 대조 |

### 1-2. 서빙 arm·명령줄 (코드 텍스트 동일)

| 항목 | rev5 출처 | 값 |
|---|---|---|
| 모델 | `lambda0.sbatch:99` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` |
| 모델 리비전 | rev5 서버 로그 `revision=None`(refs/main) · 이양 번들 `D_misc/hf_hub_revisions.tsv:5` | `dc0661c829b14e5b9246c05cfa89094a0875e052` — VESSL `/data/hf` refs/main이 이 값이 아니면 `ABORT_MODEL_REVISION`(exit 3) |
| 백엔드 | `:100` | `flashinfer` |
| PD-mux config | `:101` | `results/longctx_conflict/probes/pdmux_homog5.yml`(sha `be6dd647…`, 2026-09-09 `7e924a2` 이후 불변 — `CONFIG_DIGEST`로 강제). `manual_divisions` 키가 있으므로 `get_arch_constraints`는 발화하지 않는다 |
| ctx / mem / max_running / server seed / fixed decode SM | `:102-106` | 16384 / 0.82 / 48 / 1 / 44 |
| trace | `:107` | `$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json` |
| unset 목록 | `:91-94` | 그대로 + `PDMUX_DUAL_WORKER_TRACE`·`PDMUX_MEM_TELEMETRY` 추가(둘 다 rev5 실행에서 미설정이던 기본값을 명시적으로 보장할 뿐) |
| 셀 env | `:326-330` | `PDMUX_TELEMETRY_PATH`·`PDMUX_RUN_ID`·`PDMUX_WORKLOAD_ID`·`PDMUX_R2_POLICY=fixed`·`PDMUX_R2_FIXED_DSM=44` 동일 |
| 서버 플래그 | `:338-345` | 동일(cudagraph ON, `--disable-piecewise-cuda-graph` 미전달) |
| warmup 8요청 동시성 1 seed 7 | `:357-362` | 동일 |
| 측정 bench(`--tokenize-prompt` · `--random-range-ratio 1.0` · `--output-details`) | `:372-378` | 동일 |
| 포트 선택·`wait_health` 240×5 s·teardown(`kill`·5 s·`pkill -9`·10 s) | `:316-323,:295-302,:396-397` | 동일 |
| 연속 boot 실패 2회 → 이후 셀 미시도 | `:399-402` | 동일(`STOP=boot`. 분산 블록도 시도하지 않는다) |
| analyze·cellprint 호출 인자 | `:383-389` | 동일. 단 예산 집행기가 bench를 죽인 셀은 analyze를 건너뛴다(§6-3) |

### 1-3. 엔진 트리 (내용 동일 강제)

- 러너는 rev5처럼 `sync_engine_tree.sh`로 manifest를 다시 쓴 뒤, 그 25항목을 `lam0_908623/runtime_source_manifest.sha256`과
  **경로 접미(`…/python/` 이후)별 내용 해시**로 대조한다(`lambda0_vessl_plan.py --engine-diff`).
- 하나라도 다르면 `ABORT_ENGINE_TREE_DRIFT`(exit 3)다.
- 사전 사실(GPU 0): B5 Job(`ec6d94d`)이 VESSL에서 쓴 manifest와 908623 manifest는 **25/25 동일**하다(이 회차에서 로컬 재대조).

---

## 2. (b) rev5와 다른 것 — 전수

### 2-1. 기판

| 축 | KISTI 908623 | VESSL | 근거 |
|---|---|---|---|
| GPU | A100-SXM4-80GB, 108 SM, gpu38 | A100-SXM4-80GB, 108 SM, cc 8.0, MIG off | `results/b5_smoke/…/SUMMARY.md` B0 |
| 격자 실현 | 미실측(908623 자체) | (64,44) 포함 5개 격자에서 요청 = 실현 %smid 집합 크기, 서로소, cudagraph replay 중 격리 유지 | B3a·B3c(2026-10-04). **빈 GPU 미시 조건**이며 부하 중 실현은 이 Job의 green read-out이 기록(§7) |
| 드라이버 | 580.105.08(2026-08-20 기록. **908623 자신은 드라이버를 기록하지 않았다**) | 580.105.08(B0·B5) | 버전 문자열이 같아도 플랫폼 패치 빌드 동일성은 미확인 |
| 노드 | 전용 할당(SLURM `--gres=gpu:1`) | **공유 노드**, Job마다 노드가 다를 수 있음 | 운영모델 §1-2 |
| 클럭 | 미고정(기록 없음) | **고정 불가**. job_entry가 100 ms마다 SM 클럭·전력·스로틀 사유를 `meta/clk.csv`에 기록 | `job_entry.sh` §3 |
| CPU·RAM | `--cpus-per-task=8` · `--mem=100G`(cgroup) | 스펙 12 CPU · 80 GiB(setup plan §1, 실측은 이 Job의 `SUBSTRATE_LAM0V.txt`) | ★클라이언트 코어 예산이 다르다(§3 VC-5) |
| 모델 저장소 | Lustre scratch | CephFS `/data/hf` | boot 시간에만 영향 |
| telemetry 쓰기 위치 | Lustre scratch | CephFS `/data/runs/<job>/results/…` | §3 VC-6 |

### 2-2. 소프트웨어

| 축 | KISTI | VESSL | 근거 |
|---|---|---|---|
| Python | 3.14.2(venv, 2026-09-17 기록) | 3.14.6 | `env/venv_packages_2026-09-17.txt:2` · Dockerfile 주석 |
| torch / triton | 2.9.1+cu130 / 3.5.1 | 동일 | requirements 헤더 "identical" |
| flashinfer / sglang-kernel | 0.6.10 / 0.4.1+cu130 | 동일 핀 | `requirements.pdmux-sglang.txt` · Dockerfile |
| numpy | 2.3.5와 2.5.1 **두 dist-info 공존** | 2.5.1 | `venv_packages_2026-09-17.txt` |
| `causal_conv1d`·`mamba_ssm`·`flash_attn` | 설치됨 | **없음**(deferred) | requirements 헤더. SGLang CUDA 경로는 `sgl_kernel`·triton 구현을 쓰고 pip 패키지를 import하지 않는다(로컬 grep) — **코드 독해이며 측정된 null이 아니다** |
| 엔진 설치 | KISTI live dev tree(sync + 수동편집이 이미 적용된 상태) | 공개 이미지(pristine v0.5.10) + 시작 시 `install_runtime.sh`(sync → `devtree_manual_edits.patch` → 재sync) | manifest 25/25 동일(§1-3). **manifest 밖** SGLang 파일은 `git archive v0.5.10` + 수동편집 5파일이며, KISTI 트리에 기록되지 않은 편집이 있었는지는 확인할 수 없다 |
| 이미지 | 없음(module + venv) | `ghcr.io/asuan99/sglang-runtime@sha256:1e0d335b…`(레시피 `75758e8`) | `vessl.env` |

### 2-3. 실행 배선과 추가물

| 항목 | rev5 | 이식본 | 종류 |
|---|---|---|---|
| 경로 | `ROOT=/scratch/…` 상수(`:75`) | `ROOT="${PDMUX_PROJECT_ROOT:?}"`, `HF_HOME`은 job_entry 값 | 배선 |
| 환경 | `module load`·venv `source`(`:78-80`) | 제거(이미지가 제공) | 배선 |
| triton 캐시 | `$OUT/.triton_cache`(`:88`) | job_entry의 Job별 `TRITON_CACHE_DIR`·`XDG_CACHE_HOME`·`FLASHINFER_WORKSPACE_BASE` | 배선 |
| 결과 위치 | `lambda0_prereg/lam0_<SLURM_JOB_ID>/` | `lambda0_vessl/lam0v_<job name>/`(job_entry가 반출, `fetch.sh`가 병합) | 배선 |
| git 밖 판정 입력 | KISTI 디스크에 존재 | `/io/staging/lambda0_inputs/`에서 sha 검증 후 복사(§2-5) | 배선 |
| 예산 집행 | SLURM `--time=04:30:00`(`:9`) | **스크립트 내부 wall-clock 집행기**(§6-2) | ★추가물 1 |
| 분산 블록 | 없음(F4 seed 반복 1개뿐) | 최상단 rung seed 3·4 반복 4셀(§5) | ★추가물 2 |
| green read-out | 없음 | boot마다 1회 `PDMUX_GREEN_READOUT=1`(§7) | ★추가물 3 |
| 실현 분할 요약 | 없음(결과 감사자가 사후 계산) | 셀마다 `e2_realized_mix.py`로 기술 기록(§7) | ★추가물 3 |
| 재개 | 없음 | `PDMUX_LAM0V_RESUME_FROM`(§6-4) | ★추가물 4 |
| `timeout`·`setsid` | 없음 | warmup·bench에 `timeout -k 30 <남은 예산>`, 서버는 `setsid`로 띄움 | 추가물 1의 일부. 발화하지 않으면 동작 무변 |
| PDMUX_* 노브 | 고정 unset 목록 | **허용 목록 밖의 PDMUX_\*가 하나라도 있으면 `ABORT_ENV`**(§2-4) | 강화 |
| 운영자 노브 `PDMUX_LAMBDA0_INSTR`(`:169`)·`PDMUX_LAMBDA0_PREREG` | 경로 재지정 허용 | **제거**. 술어 입력은 job 907959 고정이고 digest로 강제된다. 두 변수는 허용 목록 밖이라 설정되면 `ABORT_ENV` | 강화(W4′ 표면 제거) |
| 실행 기록의 트리 상태 | `uncommitted=`(`git status --porcelain`, 미추적 포함) | `uncommitted_outside_results=`(추적 파일, `engine-port/results` 제외). job_entry가 `results/`를 심볼릭으로 바꾸므로 그 아래 추적 파일 전부가 변경으로 잡힌다(로컬 컨테이너 dry-run에서 4,168건) | 배선(기록의 의미 보존) |
| 바이트코드 | 기록 없음 | 종료 시(중단 포함) `lambda0_prereg/`·`lambda0_vessl/`·`e2_sticky_prereg/`의 `__pycache__`를 지운다. job_entry가 results 아래 새 파일을 전부 반출하기 때문이다 | 배선 |
| `PLAN.txt` 생성 | `… \| tee "$OUT/PLAN.txt"`(`:269`), 실패가 `tee`에 삼켜짐(追記 §B-3 2번 선재 결함) | `… > "$OUT/PLAN.txt" \|\| exit 2` 후 `cat` | 강화(fail-closed). 성공 시 출력 바이트 동일, 908623 `PLAN.txt`와 대조 기록 |

### 2-4. 노브 허용 목록 (새 자유 표면을 닫는다)

`launch.sh --env`는 임의의 `PDMUX_*`를 Job에 넣을 수 있다. rev5의 unset 목록은 고정된 이름만 지운다.
그래서 이식본은 허용 목록 밖의 `PDMUX_*`가 보이면 GPU 작업 전에 중단한다.
- 허용: job_entry/launch가 넣는 식별자(`PDMUX_COMMIT`·`_JOB_NAME`·`_CAMPAIGN`·`_TARGET`·`_IMAGE`·`_SPEC`·`_ROOT`·`_PROJECT_ROOT`·`_IO`·`_DATA`·`_OPT`), `PDMUX_LAM0V_RESUME_FROM`, `PDMUX_LAM0V_DRYRUN`.
- **테스트 전용**(`PDMUX_LAM0V_TEST_*`, `PDMUX_TEST_SKIP_SYNC`): `DRYRUN=1`이 아니면 `ABORT_TEST_KNOB_IN_MEASUREMENT`.
- `PDMUX_ARRAY_SPEC`는 비어 있어야 한다(한 Job).


### 2-5. git 밖 판정 입력의 운반

- λ_inf 술어는 job 907959의 `srv_warmup.log`·`I3a_shapeA.jsonl`·`I3b_shapeB.jsonl`을 읽는다.
  셋 다 `.gitignore` 대상(`*.log`·`*.jsonl`)이라 커밋 bundle에 실리지 않는다(rev5 §10 "결정 입력 3종이 git 밖").
- 로컬 `stage_lambda0_inputs.sh`가 세 파일을 모아 등록 digest와 대조한 뒤 `/io/staging/lambda0_inputs/`(평면 배치)로 올린다.
- Job은 그 파일을 `results/r2_correctness/job_907959/` 원위치로 `cp -p` 한다(mtime 보존 → job_entry가 다시 반출하지 않는다).
  그다음 `--verify-digests`가 908623이 기록한 digest와 대조한다. 없으면 `ABORT_STAGE`, 다르면 `ABORT_F6_DRIFT`.
- 변이 하네스가 읽는 probe C 아카이브(`c_905835/cell_*.json`)는 추적 파일이라 bundle에 실린다.

---

## 3. (c) 각 차이가 판정에 미치는 경로와 caveat

판정량은 rev5와 같다: 셀별 `achieved_rate`, `achieved_over_realized`(클라이언트 측) → 4개 라벨 + seed 반복.
telemetry는 판정에 들어가지 않는다(`lambda0_analyze.py` docstring "Everything here is CLIENT SIDE").

| ID | 차이 | 판정에 닿는 경로 | 처리 |
|---|---|---|---|
| **VC-1** | git 밖 판정 입력의 운반 | 다른 바이트가 오면 분지가 바뀔 수 있다(W4′ 계열) | digest 강제(§1-1). 운반 경로 자체는 판정에 무관 |
| **VC-2** | 노드 공유·클럭 미고정 | 처리율 수준값(λ\*)이 노드·시점에 따라 움직인다. 라벨은 같은 Job 안 rung 간 비교라 수준 이동에 상대적으로 둔감하지만, 브래킷 경계 근처에서는 라벨이 바뀔 수 있다 | `meta/clk.csv`·`clk_summary.json`(스로틀 플래그)를 VESSL λ\* 인용에 **필수 병기**. 스로틀이 켜진 셀이 있으면 그 사실을 라벨 옆에 적는다(라벨은 바꾸지 않는다) |
| **VC-3** | 드라이버·Python·numpy 차이 | numpy legacy 스트림이 다르면 도착 재현(span)이 깨지고 `achieved_over_realized`가 틀어진다. Ebar 가드는 이것을 못 잡는다(`lambda0_analyze.py` docstring) | `plan.json` 바이트 동일(=`ebar()` 재현) 강제. bench 쪽 도착 재생은 같은 numpy legacy `RandomState`이며 `range_ratio_ok`가 계속 필요조건이다 |
| **VC-4** | 빠진 pip 패키지 3개 | 커널 경로가 다르면 λ\* 수준이 달라진다 | 코드 독해로는 미사용. **측정된 null 아님** — VESSL λ\*를 KISTI와 비교할 때 이 문장을 병기 |
| **VC-5** | CPU 코어 12 vs 8(cgroup), 핀 없음 | `bench_serving` 클라이언트가 GIL 병목이면 포화 셀 처리율이 클라이언트에 묶인다(과거 "triton TPOT floor = GIL 아티팩트" 전례). a_r4(8.0 req/s × 400)가 가장 취약 | 코어 수·affinity·cgroup을 `SUBSTRATE_LAM0V.txt`에 기록. 코어가 **늘었으므로** KISTI보다 덜 묶일 쪽이지만, 양쪽 모두 미측정이므로 차이를 기전으로 읽지 않는다 |
| **VC-6** | telemetry가 CephFS에 쓰인다 | 비동기 writer의 쓰기 지연이 엔진 루프를 막으면 관측자 효과가 KISTI와 달라진다 | 측정하지 않는다(게이트 4의 대칭 on/off 측정은 이 Job의 범위 밖). telemetry 크기(셀당 25–70 MB, 908623 기준)를 기록하고 caveat로만 둔다 |
| **VC-7** | 엔진 설치 방식 | manifest 밖 파일 차이가 있으면 행동이 다를 수 있다 | manifest 25/25 강제(§1-3). manifest 밖은 caveat |
| **VC-8** | 예산 집행기 추가 | 발화하면 셀이 빠지거나 잘린다 → rev5 D23에 따라 그 shape가 `UNRESOLVED` | 잘린 셀은 analyze하지 않는다(§6-3). 발화하지 않으면 동작 무변 |
| **VC-9** | 분산 블록 추가 | rev5 셀 11개 **뒤**에 돌므로 rev5 셀의 실행 조건(순서·boot 수)은 그대로다 | 라벨 입력에서 구조적으로 분리(`variance/` 하위 디렉터리) |
| **VC-10** | green read-out 추가 | boot 시 1회 ctypes 조회. 스케줄링·배치·입장에 쓰이지 않는다(`green_readout.py` docstring "one-shot startup observation") | 관측자 효과 0으로 **주장하지 않는다** — 1회성이라 무시 가능하다고 판단하며, 이것은 코드 독해다 |
| **VC-11** | 재개 | 다른 Job(다른 노드일 수 있음)의 셀이 섞인다 | `RESUMED_CELLS.tsv`에 출처 Job·GPU UUID·드라이버 기록. 분산 요약이 `mixed_jobs`를 표시. 재개된 Job의 라벨은 "Job 혼합" 병기 필수 |

**승계 caveat**: rev5 §8 전부(Q1–Q5·L1–L3·R3C-1…4·NPC-I·N-7·N-8·N-9·N-11·R4C-1…6), 追記 A6의 λ5C-1…8,
908623 결과 감사 `VERDICT_result_lambda0_908623_2026-09-14.md` §6의 **λ0R-1…10**을 VESSL 결과에도 문자 그대로 승계한다.
특히 다음 셋은 이 Job에서도 그대로 참이다.
- **게이트 #6은 이 Job으로 닫히지 않는다**(rev5 §8 Q1). 이 Job이 재는 것은 throughput 포화이고 SLO 용량이 아니다.
- **λ0R-3**: λ\*(A)는 `--max-running-requests 48`이 구속한 처리율이다.
- **λ0R-8 계열**: λ\*(A)는 "D44에서의 용량"이 아니다. 실현 분할은 §7의 기록으로만 말한다.

---

## 4. (d) 라벨·판정 규칙

1. **rev5 라벨 그대로.** `lambda0_label.py`가 rev5 셀 11개로 shape별 `KNEE_BRACKETED`/`KNEE_NOT_BRACKETED`/
   `LADDER_TOO_HIGH`/`LADDER_TOO_LOW`/`UNRESOLVED`와 `SEED_REPEAT_*`를 낸다. 문턱 `ACH_HI 0.95`·`ACH_LO 0.90`·
   `SEED_REPEAT_TOL 0.05`(`lambda0_label.py:81-83`) 불변.
2. **이름.** 이 Job이 낸 값은 **λ\*\_VESSL(A)·λ\*\_VESSL(B)**로 부른다. 기판 표기 없는 "λ\*"로 쓰지 않는다.
3. **비교만, 합치지 않음.** `COMPARE_908623.json`은 셀별 achieved와 라벨을 KISTI 908623과 **나란히** 놓을 뿐이다.
   - 두 기판의 값을 평균·풀링하지 않는다.
   - 한쪽 셀로 다른 쪽 빠진 셀을 채우지 않는다.
   - VESSL 캠페인의 부하 사다리는 VESSL 값으로만 만든다. KISTI 값으로 만들지 않는다.
   - KISTI 결론(HE0·정책 순위 등)을 VESSL 값으로 재계산하거나 갱신하지 않는다.
4. **"기판 차이가 있다"를 쓸 수 있는 조건(사전 고정).** shape별로 다음이 **모두** 참일 때만
   "VESSL λ\*가 KISTI λ\*와 다르다"를 쓸 수 있다.
   - |Δ|/λ\*\_KISTI > 3%(CLAUDE.md 게이트 3: 3% 미만은 헤드라인이 아니다).
   - VESSL n=4 범위와 KISTI n=2 범위(a_r4/a_r4_s2, b_r3/b_r3_s2)가 겹치지 않는다.
     (rev5 λ\*는 포화 rung achieved의 최댓값이다. 그것이 최상단 rung이 아닌 rung에서 나오면 그 사실을 병기한다.)
   - 그 경우에도 문장은 "이 노드·이 시점의 관측 차이"까지만이다. 원인 귀속(드라이버·패키지·CPU·노드)은 금지다.
     §2의 차이가 동시에 여러 개 바뀌었기 때문이다.
   - 위 조건이 하나라도 거짓이면 "구별되지 않는다"로 쓴다. "같다"로 쓰지 않는다.
5. **분산 블록은 판정이 아니다.** `VARIANCE_SUMMARY.json`은 기술 통계다(§5).

---

## 5. 기준선 분산 설계

### 5-1. rev5의 seed 반복으로 충분한가 — 아니다

- rev5의 F4 반복은 shape당 **최상단 rung 1개 × seed 2개**(a_r4/a_r4_s2, b_r3/b_r3_s2)다. 즉 n=2다.
- CLAUDE.md 게이트 3은 n≥4를 요구한다. n=2로는 SD 추정이 불안정하고 "3% 미만 차이" 판단에 쓸 수 없다.
- 908623 실측 산포는 A 0.60%(3.0531 vs 3.0716), B 0.31%(0.6972 vs 0.6951)다. 이 값은 n=2이고 CI가 없다.

### 5-2. 추가안 (최소안 = 등록안)

- 같은 최상단 rung(a_r4 offered 8.0 N=400, b_r3 offered 1.15 N=200)을 **seed 3·4로 한 번씩 더** 돈다.
  각 반복은 **자기 boot**를 갖는다. shape당 n=4다(s1·s2는 rev5 셀, s3·s4는 분산 블록).
- 시드는 rev5 시드를 고른 **같은 채점**(`lambda0_plan.ebar`로 계획 N 다중집합의 max|Ēbar−1|)의 3·4위다:
  **251, 2630**. `lambda0_vessl_plan.py`가 1·2위가 rev5의 4386·4162와 같음을 assert하고, 3·4위를 등록 리터럴과 대조한다.
  E2 사전등록이 쓴 시드와도 같다.
- 두 시드 모두 그 N에서 analyzer의 Ēbar 가드(1/Ēbar ∈ [0.95, 1.05])를 통과한다(selftest).
- 실행 순서는 rev5 11셀 **뒤**에 `a_r4_s3 → b_r3_s3 → a_r4_s4 → b_r3_s4`(shape 교차)다.

### 5-3. 이 n=4가 무엇을 재는가 (범위 한정)

- 재는 것: "같은 Job·같은 노드에서, boot와 도착 seed와 실행 시점이 바뀔 때" 최상단 rung 처리율의 산포다.
  boot·seed·시점 효과를 **서로 분리하지 않는다**. s3·s4는 s1·s2보다 약 1–1.5시간 뒤에 돈다.
- 재지 않는 것:
  - **노드 간 산포.** Job이 1개면 노드도 1개다(§5-4).
  - **운영점 SLO 지표의 기준선 분산**(TTFT/ITL p95·goodput). 그것은 λ\*\_SLO가 정해진 뒤 그 캠페인이
    자기 기준선 arm으로 따로 재야 한다(게이트 #6이 선행).
- 요약 통계(사전 고정): shape별 mean·SD(n−1)·CV·범위/평균·rev5 λ\* 대비 최대 상대편차.
  포화 판정(`ratio ≤ 0.90`)과 R0 유효성(`_cell_invalid`)을 통과한 셀만 센다.
  4개 미만이면 `UNRESOLVED_N_BELOW_4`다. 이는 측정 부족이며 발견이 아니다(게이트 #21).

### 5-4. 노드 간 산포 (선택 — 사용자 결정)

- 최소안은 Job 1개다. 이 경우 노드 간 산포는 미측정으로 남는다.
- 권고안은 **같은 스크립트를 새 Job 이름으로 한 번 더**(Job 2개) 돌리는 것이다.
  각 Job이 자기 rev5 라벨과 n=4를 낸다. 노드(GPU UUID·hostname)는 공변량으로 기록된다.
- Job 2개의 결과는 Job별로 따로 보고한다. 두 Job을 풀링한 n=8은 **이 문서가 등록하지 않는다**.
  필요하면 결과를 보기 전에 별도 追記로 등록한다.

---

## 6. (e) 예산·하드캡·중단 규칙

### 6-1. 예산 (`lambda0_vessl_plan.py --budget`, 단가 $1.48/h)

- 셀 비용 = rev5의 `cell_duration_at`(`lambda0_plan.py:349-355`) + `warmup_s` + VESSL boot 여유 240 s.
  - KISTI 잔차 110 s 대신 240 s를 쓴다. 근거는 B4 boot+12요청 151 s(2026-10-04) + teardown 15 s + 여유다.
- preflight(sync·selftest·변이 하네스) 900 s(rev5 §7 "약 10분").

| 시나리오(참 λ\*/λ_inf) | rev5 11셀 | 분산 4셀 | 합계 | 비용 |
|---|---|---|---|---|
| 1.00× | 6,284 s | 2,068 s | **2.57 GPU-h** | **$3.80** |
| 0.50× | 8,546 s | 3,042 s | 3.47 GPU-h | $5.13 |
| 0.25×(rev5 최악 코너) | 14,061 s | 4,989 s | 5.54 GPU-h — 하드캡 초과, 분산 셀이 입장 규칙으로 잘린다 | (상한 $6.66) |
| 참고: 908623의 λ\* 비율(A 1.454×, B 1.033×) | — | — | 2.46 GPU-h | $3.64 |

- 참고 행은 **이 기판의 사전값이 아니다**. KISTI 값으로 VESSL 예산을 "예측"하지 않는다. 비용 규모 감각용이다.
- 표 밖 비용: 스케줄링·이미지 pull(B5에서 1분 미만), clone+install(B5에서 약 15 s), 결과 반출(908623 규모 약 400 MB를 Object FUSE로, 미측정 — 수 분으로 예상).
- **Job 분할**: Job 1개(rev5 11셀 + 분산 4셀). 운영모델 §1-3의 "Job당 ≲3–4 GPU-h"에 1.00×·0.50× 시나리오가 들어간다.
  선택 Job 2(§5-4)는 같은 크기를 한 번 더 쓴다.

### 6-2. 하드캡 = 스크립트 내부 wall-clock 집행기

E2 이식성 판정(`VERDICT_e2_substrate_portability_2026-09-18.md` Q2(c))은 SLURM이 사라지면 `--time` 하드캡도 사라지고,
boot 사이 검사만으로는 한 boot가 캡을 넘을 수 있다고 지적했다. 이식본은 세 겹으로 막는다.
1. **캡**: job_entry 시작 시각(`/data/runs/<job>/meta/timing.txt`의 `start_utc`)부터 **16,200 s = 4.5 h**($6.66).
   rev5의 `--time 04:30:00`과 같은 값이다. 상수는 `lambda0_vessl_plan.py` `HARD_CAP_S`에 있다. 측정 모드에서는 env로 바꿀 수 없다(`PDMUX_LAM0V_TEST_HARD_CAP_S`는 DRYRUN 전용, 아니면 `ABORT_TEST_KNOB_IN_MEASUREMENT`).
   셀 실행 마감은 캡에서 tail 예비 600 s(라벨·요약)를 뺀 시각이다.
2. **입장 규칙**(셀 시작 전): 남은 시간 ≥ 그 셀의 0.25× 코너 비용이어야 시작한다(`--need`).
   부족하면 `NOT_ATTEMPTED_BUDGET`을 기록하고 이후 셀을 모두 건너뛴다.
3. **실행 중 상한**: warmup·bench를 `timeout -k 30 <남은 시간>`으로 감싼다.
   별도 watchdog이 마감 시각에 `BUDGET_HARDCAP_FIRED`를 쓰고 bench·서버를 종료한다. `wait_health`도 이 플래그를 본다.
- VESSL 크레딧 차단(잔액 ≤0)은 **보조**일 뿐이다. 집행 주체는 이 스크립트다.
- **검증 방식(변이)**: 가짜 셀(DRYRUN, 같은 timeout·watchdog·입장 기계)로 시험한다.
  - 정상: 마감을 넘길 셀이 마감에서 잘리고(rc 124), 이후 셀은 `NOT_ATTEMPTED_BUDGET`이다.
  - 두 kill 층(셀별 `timeout`·watchdog)을 **모두** 제거한 변이: 셀이 끝까지 돌아 마감을 넘긴다(검출됨).
  - 한 층만 제거한 변이: 다른 층이 여전히 자른다. 즉 **등가 변이**다(교훈 254). 이 중복은 의도된 것이며 숨기지 않는다.
  - 입장 규칙은 구조 테스트(서버 기동보다 앞에 있는가)와 그 제거 변이로 고정한다.
- 캡 밖에 남는 시간: 집행기 종료 후 라벨·요약(tail 600 s 안), 그리고 job_entry의 반출. 컨테이너 총 시간 상한 ≈ 4.5 h + 반출.

### 6-3. 잘린 셀의 의미

- 예산 집행기가 bench를 죽인 셀(`rc 124/137` 또는 플래그)은 analyze하지 않는다.
  잘린 기록이 셀이 되면 처리율이 왜곡되기 때문이다.
- 그 셀은 라벨에서 "누락 셀"이고 rev5 D23대로 그 shape는 `UNRESOLVED`다. 이는 측정 실패이며 규칙 판정이 아니다(게이트 #21).
- 분산 셀이 잘리면 `UNRESOLVED_N_BELOW_4`다.

### 6-4. 재개

- 자동 재시도는 없다(VESSL Job). 재개는 **새 Job**에 `--env PDMUX_LAM0V_RESUME_FROM=<이전 job name>`을 준다.
- 조건: 이전 Job의 `plan.json`이 바이트 동일하고 등록 digest가 같아야 한다. 아니면 `ABORT_RESUME`.
- 이전 Job에서 `cell_<NAME>.complete`가 있는 셀만 복사하고 다시 돌리지 않는다. 출처는 `RESUMED_CELLS.tsv`에 남는다.
- 재개된 결과는 **Job 혼합**이다. 결과 보고에 반드시 적는다(운영모델 §3 "실패 셀만 골라 다시 돌린 사실은 반드시 적는다").
- 같은 Job 안 재실행(Workspace 디버깅)에서도 `.complete`가 있는 셀은 건너뛴다.

### 6-5. 중단 코드 (GPU 작업 전)

| 코드 | exit | 조건 |
|---|---|---|
| `ABORT_ENV` / `ABORT_TEST_KNOB_IN_MEASUREMENT` | 2 | §2-4 |
| `ABORT_STAGE` | 3 | staging 입력 없음 |
| `ABORT_F6_DRIFT` / `ABORT_F6` | 3 | §1-1 digest 불일치, 사전등록 파일 없음, digest 10개 미만 |
| `ABORT_MODEL_REVISION` | 3 | refs/main ≠ `dc0661c8…` |
| `ABORT_D12` / `ABORT_ENGINE_TREE_DRIFT` | 3 | manifest 미달 / 908623과 다름 |
| `ABORT_D18` | 2 | 술어 무응답 또는 `ANCHORED` |
| `ABORT_MUTATION_ESCAPE` / `…_HARNESS_CANNOT_RUN` | 5 / 6 | rev5 그대로 |
| `ABORT_PLAN_DRIFT` / `ABORT_CELL_COUNT` | 3 / 2 | §1-1 |
| `ABORT_RESUME` | 2 | §6-4 |

---

## 7. (f) 908623 교락의 기록 — 실현 분할을 telemetry로 남긴다

**908623 결과 감사가 지적한 것**(§4-A): λ\*(A)를 낸 a_r4의 decode-busy 시간 대부분이 비분할 `(0,108)`이었다.
legacy 루프는 split-prefill이 in-flight가 아니면 마지막 stream group으로 돌아간다(`multiplexing_mixin.py:1199-1200` 당시 인용).
그래서 "fixed D44에서 쟀다"는 shape A에서 거짓이다. 이식본은 이 교락을 **고치지 않는다**.
고치면(sticky ON) rev5와 다른 arm이 되기 때문이다. 대신 **기록한다**.

1. **telemetry 케이던스는 908623과 같다.** `PDMUX_TELEMETRY_PATH`만 셀마다 설정하고,
   `PDMUX_DUAL_WORKER_TRACE_EVERY`(기본 32)·`PDMUX_TRACE_FORCE_PREFILL`·`PDMUX_MEM_TELEMETRY`는 unset이다.
   - 그래서 관측자 부하는 908623과 같은 설계다. 새로 추가된 관측자 효과는 없다(VC-6의 파일시스템 차이는 별도).
   - **매 iteration 기록(`EVERY=1`)은 켜지 않는다.** 켜면 측정 arm의 관측자 부하가 908623과 달라져 비교 가능성이 깨진다.
     그 부하의 크기도 미측정이다(게이트 4의 대칭 on/off 측정이 없다).
     908623의 32-샘플 격자로도 실현 분할은 네 추정량이 같은 방향으로 판정할 만큼 분해됐다.
2. **셀마다 실현 분할 요약**: `results/r2_eval/e2_sticky_prereg/e2_realized_mix.py`(E2 트랙에서 감사된 추정량)를 telemetry에 돌려
   `mix_<cell>.json`을 남긴다.
   - 내용: D44(stream index 2) 점유를 네 추정량(E-cnt·E-time·E-iter·E-qcond)으로, E-pact 항등 가드, index 히스토그램.
   - **기술용이며 판정 입력이 아니다.** 실패해도 셀은 유효하다(`MIX_FAILED … (descriptive only)`).
   - 그 도구의 selftest는 908623 아카이브를 요구하므로 Job 안에서는 돌리지 않는다. 로컬에서 `SELFTEST_OK 7/7`을 확인했다(§9).
3. **green-context read-out**: boot마다 `PDMUX_GREEN_READOUT=1`로 각 stream group의 green context SM 수를 드라이버에 묻고
   `green_<cell>.json`에 쓴다.
   - "index 2 group이 실제로 44 SM green context인가"를 그 서버 프로세스 안에서 확인한다(Stage 0 D108 유형 방지).
   - "decode가 얼마 동안 그 group에서 돌았나"는 2번(telemetry)이 답한다. read-out은 이 질문에 답하지 않는다.
4. **필수 병기(등록)**: λ\*\_VESSL(A 또는 B)를 인용할 때는 그 값을 낸 셀의 D44 점유를 네 추정량으로 함께 적는다.
   "fixed D44에서의 용량"이라고 쓰지 않는다. telemetry 파일이 없거나 비면 "실현 분할 미기록"이라고 적는다.

---

## 8. 감사자가 봐야 할 자유 표면 (자기 공시)

1. **VESSL boot 여유 240 s**: 입장 규칙과 예산표만 바꾼다. 라벨은 바꾸지 않는다. 너무 작으면 마지막 셀이 실행 중 잘리고, 너무 크면 셀이 미리 빠진다.
2. **입장 비율 0.25×**: rev5 최악 코너를 그대로 썼다. 더 낙관적이면 분산 셀이 더 자주 시도되지만 실행 중 잘릴 위험이 커진다.
3. **하드캡 4.5 h**: rev5 `--time`과 같은 값이다. 단 시계 시작점이 다르다. KISTI는 SLURM 시작, 여기는 job_entry 시작(clone·install 포함)이다.
4. **분산 시드 3·4위 선택**: 1·2위와 같은 채점으로 기계적으로 정했다. 다른 시드면 Ēbar가 달라 도착 실현이 달라진다.
5. **분산 셀 순서·위치**(rev5 뒤, shape 교차): 시점 효과와 seed 효과가 섞인다(§5-3에 공시).
6. **분산 요약의 포함 조건**(포화 ∧ R0 유효): 포화가 아닌 반복이 나오면 n이 줄어든다. 그 자체가 정보다(F4 `still_saturated`와 같은 논리).
7. **§4의 4번 "다르다" 조건**(3% ∧ 범위 비중첩): 결과 보기 전에 고정했다.
8. **잘린 셀 analyze 생략**: rev5에는 없던 분기다. 발화 조건은 예산 집행기뿐이다.
9. **green read-out 추가**: 1회성 관측. 관측자 효과 0을 주장하지 않는다.
10. **엔진 트리 대조를 중단 조건으로 둔 것**: 다른 트랙의 커밋이 manifest 25파일 중 하나라도 바꾸면 이 Job은 돌지 않는다. 의도된 동작이며, 그때는 새 追記가 필요하다.
11. **staging 운반**: 입력 바이트는 digest로 강제된다. 운반 경로는 판정에 무관하다.
12. **unset 목록에 2개 추가**: 둘 다 908623에서 미설정 기본값이었다. 동작은 바뀌지 않는다.
13. **`timeout`·`setsid` 추가**: 발화하지 않으면 동작이 바뀌지 않는다. `setsid`는 서버의 세션만 바꾼다.

---

## 9. 로컬 검증 (GPU 0, 2026-10-04)

- `bash -n` 통과(`lambda0_vessl.sh`, `stage_lambda0_inputs.sh`).
- `lambda0_vessl_plan.py --selftest` 통과: 908623 `plan.json` 재현, 분산 행, Ēbar 가드, 110 s/boot에서 rev5 예산 재현, rev5 부분의 0.25× 코너 하드캡 적합.
- digest 대조: rev5 11파일 + 판정 입력 5개 + config 1개가 908623 기록과 전부 일치.
- `e2_realized_mix.py --selftest --archive lambda0_prereg/lam0_908623`: `SELFTEST_OK 7/7`.
- 공개 이미지 컨테이너(`sglang-runtime:75758e8` = `1e0d335b`, GPU 없음) dry-run, 변이 검사, 회귀 결과는 이 문서가 아니라 engine-porter 보고에 적는다.
  이유: 이 문서의 digest가 그 실행 기록에 들어가므로, 실행 결과를 다시 이 문서에 적으면 digest가 바뀐다.

## 10. 이 회차가 하지 못하는 것 · 제출 게이트

- 이 Job은 **게이트 #6을 닫지 않는다.** λ\*\_SLO는 이 Job의 범위 밖이다.
- 노드 간 산포는 Job 1개로 측정되지 않는다(§5-4).
- 관측자 효과(게이트 4)는 측정하지 않는다.
- **제출 게이트**:
  1. 이 문서가 규칙층 `GO`(또는 `GO-with-caveats`)를 받는다.
  2. 메인 세션이 커밋한다. 커밋 후 `REGISTRATION_SHA256.txt`가 기록할 digest를 대조한다.
  3. `stage_lambda0_inputs.sh --upload`로 판정 입력 3개를 올린다(스토리지만, GPU 0).
  4. CPU 회귀 통과, `check_citation_stops.py` 0위반.
  5. 사용자가 `launch.sh` dry-run 출력(명령·단가·job name)을 보고 승인한다.
