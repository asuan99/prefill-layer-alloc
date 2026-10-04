# 사전등록 追記 A13 — E2 (sticky 분할 대조) VESSL 기판 이식 (2026-10-04)

> **지위**: `e2_sticky_prereg/PREREG_E2_STICKY_REV3_2026-09-15.md`(sha `8998d532…`)와 그 구속 追記
> `PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md`(A1–A12, sha `a9c872ab…`)에 대한 **기판 이식 追記 A13**이다.
> rev3·ADDENDUM의 결정 규칙(R1–R6)·추정량(§5)·셀·seed·arm·실행 순서·라벨 어휘는 **한 글자도 바꾸지 않는다.**
> A13이 바꾸는 것은 실행 기판·배선·예산 집행기·기판 상수(P7 `T_cap`, §8 per-boot)이며, 각 변경은 아래에 전수로 적는다.
> 충돌 시 A13이 rev3·ADDENDUM의 **해당 조항만** 대체한다(대체 조항은 §8 색인).
> - 체크리스트: `e2_sticky_prereg/VERDICT_e2_substrate_portability_2026-09-18.md`(이식성 판정서) Q1 24항목·Q2 (a)–(f)·
>   "미결 해소 조건" 1–5·실현가능성 자기검사·E2C-36…39. 대응표는 §9.
> - 사용자 결정(2026-10-04): E2를 VESSL에서 실행한다. 새 범위 한정 OVERRIDE 승인 범위 **20 boot, 컨테이너 하드캡 ≈4 h**
>   (`OVERRIDE_E2_VESSL_2026-10-04.md`). 5-a/5-b arm은 sticky OFF로 정의한다(sticky ON은 프로파일 전용) — E2 자체에는 영향 없음.
> - 이 문서는 **claims-auditor 규칙층(이식) 감사 대기**다. `GO` 이전 `launch.sh --submit` 금지.
> - 작성 시점 GPU 지출 0 · 성능 판정 0건 · Claim 등급 변경 0건. 작성: engine-porter. 커밋은 메인 세션이 한다.
>
> **파일**(모두 `workspace/engine-port/results/r2_eval/e2_sticky_vessl/`)
> - `e2_sticky_vessl.sh` — 러너. `e2_sticky_prereg/e2_sticky.sbatch`의 VESSL 이식본.
> - `e2_vessl_plan.py` — 등록 상수(digest·`T_cap`·예산·하드캡)와 검사(digest·아카이브·엔진 트리·기판·green 요약). 규칙·추정량 코드는 없다.
> - `stage_e2_inputs.sh` — git 밖 셀프테스트 입력(908623 telemetry 7개)을 Object 볼륨에 올리는 로컬 도구(기본 dry-run).
> - `OVERRIDE_E2_VESSL_2026-10-04.md` — 새 범위 한정 OVERRIDE 초안.
> - 테스트: `workspace/engine-port/tests/test_e2_vessl.py`.
>
> **줄 번호 인용 규약**: rev3 계열 파일은 digest로 고정돼 있으므로 `파일:줄`로 인용한다. 이 회차의 새 파일은
> 아직 커밋되지 않았으므로 **코드 텍스트**로 인용한다.

---

## 0. 한 줄 요약

E2-α(대조 arm을 같은 Job 안에)를 VESSL A100 80GB 한 Job에서 돈다. 20 boot · 같은 규칙 · 같은 추정량 · 같은 셀.
바뀌는 것은 (1) 엔진 트리 = HEAD(908623 트리 + `multiplexing_mixin.py` 1파일, 기본 OFF 계측 훅), (2) P7 `T_cap`·§8 예산의
기판 상수를 VESSL λ0 실측으로 교체, (3) SLURM `--time` 대신 스크립트 내부 wall-clock 하드캡(4.0 h), (4) 기록 추가
(green read-out·cpu.stat·epoch 타이밍)뿐이다. **VESSL 결과는 KISTI 908623과 나란히 놓기만 하고 합치지 않는다.**

---

## 1. (a) rev3·ADDENDUM과 같은 것 — 강제 수단

### 1-1. 판정 경로 (바이트 동일, 위반 시 GPU 0에서 중단)

| 항목 | 출처 | 이식본 | 강제 |
|---|---|---|---|
| 추정량 `e2_realized_mix.py` · 규칙 `e2_label.py` | 커밋 `b917bfc`(2026-09-15 OVERRIDE 제출 커밋, 이후 변경 0) | `$PREREG/…`로 **호출**(복사 없음) | `e2_vessl_plan.py` `REGISTERED_DIGESTS` ↔ 실제 sha256. 하나라도 다르면 `ABORT_REG_DRIFT`(exit 3) |
| rev1·rev2·rev3·ADDENDUM · 판정서 6개(규칙 3·하네스 2·이식성 1) · 2026-09-15 OVERRIDE · `e2_sticky.sbatch`(실행하지 않음) | 같은 커밋(이식성 판정서는 2026-09-24 스코프 정정 반영본) | 기록 | 같은 `REGISTERED_DIGESTS`(14파일) |
| **A13 자신** | 이 문서 | — | `A13_DIGEST`. A13을 고치면 상수도 같은 커밋에서 고쳐야 하고, 아니면 `tests/test_e2_vessl.py`가 실패하고 Job은 `ABORT_REG_DRIFT` |
| 서빙 config `pdmux_homog5.yml` | `be6dd647…`(2026-09-09 `7e924a2` 이후 불변) | 같은 경로 | `CONFIG_DIGEST` |
| §5-3 추정량 셀프테스트(7/7 바이트 일치) | rev3 §5-3 + ADDENDUM A1 | **Job 안에서** `--selftest --archive` 실행(§2-5) | 실패 = `ABORT_SELFTEST`(exit 4). 입력 telemetry 7개도 digest 강제(`ABORT_ARCHIVE_DRIFT`) |
| 규칙 셀프테스트(8 도달성 + 10 되돌림 회귀) · `test_sticky_partition`(설치 트리) | `e2_sticky.sbatch` selftest 블록 | 같은 명령 | 같음(exit 4) |
| 라벨 호출 `e2_label.py --cells-dir $OUT --expect <10쌍>` | `e2_sticky.sbatch` 말미 | 같은 명령. `--expect` 문자열은 `e2_vessl_plan.expect_pairs()`가 rev3 §2 순서로 만든다 | 구조 테스트 |

### 1-2. 서빙 arm·명령줄·게이트 (코드 텍스트 동일 — 구조 테스트가 `e2_sticky.sbatch`와 대조)

| 항목 | 값 |
|---|---|
| 통제 요인(rev3 §3) | `MODEL`·`BACKEND=flashinfer`·`CFG`·`CTX=16384`·`MEM_FRACTION=0.82`·`MAX_RUNNING=48`·`SERVER_SEED=1`·`FIXED_DSM=44`, cudagraph ON, `--disable-radix-cache`·`--disable-overlap-schedule`·`--chunked-prefill-size -1`, `PDMUX_R2_POLICY=fixed` |
| 모델 리비전 | `dc0661c829b14e5b9246c05cfa89094a0875e052`. `/data/hf` refs/main이 다르면 `ABORT_MODEL_REVISION` |
| trace | `$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json`, sha `35f0e213…`(λ0 VESSL Job이 읽은 바이트). 다르면 `ABORT_TRACE` |
| server / warm-up / bench 플래그 집합 | `e2_sticky.sbatch`와 동일(테스트가 플래그 다중집합을 추출해 대조) |
| **THE ONE KNOB** | OFF = `PDMUX_STICKY_PARTITION` unset / ON = `1`. per-boot 설정 문장은 sbatch와 같은 텍스트 |
| 20 boot 순서 · seed `4386/4162/251/2630`(B는 앞 둘) | rev3 §2 리터럴. `e2_vessl_plan.units()`가 생성, 테스트가 sbatch의 생성 루프와 대조 |
| P1·P2(ON 로그 `ENABLED`, `fixed target index=2`; OFF 로그에 없음) · P3(binding resource 동일, boot마다) · P8(`run_id` ↔ `--arm`, ADDENDUM A11(c)) · P9–P12 · P13(양 arm, 프롬프트 6개 바이트 동일, `text` 리스트 동등, 빈 출력 `ABORT_CORRECTNESS`) | 같은 문장. P13 프로브 Python 본문은 sbatch와 **바이트 동일**(테스트) |
| ADDENDUM A3(프로브 시각 이후 레코드만) · A2(decode-busy 리터럴) · A6/A11(짝 `UNRESOLVED`) · A12(도메인 공시) | 전부 `e2_realized_mix.py`·`e2_label.py` 안에 있고 그 파일은 digest로 고정 |
| 추정량 호출 인자(`--probe-end`·`--bench-rc`·`--num-prompts`…) | 같은 텍스트(테스트) |
| D23(짧아진 집합으로 재라벨 금지) · boot 실패 2연속 → 이후 미시도 | 같음 |

### 1-3. ★엔진 트리 결정 — **HEAD를 쓴다** (지시 1)

**사실**(GPU 0, 이 회차 재확인):
- `git diff --stat 4d1a0ec 1d47d3e -- workspace/engine-port/src/` = **공백**. E2 rev3 기준 트리(`4d1a0ec`) = λ0 VESSL 트리(`1d47d3e`) = job 908623 트리
  (908623 manifest의 `multiplexing_mixin.py` = `e2a97b42…` = `1d47d3e` 소스 미러).
- `1d47d3e..HEAD`의 `src/` 변경은 `0922512`(E-1 사전등록 커밋)의 `multiplexing_mixin.py` 1파일뿐이다(`ba5835aa…`).
  내용 = `PDMUX_PHASE_EVENTS`(기본 OFF) 훅 다섯 곳을 **같은 줄 편집**(`; hook(...)`)으로 넣고 본문은 파일 끝에 붙였다.
  OFF에서 각 훅은 속성 검사 1회 후 반환한다. `dual_worker_trace_error_logged` 줄은 `_init_phase_events()`의 반환값 `False`를 받는다(종전 값과 같다).
  줄이 밀리지 않으므로 rev3 §4-1의 코드 인용(`:1163-1211`, `:1169-1171` 등)은 HEAD에서도 같은 줄을 가리킨다(이 회차 확인).

**결정**: 러너는 sync 후 manifest를 908623 manifest와 경로 접미별로 대조하고(`e2_vessl_plan.py --engine-diff`),
**다른 파일이 정확히 `sglang/srt/multiplex/multiplexing_mixin.py` 하나이며, 그 파일의 이쪽 digest가 `ba5835aa…`(HEAD 등록값),
908623 쪽이 `e2a97b42…`일 때만** 진행한다. 0개 차이(908623 트리 그대로)·2개 이상 차이·다른 digest는 모두 `ABORT_ENGINE_TREE_DRIFT`(exit 3).
`PDMUX_PHASE_EVENTS`는 허용 목록 밖이라 설정되면 `ABORT_ENV`이고, 러너는 unset 목록에도 넣었다.

**E2-α 판정에 면역인 이유**: R1–R5·P13은 전부 **같은 Job 안 OFF 대 ON**이다. 두 arm은 같은 sync·같은 manifest·같은 프로세스 이미지로 부팅하므로
1파일 차이는 두 arm에 **똑같이** 실린다. OFF에서 그 파일이 하는 일(훅당 속성 검사 1회)은 arm 사이에 다르지 않다.
트리 일치가 의미를 갖는 곳은 908623과의 교차 비교(R6)뿐이고, R6은 rev3 §6이 이미 "판정 아님"으로 강등했다 → **E2V-1**로 공시.

**대안(`1d47d3e` 제출)을 쓰지 않는 이유**: 그 커밋에는 이 러너·A13·OVERRIDE가 없고(새 파일), `launch.sh`의 tag 밑줄 수정(`846858a`)과
`job_entry.sh`의 cgroup CPU 기록(`8addb9f`)도 없다. 그 커밋에 이식 파일을 얹으면 감사되지 않은 새 분기가 생긴다.

---

## 2. (b) rev3·ADDENDUM과 다른 것 — 전수

### 2-1. 기판

| 축 | KISTI(908623, E2 등록 기준) | VESSL | 근거 · 강제 |
|---|---|---|---|
| GPU | A100-SXM4-80GB, 108 SM, gpu38 | A100-SXM4-80GB, 108 SM, cc 8.0, MIG off | **B0 게이트**: `nvidia-smi` 이름에 `A100`·`80GB`, cc `8.0`, 메모리 ≥ 79,000 MiB, driver major ≥ 550, MIG off, torch SM 수 108. 하나라도 아니면 `ABORT_SUBSTRATE`(이식성 판정서 해소 조건 1) |
| 격자 실현 | 미실측 | B3a·B3c(2026-10-04, 빈 GPU)에서 (64,44) 포함 격자 = 요청 SM, 서로소, cudagraph replay 중 격리 유지 | 부하 중 실현은 boot마다 green read-out(§2-7) |
| 드라이버 | 908623 자신은 미기록 | 580.105.08(B0·B5·λ0 VESSL) | 버전 문자열 동일이어도 플랫폼 패치 빌드 동일성은 미확인 |
| 노드 | 전용(SLURM) | 공유 노드, Job마다 다를 수 있음 | E2-α상 판정은 노드 축에 면역. R6만 영향(E2V-1) |
| 클럭 | 미기록 | 고정 불가. job_entry가 100 ms마다 `meta/clk.csv` 기록 | λ0 VESSL Job에서 SW power cap(0x4) 노출 17.8%(Job 전체). E2와 같은 셀의 bench 창: `a_r4` 4 seed 25.8–27.5% · `a_r2` 28.8% · `b_r3` 4 seed 49.0–49.4%(`lam0v_descriptive.json` `clk.cells.*.bench_window`) |
| CPU | `--cpus-per-task=8`(cgroup) | `nproc` 96 가시, cgroup `cpu.max` = 11 CPU quota, 핀 없음 | §2-6 |
| telemetry 쓰기 | Lustre | CephFS `/data/runs/<job>/results/…` | λ0 VESSL 691,307 레코드에서 `dropped_events` 전부 0. 이 Job도 셀마다 기록(LV-12 계열) |

### 2-2. ★P7 `T_cap` — **VESSL 값으로 교체한다** (지시 2)

rev3 §7 P7 문안 "`T_cap = 3 × (908623 OFF 실측 duration)`"을 다음으로 대체한다.

> **`T_cap = ⌈3 × round(d, 0.1 s)⌉`, `d` = λ0 VESSL Job `pdmux-lambda0-vessl-1d47d3e-20261004084945`의 같은 셀 · seed 4386 · sticky OFF 측정 duration.**
> `a_r4` d = 133.615 s → **401 s** · `a_r2` d = 142.902 s → **429 s** · `b_r3` d = 289.667 s → **870 s**.
> (출처: `lambda0_vessl/analysis_2026-10-04/lam0v_descriptive.json` `vessl.<cell>.cell.duration_s`, `client_seed` 4386. 이 회차 직접 확인.)

- **배수 3×는 불변.** 반올림 규약도 KISTI와 같다: `e2_sticky.sbatch`의 393/428/861은 ⌈3 × (rev3 §8-1 표의 0.1 s 반올림 duration 131.0/142.5/286.9)⌉이다.
  원시 duration에 바로 ⌈3d⌉를 적용하면 `a_r4`가 394가 되어 sbatch 리터럴과 어긋난다. `e2_vessl_plan.t_cap()`은 반올림 규약을 쓰고, 셀프테스트가 KISTI 리터럴 3개 재현을 확인한다.
- **근거**: 기판 상수는 그 기판에서 잰다. λ0 VESSL의 그 셀은 E2 OFF arm과 같은 조건이다(같은 셀·seed·sticky OFF·같은 서버 플래그. 차이는 엔진 훅 1파일[OFF]과 P13 프로브[첫 boot만]).
- **판정에 닿는 경로**: `T_cap`은 절단 가드다. 두 선택(908623 값 / VESSL 값)은 ON bench duration이 `a_r4` [393, 401] s, `a_r2` [428, 429] s, `b_r3` [861, 870] s
  안에 떨어질 때만 다른 결과(CAPPED vs 정상)를 낸다. 즉 ON이 OFF의 약 2.94–3.0배일 때다. 등록 최악 코너(A ×2.5, B ×1.5)는 두 선택 모두에서 캡에 닿지 않는다.
  VESSL 값이 1–9 s 느슨하다(완화 방향, ≤ 2.1%). 이 방향을 자유 표면 §6-2로 공시한다.

### 2-3. 실행 배선과 추가물

| 항목 | `e2_sticky.sbatch` | 이식본 | 종류 |
|---|---|---|---|
| 경로 | `ROOT=/scratch/…`, `module load`, venv `source`, `#SBATCH` 헤더 | `ROOT="${PDMUX_PROJECT_ROOT:?}"`, `HF_HOME`·캐시는 job_entry 값, module/venv 없음. `#SBATCH` 헤더는 **무효**(VESSL은 읽지 않는다) | 배선 |
| 결과 위치 | `e2_sticky_prereg/e2_<SLURM_JOB_ID>/` | `e2_sticky_vessl/e2v_<job name>/`(job_entry 반출, `fetch.sh` 병합). 원장 `results/e2_sticky_vessl/vessl_jobs.tsv` | 배선 |
| §5-3 셀프테스트 입력 | KISTI 디스크 | `/io/staging/e2_selftest_archive/`에서 digest 검증 후 원위치로 `cp -p`(§2-5) | 배선 |
| 예산 집행 | SLURM `--time 03:00:00` + P14 | **스크립트 내부 wall-clock 집행기**(§5-3) | ★추가물 1 |
| warm-up 타임아웃 | 없음 | `timeout -k 30 <남은 시간>` | 추가물 1의 일부. 발화하지 않으면 동작 무변 |
| bench 타임아웃 | `timeout "${TCAP}s"` | `timeout -k 30 "${BT}s"`, `BT = min(T_cap, 남은 시간)`. `T_cap`으로 잘리면 CAPPED(P7), 남은 시간으로 잘리면 BUDGET | 추가물 1의 일부. `-k 30`은 SIGTERM 무시 시 30 s 뒤 SIGKILL(rc 137)이며 둘 다 같은 분지로 처리 |
| P13 프로브 | 무제한 | `timeout -k 10 <남은 시간>` 래퍼. 본문 바이트 동일 | 추가물 1의 일부 |
| 서버 기동 | `python -m sglang.launch_server … &` | `setsid python -m …` | 세션만 바꾼다(watchdog의 `pkill`이 서버를 확실히 잡도록) |
| boot 실패 · P1/P2/P3/P13 abort 시 정리 | `kill $PID` | 같은 teardown 순서(kill, 5 s, `pkill -9`, 10 s)를 abort 전에 실행 | 고아 서버의 GPU 과금 방지(이식성 판정서 자기검사) |
| green read-out | 없음 | boot마다 `PDMUX_GREEN_READOUT=1`(양 arm) | ★추가물 2(§2-7) |
| cgroup `cpu.stat` | 없음 | boot마다 bench 직전·직후 스냅샷 `cpu_<boot>.txt` | ★추가물 3(§2-6) |
| epoch 타이밍 | 없음 | `TIMING.tsv`(입장·health·bench 시작/끝·종료, rc, 절단 종류) — `clk.csv` 정렬용 | ★추가물 3 |
| 기판 기록 | 없음 | `SUBSTRATE_E2V.txt`·`SUBSTRATE_CHECK.txt`(B0) | ★추가물 4 |
| 재개 | 없음 | `PDMUX_E2V_RESUME_FROM`, **짝 단위**(§5-4) | ★추가물 5 |
| PDMUX_* 노브 | 고정 unset 목록 | 허용 목록 밖이면 `ABORT_ENV`(§2-4) + unset 목록에 `PDMUX_DUAL_WORKER_TRACE`·`PDMUX_MEM_TELEMETRY`·`PDMUX_PHASE_EVENTS` 추가(셋 다 908623에서 미설정 기본값) | 강화 |
| `PDMUX_E2_PREREG` 운영자 노브(사전등록 경로 재지정) | 허용 | **제거**(허용 목록 밖 → `ABORT_ENV`) | 강화 |
| 바이트코드 | 기록 없음 | 종료 시 `e2_sticky_prereg/`·`e2_sticky_vessl/`의 `__pycache__` 삭제(job_entry가 results 아래 새 파일을 반출) | 배선 |
| 실행 기록의 트리 상태 | `uncommitted=` | `uncommitted_outside_results=`(λ0 VESSL과 같은 이유) | 배선 |

### 2-4. 노브 허용 목록

- 허용: job_entry/launch 식별자(`PDMUX_COMMIT`·`_JOB_NAME`·`_CAMPAIGN`·`_TARGET`·`_IMAGE`·`_SPEC`·`_ROOT`·`_PROJECT_ROOT`·`_IO`·`_DATA`·`_OPT`), `PDMUX_E2V_RESUME_FROM`, `PDMUX_E2V_DRYRUN`.
- **테스트 전용**(`PDMUX_E2V_TEST_*`, `PDMUX_TEST_SKIP_SYNC`): `DRYRUN=1`이 아니면 `ABORT_TEST_KNOB_IN_MEASUREMENT`.
- `PDMUX_ARRAY_SPEC`는 비어 있어야 한다(E2-α: 한 Job).
- **arm 노브 `PDMUX_STICKY_PARTITION`은 허용 목록 밖이다.** 러너가 boot마다 직접 설정·해제한다. 제출 환경에서 상속되면 중단한다.

### 2-5. git 밖 셀프테스트 입력의 운반

- rev3 §5-3 셀프테스트는 job 908623의 `tel_{a_r0,a_r2,a_r4,a_r4_s2,b_r0,b_r3,b_r3_s2}.jsonl`(약 238 MB)을 읽는다. `.gitignore`(`*.jsonl`) 대상이라 bundle에 실리지 않는다.
- 로컬 `stage_e2_inputs.sh`가 일곱 파일을 모아 등록 digest(`ARCHIVE_DIGESTS`)와 대조하고 `/io/staging/e2_selftest_archive/`로 올린다(`--upload`, 스토리지만).
- Job은 원위치(`lambda0_prereg/lam0_908623/`)에 없을 때만 `cp -p`로 놓고(mtime 보존 → 반출 안 됨) `--verify-archive`로 다시 대조한다. 없으면 `ABORT_STAGE`, 다르면 `ABORT_ARCHIVE_DRIFT`.
- λ0 VESSL은 이 셀프테스트를 Job 밖(로컬)에서만 돌렸다(그곳에서는 기술용이었다). **E2에서는 추정량이 판정 입력이므로 rev3 원 등록대로 Job 안에서 돌린다.**

### 2-6. ★CPU — **등가 핀을 걸지 않는다. 기록하고 caveat로 둔다** (지시 7, 이식성 판정서 Q2-(f))

- **사실**: VESSL 컨테이너는 `nproc` 96(가시), cgroup `cpu.max` = `1100000 100000` = 11 CPU quota, affinity 0–95(λ0 VESSL `SUBSTRATE_LAM0V.txt`).
  KISTI는 `--cpus-per-task=8`(cpuset 여부 미확인). 즉 코어 예산 축이 등록되지 않은 채로 바뀌었다(교훈 268).
- **결정**: `taskset`·cgroup 핀을 새로 걸지 않는다. 사유 3개:
  1. 핀은 새 통제 요인이다. 이 A13이 쓰는 VESSL 기판 상수(`T_cap`·per-boot·OFF 밴드 파일럿)는 **핀 없는** λ0 VESSL Job에서 왔다. 핀을 걸면 그 상수의 이식이 다시 깨진다.
  2. E2의 판정은 전부 같은 Job 안 paired 대조다. CPU quota·가시 코어는 두 arm에 같다.
  3. 이식성 판정서 (f)의 위험 방향은 "클라이언트 클램프가 양 arm에 같아 차분을 0 쪽으로 압축"하는 **검출력 손실**(`LAMBDA_WITHIN_CI` 쪽)이다. 부호 반전 경로가 없다는 주장은 **미검증**이다(서버와 클라이언트가 같은 quota를 공유하고 ON·OFF의 CPU 수요가 다를 수 있다 — E2VC-2). 그 라벨의 오독은 E2C-7이 이미 금지한다.
- **대신 기록한다**: job_entry가 시작·끝의 `cpu.max`·`cpu.stat`을 `meta/cgroup_cpu.txt`에 쓰고, 러너가 boot마다 bench 직전·직후 `cpu.stat`(`nr_throttled`·`throttled_usec`)을 `cpu_<boot>.txt`에 쓴다.
  결과 문서는 arm별 `throttled_usec` 증가량을 병기한다(E2V-4). 이 기록은 판정 입력이 아니다.
- 대안(감사자 선택지): `taskset -c 0-10`으로 가시 코어 = quota로 맞추는 안. 위 사유 1 때문에 권하지 않는다. 필요하면 별도 대조(λ0 VESSL 결과 감사 §7의 CPU quota 단일 노브 대조)로 따로 잰다.

### 2-7. green-context read-out (E2C-38 해소 — 부분)

- boot마다 `PDMUX_GREEN_READOUT=1`, `PDMUX_GREEN_READOUT_PATH=$OUT/green_<boot>.json`. **양 arm 모두.** 서버 프로세스 안에서 각 stream group의 green context SM 수를 드라이버에 1회 묻는다(`green_readout.py`, 908623 manifest에 이미 포함된 파일).
- λ0 VESSL Job이 이 설정으로 돌았으므로, 이 A13의 OFF arm은 908623보다 λ0 VESSL 파일럿에 더 가깝다(같은 env).
- 러너 말미 `GREEN_SUMMARY.json`: boot마다 index 2 group의 (prefill, decode) green smCount와 서버가 밝힌 sticky target index.
  **기술용이며 판정 입력이 아니다.** 결정 규칙에 연결하면 새 estimand가 된다(이식성 판정서 "미결 해소 조건" 머리말, 교훈 250).
- 해소 범위: "index 2 group이 실제로 (64,44) green context인가"에 답한다. "decode가 그 group에서 얼마 동안 돌았나"는 여전히 telemetry `stream_index`가 답한다.
  ⇒ E2C-38은 **"realized green SM 미측정" → "group 구성은 실측, 시간 점유는 `stream_index` 기준"**으로 좁혀 승계한다(§3-2).

---

## 3. (c) 판정 경로와 caveat

### 3-1. 이식 차이가 판정에 닿는 경로

| ID | 차이 | 판정에 닿는 경로 | 처리 |
|---|---|---|---|
| **E2V-1** | 엔진 트리 = 908623 + `multiplexing_mixin.py`(PHASE_EVENTS 훅, OFF) | 두 arm에 같다 → R1–R5·P13 무영향. R6(908623 교차 비교)에만 트리 차이가 실린다 | **R6 필수 병기**: "VESSL Job은 908623 트리 + `multiplexing_mixin.py` 1파일(`PDMUX_PHASE_EVENTS` 기본 OFF 훅) · 다른 클러스터 · 다른 드라이버 기록 상태 · 다른 패키지 설치 방식이다." R6은 계속 **판정 아님**. 이식성 판정서 Q2 R6 행의 "노드 축 → 클러스터·드라이버·패키지 축" 확대를 이 문장으로 이행한다 |
| **E2V-2** | P7 `T_cap` VESSL 값(+1–9 s) | ON bench가 OFF의 ≈2.94–3.0배일 때만 CAPPED 여부가 바뀐다 | §2-2. 결과 문서는 `T_cap` 값과 출처 Job을 병기 |
| **E2V-3** | 하드캡 집행기·`timeout`·`setsid` 추가 | 발화하면 짝이 빠진다 → D23대로 `UNRESOLVED`(측정 실패, 규칙 판정 아님) | 발화하지 않으면 동작 무변. `BUDGET_STOP.txt`·`TIMING.tsv`로 기록 |
| **E2V-4** | CPU quota 11 · 가시 96 · 핀 없음 | 클라이언트 클램프가 있으면 두 arm에 같이 걸려 차분이 0 쪽으로 압축(검출력 손실) | §2-6. arm별 `throttled_usec` 증가량 병기. `LAMBDA_WITHIN_CI`를 "차이 없음"으로 읽기 금지(E2C-7과 같은 방향) |
| **E2V-5** | 공유 노드·클럭 미고정·power cap | 느린 드리프트는 인접 OFF→ON에 거의 같이 실린다. **단조 드리프트는 ON 쪽에 실린다**(OFF→ON 순서 고정, E2C-7) | 결과 문서는 boot별 `clk.csv` 창(TIMING.tsv로 정렬)의 power cap 노출을 arm별로 병기한다. `LAMBDA_MOVES_*`를 기전으로 읽는 것은 E2C-1에 더해 이 caveat로도 금지(이식성 판정서 Q2-(e)의 "한 겹 더") |
| **E2V-6** | green read-out 추가(양 arm) | boot 시 1회 ctypes 조회. 스케줄링·배치·입장에 쓰이지 않는다 | 관측자 효과 0을 주장하지 않는다(λ0 VESSL LV-11과 같은 형식). 두 arm에 같다 |
| **E2V-7** | 재개(짝 단위) | 짝 안은 같은 Job이므로 R2의 paired 차분은 유지. R5(seed 간 산포)·R4″(seed 간)·R3·P3은 Job을 섞을 수 있다 | `RESUMED_PAIRS.tsv`(출처 Job·GPU UUID·드라이버). 재개가 있었으면 라벨 인용에 "Job 혼합(짝 단위)" 병기 필수. 재개된 결과로는 R5·R4″를 "같은 Job 안 산포"로 쓰지 않는다 |
| **E2V-8** | 모델·패키지·telemetry 파일시스템 | λ0 VESSL VC-3/VC-4/VC-6/VC-7과 같은 경로. 같은 Job 안 대조에는 면역, R6에만 실린다 | caveat로만(LV-11·LV-12 형식 승계: `dropped_events` 병기) |

### 3-2. E2C-36…E2C-39 — 결과 문서 승계 caveat로 **등록**한다 (지시 6)

이식성 판정서가 "판정서 참조로만 승계"로 둔 네 문장을 이 A13이 결과 문서의 **필수 승계 caveat**로 등록한다. 해소 상태를 함께 적는다.

> **E2C-36** — `a_r4`=8.0 · `a_r2`=3.0 · `b_r3`=1.15 req/s는 A100 D44 회귀에 앵커된 KISTI λ0 사다리의 rung이다. 셀 이름을 "고부하/중부하"로 해석 인용 금지.
> **해소 상태(부분)**: VESSL λ\*\_VESSL(A) = 2.9937 req/s(λ0 VESSL 라벨)이라 `a_r2`의 명목 offered 3.0은 1.002×λ\*\_VESSL(A)이다(KISTI 0.983×, LVR-3 승계).
> seed별 실현 offered(E2C-13: 2.996/2.955/2.950/2.951)로는 **0.985–1.001×λ\*\_VESSL(A)** — VESSL에서 `a_r2`는 무릎을 **양쪽으로 걸친다**. 같은 offered rate가 KISTI와 다른 ρ에 놓인다.

> **E2C-37** — 이 캠페인의 **어떤 판정 게이트도 기판 변화를 탐지하지 않는다**(`idx2 ↔ (64,44)` 대조는 `manual_divisions` 자기 대조, P3는 Job 안 일치만). "게이트가 기판 동일성을 확인했다" 인용 금지.
> **해소 상태(부분)**: B0 게이트(§2-1)가 A100 80GB·cc 8.0·108 SM·MIG off·driver ≥ 550을 **Job 시작 시 강제**한다. 이것은 "같은 종류의 기판"까지이며 "KISTI와 같은 기판"이 아니다.

> **E2C-38** — **이 캠페인은 decode가 시간 단위로 실제 44 SM green context에서 돌았는지를 직접 재지 않는다.** R1의 `STICKY_REALIZES_A`는 `stream_index`가 2였다는 뜻이다.
> **해소 상태(부분)**: boot마다 green read-out으로 "index 2 group = (64,44) green context"는 실측한다(§2-7, 기술용). 시간 점유는 여전히 `stream_index` 기준이다.

> **E2C-39** — 결과 문서는 "하드캡 안에서 끝났다"를 **실측 벽시계 없이 쓸 수 없다.**
> **해소 상태(부분)**: 하드캡 4.0 h는 스크립트 내부에서 세 겹으로 집행된다(§5-3). 단 이 캡은 **셀 구간 집행**이며 job_entry의 clone·install, 러너 tail 600 s, 반출에는 kill 층이 없다.
> 결과 문서는 `meta/timing.txt`의 `start_utc`·`end_utc`와 `BUDGET.txt`의 `remaining_at_loop_end_s`를 인용한다.

### 3-3. 908623 원자료에 기댄 caveat 수치 — VESSL 재산출 (지시 8)

전부 λ0 VESSL Job 원자료(fetch 완료, sha256 검증된 `tel_*.jsonl`)에 **쉬핑 `e2_realized_mix.py` 함수를 그대로** 적용한 값이다. 이 회차 산출.
VESSL 값은 **sticky OFF·seed별 1 boot**이며 KISTI 값과 **나란히만** 놓는다. 이 표의 어떤 값도 판정 입력이 아니다.

| caveat | KISTI 908623(등록 원문) | **VESSL λ0(재산출)** | 비고 |
|---|---|---|---|
| **E2C-21**(shape A 처치 시간 중 prefill 유휴 비율 = OFF decode-busy E-time 분모 중 idx4 가중) | `a_r4` 158.24/183.61 s = 86.2% · `a_r2` 174.63/195.24 = 89.4% · `b_r3` 2.3% | `a_r4` s1 161.05/187.27 = **86.0%**(4 seed 85.8–87.1%) · `a_r2` s1 170.80/196.59 = **86.9%** · `b_r3` 4 seed **2.4–2.6%** | 결론 불변: shape A 처치 시간의 약 86–87%가 prefill 유휴. 승계 문장의 "86–89%"는 VESSL에서 "85.8–87.1%"로 병기 |
| **E2C-6**(`a_r2` Q5 절벽) | OFF TTFT p95 1.806 s vs SLO 3.0 s | **2.840 s** vs 3.0 s(seed 4386) | 절벽에 **더** 가깝다(여유 0.16 s). 계단 함수로만 병기 규약 불변. LVR-3: 처리율 구별 불가 ≠ SLO 지표 동등 |
| **E2C-13**(seed가 실현 offered를 바꿈) | 2.996/2.955/2.950/2.951 = 0.981–0.966×λ\*(A)\_KISTI(3.0531) | 같은 계산값(기판 무관) = **1.001–0.985×λ\*\_VESSL(A)** | 해석만 바뀐다(E2C-36 해소 상태) |
| **E2C-20(a)**(인덱스별 표본 간격 중앙값, decode-busy → 다음 비-startup 스냅샷) | `a_r4` 532/208 ms · `a_r2` 413/265 · `b_r3` 333/209 | `a_r4` **563/214** · `a_r2` **461/252** · `b_r3` **342/213** | 비대칭(idx2가 2.6배 성김) 재현. R1 네 추정량 동시 성립 규칙(A8)의 근거 불변 |
| **E2C-20(b)**(busy→busy 간격 중앙값) | `a_r2` 315.7 · `a_r4` 272.5 · `b_r3` 332.6 ms | `a_r2` **314.4** · `a_r4` **294.4** · `b_r3` **341.4** ms | 규약 동일 |
| **E2C-16**(bench 전용 창 값, OFF 여유 0.85pp) | `a_r4` E-time 13.64→19.15 등 | **VESSL 미측정** | telemetry에 wall-clock이 없고 bench 창을 monotonic 축에 맞추는 규약이 등록돼 있지 않다. 사후 bench-only 재스코어로 R1 재라벨 금지(게이트 #8) 불변 |
| **E2C-19**(`b_r0` E-qcond ON 예보의 다리) | `idx0 ∩ prefill_queue_depth>0` = 8/45 | 0/45 | `b_r0`는 E2 셀 집합 밖이라 판정 영향 없음. E2C-19의 82.22%는 KISTI 값으로만 인용 |
| **E2C-9**(≈2 ms 격자 단독 인용 금지) | — | 규약 불변 | — |
| R3(ii) idx0 (OFF) | `a_r4` 2/555 · `a_r2` 0/610 · `b_r3` 1/441 | `a_r4` 0–2/554–557 · `a_r2` 0/597 · `b_r3` 0–3/440–442 | 5.0% 문턱에서 먼 쪽 |
| E-pact (OFF, ADDENDUM A6 가드) | 100%(24–35 표본) | **100%** 전 셀(`a_r4` 27–33/…, `a_r2` 33/33, `b_r3` 399–400/…) | 가드 발화 예보 없음 |
| R4 전제(`b_r3` OFF E-qcond) | 100.00%(395/395) | **100%** 4 seed(394–396/…) | R4가 기대는 "처치가 못 움직이는 양" 전제는 VESSL OFF에서 재현 |
| R5 전제(`a_r4` OFF E-cnt seed 간 산포) | n=2: 8.65/8.26 | 4 seed **7.93–9.16% = 1.23pp** | 5pp 문턱 아래 |

---

## 4. (d) 라벨·판정 규칙 — **불변**, 그리고 R1 OFF 밴드 사전 예보 (지시 5)

1. R1–R6·ADDENDUM A5(R1 4분지)·A6·A11·A12·D23·R2 부트스트랩 리터럴(`B=100000`, rng `20260915`)은 **그대로**다. `e2_label.py`는 digest로 고정이다.
2. **이름**: 이 Job의 값은 "E2 VESSL(Job 이름)"로 부른다. 기판 표기 없이 KISTI 값과 같은 문장에 넣지 않는다.
3. **R6**(908623 OFF 셀 교차 비교)은 **판정 아님**, E2V-1 병기. VESSL λ0 OFF 셀과의 비교도 같은 지위다(다른 Job).

### 4-1. ★R1 OFF 밴드 — **문턱·규칙은 바꾸지 않는다. 사전 예보로 공시한다**

**파일럿 사실**(λ0 VESSL Job, sticky OFF = E2 OFF arm과 같은 조건 — §2-2 단서 포함. 이 회차 telemetry 재산출로 `lam0v_descriptive.json`과 일치 확인):

| 셀·seed | E-cnt | E-time | E-iter | **E-qcond** | R1 OFF 절(< 20.0, 네 값 모두) |
|---|---|---|---|---|---|
| `a_r4` 4386 | 8.66 (48/554) | 13.89 | 8.76 | **18.33 (46/251)** | 통과(여유 1.67pp) |
| `a_r4` 4162 | 7.93 (44/555) | 12.89 | 8.01 | 15.94 (40/251) | 통과 |
| `a_r4` 251 | 9.16 (51/557) | 14.18 | 9.28 | 17.06 (43/252) | 통과 |
| `a_r4` 2630 | 7.93 (44/555) | 12.92 | 8.03 | 16.60 (42/253) | 통과 |
| **`a_r2` 4386** | 9.21 (55/597) | 13.12 | 9.30 | **20.57 (29/141)** | **실패(여유 −0.57pp)** |
| `a_r2` 4162·251·2630 | — | — | — | — | **VESSL 미측정**(KISTI도 미측정) |

(E-time 초: `a_r4` 4386 = 26.01/187.27 s, `a_r2` 4386 = 25.80/196.59 s. 분모 141에서 표본 1개 = 0.71pp.)

**사전 예보(등록)**:
> `a_r2`에서 `OFF_BASELINE_ABOVE_BAND`가 나올 가능성이 크다(VESSL 파일럿 n = 1 셀·1 seed, 여유 −0.57pp).
> 그 라벨은 **대조 arm(OFF)에 대한 진술**이다(ADDENDUM A5).
> ★**Q1은 R1 ON 절로 답해지지 않는다**(E2VC-1, `VERDICT_e2_vessl_rules_2026-10-04.md` 死因 1 수리): R1 라벨이 `STICKY_REALIZES_A`(또는 `cells_evaluated`를 병기한
> `STICKY_REALIZES_A_ON_REMAINING_CELLS`)가 아닌 한, 이 캠페인은 "sticky ON이 shape A의 실현 분할을 D44로 고정/유지/실현한다"를 **확립하지 않는다**.
> R1 ON 절의 성립은 엔진이 {0,2}로 강제한 **양성 대조**의 통과이며(E2C-3; live 짝에서는 R3(ii)가 먼저 적용돼 E-cnt(ON) ≥ 95가 구성상 성립) Q1의 답이 아니다.
> `OFF_BASELINE_ABOVE_BAND`를 "VESSL OFF가 KISTI OFF보다 D44를 더 점유한다"나 "sticky가 무의미하다"로 읽지 않는다.
> 결과 문서는 R1 라벨과 함께 **ON 절·OFF 절의 성립 여부를 셀·seed별로 분리해** 적고(`per_seed` 네 추정량·분자/분모), ON 절 성립을 R1 라벨 이름이나 Q1의 답으로 바꿔 쓰지 않는다.

이것은 **규칙 변경이 아니다.** `e2_label.py`가 이미 내는 필드(`on_clause_failures`·`off_clause_failures`·`per_seed`, ADDENDUM A5/A12)를 **보고**하는 규약이며, Q1의 판정은 rev3 §6 R1 라벨 그대로다.

**왜 문턱을 옮기지 않는가**: (i) 이식성 판정서의 결론 "R1–R6 재등록 불요"를 유지한다. (ii) 파일럿 한 값(20.57%)을 본 뒤 문턱을 옮기면 그 위치 자체가 자유 표면이 된다.
(iii) A5의 4분지 덕분에 OFF 실패는 ON arm에 대한 거짓 귀속을 만들지 않는다(rev1 死因 F1 형태가 재발하지 않는다).

**문턱 양방향 민감도(교훈 251, 파일럿 OFF 5 셀·seed 기준)**: OFF 문턱 t에서 OFF 절 실패 수 — t = 21.0 → 0 · **t = 20.0(등록) → 1**(`a_r2`) · t = 18.5 → 1 · t = 18.3 → 2 · t = 17.0 → 3 · t = 16.0 → 4.
느슨하게 하면 `a_r2`가 0.43pp 여유로 통과하고, 조이면 `a_r4`도 무너진다. 두 방향 모두 한 파일럿 값 근처에서 결과가 바뀐다.

**감사자 선택지**(결정은 claims-auditor. A13 작성자는 (A)를 권고):

| 안 | 내용 | 장점 | 위험 |
|---|---|---|---|
| **(A) 유지 + 예보 공시**(이 A13) | 문턱 20.0 유지, 위 예보·ON 절 분리 보고 등록 | 등록 무변경, 데이터 본 뒤 조정 없음 | 1차 라벨이 깨끗한 `STICKY_REALIZES_A`가 아닐 가능성이 크다 |
| (B) 상대 형태로 재등록 | 예: OFF 절을 "같은 셀·seed에서 ON − OFF ≥ 70pp(네 추정량 각각)"로 | 기판 수준 이동에 둔감 | 70이라는 수를 파일럿을 본 뒤 고른다. 판정량의 의미가 "OFF가 D44 밖"에서 "ON이 OFF와 충분히 다름"으로 바뀐다(새 estimand) |
| (C) OFF 절에서 E-qcond 제외 | 세 추정량만 OFF 절에 | `a_r2` 파일럿 통과 | rev3 §5-1·A8의 "네 추정량 동시" 근거를 OFF 쪽만 깬다. 파일럿을 보고 고른 추정량 선택(rev1 死因과 같은 형태: 추정량 메뉴와 규칙의 동시 변경) |
| (D) 절대 문턱 상향(예: 25) | — | — | 파일럿에 맞춘 이동. 권하지 않는다 |

(B)–(D)는 모두 실행 전 재등록이라 사후 조정은 아니다. 그러나 근거가 VESSL 한 셀·한 seed이므로 **과적합 여부를 감사가 확인해야 한다**.

---

## 5. (e) 예산·하드캡·중단 규칙

### 5-1. per-boot 부대비용 — VESSL 실측으로 교체 (지시 3)

λ0 VESSL Job `BUDGET.txt`의 연속 `admit … remaining_s` 차이 = 셀 벽시계(입장 → 다음 입장). 여기서 bench duration을 빼면 per-boot 부대비용이다
(입장 계산·boot·warm-up·사후 분석·teardown 5+10 s 포함). 이 회차 산출:

| 구분 | λ0 VESSL 실측 | 등록 상수 | KISTI(rev3 §8-1) |
|---|---|---|---|
| shape A, E2와 같은 N=400 셀 6개(`a_r2`·`a_r3`·`a_r4`·`a_r4_s2`·`a_r4_s3`·`a_r4_s4`) | 159.1–160.1 s | **A = 160.1 s**(최댓값) | 145.5 s |
| (제외) `a_r1`(N=1,260: 사후 분석이 큰 파일을 읽는다) · `a_r0`(N=660, Job 첫 boot) | 163.4 s · 251.4 s | `a_r0`만 cold 추가로 반영(아래) | — |
| shape B 7셀(`b_r0`·`b_r1`·`b_r2`·`b_r3`·`b_r3_s2`·`b_r3_s3`·`b_r3_s4`) | 118.6–120.9 s | **B = 121.0 s** | 106.5 s |
| 첫 boot 추가(JIT·캐시 cold: `a_r0` 251.4 s − `a_r4` 159.4 s) | 92.1 s | **+92.1 s**(1회) | 없음 |
| P13 프로브(첫 OFF·첫 ON boot) | 미측정 | **30 s × 2**(여유값) | 없음 |
| preflight(job_entry 시작 → 첫 boot) | λ0 732 s(그중 변이 하네스 약 10분 — E2에는 없다) | **600 s**(여유값, E2 미측정) | ≥ 400 s |

참고 분해(λ0 VESSL descriptive): boot+warm-up A 134–136 s, B 95–97 s · bench 뒤 정리 22–24 s.
λ0 F6(VESSL 소요 ≈ KISTI 예산식 ×1.36)은 쓰지 않는다. 셀 단위 실측이 더 직접적이다.

### 5-2. 예산 3층 (`e2_vessl_plan.py --budget`, 단가 $1.48/h)

bench duration은 λ0 VESSL OFF 실측(133.62 / 142.90 / 289.67 s). ON도 같다고 두는 것이 보통, rev3 §8-1 최악 코너는 ON bench ×2.5(A)·×1.5(B).

| 층 | 컨테이너 초(첫 boot 끝까지 포함, tail 제외) | GPU-h | 비용 | 20 boot 입장 |
|---|---|---|---|---|
| **보통** | 7,168.5 s | **1.991** | **$2.95** | 20/20 |
| **최악 코너** | 9,117.3 s | **2.533** | **$3.75** | 20/20 |
| **하드캡**(셀 구간) | 14,400 s | **4.000** | **$5.92** | — |
| 참고: 모든 bench가 `T_cap`에 닿는 극단 | 12,926.7 s(19번째까지) | 3.591 | $5.31 | 19/20 — P14가 `b_r3/4162/ON`을 거부 → 그 짝 `UNRESOLVED_BUDGET` |

- 표 밖: tail(P13 판정·green 요약·라벨, 600 s 예비 안), 스케줄링·이미지 pull(B5에서 1분 미만), 결과 반출(λ0 VESSL 규모 ≈400 MB, 미측정).
- 비교: KISTI 등록 1.803 / 2.338 / 3.0 GPU-h(무상 SLURM). λ0 VESSL 실지출 컨테이너 1.894 h($2.80 하한).
- **최악 코너에서 마지막 boot가 P14에 걸리기 전까지의 boot당 여유 = 219.4 s**(이식성 판정서의 KISTI 값 +97.8 s/boot에 대응). 즉 per-boot 부대비용이 평균 219 s 더 늘어야 R4″가 사라진다.

### 5-3. 하드캡 = 스크립트 내부 wall-clock 집행기 (지시 4 — ADDENDUM A4 대체)

ADDENDUM A4의 *"남은 예산 = `--time` 할당(10,800 s) − 잡 시작 이후 경과 `SECONDS`"* 를 다음으로 **대체**한다:

> **남은 예산 = (job_entry 시작 시각 + 14,400 s − 600 s) − 현재 시각.**
> job_entry 시작 시각 = `/data/runs/<job>/meta/timing.txt`의 `start_utc`(없으면 러너 시작 시각, 기록됨).
> **P14**: 각 boot 시작 전 남은 예산이 `T_cap(다음 셀) + 200 s` 미만이면(또는 하드캡이 이미 발화했으면) **그 seed 짝부터 끝까지 실행하지 않고 `UNRESOLVED_BUDGET`**. 부분 집합으로 재라벨하지 않는다(D23).

`200 s`는 rev3 P14 리터럴 그대로다(VESSL 정상 상태 부대비용 A 160 s·B 121 s를 덮는다. 첫 boot의 cold 252 s는 시작 직후라 무관).
세 겹 집행(λ0 VESSL §6-2와 같은 구조):
1. **캡**: 14,400 s = 4.0 h($5.92). 사용자 승인 범위 "≈4 h" 안. 상수는 `e2_vessl_plan.HARD_CAP_S`. 측정 모드에서 env로 못 바꾼다.
2. **입장 규칙**(boot 시작 전): 위 P14.
3. **실행 중 상한**: P13 프로브·warm-up·bench를 `timeout -k <s> <남은 시간 또는 T_cap>`으로 감싼다. 별도 watchdog이 마감에 `BUDGET_HARDCAP_FIRED`를 쓰고 bench·서버를 종료한다. `wait_health`도 그 플래그를 본다.
- bench 절단의 종류: `BT = min(T_cap, 남은 시간)`. `T_cap`이 먼저면 **CAPPED**(P7, 등록 결과 — 짝 `UNRESOLVED`), 남은 시간이 먼저거나 watchdog이면 **BUDGET**(측정 실패 — 짝 `UNRESOLVED_BUDGET`).
- **변이 검증**(테스트, DRYRUN 가짜 bench, 같은 타임아웃·watchdog·입장 기계): 정상 → 마감에서 잘리고 이후 boot 미시도.
  두 kill 층(타임아웃·watchdog)을 **모두** 제거한 변이 → 마감을 넘겨 돈다(검출). **한 층만** 제거 → 다른 층이 자른다(등가 변이, 교훈 254 — 의도된 중복, 숨기지 않는다). 입장 규칙 제거 변이 → 구조 테스트 실패.
- 캡 밖에 남는 시간: job_entry의 clone·install(λ0에서 14 s), tail 600 s, 반출. 컨테이너 총 시간 상한 ≈ 4.0 h + 반출. **과금 하드캡이 아니다**(LV-13 형식).

### 5-4. 재개 (짝 단위)

- 자동 재시도 없음. 재개는 **새 Job**에 `--env PDMUX_E2V_RESUME_FROM=<이전 job name>`.
- 조건: 이전 Job의 `UNITS.txt` 바이트 동일, 등록 digest 동일. 아니면 `ABORT_RESUME`.
- **짝 단위**: 한 seed 짝의 두 boot가 모두 `boot_<boot>.complete`일 때만 그 짝의 파일을 복사하고 다시 돌리지 않는다. 짝은 Job을 가로질러 쪼개지지 않는다.
- `.complete` = 등록 결과가 있다(bench 완료 또는 **P7 CAPPED**) ∧ 셀 JSON이 있다. **BUDGET 절단·boot 실패는 complete가 아니다** → 재개 Job이 그 짝을 다시 돈다.
  CAPPED 짝을 다시 돌리지 않는 것은 "성공할 때까지 재시도"를 막기 위해서다.
- 재개한 짝의 배너는 P3 키에 합쳐 Job 사이 일치도 검사한다. P13은 이 Job에서 처음 실행되는 OFF·ON boot에서 돈다(모든 짝이 재개면 이전 Job의 두 파일을 비교, 출처 기록).
- 재개 결과는 **Job 혼합**이다(E2V-7). `RESUMED_PAIRS.tsv`를 결과 문서에 반드시 적는다(운영모델 §3).

### 5-5. 중단 코드

| 코드 | exit | 조건 |
|---|---|---|
| `ABORT_ENV` / `ABORT_TEST_KNOB_IN_MEASUREMENT` | 2 | §2-4 |
| `ABORT_REG_DRIFT` / `ABORT_P4` | 3 | §1-1 digest 불일치 / digest 10개 미만 / manifest 미달 |
| `ABORT_SUBSTRATE` / `ABORT_MODEL_REVISION` / `ABORT_TRACE` | 3 | §2-1·§1-2 (DRYRUN에서는 기록만) |
| `ABORT_STAGE` / `ABORT_ARCHIVE_DRIFT` | 3 | §2-5 |
| `ABORT_ENGINE_TREE_DRIFT` | 3 | §1-3 (DRYRUN에서는 기록만) |
| `ABORT_SELFTEST` | 4 | §1-1 셀프테스트 |
| `ABORT_UNIT_COUNT` / `ABORT_RESUME` | 2 | 20 boot 아님 / §5-4 |
| `ABORT_NO_FREE_PORT` | 5 | 포트 없음 |
| `ABORT_STICKY_STATE_MISMATCH` / `ABORT_P2` | 6 | P1/P2 |
| `ABORT_CORRECTNESS` | 7 | P13 빈 출력 또는 기록 없음 |
| `ABORT_P3` | 8 | P3 |

---

## 6. (f) 자유 표면 자기 공시 (감사자가 볼 곳)

1. **엔진 트리 = HEAD**(§1-3): 두 arm에 같다. 1파일 차이를 R6에 공시. 다른 트랙이 `src/`를 더 바꾸면 이 Job은 돌지 않는다(의도된 동작, 그때는 새 追記).
2. **`T_cap` VESSL 값**(§2-2): 908623 값 대비 +1–9 s 완화. ON/OFF ≈ 2.94–3.0 구간에서만 결과가 갈린다.
3. **per-boot 상수 A 160.1 · B 121.0 · cold 92.1**(§5-1): 입장 규칙·예산표만 바꾼다. 라벨은 안 바꾼다. 최댓값을 골랐다(보수 쪽).
4. **P13 30 s·preflight 600 s**: **미측정 여유값**이다. 예산표만 바꾼다.
5. **하드캡 14,400 s**: 사용자 승인 "≈4 h"의 상한을 그대로 썼다. 시계 시작은 job_entry 시작(clone·install 포함). 최악 코너 여유 219 s/boot.
6. **P14 200 s 유지**: KISTI 부대비용(145.5 s)에 맞춰진 리터럴을 VESSL(160 s)에 그대로 둔다. 부족분은 실행 중 상한(3겹의 3번)이 막는다.
7. **R1 OFF 밴드**(§4-1): 문턱 유지 + 예보 공시 — 감사자가 (A)로 확정(대안 (B)–(D) 기각). ON 절 성립은 Q1의 답이 아니다(E2VC-1).
8. **CPU 핀 없음**(§2-6): 검출력 손실 방향. 부호 반전 경로 여부는 미검증(E2VC-2).
9. **green read-out 양 arm 추가**(§2-7): 1회성. 관측자 효과 0을 주장하지 않는다. λ0 VESSL OFF 파일럿과 env를 맞추는 효과도 있다.
10. **`timeout -k`·`setsid`·warm-up 타임아웃 추가**: 발화하지 않으면 동작 무변.
11. **bench 절단 종류 분기**(CAPPED vs BUDGET): rev3에 없던 분기다. BUDGET은 등록된 P14·D23의 "예산 절단"에 해당하고, CAPPED는 P7 그대로다.
12. **재개(짝 단위)**: rev3에 없던 경로다. CAPPED 짝은 재실행하지 않는다. 재개가 있으면 Job 혼합 병기.
13. **셀프테스트 입력 운반**: 바이트는 digest로 강제. 운반 경로는 판정에 무관.
14. **B0 게이트 문턱**(80GB·cc 8.0·108 SM·MIG off·driver ≥ 550): 기판 종류만 거른다. 동일성은 판정하지 않는다(E2C-37).
15. **`--expect`·순서 생성을 helper로 옮김**: 테스트가 sbatch의 리터럴 생성과 같은 결과임을 확인한다.

## 7. 로컬 검증 · 이 회차가 하지 못하는 것

- 로컬 검증(`bash -n`, 셀프테스트, `tests/test_e2_vessl.py`, 전체 CPU 회귀, `check_citation_stops.py`, 공개 이미지 컨테이너 dry-run) 결과는
  **이 문서가 아니라 engine-porter 보고**에 적는다. 이 문서의 digest가 러너에 고정되므로, 실행 결과를 이 문서에 적으면 digest가 바뀐다(λ0 VESSL §9와 같은 이유).
- 이 회차는 rev3 §10의 "하지 못하는 것" 7개를 **그대로** 승계한다: 게이트 #6 미해소 · P2 블로커 ② 미해소 · Claim D/E 불변 · 새 성능 판정 0건 ·
  shape A↔B 비교 금지 · λ0 라벨 재계산 없음 · true-dual(B4)에 대해 말하지 않음 · E3 미포함.
- **E2C-5 문자 승계**: *"이 회차는 λ0R-8(iii)(λ\*(A)로 B4/true-dual을 B1에 정규화하는 설계의 교락)을 **해소하지 않는다** — R1/R2를 어떻게 내든."*
  두 arm은 prefill 축이 같고(prefill span에서 둘 다 idx 2), true-dual이 바꾸는 축(prefill·decode 동시성)은 이 회차에서 측정되지 않는다.
  결과 문서·정본은 "E2가 λ0R-8(iii)을 해소했다/판정했다/약화시켰다"를 쓰지 않는다(R2 `banned_reading`과 같은 방향).
- **추가(이식)**: VESSL 결과는 KISTI 908623의 E2 기대값(파일럿 포함)을 "재현했다/반증했다"로 쓰지 않는다(다른 기판, 게이트 1의 연장). LVR-1…4·LV-1…14 승계.

## 8. 대체 조항 색인

| rev3·ADDENDUM 조항 | A13 대체 |
|---|---|
| rev3 §7 P4 "manifest ≥ 24 + `nemotron_h.py`" | 유지 + §1-3 엔진 트리 대조(1파일 차이 강제) 추가 |
| rev3 §7 P7 "`T_cap = 3 × (908623 OFF 실측)`" | §2-2 |
| rev3 §7 P14 + ADDENDUM A4 "남은 예산 = `--time` 할당 − `SECONDS`" | §5-3 |
| rev3 §8 예산(1.803 / 2.338 / 3.0 GPU-h, `--time 03:00:00`) | §5-1·§5-2(1.991 / 2.533 / 4.0 GPU-h) |
| rev3 §11·ADDENDUM A10의 제출 선행조건 "`sbatch`" | §10의 `launch.sh` 절차 |
| (없음) | 추가: B0 게이트, green read-out, cpu.stat·TIMING 기록, 허용 목록, 짝 단위 재개, 셀프테스트 입력 staging |

## 9. 이식성 판정서 체크리스트 대응

| 판정서 항목 | 처리 |
|---|---|
| Q1 #1–2(격자·idx 매핑) | 기판 A100 108 SM을 B0로 강제 → 기판 무관 조건 성립 |
| Q1 #5–6·E2C-36(offered·λ\*) | §3-2 E2C-36(VESSL 위치 0.985–1.001×) |
| Q1 #9(P7 `T_cap`) | §2-2 |
| Q1 #10·Q2 반전 "per-boot +98 s"(예산 상수) | §5-1·§5-2(여유 219 s/boot) |
| Q1 #11·Q2-(c)·E2C-39(`--time` 집행기) | §5-3 세 겹 + 변이 검증 + 고아 서버 정리(§2-3) |
| Q1 #13·Q2 반전(R1 OFF 밴드) | §4-1 사전 예보(파일럿 −0.57pp) + 감사자 선택지 |
| Q1 #16·#22(80 GB 전제·`max_running` 48) | B0 메모리 ≥ 79,000 MiB, P3 배너 기록 |
| Q1 #17(E2C-1 "108→44") | 기판이 108 SM으로 강제되므로 문언 그대로 성립 |
| Q1 #19–20(E2C-21·E2C-6) | §3-3 재산출 |
| Q1 #23(클러스터 리터럴) | §2-3 |
| Q1 #24·Q2-(f)(CPU 핀) | §2-6(핀 없음 + 기록 + E2V-4) |
| Q2-(a)(manual edits·manifest는 기록일 뿐) | §1-3 manifest를 908623과 라인 대 라인 대조(해소 조건 2). manual edits 5파일은 install_runtime이 적용하고 `meta/runtime_source_manifest.sha256`(최종 트리)에 기록. 그 5파일 중 manifest 25항목 밖 파일은 λ0 VC-7과 같이 caveat |
| Q2-(b)(드라이버·CUDA·패키지) | `SUBSTRATE_E2V.txt`·job_entry `substrate.json` 기록. 같은 Job 안 대조는 면역, R6만 E2V-1·E2V-8 |
| Q2-(d)(하네스 sha 변경) | `e2_sticky.sbatch`는 수정하지 않았다(digest 고정, 실행 안 함). 이식본은 새 파일이며 **하네스층 국소 재확인** 대상 |
| Q2-(e)(공유 호스트 드리프트) | E2V-5 |
| Q2 R6 행(caveat 확대) | E2V-1 |
| Q2 green read-out | §2-7, E2C-38 부분 해소 |
| 해소 조건 1(B0) | §2-1 B0 게이트(매 Job) |
| 해소 조건 2(manifest 대조) | §1-3 |
| 해소 조건 3(B4 스모크 1 boot) | **별도 GPU 스모크 없이** λ0 VESSL Job(15 boot, 같은 이미지·같은 manifest 25/25)이 대신한다: 배너 `sm_counts` 격자·`max_mamba_cache_size 48 / max_total_num_tokens 2,722,025`·`a_r4` duration 133.6 s(908623 131.0 s의 +2.0%, ±20% 안)·부대비용 160 s(145.5 + 98 이내)·green read-out 확인. 단 HEAD 트리(1파일 차이)로 부팅한 적은 없다 → 첫 boot의 P1–P3이 그 확인을 맡는다 |
| 해소 조건 4(A13) | 이 문서 |
| 해소 조건 5(새 OVERRIDE) | `OVERRIDE_E2_VESSL_2026-10-04.md` |
| 자기검사(`timeout`만으로는 고아 서버) | `setsid` + watchdog `pkill -9 -f sglang.launch_server` + abort 전 teardown |

## 10. 제출 게이트 (순서대로)

1. 이 A13이 claims-auditor **규칙층(이식) `GO`/`GO-with-caveats`**를 받는다(§4-1 선택지 결정 포함).
2. **하네스층 국소 재확인**(이식성 판정서 Q2-(d), 교훈 34의 2단).
3. 감사가 A13을 바꾸면 `e2_vessl_plan.A13_DIGEST`를 같은 커밋에서 갱신한다.
4. 메인 세션이 커밋한다. CPU 회귀 통과, `check_citation_stops.py` 0위반, `presubmit.py` 차단이 정확히 M4R·TC1 2건
   (2026-10-04 로컬 실행은 차단 5건 — 규칙 파일 절대경로로 인한 reachability 도구 미실행 4건 + E2 무관 `doc_facts` 1건[위반 2]. OVERRIDE §1).
5. `stage_e2_inputs.sh --upload`(스토리지만, GPU 0).
6. **새 OVERRIDE 승인 확인**(`OVERRIDE_E2_VESSL_2026-10-04.md` §7) — 사용자 승인은 기록됐으나 제출은 1–5 뒤다.
7. 사용자가 `launch.sh --campaign e2_sticky_vessl --target workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh` **dry-run 출력**(명령·단가·job name)을 보고 승인 → `--submit`.
8. 회수 = `fetch.sh e2_sticky_vessl <job>`의 DONE + sha256 통과.
