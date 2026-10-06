# N3(게이트 #6) 제출 기록 (메인 세션)

| shape | job name | slug | 제출(UTC) | 커밋 | 이미지 | spec | 승인 |
|---|---|---|---|---|---|---|---|
| B | `pdmux-n3-capacity-a4b0c2e-20261006064114` | `job-sxdnvi86hxsy` | 2026-10-06T06:41Z | `a4b0c2e` | `sglang-runtime@1e0d335b…` | `resourcespec-a100x1` | 사용자 2026-10-05 "N3 dry-run 진행" + 2026-10-06 "실험 job들에 대해서 다시 진행"(E2 r2 종료·회수 뒤 제출, 동시 제출 없음) |

- 제출 시 상태: `git status --porcelain --untracked-files=no` 비어 있음, presubmit 차단 = M4R·TC1 2건(등록된 범위), 사전등록 rev4(`23c4175`)·하네스층 H-1 수리(`1a90b08`) 커밋됨, 봉인 sha = `VERDICT_n3_harness_2026-10-05.md` 追記 표.
- env: `PDMUX_N3_SHAPE=B`만 추가(허용 목록). 예산 문언(§8-3): 보통 3.37 GPU-h / 최악 3.83 / 하드캡 4.0(셀 구간).
- shape A는 B 결과 감사 뒤 별도 승인.

## shape B 1회차 실패 진단 (2026-10-06, 메인 세션, GPU 소비 ≈1 min — 셀 0 측정)

- 상태 `failed`, 컨테이너 06:41:37Z→06:42:34Z(57 s), `exit_code=4`, 마지막 줄 `ABORT_SELFTEST n3_plan`. 회수 메타 `vessl/pdmux-n3-capacity-a4b0c2e-20261006064114/`(26 files, sha 통과), OUT 디렉터리 `n3b_…/`(preflight 산출물만).
- **통과한 preflight**: CONFIG OK · SUBSTRATE OK(A100 80GB cc8.0 108 SM) · 모델 revision · ShareGPT sha · clock · `install_runtime` + manual_edits patch · `--verify-digests "$ENG"` → `DIGESTS OK (3 files + 9 mirror entries)` · `ENGINE_TREE == registered HEAD manifest (0 entries differ)` · thread-local role patch.
- **실패 지점**: `n3_plan.py --selftest` → `selftest()` 기본 인자 `eng=ENG` → `verify_digests(eng)` →
  `benchmarks/pdmux_eval/analyze.py: MISSING`.
- **원인(로컬 재현으로 확정)**: `n3_plan.py:36–37`이 `ENG = Path(__file__).resolve().parents[1]`로 엔진 루트를 **자기 파일의 실경로**에서 유도한다. `job_entry.sh:100–103`은 Job 안에서 `workspace/engine-port/results/`를 `/data/runs/<job>/results`로 **옮기고 심볼릭 링크**로 바꾸므로 `resolve()`가 `/data/runs/<job>/results/n3_capacity/n3_plan.py`로 풀리고 ENG = `/data/runs/<job>`가 된다. 그 루트 밑에는 `results/…`(config·`lambda0_plan.py`)만 있고 `benchmarks/`·`src/`는 없다 → 등록 3파일 중 analyze.py만 MISSING(그래서 정확히 1건). 같은 helper의 `--verify-digests "$ENG"`는 러너가 루트를 **인자로** 넘겨서 통과했다(두 경로의 차이가 증거). 로컬에서는 `results/`가 실제 디렉터리라 selftest가 통과했다. ~~Job 레이아웃(results 심볼릭 링크)은 사전등록·하네스 감사 어디에도 전제로 적혀 있지 않았다~~ **(정정, rev5 델타 감사 R-2: 거짓)** — 하네스 판정서 `VERDICT_n3_harness_2026-10-05.md` §7은 이 레이아웃을 적었으나 OUT 경로에만 적용했고 helper의 루트 유도에는 적용하지 않았다. 같은 함정은 저장소 안에 이미 진단·수리된 선례가 있다(`results/e1_cap_vessl/e1_plan.py:45-47`, 2026-10-04, GPU 없는 컨테이너 dry run에서 발견) — N3 하네스가 그 선례를 승계하지 않았다(교훈 18 계열).
- **동일 결함 파급**(로컬 심볼릭 링크 모사 레이아웃에서 재현): `n3_analyze.py:37–40`도 같은 유도(`ENG/"benchmarks"/"pdmux_eval"`를 sys.path에 추가) → `ModuleNotFoundError: No module named 'analyze'` — selftest를 넘겼더라도 **채점 단계에서 실패**했을 결함. `n3_realized_mix.py:66–69`는 `results/` 밑 경로만 쓰므로 Job 레이아웃에서도 성립(단 λ0 archive는 로컬 전용).
- **E2/λ0가 같은 함정을 피한 이유**: `e2_vessl_plan_r2.py`는 HERE를 자기 디렉터리 sys.path에만 쓰고 엔진 루트는 러너가 `--verify-digests-r2 HERE_DIR ENGINE_ROOT` 인자로 넘김; `lambda0_vessl_plan.py`는 `results/` 밑 상대경로만 유도.
- **수리 방향(봉인 파일 변경 ⇒ rev5 + 규칙층 델타 감사 + 하네스 재감사 필요, 승인 전 미수정)**: n3_plan.py·n3_analyze.py의 엔진 루트를 `PDMUX_ENGINE_ROOT`(러너가 `$ENG`로 export) 또는 `Path(__file__).parent`(비해석, `resolve()` 제거)로 유도하고, 테스트에 "results가 심볼릭 링크인 레이아웃에서 selftest/analyze import가 성립" 케이스(수리 되돌림 변이에서 실패해야 함, 교훈 53)를 추가. 러너 `n3_capacity.sh`는 `ENG`를 `$ROOT`에서 유도하므로 무변경 가능.
- 등록 라벨·셀 측정 0건 ⇒ D23(부분 집합 재라벨 금지) 해당 없음. 재제출은 새 job name(커밋 sha 포함)로.
