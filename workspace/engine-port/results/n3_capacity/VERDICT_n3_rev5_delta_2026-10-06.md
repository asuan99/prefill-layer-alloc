# 판정서: N3 사전등록 rev4 → rev5 델타 감사 (규칙층 + 하네스층 합본, claims-auditor, 2026-10-06)

> 읽기 전용, GPU 0, 저장소 무수정. 감사자 최종 보고를 메인 세션이 옮겼다(§8 R-3 표만 R-1·권고 반영 후 값으로 메인 세션이 확정, 아래 "메인 세션 追記" 참조). 성능 판정 0건, Claim 등급 변경 0건.
> 대상: `git diff a4b0c2e -- results/n3_capacity tests/test_n3_capacity.py`(4파일, +78/−3). 범위 밖 미커밋 diff 3건(`PROJECT_STATUS.md`·`CONSENSUS.md`·`DESIGN_A1_REV2_STICKY`)은 doc_facts 카운터 955→957이며 N3 실행 경로에 닿지 않는다.
> 감사자 재실행(스크래치): `git archive a4b0c2e` + rev5 patch 트리에 `job_entry.sh:100-103`을 그대로 적용(전체 `results/`를 `data/runs/J/results`로 `cp -a`, 원위치를 심볼릭 링크로)한 모사 트리에서 러너의 helper 호출 전부(preflight → 합성 캠페인 채점 → mix CLI), `.resolve()` 복원 변이, `test_n3_capacity` 63건.

## 0. 판정: **`GO-with-caveats`**

- 死因 0건, 필수 수리 3건(전부 문서·봉인 절차, 코드 변경 없음), 새 caveat N3C-23…N3C-27.
- 코드 수리는 국소적이고 정확하다. 판정 경로·상수·arm·순서·예산·입장 규칙의 의미는 바뀌지 않았다(§2, AST 동등성 + helper 출력 일치).
- 러너 전체 흐름에서 클론 밖 경로에 의존하는 곳은 이번 두 지점뿐이다(§4).
- 진단 기록의 "어디에도 전제로 적혀 있지 않았다"는 사실이 아니다. 같은 함정의 저장소 내 선례가 있다: `results/e1_cap_vessl/e1_plan.py:45-47`(2026-10-04, "caught by the GPU-less container dry run"). N3 하네스는 이 선례를 승계하지 않았다(교훈 18 계열).

## 1. 원인 주장 (Q1) — **CONFIRMED**

- 로그 순서: `DIGESTS OK` → `CONFIG OK` → `SUBSTRATE OK` → `ENGINE_TREE == registered HEAD manifest (0 entries differ)` → `n3_plan.py:722 selftest` `assert verify_digests(eng) == []` → `['benchmarks/pdmux_eval/analyze.py: MISSING != registered 1be49511…']` → `ABORT_SELFTEST n3_plan`(exit 4). traceback 경로 `/opt/pdmux/prefill-layer-alloc/.../n3_plan.py` = 비해석 호출 경로.
- 기전: `job_entry.sh:100-103`이 `results/`를 `${RUN}/results`로 복사하고 심볼릭 링크를 건다. `.resolve()`면 HERE = `/data/runs/<job>/results/n3_capacity`, ENG = `/data/runs/<job>`. 그 밑에는 `results/`만 있어 등록 3파일 중 `analyze.py` 한 건만 MISSING, assert가 `mirror_check`보다 앞이라 거기서 멈췄다.
- `--verify-digests "$ENG"`만 통과한 이유: 러너(`n3_capacity.sh:120`)가 ENG를 인자로 넘긴다. `--selftest`는 모듈 전역 `ENG`(`.resolve()` 유도)를 쓴다.
- 모사 트리 독립 재현(rev4 줄 복원):

  | 호출 | 결과 |
  |---|---|
  | `--verify-digests` | rc 0 |
  | `--selftest` | rc 1, Job과 문자열 동일 |
  | mix `--selftest` | rc 0 |
  | `n3_analyze --selftest` | `ModuleNotFoundError: No module named 'analyze'` |
  | preflight unittest | `TestRegistration.test_plan_selftest` FAIL |

  1회차가 selftest를 넘겼더라도 abort 지점이 두 번 더 남아 있었다. rev5는 셋을 모두 해소한다.

## 2. 규칙층 변경 0 주장 (Q2) — **CONFIRMED**

- `n3_plan.py`·`n3_analyze.py` AST는 `import os`와 `HERE =` 대입을 빼면 rev4와 동일. `os` 사용은 HERE 줄 1곳.
- 바이트 불변: `n3_capacity.sh` `3cc780dc…`, `n3_realized_mix.py` `3b6f8934…`, config `01675b2d…`, `scripts/vessl/*` diff 없음.
- 사전등록 변경은 제목 줄·머리말 rev5 줄·§17-4뿐. §2·§5·§8·§14 본문 불변.
- 로컬 레이아웃(심볼릭 링크 없음)에서는 abspath와 resolve가 같은 ENG.
- 모사 트리 합성 캠페인(cliff 0.38) 채점: `CAPACITY_BRACKETED_{3,5,10}`, capacity 0.38, bracket [0.38,0.45], `OBSERVER_OK` — analyze selftest 고정 assert와 동일.

## 3. abspath 유도가 Job에서 맞는 루트를 주는가 (Q3) — **CONFIRMED (러너 호출 형태에 한정)**

- 러너는 `cd "$ROOT"` 후 helper를 `$ROOT/workspace/engine-port/results/n3_capacity/...` 절대경로로 부른다(`n3_capacity.sh:21-26,37`). 모사 트리: `P.ENG`·`N.ENG` = 클론, `A.__file__` = 클론의 `benchmarks/pdmux_eval/analyze.py`.
- n3_analyze의 `sys.path.insert(0, HERE)`가 앞서므로 `n3_plan`·`n3_realized_mix`는 비해석 HERE에서 로드(같은 inode).
- `n3_realized_mix.py`(여전히 `.resolve()`): Job에서 `M.ENG` = `/data/runs/<job>`. ENG로 만드는 경로는 `LAM0V_ARCHIVE`·`E2_TOOL` 둘뿐이고 둘 다 `results/` 밑(복사본 내용 동일). 셀 단위 `compute()`·n3_analyze가 쓰는 M 함수는 ENG 미사용. λ0 아카이브는 git-ignored라 Job mix selftest는 두 유도 모두 "합성 OK + `SELFTEST_ARCHIVE_MISSING` → rc 0"(기존 설계). 클론 밖 경로가 필요한 mix 기능 없음. P.ENG ≠ M.ENG 비대칭은 남는다 → N3C-24.
- 한계: abspath는 호출 경로에 의존. 모사 트리에서 `cd results/n3_capacity`(심볼릭 링크 경유) 후 상대 호출하면 같은 `No module named 'analyze'` 재현(실측). 러너 바이트로는 도달 불가, fail-closed → N3C-23.

## 4. 같은 함정이 다음 단계에 남아 있는가 (Q4) — **반증 실패: 경로 계열 잔존 결함 0건**

| 단계 | 호출 | ENG/경로 의존 | 모사 트리 결과 |
|---|---|---|---|
| 등록 | `--const`·`--verify-digests $ENG`·sha256sum 8+개·`--check-config` | 인자/러너 ENG | OK |
| preflight | sync·`--engine-check`·parallel_state grep(`SGLANG_ENGINE_DEV`) | 의도된 설치 트리 | Job에서 이미 통과 |
| selftest | plan / mix / analyze `--selftest` | P.ENG(수리) / M.ENG(results만) / N.ENG(수리) | 3/3 OK |
| unittest | `test_n3_capacity.{Registration,Units,FailClosedWiring,PortAndTeardown}` | 테스트 ENG = `tests/` resolve(클론 실디렉터리) | 23/23 OK |
| unittest | `test_phase_events`·`test_green_readout` | abspath(tests) + 설치 sglang | 로컬 실행 불가(sglang 없음), 경로 의존 없음(독해) |
| 셀 루프 | `--units/--budget/--need/--tk-valid/--floor/--check-green/--p3-verdict` | ENG 미사용 | OK (units 48, need 192) |
| 기술 | `n3_realized_mix <tel> --out mix_*.json` | ENG 미사용 | rc 0 |
| 채점 | `n3_analyze --out-dir --shape B --clk` | `import analyze`(N.ENG) | 합성 BRACKETED, rc 0 |
| 꼬리 | `cleanup` `rm -rf $HERE/__pycache__` 등 | 무해 | — |

- `lambda0_plan.py`·정본 `analyze.py`: 경로 유도 없음. `e2_realized_mix.py`: abspath, `results/` 상대, 아카이브 selftest 전용. N3가 부르지 않는 VESSL helper(`e2_vessl_plan*.py`·`lambda0_vessl_plan.py`·`lambda0_prereg/*`)의 `.resolve()`는 `results/` 상대경로만 써서 안전 — E2/λ0 Job이 이 함정을 피한 이유.
- 경로 계열이 아닌 미발화 위험(기록용): `test_phase_events`·`test_green_readout`은 VESSL 이미지 실행 기록이 저장소에 없다(2026-10-04 기록은 작성자 컨테이너, `VERDICT_e1_cap_vessl_rules_rev1` :14). 같은 import 체인의 `test_sticky_partition`은 E2 Job에서 통과. `MutantTest` 서브프로세스 시간 미실측(preflight 600 s는 N3 미측정). 실패 시 `ABORT_PREFLIGHT`(exit 4), 라벨 오염 없음 → N3C-26.

## 5. 테스트 항등식 여부 · 하네스 판정서 영향 (Q5)

- `TestJobSymlinkLayout`은 항등식이 아니다. 대조 변이(`.resolve()` 복원, 치환 1회 assert)가 Job과 같은 문자열(`benchmarks/pdmux_eval/analyze.py: MISSING`, `No module named 'analyze'`)로 실패한다. 양성 시험은 실제 레이아웃에서 helper를 실행하므로 `os.path.realpath` 같은 다른 해석형 오수리도 잡는다.
- 모사 트리 63건 중 TestJobSymlinkLayout 2/2 통과. 오류 2건(`TestBootBranchSnippets.test_t_h1_sticky`·`test_t_h3_floor`)은 git-ignored 프로브 파일 부재(클린 클론) 때문으로 preflight 목록 밖·rev5 무관. skip이 아니라 error인 것은 기존 결함.
- 충실도 한계: 테스트는 `job_entry.sh`를 실행하지 않고 두 하위 트리만 복사해 모사한다. preflight unittest 묶음과 `n3_analyze --selftest`(테스트는 `--help`만)는 그 레이아웃에서 돌지 않는다 — 이 감사의 전체 복사 모사가 공백을 메웠다 → N3C-25.
- `VERDICT_n3_harness_2026-10-05.md`: §7은 레이아웃을 보았으나 helper 루트 유도에 적용하지 않았다. "§2 전부 ✓"·"§6 테스트 실제 경로 행동"은 로컬 레이아웃 검증. N3H-1·3·4·5·6 문언은 무영향. **N3H-2 영향**: 봉인 sha 표의 `n3_plan.py`·`n3_analyze.py`·테스트·사전등록 행 무효 → R-3 재봉인.

## 6. 1회차 Job 처리 (Q6)

- 측정 0, 라벨 0. OUT에는 preflight 산출물만, `T_K.json`·`unit_*.complete` 없음.
- "D23 비해당"은 N3 용어가 아니다(E2 OVERRIDE R2 `:136,159`의 부분 집합 재라벨 금지 / λ0 rev2 판정서의 미시도 shape EXPECT — 정의 둘, N3 사전등록에는 없음). 내용 자체는 참 → R-1.
- 재개 금지, 시도해도 fail-closed: `PDMUX_N3_RESUME_FROM=<1회차>`는 `--tk-valid` 실패 + digest 불일치로 `ABORT_RESUME`(`n3_capacity.sh:213-217`). 재제출은 새 Job·새 job name만.
- 승인 재사용: N3 사전등록에는 OVERRIDE 구조가 없다. §14 게이트 5는 dry-run 출력(명령·단가·job name·예산 문언)을 보고 승인. 커밋이 바뀌면 job name·번들이 바뀌므로 등록 문언상 새 dry-run에 대한 사용자 승인이 필요하다. 감사자는 이전 발화를 승인으로 간주하지 않는다(판단은 사용자 몫). 게이트 1·2(이 판정서)·3(커밋)·4(CPU 회귀·citation) 재충족 필요.
- 1회차 컨테이너 0.016 h, 장부 기록만.
- 기판 장부 부작용: 1회차가 `/data/substrate_history.tsv`에 campaign `n3_capacity` 행(driver 580.105.08, GPU-1d3fda4c…)을 남겼다 → N3C-27.

## 7. 반전 시험 표

원자료: 모사 트리, n3_analyze `_synth` 합성 캠페인(cliff 0.38, shape B), 1회차 `target.log`. 새 상수·추정량 없음.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 엔진 루트 유도 | resolve / realpath / abspath | 라벨 변화 없음(resolve 계열은 Job에서 preflight abort, 측정 0). 로컬 레이아웃에서는 셋 동일 | AST 동등, 합성 BRACKETED_{3,5,10} cap 0.38 동일, 변이 plan rc1·ana rc1·unittest FAIL 1 |
| helper 호출 경로 | 절대(클론) / cwd=심볼릭 링크 안 + 상대 | 없음(후자 abort, 러너로 도달 불가) | N3C-23 |
| mix `.resolve()` 유지 | 클론 / `/data/runs/<job>` | 없음 | mix selftest rc 0 양쪽, 셀 CLI rc 0 |
| n3_analyze 모듈 탐색 순서 | HERE 우선 / 실경로 우선 | 없음(같은 바이트) | P.ENG = 클론 |
| 테스트 레이아웃 충실도 | 2개 하위 트리 / 전체 `cp -a` | 없음 | 전체 복사 모사에서 selftest 3 + unittest 23 OK |
| 1회차 재개 | 미사용 / `RESUME_FROM`=1회차 | 없음(`ABORT_RESUME`) | `T_K.json` 없음, digest 2개 변경 |
| substrate_history 1회차 행 | 포함 / 제외 | 라벨 없음(경고 정보용) | N3C-27 |

반전 0건 → 死因 아님.

## 8. 필수 수리 (제출 전, 문서·봉인 절차만)

- **R-1** 사전등록 머리말: (a) rev5 줄 "D23 비해당" → N3 자체 문언("셀 측정 0·`T_K.json` 없음이라 부분 집합 재라벨·Job 혼합 문제가 없고, `RESUME_FROM`으로 쓸 수 없다(`ABORT_RESUME`)"). (b) 3행 지위 줄을 rev5 상태로 갱신.
- **R-2** `SUBMISSIONS_2026-10-06.md` 원인 문단의 "어디에도 전제로 적혀 있지 않았다" 정정 + 선례 `e1_plan.py:45-47` 병기.
- **R-3** 재봉인 + 이 판정서를 제출 전에 커밋(REGISTRATION이 `VERDICT_n3_*.md`를 기록하고 재개 digest 대조에도 들어간다).
- 권고(필수 아님): `n3_plan.py:5` docstring "(rev4)" → "(rev5)".

## 9. 새 caveat (결과 문서·정본이 문자 그대로 승계, 인용 금지 문장)

- **N3C-23.** "rev5의 엔진 루트 유도(`os.path.abspath(__file__)`)는 호출 경로에 의존한다. 러너는 helper를 클론 경로의 절대경로로 부르므로 Job에서 성립한다. 그러나 Job 안에서 `results/`(심볼릭 링크) 밑으로 cd한 뒤 상대경로로 부르면 같은 결함이 재현된다(abort, 측정 0). 'N3 helper는 어떤 레이아웃에서도 루트를 바르게 찾는다'고 쓰지 않는다."
- **N3C-24.** "`n3_realized_mix.py`는 여전히 `.resolve()`로 루트를 유도한다. 그래서 Job 안에서 그 루트(`/data/runs/<job>`)는 `n3_plan`·`n3_analyze`의 루트(클론)와 다르다. 현재 mix가 쓰는 경로는 `results/` 밑뿐이어서 결과는 같다. Job 안의 mix selftest는 합성 부분만이며, λ0 아카이브 양성 대조는 로컬 전용이다. 'Job preflight가 mix 추정량을 E2 도구와 대조했다'고 쓰지 않는다."
- **N3C-25.** "`TestJobSymlinkLayout`은 `job_entry.sh`를 실행하지 않고 그 레이아웃을 손으로 모사한다(두 하위 트리만 복사). Job 레이아웃에서 preflight selftest 3종·unittest 묶음·합성 채점이 성립함은 rev5 델타 감사의 스크래치 모사로 확인됐다. 하네스층 감사(2026-10-05)의 검증은 로컬 레이아웃 기준이다. `job_entry.sh`의 레이아웃이 바뀌면 이 테스트는 그것을 보지 못한다."
- **N3C-26.** "Job preflight의 `test_phase_events`·`test_green_readout`은 이 저장소 기록상 VESSL 이미지에서 실행된 적이 없다(로컬은 sglang 부재로 실행 불가, `test_phase_events`의 2026-10-04 기록은 작성자 컨테이너). 실행 시간도 실측되지 않았다(등록 preflight 600 s는 N3 미측정). 2회차 Job이 이 단계에서 `ABORT_PREFLIGHT`로 끝나면 그것은 측정 실패이지 하네스 규칙의 결함 증거가 아니다."
- **N3C-27.** "shape B 1회차 Job `pdmux-n3-capacity-a4b0c2e-20261006064114`는 셀 측정 0이다. 그 Job은 어떤 수치·라벨의 근거도 아니고, `RESUME_FROM`으로 쓸 수 없다. 그러나 그 Job이 `/data/substrate_history.tsv`에 campaign `n3_capacity` 행(driver 580.105.08)을 남겼으므로, 이후 N3 Job의 `SUBSTRATE_WARNING`은 측정 0인 Job을 기준으로 발화할 수 있다. 이 경고를 '측정 셀 간 기판 혼합'으로 읽지 않는다."

## 10. 게이트 #113 · #110

- R-1·R-2는 문서 문언만. 사전등록 sha는 Job에서 검증되지 않고 기록만 된다(`registered_files`에 없음), 테스트에도 사전등록 고정 없음. 새 상수·노브·라벨·exit 코드 0개.
- 처방하지 않은 것: mix abspath 수리(봉인 바이트 변경, 이득 없음 → N3C-24), `job_entry.sh` 실구동 테스트(실현성 미검증).
- 원인은 로그·`job_entry` 소스·모사 재현으로 독립 확인. SUBMISSIONS 서술 하나(R-2)가 그 과정에서 거짓으로 판명.

## 11. 남은 제출 선행조건

1. R-1·R-2 반영 → R-3 재봉인 → 이 판정서와 함께 커밋.
2. CPU 회귀(`test_n3_capacity` 63건) + `check_citation_stops.py` 0위반.
3. `launch.sh` 새 dry-run → 사용자 승인(새 job name) → `--submit`. `RESUME_FROM` 미설정.
4. shape A는 B 결과 감사 뒤 별도 승인.

---

## 메인 세션 追記 (2026-10-06) — R-1·R-2·권고 반영 후 재봉인 표 (R-3)

감사자 표의 `n3_plan.py`(`6ce4884d…`)·사전등록(`9e2472a4…`) 값은 R-1·docstring 권고 반영 **전** 값이다. 반영 후 작업 트리 값(`n3_plan.py`는 docstring 한 줄 "(rev4)"→"(rev5)"만 추가 변경, selftest·`TestRegistration/TestUnits/TestFailClosedWiring/TestJobSymlinkLayout` 재통과):

| 파일 | sha256 |
|---|---|
| `results/n3_capacity/n3_plan.py` | `452482a0bb34ca6d145bee8be9a1dd93afbbb86b109114dee07c41c77536ce0b` |
| `results/n3_capacity/n3_analyze.py` | `eb89c0a2c48faba77615c5b3d7de3cccd8b43be58cf6e1faf17c4a1be9d5b974` |
| `tests/test_n3_capacity.py` | `ce27a209ddc3ae88feb3a603ec4756c5df93697ca0d84d1dfa0891782d19bad5` |
| `results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md` | `6505036be424f436af32c90ecd3356f74baf0dcd0f192faa38bac1a3df93849b` |
| 불변 `n3_capacity.sh` | `3cc780dca651419965b3917db392b4af37f96feb5d035bfef56622b393cb0c3f` |
| 불변 `n3_realized_mix.py` | `3b6f8934c69ad09b23fd4490a882dafbdecef05e7980f51b2679fe062d2e65c3` |
| 불변 `configs/pdmux_n3_A_agn.yml` | `01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1` |
| 불변 정본 `benchmarks/pdmux_eval/analyze.py` | `1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2` |

사용자 승인(§6 게이트 5): 2026-10-06 사용자 지시 "수리 진행 후 실행" — 수리 후 재제출을 명시 지시. 제출 기록은 `SUBMISSIONS_2026-10-06.md`.
