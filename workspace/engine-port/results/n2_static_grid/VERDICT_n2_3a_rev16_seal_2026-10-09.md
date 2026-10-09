> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 그대로 저장했다(2026-10-09).

**규칙 봉인 가능.** 死因 0, 예보 범위 안 판정 경로 구멍 0, 봉인 대상 파일 정의 결함 0. 등급은 `GO-with-caveats`이고 새 caveat는 1건(N2C-75)이다. 새 성능 판정은 0건이다.

## 확인 표

| # | 항목 | 결과 | 근거 |
|---|---|---|---|
| 1-(1) | §18-2-a 분류표 | 해소 | `git ls-files` 31개가 표에 정확히 한 번씩 나온다. 내역: 규칙 7 + 하네스 6 + smoke 2 + 이력 16(판정서 13 + 근거 문서 3). 중복 0, 누락 0. `__pycache__`는 gitignore 대상이고, `REGISTRATION_SHA256.txt`는 표 밖 규칙으로 처리됐다. |
| 1-(1) | §18-2-b 외부 의존 digest | 해소. 3/3 일치 | `n3_capacity.sh` sha256 `b20f71e1…a218b4e`, 마지막 커밋 `93c3cca`. r3 manifest `bec2cba5…893897c3`(파일 626, JSONL 129). r4 manifest `b76d3fae…2a4b9`(파일 695, JSONL 144). 셋 다 등록 정의대로 다시 계산했고 일치했다. |
| 1-(2) | `n2_mutants.py --set core`의 하네스 비의존 | 해소 | 코드 독해: `names = (FE, RD)`와 `TESTS`만 읽고 복사한다. `_harness_mutants()`(`n2_harness_mutants.py`를 importlib로 읽는 함수)는 `mset != "core"`일 때만 호출된다. `MUTANTS` 192종은 전부 `n2_probe_reduce.py` 108곳, `n2_feasibility.py` 86곳만 건드린다(하네스 파일 0). 규칙 시험 2개는 하네스 모듈을 import하지 않는다. core가 쓰는 외부 자원은 `n3_sh_path()` 존재 확인과 `N2_N3_DIR`뿐이고, 둘 다 §18-2-b에 digest가 있다. |
| 1-(3) | §18-2-c 봉인 뒤 수정 절차 | 해소 | 세 단계(§21 색인 → 좁은 델타 감사와 core 장부 재실행 → digest 갱신)가 있다. 하네스 절은 자신이 빌드된 규칙 digest를 기록하고, 두 digest가 불일치하면 dry-run을 금지한다. 예고됐던 두 수정(H14-3, H15-1)은 봉인 전에 반영됐다. |
| 2 | H14-3·H15-1·H15-2·H15-6 반영 | 새 구멍 없음 | 아래 반전 시험 표. 새로 생긴 경로는 모두 fail-closed 방향이다(VALID를 빼는 쪽). 부호나 라벨을 바꾸는 경로는 없다. |
| 3 | N2C-73 문자 일치 | 일치 | 판정서 rev15의 249자 문자열이 PREREG 본문에 그대로 들어 있다(부분 문자열 검사 참). |
| 3 | N2C-74 n/a 사유 | 타당 | N2C-74의 조건은 "`cell_windows`가 PE-time, §7-1 (c), §7-2 (b)를 계산하지 않는다"이다. HEAD 코드는 셋을 모두 계산하므로 조건이 거짓이다. 둘째 금지(합성 시험 ≠ 실제 엔진 이벤트열)는 §17-21과 N2C-3이 계속 공시한다. |
| 4 | K, 예산, `N2_REGISTERED.json` | 불변 | `n2_feasibility.py`, `test_n2_feasibility.py`, `N2_REGISTERED.json`의 diff는 `49451c1`..HEAD에서 0이다. `--verify-registered` 결과는 `{"ok": true, "problems": []}`, rc 0. `PROBE_PLAN.json`은 `reducer_sha256` 한 줄만 바뀌었고, 새 값 `8ef0c8cc…4282d`는 현재 축약기 sha와 같다. |
| — | 규칙 시험 | 141/141 OK, skip 0 | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_n2_feasibility test_n2_probe_reduce`, N3는 저장소 기본 경로, 27.5 s. 첫 실행에서 요약 줄이 출력 tail에 잘려 같은 명령을 한 번 더 돌렸다. 범위 규칙의 "1회"를 벗어났음을 공시한다. |
| — | core 장부 기록 | 지시에 따라 신뢰 | `scratchpad/r16/mut_core.txt`: CTRL OK, 사살 183, 등가 9, 행 수 192. 기록된 ID 집합이 `n2_mutants.MUTANTS`의 ID와 완전히 같다(대칭차 ∅). 변이는 재실행하지 않았다. |

## 반전 시험 표 (rev16에서 새로 생긴 자유 표면, `cell_windows` 경로)

원자료와 추정량: 합성 시험은 등록 축약기 위에서 돌렸다. 실자료 검사 하나는 N3 r4 `tel_g_r069_3704.jsonl`(decode_iteration 3471개)을 썼다. 기술 확인용이며 인용 금지다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| PE-time의 가중 합 0일 때 기본값 | 0.0 → 1.0 (감사 변이 AU1, fail-open) | 없음(예보 범위 밖에서만 발화) | AU1은 83 시험에서 생존했다. 발화하려면 길이 128 창에 iter_seq 연속 쌍이 하나도 없어야 한다. 단독 창에서는 bs≠b 또는 in-flight iteration이 하나 걸러 하나씩 ≥256회 끼어야 하는데, in-flight 쪽이면 §7-1 (c)가 먼저 D_UNREALIZED를 낸다. C 쪽에서는 그보다 먼저 `CO_WINDOW_SHORT`/`BS_UNREALIZED`가 걸린다. → caveat N2C-75 |
| §7-1 (c) 창 끝점 포함 여부 | `≤` → `<` (AU3) | 없음 | AU3 생존. 같은 호스트 시계(`perf_counter`: `telemetry.py:23`, `multiplexing_mixin.py:2251`)에서 끝점과 정확히 같은 시각이 나올 확률은 사실상 0이다. 실자료에서 자기 [t_launch, t_sync] 안에 span 시작이 든 iteration은 0/3471이었다. → N2C-75 |
| §7-1 (c)가 짝 없는 span 이벤트를 포함하는가 | 짝 있는 span만 → 묶임 (AU2) | — | AU2는 사살됐다(`test_alone_no_prefill_in_window` 실패). 등록 문언대로 묶여 있다. |
| PE-time 가중 규약(시작 간격 vs 끝 간격) | J27로 이미 묶임 | 없음 | 장부 기록상 J27 사살 |
| §7-2 (b) S-per 가중 0.95(C) | 삭제(J26) ↔ 존재 | 없음(추가 배제만 가능하고 부호는 불변) | 동거 iteration은 단독 iteration보다 느리다(예보 κ [1.3, 1.6], rev15 판정서 §1-3). 그러면 시간 가중 비율 ≥ 개수 비율이므로, 개수 0.95를 통과한 C가 시간 기준에서 탈락하기는 어렵다. |
| A1 + 트리거 뒤 비-co span → `WINDOW_SHORT` | ON·OFF 공통 | 없음(fail-closed) | rev15의 fail-open(A1 VALID가 ρ_off 짝에 쓰이던 경로)을 닫는다. J25가 이 경로를 묶는다. |
| `t_sync` 결측 → `RECORDER_UNPROVEN` | 전 창 | 없음(N3 위험 없음) | 엔진이 매 decode_iteration마다 `t_sync`를 쓴다(`multiplexing_mixin.py:2258`). 실자료 표본 2파일에서 3471/3471, 2715/2715 존재. |
| H14-3 warm-up 고아 면제 | smoke `reduce_probe`에만 적용 | 판정 경로 아님 | n_in = 0일 때만 면제한다. 짝 없음, 셀과 겹침, 중복은 VD다. Job 경로(`n2_window`)는 아직 미구현이다. |

반전을 만든 자유 표면: 0개. 반전 계산은 등록 축약기와 등록 시험을 그대로 썼고, 새 추정량은 도입하지 않았다.

## 감사자 변이 (3종, 단일 스크립트, `test_n2_probe_reduce`만 실행, `N2_N3_DIR` 무효화)

| ID | 변이 | 결과 |
|---|---|---|
| AU1 | `_pe_time`: 가중 합 0이면 1.0 반환 | SURVIVED(83 OK) → 예보 밖 등가, N2C-75 |
| AU2 | §7-1 (c)를 짝 있는 span 이벤트로만 판정 | KILLED(`test_alone_no_prefill_in_window`) |
| AU3 | §7-1 (c) 끝점을 `<`로 바꿈 | SURVIVED(83 OK) → 예보 밖 등가, N2C-75 |

N3 양성 대조 경로(`n3_theta_positive_control`)는 `cell_windows`를 거치지 않으므로, N3를 끈 것은 이 세 변이의 판정에 영향이 없다.

## 봉인 시 지시 (봉인 차단 아님)

- **S-1**: 이 판정서는 32번째 추적 파일이 되지만 §18-2-a에 없다. 봉인 커밋의 `REGISTRATION_SHA256.txt`에 "`VERDICT_n2_3a_rev16_seal_2026-10-09.md` = 이력(봉인 바이트 아님)"이라고 한 줄 적는다. PREREG에 행을 추가하면 규칙 digest가 바뀌므로, 추가하려면 봉인 전에 하고 이 판정서를 그 커밋 해시로 인용한다.
- **S-2**: `mut_core.txt`의 수정 시각(16:49)이 rev16 커밋(16:57)보다 앞선다. 그 사이 규칙 파일이 바뀌지 않았다는 증거는 장부 안에 없다(ID 집합 일치만 확인했다). 봉인 기록에 장부가 돌아간 `n2_probe_reduce.py`·`n2_feasibility.py`의 sha를 적는다. 현행은 축약기 `8ef0c8cc…4282d`, 변이 스크립트 `2c502d0c…af72`다. 확인할 수 없으면 봉인 커밋에서 `--set core`를 한 번 다시 돌린다.

## 하네스 단계 수리 / 다음 규칙 개봉 때 함께 할 것

- **H16-1**: AU1과 AU3를 사살하는 경계 시험 2개를 추가한다. 가중 합 0 창 → D_UNREALIZED/COLOC_UNREALIZED가 되는 시험, 끝점과 같은 시각의 prefill 이벤트 → D_UNREALIZED가 되는 시험이다. 규칙 파일(`test_n2_probe_reduce.py`)을 고치는 일이므로 §18-2-c 절차로 하고, 하네스 봉인과 같은 회차에 묶는 것을 권고한다. 그 전까지는 N2C-75가 적용된다.
- 기존 선행 조건 H15-3, H15-4, H15-5와 `n2_marks`의 `warmup_start/end` 기록, `test_n2_harness` C1을 `arity ∪ WARMUP_ARITY`로 확장하는 일은 변함없이 하네스 봉인 선행 조건이다.

## 신규 caveat (인용 금지 문장, 문자 그대로 승계)

- **N2C-75**: "규칙 봉인 시점의 `_pe_time` '가중 합 0 → PE-time 0(fail-closed)' 규약과 §7-1 (c)의 창 끝점 포함(`≤`) 규약은 등록 문언과 코드가 일치하지만 시험으로 묶이지 않았다(감사 변이 AU1·AU3 생존). 이 두 규약이 판정에 닿지 않았다고 쓰려면 Job 원자료에서 가중 합 0인 창 수와 창 끝점과 같은 시각의 prefill 이벤트 수가 0임을 보고한 뒤에만 쓴다."

---

# 판정서 본문 — `VERDICT_n2_3a_rev16_seal_2026-10-09.md`

> 결론(한 줄): **규칙 봉인 가능** — 死因 0, 예보 범위 안 판정 경로 구멍 0, 봉인 대상 파일 정의 결함 0. 등급 `GO-with-caveats`(신규 caveat N2C-75). 새 성능 판정 0건.

**대상**: `workspace/engine-port/results/n2_static_grid/` HEAD `c54d5c5`(rev16). 대조본은 `49451c1`(rev15 + 판정서 rev15).
**선행**: `VERDICT_n2_3a_rev15_delta_2026-10-09.md`. 봉인 차단 사유는 R15-1 하나, 후속은 H15-1…H15-6, caveat는 N2C-73/74였다.
**범위 제한(요청자 지정)**: 변이는 재실행하지 않았고 `scratchpad/r16/mut_core.txt` 기록을 신뢰했다. 규칙 시험은 1회 실행했다(요약 줄이 잘려 같은 명령을 한 번 더 돌림, 공시). 감사자 변이는 `cell_windows` 경로에서 3종만 돌렸다.
**봉인 기준**: (a) 死因 (b) 예보 범위 안 판정 경로 구멍 (c) 봉인 대상 파일 정의 결함. 이 셋만 봉인을 막는다.

## 1. R15-1 해소 확인

- **(1) 분류표 §18-2-a**: `git ls-files`의 31개가 정확히 한 번씩 나온다. 규칙 봉인 7은 PREREG, 계산기, 축약기, `N2_REGISTERED.json`, 규칙 시험 2, `n2_mutants.py`다. 하네스 봉인 6은 `n2_marks`, `n2_loadgen`, `n2_probe_run`, `fake_n2_engine`, `test_n2_harness`, `n2_harness_mutants`다. smoke 전용 2는 `n2_probe_workspace.sh`, `PROBE_PLAN.json`이다. 이력 16은 판정서 13개(rules, rev2–rev8, rev9, rev10, rev11, rev14, rev15)와 근거 문서 3개다. 중복 0, 누락 0.
- **(1) 외부 의존 §18-2-b**: 세 digest를 등록 정의(`find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum`)대로 다시 계산했고 모두 일치했다. 파일 수와 JSONL 수도 같다.
- **(2) 장부 분할**: `run(mset="core")`는 `(FE, RD)`와 `TESTS`만 읽고 복사하며, `_harness_mutants()`를 호출하지 않는다. `MUTANTS` 192종이 건드리는 파일은 축약기 108곳과 계산기 86곳뿐이다. 규칙 시험의 import는 `n2_feasibility`, `n2_probe_reduce`와 표준 라이브러리뿐이다. 시험 수 문언(규칙 141 = 58 + 83, 하네스 35)은 실행 결과 `Ran 141`과 일치한다.
- **(3) 수정 절차 §18-2-c**: 처방 문언대로 등록됐고, 예고된 두 수정은 봉인 전에 반영됐다.

## 2. 새 반영분의 판정 경로 검사

- **H15-1** `cell_windows`: PE-iter 다음에 PE-time(S-per 가중, 시작 간격, 연속 쌍 규칙)과 §7-1 (c)(창 [첫 `t_launch`, 마지막 `t_sync`] 안의 모든 `prefill_span_start/_end`, 짝 없는 것 포함)가 이어진다. `t_sync` 결측은 `RECORDER_UNPROVEN`이다. E-α A1에서 트리거 뒤 비-co span이 있으면 `WINDOW_SHORT`다. C에는 S-per 가중 0.95가 붙는다. 코드와 §7-1 (b)의 등록 문언(가중 합 0이면 0)이 일치한다. 모든 경로가 VALID를 빼는 방향(fail-closed)이다.
- 시계 정합: `t_launch`/`t_sync`(`perf_counter`, `multiplexing_mixin.py:2251`)와 `timestamp_monotonic_s`(`perf_counter`, `telemetry.py:23`)는 같은 시계다. N3 r4 실자료 3471개 iteration에서 자기 [launch, sync] 안에 span 시작이 든 경우는 0이었다. `t_sync`는 표본 2파일에서 전부 존재했다(3471/3471, 2715/2715). 따라서 `RECORDER_UNPROVEN`이 전면 발화해 예측이 도달 불가해질(N3) 위험은 없다.
- **H15-2**: 경계 시험(127/128, 126/128)과 OFF A1/A2 경로 시험이 있고, 장부상 J20이 사살됐다.
- **H14-3**: smoke `reduce_probe`에만 적용되며 판정 경로가 아니다. 면제는 n_in = 0일 때만이고, 짝 없음, 겹침, 중복은 VD다. `for a, b in warm:`의 변수 가림은 그 뒤 `b`를 배치 크기로 쓰지 않으므로 무해하다.
- **H15-6/R15-2**: §6-1의 두 문장이 정정됐다(`CO_WINDOW_SHORT`는 필터 뒤 C 값 창 기준, 함수명은 `cell_windows`).
- 반전 시험 표는 위와 같고, 반전은 0개다.

## 3. Caveat

- N2C-73: 문자 그대로 등재됐다(확인 완료).
- N2C-74: 해당 없음. 조건 사실이 거짓이므로 타당하다.
- **N2C-75**(신규): "규칙 봉인 시점의 `_pe_time` '가중 합 0 → PE-time 0(fail-closed)' 규약과 §7-1 (c)의 창 끝점 포함(`≤`) 규약은 등록 문언과 코드가 일치하지만 시험으로 묶이지 않았다(감사 변이 AU1·AU3 생존). 이 두 규약이 판정에 닿지 않았다고 쓰려면 Job 원자료에서 가중 합 0인 창 수와 창 끝점과 같은 시각의 prefill 이벤트 수가 0임을 보고한 뒤에만 쓴다."

## 4. 불변 확인

계산기, 계산기 시험, `N2_REGISTERED.json`은 diff 0이다. `--verify-registered`는 ok다. K는 (33, 15) 그대로이고 κ\* 표와 예산도 그대로다. `PROBE_PLAN.json`은 축약기 sha 한 줄만 바뀌었다(현재 축약기 sha256 `8ef0c8cc6b0ba273ab684a8e297441598ffd70453138ee1f10713a8948d4282d`와 같음). 작업 트리는 깨끗하다.

## 5. 봉인 시 지시와 후속

- S-1: 이 판정서를 `REGISTRATION_SHA256.txt`에 이력으로 기재한다. PREREG를 편집하려면 봉인 전에 한다.
- S-2: core 장부가 돌아간 파일 sha를 봉인 기록에 적는다(`mut_core.txt` 시각 16:49 < 커밋 16:57). 확인할 수 없으면 봉인 커밋에서 `--set core`를 다시 실행한다.
- H16-1: AU1·AU3 사살 시험은 §18-2-c 절차로, 하네스 봉인 회차에 넣는다.
- H15-3, H15-4, H15-5와 warm-up 마크 기록, C1 확장은 하네스 봉인 선행 조건으로 그대로 남는다.

⚠️ 규칙 봉인은 "판정 경로가 실제 엔진에서 성립한다"는 뜻이 아니다. 합성 시험에 묶였다는 뜻일 뿐이다(§17-21, N2C-3). 하네스 봉인과 Job 결과 감사가 따로 필요하다.

---
관련 경로:
- /home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/PREREG_N2_3A_2026-10-07.md (§18-2-a/b/c, §7-1 (b), §17-21, §20-1)
- /home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/n2_probe_reduce.py (`_pe_time`, `cell_windows`, `reduce_probe` warm-up 블록)
- /home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/n2_mutants.py (`run`, `_harness_mutants`)
- /home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/VERDICT_n2_3a_rev15_delta_2026-10-09.md
- 감사자 변이 스크립트: /tmp/claude-1000/-home-wonho-Experiments-KISTI/2657f214-f740-4503-8f81-241f78b1e2b2/scratchpad/aud/m.py
- core 장부 기록: /tmp/claude-1000/-home-wonho-Experiments-KISTI/2657f214-f740-4503-8f81-241f78b1e2b2/scratchpad/r16/mut_core.txt