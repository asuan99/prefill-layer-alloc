# Q1 준비 자료 — N2(3-a) 종료 여부 (2026-10-15)

> 지위: main 트랙 논의 메모. 측정 0, 새 판정 0, 정본·봉인 파일 무편집. 출발점은 `DISCUSSION_INDEX_2026-10-15.md` §4(:90–122)다.
> 경로 약칭: `N2/` = `workspace/engine-port/results/n2_static_grid/`, `PS` = `PROJECT_STATUS.md`, `MG:` = 브랜치 `multigpu-scaling`의 파일(`git show`로 읽음).
> 표기: [사실]은 파일:행 인용, [해석]은 이 메모의 판단이다. 확인하지 못한 것은 "미확인"으로 적었다.

## 0. 결론 요약

1. PROFILE 셀·boot 단위 4추정량은 **판정 입력이 아닌 기술값**이다. 그리고 점 라벨을 인용할 때 필요한 병기 항목도 아니다. 그래서 **산출하지 않고 종료할 수 있다.** 이때 "등록 기술값 미산출"을 종료 기록에 공시한다. 산출해서 인용하려면 result-analyst 산출과 claims-auditor 결과 감사 1회가 필요하다.
2. "캠페인 종료 선언"만으로는 merge 장애가 다 풀리지 않는다. 봉인은 두 층에서 작동한다. 층 1(새 Job의 preflight abort)은 Job을 내지 않으면 발화하지 않는다. 층 2(main의 CPU 회귀 중 봉인된 시험 2건 이상)는 merge 직후 **실패**한다. 그 시험 파일은 그 자체가 봉인돼 있어 고칠 수 없다. 따라서 **봉인 시험 처리 규약과 봉인 커밋 고정**이라는 추가 절차가 필요하다.
3. 이 제약은 merge만의 문제가 아니다. G0 계측 보강(Q14)처럼 `src/multiplex`를 main에서 고치는 일도 같은 시험을 깬다. 그래서 N2/N3/E2를 닫는 결정은 merge와 무관하게 G0 착수의 선행 조건이다.
4. N3(shape B 4회차)와 E2(VESSL r2)는 둘 다 결과 감사까지 끝났다. 남은 것은 새 Job이 필요한 항목뿐이다. **세 캠페인을 같이 닫고 게이트 #6은 열린 채 G0 명세로 이관하는 편**이 merge 영향이 가장 작다.
5. 권고는 **(a)를 유지**하고 세 가지를 보정한다. (i) Job 종료와 아카이브 분석 종료를 분리한다. (ii) F-1은 N2 안에서 하면 엔진 등록 재개봉이 필요하다. 그러니 (b)는 색인이 적은 것보다 비싸다. (iii) merge 전에 봉인 시험 규약을 정한다.

---

## 1. 확인 ① — PROFILE 동거 4추정량: 감사 범위와 "산출 없이 종료" 가능성

**[사실]**
- 4추정량의 정체는 "실현 분할 4추정량"(E-cnt/E-time/E-iter/E-qcond, A2-pair)이다. 동거 통계 전용량이 아니다. 셀·boot 단위 판에 burst 동거 구간이 들어갈 뿐이다. 출처: `N2/PREREG_N2_3A_2026-10-07.md` §7-3 표(:307–312), :315.
- 지위는 기술값이다. "4추정량은 판정 입력이 아니다 … 판정 입력은 PE-iter/PE-time뿐"(:316). §12-2는 이를 기술 통계 목록에 둔다(:707).
- 점 라벨을 인용할 때 필수 병기는 PE-iter·PE-time·창 W-idx·Δdropped·ctx다(:317). 셀·boot 4추정량은 그 목록에 없다.
- 산출 시점 규정은 "Job-EA 결과 봉인 **뒤** 산출한다"이다(:315). 기계적 정의는 `--post-ea-seal` 재대조이고, "거버넌스(결과 감사·커밋)는 이 함수 밖"이다(§18-2-d #28, :1089).
  - `N2/n2p_…7e09cc3…/EA_RESULT_SEAL.json`이 있다(sealed_utc 2026-10-13T06:17:02Z, rule digest `56473d6f…`).
  - EA 결과 감사는 커밋 `fa454c4`로 끝났다.
- PROFILE 러너 산출의 `blind.coloc_statistics_computed`는 `false`다(`N2/n2p_…a5de2d6…/N2_PROFILE_RESULT.json`). 즉 아직 산출되지 않았다.
- 노출 2건:
  - N2C-87은 분석자가 r1 셀 6개의 burst 시각·iteration 수를 출력한 일이다(PREREG :1229, `CHECK_profile_a5de2d6_2026-10-12.md` §0 :9–20).
  - N2C-98은 감사자가 r4 boot 3개의 동거 iteration 수를 센 일이다(:1241).
  - 두 caveat는 **E-α 인용 문안**을 제약한다. "블라인드가 Job-EA 결과 봉인까지 지켜졌다"는 쓰지 않는다. K는 "rev14 등록값 그대로이며 노출 뒤에도 바뀌지 않았다"로만 쓴다(VERDICT_result §6 (K) :57).
  - 개정 창은 2026-10-12에 닫혔다(PREREG §5-4 :218).
- 정본의 현재 서술은 "산출 허용 여부는 사용자/claims-auditor 거버넌스 확인 사항"이다(`PS:39` 다음 gate (2)). 근거는 셋이다. PROFILE이 "경량 확인"으로 지위가 바뀌었고(사용자 결정 2026-10-12, `CHECK…` :3), N2C-87·98 노출이 있었고, N2C-95가 있다(봉인 파일은 sha만 묶는다, :1238).
- PROFILE 전체(점·D-민감도)도 정식 결과 감사를 거치지 않았다. COSTMODEL CM-1은 "N2 PROFILE(경량 확인만, 수치 인용 금지)"로 적는다(`COSTMODEL.md:18`, :37).

**[해석]**
- 블라인드 의무는 Job-EA 결과 봉인 전까지만 걸려 있었다. EA는 이미 봉인·감사됐다. 그래서 지금 4추정량을 산출해도 EA 판정에 영향을 줄 경로는 없다. 노출 2건은 산출을 막는 사유가 아니라 E-α 인용 문안의 제약이다.
- **산출하지 않고 종료하는 길은 있다.** 판정 입력이 아니고 점 라벨 인용 병기에도 필요 없다. 다만 :315가 "산출한다"고 등록했으므로, 미산출은 등록 계획에서 벗어난 것이다. 종료 기록에 "사용자 결정으로 미산출, 판정 경로 영향 0"을 공시해야 한다. 규칙 파일(PREREG)을 고칠 필요는 없다고 본다. §18-2-c는 규칙 파일 *수정*의 절차이기 때문이다.
- **산출해서 인용하려면** result-analyst 산출과 claims-auditor 결과 감사 1회면 형식상 충분하다(감사 상한은 같은 대상 2회). 감사가 반드시 다룰 것은 두 가지다.
  - 금지 #41(EA 수치와 PROFILE 수치의 비교·결합) 아래에서 PROFILE burst 동거 값을 EA ρ 옆에 두지 않는 문안
  - N2C-73·95 병기
- 소비처가 약하다. 이 4추정량이 닫는 COSTMODEL 칸은 확인되지 않았다(CM-4 분리 경로는 F-1·F-3, `COSTMODEL.md:60`). 그러니 **산출하지 않는 쪽이 비용 대비 낫다.**
- 별개 쟁점: 종료 후 PROFILE 점·D-민감도를 CM-1 근거로 쓰려면, 4추정량과 무관하게 **PROFILE 결과 감사 1회**가 필요하다. 이 감사는 아카이브만 읽는 CPU 작업이다. N2 분석기(`n2_analyze.py`·`recheck_*.py`)는 엔진 digest를 검사하지 않으므로(grep: `engine_check` 호출 0) merge 뒤에도 할 수 있다.

## 2. 확인 ② — 봉인 digest 제약: 정확한 기전과 해제 조건

**[사실 — 문구의 출처]**
- "main merge 금지"는 트랙 파일(`../prefill-layer-alloc-multigpu/tracks/multigpu.md:15`)과 main 설계 메모(`DESIGN_span_and_integration_2026-10-15.md:106`, :129 "종료 선언 → merge")에만 있다. 등록 규칙이나 감사 판정서에서 온 문구가 아니다(MG `results/multigpu/*.md` grep 0건).
- 1차 근거는 둘이다.
  - `MG:workspace/engine-port/results/multigpu/PREREG_MULTIGPU_2026-10-12.md:21`: "엔진 트리는 N2/N3/E2 봉인 digest와 다르다(의도)"
  - `MG:workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh:173–180`: manifest 25→27항목. "A registration that compares the WHOLE entry set (n2_plan.engine_check, n3_plan, e2_vessl_plan) will report these two as installed-but-unregistered … intended fail-closed … those campaigns pin the pre-M2 engine"
- 브랜치의 엔진 변경은 다음과 같다(merge-base `20e184d` 기준).
  - `src/multiplex/{multiplexing_mixin, telemetry, holb_probe, chunk_probe}.py` 수정
  - `multigpu_guard.py` 신설
  - `pdmux_context_device_assert.patch` 신설
  - `sync_engine_tree.sh` 수정
  - 4개 파일 sha는 main·N3 등록값과 다르다(예: mixin main `ba5835aa…` = N3 등록, 브랜치 `c2483a14…`). `profile.py`는 같다(`d7b6a807…`).

**[사실 — 기전 3층]**

| 층 | 무엇을 검사하나 | 코드 | merge 뒤 결과 |
|---|---|---|---|
| L1 Job preflight | 설치 트리 manifest 전체 항목 집합 = 하드코딩 등록 manifest(N2·N3: 25항목, E2: 908623 + mixin 1파일 허용) | `N2/n2_plan.py:116–142,656–659`, `N2/n2_capacity.sh:149–151`; `n3_capacity/n3_plan.py:85,487–490`, `n3_capacity.sh:179–180`; `r2_eval/e2_sticky_vessl/e2_sticky_vessl_r2.sh:234–237` | 그 캠페인 Job을 merge된 HEAD에서 내면 `ABORT_ENGINE_TREE_DRIFT`(exit 3)로 측정 전에 멈춘다. 파일 4개가 그대로여도 27항목이라는 것만으로 불일치다 |
| L2 로컬 CPU 회귀 | 추적 중인 `src/` 미러 sha = 등록 manifest | `tests/test_n3_capacity.py:65–67`(`mirror_check`), `tests/test_e2_vessl.py:208–210`(`this_head` = `ba5835aa…`, `e2_vessl_plan.py:95`) | merge 즉시 **실패**한다(최소 2건 확인. 전수는 미확인 — merge 미리보기 트리에서 전체 회귀를 돌리지 않았다) |
| L3 봉인 시험의 상호 고정 | 봉인 파일 바이트 = 등록 sha | `tests/test_e2_vessl_r2.py:60–74,206–209`(`test_e2_vessl.py`를 `d163d702…`로 고정); N3 봉인 표가 `tests/test_n3_capacity.py` sha를 기재(`VERDICT_n3_rev5_delta_2026-10-06.md:133`) | L2 실패 시험을 고치거나 skip하면 L3가 실패한다. **봉인 바이트 수정으로는 해소할 수 없다** |

- 과거 결과는 영향을 받지 않는다. Job은 `PDMUX_COMMIT`을 체크아웃해 그 커밋의 `src/`를 설치하고(`scripts/vessl/job_entry.sh:101–105,124–126`), 결과는 커밋 해시(`a5de2d6`, `7e09cc3`, `93c3cca`, `22c8c64`)에 묶여 있다.
- `launch.sh`는 **실행하는 worktree의 HEAD만** 번들한다(`scripts/vessl/launch.sh:46,70–76`). 그래서 merge 뒤에도 봉인 커밋을 체크아웃한 별도 worktree에서는 원칙상 재제출할 수 있다(실행은 미확인).
- 아카이브 재분석 도구는 엔진 digest를 검사하지 않는다(`n3_plan.mirror_check`는 `--selftest`·`--verify-digests` 경로에서만 쓰인다, `n3_plan.py:931–932,1095`).
- 텍스트 충돌은 옛 방식 `git merge-tree`(읽기 전용) 미리보기에서 표지 0개였다. 양쪽이 고친 파일은 `scripts/vessl/job_entry.sh` 하나다. 실제 merge 시험은 미실행이다.
- 브랜치 코드 주석은 "tp=dp=1에서 경로·레코드 불변, 분할·batch·입장 결정 불변"이라고 적는다(MG mixin 말미 주석). 다만 이것이 N2/N3 측정 경로와 **바이트 동등하다는 GPU 실증은 미확인**이다.

**[해석 — 해제 조건]**
- "종료 선언"이 실제로 바꾸는 것은 L1뿐이다. 그 캠페인 Job을 다시 내지 않으면 L1은 발화할 일이 없다. L2·L3은 선언과 무관하게 merge 순간 main의 CPU 회귀를 빨갛게 만든다. 그래서 다음 추가 절차가 필요하다.
  1. **봉인 커밋 고정**: 각 캠페인의 최종 봉인 커밋을 종료 기록에 적는다. 선택으로 로컬 태그를 붙인다. 예: `seal/n2-rev18` → 하네스 봉인 `7e09cc3`(규칙 `5ea2a60`), `seal/n3-rev7` → `93c3cca`, `seal/e2-vessl-r2` → `22c8c64`. 이 프로젝트에 태그 관례는 없다(`git tag` = `backup-pre-signfix-2026-09-21` 1개). 그래서 태그 사용 자체가 사용자 결정이다.
  2. **봉인 시험 처리 규약**(택1, 사용자 결정):
     - (가) main 회귀에서 "종료 캠페인의 기지 실패" 목록으로 분리 공시한다. 시험 파일은 무편집이고, 재현은 봉인 커밋 worktree에서 한다.
     - (나) 회귀 실행 명령(CLAUDE.md의 `unittest discover`)을 봉인 시험 제외판으로 바꾼다. 이것은 CLAUDE.md 개정이므로 promote 경로를 탄다.
     - 어느 쪽이든 봉인 파일 바이트는 건드리지 않는다.
  3. **merge 전 미리보기 회귀**: 임시 worktree에서 merge한 트리로 전체 CPU 회귀를 돌려 실패 목록을 확정한다. 그 목록이 위 규약의 입력이다.
- harness digest 재계산이나 봉인 동결 태그는 해제의 **필수** 조건이 아니다. 봉인 파일을 바꾸지 않는 한 재계산할 대상이 없다. 오히려 재계산하면 봉인 의미가 깨진다.

## 3. 확인 ③ — N3 shape B·E2를 함께 닫을지

| 캠페인 | 최종 봉인·Job | 현재 상태 [사실] | 열린 항목 [사실] | 같이 닫을 때 [해석] | 열어 둘 때 [해석] |
|---|---|---|---|---|---|
| N2(3-a) | 규칙 rev18 `5ea2a60`(`56473d6f…`), 하네스 `7e09cc3`(`9257845f…`); PROFILE `a5de2d6`, EA `7e09cc3` | EA 결과 감사 완료, 死因 0(`PS:9`); PROFILE은 경량 확인만(`PS:39`) | F-1…F-4(VERDICT_result §10 :75); PROFILE 4추정량(§1); N2C-10 "3-a 완결 아님"(PREREG :1125 — L>65536 셀 없음) | 엔진 의존 Job 0 → L1 무관. 3-a의 L>65536 셀은 Q6(128k 프로브)·G0로 이관 | F-1은 burner 주입이 엔진 변경이라(해석, 미확인) N2 ENGINE_MANIFEST 재등록 + §18-2-c 재개봉 + 하네스 재봉인이 필요 → N2를 열어 둬도 merge와 충돌 |
| N3(게이트 #6) shape B | rev7 `93c3cca`, Job `…93c3cca-20261007064124` | 4회차 결과 감사 死因 0, 라벨 3개 기계적 `CONFIRMED`(`PS:84`) | shape A 미실행(사용자 결정 2026-10-11 "보류, N2 먼저", `PS:47`); 확정 조건 ①–④(`PS:100`); 게이트 #6 "부분 측정, 미닫힘"(`PS:102`) | 캠페인을 닫아도 **게이트 #6은 열린 채** 남는다. ③ σ_Job은 Q5(Job ≥3), 용량 정의는 Q9, shape A·k 위치는 G0 PREREG(Q24)로 이관 | shape A를 rev7 봉인으로 돌리려면 merge 전에 main HEAD에서 하거나, merge 뒤 봉인 커밋 worktree에서 해야 한다. 어느 쪽이든 Q14 계측이 없는 옛 트리라 G0와 비교할 수 없다(#262: Job 간 비교 금지) |
| E2 sticky VESSL | r2 `22c8c64`, Job `…22c8c64-20261006020317` | 결과 감사 死因 0, 라벨 기계적 `CONFIRMED`(`PS:143`) | Q1(sticky의 D44 실현)·R2 기전은 `NOT-YET-SUPPORTED`(`VERDICT_result_e2_vessl_r2_2026-10-06.md` 판정표); 다음 gate = "N2가 E2VR-6 승계"(`PS:151`) → 예정 Job 없음 | 비용 거의 0. E2VR-6은 이미 N2가 승계 | 얻는 것 없음. L2(`test_e2_vessl.py:208`)는 선언과 무관하게 merge 때 실패 |

- [해석] 세 캠페인 모두 남은 일은 새 Job이고, 남은 분석은 아카이브만 쓴다. 하나라도 열어 두면 merge 뒤 그 캠페인 Job은 봉인 커밋 worktree에서만 낼 수 있다. 그 트리에는 G0 계측과 엔진 개선이 들어가지 않는다. **함께 닫는 편이 일관적이다.** 단 "N3 종료 = 게이트 #6 닫음"으로 쓰면 금지 문안이다(N3 금지 17항목 최상위, `PS:129`). 종료 기록에 "게이트 #6은 미닫힘, G0로 이관"을 명시해야 한다.
- [사실] λ0 VESSL도 같은 종류의 L1 검사를 가진다(`r2_eval/lambda0_vessl/lambda0_vessl.sh`). 그러나 그 CPU 시험은 합성 manifest만 쓴다(`tests/test_lambda0_vessl.py:142–150`). 그래서 L2 영향은 확인되지 않았다(미확인, 전체 회귀로 확정 필요).

## 4. 선택지 재평가

| 선택지 | merge 시점 | CM-4 크기 분리 | 추가 엔진 작업 | 봉인 처리 | 평가 |
|---|---|---|---|---|---|
| (a) N2 종료 + 흡수(F-2·F-1 → 명세 계측, F-3 → 5-a, F-4 포기) | 가장 빠름(§2 절차 3개 후) | G0 이후 | G0 계측(Q14)만 — merge된 트리 하나에서 | N2/N3/E2 동시 종료 + 시험 규약 | **권고 유지** |
| (b) F-1·F-2 후 종료 | F-1 완료 후 | N2 안에서 먼저 | burner 주입(미확인) + N2 엔진 등록 재개봉 + 하네스 재봉인 + 정확성 게이트 | 재개봉 1회 추가 후 종료 | 색인 평가보다 비쌈: F-1 자체가 N2 엔진 봉인을 바꾼다(해석) |
| (c) 전부 | 가장 늦음 | N2 안 | (b) + F-3 별도 사전등록 + F-4 격자 Job | 재개봉 여러 회 | 비권고 |

권고 (a)의 보정은 다음 세 가지다. 권고 자체는 바뀌지 않는다.
1. **Job 종료와 분석 종료를 분리한다.** 종료는 "이 봉인으로 새 Job을 내지 않는다"는 뜻이다. PROFILE 결과 감사처럼 아카이브만 쓰는 분석은 merge 뒤에도 할 수 있다.
2. **F-1의 흡수처는 G0 명세의 "전력 정합 대조" 계측 항목이다.** burner 주입 경로와 ≈376 W 도달 가능성은 G0 PREREG 전에 engine-porter가 확인할 미확인 항목으로 넘긴다.
3. **봉인 시험 규약을 merge의 선행 조건으로 둔다**(§2 해석 2–3).

## 5. 사용자에게 물을 하위 결정

| # | 결정 | 선택지 | 메모 |
|---|---|---|---|
| D1 | N2(rev18) 캠페인 종료(새 Job 없음) | 예 / 아니오 | 아니오면 (b)·(c) |
| D2 | F-4(확장 격자) 포기 확인 | 포기 / G0에서 D 스윕으로 흡수 | 포기하면 "ρ의 D·b·c 무관"은 영구히 `NOT-YET-SUPPORTED`(금지 35 유지) |
| D3 | F-3 → 5-a 흡수 확인 | 예 / 별도 사전등록 유지 | N2C-107을 닫는 유일 경로(VERDICT §10)라는 지위를 5-a PREREG가 승계해야 함 |
| D4 | F-1·F-2 → 공통 측정 명세(Q11·Q14) 흡수 | 예 / 아니오 | F-1 실현 가능성은 미확인 |
| D5 | PROFILE 셀·boot 4추정량 | 미산출(공시) / 산출 + 결과 감사 1회 | 권고: 미산출 |
| D6 | PROFILE 점·D-민감도 결과 감사 | 하지 않음(CM-1에서 영구 인용 금지) / claims-auditor 1회(CPU, merge 뒤 가능) | CM-1 근거로 쓸지에 달림 |
| D7 | N3(rev7) 캠페인 종료 + 게이트 #6 G0 이관 | 예 / shape A를 merge 전 실행 | 2026-10-11 "shape A 보류" 결정을 대체함 |
| D8 | E2(VESSL r2) 캠페인 종료 | 예 / 아니오 | — |
| D9 | 봉인 시험 처리 규약 | (가) 기지 실패 공시 / (나) 회귀 명령 개정(CLAUDE.md) | 봉인 바이트 수정은 선택지에 없음 |
| D10 | 봉인 커밋 로컬 태그 | 붙임 / 종료 기록에 해시만 | 태그는 push하지 않음 |
| D11 | 3-a 잔여(L>65536 셀, N2C-10) 이관처 | Q6 128k 프로브 / G0 / 보류 | — |

## 6. 종료를 택했을 때의 실행 체크리스트

| 순서 | 누가 | 무엇을 | 어디에 |
|---|---|---|---|
| 1 | 사용자 | D1–D11 결정 | — |
| 2 | main 세션 | 종료 기록 작성(봉인 커밋·digest, 미산출 등록 기술값, 승계 caveat N2C-1…108·N3VR·E2VR 목록, "게이트 #6 미닫힘", "N2C-10 3-a 미완결", 금지 32–42 유지). 봉인 파일(PREREG·`REGISTRATION_SHA256.txt`·하네스·시험)은 **무편집** | `N2/CLOSURE_n2_3a_<날짜>.md`, `n3_capacity/CLOSURE_…`, `r2_eval/e2_sticky_vessl/CLOSURE_…`(새 파일) |
| 3 | claims-auditor(선택, D5/D6이 "감사"일 때) | PROFILE 결과 감사 1회. 금지 #41, N2C-73·87·95·98 병기를 점검 | `N2/VERDICT_result_n2_profile_a5de2d6_<날짜>.md` |
| 4 | experiment-runner | 임시 worktree에서 `main`+`multigpu-scaling` 시험 merge → 전체 CPU 회귀 + `check_citation_stops.py` → 실패 목록 확정(Job 0) | scratchpad(커밋 없음) |
| 5 | main 세션 | D9 규약 반영. (가)면 기지 실패 목록 파일, (나)면 CLAUDE.md 회귀 명령 개정 | (가) `workspace/engine-port/tests/` 밖의 목록 문서 / (나) promote 경로 |
| 6 | git-committer | (D10이면) 로컬 태그 3개. 종료 기록 커밋 | main |
| 7 | main `promote` | `PS` (a⁶) 다음 gate를 종료로 교체, CONSENSUS §5 항목 20–23 갱신(N3·N2), E2 항목 종료 표기, COSTMODEL CM-1(PROFILE 지위)·CM-4(F-1…F-4 → 명세 이관) 행, MEMORY 인덱스 | 정본 |
| 8 | multigpu 세션 | 트랙 파일 :15 문구를 "N2/N3/E2 종료(<커밋>) — merge 가능, 기지 실패 규약 참조"로 갱신 | `../prefill-layer-alloc-multigpu/tracks/multigpu.md` |
| 9 | git-committer(사용자 지시 시) | `multigpu-scaling` → main merge. `job_entry.sh` 양쪽 변경 확인 | main |
| 10 | main 세션 | G0 PREREG에 merge 뒤 엔진 digest(27항목)를 새로 등록. 봉인 캠페인 수치와 비교 금지 명시(N2 금지 41, N3 #262) | G0 PREREG |

## 7. 미확인 목록

- merge한 트리의 전체 CPU 회귀 실패 수(최소 2건만 확인: `test_n3_capacity.py:65`, `test_e2_vessl.py:208`). λ0·기타 봉인 시험의 L2 영향.
- 실제 `git merge`의 충돌 여부(옛 방식 merge-tree 미리보기 표지 0개만 확인).
- 브랜치 엔진의 tp=dp=1 경로가 N2/N3 측정 경로와 GPU에서 동등한지(코드 주석 주장만 있음).
- F-1 burner를 서버 프로세스 green-ctx stream에 주입하는 경로와 ≈376 W 도달 가능성(VERDICT §10 그대로).
- 봉인 커밋 worktree에서 재제출하는 경로의 실제 동작(launch.sh가 HEAD만 쓴다는 것까지만 확인).
- PROFILE 4추정량이 닫는 COSTMODEL 칸(찾지 못함).
