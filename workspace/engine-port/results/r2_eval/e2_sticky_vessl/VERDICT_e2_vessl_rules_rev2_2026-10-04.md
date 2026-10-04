# 판정서 rev2 — E2 sticky 대조 VESSL 이식: 국소 재감사(필수 수리 6) + 하네스층 국소 확인 (2026-10-04)

> **지위**: claims-auditor 회차. 1회차 판정서 `VERDICT_e2_vessl_rules_2026-10-04.md`(`NO-GO`, 死因 1)의 수리를 확인한다. 읽기 전용, GPU 0, 저장소 파일 수정 0건. 감사자는 reachability 도구를 spec **사본**에 돌렸고 출력은 scratchpad로 보냈다. 감사자 최종 보고를 메인 세션이 이 파일에 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.

## 최종 판정: **`GO-with-caveats`**

- 死因 1은 해소됐다. 남은 死因은 0건이다.
- caveat는 E2VC-1…E2VC-11이다.
- 이 등급은 **실행 승인**이다. 게이트 #6, λ0R-8(iii), P2 블로커 ②, Claim D/E 중 어느 것도 닫지 않는다.
- **제출은 아래 "남은 제출 선행조건"이 모두 끝난 뒤에만 한다.**

## (1) 死因 1 교체 문안과 A13의 나머지 diff

**교체 문안: E2VC-1 취지와 일치한다.**
- A13 §4-1(L256–262): "Q1은 R1 ON 절로 답해지지 않는다" · 확립 조건은 R1 라벨이 `STICKY_REALIZES_A`(또는 `_ON_REMAINING_CELLS`+`cells_evaluated`)일 때로 한정 · ON 절은 E2C-3에 따른 양성 대조이며 R3(ii) 때문에 live 짝에서 E-cnt(ON) ≥ 95가 구성상 성립 · OFF 라벨의 두 오독 금지 · "ON 절 성립을 R1 라벨 이름이나 Q1의 답으로 바꿔 쓰지 않는다". 전부 들어갔다.
- L264("보고하는 규약이며, Q1의 판정은 rev3 §6 R1 라벨 그대로"), §6-7((A) 확정·(B)–(D) 기각), §2-6(3)·§6-8("미검증", E2VC-2), OVERRIDE §4: 일치.

**그 밖의 A13 diff = 0 (감사자 독립 실증).** 현재 A13(442줄, `82055fe1…`)에서 보고된 네 곳(L162, §4-1 인용 블록, L264, L369–370)을 1회차 원문으로 되돌려 해시하면 440줄, sha256 `a5e703fed3a43e6aaaff82eeee62a803adf464cf842691705ae9ffd3850ad1cc` = 1회차 봉인 원본과 바이트 동일. `e2_vessl_plan.A13_DIGEST`는 현재 바이트와 맞고, `test_e2_vessl.py`는 감사자 재실행에서 전부 OK.

**§4-1 (A)–(D) 선택지 표 잔존: 死因 아님, 수정 지시 없음.** (B)–(D)를 적용하려면 digest로 고정된 `e2_label.py`를 바꿔야 하고 그러면 `ABORT_REG_DRIFT`(exit 3)다. 실행 경로에서 판정을 바꿀 길이 0이다. E2VC-10으로 묶는다.

## (2) 수리 2: digest 출처 주석과 두 번째 출처 예외

- engine-porter의 발견을 확인했다(1회차 감사자 누락). `OVERRIDE_E2_SUBMIT_2026-09-15.md`는 b917bfc 추가·dff7bb7 최종 수정. `git show b917bfc:` sha `251f065e…`, `git show dff7bb7:` sha `0e5833e3…` = 등록값 ⇒ 등록 바이트는 dff7bb7판. 이식성 판정서는 40a7be2 추가·b330ae8 수정. 강제되는 것은 바이트이므로 판정 영향 0.
- `e2_vessl_plan.py` 주석(L40–49)은 정확하다. 수리 완료.
- A13 §1-1 표 둘째 행 "같은 커밋(…)"은 부정확하지만 **A13은 수정하지 않는다** — 정오표 caveat E2VC-9. 사유: 강제되는 것은 바이트 digest이고 값은 옳다 · 출처 정본은 helper 주석(교훈 80의 위험 위치는 수리됨) · A13 재수정은 digest 재봉인 비용.

## (3) 수리 3: reachability 경로 치환

- 코드(`design_reachability.py` `resolve_legacy()`, `presubmit.py` 로그 태그): `LEGACY_ROOT + "/"` 접두에만 작동 · 대상 파일 존재 시에만(`out`은 상위 디렉터리 존재 시) · 없으면 `PATH_REMAP_SKIPPED` 후 원경로 유지(시끄럽게 실패) · 치환마다 `PATH_REMAP` 로그 · spec 바이트 불변.
- **독립 재현**(spec 사본, `out`만 scratchpad): M4R → `SINGLE_LABEL_FORCED` · TC1 rev3-A → `RESTRICTIONS_INERT` · cp0_arm_reqactive → `DISCRIMINATING` · cp0_capacity → `DISCRIMINATING`. 넷 다 `PATH_REMAP rule_path` 발화. 2026-09-15 라벨과 같다. 재기록된 추적 판정 JSON 4개는 `git status`상 변경 없음.
- **반전 시도**: (a) 치환은 당시 파일이 아니라 현재 저장소 규칙 파일을 평가한다 — 이번 4건은 바이트 재현이라 반전 없음, 일반 경로로는 열려 있음 → E2VC-11. (b) 규칙 모듈 내부 절대경로는 치환하지 않고 시끄럽게 실패. (c) `?`→비차단 경로는 cp_baseline 2건이 2026-09-15와 같은 라벨로 복귀한 것뿐. **반증 실패, 死因 아님.**

## (4) 수리 5: 실제 경로의 kill 층 (E2VC-4)

`TestRealPathKillLayers`는 **사본이 아니라 실제 소스를 검증한다.** (a) DRYRUN 분기 이후 실제 텍스트에서 bench `timeout -k 30 "${BT}s" "${BENCH[@]}"`·`BT=$TCAP; KIND=P7_CAPPED` 확인, 제거 변이 실패 (b) watchdog `pkill -f` 패턴을 실제 `SRV=(`/`BENCH=(` 블록 명령 머리에 정규식 대조(TERM·9 모두), 오타 변이 2종 실패 — 추가 확인: watchdog는 `BUDGET_HARDCAP_FIRED`를 먼저 쓰고 `pkill`하므로 rc 143/124 모두 BUDGET으로 분류 (c) `.complete` 분류는 실제 러너 소스 조각을 stub과 실행하는 행동 테스트 5사례, BUDGET 가드 삭제 변이 실패 (d) preflight 6개·tail 3개 명령이 `timeout -k 10 "$(cap_s)"`/`"$(tail_s)"`로 감싸짐, 래퍼 제거 변이 실패.

**E2VC-4 대부분 해소.** 잔여: ① 종단 간 GPU 실행이 없어 실제 `pkill` 동작은 텍스트 대조뿐 ② 셀마다 도는 추정량 호출(`e2_realized_mix.py <tel>`, 러너 L502)은 timeout 없음·watchdog 패턴 밖 ③ job_entry clone·install과 반출에는 kill 층 없음.

**측정 경로 불변**: SRV/WARM/BENCH 배열 텍스트가 1회차와 같다. 플래그 다중집합·상수·노브 텍스트 대조 통과. warm-up은 1회차와 같은 `timeout -k 30 "$CAP"`.

## (5) 하네스층 봉인 (감사자 실측 sha256, engine-porter 보고와 8/8 일치)

| 파일 | sha256 |
|---|---|
| `workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh` | `921dfd2df4c532ee2e6e0452533ca5a0cb09fd541daca0dcd4cda927af00a6c4` |
| `…/e2_sticky_vessl/e2_vessl_plan.py` | `57220a46bc8e57d60f470d45accabdbdba5e99135435a0e19c06eb3c5354700f` |
| `workspace/engine-port/tests/test_e2_vessl.py` | `d163d7020cbe4d737904036b906d63be548529c452e84f924bca415b5a99207c` |
| `…/e2_sticky_vessl/PREREG_E2_VESSL_A13_2026-10-04.md` | `82055fe17f1d438a80b48d9412a21e4d7d300d5f2ce1ed15b284762247a719c0` |
| `…/e2_sticky_vessl/OVERRIDE_E2_VESSL_2026-10-04.md` | `54a810ca4c4f0d57445f88d489afe58a6713eb21bc6b6bc2fd9d63e7e5f9202d` (예외 — 아래) |
| `…/e2_sticky_vessl/stage_e2_inputs.sh` | `4b27bb0ab53149bf8fb9cd0179da71e5b86b4569fe849e79f7c5a034642d54cb` |
| `scripts/discipline/design_reachability.py` | `7eb2c190d0153489d1d6632e841d2626977a830c82594200e0dd0c4706abbc55` |
| `workspace/engine-port/scripts/discipline/presubmit.py` | `17821c4944d2d56d3804cf838ed61b0fb9092402e5af38bd01099f50d161093e` |

**봉인 규약**: 러너·helper·테스트·A13·staging 도구·규율 도구 2개는 이 바이트 그대로 커밋돼야 이 판정이 적용된다. 바뀌면 하네스층 국소 재확인을 다시 하고, A13이 바뀌면 규칙층도 다시 본다. **OVERRIDE만 예외** — 선행조건 3의 추가 기록(§1 추가 기록, §7 상태 칸, 사용자 재확인 기록)으로 sha가 바뀐다. 러너는 OVERRIDE를 기록만 하고 판정 경로에서 읽지 않는다. 커밋 sha의 OVERRIDE가 새 봉인이 되고, 재개 Job은 이 봉인과 같아야 한다(`ABORT_RESUME` digest 비교).

## 반전 시험 표 (이번 회차분)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| A13 수리 범위 | 보고된 4곳 ↔ 그 밖 | 없음 | 4곳 되돌린 재구성 sha = 원본 `a5e703fe…` |
| (A)–(D) 표 잔존 | (B)–(D) 채택 시도 | 없음(실행 불가) | `e2_label.py` digest 고정 → `ABORT_REG_DRIFT` |
| A13 §1-1 출처 문구 | "같은 커밋" ↔ 실제 출처 | 없음 | 14개 값 일치, OVERRIDE = dff7bb7판 `0e5833e3…` |
| 경로 치환 | 레거시 → 현재 규칙 바이트 | 이번 회차 없음 | 4 라벨 재현, 추적 판정 JSON 바이트 동일 |
| kill 층 | 래퍼·패턴·가드 제거 변이 | 테스트가 검출 | `TestRealPathKillLayers` 대조 변이 5종 |

## caveat (결과 문서·정본이 문자 그대로 승계)

1회차의 E2VC-1, E2VC-2, E2VC-5, E2VC-6, E2VC-7, E2VC-8 유지. 아래는 축소·추가분.

- **E2VC-3 (축소)**: "4.0 h는 job_entry 시작부터 잰 캡이다. 셀 구간은 T0+13,800 s에서, preflight는 같은 마감에서, tail은 T0+14,400 s에서 timeout으로 끊긴다. job_entry의 clone·install과 결과 반출, 셀마다 도는 추정량 호출에는 kill 층이 없다. '컨테이너가 4 h 안에 끝났다'는 `meta/timing.txt`의 `start_utc`/`end_utc` 실측 없이 쓰지 않는다."
- **E2VC-4 (축소)**: "실제 경로의 bench timeout, watchdog 패턴, BUDGET 시 `.complete` 미기록은 실제 러너 소스에 대한 구조·행동 테스트와 대조 변이로 검증됐다. 실제 GPU 프로세스에 대한 종단 간 kill 동작은 검증되지 않았다."
- **E2VC-9 (출처 정오표)**: "A13 §1-1 표 둘째 행의 '같은 커밋'은 OVERRIDE_E2_SUBMIT_2026-09-15(등록 바이트 = dff7bb7판)와 이식성 판정서(40a7be2 추가·b330ae8판)에 대해 거짓이다. 등록 바이트의 출처는 `e2_vessl_plan.py`의 `REGISTERED_DIGESTS` 주석을 따른다."
- **E2VC-10 (기각 대안)**: "A13 §4-1의 (A)–(D) 선택지 표 가운데 (B)·(C)·(D)는 감사자가 기각한 대안이다(파일럿 한 값을 본 뒤 고른 수라 채택 시 예보 라벨이 뒤집힌다). 등록 규칙의 일부가 아니며 결과 해석에 쓰지 않는다. R1 OFF 문턱은 20.0이다."
- **E2VC-11 (경로 치환)**: "`PATH_REMAP`을 거쳐 나온 reachability 판정은 spec 작성 시점이 아니라 현재 저장소 규칙 파일 바이트에 대한 판정이다. 2026-10-04 E2 제출 전 판정 4건은 추적된 판정 JSON과 바이트 동일하게 재현됐다. 이 사실을 다른 spec의 재현으로 일반화하지 않는다."

## 반증 실패 기록

A13 diff 범위(재구성 해시 일치) · 교체 문안 5곳과 E2VC-1·E2VC-2 취지 일치 · `A13_DIGEST` 일치 · 등록 digest 출처(OVERRIDE dff7bb7판 독립 확인) · reachability 4 라벨 독립 재현과 추적 JSON 무변경 · 새 테스트의 대조 변이 실재 · 측정 명령줄 불변 · 봉인 8/8 일치 · 테스트 스위트 감사자 재실행 OK.

## 남은 제출 선행조건 (순서대로, 한 창 규율)

1. **사용자 재확인(필수 수리 4)**: OVERRIDE §5-2 위험표(셀 구간 캡, 무상→과금, kill 층 없는 구간 E2VC-3)를 사용자에게 다시 보여 확인받고, "컨테이너 하드캡 ≈4 h"와의 차이 확인을 OVERRIDE §7에 기록한다. §7의 "권한다"는 **필수**로 읽는다.
2. **doc_facts 차단 해소(doc-steward)** 후 `presubmit.py` 재실행 — 차단이 정확히 M4R `SINGLE_LABEL_FORCED`·TC1 `RESTRICTIONS_INERT` 2건이고 두 줄에 `PATH_REMAP` 태그가 붙어 있어야 한다.
3. **OVERRIDE §1 추가 기록**: 수리 후 presubmit 출력을 문자 그대로 추가 기록. 수리는 spec 편집이 아니라 도구 수준 치환이었다는 점, doc_facts 수치를 현재 출력에 맞게.
4. **커밋**: 봉인 파일 전부, `design_reachability.py`, `presubmit.py`, 판정서 2개. 봉인 sha와 같은지 확인(OVERRIDE는 3번 반영본). CPU 회귀·`check_citation_stops.py` 0위반.
5. **`stage_e2_inputs.sh --upload`**: 스토리지만, GPU 0.
6. **`launch.sh` dry-run**: 사용자가 명령·단가·job name 승인 뒤 `--submit`.
7. 회수는 `fetch.sh` DONE + sha256까지. job name을 OVERRIDE §7에 기록하고 소진 처리.
8. 재개 Job은 OVERRIDE §5-1에 따라 **사용자 재승인 대상**.

## 실현가능성 자기검사 (게이트 #113)

이번 회차는 A13 추가 수정을 요구하지 않았다(표 잔존·출처 문구는 caveat). digest 연쇄 재봉인을 만들지 않았다. 요구한 것은 OVERRIDE 추가 기록·사용자 확인·doc-steward 수정뿐이며 전부 CPU, 판정 경로 밖이다. 이전 판정서의 지적을 상수로 승격하지 않고 원본 해시 재구성·`git show`·도구 사본 실행으로 독립 재검증했다(게이트 #110). 등급 자기검사: 死因을 만들지 못했으므로 `NO-GO`가 아니고, caveat가 있으므로 `GO`도 아니다.
