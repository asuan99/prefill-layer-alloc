# 감사 판정서 — P0-A **H1–H7 수리** 보충 하네스층 감사 (2026-08-23, claims-auditor)

**판정: `CONFIRMED with conditions`** · GPU 지출 **0** · 대상 트리 수정 **0건** ·
새 성능 판정 **0건**.

> ★**작성 경위**: 감사자가 운영 규율상 파일을 쓰지 않고 전문을 반환했고, 메인 세션이 지정된
> 경로에 보존했다. 기존 두 판정서는 건드리지 않았다.

> **(ii) 890893의 판정을 바꿀 수 있는 새 결함 = 0건.** 직접 실증.
> **(i) 의도한 결함을 닫았는가 = 7건 중 6건 YES, ★H5는 NO** — H5는 *하지 않은 수리를 문서와
> 코드 주석이 했다고 적은* 형태다(**교훈 #67 재발**). 890893을 흔들지 않지만 결과 감사의
> **C7을 "미검사"에서 "수리가 실패했음이 실측됨"으로 격상**시킨다.

---

## 1. ★★ 890893 판정 영향 — **없음**(재채점 실증)

`git show 0ba9394`(G-수리분 = H-수리 **이전**) 하네스로 `p0a_raw_890893.json`을 재채점.
규칙 모듈은 두 판본에서 **sha256 동일**(`40709b24…3e43b`) ⇒ 변인은 하네스뿐.

| 비교 | 결과 |
|---|---|
| 배포판 `p0a_verdict_890893.json` vs 현행 코드 재채점 | ★**전 키 바이트 동일**(only-in 0/0, differing 0) — 결과 감사 재현 |
| 현행(H-수리) vs **H-수리 이전** 재채점 | `verdict`·`why`·`escape`·`delta`·`exact_set_match`·`null_channel`·`green_pair_*`·`min_hits_by_leg`·`delta_split_by_leg`·`cross_legs`·`graph_leg_failures` **전부 동일**. 차이는 **순수 가산 3건뿐**: `run_sweeps_per_plain_leg`(H6) · `adapter.escape_union_size`(H7) · `replicated_any` 키 개명(H6) |

★`run()`의 GPU 동작도 H 전후 동일(`SWEEPS_CROSS=1`은 이전 리터럴과 같은 값, `cross_spec.items()`
산출 쌍·순서가 `zip(...)`과 **원소 단위로 동일** — ★§3에서 이게 오히려 문제다).
⇒ ★**`CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`는 어떤 H-수리에도 의존하지 않는다.**

---

## 2. 일괄 치환 손상 전수 — **자백한 1건 외 0건**

| 검사 | 결과 |
|---|---|
| 함수 단위 AST 비교(preH↔postH) | 변경 **6개** + 신규 함수 1 + 신규 상수 1. **부수 변경 0건** ⇒ H2 삽입이 다른 함수를 안 건드렸다 |
| `world_from_raw` 제어흐름 | 양 판본 **`return` 8개, 중첩 구조 동일**. 네 도구실패 경로 전부 `_attach_failure_accounting`가 `if` **안**에서 `return` 바로 앞 형제 |
| ★**자기 수리의 변이 테스트**(교훈 #53) | 자백한 dedent를 **재주입** → `--selftest` **rc=1**, `A3a`·`A4`·`A5`(42세계 중단)·`A6` 6행 FAIL ⇒ 파손 상태와 현 상태가 **기계적으로 구분된다** |
| 오형 원자료 퍼즈 **14종** | preH↔postH **판정 차이 0건 · 신규 예외 0건** ⇒ `_attach_failure_accounting`는 예외 안전 |

---

## 3. H1–H7 개별 판정

**H1(문서) — 닫힘. 자기 선언도 참**: verdict 기록처가 `:774`/`:778` **두 곳뿐**,
게이트 4개 전부 `ABSENT` 한 방향(6 픽스처 실측), §5.3·§10.2에 실제 등재.

**H2 — 닫힘. 서술 전용 실증**: 6 픽스처에서 **판정 6/6 동일**(healthy PRESERVED, 나머지 ABSENT),
차이는 `adapter` 신규 키뿐. 잔여: sweep 수·meta 불일치 2행엔 `tool_status_by_leg`가 없다.

**H3(문서) — 닫힘**: `status()`가 `cap == expected`일 때만 `ok` ⇒ 25 + 2×25 = **75**.
경미한 과소진술(코드는 replay도 75/75 요구, 문서는 캡처만) — **안전 방향**.

**H4 — 닫힘. 일곱 중 가장 값진 수리**

| 변이 | A7c | A7d | A7e | `--selftest` |
|---|---|---|---|---|
| 배포 상태 | PASS | PASS | PASS | **ALL PASS rc=0** |
| ★`_ = CROSS_LEGS` + 리터럴 복귀(구 A7c 통과 형태) | **FAIL** | PASS | **FAIL** | **rc=1** |
| sweep 수를 위치 숫자로 | PASS | **FAIL** | PASS | — |
| `graph_green`에 `2` 하드와이어 | PASS | **FAIL** | PASS | — |

비공허성: A7d가 보는 leg 호출 **6개**, A7c가 훑는 문자열 상수 **60개** ⇒ 항등식 아님.

**H5 — ★★닫히지 않았다. 문서·주석이 하지 않은 수리를 적었다**
`run():478-485` 주석은 *"`zip`이라 재정렬하면 두 서술 레그가 조용히 뒤바뀐다 ⇒ 상수로 키를
잡아 다시 묶는다"* 고 적지만, 채택된 형태가 **위치 인덱스 키**라 `.items()`가 `zip`과
**원소·순서까지 동일**하다. 실측(`CROSS_LEGS` 원소 뒤집기):
```
as shipped           : capture_green_replay_plain -> capture=GREEN(34) replay=PLAIN(108)
CROSS_LEGS reordered : capture_plain_replay_green -> capture=GREEN(34) replay=PLAIN(108)  ← 의미 역전
```
**A7c/A7d/A7e 전부 PASS.** ⇒ H5가 막겠다고 선언한 사고가 **그대로 재현된다.**
★그리고 `PREREG_P0A_2026-08-22.md:35-40`이 *"H1–H7은 전부 비차단이며 rev8이 **전부 닫았다**"* 라
쓰고 H5를 *"dict로 복원 + A7e"* 로 열거한다 — **H5에 대해 거짓**. **교훈 #67 재발**.

**H6 — 닫힘**(3건): `sweeps_per_plain_leg` 편입 확인 · `absent` `why` 분기 실측 ·
`replicated_any` 개명 안전(이 키를 이름으로 읽는 코드 **0건**, grep 전수).

**H7 — 닫힘**(합성 세계): 잡음 1+1 서로소 → `LOST (UNREPLICATED)`, `escape_union_size` **2** /
21+21 서로소 → 동일 라벨, **42** / 같은 라벨 두 번 → `LOST (PARTIAL)`, **1**.
판정 pre=post ⇒ **문턱 아님**.

---

## 4. AST 가드(§10-7) — 통과하고 **신규 코드에서도 물린다**

도달 함수 **15 → 16**, `_attach_failure_accounting`이 **폐포 안**(공허 아님).
배포 파일 `target_numbers=[] driver_keys=[]`. ★신규 코드 3곳 누출 심기 **전부 탐지**
(`total_sm_reported` · 드라이버 `rc` · `nsmid_observed`).
`--selftest` **ALL PASS rc=0**(PASS 81행, A5 **516,096 세계**, 변이 13×3, A7a–e 전부) ·
`--selftest-mutants` **ALL PASS rc=0**.

---

## 5. ★ 신규 지적 (890893 판정 영향 **0**)

| # | 내용 | 등급 |
|---|---|---|
| **F1** | **H5가 닫지 않았다**. 주석과 prereg rev8 헤더가 **하지 않은 수리를 적는다** | ★중대(문서·계약) |
| **F2** | **A7e가 H5를 지키지 못한다**. `zip` 완전 원복 + `_decoy = {CROSS_LEGS[0]: None, CROSS_LEGS[1]: None}` 한 줄 → `--selftest` **ALL PASS rc=0**(실측). A7e는 `run()` 안 **모든** `ast.Dict`에 `max()`를 걸 뿐 **루프 구동 dict인지 안 본다**. **중복 키**도 센다 | ★중대(검사 약함) |
| **F3** | **(capture, replay) 순서 무검사가 end-to-end 확인됨** — `mut_swap` `--selftest` **ALL PASS rc=0**. 스왑이 살아 있었으면 34/108이 **정확히 뒤바뀌어** 기록되고 아티팩트에 표시가 없다. 오늘 배선이 옳음은 **코드로** 확인 | ★결과감사 **C7 격상** |
| **F4** | ★**§10.2 계약표가 G1 이전 술어를 말한다**(`:705` *"두 sweep 모두 비지 않을 때만 참(§8-25)"*). **§8-25·§5·코드는 교집합** ⇒ **표가 자기 근거로 인용한 절과 반대**. G-era 잔여, 후속검증·H 라운드 둘 다 놓침 | 중 |
| **F5** | `graph_leg_failures`가 **두 깊이**에 존재(정상 경로 최상위 / ABSENT 경로 `adapter` 안) — 한쪽만 읽는 소비자는 조용히 잃는다 | 경 |
| **F6** | A7d **키워드 인자 사각**(`sweeps=1` → 마지막 위치 인자가 `dev`라 PASS). 현 6 호출 전부 위치라 미발화 | 경 |
| **F7** | `escape_union_size`가 prereg에 **필드명으로는 없다** | 경 |

★공정성: `mut_dupkey`·`mut_kw`는 `--selftest` 전체를 10분 안에 못 끝내 `_run_wiring()` 직접
평가로 근거를 세웠다(스위트가 보는 값과 동일하나 end-to-end 아님).
`mut_swap`·`mut_decoy`·`mut_lit`은 **end-to-end 확인**.

---

## 6. 조건 (결과 감사 C1–C8에 **추가**)

* **C9** — *"H1–H7이 전부 닫혔다"* 라고 쓰지 마라. **H5는 닫히지 않았다**(F1).
* **C10** — 34/108 인용 시 *"레그↔스트림 순서는 **사람이 읽어** 확인했고 기계 검사는 없다"* 병기.
* **C11** — `§10.2` 행을 교집합 술어로 정정하기 전에는 그 표를 **구현 근거로 쓰지 마라**(F4).
* **C12** — `graph_leg_failures`는 **경로 의존 위치**다. 인용은 경로를 밝혀서(F5).
* 승계: 결과 감사 **C1–C8 전부 유효**, 금지 문장 전부 유지, **HE0 불변 · 정책 순위 0건**.
* ★**신설 금지 문장**: *"H-수리 감사가 하네스를 통과시켰다"* — 이 절이 말하는 것은
  **"890893의 판정이 H-수리에 의존하지 않음을 재채점으로 실증했고, H5 하나가 닫히지 않았음을
  찾았다"** 뿐이다.

---

## 7. 이를 확정할 실험 (전부 **GPU 0**)

| # | 목적 | 설계 | 통과 기준 |
|---|---|---|---|
| **E1** | F1·F3 **실제 폐쇄** | `_leg_graph`가 호출자에게서 받은 **역할 라벨**을 레그 dict에 쓴다(`capture_stream_role`/`replay_stream_role`). 어댑터가 `cross_legs[…]["capture_stream_role"] == "green_decode"` assert | `mut_swap`에서 **반드시 FAIL**, intact PASS. ★AST가 아니라 **런타임 값**을 묶는 것이 요점 |
| **E2** | F2 폐쇄 | A7e를 *"루프의 `For.iter`가 `<Name>.items()`이고 그 `<Name>`의 Assign이 dict이며 키 **인덱스 집합**이 `range(len(CROSS_LEGS))`와 같다"* 로 교체 | `mut_decoy`·`mut_dupkey` **FAIL**, intact PASS |
| **E3** | F6 폐쇄 + 비공허 고정 | A7d에 숫자 **키워드** `sweeps=` 거부 + *"매치된 leg 호출이 정확히 6개"* arity assert | `mut_kw` **FAIL**, 호출 삭제 시 arity **FAIL** |
| **E4** | F4·F7 | `§10.2` 술어 정정 · §5/§11에 `escape_union_size` 명기 · rev8 헤더 H5 문구 정정 | 문서 |
| — | **판정의 두 잔여는 불변** | 결과감사 **F1**(pytorch v2.9.1 `CUDAGraph::replay()` launch 스트림 한 줄, GPU 0) · **F2**(캡처만/replay 없음 대조, ≈0.02 GPU-hr) | ★이 둘만이 890893이 *말할 수 있는 것*을 넓힌다 |

★E1–E4 어느 것도 **890893 재실행 사유가 아니다**(§1). 다음 GPU 지출은 결과감사 F2뿐.

**GPU 0 · job 제출 0 · 대상 수정 0 · 새 성능 판정 0 · 등급 변경 0 · 정책 순위 0 · HE0 불변 ·
gate #13/#16·switch-cost "닫았다" 금지 전부 유지.**
