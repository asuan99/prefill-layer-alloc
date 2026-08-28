# 설계층 도달가능성 — 두 트랙 모두 **캠페인 구매 불가** (GPU 0)

2026-08-28 · 도구 [`scripts/discipline/design_reachability.py`](../../../scripts/discipline/design_reachability.py) ·
**새 성능 판정 0건** · 정책 순위·HE0 불변

두 재감사(TC1 rev2 `NO-GO` F5–F7 · M4R rev2 `NO-GO` F1′–F4′)가 **공통으로 요구한 메타검사**를 구현해
두 규칙에 돌렸다. 규칙 파일의 `T2_reachable`은 *"라벨이 격자 어딘가에 나타나는가"* 를 묻는 **격자 내부 성질**이라
설계층 도달불가를 원리적으로 못 잡는다(TC1 rev1 B13 · M4R rev2 F3′). 이 도구는 반대를 묻는다 —
**등록된 설계와 이미 측정된 데이터가 실제로 낼 수 있는 실질 라벨은 무엇인가.**
모든 제약에 **출처 문자열을 강제**한다(출처 없는 제약은 거부) — 그렇지 않으면 증거가 아니라 가정으로 설계가 좁혀진다.

## 결과

| 트랙 | 격자 실질 라벨 | **설계가 낼 수 있는 것** | 판정 |
|---|---|---|---|
| **M4R rev2** | RESIDUAL_{PRESENT, ABSENT, INCONCLUSIVE} | **`RESIDUAL_INCONCLUSIVE` 하나** | ★`SINGLE_LABEL_FORCED` |
| **TC1 rev2 · 시나리오 A**<br>(argmax = d44) | FLIP · REVFLIP · BOTHWIN · BOTHLOSE · UNDERPWR · INCONC | **없음** | ★★`NOTHING_PURCHASABLE` |
| **TC1 rev2 · 시나리오 B**<br>(argmax = d34) | 〃 | FLIP · BOTHLOSE · UNDERPWR · INCONC | `DISCRIMINATING` |

## 읽기

**M4R rev2**: `sm_match`(39,849 구간 반례 0) · `iters`(Δ=16이 99.16%, 이 세션 직접 확인 1418/1430) ·
`cadence`(상수가 삭제됐는데 라벨은 남음) 세 가드가 **발화할 세계가 없고**, 7/7 셀의 블록 t-CI가 1을 포함한다.
⇒ **판정이 데이터 도착 전에 `RESIDUAL_INCONCLUSIVE`로 고정**돼 있다. 하네스를 돌려도 사는 것이 없다.

**TC1 rev2**: ★**캠페인 50 job의 정보량 전체가 Stage 1 argmax 추첨 하나에 걸려 있다.**
정본이 두 번(§1-7 3.220 최대 · §1-33 goodput argmax d44 3.7973) 지지하는 **시나리오 A에서는 실질 라벨이 하나도 없다** —
`visits_argmax = no` ⇒ `CONTROLLER_DEGENERATE`가 모든 세계를 삼킨다. 그리고 A와 B를 가르는
**d34↔d44 격차는 1.5% = 등록 문턱 δ의 절반**이다(§1-33: d44 3.7973 vs d64 3.7752, 정본 자신이 "헤드라인 아님"이라 등재).
⇒ **자기 문턱보다 작은 차이가 캠페인의 정보량을 결정한다.**

## 이 도구가 잡는 실패 형태 (신규 등재 후보)

> **도달가능성 검사가 격자 안에서만 돌면, 설계가 답을 미리 정해 놓아도 통과한다.**
> 규칙 파일의 `T2`류는 축 격자의 성질이고, 실험이 만들 수 있는 세계는 그 부분집합이다.
> 부분집합에서 실질 라벨이 0개거나 1개면 **결정량은 항등식이 아니어도 결정은 항등식**이다.

★게이트 #40(결정량 자체가 항등식일 수 있다)의 **설계층 판본**이며, 두 트랙에서 **각각 독립적으로** 발화했다.

## 처방 (양 트랙 공통, 전부 GPU 0)
1. 이 도구를 **사전등록 "제출 전 체크리스트"에 등재**한다. 현재 등록된 실행 지점이 없다
   (`check_line_citations.py`·`check_doc_facts.py`가 같은 상태로 남아 있는 것과 같은 결손 — 감사 B11 계열).
2. 규칙 파일마다 `reachability_spec.json`을 **사전등록 산출물로 요구**하고, `NOTHING_PURCHASABLE`/
   `SINGLE_LABEL_FORCED`이면 **제출 금지**로 못 박는다.
3. TC1: `visits_argmax`를 감사 처방대로 **시도·봉쇄 축**(`DEAD`/`BLOCKED`/`MISPOSITIONED`/`REACHES`)으로
   재정의하고, 뒤의 둘은 **차단이 아니라 관측**으로 둔다 — 그러면 시나리오 A에서도 실질 라벨이 산다.
4. M4R: `sm_match`는 이 기판에서 항등식이므로, **sticky ON 신규 측정** 없이는 어떤 판본도 이 벽을 못 넘는다.

## 금지 문장
*"도달가능성 검사를 통과했으므로 제출할 수 있다"*(다른 死因이 독립으로 남는다) ·
*"TC1은 argmax가 d34면 살아난다"*(시나리오 B는 나머지 死因 F6·F7을 닫지 않는다) ·
*"M4R은 GPU 0이다"*(★rev1 §8이 이미 금지했고 rev2가 위반했다 — 아래 정정 참조).
