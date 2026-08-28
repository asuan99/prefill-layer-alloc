# 제출 전 체크리스트 — **실행 지점**

2026-08-28 신설 · GPU 0 · 새 성능 판정 0건

## 왜 이 파일이 있나

이 저장소는 규율 도구를 셋 갖고 있는데 **셋 다 등록된 실행 지점이 없었다.**

| 도구 | 신설 | 지적 |
|---|---|---|
| `check_line_citations.py` | 2026-08-26 | 감사 B11 — *"사전등록 제출 전 체크리스트에 아직 등재 안 됨"* |
| `check_doc_facts.py` | 2026-08-26 | 〃 |
| `check_citation_stops.py` | 2026-08-18 | 게이트 #47 — 인용금지가 정본 등재만으로 전파되지 않는다 |
| `design_reachability.py` | 2026-08-28 | TC1 rev2·M4R rev2 두 재감사가 같은 지적을 반복 |

**도구가 있는데 아무도 안 부르면 도구가 없는 것과 같다.** `presubmit.py`가 그 실행 지점이다.

```bash
cd workspace/engine-port/scripts/discipline
python3 presubmit.py --registry presubmit_registry.json ; echo $?
#  0 = 제출 가능   1 = 제출 금지
```

## 차단 조건
- line-citation 드리프트 · doc-fact 위반
- **도달가능성이 `NOTHING_PURCHASABLE` 또는 `SINGLE_LABEL_FORCED`**

## 설계 원칙 두 개

1. ★**디렉터리를 쓸어 담지 않는다.** `presubmit_registry.json`에 적힌 항목만 검사한다.
   같은 저장소에서 **다른 세션이 동시에 작업할 수 있고**(2026-08-28 실제 발생),
   그쪽의 진행 중 파일을 이 검사가 실패시키면 안 된다.
2. ★**미실행을 통과로 세지 않는다.** `check_line_citations.py`를 문서 인자 없이 부르면
   *"no documents given"*으로 끝나는데 그건 **통과가 아니라 미실행**이다 —
   `--all`(매니페스트 전량)을 강제한다. 게이트 #21(측정 실패를 게이트 실패로 라벨링 마라)의 거울상.

## 결과 블록을 손으로 적지 않는다

★rev1의 이 자리에는 실행 결과를 **손으로 베껴** 두었고, 레지스트리가 바뀌자 **곧바로 stale**이 됐다
(더 이상 등록돼 있지 않은 spec 2건을 BLOCK으로 보여주고 있었다 — TC1 rev3 감사 지적, 교훈 항목81 재발 지점).
현재 상태는 **도구를 돌려서** 본다:

```bash
cd workspace/engine-port/scripts/discipline
python3 presubmit.py --registry presubmit_registry.json
```

## 설계 원칙 셋째 — ★레지스트리는 append-only

TC1 rev3 감사가 이 파일의 구멍을 **실증**했다. *"레지스트리에서 빼면 통과한다"* 는 가설이 아니라
**이미 일어난 일**이었다 — 커밋 `d11243a`가 TC1을 `NOTHING_PURCHASABLE`로 막던
`reach_spec_A.json`을 **rev3 spec을 추가한 같은 커밋에서 삭제**했다.
**판정 대상이 자기 판정 범위를 정할 수 있으면 게이트가 아니다.**

⇒ spec을 빼려면 `{"spec": …, "status": "superseded", "superseded_by": …}`로 **표시**한다.
통째 삭제는 `presubmit.py`가 `HEAD` 판본과 대조해 **차단**한다(`registry_append_only`).

## 도구 자신도 감사받았다 — `design_reachability.py` TOOL_REV 2

rev1은 **자기 요구자를 만족시키지 못했다**: TC1 rev3 spec의 제약이 `design_sub = grid_sub/3`으로
**균일 재척도**만 했는데 `unreachable_by_design: []`을 *"잃은 게 없다"* 로 읽어 `DISCRIMINATING`을 냈다.
rev2가 그것을 잡는다 — 그 spec은 이제 **`RESTRICTIONS_INERT`로 차단**된다.

신설 차단 코드 셋: `RESTRICTIONS_INERT` · `RESTRICTION_DROPPED_UNEXPLAINED`(선행 spec이 제약한 축을
사유 없이 뺐다) · `PRIOR_UNREGISTERED`(라벨별 사전확률 미등록).

★**도구의 알려진 한계를 문서에 박았다**: 이 도구는 **범주 격자만** 본다. rev2의 강제는 마침
범주적(`visits=no`)이라 잡혔고, **rev3의 강제는 연속층**(se·문턱·TOST 마진·사전분포)에 있어 **못 잡았다**.
`DISCRIMINATING`은 *"범주가 금지하지 않는다"* 일 뿐 *"판정을 낼 수 있다"* 가 아니다.

## 알려진 미정리 1건 (doc-steward 소관)

`design_reachability.py`가 **저장소 루트 `scripts/discipline/`**에 만들어졌는데, 기존 도구 둘은
**`workspace/engine-port/scripts/discipline/`**에 있다. 정본 문서들의 `scripts/discipline/…` 표기는
engine-port 트리 기준이다. ★**지금 옮기지 않았다** — 2026-08-28 시점에 **다른 세션이 그 경로로 이미
호출 중**이라(`results/cp_baseline/`) 임의 이동은 그쪽을 깬다. `presubmit.py`는 **두 위치를 다 찾는다.**
경로 정본화는 doc-steward가 양쪽 세션을 조율해 처리한다.

## 이 체크리스트가 보지 **않는** 것
사전등록의 死因 · estimand 식별성 · 대조의 항등식 여부 · 검정력.
**통과는 "규율 검사를 통과했다"일 뿐 "제출해도 된다"가 아니다.**
