# 제출 전 체크리스트 — **실행 지점**

2026-08-28 신설 · GPU 0 · 새 성능 판정 0건

## 왜 이 파일이 있나

이 저장소는 규율 도구를 셋 갖고 있는데 **셋 다 등록된 실행 지점이 없었다.**

| 도구 | 신설 | 지적 |
|---|---|---|
| `check_line_citations.py` | 2026-08-26 | 감사 B11 — *"사전등록 제출 전 체크리스트에 아직 등재 안 됨"* |
| `check_doc_facts.py` | 2026-08-26 | 〃 |
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

## 현재 실행 결과 (2026-08-28)

```
OK     line_citations  50 compared, 0 violation(s)
OK     doc_facts       11 fact(s), 14 occurrence(s), 0 violation(s)
OK     reachability    reach_spec_B.json        -> DISCRIMINATING
BLOCK  reachability    reachability_spec.json   -> SINGLE_LABEL_FORCED     (M4R rev2)
BLOCK  reachability    reach_spec_A.json        -> NOTHING_PURCHASABLE     (TC1 rev2, argmax=d44)
exit=1  제출 금지
```

⇒ **두 재감사가 죽인 설계 둘이 이제 제출 게이트에서 기계적으로 막힌다.**

## 알려진 미정리 1건 (doc-steward 소관)

`design_reachability.py`가 **저장소 루트 `scripts/discipline/`**에 만들어졌는데, 기존 도구 둘은
**`workspace/engine-port/scripts/discipline/`**에 있다. 정본 문서들의 `scripts/discipline/…` 표기는
engine-port 트리 기준이다. ★**지금 옮기지 않았다** — 2026-08-28 시점에 **다른 세션이 그 경로로 이미
호출 중**이라(`results/cp_baseline/`) 임의 이동은 그쪽을 깬다. `presubmit.py`는 **두 위치를 다 찾는다.**
경로 정본화는 doc-steward가 양쪽 세션을 조율해 처리한다.

## 이 체크리스트가 보지 **않는** 것
사전등록의 死因 · estimand 식별성 · 대조의 항등식 여부 · 검정력.
**통과는 "규율 검사를 통과했다"일 뿐 "제출해도 된다"가 아니다.**
