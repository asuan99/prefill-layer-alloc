---
name: promote
description: main 세션 전용. 트랙 파일(`tracks/<트랙>.md`, 이하 TRACK.md) 중 "promote 필요: 예"인 항목(결과 분석·사용자 결정·감사 판정)을 COSTMODEL.md·PROJECT_STATUS.md·CONSENSUS.md(§6 claim 등급 포함, 필요 시 MEMORY.md)에 압축 반영하고 축 지도를 갱신한다. "/promote", "정본 반영", "트랙 결과 올려" 할 때 사용. 정본·메모리의 유일한 작성 경로이므로 다른 세션이 promote 중이 아닐 때 한 번에 하나만 실행.
---

# promote — 트랙 → 정본 반영 (단일 작성자, 2026-10-13 신설 · 2026-10-19 개정)

2026-10-19 개정 이유: 결과 분석만 올라가고 사용자 결정·감사 판정은 올라갈 길이 없어 PROJECT_STATUS가 늦어졌고
(10-15 Q1·2회차 감사), 위계 2순위였던 `reports/paper/`는 반영 대상에 없어 claim 표가 10-04에 멈췄다.
그래서 수집 대상에 결정·판정을 넣고, claim 등급 정본을 CONSENSUS §6으로 옮겼다(`reports/paper/`는 동결 이력).

## 1. 수집
```bash
REPO=~/Experiments/KISTI/prefill-layer-alloc
grep -n 'promote 필요: 예' $REPO/tracks/*.md $REPO/../prefill-layer-alloc-*/tracks/*.md
/usr/bin/bash $REPO/tools/claude/inbox.sh     # 지연 경보(⚠) 줄도 함께 본다
```
표시는 세 종류다: `예`(결과 분석) · `예 (결정)`(사용자 결정) · `예 (감사)`(감사 판정). 그 항목과 링크된 결과 문서·
결정 원문·감사 판정서만 읽는다. 지연 경보가 뜬 트랙은 트랙 파일이 낡았을 수 있으니 결과 디렉터리 최신 커밋을 직접 확인한다.

## 2. 게이트
- 결과 주장은 claims-auditor 판정이 있어야 반영한다. 없으면 반영하지 않고 TRACK.md에 "감사 필요"로 되돌린다.
- 반영 문장은 판정서의 허용 문안을 그대로 쓰고, 금지 서술 목록은 링크로만 승계한다.
- 사용자 결정은 결정 원문 파일(예: `DECISIONS_*.md`)의 문장을 그대로 옮긴다. 결정은 판정이 아니므로 claim 등급을 바꾸지 않는다.

## 3. 반영 (doc-steward 위임)
- **COSTMODEL.md**: 해당 항의 **현재 형태를 덮어쓴다**(항당 6줄 이하) + "진척 요약" 표 상태 갱신 +
  "아직 답하지 못한 것" 목록 갱신. 이것이 사용자에게 보이는 진척판이다.
- **PROJECT_STATUS.md**: 원장이다 — 새 항목은 최상단 "최종 갱신" 아래 **5줄 이하**로만 추가하고 "최종 갱신" 줄 날짜를
  바꾼다. 결정·감사 항목도 여기에 한 줄씩 들어간다. 코드·테스트가 줄 번호와 게이트 번호로 인용하므로 기존 줄을 재배열·삭제하지 않는다.
- **CONSENSUS.md**: §1은 새 확정 결론일 때만 한 행. **claim 등급이 바뀌면 §6 표의 해당 칸만** 고친다. 방법론 교훈은
  기존 항목에 병합할 수 없을 때만 새 번호. 기존 줄 번호를 밀지 않도록 새 절은 파일 끝에 둔다.
- `reports/paper/`는 동결 이력이므로 쓰지 않는다.
- **MEMORY.md**: 트랙당 한 줄(150자 이하). 상세는 topic 파일.
- **축 지도** `reports/anchor/axes_map.html`: 바뀐 내용에 해당하는 데이터 블록 항목(축 카드의 확정·열린 것, 트랙 카드,
  claim 표, §0 현재 상태 문단)을 고치고 같은 URL로 재발행한다(README의 갱신 규칙). 새 판정은 넣지 않는다.
- 반영한 TRACK.md 항목은 `promote 필요: 아니오`로 바꿔 그 worktree에 커밋한다.

## 4. 커밋 · 출력
citation stop 0위반 → main 커밋. 사용자에게는 **"COSTMODEL에서 바뀐 칸"** 목록, PROJECT_STATUS에 올린 결정·감사 줄 수,
축 지도 재발행 여부, INBOX(inbox.sh) 변화만 보고한다.
