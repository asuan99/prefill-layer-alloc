---
name: promote
description: main 세션 전용. 트랙 파일(`tracks/<트랙>.md`, 이하 TRACK.md) 중 "promote 필요: 예"인 항목을 COSTMODEL.md·PROJECT_STATUS.md(필요 시 CONSENSUS·MEMORY.md)에 압축 반영한다. "/promote", "정본 반영", "트랙 결과 올려" 할 때 사용. 정본·메모리의 유일한 작성 경로이므로 다른 세션이 promote 중이 아닐 때 한 번에 하나만 실행.
---

# promote — 트랙 → 정본 반영 (단일 작성자, 2026-10-13 신설)

## 1. 수집
```bash
REPO=~/Experiments/KISTI/prefill-layer-alloc
grep -l 'promote 필요: 예' $REPO/tracks/*.md $REPO/../prefill-layer-alloc-*/tracks/*.md
```
그 항목과 링크된 결과 문서·감사 판정서만 읽는다.

## 2. 게이트
- 결과 주장은 claims-auditor 판정이 있어야 반영한다. 없으면 반영하지 않고 TRACK.md에 "감사 필요"로 되돌린다.
- 반영 문장은 판정서의 허용 문안을 그대로 쓰고, 금지 서술 목록은 링크로만 승계한다.

## 3. 반영 (doc-steward 위임)
- **COSTMODEL.md**: 해당 항의 **현재 형태를 덮어쓴다**(항당 6줄 이하) + "진척 요약" 표 상태 갱신 +
  "아직 답하지 못한 것" 목록 갱신. 이것이 사용자에게 보이는 진척판이다.
- **PROJECT_STATUS.md**: 원장이다 — 새 항목은 최상단 "최종 갱신" 아래 **5줄 이하**로만 추가한다. 이 파일은
  코드·테스트가 줄 번호와 게이트 번호로 인용하므로 기존 줄을 재배열·삭제하지 않는다.
- **CONSENSUS.md** §1: 새 확정 결론일 때만 한 행. 방법론 교훈은 기존 항목에 병합할 수 없을 때만 새 번호.
- **MEMORY.md**: 트랙당 한 줄(150자 이하). 상세는 topic 파일.
- 반영한 TRACK.md 항목은 `promote 필요: 아니오`로 바꿔 그 worktree에 커밋한다.

## 4. 커밋 · 출력
citation stop 0위반 → main 커밋. 사용자에게는 **"COSTMODEL에서 바뀐 칸"** 목록과 INBOX(inbox.sh) 변화만 보고한다.
