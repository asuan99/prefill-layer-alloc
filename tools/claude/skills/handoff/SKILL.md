---
name: handoff
description: 세션을 마무리하거나 체크포인트를 남길 때 이 세션 트랙의 TRACK.md를 갱신하고 그 worktree에 커밋한다. "정리하고 마무리", "핸드오프 남겨", "저장해두자", "체크포인트" 할 때 사용. 정본(PROJECT_STATUS·CONSENSUS·COSTMODEL)과 MEMORY.md는 건드리지 않는다 — 그건 main 세션의 promote 몫. catch-up의 짝.
---

# handoff — 트랙 체크포인트 (2026-10-13 개정)

원칙: **트랙 세션은 자기 TRACK.md와 자기 결과 디렉터리에만 쓴다.** 공유 파일(정본·MEMORY.md·
날짜별 handoff 문서)에 쓰지 않으므로 병행 세션끼리 충돌하지 않는다. (2026-10-13 이전에는 네 세션이
한 날짜 파일 `session_handoff_2026-10-13.md`에 덮어쓰고 복원하는 일이 있었다.)

## 1. 델타 수집

```bash
W=<이 트랙 worktree>   # catch-up 표 참조
git -C $W log --oneline -15; git -C $W status --short; git -C $W diff --stat
```
대화에서 실제로 도달한 결정·측정·변경·열린 항목을 정리한다.

## 2. 트랙 파일(`<worktree>/tracks/<트랙>.md`, 이하 TRACK.md) 갱신 (60줄 예산 — 양식은 `tools/claude/TRACK_TEMPLATE.md`)

- "현재 상태" "다음 행동" "사용자 결정 대기" "실행 중·미회수 Job"은 **덮어쓴다**(append 아님).
- "최근 로그"에 한 줄 추가, 5개를 넘으면 오래된 것부터 지운다.
- 이번 세션에 결과 분석이 있었으면 그 분석의 **"cost model 기여" 절**을 "cost model 기여" 절에 한두 줄로
  반영하고 `promote 필요: 예`로 표시한다.
- 사용자 결정 대기 항목은 3줄 형식: **무엇 / 왜 지금 / 선택지 + 권고**. 처리된 결정은 지운다.
- 60줄을 넘으면 서술을 줄인다(이력은 git, 세부는 결과 디렉터리 문서로 링크).
- GPU Job 기록(job name·커밋·상태·fetch 여부)은 캠페인 결과 디렉터리의 원장에 두고 TRACK.md엔 미회수·실행 중만.

## 3. 커밋 (묻지 않는다, push 금지)

그 worktree에서 `python3 workspace/engine-port/scripts/discipline/check_citation_stops.py` 0위반 확인 →
TRACK.md + 이번 세션 산출물 커밋. 루트 `.out`/`.err` 제외. push는 하지 않고 묻지도 보고하지도 않는다.

## 4. 출력 (5줄)

커밋 해시 · 다음 시작점 한 줄 · 사용자 결정 대기 수 · promote 필요 여부 · 실행 중 Job.

## 하지 않는 것

날짜별 `handoff-report/session_handoff_*.md` 생성(폐지 — 옛 파일은 이력으로 남김) ·
PROJECT_STATUS/CONSENSUS/COSTMODEL/MEMORY.md 수정 · doc-steward 호출.
