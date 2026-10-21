---
name: catch-up
description: 세션 시작 시(또는 컨텍스트 유실 시) 한 트랙의 현재 상태를 짧게 복원한다. "/catch-up <트랙>"(main|amd|opchar|power-cap), 인자가 없으면 트랙 목록과 사용자 결정 대기만 보여 주고 어느 트랙인지 묻는다. "어디까지 했지", "이어서 하자", 세션 처음 열 때 사용. 확정 결론을 재도출하지 않는다. handoff 스킬의 짝(handoff가 TRACK.md에 저장 → catch-up이 복원).
---

# catch-up — 트랙 단위 오리엔테이션 (2026-10-13 개정)

원칙: **읽는 양을 고정한다.** 병행 세션이 서로의 문맥을 끌어오지 않도록 **한 트랙만** 복원한다.
`PROJECT_STATUS.md`(1.4만 줄 원장)·`CONSENSUS.md`·감사 판정서는 통째로 읽지 않는다.

## 1. 트랙 결정

```bash
ROOT=~/Experiments/KISTI; REPO=$ROOT/prefill-layer-alloc
git -C $REPO worktree list
/usr/bin/bash $REPO/tools/claude/inbox.sh       # 전 트랙의 "사용자 결정 대기"만 모은 뷰
```

| 트랙 | 트랙 파일 (자기 worktree의 `tracks/<트랙>.md`) |
|---|---|
| main (anchor 사다리·N2) | `$REPO/tracks/main.md` |
| amd | `$ROOT/prefill-layer-alloc-amd/tracks/amd.md` |
| opchar | `$ROOT/prefill-layer-alloc-opchar/tracks/opchar.md` |
| power-cap (worktree 없음) | `$REPO/tracks/power-cap.md` |

트랙 파일은 각 branch에 커밋되므로 병합해도 경로가 겹치지 않는다. 아래에서 "TRACK.md"는 이 트랙 파일을 뜻한다.
새 트랙이 생기면 이 표에 한 줄 추가하고 그 worktree에 `tracks/<트랙>.md`를 만든다(`inbox.sh`는 자동 탐색).
인자가 없으면: inbox.sh 출력 + 각 TRACK.md 첫 3줄만 보여 주고 "어느 트랙을 이어갈까요?"로 끝낸다.

## 2. 읽기 (이것만)

1. 그 트랙의 `TRACK.md` 전부
2. `$REPO/COSTMODEL.md`의 "진척 요약" 표 + 그 트랙이 기여하는 항의 절
3. 현재 결론은 이미 컨텍스트에 있는 CLAUDE.md "현재 결론"으로 충분하다
4. 그 worktree의 `git log --oneline -8`, `git status --short`
5. `vesslctl job list` — 그 트랙 캠페인 이름의 Job만 보고, 켜진 Workspace는 트랙 무관 경고
6. 근거가 더 필요할 때만 TRACK.md가 링크한 파일을 연다

## 3. 브리핑 (15줄 이하)

- 트랙 목표 + 기여하는 cost model 항 (1줄)
- 현재 상태 (≤3줄)
- 다음 행동 (≤3줄)
- 이 트랙의 사용자 결정 대기 (있으면)
- 미커밋·실행 중·미회수 Job (있으면)
- 다른 트랙 결정 대기 수 (1줄)
- 끝에 "이어서 뭘 할까요?" + 다음 행동 후보 2–3개

## 4. 규율

- 확정 결론을 재증명하지 않는다. 사용자가 도전하면 그때 claims-auditor.
- **다른 트랙의 파일은 읽기만 한다.** 이 세션이 쓰는 곳은 자기 worktree뿐이다.
- TRACK.md와 정본이 모순되면 정본이 우선 — 브리핑에 한 줄로 알리고 main의 `promote` 때 정리한다.
