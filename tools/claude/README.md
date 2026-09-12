# `tools/claude/` — Claude Code 툴 정의의 버전관리 정본

## 왜 있나 (게이트 #157)

이 프로젝트의 감사·규율 규칙은 **에이전트/스킬 파일 본문**에 들어 있고, 정본 문서가 그
파일을 **SHA-256으로 앵커**한다(예: `PROJECT_STATUS.md`의 게이트 #157이
`.claude/agents/claims-auditor.md` 해시를 적는다). 그런데 그 파일들이 사는
`/scratch/ehmoon/whlee`는 **git 저장소가 아니다**(`.git`이 빈 디렉터리). 따라서
2026-09-12까지 툴 파일에는 **이력이 없었다** — 규칙이 diff 없이 바뀔 수 있었고, provenance는
사람이 손으로 붙여 넣은 해시 한 줄뿐이었다.

Claude Code는 **세션을 시작한 워크스페이스 루트**에서 에이전트·스킬을 찾으므로 live 파일을
repo 안으로 옮겨버릴 수는 없다. 그래서 **추적 사본을 이 repo가 갖고**, 스크립트가 두 위치
사이에서 바이트를 옮기며 일치를 증명한다.

## 무엇이 정본인가

| | 경로 | 역할 |
|---|---|---|
| 추적 사본 | `prefill-layer-alloc/tools/claude/` | **버전관리 정본**. 리뷰·이력·해시가 여기 붙는다. |
| live 사본 | `/scratch/ehmoon/whlee/{CLAUDE.md,.claude/{agents,skills}}` | Claude Code가 실제로 읽는 것 |

`workspace/CLAUDE.workspace.md` = live의 루트 `CLAUDE.md`. 이름을 바꿔 둔 이유는 이
하위 디렉터리가 **또 하나의 프로젝트 지침으로 자동 로드되지 않게** 하려는 것이다.

## 사용법

```bash
tools/claude/sync_claude_tools.sh            # --check (기본): 드리프트 있으면 exit 3
tools/claude/sync_claude_tools.sh --import    # live → 추적 사본  (제자리 편집을 담아 커밋할 때)
tools/claude/sync_claude_tools.sh --install   # 추적 사본 → live  (새 클론·다른 머신 복원)
tools/claude/sync_claude_tools.sh --manifest  # 추적 사본 SHA-256 재생성
tools/claude/test_sync_claude_tools.sh        # 드리프트 검사 자신의 변이 테스트 10케이스
```

`claude_tools.manifest.sha256` = 추적 사본 11개 파일의 해시. 정본 문서가 툴 파일을 인용할 때
이 파일의 해시를 쓰면 손으로 붙여 넣은 해시 한 줄보다 낫다(커밋 SHA로 같이 고정된다).

## 정직한 한계

- `--import` **직후의** `--check`는 항등식이라 아무 정보가 없다. 의미 있는 것은 **그 뒤에
  생긴** 드리프트뿐이다(de-confound 교훈 9).
- 검사는 해시뿐 아니라 **파일 집합**도 비교한다. 삭제·신규 추가는 알려진 경로의 해시 목록으로는
  안 보이기 때문이다(교훈 89).
- 이 장치는 live 파일을 **불변으로 만들지 않는다**. 변경을 **보이게** 만든다. 세션이 live를
  고치고 `--import`를 잊으면 드리프트는 다음 `--check`까지 남는다 — `catch-up` 또는 커밋
  시점에 돌리는 것이 규율이다(`.claude/agents/git-committer.md` "저장소는 하나다" 절).
- Claude Code가 심볼릭 링크로 이 디렉터리를 읽게 하는 방식(사본 0개)은 채택하지 않았다:
  에이전트·스킬 **디스커버리가 링크를 따라가는지**를 현 세션에서 검증할 수 없고, 실패하면
  다음 세션에서 에이전트 7개가 조용히 사라진다. 사본 + 기계적 드리프트 검사가 그보다 안전하다.
