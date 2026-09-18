---
name: catch-up
description: 새 세션 시작 시(또는 컨텍스트 유실 시) 이 프로젝트(prefill-layer-alloc, Hybrid-LLM PD-mux 서빙)의 현재 상태를 정본·메모리·최근 git/결과/실행중 job에서 재구성해 브리핑한다. "어디까지 했지", "지금 상태 정리", "이어서 하자", 세션 처음 열 때 사용. 확정 결론을 재도출하지 않고 요약만 한다. handoff 스킬의 짝(handoff가 저장 → catch-up이 복원).
---

# catch-up — 세션 오리엔테이션

목적: **직전 세션의 작업 문맥을 재구성**해 사용자가 다시 설명하지 않아도 이어서 일할 수
있게 한다. 확정된 결론은 **재도출하지 말고 요약**만 한다(CLAUDE.md 규칙). 새 결론을
만드는 스킬이 아니다.

## 1. 정본 읽기 (우선순위 순, 충돌 시 위가 이김)

1. `prefill-layer-alloc/PROJECT_STATUS.md` — 논문 방향 / 확정된 결과 / 철회된 가설 /
   증거 수준 표 / **다음 실험 gate**.
2. `prefill-layer-alloc/reports/CONSENSUS.md` — **§0(한 줄)** + §1
   확정 표 헤드라인 + **§5 열린 항목**(무엇이 완료/종결됐고 무엇이 남았는지).
3. `prefill-layer-alloc/reports/paper/CLAIM_EVIDENCE_MATRIX.md` + `EXPERIMENT_ROADMAP.md` — 현재 claim별
   판정과 다음에 필요한 실험(P1–P6, B0–B8, W1–W9).
4. **가장 최근 handoff 문서**: `prefill-layer-alloc/handoff-report/session_handoff_*.md` 중
   날짜가 가장 늦은 것 (`ls -t prefill-layer-alloc/handoff-report/session_handoff_*.md | head -1`).
   없으면 skip. (핸드오프는 전용 `handoff-report/`에 모임 — reports/와 분리.)
5. 투고 전략이 관련되면 `.../reports/paper/venue_positioning.md`.
6. 메모리 인덱스(이미 컨텍스트에 로드됨): `MEMORY.md`. 필요한 개별 메모리 파일만 추가로 읽는다.

## 2. 작업 상태 점검 (아직 정본에 안 들어간 것)

```bash
REPO=~/Experiments/KISTI/prefill-layer-alloc   # 작업 루트가 바뀌면 이 줄만 고친다
git -C "$REPO" log --oneline -12               # 저장소는 여기 하나뿐(작업 루트는 git 아님)
git -C "$REPO" status --short                  # 미커밋 변경
# 최근 결과·로그 (원시 telemetry는 git 밖 — 이양 번들에서 푼 것)
ls -t "$REPO"/workspace/engine-port/results/*/ | head
ls -t "$REPO"/*.out 2>/dev/null | head
```
**실행 중인 GPU 작업**: 2026-09-18 현재 **없다.** 개발·검증은 로컬(CPU)에서만 하고 GPU
실행은 대여 서버가 정해진 뒤 시작한다(로컬 5060 Ti는 PD-mux가 거부 — CLAUDE.md "환경 / 실행").
대여 서버가 생기면 그 접속 방식으로 확인하고 experiment-runner에 넘긴다.

## 3. 브리핑 출력 (간결하게)

아래 순서로 요약한다. 각 항목 1–3줄. 정본 경로를 링크로 건다.

- **현재 결론(bottom line)** — PD-분리 이득 / layer-aware 死 / 단일-GPU 동적 死 /
  실무 권고(peak-decode 기준 decode-heavy static). 재논증 금지, 한 줄씩.
- **라이브 트랙(R2)** — H-Architecture(true dual-worker) / H-Policy(offline decode-floor
  predictor) 상태와 실험 gate(Claim D/E acceptance).
- **직전 세션에서 한 것** — 최근 handoff 문서 + git log 요약.
- **열린 항목 / 다음 실험 gate** — CONSENSUS §5 + ROADMAP에서 미완인 것.
- **미커밋 변경 / 실행 중 job** — 있으면 명시(있는데 방치되면 손실 위험).
- **재확인할 방법론 게이트** — 정책 결론엔 서빙 실증·n≥4·paired CI·변화 trace 필요(CLAUDE.md §방법론).

## 4. 규율

- 확정 결론을 다시 증명하려 하지 마라. "이미 확정" 항목은 **요약**하고, 사용자가 도전하면
  그때 claims-auditor에 위임.
- 정본끼리 모순이 보이면 doc-steward에 넘겨 정리(직접 덮어쓰지 마라).
- 브리핑 끝에 **"이어서 뭘 할까요?"** 로 다음 액션 후보 2–3개(열린 항목 기반)를 제시.
