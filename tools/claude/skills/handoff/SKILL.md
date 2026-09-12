---
name: handoff
description: 세션을 마무리하거나 중간 체크포인트를 남길 때, 이번 세션의 결정·측정·코드/문서 변경·열린 항목을 dated handoff 문서 + 메모리에 저장해 다음 세션이 catch-up으로 이어받게 한다. "정리하고 마무리", "핸드오프 남겨", "저장해두자", "체크포인트" 할 때 사용. 정본 갱신·메모리 반영은 doc-steward 규율에 위임. catch-up 스킬의 짝.
---

# handoff — 세션 체크포인트

목적: **이번 세션의 작업 델타를 durable하게 저장**해, 새 세션의 `catch-up`이 문맥을
복원할 수 있게 한다. 저장 안 하면 아직 정본에 안 들어간 결정·상태가 유실된다.

## 1. 이번 세션 델타 수집

```bash
git -C /scratch/ehmoon/whlee/prefill-layer-alloc log --oneline -15
git -C /scratch/ehmoon/whlee/prefill-layer-alloc status --short          # 미커밋
git -C /scratch/ehmoon/whlee/prefill-layer-alloc diff --stat             # 무엇이 바뀜
ls -t /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/*/ | head
```
그리고 **대화에서 실제로 도달한 것**을 정리한다: 내린 결정, 돌린 실험/측정, 만든/고친
코드·문서, 남은 열린 항목·다음 단계.

## 2. dated handoff 문서 작성

경로: **`prefill-layer-alloc/handoff-report/session_handoff_<YYYY-MM-DD>.md`**
(핸드오프 **전용** 디렉터리 — 다른 reports와 섞지 않는다. 관리·활용 편의). 디렉터리가
없으면 생성. 오늘 날짜 = 세션 컨텍스트의 currentDate, **절대 날짜**. 같은 날짜 문서가 이미
있으면 append/갱신. 섹션:

- **이번 세션 요약** — 1문단.
- **결정·측정** — 무엇을, 어떤 근거로. 수치는 결과 경로와 함께(문서엔 요약, 원자료는 `results/<campaign>/`).
- **코드·문서 변경** — 파일 경로 + 무엇이 왜 바뀜. 미커밋이면 명시.
- **열린 항목 / 다음 세션 시작점** — 구체적으로. 다음 실험 gate·필요한 런 수·블로커.
- **미완·주의** — 검증 안 된 것, claims-auditor에 아직 안 건 주장, 방치된 job.

## 3. 정본·메모리 반영 (doc-steward에 위임)

세션에서 **경험적 결론이 확정/철회**됐거나 프로젝트 상태가 실질적으로 바뀌었으면
**doc-steward** 에이전트에 위임해 규율대로 반영한다:

- PROJECT_STATUS.md(확정된 결과/철회된 가설/증거 수준/다음 실험 gate) 및 해당 트랙 보고서 갱신.
- `MEMORY.md` 인덱스 + 관련 메모리 파일 갱신(비자명한 판정·방법론 교훈만; 코드/깃으로
  재현 가능한 건 넣지 않음). 메모리 경로:
  `/home01/ehmoon/.claude/projects/-scratch-ehmoon-whlee/memory/`.
- 절대 날짜, 증거 수준어(부분 지지/강한 지지(범위 한정)/미검증/철회) 일치, stale 배너.

단순 작업 상태(결론 변화 없음)면 handoff 문서만으로 충분 — 정본을 건드리지 마라.

## 4. 규율 (필수)

- **overclaim 금지.** 결과를 "확정"으로 쓰려면 서빙 직접 측정 + n≥4 + paired CI가 있어야
  한다. 애매하면 **claims-auditor**에 먼저 반증을 걸고, 통과 못 하면 "미검증/부분 지지"로 기록.
- 정책 비교 수치를 handoff에 쓸 땐 변화 trace·switch_count·TTFT/ITL percentile 유무를 명시.
- ★**커밋은 바로 한다 — 묻지 마라**(2026-08-25 지시). handoff 문서와 그에 딸린 정본·메모리
  변경을 **한 커밋으로 묶어** 저장한다. 이유: handoff는 세션 종료 체크포인트이므로 커밋을
  미루면 다음 세션 `catch-up`이 **미커밋 상태를 물려받는다** — durable하게 만드는 것이
  목적인데 확인 왕복이 그 목적과 어긋난다.
  순서: 문서 작성 → doc-steward 반영 → `check_citation_stops.py` **0위반 확인** → 커밋.
  (루트 `.out`/`.err`는 커밋 대상 아님 — 스테이징에서 제외.)
- ★★**push는 하지 않는다.** 묻지도, 보고하지도 마라(원격 URL에 토큰이 평문 노출돼 있다).

## 5. 마무리 출력

작성한 handoff 문서 경로, **커밋 해시**, doc-steward가 갱신한 정본/메모리 목록, 실행 중
job(있으면), 그리고 **다음 세션이 `catch-up`으로 이어받을 시작점** 한 줄을 보고한다.
