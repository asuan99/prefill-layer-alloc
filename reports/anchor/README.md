# `reports/anchor/` — 읽는 순서용 지도

- `axes_map.html` — artifact **"서빙 축 지도"**(https://claude.ai/artifact/9eXREM836RYTxWu9SnNSik, 비공개, 2026-10-15 v3)의 저장소 사본.
  축 11개(E 엔진 6 · M 모델 1 · U 워크로드 1 · H 기판 1 · S 측정 2), 축 간 결합 14개, claim ↔ 축, 트랙 5개, 인용 규율, 문서 지도.

## 지위

- **정본 아님. 새 판정 0건.** 수치·문장은 전부 `PROJECT_STATUS.md`·`reports/CONSENSUS.md`·`COSTMODEL.md`·`reports/paper/`·
  트랙 파일·`workspace/engine-port/results/handoff_review_2026-10-15/`에서 전사했고, 항목마다 근거 `경로:줄`을 달았다. 충돌하면 정본이 이긴다.
- 상위 좌표계는 `COSTMODEL.md:11`의 `비용 = f(E, M, U)`에 기판 H(사각지대 D-8)와 측정층 S를 더한 것이다. 새 축 정의를 만들지 않는다.
- 2026-09-30 "Hybrid PD-mux Anchor"(`reports/system_design_anchor_2026-09-30.md`, 스냅샷 `handoff-report/artifacts_2026-09-30/`)는
  스케줄러 설계 기준선이고, 이 지도는 축·결합·최신본 위치를 보여 준다. 서로 대체하지 않는다.

## 갱신 규칙

1. 데이터는 HTML 안 `<script type="application/json" id="axes-data">` 블록 하나다. 고칠 때는 그 블록만 고친다.
2. 정본이 바뀌면(promote 직후) 블록을 갱신하고 **같은 URL로 재발행**한다. 파일명에 날짜를 붙이지 않는다.
3. 커밋 전: 근거 포인터 존재 검사(아래) + `check_citation_stops.py`.

```bash
python3 workspace/engine-port/scripts/discipline/check_citation_stops.py --file reports/anchor/axes_map.html
```

## 2026-10-15 v1 작성 때 드러난 stale 지점 (지도에 `stale` 칩으로 표시, 수리는 각 담당 몫)

- `tracks/main.md`가 Q1을 아직 "결정 대기"로 적음 — 실제 결정은 `handoff_review_2026-10-15/DECISIONS_Q1_consolidated_2026-10-15.md:4` (handoff 몫).
- `PROJECT_STATUS.md` "최종 갱신" 줄 2026-10-13, Q1 결정·2회차 감사 미반영 (promote 몫).
- `reports/paper/CLAIM_EVIDENCE_MATRIX.md` 2026-10-04 이후 미갱신 (promote 몫).
- `reports/paper/DOCUMENT_STATUS.md:127-175`가 CONSENSUS를 historical로 분류하나 실제로는 rev95까지 갱신 중 (doc-steward 몫).

## 검토 이력

- 2026-10-15 v1→v3: doc-steward 읽기 전용 대조 1회. 등급 칩 과장 6건(serving→code/scoped 등), 누락 단서 4건, 포인터 7건 수정.
  `check_citation_stops.py --file` 위반 1건(claim C 크기 문구) 제거 → 0건.
