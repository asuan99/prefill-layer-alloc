# `reports/anchor/` — 읽는 순서용 지도

- `axes_map.html` — artifact **"서빙 축 지도"**(https://claude.ai/artifact/9eXREM836RYTxWu9SnNSik, 비공개, 2026-10-15 v5)의 저장소 사본.
  축 11개(E 엔진 6 · M 모델 1 · U 워크로드 1 · H 기판 1 · S 측정 2), 축 간 결합 14개, claim ↔ 축, 트랙 5개, 문서 지도.
  v4부터 축마다 흐름도(축 안의 기전, `axes[].flow`의 노드·간선 데이터)와 관계도(결합 목록에서 자동 생성)를, §3에 전체 관계도를 그린다.

## 지위

- **정본 아님. 새 판정 0건.** 수치·문장은 전부 `PROJECT_STATUS.md`·`reports/CONSENSUS.md`·`COSTMODEL.md`·`reports/paper/`·
  트랙 파일·`workspace/engine-port/results/handoff_review_2026-10-15/`에서 전사했고, 항목마다 근거 `경로:줄`을 달았다. 충돌하면 정본이 이긴다.
- 상위 좌표계는 `COSTMODEL.md:11`의 `비용 = f(E, M, U)`에 기판 H(사각지대 D-8)와 측정층 S를 더한 것이다. 새 축 정의를 만들지 않는다.
- 2026-09-30 "Hybrid PD-mux Anchor"(`reports/system_design_anchor_2026-09-30.md`, 스냅샷 `handoff-report/artifacts_2026-09-30/`)는
  스케줄러 설계 기준선이고, 이 지도는 축·결합·최신본 위치를 보여 준다. 서로 대체하지 않는다.

## 갱신 규칙

1. 흐름도는 `axes[].flow`(`n`: [id, 열, 행, 라벨, 종류 in/mech/out/bad/open], `e`: [출발, 도착, 라벨, hot/open])로 그린다. 노드·간선 라벨은 정본 문장의 요약이므로 새 판정을 넣지 않는다.
2. 데이터는 HTML 안 `<script type="application/json" id="axes-data">` 블록 하나다. 고칠 때는 그 블록만 고친다.
3. 정본이 바뀌면(promote 직후) 블록을 갱신하고 **같은 URL로 재발행**한다. 파일명에 날짜를 붙이지 않는다.
4. 커밋 전: 근거 포인터 존재 검사(아래) + `check_citation_stops.py`.

```bash
python3 workspace/engine-port/scripts/discipline/check_citation_stops.py --file reports/anchor/axes_map.html
```

## 검토 이력

- 2026-10-15 v1→v3: doc-steward 읽기 전용 대조 1회. 등급 칩 과장 6건(serving→code/scoped 등), 누락 단서 4건, 포인터 7건 수정.
  `check_citation_stops.py --file` 위반 1건(claim C 크기 문구) 제거 → 0건.
- 2026-10-15 v4: 사용자 요청으로 축별 흐름도·관계도와 전체 관계도 추가. headless Chrome으로 데스크톱 1200px·폰 400px 렌더 확인(흐름도는 폰에서 상자 안 가로 스크롤).
- 2026-10-15 v5: 사용자 요청으로 인용 규율 절(인용 가능 수치·인용 정지·금지 문장)과 날짜·stale 표시를 페이지에서 뺐다. 인용 규율 원문은 `reports/CONSENSUS.md`·`PROJECT_STATUS.md`에 그대로 있다. 날짜 갱신 문제는 에이전트·스킬 수정 논의로 넘김.
- 2026-10-19 v6: 정본 위계 개정(사용자 결정)에 맞춰 claim 근거를 `reports/CONSENSUS.md` §6으로, 문서 지도의 위계·다음 행동 위치를 고쳤다. 이후 이 지도의 갱신은 `promote` 스킬 3절 마지막 단계가 맡는다.
- 2026-10-21 v7: promote 4차 — §0 현재 상태 문단과 트랙 카드(main 상태·결정 대기, multigpu worktree 폐기·main 흡수)만 고쳤다. 새 판정 0, 재발행은 main 세션이 한다.
