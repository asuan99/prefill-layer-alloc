# 2026-09-28 artifact 스냅샷 (보고용 로드맵 v17 · 검토용 v6)

- `pdmux_roadmap_v18.html` — 교수 보고용 "Hybrid PD-mux 로드맵"(claude.ai artifact `6URdjxzXLB2rCrG7Qq3JWS` **v19**, 시뮬레이션 재작업본: 워크로드×배분 분리·요청별 줄·층 단위 span·decode 비면 prefill 108). 이전 판 `pdmux_roadmap_v17.html`은 기여 3개 확정 시점 스냅샷.
  절: 1 무엇을 연구 · 2 용어 · 3 Hybrid 특성(H1·H2·H3 차트) · 4 워크로드 지도 · 5 동작 시나리오(그림 1–3 + 캔버스 시뮬레이션) · 6 핵심 기여 3개.
- `pdmux_review_v6.html` — 내부 "PD-mux 실험 검토"(artifact `NhCAwkHtZb5LeYJqQ4e8fH` v6).
  절: 1 확정된 것 · 2 실행 단계(0–5, 결정 ①②) · 3 기여(계층 + 세부 표 ①–⑨) · 4 뺀 실험 · 5 예산 · 6 원칙 · 7 결정 체크리스트 · 대응표.
- `gen_charts.py`, `gen_h1.py` — H1·H3 SVG 차트 생성기(입력 수치는 정본 인용 가능 값만; 2026-09-28 감사로 2,940×/27× 배율과 residency 범위 3–10/43–98 % 제거).
- `edit_*.py` — 2026-09-28 감사 반영 편집 스크립트(재현용, 순서: audit2 → core3 → audit3 → audit3b → link).

**정본이 아니다.** markdown 원본은 `handoff-report/roadmap_review_2026-09-27.md`(rev2)이며, 이 HTML의 기여 3개 재구성·4-a 추가 arm·agnostic 기준선 arm 제안은 `session_handoff_2026-09-28.md`와 `reports/audit/2026-09-28_contributions/`에 기록돼 있다. 캔버스 시뮬레이션은 개념 애니메이션이며 측정이 아니다.
