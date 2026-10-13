# TRACK: multigpu — A100 ×1/2/4/8 DP·TP 확장   (worktree `prefill-layer-alloc-multigpu`, branch `multigpu-scaling`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 main의 promote.

## 목표
PD-mux 이득의 N-스케일링(1차 estimand G_sym = λ*_agnostic / λ*_fused_sym) + multi-GPU 고유 기전.
## cost model 기여
- 채우는 칸: E 축 확장(N, TP) — TP 통신 항이 CM-1에 더해지는 형태, GPU당 decode SM 요구 변화
- 이번에 알게 된 것: 없음(M2g는 정확성 게이트, 성능 판정 0)
- promote 필요: 아니오
## 현재 상태
- M2g 정확성 게이트: J1·J2 통과, J4·J8 하네스 수리(a41e725) 후 재실행으로 TP4·TP8 4 arm 포함. PASS 표 4개 봉인(99ed704).
- TP4·TP8 G2 `TP_NUMERIC_DIVERGENT`는 결함 판정 아님 — M3 TP 결과 옆 병기 의무(MGH-3, C4).
- 브랜치의 `src/multiplex` 변경은 N2/N3/E2 봉인 digest를 깨므로 그 캠페인 종료 전 main merge 금지.
## 다음 행동
1. M3 제출 전 "M3 경로 무변경"(313be0a 이후 a41e725) 경량 확인 → dry-run → 승인 → 직렬 제출
2. 이후 `mg_label.py` 결합 → result-analyst → claims-auditor 결과 감사
## 사용자 결정 대기
- **M3 PREREG 개정 여부** — 연구 감사 의견(main 핸드오프 (D)절): shape C 부재·`max_prefill_tokens` 미등록·N 간 비교엔 절대 SLO 필요·TP에서 GPU당 decode SM 요구 변화 / 현행 M3 제출 · PREREG 개정 후 재감사(권고: 개정 범위만 먼저 결정)
- **M3 제출 승인** — 모형 예보 예약 약 124 GPU-h, 최장 약 9 h
## 실행 중·미회수 Job
- 없음(M2g J4·J8 재실행 회수 완료)
## 금지 서술 (링크만)
`workspace/engine-port/results/multigpu/PREREG_MULTIGPU_2026-10-12.md` §7·§15–§18, 각 `VERDICT_*.md`
## 최근 로그
- 2026-10-13: M2g J4·J8 재실행 회수, PASS 표 4개 봉인
- 2026-10-12: M1+V0 프로브, PREREG rev11·하네스 봉인
