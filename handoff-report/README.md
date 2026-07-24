# handoff-report/ — 세션 핸드오프 전용 아카이브

이 디렉터리는 `prefill-layer-alloc` 세션 간 인수인계(handoff) 기록만 모아둔
전용 아카이브다. `reports/`(현재 연구 정본)와는 **분리**해서 관리한다 — 핸드오프는
그 세션 시점의 point-in-time 기록이지, 갱신되는 연구 결론이 아니다.

★ **핸드오프 문서는 전부 역사적 기록이다. 현재 결론이 아니다.** 프로젝트 전체
현재 상태는 항상 루트 `PROJECT_STATUS.md`를 우선하고, 확정/철회 상태는
`reports/CONSENSUS.md`를 우선한다. 핸드오프 문서가 이 두 정본과 충돌하면
정본이 이긴다.

2026-07-24부로 세션 핸드오프 스킬은 이 경로(`handoff-report/`)에 저장한다.

## 목록 (날짜순)

| 문서 | 날짜 | 요지 | 현재 상태 |
|---|---|---|---|
| [`session_handoff_2026-07-07.md`](session_handoff_2026-07-07.md) | 07-07 | R-시리즈 재정립, coordinated layer-aware 반증, SLO-aware track 시작 | 역사적 기록 — 이후 `reports/research_arc.md`·`reports/CONSENSUS.md`가 후속 결론을 흡수 |
| [`session_handoff_2026-07-13.md`](session_handoff_2026-07-13.md) | 07-13 | cudagraph 재측정, prefill-side layer-aware (B,L) knee로 매장, SLO-aware Step D/E/F | 역사적 기록 — 07-15가 후속 |
| [`session_handoff_2026-07-15.md`](session_handoff_2026-07-15.md) | 07-15 | layer-aware 死 확정 + SLO-aware 동적 트랙 종결(HE0) + 실trace 열림 | **부분 반증됨** — 아래 참조 |
| [`session_handoff_2026-07-24.md`](session_handoff_2026-07-24.md) | 07-24 | "단일-GPU 동적 제어 완전 종결" 선언의 overclaim 여부를 claims-auditor로 재검증 → 정본 스코프 축소 + HE0-reopen 게이트 설계(측정 없음, 설계 세션) | 역사적 기록 |

## `session_handoff_2026-07-15.md`의 부분 반증 (2026-07-17, `deprecated/reports/quarantine_engine_port/README.md`에서 이관 보존)

> **부분 반증 + 대체됨.** "cudagraph 최적 static=d16"·"실전=d16 고정"은
> **저-decode-부하 synthetic 아티팩트**다(실 trace/변화 trace서 d16이 최하위,
> 최적은 d44). 전체 흐름은 [`../reports/research_arc.md`](../reports/research_arc.md)가
> 대체한다.

이 반증 이후에도 유지되는 부분: layer-type 런타임 정책 全死, HE0(동적이
best-static을 못 넘음) — 단 근거는 이 문서의 synthetic 벤치가 아니라 **변화
trace n≥4**(`reports/CONSENSUS.md` §1-7)다.

## 형제 참조

07-13은 07-07을, 07-15는 07-13을 각각 "이전 핸드오프"로 링크한다(같은 폴더로
함께 이동했으므로 그대로 유효).
