# Document status

최종 갱신: 2026-08-21(doc-steward — **gate #13 job-축 캠페인 완주(양
arm `PASS`) 반영으로 `CLAIM_EVIDENCE_MATRIX.md` Claim A 각주의
"job/node/day 축 부재"를 `reports/CONSENSUS.md` rev41·§3 항목71과
재동기화(job 축만 갱신, batch·node(≥3)·day(≥3) 축 부재는 그대로,
등급·인용정지 (a)(b) 무변경). 새 성능 판정 0건, 분류표 무변경,
changelog 기록뿐.** 이전: 2026-08-16(3차, doc-steward — **R1(he2 재채점) 재분석 완료
반영으로 `CLAIM_EVIDENCE_MATRIX.md` Claim D 행을 `reports/CONSENSUS.md`
rev34·§1-19와 재동기화(집계·술어별 확정 수치 등재, 등급 무변경). 새
성능 판정 아님, 분류표 무변경, changelog 기록뿐.** 이전: 2026-08-14(doc-steward, 2차 — **정본 결함 정정 반영(A-1)으로
`CLAIM_EVIDENCE_MATRIX.md` Claim A 각주의 "구간 평균 ε≈0.48–0.56"을
`reports/CONSENSUS.md` rev27과 재동기화(변환식·E-1a 원자료 직접측정과
일치하는 ε≈0.49–0.61로 정정, ITL 비 2.36–2.91× 자체는 불변, 원문
미덮어쓰기·괄호 정정). 새 성능 판정 아님, 분류표 무변경, changelog
기록뿐.** 이전: 2026-08-14(doc-steward — **정본 정정 1건(하드웨어 오식별) +
신규 결과 등재 1건(E-3) 반영으로 `CLAIM_EVIDENCE_MATRIX.md`를
`reports/CONSENSUS.md` rev26과 재동기화(Claim A 각주 정정 + "P1 트랙"
항목7 신설). 분류표 무변경, changelog 기록뿐.** 이전: 2026-08-11(doc-steward — **E1 addendum(jobs 877756/
877757 재집계) 반영으로 `CLAIM_EVIDENCE_MATRIX.md`·
`EXPERIMENT_ROADMAP.md`를 `reports/CONSENSUS.md` rev23과
재동기화(Gate 2-S 전제 4셀 전부 `VERIFIED_AT_SAMPLED_INSTANTS` —
등급 하향된 조건부 채택, "P1 트랙" 로드맵 완료분 이동 + 후속
E1-a/b/c/d 신설). 분류표 무변경, changelog 기록뿐.** 이전:
2026-08-11(doc-steward — **G1-c(job 877974) 반영으로
`CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md`를
`reports/CONSENSUS.md` rev22와 재동기화(Gate 2-S 인용 제한 7건
중 4번 갱신, "P1 트랙" 로드맵 완료분 이동). 분류표 무변경, changelog
기록뿐.** 이전: 2026-08-11(doc-steward — **정본 동기화 기록.**
`CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md`를
`reports/CONSENSUS.md` rev14→rev21(2026-08-06~2026-08-11) 및
`PROJECT_STATUS.md`의 같은 구간과 대조·동기화했다. Claim A–F/
P0–P6·벡터1·벡터2 기존 본문은 무변경(그 구간이 인용되지 않음,
`CONSENSUS.md` rev14–20 changelog가 매 rev 확인) — 두 문서에
지금까지 없었던 **"P1 트랙"**(PD-mux 자체 vs fused, Gate 1/Gate
2/E-A/Gate 2-S) 절을 신설해 현재 판정과 2026-08-11 Gate 2-S 인용
제한 7건을 이관했다. `DOCUMENT_STATUS.md`(이 문서) 자체의 "Canonical/
Historical/Superseded" 분류는 이 동기화로 바뀌지 않는다(구조는
그대로 정확) — 갱신은 changelog 기록뿐. 이전: 2026-07-25(아래 "Historical evidence" 표에 `spatial_decoupling_
design_review_2026-07-25.md` 부분-supersede 항목 추가). ★**정책 변경(2026-07-24, 사용자 승인)**: 기존 "파일은 link
보존을 위해 이동·삭제하지 않는다" 방침을 top-level 축소를 위해 **override**했다.
아래 표에 나열된 문서 중 root `reports/*`(전체) 와 두 `deprecated_reports/`는
`git mv`로 통합 `deprecated/`(신설) 아래로 **물리 이동**했다(내용·claim 판정은
불변, 경로만 변경). 이번 override 이후 다시 원 방침("link 보존을 위해 이동·삭제
하지 않는다")으로 돌아간다 — 즉 오늘부로 새 경로가 **다시 고정**이다. 이관 상세는
[`../../deprecated/README.md`](../../deprecated/README.md).

## Canonical

| 문서 | 역할 |
|---|---|
| repository root `PROJECT_STATUS.md` | 전체 현재 상태의 유일 정본 |
| `reports/paper/*` | claim, R1, experiment, code, 문서 상태 정본 |
| `reports/paper/venue_positioning.md` | 투고 전략 positioning. **claim/evidence 문서 아님** — CONSENSUS/CLAIM_EVIDENCE_MATRIX 판정을 인용·요약만 하며 새 증거의 근거로 쓰지 않는다 |

## Historical evidence, 유지

| 문서 | 사용 가능한 내용 | 제한 |
|---|---|---|
| `CONSENSUS.md` | 7월 19일까지 established/retired single-worker 결과 | dual/R2 이전 historical consensus |
| `per_layer_type_postmortem.md` | layer-granular negative mechanism | context floor 절대값은 one-model/no-CG |
| `research_arc.md` | 연구 변화 이력 | current hypothesis 정본 아님 |
| `dual_worker_design.md` | R1 observer 설계 이력 | R2 target과 분리해서 읽어야 함 |
| `../spatial_decoupling_design_review_2026-07-25.md` | §2(engine-porter, SGLang v0.5.10 disaggregation 기판 file:line 실현가능성)·§3·§4·§5(long-ctx 게이트 연결점)는 유효 | §1(신규성, disaggregation 축 평가)은 축 오조준으로 **부분 supersede** — `venue_positioning.md` §0.1(multiplexing 축 정정)이 대체. 문서 상단에 HISTORICAL 노트 삽입됨 |

## Superseded/stale

| 문서 | 문제 | 대체 |
|---|---|---|
| `realtrace_findings_and_open_branches.md` | 완료된 branch와 설계 상태 혼재 | CONSENSUS + roadmap |
| `slo_aware_scheduling_design.md` | 중간 positive interpretation 포함 | CONSENSUS |
| `policy_comparison.md` | no-CG goodput와 dynamic-win 본문 잔존 | claim matrix |
| `system_vs_engine_vs_sim.md` | no-CG/current-CG 판정 충돌 | PROJECT_STATUS |
| `prefill_vs_decode_execution.md` | phase 특성을 과도하게 일반화 | postmortem + PROJECT_STATUS |
| `r1_dual_worker_progress.md` | 성공 전 진행 로그 | R1_REANALYSIS |
| `r1_dual_worker/job_862512.md` | observer 차이를 architecture에 귀속 | R1_REANALYSIS |
| `deprecated/reports/historical_root/PAPER_RESULTS.md`(2026-07-24 이전 경로: root `reports/PAPER_RESULTS.md`) | 철회된 layer-aware 1.37–2.02× claim | PROJECT_STATUS |
| `deprecated/reports/historical_root/FINAL_SUMMARY.md`(이전: root `reports/FINAL_SUMMARY.md`) | 이미 superseded된 simulation capstone | PROJECT_STATUS |
| root `CLAUDE_CODE_SESSION_SUMMARY.md`(경로 불변, 2026-07-24 내부 링크만 새 경로로 갱신) | R1/R2 이전 handoff | PROJECT_STATUS |
| `deprecated/reports/historical_root/`(이전: root `reports/`의 초기 논문 트랙 전체 — figures/·figures_v3/·figures_v4/ 포함) | simulator/characterization과 철회 claim 혼재 | historical_root/README.md + PROJECT_STATUS |
| `deprecated/reports/quarantine_root/`(이전: root `deprecated_reports/`, 14 files) | 초기 triage·마이그레이션·모델별 평가 이력 | PROJECT_STATUS |
| `deprecated/reports/quarantine_engine_port/`(이전: `workspace/engine-port/reports/deprecated_reports/`, 12 files) | P0–P1.7 triage·구 핸드오프 이력 | CONSENSUS.md |
| `RESUME.md` 연구 상태 절 | 오래된 P1.4/P1.7 상태 | 환경 정보만 유지 |

Deprecated 문서는 historical artifact로 인용할 때 측정 환경과 정정 문서를 함께
명시한다.
