# Document status

최종 갱신: 2026-07-24. 파일은 link 보존을 위해 이동·삭제하지 않는다.

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
| root `reports/PAPER_RESULTS.md` | 철회된 layer-aware 1.37–2.02× claim | PROJECT_STATUS |
| root `reports/FINAL_SUMMARY.md` | 이미 superseded된 simulation capstone | PROJECT_STATUS |
| root `CLAUDE_CODE_SESSION_SUMMARY.md` | R1/R2 이전 handoff | PROJECT_STATUS |
| root `reports/`의 초기 논문 트랙 | simulator/characterization과 철회 claim 혼재 | root reports README + PROJECT_STATUS |
| `RESUME.md` 연구 상태 절 | 오래된 P1.4/P1.7 상태 | 환경 정보만 유지 |

Deprecated 문서는 historical artifact로 인용할 때 측정 환경과 정정 문서를 함께
명시한다.
