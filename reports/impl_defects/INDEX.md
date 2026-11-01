# 구현 결함 수정·측정 — 파일 위치 안내 (2026-11-01)

이 작업의 엔진 코드·측정 결과·그림은 **main이 아니라 브랜치 `impl-defects-e123`**에 있다. 그 브랜치는 별도 worktree 폴더 `prefill-layer-alloc-implfix/`(이 저장소 폴더 옆)로 체크아웃돼 있다. main에 병합하지 않은 이유: main에는 엔진 digest로 봉인된 다른 측정이 묶여 있다.

| 폴더 | 브랜치 | 내용 |
|---|---|---|
| `prefill-layer-alloc/` | main | 정본(PROJECT_STATUS·CONSENSUS·COSTMODEL), 트랙 파일, 축 지도, Phase 0 CPU 확인·범위 감사 |
| `prefill-layer-alloc-implfix/` | impl-defects-e123 | 엔진 수정(E1–E3·E5), GPU 정확성 검사, 정책 측정 PREREG·결과·그림 |

## main에 있는 것
- Phase 0 CPU 확인: [PHASE0_CPU_CHECK](../../workspace/engine-port/results/impl_defects/PHASE0_CPU_CHECK_2026-10-28.md)
- 루프 의존 범위 감사: [AUDIT_scope_loop_dependence_r1](../../workspace/engine-port/results/impl_defects/AUDIT_scope_loop_dependence_r1_2026-10-29.md)
- 범위 정정 제안·diff: [PROPOSAL](../hymux_review/PROPOSAL_scope_wording_impl_defects_2026-10-29.md)

## impl-defects-e123(worktree `prefill-layer-alloc-implfix/`)에 있는 것
- 엔진 수정 보고: [IMPL_FIX_E1_E3](../../../prefill-layer-alloc-implfix/reports/hymux_review/IMPL_FIX_E1_E3_2026-10-29.md)
- GPU 정확성 검사 결과: [rev0.1 FAIL](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/impl_defects_gate/RESULT_impl_gate_c03d622_2026-10-30.md) · [rev0.2 UNRESOLVED](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/impl_defects_gate/RESULT_impl_gate_077434a_2026-10-30.md)
- 정책 측정 설계 보고(r 정의·수식): [REPORT_policy_implfix_design](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/REPORT_policy_implfix_design_2026-10-31.md)
- 정책 측정 결과(감사 r1 반영): [RESULT_policy_implfix](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/RESULT_policy_implfix_2026-11-01.md) · [감사 r1](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/AUDIT_result_policy_implfix_r1_2026-11-01.md)

### 그림 (기술용, 사전 등록 아님)
| 그림 | 무엇을 볼지 |
|---|---|
| [fig1 r 분포](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig1_rspan_ecdf.png) | 가는 선 = 봉인 계산(warmup·프로브 섞임), 굵은 반투명 선 = 등록 부하만 사후 재계산. 녹색 띠 0.75–1.25가 박자 범위 |
| [fig2 쌍별 효과](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig2_forest.png) | 각 비교의 ln(ON/REF)과 95% CI, 회색 띠 ±3% |
| [fig3 실현 차이](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig3_dose.png) | 두 정책이 다르게 자른 비율, 50% 기준선 |
| [fig4 용량 사다리](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig4_capacity_ladder.png) | OFF 부하별 goodput, 선택된 용량 λ\* |
| [fig5 EVAL 셀](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig5_eval_cell_lohi.png) | 저부하는 같고, 고부하(과부하) 구간에서만 플래그 ON(빨강)이 낮음 |
| [fig6 동적 전환](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig6_dynamic_switch_dwell.png) | 동적 OFF는 전환 0회(같은 분할 null), 플래그 ON에서 전환 발생 |
| [fig7 static D와 D\* 예측](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig7_static_D_dstar.png) | D별 goodput과 예측 띠, 적중·불일치(E-8은 n=4 미분해) |
| [fig8 drain 대 handoff](../../../prefill-layer-alloc-implfix/workspace/engine-port/results/policy_implfix/figures/fig8_e3_drain_vs_handoff.png) | ITL p95/p99 동일, 막대는 컨트롤러 결정 변경 수(물리 handoff 미관측) |
