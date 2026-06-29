# 그래프 색인 — 각 그림이 무엇을 보여주나

분류: **A. 결론/논문용**(`reports/figures/`, 8개) · **B. 특성화**(`workspace/characterization/results_v2/figures/`, E0–E5)

---

## A0. 특성화 그래프 (신규, E3+E5 실측 반영) — *기반 메커니즘*

| 그림 | 패널 | **무엇을 보여주나** | 답하는 질문 |
|---|---|---|---|
| **char_phase_layertype.png** | 3-panel | (A) decode latency **vs batch**(attn 급증=메모리바운드 KV×B / ssm 평탄) · (B) decode **vs 시퀀스길이**(attn **O(L)** / ssm **O(1)**) · (C) **phase×layer-type** 매트릭스(prefill 대칭·compute / decode 비대칭·memory) | "attn/ssm이 phase에 따라 어떤 차이?" |
| **policy_scaling.png** | 2-panel | (A) **정책별 decode ITL vs batch**(fused>co>agnostic≈layer_aware, 실측) · (B) **vs 시퀀스길이**(fused/co가 context↑서 더 빨리 악화; attn O(L)) | "batch·seqlen에 따른 각 정책 실행시간?" |
| **mechanism_schematic.png** | 도식 | 한 forward의 **SM 배분**: attn-decode 레이어는 SM 예약(~94), ssm 레이어는 decode가 ~54서 포화→나머지 prefill 환원 | "layer_aware가 *어떻게* 작동?" (S2 시스템 도식) |
| **attn_ssm_diff.png** | 2-panel | (A) **decode SM-민감도**(attn은 SM 필요·가파름 / ssm은 ~54서 포화·평탄) · (B) **한 forward latency 구성비**(ssm 지배; attn-decode는 batch↑서 증가) | "attn/ssm 차이를 실측으로" |

> 이 둘이 **E0–E7 기반 측정(배치·시퀀스·phase 의존성)을 논문 figure로 가시화** — 기존 결론 figure가 못 보여주던 부분.

## A. 결론/논문용 그래프 (reports/figures/)

| 그림 | 패널 | **무엇을 보여주나** | 뒷받침 주장 |
|---|---|---|---|
| **summary_dashboard.png** | 3-panel | (A) la/agnostic goodput 1.37/2.02/1.82× (zamba2 크기별) · (B) attn-decode 비용 판별자(zamba2 0.59ms vs falcon 0.056ms) · (C) vLLM sim-vs-실측 decode 비율 | 프로젝트 전체 헤드라인 1장 |
| **layer_aware_result.png** | 3-panel | (A) **goodput vs SLO** 곡선 — SLO≥~48ms서 layer_aware 우위 음영 · (B) throughput↑/TTFT↓/ITL tradeoff 막대 · (C) **메커니즘**: 54-레이어 forward의 SM 배분(attn만 예약, ssm은 prefill 환원) | layer_aware 양성 결과 + 작동 원리 (zamba2_2.7b) |
| **layer_aware_size_trend.png** | 2-panel | (A) **la/agnostic 비율** 1.37/2.02/1.82× (peak 2.7B) · (B) 정책별 절대 goodput(co/agnostic/layer_aware) 모델별 | layer_aware가 1.2B–7B 전 구간 이득 |
| **temporal_vs_spatial.png** | 2-panel | (A) zamba2(temporal): 9개 *비싼* attn 스파이크 + 45개 싼 ssm (분리 가능) · (B) falcon(spatial): 매 레이어 attn(GQA로 작음)+ssm 병렬 | **왜 temporal-only인가** (per-layer decode 구조) |
| **layer_aware_applicability.png** | 2-panel | zamba2(temporal, attn/ssm 분리→적용 ✓) vs falcon(spatial, 병렬→N/A) 레이어 배치 도식 | 적용 범위 = temporal 하이브리드 |
| **prefill_sm_sensitivity.png** | 2-panel | (A) ssm-prefill SM-민감도 곡선 (1.2/2.7/7B 거의 겹침) · (B) attn-prefill 곡선 — 108→14 SM 비율 | prefill SM-민감도는 **크기-불변**(7B도 민감) |
| **framework_comparison.png** | 1-panel | 4-way **decode ITL** 막대(fused/co/agnostic/layer_aware) × 3모델 + vLLM 실측 109ms 앵커 | fused가 ITL 최대, 예약이 4–6× bound |
| **vllm_calibration.png** | 2-panel | (A) **sim fused ITL vs vLLM 실측** (과대비 1.66/1.64/2.34×, zamba2) · (B) **decode 크기-스케일링 일치**(sim 비 ≈ 실측 비) | sim 절대값 off(no-GQA·크기↑서↑), 상대 추세 valid |

### A 그림 — 파일 위치 & 생성 스크립트

> 모든 그림 `prefill-layer-alloc/reports/figures/*.png` · 스크립트 `workspace/characterization/experiments/<dir>/` · 실행 `cd workspace/characterization && ../../bin/python -m experiments.<dir>.<script>`

| 그림 | repo 경로 | 생성 스크립트 |
|---|---|---|
| char_phase_layertype | `reports/figures/char_phase_layertype.png` | `e6_queue_sim/make_char_phase_fig.py` |
| mechanism_schematic | `reports/figures/mechanism_schematic.png` | `e6_queue_sim/make_mechanism_fig.py` |
| attn_ssm_diff | `reports/figures/attn_ssm_diff.png` | `e6_queue_sim/make_attn_ssm_diff_fig.py` |
| policy_scaling | `reports/figures/policy_scaling.png` | `e6_queue_sim/make_policy_scaling_fig.py` |
| summary_dashboard | `reports/figures/summary_dashboard.png` | `e6_queue_sim/make_summary_fig.py` |
| layer_aware_result | `reports/figures/layer_aware_result.png` | `e6_queue_sim/make_layer_aware_figs.py` |
| layer_aware_applicability | `reports/figures/layer_aware_applicability.png` | `e6_queue_sim/make_layer_aware_figs.py` |
| layer_aware_size_trend | `reports/figures/layer_aware_size_trend.png` | `e6_queue_sim/make_size_trend_fig.py` |
| temporal_vs_spatial | `reports/figures/temporal_vs_spatial.png` | `e6_queue_sim/make_temporal_spatial_fig.py` |
| prefill_sm_sensitivity | `reports/figures/prefill_sm_sensitivity.png` | `e7_prefill_sm/make_prefill_sm_fig.py` |
| framework_comparison | `reports/figures/framework_comparison.png` | `e6_queue_sim/make_framework_fig.py` |
| vllm_calibration | `reports/figures/vllm_calibration.png` | `e6_queue_sim/make_calibration_fig.py` |

### A의 그림으로 답하는 핵심 질문
- "layer_aware가 정말 이득인가?" → **layer_aware_result(A)·size_trend**
- "어떻게 작동하나?" → **layer_aware_result(C) 메커니즘**
- "왜 temporal만?" → **temporal_vs_spatial · applicability**
- "vLLM 대비 어떤가?" → **framework_comparison · vllm_calibration**
- "한 장 요약?" → **summary_dashboard**

---

## B. 특성화 그래프 (E0–E5, results_v2/figures/) — 기반 측정

| 그림 | 무엇을 보여주나 |
|---|---|
| `e0_grid_sat_vs_batch` | 포화 여부 × 배치 격자(해석 모델) |
| `e0_overhead_budget` | prefill budget 대비 오버헤드 |
| `e1_component_breakdown` | prefill 비용 구성 분해(proj-GEMM vs mixer) |
| `e1_scan_share_vs_batch` | 배치에 따른 ssm-scan 비중 |
| `e2_asymmetry_satsm` | **attn/ssm 포화-SM 비대칭** |
| `e2_bwutil_at_sat` · `e2_mechanism_bwutil` | 포화 지점 HBM BW util(메모리-바운드 정도) |
| `e2_saturation_vs_batch` · `e2_smcurve_zamba2_1.2b` | 배치·SM에 따른 포화 곡선 |
| `e3_decode_floor_attn` · `e3_decode_floor_ssm` | **decode floor(보호 최소 SM)** — attn↑·ssm↓ |
| `e4_decode_interference` | 동시 실행 시 decode 간섭 |
| `e4_pp_throughput` | prefill-parallel throughput |
| `e5_context_dependence` | **context 길이 의존성**(attn O(L) vs ssm O(1)) |
| `e5_overlap_matrix` | prefill∥decode 공존 행렬 |
| `e5_spatial_specific_gain` | 공간-분할 특이 이득(있나) |
| `e5_speedup_vs_batch` | 배치별 speedup |
| `serving_overlap_and_budget` | serving 오버랩·budget |
| `study_summary` · `verdicts_summary` | v2 특성화 종합·판정 |

> A(결론용)는 **논문 figure 후보**, B(특성화)는 **방법/부록 근거**다. A는 "주장"을, B는 그 주장의 "기반 측정"을 보여준다.
