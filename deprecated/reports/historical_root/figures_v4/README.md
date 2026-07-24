# figures_v4 — 발표 덱 (decode-side layer-type 반증 전용)

서사: **decode-side layer-type 분할 반증**(sim→binary→graduated→coordinated). prefill-side(PF)와
workload-regime(tuned-uniform/SLO-aware)는 이 덱에서 **의도적으로 축소** — 로드맵(F9) 한 장에만 등장.
**F1–F8은 decode-side 데이터만** 사용(PF·workload 데이터 미혼입).

생성 스크립트: `workspace/characterization/experiments/e9_deck_figs/` (e8_engine_figs 헬퍼 재활용).
출력: PNG(150 DPI) → PIL PDF 변환. 재생성:

```bash
PY=/scratch/ehmoon/whlee/sglang_engine_venv/bin/python
cd workspace/characterization/experiments/e9_deck_figs
for f in make_f*.py; do $PY "$f"; done
```

## 데이터 출처표

| 그림 | 종류 | 데이터 출처 (경로 · job) | 핵심 수치 |
|---|---|---|---|
| **F1** sim_prediction_vs_binary_collapse | measured+derived | [measured] binary 붕괴 `triage/notes.md` P1.7g (agnostic 33.5→la 72.0, +115%; control job **834486** Granite floor16) · [derived] sim 0-cost→≈baseline `reports/p1_4_layer_aware_평가_kr.md` ("순이득≈0") | 34 → 72 ms (+115%) |
| **F2** four_pass_diagnosis_waterfall | measured (서술) | `reports/sm_policy_report.html` §07 "four passes" | A ~0 · B ~8%(+2.7) · C +35.8 · D=granularity(~0.8ms window) |
| **F3** decode_sm_sensitivity_asymmetry | measured | `results/r0c/knee_result_835571.txt` (job 835571) | attn/mamba @108 8.7× → @8 35×; attn 3.14/mamba 0.36 ms |
| **F4** graduated_sm_pool_sweep | measured | `results/r0b/r0b_summary.csv` (jobs 835044–835209) · 기준 `r0c/_rows_agn_*835303` | 96/96 44 · 84/84 45 · 72/72 121 · 54/84 119 · 54/16 155 ms vs agn 42.5 |
| **F5** coordination_contrast | measured | coordinated `r0c/_rows_agn_*835303` (42.5) · uncoordinated `r0b/r0b_summary.csv` g_54_54 (job **835044**, 121.8); 교차확인 `reports/r_series_status.md` §R0b | 같은 54 SM: 42 vs 121 ms (2.9×) |
| **F6** coordinated_implementations_vs_agnostic | measured | R0d job **835918** · v4 job **837718** · (a) job **837520** · agn `r0c` 835303 | (a) 85 · v4 95 · R0d 124 vs floor 42.5 ms (전부 floor 위) |
| **F7** window_vs_kernel_timeline | **conceptual** (라벨=measured) | `reports/prefill_vs_decode_execution.md` §4–6 (mamba window 0.36 / prefill kernel 2.2 ms) | 6× (kernel/window) |
| **F8** switching_overhead_accumulation | **conceptual** | `reports/session_handoff_2026-07-07.md` R0d "19윈도우 파편화" | 19 switches/step, crossover ~11 |
| **F9** design_space_roadmap | **conceptual** | 없음 (개념도, 수치·job 미표기) | — |

경로는 `results/*`는 `workspace/engine-port/results/` 기준, `reports/*`는 `workspace/engine-port/reports/` 기준.

## 색상 팔레트 (sm_policy_report.html CSS 변수 기준 통일)
attn `#e8710a` · mamba `#1a73e8` · agnostic `#158b7f` · layer-aware `#c6790f` · fused/collapse `#c0503a`.

## 데이터 위치 미확인 (보고)
- F1의 **sim LUT 정량 예측치**: `characterization/results_v2/e5*`에 per-layer solo latency는 있으나,
  "layer-aware TPOT 예측"으로 직접 라벨된 값은 없음 → F1의 예측 막대는 p1_4의 **[derived] "순이득≈0"**
  (0-cost 가정 → ≈baseline)으로 처리. sim의 명시적 TPOT 예측 단일값이 필요하면 별도 확인 요망.
