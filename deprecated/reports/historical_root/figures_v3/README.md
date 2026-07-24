# figures_v3 — 논문/발표용 그림 (engine-port 실측 기반)

Results 절(negative-result 중심)용 그림. **모든 수치는 저장소 내 실제 결과 파일에서 직접 파싱**
(sim 예측치·재구성치 아님). 생성 스크립트: `workspace/characterization/experiments/e8_engine_figs/`.
출력: PNG(150 DPI) → PIL로 PDF 변환. 재생성:

```bash
PY=/scratch/ehmoon/whlee/sglang_engine_venv/bin/python
cd workspace/characterization/experiments/e8_engine_figs
$PY make_f1_coord_la_vs_agnostic_floor.py
$PY make_f2_window_vs_kernel_timeline.py
$PY make_f3_decode_vs_prefill_differential_dual.py
```

## 완료 (핵심 3 — 이것만으로 논문 핵심 주장 성립)

| 그림 | 파일 | 데이터 출처 (경로 · job) | 핵심 수치 (measured) |
|---|---|---|---|
| **F1** coord_la_vs_agnostic_floor | `f1_*.pdf/.png` | `r0d_coord_la/inefficient_v1/_rows_*835906` · `r0d_coord_la/_rows_*835918` · `v4_multiwin/_rows_*837718` · `a_substrate/_rows_*837520` · 기준선 `r0c/_rows_agn_*835303`,`r0c/_rows_16_*835300` | TPOT p50 @in3600/o32 rate2: inefficient_v1 **698** · R0d **124** · v4 **95** · (a) **85** vs 기준선 agnostic **42.5** / tuned(d16) **45.7** ms. 4개 전부 floor 위(2.0–16.4×). |
| **F2** window_vs_kernel_timeline | `f2_*.pdf/.png` | decode `r0c/knee_result_835571.txt` · prefill `prefill_knee/knee_result_837931.txt` | per-layer @full SM: mamba-decode **0.36ms**, attn-decode **3.14ms**, mamba-prefill **9.03ms**. prefill 커널이 mamba 창의 **25×** → (D) granularity. 레이어수 attn 9 / mamba 54. |
| **F3** decode_vs_prefill_differential_dual (**F4 흡수**) | `f3_*.pdf/.png` | 좌 decode `r0c/knee_result_835571.txt` · 우 prefill `prefill_knee/knee_result_837931.txt` | attn/mamba SM-민감도 differential: **decode 8.7→35×** vs **prefill 1.2→1.4×**. SM 108→8, log-log. |

경로는 전부 `workspace/engine-port/results/` 기준.

## 통합/정리 결정
- **F4 → F3 좌패널로 흡수** (동일 `r0c/knee_result_835571.txt`, 별도 그림 불필요).
- 색상: attn `#e8710a` / mamba `#1a73e8` (characterization figures와 일치), agnostic `#158b7f` /
  layer-aware `#c6790f` / fused `#c0503a` (sm_policy_report.html CSS 변수).

## 남은 그림 (미생성 — 데이터 위치 상태)
| 그림 | 상태 | 데이터 |
|---|---|---|
| F6 four_pass_diagnosis_waterfall | 가능 | `reports/sm_policy_report.html` §07 "four passes" |
| F7 coordination_contrast | 가능 | `r0b/r0b_summary.csv` (jobs 835044–835209) |
| F8 agnostic_vs_fused_4models | 가능 | `triage/p1_7bench_srv_*` (832701/832575/832638), P1.6 831609 |
| F13 regime_dependent_optimal_split | 가능(F8과 A-클러스터 통합 권고) | `r0c/` d24/d44 sweep |
| F14 slo_aware_convergence | 가능 | `slo_sched/` + `reports/policy_comparison.md` §2 v7b |
| F9/F10 (재출력) | 가능 | `sm_policy_report.html` §07 mamba_models/attn_models SVG |
| F5·F11·F12 | **부록 1개로 병합 권고** | F5 sim LUT는 `characterization/results_v2`서 **미검출 → 확인 필요** |
