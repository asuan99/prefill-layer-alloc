# 실험 전체 색인 (테이블)

작성일: 2026-06-22 · 프로젝트 `prefill-layer-alloc` · GPU A100-SXM4-80GB(108 SM)
대상: Zamba2 1.2/2.7/7B(temporal 하이브리드) · Falcon-H1 3/7B(spatial 하이브리드)

---

## A. 실험 단계 (E0–E7) — 무엇을 측정하고 무엇을 알았나

| 단계 | 측정 대상 | 핵심 산출 | 핵심 결과 |
|---|---|---|---|
| **E0** analytical | compute/memory 해석 모델 | roofline·이론 추정 | attn=compute·ssm=memory 비대칭의 이론 근거 |
| **E1** prefill_decomp | prefill 비용 분해(GEMM vs mixer) | 레이어별 prefill 구성 | prefill은 proj-GEMM 지배(compute-bound) |
| **E2** sm_saturation | 커널이 몇 SM에서 포화되나 | SM-스케일 곡선 | attn/ssm 포화 SM 비대칭 |
| **E3** decode_floor | decode를 solo 속도로 유지하는 *최소 SM*(보호 비용) | `decode_floor_*.csv` | attn floor↑(54~108), ssm floor↓; 7B attn 미포화 |
| **E4** concurrent | green-context 2-파티션 *동시 실행* 타이밍 | `create_two_partitions`·`measure_concurrent` | green_ctx가 two_stream 못 이김(분할 死의 근거) |
| **E5** serving | prefill ∥ decode 공존 행렬(LUT) | `serving_coexec_full_*.csv`(e5_sim_b8_opt 등) | 정책별 prefill/decode_stream LUT — sim의 입력 |
| **E6** queue_sim | LUT 구동 *큐 시뮬레이터*(PD-mux 정책) | `run_layer_aware`·`run_queue_sim` | **layer_aware 결과·4-way framework 비교** |
| **E7** prefill_sm | prefill SM-민감도 스윕(14~108 SM) | `prefill_sm_*.csv` | prefill 민감도 크기-불변(7B도 민감) |

## B. E5 LUT 변종 (결과 디렉토리)

| 디렉토리 | 의미 |
|---|---|
| `e5` / `e5_dp` | 초기 serving coexec / decode-protect 포함 |
| `e5_sim` / `e5_sim_b8` / `e5_sim_b8_opt` | sim 입력 LUT(budget=8, opt-decode=flash) ← **현재 표준** |
| `e5_fused` / `e5_fused_opt` | 진짜 fused-step 측정(fusion_saving≈0 확인) |

## C. SLURM 잡 이력 (이번 세션) — 측정과 결과

| job | 이름 | 측정 | 결과 |
|---|---|---|---|
| 778472 | v2-fused | 진짜 fused-step | **fusion_saving≈0**(fused≈prefill+decode) |
| 783157/8 | v2-fused/e5 | production decode(flash_attn_with_kvcache) | flash 가속은 KV크기(GQA비) 의존 |
| 783848/50 | vllm-smoke | vLLM 로드·생성 de-risk | vLLM이 SXM4서 두 모델 서빙 OK |
| 783851/60 | vllm-bench | (실패→진단) | offline metrics=None / prometheus 미들웨어 크래시 |
| **783863** | vllm-bench | **vLLM 실서빙 TTFT/TPOT/throughput** | **fused baseline 실측 검증; "16×GQA" 반증** |
| 785877 | la7b_e3e5 | 7B E3 floor+E5(초기) | E5 ssm-decode **OSError**(Triton 캐시) → 부분 |
| 793540 | la_e3e5 | 1.2B E3+E5 | 깨끗(188/200) → la/agnostic **1.37×** |
| 797524 | prefill_sm | prefill SM-민감도(1.2/2.7/7B) | **크기-불변**(ssm~2×·attn~7×); premise② 반증 |
| 797630 | e5_7b | 7B E5 재실행(TRITON fix) | dec=ssm OSError 해소; 단 E3 ssm floor 여전 누락→**1.00× artifact** |
| **797832** | re7b_e3e5 | **7B E3 ssm floor + E5 재측정** | **la/agnostic 1.82× — 7B도 스케일(반전)** |
| **799168** | vllm-z1b7b | **z1.2b·z7b vLLM 실서빙** | size-scaling 일치; calibration 1.66/2.34×(no-GQA 크기↑서 과대↑) |

## D. 모델 × 측정 커버리지

| 측정 | z1.2b | z2.7b | z7b | f3b | f7b |
|---|--|--|--|--|--|
| E3 decode floor | ✅ | ✅ | ✅(797832) | ✅ | ✅ |
| E5 LUT(opt+protect) | ✅ | ✅ | ✅(797832) | ✅ | ✅(797630) |
| E7 prefill SM-민감도 | ✅ | ✅ | ✅ | — | — |
| E6 layer_aware(4-way) | ✅1.37× | ✅2.02× | ✅1.82× | N/A(spatial) | N/A(spatial) |
| vLLM 실측 | ✅(799168) | ✅ | ✅(799168) | ✅ | — |

## E. 핵심 결론 (분석 단위)

| # | 질문 | 결과 | 근거 |
|---|---|---|---|
| 1 | layer-type *공간 분할*이 co-schedule 이기나 | **死**(1576셀 0승) | E2/E4, §3.1/§3.2 |
| 2 | layer-type-aware *예약*이 agnostic 이기나 | **生 1.37~2.02×**(temporal, 1.2B~7B) | E6 run_layer_aware |
| 3 | 적용 범위 | **temporal-only**(spatial=분리불가+GQA로 쌈) | temporal_vs_spatial |
| 4 | fused vs 분할 | fusion_saving≈0; fused decode ITL 4~6× 높음 | E5_fused·E6 framework |
| 5 | GQA가 decode 16× 가르나 | **반증** — full-model은 ssm 지배로 ~1.3× | vLLM 783863 |
| 6 | prefill SM-민감도 크기 의존? | **불변**(7B도 민감) | E7 797524 |
| 7 | 7B 미스케일? | **반전: 스케일함(1.82×)** — 1.00×는 E3 ssm floor 누락 artifact | 797832 |
| 8 | sim이 vLLM과 정량 일치? | **directional only** — 절대 fused ITL 1.66/1.64/**2.34×** 과대(no-GQA, 크기↑서↑)·thru 0.8× 과소; **단 크기-스케일링은 일치**(decode비 1/1.5/2.6 ≈ 실측) → 정량 비교는 프로토타입 필요 | 799168 framework_comparison |

## F. 산출 보고서 (reports/)

| 보고서 | 내용 |
|---|---|
| `FINAL_SUMMARY.md` | 종합 capstone |
| `layer_aware_benefit_report.md` | layer_aware 양성 결과·메커니즘·크기추세 |
| `layer_aware_7b_verification.md` | 7B 회귀(3회 반전·artifact 교훈) |
| `temporal_vs_spatial_hybrid.md` | sequential vs parallel 하이브리드 |
| `vllm_validation.md` | vLLM 실측 검증·16× 반증 |
| `framework_comparison.md` | 4-way+fused·calibration |
| `metrics_and_load_dependence.md` | 지표 사전·부하 의존성 |
| `partition_engine_design.md` | 실엔진 프로토타입(Path C) 스코핑 |
| `queue_simulator_design.md` | sim 설계·정책 §1–16 |

## G. 그림 (reports/figures/)

`summary_dashboard` · `layer_aware_result` · `layer_aware_size_trend` · `temporal_vs_spatial` · `prefill_sm_sensitivity` · `layer_aware_applicability` · `framework_comparison` · `vllm_calibration`

## H. 실엔진 serving 재측정 (engine-port, 2026-07) — sim 결론 supersede

> ⚠️ 위 §C–§G의 sim 헤드라인(la/agnostic 1.37–2.02×)은 실엔진(sglang v0.5.10) serving 측정으로 **기각**됨. 정본: [`workspace/engine-port/reports/sm_policy_report.html`](../../../reports/sm_policy_report.html) (§07 4-pass 은퇴). 4개 하이브리드 전부 layer-aware ≤ agnostic.

**R0a — Zamba2-2.7B clean async re-measure** (§04 prefill-bound 패널의 구 GIL-client 데이터 교체). 클라이언트 `python -m sglang.bench_serving` (async, random-ids), triton, ctx4096, in3600/out32, num-prompts 120, rates 1/2/3/4/6, SLO TTFT≤3s·TPOT≤60ms.

| job | policy | rep | 조건 | 결과(goodput@SLO peak) |
|---|---|---|---|---|
| 834914 | agnostic | 1 | Zamba2-2.7B in3600/out32 triton | **2.07 @rate2** (최고) |
| 834915 | fused | 1 | 〃 | 0.79 @rate1 (rate2↓ TPOT 붕괴) |
| 834916 | layer_aware | 1 | 〃 | 0.82 @rate1 → **0 from rate2** (최악) |
| 834927 | agnostic | 2 | 〃 (변산 반복) | **2.07 @rate2** (rep1과 동일) |
| 834929 | fused | 2 | 〃 | 0.81 @rate1 (rep1과 동일) |
| 834928 | layer_aware | 2 | 〃 | 0.81 @rate1 → **0 from rate2** (rep1과 동일) |

- 산출: `workspace/engine-port/results/r0a/` — `r0a_summary.csv`, per-run `zamba2_async_{policy}_r{rate}_rep{n}.csv`, 서버/벤치 로그, sbatch `r0a_zamba2_bench.sbatch`.
- 판정: **layer-aware ≤ agnostic 전 rate** (동률 이하) → 은퇴 확정(§07 정합). 구 GIL-client 산출(831146/831166/831541)은 [`triage/DEPRECATED_gil_client.md`](../../../workspace/engine-port/triage/DEPRECATED_gil_client.md) 참조.
- 병행 clean async(기확보): NemotronH 831609–831612 · Zamba2 in2000/out96 832701/832702 · Granite 832639/832690 · Falcon-H1 832575/832576.
